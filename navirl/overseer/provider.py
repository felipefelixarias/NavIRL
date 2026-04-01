from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

import cv2


class ProviderUnavailableError(RuntimeError):
    pass


class ProviderCallError(RuntimeError):
    pass


@dataclass(slots=True)
class ProviderConfig:
    provider: str = "codex"
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str = "NAVIRL_VLM_API_KEY"
    native_cmd: str | None = None
    timeout_s: float = 45.0
    max_images: int = 4

    def normalized_provider(self) -> str:
        return str(self.provider or "codex").strip().lower()


def _extract_json_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ProviderCallError("Empty VLM response.")

    if text.startswith("{") and text.endswith("}"):
        return text

    marker = "```json"
    if marker in text:
        start = text.find(marker)
        start = text.find("\n", start)
        end = text.find("```", start + 1)
        if start >= 0 and end > start:
            return text[start:end].strip()

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        return text[first : last + 1]

    raise ProviderCallError("Unable to find JSON object in VLM response.")


def _parse_json_object(raw: str) -> dict:
    try:
        return json.loads(_extract_json_text(raw))
    except json.JSONDecodeError as exc:
        raise ProviderCallError(f"Invalid JSON from VLM response: {exc}") from exc


def _encode_images_as_data_urls(image_paths: list[str], max_images: int) -> list[str]:
    out: list[str] = []
    for p in image_paths[:max_images]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        edge = max(h, w)
        if edge > 720:
            scale = 720.0 / float(edge)
            img = cv2.resize(
                img,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        enc = base64.b64encode(buf.tobytes()).decode("ascii")
        out.append(f"data:image/jpeg;base64,{enc}")
    return out


def _strict_json_schema_for_codex(schema: dict) -> dict:
    """Codex JSON schema mode requires object nodes to set additionalProperties=false."""

    def _convert(node):
        if isinstance(node, dict):
            out = {str(k): _convert(v) for k, v in node.items()}
            node_type = out.get("type")
            is_object = (
                node_type == "object"
                or (isinstance(node_type, list) and "object" in node_type)
                or ("properties" in out and "type" not in out)
            )
            if is_object:
                props = out.get("properties")
                if not isinstance(props, dict):
                    props = {}
                out["properties"] = props
                out["required"] = [str(k) for k in props.keys()]
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [_convert(v) for v in node]
        return node

    converted = _convert(schema)
    if not isinstance(converted, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    return converted


def _resolve_native_command(config: ProviderConfig) -> str:
    if config.native_cmd:
        return str(config.native_cmd)

    provider = config.normalized_provider()
    if provider == "codex":
        return os.getenv("NAVIRL_CODEX_CMD", "").strip()
    if provider == "claude":
        return os.getenv("NAVIRL_CLAUDE_CMD", "").strip()
    return os.getenv("NAVIRL_VLM_NATIVE_CMD", "").strip()


def _run_native_json(
    *,
    prompt: str,
    image_paths: list[str],
    schema: dict,
    config: ProviderConfig,
) -> dict:
    cmd_template = _resolve_native_command(config)
    if not cmd_template:
        raise ProviderUnavailableError(
            "No native command configured. Set NAVIRL_VLM_NATIVE_CMD or provider-specific command env var."
        )

    with tempfile.TemporaryDirectory(prefix="navirl-aegis-native-") as td:
        td_path = Path(td)
        prompt_path = td_path / "prompt.json"
        schema_path = td_path / "schema.json"
        output_path = td_path / "response.json"
        selected_images = [str(p) for p in image_paths[: max(0, int(config.max_images))]]
        provider = config.normalized_provider()
        response_schema = _strict_json_schema_for_codex(schema) if provider == "codex" else schema

        prompt_payload = {
            "prompt": prompt,
            "image_paths": selected_images,
            "schema": response_schema,
        }
        prompt_path.write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
        schema_path.write_text(json.dumps(response_schema, indent=2), encoding="utf-8")
        image_flags = " ".join(f"-i {shlex.quote(p)}" for p in selected_images)
        image_paths_json = json.dumps(selected_images)

        fmt = {
            "prompt_file": str(prompt_path),
            "schema_file": str(schema_path),
            "output_file": str(output_path),
            "image_flags": image_flags,
            "image_paths_json": image_paths_json,
        }
        if "{" in cmd_template:
            cmd = shlex.split(cmd_template.format(**fmt))
        else:
            cmd = shlex.split(cmd_template) + [str(prompt_path)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(config.timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderCallError(
                "Native provider command timed out "
                f"after {max(1.0, float(config.timeout_s)):.1f}s: {' '.join(cmd)}"
            ) from exc
        if proc.returncode != 0:
            raise ProviderCallError(
                f"Native provider command failed with code {proc.returncode}: {proc.stderr.strip()}"
            )

        stdout = proc.stdout.strip()
        if stdout:
            return _parse_json_object(stdout)

        if output_path.exists():
            return _parse_json_object(output_path.read_text(encoding="utf-8"))

        raise ProviderCallError("Native provider produced no JSON output.")


def _run_openai_compatible_json(
    *,
    prompt: str,
    image_paths: list[str],
    schema: dict,
    config: ProviderConfig,
) -> dict:
    del schema
    api_key = os.getenv(str(config.api_key_env), "").strip()
    if not api_key:
        raise ProviderUnavailableError(
            f"Missing API key in env var {config.api_key_env} for openai-compatible provider."
        )

    endpoint = (
        str(config.endpoint).strip()
        if config.endpoint
        else os.getenv("NAVIRL_VLM_ENDPOINT", "").strip()
        or "https://api.openai.com/v1/chat/completions"
    )
    model = (
        str(config.model).strip()
        if config.model
        else os.getenv("NAVIRL_VLM_MODEL", "").strip() or "gpt-4.1-mini"
    )

    content: list[dict] = [{"type": "text", "text": prompt}]
    for data_url in _encode_images_as_data_urls(image_paths, max_images=config.max_images):
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Aegis Overseer. Return strict JSON only, with no markdown, no prose outside JSON."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=max(1.0, float(config.timeout_s))) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ProviderCallError(f"Provider HTTP {exc.code}: {body[:240]}") from exc
    except error.URLError as exc:
        raise ProviderCallError(f"Provider network error: {exc}") from exc

    choices = raw.get("choices", [])
    if not choices:
        raise ProviderCallError("Provider response missing choices.")
    msg = choices[0].get("message", {})
    content_out = msg.get("content")
    if isinstance(content_out, list):
        text = ""
        for item in content_out:
            if isinstance(item, dict) and item.get("type") == "text":
                text += str(item.get("text", ""))
        content_out = text
    if not isinstance(content_out, str):
        raise ProviderCallError("Provider message content is not text.")

    return _parse_json_object(content_out)


def run_structured_vlm(
    *,
    prompt: str,
    image_paths: list[str],
    schema: dict,
    config: ProviderConfig,
) -> dict:
    provider = config.normalized_provider()
    if provider in {"codex", "claude", "native"}:
        return _run_native_json(
            prompt=prompt,
            image_paths=image_paths,
            schema=schema,
            config=config,
        )

    if provider in {"openai_compatible", "kimi"}:
        return _run_openai_compatible_json(
            prompt=prompt,
            image_paths=image_paths,
            schema=schema,
            config=config,
        )

    raise ProviderUnavailableError(
        f"Unsupported provider '{config.provider}'. "
        "Use one of: codex, claude, native, openai_compatible, kimi."
    )
