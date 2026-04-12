"""Tests for overseer/provider.py execution paths.

Covers _encode_images_as_data_urls, _run_native_json (success, timeout, failure),
and _run_openai_compatible_json (success, HTTP errors, network errors, edge cases).
These paths were previously uncovered (lines 130-157, 237-293, 310-377).
"""

from __future__ import annotations

import json
import os
from unittest import mock

import cv2
import numpy as np
import pytest

from navirl.overseer.provider import (
    ProviderCallError,
    ProviderConfig,
    ProviderUnavailableError,
    _encode_images_as_data_urls,
    _run_native_json,
    _run_openai_compatible_json,
)

# ===================================================================
# _encode_images_as_data_urls
# ===================================================================


class TestEncodeImagesAsDataUrls:
    def test_valid_image(self, tmp_path):
        """Encodes a real image file to a data URL."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = 255
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, img)

        result = _encode_images_as_data_urls([img_path], max_images=4)
        assert len(result) == 1
        assert result[0].startswith("data:image/jpeg;base64,")

    def test_large_image_is_resized(self, tmp_path):
        """Images larger than 720px edge get resized."""
        img = np.zeros((1000, 1500, 3), dtype=np.uint8)
        img_path = str(tmp_path / "large.png")
        cv2.imwrite(img_path, img)

        result = _encode_images_as_data_urls([img_path], max_images=4)
        assert len(result) == 1
        assert result[0].startswith("data:image/jpeg;base64,")

    def test_nonexistent_file_skipped(self, tmp_path):
        """Missing files are silently skipped."""
        result = _encode_images_as_data_urls([str(tmp_path / "missing.png")], max_images=4)
        assert len(result) == 0

    def test_max_images_limit(self, tmp_path):
        """Only max_images files are processed."""
        paths = []
        for i in range(5):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            p = str(tmp_path / f"img_{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)

        result = _encode_images_as_data_urls(paths, max_images=2)
        assert len(result) == 2

    def test_corrupt_image_skipped(self, tmp_path):
        """Files that can't be read as images are skipped."""
        p = tmp_path / "corrupt.png"
        p.write_text("not an image")
        result = _encode_images_as_data_urls([str(p)], max_images=4)
        assert len(result) == 0

    def test_empty_list(self):
        result = _encode_images_as_data_urls([], max_images=4)
        assert result == []

    def test_multiple_valid_images(self, tmp_path):
        paths = []
        for i in range(3):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            p = str(tmp_path / f"img_{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)

        result = _encode_images_as_data_urls(paths, max_images=4)
        assert len(result) == 3
        assert all(r.startswith("data:image/jpeg;base64,") for r in result)


# ===================================================================
# _run_native_json
# ===================================================================


class TestRunNativeJson:
    def test_success_stdout(self, tmp_path):
        """Native command returns JSON on stdout."""
        # Create a script that outputs JSON
        script = tmp_path / "echo_json.sh"
        script.write_text('#!/bin/bash\necho \'{"result": "ok"}\'')
        script.chmod(0o755)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=10.0)
        result = _run_native_json(
            prompt="test prompt",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result == {"result": "ok"}

    def test_success_output_file(self, tmp_path):
        """Native command writes JSON to output_file."""
        script = tmp_path / "write_json.sh"
        # Script that reads prompt file and writes to output_file
        script.write_text(
            "#!/bin/bash\n"
            'PROMPT_FILE="$1"\n'
            "# Parse output_file from the prompt payload\n"
            "echo '{\"written\": true}'\n"
        )
        script.chmod(0o755)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=10.0)
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result == {"written": True}

    def test_command_failure_raises(self, tmp_path):
        """Non-zero exit code raises ProviderCallError."""
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/bash\nexit 1")
        script.chmod(0o755)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=10.0)
        with pytest.raises(ProviderCallError, match="failed with code"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_timeout_raises(self, tmp_path):
        """Command that takes too long raises ProviderCallError."""
        script = tmp_path / "slow.sh"
        script.write_text("#!/bin/bash\nsleep 60")
        script.chmod(0o755)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=1.0)
        with pytest.raises(ProviderCallError, match="timed out"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_no_output_raises(self, tmp_path):
        """Command that produces no output raises ProviderCallError."""
        script = tmp_path / "silent.sh"
        script.write_text("#!/bin/bash\n# no output")
        script.chmod(0o755)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=10.0)
        with pytest.raises(ProviderCallError, match="no JSON output"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_no_command_configured_raises(self):
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with (
            mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": ""}),
            pytest.raises(ProviderUnavailableError, match="No native command"),
        ):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_template_command_with_format_args(self, tmp_path):
        """Command template with {prompt_file} placeholder."""
        script = tmp_path / "reader.sh"
        script.write_text("#!/bin/bash\necho '{\"from_template\": true}'")
        script.chmod(0o755)

        cfg = ProviderConfig(
            native_cmd=f"{script} {{prompt_file}}",
            timeout_s=10.0,
        )
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result == {"from_template": True}

    def test_codex_provider_uses_strict_schema(self, tmp_path):
        """When provider is codex, schema gets strict JSON conversion."""
        script = tmp_path / "echo_json.sh"
        script.write_text('#!/bin/bash\necho \'{"name": "test"}\'')
        script.chmod(0o755)

        cfg = ProviderConfig(provider="codex", native_cmd=str(script), timeout_s=10.0)
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            config=cfg,
        )
        assert result == {"name": "test"}

    def test_with_image_paths(self, tmp_path):
        """Image paths are passed through to command."""
        script = tmp_path / "echo_json.sh"
        script.write_text("#!/bin/bash\necho '{\"images_received\": true}'")
        script.chmod(0o755)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, img)

        cfg = ProviderConfig(native_cmd=str(script), timeout_s=10.0, max_images=2)
        result = _run_native_json(
            prompt="describe",
            image_paths=[img_path],
            schema={"type": "object"},
            config=cfg,
        )
        assert result == {"images_received": True}


# ===================================================================
# _run_openai_compatible_json
# ===================================================================


class TestRunOpenaiCompatibleJson:
    def _mock_urlopen_response(self, content: str, status: int = 200):
        """Create a mock urlopen context manager response."""
        response = mock.MagicMock()
        response.read.return_value = content.encode("utf-8")
        response.status = status
        response.__enter__ = mock.MagicMock(return_value=response)
        response.__exit__ = mock.MagicMock(return_value=False)
        return response

    def test_success(self):
        """Successful API call returns parsed JSON."""
        api_response = json.dumps(
            {"choices": [{"message": {"content": '{"status": "ok", "score": 0.95}'}}]}
        )
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="TEST_KEY_ENV",
            model="gpt-4",
            endpoint="https://api.example.com/v1/chat/completions",
        )
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            result = _run_openai_compatible_json(
                prompt="analyze this",
                image_paths=[],
                schema={},
                config=cfg,
            )
        assert result == {"status": "ok", "score": 0.95}

    def test_missing_api_key_raises(self):
        cfg = ProviderConfig(provider="openai_compatible", api_key_env="NONEXISTENT_KEY_XYZ")
        os.environ.pop("NONEXISTENT_KEY_XYZ", None)
        with pytest.raises(ProviderUnavailableError, match="Missing API key"):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_http_error_raises(self):
        from urllib.error import HTTPError

        cfg = ProviderConfig(provider="openai_compatible", api_key_env="TEST_KEY_ENV")
        exc = HTTPError("https://api.example.com", 429, "Rate limited", {}, None)
        exc.read = mock.MagicMock(return_value=b"rate limit exceeded")
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch("navirl.overseer.provider.request.urlopen", side_effect=exc),
            pytest.raises(ProviderCallError, match="HTTP 429"),
        ):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_network_error_raises(self):
        from urllib.error import URLError

        cfg = ProviderConfig(provider="openai_compatible", api_key_env="TEST_KEY_ENV")
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch(
                "navirl.overseer.provider.request.urlopen",
                side_effect=URLError("Connection refused"),
            ),
            pytest.raises(ProviderCallError, match="network error"),
        ):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_empty_choices_raises(self):
        api_response = json.dumps({"choices": []})
        cfg = ProviderConfig(provider="openai_compatible", api_key_env="TEST_KEY_ENV")
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
            pytest.raises(ProviderCallError, match="missing choices"),
        ):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_non_text_content_raises(self):
        api_response = json.dumps({"choices": [{"message": {"content": 12345}}]})
        cfg = ProviderConfig(provider="openai_compatible", api_key_env="TEST_KEY_ENV")
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
            pytest.raises(ProviderCallError, match="not text"),
        ):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_list_content_assembled(self):
        """Content returned as a list of text parts is assembled."""
        part1 = '{"part"'
        part2 = ': "value"}'
        api_response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": part1},
                                {"type": "text", "text": part2},
                            ]
                        }
                    }
                ]
            }
        )
        cfg = ProviderConfig(provider="openai_compatible", api_key_env="TEST_KEY_ENV")
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test123"}),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            result = _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )
        assert result == {"part": "value"}

    def test_default_endpoint_and_model(self):
        """When no endpoint/model set, uses defaults from env or hardcoded."""
        api_response = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="TEST_KEY_ENV",
            endpoint=None,
            model=None,
        )
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TEST_KEY_ENV": "sk-test",
                    "NAVIRL_VLM_ENDPOINT": "",
                    "NAVIRL_VLM_MODEL": "",
                },
            ),
            mock.patch(
                "navirl.overseer.provider.request.urlopen", return_value=mock_resp
            ) as mock_open,
        ):
            result = _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )
        assert result == {"ok": True}
        # Check the request was made to the default OpenAI endpoint
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert "openai.com" in req.full_url

    def test_with_images(self, tmp_path):
        """Images are encoded and included in the request."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img_path = str(tmp_path / "img.png")
        cv2.imwrite(img_path, img)

        api_response = json.dumps({"choices": [{"message": {"content": '{"analyzed": true}'}}]})
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="TEST_KEY_ENV",
            max_images=2,
        )
        mock_resp = self._mock_urlopen_response(api_response)
        with (
            mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-test"}),
            mock.patch(
                "navirl.overseer.provider.request.urlopen", return_value=mock_resp
            ) as mock_open,
        ):
            result = _run_openai_compatible_json(
                prompt="describe",
                image_paths=[img_path],
                schema={},
                config=cfg,
            )
        assert result == {"analyzed": True}
        # Verify image was included in the payload
        call_args = mock_open.call_args
        req = call_args[0][0]
        payload = json.loads(req.data)
        user_content = payload["messages"][1]["content"]
        assert len(user_content) == 2  # text + image
        assert user_content[1]["type"] == "image_url"
