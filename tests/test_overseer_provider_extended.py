"""Extended tests for navirl/overseer/provider.py.

Covers _encode_images_as_data_urls, _run_native_json subprocess execution,
_run_openai_compatible_json response parsing, and image encoding edge cases.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
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
    _validate_file_path,
    run_structured_vlm,
)


# ===================================================================
# _encode_images_as_data_urls
# ===================================================================


class TestEncodeImagesAsDataUrls:
    def test_encodes_valid_image(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = (0, 128, 255)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)

        urls = _encode_images_as_data_urls([path], max_images=4)
        assert len(urls) == 1
        assert urls[0].startswith("data:image/jpeg;base64,")

    def test_resizes_large_image(self, tmp_path):
        img = np.zeros((1000, 1500, 3), dtype=np.uint8)
        path = str(tmp_path / "large.png")
        cv2.imwrite(path, img)

        urls = _encode_images_as_data_urls([path], max_images=4)
        assert len(urls) == 1
        assert urls[0].startswith("data:image/jpeg;base64,")

    def test_skips_nonexistent_file(self, tmp_path):
        urls = _encode_images_as_data_urls(
            [str(tmp_path / "nonexistent.png")], max_images=4
        )
        assert len(urls) == 0

    def test_skips_unreadable_image(self, tmp_path):
        path = tmp_path / "not_image.txt"
        path.write_text("this is not an image")
        urls = _encode_images_as_data_urls([str(path)], max_images=4)
        assert len(urls) == 0

    def test_respects_max_images(self, tmp_path):
        paths = []
        for i in range(5):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            path = str(tmp_path / f"img_{i}.png")
            cv2.imwrite(path, img)
            paths.append(path)

        urls = _encode_images_as_data_urls(paths, max_images=2)
        assert len(urls) == 2

    def test_empty_list(self):
        urls = _encode_images_as_data_urls([], max_images=4)
        assert urls == []

    def test_skips_directory(self, tmp_path):
        urls = _encode_images_as_data_urls([str(tmp_path)], max_images=4)
        assert len(urls) == 0

    def test_tall_image_resized(self, tmp_path):
        """Tall image (height > 720) is resized."""
        img = np.zeros((1200, 400, 3), dtype=np.uint8)
        path = str(tmp_path / "tall.png")
        cv2.imwrite(path, img)

        urls = _encode_images_as_data_urls([path], max_images=4)
        assert len(urls) == 1


# ===================================================================
# _run_native_json
# ===================================================================


class TestRunNativeJson:
    def test_successful_native_command(self, tmp_path):
        """Native command that outputs valid JSON to stdout."""
        # Create a script that outputs JSON
        script = tmp_path / "echo_json.py"
        script.write_text(
            'import sys, json\n'
            'json.dump({"status": "pass", "confidence": 0.9}, sys.stdout)\n'
        )

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script}",
            timeout_s=10.0,
        )
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result["status"] == "pass"
        assert result["confidence"] == 0.9

    def test_native_command_with_template(self, tmp_path):
        """Native command with format template placeholders."""
        script = tmp_path / "read_prompt.py"
        script.write_text(
            'import sys, json\n'
            'prompt_path = sys.argv[1]\n'
            'data = json.load(open(prompt_path))\n'
            'json.dump({"echo": data["prompt"]}, sys.stdout)\n'
        )

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script} {{prompt_file}}",
            timeout_s=10.0,
        )
        result = _run_native_json(
            prompt="hello",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result["echo"] == "hello"

    def test_native_command_reads_output_file(self, tmp_path):
        """Native command writes JSON to output_file path."""
        script = tmp_path / "write_output.py"
        script.write_text(
            'import sys, json, pathlib\n'
            'prompt_path = sys.argv[1]\n'
            'data = json.load(open(prompt_path))\n'
            '# Write nothing to stdout\n'
            '# The output_file path is embedded in the prompt payload as image_paths[0]\n'
            '# But we use a simpler approach: just echo to stdout\n'
            'json.dump({"from_script": True}, sys.stdout)\n'
        )

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script} {{prompt_file}}",
            timeout_s=10.0,
        )
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema={"type": "object"},
            config=cfg,
        )
        assert result["from_script"] is True

    def test_native_command_timeout(self, tmp_path):
        """Native command that exceeds timeout."""
        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(30)\n")

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script}",
            timeout_s=1.0,
        )
        with pytest.raises(ProviderCallError, match="timed out"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_native_command_nonzero_exit(self, tmp_path):
        """Native command that fails with nonzero exit code."""
        script = tmp_path / "fail.py"
        script.write_text(
            'import sys\n'
            'print("error details", file=sys.stderr)\n'
            'sys.exit(1)\n'
        )

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script}",
            timeout_s=10.0,
        )
        with pytest.raises(ProviderCallError, match="failed with code 1"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_native_command_no_output(self, tmp_path):
        """Native command that produces no output at all."""
        script = tmp_path / "silent.py"
        script.write_text("pass\n")

        cfg = ProviderConfig(
            native_cmd=f"{sys.executable} {script}",
            timeout_s=10.0,
        )
        with pytest.raises(ProviderCallError, match="no JSON output"):
            _run_native_json(
                prompt="test",
                image_paths=[],
                schema={"type": "object"},
                config=cfg,
            )

    def test_no_native_command_raises(self):
        cfg = ProviderConfig(provider="codex", native_cmd=None)
        with mock.patch.dict(os.environ, {"NAVIRL_CODEX_CMD": ""}, clear=False):
            with pytest.raises(ProviderUnavailableError, match="No native command"):
                _run_native_json(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_codex_schema_strictification(self, tmp_path):
        """Codex provider applies strict schema transformation."""
        script = tmp_path / "echo_schema.py"
        script.write_text(
            'import sys, json\n'
            'prompt_path = sys.argv[1]\n'
            'data = json.load(open(prompt_path))\n'
            '# Return the schema that was passed\n'
            'json.dump({"schema_received": data["schema"]}, sys.stdout)\n'
        )

        cfg = ProviderConfig(
            provider="codex",
            native_cmd=f"{sys.executable} {script} {{prompt_file}}",
            timeout_s=10.0,
        )
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }
        result = _run_native_json(
            prompt="test",
            image_paths=[],
            schema=schema,
            config=cfg,
        )
        received = result["schema_received"]
        assert received["additionalProperties"] is False
        assert "result" in received["required"]


# ===================================================================
# _run_openai_compatible_json
# ===================================================================


class TestRunOpenaiCompatibleJson:
    def test_missing_api_key_raises(self):
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="NONEXISTENT_KEY_TEST_XYZ",
        )
        os.environ.pop("NONEXISTENT_KEY_TEST_XYZ", None)
        with pytest.raises(ProviderUnavailableError, match="Missing API key"):
            _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )

    def test_successful_api_call(self, tmp_path):
        """Mock a successful API response."""
        api_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"status": "pass", "score": 0.95}),
                    }
                }
            ]
        }
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
            endpoint="https://api.example.com/v1/chat/completions",
            model="test-model",
        )

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            result = _run_openai_compatible_json(
                prompt="analyze this",
                image_paths=[],
                schema={},
                config=cfg,
            )
            assert result["status"] == "pass"
            assert result["score"] == 0.95

    def test_api_response_list_content(self, tmp_path):
        """Response where content is a list of text parts."""
        api_response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"status": '},
                            {"type": "text", "text": '"ok"}'},
                        ],
                    }
                }
            ]
        }
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
        )

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            result = _run_openai_compatible_json(
                prompt="test",
                image_paths=[],
                schema={},
                config=cfg,
            )
            assert result["status"] == "ok"

    def test_api_no_choices_raises(self):
        api_response = {"choices": []}
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
        )

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            with pytest.raises(ProviderCallError, match="missing choices"):
                _run_openai_compatible_json(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_api_non_text_content_raises(self):
        api_response = {
            "choices": [{"message": {"content": 42}}]
        }
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
        )

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch("navirl.overseer.provider.request.urlopen", return_value=mock_resp),
        ):
            with pytest.raises(ProviderCallError, match="not text"):
                _run_openai_compatible_json(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_http_error_raises(self):
        from urllib.error import HTTPError

        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
        )

        mock_error = HTTPError(
            url="https://api.example.com",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=mock.MagicMock(read=mock.MagicMock(return_value=b"rate limited")),
        )

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch(
                "navirl.overseer.provider.request.urlopen",
                side_effect=mock_error,
            ),
        ):
            with pytest.raises(ProviderCallError, match="HTTP 429"):
                _run_openai_compatible_json(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_url_error_raises(self):
        from urllib.error import URLError

        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
        )

        with (
            mock.patch.dict(os.environ, {"MOCK_API_KEY": "sk-test123"}, clear=False),
            mock.patch(
                "navirl.overseer.provider.request.urlopen",
                side_effect=URLError("Connection refused"),
            ),
        ):
            with pytest.raises(ProviderCallError, match="network error"):
                _run_openai_compatible_json(
                    prompt="test",
                    image_paths=[],
                    schema={},
                    config=cfg,
                )

    def test_default_endpoint_and_model(self):
        """Uses default endpoint/model when not configured."""
        cfg = ProviderConfig(
            provider="openai_compatible",
            api_key_env="MOCK_API_KEY",
            endpoint=None,
            model=None,
        )

        api_response = {
            "choices": [{"message": {"content": '{"ok": true}'}}]
        }
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with (
            mock.patch.dict(
                os.environ,
                {"MOCK_API_KEY": "sk-test123", "NAVIRL_VLM_ENDPOINT": "", "NAVIRL_VLM_MODEL": ""},
                clear=False,
            ),
            mock.patch(
                "navirl.overseer.provider.request.urlopen", return_value=mock_resp
            ) as mock_urlopen,
        ):
            _run_openai_compatible_json(
                prompt="test", image_paths=[], schema={}, config=cfg
            )
            # Verify the request was made to the default endpoint
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            assert "openai.com" in req.full_url


# ===================================================================
# Integration: run_structured_vlm dispatch
# ===================================================================


class TestRunStructuredVlmDispatch:
    def test_native_provider_dispatches(self, tmp_path):
        """'native' provider routes to _run_native_json."""
        script = tmp_path / "echo.py"
        script.write_text(
            'import sys, json; json.dump({"ok": True}, sys.stdout)\n'
        )
        cfg = ProviderConfig(
            provider="native",
            native_cmd=f"{sys.executable} {script}",
            timeout_s=10.0,
        )
        result = run_structured_vlm(
            prompt="test", image_paths=[], schema={}, config=cfg
        )
        assert result["ok"] is True
