"""Tests for uncovered paths in navirl.config.serialization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from navirl.config.serialization import (
    _auto_cast,
    _flatten_to_args,
    _import_toml_read,
    _import_toml_write,
    _import_yaml,
    _normalize_output_path,
    _resolve_existing_path,
    cli_args_to_config,
    config_to_cli_args,
    diff_configs,
    load_config,
    merge_configs,
    save_config,
)

# ---------------------------------------------------------------------------
# save_config error paths
# ---------------------------------------------------------------------------


class TestSaveConfigErrors:
    """Cover error branches in save_config."""

    def test_mkdir_os_error(self, tmp_path):
        """OSError during directory creation should propagate."""
        path = tmp_path / "sub" / "config.json"
        with (
            patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
            pytest.raises(OSError, match="Cannot create directory"),
        ):
            save_config({"key": "val"}, path)

    def test_json_write_os_error(self, tmp_path):
        """OSError writing JSON file."""
        path = tmp_path / "config.json"
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="Cannot write JSON"),
        ):
            save_config({"key": "val"}, path)

    def test_json_write_type_error(self, tmp_path):
        """TypeError during JSON serialization."""
        path = tmp_path / "config.json"
        with (
            patch("json.dump", side_effect=TypeError("not serializable")),
            pytest.raises(ValueError, match="cannot be serialized to JSON"),
        ):
            save_config({"key": "val"}, path)

    def test_yaml_write_os_error(self, tmp_path):
        """OSError writing YAML file."""
        path = tmp_path / "config.yaml"
        mock_yaml = MagicMock()
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(OSError, match="Cannot write YAML"),
        ):
            save_config({"key": "val"}, path)

    def test_yaml_write_value_error(self, tmp_path):
        """ValueError during YAML serialization."""
        path = tmp_path / "config.yaml"
        mock_yaml = MagicMock()
        mock_yaml.dump.side_effect = ValueError("bad value")
        with (
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(ValueError, match="cannot be serialized to YAML"),
        ):
            save_config({"key": "val"}, path)

    def test_yaml_write_generic_error(self, tmp_path):
        """Generic exception during YAML serialization."""
        path = tmp_path / "config.yaml"
        mock_yaml = MagicMock()
        mock_yaml.dump.side_effect = RuntimeError("yaml internal error")
        with (
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(ValueError, match="cannot be serialized to YAML"),
        ):
            save_config({"key": "val"}, path)

    def test_toml_write_os_error(self, tmp_path):
        """OSError writing TOML file."""
        path = tmp_path / "config.toml"
        mock_toml = MagicMock()
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            patch(
                "navirl.config.serialization._import_toml_write",
                return_value=mock_toml,
            ),
            pytest.raises(OSError, match="Cannot write TOML"),
        ):
            save_config({"key": "val"}, path)

    def test_toml_write_type_error(self, tmp_path):
        """TypeError during TOML serialization."""
        path = tmp_path / "config.toml"
        mock_toml = MagicMock()
        mock_toml.dump.side_effect = TypeError("not serializable")
        with (
            patch(
                "navirl.config.serialization._import_toml_write",
                return_value=mock_toml,
            ),
            pytest.raises(ValueError, match="cannot be serialized to TOML"),
        ):
            save_config({"key": "val"}, path)

    def test_toml_write_generic_error(self, tmp_path):
        """Generic exception during TOML serialization."""
        path = tmp_path / "config.toml"
        mock_toml = MagicMock()
        mock_toml.dump.side_effect = RuntimeError("toml internal error")
        with (
            patch(
                "navirl.config.serialization._import_toml_write",
                return_value=mock_toml,
            ),
            pytest.raises(ValueError, match="cannot be serialized to TOML"),
        ):
            save_config({"key": "val"}, path)


# ---------------------------------------------------------------------------
# load_config error paths
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    """Cover error branches in load_config."""

    def test_json_read_os_error(self, tmp_path):
        """OSError reading JSON file."""
        path = tmp_path / "config.json"
        path.write_text("{}")
        with (
            patch("builtins.open", side_effect=OSError("permission denied")),
            pytest.raises(OSError, match="Cannot read JSON"),
        ):
            load_config(path)

    def test_yaml_non_dict(self, tmp_path):
        """YAML file containing non-dict should raise ValueError."""
        path = tmp_path / "config.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must contain a dictionary"):
            load_config(path)

    def test_yaml_read_os_error(self, tmp_path):
        """OSError reading YAML file."""
        path = tmp_path / "config.yaml"
        path.write_text("key: val")
        with (
            patch("builtins.open", side_effect=OSError("read error")),
            patch(
                "navirl.config.serialization._import_yaml",
                return_value=MagicMock(),
            ),
            pytest.raises(OSError, match="Cannot read YAML"),
        ):
            load_config(path)

    def test_yaml_read_value_error(self, tmp_path):
        """ValueError reading YAML."""
        path = tmp_path / "config.yaml"
        path.write_text("key: val")
        mock_yaml = MagicMock()
        mock_yaml.safe_load.side_effect = ValueError("bad yaml")
        with (
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(ValueError, match="Invalid YAML"),
        ):
            load_config(path)

    def test_yaml_read_generic_yaml_error(self, tmp_path):
        """Generic exception with 'yaml' in message during YAML read."""
        path = tmp_path / "config.yaml"
        path.write_text("key: val")
        mock_yaml = MagicMock()
        mock_yaml.safe_load.side_effect = RuntimeError("yaml parse failed")
        with (
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(ValueError, match="Invalid YAML"),
        ):
            load_config(path)

    def test_yaml_read_generic_non_yaml_error(self, tmp_path):
        """Generic exception without 'yaml' in message during YAML read."""
        path = tmp_path / "config.yaml"
        path.write_text("key: val")
        mock_yaml = MagicMock()
        mock_yaml.safe_load.side_effect = RuntimeError("unknown failure")
        with (
            patch("navirl.config.serialization._import_yaml", return_value=mock_yaml),
            pytest.raises(OSError, match="Cannot read YAML"),
        ):
            load_config(path)

    def test_toml_non_dict(self, tmp_path):
        """TOML loader returning non-dict should raise ValueError."""
        path = tmp_path / "config.toml"
        path.write_bytes(b"key = 'val'")
        mock_loader = MagicMock(return_value=["not", "a", "dict"])
        with (
            patch(
                "navirl.config.serialization._import_toml_read",
                return_value=mock_loader,
            ),
            pytest.raises(ValueError, match="must contain a dictionary"),
        ):
            load_config(path)

    def test_toml_read_os_error(self, tmp_path):
        """OSError reading TOML file."""
        path = tmp_path / "config.toml"
        path.write_bytes(b"key = 'val'")
        with (
            patch("builtins.open", side_effect=OSError("read error")),
            patch(
                "navirl.config.serialization._import_toml_read",
                return_value=MagicMock(),
            ),
            pytest.raises(OSError, match="Cannot read TOML"),
        ):
            load_config(path)

    def test_toml_read_value_error(self, tmp_path):
        """ValueError reading TOML."""
        path = tmp_path / "config.toml"
        path.write_bytes(b"key = 'val'")
        mock_loader = MagicMock(side_effect=ValueError("bad toml"))
        with (
            patch(
                "navirl.config.serialization._import_toml_read",
                return_value=mock_loader,
            ),
            pytest.raises(ValueError, match="Invalid TOML"),
        ):
            load_config(path)

    def test_toml_read_generic_toml_error(self, tmp_path):
        """Generic exception with 'toml' in message during TOML read."""
        path = tmp_path / "config.toml"
        path.write_bytes(b"key = 'val'")
        mock_loader = MagicMock(side_effect=RuntimeError("toml parse failed"))
        with (
            patch(
                "navirl.config.serialization._import_toml_read",
                return_value=mock_loader,
            ),
            pytest.raises(ValueError, match="Invalid TOML"),
        ):
            load_config(path)

    def test_toml_read_generic_non_toml_error(self, tmp_path):
        """Generic exception without 'toml' in message during TOML read."""
        path = tmp_path / "config.toml"
        path.write_bytes(b"key = 'val'")
        mock_loader = MagicMock(side_effect=RuntimeError("unknown"))
        with (
            patch(
                "navirl.config.serialization._import_toml_read",
                return_value=mock_loader,
            ),
            pytest.raises(OSError, match="Cannot read TOML"),
        ):
            load_config(path)


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


class TestImportHelpers:
    """Cover import fallback chains."""

    def test_import_yaml_missing(self):
        """Missing yaml should raise ImportError."""
        import sys

        with patch.dict(sys.modules, {"yaml": None}), pytest.raises(ImportError, match="PyYAML"):
            _import_yaml()

    def test_import_toml_write_fallback_to_toml(self):
        """When tomli_w is missing, fall back to toml."""
        import sys

        mock_toml = MagicMock()
        with patch.dict(sys.modules, {"tomli_w": None, "toml": mock_toml}):
            result = _import_toml_write()
            assert result is mock_toml

    def test_import_toml_write_all_missing(self):
        """When all toml writers are missing, raise ImportError."""
        import sys

        with (
            patch.dict(sys.modules, {"tomli_w": None, "toml": None}),
            pytest.raises(ImportError, match="tomli-w or toml"),
        ):
            _import_toml_write()

    def test_import_toml_read_fallback_to_tomli(self):
        """When tomllib is missing, fall back to tomli."""
        import sys

        mock_tomli = MagicMock()
        with patch.dict(sys.modules, {"tomllib": None, "tomli": mock_tomli}):
            result = _import_toml_read()
            assert result is mock_tomli.load

    def test_import_toml_read_fallback_to_toml(self):
        """When tomllib and tomli are missing, fall back to toml wrapper."""
        import sys

        mock_toml = MagicMock()
        mock_toml.loads.return_value = {"key": "val"}
        with patch.dict(sys.modules, {"tomllib": None, "tomli": None, "toml": mock_toml}):
            loader = _import_toml_read()
            # The loader should be a wrapper function
            assert callable(loader)
            # Test the wrapper reads and decodes
            mock_fh = MagicMock()
            mock_fh.read.return_value = b'key = "val"'
            loader(mock_fh)
            mock_toml.loads.assert_called_once()

    def test_import_toml_read_all_missing(self):
        """When all toml readers are missing, raise ImportError."""
        import sys

        with (
            patch.dict(sys.modules, {"tomllib": None, "tomli": None, "toml": None}),
            pytest.raises(ImportError, match="tomllib"),
        ):
            _import_toml_read()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Cover remaining helper function branches."""

    def test_resolve_existing_path_no_suffix_no_match(self, tmp_path):
        """Path without suffix that doesn't match any extension returns as-is."""
        path = tmp_path / "nonexistent"
        result = _resolve_existing_path(path)
        assert result == path

    def test_normalize_output_path_unknown_format(self):
        """Unknown format returns path unchanged."""
        path = Path("config")
        result = _normalize_output_path(path, "xml")
        assert result == path

    def test_flatten_bool_false_omitted(self):
        """Boolean False values should be omitted from CLI args."""
        out: list[str] = []
        _flatten_to_args({"flag": False}, "--", "", out)
        assert out == []

    def test_auto_cast_yes_no(self):
        assert _auto_cast("yes") is True
        assert _auto_cast("no") is False
        assert _auto_cast("null") is None

    def test_cli_args_non_flag_skipped(self):
        """Tokens not starting with -- should be skipped."""
        result = cli_args_to_config(["positional", "--key", "val", "other"])
        assert result == {"key": "val"}
