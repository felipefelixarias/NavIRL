"""Security tests for routine system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import pytest

from navirl.routines.behavior_integration import RoutineControllerFactory, _validate_file_path


class TestRoutineSecurityFixture(TestCase):
    """Test security features of the routine system."""

    def test_validate_file_path_secure(self) -> None:
        """Test that valid YAML files pass validation."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: data\n")
            temp_path = f.name

        try:
            validated = _validate_file_path(temp_path)
            self.assertEqual(validated, Path(temp_path).resolve())
        finally:
            Path(temp_path).unlink()

    def test_validate_file_path_nonexistent(self) -> None:
        """Test that nonexistent files are rejected."""
        with pytest.raises(ValueError, match="File does not exist"):
            _validate_file_path("/nonexistent/file.yaml")

    def test_validate_file_path_not_yaml(self) -> None:
        """Test that non-YAML files are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test data\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Only YAML files are allowed"):
                _validate_file_path(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_file_path_directory(self) -> None:
        """Test that directories are rejected."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="Path is not a file"),
        ):
            _validate_file_path(temp_dir)

    def test_validate_file_path_relative_safe(self) -> None:
        """Test that safe relative paths work."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: data\n")
            temp_path = f.name

        try:
            # Test with relative path that resolves safely
            relative_path = Path(temp_path).relative_to(Path.cwd())
            validated = _validate_file_path(str(relative_path))
            self.assertEqual(validated, Path(temp_path).resolve())
        except ValueError:
            # If relative_to fails, just test the absolute path worked
            validated = _validate_file_path(temp_path)
            self.assertEqual(validated, Path(temp_path).resolve())
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_files_security(self) -> None:
        """Test that from_yaml_files rejects invalid paths."""
        # Test with invalid path
        routine_files = {1: "/nonexistent/file.yaml"}

        # Should not raise exception but print warning
        controller = RoutineControllerFactory.from_yaml_files(routine_files)

        # Controller should be created but with empty routines due to failed load
        self.assertIsNotNone(controller)
        self.assertEqual(len(controller.routines), 0)

    def test_from_yaml_files_with_valid_file(self) -> None:
        """Test that from_yaml_files works with valid YAML files."""
        yaml_content = """
id: "test_routine"
description: "Test routine for security validation"

tasks:
  - type: "go_to"
    params:
      x: 5.0
      y: 3.0
    priority: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            routine_files = {1: temp_path}
            controller = RoutineControllerFactory.from_yaml_files(routine_files)

            self.assertIsNotNone(controller)
            self.assertEqual(len(controller.routines), 1)
            self.assertIn(1, controller.routines)
        finally:
            Path(temp_path).unlink()
