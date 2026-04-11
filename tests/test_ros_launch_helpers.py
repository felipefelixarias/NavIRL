from __future__ import annotations

import os
import stat
import textwrap

import pytest
import yaml

from navirl.ros.launch_helpers import (
    create_param_file,
    generate_launch_description,
    setup_workspace,
)

# ---------------------------------------------------------------------------
# generate_launch_description
# ---------------------------------------------------------------------------


class TestGenerateLaunchDescription:
    """Tests for generate_launch_description."""

    def test_default_params(self):
        """Minimal config produces valid launch source with defaults."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "from launch import LaunchDescription" in src
        assert "from launch_ros.actions import Node" in src
        assert 'package="navirl"' in src
        assert 'executable="navirl_node"' in src
        assert 'output="screen"' in src
        assert "'agent_type': 'irl'" in src
        assert "'action_rate': 10.0" in src
        assert "'observation_type': 'lidar'" in src

    def test_custom_agent_config(self):
        """Custom agent_config values appear in the generated source."""
        cfg = {
            "agent_type": "bc",
            "model_path": "/models/bc.pt",
            "action_rate": 20.0,
            "observation_type": "rgbd",
        }
        src = generate_launch_description(cfg)
        assert "'agent_type': 'bc'" in src
        assert "'model_path': '/models/bc.pt'" in src
        assert "'action_rate': 20.0" in src
        assert "'observation_type': 'rgbd'" in src

    def test_custom_executable_and_package(self):
        """Custom executable and package names are reflected."""
        src = generate_launch_description(
            {"agent_type": "irl"},
            executable="my_node",
            package="my_pkg",
        )
        assert 'executable="my_node"' in src
        assert 'package="my_pkg"' in src

    def test_output_log(self):
        """Non-default output mode is reflected."""
        src = generate_launch_description({"agent_type": "irl"}, output="log")
        assert 'output="log"' in src

    def test_namespace(self):
        """Namespace argument appears in the Node constructor."""
        src = generate_launch_description({"agent_type": "irl"}, namespace="robot1")
        assert 'namespace="robot1"' in src

    def test_no_namespace_by_default(self):
        """No namespace= line when namespace is empty string."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "namespace=" not in src

    def test_extra_remappings(self):
        """Extra remappings are rendered as tuples in the Node call."""
        remaps = {"/cmd_vel": "/robot1/cmd_vel", "/scan": "/robot1/scan"}
        src = generate_launch_description({"agent_type": "irl"}, extra_remappings=remaps)
        assert "remappings=" in src
        assert '("/cmd_vel", "/robot1/cmd_vel")' in src
        assert '("/scan", "/robot1/scan")' in src

    def test_no_remappings_by_default(self):
        """No remappings line when extra_remappings is None."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "remappings=" not in src

    def test_extra_config_keys_merged(self):
        """Keys beyond the four standard ones are merged into parameters."""
        cfg = {"agent_type": "irl", "custom_param": 42}
        src = generate_launch_description(cfg)
        assert "'custom_param': 42" in src

    def test_agent_type_defaults_to_irl(self):
        """If agent_type is missing from config, it defaults to 'irl'."""
        src = generate_launch_description({})
        assert "'agent_type': 'irl'" in src

    def test_model_path_defaults_to_empty(self):
        """If model_path is missing, it defaults to empty string."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "'model_path': ''" in src

    def test_returns_string(self):
        """Return type is str."""
        result = generate_launch_description({"agent_type": "irl"})
        assert isinstance(result, str)

    def test_contains_node_constructor(self):
        """Generated source contains a Node() call."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "Node(" in src

    def test_contains_launch_description(self):
        """Generated source contains LaunchDescription."""
        src = generate_launch_description({"agent_type": "irl"})
        assert "LaunchDescription(" in src

    def test_action_rate_cast_to_float(self):
        """action_rate is cast to float even if provided as int."""
        src = generate_launch_description({"agent_type": "irl", "action_rate": 5})
        assert "'action_rate': 5.0" in src


# ---------------------------------------------------------------------------
# create_param_file
# ---------------------------------------------------------------------------


class TestCreateParamFile:
    """Tests for create_param_file."""

    def test_explicit_path(self, tmp_path):
        """Writing to an explicit path returns that path and creates the file."""
        dest = tmp_path / "params.yaml"
        result = create_param_file({"speed": 1.0}, path=dest)
        assert result == str(dest.resolve())
        assert dest.exists()

    def test_yaml_content(self, tmp_path):
        """Written YAML matches ROS2 parameter format."""
        dest = tmp_path / "params.yaml"
        create_param_file({"speed": 1.0, "mode": "fast"}, path=dest)
        data = yaml.safe_load(dest.read_text())
        assert data == {
            "navirl_node": {
                "ros__parameters": {"speed": 1.0, "mode": "fast"},
            }
        }

    def test_custom_node_name(self, tmp_path):
        """Custom node_name is reflected as the top-level YAML key."""
        dest = tmp_path / "params.yaml"
        create_param_file({"x": 1}, path=dest, node_name="my_node")
        data = yaml.safe_load(dest.read_text())
        assert "my_node" in data
        assert data["my_node"]["ros__parameters"]["x"] == 1

    def test_temp_file_created_when_path_is_none(self):
        """When path is None a temp file is created and its path returned."""
        result = create_param_file({"a": 1})
        try:
            assert os.path.isabs(result)
            assert os.path.isfile(result)
            assert "navirl_params_" in os.path.basename(result)
            assert result.endswith(".yaml")
        finally:
            os.unlink(result)

    def test_temp_file_permissions(self):
        """Temp files are created with 0o600 permissions."""
        result = create_param_file({"a": 1})
        try:
            mode = os.stat(result).st_mode
            # Check owner read/write, no group/other access
            assert mode & 0o777 == 0o600
        finally:
            os.unlink(result)

    def test_temp_file_yaml_content(self):
        """Temp file contains valid YAML with correct structure."""
        result = create_param_file({"key": "value"})
        try:
            with open(result) as f:
                data = yaml.safe_load(f)
            assert data["navirl_node"]["ros__parameters"]["key"] == "value"
        finally:
            os.unlink(result)

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories are created if they don't exist."""
        dest = tmp_path / "a" / "b" / "c" / "params.yaml"
        result = create_param_file({"x": 1}, path=dest)
        assert os.path.isfile(result)

    def test_returns_absolute_path(self, tmp_path):
        """Returned path is always absolute."""
        dest = tmp_path / "params.yaml"
        result = create_param_file({"x": 1}, path=dest)
        assert os.path.isabs(result)

    def test_empty_config(self, tmp_path):
        """An empty config dict produces valid YAML with empty parameters."""
        dest = tmp_path / "params.yaml"
        create_param_file({}, path=dest)
        data = yaml.safe_load(dest.read_text())
        assert data["navirl_node"]["ros__parameters"] == {}

    def test_path_as_string(self, tmp_path):
        """Accepts path as a plain string."""
        dest = str(tmp_path / "params.yaml")
        result = create_param_file({"a": 1}, path=dest)
        assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# setup_workspace
# ---------------------------------------------------------------------------


class TestSetupWorkspace:
    """Tests for setup_workspace."""

    def test_returns_absolute_path(self, tmp_path):
        """Returned path is the resolved absolute workspace root."""
        ws = setup_workspace(tmp_path / "ws")
        assert ws.is_absolute()
        assert ws == (tmp_path / "ws").resolve()

    def test_directory_structure(self, tmp_path):
        """All expected directories are created."""
        ws = setup_workspace(tmp_path / "ws")
        pkg = "navirl_ros"
        assert (ws / "src" / pkg).is_dir()
        assert (ws / "src" / pkg / pkg).is_dir()
        assert (ws / "src" / pkg / "resource").is_dir()

    def test_package_xml_exists(self, tmp_path):
        """package.xml is created with correct content."""
        ws = setup_workspace(tmp_path / "ws")
        pkg_xml = ws / "src" / "navirl_ros" / "package.xml"
        assert pkg_xml.is_file()
        content = pkg_xml.read_text()
        assert "<name>navirl_ros</name>" in content
        assert "<version>0.1.0</version>" in content
        assert "ament_python" in content
        assert "<depend>rclpy</depend>" in content

    def test_setup_py_exists(self, tmp_path):
        """setup.py is created with correct package name."""
        ws = setup_workspace(tmp_path / "ws")
        setup_py = ws / "src" / "navirl_ros" / "setup.py"
        assert setup_py.is_file()
        content = setup_py.read_text()
        assert 'package_name = "navirl_ros"' in content
        assert "setuptools" in content

    def test_setup_cfg_exists(self, tmp_path):
        """setup.cfg is created."""
        ws = setup_workspace(tmp_path / "ws")
        setup_cfg = ws / "src" / "navirl_ros" / "setup.cfg"
        assert setup_cfg.is_file()
        content = setup_cfg.read_text()
        assert "navirl_ros" in content

    def test_resource_marker(self, tmp_path):
        """Resource marker file exists (required by ament)."""
        ws = setup_workspace(tmp_path / "ws")
        marker = ws / "src" / "navirl_ros" / "resource" / "navirl_ros"
        assert marker.is_file()

    def test_init_py(self, tmp_path):
        """__init__.py is created with a docstring."""
        ws = setup_workspace(tmp_path / "ws")
        init = ws / "src" / "navirl_ros" / "navirl_ros" / "__init__.py"
        assert init.is_file()
        assert "ROS2 package wrapper" in init.read_text()

    def test_custom_package_name(self, tmp_path):
        """Custom package_name is used throughout the workspace."""
        ws = setup_workspace(tmp_path / "ws", package_name="my_robot")
        assert (ws / "src" / "my_robot").is_dir()
        assert (ws / "src" / "my_robot" / "my_robot" / "__init__.py").is_file()
        assert (ws / "src" / "my_robot" / "resource" / "my_robot").is_file()
        pkg_xml = (ws / "src" / "my_robot" / "package.xml").read_text()
        assert "<name>my_robot</name>" in pkg_xml

    def test_idempotent(self, tmp_path):
        """Calling setup_workspace twice does not raise."""
        ws1 = setup_workspace(tmp_path / "ws")
        ws2 = setup_workspace(tmp_path / "ws")
        assert ws1 == ws2
