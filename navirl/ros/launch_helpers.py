"""Utilities for generating ROS2 launch configurations.

These helpers produce launch-file content and parameter YAML files
programmatically, making it easy to spin up NavIRL nodes from Python
without manually authoring XML or YAML.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # PyYAML -- already a NavIRL dependency


# ---------------------------------------------------------------------------
# Launch description generator
# ---------------------------------------------------------------------------

def generate_launch_description(
    agent_config: Dict[str, Any],
    *,
    executable: str = "navirl_node",
    package: str = "navirl",
    namespace: str = "",
    output: str = "screen",
    extra_remappings: Optional[Dict[str, str]] = None,
) -> str:
    """Return a complete ROS2 Python launch-file string.

    Parameters
    ----------
    agent_config : dict
        Must include at least ``agent_type``.  May also contain
        ``model_path``, ``action_rate``, ``observation_type``, and any
        extra ROS2 parameters.
    executable : str
        Name of the ROS2 executable to launch.
    package : str
        Name of the ROS2 package containing the executable.
    namespace : str
        Optional ROS2 namespace.
    output : str
        ``"screen"`` (default) or ``"log"``.
    extra_remappings : dict or None
        Topic remappings as ``{from: to}``.

    Returns
    -------
    str
        Python source code for a ROS2 launch file.
    """
    params = {
        "agent_type": agent_config.get("agent_type", "irl"),
        "model_path": agent_config.get("model_path", ""),
        "action_rate": float(agent_config.get("action_rate", 10.0)),
        "observation_type": agent_config.get("observation_type", "lidar"),
    }
    # Merge any additional parameters
    for key, val in agent_config.items():
        if key not in params:
            params[key] = val

    remappings_lines = ""
    if extra_remappings:
        pairs = ", ".join(
            f'("{src}", "{dst}")' for src, dst in extra_remappings.items()
        )
        remappings_lines = f"        remappings=[{pairs}],"

    params_repr = repr(params)

    namespace_arg = f'        namespace="{namespace}",' if namespace else ""

    launch_src = textwrap.dedent(f"""\
        \"\"\"Auto-generated NavIRL ROS2 launch file.\"\"\"

        from launch import LaunchDescription
        from launch_ros.actions import Node


        def generate_launch_description():
            return LaunchDescription([
                Node(
                    package="{package}",
                    executable="{executable}",
        {namespace_arg}
                    output="{output}",
                    parameters=[{params_repr}],
        {remappings_lines}
                ),
            ])
    """)
    return launch_src


# ---------------------------------------------------------------------------
# Parameter file writer
# ---------------------------------------------------------------------------

def create_param_file(
    config: Dict[str, Any],
    path: str | Path | None = None,
    node_name: str = "navirl_node",
) -> str:
    """Write a ROS2 YAML parameter file and return the file path.

    Parameters
    ----------
    config : dict
        Flat dictionary of parameter names to values.
    path : str or Path or None
        Destination path.  If *None*, writes to a temp file.
    node_name : str
        The fully-qualified node name used as the YAML key.

    Returns
    -------
    str
        Absolute path to the written YAML file.
    """
    # ROS2 YAML format:
    # <node_name>:
    #   ros__parameters:
    #     key: value
    yaml_dict = {
        node_name: {
            "ros__parameters": dict(config),
        }
    }

    if path is None:
        import tempfile

        fd, path = tempfile.mkstemp(prefix="navirl_params_", suffix=".yaml")
        os.close(fd)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    return str(path.resolve())


# ---------------------------------------------------------------------------
# Workspace scaffolding
# ---------------------------------------------------------------------------

def setup_workspace(
    path: str | Path,
    package_name: str = "navirl_ros",
) -> Path:
    """Create a minimal colcon workspace layout.

    The resulting tree::

        <path>/
        ├── src/
        │   └── <package_name>/
        │       ├── package.xml
        │       ├── setup.py
        │       ├── setup.cfg
        │       ├── resource/
        │       │   └── <package_name>
        │       └── <package_name>/
        │           └── __init__.py
        └── (build/, install/, log/ are created by colcon)

    Parameters
    ----------
    path : str or Path
        Root directory for the workspace.
    package_name : str
        Name of the ROS2 Python package to create.

    Returns
    -------
    Path
        Absolute path to the workspace root.
    """
    ws = Path(path).resolve()
    pkg_dir = ws / "src" / package_name
    pkg_py = pkg_dir / package_name
    resource_dir = pkg_dir / "resource"

    for d in (pkg_py, resource_dir):
        d.mkdir(parents=True, exist_ok=True)

    # package.xml
    package_xml = textwrap.dedent(f"""\
        <?xml version="1.0"?>
        <package format="3">
          <name>{package_name}</name>
          <version>0.1.0</version>
          <description>NavIRL ROS2 integration package</description>
          <maintainer email="navirl@example.com">NavIRL Team</maintainer>
          <license>MIT</license>

          <depend>rclpy</depend>
          <depend>geometry_msgs</depend>
          <depend>sensor_msgs</depend>
          <depend>nav_msgs</depend>
          <depend>visualization_msgs</depend>
          <depend>std_msgs</depend>

          <buildtool_depend>ament_python</buildtool_depend>

          <export>
            <build_type>ament_python</build_type>
          </export>
        </package>
    """)
    (pkg_dir / "package.xml").write_text(package_xml)

    # setup.py
    setup_py = textwrap.dedent(f"""\
        from setuptools import setup

        package_name = "{package_name}"

        setup(
            name=package_name,
            version="0.1.0",
            packages=[package_name],
            data_files=[
                ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"]),
            ],
            install_requires=["setuptools"],
            zip_safe=True,
            maintainer="NavIRL Team",
            maintainer_email="navirl@example.com",
            description="NavIRL ROS2 integration package",
            license="MIT",
            entry_points={{
                "console_scripts": [
                    "navirl_node = navirl.ros.node:main",
                ],
            }},
        )
    """)
    (pkg_dir / "setup.py").write_text(setup_py)

    # setup.cfg
    setup_cfg = textwrap.dedent(f"""\
        [develop]
        script_dir=$base/lib/{package_name}
        [install]
        install_scripts=$base/lib/{package_name}
    """)
    (pkg_dir / "setup.cfg").write_text(setup_cfg)

    # resource marker (required by ament)
    (resource_dir / package_name).touch()

    # Python package __init__
    (pkg_py / "__init__.py").write_text(
        f'"""ROS2 package wrapper for NavIRL."""\n'
    )

    return ws
