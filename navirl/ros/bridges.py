"""Simulator bridges -- interface NavIRL with physics simulators via ROS2.

Each bridge provides a uniform API:

    connect()            -- establish the connection / verify readiness
    reset()              -- reset the environment to an initial state
    step(action)         -- advance one simulation step
    get_observation()    -- return the latest observation dict
    send_action(action)  -- send a velocity command to the simulated robot
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded ROS2 imports
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]


def _ensure_ros2(cls_name: str) -> None:
    if not _ROS2_AVAILABLE:
        raise ImportError(
            f"{cls_name} requires ROS2 (rclpy).  "
            "Install ROS2 and source the workspace first."
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _SimBridgeBase(abc.ABC):
    """Interface shared by all simulator bridges."""

    def __init__(self, node_name: str = "navirl_bridge") -> None:
        self._node_name = node_name
        self._node: Any = None
        self._connected: bool = False

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the bridge connection."""

    @abc.abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the simulator and return the initial observation."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """Apply *action* and return ``{obs, reward, done, info}``."""

    @abc.abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Return the latest observation without stepping."""

    @abc.abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Send a velocity command to the robot."""

    @property
    def connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# Gazebo bridge
# ---------------------------------------------------------------------------

class GazeboBridge(_SimBridgeBase):
    """Interface with Gazebo Classic / Ignition Gazebo through ROS2 services.

    Wraps the common Gazebo ROS2 service calls:

    * ``/spawn_entity``  -- spawn the robot URDF
    * ``/reset_simulation`` or ``/reset_world``
    * ``/get_entity_state``
    """

    def __init__(
        self,
        node_name: str = "navirl_gazebo_bridge",
        robot_model: str = "turtlebot3_burger",
        world_name: str = "default",
    ) -> None:
        _ensure_ros2("GazeboBridge")
        super().__init__(node_name)
        self._robot_model = robot_model
        self._world_name = world_name
        self._latest_obs: Dict[str, Any] = {}

    def connect(self) -> None:
        """Initialize rclpy (if needed) and create the helper node."""
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(self._node_name)
        logger.info("GazeboBridge: node created (%s)", self._node_name)

        # Verify Gazebo is reachable by listing services
        svc_names = [n for n, _ in self._node.get_service_names_and_types()]
        gazebo_services = [s for s in svc_names if "gazebo" in s.lower() or "spawn" in s.lower()]
        if gazebo_services:
            logger.info("GazeboBridge: found Gazebo services: %s", gazebo_services)
        else:
            logger.warning(
                "GazeboBridge: no Gazebo services detected -- "
                "ensure Gazebo is running with ROS2 plugins loaded."
            )
        self._connected = True

    def reset(self) -> Dict[str, Any]:
        """Call the Gazebo reset service and return the initial observation."""
        if not self._connected:
            raise RuntimeError("Call connect() before reset().")

        try:
            from std_srvs.srv import Empty as EmptySrv

            client = self._node.create_client(EmptySrv, "/reset_simulation")
            if client.wait_for_service(timeout_sec=5.0):
                future = client.call_async(EmptySrv.Request())
                rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
                logger.info("GazeboBridge: simulation reset.")
            else:
                logger.warning("GazeboBridge: /reset_simulation service not available.")
        except Exception as exc:
            logger.error("GazeboBridge reset failed: %s", exc)

        self._latest_obs = {}
        return self.get_observation()

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self.send_action(action)
        # Allow physics to advance (one-shot spin)
        rclpy.spin_once(self._node, timeout_sec=0.05)
        obs = self.get_observation()
        return {"obs": obs, "reward": 0.0, "done": False, "info": {}}

    def get_observation(self) -> Dict[str, Any]:
        """Spin once to pick up the latest messages and return cached obs."""
        if self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.02)
        return dict(self._latest_obs)

    def send_action(self, action: np.ndarray) -> None:
        if self._node is None:
            raise RuntimeError("Call connect() first.")
        from geometry_msgs.msg import Twist

        pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        twist = Twist()
        action = np.asarray(action, dtype=np.float64).ravel()
        twist.linear.x = float(action[0]) if action.size > 0 else 0.0
        twist.angular.z = float(action[1]) if action.size > 1 else 0.0
        pub.publish(twist)

    def spawn_robot(
        self,
        urdf_path: str,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
    ) -> bool:
        """Spawn a robot model in Gazebo via ``/spawn_entity``."""
        if not self._connected:
            raise RuntimeError("Call connect() first.")
        try:
            from gazebo_msgs.srv import SpawnEntity

            client = self._node.create_client(SpawnEntity, "/spawn_entity")
            if not client.wait_for_service(timeout_sec=5.0):
                logger.error("GazeboBridge: /spawn_entity service not available.")
                return False

            request = SpawnEntity.Request()
            request.name = self._robot_model
            with open(urdf_path) as f:
                request.xml = f.read()
            request.initial_pose.position.x = x
            request.initial_pose.position.y = y

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
            result = future.result()
            if result and result.success:
                logger.info("GazeboBridge: robot spawned at (%.2f, %.2f).", x, y)
                return True
            else:
                logger.error("GazeboBridge: spawn failed -- %s", getattr(result, "status_message", "unknown"))
                return False
        except Exception as exc:
            logger.error("GazeboBridge spawn_robot error: %s", exc)
            return False

    def get_model_state(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Query Gazebo for a model's pose."""
        model_name = model_name or self._robot_model
        try:
            from gazebo_msgs.srv import GetEntityState

            client = self._node.create_client(GetEntityState, "/get_entity_state")
            if not client.wait_for_service(timeout_sec=3.0):
                return {}
            req = GetEntityState.Request()
            req.name = model_name
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=3.0)
            result = future.result()
            if result and result.success:
                p = result.state.pose.position
                return {"x": p.x, "y": p.y, "z": p.z}
        except Exception as exc:
            logger.error("get_model_state error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Isaac Sim bridge (stub)
# ---------------------------------------------------------------------------

class IsaacBridge(_SimBridgeBase):
    """Interface stub for NVIDIA Isaac Sim.

    Isaac Sim exposes its own ROS2 bridge; this class provides the
    NavIRL-standard API on top of it.  Full implementation depends on
    the ``omni.isaac.ros2_bridge`` extension being loaded in the USD stage.
    """

    def __init__(
        self,
        node_name: str = "navirl_isaac_bridge",
        scene_path: str = "",
    ) -> None:
        _ensure_ros2("IsaacBridge")
        super().__init__(node_name)
        self._scene_path = scene_path

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(self._node_name)
        logger.info(
            "IsaacBridge: node created.  Ensure Isaac Sim is running with "
            "the ROS2 bridge extension enabled."
        )
        self._connected = True

    def reset(self) -> Dict[str, Any]:
        logger.info("IsaacBridge: reset requested (stub -- implement via Isaac API).")
        return self.get_observation()

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self.send_action(action)
        if self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.02)
        obs = self.get_observation()
        return {"obs": obs, "reward": 0.0, "done": False, "info": {}}

    def get_observation(self) -> Dict[str, Any]:
        if self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.02)
        return {}

    def send_action(self, action: np.ndarray) -> None:
        if self._node is None:
            raise RuntimeError("Call connect() first.")
        from geometry_msgs.msg import Twist

        pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        twist = Twist()
        action = np.asarray(action, dtype=np.float64).ravel()
        twist.linear.x = float(action[0]) if action.size > 0 else 0.0
        twist.angular.z = float(action[1]) if action.size > 1 else 0.0
        pub.publish(twist)


# ---------------------------------------------------------------------------
# Habitat bridge (stub)
# ---------------------------------------------------------------------------

class HabitatBridge(_SimBridgeBase):
    """Interface stub for Meta Habitat simulator.

    Habitat does not natively use ROS2, so this bridge wraps
    ``habitat-sim`` with a ROS2 node that publishes sensor data
    and subscribes to velocity commands.  Full implementation requires
    ``habitat-sim`` to be installed.
    """

    def __init__(
        self,
        node_name: str = "navirl_habitat_bridge",
        scene_path: str = "",
        sensor_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ensure_ros2("HabitatBridge")
        super().__init__(node_name)
        self._scene_path = scene_path
        self._sensor_config = sensor_config or {}
        self._habitat_sim: Any = None

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(self._node_name)

        try:
            import habitat_sim  # type: ignore[import-untyped]
            logger.info("HabitatBridge: habitat-sim detected.")
        except ImportError:
            logger.warning(
                "HabitatBridge: habitat-sim not found.  "
                "Install it to use Habitat environments."
            )
        self._connected = True

    def reset(self) -> Dict[str, Any]:
        logger.info("HabitatBridge: reset requested (stub -- implement via habitat-sim).")
        return self.get_observation()

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self.send_action(action)
        obs = self.get_observation()
        return {"obs": obs, "reward": 0.0, "done": False, "info": {}}

    def get_observation(self) -> Dict[str, Any]:
        return {}

    def send_action(self, action: np.ndarray) -> None:
        logger.debug("HabitatBridge: send_action stub (action=%s).", action)
