"""NavIRLNode -- main ROS2 node bridging NavIRL agents with ROS2 topics.

Subscribes to sensor topics, converts them to NavIRL observations, runs
the agent policy, and publishes velocity commands.

Usage::

    ros2 run navirl navirl_node --ros-args -p agent_type:=social_force \
        -p model_path:=/path/to/model -p action_rate:=10.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import conversions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded ROS2 imports
# ---------------------------------------------------------------------------
try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray

    _ROS2_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = _exc
    # Provide a placeholder so the class definition does not crash when
    # ROS2 is absent -- users will get a clear error at instantiation time.
    Node = object  # type: ignore[assignment,misc]

# Optional: person tracking message (custom or from pedsim)
try:
    from spencer_tracking_msgs.msg import TrackedPersons as PersonArray

    _HAS_PERSON_MSG = True
except ImportError:
    try:
        from pedsim_msgs.msg import TrackedPersons as PersonArray

        _HAS_PERSON_MSG = True
    except ImportError:
        _HAS_PERSON_MSG = False
        PersonArray = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class NavIRLNode(Node):  # type: ignore[misc]
    """Bridge between a NavIRL agent and the ROS2 topic graph.

    Parameters (ROS2 parameters)
    ----------------------------
    agent_type : str
        Name of the NavIRL agent class to instantiate (default ``"irl"``).
    model_path : str
        Filesystem path to a trained model checkpoint (default ``""``).
    action_rate : float
        Frequency (Hz) of the control loop timer (default ``10.0``).
    observation_type : str
        Which observation representation to build -- one of
        ``"lidar"``, ``"image"``, ``"full"`` (default ``"lidar"``).
    """

    # Subscribers
    _sub_scan: Any
    _sub_odom: Any
    _sub_image: Any
    _sub_persons: Any

    # Publishers
    _pub_cmd_vel: Any
    _pub_status: Any
    _pub_debug_markers: Any

    def __init__(self, **kwargs: Any) -> None:
        if not _ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 (rclpy) is not available.  Install ROS2 Humble or later "
                "and source the workspace before using NavIRLNode.  "
                f"Original error: {_ROS2_IMPORT_ERROR}"
            )

        super().__init__("navirl_node", **kwargs)

        # -- Declare parameters ------------------------------------------------
        self.declare_parameter("agent_type", "irl")
        self.declare_parameter("model_path", "")
        self.declare_parameter("action_rate", 10.0)
        self.declare_parameter("observation_type", "lidar")

        self._agent_type: str = self.get_parameter("agent_type").get_parameter_value().string_value
        self._model_path: str = self.get_parameter("model_path").get_parameter_value().string_value
        self._action_rate: float = (
            self.get_parameter("action_rate").get_parameter_value().double_value
        )
        self._obs_type: str = (
            self.get_parameter("observation_type").get_parameter_value().string_value
        )

        self.get_logger().info(
            f"NavIRLNode starting  agent_type={self._agent_type}  "
            f"model_path={self._model_path!r}  action_rate={self._action_rate} Hz  "
            f"observation_type={self._obs_type}"
        )

        # -- Internal state ----------------------------------------------------
        self._latest_scan: np.ndarray | None = None
        self._latest_odom: dict[str, Any] | None = None
        self._latest_image: np.ndarray | None = None
        self._latest_persons: np.ndarray | None = None
        self._agent: Any = None  # loaded lazily
        self._goal: tuple | None = None
        self._step_count: int = 0

        # -- QoS profiles ------------------------------------------------------
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # -- Subscribers -------------------------------------------------------
        self._sub_scan = self.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
        self._sub_odom = self.create_subscription(Odometry, "/odom", self._odom_cb, sensor_qos)
        self._sub_image = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, sensor_qos
        )
        if _HAS_PERSON_MSG:
            self._sub_persons = self.create_subscription(
                PersonArray, "/tracked_persons", self._persons_cb, sensor_qos
            )
        else:
            self._sub_persons = None
            self.get_logger().warn(
                "Person tracking message type not found -- /tracked_persons subscription disabled."
            )

        # -- Publishers --------------------------------------------------------
        self._pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_status = self.create_publisher(String, "/navirl/status", 10)
        self._pub_debug_markers = self.create_publisher(MarkerArray, "/navirl/debug_markers", 10)

        # -- Timer-based control loop ------------------------------------------
        period_s = 1.0 / max(self._action_rate, 0.1)
        self._timer = self.create_timer(period_s, self._action_loop)

        self._load_agent()
        self._publish_status("initialized")

    # ------------------------------------------------------------------
    # Agent loading
    # ------------------------------------------------------------------

    def _load_agent(self) -> None:
        """Instantiate the NavIRL agent from configuration."""
        try:
            # Attempt to load via NavIRL's agent registry
            from navirl.agents import load_agent  # type: ignore[import-untyped]

            self._agent = load_agent(
                agent_type=self._agent_type,
                model_path=self._model_path if self._model_path else None,
            )
            self.get_logger().info(f"Agent loaded: {self._agent_type}")
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load agent '{self._agent_type}': {exc}.  "
                "The node will publish zero-velocity until an agent is available."
            )
            self._agent = None

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _scan_cb(self, msg: Any) -> None:
        self._latest_scan = conversions.laser_scan_to_lidar_obs(msg)

    def _odom_cb(self, msg: Any) -> None:
        self._latest_odom = conversions.odometry_to_state(msg)

    def _image_cb(self, msg: Any) -> None:
        self._latest_image = conversions.image_to_numpy(msg)

    def _persons_cb(self, msg: Any) -> None:
        self._latest_persons = conversions.person_array_to_social_obs(msg)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _action_loop(self) -> None:
        """Called at *action_rate* Hz by the timer."""
        observation = self._build_observation()
        if observation is None:
            return  # not enough sensor data yet

        action = self._query_agent(observation)
        self._publish_action(action)
        self._publish_debug_markers(observation, action)
        self._step_count += 1

        if self._step_count % 100 == 0:
            self._publish_status(f"running step={self._step_count}")

    def _build_observation(self) -> dict[str, Any] | None:
        """Assemble a NavIRL-compatible observation dict from cached data."""
        obs: dict[str, Any] = {}

        if self._obs_type in ("lidar", "full"):
            if self._latest_scan is None:
                return None
            obs["lidar"] = self._latest_scan

        if self._obs_type in ("image", "full"):
            if self._latest_image is None and self._obs_type == "image":
                return None
            if self._latest_image is not None:
                obs["image"] = self._latest_image

        if self._latest_odom is not None:
            obs["robot_state"] = self._latest_odom

        if self._latest_persons is not None:
            obs["pedestrians"] = self._latest_persons

        if self._goal is not None:
            obs["goal"] = np.array(self._goal, dtype=np.float64)

        return obs if obs else None

    def _query_agent(self, observation: dict[str, Any]) -> np.ndarray:
        """Get an action from the loaded agent, or return zeros."""
        if self._agent is None:
            return np.zeros(2, dtype=np.float64)
        try:
            action = self._agent.act(observation)
            return np.asarray(action, dtype=np.float64)
        except Exception as exc:
            self.get_logger().error(f"Agent.act() failed: {exc}", throttle_duration_sec=5.0)
            return np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------

    def _publish_action(self, action: np.ndarray) -> None:
        """Convert a NavIRL action to a Twist and publish on /cmd_vel."""
        twist = Twist()
        if action.size >= 2:
            twist.linear.x = float(action[0])
            twist.angular.z = float(action[1])
        elif action.size == 1:
            twist.linear.x = float(action[0])
        self._pub_cmd_vel.publish(twist)

    def _publish_status(self, status_text: str) -> None:
        msg = String()
        msg.data = status_text
        self._pub_status.publish(msg)

    def _publish_debug_markers(self, observation: dict[str, Any], action: np.ndarray) -> None:
        """Publish visualization markers for debugging in RViz."""
        marker_array = MarkerArray()

        # Marker for the chosen action direction
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "navirl_action"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = float(max(abs(action[0]) if action.size > 0 else 0, 0.05))
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker_array.markers.append(marker)

        self._pub_debug_markers.publish(marker_array)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        """Programmatically set the navigation goal."""
        self._goal = (x, y)
        self.get_logger().info(f"Goal set to ({x:.2f}, {y:.2f})")
        self._publish_status(f"goal_set x={x:.2f} y={y:.2f}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    """``ros2 run`` entry-point."""
    if not _ROS2_AVAILABLE:
        raise ImportError(
            "ROS2 (rclpy) is not installed.  Please install ROS2 and source the workspace."
        )
    rclpy.init(args=args)
    node = NavIRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
