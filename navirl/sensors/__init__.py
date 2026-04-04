"""Sensor simulation package for NavIRL.

Provides simulated sensors (LiDAR, camera, depth, IMU, pedestrian detector,
occupancy grid) with configurable noise models, plus sensor fusion and
Kalman-filter state estimation.
"""

from __future__ import annotations

from navirl.sensors.base import (
    DropoutNoise,
    GaussianNoise,
    NoiseModel,
    QuantizationNoise,
    SaltPepperNoise,
    SensorBase,
)
from navirl.sensors.camera import (
    CameraConfig,
    CameraSensor,
    DepthSensor,
    DepthSensorConfig,
)
from navirl.sensors.fusion import (
    EKFConfig,
    KalmanStateEstimator,
    SensorFusion,
)
from navirl.sensors.imu import (
    IMUConfig,
    IMUSensor,
)
from navirl.sensors.lidar import (
    LidarConfig,
    LidarSensor,
)
from navirl.sensors.occupancy_grid import (
    OccupancyGridConfig,
    OccupancyGridSensor,
)
from navirl.sensors.pedestrian_detector import (
    PedestrianDetector,
    PedestrianDetectorConfig,
    PedestrianTracker,
)

__all__ = [
    # Base
    "SensorBase",
    "NoiseModel",
    "GaussianNoise",
    "SaltPepperNoise",
    "DropoutNoise",
    "QuantizationNoise",
    # LiDAR
    "LidarSensor",
    "LidarConfig",
    # Camera / Depth
    "CameraSensor",
    "CameraConfig",
    "DepthSensor",
    "DepthSensorConfig",
    # IMU
    "IMUSensor",
    "IMUConfig",
    # Pedestrian detection and tracking
    "PedestrianDetector",
    "PedestrianDetectorConfig",
    "PedestrianTracker",
    # Occupancy grid
    "OccupancyGridSensor",
    "OccupancyGridConfig",
    # Fusion
    "SensorFusion",
    "KalmanStateEstimator",
    "EKFConfig",
]
