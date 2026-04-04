"""Trajectory prediction algorithms for human and robot agents.

This package provides predictors for anticipating future agent paths including:
- Constant velocity and linear motion models
- Kalman filter-based prediction
- Goal-conditioned trajectory prediction
- Neural network models (Social LSTM, Social GAN, Trajectron)
"""

from __future__ import annotations

from navirl.prediction.base import PredictionResult, TrajectoryPredictor
from navirl.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    KalmanPredictor,
    LinearPredictor,
)
from navirl.prediction.goal_conditioned import GoalConditionedPredictor, IntentPredictor

__all__ = [
    "PredictionResult",
    "TrajectoryPredictor",
    "ConstantVelocityPredictor",
    "LinearPredictor",
    "KalmanPredictor",
    "GoalConditionedPredictor",
    "IntentPredictor",
]

# Neural network predictors are available only when torch is installed.
try:
    from navirl.prediction.social_gan import SocialGAN, SocialGANPredictor  # noqa: F401
    from navirl.prediction.social_lstm import SocialLSTM, SocialLSTMPredictor  # noqa: F401
    from navirl.prediction.trajectron import Trajectron, TrajectronPredictor  # noqa: F401

    __all__ += [
        "SocialLSTM",
        "SocialLSTMPredictor",
        "SocialGAN",
        "SocialGANPredictor",
        "Trajectron",
        "TrajectronPredictor",
    ]
except ImportError:
    pass
