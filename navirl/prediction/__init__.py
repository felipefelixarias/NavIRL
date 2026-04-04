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
    "ConstantVelocityPredictor",
    "GoalConditionedPredictor",
    "IntentPredictor",
    "KalmanPredictor",
    "LinearPredictor",
    "PredictionResult",
    "TrajectoryPredictor",
]

# Neural network predictors are available only when torch is installed.
try:
    from navirl.prediction.social_gan import SocialGAN, SocialGANPredictor
    from navirl.prediction.social_lstm import SocialLSTM, SocialLSTMPredictor
    from navirl.prediction.trajectron import Trajectron, TrajectronPredictor

    __all__ += [
        "SocialGAN",
        "SocialGANPredictor",
        "SocialLSTM",
        "SocialLSTMPredictor",
        "Trajectron",
        "TrajectronPredictor",
    ]
except ImportError:
    pass
