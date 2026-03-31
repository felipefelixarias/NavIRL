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
