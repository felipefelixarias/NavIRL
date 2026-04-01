"""
NavIRL Training Infrastructure
==============================

Comprehensive training pipeline for pedestrian navigation agents, including
experience replay, parallel environment execution, curriculum learning,
experiment management, and extensible callback systems.

Submodules
----------
buffer
    Experience replay buffers: basic, prioritized, n-step, hindsight,
    sequence, rollout, multi-agent, and demonstration.
parallel
    Vectorized environment wrappers for multiprocessing and async execution.
trainer
    Main training loop orchestration with evaluation, checkpointing,
    and structured logging.
curriculum
    Curriculum learning schedulers for progressive difficulty scaling.
callbacks
    Composable training callbacks for evaluation, logging, checkpointing,
    video recording, gradient monitoring, and more.
schedulers
    Learning rate and hyperparameter schedules: linear, cosine, cyclic,
    warmup, one-cycle, plateau, and composite.
experiment
    Experiment management, grid/random hyperparameter search, and
    SQLite-backed results database.
"""

from __future__ import annotations

from navirl.training.buffer import (
    DemonstrationBuffer,
    HindsightReplayBuffer,
    MultiAgentBuffer,
    NStepBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    SequenceBuffer,
)
from navirl.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    CurriculumCallback,
    EarlyStoppingCallback,
    EvalCallback,
    GradientMonitorCallback,
    HyperparameterSearchCallback,
    LoggingCallback,
    ProgressBarCallback,
    SchedulerCallback,
    TensorBoardCallback,
    VideoRecordCallback,
    WandbCallback,
)
from navirl.training.curriculum import (
    CurriculumManager,
    CurriculumScheduler,
    DifficultyDimension,
    LinearCurriculum,
    PerformanceCurriculum,
    StagedCurriculum,
)
from navirl.training.experiment import (
    Experiment,
    ExperimentGrid,
    ExperimentRandom,
    ResultsDB,
)
from navirl.training.parallel import (
    AsyncVecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecEnvWrapper,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from navirl.training.schedulers import (
    CompositeSchedule,
    CosineAnnealingSchedule,
    CyclicSchedule,
    ExplorationSchedule,
    ExponentialSchedule,
    LinearSchedule,
    OneCycleSchedule,
    PolynomialSchedule,
    ReduceOnPlateauSchedule,
    Schedule,
    StepSchedule,
    WarmupSchedule,
)
from navirl.training.trainer import (
    EvalResult,
    Trainer,
    TrainerConfig,
    TrainingLogger,
)

__all__ = [
    # buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "NStepBuffer",
    "HindsightReplayBuffer",
    "SequenceBuffer",
    "RolloutBuffer",
    "MultiAgentBuffer",
    "DemonstrationBuffer",
    # parallel
    "SubprocVecEnv",
    "DummyVecEnv",
    "VecEnvWrapper",
    "VecNormalize",
    "VecFrameStack",
    "VecMonitor",
    "AsyncVecEnv",
    # trainer
    "Trainer",
    "TrainerConfig",
    "TrainingLogger",
    "EvalResult",
    # curriculum
    "CurriculumScheduler",
    "LinearCurriculum",
    "PerformanceCurriculum",
    "StagedCurriculum",
    "CurriculumManager",
    "DifficultyDimension",
    # callbacks
    "Callback",
    "CallbackList",
    "EvalCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "CurriculumCallback",
    "WandbCallback",
    "TensorBoardCallback",
    "ProgressBarCallback",
    "VideoRecordCallback",
    "GradientMonitorCallback",
    "SchedulerCallback",
    "HyperparameterSearchCallback",
    # schedulers
    "Schedule",
    "LinearSchedule",
    "CosineAnnealingSchedule",
    "StepSchedule",
    "ExponentialSchedule",
    "CyclicSchedule",
    "WarmupSchedule",
    "ReduceOnPlateauSchedule",
    "PolynomialSchedule",
    "OneCycleSchedule",
    "CompositeSchedule",
    "ExplorationSchedule",
    # experiment
    "Experiment",
    "ExperimentGrid",
    "ExperimentRandom",
    "ResultsDB",
]
