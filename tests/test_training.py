"""Tests for navirl/training/ module: buffer, curriculum, schedulers, callbacks, trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from navirl.training.buffer import PrioritizedReplayBuffer, ReplayBuffer
from navirl.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    LoggingCallback,
)
from navirl.training.curriculum import (
    CurriculumManager,
    DifficultyDimension,
    LinearCurriculum,
    PerformanceCurriculum,
    StagedCurriculum,
)
from navirl.training.experiment import (
    Experiment,
    ExperimentGrid,
    ExperimentRandom,
    ExperimentStatus,
    ResultsDB,
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
    StepSchedule,
    WarmupSchedule,
)
from navirl.training.trainer import EvalResult, TrainerConfig, TrainingLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=(2,))


@pytest.fixture
def prioritized_buffer():
    return PrioritizedReplayBuffer(
        capacity=64, obs_shape=(4,), action_shape=(2,), alpha=0.6, beta=0.4
    )


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_size(self, replay_buffer):
        assert len(replay_buffer) == 0
        replay_buffer.add(
            obs=np.zeros(4),
            action=np.zeros(2),
            reward=1.0,
            next_obs=np.ones(4),
            done=False,
        )
        assert len(replay_buffer) == 1

    def test_add_multiple(self, replay_buffer):
        for i in range(50):
            replay_buffer.add(
                obs=np.full(4, i, dtype=np.float32),
                action=np.zeros(2),
                reward=float(i),
                next_obs=np.full(4, i + 1, dtype=np.float32),
                done=i % 10 == 9,
            )
        assert len(replay_buffer) == 50

    def test_circular_overwrite(self, replay_buffer):
        for i in range(150):
            replay_buffer.add(
                obs=np.full(4, i, dtype=np.float32),
                action=np.zeros(2),
                reward=0.0,
                next_obs=np.zeros(4),
                done=False,
            )
        assert len(replay_buffer) == 100

    def test_sample_returns_correct_keys(self, replay_buffer):
        for _ in range(10):
            replay_buffer.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), False)
        batch = replay_buffer.sample(5)
        assert set(batch.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}
        assert batch["obs"].shape == (5, 4)
        assert batch["actions"].shape == (5, 2)

    def test_sample_batch_size(self, replay_buffer):
        for _ in range(20):
            replay_buffer.add(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        batch = replay_buffer.sample(8)
        assert batch["rewards"].shape == (8,)


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class TestPrioritizedReplayBuffer:
    def test_add_and_sample(self, prioritized_buffer):
        for i in range(20):
            prioritized_buffer.add(
                obs=np.full(4, i, dtype=np.float32),
                action=np.zeros(2),
                reward=float(i),
                next_obs=np.zeros(4),
                done=False,
            )
        batch = prioritized_buffer.sample(8)
        assert "obs" in batch
        assert "weights" in batch
        assert "indices" in batch
        assert batch["weights"].shape == (8,)

    def test_update_priorities(self, prioritized_buffer):
        for _i in range(10):
            prioritized_buffer.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), False)
        batch = prioritized_buffer.sample(5)
        indices = batch["indices"]
        new_priorities = np.ones(5) * 10.0
        prioritized_buffer.update_priorities(indices, new_priorities)
        # Should not crash; priorities are updated internally

    def test_capacity_limit(self, prioritized_buffer):
        for _i in range(100):
            prioritized_buffer.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), False)
        assert len(prioritized_buffer) == 64


# ---------------------------------------------------------------------------
# Curriculum Learning
# ---------------------------------------------------------------------------


class TestDifficultyDimension:
    def test_default_current_value(self):
        dim = DifficultyDimension(name="density", min_value=0.0, max_value=10.0)
        assert dim.current_value == 0.0

    def test_set_from_difficulty(self):
        dim = DifficultyDimension(name="speed", min_value=0.5, max_value=2.0)
        dim.set_from_difficulty(0.5)
        assert dim.current_value == pytest.approx(1.25)

    def test_set_from_difficulty_clamp(self):
        dim = DifficultyDimension(name="x", min_value=0, max_value=1)
        dim.set_from_difficulty(1.5)
        assert dim.current_value == 1.0
        dim.set_from_difficulty(-0.5)
        assert dim.current_value == 0.0


class TestLinearCurriculum:
    def test_start(self):
        lc = LinearCurriculum(start_difficulty=0.0, end_difficulty=1.0, total_steps=100)
        assert lc.get_difficulty(0) == pytest.approx(0.0)

    def test_end(self):
        lc = LinearCurriculum(start_difficulty=0.0, end_difficulty=1.0, total_steps=100)
        assert lc.get_difficulty(100) == pytest.approx(1.0)

    def test_midpoint(self):
        lc = LinearCurriculum(start_difficulty=0.0, end_difficulty=1.0, total_steps=100)
        assert lc.get_difficulty(50) == pytest.approx(0.5)

    def test_beyond_total_steps(self):
        lc = LinearCurriculum(total_steps=100)
        assert lc.get_difficulty(200) == pytest.approx(1.0)

    def test_update_noop(self):
        lc = LinearCurriculum()
        lc.update(50)  # Should not raise


class TestPerformanceCurriculum:
    def test_initial_difficulty(self):
        pc = PerformanceCurriculum()
        assert pc.get_difficulty(0) == 0.0

    def test_increase_on_high_performance(self):
        pc = PerformanceCurriculum(threshold=0.7, increase_rate=0.1)
        pc.update(1, {"eval/success_rate": 0.8})
        assert pc.get_difficulty(1) == pytest.approx(0.1)

    def test_decrease_on_low_performance(self):
        pc = PerformanceCurriculum(threshold=0.7, increase_rate=0.1, decrease_rate=0.05)
        pc.update(1, {"eval/success_rate": 0.8})
        pc.update(2, {"eval/success_rate": 0.3})
        assert pc.get_difficulty(2) == pytest.approx(0.05)

    def test_no_metrics(self):
        pc = PerformanceCurriculum()
        pc.update(1)
        assert pc.get_difficulty(1) == 0.0

    def test_clamp_to_one(self):
        pc = PerformanceCurriculum(threshold=0.0, increase_rate=0.6)
        for i in range(10):
            pc.update(i, {"eval/success_rate": 1.0})
        assert pc.get_difficulty(10) <= 1.0


class TestStagedCurriculum:
    @pytest.fixture
    def stages(self):
        return [
            {"name": "easy", "difficulty": 0.0, "promotion_threshold": 0.8},
            {"name": "medium", "difficulty": 0.5, "promotion_threshold": 0.7},
            {"name": "hard", "difficulty": 1.0, "promotion_threshold": 1.0},
        ]

    def test_initial_stage(self, stages):
        sc = StagedCurriculum(stages)
        assert sc.current_stage == 0
        assert sc.current_stage_name == "easy"

    def test_promotion(self, stages):
        sc = StagedCurriculum(stages)
        sc.update(1, {"eval/success_rate": 0.9})
        assert sc.current_stage == 1
        assert sc.current_stage_name == "medium"

    def test_no_demotion(self, stages):
        sc = StagedCurriculum(stages)
        sc.update(1, {"eval/success_rate": 0.9})
        sc.update(2, {"eval/success_rate": 0.1})
        assert sc.current_stage == 1  # doesn't go back

    def test_empty_stages_error(self):
        with pytest.raises(ValueError):
            StagedCurriculum([])

    def test_multi_promotion(self, stages):
        sc = StagedCurriculum(stages)
        sc.update(1, {"eval/success_rate": 0.9})
        sc.update(2, {"eval/success_rate": 0.8})
        assert sc.current_stage == 2


class TestCurriculumManager:
    def test_manager_updates_dimensions(self):
        dims = [
            DifficultyDimension("density", 1.0, 20.0),
            DifficultyDimension("speed", 0.5, 2.0),
        ]
        scheduler = LinearCurriculum(0.0, 1.0, total_steps=100)
        mgr = CurriculumManager(dims, scheduler)
        mgr.update(50)
        config = mgr.get_env_config()
        assert config["density"] == pytest.approx(10.5)
        assert config["speed"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    def test_start_value(self):
        s = LinearSchedule(1.0, 0.0, 100)
        assert s.value(0) == pytest.approx(1.0)

    def test_end_value(self):
        s = LinearSchedule(1.0, 0.0, 100)
        assert s.value(100) == pytest.approx(0.0)

    def test_callable(self):
        s = LinearSchedule(1.0, 0.0, 100)
        assert s(50) == pytest.approx(0.5)


class TestCosineAnnealingSchedule:
    def test_start_max(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, total_steps=100)
        assert s.value(0) == pytest.approx(1.0)

    def test_end_min(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, total_steps=100)
        assert s.value(100) == pytest.approx(0.0, abs=1e-9)

    def test_warmup(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, total_steps=100, warmup_steps=20)
        assert s.value(0) == pytest.approx(0.0)
        assert s.value(20) == pytest.approx(1.0)


class TestStepSchedule:
    def test_initial(self):
        s = StepSchedule(initial_value=1.0, factor=0.1, step_size=10)
        assert s.value(0) == pytest.approx(1.0)

    def test_after_step(self):
        s = StepSchedule(initial_value=1.0, factor=0.1, step_size=10)
        assert s.value(10) == pytest.approx(0.1)

    def test_min_value(self):
        s = StepSchedule(initial_value=1.0, factor=0.1, step_size=10, min_value=0.05)
        assert s.value(100) >= 0.05


class TestExponentialSchedule:
    def test_decay(self):
        s = ExponentialSchedule(1.0, decay_rate=0.5, decay_steps=1)
        assert s.value(1) == pytest.approx(0.5)

    def test_min_value(self):
        s = ExponentialSchedule(1.0, decay_rate=0.1, decay_steps=1, min_value=0.5)
        assert s.value(100) >= 0.5


class TestCyclicSchedule:
    def test_triangular_period(self):
        s = CyclicSchedule(base_value=0.0, max_value=1.0, cycle_steps=100, mode="triangular")
        assert s.value(0) == pytest.approx(0.0)
        assert s.value(50) == pytest.approx(0.0, abs=1e-6)  # back to base at midpoint change

    def test_cosine_mode(self):
        s = CyclicSchedule(base_value=0.0, max_value=1.0, cycle_steps=100, mode="cosine")
        val = s.value(25)
        assert 0.0 <= val <= 1.0


class TestExplorationSchedule:
    def test_linear_decay(self):
        s = ExplorationSchedule(1.0, 0.01, total_steps=100, mode="linear")
        assert s.value(0) == pytest.approx(1.0)
        assert s.value(100) == pytest.approx(0.01)

    def test_exponential_decay(self):
        s = ExplorationSchedule(1.0, 0.01, total_steps=100, mode="exponential")
        assert s.value(0) == pytest.approx(1.0)
        assert s.value(100) == pytest.approx(0.01, abs=1e-3)


class TestWarmupSchedule:
    def test_warmup_start(self):
        inner = LinearSchedule(1.0, 0.0, 100)
        s = WarmupSchedule(inner, warmup_steps=10, warmup_start=0.0)
        assert s.value(0) == pytest.approx(0.0)

    def test_warmup_end(self):
        inner = LinearSchedule(1.0, 0.0, 100)
        s = WarmupSchedule(inner, warmup_steps=10, warmup_start=0.0)
        assert s.value(10) == pytest.approx(inner.value(10))


class TestReduceOnPlateau:
    def test_no_reduction_with_improvement(self):
        s = ReduceOnPlateauSchedule(initial_value=1.0, patience=3, factor=0.5)
        for val in [0.1, 0.2, 0.3, 0.4]:
            s.report(val)
        assert s.value(0) == pytest.approx(1.0)

    def test_reduction_on_plateau(self):
        s = ReduceOnPlateauSchedule(initial_value=1.0, patience=2, factor=0.5)
        s.report(0.5)
        s.report(0.4)
        s.report(0.4)
        assert s.value(0) == pytest.approx(0.5)


class TestPolynomialSchedule:
    def test_linear_polynomial(self):
        s = PolynomialSchedule(1.0, 0.0, total_steps=100, power=1.0)
        assert s.value(50) == pytest.approx(0.5)

    def test_quadratic(self):
        s = PolynomialSchedule(1.0, 0.0, total_steps=100, power=2.0)
        assert s.value(50) == pytest.approx(0.25)


class TestOneCycleSchedule:
    def test_warmup_phase(self):
        s = OneCycleSchedule(max_value=1.0, total_steps=100, pct_start=0.3)
        assert s.value(0) < s.value(15) < s.value(30)

    def test_decay_phase(self):
        s = OneCycleSchedule(max_value=1.0, total_steps=100, pct_start=0.3)
        assert s.value(30) > s.value(65) > s.value(100)


class TestCompositeSchedule:
    def test_two_phases(self):
        phase1 = LinearSchedule(1.0, 0.5, 50)
        phase2 = LinearSchedule(0.5, 0.0, 50)
        s = CompositeSchedule([(phase1, 50), (phase2, 50)])
        assert s.value(0) == pytest.approx(1.0)
        assert s.value(50) == pytest.approx(0.5)
        assert s.value(100) == pytest.approx(0.0, abs=1e-6)

    def test_past_all_phases(self):
        s = CompositeSchedule([(LinearSchedule(1.0, 0.0, 10), 10)])
        assert s.value(100) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_callback_default_returns_true(self):
        cb = Callback.__new__(Callback)
        assert cb.on_step({}) is True

    def test_callback_list_chain(self):
        called = []

        class TrackingCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_step(self, locals_):
                called.append(self.name)
                return True

        cb_list = CallbackList([TrackingCallback("a"), TrackingCallback("b")])
        cb_list.on_step({})
        assert called == ["a", "b"]

    def test_callback_list_early_stop(self):

        class StopCallback(Callback):
            def on_step(self, locals_):
                return False

        class ContinueCallback(Callback):
            def on_step(self, locals_):
                return True

        cb_list = CallbackList([ContinueCallback(), StopCallback()])
        assert cb_list.on_step({}) is False

    def test_logging_callback(self, tmp_path):
        log_file = str(tmp_path / "train.jsonl")
        cb = LoggingCallback(log_freq=1, log_file=log_file, verbose=0)
        cb.on_training_start({})
        cb.on_step({"loss": 0.5})
        cb.on_training_end({})
        lines = Path(log_file).read_text().strip().splitlines()
        assert len(lines) >= 1

    def test_checkpoint_callback_creates_dir(self, tmp_path):
        cb = CheckpointCallback(save_freq=1, save_path=str(tmp_path / "ckpt"), verbose=0)
        cb.on_training_start({})
        assert (tmp_path / "ckpt").is_dir()


# ---------------------------------------------------------------------------
# TrainerConfig & TrainingLogger
# ---------------------------------------------------------------------------


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.total_timesteps == 1_000_000
        assert cfg.seed == 42

    def test_to_dict_from_dict(self):
        cfg = TrainerConfig(total_timesteps=5000, seed=0)
        d = cfg.to_dict()
        cfg2 = TrainerConfig.from_dict(d)
        assert cfg2.total_timesteps == 5000
        assert cfg2.seed == 0

    def test_from_dict_ignores_unknown(self):
        cfg = TrainerConfig.from_dict({"total_timesteps": 100, "unknown_key": 42})
        assert cfg.total_timesteps == 100


class TestTrainingLogger:
    def test_log_scalar_and_history(self):
        logger = TrainingLogger()
        logger.log_scalar("loss", 0.5, step=1)
        logger.log_scalar("loss", 0.3, step=2)
        history = logger.get_history("loss")
        assert len(history) == 2
        assert history[0] == (1, 0.5)

    def test_log_dict(self):
        logger = TrainingLogger()
        logger.log_dict({"a": 1.0, "b": 2.0}, step=1)
        assert len(logger.get_history("a")) == 1
        assert len(logger.get_history("b")) == 1

    def test_close(self):
        logger = TrainingLogger()
        logger.close()  # Should not raise


class TestEvalResult:
    def test_to_dict(self):
        er = EvalResult(
            mean_reward=10.0,
            std_reward=2.0,
            mean_length=50.0,
            success_rate=0.8,
        )
        d = er.to_dict()
        assert d["mean_reward"] == 10.0
        assert d["success_rate"] == 0.8


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_lifecycle(self):
        exp = Experiment(name="test_run", config={"lr": 1e-3})
        assert exp.status == ExperimentStatus.PENDING
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING
        exp.complete({"reward": 100.0})
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.results["reward"] == 100.0

    def test_fail(self):
        exp = Experiment(name="fail_run")
        exp.start()
        exp.fail("OOM")
        assert exp.status == ExperimentStatus.FAILED
        assert exp.error == "OOM"

    def test_serialization(self):
        exp = Experiment(name="ser_test", config={"lr": 0.01})
        exp.start()
        exp.complete({"loss": 0.1})
        d = exp.to_dict()
        exp2 = Experiment.from_dict(d)
        assert exp2.name == "ser_test"
        assert exp2.status == ExperimentStatus.COMPLETED


class TestExperimentGrid:
    def test_generate_configs(self):
        grid = ExperimentGrid(param_grid={"lr": [0.01, 0.001], "batch_size": [32, 64]})
        configs = grid.generate_configs()
        assert len(configs) == 4
        assert grid.total_combinations == 4

    def test_generate_experiments(self):
        grid = ExperimentGrid(param_grid={"lr": [0.01], "gamma": [0.99]})
        exps = grid.generate_experiments()
        assert len(exps) == 1
        assert exps[0].status == ExperimentStatus.PENDING


class TestExperimentRandom:
    def test_generate_configs(self):
        rand = ExperimentRandom(
            param_distributions={"lr": [1e-4, 1e-3, 1e-2], "gamma": (0.9, 1.0)},
            seed=42,
        )
        configs = rand.generate_configs(n_trials=5)
        assert len(configs) == 5

    def test_generate_experiments(self):
        rand = ExperimentRandom(
            param_distributions={"lr": [0.01]},
            seed=0,
        )
        exps = rand.generate_experiments(n_trials=3)
        assert len(exps) == 3


class TestResultsDB:
    def test_log_and_query(self, tmp_path):
        db_path = tmp_path / "results.db"
        with ResultsDB(db_path) as db:
            exp = Experiment(name="run1", config={"lr": 0.01})
            exp.start()
            exp.complete({"reward": 50.0})
            row_id = db.log_experiment(exp)
            assert row_id >= 1

            results = db.query()
            assert len(results) == 1
            assert results[0].name == "run1"

    def test_get_best(self, tmp_path):
        db_path = tmp_path / "results.db"
        with ResultsDB(db_path) as db:
            for i, reward in enumerate([10, 50, 30]):
                exp = Experiment(name=f"run{i}", config={})
                exp.start()
                exp.complete({"reward": reward})
                db.log_experiment(exp)

            best = db.get_best("reward", n=1, mode="max")
            assert len(best) == 1
            assert best[0].results["reward"] == 50

    def test_query_filters(self, tmp_path):
        db_path = tmp_path / "results.db"
        with ResultsDB(db_path) as db:
            exp = Experiment(name="special")
            exp.start()
            exp.complete({"reward": 1.0})
            db.log_experiment(exp)

            exp2 = Experiment(name="other")
            db.log_experiment(exp2)

            results = db.query({"name": "special"})
            assert len(results) == 1
