"""Tests for navirl.core.seeds module."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from navirl.core.seeds import set_global_seed


class TestSetGlobalSeed:
    def test_python_random_deterministic(self):
        set_global_seed(42)
        a = [random.random() for _ in range(10)]
        set_global_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_random_deterministic(self):
        set_global_seed(42)
        a = np.random.rand(10).tolist()
        set_global_seed(42)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_env_var_set(self):
        set_global_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a = random.random()
        set_global_seed(99)
        b = random.random()
        assert a != b

    def test_zero_seed(self):
        set_global_seed(0)
        a = random.random()
        set_global_seed(0)
        b = random.random()
        assert a == b
