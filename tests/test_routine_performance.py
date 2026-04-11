"""Performance tests for routine system."""

from __future__ import annotations

from unittest import TestCase

from navirl.models.behavior_tree import Blackboard
from navirl.routines.compiler import GoToTarget


class TestRoutinePerformanceFixture(TestCase):
    """Test performance optimizations in the routine system."""

    def test_goto_target_caches_goto_goal(self) -> None:
        """Test that GoToTarget reuses the same GoToGoal instance."""
        # Create a GoToTarget node
        goto_target = GoToTarget(5.0, 3.0)

        # Verify that the GoToGoal instance is cached and reused
        goto_goal_instance1 = goto_target._go_to_goal
        goto_goal_instance2 = goto_target._go_to_goal

        # Should be the same instance (not created fresh each time)
        self.assertIs(goto_goal_instance1, goto_goal_instance2)

        # Verify it's actually a GoToGoal instance
        from navirl.models.behavior_tree import GoToGoal

        self.assertIsInstance(goto_goal_instance1, GoToGoal)

    def test_goto_target_reset_state(self) -> None:
        """Test that GoToTarget properly resets its cached state."""
        # Create a GoToTarget node
        goto_target = GoToTarget(5.0, 3.0)

        # Verify initial state
        initial_goto_goal = goto_target._go_to_goal

        # Reset the state
        goto_target.reset_state()

        # The cached GoToGoal should still be the same instance
        # (our implementation calls reset on the instance rather than creating a new one)
        self.assertIs(goto_target._go_to_goal, initial_goto_goal)
