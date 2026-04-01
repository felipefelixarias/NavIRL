"""Pedestrian behavior models with personality traits and decision-making.

Provides configurable behavior archetypes (aggressive, passive, distracted,
hurried, elderly, child, smartphone user, person with disability) that
influence speed adaptation, yielding decisions, path choice, and attention.
All models are numpy-only and expose a uniform interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navirl.humans.pedestrian_state import PedestrianState, PersonalityTag

# ---------------------------------------------------------------------------
# Attention model
# ---------------------------------------------------------------------------

@dataclass
class AttentionModel:
    """Models a pedestrian's spatial awareness and distraction.

    Parameters
    ----------
    awareness_radius : float
        Maximum distance (m) at which the pedestrian notices other agents.
    field_of_view : float
        Angular field of view in radians (symmetric about heading).
    distraction_probability : float
        Per-step probability of entering a distracted state.
    distraction_duration_mean : float
        Mean duration (s) of a distraction episode.
    reaction_time : float
        Delay (s) before responding to stimuli.
    """

    awareness_radius: float = 10.0
    field_of_view: float = math.pi * 2.0 / 3.0  # 120 deg
    distraction_probability: float = 0.0
    distraction_duration_mean: float = 2.0
    reaction_time: float = 0.2

    _distracted: bool = field(default=False, init=False, repr=False)
    _distraction_remaining: float = field(default=0.0, init=False, repr=False)

    # -- public API -----------------------------------------------------------

    @property
    def is_distracted(self) -> bool:
        """Whether the pedestrian is currently distracted."""
        return self._distracted

    def step(self, dt: float, rng: np.random.Generator) -> None:
        """Advance the distraction state by *dt* seconds.

        Parameters
        ----------
        dt : float
            Simulation time step.
        rng : numpy.random.Generator
            Random number generator.
        """
        if self._distracted:
            self._distraction_remaining -= dt
            if self._distraction_remaining <= 0.0:
                self._distracted = False
                self._distraction_remaining = 0.0
        else:
            if rng.random() < self.distraction_probability * dt:
                self._distracted = True
                self._distraction_remaining = float(
                    rng.exponential(self.distraction_duration_mean)
                )

    def can_perceive(
        self,
        observer: PedestrianState,
        target_pos: np.ndarray,
    ) -> bool:
        """Check whether *observer* can perceive an entity at *target_pos*.

        Perception requires the target to be within awareness radius and
        inside the field of view, and the observer must not be distracted.

        Parameters
        ----------
        observer : PedestrianState
            The observing pedestrian.
        target_pos : numpy.ndarray
            2-D position of the target.

        Returns
        -------
        bool
            ``True`` if perceived.
        """
        if self._distracted:
            return False

        diff = target_pos - observer.position
        dist = float(np.linalg.norm(diff))
        if dist > self.awareness_radius:
            return False

        if dist < 1e-9:
            return True

        angle_to = math.atan2(diff[1], diff[0])
        delta = angle_to - observer.heading
        # Wrap to (-pi, pi]
        delta = math.atan2(math.sin(delta), math.cos(delta))
        return abs(delta) <= self.field_of_view / 2.0

    def effective_awareness_radius(self) -> float:
        """Return the radius reduced when distracted.

        Returns
        -------
        float
            Effective awareness radius (m).
        """
        return self.awareness_radius * (0.3 if self._distracted else 1.0)

    def reset(self) -> None:
        """Reset distraction state."""
        self._distracted = False
        self._distraction_remaining = 0.0


# ---------------------------------------------------------------------------
# Personality parameters
# ---------------------------------------------------------------------------

@dataclass
class PersonalityParams:
    """Numeric parameters that characterise a personality archetype.

    Parameters
    ----------
    speed_factor : float
        Multiplier for preferred speed relative to the population mean.
    assertiveness : float
        Willingness to assert right-of-way (0 = always yields, 1 = never).
    personal_space_factor : float
        Multiplier for personal-space radius.
    patience : float
        Tolerance for waiting (higher = more patient).
    compliance : float
        Tendency to follow social conventions (0..1).
    acceleration_factor : float
        Multiplier for acceleration/deceleration capability.
    """

    speed_factor: float = 1.0
    assertiveness: float = 0.5
    personal_space_factor: float = 1.0
    patience: float = 0.5
    compliance: float = 0.7
    acceleration_factor: float = 1.0


#: Pre-defined personality parameter sets.
PERSONALITY_DEFAULTS: dict[PersonalityTag, PersonalityParams] = {
    PersonalityTag.NORMAL: PersonalityParams(
        speed_factor=1.0,
        assertiveness=0.5,
        personal_space_factor=1.0,
        patience=0.5,
        compliance=0.7,
        acceleration_factor=1.0,
    ),
    PersonalityTag.AGGRESSIVE: PersonalityParams(
        speed_factor=1.25,
        assertiveness=0.9,
        personal_space_factor=0.7,
        patience=0.15,
        compliance=0.3,
        acceleration_factor=1.3,
    ),
    PersonalityTag.PASSIVE: PersonalityParams(
        speed_factor=0.85,
        assertiveness=0.1,
        personal_space_factor=1.3,
        patience=0.9,
        compliance=0.9,
        acceleration_factor=0.8,
    ),
    PersonalityTag.DISTRACTED: PersonalityParams(
        speed_factor=0.75,
        assertiveness=0.3,
        personal_space_factor=0.9,
        patience=0.4,
        compliance=0.5,
        acceleration_factor=0.7,
    ),
    PersonalityTag.HURRIED: PersonalityParams(
        speed_factor=1.4,
        assertiveness=0.75,
        personal_space_factor=0.6,
        patience=0.05,
        compliance=0.4,
        acceleration_factor=1.4,
    ),
    PersonalityTag.ELDERLY: PersonalityParams(
        speed_factor=0.6,
        assertiveness=0.2,
        personal_space_factor=1.4,
        patience=0.8,
        compliance=0.9,
        acceleration_factor=0.5,
    ),
    PersonalityTag.CHILD: PersonalityParams(
        speed_factor=0.9,
        assertiveness=0.6,
        personal_space_factor=0.5,
        patience=0.1,
        compliance=0.3,
        acceleration_factor=1.1,
    ),
}


def get_personality_params(tag: PersonalityTag) -> PersonalityParams:
    """Return the :class:`PersonalityParams` for *tag*, falling back to NORMAL.

    Parameters
    ----------
    tag : PersonalityTag
        Personality archetype.

    Returns
    -------
    PersonalityParams
        Parameter set.
    """
    return PERSONALITY_DEFAULTS.get(tag, PERSONALITY_DEFAULTS[PersonalityTag.NORMAL])


# ---------------------------------------------------------------------------
# BehaviorModel
# ---------------------------------------------------------------------------

class BehaviorModel:
    """Configurable pedestrian behavior model.

    Combines personality traits, an attention model, and decision-making
    logic to produce speed adaptation, yielding decisions, and comfort/stress
    updates each simulation step.

    Parameters
    ----------
    personality : PersonalityTag
        Archetype for this pedestrian.
    params : PersonalityParams or None
        Custom parameter overrides; if ``None``, defaults for *personality*
        are used.
    attention : AttentionModel or None
        Custom attention model; if ``None``, defaults are created.
    base_preferred_speed : float
        Population-mean preferred walking speed (m/s).
    base_personal_space : float
        Population-mean personal space radius (m).
    base_max_speed : float
        Population-mean maximum speed (m/s).
    rng_seed : int or None
        Seed for the internal RNG.
    """

    def __init__(
        self,
        personality: PersonalityTag = PersonalityTag.NORMAL,
        params: PersonalityParams | None = None,
        attention: AttentionModel | None = None,
        base_preferred_speed: float = 1.2,
        base_personal_space: float = 0.6,
        base_max_speed: float = 1.5,
        rng_seed: int | None = None,
    ) -> None:
        self.personality: PersonalityTag = personality
        self.params: PersonalityParams = params or get_personality_params(personality)
        self.attention: AttentionModel = attention or self._default_attention()
        self.base_preferred_speed: float = base_preferred_speed
        self.base_personal_space: float = base_personal_space
        self.base_max_speed: float = base_max_speed
        self.rng: np.random.Generator = np.random.default_rng(rng_seed)

        # Derived values.
        self.preferred_speed: float = base_preferred_speed * self.params.speed_factor
        self.max_speed: float = base_max_speed * self.params.speed_factor
        self.personal_space: float = base_personal_space * self.params.personal_space_factor

    # -- factory helpers ------------------------------------------------------

    def _default_attention(self) -> AttentionModel:
        """Build a default :class:`AttentionModel` for the current personality."""
        if self.personality == PersonalityTag.DISTRACTED:
            return AttentionModel(
                awareness_radius=5.0,
                field_of_view=math.pi / 3.0,
                distraction_probability=0.15,
                distraction_duration_mean=3.0,
                reaction_time=0.6,
            )
        if self.personality == PersonalityTag.ELDERLY:
            return AttentionModel(
                awareness_radius=6.0,
                field_of_view=math.pi / 2.0,
                distraction_probability=0.0,
                reaction_time=0.5,
            )
        if self.personality == PersonalityTag.CHILD:
            return AttentionModel(
                awareness_radius=5.0,
                field_of_view=math.pi,
                distraction_probability=0.1,
                distraction_duration_mean=1.5,
                reaction_time=0.3,
            )
        return AttentionModel()

    # -- class methods --------------------------------------------------------

    @classmethod
    def create_smartphone_user(
        cls,
        base_preferred_speed: float = 1.2,
        rng_seed: int | None = None,
    ) -> BehaviorModel:
        """Create a smartphone-using pedestrian behavior model.

        Smartphone users walk slower, have a narrow field of view and a high
        distraction probability.

        Parameters
        ----------
        base_preferred_speed : float
            Population-mean preferred speed.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        BehaviorModel
            Configured for smartphone use.
        """
        params = PersonalityParams(
            speed_factor=0.7,
            assertiveness=0.2,
            personal_space_factor=0.8,
            patience=0.3,
            compliance=0.4,
            acceleration_factor=0.6,
        )
        attention = AttentionModel(
            awareness_radius=3.0,
            field_of_view=math.pi / 4.0,
            distraction_probability=0.25,
            distraction_duration_mean=5.0,
            reaction_time=0.8,
        )
        return cls(
            personality=PersonalityTag.DISTRACTED,
            params=params,
            attention=attention,
            base_preferred_speed=base_preferred_speed,
            rng_seed=rng_seed,
        )

    @classmethod
    def create_elderly(
        cls,
        base_preferred_speed: float = 1.2,
        rng_seed: int | None = None,
    ) -> BehaviorModel:
        """Create an elderly pedestrian behavior model.

        Elderly pedestrians walk slowly, have limited acceleration, large
        personal space, and are very compliant with social conventions.

        Parameters
        ----------
        base_preferred_speed : float
            Population-mean preferred speed.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        BehaviorModel
            Configured for elderly pedestrian.
        """
        return cls(
            personality=PersonalityTag.ELDERLY,
            base_preferred_speed=base_preferred_speed,
            rng_seed=rng_seed,
        )

    @classmethod
    def create_child(
        cls,
        base_preferred_speed: float = 1.2,
        rng_seed: int | None = None,
    ) -> BehaviorModel:
        """Create a child pedestrian behavior model.

        Children are impulsive, have small personal space, low compliance,
        and moderate speed with frequent distraction bursts.

        Parameters
        ----------
        base_preferred_speed : float
            Population-mean preferred speed.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        BehaviorModel
            Configured for a child.
        """
        return cls(
            personality=PersonalityTag.CHILD,
            base_preferred_speed=base_preferred_speed,
            rng_seed=rng_seed,
        )

    @classmethod
    def create_disability(
        cls,
        speed_factor: float = 0.4,
        radius: float = 0.45,
        base_preferred_speed: float = 1.2,
        rng_seed: int | None = None,
    ) -> BehaviorModel:
        """Create a person-with-disability behavior model.

        Configurable speed reduction, larger radius (e.g. wheelchair),
        high compliance, very patient, and large personal space.

        Parameters
        ----------
        speed_factor : float
            Speed multiplier relative to population mean.
        radius : float
            Physical radius (m).
        base_preferred_speed : float
            Population-mean preferred speed.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        BehaviorModel
            Configured for a person with disability.
        """
        params = PersonalityParams(
            speed_factor=speed_factor,
            assertiveness=0.15,
            personal_space_factor=1.8,
            patience=0.95,
            compliance=0.95,
            acceleration_factor=0.35,
        )
        attention = AttentionModel(
            awareness_radius=8.0,
            field_of_view=math.pi,
            distraction_probability=0.0,
            reaction_time=0.4,
        )
        model = cls(
            personality=PersonalityTag.PASSIVE,
            params=params,
            attention=attention,
            base_preferred_speed=base_preferred_speed,
            rng_seed=rng_seed,
        )
        # Store the physical radius hint for the caller.
        model._disability_radius = radius  # noqa: SLF001
        return model

    # -- decision methods -----------------------------------------------------

    def should_yield(
        self,
        ego: PedestrianState,
        other: PedestrianState,
    ) -> bool:
        """Decide whether *ego* should yield to *other*.

        The decision is probabilistic, influenced by assertiveness, relative
        speed, and approach angle.

        Parameters
        ----------
        ego : PedestrianState
            The pedestrian making the decision.
        other : PedestrianState
            The approaching pedestrian.

        Returns
        -------
        bool
            ``True`` if *ego* decides to yield.
        """
        # Base yield probability inversely proportional to assertiveness.
        p_yield = 1.0 - self.params.assertiveness

        # Faster agents are less likely to yield.
        speed_ratio = other.speed / max(ego.speed, 1e-6)
        if speed_ratio > 1.0:
            p_yield *= 1.2
        else:
            p_yield *= 0.8

        # Pedestrians approaching from the right get priority (social convention).
        bearing = ego.bearing_to(other)
        relative_bearing = math.atan2(
            math.sin(bearing - ego.heading), math.cos(bearing - ego.heading)
        )
        if relative_bearing < 0:  # Other is on the right.
            p_yield += 0.15 * self.params.compliance

        p_yield = max(0.0, min(1.0, p_yield))
        return bool(self.rng.random() < p_yield)

    def adapt_speed(
        self,
        ego: PedestrianState,
        nearby_states: list[PedestrianState],
    ) -> float:
        """Compute an adapted speed given nearby pedestrians.

        Slows down when surrounded by slower pedestrians and when space is
        tight; speeds up when path is clear.

        Parameters
        ----------
        ego : PedestrianState
            Current state.
        nearby_states : list[PedestrianState]
            States of perceived neighbours.

        Returns
        -------
        float
            Adapted speed (m/s).
        """
        if not nearby_states:
            return self.preferred_speed

        # Mean speed of neighbours within awareness radius.
        neighbour_speeds: list[float] = []
        min_dist = float("inf")
        for n in nearby_states:
            d = ego.distance_to(n)
            if d < self.attention.effective_awareness_radius():
                neighbour_speeds.append(n.speed)
                min_dist = min(min_dist, d)

        if not neighbour_speeds:
            return self.preferred_speed

        avg_neighbour_speed = float(np.mean(neighbour_speeds))

        # Blend towards neighbour speed weighted by proximity.
        proximity_weight = max(0.0, 1.0 - min_dist / self.attention.awareness_radius)
        blended = (1.0 - proximity_weight * 0.5) * self.preferred_speed + (
            proximity_weight * 0.5
        ) * avg_neighbour_speed

        # Assertive pedestrians less influenced.
        final = self.preferred_speed * self.params.assertiveness + blended * (
            1.0 - self.params.assertiveness
        )

        return max(0.0, min(self.max_speed, final))

    def choose_side(self, ego: PedestrianState, other: PedestrianState) -> int:
        """Decide which side to pass *other* on: +1 (left) or -1 (right).

        Compliant pedestrians prefer passing on the right (right-hand traffic
        convention).

        Parameters
        ----------
        ego : PedestrianState
            The deciding pedestrian.
        other : PedestrianState
            The oncoming pedestrian.

        Returns
        -------
        int
            +1 to pass on the left, -1 to pass on the right.
        """
        # Convention: pass on the right => return -1.
        if self.rng.random() < self.params.compliance:
            return -1
        return 1 if self.rng.random() < 0.5 else -1

    def compute_comfort(
        self,
        ego: PedestrianState,
        nearby_states: list[PedestrianState],
    ) -> float:
        """Compute a comfort score in [0, 1] based on proxemics.

        Comfort drops when neighbours are inside personal space, when the
        pedestrian is distracted, or when speed deviates from preferred.

        Parameters
        ----------
        ego : PedestrianState
            Current state.
        nearby_states : list[PedestrianState]
            Perceived neighbours.

        Returns
        -------
        float
            Comfort in [0, 1].
        """
        comfort = 1.0

        # Personal space intrusions.
        for n in nearby_states:
            d = ego.distance_to(n)
            threshold = self.personal_space * 1.5
            if d < threshold:
                violation_ratio = 1.0 - d / threshold
                comfort -= 0.3 * violation_ratio

        # Speed deviation.
        speed_dev = abs(ego.speed - self.preferred_speed) / max(self.preferred_speed, 1e-6)
        comfort -= 0.15 * min(speed_dev, 1.0)

        # Distraction penalty.
        if self.attention.is_distracted:
            comfort -= 0.1

        return max(0.0, min(1.0, comfort))

    def compute_stress(
        self,
        ego: PedestrianState,
        nearby_states: list[PedestrianState],
        time_pressure: float = 0.0,
    ) -> float:
        """Compute a stress score in [0, 1].

        Parameters
        ----------
        ego : PedestrianState
            Current state.
        nearby_states : list[PedestrianState]
            Perceived neighbours.
        time_pressure : float
            External time-pressure signal in [0, 1].

        Returns
        -------
        float
            Stress in [0, 1].
        """
        stress = 0.0

        # Close neighbours raise stress.
        for n in nearby_states:
            d = ego.distance_to(n)
            if d < ego.radius * 3.0:
                stress += 0.2 * (1.0 - d / (ego.radius * 3.0))

        # Impatient pedestrians feel time pressure more.
        stress += time_pressure * (1.0 - self.params.patience)

        # Being stuck.
        if ego.speed < 0.05 and ego.distance_to_goal() > 1.0:
            stress += 0.15 * (1.0 - self.params.patience)

        return max(0.0, min(1.0, stress))

    def step(
        self,
        ego: PedestrianState,
        nearby_states: list[PedestrianState],
        dt: float,
        time_pressure: float = 0.0,
    ) -> dict[str, Any]:
        """Run one simulation step of the behavior model.

        Updates the attention model and returns recommended speed, comfort,
        and stress values.

        Parameters
        ----------
        ego : PedestrianState
            Current state of the pedestrian.
        nearby_states : list[PedestrianState]
            States of nearby pedestrians.
        dt : float
            Time step (s).
        time_pressure : float
            External time-pressure factor in [0, 1].

        Returns
        -------
        dict
            Keys: ``adapted_speed``, ``comfort``, ``stress``,
            ``is_distracted``.
        """
        self.attention.step(dt, self.rng)

        # Filter neighbours by perception.
        perceived = [
            n
            for n in nearby_states
            if self.attention.can_perceive(ego, n.position)
        ]

        adapted_speed = self.adapt_speed(ego, perceived)
        comfort = self.compute_comfort(ego, perceived)
        stress = self.compute_stress(ego, perceived, time_pressure)

        return {
            "adapted_speed": adapted_speed,
            "comfort": comfort,
            "stress": stress,
            "is_distracted": self.attention.is_distracted,
        }

    def reset(self) -> None:
        """Reset internal state (attention, RNG not re-seeded)."""
        self.attention.reset()


# ---------------------------------------------------------------------------
# Population-level utilities
# ---------------------------------------------------------------------------

def sample_personality(
    rng: np.random.Generator,
    weights: dict[PersonalityTag, float] | None = None,
) -> PersonalityTag:
    """Sample a personality tag from a categorical distribution.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    weights : dict or None
        Mapping from :class:`PersonalityTag` to relative weight.  If
        ``None``, a balanced distribution is used.

    Returns
    -------
    PersonalityTag
        Sampled personality.
    """
    if weights is None:
        weights = {
            PersonalityTag.NORMAL: 0.45,
            PersonalityTag.AGGRESSIVE: 0.1,
            PersonalityTag.PASSIVE: 0.15,
            PersonalityTag.DISTRACTED: 0.1,
            PersonalityTag.HURRIED: 0.1,
            PersonalityTag.ELDERLY: 0.05,
            PersonalityTag.CHILD: 0.05,
        }

    tags = list(weights.keys())
    probs = np.array([weights[t] for t in tags], dtype=np.float64)
    probs /= probs.sum()
    idx = int(rng.choice(len(tags), p=probs))
    return tags[idx]


def create_behavior_model(
    personality: PersonalityTag,
    rng_seed: int | None = None,
    **kwargs: Any,
) -> BehaviorModel:
    """Factory function for creating a :class:`BehaviorModel`.

    Parameters
    ----------
    personality : PersonalityTag
        Archetype.
    rng_seed : int or None
        Optional RNG seed.
    **kwargs
        Forwarded to :class:`BehaviorModel`.

    Returns
    -------
    BehaviorModel
        Configured model.
    """
    return BehaviorModel(personality=personality, rng_seed=rng_seed, **kwargs)
