"""Extended component registry for NavIRL.

Provides a generic, type-safe registry that maps string names to classes
(or factory callables), with optional metadata and config-based instantiation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic registry
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """Internal bookkeeping for a registered component."""

    cls: type | Callable[..., Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """Generic registry for any component type.

    Usage::

        agents = ComponentRegistry("agents")
        agents.register("ppo", PPOAgent, metadata={"family": "on-policy"})
        cls = agents.get("ppo")
        instance = agents.from_config({"name": "ppo", "lr": 1e-4})

    Parameters
    ----------
    name : str
        Human-readable name of the registry (e.g., ``"agents"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, _Entry] = {}

    # -- mutation -----------------------------------------------------------

    def register(
        self,
        name: str,
        cls: type | Callable[..., Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component under *name*.

        Parameters
        ----------
        name : str
            Lookup key.
        cls : type | callable
            The class or factory callable.
        metadata : dict, optional
            Arbitrary metadata (version, description, etc.).

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in self._entries:
            raise ValueError(f"'{name}' is already registered in the '{self.name}' registry.")
        self._entries[name] = _Entry(cls=cls, metadata=metadata or {})

    # -- queries ------------------------------------------------------------

    def get(self, name: str) -> type | Callable[..., Any]:
        """Return the class / factory registered under *name*.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"'{name}' not found in '{self.name}' registry. " f"Available: {available}"
            )
        return self._entries[name].cls

    def list_registered(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(name, metadata)`` pairs for all registered components."""
        return [(n, e.metadata) for n, e in self._entries.items()]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- instantiation from config ------------------------------------------

    def from_config(self, config: dict[str, Any]) -> Any:
        """Instantiate a component from a configuration dict.

        The dict must contain a ``"name"`` key that maps to a registered
        component.  All remaining keys are forwarded as keyword arguments.

        Parameters
        ----------
        config : dict
            Must include ``"name"``; other keys become constructor kwargs.

        Returns
        -------
        Any
            An instance of the registered component.
        """
        config = dict(config)  # shallow copy
        name = config.pop("name")
        cls = self.get(name)
        return cls(**config)


# ---------------------------------------------------------------------------
# Pre-built domain registries
# ---------------------------------------------------------------------------

agents_registry = ComponentRegistry("agents")
environments_registry = ComponentRegistry("environments")
models_registry = ComponentRegistry("models")
sensors_registry = ComponentRegistry("sensors")
rewards_registry = ComponentRegistry("rewards")
scenarios_registry = ComponentRegistry("scenarios")
