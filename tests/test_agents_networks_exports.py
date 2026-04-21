"""Tests for the lazy export surface of ``navirl.agents.networks``.

These tests run without PyTorch installed because they only introspect the
export-to-module mapping and the package's own source files. They guard against
drift between the mapping in ``navirl/agents/networks/__init__.py`` and the
public classes/functions defined in each submodule.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from navirl.agents import networks as networks_pkg
from navirl.agents.networks import _EXPORTS

_SUBMODULES = sorted({mod.rsplit(".", 1)[-1] for mod in _EXPORTS.values()})


def _public_defs(module_file: Path) -> set[str]:
    """Return the set of module-level public class and function names."""
    tree = ast.parse(module_file.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and not node.name.startswith("_"):
            names.add(node.name)
    return names


class TestExportsMapping:
    def test_all_matches_exports(self):
        assert set(networks_pkg.__all__) == set(_EXPORTS)

    def test_no_duplicate_exports(self):
        assert len(networks_pkg.__all__) == len(set(networks_pkg.__all__))

    @pytest.mark.parametrize("target_module", _SUBMODULES)
    def test_target_module_resolvable(self, target_module):
        pkg_dir = Path(networks_pkg.__file__).parent
        assert (pkg_dir / f"{target_module}.py").is_file(), (
            f"_EXPORTS references non-existent submodule: {target_module}"
        )

    @pytest.mark.parametrize("target_module", _SUBMODULES)
    def test_submodule_public_defs_are_all_exported(self, target_module):
        """Every public class/function in a submodule must be in _EXPORTS.

        Catches the drift case where someone adds a new public network component
        but forgets to wire it into the lazy export map.
        """
        pkg_dir = Path(networks_pkg.__file__).parent
        defs = _public_defs(pkg_dir / f"{target_module}.py")
        exported_names = {
            name
            for name, mod in _EXPORTS.items()
            if mod == f"navirl.agents.networks.{target_module}"
        }
        missing = defs - exported_names
        assert not missing, (
            f"Public defs in {target_module} not listed in _EXPORTS: {sorted(missing)}"
        )

    def test_every_export_target_name_exists_in_target_module_source(self):
        """Every _EXPORTS key must actually be defined in the target module.

        We check the source (not runtime), so this works without PyTorch.
        """
        pkg_dir = Path(networks_pkg.__file__).parent
        by_module: dict[str, set[str]] = {}
        for name, mod in _EXPORTS.items():
            by_module.setdefault(mod, set()).add(name)
        for mod, names in by_module.items():
            target = mod.rsplit(".", 1)[-1]
            defs = _public_defs(pkg_dir / f"{target}.py")
            missing = names - defs
            assert not missing, (
                f"_EXPORTS lists names not defined in {target}.py: {sorted(missing)}"
            )


class TestLazyAttributeAccess:
    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute 'DefinitelyNotAThing'"):
            networks_pkg.DefinitelyNotAThing  # noqa: B018

    def test_dir_includes_all_exports(self):
        visible = set(dir(networks_pkg))
        assert set(networks_pkg.__all__).issubset(visible)


try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestLazyAttributeResolutionWithTorch:
    """Guard that the lazy __getattr__ actually returns the right object."""

    @pytest.mark.parametrize("export_name", sorted(_EXPORTS))
    def test_each_export_resolves_to_target_module_member(self, export_name):
        fresh = importlib.reload(networks_pkg)
        value = getattr(fresh, export_name)
        target_module = importlib.import_module(_EXPORTS[export_name])
        assert value is getattr(target_module, export_name)
