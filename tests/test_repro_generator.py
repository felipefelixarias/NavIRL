"""Tests for the reproducibility package generator and templates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navirl.repro.generator import (
    GeneratorConfig,
    discover_run_dirs,
    discover_scenarios,
    generate_canonical_package,
    generate_repro_package,
)
from navirl.repro.templates import get_checklist_template, get_package_readme_template


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_checklist_template_loads(self):
        content = get_checklist_template()
        assert "Reproducibility Checklist" in content
        assert "Environment Pins" in content
        assert "Legal and Data Compliance" in content

    def test_package_readme_template_loads(self):
        content = get_package_readme_template()
        assert "{name}" in content
        assert "{version}" in content
        assert "Replay Instructions" in content

    def test_checklist_template_has_all_sections(self):
        content = get_checklist_template()
        required_sections = [
            "Package Identity",
            "Environment Pins",
            "Scenario Configs",
            "Results and Metrics",
            "Artifact Integrity",
            "Legal and Data Compliance",
            "Replay Validation",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_readme_template_has_placeholders(self):
        content = get_package_readme_template()
        required_placeholders = [
            "{name}",
            "{version}",
            "{created_at}",
            "{description}",
            "{python_version}",
            "{metrics_table}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"


# ---------------------------------------------------------------------------
# discover_run_dirs
# ---------------------------------------------------------------------------


class TestDiscoverRunDirs:
    def test_finds_run_dirs(self, tmp_path: Path):
        run1 = tmp_path / "runs" / "run_001" / "bundle"
        run1.mkdir(parents=True)
        (run1 / "summary.json").write_text("{}")

        run2 = tmp_path / "runs" / "run_002" / "bundle"
        run2.mkdir(parents=True)
        (run2 / "summary.json").write_text("{}")

        dirs = discover_run_dirs(tmp_path / "runs")
        assert len(dirs) == 2

    def test_returns_empty_for_nonexistent(self, tmp_path: Path):
        dirs = discover_run_dirs(tmp_path / "nonexistent")
        assert dirs == []

    def test_returns_empty_for_no_summaries(self, tmp_path: Path):
        d = tmp_path / "runs"
        d.mkdir()
        dirs = discover_run_dirs(d)
        assert dirs == []


# ---------------------------------------------------------------------------
# discover_scenarios
# ---------------------------------------------------------------------------


class TestDiscoverScenarios:
    def test_finds_scenario_yaml(self, tmp_path: Path):
        run_dir = tmp_path / "runs" / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "scenario.yaml").write_text("id: test\n")

        scenarios = discover_scenarios(tmp_path / "runs")
        assert len(scenarios) == 1
        assert scenarios[0].name == "scenario.yaml"

    def test_returns_empty_for_nonexistent(self, tmp_path: Path):
        scenarios = discover_scenarios(tmp_path / "nonexistent")
        assert scenarios == []

    def test_falls_back_to_any_yaml(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        (run_dir / "custom.yaml").write_text("id: custom\n")

        scenarios = discover_scenarios(run_dir)
        assert len(scenarios) == 1
        assert scenarios[0].name == "custom.yaml"


# ---------------------------------------------------------------------------
# GeneratorConfig
# ---------------------------------------------------------------------------


class TestGeneratorConfig:
    def test_defaults(self):
        config = GeneratorConfig(name="test")
        assert config.version == "1.0"
        assert config.include_checklist is True
        assert config.include_readme is True
        assert config.metadata == {}

    def test_custom_values(self):
        config = GeneratorConfig(
            name="study",
            version="2.0",
            description="A study",
            include_checklist=False,
            metadata={"author": "Alice"},
        )
        assert config.name == "study"
        assert config.version == "2.0"
        assert config.include_checklist is False
        assert config.metadata["author"] == "Alice"


# ---------------------------------------------------------------------------
# generate_repro_package
# ---------------------------------------------------------------------------


class TestGenerateReproPackage:
    def _make_run_dir(self, tmp_path: Path) -> Path:
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("id: hallway_pass\n")
        (run1 / "summary.json").write_text('{"success_rate": 1.0}\n')
        return run_dir

    def test_generates_complete_package(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(
            name="test-gen",
            run_dir=run_dir,
            out_dir=out,
            description="Test generation",
        )
        package = generate_repro_package(config)

        assert package.name == "test-gen"
        assert (out / "MANIFEST.json").is_file()
        assert (out / "CHECKLIST.md").is_file()
        assert (out / "README.md").is_file()

    def test_generates_without_checklist(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(
            name="no-checklist",
            run_dir=run_dir,
            out_dir=out,
            include_checklist=False,
        )
        generate_repro_package(config)

        assert (out / "MANIFEST.json").is_file()
        assert not (out / "CHECKLIST.md").exists()

    def test_generates_without_readme(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(
            name="no-readme",
            run_dir=run_dir,
            out_dir=out,
            include_readme=False,
        )
        generate_repro_package(config)

        assert (out / "MANIFEST.json").is_file()
        assert not (out / "README.md").exists()

    def test_readme_contains_package_info(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(
            name="readme-test",
            version="3.0",
            description="A readable study",
            run_dir=run_dir,
            out_dir=out,
        )
        generate_repro_package(config)

        readme = (out / "README.md").read_text()
        assert "readme-test" in readme
        assert "3.0" in readme
        assert "A readable study" in readme

    def test_checklist_file_matches_template(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(name="checklist-test", run_dir=run_dir, out_dir=out)
        generate_repro_package(config)

        checklist = (out / "CHECKLIST.md").read_text()
        template = get_checklist_template()
        assert checklist == template

    def test_generates_with_metadata(self, tmp_path: Path):
        run_dir = self._make_run_dir(tmp_path)
        out = tmp_path / "package"

        config = GeneratorConfig(
            name="meta-gen",
            run_dir=run_dir,
            out_dir=out,
            metadata={"author": "Bob", "lab": "NavLab"},
        )
        package = generate_repro_package(config)

        assert package.metadata["author"] == "Bob"
        assert package.metadata["lab"] == "NavLab"

    def test_generates_from_empty_run_dir(self, tmp_path: Path):
        run_dir = tmp_path / "empty_runs"
        run_dir.mkdir()
        out = tmp_path / "package"

        config = GeneratorConfig(name="empty-gen", run_dir=run_dir, out_dir=out)
        package = generate_repro_package(config)

        assert package.name == "empty-gen"
        assert (out / "MANIFEST.json").is_file()


# ---------------------------------------------------------------------------
# generate_canonical_package
# ---------------------------------------------------------------------------


class TestGenerateCanonicalPackage:
    def test_from_scenario_file(self, tmp_path: Path):
        scenario = tmp_path / "hallway_pass.yaml"
        scenario.write_text("id: hallway_pass\nseed: 7\n")

        out = tmp_path / "package"
        package = generate_canonical_package(scenario, out)

        assert package.name == "hallway_pass"
        assert (out / "MANIFEST.json").is_file()
        assert (out / "README.md").is_file()
        assert (out / "CHECKLIST.md").is_file()
        # Staging dir should be cleaned up
        assert not (out / "_staging").exists()

    def test_custom_name(self, tmp_path: Path):
        scenario = tmp_path / "test.yaml"
        scenario.write_text("id: test\n")

        out = tmp_path / "package"
        package = generate_canonical_package(scenario, out, name="custom-name")

        assert package.name == "custom-name"

    def test_raises_on_missing_scenario(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Scenario not found"):
            generate_canonical_package(tmp_path / "nonexistent.yaml", tmp_path / "out")

    def test_includes_scenario_in_package(self, tmp_path: Path):
        scenario = tmp_path / "my_scenario.yaml"
        scenario.write_text("id: my_scenario\ngrid: {rows: 5}\n")

        out = tmp_path / "package"
        package = generate_canonical_package(scenario, out)

        # The scenario should be in the scenarios subdirectory
        scenario_files = list((out / "scenarios").glob("*.yaml"))
        assert len(scenario_files) > 0

    def test_metadata_includes_source(self, tmp_path: Path):
        scenario = tmp_path / "src.yaml"
        scenario.write_text("id: src\n")

        out = tmp_path / "package"
        package = generate_canonical_package(scenario, out)

        assert "source_scenario" in package.metadata

    def test_with_description_and_version(self, tmp_path: Path):
        scenario = tmp_path / "study.yaml"
        scenario.write_text("id: study\n")

        out = tmp_path / "package"
        package = generate_canonical_package(
            scenario, out, version="2.0", description="Hallway study"
        )

        assert package.version == "2.0"
        assert package.description == "Hallway study"


# ---------------------------------------------------------------------------
# CLI integration for repro generate
# ---------------------------------------------------------------------------


class TestCLIGenerateParser:
    def test_repro_generate_parser(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["repro", "generate", "/path/to/scenario.yaml", "--name", "my-pkg", "--version", "2.0"]
        )
        assert args.scenario == "/path/to/scenario.yaml"
        assert args.name == "my-pkg"
        assert args.version == "2.0"

    def test_repro_generate_defaults(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["repro", "generate", "/path/scenario.yaml"])
        assert args.scenario == "/path/scenario.yaml"
        assert args.name is None
        assert args.version == "1.0"
        assert args.out == "out/repro"


# ---------------------------------------------------------------------------
# Imports from navirl.repro
# ---------------------------------------------------------------------------


class TestReproExports:
    def test_generator_exports_available(self):
        from navirl.repro import (
            GeneratorConfig,
            discover_run_dirs,
            discover_scenarios,
            generate_canonical_package,
            generate_repro_package,
        )

        assert GeneratorConfig is not None
        assert callable(discover_run_dirs)
        assert callable(discover_scenarios)
        assert callable(generate_canonical_package)
        assert callable(generate_repro_package)
