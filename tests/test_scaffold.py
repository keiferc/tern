import importlib.resources
import pathlib
import subprocess
import unittest.mock

import pytest
import yaml

import tern.config as tern_config
import tern.scaffold as tern_scaffold
import tern.templates as tern_templates


@pytest.fixture
def mock_sbx_ok():
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        yield mock_run


@pytest.fixture
def scaffolded(tmp_path: pathlib.Path) -> pathlib.Path:
    tern_dir = tmp_path / ".tern"
    tern_scaffold.scaffold(tern_dir)
    return tern_dir


# ── scaffold ──────────────────────────────────────────────────────────────────


def test_scaffold_creates_all_template_files(scaffolded: pathlib.Path):
    expected = {
        "spec.yaml",
        "config.yaml",
        "CONSTITUTION.md",
        "planner.md",
        "maker.md",
        "checker.md",
        "summarizer.md",
    }
    assert {f.name for f in scaffolded.iterdir()} == expected


def test_scaffold_config_yaml_matches_template_bytes(scaffolded: pathlib.Path):
    template_bytes = (
        importlib.resources.files(tern_templates).joinpath("config.yaml").read_bytes()
    )
    assert (scaffolded / "config.yaml").read_bytes() == template_bytes


def test_scaffold_config_yaml_is_valid(scaffolded: pathlib.Path):
    cfg = tern_config.load_config(scaffolded)
    template = yaml.safe_load(
        importlib.resources.files(tern_templates).joinpath("config.yaml").read_bytes()
    )
    assert cfg.models == template["models"]
    assert cfg.checker_tools == template["checker"]["tools"]
    assert cfg.max_iterations == template["max_iterations"]


def test_scaffold_spec_yaml_is_valid(scaffolded: pathlib.Path):
    spec = tern_config.load_spec(scaffolded)
    assert spec.schema_version == "1"
    assert spec.kind == "agent"
    assert spec.name == "tern"
    assert "api-inference.huggingface.co:443" in spec.allowed_domains


# ── validate_kit ──────────────────────────────────────────────────────────────


def test_validate_kit_calls_sbx_with_dir(tmp_path: pathlib.Path, mock_sbx_ok):
    tern_scaffold.validate_kit(tmp_path)
    call_args = mock_sbx_ok.call_args[0][0]
    assert call_args[:3] == ["sbx", "kit", "validate"]
    assert str(tmp_path) in call_args


def test_validate_kit_raises_on_nonzero(tmp_path: pathlib.Path):
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="invalid kit"
        )
        with pytest.raises(RuntimeError, match="invalid kit"):
            tern_scaffold.validate_kit(tmp_path)


def test_validate_kit_raises_when_sbx_not_found(tmp_path: pathlib.Path):
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="sbx not found"):
            tern_scaffold.validate_kit(tmp_path)


def test_validate_kit_raises_on_timeout(tmp_path: pathlib.Path):
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sbx", timeout=30),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            tern_scaffold.validate_kit(tmp_path)


# ── scaffold_and_validate ─────────────────────────────────────────────────────


def test_scaffold_and_validate_cleans_up_on_failure(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="bad kit"
        )
        with pytest.raises(RuntimeError):
            tern_scaffold.scaffold_and_validate(tern_dir)
    assert not tern_dir.exists()
