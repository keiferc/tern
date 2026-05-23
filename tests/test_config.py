import argparse
import importlib.resources
import pathlib
import subprocess
import unittest.mock

import pytest
import yaml

import tern.config as config
import tern.main as main
import tern.templates as tern_templates


# ========================================================================= #
#                                                                           #
#                               Fixtures                                    #
#                                                                           #
# ========================================================================= #


@pytest.fixture
def tern_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / ".tern"
    d.mkdir()
    return d


@pytest.fixture
def valid_config_yaml(tern_dir: pathlib.Path) -> pathlib.Path:
    data = {
        "models": {"default": "anthropic:claude-sonnet-4-6", "maker": "openai:gpt-4o"},
        "checker": {"tools": ["uv run ruff check .", "uv run pytest"]},
        "max_iterations": {"default": 20, "planner": 10, "maker_checker_cycles": 3},
    }
    path = tern_dir / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture
def valid_spec_yaml(tern_dir: pathlib.Path) -> pathlib.Path:
    data = {
        "schemaVersion": "1",
        "kind": "mixin",
        "name": "tern",
        "network": {"allowedDomains": ["api.anthropic.com:443"]},
    }
    path = tern_dir / "spec.yaml"
    path.write_text(yaml.dump(data))
    return path


# ========================================================================= #
#                                                                           #
#                           load_config                                     #
#                                                                           #
# ========================================================================= #


def test_load_config_valid(tern_dir: pathlib.Path, valid_config_yaml: pathlib.Path):
    cfg = config.load_config(tern_dir)
    assert cfg.models["default"] == "anthropic:claude-sonnet-4-6"
    assert cfg.models["maker"] == "openai:gpt-4o"
    assert cfg.checker_tools == ["uv run ruff check .", "uv run pytest"]
    assert cfg.max_iterations["default"] == 20
    assert cfg.max_iterations["planner"] == 10


@pytest.mark.parametrize("missing_section", ["models", "checker", "max_iterations"])
def test_load_config_missing_required_section_raises(
    tern_dir: pathlib.Path, missing_section: str
):
    data = {
        "models": {"default": "anthropic:claude-sonnet-4-6"},
        "checker": {"tools": []},
        "max_iterations": {"default": 20},
    }
    del data[missing_section]
    (tern_dir / "config.yaml").write_text(yaml.dump(data))
    with pytest.raises(ValueError, match=f"required section: {missing_section}"):
        config.load_config(tern_dir)


def test_load_config_missing_default_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"maker": "openai:gpt-4o"},
                "checker": {"tools": []},
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="models.default"):
        config.load_config(tern_dir)


def test_load_config_missing_checker_tools_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {},
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="checker.tools"):
        config.load_config(tern_dir)


def test_load_config_missing_maker_checker_cycles_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {"tools": ["uv run pytest"]},
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="max_iterations.maker_checker_cycles"):
        config.load_config(tern_dir)


def test_load_config_missing_max_iterations_default_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {"tools": ["uv run pytest"]},
                "max_iterations": {"maker": 10},
            }
        )
    )
    with pytest.raises(ValueError, match="max_iterations.default"):
        config.load_config(tern_dir)


def test_load_config_null_max_iterations_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {"tools": ["uv run pytest"]},
                "max_iterations": None,
            }
        )
    )
    with pytest.raises(ValueError, match="max_iterations must be a mapping"):
        config.load_config(tern_dir)


def test_load_config_null_models_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        "models:\nchecker:\n  tools: []\nmax_iterations:\n  default: 20\n"
    )
    with pytest.raises(ValueError, match="models must be a mapping"):
        config.load_config(tern_dir)


def test_load_config_null_checker_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": None,
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="checker must be a mapping"):
        config.load_config(tern_dir)


def test_load_config_not_a_mapping_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="mapping"):
        config.load_config(tern_dir)


def test_load_config_checker_tools_scalar_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {"tools": "uv run ruff"},
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="checker.tools must be a list"):
        config.load_config(tern_dir)


def test_load_config_checker_not_a_mapping_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": ["uv run ruff"],
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="checker must be a mapping"):
        config.load_config(tern_dir)


def test_load_config_models_not_a_mapping_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": ["anthropic:claude-sonnet-4-6"],
                "checker": {"tools": []},
                "max_iterations": {"default": 20},
            }
        )
    )
    with pytest.raises(ValueError, match="models must be a mapping"):
        config.load_config(tern_dir)


def test_load_config_max_iterations_not_a_mapping_raises(tern_dir: pathlib.Path):
    (tern_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "models": {"default": "anthropic:claude-sonnet-4-6"},
                "checker": {"tools": ["uv run pytest"]},
                "max_iterations": [20],
            }
        )
    )
    with pytest.raises(ValueError, match="max_iterations must be a mapping"):
        config.load_config(tern_dir)


# ========================================================================= #
#                                                                           #
#                           load_spec                                       #
#                                                                           #
# ========================================================================= #


def test_load_spec_valid(tern_dir: pathlib.Path, valid_spec_yaml: pathlib.Path):
    spec = config.load_spec(tern_dir)
    assert spec.schema_version == "1"
    assert spec.kind == "mixin"
    assert spec.name == "tern"
    assert spec.allowed_domains == ["api.anthropic.com:443"]


def test_load_spec_camelcase_mapping(
    tern_dir: pathlib.Path, valid_spec_yaml: pathlib.Path
):
    spec = config.load_spec(tern_dir)
    assert hasattr(spec, "schema_version")
    assert hasattr(spec, "allowed_domains")


@pytest.mark.parametrize("missing_field", ["schemaVersion", "kind", "name"])
def test_load_spec_missing_required_field_raises(
    tern_dir: pathlib.Path, missing_field: str
):
    data = {"schemaVersion": "1", "kind": "mixin", "name": "tern"}
    del data[missing_field]
    (tern_dir / "spec.yaml").write_text(yaml.dump(data))
    with pytest.raises(ValueError, match=missing_field):
        config.load_spec(tern_dir)


def test_load_spec_allowed_domains_defaults_empty(tern_dir: pathlib.Path):
    data = {"schemaVersion": "1", "kind": "mixin", "name": "tern"}
    (tern_dir / "spec.yaml").write_text(yaml.dump(data))
    spec = config.load_spec(tern_dir)
    assert spec.allowed_domains == []


def test_load_spec_null_network_defaults_empty(tern_dir: pathlib.Path):
    data = {"schemaVersion": "1", "kind": "mixin", "name": "tern", "network": None}
    (tern_dir / "spec.yaml").write_text(yaml.dump(data))
    spec = config.load_spec(tern_dir)
    assert spec.allowed_domains == []


def test_load_spec_not_a_mapping_raises(tern_dir: pathlib.Path):
    (tern_dir / "spec.yaml").write_text("- item1\n")
    with pytest.raises(ValueError, match="mapping"):
        config.load_spec(tern_dir)


def test_load_spec_network_not_a_mapping_raises(tern_dir: pathlib.Path):
    data = {"schemaVersion": "1", "kind": "mixin", "name": "tern", "network": ["item"]}
    (tern_dir / "spec.yaml").write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="network must be a mapping"):
        config.load_spec(tern_dir)


def test_load_spec_allowed_domains_scalar_raises(tern_dir: pathlib.Path):
    data = {
        "schemaVersion": "1",
        "kind": "mixin",
        "name": "tern",
        "network": {"allowedDomains": "api.anthropic.com:443"},
    }
    (tern_dir / "spec.yaml").write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="allowedDomains must be a list"):
        config.load_spec(tern_dir)


# ========================================================================= #
#                                                                           #
#                        load_agent_prompt                                  #
#                                                                           #
# ========================================================================= #


def test_load_agent_prompt_non_empty(tern_dir: pathlib.Path):
    (tern_dir / "planner.md").write_text("You are a senior data scientist.")
    result = config.load_agent_prompt(tern_dir, "planner")
    assert result == "You are a senior data scientist."


def test_load_agent_prompt_empty_returns_none(tern_dir: pathlib.Path):
    (tern_dir / "maker.md").write_text("")
    assert config.load_agent_prompt(tern_dir, "maker") is None


def test_load_agent_prompt_whitespace_only_returns_none(tern_dir: pathlib.Path):
    (tern_dir / "checker.md").write_text("   \n  ")
    assert config.load_agent_prompt(tern_dir, "checker") is None


def test_load_agent_prompt_unknown_agent_raises(tern_dir: pathlib.Path):
    with pytest.raises(ValueError, match="unknown agent"):
        config.load_agent_prompt(tern_dir, "router")


def test_load_agent_prompt_missing_file_raises_with_agent_name(
    tern_dir: pathlib.Path,
):
    with pytest.raises(FileNotFoundError, match="planner"):
        config.load_agent_prompt(tern_dir, "planner")


# ========================================================================= #
#                                                                           #
#                           cmd_init                                        #
#                                                                           #
# ========================================================================= #


@pytest.fixture
def mock_sbx_ok():
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        yield mock_run


def test_cmd_init_creates_all_files(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    tern_dir = tmp_path / ".tern"
    expected = {
        "spec.yaml",
        "config.yaml",
        "CONSTITUTION.md",
        "planner.md",
        "maker.md",
        "checker.md",
        "summarizer.md",
    }
    assert {f.name for f in tern_dir.iterdir()} == expected


def test_cmd_init_default_model_in_config(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    cfg = config.load_config(tmp_path / ".tern")
    assert cfg.models["default"] == "anthropic:claude-sonnet-4-6"


def test_cmd_init_default_checker_tools(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    cfg = config.load_config(tmp_path / ".tern")
    template = yaml.safe_load(
        importlib.resources.files(tern_templates).joinpath("config.yaml").read_bytes()
    )
    assert cfg.checker_tools == template["checker"]["tools"]


def test_cmd_init_default_max_iterations(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    cfg = config.load_config(tmp_path / ".tern")
    template = yaml.safe_load(
        importlib.resources.files(tern_templates).joinpath("config.yaml").read_bytes()
    )
    assert cfg.max_iterations == template["max_iterations"]


def test_cmd_init_valid_spec_yaml(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    spec = config.load_spec(tmp_path / ".tern")
    assert spec.schema_version == "1"
    assert spec.kind == "mixin"
    assert spec.name == "tern"
    assert "api-inference.huggingface.co:443" in spec.allowed_domains


def test_cmd_init_calls_sbx_validate(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    call_args = mock_sbx_ok.call_args[0][0]
    assert call_args[:3] == ["sbx", "kit", "validate"]


def test_cmd_init_sbx_nonzero_raises(tmp_path: pathlib.Path):
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="invalid kit"
        )
        with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="invalid kit"):
                main.cmd_init(argparse.Namespace())


def test_cmd_init_sbx_not_found_raises(tmp_path: pathlib.Path):
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="sbx not found"):
                main.cmd_init(argparse.Namespace())


def test_cmd_init_sbx_timeout_raises(tmp_path: pathlib.Path):
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sbx", timeout=30),
    ):
        with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="timed out"):
                main.cmd_init(argparse.Namespace())


def test_cmd_init_cleans_up_on_validation_failure(tmp_path: pathlib.Path):
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="bad kit"
        )
        with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(RuntimeError):
                main.cmd_init(argparse.Namespace())

    assert not (tmp_path / ".tern").exists()


def test_cmd_init_existing_tern_dir_is_noop(
    tmp_path: pathlib.Path, capsys, mock_sbx_ok
):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    sentinel = tern_dir / "sentinel.txt"
    sentinel.write_text("original")

    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        main.cmd_init(argparse.Namespace())

    assert sentinel.read_text() == "original"
    assert {f.name for f in tern_dir.iterdir()} == {"sentinel.txt"}
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()


def test_cmd_up_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 1


# ========================================================================= #
#                                                                           #
#                           CLI registration                                #
#                                                                           #
# ========================================================================= #


def test_cli_init_subcommand_registered():
    parser = main.get_cli_args()
    args = parser.parse_args(["init"])
    assert args.command == "init"


def test_cli_up_subcommand_registered():
    parser = main.get_cli_args()
    args = parser.parse_args(["up"])
    assert args.command == "up"
