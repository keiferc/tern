import pathlib

import pytest
import yaml

import tern.config as config


# ── fixtures ──────────────────────────────────────────────────────────────


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


# ── load_config ───────────────────────────────────────────────────────────


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


# ── load_spec ─────────────────────────────────────────────────────────────


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


# ── load_agent_prompt ─────────────────────────────────────────────────────


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
