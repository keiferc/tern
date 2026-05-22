import dataclasses
import pathlib

import yaml

# ========================================================================= #
#                                                                           #
#                               Constants                                   #
#                                                                           #
# ========================================================================= #

VALID_AGENTS: frozenset[str] = frozenset({"planner", "maker", "checker", "summarizer"})

# ========================================================================= #
#                                                                           #
#                               Dataclasses                                 #
#                                                                           #
# ========================================================================= #


@dataclasses.dataclass
class Config:
    models: dict[str, str]
    checker_tools: list[str]
    max_iterations: dict[str, int]


@dataclasses.dataclass
class Spec:
    schema_version: str
    kind: str
    name: str
    allowed_domains: list[str]


# ========================================================================= #
#                                                                           #
#                               Loaders                                     #
#                                                                           #
# ========================================================================= #


def _require_mapping(raw: dict, key: str, label: str) -> dict:
    value = raw.get(key)
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return value or {}


def _require_list(d: dict, key: str, label: str) -> list:
    value = d.get(key)
    if value is None:
        raise ValueError(f"missing required field: {label}")
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list, got {type(value).__name__}")
    return value


def _optional_list(d: dict, key: str, label: str) -> list:
    value = d.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list, got {type(value).__name__}")
    return value


def load_config(tern_dir: pathlib.Path) -> Config:
    path = tern_dir / "config.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"config.yaml must be a YAML mapping, got {type(raw).__name__}"
        )

    models = _require_mapping(raw, "models", "config.yaml models")
    if "default" not in models:
        raise ValueError("config.yaml missing required field: models.default")

    checker = _require_mapping(raw, "checker", "config.yaml checker")
    checker_tools = _require_list(checker, "tools", "config.yaml checker.tools")

    max_iterations = _require_mapping(
        raw, "max_iterations", "config.yaml max_iterations"
    )
    if "default" not in max_iterations:
        raise ValueError("config.yaml missing required field: max_iterations.default")

    return Config(
        models=models, checker_tools=checker_tools, max_iterations=max_iterations
    )


def load_spec(tern_dir: pathlib.Path) -> Spec:
    path = tern_dir / "spec.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"spec.yaml must be a YAML mapping, got {type(raw).__name__}")

    for field in ("schemaVersion", "kind", "name"):
        if field not in raw:
            raise ValueError(f"spec.yaml missing required field: {field}")

    network = _require_mapping(raw, "network", "spec.yaml network")
    allowed_domains = _optional_list(
        network, "allowedDomains", "spec.yaml network.allowedDomains"
    )

    return Spec(
        schema_version=raw["schemaVersion"],
        kind=raw["kind"],
        name=raw["name"],
        allowed_domains=allowed_domains,
    )


def load_agent_prompt(tern_dir: pathlib.Path, agent: str) -> str | None:
    if agent not in VALID_AGENTS:
        raise ValueError(
            f"unknown agent '{agent}'; valid agents: {sorted(VALID_AGENTS)}"
        )

    path = tern_dir / f"{agent}.md"
    try:
        content = path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"agent prompt file not found for '{agent}': {path}")
    return content if content.strip() else None
