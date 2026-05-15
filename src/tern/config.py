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


def load_config(tern_dir: pathlib.Path) -> Config:
    path = tern_dir / "config.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"config.yaml must be a YAML mapping, got {type(raw).__name__}"
        )

    models: dict[str, str] = raw.get("models") or {}
    if "default" not in models:
        raise ValueError("config.yaml missing required field: models.default")

    checker_tools = (raw.get("checker") or {}).get("tools")
    if checker_tools is None:
        raise ValueError("config.yaml missing required field: checker.tools")

    return Config(models=models, checker_tools=checker_tools)


def load_spec(tern_dir: pathlib.Path) -> Spec:
    path = tern_dir / "spec.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"spec.yaml must be a YAML mapping, got {type(raw).__name__}")

    for field in ("schemaVersion", "kind", "name"):
        if field not in raw:
            raise ValueError(f"spec.yaml missing required field: {field}")

    allowed_domains: list[str] = (raw.get("network") or {}).get("allowedDomains", [])

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
    content = path.read_text()
    return content if content.strip() else None
