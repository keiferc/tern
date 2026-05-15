# Tern

Not an intern or an extern, just a tern: a provider-agnostic, flexible, multi-agent coding assistance system. Also a seabird. Primarily used a an exercise in building AI agents.

## Features

- Security
    - Agent runs in a microVM isolated from host (i.e., Docker Sandbox)
    - Agent's filesystem access is limited to the project directory in which it is installed
    - Agent's internet access is limited to a customizable allowlist
    - microVM proxy supports credential injection
    - Agent has deterministic human-in-the-loop (HITL) controls for dependency use approval
- Customizability
    - Subagent models and profiles are customizable through Docker Sandbox Kits

## Requirements

- [Docker Sandbox](https://www.docker.com/products/docker-sandboxes/) >= 0.28

## Download and Installation

TODO

## Usage

```bash
sbx login
uv run tern
```
TODO

## Agent Architecture

```mermaid
---
config:
  layout: dagre
---
flowchart TB
    s{{"START"}} --> user(("user"))
    user --> have_prompt{"have prompt?"} & approve_plan{"approve plan?"} & approve_deps{"approve dependencies?"}
    have_prompt -- no --> need_handoff{"need handoff?"}
    need_handoff -- yes --> summarizer["summarizer subagent"]
    need_handoff -- no --> e{{"END"}}
    summarizer --> e
    have_prompt -- yes --> planner["planner subagent"]
    planner --> user
    approve_plan -- no --> have_prompt
    approve_plan -- yes --> maker["maker subagent"]
    maker --> dep_check["dependency checker"]
    dep_check --> new_deps{"new dependencies?"}
    new_deps -- no --> qa_tools["QA tools runner"]
    new_deps -- yes --> user
    approve_deps -- no --> maker
    approve_deps -- yes --> qa_tools
    qa_tools --> checker["checker subagent"]
    checker --> have_issues{"have issues?"}
    have_issues -- yes --> maker
    have_issues -- no --> user
```

## Contributing

### Additional Requirements
- [uv](https://docs.astral.sh/uv/) >= 0.11

### Installation
```bash
$ uv python install 3.14 # if necessary
$ uv sync
$ uv run pre-commit autoupdate
$ uv run pre-commit install
```

### Guidelines
- Write self-documenting code
- Manage dependencies with `uv` (e.g., `uv add polars`)
- Verify types with `ty` (e.g., `uv run ty check`)
- Use `pytest` for tests (e.g., `uv run pytest`)
- Ensure style compliance with `ruff` (e.g., `uv run ruff check --fix`, `uv run ruff format`)
- Containerize releases with Docker
- Submit pull requests to `dev`
