# Tern

Not an intern or an extern, just a tern: a provider-agnostic, human-in-the-loop, multi-agent coding assistant with deterministic security controls and user-configurable tool permissions, guardrails, subagent model choice, and prompt templates. Also a seabird. Primarily used a an exercise in building AI agents.

## Features

- Security
    - Agent runs in a microVM isolated from host (i.e., Docker Sandbox)
    - Agent's filesystem access is limited to the project directory in which it is installed
    - Agent's internet access is limited to a customizable allowlist
    - API credentials are supplied explicitly through Docker Sandbox secrets with nothing exposed to the sandbox internals.
    - Agent has deterministic human-in-the-loop (HITL) controls for dependency use approval
- Customizability
    - Choose subagent models based on your risk level, task complexity, and cost (e.g., can use open source model for Summarizer subagent, GPT for Maker subagent, and Claude for Checker subagent)
    - Subagent personas and success rubrics are customizable
    - Customize your soft guardrails
- Supported Model Providers
    - Anthropic
    - OpenAI
    - Ollama (Pending)

## Requirements

- [Docker Sandbox (sbx)](https://www.docker.com/products/docker-sandboxes/) 0.28+
- [uv](https://docs.astral.sh/uv/) 0.11+

## Download and Installation

```bash
uv add "tern @ git+https://github.com/keiferc/tern.git"
```

## Usage

### Quickstart

```bash
uv run tern up
echo $ANTHROPIC_API_KEY | sbx secret set tern-<PROJECT_DIR> anthropic
uv run tern on
```

### Full Usage

```bash
usage: tern [-h] {up,on,down} ...

Provider-agnostic multi-agent coding assistant

positional arguments:
  {up,on,down}
    up          Initialize scaffold and/or sandbox
    on          Connect to sandbox and start REPL
    down        Remove scaffold and/or sandbox

options:
  -h, --help    show this help message and exit

subcommand flags:
  up    --scaffold   initialize .tern/ scaffold only
        --sandbox    initialize sandbox only
  down  --scaffold   remove .tern/ scaffold only
        --sandbox    remove sandbox only
```

### Customization

```bash
.tern
├── checker.md # for customizing Checker subagent
├── config.yaml # for specifying models and allowable agent tools
├── CONSTITUTION.md # for defining rules that apply to all agents
├── maker.md # for customizing Maker subagent
├── planner.md # for customizing Planner subagent
├── spec.yaml # Docker Sandbox Mixin Kit for customizing sandbox
└── summarizer.md # for customizing Summarizer subagent
```

### Store API keys (once per project)
Tern uses explicit API keys stored as Docker Sandbox secrets. OAuth-backed sbx
provider credentials are not supported for Tern's custom sandbox kit.
See [`sbx secret` docs](https://docs.docker.com/reference/cli/sbx/secret/) for more info.

```bash
# Assuming you saved the API key as an environmental variable...
echo $ANTRHOPIC_API_KEY | sbx secret set tern-<PROJECT_DIR> anthropic # for Anthropic models
echo $OPENAI_API_KEY | sbx secret set tern-<PROJECT_DIR> openai # for OpenAI models
```

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

## Known Bugs

- If you have both a Claude API key and a Claude subscription and you logged in with the API key and re-logged in with your subscription, sbx would still set the API key variable, so Claude would spend from your API credits instead of from your subscription quota, even after sandbox removal. Fix: since ~/.claude is persistent across sandboxes, rm -rf ~/.claude, then /login with your subscription plan.

## Contributing

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
