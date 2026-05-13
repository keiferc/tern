# Tern

Not an intern or an extern, just a tern: a provider-agnostic, flexible, multi-agent coding assistance system. Also a seabird. Primarily used a an exercise in building AI agents.

## Requirements

- [Docker Sandbox](https://www.docker.com/products/docker-sandboxes/) >= 0.28
- [Docker](https://www.docker.com) >= 29.3

## Download and Installation

TODO

## Usage

```bash
sbx login
uv run tern
```
TODO

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
