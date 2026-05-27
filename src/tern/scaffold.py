import importlib.resources
import pathlib
import shutil
import subprocess

import tern.templates as tern_templates

_TEMPLATE_FILES = (
    "spec.yaml",
    "config.yaml",
    "CONSTITUTION.md",
    "planner.md",
    "maker.md",
    "checker.md",
    "summarizer.md",
)


def scaffold_and_validate(tern_dir: pathlib.Path) -> None:
    scaffold(tern_dir)
    try:
        validate_kit(tern_dir)
    except RuntimeError:
        shutil.rmtree(tern_dir, ignore_errors=True)
        raise


def scaffold(tern_dir: pathlib.Path) -> None:
    tern_dir.mkdir()
    pkg = importlib.resources.files(tern_templates)
    for name in _TEMPLATE_FILES:
        src = pkg.joinpath(name)
        (tern_dir / name).write_bytes(src.read_bytes())


def validate_kit(tern_dir: pathlib.Path) -> None:
    try:
        result = subprocess.run(
            ["sbx", "kit", "validate", str(tern_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "sbx not found — install from https://docs.docker.com/ai/sandboxes/"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("sbx kit validate timed out after 30 s")

    if result.returncode != 0:
        raise RuntimeError(f"sbx kit validate failed:\n{result.stderr.strip()}")
