import argparse
import os
import pathlib
import subprocess
import sys

import tern.scaffold as tern_scaffold


def main() -> None:
    parser = get_cli_args()
    args = parser.parse_args()

    if args.command == "up":
        cmd_up(args)
    elif args.command == "down":
        cmd_down(args)
    elif args.command == "_repl":
        cmd_repl(args)


def get_cli_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tern",
        description="Provider-agnostic multi-agent coding assistant",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{up,down}"
    )

    subparsers.add_parser("up", help="Start a tern session, initializing if needed")
    subparsers.add_parser("down", help="Remove initialized tern sandbox")
    subparsers.add_parser("_repl", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()  # hide _repl from help table

    return parser


def cmd_up(args: argparse.Namespace) -> None:
    cwd = pathlib.Path.cwd()
    tern_dir = cwd / ".tern"
    sandbox = f"shell-{cwd.name}"

    if tern_dir.exists():
        print(f".tern/ already exists at {tern_dir} — skipping scaffold.")
    else:
        tern_scaffold.scaffold_and_validate(tern_dir)

    result = _sbx(
        ["sbx", "create", "shell", ".", "--kit", "./.tern/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "already exists" not in (result.stderr or ""):
            print(
                result.stderr.strip()
                or f"error: sandbox creation failed (exit {result.returncode})"
            )
            sys.exit(result.returncode)
    else:
        print("tern sandbox ready.")

    # ${WORKDIR} is only substituted in spec.yaml initFiles, not commands — sync explicitly
    result = _sbx(
        [
            "sbx",
            "exec",
            "-w",
            str(cwd),
            sandbox,
            "env",
            "UV_PROJECT_ENVIRONMENT=/home/agent/.venv",
            "uv",
            "sync",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            result.stderr.strip()
            or f"error: venv setup failed (exit {result.returncode})"
        )
        sys.exit(result.returncode)

    result = _sbx(
        ["sbx", "exec", "-w", str(cwd), sandbox, "/home/agent/.venv/bin/tern", "_repl"]
    )
    sys.exit(result.returncode)


def cmd_down(args: argparse.Namespace) -> None:
    sandbox = f"shell-{pathlib.Path.cwd().name}"
    result = _sbx(["sbx", "rm", sandbox])
    sys.exit(result.returncode)


def cmd_repl(args: argparse.Namespace) -> None:
    if not os.environ.get("SANDBOX_VM_ID"):
        print("error: tern _repl must be run inside a Docker Sandbox")
        sys.exit(1)
    print("tern ready.")


def _sbx(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, **kwargs)
    except FileNotFoundError:
        print(
            "error: sbx not found — install from https://docs.docker.com/ai/sandboxes/"
        )
        sys.exit(1)
