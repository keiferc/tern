import argparse
import pathlib
import sys

import tern.scaffold as tern_scaffold


def main() -> None:
    parser = get_cli_args()
    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "up":
        cmd_up(args)


def get_cli_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tern",
        description="Provider-agnostic multi-agent coding assistant",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialise tern in the current project")
    subparsers.add_parser("up", help="Start a tern session (not yet implemented)")

    return parser


def cmd_init(args: argparse.Namespace) -> None:
    tern_dir = pathlib.Path.cwd() / ".tern"

    if tern_dir.exists():
        print(
            f"warning: .tern/ already exists at {tern_dir} — skipping init. "
            "Remove .tern/ and re-run to reinitialise."
        )
        return

    tern_scaffold.scaffold_and_validate(tern_dir)
    print(f"Initialised tern in {tern_dir}")


def cmd_up(args: argparse.Namespace) -> None:
    print("tern up: not yet implemented")
    sys.exit(1)
