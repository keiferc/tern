import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import typing as T
import uuid

import langchain_core.runnables.config as lc_runnables_config
import langgraph.checkpoint.memory as lg_memory
import langgraph.types as lg_types

import tern.agent as tern_agent
import tern.config as tern_config
import tern.scaffold as tern_scaffold
import tern.ui as tern_ui


# ========================================================================= #
#                                                                           #
#                               Constants                                   #
#                                                                           #
# ========================================================================= #

_AUTH_ERROR_MSG = (
    "error: API authentication failed. Set an explicit API key as a Docker Sandbox "
    "secret on the host (n.b., OAuth not supported)."
)

_PROMPTS: dict[str, str] = {
    "new_objective": "objective: ",
    "plan_approval": "approve / feedback: ",
    "dep_approval": "approve / feedback: ",
}


# ========================================================================= #
#                                                                           #
#                                   CLI                                     #
#                                                                           #
# ========================================================================= #


def main() -> None:
    parser = get_cli_args()
    args = parser.parse_args()

    if args.command == "up":
        cmd_up(args)
    elif args.command == "on":
        cmd_on(args)
    elif args.command == "down":
        cmd_down(args)
    elif args.command == "_repl":
        cmd_repl(args)


def get_cli_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tern",
        description="Provider-agnostic multi-agent coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "subcommand flags:\n"
            "  up    --scaffold   initialize .tern/ scaffold only\n"
            "        --sandbox    initialize sandbox only\n"
            "  down  --scaffold   remove .tern/ scaffold only\n"
            "        --sandbox    remove sandbox only\n"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{up,on,down}"
    )

    up_parser = subparsers.add_parser("up", help="Initialize scaffold and/or sandbox")
    up_parser.add_argument(
        "--scaffold", action="store_true", help="Initialize .tern/ scaffold only"
    )
    up_parser.add_argument(
        "--sandbox", action="store_true", help="Initialize sandbox only"
    )

    subparsers.add_parser("on", help="Connect to sandbox and start REPL")

    down_parser = subparsers.add_parser("down", help="Remove scaffold and/or sandbox")
    down_parser.add_argument(
        "--scaffold", action="store_true", help="Remove .tern/ scaffold only"
    )
    down_parser.add_argument(
        "--sandbox", action="store_true", help="Remove sandbox only"
    )

    subparsers.add_parser("_repl", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()  # hide _repl from help table

    return parser


def cmd_up(args: argparse.Namespace) -> None:
    cwd = pathlib.Path.cwd()
    tern_dir = cwd / ".tern"
    sandbox = f"tern-{cwd.name}"

    scaffold_only = getattr(args, "scaffold", False) and not getattr(
        args, "sandbox", False
    )
    sandbox_only = getattr(args, "sandbox", False) and not getattr(
        args, "scaffold", False
    )

    if not sandbox_only:
        _init_scaffold(tern_dir)

    if scaffold_only:
        return

    if sandbox_only and not tern_dir.exists():
        print(
            f"error: scaffold not found at {tern_dir}. Run `tern up --scaffold` first."
        )
        sys.exit(1)

    sys.exit(_init_sandbox(sandbox, tern_dir))


def cmd_on(args: argparse.Namespace) -> None:
    cwd = pathlib.Path.cwd()
    tern_dir = cwd / ".tern"
    sandbox = f"tern-{cwd.name}"

    if not tern_dir.exists():
        print(f"error: scaffold not found at {tern_dir}. Run `tern up` first.")
        sys.exit(1)

    tern_ui.print_stage("Loading scaffold")

    with tern_ui.Spinner(f"Connecting to sandbox '{sandbox}'"):
        exists = _sandbox_exists(sandbox)

    if not exists:
        print(f"error: sandbox '{sandbox}' not found. Run `tern up` first.")
        sys.exit(1)

    result = _sbx(
        [
            "sbx",
            "exec",
            "-it",
            "-w",
            str(cwd),
            sandbox,
            "sh",
            "-lc",
            "echo 'Preparing tern agent...' && uv sync --quiet && /home/agent/.venv/bin/tern _repl",
        ]
    )
    sys.exit(result.returncode)


def cmd_down(args: argparse.Namespace) -> None:
    cwd = pathlib.Path.cwd()
    tern_dir = cwd / ".tern"
    sandbox = f"tern-{cwd.name}"

    scaffold_only = getattr(args, "scaffold", False) and not getattr(
        args, "sandbox", False
    )
    sandbox_only = getattr(args, "sandbox", False) and not getattr(
        args, "scaffold", False
    )

    if not sandbox_only:
        _remove_scaffold(tern_dir)

    if scaffold_only:
        return

    result = _sbx(["sbx", "rm", sandbox])
    sys.exit(result.returncode)


def cmd_repl(args: argparse.Namespace) -> None:
    if not os.environ.get("SANDBOX_VM_ID"):
        print("error: tern _repl must be run inside a Docker Sandbox")
        sys.exit(1)

    sys.stdout.reconfigure(line_buffering=True)  # ty: ignore[unresolved-attribute]

    cwd = pathlib.Path.cwd()
    graph, graph_config = _init_repl_graph(cwd)

    if not _invoke(graph, tern_agent.INITIAL_STATE, graph_config):
        sys.exit(1)

    while True:
        snapshot = graph.get_state(graph_config)
        if not snapshot.next:
            break

        state = dict(snapshot.values)
        checkpoint = _detect_checkpoint(state)
        _print_checkpoint(checkpoint, state)

        user_input = _prompt(checkpoint)
        if user_input is None:
            _handle_exit(graph, graph_config, state)
            break

        if user_input.lower() == "exit":
            _handle_exit(graph, graph_config, state)
            break

        if not user_input:
            continue

        graph.update_state(graph_config, _compute_update(checkpoint, user_input))
        if not _invoke(graph, lg_types.Command(resume=True), graph_config):
            sys.exit(1)


# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _init_scaffold(tern_dir: pathlib.Path) -> None:
    if tern_dir.exists():
        print(f"Scaffold already exists at {tern_dir}.")
    else:
        tern_scaffold.scaffold_and_validate(tern_dir)
        print(f"Initialized scaffold at {tern_dir}.")


def _init_sandbox(sandbox: str, tern_dir: pathlib.Path) -> int:
    if _sandbox_exists(sandbox):
        print(f"Sandbox '{sandbox}' already exists.")
        return 0
    result = _sbx(["sbx", "run", "--kit", str(tern_dir), "tern"])
    if result.returncode == 0:
        print(f"Initialized sandbox '{sandbox}'.")
    return result.returncode


def _remove_scaffold(tern_dir: pathlib.Path) -> None:
    if not tern_dir.exists():
        print(f"Scaffold not found at {tern_dir}.")
        return
    try:
        answer = input(f"Remove {tern_dir}? [y/N]: ").strip().lower()
    except KeyboardInterrupt, EOFError:
        print()
        return
    if answer == "y":
        shutil.rmtree(tern_dir)
        print(f"Removed {tern_dir}.")


def _init_repl_graph(
    cwd: pathlib.Path,
) -> tuple[T.Any, lc_runnables_config.RunnableConfig]:
    tern_dir = cwd / ".tern"
    config = tern_config.load_config(tern_dir)
    graph = tern_agent.build_agent(
        config, tern_dir, checkpointer=lg_memory.MemorySaver()
    )
    graph_config: lc_runnables_config.RunnableConfig = {
        "configurable": {"thread_id": str(uuid.uuid4())}
    }
    return graph, graph_config


def _invoke(
    graph: T.Any,
    payload: T.Any,
    graph_config: lc_runnables_config.RunnableConfig,
) -> bool:
    try:
        graph.invoke(payload, graph_config)
        return True
    except Exception as exc:
        if _is_auth_error(exc):
            print(_AUTH_ERROR_MSG, file=sys.stderr)
            return False
        raise


def _print_checkpoint(checkpoint: str, state: dict) -> None:
    if checkpoint == "plan_approval":
        print(f"\n{state.get('plan', '')}")
    elif checkpoint == "dep_approval":
        print(f"\nNew dependencies: {', '.join(state.get('new_deps', []))}")
    elif checkpoint == "new_objective":
        if state.get("issues") and state.get("plan_approved") is None:
            print("\nCycle limit reached. Unresolved issues:")
            for issue in state.get("issues", []):
                print(f"  {issue}")
        elif state.get("qa_output") is not None and not state.get("issues"):
            print("\nMilestone complete.")
            written = state.get("written_files", [])
            if written:
                print("Files written:")
                for f in written:
                    print(f"  {f}")


def _prompt(checkpoint: str) -> str | None:
    tern_ui.print_separator()
    try:
        return input(tern_ui.format_prompt(_PROMPTS[checkpoint])).strip()
    except KeyboardInterrupt, EOFError:
        print()
        return None


def _detect_checkpoint(state: dict) -> str:
    if state.get("new_deps") and state.get("deps_approved") is None:
        return "dep_approval"
    if state.get("issues") and state.get("plan_approved") is None:
        return "new_objective"
    if state.get("plan") is not None and state.get("plan_approved") is not True:
        return "plan_approval"
    return "new_objective"


def _compute_update(checkpoint: str, user_input: str) -> dict:
    if checkpoint == "plan_approval":
        if user_input.lower() == "approve":
            return {"plan_approved": True, "feedback": []}
        return {"feedback": [user_input]}
    if checkpoint == "dep_approval":
        if user_input.lower() == "approve":
            return {"deps_approved": True, "feedback": []}
        return {
            "deps_approved": None,
            "plan_approved": False,
            "new_deps": [],
            "issues": [],
            "feedback": [user_input],
        }
    return {
        "objective": user_input,
        "qa_output": None,
        "deps_approved": None,
        "plan_approved": None,
        "feedback": [],
        "new_deps": [],
        "maker_checker_cycles": 0,
        "session_objectives": [user_input],
    }


def _handle_exit(
    graph: T.Any,
    graph_config: lc_runnables_config.RunnableConfig,
    state: dict,
) -> None:
    try:
        answer = input("Generate handoff document? [y/N]: ").strip().lower()
    except KeyboardInterrupt, EOFError:
        print()
        return
    if answer == "y" and state.get("objective"):
        graph.update_state(graph_config, {"need_handoff": True})
        _invoke(graph, None, graph_config)


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "authentication" in name
        or "authentication" in msg
        or "api key" in msg
        or "api_key" in msg
        or "unauthorized" in msg
        or "invalid x-api-key" in msg
    )


def _sandbox_exists(sandbox: str) -> bool:
    result = _sbx(["sbx", "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0] == sandbox:
            return True
    return False


def _sbx(argv: list[str], **kwargs: T.Any) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, **kwargs)
    except FileNotFoundError:
        print(
            "error: sbx not found — install from https://docs.docker.com/ai/sandboxes/"
        )
        sys.exit(1)
