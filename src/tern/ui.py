import os
import shutil
import sys
import threading
import typing as T


def _use_ansi() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def print_stage(name: str) -> None:
    print(flush=True)
    if _use_ansi():
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        bar = "─" * max(0, cols - len(name) - 5)
        sys.stdout.write(f"\033[1m── {name} {bar}\033[0m\n")
        sys.stdout.flush()
    else:
        print(f"── {name}", flush=True)


def print_separator() -> None:
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    if _use_ansi():
        sys.stdout.write(f"\033[2m{'─' * cols}\033[0m\n")
        sys.stdout.flush()
    else:
        print("─" * cols, flush=True)


def format_prompt(text: str) -> str:
    if _use_ansi():
        return f"\001\033[1m\002{text}\001\033[0m\002"
    return text


class Spinner:
    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if _use_ansi():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            print(f"{self._message}...", flush=True)
        return self

    def __exit__(self, *_: T.Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.1):
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r{frame} {self._message}")
            sys.stdout.flush()
            i += 1
