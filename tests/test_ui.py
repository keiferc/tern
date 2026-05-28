import unittest.mock

import pytest

import tern.ui as tern_ui


# ── _use_ansi ─────────────────────────────────────────────────────────────────


def test_use_ansi_false_when_no_color_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NO_COLOR", "1")
    with unittest.mock.patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        assert tern_ui._use_ansi() is False


def test_use_ansi_false_when_not_tty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    with unittest.mock.patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = False
        assert tern_ui._use_ansi() is False


def test_use_ansi_true_when_tty_and_no_no_color(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    with unittest.mock.patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        assert tern_ui._use_ansi() is True


# ── print_stage ───────────────────────────────────────────────────────────────


def test_print_stage_contains_name(capsys: pytest.CaptureFixture):
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        tern_ui.print_stage("Planning")
    assert "Planning" in capsys.readouterr().out


def test_print_stage_has_leading_blank_line(capsys: pytest.CaptureFixture):
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        tern_ui.print_stage("Planning")
    out = capsys.readouterr().out
    assert out.startswith("\n")


def test_print_stage_contains_banner_marker(capsys: pytest.CaptureFixture):
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        tern_ui.print_stage("Reviewing")
    assert "──" in capsys.readouterr().out


# ── print_separator ───────────────────────────────────────────────────────────


def test_print_separator_outputs_non_empty_line(capsys: pytest.CaptureFixture):
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        tern_ui.print_separator()
    out = capsys.readouterr().out.strip()
    assert len(out) > 0
    assert all(c == "─" for c in out)


# ── format_prompt ─────────────────────────────────────────────────────────────


def test_format_prompt_plain_when_no_ansi():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        result = tern_ui.format_prompt("objective: ")
    assert result == "objective: "


def test_format_prompt_contains_text_when_ansi():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=True):
        result = tern_ui.format_prompt("objective: ")
    assert result == "\001\033[1m\002objective: \001\033[0m\002"


# ── Spinner ───────────────────────────────────────────────────────────────────


def test_spinner_non_tty_prints_message(capsys: pytest.CaptureFixture):
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        with tern_ui.Spinner("Connecting"):
            pass
    assert "Connecting..." in capsys.readouterr().out


def test_spinner_non_tty_does_not_start_thread():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        spinner = tern_ui.Spinner("Loading")
        with spinner:
            pass
    assert spinner._thread is None


def test_spinner_tty_starts_and_stops_thread():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=True):
        with unittest.mock.patch("sys.stdout"):
            spinner = tern_ui.Spinner("Loading")
            with spinner:
                assert spinner._thread is not None
                assert spinner._thread.is_alive()
    assert not spinner._thread.is_alive()


def test_spinner_stop_event_set_on_exit():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        spinner = tern_ui.Spinner("Loading")
        with spinner:
            pass
    assert spinner._stop.is_set()


def test_spinner_stop_event_set_on_exception():
    with unittest.mock.patch("tern.ui._use_ansi", return_value=False):
        spinner = tern_ui.Spinner("Loading")
        try:
            with spinner:
                raise ValueError("test error")
        except ValueError:
            pass
    assert spinner._stop.is_set()
