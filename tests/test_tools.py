import pathlib

import pytest

import tern.tools as tern_tools


def test_write_file_creates_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tern_tools.write_file.invoke({"path": "output.py", "content": "x = 1"})
    assert (tmp_path / "output.py").read_text(encoding="utf-8") == "x = 1"


def test_write_file_creates_parent_dirs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tern_tools.write_file.invoke({"path": "src/nested/output.py", "content": "y = 2"})
    assert (tmp_path / "src" / "nested" / "output.py").read_text(
        encoding="utf-8"
    ) == "y = 2"


def test_write_file_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        tern_tools.write_file.invoke({"path": "../outside.py", "content": "z = 3"})


def test_write_file_overwrites_silently(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tern_tools.write_file.invoke({"path": "output.py", "content": "x = 1"})
    tern_tools.write_file.invoke({"path": "output.py", "content": "x = 2"})
    assert (tmp_path / "output.py").read_text(encoding="utf-8") == "x = 2"


@pytest.mark.parametrize("filename", [".env", "server.key", "my_secret"])
def test_write_file_raises_for_sensitive_file(
    filename: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="sensitive-file pattern"):
        tern_tools.write_file.invoke({"path": filename, "content": "secret"})
