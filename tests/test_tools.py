import pathlib
import unittest.mock

import pytest

import tern.tools as tern_tools


# ── write_file ────────────────────────────────────────────────────────────────


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


# ── read_file ─────────────────────────────────────────────────────────────────


def test_read_file_reads_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.py").write_text("print('hello')")
    assert tern_tools.read_file.invoke({"path": "hello.py"}) == "print('hello')"


def test_read_file_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        tern_tools.read_file.invoke({"path": "../outside.txt"})


@pytest.mark.parametrize(
    "filename",
    [
        ".env",
        ".env.local",
        "server.key",
        "cert.pem",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "aws_credentials",
        "my_secret",
        "access_token",
    ],
)
def test_read_file_raises_for_sensitive_filename(
    filename: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / filename).write_text("secret")
    with pytest.raises(ValueError, match="sensitive-file pattern"):
        tern_tools.read_file.invoke({"path": filename})


def test_read_file_allows_env_example(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("API_KEY=changeme")
    result = tern_tools.read_file.invoke({"path": ".env.example"})
    assert result == "API_KEY=changeme"


def test_read_file_allows_tokenizer_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tokenizer.py").write_text("# tokenizer")
    result = tern_tools.read_file.invoke({"path": "tokenizer.py"})
    assert result == "# tokenizer"


# ── list_files ────────────────────────────────────────────────────────────────


def test_list_files_lists_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "a.py").write_text("")
    (subdir / "b.py").write_text("")
    result = tern_tools.list_files.invoke({"path": "src"})
    assert "src/a.py" in result
    assert "src/b.py" in result


def test_list_files_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        tern_tools.list_files.invoke({"path": "../outside"})


# ── web_fetch ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/file"])
def test_web_fetch_rejects_non_http_scheme(url: str):
    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        result = tern_tools.web_fetch.invoke({"url": url})
    assert result.startswith("Error:")
    mock_urlopen.assert_not_called()


def test_web_fetch_passes_timeout_to_urlopen():
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"hello"
    with unittest.mock.patch(
        "urllib.request.urlopen", return_value=mock_resp
    ) as mock_open:
        tern_tools.web_fetch.invoke({"url": "https://example.com"})
    mock_open.assert_called_once_with("https://example.com", timeout=30)


def test_web_fetch_propagates_urlopen_failure():
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=OSError("connection refused")
    ):
        with pytest.raises(OSError, match="connection refused"):
            tern_tools.web_fetch.invoke({"url": "https://example.com"})


def test_web_fetch_truncates_long_response():
    content = b"x" * 25000
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = content
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = tern_tools.web_fetch.invoke({"url": "https://example.com"})
    assert result[:20000] == "x" * 20000
    assert result.endswith("\n[... truncated]")


def test_web_fetch_does_not_truncate_short_response():
    content = b"hello world"
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = content
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = tern_tools.web_fetch.invoke({"url": "https://example.com"})
    assert result == "hello world"
