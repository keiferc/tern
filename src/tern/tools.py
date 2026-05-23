import fnmatch
import pathlib
import urllib.parse
import urllib.request

import langchain_core.tools as lc_tools


# ========================================================================= #
#                                                                           #
#                               Constants                                   #
#                                                                           #
# ========================================================================= #

_SENSITIVE_FILE_ALLOWLIST = frozenset({".env.example"})
_SENSITIVE_FILE_PATTERNS = (
    "*.env*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "*credentials*",
    "*secret*",
    "*_token*",
)


# ========================================================================= #
#                                                                           #
#                               Tools                                       #
#                                                                           #
# ========================================================================= #


@lc_tools.tool
def web_fetch(url: str) -> str:
    """Fetch the text content of a URL."""
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        return f"Error: web_fetch only supports http/https, got scheme {scheme!r}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
        if len(text) > 20000:
            return text[:20000] + "\n[... truncated]"
        return text


@lc_tools.tool
def read_file(path: str) -> str:
    """Read a file within the project working directory and return its content."""
    resolved = _safe_resolve(path)
    name = resolved.name
    if name not in _SENSITIVE_FILE_ALLOWLIST and any(
        fnmatch.fnmatch(name, pat) for pat in _SENSITIVE_FILE_PATTERNS
    ):
        raise ValueError(f"read_file: {name!r} matches a sensitive-file pattern")
    return resolved.read_text(encoding="utf-8")


@lc_tools.tool
def list_files(path: str) -> str:
    """List files in a directory within the project working directory."""
    cwd = pathlib.Path.cwd().resolve()
    return "\n".join(
        str(p.relative_to(cwd)) for p in sorted(_safe_resolve(path).iterdir())
    )


@lc_tools.tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file at path, creating parent directories as needed. Returns the absolute path written."""
    resolved = _safe_resolve(path)
    name = resolved.name
    if name not in _SENSITIVE_FILE_ALLOWLIST and any(
        fnmatch.fnmatch(name, pat) for pat in _SENSITIVE_FILE_PATTERNS
    ):
        raise ValueError(f"write_file: {name!r} matches a sensitive-file pattern")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return str(resolved)


# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _safe_resolve(path_str: str) -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    resolved = (cwd / path_str).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(f"path outside working directory: {path_str!r}")
    return resolved
