"""User-defined tools from declarative YAML files.

Drop a file in ``~/.hermes/tools/<tool>.yaml`` and it becomes a first-class
tool the agent can call — no Python plugin required::

    # ~/.hermes/tools/my_search.yaml
    name: my_search
    description: "Search my internal documentation"
    command: 'curl -s "https://internal-docs/search?q=$QUERY"'
    parameters:
      query:
        type: string
        description: "Search query"
        required: true
    timeout: 60          # optional, seconds (capped)

Each file defines exactly one tool. It is registered under the ``custom``
toolset (which auto-appears in the default tool set and can be toggled like any
other toolset).

Security — why this is injection-safe
-------------------------------------
Parameter values supplied by the model are passed to the command as
**environment variables** (both the parameter name and its upper-cased form),
never interpolated into the command string. bash does not re-parse the *result*
of a variable expansion for command substitution, so a value such as
``$(rm -rf ~)`` or ``"; rm -rf ~ ; "`` is treated as a literal string, not
executed. The command template is authored by the user (trusted); only the
argument *values* come from the model.

Parameters may not map to execution-sensitive environment variables (``PATH``,
``LD_PRELOAD``, …) — those are rejected at load time (see ``_RESERVED_ENV``) so
a value can't hijack how the shell resolves commands. Each invocation still
goes through the normal command-approval pipeline (#68187), and output is
captured with a hard size bound while the command runs in its own process
group so a timeout kills its descendants too.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_TOOLSET = "custom"
_EMOJI = "🔧"
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600
_MAX_OUTPUT_CHARS = 100_000
_ALLOWED_PARAM_TYPES = {"string", "number", "integer", "boolean"}
# Tool and parameter names must be valid identifiers so they are safe as both
# function-call names and environment-variable names.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Parameter values are exported as environment variables (see _make_handler).
# These names control how the shell / dynamic loader resolves and runs
# executables, so a model-supplied value landing in one would turn an argument
# into command hijacking or code execution (e.g. a `path` parameter clobbering
# PATH, or `ld_preload` -> LD_PRELOAD). Parameters may not map to them; the
# check is case-insensitive so it covers both the exact and upper-cased
# spelling the handler exports.
_RESERVED_ENV = frozenset({
    "PATH", "IFS", "ENV", "BASH_ENV", "BASHOPTS", "SHELLOPTS", "PS4",
    "PROMPT_COMMAND", "GLOBIGNORE", "CDPATH",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_PROFILE",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
})


def register(ctx) -> None:
    """Discover ``~/.hermes/tools/*.yaml`` and register each as a tool.

    Called once by the plugin loader. Never raises: a malformed file or a
    name collision is logged and skipped so it can't break agent startup.
    """
    for path in _iter_tool_files():
        try:
            spec = _load_spec(path)
        except Exception as exc:
            logger.warning("yaml_tools: skipping %s — %s", path, exc)
            continue
        name, schema, command, timeout = spec
        handler = _make_handler(command, list(schema["parameters"]["properties"]), timeout)
        try:
            ctx.register_tool(
                name=name,
                toolset=_TOOLSET,
                schema=schema,
                handler=handler,
                description=schema.get("description", ""),
                emoji=_EMOJI,
            )
        except Exception as exc:
            # Most likely a name collision with a built-in or another YAML
            # tool. We never override built-ins, so just skip this one.
            logger.warning(
                "yaml_tools: could not register tool %r from %s — %s",
                name, path, exc,
            )
        else:
            logger.debug("yaml_tools: registered custom tool %r from %s", name, path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _tools_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "tools"


def _iter_tool_files():
    d = _tools_dir()
    if not d.is_dir():
        return
    for path in sorted(d.iterdir()):
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            yield path


# ---------------------------------------------------------------------------
# Parsing / schema construction
# ---------------------------------------------------------------------------

def _load_spec(path: Path) -> Tuple[str, dict, str, int]:
    """Parse and validate one tool file.

    Returns ``(name, schema, command, timeout)`` or raises ``ValueError`` with
    a human-readable reason.
    """
    from utils import fast_safe_load

    raw = fast_safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("top-level YAML must be a mapping")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' is required and must be a non-empty string")
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid tool name {name!r}: use letters, digits and underscores, "
            "starting with a letter or underscore"
        )

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("'description' must be a string")

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("'command' is required and must be a non-empty string")

    timeout = _coerce_timeout(raw.get("timeout"))

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("'parameters' must be a mapping of name -> spec")

    properties: dict = {}
    required: list = []
    for pname, pspec in parameters.items():
        pname = str(pname)
        if not _NAME_RE.match(pname):
            raise ValueError(
                f"invalid parameter name {pname!r}: use letters, digits and "
                "underscores, starting with a letter or underscore"
            )
        if pname.upper() in _RESERVED_ENV:
            raise ValueError(
                f"parameter name {pname!r} is reserved: it maps to the "
                f"environment variable {pname.upper()}, which controls how the "
                "shell resolves and runs commands — rename the parameter"
            )
        pspec = pspec or {}
        if not isinstance(pspec, dict):
            raise ValueError(f"parameter {pname!r} spec must be a mapping")
        ptype = pspec.get("type", "string")
        if ptype not in _ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"parameter {pname!r} has unsupported type {ptype!r}; "
                f"allowed: {sorted(_ALLOWED_PARAM_TYPES)}"
            )
        prop: dict = {"type": ptype}
        pdesc = pspec.get("description")
        if pdesc is not None:
            prop["description"] = str(pdesc)
        enum = pspec.get("enum")
        if isinstance(enum, list) and enum:
            prop["enum"] = enum
        properties[pname] = prop
        if pspec.get("required"):
            required.append(pname)

    schema = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return name, schema, command, timeout


def _coerce_timeout(value: Any) -> int:
    if value is None:
        return _DEFAULT_TIMEOUT
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'timeout' must be a whole number of seconds, got {value!r}")
    if seconds <= 0:
        raise ValueError("'timeout' must be a positive number of seconds")
    return min(seconds, _MAX_TIMEOUT)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _make_handler(command: str, param_names: list, timeout: int) -> Callable:
    def handler(args: Optional[dict] = None, **kwargs) -> str:
        from tools.registry import tool_error, tool_result

        args = args or {}
        env = os.environ.copy()
        for pname in param_names:
            value = args.get(pname)
            if value is None:
                continue
            rendered = _stringify(value)
            # Expose the value under both the exact parameter name and its
            # upper-cased form so `$query` and `$QUERY` both work in templates.
            # Never write an execution-sensitive variable (defense in depth —
            # such parameter names are already rejected at load time).
            for candidate in (pname, pname.upper()):
                if candidate.upper() in _RESERVED_ENV:
                    continue
                env[candidate] = rendered

        bash = _find_bash()
        if bash is None:
            return tool_error(
                "bash is required to run YAML tools but was not found on PATH",
                success=False,
            )

        # #68187: custom-tool calls go through the normal approval pipeline.
        # The command template is user-authored, but a dangerous or hardline
        # pattern in it must still be gated exactly like a `terminal` command
        # (hardline block, deny rules, yolo bypass, interactive prompt).
        try:
            from tools.approval import check_dangerous_command
            approval = check_dangerous_command(command, env_type="local")
        except Exception as exc:  # pragma: no cover - the gate must not fail open
            return tool_error(f"Approval check failed: {exc}", success=False)
        if not approval.get("approved", False):
            return tool_error(
                approval.get("message") or "Command blocked by the approval guard.",
                success=False,
            )

        try:
            output, timed_out, returncode = _run_bounded(bash, command, env, timeout)
        except Exception as exc:  # pragma: no cover - defensive
            return tool_error(f"Command failed to start: {exc}", success=False)
        if timed_out:
            return tool_error(f"Command timed out after {timeout}s", success=False)
        if returncode != 0:
            return tool_error(
                f"Command exited with status {returncode}",
                success=False,
                output=output,
            )
        return tool_result(output=output, exit_code=0)

    return handler


def _run_bounded(
    bash: str, command: str, env: dict, timeout: int,
) -> Tuple[str, bool, Optional[int]]:
    """Run ``bash -c command`` with bounded memory and clean timeout kill.

    Returns ``(output, timed_out, returncode)``. Output is captured
    incrementally and stops accumulating at ``_MAX_OUTPUT_CHARS`` (the child
    keeps draining so it never blocks on a full pipe), so a runaway command
    cannot exhaust memory before truncation. The child runs in its own process
    group; on timeout the whole group is killed so orphaned descendants don't
    leak.
    """
    import threading

    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # own process group for killpg
    else:  # pragma: no cover - windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        [bash, "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        **kwargs,
    )

    chunks: list = []
    captured = 0
    truncated = False

    def _drain() -> None:
        nonlocal captured, truncated
        try:
            for piece in iter(lambda: proc.stdout.read(8192), ""):
                room = _MAX_OUTPUT_CHARS - captured
                if room > 0:
                    chunks.append(piece[:room])
                    captured += min(len(piece), room)
                if len(piece) > room:
                    truncated = True  # keep looping to drain (and discard) the rest
        except Exception:  # pragma: no cover - pipe closed under us
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_group(proc)
        proc.wait()
        reader.join(timeout=2)
        return "".join(chunks), True, None
    reader.join(timeout=2)
    output = "".join(chunks)
    if truncated:
        output += "\n… [output truncated]"
    return output, False, proc.returncode


def _terminate_group(proc: "subprocess.Popen") -> None:
    """SIGKILL the child's whole process group (POSIX), else kill the child."""
    try:
        if os.name == "posix":
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - windows
            proc.kill()
    except Exception:  # pragma: no cover - already gone
        pass


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _find_bash() -> Optional[str]:
    import shutil
    return shutil.which("bash") or shutil.which("bash.exe")
