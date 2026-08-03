"""Logging configuration through environment variables, exercised in subprocesses."""

import os
import re
import sys
import subprocess
import json

import pytest

# ------------------------------------------------------------
# Test 46: Logging: file target + JSON format
# ------------------------------------------------------------
def test_logging_json_stderr_subprocess(tmp_path):
    #import os, sys, json, subprocess

    code = r"""
import numpy as np
import zeusdb_vector_database as zdb
v = zdb.VectorDatabase()
idx = v.create('hnsw', dim=8)
vals = np.random.rand(3, 8).astype('float32').tolist()
idx.add({'vectors': vals})
"""

    env = os.environ.copy()
    env.update({
        "ZEUSDB_LOG_LEVEL": "debug",
        "ZEUSDB_LOG_FORMAT": "json",
        "ZEUSDB_LOG_TARGET": "stderr",  # preferred; some builds may still log to stdout
    })

    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    stderr = proc.stderr.decode("utf-8", errors="ignore")
    stdout = proc.stdout.decode("utf-8", errors="ignore")

    def first_json_line(text: str):
        for ln in text.splitlines():
            if ln.strip().startswith("{"):
                return ln
        return None

    line = first_json_line(stderr) or first_json_line(stdout)
    assert line, f"No JSON log lines found.\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
    entry = json.loads(line)
    assert isinstance(entry, dict)
    assert "level" in entry and "fields" in entry and "timestamp" in entry

# ------------------------------------------------------------
# Test 47: Logging: autolog disabled prevents file init
# ------------------------------------------------------------
def test_logging_disabled_autoinit(tmp_path):
    #import os, sys, subprocess

    log_file = tmp_path / "noinit.log"

    code = r"""
import numpy as np
import zeusdb_vector_database as zdb
# If autologging is truly disabled, creating/using the index shouldn't create a file-based subscriber.
v = zdb.VectorDatabase()
idx = v.create('hnsw', dim=8)
vals = np.random.rand(2, 8).astype('float32').tolist()
idx.add({'vectors': vals})
"""

    env = os.environ.copy()
    env.update({
        "ZEUSDB_DISABLE_AUTOLOG": "1",
        "ZEUSDB_LOG_LEVEL": "debug",          # would normally create logs
        "ZEUSDB_LOG_FORMAT": "json",
        "ZEUSDB_LOG_TARGET": "file",
        "ZEUSDB_LOG_FILE": str(log_file),
    })

    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0

    # With autolog disabled, the file subscriber should not be initialized.
    # Accept either "file not created" or "created but empty" as success.
    assert (not log_file.exists()) or log_file.stat().st_size == 0

# ------------------------------------------------------------
# The environment matrix from benchmark 41
# ------------------------------------------------------------
# The script set three environment combinations in one process and re-imported
# the package each time. That cannot work. _auto_configure_logging() is guarded
# by a module level _logging_configured flag and _configure_rust_logging uses
# os.environ.setdefault, so the second and third settings had no effect. One
# subprocess per combination is the only shape that exercises the matrix.
_WORKLOAD = r"""
import numpy as np
import zeusdb_vector_database as zdb
v = zdb.VectorDatabase()
idx = v.create('hnsw', dim=8)
vals = np.random.rand(4, 8).astype('float32').tolist()
idx.add({'vectors': vals})
idx.search(vals[0], top_k=2)
"""


def _run_with_logging_env(**overrides):
    """Run the workload with a clean environment plus the named settings.

    Every ZEUSDB_ variable is stripped first. The parent pytest process has
    already imported the package, and _configure_rust_logging writes
    ZEUSDB_LOG_LEVEL, ZEUSDB_LOG_FORMAT and ZEUSDB_LOG_TARGET into its own
    os.environ, so an inherited copy would leak the parent's configuration into
    the child and the combination under test would not be the one running.
    PYTEST_CURRENT_TEST is dropped for the same reason, since it drives
    _detect_environment.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ZEUSDB_") and k != "PYTEST_CURRENT_TEST"}
    env.update(overrides)

    proc = subprocess.run(
        [sys.executable, "-c", _WORKLOAD],
        env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return (proc.stdout.decode("utf-8", errors="ignore"),
            proc.stderr.decode("utf-8", errors="ignore"))


def _json_lines(text):
    entries = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            entries.append(json.loads(stripped))
    return entries

# ------------------------------------------------------------
# Test 83: Logging: debug level, JSON format, target left unset
# ------------------------------------------------------------
def test_logging_env_debug_json_defaults_to_stderr():
    stdout, stderr = _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
    )

    # With no ZEUSDB_LOG_TARGET and no log file, _configure_rust_logging
    # defaults the target to stderr, so nothing lands on stdout.
    entries = _json_lines(stderr)
    assert entries, f"No JSON log lines on stderr.\nSTDERR:\n{stderr}"
    assert _json_lines(stdout) == []

    for entry in entries:
        assert set(entry) >= {"level", "fields", "timestamp", "target"}
        assert "message" in entry["fields"]
        assert entry["target"].startswith("zeusdb_vector_database")

    levels = {entry["level"] for entry in entries}
    # debug is the point of this combination, so a DEBUG record must appear.
    assert "DEBUG" in levels
    assert "INFO" in levels

    messages = [entry["fields"]["message"] for entry in entries]
    assert any("HNSW index created successfully" in m for m in messages)
    assert any("Vector addition completed" in m for m in messages)

# ------------------------------------------------------------
# Test 84: Logging: trace level, human format
# ------------------------------------------------------------
def test_logging_env_trace_human():
    stdout, stderr = _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="trace",
        ZEUSDB_LOG_FORMAT="human",
    )

    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    assert lines, f"No log lines on stderr.\nSTDERR:\n{stderr}"
    assert stdout == ""

    # Human format is not JSON.
    assert not any(ln.lstrip().startswith("{") for ln in lines)

    # Every line opens with an RFC 3339 timestamp followed by a level token.
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+(TRACE|DEBUG|INFO|WARN|ERROR)\b")
    matches = [pattern.match(ln) for ln in lines]
    assert all(matches), lines[:3]

    levels = {m.group(1) for m in matches if m is not None}
    # trace is the only level that produces TRACE records, and it is the clause
    # that distinguishes this combination from the debug one.
    assert "TRACE" in levels
    assert "INFO" in levels

    # The initialization record reports the configuration it resolved.
    init = [ln for ln in lines if "ZeusDB logging initialized successfully" in ln]
    assert len(init) == 1
    assert "format=human" in init[0]
    assert "target=stderr" in init[0]
    assert init[0].split()[1] == "TRACE"

# ------------------------------------------------------------
# Test 85: Logging: error level, file target
# ------------------------------------------------------------
def test_logging_env_error_level_file_target(tmp_path):
    log_file = tmp_path / "zeus.log"

    stdout, stderr = _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="error",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
    )

    # The file target diverts everything away from the console.
    assert stdout == ""
    assert stderr == ""

    # One file, the one the caller named. Both layers open it, and nothing
    # creates a dated sibling any more.
    written = sorted(p.name for p in tmp_path.glob("*"))
    assert written == ["zeus.log"]

    # At error level a clean run produces no records at all.
    assert log_file.stat().st_size == 0

# ------------------------------------------------------------
# Test 86: Logging: the file target writes to the file that was named
# ------------------------------------------------------------
def test_logging_env_file_target_writes_to_the_named_file(tmp_path):
    """ZEUSDB_LOG_FILE names the file, and the records land in it.

    It previously always went through a daily rotating appender that read the
    value as a directory plus a base name, so ZEUSDB_LOG_FILE=zeus.log wrote
    zeus.log.2026-08-03 and the named path stayed empty. Rotation is now asked
    for through ZEUSDB_LOG_ROTATION rather than imposed.
    """
    log_file = tmp_path / "zeus.log"

    _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
    )

    assert log_file.exists()
    assert log_file.stat().st_size > 0

    # No dated sibling is created alongside it.
    assert [p.name for p in tmp_path.glob("*")] == ["zeus.log"]

    entries = _json_lines(log_file.read_text(encoding="utf-8", errors="ignore"))
    assert entries
    assert "DEBUG" in {entry["level"] for entry in entries}
    messages = [entry["fields"]["message"] for entry in entries]
    assert any("HNSW index created successfully" in m for m in messages)

    # The resolved destination is named in the log itself, so a caller who
    # cannot find the output can read where it went.
    assert any("ZeusDB file logging writing to resolved path" in m for m in messages)
    resolved = [
        entry["fields"]["log_file"]
        for entry in entries
        if "log_file" in entry["fields"]
    ]
    assert resolved
    assert os.path.basename(resolved[0]) == "zeus.log"

# ------------------------------------------------------------
# Test 101: ZEUSDB_LOG_ROTATION=never writes the file that was named
# ------------------------------------------------------------
def test_log_rotation_never_writes_the_named_file(tmp_path):
    """never is the default, and naming it explicitly does the same thing.

    This is the pair of test 102. Between them they pin both values of the knob
    that decides whether the file target rotates.
    """
    log_file = tmp_path / "zeus.log"

    _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
        ZEUSDB_LOG_ROTATION="never",
    )

    assert [p.name for p in tmp_path.glob("*")] == ["zeus.log"]
    assert log_file.stat().st_size > 0

    entries = _json_lines(log_file.read_text(encoding="utf-8", errors="ignore"))
    init = [e for e in entries if e["fields"].get("operation") == "logging_init_file"]
    assert len(init) == 1
    assert init[0]["fields"]["rotation"] == "never"
    assert os.path.basename(init[0]["fields"]["log_file"]) == "zeus.log"

# ------------------------------------------------------------
# Test 102: ZEUSDB_LOG_ROTATION=daily bounds disk growth
# ------------------------------------------------------------
def test_log_rotation_daily_writes_a_dated_file(tmp_path):
    """daily is the way to keep the file from growing without bound.

    The rotating appender reads ZEUSDB_LOG_FILE as a directory plus a base name
    and appends the UTC date, so the Rust records land in a file that is not the
    one the caller typed. The startup record names the resolved path, which is
    what makes the dated file discoverable.
    """
    log_file = tmp_path / "zeus.log"

    _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
        ZEUSDB_LOG_ROTATION="daily",
    )

    written = sorted(p.name for p in tmp_path.glob("*"))
    dated = [name for name in written if name != "zeus.log"]
    assert len(dated) == 1
    dated = dated[0]
    assert dated.startswith("zeus.log.")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", dated[len("zeus.log."):])

    # The Rust records are in the dated file, not the name the caller typed,
    # which is the whole reason the resolved path has to be reported.
    entries = _json_lines((tmp_path / dated).read_text(encoding="utf-8", errors="ignore"))
    assert entries
    assert "DEBUG" in {entry["level"] for entry in entries}
    messages = [entry["fields"]["message"] for entry in entries]
    assert any("HNSW index created successfully" in m for m in messages)

    init = [e for e in entries if e["fields"].get("operation") == "logging_init_file"]
    assert len(init) == 1
    assert init[0]["fields"]["rotation"] == "daily"
    assert os.path.basename(init[0]["fields"]["log_file"]) == dated

    # The Python file handler opens the name it was given and does not rotate,
    # so under daily the named path is created and stays empty, because nothing
    # in this package logs through the Python logger. Asserted rather than
    # expected. Under never the two layers share one file and this does not
    # arise.
    assert log_file.exists()
    assert log_file.stat().st_size == 0

# ------------------------------------------------------------
# Test 103: an unrecognised rotation value falls back to never and says so
# ------------------------------------------------------------
def test_log_rotation_unrecognised_value_warns_and_falls_back(tmp_path):
    """A knob that bounds disk growth must not fail silently.

    Falling back to never without a word would leave a caller who mistyped
    daily believing rotation was on while the file grew unbounded. The warning
    is emitted at warn, which is the default level, so it is visible without
    turning logging up.
    """
    log_file = tmp_path / "zeus.log"

    _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
        ZEUSDB_LOG_ROTATION="dayly",
    )

    assert [p.name for p in tmp_path.glob("*")] == ["zeus.log"]

    entries = _json_lines(log_file.read_text(encoding="utf-8", errors="ignore"))
    warnings = [
        entry for entry in entries
        if entry["level"] == "WARN"
        and "Unrecognised ZEUSDB_LOG_ROTATION" in entry["fields"]["message"]
    ]
    assert len(warnings) == 1
    assert warnings[0]["fields"]["value"] == "dayly"

    init = [e for e in entries if e["fields"].get("operation") == "logging_init_file"
            and e["level"] == "INFO"]
    assert len(init) == 1
    assert init[0]["fields"]["rotation"] == "never"

# ------------------------------------------------------------
# Test 91: the documented disable flag reaches both layers
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "flag", ["ZEUSDB_DISABLE_AUTO_LOGGING", "ZEUSDB_DISABLE_AUTOLOG"]
)
def test_disable_flag_stops_both_layers(tmp_path, flag):
    """The published name and the deprecated alias both disable both layers.

    ZEUSDB_DISABLE_AUTO_LOGGING is what the documentation site, the package
    README and the Python layer name. The Rust layer read ZEUSDB_DISABLE_AUTOLOG
    and nothing else, so the documented variable silently did nothing to it.
    Both layers now read both names.
    """
    log_file = tmp_path / "disabled.log"

    _run_with_logging_env(**{
        flag: "1",
        "ZEUSDB_LOG_LEVEL": "trace",
        "ZEUSDB_LOG_FORMAT": "json",
        "ZEUSDB_LOG_TARGET": "file",
        "ZEUSDB_LOG_FILE": str(log_file),
    })

    # Neither layer installed a writer, so nothing created the named file.
    assert not log_file.exists()
    assert list(tmp_path.glob("*")) == []

# ------------------------------------------------------------
# Test 92: a value outside the truthy set does not disable anything
# ------------------------------------------------------------
def test_disable_flag_requires_a_truthy_value(tmp_path):
    """The flag reads the way the Python layer has always read it.

    The Rust layer used to disable on the variable merely being present, so
    ZEUSDB_DISABLE_AUTOLOG=0 silenced the Rust half while the Python half
    stayed configured. Both now require true, 1 or yes.
    """
    log_file = tmp_path / "still-on.log"

    _run_with_logging_env(
        ZEUSDB_DISABLE_AUTO_LOGGING="0",
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
    )

    assert log_file.exists()
    assert log_file.stat().st_size > 0
