"""Logging configuration through environment variables, exercised in subprocesses."""

import os
import re
import sys
import subprocess
import json

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
# the package each time. That cannot work. auto_configure_logging() is guarded
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

    # The path the caller named is created by the Python file handler and left
    # empty. The Rust layer writes to a daily rotated sibling instead, which is
    # created even when the level is high enough that nothing is written to it.
    written = sorted(p.name for p in tmp_path.glob("*"))
    assert written[0] == "zeus.log"
    assert len(written) == 2
    assert log_file.stat().st_size == 0

    rotated = written[1]
    assert rotated.startswith("zeus.log.")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rotated[len("zeus.log."):])

    # At error level a clean run produces no records at all, in either file.
    assert (tmp_path / rotated).stat().st_size == 0

# ------------------------------------------------------------
# Test 86: Logging: the file target writes to a rotated sibling
# ------------------------------------------------------------
def test_logging_env_file_target_writes_to_rotated_sibling(tmp_path):
    """Current behaviour, asserted rather than expected.

    ZEUSDB_LOG_FILE names a path, and no records are ever written to it. The
    Rust layer builds a daily rotating appender whose stem is that path, so the
    records land in a sibling carrying a date suffix, while the Python file
    handler creates the named path and leaves it empty at these levels. The
    expectation this violates is that a caller who names a log file finds the
    log in it. This is also why test_logging_disabled_autoinit cannot fail on
    the behaviour it names, since the named path is empty either way.
    """
    log_file = tmp_path / "zeus.log"

    _run_with_logging_env(
        ZEUSDB_LOG_LEVEL="debug",
        ZEUSDB_LOG_FORMAT="json",
        ZEUSDB_LOG_TARGET="file",
        ZEUSDB_LOG_FILE=str(log_file),
    )

    assert log_file.exists()
    assert log_file.stat().st_size == 0

    rotated = [p for p in tmp_path.glob("zeus.log.*")]
    assert len(rotated) == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rotated[0].name[len("zeus.log."):])

    entries = _json_lines(rotated[0].read_text(encoding="utf-8", errors="ignore"))
    assert entries
    assert "DEBUG" in {entry["level"] for entry in entries}
    messages = [entry["fields"]["message"] for entry in entries]
    assert any("HNSW index created successfully" in m for m in messages)
