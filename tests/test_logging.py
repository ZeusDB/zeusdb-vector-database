"""Logging configuration through environment variables, exercised in subprocesses."""

import os
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
