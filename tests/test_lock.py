"""The lock that keeps a second writer off a directory, from Python.

`<path>.zdblock` beside the directory. A journaled index holds it for its
life, a save for the save, and a load while it reads. A second process is
refused with a RuntimeError, CollectionLocked, and a process killed holding
the lock releases it with its exit. A child forked from the process that
opened a journal refuses every mutation with JournalForked. A journaled
directory this process cannot write opens read only, with a RuntimeWarning.
"""

import gc
import json
import os
import stat
import subprocess
import sys
import time

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

DIM = 8

# A process that journals the directory named first, adds one record, says
# it is ready in the file named second, and waits for the file named third.
HOLDER = """
import sys, time
import numpy as np
from zeusdb_vector_database import VectorDatabase
path, ready, go = sys.argv[1:4]
index = VectorDatabase().create("hnsw", dim=8, expected_size=10)
index.journal_to(path)
index.add({"ids": ["theirs"], "embeddings": np.full((1, 8), 0.5, dtype=np.float32)})
open(ready, "w").write("ready")
start = time.time()
import os
while not os.path.exists(go) and time.time() - start < 120:
    time.sleep(0.01)
"""


def build(n):
    rng = np.random.default_rng(n)
    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=50)
    if n:
        result = index.add({"ids": [f"r{i}" for i in range(n)],
                            "embeddings": rng.standard_normal((n, DIM)).astype(np.float32)})
        assert result.is_success(), result.errors
    return index


def holder(tmp_path, name):
    path = tmp_path / name
    ready = tmp_path / f"{name}.ready"
    go = tmp_path / f"{name}.go"
    child = subprocess.Popen([sys.executable, "-c", HOLDER, str(path), str(ready), str(go)])
    start = time.time()
    while not ready.exists():
        assert child.poll() is None, "the holder exited before it was ready"
        assert time.time() - start < 60, "the holder never became ready"
        time.sleep(0.01)
    return child, path, go


def test_another_process_holding_the_journal_refuses_a_load_and_a_save(tmp_path):
    child, path, go = holder(tmp_path, "theirs.zdb")
    try:
        with pytest.raises(RuntimeError, match="is held by"):
            VectorDatabase().load(str(path))
        with pytest.raises(RuntimeError, match="is held by"):
            build(1).save(str(path))
    finally:
        go.write_text("")
        assert child.wait(timeout=60) == 0
    index = VectorDatabase().load(str(path))
    assert "theirs" in index


def test_a_killed_holder_releases_the_lock(tmp_path):
    """The system releases a killed holder's lock with its process. Windows
    may take a few milliseconds after the process has exited, so the load is
    tried again for a bounded time."""
    child, path, _ = holder(tmp_path, "killed.zdb")
    child.kill()
    child.wait(timeout=60)
    deadline = time.time() + 10
    while True:
        try:
            index = VectorDatabase().load(str(path))
            break
        except RuntimeError as error:
            if "is held by" not in str(error) or time.time() > deadline:
                raise
            time.sleep(0.01)
    assert "theirs" in index, "the record the holder added survived the kill"


def test_a_journaled_directory_this_process_cannot_write_opens_read_only(tmp_path):
    """The lock file made read only stands for a medium that refuses writes:
    the load cannot open it for writing, so it replays the journal in memory,
    warns, and refuses every change. A save into another directory copies
    it, and the copy opens writable."""
    index = build(3)
    path = tmp_path / "ro.zdb"
    index.journal_to(str(path))
    result = index.add({"ids": ["late"], "embeddings": np.full((1, DIM), 0.25, dtype=np.float32)})
    assert result.is_success(), result.errors
    del index
    gc.collect()
    lock = tmp_path / "ro.zdb.zdblock"
    os.chmod(lock, stat.S_IREAD)
    try:
        if os.access(lock, os.W_OK):
            pytest.skip("this process writes through a read-only permission")
        with pytest.warns(RuntimeWarning, match="opened read only"):
            loaded = VectorDatabase().load(str(path))
        assert len(loaded) == 4 and "late" in loaded
        assert loaded.journal_path is None
        assert not any(key.startswith("journal_") for key in loaded.get_stats())
        refused = loaded.add({"ids": ["new"], "embeddings": np.zeros((1, DIM), dtype=np.float32)})
        assert refused.total_errors == 1 and "cannot write" in refused.errors[0], refused.errors
        for change in (lambda: loaded.remove_point("r0"),
                       lambda: loaded.checkpoint(),
                       lambda: loaded.journal_to(str(tmp_path / "elsewhere.zdb")),
                       lambda: loaded.save(str(path))):
            with pytest.raises(RuntimeError, match="cannot write"):
                change()
        copy = tmp_path / "copy.zdb"
        loaded.save(str(copy))
        assert "journal" not in json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
        del loaded
        gc.collect()
        writable = VectorDatabase().load(str(copy))
        assert len(writable) == 4
        assert writable.add({"ids": ["new"], "embeddings": np.zeros((1, DIM), dtype=np.float32)}).is_success()
    finally:
        os.chmod(lock, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork exists on POSIX alone")
def test_a_forked_child_refuses_every_mutation_and_leaves_the_parent_whole(tmp_path):
    """The child shares the parent's open journal, so every mutation there is
    refused, and dropping the index there neither unlocks the directory nor
    stops the parent journaling."""
    index = build(2)
    path = tmp_path / "forked.zdb"
    index.journal_to(str(path))
    said = tmp_path / "child.said"
    pid = os.fork()
    if pid == 0:
        try:
            words = []
            result = index.add({"ids": ["child"], "embeddings": np.zeros((1, DIM), dtype=np.float32)})
            words.append("add refused" if result.total_errors == 1 and "forked" in result.errors[0]
                         else f"add {result.errors}")
            for name, change in (("remove", lambda: index.remove_point("r0")),
                                 ("checkpoint", lambda: index.checkpoint())):
                try:
                    change()
                    words.append(f"{name} ran")
                except RuntimeError as error:
                    words.append(f"{name} refused" if "forked" in str(error) else f"{name} {error}")
            del index
            gc.collect()
            said.write_text(", ".join(words))
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    assert said.read_text() == "add refused, remove refused, checkpoint refused"
    result = index.add({"ids": ["parent"], "embeddings": np.full((1, DIM), 0.5, dtype=np.float32)})
    assert result.is_success(), result.errors
    with pytest.raises(RuntimeError, match="is held by"):
        VectorDatabase().load(str(path))
    del index
    gc.collect()
    reopened = VectorDatabase().load(str(path))
    assert "parent" in reopened and "child" not in reopened and "r0" in reopened
