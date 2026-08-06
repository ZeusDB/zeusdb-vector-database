"""Run every file in ``examples/`` and check the output it documents.

The README is executed by ``test_readme_examples.py``. This is the same idea
applied to the examples directory, and it exists for the same reason: an example
nobody runs is an example that rots.

Each example carries a module level ``EXPECTED_OUTPUT`` string holding the
complete transcript it prints. Each is run in a subprocess, in a temporary
working directory, and its stdout is compared against that string.

Two allowances are applied to both sides of the comparison, uniformly across the
set.

Lines whose first non-space character is non-ASCII are dropped. That is what
removes the progress banner ``save()`` and ``load()`` print, which is emitted by
the Rust layer straight to the file descriptor and so does not even arrive in
document order when stdout is a pipe.

A literal ``...`` in an expected line matches any run of characters. It is used
only where a value genuinely moves between runs, which is wall clock timing and
anything downstream of product quantization training, since that trains with an
unseeded k-means. Every such line is paired with a stable verdict token in the
example itself, so the assertion still fails if the behaviour changes rather
than only the digits.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))

# Generous enough that a loaded machine does not fail the suite, short enough
# that a hung example is reported rather than waited on.
TIMEOUT_S = 300


def _expected_output(path):
    """Read EXPECTED_OUTPUT out of a file without importing or running it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_OUTPUT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def _visible(text):
    """Drop the save() and load() progress banner, and trailing blank lines."""
    kept = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped and ord(stripped[0]) > 127:
            continue
        kept.append(line.rstrip())
    while kept and not kept[-1]:
        kept.pop()
    return kept


def _matches(expected, actual):
    """Compare one line, treating ``...`` as any run of characters."""
    if "..." not in expected:
        return expected == actual
    parts = expected.split("...")
    if not actual.startswith(parts[0]) or not actual.endswith(parts[-1]):
        return False
    cursor = len(parts[0])
    for part in parts[1:-1]:
        found = actual.find(part, cursor)
        if found < 0:
            return False
        cursor = found + len(part)
    return cursor <= len(actual) - len(parts[-1])


def test_examples_were_found():
    """A glob that silently matched nothing would make every other test pass."""
    assert len(SCRIPTS) >= 6, f"found only {[p.name for p in SCRIPTS]}"
    for path in SCRIPTS:
        expected = _expected_output(path)
        assert expected, f"{path.name} has no EXPECTED_OUTPUT"
        assert len(_visible(expected)) >= 5, f"{path.name} documents almost no output"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs_and_prints_what_it_documents(script, tmp_path):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=TIMEOUT_S,
    )
    assert completed.returncode == 0, (
        f"{script.name} exited {completed.returncode}\n\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )

    got = _visible(completed.stdout)
    want = _visible(_expected_output(script))
    assert len(got) == len(want), (
        f"{script.name} printed {len(got)} lines against {len(want)} documented.\n"
        f"expected:\n" + "\n".join(want) + "\n\ngot:\n" + "\n".join(got)
    )
    for number, (expected, actual) in enumerate(zip(want, got), start=1):
        assert _matches(expected, actual), (
            f"{script.name} line {number} does not match what it documents.\n"
            f"expected: {expected!r}\n     got: {actual!r}"
        )


def test_no_em_dashes():
    # Spelled as an escape so this file does not contain the character it bans.
    em_dash = "\u2014"
    files = list(SCRIPTS) + [EXAMPLES / "README.md"]
    offenders = [p.name for p in files if em_dash in p.read_text(encoding="utf-8")]
    assert not offenders, f"em-dash found in {offenders}"


def test_examples_readme_lists_every_example():
    listing = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    missing = [p.name for p in SCRIPTS if p.name not in listing]
    assert not missing, f"examples/README.md does not mention {missing}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
