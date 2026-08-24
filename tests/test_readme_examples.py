"""Run the README's Python examples and check their documented output.

The README drifted from the code because nothing executed it. This module
executes it.

Every fenced ``python`` block in README.md is run, in document order. Blocks
share a namespace within a level-2 section and start fresh at the next one, so
each section reads as one program. A block preceded by ``<!-- zeusdb:skip -->``
is not run; that marker is on the logging examples alone, which set process wide
environment variables and install a global logging subscriber.

Where a block is followed by an italic ``*Output*`` line and a plain fence, the
captured stdout is compared against it. Lines beginning with a non-ASCII
character are dropped from the comparison first, which is what removes the
progress banner ``save()`` and ``load()`` print.
"""

import io
import os
import re
import contextlib
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

SKIP_MARKER = "<!-- zeusdb:skip -->"
OUTPUT_LEADS = ("*Output*", "*Results Output:*", "*Output, with the progress lines omitted*")


class Block:
    """One executable README example and the output it documents."""

    def __init__(self, section, line, source, expected):
        self.section = section
        self.line = line
        self.source = source
        self.expected = expected

    @property
    def id(self):
        return f"L{self.line}"


def _visible(text):
    """Drop the progress lines save() and load() print, and trailing blanks."""
    kept = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped and ord(stripped[0]) > 127:
            continue
        kept.append(line.rstrip())
    while kept and not kept[-1]:
        kept.pop()
    return kept


def _parse(markdown):
    """Split the README into executable blocks, grouped by level-2 section."""
    lines = markdown.splitlines()
    blocks = []
    section = "(preamble)"
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("## "):
            section = line[3:].strip()
            i += 1
            continue

        if line.startswith("```python"):
            skip = any(lines[j].strip() == SKIP_MARKER for j in range(max(0, i - 3), i))
            start = i + 1
            end = start
            while end < len(lines) and not lines[end].startswith("```"):
                end += 1
            source = "\n".join(lines[start:end])
            i = end + 1

            expected = None
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() in OUTPUT_LEADS:
                j += 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].startswith("```"):
                    fence_start = j + 1
                    fence_end = fence_start
                    while fence_end < len(lines) and not lines[fence_end].startswith("```"):
                        fence_end += 1
                    expected = "\n".join(lines[fence_start:fence_end])

            if not skip:
                blocks.append(Block(section, start, source, expected))
            continue

        i += 1

    return blocks


BLOCKS = _parse(README.read_text(encoding="utf-8"))


def test_readme_has_examples():
    """A parser that silently matched nothing would make every other test pass."""
    assert len(BLOCKS) >= 25
    assert sum(1 for b in BLOCKS if b.expected is not None) >= 15


@pytest.mark.parametrize("section", sorted({b.section for b in BLOCKS}))
def test_readme_section(section, tmp_path, monkeypatch):
    """Run one section's blocks in order and check the documented output."""
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__readme__"}

    for block in [b for b in BLOCKS if b.section == section]:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(block.source, f"README.md:{block.line}", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - only on a broken example
            pytest.fail(
                f"README.md:{block.line} in section {section!r} raised "
                f"{type(exc).__name__}: {exc}\n\n{block.source}"
            )

        if block.expected is None:
            continue

        got = _visible(buffer.getvalue())
        want = _visible(block.expected)
        assert got == want, (
            f"README.md:{block.line} in section {section!r} printed output that "
            f"does not match the documented block.\n"
            f"expected:\n" + "\n".join(want) + "\n\ngot:\n" + "\n".join(got)
        )


def test_no_em_dashes():
    assert "—" not in README.read_text(encoding="utf-8")


def test_readme_names_only_entry_points_that_exist():
    """Catch a method named in prose that the extension no longer exposes."""
    import zeusdb_vector_database
    from zeusdb_vector_database import VectorDatabase, AddResult, HNSWIndex

    text = README.read_text(encoding="utf-8")
    named = set(re.findall(r"[`.]([a-z_][a-z0-9_]*)\(\)", text))
    known = (
        set(dir(HNSWIndex))
        | set(dir(AddResult))
        | set(dir(VectorDatabase))
        # The package's own exports, read rather than listed, so a function
        # added to __all__ and documented does not also have to be named here.
        # The three logging initializers used to be in the literal below.
        | set(dir(zeusdb_vector_database))
        | {
            "create", "load", "search", "add",
            "print", "len", "sorted", "matched", "range",
            "enumerate", "round", "compile", "getLogger", "setLevel", "basicConfig",
            "default_rng", "random", "tolist", "keys", "items", "listdir",
        }
    )
    missing = sorted(name for name in named if name not in known)
    assert not missing, f"README names entry points that do not exist: {missing}"


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(pytest.main([__file__, "-v"]))
