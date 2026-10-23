"""Decide whether a change touches documentation alone, for the jobs in ci.yml.

Each job runs this as its first step after the checkout and runs every later
step only when the answer is false, so a change to documentation alone still
reports every check, quickly. The answer goes to $GITHUB_OUTPUT as
documentation-only=true or documentation-only=false.

A push compares the commit before the push with the pushed commit. A pull
request compares the merge base of its base and head with its head. The
answer is true only when that comparison names at least one path and every
path it names is in DOCUMENTATION. A new branch, a commit the checkout does
not hold, an event other than these two, or any error answers false, so such
a change runs everything.

The checkout must hold the whole history, which `fetch-depth: 0` gives.
"""

import os
import subprocess
import sys

# Files no test reads and no job in ci.yml builds from. README.md and
# examples/ are not here, because tests/test_readme_examples.py and
# tests/test_examples.py run them.
DOCUMENTATION = frozenset(
    {
        "ATTRIBUTION.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "LICENSE",
        "NOTICE",
        "SECURITY.md",
    }
)

# The `before` of a push that creates a branch.
NO_COMMIT = "0" * 40


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def changed_paths(event, environ):
    """The paths the change touches, or None where it cannot be compared."""
    if event == "push":
        before, after = environ.get("BEFORE_SHA", ""), environ.get("AFTER_SHA", "")
        if not before or before == NO_COMMIT or not after:
            return None
        commits = [before, after]
    elif event == "pull_request":
        base, head = environ.get("BASE_SHA", ""), environ.get("HEAD_SHA", "")
        if not base or not head:
            return None
        commits = [git("merge-base", base, head).decode().strip(), head]
    else:
        return None
    # Without renames, so a file moved onto a documentation path also names
    # the path it left.
    listing = git("diff", "--name-only", "--no-renames", "-z", *commits)
    return [path for path in listing.decode("utf-8").split("\0") if path]


def main():
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    try:
        paths = changed_paths(event, os.environ)
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError) as error:
        stderr = getattr(error, "stderr", b"") or b""
        print(f"The change could not be compared with its base: {error}")
        print(stderr.decode("utf-8", "replace").strip())
        paths = None

    if paths is None:
        answer = False
        print(f"No comparison for this {event or 'unknown'} event, so every step runs.")
    else:
        for path in paths:
            print(f"  {'documentation' if path in DOCUMENTATION else 'other        '}  {path}")
        answer = bool(paths) and all(path in DOCUMENTATION for path in paths)
        if answer:
            print("The change touches documentation alone, so the later steps are skipped.")
        else:
            print("The change touches more than documentation, so every step runs.")

    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        print("GITHUB_OUTPUT is not set", file=sys.stderr)
        return 1
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(f"documentation-only={'true' if answer else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
