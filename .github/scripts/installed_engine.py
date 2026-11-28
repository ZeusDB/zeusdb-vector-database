"""Check that the running interpreter imports the package's extension from its
own environment, and whether the extension carries the test-support feature.

    python .github/scripts/installed_engine.py [--test-support]

The suite jobs of publish-pypi.yml and the musl job of ci.yml run it after
they install a wheel and before they run the suite, so the suite runs on the
wheel installed and on no other build. It prints the interpreter's version,
the package's version, and the extension's path and SHA-256.

It fails where the extension lies outside the interpreter's site-packages.
Without --test-support it fails where the extension holds `_Pause`, the class
only the binding's test-support feature compiles, and with it it fails where
the extension does not.
"""

import hashlib
import pathlib
import sys
import sysconfig

import zeusdb_vector_database as zvd


def main(args):
    support = "--test-support" in args
    engine = pathlib.Path(zvd._engine.__file__).resolve()
    site = pathlib.Path(sysconfig.get_paths()["platlib"]).resolve()
    digest = hashlib.sha256(engine.read_bytes()).hexdigest()
    print(sys.version.split()[0], zvd.__version__, engine, digest)
    if site not in engine.parents:
        return f"the extension {engine} lies outside this interpreter's site-packages {site}"
    pause = hasattr(zvd._engine, "_Pause")
    if pause and not support:
        return "_engine holds _Pause, which only the test-support feature compiles"
    if support and not pause:
        return "_engine holds no _Pause, so this build lacks the test-support feature"
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
