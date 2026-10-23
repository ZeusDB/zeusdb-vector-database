"""List the libraries each wheel bundles, for the audit job in publish-pypi.yml.

For every wheel it prints the files in the wheel's .libs or .dylibs folder,
and for every ELF file in the wheel the libraries it needs and its run path.
maturin copies each library a musllinux extension needs, other than musl's
libc.so, into the wheel's .libs folder and points the extension at the copy.
A musllinux wheel therefore fails here when it bundles no library, or when an
ELF file in it needs a library other than libc.so that the wheel does not
carry.

The audit job runs it with the Python of its abi3audit environment, whose
pyelftools .github/requirements/abi3audit.txt pins by hash.
"""

import io
import sys
import zipfile

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

# The one library a musllinux wheel may need without carrying it.
MUSL_LIBC = "libc.so"


def platform_tags(wheel):
    """The platform tags in a wheel's file name."""
    name = wheel.replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".whl").split("-")[-1].split(".")


def dynamic_entries(data):
    """The needed libraries and the run path of one ELF file."""
    needed, run_path = [], []
    # The dynamic segment rather than the dynamic section, since it is what
    # the loader reads.
    for segment in ELFFile(io.BytesIO(data)).iter_segments():
        if not isinstance(segment, DynamicSegment):
            continue
        for tag in segment.iter_tags():
            if tag.entry.d_tag == "DT_NEEDED":
                needed.append(tag.needed)
            elif tag.entry.d_tag == "DT_RPATH":
                run_path.append(tag.rpath)
            elif tag.entry.d_tag == "DT_RUNPATH":
                run_path.append(tag.runpath)
    return needed, run_path


def check(wheel):
    """Print one wheel's libraries and return what is wrong with them."""
    errors = []
    musllinux = any(tag.startswith("musllinux_") for tag in platform_tags(wheel))
    print(wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        bundled = sorted(
            name for name in names if name.split("/", 1)[0].endswith((".libs", ".dylibs"))
        )
        carried = {name.rsplit("/", 1)[-1] for name in bundled}
        print(f"  bundles {', '.join(bundled) if bundled else 'no library'}")
        for name in names:
            data = archive.read(name)
            if not data.startswith(b"\x7fELF"):
                continue
            needed, run_path = dynamic_entries(data)
            print(f"  {name}")
            print(f"    needs {', '.join(needed) if needed else 'no library'}")
            print(f"    run path {':'.join(run_path) if run_path else 'none'}")
            if musllinux:
                for library in needed:
                    if library != MUSL_LIBC and library not in carried:
                        errors.append(f"{name} needs {library}, which the wheel does not carry")
    if musllinux and not bundled:
        errors.append("the musllinux wheel bundles no library")
    return errors


def main():
    wheels = sys.argv[1:]
    if not wheels:
        print("::error::no wheel was given")
        return 1
    failed = False
    for wheel in wheels:
        for error in check(wheel):
            print(f"::error::{wheel}: {error}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
