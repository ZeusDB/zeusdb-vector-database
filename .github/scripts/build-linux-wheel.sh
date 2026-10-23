#!/usr/bin/env bash
# Builds one Linux wheel inside a pinned build image, for publish-pypi.yml.
#
# The linux and musllinux jobs run it through `docker run`, with the
# repository at /io as the working directory and the maturin binary installed
# from .github/requirements/maturin.txt mounted read only at
# /usr/local/bin/maturin. The patchelf binary installed from
# .github/requirements/patchelf.txt is mounted read only at
# /opt/patchelf/bin/patchelf. Both binaries are static, so they run in every
# image.
#
# The first argument is the Rust target triple. The second is the wheel's
# compatibility, either a value for maturin's --compatibility or "auto", which
# leaves maturin to choose the tag from what the build links, as the manylinux
# images have always done.
set -euo pipefail

target="$1"
compatibility="$2"

# The manylinux images carry no rustup, and the musl images carry their own.
# Where it is missing, this is the rustup the runner images carry,
# fetched by version and checked against the SHA-256 the Rust project
# publishes beside it.
rustup_version="1.29.1"
rustup_sha256="dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71"

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 --silent --show-error --fail \
    --output /tmp/rustup-init \
    "https://static.rust-lang.org/rustup/archive/${rustup_version}/x86_64-unknown-linux-gnu/rustup-init"
  echo "${rustup_sha256}  /tmp/rustup-init" | sha256sum --check --strict
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain none
  export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
fi

# The toolchain rust-toolchain.toml names, with the one target this wheel
# needs. rustup checks every component against the release's manifest, and
# --no-self-update keeps rustup at the version above or the image's own.
channel=$(sed -n 's/^channel *= *"\([^"]*\)".*$/\1/p' rust-toolchain.toml)
if [ -z "$channel" ]; then
  echo "::error::rust-toolchain.toml names no channel"
  exit 1
fi
rustup toolchain install "$channel" --profile minimal --target "$target" --no-self-update
# Named for every later command, ahead of rust-toolchain.toml, so cargo runs
# the toolchain just installed and rustup fetches nothing else that file lists.
export RUSTUP_TOOLCHAIN="$channel"
rustup --version
rustc --version

# maturin runs patchelf by name when it copies a library into the wheel, so
# the mounted binary goes first on PATH, ahead of any patchelf the image
# carries.
export PATH="/opt/patchelf/bin:$PATH"
if [ "$(command -v patchelf)" != /opt/patchelf/bin/patchelf ]; then
  echo "::error::the patchelf on PATH is not the one mounted at /opt/patchelf/bin"
  exit 1
fi
patchelf --version

# By its full path, so the mounted binary builds the wheel whatever else the
# image carries.
maturin=/usr/local/bin/maturin
"$maturin" --version

args=(build --release --locked --out dist --target "$target")
if [ "$compatibility" != "auto" ]; then
  args+=(--compatibility "$compatibility")
fi
"$maturin" "${args[@]}"
