# Repository conventions

## Public text

Every tracked file in this repository is publicly readable. `README.md`,
`pyproject.toml`, `Cargo.toml`, `Cargo.lock`, `LICENSE`, `NOTICE`, `src/`,
`bindings/` and `crates/` are published in the source distribution as well.

- Analysis, assessment and rationale go in the working report outside this
  repository, never in a tracked file.
- A governance or configuration file carries what it does and nothing about how
  well it works.
- No tracked file names an account, states how many people work on the project,
  or says that a control is inactive.
- No tracked file carries a local filesystem path.

## Line endings

`.gitattributes` pins the tracked source types to LF. Do not add a blanket
`* text=auto` rule.

## Versions

`pyproject.toml`, `[workspace.package]` in the root `Cargo.toml` and
`src/zeusdb_vector_database/__init__.py` declare the version, and a release tag
is the letter `v` followed by the same PEP 440 version in its canonical
spelling. All four must agree. Every crate manifest under `bindings/` and
`crates/` takes the version with `version.workspace = true` and declares none
of its own.

## Gates

Run from the repository root:

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --locked -- -D warnings
    cargo test --workspace --locked --no-default-features
    cargo deny --locked check
    pytest tests

Exactly one `clippy.toml` and one `deny.toml` exist, both at the root. A copy
in a crate directory replaces the root file for that crate without a warning.

Every cargo step passes `--locked`.
