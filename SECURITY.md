# Security policy

## Supported versions

Only the latest release on PyPI receives fixes. A fix ships as a new release rather than as a patch to an older one, so upgrade to the current release before reporting a problem that a newer release may already have closed.

| Version | Supported |
| --- | --- |
| The latest release on PyPI | Yes |
| Anything older | No |

## Reporting a vulnerability

Report privately. Do not open a public issue for a vulnerability.

1. Use GitHub private vulnerability reporting for this repository, from the Security tab, where it is offered.
2. Otherwise, email contact@zeusdb.com with the subject line "zeusdb-vector-database security".

Include the package version, the platform, and the smallest input that reproduces the problem. A saved index directory that crashes the loader is a vulnerability report here, because the loader is meant to refuse a hostile file rather than die on it. So is a parameter value that ends the interpreter instead of raising.

## What to expect

An acknowledgement within seven days, and an assessment of whether the report is a vulnerability within fourteen days of that. Where it is, a fix in the next release, and an entry in CHANGELOG.md that credits the reporter unless they ask otherwise.

## Scope

In scope are the `zeusdb-vector-database` package on PyPI, this repository's source under `bindings/`, `crates/` and `src/`, and the workflows under `.github/workflows/`.

Out of scope, and reported in their own repositories, are the `zeusdb` umbrella package, `langchain-zeusdb` and `llama-index-vector-stores-zeusdb`.

Advisories against the Rust crates the wheel links are checked daily by cargo-deny in `.github/workflows/dependency-scan.yml` and appear in the Security tab of this repository. A report that an advisory is unaddressed is welcome, with its RUSTSEC identifier.

## How releases are built

Releases are built and published by `.github/workflows/publish-pypi.yml` through PyPI trusted publishing. Every action in the workflow is pinned to a commit SHA, every wheel is checked against the SHA-256 recorded when it was built before it is attested or uploaded, and each release carries a build provenance attestation. A downloaded wheel can be checked against that attestation with `gh attestation verify <wheel> --repo ZeusDB/zeusdb-vector-database`.
