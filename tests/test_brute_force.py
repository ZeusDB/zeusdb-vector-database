"""The page a search returns, against a nearest neighbour search computed in numpy.

# Why this file exists

Nothing in this suite compared the index against an independent implementation
of the thing it computes. Every search test asserted a shape, a count, a filter
or a relative ordering, and a traversal that returned plausible neighbours in
the wrong order would have passed all of them.

# The regime

HNSW is approximate, so a comparison against brute force is only an equality
inside a regime where the traversal is exact. This file uses two, and both were
measured rather than assumed.

**The traversal, at `ef_search` of at least the record count and `m` of at least
16.** Two things have to hold. The beam has to be wide enough to hold every node
it reaches, which `ef_search >= n` gives: raising it above `n` changes nothing,
measured at `n = 1000` where `ef_search` of `n` and of `4n` returned the same
page in every cell. And every node has to be reachable, which is a property of
the graph rather than of the search. At `m = 2` it is badly false. Sixty of 500
records at `dim=128` under `l2` are returned by no query at all, including a
query that is the record's own vector, which is the mechanism behind the
README's "loses recall that no search width recovers". Orphans were counted at
five seeds across six metric-and-dimension cells for each of `m` in 8, 12, 16,
24 and 32 at `n` of 25 through 800. The last orphan is at `m = 8`, `n = 400`.
There were none at `m >= 12` anywhere, and none at any `m >= 8` for `n <= 200`.
This file runs at `m >= 16` and `n <= 200`, inside both margins.
`test_every_record_in_the_regime_is_reachable` holds the precondition, so a
change that orphans a node fails there rather than intermittently here.

**The exact-scan path, at any `m`.** A filter matching at or below
`FULL_SCAN_THRESHOLD`, which is 5,000, scores every matching record and ranks
them, so it never consults the graph. It is exact where the traversal is
orphaned. At `m = 2`, `n = 500`, `dim = 128` under `l2` the scan matched brute
force on all thirty queries where the traversal at the same `m` misses.

# What is excluded, and why

`quantized_only` is not compared. It sheds the raw vectors at training, so
there is nothing it holds to compare against. Measured at 1,200 records of 64
dimensions with 8 subvectors, its recall at 10 against brute force over the
inserted vectors is 0.420, and against brute force over the reconstructions
`get_records()` hands back it is 0.667, because the ADC table quantizes the
query as well. Neither number is an exactness claim, and pinning either would
pin today's codebook.

# Ties

Two records at the same distance make the page order underdetermined, so
nothing here asserts an order across one. `assert_exact_page` compares the page
against brute force by the two properties a tie leaves well defined. Every
returned score matches that record's brute force distance, and no record left
out is nearer than the last one returned. The strict ordered comparison runs
only where the corpus carries no tie within tolerance at the page boundary.

# The budget and the deeper run

The committed run is a few seconds. `ZEUSDB_BRUTE_SEEDS` raises the seed count
for a soak by hand and is absent in CI, which is what `ZEUSDB_FUZZ_CASES` does
for the dump fuzzer.
"""

import os

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

# Seeds per parameter cell. A budget rather than a target: this runs on every
# commit, and a gate nobody can afford to run finds nothing.
SEEDS = int(os.environ.get("ZEUSDB_BRUTE_SEEDS", "3"))

# The comparison is between an f64 numpy reduction and an f32 kernel that
# accumulates in a different order, and for cosine against a vector the index
# normalised in f32 on the way in, so the two scores agree to f32 precision
# rather than exactly.
#
# The tolerance carries a term in the magnitude because the disagreement is
# relative. The worst measured across every traversal cell in this file is
# 1.686e-05, on an l1 distance of 131.6 at dim 128, which is 1.3e-07 of the
# value. An absolute tolerance sized for that would be loose at cosine, where
# every distance is between 0 and 2. Measured worst against this function is a
# factor of 14 inside it.
def tolerance(value):
    return 1e-6 + 1e-6 * abs(value)


# `dot` is here on the same terms as the other three. It is the one space whose
# distance can be negative, since it returns `1 - dot` and an inner product
# above one gives one, so it is also the one that exercises the ordering with a
# sign the vendored traversal used to assert against.
SPACES = ("cosine", "l2", "l1", "dot")


def brute_force(space, corpus, query):
    """The distance the index scores, computed independently in numpy at f64."""
    c = corpus.astype(np.float64)
    q = query.astype(np.float64)
    if space == "cosine":
        return 1.0 - (c @ q) / (np.linalg.norm(c, axis=1) * np.linalg.norm(q))
    if space == "l2":
        return np.linalg.norm(c - q, axis=1)
    if space == "l1":
        return np.abs(c - q).sum(axis=1)
    if space == "dot":
        # What `DotDist` computes, and deliberately not a normalised one: this
        # space does not normalise what it is given, so brute force must not
        # either.
        return 1.0 - (c @ q)
    raise AssertionError(space)


def assert_exact_page(page, ids, distances, top_k, where):
    """The page is a correct top-k under `distances`, tie included.

    Four properties. The page holds no record twice, the scores come back in
    non-decreasing order, each score is the brute force distance of the record
    it is attached to, and nothing left out is nearer than the last record
    returned. The last is the one that catches a traversal that missed a
    neighbour, and it is stated as an inequality so a tie at the boundary is not
    a failure.

    Where the boundary carries no tie the ordered id list is compared outright,
    which the inequality alone would not catch.
    """
    by_id = dict(zip(ids, distances))
    assert len(page) == min(top_k, len(ids)), f"{where}: page length {len(page)}"

    got_ids = [row["id"] for row in page]
    assert len(set(got_ids)) == len(got_ids), f"{where}: a record appears twice"

    scores = [row["score"] for row in page]
    for earlier, later in zip(scores, scores[1:]):
        assert earlier <= later + tolerance(later), (
            f"{where}: page is not sorted, {scores}"
        )

    for row in page:
        want = by_id[row["id"]]
        got = row["score"]
        assert abs(got - want) <= tolerance(want), (
            f"{where}: {row['id']} scored {got} against brute force {want}"
        )

    returned = set(got_ids)
    cutoff = max(by_id[identifier] for identifier in got_ids)
    for identifier, distance in by_id.items():
        if identifier in returned:
            continue
        assert distance >= cutoff - tolerance(cutoff), (
            f"{where}: {identifier} at {distance} is nearer than the page's last "
            f"at {cutoff}, so the page is not a top-{top_k}"
        )

    # An ordered comparison, where no tie makes the order a choice.
    ranked = sorted(distances)
    boundary_tie = any(
        abs(ranked[i] - ranked[i + 1]) <= tolerance(ranked[i])
        for i in range(min(top_k, len(ranked) - 1))
    )
    if not boundary_tie:
        order = np.argsort(distances, kind="stable")
        want = [ids[i] for i in order[:top_k]]
        assert got_ids == want, f"{where}: page {got_ids} against brute force {want}"


def corpus_for(seed, n, dim):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def build(space, dim, m, n, corpus, metadata=None):
    index = VectorDatabase().create(
        "hnsw", dim=dim, space=space, m=m, expected_size=max(n, 8)
    )
    record = {"ids": [f"r{i}" for i in range(n)], "embeddings": corpus}
    if metadata is not None:
        record["metadatas"] = metadata
    result = index.add(record)
    assert result.total_errors == 0, result.errors
    return index


# ============================================================================
# THE PRECONDITION
# ============================================================================


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("dim", [3, 128])
def test_every_record_in_the_regime_is_reachable(space, dim):
    """No record is orphaned at the `m` and `n` the exactness tests run at.

    A record its own vector does not retrieve is one no query can reach, and no
    `ef_search` recovers it, so the exactness claim below rests on this. It is
    held rather than assumed because it is a property of the graph builder, and
    a change there would otherwise surface as an intermittent failure in the
    tests that depend on it.
    """
    n = 200
    corpus = corpus_for(11, n, dim)
    index = build(space, dim, 16, n, corpus)
    orphans = []
    for i in range(n):
        page = index.search(vector=corpus[i].tolist(), top_k=n, ef_search=n)
        if f"r{i}" not in {row["id"] for row in page}:
            orphans.append(f"r{i}")
    assert orphans == [], (
        f"{len(orphans)} of {n} records are reachable by no query at m=16, "
        f"which voids the regime the exactness tests run in: {orphans[:10]}"
    )


# ============================================================================
# THE TRAVERSAL
# ============================================================================


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("dim", [2, 3, 8, 32, 128])
@pytest.mark.parametrize("m", [16, 32])
def test_the_traversal_matches_brute_force_inside_the_regime(space, dim, m):
    """At `ef_search >= n` and `m >= 16` the page is the exact nearest set."""
    for seed in range(SEEDS):
        for n, top_k in ((32, 5), (200, 10)):
            corpus = corpus_for(seed * 131 + dim * 7 + n, n, dim)
            index = build(space, dim, m, n, corpus)
            rng = np.random.default_rng(seed * 977 + dim)
            for probe in range(4):
                query = rng.standard_normal(dim).astype(np.float32)
                page = index.search(vector=query.tolist(), top_k=top_k, ef_search=n)
                assert_exact_page(
                    page,
                    [f"r{i}" for i in range(n)],
                    brute_force(space, corpus, query),
                    top_k,
                    f"{space} dim={dim} m={m} n={n} seed={seed} probe={probe}",
                )


@pytest.mark.parametrize("space", SPACES)
def test_a_query_that_is_a_record_returns_the_record_brute_force_names(space):
    """A stored vector as the query, against what brute force says is nearest.

    Under cosine, l1 and l2 that is the record itself at distance zero, because
    all three are metrics. **Under dot it need not be.** An inner product is not
    a metric and has no zero: a stored vector scores `1 - v.v` against itself,
    and a longer vector pointing much the same way scores lower still, so the
    nearest neighbour of a record can be a different record. That is a property
    of inner product search rather than a defect, and asserting the metric rule
    here would have asserted something untrue.
    """
    n, dim = 120, 16
    corpus = corpus_for(4, n, dim)
    index = build(space, dim, 16, n, corpus)
    for i in range(0, n, 7):
        page = index.search(vector=corpus[i].tolist(), top_k=1, ef_search=n)
        truth = brute_force(space, corpus, corpus[i])
        nearest = int(np.argmin(truth))
        assert page[0]["id"] == f"r{nearest}", (
            f"{space}: query r{i} found {page[0]['id']} where brute force says r{nearest}"
        )
        assert abs(page[0]["score"] - truth[nearest]) <= tolerance(truth[nearest])
        if space != "dot":
            assert nearest == i, f"{space}: r{i} is not its own nearest neighbour"
            assert abs(page[0]["score"]) <= tolerance(1.0), page[0]["score"]


@pytest.mark.parametrize("space", SPACES)
def test_top_k_is_a_prefix_of_a_wider_page(space):
    """Asking for fewer results returns the front of the longer page.

    Independent of brute force, and it fails on a traversal whose ordering
    depends on how many results were asked for.
    """
    n, dim = 200, 24
    corpus = corpus_for(21, n, dim)
    index = build(space, dim, 16, n, corpus)
    rng = np.random.default_rng(22)
    for _ in range(4):
        query = rng.standard_normal(dim).astype(np.float32).tolist()
        wide = [row["id"] for row in index.search(vector=query, top_k=40, ef_search=n)]
        for k in (1, 5, 10, 25):
            short = [
                row["id"] for row in index.search(vector=query, top_k=k, ef_search=n)
            ]
            assert short == wide[:k], f"{space}: top_{k} is not a prefix of top_40"


# ============================================================================
# THE EXACT-SCAN PATH
# ============================================================================


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("m", [2, 16])
def test_the_scan_matches_brute_force_at_any_m(space, m):
    """A filter under the scan threshold ranks every match, so `m` cannot matter.

    `m = 2` is in the matrix deliberately. The traversal at that degree leaves
    records no query reaches, and this path returns the exact page anyway, which
    is what shows the two are not the same code.
    """
    n, dim, top_k = 400, 48, 10
    for seed in range(SEEDS):
        corpus = corpus_for(seed * 53 + 3, n, dim)
        # Two thirds carry the field, so the scan ranks a subset rather than all.
        keep = [i % 3 != 0 for i in range(n)]
        index = build(
            space,
            dim,
            m,
            n,
            corpus,
            metadata=[{"g": "in" if k else "out"} for k in keep],
        )
        subset = [i for i in range(n) if keep[i]]
        rng = np.random.default_rng(seed * 61 + 5)
        for probe in range(3):
            query = rng.standard_normal(dim).astype(np.float32)
            page = index.search(vector=query.tolist(), top_k=top_k, filter={"g": "in"})
            assert_exact_page(
                page,
                [f"r{i}" for i in subset],
                brute_force(space, corpus[subset], query),
                top_k,
                f"scan {space} m={m} seed={seed} probe={probe}",
            )


@pytest.mark.parametrize("space", SPACES)
def test_the_scan_and_the_traversal_agree_inside_the_regime(space):
    """The same query down both paths, where both are exact.

    A filter matching every record still takes the scan, so this compares the
    two implementations against each other as well as against numpy.
    """
    n, dim, top_k = 200, 32, 10
    corpus = corpus_for(99, n, dim)
    index = build(space, dim, 16, n, corpus, metadata=[{"g": 1}] * n)
    rng = np.random.default_rng(100)
    for _ in range(4):
        query = rng.standard_normal(dim).astype(np.float32).tolist()
        walked = index.search(vector=query, top_k=top_k, ef_search=n)
        scanned = index.search(vector=query, top_k=top_k, filter={"g": 1})
        assert [r["id"] for r in walked] == [r["id"] for r in scanned], space
        for a, b in zip(walked, scanned):
            assert abs(a["score"] - b["score"]) <= tolerance(b["score"]), (a, b)


# ============================================================================
# TIES
# ============================================================================


@pytest.mark.parametrize("space", SPACES)
def test_a_tied_page_holds_every_tied_record(space):
    """Four records at each of three vectors, so every distance is a tie.

    The order within a tie is the graph's and is not asserted. What is asserted
    is that the page is a correct top-k under the tie, which is the strongest
    statement brute force supports when the ranking is not a total order.
    """
    block = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    corpus = np.repeat(block, 4, axis=0)
    index = build(space, 4, 16, len(corpus), corpus)
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    distances = brute_force(space, corpus, query)
    page = index.search(vector=query.tolist(), top_k=4, ef_search=len(corpus))
    assert {row["id"] for row in page} == {"r0", "r1", "r2", "r3"}, [
        row["id"] for row in page
    ]
    assert_exact_page(
        page, [f"r{i}" for i in range(len(corpus))], distances, 4, f"tie {space}"
    )


@pytest.mark.parametrize("space", SPACES)
def test_a_tied_page_is_the_same_page_every_build(space):
    """The tie order is unspecified but it is not arbitrary.

    Two indexes built from the same records in the same order return the same
    page, so a tie does not make a search non-deterministic.
    """
    block = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    corpus = np.repeat(block, 4, axis=0)
    query = [1.0, 0.0, 0.0, 0.0]
    pages = set()
    for _ in range(3):
        index = build(space, 4, 16, len(corpus), corpus)
        page = index.search(vector=query, top_k=len(corpus), ef_search=len(corpus))
        pages.add(tuple(row["id"] for row in page))
    assert len(pages) == 1, f"{space}: a tie made the page vary across builds: {pages}"
