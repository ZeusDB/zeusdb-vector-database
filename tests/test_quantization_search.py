"""Product quantization fetch calibration, recall floors and scoring by metric.

The half of the quantization suite that searches a trained index. The crossover
below which the defaults hold, the corpus term and the calibrated rerank fetch,
the page term, the creation warnings and the break-even, codebook
reproducibility, the scale a quantized L2 index reports, recall against an exact
ranking on every metric, and cosine scoring against the reconstruction. Its
pair, test_quantization_training.py, covers how an index is configured, trained
and stored.
"""

import time
import warnings
import pytest
import numpy as np
from helpers import repair_manifest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Tests 117 and 118: what the defaults deliver below the crossover
# ------------------------------------------------------------
CROSSOVER_DIM = 1536
CROSSOVER_RECORDS = 3000

# Ratio of quantized to unquantized median query time this fixture is allowed to
# reach. The bound is what fails a fetch default that has gone materially wrong.
# The fetch below the crossover is the floor of 250 candidates, and the next step
# up in the corpus term, being 1,000 candidates at 50,000 records, reads 2.19
# times an unquantized search on this shape of data.
#
# It is 1.9 rather than the 1.5 it was, and the reason is the runner rather than
# the fetch.
#
# Measured on a quiet machine over five repeats of the round robin below, on the
# build before the graph cutover and on the build after it.
#
#   build    raw median   quantized median   ratio range
#   before      1.267 ms           1.177 ms   0.912 to 1.021
#   after       0.688 ms           0.556 ms   0.761 to 0.975
#
# So the cutover moved the ratio down. Both arms got faster and the quantized
# arm got faster by more, at 2.12 times against 1.84, because a reranked search
# widens its traversal to the fetch and the traversal is what the cutover sped
# up. There is no product reason to raise the bound.
#
# CI reads 1.506, at 0.576 ms quantized against 0.383 ms raw. Those absolutes do
# not resemble either column above: the runner is 1.80 times faster than a quiet
# machine on the raw arm and 1.04 times slower on the quantized one, for the same
# two work shapes. The round robin interleaves the arms query by query, so a busy
# window lands on both and cannot produce that; a machine whose cores outrun its
# memory can, because the one part of the quantized arm the cutover did not touch
# is the rescoring of 250 candidates against raw vectors, which gathers 1.5 MB
# per query at this dimension. **That reading of the runner is an inference from
# its two numbers and is not measured, since these tests cannot run there.**
#
# 1.9 sits 0.39 above the reading the runner gives with a correct fetch and below
# the 2.19 a wrong one gives on a quiet machine. On the runner the separation is
# wider than that, not narrower, because a fetch four times the floor is four
# times the rescoring and the rescoring is what that machine is slow at.
CROSSOVER_MAX_TIME_RATIO = 1.9


@pytest.fixture(scope="module")
def crossover_pair():
    """One unquantized index and one quantized index over the same records.

    3,000 records of dimension 1536, which is below the crossover where a
    reranked quantized search stops being faster than an unquantized one. The
    two tests below assert the two properties the defaults are chosen to hold
    there, being recall and query time, and they share the build because
    building it twice would double what the suite pays for them.

    The dimension is high because the margin depends on it. A quantized
    traversal replaces a distance over dim floats with one over subvectors
    bytes, so the wider the vector the more it saves, and at dim 128 with 8,000
    records the reranked search is 1.39 times an unquantized one where at dim
    1536 with 3,000 it is 0.73 times.
    """
    dim, records, queries = CROSSOVER_DIM, CROSSOVER_RECORDS, 60

    rng = np.random.default_rng(20260807)
    centres = rng.standard_normal((50, dim))
    points = centres[rng.integers(0, 50, records)] + rng.standard_normal((records, dim))
    data = (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)
    ids = [f"c_{i}" for i in range(records)]

    picks = rng.choice(records, queries, replace=False)
    truth = [{ids[j] for j in row}
             for row in np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]]

    built = {}
    for label, quantization in (("raw", None),
                                ("quantized", {"type": "pq", "training_size": 1000,
                                               "storage_mode": "quantized_with_raw"})):
        kwargs = dict(dim=dim, expected_size=records)
        if quantization:
            kwargs["quantization_config"] = quantization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            index = VectorDatabase().create("hnsw", **kwargs)
        assert index.add({"ids": ids, "embeddings": data}).is_success()
        built[label] = index
    assert built["quantized"].is_quantized()

    return {"indexes": built, "queries": [data[p] for p in picks], "truth": truth}


def test_default_fetch_holds_recall_below_the_crossover(crossover_pair):
    """The defaults return almost the page an unquantized index returns.

    Below 12,500 records at top_k 10 the fetch is the floor of 250 candidates.
    What that has to reach is the group of records the codes cannot tell apart
    from the query, which on clustered data is the query's own cluster. This
    fixture puts 60 records in a cluster, so the floor covers it several times
    over, and recall measures at 1.0000 here.

    The bound is set below the level this fixture measures, because the
    codebook is trained by an unseeded k-means and a rebuild draws a different
    one, which moves a quantized recall figure by about 0.013.
    """
    pair = crossover_pair
    scores = {}
    for label, index in pair["indexes"].items():
        hits = 0
        for query, truth in zip(pair["queries"], pair["truth"]):
            hits += len({h["id"] for h in index.search(query, top_k=10)} & truth)
        scores[label] = hits / (10.0 * len(pair["queries"]))

    assert scores["raw"] > 0.99, f"the unquantized index is not the baseline: {scores}"
    assert scores["quantized"] > 0.95, (
        f"the default fetch lost recall: {scores['quantized']} against the "
        f"unquantized index's {scores['raw']}")


def test_default_quantized_search_is_not_slower_below_the_crossover(crossover_pair):
    """Below the crossover the defaults cost roughly what an unquantized index does.

    Above it they cost a multiple of it, and that is the property this test pins
    to a size. The fetch is a share of the corpus, the traversal widens to the
    fetch because HNSW cannot return more results than its candidate list holds,
    and an HNSW search costs roughly linear time in that width. So a reranked
    quantized search costs time proportional to the record count where an
    unquantized one costs time proportional to its logarithm, and the two cross
    once.

    What is asserted is the ratio of the two medians against
    CROSSOVER_MAX_TIME_RATIO, not that one is faster than the other. A quiet
    machine measures 0.761 to 0.975 here and a shared runner has read 1.506, so
    an assertion that the quantized search wins is an assertion about the runner
    rather than about the fetch. The bound is 1.9, which a fetch several times
    the floor would break and the runner does not.

    Timed round robin, one query to each index in turn, so a load spike lands on
    both rather than on whichever ran second, and compared on the median rather
    than the mean for the same reason.

    Where the two cross depends on the data as well as the record count. On
    clustered vectors of dim 768 it is between 10,000 and 15,000 records. On an
    anisotropic corpus that resembles real embeddings a reranked quantized
    search already reads 1.80 times an unquantized one at 10,000 records,
    because the unquantized search converges faster there while the fetch does
    not shrink. This fixture is clustered and sits below both.
    """
    pair = crossover_pair
    queries = pair["queries"]
    samples = {label: [] for label in pair["indexes"]}

    for index in pair["indexes"].values():   # warm both before timing
        for query in queries[:10]:
            index.search(query, top_k=10)

    for round_index in range(120):
        query = queries[round_index % len(queries)]
        for label, index in pair["indexes"].items():
            start = time.perf_counter()
            index.search(query, top_k=10)
            samples[label].append(time.perf_counter() - start)

    median = {label: sorted(values)[len(values) // 2]
              for label, values in samples.items()}
    ratio = median["quantized"] / median["raw"]
    assert ratio < CROSSOVER_MAX_TIME_RATIO, (
        f"the default quantized search costs {ratio:.3f} times an unquantized "
        f"one at {CROSSOVER_RECORDS} records, against a bound of "
        f"{CROSSOVER_MAX_TIME_RATIO}, being "
        f"{median['quantized'] * 1000:.3f} ms against "
        f"{median['raw'] * 1000:.3f} ms")


# ------------------------------------------------------------
# Tests 119 and 120: recall where the corpus term sets the fetch
# ------------------------------------------------------------
CORPUS_TERM_DIM = 256
CORPUS_TERM_RECORDS = 25000


@pytest.fixture(scope="module", params=[50, 200], ids=["coarse", "fine"])
def corpus_term_index(request):
    """A quantized index at a size where the corpus term sets the fetch.

    The floor of 250 candidates governs up to 12,500 records at top_k 10, so
    25,000 records puts the corpus term in charge at 500 candidates. The two
    parameters are two cluster structures over the same record count, because
    what a fetch has to reach is the size of the group the codes cannot
    separate and on clustered data that is the cluster. 50 clusters puts 500
    records in one and 200 clusters puts 125, and the default has to hold
    recall on both.

    Measured at dim 768 over 100 queries, the 90th percentile depth of the
    deepest true neighbour is 469 at 50 clusters and 461 at 200 clusters over
    100,000 records, which is the same 500 records to a cluster. Recall at the
    default measures 1.0000 on both.
    """
    clusters = request.param
    dim, records, queries = CORPUS_TERM_DIM, CORPUS_TERM_RECORDS, 60

    rng = np.random.default_rng(20260808)
    centres = rng.standard_normal((clusters, dim))
    points = centres[rng.integers(0, clusters, records)] + rng.standard_normal((records, dim))
    data = (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)
    ids = [f"t_{i}" for i in range(records)]

    picks = rng.choice(records, queries, replace=False)
    truth = [{ids[j] for j in row}
             for row in np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=dim, expected_size=records,
            quantization_config={"type": "pq", "training_size": 1000,
                                 "storage_mode": "quantized_with_raw"},
        )
    for start in range(0, records, 5000):
        stop = min(start + 5000, records)
        assert index.add({"ids": ids[start:stop],
                          "embeddings": data[start:stop]}).is_success()
    assert index.is_quantized()

    return {"index": index, "clusters": clusters,
            "queries": [data[p] for p in picks], "truth": truth}


def test_default_fetch_holds_recall_where_the_corpus_term_governs(corpus_term_index):
    """The defaults hold recall at a size the floor does not reach.

    Both cluster structures are covered because the required depth is set by
    the data rather than by the record count. The default fetch of 500
    candidates covers a 500 record cluster once and a 125 record cluster four
    times, so the coarse parameter is the binding one and the fine one has
    margin.

    The bound sits below the measured level because the codebook is trained by
    an unseeded k-means and a rebuild draws a different one, which moves a
    quantized recall figure by about 0.013.
    """
    case = corpus_term_index
    hits = 0
    for query, truth in zip(case["queries"], case["truth"]):
        hits += len({h["id"] for h in case["index"].search(query, top_k=10)} & truth)
    recall = hits / (10.0 * len(case["queries"]))

    assert recall > 0.95, (
        f"the default fetch lost recall at {CORPUS_TERM_RECORDS} records over "
        f"{case['clusters']} clusters: {recall}")


# ------------------------------------------------------------
# Test 118: the rerank fetch is calibrated at training completion
# ------------------------------------------------------------
CALIBRATION_DIM = 256
CALIBRATION_TRAINING = 2000
CALIBRATION_RECORDS = 8000


def _calibration_vectors(n, dim, seed):
    """Fifty Gaussian centres at sigma 1.0, then L2 normalised."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((50, dim))
    points = centres[rng.integers(0, 50, n)] + rng.standard_normal((n, dim))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def _calibration_index(storage_mode="quantized_with_raw",
                       records=CALIBRATION_RECORDS, seed=20260808):
    data = _calibration_vectors(records, CALIBRATION_DIM, seed)
    ids = [f"c_{i}" for i in range(records)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=CALIBRATION_DIM, expected_size=records,
            quantization_config={"type": "pq",
                                 "training_size": CALIBRATION_TRAINING,
                                 "storage_mode": storage_mode})
    assert index.add({"ids": ids, "embeddings": data}).is_success()
    return index, ids, data


@pytest.fixture(scope="module")
def calibrated_index():
    """One trained quantized_with_raw index, reused across the tests below."""
    index, ids, data = _calibration_index()
    assert index.is_quantized(), "training did not complete"
    return {"index": index, "ids": ids, "data": data}


def test_the_rerank_fetch_is_calibrated_at_training(calibrated_index):
    """Training measures the fetch on its own records and reports it.

    The calibration is a leave one out measurement over the training sample,
    so what it can report is bounded by that sample rather than by the corpus.
    The fetch it produces at the live record count is a separate figure and it
    is larger, because the depth grows with the record count.
    """
    stats = calibrated_index["index"].get_stats()

    assert stats["rerank_calibrated"] == "true"
    assert int(stats["rerank_calibration_records"]) == CALIBRATION_TRAINING
    assert int(stats["rerank_calibration_queries"]) > 0
    assert float(stats["rerank_calibration_target_recall"]) == pytest.approx(0.99)
    assert int(stats["rerank_calibration_ms"]) >= 0

    measured = int(stats["rerank_calibration_fetch"])
    assert 1 <= measured <= CALIBRATION_TRAINING, (
        f"a fetch of {measured} cannot come from {CALIBRATION_TRAINING} records")

    # On fifty clusters over 2,000 training records a cluster holds 40, so the
    # measured depth is a small share of the sample rather than most of it.
    assert measured < CALIBRATION_TRAINING // 2

    # The reported default is what a search at top_k 10 will actually fetch,
    # and it is above the measured value because there are four times as many
    # records as the calibration saw.
    assert int(stats["rerank_default_fetch"]) >= measured


def test_the_calibrated_fetch_holds_recall(calibrated_index):
    """The page the calibrated default returns is the page exact search returns.

    The bound sits below the level the calibration targets because the codebook
    is trained by an unseeded k-means and a draw moves a quantized recall figure
    by about 0.013.
    """
    case = calibrated_index
    index, ids, data = case["index"], case["ids"], case["data"]
    rng = np.random.default_rng(4242)
    picks = rng.choice(len(ids), 100, replace=False)
    truth = np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]

    hits = 0
    for row, pick in enumerate(picks):
        found = {h["id"] for h in index.search(data[pick], top_k=10)}
        hits += len(found & {ids[j] for j in truth[row]})
    recall = hits / (10.0 * len(picks))

    assert recall > 0.95, f"the calibrated default lost recall: {recall}"


def test_an_explicit_rerank_overrides_the_calibration(calibrated_index):
    """A named factor is a multiple of the page and the calibration is ignored.

    A factor of 1 fetches ten candidates at top_k 10, which is far below the
    calibrated fetch, so the page it returns is measurably worse. Zero returns
    the ADC ordering, which is worse still.
    """
    case = calibrated_index
    index, ids, data = case["index"], case["ids"], case["data"]
    rng = np.random.default_rng(4243)
    picks = rng.choice(len(ids), 60, replace=False)
    truth = np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]

    def recall(**kwargs):
        hits = 0
        for row, pick in enumerate(picks):
            found = {h["id"] for h in index.search(data[pick], top_k=10, **kwargs)}
            hits += len(found & {ids[j] for j in truth[row]})
        return hits / (10.0 * len(picks))

    calibrated = recall()
    narrow = recall(rerank=1)
    off = recall(rerank=0)

    assert narrow < calibrated - 0.05, (
        f"rerank=1 did not override the calibration, {narrow} against {calibrated}")
    assert off < calibrated - 0.05, (
        f"rerank=0 did not turn reranking off, {off} against {calibrated}")


def test_an_untrained_index_reports_no_calibration():
    """A calibration exists only once training has produced a codebook."""
    data = _calibration_vectors(500, CALIBRATION_DIM, 20260809)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=CALIBRATION_DIM, expected_size=CALIBRATION_RECORDS,
            quantization_config={"type": "pq",
                                 "training_size": CALIBRATION_TRAINING,
                                 "storage_mode": "quantized_with_raw"})
    assert index.add({"ids": [f"u_{i}" for i in range(500)],
                      "embeddings": data}).is_success()
    assert not index.is_quantized()

    stats = index.get_stats()
    assert stats["rerank_calibrated"] == "false"
    assert "rerank_calibration_fetch" not in stats

    # The fallback is the largest of the corpus term, the floor and the page
    # term, which at 500 records is the floor.
    assert int(stats["rerank_default_fetch"]) == 250


def test_quantized_only_is_not_calibrated():
    """quantized_only never reranks, so it is not calibrated and pays nothing."""
    index, _, _ = _calibration_index(storage_mode="quantized_only",
                                     records=CALIBRATION_TRAINING + 500)
    assert index.is_quantized()

    stats = index.get_stats()
    assert stats["rerank_calibrated"] == "false"
    assert "rerank_calibration_ms" not in stats


def test_the_calibration_survives_a_save_and_load(tmp_path, calibrated_index):
    """The measurement is stored with the index rather than recomputed."""
    index = calibrated_index["index"]
    before = index.get_stats()
    path = str(tmp_path / "calibrated.zdb")
    index.save(path)

    loaded = VectorDatabase().load(path)
    after = loaded.get_stats()

    for key in ("rerank_calibrated", "rerank_calibration_fetch",
                "rerank_calibration_records", "rerank_calibration_queries",
                "rerank_calibration_target_recall"):
        assert after[key] == before[key], f"{key} did not survive the round trip"


def test_an_index_saved_without_a_calibration_loads_and_uses_the_fallback(
        tmp_path, calibrated_index):
    """A directory written before the calibration existed still opens.

    quantization.json gained one field. Removing it reproduces a directory
    written by an earlier build, which has to load, search, and take the corpus
    term the way that build did.
    """
    import json

    path = tmp_path / "legacy.zdb"
    calibrated_index["index"].save(str(path))

    quant_path = path / "quantization.json"
    payload = json.loads(quant_path.read_text())
    assert payload.pop("rerank_calibration", None) is not None, (
        "the field this test removes was not written")
    quant_path.write_text(json.dumps(payload, indent=2))
    repair_manifest(quant_path.parent, "quantization.json")

    loaded = VectorDatabase().load(str(path))
    stats = loaded.get_stats()
    assert stats["rerank_calibrated"] == "false"

    # The fallback at this record count is the floor of 250, since the corpus
    # term reaches it only at 12,500 records.
    assert int(stats["rerank_default_fetch"]) == 250

    data = calibrated_index["data"]
    ids = calibrated_index["ids"]
    page = loaded.search(data[0], top_k=5)
    assert len(page) == 5
    assert page[0]["id"] == ids[0]


def test_the_calibration_reports_the_points_it_fitted(calibrated_index):
    """The exponent is fitted over fractions of the sample, and they are shown.

    One fetch per quarter of the training sample, the last of them being the
    fetch over the whole of it, each no deeper than the records it was measured
    over. Those four numbers are what the reported exponent comes from.
    """
    stats = calibrated_index["index"].get_stats()

    fitted = [int(part) for part in stats["rerank_calibration_fit_fetches"].split(",")]
    assert len(fitted) == 4, f"expected four fitting points, got {fitted}"
    assert fitted[-1] == int(stats["rerank_calibration_fetch"])

    sample = int(stats["rerank_calibration_records"])
    for position, measured in enumerate(fitted, start=1):
        bound = sample * position // 4
        assert 1 <= measured <= bound, (
            f"a fetch of {measured} cannot come from {bound} records")

    exponent = float(stats["rerank_calibration_exponent"])
    assert 0.40 <= exponent <= 1.00, f"exponent {exponent} escaped the clamp"


# ------------------------------------------------------------
# Test 129: the calibration measures the page as well as the corpus
# ------------------------------------------------------------
def _page_recall(index, data, ids, picks, truth, page, **kwargs):
    """Mean recall at `page` over `picks`, against exact cosine neighbours."""
    hits = 0
    for row, pick in enumerate(picks):
        found = [hit["id"] for hit in index.search(data[pick], **kwargs)][:page]
        hits += len(set(found) & {ids[j] for j in truth[row, :page]})
    return hits / (page * len(picks))


def test_the_calibration_reports_the_pages_it_fitted(calibrated_index):
    """One fetch per page, and an exponent fitted through them.

    The reference page has to be one of the pages measured, since the fetch the
    search scales from is measured there and the scaling is exactly one there.
    """
    stats = calibrated_index["index"].get_stats()

    pages = [int(part) for part in stats["rerank_calibration_pages"].split(",")]
    fetches = [int(part) for part in stats["rerank_calibration_page_fetches"].split(",")]
    assert len(pages) == len(fetches) == 3, f"{pages} against {fetches}"
    assert 10 in pages, "the reference page is not among the pages measured"
    assert fetches[pages.index(10)] == int(stats["rerank_calibration_fetch"]), (
        "the fetch at the reference page is not the fetch the search scales from")

    # A deeper page needs a deeper fetch, and the sample bounds every one of them.
    assert fetches == sorted(fetches), f"the fetch fell as the page grew: {fetches}"
    sample = int(stats["rerank_calibration_records"])
    assert all(1 <= f <= sample for f in fetches), fetches

    page_exponent = float(stats["rerank_calibration_page_exponent"])
    assert 0.0 <= page_exponent <= 1.0, f"{page_exponent} escaped the clamp"

    # Sublinear. A fetch proportional to the page is what a constant multiple of
    # top_k assumes and no corpus measured here needs it.
    assert page_exponent < 1.0, (
        f"the page requirement measured linear at {page_exponent}, which no "
        "corpus measured for this change did")


# A corpus with depth at a page of 100
#
# Fifty Gaussian clusters over 8,000 records put a whole page of 100 inside one
# cluster of 160, and a fetch sized for a page of ten already covers that
# cluster, so recall at 100 reads 1.0000 either way and the corpus cannot tell
# the two behaviours apart. This one has no cluster structure and a power law
# covariance spectrum, which is the model this project uses for embedding-like
# data, and on it the hundredth true neighbour really does sit deeper than the
# tenth.
PAGE_DIM = 256
PAGE_RECORDS = 10000
PAGE_TRAINING = 2000


def _anisotropic(n, dim, seed):
    """Unit vectors with a power law covariance spectrum and no clusters."""
    rng = np.random.default_rng(seed)
    scale = np.power(np.arange(1, dim + 1, dtype=np.float64), -0.7)
    points = rng.standard_normal((n, dim)) * scale
    rotation, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    points = points @ rotation.T
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def page_index():
    """One trained quantized_with_raw index over a corpus with page depth."""
    data = _anisotropic(PAGE_RECORDS, PAGE_DIM, 20260812)
    ids = [f"p_{i}" for i in range(PAGE_RECORDS)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=PAGE_DIM, expected_size=PAGE_RECORDS,
            quantization_config={"type": "pq", "training_size": PAGE_TRAINING,
                                 "storage_mode": "quantized_with_raw"})
    assert index.add({"ids": ids, "embeddings": data}).is_success()
    assert index.is_quantized(), "training did not complete"

    rng = np.random.default_rng(5151)
    picks = rng.choice(PAGE_RECORDS, 100, replace=False)
    truth = np.argsort(-(data[picks] @ data.T), axis=1)[:, :100]
    return {"index": index, "ids": ids, "data": data, "picks": picks, "truth": truth}


def test_the_default_fetch_scales_with_the_requested_page(page_index):
    """A larger page fetches deeper, and the reference page is untouched.

    `rerank_default_fetch` is reported at the reference page, so the fetch a
    page of 100 asks for is read off the search rather than off the stats. The
    two searches below differ only in the page they request.
    """
    case = page_index
    index, data, ids = case["index"], case["data"], case["ids"]
    stats = index.get_stats()
    reference = int(stats["rerank_default_fetch"])
    page_exponent = float(stats["rerank_calibration_page_exponent"])

    # What the arithmetic says the fetch at a page of 100 is. The floor and the
    # cap are wide open at this record count and this page.
    assert reference * 10 ** page_exponent > reference, (
        "the page term did not deepen the fetch")

    # And the search really pays for it. A page of 100 with rerank named at the
    # reference fetch returns a worse page than the default does.
    default = _page_recall(index, data, ids, case["picks"], case["truth"], 100,
                           top_k=100)
    reference_only = _page_recall(index, data, ids, case["picks"], case["truth"],
                                  100, top_k=reference, rerank=1, ef_search=100)

    assert default > reference_only, (
        f"recall at 100 is {default:.4f} on the default and {reference_only:.4f} "
        "on the fetch a page of ten asks for, so the page term bought nothing")


def test_recall_at_a_hundred_clears_the_bound_the_old_fetch_missed(page_index):
    """Recall at a page of 100, against a bound the page of ten fetch fails.

    The fetch used to be measured for a page of ten and applied at every page,
    so a search at `top_k=100` paid for a hundred results with a fetch sized for
    ten. The bound below is what the page term buys and it is chosen so that the
    old behaviour, reproduced in the second arm, does not reach it.

    The second arm names `ef_search`. An unset one resolves to `max(2 * top_k,
    100)` and the crate then raises the traversal to the candidate count, so
    leaving it unset would give that arm a traversal twice as wide as the arm it
    is reproducing.
    """
    case = page_index
    index, data, ids = case["index"], case["data"], case["ids"]
    reference = int(index.get_stats()["rerank_default_fetch"])

    after = _page_recall(index, data, ids, case["picks"], case["truth"], 100,
                         top_k=100)
    before = _page_recall(index, data, ids, case["picks"], case["truth"], 100,
                          top_k=reference, rerank=1, ef_search=100)

    assert after >= 0.97, f"recall at 100 is {after:.4f} under the page term"
    assert before < 0.97, (
        f"the fetch a page of ten asks for already reached {before:.4f} at a "
        "page of 100, so this bound no longer separates the two")


def test_recall_at_ten_is_untouched_by_the_page_term(calibrated_index):
    """The reference page asks for exactly what it asked for before.

    This is the whole guarantee. The page ratio is one at the reference page
    whatever the exponent is, so the fetch, the candidate set and the page are
    identical to the ones the calibration shipped without a page term.
    """
    case = calibrated_index
    index, data, ids = case["index"], case["data"], case["ids"]
    reference = int(index.get_stats()["rerank_default_fetch"])

    rng = np.random.default_rng(5151)
    picks = rng.choice(len(ids), 100, replace=False)
    truth = np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]

    default = _page_recall(index, data, ids, picks, truth, 10, top_k=10)
    explicit = _page_recall(index, data, ids, picks, truth, 10,
                            top_k=reference, rerank=1, ef_search=100)

    # One result slot out of the 1,000 this compares is the tolerance, because
    # the second arm asks for a page of `reference` and cuts it, so a tie at
    # equal rescored distance can fall the other way. The fetch itself is
    # identical by construction and the Rust suite asserts that directly.
    assert default == pytest.approx(explicit, abs=0.002), (
        f"recall at 10 is {default:.6f} on the default and {explicit:.6f} at the "
        "same fetch named explicitly, so the default is no longer that fetch")


def test_the_page_term_survives_a_save_and_load(tmp_path, calibrated_index):
    """The page fetches and the exponent are stored, not recomputed."""
    index = calibrated_index["index"]
    before = index.get_stats()
    path = str(tmp_path / "paged.zdb")
    index.save(path)

    after = VectorDatabase().load(path).get_stats()
    for key in ("rerank_calibration_page_fetches", "rerank_calibration_pages",
                "rerank_calibration_page_exponent"):
        assert after[key] == before[key], f"{key} did not survive the round trip"


def test_an_index_calibrated_without_a_page_term_takes_the_default(
        tmp_path, calibrated_index):
    """A directory written before the page term still opens and still deepens.

    quantization.json gained two fields inside the calibration. Removing them
    reproduces a directory written by the previous build, which has to load,
    keep its record scaling, and fall back to the shipped page exponent rather
    than to no page term at all.
    """
    import json

    path = tmp_path / "no_page_term.zdb"
    calibrated_index["index"].save(str(path))

    quant_path = path / "quantization.json"
    payload = json.loads(quant_path.read_text())
    calibration = payload["rerank_calibration"]
    assert calibration.pop("page_fetches", None) is not None
    assert calibration.pop("page_exponent", None) is not None
    quant_path.write_text(json.dumps(payload, indent=2))
    repair_manifest(quant_path.parent, "quantization.json")

    loaded = VectorDatabase().load(str(path))
    stats = loaded.get_stats()

    assert stats["rerank_calibrated"] == "true"
    assert stats["rerank_calibration_fetch"] == (
        calibrated_index["index"].get_stats()["rerank_calibration_fetch"])
    assert stats["rerank_calibration_page_fetches"] == "0,0,0", (
        "a calibration that measured no pages should report none")
    assert float(stats["rerank_calibration_page_exponent"]) > 0.0, (
        "the fallback page exponent is what such an index deepens by")

    # It still searches, and the reference page is untouched.
    assert stats["rerank_default_fetch"] == (
        calibrated_index["index"].get_stats()["rerank_default_fetch"])
    page = loaded.search(calibrated_index["data"][0], top_k=100)
    assert len(page) == 100


def test_the_calibration_holds_recall_on_records_that_arrive_in_order():
    """Records grouped by cluster are the case the seeded shuffle exists for.

    Training fires on the record that reaches training_size, so an insertion
    order that groups the corpus puts a slice of it in the codebook and in the
    calibration. Here the first 2,000 of 8,000 records are twelve of the fifty
    clusters. The sample is shuffled before either reads it, so the fractions
    the exponent is fitted over are random draws over that slice rather than
    narrower slices again, and the fetch the calibration produces still holds
    recall over the whole corpus.

    The bound matches the sibling test on randomly ordered records, because a
    codebook fitted per contiguous coordinate slice depends on the per
    coordinate marginals rather than on the joint distribution.
    """
    records = CALIBRATION_RECORDS
    rng = np.random.default_rng(20260810)
    centres = rng.standard_normal((50, CALIBRATION_DIM))
    labels = np.sort(rng.integers(0, 50, records))
    points = centres[labels] + rng.standard_normal((records, CALIBRATION_DIM))
    data = (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)
    ids = [f"o_{i}" for i in range(records)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=CALIBRATION_DIM, expected_size=records,
            quantization_config={"type": "pq",
                                 "training_size": CALIBRATION_TRAINING,
                                 "storage_mode": "quantized_with_raw"})
    assert index.add({"ids": ids, "embeddings": data}).is_success()
    assert index.is_quantized(), "training did not complete"

    # The training sample really is a slice: the first CALIBRATION_TRAINING
    # records hold well under half of the clusters.
    assert len(set(labels[:CALIBRATION_TRAINING].tolist())) < 25

    stats = index.get_stats()
    assert stats["rerank_calibrated"] == "true"

    picks = rng.choice(records, 100, replace=False)
    truth = np.argsort(-(data[picks] @ data.T), axis=1)[:, :10]
    hits = 0
    for row, pick in enumerate(picks):
        found = {h["id"] for h in index.search(data[pick], top_k=10)}
        hits += len(found & {ids[j] for j in truth[row]})
    recall = hits / (10.0 * len(picks))

    assert recall > 0.95, (
        f"the calibrated default lost recall on ordered records: {recall}, "
        f"fetch {stats['rerank_default_fetch']}")


# ------------------------------------------------------------
# Test 119: the low dimension warning
# ------------------------------------------------------------
def test_no_creation_warning_claims_a_saving_for_quantized_with_raw():
    """Nothing create() says credits quantized_with_raw with saving memory.

    Two warnings used to. The low dimension warning quoted a modelled share and
    a five dimension table, all of it measured while an unquantized index held
    every vector twice, and the break even warning named a record count above
    which the mode started saving. There is one copy of a raw vector now, held
    in a store the graph is handed, so the mode adds two codes and two tables to
    what an unquantized index holds and takes nothing away. Measured resident on
    50,000 real embeddings, it holds 1.08 times an unquantized index at dim 1536
    and 1.14 times at dim 128.

    The low dimension warning is gone rather than corrected. Its gate needed the
    graph's neighbour lists in the denominator, and those measured 823 bytes a
    record on sift-128 against 1,152 on dbpedia-openai at the same record count
    and the same degree, so no constant fits them.
    """
    vdb = VectorDatabase()
    forbidden = ("less memory than an unquantized index",
                 "times an unquantized index",
                 "holds about")

    for dim in (64, 128, 256, 768, 1536):
        for declared in (20_000, 100_000):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                vdb.create("hnsw", dim=dim, expected_size=declared,
                           quantization_config={"type": "pq",
                                                "storage_mode": "quantized_with_raw"})
            messages = [str(w.message) for w in caught]
            for phrase in forbidden:
                assert not [m for m in messages if phrase in m], (
                    f"dim {dim}, expected_size {declared}: a warning still "
                    f"claims a saving, containing {phrase!r}")
            assert [m for m in messages
                    if "is the accuracy mode rather than the memory mode" in m], (
                f"dim {dim}, expected_size {declared}: the mode warning is missing")

    # The removed model is gone from the class rather than merely unused.
    assert not hasattr(vdb, "_memory_saving_share")
    assert not hasattr(vdb, "MEASURED_RECORD_OVERHEAD_BYTES")
    assert not hasattr(vdb, "QUANTIZATION_REPAYS_SAVING_SHARE")


def test_every_creation_warning_names_the_callers_own_line():
    """A UserWarning points at the create() the caller wrote.

    Every quantization warning carried stacklevel=2 and none of them reached
    the caller. The mode warning is raised inside
    `_validate_quantization_config`, so 2 named `create`'s own body, and the
    four in `_check_memory_usage` are a frame deeper still, so 2 named the line
    inside `_validate_quantization_config` that calls it. A warning attributed
    to library internals tells a caller nothing about which of their calls
    caused it. `_warn_if_selection_disabled` had this right already.
    """
    vdb = VectorDatabase()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vdb.create("hnsw", dim=64, expected_size=3000,                     # noqa: E501
                   quantization_config={"type": "pq", "training_size": 1000})
        vdb.create("hnsw", dim=64, expected_size=100,
                   quantization_config={"type": "pq", "training_size": 1000})
        vdb.create("hnsw", dim=1536, expected_size=100_000,
                   quantization_config={"type": "pq", "subvectors": 512,
                                        "training_size": 1000})
        vdb.create("hnsw", dim=1536, expected_size=100_000,
                   quantization_config={"type": "pq", "subvectors": 8,
                                        "training_size": 1000})
        vdb.create("hnsw", dim=64, expected_size=100_000,
                   quantization_config={"type": "pq", "training_size": 1000,
                                        "storage_mode": "quantized_with_raw"})
        vdb.create("hnsw", dim=64, m=16, ef_construction=20)

    fired = {str(w.message).split(":")[0].split(",")[0]: w for w in caught}
    assert len(caught) == 6, [str(w.message)[:60] for w in caught]
    for w in caught:
        assert w.filename == __file__, (
            f"a warning was attributed to {w.filename} rather than to the "
            f"caller: {str(w.message)[:80]}")
    assert fired, "no warning fired, so nothing was checked"


def test_the_break_even_counts_both_codes_and_both_tables():
    """The one surviving memory model, checked against its own arithmetic.

    It needs no fitted constant. The fixed bytes are the codebook and the
    centroid distance table, both exact from dim, subvectors and bits, and the
    per record saving is the vector less the two codes that replace it.

    It counts the codes and the tables and nothing else, so it names a record
    count at or below the true one. Measured resident against the same three
    terms, the unmodelled remainder came to 135 bytes a record on sift-128 at
    50,000 records, 327 on dbpedia-openai at 50,000 and 851 on dbpedia-openai
    at 12,500, all of them positive, so the true crossing is above this figure
    rather than below it.
    """
    vdb = VectorDatabase()

    for dim in (64, 128, 256, 768, 1536):
        subvectors = vdb._default_subvectors(dim)
        fixed = vdb._fixed_quantization_bytes(dim, subvectors, 8)
        assert fixed == 256 * dim * 4 + subvectors * (256 * 255 // 2) * 4

        expected = -(-fixed // (dim * 4 - 2 * subvectors))
        with pytest.warns(UserWarning,
                          match=rf"starts saving above {expected} records"):
            vdb.create("hnsw", dim=dim, expected_size=expected - 1,
                       quantization_config={"type": "pq", "training_size": 1000,
                                            "storage_mode": "quantized_only"})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vdb.create("hnsw", dim=dim, expected_size=expected,
                       quantization_config={"type": "pq", "training_size": 1000,
                                            "storage_mode": "quantized_only"})
        assert not [w for w in caught
                    if "unquantized index at expected_size" in str(w.message)], (
            f"dim {dim}: the warning fired at its own break even of {expected}")


# ------------------------------------------------------------
# Training reproducibility. The trainer draws under a fixed seed, the level
# generator draws under a fixed seed, and the training rebuild inserts in
# internal id order, so building twice on identical data is building the same
# index. Each check would fail on the unseeded trainer on every run.
# ------------------------------------------------------------

def _repro_corpus(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1200, 32)).astype(np.float32)


def _repro_index(data, tmp_path, name):
    """One small quantized index, trained on `data`, saved under `name`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        index = VectorDatabase().create(
            "hnsw", dim=32, space="cosine", expected_size=1200,
            quantization_config={"type": "pq", "subvectors": 4, "bits": 4,
                                 "training_size": 1000,
                                 "storage_mode": "quantized_with_raw"})
    ids = [f"r{i:05d}" for i in range(data.shape[0])]
    result = index.add({"ids": ids, "embeddings": data})
    assert result.total_inserted == data.shape[0]
    assert index.is_quantized()
    path = tmp_path / name
    index.save(str(path))
    return index, path


def test_two_trainings_produce_one_codebook(tmp_path):
    data = _repro_corpus(65)
    _, first = _repro_index(data, tmp_path, "first")
    _, second = _repro_index(data, tmp_path, "second")

    a = (first / "pq_centroids.bin").read_bytes()
    b = (second / "pq_centroids.bin").read_bytes()
    assert a == b, "two trainings of identical data wrote different codebooks"


def test_identical_data_builds_identical_search_results(tmp_path):
    data = _repro_corpus(65)
    first, _ = _repro_index(data, tmp_path, "first")
    second, _ = _repro_index(data, tmp_path, "second")

    queries = np.random.default_rng(66).standard_normal((25, 32)).astype(np.float32)
    for rerank in (None, 0):
        kwargs = {} if rerank is None else {"rerank": rerank}
        for q in queries:
            one = [(h["id"], float(h["score"]).hex())
                   for h in first.search(q, top_k=10, **kwargs)]
            two = [(h["id"], float(h["score"]).hex())
                   for h in second.search(q, top_k=10, **kwargs)]
            assert one == two, f"rerank={rerank}: results diverge"

    # The calibration is part of what training produces, so it matches too.
    first_stats = first.get_stats()
    second_stats = second.get_stats()
    for key in ("rerank_calibration_fetch", "rerank_calibration_exponent",
                "rerank_calibration_fit_fetches", "rerank_default_fetch"):
        assert first_stats[key] == second_stats[key], key


def test_different_data_trains_a_different_codebook(tmp_path):
    _, first = _repro_index(_repro_corpus(65), tmp_path, "first")
    _, second = _repro_index(_repro_corpus(66), tmp_path, "second")

    a = (first / "pq_centroids.bin").read_bytes()
    b = (second / "pq_centroids.bin").read_bytes()
    assert a != b, "a fixed seed must not mean a fixed codebook"


# ------------------------------------------------------------
# The metric a quantized index scores on, and the scale it reports
# ------------------------------------------------------------

def _pq_corpus(n=3000, dim=32, seed=1105, scale=3.0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, dim)).astype(np.float32) * scale)


def _pq_index(space, storage_mode, corpus, subvectors=8, metadatas=None):
    index = VectorDatabase().create(
        "hnsw", dim=corpus.shape[1], space=space, m=16, ef_construction=200,
        expected_size=len(corpus),
        quantization_config={"type": "pq", "subvectors": subvectors, "bits": 8,
                             "training_size": 1000, "storage_mode": storage_mode},
    )
    payload = {"ids": [f"r{i}" for i in range(len(corpus))],
               "vectors": corpus.tolist()}
    if metadatas is not None:
        payload["metadatas"] = metadatas
    index.add(payload)
    assert index.is_quantized(), "the corpus must be large enough to train"
    return index


def test_quantized_l2_reports_the_same_scale_as_raw_l2():
    """The property the square root restores.

    `DistPQ::eval` sums a table of squared L2 distances and takes no root,
    while `L2Dist` does, so one index reported two scales. A quantized score is
    a distance to the record's reconstruction rather than to the vector that was
    inserted, so it does not equal the raw score; what it must do is live on the
    same scale, which a squared quantity does not.
    """
    corpus = _pq_corpus()
    # Held out rather than a corpus member, so neither page's top score is the
    # exact zero a record matching itself produces.
    query = np.random.default_rng(77).normal(size=corpus.shape[1]).astype(np.float32) * 3.0

    raw = VectorDatabase().create("hnsw", dim=corpus.shape[1], space="l2",
                                  m=16, ef_construction=200, expected_size=len(corpus))
    raw.add({"ids": [f"r{i}" for i in range(len(corpus))], "vectors": corpus.tolist()})
    raw_page = raw.search(query.tolist(), top_k=5, ef_search=200)

    quantized = _pq_index("l2", "quantized_only", corpus)
    q_page = quantized.search(query.tolist(), top_k=5, ef_search=200)

    # Every quantized score is the rooted L2 to that record's own
    # reconstruction, which is what a raw index would report if the
    # reconstruction were the stored vector.
    got = quantized.get_records([h["id"] for h in q_page], return_vector=True)
    recon = {r["id"]: np.asarray(r["vector"], dtype=np.float32) for r in got}
    for hit in q_page:
        expected = float(np.linalg.norm(query - recon[hit["id"]]))
        assert abs(hit["score"] - expected) <= 1e-3 + 1e-4 * expected, (
            f"{hit['id']} reported {hit['score']} against a rooted L2 of {expected}"
        )

    # And the two pages are within a factor of each other rather than a square
    # apart. Before the root the quantized top score was 159.416 where the scan
    # reported 12.626 on one index.
    assert 0.2 <= q_page[0]["score"] / raw_page[0]["score"] <= 5.0


def test_one_l2_index_reports_one_scale_whatever_the_filter():
    """The traversal and the exact scan a narrow filter takes must agree.

    The scan scores with `raw_distance_fn`, which roots, and the traversal
    scores with the ADC sum, which did not. Which number a caller saw therefore
    depended on how selective their filter was.
    """
    corpus = _pq_corpus()
    metadatas = [{"tag": "rare" if i < 30 else "common"} for i in range(len(corpus))]
    index = _pq_index("l2", "quantized_only", corpus, metadatas=metadatas)
    query = corpus[0]

    plain = index.search(query.tolist(), top_k=5, ef_search=200)
    narrow = index.search(query.tolist(), filter={"tag": "rare"}, top_k=5, ef_search=200)

    # r0 is in the corpus and carries the rare tag, so both paths return it.
    plain_r0 = next(h["score"] for h in plain if h["id"] == "r0")
    narrow_r0 = next(h["score"] for h in narrow if h["id"] == "r0")
    assert abs(plain_r0 - narrow_r0) <= 1e-3 + 1e-4 * abs(narrow_r0), (
        f"traversal reported {plain_r0} and the exact scan {narrow_r0}"
    )


def test_the_root_does_not_move_the_quantized_l2_ordering():
    """A square root is monotone, so the page it reorders is no page at all."""
    corpus = _pq_corpus()
    index = _pq_index("l2", "quantized_only", corpus)
    for qi in (0, 5, 19):
        page = index.search(corpus[qi].tolist(), top_k=10, ef_search=200)
        scores = [h["score"] for h in page]
        assert scores == sorted(scores)
        assert all(s >= 0.0 for s in scores)


def test_an_unquantized_index_is_untouched_by_the_root():
    """Nothing here may move a raw page."""
    corpus = _pq_corpus(n=500)
    query = corpus[3]
    for space in ("cosine", "l2", "l1", "dot"):
        index = VectorDatabase().create("hnsw", dim=corpus.shape[1], space=space,
                                        m=16, ef_construction=200, expected_size=500)
        index.add({"ids": [f"r{i}" for i in range(len(corpus))],
                   "vectors": corpus.tolist()})
        page = index.search(query.tolist(), top_k=5, ef_search=200)
        if space == "l2":
            truth = np.linalg.norm(corpus - query, axis=1)
        elif space == "l1":
            truth = np.abs(corpus - query).sum(axis=1)
        elif space == "dot":
            truth = 1.0 - corpus.astype(np.float64) @ query.astype(np.float64)
        else:
            cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
            truth = 1.0 - cn @ (query / np.linalg.norm(query))
        for hit in page:
            expected = float(truth[int(hit["id"][1:])])
            assert abs(hit["score"] - expected) <= 1e-3 + 1e-4 * abs(expected), (
                f"{space} moved: {hit['id']} reported {hit['score']} against {expected}"
            )


# ------------------------------------------------------------
# Recall floors against exact rankings
# ------------------------------------------------------------

_FLOOR_N, _FLOOR_D, _FLOOR_NQ, _FLOOR_K = 3000, 32, 40, 10


def _floor_corpus():
    rng = np.random.default_rng(1105)
    return rng.normal(size=(_FLOOR_N, _FLOOR_D)).astype(np.float32) * 3.0


def _exact_top_k(corpus, queries, space, k):
    if space == "l2":
        dd = np.linalg.norm(corpus[None, :, :] - queries[:, None, :], axis=2)
    elif space == "l1":
        dd = np.abs(corpus[None, :, :] - queries[:, None, :]).sum(axis=2)
    elif space == "dot":
        dd = 1.0 - queries.astype(np.float64) @ corpus.astype(np.float64).T
    else:
        cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        dd = 1.0 - qn @ cn.T
    return [set(row.tolist()) for row in np.argsort(dd, axis=1)[:, :k]]


def _measure_recall(index, corpus, queries, truth, k, rerank=None):
    hit = 0
    for qi in range(len(queries)):
        kwargs = dict(vector=queries[qi].tolist(), top_k=k, ef_search=200)
        if rerank is not None:
            kwargs["rerank"] = rerank
        got = {int(h["id"][1:]) for h in index.search(**kwargs)}
        hit += len(got & truth[qi])
    return hit / (len(queries) * k)


# space, storage mode, rerank, floor. The floors are what this configuration
# measured on this corpus, less a margin, and not a figure anyone hoped for.
# Corpus: 3,000 Gaussian records of dimension 32, subvectors 8, bits 8,
# training_size 1,000, 40 queries drawn from the corpus, ef_search 200,
# top_k 10. Measured: cosine 1.0000 / 0.6425 / 1.0000 / 0.6425, l2 1.0000 /
# 0.5800 / 1.0000 / 0.5800, l1 raw 1.0000, dot raw 1.0000.
#
# The two quantized cosine rows measured 0.6175 while the scorer ranked by
# squared L2 to the reconstruction and 0.6425 since it ranks by cosine. The
# floors below are not set between those two, because a floor that tight would
# fail on float differences that are not a regression. What catches a return to
# the old ordering is
# `test_quantized_cosine_reports_the_cosine_distance_to_the_reconstruction`
# and `test_quantized_cosine_ranks_by_cosine_and_not_by_squared_l2`, which pin
# the scorer against the reconstructions themselves. These catch a collapse.
_RECALL_FLOORS = [
    ("cosine", None, None, 0.99),
    ("cosine", "quantized_only", None, 0.58),
    ("cosine", "quantized_with_raw", None, 0.99),
    ("cosine", "quantized_with_raw", 0, 0.58),
    ("l2", None, None, 0.99),
    ("l2", "quantized_only", None, 0.52),
    ("l2", "quantized_with_raw", None, 0.99),
    ("l2", "quantized_with_raw", 0, 0.52),
    ("l1", None, None, 0.99),
    ("dot", None, None, 0.99),
]


@pytest.mark.parametrize("space,storage_mode,rerank,floor", _RECALL_FLOORS)
def test_recall_against_an_exact_ranking(space, storage_mode, rerank, floor):
    """Every metric and storage mode that survives, held to what it measured.

    `l1` appears unquantized only, because `create()` refuses the quantized
    pair. A quantized graph ranks by squared L2 whatever the space, and squared
    L2 does not order the same points L1 does.
    """
    corpus = _floor_corpus()
    queries = corpus[:_FLOOR_NQ]
    truth = _exact_top_k(corpus, queries, space, _FLOOR_K)

    kwargs = dict(index_type="hnsw", dim=_FLOOR_D, space=space, m=16,
                  ef_construction=200, expected_size=_FLOOR_N)
    if storage_mode:
        kwargs["quantization_config"] = {
            "type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000,
            "storage_mode": storage_mode,
        }
    index = VectorDatabase().create(**kwargs)
    index.add({"ids": [f"r{i}" for i in range(_FLOOR_N)], "vectors": corpus.tolist()})
    if storage_mode:
        assert index.is_quantized()

    recall = _measure_recall(index, corpus, queries, truth, _FLOOR_K, rerank)
    assert recall >= floor, (
        f"{space} {storage_mode} rerank={rerank} recall@{_FLOOR_K} "
        f"{recall:.4f} below the floor {floor}"
    )


def test_candidate_recall_is_measured_before_rerank():
    """Rerank must not be able to hide a broken candidate generator.

    Rescoring against raw vectors is exact, so a reranked page can read well
    while the graph underneath it returns the wrong candidates. `rerank=0`
    reports the page the traversal actually produced, and that is the number
    held here. It has to be well below the reranked figure, otherwise this test
    is measuring the rescoring it exists to see past.
    """
    corpus = _floor_corpus()
    queries = corpus[:_FLOOR_NQ]
    truth = _exact_top_k(corpus, queries, "l2", _FLOOR_K)

    index = VectorDatabase().create(
        "hnsw", dim=_FLOOR_D, space="l2", m=16, ef_construction=200,
        expected_size=_FLOOR_N,
        quantization_config={"type": "pq", "subvectors": 8, "bits": 8,
                             "training_size": 1000,
                             "storage_mode": "quantized_with_raw"},
    )
    index.add({"ids": [f"r{i}" for i in range(_FLOOR_N)], "vectors": corpus.tolist()})
    assert index.is_quantized()

    candidates = _measure_recall(index, corpus, queries, truth, _FLOOR_K, rerank=0)
    reranked = _measure_recall(index, corpus, queries, truth, _FLOOR_K)

    # Measured 0.5800 and 1.0000 on this corpus.
    assert candidates >= 0.52, f"candidate recall {candidates:.4f} below its floor"
    assert reranked >= 0.99, f"reranked recall {reranked:.4f} below its floor"
    assert reranked - candidates >= 0.15, (
        "rerank is not moving the page, so this is not measuring candidates"
    )


def test_quantized_only_ignores_rerank_entirely():
    """It holds no raw vectors, so there is nothing exact to rescore against.

    Pinned because the blast radius of a wrong candidate ordering depends on it:
    a mode that always reranks is exposed only through `rerank=0`, and this one
    is exposed always.
    """
    corpus = _floor_corpus()
    queries = corpus[:_FLOOR_NQ]
    truth = _exact_top_k(corpus, queries, "l2", _FLOOR_K)
    index = _pq_index("l2", "quantized_only", corpus)

    assert _measure_recall(index, corpus, queries, truth, _FLOOR_K) == pytest.approx(
        _measure_recall(index, corpus, queries, truth, _FLOOR_K, rerank=0)
    )


# ------------------------------------------------------------
# The metric a quantized cosine index scores on
# ------------------------------------------------------------


def _reconstructions(index, ids):
    """What a `quantized_only` index holds for each record.

    `get_records(return_vector=True)` returns the reconstruction where no raw
    vector survives, which is the vector both scorers see.
    """
    out = {}
    for r in index.get_records(ids, return_vector=True):
        out[r["id"]] = np.asarray(r["vector"], dtype=np.float64)
    return out


def test_quantized_cosine_reports_the_cosine_distance_to_the_reconstruction():
    """The score is on the scale a raw cosine index reports.

    A quantized graph sums a table of squared L2 distances, and under cosine
    that is `1 + norm(c)^2 - 2 dot(q, c)` against a reconstruction whose norm is
    not one. It used to be reported as it stood, which ran at about 1.86 times
    the cosine distance on 25,000 dbpedia records and could not be converted by
    the caller, because the conversion needs that record's own `norm(c)`.

    `DistPQ` under `PqMetric::Cosine` does the conversion inside the scorer,
    where `norm(c)` is available as a sum over the codes.
    """
    corpus = _pq_corpus(n=3000, dim=32)
    query = np.random.default_rng(78).normal(size=corpus.shape[1]).astype(np.float32)
    index = _pq_index("cosine", "quantized_only", corpus)

    page = index.search(query.tolist(), top_k=8, ef_search=200)
    recon = _reconstructions(index, [h["id"] for h in page])
    qn = query.astype(np.float64) / np.linalg.norm(query)

    for hit in page:
        c = recon[hit["id"]]
        expected = 1.0 - float(qn @ c) / float(np.linalg.norm(c))
        assert abs(hit["score"] - expected) <= 2e-4 + 1e-4 * abs(expected), (
            f"{hit['id']} reported {hit['score']} against a cosine distance of {expected}"
        )


def test_quantized_cosine_ranks_by_cosine_and_not_by_squared_l2():
    """The two orderings differ, and the page is the cosine one.

    Squared L2 to the reconstruction carries `norm(c)^2`, which varies from
    record to record, so it is not a monotone function of the cosine distance
    and the two do not rank the same records the same way. That is why the score
    could not be converted on the page and the scorer had to move.
    """
    corpus = _pq_corpus(n=3000, dim=32)
    index = _pq_index("cosine", "quantized_only", corpus)
    ids = [f"r{i}" for i in range(len(corpus))]
    recon = _reconstructions(index, ids)
    stack = np.stack([recon[i] for i in ids])
    norms = np.linalg.norm(stack, axis=1)

    rng = np.random.default_rng(79)
    disagreements = 0
    for _ in range(25):
        q = rng.normal(size=corpus.shape[1])
        q = q / np.linalg.norm(q)
        sim = stack @ q
        by_cosine = np.argsort(-(sim / norms))[:10]
        by_sq_l2 = np.argsort(norms**2 - 2.0 * sim)[:10]
        page = [int(h["id"][1:]) for h in
                index.search(q.astype(np.float32).tolist(), top_k=10, ef_search=800)]
        if list(by_cosine) != list(by_sq_l2):
            disagreements += 1
        assert page == list(by_cosine), (
            f"page {page} is not the exhaustive cosine order {list(by_cosine)}"
        )
    assert disagreements > 0, (
        "the two orderings never disagreed on this corpus, so this test proves nothing"
    )


def test_one_cosine_index_reports_one_scale_whatever_the_filter():
    """The traversal and the exact scan a narrow filter takes must agree.

    The scan scores a reconstruction with the space's raw distance, and
    `CosineDist` is `1 - dot` on a pair assumed to be unit. A reconstruction is
    not unit, so the scan answered a third quantity, neither the squared L2 the
    traversal returned nor the cosine distance either of them should have. On
    25,000 dbpedia records one record read 0.19678 from the traversal and
    0.19118 from the scan against a true cosine distance of 0.10381, and the
    gap between the two paths ran with the distance rather than being constant.
    """
    corpus = _pq_corpus(n=3000, dim=32)
    metadatas = [{"tag": "rare" if i < 30 else "common"} for i in range(len(corpus))]
    index = _pq_index("cosine", "quantized_only", corpus, metadatas=metadatas)

    checked = 0
    for qi in (0, 7, 13):
        query = corpus[qi].tolist()
        plain = {h["id"]: h["score"] for h in index.search(query, top_k=30, ef_search=400)}
        narrow = {h["id"]: h["score"] for h in
                  index.search(query, filter={"tag": "rare"}, top_k=30, ef_search=400)}
        for ext_id, scanned in narrow.items():
            if ext_id not in plain:
                continue
            checked += 1
            assert abs(plain[ext_id] - scanned) <= 1e-4 + 1e-4 * abs(scanned), (
                f"{ext_id}: traversal reported {plain[ext_id]} and the scan {scanned}"
            )
    assert checked >= 3, "no record was returned by both paths, so nothing was compared"


def test_a_quantized_cosine_score_is_comparable_with_a_raw_cosine_score():
    """Two indexes over one corpus, one raw and one quantized, one scale.

    Not equal, because a quantized index scores against the reconstruction and
    a raw one against the vector it was given. On one scale, because the
    difference between them is the quantization error rather than a factor of
    two.
    """
    corpus = _pq_corpus(n=3000, dim=32)
    query = np.random.default_rng(80).normal(size=corpus.shape[1]).astype(np.float32)

    raw = VectorDatabase().create("hnsw", dim=corpus.shape[1], space="cosine",
                                  m=16, ef_construction=200, expected_size=len(corpus))
    raw.add({"ids": [f"r{i}" for i in range(len(corpus))], "vectors": corpus.tolist()})
    raw_scores = {h["id"]: h["score"] for h in raw.search(query.tolist(), top_k=20,
                                                          ef_search=400)}

    quantized = _pq_index("cosine", "quantized_only", corpus)
    q_page = quantized.search(query.tolist(), top_k=20, ef_search=400)

    shared = [h for h in q_page if h["id"] in raw_scores]
    assert len(shared) >= 5, "the two pages barely overlap, so there is nothing to compare"
    for hit in shared:
        gap = abs(hit["score"] - raw_scores[hit["id"]])
        assert gap <= 0.25, (
            f"{hit['id']}: quantized {hit['score']} against raw "
            f"{raw_scores[hit['id']]}, a gap of {gap}"
        )


@pytest.mark.parametrize("storage_mode", ["quantized_only", "quantized_with_raw"])
def test_a_quantized_cosine_page_survives_a_save_and_load(storage_mode, tmp_path):
    """The scorer follows the graph through a directory.

    The metric is chosen from the space at `create()` and from the dump's own
    discriminant at load, so a saved directory scores the way the graph inside
    it was wired. Nothing about the format moved, so the codebook, the codes and
    the graph are the same bytes as before.
    """
    corpus = _pq_corpus(n=3000, dim=32)
    query = np.random.default_rng(81).normal(size=corpus.shape[1]).astype(np.float32)
    index = _pq_index("cosine", storage_mode, corpus)
    before = [(h["id"], h["score"]) for h in
              index.search(query.tolist(), top_k=10, ef_search=200)]

    path = tmp_path / f"cosine-{storage_mode}.zdb"
    index.save(str(path))
    loaded = VectorDatabase().load(str(path))
    after = [(h["id"], h["score"]) for h in
             loaded.search(query.tolist(), top_k=10, ef_search=200)]

    assert [a for a, _ in before] == [a for a, _ in after]
    for (_, b), (_, a) in zip(before, after):
        assert abs(b - a) <= 1e-5
