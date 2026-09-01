//! The brute-force verifier.
//!
//! Every page the index returns is compared with one computed by scoring
//! every admitted live record with the same merge and keeping the best `k` by
//! score and then by lower id. The two accumulate in the same order, so the
//! comparison is on exact equality of ids and score bits.
//!
//! What is compared. Every search loop, three admit shapes, being the live
//! set as a bitmap, a filter bitmap admitting a tenth, and a sorted candidate
//! list of fifty ids standing for a chained arm. Then a fifth of the records
//! removed under each of the three policies, with the admit sets rebuilt to
//! exclude them and once left as they were, then `vector` against the corpus
//! for every id, then a save and a restore, then a compaction.
//!
//! The term frequency weighting has a verifier of its own, `verify_bm25`,
//! whose brute force counts the document frequencies over the admitted
//! records itself and applies the formula from its definition, under both
//! corpus scopes, through the same states.

use std::collections::HashMap;

use zeusdb_vector_core::{
    Admit, ArtefactRecord, Bitmap, Bounds, Budget, Candidates, Hit, Inventory, Ledger, Persist,
    Prepared, RecordId, Restore, SparseRef, VectorIndex,
};

use crate::corpus::{self, Corpus, Rng};
use crate::index::{PostingsIndex, SparseConfig, Unlink, Weighting};
use crate::search::Mode;
use zeusdb_vector_core::IdfScope;

const MODES: [Mode; 6] = [
    Mode::Auto,
    Mode::PerPosting,
    Mode::PerCandidate,
    Mode::BitmapPerPosting,
    Mode::Enumerate,
    Mode::Floor,
];

fn build(corpus: &Corpus, unlink: Unlink) -> PostingsIndex {
    build_weighted(corpus, unlink, Weighting::Dot)
}

fn build_weighted(corpus: &Corpus, unlink: Unlink, weighting: Weighting) -> PostingsIndex {
    let mut index = PostingsIndex::new(SparseConfig {
        unlink,
        weighting,
        ..SparseConfig::default()
    });
    for (i, d) in corpus.docs.iter().enumerate() {
        // Internal ids start at 1, as the collection's counter does.
        index
            .insert(RecordId(i as u32 + 1), d.as_ref(), Prepared::none())
            .unwrap();
    }
    index
}

fn live_bitmap(n: usize, dead: &Bitmap) -> Bitmap {
    let mut b = Bitmap::default();
    for id in 1..=n {
        if !dead.contains(id) {
            b.insert(id);
        }
    }
    b
}

/// A filter bitmap admitting `permille` per thousand of the ids, drawn by a
/// seeded generator so every arm sees the same set.
fn filter_bitmap(n: usize, permille: usize, seed: u64, dead: &Bitmap) -> Bitmap {
    let mut rng = Rng::new(seed);
    let mut b = Bitmap::default();
    for id in 1..=n {
        if rng.below(1000) < permille && !dead.contains(id) {
            b.insert(id);
        }
    }
    b
}

/// Every admitted live record with a nonzero dot product, best first, ties
/// by lower id.
fn brute(
    corpus: &Corpus,
    dead: &Bitmap,
    admit: &dyn Admit,
    q: SparseRef<'_>,
    k: usize,
) -> Vec<Hit> {
    let mut cands: Vec<(f32, u32)> = Vec::new();
    for (i, d) in corpus.docs.iter().enumerate() {
        let id = RecordId(i as u32 + 1);
        if dead.contains(id.slot()) || !admit.admits(id) {
            continue;
        }
        let s = d.as_ref().dot(q);
        if s != 0.0 {
            cands.push((s, id.0));
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    cands.truncate(k);
    cands
        .into_iter()
        .map(|(s, id)| Hit {
            id: RecordId(id),
            score: s,
        })
        .collect()
}

fn same(a: &[Hit], b: &[Hit]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.id == y.id && x.score.to_bits() == y.score.to_bits())
}

#[derive(Default)]
struct Manifest(HashMap<String, ArtefactRecord>);

impl Ledger for Manifest {
    fn record(&mut self, name: &str, record: ArtefactRecord) {
        self.0.insert(name.to_string(), record);
    }
}

impl Inventory for Manifest {
    fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
        self.0.get(name).copied()
    }
}

struct Tally {
    pages: usize,
    differ: usize,
}

/// Compare every loop under every admit shape against brute force, for one
/// index state. `Floor` is compared only where nothing is dead and the admit
/// set is the live set, since it applies no predicate.
fn check(
    tally: &mut Tally,
    label: &str,
    corpus: &Corpus,
    index: &PostingsIndex,
    dead: &Bitmap,
    admits: &[(&str, &dyn Admit)],
    k: usize,
) {
    for (aname, admit) in admits {
        for mode in MODES {
            if mode == Mode::Floor && (*aname != "live" || dead.count() > 0) {
                continue;
            }
            for q in &corpus.queries {
                let expect = brute(corpus, dead, *admit, q.as_ref(), k);
                let got = index
                    .search_mode(mode, q.as_ref(), k, *admit, false)
                    .unwrap();
                assert!(got.exact);
                tally.pages += 1;
                if !same(&expect, &got.items) {
                    tally.differ += 1;
                    if tally.differ == 1 {
                        eprintln!(
                            "first mismatch {label} {aname} {mode:?}: expect {:?} got {:?}",
                            &expect[..expect.len().min(3)],
                            &got.items[..got.items.len().min(3)]
                        );
                    }
                }
            }
        }
    }
    // `vector` answers for live records and not for dead ones.
    for (i, d) in corpus.docs.iter().enumerate() {
        let id = RecordId(i as u32 + 1);
        match index.vector(id) {
            Some(v) if !dead.contains(id.slot()) => {
                assert!(
                    v.dims == &d.dims[..] && v.values == &d.values[..],
                    "{label} vector {}",
                    id.0
                );
            }
            None if dead.contains(id.slot()) => {}
            other => panic!(
                "{label} vector {} answered {:?}",
                id.0,
                other.map(|v| v.dims.len())
            ),
        }
    }
}

/// The whole verification over one regime at one size.
pub(crate) fn verify(regime: &str, n: usize, nq: usize) -> (usize, usize) {
    let corpus = corpus::corpus(regime, n, nq);
    let k = 10;
    let mut tally = Tally {
        pages: 0,
        differ: 0,
    };
    let none = Bitmap::default();
    let live = live_bitmap(n, &none);
    let narrow = filter_bitmap(n, 100, 7, &none);
    let mut sorted_ids: Vec<RecordId> =
        (0..50).map(|i| RecordId((i * 97 % n + 1) as u32)).collect();
    sorted_ids.sort();
    sorted_ids.dedup();
    let chained = Candidates::Sorted(sorted_ids.clone());
    let admits: Vec<(&str, &dyn Admit)> = vec![
        ("live", &live),
        ("filter_10pct", &narrow),
        ("chained_50", &chained),
    ];

    let mut lazy = build(&corpus, Unlink::Lazy);
    check(&mut tally, "fresh_lazy", &corpus, &lazy, &none, &admits, k);

    // The trait's own search, through a trait object, as the collection
    // holds the index.
    {
        let object: &dyn VectorIndex<zeusdb_vector_core::Sparse> = &lazy;
        for q in &corpus.queries {
            let hits = object
                .search(q.as_ref(), k, &Candidates::All, &Budget::default())
                .unwrap();
            tally.pages += 1;
            if !same(&hits.items, &brute(&corpus, &none, &live, q.as_ref(), k)) {
                tally.differ += 1;
            }
            let cost = object.cost(q.as_ref(), k, None);
            assert!(cost.exact && cost.work_ns >= 0.0);
        }
        assert_eq!(object.len(), n);
    }

    // Remove a fifth under each policy and check again, first with the admit
    // sets rebuilt to exclude the dead as the collection's would be, then
    // with the sets left as they were, so the index's own dead test is
    // exercised.
    let mut rng = Rng::new(99);
    let mut dead = Bitmap::default();
    let mut doomed = Vec::new();
    for id in 1..=n {
        if rng.below(100) < 20 {
            dead.insert(id);
            doomed.push(RecordId(id as u32));
        }
    }
    let mut strand = build(&corpus, Unlink::Strand);
    let mut eager = build(&corpus, Unlink::Eager);
    for id in &doomed {
        lazy.remove(*id).unwrap();
        strand.remove(*id).unwrap();
        eager.remove(*id).unwrap();
    }
    assert_eq!(lazy.len(), n - doomed.len());
    assert_eq!(
        strand.stranded(),
        doomed
            .iter()
            .map(|id| corpus.docs[id.slot() - 1].dims.len())
            .sum::<usize>()
    );
    assert_eq!(eager.stranded(), 0);
    assert!(lazy.stranded() < strand.stranded());

    let live_after = live_bitmap(n, &dead);
    let narrow_after = filter_bitmap(n, 100, 7, &dead);
    let chained_after = Candidates::Sorted(
        sorted_ids
            .iter()
            .copied()
            .filter(|id| !dead.contains(id.slot()))
            .collect(),
    );
    let admits_after: Vec<(&str, &dyn Admit)> = vec![
        ("live", &live_after),
        ("filter_10pct", &narrow_after),
        ("chained_50", &chained_after),
    ];
    let admits_stale: Vec<(&str, &dyn Admit)> = vec![
        ("live_stale", &live),
        ("filter_10pct_stale", &narrow),
        ("chained_50_stale", &chained),
    ];
    for (label, index) in [
        ("removed_lazy", &lazy),
        ("removed_strand", &strand),
        ("removed_eager", &eager),
    ] {
        check(&mut tally, label, &corpus, index, &dead, &admits_after, k);
        check(&mut tally, label, &corpus, index, &dead, &admits_stale, k);
    }

    // Persist and restore, then the same pages.
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::default();
    lazy.write("sparse.", dir.path(), &mut manifest).unwrap();
    let bounds = Bounds {
        min_records: 0,
        max_records: n,
        max_bytes: 1 << 34,
    };
    let restored =
        PostingsIndex::restore(lazy.config(), "sparse.", dir.path(), &manifest, &bounds).unwrap();
    assert_eq!(restored.len(), lazy.len());
    assert_eq!(restored.stranded(), 0);
    check(
        &mut tally,
        "restored_lazy",
        &corpus,
        &restored,
        &dead,
        &admits_after,
        k,
    );

    // Compaction keeps the pages and reclaims everything.
    strand.compact();
    lazy.compact();
    assert_eq!(strand.stranded(), 0);
    assert_eq!(lazy.stranded(), 0);
    assert_eq!(strand.postings_total(), eager.postings_total());
    check(
        &mut tally,
        "compacted_strand",
        &corpus,
        &strand,
        &dead,
        &admits_after,
        k,
    );
    check(
        &mut tally,
        "compacted_lazy",
        &corpus,
        &lazy,
        &dead,
        &admits_stale,
        k,
    );

    (tally.pages, tally.differ)
}

/// The term frequency weighting by brute force, written from the formula
/// rather than from the index's scorer. The document frequencies are counted
/// over the admitted live records by a walk of the corpus, or over every
/// live record under the global scope, and each admitted live record is
/// scored by a merge in ascending dimension order with the same single
/// precision arithmetic the scan applies, so the comparison is on exact
/// equality. The mean length is the index's own, checked separately against
/// a fresh sum.
#[allow(clippy::too_many_arguments)]
fn brute_bm25(
    corpus: &Corpus,
    dead: &Bitmap,
    admit: &dyn Admit,
    scope: IdfScope,
    q: SparseRef<'_>,
    k: usize,
    k1: f32,
    b: f32,
    mean: f64,
) -> Vec<Hit> {
    let admitted = |i: usize| {
        let id = RecordId(i as u32 + 1);
        !dead.contains(id.slot()) && admit.admits(id)
    };
    let live = |i: usize| !dead.contains(i + 1);
    let counted: &dyn Fn(usize) -> bool = match scope {
        IdfScope::Corpus => &admitted,
        IdfScope::Global => &live,
    };
    let mut documents = 0usize;
    let mut df = vec![0usize; q.dims.len()];
    for (i, d) in corpus.docs.iter().enumerate() {
        if !counted(i) {
            continue;
        }
        documents += 1;
        for (j, dim) in q.dims.iter().enumerate() {
            if d.dims.binary_search(dim).is_ok() {
                df[j] += 1;
            }
        }
    }
    let n = documents as f64;
    let weights: Vec<f32> = q
        .values
        .iter()
        .zip(&df)
        .map(|(&qw, &df)| {
            if df == 0 {
                0.0
            } else {
                let df = df as f64;
                (qw as f64 * (1.0 + (n - df + 0.5) / (df + 0.5)).ln() * (k1 as f64 + 1.0)) as f32
            }
        })
        .collect();
    let c0 = k1 * (1.0 - b);
    let c1 = (k1 as f64 * b as f64 / if mean > 0.0 { mean } else { 1.0 }) as f32;
    let mut cands: Vec<(f32, u32)> = Vec::new();
    for (i, d) in corpus.docs.iter().enumerate() {
        if !admitted(i) {
            continue;
        }
        let len = d.values.iter().map(|&v| v as f64).sum::<f64>() as f32;
        let mut s = 0f32;
        for (j, dim) in q.dims.iter().enumerate() {
            if let Ok(at) = d.dims.binary_search(dim) {
                let tf = d.values[at];
                s += weights[j] * tf / (tf + c0 + c1 * len);
            }
        }
        if s != 0.0 {
            cands.push((s, i as u32 + 1));
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    cands.truncate(k);
    cands
        .into_iter()
        .map(|(s, id)| Hit {
            id: RecordId(id),
            score: s,
        })
        .collect()
}

/// A fresh mean length from the corpus, for the check of the running sum.
fn fresh_mean(corpus: &Corpus, dead: &Bitmap) -> f64 {
    let (mut total, mut live) = (0f64, 0usize);
    for (i, d) in corpus.docs.iter().enumerate() {
        if dead.contains(i + 1) {
            continue;
        }
        total += d.values.iter().map(|&v| v as f64).sum::<f64>() as f32 as f64;
        live += 1;
    }
    if live == 0 {
        0.0
    } else {
        total / live as f64
    }
}

/// Every loop under every admit shape and both scopes against the brute
/// force weighting, for one index state.
#[allow(clippy::too_many_arguments)]
fn check_bm25(
    tally: &mut Tally,
    label: &str,
    corpus: &Corpus,
    index: &PostingsIndex,
    dead: &Bitmap,
    admits: &[(&str, &dyn Admit)],
    k: usize,
    k1: f32,
    b: f32,
) {
    let mean = index.mean_length();
    let fresh = fresh_mean(corpus, dead);
    assert!(
        (mean - fresh).abs() <= 1e-9 * fresh.max(1.0),
        "{label}: the running length total drifted, {mean} against {fresh}"
    );
    for (aname, admit) in admits {
        for scope in [IdfScope::Corpus, IdfScope::Global] {
            for mode in MODES {
                if mode == Mode::Floor && (*aname != "live" || dead.count() > 0) {
                    continue;
                }
                for q in &corpus.queries {
                    let expect =
                        brute_bm25(corpus, dead, *admit, scope, q.as_ref(), k, k1, b, mean);
                    let got = index
                        .search_scoped(mode, q.as_ref(), k, *admit, false, scope)
                        .unwrap();
                    assert!(got.exact);
                    tally.pages += 1;
                    if !same(&expect, &got.items) {
                        tally.differ += 1;
                        if tally.differ == 1 {
                            eprintln!(
                                "first mismatch {label} {aname} {scope:?} {mode:?}: expect {:?} got {:?}",
                                &expect[..expect.len().min(3)],
                                &got.items[..got.items.len().min(3)]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The whole verification of the term frequency weighting over one regime
/// at one size: fresh, a fifth removed under the lazy policy with the admit
/// sets rebuilt and left stale, restored, and compacted.
pub(crate) fn verify_bm25(regime: &str, n: usize, nq: usize) -> (usize, usize) {
    let corpus = corpus::corpus(regime, n, nq);
    let (k, k1, b) = (10, 1.2f32, 0.75f32);
    let mut tally = Tally {
        pages: 0,
        differ: 0,
    };
    let none = Bitmap::default();
    let live = live_bitmap(n, &none);
    let narrow = filter_bitmap(n, 100, 7, &none);
    let mut sorted_ids: Vec<RecordId> =
        (0..50).map(|i| RecordId((i * 97 % n + 1) as u32)).collect();
    sorted_ids.sort();
    sorted_ids.dedup();
    let chained = Candidates::Sorted(sorted_ids.clone());
    let admits: Vec<(&str, &dyn Admit)> = vec![
        ("live", &live),
        ("filter_10pct", &narrow),
        ("chained_50", &chained),
    ];
    let mut lazy = build_weighted(&corpus, Unlink::Lazy, Weighting::Bm25 { k1, b });
    check_bm25(
        &mut tally, "fresh", &corpus, &lazy, &none, &admits, k, k1, b,
    );

    // The trait's own search under a set admitting everything is the
    // global weighting over the live set, and the trait's statistics under
    // the filter count what the filter admits.
    {
        let object: &dyn VectorIndex<zeusdb_vector_core::Sparse> = &lazy;
        let mean = lazy.mean_length();
        for q in &corpus.queries {
            let hits = object
                .search(q.as_ref(), k, &Candidates::All, &Budget::default())
                .unwrap();
            tally.pages += 1;
            let expect = brute_bm25(
                &corpus,
                &none,
                &live,
                IdfScope::Global,
                q.as_ref(),
                k,
                k1,
                b,
                mean,
            );
            if !same(&hits.items, &expect) {
                tally.differ += 1;
            }
            let stats = object.corpus_stats(&q.dims, &narrow).unwrap();
            assert_eq!(stats.documents, narrow.count());
        }
    }

    let mut rng = Rng::new(99);
    let mut dead = Bitmap::default();
    let mut doomed = Vec::new();
    for id in 1..=n {
        if rng.below(100) < 20 {
            dead.insert(id);
            doomed.push(RecordId(id as u32));
        }
    }
    for id in &doomed {
        lazy.remove(*id).unwrap();
    }
    let live_after = live_bitmap(n, &dead);
    let narrow_after = filter_bitmap(n, 100, 7, &dead);
    let chained_after = Candidates::Sorted(
        sorted_ids
            .iter()
            .copied()
            .filter(|id| !dead.contains(id.slot()))
            .collect(),
    );
    let admits_after: Vec<(&str, &dyn Admit)> = vec![
        ("live", &live_after),
        ("filter_10pct", &narrow_after),
        ("chained_50", &chained_after),
    ];
    let admits_stale: Vec<(&str, &dyn Admit)> = vec![
        ("live_stale", &live),
        ("filter_10pct_stale", &narrow),
        ("chained_50_stale", &chained),
    ];
    check_bm25(
        &mut tally,
        "removed",
        &corpus,
        &lazy,
        &dead,
        &admits_after,
        k,
        k1,
        b,
    );
    check_bm25(
        &mut tally,
        "removed_stale",
        &corpus,
        &lazy,
        &dead,
        &admits_stale,
        k,
        k1,
        b,
    );

    let dir = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::default();
    lazy.write("bm25.", dir.path(), &mut manifest).unwrap();
    let bounds = Bounds {
        min_records: 0,
        max_records: n,
        max_bytes: 1 << 34,
    };
    let restored =
        PostingsIndex::restore(lazy.config(), "bm25.", dir.path(), &manifest, &bounds).unwrap();
    check_bm25(
        &mut tally,
        "restored",
        &corpus,
        &restored,
        &dead,
        &admits_after,
        k,
        k1,
        b,
    );

    lazy.compact();
    check_bm25(
        &mut tally,
        "compacted",
        &corpus,
        &lazy,
        &dead,
        &admits_stale,
        k,
        k1,
        b,
    );

    (tally.pages, tally.differ)
}

#[test]
fn the_text_regime_weighted_by_term_frequency_matches_brute_force() {
    let (pages, differ) = verify_bm25("text", 3_000, 40);
    assert_eq!(
        differ, 0,
        "{differ} of {pages} pages differ from brute force"
    );
    assert!(pages > 6_000);
}

/// The `splade` structure under the term frequency weighting, which takes
/// whole numbers alone, so over the regime that rounds its weights up.
#[test]
fn the_splade_counts_regime_weighted_by_term_frequency_matches_brute_force() {
    let (pages, differ) = verify_bm25("splade-counts", 1_000, 30);
    assert_eq!(
        differ, 0,
        "{differ} of {pages} pages differ from brute force"
    );
    assert!(pages > 4_000);
}

#[test]
#[ignore]
fn the_text_regime_weighted_by_term_frequency_matches_brute_force_at_scale() {
    let (pages, differ) = verify_bm25("text", 50_000, 100);
    eprintln!("text bm25 at scale: {pages} pages, {differ} differ");
    assert_eq!(differ, 0);
}

#[test]
fn the_text_regime_matches_brute_force() {
    let (pages, differ) = verify("text", 4_000, 60);
    assert_eq!(
        differ, 0,
        "{differ} of {pages} pages differ from brute force"
    );
    assert!(pages > 8_000);
}

#[test]
fn the_splade_regime_matches_brute_force() {
    let (pages, differ) = verify("splade", 1_500, 40);
    assert_eq!(
        differ, 0,
        "{differ} of {pages} pages differ from brute force"
    );
    assert!(pages > 5_000);
}

/// The scale the structure was measured at. Run with `--ignored`, since a
/// brute-force page over fifty thousand records takes a millisecond and the
/// two regimes together compare over a hundred thousand of them.
#[test]
#[ignore]
fn the_text_regime_matches_brute_force_at_scale() {
    let (pages, differ) = verify("text", 50_000, 200);
    eprintln!("text at scale: {pages} pages, {differ} differ");
    assert_eq!(differ, 0);
}

#[test]
#[ignore]
fn the_splade_regime_matches_brute_force_at_scale() {
    let (pages, differ) = verify("splade", 20_000, 200);
    eprintln!("splade at scale: {pages} pages, {differ} differ");
    assert_eq!(differ, 0);
}
