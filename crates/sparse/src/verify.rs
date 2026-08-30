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

use std::collections::HashMap;

use zeusdb_vector_core::{
    Admit, ArtefactRecord, Bitmap, Bounds, Budget, Candidates, Hit, Inventory, Ledger, Persist,
    Prepared, RecordId, Restore, SparseRef, VectorIndex,
};

use crate::corpus::{self, Corpus, Rng};
use crate::index::{PostingsIndex, SparseConfig, Unlink};
use crate::search::Mode;

const MODES: [Mode; 6] = [
    Mode::Auto,
    Mode::PerPosting,
    Mode::PerCandidate,
    Mode::BitmapPerPosting,
    Mode::Enumerate,
    Mode::Floor,
];

fn build(corpus: &Corpus, unlink: Unlink) -> PostingsIndex {
    let mut index = PostingsIndex::new(SparseConfig {
        unlink,
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
