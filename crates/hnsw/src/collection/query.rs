//! A query over one or more arms, being the search a collection with more
//! than one space answers.
//!
//! # What a query is
//!
//! One or more arms, each naming a space by the kind of query it carries,
//! a filter the arms share, the page size `k`, how many candidates each arm
//! contributes, and how the arms' pages are fused. A one arm query is that
//! arm's search and returns its page, so the single space search methods on
//! the collection are one arm queries. A query over several arms runs each
//! under the same admit set and fuses the pages by rank; see the fusion
//! module in the engine's floor for what a fused page is and what it is
//! not.
//!
//! # What the planner decides
//!
//! Less than the word suggests, and deliberately. Every arm produces its
//! page over the whole admitted set, because the fused page is a function
//! of those pages and nothing else. An arm narrowed to another arm's
//! candidates returns a different page, which is a different query rather
//! than a cheaper way to run this one, and an arm left out changes the page
//! unless its page was empty, in which case it costs nothing to run. So
//! the planner does not choose between fusing the arms and chaining them,
//! and it drops none. What it decides is the admit set, once, for every
//! arm, including the one thing it can vary there without moving a page: a
//! filter admitting every live record is handed on as no filter at all,
//! which spares a term weighted sparse arm the walk over the whole live set
//! it would otherwise make, a fifth of such a query, and a dot product arm
//! the bit test per posting, an eighth, while a dense traversal measures
//! the same under either shape. It then prices every arm under that set,
//! so a caller can see what a query
//! costs before it runs, through [`Collection::explain`], and beside the
//! page after it ran, through [`Page::plan`].
//!
//! # Locks
//!
//! The guards a single space search takes, in the declared order,
//! `id_map < rev_map < dense index < dense codes < sparse index <
//! vector_metadata < columns`, each space's taken only where an arm reads
//! it. A text arm is tokenized under no guard at all, since the tokenizer
//! may be the caller's own, and its terms are counted into term ids under
//! the dictionary's guard before any of the above is taken, and that guard
//! is released before the first of them is, which is the rule the text
//! layer declares.

use super::search::{AdmitPlan, Scored, MAX_TOP_K};
use super::Collection;
use crate::{RawVectors, SearchParams};
use serde_json::Value;
use std::collections::HashMap;
use zeusdb_vector_core::{
    fuse, Budget, Contribution, Cost, Error, Filter, Fusion, IdfScope, Selectivity, SpaceKind,
    SpaceName, SparseRef, SparseVector, VectorIndex,
};
use zeusdb_vector_text::count_query;

/// Arms a query may name.
///
/// Each arm is one search, so a query costs the sum of its arms, and a
/// caller asking for more than this many is asking for something other
/// than a page.
pub const MAX_ARMS: usize = 8;

/// Candidates each arm of a query over several arms contributes, per unit
/// of `k`, where the query names no fetch depth.
///
/// A fused page is cut from the union of the arms' pages, so a record just
/// outside one arm's page cannot be lifted by its rank on the other, and
/// the deeper each arm fetches the more the fusion has to work with.
/// Measured on a synthetic corpus of 20,000 records where both arms carry
/// signal about a known target, recall@10 of the target went from 0.684 on
/// the better arm alone to 0.842 fused at a fetch of `k`, 0.892 at five
/// times `k` and 0.919 at ten. Five is the largest multiple the dense arm
/// pays nothing measurable for at a page of ten, since the traversal's
/// default width is the larger of twice the fetch and one hundred, so
/// fetching fifty runs the traversal a single search runs and fetching a
/// hundred doubles its width: a fused query at fetch fifty measured 1.06 to
/// 1.13 of one at fetch ten with every interval including 1.0, and at a
/// hundred 1.35 to 1.63. A postings scan's cost does not move with the
/// fetch. A caller with a large `k` who wants the dense arm at its default
/// width sets `fetch`.
pub const DEFAULT_FETCH_PER_K: usize = 5;

/// One arm of a query. The kind of query names the space.
#[derive(Clone, Copy, Debug)]
pub enum Arm<'a> {
    /// The dense space, asked with a vector on the caller's scale, which is
    /// checked and processed for the space as a stored vector is.
    Dense {
        vector: &'a [f32],
        /// The traversal width, or the space's default for the page size.
        ef: Option<usize>,
        /// The rerank depth on a quantized space. See `Collection::search`.
        rerank: Option<usize>,
    },
    /// The sparse space, asked with term ids and weights.
    Sparse {
        vector: SparseRef<'a>,
        idf: IdfScope,
    },
    /// The sparse space's text layer, asked with a string the layer's
    /// tokenizer splits, under no guard, and the layer counts into term
    /// ids as it counted the records.
    Text { text: &'a str, idf: IdfScope },
    /// The sparse space's text layer, asked with terms a caller has already
    /// split as the layer's tokenizer would, in order and repeats included,
    /// which the layer counts into term ids as it counted the records. What
    /// a caller hands over after running the tokenizer itself through
    /// `Collection::tokenize`, under whatever the tokenizer needs.
    Terms { terms: &'a [String], idf: IdfScope },
}

impl Arm<'_> {
    /// The kind of space the arm names.
    pub fn kind(&self) -> SpaceKind {
        match self {
            Arm::Dense { .. } => SpaceKind::Dense,
            Arm::Sparse { .. } | Arm::Text { .. } | Arm::Terms { .. } => SpaceKind::Sparse,
        }
    }
}

/// A query.
///
/// `Copy`, since every field is a borrow or a small value, and without
/// `Debug`, since a compiled filter has none.
#[derive(Clone, Copy)]
pub struct Query<'a> {
    /// The arms, in the order their contributions are reported.
    pub arms: &'a [Arm<'a>],
    /// The filter every arm runs under.
    pub filter: Option<&'a Filter>,
    /// The page size.
    pub k: usize,
    /// Candidates each arm contributes to the fusion. Unset, it is `k` for
    /// a one arm query, whose page is the arm's own, and [`DEFAULT_FETCH_PER_K`]
    /// times `k` for a query over several arms; see that constant.
    pub fetch: Option<usize>,
    /// How the arms' pages become one. Read where there is more than one
    /// arm.
    pub fusion: Fusion,
}

impl<'a> Query<'a> {
    /// A query over these arms at page size `k`, with no filter, the fetch
    /// depth and the fusion at their defaults.
    pub fn new(arms: &'a [Arm<'a>], k: usize) -> Self {
        Query {
            arms,
            filter: None,
            k,
            fetch: None,
            fusion: Fusion::default(),
        }
    }
}

/// One record on the page.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryHit {
    pub id: String,
    /// The fused score where the query had several arms, and the arm's own
    /// score where it had one.
    pub score: f32,
    /// The record's rank and score on every arm's page it appeared on, in
    /// arm order, which is what the fused score was made from.
    pub contributions: Vec<Contribution>,
    pub metadata: HashMap<String, Value>,
}

/// The page, with the plan that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    pub hits: Vec<QueryHit>,
    pub plan: Plan,
}

/// What the planner decided, being the admit set every arm ran under,
/// what each arm was asked for and what it was priced at, and how the
/// pages were fused.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub admit: AdmitShape,
    pub arms: Vec<ArmPlan>,
    /// The fusion applied, or `None` for a one arm query, whose page is the
    /// arm's own.
    pub fusion: Option<Fusion>,
}

/// The shape of the admit set every arm ran under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitShape {
    /// Every live record, for no filter or for a filter that admits every
    /// live record above the dense scan threshold.
    All,
    /// A filter's bitmap, of this many records.
    Bitmap { admitted: usize },
    /// The matches of a metadata walk that finished, as a sorted list of
    /// this many records.
    Sorted { admitted: usize },
    /// A bound of this many records the declared fields left, conjoined
    /// with the metadata predicate, because the walk inside it gave up.
    Bounded { bound: usize },
    /// The metadata predicate alone, because the walk over every record
    /// gave up.
    Predicate,
}

impl AdmitShape {
    /// What a planner tells an arm about the set, where it can say
    /// anything. A bitmap and a sorted list are exact. A bound is an upper
    /// limit whose expected value is taken as the limit, since the walk
    /// that gave up inside it counted nothing it could report. Everything
    /// and a predicate say nothing, and an arm then prices an unfiltered
    /// search.
    pub fn selectivity(&self) -> Option<Selectivity> {
        match *self {
            AdmitShape::All | AdmitShape::Predicate => None,
            AdmitShape::Bitmap { admitted } | AdmitShape::Sorted { admitted } => Some(
                Selectivity::exact(u32::try_from(admitted).unwrap_or(u32::MAX)),
            ),
            AdmitShape::Bounded { bound } => {
                let bound = u32::try_from(bound).unwrap_or(u32::MAX);
                Some(Selectivity {
                    min: 0,
                    expected: bound,
                    max: bound,
                })
            }
        }
    }
}

impl AdmitPlan<'_> {
    fn shape(&self) -> AdmitShape {
        match self {
            AdmitPlan::All => AdmitShape::All,
            AdmitPlan::Bitmap(bitmap) => AdmitShape::Bitmap {
                admitted: bitmap.count(),
            },
            AdmitPlan::Matched(ids) => AdmitShape::Sorted {
                admitted: ids.len(),
            },
            AdmitPlan::Bounded(bound, _) => AdmitShape::Bounded {
                bound: bound.count(),
            },
            AdmitPlan::Predicate(_) => AdmitShape::Predicate,
        }
    }
}

/// One arm as planned.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmPlan {
    pub space: SpaceName,
    pub kind: SpaceKind,
    /// Candidates the arm was asked for, which on a reranking dense arm is
    /// the over-fetch and not the page it contributes.
    pub fetch: usize,
    /// What the arm said the search would cost under the admit set, in
    /// estimated nanoseconds, and whether its page is exact.
    ///
    /// The index's estimate of its own work, and nothing else. It leaves
    /// out what the collection pays around the arm, being the filter's
    /// evaluation over the columns and the page's assembly, which together
    /// measured near forty microseconds a query at fifty thousand records
    /// whatever the filter admitted, and it is as good as the units the
    /// index timed on itself: a graph of a hundred dimensions priced its
    /// searches at a third to a half of their measured time at every
    /// selectivity, and one of fifteen hundred within a fifth at the broad
    /// ones.
    pub cost: Cost,
}

/// An arm with its query settled before any guard is taken: the dense
/// vector checked and processed for its space with its parameters resolved,
/// or the sparse vector owned, a text arm's counted from the dictionary.
enum Resolved {
    Dense {
        vector: Vec<f32>,
        params: SearchParams,
    },
    Sparse {
        vector: SparseVector,
        idf: IdfScope,
    },
}

impl Collection {
    /// The plan a query would run under, without running it.
    ///
    /// Takes the guards the query would take, decides the admit set from
    /// the filter, which for a filter over undeclared fields is the walk
    /// itself, prices every arm and releases. What it reports is what
    /// [`Collection::query`] reports beside its page.
    pub fn explain(&self, query: &Query<'_>) -> Result<Plan, Error> {
        self.execute(query, false).map(|(plan, _)| plan)
    }

    /// Run a query and return its page with the plan that produced it.
    ///
    /// Every rule a query has to satisfy is checked before any guard is
    /// taken, so a bad query changes nothing and costs nothing. The page is
    /// best first, cut to `k`, and may be shorter, since an arm returns no
    /// record that scored nothing and a filter may admit fewer records than
    /// were asked for. Among equal fused scores the external id orders the
    /// page, which is the rule every exact single space page applies, so a
    /// page is the same page from one run to the next.
    pub fn query(&self, query: &Query<'_>) -> Result<Page, Error> {
        let (plan, hits) = self.execute(query, true)?;
        Ok(Page {
            hits: hits.unwrap_or_default(),
            plan,
        })
    }

    /// Hold a query to its rules and settle every arm's query, taking no
    /// guard but a text arm's dictionary, alone.
    fn resolve_arms(&self, query: &Query<'_>) -> Result<(Vec<Resolved>, usize), Error> {
        if query.arms.is_empty() {
            return Err(Error::QueryArmsEmpty);
        }
        if query.arms.len() > MAX_ARMS {
            return Err(Error::QueryArmsTooMany {
                max: MAX_ARMS,
                arms: query.arms.len(),
            });
        }
        if query.k > MAX_TOP_K {
            return Err(Error::TopKTooLarge {
                max: MAX_TOP_K,
                top_k: query.k,
            });
        }
        let fetch = query.fetch.unwrap_or(if query.arms.len() > 1 {
            query.k.saturating_mul(DEFAULT_FETCH_PER_K)
        } else {
            query.k
        });
        if fetch > MAX_TOP_K {
            return Err(Error::FetchTooLarge {
                max: MAX_TOP_K,
                fetch,
            });
        }
        query.fusion.validate()?;
        let mut resolved = Vec::with_capacity(query.arms.len());
        for arm in query.arms {
            resolved.push(match *arm {
                Arm::Dense { vector, ef, rerank } => {
                    // The bounds, the width default and the rerank plan,
                    // resolved once here as the single space search does,
                    // and before any guard because the rerank plan reads
                    // the graph's own.
                    let params = self.search_params(fetch, ef, false, rerank)?;
                    let vector = self.validate_query(vector.to_vec())?;
                    Resolved::Dense { vector, params }
                }
                Arm::Sparse { vector, idf } => {
                    vector.validate()?;
                    self.sparse().ok_or(Error::NoSparseSpace)?;
                    Resolved::Sparse {
                        vector: SparseVector {
                            dims: vector.dims.to_vec(),
                            values: vector.values.to_vec(),
                        },
                        idf,
                    }
                }
                Arm::Text { text, idf } => {
                    // The tokenizer under no guard, then the terms under
                    // the dictionary's alone; see `Collection::tokenize`.
                    let terms = self.tokenize(text)?;
                    Resolved::Sparse {
                        vector: self.count_terms(&terms)?,
                        idf,
                    }
                }
                Arm::Terms { terms, idf } => Resolved::Sparse {
                    vector: self.count_terms(terms)?,
                    idf,
                },
            });
        }
        Ok((resolved, fetch))
    }

    /// Terms looked up in the text layer's dictionary and counted, under
    /// its read guard taken alone and released before any search guard is
    /// taken. A term no record has carried is dropped.
    fn count_terms(&self, terms: &[String]) -> Result<SparseVector, Error> {
        let layer = self.text_layer()?;
        let dictionary = layer.dictionary.read().unwrap();
        Ok(count_query(&dictionary, terms))
    }

    /// Plan, and run where asked.
    fn execute(
        &self,
        query: &Query<'_>,
        run: bool,
    ) -> Result<(Plan, Option<Vec<QueryHit>>), Error> {
        let (arms, fetch) = self.resolve_arms(query)?;
        let wants_dense = arms.iter().any(|arm| matches!(arm, Resolved::Dense { .. }));
        let wants_sparse = arms
            .iter()
            .any(|arm| matches!(arm, Resolved::Sparse { .. }));

        // The guards, in the declared order, a space's only where an arm
        // reads it.
        let id_map = self.id_map.read().unwrap();
        let rev_map = self.rev_map.read().unwrap();
        let dense = wants_dense.then(|| self.dense().index.read().unwrap());
        let pq_codes = wants_dense.then(|| self.dense().pq_codes.read().unwrap());
        let sparse_space = wants_sparse.then(|| {
            self.sparse()
                .expect("every sparse arm found the sparse space when it was resolved")
        });
        let sparse = sparse_space.map(|space| space.index.read().unwrap());
        let vector_metadata = self.vector_metadata.read().unwrap();
        let columns = self.columns.read().unwrap();

        // The admit set, once, for every arm.
        let admit = self.admit_plan(query.filter, &columns, &rev_map, &vector_metadata, &id_map);
        let shape = admit.shape();
        let selectivity = shape.selectivity();
        let live = rev_map.len();

        let planned: Vec<ArmPlan> = arms
            .iter()
            .map(|arm| match arm {
                Resolved::Dense { vector, params } => {
                    let index = dense.as_ref().expect("a dense arm took the dense guard");
                    let fetch_k = params.fetch_k(live);
                    ArmPlan {
                        space: self.spaces[0].name.clone(),
                        kind: SpaceKind::Dense,
                        fetch: fetch_k,
                        cost: index.cost(vector, fetch_k, selectivity.as_ref()),
                    }
                }
                Resolved::Sparse { vector, .. } => {
                    let index = sparse.as_ref().expect("a sparse arm took the sparse guard");
                    ArmPlan {
                        space: self.sparse_name(),
                        kind: SpaceKind::Sparse,
                        fetch,
                        cost: index.cost(vector.as_ref(), fetch, selectivity.as_ref()),
                    }
                }
            })
            .collect();
        let plan = Plan {
            admit: shape,
            arms: planned,
            fusion: (arms.len() > 1).then_some(query.fusion),
        };
        if !run {
            return Ok((plan, None));
        }

        // Every arm's page, best first, under the one admit set.
        let mut pages: Vec<Vec<(&String, f32)>> = Vec::with_capacity(arms.len());
        for arm in &arms {
            match arm {
                Resolved::Dense { vector, params } => {
                    let index = dense.as_ref().expect("a dense arm took the dense guard");
                    let codes = pq_codes.as_ref().expect("a dense arm took the codes guard");
                    let fetch_k = params.fetch_k(live);
                    let budget = Self::dense_budget(params);
                    let hits = admit.run(|admit| index.search(vector, fetch_k, admit, &budget))?;
                    let mut page = Scored::resolve(hits, &rev_map).cut(fetch_k).items;
                    let raws = RawVectors {
                        id_map: &id_map,
                        graph: index.graph(),
                    };
                    self.rescore_page(&mut page, vector, raws, codes, params);
                    pages.push(page);
                }
                Resolved::Sparse { vector, idf } => {
                    let index = sparse.as_ref().expect("a sparse arm took the sparse guard");
                    let budget = Budget {
                        boundary_ties: true,
                        idf: *idf,
                        ..Budget::default()
                    };
                    let hits =
                        admit.run(|admit| index.search(vector.as_ref(), fetch, admit, &budget))?;
                    let mut page: Vec<(&String, f32)> = Vec::with_capacity(hits.items.len());
                    for hit in hits.items {
                        if let Some(ext_id) = rev_map.get(&hit.id.slot()) {
                            page.push((ext_id, hit.score));
                        }
                    }
                    // Higher is better, and the tie goes to the external
                    // id, as an exact dense page's does.
                    page.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                    page.truncate(fetch);
                    pages.push(page);
                }
            }
        }

        let metadata_of = |id: &String| vector_metadata.get(id).cloned().unwrap_or_default();
        let hits: Vec<QueryHit> = if pages.len() == 1 {
            // One arm, so its page is the page and its score is the score.
            pages[0]
                .iter()
                .take(query.k)
                .enumerate()
                .map(|(position, (id, score))| QueryHit {
                    id: (*id).clone(),
                    score: *score,
                    contributions: vec![Contribution {
                        arm: 0,
                        rank: position + 1,
                        score: *score,
                    }],
                    metadata: metadata_of(id),
                })
                .collect()
        } else {
            let slices: Vec<&[(&String, f32)]> = pages.iter().map(|p| p.as_slice()).collect();
            fuse(query.fusion, &slices)
                .into_iter()
                .take(query.k)
                .map(|fused| QueryHit {
                    id: fused.id.clone(),
                    score: fused.score,
                    contributions: fused.contributions,
                    metadata: metadata_of(fused.id),
                })
                .collect()
        };
        Ok((plan, Some(hits)))
    }

    /// The sparse space's name, where one was declared.
    fn sparse_name(&self) -> SpaceName {
        self.spaces
            .iter()
            .find(|named| matches!(named.space, super::Space::Sparse(_)))
            .map(|named| named.name.clone())
            .expect("a sparse arm found the sparse space when it was resolved")
    }
}
