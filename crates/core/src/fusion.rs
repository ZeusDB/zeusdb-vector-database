//! Fusing the pages of several arms into one.
//!
//! A collection with more than one space answers a query by asking each
//! space for a page and combining the pages, and this module is the
//! combination. It sees ranked lists of ids and nothing else, so it names
//! no index type and the collection is the only caller.
//!
//! # What a score means, and why the fusion reads none
//!
//! Each arm's page carries a [`ScoreKind`]. A dense arm reports a distance,
//! lower better, on the scale of its metric, and a sparse arm reports a
//! similarity, higher better, on whatever scale its scoring rule has. The
//! two are not comparable, and a sparse score under a term frequency
//! weighting is not even stable in itself, because it is a function of the
//! corpus at the moment of the query and an unrelated removal moves it. A
//! fusion that normalised by a stored range would therefore be wrong by
//! construction, and one that normalised by the page's own extremes is
//! undefined where the extremes coincide and amplifies noise where they are
//! close. Reciprocal rank fusion reads the order of each page and nothing
//! else, so it needs no scale, no direction and no stored statistic. What it
//! gives up is the gap: a strong hit and a marginal one at the same rank
//! count the same.
//!
//! # What the fused list is
//!
//! A deterministic function of the input pages. Every record on any page
//! is on the fused list, a record on one page alone included, with a fused
//! score that is the sum over the pages it appears on of `1 / (k + rank)`,
//! rank counted from one. The list is ordered by fused score and then by
//! id, so two records with the same ranks on the same pages, which tie
//! exactly, are ordered by a key that does not depend on the order the
//! pages were given in. The id type is the caller's, so the tie is broken
//! by whatever key the caller ranks its own pages by, and the list is not
//! cut, so the caller cuts it to the page it wants under the same key.
//!
//! [`ScoreKind`]: crate::space::ScoreKind

use std::collections::HashMap;
use std::hash::Hash;

use crate::error::Error;

/// The reciprocal rank constant a query applies unless told otherwise.
///
/// The value the rule is most often published with, and measured rather
/// than taken on faith. On this engine's own synthetic corpora, at a fetch
/// equal to the page every value from 0 to 1,000 fuses to the same page,
/// since the constant only reorders within the union of two pages the size
/// of the result. At a fetch of ten times the page, where the union is
/// larger than the result, values under 30 lose up to 0.07 of recall@10
/// against this one and every value from 60 to 1,000 sits within 0.002 of
/// it, so the published figure is on the plateau and nothing here depends
/// on its exact value.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// How the pages of several arms become one page.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Fusion {
    /// Reciprocal rank fusion. A record's fused score is the sum over the
    /// pages it appears on of `1 / (k + rank)`, with rank counted from one
    /// on each page. `k` damps the gap between the top ranks: at zero the
    /// first record on a page is worth twice the second, and at sixty the
    /// two are within two percent of each other.
    ReciprocalRank { k: f32 },
}

impl Fusion {
    /// Reciprocal rank fusion at the published constant.
    pub const RRF: Fusion = Fusion::ReciprocalRank { k: DEFAULT_RRF_K };

    /// The rules a fusion has to satisfy before a query is run under it.
    ///
    /// The reciprocal rank constant is finite and at least zero, since
    /// `k + rank` is a divisor and a negative `k` can put it at zero.
    pub fn validate(&self) -> Result<(), Error> {
        match *self {
            Fusion::ReciprocalRank { k } => {
                if !(k.is_finite() && k >= 0.0) {
                    return Err(Error::FusionConstantInvalid { value: k });
                }
            }
        }
        Ok(())
    }
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion::RRF
    }
}

/// One record's place on one arm's page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contribution {
    /// Which page, as the index of the page in the list handed to [`fuse`].
    pub arm: usize,
    /// The record's rank on that page, counted from one.
    pub rank: usize,
    /// The score the arm reported for it, whose meaning is the arm's
    /// [`ScoreKind`](crate::space::ScoreKind).
    pub score: f32,
}

/// One record on the fused list, with what each page contributed.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedHit<I> {
    pub id: I,
    /// The fused score. Higher is better, and it is comparable only with
    /// other fused scores from the same query.
    pub score: f32,
    /// The record's place on every page it appeared on, in page order.
    pub contributions: Vec<Contribution>,
}

/// Fuse several pages into one list, best first.
///
/// Each page is a ranked list of ids with the score its arm gave each,
/// best first. The order is read as the ranking and the scores are carried
/// through untouched, so what an arm's score means does not matter to the
/// result. The list holds every id that appears on any page, ordered by
/// fused score and then by id, and is not cut. See the module documentation
/// for what that leaves to the caller.
pub fn fuse<I: Clone + Ord + Hash>(fusion: Fusion, pages: &[&[(I, f32)]]) -> Vec<FusedHit<I>> {
    let Fusion::ReciprocalRank { k } = fusion;
    let k = k as f64;
    // Accumulated in double precision so the sum of reciprocals is the same
    // number whichever page contributed first, and stored single at the end.
    // A record's first appearance opens its entry, and every later page adds
    // to it, found through a map from the id to its entry. A linear search
    // over the open entries instead cost a quarter of a fused query at a
    // fetch of a hundred, since every comparison was a string comparison.
    let total: usize = pages.iter().map(|page| page.len()).sum();
    let mut fused: Vec<(I, f64, Vec<Contribution>)> = Vec::with_capacity(total);
    let mut entry_of: HashMap<&I, usize> = HashMap::with_capacity(total);
    for (arm, page) in pages.iter().enumerate() {
        for (position, (id, score)) in page.iter().enumerate() {
            let rank = position + 1;
            let weight = 1.0 / (k + rank as f64);
            let contribution = Contribution {
                arm,
                rank,
                score: *score,
            };
            match entry_of.get(id) {
                Some(&at) => {
                    fused[at].1 += weight;
                    fused[at].2.push(contribution);
                }
                None => {
                    entry_of.insert(id, fused.len());
                    fused.push((id.clone(), weight, vec![contribution]));
                }
            }
        }
    }
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
        .into_iter()
        .map(|(id, score, contributions)| FusedHit {
            id,
            score: score as f32,
            contributions,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distance page, scored ascending as a dense arm reports.
    fn distances(ids: &[u32]) -> Vec<(u32, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, i as f32 * 0.5))
            .collect()
    }

    /// A similarity page, scored descending as a sparse arm reports.
    fn similarities(ids: &[u32]) -> Vec<(u32, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, 10.0 - i as f32))
            .collect()
    }

    fn fused_of(fusion: Fusion, pages: &[Vec<(u32, f32)>]) -> Vec<FusedHit<u32>> {
        let slices: Vec<&[(u32, f32)]> = pages.iter().map(|p| p.as_slice()).collect();
        fuse(fusion, &slices)
    }

    fn ids(fused: &[FusedHit<u32>]) -> Vec<u32> {
        fused.iter().map(|h| h.id).collect()
    }

    /// The fused list is the reciprocal rank sum by hand, holds a record on
    /// one page alone, and names every page a record appeared on.
    #[test]
    fn the_fused_list_is_the_reciprocal_rank_sum_by_hand() {
        let dense = distances(&[1, 2, 3, 4]);
        let sparse = similarities(&[3, 1, 5]);
        let fused = fused_of(Fusion::ReciprocalRank { k: 60.0 }, &[dense, sparse]);
        let rrf = |ranks: &[usize]| -> f32 {
            ranks.iter().map(|&r| 1.0 / (60.0 + r as f64)).sum::<f64>() as f32
        };
        // 1 is first on the dense page and second on the sparse one, 3 is
        // third and first, and the two sums are equal, so the tie goes to
        // the lower id. 2 and 5 are on one page each, 4 is last.
        assert_eq!(ids(&fused), vec![1, 3, 2, 5, 4]);
        assert_eq!(fused[0].score, rrf(&[1, 2]));
        assert_eq!(fused[1].score, rrf(&[3, 1]));
        assert_eq!(fused[2].score, rrf(&[2]));
        assert_eq!(fused[3].score, rrf(&[3]));
        assert_eq!(fused[4].score, rrf(&[4]));
        assert_eq!(
            fused[0].contributions,
            vec![
                Contribution {
                    arm: 0,
                    rank: 1,
                    score: 0.0
                },
                Contribution {
                    arm: 1,
                    rank: 2,
                    score: 9.0
                }
            ]
        );
        assert_eq!(
            fused[3].contributions,
            vec![Contribution {
                arm: 1,
                rank: 3,
                score: 8.0
            }]
        );
    }

    /// The order of the pages does not change the list, a page's scores do
    /// not change it, and running it twice gives the same list.
    #[test]
    fn the_fused_list_does_not_depend_on_page_order_or_scores() {
        let a = distances(&[7, 2, 9, 4, 1]);
        let b = similarities(&[4, 7, 6]);
        let ab = fused_of(Fusion::RRF, &[a.clone(), b.clone()]);
        let ba = fused_of(Fusion::RRF, &[b.clone(), a.clone()]);
        assert_eq!(ids(&ab), ids(&ba));
        assert!(ab
            .iter()
            .zip(&ba)
            .all(|(x, y)| x.score.to_bits() == y.score.to_bits()));
        // The same pages with every score replaced.
        let rescored: Vec<(u32, f32)> = a.iter().map(|&(id, _)| (id, 123.0)).collect();
        let again = fused_of(Fusion::RRF, &[rescored, b.clone()]);
        assert_eq!(ids(&again), ids(&ab));
        assert_eq!(fused_of(Fusion::RRF, &[a, b]), ab);
    }

    /// A one page fusion is the page, an empty page contributes nothing, and
    /// no pages fuse to nothing.
    #[test]
    fn one_page_fuses_to_itself_and_an_empty_page_adds_nothing() {
        let a = similarities(&[5, 3, 8]);
        let alone = fused_of(Fusion::RRF, std::slice::from_ref(&a));
        assert_eq!(ids(&alone), vec![5, 3, 8]);
        assert!(alone.windows(2).all(|w| w[0].score > w[1].score));
        let empty = distances(&[]);
        assert_eq!(ids(&fused_of(Fusion::RRF, &[a, empty])), vec![5, 3, 8]);
        assert!(fuse::<u32>(Fusion::RRF, &[]).is_empty());
    }

    /// The constant damps the top of the page: at zero the first record on
    /// one page outranks the second on both, and at sixty the two do not.
    #[test]
    fn the_constant_decides_how_much_the_top_rank_is_worth() {
        let a = distances(&[1, 2]);
        let b = similarities(&[3, 2]);
        // At k = 0 record 1 scores 1 and record 2 scores 1/2 + 1/2 = 1, a
        // tie, so the lower id leads; at any k above zero record 2 leads.
        assert_eq!(
            ids(&fused_of(
                Fusion::ReciprocalRank { k: 0.0 },
                &[a.clone(), b.clone()]
            ))[0],
            1
        );
        assert_eq!(ids(&fused_of(Fusion::RRF, &[a, b]))[0], 2);
    }

    /// The constant is held to its rule.
    #[test]
    fn the_constant_is_finite_and_at_least_zero() {
        assert!(Fusion::RRF.validate().is_ok());
        assert!(Fusion::ReciprocalRank { k: 0.0 }.validate().is_ok());
        for value in [-1.0f32, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                Fusion::ReciprocalRank { k: value }.validate(),
                Err(Error::FusionConstantInvalid { .. })
            ));
        }
        assert_eq!(Fusion::default(), Fusion::RRF);
    }
}
