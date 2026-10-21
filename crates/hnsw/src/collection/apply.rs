//! A journaled operation applied at the values its record names.
//!
//! `apply` is the other end of the seam. Every mutation hands its record to
//! the collection's sink before it runs, and `apply` takes such a record,
//! decoded, and runs the same mutation, so a collection built by a script of
//! operations and one built by applying the script's records to a
//! checkpoint are the same collection, artefact for artefact. A record
//! carries three values the mutation would otherwise issue, being the
//! internal id, the graph level and a term's id, and `apply` installs at
//! them rather than issuing its own, checking each against what the
//! collection would have issued.
//!
//! A check that fails means the records and the collection do not belong
//! together: a checkpoint and a journal from different histories, a journal
//! applied twice, or a record built by hand. `apply` refuses with
//! [`Error::JournalReplayMismatch`] naming what disagreed, leaves the
//! collection as it was before that record, and the caller stops there.
//! Nothing here panics on a record's content, since the record came off a
//! disk.
//!
//! `apply` hands nothing to the sink, and refuses to run while one is
//! attached, so a replay is never recorded as new mutations.
//!
//! `apply_withdrawn` takes a record of a call whose commit failed, which the
//! live collection refused, and does what the live collection kept of it:
//! an insert's internal id stays issued, since admission issued it before
//! the commit, and a term stays interned, since the dictionary kept it.
//! Nothing else of such a call is applied.

use super::insert::Admitted;
use super::{validate_index_parameters, Collection};
use std::collections::HashMap;
use zeusdb_vector_core::{Error, Operation, SparseVector, MAX_INTERNAL_ID};

impl Collection {
    /// Apply one recorded operation to the collection.
    ///
    /// Under the mutation guard, as the entry point it re-runs was. What
    /// each kind checks before it runs, and what it does.
    ///
    /// - `Insert`. The internal id is held under the id ceiling and equal to
    ///   the one the counter would issue next, the record's id must not be
    ///   held already, the vector is the declared width and finite, which
    ///   is the refusal `add` makes at its door, and a sparse half is held
    ///   to the space's rules under its weighting, as admission holds it.
    ///   The id is then issued, no level is drawn, and the record is
    ///   installed at the level it names through the path `add` installs
    ///   through, training trigger included. A training that fires here
    ///   stamps itself from the clock, and the `Train` record that follows
    ///   it in the journal restamps it.
    /// - `Remove`. Every id must be held, since the record named what was
    ///   held. Then the removal.
    /// - `UpdateMetadata`. The id must be held. Then the replacement.
    /// - `Intern`. The term id must be the one the dictionary would issue
    ///   and the term must be new to it. Then the interning.
    /// - `Rebuild`. The three values are held to the rules `create()`
    ///   applies, through the same function. Then the rebuild.
    /// - `Train`. The codebook must be fitted, since the record was written
    ///   as a training began. The stamp is installed and nothing else runs,
    ///   because the training itself fired as the insert that filled the
    ///   set was applied.
    /// - `Clear`, `Compact`, `RebuildQuantized` and `AddMetadata` run their
    ///   bodies.
    ///
    /// Returns [`Error::JournalReplayMismatch`] for a check that fails, the
    /// ordinary refusal for a value the ordinary path would refuse, and
    /// whatever the body returns for a body that fails.
    pub fn apply(&self, operation: Operation) -> Result<(), Error> {
        // What a clear traded out, declared before the holder of `writers`
        // so it drops once the holder has let `writers` go.
        let mut cleared = None;
        let writers = self.hold_writers();
        if self.sink_attached() {
            return Err(Error::Engine(
                "A recorded operation cannot be applied while a sink is attached, since a \
                 replay is not a new mutation. Detach the sink, apply, and attach it again."
                    .to_string(),
            ));
        }
        let mismatch = |detail: String| Error::JournalReplayMismatch { detail };
        let applied = match operation {
            Operation::Insert {
                id,
                internal_id,
                level,
                vector,
                metadata,
                sparse,
            } => self
                .apply_insert(
                    id,
                    internal_id,
                    level,
                    vector,
                    metadata.into_iter().collect(),
                    sparse,
                )
                .and_then(|()| self.maybe_trigger_training(&writers).map_err(Error::Engine)),
            Operation::Remove { ids } => {
                let held = self.held_ids(&ids);
                if held.len() != ids.len() {
                    let absent = ids
                        .iter()
                        .find(|id| !held.contains(id))
                        .cloned()
                        .unwrap_or_default();
                    return Err(mismatch(format!(
                        "the record removes '{}' and the collection does not hold it",
                        absent
                    )));
                }
                let missing = self.remove_points_internal(&writers, &ids);
                debug_assert!(missing.is_empty(), "every id was held a moment ago");
                Ok(())
            }
            Operation::UpdateMetadata { id, metadata } => {
                if !self.contains(&id) {
                    return Err(mismatch(format!(
                        "the record replaces the metadata of '{}' and the collection does not \
                         hold it",
                        id
                    )));
                }
                let replaced = self.update_metadata_locked(&id, metadata.into_iter().collect());
                debug_assert!(replaced, "the record was held a moment ago");
                Ok(())
            }
            Operation::Clear => self.clear_locked().map(|(_, old)| cleared = Some(old)),
            Operation::Compact => self.compact_locked(&writers).map(|_| ()),
            Operation::Rebuild {
                m,
                expected_size,
                ef_construction,
            } => {
                let fits = |value: u64, what: &str| {
                    usize::try_from(value).map_err(|_| {
                        mismatch(format!(
                            "the record's {} of {} is above what this build addresses",
                            what, value
                        ))
                    })
                };
                let m = fits(m, "m")?;
                let expected_size = fits(expected_size, "expected_size")?;
                let ef_construction = fits(ef_construction, "ef_construction")?;
                validate_index_parameters(
                    self.dense().dim,
                    &self.dense().metric,
                    m,
                    ef_construction,
                    expected_size,
                    "",
                )?;
                self.rebuild_locked(&writers, m, expected_size, ef_construction)
                    .map(|_| ())
            }
            Operation::Intern { term_id, term } => self.intern_recorded(term_id, &term),
            Operation::Train { completed_at } => {
                if !self.can_use_quantization() {
                    return Err(mismatch(
                        "the record names a training and the collection's codebook is not \
                         fitted"
                            .to_string(),
                    ));
                }
                *self.dense().training_completed_at.write().unwrap() = Some(completed_at);
                Ok(())
            }
            Operation::AddMetadata { pairs } => {
                self.add_metadata_locked(pairs.into_iter().collect());
                Ok(())
            }
            Operation::RebuildQuantized => self
                .rebuild_with_quantization_locked(&writers)
                .map(|_| ())
                .map_err(Error::Engine),
            // The reader settles a call by its framing records and hands
            // none of them on, so one reaching here was built by hand.
            Operation::Commit { first } => Err(mismatch(format!(
                "the record is a commit naming sequence {}, which closes a call and is not an \
                 operation",
                first
            ))),
            Operation::Abort { first, .. } => Err(mismatch(format!(
                "the record is an abort naming sequence {}, which withdraws a call and is not an \
                 operation",
                first
            ))),
        };
        debug_assert!(
            self.positions_agree(),
            "an applied record leaves every structure at the positions the records hold"
        );
        applied
    }

    /// Apply one record of a call whose commit failed, as the live
    /// collection left it.
    ///
    /// Under the mutation guard, as `apply` is. An `Insert` issues the
    /// internal id it names, which must be the one the counter would issue
    /// next, and installs nothing. An `Intern` interns its term, held as
    /// `apply` holds it. Every other kind is passed over. A framing record is
    /// refused, as `apply` refuses one.
    pub fn apply_withdrawn(&self, operation: Operation) -> Result<(), Error> {
        let _writers = self.hold_writers();
        if self.sink_attached() {
            return Err(Error::Engine(
                "A recorded operation cannot be applied while a sink is attached, since a \
                 replay is not a new mutation. Detach the sink, apply, and attach it again."
                    .to_string(),
            ));
        }
        let mismatch = |detail: String| Error::JournalReplayMismatch { detail };
        match operation {
            Operation::Insert {
                id, internal_id, ..
            } => {
                let expected = usize::try_from(internal_id)
                    .ok()
                    .filter(|&value| value <= MAX_INTERNAL_ID)
                    .ok_or_else(|| {
                        mismatch(format!(
                            "the withdrawn record names internal id {} for '{}', above the id \
                             ceiling",
                            internal_id, id
                        ))
                    })?;
                self.issue_expected_id(expected).map(|_| ())
            }
            Operation::Intern { term_id, term } => self.intern_recorded(term_id, &term),
            Operation::Commit { first } | Operation::Abort { first, .. } => Err(mismatch(format!(
                "the withdrawn record is a framing record naming sequence {}, which is not an \
                 operation",
                first
            ))),
            Operation::Remove { .. }
            | Operation::UpdateMetadata { .. }
            | Operation::Clear
            | Operation::Compact
            | Operation::Rebuild { .. }
            | Operation::Train { .. }
            | Operation::AddMetadata { .. }
            | Operation::RebuildQuantized => Ok(()),
        }
    }

    /// The `Intern` arm of `apply` and of `apply_withdrawn`: the term id
    /// must be the one the dictionary would issue and the term must be new
    /// to it. Then the interning.
    fn intern_recorded(&self, term_id: u32, term: &str) -> Result<(), Error> {
        let mismatch = |detail: String| Error::JournalReplayMismatch { detail };
        let layer = self.text_layer()?;
        let mut dictionary = layer.dictionary.write().unwrap();
        let next = dictionary.next_id().unwrap_or(u32::MAX);
        if term_id != next {
            return Err(mismatch(format!(
                "the record interns '{}' at {} and the dictionary would issue {}",
                term, term_id, next
            )));
        }
        if let Some(held) = dictionary.id_of(term) {
            return Err(mismatch(format!(
                "the record interns '{}' at {} and the dictionary holds it at {}",
                term, term_id, held
            )));
        }
        let issued = dictionary.intern(term)?;
        debug_assert_eq!(issued, term_id, "the dictionary issued the next id");
        Ok(())
    }

    /// The `Insert` arm of `apply`, with the mutation guard held, up to the
    /// training trigger, which the arm sets off once this returns.
    fn apply_insert(
        &self,
        id: String,
        internal_id: u64,
        level: u8,
        vector: Vec<f32>,
        metadata: HashMap<String, serde_json::Value>,
        sparse: Option<SparseVector>,
    ) -> Result<(), Error> {
        let mismatch = |detail: String| Error::JournalReplayMismatch { detail };

        // The internal id, held to the last one the index's id store holds
        // before it is compared with the counter, so a record naming an id
        // the store refuses is refused here, before anything is issued.
        let expected = usize::try_from(internal_id)
            .ok()
            .filter(|&value| value <= MAX_INTERNAL_ID)
            .ok_or_else(|| {
                mismatch(format!(
                    "the record names internal id {} for '{}', above the id ceiling",
                    internal_id, id
                ))
            })?;
        if self.contains(&id) {
            return Err(mismatch(format!(
                "the record inserts '{}' and the collection already holds it",
                id
            )));
        }

        // The vector, as `add` holds it at its door.
        if vector.len() != self.dense().dim {
            return Err(Error::VectorDimension {
                expected: self.dense().dim,
                got: vector.len(),
            });
        }
        if let Some((index, &value)) = vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Error::VectorNotFinite { index, value });
        }

        // The sparse half, as admission holds it.
        if let Some(sparse) = &sparse {
            let space = self.sparse().ok_or(Error::NoSparseSpace)?;
            space.config().weighting.validate_record(sparse.as_ref())?;
        }

        // The id, held to what the id store can name, as admission holds it.
        let arena = self.ids.read().unwrap().arena_len();
        zeusdb_vector_core::refuse_unnamable(&id, arena)?;

        // The id the record names, issued only if it is the next, and the
        // level the record names, with nothing drawn.
        let issued = self.issue_expected_id(expected)?;
        self.install(Admitted {
            id,
            vector,
            sparse,
            metadata,
            internal_id: issued,
            level: level as usize,
        })?;
        {
            let mut count = self.vector_count.lock().unwrap();
            *count += 1;
        }
        Ok(())
    }
}
