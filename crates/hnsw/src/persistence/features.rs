//! The features a directory lists in `manifest.json`.
//!
//! A feature is a part of a saved directory that a build can be made
//! without: a quantizer, a sparse space and its text layer, a journal, an
//! identity. Every save lists the features the directory holds under
//! `features`, each with a mark, and a load reads the list before any
//! artefact. A build that meets a feature it does not know opens the
//! directory where the mark is `compatible`, meaning a build without the
//! feature opens the directory correctly and its save writes a correct
//! directory that no longer holds it. Where the mark is `read_only` it opens
//! the directory read only, meaning a build without the feature reads every
//! record correctly and a directory it wrote would not hold the feature, so
//! the collection refuses every change, every save, a checkpoint, a journal
//! and a backup. Any other mark, `incompatible` among them, is refused,
//! naming the feature. The marks are read at every major.
//!
//! | Name | Mark | Listed where |
//! | --- | --- | --- |
//! | `identity` | compatible | a 4.x save, which records `identity`; part of the base at 5.x and never listed there |
//! | `int8` | incompatible | `quantization.json` takes the scalar layout, trained or not |
//! | `journal` | incompatible | the manifest names a journal |
//! | `pq` | incompatible | `quantization.json` takes the product quantized layout, trained or not |
//! | `sparse` | incompatible | `config.json` declares a sparse space |
//! | `text` | incompatible | that space declares a tokenizer |
//!
//! The base every 5.x directory shares lists nothing: the manifest with its
//! identity, `config.json`, `metadata.json`, the framed `mappings.bin`, the
//! framed `vectors.bin` where raw vectors are held, and the graph dump. A
//! field those files default when it is absent is part of the base too. A
//! 5.x manifest carries the record even where it lists nothing, and one
//! carrying none is refused. The 4.x base is the same without the identity,
//! which a 4.x save lists.
//!
//! **A name and its mark are on disk. Never reuse a name and never change a
//! mark.** A later feature is a new name, listed only by a directory that
//! holds it, and a later change to a feature is a new name as well, so a
//! directory that holds neither opens on every build that reads its base.

use std::collections::BTreeMap;

use tracing::warn;
use zeusdb_vector_core::Error;

use super::LOG_TARGET;

/// A build that does not know the feature may open the directory without it.
const COMPATIBLE: &str = "compatible";

/// A build that does not know the feature must refuse the directory.
const INCOMPATIBLE: &str = "incompatible";

/// A build that does not know the feature may read the directory and must
/// change nothing.
const READ_ONLY: &str = "read_only";

/// Every feature this build knows and the mark it gives each, in name order.
const KNOWN: [(&str, &str); 6] = [
    ("identity", COMPATIBLE),
    ("int8", INCOMPATIBLE),
    ("journal", INCOMPATIBLE),
    ("pq", INCOMPATIBLE),
    ("sparse", INCOMPATIBLE),
    ("text", INCOMPATIBLE),
];

/// The features this build knows, as a refusal spells them.
const KNOWN_LABEL: &str = "identity, int8, journal, pq, sparse and text";

/// The mark this build gives each, as a refusal spells them.
const MARKS_LABEL: &str = "identity compatible and int8, journal, pq, sparse and text incompatible";

/// The features a build knows with the mark it gives each, in name order,
/// and whether it reads the `read_only` mark on a feature it does not know.
pub(super) struct Table {
    known: &'static [(&'static str, &'static str)],
    known_label: &'static str,
    reads_read_only: bool,
}

/// This build's table.
pub(super) const THIS_BUILD: Table = Table {
    known: &KNOWN,
    known_label: KNOWN_LABEL,
    reads_read_only: true,
};

impl Table {
    /// The mark this table gives `name`, where it knows the feature.
    fn mark_of(&self, name: &str) -> Option<&'static str> {
        self.known
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, mark)| *mark)
    }

    /// Whether a feature this table does not know, under `mark`, opens.
    fn opens(&self, mark: &str) -> bool {
        mark == COMPATIBLE || (self.reads_read_only && mark == READ_ONLY)
    }
}

/// What reading a features record found: the features this build does not
/// know whose mark is `read_only`, in name order, which make the collection
/// read only.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Reading {
    pub(super) read_only: Vec<String>,
}

/// What a directory holds, of the features this build knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Held {
    pub(super) identity: bool,
    pub(super) int8: bool,
    pub(super) journal: bool,
    pub(super) pq: bool,
    pub(super) sparse: bool,
    pub(super) text: bool,
}

impl Held {
    /// The names held, in name order.
    fn names(&self) -> Vec<&'static str> {
        [
            ("identity", self.identity),
            ("int8", self.int8),
            ("journal", self.journal),
            ("pq", self.pq),
            ("sparse", self.sparse),
            ("text", self.text),
        ]
        .into_iter()
        .filter_map(|(name, held)| held.then_some(name))
        .collect()
    }

    /// The record a save writes for a directory holding these, each name
    /// under the mark this build gives it.
    pub(super) fn record(&self) -> BTreeMap<String, String> {
        self.names()
            .into_iter()
            .filter_map(|name| mark_of(name).map(|mark| (name.to_string(), mark.to_string())))
            .collect()
    }
}

/// The mark this build gives `name`, where it knows the feature.
fn mark_of(name: &str) -> Option<&'static str> {
    THIS_BUILD.mark_of(name)
}

/// Read a features record against `table`, before any artefact and on any
/// major.
///
/// Every feature the table does not know whose mark does not open is
/// refused, all of them named in name order, since the build cannot tell
/// what opening the directory without them would lose. Then a feature the
/// table knows listed under another mark than its own is refused, since no
/// save writes one. A feature the table does not know whose mark is
/// `compatible` is logged, and the directory opens without it. One whose
/// mark is `read_only` is handed back, and the directory opens read only.
pub(super) fn read(record: &BTreeMap<String, String>, table: &Table) -> Result<Reading, Error> {
    let unsupported: Vec<String> = record
        .iter()
        .filter(|(name, mark)| table.mark_of(name).is_none() && !table.opens(mark))
        .map(|(name, _)| name.clone())
        .collect();
    if !unsupported.is_empty() {
        return Err(Error::FeatureUnsupported {
            features: unsupported,
            known: table.known_label,
        });
    }
    if let Some((name, mark)) = record
        .iter()
        .find(|(name, mark)| table.mark_of(name).is_some_and(|own| own != mark.as_str()))
    {
        return Err(invalid(format!("it marks {} {}", name, spelled(mark))));
    }
    let mut reading = Reading::default();
    for (name, mark) in record
        .iter()
        .filter(|(name, _)| table.mark_of(name).is_none())
    {
        if mark == READ_ONLY {
            reading.read_only.push(name.clone());
            continue;
        }
        warn!(target: LOG_TARGET, operation = "load", feature = name.as_str(),
            "manifest.json lists the feature '{}', which this build does not know and the \
             manifest marks compatible, so the directory opens without it, and a save from \
             this build will not list it",
            name
        );
    }
    Ok(reading)
}

/// Whether a features record lists a feature this build does not know whose
/// mark is `read_only`, read with no refusal and no record logged, for the
/// lock a load takes before the load proper reads the record.
pub(super) fn lists_read_only(record: &BTreeMap<String, String>) -> bool {
    record
        .iter()
        .any(|(name, mark)| mark_of(name).is_none() && mark == READ_ONLY)
}

/// Hold a features record to what the directory holds: of the features this
/// build knows, the record lists exactly the ones the directory holds.
pub(super) fn check_held(record: &BTreeMap<String, String>, held: Held) -> Result<(), Error> {
    let listed: Vec<&str> = record
        .keys()
        .map(String::as_str)
        .filter(|name| mark_of(name).is_some())
        .collect();
    let holds = held.names();
    if listed == holds {
        return Ok(());
    }
    Err(invalid(format!(
        "it lists {}, and the directory holds {}",
        sentence(&listed),
        sentence(&holds)
    )))
}

fn invalid(detail: String) -> Error {
    Error::FeaturesInvalid {
        detail,
        marks: MARKS_LABEL,
    }
}

/// The refusal a 5.x manifest carrying no features record meets, since
/// every save at 5.x lists its features, possibly none.
pub(super) fn missing() -> Error {
    invalid("it carries no features record, which every manifest at format 5.x carries".to_string())
}

/// A mark as a refusal spells it: the two this build writes bare, any other
/// quoted.
fn spelled(mark: &str) -> String {
    if mark == COMPATIBLE || mark == INCOMPATIBLE {
        mark.to_string()
    } else {
        format!("'{}'", mark)
    }
}

/// Names joined as a sentence lists them, or `no feature`.
fn sentence(names: &[&str]) -> String {
    match names.split_last() {
        None => "no feature".to_string(),
        Some((last, [])) => last.to_string(),
        Some((last, rest)) => format!("{} and {}", rest.join(", "), last),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, mark)| (name.to_string(), mark.to_string()))
            .collect()
    }

    /// The table, the two labels and the order the names are kept in agree,
    /// so a refusal never names a feature or a mark the table does not.
    #[test]
    fn the_table_and_its_labels_agree() {
        let names: Vec<&str> = KNOWN.iter().map(|(name, _)| *name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "the table is in name order");
        assert_eq!(sentence(&names), KNOWN_LABEL);
        let compatible: Vec<&str> = KNOWN
            .iter()
            .filter(|(_, mark)| *mark == COMPATIBLE)
            .map(|(name, _)| *name)
            .collect();
        let incompatible: Vec<&str> = KNOWN
            .iter()
            .filter(|(_, mark)| *mark == INCOMPATIBLE)
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            format!(
                "{} compatible and {} incompatible",
                sentence(&compatible),
                sentence(&incompatible)
            ),
            MARKS_LABEL
        );
        let every = Held {
            identity: true,
            int8: true,
            journal: true,
            pq: true,
            sparse: true,
            text: true,
        };
        assert_eq!(every.names(), names);
        assert_eq!(every.record(), record(&KNOWN));
    }

    /// A record naming only what this build knows, under its own marks, is
    /// read, and so is an empty one, with no feature making it read only.
    #[test]
    fn a_record_of_known_features_is_read() {
        let read = |record: &BTreeMap<String, String>| read(record, &THIS_BUILD).unwrap();
        assert_eq!(read(&record(&[])), Reading::default());
        assert_eq!(read(&record(&KNOWN)), Reading::default());
        assert_eq!(
            read(&record(&[
                ("identity", "compatible"),
                ("pq", "incompatible")
            ])),
            Reading::default()
        );
    }

    /// Every feature this build does not know whose mark neither opens nor
    /// reads is refused, named in name order, whatever the mark.
    #[test]
    fn an_unknown_feature_not_marked_compatible_or_read_only_is_refused_by_name() {
        for mark in ["incompatible", "", "Compatible", "Read_only", "read-only"] {
            match read(
                &record(&[("identity", "compatible"), ("later", mark)]),
                &THIS_BUILD,
            ) {
                Err(Error::FeatureUnsupported { features, known }) => {
                    assert_eq!(features, vec!["later".to_string()], "{mark:?}");
                    assert_eq!(known, KNOWN_LABEL);
                }
                other => panic!("{mark:?}: expected a refusal, got {other:?}"),
            }
        }
        match read(
            &record(&[
                ("zeta", "incompatible"),
                ("alpha", "incompatible"),
                ("note", "compatible"),
                ("probe", "read_only"),
            ]),
            &THIS_BUILD,
        ) {
            Err(Error::FeatureUnsupported { features, .. }) => {
                assert_eq!(features, vec!["alpha".to_string(), "zeta".to_string()])
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// A record listing `probe` as `read_only` is read as the feature's own
    /// by a table that knows it, as a read-only feature by this build's
    /// table, and refused by name by a table that does not read the mark,
    /// which is the rule a build before the mark follows.
    #[test]
    fn the_read_only_mark_is_read_by_the_table_that_reads_it() {
        let listed = record(&[("identity", "compatible"), ("probe", "read_only")]);
        let knowing = Table {
            known: &[("identity", COMPATIBLE), ("probe", READ_ONLY)],
            known_label: "identity and probe",
            reads_read_only: true,
        };
        assert_eq!(read(&listed, &knowing).unwrap(), Reading::default());
        assert_eq!(
            read(&listed, &THIS_BUILD).unwrap(),
            Reading {
                read_only: vec!["probe".to_string()]
            }
        );
        assert!(lists_read_only(&listed));
        assert!(!lists_read_only(&record(&KNOWN)));
        let before = Table {
            known: &KNOWN,
            known_label: KNOWN_LABEL,
            reads_read_only: false,
        };
        match read(&listed, &before) {
            Err(Error::FeatureUnsupported { features, known }) => {
                assert_eq!(features, vec!["probe".to_string()]);
                assert_eq!(known, KNOWN_LABEL);
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        let two = record(&[("beta", "read_only"), ("alpha", "read_only")]);
        assert_eq!(
            read(&two, &THIS_BUILD).unwrap().read_only,
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    /// An unknown feature marked compatible is read, and a known feature
    /// under another mark than its own is refused, naming the mark.
    #[test]
    fn marks_are_held_to_the_table() {
        assert_eq!(
            read(
                &record(&[("identity", "compatible"), ("note", "compatible")]),
                &THIS_BUILD
            )
            .unwrap(),
            Reading::default()
        );
        let refused = |entries: &[(&str, &str)]| match read(&record(entries), &THIS_BUILD) {
            Err(Error::FeaturesInvalid { detail, marks }) => {
                assert_eq!(marks, MARKS_LABEL);
                detail
            }
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert_eq!(
            refused(&[("identity", "incompatible")]),
            "it marks identity incompatible"
        );
        assert_eq!(
            refused(&[("journal", "compatible")]),
            "it marks journal compatible"
        );
        assert_eq!(refused(&[("pq", "read_only")]), "it marks pq 'read_only'");
    }

    /// The known features listed must be the ones held, both ways, and a
    /// feature this build does not know takes no part.
    #[test]
    fn a_record_lists_the_held_features_and_no_other() {
        let held = Held {
            identity: true,
            pq: true,
            ..Held::default()
        };
        assert!(check_held(&held.record(), held).is_ok());
        let mut with_note = held.record();
        with_note.insert("note".to_string(), "compatible".to_string());
        assert!(check_held(&with_note, held).is_ok());
        let detail = |listed: &[(&str, &str)], held: Held| match check_held(&record(listed), held) {
            Err(Error::FeaturesInvalid { detail, .. }) => detail,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert_eq!(
            detail(&[("identity", "compatible")], held),
            "it lists identity, and the directory holds identity and pq"
        );
        assert_eq!(
            detail(&[], held),
            "it lists no feature, and the directory holds identity and pq"
        );
        assert_eq!(
            detail(
                &[
                    ("identity", "compatible"),
                    ("journal", "incompatible"),
                    ("pq", "incompatible")
                ],
                held
            ),
            "it lists identity, journal and pq, and the directory holds identity and pq"
        );
        assert_eq!(
            detail(&[("identity", "compatible")], Held::default()),
            "it lists identity, and the directory holds no feature"
        );
    }
}
