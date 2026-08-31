//! A text counted into a sparse vector.

use zeusdb_vector_core::{Error, SparseVector};

use crate::dictionary::TermDictionary;
use crate::tokenizer::Tokenizer;

/// The ids sorted and run-length counted into a vector whose dimensions are
/// strictly increasing, which is what the sparse index requires.
fn count(mut ids: Vec<u32>) -> SparseVector {
    ids.sort_unstable();
    let mut dims: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    for id in ids {
        match dims.last() {
            Some(&last) if last == id => {
                *values.last_mut().expect("a dimension has a value") += 1.0;
            }
            _ => {
                dims.push(id);
                values.push(1.0);
            }
        }
    }
    SparseVector { dims, values }
}

/// A record's text as term ids and term frequencies, every new term given
/// an id. An empty term is dropped.
pub fn vectorize_record(
    tokenizer: &dyn Tokenizer,
    dictionary: &mut TermDictionary,
    text: &str,
) -> Result<SparseVector, Error> {
    let mut ids: Vec<u32> = Vec::new();
    let mut failed: Option<Error> = None;
    tokenizer.tokenize(text, &mut |term| {
        if term.is_empty() || failed.is_some() {
            return;
        }
        match dictionary.intern(term) {
            Ok(id) => ids.push(id),
            Err(e) => failed = Some(e),
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(count(ids)),
    }
}

/// A query's text as term ids and term frequencies. A term no record has
/// carried has no id and is dropped, so a query issues no ids.
pub fn vectorize_query(
    tokenizer: &dyn Tokenizer,
    dictionary: &TermDictionary,
    text: &str,
) -> SparseVector {
    let mut ids: Vec<u32> = Vec::new();
    tokenizer.tokenize(text, &mut |term| {
        if let Some(id) = dictionary.id_of(term) {
            ids.push(id);
        }
    });
    count(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SimpleTokenizer;

    /// A record's terms are counted under ids in arrival order, the vector
    /// is sorted by id, and a query finds the same ids without issuing any.
    #[test]
    fn a_record_is_counted_and_a_query_looks_up_without_interning() {
        let mut d = TermDictionary::new();
        let v = vectorize_record(&SimpleTokenizer, &mut d, "The fox, the dog. THE end").unwrap();
        assert_eq!(v.dims, vec![0, 1, 2, 3]);
        assert_eq!(v.values, vec![3.0, 1.0, 1.0, 1.0]);
        assert_eq!(d.terms(), ["the", "fox", "dog", "end"]);
        assert!(v.as_ref().validate().is_ok());

        let q = vectorize_query(&SimpleTokenizer, &d, "dog dog cat THE");
        assert_eq!(q.dims, vec![0, 2]);
        assert_eq!(q.values, vec![1.0, 2.0]);
        assert_eq!(d.len(), 4, "a query issues no id");

        let empty = vectorize_record(&SimpleTokenizer, &mut d, " ,, ").unwrap();
        assert!(empty.dims.is_empty());
    }
}
