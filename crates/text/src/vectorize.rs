//! A text counted into a sparse vector.
//!
//! Two forms of each direction. `vectorize_record` and `vectorize_query`
//! run the tokenizer and count in one pass, allocating no term, for a
//! caller holding a tokenizer it can run anywhere. `tokenize` collects the
//! terms so that a caller can run the tokenizer under nothing and hand the
//! terms to `count_record` or `count_query` under the dictionary's guard,
//! which is what the collection does, since a tokenizer may be the caller's
//! own and need what the caller needs to run.

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

/// Every term of `text` as the tokenizer hands them over, in order and
/// repeats included, an empty term dropped. Collected, so the tokenizer can
/// run under no guard and the terms be counted under one afterwards.
pub fn tokenize(tokenizer: &dyn Tokenizer, text: &str) -> Result<Vec<String>, Error> {
    let mut terms: Vec<String> = Vec::new();
    tokenizer.tokenize(text, &mut |term| {
        if !term.is_empty() {
            terms.push(term.to_string());
        }
    })?;
    Ok(terms)
}

/// Terms already tokenized, counted as a record's: every new term is given
/// an id and an empty term is dropped. What `vectorize_record` does after
/// the tokenizer has run.
pub fn count_record<I, S>(dictionary: &mut TermDictionary, terms: I) -> Result<SparseVector, Error>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut ids: Vec<u32> = Vec::new();
    for term in terms {
        let term = term.as_ref();
        if term.is_empty() {
            continue;
        }
        ids.push(dictionary.intern(term)?);
    }
    Ok(count(ids))
}

/// Terms already tokenized, counted as a query's: a term no record has
/// carried has no id and is dropped, so a query issues no ids. What
/// `vectorize_query` does after the tokenizer has run.
pub fn count_query<I, S>(dictionary: &TermDictionary, terms: I) -> SparseVector
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let ids: Vec<u32> = terms
        .into_iter()
        .filter_map(|term| dictionary.id_of(term.as_ref()))
        .collect();
    count(ids)
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
    })?;
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
) -> Result<SparseVector, Error> {
    let mut ids: Vec<u32> = Vec::new();
    tokenizer.tokenize(text, &mut |term| {
        if let Some(id) = dictionary.id_of(term) {
            ids.push(id);
        }
    })?;
    Ok(count(ids))
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

        let q = vectorize_query(&SimpleTokenizer, &d, "dog dog cat THE").unwrap();
        assert_eq!(q.dims, vec![0, 2]);
        assert_eq!(q.values, vec![1.0, 2.0]);
        assert_eq!(d.len(), 4, "a query issues no id");

        let empty = vectorize_record(&SimpleTokenizer, &mut d, " ,, ").unwrap();
        assert!(empty.dims.is_empty());
    }

    /// The two step forms agree with the one step forms, term for term
    /// and id for id, and a tokenizer's failure comes back from both.
    #[test]
    fn the_two_step_forms_agree_with_the_one_step_forms() {
        let text = "The fox, the dog. THE end";
        let terms = tokenize(&SimpleTokenizer, text).unwrap();
        assert_eq!(terms, ["the", "fox", "the", "dog", "the", "end"]);
        assert!(tokenize(&SimpleTokenizer, " ,, ").unwrap().is_empty());

        let mut one = TermDictionary::new();
        let mut two = TermDictionary::new();
        let by_text = vectorize_record(&SimpleTokenizer, &mut one, text).unwrap();
        let by_terms = count_record(&mut two, &terms).unwrap();
        assert_eq!(by_text, by_terms);
        assert_eq!(one.terms(), two.terms());
        assert_eq!(
            count_record(&mut two, ["", "fox", ""]).unwrap().dims,
            vec![1]
        );

        let query = "dog dog cat THE";
        assert_eq!(
            vectorize_query(&SimpleTokenizer, &one, query).unwrap(),
            count_query(&two, tokenize(&SimpleTokenizer, query).unwrap())
        );
        assert_eq!(two.len(), 4);

        struct Broken;
        impl Tokenizer for Broken {
            fn tokenize(&self, _text: &str, _term: &mut dyn FnMut(&str)) -> Result<(), Error> {
                Err(Error::TokenizerFailed("no terms today".into()))
            }
        }
        let failed = tokenize(&Broken, "anything").unwrap_err();
        assert!(matches!(failed, Error::TokenizerFailed(_)));
        assert_eq!(failed.to_string(), "The tokenizer raised no terms today");
        assert!(vectorize_record(&Broken, &mut one, "x").is_err());
        assert!(vectorize_query(&Broken, &one, "x").is_err());
    }
}
