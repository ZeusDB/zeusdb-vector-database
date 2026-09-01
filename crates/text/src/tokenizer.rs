//! The tokenizer trait, the one built-in implementation, and the value a
//! space records its tokenizer as.

use std::fmt;

use serde::{Deserialize, Serialize};
use zeusdb_vector_core::Error;

/// How a space's tokenizer is declared, as a value.
///
/// The variants are what an index can write down and read back so that a
/// query is tokenized the way the records were. A caller's own
/// implementation has no such value, and is recorded as `External`.
///
/// Written into `config.json` as `"simple"` or `"external"`. An index that
/// recorded `external` is opened with the implementation handed to it or
/// not at all.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerConfig {
    /// [`SimpleTokenizer`]. It has no parameters, so the variant is the
    /// whole of its value.
    Simple,
    /// An implementation the caller supplied. An index that used one must be
    /// handed the same implementation again when it is opened, since nothing
    /// here can reproduce it.
    External,
}

impl TokenizerConfig {
    /// The value's name, as a declaration or a message spells it.
    pub fn name(&self) -> &'static str {
        match self {
            TokenizerConfig::Simple => "simple",
            TokenizerConfig::External => "external",
        }
    }
}

/// Splits a text into terms.
///
/// One method, and it hands each term to a closure rather than returning a
/// list, so an implementation that borrows its terms from the text allocates
/// nothing. Terms are handed over in order and a repeated term is handed
/// over each time it occurs, since the count is what a term frequency space
/// stores. An empty term is dropped by the caller.
///
/// A record's terms and a query's terms come through the same method, so a
/// query is tokenized as the records were.
///
/// An implementation may fail, and what it returns reaches the caller of the
/// operation that ran it. The built-in tokenizer never fails. A caller's own
/// carries whatever it raised in [`Error::TokenizerFailed`], so a binding can
/// hand the caller their own failure back.
///
/// The engine never calls this with one of its guards held. A caller's
/// implementation may need something of the caller's to run, such as an
/// interpreter lock, and a thread holding that while waiting for a guard the
/// tokenizing thread holds would wait forever. So the collection tokenizes
/// first, under nothing, and counts the terms it collected under the
/// dictionary's guard afterwards.
pub trait Tokenizer: Send + Sync {
    /// Every term of `text`, in order, repeats included.
    fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error>;

    /// What this tokenizer is declared as. A caller's implementation is
    /// `External` unless it says otherwise.
    fn config(&self) -> TokenizerConfig {
        TokenizerConfig::External
    }
}

impl fmt::Debug for dyn Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tokenizer({})", self.config().name())
    }
}

/// The built-in tokenizer. A term is a maximal run of characters that are
/// letters or digits, lowercased. Everything else separates terms and is
/// dropped.
///
/// Letters, digits and case are what the standard library says they are for
/// the whole of Unicode, so a script with case is lowercased and a script
/// without spaces between words, which has no character that is neither a
/// letter nor a digit, arrives as one term per run of text. A lowercase
/// mapping can be longer than the character it maps, which is why a term
/// that is not already lowercase is built in a buffer rather than sliced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
        let mut lowered = String::new();
        for run in text.split(|c: char| !c.is_alphanumeric()) {
            if run.is_empty() {
                continue;
            }
            if run.chars().all(|c| !c.is_uppercase()) {
                term(run);
            } else {
                lowered.clear();
                lowered.extend(run.chars().flat_map(char::to_lowercase));
                term(&lowered);
            }
        }
        Ok(())
    }

    fn config(&self) -> TokenizerConfig {
        TokenizerConfig::Simple
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        SimpleTokenizer
            .tokenize(text, &mut |t| out.push(t.to_string()))
            .unwrap();
        out
    }

    /// Runs of letters and digits are terms, lowercased, and everything
    /// between them is dropped.
    #[test]
    fn the_built_in_tokenizer_splits_lowercases_and_nothing_else() {
        assert_eq!(
            terms("The quick, brown fox!"),
            ["the", "quick", "brown", "fox"]
        );
        assert_eq!(terms("  "), Vec::<String>::new());
        assert_eq!(terms("x2 X2"), ["x2", "x2"]);
        // Repeats are handed over each time.
        assert_eq!(terms("a a A"), ["a", "a", "a"]);
        // Case is Unicode case.
        assert_eq!(terms("Straße ÉCOLE"), ["straße", "école"]);
        assert_eq!(SimpleTokenizer.config(), TokenizerConfig::Simple);
    }

    /// What it does to text that is not plain prose, held here so the limit
    /// is stated rather than discovered.
    #[test]
    fn the_built_in_tokenizer_has_the_limits_it_declares() {
        // A URL is its pieces.
        assert_eq!(
            terms("https://example.org/a-b?x=1"),
            ["https", "example", "org", "a", "b", "x", "1"]
        );
        // An email address is its pieces.
        assert_eq!(terms("ross@example.org"), ["ross", "example", "org"]);
        // A hyphenated word is two terms, and a contraction is two.
        assert_eq!(
            terms("state-of-the-art don't"),
            ["state", "of", "the", "art", "don", "t"]
        );
        // An underscore separates, since it is neither a letter nor a digit.
        assert_eq!(terms("snake_case"), ["snake", "case"]);
        // A script written without spaces is one term per run.
        assert_eq!(terms("東京は日本の首都です。"), ["東京は日本の首都です"]);
        // No stemming and no stopwords.
        assert_eq!(
            terms("the cats are running"),
            ["the", "cats", "are", "running"]
        );
    }

    /// A caller's implementation is external unless it says otherwise.
    #[test]
    fn a_callers_tokenizer_is_external_by_default() {
        struct Whitespace;
        impl Tokenizer for Whitespace {
            fn tokenize(&self, text: &str, term: &mut dyn FnMut(&str)) -> Result<(), Error> {
                text.split_whitespace().for_each(term);
                Ok(())
            }
        }
        assert_eq!(Whitespace.config(), TokenizerConfig::External);
        let boxed: Box<dyn Tokenizer> = Box::new(Whitespace);
        assert_eq!(format!("{:?}", boxed), "Tokenizer(external)");
    }
}
