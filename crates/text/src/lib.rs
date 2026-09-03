//! The text layer, above the sparse index and below the collection.
//!
//! A sparse space stores term ids and weights and never sees a string. What
//! turns a string into those is here: a [`Tokenizer`] that splits a text
//! into terms, a [`TermDictionary`] that gives each distinct term a stable
//! id, and the functions in `vectorize` that count a text's terms into a
//! sparse vector, one for a record and one for a query, each in a one step
//! form that runs the tokenizer and a two step form that takes terms already
//! tokenized, so the tokenizer can run under no guard.
//!
//! # The pre-tokenized path is the primary route
//!
//! A caller that runs its own encoder, or its own tokenizer, hands the sparse
//! space `(term id, weight)` pairs directly and never touches this crate. The
//! sparse index does not depend on it, so a record that arrives as ids pays
//! nothing for the existence of the text layer.
//!
//! # What the built-in tokenizer does, and what it does not
//!
//! [`SimpleTokenizer`] splits where a character is neither a letter nor a
//! digit, lowercases what is left, and does nothing else. It knows no
//! language. It does not stem, it drops no stopword, it forms no n-gram, it
//! does not segment a script written without spaces, and it does not treat a
//! URL, an email address or a hyphenated word as one term. A caller who needs
//! any of that implements [`Tokenizer`], which is one method. Nothing of that
//! kind is added here, because there is no point at which it would be
//! complete, and the engine's users already run an embedding pipeline that
//! owns their text.
//!
//! # The declaration is a value
//!
//! [`TokenizerConfig`] says what tokenizer a space was declared with, as a
//! value rather than as a name looked up in a registry, so an index opened
//! later tokenizes a query the way its records were tokenized. A tokenizer
//! the caller supplied has no value the engine can write down, and is
//! recorded as such: an index that used one must be handed the same
//! implementation again when it is opened.
#![warn(unreachable_pub)]

mod dictionary;
mod tokenizer;
mod vectorize;

pub use dictionary::TermDictionary;
pub use tokenizer::{SimpleTokenizer, Tokenizer, TokenizerConfig};
pub use vectorize::{
    count_query, count_record, count_record_with, tokenize, vectorize_query, vectorize_record,
};
