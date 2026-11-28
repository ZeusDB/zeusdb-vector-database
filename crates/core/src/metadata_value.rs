//! A record's metadata read from JSON to the depth the binding takes.
//!
//! The binding refuses a metadata value nested past [`METADATA_MAX_DEPTH`]
//! levels, counting the record's mapping as level one and each value inside a
//! list or a mapping as one level below it. `serde_json`'s own limit counts the
//! containers of the whole text instead, and `metadata.json` wraps a record's
//! mapping in the file's object while a journal record holds the mapping
//! alone, so that limit refuses values the binding takes. The read of
//! `metadata.json` and the journal's read of a record's metadata turn that
//! limit off and read every value, kept or let go, through [`MetadataValue`].
//! It counts the levels the binding counts and refuses a value past the bound
//! before it reads it, so every value the binding takes comes back, and no
//! text makes either read recurse past the bound however deep it nests.

use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

/// The deepest level a metadata value reaches. The record's mapping is level
/// one, and a value inside a list or a mapping is one level below the list or
/// the mapping. The binding refuses a value past it where the value is
/// written, and a read of `metadata.json` or of a journal record refuses one
/// past it where the value is read.
pub const METADATA_MAX_DEPTH: usize = 128;

/// The words a value past [`METADATA_MAX_DEPTH`] is refused in, being
/// `serde_json`'s own for its limit.
const TOO_DEEP: &str = "recursion limit exceeded";

/// One metadata value at `level`, read into the `Value` that `serde_json`'s
/// own reader of a `Value` builds where `keep` is set, and read and let go
/// otherwise. Every value nested in it is read through this seed one level
/// down, and a value past [`METADATA_MAX_DEPTH`] is refused before any of it
/// is read, so the read recurses once a level and no further.
#[derive(Clone, Copy, Debug)]
pub struct MetadataValue {
    level: usize,
    keep: bool,
}

impl MetadataValue {
    /// A record's mapping, at level one, kept.
    pub fn record() -> Self {
        MetadataValue {
            level: 1,
            keep: true,
        }
    }

    /// The value of one field of a record's mapping, at level two, kept
    /// where `keep` is set and read and let go otherwise.
    pub fn field(keep: bool) -> Self {
        MetadataValue { level: 2, keep }
    }

    /// The seed for a value inside this one.
    fn inner(self) -> Self {
        MetadataValue {
            level: self.level + 1,
            keep: self.keep,
        }
    }
}

impl<'de> DeserializeSeed<'de> for MetadataValue {
    type Value = Value;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        if self.level > METADATA_MAX_DEPTH {
            return Err(D::Error::custom(TOO_DEEP));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for MetadataValue {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any valid JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Number::from_f64(value).map_or(Value::Null, Value::Number))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(if self.keep {
            Value::String(value.to_owned())
        } else {
            Value::Null
        })
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(if self.keep {
            Value::String(value)
        } else {
            Value::Null
        })
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        self.deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(self.inner())? {
            if self.keep {
                items.push(item);
            }
        }
        Ok(if self.keep {
            Value::Array(items)
        } else {
            Value::Null
        })
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = map.next_value_seed(self.inner())?;
            if self.keep {
                values.insert(key, value);
            }
        }
        Ok(if self.keep {
            Value::Object(values)
        } else {
            Value::Null
        })
    }
}

/// One record's mapping read from `text` through [`MetadataValue`] with
/// `serde_json`'s own limit off, and the whole text held to it.
pub fn record_metadata(text: &[u8]) -> Result<Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(text);
    deserializer.disable_recursion_limit();
    let value = MetadataValue::record().deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `levels` lists, one inside the next, around `innermost`, as the text
    /// of a field's value.
    fn lists(levels: usize, innermost: &str) -> String {
        format!("{}{}{}", "[".repeat(levels), innermost, "]".repeat(levels))
    }

    /// The text of a record's mapping whose one field holds `value`.
    fn record(value: &str) -> String {
        format!("{{\"k\":{value}}}")
    }

    /// What the reader builds is what `serde_json`'s reader of a `Value`
    /// builds, for every kind of value and a key given twice.
    #[test]
    fn the_reader_builds_the_value_serde_json_builds() {
        for text in [
            r#"{}"#,
            r#"{"a":null,"b":true,"c":false,"d":0,"e":-1,"f":18446744073709551615}"#,
            r#"{"g":-9223372036854775808,"h":1.5,"i":-0.0,"j":1e300,"k":2.2250738585072014e-308}"#,
            r#"{"l":"text","m":"é😀","n":"esc\"aped\n","o":[],"p":{}}"#,
            r#"{"q":[1,[2,[3,{"r":[{}]}]]],"s":{"t":{"u":[null,"v"]}},"w":1,"w":2}"#,
            r#"{"x":123456789012345678901234567890,"y":0.1,"z":[1.0,2.5e-10]}"#,
        ] {
            let ours = record_metadata(text.as_bytes()).unwrap();
            let theirs: Value = serde_json::from_str(text).unwrap();
            assert_eq!(ours, theirs, "{text}");
        }
    }

    /// A value at level 128 is read, being 127 containers in a field with the
    /// innermost empty or 126 around a number, and one past it is refused in
    /// `serde_json`'s words, kept or let go.
    #[test]
    fn a_value_past_level_128_is_refused_and_one_at_it_is_read() {
        for (levels, innermost) in [(127, ""), (126, "1")] {
            let text = record(&lists(levels, innermost));
            let value = record_metadata(text.as_bytes()).unwrap();
            let expected: Value = {
                let mut de = serde_json::Deserializer::from_str(&text);
                de.disable_recursion_limit();
                serde::Deserialize::deserialize(&mut de).unwrap()
            };
            assert_eq!(value, expected);
        }
        for (levels, innermost) in [(128, ""), (127, "1")] {
            let text = record(&lists(levels, innermost));
            let refused = record_metadata(text.as_bytes()).unwrap_err();
            assert!(refused.to_string().starts_with(TOO_DEEP), "{refused}");
            let value = lists(levels, innermost);
            let mut de = serde_json::Deserializer::from_str(&value);
            de.disable_recursion_limit();
            let skipped = MetadataValue::field(false)
                .deserialize(&mut de)
                .unwrap_err();
            assert!(skipped.to_string().starts_with(TOO_DEEP), "{skipped}");
        }
    }

    /// A text nested 100,000 levels, of lists and of mappings, is refused
    /// with the words of the limit, kept or let go, and the read recurses no
    /// deeper than the bound.
    #[test]
    fn a_value_nested_100000_levels_is_refused_without_recursing_past_the_bound() {
        let deep_lists = lists(100_000, "");
        let deep_maps = format!("{}1{}", "{\"a\":".repeat(100_000), "}".repeat(100_000));
        for value in [deep_lists, deep_maps] {
            let text = record(&value);
            let refused = record_metadata(text.as_bytes()).unwrap_err();
            assert!(refused.to_string().starts_with(TOO_DEEP), "{refused}");
            let mut de = serde_json::Deserializer::from_str(&value);
            de.disable_recursion_limit();
            let skipped = MetadataValue::field(false)
                .deserialize(&mut de)
                .unwrap_err();
            assert!(skipped.to_string().starts_with(TOO_DEEP), "{skipped}");
        }
    }
}
