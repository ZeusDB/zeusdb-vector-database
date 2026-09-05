//! Scalar quantization to one signed byte a value, with one scale a
//! dimension.
//!
//! A vector of `dim` values is held as `dim` bytes and decoded as
//! `code * scale[j]`, so the codes hold a quarter of what the values held
//! and the scale array is `dim` floats once per index rather than per
//! record. The graph scores against the codes through [`Int8Dist`], which
//! decodes each value inside the kernel, so the vector is never held at
//! full width anywhere in the index; see `distance.rs`.
//!
//! # The scale
//!
//! Symmetric about zero, one scale a dimension, fitted as the largest
//! magnitude seen in that dimension over the fitting sample divided by 127.
//! Every value in the sample then rounds to a code in `-127..=127`, and a
//! value seen later that sits beyond the range saturates at the edge. The
//! code `-128` is never written, so the codes are symmetric and a negated
//! vector encodes to negated codes.
//!
//! A dimension whose sample holds only zeros gets a scale of one, so that
//! it decodes to zero and a later non-zero value in it rounds to zero as
//! well rather than saturating. That dimension carries nothing the sample
//! could see, and a scale of one is the least surprising value for it.
//!
//! # What is exact
//!
//! Decoding is one multiply, so `reconstruct` is a function of the codes and
//! the scales alone and two builds over the same sample produce the same
//! scales, the same codes and the same reconstruction. Rounding is half to
//! even, so a value that sits exactly between two codes takes the even one,
//! which is the rule `f32::round_ties_even` states.
//!
//! [`Int8Dist`]: crate::distance::Int8Dist

/// The largest magnitude a code can hold. The range is `-QMAX..=QMAX`.
const QMAX: f32 = 127.0;

/// A fitted scalar quantizer: one scale a dimension.
///
/// Immutable once built, so a kernel reads its scales without a guard. A
/// space that quantizes this way holds none until its sample is in, and
/// then holds one for the life of the index.
#[derive(Clone, Debug, PartialEq)]
pub struct Int8Codec {
    /// One scale a dimension, every one finite and positive.
    scales: Vec<f32>,
}

impl Int8Codec {
    /// Fit the scales to `sample`, every row of which is `dim` values.
    ///
    /// Refuses an empty sample, a zero width, and a row of the wrong width
    /// or holding a value that is not finite, naming the row.
    pub fn fit(dim: usize, sample: &[Vec<f32>]) -> Result<Self, String> {
        if dim == 0 {
            return Err("a scalar codec quantizes vectors of at least one value".to_string());
        }
        if sample.is_empty() {
            return Err("a scalar codec cannot be fitted to an empty sample".to_string());
        }
        let mut largest = vec![0f32; dim];
        for (row, vector) in sample.iter().enumerate() {
            if vector.len() != dim {
                return Err(format!(
                    "row {} of the sample holds {} values where the codec quantizes {}",
                    row,
                    vector.len(),
                    dim
                ));
            }
            for (slot, &value) in largest.iter_mut().zip(vector) {
                if !value.is_finite() {
                    return Err(format!(
                        "row {} of the sample holds {}, which is not finite",
                        row, value
                    ));
                }
                let magnitude = value.abs();
                if magnitude > *slot {
                    *slot = magnitude;
                }
            }
        }
        let scales = largest
            .into_iter()
            .map(|magnitude| {
                if magnitude > 0.0 {
                    magnitude / QMAX
                } else {
                    1.0
                }
            })
            .collect();
        Ok(Int8Codec { scales })
    }

    /// A codec over scales read back from a saved index.
    ///
    /// Refuses an empty array and any scale that is not finite and
    /// positive, since a zero or negative scale decodes every code to the
    /// same value or to the wrong sign and a non-finite one poisons every
    /// distance.
    pub fn from_scales(scales: Vec<f32>) -> Result<Self, String> {
        if scales.is_empty() {
            return Err("a scalar codec carries at least one scale".to_string());
        }
        if let Some((at, &scale)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || **scale <= 0.0)
        {
            return Err(format!(
                "scale {} is {}, and every scale is finite and positive",
                at, scale
            ));
        }
        Ok(Int8Codec { scales })
    }

    /// Values in a vector this codec encodes.
    #[inline]
    pub fn dim(&self) -> usize {
        self.scales.len()
    }

    /// The scale of every dimension, in order.
    #[inline]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Bytes the codec asks the allocator for, being the scale array.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.scales.capacity() * std::mem::size_of::<f32>()
    }

    /// Encode one vector, saturating a value beyond a dimension's range at
    /// the edge of the code range. Refuses a vector of the wrong width.
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<i8>, String> {
        if vector.len() != self.scales.len() {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.scales.len(),
                vector.len()
            ));
        }
        let mut codes = vec![0i8; vector.len()];
        self.quantize_into(vector, &mut codes);
        Ok(codes)
    }

    /// Encode `vector` into `codes`, which is the same width. The width is
    /// asserted rather than checked because every caller has already been
    /// through [`Int8Codec::quantize`]'s check or holds a store of this
    /// width.
    #[inline]
    pub fn quantize_into(&self, vector: &[f32], codes: &mut [i8]) {
        assert_eq!(vector.len(), self.scales.len());
        assert_eq!(codes.len(), self.scales.len());
        for ((code, &value), &scale) in codes.iter_mut().zip(vector).zip(&self.scales) {
            *code = encode(value, scale);
        }
    }

    /// Count the values of `vector` that saturate, being the values `encode`
    /// clips at the edge of the range, for a caller measuring how far a
    /// corpus reaches past the sample it was fitted on.
    ///
    /// The test is applied after the rounding `encode` applies, so a value
    /// at the edge of its dimension's range, whose division can land a
    /// rounding error above 127 and which `encode` rounds to 127 and clips
    /// nothing from, is not counted. A sample value therefore never counts
    /// against the sample it was fitted on.
    pub fn saturated(&self, vector: &[f32]) -> usize {
        vector
            .iter()
            .zip(&self.scales)
            .filter(|(&value, &scale)| (value / scale).round_ties_even().abs() > QMAX)
            .count()
    }

    /// The vector a code stands for, being `code * scale` in every
    /// dimension. Refuses a code of the wrong width.
    pub fn reconstruct(&self, codes: &[i8]) -> Result<Vec<f32>, String> {
        if codes.len() != self.scales.len() {
            return Err(format!(
                "Code length mismatch: expected {}, got {}",
                self.scales.len(),
                codes.len()
            ));
        }
        Ok(codes
            .iter()
            .zip(&self.scales)
            .map(|(&code, &scale)| decode(code, scale))
            .collect())
    }
}

/// One value to its code: scaled, rounded half to even, and saturated.
#[inline(always)]
fn encode(value: f32, scale: f32) -> i8 {
    let scaled = (value / scale).round_ties_even();
    // A NaN would compare false against both bounds and reach the cast,
    // which saturates to zero; `fit` and every insertion path refuse one
    // before it gets here, so this is a statement rather than a guard.
    scaled.clamp(-QMAX, QMAX) as i8
}

/// One code to the value it stands for. This is the one multiply every
/// kernel decodes with, so a value decoded here and a value decoded inside
/// the kernel are the same `f32` bit for bit.
#[inline(always)]
pub(crate) fn decode(code: i8, scale: f32) -> f32 {
    code as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Vec<f32>> {
        vec![
            vec![0.5, -2.0, 0.0, 12.7],
            vec![-1.0, 1.0, 0.0, -0.3],
            vec![0.25, 3.5, 0.0, 6.35],
        ]
    }

    /// The scale is the largest magnitude over 127, a zero dimension takes
    /// one, and every sample value rounds to a code inside the range.
    #[test]
    fn the_scales_are_fitted_per_dimension() {
        let codec = Int8Codec::fit(4, &sample()).unwrap();
        assert_eq!(codec.dim(), 4);
        let scales = codec.scales();
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-9);
        assert!((scales[1] - 3.5 / 127.0).abs() < 1e-9);
        assert_eq!(scales[2], 1.0);
        assert!((scales[3] - 12.7 / 127.0).abs() < 1e-9);
        for vector in sample() {
            let codes = codec.quantize(&vector).unwrap();
            assert!(codes.iter().all(|&c| (-127..=127).contains(&c)));
            assert_eq!(codec.saturated(&vector), 0);
        }
        assert_eq!(
            codec.quantize(&[1.0, 3.5, 0.0, 12.7]).unwrap(),
            vec![127, 127, 0, 127]
        );
        assert_eq!(
            codec.quantize(&[-1.0, -3.5, 0.0, -12.7]).unwrap(),
            vec![-127, -127, 0, -127]
        );
    }

    /// A value beyond the fitted range saturates at the edge rather than
    /// wrapping, and the count of such values is reported.
    #[test]
    fn a_value_past_the_range_saturates() {
        let codec = Int8Codec::fit(4, &sample()).unwrap();
        let codes = codec.quantize(&[5.0, -40.0, 0.4, 0.0]).unwrap();
        assert_eq!(codes, vec![127, -127, 0, 0]);
        assert_eq!(codec.saturated(&[5.0, -40.0, 0.4, 0.0]), 2);
    }

    /// The count agrees with `encode` at the edge: a value at exactly the
    /// range a dimension was fitted to, and one within half a step of it,
    /// encode to 127 and are not counted, whatever rounding the division
    /// lands on, and one a full step past it is.
    #[test]
    fn the_saturation_count_agrees_with_the_encoder_at_the_edge() {
        let magnitudes = [0.3f32, 1.7, 12.7, 0.07, 1e-3, 255.0, 0.9999, 3.3];
        let sample: Vec<Vec<f32>> = vec![magnitudes.to_vec()];
        let codec = Int8Codec::fit(magnitudes.len(), &sample).unwrap();
        for &sign in &[1.0f32, -1.0] {
            let at_edge: Vec<f32> = magnitudes.iter().map(|m| m * sign).collect();
            assert_eq!(codec.saturated(&at_edge), 0);
            assert!(codec
                .quantize(&at_edge)
                .unwrap()
                .iter()
                .all(|c| c.abs() == 127));
            let within: Vec<f32> = codec.scales().iter().map(|s| 127.4 * s * sign).collect();
            assert_eq!(codec.saturated(&within), 0);
            let past: Vec<f32> = codec.scales().iter().map(|s| 128.0 * s * sign).collect();
            assert_eq!(codec.saturated(&past), magnitudes.len());
            assert!(codec
                .quantize(&past)
                .unwrap()
                .iter()
                .all(|c| c.abs() == 127));
        }
    }

    /// Reconstruction is the one multiply, so a code decodes to exactly
    /// `code * scale` and a round trip lands within half a scale of the
    /// value it started from.
    #[test]
    fn reconstruction_is_one_multiply_and_within_half_a_step() {
        let codec = Int8Codec::fit(4, &sample()).unwrap();
        for vector in sample() {
            let codes = codec.quantize(&vector).unwrap();
            let back = codec.reconstruct(&codes).unwrap();
            for ((&value, &decoded), (&code, &scale)) in vector
                .iter()
                .zip(&back)
                .zip(codes.iter().zip(codec.scales()))
            {
                assert_eq!(decoded.to_bits(), (code as f32 * scale).to_bits());
                assert!((value - decoded).abs() <= scale * 0.5 + 1e-6);
            }
        }
    }

    /// Ties round to the even code, which is the rule stated in the module
    /// documentation and the rule a reference implementation reproduces.
    #[test]
    fn a_tie_rounds_to_the_even_code() {
        let codec = Int8Codec::from_scales(vec![1.0, 1.0]).unwrap();
        assert_eq!(codec.quantize(&[0.5, 1.5]).unwrap(), vec![0, 2]);
        assert_eq!(codec.quantize(&[-0.5, -2.5]).unwrap(), vec![0, -2]);
    }

    /// Every malformed input is refused by name.
    #[test]
    fn every_refusal_names_its_reason() {
        assert!(Int8Codec::fit(0, &sample())
            .unwrap_err()
            .contains("at least one value"));
        assert!(Int8Codec::fit(4, &[]).unwrap_err().contains("empty sample"));
        assert!(Int8Codec::fit(3, &sample()).unwrap_err().contains("row 0"));
        let mut bad = sample();
        bad[1][2] = f32::NAN;
        assert!(Int8Codec::fit(4, &bad).unwrap_err().contains("row 1"));
        assert!(Int8Codec::from_scales(vec![])
            .unwrap_err()
            .contains("at least one scale"));
        assert!(Int8Codec::from_scales(vec![1.0, 0.0])
            .unwrap_err()
            .contains("scale 1 is 0"));
        assert!(Int8Codec::from_scales(vec![-1.0])
            .unwrap_err()
            .contains("scale 0"));
        assert!(Int8Codec::from_scales(vec![f32::INFINITY]).is_err());
        let codec = Int8Codec::fit(4, &sample()).unwrap();
        assert!(codec
            .quantize(&[1.0, 2.0])
            .unwrap_err()
            .contains("expected 4, got 2"));
        assert!(codec
            .reconstruct(&[1, 2])
            .unwrap_err()
            .contains("expected 4, got 2"));
    }

    /// The memory figure is the scale array and the struct.
    #[test]
    fn the_memory_figure_is_the_scale_array() {
        let codec = Int8Codec::from_scales(vec![1.0; 100]).unwrap();
        assert_eq!(
            codec.memory_bytes(),
            std::mem::size_of::<Int8Codec>() + 100 * 4
        );
    }
}
