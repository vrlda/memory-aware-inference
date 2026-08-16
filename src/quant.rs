//! Lossless readers for GGUF quantized blocks.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantError(pub String);

pub type Result<T> = std::result::Result<T, QuantError>;

pub const Q4_K_BLOCK_ELEMENTS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q5_K_BLOCK_ELEMENTS: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;

pub fn dequantize_q4_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    if elements == 0 || !elements.is_multiple_of(Q4_K_BLOCK_ELEMENTS) {
        return Err(QuantError(
            "Q4_K element count must be a non-zero multiple of 256".into(),
        ));
    }
    let blocks = elements / Q4_K_BLOCK_ELEMENTS;
    let expected_bytes = blocks
        .checked_mul(Q4_K_BLOCK_BYTES)
        .ok_or_else(|| QuantError("Q4_K byte count overflows".into()))?;
    if bytes.len() != expected_bytes {
        return Err(QuantError(format!(
            "Q4_K data has {} bytes; expected {}",
            bytes.len(),
            expected_bytes
        )));
    }

    let mut output = Vec::with_capacity(elements);
    for block in bytes.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = &block[16..];
        let mut scale_index = 0;
        for quant_group in 0..4 {
            let (scale, minimum) = get_scale_min(scale_index, scales);
            let d1 = d * f32::from(scale);
            let m1 = min * f32::from(minimum);
            let (scale, minimum) = get_scale_min(scale_index + 1, scales);
            let d2 = d * f32::from(scale);
            let m2 = min * f32::from(minimum);
            let quant_bytes = &quants[quant_group * 32..][..32];
            output.extend(
                quant_bytes
                    .iter()
                    .map(|quant| d1 * f32::from(quant & 0x0f) - m1),
            );
            output.extend(
                quant_bytes
                    .iter()
                    .map(|quant| d2 * f32::from(quant >> 4) - m2),
            );
            scale_index += 2;
        }
    }
    Ok(output)
}

pub fn dequantize_q5_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    if elements == 0 || !elements.is_multiple_of(Q5_K_BLOCK_ELEMENTS) {
        return Err(QuantError(
            "Q5_K element count must be a non-zero multiple of 256".into(),
        ));
    }
    let blocks = elements / Q5_K_BLOCK_ELEMENTS;
    let expected_bytes = blocks
        .checked_mul(Q5_K_BLOCK_BYTES)
        .ok_or_else(|| QuantError("Q5_K byte count overflows".into()))?;
    if bytes.len() != expected_bytes {
        return Err(QuantError(format!(
            "Q5_K data has {} bytes; expected {}",
            bytes.len(),
            expected_bytes
        )));
    }

    let mut output = Vec::with_capacity(elements);
    for block in bytes.chunks_exact(Q5_K_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let high_bits = &block[16..48];
        let quants = &block[48..];
        let mut high_low_mask = 1_u8;
        let mut high_high_mask = 2_u8;
        let mut scale_index = 0;
        for quant_group in 0..4 {
            let (scale_1, minimum_1) = get_scale_min(scale_index, scales);
            let (scale_2, minimum_2) = get_scale_min(scale_index + 1, scales);
            let d1 = d * f32::from(scale_1);
            let m1 = min * f32::from(minimum_1);
            let d2 = d * f32::from(scale_2);
            let m2 = min * f32::from(minimum_2);
            let quant_bytes = &quants[quant_group * 32..][..32];
            output.extend(quant_bytes.iter().enumerate().map(|(index, quant)| {
                let high = if high_bits[index] & high_low_mask != 0 {
                    16
                } else {
                    0
                };
                d1 * f32::from((quant & 0x0f) + high) - m1
            }));
            output.extend(quant_bytes.iter().enumerate().map(|(index, quant)| {
                let high = if high_bits[index] & high_high_mask != 0 {
                    16
                } else {
                    0
                };
                d2 * f32::from((quant >> 4) + high) - m2
            }));
            scale_index += 2;
            high_low_mask <<= 2;
            high_high_mask <<= 2;
        }
    }
    Ok(output)
}

pub fn dequantize_q6_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    if elements == 0 || !elements.is_multiple_of(Q6_K_BLOCK_ELEMENTS) {
        return Err(QuantError(
            "Q6_K element count must be a non-zero multiple of 256".into(),
        ));
    }
    let blocks = elements / Q6_K_BLOCK_ELEMENTS;
    let expected_bytes = blocks
        .checked_mul(Q6_K_BLOCK_BYTES)
        .ok_or_else(|| QuantError("Q6_K byte count overflows".into()))?;
    if bytes.len() != expected_bytes {
        return Err(QuantError(format!(
            "Q6_K data has {} bytes; expected {}",
            bytes.len(),
            expected_bytes
        )));
    }

    let mut output = Vec::with_capacity(elements);
    for block in bytes.chunks_exact(Q6_K_BLOCK_BYTES) {
        let quants_low = &block[..128];
        let quants_high = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let output_start = output.len();
        output.resize(output_start + Q6_K_BLOCK_ELEMENTS, 0.0);
        for half in 0..2 {
            let low = &quants_low[half * 64..][..64];
            let high = &quants_high[half * 32..][..32];
            let block_scales = &scales[half * 8..][..8];
            for index in 0..32 {
                let scale_index = index / 16;
                let high_bits = high[index];
                let q1 = i32::from((low[index] & 0x0f) | ((high_bits & 0x03) << 4)) - 32;
                let q2 = i32::from((low[index + 32] & 0x0f) | ((high_bits >> 2 & 0x03) << 4)) - 32;
                let q3 = i32::from((low[index] >> 4) | ((high_bits >> 4 & 0x03) << 4)) - 32;
                let q4 = i32::from((low[index + 32] >> 4) | ((high_bits >> 6) << 4)) - 32;
                let scale = |offset: usize| f32::from(i8::from_ne_bytes([block_scales[offset]]));
                let base = output_start + half * 128;
                output[base + index] = d * scale(scale_index) * q1 as f32;
                output[base + index + 32] = d * scale(scale_index + 2) * q2 as f32;
                output[base + index + 64] = d * scale(scale_index + 4) * q3 as f32;
                output[base + index + 96] = d * scale(scale_index + 6) * q4 as f32;
            }
        }
    }
    Ok(output)
}

fn get_scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction;
            let mut exponent = -14_i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (u32::try_from(exponent + 127).unwrap_or(0) << 23) | (fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent) + 112) << 23 | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_one_q4_k_block() {
        let mut block = vec![0_u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        for scale in &mut block[4..16] {
            *scale = 1;
        }
        for quant in &mut block[16..] {
            *quant = 0x10;
        }

        let values = dequantize_q4_k(&block, 256).expect("valid Q4_K block");

        assert_eq!(values.len(), 256);
        for group in values.chunks_exact(64) {
            assert!(group[..32].iter().all(|value| *value == 0.0));
            assert!(group[32..].iter().all(|value| *value == 1.0));
        }
    }

    #[test]
    fn rejects_q4_k_shape_and_byte_mismatches() {
        assert!(dequantize_q4_k(&[0; Q4_K_BLOCK_BYTES], 255).is_err());
        assert!(dequantize_q4_k(&[0; Q4_K_BLOCK_BYTES - 1], 256).is_err());
        assert!(dequantize_q4_k(&[0; Q4_K_BLOCK_BYTES + 1], 256).is_err());
    }

    #[test]
    fn decodes_one_q5_k_block_with_high_bits() {
        let mut block = vec![0_u8; Q5_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        for scale in &mut block[4..16] {
            *scale = 1;
        }
        for quant in &mut block[48..] {
            *quant = 0x10;
        }
        for high in &mut block[16..48] {
            *high = 0xff;
        }
        let values = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMENTS).expect("valid Q5_K block");
        assert_eq!(values.len(), Q5_K_BLOCK_ELEMENTS);
        for group in values.chunks_exact(64) {
            assert!(group[..32].iter().all(|value| *value == 16.0));
            assert!(group[32..].iter().all(|value| *value == 17.0));
        }
    }

    #[test]
    fn decodes_one_q6_k_block_with_high_bits_and_scales() {
        let mut block = vec![0_u8; Q6_K_BLOCK_BYTES];
        block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[..128].fill(0x21);
        block[128..192].fill(0xe4);
        block[192..208].fill(1);
        let values = dequantize_q6_k(&block, Q6_K_BLOCK_ELEMENTS).expect("valid Q6_K block");
        assert_eq!(values.len(), Q6_K_BLOCK_ELEMENTS);
        for chunk in values.chunks_exact(128) {
            assert!(chunk[..16].iter().all(|value| *value == -31.0));
            assert!(chunk[16..32].iter().all(|value| *value == -31.0));
            assert!(chunk[32..48].iter().all(|value| *value == -15.0));
            assert!(chunk[48..64].iter().all(|value| *value == -15.0));
            assert!(chunk[64..80].iter().all(|value| *value == 2.0));
            assert!(chunk[80..96].iter().all(|value| *value == 2.0));
            assert!(chunk[96..112].iter().all(|value| *value == 18.0));
            assert!(chunk[112..128].iter().all(|value| *value == 18.0));
        }
    }
}
