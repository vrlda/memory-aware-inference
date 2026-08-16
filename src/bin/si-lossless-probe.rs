//! Probe lossless block codecs on the immutable BF16 model representation.
//! This intentionally does not alter runtime inference; it only measures
//! whether compressed tiles can reduce weight traffic before a codec is
//! considered for runtime integration.

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(target_os = "macos")]
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};

#[cfg(target_os = "macos")]
const CODECS: &[&str] = &["zlib", "bf16-plane-zlib", "bf16-bitpack", "bf16-rle"];

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("error: lossless compression probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::ModelStore::open(&args.model, args.verify_manifest)
        .map_err(|error| error.to_string())?;
    let default_tensors = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
    ];
    let tensor_names = if args.tensors.is_empty() {
        default_tensors
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>()
    } else {
        args.tensors
    };
    println!(
        "probe=model={} block_kib={} codecs=zlib,bf16-plane-zlib,bf16-bitpack,bf16-rle",
        args.model.display(),
        args.block_bytes / 1024
    );
    for tensor_name in tensor_names {
        let tensor = store
            .tensor(&tensor_name)
            .map_err(|error| error.to_string())?;
        if tensor.info.dtype != "BF16" {
            return Err(format!("{tensor_name} is not BF16"));
        }
        println!(
            "tensor={} shape={:?} bytes={}",
            tensor_name,
            tensor.info.shape,
            tensor.bytes.len()
        );
        for codec_name in CODECS {
            let mut compressed_bytes = 0_usize;
            let mut encoded_blocks = 0_usize;
            let mut encode_elapsed = std::time::Duration::ZERO;
            let mut decode_elapsed = std::time::Duration::ZERO;
            for block in tensor.bytes.chunks(args.block_bytes) {
                let started = std::time::Instant::now();
                let compressed = encode_block(block, codec_name)?;
                encode_elapsed += started.elapsed();
                let started = std::time::Instant::now();
                decode_block(&compressed, block, codec_name)?;
                decode_elapsed += started.elapsed();
                compressed_bytes = compressed_bytes.saturating_add(compressed.len());
                encoded_blocks += 1;
            }
            let ratio = tensor.bytes.len() as f64 / compressed_bytes.max(1) as f64;
            let encoded_gib_s = tensor.bytes.len() as f64
                / encode_elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
                / 1.0e9;
            let decoded_gib_s = tensor.bytes.len() as f64
                / decode_elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
                / 1.0e9;
            println!(
                "codec={} blocks={} compressed_bytes={} ratio={:.4} encode_gib_s={:.3} decode_gib_s={:.3} exact=true",
                codec_name,
                encoded_blocks,
                compressed_bytes,
                ratio,
                encoded_gib_s,
                decoded_gib_s,
            );
        }
        if args.metal_matvec {
            if tensor.info.shape.len() != 2 {
                println!("metal_matvec=skipped reason=rank_not_two");
            } else if tensor.bytes.len() > 100 * 1024 * 1024 {
                println!(
                    "metal_matvec=skipped reason=tensor_too_large bytes={} limit={}",
                    tensor.bytes.len(),
                    100 * 1024 * 1024
                );
            } else {
                run_metal_matvec_probe(&tensor, args.repetitions)?;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_metal_matvec_probe(
    tensor: &super_inference::model::TensorView<'_>,
    repetitions: usize,
) -> Result<(), String> {
    let rows = tensor.info.shape[0];
    let columns = tensor.info.shape[1];
    let tile_rows = 8;
    let (packed, offsets) = encode_bf16_bitpack_matrix(tensor.bytes, rows, columns, tile_rows)?;
    let input = (0..columns)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    std::env::set_var("SI_LOSSLESS_GPU", "1");
    let context = super_inference::metal::MetalContext::new()?;
    let baseline_warmup = context.bf16_matvec_tensor(tensor, &input)?;
    let packed_warmup =
        context.bf16_bitpack_matvec(&packed, &offsets, rows, columns, tile_rows, &input)?;
    let max_abs_diff = baseline_warmup
        .iter()
        .zip(&packed_warmup)
        .map(|(baseline, packed)| (baseline - packed).abs())
        .fold(0.0_f32, f32::max);
    let baseline_started = std::time::Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(context.bf16_matvec_tensor(tensor, &input)?);
    }
    let baseline_elapsed = baseline_started.elapsed();
    let packed_started = std::time::Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(
            context.bf16_bitpack_matvec(&packed, &offsets, rows, columns, tile_rows, &input)?,
        );
    }
    let packed_elapsed = packed_started.elapsed();
    let baseline_ms = baseline_elapsed.as_secs_f64() * 1_000.0 / repetitions as f64;
    let packed_ms = packed_elapsed.as_secs_f64() * 1_000.0 / repetitions as f64;
    println!(
        "metal_matvec=bitpack tile_rows={} packed_bytes={} ratio={:.4} baseline_ms={:.3} bitpack_ms={:.3} speedup={:.3} max_abs_diff={:.7} exact={}",
        tile_rows,
        packed.len(),
        tensor.bytes.len() as f64 / packed.len().max(1) as f64,
        baseline_ms,
        packed_ms,
        baseline_ms / packed_ms.max(f64::MIN_POSITIVE),
        max_abs_diff,
        max_abs_diff <= 1.0e-5,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_block(source: &[u8], codec: &str) -> Result<Vec<u8>, String> {
    match codec {
        "zlib" => zlib_encode(source),
        "bf16-plane-zlib" => zlib_encode(&bf16_plane_transform(source)?),
        "bf16-bitpack" => encode_bf16_bitpack(source),
        "bf16-rle" => encode_bf16_rle(source),
        _ => Err(format!("unsupported codec {codec}")),
    }
}

#[cfg(target_os = "macos")]
fn decode_block(compressed: &[u8], expected: &[u8], codec: &str) -> Result<(), String> {
    let decoded = match codec {
        "zlib" => zlib_decode(compressed, expected.len())?,
        "bf16-plane-zlib" => bf16_plane_inverse(&zlib_decode(compressed, expected.len())?)?,
        "bf16-bitpack" => decode_bf16_bitpack(compressed, expected.len())?,
        "bf16-rle" => decode_bf16_rle(compressed)?,
        _ => return Err(format!("unsupported codec {codec}")),
    };
    if decoded != expected {
        return Err("lossless codec round-trip changed BF16 bits".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn zlib_encode(source: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    std::io::Write::write_all(&mut encoder, source)
        .map_err(|error| format!("zlib encode: {error}"))?;
    encoder
        .finish()
        .map_err(|error| format!("zlib finish: {error}"))
}

#[cfg(target_os = "macos")]
fn zlib_decode(source: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    let mut decoder = ZlibDecoder::new(source);
    let mut decoded = Vec::with_capacity(expected_len);
    std::io::Read::read_to_end(&mut decoder, &mut decoded)
        .map_err(|error| format!("zlib decode: {error}"))?;
    Ok(decoded)
}

#[cfg(target_os = "macos")]
fn bf16_plane_transform(source: &[u8]) -> Result<Vec<u8>, String> {
    if !source.len().is_multiple_of(2) {
        return Err("BF16 plane transform requires 2-byte alignment".into());
    }
    let values = source
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect::<Vec<_>>();
    let mut transformed = Vec::with_capacity(source.len());
    transformed.extend(values.iter().map(|value| (value >> 8) as u8));
    transformed.extend(values.iter().map(|value| *value as u8));
    Ok(transformed)
}

#[cfg(target_os = "macos")]
fn bf16_plane_inverse(source: &[u8]) -> Result<Vec<u8>, String> {
    if !source.len().is_multiple_of(2) {
        return Err("BF16 plane stream has invalid length".into());
    }
    let values = source.len() / 2;
    let (high, low) = source.split_at(values);
    let mut decoded = Vec::with_capacity(source.len());
    for (high, low) in high.iter().zip(low) {
        decoded.extend_from_slice(&u16::from_be_bytes([*high, *low]).to_le_bytes());
    }
    Ok(decoded)
}

#[cfg(target_os = "macos")]
fn encode_bf16_rle(source: &[u8]) -> Result<Vec<u8>, String> {
    if !source.len().is_multiple_of(2) {
        return Err("BF16 RLE requires 2-byte alignment".into());
    }
    let values = source
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect::<Vec<_>>();
    let mut encoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < values.len() {
        let value = values[index];
        let mut run = 1_usize;
        while index + run < values.len() && values[index + run] == value && run < u16::MAX as usize
        {
            run += 1;
        }
        encoded.extend_from_slice(&(run as u16).to_le_bytes());
        encoded.extend_from_slice(&value.to_le_bytes());
        index += run;
    }
    Ok(encoded)
}

#[cfg(target_os = "macos")]
fn decode_bf16_rle(source: &[u8]) -> Result<Vec<u8>, String> {
    if !source.len().is_multiple_of(4) {
        return Err("BF16 RLE stream has invalid length".into());
    }
    let mut decoded = Vec::new();
    for pair in source.chunks_exact(4) {
        let count = usize::from(u16::from_le_bytes([pair[0], pair[1]]));
        if count == 0 {
            return Err("BF16 RLE stream contains a zero run".into());
        }
        let value = [pair[2], pair[3]];
        for _ in 0..count {
            decoded.extend_from_slice(&value);
        }
    }
    Ok(decoded)
}

#[cfg(target_os = "macos")]
fn encode_bf16_bitpack(source: &[u8]) -> Result<Vec<u8>, String> {
    if !source.len().is_multiple_of(2) || source.is_empty() {
        return Err("BF16 bitpack requires non-empty 2-byte-aligned input".into());
    }
    let values = source
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect::<Vec<_>>();
    let first = values[0];
    let mut invariant_mask = 0_u16;
    let mut constants = 0_u16;
    for bit in 0..16 {
        let mask = 1_u16 << bit;
        let set = first & mask;
        if values.iter().all(|value| value & mask == set) {
            invariant_mask |= mask;
            constants |= set;
        }
    }
    let variable_bits = 16 - invariant_mask.count_ones();
    let packed_bytes = (values.len() * variable_bits as usize).div_ceil(8);
    let mut encoded = Vec::with_capacity(4 + packed_bytes);
    encoded.extend_from_slice(&invariant_mask.to_le_bytes());
    encoded.extend_from_slice(&constants.to_le_bytes());
    let mut accumulator = 0_u8;
    let mut available = 0_u8;
    for value in values {
        for bit in 0..16 {
            let mask = 1_u16 << bit;
            if invariant_mask & mask != 0 {
                continue;
            }
            if value & mask != 0 {
                accumulator |= 1 << available;
            }
            available += 1;
            if available == 8 {
                encoded.push(accumulator);
                accumulator = 0;
                available = 0;
            }
        }
    }
    if available > 0 {
        encoded.push(accumulator);
    }
    Ok(encoded)
}

#[cfg(target_os = "macos")]
fn encode_bf16_bitpack_matrix(
    source: &[u8],
    rows: usize,
    columns: usize,
    tile_rows: usize,
) -> Result<(Vec<u8>, Vec<u32>), String> {
    if rows == 0 || columns == 0 || tile_rows == 0 {
        return Err("BF16 bitpack matrix dimensions must be non-zero".into());
    }
    let row_bytes = columns
        .checked_mul(2)
        .ok_or("BF16 bitpack matrix row byte length overflow")?;
    let expected_len = rows
        .checked_mul(row_bytes)
        .ok_or("BF16 bitpack matrix dimensions overflow")?;
    if source.len() != expected_len {
        return Err("BF16 bitpack matrix byte length does not match dimensions".into());
    }
    let tile_count = rows.div_ceil(tile_rows);
    let mut packed = Vec::new();
    let mut offsets = Vec::with_capacity(tile_count + 1);
    for row_start in (0..rows).step_by(tile_rows) {
        let row_count = tile_rows.min(rows - row_start);
        let start = row_start * row_bytes;
        let end = start + row_count * row_bytes;
        offsets.push(
            u32::try_from(packed.len())
                .map_err(|_| "BF16 bitpack matrix exceeds 4 GiB offset range")?,
        );
        packed.extend_from_slice(&encode_bf16_bitpack(&source[start..end])?);
    }
    offsets.push(
        u32::try_from(packed.len())
            .map_err(|_| "BF16 bitpack matrix exceeds 4 GiB offset range")?,
    );
    Ok((packed, offsets))
}

#[cfg(target_os = "macos")]
fn decode_bf16_bitpack(source: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    if source.len() < 4 || !expected_len.is_multiple_of(2) {
        return Err("BF16 bitpack stream has invalid dimensions".into());
    }
    let invariant_mask = u16::from_le_bytes([source[0], source[1]]);
    let constants = u16::from_le_bytes([source[2], source[3]]);
    if constants & !invariant_mask != 0 {
        return Err("BF16 bitpack constants contain a variable bit".into());
    }
    let values = expected_len / 2;
    let variable_bits = 16 - invariant_mask.count_ones();
    let expected_packed = (values * variable_bits as usize).div_ceil(8);
    if source.len() != 4 + expected_packed {
        return Err("BF16 bitpack payload length does not match metadata".into());
    }
    let mut decoded = Vec::with_capacity(expected_len);
    let mut byte_index = 4;
    let mut bit_index = 0_u8;
    for _ in 0..values {
        let mut value = constants;
        for bit in 0..16 {
            let mask = 1_u16 << bit;
            if invariant_mask & mask != 0 {
                continue;
            }
            if source[byte_index] & (1 << bit_index) != 0 {
                value |= mask;
            }
            bit_index += 1;
            if bit_index == 8 {
                byte_index += 1;
                bit_index = 0;
            }
        }
        decoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(decoded)
}

struct Args {
    model: PathBuf,
    tensors: Vec<String>,
    block_bytes: usize,
    verify_manifest: bool,
    metal_matvec: bool,
    repetitions: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut model = None;
        let mut tensors = Vec::new();
        let mut block_bytes = 64 * 1024;
        let mut verify_manifest = false;
        let mut metal_matvec = false;
        let mut repetitions = 3;
        let mut index = 0;
        while index < arguments.len() {
            if arguments[index] == "--verify-manifest" {
                verify_manifest = true;
                index += 1;
                continue;
            }
            if arguments[index] == "--metal-matvec" {
                metal_matvec = true;
                index += 1;
                continue;
            }
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--tensor" => tensors.push(value.clone()),
                "--block-kib" => {
                    let kib = value
                        .parse::<usize>()
                        .map_err(|_| "--block-kib must be an integer".to_owned())?;
                    block_bytes = kib.checked_mul(1024).ok_or("--block-kib overflows")?;
                    if block_bytes == 0 {
                        return Err("--block-kib must be non-zero".into());
                    }
                }
                "--repetitions" => {
                    repetitions = value
                        .parse::<usize>()
                        .map_err(|_| "--repetitions must be an integer".to_owned())?;
                    if repetitions == 0 {
                        return Err("--repetitions must be non-zero".into());
                    }
                }
                "--help" | "-h" => return Err(Self::usage().into()),
                unknown => return Err(format!("unknown option: {unknown}")),
            }
            index += 2;
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            tensors,
            block_bytes,
            verify_manifest,
            metal_matvec,
            repetitions,
        })
    }

    fn usage() -> &'static str {
        "Usage: si-lossless-probe --model PATH [--tensor NAME] [--block-kib N] [--repetitions N] [--verify-manifest] [--metal-matvec]"
    }
}

#[cfg(test)]
mod tests {
    use super::Args;
    use std::path::PathBuf;

    #[test]
    fn parses_lossless_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model".into(),
            "--tensor".into(),
            "head".into(),
            "--block-kib".into(),
            "32".into(),
            "--verify-manifest".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.model, PathBuf::from("model"));
        assert_eq!(args.tensors, vec!["head"]);
        assert_eq!(args.block_bytes, 32 * 1024);
        assert!(args.verify_manifest);
        assert!(!args.metal_matvec);
        assert_eq!(args.repetitions, 3);
    }

    #[test]
    fn parses_metal_matvec_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model".into(),
            "--metal-matvec".into(),
            "--repetitions".into(),
            "5".into(),
        ])
        .expect("arguments should parse");
        assert!(args.metal_matvec);
        assert_eq!(args.repetitions, 5);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn invariant_bitpack_round_trips_bf16_bits() {
        let source = [0x00_u8, 0x3f, 0x01, 0x3f, 0x00, 0x3f, 0xff, 0x3e];
        let encoded = super::encode_bf16_bitpack(&source).expect("encode should succeed");
        let decoded =
            super::decode_bf16_bitpack(&encoded, source.len()).expect("decode should succeed");
        assert_eq!(decoded, source);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn row_aligned_bitpack_round_trips_matrix_tiles() {
        let source = [
            0x00_u8, 0x3f, 0x01, 0x3f, 0x00, 0x3f, 0xff, 0x3e, 0x02, 0x3f, 0x03, 0x3f,
        ];
        let (packed, offsets) = super::encode_bf16_bitpack_matrix(&source, 3, 2, 2)
            .expect("matrix encoding should succeed");
        assert_eq!(offsets.len(), 3);
        let mut decoded = Vec::new();
        for tile in 0..2 {
            let start = offsets[tile] as usize;
            let end = offsets[tile + 1] as usize;
            decoded.extend(
                super::decode_bf16_bitpack(
                    &packed[start..end],
                    (if tile == 0 { 2 } else { 1 }) * 4,
                )
                .expect("matrix tile should decode"),
            );
        }
        assert_eq!(decoded, source);
    }
}
