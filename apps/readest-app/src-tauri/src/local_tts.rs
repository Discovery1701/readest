//! Local neural TTS (ZipVoice via sherpa-onnx), desktop only.
//!
//! This backs a self-hosted "local TTS" engine: no cloud, no account. Models
//! live next to the portable executable under `<exe_dir>/models/zipvoice/`
//! (deliberately NOT the user profile, so a portable build stays self-contained).
//!
//! Stage 0: load + synthesize + WAV encoding + status only. Model download is a
//! later step; until then the files must already be present in that directory.
//!
//! Expected layout under models/zipvoice/:
//!   vocos_24khz.onnx
//!   lexicon-en-us.txt
//!   sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/
//!     encoder.int8.onnx  decoder.int8.onnx  tokens.txt  lexicon.txt
//!     test_wavs/leijun-1.wav
//!
//! The reference clip (`leijun-1.wav`) + its transcript define the default
//! voice via ZipVoice zero-shot cloning, so it works out of the box.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::Serialize;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsZipvoiceModelConfig, Wave,
};

/// Sub-directory of the exe folder that holds the models.
const MODEL_SUBDIR: &str = "models/zipvoice";
/// Directory name produced by extracting the ZipVoice tarball.
const ZIPVOICE_DIR: &str = "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia";
/// Reference clip shipped inside the ZipVoice model dir.
const REFERENCE_WAV: &str = "test_wavs/leijun-1.wav";
/// Transcript of the reference clip (must match it word-for-word).
const REFERENCE_TEXT: &str = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";

/// Files that must exist (relative to the model dir) for the engine to run.
const REQUIRED_FILES: &[&str] = &[
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokens.txt",
    "lexicon.txt",
    "test_wavs/leijun-1.wav",
];

struct TtsEngine {
    tts: OfflineTts,
    reference: Vec<f32>,
    reference_rate: i32,
}

static ENGINE: OnceLock<Mutex<Option<TtsEngine>>> = OnceLock::new();

fn engine_cell() -> &'static Mutex<Option<TtsEngine>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

/// Directory of the running executable (portable build: where the .exe sits).
fn exe_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe failed: {e}"))?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "executable has no parent directory".to_string())
}

fn model_dir() -> Result<PathBuf, String> {
    Ok(exe_dir()?.join(MODEL_SUBDIR))
}

#[derive(Serialize)]
pub struct LocalTtsStatus {
    /// Absolute path of the model directory (for "place files here" hints).
    pub model_dir: String,
    /// True when every required file is present.
    pub ready: bool,
    /// Required files that are missing, relative to `model_dir`.
    pub missing: Vec<String>,
}

#[tauri::command]
pub fn local_tts_status() -> LocalTtsStatus {
    let dir = model_dir().unwrap_or_default();
    let missing = missing_files(&dir);
    LocalTtsStatus {
        model_dir: dir.to_string_lossy().into_owned(),
        ready: missing.is_empty(),
        missing,
    }
}

fn missing_files(dir: &Path) -> Vec<String> {
    let zp = dir.join(ZIPVOICE_DIR);
    let mut missing = Vec::new();
    for rel in REQUIRED_FILES {
        if !zp.join(rel).exists() {
            missing.push(format!("{ZIPVOICE_DIR}/{rel}"));
        }
    }
    if !dir.join("vocos_24khz.onnx").exists() {
        missing.push("vocos_24khz.onnx".to_string());
    }
    if !dir.join("lexicon-en-us.txt").exists() {
        missing.push("lexicon-en-us.txt".to_string());
    }
    missing
}

fn build_engine(dir: &Path) -> Result<TtsEngine, String> {
    let zp = dir.join(ZIPVOICE_DIR);
    let p = |rel: &str| zp.join(rel).to_string_lossy().into_owned();

    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            zipvoice: OfflineTtsZipvoiceModelConfig {
                tokens: Some(p("tokens.txt")),
                encoder: Some(p("encoder.int8.onnx")),
                decoder: Some(p("decoder.int8.onnx")),
                vocoder: Some(dir.join("vocos_24khz.onnx").to_string_lossy().into_owned()),
                lexicon: Some(format!(
                    "{},{}",
                    p("lexicon.txt"),
                    dir.join("lexicon-en-us.txt").to_string_lossy()
                )),
                feat_scale: 0.1,
                t_shift: 0.5,
                target_rms: 0.1,
                guidance_scale: 1.0,
                ..Default::default()
            },
            num_threads: 2,
            debug: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config)
        .ok_or_else(|| "OfflineTts::create returned None (check model files)".to_string())?;

    let ref_path = zp.join(REFERENCE_WAV).to_string_lossy().into_owned();
    let wave = Wave::read(&ref_path)
        .ok_or_else(|| format!("failed to read reference wav: {ref_path}"))?;

    log::info!(
        "local_tts: engine created, sample_rate={}, reference={} samples",
        tts.sample_rate(),
        wave.samples().len()
    );

    Ok(TtsEngine {
        tts,
        reference: wave.samples().to_vec(),
        reference_rate: wave.sample_rate(),
    })
}

fn synth_blocking(text: &str, speed: f32) -> Result<Vec<u8>, String> {
    let dir = model_dir()?;
    let missing = missing_files(&dir);
    if !missing.is_empty() {
        return Err(format!(
            "local TTS models missing in {}: {}",
            dir.to_string_lossy(),
            missing.join(", ")
        ));
    }

    let cell = engine_cell();
    let mut guard = cell.lock().map_err(|e| format!("tts lock poisoned: {e}"))?;
    if guard.is_none() {
        *guard = Some(build_engine(&dir)?);
    }
    let engine = guard.as_ref().ok_or_else(|| "engine not initialized".to_string())?;

    let gen = GenerationConfig {
        speed,
        reference_audio: Some(engine.reference.clone()),
        reference_sample_rate: engine.reference_rate,
        reference_text: Some(REFERENCE_TEXT.to_string()),
        num_steps: 4,
        ..Default::default()
    };

    let audio = engine
        .tts
        .generate_with_config(text, &gen, None::<fn(&[f32], f32) -> bool>)
        .ok_or_else(|| "generation failed".to_string())?;

    Ok(f32_to_wav(audio.samples(), audio.sample_rate()))
}

/// Synthesize `text` and return a 16-bit mono WAV, base64-encoded for IPC.
///
/// Base64 keeps the payload compact across the Tauri boundary (a raw Vec<u8>
/// would serialize as a JSON number array). The frontend decodes it back into
/// an ArrayBuffer for `decodeAudioData`.
#[tauri::command]
pub async fn local_tts_synthesize(text: String, speed: Option<f32>) -> Result<String, String> {
    let speed = speed.unwrap_or(1.0);
    let wav = tauri::async_runtime::spawn_blocking(move || synth_blocking(&text, speed))
        .await
        .map_err(|e| format!("tts task join error: {e}"))??;
    Ok(B64.encode(wav))
}

/// Encode normalized f32 PCM (mono) into a 16-bit WAV byte buffer.
fn f32_to_wav(samples: &[f32], sample_rate: i32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let byte_rate = sample_rate as u32 * channels as u32 * (bits as u32 / 8);
    let block_align = channels * (bits / 8);
    let data_len = (samples.len() * 2) as u32;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}