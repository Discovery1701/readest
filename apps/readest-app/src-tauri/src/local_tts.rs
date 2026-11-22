//! Local neural TTS (ZipVoice via sherpa-onnx), desktop only.
//!
//! This backs a self-hosted "local TTS" engine: no cloud, no account. Models
//! live next to the portable executable under `<exe_dir>/models/zipvoice/`
//! (deliberately NOT the user profile, so a portable build stays self-contained).
//!
//! `local_tts_prepare` downloads the models on first use; `local_tts_synthesize`
//! turns text into a 16-bit mono WAV (base64 for IPC). The reference clip
//! (`leijun-1.wav`) + its transcript define the default voice via ZipVoice
//! zero-shot cloning, so it works out of the box.
//!
//! Expected layout under models/zipvoice/ after prepare:
//!   vocos_24khz.onnx
//!   lexicon-en-us.txt
//!   sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/
//!     encoder.int8.onnx  decoder.int8.onnx  tokens.txt  lexicon.txt
//!     test_wavs/leijun-1.wav

use std::io::Cursor;
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

const URL_ZIPVOICE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2";
const URL_VOCODER: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx";
const URL_LEXICON_EN: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt";

/// Files that must exist (relative to the ZipVoice model subdir).
const REQUIRED_ZW_FILES: &[&str] = &[
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

/// Append a line to `<exe_dir>/tts-debug.log` so field issues are diagnosable
/// without devtools. Best-effort: logging failures are ignored.
fn debug_log(msg: &str) {
    use std::io::Write;
    if let Ok(dir) = exe_dir() {
        let path = dir.join("tts-debug.log");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{msg}");
        }
    }
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

fn missing_files(dir: &Path) -> Vec<String> {
    let zp = dir.join(ZIPVOICE_DIR);
    let mut missing = Vec::new();
    for rel in REQUIRED_ZW_FILES {
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

fn status_of(dir: &Path) -> LocalTtsStatus {
    let missing = missing_files(dir);
    LocalTtsStatus {
        model_dir: dir.to_string_lossy().into_owned(),
        ready: missing.is_empty(),
        missing,
    }
}

#[tauri::command]
pub fn local_tts_status() -> LocalTtsStatus {
    match model_dir() {
        Ok(dir) => status_of(&dir),
        Err(e) => LocalTtsStatus {
            model_dir: String::new(),
            ready: false,
            missing: vec![e],
        },
    }
}

/// Frontend-side tracing helper (writes into the same `<exe>/tts-debug.log`).
#[tauri::command]
pub fn local_tts_log(msg: String) {
    debug_log(&format!("[frontend] {msg}"));
}

async fn http_get_bytes(url: &str) -> Result<Vec<u8>, String> {
    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("GET {url} failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url} -> HTTP {}", resp.status()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("read body {url}: {e}"))?;
    Ok(bytes.to_vec())
}

fn extract_tar_bz2(data: &[u8], dest: &Path) -> Result<(), String> {
    let decoder = bzip2::read::BzDecoder::new(Cursor::new(data));
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest)
        .map_err(|e| format!("untar into {} failed: {e}", dest.display()))?;
    Ok(())
}

/// Download any missing model files into `<exe_dir>/models/zipvoice/`.
///
/// Idempotent: already-present files are skipped. ~156 MB on first run.
#[tauri::command]
pub async fn local_tts_prepare() -> Result<LocalTtsStatus, String> {
    let dir = model_dir()?;
    debug_log(&format!("prepare: start dir={}", dir.display()));
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let zp = dir.join(ZIPVOICE_DIR);

    if REQUIRED_ZW_FILES.iter().any(|rel| !zp.join(rel).exists()) {
        log::info!("local_tts: downloading ZipVoice package…");
        let data = http_get_bytes(URL_ZIPVOICE).await?;
        let dest = dir.clone();
        tauri::async_runtime::spawn_blocking(move || extract_tar_bz2(&data, &dest))
            .await
            .map_err(|e| format!("untar task join: {e}"))??;
    }

    if !dir.join("vocos_24khz.onnx").exists() {
        log::info!("local_tts: downloading vocoder…");
        let data = http_get_bytes(URL_VOCODER).await?;
        tokio::fs::write(dir.join("vocos_24khz.onnx"), &data)
            .await
            .map_err(|e| format!("write vocoder: {e}"))?;
    }

    if !dir.join("lexicon-en-us.txt").exists() {
        log::info!("local_tts: downloading en lexicon…");
        let data = http_get_bytes(URL_LEXICON_EN).await?;
        tokio::fs::write(dir.join("lexicon-en-us.txt"), &data)
            .await
            .map_err(|e| format!("write lexicon: {e}"))?;
    }

    Ok(status_of(&dir))
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

    debug_log(&format!("build_engine: creating OfflineTts (dir={})", dir.display()));
    let tts = match OfflineTts::create(&config) {
        Some(t) => {
            debug_log("build_engine: OfflineTts created");
            t
        }
        None => {
            debug_log("build_engine: OfflineTts::create returned None");
            return Err("OfflineTts::create returned None (check model files)".to_string());
        }
    };

    let ref_path = zp.join(REFERENCE_WAV).to_string_lossy().into_owned();
    let wave =
        Wave::read(&ref_path).ok_or_else(|| format!("failed to read reference wav: {ref_path}"))?;

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
            "local TTS models missing in {} (call local_tts_prepare): {}",
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

    debug_log(&format!("synth: text_len={} generating…", text.chars().count()));
    let audio = match engine
        .tts
        .generate_with_config(text, &gen, None::<fn(&[f32], f32) -> bool>)
    {
        Some(a) => a,
        None => {
            debug_log("synth: generate_with_config returned None");
            return Err("generation failed".to_string());
        }
    };
    let sr = audio.sample_rate();
    let n = audio.samples().len();
    debug_log(&format!("synth: ok samples={n} rate={sr}"));
    Ok(f32_to_wav(audio.samples(), sr))
}

/// Synthesize `text` and return a 16-bit mono WAV, base64-encoded for IPC.
///
/// Base64 keeps the payload compact across the Tauri boundary (a raw Vec<u8>
/// would serialize as a JSON number array). The frontend decodes it back into
/// an ArrayBuffer for `decodeAudioData`.
#[tauri::command]
pub async fn local_tts_synthesize(text: String, speed: Option<f32>) -> Result<String, String> {
    let speed = speed.unwrap_or(1.0);
    let result = tauri::async_runtime::spawn_blocking(move || synth_blocking(&text, speed))
        .await
        .map_err(|e| format!("tts task join error: {e}"))?;
    match result {
        Ok(wav) => Ok(B64.encode(wav)),
        Err(e) => {
            debug_log(&format!("synthesize error: {e}"));
            Err(e)
        }
    }
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