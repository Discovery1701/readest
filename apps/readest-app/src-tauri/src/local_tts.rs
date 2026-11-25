//! Local neural TTS via sherpa-onnx, desktop only.
//!
//! Model-agnostic: a catalog describes downloadable models. The built-in list
//! can be extended or overridden at runtime by `<exe>/models/catalog.json`, so
//! adding or switching models does not require a rebuild.
//!
//! Each model becomes a selectable "voice"; `local_tts_synthesize` receives the
//! model id and lazily downloads + builds that model on first use. Result is a
//! 16-bit mono WAV, base64-encoded for IPC.
//!
//! Layout: `<exe>/models/<spec.dir>/…` with the model files unpacked in place.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsMatchaModelConfig,
    OfflineTtsModelConfig, OfflineTtsVitsModelConfig, OfflineTtsZipvoiceModelConfig, Wave,
};

/// Sub-directory of the exe folder that holds models + catalog.
const MODELS_SUBDIR: &str = "models";
/// Optional runtime catalog, merged over the built-in model list.
const CATALOG_FILE: &str = "catalog.json";
/// Reference clip inside the ZipVoice model dir (zero-shot voice prompt).
const ZIPVOICE_REFERENCE: &str = "test_wavs/leijun-1.wav";
/// Transcript of the ZipVoice reference clip (must match word for word).
const ZIPVOICE_REFERENCE_TEXT: &str = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";
const ZIPVOICE_ENCODER: &str = "encoder.int8.onnx";
const ZIPVOICE_DECODER: &str = "decoder.int8.onnx";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    #[default]
    Vits,
    Matcha,
    Zipvoice,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ModelParams {
    #[serde(default)]
    pub noise_scale: Option<f32>,
    #[serde(default)]
    pub noise_scale_w: Option<f32>,
    #[serde(default)]
    pub length_scale: Option<f32>,
    #[serde(default)]
    pub num_threads: Option<i32>,
    #[serde(default)]
    pub sid: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub kind: ModelKind,
    /// Directory name under `<exe>/models/` (also the tarball's top directory).
    pub dir: String,
    /// Tarball downloaded on first use (may be omitted if files are shipped).
    #[serde(default)]
    pub url: Option<String>,
    /// Separate vocoder file downloaded next to the model (Matcha / ZipVoice).
    #[serde(default)]
    pub vocoder_url: Option<String>,
    #[serde(default)]
    pub vocoder: Option<String>,
    /// Preferred acoustic model file; falls back to the first `*.onnx` found.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tokens: Option<String>,
    #[serde(default)]
    pub lexicons: Vec<String>,
    /// Text-normalization FSTs (date/number), joined with commas for sherpa.
    #[serde(default)]
    pub rule_fsts: Vec<String>,
    #[serde(default)]
    pub dict_dir: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub size_mb: Option<u32>,
    #[serde(default)]
    pub params: ModelParams,
}

#[derive(Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub label: String,
    pub lang: String,
    /// True when the model files are already present on disk.
    pub ready: bool,
    pub dir: String,
    pub missing: Vec<String>,
    pub size_mb: Option<u32>,
}

struct TtsEngine {
    id: String,
    /// Per-model lock: one `OfflineTts` instance is not safe to drive
    /// concurrently, but unrelated models must not queue behind each other.
    tts: Mutex<OfflineTts>,
    /// ZipVoice only: reference PCM + sample rate cached at build time.
    reference: Option<(Vec<f32>, i32)>,
}

/// Small LRU of loaded engines, most-recently-used last. Reloading a 100 MB+
/// model costs several seconds, and the preload path plus real playback can
/// race on different model ids - a single slot would thrash and make every
/// sentence pay the model load again.
const MAX_ENGINES: usize = 2;

static ENGINES: OnceLock<Mutex<Vec<Arc<TtsEngine>>>> = OnceLock::new();

fn engine_cell() -> &'static Mutex<Vec<Arc<TtsEngine>>> {
    ENGINES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Directory of the running executable (portable build: where the .exe sits).
fn exe_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe failed: {e}"))?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "executable has no parent directory".to_string())
}

fn models_root() -> Result<PathBuf, String> {
    Ok(exe_dir()?.join(MODELS_SUBDIR))
}

/// Debug logging is opt-in. Enable it either by setting `READEST_TTS_DEBUG=1`
/// or by creating an empty `<exe>/models/tts-debug.on` file (delete the file
/// to turn it off again). Checked on every call, so the switch can be flipped
/// without restarting the app - one `exists()` per log line is negligible.
fn debug_enabled() -> bool {
    if matches!(
        std::env::var("READEST_TTS_DEBUG").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    ) {
        return true;
    }
    models_root()
        .map(|root| root.join("tts-debug.on").exists())
        .unwrap_or(false)
}

/// Append a line to `<exe_dir>/tts-debug.log` when debug logging is enabled,
/// so field issues are diagnosable without devtools. Best-effort: logging
/// failures are ignored.
fn debug_log(msg: &str) {
    if !debug_enabled() {
        return;
    }
    use std::io::Write;
    if let Ok(dir) = exe_dir() {
        let path = dir.join("tts-debug.log");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{msg}");
        }
    }
}

fn builtin_models() -> Vec<ModelSpec> {
    let tts_models = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";
    let vocoder_models =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models";
    let mut list = vec![
        ModelSpec {
            id: "piper-zh-xiaoya".to_string(),
            label: "Piper 中文·小雅（快）".to_string(),
            kind: ModelKind::Vits,
            dir: "vits-piper-zh_CN-xiao_ya-medium-int8".to_string(),
            url: Some(format!(
                "{tts_models}/vits-piper-zh_CN-xiao_ya-medium-int8.tar.bz2"
            )),
            vocoder_url: None,
            vocoder: None,
            model: Some("zh_CN-xiao_ya-medium.onnx".to_string()),
            tokens: Some("tokens.txt".to_string()),
            lexicons: vec!["lexicon.txt".to_string()],
            rule_fsts: Vec::new(),
            dict_dir: None,
            lang: Some("zh-CN".to_string()),
            size_mb: Some(13),
            params: ModelParams {
                num_threads: Some(4),
                ..Default::default()
            },
        },
        ModelSpec {
            id: "melo-tts-zh-en".to_string(),
            label: "MeloTTS 中英".to_string(),
            kind: ModelKind::Vits,
            dir: "vits-melo-tts-zh_en".to_string(),
            url: Some(format!("{tts_models}/vits-melo-tts-zh_en.tar.bz2")),
            vocoder_url: None,
            vocoder: None,
            model: Some("model.onnx".to_string()),
            tokens: Some("tokens.txt".to_string()),
            lexicons: vec!["lexicon.txt".to_string()],
            // sherpa-onnx re-applies rule_fsts on every generate call; the
            // MeloTTS date/number FSTs added ~36 s per sentence on Windows, so
            // they are deliberately left out.
            rule_fsts: Vec::new(),
            dict_dir: None,
            lang: Some("zh-CN".to_string()),
            size_mb: Some(159),
            params: ModelParams {
                num_threads: Some(4),
                ..Default::default()
            },
        },
        ModelSpec {
            id: "matcha-zh-baker".to_string(),
            label: "Matcha 中文".to_string(),
            kind: ModelKind::Matcha,
            dir: "matcha-icefall-zh-baker".to_string(),
            url: Some(format!("{tts_models}/matcha-icefall-zh-baker.tar.bz2")),
            vocoder_url: Some(format!("{vocoder_models}/vocos-22khz-univ.onnx")),
            vocoder: Some("vocos-22khz-univ.onnx".to_string()),
            model: Some("model-steps-3.onnx".to_string()),
            tokens: Some("tokens.txt".to_string()),
            lexicons: vec!["lexicon.txt".to_string()],
            rule_fsts: Vec::new(),
            dict_dir: Some("dict".to_string()),
            lang: Some("zh-CN".to_string()),
            size_mb: Some(146),
            params: ModelParams {
                num_threads: Some(4),
                ..Default::default()
            },
        },
        ModelSpec {
            id: "zipvoice-zh-en".to_string(),
            label: "ZipVoice 中英（实验，较慢）".to_string(),
            kind: ModelKind::Zipvoice,
            dir: "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia".to_string(),
            url: Some(format!(
                "{tts_models}/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2"
            )),
            vocoder_url: Some(format!("{vocoder_models}/vocos_24khz.onnx")),
            vocoder: Some("vocos_24khz.onnx".to_string()),
            model: None,
            tokens: Some("tokens.txt".to_string()),
            lexicons: vec!["lexicon.txt".to_string()],
            rule_fsts: Vec::new(),
            dict_dir: None,
            lang: Some("zh-CN".to_string()),
            size_mb: Some(156),
            params: ModelParams {
                num_threads: Some(4),
                ..Default::default()
            },
        },
    ];
    list.shrink_to_fit();
    list
}

fn load_catalog(root: &Path) -> Vec<ModelSpec> {
    let path = root.join(CATALOG_FILE);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    match serde_json::from_str::<Vec<ModelSpec>>(&text) {
        Ok(entries) => entries,
        Err(e) => {
            debug_log(&format!("catalog: parse {} failed: {e}", path.display()));
            Vec::new()
        }
    }
}

/// Built-in models with `catalog.json` entries merged in (same id replaces).
fn all_specs(root: &Path) -> Vec<ModelSpec> {
    let mut specs = builtin_models();
    for extra in load_catalog(root) {
        if let Some(pos) = specs.iter().position(|s| s.id == extra.id) {
            specs[pos] = extra;
        } else {
            specs.push(extra);
        }
    }
    specs
}

fn find_spec<'a>(specs: &'a [ModelSpec], id: &str) -> Result<&'a ModelSpec, String> {
    specs
        .iter()
        .find(|s| s.id == id)
        .ok_or_else(|| format!("unknown local TTS model: {id}"))
}

fn model_dir(root: &Path, spec: &ModelSpec) -> PathBuf {
    root.join(&spec.dir)
}

/// A real ONNX file, not a Git LFS pointer stub. MeloTTS ships a 133-byte
/// `model.int8.onnx` pointer, which must never be picked as the acoustic model.
fn is_model_onnx(path: &Path) -> bool {
    let is_onnx = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("onnx"))
        .unwrap_or(false);
    is_onnx && path.metadata().is_ok_and(|m| m.len() > 4096)
}

/// True when a usable acoustic model file is present. The separately
/// downloaded vocoder lives in the same directory but cannot synthesize alone,
/// so it must not mark the model as ready.
fn model_ready(dir: &Path, spec: &ModelSpec) -> bool {
    resolve_model(spec, dir).is_ok()
}

fn info_of(root: &Path, spec: &ModelSpec) -> ModelInfo {
    let dir = model_dir(root, spec);
    let ready = model_ready(&dir, spec);
    let mut missing = Vec::new();
    if !ready {
        missing.push(spec.dir.clone());
    }
    ModelInfo {
        id: spec.id.clone(),
        label: spec.label.clone(),
        lang: spec.lang.clone().unwrap_or_else(|| "zh-CN".to_string()),
        ready,
        dir: dir.to_string_lossy().into_owned(),
        missing,
        size_mb: spec.size_mb,
    }
}

#[tauri::command]
pub fn local_tts_list_models() -> Vec<ModelInfo> {
    match models_root() {
        Ok(root) => all_specs(&root)
            .iter()
            .map(|spec| info_of(&root, spec))
            .collect(),
        Err(e) => {
            debug_log(&format!("list_models: {e}"));
            Vec::new()
        }
    }
}

#[tauri::command]
pub fn local_tts_status(model_id: Option<String>) -> ModelInfo {
    let Ok(root) = models_root() else {
        return ModelInfo {
            id: String::new(),
            label: String::new(),
            lang: "zh-CN".to_string(),
            ready: false,
            dir: String::new(),
            missing: vec!["cannot resolve executable directory".to_string()],
            size_mb: None,
        };
    };
    let specs = all_specs(&root);
    let spec = model_id
        .as_deref()
        .and_then(|id| specs.iter().find(|s| s.id == id))
        .or_else(|| specs.first());
    match spec {
        Some(spec) => info_of(&root, spec),
        None => ModelInfo {
            id: String::new(),
            label: String::new(),
            lang: "zh-CN".to_string(),
            ready: false,
            dir: String::new(),
            missing: vec!["no models configured".to_string()],
            size_mb: None,
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

async fn download_into(url: &str, dest: &Path) -> Result<(), String> {
    let data = http_get_bytes(url).await?;
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    tokio::fs::write(dest, &data)
        .await
        .map_err(|e| format!("write {}: {e}", dest.display()))
}

/// Download the tarball (if needed) and the separate vocoder for `model_id`.
#[tauri::command]
pub async fn local_tts_prepare(model_id: String) -> Result<ModelInfo, String> {
    let root = models_root()?;
    let specs = all_specs(&root);
    let spec = find_spec(&specs, &model_id)?;
    debug_log(&format!("prepare: model={} root={}", spec.id, root.display()));
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|e| format!("mkdir {}: {e}", root.display()))?;

    let dir = model_dir(&root, spec);
    if !model_ready(&dir, spec) {
        let url = spec
            .url
            .as_ref()
            .ok_or_else(|| format!("model {} has no download url", spec.id))?;
        debug_log(&format!("prepare: downloading tarball {url}"));
        let data = http_get_bytes(url).await?;
        let dest = root.clone();
        tauri::async_runtime::spawn_blocking(move || extract_tar_bz2(&data, &dest))
            .await
            .map_err(|e| format!("untar task join: {e}"))??;
    }

    let dir = model_dir(&root, spec);
    if let (Some(url), Some(name)) = (&spec.vocoder_url, &spec.vocoder) {
        let dest = dir.join(name);
        if !dest.exists() {
            debug_log(&format!("prepare: downloading vocoder {url}"));
            download_into(url, &dest).await?;
        }
    }

    let info = info_of(&root, spec);
    debug_log(&format!("prepare: ready={}", info.ready));
    if info.ready {
        // Warm the engine off the request path. Loading a model (170 MB for
        // MeloTTS) takes several seconds; paying that on the first sentence
        // looks like a hang to the reader.
        let warm_root = root.clone();
        let warm_spec = spec.clone();
        let _ = tauri::async_runtime::spawn_blocking(move || {
            let already = engine_cell()
                .lock()
                .map(|guard| guard.iter().any(|e| e.id == warm_spec.id))
                .unwrap_or(false);
            if already {
                return;
            }
            match build_engine(&warm_root, &warm_spec) {
                Ok(engine) => {
                    debug_log(&format!("prepare: engine warmed ({})", warm_spec.id));
                    if let Ok(mut guard) = engine_cell().lock() {
                        if guard.len() >= MAX_ENGINES {
                            guard.remove(0);
                        }
guard.push(Arc::new(engine));
                    }
                }
                Err(e) => debug_log(&format!("prepare: warm failed: {e}")),
            }
        });
    }
    Ok(info)
}

fn resolve_model(spec: &ModelSpec, dir: &Path) -> Result<String, String> {
    if let Some(name) = &spec.model {
        let candidate = dir.join(name);
        if is_model_onnx(&candidate) {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    let mut onnx: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_model_onnx(p))
        .filter(|p| {
            spec.vocoder
                .as_ref()
                .is_none_or(|v| p.file_name().and_then(|s| s.to_str()) != Some(v.as_str()))
        })
        .collect();
    onnx.sort();
    onnx.into_iter()
        .next()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| format!("no .onnx model file found in {}", dir.display()))
}

fn build_engine(root: &Path, spec: &ModelSpec) -> Result<TtsEngine, String> {
    let dir = model_dir(root, spec);
    let p = |rel: &str| dir.join(rel).to_string_lossy().into_owned();
    let model = resolve_model(spec, &dir)?;
    let tokens = p(spec.tokens.as_deref().unwrap_or("tokens.txt"));
    let lexicons: Vec<String> = spec
        .lexicons
        .iter()
        .map(|rel| dir.join(rel))
        .filter(|path| path.exists())
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    let lexicon = if lexicons.is_empty() {
        None
    } else {
        Some(lexicons.join(","))
    };
    let num_threads = spec.params.num_threads.unwrap_or(4);

    let mut model_cfg = OfflineTtsModelConfig {
        num_threads,
        debug: false,
        ..Default::default()
    };
    match spec.kind {
        ModelKind::Vits => {
            model_cfg.vits = OfflineTtsVitsModelConfig {
                model: Some(model),
                tokens: Some(tokens),
                lexicon,
                noise_scale: spec.params.noise_scale.unwrap_or(0.667),
                noise_scale_w: spec.params.noise_scale_w.unwrap_or(0.8),
                length_scale: spec.params.length_scale.unwrap_or(1.0),
                ..Default::default()
            };
        }
        ModelKind::Matcha => {
            model_cfg.matcha = OfflineTtsMatchaModelConfig {
                acoustic_model: Some(model),
                vocoder: spec.vocoder.as_ref().map(|name| p(name)),
                lexicon,
                tokens: Some(tokens),
                dict_dir: spec.dict_dir.as_ref().map(|rel| p(rel)),
                noise_scale: spec.params.noise_scale.unwrap_or(0.667),
                length_scale: spec.params.length_scale.unwrap_or(1.0),
                ..Default::default()
            };
        }
        ModelKind::Zipvoice => {
            model_cfg.zipvoice = OfflineTtsZipvoiceModelConfig {
                tokens: Some(tokens),
                encoder: Some(p(ZIPVOICE_ENCODER)),
                decoder: Some(p(ZIPVOICE_DECODER)),
                vocoder: spec.vocoder.as_ref().map(|name| p(name)),
                lexicon,
                feat_scale: 0.1,
                t_shift: 0.5,
                target_rms: 0.1,
                guidance_scale: 1.0,
                ..Default::default()
            };
        }
    }

    let rule_fsts: Vec<String> = spec
        .rule_fsts
        .iter()
        .map(|rel| dir.join(rel))
        .filter(|path| path.exists())
        .map(|path| path.to_string_lossy().into_owned())
        .collect();

    let config = OfflineTtsConfig {
        model: model_cfg,
        rule_fsts: if rule_fsts.is_empty() {
            None
        } else {
            Some(rule_fsts.join(","))
        },
        // The Rust struct derives Default (0 / 0.0), while the C++ defaults are
        // 1 / 0.2. Set them explicitly so long text is processed sentence by
        // sentence and sentences are separated by a short silence.
        max_num_sentences: 1,
        silence_scale: 0.2,
        ..Default::default()
    };

    debug_log(&format!(
        "build_engine: creating OfflineTts (model={}, kind={:?}, dir={})",
        spec.id,
        spec.kind,
        dir.display()
    ));
    let tts = match OfflineTts::create(&config) {
        Some(t) => {
            debug_log(&format!("build_engine: OfflineTts created (sr={})", t.sample_rate()));
            t
        }
        None => {
            debug_log("build_engine: OfflineTts::create returned None");
            return Err("OfflineTts::create returned None (check model files)".to_string());
        }
    };

    let reference = if spec.kind == ModelKind::Zipvoice {
        let path = dir.join(ZIPVOICE_REFERENCE);
        match Wave::read(&path.to_string_lossy()) {
            Some(wave) => Some((wave.samples().to_vec(), wave.sample_rate())),
            None => {
                debug_log(&format!(
                    "build_engine: failed to read reference {}",
                    path.display()
                ));
                None
            }
        }
    } else {
        None
    };

    Ok(TtsEngine {
        id: spec.id.clone(),
        tts: Mutex::new(tts),
        reference,
    })
}

fn synth_blocking(model_id: &str, text: &str, speed: f32) -> Result<Vec<u8>, String> {
    let root = models_root()?;
    let specs = all_specs(&root);
    let spec = find_spec(&specs, model_id)?;
    let dir = model_dir(&root, spec);
    if !model_ready(&dir, spec) {
        return Err(format!(
            "model {} is not downloaded in {} (call local_tts_prepare first)",
            spec.id,
            dir.display()
        ));
    }

    let cell = engine_cell();
    // Hold the registry lock only for lookup/insertion: synthesis itself must
    // not serialize requests for unrelated models.
    let entry = {
        let mut guard = cell.lock().map_err(|e| format!("tts lock poisoned: {e}"))?;
        if let Some(idx) = guard.iter().position(|e| e.id == spec.id) {
            let entry = guard.remove(idx);
            guard.push(entry.clone());
            entry
        } else {
            let entry = Arc::new(build_engine(&root, spec)?);
            if guard.len() >= MAX_ENGINES {
                guard.remove(0);
            }
            guard.push(entry.clone());
            entry
        }
    };

    let mut gen = GenerationConfig {
        speed,
        sid: spec.params.sid.unwrap_or(0),
        ..Default::default()
    };
    if let Some((reference, rate)) = &entry.reference {
        gen.reference_audio = Some(reference.clone());
        gen.reference_sample_rate = *rate;
        gen.reference_text = Some(ZIPVOICE_REFERENCE_TEXT.to_string());
        gen.num_steps = 4;
    }

    debug_log(&format!(
        "synth: model={} text_len={} generating…",
        spec.id,
        text.chars().count()
    ));
    let tts = entry
        .tts
        .lock()
        .map_err(|e| format!("tts lock poisoned: {e}"))?;
    let audio = match tts.generate_with_config(text, &gen, None::<fn(&[f32], f32) -> bool>) {
        Some(a) => a,
        None => {
            debug_log("synth: generate_with_config returned None");
            return Err("generation failed".to_string());
        }
    };
    let sr = audio.sample_rate();
    let n = audio.samples().len();
    debug_log(&format!("synth: ok samples={n} rate={sr}"));
    let wav = f32_to_wav(audio.samples(), sr);
    drop(audio);
    Ok(wav)
}

/// Synthesize `text` with `model_id` and return a 16-bit mono WAV, base64-encoded.
///
/// Base64 keeps the payload compact across the Tauri boundary (a raw Vec<u8>
/// would serialize as a JSON number array). The frontend decodes it back into
/// an ArrayBuffer for `decodeAudioData`.
#[tauri::command]
pub async fn local_tts_synthesize(
    model_id: String,
    text: String,
    speed: Option<f32>,
) -> Result<String, String> {
    let speed = speed.unwrap_or(1.0);
    let result =
        tauri::async_runtime::spawn_blocking(move || synth_blocking(&model_id, &text, speed))
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