import { invoke } from '@tauri-apps/api/core';
import { isTauriAppPlatform } from '@/services/environment';
import type { TTSVoice } from '../types';
import type {
  SpeechProvider,
  SpeechSynthesisRequest,
  SpeechSynthesisResult,
} from './types';

export const LOCAL_NEURAL_PROVIDER_ID = 'local-neural';
/** Fallback when the backend catalog cannot be read. */
export const LOCAL_NEURAL_DEFAULT_VOICE_ID = 'piper-zh-xiaoya';

interface LocalTtsModelInfo {
  id: string;
  label: string;
  lang: string;
  ready: boolean;
  dir: string;
  missing: string[];
  size_mb?: number | null;
}

const logToBackend = (msg: string) => {
  void invoke('local_tts_log', { msg }).catch(() => {});
};

// Backend returns base64 of a 16-bit mono WAV. Tolerate a data: prefix.
const decodeBase64Wav = (base64: string): ArrayBuffer => {
  const comma = base64.indexOf(',');
  const payload = (comma >= 0 ? base64.slice(comma + 1) : base64).replace(/\s/g, '');
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer as ArrayBuffer;
};

// invoke() cannot be cancelled; wrap it so stop()/chapter switching can bail out
// instead of hanging inside synthesis (the failure mode the Edge path suffers).
const abortable = <T>(promise: Promise<T>, signal: AbortSignal): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      reject(new Error('Aborted'));
      return;
    }
    const onAbort = () => reject(new Error('Aborted'));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });

/**
 * Local neural TTS backed by sherpa-onnx.
 *
 * The selectable voices are the models advertised by the Rust catalog
 * (`local_tts_list_models`), so switching models needs no rebuild. A model is
 * downloaded the first time it is selected, not upfront.
 */
export class LocalNeuralSpeechProvider implements SpeechProvider {
  readonly id = LOCAL_NEURAL_PROVIDER_ID;
  readonly label = '本地语音';
  readonly fallbackVoiceId = LOCAL_NEURAL_DEFAULT_VOICE_ID;
  readonly cacheable = true;

  #models: LocalTtsModelInfo[] | null = null;
  #modelsPromise: Promise<LocalTtsModelInfo[]> | null = null;
  #prepare = new Map<string, Promise<boolean>>();
  /**
   * Short-lived PCM cache keyed by model+voice+text. `TTSController` first
   * synthesizes upcoming sentences to preload them and then synthesizes the
   * same text again for playback; without this the reader waits twice.
   */
  #audioCache = new Map<string, Uint8Array>();

  async init(): Promise<boolean> {
    logToBackend(`init: isTauri=${isTauriAppPlatform()}`);
    if (!isTauriAppPlatform()) return false;
    // Availability is Tauri-based only. Do NOT gate it on the catalog probe:
    // any transient invoke failure would flip initialized=false and grey the
    // whole voice group out. Model prep is lazy and happens on selection.
    void this.#loadModels();
    return true;
  }

  async getAllVoices(): Promise<TTSVoice[]> {
    const models = await this.#loadModels();
    if (models.length === 0) {
      return [
        {
          id: LOCAL_NEURAL_DEFAULT_VOICE_ID,
          name: 'Piper 中文·小雅（快）',
          lang: 'zh-CN',
        },
      ];
    }
    return models.map((model) => ({
      id: model.id,
      name: model.ready
        ? model.label
        : `${model.label} · 需下载${model.size_mb ? ` ~${model.size_mb}MB` : ''}`,
      lang: model.lang || 'zh-CN',
    }));
  }

  async synthesize(
    req: SpeechSynthesisRequest,
    signal: AbortSignal,
  ): Promise<SpeechSynthesisResult> {
    const modelId = req.voice || LOCAL_NEURAL_DEFAULT_VOICE_ID;
    const cacheKey = `${modelId}\u0000${req.pitch}\u0000${req.text}`;
    const cached = this.#audioCache.get(cacheKey);
    if (cached) {
      // decodeAudioData() detaches the buffer it is handed, so return a copy.
      return { audio: cached.slice().buffer as ArrayBuffer, boundaries: [] };
    }
    logToBackend(`synth: model=${modelId} len=${req.text.length}`);
    const ready = await abortable(this.#ensurePrepared(modelId), signal);
    logToBackend(`synth: model=${modelId} prepared=${ready}`);
    if (!ready) {
      throw new Error(`Local TTS model ${modelId} is not ready`);
    }

    // Fix speed=1: playback rate is handled by the buffered player's WSOLA so a
    // sentence's audio stays reusable across rates (matches the Edge provider).
    const base64 = await abortable(
      invoke<string>('local_tts_synthesize', {
        modelId,
        text: req.text,
        speed: 1,
      }),
      signal,
    );
    logToBackend(`synth: base64 len=${base64?.length ?? 0}`);

    const audio = decodeBase64Wav(base64);
    if (audio.byteLength === 0) {
      throw new Error('Local TTS returned empty audio');
    }

    if (this.#audioCache.size >= 16) {
      const oldest = this.#audioCache.keys().next().value;
      if (oldest !== undefined) this.#audioCache.delete(oldest);
    }
    this.#audioCache.set(cacheKey, new Uint8Array(audio));

    return { audio, boundaries: [] };
  }

  #loadModels(): Promise<LocalTtsModelInfo[]> {
    if (this.#models) return Promise.resolve(this.#models);
    if (!this.#modelsPromise) {
      this.#modelsPromise = invoke<LocalTtsModelInfo[]>('local_tts_list_models')
        .then((models) => {
          this.#models = Array.isArray(models) ? models : [];
          logToBackend(
            `list_models: ${this.#models!.map((m) => `${m.id}:${m.ready}`).join(',')}`,
          );
          return this.#models!;
        })
        .catch((err) => {
          logToBackend(`list_models: error ${String(err)}`);
          this.#modelsPromise = null;
          return [];
        });
    }
    return this.#modelsPromise;
  }

  #ensurePrepared(modelId: string): Promise<boolean> {
    let pending = this.#prepare.get(modelId);
    if (!pending) {
      logToBackend(`prepare: invoking local_tts_prepare(${modelId})`);
      pending = invoke<LocalTtsModelInfo>('local_tts_prepare', { modelId })
        .then((info) => {
          const ready = !!info?.ready;
          logToBackend(`prepare(${modelId}): ready=${ready}`);
          if (!ready) this.#prepare.delete(modelId); // allow a later retry
          // A freshly downloaded model now shows without the "download" hint.
          this.#models = null;
          this.#modelsPromise = null;
          return ready;
        })
        .catch((err) => {
          logToBackend(`prepare(${modelId}): error ${String(err)}`);
          this.#prepare.delete(modelId);
          return false;
        });
      this.#prepare.set(modelId, pending);
    }
    return pending;
  }
}