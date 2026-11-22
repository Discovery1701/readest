import { invoke } from '@tauri-apps/api/core';
import { isTauriAppPlatform } from '@/services/environment';
import type { TTSVoice } from '../types';
import type {
  SpeechProvider,
  SpeechSynthesisRequest,
  SpeechSynthesisResult,
} from './types';

export const LOCAL_NEURAL_PROVIDER_ID = 'local-neural';
export const LOCAL_NEURAL_VOICE_ID = 'zipvoice-zh-CN';

interface LocalTtsPrepareResult {
  model_dir?: string;
  ready?: boolean;
  missing?: string[];
}

// Backend returns base64 of a 16-bit mono 24kHz WAV. Tolerate a data: prefix.
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

export class LocalNeuralSpeechProvider implements SpeechProvider {
  readonly id = LOCAL_NEURAL_PROVIDER_ID;
  readonly label = '本地语音（ZipVoice）';
  readonly fallbackVoiceId = LOCAL_NEURAL_VOICE_ID;
  readonly cacheable = true;

  #prepare: Promise<boolean> | null = null;

  async init(): Promise<boolean> {
    if (!isTauriAppPlatform()) return false;
    // Availability is Tauri-based only. Do NOT gate it on the status probe:
    // any transient invoke failure would flip initialized=false and grey the
    // whole voice group out even though the engine works. Preparation is lazy
    // and skips straight through when the model files are already present.
    void this.#ensurePrepared();
    return true;
  }

  async getAllVoices(): Promise<TTSVoice[]> {
    return [
      {
        id: LOCAL_NEURAL_VOICE_ID,
        name: 'ZipVoice 中文',
        lang: 'zh-CN',
      },
    ];
  }

  async synthesize(
    req: SpeechSynthesisRequest,
    signal: AbortSignal,
  ): Promise<SpeechSynthesisResult> {
    const ready = await abortable(this.#ensurePrepared(), signal);
    if (!ready) {
      throw new Error('Local TTS model is not ready');
    }

    // Fix speed=1: playback rate is handled by the buffered player's WSOLA so a
    // sentence's audio stays reusable across rates (matches the Edge provider).
    const base64 = await abortable(
      invoke<string>('local_tts_synthesize', {
        text: req.text,
        speed: 1,
      }),
      signal,
    );

    const audio = decodeBase64Wav(base64);
    if (audio.byteLength === 0) {
      throw new Error('Local TTS returned empty audio');
    }

    return { audio, boundaries: [] };
  }

  #ensurePrepared(): Promise<boolean> {
    if (!this.#prepare) {
      this.#prepare = invoke<LocalTtsPrepareResult>('local_tts_prepare')
        .then((result) => {
          const ready = !!result?.ready;
          if (!ready) this.#prepare = null; // allow a later retry
          return ready;
        })
        .catch((err) => {
          console.warn('[TTS] local_tts_prepare failed', err);
          this.#prepare = null;
          return false;
        });
    }
    return this.#prepare;
  }
}