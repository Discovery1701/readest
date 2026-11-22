import type { AppService } from '@/types/system';
import { BufferedTTSClient } from './BufferedTTSClient';
import type { TTSCapabilities } from './TTSClient';
import type { TTSVoicesGroup } from './types';
import type { TTSController } from './TTSController';
import { LocalNeuralSpeechProvider } from './providers/localNeural';

/**
 * Offline neural TTS (ZipVoice via sherpa-onnx, desktop only). Reuses the
 * buffered playback pipeline; audio is decoded with decodeAudioData exactly
 * like the Edge engine.
 */
export class LocalNeuralTTSClient extends BufferedTTSClient {
  constructor(controller?: TTSController, appService?: AppService | null) {
    super(new LocalNeuralSpeechProvider(), controller, appService);
  }

  // Expose the local voice regardless of the book's language: ZipVoice is a
  // zh-en model, so BufferedTTSClient's language-matched filter would otherwise
  // grey the whole group out on non-Chinese books. Only `!initialized` disables.
  override async getVoices(_lang: string): Promise<TTSVoicesGroup[]> {
    const voices = await this.getAllVoices();
    return [
      {
        id: this.name,
        name: this.provider.label,
        voices,
        disabled: !this.initialized,
      },
    ];
  }

  override getCapabilities(): TTSCapabilities {
    return {
      ...super.getCapabilities(),
      // No word timings from the local model -> sentence-level highlight.
      wordBoundaries: false,
    };
  }
}