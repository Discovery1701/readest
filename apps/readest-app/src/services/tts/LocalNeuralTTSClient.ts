import type { AppService } from '@/types/system';
import { BufferedTTSClient } from './BufferedTTSClient';
import type { TTSCapabilities } from './TTSClient';
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

  override getCapabilities(): TTSCapabilities {
    return {
      ...super.getCapabilities(),
      // No word timings from the local model -> sentence-level highlight.
      wordBoundaries: false,
    };
  }
}