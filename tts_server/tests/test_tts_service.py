from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.tts_service import build_tts_service


class TtsServiceTests(unittest.TestCase):
    def _build_reference_audio(self, output_path: Path, sample_rate: int = 22050) -> None:
        samples = int(sample_rate * 1.0)
        t = np.arange(samples, dtype=np.float32) / float(sample_rate)
        waveform = 0.2 * np.sin(2.0 * np.pi * 200.0 * t)
        sf.write(str(output_path), waveform, sample_rate)

    def test_batch_mode_returns_non_empty_audio_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_audio = Path(temp_dir) / "reference.wav"
            self._build_reference_audio(reference_audio, sample_rate=22050)

            build_tts_service.cache_clear()
            service = build_tts_service()
            service.initialize()

            result = service.synthesize_batch(
                text="Batch mode test synthesis.",
                reference_audio_path=reference_audio,
            )
            self.assertGreater(len(result.audio_bytes), 0)
            self.assertEqual(result.sample_rate, 16000)

    def test_stream_mode_returns_non_empty_aggregated_audio_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_audio = Path(temp_dir) / "reference.wav"
            self._build_reference_audio(reference_audio, sample_rate=22050)

            build_tts_service.cache_clear()
            service = build_tts_service()
            service.initialize()

            result = service.synthesize_stream(
                text="Stream mode test synthesis.",
                reference_audio_path=reference_audio,
            )
            self.assertGreater(len(result.chunk_bytes), 0)
            self.assertGreater(len(result.aggregated_audio_bytes), 0)
            self.assertEqual(result.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
