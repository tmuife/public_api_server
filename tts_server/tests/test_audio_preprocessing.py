from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.utils.audio_processing import prepare_reference_audio


class AudioPreprocessingTests(unittest.TestCase):
    def test_non_16k_audio_is_resampled_with_soxr_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "ref_22k.wav"
            sample_rate = 22050
            duration_seconds = 1.0
            sample_count = int(sample_rate * duration_seconds)
            timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
            waveform = 0.15 * np.sin(2.0 * np.pi * 330.0 * timeline)
            sf.write(str(ref_path), waveform, sample_rate)

            prepared = prepare_reference_audio(
                ref_path,
                target_sample_rate=16000,
                target_channels=1,
            )

            self.assertEqual(prepared.sample_rate, 16000)
            self.assertEqual(prepared.waveform.ndim, 2)
            self.assertEqual(prepared.waveform.shape[1], 1)
            self.assertGreater(prepared.waveform.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
