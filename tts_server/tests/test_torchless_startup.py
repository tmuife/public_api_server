from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


class _BlockTorchImporter:
    def find_spec(self, fullname, path, target=None):
        root_name = fullname.split(".")[0]
        if root_name in {"torch", "torchaudio"}:
            raise ImportError(f"Blocked optional dependency import: {fullname}")
        return None


class TorchlessStartupTests(unittest.TestCase):
    def _write_reference_audio(self, path: Path, sample_rate: int = 22050) -> None:
        samples = int(sample_rate * 1.0)
        timeline = np.arange(samples, dtype=np.float32) / float(sample_rate)
        waveform = 0.15 * np.sin(2.0 * np.pi * 240.0 * timeline)
        sf.write(str(path), waveform, sample_rate)

    def test_service_startup_and_inference_without_torch_modules(self) -> None:
        blocker = _BlockTorchImporter()
        sys.meta_path.insert(0, blocker)
        try:
            for module_name in list(sys.modules):
                root = module_name.split(".")[0]
                if root in {"app", "main", "torch", "torchaudio"}:
                    sys.modules.pop(module_name, None)

            service_module = importlib.import_module("app.services.tts_service")
            service_module.build_tts_service.cache_clear()
            service = service_module.build_tts_service()
            service.initialize()

            with tempfile.TemporaryDirectory() as temp_dir:
                reference_audio = Path(temp_dir) / "reference.wav"
                self._write_reference_audio(reference_audio)
                batch_result = service.synthesize_batch(
                    text="Torchless startup test.",
                    reference_audio_path=reference_audio,
                )
                self.assertGreater(len(batch_result.audio_bytes), 0)
        finally:
            sys.meta_path = [item for item in sys.meta_path if item is not blocker]


if __name__ == "__main__":
    unittest.main()
