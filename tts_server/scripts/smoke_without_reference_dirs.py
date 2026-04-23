from __future__ import annotations

import importlib
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_DIRS = (
    Path("MOSS-TTS-Nano-main"),
    Path("Kokoro-FastAPI-master"),
)
DEFAULT_SMOKE_API_KEY = "smoke-test-token-12345"


class _BlockTorchImporter:
    def find_spec(self, fullname: str, path: object, target: object = None):  # noqa: ANN001
        root_name = fullname.split(".")[0]
        if root_name in {"torch", "torchaudio"}:
            raise ImportError(f"Blocked optional dependency import: {fullname}")
        return None


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    detail: str = ""


def _generate_reference_audio(path: Path, sample_rate: int = 22050, seconds: float = 1.0) -> None:
    sample_count = int(sample_rate * seconds)
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    waveform = 0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)
    sf.write(str(path), waveform, sample_rate)


def _clear_runtime_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("main."):
            sys.modules.pop(module_name, None)
            continue
        if module_name == "app" or module_name.startswith("app."):
            sys.modules.pop(module_name, None)
            continue
        if module_name == "torch" or module_name.startswith("torch."):
            sys.modules.pop(module_name, None)
            continue
        if module_name == "torchaudio" or module_name.startswith("torchaudio."):
            sys.modules.pop(module_name, None)


def _resolve_smoke_api_key() -> str:
    value = str(os.getenv("SMOKE_API_KEY", DEFAULT_SMOKE_API_KEY)).strip()
    return value if value else DEFAULT_SMOKE_API_KEY


def _configure_auth_env() -> str:
    api_key = _resolve_smoke_api_key()
    os.environ.setdefault("AUTH_REQUIRED", "true")
    os.environ["API_KEY"] = api_key
    return api_key


def _build_auth_headers() -> dict[str, str]:
    api_key = _resolve_smoke_api_key()
    return {"Authorization": f"Bearer {api_key}"}


def _scenario_runtime_batch_stream_without_reference_dirs() -> None:
    _clear_runtime_modules()
    service_module = importlib.import_module("app.services.tts_service")
    service_module.build_tts_service.cache_clear()
    service = service_module.build_tts_service()
    service.initialize()

    with tempfile.TemporaryDirectory() as temp_dir:
        ref_audio_path = Path(temp_dir) / "reference.wav"
        _generate_reference_audio(ref_audio_path, sample_rate=22050)

        batch = service.synthesize_batch(
            text="Smoke test after removing reference directories.",
            reference_audio_path=ref_audio_path,
        )
        stream = service.synthesize_stream(
            text="Smoke test after removing reference directories.",
            reference_audio_path=ref_audio_path,
        )

    if len(batch.audio_bytes) <= 0:
        raise RuntimeError("Batch synthesis returned empty bytes")
    if len(stream.aggregated_audio_bytes) <= 0:
        raise RuntimeError("Stream synthesis returned empty bytes")


def _scenario_openai_endpoints_without_reference_dirs() -> None:
    _clear_runtime_modules()
    _configure_auth_env()
    main_module = importlib.import_module("main")
    app = main_module.create_app()
    headers = _build_auth_headers()

    with TestClient(app, base_url="http://testserver/v1") as client:
        models_response = client.get("/models", headers=headers)
        if models_response.status_code != 200:
            raise RuntimeError(f"/v1/models failed with status {models_response.status_code}")

        speech_response = client.post(
            "/audio/speech",
            json={
                "model": "tts-1",
                "input": "Smoke test openai compatible endpoint.",
                "voice": "alloy",
                "response_format": "wav",
                "stream": False,
            },
            headers=headers,
        )
        if speech_response.status_code != 200:
            raise RuntimeError(f"/v1/audio/speech failed with status {speech_response.status_code}")
        if len(speech_response.content) <= 0:
            raise RuntimeError("/v1/audio/speech returned empty audio bytes")

        canonical_voice_response = client.post(
            "/audio/speech",
            json={
                "model": "tts-1",
                "input": "Smoke test canonical moss voice mapping.",
                "voice": "Junhao",
                "response_format": "wav",
                "stream": False,
            },
            headers=headers,
        )
        if canonical_voice_response.status_code != 200:
            raise RuntimeError(
                f"/v1/audio/speech (canonical voice) failed with status {canonical_voice_response.status_code}"
            )


def _scenario_torchless_startup_without_reference_dirs() -> None:
    blocker = _BlockTorchImporter()
    sys.meta_path.insert(0, blocker)
    try:
        _clear_runtime_modules()
        service_module = importlib.import_module("app.services.tts_service")
        service_module.build_tts_service.cache_clear()
        service = service_module.build_tts_service()
        service.initialize()

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_audio = Path(temp_dir) / "reference.wav"
            _generate_reference_audio(reference_audio, sample_rate=22050)

            batch = service.synthesize_batch(
                text="Torchless startup smoke test.",
                reference_audio_path=reference_audio,
            )
            if len(batch.audio_bytes) <= 0:
                raise RuntimeError("Torchless startup path returned empty bytes")

        _configure_auth_env()
        main_module = importlib.import_module("main")
        app = main_module.create_app()
        headers = _build_auth_headers()
        with TestClient(app, base_url="http://testserver/v1") as client:
            response = client.post(
                "/audio/speech",
                json={
                    "model": "tts-1",
                    "input": "Torchless endpoint smoke test.",
                    "voice": "alloy",
                    "response_format": "wav",
                    "stream": False,
                },
                headers=headers,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Torchless /v1/audio/speech failed with status {response.status_code}")
            if len(response.content) <= 0:
                raise RuntimeError("Torchless /v1/audio/speech returned empty bytes")
    finally:
        sys.meta_path = [item for item in sys.meta_path if item is not blocker]


def _run_scenario(name: str, scenario: Callable[[], None]) -> ScenarioResult:
    print(f"[RUN ] {name}")
    try:
        scenario()
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        return ScenarioResult(name=name, passed=False, detail=str(exc))

    print(f"[PASS] {name}")
    return ScenarioResult(name=name, passed=True, detail="")


def _detach_reference_dirs() -> list[tuple[Path, Path]]:
    renamed: list[tuple[Path, Path]] = []
    for directory in REFERENCE_DIRS:
        if not directory.exists():
            continue
        shadow = directory.with_name(f".{directory.name}.tmp-smoke")
        if shadow.exists():
            raise RuntimeError(f"Shadow path already exists: {shadow}")
        os.replace(directory, shadow)
        renamed.append((directory, shadow))
    return renamed


def _restore_reference_dirs(renamed: list[tuple[Path, Path]]) -> None:
    for original, shadow in reversed(renamed):
        if shadow.exists():
            os.replace(shadow, original)


def main() -> int:
    renamed: list[tuple[Path, Path]] = []
    scenarios: list[tuple[str, Callable[[], None]]] = [
        (
            "Runtime batch/stream remains usable without reference directories",
            _scenario_runtime_batch_stream_without_reference_dirs,
        ),
        (
            "OpenAI-compatible /v1 endpoints remain callable without reference directories",
            _scenario_openai_endpoints_without_reference_dirs,
        ),
        (
            "ONNX-only startup remains callable when torch and torchaudio are unavailable",
            _scenario_torchless_startup_without_reference_dirs,
        ),
    ]

    try:
        renamed = _detach_reference_dirs()
        results = [_run_scenario(name, scenario) for name, scenario in scenarios]
        passed_count = sum(1 for item in results if item.passed)
        total_count = len(results)

        print(f"Acceptance summary: {passed_count}/{total_count} scenarios passed")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            detail = f" ({result.detail})" if result.detail else ""
            print(f"- [{status}] {result.name}{detail}")

        return 0 if passed_count == total_count else 1
    finally:
        _restore_reference_dirs(renamed)


if __name__ == "__main__":
    raise SystemExit(main())
