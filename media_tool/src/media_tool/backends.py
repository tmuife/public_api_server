import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .errors import MediaToolError

logger = logging.getLogger(__name__)


def is_backend_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        word in message
        for word in (
            "coreml",
            "mlmodel",
            "mps",
            "not implemented",
            "notimplemented",
            "unsupported operator",
            "not currently implemented",
        )
    )


class OnnxModel:
    def __init__(self, path: Path, mode: str):
        self.path, self.mode = path, mode
        available = ort.get_available_providers()
        if mode == "coreml" and "CoreMLExecutionProvider" not in available:
            raise MediaToolError("显式选择 CoreML，但当前 ONNX Runtime 不支持该后端")
        providers = ["CPUExecutionProvider"]
        if mode != "cpu" and "CoreMLExecutionProvider" in available:
            providers.insert(0, "CoreMLExecutionProvider")
        try:
            self.session = self._create(providers)
        except Exception as exc:
            if mode != "auto" or len(providers) == 1 or not is_backend_error(exc):
                raise MediaToolError(f"模型初始化失败 {path.name}: {exc}") from exc
            logger.warning("%s CoreML 初始化不兼容，切换 CPU：%s", path.name, exc)
            self.session = self._create(["CPUExecutionProvider"])
        self.session.disable_fallback()

    def _create(self, providers: list[str]) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        return ort.InferenceSession(str(self.path), sess_options=options, providers=providers)

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        try:
            return self.session.run(None, feeds)
        except Exception as exc:
            providers = self.session.get_providers()
            if (
                self.mode == "auto"
                and "CoreMLExecutionProvider" in providers
                and is_backend_error(exc)
            ):
                logger.warning("%s CoreML 试推理不兼容，切换 CPU：%s", self.path.name, exc)
                self.session = self._create(["CPUExecutionProvider"])
                self.session.disable_fallback()
                return self.session.run(None, feeds)
            raise MediaToolError(f"模型推理失败 {self.path.name}: {exc}") from exc

    def describe(self) -> dict:
        return {
            "model": self.path.name,
            "providers": self.session.get_providers(),
            "note": "提供器列表不代表全部算子均由加速器执行",
        }
