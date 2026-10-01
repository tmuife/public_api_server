import logging
from pathlib import Path

import numpy as np
import torch

from .alignment import aligned_crop, paste_enhanced
from .backends import is_backend_error
from .errors import MediaToolError
from .vendor.gfpgan.gfpganv1_clean_arch import GFPGANv1Clean

logger = logging.getLogger(__name__)


class FaceEnhancer:
    def __init__(self, path: Path, mode: str, blend: float):
        available = torch.backends.mps.is_available()
        if mode == "mps" and not available:
            raise MediaToolError("显式选择 MPS，但当前系统/PyTorch 不支持 MPS")
        self.device = torch.device("mps" if mode != "cpu" and available else "cpu")
        self.mode, self.blend = mode, blend
        self.network = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = checkpoint.get("params_ema", checkpoint.get("params"))
        if weights is None:
            raise MediaToolError("GFPGAN 权重缺少 params_ema/params")
        self.network.load_state_dict(weights, strict=True)
        self.network.eval().to(self.device)

    def restore(self, crop: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float()
        tensor = (tensor / 127.5 - 1).unsqueeze(0)
        try:
            with torch.inference_mode():
                result = self.network(
                    tensor.to(self.device), return_rgb=False, randomize_noise=False
                )[0]
        except (RuntimeError, NotImplementedError) as exc:
            if self.mode != "auto" or self.device.type != "mps" or not is_backend_error(exc):
                raise MediaToolError(f"GFPGAN 推理失败：{exc}") from exc
            logger.warning("GFPGAN MPS 不兼容，切换 CPU：%s", exc)
            self.device = torch.device("cpu")
            self.network.to(self.device)
            with torch.inference_mode():
                result = self.network(tensor, return_rgb=False, randomize_noise=False)[0]
        if not torch.isfinite(result).all():
            raise MediaToolError("GFPGAN 输出包含非有限数值")
        result = result[0].clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
        return np.rint((result[:, :, ::-1] + 1) * 127.5).astype(np.uint8)

    def enhance(self, frame, face):
        crop, matrix = aligned_crop(frame, face.kps, 512, enhance=True)
        return paste_enhanced(frame, self.restore(crop), matrix, self.blend)

    def check(self):
        output = self.restore(np.zeros((512, 512, 3), np.uint8))
        if output.shape != (512, 512, 3):
            raise MediaToolError("GFPGAN 试推理输出规格错误")
