from __future__ import annotations


class PassthroughFrameProcessor:
    def process(self, frame_bytes: bytes, content_type: str) -> bytes:
        _ = content_type
        return frame_bytes
