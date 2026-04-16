# splade_embedding_server

`splade_embedding_server` 是本仓库中的一个独立 FastAPI 子服务，当前主要用于嵌入相关能力（与 `embedding_server` 路由结构接近）。

## 当前状态

- 依赖管理：`uv`（已从 Poetry 迁移）
- 启动方式：`uv run python main.py`
- 默认文档地址：`http://127.0.0.1:<APP_PORT>/docs`

## OCR 功能说明

当前版本 **未启用 OCR**。

在 [`app/services/paddleocr_service.py`](./app/services/paddleocr_service.py) 中，PaddleOCR 相关初始化与导入已注释（例如 `from paddleocr import PaddleOCR` 与 `self.ocr = PaddleOCR(...)`），因此 OCR 能力目前不对外提供。

如果后续需要恢复 OCR，需要先恢复该文件中的 PaddleOCR 相关代码，并补齐对应依赖与运行环境。
