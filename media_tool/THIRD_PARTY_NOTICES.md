# 第三方推理代码与模型来源

本项目的服务核心抽取自同仓库的 `swap_face_server` 和 `enhance_face_server`，运行时不依赖它们。模型推理算法和网络定义按以下上游版本保留；代码许可证保存在 `src/media_tool/vendor/licenses/`，随 Python 包交付。

| 来源 | 使用内容 | 版本/许可 | 本地位置 |
|---|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface/tree/v0.7/python-package/insightface) | SCRFD 解码/NMS、ArcFace 模板/预处理、INSwapper 身份映射与贴回 | v0.7，MIT | `detectors/insightface.py`、`face_analyzer.py`、`alignment.py`、`face_swapper.py` |
| [GFPGAN](https://github.com/TencentARC/GFPGAN/tree/v1.3.8) | clean 推理网络、网络构造、输入输出转换 | v1.3.8，Apache-2.0 及上游所列第三方条款 | `vendor/gfpgan/`、`face_enhancer.py` |
| [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/v1.4.2) | `default_init_weights` 的最小实现 | v1.4.2，Apache-2.0 | `vendor/gfpgan/initialization.py` |
| [facexlib](https://github.com/xinntao/facexlib/tree/v0.3.0) | GFPGAN FFHQ 五点对齐模板 | v0.3.0，MIT | `alignment.py` |
| [scikit-image](https://github.com/scikit-image/scikit-image/tree/v0.25.2) | Umeyama 五点相似变换的数值运算顺序 | v0.25.2，BSD-3-Clause | `alignment.py` |

GFPGAN 网络只移除了训练注册器并替换必要的初始化导入；参数名称和网络结构保留，使用原始 `GFPGANv1.4.pth` 加载。没有引入 BasicSR 的训练、退化或数据加载模块。局部增强贴回使用本项目的羽化遮罩。

独立模型准备：YuNet、ArcFace、INSwapper 来自旧换脸项目的本地权重，GFPGAN 来自旧增强项目。补齐的 SCRFD 来自 [immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx)，属于 InsightFace SCRFD 模型。文件大小及 SHA-256 记录在 `models-manifest.json`。模型权重的使用条款遵循各模型发布者，与推理代码的许可证分别管理。

验证中使用的公开图片来自 [OpenCV 4.10.0 samples/data](https://github.com/opencv/opencv/tree/4.10.0/samples/data)，仅保存在临时测试目录，不随项目分发。默认测试和正常运行不下载图片或模型。
