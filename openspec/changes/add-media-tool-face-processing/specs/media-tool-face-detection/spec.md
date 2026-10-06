## ADDED Requirements

### Requirement: 通过配置选择两种检测器

系统 MUST 在首版同时支持 InsightFace/SCRFD 与 OpenCV YuNet，通过 `detect_method=insightface|yunet` 选择，默认 `insightface`。一次运行 MUST 仅初始化所选检测器；不支持的值 MUST 报错，不接受隐式别名或自动切换。

#### Scenario: 选择 InsightFace
- **WHEN** 配置 `detect_method=insightface`，且所需模型有效
- **THEN** 参考图、替换图、素材图片和视频帧均使用 SCRFD 检测，不创建 YuNet 检测器

#### Scenario: 选择 YuNet
- **WHEN** 配置 `detect_method=yunet`，且所需模型有效
- **THEN** 所有人脸检测使用 OpenCV YuNet，不创建 SCRFD 会话

#### Scenario: 检测方法无效
- **WHEN** 配置 `detect_method=unknown`
- **THEN** 预检查报告仅支持 `insightface` 与 `yunet` 并退出

### Requirement: 统一检测结果结构

两个检测器 MUST 输出相同的人脸结构：原图坐标下形状为 `(4,)` 的浮点边界框 `[x1,y1,x2,y2]`、形状为 `(5,2)` 的浮点关键点和检测置信度。关键点顺序 MUST 为图像中的左眼、右眼、鼻尖、左嘴角、右嘴角。系统 MUST 保留有效浮点精度，不先转整数，并使下游处理不依赖检测器特有格式。

#### Scenario: YuNet 结果转换
- **WHEN** YuNet 返回位置、宽高、五点与置信度组成的检测行
- **THEN** 适配器转换为统一的四元素边界框及五点顺序，识别与换脸可以直接使用

#### Scenario: SCRFD 结果转换
- **WHEN** SCRFD 输出经过解码和 NMS 的人脸
- **THEN** 适配器输出与 YuNet 相同结构和坐标语义

### Requirement: 检测缩放与原图坐标一致

系统 MUST 根据 `detection_max_side` 构造合法检测输入，并将缩放、填充后的边界框与关键点正确变换回原图坐标。检测缩小 MUST 不降低用于换脸和融合的原始帧分辨率。

#### Scenario: 大分辨率图片检测
- **WHEN** 对长边 3840 的图片使用长边 640 的检测配置
- **THEN** 返回的框与关键点定位到 3840 长边的原图，后续在原图执行换脸和融合

#### Scenario: 非方形图片的 SCRFD 填充
- **WHEN** 横向图片被等比缩放并填充为 SCRFD 输入
- **THEN** 输出坐标消除填充偏移并恢复缩放，不产生横纵拉伸

### Requirement: 置信度过滤与空检测语义

两个检测器 MUST 使用 `detection_score_threshold` 过滤低置信人脸并进行非极大值抑制。无有效人脸 MUST 返回空列表；模型推理异常 MUST 报错，不能伪装为空检测。

#### Scenario: 未检测到人脸
- **WHEN** 正常检测完成但没有超过置信度阈值的人脸
- **THEN** 返回空列表，下游不执行换脸或增强

#### Scenario: 检测会话执行失败
- **WHEN** 检测器因模型或运行时异常无法执行
- **THEN** 错误传播到当前文件或启动检查，不将素材标记为未匹配

### Requirement: 两种检测器共用识别模型

两种检测器 MUST 使用同一 `recognition/w600k_r50.onnx`、相同的识别预处理和 L2 归一化算法。每次运行 MUST 只创建一份识别模型，所选检测器 MUST 一致用于参考、替换和素材。运行报告 MUST 标注检测方法；切换方法 MUST 重新提取参考特征。

#### Scenario: 修改检测方法后重新运行
- **WHEN** 用户从 InsightFace 切换到 YuNet 后重新启动工具
- **THEN** 参考与替换图通过 YuNet 重新检测和提取特征，并沿用同一识别模型文件及距离定义
