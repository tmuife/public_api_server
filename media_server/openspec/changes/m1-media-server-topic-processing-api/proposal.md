## Why

`media_client` 已经具备视频预处理、帧加密、写入 Kafka、以及从结果 topic 合成视频的基础能力，但与之配套的 `media_server` 仍是空项目，缺少一个可直接接入当前链路的处理端。为了形成最小业务闭环，需要在 `media_server` 中定义并实现一个标准 FastAPI 服务：接收客户端传入的输入 topic，消费加密帧，完成服务端处理，再把结果写回约定的输出 topic。

当前约束已经明确：

- 只实现一个业务接口，不扩展额外管理面接口。
- 服务端必须与 `media_client` 当前的 Kafka topic、AES-256-GCM、消息字段规则完全兼容。
- 处理逻辑本期先留空，使用占位处理器打通链路即可。
- 当前目录是多子项目仓库，本次改动只允许落在 `media_server`。

## What Changes

- 在 `media_server` 中建立 FastAPI 服务基线：
  - `main.py` 作为唯一服务入口。
  - 业务代码放在 `app/` 下，按 `routers/services/utils` 做最小分层。
  - 成功响应统一为 `{ code: 0, message: "success", data: ... }`。
  - 业务接口默认 Bearer 鉴权，`/health`、`/docs`、`/openapi.json` 公开。
- 新增单一业务接口：
  - `POST /videos/process-topic`
  - 请求体只包含 `topic`
  - `topic` 必须是 `media_client` 预处理阶段生成的输入 topic，格式为 `{job_name}_input`
  - 接口采用异步触发模型：请求负责启动处理任务并快速返回，实际帧消费与处理在后台执行
- 标准化服务端 topic 处理链路：
  - 校验输入 topic 命名与类型
  - 从 topic 推导 `job_name`
  - 推导输出 topic 为 `{job_name}_output`
  - 在后台任务中消费输入 topic 中的加密帧
  - 使用与客户端一致的 `AES-256-GCM` 解密每帧
  - 调用占位处理逻辑（当前先 passthrough）
  - 将处理后的帧重新加密并发布到输出 topic
- 固化消息契约兼容性：
  - 输入消息必须符合 frame request contract：
    - `job_name`
    - `frame_index`
    - `nonce_b64`
    - `ciphertext_b64`
    - `tag_b64`
    - `content_type`
  - 输出消息必须符合 frame result contract：
    - `job_name`
    - `frame_index`
    - `status`
    - `nonce_b64`
    - `ciphertext_b64`
    - `tag_b64`
    - 当 `status=error` 时必须附带非空 `error_code`
- 定义无状态处理模型：
  - `media_server` 不维护作业 workspace，不持有 manifest，不参与视频合成
  - 服务端只负责 topic 级消费、处理、回写
- 定义异步执行与消费归属模型：
  - `POST /videos/process-topic` 只负责异步提交处理任务
  - 同一输入 topic 的消费归属通过固定 Kafka `group.id` 控制，不引入外部锁或调度层
  - 进程内若保留任务注册表，也只作为重复触发去重优化，不作为一致性保证
- 定义“消费完成”的收敛方式：
  - 由于当前输入消息中没有 `frame_count` 和 EOF 标志，服务端采用“首帧等待 + 空闲超时 + 总处理时长上限”来判定一次 topic 处理何时完成
  - 本期不扩展输入协议，不额外增加 EOF 消息或总帧数字段
- 补齐文档与测试：
  - `README.md` 补运行方式、环境变量、接口调用示例
  - 增加配置、topic/job 规则、加解密契约、API 行为的基础测试

## Capabilities

### New Capabilities

- `media-server-fastapi-foundation`: 定义 `media_server` 的 FastAPI 启动方式、鉴权、统一响应与异常处理。
- `media-server-topic-process-api`: 定义 `POST /videos/process-topic` 接口的请求、异步受理响应和错误语义。
- `media-server-kafka-frame-worker`: 定义从 `{job_name}_input` 消费帧、处理后写入 `{job_name}_output` 的服务端职责边界。
- `media-server-async-processing-dispatch`: 定义异步任务提交、后台执行与进程内重复提交去重策略。
- `media-server-processing-window-control`: 定义没有总帧数/结束标志时的消费完成判定策略。
- `encrypted-frame-contract-foundation`: 在服务端侧补充与 `media_client` 对齐的消费校验、解密、重新加密与结果回写约束。

## Impact

- 受影响代码：
  - `media_server/main.py`
  - `media_server/app/config.py`
  - `media_server/app/errors.py`
  - `media_server/app/schemas.py`
  - `media_server/app/middleware/auth.py`
  - `media_server/app/routers/videos.py`
  - `media_server/app/services/topic_process_service.py`
  - `media_server/app/services/kafka_service.py`
  - `media_server/app/utils/contracts.py`
  - `media_server/app/utils/crypto.py`
  - `media_server/app/utils/job.py`
  - `media_server/app/utils/runtime.py`
  - `media_server/tests/*`
  - `media_server/README.md`
- 外部依赖与运行环境影响：
  - 继续使用 `confluent-kafka` 与 `cryptography`
  - Kafka 集群需允许输入/输出 topic 的消费与生产
  - 服务端需与 `media_client` 使用相同 AES key 配置
- 数据与安全影响：
  - Kafka 消息与日志中禁止出现 token/password/secret/key 等敏感字段
  - 处理失败时也必须回写符合 result contract 的加密消息，不能只发错误码
  - 同一输入流的单次消费语义主要依赖固定 Kafka `group.id` 与 offset commit，避免重复回写
