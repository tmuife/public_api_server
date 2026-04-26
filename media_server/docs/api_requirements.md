服务端对接标准（可直接给后端）

compose 请求里的 topic 要求
字符限制：^[a-zA-Z0-9._-]+$ app/utils/job.py:42
推荐命名：{job_name}_output（文档约定） README.md:150
job_name 应匹配预处理生成的任务名（推荐 ^job_[0-9]{13}_[a-z0-9]{8}$） app/utils/job.py:8
该 job_name 必须能在本地 workspace 找到，且 manifest 里的 job_name 一致，否则会 job_not_found / manifest_job_mismatch app/services/compose_service.py:39
服务端写入结果 topic 的消息格式（必须）
必填字段：job_name, frame_index, status, nonce_b64, ciphertext_b64, tag_b64 app/utils/contracts.py:14
frame_index：整数、从 0 开始（zero-based），且最终要覆盖 0..frame_count-1
status：只能是 "ok" 或 "error"
当 status="error" 时，还必须带非空 error_code app/utils/contracts.py:80
字段名不能包含这些敏感词：token/password/secret/master_key/wrapped_key/key/credential app/utils/contracts.py:23
标准消息示例
成功帧（status=ok）：

{
"job_name": "job_1776945123456_ab12cd34",
"frame_index": 0,
"status": "ok",
"nonce_b64": "base64-12bytes-nonce",
"ciphertext_b64": "base64-ciphertext",
"tag_b64": "base64-16bytes-tag"
}

失败帧（status=error）：

{
"job_name": "job_1776945123456_ab12cd34",
"frame_index": 0,
"status": "error",
"nonce_b64": "base64-12bytes-nonce",
"ciphertext_b64": "base64-ciphertext",
"tag_b64": "base64-16bytes-tag",
"error_code": "MODEL_TIMEOUT"
}

补充一个容易踩坑点：当前实现里即使 status=error，nonce_b64/ciphertext_b64/tag_b64 仍然是必填；如果后端只想回错误码不回密文，客
户端需要改校验逻辑。
