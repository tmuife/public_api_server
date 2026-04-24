# AGENT.md

## Purpose

`media_client` 是一个轻量级 Python 客户端脚手架，用于连接 `public_api_server` 系列服务，完成基础初始化与服务连通性检查。

## Current capabilities

- 通过 `init` 命令初始化本地 `.env`（从 `.env.example` 复制）
- 通过 `doctor` 命令检查目标服务健康状态
- 支持通过环境变量注入 API 地址、鉴权 token、超时时间

## How to run

```bash
cd media_client
.venv/bin/python main.py --help
.venv/bin/python main.py init
.venv/bin/python main.py doctor
```

## Core runtime parameters

- `MEDIA_API_BASE_URL`: 目标 API 根地址，默认 `http://127.0.0.1:8000`
- `MEDIA_API_HEALTH_PATH`: 健康检查路径，默认 `/health`
- `MEDIA_API_ACCESS_TOKEN`: 可选鉴权 token（通过 `access_token` 请求头发送）
- `MEDIA_API_TIMEOUT_SECONDS`: 请求超时秒数，默认 `10`

## Notes

- 当前实现只依赖 Python 标准库，便于快速启动与集成。
- 当后续需要业务接口调用时，可在当前 CLI 基础上继续扩展子命令。
