from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from client.config import AppSettings
from client.kafka_bus import KafkaDoctorClient, KafkaStageError


@dataclass
class StageResult:
    stage: str
    status: str
    error_code: str | None = None
    hint: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "status": self.status,
        }
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


def _http_get(url: str, timeout_seconds: float, token: str | None = None) -> tuple[int, str]:
    request = urllib.request.Request(url=url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body


def _fetch_openapi(settings: AppSettings) -> tuple[dict[str, Any], StageResult]:
    url = f"{settings.api.base_url}{settings.api.openapi_path}"
    try:
        status, body = _http_get(url, settings.api.timeout_seconds)
    except urllib.error.HTTPError as exc:
        return {}, StageResult(
            stage="api.openapi",
            status="fail",
            error_code="http_error",
            hint="Ensure /openapi.json is publicly accessible",
            details={"status_code": exc.code, "url": url},
        )
    except urllib.error.URLError as exc:
        return {}, StageResult(
            stage="api.openapi",
            status="fail",
            error_code="network_error",
            hint="Check MEDIA_API_BASE_URL and network reachability",
            details={"url": url, "reason": str(exc.reason)},
        )

    if status >= 400:
        return {}, StageResult(
            stage="api.openapi",
            status="fail",
            error_code="http_error",
            hint="Ensure /openapi.json is available",
            details={"status_code": status, "url": url},
        )

    try:
        document = json.loads(body)
    except ValueError:
        return {}, StageResult(
            stage="api.openapi",
            status="fail",
            error_code="invalid_openapi",
            hint="OpenAPI endpoint must return valid JSON",
            details={"url": url},
        )

    return document, StageResult(
        stage="api.openapi",
        status="pass",
        details={"status_code": status, "url": url},
    )


def _discover_auth_probe_path(openapi_doc: dict[str, Any]) -> str | None:
    paths = openapi_doc.get("paths", {})
    default_security = openapi_doc.get("security", [])

    for method in ("get", "head", "options", "post", "put", "patch", "delete"):
        for path, operations in paths.items():
            operation = operations.get(method)
            if not isinstance(operation, dict):
                continue

            operation_security = operation.get("security", default_security)
            if not operation_security:
                continue

            if "{" in path:
                continue

            parameters = operation.get("parameters", [])
            if any(parameter.get("required") for parameter in parameters):
                continue

            if method in {"post", "put", "patch"}:
                request_body = operation.get("requestBody") or {}
                if request_body.get("required"):
                    continue

            if path in {"/health", "/docs", "/openapi.json", "/docs/oauth2-redirect"}:
                continue

            return path

    return None


def _check_api_baseline(settings: AppSettings) -> tuple[list[StageResult], dict[str, Any]]:
    results: list[StageResult] = []

    health_url = f"{settings.api.base_url}{settings.api.health_path}"
    try:
        status, _body = _http_get(health_url, settings.api.timeout_seconds)
        if status >= 400:
            results.append(
                StageResult(
                    stage="api.health_public",
                    status="fail",
                    error_code="health_not_public",
                    hint="/health must be available without authentication",
                    details={"status_code": status, "url": health_url},
                )
            )
        else:
            results.append(
                StageResult(
                    stage="api.health_public",
                    status="pass",
                    details={"status_code": status, "url": health_url},
                )
            )
    except urllib.error.HTTPError as exc:
        results.append(
            StageResult(
                stage="api.health_public",
                status="fail",
                error_code="health_not_public",
                hint="/health must be available without authentication",
                details={"status_code": exc.code, "url": health_url},
            )
        )
    except urllib.error.URLError as exc:
        results.append(
            StageResult(
                stage="api.health_public",
                status="fail",
                error_code="network_error",
                hint="Check API host reachability",
                details={"url": health_url, "reason": str(exc.reason)},
            )
        )

    for stage_name, route_path in (
        ("api.docs_public", settings.api.docs_path),
        ("api.docs_oauth2_redirect_public", settings.api.docs_oauth2_redirect_path),
    ):
        url = f"{settings.api.base_url}{route_path}"
        try:
            status, _body = _http_get(url, settings.api.timeout_seconds)
            if status >= 400:
                results.append(
                    StageResult(
                        stage=stage_name,
                        status="fail",
                        error_code="docs_not_public",
                        hint="Docs endpoints should be accessible without authentication",
                        details={"status_code": status, "url": url},
                    )
                )
            else:
                results.append(
                    StageResult(
                        stage=stage_name,
                        status="pass",
                        details={"status_code": status, "url": url},
                    )
                )
        except urllib.error.HTTPError as exc:
            results.append(
                StageResult(
                    stage=stage_name,
                    status="fail",
                    error_code="docs_not_public",
                    hint="Docs endpoints should be accessible without authentication",
                    details={"status_code": exc.code, "url": url},
                )
            )
        except urllib.error.URLError as exc:
            results.append(
                StageResult(
                    stage=stage_name,
                    status="fail",
                    error_code="network_error",
                    hint="Check API host reachability",
                    details={"url": url, "reason": str(exc.reason)},
                )
            )

    openapi_doc, openapi_result = _fetch_openapi(settings)
    results.append(openapi_result)

    security_schemes = (
        openapi_doc.get("components", {})
        .get("securitySchemes", {})
    )
    bearer_enabled = any(
        scheme.get("type") == "http" and str(scheme.get("scheme", "")).lower() == "bearer"
        for scheme in security_schemes.values()
        if isinstance(scheme, dict)
    )

    if bearer_enabled:
        results.append(
            StageResult(
                stage="api.docs_bearer_scheme",
                status="pass",
                details={"security_schemes": sorted(security_schemes.keys())},
            )
        )
    else:
        results.append(
            StageResult(
                stage="api.docs_bearer_scheme",
                status="fail",
                error_code="missing_bearer_scheme",
                hint="OpenAPI must define HTTP Bearer scheme for docs login",
            )
        )

    probe_path = settings.api.auth_probe_path or _discover_auth_probe_path(openapi_doc)
    if not probe_path:
        results.append(
            StageResult(
                stage="api.auth_enforcement",
                status="fail",
                error_code="auth_probe_unavailable",
                hint="Set MEDIA_API_AUTH_PROBE_PATH to a protected endpoint",
            )
        )
        return results, {}

    protected_url = f"{settings.api.base_url}{probe_path}"
    try:
        _status_unauth, _ = _http_get(protected_url, settings.api.timeout_seconds)
        results.append(
            StageResult(
                stage="api.auth_enforcement",
                status="fail",
                error_code="auth_not_enforced",
                hint="Protected endpoint should return 401 without Bearer token",
                details={"url": protected_url, "status_code": _status_unauth},
            )
        )
        return results, {"auth_probe_path": probe_path}
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            results.append(
                StageResult(
                    stage="api.auth_enforcement",
                    status="fail",
                    error_code="auth_not_enforced",
                    hint="Protected endpoint should return 401 without Bearer token",
                    details={"url": protected_url, "status_code": exc.code},
                )
            )
            return results, {"auth_probe_path": probe_path}
    except urllib.error.URLError as exc:
        results.append(
            StageResult(
                stage="api.auth_enforcement",
                status="fail",
                error_code="network_error",
                hint="Check API host reachability",
                details={"url": protected_url, "reason": str(exc.reason)},
            )
        )
        return results, {"auth_probe_path": probe_path}

    try:
        status_auth, _ = _http_get(protected_url, settings.api.timeout_seconds, token=settings.api.api_key)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            results.append(
                StageResult(
                    stage="api.auth_enforcement",
                    status="fail",
                    error_code="invalid_media_api_key",
                    hint="MEDIA_API_KEY does not authenticate protected endpoints",
                    details={"url": protected_url, "status_code": exc.code},
                )
            )
            return results, {"auth_probe_path": probe_path}

        # Token passed auth, but endpoint may still fail for business reasons.
        status_auth = exc.code

    results.append(
        StageResult(
            stage="api.auth_enforcement",
            status="pass",
            details={"url": protected_url, "status_code_with_token": status_auth},
        )
    )
    return results, {"auth_probe_path": probe_path}


def _check_kafka_baseline(
    settings: AppSettings,
    kafka_client: KafkaDoctorClient,
) -> list[StageResult]:
    results: list[StageResult] = []
    kafka_settings = settings.kafka
    if kafka_settings is None:
        return results

    try:
        metadata_details = kafka_client.probe_metadata(timeout_seconds=kafka_settings.doctor_timeout_seconds)
        results.append(
            StageResult(
                stage="kafka.metadata",
                status="pass",
                details=metadata_details,
            )
        )
    except KafkaStageError as exc:
        results.append(
            StageResult(
                stage="kafka.metadata",
                status="fail",
                error_code=exc.code,
                hint=exc.hint,
                details={"message": str(exc)},
            )
        )
        return results

    try:
        rw_details = kafka_client.probe_read_write(timeout_seconds=kafka_settings.doctor_timeout_seconds)
        results.append(
            StageResult(
                stage="kafka.read_write",
                status="pass",
                details=rw_details,
            )
        )
    except KafkaStageError as exc:
        results.append(
            StageResult(
                stage="kafka.read_write",
                status="fail",
                error_code=exc.code,
                hint=exc.hint,
                details={"message": str(exc)},
            )
        )

    return results


def run_doctor(
    settings: AppSettings,
    *,
    skip_kafka: bool = False,
    include_api_baseline: bool = True,
    kafka_client: KafkaDoctorClient | None = None,
) -> dict[str, Any]:
    stage_results: list[StageResult] = [
        StageResult(
            stage="config",
            status="pass",
            details={"settings": settings.redacted()},
        )
    ]

    api_extra: dict[str, Any] = {}
    if include_api_baseline:
        api_results, api_extra = _check_api_baseline(settings)
        stage_results.extend(api_results)

    if skip_kafka:
        stage_results.append(
            StageResult(
                stage="kafka",
                status="skip",
                details={"reason": "--skip-kafka"},
            )
        )
    else:
        client = kafka_client
        if client is None:
            if settings.kafka is None:
                stage_results.append(
                    StageResult(
                        stage="kafka.metadata",
                        status="fail",
                        error_code="config_error",
                        hint="Kafka settings are required unless --skip-kafka is set",
                    )
                )
            else:
                client = KafkaDoctorClient(settings.kafka)

        if client is not None:
            stage_results.extend(_check_kafka_baseline(settings, client))

    ok = all(result.status in {"pass", "skip"} for result in stage_results)

    return {
        "ok": ok,
        "summary": {
            "total_stages": len(stage_results),
            "failed_stages": [result.stage for result in stage_results if result.status == "fail"],
            "auth_probe_path": api_extra.get("auth_probe_path"),
        },
        "stages": [result.to_dict() for result in stage_results],
    }
