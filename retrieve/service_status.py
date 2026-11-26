"""Gatus endpoint-status helpers for Remembrancer's live checker integration."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import urllib.error
import urllib.request

SERVICE_STATUS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
SERVICE_STATUS_CACHE_TTL = 30.0


class ServiceStatusError(RuntimeError):
    """Raised when the checker read cannot be completed."""


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_endpoints(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("endpoints", [])
    if not isinstance(payload, list):
        raise ServiceStatusError("unexpected checker payload: expected endpoint list")
    out: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict):
            out.append(row)
    return out


def fetch_statuses(api_url: str, timeout: int = 6) -> list[dict[str, Any]]:
    """Fetch checker endpoint data with in-process TTL cache and typed failures."""
    if not isinstance(api_url, str) or not api_url.strip():
        raise ServiceStatusError("invalid api_url")

    now = time.time()
    cached = SERVICE_STATUS_CACHE.get(api_url)
    if cached and now - cached[0] < SERVICE_STATUS_CACHE_TTL:
        return cached[1]

    req = urllib.request.Request(api_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if status >= 400:
                raise ServiceStatusError(f"checker returned status {status}")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServiceStatusError(f"checker returned status {exc.code}") from exc
    except Exception as exc:
        raise ServiceStatusError(f"checker request failed: {exc}") from exc

    endpoints = _extract_endpoints(payload)
    SERVICE_STATUS_CACHE[api_url] = (now, endpoints)
    return endpoints


def _latency_ms(result: dict[str, Any]) -> float | None:
    raw = result.get("duration")
    if raw is None:
        raw = result.get("response_time")
    if raw is None:
        return None
    try:
        return float(raw) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _result_success(result: dict[str, Any]) -> bool:
    return bool(result.get("success"))


def _checked_at(result: dict[str, Any]) -> str:
    raw = result.get("time") if result.get("time") is not None else result.get("timestamp")
    return str(raw) if raw else ""


def summarize(endpoints: list[dict], name: str | None = None,
              now: datetime | str | None = None, max_age_s: int = 300) -> list[dict[str, Any]]:
    """Map checker endpoint objects to compact row dictionaries.

    `up` is the last result's success, `latency_ms` is last result duration
    (ns→ms), `uptime_recent` is fraction of successful recent results,
    and `checked_at` is the last result timestamp (ISO string).
    """
    if now is None:
        cutoff = datetime.now(timezone.utc)
    elif isinstance(now, str):
        cutoff = _parse_iso(now) or datetime.now(timezone.utc)
    else:
        cutoff = now
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
    filtered: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        endpoint_name = (endpoint.get("name") or "")
        if not isinstance(endpoint_name, str) or endpoint_name == "":
            continue
        if name is not None and endpoint_name != name:
            continue
        results = [r for r in endpoint.get("results", []) if isinstance(r, dict)]
        if results:
            last = results[-1]
            checked = _checked_at(last)
            checked_dt = _parse_iso(checked)
            stale = checked_dt is not None and (cutoff - checked_dt).total_seconds() > float(max_age_s)
            up = _result_success(last)
            uptime = sum(1 for row in results if _result_success(row)) / float(len(results))
            latency = _latency_ms(last)
        else:
            checked = ""
            checked_dt = None
            stale = False
            up = False
            uptime = 0.0
            latency = None
        filtered.append({
            "name": endpoint_name,
            "group": endpoint.get("group", "") or "",
            "up": up,
            "latency_ms": latency,
            "uptime_recent": uptime,
            "url": endpoint.get("url", "") or "",
            "checked_at": checked,
            "stale": stale,
        })
    if name is None:
        filtered.sort(key=lambda row: row.get("name", ""))
    return filtered
