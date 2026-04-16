"""Open-Meteo weather client for campus weather context and outdoor sensor sanity checks.

Fetches current conditions and hourly forecast for a single campus coordinate.
All nodes share one coordinate because the campus fits within one Open-Meteo grid cell.

Thread-safe: uses a module-level lock so concurrent callers don't issue duplicate requests.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600  # 10 minutes; Open-Meteo updates ~every 15 min
_REQUEST_TIMEOUT = 8

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_ts: float = 0.0


@dataclass
class CurrentWeather:
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    weathercode: int
    description: str

    @property
    def display(self) -> str:
        return (
            f"{self.description}, {self.temperature_c:.1f}°C, "
            f"humidity {self.humidity_pct:.0f}%, wind {self.wind_speed_kmh:.1f} km/h"
        )


# WMO Weather interpretation codes (subset used for display)
_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def _fetch_from_api(lat: float, lon: float) -> dict[str, Any]:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        f"&forecast_days=1"
        f"&timezone=Asia%2FBangkok"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_weather(lat: float, lon: float) -> dict[str, Any] | None:
    """Return raw Open-Meteo response, using cache if fresh enough.

    Returns None if the API is unreachable and no cache is available.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    with _lock:
        if _cache is not None and (now - _cache_ts) < _CACHE_TTL_SECONDS:
            return _cache

    try:
        data = _fetch_from_api(lat, lon)
        with _lock:
            _cache = data
            _cache_ts = time.monotonic()
        return data
    except Exception as exc:
        logger.warning("Open-Meteo request failed: %s", exc)
        with _lock:
            if _cache is not None:
                logger.info("Returning stale weather cache (age %.0fs)", now - _cache_ts)
                return _cache
        return None


def get_current(lat: float, lon: float) -> CurrentWeather | None:
    """Return parsed current weather, or None if unavailable."""
    data = get_weather(lat, lon)
    if data is None:
        return None
    try:
        current = data["current"]
        code = int(current.get("weather_code", 0))
        return CurrentWeather(
            temperature_c=float(current["temperature_2m"]),
            humidity_pct=float(current["relative_humidity_2m"]),
            wind_speed_kmh=float(current["wind_speed_10m"]),
            weathercode=code,
            description=_WMO_DESCRIPTIONS.get(code, f"Code {code}"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse Open-Meteo current weather: %s", exc)
        return None


def get_hourly_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    """Return list of hourly forecast dicts for today (up to 24 entries).

    Each entry: {"time": str, "temperature_c": float, "humidity_pct": float,
                 "wind_speed_kmh": float, "description": str}

    Returns empty list if unavailable.
    """
    data = get_weather(lat, lon)
    if data is None:
        return []
    try:
        hourly = data["hourly"]
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        humids = hourly["relative_humidity_2m"]
        winds = hourly["wind_speed_10m"]
        codes = hourly["weather_code"]
        return [
            {
                "time": times[i],
                "temperature_c": float(temps[i]),
                "humidity_pct": float(humids[i]),
                "wind_speed_kmh": float(winds[i]),
                "description": _WMO_DESCRIPTIONS.get(int(codes[i]), f"Code {codes[i]}"),
            }
            for i in range(len(times))
        ]
    except (KeyError, TypeError, IndexError) as exc:
        logger.warning("Failed to parse Open-Meteo hourly forecast: %s", exc)
        return []
