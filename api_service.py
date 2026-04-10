"""FastAPI-facing service layer for real-time IoT model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import threading
import time
import urllib.request
import urllib.parse
import json
from typing import Any, NamedTuple

import pandas as pd

from .data_fetcher import DataFetchError, DataFetcher, DataFetcherConfig
from .data_quality_gate import DataQualityError, DataQualityGate
from .predictor import DeepTimeSeriesPredictor, PredictorConfig, PredictorError


class ApiServiceError(RuntimeError):
    """Raised when API-level inference operations fail."""


# ---------------------------------------------------------------------------
# Sensor threshold definitions
# ---------------------------------------------------------------------------

class _Threshold(NamedTuple):
    lo: float | None
    hi: float | None
    lo_reason: str = ""
    hi_reason: str = ""


_SENSOR_THRESHOLDS: dict[str, _Threshold] = {
    "_Temp":  _Threshold(lo=18.0, hi=33.0,   lo_reason="< 18.0 C",      hi_reason="> 33.0 C"),
    "_Humid": _Threshold(lo=30.0, hi=75.0,   lo_reason="< 30 %",        hi_reason="> 75 %"),
    "_CO2":   _Threshold(lo=None, hi=1200.0, lo_reason="",              hi_reason="> 1200 ppm"),
    "_TVOC":  _Threshold(lo=None, hi=300.0,  lo_reason="",              hi_reason="> 300 ppb"),
}

_DAY_ROWS:  int = 288
_HOUR_ROWS: int = 12
_WEEK_ROWS: int = 2016

_ALERT_COOLDOWN_SECONDS: float = 3600.0

_ICT = timezone(timedelta(hours=7))

NODE_ORDER = ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _now_ict() -> datetime:
    return datetime.now(_ICT)


def _fmt_ts(dt: datetime) -> str:
    """e.g. 13:00 ICT — 10/04/2026"""
    return dt.strftime("%H:%M ICT \u2014 %d/%m/%Y")


def _fmt_val(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def _build_node_table(latest: pd.Series, df_columns: list[str]) -> str:
    """
    Builds a fixed-width per-node table string.

    Node   Temp    Hum    CO2    TVOC
    M1     29.2   70.6    400     0.0
    M4     --      --      --      --
    """
    header = f"{'Node':<6} {'Temp':>6} {'Hum':>6} {'CO2':>6} {'TVOC':>6}"
    divider = "-" * len(header)
    rows = [header, divider]

    for node in NODE_ORDER:
        def get(suffix: str) -> float | None:
            col = f"{node}{suffix}"
            if col not in df_columns:
                return None
            val = latest.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        temp = get("_Temp")
        hum  = get("_Humid")
        co2  = get("_CO2")
        tvoc = get("_TVOC")

        # Node considered offline if all values are None
        if all(v is None for v in (temp, hum, co2, tvoc)):
            rows.append(f"{node:<6} {'--':>6} {'--':>6} {'--':>6} {'--':>6}")
        else:
            rows.append(
                f"{node:<6} {_fmt_val(temp):>6} {_fmt_val(hum):>6}"
                f" {_fmt_val(co2, 0):>6} {_fmt_val(tvoc, 1):>6}"
            )

    return "\n".join(rows)


def _build_alert_lines(breaches: list[tuple[str, float, str]]) -> list[str]:
    lines = []
    for col, val, reason in breaches:
        # col is like "M8_CO2" — split into node + metric
        parts = col.split("_", 1)
        node   = parts[0] if len(parts) == 2 else col
        metric = parts[1] if len(parts) == 2 else ""
        lines.append(f"  {node:<5} {metric:<6} {val:.0f}  ({reason})")
    return lines


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApiServiceConfig:
    csv_url: str
    checkpoint_path: Path
    train_tail_rows: int = 720


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class InferenceAPIService:
    """Provides thread-safe inference and retraining methods for REST endpoints."""

    def __init__(self, config: ApiServiceConfig) -> None:
        self._config = config
        self._fetcher = DataFetcher(DataFetcherConfig(csv_url=config.csv_url))
        self._quality_gate = DataQualityGate(zscore_threshold=3.0)
        self._predictor = DeepTimeSeriesPredictor(config=PredictorConfig())

        self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        self._alert_cooldowns: dict[str, float] = {}
        # Track which UTC hour we last sent a scheduled report (avoids double-send)
        self._last_report_hour: int = -1

        self._checkpoint_checked = False
        self._last_error: str | None = None
        self._last_prediction_utc: str | None = None
        self._lock = threading.RLock()

        self._cached_clean_df: pd.DataFrame | None = None
        self._stop_event = threading.Event()

        self._credentials_ready = threading.Event()
        if self._bot_token and self._chat_id:
            self._credentials_ready.set()

        self._bg_thread = threading.Thread(
            target=self._background_fetch_loop, daemon=True, name="bg-fetch"
        )
        self._bg_thread.start()

        self._tg_thread = threading.Thread(
            target=self._telegram_polling_loop, daemon=True, name="tg-poll"
        )
        self._tg_thread.start()

    # ------------------------------------------------------------------
    # Background fetch loop
    # ------------------------------------------------------------------

    def _background_fetch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw_df = self._fetcher.fetch()
                clean_df, _ = self._quality_gate.process(raw_df)

                with self._lock:
                    try:
                        self._ensure_model_ready(clean_df, force_retrain=False)
                    except Exception as exc:
                        self._last_error = f"Background Model Prep Error: {exc}"

                    self._cached_clean_df = clean_df
                    if not self._last_error or "Prep Error" not in self._last_error:
                        self._last_error = None

                self._check_and_send_alerts(clean_df)
                self._check_and_send_hourly_report(clean_df)

            except Exception as exc:
                with self._lock:
                    self._last_error = f"Background Fetch Error: {exc}"

            self._stop_event.wait(timeout=30.0)

    # ------------------------------------------------------------------
    # Telegram send
    # ------------------------------------------------------------------

    def _send_telegram_message(self, text: str, target_chat_id: str | None = None) -> None:
        target = target_chat_id or self._chat_id
        if not self._bot_token or not target:
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = urllib.parse.urlencode(
                {"chat_id": target, "text": text, "parse_mode": "HTML"}
            ).encode("utf-8")
            req = urllib.request.Request(url, data=payload)
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as exc:
            print(f"Telegram send error: {exc}")

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_sensor_alerts(latest: pd.Series) -> list[tuple[str, float, str]]:
        breaches: list[tuple[str, float, str]] = []
        for col in latest.index:
            col_str = str(col)
            if "_" not in col_str:
                continue
            suffix = next((s for s in _SENSOR_THRESHOLDS if col_str.endswith(s)), None)
            if suffix is None:
                continue
            thresh = _SENSOR_THRESHOLDS[suffix]
            val = float(latest[col])
            if thresh.hi is not None and val > thresh.hi:
                breaches.append((col_str, val, thresh.hi_reason))
            elif thresh.lo is not None and val < thresh.lo:
                breaches.append((col_str, val, thresh.lo_reason))
        return breaches

    def _check_and_send_alerts(self, clean_df: pd.DataFrame) -> None:
        if clean_df.empty or not self._bot_token:
            return

        latest = clean_df.iloc[-1]
        now    = time.monotonic()
        breaches = self._evaluate_sensor_alerts(latest)

        alert_lines: list[str] = []
        with self._lock:
            for col, val, reason in breaches:
                last_sent = self._alert_cooldowns.get(col, 0.0)
                if now - last_sent > _ALERT_COOLDOWN_SECONDS:
                    self._alert_cooldowns[col] = now
                    parts  = col.split("_", 1)
                    node   = parts[0] if len(parts) == 2 else col
                    metric = parts[1] if len(parts) == 2 else ""
                    alert_lines.append(f"  {node:<5} {metric:<6} {val:.0f}  ({reason})")

        if alert_lines:
            dt  = _now_ict()
            msg = (
                f"<b>[ALERT] UEH Campus V \u2014 {_fmt_ts(dt)}</b>\n"
                "<pre>"
                + "\n".join(alert_lines)
                + "</pre>"
            )
            self._send_telegram_message(msg)

    # ------------------------------------------------------------------
    # Hourly scheduled report
    # ------------------------------------------------------------------

    def _check_and_send_hourly_report(self, clean_df: pd.DataFrame) -> None:
        if clean_df.empty or not self._bot_token:
            return

        dt = _now_ict()
        current_hour = dt.hour

        with self._lock:
            # Trigger once per hour when minute < 2 (30s fetch loop may land anywhere in min 0-1)
            if dt.minute >= 2 or current_hour == self._last_report_hour:
                return
            self._last_report_hour = current_hour

        self._send_status_report(clean_df, target_chat_id=None)

    def _send_status_report(self, clean_df: pd.DataFrame, target_chat_id: str | None) -> None:
        latest   = clean_df.iloc[-1]
        breaches = self._evaluate_sensor_alerts(latest)
        dt       = _now_ict()

        status_line = (
            f"Anomalies: <b>{len(breaches)}</b>"
            if breaches
            else "Status: <b>NORMAL</b>"
        )

        table = _build_node_table(latest, list(clean_df.columns))

        msg = (
            f"<b>[HOURLY REPORT] {_fmt_ts(dt)}</b>\n"
            f"{status_line}\n\n"
            "<pre>"
            f"{table}"
            "</pre>"
        )
        self._send_telegram_message(msg, target_chat_id)

    # ------------------------------------------------------------------
    # Telegram polling + command dispatch
    # ------------------------------------------------------------------

    def _telegram_polling_loop(self) -> None:
        last_update_id: int = 0
        while not self._stop_event.is_set():
            self._credentials_ready.wait(timeout=60.0)
            if not self._bot_token or not self._chat_id:
                continue
            try:
                url = (
                    f"https://api.telegram.org/bot{self._bot_token}"
                    f"/getUpdates?offset={last_update_id}&timeout=30"
                )
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=35) as resp:
                    payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

                if payload.get("ok"):
                    for update in payload["result"]:
                        last_update_id = update["update_id"] + 1

                        msg_node = update.get("message") or update.get("channel_post")
                        if not (msg_node and "text" in msg_node):
                            continue

                        text    = msg_node["text"].lower()
                        chat_id = str(msg_node["chat"]["id"])

                        if text.startswith("/getcurrent_detail"):
                            self._handle_telegram_command("detail", chat_id)
                        elif text.startswith("/getcurrent_short"):
                            self._handle_telegram_command("short", chat_id)
                        elif text.startswith("/getcurrent_alert"):
                            self._handle_telegram_command("alert", chat_id)

            except Exception as exc:
                print(f"Telegram polling error: {exc}")
            self._stop_event.wait(timeout=5.0)

    def _handle_telegram_command(self, cmd_type: str, target_chat_id: str) -> None:
        with self._lock:
            df = self._cached_clean_df

        if df is None or df.empty:
            self._send_telegram_message(
                "System warming up. Please try again in a moment.",
                target_chat_id,
            )
            return

        latest   = df.iloc[-1]
        dt       = _now_ict()
        breaches = self._evaluate_sensor_alerts(latest)

        if cmd_type == "detail":
            table = _build_node_table(latest, list(df.columns))
            msg = (
                f"<b>[DETAIL] {_fmt_ts(dt)}</b>\n\n"
                "<pre>"
                f"{table}"
                "</pre>"
            )
            self._send_telegram_message(msg, target_chat_id)

        elif cmd_type == "short":
            temps = [float(latest[c]) for c in df.columns if str(c).endswith("_Temp") and not pd.isna(latest[c])]
            hums  = [float(latest[c]) for c in df.columns if str(c).endswith("_Humid") and not pd.isna(latest[c])]
            avg_t = sum(temps) / len(temps) if temps else None
            avg_h = sum(hums)  / len(hums)  if hums  else None
            msg = (
                f"<b>[SUMMARY] {_fmt_ts(dt)}</b>\n\n"
                f"Avg Temp:     <b>{_fmt_val(avg_t)} C</b>\n"
                f"Avg Humidity: <b>{_fmt_val(avg_h)} %</b>\n"
                f"Anomalies:    <b>{len(breaches)}</b>"
            )
            if breaches:
                lines = _build_alert_lines(breaches)
                msg += "\n<pre>" + "\n".join(lines) + "</pre>"
            self._send_telegram_message(msg, target_chat_id)

        elif cmd_type == "alert":
            if not breaches:
                msg = f"<b>[ALERT] {_fmt_ts(dt)}</b>\n\nNo active anomalies."
            else:
                lines = _build_alert_lines(breaches)
                msg = (
                    f"<b>[ALERT] {_fmt_ts(dt)}</b>\n"
                    "<pre>"
                    + "\n".join(lines)
                    + "</pre>"
                )
            self._send_telegram_message(msg, target_chat_id)

    # ------------------------------------------------------------------
    # REST API methods (unchanged)
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "model_fitted": self._predictor.is_fitted,
                "checkpoint_path": self._config.checkpoint_path.as_posix(),
                "checkpoint_checked": self._checkpoint_checked,
                "last_prediction_utc": self._last_prediction_utc,
                "last_error": self._last_error,
            }

    def get_latest_raw_data(self) -> dict[str, Any]:
        with self._lock:
            try:
                clean_df = self._load_latest_clean_data()
            except (DataFetchError, DataQualityError) as exc:
                self._last_error = str(exc)
                raise ApiServiceError(str(exc)) from exc

        if clean_df.empty:
            raise ApiServiceError("Clean data is empty.")

        latest_row = clean_df.iloc[-1]
        response = latest_row.to_dict()
        response["ThoiGian"] = pd.Timestamp(clean_df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

        remapped = {}
        for k, v in response.items():
            if isinstance(v, (int, float)):
                v = f"{float(v):.2f}"
            if k.endswith("_Humid"):
                remapped[k.replace("_Humid", "_Hum")] = v
            else:
                remapped[k] = v
        return remapped

    def get_latest_day_raw_data(self) -> dict[str, Any]:
        return self.get_historical_data_legacy(_DAY_ROWS)

    def get_latest_hour_raw_data(self) -> dict[str, Any]:
        return self.get_historical_data_legacy(_HOUR_ROWS)

    def get_latest_7_days_raw_data(self) -> dict[str, Any]:
        return self.get_historical_data_legacy(_WEEK_ROWS)

    def get_historical_data_legacy(self, row_count: int) -> dict[str, Any]:
        _ROW_TO_DELTA = {
            _HOUR_ROWS: pd.Timedelta(hours=1),
            _DAY_ROWS:  pd.Timedelta(hours=24),
            _WEEK_ROWS: pd.Timedelta(days=7),
        }
        delta = _ROW_TO_DELTA.get(row_count)

        with self._lock:
            try:
                clean_df = self._load_latest_clean_data()
            except (DataFetchError, DataQualityError) as exc:
                self._last_error = str(exc)
                raise ApiServiceError(str(exc)) from exc

        if clean_df.empty:
            raise ApiServiceError("Clean data is empty.")

        if delta is not None and isinstance(clean_df.index, pd.DatetimeIndex):
            cutoff  = clean_df.index[-1] - delta
            tail_df = clean_df[clean_df.index > cutoff]
            if len(tail_df) > row_count:
                tail_df = tail_df.tail(row_count)
        else:
            tail_df = clean_df.tail(row_count)

        data_list = []
        for timestamp, row in tail_df.iterrows():
            row_dict = row.to_dict()
            mapped   = {}
            for k, v in row_dict.items():
                if isinstance(v, (int, float)):
                    v = f"{float(v):.2f}"
                if k.endswith("_Humid"):
                    mapped[k.replace("_Humid", "_Hum")] = v
                else:
                    mapped[k] = v
            mapped["ThoiGian"] = pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            data_list.append(mapped)

        return {"data": data_list, "count": len(data_list)}
