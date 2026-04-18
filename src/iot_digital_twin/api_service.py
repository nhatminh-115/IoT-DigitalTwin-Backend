"""FastAPI-facing service layer for real-time IoT model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import urllib.parse
import json
from typing import Any, NamedTuple

import pandas as pd
import requests

from .data_fetcher import DataFetchError, DataFetcher, DataFetcherConfig
from .data_quality_gate import DataQualityError, DataQualityGate
from .predictor import DeepTimeSeriesPredictor, PredictorConfig, PredictorError
from .anomaly_detector import AnomalyDetector
from .llm_service import AIAssistant
from . import viz_engine
from .video_generator import generate_video, video_path, CSV_URL as _VIDEO_CSV_URL
from . import worker_state
from . import weather_client

logger = logging.getLogger(__name__)

_OUTDOOR_CONFIG_PATH = Path("config/outdoor_nodes.json")
_WEATHER_SANITY_COOLDOWN_SECONDS = 14400  # 4 hours between repeated sensor-drift alerts


def _load_outdoor_config() -> dict[str, Any]:
    """Load outdoor node config. Returns empty config if file is missing."""
    try:
        return json.loads(_OUTDOOR_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("outdoor_nodes.json not found — weather sanity check disabled")
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("outdoor_nodes.json parse error: %s — weather sanity check disabled", exc)
        return {}


class ApiServiceError(RuntimeError):
    """Raised when API-level inference operations fail."""


# ---------------------------------------------------------------------------
# Sensor threshold definitions — single source of truth used by both alert
# dispatch and Telegram command handlers.
# ---------------------------------------------------------------------------

class _Threshold(NamedTuple):
    lo: float | None  # inclusive lower bound; None = unbounded
    hi: float | None  # inclusive upper bound; None = unbounded
    lo_reason: str = ""
    hi_reason: str = ""


_SENSOR_THRESHOLDS: dict[str, _Threshold] = {
    "_Temp":  _Threshold(lo=18.0,   hi=33.0,   lo_reason="Threshold dropped < 18.0 °C",   hi_reason="Threshold exceeded > 33.0 °C"),
    "_Humid": _Threshold(lo=30.0,   hi=75.0,   lo_reason="Severely dry < 30 %",            hi_reason="Hyper-humid > 75 %"),
    "_CO2":   _Threshold(lo=None,   hi=1200.0, lo_reason="",                               hi_reason="Toxic levels > 1200 ppm"),
    "_TVOC":  _Threshold(lo=None,   hi=300.0,  lo_reason="",                               hi_reason="Hazardous TVOC detected"),
}

# Number of 3-minute readings per window (3-min cadence from HomeAssistant).
_DAY_ROWS: int = 480   # 24 h × 20 readings/h
_HOUR_ROWS: int = 20   # 60 min / 3 min
_WEEK_ROWS: int = 3360 # 7 × 480

# Cooldown between repeated alerts for the same sensor channel (seconds).
_SENSOR_ALERT_COOLDOWN_SECONDS: float = 3600.0
_AE_ALERT_COOLDOWN_SECONDS: float = 3600.0

_HYSTERESIS_MARGIN: dict[str, float] = {
    "_Temp":  0.5,
    "_Humid": 2.0,
    "_CO2":   50.0,
    "_TVOC":  20.0,
}
_PERSISTENT_REALERT_SECONDS: float = 43200.0   # 12 hours
_ALERT_BUFFER_WINDOW_SECONDS: float = 120.0    # 2 minutes

_ICT = timezone(timedelta(hours=7))

NODE_ORDER = ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]

NODE_NAMES: dict[str, str] = {
    "M1":  "Canteen Garden",
    "M4":  "Studio ISCM",
    "M6":  "ISCM Staircase",
    "M7":  "Sky Garden",
    "M8":  "ISCM Balcony",
    "M9":  "Hotel Kitchen",
    "M10": "Hotel Corridor",
    "M11": "Hotel Balcony",
}


def _now_ict() -> datetime:
    return datetime.now(_ICT)


def _fmt_ts(dt: datetime) -> str:
    """e.g. 13:00 ICT — 10/04/2026"""
    return dt.strftime("%H:%M ICT \u2014 %d/%m/%Y")


def _fmt_val(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def _build_node_table(latest: pd.Series, df_columns: list[str], show_names: bool = False) -> str:
    if show_names:
        header  = f"{'Node':<6} {'Location':<16} {'Temp':>6} {'Hum':>6} {'CO2':>6} {'TVOC':>6}"
    else:
        header  = f"{'Node':<6} {'Temp':>6} {'Hum':>6} {'CO2':>6} {'TVOC':>6}"
    divider = "-" * len(header)
    rows    = [header, divider]

    for node in NODE_ORDER:
        def get(suffix: str, _node: str = node) -> float | None:
            col = f"{_node}{suffix}"
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

        if show_names:
            name = NODE_NAMES.get(node, "")
            if all(v is None for v in (temp, hum, co2, tvoc)):
                rows.append(f"{node:<6} {name:<16} {'--':>6} {'--':>6} {'--':>6} {'--':>6}")
            else:
                rows.append(
                    f"{node:<6} {name:<16} {_fmt_val(temp):>6} {_fmt_val(hum):>6}"
                    f" {_fmt_val(co2, 0):>6} {_fmt_val(tvoc, 1):>6}"
                )
        else:
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
        parts  = col.split("_", 1)
        node   = parts[0] if len(parts) == 2 else col
        metric = parts[1] if len(parts) == 2 else ""
        lines.append(f"  {node:<5} {metric:<6} {val:.0f}  ({reason})")
    return lines


def _breach_severity(col: str, value: float) -> float:
    """Returns threshold-distance severity used for deduplicating repeated breaches."""
    suffix = next((s for s in _SENSOR_THRESHOLDS if col.endswith(s)), None)
    if suffix is None:
        return 0.0
    threshold = _SENSOR_THRESHOLDS[suffix]
    if threshold.hi is not None and value > threshold.hi:
        return float(value - threshold.hi)
    if threshold.lo is not None and value < threshold.lo:
        return float(threshold.lo - value)
    return 0.0


def _dedupe_breaches(
    breaches: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """Collapses duplicate channel alerts and keeps the most severe value per channel."""
    deduped: dict[str, tuple[str, float, str]] = {}
    for col, val, reason in breaches:
        previous = deduped.get(col)
        if previous is None:
            deduped[col] = (col, val, reason)
            continue
        _, prev_val, _ = previous
        if _breach_severity(col, val) >= _breach_severity(col, prev_val):
            deduped[col] = (col, val, reason)
    return list(deduped.values())


@dataclass(frozen=True)
class ApiServiceConfig:
    """Configuration for API inference workflow.

    Attributes:
        csv_url: Public CSV endpoint of the data source.
        checkpoint_path: Local path of predictor checkpoint file.
        train_tail_rows: Number of most recent rows used for training.
    """

    csv_url: str
    checkpoint_path: Path
    train_tail_rows: int = 1200  # ~60 h at 3-min cadence


class InferenceAPIService:
    """Provides thread-safe inference and retraining methods for REST endpoints."""

    def __init__(self, config: ApiServiceConfig) -> None:
        logger.info("InferenceAPIService init with AE support")
        """Initializes service dependencies and state."""
        self._config = config
        self._fetcher = DataFetcher(DataFetcherConfig(csv_url=config.csv_url))
        self._quality_gate = DataQualityGate(zscore_threshold=3.0)
        self._predictor = DeepTimeSeriesPredictor(config=PredictorConfig())

        # Autoencoder anomaly detector (soft layer, complements rule-based)
        try:
            self._anomaly_detector: AnomalyDetector | None = AnomalyDetector(
                checkpoint_path=config.checkpoint_path.parent / "autoencoder_checkpoint.pt",
                meta_path=config.checkpoint_path.parent / "autoencoder_meta.json",
            )
        except Exception as exc:
            logger.warning("AnomalyDetector init failed (non-fatal): %s", exc)
            self._anomaly_detector = None
        self._ae_cooldown_until: float = 0.0   # suppress repeat AE alerts for 1h

        # Telegram Bot configuration
        self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        # Daily video generation — tracks last generated date to avoid duplicates.
        self._last_video_date: date | None = None

        # Mutable alert state accessed only from _bg_thread — no lock needed.
        self._active_breaches: set[str] = set()
        self._alert_buffer: list[tuple[str, float, str]] = []  # (col, val, reason)
        self._buffer_flush_at: float = 0.0
        self._sensor_last_alert_sent_at: dict[str, float] = {}
        self._persistent_alert_sent_at: dict[str, float] = {}
        self._weather_sanity_alerted_at: dict[str, float] = {}  # node_metric → last alert ts

        # Outdoor node config for weather sanity check (loaded once at init).
        self._outdoor_config: dict[str, Any] = _load_outdoor_config()

        # Protected by _lock.
        self._last_report_hour: int = -1

        self._checkpoint_checked = False
        self._last_error: str | None = None
        self._last_prediction_utc: str | None = None
        self._last_fetch_utc: str | None = None
        self._lock = threading.RLock()

        self._cached_clean_df: pd.DataFrame | None = None

        # Supabase logging client — non-fatal; None disables all logging.
        self._supabase = None
        try:
            supabase_url = os.environ.get("SUPABASE_URL", "").strip()
            supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
            if supabase_url and supabase_key:
                from supabase import create_client
                self._supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase logging client initialized.")
            else:
                logger.warning("Supabase logging disabled: SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        except Exception as exc:
            logger.warning("Supabase logging init failed (non-fatal): %s", exc)

        # Signals background threads to stop gracefully on shutdown.
        self._stop_event = threading.Event()

        # Becomes set once valid credentials are detected; avoids busy-wait in polling loop.
        self._credentials_ready = threading.Event()
        if self._bot_token and self._chat_id:
            self._credentials_ready.set()

        # AI assistant — non-fatal: bot works even if Groq is unavailable.
        self._ai_assistant: AIAssistant | None = None
        try:
            groq_key = os.environ.get("GROQ_API_KEY", "").strip()
            if groq_key:
                self._ai_assistant = AIAssistant(
                    groq_api_key=groq_key,
                    get_df=lambda: self._cached_clean_df,
                )
                logger.info("AIAssistant initialized successfully.")
            else:
                logger.warning("AIAssistant skipped: GROQ_API_KEY not set.")
        except Exception as exc:
            logger.warning("AIAssistant init failed (non-fatal): %s", exc)

        self._app_mode = os.environ.get("APP_MODE", "combined")

        self._bg_thread = threading.Thread(
            target=self._background_fetch_loop, daemon=True, name="bg-fetch"
        )
        self._bg_thread.start()

        self._tg_thread = threading.Thread(
            target=self._telegram_polling_loop, daemon=True, name="tg-poll"
        )
        if self._app_mode in ("combined", "worker"):
            self._tg_thread.start()

    def _background_fetch_loop(self) -> None:
        """Continuously fetches and caches clean data; pre-warms the predictor.

        Runs on a daemon thread. Exits cleanly when ``_stop_event`` is set.
        """
        _heartbeat_counter = 0
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
                    self._last_fetch_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    if not self._last_error or "Prep Error" not in self._last_error:
                        self._last_error = None

                if self._app_mode != "api":
                    self._check_and_send_alerts(clean_df)
                    self._check_and_send_hourly_report(clean_df)
                    self._check_autoencoder_anomaly(clean_df)
                    self._check_and_generate_daily_video()
                    self._check_outdoor_sensor_sanity(clean_df)

                # Publish state for API processes to read via /health.
                if self._app_mode in ("combined", "worker"):
                    with self._lock:
                        _state_snapshot = {
                            "last_fetch_utc": self._last_fetch_utc,
                            "last_error": self._last_error,
                            "cache_rows": (
                                int(len(self._cached_clean_df))
                                if self._cached_clean_df is not None
                                else None
                            ),
                            "model_fitted": self._predictor.is_fitted,
                            "bg_thread_alive": True,
                            "tg_thread_alive": self._tg_thread.is_alive(),
                        }
                    try:
                        worker_state.write_state(_state_snapshot)
                    except Exception as exc:
                        logger.warning("worker_state write failed: %s", exc)

            except Exception as exc:
                with self._lock:
                    self._last_error = f"Background Fetch Error: {exc}"

            # Insert system heartbeat every 10 iterations (~5 minutes at 30s cadence).
            # Only the worker/combined process should own the heartbeat record.
            _heartbeat_counter += 1
            if _heartbeat_counter >= 10 and self._app_mode != "api":
                _heartbeat_counter = 0
                with self._lock:
                    _last_err = self._last_error
                def _insert_heartbeat(last_error: str | None) -> None:
                    try:
                        if self._supabase is None:
                            return
                        now = datetime.now(timezone.utc)
                        self._supabase.table("system_heartbeat").insert(
                            {
                                "ts": now.isoformat(),
                                "status": "ok",
                                "last_error": last_error,
                            }
                        ).execute()
                        cutoff = (now - timedelta(days=3)).isoformat()
                        self._supabase.table("system_heartbeat").delete().lt("ts", cutoff).execute()
                    except Exception:
                        pass
                threading.Thread(target=_insert_heartbeat, args=(_last_err,), daemon=True).start()

            # Throttle to prevent rate-limits from Google Sheets.
            self._stop_event.wait(timeout=30.0)

    def _log_to_supabase(self, table: str, record: dict) -> None:
        """Fire-and-forget insert to Supabase. Errors are swallowed silently."""
        if self._supabase is None:
            return
        def _insert() -> None:
            try:
                self._supabase.table(table).insert(record).execute()
            except Exception:
                pass
        threading.Thread(target=_insert, daemon=True).start()

    def _send_telegram_message(self, text: str, target_chat_id: str | None = None) -> None:
        """Sends a message via the Telegram Bot API.

        Silently swallows network errors so callers are not disrupted by
        transient Telegram outages.
        """
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
                resp.read()  # consume body to allow keep-alive reuse
        except Exception as exc:
            logger.error("Telegram send error: %s", exc)

    def _send_telegram_photo(
        self,
        buf: io.BytesIO,
        caption: str,
        target_chat_id: str,
    ) -> None:
        """Sends a photo via the Telegram Bot API using multipart upload."""
        if not self._bot_token:
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendPhoto"
            buf.seek(0)
            requests.post(
                url,
                data={"chat_id": target_chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("chart.png", buf, "image/png")},
                timeout=30,
            )
        except Exception as exc:
            logger.error("sendPhoto failed: %s", exc)

    def _send_telegram_video(
        self,
        video_file: Path,
        caption: str,
        target_chat_id: str,
    ) -> None:
        """Send a local MP4 file via Telegram sendVideo."""
        if not self._bot_token:
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendVideo"
            with video_file.open("rb") as f:
                requests.post(
                    url,
                    data={"chat_id": target_chat_id, "caption": caption,
                          "supports_streaming": "true"},
                    files={"video": (video_file.name, f, "video/mp4")},
                    timeout=120,
                )
        except Exception as exc:
            logger.error("sendVideo failed: %s", exc)

    def _send_inline_keyboard(
        self,
        text: str,
        buttons: list[list[dict]],
        target_chat_id: str,
    ) -> int | None:
        """Send message with inline keyboard. Returns message_id."""
        if not self._bot_token:
            return None
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": target_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "reply_markup": {"inline_keyboard": buttons},
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload,
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("result", {}).get("message_id")
        except Exception as exc:
            logger.error("sendInlineKeyboard failed: %s", exc)
            return None

    def _edit_message_text(
        self,
        chat_id: str,
        message_id: int,
        text: str,
        buttons: list[list[dict]] | None = None,
    ) -> None:
        """Edit an existing message text and optionally update its keyboard."""
        if not self._bot_token:
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/editMessageText"
            body: dict = {"chat_id": chat_id, "message_id": message_id,
                          "text": text, "parse_mode": "HTML"}
            if buttons is not None:
                body["reply_markup"] = {"inline_keyboard": buttons}
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(url, data=payload,
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as exc:
            logger.error("editMessageText failed: %s", exc)

    def _answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        """Dismiss the loading spinner on an inline button press."""
        if not self._bot_token:
            return
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/answerCallbackQuery"
            payload = json.dumps({"callback_query_id": callback_query_id, "text": text}).encode("utf-8")
            req = urllib.request.Request(url, data=payload,
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers — sensor threshold evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_sensor_alerts(latest: pd.Series) -> list[tuple[str, float, str]]:
        """Returns list of (column, value, reason) for all breached thresholds.

        Uses the module-level ``_SENSOR_THRESHOLDS`` table as the single
        source of truth, keeping alert logic consistent across callers.
        """
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
        now_mono = time.monotonic()
        breaches = _dedupe_breaches(self._evaluate_sensor_alerts(latest))
        current_breach_cols = {col for col, _, _ in breaches}

        # Always log every breach to Supabase regardless of Telegram state.
        for col, val, reason in breaches:
            parts = col.split("_", 1)
            node = parts[0] if len(parts) == 2 else col
            metric = parts[1] if len(parts) == 2 else ""
            threshold_type = "HIGH" if ">" in reason or "exceeded" in reason.lower() else "LOW"
            self._log_to_supabase("alert_logs", {
                "node_id": node,
                "metric": metric,
                "value": val,
                "threshold_type": threshold_type,
                "reason": reason,
            })

        # Categorise each current breach as new or persistent.
        for col, val, reason in breaches:
            if col not in self._active_breaches:
                self._active_breaches.add(col)
                last_sent = self._sensor_last_alert_sent_at.get(col, 0.0)
                if now_mono - last_sent >= _SENSOR_ALERT_COOLDOWN_SECONDS:
                    self._sensor_last_alert_sent_at[col] = now_mono
                    self._persistent_alert_sent_at[col] = now_mono
                    self._alert_buffer.append((col, val, reason))
                    if self._buffer_flush_at == 0.0:
                        self._buffer_flush_at = now_mono + _ALERT_BUFFER_WINDOW_SECONDS
                else:
                    # Keep breach as active but suppress noisy re-alerts inside cooldown.
                    self._persistent_alert_sent_at[col] = max(
                        self._persistent_alert_sent_at.get(col, 0.0),
                        last_sent,
                    )
                    logger.info(
                        "Alert suppressed by cooldown: %s (%.0fs remaining)",
                        col,
                        max(0.0, _SENSOR_ALERT_COOLDOWN_SECONDS - (now_mono - last_sent)),
                    )
            else:
                last_sent = self._persistent_alert_sent_at.get(col, 0.0)
                if now_mono - last_sent > _PERSISTENT_REALERT_SECONDS:
                    self._persistent_alert_sent_at[col] = now_mono
                    self._alert_buffer.append((col, val, "\x00" + reason))  # \x00 = reminder marker
                    if self._buffer_flush_at == 0.0:
                        self._buffer_flush_at = now_mono + _ALERT_BUFFER_WINDOW_SECONDS

        # Resolve breaches that are no longer detected, with hysteresis.
        for col in list(self._active_breaches - current_breach_cols):
            suffix = next((s for s in _HYSTERESIS_MARGIN if col.endswith(s)), None)
            if suffix is None:
                self._active_breaches.discard(col)
                self._persistent_alert_sent_at.pop(col, None)
                continue
            raw = latest.get(col)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                self._active_breaches.discard(col)
                self._persistent_alert_sent_at.pop(col, None)
                continue
            val_f = float(raw)
            margin = _HYSTERESIS_MARGIN[suffix]
            thresh = _SENSOR_THRESHOLDS[suffix]
            in_hysteresis = (
                (thresh.hi is not None and val_f > thresh.hi - margin)
                or (thresh.lo is not None and val_f < thresh.lo + margin)
            )
            if not in_hysteresis:
                self._active_breaches.discard(col)
                self._persistent_alert_sent_at.pop(col, None)

        # Flush buffer once the window expires.
        if self._alert_buffer and self._buffer_flush_at > 0.0 and now_mono >= self._buffer_flush_at:
            new_items = _dedupe_breaches(
                [(c, v, r) for c, v, r in self._alert_buffer if not r.startswith("\x00")]
            )
            reminder_items = _dedupe_breaches(
                [(c, v, r[1:]) for c, v, r in self._alert_buffer if r.startswith("\x00")]
            )
            dt = _now_ict()
            if new_items:
                lines = _build_alert_lines(new_items)
                msg = (
                    f"<b>[ALERT] UEH Campus V \u2014 {_fmt_ts(dt)}</b>\n"
                    "<pre>" + "\n".join(lines) + "</pre>"
                )
                self._send_telegram_message(msg)
            if reminder_items:
                lines = _build_alert_lines(reminder_items)
                msg = (
                    f"<b>[REMINDER] V\u1eabn c\u00f2n anomaly sau 12h \u2014 {_fmt_ts(dt)}</b>\n"
                    "<pre>" + "\n".join(lines) + "</pre>"
                )
                self._send_telegram_message(msg)
            self._alert_buffer.clear()
            self._buffer_flush_at = 0.0

    def _check_outdoor_sensor_sanity(self, clean_df: pd.DataFrame) -> None:
        """Compare outdoor/semi-outdoor node readings against Open-Meteo.

        Fires a Telegram alert when a node deviates beyond its configured threshold,
        suggesting sensor drift, obstruction, or hardware fault. Each node+metric
        combination is rate-limited to one alert per _WEATHER_SANITY_COOLDOWN_SECONDS.
        """
        if not self._outdoor_config or clean_df.empty or not self._bot_token:
            return

        campus = self._outdoor_config.get("campus_latlon", {})
        lat = campus.get("lat")
        lon = campus.get("lon")
        nodes_cfg: dict[str, Any] = self._outdoor_config.get("nodes", {})
        if not lat or not lon or not nodes_cfg:
            return

        current = weather_client.get_current(lat, lon)
        if current is None:
            return  # API unavailable — skip silently, fallback already logged in client

        latest = clean_df.iloc[-1]
        now = time.monotonic()

        # Mapping from config threshold key to (column suffix, Open-Meteo value)
        metric_map: dict[str, tuple[str, float]] = {
            "Temp": ("_Temp", current.temperature_c),
            "Humid": ("_Humid", current.humidity_pct),
        }

        alerts: list[str] = []
        for node_id, node_cfg in nodes_cfg.items():
            node_type: str = node_cfg.get("type", "outdoor")
            thresholds: dict[str, float] = node_cfg.get("sanity_thresholds", {})

            for metric, (col_suffix, weather_val) in metric_map.items():
                threshold = thresholds.get(metric)
                if threshold is None:
                    continue

                col = f"{node_id}{col_suffix}"
                if col not in latest.index:
                    continue

                sensor_val = float(latest[col])
                deviation = abs(sensor_val - weather_val)
                if deviation <= threshold:
                    continue

                cooldown_key = f"{node_id}_{metric}"
                last_sent = self._weather_sanity_alerted_at.get(cooldown_key, 0.0)
                if now - last_sent < _WEATHER_SANITY_COOLDOWN_SECONDS:
                    continue

                self._weather_sanity_alerted_at[cooldown_key] = now
                unit = "°C" if metric == "Temp" else "%"
                label = "outdoor" if node_type == "outdoor" else "semi-outdoor"
                alerts.append(
                    f"  {node_id} ({label}) {metric}: sensor={sensor_val:.1f}{unit}, "
                    f"weather={weather_val:.1f}{unit}, diff={deviation:.1f}{unit}"
                )

        if alerts:
            body = "\n".join(alerts)
            msg = (
                f"<b>[SENSOR DRIFT?] Outdoor sensor vs weather mismatch</b>\n"
                f"<pre>{body}</pre>\n"
                f"Check for obstruction, direct sunlight, or hardware fault."
            )
            self._send_telegram_message(msg)

    def _check_and_send_hourly_report(self, clean_df: pd.DataFrame) -> None:
        if clean_df.empty or not self._bot_token:
            return

        dt           = _now_ict()
        current_hour = dt.hour

        with self._lock:
            if dt.minute >= 2 or current_hour == self._last_report_hour:
                return
            self._last_report_hour = current_hour

        latest   = clean_df.iloc[-1]
        breaches = self._evaluate_sensor_alerts(latest)
        status   = f"Anomalies: <b>{len(breaches)}</b>" if breaches else "Status: <b>NORMAL</b>"
        table    = _build_node_table(latest, list(clean_df.columns))

        msg = (
            f"<b>[HOURLY REPORT] {_fmt_ts(dt)}</b>\n"
            f"{status}\n\n"
            "<pre>" + table + "</pre>"
        )
        if breaches:
            active_lines = _build_alert_lines(breaches)
            msg += "\n<b>Active alerts:</b>\n<pre>" + "\n".join(active_lines) + "</pre>"
        self._send_telegram_message(msg)

    def _check_autoencoder_anomaly(self, clean_df: pd.DataFrame) -> None:
        if self._anomaly_detector is None or clean_df.empty or not self._bot_token:
            return

        result = self._anomaly_detector.score(clean_df)
        if result is None or not result.is_anomaly:
            return

        now = time.monotonic()
        with self._lock:
            if now < self._ae_cooldown_until:
                return
            self._ae_cooldown_until = now + _AE_ALERT_COOLDOWN_SECONDS

        top = ", ".join(f"{f} ({e:.3f})" for f, e in result.top_features)
        dt  = _now_ict()
        msg = (
            f"<b>[AE ANOMALY] {_fmt_ts(dt)}</b>\n"
            f"Score: {result.reconstruction_error:.4f}  (threshold {result.threshold:.4f})\n"
            f"<pre>{top}</pre>"
        )
        self._send_telegram_message(msg)
        self._log_to_supabase("alert_logs", {
            "node_id": None,
            "metric": "MULTIVARIATE",
            "value": result.reconstruction_error,
            "threshold_type": "AE",
            "reason": top,
        })

    def _check_and_generate_daily_video(self) -> None:
        """Trigger daily video generation at 00:10 ICT. Fire-and-forget thread."""
        now = _now_ict()
        if now.hour != 0 or now.minute < 10 or now.minute >= 15:
            return
        yesterday = (now - timedelta(days=1)).date()
        with self._lock:
            if self._last_video_date == yesterday:
                return
            self._last_video_date = yesterday

        def _run() -> None:
            try:
                out = generate_video(target_date=yesterday, csv_url=_VIDEO_CSV_URL)
                caption = (
                    f"<b>Daily Heatmap — {yesterday.strftime('%d/%m/%Y')}</b>\n"
                    f"Temp · Humid · CO2 · TVOC  |  48 frames"
                )
                self._send_telegram_video(out, caption, self._chat_id)
            except Exception as exc:
                logger.error("Daily video generation failed: %s", exc)
                self._send_telegram_message(
                    f"[VIDEO] Daily video generation failed: {exc}", self._chat_id
                )

        threading.Thread(target=_run, daemon=True, name="daily-video").start()

    def _handle_video_command(
        self,
        arg: str,
        target_chat_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> None:
        """Handle /video [YYYY-MM-DD] — serve cached video or generate on demand."""
        if arg:
            try:
                target = date.fromisoformat(arg.strip())
            except ValueError:
                self._send_telegram_message(
                    "Invalid date format. Use /video or /video YYYY-MM-DD",
                    target_chat_id,
                )
                return
        else:
            target = (_now_ict() - timedelta(days=1)).date()

        cached = video_path(target)
        if cached.exists():
            caption = (
                f"<b>Daily Heatmap — {target.strftime('%d/%m/%Y')}</b>\n"
                f"Temp · Humid · CO2 · TVOC  |  48 frames"
            )
            self._send_telegram_video(cached, caption, target_chat_id)
            return

        self._send_telegram_message(
            f"Generating video for {target} — this takes a minute...",
            target_chat_id,
        )

        def _run() -> None:
            try:
                out = generate_video(target_date=target, csv_url=_VIDEO_CSV_URL)
                caption = (
                    f"<b>Daily Heatmap — {target.strftime('%d/%m/%Y')}</b>\n"
                    f"Temp · Humid · CO2 · TVOC  |  48 frames"
                )
                self._send_telegram_video(out, caption, target_chat_id)
            except Exception as exc:
                logger.error("On-demand video failed: %s", exc)
                self._send_telegram_message(f"[VIDEO] Generation failed: {exc}", target_chat_id)

        threading.Thread(target=_run, daemon=True, name="ondemand-video").start()

    def _telegram_polling_loop(self) -> None:
        """Long-polls the Telegram Bot API and dispatches incoming commands.

        Blocks on ``_credentials_ready`` until a valid bot token and chat ID
        are available, avoiding a busy-wait when credentials are absent.
        Exits cleanly when ``_stop_event`` is set.
        """
        last_update_id: int = 0
        while not self._stop_event.is_set():
            # Block until credentials become available (set in __init__ or later).
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

                        # Handle inline keyboard button presses
                        cq = update.get("callback_query")
                        if cq:
                            self._handle_callback_query(cq)
                            continue

                        # Support DMs/groups ('message') and broadcast channels ('channel_post').
                        msg_node = update.get("message") or update.get("channel_post")
                        if not (msg_node and "text" in msg_node):
                            continue

                        # Strip @botname suffix added by Telegram command menu autocomplete
                        # e.g. "/compare@UEH_V_DigitalTwin_bot" → "/compare"
                        raw_text = msg_node["text"]
                        text = raw_text.split("@")[0].lower() if raw_text.startswith("/") else raw_text.lower()
                        chat_id = str(msg_node["chat"]["id"])
                        sender = msg_node.get("from") or {}
                        user_id = str(sender.get("id", "")) or None
                        username = sender.get("username") or sender.get("first_name") or None

                        ctx = dict(target_chat_id=chat_id, user_id=user_id, username=username)

                        if text.startswith("/getcurrent_detail"):
                            self._handle_telegram_command("detail", **ctx)
                        elif text.startswith("/getcurrent_short"):
                            self._handle_telegram_command("short", **ctx)
                        elif text.startswith("/getcurrent_alert"):
                            self._handle_telegram_command("alert", **ctx)
                        elif text.startswith("/ask"):
                            query = raw_text.split("@")[0][len("/ask"):].strip()
                            self._handle_ask_command(query, **ctx)
                        elif text.startswith("/video"):
                            arg = raw_text.split("@")[0][len("/video"):].strip()
                            self._handle_video_command(arg, **ctx)
                        else:
                            self._dispatch_viz_command(text, **ctx)

            except Exception as exc:
                logger.error("Telegram polling error: %s", exc)
                self._stop_event.wait(timeout=5.0)

    # ── Regex-based viz dispatcher ─────────────────────────────────────────

    _VIZ_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^/chart_(hour|day|week)_(all|m\d+)_(temp|humid|co2|tvoc)$"), "chart"),
        (re.compile(r"^/predict_(m\d+)_(temp|humid|co2|tvoc)$"),                   "predict"),
        (re.compile(r"^/heatmap_(temp|humid|co2|tvoc)$"),                          "heatmap"),
        (re.compile(r"^/compare_(m\d+)_(m\d+)$"),                                  "compare"),
        (re.compile(r"^/rank_(temp|humid|co2|tvoc)$"),                             "rank"),
    ]

    _VIZ_PREFIXES = ("/chart_", "/predict_", "/heatmap_", "/compare_", "/rank_")

    _METRIC_BUTTONS: list[list[dict]] = [[
        {"text": "Temp",  "callback_data": "viz:sel:metric:temp"},
        {"text": "Humid", "callback_data": "viz:sel:metric:humid"},
        {"text": "CO2",   "callback_data": "viz:sel:metric:co2"},
        {"text": "TVOC",  "callback_data": "viz:sel:metric:tvoc"},
    ]]

    _RANGE_BUTTONS: list[list[dict]] = [[
        {"text": "Hour", "callback_data": "viz:chart:range:hour"},
        {"text": "Day",  "callback_data": "viz:chart:range:day"},
        {"text": "Week", "callback_data": "viz:chart:range:week"},
    ]]

    _NODE_BUTTONS: list[list[dict]] = [
        [
            {"text": f"{n} — {NODE_NAMES[n]}", "callback_data": f"viz:sel:node:{n}"}
            for n in NODE_ORDER[:4]
        ],
        [
            {"text": f"{n} — {NODE_NAMES[n]}", "callback_data": f"viz:sel:node:{n}"}
            for n in NODE_ORDER[4:]
        ],
    ]

    # Node buttons for /chart — same list but callback carries the pending range.
    # Built dynamically in the callback handler once the range is known.

    _VIZ_HELP = (
        "Available commands:\n"
        "  /chart_&lt;hour|day|week&gt;_&lt;all|node&gt;_&lt;temp|humid|co2|tvoc&gt;\n"
        "  /predict_&lt;node&gt;_&lt;temp|humid|co2|tvoc&gt;\n"
        "  /heatmap_&lt;temp|humid|co2|tvoc&gt;\n"
        "  /compare_&lt;nodeA&gt;_&lt;nodeB&gt;\n"
        "  /rank_&lt;temp|humid|co2|tvoc&gt;\n"
        "Nodes: M1 (Canteen Garden), M4 (Studio ISCM), M6 (ISCM Staircase),\n"
        "       M7 (Sky Garden), M8 (ISCM Balcony), M9 (Hotel Kitchen),\n"
        "       M10 (Hotel Corridor), M11 (Hotel Balcony)"
    )

    def _start_chart_keyboard(self, target_chat_id: str, **_) -> None:
        self._send_inline_keyboard(
            "Select time range:", self._RANGE_BUTTONS, target_chat_id
        )

    def _start_heatmap_keyboard(self, target_chat_id: str, **_) -> None:
        self._send_inline_keyboard(
            "Select metric for heatmap:", self._METRIC_BUTTONS, target_chat_id
        )

    def _start_rank_keyboard(self, target_chat_id: str, **_) -> None:
        self._send_inline_keyboard(
            "Select metric to rank:", self._METRIC_BUTTONS, target_chat_id
        )

    def _start_compare_keyboard(self, target_chat_id: str, **_) -> None:
        self._send_inline_keyboard(
            "Select first node to compare:", self._NODE_BUTTONS, target_chat_id
        )

    def _start_predict_keyboard(self, target_chat_id: str, **_) -> None:
        self._send_inline_keyboard(
            "Select node to forecast:", self._NODE_BUTTONS, target_chat_id
        )

    def _dispatch_viz_command(
        self,
        text: str,
        target_chat_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> None:
        # Bare command → trigger inline keyboard
        _KEYBOARD_MAP = {
            "/chart":   self._start_chart_keyboard,
            "/heatmap": self._start_heatmap_keyboard,
            "/rank":    self._start_rank_keyboard,
            "/compare": self._start_compare_keyboard,
            "/predict": self._start_predict_keyboard,
        }
        if text in _KEYBOARD_MAP:
            _KEYBOARD_MAP[text](target_chat_id=target_chat_id)
            return

        if not any(text.startswith(p) for p in self._VIZ_PREFIXES):
            return  # not a viz command — ignore silently

        for pattern, kind in self._VIZ_PATTERNS:
            m = pattern.match(text)
            if m:
                groups = m.groups()
                threading.Thread(
                    target=self._run_viz_command,
                    args=(kind, groups, target_chat_id, user_id, username),
                    daemon=True,
                ).start()
                return

        # Matched prefix but failed full pattern — bad params
        self._send_telegram_message(
            f"Invalid command format.\n\n{self._VIZ_HELP}", target_chat_id
        )

    def _handle_callback_query(self, cq: dict) -> None:
        query_id   = cq["id"]
        data       = cq.get("data", "")
        chat_id    = str(cq["message"]["chat"]["id"])
        message_id = cq["message"]["message_id"]
        sender     = cq.get("from") or {}
        user_id    = str(sender.get("id", "")) or None
        username   = sender.get("username") or sender.get("first_name") or None

        self._answer_callback_query(query_id)

        if not data.startswith("viz:"):
            return

        parts = data.split(":")
        # parts[0] = "viz", parts[1] = step/command, parts[2+] = values

        ctx = dict(target_chat_id=chat_id, user_id=user_id, username=username)

        # viz:chart:range:<range> → show node selection
        if parts[1] == "chart" and parts[2] == "range":
            range_str = parts[3]
            buttons = [
                [{"text": "All Nodes", "callback_data": f"viz:chart:node:{range_str}:all"}],
                [
                    {"text": f"{n} — {NODE_NAMES[n]}",
                     "callback_data": f"viz:chart:node:{range_str}:{n}"}
                    for n in NODE_ORDER[:4]
                ],
                [
                    {"text": f"{n} — {NODE_NAMES[n]}",
                     "callback_data": f"viz:chart:node:{range_str}:{n}"}
                    for n in NODE_ORDER[4:]
                ],
            ]
            self._edit_message_text(chat_id, message_id,
                                    f"Range: <b>{range_str}</b>  —  Select node:",
                                    buttons)

        # viz:chart:node:<range>:<node> → show metric selection
        elif parts[1] == "chart" and parts[2] == "node":
            range_str, node = parts[3], parts[4]
            node_label = "All Nodes" if node == "all" else f"{node} — {NODE_NAMES.get(node, node)}"
            buttons = [[
                {"text": "Temp",  "callback_data": f"viz:chart:go:{range_str}:{node}:temp"},
                {"text": "Humid", "callback_data": f"viz:chart:go:{range_str}:{node}:humid"},
                {"text": "CO2",   "callback_data": f"viz:chart:go:{range_str}:{node}:co2"},
                {"text": "TVOC",  "callback_data": f"viz:chart:go:{range_str}:{node}:tvoc"},
            ]]
            self._edit_message_text(
                chat_id, message_id,
                f"Range: <b>{range_str}</b>  |  Node: <b>{node_label}</b>  —  Select metric:",
                buttons,
            )

        # viz:chart:go:<range>:<node>:<metric> → render chart
        elif parts[1] == "chart" and parts[2] == "go":
            range_str, node, metric = parts[3], parts[4], parts[5]
            node_label = "All Nodes" if node == "all" else node
            self._edit_message_text(chat_id, message_id,
                                    f"Generating chart — {range_str} / {node_label} / {metric}...")
            threading.Thread(
                target=self._run_viz_command,
                args=("chart", (range_str, node, metric), chat_id, user_id, username),
                daemon=True,
            ).start()

        # viz:sel:metric:<metric> — used by heatmap and rank (need context from message text)
        elif parts[1] == "sel" and parts[2] == "metric":
            metric = parts[3]
            original_text = cq["message"].get("text", "")
            if "heatmap" in original_text.lower():
                self._edit_message_text(chat_id, message_id,
                                        f"Generating /heatmap_{metric}...")
                threading.Thread(
                    target=self._run_viz_command,
                    args=("heatmap", (metric,), chat_id, user_id, username),
                    daemon=True,
                ).start()
            else:  # rank
                self._edit_message_text(chat_id, message_id,
                                        f"Generating /rank_{metric}...")
                threading.Thread(
                    target=self._run_viz_command,
                    args=("rank", (metric,), chat_id, user_id, username),
                    daemon=True,
                ).start()

        # viz:sel:node:<node> — used by compare (pick nodeA) and predict (pick node)
        elif parts[1] == "sel" and parts[2] == "node":
            node = parts[3]
            original_text = cq["message"].get("text", "")
            if "compare" in original_text.lower() or "Select first node" in original_text:
                # First node selected, now pick second
                buttons = [
                    [
                        {"text": f"{n} — {NODE_NAMES[n]}",
                         "callback_data": f"viz:compare:go:{node}:{n}"}
                        for n in NODE_ORDER[:4] if n != node
                    ],
                    [
                        {"text": f"{n} — {NODE_NAMES[n]}",
                         "callback_data": f"viz:compare:go:{node}:{n}"}
                        for n in NODE_ORDER[4:] if n != node
                    ],
                ]
                self._edit_message_text(
                    chat_id, message_id,
                    f"Node A: <b>{node} — {NODE_NAMES.get(node, node)}</b>\n\nSelect node B:",
                    buttons,
                )
            else:  # predict: node selected, now pick metric
                buttons = [[
                    {"text": "Temp",  "callback_data": f"viz:predict:go:{node}:temp"},
                    {"text": "Humid", "callback_data": f"viz:predict:go:{node}:humid"},
                    {"text": "CO2",   "callback_data": f"viz:predict:go:{node}:co2"},
                    {"text": "TVOC",  "callback_data": f"viz:predict:go:{node}:tvoc"},
                ]]
                self._edit_message_text(
                    chat_id, message_id,
                    f"Node: <b>{node} — {NODE_NAMES.get(node, node)}</b>\n\nSelect metric to forecast:",
                    buttons,
                )

        # viz:compare:go:<nodeA>:<nodeB>
        elif parts[1] == "compare" and parts[2] == "go":
            node_a, node_b = parts[3], parts[4]
            self._edit_message_text(chat_id, message_id,
                                    f"Generating /compare_{node_a}_{node_b}...")
            threading.Thread(
                target=self._run_viz_command,
                args=("compare", (node_a.lower(), node_b.lower()), chat_id, user_id, username),
                daemon=True,
            ).start()

        # viz:predict:go:<node>:<metric>
        elif parts[1] == "predict" and parts[2] == "go":
            node, metric = parts[3], parts[4]
            self._edit_message_text(chat_id, message_id,
                                    f"Generating /predict_{node}_{metric}...")
            threading.Thread(
                target=self._run_viz_command,
                args=("predict", (node.lower(), metric), chat_id, user_id, username),
                daemon=True,
            ).start()

    def _run_viz_command(
        self,
        kind: str,
        groups: tuple[str, ...],
        target_chat_id: str,
        user_id: str | None,
        username: str | None,
    ) -> None:
        start_time = time.monotonic()
        with self._lock:
            df = self._cached_clean_df

        if df is None or df.empty:
            self._send_telegram_message(
                "System is starting up. Please try again in a moment.", target_chat_id
            )
            return

        try:
            buf: io.BytesIO | None = None
            caption = ""
            text_reply: str | None = None

            if kind == "chart":
                range_str, node, metric = groups
                buf = viz_engine.chart(df, range_str, node, metric)
                node_label = "all" if node == "all" else node.upper()
                caption = f"/chart_{range_str}_{node_label}_{metric}"

            elif kind == "predict":
                node, metric = groups[0].upper(), groups[1]
                if node not in NODE_ORDER:
                    self._send_telegram_message(
                        f"Unknown node {node}. Valid nodes: {', '.join(NODE_ORDER)}",
                        target_chat_id,
                    )
                    return
                with self._lock:
                    predictor = self._predictor
                if not predictor.is_fitted:
                    self._send_telegram_message("Model not ready yet. Please try again shortly.", target_chat_id)
                    return
                buf = viz_engine.predict(df, node, metric, predictor)
                caption = f"/predict_{node}_{metric}"

            elif kind == "heatmap":
                metric = groups[0]
                root   = Path(__file__).parent.parent.parent
                views = [
                    (root / "node_coords_v0.json", root / "campus_3d.png"),
                    (root / "node_coords_v1.json", root / "campus_3d_1.png"),
                ]
                for coords, image in views:
                    v_buf = viz_engine.heatmap(df, metric, coords, image)
                    self._send_telegram_photo(v_buf, f"/heatmap_{metric}", target_chat_id)
                buf = None
                caption = ""

            elif kind == "compare":
                node_a, node_b = groups[0].upper(), groups[1].upper()
                if node_a not in NODE_ORDER:
                    self._send_telegram_message(
                        f"Unknown node {node_a}. Valid nodes: {', '.join(NODE_ORDER)}",
                        target_chat_id,
                    )
                    return
                if node_b not in NODE_ORDER:
                    self._send_telegram_message(
                        f"Unknown node {node_b}. Valid nodes: {', '.join(NODE_ORDER)}",
                        target_chat_id,
                    )
                    return
                if node_a == node_b:
                    self._send_telegram_message(
                        "Please select two different nodes.", target_chat_id
                    )
                    return
                buf = viz_engine.compare(df, node_a, node_b)
                caption = f"/compare_{node_a}_{node_b}"

            elif kind == "rank":
                metric = groups[0]
                text_reply = viz_engine.rank(df, metric)

            if buf is not None:
                self._send_telegram_photo(buf, caption, target_chat_id)
            elif text_reply is not None:
                self._send_telegram_message(text_reply, target_chat_id)

        except Exception as exc:
            logger.error("viz command %s failed: %s", kind, exc)
            self._send_telegram_message(f"Error processing command: {exc}", target_chat_id)
            return

        latency_ms = int((time.monotonic() - start_time) * 1000)
        username_hash = hashlib.sha256(username.encode()).hexdigest()[:16] if username else None
        self._log_to_supabase("bot_logs", {
            "user_id": user_id,
            "username_hash": username_hash,
            "chat_id": target_chat_id,
            "command": kind,
            "query": None,
            "response": None,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "is_cached": False,
        })

    def _handle_telegram_command(
        self,
        cmd_type: str,
        target_chat_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> None:
        start_time = time.monotonic()
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
        msg: str

        if cmd_type == "detail":
            table  = _build_node_table(latest, list(df.columns), show_names=True)
            status = f"Anomalies: <b>{len(breaches)}</b>" if breaches else "Status: <b>NORMAL</b>"
            msg = (
                f"<b>[DETAIL] {_fmt_ts(dt)}</b>\n"
                f"{status}\n\n"
                "<pre>" + table + "</pre>"
            )
            if breaches:
                msg += "\n<b>Active alerts:</b>\n<pre>" + "\n".join(_build_alert_lines(breaches)) + "</pre>"
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
                msg += "\n<pre>" + "\n".join(_build_alert_lines(breaches)) + "</pre>"
            self._send_telegram_message(msg, target_chat_id)

        elif cmd_type == "alert":
            if not breaches:
                msg = f"<b>[ALERT] {_fmt_ts(dt)}</b>\n\nNo active anomalies."
            else:
                msg = (
                    f"<b>[ALERT] {_fmt_ts(dt)}</b>\n"
                    "<pre>" + "\n".join(_build_alert_lines(breaches)) + "</pre>"
                )
            self._send_telegram_message(msg, target_chat_id)
        else:
            return

        latency_ms = int((time.monotonic() - start_time) * 1000)
        username_hash = hashlib.sha256(username.encode()).hexdigest()[:16] if username else None
        self._log_to_supabase("bot_logs", {
            "user_id": user_id,
            "username_hash": username_hash,
            "chat_id": target_chat_id,
            "command": cmd_type,
            "query": None,
            "response": msg,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "is_cached": False,
        })

    def _handle_ask_command(
        self,
        query: str,
        target_chat_id: str,
        user_id: str | None = None,
        username: str | None = None,
    ) -> None:
        start_time = time.monotonic()
        if not query:
            self._send_telegram_message(
                "[AI] Please provide a question after /ask.\n"
                "Example: /ask Is M1 temperature normal right now?",
                target_chat_id,
            )
            return

        if self._ai_assistant is None:
            self._send_telegram_message(
                "[AI] AI assistant is not available. Check GROQ_API_KEY environment variable.",
                target_chat_id,
            )
            return

        try:
            result = self._ai_assistant.answer(query)
            answer_text = result["answer"]
            self._send_telegram_message(f"[AI] {answer_text}", target_chat_id)

            latency_ms = result.get("latency_ms") or int((time.monotonic() - start_time) * 1000)
            username_hash = hashlib.sha256(username.encode()).hexdigest()[:16] if username else None
            self._log_to_supabase("bot_logs", {
                "user_id": user_id,
                "username_hash": username_hash,
                "chat_id": target_chat_id,
                "command": "ask",
                "query": query,
                "response": answer_text,
                "model": result.get("model"),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "latency_ms": latency_ms,
                "is_cached": result.get("is_cached", False),
            })
        except Exception as exc:
            logger.error("AIAssistant.answer failed: %s", exc)
            self._send_telegram_message(
                f"[AI] Error processing your question: {exc}",
                target_chat_id,
            )

    def health(self) -> dict[str, Any]:
        """Returns runtime health information for service monitoring."""
        if self._app_mode == "api":
            return self._health_from_worker_state()
        return self._health_local()

    def _health_from_worker_state(self) -> dict[str, Any]:
        """Health derived from the shared state file written by the worker process."""
        state = worker_state.read_state()
        if state is None:
            return {
                "status": "unhealthy",
                "last_error": "Worker state unavailable: state file missing or corrupt.",
                "bg_thread_alive": False,
                "tg_thread_alive": False,
                "cache_rows": None,
                "model_fitted": self._predictor.is_fitted,
                "checkpoint_path": self._config.checkpoint_path.as_posix(),
                "checkpoint_checked": self._checkpoint_checked,
                "last_prediction_utc": self._last_prediction_utc,
                "last_fetch_utc": None,
                "data_age_seconds": None,
            }

        last_fetch_utc: str | None = state.get("last_fetch_utc")
        data_age_seconds: int | None = None
        if last_fetch_utc:
            try:
                last_fetch_dt = datetime.fromisoformat(last_fetch_utc)
                if last_fetch_dt.tzinfo is None:
                    last_fetch_dt = last_fetch_dt.replace(tzinfo=timezone.utc)
                data_age_seconds = max(0, int((datetime.now(timezone.utc) - last_fetch_dt).total_seconds()))
            except ValueError:
                data_age_seconds = None

        bg_alive: bool = state.get("bg_thread_alive", False)
        last_error: str | None = state.get("last_error")

        if (not bg_alive) or (data_age_seconds is not None and data_age_seconds > 900):
            status = "unhealthy"
        elif last_error or data_age_seconds is None or data_age_seconds >= 300:
            status = "degraded"
        else:
            status = "ok"

        return {
            "status": status,
            "model_fitted": state.get("model_fitted", self._predictor.is_fitted),
            "checkpoint_path": self._config.checkpoint_path.as_posix(),
            "checkpoint_checked": self._checkpoint_checked,
            "last_prediction_utc": self._last_prediction_utc,
            "last_error": last_error,
            "last_fetch_utc": last_fetch_utc,
            "data_age_seconds": data_age_seconds,
            "bg_thread_alive": bg_alive,
            "tg_thread_alive": state.get("tg_thread_alive", False),
            "cache_rows": state.get("cache_rows"),
        }

    def _health_local(self) -> dict[str, Any]:
        """Health derived from local thread and cache state (combined/worker mode)."""
        bg_thread_alive = self._bg_thread.is_alive()
        tg_thread_alive = self._tg_thread.is_alive()

        with self._lock:
            cache_rows = None if self._cached_clean_df is None else int(len(self._cached_clean_df))
            data_age_seconds: int | None = None
            if self._last_fetch_utc:
                try:
                    last_fetch_dt = datetime.fromisoformat(self._last_fetch_utc)
                    if last_fetch_dt.tzinfo is None:
                        last_fetch_dt = last_fetch_dt.replace(tzinfo=timezone.utc)
                    age_seconds = (datetime.now(timezone.utc) - last_fetch_dt).total_seconds()
                    data_age_seconds = max(0, int(age_seconds))
                except ValueError:
                    data_age_seconds = None

            status = "ok"
            if (not bg_thread_alive) or (data_age_seconds is not None and data_age_seconds > 900):
                status = "unhealthy"
            elif self._last_error or data_age_seconds is None or (data_age_seconds >= 300):
                status = "degraded"

            return {
                "status": status,
                "model_fitted": self._predictor.is_fitted,
                "checkpoint_path": self._config.checkpoint_path.as_posix(),
                "checkpoint_checked": self._checkpoint_checked,
                "last_prediction_utc": self._last_prediction_utc,
                "last_error": self._last_error,
                "last_fetch_utc": self._last_fetch_utc,
                "data_age_seconds": data_age_seconds,
                "bg_thread_alive": bg_thread_alive,
                "tg_thread_alive": tg_thread_alive,
                "cache_rows": cache_rows,
            }

    def get_latest_raw_data(self) -> dict[str, Any]:
        """Returns the latest clean sensor row as a flattened dictionary.

        Notes:
            Numeric values are serialized as 2-decimal-place strings to satisfy
            the Unity legacy ``SensorData`` contract, which expects string fields.
            ``_Humid`` suffixes are remapped to ``_Hum`` for the same reason.
            If a future non-Unity client requires raw floats, introduce a separate
            endpoint rather than modifying this method.
        """
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
            response["Timestamp"] = pd.Timestamp(clean_df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

            # Map Humid back to Hum for Unity's legacy SensorData matching
            remapped = {}
            for k, v in response.items():
                if isinstance(v, (int, float)):
                    # Format strictly to 2 decimal places as a string (since Unity parses them into strings)
                    v = f"{float(v):.2f}"
                
                if k.endswith("_Humid"):
                    remapped[k.replace("_Humid", "_Hum")] = v
                else:
                    remapped[k] = v
            return remapped

    def get_latest_day_raw_data(self) -> dict[str, Any]:
        """Returns the last 288 rows for the legacy Unity charting system."""
        return self.get_historical_data_legacy(_DAY_ROWS)

    def get_latest_hour_raw_data(self) -> dict[str, Any]:
        """Returns the last 12 rows (1 hour) for the legacy Unity charting system."""
        return self.get_historical_data_legacy(_HOUR_ROWS)

    def get_latest_7_days_raw_data(self) -> dict[str, Any]:
        """Returns the last 2016 rows (7 days) for the legacy Unity charting system."""
        return self.get_historical_data_legacy(_WEEK_ROWS)

    def get_historical_data_legacy(self, row_count: int) -> dict[str, Any]:
        """Generic helper to return previous N rows in legacy Unity-compatible format."""
        with self._lock:
            try:
                clean_df = self._load_latest_clean_data()
            except (DataFetchError, DataQualityError) as exc:
                self._last_error = str(exc)
                raise ApiServiceError(str(exc)) from exc

            if clean_df.empty:
                raise ApiServiceError("Clean data is empty.")

            tail_df = clean_df.tail(row_count)

            data_list = []
            for timestamp, row in tail_df.iterrows():
                row_dict = row.to_dict()
                mapped = {}
                for k, v in row_dict.items():
                    if pd.isna(v):
                        continue
                    if isinstance(v, (int, float)):
                        v = f"{float(v):.2f}"
                    
                    if str(k).endswith("_Humid"):
                        mapped[str(k).replace("_Humid", "_Hum")] = v
                    else:
                        mapped[str(k)] = v
                mapped["created"] = pd.Timestamp(timestamp).strftime("%H:%M:%S")
                data_list.append(mapped)

            date_str = pd.Timestamp(clean_df.index[-1]).strftime("%Y-%m-%d")
            return {
                "date": date_str,
                "count": len(data_list),
                "data": data_list
            }

    def get_latest_prediction(self, module: str | None = None) -> dict[str, Any]:
        """Fetches latest data and returns one-step-ahead prediction payload.

        Args:
            module: Optional module identifier (`M1`, `M4`, `M6`...`M11`).

        Returns:
            JSON-serializable dictionary for API response.

        Raises:
            ApiServiceError: If pipeline fetch/clean/predict fails.
        """
        with self._lock:
            try:
                clean_df = self._load_latest_clean_data()
                self._ensure_model_ready(clean_df, force_retrain=False)
                prediction = self._predictor.predict_next_step(clean_df)
            except (DataFetchError, DataQualityError, PredictorError) as exc:
                self._last_error = str(exc)
                raise ApiServiceError(str(exc)) from exc

            issue_time = clean_df.index[-1]
            step_delta = self._infer_sampling_delta(clean_df.index)
            target_time = issue_time + step_delta
            current = clean_df.iloc[-1]

            selected_columns = list(clean_df.columns)
            if module is not None:
                selected_columns = [column for column in clean_df.columns if column.startswith(f"{module}_")]
                if not selected_columns:
                    raise ApiServiceError(f"Module {module} is not available in current schema.")

            response = {
                "forecast_issue_time": pd.Timestamp(issue_time).isoformat(),
                "forecast_target_time": pd.Timestamp(target_time).isoformat(),
                "module": module,
                "channels": {
                    column: {
                        "current": float(current[column]),
                        "predicted": float(prediction[column]),
                        "delta": float(prediction[column] - current[column]),
                    }
                    for column in selected_columns
                },
            }
            self._last_prediction_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._last_error = None
            return response

    def get_unity_payload(self, module: str) -> dict[str, Any]:
        """Returns compact feature-centric payload for Unity clients.

        Args:
            module: Module code such as `M1`.

        Returns:
            Dictionary with feature-keyed current/predicted/delta fields.
        """
        snapshot = self.get_latest_prediction(module=module)

        current: dict[str, float] = {}
        predicted: dict[str, float] = {}
        delta: dict[str, float] = {}
        features: list[dict[str, float | str]] = []
        for channel_name, values in snapshot["channels"].items():
            _, feature = channel_name.split("_", 1)
            current[feature] = float(values["current"])
            predicted[feature] = float(values["predicted"])
            delta[feature] = float(values["delta"])
            features.append(
                {
                    "name": feature,
                    "current": float(values["current"]),
                    "predicted": float(values["predicted"]),
                    "delta": float(values["delta"]),
                }
            )

        return {
            "module": module,
            "forecast_issue_time": snapshot["forecast_issue_time"],
            "forecast_target_time": snapshot["forecast_target_time"],
            "current": current,
            "predicted": predicted,
            "delta": delta,
            "features": features,
        }

    def get_unity_payload_all(self) -> dict[str, Any]:
        """Returns Unity payload grouped by all available modules in one response."""
        snapshot = self.get_latest_prediction(module=None)

        modules: dict[str, Any] = {}
        channels = snapshot.get("channels", {})
        for channel_name, values in channels.items():
            if "_" not in channel_name:
                continue

            module, feature = channel_name.split("_", 1)
            if module not in modules:
                modules[module] = {
                    "module": module,
                    "forecast_issue_time": snapshot["forecast_issue_time"],
                    "forecast_target_time": snapshot["forecast_target_time"],
                    "current": {},
                    "predicted": {},
                    "delta": {},
                    "features": [],
                }

            modules[module]["current"][feature] = float(values["current"])
            modules[module]["predicted"][feature] = float(values["predicted"])
            modules[module]["delta"][feature] = float(values["delta"])
            modules[module]["features"].append(
                {
                    "name": feature,
                    "current": float(values["current"]),
                    "predicted": float(values["predicted"]),
                    "delta": float(values["delta"]),
                }
            )

        module_list = [modules[key] for key in sorted(modules.keys())]

        return {
            "forecast_issue_time": snapshot["forecast_issue_time"],
            "forecast_target_time": snapshot["forecast_target_time"],
            "modules": module_list,
        }

    def retrain_now(self) -> dict[str, Any]:
        """Forces immediate retraining from latest data and saves checkpoint."""
        with self._lock:
            try:
                clean_df = self._load_latest_clean_data()
                self._ensure_model_ready(clean_df, force_retrain=True)
            except (DataFetchError, DataQualityError, PredictorError) as exc:
                self._last_error = str(exc)
                raise ApiServiceError(str(exc)) from exc

            self._last_error = None
            return {
                "status": "retrained",
                "rows_used": int(min(len(clean_df), self._config.train_tail_rows)),
                "checkpoint_path": self._config.checkpoint_path.as_posix(),
            }

    def _load_latest_clean_data(self) -> pd.DataFrame:
        """Returns the latest cached data updated by the background thread."""
        if self._cached_clean_df is None:
            raise DataFetchError("System is warming up or fetching data. Please wait.")
        return self._cached_clean_df

    def _ensure_model_ready(self, clean_df: pd.DataFrame, force_retrain: bool) -> None:
        """Ensures predictor is fitted, with optional forced retraining."""
        if not self._checkpoint_checked and not force_retrain:
            try:
                self._predictor.load_checkpoint(self._config.checkpoint_path)
            except PredictorError:
                # Expected during schema/preprocessing updates; fallback is retrain.
                pass
            finally:
                self._checkpoint_checked = True

        schema_mismatch = self._predictor.is_fitted and set(self._predictor.feature_columns) != set(clean_df.columns)

        if force_retrain or not self._predictor.is_fitted or schema_mismatch:
            train_df = clean_df.tail(max(self._config.train_tail_rows, self._predictor.config.sequence_length + 2))
            self._predictor.fit(train_df)
            self._predictor.save_checkpoint(self._config.checkpoint_path)
            self._checkpoint_checked = True

    @staticmethod
    def _infer_sampling_delta(index: pd.Index) -> pd.Timedelta:
        """Infers median positive cadence from datetime index."""
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return pd.Timedelta(minutes=3)

        diffs = index.to_series().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return pd.Timedelta(minutes=3)
        return pd.to_timedelta(diffs.median())
