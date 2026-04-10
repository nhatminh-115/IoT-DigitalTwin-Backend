"""FastAPI-facing service layer for real-time IoT model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

# Number of 5-minute readings that constitute one day (288 = 24 h × 12 readings/h).
_DAY_ROWS: int = 288
_HOUR_ROWS: int = 12
_WEEK_ROWS: int = 2016

# Cooldown between repeated alerts for the same sensor channel (seconds).
_ALERT_COOLDOWN_SECONDS: float = 3600.0

# Interval between routine status reports (seconds).
_ROUTINE_REPORT_INTERVAL_SECONDS: float = 900.0


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
    train_tail_rows: int = 720


class InferenceAPIService:
    """Provides thread-safe inference and retraining methods for REST endpoints."""

    def __init__(self, config: ApiServiceConfig) -> None:
        """Initializes service dependencies and state."""
        self._config = config
        self._fetcher = DataFetcher(DataFetcherConfig(csv_url=config.csv_url))
        self._quality_gate = DataQualityGate(zscore_threshold=3.0)
        self._predictor = DeepTimeSeriesPredictor(config=PredictorConfig())

        # Telegram Bot configuration
        self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        # All mutable alert state is protected by _lock.
        self._alert_cooldowns: dict[str, float] = {}
        self._last_routine_report: float = time.monotonic()

        self._checkpoint_checked = False
        self._last_error: str | None = None
        self._last_prediction_utc: str | None = None
        self._lock = threading.RLock()

        self._cached_clean_df: pd.DataFrame | None = None

        # Signals background threads to stop gracefully on shutdown.
        self._stop_event = threading.Event()

        # Becomes set once valid credentials are detected; avoids busy-wait in polling loop.
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

    def _background_fetch_loop(self) -> None:
        """Continuously fetches and caches clean data; pre-warms the predictor.

        Runs on a daemon thread. Exits cleanly when ``_stop_event`` is set.
        """
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

                # Alert evaluation does not need the global lock; it has its own
                # internal locking for the cooldown dict.
                self._check_and_send_alerts(clean_df)

            except Exception as exc:
                with self._lock:
                    self._last_error = f"Background Fetch Error: {exc}"

            # Throttle to prevent rate-limits from Google Sheets.
            self._stop_event.wait(timeout=30.0)

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
            print(f"Telegram send error: {exc}")

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
        """Evaluates thresholds and dispatches Telegram alerts with per-channel cooldown.

        Alert cooldown state is read and written under ``_lock`` to prevent
        races between the background fetch thread and any future callers.
        """
        if clean_df.empty or not self._bot_token:
            return

        latest = clean_df.iloc[-1]
        now = time.monotonic()
        breaches = self._evaluate_sensor_alerts(latest)

        alert_lines: list[str] = []
        with self._lock:
            for col, val, reason in breaches:
                last_sent = self._alert_cooldowns.get(col, 0.0)
                if now - last_sent > _ALERT_COOLDOWN_SECONDS:
                    alert_lines.append(f"⚠️ <b>{col}</b>: {val:.1f} <i>({reason})</i>")
                    self._alert_cooldowns[col] = now

            should_report = (now - self._last_routine_report) > _ROUTINE_REPORT_INTERVAL_SECONDS
            if should_report:
                self._last_routine_report = now

        if alert_lines:
            body = "\n".join(alert_lines)
            msg = (
                "🔴 <b>[CRITICAL ALERT] IoT Facility Monitor</b>\n"
                "──────────────────────────────\n"
                f"{body}\n\n"
                "<i>Status: Immediate inspection requested.</i>"
            )
            self._send_telegram_message(msg)

        if should_report:
            temps = [float(latest[c]) for c in clean_df.columns if str(c).endswith("_Temp")]
            avg_temp = sum(temps) / len(temps) if temps else 0.0
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            msg = (
                "✅ <b>[SYSTEM STATUS REPORT]</b>\n"
                f"Time: {ts}\n"
                "──────────────────────────────\n"
                "Overall Status: <b>NORMAL</b>\n\n"
                f"🌡️ Avg Temperature: <b>{avg_temp:.1f} °C</b>\n\n"
                "<i>Active Modules: Live</i>"
            )
            self._send_telegram_message(msg)

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

                        # Support DMs/groups ('message') and broadcast channels ('channel_post').
                        msg_node = update.get("message") or update.get("channel_post")
                        if not (msg_node and "text" in msg_node):
                            continue

                        text = msg_node["text"].lower()
                        chat_id = str(msg_node["chat"]["id"])

                        if text.startswith("/getcurrent_detail"):
                            self._handle_telegram_command("detail", target_chat_id=chat_id)
                        elif text.startswith("/getcurrent_short"):
                            self._handle_telegram_command("short", target_chat_id=chat_id)
                        elif text.startswith("/getcurrent_alert"):
                            self._handle_telegram_command("alert", target_chat_id=chat_id)

            except Exception as exc:
                print(f"Telegram polling error: {exc}")
                self._stop_event.wait(timeout=5.0)

    def _handle_telegram_command(self, cmd_type: str, target_chat_id: str) -> None:
        """Handles an inbound Telegram command and sends the appropriate reply.

        Args:
            cmd_type: One of ``"detail"``, ``"short"``, or ``"alert"``.
            target_chat_id: Destination chat ID for the reply message.
        """
        with self._lock:
            df = self._cached_clean_df

        if df is None or df.empty:
            self._send_telegram_message(
                "⏳ <i>System is currently warming up. Please try again later.</i>",
                target_chat_id,
            )
            return

        latest = df.iloc[-1]
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # Reuse the shared threshold evaluator — no duplicated logic.
        breaches = self._evaluate_sensor_alerts(latest)
        anomaly_lines = [
            f"⚠️ <b>{col}</b>: {val:.1f} <i>({reason})</i>"
            for col, val, reason in breaches
        ]

        if cmd_type == "short":
            temps = [float(latest[c]) for c in df.columns if str(c).endswith("_Temp")]
            hums  = [float(latest[c]) for c in df.columns if str(c).endswith("_Humid")]
            avg_t = sum(temps) / len(temps) if temps else 0.0
            avg_h = sum(hums) / len(hums) if hums else 0.0
            body = (
                f"📊 <b>[QUICK SUMMARY]</b>\nTime: {timestamp}\n"
                "──────────────────────────────\n"
                f"🌡️ Avg Temp: <b>{avg_t:.1f} °C</b>\n"
                f"💧 Avg Humidity: <b>{avg_h:.1f} %</b>\n\n"
                f"<b>Anomalies Detected: {len(anomaly_lines)}</b>"
            )
            if anomaly_lines:
                body += "\n" + "\n".join(anomaly_lines)
            self._send_telegram_message(body, target_chat_id)

        elif cmd_type == "alert":
            if not anomaly_lines:
                self._send_telegram_message(
                    "✅ <b>[ALERT STATUS]</b>\n"
                    "──────────────────────────────\n"
                    "Zero active anomalies across all tracked dimensions. System optimal.",
                    target_chat_id,
                )
            else:
                msg = (
                    f"🔴 <b>[ACTIVE ANOMALIES]</b>\nTime: {timestamp}\n"
                    "──────────────────────────────\n"
                    + "\n".join(anomaly_lines)
                )
                self._send_telegram_message(msg, target_chat_id)

        elif cmd_type == "detail":
            detail_lines = [
                f"▪️ <b>{col}</b>: {float(latest[col]):.1f}"
                for col in df.columns
                if "_" in str(col)
            ]
            msg = (
                f"📋 <b>[DETAILED DIAGNOSTIC LOG]</b>\nTime: {timestamp}\n"
                "──────────────────────────────\n"
                + "\n".join(detail_lines)
            )
            self._send_telegram_message(msg, target_chat_id)

    def health(self) -> dict[str, Any]:
        """Returns runtime health information for service monitoring."""
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
            response["ThoiGian"] = pd.Timestamp(clean_df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

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
