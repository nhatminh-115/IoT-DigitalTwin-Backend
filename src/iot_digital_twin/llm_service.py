"""LLM service: semantic cache, Groq client, and Google Sheets context fetcher."""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_ALL_NODES = ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]

def _metric_thresholds() -> dict[str, tuple[float | None, float | None]]:
    """Lazily maps api_service thresholds into LLM metric names.

    Local import avoids circular import at module load time:
    api_service imports this module to initialize AIAssistant.
    """
    from .api_service import _SENSOR_THRESHOLDS

    thresholds: dict[str, tuple[float | None, float | None]] = {}
    for suffix, bounds in _SENSOR_THRESHOLDS.items():
        thresholds[suffix.lstrip("_")] = (bounds.lo, bounds.hi)
    return thresholds


def _build_threshold_context() -> str:
    thresholds = _metric_thresholds()
    lines = ["Safety thresholds for UEH Campus V:"]
    metric_specs = [
        ("Temp", "Temperature", "°C"),
        ("Humid", "Humidity", "%"),
        ("CO2", "CO2", "ppm"),
        ("TVOC", "TVOC", "ppb"),
    ]
    for metric_key, metric_label, unit in metric_specs:
        lo, hi = thresholds.get(metric_key, (None, None))
        if lo is not None and hi is not None:
            lines.append(f"- {metric_label}: {lo:g}–{hi:g} {unit}")
        elif lo is None and hi is not None:
            lines.append(f"- {metric_label}: <= {hi:g} {unit}")
        elif lo is not None and hi is None:
            lines.append(f"- {metric_label}: >= {lo:g} {unit}")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    threshold_context = _build_threshold_context()
    return f"""
You are an AI assistant for smart environmental monitoring at UEH Campus V (University of Economics Ho Chi Minh City).
The campus has 8 sensor nodes: M1, M4, M6, M7, M8, M9, M10, M11.
Each node measures: temperature (°C), humidity (%), CO2 (ppm), TVOC (ppb).

{threshold_context}

Response guidelines:
- Respond in English, concise, and professional.
- Always use the most recent data available in the context. Specify the timestamp of the latest reading.
- Only say "no data" if the DATA section below is actually empty.
- Do not fabricate numbers.
- Provide a brief assessment of safety levels based on standard thresholds.
- Prioritize reading the SUMMARY section for quick responses; use the DATA TABLE to cite specific figures.
""".strip()

# Frozen detection: 2 h = 40 rows at 3-min cadence.
_FROZEN_ROWS = 40


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Dictionary-based cache with fuzzy string matching and TTL."""

    def __init__(self, ttl_seconds: float = 300.0, similarity_threshold: int = 80) -> None:
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._store: dict[str, tuple[str, float]] = {}

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def get(self, query: str) -> tuple[str | None, bool]:
        try:
            from thefuzz import fuzz
        except ImportError:
            return None, False

        now = time.monotonic()
        normalized = self._normalize(query)
        best_score = 0
        best_key: str | None = None

        for key in list(self._store.keys()):
            _, expire = self._store[key]
            if now > expire:
                del self._store[key]
                continue
            score = fuzz.partial_ratio(normalized, key)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and best_score >= self._threshold:
            answer, _ = self._store[best_key]
            logger.debug("Cache hit (score=%d): %s", best_score, query[:60])
            return answer, True
        return None, False

    def set(self, query: str, answer: str) -> None:
        key = self._normalize(query)
        self._store[key] = (answer, time.monotonic() + self._ttl)


# ---------------------------------------------------------------------------
# Google Sheets context fetcher
# ---------------------------------------------------------------------------

class SheetContextFetcher:
    """Slices the cached clean DataFrame and formats it for LLM consumption.

    Produces a two-part context block:
      1. SUMMARY — pre-computed facts (active nodes, frozen sensors, alerts).
         LLM reads this to answer status questions without parsing the table.
      2. DATA TABLE — raw readings for the requested time window.
    """

    # 3-min cadence: 10=30min, 120=6h, 480=24h, 5=15min for status table.
    _WINDOW_ROWS = {
        "module_status": 5,    # ~15 min table (active/silent computed time-based)
        "current":       10,   # ~30 min
        "trend":         120,  # ~6 h
        "today":         480,  # ~24 h
    }
    _MAX_ROWS_TO_LLM = 20

    def __init__(self, get_df: Callable[[], pd.DataFrame | None]) -> None:
        self._get_df = get_df

    # ------------------------------------------------------------------
    # Intent & node detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        status_kw = ("dead", "offline", "disconnected", "not working", "alive",
                     "which module", "which node", "status", "state")
        trend_kw  = ("trend", "fluctuation", "change", "history",
                     "this morning", "this afternoon", "this evening", "6h", "past hour", "hours")
        today_kw  = ("today", "all day", "24h")
        if any(k in q for k in status_kw):
            return "module_status"
        if any(k in q for k in trend_kw):
            return "trend"
        if any(k in q for k in today_kw):
            return "today"
        return "current"

    @staticmethod
    def _extract_node(query: str) -> str | None:
        match = re.search(r"\bM(10|11|1|4|6|7|8|9)\b", query, re.IGNORECASE)
        return match.group(0).upper() if match else None

    # ------------------------------------------------------------------
    # Pre-computed analysis (Python side — zero LLM tokens)
    # ------------------------------------------------------------------

    @staticmethod
    def _active_nodes(df: pd.DataFrame, window_minutes: int = 60) -> tuple[list[str], list[str]]:
        """Returns (active, silent) based on whether a node reported in the last window_minutes (default 60 min)."""
        cutoff = df.index[-1] - pd.Timedelta(minutes=window_minutes)
        recent = df[df.index >= cutoff]
        active, silent = [], []
        for node in _ALL_NODES:
            cols = [c for c in recent.columns if c.startswith(f"{node}_")]
            has_data = any(recent[c].notna().any() for c in cols)
            (active if has_data else silent).append(node)
        return active, silent



    @staticmethod
    def _frozen_sensors(df: pd.DataFrame, min_rows: int = _FROZEN_ROWS) -> list[str]:
        """Returns columns whose value has been unchanged for >= min_rows readings."""
        tail = df.tail(min_rows)
        if len(tail) < min_rows:
            return []
        frozen = []
        for col in tail.columns:
            series = tail[col].dropna()
            if len(series) >= min_rows and series.nunique() == 1:
                frozen.append(col)
        return frozen

    @staticmethod
    def _threshold_alerts(df: pd.DataFrame) -> list[str]:
        """Returns alert strings for the latest row that breach known thresholds."""
        latest = df.iloc[-1]
        alerts = []
        thresholds = _metric_thresholds()
        for col in latest.index:
            col_str = str(col)
            metric = next((m for m in thresholds if col_str.endswith(f"_{m}")), None)
            if metric is None:
                continue
            val = latest[col_str]
            if pd.isna(val):
                continue
            lo, hi = thresholds[metric]
            if hi is not None and float(val) > hi:
                alerts.append(f"{col_str}={val:.1f} [HIGH]")
            elif lo is not None and float(val) < lo:
                alerts.append(f"{col_str}={val:.1f} [LOW]")
        return alerts

    def _build_summary(self, df: pd.DataFrame) -> str:
        try:
            latest_ts = pd.Timestamp(df.index[-1]).strftime("%d/%m/%Y %H:%M ICT")
        except Exception:
            latest_ts = "unknown"

        active, silent  = self._active_nodes(df)
        frozen          = self._frozen_sensors(df)
        alerts          = self._threshold_alerts(df)

        lines = [
            "=== SUMMARY ===",
            f"Latest data    : {latest_ts}",
            f"Active nodes   : {', '.join(active) if active else 'none'} ({len(active)}/8)",
            f"Silent nodes   : {', '.join(silent) if silent else 'none'}",
            f"Frozen sensors : {', '.join(frozen) if frozen else 'none'} (unchanged >=2h / 40 readings)",
            f"Alerts         : {', '.join(alerts) if alerts else 'none'}",
            "================",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data table formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(val: object, decimals: int = 1) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "--"
        return f"{float(val):.{decimals}f}"

    def _format_table(self, df: pd.DataFrame, label: str, node: str | None) -> str:
        if df.empty:
            return f"{label}: no data."

        nodes = [node] if node else _ALL_NODES
        tail  = df.tail(self._MAX_ROWS_TO_LLM)

        lines = [f"=== DATA TABLE: {label} ({len(df)} rows, showing last {len(tail)}) ==="]
        lines.append(f"{'Timestamp':<16} {'Node':<5} {'Temp':>6} {'Hum':>6} {'CO2':>6} {'TVOC':>6}")
        lines.append("-" * 54)

        for ts, row in tail.iterrows():
            try:
                ts_str = pd.Timestamp(ts).strftime("%d/%m %H:%M")
            except Exception:
                ts_str = str(ts)[:15]

            for n in nodes:
                temp = self._fmt(row.get(f"{n}_Temp"))
                hum  = self._fmt(row.get(f"{n}_Humid"))
                co2  = self._fmt(row.get(f"{n}_CO2"), 0)
                tvoc = self._fmt(row.get(f"{n}_TVOC"))
                if temp == "--" and hum == "--" and co2 == "--" and tvoc == "--":
                    continue
                lines.append(f"{ts_str:<16} {n:<5} {temp:>6} {hum:>6} {co2:>6} {tvoc:>6}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_context(self, query: str) -> str:
        df = self._get_df()
        if df is None or df.empty:
            return "No sensor data is available. The system is starting up."

        intent = self._detect_intent(query)
        node   = self._extract_node(query)
        rows   = self._WINDOW_ROWS[intent]
        label  = {
            "module_status": "Node status check",
            "current":       "Latest readings (~30 min)",
            "trend":         "Recent trend (~6 h)",
            "today":         "Today's readings (~24 h)",
        }[intent]

        summary   = self._build_summary(df)

        window_df = df.tail(rows)
        if node:
            node_cols = [c for c in window_df.columns if c.startswith(f"{node}_")]
            window_df = window_df[node_cols] if node_cols else window_df

        table = self._format_table(window_df, label, node)

        return f"{summary}\n\n{table}"


# ---------------------------------------------------------------------------
# Groq LLM client
# ---------------------------------------------------------------------------

class GroqClient:
    _MODEL = "llama-3.3-70b-versatile"
    _MAX_TOKENS = 512
    _TEMPERATURE = 0.3

    def __init__(self, api_key: str) -> None:
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError as exc:
            raise ImportError("groq package is required: pip install groq>=0.9.0") from exc

    def ask(self, user_query: str, context: str) -> dict:
        user_message = f"Sensor data:\n\n{context}\n\nQuestion: {user_query}"
        system_prompt = _build_system_prompt()
        t0 = time.monotonic()
        completion = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=self._MAX_TOKENS,
            temperature=self._TEMPERATURE,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        usage = completion.usage
        return {
            "answer": completion.choices[0].message.content or "(no response)",
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "model": self._MODEL,
            "latency_ms": latency_ms,
        }


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class AIAssistant:
    """Wires cache + sheet fetcher + LLM into a single answer() call."""

    def __init__(self, groq_api_key: str, get_df: Callable[[], pd.DataFrame | None]) -> None:
        self._cache   = SemanticCache(ttl_seconds=300.0, similarity_threshold=80)
        self._fetcher = SheetContextFetcher(get_df)
        self._llm     = GroqClient(api_key=groq_api_key)

    def answer(self, query: str) -> dict:
        """Returns dict with answer text and LLM metadata for logging."""
        cached_text, is_cached = self._cache.get(query)
        if is_cached and cached_text is not None:
            return {
                "answer": cached_text,
                "input_tokens": 0,
                "output_tokens": 0,
                "model": GroqClient._MODEL,
                "latency_ms": 0,
                "is_cached": True,
            }
        context = self._fetcher.build_context(query)
        result  = self._llm.ask(query, context)
        self._cache.set(query, result["answer"])
        result["is_cached"] = False
        return result
