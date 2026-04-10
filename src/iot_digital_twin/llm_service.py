"""LLM service: semantic cache, Groq client, and Google Sheets context fetcher."""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_THRESHOLD_CONTEXT = """
Safety thresholds for UEH Campus V:
- Temperature: 18–33 °C (below 18 = too cold, above 33 = too hot)
- Humidity: 30–75 % (below 30 = severely dry, above 75 = hyper-humid)
- CO2: safe below 1000 ppm, concerning 1000–1200 ppm, toxic above 1200 ppm
- TVOC: safe below 150 ppb, concerning 150–300 ppb, hazardous above 300 ppb
""".strip()

_SYSTEM_PROMPT = f"""
Bạn là trợ lý AI giám sát môi trường thông minh cho UEH Campus V (Đại học Kinh tế TP.HCM).
Campus có 8 nodes cảm biến: M1, M4, M6, M7, M8, M9, M10, M11.
Mỗi node đo: nhiệt độ (°C), độ ẩm (%), CO2 (ppm), TVOC (ppb).

{_THRESHOLD_CONTEXT}

Quy tắc trả lời:
- Trả lời bằng tiếng Việt, ngắn gọn, chuyên nghiệp.
- Luôn trích dẫn số liệu cụ thể từ dữ liệu được cung cấp.
- Nếu không có dữ liệu liên quan, nói rõ "Không có dữ liệu cho thời điểm này."
- Không bịa đặt số liệu.
- Đưa ra nhận xét ngắn về mức độ an toàn dựa trên ngưỡng chuẩn.
""".strip()


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

    def get(self, query: str) -> str | None:
        try:
            from thefuzz import fuzz
        except ImportError:
            return None

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
            return answer
        return None

    def set(self, query: str, answer: str) -> None:
        key = self._normalize(query)
        self._store[key] = (answer, time.monotonic() + self._ttl)


# ---------------------------------------------------------------------------
# Google Sheets context fetcher — operates on the in-memory DataFrame
# ---------------------------------------------------------------------------

class SheetContextFetcher:
    """Slices the cached clean DataFrame and formats it for LLM consumption.

    The DataFrame is obtained via a callable so it always reflects the latest
    background fetch without holding a direct reference to the service.
    Columns follow the canonical schema: M1_Temp, M1_Humid, M1_CO2, M1_TVOC, …
    Index is a DatetimeIndex (UTC or naive — handled transparently).
    """

    # Approximate rows per time window (5-min sampling cadence).
    _WINDOW_ROWS = {
        "current": 6,    # ~30 min
        "trend":   72,   # ~6 h
        "today":   288,  # ~24 h
    }
    _MAX_ROWS_TO_LLM = 20  # cap to save tokens

    def __init__(self, get_df: Callable[[], pd.DataFrame | None]) -> None:
        self._get_df = get_df

    # ------------------------------------------------------------------
    # Intent & node detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        trend_kw = ("xu hướng", "trend", "biến động", "thay đổi", "lịch sử",
                    "sáng nay", "chiều nay", "tối nay", "6h", "giờ qua", "tiếng")
        today_kw = ("hôm nay", "today", "ngày hôm nay", "cả ngày", "24h")
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
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(val: object, decimals: int = 1) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "--"
        return f"{float(val):.{decimals}f}"

    def _format_df(self, df: pd.DataFrame, label: str, node: str | None) -> str:
        if df.empty:
            return f"{label}: Không có dữ liệu."

        nodes = [node] if node else ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]
        # Only keep rows from the tail (cap tokens)
        tail = df.tail(self._MAX_ROWS_TO_LLM)

        lines = [f"{label} ({len(df)} readings, showing last {len(tail)}):"]
        lines.append(f"{'Timestamp (ICT)':<18} {'Node':<5} {'Temp':>6} {'Hum':>6} {'CO2':>6} {'TVOC':>6}")
        lines.append("-" * 56)

        for ts, row in tail.iterrows():
            try:
                ts_dt = pd.Timestamp(ts)
                # HomeAssistant writes ICT timestamps directly — no conversion needed.
                ts_str = ts_dt.strftime("%d/%m %H:%M")
            except Exception:
                ts_str = str(ts)[:16]

            for n in nodes:
                temp = self._fmt(row.get(f"{n}_Temp"))
                hum  = self._fmt(row.get(f"{n}_Humid"))
                co2  = self._fmt(row.get(f"{n}_CO2"), 0)
                tvoc = self._fmt(row.get(f"{n}_TVOC"))
                if temp == "--" and hum == "--" and co2 == "--" and tvoc == "--":
                    continue
                lines.append(f"{ts_str:<18} {n:<5} {temp:>6} {hum:>6} {co2:>6} {tvoc:>6}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_context(self, query: str) -> str:
        df = self._get_df()
        if df is None or df.empty:
            return "Không có dữ liệu cảm biến. Hệ thống đang khởi động."

        intent = self._detect_intent(query)
        node   = self._extract_node(query)
        rows   = self._WINDOW_ROWS[intent]
        label  = {
            "current": "Latest readings (~30 min)",
            "trend":   "Recent trend (~6 h)",
            "today":   "Today's readings (~24 h)",
        }[intent]

        window_df = df.tail(rows)

        # If a specific node was requested, drop columns for other nodes.
        if node:
            node_cols = [c for c in window_df.columns if c.startswith(f"{node}_")]
            window_df = window_df[node_cols] if node_cols else window_df

        return self._format_df(window_df, label, node)


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

    def ask(self, user_query: str, context: str) -> str:
        user_message = f"Dữ liệu cảm biến:\n\n{context}\n\nCâu hỏi: {user_query}"
        completion = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=self._MAX_TOKENS,
            temperature=self._TEMPERATURE,
        )
        return completion.choices[0].message.content or "(no response)"


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class AIAssistant:
    """Wires cache + sheet fetcher + LLM into a single answer() call."""

    def __init__(self, groq_api_key: str, get_df: Callable[[], pd.DataFrame | None]) -> None:
        self._cache   = SemanticCache(ttl_seconds=300.0, similarity_threshold=80)
        self._fetcher = SheetContextFetcher(get_df)
        self._llm     = GroqClient(api_key=groq_api_key)

    def answer(self, query: str) -> str:
        cached = self._cache.get(query)
        if cached:
            return cached
        context = self._fetcher.build_context(query)
        reply   = self._llm.ask(query, context)
        self._cache.set(query, reply)
        return reply
