"""Data fetching utilities for Google Sheet CSV ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import io
import re
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class DataFetchError(RuntimeError):
    """Raised when remote CSV data cannot be fetched or parsed."""


@dataclass(frozen=True)
class DataFetcherConfig:
    """Configuration for fetching remote time-series data.

    Attributes:
        csv_url: Public Google Sheet CSV URL.
        timeout_sec: Request timeout in seconds.
        max_retries: Total retry attempts for transient network issues.
        backoff_factor: Exponential backoff factor between retries.
    """

    csv_url: str
    timeout_sec: int = 12
    max_retries: int = 3
    backoff_factor: float = 0.8


class DataFetcher:
    """Fetches and normalizes multivariate IoT data from a CSV endpoint.

    The class performs robust HTTP access with retries and converts heterogeneous
    spreadsheet column naming into canonical sensor names such as `M1_Temp`.
    """

    _MODULES = {"M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"}
    _FEATURES = {"TEMP", "HUMID", "HUM", "CO2", "TVOC"}

    def __init__(self, config: DataFetcherConfig) -> None:
        """Initializes the fetcher and underlying HTTP session.

        Args:
            config: Runtime configuration for URL and retry behavior.
        """
        self._config = config
        self._session = requests.Session()
        retries = Retry(
            total=config.max_retries,
            read=config.max_retries,
            connect=config.max_retries,
            backoff_factor=config.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def fetch(self) -> pd.DataFrame:
        """Fetches, parses, and normalizes the latest dataset.

        Returns:
            A DataFrame indexed by parsed timestamp.

        Raises:
            DataFetchError: If network retrieval or CSV parsing fails.
        """
        try:
            response = self._session.get(self._config.csv_url, timeout=self._config.timeout_sec)
            if response.status_code != 200:
                raise DataFetchError(
                    f"Data source returned HTTP {response.status_code}."
                )
        except requests.RequestException as exc:
            raise DataFetchError("Unable to reach Google Sheet CSV endpoint.") from exc

        try:
            dataframe = pd.read_csv(io.StringIO(response.text), low_memory=False)
        except Exception as exc:
            raise DataFetchError("Fetched payload is not a valid CSV document.") from exc

        if dataframe.empty:
            raise DataFetchError("The remote CSV payload is empty.")

        return self._normalize_schema(dataframe)

    def _normalize_schema(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Maps source columns to canonical schema.

        Args:
            dataframe: Raw DataFrame from CSV.

        Returns:
            Canonicalized DataFrame with numeric sensor columns.

        Raises:
            DataFetchError: If no valid sensor channels are found.
        """
        dataframe = dataframe.copy()
        dataframe.columns = [str(column).strip() for column in dataframe.columns]

        timestamp_col = self._find_timestamp_column(dataframe)
        if timestamp_col is None:
            raise DataFetchError(
                "Timestamp column is missing. A valid datetime column is required "
                "for seasonal alignment and time-axis plotting."
            )

        dataframe[timestamp_col] = pd.to_datetime(dataframe[timestamp_col], errors="coerce")
        dataframe = dataframe.dropna(subset=[timestamp_col])
        if dataframe.empty:
            raise DataFetchError("All timestamp values are invalid after datetime parsing.")

        dataframe = dataframe.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="last")
        dataframe = dataframe.set_index(timestamp_col)
        dataframe.index.name = "Timestamp"

        renamed = {}
        for column in dataframe.columns:
            canonical = self._canonical_sensor_name(column)
            if canonical is not None:
                renamed[column] = canonical

        if not renamed:
            raise DataFetchError(
                "No valid sensor channels were found. Expected module-feature columns "
                "for M1, M4, M6-M11 with Temp, Humid, CO2, TVOC."
            )

        dataframe = dataframe.rename(columns=renamed)
        dataframe = dataframe[list(renamed.values())]
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()].copy()

        for column in dataframe.columns:
            dataframe[column] = self._to_numeric_series(dataframe[column])

        if dataframe.empty:
            raise DataFetchError("No rows remained after schema normalization.")

        return dataframe

    @staticmethod
    def _find_timestamp_column(dataframe: pd.DataFrame) -> str | None:
        """Finds the best timestamp column by parseability and naming heuristics.

        The method first filters columns whose names semantically suggest time,
        then selects the candidate with the highest datetime parse success ratio.
        """
        columns = [str(column).strip() for column in dataframe.columns]
        lower_cols = [column.lower() for column in columns]

        semantic_keywords = ("timestamp", "datetime", "date", "time", "created", "thoigian")
        candidate_cols = [
            columns[idx]
            for idx, lowered in enumerate(lower_cols)
            if any(keyword in lowered for keyword in semantic_keywords)
        ]

        if not candidate_cols:
            return None

        best_col: str | None = None
        best_score = -1.0
        for column in candidate_cols:
            parsed = pd.to_datetime(dataframe[column], errors="coerce")
            score = float(parsed.notna().mean())
            if score > best_score:
                best_score = score
                best_col = column

        if best_score <= 0.0:
            return None
        return best_col

    @staticmethod
    def _to_numeric_series(series: pd.Series) -> pd.Series:
        """Converts mixed-format sensor values into numeric dtype.

        Handles locale decimal commas (e.g., "28,7") and string sentinels such as
        "unknown" by coercing invalid values to NaN.
        """
        normalized = (
            series.astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .replace({"unknown": pd.NA, "Unknown": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        return pd.to_numeric(normalized, errors="coerce")

    def _canonical_sensor_name(self, raw_name: str) -> str | None:
        """Converts heterogeneous labels into canonical `M#_Feature` names."""
        tokenized = re.sub(r"[^A-Za-z0-9]", "", str(raw_name).upper())

        module_match = re.search(r"M(?:10|11|1|4|6|7|8|9)", tokenized)
        feature_match = re.search(r"TEMP|HUMID|HUM|CO2|TVOC", tokenized)

        if module_match is None or feature_match is None:
            return None

        module = module_match.group(0)
        feature = feature_match.group(0)

        if module not in self._MODULES or feature not in self._FEATURES:
            return None

        if feature in {"CO2", "TVOC"}:
            feature_name = feature
        elif feature in {"HUM", "HUMID"}:
            feature_name = "Humid"
        else:
            feature_name = feature.title()
        return f"{module}_{feature_name}"
