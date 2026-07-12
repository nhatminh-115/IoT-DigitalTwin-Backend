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
        """Maps row-by-device data to canonical wide schema via pivoting."""
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

        # Find key columns case-insensitively
        header_lower = [col.lower() for col in dataframe.columns]
        def get_col_name(names):
            for name in names:
                if name.lower() in header_lower:
                    return dataframe.columns[header_lower.index(name.lower())]
            return None

        col_device = get_col_name(['device id', 'device', 'node id', 'node'])
        col_temp = get_col_name(['temp', 'temperature', 'nhiệt độ'])
        col_hum = get_col_name(['humidity', 'hum', 'độ ẩm'])
        col_co2 = get_col_name(['eco2', 'co2'])
        col_tvoc = get_col_name(['tvoc'])
        col_pm1 = get_col_name(['pm 1.0', 'pm1.0', 'pm 1', 'pm1'])
        col_pm25 = get_col_name(['pm 2.5', 'pm2.5', 'pm 25', 'pm25'])
        col_pm10 = get_col_name(['pm 10', 'pm10'])

        if not col_device:
            raise DataFetchError("Device/Node ID column is missing.")

        # Rename columns to standardized keys
        renamed = {col_device: 'Device'}
        metrics_found = []
        
        if col_temp: renamed[col_temp] = 'Temp'; metrics_found.append('Temp')
        if col_hum: renamed[col_hum] = 'Hum'; metrics_found.append('Hum')
        if col_co2: renamed[col_co2] = 'CO2'; metrics_found.append('CO2')
        if col_tvoc: renamed[col_tvoc] = 'TVOC'; metrics_found.append('TVOC')
        if col_pm1: renamed[col_pm1] = 'PM1'; metrics_found.append('PM1')
        if col_pm25: renamed[col_pm25] = 'PM25'; metrics_found.append('PM25')
        if col_pm10: renamed[col_pm10] = 'PM10'; metrics_found.append('PM10')

        dataframe = dataframe.rename(columns=renamed)
        dataframe = dataframe[['Device'] + metrics_found]

        # Normalize numeric metrics
        for m in metrics_found:
            dataframe[m] = dataframe[m].astype(str).str.strip().str.replace(',', '.', regex=False)
            dataframe[m] = pd.to_numeric(dataframe[m], errors='coerce')

        # Normalize Device IDs (e.g. 'esp09' -> 'M9')
        def norm_device(raw):
            if not isinstance(raw, str):
                raw = str(raw)
            s = raw.strip().lower()
            digits = "".join([c for c in s if c.isdigit()])
            if digits:
                return f"M{int(digits)}"
            return raw.upper()

        dataframe['Device'] = dataframe['Device'].apply(norm_device)
        dataframe = dataframe[dataframe['Device'].str.startswith('M')]
        
        # Keep only M1 to M16
        dataframe['DeviceNum'] = dataframe['Device'].str[1:].astype(int)
        dataframe = dataframe[(dataframe['DeviceNum'] >= 1) & (dataframe['DeviceNum'] <= 16)]

        # Pivot to wide format: Index is timestamp, Columns are Device
        pivoted = dataframe.pivot_table(
            index=timestamp_col,
            columns='Device',
            values=metrics_found,
            aggfunc='last'
        )

        if pivoted.empty:
            raise DataFetchError("No valid rows remained after pivoting data.")

        # Flatten multiindex columns: (Metric, Device) -> "Device_Metric"
        # Standardize "Hum" to "Humid" to keep compatibility with ML code
        flat_cols = []
        for metric, device in pivoted.columns:
            feat_name = metric
            if metric == 'Hum':
                feat_name = 'Humid'
            flat_cols.append(f"{device}_{feat_name}")
        pivoted.columns = flat_cols

        # Resample to 3-minute intervals to align timestamps and downsample
        pivoted = pivoted.resample('3min').mean()
        
        # Interpolate missing values
        pivoted = pivoted.interpolate(method='linear', limit_direction='both')
        pivoted = pivoted.ffill().bfill()
        
        if pivoted.empty or pivoted.shape[1] == 0:
            raise DataFetchError("Pivoted DataFrame is empty.")

        pivoted.index.name = "Timestamp"
        return pivoted

    @staticmethod
    def _find_timestamp_column(dataframe: pd.DataFrame) -> str | None:
        """Finds the best timestamp column by parseability and naming heuristics."""
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
