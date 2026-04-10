"""Robust data quality controls for unstable IoT sensor streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when data quality checks fail critically."""


@dataclass
class DataQualityReport:
    """Summarizes actions taken by data preprocessing.

    Attributes:
        input_rows: Number of records received.
        output_rows: Number of records after quality controls.
        flatline_columns: Columns identified as flatlined sensors.
        outlier_counts: Number of values removed via z-score per column.
        dropped_columns: Columns dropped due to irrecoverable corruption.
        total_missing_after_cleaning: Remaining missing values after interpolation.
    """

    input_rows: int
    output_rows: int
    flatline_columns: List[str]
    outlier_counts: Dict[str, int]
    dropped_columns: List[str]
    total_missing_after_cleaning: int


class DataQualityGate:
    """Executes strict data validation and repair in a deterministic pipeline.

    Preprocessing math:
    1. Micro-flatline detection uses rolling variance over a window W.
       For each channel x_t, compute
           Var_t = Var(x_{t-W+1}, ..., x_t).
       If Var_t < epsilon, the point is flagged as flatline-like and set to NaN,
       then recovered by interpolation.
    2. Outlier removal uses z-score thresholding:
       z = (x - mean) / std
       Values with |z| > threshold are treated as spikes and set to NaN.
    3. Missing values are repaired with linear interpolation over time.

    This sequence removes anomalous points before interpolation, reducing bias.
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        flatline_variance_epsilon: float = 1e-4,
        flatline_window_size: int = 20,  # ~60 min at 3-min cadence
        min_required_rows: int = 40,     # ~2 h at 3-min cadence
    ) -> None:
        """Initializes gate thresholds.

        Args:
            zscore_threshold: Absolute z-score cutoff for spike rejection.
            flatline_variance_epsilon: Rolling variance epsilon for micro-flatline detection.
            flatline_window_size: Rolling window length used for variance estimation.
            min_required_rows: Minimum sample count needed for stable inference.
        """
        self._zscore_threshold = zscore_threshold
        self._flatline_variance_epsilon = flatline_variance_epsilon
        self._flatline_window_size = max(3, int(flatline_window_size))
        self._min_required_rows = min_required_rows

    def process(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
        """Validates and cleans a multivariate time-series DataFrame.

        Args:
            dataframe: Raw or normalized multivariate sensor matrix.

        Returns:
            A tuple containing the cleaned DataFrame and a processing report.

        Raises:
            DataQualityError: If data is insufficient or fully corrupted.
        """
        if dataframe is None or dataframe.empty:
            raise DataQualityError("Input DataFrame is empty.")

        numeric_df = dataframe.copy()
        for column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

        if len(numeric_df) < self._min_required_rows:
            raise DataQualityError(
                f"Insufficient rows for inference. Required at least {self._min_required_rows}, "
                f"received {len(numeric_df)}."
            )

        flatline_columns, flatline_masks = self._detect_flatlines(numeric_df)
        for column, mask in flatline_masks.items():
            if mask.any():
                numeric_df.loc[mask, column] = np.nan

        numeric_df, outlier_counts = self._remove_outliers_zscore(numeric_df)
        numeric_df = numeric_df.interpolate(method="linear", limit_direction="both")

        dropped_columns = []
        fully_missing = [column for column in numeric_df.columns if numeric_df[column].isna().all()]
        if fully_missing:
            numeric_df = numeric_df.drop(columns=fully_missing)
            dropped_columns.extend(fully_missing)

        if numeric_df.empty or numeric_df.shape[1] == 0:
            raise DataQualityError("All channels were corrupted after quality controls.")

        total_missing = int(numeric_df.isna().sum().sum())
        if total_missing > 0:
            numeric_df = numeric_df.ffill().bfill()
            total_missing = int(numeric_df.isna().sum().sum())

        if total_missing > 0:
            raise DataQualityError("Residual missing values remain after interpolation.")

        report = DataQualityReport(
            input_rows=len(dataframe),
            output_rows=len(numeric_df),
            flatline_columns=flatline_columns,
            outlier_counts=outlier_counts,
            dropped_columns=dropped_columns,
            total_missing_after_cleaning=total_missing,
        )
        return numeric_df, report

    def _detect_flatlines(self, dataframe: pd.DataFrame) -> tuple[List[str], Dict[str, pd.Series]]:
        """Detects micro-flatline segments using rolling variance thresholding.

        For a channel x, rolling variance is computed on a trailing window of
        size W. A timestamp t is flagged when Var_t < epsilon. This captures
        malfunction patterns with tiny floating-point oscillations that evade
        strict zero-variance checks.

        Args:
            dataframe: Numeric sensor matrix.

        Returns:
            Tuple of (flatline_columns, per-column boolean masks).
        """
        flatline_columns: List[str] = []
        flatline_masks: Dict[str, pd.Series] = {}

        min_periods = max(3, self._flatline_window_size // 2)
        for column in dataframe.columns:
            series = dataframe[column]
            rolling_variance = series.rolling(
                window=self._flatline_window_size,
                min_periods=min_periods,
            ).var(ddof=0)

            flatline_mask = (rolling_variance < self._flatline_variance_epsilon) & series.notna()
            flatline_mask = flatline_mask.fillna(False)
            flatline_masks[column] = flatline_mask

            if bool(flatline_mask.any()):
                flatline_columns.append(column)

        return flatline_columns, flatline_masks

    def _remove_outliers_zscore(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
        """Removes outliers by z-score thresholding.

        Args:
            dataframe: Numeric DataFrame with potential spikes.

        Returns:
            Cleaned DataFrame and count of replaced outliers per column.
        """
        cleaned = dataframe.copy()
        outlier_counts: Dict[str, int] = {}

        for column in cleaned.columns:
            series = cleaned[column]
            mean = float(series.mean(skipna=True))
            std = float(series.std(skipna=True, ddof=0))

            if not np.isfinite(std) or std == 0.0:
                outlier_counts[column] = 0
                continue

            z_scores = (series - mean) / std
            mask = z_scores.abs() > self._zscore_threshold
            outlier_counts[column] = int(mask.sum())
            cleaned.loc[mask, column] = np.nan

        return cleaned, outlier_counts
