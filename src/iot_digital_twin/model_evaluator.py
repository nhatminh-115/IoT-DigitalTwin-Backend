"""Continuous model evaluation utilities for delayed prediction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class EvaluationError(RuntimeError):
    """Raised when model evaluation artifacts cannot be computed."""


@dataclass
class EvaluationArtifacts:
    """Container for historical evaluation outputs.

    Attributes:
        ground_truth: Observed values y_t for aligned timestamps.
        prediction_mean: Historical one-step predictions \hat{y}_t.
        prediction_std: Predictive standard deviation per timestamp and feature.
        residuals: Error matrix e_t = y_t - \hat{y}_t.
        rolling_mae: Rolling mean absolute error time-series.
        rolling_rmse: Rolling root mean squared error time-series.
    """

    ground_truth: pd.DataFrame
    prediction_mean: pd.DataFrame
    prediction_std: pd.DataFrame
    residuals: pd.DataFrame
    rolling_mae: pd.Series
    rolling_rmse: pd.Series


class ModelEvaluator:
    """Builds long-term robustness metrics for multivariate forecasters."""

    def __init__(self, rolling_window: int = 20, ci_z_value: float = 1.96) -> None:
        """Initializes evaluation hyperparameters.

        Args:
            rolling_window: Window length for rolling MAE and RMSE.
            ci_z_value: Two-sided normal z-score for confidence interval width.
        """
        self._rolling_window = rolling_window
        self._ci_z_value = ci_z_value

    @property
    def ci_z_value(self) -> float:
        """Returns z-score used for prediction interval construction."""
        return self._ci_z_value

    def evaluate(
        self,
        ground_truth: pd.DataFrame,
        prediction_mean: pd.DataFrame,
        prediction_std: pd.DataFrame,
        selected_features: list[str],
    ) -> EvaluationArtifacts:
        """Computes residual and rolling error diagnostics.

        Mathematical definition:
        - Residual: e_t = y_t - \hat{y}_t
        - MAE_t(window): mean(|e_i|) over the rolling window ending at t
        - RMSE_t(window): sqrt(mean(e_i^2)) over the rolling window ending at t

        For multivariate module-level monitoring, MAE and RMSE are aggregated by
        averaging across selected features for each timestamp before rolling.

        Args:
            ground_truth: Observed target matrix indexed by datetime.
            prediction_mean: Predicted historical target matrix.
            prediction_std: Predicted standard deviation matrix.
            selected_features: Feature columns to aggregate for dashboard metrics.

        Returns:
            EvaluationArtifacts with residual and rolling diagnostics.

        Raises:
            EvaluationError: If inputs are empty or feature list is invalid.
        """
        if ground_truth.empty or prediction_mean.empty or prediction_std.empty:
            raise EvaluationError("Evaluation inputs cannot be empty.")

        if not selected_features:
            raise EvaluationError("At least one feature must be selected for evaluation.")

        missing_features = [
            feature for feature in selected_features if feature not in ground_truth.columns
        ]
        if missing_features:
            raise EvaluationError(
                f"Selected features are missing in evaluation frame: {missing_features}."
            )

        common_index = ground_truth.index.intersection(prediction_mean.index).intersection(prediction_std.index)
        if common_index.empty:
            raise EvaluationError("No aligned timestamps between targets and predictions.")

        aligned_truth = ground_truth.loc[common_index].copy()
        aligned_mean = prediction_mean.loc[common_index].copy()
        aligned_std = prediction_std.loc[common_index].copy()

        residuals = aligned_truth - aligned_mean

        # Aggregate per-timestamp errors across module features.
        # This yields a single scalar error trajectory used in rolling drift charts.
        abs_error = residuals[selected_features].abs().mean(axis=1)
        squared_error = np.square(residuals[selected_features]).mean(axis=1)

        rolling_mae = abs_error.rolling(window=self._rolling_window, min_periods=1).mean()
        rolling_rmse = np.sqrt(squared_error.rolling(window=self._rolling_window, min_periods=1).mean())

        return EvaluationArtifacts(
            ground_truth=aligned_truth,
            prediction_mean=aligned_mean,
            prediction_std=aligned_std,
            residuals=residuals,
            rolling_mae=rolling_mae.rename("Rolling MAE"),
            rolling_rmse=rolling_rmse.rename("Rolling RMSE"),
        )

    def build_prediction_interval(
        self,
        mean_series: pd.Series,
        std_series: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Constructs two-sided confidence interval around predicted mean.

        CI_t = [mu_t - z * sigma_t, mu_t + z * sigma_t]

        Args:
            mean_series: Predicted mean trajectory.
            std_series: Predicted standard deviation trajectory.

        Returns:
            Lower and upper confidence interval bounds.
        """
        interval_radius = self._ci_z_value * std_series
        lower = (mean_series - interval_radius).rename("CI Lower")
        upper = (mean_series + interval_radius).rename("CI Upper")
        return lower, upper
