"""Streamlit dashboard for real-time IoT Digital Twin monitoring and inference."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import re
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.iot_digital_twin.data_fetcher import DataFetchError, DataFetcher, DataFetcherConfig
from src.iot_digital_twin.data_quality_gate import DataQualityError, DataQualityGate, DataQualityReport
from src.iot_digital_twin.model_evaluator import EvaluationArtifacts, EvaluationError, ModelEvaluator
from src.iot_digital_twin.predictor import DeepTimeSeriesPredictor, PredictorConfig, PredictorError


CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnBhE8u7fdXKEooOlqgGYtSZLeUxrQUu9e_q6MnMrGbakxXESMYVf0utORhEG3pEqWffGhX6J-V2cC/pub?output=csv"
)
MODEL_CHECKPOINT_PATH = Path("artifacts") / "lstm_checkpoint.pt"
PREDICTION_LOG_PATH = Path("artifacts") / "prediction_log.csv"


class StreamlitDashboard:
    """Coordinates data ingestion, quality controls, forecasting, and UI rendering."""

    def __init__(self) -> None:
        """Initializes service objects and streamlit session state."""
        fetcher_config = DataFetcherConfig(csv_url=CSV_URL)
        predictor_config = PredictorConfig()
        self.model_checkpoint_path = MODEL_CHECKPOINT_PATH
        self.prediction_log_path = PREDICTION_LOG_PATH

        self.fetcher = DataFetcher(fetcher_config)
        self.quality_gate = DataQualityGate(zscore_threshold=3.0)
        self.evaluator = ModelEvaluator(rolling_window=20, ci_z_value=1.96)

        if "predictor" not in st.session_state:
            st.session_state.predictor = DeepTimeSeriesPredictor(config=predictor_config)

        self.predictor: DeepTimeSeriesPredictor = st.session_state.predictor
        if "checkpoint_loaded" not in st.session_state:
            st.session_state.checkpoint_loaded = False
        if "force_retrain_once" not in st.session_state:
            st.session_state.force_retrain_once = False

    def run(self) -> None:
        """Executes one dashboard refresh cycle."""
        st.set_page_config(page_title="IoT Digital Twin Forecast Dashboard", layout="wide")
        st.title("Real-Time Multivariate Forecasting Dashboard")
        st.caption("Data source refresh interval: 3 minutes")
        st.caption(f"Model checkpoint path: {self.model_checkpoint_path.as_posix()}")

        refresh_millis = st.sidebar.number_input(
            "Auto-refresh interval (ms)", min_value=30_000, max_value=900_000, value=180_000, step=30_000
        )
        st_autorefresh(interval=int(refresh_millis), key="dashboard_autorefresh")

        if st.sidebar.button("Refresh now", use_container_width=True):
            st.rerun()

        if st.sidebar.button("Retrain now", use_container_width=True):
            st.session_state.force_retrain_once = True
            st.rerun()

        selected_tail = st.sidebar.slider("Historical window length", min_value=60, max_value=600, value=240, step=20)
        eval_horizon = st.sidebar.slider(
            "Evaluation horizon (recent steps)", min_value=240, max_value=3000, value=900, step=60
        )
        retrain_each_refresh = st.sidebar.checkbox("Retrain model on each refresh", value=False)
        mc_samples = st.sidebar.slider("MC dropout samples", min_value=5, max_value=40, value=10, step=5)
        log_rows_to_show = st.sidebar.slider("Prediction log rows", min_value=50, max_value=5000, value=500, step=50)
        show_advanced_metrics = st.sidebar.checkbox("Advanced metrics", value=False)

        try:
            if not st.session_state.checkpoint_loaded:
                try:
                    st.session_state.checkpoint_loaded = self.predictor.load_checkpoint(
                        self.model_checkpoint_path
                    )
                except PredictorError as exc:
                    self._handle_checkpoint_load_error(exc)

            raw_df = self.fetcher.fetch()
            clean_df, report = self.quality_gate.process(raw_df)

            train_df = clean_df.tail(max(720, self.predictor.config.sequence_length + 1))
            schema_mismatch = self.predictor.is_fitted and set(self.predictor.feature_columns) != set(clean_df.columns)
            if schema_mismatch:
                st.warning(
                    "Model schema differs from current data channels. Retraining now to align columns."
                )
            should_retrain = (
                retrain_each_refresh
                or not self.predictor.is_fitted
                or schema_mismatch
                or bool(st.session_state.force_retrain_once)
            )
            if should_retrain:
                self.predictor.fit(train_df)
                self.predictor.save_checkpoint(self.model_checkpoint_path)
                st.session_state.force_retrain_once = False
                st.sidebar.success("Retraining completed and checkpoint updated.")

            prediction = self.predictor.predict_next_step(clean_df)
            step_delta = self._infer_sampling_delta(clean_df.index)
            self._append_prediction_log(clean_df, prediction, step_delta)
            self._reconcile_prediction_log_with_ground_truth(clean_df, step_delta)

            eval_input = clean_df.tail(max(eval_horizon, self.predictor.config.sequence_length + 1))
            ground_truth, pred_mean, pred_std = self.predictor.predict_historical_with_uncertainty(
                eval_input,
                mc_samples=int(mc_samples),
            )

            selected_module = self._select_module_sidebar(clean_df)
            selected_features = self._module_features(clean_df, selected_module)
            evaluation = self.evaluator.evaluate(
                ground_truth=ground_truth,
                prediction_mean=pred_mean,
                prediction_std=pred_std,
                selected_features=selected_features,
            )

            self._render_quality_summary(report)
            self._render_metrics(clean_df, prediction, selected_features)
            self._render_module_charts(
                evaluation,
                selected_module,
                selected_features,
                selected_tail,
                clean_df,
                prediction,
            )
            self._render_rolling_error_chart(evaluation)
            if show_advanced_metrics:
                self._render_feature_rolling_error_chart(evaluation, selected_features)
            self._render_residual_histogram(evaluation, selected_features)
            self._render_prediction_log_table(
                rows_to_show=int(log_rows_to_show),
                selected_module=selected_module,
            )

        except (DataFetchError, DataQualityError, PredictorError, EvaluationError) as exc:
            st.error(str(exc))
            st.info("The pipeline halted safely due to upstream data or model constraints.")
        except Exception as exc:
            st.error(f"Unexpected runtime failure: {exc}")
            st.text(traceback.format_exc())

    def _select_module_sidebar(self, clean_df: pd.DataFrame) -> str:
        """Renders module selection and returns selected module identifier."""
        modules = sorted({column.split("_")[0] for column in clean_df.columns if "_" in column})
        if not modules:
            raise EvaluationError("No module-feature channels available for module drill-down.")
        return st.sidebar.selectbox("Select module", options=modules, index=0)

    def _module_features(self, dataframe: pd.DataFrame, module: str) -> list[str]:
        """Returns canonical ordered features for a selected module."""
        feature_order = ["Temp", "Humid", "CO2", "TVOC"]
        expected = [f"{module}_{feature}" for feature in feature_order]
        available = [column for column in expected if column in dataframe.columns]
        if not available:
            raise EvaluationError(f"No features found for selected module {module}.")
        return available

    def _render_quality_summary(self, report: DataQualityReport) -> None:
        """Displays preprocessing diagnostics for operational visibility."""
        with st.expander("Data Quality Diagnostics", expanded=True):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Rows In", report.input_rows)
            col_b.metric("Rows Out", report.output_rows)
            col_c.metric("Flatline Channels", len(report.flatline_columns))
            col_d.metric("Dropped Channels", len(report.dropped_columns))

            if report.flatline_columns:
                st.write("Flatline channels:", ", ".join(report.flatline_columns))
            if report.dropped_columns:
                st.write("Dropped channels:", ", ".join(report.dropped_columns))

            outlier_total = int(np.sum(list(report.outlier_counts.values())))
            st.write(f"Outliers replaced via z-score filtering: {outlier_total}")

    def _render_metrics(
        self,
        clean_df: pd.DataFrame,
        prediction: pd.Series,
        selected_features: list[str],
    ) -> None:
        """Renders latest observed values and next-step predictions."""
        st.subheader("Current vs Predicted Next Step")

        latest = clean_df.iloc[-1]
        metric_cols = st.columns(4)

        for idx, column in enumerate(selected_features):
            current_val = float(latest[column])
            predicted_val = float(prediction[column])
            delta = predicted_val - current_val
            metric_cols[idx % 4].metric(
                label=column,
                value=f"{current_val:.3f}",
                delta=f"{delta:+.3f}",
            )

        with st.expander("Full prediction vector"):
            pred_df = pd.DataFrame({
                "current": latest,
                "predicted_t+1": prediction,
                "delta": prediction - latest,
            })
            st.dataframe(pred_df)

    def _handle_checkpoint_load_error(self, error: PredictorError) -> None:
        """Handles checkpoint loading failure with safe migration behavior.

        When preprocessing logic changes, old checkpoints may become incompatible.
        In that case, the file is archived with a legacy timestamp suffix and the
        pipeline proceeds to retrain on current data.
        """
        message = str(error)
        if "preprocessing mode is incompatible" in message and self.model_checkpoint_path.exists():
            legacy_path = self._archive_incompatible_checkpoint(self.model_checkpoint_path)
            st.warning(
                "Checkpoint is from an older preprocessing pipeline and was archived. "
                f"Legacy file: {legacy_path.as_posix()}. Retraining a new checkpoint now."
            )
            st.session_state.checkpoint_loaded = True
            return

        st.warning(f"Checkpoint load failed ({message}). Model will be trained from current data.")

    @staticmethod
    def _archive_incompatible_checkpoint(checkpoint_path: Path) -> Path:
        """Renames an incompatible checkpoint to a legacy backup file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        legacy_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.legacy_{timestamp}{checkpoint_path.suffix}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.replace(legacy_path)
        return legacy_path

    def _render_module_charts(
        self,
        evaluation: EvaluationArtifacts,
        selected_module: str,
        selected_features: list[str],
        selected_tail: int,
        clean_df: pd.DataFrame,
        next_prediction: pd.Series,
    ) -> None:
        """Renders module charts with delayed overlays and explicit t+1 forecast points.

        The delayed overlay compares y_t and historical y_hat_t on aligned past
        timestamps. Additionally, the chart shows the forward-looking forecast
        from origin time t to target time t+1 using markers and a dashed segment.
        """
        st.subheader(f"Module Drill-Down: {selected_module}")

        truth_tail = evaluation.ground_truth[selected_features].tail(selected_tail)
        pred_tail = evaluation.prediction_mean[selected_features].tail(selected_tail)
        std_tail = evaluation.prediction_std[selected_features].tail(selected_tail)
        step_delta = self._infer_sampling_delta(clean_df.index)

        forecast_origin_time = clean_df.index[-1]
        forecast_target_time = forecast_origin_time + step_delta

        for feature in selected_features:
            lower_ci, upper_ci = self.evaluator.build_prediction_interval(
                pred_tail[feature],
                std_tail[feature],
            )

            forecast_origin_value = float(clean_df.iloc[-1][feature])
            forecast_target_value = float(next_prediction[feature])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=truth_tail.index,
                    y=truth_tail[feature],
                    mode="lines",
                    line=dict(width=2),
                    name="Ground Truth y_t",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_tail.index,
                    y=pred_tail[feature],
                    mode="lines",
                    line=dict(width=2, dash="dash"),
                    name="Historical Prediction y_hat_t",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_tail.index,
                    y=upper_ci,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_tail.index,
                    y=lower_ci,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(31, 119, 180, 0.15)",
                    name=f"{int(self.evaluator.ci_z_value / 1.96 * 95)}% Prediction Interval",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[forecast_origin_time],
                    y=[forecast_origin_value],
                    mode="markers",
                    marker=dict(size=9, symbol="circle"),
                    name="Forecast Origin (t)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[forecast_origin_time, forecast_target_time],
                    y=[forecast_origin_value, forecast_target_value],
                    mode="lines+markers",
                    line=dict(width=2, dash="dot"),
                    marker=dict(size=9, symbol="diamond"),
                    name="Forward Forecast to t+1",
                )
            )

            fig.add_shape(
                type="line",
                x0=forecast_origin_time,
                x1=forecast_origin_time,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(width=1, dash="dot", color="gray"),
            )
            fig.add_annotation(
                x=forecast_origin_time,
                y=1,
                xref="x",
                yref="paper",
                text="t (forecast origin)",
                showarrow=False,
                yshift=8,
                xanchor="left",
                font=dict(size=11, color="gray"),
            )

            fig.update_layout(
                title=f"{feature}: Ground Truth vs Historical Predictions",
                xaxis_title="Datetime",
                yaxis_title="Sensor Value",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="x unified",
            )
            fig.update_xaxes(
                tickformat="%Y-%m-%d\n%H:%M",
                showgrid=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _infer_sampling_delta(index: pd.Index) -> pd.Timedelta:
        """Infers sampling interval from datetime index for plotting future t+1.

        Uses the median positive time difference between consecutive timestamps.
        If inference is unstable, defaults to 3 minutes based on source cadence.
        """
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return pd.Timedelta(minutes=3)

        diffs = index.to_series().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return pd.Timedelta(minutes=3)
        return pd.to_timedelta(diffs.median())

    def _append_prediction_log(
        self,
        clean_df: pd.DataFrame,
        prediction: pd.Series,
        step_delta: pd.Timedelta,
    ) -> None:
        """Appends a single forecast record to durable CSV storage.

        The log is append-only and intended for long-term operational tracking.
        Each row captures issue time t, target time t+1, and all predicted channels.
        """
        if clean_df.empty or prediction.empty:
            return

        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)

        issue_time = clean_df.index[-1]
        target_time = issue_time + step_delta

        log_row: dict[str, object] = {
            "logged_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "forecast_issue_time": pd.Timestamp(issue_time).isoformat(),
            "forecast_target_time": pd.Timestamp(target_time).isoformat(),
            "ground_truth_matched": False,
        }
        for feature, value in prediction.items():
            log_row[f"pred_{feature}"] = float(value)
            log_row[f"gt_{feature}"] = np.nan
            log_row[f"gt_matched_{feature}"] = False
            log_row[f"residual_{feature}"] = np.nan
            log_row[f"abs_error_{feature}"] = np.nan

        row_df = pd.DataFrame([log_row])
        existing_df = self._load_prediction_log_safe()
        if not existing_df.empty:
            existing_df = self._normalize_legacy_prediction_columns(existing_df)

        combined = pd.concat([existing_df, row_df], ignore_index=True, sort=False)
        if {"forecast_issue_time", "forecast_target_time"}.issubset(set(combined.columns)):
            combined = combined.drop_duplicates(
                subset=["forecast_issue_time", "forecast_target_time"],
                keep="last",
            )
        self._save_prediction_log(combined)

    def _reconcile_prediction_log_with_ground_truth(
        self,
        clean_df: pd.DataFrame,
        step_delta: pd.Timedelta,
    ) -> None:
        """Backfills ground truth for previously issued predictions.

        For each unresolved log row, the method searches the nearest observed
        timestamp around the forecast target time within a tolerance window.
        When matched, it writes ground truth, residual, and absolute error:
            e = y - y_hat
            |e| = abs(y - y_hat)
        """
        if not self.prediction_log_path.exists() or clean_df.empty:
            return

        log_df = self._load_prediction_log_safe()

        if log_df.empty or "forecast_target_time" not in log_df.columns:
            return

        log_df = self._normalize_legacy_prediction_columns(log_df)

        if "ground_truth_matched" not in log_df.columns:
            log_df["ground_truth_matched"] = False
        log_df["ground_truth_matched"] = self._coerce_to_bool_series(log_df["ground_truth_matched"])

        log_df["forecast_target_time"] = pd.to_datetime(log_df["forecast_target_time"], errors="coerce")

        clean_sorted = clean_df.sort_index()
        clean_times = clean_sorted.index
        if not isinstance(clean_times, pd.DatetimeIndex) or len(clean_times) == 0:
            return

        tolerance = max(pd.Timedelta(minutes=1), step_delta / 2)
        prediction_features = [
            column[len("pred_") :]
            for column in log_df.columns
            if column.startswith("pred_")
        ]
        prediction_features = [feature for feature in prediction_features if feature in clean_sorted.columns]
        if not prediction_features:
            return

        for feature in prediction_features:
            feature_match_col = f"gt_matched_{feature}"
            if feature_match_col not in log_df.columns:
                log_df[feature_match_col] = False
            log_df[feature_match_col] = self._coerce_to_bool_series(log_df[feature_match_col])

        feature_match_columns = [f"gt_matched_{feature}" for feature in prediction_features]
        unresolved_mask = (
            (~log_df[feature_match_columns].all(axis=1))
            & log_df["forecast_target_time"].notna()
        )
        if not bool(unresolved_mask.any()):
            return

        # Coerce numeric columns to float to avoid PyArrow/string validation type errors
        for feature in prediction_features:
            for prefix in ("pred_", "gt_", "residual_", "abs_error_"):
                col_name = f"{prefix}{feature}"
                if col_name in log_df.columns:
                    log_df[col_name] = pd.to_numeric(log_df[col_name], errors="coerce")

        updated_any = False
        unresolved_indices = log_df.index[unresolved_mask]
        for row_idx in unresolved_indices:
            target_time = log_df.at[row_idx, "forecast_target_time"]
            nearest_pos = clean_times.get_indexer([target_time], method="nearest")[0]
            if nearest_pos < 0:
                continue

            nearest_time = clean_times[nearest_pos]
            if abs(nearest_time - target_time) > tolerance:
                continue

            for feature in prediction_features:
                pred_col = f"pred_{feature}"
                gt_col = f"gt_{feature}"
                gt_matched_col = f"gt_matched_{feature}"
                residual_col = f"residual_{feature}"
                abs_error_col = f"abs_error_{feature}"

                pred_val = pd.to_numeric(log_df.at[row_idx, pred_col], errors="coerce")
                gt_val = pd.to_numeric(clean_sorted.at[nearest_time, feature], errors="coerce")
                if pd.isna(pred_val) or pd.isna(gt_val):
                    continue

                residual = float(gt_val - pred_val)
                log_df.at[row_idx, gt_col] = float(gt_val)
                log_df.at[row_idx, gt_matched_col] = True
                log_df.at[row_idx, residual_col] = residual
                log_df.at[row_idx, abs_error_col] = abs(residual)

            log_df.at[row_idx, "ground_truth_time"] = pd.Timestamp(nearest_time).isoformat()
            log_df.at[row_idx, "ground_truth_matched"] = bool(
                log_df.loc[row_idx, feature_match_columns].all()
            )
            updated_any = True

        if updated_any:
            self._save_prediction_log(log_df)

    def _render_prediction_log_table(self, rows_to_show: int, selected_module: str) -> None:
        """Renders long-horizon prediction log with prediction-vs-truth details."""
        st.subheader("Prediction Log (Long-Term)")
        st.caption(f"Log file: {self.prediction_log_path.as_posix()}")

        if not self.prediction_log_path.exists():
            st.info("Prediction log is not available yet.")
            return

        log_df = self._load_prediction_log_safe()

        if log_df.empty:
            st.info("Prediction log is empty.")
            return

        log_df = self._normalize_legacy_prediction_columns(log_df)

        if "forecast_issue_time" in log_df.columns:
            log_df["forecast_issue_time"] = pd.to_datetime(log_df["forecast_issue_time"], errors="coerce")
        if "forecast_target_time" in log_df.columns:
            log_df["forecast_target_time"] = pd.to_datetime(log_df["forecast_target_time"], errors="coerce")
        if "ground_truth_time" in log_df.columns:
            log_df["ground_truth_time"] = pd.to_datetime(log_df["ground_truth_time"], errors="coerce")

        if {"forecast_issue_time", "forecast_target_time"}.issubset(set(log_df.columns)):
            log_df = log_df.sort_values(
                by=["forecast_issue_time", "forecast_target_time"],
                ascending=[True, True],
            ).drop_duplicates(
                subset=["forecast_issue_time", "forecast_target_time"],
                keep="last",
            )

        view_mode = st.radio(
            "Log table mode",
            options=["Selected module", "All modules"],
            horizontal=True,
            key="prediction_log_view_mode",
        )

        if "forecast_issue_time" in log_df.columns:
            log_tail = log_df.sort_values(by="forecast_issue_time", ascending=False).head(rows_to_show).copy()
        else:
            log_tail = log_df.tail(rows_to_show).copy()

        if view_mode == "Selected module":
            module_columns = self._module_log_columns(log_tail.columns, selected_module)
            base_columns = [
                column
                for column in [
                    "forecast_issue_time",
                    "forecast_target_time",
                ]
                if column in log_tail.columns
            ]
            display_columns = base_columns + module_columns
            if display_columns:
                log_tail = log_tail[display_columns]

        if "ground_truth_matched" in log_tail.columns:
            log_tail = log_tail.drop(columns=["ground_truth_matched"])

        st.dataframe(log_tail, use_container_width=True)

    def _load_prediction_log_safe(self) -> pd.DataFrame:
        """Loads prediction log robustly even when historical schemas differ.

        This parser repairs common drift patterns where new fields were appended
        without rewriting headers. In particular, it handles rows with one extra
        boolean field (`ground_truth_matched`) inserted after target time.
        """
        if not self.prediction_log_path.exists():
            return pd.DataFrame()

        try:
            with self.prediction_log_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
        except Exception:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        header = list(rows[0])
        records: list[dict[str, object]] = []
        repaired_schema = False

        for row in rows[1:]:
            if not row:
                continue

            if len(row) == len(header):
                record = dict(zip(header, row))
                records.append(record)
                continue

            if (
                len(row) == len(header) + 1
                and "ground_truth_matched" not in header
                and len(row) >= 4
                and row[3] in {"True", "False", "true", "false"}
            ):
                header = header[:3] + ["ground_truth_matched"] + header[3:]
                for previous in records:
                    previous.setdefault("ground_truth_matched", "")
                record = dict(zip(header, row))
                records.append(record)
                repaired_schema = True
                continue

            if len(row) < len(header):
                padded = row + [""] * (len(header) - len(row))
                records.append(dict(zip(header, padded)))
                continue

            records.append(dict(zip(header, row[: len(header)])))

        if not records:
            return pd.DataFrame(columns=header)

        dataframe = pd.DataFrame.from_records(records, columns=header)
        dataframe, repaired_values = self._repair_prediction_log_timestamps(dataframe)

        # Persist repaired schema once so future reads remain fast and stable.
        if repaired_schema or repaired_values:
            self._save_prediction_log(dataframe)

        return dataframe

    def _save_prediction_log(self, log_df: pd.DataFrame) -> None:
        """Writes normalized prediction log to disk with consistent schema."""
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(self.prediction_log_path, index=False)

    @staticmethod
    def _repair_prediction_log_timestamps(log_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Repairs missing/invalid forecast target timestamps in legacy log rows.

        Some historical rows were written with empty `forecast_target_time` during
        schema transitions. This method reconstructs missing targets using
        `forecast_issue_time + inferred_step_delta`.
        """
        repaired = False
        dataframe = log_df.copy()

        if "forecast_issue_time" not in dataframe.columns:
            return dataframe, repaired

        issue_dt = pd.to_datetime(dataframe["forecast_issue_time"], errors="coerce")
        target_dt = (
            pd.to_datetime(dataframe["forecast_target_time"], errors="coerce")
            if "forecast_target_time" in dataframe.columns
            else pd.Series(pd.NaT, index=dataframe.index)
        )

        valid_delta_mask = issue_dt.notna() & target_dt.notna() & (target_dt >= issue_dt)
        if bool(valid_delta_mask.any()):
            inferred_delta = (target_dt[valid_delta_mask] - issue_dt[valid_delta_mask]).median()
        else:
            inferred_delta = pd.Timedelta(minutes=3)

        if pd.isna(inferred_delta) or inferred_delta <= pd.Timedelta(0):
            inferred_delta = pd.Timedelta(minutes=3)

        missing_target_mask = issue_dt.notna() & target_dt.isna()
        if bool(missing_target_mask.any()):
            target_dt.loc[missing_target_mask] = issue_dt.loc[missing_target_mask] + inferred_delta
            repaired = True

        dataframe["forecast_issue_time"] = issue_dt
        dataframe["forecast_target_time"] = target_dt

        if "ground_truth_time" in dataframe.columns:
            dataframe["ground_truth_time"] = pd.to_datetime(dataframe["ground_truth_time"], errors="coerce")

        return dataframe, repaired

    @staticmethod
    def _module_log_columns(columns: pd.Index, module: str) -> list[str]:
        """Returns module-specific prediction, truth, and error columns."""
        pattern = re.compile(rf"^(pred|gt|residual|abs_error)_{re.escape(module)}_")
        matched_pattern = re.compile(rf"^gt_matched_{re.escape(module)}_")
        ordered = [column for column in columns if pattern.match(str(column))]
        return [column for column in ordered if not matched_pattern.match(str(column))]

    @staticmethod
    def _normalize_legacy_prediction_columns(log_df: pd.DataFrame) -> pd.DataFrame:
        """Upgrades legacy log schema to `pred_`-prefixed prediction columns.

        Older log rows may store prediction columns directly as sensor names such
        as `M1_Temp`. They are migrated in-memory to `pred_M1_Temp` style to
        unify downstream processing.
        """
        upgraded = log_df.copy()
        for column in list(upgraded.columns):
            if re.match(r"^M(?:1|4|6|7|8|9|10|11)_(Temp|Humid|CO2|TVOC)$", str(column)):
                pred_col = f"pred_{column}"
                if pred_col not in upgraded.columns:
                    upgraded = upgraded.rename(columns={column: pred_col})
        return upgraded

    @staticmethod
    def _coerce_to_bool_series(series: pd.Series) -> pd.Series:
        """Converts mixed string/number/boolean values to strict boolean flags."""
        if series.dtype == bool:
            return series

        normalized = series.astype(str).str.strip().str.lower()
        true_tokens = {"true", "1", "yes", "y", "t"}
        false_tokens = {"false", "0", "no", "n", "f", "", "nan", "none"}

        mapped = pd.Series(False, index=series.index, dtype=bool)
        mapped[normalized.isin(true_tokens)] = True
        mapped[normalized.isin(false_tokens)] = False
        return mapped

    def _render_rolling_error_chart(self, evaluation: EvaluationArtifacts) -> None:
        """Renders rolling MAE and RMSE to monitor drift over time."""
        st.subheader("Rolling Error Metrics (window=20 steps)")
        error_fig = go.Figure()
        error_fig.add_trace(
            go.Scatter(
                x=evaluation.rolling_mae.index,
                y=evaluation.rolling_mae.values,
                mode="lines",
                name="Rolling MAE",
            )
        )
        error_fig.add_trace(
            go.Scatter(
                x=evaluation.rolling_rmse.index,
                y=evaluation.rolling_rmse.values,
                mode="lines",
                name="Rolling RMSE",
            )
        )
        error_fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="Error",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(error_fig, use_container_width=True)

    def _render_feature_rolling_error_chart(
        self,
        evaluation: EvaluationArtifacts,
        selected_features: list[str],
    ) -> None:
        """Renders optional per-feature rolling MAE/RMSE curves."""
        st.subheader("Advanced: Per-Feature Rolling MAE/RMSE")

        rolling_window = int(self.evaluator._rolling_window)
        feature_fig = go.Figure()
        color_palette = [
            "#636EFA",
            "#00CC96",
            "#EF553B",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
        ]

        for idx, feature in enumerate(selected_features):
            feature_color = color_palette[idx % len(color_palette)]
            feature_residual = evaluation.residuals[feature]
            feature_abs_error = feature_residual.abs()
            feature_squared_error = np.square(feature_residual)

            feature_rolling_mae = feature_abs_error.rolling(window=rolling_window, min_periods=1).mean()
            feature_rolling_rmse = np.sqrt(
                feature_squared_error.rolling(window=rolling_window, min_periods=1).mean()
            )

            feature_fig.add_trace(
                go.Scatter(
                    x=feature_rolling_mae.index,
                    y=feature_rolling_mae.values,
                    mode="lines",
                    name=f"{feature} (MAE/RMSE)",
                    legendgroup=feature,
                    line=dict(color=feature_color, width=2),
                    hovertemplate="%{x}<br>MAE=%{y:.6f}<extra></extra>",
                )
            )
            feature_fig.add_trace(
                go.Scatter(
                    x=feature_rolling_rmse.index,
                    y=feature_rolling_rmse.values,
                    mode="lines",
                    line=dict(color=feature_color, dash="dash", width=2),
                    name=f"{feature} RMSE",
                    legendgroup=feature,
                    showlegend=False,
                    hovertemplate="%{x}<br>RMSE=%{y:.6f}<extra></extra>",
                )
            )

        feature_fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="Rolling Error",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                groupclick="togglegroup",
            ),
        )
        st.plotly_chart(feature_fig, use_container_width=True)

    def _render_residual_histogram(
        self,
        evaluation: EvaluationArtifacts,
        selected_features: list[str],
    ) -> None:
        """Renders residual histogram and statistical interpretation note."""
        st.subheader("Residual Distribution Analysis")

        residual_fig = go.Figure()
        for feature in selected_features:
            residual_fig.add_trace(
                go.Histogram(
                    x=evaluation.residuals[feature],
                    opacity=0.6,
                    name=feature,
                    nbinsx=40,
                )
            )

        residual_fig.update_layout(
            barmode="overlay",
            xaxis_title="Residual e_t = y_t - y_hat_t",
            yaxis_title="Frequency",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(residual_fig, use_container_width=True)

        st.caption(
            "Academic note: A near-Gaussian residual distribution centered close to zero "
            "suggests remaining error behaves approximately as white noise, indicating "
            "the model has captured most systematic temporal structure."
        )


def main() -> None:
    """Application entry point."""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
