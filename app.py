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

_CHART_COLORS = {
    "truth": "#60A5FA",
    "pred": "#34D399",
    "ci_fill": "rgba(52, 211, 153, 0.10)",
    "forecast_line": "#FBBF24",
    "origin_marker": "#A78BFA",
    "separator": "rgba(148, 163, 184, 0.2)",
}
_COLOR_PALETTE = ["#60A5FA", "#34D399", "#F87171", "#FBBF24", "#A78BFA", "#FB923C"]

_CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ---- header banner ---- */
.db-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid rgba(148,163,184,0.12);
    padding: 1.4rem 2rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
}
.db-title {
    color: #F1F5F9;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.025em;
}
.db-meta { color: #64748B; font-size: 0.8rem; margin-top: 5px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34D399;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-top: 8px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #34D399;
    border-radius: 50%;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ---- section labels ---- */
.section-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.65rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid rgba(148,163,184,0.15);
}

/* ---- module heading ---- */
.module-heading {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem;
    padding: 0.4rem 0.8rem;
    background: rgba(148,163,184,0.06);
    border-left: 3px solid #3B82F6;
    border-radius: 0 6px 6px 0;
}

/* ---- metric cards ---- */
[data-testid="metric-container"] {
    background: #1E293B !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: #1E293B;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(148,163,184,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.82rem;
    color: #64748B;
    border: none !important;
    padding: 6px 20px;
}
.stTabs [aria-selected="true"] {
    background: #334155 !important;
    color: #F1F5F9 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ---- settings expander ---- */
[data-testid="stExpander"] {
    background: #1E293B;
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-radius: 10px;
}
</style>
"""


class StreamlitDashboard:
    """Coordinates data ingestion, quality controls, forecasting, and UI rendering."""

    def __init__(self) -> None:
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

    def run(self) -> None:
        """Executes one dashboard refresh cycle."""
        st.set_page_config(
            page_title="IoT Digital Twin",
            page_icon="",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.markdown(_CSS, unsafe_allow_html=True)

        # Header
        st.markdown(
            """
            <div class="db-header">
                <div class="db-title">IoT Digital Twin — Multivariate Forecast</div>
                <div class="db-meta">LSTM · DTW-guided · Real-time sensor telemetry</div>
                <div class="status-pill"><span class="status-dot"></span>LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top action bar
        btn_col, _, settings_col = st.columns([1, 6, 2])
        if btn_col.button("Refresh now", width="stretch"):
            st.rerun()

        # Collapsible settings — defaults are sane, most users never touch this
        with settings_col:
            with st.expander("Settings", expanded=False):
                refresh_millis = st.number_input(
                    "Auto-refresh (ms)", min_value=30_000, max_value=900_000,
                    value=180_000, step=30_000,
                )
                selected_tail = st.slider("Historical window", 60, 600, 240, 20)
                eval_horizon = st.slider("Eval horizon (steps)", 240, 3000, 900, 60)
                mc_samples = st.slider("MC dropout samples", 5, 40, 10, 5)
                show_advanced_metrics = st.checkbox("Advanced per-feature metrics", value=False)
                log_rows_to_show = st.slider("Log rows", 50, 5000, 500, 50)

        st_autorefresh(interval=int(refresh_millis), key="dashboard_autorefresh")

        # Pipeline
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
            schema_mismatch = (
                self.predictor.is_fitted
                and set(self.predictor.feature_columns) != set(clean_df.columns)
            )
            if schema_mismatch:
                st.warning("Model schema differs from current data. Retraining now.")

            should_retrain = not self.predictor.is_fitted or schema_mismatch
            if should_retrain:
                with st.spinner("Training model..."):
                    self.predictor.fit(train_df)
                    self.predictor.save_checkpoint(self.model_checkpoint_path)

            prediction = self.predictor.predict_next_step(clean_df)
            step_delta = self._infer_sampling_delta(clean_df.index)
            self._append_prediction_log(clean_df, prediction, step_delta)
            self._reconcile_prediction_log_with_ground_truth(clean_df, step_delta)

            eval_input = clean_df.tail(max(eval_horizon, self.predictor.config.sequence_length + 1))
            ground_truth, pred_mean, pred_std = self.predictor.predict_historical_with_uncertainty(
                eval_input, mc_samples=int(mc_samples),
            )

            all_modules = self._get_all_modules(clean_df)
            all_features_per_module = {
                m: self._module_features(clean_df, m) for m in all_modules
            }
            all_features_flat = [f for feats in all_features_per_module.values() for f in feats]

            evaluation = self.evaluator.evaluate(
                ground_truth=ground_truth,
                prediction_mean=pred_mean,
                prediction_std=pred_std,
                selected_features=all_features_flat,
            )

            tab_forecast, tab_health, tab_log = st.tabs(
                ["Forecast", "Model Health", "Prediction Log"]
            )

            with tab_forecast:
                self._render_metrics_all_modules(clean_df, prediction, all_features_per_module)
                st.divider()
                self._render_all_module_charts(
                    evaluation, all_features_per_module,
                    selected_tail, clean_df, prediction,
                )

            with tab_health:
                self._render_quality_summary(report)
                st.divider()
                self._render_rolling_error_chart(evaluation)
                if show_advanced_metrics:
                    st.divider()
                    self._render_feature_rolling_error_chart(evaluation, all_features_flat)
                st.divider()
                self._render_residual_histogram(evaluation, all_features_flat)

            with tab_log:
                self._render_prediction_log_table(rows_to_show=int(log_rows_to_show))

        except (DataFetchError, DataQualityError, PredictorError, EvaluationError) as exc:
            st.error(str(exc))
            st.info("Pipeline halted safely due to upstream data or model constraints.")
        except Exception as exc:
            st.error(f"Unexpected runtime failure: {exc}")
            st.text(traceback.format_exc())

    # -------------------------------------------------------------------------
    # Module helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_all_modules(clean_df: pd.DataFrame) -> list[str]:
        modules = sorted({col.split("_")[0] for col in clean_df.columns if "_" in col})
        if not modules:
            raise EvaluationError("No module-feature channels found in data.")
        return modules

    def _module_features(self, dataframe: pd.DataFrame, module: str) -> list[str]:
        feature_order = ["Temp", "Humid", "CO2", "TVOC"]
        expected = [f"{module}_{feat}" for feat in feature_order]
        available = [col for col in expected if col in dataframe.columns]
        if not available:
            raise EvaluationError(f"No features found for module {module}.")
        return available

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _render_quality_summary(self, report: DataQualityReport) -> None:
        st.markdown('<div class="section-label">Data Quality Diagnostics</div>', unsafe_allow_html=True)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows In", report.input_rows)
        col_b.metric("Rows Out", report.output_rows)
        flatline_count = len(report.flatline_columns)
        dropped_count = len(report.dropped_columns)
        col_c.metric(
            "Flatline Channels", flatline_count,
            delta=f"{flatline_count} flagged" if flatline_count else None,
            delta_color="inverse",
        )
        col_d.metric(
            "Dropped Channels", dropped_count,
            delta=f"{dropped_count} removed" if dropped_count else None,
            delta_color="inverse",
        )
        outlier_total = int(np.sum(list(report.outlier_counts.values())))
        with st.expander("Details", expanded=False):
            if report.flatline_columns:
                st.markdown(f"**Flatline:** {', '.join(report.flatline_columns)}")
            if report.dropped_columns:
                st.markdown(f"**Dropped:** {', '.join(report.dropped_columns)}")
            st.markdown(f"**Outliers replaced (z-score):** {outlier_total}")

    def _render_metrics_all_modules(
        self,
        clean_df: pd.DataFrame,
        prediction: pd.Series,
        all_features_per_module: dict[str, list[str]],
    ) -> None:
        st.markdown('<div class="section-label">Current vs Predicted Next Step</div>', unsafe_allow_html=True)
        latest = clean_df.iloc[-1]

        for module, features in all_features_per_module.items():
            st.markdown(f'<div class="module-heading">{module}</div>', unsafe_allow_html=True)
            cols = st.columns(len(features))
            for idx, col in enumerate(features):
                current_val = float(latest[col])
                predicted_val = float(prediction[col])
                label = col.split("_", 1)[-1] if "_" in col else col
                cols[idx].metric(
                    label=label,
                    value=f"{current_val:.3f}",
                    delta=f"{predicted_val - current_val:+.3f}",
                )

    @staticmethod
    def _apply_chart_theme(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, color="#CBD5E1"),
                x=0, xanchor="left", pad=dict(b=6),
            ),
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#0F172A",
            font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#94A3B8"),
            height=height,
            margin=dict(l=50, r=16, t=52, b=44),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            ),
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(148,163,184,0.08)", showgrid=True),
            yaxis=dict(gridcolor="rgba(148,163,184,0.08)", showgrid=True),
        )
        return fig

    def _render_all_module_charts(
        self,
        evaluation: EvaluationArtifacts,
        all_features_per_module: dict[str, list[str]],
        selected_tail: int,
        clean_df: pd.DataFrame,
        next_prediction: pd.Series,
    ) -> None:
        step_delta = self._infer_sampling_delta(clean_df.index)
        forecast_origin_time = clean_df.index[-1]
        forecast_target_time = forecast_origin_time + step_delta

        for module, features in all_features_per_module.items():
            st.markdown(f'<div class="module-heading">{module}</div>', unsafe_allow_html=True)

            truth_tail = evaluation.ground_truth[features].tail(selected_tail)
            pred_tail = evaluation.prediction_mean[features].tail(selected_tail)
            std_tail = evaluation.prediction_std[features].tail(selected_tail)

            for row_start in range(0, len(features), 2):
                batch = features[row_start : row_start + 2]
                cols = st.columns(2)

                for col_idx, feature in enumerate(batch):
                    lower_ci, upper_ci = self.evaluator.build_prediction_interval(
                        pred_tail[feature], std_tail[feature],
                    )
                    origin_val = float(clean_df.iloc[-1][feature])
                    target_val = float(next_prediction[feature])
                    label = feature.split("_", 1)[-1] if "_" in feature else feature

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=truth_tail.index, y=truth_tail[feature],
                        mode="lines",
                        line=dict(color=_CHART_COLORS["truth"], width=2),
                        name="Ground Truth",
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_tail.index, y=pred_tail[feature],
                        mode="lines",
                        line=dict(color=_CHART_COLORS["pred"], width=2, dash="dash"),
                        name="Prediction",
                    ))
                    # CI band
                    fig.add_trace(go.Scatter(
                        x=pred_tail.index, y=upper_ci,
                        mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_tail.index, y=lower_ci,
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=_CHART_COLORS["ci_fill"],
                        name="95% CI", hoverinfo="skip",
                    ))
                    # t+1 forecast segment
                    fig.add_trace(go.Scatter(
                        x=[forecast_origin_time, forecast_target_time],
                        y=[origin_val, target_val],
                        mode="lines+markers",
                        line=dict(color=_CHART_COLORS["forecast_line"], width=2, dash="dot"),
                        marker=dict(
                            size=[7, 10],
                            symbol=["circle", "diamond"],
                            color=[_CHART_COLORS["origin_marker"], _CHART_COLORS["forecast_line"]],
                        ),
                        name="t+1 Forecast",
                    ))
                    fig.add_shape(
                        type="line",
                        x0=forecast_origin_time, x1=forecast_origin_time,
                        y0=0, y1=1, xref="x", yref="paper",
                        line=dict(width=1, dash="dot", color=_CHART_COLORS["separator"]),
                    )
                    fig.add_annotation(
                        x=forecast_origin_time, y=1, xref="x", yref="paper",
                        text="now", showarrow=False, yshift=8, xanchor="left",
                        font=dict(size=10, color="#475569"),
                    )

                    self._apply_chart_theme(fig, title=label, height=300)
                    fig.update_xaxes(tickformat="%H:%M\n%b %d")
                    fig.update_yaxes(title_text="Value", title_font=dict(size=10))

                    with cols[col_idx]:
                        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def _infer_sampling_delta(index: pd.Index) -> pd.Timedelta:
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return pd.Timedelta(minutes=3)
        diffs = index.to_series().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return pd.Timedelta(minutes=3)
        return pd.to_timedelta(diffs.median())

    def _render_rolling_error_chart(self, evaluation: EvaluationArtifacts) -> None:
        st.markdown('<div class="section-label">Rolling Error — MAE &amp; RMSE (window = 20 steps)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=evaluation.rolling_mae.index, y=evaluation.rolling_mae.values,
            mode="lines", line=dict(color=_CHART_COLORS["truth"], width=2),
            name="Rolling MAE",
        ))
        fig.add_trace(go.Scatter(
            x=evaluation.rolling_rmse.index, y=evaluation.rolling_rmse.values,
            mode="lines", line=dict(color=_CHART_COLORS["forecast_line"], width=2, dash="dash"),
            name="Rolling RMSE",
        ))
        self._apply_chart_theme(fig, height=300)
        fig.update_yaxes(title_text="Error", title_font=dict(size=10))
        st.plotly_chart(fig, width="stretch")

    def _render_feature_rolling_error_chart(
        self,
        evaluation: EvaluationArtifacts,
        all_features: list[str],
    ) -> None:
        st.markdown('<div class="section-label">Per-Feature Rolling MAE / RMSE</div>', unsafe_allow_html=True)
        rolling_window = int(self.evaluator._rolling_window)
        fig = go.Figure()

        for idx, feature in enumerate(all_features):
            color = _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]
            label = feature.split("_", 1)[-1] if "_" in feature else feature
            residual = evaluation.residuals[feature]
            rolling_mae = residual.abs().rolling(window=rolling_window, min_periods=1).mean()
            rolling_rmse = np.sqrt(
                np.square(residual).rolling(window=rolling_window, min_periods=1).mean()
            )
            fig.add_trace(go.Scatter(
                x=rolling_mae.index, y=rolling_mae.values,
                mode="lines", name=f"{label} MAE",
                legendgroup=feature, line=dict(color=color, width=2),
                hovertemplate="%{x}<br>MAE=%{y:.4f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=rolling_rmse.index, y=rolling_rmse.values,
                mode="lines", name=f"{label} RMSE",
                legendgroup=feature, line=dict(color=color, dash="dash", width=1.5),
                showlegend=False,
                hovertemplate="%{x}<br>RMSE=%{y:.4f}<extra></extra>",
            ))

        self._apply_chart_theme(fig, height=320)
        fig.update_yaxes(title_text="Rolling Error", title_font=dict(size=10))
        st.plotly_chart(fig, width="stretch")

    def _render_residual_histogram(
        self,
        evaluation: EvaluationArtifacts,
        all_features: list[str],
    ) -> None:
        st.markdown('<div class="section-label">Residual Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for idx, feature in enumerate(all_features):
            label = feature.split("_", 1)[-1] if "_" in feature else feature
            fig.add_trace(go.Histogram(
                x=evaluation.residuals[feature],
                opacity=0.65, name=label, nbinsx=40,
                marker_color=_COLOR_PALETTE[idx % len(_COLOR_PALETTE)],
            ))
        self._apply_chart_theme(fig, height=300)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Residual  e = y − ŷ", title_font=dict(size=10))
        fig.update_yaxes(title_text="Frequency", title_font=dict(size=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Near-Gaussian residuals centered near zero indicate the model has captured "
            "most systematic temporal structure; remaining error approximates white noise."
        )

    def _render_prediction_log_table(self, rows_to_show: int) -> None:
        st.markdown('<div class="section-label">Prediction Log — Long-Term Tracking</div>', unsafe_allow_html=True)

        if not self.prediction_log_path.exists():
            st.info("Prediction log not available yet.")
            return

        log_df = self._load_prediction_log_safe()
        if log_df.empty:
            st.info("Prediction log is empty.")
            return

        log_df = self._normalize_legacy_prediction_columns(log_df)

        for col_name in ("forecast_issue_time", "forecast_target_time", "ground_truth_time"):
            if col_name in log_df.columns:
                log_df[col_name] = pd.to_datetime(log_df[col_name], errors="coerce")

        if {"forecast_issue_time", "forecast_target_time"}.issubset(set(log_df.columns)):
            log_df = log_df.sort_values(
                by=["forecast_issue_time", "forecast_target_time"],
            ).drop_duplicates(
                subset=["forecast_issue_time", "forecast_target_time"], keep="last",
            )

        if "forecast_issue_time" in log_df.columns:
            log_tail = log_df.sort_values("forecast_issue_time", ascending=False).head(rows_to_show).copy()
        else:
            log_tail = log_df.tail(rows_to_show).copy()

        if "ground_truth_matched" in log_tail.columns:
            log_tail = log_tail.drop(columns=["ground_truth_matched"])

        st.caption(f"Showing {min(rows_to_show, len(log_tail))} of {len(log_df)} rows · {self.prediction_log_path.as_posix()}")
        st.dataframe(log_tail, width="stretch")

    # -------------------------------------------------------------------------
    # Checkpoint handling
    # -------------------------------------------------------------------------

    def _handle_checkpoint_load_error(self, error: PredictorError) -> None:
        message = str(error)
        if "preprocessing mode is incompatible" in message and self.model_checkpoint_path.exists():
            legacy_path = self._archive_incompatible_checkpoint(self.model_checkpoint_path)
            st.warning(
                f"Incompatible checkpoint archived to {legacy_path.as_posix()}. Retraining now."
            )
            st.session_state.checkpoint_loaded = True
            return
        st.warning(f"Checkpoint load failed ({message}). Retraining from current data.")

    @staticmethod
    def _archive_incompatible_checkpoint(checkpoint_path: Path) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        legacy_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.legacy_{timestamp}{checkpoint_path.suffix}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.replace(legacy_path)
        return legacy_path

    # -------------------------------------------------------------------------
    # Prediction log I/O
    # -------------------------------------------------------------------------

    def _append_prediction_log(
        self,
        clean_df: pd.DataFrame,
        prediction: pd.Series,
        step_delta: pd.Timedelta,
    ) -> None:
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
                subset=["forecast_issue_time", "forecast_target_time"], keep="last",
            )
        self._save_prediction_log(combined)

    def _reconcile_prediction_log_with_ground_truth(
        self,
        clean_df: pd.DataFrame,
        step_delta: pd.Timedelta,
    ) -> None:
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
            col[len("pred_"):] for col in log_df.columns if col.startswith("pred_")
        ]
        prediction_features = [f for f in prediction_features if f in clean_sorted.columns]
        if not prediction_features:
            return

        for feature in prediction_features:
            col = f"gt_matched_{feature}"
            if col not in log_df.columns:
                log_df[col] = False
            log_df[col] = self._coerce_to_bool_series(log_df[col])

        feature_match_columns = [f"gt_matched_{f}" for f in prediction_features]
        unresolved_mask = (
            (~log_df[feature_match_columns].all(axis=1))
            & log_df["forecast_target_time"].notna()
        )
        if not bool(unresolved_mask.any()):
            return

        for feature in prediction_features:
            for prefix in ("pred_", "gt_", "residual_", "abs_error_"):
                col_name = f"{prefix}{feature}"
                if col_name in log_df.columns:
                    log_df[col_name] = pd.to_numeric(log_df[col_name], errors="coerce")

        updated_any = False
        for row_idx in log_df.index[unresolved_mask]:
            target_time = log_df.at[row_idx, "forecast_target_time"]
            nearest_pos = clean_times.get_indexer([target_time], method="nearest")[0]
            if nearest_pos < 0:
                continue
            nearest_time = clean_times[nearest_pos]
            if abs(nearest_time - target_time) > tolerance:
                continue

            for feature in prediction_features:
                pred_val = pd.to_numeric(log_df.at[row_idx, f"pred_{feature}"], errors="coerce")
                gt_val = pd.to_numeric(clean_sorted.at[nearest_time, feature], errors="coerce")
                if pd.isna(pred_val) or pd.isna(gt_val):
                    continue
                residual = float(gt_val - pred_val)
                log_df.at[row_idx, f"gt_{feature}"] = float(gt_val)
                log_df.at[row_idx, f"gt_matched_{feature}"] = True
                log_df.at[row_idx, f"residual_{feature}"] = residual
                log_df.at[row_idx, f"abs_error_{feature}"] = abs(residual)

            log_df.at[row_idx, "ground_truth_time"] = pd.Timestamp(nearest_time).isoformat()
            log_df.at[row_idx, "ground_truth_matched"] = bool(
                log_df.loc[row_idx, feature_match_columns].all()
            )
            updated_any = True

        if updated_any:
            self._save_prediction_log(log_df)

    def _load_prediction_log_safe(self) -> pd.DataFrame:
        if not self.prediction_log_path.exists():
            return pd.DataFrame()
        try:
            with self.prediction_log_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
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
                records.append(dict(zip(header, row)))
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
                records.append(dict(zip(header, row)))
                repaired_schema = True
                continue
            if len(row) < len(header):
                records.append(dict(zip(header, row + [""] * (len(header) - len(row)))))
                continue
            records.append(dict(zip(header, row[: len(header)])))

        if not records:
            return pd.DataFrame(columns=header)

        dataframe = pd.DataFrame.from_records(records, columns=header)
        dataframe, repaired_values = self._repair_prediction_log_timestamps(dataframe)
        if repaired_schema or repaired_values:
            self._save_prediction_log(dataframe)
        return dataframe

    def _save_prediction_log(self, log_df: pd.DataFrame) -> None:
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(self.prediction_log_path, index=False)

    @staticmethod
    def _repair_prediction_log_timestamps(log_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
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
        inferred_delta = (
            (target_dt[valid_delta_mask] - issue_dt[valid_delta_mask]).median()
            if bool(valid_delta_mask.any())
            else pd.Timedelta(minutes=3)
        )
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
        pattern = re.compile(rf"^(pred|gt|residual|abs_error)_{re.escape(module)}_")
        matched_pattern = re.compile(rf"^gt_matched_{re.escape(module)}_")
        ordered = [col for col in columns if pattern.match(str(col))]
        return [col for col in ordered if not matched_pattern.match(str(col))]

    @staticmethod
    def _normalize_legacy_prediction_columns(log_df: pd.DataFrame) -> pd.DataFrame:
        upgraded = log_df.copy()
        for column in list(upgraded.columns):
            if re.match(r"^M(?:1|4|6|7|8|9|10|11)_(Temp|Humid|CO2|TVOC)$", str(column)):
                pred_col = f"pred_{column}"
                if pred_col not in upgraded.columns:
                    upgraded = upgraded.rename(columns={column: pred_col})
        return upgraded

    @staticmethod
    def _coerce_to_bool_series(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        normalized = series.astype(str).str.strip().str.lower()
        mapped = pd.Series(False, index=series.index, dtype=bool)
        mapped[normalized.isin({"true", "1", "yes", "y", "t"})] = True
        return mapped


def main() -> None:
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
