"""Visualization engine — generates chart images for Telegram bot commands."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .predictor import DeepTimeSeriesPredictor

# ─── Theme ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#8b949e",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "legend.framealpha": 0.0,
    "font.family":       "DejaVu Sans",
})

# ─── Constants ────────────────────────────────────────────────────────────────

NODES = ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]

PALETTE = [
    "#4cc9f0", "#f72585", "#7bed9f", "#ffd32a",
    "#a29bfe", "#fd9644", "#00d2d3", "#ff6b81",
]

# metric key -> (column suffix, display unit)
METRIC_META: dict[str, tuple[str, str]] = {
    "temp":  ("_Temp",  "°C"),
    "humid": ("_Humid", "%"),
    "co2":   ("_CO2",   "ppm"),
    "tvoc":  ("_TVOC",  "ppb"),
}

# Alert thresholds mirrored from api_service — hi bound only.
_THRESH_HI: dict[str, float | None] = {
    "_Temp":  33.0,
    "_Humid": 75.0,
    "_CO2":   1200.0,
    "_TVOC":  300.0,
}

# Fallback node positions on a 0–7.5 × 0–6.8 grid.
_NODE_GRID_POS: dict[str, tuple[float, float]] = {
    "M1":  (1.0, 5.5),
    "M4":  (4.5, 5.5),
    "M6":  (1.0, 3.0),
    "M7":  (3.0, 3.0),
    "M8":  (5.5, 3.0),
    "M9":  (1.0, 0.8),
    "M10": (3.5, 0.8),
    "M11": (6.5, 0.8),
}

_ROOMS: list[tuple[str, float, float, float, float]] = [
    ("Lab A (M1)",    0.1, 4.6, 2.5, 1.8),
    ("Office (M4)",   3.2, 4.6, 2.9, 1.8),
    ("Hall (M6)",     0.1, 1.9, 2.0, 2.3),
    ("Room (M7)",     2.5, 1.9, 1.8, 2.3),
    ("Lab B (M8)",    4.5, 1.9, 2.9, 2.3),
    ("Storage (M9)",  0.1, 0.1, 1.8, 1.5),
    ("Server (M10)",  2.4, 0.1, 2.4, 1.5),
    ("Meeting (M11)", 5.2, 0.1, 2.2, 1.5),
]

_HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "campus_heat",
    ["#2196f3", "#00c853", "#ffeb3b", "#ff9800", "#f44336"],
    N=256,
)

# Fixed value range per metric — keeps colors consistent across time.
_METRIC_VRANGE: dict[str, tuple[float, float]] = {
    "temp":  (25.0, 34.0),
    "humid": (40.0, 80.0),
    "co2":   (400.0, 1200.0),
    "tvoc":  (0.0,  300.0),
}

_ICT = timezone(timedelta(hours=7))

# Project root: three levels above this file (src/iot_digital_twin/viz_engine.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _now_ict() -> datetime:
    return datetime.now(_ICT)


def _footer(fig: plt.Figure) -> None:
    fig.text(
        0.99, 0.01,
        f"Generated {_now_ict().strftime('%H:%M  %d/%m/%Y')}  |  UEH Campus V",
        ha="right", va="bottom", fontsize=7, color="#484f58",
    )


def _to_bytes(fig: plt.Figure, dpi: int = 130) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _filter_window(df: pd.DataFrame, range_str: str) -> pd.DataFrame:
    """Return rows within the requested time window using real timestamps."""
    deltas = {"hour": timedelta(hours=1), "day": timedelta(hours=24), "week": timedelta(days=7)}
    delta = deltas.get(range_str)
    if delta is None:
        return df
    now = _now_ict()
    cutoff = now - delta
    if df.index.tz is None:
        # index not tz-aware: attach ICT and filter
        idx_ict = df.index.tz_localize(_ICT, ambiguous="NaT", nonexistent="NaT")
        mask = idx_ict >= cutoff
        return df.loc[mask]
    return df[df.index >= cutoff]


def _metric_cols(df: pd.DataFrame, col_suffix: str) -> list[str]:
    return [c for c in df.columns if c.endswith(col_suffix)]


def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def _idw(
    gx: np.ndarray,
    gy: np.ndarray,
    positions: dict[str, tuple[float, float]],
    values: dict[str, float],
    power: int = 2,
) -> np.ndarray:
    gz = np.zeros_like(gx, dtype=float)
    w_sum = np.zeros_like(gx, dtype=float)
    for node, (px, py) in positions.items():
        if node not in values:
            continue
        dist = np.hypot(gx - px, gy - py) + 1e-6
        w = 1.0 / dist**power
        gz += w * values[node]
        w_sum += w
    return gz / np.where(w_sum > 0, w_sum, 1.0)


def _node_marker_color(val: float, vmin: float, vmax: float) -> str:
    span = vmax - vmin
    if val > vmin + span * 0.65:
        return "#ff4444"
    if val > vmin + span * 0.35:
        return "#ffd32a"
    return "#4cc9f0"


# ─── Public API ───────────────────────────────────────────────────────────────

def chart(df: pd.DataFrame, range_str: str, metric: str) -> io.BytesIO:
    """Shaded band (min/max across all nodes) + mean line for one metric."""
    col_suffix, unit = METRIC_META[metric]
    window_df = _filter_window(df, range_str)

    cols = _metric_cols(window_df, col_suffix)
    if not cols or window_df.empty:
        raise ValueError(f"No data for metric={metric} range={range_str}.")

    numeric = window_df[cols].apply(pd.to_numeric, errors="coerce")
    mean_s = _smooth(numeric.mean(axis=1))
    min_s  = _smooth(numeric.min(axis=1))
    max_s  = _smooth(numeric.max(axis=1))

    if range_str == "week":
        mean_s = mean_s.resample("30min").mean().dropna()
        min_s  = min_s.resample("30min").mean().dropna()
        max_s  = max_s.resample("30min").mean().dropna()

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")

    idx = mean_s.index
    ax.fill_between(idx, min_s, max_s, alpha=0.18, color="#4cc9f0",
                    label="All nodes — min/max band")
    ax.plot(idx, mean_s, color="#4cc9f0", linewidth=2.2,
            label=f"Mean  ({mean_s.iloc[-1]:.1f} {unit})", zorder=3)

    for col, color in zip(cols, PALETTE):
        node = col.replace(col_suffix, "")
        s = _smooth(pd.to_numeric(window_df[col], errors="coerce").dropna())
        if not s.empty:
            ax.plot(s.index, s, linewidth=0.9, alpha=0.45, color=color, label=node)

    fmt = "%H:%M" if range_str == "hour" else "%d/%m %H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax.set_ylabel(unit, fontsize=9)
    ax.set_title(
        f"/chart_{range_str}_{metric}  |  All Nodes  |  {_now_ict().strftime('%H:%M')}",
        color="#e6edf3", fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7.5, frameon=True, facecolor="#161b22", edgecolor="#30363d",
              loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4cc9f0")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(labelsize=8)

    _footer(fig)
    plt.tight_layout()
    return _to_bytes(fig)


def compare(df: pd.DataFrame, node_a: str, node_b: str) -> io.BytesIO:
    """All-metric comparison for two nodes over last 3 hours (2x2 subplots)."""
    now = _now_ict()
    cutoff = now - timedelta(hours=3)
    window_df = (
        df[df.index >= cutoff]
        if df.index.tz is not None
        else df[df.index.tz_localize(_ICT, ambiguous="NaT") >= cutoff]
        if False  # avoid modifying index; just use tail as fallback
        else df.tail(60)
    )
    # simpler: use _filter_window with a "3h" pseudo-range, or just tail
    window_df = _filter_window(df, "day").tail(60)  # last 60 rows ≈ 3h at 3-min cadence

    COLOR_A, COLOR_B = "#4cc9f0", "#ff6b81"
    METRICS = [
        ("Temp",  "°C",  "#f85149"),
        ("Humid", "%",   "#3fb950"),
        ("CO2",   "ppm", "#e3b341"),
        ("TVOC",  "ppb", "#a371f7"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"/compare_{node_a}_{node_b}  |  Last 3 Hours  |  {_now_ict().strftime('%H:%M')}",
        color="#e6edf3", fontsize=13, fontweight="bold", y=0.99,
    )

    for ax, (col_name, unit, accent) in zip(axes.flat, METRICS):
        col_a = f"{node_a}_{col_name}"
        col_b = f"{node_b}_{col_name}"

        a_raw = pd.to_numeric(window_df[col_a], errors="coerce") if col_a in window_df.columns else pd.Series(dtype=float)
        b_raw = pd.to_numeric(window_df[col_b], errors="coerce") if col_b in window_df.columns else pd.Series(dtype=float)
        a_vals = _smooth(a_raw)
        b_vals = _smooth(b_raw)
        idx = window_df.index

        label_a = f"{node_a}  ({a_vals.iloc[-1]:.1f} {unit})" if not a_vals.empty else node_a
        label_b = f"{node_b}  ({b_vals.iloc[-1]:.1f} {unit})" if not b_vals.empty else node_b

        ax.plot(idx, a_vals, color=COLOR_A, linewidth=2.0, label=label_a, zorder=3)
        ax.plot(idx, b_vals, color=COLOR_B, linewidth=2.0, label=label_b, zorder=3)

        if not a_vals.empty and not b_vals.empty:
            ax.fill_between(idx, a_vals, b_vals,
                            where=b_vals >= a_vals, alpha=0.12, color=COLOR_B)
            ax.fill_between(idx, a_vals, b_vals,
                            where=b_vals < a_vals,  alpha=0.12, color=COLOR_A)
            d = b_vals.iloc[-1] - a_vals.iloc[-1]
            sign = "+" if d >= 0 else ""
            mid_y = (a_vals.iloc[-1] + b_vals.iloc[-1]) / 2
            ax.annotate(
                f"Delta: {sign}{d:.1f} {unit}",
                xy=(idx[-1], mid_y),
                xytext=(-90, 0), textcoords="offset points",
                fontsize=8, color="#e6edf3", alpha=0.85,
                bbox=dict(fc="#21262d", ec="#30363d", boxstyle="round,pad=0.3"),
            )

        ax.set_title(f"{col_name} ({unit})", color=accent, fontsize=9, fontweight="bold", pad=5)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(accent)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("#30363d")
        ax.tick_params(labelsize=7.5)

    _footer(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return _to_bytes(fig)


def predict(
    df: pd.DataFrame,
    node: str,
    metric: str,
    predictor: "DeepTimeSeriesPredictor",
) -> io.BytesIO:
    """Historical 1-step predictions + MC uncertainty + next-step forecast."""
    col_suffix, unit = METRIC_META[metric]
    col = f"{node}{col_suffix}"

    if col not in df.columns:
        raise ValueError(f"Column {col} not in dataframe.")

    # Use last 2h (40 rows) for the rolling evaluation
    tail = df.tail(40)
    ground_truth, pred_mean, pred_std = predictor.predict_historical_with_uncertainty(
        tail, mc_samples=20
    )

    gt_col   = ground_truth[col]
    mean_col = pred_mean[col]
    std_col  = pred_std[col]

    # Single next-step forecast
    next_pred = predictor.predict_next_step(df)
    next_val  = float(next_pred[col]) if col in next_pred.index else None
    next_std  = float(std_col.iloc[-1]) if not std_col.empty else 0.0
    last_ts   = pd.Timestamp(gt_col.index[-1])
    next_ts   = last_ts + timedelta(minutes=3)

    fig, (ax_main, ax_re) = plt.subplots(
        2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor("#0d1117")

    # ── Main chart ─────────────────────────────────────────────────────────
    ax_main.set_facecolor("#161b22")
    ax_main.plot(gt_col.index, gt_col.values, color="#4cc9f0", linewidth=2.0,
                 label=f"{col} — Observed", zorder=4)
    ax_main.plot(mean_col.index, mean_col.values, color="#f85149", linewidth=1.5,
                 linestyle="--", alpha=0.85, label="1-step prediction (MC mean)", zorder=3)
    ax_main.fill_between(
        mean_col.index,
        mean_col.values - std_col.values,
        mean_col.values + std_col.values,
        color="#f85149", alpha=0.15, label="±1σ (MC dropout)", zorder=2,
    )

    if next_val is not None:
        ax_main.errorbar(
            [next_ts], [next_val], yerr=[[next_std], [next_std]],
            fmt="o", color="#ffd32a", markersize=8, capsize=5, zorder=6,
            label=f"Next step: {next_val:.1f} {unit}",
        )
        ax_main.annotate(
            f"+3 min\n{next_val:.1f} {unit}",
            xy=(next_ts, next_val),
            xytext=(10, 14), textcoords="offset points",
            fontsize=9, color="#ffd32a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ffd32a", lw=1.2),
            bbox=dict(fc="#1e1e2e", ec="#ffd32a", boxstyle="round,pad=0.35", lw=0.8),
        )

    thresh_hi = _THRESH_HI.get(col_suffix)
    if thresh_hi is not None:
        ax_main.axhline(thresh_hi, color="#ff4444", linewidth=0.9, linestyle="--",
                        alpha=0.6, label=f"Alert threshold ({thresh_hi} {unit})")

    ax_main.set_title(
        f"/predict_{node}_{metric}  |  MC Dropout Uncertainty  |  {_now_ict().strftime('%H:%M')}",
        color="#e6edf3", fontsize=12, fontweight="bold", pad=10,
    )
    ax_main.set_ylabel(unit, fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.4)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_main.legend(loc="upper left", fontsize=8.5, frameon=True,
                   facecolor="#1a1d27", edgecolor="#30363d")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_color("#f85149")
    ax_main.spines["left"].set_linewidth(1.5)
    ax_main.tick_params(labelsize=8)
    ax_main.text(
        0.01, -0.04,
        "Uncertainty via MC dropout (20 passes). Single-step look-ahead only — not a long-horizon forecast.",
        transform=ax_main.transAxes, fontsize=7, color="#484f58",
    )

    # ── Uncertainty subplot ────────────────────────────────────────────────
    ax_re.set_facecolor("#0f1318")
    ax_re.fill_between(std_col.index, std_col.values, color="#a371f7", alpha=0.4)
    ax_re.plot(std_col.index, std_col.values, color="#a371f7", linewidth=1.2,
               label="Predictive σ (MC dropout)")
    ax_re.set_ylabel(f"σ ({unit})", fontsize=7.5)
    ax_re.set_ylim(0, None)
    ax_re.grid(True, linestyle="--", alpha=0.35)
    ax_re.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_re.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax_re.spines["top"].set_visible(False)
    ax_re.spines["right"].set_visible(False)
    ax_re.spines["left"].set_color("#a371f7")
    ax_re.spines["left"].set_linewidth(1.4)
    ax_re.spines["bottom"].set_color("#30363d")
    ax_re.tick_params(labelsize=7.5)

    _footer(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return _to_bytes(fig)


def heatmap(
    df: pd.DataFrame,
    metric: str,
    coords_path: Path | None = None,
    image_path: Path | None = None,
) -> io.BytesIO:
    """IDW spatial heatmap. Uses campus image overlay if files exist, else floor-plan grid."""
    col_suffix, unit = METRIC_META[metric]
    if df.empty:
        raise ValueError("DataFrame is empty.")

    latest = df.iloc[-1]
    node_values: dict[str, float] = {}
    for node in NODES:
        col = f"{node}{col_suffix}"
        if col in df.columns:
            v = latest.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(float(v))):
                node_values[node] = float(v)

    if not node_values:
        raise ValueError(f"No node values available for metric={metric}.")

    vmin, vmax = _METRIC_VRANGE.get(metric, (min(node_values.values()), max(node_values.values())))

    # Try campus image overlay
    if coords_path and coords_path.exists() and image_path and image_path.exists():
        try:
            import matplotlib.image as mpimg
            raw_coords: dict[str, list[int]] = json.loads(coords_path.read_text(encoding="utf-8"))
            pixel_pos = {n: (float(xy[0]), float(xy[1])) for n, xy in raw_coords.items()}
            campus_img = mpimg.imread(str(image_path))
            return _heatmap_image(campus_img, pixel_pos, node_values, metric, unit, vmin, vmax)
        except Exception:
            pass  # fall through to grid

    return _heatmap_grid(node_values, metric, unit, vmin, vmax)


def _heatmap_image(
    img: np.ndarray,
    pixel_pos: dict[str, tuple[float, float]],
    node_values: dict[str, float],
    metric: str,
    unit: str,
    vmin: float,
    vmax: float,
) -> io.BytesIO:
    h, w = img.shape[:2]
    visible_pos = {n: pos for n, pos in pixel_pos.items() if n in node_values}

    # IDW on a coarse grid, then bilinear-interpolated by imshow — smoother result.
    res = 300
    gx, gy = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))
    gz = _idw(gx, gy, visible_pos, node_values)

    fig, ax = plt.subplots(figsize=(15, 8.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.imshow(img, origin="upper", zorder=1)
    ax.imshow(
        gz,
        extent=[0, w, h, 0],   # left, right, bottom(=img_h), top(=0) — origin='upper'
        origin="upper",
        cmap=_HEAT_CMAP,
        vmin=vmin, vmax=vmax,
        alpha=0.4,
        interpolation="bilinear",
        zorder=2,
    )

    for node, (px, py) in visible_pos.items():
        val = node_values[node]
        ax.scatter(px, py, s=120, color="white", edgecolors="black",
                   linewidths=1.5, zorder=10)
        ax.text(px, py + 22, f"{node}  {val:.1f} {unit}",
                color="white", fontsize=9, fontweight="bold", ha="center",
                bbox=dict(facecolor="black", alpha=0.65, pad=2, linewidth=0),
                zorder=11)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=_HEAT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, aspect=35)
    cbar.set_label(f"{metric.capitalize()} ({unit})", color="#8b949e", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=7.5)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#8b949e")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(
        f"/heatmap_{metric}  |  Campus Overlay  |  {_now_ict().strftime('%H:%M')}",
        color="#e6edf3", fontsize=13, fontweight="bold", pad=14,
    )
    fig.text(
        0.01, 0.01,
        "Spatial interpolation (IDW) — indicative only. Mixed indoor/outdoor nodes may reduce accuracy.",
        ha="left", va="bottom", fontsize=7, color="#484f58",
    )
    _footer(fig)
    plt.tight_layout()
    return _to_bytes(fig, dpi=130)


def _heatmap_grid(
    node_values: dict[str, float],
    metric: str,
    unit: str,
    vmin: float,
    vmax: float,
) -> io.BytesIO:
    gx, gy = np.meshgrid(np.linspace(0, 7.5, 300), np.linspace(0, 6.8, 300))
    gz = _idw(gx, gy, _NODE_GRID_POS, node_values)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.pcolormesh(gx, gy, gz, cmap=_HEAT_CMAP, vmin=vmin, vmax=vmax,
                       shading="gouraud", alpha=0.88, zorder=1)

    for label, rx, ry, rw, rh in _ROOMS:
        rect = patches.FancyBboxPatch(
            (rx, ry), rw, rh, boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor="#e6edf3", facecolor="none",
            linestyle="--", zorder=3, alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(rx + rw / 2, ry + rh / 2 + 0.2, label,
                ha="center", va="center", fontsize=7.5,
                color="#ffffff", fontweight="bold", zorder=5,
                bbox=dict(fc="#00000066", ec="none", boxstyle="round,pad=0.15"))

    for node, (px, py) in _NODE_GRID_POS.items():
        if node not in node_values:
            continue
        val = node_values[node]
        color = _node_marker_color(val, vmin, vmax)
        ax.scatter(px, py, s=180, color=color, zorder=6, edgecolors="#ffffff", linewidths=1.2)
        ax.text(px, py - 0.42, f"{val:.1f} {unit}",
                ha="center", va="top", fontsize=8, color="#ffffff", fontweight="bold", zorder=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label(f"{metric.capitalize()} ({unit})", color="#8b949e", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=7.5)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#8b949e")

    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"/heatmap_{metric}  |  Floor Plan  |  {_now_ict().strftime('%H:%M')}",
        color="#e6edf3", fontsize=12, fontweight="bold", pad=12,
    )

    span = vmax - vmin
    for label, color in [
        (f"Low  (< {vmin + span * 0.35:.1f} {unit})", "#4cc9f0"),
        (f"Mid  ({vmin + span * 0.35:.1f}–{vmin + span * 0.65:.1f} {unit})", "#ffd32a"),
        (f"High (> {vmin + span * 0.65:.1f} {unit})", "#ff4444"),
    ]:
        ax.scatter([], [], color=color, s=80, label=label)
    ax.legend(loc="upper right", fontsize=8, frameon=True,
              facecolor="#161b22", edgecolor="#30363d")

    _footer(fig)
    plt.tight_layout()
    return _to_bytes(fig, dpi=140)


def rank(df: pd.DataFrame, metric: str) -> str:
    """HTML-formatted ranking of all nodes by current metric value."""
    col_suffix, unit = METRIC_META[metric]
    if df.empty:
        return "No data available."

    latest = df.iloc[-1]
    scored: list[tuple[str, float]] = []
    for node in NODES:
        col = f"{node}{col_suffix}"
        if col in df.columns:
            v = latest.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(float(v))):
                scored.append((node, float(v)))

    if not scored:
        return f"No node data for metric {metric}."

    scored.sort(key=lambda x: x[1], reverse=True)
    medals = ["\U0001f947", "\U0001f948", "\U0001f949"]  # gold, silver, bronze
    now_str = _now_ict().strftime("%H:%M ICT \u2014 %d/%m/%Y")

    lines = [f"<b>[RANK] {metric.upper()} \u2014 {now_str}</b>\n"]
    for i, (node, val) in enumerate(scored):
        prefix = medals[i] if i < 3 else f"  {i + 1}."
        suffix = "  <i>(highest)</i>" if i == 0 else "  <i>(lowest)</i>" if i == len(scored) - 1 else ""
        lines.append(f"{prefix} <b>{node}</b>: {val:.1f} {unit}{suffix}")

    return "\n".join(lines)
