"""Daily heatmap video generation — core logic used by api_service and CLI."""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v2 as iio

from . import viz_engine
from .data_fetcher import DataFetcher, DataFetcherConfig

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

ICT           = timezone(timedelta(hours=7))
DEFAULT_FPS   = 6
VIDEO_QUALITY = 7
METRICS       = ["temp", "humid", "co2", "tvoc"]
NODES         = viz_engine.NODES
PALETTE       = viz_engine.PALETTE
NODE_NAMES    = viz_engine.NODE_NAMES
RETENTION_DAYS = 7
HF_ASSET_REPO  = "Nhatminh1234/iot-campus-assets"
HF_VIDEO_PREFIX = "videos/"

CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSnBhE8u7fdXKEooOlqgGYtSZLeUxrQUu9e_q6MnMrGbakxXESMYVf0utORhEG3pEqWffGhX6J-V2cC"
    "/pub?output=csv"
)

_METRIC_META   = viz_engine.METRIC_META
_METRIC_VRANGE: dict[str, tuple[float, float]] = {
    "temp":  (25.0, 34.0),
    "humid": (40.0, 80.0),
    "co2":   (400.0, 1200.0),
    "tvoc":  (0.0,  300.0),
}
_HEAT_CMAP = viz_engine._HEAT_CMAP
_GRID_POS  = viz_engine._NODE_GRID_POS

_PROJECT_ROOT      = Path(__file__).parent.parent.parent
_COORDS_CANDIDATES = [
    (_PROJECT_ROOT / "node_coords_v1.json", _PROJECT_ROOT / "campus_3d_1.png"),
    (_PROJECT_ROOT / "node_coords_v0.json", _PROJECT_ROOT / "campus_3d.png"),
]

# ── IDW ────────────────────────────────────────────────────────────────────────

def _idw(
    gx: np.ndarray,
    gy: np.ndarray,
    positions: dict[str, tuple[float, float]],
    values: dict[str, float],
    power: float = 2.0,
) -> np.ndarray:
    gz    = np.zeros_like(gx, dtype=float)
    w_sum = np.zeros_like(gx, dtype=float)
    for node, (px, py) in positions.items():
        if node not in values:
            continue
        d2 = (gx - px) ** 2 + (gy - py) ** 2
        d2 = np.where(d2 < 1e-10, 1e-10, d2)
        w      = 1.0 / d2 ** (power / 2)
        gz    += w * values[node]
        w_sum += w
    return np.where(w_sum > 0, gz / w_sum, 0.0)


def _precompute_idw(
    df_day:     pd.DataFrame,
    campus_img: np.ndarray | None,
    pixel_pos:  dict[str, tuple[float, float]] | None,
) -> dict[str, list[tuple[np.ndarray, dict[str, float]]]]:
    if campus_img is not None and pixel_pos is not None:
        h, w   = campus_img.shape[:2]
        gx, gy = np.meshgrid(np.linspace(0, w, 180), np.linspace(0, h, 180))
    else:
        gx, gy = np.meshgrid(np.linspace(0, 7.5, 200), np.linspace(0, 6.8, 200))

    result: dict[str, list[tuple[np.ndarray, dict[str, float]]]] = {}
    for metric in METRICS:
        col_suffix = _METRIC_META[metric][0]
        entries: list[tuple[np.ndarray, dict[str, float]]] = []
        for row_idx in range(len(df_day)):
            row = df_day.iloc[row_idx]
            node_values: dict[str, float] = {}
            for node in NODES:
                col = f"{node}{col_suffix}"
                if col in df_day.columns:
                    v = row.get(col)
                    if v is not None and not (isinstance(v, float) and np.isnan(float(v))):
                        node_values[node] = float(v)
            if campus_img is not None and pixel_pos is not None:
                visible = {n: p for n, p in pixel_pos.items() if n in node_values}
                gz = _idw(gx, gy, visible, node_values).astype(np.float32)
            else:
                gz = _idw(gx, gy, _GRID_POS, node_values).astype(np.float32)
            entries.append((gz, node_values))
        result[metric] = entries
    return result


# ── Draw helpers ───────────────────────────────────────────────────────────────

def _draw_heatmap(
    ax: plt.Axes,
    gz: np.ndarray,
    node_values: dict[str, float],
    metric: str,
    unit: str,
    vmin: float,
    vmax: float,
    campus_img: np.ndarray | None,
    pixel_pos: dict[str, tuple[float, float]] | None,
) -> None:
    ax.set_facecolor("#0d1117")
    ax.axis("off")
    if campus_img is not None and pixel_pos is not None:
        h, w    = campus_img.shape[:2]
        visible = {n: p for n, p in pixel_pos.items() if n in node_values}
        ax.imshow(campus_img, origin="upper", zorder=1)
        ax.imshow(gz, extent=[0, w, h, 0], origin="upper",
                  cmap=_HEAT_CMAP, vmin=vmin, vmax=vmax,
                  alpha=0.45, interpolation="bilinear", zorder=2)
        for node, (px, py) in visible.items():
            val  = node_values[node]
            name = NODE_NAMES.get(node, node)
            ax.scatter(px, py, s=55, color="white", edgecolors="black",
                       linewidths=1.0, zorder=10)
            ax.text(px, py + 20, f"{node} — {name}\n{val:.1f} {unit}",
                    color="white", fontsize=6, fontweight="bold", ha="center",
                    bbox=dict(facecolor="#000000aa", pad=1.5, linewidth=0), zorder=11)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    else:
        gx, gy = np.meshgrid(np.linspace(0, 7.5, 200), np.linspace(0, 6.8, 200))
        ax.pcolormesh(gx, gy, gz, cmap=_HEAT_CMAP, vmin=vmin, vmax=vmax,
                      shading="gouraud", alpha=0.88, zorder=1)
        for node, (px, py) in _GRID_POS.items():
            if node not in node_values:
                continue
            val = node_values[node]
            ax.scatter(px, py, s=70, color="white", edgecolors="black",
                       linewidths=0.9, zorder=6)
            ax.text(px, py - 0.32, f"{node}\n{val:.1f} {unit}",
                    ha="center", va="top", fontsize=6, color="white",
                    fontweight="bold", zorder=7)
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 6.8)
    ax.set_title(metric.upper(), color="#e6edf3", fontsize=9, fontweight="bold", pad=3)


def _draw_chart(
    ax: plt.Axes,
    df_day: pd.DataFrame,
    metric: str,
    unit: str,
    cursor_ts: pd.Timestamp,
    vmin: float,
    vmax: float,
) -> None:
    col_suffix = _METRIC_META[metric][0]
    span       = vmax - vmin
    ax.set_facecolor("#161b22")
    ax.set_xlim(df_day.index[0], df_day.index[-1])
    ax.set_ylim(vmin - span * 0.04, vmax + span * 0.04)
    for i, node in enumerate(NODES):
        col = f"{node}{col_suffix}"
        if col in df_day.columns:
            ax.plot(df_day.index, df_day[col], color=PALETTE[i],
                    linewidth=0.85, alpha=0.8)
    ax.axvline(cursor_ts, color="#ffffff", linewidth=1.1, alpha=0.9, zorder=10)
    ax.tick_params(colors="#8b949e", labelsize=5.5, length=2, pad=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
        spine.set_linewidth(0.6)
    ax.grid(True, color="#21262d", linewidth=0.4, alpha=0.6)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=5)


# ── Frame ──────────────────────────────────────────────────────────────────────

def _render_frame(
    df_day:      pd.DataFrame,
    ts:          pd.Timestamp,
    frame_idx:   int,
    precomputed: dict[str, list[tuple[np.ndarray, dict[str, float]]]],
    campus_img:  np.ndarray | None,
    pixel_pos:   dict[str, tuple[float, float]] | None,
) -> np.ndarray:
    fig = plt.figure(figsize=(24, 13), facecolor="#0d1117", dpi=96)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.08, wspace=0.04,
                            left=0.01, right=0.99, top=0.92, bottom=0.02)
    _INSET   = [0.02, 0.03, 0.38, 0.26]
    hm_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, metric in enumerate(METRICS):
        r, c             = hm_slots[i]
        col_suffix, unit = _METRIC_META[metric]
        vmin, vmax       = _METRIC_VRANGE[metric]
        ax               = fig.add_subplot(gs[r, c])
        gz, node_values  = precomputed[metric][frame_idx]
        _draw_heatmap(ax, gz, node_values, metric, unit, vmin, vmax, campus_img, pixel_pos)
        ax_inset = ax.inset_axes(_INSET)
        ax_inset.patch.set_facecolor("#0d1117")
        ax_inset.patch.set_alpha(0.82)
        _draw_chart(ax_inset, df_day, metric, unit, ts, vmin, vmax)

    fig.suptitle(
        f"UEH Campus V  —  Daily Heatmap  |  {ts.strftime('%H:%M ICT   %d/%m/%Y')}",
        color="#e6edf3", fontsize=13, fontweight="bold", y=0.965,
    )
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    frame = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(h_px, w_px, 4)[:, :, :3]
        .copy()
    )
    plt.close(fig)
    return frame


# ── Data loading ───────────────────────────────────────────────────────────────

def load_day(target_date: date, csv_url: str = CSV_URL) -> pd.DataFrame:
    fetcher = DataFetcher(DataFetcherConfig(csv_url=csv_url))
    df      = fetcher.fetch()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("Asia/Bangkok")
    else:
        df.index = df.index.tz_convert("Asia/Bangkok")
    day_start = pd.Timestamp(target_date, tz="Asia/Bangkok")
    day_end   = day_start + pd.Timedelta(days=1)
    df_day    = df[(df.index >= day_start) & (df.index < day_end)].copy()
    if df_day.empty:
        raise ValueError(f"No data for {target_date}.")
    df_day = df_day.resample("30min").mean()
    df_day = df_day.interpolate(method="time", limit=6)
    logger.info("Loaded %d frames for %s", len(df_day), target_date)
    return df_day


# ── HF upload / prune ──────────────────────────────────────────────────────────

def _upload_to_hf(local_path: Path) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import upload_file
        repo_path = f"{HF_VIDEO_PREFIX}{local_path.name}"
        upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_path,
                    repo_id=HF_ASSET_REPO, repo_type="dataset", token=token)
        logger.info("Uploaded video -> hf://datasets/%s/%s", HF_ASSET_REPO, repo_path)
    except Exception as exc:
        logger.warning("HF upload failed: %s", exc)


def _prune_hf_videos() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import delete_file, list_repo_files
        cutoff = datetime.now(ICT) - timedelta(days=RETENTION_DAYS)
        for repo_path in list_repo_files(HF_ASSET_REPO, repo_type="dataset", token=token):
            if not repo_path.startswith(HF_VIDEO_PREFIX) or not repo_path.endswith(".mp4"):
                continue
            try:
                ds = repo_path.split("/")[-1].replace("daily_heatmap_", "").replace(".mp4", "")
                if datetime.fromisoformat(ds).replace(tzinfo=ICT) < cutoff:
                    delete_file(repo_path, repo_id=HF_ASSET_REPO,
                                repo_type="dataset", token=token)
                    logger.info("Pruned HF video: %s", repo_path)
            except ValueError:
                pass
    except Exception as exc:
        logger.warning("HF prune failed: %s", exc)


def _prune_local_videos(output_dir: Path) -> None:
    cutoff = datetime.now(ICT) - timedelta(days=RETENTION_DAYS)
    for f in output_dir.glob("daily_heatmap_*.mp4"):
        try:
            ds = f.stem.replace("daily_heatmap_", "")
            if datetime.fromisoformat(ds).replace(tzinfo=ICT) < cutoff:
                f.unlink()
                logger.info("Pruned local video: %s", f.name)
        except ValueError:
            pass


# ── Public API ─────────────────────────────────────────────────────────────────

def video_path(target_date: date, output_dir: Path = _PROJECT_ROOT) -> Path:
    """Canonical local path for a given date's video."""
    return output_dir / f"daily_heatmap_{target_date}.mp4"


def generate_video(
    target_date: date,
    output_path: Path | None = None,
    fps:         int = DEFAULT_FPS,
    csv_url:     str = CSV_URL,
) -> Path:
    """Generate daily heatmap video, upload to HF, prune old files."""
    if output_path is None:
        output_path = video_path(target_date)

    viz_engine._ensure_campus_images()

    campus_img: np.ndarray | None = None
    pixel_pos:  dict[str, tuple[float, float]] | None = None
    for coords_p, img_p in _COORDS_CANDIDATES:
        if coords_p.exists() and img_p.exists():
            raw        = json.loads(coords_p.read_text(encoding="utf-8"))
            pixel_pos  = {n: (float(xy[0]), float(xy[1])) for n, xy in raw.items()}
            campus_img = mpimg.imread(str(img_p))
            logger.info("Campus overlay: %s", img_p.name)
            break

    df_day     = load_day(target_date, csv_url)
    timestamps = list(df_day.index)
    n          = len(timestamps)

    logger.info("Precomputing IDW grids (%d × %d)...", n, len(METRICS))
    precomputed = _precompute_idw(df_day, campus_img, pixel_pos)

    logger.info("Rendering + encoding %d frames -> %s", n, output_path)
    writer = iio.get_writer(str(output_path), fps=fps, codec="libx264",
                            quality=VIDEO_QUALITY, pixelformat="yuv420p",
                            macro_block_size=None)
    try:
        for i, ts in enumerate(timestamps):
            frame = _render_frame(df_day, ts, i, precomputed, campus_img, pixel_pos)
            writer.append_data(frame)
            del frame
    finally:
        writer.close()

    size_mb = output_path.stat().st_size / 1_048_576
    logger.info("Video ready: %s (%.1f MB, %d frames @ %d fps)",
                output_path.name, size_mb, n, fps)

    _upload_to_hf(output_path)
    _prune_hf_videos()
    _prune_local_videos(output_path.parent)
    return output_path
