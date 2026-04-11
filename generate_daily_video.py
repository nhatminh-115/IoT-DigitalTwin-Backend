"""CLI entry point — daily heatmap video generator."""
import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from iot_digital_twin.video_generator import CSV_URL, DEFAULT_FPS, generate_video, video_path

ICT = timezone(timedelta(hours=7))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily heatmap timelapse.")
    parser.add_argument("--date",   default=None, help="YYYY-MM-DD (default: yesterday ICT)")
    parser.add_argument("--fps",    type=int, default=DEFAULT_FPS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    target = (
        date.fromisoformat(args.date)
        if args.date
        else (datetime.now(ICT) - timedelta(days=1)).date()
    )
    out = Path(args.output) if args.output else video_path(target)
    generate_video(target_date=target, output_path=out, fps=args.fps)
