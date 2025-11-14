import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional


class HeatmapMetadataLogger:
    """Append-only JSONL logger for storing per-frame heatmap projections."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # reset file content
        self.path.write_text("", encoding="utf-8")

    def log(self, frame_index: int, points: Optional[Iterable[Dict[str, Any]]]) -> None:
        payload = {
            "frame_index": int(frame_index),
            "points": list(points or [])
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Placeholder for compatibility with contexts where a close hook is expected."""
        return None
