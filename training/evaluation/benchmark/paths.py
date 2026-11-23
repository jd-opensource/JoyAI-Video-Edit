import hashlib
import json
from pathlib import Path


def resolve_video_path(source: str, dataset_root: Path) -> Path:
    path = Path(source)
    if not path.is_absolute():
        if path.parts and path.parts[0] == dataset_root.name:
            path = Path(*path.parts[1:])
        path = dataset_root / path
    return path.resolve()


def benchmark_stem(video_path: Path, prompt: str) -> str:
    identity = json.dumps([str(video_path.resolve()), prompt.strip()], ensure_ascii=False).encode("utf-8")
    return f"{video_path.stem}_{hashlib.sha256(identity).hexdigest()[:16]}"
