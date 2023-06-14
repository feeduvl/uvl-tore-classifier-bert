from contextlib import contextmanager
from pathlib import Path


@contextmanager
def create_file(file_path: Path, mode="w+", encoding="utf-8", buffering=1):
    if file_path.exists():
        file_path.unlink()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(file_path, mode=mode, encoding=encoding, buffering=buffering)
    try:
        yield f
    finally:
        f.close()
