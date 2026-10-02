import json
import os
import time
import uuid
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)  # Windows scanners can briefly hold an existing checkpoint open.
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Lock:
    """Fail immediately if another local writer is active; hooks never wait on training."""
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *args):
        os.close(self.fd)
        self.path.unlink(missing_ok=True)
