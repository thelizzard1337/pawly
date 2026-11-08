import os
from pathlib import Path
from typing import Protocol
from uuid import UUID


class Storage(Protocol):
    def write(self, key: str, content: bytes) -> None: ...
    def path(self, key: str) -> Path: ...
    def delete(self, key: str) -> None: ...
    def check(self) -> None: ...


class LocalFileStorage:
    """All methods perform blocking IO and must run in a thread from async callers."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, key: str) -> Path:
        if str(UUID(key.removesuffix(".jpg"))) + ".jpg" != key:
            raise ValueError("Invalid storage key")
        return self.root / key

    def write(self, key: str, content: bytes) -> None:
        target = self.path(key)
        temporary = target.with_suffix(".tmp")
        try:
            with temporary.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        path = self.path(key)
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)

    def check(self) -> None:
        from uuid import uuid4

        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / f".probe-{uuid4()}"
        try:
            with probe.open("xb") as output:
                output.write(b"ok")
                output.flush()
        finally:
            probe.unlink(missing_ok=True)

    def old_files(self, before: float) -> list[Path]:
        return [
            path
            for path in self.root.iterdir()
            if path.suffix in (".jpg", ".tmp") and path.stat().st_mtime < before
        ]
