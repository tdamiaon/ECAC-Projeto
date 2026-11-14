"""Helpers shared by the three extractors: devices, the model folder, files, CSV output with resume, and the run record."""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path

PACKAGE_VERSION = "1.0"
RANDOM_WEIGHTS = os.environ.get("ECAC_EXTRACTORS_RANDOM_WEIGHTS") == "1"   # only for testing the code offline


def models_dir(path=None) -> Path:
    """Folder where the downloaded weights are kept: the argument, else $ECAC_MODELS_DIR, else ~/.cache/ecac_extractors."""
    p = Path(path or os.environ.get("ECAC_MODELS_DIR") or Path.home() / ".cache" / "ecac_extractors")
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(device: str = "auto") -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, target: Path, expected_sha256: str | None = None) -> Path:
    """Download url to target (atomically), unless it is already there; check the SHA-256 when one is given."""
    target = Path(target)
    if not target.exists():
        from tqdm.auto import tqdm
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        print(f"downloading {target.name} from {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
                for block in iter(lambda: r.read(1 << 20), b""):
                    f.write(block)
                    bar.update(len(block))
        tmp.replace(target)
    if expected_sha256 and sha256(target) != expected_sha256:
        raise RuntimeError(f"{target} does not have the expected SHA-256; delete it and download it again")
    return target


def find_files(folder, extensions) -> list[Path]:
    """Every file under folder (subfolders included) with one of the extensions, sorted by name. Hidden files are skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no folder {folder}")
    files = [p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")]
    return sorted(files, key=lambda p: (p.stem, str(p)))


class RowWriter:
    """Writes one row per item to a CSV file, and resumes an interrupted run: ids already written are skipped."""

    def __init__(self, path, id_column: str, columns: list[str]):
        self.path = Path(path)
        self.header = [id_column] + list(columns)
        self.done: set[str] = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            self._drop_partial_last_line()
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                if next(reader, None) != self.header:
                    raise SystemExit(f"{self.path} exists with other columns: move it away or use another --out")
                self.done = {row[0] for row in reader if len(row) == len(self.header)}
            self._f = open(self.path, "a", newline="", encoding="utf-8")
            self._w = csv.writer(self._f)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "w", newline="", encoding="utf-8")
            self._w = csv.writer(self._f)
            self._w.writerow(self.header)

    def _drop_partial_last_line(self):
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            with open(self.path, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)

    def write(self, item_id: str, values):
        self._w.writerow([item_id] + [f"{float(v):.6g}" for v in values])
        self.done.add(item_id)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()


def environment() -> dict:
    versions = {}
    for name in ("numpy", "torch", "torchvision", "torchlibrosa", "sentence_transformers", "transformers",
                 "huggingface_hub", "soundfile", "soxr", "librosa", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = None
    return {"ecac_extractors": PACKAGE_VERSION, "python": sys.version.split()[0],
            "platform": platform.platform(), "libraries": versions}


def record_run(out_dir, domain: str, info: dict) -> Path:
    """Adds this run to models_used.json in out_dir: the model, its exact version, the parameters and what failed."""
    path = Path(out_dir) / "models_used.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["environment"] = environment()
    info = dict(info, finished=datetime.datetime.now().isoformat(timespec="seconds"))
    data[domain] = info
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
