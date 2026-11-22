"""Unit checkpoints for long runs.

A run is split into units (a cell, a cell at one tolerance). When a unit
finishes, its rows are written to <directory>/<unit>.json together with the
parameters that produced them; a later run with the same parameters reads the
file instead of computing the unit again, so an interruption costs at most the
unit that was running.

The rows a caller gets back are always the rows as stored, whether the unit was
just computed or read back: numpy scalars become Python numbers and tuples
become lists in both cases. Outputs assembled from the rows are therefore the
same for a run that was interrupted and resumed as for one that was not.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

UNIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def replace_with_retry(tmp: Path, path: Path, attempts: int = 40, delay: float = 0.25) -> None:
    """os.replace with retries: Windows refuses to replace a file that another
    process holds open, and virus scanners and indexers briefly open new files."""
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _partial(path: Path) -> Path:
    """Where a file is written before it is moved into place: same directory and
    extension, and a name no pattern for the finished file (x_hat_*.npy) matches."""
    return path.with_name(".partial_" + path.name)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text so that a reader sees the old file or the new one, never half of
    one; a file whose existence marks finished work must be written this way."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(path)
    tmp.write_text(text, encoding="utf-8")
    replace_with_retry(tmp, path)


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """np.save, written in full before it appears under its name."""
    path = Path(path)
    tmp = _partial(path)
    try:
        np.save(tmp, arr)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    replace_with_retry(tmp, path)


def write_csv_atomic(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    """A CSV written in full before it appears under its name (a field a row
    lacks is left empty), so a run stopped while writing leaves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    replace_with_retry(tmp, path)


def write_json_atomic(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, default=_plain))


def digests(paths: Iterable[Path]) -> dict[str, str]:
    """file name -> sha256 of its bytes: the identity of a unit's input files or
    of the code that computes it, for the parameters of a UnitStore."""
    out: dict[str, str] = {}
    for p in map(Path, paths):
        if p.name in out:
            raise ValueError(f"two files named {p.name}")
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _normalised(obj):
    """obj as it reads back from JSON."""
    return json.loads(json.dumps(obj, default=_plain))


class UnitStore:
    """The finished units of one run, one JSON file per unit."""

    def __init__(self, directory: Path, params: dict, log: Callable[[str], None] | None = None):
        self.directory = Path(directory)
        self.params = _normalised(params)
        self.log = log or (lambda msg: print(msg, flush=True))

    def key(self) -> str:
        """A short name for the parameters, for files that belong to one parameter
        set, such as the state of a computation that continues across units."""
        return hashlib.sha256(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:12]

    def path(self, unit: str) -> Path:
        if not UNIT_NAME.match(unit):
            raise ValueError(f"unit name {unit!r} is not a plain file name")
        return self.directory / f"{unit}.json"

    def load(self, unit: str) -> list[dict] | None:
        """The stored rows, or None when the unit has not finished under these parameters."""
        p = self.path(unit)
        if not p.exists():
            return None
        try:
            stored = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if stored.get("params") != self.params or stored.get("unit") != unit:
            return None
        return stored["rows"]

    def save(self, unit: str, rows: list[dict]) -> list[dict]:
        write_json_atomic(self.path(unit), {"unit": unit, "params": self.params, "rows": rows})
        return _normalised(rows)

    def rows(self, unit: str, compute: Callable[[], list[dict]], outputs: Iterable[Path] = ()) -> list[dict]:
        """The unit's rows: read back when it finished before, else computed and saved.
        outputs are files the computation writes besides its rows; the unit counts
        as finished only while they all exist."""
        rows = self.load(unit) if all(Path(p).exists() for p in outputs) else None
        if rows is not None:
            self.log(f"[checkpoint] {unit}: finished earlier, read from {self.path(unit)}")
            return rows
        return self.save(unit, compute())
