from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path, run_id: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._seq = 0
        # modo "x": nunca se mezcla un run nuevo con un log viejo.
        self._handle = self._path.open("x", encoding="utf-8")

    def append(self, tipo: str, **datos: Any) -> None:
        evento = {"seq": self._seq, "run_id": self._run_id, "tipo": tipo, **datos}
        self._handle.write(json.dumps(evento, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()
        self._seq += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()