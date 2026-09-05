"""Coordinate deterministic multi-node shards and shared-filesystem task leases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class DistributedLayout:
    """Describe one node's place in a homogeneous static worker topology."""

    node_rank: int
    num_nodes: int
    local_worker_count: int

    def validate(self) -> None:
        """Reject layouts that cannot produce unique global worker identifiers."""
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if not 0 <= self.node_rank < self.num_nodes:
            raise ValueError(
                f"node_rank must be in [0, {self.num_nodes - 1}], got {self.node_rank}"
            )
        if self.local_worker_count < 1:
            raise ValueError("local_worker_count must be positive")

    @property
    def global_worker_count(self) -> int:
        """Return the total number of workers across all nodes."""
        return self.num_nodes * self.local_worker_count

    def global_worker_id(self, local_worker_id: int) -> int:
        """Map a node-local worker number to its stable global worker number."""
        if not 0 <= local_worker_id < self.local_worker_count:
            raise ValueError(f"Invalid local worker id: {local_worker_id}")
        return self.node_rank * self.local_worker_count + local_worker_id


def assign_node_tasks(
    videos: list[Path], layout: DistributedLayout
) -> list[list[Path]]:
    """Return this node's deterministic slices of one globally ordered manifest."""
    layout.validate()
    return [
        videos[layout.global_worker_id(local_id) :: layout.global_worker_count]
        for local_id in range(layout.local_worker_count)
    ]


def validate_run_id(value: str) -> str:
    """Validate a user-visible run namespace before using it as a directory name."""
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "--run-id must be 1-128 characters using letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )
    return value


def manifest_digest(payload: dict[str, Any]) -> str:
    """Return a stable digest for cross-node manifest agreement checks."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_shared_manifest(run_root: Path, payload: dict[str, Any]) -> str:
    """Create one immutable manifest or verify exact agreement with another node."""
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    expected = {**payload, "manifest_sha256": manifest_digest(payload)}
    encoded = json.dumps(expected, indent=2, ensure_ascii=False).encode("utf-8")
    try:
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing: dict[str, Any] | None = None
        for _ in range(50):
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                break
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
        if existing != expected:
            existing_digest = (
                existing.get("manifest_sha256") if isinstance(existing, dict) else None
            )
            raise RuntimeError(
                "Distributed run manifest mismatch. Every node must use the same input "
                "snapshot, topology and inference parameters. Choose a new --run-id after "
                f"correcting the mismatch (existing={existing_digest}, "
                f"current={expected['manifest_sha256']})."
            )
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return expected["manifest_sha256"]


class TaskLease:
    """Hold an atomic, heartbeating lease for one video on shared storage."""

    def __init__(
        self,
        path: Path,
        global_worker_id: int,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("lease timeout must be positive")
        self.path = path
        self.global_worker_id = global_worker_id
        self.timeout_seconds = timeout_seconds
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def owner_path(self) -> Path:
        """Return the diagnostic owner record within the lease directory."""
        return self.path / "owner.json"

    @property
    def heartbeat_path(self) -> Path:
        """Return the file whose modification time proves the owner is alive."""
        return self.path / "heartbeat"

    def acquire(self) -> bool:
        """Atomically acquire the lease, recovering it only after heartbeat expiry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                self.path.mkdir()
            except FileExistsError:
                if not self._recover_stale_lease():
                    return False
                continue
            owner = {
                "format_version": 1,
                "token": self.token,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "global_worker_id": self.global_worker_id,
                "started_unix": time.time(),
            }
            self.owner_path.write_text(
                json.dumps(owner, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.heartbeat_path.touch()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"camera-lease-{self.global_worker_id}",
                daemon=True,
            )
            self._thread.start()
            return True
        return False

    def _recover_stale_lease(self) -> bool:
        """Move one expired lease aside atomically so only one contender recovers it."""
        heartbeat = self.heartbeat_path
        try:
            modified = heartbeat.stat().st_mtime if heartbeat.exists() else self.path.stat().st_mtime
        except FileNotFoundError:
            return True
        if time.time() - modified <= self.timeout_seconds:
            return False
        try:
            owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
            if owner.get("hostname") == socket.gethostname():
                pid = int(owner["pid"])
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return False
                else:
                    return False
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        try:
            current = (
                heartbeat.stat().st_mtime
                if heartbeat.exists()
                else self.path.stat().st_mtime
            )
        except FileNotFoundError:
            return True
        if current != modified or time.time() - current <= self.timeout_seconds:
            return False
        stale = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex}")
        try:
            self.path.rename(stale)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        shutil.rmtree(stale, ignore_errors=True)
        return True

    def _heartbeat_loop(self) -> None:
        """Refresh the lease and fence the owner before a stale takeover is possible."""
        interval = max(1.0, min(60.0, self.timeout_seconds / 3.0))
        last_success = time.monotonic()
        while not self._stop.wait(interval):
            try:
                owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
                if owner.get("token") != self.token:
                    self._lost.set()
                    return
                self.heartbeat_path.touch()
            except (OSError, json.JSONDecodeError):
                if time.monotonic() - last_success >= self.timeout_seconds / 2.0:
                    self._lost.set()
                    return
            else:
                last_success = time.monotonic()

    def assert_owned(self) -> None:
        """Refuse publication after heartbeat loss or ownership replacement."""
        if self._lost.is_set():
            raise RuntimeError(f"Lease heartbeat was lost: {self.path}")
        try:
            owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot verify lease ownership: {self.path}") from error
        if owner.get("token") != self.token:
            raise RuntimeError(f"Lease ownership changed while processing: {self.path}")

    def release(self) -> None:
        """Release only a lease still owned by this exact process invocation."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._lost.is_set():
            return
        try:
            owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if owner.get("token") == self.token:
            shutil.rmtree(self.path, ignore_errors=True)
