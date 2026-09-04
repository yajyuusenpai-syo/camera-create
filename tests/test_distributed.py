"""Test deterministic global sharding, immutable manifests, and task leases."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from camera_create.batch import BatchOptions, run_batch
from camera_create.cli import build_parser, main
from camera_create.config import ModelPaths
from camera_create.distributed import (
    DistributedLayout,
    TaskLease,
    assign_node_tasks,
    ensure_shared_manifest,
    validate_run_id,
)
from camera_create.pipeline import PipelineOptions


def test_global_worker_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    videos = [tmp_path / f"video_{index:02d}.mp4" for index in range(12)]
    node_zero = assign_node_tasks(videos, DistributedLayout(0, 2, 2))
    node_one = assign_node_tasks(videos, DistributedLayout(1, 2, 2))

    assert node_zero == [videos[0::4], videos[1::4]]
    assert node_one == [videos[2::4], videos[3::4]]
    assigned = [path for tasks in node_zero + node_one for path in tasks]
    assert sorted(assigned) == sorted(videos)
    assert len(set(assigned)) == len(videos)


def test_eight_nodes_with_32_workers_cover_50k_tasks_once(tmp_path: Path) -> None:
    videos = [tmp_path / f"video_{index:05d}.mp4" for index in range(50_000)]
    assigned: list[Path] = []
    shard_sizes: list[int] = []
    for node_rank in range(8):
        shards = assign_node_tasks(videos, DistributedLayout(node_rank, 8, 32))
        assigned.extend(path for shard in shards for path in shard)
        shard_sizes.extend(len(shard) for shard in shards)

    assert len(assigned) == 50_000
    assert len(set(assigned)) == 50_000
    assert set(assigned) == set(videos)
    assert max(shard_sizes) - min(shard_sizes) <= 1


def test_layout_and_run_id_validation() -> None:
    DistributedLayout(7, 8, 32).validate()
    assert validate_run_id("camera-50k_v1") == "camera-50k_v1"
    with pytest.raises(ValueError):
        DistributedLayout(8, 8, 32).validate()
    with pytest.raises(ValueError):
        validate_run_id("../unsafe")


def test_shared_manifest_is_immutable(tmp_path: Path) -> None:
    payload = {"format_version": 1, "videos": ["a.mp4"], "num_nodes": 2}
    first = ensure_shared_manifest(tmp_path, payload)
    second = ensure_shared_manifest(tmp_path, payload)
    assert first == second
    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert stored["manifest_sha256"] == first
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        ensure_shared_manifest(tmp_path, {**payload, "videos": ["b.mp4"]})


def test_task_lease_is_exclusive_and_releasable(tmp_path: Path) -> None:
    path = tmp_path / "leases" / "job.lease"
    first = TaskLease(path, global_worker_id=0, timeout_seconds=60)
    second = TaskLease(path, global_worker_id=1, timeout_seconds=60)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
    assert not path.exists()


def test_task_lease_recovers_expired_owner(tmp_path: Path) -> None:
    path = tmp_path / "leases" / "job.lease"
    path.mkdir(parents=True)
    heartbeat = path / "heartbeat"
    heartbeat.touch()
    expired = time.time() - 120
    os.utime(heartbeat, (expired, expired))

    lease = TaskLease(path, global_worker_id=2, timeout_seconds=30)
    assert lease.acquire()
    lease.release()


def test_empty_multinode_run_writes_isolated_node_summaries(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    checkpoint_root = tmp_path / "checkpoints"
    pi3x = tmp_path / "models" / "pi3x"
    moge3 = tmp_path / "models" / "moge3"
    vipe = tmp_path / "models" / "vipe"
    input_root.mkdir()
    for model in (pi3x, moge3, vipe):
        model.mkdir(parents=True)
        (model / "placeholder").touch()
    options = BatchOptions(
        input_root=input_root,
        checkpoint_root=checkpoint_root,
        model_paths=ModelPaths(pi3x, moge3, vipe),
        pipeline_options=PipelineOptions(),
        gpu_ids=(0, 1),
        workers_per_gpu=2,
        node_rank=0,
        num_nodes=2,
        run_id="empty-distributed-test",
    )

    first = run_batch(options)
    second = run_batch(replace(options, node_rank=1))
    run_root = checkpoint_root / "run_empty-distributed-test"

    assert first["global_workers"] == 8
    assert second["videos_assigned"] == 0
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "summary_node_000.json").is_file()
    assert (run_root / "summary_node_001.json").is_file()
    assert not (run_root / "summary.json").exists()


def test_dlc_environment_and_accelerate_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("DLC_JOB_ID", "dlc-camera-job")
    parser = build_parser()
    defaults = parser.parse_args(["--input", "/shared/videos"])
    aliases = parser.parse_args(
        [
            "--input",
            "/shared/videos",
            "--num_machines",
            "8",
            "--machine_rank",
            "4",
            "--num_processes",
            "64",
            "--main_process_ip",
            "10.0.0.2",
            "--main_process_port",
            "29600",
        ]
    )

    assert defaults.num_nodes == 8
    assert defaults.node_rank == 3
    assert defaults.main_process_ip == "10.0.0.1"
    assert defaults.main_process_port == 29500
    assert defaults.run_id == "dlc-camera-job"
    assert aliases.num_nodes == 8
    assert aliases.node_rank == 4
    assert aliases.launcher_num_processes == 64
    assert aliases.main_process_ip == "10.0.0.2"
    assert aliases.main_process_port == 29600


def test_accelerate_process_environment_is_reduced_to_machine_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "27")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")

    args = build_parser().parse_args(["--input", "/shared/videos"])

    assert args.num_nodes == 8
    assert args.node_rank == 3


def test_nonzero_local_launcher_rank_exits_without_starting_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert main(["--input", str(tmp_path)]) == 0
    assert '"status": "idle_launcher_process"' in capsys.readouterr().out
