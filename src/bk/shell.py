# SPDX-License-Identifier: Apache-2.0
"""Cooperative Bash visibility from reservations and cached device identity."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
from collections.abc import Sequence
from datetime import datetime
from importlib import resources
from pathlib import Path

from .collector_status import classify_collector_document
from .config import Config
from .fileio import open_existing_regular
from .gpu import GpuSnapshot, has_stable_device_identifier, snapshot
from .identity import current_actor
from .models import STATUS_ACTIVE, Actor, BookingError
from .policy import validate_ledger_policy
from .storage import LedgerStore
from .timeparse import parse_iso, to_iso, utc_now

ENV_SCHEMA_VERSION = "gpubk.env.v1"
MAX_COLLECTOR_BYTES = 1024 * 1024


def environment_payload(
    config: Config,
    store: LedgerStore,
    *,
    actor: Actor | None = None,
    now: datetime | None = None,
) -> dict:
    """Fail closed; future or expired reservations never grant device visibility."""
    current = now or utc_now()
    identity = actor or current_actor()
    result = {
        "schema_version": ENV_SCHEMA_VERSION,
        "generated_at": to_iso(current),
        "uid": identity.uid,
        "status": "ready",
        "gpus": [],
        "reservation_ids": [],
        "cuda_visible_devices": "",
        "error": None,
    }
    try:
        if config.broker_socket is not None:
            from .broker import BrokerClient

            client = BrokerClient(config)
            client.timeout = min(client.timeout, 2.0)
            client.call("ping", {})
        ledger = store.load_read_only()
        if store.last_warning:
            raise BookingError("cannot grant visibility from a recovered backup ledger")
        validate_ledger_policy(ledger, config)
        own = [
            item
            for item in ledger.get("reservations", [])
            if item.get("uid") == identity.uid
            and item.get("status") == STATUS_ACTIVE
            and parse_iso(item["start_at"]) <= current < parse_iso(item["end_at"])
        ]
        selected = sorted({gpu for item in own for gpu in item["gpus"]})
        if any(
            isinstance(gpu, bool)
            or not isinstance(gpu, int)
            or gpu < 0
            or gpu >= config.gpu_count
            for gpu in selected
        ):
            raise BookingError("reservation contains an invalid GPU index")
        selected = [gpu for gpu in selected if gpu not in config.disabled_gpus]
        if selected:
            tokens = _device_tokens(config, selected, current)
            result["cuda_visible_devices"] = ",".join(tokens)
        result["gpus"] = selected
        result["reservation_ids"] = [str(item["id"]) for item in own]
    except (BookingError, OSError, ValueError, TypeError, KeyError) as exc:
        result["status"] = "unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def bash_environment(payload: dict) -> str:
    value = payload["cuda_visible_devices"] if payload.get("status") == "ready" else ""
    return f"export CUDA_VISIBLE_DEVICES={shlex.quote(value)}\n"


def run_env_cli(argv: Sequence[str], config: Config, store: LedgerStore) -> int:
    parser = argparse.ArgumentParser(prog="bk env")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--bash", action="store_true", help="print a safely quoted Bash export"
    )
    output.add_argument(
        "--json", action="store_true", help="print the structured visibility result"
    )
    output.add_argument(
        "--value",
        action="store_true",
        help="print only comma-separated stable device IDs",
    )
    args = parser.parse_args(argv)
    payload = environment_payload(config, store)
    if args.bash:
        print(bash_environment(payload), end="")
    elif args.value:
        print(payload["cuda_visible_devices"])
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "ready" else 2


def shell_path() -> Path:
    path = Path(str(resources.files("bk").joinpath("data", "shell", "gpubk.bash")))
    if not path.is_file():
        raise BookingError(
            "the installed package does not expose its Bash integration file"
        )
    return path.resolve()


def run_shell_cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="bk shell")
    parser.add_argument(
        "action", choices=("init", "path", "show"), nargs="?", default="init"
    )
    args = parser.parse_args(argv)
    path = shell_path()
    if args.action == "path":
        print(path)
    elif args.action == "show":
        print(path.read_text(encoding="utf-8"), end="")
    else:
        print(f"source {shlex.quote(str(path))}")
    return 0


def _device_tokens(config: Config, selected: list[int], now: datetime) -> list[str]:
    if os.environ.get("BK_GPU_SIM_FILE") and config.broker_socket is None:
        devices = snapshot(config)
        by_index = {item.index: item for item in devices if item.source == "simulation"}
    else:
        path = config.data_dir / "usage" / "collector.json"
        fd = open_existing_regular(path)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            owner = (
                config.monitor_uid
                if config.monitor_uid is not None
                else config.broker_uid
                if config.broker_uid is not None
                else config.config_owner_uid
                if config.config_owner_uid is not None
                else os.getuid()
            )
            if metadata.st_uid != owner or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise BookingError(
                    "collector snapshot is not owned by the configured monitor or is writable by others"
                )
            raw = stream.read(MAX_COLLECTOR_BYTES + 1)
        if len(raw) > MAX_COLLECTOR_BYTES:
            raise BookingError("collector snapshot exceeds its size limit")
        collector = classify_collector_document(
            json.loads(raw), now=now, expected_gpu_count=config.gpu_count
        )
        if collector.get("fresh") is not True:
            raise BookingError(
                f"device identity snapshot is unavailable: {collector['state']}"
            )
        by_index = {
            item["gpu"]: GpuSnapshot(
                index=item["gpu"],
                name="cached",
                device_uuid=item.get("device_uuid", ""),
            )
            for item in collector["devices"]
            if item.get("stable_device_identifier") is True
            and (
                item.get("device_telemetry") is True
                or item.get("identity_only") is True
            )
        }
    tokens = []
    for gpu in selected:
        device = by_index.get(gpu)
        if device is None or not has_stable_device_identifier(device):
            raise BookingError(f"GPU {gpu} has no trusted stable device identifier")
        tokens.append(device.device_uuid)
    if len(set(tokens)) != len(tokens):
        raise BookingError("collector snapshot contains duplicate device identifiers")
    return tokens
