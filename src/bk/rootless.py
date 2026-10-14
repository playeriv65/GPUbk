"""Explicit user-owned broker deployment for the laboratory fork."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import threading
from pathlib import Path
from typing import Optional

from .models import BookingError
from .userdirs import xdg_user_directory


def client_pointer_path() -> Path:
    return xdg_user_directory("XDG_CONFIG_HOME", ".config") / "gpubk" / "server.json"


def connected_config_path() -> Optional[Path]:
    path = client_pointer_path()
    if not os.path.lexists(path):
        return None
    from .fileio import open_existing_regular
    from .config import _open_trusted_config_parent

    parent_fd = _open_trusted_config_parent(path.parent)
    os.close(parent_fd)
    fd = open_existing_regular(path)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > 8192
        ):
            raise BookingError(
                "client server pointer must be a private file owned by this UID"
            )
        document = json.load(stream)
    if not isinstance(document, dict) or set(document) != {"config_file"}:
        raise BookingError("invalid client server pointer")
    value = document["config_file"]
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 4096:
        raise BookingError(
            "client server configuration path must be bounded nonempty text"
        )
    result = Path(value)
    if not result.is_absolute():
        raise BookingError("client server configuration path must be absolute")
    return result


def initialize_server(
    directory: Path,
    *,
    gpu_count: int,
    cpus: int = 0,
    memory_mb: int = 0,
    no_show_seconds: int = 3600,
) -> Path:
    from .config import Config, MAX_GPU_COUNT, _open_trusted_config_parent, load_config

    directory = Path(os.path.abspath(directory.expanduser()))
    if (
        isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or not 1 <= gpu_count <= MAX_GPU_COUNT
    ):
        raise BookingError(f"gpu_count must be between 1 and {MAX_GPU_COUNT}")
    Config(
        directory,
        gpu_count=gpu_count,
        host_cpu_capacity=cpus,
        host_memory_capacity_mb=memory_mb,
        no_show_grace_seconds=no_show_seconds,
    )
    if os.path.lexists(directory):
        raise BookingError("server directory already exists; choose a new directory")
    fd = _open_trusted_config_parent(directory.parent)
    os.close(fd)
    socket_path = directory / "run" / "broker.sock"
    if len(os.fsencode(socket_path)) >= 108:
        raise BookingError("server socket path is too long; choose a shorter directory")
    directory.mkdir(mode=0o755)
    for child in ("data", "run"):
        (directory / child).mkdir(mode=0o755)
        (directory / child).chmod(0o755)
    directory.chmod(0o755)
    configuration = {
        "config_version": 1,
        "rootless": True,
        "data_dir": str(directory / "data"),
        "gpu_count": gpu_count,
        "broker_uid": os.getuid(),
        "monitor_uid": os.getuid(),
        "broker_socket": str(socket_path),
        "broker_socket_mode": "0666",
        "file_mode": "0644",
        "dir_mode": "0755",
        "host_cpu_capacity": cpus,
        "host_memory_capacity_mb": memory_mb,
        "no_show_grace_seconds": no_show_seconds,
        "monitor_interval_seconds": 10,
        "monitor_rollup_seconds": 60,
        "monitor_pause_on_exclusive": True,
    }
    path = directory / "config.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2)
        stream.write("\n")
    path.chmod(0o644)
    load_config(path)
    return path


def connect_server(path: Path) -> Path:
    from .config import load_config, _open_trusted_config_parent

    config = load_config(path)
    if config.broker_socket is None:
        raise BookingError("configuration does not define a shared broker")
    destination = client_pointer_path()
    destination.parent.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        destination.parent.stat().st_uid != os.getuid()
        or destination.parent.is_symlink()
    ):
        raise BookingError("client configuration directory must belong to this UID")
    destination.parent.chmod(0o700)
    parent_fd = _open_trusted_config_parent(destination.parent)
    os.close(parent_fd)
    import tempfile

    fd, temporary = tempfile.mkstemp(prefix=".server-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"config_file": str(config.config_file)}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return destination


def run_server_cli(argv: list[str]) -> int:
    from .config import load_config
    from .timeparse import parse_memory_mb

    parser = argparse.ArgumentParser(prog="bk server")
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="create a new user-owned shared server")
    init.add_argument("--directory", type=Path, required=True)
    init.add_argument("--gpu-count", type=int, required=True)
    init.add_argument("--cpus", type=int, default=0)
    init.add_argument("--ram", help="declared host memory capacity, e.g. 256g")
    init.add_argument("--no-show-seconds", type=int, default=3600)
    connect = sub.add_parser(
        "connect", help="save this user's shared configuration pointer"
    )
    connect.add_argument("config_file", type=Path)
    serve = sub.add_parser(
        "serve", help="run the broker and monitor in this foreground process"
    )
    serve.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        path = initialize_server(
            args.directory,
            gpu_count=args.gpu_count,
            cpus=args.cpus,
            memory_mb=parse_memory_mb(args.ram) if args.ram else 0,
            no_show_seconds=args.no_show_seconds,
        )
        print(
            f"configuration: {path}\nconnect: bk server connect {shlex.quote(str(path))}\nserve: bk server serve --config {shlex.quote(str(path))}"
        )
        return 0
    if args.action == "connect":
        print(f"connected: {connect_server(args.config_file)}")
        return 0
    from .broker import BrokerServer
    from .monitor import run_monitor

    config = load_config(args.config)
    if not config.rootless:
        raise BookingError(
            "bk server serve requires an explicit rootless configuration"
        )
    server = BrokerServer(config)
    failed = []
    stop_event = threading.Event()

    def serve_broker():
        try:
            server.serve_forever()
        except Exception as exc:
            failed.append(exc)
            stop_event.set()

    try:
        server.prepare()
        thread = threading.Thread(target=serve_broker, name="gpubk-broker", daemon=True)
        thread.start()
        result = run_monitor(config, server.store, stop_event=stop_event)
        if failed:
            raise BookingError(f"broker stopped: {failed[0]}") from failed[0]
        return result
    finally:
        server.close()
        if "thread" in locals():
            thread.join(timeout=5)
