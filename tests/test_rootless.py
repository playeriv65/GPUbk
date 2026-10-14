"""CPU-only tests of explicit user-owned shared broker deployment."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bk.broker import BrokerServer
from bk.config import load_config
from bk.models import BookingError
from bk.rootless import (
    client_pointer_path,
    connect_server,
    connected_config_path,
    initialize_server,
    run_server_cli,
)


class RootlessDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bk-rootless-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("client-config", "client-data", "client-state"):
            (self.root / name).mkdir(mode=0o700)
        self.environment = mock.patch.dict(
            "os.environ",
            {
                "XDG_CONFIG_HOME": str(self.root / "client-config"),
                "XDG_DATA_HOME": str(self.root / "client-data"),
                "XDG_STATE_HOME": str(self.root / "client-state"),
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _initialize(self, **kwargs):
        return initialize_server(self.root / "server", gpu_count=2, **kwargs)

    def _write_pointer(self, document, *, mode=0o600):
        path = client_pointer_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(mode)
        return path

    def test_initializer_creates_service_owned_readable_shared_layout(self):
        path = self._initialize(cpus=8, memory_mb=64000)
        config = load_config(path)
        self.assertTrue(config.rootless)
        self.assertEqual(config.broker_uid, os.getuid())
        self.assertEqual(config.monitor_uid, os.getuid())
        self.assertEqual(config.config_owner_uid, os.getuid())
        self.assertEqual(config.gpu_count, 2)
        self.assertEqual(config.host_cpu_capacity, 8)
        self.assertEqual(config.host_memory_capacity_mb, 64000)
        self.assertEqual(config.no_show_grace_seconds, 3600)
        self.assertGreaterEqual(config.monitor_interval_seconds, 10)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
        for directory in (path.parent, config.data_dir, config.broker_socket.parent):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
            self.assertEqual(directory.stat().st_uid, os.getuid())

    def test_initialization_never_overwrites_an_existing_directory(self):
        directory = self.root / "server"
        directory.mkdir()
        canary = directory / "keep.txt"
        canary.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(BookingError, "already exists"):
            self._initialize()
        self.assertEqual(canary.read_text(), "keep")

    def test_initializer_rejects_writable_non_sticky_ancestor(self):
        parent = self.root / "untrusted"
        parent.mkdir(mode=0o777)
        parent.chmod(0o777)
        with self.assertRaisesRegex(PermissionError, "writable"):
            initialize_server(parent / "server", gpu_count=2)
        self.assertFalse((parent / "server").exists())

    def test_invalid_budget_is_rejected_before_creating_any_files(self):
        for settings in ({"cpus": -1}, {"memory_mb": -1}, {"no_show_seconds": -1}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self._initialize(**settings)
                self.assertFalse((self.root / "server").exists())

    def test_symlink_server_directory_is_never_followed(self):
        target = self.root / "target"
        target.mkdir()
        (self.root / "server").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(BookingError, "already exists"):
            self._initialize()
        self.assertEqual(list(target.iterdir()), [])

    def test_connect_writes_private_atomic_pointer_and_default_config_load_uses_it(
        self,
    ):
        path = self._initialize()
        destination = connect_server(path)
        self.assertEqual(destination, client_pointer_path())
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(destination.parent.stat().st_mode), 0o700)
        self.assertEqual(connected_config_path(), path)
        self.assertEqual(load_config().config_file, path)
        self.assertEqual(list(destination.parent.glob(".server-*")), [])

    def test_connect_replaces_an_existing_pointer_without_touching_its_symlink_target(
        self,
    ):
        path = self._initialize()
        destination = client_pointer_path()
        destination.parent.mkdir(parents=True)
        target = self.root / "canary.txt"
        target.write_text("keep", encoding="utf-8")
        destination.symlink_to(target)
        connect_server(path)
        self.assertFalse(destination.is_symlink())
        self.assertEqual(target.read_text(), "keep")
        self.assertEqual(connected_config_path(), path)

    def test_connect_rejects_writable_non_sticky_pointer_ancestor(self):
        path = self._initialize()
        untrusted = self.root / "untrusted-client"
        untrusted.mkdir(mode=0o777)
        untrusted.chmod(0o777)
        with (
            mock.patch.dict("os.environ", {"XDG_CONFIG_HOME": str(untrusted)}),
            self.assertRaises((PermissionError, BookingError)),
        ):
            connect_server(path)

    def test_pointer_reader_rejects_symlink_and_public_files(self):
        pointer = self._write_pointer(
            {"config_file": "/absolute/config.json"}, mode=0o644
        )
        with self.assertRaisesRegex(BookingError, "private file"):
            connected_config_path()
        pointer.unlink()
        target = self.root / "pointer-target.json"
        target.write_text('{"config_file":"/absolute/config.json"}', encoding="utf-8")
        pointer.symlink_to(target)
        with self.assertRaises(OSError):
            connected_config_path()

    def test_pointer_reader_rejects_unknown_keys_relative_path_and_oversized_document(
        self,
    ):
        documents = (
            {"config_file": "relative.json"},
            {"config_file": "/absolute/config.json", "unknown": True},
            {"config_file": "/" + "x" * 9000},
        )
        for document in documents:
            with self.subTest(document=str(document)[:80]):
                path = self._write_pointer(document)
                with self.assertRaises((BookingError, ValueError)):
                    connected_config_path()
                path.unlink()

    def test_malformed_pointer_path_types_fail_with_actionable_errors(self):
        for value in (None, 1, False, [], {}, ""):
            with self.subTest(value=value):
                path = self._write_pointer({"config_file": value})
                with self.assertRaises((BookingError, ValueError)):
                    connected_config_path()
                path.unlink()

    def test_foreign_client_reads_only_explicit_rootless_owner_matched_config(self):
        path = self._initialize()
        actual_owner = os.getuid()
        # Logical UID validation only; this does not claim an actual cross-UID run.
        with mock.patch("bk.config.os.getuid", return_value=actual_owner + 100):
            config = load_config(path)
        self.assertEqual(config.broker_uid, actual_owner)
        document = json.loads(path.read_text())
        document["rootless"] = False
        path.write_text(json.dumps(document), encoding="utf-8")
        with (
            mock.patch("bk.config.os.getuid", return_value=actual_owner + 100),
            self.assertRaises(PermissionError),
        ):
            load_config(path)

    def test_rootless_file_must_match_broker_owner_and_not_be_writable_by_clients(self):
        path = self._initialize()
        document = json.loads(path.read_text())
        document["broker_uid"] = os.getuid() + 100
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(PermissionError, "broker UID"):
            load_config(path)
        document["broker_uid"] = os.getuid()
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o666)
        with self.assertRaisesRegex(PermissionError, "writable"):
            load_config(path)

    def test_explicit_rootless_config_is_accepted_by_broker_constructor(self):
        path = self._initialize()
        server = BrokerServer(load_config(path))
        self.addCleanup(server.close)
        self.assertEqual(server.config.config_owner_uid, os.getuid())

    def test_serve_propagates_broker_startup_failure(self):
        path = self._initialize()
        with (
            mock.patch(
                "bk.broker.BrokerServer._prepare",
                side_effect=BookingError("test broker startup failure"),
            ),
            mock.patch("bk.monitor.run_monitor", return_value=0),
            self.assertRaisesRegex(BookingError, "test broker startup failure"),
        ):
            run_server_cli(["serve", "--config", str(path)])


if __name__ == "__main__":
    unittest.main()
