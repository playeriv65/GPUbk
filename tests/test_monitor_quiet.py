"""Verify protected timing windows never invoke the monitor's GPU provider."""

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from bk.config import Config
from bk.gpu import GpuSnapshot
from bk.models import Actor, BookingRequest
from bk.monitor import UsageMonitor
from bk.scheduler import add_booking
from bk.shell import environment_payload
from bk.storage import LedgerStore
from bk.usage_store import UsageAuditStore


class QuietMonitorTests(unittest.TestCase):
    def test_exclusive_interval_pauses_sampling_preserves_identity_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
            config = Config(Path(tmp) / "data", monitor_pause_on_exclusive=True)
            store = LedgerStore(config.data_dir)
            usage = UsageAuditStore(config.data_dir)
            device = GpuSnapshot(
                0,
                "synthetic",
                source="simulation",
                device_uuid="GPU-00000000-0000-0000-0000-000000000000",
            )
            provider = mock.Mock(return_value=[device])
            monitor = UsageMonitor(config, store, usage, snapshot_provider=provider)
            monitor.collect(now)
            actor = Actor(os.getuid(), "owner")
            with mock.patch("bk.scheduler.utc_now", return_value=now):
                add_booking(
                    store, config, BookingRequest(actor, 1, 300, now, mode="exclusive")
                )
            provider.reset_mock()
            monitor.collect(now + timedelta(seconds=60))
            provider.assert_not_called()
            with mock.patch.dict(os.environ, {"BK_GPU_SIM_FILE": ""}):
                payload = environment_payload(
                    config, store, actor=actor, now=now + timedelta(seconds=60)
                )
            self.assertEqual(
                payload["cuda_visible_devices"], device.device_uuid, payload
            )
            collector = usage.load_collector_status(
                now=now + timedelta(seconds=60), expected_gpu_count=1
            )
            self.assertEqual(collector["state"], "degraded")
            self.assertFalse(collector["devices"][0]["process_telemetry"])
            monitor.collect(now + timedelta(seconds=301))
            provider.assert_called_once()

    def test_pending_managed_exclusive_job_can_still_collect_no_show_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
            config = Config(Path(tmp) / "data", monitor_pause_on_exclusive=True)
            store = LedgerStore(config.data_dir)
            actor = Actor(os.getuid(), "owner")
            with mock.patch("bk.scheduler.utc_now", return_value=now):
                created = add_booking(
                    store, config, BookingRequest(actor, 1, 300, now, mode="exclusive")
                )

            def pending(ledger):
                ledger["reservations"][0]["job"] = {"status": "pending"}
                return ledger, None, [], True

            store.transaction(pending)
            provider = mock.Mock(return_value=[])
            monitor = UsageMonitor(
                config,
                store,
                UsageAuditStore(config.data_dir),
                snapshot_provider=provider,
            )
            monitor.collect(now)
            provider.assert_called_once()
            self.assertIsNotNone(created)

    def test_default_budget_larger_than_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(
                Config(Path("/tmp/unused")), host_cpu_capacity=2, default_host_cpus=3
            )

    def test_env_configuration_never_probes_gpu_when_topology_is_unspecified(self):
        from bk.config import load_config

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"BK_DATA_DIR": tmp}, clear=True),
            mock.patch(
                "bk.config._auto_gpu_count", side_effect=AssertionError("driver query")
            ),
        ):
            self.assertEqual(load_config(probe_gpu=False).gpu_count, 1)


if __name__ == "__main__":
    unittest.main()
