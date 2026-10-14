"""CPU-only tests for the lab fork's no-show safety policy."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest import mock

from bk.config import Config
from bk.gpu import GpuProcessSnapshot, GpuSnapshot
from bk.models import Actor, BookingError
from bk.policy import bind_ledger_policy
from bk.reclaim import reclaim_no_shows, record_no_show_observations
from bk.storage import LedgerStore
from bk.timeparse import to_iso
from bk.worker import claim_due_jobs


class NoShowReclamationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Config(
            data_dir=Path(self.temporary.name),
            gpu_count=2,
            monitor_interval_seconds=10,
            monitor_rollup_seconds=60,
            no_show_grace_seconds=60,
        )
        self.store = LedgerStore(self.config.data_dir)
        self.actor = Actor(os.getuid(), "test-owner")
        self.start = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
        self.devices = [
            GpuSnapshot(
                index=gpu,
                name="CPU-only fixture",
                source="simulation",
                process_telemetry_available=True,
                memory_total_mb=96000,
                device_uuid=f"GPU-00000000-0000-0000-0000-00000000000{gpu}",
            )
            for gpu in range(2)
        ]
        self._seed()

    def _seed(self, *, managed=False, uid=None, gpus=None):
        item = {
            "id": "reservation-1",
            "uid": self.actor.uid if uid is None else uid,
            "username": "test-user",
            "mode": "exclusive",
            "gpus": [0] if gpus is None else gpus,
            "status": "active",
            "start_at": to_iso(self.start),
            "end_at": to_iso(self.start + timedelta(hours=2)),
        }
        if managed:
            item["job"] = {
                "status": "pending",
                "spec_id": "preserve-private-spec",
                "claimed_at": None,
                "started_at": None,
            }

        def mutate(ledger):
            ledger["reservations"] = [item]
            bind_ledger_policy(ledger, self.config)
            return ledger, None, [], True

        self.store.transaction(mutate)

    def _observe(self, second, devices=None):
        return record_no_show_observations(
            self.store,
            self.config,
            self.devices if devices is None else devices,
            self.start + timedelta(seconds=second),
        )

    def _history(self, start=0, end=60):
        for second in range(start, end + 1, 10):
            self._observe(second)

    def _reclaim(self, second=60):
        return reclaim_no_shows(
            self.store,
            self.config,
            self.actor,
            now=self.start + timedelta(seconds=second),
        )

    def _reason(self, second=60):
        return self._reclaim(second)["retained"][0]["reason"]

    def _update(self, change):
        def mutate(ledger):
            change(ledger["reservations"][0])
            return ledger, None, [], True

        self.store.transaction(mutate)

    def test_manual_no_show_is_reclaimed_after_a_full_fresh_history_and_audited(self):
        self._history()
        result = self._reclaim()
        self.assertEqual(result["reclaimed"], ["reservation-1"])
        self.assertEqual(self.store.load()["reservations"][0]["status"], "cancelled")
        logs = [
            json.loads(line) for line in self.store.log_path.read_text().splitlines()
        ]
        self.assertEqual(logs[-1]["action"], "no-show-reclaim")
        self.assertEqual(logs[-1]["uid"], self.actor.uid)
        self.assertIn("transaction_id", logs[-1])
        self.assertEqual(self._reclaim()["reclaimed"], [])

    def test_pending_job_reclamation_cancels_claim_but_preserves_specification(self):
        self._seed(managed=True)
        self._history()
        with mock.patch("os.kill", side_effect=AssertionError("must never signal")):
            self.assertEqual(self._reclaim()["reclaimed"], ["reservation-1"])
        job = self.store.load()["reservations"][0]["job"]
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["spec_id"], "preserve-private-spec")

    def test_grace_and_missing_history_retain_reservations(self):
        self.assertEqual(self._reason(59), "within-start-grace")
        self.assertEqual(self._reason(), "missing-evidence")
        self._observe(60)
        self.assertEqual(self._reason(), "insufficient-absence-history")

    def test_stale_or_future_evidence_fails_closed(self):
        self._history()
        self.assertEqual(self._reason(86), "stale-evidence")
        self.assertEqual(self._reason(59), "within-start-grace")
        self._update(
            lambda item: item["no_show"].update(
                last_observed_at=to_iso(self.start + timedelta(seconds=70))
            )
        )
        self.assertEqual(self._reason(60), "stale-evidence")

    def test_missing_telemetry_resets_absence_window(self):
        self._history(end=30)
        self._observe(40, [replace(self.devices[0], process_telemetry_available=False)])
        self.assertEqual(self._reason(), "process-telemetry-unavailable")
        self._history(start=50, end=100)
        self.assertEqual(self._reason(100), "insufficient-absence-history")
        self._observe(110)
        self.assertEqual(self._reclaim(110)["reclaimed"], ["reservation-1"])

    def test_monitor_restart_gap_requires_new_full_history(self):
        self._history(end=30)
        self._history(start=70, end=120)
        self.assertEqual(self._reason(120), "insufficient-absence-history")
        self._observe(130)
        self.assertEqual(self._reclaim(130)["reclaimed"], ["reservation-1"])

    def test_zero_utilization_with_a_live_owner_process_is_use(self):
        process = GpuProcessSnapshot(
            pid=12345,
            uid=self.actor.uid,
            host_start_id="boot:12345:100",
            gpu_memory_mb=0,
            sm_utilization_percent=0,
        )
        self._observe(
            0, [replace(self.devices[0], processes=(process,), utilization_percent=0)]
        )
        self._history(start=10, end=80)
        self.assertEqual(self._reason(80), "ever-started")

    def test_observed_use_is_sticky_across_device_edits_and_monitor_restarts(self):
        process = GpuProcessSnapshot(
            pid=10, uid=self.actor.uid, host_start_id="boot:10:200"
        )
        self._observe(0, [replace(self.devices[0], processes=(process,))])
        self._update(lambda item: item.update(gpus=[1]))
        self._history(start=100, end=160)
        self.assertEqual(self._reason(160), "ever-started")

    def test_claimed_running_and_uncertain_jobs_are_never_reclaimed(self):
        for status in ("claimed", "running", "uncertain", "succeeded"):
            with self.subTest(status=status):
                self._seed(managed=True)
                self._history()
                self._update(
                    lambda item, status=status: item["job"].update(status=status)
                )
                self.assertEqual(self._reason(), "job-started-or-claimed")

    def test_pending_job_with_any_durable_launch_marker_is_retained(self):
        for key in (
            "claimed_at",
            "started_at",
            "claim_token",
            "runner_pid",
            "worker_lease_id",
        ):
            with self.subTest(key=key):
                self._seed(managed=True)
                self._history()
                self._update(
                    lambda item, key=key: item["job"].update(
                        {key: 123 if key == "runner_pid" else "marker"}
                    )
                )
                self.assertEqual(self._reason(), "job-started-or-claimed")

    def test_ambiguous_and_inferred_container_identity_blocks_absence(self):
        for source in ("host", "container-ambiguous", "container-reservation"):
            with self.subTest(source=source):
                self._seed()
                self._history(end=50)
                process = GpuProcessSnapshot(
                    pid=99, identity_source=source, host_start_id="boot:99:100"
                )
                self._observe(60, [replace(self.devices[0], processes=(process,))])
                self.assertEqual(self._reason(), "process-identity-uncertain")

    def test_verified_other_owner_process_does_not_hide_this_owners_absence(self):
        process = GpuProcessSnapshot(
            pid=99, uid=self.actor.uid + 100, host_start_id="boot:99:100"
        )
        devices = [replace(self.devices[0], processes=(process,))]
        for second in range(0, 61, 10):
            self._observe(second, devices)
        self.assertEqual(self._reclaim()["reclaimed"], ["reservation-1"])

    def test_unattributed_device_activity_fails_closed(self):
        self._history(end=50)
        self._observe(60, [replace(self.devices[0], utilization_percent=80)])
        self.assertEqual(self._reason(), "unattributed-device-activity")

    def test_every_reserved_device_needs_trusted_process_telemetry_and_identity(self):
        self._seed(gpus=[0, 1])
        self._history(end=50)
        self._observe(60, [self.devices[0]])
        self.assertEqual(self._reason(), "process-telemetry-unavailable")
        self._observe(70, [self.devices[0], replace(self.devices[1], device_uuid="")])
        self.assertEqual(self._reason(70), "device-identity-unavailable")

    def test_gpu_identity_change_resets_absence_history(self):
        self._history(end=50)
        self._observe(
            60,
            [
                replace(
                    self.devices[0],
                    device_uuid="GPU-11111111-1111-1111-1111-111111111111",
                )
            ],
        )
        self.assertEqual(self._reason(), "insufficient-absence-history")

    def test_booking_edit_invalidates_old_proof(self):
        self._history()
        self._update(lambda item: item.update(gpus=[1]))
        self.assertEqual(self._reason(), "evidence-binding-changed")

    def test_out_of_order_sample_cannot_erase_newer_owner_usage(self):
        self._history(end=50)
        process = GpuProcessSnapshot(
            pid=99, uid=self.actor.uid, host_start_id="boot:99:100"
        )
        self._observe(60, [replace(self.devices[0], processes=(process,))])
        self._observe(50)
        self.assertEqual(self._reason(), "ever-started")

    def test_disabled_policy_never_collects_or_reclaims(self):
        self.config = replace(self.config, no_show_grace_seconds=0)
        self.assertTrue(self._observe(0)["disabled"])
        self.assertTrue(self._reclaim()["disabled"])
        self.assertNotIn("no_show", self.store.load()["reservations"][0])

    def test_private_reclaimer_cannot_reclaim_another_uid(self):
        self._seed(uid=self.actor.uid + 100)
        self._history()
        self.assertEqual(self._reclaim()["reclaimed"], [])
        self.assertEqual(self.store.load()["reservations"][0]["status"], "active")

    def test_broker_owner_can_reclaim_other_uids_but_clients_cannot(self):
        self.config = replace(
            self.config,
            data_dir=Path(self.temporary.name) / "broker-data",
            broker_uid=self.actor.uid,
            broker_socket=Path(self.temporary.name) / "broker.sock",
            broker_socket_mode=0o666,
            file_mode=0o644,
            dir_mode=0o755,
            rootless=True,
        )
        self.store = LedgerStore(
            self.config.data_dir,
            file_mode=self.config.file_mode,
            dir_mode=self.config.dir_mode,
        )
        self._seed(uid=self.actor.uid + 100)
        self._history()
        self.assertEqual(self._reclaim()["reclaimed"], ["reservation-1"])
        with (
            mock.patch("bk.reclaim.os.getuid", return_value=self.actor.uid + 100),
            self.assertRaisesRegex(BookingError, "broker owner"),
        ):
            reclaim_no_shows(
                self.store,
                self.config,
                Actor(self.actor.uid + 100, "client"),
                now=self.start,
            )

    def test_only_configured_monitor_or_broker_owner_records_evidence(self):
        self.config = replace(self.config, monitor_uid=self.actor.uid + 100)
        with self.assertRaisesRegex(BookingError, "configured monitor"):
            self._observe(0)

    def test_forged_actor_is_rejected(self):
        with self.assertRaisesRegex(BookingError, "executing UID"):
            reclaim_no_shows(
                self.store,
                self.config,
                Actor(self.actor.uid + 100, "forged"),
                now=self.start,
            )

    def test_concurrent_worker_claim_and_reclaim_cannot_both_succeed(self):
        self._seed(managed=True)
        self._history()
        start_barrier = Barrier(2)

        def claim():
            start_barrier.wait(timeout=5)
            return claim_due_jobs(
                self.store,
                self.actor,
                self.start + timedelta(seconds=60),
                worker_id="test-worker",
                worker_lease_id="test-lease",
                runner_host="cpu-test",
                runner_pid=12345,
                claim_timeout_seconds=30,
                limit=1,
            )

        def reclaim():
            start_barrier.wait(timeout=5)
            return self._reclaim()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claim_future = pool.submit(claim)
            reclaim_future = pool.submit(reclaim)
            claimed = claim_future.result(timeout=10)
            reclaimed = reclaim_future.result(timeout=10)["reclaimed"]
        self.assertEqual(bool(claimed) + bool(reclaimed), 1)
        stored = self.store.load()["reservations"][0]
        self.assertEqual(stored["job"]["status"], "claimed" if claimed else "cancelled")
        self.assertEqual(stored["status"], "active" if claimed else "cancelled")

    def test_worker_does_not_claim_reclaimed_pending_job(self):
        self._seed(managed=True)
        self._history()
        self._reclaim()
        claimed = claim_due_jobs(
            self.store,
            self.actor,
            self.start + timedelta(seconds=60),
            worker_id="test-worker",
            worker_lease_id="test-lease",
            runner_host="cpu-test",
            runner_pid=12345,
            claim_timeout_seconds=30,
            limit=1,
        )
        self.assertEqual(claimed, [])


if __name__ == "__main__":
    unittest.main()
