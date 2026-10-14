"""CPU-only admission tests for declared host CPU and RAM budgets."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest import mock

from bk.config import Config
from bk.models import Actor, BookingError, BookingRequest, EditRequest
from bk.scheduler import add_booking, edit_booking, find_applied_create
from bk.storage import LedgerStore
from bk.timeparse import parse_iso


class HostResourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bk-host-budgets-")
        self.addCleanup(self.temporary.cleanup)
        self.config = Config(
            data_dir=Path(self.temporary.name),
            gpu_count=3,
            host_cpu_capacity=4,
            host_memory_capacity_mb=8192,
            default_host_cpus=1,
            default_host_memory_mb=1024,
        )
        self.store = LedgerStore(self.config.data_dir)
        self.now = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
        self.start = self.now + timedelta(hours=1)

    def _request(self, uid=1001, gpu=0, **changes):
        fields = {
            "actor": Actor(uid, f"user{uid}"),
            "count": 1,
            "duration_seconds": 3600,
            "start_at": self.start,
            "mode": "exclusive",
            "preferred_gpus": [gpu],
            "host_cpus": 1,
            "host_memory_mb": 1024,
        }
        fields.update(changes)
        return BookingRequest(**fields)

    def _book(self, request):
        with mock.patch("bk.scheduler.utc_now", return_value=self.now):
            return add_booking(self.store, self.config, request)

    def _edit(self, request, *, now=None):
        with mock.patch("bk.scheduler.utc_now", return_value=now or self.now):
            return edit_booking(self.store, self.config, request)

    def test_declared_budgets_are_stored_once_for_a_multiple_gpu_reservation(self):
        self.config = replace(self.config, host_cpu_capacity=2)
        booked = self._book(
            self._request(
                count=2, preferred_gpus=[0, 1], host_cpus=2, host_memory_mb=4096
            )
        )
        self.assertEqual(booked.reservation["gpus"], [0, 1])
        self.assertEqual(booked.reservation["host_cpus"], 2)
        self.assertEqual(booked.reservation["host_memory_mb"], 4096)
        with self.assertRaisesRegex(BookingError, "host_cpus"):
            self._book(self._request(uid=1002, gpu=2))

    def test_omitted_budgets_use_configured_defaults_when_capacities_are_enabled(self):
        self.config = replace(
            self.config, default_host_cpus=2, default_host_memory_mb=2048
        )
        booked = self._book(self._request(host_cpus=None, host_memory_mb=None))
        self.assertEqual(booked.reservation["host_cpus"], 2)
        self.assertEqual(booked.reservation["host_memory_mb"], 2048)

    def test_disabled_capacities_preserve_zero_dependency_gpu_only_behavior(self):
        self.config = replace(
            self.config, host_cpu_capacity=0, host_memory_capacity_mb=0
        )
        booked = self._book(self._request(host_cpus=None, host_memory_mb=None))
        self.assertNotIn("host_cpus", booked.reservation)
        self.assertNotIn("host_memory_mb", booked.reservation)
        explicit = self._book(
            self._request(uid=1002, gpu=1, host_cpus=1000, host_memory_mb=1000000)
        )
        self.assertEqual(explicit.reservation["host_cpus"], 1000)
        self.assertEqual(explicit.reservation["host_memory_mb"], 1000000)

    def test_cpu_budget_conflicts_across_different_free_gpus(self):
        self._book(self._request(host_cpus=4))
        before = self.store.load()
        with self.assertRaisesRegex(BookingError, "host_cpus.*conflict"):
            self._book(self._request(uid=1002, gpu=1))
        self.assertEqual(self.store.load(), before)

    def test_ram_budget_conflicts_across_different_free_gpus(self):
        self._book(self._request(host_memory_mb=8192))
        before = self.store.load()
        with self.assertRaisesRegex(BookingError, "host_memory_mb.*conflict"):
            self._book(self._request(uid=1002, gpu=1))
        self.assertEqual(self.store.load(), before)

    def test_rejected_multiple_gpu_request_never_partially_reserves_free_gpus(self):
        self._book(self._request(host_cpus=4))
        with self.assertRaisesRegex(BookingError, "host_cpus"):
            self._book(self._request(uid=1002, count=2, preferred_gpus=[1, 2]))
        stored = self.store.load()["reservations"]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["gpus"], [0])

    def test_adjacent_host_usage_segments_are_not_summed_across_whole_request(self):
        self._book(
            self._request(
                gpu=0, duration_seconds=1800, host_cpus=3, host_memory_mb=6144
            )
        )
        self._book(
            self._request(
                uid=1002,
                gpu=1,
                start_at=self.start + timedelta(minutes=30),
                duration_seconds=1800,
                host_cpus=3,
                host_memory_mb=6144,
            )
        )
        bridge = self._book(
            self._request(
                uid=1003,
                gpu=2,
                start_at=self.start + timedelta(minutes=15),
                duration_seconds=1800,
                host_cpus=1,
                host_memory_mb=2048,
            )
        )
        self.assertEqual(
            parse_iso(bridge.reservation["start_at"]),
            self.start + timedelta(minutes=15),
        )
        self.assertFalse(bridge.queued)

    def test_interval_end_and_start_can_share_the_exact_host_capacity(self):
        self._book(
            self._request(duration_seconds=1800, host_cpus=4, host_memory_mb=8192)
        )
        after = self._book(
            self._request(
                uid=1002,
                gpu=1,
                start_at=self.start + timedelta(minutes=30),
                host_cpus=4,
                host_memory_mb=8192,
            )
        )
        self.assertFalse(after.queued)

    def test_queued_cpu_request_moves_until_cross_gpu_host_budget_fits(self):
        self._book(self._request(host_cpus=4))
        queued = self._book(self._request(uid=1002, gpu=1, allow_queue=True))
        self.assertTrue(queued.queued)
        self.assertEqual(queued.reservation["gpus"], [1])
        self.assertEqual(
            parse_iso(queued.reservation["start_at"]), self.start + timedelta(hours=1)
        )

    def test_queued_ram_request_moves_until_cross_gpu_host_budget_fits(self):
        self._book(self._request(host_memory_mb=8192))
        queued = self._book(self._request(uid=1002, gpu=1, allow_queue=True))
        self.assertTrue(queued.queued)
        self.assertEqual(
            parse_iso(queued.reservation["start_at"]), self.start + timedelta(hours=1)
        )

    def test_future_renewal_conflicts_with_cpu_budget_on_another_gpu_and_is_atomic(
        self,
    ):
        booked = self._book(self._request(duration_seconds=1800, host_cpus=3))
        self._book(
            self._request(
                uid=1002,
                gpu=1,
                start_at=self.start + timedelta(minutes=30),
                host_cpus=2,
            )
        )
        before = self.store.load()
        with self.assertRaisesRegex(BookingError, "host_cpus"):
            self._edit(
                EditRequest(
                    Actor(1001, "user1001"),
                    booked.reservation["id"],
                    duration_seconds=3600,
                )
            )
        self.assertEqual(self.store.load(), before)

    def test_future_renewal_conflicts_with_ram_budget_on_another_gpu(self):
        booked = self._book(self._request(duration_seconds=1800, host_memory_mb=6144))
        self._book(
            self._request(
                uid=1002,
                gpu=1,
                start_at=self.start + timedelta(minutes=30),
                host_memory_mb=4096,
            )
        )
        with self.assertRaisesRegex(BookingError, "host_memory_mb"):
            self._edit(
                EditRequest(
                    Actor(1001, "user1001"),
                    booked.reservation["id"],
                    duration_seconds=3600,
                )
            )

    def test_late_extension_preserves_host_budget_and_start_time(self):
        booked = self._book(
            self._request(duration_seconds=1800, host_cpus=2, host_memory_mb=3072)
        )
        edited = self._edit(
            EditRequest(
                Actor(1001, "user1001"), booked.reservation["id"], duration_seconds=3600
            ),
            now=self.start + timedelta(minutes=5),
        )
        self.assertEqual(edited.reservation["host_cpus"], 2)
        self.assertEqual(edited.reservation["host_memory_mb"], 3072)
        self.assertEqual(parse_iso(edited.reservation["start_at"]), self.start)
        self.assertEqual(
            parse_iso(edited.reservation["end_at"]), self.start + timedelta(hours=1)
        )

    def test_late_extension_rechecks_host_capacity_on_another_gpu(self):
        booked = self._book(self._request(duration_seconds=1800, host_cpus=3))
        self._book(
            self._request(
                uid=1002,
                gpu=1,
                start_at=self.start + timedelta(minutes=30),
                host_cpus=2,
            )
        )
        with self.assertRaisesRegex(BookingError, "host_cpus"):
            self._edit(
                EditRequest(
                    Actor(1001, "user1001"),
                    booked.reservation["id"],
                    duration_seconds=3600,
                ),
                now=self.start + timedelta(minutes=5),
            )

    def test_started_reservation_cannot_change_host_budget(self):
        booked = self._book(self._request(host_cpus=2, host_memory_mb=2048))
        for changes in ({"host_cpus": 3}, {"host_memory_mb": 3072}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(BookingError, "only change its end time"):
                    self._edit(
                        EditRequest(
                            Actor(1001, "user1001"),
                            booked.reservation["id"],
                            duration_seconds=7200,
                            **changes,
                        ),
                        now=self.start + timedelta(minutes=5),
                    )

    def test_future_edit_updates_declared_budget_and_records_before_after(self):
        booked = self._book(self._request(host_cpus=1, host_memory_mb=1024))
        edited = self._edit(
            EditRequest(
                Actor(1001, "user1001"),
                booked.reservation["id"],
                host_cpus=2,
                host_memory_mb=2048,
                op_id="change-host-budget",
            )
        )
        self.assertEqual(edited.reservation["host_cpus"], 2)
        self.assertEqual(edited.reservation["host_memory_mb"], 2048)
        history = edited.reservation["edit_operations"][-1]
        self.assertEqual(history["before"]["host_cpus"], 1)
        self.assertEqual(history["before"]["host_memory_mb"], 1024)
        self.assertEqual(history["after"]["host_cpus"], 2)
        self.assertEqual(history["after"]["host_memory_mb"], 2048)

    def test_creation_idempotency_rejects_changed_cpu_or_ram_budget(self):
        request = self._request(host_cpus=2, host_memory_mb=2048, op_id="stable-create")
        first = self._book(request)
        self.assertFalse(self._book(request).created)
        for changes in ({"host_cpus": 3}, {"host_memory_mb": 3072}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(BookingError, "different write"):
                    self._book(replace(request, **changes))
        self.assertEqual(len(self.store.load()["reservations"]), 1)
        self.assertEqual(first.reservation["host_cpus"], 2)

    def test_replay_lookup_rejects_changed_host_budget(self):
        request = self._request(host_cpus=2, op_id="lookup-budget")
        self._book(request)
        with (
            mock.patch("bk.scheduler.utc_now", return_value=self.now),
            self.assertRaisesRegex(BookingError, "different write"),
        ):
            find_applied_create(
                self.store.load(), self.config, replace(request, host_cpus=3)
            )

    def test_edit_idempotency_rejects_reused_operation_id_for_another_budget(self):
        booked = self._book(self._request())
        edit = EditRequest(
            Actor(1001, "user1001"),
            booked.reservation["id"],
            host_cpus=2,
            host_memory_mb=2048,
            op_id="stable-edit",
        )
        self._edit(edit)
        self.assertFalse(self._edit(edit).created)
        with self.assertRaisesRegex(BookingError, "different write"):
            self._edit(replace(edit, host_cpus=3))

    def test_boolean_negative_zero_and_over_capacity_budgets_are_rejected(self):
        for field, values in (
            ("host_cpus", (True, False, -1, 0, 5)),
            ("host_memory_mb", (True, False, -1, 0, 8193)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(BookingError, field):
                        self._book(self._request(**{field: value}))
        self.assertEqual(self.store.load()["reservations"], [])

    def test_capacity_or_default_change_requires_explicit_policy_migration(self):
        self._book(self._request())
        for changes in (
            {"host_cpu_capacity": 5},
            {"host_memory_capacity_mb": 16384},
            {"default_host_cpus": 2},
            {"default_host_memory_mb": 2048},
            {"host_cpu_capacity": 0, "host_memory_capacity_mb": 0},
        ):
            with self.subTest(changes=changes):
                changed = replace(self.config, **changes)
                with (
                    mock.patch("bk.scheduler.utc_now", return_value=self.now),
                    self.assertRaisesRegex(BookingError, "ledger policy"),
                ):
                    add_booking(self.store, changed, self._request(uid=1002, gpu=1))

    def test_configuration_rejects_boolean_or_negative_capacity_and_invalid_defaults(
        self,
    ):
        for field in ("host_cpu_capacity", "host_memory_capacity_mb"):
            for value in (True, False, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(self.config, **{field: value})
        for field in ("default_host_cpus", "default_host_memory_mb"):
            for value in (True, False, -1, 0):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(self.config, **{field: value})

    def test_concurrent_cross_gpu_requests_cannot_both_exceed_host_budget(self):
        gate = Barrier(2)

        def submit(request):
            gate.wait(timeout=5)
            try:
                result = add_booking(self.store, self.config, request)
                return ("created", result.reservation)
            except BookingError as error:
                return ("rejected", str(error))

        requests = [
            self._request(uid=1001, gpu=0, host_cpus=4),
            self._request(uid=1002, gpu=1, host_cpus=4),
        ]
        with (
            mock.patch("bk.scheduler.utc_now", return_value=self.now),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(submit, requests))
        self.assertEqual(
            sorted(result[0] for result in results), ["created", "rejected"]
        )
        self.assertEqual(len(self.store.load()["reservations"]), 1)
        self.assertIn(
            "host_cpus",
            next(result[1] for result in results if result[0] == "rejected"),
        )


if __name__ == "__main__":
    unittest.main()
