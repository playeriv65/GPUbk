import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from bk.collector_status import collector_document
from bk.config import Config
from bk.models import Actor, BookingRequest
from bk.scheduler import add_booking, cancel_booking
from bk.shell import bash_environment, environment_payload, shell_path
from bk.storage import LedgerStore
from bk.timeparse import to_iso
from bk.usage_store import UsageAuditStore

ROOT = Path(__file__).resolve().parents[1]
UUIDS = [
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000001",
]


class ShellVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = Config(data_dir=self.root / "data", gpu_count=2)
        self.store = LedgerStore(self.config.data_dir)
        self.actor = Actor(os.getuid(), "test-user")
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.start = self.now.replace(
            minute=self.now.minute - self.now.minute % 5, second=0
        )
        self._write_collector()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_collector(self, **changes):
        devices = [
            {
                "gpu": gpu,
                "source": "nvml",
                "device_telemetry": True,
                "stable_device_identifier": True,
                "device_uuid": UUIDS[gpu],
                "process_telemetry": True,
                "process_utilization": True,
            }
            for gpu in range(2)
        ]
        values = {
            "monitor_id": "shell-test",
            "status": "running",
            "uid": os.getuid(),
            "pid": os.getpid(),
            "hostname": "test-host",
            "heartbeat_interval_seconds": 60,
            "sample_interval_seconds": 10,
            "rollup_seconds": 60,
            "started_at": self.now - timedelta(minutes=10),
            "sampled_at": self.now,
            "devices": devices,
            "stable_device_identifier_gap": [],
            "process_telemetry_gap": [],
            "process_identity_gap": [],
            "process_utilization_gap": [],
        }
        values.update(changes)
        UsageAuditStore(self.config.data_dir).save_collector_status(
            collector_document(**values)
        )

    def _book(self, gpu, *, actor=None, start=None):
        with mock.patch("bk.scheduler.utc_now", return_value=self.now):
            return add_booking(
                self.store,
                self.config,
                BookingRequest(
                    actor=actor or self.actor,
                    count=1,
                    duration_seconds=3600,
                    start_at=start or self.start,
                    preferred_gpus=[gpu],
                ),
            ).reservation

    def _payload(self, **kwargs):
        with mock.patch.dict(os.environ, {"BK_GPU_SIM_FILE": ""}):
            return environment_payload(
                self.config, self.store, actor=self.actor, now=self.now, **kwargs
            )

    def test_union_excludes_other_users_future_expired_and_cancelled(self):
        first = self._book(0)
        second = self._book(1)
        payload = self._payload()
        self.assertEqual(payload["cuda_visible_devices"], ",".join(UUIDS))
        self.assertEqual(set(payload["reservation_ids"]), {first["id"], second["id"]})
        cancel_booking(self.store, first["id"], self.actor)
        self.assertEqual(self._payload()["cuda_visible_devices"], UUIDS[1])
        self._book(0, actor=Actor(self.actor.uid + 1, "other"))
        self.assertEqual(self._payload()["cuda_visible_devices"], UUIDS[1])
        cancel_booking(self.store, second["id"], self.actor)
        self._book(1, start=self.start + timedelta(hours=2))
        self.assertEqual(self._payload()["cuda_visible_devices"], "")
        with mock.patch.dict(os.environ, {"BK_GPU_SIM_FILE": ""}):
            late = environment_payload(
                self.config,
                self.store,
                actor=self.actor,
                now=self.start + timedelta(hours=4),
            )
        self.assertEqual(late["cuda_visible_devices"], "")
        self.assertEqual(bash_environment(late), "export CUDA_VISIBLE_DEVICES=''\n")

    def test_cached_snapshot_never_probes_driver(self):
        self._book(1)
        with mock.patch(
            "bk.shell.snapshot", side_effect=AssertionError("driver probe")
        ):
            self.assertEqual(self._payload()["cuda_visible_devices"], UUIDS[1])

    def test_stale_missing_or_unsafe_identity_fails_closed(self):
        self._book(1)
        path = self.config.data_dir / "usage/collector.json"
        payload = json.loads(path.read_text())
        payload["written_at"] = to_iso(self.now - timedelta(minutes=5))
        payload["sampled_at"] = payload["written_at"]
        path.write_text(json.dumps(payload))
        self.assertEqual(self._payload()["status"], "unavailable")
        self._write_collector()
        payload = json.loads(path.read_text())
        payload["devices"][1]["device_uuid"] = "GPU-$(touch unwanted)"
        path.write_text(json.dumps(payload))
        self.assertEqual(self._payload()["cuda_visible_devices"], "")
        self._write_collector()
        path.chmod(0o666)
        self.assertEqual(self._payload()["status"], "unavailable")
        path.unlink()
        self.assertEqual(self._payload()["status"], "unavailable")

    def test_other_device_telemetry_gap_preserves_selected_trusted_identity(self):
        self._book(1)
        path = self.config.data_dir / "usage/collector.json"
        document = json.loads(path.read_text())
        document["status"] = "degraded"
        document["devices"][0].update(
            source="none",
            device_telemetry=False,
            stable_device_identifier=False,
            device_uuid="",
            process_telemetry=False,
            process_utilization=False,
        )
        for key in (
            "stable_device_identifier_gap",
            "process_telemetry_gap",
            "process_identity_gap",
        ):
            document[key] = [0]
        document["process_utilization_gap"] = []
        path.write_text(json.dumps(document))
        self.assertEqual(self._payload()["cuda_visible_devices"], UUIDS[1])
        self._book(0)
        self.assertEqual(self._payload()["status"], "unavailable")
        self.assertEqual(self._payload()["cuda_visible_devices"], "")

    def test_absent_broker_does_not_grant_cached_reservations(self):
        self._book(1)
        config = replace(
            self.config,
            broker_socket=self.root / "missing.sock",
            broker_uid=os.getuid(),
            file_mode=0o644,
            dir_mode=0o755,
            broker_socket_mode=0o666,
        )
        payload = environment_payload(
            config, self.store, actor=self.actor, now=self.now
        )
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(bash_environment(payload), "export CUDA_VISIBLE_DEVICES=''\n")

    def _bash(self, script, *, interactive=False, extra_env=None):
        bin_dir = self.root / "bin"
        bin_dir.mkdir(exist_ok=True)
        wrapper = bin_dir / "bk"
        wrapper.write_text(
            "#!"
            + sys.executable
            + "\nfrom bk.entrypoint import main\nraise SystemExit(main())\n"
        )
        wrapper.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": str(bin_dir) + os.pathsep + env["PATH"],
                "PYTHONPATH": str(ROOT / "src"),
                "BK_GPU_COUNT": "2",
                "BK_DATA_DIR": str(self.config.data_dir),
                "BK_GPU_SIM_FILE": "",
                "BASH_ENV": "",
                "CUDA_VISIBLE_DEVICES": "all",
            }
        )
        env.pop("GPUBK_PREVIOUS_BASH_ENV", None)
        if extra_env:
            env.update(extra_env)
        command = ["/bin/bash", "--noprofile", "--norc"]
        if interactive:
            command.append("-i")
        return subprocess.run(
            command,
            input=script,
            text=True,
            capture_output=True,
            env=env,
            timeout=20,
            check=False,
        )

    def test_live_bash_refresh_child_inheritance_and_previous_bash_env(self):
        self._book(1)
        previous = self.root / "previous env.bash"
        previous.write_text("export PREVIOUS_HOOK_RAN=yes\n")
        result = self._bash(
            f"source {shlex.quote(str(shell_path()))}\n"
            "printf 'parent=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
            'bash -c \'printf "child=%s previous=%s\\n" "$CUDA_VISIBLE_DEVICES" "$PREVIOUS_HOOK_RAN"\'\n',
            extra_env={"BASH_ENV": str(previous)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("parent=" + UUIDS[1], result.stdout)
        self.assertIn("child=" + UUIDS[1] + " previous=yes", result.stdout)

    def test_live_interactive_hook_preserves_prompt_array_and_is_idempotent(self):
        self._book(1)
        hook = shlex.quote(str(shell_path()))
        result = self._bash(
            "PROMPT_COMMAND=('printf original-one' 'printf original-two')\n"
            f"source {hook}\nsource {hook}\n"
            'printf \'\\nvisibility=%s hooks=%s\\n\' "$CUDA_VISIBLE_DEVICES" "${#PROMPT_COMMAND[@]}"\nexit\n',
            interactive=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("visibility=" + UUIDS[1] + " hooks=3", result.stdout)
        self.assertIn("original-oneoriginal-two", result.stdout)

    def test_live_interactive_hook_preserves_prompt_string(self):
        hook = shlex.quote(str(shell_path()))
        result = self._bash(
            "PROMPT_COMMAND='printf original-string'\n"
            f"source {hook}\n"
            'printf \'\\nhooks=%s empty=%s\\n\' "${#PROMPT_COMMAND[@]}" "$CUDA_VISIBLE_DEVICES"\nexit\n',
            interactive=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("hooks=2 empty=\n", result.stdout)
        self.assertIn("original-string", result.stdout)

    def test_live_bash_expiry_refreshes_parent_and_new_child(self):
        self._book(1)
        path = str(self.store.ledger_path)
        expired_start = to_iso(self.now - timedelta(hours=2))
        expired_end = to_iso(self.now - timedelta(hours=1))
        expire = (
            "import json; from pathlib import Path; "
            f"p=Path({path!r}); d=json.loads(p.read_text()); "
            f"d['reservations'][0]['start_at']={expired_start!r}; "
            f"d['reservations'][0]['end_at']={expired_end!r}; p.write_text(json.dumps(d))"
        )
        result = self._bash(
            f"source {shlex.quote(str(shell_path()))}\n"
            "printf 'before=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
            f"{shlex.quote(sys.executable)} -c {shlex.quote(expire)}\n"
            "bk_gpu_refresh\nprintf 'expired=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
            'bash -c \'printf "new-child=%s\\n" "$CUDA_VISIBLE_DEVICES"\'\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("before=" + UUIDS[1], result.stdout)
        self.assertIn("expired=\n", result.stdout)
        self.assertIn("new-child=\n", result.stdout)

    def test_live_bash_release_and_service_failure_clear_visibility(self):
        reservation = self._book(1)
        hook = shlex.quote(str(shell_path()))
        result = self._bash(
            f"source {hook}\n"
            "printf 'before=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
            f"bk agent cancel {reservation['id']} --compact >/dev/null\n"
            "bk_gpu_refresh\nprintf 'released=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n"
            "export BK_DATA_DIR=/dev/null/unavailable\n"
            "export CUDA_VISIBLE_DEVICES=all\nbk_gpu_refresh\n"
            "printf 'failed=%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("before=" + UUIDS[1], result.stdout)
        self.assertIn("released=\n", result.stdout)
        self.assertIn("failed=\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
