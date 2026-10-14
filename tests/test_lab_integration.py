"""CPU-only CLI, MCP and Linux broker acceptance for the laboratory fork."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path

from bk.advisor import build_gpu_advice
from bk.broker import BrokerLedgerStore, BrokerServer
from bk.config import load_config
from bk.mcp_server import BkMcpBackend
from bk.monitor import UsageMonitor
from bk.rootless import initialize_server
from bk.usage_store import UsageAuditStore
from bk.timeparse import parse_iso, to_iso


class LabBrokerIntegrationTests(unittest.TestCase):
    def test_cli_budget_recommendation_retry_visibility_cancel_and_skill_only_install(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="bk-lab-") as tmp:
            root = Path(tmp)
            simulation = root / "simulation.json"
            uuids = [
                f"GPU-00000000-0000-0000-0000-00000000000{gpu}" for gpu in range(2)
            ]
            simulation.write_text(
                json.dumps(
                    {
                        "gpus": [
                            {
                                "index": gpu,
                                "name": "synthetic",
                                "uuid": uuids[gpu],
                                "memory_total_mb": 96000,
                                "memory_used_mb": 0,
                                "utilization_percent": 0,
                                "processes": [],
                            }
                            for gpu in range(2)
                        ]
                    }
                )
            )
            config_path = initialize_server(
                root / "server", gpu_count=2, cpus=8, memory_mb=65536
            )
            config = load_config(config_path)
            env = {
                **os.environ,
                "BK_CONFIG_FILE": str(config_path),
                "BK_GPU_SIM_FILE": str(simulation),
                "XDG_STATE_HOME": str(root / "state"),
                "XDG_CONFIG_HOME": str(root / "client"),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            }

            def command(*args, expected=0):
                result = subprocess.run(
                    [sys.executable, "-m", "bk", *args],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(
                    result.returncode, expected, result.stderr + result.stdout
                )
                return result.stdout

            from unittest import mock

            with mock.patch.dict(os.environ, env):
                server = BrokerServer(config, advice_provider=build_gpu_advice)
                server.prepare()
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    monitor = UsageMonitor(
                        config,
                        server.store,
                        UsageAuditStore(
                            config.data_dir,
                            file_mode=config.file_mode,
                            dir_mode=config.dir_mode,
                        ),
                    )
                    monitor.collect()
                    args = (
                        "1",
                        "1h",
                        "--gpu",
                        "1",
                        "--mem",
                        "12g",
                        "--cpus",
                        "4",
                        "--ram",
                        "16g",
                        "--op-id",
                        "integration-book",
                        "--json",
                    )
                    booking = json.loads(command(*args))
                    self.assertEqual(booking["status"], "created")
                    identifier = booking["reservation"]["id"]
                    self.assertEqual(booking["reservation"]["host_cpus"], 4)
                    self.assertEqual(json.loads(command(*args))["status"], "exists")
                    future = json.loads(
                        command(
                            "agent",
                            "recommend",
                            "1",
                            "1h",
                            "--gpu",
                            "0",
                            "--mem",
                            "12g",
                            "--cpus",
                            "5",
                            "--ram",
                            "16g",
                            "--compact",
                        )
                    )
                    self.assertTrue(future["available"])
                    self.assertGreater(
                        future["recommendation"]["start_at"],
                        booking["reservation"]["start_at"],
                    )
                    visible = json.loads(command("env", "--json"))
                    self.assertEqual(visible["cuda_visible_devices"], uuids[1])
                    command("agent", "cancel", identifier, "--compact")
                    self.assertEqual(
                        json.loads(command("env", "--json"))["cuda_visible_devices"], ""
                    )
                    api = BkMcpBackend(config, BrokerLedgerStore(config))
                    result = api.book(
                        1,
                        "1h",
                        "mcp-budget",
                        gpus=[0],
                        expected_memory="12g",
                        host_cpus=3,
                        host_memory_mb=4096,
                    )
                    self.assertEqual(result["reservation"]["host_cpus"], 3)
                    self.assertEqual(result["reservation"]["host_memory_mb"], 4096)
                    planned = api.book(
                        1,
                        "1h",
                        "future-budget",
                        gpus=[1],
                        expected_memory="12g",
                        start=to_iso(
                            parse_iso(result["reservation"]["end_at"])
                            + timedelta(hours=1)
                        ),
                        host_cpus=2,
                        host_memory_mb=2048,
                    )
                    planned_id = planned["reservation"]["id"]
                    edited = api.edit(
                        planned_id, "future-mcp-edit", host_cpus=4, host_memory_mb=4096
                    )
                    self.assertEqual(edited["reservation"]["host_cpus"], 4)
                    cli_edited = json.loads(
                        command(
                            "agent",
                            "edit",
                            planned_id,
                            "--cpus",
                            "3",
                            "--ram",
                            "8g",
                            "--op-id",
                            "future-cli-edit",
                            "--compact",
                        )
                    )
                    self.assertEqual(cli_edited["reservation"]["host_cpus"], 3)
                    self.assertEqual(cli_edited["reservation"]["host_memory_mb"], 8192)
                    canary = root / "AGENTS.md"
                    canary.write_text("existing instructions\n")
                    target = root / "skills" / "gpubk"
                    command(
                        "skill", "install", "--agent", "codex", "--target", str(target)
                    )
                    self.assertTrue((target / "SKILL.md").is_file())
                    self.assertEqual(canary.read_text(), "existing instructions\n")
                finally:
                    server.close()
                    thread.join(timeout=5)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
