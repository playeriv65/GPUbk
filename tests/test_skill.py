# Modified for the playeriv65 laboratory fork; see NOTICE.
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bk.models import BookingError
from bk.skill import SKILL_NAME, default_skill_path, install_skill, skill_text


class BundledSkillTests(unittest.TestCase):
    def test_client_selection_uses_discoverable_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch.dict(os.environ, {"HOME": tmp}, clear=True):
                self.assertEqual(
                    default_skill_path("codex"), home / ".agents/skills/gpubk"
                )
                self.assertEqual(
                    default_skill_path("claude"), home / ".claude/skills/gpubk"
                )
                self.assertEqual(
                    default_skill_path("auto"), home / ".agents/skills/gpubk"
                )
                (home / ".claude").mkdir()
                self.assertEqual(
                    default_skill_path("auto"), home / ".claude/skills/gpubk"
                )
                (home / ".codex").mkdir()
                with self.assertRaisesRegex(BookingError, "both Codex and Claude"):
                    default_skill_path("auto")
                os.environ["CODEX_HOME"] = str(home / "custom-codex")
                self.assertEqual(
                    default_skill_path("auto"), home / "custom-codex/skills/gpubk"
                )

    def test_install_changes_only_skill_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".claude").mkdir()
            instructions = home / ".claude/CLAUDE.md"
            instructions.write_text("Keep these instructions.\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"HOME": tmp}, clear=True):
                installed = install_skill(agent="claude")
            self.assertEqual(installed, home / ".claude/skills/gpubk")
            self.assertEqual(
                instructions.read_text(encoding="utf-8"), "Keep these instructions.\n"
            )
            self.assertFalse((home / ".codex/AGENTS.md").exists())
            self.assertFalse((home / ".agents/AGENTS.md").exists())

    def test_default_path_uses_only_an_absolute_codex_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            explicit = root / "codex-home"

            with mock.patch.dict(
                os.environ,
                {"HOME": str(home), "CODEX_HOME": str(explicit)},
                clear=True,
            ):
                self.assertEqual(default_skill_path(), explicit / "skills" / SKILL_NAME)
            for value in ("", "relative/codex-home"):
                with mock.patch.dict(
                    os.environ,
                    {"HOME": str(home), "CODEX_HOME": value},
                    clear=True,
                ):
                    self.assertEqual(
                        default_skill_path(),
                        home / ".codex" / "skills" / SKILL_NAME,
                    )

    def test_bundled_skill_has_expected_trigger_metadata(self):
        text = skill_text()

        self.assertEqual(SKILL_NAME, "gpubk")
        self.assertIn(f"name: {SKILL_NAME}", text)
        self.assertIn("expected VRAM", text)
        self.assertIn("operation ID", text)
        self.assertIn("edit_my_gpu_booking", text)
        self.assertIn("bk agent edit", text)
        self.assertIn("collector.fresh", text)
        self.assertIn("--require-monitor", text)
        self.assertIn("--exclude-gpu", text)
        self.assertIn("policy.disabled_gpus", text)
        self.assertIn("remote-only catalog", text)
        self.assertIn("cluster delete", text)

        protocol = (
            Path(__file__).parents[1]
            / "src"
            / "bk"
            / "data"
            / "codex-skill"
            / "gpubk"
            / "references"
            / "protocol.md"
        ).read_text(encoding="utf-8")
        self.assertIn("enabled_gpus", protocol)
        self.assertIn("disabled_gpus", protocol)
        self.assertIn("request-excluded", protocol)
        self.assertIn("remote-only catalog", protocol)
        self.assertIn("/etc/gpubk/cluster.json", protocol)

    def test_install_is_complete_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / SKILL_NAME

            installed = install_skill(destination)

            self.assertEqual(installed, destination)
            self.assertTrue((destination / "SKILL.md").is_file())
            self.assertTrue((destination / "agents" / "openai.yaml").is_file())
            self.assertTrue((destination / "references" / "protocol.md").is_file())
            with self.assertRaisesRegex(BookingError, "already exists"):
                install_skill(destination)

    def test_force_only_replaces_a_recognized_gpubk_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "unrelated"
            wrong.mkdir()
            (wrong / "SKILL.md").write_text("name: unrelated\n", encoding="utf-8")

            with self.assertRaisesRegex(BookingError, "unrecognized directory"):
                install_skill(wrong, force=True)

            destination = Path(tmp) / SKILL_NAME
            install_skill(destination)
            (destination / "stale.txt").write_text("stale", encoding="utf-8")
            install_skill(destination, force=True)
            self.assertFalse((destination / "stale.txt").exists())

    def test_force_refuses_the_active_working_tree_without_removing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / SKILL_NAME
            install_skill(destination)
            sentinel = destination / "user-file.txt"
            sentinel.write_text("keep", encoding="utf-8")
            original = Path.cwd()
            os.chdir(destination / "references")
            try:
                with self.assertRaisesRegex(BookingError, "current working directory"):
                    install_skill(destination, force=True)
            finally:
                os.chdir(original)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_force_rejects_a_symbolic_link_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            install_skill(target)
            sentinel = target / "user-file.txt"
            sentinel.write_text("keep", encoding="utf-8")
            destination = root / SKILL_NAME
            destination.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(BookingError, "unrecognized directory"):
                install_skill(destination, force=True)

            self.assertTrue(destination.is_symlink())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_failed_force_install_restores_the_previous_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / SKILL_NAME
            install_skill(destination)
            sentinel = destination / "user-file.txt"
            sentinel.write_text("keep", encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_new_skill(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated replacement failure")
                return real_replace(source, target)

            with (
                mock.patch("bk.skill.os.replace", side_effect=fail_new_skill),
                self.assertRaisesRegex(OSError, "simulated replacement failure"),
            ):
                install_skill(destination, force=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(destination.parent.glob(".gpubk.*")), [])


if __name__ == "__main__":
    unittest.main()
