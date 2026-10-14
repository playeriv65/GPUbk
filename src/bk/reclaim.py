"""Lab fork addition: conservative no-show reclamation without process signals.

Evidence is collected from the monitor's existing snapshots. A low utilization
reading alone never establishes absence, and incomplete telemetry fails closed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta

from .config import Config
from .gpu import GpuSnapshot, has_process_telemetry, has_stable_device_identifier
from .models import (
    JOB_CANCELLED,
    JOB_PENDING,
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    Actor,
    BookingError,
)
from .policy import validate_ledger_policy
from .storage import LedgerStore
from .timeparse import parse_iso, to_iso, utc_now
from .usage import GPU_LIVE_BUSY, assess_gpu_live_states

EVIDENCE_VERSION = 1
# Launch markers must be checked together with job state under the ledger lock.
LAUNCH_MARKERS = (
    "claimed_at",
    "claim_token",
    "started_at",
    "runner_pid",
    "worker_id",
    "worker_lease_id",
    "recovered_at",
)


def _grace_seconds(config: Config) -> int:
    return int(getattr(config, "no_show_grace_seconds", 0))


def _maximum_gap_seconds(config: Config) -> float:
    # Permit bounded scheduling jitter, not a missing monitoring window.
    return max(1.0, float(config.monitor_interval_seconds) * 2.5)


def _authorize_observer(config: Config) -> None:
    current_uid = os.getuid()
    allowed = {
        uid for uid in (config.monitor_uid, config.broker_uid) if uid is not None
    }
    if allowed and current_uid not in allowed:
        raise BookingError(
            "only the configured monitor or broker owner may record no-show evidence"
        )
    if config.dir_mode & 0o022 and not allowed:
        raise BookingError(
            "shared no-show evidence requires a configured monitor or broker owner"
        )


def _authorize_reclaimer(config: Config, actor: Actor) -> bool:
    """Return whether the service actor may reclaim reservations of all UIDs."""
    if actor.uid != os.getuid():
        raise BookingError("no-show reclamation actor must match the executing UID")
    if config.broker_uid is not None:
        if actor.uid != config.broker_uid:
            raise BookingError("only the configured broker owner may reclaim no-shows")
        return True
    return False


def _binding(reservation: dict) -> dict:
    return {
        "uid": int(reservation["uid"]),
        "start_at": str(reservation["start_at"]),
        "gpus": [int(gpu) for gpu in reservation["gpus"]],
    }


def _job_has_started(reservation: dict) -> bool:
    job = reservation.get("job")
    if not isinstance(job, dict):
        return False
    return job.get("status") != JOB_PENDING or any(
        job.get(key) for key in LAUNCH_MARKERS
    )


def _snapshot_absence(
    reservation: dict, snapshots: Sequence[GpuSnapshot]
) -> tuple[bool, bool, str, list[str]]:
    """Return trusted absence, observed owner use, reason and device identities."""
    uid = int(reservation["uid"])
    by_gpu = {device.index: device for device in snapshots}
    identities = []
    observed_use = False
    reason = "absent"
    live_states = assess_gpu_live_states(snapshots, max(reservation["gpus"]) + 1)
    for gpu in reservation["gpus"]:
        device = by_gpu.get(gpu)
        if device is not None:
            observed_use = observed_use or any(
                process.uid == uid or process.host_uid == uid
                for process in device.processes
            )
        if (
            device is None
            or device.source == "none"
            or not has_process_telemetry(device)
        ):
            reason = "process-telemetry-unavailable"
            continue
        if not has_stable_device_identifier(device):
            reason = "device-identity-unavailable"
            continue
        identities.append(device.device_uuid)
        for process in device.processes:
            if (
                process.uid is None
                or process.identity_source != "host"
                or not process.host_start_id
            ):
                reason = "process-identity-uncertain"
        # Activity with no visible process can reflect an invisible holder.
        if not device.processes and live_states[gpu].status == GPU_LIVE_BUSY:
            reason = "unattributed-device-activity"
    if observed_use:
        return False, True, "owner-process-observed", identities
    return reason == "absent", False, reason, identities


def record_no_show_observations(
    store: LedgerStore,
    config: Config,
    snapshots: Sequence[GpuSnapshot],
    sampled_at: datetime,
) -> dict:
    """Persist one sample of proof; this function does not sample or reclaim GPUs.

    A telemetry gap resets the absence window. Observed use and durable launch
    markers are sticky for the reservation, including across monitor restarts.
    """
    if _grace_seconds(config) == 0:
        return {"observed": 0, "changed": 0, "disabled": True}
    _authorize_observer(config)
    maximum_gap = _maximum_gap_seconds(config)

    def mutate(ledger: dict):
        validate_ledger_policy(ledger, config)
        observed = 0
        changed = 0
        for reservation in ledger.get("reservations", []):
            if reservation.get("status") != STATUS_ACTIVE:
                continue
            start = parse_iso(reservation["start_at"])
            end = parse_iso(reservation["end_at"])
            if not start <= sampled_at < end:
                continue
            observed += 1
            binding = _binding(reservation)
            previous = reservation.get("no_show")
            previous = previous if isinstance(previous, dict) else {}
            previous_valid = (
                previous.get("version") == EVIDENCE_VERSION
                and previous.get("binding") == binding
            )
            if previous_valid and previous.get("ever_started"):
                # A used reservation can never become a no-show. Avoid repeated
                # ledger writes for running experiments and preserve this fact.
                continue
            ever_started = bool(previous.get("ever_started")) or _job_has_started(
                reservation
            )
            absent, owner_use, reason, identities = _snapshot_absence(
                reservation, snapshots
            )
            ever_started = ever_started or owner_use
            previous_at = (
                _timestamp(previous.get("last_observed_at")) if previous_valid else None
            )
            if previous_at is not None and sampled_at <= previous_at:
                # An out-of-order sample must not overwrite newer trusted proof.
                if ever_started and not previous.get("ever_started"):
                    reservation["no_show"] = {
                        **previous,
                        "ever_started": True,
                        "reason": "ever-started",
                    }
                    changed += 1
                continue
            consecutive = (
                previous_at is not None
                and (sampled_at - previous_at).total_seconds() <= maximum_gap
                and previous.get("device_uuids") == identities
                and previous.get("reason") == "absent"
            )
            absent_since = (
                previous.get("absent_since") if absent and consecutive else None
            )
            if absent and absent_since is None:
                absent_since = to_iso(sampled_at)
            evidence = {
                "version": EVIDENCE_VERSION,
                "binding": binding,
                "device_uuids": identities,
                "last_observed_at": to_iso(sampled_at),
                "absent_since": absent_since,
                "sample_count": int(previous.get("sample_count", 0)) + 1
                if consecutive
                else 1,
                "ever_started": ever_started,
                "reason": "ever-started" if ever_started else reason,
            }
            if evidence != previous:
                reservation["no_show"] = evidence
                changed += 1
        return (
            ledger,
            {"observed": observed, "changed": changed, "disabled": False},
            [],
            bool(changed),
        )

    return store.transaction(mutate)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_iso(value)
    except (TypeError, ValueError):
        return None


def _retention_reason(reservation: dict, config: Config, now: datetime) -> str | None:
    if reservation.get("status") != STATUS_ACTIVE:
        return "not-active"
    start = parse_iso(reservation["start_at"])
    if now < start + timedelta(seconds=_grace_seconds(config)):
        return "within-start-grace"
    if parse_iso(reservation["end_at"]) <= now:
        return "reservation-ended"
    if _job_has_started(reservation):
        return "job-started-or-claimed"
    evidence = reservation.get("no_show")
    if not isinstance(evidence, dict) or evidence.get("version") != EVIDENCE_VERSION:
        return "missing-evidence"
    if evidence.get("binding") != _binding(reservation):
        return "evidence-binding-changed"
    if evidence.get("ever_started"):
        return "ever-started"
    if evidence.get("reason") != "absent":
        return str(evidence.get("reason", "absence-unverified"))
    identities = evidence.get("device_uuids")
    if not isinstance(identities, list) or len(identities) != len(reservation["gpus"]):
        return "device-identity-unavailable"
    last = _timestamp(evidence.get("last_observed_at"))
    since = _timestamp(evidence.get("absent_since"))
    if last is None or since is None or since < start or last < since:
        return "invalid-evidence"
    age = (now - last).total_seconds()
    if age < 0 or age > _maximum_gap_seconds(config):
        return "stale-evidence"
    if (last - since).total_seconds() < _grace_seconds(config):
        return "insufficient-absence-history"
    return None


def reclaim_no_shows(
    store: LedgerStore,
    config: Config,
    actor: Actor,
    *,
    now: datetime | None = None,
) -> dict:
    """Reclaim verified no-shows once, atomically with worker claims and edits.

    This never signals a process or removes a job specification. Pending jobs
    become cancelled, preventing later worker claims after capacity is released.
    """
    all_uids = _authorize_reclaimer(config, actor)
    current = now or utc_now()
    if _grace_seconds(config) == 0:
        return {"reclaimed": [], "retained": [], "disabled": True}

    def mutate(ledger: dict):
        validate_ledger_policy(ledger, config)
        reclaimed = []
        retained = []
        logs = []
        for reservation in ledger.get("reservations", []):
            if reservation.get("status") != STATUS_ACTIVE:
                continue
            if not all_uids and int(reservation["uid"]) != actor.uid:
                continue
            reason = _retention_reason(reservation, config, current)
            if reason is not None:
                retained.append({"reservation_id": reservation["id"], "reason": reason})
                continue
            reservation["status"] = STATUS_CANCELLED
            reservation["updated_at"] = to_iso(current)
            reservation["no_show"]["reclaimed_at"] = to_iso(current)
            reservation["no_show"]["reclaimed_by_uid"] = actor.uid
            job = reservation.get("job")
            if isinstance(job, dict):
                job["status"] = JOB_CANCELLED
                job["finished_at"] = to_iso(current)
                job["message"] = "no-show reclaimed after verified absence"
            reservation_id = str(reservation["id"])
            reclaimed.append(reservation_id)
            logs.append(
                {
                    "ts": to_iso(current),
                    "uid": actor.uid,
                    "username": actor.username,
                    "action": "no-show-reclaim",
                    "reservation_id": reservation_id,
                    "gpus": reservation["gpus"],
                    "mode": reservation["mode"],
                    "start_at": reservation["start_at"],
                    "end_at": reservation["end_at"],
                    "result": "ok",
                    "message": "reclaimed after continuous verified absence; no process was signalled",
                }
            )
        result = {"reclaimed": reclaimed, "retained": retained, "disabled": False}
        return ledger, result, logs, bool(reclaimed)

    return store.transaction(mutate)
