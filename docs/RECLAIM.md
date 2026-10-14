# Reclaim reservations that never started

This laboratory extension uses the monitor's existing GPU snapshots. It does not
launch another driver sampler, signal processes or remove private command specs.

`no_show_grace_seconds` is disabled at zero in existing private/system configs.
The rootless initializer enables 3600 seconds. The policy is bound to the ledger;
clients cannot override it to bypass admission or reclaim another UID's booking.

Evidence is stored per reservation and binds its UID, start time, selected GPU
indices and stable device UUIDs. All reserved devices must have complete,
trustworthy process observations. The first sample starts an absence window.
Consecutive samples can extend it only while their gaps remain bounded by the
configured sampling cadence. Missing devices, changed identities, unknown or
ambiguous process ownership, an invisible busy device, stale telemetry or a
collector restart gap reset or block the proof.

Any observed owner process counts as use even at zero utilization. Once a
reservation has started, the evidence remains marked as started; a later idle
period never becomes a no-show. Worker claims, launch/PID markers and non-pending
job states also block reclamation.

Reclamation requires the booking to have started at least one grace interval ago
and a full grace interval of sampled absence with a fresh last sample. The
service rechecks all conditions inside one locked ledger transaction, so a
worker claim and a reclaim cannot both win. The booking is cancelled with an
audited reason; a never-claimed pending job becomes cancelled too. Repeating the
operation has no further effect.

The rootless server invokes the policy from its monitor. Its owner can also run
one bounded pass over persisted evidence:

```bash
bk reclaim --once --json
```

This command does not query GPUs. Another client UID cannot use the shared
reclaimer. In a private ledger, reclamation is limited to the executing user's
own reservations.

Exclusive timing windows intentionally pause the monitor after a command is
claimed, or immediately for a manual reservation. Paused observations cannot
prove absence. Pending managed jobs can still collect evidence before launch;
manual exclusive reservations are conservatively retained until normal expiry.

The proof is sampled and cooperative. It does not establish that a short manual
CUDA context never existed between samples, and it cannot atomically control an
external process that bypasses the managed launcher. Real process ownership and
handover still need host acceptance testing. Reservation cancellation is not
physical device revocation.
