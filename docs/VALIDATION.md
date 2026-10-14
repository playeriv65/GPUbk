# Laboratory fork validation

The 2026-10-14 development validation used Linux and Python 3.12.13 with the
checked-in uv lock file. It ran no GPU computation and made no hardware timing
claims.

- Ruff and bytecode compilation passed.
- Bandit reported no medium-or-higher findings in source and tools.
- The complete regression suite passed: 1192 tests, one skipped.
- The same suite passed under `python -O`: 1192 tests, one skipped.
- Combined branch-aware coverage was 76%, above the configured 65% floor.
- Source and wheel distributions built successfully.

The new tests exercise atomic cross-GPU host-budget conflicts, concurrent
admission, recommendations, retries, edits and renewals. Reclamation tests cover
continuous absence, observed idle processes, stale or ambiguous evidence, worker
claims, ownership and audited idempotence. Disposable Bash processes exercise
prompt hooks, new shells, recursion protection and empty visibility on failure.

The CLI/MCP integration test uses a real local Unix broker socket and synthetic
device identities. It exercises booking, retry, queue recommendation, budget
edits, visibility, cancellation and Skill-only installation. Actual current-UID
credentials are tested; foreign-UID policy tests inject peer credentials.

A disposable foreground `server serve` process was also exercised with synthetic
devices in tmux. A client discovered its saved connection pointer, booked GPU 1,
obtained its UUID visibility and cleared it on release. Stopping that owned
service removed its socket and left visibility empty despite an active booking.

A private historical request trace was also replayed through the fork's
scheduler with synthetic eight-device capacity. Its adapter rounds starts down
and durations up to one-minute slots, maps identities to synthetic UIDs, and
releases only an entire owned reservation matching the requested device set.
No exclusive-overlap or host-budget invariant failed. No-show behavior is tested
separately because messages do not prove process absence. Raw messages,
identities and replay outputs remain outside this public repository.

## Remaining host acceptance

Verify access with a second real UID, live NVML process attribution, a managed
CUDA launch and handover on the intended shared host before general deployment.
Administrator-controlled reboot persistence is separate from the rootless tmux
service. Environment visibility and resource budgets remain cooperative.
Exclusive monitor pauses retain manual no-shows until ordinary expiry, and
external diagnostic samplers remain outside GPUBK's control.
