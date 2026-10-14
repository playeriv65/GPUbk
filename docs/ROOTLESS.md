# Rootless shared deployment

This laboratory fork supports a broker configuration owned by the service UID,
as an explicit alternative to the upstream root-owned configuration. The broker
still derives caller identity from Linux SO_PEERCRED and is the only writer.

## Server owner

Install the fork with uv and choose a new protected directory. Parent directories
must be owned by root or the service account and must not be group/world writable,
except for root-owned sticky directories. All intended clients must be able to
traverse the path. The installer does not change ancestor permissions.

```bash
bk server init --directory /absolute/shared/owner/gpubk-server \
  --gpu-count 8 --cpus 64 --ram 256g
bk server serve --config /absolute/shared/owner/gpubk-server/config.json
```

Run the foreground service in a tmux window. It hosts both the broker and the
monitor, propagates broker failure, and closes its own socket on shutdown.
It does not install system services, change /etc, create users/groups or enable
linger. tmux survives SSH disconnection; it does not survive a host reboot.

The new config contains `rootless: true`, the owner's numeric `broker_uid` and
`monitor_uid`, readable protected ledger modes, a local Unix socket, host resource
capacities, a 3600-second no-show grace, a ten-second monitor interval and
`monitor_pause_on_exclusive: true`. GPU capacities are obtained from telemetry.
The GPU count is explicit so shell visibility never needs topology discovery.

The service account is a trusted operator: it can change its policy and stored
state. This is cooperative scheduling, not OS-enforced device isolation.

## Each client

```bash
bk server connect /absolute/shared/owner/gpubk-server/config.json
bk skill install --agent codex
# or: bk skill install --agent claude
bk doctor --require-monitor --strict
```

`connect` saves a private, atomic configuration pointer under the client's XDG
configuration directory. New CLI processes discover the same service without
editing shell startup files or exporting a global BK_CONFIG_FILE. Explicit
BK_CONFIG_FILE or BK_DATA_DIR still selects an intentional alternative instance.

The configured file must belong to its declared broker UID, be protected from
other writers, and have a trusted ancestor chain. Clients cannot change another
UID's reservations. Socket access does not grant direct ledger write permission.

The skill installer supports `--agent auto|codex|claude`, `--target` and `--force`.
Automatic selection honors explicit CODEX_HOME, otherwise detects installed
client directories; an ambiguous choice asks for an explicit client selection.
Codex uses its user discovery directory and Claude uses ~/.claude/skills.
Installation never rewrites AGENTS.md or CLAUDE.md. Confirm discovery in a new
agent session; implicit selection is model-driven.

## Host budgets and timing

Use `--cpus N` and `--ram 32g` on bookings, Agent recommendations and Agent edits.
MCP exposes numeric `host_cpus` and `host_memory_mb`. Budgets apply once per whole
reservation, not once per GPU. Capacity must fit in every overlapping time
segment. Future edits and extension are checked inside the same transaction;
started reservations can change their end only. Budgets are cooperative
declarations, not measured CPU impact, cgroup limits or physical RAM guarantees.

When an exclusive reservation is active, the configured monitor suppresses
periodic NVML/nvidia-smi sampling and reports unavailable volatile telemetry.
Stable cached UUIDs remain available for Bash visibility. A pending managed
command can still be observed before it is claimed. Driver queries performed by
other programs or pending launchers remain outside this monitor's control;
validate the complete deployment before collecting formal performance results.

## Acceptance boundary

Automated validation uses synthetic GPU snapshots, Linux local IPC, fake clock
evidence and actual disposable Bash processes. Kernel identity is exercised for
the current UID; foreign-UID authorization is also tested through injected peer
credentials. A real second-user filesystem/socket test, reboot persistence and
real GPU launch still require a host acceptance run. Do not infer these from
simulation or advertise hardware quotas.
