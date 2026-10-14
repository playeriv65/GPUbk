# Bash visibility and agent Skills

The Skill installer writes only the selected Skill directory. It never modifies
`AGENTS.md`, `CLAUDE.md`, client settings, or shell startup files.

```bash
bk skill install --agent codex
bk skill install --agent claude
bk skill path --agent codex
```

Codex uses `~/.agents/skills/gpubk`, or `$CODEX_HOME/skills/gpubk` when a custom
absolute `CODEX_HOME` is configured. Claude Code uses `~/.claude/skills/gpubk`.
Automatic selection prefers an explicit `CODEX_HOME`; otherwise it detects the
installed client directory. When both clients are detected, select one explicitly.
`--target` accepts an exact destination and `--force` safely replaces only a
recognized existing GPUBK Skill. Start a new client session after installation.
Test discovery with a natural-language GPU reservation request. Skill selection
is a client behavior; the installer cannot guarantee a model's automatic choice.

## Enable visibility in a Bash session

```bash
eval "$(bk shell init)"
```

This explicitly sources the packaged Bash integration. It refreshes
`CUDA_VISIBLE_DEVICES` at startup and before each interactive prompt. Existing
`PROMPT_COMMAND` entries remain intact; sourcing twice does not duplicate hooks.
It sets `BASH_ENV` so newly started noninteractive Bash processes refresh their
reservation state, while preserving an existing startup file through
`GPUBK_PREVIOUS_BASH_ENV`. Existing `BASH_ENV` values should be literal file paths.

For persistence, add the same source command to the user's Bash startup file
manually. GPUBK does not install it there automatically. Each independently
started shell must source the hook to participate. A shell showing an old prompt
updates on its next prompt; already running processes retain their environment.

```bash
bk_gpu_refresh
bk env --json
bk env --bash
```

`bk env --bash` prints an export; a child command cannot change its parent shell.
When manually evaluating an export, clear visibility first so command failure also
leaves an empty value:

```bash
export CUDA_VISIBLE_DEVICES=''
eval "$(bk env --bash)"
```

## Visibility policy

Visibility is the union of the current Linux UID's active reservations whose
start time has arrived and end time remains in the future. Future, expired,
cancelled, disabled-device, and other-UID reservations grant no visibility.
Agents sharing a UID share that UID's visibility.

Real devices require stable UUIDs from a fresh collector snapshot owned by the
configured monitor. Prompt refresh reads files and contacts the configured broker;
it never queries NVML or `nvidia-smi`. The `bk env` command disables hardware
auto-detection. Configure `gpu_count` explicitly in the shared server policy. A missing
broker, stale snapshot, corrupt ledger, recovered backup, or unsafe/missing UUID
leaves visibility empty. There is no numeric-device fallback. The local
`BK_GPU_SIM_FILE` path is an explicit CPU-only simulation facility.

Release and expiry set `CUDA_VISIBLE_DEVICES` to an **empty string**, rather than
unsetting it. Unsetting it would normally expose every accessible GPU. This is
cooperative protection against accidental use; users can change their environment,
and programs that ignore CUDA visibility require separate device access controls.

## Declare host CPU and RAM

The booking, recommendation, and edit interfaces accept host budgets separately
from GPU memory:

```bash
bk agent recommend 1 1h --gpu 1 --mem 12g --cpus 8 --ram 32g --compact
bk 1 1h --gpu 1 --mem 12g --cpus 8 --ram 32g --op-id my-stable-id --json
bk agent edit <reservation-id> --cpus 4 --ram 16g --op-id my-edit-id --compact
```

`--mem` means VRAM **per GPU**; `--ram` means RAM for the entire reservation.
Host CPU and RAM admission is checked against all overlapping bookings, including
bookings on other GPUs. The server policy controls capacities and defaults.
Disabled capacity checks do not impose a host limit. These cooperative budgets
do not throttle CPU use or physically restrict process memory.
Budget edits apply only before a reservation starts; started reservations can
change only their end time.

MCP clients use `host_cpus` and `host_memory_mb`, with RAM in MiB; 32 GiB is
`32768` MiB. Preserve requested budgets when creating or retrying a reservation.
