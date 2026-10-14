# Operating contract

- This repository is the laboratory fork of lzzmm/GPUbk. Preserve upstream attribution and Apache-2.0 notices.
- Read README.md, docs/ROOTLESS.md and docs/ARCHITECTURE.md before changing deployment or admission.
- Use Python 3.12 and uv with the checked-in lock file for development.
- Keep one broker as the authority. Authenticate clients with kernel peer credentials; never accept a supplied UID as identity.
- Perform conflict, GPU-memory and host-budget checks inside the locked ledger transaction.
- Preserve existing reservations, workloads, private specs and user configuration.
- Install only Agent Skills. Do not modify user AGENTS.md, CLAUDE.md, shell files or MCP configuration automatically.
- Keep host policy in configuration, and private identities, traces, credentials and run artifacts outside public Git history.
- A reservation expiry is not proof that a physical GPU is free. Unknown occupancy must not be interpreted as absence.
- No-show reclamation requires trustworthy evidence and must preserve started work. Never signal another user's processes.
- Keep periodic device sampling out of protected timing windows. Prompt refresh must not query GPU drivers.
- Validate with synthetic devices and disposable Bash processes before requesting a real GPU acceptance run.
- Use English for comments and new documentation. Keep this file and CLAUDE.md identical.
- Run Ruff, relevant tests, the full regression suite and package checks before publishing changes.
