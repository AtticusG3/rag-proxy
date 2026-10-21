# Vendored agent skills

These are [Matt Pocock's skills](https://github.com/mattpocock/skills) ("Skills For Real Engineers"), copied in as editable files, the same way `npx skills@latest add mattpocock/skills` installs them for Cursor, Codex and other agents that read `.agents/skills/`.

- **Source:** https://github.com/mattpocock/skills
- **Pinned at:** `5beb35b040e3de97e54fa9e386d7e42d15c295c2` (19 Oct 2026)
- **What's here:** exactly the promoted set the upstream Claude Code plugin ships (`skills/engineering/*` and `skills/productivity/*`), folders unchanged
- **Licence:** MIT, Copyright (c) 2026 Matt Pocock. See [`LICENSE-mattpocock-skills`](./LICENSE-mattpocock-skills)

The repo's own skills (`rag-proxy-change`, `rag-proxy-test`, `rag-proxy-debug`, `rag-proxy-deploy`) stay in `.cursor/skills/`.

## Per-repo config

`/setup-matt-pocock-skills` has already been done for this repo. Its output is:

- the `## Agent skills` block in `AGENTS.md`
- `docs/agents/issue-tracker.md` (GitHub, `AtticusG3/rag-proxy`)
- `docs/agents/triage-labels.md` (default labels)
- `docs/agents/domain.md` (single-context: `GLOSSARY.md` + `docs/adr/` at the root, created lazily)

Edit those files directly if things change. Re-run the setup skill only to switch trackers.

## Claude Code

Claude Code reads `.claude/skills/`, not `.agents/skills/`. Either install the plugin (`claude plugin install mattpocock-skills@claude-plugins-official`) or copy/symlink this folder. Don't do both, or you'll get every skill twice.

## Updating

```bash
npx skills@latest update      # if installed via the skills CLI
# or, by hand:
git clone https://github.com/mattpocock/skills /tmp/mp-skills
# copy the folders listed in /tmp/mp-skills/.claude-plugin/plugin.json over the ones here,
# then update the "Pinned at" commit above
```

Local edits to a vendored skill will be overwritten by an update, so keep repo-specific rules in `AGENTS.md` or `docs/agents/` instead.
