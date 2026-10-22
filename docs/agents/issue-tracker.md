# Issue tracker: Gitea

Issues, specs and PRs for this repo live on Kevyn's Gitea at **`git.kevynwatkins.com`**, repo **`default/rag-proxy`**. GitHub (`AtticusG3/rag-proxy`) is only a mirror. Its issues #23-#45 were copied here on 22 Oct 2026 (GitHub #n = Gitea #n-17) and the GitHub copies are frozen; don't use them.

There's no `gh` equivalent assumed here. Use the Gitea REST API (`/api/v1`) with a token in the environment, or the official [`tea`](https://gitea.com/gitea/tea) CLI if it's installed and logged in. The API calls below are the ones the setup was done with.

```bash
G=https://git.kevynwatkins.com/api/v1/repos/default/rag-proxy
H="Authorization: token $GITEA_TOKEN"   # never echo the token or put it in a file in the repo
```

The web UI sits behind Cloudflare and may refuse non-browser clients. The `/api/v1` paths are what to script against.

## Conventions

- **Create an issue**: `curl -sS -X POST -H "$H" -H 'Content-Type: application/json' "$G/issues" -d @issue.json`, where `issue.json` is `{"title": "...", "body": "...", "labels": [<label ids>]}`. Labels are passed by **id**, not by name. Get them with `curl -sS -H "$H" "$G/labels?limit=50"`.
- **Read an issue**: `curl -sS -H "$H" "$G/issues/<n>"`, and comments with `"$G/issues/<n>/comments"`.
- **List issues**: `curl -sS -H "$H" "$G/issues?state=open&type=issues&labels=<name>&limit=50"`. `type=issues` matters, because Gitea returns PRs from this endpoint too.
- **Comment**: `POST "$G/issues/<n>/comments"` with `{"body": "..."}`
- **Labels**: `POST "$G/issues/<n>/labels"` with `{"labels": [<ids>]}` to add; `DELETE "$G/issues/<n>/labels/<id>"` to remove.
- **Close**: `PATCH "$G/issues/<n>"` with `{"state": "closed"}`, after a comment saying why.
- **Edit a body**: `PATCH "$G/issues/<n>"` with `{"body": "..."}`

Issues and PRs share one number space on Gitea, so a bare `#42` may be either. Resolve it with `GET "$G/pulls/42"` and fall back to `GET "$G/issues/42"`.

### Parents and children (Gitea has no sub-issues)

- The child's body starts with `Parent: #<parent>`.
- The parent's body carries a checklist of its children (`- [ ] #<child> <title>`), ticked as they close.
- The parent is **blocked by** each child through Gitea's native issue dependencies, so Gitea won't let the parent close while a child is open: `POST "$G/issues/<parent>/dependencies"` with `{"owner": "default", "repo": "rag-proxy", "index": <child>}`.

### Blocking between tickets

Use the same dependencies endpoint: `POST "$G/issues/<blocked>/dependencies"` with `{"owner": "default", "repo": "rag-proxy", "index": <blocker>}`. Also write it in the ticket's `## Blocked by` section, so it's readable without the UI. List the blockers of an issue with `GET "$G/issues/<n>/dependencies"`. A ticket is unblocked when all of them are closed.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues: `GET "$G/pulls?state=open"`, `GET "$G/pulls/<n>.diff"`, and comment/label through the issue endpoints, since a PR is an issue on Gitea.

## When a skill says "publish to the issue tracker"

Create a Gitea issue on `default/rag-proxy` as above.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: one issue labelled `wayfinder:map` (create the label if missing), holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: an issue with `Parent: #<map>` at the top, listed in the map's checklist, and the map blocked by it (see **Parents and children**). Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`).
- **Blocking**: Gitea issue dependencies (see above), mirrored in a `Blocked by` line.
- **Frontier query**: the map's open children with no open blockers and no assignee; first in checklist order wins.
- **Claim**: `PATCH "$G/issues/<n>"` with `{"assignees": ["kevyn"]}` (or the agent's account), the session's first write.
- **Resolve**: comment the answer, close the issue, then add a context pointer (gist + link) to the map's Decisions-so-far.

## Labels used in this repo

On top of the triage labels in `triage-labels.md`:

| Label | Use |
| --- | --- |
| `prd` | A spec / PRD issue; its tickets are its children. Current PRD: #6 |
| `bug`, `enhancement`, `documentation` | Category (at most one) |
| `review-finding` | Came out of a code review. Body has file/line permalinks pinned to a commit, the evidence, and a suggested fix. Review tracking issue: #12 |
| `severity:high` / `severity:medium` / `severity:low` | Impact if it bites (review findings) |
| `priority:p1` / `priority:p2` / `priority:p3` | Order to work in: do next / soon / when there's time |

## Ticket shape

Tickets follow `/to-tickets`: `Parent: #<n>` at the top, `## What to build` (end-to-end behaviour, not a layer list), `## Acceptance criteria` checkboxes, and `## Blocked by`.

## Branches and PRs

- Branch from up-to-date `main` on Gitea. Never push straight to `main`.
- Open the PR on Gitea against `main` (`POST "$G/pulls"` with `{"head": "<branch>", "base": "main", "title": "...", "body": "..."}`) and put `Closes #<n>` in the body for the ticket(s) it finishes.
- Don't merge your own PR; Kevyn merges.
