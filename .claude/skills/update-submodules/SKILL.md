---
name: update-submodules
description: Update a vendored fork (torchtitan, torchft, or any future fork) to a newer upstream version while preserving Panocular's custom changes that live on top of upstream. Use this whenever the user wants to "update torchtitan/torchft", "update the forks", "update the submodules", "pull in upstream changes", "rebase our fork", "bump a fork", "sync the fork with upstream", or resolve conflicts between our changes and new upstream code. Trigger even if the user only names one fork.
---

# Update a fork from upstream

## Mental model — read this first

This repo (`panofabric-engine`) depends on **Panocular forks** (`PanocularAI/torchtitan`,
`PanocularAI/torchft`) rather than the upstream projects directly.

**These are NOT submodules any more.** They are ordinary **sibling clones** next to this
repo (`../torchtitan`, `../torchft`), created by `make forks` / `make dev-forks`, and the
engine pins them **by SHA** in two places that must stay in sync:

* `pyproject.toml` — the `[train]` extra's `torchtitan @ git+...@<sha>` / `torchft @ ...@<sha>`
* `Makefile` — `TORCHTITAN_REF` / `TORCHFT_REF`

A third pin lives in the private control-plane repo: `panofabric/Makefile`'s
`TORCHTITAN_REQ` / `TORCHFT_REQ` (needed because `uv pip` accepts git URLs only from the
command line, never from a package's metadata). Update it too, or `make install-engine`
drifts from the engine.

Each fork's `main` is structured the same way:

```
  our custom commits           ← Panocular's changes, replayed on top   ┐
  ...                                                                    │  "the stack"
  our custom commits                                                     ┘
  <upstream snapshot HEAD>     ← a commit from the upstream project
  ...upstream history...
```

So our changes are a **small stack of commits replayed on top of an upstream snapshot**.
**Updating = rebase that stack onto a newer upstream commit, resolve conflicts, push the fork,
then update the SHA pins.** That's the whole job for *every* fork — only the remote URLs, the
files our stack touches, and the verification commands differ. Everything below is the
careful, generic version of those five moves.

**Never hardcode the stack's SHAs or commit count** — always rediscover it dynamically per
fork (Step 2), because the stack grows over time and differs between forks.

## Submodule registry

Run `cat .gitmodules` from the repo root to see the current set. Known mappings:

| Submodule path | Fork (origin) | Upstream remote to add |
|---|---|---|
| `torchtitan` | `git@github.com:PanocularAI/torchtitan.git` | `https://github.com/pytorch/torchtitan.git` |
| `torchft` | `git@github.com:PanocularAI/torchft.git` | `https://github.com/pytorch/torchft.git` |

If the user says "update the submodules" (still the common phrasing) without naming one, ask
which — or do both — and run the steps below **once per fork**.

## Guardrails

- **Pushing the fork and force-pushing rewrite shared history** — these are outward-facing.
  Confirm before pushing, and strongly prefer a **new branch + PR** over force-pushing `main`
  (Step 6). Never `git push --force`; only `--force-with-lease`.
- Work on a throwaway branch so a botched rebase is `git rebase --abort` + delete-branch, never a
  corrupted `main`.
- Don't run full training/long jobs to "verify" — verify with builds/imports/tests (Step 5).
- Keep the user in the loop at the decision points: target upstream ref, each conflict
  resolution, whether any of our commits are now obsolete, and how to publish.

---

The steps below use `$SM` for the fork's directory name (e.g. `torchtitan`) and reach it as
`../$SM` from this repo. Set it once per fork and repeat the whole sequence.

## Step 0 — Orient and check preconditions

```bash
cd "../$SM"                       # the sibling clone
git status                        # MUST be clean; if dirty, stop and ask the user
git remote -v                     # expect: origin -> PanocularAI/<name>
```

Each fork also carries a `FORK-DELTA.md` listing exactly which files our stack modifies —
read it first, because that list *is* the conflict surface you are about to hit.

Ensure the upstream remote exists (URL from the registry above), then fetch everything:

```bash
git remote get-url upstream 2>/dev/null \
  || git remote add upstream <UPSTREAM_URL_FROM_REGISTRY>
git fetch origin
git fetch upstream
```

Note the SHA the engine currently pins — it may lag `origin/main`:

```bash
git rev-parse HEAD                                   # this clone's checkout
git rev-parse origin/main                            # tip of the fork
grep -E 'TORCHTITAN_REF|TORCHFT_REF' ../panofabric-engine/Makefile   # what we pin
```

## Step 1 — Decide the target upstream ref

Default to the latest upstream `main` (`upstream/main`). If the user wants stability, offer a
release tag instead (`git tag -l | sort -V | tail`). Show how much is incoming before committing
to the work:

```bash
OLD_BASE=$(git merge-base origin/main upstream/main)
git log --oneline "$OLD_BASE"..upstream/main | wc -l      # how many upstream commits we'd pull in
git log --oneline "$OLD_BASE"..upstream/main | head -40   # a taste of what's new
```

Pin the chosen ref for the rest of the run, e.g. `TARGET=upstream/main` (or a tag / exact SHA).
Prefer recording the resolved SHA so the result is reproducible.

## Step 2 — Identify OUR stack (the commits to replay)

These are the commits on the fork that aren't in upstream — this is the **authoritative** way to
find the stack:

```bash
git log --oneline upstream/main..origin/main
git log --format='%h %an <%ae> %s' upstream/main..origin/main   # inspect authors
```

Sanity-check the list looks like our changes (small, focused on the dirs we customize). Author
email is only a weak hint — it varies (some commits are `@panocular.ai`, others a personal/GitHub
noreply address), so **don't filter by email**; trust the `upstream/main..origin/main` range. If
the range contains commits that clearly aren't ours, or it's surprisingly large/non-linear, stop
and investigate — the fork may have merged upstream in a way that needs a different `--onto` base
or a merge instead of a rebase. Show the user the list and confirm it's the intended set.

## Step 3 — Rebase the stack onto the target

Work on a dated branch built from the current fork tip, then replay only our commits onto the new
upstream base. Using `--onto <target> <old-base>` guarantees we move exactly the stack from
Step 2 and nothing else:

```bash
git checkout -b "${SM}-update-$(date +%Y%m%d)" origin/main
git rebase --onto "$TARGET" "$OLD_BASE"
```

Clean rebase → skip to Step 5. Otherwise → Step 4.

## Step 4 — Resolve conflicts (the part that needs judgment)

The goal is to **preserve the intent of our change while adapting it to upstream's new
API/structure** — not to blindly keep "ours" or "theirs".

For each conflict:
1. `git status` to see conflicted files. (Our stacks tend to touch a small, stable set —
   e.g. torchtitan's `torchtitan/experiments/ft/` and training entry; torchft's reconfiguration
   logic. Confirm by reading the stack's own diffs from Step 2.)
2. Read the upstream side to understand what changed (renamed functions, moved modules, changed
   signatures, config-registry refactors are common upstream).
3. Read our side to understand the behavior we're adding.
4. Reconcile: reapply our behavior on top of upstream's new shape. If upstream **moved/renamed**
   the thing we patched, follow it there. If a file we touched was **deleted upstream**, find
   where the logic went.
5. `git add <files>` then `git rebase --continue`.

Watch for our change becoming **obsolete** — upstream may have landed equivalent functionality.
If a commit is now redundant, raise it with the user; dropping it via `git rebase --skip` (or
editing it down) keeps our stack minimal. **Note anything dropped or materially reworked in the
final summary.**

If it gets messy, `git rebase --abort` returns you to a clean `origin/main` to retry.

## Step 5 — Verify without a full run

Confirm the tree is coherent — this catches the most common rebase breakage (stale imports,
moved symbols). Verification is fork-specific:

- **Pure-Python fork (e.g. torchtitan):** from this repo's root, using the project venv:
  ```bash
  python -c "import torchtitan; print(torchtitan.__file__)"
  python -c "import torchtitan.experiments.torchft.manager, torchtitan.experiments.torchft.checkpoint"
  python -m pytest tests -q                 # the recipe smoke test — see below
  ```
  **Run `tests/test_config_registry.py`.** It constructs every preset in every recipe and is
  the cheapest possible detector of exactly the drift a rebase causes (a moved symbol, a
  renamed kwarg). It is why this suite exists.
  Also eyeball that `run_train.sh`'s entry points still exist (`MODULE`, `TRAIN_FILE`,
  `CONFIG_NAME`), since upstream refactors config registries periodically. Do NOT launch
  `run_train.sh`.

- **Fork with a native/Rust extension (e.g. torchft):** importing requires a build. Rebuild
  the extension before the import check (it ships a Rust core via maturin, needing Rust,
  `protoc` 32.0 and CPython <= 3.13):
  ```bash
  pip install -e ../torchft       # or: (cd ../torchft && maturin develop)
  python -c "import torchft; print(torchft.__file__)"
  (cd ../torchft && cargo build 2>/dev/null && python -m pytest tests -q)
  ```
  If a full rebuild isn't feasible in this environment, say so explicitly rather than claiming the
  import passed.

Report exactly what you ran and what passed/failed — don't claim verification you didn't do.

## Step 6 — Publish the fork (confirm first)

Show the user `git log --oneline "$TARGET"..HEAD` (our replayed stack on the new base) and the
diff range, then ask how to publish. Preferred, safest path — push a branch and open a PR:

```bash
git push origin HEAD
gh pr create --repo PanocularAI/$SM --base main --head "${SM}-update-$(date +%Y%m%d)" \
  --title "Rebase onto upstream $SM <date/ref>" \
  --body "Replays our stack onto <TARGET>. Conflicts resolved in <files>. <obsolete commits noted>."
```

Only if the user explicitly wants `main` moved directly (understanding it rewrites shared history)
update it with a lease guard:

```bash
git push --force-with-lease origin "${SM}-update-$(date +%Y%m%d):main"
```

## Step 7 — Update the SHA pins

There is no submodule pointer to move. Update the pins instead — **all three**, or the engine,
its Makefile and the control plane drift apart:

```bash
NEW=$(git -C "../$SM" rev-parse origin/main)   # or the branch tip you pushed

# 1 + 2. The engine: pyproject.toml [train] extra, and the Makefile ref.
#        Both live in this repo; grep the old sha and replace it in both.
grep -rn "<old-sha>" pyproject.toml Makefile

# 3. The private control plane (TORCHTITAN_REQ / TORCHFT_REQ).
grep -n "$SM" ../panofabric/Makefile
```

Then confirm nothing still names a dead ref — a deleted branch or a stale sha is exactly how
`make install-engine` broke before:

```bash
git ls-remote https://github.com/PanocularAI/$SM.git | grep "$NEW"   # MUST match
```

Commit the pin bumps. Don't push this repo or open its PR unless asked.

End with a summary per fork: old vs new upstream base, the stack after rebase, conflicts
resolved, anything dropped as obsolete, which pins you moved, and what was verified.

## Quick reference

| Action | Command |
|---|---|
| Find our stack | `git log upstream/main..origin/main` |
| Old base | `git merge-base origin/main upstream/main` |
| Replay stack | `git rebase --onto <target> $(git merge-base origin/main upstream/main)` |
| Abort | `git rebase --abort` |
| Publish | branch + `gh pr create` (preferred) or `git push --force-with-lease origin <branch>:main` |
| Bump pins | `pyproject.toml` `[train]` + `Makefile` refs here, **and** `panofabric/Makefile`'s `*_REQ` |
| Check a pin is real | `git ls-remote https://github.com/PanocularAI/$SM.git \| grep <sha>` |
