---
name: lint
description: Run FalkorDB's stricter opt-in Clippy pass (pedantic/nursery/cargo lints) beyond the basic CI gate, optionally auto-fixing. Use when asked for a deep/strict lint pass or to auto-fix clippy warnings; for the standard fmt+clippy CI gate use the build skill.
allowed-tools: Bash
---

# Lint (advanced)

The `build` skill runs the CI gate (`cargo fmt --all -- --check` +
`CXX=clang++ cargo clippy --all-targets`). This skill is the stricter,
opt-in pass: pedantic + nursery + cargo lints, and an optional auto-fix.

## Deep clippy pass

```bash
CXX=clang++ cargo clippy --all-targets -- \
  -W clippy::pedantic -W clippy::nursery -W clippy::cargo \
  -A clippy::missing-errors-doc -A clippy::missing-panics-doc
```

## Auto-fix where possible

```bash
CXX=clang++ cargo clippy --all-targets --fix --allow-dirty --allow-staged -- \
  -W clippy::pedantic -W clippy::nursery -W clippy::cargo \
  -A clippy::missing-errors-doc -A clippy::missing-panics-doc
```
`--fix` rewrites your source in place — review the diff afterward.

## Notes

- `CXX=clang++` is required for clippy (GraphBLAS FFI), same reason as the
  `build` skill.
- `missing-errors-doc` / `missing-panics-doc` are deliberately allowed
  (`-A`) to avoid noise on every fallible/panicking function.
- These settings are stricter than CI's gate and `nursery` lints are
  experimental (occasional false positives) — treat this as a cleanup aid,
  not a merge blocker.

If any lints fail, report the errors clearly and help fix them.
