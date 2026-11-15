---
name: lint
description: Auto-fix FalkorDB's strict Clippy lints (pedantic/nursery/cargo, enforced by CI with -D warnings) and resolve what --fix cannot. Use when asked to fix clippy warnings or when CI's Lint job fails; for the plain fmt+clippy check use the build skill.
allowed-tools: Bash
---

# Lint (auto-fix)

CI fails on any clippy warning. The lint set (pedantic, nursery, cargo, minus
`missing_errors_doc` and `missing_panics_doc`) lives in `[workspace.lints]` in
`Cargo.toml` and `[lints]` in `native-deps/Cargo.toml`, so plain `cargo clippy`
shows exactly what CI checks. The `build` skill has the check commands.

## Auto-fix

```bash
CXX=clang++ cargo clippy --fix --allow-dirty -p falkordb-rs -p graph --all-targets
CXX=clang++ cargo clippy --fix --allow-dirty -p falkordb-rs -p graph --all-targets --features index-falkordb
cargo fmt --all
```

`--fix` rewrites source in place; review the diff. Run it for both feature
sets: code under `index-falkordb` is only seen by the second run.

## When --fix rolls back ("errors present after applying fixes")

One suggestion did not compile, so cargo reverted the whole crate. Rerun with
`--broken-code` to keep the fixes, then repair the few that broke. Known cases:

- `redundant_closure_for_method_calls`: `|r| r.string()` → `Reader::string`
  loses a closure that must work for any borrow lifetime. Keep the closure and
  allow the lint on the function.
- `missing_const_for_fn`: a fn made `const` in one feature set may call
  non-const code under the other (`#[cfg(feature = ...)]` bodies). Allow it.
- `wildcard_imports`: the expanded list differs per feature set. Keep `*`.

## What --fix leaves

Fix it by hand, or allow it with the reason: on the item for a one-off, or in
the crate's `lib.rs` for a style the crate does not follow. Changing a
signature (`&T` → `T`, `Vec` → slice, dropping a `Result` that cannot fail)
also means updating its callers.

In `native-deps`, do not edit `src/recipes/`, `src/toolchain.rs` or
`src/util.rs` for a lint: they feed the dependency cache key, so any edit
rebuilds GraphBLAS, LAGraph and RediSearch for everyone. Allow the lint on the
`mod` line in `src/lib.rs`.
