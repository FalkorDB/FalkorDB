---
name: lint
description: Auto-fix FalkorDB's strict Clippy lints (pedantic/nursery/cargo, enforced by CI with -D warnings) and resolve what --fix cannot. Use when asked to fix clippy warnings or when CI's Lint job fails; for the plain fmt+clippy check use the build skill.
allowed-tools: Bash
---

# Lint (auto-fix)

CI fails on any clippy warning. The lint set (pedantic, nursery, cargo, minus
`missing_errors_doc`, `missing_panics_doc` and `doc_markdown`) lives in
`[workspace.lints]` in `Cargo.toml` and `[lints]` in `native-deps/Cargo.toml`,
so plain `cargo clippy` shows exactly what CI checks. The `build` skill has
the check commands.

## Auto-fix

```bash
CXX=clang++ cargo clippy --fix --allow-dirty --workspace --all-targets
CXX=clang++ cargo clippy --fix --allow-dirty -p falkordb-rs -p graph --all-targets --features index-falkordb
cargo fmt --all
```

`--fix` rewrites source in place; review the diff (its `map_or` rewrites can
delete comments inside match arms). Run it for both feature sets: code under
`index-falkordb` is only seen by the second run.

## When --fix rolls back ("errors present after applying fixes")

One suggestion did not compile, so cargo reverted the whole crate. Rerun with
`--broken-code` to keep the fixes, then repair the few that broke. Known cases:

- `redundant_closure_for_method_calls`: `|r| r.string()` → `Reader::string`
  loses a closure that must work for any borrow lifetime. Keep the closure.
- `missing_const_for_fn`: a fn made `const` in one feature set may call
  non-const code under the other (`#[cfg(feature = ...)]` bodies).
- `wildcard_imports`: the expanded list differs per feature set. Keep `*`.
- `redundant_clone` on a `Range` drops the clone but not the need for `mut`.

## What --fix leaves

Fix it by hand, or put a scoped `#[allow(clippy::<lint>, reason = "...")]` on
the narrowest item it covers (a bench/test module at most), never crate-wide.
Changing a signature (`&T` → `T`, `Vec` → slice, dropping a `Result` that
cannot fail) also means updating its callers.

In `native-deps`, do not edit `src/recipes/`, `src/toolchain.rs` or
`src/util.rs` for a lint: they feed the dependency cache key, so any edit
rebuilds GraphBLAS, LAGraph and RediSearch for everyone. Allow the lint on the
`mod` line in `src/lib.rs`.
