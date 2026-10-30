# feral-ordering-core

Shared input/output types for FERAL's fill-reducing ordering crates.

This crate defines the locked contract surface that every sibling
ordering crate in FERAL (`feral-amd`, `feral-metis`, `feral-scotch`,
`feral-kahip`) implements:

- `CscPattern<'_>` — borrowed, full-symmetric, 0-based, `i32`-indexed
  sparsity pattern.
- `OrderingStats` — producer-agnostic diagnostic counters (time,
  optional fill / flop estimates).
- `OrderingError` — shared error shape with a static `Internal(&str)`
  escape hatch.
- `CONTRACT_VERSION: u32` — bumped on breaking changes.

Zero dependencies beyond `std`. The full design rationale lives in
`dev/plans/ordering-crate-contract.md`.

## Per-crate contract function

Each ordering crate exposes exactly one contract-conforming function:

```rust,ignore
pub fn xxx_order(
    pattern: &feral_ordering_core::CscPattern<'_>,
    opts: &XxxOptions,
) -> Result<
    (Vec<i32>, feral_ordering_core::OrderingStats, XxxStats),
    feral_ordering_core::OrderingError,
>;
```

`perm[k] = j` means new index `k` corresponds to old index `j`
(new-to-old). Convenience overloads (e.g. defaults, skipping stats)
are allowed but the three-tuple signature above is mandatory.

## License

MIT.
