# Effects v3 conformance fixtures

Generated. Do not hand-edit — see `graph/src/effects/v3/fixtures.rs`, which
regenerates these and fails the build if they drift:

```sh
UPDATE_EFFECTS_FIXTURES=1 cargo test -p graph effects::v3::fixtures
```

Each case is a pair. `<case>.hex` is a complete `GRAPH.EFFECT` payload —
version byte, flags byte, then records — as hex, 32 bytes a line, with `#`
comments and whitespace to be stripped by the reader. `<case>.json` says what
those bytes decode to.

## What these are for

Two engines have to agree on this format, and each round-tripping against
itself proves only that it is self-consistent. An engine can read every field
one width narrow and pass all of its own tests; that is the failure that
segfaulted C in `AttributeSet_Update` when it was handed a Rust buffer. These
files are the shared statement neither side can quietly move.

The minimal test on the far side needs no JSON parser: read the `.hex`, decode
it, re-encode it, compare. A decoder that reads a field wrong writes it back
wrong, so byte equality after a round trip catches width, order and truncation
errors on its own. Rust runs that same test, on these same files rather than
on bytes regenerated beside them.

## They are not ground truth

They are generated from the Rust encoder. If it is wrong about C’s format,
they enshrine the error faithfully. The ground truth is C’s own source — the
widths, the field order and the tag numbering in `effects.c` and `effects.h`,
which the encoder cites field by field.

So a disagreement between C and a fixture is not settled by the fixture. It is
settled against C’s format definition, and the fixture regenerated if Rust was
the one that was wrong. What these files are for is narrower: freezing what has
been agreed, so neither side moves it by accident.

## What the `.json` is, and is not

Documentation. It is generated from the same records the `.hex` is, so nothing
cross-checks one against the other and it adds no verification power.

It earns its place at review time. When a change moves the wire, a `.hex` diff
is unreadable and the `.json` diff says what actually changed — which record,
which field, which value. It also lets someone writing assertions on the far
side see what a case contains without decoding it by hand.

## Two things the corpus is deliberately pinning

- **`schema_type` is 0-based; `entity_type` is 1-based.** C's
`GraphEntityType` reserves 0 for UNKNOWN, so the schema records and the
constraint records number node-vs-edge differently. Both appear here.
- **`T_INTERN | T_STRING` and `T_STRING` are the same value.** The intern bit
is a hint about the primary's string pool. `values_all_kinds` carries one of
each, so a decoder that switches on the whole tag word instead of masking
fails on exactly one of them.

Payloads are uncompressed. Compression is an encoder choice — a level, and a
zstd version — and pinning one here would bind the far side to a compressor
rather than to a format.

## A known gap: no descending segments

This corpus was cut before the segment header gained a direction bit, so it
does not exercise it at all. As of `08dca4a1e` on `feat/effects-v3`
(`graph/src/effects/v3/id_list.rs:518-519`):

- bit 6 is `descending`, **not** reserved. A descending `Range` reads its base
  as the first and *highest* id; an `Ascending` blob is a set and has no
  direction, so the two directions of the same ids differ in exactly that bit.
- bit 7 alone is reserved and must be rejected.
- `Repeat` has no direction, so the bit is rejected there rather than ignored
  (`id_list.rs:721`).

So there are no `Range` or `Ascending` fixtures with bit 6 set, and the
accepting side of that rule is unpinned by these files. The rejecting side
needs no fixture and is covered in `tests/unit/test_effects_v3_roundtrip.c`,
which mutates `seg_repeat`'s header. A regeneration should add an ascending and
a descending form of the same ids, since that pair is what makes the bit's
meaning checkable rather than merely present.

## Cases

| case | records | bytes |
| --- | --- | ---: |
| `seg_range` | 1 | 19 |
| `seg_range_single` | 1 | 19 |
| `seg_range_many` | 1 | 28 |
| `seg_repeat` | 1 | 19 |
| `seg_ascending` | 1 | 209 |
| `collapse_below` | 1 | 87 |
| `collapse_above` | 1 | 87 |
| `seg_mixed` | 1 | 25 |
| `rec_create_node` | 1 | 92 |
| `rec_create_edge` | 1 | 100 |
| `rec_update_node` | 1 | 88 |
| `rec_update_edge` | 1 | 86 |
| `rec_delete_node` | 1 | 31 |
| `rec_delete_edge` | 1 | 35 |
| `rec_set_labels` | 1 | 27 |
| `rec_remove_labels` | 1 | 23 |
| `rec_add_schema_node` | 1 | 29 |
| `rec_add_schema_edge` | 1 | 28 |
| `rec_add_attribute` | 1 | 21 |
| `rec_create_index` | 1 | 96 |
| `rec_drop_index` | 1 | 48 |
| `rec_create_constraint` | 1 | 53 |
| `rec_drop_constraint` | 1 | 62 |
| `values_all_kinds` | 1 | 265 |
| `payload_multi_record` | 4 | 134 |
