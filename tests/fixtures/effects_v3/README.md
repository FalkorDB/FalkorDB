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

## Cross-fixture invariants: file-by-file coverage is not total coverage

Three of this corpus's strongest claims are **relationships between two files**,
not properties of either one. A harness that iterates the corpus a file at a
time structurally cannot express them: each file decodes, re-encodes and
round-trips correctly in isolation whether or not the invariant holds. An
encoder can be internally consistent on every case and collectively wrong.

| pair | differs in | what it pins |
| --- | --- | --- |
| `dir_ascending` / `dir_descending` | 2 bytes | bit 6 carries direction |
| `collapse_above` / `dir_descending_bitmap` | 1 byte | a bitmap is a set, so the bit is the *only* carrier |
| `collapse_below` / `collapse_above` | one id | where the collapse rule changes its mind |

All three are asserted in `tests/unit/test_effects_v3_corpus.c` —
`directionPairs` and `collapsePair`. If you take a subset of this corpus, these
are the cases that stop meaning anything when their partner is dropped.

## The direction pairs

Two pairs in this corpus do something no self-consistency check can.

    dir_ascending          / dir_descending          differ in exactly 2 bytes
    collapse_above         / dir_descending_bitmap   differ in exactly 1 byte

An engine that ignores the header's `descending` bit reads a descending payload
as its ascending twin and returns the ids **reversed rather than refusing the
buffer**. A round trip still passes: the decoder re-encodes what it thinks it
read, and the bytes come back identical. Nothing an engine can check against
itself detects this. Only a same-shape pair does, and only because the
ascending control sits here beside it.

The `Range` pair moves two bytes: the header's bit 6, and the base, which is
the *lowest* id ascending (200) and the *highest* descending (207). The bitmap
pair moves **one** — a roaring blob is a set and has no direction, so the
66-byte blob is byte-identical and bit 6 is the only thing that says which way
the ids came. That makes it the strongest statement available about the bit.

Both are asserted in `tests/unit/test_effects_v3_corpus.c`, which also checks
that every case carries the code its name claims. That is a different failure
from the ones the manifest catches: a fixture called `value_width_8_bytes` that
quietly encoded as width code 0 hashes consistently and is listed, so it would
sit here looking like coverage while exercising the same path as every other
case.

## The collapse pair

`collapse_below` and `collapse_above` are the collapse rule's decision written
in bytes. The same ascending shape, **one id apart**, landing on opposite sides
of

    range_bytes >= 32  AND  5 + bitmap_bytes < range_bytes

18 ids stay as 18 `Range` segments; 19 collapse into a single `Ascending`
segment. Both payloads are **87 bytes** — the cost tie made visible, the pair
straddling the point where the arithmetic changes its mind.

That is why it pins more than its size suggests: matching both halves requires
the cost prediction, the collapse decision, the run tracking, the segmentation
and the roaring construction path all to agree with the other engine, against
bytes this engine did not produce. An encoder wrong by a few bytes in its cost
arithmetic produces two files that are each internally consistent and
collectively on the wrong side of the boundary.

## What the corpus still does not pin

Earlier revisions of this file recorded three gaps — no descending segments, no
value width codes 2 or 3, no non-zero count width. All three are now covered by
the eight `dir_*`, `value_width_*` and `count_width_*` cases. What is left:

- **Descending only ever appears under `DELETE_NODE`.** All four `dir_*` cases
  are that record, so the bit is unexercised in combination with any other
  record's blocks.
- **No case combines a non-zero count width with the collapse boundary.** The
  two `count_width_*` cases are single ranges well clear of it, so an encoder
  whose cost arithmetic mishandles a wide count would not be caught here.
- **Count width code 3 (eight bytes) is unreachable for an ENCODER, and must
  still be handled by a DECODER.** A record's id count is a `u32` and a
  segment's count is bounded by it, so four bytes is the widest a conforming
  encoder can ever need, and no fixture should be added for code 3. But a
  malformed or hostile payload can set those bits anyway, so a decoder that
  switched on only the reachable codes would turn this note into a parse hole
  the moment someone read it as permission. C is safe here for the right
  reason: `_ReadWidth` computes `EFFECTS_V3_WIDTH_BYTES(code)`, i.e. `1 <<
  code`, and reads that many bytes generically rather than enumerating the
  codes it expects. Same shape as the descending bit, where the danger was a
  decoder ignoring a bit it thought it did not need.

## Cases

33 cases.

| case | records | bytes |
| --- | --- | ---: |
| `collapse_above` | 1 | 87 |
| `collapse_below` | 1 | 87 |
| `count_width_2_bytes` | 1 | 20 |
| `count_width_4_bytes` | 1 | 22 |
| `dir_ascending` | 1 | 19 |
| `dir_descending` | 1 | 19 |
| `dir_descending_bitmap` | 1 | 87 |
| `dir_descending_to_zero` | 1 | 19 |
| `payload_multi_record` | 4 | 134 |
| `rec_add_attribute` | 1 | 21 |
| `rec_add_schema_edge` | 1 | 28 |
| `rec_add_schema_node` | 1 | 29 |
| `rec_create_constraint` | 1 | 53 |
| `rec_create_edge` | 1 | 100 |
| `rec_create_index` | 1 | 96 |
| `rec_create_node` | 1 | 92 |
| `rec_delete_edge` | 1 | 35 |
| `rec_delete_node` | 1 | 31 |
| `rec_drop_constraint` | 1 | 62 |
| `rec_drop_index` | 1 | 48 |
| `rec_remove_labels` | 1 | 23 |
| `rec_set_labels` | 1 | 27 |
| `rec_update_edge` | 1 | 86 |
| `rec_update_node` | 1 | 88 |
| `seg_ascending` | 1 | 209 |
| `seg_mixed` | 1 | 25 |
| `seg_range` | 1 | 19 |
| `seg_range_many` | 1 | 28 |
| `seg_range_single` | 1 | 19 |
| `seg_repeat` | 1 | 19 |
| `value_width_4_bytes` | 1 | 22 |
| `value_width_8_bytes` | 1 | 26 |
| `values_all_kinds` | 1 | 265 |
