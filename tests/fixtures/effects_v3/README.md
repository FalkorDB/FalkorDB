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

## Where the guarantee ends: no query-derived payloads

Every case here is built from `Record` values, not by running a query. That is
a deliberate boundary, not a stage the corpus has yet to reach.

C filters redundant label operations and Rust does not — sending a fact already
set is idempotent, so C optimises and Rust is more verbose. That was ruled
**implementation, not divergence**: the two engines legitimately emit different
bytes for the same query, and that is sanctioned rather than tolerated.

So **no test may assert cross-engine byte equality for a query-derived
payload.** A query-level corpus would be asserting something the format does
not claim. What the format promises is that a given *record* encodes to given
bytes; which records a given query produces is each engine's business.

That is why this directory contains no "run this Cypher, compare the wire"
cases, and why adding them would be a mistake rather than an improvement. The
record level is not the corpus's first phase — it is the whole of what the
corpus can be.

## What completing the grid actually found

Filling in a coverage grid reads like housekeeping. This one produced a live
cross-engine bug that nobody was looking for, and the chain is worth stating in
full because every link looks like tidying on its own:

1. The grid measured 26 blocks with a present case and no empty one.
2. That produced a request to classify all 26 — is each empty form legal or
   refusable?
3. Answering it made the Rust side explain their *emitter* to justify one entry,
   `SET_LABELS.labels`: `digest_labels` has no empty guard and
   `remove_node_labels` empties the label vector while leaving the key, so
   `MATCH (n) SET n:Foo REMOVE n:Foo` emits a zero-label `SET_LABELS`.
4. Reading the C side against that: decode accepts it, and **apply refuses it**
   — `effects_v3_apply.c`, "carries no labels". A refused effect is divergence,
   and the same buffer is refused on every retry, so it is a forced-resync loop
   against a current master. C's encoder was then found to emit the same record
   deliberately, so it is *both* engines emitting a record one of them refuses.

None of that was visible from any single fixture, any single engine, or any
green test run. It came out of asking what the corpus does *not* contain and
insisting on a reason for each answer.

## The rule: every optional block needs a present case AND an absent case

The 33 cases read as one per feature, which is why a missing *absent*-block
case was never obviously missing — it was not a hole in a grid, it was a case
nobody thought of. It cost a real bug: the C decoder returned MALFORMED for a
create with zero attributes, refusing `CREATE (:Person)`, while a full green
corpus run and 26 green flow tests said nothing (fixed in `377f82a85`). No
fixture here has an empty attribute set, so nothing contradicted it.

So the rule, stated so the holes are visible as gaps in a grid:

> Every block whose cardinality can be zero needs a case where it is present
> and a case where it is empty. If the empty form is *invalid*, the absent case
> is a rejection case rather than a fixture.

That second clause matters, because "empty" is not one thing. There are three
kinds, and they need three different artifacts:

- **Legal and degenerate** — the empty form is a payload the engine really
  emits and a decoder must accept. `CREATE (:Person)` has no attributes;
  `CREATE ({x: 1})` has no labels. These need real generated fixtures.
- **Invalid** — the empty form states nothing and must be refused. These need
  no fixture: they are hand-built mutations in the rejection cases, like the
  reserved-bit and descending-on-Repeat cases already there.
- **Mandated empty** — the empty form is the *only* legal one, and it is the
  PRESENT form that must be refused. `DROP_INDEX` carries an empty field list
  and no options (`docs/effects-v3.md:547`); `DROP_CONSTRAINT` omits the
  status the create carries. So the two blocks this grid shows as "empty with
  no present counterpart" are not gaps — they are correct, and what they need
  is a rejection case for the present form.

**The spec is silent on block cardinality, and that is the root cause of all
26.** Searching `docs/effects-v3.md` for `empty` or `zero` returns nine hits:
the reserved header bit, the descending base underflow, id widths, compression,
a test note. Not one is a rule about whether a block may be empty. The corpus
gap and the spec gap are the same gap, so filling in the corpus without filling
in the spec would leave the next person deriving these classifications again.

### Do not let one engine assert this alone

Rust's block decoders accept `n = 0` unconditionally — `LabelSet::decode` and
`AttrIds::decode` both `take_n(n)` with no zero check, and
`AttrValues::decode_sized` computes `count * attrs_per_row`, which is zero and
loops zero times (`graph/src/effects/v3/blocks.rs`). There is no per-block zero
rejection anywhere. So every entry in the "invalid" list is a property **C would
be asserting alone** until Rust agrees to it, and a decoder that refuses what
its peer emits is the divergence this whole format exists to prevent. The
mutations are therefore not built until the classification is confirmed on both
sides — deliberately, not pending effort.

One inference that looked safe and was wrong, recorded because it is the shape
of the mistake: *a record carrying zero ids* is not refusable by analogy with
the segment-level checks. `Segment::decode` does reject `count == 0` and
`len == 0`, but `read_ids` decodes `n_segments` segments and then only asserts
`len == count` — so `n_segments = 0` with `count = 0` never calls
`Segment::decode` at all and returns an empty list. **Rust accepts a zero-id
record today.** Verified against `graph/src/effects/v3/id_list.rs`. Segment-level
and record-level are different checks and only the first exists.

Measured across all 33 cases, only ONE block has both forms today:

| block | present | empty |
| --- | ---: | ---: |
| `DELETE_NODE.labels` | 2 | 16 |

Every other optional block appears in one form only — 26 of them present with
no empty counterpart, and two (`DROP_CONSTRAINT.status`, `DROP_INDEX.options`)
empty with no present counterpart. The empty-attribute-set gap was one of those
26, not a one-off.

**Enforcement belongs in the generator, not here.** Checking this grid means
reading each record's blocks as structures, and the C harness deliberately
reads bytes and no JSON — the `.json` files are documentation with no
verification power. The side that holds the records in hand is the generator,
which can assert the grid is complete as it writes the corpus.

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
- **A zero-count record is illegal.** Ruled after
  the grid work found both accepting it while both rejected the same idea one
  level down at the segment. Rust has landed it as an `EmptyRecord` error and
  C's check goes at the record header where `count` is read; the rejection case
  is in `tests/unit/test_effects_v3_roundtrip.c`, gated until C's lands. So:
  **ruled on both engines, implemented on Rust, pending on C** — not yet an
  invariant either engine can rely on the other to hold.
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
