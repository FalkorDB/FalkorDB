# Effects wire format v3

Status: **proposed**, for discussion. Implemented behind `EFFECTS_VERSION = 3` in
the Rust engine; needs a matching C implementation before it is usable.

## Why a new version

The goal is **engine migration**: a deployment running the C engine must be able to
upgrade to Rust, and downgrade back, without dumping and reloading. RDB
loadability landed in #2459, so a Rust replica can now full-sync from a C master.
Effects are the remaining gap: after the sync, the master keeps streaming
`GRAPH.EFFECT`, and the two engines cannot read each other's buffers.

Cloning C's v2 byte-for-byte would work against shipped C, but it means adopting
the two v2 decisions that are actively hostile to a second implementation:

1. **`CREATE_NODE` carries no entity id** (`CREATE_EDGE` no edge id). The replica
   derives ids from creation order. Any disagreement between the two engines about
   what "creation order" means corrupts a replica silently, and the record itself
   carries nothing to check against.
2. **`SET_LABELS` / `REMOVE_LABELS` carry a `GxB_Vector_serialize` blob** — a
   GraphBLAS-internal serialization embedded in the replication stream. The two
   engines build against separate GraphBLAS copies (Rust pins v10.5.0 in
   `graphblas.sh`; C vendors `deps/GraphBLAS`), so the wire format inherits a
   dependency on those staying compatible. That is a poor thing to couple a
   migration path to.

Versioning is C's own established mechanism, not a new idea: C's reader accepts
any version **≤** its own (`if (*v > EFFECTS_VERSION) return false`, `effects_apply.c:797`)
and already branches per version — `if (unlikely(version == 1)) ApplyLabels(...) else ApplyLabels_V2(...)`
(`effects_apply.c:865,873`). Notably, the v1→v2 bump existed to change *the label
records* — the same record family this proposal changes again.

## The invariant

    GRAPH.EFFECT <graph-key> <payload>      one command per query, never one per record

      payload   u8 version = 3
                record · record · record …  one per (opcode, shape), in apply order

      record    u32 opcode · u32 count · <the shape, once> · <count rows>

| record | **partition key** — the `GROUP BY` | per row |
| --- | --- | --- |
| `CREATE_NODE` | `LabelSet` + `AttrSet` ids | id, values |
| `CREATE_EDGE` | `RelType` + `AttrSet` ids | id, src, dst, values |
| `UPDATE_NODE` / `UPDATE_EDGE` | `AttrSet` ids only | id, values |
| `SET_LABELS` / `REMOVE_LABELS` | `LabelSet` | id |
| `DELETE_NODE` | — nothing to hoist | id |
| `DELETE_EDGE` | `RelType` | id, src, dst |
| `ADD_SCHEMA`, `ADD_ATTRIBUTE`, index, constraint | inherently singular — no partition key, no count | — |

Four rules, and everything else follows from them:

1. **One `GRAPH.EFFECT` per query.** Batching removes per-record framing; it never
   multiplies commands. There is exactly one `ctx.replicate("GRAPH.EFFECT", args)`
   (`src/graph_core.rs:1582`), fed by one `build_effects_buffer` call at commit
   (`graph/src/runtime/ops/commit.rs:111`).
2. **One record per `(opcode, partition key)`.** Shapes are compared only within a record
   type, so a query that creates *and* updates *and* deletes groups each family
   independently.
3. **Inside a record**: the count once, the ids once, the shape once, then
   `count × n` values.
4. **Row *k* belongs to the k-th id in the `IdList`, as written.** Id order is
   the only thing binding values to entities — no per-row id is sent. This is why
   An `Ascending` segment's decode must check its cardinality: a bitmap one id short
   would land every subsequent row on the wrong entity.

"Partition key" and "shape" are the same thing — the first is its role, and the
rest of this document uses *shape* as the short name. Rows sharing it become one
record, exactly as a `GROUP BY` would. It is deliberately not called a *sharding*
key: nothing is distributed across nodes, and all of a query's records still travel
in one command.

Shape is the attribute **ids**, never the values, and for edges it is the
relationship type exactly as labels are for nodes.

### We hoist, but deliberately do not reference

MySQL's `TABLE_MAP_EVENT` and Postgres's `Relation` message declare a shape in one
event and point at it by number from later events. That requires the decoder to
carry state between events, so a buffer is only decodable in the context of those
before it. v3 states the shape inline in the record that uses it, keeping every
record self-contained — which is what keeps AOF replay from an arbitrary offset,
and a replica that missed a buffer, both well-defined.

It also costs nothing to skip. The `AttrSet` header is `2 + 2n` bytes per record:
8 B out of 388,944 (0.002%) for a batched 3-attribute create. A declare-once table
referenced by a `u16` would be a strict loss rather than a trade-off, because
grouping makes **shapes == records by construction** — there is never a second
record to amortise a declaration over, so the reference bytes are pure addition
(1 shape: 8 → 10 B; 20 shapes: 80 → 120 B). The only regime where the header is
proportionally large is one shape per node (10.3%), and there every shape is
distinct, so a table cannot help at all.

## The payload header

    GRAPH.EFFECT <graph-key> <payload>

      u8  version = 3
      u8  flags                      bit 0 = compressed
      u32 uncompressed_length        only when compressed
      records… | zstd frame

The header is never compressed, so a reader always knows what it holds before
committing to decode anything.

**The flags byte is reserved whether or not compression is on.** Adding a byte
to the header later would cost another version bump for something compression
needs, which is the whole reason it is here from the start rather than when the
feature is first switched on.

A reader that meets a flag bit outside the mask it understands **rejects the
buffer**. An old node meeting a future payload must fail loudly; decoding the
records anyway would apply a prefix of something whose shape it does not know.
The declared uncompressed length is checked against what the frame actually
expands to, for the same reason: a mismatch means header and payload disagree.

### Compression is default off

`GRAPH.CONFIG SET EFFECTS_COMPRESSION <bytes>` — the smallest payload worth
compressing; **0, the default, disables it**. zstd level 1.

Off by default because compression is a bandwidth trade, not a CPU one: measured
at 3.245 cycles/byte to compress against 0.067 to copy into the replica output
buffer, so on a fast link it spends far more than the bytes are worth. Worth
turning on when the replication link, rather than the write thread, is the
constraint.

Even when enabled the smaller form wins — zstd inflates an already-minimal
buffer, and batched records are already most of the way there. On the motivating
query the three sizes are **340,001 B (v2) → 120,028 B (v3) → 10,446 B
(v3 + zstd-1)**.

Compression runs last, after the record stream is complete, because it rewrites
everything after the header.

#### Level 1's ratio is a coin flip on alignment

Worth knowing before quoting a ratio from this document. zstd level 1's fast
match-finder phase-locks onto a payload's record period, so the ratio depends on
where the records happen to start. Two payloads that differ by a few bytes of
prefix and are otherwise the same ten thousand value rows compress to **10,446**
and **31,136** bytes. Padding the same body one byte at a time shows the shape
of it — one alignment in nine is ~3x better than the other eight:

| pad | level 1 | level 3 |
| --- | --- | --- |
| 0 | 31,098 | 27,253 |
| 1 | **10,399** | 27,250 |
| 2 | 31,105 | 27,254 |
| 3–8 | ~31,10x | ~27,25x |

Level 3 is ~27,250 at every alignment: worse than level 1's lucky case, better
than its common one, and stable. The level is still 1, because raising it is a
CPU trade that has not been measured on the write thread — but a ratio quoted
from level 1 is a measurement of one payload's alignment, not of the format.

**Every compressed figure in this document is a level-1 figure**, including the
`v3 + zstd-1` column below and the motivating query's 10,446 B. If the level
ever moves, those numbers must be re-measured *in the same change* — otherwise
the document keeps a lucky-alignment ratio that a second implementation will try
and fail to reproduce. The level does not affect interoperability: a decoder
reads any zstd frame, and byte-for-byte comparison between engines is done on
uncompressed payloads.

## Design rule

**Every batchable record is `u32 opcode · u32 count · blocks…`, built from five
shared blocks.** There is no separate "batch" record type: `count == 1` and
`count == 10_000` are the same record, so nothing is left un-batchable and the
decoder has one shape per opcode. `ADD_SCHEMA` and `ADD_ATTRIBUTE` are
inherently singular and carry no count — the one exception.

The blocks are written and tested once each, then recombined:

| block | layout |
| --- | --- |
| `IdList` | `u32 n_segments` · `Segment × n_segments` — ordered, duplicates kept |
| `RelType` | `i32` — one per record |
| `LabelSet` | `u16 n` · `i32 × n` — **n, not count**; hoisted, always plain |
| `AttrSet` | `u16 n` · `u16 attr_id × n` · `SIValue × (count × n)` — attribute ids stated **once**, rows row-major |

### An `IdList` is a sequence of segments

There is no plain form, no dictionary, and no encoding discriminator for the
list as a whole. A list is a **count of segments followed by the segments**, and
each segment writes itself:

```text
Segment := u8 header · payload
  header bits 0-1  kind: 0 = Range, 1 = Ascending, 2 = Repeat
         bits 2-3  value width code   (Range base, Repeat id)
         bits 4-5  count width code   (Range len,  Repeat count)
         bits 6-7  reserved, MUST be zero — a decoder rejects a header that
                   sets them rather than masking them off

  Range     := uint(value_width) base · uint(count_width) len
  Repeat    := uint(value_width) id   · uint(count_width) count
  Ascending := u32 blob_len · roaring64[blob_len]
```

A width code is `0,1,2,3` for `1,2,4,8` bytes. Ids and lengths are written at
the narrowest width that holds them; FalkorDB allocates entity ids densely from
zero and reuses them from a free list, so the width tracks the graph's size.

**`Range` alone can describe any list**, and `Repeat` is what stops that being
expensive. A single id is `len == 1`, so a shuffled list is one segment per id.
A *repeated* id would be one segment each too — and edge endpoints are nothing
but repeats, since every edge out of a supernode carries the same source. So
`Repeat { id, count }` is its own kind:

* it is **not** a degenerate range — a range ascends by one per step, a repeat
  does not move;
* and it can **never** become a bitmap, which holds a value once.

That is the third kind's whole justification, and it is what replaces v2's
dictionary encoding. Measured on 10,000 edges out of one supernode: the
dictionary was 10,058 B, segments without `Repeat` were 60,031 B, and with it
the whole record is **43 B** — the sources are one `Repeat`, the destinations
one `Range`, the edge ids another.

**`Ascending` is what several ranges collapse into** when a bitmap is cheaper.
It only ever describes ranges that were already ascending, because a bitmap
holds neither a repeat nor a step backwards.

The whole shape is decided as the ids arrive, not rediscovered at encode time —
so a bulk create or a delete-by-label is one `Range` from the first push to the
last and never allocates. The segments *are* the encoding.

#### Both counts are stated, deliberately

The segment count is on the wire, and so is every segment's length, including
the last. Either could be inferred — the record's own id count fixes the total —
and doing so would save four bytes per record plus one for the final length.
Measured on ten thousand single-id records that is 40,000 and 10,000 bytes
respectively, and it is refused for the same reason in both cases: **a segment
list has to be well-formed on its own, not only inside the record carrying it.**
Inferring either makes a truncated list indistinguishable from a complete one,
and lets a wrong record count be silently absorbed by the final segment —
binding rows to the wrong entities instead of failing.

### When ranges collapse into a bitmap

The rule is normative. Two engines must reach the same segmentation for the same
ids, or the same write produces different bytes.

An encoder tracks the ascending **run** it is currently building — the segments
since the last id that did not ascend — and collapses it when the bitmap is
strictly cheaper than the ranges it would replace:

```text
collapse when  range_bytes >= 32  and  5 + bitmap_bytes < range_bytes
```

`range_bytes` is the run's segments as they would be written, counting only
those that have stopped growing. The `5` is the `Ascending` segment's own header
byte and `u32` length prefix. The `32` is a floor: roaring cannot serialize
smaller than 30 bytes, so a cheaper run has nothing to weigh.

Three properties of that rule matter more than its constants:

**It is evaluated on every new segment, and the trigger is the run's own shape.**
Not a counter over the list, and not a pass at encode time. A counter would make
the comparison point arbitrary — twenty ascending segments weighed at sixteen and
the same twenty weighed together can disagree — so the bytes would depend on when
an encoder chose to look rather than on the ids.

**`bitmap_bytes` is arithmetic, not a trial build.** Roaring's serialized size is
a closed-form function of how many ids there are, how many maximal runs they
form, and how they spread across 65,536-wide buckets, and a segment list already
knows all three. Nothing is ever built in order to be measured and discarded; a
bitmap is constructed once, after it has already won. The formula is in
`RunCost`/`Run` in `graph/src/effects/v3/id_list.rs`, and
`predicted_matches_roaring` pins it against the crate on every bucket shape.

**A decline doubles nothing and retires nothing.** The comparison is not
monotone — crossing a bucket boundary adds a container and steps the bitmap's
size up — so a run that loses now can win later and is re-weighed on the next
segment.

### Constructing a bitmap is normative too, not just optimizing it

Roaring's `optimize()` is **path-dependent**. From an `Array` store a container
converts to runs only on a strict win; from a `Run` store it *stays* runs unless
strictly beaten — and a run-flavoured bitmap carries a different header. So the
same set serializes to different bytes depending on how it was built: four
three-id buckets are **73 bytes** built by ranges and **76** built id by id.

Calling `optimize()` is therefore necessary but not sufficient. **A bitmap
segment MUST be built with one range insertion per contributing range**
(`RoaringTreemap::insert_range`, `roaring64_bitmap_add_range`), never id by id.
`construction_order_changes_the_bytes` pins the difference.

The pinned `roaring` version is part of this too: the crate's container
heuristics are internal and not stable across releases by contract, so a patch
bump can move the wire. It is pinned exactly rather than caret-matched.

### Why the ids are never expanded while decoding

`read_ids` returns the segments, not the ids. That is a hard requirement rather
than an optimisation: one valid segment describes four billion ids in seven
bytes, so expanding while decoding makes a decoder whose cost is unbounded by
its input — and `GRAPH.EFFECT` is applied inline on the server's main thread,
with no timeout and no cancellation. Expanding is the applier's decision, made
where the graph is in scope and the memory is inherent to the write.

Decoding into segments also keeps the segmentation the peer chose, so a buffer
decoded and re-encoded comes back byte-identical. Re-pushing the ids would
re-run the collapse rule and could group them differently.

### Duplicates and order are the ordinary case

A roaring bitmap is a *set*: it deduplicates and it sorts. Edge endpoints do
neither — many edges share a source, and each endpoint must stay positionally
aligned with its edge id. v2 needed a separate dictionary encoding for that;
segments do not, because a repeat is just another `Range` of one.

An `Ascending` segment's decode **must check its cardinality against the ids the
record still owes**, and the segment list as a whole must total the record's
count. Row *k* belongs to the k-th id as written, so a bitmap one id short would
land every later row on the wrong entity rather than fail.

### Values stay fixed-width

`SIValue` payloads are not varint-encoded. Varints would shrink small integers,
but they make every field's position depend on the values before it, so a decoder
cannot skip a record without parsing it and the two engines must agree on a
second variable-width scheme. The bytes they save are the ones compression
already removes.

## Unchanged from v2 — the primitives

Primitives (from [`effects-v2-c-wire-format.md`](effects-v2-c-wire-format.md), read from C source):

| primitive | bytes |
| --- | --- |
| `EffectType` | 4 |
| `SchemaType`, `GraphEntityType`, `ConstraintType`, `IndexFieldType` | 4 |
| `EntityID` | 8 |
| `AttributeID` | 2 |
| `LabelID` | 4 |
| `SIType` | 4 (bitmask, not an ordinal) |
| constraint attr count | 1 |

Strings are `uint64 len = strlen + 1`, then `len` bytes **including** the trailing
NUL. `SIValue` is `uint32 SIType` then payload; `T_POINT` is 2×f32, `T_ARRAY` and
`T_VECTOR_F32` use `uint32` counts.

Records are **packed** — C writes them through `#pragma pack(push, 1)` structs, so
there is no alignment padding to reproduce.

An `AttributeSet` is `ushort count`, then `count` × (`AttributeID` id, `SIValue`
value).

### What v2 looked like, for reference

Every record changes shape in v3 (see below) — a count and the shared blocks — so
nothing in the record list survives untouched. Two v2 details are worth recording
because they are easy to get wrong when reading C:

- `1`/`2 UPDATE_*` are **one record per (entity, attribute)** in v2 — not one record
  per entity with a count and N pairs, which is what Rust emitted. A structural
  difference, not a width one.
- `10 ADD_ATTRIBUTE` has **no node/relationship discriminator**. Rust used to write
  one; that was a consequence of two attribute dictionaries, which #2459 unified, so
  dropping it is a correctness alignment rather than a loss.

For the v2 byte layouts themselves see
[`effects-v2-c-wire-format.md`](effects-v2-c-wire-format.md), read from C source.

## Changed in v3

    3 CREATE_NODE
      v2   ushort n_labels · LabelID[n_labels] · AttributeSet      (one record per node, no id)
      v3   u32 count · IdList · LabelSet · AttrSet

    4 CREATE_EDGE
      v2   ushort n_rels · RelationID · EntityID src · EntityID dest · AttributeSet
      v3   u32 count · IdList · RelType · IdList(src) · IdList(dst) · AttrSet

    1·2 UPDATE_NODE / UPDATE_EDGE
      v2   EntityID id · AttributeID attr_id · SIValue value       (one record per (entity, attribute))
      v3   u32 count · IdList · AttrSet

    5·6 DELETE_NODE / DELETE_EDGE
      v2   EntityID id                       | EntityID id · RelationID · src · dest
      v3   u32 count · IdList                | u32 count · IdList · RelType · IdList(src) · IdList(dst)

    7·8 SET_LABELS / REMOVE_LABELS
      v2   GrB_Index blob_size · GxB_Vector_serialize(nodes)[blob_size]
      v3   u32 count · IdList(nodes) · LabelSet

    9 ADD_SCHEMA
      v2   t · SchemaType · string name          (no id — the replica infers it)
      v3   t · SchemaType · LabelID|RelationID id · string name

    10 ADD_ATTRIBUTE
      v2   t · string name                       (no id)
      v3   t · AttributeID id · string name

    11 CREATE_INDEX   — one record per field, unchanged from v2
      t · SchemaType · LabelID id · string label · AttributeID id · string attr
        · IndexFieldType · SIValue options
    12 DROP_INDEX      — mirrors 11 without the options
    13 CREATE_CONSTRAINT
      t · ConstraintType · GraphEntityType · LabelID id · string label
        · uint8 n · (AttributeID id, string name) * n
    14 DROP_CONSTRAINT — mirrors 13

Three width traps in these, all read from C source rather than inferred:
`IndexFieldType` is a **bit flag set**, so a range index is
`NUMERIC|GEO|STR = 0x0E` and not a discriminant of its own; `GraphEntityType` is
**1-based**, because `GETYPE_UNKNOWN` takes 0, so a node is 1; and the
constraint property count is a **`uint8`**, not the `u16` used everywhere else.

`CREATE_INDEX` is deliberately not batched. Index DDL is rare, and C applies it
idempotently per field — `Index_SetLanguage` tolerates a re-set,
`Index_SetStopwords` is guarded — which is what lets a replica take the fields
of one index in any order.

### Entity ids become explicit

The replica stops inferring ids from creation order, so an id disagreement fails a
bounds/existence check on the spot instead of silently writing the right value to
the wrong entity. This is the item the earlier attempt was parked on; v3 removes
the problem rather than reimplementing it.

### Schema and attribute records carry their id

C states the invariant in its own source (`effects_apply.c`, above `VerifySchema`):
the id is authoritative *"only because every schema mutation is itself an effect,
applied in the same order on every replica — the name is a cheap cross-check that
surfaces divergence instead of silently trusting a stale/incorrect id."*

The two records that perform that assignment are the only ones carrying no id.
Audited against `origin/master` (2caafdaae):

| record | schema ids on the wire | can it detect divergence? |
| --- | --- | --- |
| `9 ADD_SCHEMA`, `10 ADD_ATTRIBUTE` | **none** — name only, id inferred from append order | only "this name already exists locally" |
| `1`/`2 UPDATE_*` | `AttributeID`, bare | no |
| `3 CREATE_NODE`, `4 CREATE_EDGE` | `LabelID`/`RelationID` + `AttributeID`, bare | **no, and cannot** — `static void ApplyCreateNode` has no return value |
| `6 DELETE_EDGE`, `7`/`8 *_LABELS` | `RelationID`/`LabelID`, bare | entity existence only |
| `11`–`14` index, constraint | id **+ the name** | yes — `VerifySchema`/`VerifyAttribute`, whose only call sites are `create_index_effect.c:113,119`, `drop_index_effect.c:91,96`, `create_constraint_effect.c:103,106`, `drop_constraint_effect.c:97,100` |

So records 9 and 10 *establish* the id spaces, 1–8 *consume* them with bare ids, and
only 11–14 check anything. A numbering disagreement is introduced by a record that
cannot report it, then silently trusted by every record that uses it.

The existing "already exists locally" check in `ApplyAddSchema` / `ApplyAddAttribute`
is necessary but not sufficient: it catches a replica that already holds the name,
but not a replica whose dictionary is a *different length*, where appending the same
new name yields a different id. That is the case that actually bit us — Rust's split
node/relationship attribute-id spaces against an RDB-seeded replica holding the
unified numbering: a property effect carried a bare `u16` and landed the value on the
wrong attribute, and healed on resync, so it hid.

v3 therefore carries the id on both records. The replica computes the id it *would*
assign, asserts it equals the id on the wire, and fails the buffer on mismatch — so
divergence surfaces where it is introduced rather than never.

The bulk records deliberately keep bare ids: a name per row would undo the batching,
and once assignment is pinned they no longer need one. Two consequences:

- **Ordering is normative.** `9`/`10` must precede any record referencing the ids
  they introduce. Rust already does this — `build_effects_buffer` emits schema
  additions (`pending.rs:1468`) before created nodes (`:1509`).
- **`ApplyCreateNode` / `ApplyCreateEdge` must return `bool`.** They are `static void`
  in C today, so they cannot report a mismatch at all. v3's explicit entity ids are
  only *enforceable* — as opposed to merely present — once those signatures change.
  This is a C-side prerequisite, not an optional hardening.

### Labels are grouped by label, not interleaved as pairs

`SET_LABELS` carries *the labels, and all their nodes* — not one
`(node_id, label_id)` pair per node. That is what makes the node ids a contiguous
a single `Range` segment describes: for 10,000 consecutive nodes gaining one label,
the pair form is 120,008 B and the grouped form is **46 B** (2,609× smaller raw,
261× after zstd-1; measured).

Grouping is sound here because label add/remove are idempotent set operations, so
order within a record carries no information — unlike edge endpoints, which is why
those stay an `IdList`.

A query that applies *different* label sets to different nodes splits into one
record per distinct label set, exactly as differing attribute shapes do. That
split is cheap in practice because a label list is **parsed, not evaluated**:
`SetItem::Label { var, labels: OrderSet<L> }` (`graph/src/parser/ast.rs:870`) is
filled by `parse_labels()` at parse time, so the number of distinct label sets a
query can produce is bounded by the query text — a handful, never one per row.
Attribute shapes have no such bound, since a map key can be computed.

### Attribute ids are stated once per record

`AttrSet` hoists the attribute ids out of the rows: `u16 n · u16 attr_id × n`,
then `count × n` values. For 10,000 nodes with 3 properties that removes
`2 × 10_000 × 3 = 60_000` bytes of repeated ids (468,936 B → 388,944 B, −17.1%).

The saving is stated as a raw byte count deliberately. Its effect *after*
compression is data-dependent and occasionally negative — the repeated ids also
acted as row delimiters that zstd-1's match finder was using — so the raw figure,
which always applies and needs no measurement, is the one the design rests on.

**`T_NULL` in a value slot means "remove this attribute".** FalkorDB never
*stores* a null property, so `SET n.x = NULL` is a removal, and an `UPDATE_NODE`
or `UPDATE_EDGE` row carrying `T_NULL` is what replicates it. A replica that
filtered nulls out before applying a row would turn every removal into a no-op.

That leaves no second meaning for the tag, so **a shape is exact**: every entity
in a record has precisely the attribute ids the record's `AttrSet` lists, and no
row is ever padded. Grouping two entities with different-but-overlapping
attribute sets into one wider record would cut the record count, and must not be
done under this encoding — the pad it inserts is byte-identical to a removal, so
the replica would delete a property the primary still holds. Widening the
partition key requires first carrying presence separately from value, for
instance a per-record bitmap of occupied cells.

**Rows stay row-major; columnar was measured and rejected.** Grouping values by
attribute instead of by row saves exactly zero bytes uncompressed, and its
compressed result changes sign with the data: up to −73% on compressible values,
but **+119%** on high-entropy values at 20 columns (160,314 B → 350,687 B,
zstd-1). A second wire layout that two independently written engines must match
byte-for-byte, and that gives up streaming row-at-a-time decode, is not worth a
benefit that is not reliably positive.

### What a shape split costs

A record holds one shape, so a query that varies the shape splits into many
records. Measured against the shipped codec (`the_motivating_query_end_to_end`),
10,000 nodes created by `UNWIND range(0,9999) AS i CREATE (:L {v:i})`, where v2
emitted one 34-byte record per node:

| distinct shapes | v2 | v3 |
| --- | --- | --- |
| 1 — all share one shape | 340,001 B | **120,051 B, −64.7%** |
| 20 | 340,001 B | **140,401 B, −58.7%** |
| 500 | 340,001 B | **150,001 B, −55.9%** |
| 2,000 | 340,001 B | **180,001 B, −47.1%** |
| 10,000 — every node its own | 340,001 B | **339,745 B, −0.1%** |

**There is no floor to pay for.** An earlier draft of this design budgeted +15%
on the pathological case, and a still-earlier one with a separate batch opcode
cost +99%. Merging the full-width and narrowed id encodings removed it: narrowing
the id gives back what the count field costs, so a shape carried by one node
lands at parity with v2. A single record is 33 bytes for a one-byte id, 34 for a
two-byte id (exactly v2) and 36 past 2^16. A regression test asserts the worst
case stays at or under v2's 340,001 bytes.

## Building a v3 payload

`graph/src/effects/v3/emit.rs`, the mirror of `effects_v3_apply.rs` —
and deliberately shaped like it:

    emit    Pending  --digest-->  Record  --encode-->  bytes
    apply   bytes    --decode-->  Record  --apply-->   Graph

`Record` is the pivot both directions turn on. Digesting into it rather than
writing bytes straight out of `Pending` costs nothing: the grouping already
allocated the ids, labels, attribute ids and row values a `Record` holds, so
they are moved into a value instead of passed as four loose slices. What it buys
is a testable surface for the grouping — the emit tests assert on records rather
than on decoded buffers — and a round-trip that compares values:
`digest(pending) == read_buffer(encode(digest(pending)))`.

The v2 emitter sits beside it in `effects/v2/`, whole and separate, so
retiring the old format is deleting a file rather than untangling one. Neither
lives in `pending.rs`: a mutation accumulator should not know what a replication
frame looks like. `Pending`'s fields are `pub(crate)` and the wire knowledge
lives in the emitters.

An earlier revision put a per-format view struct between them. Those views were
a pure projection — every field a reference to the same type in `Pending` — so
they documented the dependency surface and did nothing else, while costing a
struct and a constructor per format. `digest`'s return type states that surface
better than a mirror of the source could.

**The shapes are deliberately not pre-grouped inside `Pending`.** Attributes
accumulate incrementally — `set_node_attribute` inserts into a sorted vec — so
`CREATE (n) SET n.a=1 SET n.b=2` changes that node's shape twice, and
pre-bucketing would mean re-bucketing on every attribute write. It would also
charge every write for a grouping most of them never use, since effects are only
built when something is actually replicating. The grouping stays a view computed
at commit.

The grouping is cheap because `Pending` already holds most of it: `created_nodes`
is a `RoaringTreemap`, so ids arrive ascending for free; `created_rels_by_type`
is already partitioned by relationship type; and the attribute vectors are
already "sorted by id, unique", which is exactly the shape key.

Three rules the emitter has to enforce that the format alone does not:

**Label order must be normalized.** `set_labels` keeps caller order, so `[7,8]`
and `[8,7]` arrive for the same set. They are one shape, and `LabelSet` is
emitted ascending — which is also what makes two engines that agree on the set
produce the same bytes.

**Group iteration must be ordered.** A `HashMap` iterates arbitrarily, so the
same query could emit its records in a different order on two runs. That alone
would defeat the byte-for-byte comparison the cross-engine harness rests on, so
groups are sorted by key before emission.

**A new attribute is announced once, not once per entity kind.** #2459 unified
the two attribute dictionaries, so `get_node_attribute_names` and
`get_relationship_attribute_names` return the same store. v2 could iterate both
harmlessly because its `ADD_ATTRIBUTE` carries a node/relationship
discriminator and the replica's registration is get-or-create. v3's does not
carry one — correctly, since C has a single dictionary — so iterating both would
announce every new attribute twice under the same id.

### `DELETE_EDGE` needs the type captured at deletion time

v3 groups deleted edges by relationship type, and the type cannot be recovered
when the buffer is built: the edge is already gone. `index_remove_edge_docs` is
not a substitute — it is only populated for types that carry an index
(`graph.rs`, guarded by `has_index`).

So `delete_relationships` and `delete_implicit_edges` now return
`DeletedEdge = (RelationshipId, type_id, NodeId, NodeId)`. Both already resolved
the type internally and were discarding it.

## Applying a v3 payload

`graph/src/effects/v3/apply.rs`. It lives in the `graph` crate rather
than beside the v2 handler because it touches only graph types — no
`redis_module` — which also lets it be tested against a real `Graph`.

**The v2 path's batching machinery disappears.** v2 has to reconstruct batches it
never received: it accumulates runs of adjacent `DELETE_NODE` records and flushes
on a type change, because applying a node's edges one at a time made a replica
~40x slower than the master that produced the writes. In v3 a record already
covers every entity of its shape, so each one applies as a single bulk call. The
look-ahead, the two pending batches and the flush logic all go.

**The id checks are the point.** On `ADD_SCHEMA` and `ADD_ATTRIBUTE` the replica
computes the id it *would* assign and rejects the whole buffer if it disagrees
with the wire. C's reader has a weaker check — it refuses a name that already
exists locally — which catches a replica that is *ahead* but not one whose
dictionary is a different length, where appending the same new name yields a
different id. Index and constraint records additionally resolve label and
attribute by id and confirm the name matches, mirroring C's `VerifySchema` and
`VerifyAttribute`.

Two behaviours worth stating because they are choices, not consequences:

- **A constraint installs the master's outcome; the replica does not
  re-validate.** A replica scanning independently would do so at a different
  time against different write interleavings and could legitimately reach a
  different status.
- **Index creation uses the sync variant.** A replica must not spawn population
  threads, and must not reorder its work against the effect stream.

`OPTIONS {...}` now survives the trip. v2 dropped the map on the wire and forced
those statements to replicate as verbatim `GRAPH.QUERY`; v3 carries it, and the
replica rebuilds the same `IndexOptions` through the same
`map_to_index_options` the master used rather than approximating it.

## v2 against v3, measured

`cargo bench -p graph --bench effects_versions`, on
`UNWIND range(0,N-1) AS i CREATE (:Person {name:'n'+i, age:i%80})` — a string
property and an int, which is a more representative payload than the
single-int case quoted elsewhere in this document.

### Payload

| nodes | v2 | v3 | v3/v2 | v3 + zstd-1 | vs v2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 40 B | 56 B | **140%** | 56 B | 140% |
| 100 | 3,991 B | 2,819 B | 70.6% | 319 B | 8.0% |
| 1,000 | 40,891 B | 28,920 B | 70.7% | 1,766 B | 4.3% |
| 10,000 | 418,891 B | 298,920 B | 71.4% | 11,101 B | 2.7% |
| 100,000 | 4,288,891 B | 3,088,922 B | 72.0% | 39,060 B | 0.9% |
| 1,000,000 | 43,888,891 B | 31,888,922 B | 72.7% | 1,607,295 B | 3.7% |

The v3 and compressed columns are re-measured on the segment format; the v2
column is historical, from when v2 existed to measure.

Two rows are worth reading against the rest. The **1-node** row is 40% larger
under v3, not 30%: a segment states its own count and length, so the smallest
possible record pays for structure it cannot amortise. And the **1,000,000** row
compresses far worse than the 100,000 one — 3.7% against 0.9% — which is not a
property of the format but of zstd level 1; see the alignment note above.

Two things this says that the single-int figures do not. **A single-node write is
30% larger under v3** — the 4-byte opcode, the 4-byte count and the 4-byte label
ids cost more than narrowing the id saves when there is only one row to amortise
them over. Forty bytes against fifty-two decides nothing, but the direction is
worth stating. And **the steady-state saving here is ~28%, not the 65% the
single-int query shows**, because a string property is the same bytes in both
formats: v3 removes framing, and the more of the payload is values, the less
framing there is to remove. Compression is what changes the order of magnitude.

### Encode, on the write thread

| nodes | v2 | v3 | v3 + zstd-1 |
| --- | --- | --- | --- |
| 1 | 102 ns | 324 ns | 1.38 µs |
| 100 | 2.08 µs | 5.98 µs | 12.2 µs |
| 1,000 | 21.6 µs | 54.6 µs | 77.8 µs |
| 10,000 | 207 µs | 482 µs | 656 µs |
| 100,000 | 2.39 ms | 5.55 ms | 6.90 ms |
| 1,000,000 | 68.6 ms | 146 ms | 162 ms |

**v3 encoding costs 2.1–3.2× what v2 costs**, and that is the honest headline.
v2 walks its entities and writes; v3 hashes a shape key per entity, builds and
sorts the groups, and gathers row-major values. The bytes are bought with CPU on
the GIL-holding write thread, and at a million nodes that is +78 ms.

### Decode

| nodes | v3 | v3 + zstd-1 |
| --- | --- | --- |
| 1 | 105 ns | 103 ns |
| 100 | 3.69 µs | 6.38 µs |
| 1,000 | 37.8 µs | 49.8 µs |
| 10,000 | 404 µs | 505 µs |
| 100,000 | 4.02 ms | 4.83 ms |
| 1,000,000 | 41.7 ms | 51.1 ms |

v2 has no standalone decoder to compare against — its bytes are read and applied
to a `Graph` in one pass, in the root crate — so this is v3 only. Decompression
adds 12–25%.

### Where the v3 writer spends its time

100,000 nodes of one shape, `cargo bench -p graph --bench effects_versions --
writer_breakdown`. **Wall-clock**, via criterion — every table in this section
is time; the instruction counts further down are the only ones that are not. The
id and attribute blocks are measured directly; the grouping and row gathering are
what is left of the whole encode once those are removed.

| stage | before | after | share now |
| --- | --- | --- | --- |
| grouping + row gathering | 4.72 ms | **2.04 ms** | 67% |
| writing values (`AttrSet`) | 0.70 ms | 0.70 ms | 23% |
| roaring ids (bitmap segment) | 0.31 ms | 0.32 ms | 10% |
| whole encode | 5.73 ms | **3.05 ms** | 100% |

The writer started at 18% of its time actually writing bytes and 82% in the
batching machinery — deciding which entities share a shape and collecting their
values row-major. Two changes took that to 33/67 and cut the whole encode by
**46%**:

- **The shape key no longer allocates per node.** It is built into a scratch
  pair that is cleared and refilled, and cloned only when the shape turns out to
  be new. Collecting a fresh `Vec` per node meant 200,000 heap allocations for
  100,000 nodes, for keys that are overwhelmingly duplicates.
- **The map is consulted only when the shape changes.** Removing the allocation
  alone moved almost nothing — the cost was *hashing* the key, not building it.
  Entities of one shape arrive in runs, so remembering the last shape and
  comparing against it turns the common case into one hash for the whole query
  and a two-element slice comparison per node.
- **`gather_rows` merges instead of searching.** Both the shape and the entity's
  attribute vector are sorted by attribute id, so a merge walk replaces a linear
  `find` per cell. That was quadratic in the shape's width — unnoticeable at two
  attributes, not at twenty.

Roaring is not the expense, at 10%, which is worth saying because it is the part
that looks expensive.

Encode against v2 across the scales, after:

| nodes | v2 | v3 before | v3 after | |
| --- | --- | --- | --- | --- |
| 100 | 2.08 µs | 5.98 µs | 3.86 µs | 1.86× v2 |
| 1,000 | 21.7 µs | 54.6 µs | 33.0 µs | 1.52× |
| 10,000 | 206 µs | 482 µs | 286 µs | 1.39× |
| 100,000 | 2.38 ms | 5.55 ms | 3.07 ms | 1.29× |
| 1,000,000 | 69.8 ms | 146 ms | **78.9 ms** | **1.13×** |

A single-node write got slightly worse — 324 ns to 411 ns — because the grouping
now keeps three containers instead of one and clones the key twice for the first
shape. Eighty-seven nanoseconds on a one-node write is not worth complicating
the common path to recover.

### Every mode, every scale, both sides

The query throughout is

```cypher
UNWIND range(0,N-1) AS i CREATE (:Person {name:'n'+i, age:i%80})
```

One label, one string property, one int — a single shape, which is v3's best
case. 66 characters however many nodes it creates.

**Instructions retired** (`ri_instructions` from `proc_pid_rusage` — not cycles,
not wall-clock), plus bytes added to the replication stream. Each measurement
uses a fresh graph and drops it outside the window; without that, million-node
graphs accumulate and the machine measures paging.

**Verbatim here is a build with `build_effects` forced false**, so no payload is
constructed at all. That is the question worth asking — *what do effects cost
against a system that does not have them* — and the stock verbatim path cannot
answer it, because it builds a payload and discards it (see below). The
no-effects master at 100,000 nodes reads 820.7 Mi against the 821.2 Mi measured
on a server with no replica at all, which is the anchor saying the two runs are
comparable.

| nodes | mode | master Mi | replica Mi | total Mi | vs verbatim | wire bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **verbatim** | 13.4 | 4.6 | 18.0 | 1.00× | 103 |
| | v2 | 13.4 | 4.1 | 17.5 | 0.97× | 156 |
| | v3 | 13.4 | 4.1 | 17.5 | 0.97× | 162 |
| 100 | **verbatim** | 14.1 | 5.3 | 19.4 | 1.00× | 106 |
| | v2 | 14.1 | 5.6 | 19.7 | 1.02× | 4,111 |
| | v3 | 14.2 | 4.7 | 18.9 | 0.97× | 2,958 |
| | v3+zstd | 14.3 | 4.7 | 19.0 | 0.98× | 421 |
| 1,000 | **verbatim** | 21.8 | 13.0 | 34.8 | 1.00× | 108 |
| | v2 | 22.5 | 20.8 | 43.3 | **1.24×** | 41,013 |
| | v3 | 22.7 | 10.9 | 33.6 | 0.97× | 29,060 |
| | v3+zstd | 23.1 | 11.1 | 34.2 | 0.98× | 1,878 |
| 10,000 | **verbatim** | 96.6 | 84.2 | 180.8 | 1.00× | 111 |
| | v2 | 101.9 | 160.3 | 262.2 | **1.45×** | 419,015 |
| | v3 | 103.6 | 68.6 | 172.2 | 0.95× | 299,062 |
| | v3+zstd | 106.1 | 70.5 | 176.6 | 0.98× | 11,222 |
| 100,000 | **verbatim** | 820.7 | 803.5 | 1,624.2 | 1.00× | 113 |
| | v2 | 868.7 | 1,628.2 | 2,496.9 | **1.54×** | 4,289,017 |
| | v3 | 883.6 | 648.8 | 1,532.4 | 0.94× | 3,089,074 |
| | v3+zstd | 912.3 | 656.1 | 1,568.4 | 0.97× | 39,226 |
| 1,000,000 | **verbatim** | 8,544.2 | 7,696.5 | 16,240.7 | 1.00× | 115 |
| | v2 | 9,092.6 | 17,860.0 | 26,952.6 | **1.66×** | 43,889,019 |
| | v3 | 9,354.7 | 6,762.9 | 16,117.6 | 0.99× | 31,889,281 |
| | v3+zstd | 9,513.0 | 6,586.2 | 16,099.2 | 0.99× | 259,319 |

**v3 is break-even with not having effects at all** — 0.94× to 0.99×, never
worse. It spends 3–10% more on the master to encode and gets it back on the
replica, which applies records instead of re-running the query.

**v2 is not.** It is level to n=100 and then diverges: 1.24×, 1.45×, 1.54×,
**1.66×** at a million nodes. Per-entity records mean a graph operation per
entity, and the replica pays 2.3× what re-executing the query would have cost.

**Verbatim's wire cost is constant at about 110 bytes.** It replicates the query,
and the query does not grow; effects replicate the result, so v3 is 277,000×
verbatim at a million nodes. Neither effects format is remotely competitive on
bandwidth and no byte-shaving inside the format changes that. **For a
deterministic bulk write, replicating the query is five orders of magnitude
smaller than replicating its effects.**

So the case for effects is not bytes, and against a system without them the CPU
is a wash. It is:

- **Correctness.** A non-deterministic query cannot be replicated verbatim at
  all — `rand()`, `timestamp()`, anything reading the clock. Those must be
  effects whatever they cost.
- **Engine independence.** A verbatim query needs a replica implementing the
  same query language the same way. A C replica cannot run Rust's planner, which
  is the whole reason the migration needs a wire format rather than a shared
  dialect.

What the table does say is that **v3 makes effects cost what not having them
costs**, where v2 charged up to 1.66×. That is the honest claim.

### Compression is nearly free, and it is what makes the bytes bearable

| nodes | v3 CPU | + zstd‑1 | CPU cost | v3 wire | + zstd‑1 | saving |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 33.6 Mi | 34.2 Mi | +1.8% | 29,060 B | 1,878 B | 15× |
| 10,000 | 172.2 | 176.6 | +2.6% | 299,062 | 11,222 | 27× |
| 100,000 | 1,532.4 | 1,568.5 | +2.4% | 3,089,074 | 39,226 | 79× |
| 1,000,000 | 16,117.6 | 16,099.2 | **−0.1%** | 31,889,281 | 259,319 | **123×** |

Whole-system, compression costs about 2% of instructions and removes 93–99% of
the bytes. At a million nodes it is free within noise: what the compressor
spends, the master saves not copying 31 MB into the replica output buffer and
the replica saves not reading it.

That is a better case than the encoder-only measurement suggested, where zstd
looked like 3.245 cycles/byte against a 0.067 memcpy. The difference is that the
encoder-only view priced the compression and not the copying it avoids, on
either side. It still cannot be the default — **C vendors neither zstd nor
lz4** — but on the evidence it should be on wherever both ends can read it.

#### Why the stock verbatim path could not be the baseline

`build_effects` is set from replica presence alone (`graph_core.rs:686`), and
`should_use_effects` consults `EFFECTS_THRESHOLD` only afterwards, on a buffer
that has already been built. A query replicating verbatim therefore pays to
construct a payload it then discards.

Three readings show that rather than argue it. The reference is a server with no
replica attached, which does no replication work at all: that measured 813.6 and
821.2 Mi at 100,000 nodes in two separate runs, so call the floor ~815–821 Mi and
treat anything inside that band as sitting on it.

| reading | master Mi at 100,000 | vs the floor |
| --- | --- | --- |
| stock verbatim, before the writer optimization | 1,052.0 | ~230 above |
| stock verbatim, after it | 893.7 | ~75 above |
| `build_effects` forced false | 820.7 | on it |

The first line is the tell: verbatim's master sat level with v3's own 1,054.9
rather than near the floor, because it was building the same payload v3 sends and
then throwing it away. The second is the proof — optimizing the writer made the
verbatim path 158 Mi cheaper **without anything on that path changing**, which can
only happen if it was running the writer. The third removes the encode outright
and lands on the floor, so the gap was the discarded payload.

The three master readings come from three separate runs, as does each floor
measurement; the 1% spread between the two floor readings is the scale of the
drift, and every gap being argued here is far larger than that.

That is why the table above uses the forced-false build. Measured against the
stock verbatim path, v3 looked 0.87× — better than it is, because the baseline
was paying v3's own encoding cost.

That is the honest position, and it does not change the case for v3, because CPU
was never the case for v3. Verbatim replication cannot carry a non-deterministic
query, cannot be read by an engine that does not share the query language
implementation, and sends the query rather than its result — which is why C uses
effects at all. v3 earns its place by being **32% cheaper than the effects
format we have** and by being a format two engines can both read. The wasted
encode on the verbatim path is worth fixing on its own merits.

## Migration and rollout

Both engines must speak v3 before either emits it. C accepts `≤ its own version`,
so the ordering constraint is the ordinary one — **replicas before masters**:

1. Ship a v3-capable C release (reads v2 and v3; still emits v2).
2. Ship a v3-capable Rust build (reads v3; emits v3).
3. Upgrade replicas, then masters.

### The open question

A v3-only Rust cannot be a replica of an **already shipped** C master, because
that master emits v2. Two ways to close that:

- **(a) Migrate via a v3-capable C release.** Upgrade C in place first, then swap
  the engine. No v2 code in Rust at all. Cheapest, but it forbids migrating
  directly from an older C.
- **(b) Teach Rust to decode v2 as well.** Allows attaching Rust to any shipped C,
  but it drags both v2 warts back in on the read path: deriving ids from creation
  order, and `GxB_Vector_deserialize` against a blob some other GraphBLAS wrote.

This proposal assumes **(a)** and treats (b) as a separate, optional follow-up,
justified only if migrating from pre-v3 C without an intermediate upgrade is a
hard requirement. Note (b) is decode-only: Rust would still never *emit* v2, so a
downgrade target must be v3-capable either way.

## What is implemented

In `graph/src/effects/v3/`, codec only — no `Pending`, no `Graph`, so
the format can be reviewed and tested byte-for-byte on its own.

| | status |
| --- | --- |
| the five blocks, encode + decode | done |
| records 1–10, encode + decode, `read_buffer` | done |
| ids on `ADD_SCHEMA` / `ADD_ATTRIBUTE` | done |
| grouping `Pending` into v3 records (`effects_v3_emit.rs`) | done, wired behind `EFFECTS_VERSION` |
| records 11–14 (index, constraint) | done |
| applying v3 on the replica (`effects_v3_apply.rs`) | done, always active — reads are version-dispatched |
| the C side | not yet — nothing can ship until this exists |

### The switch

`GRAPH.CONFIG SET EFFECTS_VERSION 2|3` picks what a node **emits**, and defaults
to **2**. Reads are always dispatched on the version byte that arrived, so a
node reads both regardless of what it writes.

That asymmetry is the whole rollout mechanism, and it is the same one C uses
(`if (*v > EFFECTS_VERSION)`, `effects_apply.c:689`): **upgrade readers, then
flip writers**. A master emitting v3 at a peer that cannot read it is silent data
loss rather than a degraded mode, which is why the default cannot move until the
C engine reads v3.

The value is range-checked to 2 or 3 at the config layer. Anything else would
stamp a version byte no peer can read, and the failure would land on the replica
rather than on the operator who typed it.

### Verified against live servers

Two module instances, one replicating from the other, 2026-08-26:

- `EFFECTS_VERSION` defaults to 2; `SET` to 1 or 99 is refused with
  `expected 2 or 3` and leaves the previous value in place.
- With the master at 3, every write reached the replica as `GRAPH.EFFECT` with a
  `\x03` version byte followed by opcode 9 (`ADD_SCHEMA`).
- Flipping the master back to 2 mid-stream produced `\x02` on the wire, and the
  same replica applied it without a restart.
- Master and replica agreed exactly afterwards: 988 nodes, 987 `:Person`, 65
  `:Young`, 1 `:Later`, and matching property values — across creates, a label
  set, a delete, and the v2 write that followed.
- With `EFFECTS_COMPRESSION 1024`, a 5,000-node create reached the replica as
  `\x03` `\x01` then the length `0x0002460e` and a frame beginning `28 b5 2f fd`
  — zstd's magic number — and the two sides again agreed exactly.

## Verification

Byte-level unit tests per record are necessary but not sufficient — a wrong width
does not fail on the far side, it makes C read misaligned bytes as a type and write
through the resulting pointer (the `AttributeSet_Update` segfault seen when feeding
C our v2-ish buffers).

What the unit tests do cover (`cargo test -p graph --lib effects`, and
`--lib v3_tests` for the emit path):

- **Byte-pinned records**, so a width change fails here rather than on the far side.
- **`encode(decode(x)) == x`** across all ten record types. Stronger than checking
  the decoded values: it catches a writer and reader that agree with each other
  but not with the format, which value comparison cannot see.
- **Every truncation** of a buffer is an error rather than a panic, and every
  count read off the wire is checked against the bytes remaining before it
  reaches `Vec::with_capacity`, so a corrupt length cannot OOM us.
- The **cardinality checks**: an `Ascending` segment against what the record still
  owes, and the segment list's total against the record's count.
- On the emit path: that label order does not split a shape, that differing
  shapes do split, that a created node's labels do not also produce a
  `SET_LABELS` record, that deleted edges group by type, and that two
  `Pending`s populated in opposite orders produce **byte-identical** buffers.
- On the apply path: a schema or attribute id disagreement aborts the buffer, a
  stale label id is caught by its name, and a **master-to-replica round trip** —
  build a real `Pending`, emit, apply to an empty graph, compare observable
  state. Encoder and decoder agreeing with each other is not the same as either
  being right.

The real gate is the cross-engine harness driving the real
`falkordb/falkordb-server:edge-c` image, in both directions, for each of: live
replication, AOF replay, and RDB full-sync followed by streamed effects. That
harness is kept out of the repo — it is investigation tooling, not a regression
test — and it cannot run at all until the C side exists.

Encoding costs are measured with criterion, `cargo bench -p graph --bench
effects_v3`. Note the benchmark profile disables LTO: release's `lto = true`
cannot link the prebuilt GraphBLAS objects, so those are not shipped-binary
numbers.
