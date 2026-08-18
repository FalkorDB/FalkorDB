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

## Design rule

**v3 is v2 with exactly two record families redefined.** Every primitive width,
the string encoding, the `SIValue` tagging, and records 9–14 are unchanged from v2.

Keeping the delta this small is deliberate: it bounds the C diff to two apply
functions, keeps the review focused on the semantic change rather than on
re-litigating widths, and means the Rust work needed to speak v2 and v3 is almost
entirely shared.

## Unchanged from v2

Primitives (from `EFFECTS-WIRE-FORMAT.md`, read from C source):

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

Records unchanged from v2:

    1 UPDATE_NODE   t, EntityID id, AttributeID attr_id, SIValue value
    2 UPDATE_EDGE   (same)
    5 DELETE_NODE   t, EntityID id
    6 DELETE_EDGE   t, EntityID id, RelationID r, EntityID src, EntityID dest
    9 ADD_SCHEMA    t, SchemaType, string name
   10 ADD_ATTRIBUTE t, string name
   11 CREATE_INDEX  t, SchemaType, LabelID, string label, AttributeID, string attr,
                    IndexFieldType, SIValue options   (one record per field)
   12 DROP_INDEX    mirrors 11, without options
   13 CREATE_CONSTRAINT t, ConstraintType, GraphEntityType, LabelID, string label,
                        uint8 n, (AttributeID, string) * n
   14 DROP_CONSTRAINT   mirrors 13

`11`–`14` keep the `(id, name)` pairs that `VerifySchema` / `VerifyAttribute`
cross-check on the way in.

Note `1`/`2` are **one record per (entity, attribute)** — not one record per entity
with a count and N pairs, which is what Rust emitted. That is a structural
difference, not a width one, and it is easy to miss.

`10 ADD_ATTRIBUTE` has no node/relationship discriminator. Rust used to write one;
that was a consequence of two attribute dictionaries, which #2459 unified, so
dropping it is now a correctness alignment rather than a loss.

## Changed in v3

### 3 CREATE_NODE

    v2:  t, ushort label_count, LabelID[label_count], AttributeSet
    v3:  t, EntityID id, ushort label_count, LabelID[label_count], AttributeSet

### 4 CREATE_EDGE

    v2:  t, ushort rel_count, RelationID, EntityID src, EntityID dest, AttributeSet
    v3:  t, EntityID id, ushort rel_count, RelationID, EntityID src, EntityID dest, AttributeSet

The id becomes explicit. The replica stops inferring it, so an id disagreement
fails a bounds/existence check on the spot instead of silently writing the right
value to the wrong entity. This is the item the earlier attempt was parked on; v3
removes the problem rather than reimplementing it.

### 7 SET_LABELS / 8 REMOVE_LABELS

    v2:  t, GrB_Index blob_size, GxB_Vector_serialize(nodes)[blob_size]
    v3:  t, uint64 count, (EntityID node_id, LabelID label_id) * count

Explicit pairs instead of a serialized GraphBLAS vector. Same information — in v2
the label rides in the vector's *values* — with no dependency on either engine's
GraphBLAS build, and readable by anything.

`#1978` batched label updates to make one effect cover many nodes; the pair list
keeps that (one record, `count` pairs) without the blob.

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

## Verification

Byte-level unit tests per record are necessary but not sufficient — a wrong width
does not fail on the far side, it makes C read misaligned bytes as a type and write
through the resulting pointer (the `AttributeSet_Update` segfault seen when feeding
C our v2-ish buffers).

The real gate is the cross-engine harness (`scripts-crossengine-*`) driving the
real `falkordb/falkordb-server:edge-c` image, in both directions, for each of:
live replication, AOF replay, and RDB full-sync followed by streamed effects.
