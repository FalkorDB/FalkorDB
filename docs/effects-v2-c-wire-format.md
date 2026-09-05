# C effect wire format, v2

Read from C source on the `master` branch. Widths re-verified against
`origin/master` @ `2caafdaae` (2026-08-20); the version rule below was corrected
at the same time.

Every width below is read from source, not guessed. One wrong width does not
produce an error on the far side — C reads the misaligned bytes as a type and
writes through the resulting pointer, which is the `AttributeSet_Update` segfault.

Reference: `src/effects/effects.{c,h}`, `effects_internal.h`,
`{create,drop}_{index,constraint}_effect.c`.

## Buffer header

    uint8 version = EFFECTS_VERSION = 2

C accepts any version **at or below** its own — `if (*v > EFFECTS_VERSION)`
(`effects_apply.c:689`) — and already branches per version further down
(`if (unlikely (version == 1))`, `:757`, `:765`). So raising the version is C's
own established mechanism, not a break. An earlier draft of this file claimed
the reader "rejects anything else"; that was wrong, and it is the whole reason
v3 is viable.

## Primitive widths

| primitive | C | bytes |
| --- | --- | --- |
| `EffectType` | enum | **4** |
| `SchemaType` | enum | 4 |
| `GraphEntityType` | enum | 4 |
| `ConstraintType` | enum | 4 |
| `IndexFieldType` | enum (bit flags) | 4 |
| `EntityID` | `GrB_Index` | 8 |
| `AttributeID` | `uint16_t` | **2** |
| `LabelID` | `int` | **4** |
| label count (create node) | `ushort` | 2 |
| constraint attr count | `uint8_t` | **1** |
| `SIType` | enum bitmask | **4** |

### string

    uint64 len = strlen(s) + 1     // INCLUDES the terminator
    bytes[len]                     // trailing NUL included

Rust wrote `s.len()` and no NUL. Same defect class as the UDF aux field, but on
every string in every effect.

### SIValue

    uint32 SIType
    payload

| type | value | payload |
| --- | --- | --- |
| `T_MAP` | 1<<0 | pair count, then (key string, value) — used by index options |
| `T_ARRAY` | 1<<3 | **uint32** len, then values |
| `T_DATETIME` | 1<<5 | int64 |
| `T_DATE` | 1<<7 | int64 |
| `T_TIME` | 1<<8 | int64 |
| `T_DURATION` | 1<<10 | int64 |
| `T_STRING` | 1<<11 | string (above) |
| `T_BOOL` | 1<<12 | 1 byte |
| `T_INT64` | 1<<13 | int64 |
| `T_DOUBLE` | 1<<14 | double |
| `T_NULL` | 1<<15 | *nothing* |
| `T_POINT` | 1<<17 | **2 × float32** (8 bytes total) |
| `T_VECTOR_F32` | 1<<18 | **uint32** dim, then raw f32 bytes |

Rust used sequential 1-byte tags 0–12, f64 for point components, and u64 lengths
for arrays and vectors.

## Records

`t` below is the 4-byte `EffectType`.

### 1 UPDATE_NODE / 2 UPDATE_EDGE — packed struct, then value

    t, EntityID id, AttributeID attr_id      (4 + 8 + 2, packed)
    SIValue value

**One record per (entity, attribute).** Rust emitted one record per entity with a
`u16` count and N pairs — a structural difference, not a width one.

### 3 CREATE_NODE

    t, ushort label_count, LabelID[label_count], AttributeSet

No entity id: the replica derives it from creation order.

### 7 SET_LABELS / 8 REMOVE_LABELS — **hard part 1**

    t
    GrB_Index blob_size          (8)
    GxB_Vector_serialize(nodes)  blob_size bytes

Not a node id and label list: C serializes a **GraphBLAS vector** of the affected
nodes (`EffectsBuffer_AddLabelsEffect` takes only `GrB_Vector nodes`, no label id —
the label rides in the vector's values). From #1978 "Batch label update".

Feasible for us — we link GraphBLAS and already wrap vectors in
`graph/src/graph/vector.rs`, so `GxB_Vector_serialize` is callable — but this is
FFI work against GraphBLAS's own serialization, not a field-width change.

### 9 ADD_SCHEMA

    t, SchemaType, string name

### 10 ADD_ATTRIBUTE

    t, string name

**No node/rel discriminator.** Rust pushes `ATTR_NODE`/`ATTR_REL`. C has none
because it has one attribute dictionary — which is exactly what PR A's #2457 fix
aligned us to, so dropping the discriminator is now correct rather than a loss.

## Hard part 2: implicit entity ids

C's CREATE_NODE carries **no node id**, and CREATE_EDGE carries no edge id — only
`rel_count`, `RelationID`, src, dest. The replica assigns ids by creation order.
Rust sends explicit ids today.

Matching C means our apply path must derive ids the same way, and id agreement
between master and replica stops being checkable from the record itself. This is a
semantics change on the apply side, not just an encoding one, and it is where a
mistake would corrupt a replica silently rather than fail loudly.

### 11 CREATE_INDEX — one record per field

    t
    SchemaType st
    int label_id
    string label
    AttributeID attr_id
    string attr
    IndexFieldType field_type
    SIValue options            // T_MAP

Applied idempotently per field: `Index_SetLanguage` tolerates re-set,
`Index_SetStopwords` is guarded by `Index_ContainsStopwords`, and the
pending-changes counter supports being bumped once per field. NOT batched by
stream adjacency.

### 12 DROP_INDEX — mirrors 11

### 13 CREATE_CONSTRAINT

    t
    ConstraintType ct
    GraphEntityType et
    int label_id
    string label
    uint8 n
    (AttributeID attr_id, string attr) * n

### 14 DROP_CONSTRAINT — mirrors 13

## Why (id, name) pairs

`VerifySchema` / `VerifyAttribute`: the id is authoritative because every schema
mutation is itself an effect applied in the same order everywhere; the name is a
cheap cross-check that surfaces divergence instead of trusting a stale id. Returns
NULL/false on divergence, and `Effects_Apply` returns false so the caller stops
propagating down a replication sub-chain.

This is the mechanism PR A's #2457 fix worked around from the other side: with a
name beside the id, a split id space would have been caught rather than silently
writing to the wrong attribute.

## Two-announcement constraint protocol

`ApplyCreateConstraint` returns true both for a genuine create and for
`CONSTRAINT_ALREADY_EXISTS`, because a pending constraint is announced twice —
once when created, once when it becomes active (`Constraint_Replicate`). Every
replica in a chain may legitimately see it more than once.
