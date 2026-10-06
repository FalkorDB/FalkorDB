# IdList (effects v3 replication) — Lean verification report

Target: `graph/src/effects/v3/id_list.rs` (plus `Reader` in `graph/src/effects/reader.rs`,
`narrow_int::width_for`). Lean 4.34.0, core only.

```
cd proofs/id_list && lake build        # Build completed successfully
grep -n "sorry\|admit\|^axiom\|native_decide" FalkorIdList/*.lean   # nothing
```
`#print axioms` on the headline theorems: only `propext`, `Classical.choice`, `Quot.sound`.
89 theorems (43 in `Push.lean`, 46 in `Wire.lean`).

Rust repros: `graph/tests/lean_id_list.rs`
(`cargo test -p graph --test lean_id_list -- --nocapture`): 6 pass, 2 fail on purpose (the
two confirmed issues below).

## What is modelled

| Lean | Rust |
| --- | --- |
| `Seg` (`range`/`rep`/`rdesc`/`asc`/`dsc`) | `enum Segment` id_list.rs:450 — bitmaps held as their set (ascending list); `len/min/max` caches derived |
| `Seg.min`, `Seg.max`, `Seg.len`, `Seg.iter` | `Segment::min` :570, `max` :557, `len` :545, `iter` :763 |
| `flat` | `IdList::iter` :1126 |
| `ins`, `insRange` | `RoaringTreemap::insert` / `insert_range` (set union) |
| `St` | `IdList { segments, len, run.start, run.desc }` :783/:182 |
| `claim`, `claimStep`, `extend?`, `push`, `push.pushTail` | `claim_direction` :857, `push` :886 (line by line, same branch order) |
| `collapse c` | `maybe_collapse_run` :1029; `none` = its `assert!` :1051 |
| `fromSegments` | `IdList::from_segments` :1176 |
| `le`/`ofLe`, `take`, `readU` | `to_le_bytes`/`from_le_bytes`, `Reader::take`/`u8..u64` (reader.rs:40, :99-113), `read_narrow` :1296 |
| `widthFor`, `widthCode`, `widthOfCode` | `width_for` (narrow_int.rs:19), `width_code` :78, `width_of_code` :89 |
| `pairHeader`, `writePair`, `encodeSeg`, `encodeIds` | header layout :534-541, `write_pair` :652, `Segment::encode` :601 (incl. `BitmapLengthLied`), `EffectEncode<3> for IdList` :1200 |
| `decodeSeg`, `loop`, `readIds` | `Segment::decode` :673, `read_ids` :1337 incl. `guard_count` (reader.rs:70); `Err.panic` = debug-build `u64` underflow in `count - len` |
| `Roaring` + `RoaringOK` | `RoaringTreemap::{serialize_into, serialized_size, deserialize_from}` — abstract codec with stated contract |

`Run::prefers_bitmap` (the roaring-size arithmetic) is **abstracted to a free Boolean per
push**: every theorem quantifies over all decision sequences, so correctness does not depend
on the arithmetic (it only affects which encoding is chosen).

## Proven (plain English)

Builder (`Push.lean`)
- `push_ok` / `pushAll_ok` / `fromIter_iter`: for ANY sequence of `u64` ids (unsorted,
  duplicates, 0, `u64::MAX`, empty) and ANY collapse decisions, the segment list iterates to
  exactly the pushed ids in push order, `len` = number of pushes, and the collapse `assert!`
  never fires. Proven via the invariant `Inv` (run = `segments[start..]` holds only ranges,
  every range in it reads in the run's direction, successive segments strictly beyond each
  other, the last range-like segment is always inside the run).
- `collapse_ok`: the collapsed bitmap iterated in the run's direction equals the run's ids
  (#2842's reversal and the "17 invented / 17 dropped" bug are both excluded by `Inv`).
- `segs_le_len`: never more segments than ids.
- `iter_bounds`, `iter_length`, `iter_lt_W`: well-formed segments iterate without panic/wrap.

Wire (`Wire.lean`)
- `ofLe_le`, `widthFor_fits`, `widthOfCode_widthCode`: narrow LE ints round-trip at the
  width `width_for` picks.
- `pairHeader_fields`, `ascHeader_fields` (exhaustive `decide`): every header the encoder can
  write decodes to its kind/direction/widths and never sets the reserved bit.
- `decode_encodeSeg`: each well-formed segment decodes back to itself, consuming exactly its bytes.
- `readIds_encodeIds`: `read_ids(encode(segs) ++ rest, count)` = `(segs, rest)` whenever
  segs are well-formed, total `count < 2^32`, and number ≤ count — hence re-encoding a decoded
  own buffer is byte-identical.
- `push_encode_decode` (headline): `decode(encode(push* xs)) = xs` for every `xs` with
  `xs.length < 2^32`.
- `readIds_never_panics`: no input reaches the `u64::from(count) - len` underflow (or any other
  modelled panic).
- `readIds_safe` (totality/soundness on arbitrary bytes): an accepted list has only
  well-formed segments (so `iter` cannot panic or wrap — ranges fit `u64`, descending ranges
  stay ≥ 0), totals exactly `count` ids, has ≤ count segments, was read from a prefix of the
  input (never over-reads) of ≥ 4 + 3·segments bytes.
- Concrete refusals (`refuses_*`, by `rfl`): reserved bit, descending Repeat, kind 3, u64 wrap,
  step below zero, count too long/short, truncation (EOF), too many segments for the bytes.

## CONFIRMED issues (Rust repro fails)

1. **Pushing onto a decoded `IdList` reorders ids or panics** — `IdList::from_segments`
   (id_list.rs:1176) sets `run.start = last segment`, `desc = None`, fresh tally, regardless of
   what that segment is. `Inv` does not hold there, and `push` is `pub`.
   - Lean: `decoded_then_pushed_reorders` (decoded `[10,9,8]` + pushes `20,22` with a collapse
     → `[8,9,10,20,22]`), `decoded_then_pushed_asserts` (decoded `[7,7]` → assert).
   - Rust: `push_after_decode_reorders_a_descending_tail` — decoded `[10,9,8]`, then
     `20,22,24,...`:
     `expected [10, 9, 8, 20, 22, 24]  got [8, 9, 10, 20, 22, 24]` (silent reorder, row↔id
     binding broken, same class as #2842).
   - Rust: `push_after_decode_of_a_repeat_hits_the_collapse_assert` — decoded `[7,7]`, then
     `20,22,...`: `panicked at graph/src/effects/v3/id_list.rs:1051:13: a run under
     consideration holds only ranges, found Repeat { id: 7, count: 2 }`.
   - Severity: latent. No production path pushes onto a decoded list today (emit builds fresh
     lists; apply only iterates). The doc comment on `from_segments` says "a later push would
     not collapse across the boundary", which is false.
   - Fix: in `from_segments`, set `run.start = segments.len()` (run begins *after* every
     decoded segment, exactly like `maybe_collapse_run`'s `restart(start + 1)`), or make the
     decoded type non-pushable.

## Non-canonical decoding (confirmed accepted; not a misparse of honest buffers)

decode∘encode is the identity (proven); encode∘decode is not — the decoder accepts several
byte strings per list. Relevant if any cross-engine / replay comparison is done on raw bytes
of a buffer that did not come from this encoder.
- Non-minimal widths: `[01 00 00 00 0c 05 00..00 01]` decodes to `[5]`; canonical is
  `[01 00 00 00 00 05 01]` (Lean `nonminimal_width_accepted`, Rust `noncanonical_wide_range_is_accepted`).
- Ascending header width bits 2-5 are never read (Rust `noncanonical_bitmap_header_width_bits_are_ignored`).
- Trailing bytes inside the stated blob length are ignored — roaring reads from the slice and
  never checks it consumed it all (Rust `noncanonical_bitmap_blob_trailing_bytes_are_ignored`).
- Lone id with the descending bit is accepted (known; `lone_descending_accepted`).

## Suspected, not confirmed

- **Duplicate treemap keys silently drop a partition.** roaring 0.11.5
  `RoaringTreemap::deserialize_from` does `BTreeMap::insert(key, bitmap)` per entry, so a blob
  with the same high key twice keeps the *last* bitmap. With the record count set to the
  survivor's cardinality, Rust accepts it and yields only the second partition's ids
  (Rust `duplicate_treemap_key_silently_drops_a_partition` passes: 80 refused, 40 accepted →
  ids of B only). CRoaring's 64-bit map deserialization may keep the *first* (emplace)
  or reject; if so, the C and Rust engines decode the same bytes to different ids. Not
  checked against the C module. Fix: after deserializing, re-serialize and compare
  `serialized_size()` to `blob_len` (also closes the trailing-bytes hole), or reject
  unsorted/duplicate keys.

## Modelling gaps / assumptions

- Roaring is abstract (`RoaringOK`: ser/deser round-trip on nonempty sorted u64 sets,
  `serialized_size = ser.length < 2^32`, deserialized sets are sorted u64 sets). Actual blob
  bytes, `optimize()`, and the `prefers_bitmap` cost arithmetic are not modelled (the latter
  is quantified away; `predicted_matches_roaring` tests it).
- Segment `len: u32` overflow inside `push` (needs ≥ 2^32 pushes; `count()` already panics
  there) is not modelled; theorems assume `xs.length < 2^32` where the wire needs it.
- Cached `len/min/max` of bitmap segments are derived from the set rather than stored; the
  proof therefore does not check cache maintenance (by inspection: `max = id` after
  `insert(id)` with `id > max` is correct).
- Allocation bounds inside roaring's deserializer (e.g. the `u32` container count in the
  no-run cookie is checked `≤ 65536` before allocating) were read, not modelled.
- `Reader` is modelled as a byte list; `Except` short-circuits exactly where `?` does.
