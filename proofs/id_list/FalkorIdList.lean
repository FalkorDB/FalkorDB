/-
# `IdList` (graph/src/effects/v3/id_list.rs) — machine-checked

* `FalkorIdList.Push` — the segment builder (`IdList::push`, collapse to roaring).
* `FalkorIdList.Wire` — the byte format (`Segment::encode/decode`, `read_ids`).
* `FalkorIdList.Cost`/`CostProps` — the collapse arithmetic (`bucket_body`, `bitmap_header`,
  `Run::{restart, absorb, prefers_bitmap, add_range, freeze_bucket, freeze_bitmap, bitmap_bytes}`,
  `Segment::encoded_len`): `bucketBody_spec` (cheapest container, tie→run), `bitmapHeader_spec`,
  `bitmapBytes_closeout` (O(1) close-out = freeze_bucket∘freeze_bitmap), `addLoop_eq`/`pieces_sum`/
  `pieces_bucket`/`addRange_top` (the loop visits exactly the bucket pieces of `[base, base+len)` and
  terminates at `u64::MAX`), `absorb_rangeBytes`/`absorbAll_rangeBytes` (range side = Σ encoded_len),
  `encodedLen_eq` (encoded_len = bytes written), `collapse_shrinks` (when prefers_bitmap fires the bitmap
  segment is strictly shorter than the closed ranges, given serialized_size = the tally),
  `no_collapse_below_floor`.
* `FalkorIdList.PushT` — the collapse decision is no longer a free Bool: the tally is threaded through
  `push` where the Rust touches `self.run`, and `c := prefers_bitmap()`. `pushT_st`, `pushAllT_ok`,
  `fromIterT`: every push sequence round-trips with the real arithmetic. `#guard`s replay two Rust
  shape tests (40 gapped singletons → 1 bitmap; two runs stay two ranges).
* `FalkorIdList.Misc` — `is_empty`, `count`, `to_roaring` (`toRoaring_spec`: exactly the id set, sorted),
  the three `PartialEq`s (`eqList_iff`: by ids, not segmentation), both `From`s (`from_eq`), `Debug`
  (`fmt_ignores_run`).

Gap: `bitmap_bytes` = roaring's `serialized_size()` is a hypothesis of `collapse_shrinks`
(roaring is a crate; Rust test `predicted_matches_roaring` checks it). Observation (not a bug): the
comparison charges only the run's *closed* segments, while the collapsed bitmap also holds the newly
pushed id, so in an edge case the collapsed list can be a few bytes longer than not collapsing.

Headline results: `IdListWire.push_encode_decode`, `IdListWire.readIds_safe`,
`IdListWire.readIds_never_panics`, `IdListPush.decoded_then_pushed_reorders`.
See REPORT.md.
-/
import FalkorIdList.Push
import FalkorIdList.Wire
import FalkorIdList.Misc
