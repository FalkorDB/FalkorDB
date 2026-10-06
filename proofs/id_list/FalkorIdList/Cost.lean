import FalkorIdList.Push
import FalkorIdList.Wire
/-!
# The collapse arithmetic (`Run`, `bucket_body`, `bitmap_header`, `encoded_len`)

`id_list.rs:182-440`, `:582`. All `usize`/`u64` arithmetic here is bounded
(ids per bucket ≤ 2^16, runs per bucket ≤ 2^15, headers linear in the bucket
count), so it is modelled in `Nat`; the one place the Rust could wrap — the
`lo` loop index at the top of the id space — is modelled with its
`checked_add` break and proven to terminate and cover the range (`addLoop_*`).
-/
namespace IdListCost
open IdListPush IdListWire

def ARRAY_ELEMENT_BYTES : Nat := 2
def BITSET_CONTAINER_BYTES : Nat := 8192
def RUN_COUNT_BYTES : Nat := 2
def RUN_INTERVAL_BYTES : Nat := 4
def MAGIC_BYTES : Nat := 4
def CONTAINER_COUNT_BYTES : Nat := 4
def CONTAINER_DESC_BYTES : Nat := 4
def CONTAINER_OFFSET_BYTES : Nat := 4
def OFFSET_TABLE_MIN_CONTAINERS : Nat := 4
def TREEMAP_COUNT_BYTES : Nat := 8
def ASCENDING_SEGMENT_OVERHEAD : Nat := 1 + 4
def BITMAP_KEY_BYTES : Nat := 4
def ROARING_FLOOR_BYTES : Nat := 32

/-- `bucket_body` (`id_list.rs:267`). -/
def bucketBody (ids runs : Nat) : Nat × Bool :=
  let sparse := ids * ARRAY_ELEMENT_BYTES
  let plain := if sparse < BITSET_CONTAINER_BYTES then sparse else BITSET_CONTAINER_BYTES
  let asRun := RUN_COUNT_BYTES + RUN_INTERVAL_BYTES * runs
  if asRun ≤ plain then (asRun, true) else (plain, false)

/-- `bitmap_header` (`id_list.rs:289`). -/
def bitmapHeader (buckets : Nat) (hasRun : Bool) : Nat :=
  if hasRun then
    let runFlags := (buckets + 7) / 8          -- `div_ceil(8)`
    if buckets ≥ OFFSET_TABLE_MIN_CONTAINERS then
      MAGIC_BYTES + (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets + runFlags
    else MAGIC_BYTES + CONTAINER_DESC_BYTES * buckets + runFlags
  else MAGIC_BYTES + CONTAINER_COUNT_BYTES + (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets

/-- `struct Run`'s cost tally (`start`/`desc` live in `IdListPush.St`). -/
structure Tally where
  rangeBytes : Nat := 0
  started : Bool := false
  closedBitmaps : Nat := 0
  bitmapKey : Nat := 0
  closedBuckets : Nat := 0
  closedBody : Nat := 0
  closedHasRun : Bool := false
  bucket : Nat := 0
  bucketIds : Nat := 0
  bucketRuns : Nat := 0
deriving Repr, DecidableEq

/-- `Run::restart` (`:311`): `Self { start, ..Self::default() }` — the tally is reset. -/
def restart : Tally := {}

/-- `Run::freeze_bucket` (`:409`). -/
def freezeBucket (t : Tally) : Tally :=
  let (body, isRun) := bucketBody t.bucketIds t.bucketRuns
  { t with closedBody := t.closedBody + body, closedHasRun := t.closedHasRun || isRun,
           closedBuckets := t.closedBuckets + 1 }

/-- `Run::freeze_bitmap` (`:416`). -/
def freezeBitmap (t : Tally) : Tally :=
  { t with closedBitmaps := t.closedBitmaps + BITMAP_KEY_BYTES +
             bitmapHeader t.closedBuckets t.closedHasRun + t.closedBody,
           closedBuckets := 0, closedBody := 0, closedHasRun := false }

/-- `Run::bitmap_bytes` (`:430`). -/
def bitmapBytes (t : Tally) : Nat :=
  if !t.started then TREEMAP_COUNT_BYTES else
  let (body, isRun) := bucketBody t.bucketIds t.bucketRuns
  TREEMAP_COUNT_BYTES + t.closedBitmaps + BITMAP_KEY_BYTES +
    bitmapHeader (t.closedBuckets + 1) (t.closedHasRun || isRun) + t.closedBody + body

/-- `Run::prefers_bitmap` (`:337`). -/
def prefersBitmap (t : Tally) : Bool :=
  decide (t.rangeBytes ≥ ROARING_FLOOR_BYTES) &&
    decide (ASCENDING_SEGMENT_OVERHEAD + bitmapBytes t < t.rangeBytes)

/-- One piece of `add_range`'s loop body (`:363-399`), for the bucket `bucket`. -/
def step (t : Tally) (bucket ids : Nat) : Tally :=
  if !t.started then
    { t with started := true, bitmapKey := bucket / 65536, bucket := bucket,
             bucketIds := ids, bucketRuns := 1 }
  else if bucket = t.bucket then
    { t with bucketIds := t.bucketIds + ids, bucketRuns := t.bucketRuns + 1 }
  else
    let t := freezeBucket t
    let t := if bucket / 65536 ≠ t.bitmapKey then { freezeBitmap t with bitmapKey := bucket / 65536 } else t
    { t with bucket := bucket, bucketIds := ids, bucketRuns := 1 }

/-- `bucket << 16 | 0xFFFF`. -/
def bucketEnd (bucket : Nat) : Nat := bucket * 65536 + 65535

theorem le_pieceEnd (lo e : Nat) (h : lo ≤ e) : lo ≤ min e (bucketEnd (lo / 65536)) := by
  unfold bucketEnd
  have h1 := Nat.mod_add_div lo 65536
  have h2 := Nat.mod_lt lo (show 65536 > 0 by decide)
  omega

/-- The `while lo <= end` loop of `add_range` (`:356`), `lo`/`end` < 2^64, with
`piece_end.checked_add(1)`'s `None => break` at the top of the id space. -/
def addLoop (t : Tally) (lo e : Nat) : Tally :=
  if h : lo ≤ e ∧ e < W then
    let bucket := lo / 65536
    let pieceEnd := min e (bucketEnd bucket)
    let t' := step t bucket (pieceEnd - lo + 1)
    if h2 : pieceEnd + 1 < W then addLoop t' (pieceEnd + 1) e else t'
  else t
termination_by W - lo
decreasing_by have := le_pieceEnd lo e h.1; omega

/-- The pieces `add_range` visits: `(bucket, ids)`. -/
def pieces (lo e : Nat) : List (Nat × Nat) :=
  if h : lo ≤ e ∧ e < W then
    let bucket := lo / 65536
    let pieceEnd := min e (bucketEnd bucket)
    (bucket, pieceEnd - lo + 1) :: (if h2 : pieceEnd + 1 < W then pieces (pieceEnd + 1) e else [])
  else []
termination_by W - lo
decreasing_by have := le_pieceEnd lo e h.1; omega

/-- `Run::add_range` (`:348`): `end = base + (len - 1)`. -/
def addRange (t : Tally) (base len : Nat) : Tally := addLoop t base (base + (len - 1))

/-- `Segment::encoded_len` (`:582`); `R.size` is `serialized_size()`. -/
def encodedLen (R : Roaring) : Seg → Nat
  | .range b l | .rdesc b l => 1 + widthFor b + widthFor l
  | .rep i c => 1 + widthFor i + widthFor c
  | .asc bm | .dsc bm => 1 + 4 + R.size bm

/-- `Run::absorb` (`:323`). -/
def absorb (R : Roaring) (t : Tally) (s : Seg) : Tally :=
  addRange { t with rangeBytes := t.rangeBytes + encodedLen R s } s.min s.len

end IdListCost
