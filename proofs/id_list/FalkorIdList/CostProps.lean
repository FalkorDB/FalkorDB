import FalkorIdList.Cost
/-! Theorems about the collapse arithmetic. -/
namespace IdListCost
open IdListPush IdListWire

/-- `bucket_body` picks the cheapest container: the run store when it does not
lose (tie → run, as `optimize()`), else `min(array, bitset)`. -/
theorem bucketBody_spec (ids runs : Nat) :
    (bucketBody ids runs).1 = min (2 + 4 * runs) (min (ids * 2) 8192) ∧
    ((bucketBody ids runs).2 = true ↔ 2 + 4 * runs ≤ min (ids * 2) 8192) := by
  unfold bucketBody ARRAY_ELEMENT_BYTES BITSET_CONTAINER_BYTES RUN_COUNT_BYTES RUN_INTERVAL_BYTES
  by_cases h1 : ids * 2 < 8192
  · simp only [h1, ite_true]; split <;> simp <;> omega
  · simp only [h1, ite_false]; split <;> simp <;> omega

theorem bucketBody_le (ids runs : Nat) :
    (bucketBody ids runs).1 ≤ 2 + 4 * runs ∧ (bucketBody ids runs).1 ≤ ids * 2 ∧
    (bucketBody ids runs).1 ≤ 8192 := by
  have := (bucketBody_spec ids runs).1
  rw [this]; omega

/-- `bitmap_header`'s three layouts. -/
theorem bitmapHeader_spec (n : Nat) :
    bitmapHeader n false = 8 + 8 * n ∧
    (n ≥ 4 → bitmapHeader n true = 4 + 8 * n + (n + 7) / 8) ∧
    (n < 4 → bitmapHeader n true = 4 + 4 * n + (n + 7) / 8) := by
  refine ⟨?_, fun h => ?_, fun h => ?_⟩ <;>
    simp [bitmapHeader, MAGIC_BYTES, CONTAINER_COUNT_BYTES, CONTAINER_DESC_BYTES,
      CONTAINER_OFFSET_BYTES, OFFSET_TABLE_MIN_CONTAINERS] <;> (try split) <;> omega

/-- `restart` discards the tally. -/
theorem restart_spec : restart = ({} : Tally) := rfl

/-- `bitmap_bytes` closes the open bucket and bitmap *arithmetically*: it is
exactly what `freeze_bucket` then `freeze_bitmap` would commit, plus the treemap
prefix — without committing it. -/
theorem bitmapBytes_closeout (t : Tally) (h : t.started = true) :
    bitmapBytes t = TREEMAP_COUNT_BYTES + (freezeBitmap (freezeBucket t)).closedBitmaps := by
  simp [bitmapBytes, h, freezeBitmap, freezeBucket]; omega

theorem bitmapBytes_empty (t : Tally) (h : t.started = false) : bitmapBytes t = 8 := by
  simp [bitmapBytes, h, TREEMAP_COUNT_BYTES]

theorem prefersBitmap_iff (t : Tally) :
    prefersBitmap t = true ↔ (32 ≤ t.rangeBytes ∧ 5 + bitmapBytes t < t.rangeBytes) := by
  unfold prefersBitmap ROARING_FLOOR_BYTES ASCENDING_SEGMENT_OVERHEAD
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]

/-- Induction along `add_range`'s loop (by `W - lo`). -/
theorem pieces_ind (e : Nat) (P : Nat → Prop)
    (step : ∀ lo, lo ≤ e ∧ e < W →
      (min e (bucketEnd (lo / 65536)) + 1 < W → P (min e (bucketEnd (lo / 65536)) + 1)) → P lo)
    (stop : ∀ lo, ¬ (lo ≤ e ∧ e < W) → P lo) (lo : Nat) : P lo :=
  if h : lo ≤ e ∧ e < W then
    step lo h (fun _ => pieces_ind e P step stop (min e (bucketEnd (lo / 65536)) + 1))
  else stop lo h
termination_by W - lo
decreasing_by have := le_pieceEnd lo e h.1; omega

/-! ### `add_range`: the loop visits exactly the bucket pieces, and terminates
even at the top of the id space (the `checked_add` break). -/

theorem addLoop_eq (t : Tally) (lo e : Nat) :
    addLoop t lo e = (pieces lo e).foldl (fun t p => step t p.1 p.2) t := by
  induction lo using pieces_ind e generalizing t with
  | step lo h ih =>
    rw [addLoop, pieces]; simp only [h, and_self, dite_true, List.foldl_cons]
    split
    · rename_i hw; exact ih hw _
    · rfl
  | stop lo h =>
    rw [addLoop, pieces]; simp [h]

/-- The pieces cover `[lo, e]` exactly: their sizes sum to the range length. -/
theorem pieces_sum (e : Nat) (he : e < W) : ∀ lo, lo ≤ e →
    ((pieces lo e).map Prod.snd).sum = e + 1 - lo := by
  intro lo
  induction lo using pieces_ind e with
  | step lo h ih =>
    intro hle
    rw [pieces]; simp only [h, and_self, dite_true, List.map_cons, List.sum_cons]
    have hp := le_pieceEnd lo e h.1
    have hpe : min e (bucketEnd (lo / 65536)) ≤ e := Nat.min_le_left _ _
    split
    · rename_i hw
      by_cases hlt : min e (bucketEnd (lo / 65536)) + 1 ≤ e
      · rw [ih hw hlt]; omega
      · rw [pieces]; have : ¬ (min e (bucketEnd (lo / 65536)) + 1 ≤ e ∧ e < W) := by omega
        simp only [this, dite_false, List.map_nil, List.sum_nil]; omega
    · simp only [List.map_nil, List.sum_nil]; unfold W at *; omega
  | stop lo h => intro hle; exact absurd ⟨hle, he⟩ h

/-- Every piece lies inside one 2^16 bucket (`ids ≤ 65536`), and is non-empty. -/
theorem pieces_bucket (e lo : Nat) : ∀ p ∈ pieces lo e, 1 ≤ p.2 ∧ p.2 ≤ 65536 := by
  induction lo using pieces_ind e with
  | step lo h ih =>
    rw [pieces]; simp only [h, and_self, dite_true, List.mem_cons]
    rintro p (rfl | hp)
    · have := le_pieceEnd lo e h.1
      have : min e (bucketEnd (lo / 65536)) ≤ bucketEnd (lo / 65536) := Nat.min_le_right _ _
      have h1 := Nat.mod_add_div lo 65536
      simp only [bucketEnd] at *; omega
    · split at hp
      · rename_i hw; exact ih hw p hp
      · cases hp
  | stop lo h => simp [pieces, h]

/-- The top of the id space: a range ending at `u64::MAX` is fully visited and
the loop stops (the regression the `checked_add` break fixed). -/
theorem addRange_top (len : Nat) (h : 1 ≤ len) (hl : len ≤ W) :
    ((pieces (W - len) (W - 1)).map Prod.snd).sum = len := by
  rw [pieces_sum _ (by unfold W; omega) _ (by omega)]; unfold W at *; omega

theorem step_rangeBytes (t : Tally) (b n : Nat) : (step t b n).rangeBytes = t.rangeBytes := by
  unfold step freezeBucket freezeBitmap; split
  · rfl
  · split
    · rfl
    · simp only; split <;> rfl

theorem addRange_rangeBytes (t : Tally) (b l : Nat) : (addRange t b l).rangeBytes = t.rangeBytes := by
  unfold addRange; rw [addLoop_eq]
  generalize pieces b (b + (l - 1)) = ps
  induction ps generalizing t with
  | nil => rfl
  | cons p ps ih => simp only [List.foldl_cons]; rw [ih, step_rangeBytes]

/-- `absorb` charges the segment's exact wire cost to the range side. -/
theorem absorb_rangeBytes (R : Roaring) (t : Tally) (s : Seg) :
    (absorb R t s).rangeBytes = t.rangeBytes + encodedLen R s := by
  unfold absorb; rw [addRange_rangeBytes]

/-- Absorbing a list of closed segments from a fresh tally: `range_bytes` is the
sum of their encoded lengths. -/
theorem absorbAll_rangeBytes (R : Roaring) (segs : List Seg) :
    (segs.foldl (absorb R) restart).rangeBytes = (segs.map (encodedLen R)).sum := by
  suffices ∀ t : Tally, (segs.foldl (absorb R) t).rangeBytes = t.rangeBytes + (segs.map (encodedLen R)).sum by
    rw [this]; simp [restart]
  induction segs with
  | nil => intro t; simp
  | cons s ss ih => intro t; simp only [List.foldl_cons, List.map_cons, List.sum_cons]; rw [ih, absorb_rangeBytes]; omega

/-- **`encoded_len` is exact**: it is the length of the bytes `encode` writes. -/
theorem encodedLen_eq (R : Roaring) (s : Seg) (bytes : List Nat) (h : encodeSeg R s = .ok bytes) :
    bytes.length = encodedLen R s := by
  cases s <;> simp only [encodeSeg, encodeSeg.bitmap, Except.ok.injEq] at h
  all_goals first
    | (subst h; simp [writePair, le_length, encodedLen]; omega)
    | (split at h
       · cases h
       · rename_i hn; cases h; simp [le_length, encodedLen]; omega)

/-- **The collapse decision is sound**: when `prefers_bitmap` fires, the bitmap
segment `encode` writes (header + `u32` length + `serialized_size()` bytes) is
strictly shorter than the run's closed segments as ranges — given that roaring's
`serialized_size` of the run's id set is what the tally computed (`hsize`; this
is the crate's format spec, which `predicted_matches_roaring` checks in Rust). -/
theorem collapse_shrinks (R : Roaring) (t : Tally) (bm : List Nat)
    (hsize : R.size bm = bitmapBytes t) (h : prefersBitmap t = true) :
    encodedLen R (.asc bm) < t.rangeBytes ∧ encodedLen R (.dsc bm) < t.rangeBytes := by
  have := (prefersBitmap_iff t).mp h
  simp [encodedLen, hsize]; omega

/-- Below roaring's floor nothing collapses, whatever the arithmetic says. -/
theorem no_collapse_below_floor (t : Tally) (h : t.rangeBytes < 32) : prefersBitmap t = false := by
  cases hp : prefersBitmap t
  · rfl
  · have := (prefersBitmap_iff t).mp hp; omega

end IdListCost
