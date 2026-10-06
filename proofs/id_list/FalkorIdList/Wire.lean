/-
# The `IdList` wire format — byte-exact encoder and decoder

| here | there (`graph/src/effects/v3/id_list.rs` unless noted) |
| --- | --- |
| `le n v` / `ofLe`    | `to_le_bytes` of `v as uN` / `uN::from_le_bytes` (`Reader::u16/u32/u64`, `graph/src/effects/reader.rs:99-113`) |
| `widthFor`           | `narrow_int::width_for` (`graph/src/narrow_int.rs:19`) |
| `widthCode` / `widthOfCode` | `width_code` (:78) / `width_of_code` (:89) |
| `writePair`          | `Segment::write_pair` (:652) + `write_narrow` (:1282) |
| `encodeSeg`          | `Segment::encode` (:601) |
| `encodeIds`          | `impl EffectEncode<3> for IdList` (:1200) |
| `take` / `readU`     | `Reader::take` (reader.rs:40) / `Reader::u8..u64`, `read_narrow` (:1296) |
| `decodeSeg`          | `Segment::decode` (:673) |
| `readIds` / `loop`   | `read_ids` (:1337), incl. `Reader::guard_count` (reader.rs:70) |
| `Roaring`            | `RoaringTreemap::{serialize_into, serialized_size, deserialize_from}` — abstract |

Bytes are `Nat`s; the decoder theorems hold for **every** input list, with the
`< 256` hypothesis only where a `u64` read must stay `< 2^64`.

The roaring blob is abstract: `Roaring` is a structure of three functions over
the *set* (as its ascending list), and `RoaringOK` states the two things the
format relies on — a serialized set deserializes back to itself, and whatever
deserializes is a set of `u64`s. Nothing is assumed about *which* blobs are
accepted; see `REPORT.md` for what roaring 0.11.5 actually accepts.

A Rust `panic!`/arithmetic overflow is `Err.panic`; `readIds_never_panics`
proves it unreachable.
-/
import FalkorIdList.Push

namespace IdListWire

open IdListPush (Seg W flat)
open IdListPush.Seg

/-! ## Little-endian integers -/

/-- `(v as uN).to_le_bytes()` for `N = 8n`. -/
def le : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, v => v % 256 :: le n (v / 256)

/-- `uN::from_le_bytes`. -/
def ofLe : List Nat → Nat
  | [] => 0
  | b :: t => b + 256 * ofLe t

theorem le_length (n v : Nat) : (le n v).length = n := by
  induction n generalizing v <;> simp [le, *]

theorem le_bytes (n v : Nat) : ∀ b ∈ le n v, b < 256 := by
  induction n generalizing v with
  | zero => simp [le]
  | succ n ih => intro b hb; simp [le] at hb; rcases hb with rfl | hb; omega; exact ih _ b hb

theorem ofLe_le (n v : Nat) : ofLe (le n v) = v % 256 ^ n := by
  induction n generalizing v with
  | zero => simp [le, ofLe, Nat.mod_one]
  | succ n ih =>
    simp only [le, ofLe, ih, Nat.pow_succ]
    rw [Nat.mul_comm (256 ^ n) 256, Nat.mod_mul]

theorem ofLe_lt : ∀ (bs : List Nat), (∀ b ∈ bs, b < 256) → ofLe bs < 256 ^ bs.length
  | [], _ => by simp [ofLe]
  | b :: t, h => by
    have := ofLe_lt t (fun x hx => h x (by simp [hx]))
    have hb := h b (by simp)
    simp only [ofLe, List.length_cons, Nat.pow_succ]
    omega

/-! ## Widths -/

/-- `narrow_int::width_for`. -/
def widthFor (v : Nat) : Nat :=
  if v ≤ 0xFF then 1 else if v ≤ 0xFFFF then 2 else if v ≤ 0xFFFFFFFF then 4 else 8

/-- `width_code` (:78). -/
def widthCode (w : Nat) : Nat :=
  match w with
  | 1 => 0
  | 2 => 1
  | 4 => 2
  | _ => 3

/-- `width_of_code` (:89) — `code & 0b11`. -/
def widthOfCode (c : Nat) : Nat :=
  match c &&& 3 with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem widthFor_fits (v : Nat) (h : v < W) : v < 256 ^ widthFor v := by
  unfold widthFor W at *
  split <;> (try split) <;> (try split) <;> simp <;> omega

theorem widthFor_cases (v : Nat) : widthFor v = 1 ∨ widthFor v = 2 ∨ widthFor v = 4 ∨ widthFor v = 8 := by
  unfold widthFor; split <;> (try split) <;> (try split) <;> simp

theorem widthCode_lt (v : Nat) : widthCode (widthFor v) < 4 := by
  rcases widthFor_cases v with h | h | h | h <;> rw [h] <;> decide

theorem widthOfCode_widthCode (v : Nat) : ∀ c, c &&& 3 = widthCode (widthFor v) →
    widthOfCode c = widthFor v := by
  intro c hc
  unfold widthOfCode; rw [hc]
  rcases widthFor_cases v with h | h | h | h <;> rw [h] <;> rfl

theorem widthFor_ge (v : Nat) : 1 ≤ widthFor v := by
  rcases widthFor_cases v with h | h | h | h <;> omega

/-! ## The header byte (:534-541) -/

def SEG_KIND_MASK : Nat := 0b0000_0011
def SEG_KIND_RANGE : Nat := 0
def SEG_KIND_ASCENDING : Nat := 1
def SEG_KIND_REPEAT : Nat := 2
def SEG_DESCENDING : Nat := 0b0100_0000
def SEG_RESERVED : Nat := 0b1000_0000

/-- What `write_pair` puts in the header: `kind | (vc << 2) | (cc << 4)`, where
`kind` already carries `SEG_DESCENDING` for a `RangeDescending`. -/
def pairHeader (kind vc cc : Nat) : Nat := kind ||| (vc <<< 2) ||| (cc <<< 4)

/-- The decoder's view of every header this encoder can write — by exhaustion
over all 2 × 3 × 4 × 4 = 96 pair headers (`Repeat` never carries the bit). -/
theorem pairHeader_fields : ∀ k ∈ [0, 0x40, 2], ∀ vc < 4, ∀ cc < 4,
    let h := pairHeader k vc cc
    h &&& SEG_RESERVED = 0 ∧ (h &&& SEG_DESCENDING ≠ 0 ↔ k = 0x40) ∧
    h &&& SEG_KIND_MASK = k &&& 3 ∧ (h >>> 2) &&& 3 = vc ∧ (h >>> 4) &&& 3 = cc ∧ h < 256 := by
  decide

theorem ascHeader_fields : ∀ d ∈ [0, 0x40],
    let h := SEG_KIND_ASCENDING ||| d
    h &&& SEG_RESERVED = 0 ∧ (h &&& SEG_DESCENDING ≠ 0 ↔ d = 0x40) ∧ h &&& SEG_KIND_MASK = 1 ∧ h < 256 := by
  decide

/-! ## Encoder -/

inductive Err where
  | eof | badEncoding | badRange | cardinality | implausible | badRoaring | lengthLied | panic
  deriving DecidableEq, Repr

/-- `RoaringTreemap`'s codec, over the set as its ascending list. -/
structure Roaring where
  ser : List Nat → List Nat
  size : List Nat → Nat
  deser : List Nat → Option (List Nat)

/-- What the format needs from roaring. -/
structure RoaringOK (R : Roaring) : Prop where
  roundtrip : ∀ s, s ≠ [] → s.Pairwise (· < ·) → (∀ x ∈ s, x < W) → R.deser (R.ser s) = some s
  size_eq : ∀ s, R.size s = (R.ser s).length
  size_lt : ∀ s, (∀ x ∈ s, x < W) → R.size s < 4294967296
  bytes : ∀ s, ∀ b ∈ R.ser s, b < 256
  /-- Anything that deserializes is a set of `u64`s. -/
  deser_set : ∀ blob s, R.deser blob = some s → s.Pairwise (· < ·) ∧ ∀ x ∈ s, x < W

/-- `Segment::write_pair` (:652). -/
def writePair (kind value count : Nat) : List Nat :=
  let vw := widthFor value
  let cw := widthFor count
  pairHeader kind (widthCode vw) (widthCode cw) :: (le vw value ++ le cw count)

/-- `Segment::encode` (:601). -/
def encodeSeg (R : Roaring) : Seg → Except Err (List Nat)
  | .range b l => .ok (writePair SEG_KIND_RANGE b l)
  | .rdesc b l => .ok (writePair (SEG_KIND_RANGE ||| SEG_DESCENDING) b l)
  | .rep i c => .ok (writePair SEG_KIND_REPEAT i c)
  | .asc bm => bitmap 0 bm
  | .dsc bm => bitmap SEG_DESCENDING bm
where
  bitmap (d : Nat) (bm : List Nat) : Except Err (List Nat) :=
    let n := R.size bm
    let blob := R.ser bm
    -- `EncodeError::BitmapLengthLied` (:622)
    if blob.length ≠ n then .error .lengthLied
    else .ok ((SEG_KIND_ASCENDING ||| d) :: (le 4 n ++ blob))

def encodeSegs (R : Roaring) : List Seg → Except Err (List Nat)
  | [] => .ok []
  | s :: t => do
    let a ← encodeSeg R s
    let b ← encodeSegs R t
    pure (a ++ b)

/-- `EffectEncode<3> for IdList` (:1200): `u32 n_segments` (`as u32`), then each. -/
def encodeIds (R : Roaring) (segs : List Seg) : Except Err (List Nat) := do
  let body ← encodeSegs R segs
  pure (le 4 segs.length ++ body)

/-! ## Decoder -/

/-- `Reader::take` (reader.rs:40): refuses rather than over-reads. -/
def take (n : Nat) (bs : List Nat) : Except Err (List Nat × List Nat) :=
  if bs.length < n then .error .eof else .ok (bs.take n, bs.drop n)

/-- `Reader::u8/u16/u32/u64` and `read_narrow` (:1296). -/
def readU (n : Nat) (bs : List Nat) : Except Err (Nat × List Nat) :=
  match take n bs with
  | .error e => .error e
  | .ok (b, r) => .ok (ofLe b, r)

/-- `Segment::decode` (:673). `remaining` is how many ids the record still owes. -/
def decodeSeg (R : Roaring) (bs : List Nat) (remaining : Nat) : Except Err (Seg × List Nat) :=
  match bs with
  | [] => .error .eof
  | h :: bs =>
    if h &&& SEG_RESERVED ≠ 0 then .error .badEncoding else
    if h &&& SEG_KIND_MASK = SEG_KIND_ASCENDING then
      match readU 4 bs with
      | .error e => .error e
      | .ok (blobLen, bs) =>
        match take blobLen bs with
        | .error e => .error e
        | .ok (blob, bs) =>
          match R.deser blob with
          | none => .error .badRoaring
          | some s =>
            if s.length = 0 ∨ s.length > remaining then .error .cardinality
            else .ok (if h &&& SEG_DESCENDING ≠ 0 then .dsc s else .asc s, bs)
    else
      match readU (widthOfCode (h >>> 2)) bs with
      | .error e => .error e
      | .ok (value, bs) =>
        match readU (widthOfCode (h >>> 4)) bs with
        | .error e => .error e
        | .ok (count, bs) =>
          if count = 0 ∨ count > remaining then .error .badRange else
          -- `count as u32` is `count % 2^32`
          if h &&& SEG_KIND_MASK = SEG_KIND_RANGE then
            if h &&& SEG_DESCENDING ≠ 0 then
              -- `value.checked_sub(count - 1)`
              if value < count - 1 then .error .badRange else .ok (.rdesc value (count % 4294967296), bs)
            else
              -- `value.checked_add(count - 1)` on a `u64`
              if value + (count - 1) ≥ W then .error .badRange else .ok (.range value (count % 4294967296), bs)
          else if h &&& SEG_KIND_MASK = SEG_KIND_REPEAT then
            if h &&& SEG_DESCENDING ≠ 0 then .error .badEncoding else .ok (.rep value (count % 4294967296), bs)
          else .error .badEncoding

/-- The `for _ in 0..n_segments` loop of `read_ids` (:1357). `len` is `u64`;
`u64::from(count) - len` is a panic in debug if it underflows. -/
def loop (R : Roaring) (count : Nat) : Nat → Nat → List Seg → List Nat → Except Err (List Seg × Nat × List Nat)
  | 0, len, acc, bs => .ok (acc, len, bs)
  | k + 1, len, acc, bs =>
    if len > count then .error .panic else
    match decodeSeg R bs (count - len) with
    | .error e => .error e
    | .ok (seg, bs) => loop R count k (len + seg.len) (acc ++ [seg]) bs

/-- `read_ids` (:1337). -/
def readIds (R : Roaring) (bs : List Nat) (count : Nat) : Except Err (List Seg × List Nat) :=
  match readU 4 bs with
  | .error e => .error e
  | .ok (nSeg, bs) =>
    if nSeg > count then .error .implausible else
    -- `guard_count(n, MIN_SEGMENT_BYTES)`: `saturating_mul` then compare
    if min (nSeg * 3) (2 ^ 64 - 1) > bs.length then .error .implausible else
    match loop R count nSeg 0 [] bs with
    | .error e => .error e
    | .ok (segs, len, bs) =>
      if len ≠ count then .error .cardinality else .ok (segs, bs)

/-! ## Round trip -/

theorem take_append (a rest : List Nat) : take a.length (a ++ rest) = .ok (a, rest) := by
  simp [take]

theorem readU_le (n v : Nat) (rest : List Nat) (hv : v < 256 ^ n) :
    readU n (le n v ++ rest) = .ok (v, rest) := by
  unfold readU
  have := take_append (le n v) rest
  rw [le_length] at this
  rw [this]; simp [ofLe_le, Nat.mod_eq_of_lt hv]

/-- A well-formed segment of at most `remaining` ids that fits the wire. -/
def Encodable (s : Seg) (remaining : Nat) : Prop :=
  Seg.WF s ∧ s.len ≤ remaining ∧ remaining < 4294967296

theorem decode_writePair (R : Roaring) (k v c rest remaining) (hk : k ∈ [0, 0x40, 2])
    (hv : v < W) (hc : c < W) :
    decodeSeg R (writePair k v c ++ rest) remaining =
      (if c = 0 ∨ c > remaining then .error .badRange else
       let c32 := c % 4294967296
       if k &&& 3 = SEG_KIND_RANGE then
         if k = 0x40 then (if v < c - 1 then .error .badRange else .ok (.rdesc v c32, rest))
         else (if v + (c - 1) ≥ W then .error .badRange else .ok (.range v c32, rest))
       else if k &&& 3 = SEG_KIND_REPEAT then
         if k = 0x40 then .error .badEncoding else .ok (.rep v c32, rest)
       else .error .badEncoding) := by
  obtain ⟨h1, h2, h3, h4, h5, _⟩ :=
    pairHeader_fields k hk (widthCode (widthFor v)) (widthCode_lt v) (widthCode (widthFor c)) (widthCode_lt c)
  have hk1 : k &&& 3 ≠ 1 := by simp at hk; rcases hk with rfl | rfl | rfl <;> decide
  simp only [SEG_RESERVED, SEG_DESCENDING, SEG_KIND_MASK, SEG_KIND_ASCENDING, ne_eq] at h1 h2 h3
  simp only [writePair, decodeSeg, List.cons_append, SEG_RESERVED, SEG_DESCENDING, SEG_KIND_MASK,
    SEG_KIND_ASCENDING, h1, h3, hk1, ne_eq, not_true_eq_false, if_false]
  rw [widthOfCode_widthCode v _ h4, List.append_assoc, readU_le _ _ _ (widthFor_fits v hv)]
  simp only
  rw [widthOfCode_widthCode c _ h5, readU_le _ _ _ (widthFor_fits c hc)]
  simp only [ne_eq, h2]

theorem decode_encodeSeg (R : Roaring) (hR : RoaringOK R) (s : Seg) (remaining : Nat)
    (he : Encodable s remaining) (rest : List Nat) :
    ∃ bytes, encodeSeg R s = .ok bytes ∧ decodeSeg R (bytes ++ rest) remaining = .ok (s, rest) := by
  obtain ⟨hw, hl, hr⟩ := he
  cases s with
  | range b l =>
    simp [Seg.WF] at hw; simp [Seg.len] at hl
    refine ⟨_, rfl, ?_⟩
    rw [decode_writePair R _ _ _ _ _ (by simp [SEG_KIND_RANGE]) (by unfold W at *; omega) (by unfold W at *; omega)]
    simp [SEG_KIND_RANGE]
    repeat (rw [if_neg (by unfold W at *; omega)])
    rw [Nat.mod_eq_of_lt (by omega)]
  | rdesc b l =>
    simp [Seg.WF] at hw; simp [Seg.len] at hl
    refine ⟨_, rfl, ?_⟩
    rw [decode_writePair R _ _ _ _ _ (by simp [SEG_KIND_RANGE, SEG_DESCENDING]) (by omega) (by unfold W at *; omega)]
    simp [SEG_KIND_RANGE, SEG_DESCENDING]
    repeat (rw [if_neg (by unfold W at *; omega)])
    rw [Nat.mod_eq_of_lt (by omega)]
  | rep i c =>
    simp [Seg.WF] at hw; simp [Seg.len] at hl
    refine ⟨_, rfl, ?_⟩
    rw [decode_writePair R _ _ _ _ _ (by simp [SEG_KIND_REPEAT]) (by omega) (by unfold W at *; omega)]
    simp [SEG_KIND_REPEAT, SEG_KIND_RANGE]
    repeat (rw [if_neg (by unfold W at *; omega)])
    rw [Nat.mod_eq_of_lt (by omega)]
  | asc bm =>
    simp [Seg.WF] at hw; simp [Seg.len] at hl
    obtain ⟨hne, hs, hlt⟩ := hw
    have hsz := hR.size_eq bm
    have hszl := hR.size_lt bm hlt
    refine ⟨_, by simp [encodeSeg, encodeSeg.bitmap, hsz]; rfl, ?_⟩
    have hsz' : (R.ser bm).length < 256 ^ 4 := by rw [← hsz]; exact hszl
    have hlen : ¬ (bm.length = 0 ∨ bm.length > remaining) := by
      have := List.length_pos_iff.2 hne; omega
    simp only [decodeSeg, List.cons_append, SEG_RESERVED, SEG_DESCENDING, SEG_KIND_MASK,
      SEG_KIND_ASCENDING, Nat.reduceAnd, Nat.reduceOr]
    rw [List.append_assoc, readU_le _ _ _ hsz']
    simp [take_append, hR.roundtrip bm hne hs hlt, hlen]
    exact ⟨hne, hl⟩
  | dsc bm =>
    simp [Seg.WF] at hw; simp [Seg.len] at hl
    obtain ⟨hne, hs, hlt⟩ := hw
    have hsz := hR.size_eq bm
    have hszl := hR.size_lt bm hlt
    refine ⟨_, by simp [encodeSeg, encodeSeg.bitmap, hsz]; rfl, ?_⟩
    have hsz' : (R.ser bm).length < 256 ^ 4 := by rw [← hsz]; exact hszl
    have hlen : ¬ (bm.length = 0 ∨ bm.length > remaining) := by
      have := List.length_pos_iff.2 hne; omega
    simp only [decodeSeg, List.cons_append, SEG_RESERVED, SEG_DESCENDING, SEG_KIND_MASK,
      SEG_KIND_ASCENDING, Nat.reduceAnd, Nat.reduceOr]
    rw [List.append_assoc, readU_le _ _ _ hsz']
    simp [take_append, hR.roundtrip bm hne hs hlt, hlen]
    exact ⟨hne, hl⟩

theorem encodeSeg_len (R : Roaring) (s : Seg) (bytes : List Nat) (h : encodeSeg R s = .ok bytes) :
    3 ≤ bytes.length := by
  cases s with
  | range b l | rdesc b l | rep b l =>
    simp [encodeSeg] at h; subst h
    simp [writePair, le_length]; have := widthFor_ge b; have := widthFor_ge l; omega
  | asc bm | dsc bm =>
    simp only [encodeSeg, encodeSeg.bitmap] at h
    split at h
    · cases h
    · cases h; simp [le_length]; omega

theorem encodeSeg_exists (R : Roaring) (hR : RoaringOK R) (s : Seg) : ∃ bytes, encodeSeg R s = .ok bytes := by
  cases s with
  | range b l | rdesc b l | rep b l => exact ⟨_, rfl⟩
  | asc bm =>
    exact ⟨(SEG_KIND_ASCENDING ||| 0) :: (le 4 (R.size bm) ++ R.ser bm), by
      simp only [encodeSeg, encodeSeg.bitmap]; rw [if_neg (by rw [hR.size_eq]; simp)]⟩
  | dsc bm =>
    exact ⟨(SEG_KIND_ASCENDING ||| SEG_DESCENDING) :: (le 4 (R.size bm) ++ R.ser bm), by
      simp only [encodeSeg, encodeSeg.bitmap]; rw [if_neg (by rw [hR.size_eq]; simp)]⟩

def sumLen (segs : List Seg) : Nat := (segs.map Seg.len).sum

theorem flat_length {segs : List Seg} (hw : ∀ s ∈ segs, Seg.WF s) : (flat segs).length = sumLen segs := by
  induction segs with
  | nil => simp [flat, sumLen]
  | cons s t ih =>
    have := IdListPush.iter_length (hw s (by simp))
    have ih' := ih (fun x hx => hw x (by simp [hx]))
    simp only [flat, List.flatMap_cons, List.length_append, sumLen, List.map_cons, List.sum_cons] at *
    omega

theorem encodeSegs_loop (R : Roaring) (hR : RoaringOK R) (count : Nat) (hc : count < 4294967296) :
    ∀ (segs : List Seg), (∀ s ∈ segs, Seg.WF s) →
    ∃ body, encodeSegs R segs = .ok body ∧ 3 * segs.length ≤ body.length ∧
      ∀ len acc rest, len + sumLen segs = count →
        loop R count segs.length len acc (body ++ rest) = .ok (acc ++ segs, count, rest)
  | [], _ => ⟨[], rfl, by simp, fun len acc rest h => by simp [sumLen] at h; simp [loop, h]⟩
  | s :: t, hw => by
    obtain ⟨body, hb, hbl, hloop⟩ := encodeSegs_loop R hR count hc t (fun x hx => hw x (by simp [hx]))
    have hws := hw s (by simp)
    -- the segment encodes; its bytes decode back whatever follows
    have hex : ∀ rem, s.len ≤ rem → rem < 4294967296 → ∀ rest,
        ∃ bytes, encodeSeg R s = .ok bytes ∧ decodeSeg R (bytes ++ rest) rem = .ok (s, rest) :=
      fun rem h1 h2 rest => decode_encodeSeg R hR s rem ⟨hws, h1, h2⟩ rest
    obtain ⟨bytes, hbytes⟩ := encodeSeg_exists R hR s
    refine ⟨bytes ++ body, by simp [encodeSegs, hbytes, hb]; rfl, ?_, ?_⟩
    · have := encodeSeg_len R s bytes hbytes; simp; omega
    · intro len acc rest hlen
      simp only [sumLen, List.map_cons, List.sum_cons] at hlen
      have hle : ¬ len > count := by omega
      obtain ⟨bytes', hb', hdec⟩ := hex (count - len) (by omega) (by omega) (body ++ rest)
      rw [hbytes] at hb'; cases hb'
      simp only [List.length_cons, loop, hle, if_false, List.append_assoc, hdec]
      rw [hloop (len + s.len) (acc ++ [s]) rest (by simp only [sumLen] at *; omega)]
      simp

/-- **Round trip.** Whatever the builder produced — well-formed segments,
at most `count` of them, totalling `count < 2^32` ids — `read_ids` reads back
*exactly those segments* and stops exactly where the list ends. Hence
decode∘encode is the identity on segments, and re-encoding a decoded list
is byte-identical (the property `read_ids`' doc comment rests on). -/
theorem readIds_encodeIds (R : Roaring) (hR : RoaringOK R) (segs : List Seg)
    (hw : ∀ s ∈ segs, Seg.WF s) (count : Nat) (hsum : sumLen segs = count)
    (hc : count < 4294967296) (hn : segs.length ≤ count) (rest : List Nat) :
    ∃ bytes, encodeIds R segs = .ok bytes ∧ readIds R (bytes ++ rest) count = .ok (segs, rest) := by
  obtain ⟨body, hb, hbl, hloop⟩ := encodeSegs_loop R hR count hc segs hw
  refine ⟨le 4 segs.length ++ body, by simp [encodeIds, hb]; rfl, ?_⟩
  simp only [readIds, List.append_assoc]
  rw [readU_le _ _ _ (by show segs.length < 256 ^ 4; simp; omega)]
  simp only
  rw [if_neg (by omega), if_neg (by simp; omega)]
  rw [hloop 0 [] rest (by omega)]
  simp

/-! ## The whole pipeline -/

/-- **`decode(encode(push* xs)) = xs`** for every id sequence the encoder
accepts: any `u64`s, sorted or not, with duplicates, `0` and `u64::MAX`,
empty or not, as long as the record count fits its `u32` — and whatever the
collapse arithmetic decides at each push. -/
theorem push_encode_decode (R : Roaring) (hR : RoaringOK R) (cs : List Bool) (xs : List Nat)
    (hx : ∀ x ∈ xs, x < W) (hlen : xs.length < 4294967296) (rest : List Nat) :
    ∃ st bytes segs, IdListPush.pushAll cs IdListPush.St.empty xs = some st ∧
      encodeIds R st.segs = .ok bytes ∧
      readIds R (bytes ++ rest) xs.length = .ok (segs, rest) ∧
      segs = st.segs ∧ flat segs = xs := by
  obtain ⟨st, h1, h2, h3, _⟩ := IdListPush.fromIter_iter cs xs hx
  have hw := h2.wf
  have hsum : sumLen st.segs = xs.length := by rw [← flat_length hw, h3]
  have hn : st.segs.length ≤ xs.length := by
    have := IdListPush.segs_le_len hw; rw [h3] at this; exact this
  obtain ⟨bytes, he, hd⟩ := readIds_encodeIds R hR st.segs hw xs.length hsum hlen hn rest
  exact ⟨st, bytes, st.segs, h1, he, hd, rfl, h3⟩

/-! ## Decoder safety on arbitrary bytes -/

theorem take_ok {n : Nat} {bs a rest : List Nat} (h : take n bs = .ok (a, rest)) :
    bs = a ++ rest ∧ a.length = n := by
  unfold take at h
  split at h
  · cases h
  · rename_i hlt; cases h; exact ⟨(List.take_append_drop n bs).symm, by simp; omega⟩

theorem readU_ok {n : Nat} {bs rest : List Nat} {v : Nat} (h : readU n bs = .ok (v, rest)) :
    ∃ pre, bs = pre ++ rest ∧ pre.length = n ∧ v = ofLe pre := by
  unfold readU at h
  split at h
  · cases h
  · rename_i b r hb; cases h; obtain ⟨h1, h2⟩ := take_ok hb; exact ⟨b, h1, h2, rfl⟩

theorem readU_lt {n : Nat} {bs rest : List Nat} {v : Nat} (h : readU n bs = .ok (v, rest))
    (hb : ∀ b ∈ bs, b < 256) : v < 256 ^ n := by
  obtain ⟨pre, rfl, hl, rfl⟩ := readU_ok h
  have := ofLe_lt pre (fun b h => hb b (by simp [h])); rwa [hl] at this

theorem widthFor_code_ge (c : Nat) : 1 ≤ widthOfCode c := by
  unfold widthOfCode; split <;> decide

theorem widthOfCode_le (c : Nat) : 256 ^ widthOfCode c ≤ W := by
  unfold widthOfCode W
  split <;> decide

@[simp] theorem take_ne_panic (n : Nat) (bs : List Nat) : take n bs ≠ .error .panic := by
  unfold take; split <;> simp

@[simp] theorem readU_ne_panic (n : Nat) (bs : List Nat) : readU n bs ≠ .error .panic := by
  unfold readU; split
  · rename_i e he; intro h; cases h; exact take_ne_panic n bs he
  · simp

theorem decodeSeg_never_panics (R : Roaring) (bs : List Nat) (rem : Nat) :
    decodeSeg R bs rem ≠ .error .panic := by
  unfold decodeSeg
  repeat' split
  all_goals (intro h; simp_all)

/-- Every accepted segment fits what the record still owes (`len > remaining`
is refused on every path; `count as u32` can only shrink it). -/
theorem decodeSeg_len {R : Roaring} {bs rest : List Nat} {rem : Nat} {s : Seg}
    (h : decodeSeg R bs rem = .ok (s, rest)) : s.len ≤ rem := by
  unfold decodeSeg at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; simp [Seg.len]; omega)
    | (cases h; simp [Seg.len]; have := Nat.mod_le ‹Nat› 4294967296; omega)

theorem decodeSeg_ok (R : Roaring) (hR : RoaringOK R) {bs rest : List Nat} {rem : Nat} {s : Seg}
    (h : decodeSeg R bs rem = .ok (s, rest)) (hb : ∀ b ∈ bs, b < 256) (hrem : rem < 4294967296) :
    Seg.WF s ∧ 1 ≤ s.len ∧ s.len ≤ rem ∧ ∃ pre, bs = pre ++ rest ∧ 3 ≤ pre.length := by
  unfold decodeSeg at h
  split at h
  · cases h
  rename_i hd tl
  have hbt : ∀ b ∈ tl, b < 256 := fun b hm => hb b (by simp [hm])
  split at h
  · cases h
  split at h
  · -- bitmap
    split at h
    · cases h
    rename_i blobLen bs1 h1
    obtain ⟨p1, rfl, hp1, _⟩ := readU_ok h1
    split at h
    · cases h
    rename_i blob bs2 h2
    obtain ⟨rfl, _⟩ := take_ok h2
    split at h
    · cases h
    rename_i set hset
    obtain ⟨hsorted, hlt⟩ := hR.deser_set _ _ hset
    split at h
    · cases h
    rename_i hlen
    cases h
    have hne : set ≠ [] := by intro e; subst e; simp at hlen
    refine ⟨?_, ?_, ?_, ⟨hd :: (p1 ++ blob), by simp, by simp; omega⟩⟩
    · split <;> exact ⟨hne, hsorted, hlt⟩
    · split <;> simp [Seg.len] <;> omega
    · split <;> simp [Seg.len] <;> omega
  · -- range / repeat
    split at h
    · cases h
    rename_i value bs1 h1
    have hv := readU_lt h1 hbt
    have hvW := Nat.lt_of_lt_of_le hv (widthOfCode_le _)
    obtain ⟨p1, rfl, hp1, _⟩ := readU_ok h1
    split at h
    · cases h
    rename_i count bs2 h2
    obtain ⟨p2, rfl, hp2, _⟩ := readU_ok h2
    split at h
    · cases h
    rename_i hcnt
    have hc32 : count % 4294967296 = count := Nat.mod_eq_of_lt (by omega)
    have hw1 := widthFor_code_ge (hd >>> 2)
    have hw2 := widthFor_code_ge (hd >>> 4)
    have hpre : ∃ pre, hd :: (p1 ++ (p2 ++ rest)) = pre ++ rest ∧ 3 ≤ pre.length :=
      ⟨hd :: (p1 ++ p2), by simp, by simp; omega⟩
    split at h
    · split at h
      · split at h
        · cases h
        · cases h; rw [hc32]
          exact ⟨⟨by omega, by omega, hvW⟩, by simp [Seg.len]; omega, by simp [Seg.len]; omega, hpre⟩
      · split at h
        · cases h
        · cases h; rw [hc32]
          exact ⟨⟨by omega, by unfold W at *; omega⟩, by simp [Seg.len]; omega, by simp [Seg.len]; omega, hpre⟩
    · split at h
      · split at h
        · cases h
        · cases h; rw [hc32]
          exact ⟨⟨by omega, hvW⟩, by simp [Seg.len]; omega, by simp [Seg.len]; omega, hpre⟩
      · cases h

theorem loop_safe (R : Roaring) (hR : RoaringOK R) (count : Nat) (hc : count < 4294967296) :
    ∀ (k len : Nat) (acc : List Seg) (bs : List Nat) (segs : List Seg) (len' : Nat) (rest : List Nat),
    len ≤ count → (∀ b ∈ bs, b < 256) →
    loop R count k len acc bs = .ok (segs, len', rest) →
    ∃ new, segs = acc ++ new ∧ new.length = k ∧ (∀ s ∈ new, Seg.WF s) ∧
      len' = len + sumLen new ∧ len' ≤ count ∧ ∃ pre, bs = pre ++ rest ∧ 3 * k ≤ pre.length
  | 0, len, acc, bs, segs, len', rest, hle, _, h => by
    simp [loop] at h; obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨[], by simp, rfl, by simp, by simp [sumLen], hle, [], by simp, by simp⟩
  | k + 1, len, acc, bs, segs, len', rest, hle, hb, h => by
    simp only [loop, show ¬ len > count by omega, if_false] at h
    split at h
    · cases h
    rename_i seg bs1 hdec
    obtain ⟨hw, h1, h2, pre1, rfl, hp1⟩ := decodeSeg_ok R hR hdec hb (by omega)
    obtain ⟨new, hnew, hlen, hwn, hsum, hle', pre2, hpre2, hp2⟩ :=
      loop_safe R hR count hc k (len + seg.len) (acc ++ [seg]) bs1 segs len' rest (by omega)
        (fun b hm => hb b (by simp [hm])) h
    refine ⟨seg :: new, by simp [hnew], by simp [hlen], ?_, ?_, hle', pre1 ++ pre2, by simp [hpre2],
      by simp; omega⟩
    · intro s hs; simp at hs; rcases hs with rfl | hs
      · exact hw
      · exact hwn s hs
    · simp only [sumLen, List.map_cons, List.sum_cons] at hsum ⊢; omega

theorem loop_never_panics (R : Roaring) (count : Nat) :
    ∀ (k len : Nat) (acc : List Seg) (bs : List Nat), len ≤ count →
    loop R count k len acc bs ≠ .error .panic
  | 0, _, _, _, _ => by simp [loop]
  | k + 1, len, acc, bs, hle => by
    simp only [loop, show ¬ len > count by omega, if_false]
    split
    · rename_i e he; intro h; cases h; exact decodeSeg_never_panics R bs _ he
    · rename_i seg bs1 hdec
      have := decodeSeg_len hdec
      exact loop_never_panics R count k _ _ _ (by omega)

/-- **The debug-build `u64` subtraction in `read_ids` never underflows**: no
input, however malformed, reaches a panic. -/
theorem readIds_never_panics (R : Roaring) (bs : List Nat) (count : Nat) :
    readIds R bs count ≠ .error .panic := by
  unfold readIds
  split
  · rename_i e he; intro h; cases h; exact readU_ne_panic _ _ he
  · split
    · intro h; cases h
    · split
      · intro h; cases h
      · split
        · rename_i e he; intro h; cases h; exact loop_never_panics R count _ 0 [] _ (Nat.zero_le _) he
        · split <;> (intro h; cases h)

/-- **Decoder totality and soundness.** On any bytes, `read_ids` either
refuses, or returns segments that (a) are each well-formed — so iterating them
cannot panic or wrap, (b) total exactly `count` ids, (c) number at most `count`,
and (d) were read from a prefix of the input of at least `4 + 3·segments`
bytes: it never over-reads, and its work is bounded by what it consumed. -/
theorem readIds_safe (R : Roaring) (hR : RoaringOK R) (bs : List Nat) (count : Nat)
    (hb : ∀ b ∈ bs, b < 256) (hc : count < 4294967296) {segs : List Seg} {rest : List Nat}
    (h : readIds R bs count = .ok (segs, rest)) :
    (∀ s ∈ segs, Seg.WF s) ∧ (flat segs).length = count ∧ segs.length ≤ count ∧
      ∃ pre, bs = pre ++ rest ∧ 4 + 3 * segs.length ≤ pre.length := by
  unfold readIds at h
  split at h
  · cases h
  rename_i nSeg bs1 h1
  obtain ⟨p1, rfl, hp1, _⟩ := readU_ok h1
  split at h
  · cases h
  rename_i hn
  split at h
  · cases h
  split at h
  · cases h
  rename_i segs' len bs2 hloop
  split at h
  · cases h
  rename_i hlen
  cases h
  obtain ⟨new, hnew, hnl, hw, hsum, _, pre, hpre, hpl⟩ :=
    loop_safe R hR count hc nSeg 0 [] bs1 segs len rest (Nat.zero_le _)
      (fun b hm => hb b (by simp [hm])) hloop
  simp at hnew; subst hnew
  refine ⟨hw, by rw [flat_length hw]; omega, by omega, p1 ++ pre, by simp [hpre], ?_⟩
  simp only [List.length_append, hp1]; omega

/-! ## The decoder is not canonical

Every one of these decodes to the same ids as the canonical encoding and is a
different byte string, so encode∘decode is not the identity on byte strings
(decode∘encode is — `readIds_encodeIds`). Reproduced against the Rust decoder in
`graph/tests/lean_id_list.rs`. -/

/-- A dummy codec: these examples never reach the bitmap path. -/
def noRoaring : Roaring := ⟨fun _ => [], fun _ => 0, fun _ => none⟩

/-- `Range{5,1}` canonically: `n=1`, header `0x00`, base `05`, len `01`. -/
theorem canonical_range : encodeIds noRoaring [.range 5 1] = .ok [1, 0, 0, 0, 0x00, 5, 1] := rfl

/-- The same segment with its base at 8 bytes (header `0x0c`) is accepted.
Rust: `noncanonical_wide_range_is_accepted`. -/
theorem nonminimal_width_accepted :
    readIds noRoaring [1, 0, 0, 0, 0x0c, 5, 0, 0, 0, 0, 0, 0, 0, 1] 1 = .ok ([.range 5 1], []) := rfl

/-- A lone id marked descending is accepted (no encoder writes one). -/
theorem lone_descending_accepted :
    readIds noRoaring [1, 0, 0, 0, 0x40, 5, 1] 1 = .ok ([.rdesc 5 1], []) := rfl

/-- The refusals the doc comments promise, as concrete cases. -/
theorem refuses_reserved_bit : readIds noRoaring [1, 0, 0, 0, 0x80, 5, 1] 1 = .error .badEncoding := rfl
theorem refuses_descending_repeat : readIds noRoaring [1, 0, 0, 0, 0x42, 5, 2] 2 = .error .badEncoding := rfl
theorem refuses_kind3 : readIds noRoaring [1, 0, 0, 0, 0x03, 5, 1] 1 = .error .badEncoding := rfl
theorem refuses_wrap :
    readIds noRoaring ([1, 0, 0, 0, 0x0c] ++ le 8 (W - 1) ++ [2]) 2 = .error .badRange := rfl
theorem refuses_below_zero : readIds noRoaring [1, 0, 0, 0, 0x40, 0, 2] 2 = .error .badRange := rfl
theorem refuses_short_count : readIds noRoaring [1, 0, 0, 0, 0x00, 5, 2] 1 = .error .badRange := rfl
theorem refuses_long_count : readIds noRoaring [1, 0, 0, 0, 0x00, 5, 1] 2 = .error .cardinality := rfl
theorem refuses_truncated : readIds noRoaring [1, 0, 0, 0, 0x0c, 5, 0] 1 = .error .eof := rfl
theorem refuses_implausible_bytes : readIds noRoaring [1, 0, 0, 0, 0x00, 5] 1 = .error .implausible := rfl
theorem refuses_too_many_segments : readIds noRoaring [2, 0, 0, 0, 0x00, 5, 1] 1 = .error .implausible := rfl

end IdListWire
