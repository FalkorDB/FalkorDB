import FalkorEffectsCodec.Value
/-!
# `Reader` as the Rust has it (`buf`, `pos`), and `EffectWrite` as a trait

`Prim.lean` models a reader action as a function on the unconsumed suffix. Here
the actual cursor — `struct Reader { buf, pos }` (`graph/src/effects/reader.rs`)
— is modelled field for field, and every method is proven to *refine* the
suffix model: running it on `⟨buf, pos⟩` does what the suffix action does on
`buf.drop pos` (`rest`). So every theorem stated over `Prim.R` holds of the
real cursor.
-/
namespace FalkorCodec

structure Rd where
  buf : Bytes
  pos : Nat

/-- `Reader::new` (`reader.rs:18`). -/
def Rd.new (buf : Bytes) : Rd := ⟨buf, 0⟩
/-- `Reader::remaining` (`:24`). -/
def Rd.remaining (r : Rd) : Nat := r.buf.length - r.pos
/-- `Reader::is_empty` (`:30`). -/
def Rd.isEmpty (r : Rd) : Bool := r.remaining == 0
/-- `Reader::rest` (`:36`): `&self.buf[self.pos..]`. -/
def Rd.rest (r : Rd) : Bytes := r.buf.drop r.pos

/-- `Reader::take` (`:42`). The slice `buf[pos..pos+n]` is reached only after
`remaining() >= n`, so it cannot panic. -/
def Rd.take (r : Rd) (n : Nat) : Except DErr (Bytes × Rd) :=
  if r.remaining < n then .error (.eof n r.remaining)
  else .ok ((r.buf.drop r.pos).take n, { r with pos := r.pos + n })

/-- `Reader::guard_count` (`:70`): `count.saturating_mul(min as u64) > remaining as u64`. -/
def Rd.guardCount (r : Rd) (count minEach : Nat) : Except DErr Nat :=
  let need := min (count * minEach) (2 ^ 64 - 1)
  if need > r.remaining then .error (.implausible count r.remaining) else .ok count

/-- `take_array::<N>` (`:93`): `copy_from_slice(take(N)?)`. -/
def Rd.takeArray (r : Rd) (n : Nat) : Except DErr (Bytes × Rd) := r.take n

/-- `Reader::i64` (`:115`), `f32` (`:119`), `f64` (`:123`): `from_le_bytes` of the
array, i.e. the little-endian bit pattern (`f32/f64::from_le_bytes` is a bit
transmute; NaN payloads and `-0.0` are preserved as bits). -/
def Rd.bits (r : Rd) (k : Nat) : Except DErr (BitVec (8 * k) × Rd) :=
  match r.takeArray k with
  | .error e => .error e
  | .ok (a, r') => .ok (BitVec.ofNat (8 * k) (unle a), r')
def Rd.i64 (r : Rd) := r.bits 8
def Rd.f32 (r : Rd) := r.bits 4
def Rd.f64 (r : Rd) := r.bits 8

/-- `chunks_exact(W)` over a buffer of `n * W` bytes. -/
def chunks (k : Nat) : Nat → Bytes → List Bytes
  | 0, _ => []
  | n + 1, bs => bs.take k :: chunks k n (bs.drop k)

/-- `Reader::take_n` (`:138`): guard, take `n*W` in one go, then `chunks_exact`. -/
def Rd.takeN {α} (r : Rd) (k count : Nat) (fromLe : Bytes → α) : Except DErr (List α × Rd) :=
  match r.guardCount count k with
  | .error e => .error e
  | .ok n =>
    match r.take (n * k) with
    | .error e => .error e
    | .ok (bytes, r') => .ok ((chunks k n bytes).map fromLe, r')

/-! ### Refinement: the cursor does what the suffix model does -/

@[simp] theorem exc_map_ok {ε α β} (f : α → β) (a : α) : (Except.ok a : Except ε α).map f = .ok (f a) := rfl
@[simp] theorem exc_map_error {ε α β} (f : α → β) (e : ε) : (Except.error e : Except ε α).map f = .error e := rfl

/-- Well-formed cursor: `pos ≤ buf.len()` (what makes `remaining` not wrap). -/
def Rd.WF (r : Rd) : Prop := r.pos ≤ r.buf.length

theorem Rd.new_wf (buf : Bytes) : (Rd.new buf).WF := Nat.zero_le _
theorem Rd.new_rest (buf : Bytes) : (Rd.new buf).rest = buf := by simp [Rd.new, Rd.rest]

theorem Rd.remaining_eq (r : Rd) : r.remaining = r.rest.length := by
  simp [Rd.remaining, Rd.rest]

theorem Rd.isEmpty_iff (r : Rd) : r.isEmpty = true ↔ r.rest = [] := by
  simp [Rd.isEmpty, Rd.remaining_eq, List.length_eq_zero_iff]

/-- `Reader::take` refines `Prim.take`, and keeps the cursor well-formed. -/
theorem Rd.take_refines (r : Rd) (n : Nat) :
    (r.take n).map (fun p => (p.1, p.2.rest)) = FalkorCodec.take n r.rest := by
  unfold Rd.take FalkorCodec.take
  rw [Rd.remaining_eq]
  split
  · rfl
  · simp [Rd.rest, List.drop_drop, Nat.add_comm]

theorem Rd.take_wf (r : Rd) (hw : r.WF) (n : Nat) (out : Bytes) (r' : Rd) (h : r.take n = .ok (out, r')) :
    r'.WF ∧ r'.buf = r.buf := by
  unfold Rd.take at h
  by_cases hn : r.remaining < n
  · simp [hn] at h
  · simp only [hn, ite_false, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨_, rfl⟩ := h
    unfold Rd.WF at *; simp only [Rd.remaining] at hn
    exact ⟨by simp only; omega, rfl⟩

/-- `guard_count`'s saturating multiply is exact: it refuses iff `count * W`
exceeds what is left (since `remaining < 2^64 - 1`; at exactly `2^64 - 1` the
saturated product ties and the following `take` reports EOF instead). -/
theorem Rd.guardCount_refines (r : Rd) (count k : Nat) (hr : r.remaining < 2 ^ 64 - 1) :
    (r.guardCount count k).map (fun n => (n, r.rest)) = FalkorCodec.guardCount count k r.rest := by
  unfold Rd.guardCount FalkorCodec.guardCount
  rw [← Rd.remaining_eq]
  have : (min (count * k) (2 ^ 64 - 1) > r.remaining) ↔ (count * k > r.remaining) := by
    omega
  by_cases h : count * k > r.remaining
  · simp [this.mpr h, h]
  · have h' : ¬ (min (count * k) (2 ^ 64 - 1) > r.remaining) := fun x => h (this.mp x)
    simp [h', h]

/-- The fixed-width reads are the unsigned read's bit pattern. -/
theorem Rd.bits_refines (r : Rd) (k : Nat) :
    (r.bits k).map (fun p => (p.1.toNat, p.2.rest)) = (rdU k r.rest).map
      (fun p => (p.1 % 2 ^ (8 * k), p.2)) := by
  have := Rd.take_refines r k
  unfold Rd.bits Rd.takeArray rdU
  simp only [bind_apply]
  revert this
  cases r.take k with
  | error e => intro h; simp at h; rw [← h]; rfl
  | ok p => intro h; simp at h; rw [← h]; simp [BitVec.toNat_ofNat]

theorem readN_take_chunks (k : Nat) : ∀ (n : Nat) (bs rest : Bytes), bs.length = n * k →
    readN n (FalkorCodec.take k) (bs ++ rest) = .ok (chunks k n bs, rest) := by
  intro n
  induction n with
  | zero => intro bs rest h; simp at h; subst h; rfl
  | succ n ih =>
    intro bs rest h
    have hk : k ≤ bs.length := by rw [h, Nat.succ_mul]; omega
    simp only [readN, bind_apply, FalkorCodec.take, List.length_append]
    have : ¬ (bs.length + rest.length < k) := by omega
    simp only [this, ite_false, List.take_append_of_le_length hk, List.drop_append_of_le_length hk]
    rw [ih (bs.drop k) rest (by simp [h, Nat.succ_mul])]
    rfl

/-- **`take_n` = element-by-element.** One bounds check and one slice, then
`chunks_exact`, is the same as `count` reads of `W` bytes each. -/
theorem Rd.takeN_refines (r : Rd) (k count : Nat) (hr : r.remaining < 2 ^ 64 - 1) :
    (r.takeN k count id).map (fun p => (p.1, p.2.rest)) =
      (do let n ← FalkorCodec.guardCount count k; readN n (FalkorCodec.take k)) r.rest := by
  have hrl := Rd.remaining_eq r
  unfold Rd.takeN Rd.guardCount
  simp only [bind_apply, FalkorCodec.guardCount]
  by_cases hg : count * k > r.remaining
  · have : min (count * k) (2 ^ 64 - 1) > r.remaining := by omega
    simp only [this, ite_true, hrl ▸ hg]; simp [hrl]
  · have : ¬ min (count * k) (2 ^ 64 - 1) > r.remaining := by omega
    have hg' : ¬ count * k > r.rest.length := by omega
    simp only [this, ite_false, hg']
    unfold Rd.take
    have hlt : ¬ r.remaining < count * k := by omega
    simp only [hlt, ite_false, List.map_id_fun, id_eq, exc_map_ok]
    have hl : (List.take (count * k) r.rest).length = count * k := by simp; omega
    have := readN_take_chunks k count (List.take (count * k) r.rest) (List.drop (count * k) r.rest) hl
    rw [List.take_append_drop] at this
    rw [this]; simp [Rd.rest, List.drop_drop, Nat.add_comm]

/-! ### `EffectWrite` (`graph/src/effects/writer.rs`) -/

/-- The trait: an accumulator with `bytes` and `written`; `reserve` defaults to a
no-op. `Vec<u8>` is the instance the engine uses. -/
structure EffectWrite (W : Type) where
  bytes : W → Bytes → W
  written : W → Nat
  reserve : W → Nat → W := fun w _ => w

/-- `impl EffectWrite for Vec<u8>` (`writer.rs:163-180`): `extend_from_slice`,
`len()`, `Vec::reserve` (capacity only — the contents are unchanged). -/
def vecWrite : EffectWrite Bytes where
  bytes w b := w ++ b
  written w := w.length
  reserve w _ := w

/-- `EffectWrite::schema_id` (`writer.rs:95`): `bytes(v.to_le_bytes())`. -/
def schemaId {W} (E : EffectWrite W) (w : W) (v : Nat) : W := E.bytes w (w32 v)

/-- The trait's laws, for the `Vec` instance: `bytes` appends, `written` counts,
`reserve` does not change the contents. -/
theorem vec_bytes (w b : Bytes) : vecWrite.bytes w b = w ++ b := rfl
theorem vec_written (w b : Bytes) : vecWrite.written (vecWrite.bytes w b) = vecWrite.written w + b.length := by
  simp [vecWrite]
theorem vec_reserve (w : Bytes) (n : Nat) : vecWrite.reserve w n = w := rfl
theorem default_reserve {W} (bytes : W → Bytes → W) (written : W → Nat) (w : W) (n : Nat) :
    ({ bytes, written } : EffectWrite W).reserve w n = w := rfl

/-- `schema_id` writes the id as 4 LE bytes, and `u32` reads it back. -/
theorem schemaId_roundtrip (w : Bytes) (v : Nat) (hv : v < 2 ^ 32) (rest : Bytes) :
    schemaId vecWrite w v = w ++ w32 v ∧ u32 (w32 v ++ rest) = .ok (v, rest) :=
  ⟨rfl, u32_w32 hv rest⟩

end FalkorCodec
