import CchCoarsenReview.Cpack
/-!
# `hkey`: the matching keys and the cap test in one pass over `A`'s CSR

**Rust** (m1opt.patch @7df8e5887): `host_keys`, m1_coarsen_bench.rs:1354-1388, called from
`coarsen_level` (:819-820) with `capped = 2·maxw > cap`. For each entry `(i, j)` of `A` it skips the
pair when `capped && w(i) + w(j) > cap` (:1373-1375) and otherwise emits the key
`(A(i,j) as u32 as u64) << 32 | fmix32(j ^ salt)` (:1376-1382).

**Replaced** (the switch off, :792-828): when `2·maxw > cap` the cap mask
`S = select(VALUELE cap, eWiseMult(PLUS, mxm(PLUS_FIRST, D, A), mxm(PLUS_SECOND, A, D)))` with
`D = diag(w)` (:792-814; `caprc`'s two-product form, :798-802, is superseded by `hkey`), then
`K = apply(key_op, A)` under `S`'s structure (`GrB_DESC_S`, :821-827); with no cap mask, the apply is
unmasked. `key_op` (:528-540) computes the same key as the host pass.

`fmix32` and its inverse `unfmix32` (:511-526) are modelled exactly (`BitVec 32`): `unfmix_fmix`
proves the key's low half decodes back to `j ^ salt`, which the matching relies
on (`unkey_op`, :542-554), and that keys in one row are distinct ("unique per row, so MAX is
deterministic", :527).
-/

namespace CchCoarsen

/-- `fmix32` (:511-518): `h ^= h >> 16; h *= 0x85eb_ca6b; h ^= h >> 13; h *= 0xc2b2_ae35; h ^= h >> 16`. -/
def fmix32 (h0 : BitVec 32) : BitVec 32 :=
  let h1 := h0 ^^^ (h0 >>> 16)
  let h2 := h1 * 0x85ebca6b#32
  let h3 := h2 ^^^ (h2 >>> 13)
  let h4 := h3 * 0xc2b2ae35#32
  h4 ^^^ (h4 >>> 16)

/-- `unfmix32` (:519-526). -/
def unfmix32 (h0 : BitVec 32) : BitVec 32 :=
  let h1 := h0 ^^^ (h0 >>> 16)
  let h2 := h1 * 0x7ed1b41d#32
  let h3 := h2 ^^^ ((h2 >>> 13) ^^^ (h2 >>> 26))
  let h4 := h3 * 0xa5cb9243#32
  h4 ^^^ (h4 >>> 16)

theorem xs16_inv (h : BitVec 32) : (h ^^^ (h >>> 16)) ^^^ ((h ^^^ (h >>> 16)) >>> 16) = h := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight]
  have : h.getLsbD (16 + (16 + i)) = false := BitVec.getLsbD_of_ge h _ (by omega)
  rw [this]
  cases h.getLsbD i <;> cases h.getLsbD (16 + i) <;> rfl

theorem xs13_inv (h : BitVec 32) :
    let g := h ^^^ (h >>> 13)
    g ^^^ ((g >>> 13) ^^^ (g >>> 26)) = h := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight]
  have : h.getLsbD (13 + (26 + i)) = false := BitVec.getLsbD_of_ge h _ (by omega)
  rw [this]
  have e : 13 + (13 + i) = 26 + i := by omega
  rw [e]
  cases h.getLsbD i <;> cases h.getLsbD (13 + i) <;> cases h.getLsbD (26 + i) <;> rfl

theorem mul_inv_cancel (x a b : BitVec 32) (h : a * b = 1#32) : x * a * b = x := by
  rw [BitVec.mul_assoc, h, BitVec.mul_one]

/-- **`unfmix32` inverts `fmix32`** (both multipliers' inverses checked by `decide` mod 2³²). -/
theorem unfmix_fmix (x : BitVec 32) : unfmix32 (fmix32 x) = x := by
  unfold unfmix32 fmix32
  simp only
  rw [xs16_inv, mul_inv_cancel _ _ _ (by decide), xs13_inv, mul_inv_cancel _ _ _ (by decide), xs16_inv]

theorem fmix_injective {x y : BitVec 32} (h : fmix32 x = fmix32 y) : x = y := by
  rw [← unfmix_fmix x, ← unfmix_fmix y, h]

/-- The key of entry `(i, j)` with value `v` (:1376-1377 host, :536-538 `key_op`):
`(v as u32 as u64) << 32 | fmix32(j as u32 ^ salt as u32) as u64`. -/
def keyOf (salt : Nat) (v : Int) (j : Nat) : BitVec 64 :=
  ((BitVec.ofInt 32 v).setWidth 64 <<< 32) ||| (fmix32 (BitVec.ofNat 32 j ^^^ BitVec.ofNat 32 salt)).setWidth 64

/-- `unkey_op` (:542-554): `(unfmix32(k as u32) ^ salt as u32) as u64`. -/
def unkeyOf (salt : Nat) (k : BitVec 64) : BitVec 64 :=
  (unfmix32 (k.setWidth 32) ^^^ BitVec.ofNat 32 salt).setWidth 64

theorem low_of_key (a b : BitVec 32) : ((a.setWidth 64 <<< 32) ||| b.setWidth 64).setWidth 32 = b := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp [hi]

theorem high_of_key (a b : BitVec 32) :
    (((a.setWidth 64 <<< 32) ||| b.setWidth 64) >>> 32).setWidth 32 = a := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp [BitVec.getLsbD_ushiftRight, hi]
  have h1 : 32 + i < 64 := by omega
  have h2 : ¬ (32 + i < 32) := by omega
  have h3 : i < 64 := by omega
  simp [h1, h2, h3]

/-- **The key decodes to its column**: `unkey(key(v, j)) = j` for every `j < 2³²`. -/
theorem unkey_key (salt : Nat) (v : Int) (j : Nat) :
    unkeyOf salt (keyOf salt v j) = (BitVec.ofNat 32 j).setWidth 64 := by
  unfold unkeyOf keyOf
  rw [low_of_key, unfmix_fmix, BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

/-- **Keys are unique in a row** (any values): two columns below 2³² with equal keys are equal, so the
`MAX` over a row picks one column. -/
theorem key_col_injective (salt : Nat) {v v' : Int} {j j' : Nat} (hj : j < 2 ^ 32) (hj' : j' < 2 ^ 32)
    (h : keyOf salt v j = keyOf salt v' j') : j = j' := by
  have := congrArg (unkeyOf salt) h
  rw [unkey_key, unkey_key] at this
  have h2 := congrArg (BitVec.setWidth 32) this
  simp only [BitVec.setWidth_setWidth_of_le _ (show 32 ≤ 64 by omega), BitVec.setWidth_eq] at h2
  have h3 := congrArg BitVec.toNat h2
  simp [BitVec.toNat_ofNat] at h3
  omega

/-- The key's high half is the entry's value (as `u32`): `MAX` prefers the heaviest edge. -/
theorem key_high (salt : Nat) (v : Int) (j : Nat) :
    ((keyOf salt v j) >>> 32).setWidth 32 = BitVec.ofInt 32 v := by
  unfold keyOf; rw [high_of_key]

/-! ## The pass against the cap mask -/

/-- `host_keys`, row `i` (:1368-1384), for any key function. -/
def hostKeyRows {β : Type} (A : Rows Int) (wh : Nat → Int) (cap : Int) (capped : Bool) (key : Int → Nat → β) :
    Rows β := fun i =>
  (A i).filterMap fun e => if capped = true ∧ wh i + wh e.1 > cap then none else some (e.1, key e.2 e.1)

/-- `GrB_Matrix_diag(d, w, 0)`: `w` on the diagonal. -/
def diagM (w : Nat → Option Int) : Mat Int := fun i j => if i = j then w i else none

/-- The weight vector as GraphBLAS holds it (an entry at the live vertices, `wvec`). -/
def wv (wh : Nat → Int) : Nat → Option Int := fun i => if live wh i then some (wh i) else none

/-- The cap mask (:796-812, the non-`caprc` branch). -/
def capMask (n : Nat) (wh : Nat → Int) (cap : Int) (A : Mat Int) : Mat Int :=
  selLE cap (ewMult (· + ·) (mxm (fun a _ => a) n (diagM (wv wh)) A) (mxm (fun _ b => b) n A (diagM (wv wh))))

/-- The switch-off keys (:821-827): masked by the cap mask when there is one. -/
def keysGB {β : Type} (n : Nat) (wh : Nat → Int) (cap : Int) (capped : Bool) (key : Int → Nat → β)
    (A : Mat Int) : Mat β :=
  if capped then applyS (capMask n wh cap A) (fun _ j v => key v j) A else apply (fun _ j v => key v j) A

/-- `A`'s preconditions here: the unloaded CSR of an `n × n` matrix whose entries join live vertices. -/
structure KeyPre (n : Nat) (A : Rows Int) (wh : Nat → Int) : Prop where
  rowStrict : ∀ i, StrictRow (A i)
  colLt : ∀ i, ∀ e ∈ A i, e.1 < n
  rowsBeyond : ∀ i, n ≤ i → A i = []
  liveEnds : ∀ i, ∀ e ∈ A i, live wh i ∧ live wh e.1

/-- `mxm` with a diagonal operand collapses to the one diagonal index. -/
theorem mxm_diag_left (n : Nat) (w : Nat → Option Int) (A : Mat Int) (i j : Nat) (hi : i < n) :
    mxm (fun a _ => a) n (diagM w) A i j =
      if (w i).isSome ∧ (A i j).isSome then w i else none := by
  unfold mxm
  have hb : ∀ t, both (diagM w) A i j t = (decide (t = i) && ((w i).isSome && (A i j).isSome)) := by
    intro t; unfold both diagM
    by_cases h : i = t
    · subst h; simp
    · have : t ≠ i := Ne.symm h
      simp [h, this]
  rw [show both (diagM w) A i j = _ from funext hb]
  by_cases hc : (w i).isSome ∧ (A i j).isSome
  · have hany : (List.range n).any (fun t => decide (t = i) && ((w i).isSome && (A i j).isSome)) = true :=
      List.any_eq_true.2 ⟨i, List.mem_range.2 hi, by simp [hc.1, hc.2]⟩
    simp only [hany, ↓reduceIte]
    rw [lsum_filter]
    have := lsum_single List.nodup_range (List.mem_range.2 hi) (fun t => ((diagM w) i t).getD 0)
    simp only [hc.1, hc.2, Bool.and_true] at this ⊢
    rw [show (fun a => if decide (a = i) = true then ((diagM w) i a).getD 0 else 0) =
      (fun t => if t = i then ((diagM w) i t).getD 0 else 0) from by funext t; simp, this]
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hc.1
    simp [diagM, hx]
  · have hany : (List.range n).any (fun t => decide (t = i) && ((w i).isSome && (A i j).isSome)) = false := by
      rw [List.any_eq_false]; intro t _; simp only [Bool.and_eq_true, decide_eq_true_eq, not_and]
      intro _ h1 h2; exact hc ⟨h1, h2⟩
    rw [hany]; simp [hc]

theorem mxm_diag_right (n : Nat) (w : Nat → Option Int) (A : Mat Int) (i j : Nat) (hj : j < n) :
    mxm (fun _ b => b) n A (diagM w) i j =
      if (A i j).isSome ∧ (w j).isSome then w j else none := by
  unfold mxm
  have hb : ∀ t, both A (diagM w) i j t = (decide (t = j) && ((A i j).isSome && (w j).isSome)) := by
    intro t; unfold both diagM
    by_cases h : t = j
    · subst h; simp
    · simp [h]
  rw [show both A (diagM w) i j = _ from funext hb]
  by_cases hc : (A i j).isSome ∧ (w j).isSome
  · have hany : (List.range n).any (fun t => decide (t = j) && ((A i j).isSome && (w j).isSome)) = true :=
      List.any_eq_true.2 ⟨j, List.mem_range.2 hj, by simp [hc.1, hc.2]⟩
    simp only [hany, ↓reduceIte]
    rw [lsum_filter]
    have := lsum_single List.nodup_range (List.mem_range.2 hj) (fun t => ((diagM w) t j).getD 0)
    simp only [hc.1, hc.2, Bool.and_true] at this ⊢
    rw [show (fun a => if decide (a = j) = true then ((diagM w) a j).getD 0 else 0) =
      (fun t => if t = j then ((diagM w) t j).getD 0 else 0) from by funext t; simp, this]
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hc.2
    simp [diagM, hx]
  · have hany : (List.range n).any (fun t => decide (t = j) && ((A i j).isSome && (w j).isSome)) = false := by
      rw [List.any_eq_false]; intro t _; simp only [Bool.and_eq_true, decide_eq_true_eq, not_and]
      intro _ h1 h2; exact hc ⟨h1, h2⟩
    rw [hany]; simp [hc]

/-- The cap mask stores `w(i) + w(j)` at `A`'s entries between live vertices, kept when `≤ cap`. -/
theorem capMask_spec {n : Nat} {A : Rows Int} {wh : Nat → Int} (hp : KeyPre n A wh) (cap : Int) (i j : Nat) :
    (capMask n wh cap (den A) i j).isSome ↔ ((den A i j).isSome ∧ wh i + wh j ≤ cap) := by
  by_cases hA : (den A i j).isSome
  · obtain ⟨v, hv⟩ := (lookup_isSome_iff _ _).1 hA
    have hi : i < n := by
      by_cases h : n ≤ i
      · rw [hp.rowsBeyond i h] at hv; simp at hv
      · omega
    have hj : j < n := hp.colLt i (j, v) hv
    have hl := hp.liveEnds i (j, v) hv
    have hwi : wv wh i = some (wh i) := by simp [wv, hl.1]
    have hwj : wv wh j = some (wh j) := by simp [wv, hl.2]
    unfold capMask selLE ewMult
    rw [mxm_diag_left n _ _ i j hi, mxm_diag_right n _ _ i j hj, hwi, hwj]
    simp only [Option.isSome_some, hA, and_self, ite_true]
    by_cases hc : wh i + wh j ≤ cap <;> simp [hc]
  · unfold capMask selLE ewMult
    have hi : ¬ ((wv wh i).isSome ∧ (den A i j).isSome) := fun h => hA h.2
    by_cases hin : i < n
    · rw [mxm_diag_left n _ _ i j hin]; simp [hA]
    · have : A i = [] := hp.rowsBeyond i (by omega)
      have hm : mxm (fun a _ => a) n (diagM (wv wh)) (den A) i j = none := by
        unfold mxm
        have : (List.range n).any (both (diagM (wv wh)) (den A) i j) = false := by
          rw [List.any_eq_false]; intro t ht
          have : t ≠ i := fun h => hin (h ▸ List.mem_range.1 ht)
          simp [both, diagM, Ne.symm this]
        rw [this]; rfl
      simp [hm, hA]

theorem lookup_filterMap_key {β : Type} {row : List (Nat × Int)} (hn : (row.map (·.1)).Nodup)
    (p : Nat → Bool) (key : Int → Nat → β) (j : Nat) :
    (row.filterMap fun e => if p e.1 = true then none else some (e.1, key e.2 e.1)).lookup j =
      (row.lookup j).bind (fun v => if p j = true then none else some (key v j)) := by
  induction row with
  | nil => simp
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    have hna : row.lookup a = none := lookup_eq_none_of (fun e he hea => hn.1 ⟨e, he, hea⟩)
    rw [List.filterMap_cons, List.lookup_cons]
    by_cases hja : j = a
    · subst hja
      by_cases hc : p j = true
      · simp only [hc, ite_true]
        rw [ih hn.2, hna]; simp
      · simp [hc]
    · have hb : (j == a) = false := by simp [hja]
      rw [hb]
      by_cases hc : p a = true
      · simp only [hc, ite_true]; exact ih hn.2
      · simp only [hc, Bool.false_eq_true, ite_false, List.lookup_cons, hb]; exact ih hn.2

theorem filterMap_congr' {α β : Type} {l : List α} {f g : α → Option β} (h : ∀ a ∈ l, f a = g a) :
    l.filterMap f = l.filterMap g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.filterMap_cons, List.filterMap_cons, h a (by simp), ih (fun x hx => h x (by simp [hx]))]

/-- **`hkey` = the masked key apply**: for any key function, the host pass stores at `(i, j)` exactly
what `apply(key, A)` under the cap mask (or unmasked when `capped` is false) stores. -/
theorem hkey_eq {β : Type} {n : Nat} {A : Rows Int} {wh : Nat → Int} (hp : KeyPre n A wh) (cap : Int)
    (capped : Bool) (key : Int → Nat → β) (i j : Nat) :
    den (hostKeyRows A wh cap capped key) i j = keysGB n wh cap capped key (den A) i j := by
  unfold keysGB
  have hrow : den (hostKeyRows A wh cap capped key) i j =
      ((A i).lookup j).bind (fun v => if capped = true ∧ wh i + wh j > cap then none else some (key v j)) := by
    have := lookup_filterMap_key (hp.rowStrict i).keys_nodup
      (fun c => decide (capped = true ∧ wh i + wh c > cap)) key j
    simpa [den, hostKeyRows] using this
  rw [hrow]
  cases hcp : capped
  · simp only [Bool.false_eq_true, false_and, ite_false, apply, den]
    cases (A i).lookup j <;> rfl
  · simp only [ite_true, applyS]
    have hm := capMask_spec hp cap i j
    cases hl : (A i).lookup j with
    | none => simp [den, hl]
    | some v =>
      have hA : (den A i j).isSome := by simp [den, hl]
      by_cases hc : wh i + wh j ≤ cap
      · have : (capMask n wh cap (den A) i j).isSome := hm.2 ⟨hA, hc⟩
        have hgt : ¬ (wh i + wh j > cap) := by omega
        simp [this, hgt, den, hl]
      · have : ¬ (capMask n wh cap (den A) i j).isSome := fun h => hc (hm.1 h).2
        have hgt : wh i + wh j > cap := by omega
        simp [this, hgt]

/-- **`host_keys`'s rows are strictly ascending** (a filter of `A`'s sorted row): the keys matrix is
packed with `jumbled = false` (`pack_csr_t`, :306), so this is what GraphBLAS assumes of it. -/
theorem hkey_row_shape {β : Type} {A : Rows Int} (hA : ∀ i, StrictRow (A i)) (wh : Nat → Int) (cap : Int)
    (capped : Bool) (key : Int → Nat → β) (i : Nat) : StrictRow (hostKeyRows A wh cap capped key i) := by
  unfold hostKeyRows StrictRow
  refine List.Pairwise.filterMap _ ?_ (hA i)
  intro a a' hlt b hb b' hb'
  by_cases h1 : capped = true ∧ wh i + wh a.1 > cap
  · simp [h1] at hb
  · by_cases h2 : capped = true ∧ wh i + wh a'.1 > cap
    · simp [h2] at hb'
    · simp only [h1, h2, ite_false, Option.some.injEq] at hb hb'
      rw [← hb, ← hb']; exact hlt

/-- When `2·maxw ≤ cap` (`capped` false) no pair breaks the cap anyway: the unmasked keys are exactly
the capped ones. -/
theorem hkey_uncapped_vacuous {A : Rows Int} {wh : Nat → Int} {cap maxw : Int}
    (hmax : ∀ i, live wh i → wh i ≤ maxw) (h2 : 2 * maxw ≤ cap)
    (hl : ∀ i, ∀ e ∈ A i, live wh i ∧ live wh e.1) {β : Type} (key : Int → Nat → β) :
    hostKeyRows A wh cap false key = hostKeyRows A wh cap true key := by
  funext i
  simp only [hostKeyRows]
  refine filterMap_congr' (fun e he => ?_)
  have := hl i e he
  have h1 := hmax i this.1; have h2' := hmax e.1 this.2
  have : ¬ (wh i + wh e.1 > cap) := by omega
  simp [this]

end CchCoarsen
