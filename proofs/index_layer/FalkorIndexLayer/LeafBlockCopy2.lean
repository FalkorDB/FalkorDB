import FalkorIndexLayer.LeafBlockCopy
/-
# `block_copy_merge`, byte for byte
-/
namespace IndexLayer.Leaf

/-- Rust tuple `<` on `(u64, u64)`. -/
def tupLt (a b : P) : Bool := decide (a.1 < b.1) || (decide (a.1 = b.1) && decide (a.2 < b.2))

theorem tupLt_pltB (a b : P) : tupLt a b = pltB a b := by
  simp only [tupLt, pltB, lexLe]
  rw [Bool.eq_iff_iff]
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', beq_eq_false_iff_ne, ne_eq]
  constructor
  · rintro (h | ⟨h1, h2⟩)
    · exact ⟨Or.inl h, fun e => by rw [e] at h; omega⟩
    · exact ⟨Or.inr ⟨h1, by omega⟩, fun e => by rw [e] at h2; omega⟩
  · rintro ⟨h | ⟨h1, h2⟩, hne⟩
    · exact Or.inl h
    · right; refine ⟨h1, ?_⟩
      rcases Nat.lt_or_ge a.2 b.2 with h | h
      · exact h
      · exact absurd (Prod.ext h1 (by omega)) hne

/-- The loop of `block_copy_merge`: copy the leaf's index/doc bytes over
`[leaf_pos, position)`, then the batch entry unless it is a duplicate; finally
the leaf tail. Returns `(new_index, new_docs)`. -/
def bcmLoop (b : List Nat) (n io docsOff dw : Nat) (key doc slotOf : Nat → Nat) :
    Nat → Option P → List P → List Nat × List Nat
  | pos, _, [] => (slice b (io + pos) (io + n), slice b (docsOff + pos * dw) (docsOff + n * dw))
  | pos, prev, x :: bs =>
    let position := gallop (fun i => tupLt (key i, doc i) x) pos n
    let dup := prev == some x || (decide (position < n) && key position == x.1 && doc position == x.2)
    let r := bcmLoop b n io docsOff dw key doc slotOf position (some x) bs
    (slice b (io + pos) (io + position) ++ (if dup then [] else [slotOf x.1]) ++ r.1,
     slice b (docsOff + pos * dw) (docsOff + position * dw) ++ (if dup then [] else (le 8 x.2).take dw) ++ r.2)

/-- `CompactIndexedLeaf::block_copy_merge`. -/
def ciBcm (b : List Nat) (batch : List P) : List Nat :=
  let n := readU16 b 0
  let min := readU64 b 2
  let vw := byteAt b 10
  let dc := readU16 b 11
  let dw := byteAt b 13
  let io := (layout b).1
  let docsOff := (layout b).2
  let slotOf := fun k => partitionPoint (slotPred b k) 0 dc
  let r := bcmLoop b n io docsOff dw (ciKey b) (ciDoc b) slotOf 0 none batch
  header r.1.length min vw dc dw ++ slice b 14 io ++ r.1 ++ r.2

def idxOf (idx : List Nat) (slot : Nat → Nat) (e : P × Option Nat) : Nat :=
  match e.2 with | some i => idx.getD i 0 | none => slot e.1.1

theorem pp_bounds (p : Nat → Bool) : ∀ (f lo hi : Nat), lo ≤ hi → lo ≤ pp p f lo hi ∧ pp p f lo hi ≤ hi
  | 0, lo, hi, h => ⟨Nat.le_refl _, h⟩
  | f + 1, lo, hi, h => by
    simp only [pp]
    split
    · split
      · have := pp_bounds p f (lo + (hi - lo) / 2 + 1) hi (by omega); omega
      · have := pp_bounds p f lo (lo + (hi - lo) / 2) (by omega); omega
    · exact ⟨Nat.le_refl _, h⟩

theorem gallop_bounds (p : Nat → Bool) (start count : Nat) (h : start ≤ count) :
    start ≤ gallop p start count ∧ gallop p start count ≤ count := by
  unfold gallop partitionPoint
  dsimp only
  have hg := gallopStep_spec p start count count 1 (Nat.le_refl _) (Or.inl rfl)
  simp only at hg
  generalize gallopStep p start count count 1 = st at hg
  obtain ⟨g1, g2, -⟩ := hg
  have hlo : start + st / 2 ≤ min (start + st) count := by
    rcases g2 with e | e
    · subst e; simp; omega
    · have : st / 2 ≤ st := Nat.div_le_self _ _
      omega
  have := pp_bounds p (min (start + st) count - (start + st / 2) + 1) _ _ hlo
  omega

theorem range_getD (l : List Nat) (a k : Nat) (h : a + k ≤ l.length) :
    (List.range' a k).map (fun i => l.getD i 0) = (l.drop a).take k := by
  apply List.ext_getElem (by simp; omega)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range', Nat.one_mul, List.getElem_take, List.getElem_drop]
  exact getD_of_lt _ _ _ (by simp at h1; omega)

theorem range_flat (l : List Nat) (w a k : Nat) (h : a + k ≤ l.length) :
    ((List.range' a k).map (fun i => l.getD i 0)).flatMap (le w) = ((l.drop a).take k).flatMap (le w) := by
  rw [range_getD l a k h]

/-- The loop's output is the per-emission image of `bcmL`. -/
theorem bcmLoop_eq (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (L : List P)
    (hLd : L = ciDecode h.min dv idx docs) (slot : Nat → Nat) :
    ∀ (batch : List P) (pos : Nat) (prev : Option P), pos ≤ h.n →
    bcmLoop (ciBytes h dv idx docs) h.n (14 + h.dc * h.vw) (14 + h.dc * h.vw + h.n) h.dw
        (ciKey (ciBytes h dv idx docs)) (ciDoc (ciBytes h dv idx docs)) slot pos prev batch =
      ((bcmL L pos batch prev).map (idxOf idx slot),
       (bcmL L pos batch prev).flatMap (pickEnc (fun i => le h.dw (docs.getD i 0)) (fun x => le h.dw x.2))) := by
  obtain ⟨-, sI, sD, -⟩ := ciSlices h dv idx docs ok
  have hrd := (ciRead h dv idx docs ok)
  have hLn : L.length = h.n := by rw [hLd]; simp [ciDecode, ok.idxlen, ok.doclen]
  have hkd : ∀ i (hi : i < h.n), ciKey (ciBytes h dv idx docs) i = (Lg L i).1 ∧
      ciDoc (ciBytes h dv idx docs) i = (Lg L i).2 := by
    intro i hi
    have e := hrd.2.2
    rw [← hLd] at e
    unfold ciPairs pairsOf at e
    rw [hrd.1] at e
    have hi' : i < L.length := by rw [hLn]; exact hi
    have := congrArg (fun l => l.getD i (0, 0)) e
    rw [getD_of_lt _ _ _ (by simp; exact hi), getD_of_lt _ _ _ hi', List.getElem_map, List.getElem_range] at this
    rw [Lg_eq L i hi', ← this]
    exact ⟨rfl, rfl⟩
  have seg : ∀ a c, a ≤ c → c ≤ h.n →
      slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + a) (14 + h.dc * h.vw + c) =
        ((List.range' a (c - a)).map (fun i => (Lg L i, some i))).map (idxOf idx slot) ∧
      slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + h.n + a * h.dw) (14 + h.dc * h.vw + h.n + c * h.dw) =
        ((List.range' a (c - a)).map (fun i => (Lg L i, some i))).flatMap
          (pickEnc (fun i => le h.dw (docs.getD i 0)) (fun x => le h.dw x.2)) := by
    intro a c hac hc
    constructor
    · rw [sI a c hac hc, List.map_map, ← range_getD idx a (c - a) (by rw [ok.idxlen]; omega)]
      rfl
    · rw [sD a c hac hc, List.flatMap_map, ← range_flat docs h.dw a (c - a) (by rw [ok.doclen]; omega),
        List.flatMap_map]
      rfl
  intro batch
  induction batch with
  | nil =>
    intro pos prev hp
    simp only [bcmLoop, bcmL]
    rw [hLn, (seg pos h.n hp (Nat.le_refl _)).1, (seg pos h.n hp (Nat.le_refl _)).2]
  | cons x bs ih =>
    intro pos prev hp
    have hcong : gallop (fun i => tupLt (ciKey (ciBytes h dv idx docs) i, ciDoc (ciBytes h dv idx docs) i) x) pos h.n =
        gallop (fun i => pltB (Lg L i) x) pos L.length := by
      rw [hLn]
      apply gallop_congr
      intro i hi
      obtain ⟨k1, k2⟩ := hkd i hi
      rw [k1, k2, tupLt_pltB]
    simp only [bcmLoop, bcmL]
    rw [hcong]
    have hgb := gallop_bounds (fun i => pltB (Lg L i) x) pos L.length (by rw [hLn]; exact hp)
    generalize hgp : gallop (fun i => pltB (Lg L i) x) pos L.length = p at hgb
    have hpn : p ≤ h.n := by rw [← hLn]; exact hgb.2
    have hdup : (decide (p < h.n) && ciKey (ciBytes h dv idx docs) p == x.1 && ciDoc (ciBytes h dv idx docs) p == x.2) =
        (decide (p < L.length) && Lg L p == x) := by
      rw [hLn]
      by_cases hc : p < h.n
      · obtain ⟨k1, k2⟩ := hkd p hc
        rw [k1, k2]; simp only [hc, decide_true, Bool.true_and]
        rw [Bool.eq_iff_iff]; simp [Prod.ext_iff]
      · simp [hc]
    rw [hdup, ih p (some x) hpn, (seg pos p hgb.1 hpn).1, (seg pos p hgb.1 hpn).2,
      le8_take _ _ (W_le8 _ ok.hok.2.2.2.2)]
    split <;> simp [idxOf, pickEnc, List.append_assoc]

theorem bcmL_leaf (L : List P) : ∀ (batch : List P) (pos : Nat) (prev : Option P) (x : P) (i : Nat),
    pos ≤ L.length → (x, some i) ∈ bcmL L pos batch prev → ∃ hi : i < L.length, L[i] = x
  | [], pos, prev, x, i, hp, h => by
    simp only [bcmL, List.mem_map, List.mem_range', Prod.mk.injEq, Option.some.injEq] at h
    obtain ⟨j, ⟨k, hk, rfl⟩, h1, h2⟩ := h
    subst h2
    have : pos + 1 * k < L.length := by omega
    exact ⟨this, by rw [← h1, Lg_eq L _ this]⟩
  | y :: bs, pos, prev, x, i, hp, h => by
    have hb := gallop_bounds (fun i => pltB (Lg L i) y) pos L.length hp
    simp only [bcmL] at h
    rcases List.mem_append.mp h with h | h
    · rcases List.mem_append.mp h with h | h
      · simp only [List.mem_map, List.mem_range', Prod.mk.injEq, Option.some.injEq] at h
        obtain ⟨j, ⟨k, hk, rfl⟩, h1, h2⟩ := h
        subst h2
        have : pos + 1 * k < L.length := by omega
        exact ⟨this, by rw [← h1, Lg_eq L _ this]⟩
      · split at h <;> simp at h
    · exact bcmL_leaf L bs _ _ x i hb.2 h

/-- A present key's `partition_point` slot holds it. -/
theorem slot_present (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hsort : Sorted dv)
    (k : Nat) (hk : ∃ s, ∃ hs : s < dv.length, h.min + dv[s] = k) :
    ∃ hs : partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc < dv.length,
      h.min + dv[partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc] = k := by
  obtain ⟨dsOk, dsErr⟩ := distinctSlot_spec h dv idx docs ok hsort k
  obtain ⟨-, -, -, r4, -⟩ := ciBytes_hdr h ok.hok dv idx docs
  cases hd : distinctSlot (ciBytes h dv idx docs) k with
  | ok s =>
    obtain ⟨hs, hv⟩ := dsOk s hd
    unfold distinctSlot at hd; simp only [r4] at hd
    split at hd
    · cases hd; exact ⟨hs, hv⟩
    · cases hd
  | err s =>
    obtain ⟨-, -, -, hne⟩ := dsErr s hd
    obtain ⟨t, ht, hv⟩ := hk
    exact absurd hv (hne t ht)

/-- **PROVEN** (`CompactIndexedLeaf::block_copy_merge` = merge of the entry
lists): when every batch key is already in the distinct table (the caller's
`distinct_slot(k).is_ok()` gate), the block-copied page reads back as
`merge_sorted(leaf, batch)`. -/
theorem ciBcm_spec (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hsort : Sorted dv)
    (hdcn : h.dc < h.n) (hLs : (ciDecode h.min dv idx docs).Pairwise (lt lexLe))
    (batch : List P) (hbs : batch.Pairwise (fun a b => lexLe a b = true)) (hfit : CFits h.min h.vw h.dw batch)
    (hpres : ∀ x ∈ batch, ∃ s, ∃ hs : s < dv.length, h.min + dv[s] = x.1)
    (hcnt : (bcmL (ciDecode h.min dv idx docs) 0 batch none).length < 2 ^ 16) :
    ciPairs (ciBcm (ciBytes h dv idx docs) batch) = mergeD lexLe (ciDecode h.min dv idx docs) batch none := by
  obtain ⟨r1, r2, r3, r4, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
  obtain ⟨sV, -, -, -⟩ := ciSlices h dv idx docs ok
  have hlay := ciLayout h dv idx docs ok hdcn
  generalize hL : ciDecode h.min dv idx docs = L at hLs hcnt ⊢
  have hLn : L.length = h.n := by rw [← hL]; simp [ciDecode, ok.idxlen, ok.doclen]
  have hLi : ∀ i (hi : i < L.length), ∃ hs : idx[i]'(by rw [ok.idxlen, ← hLn]; exact hi) < dv.length,
      L[i] = (h.min + dv[idx[i]'(by rw [ok.idxlen, ← hLn]; exact hi)], docs[i]'(by rw [ok.doclen, ← hLn]; exact hi)) := by
    intro i hi
    have hs : idx[i]'(by rw [ok.idxlen, ← hLn]; exact hi) < dv.length := by
      rw [ok.dvlen]; exact ok.slots _ (List.getElem_mem _)
    refine ⟨hs, ?_⟩
    simp only [← hL, ciDecode, List.getElem_map, List.getElem_zip]
    rw [getD_of_lt _ _ _ hs]
  unfold ciBcm
  simp only [r1, r2, r3, r4, r5, hlay]
  rw [bcmLoop_eq h dv idx docs ok L hL.symm _ batch 0 none (Nat.zero_le _)]
  have v0 := sV 0 h.dc (Nat.zero_le _) (Nat.le_refl _)
  simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at v0
  rw [v0, List.take_of_length_le (l := dv) (by simp [ok.dvlen])]
  generalize hO : bcmL L 0 batch none = O at hcnt
  have hOleaf : ∀ x i, (x, some i) ∈ O → ∃ hi : i < L.length, L[i] = x := fun x i hx =>
    bcmL_leaf L batch 0 none x i (Nat.zero_le _) (by rw [hO]; exact hx)
  have hOin : ∀ e ∈ O, e.1 ∈ L ∨ e.1 ∈ batch := by
    intro e he
    have := (bcmL_spec L hLs batch 0 none hbs (Nat.zero_le _) (fun y h => by cases h)).2 e.1
    rw [hO] at this
    rcases this.mp (List.mem_map_of_mem he) with h' | ⟨h', -⟩
    · exact Or.inl (by simpa using h')
    · exact Or.inr h'
  have hdocs := flatMap_raw O L (fun (x : P) => le h.dw x.2) (fun i => le h.dw (docs.getD i 0))
    (fun i hi => by
      obtain ⟨hs0, e0⟩ := hLi i hi
      rw [e0, getD_of_lt _ _ _ (by rw [ok.doclen, ← hLn]; exact hi)])
    hOleaf (fun (x : P) => le h.dw x.2) (fun _ => rfl)
  rw [hdocs]
  have hslot := fun k hk => slot_present h dv idx docs ok hsort k hk
  have ok' : CIOk ⟨(O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc))).length,
      h.min, h.vw, h.dc, h.dw⟩ dv (O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc)))
      (O.map (fun e => e.1.2)) := by
    refine ⟨⟨by simpa using hcnt, ok.hok.2.1, ok.hok.2.2.1, ok.hok.2.2.2.1, ok.hok.2.2.2.2⟩,
      ok.dvlen, rfl, by simp, ?_, ok.dvfit, ?_⟩
    · intro sl hsl
      obtain ⟨⟨x, oi⟩, he, rfl⟩ := List.mem_map.mp hsl
      cases oi with
      | some i =>
        obtain ⟨hi, -⟩ := hOleaf x i he
        simp only [idxOf]
        rw [getD_of_lt _ _ _ (by rw [ok.idxlen, ← hLn]; exact hi)]
        exact ok.slots _ (List.getElem_mem _)
      | none =>
        simp only [idxOf]
        rcases hOin _ he with h' | h'
        · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem h'
          obtain ⟨hs, e1⟩ := hLi i hi
          have hv : h.min + dv[idx[i]'(by rw [ok.idxlen, ← hLn]; exact hi)] = L[i].1 := by rw [e1]
          obtain ⟨hs2, -⟩ := hslot L[i].1 ⟨_, hs, hv⟩
          show _ < h.dc; rw [ok.dvlen] at hs2; exact hs2
        · obtain ⟨hs2, -⟩ := hslot x.1 (hpres x h')
          show _ < h.dc; rw [ok.dvlen] at hs2; exact hs2
    · intro d hd
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
      rcases hOin e he with h' | h'
      · obtain ⟨i, hi, e2⟩ := List.getElem_of_mem h'
        obtain ⟨hs3, e3⟩ := hLi i hi
        rw [← e2, e3]; exact ok.docfit _ (List.getElem_mem _)
      · exact (hfit _ h').2.2
  have e : header (O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc))).length
      h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++
      O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc)) ++
      O.flatMap (fun e => le h.dw e.1.2) =
      ciBytes ⟨(O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc))).length,
        h.min, h.vw, h.dc, h.dw⟩ dv (O.map (idxOf idx (fun k => partitionPoint (slotPred (ciBytes h dv idx docs) k) 0 h.dc)))
        (O.map (fun e => e.1.2)) := by
    simp [ciBytes, List.flatMap_map]
  rw [e, (ciRead _ _ _ _ ok').2.2, ← bcmL_eq_merge L batch hLs hbs, hO]
  unfold ciDecode
  rw [List.zip_map', List.map_map]
  apply List.map_congr_left
  intro ⟨x, oi⟩ he
  simp only [Function.comp]
  cases oi with
  | some i =>
    obtain ⟨hi, hx⟩ := hOleaf x i he
    obtain ⟨hs, e2⟩ := hLi i hi
    simp only [idxOf]
    rw [getD_of_lt idx i 0 (by rw [ok.idxlen, ← hLn]; exact hi), getD_of_lt _ _ _ hs, ← hx, e2]
  | none =>
    simp only [idxOf]
    have hk : ∃ s, ∃ hs : s < dv.length, h.min + dv[s] = x.1 := by
      rcases hOin _ he with h' | h'
      · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem h'
        obtain ⟨hs, e1⟩ := hLi i hi
        exact ⟨_, hs, by rw [e1]⟩
      · exact hpres x h'
    obtain ⟨hs2, hv⟩ := hslot x.1 hk
    rw [getD_of_lt _ _ _ hs2, hv]

end IndexLayer.Leaf
