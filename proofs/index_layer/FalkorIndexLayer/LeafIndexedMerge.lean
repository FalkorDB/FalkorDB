import FalkorIndexLayer.LeafIndexedOps
/-
# `CompactIndexedLeaf::merge` (`compact_indexed.rs`, origin/main 3fec7d7c9)
-/
namespace IndexLayer.Leaf

def natLe (a b : Nat) : Bool := decide (a ≤ b)

theorem natLe_lin : LinOrd natLe where
  refl a := by simp [natLe]
  trans a b c h1 h2 := by simp only [natLe, decide_eq_true_eq] at *; omega
  total a b := by simp only [natLe, decide_eq_true_eq]; omega
  antisymm a b h1 h2 := by simp only [natLe, decide_eq_true_eq] at *; omega

/-- `slot_of(key) = merged.partition_point(|&v| v < key)`. -/
def slotOf (merged : List Nat) (key : Nat) : Nat :=
  partitionPoint (fun s => decide (merged.getD s 0 < key)) 0 merged.length

/-- On a strictly increasing list containing `key`, `slot_of` finds it. -/
theorem slotOf_spec (l : List Nat) (hs : l.Pairwise (lt natLe)) (key : Nat) (hk : key ∈ l) :
    ∃ h : slotOf l key < l.length, l[slotOf l key] = key := by
  obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hk
  have strict : ∀ i j (hi : i < l.length) (hj : j < l.length), i < j → l[i] < l[j] := by
    intro i j hi hj hij
    have := List.pairwise_iff_getElem.mp hs i j hi hj hij
    simp only [lt, natLe, decide_eq_true_eq] at this; omega
  have hm : Mono (fun s => decide (l.getD s 0 < l[t])) 0 l.length := by
    intro i j _ hij hj hp
    simp only [decide_eq_true_eq] at hp ⊢
    rw [getD_of_lt _ _ _ hj] at hp; rw [getD_of_lt _ _ _ (by omega)]
    rcases Nat.lt_or_ge i j with h | h
    · have := strict i j (by omega) hj h; omega
    · have : i = j := by omega
      subst this; exact hp
  obtain ⟨-, p2, p3, p4⟩ := partitionPoint_spec _ 0 l.length (Nat.zero_le _) hm
  have hr : slotOf l l[t] = t := by
    unfold slotOf
    rcases Nat.lt_trichotomy (partitionPoint (fun s => decide (l.getD s 0 < l[t])) 0 l.length) t with h | h | h
    · have := p4 _ (Nat.le_refl _) (by omega)
      have h2 := strict _ t (by omega) ht h
      simp only [decide_eq_false_iff_not, Nat.not_lt] at this
      rw [getD_of_lt _ _ _ (by omega)] at this; omega
    · exact h
    · have := p3 t (Nat.zero_le _) h
      simp only [decide_eq_true_eq] at this; rw [getD_of_lt _ _ _ ht] at this; omega
  rw [hr]; exact ⟨ht, rfl⟩

theorem flatMap_take8 (M : List Nat) (min vw : Nat) (hw : W vw) :
    M.flatMap (fun v => (le 8 (v - min)).take vw) = (M.map (· - min)).flatMap (le vw) := by
  induction M with
  | nil => rfl
  | cons v M ih =>
    rw [List.flatMap_cons, List.map_cons, List.flatMap_cons, ih, le8_take _ _ (W_le8 _ hw)]

/-- `CompactIndexedLeaf::merge`, byte for byte. -/
def ciMerge (b : List Nat) (batch : List P) : List Nat :=
  let min := readU64 b 2
  let vw := byteAt b 10
  let dc := readU16 b 11
  let dw := byteAt b 13
  let docsOff := (layout b).2
  let oldDistinct := (List.range dc).map (slotValue b)
  let merged := mergeD natLe oldDistinct (batch.map (·.1)) none
  let out := mergeWI lexLe (ciPairs b) 0 batch none
  let index := out.map (fun e => slotOf merged e.1.1)
  let docs := out.flatMap (pickEnc (fun vi => slice b (docsOff + vi * dw) (docsOff + vi * dw + dw))
    (fun x => (le 8 x.2).take dw))
  header index.length min vw merged.length dw ++ merged.flatMap (fun v => (le 8 (v - min)).take vw) ++ index ++ docs

theorem lexLe_fst (a b : P) (h : lexLe a b = true) : a.1 ≤ b.1 := by
  simp only [lexLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h; omega

/-- **PROVEN** (`CompactIndexedLeaf::merge` = merge of the entry lists): with a
sorted distinct table and sorted leaf/batch entries that fit the page's widths,
the rebuilt page (new distinct table = sorted union of old values and batch
keys, slots found by `partition_point`) reads back as `merge_sorted(leaf, batch)`. -/
theorem ciMerge_spec (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hsort : Sorted dv)
    (hdcn : h.dc < h.n) (hLs : (ciDecode h.min dv idx docs).Pairwise (fun a b => lexLe a b = true))
    (batch : List P) (hbs : batch.Pairwise (fun a b => lexLe a b = true))
    (hfit : CFits h.min h.vw h.dw batch)
    (hlen : (mergeWI lexLe (ciDecode h.min dv idx docs) 0 batch none).length < 2 ^ 16)
    (hdl : (mergeD natLe (dv.map (h.min + ·)) (batch.map (·.1)) none).length < 2 ^ 16) :
    ciPairs (ciMerge (ciBytes h dv idx docs) batch) = mergeD lexLe (ciDecode h.min dv idx docs) batch none := by
  obtain ⟨r1, r2, r3, r4, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
  obtain ⟨-, -, sD, -⟩ := ciSlices h dv idx docs ok
  have hlay := ciLayout h dv idx docs ok hdcn
  have hdec := (ciRead h dv idx docs ok).2.2
  have hold : (List.range h.dc).map (slotValue (ciBytes h dv idx docs)) = dv.map (h.min + ·) := by
    apply List.ext_getElem (by simp [ok.dvlen])
    intro i h1 h2
    simp only [List.getElem_map, List.getElem_range]
    exact ci_value h dv idx docs ok i (by simpa using h1)
  unfold ciMerge
  simp only [r2, r3, r4, r5, hlay, hdec, hold]
  generalize hL : ciDecode h.min dv idx docs = L at *
  generalize hM : mergeD natLe (dv.map (h.min + ·)) (batch.map (·.1)) none = M at *
  generalize hO : mergeWI lexLe L 0 batch none = O at *
  have hOfst : O.map (·.1) = mergeD lexLe L batch none := by rw [← hO]; exact mergeWI_fst lexLe L 0 batch none
  -- facts about the leaf entries
  have hLmem : ∀ p ∈ L, (∃ s ∈ idx, p.1 = h.min + dv.getD s 0) ∧ p.2 ∈ docs := by
    intro p hp
    rw [← hL] at hp
    obtain ⟨sd, hsd, rfl⟩ := List.mem_map.mp hp
    exact ⟨⟨sd.1, (List.of_mem_zip hsd).1, rfl⟩, (List.of_mem_zip hsd).2⟩
  -- merged distinct table: strictly increasing union
  have hdvs : (dv.map (h.min + ·)).Pairwise (fun a b => natLe a b = true) := by
    have := List.pairwise_map.mpr (hsort.imp (fun {a b} (hab : a ≤ b) => show h.min + a ≤ h.min + b by omega))
    exact this.imp (fun hab => by simpa [natLe] using hab)
  have hbk : (batch.map (·.1)).Pairwise (fun a b => natLe a b = true) := by
    rw [List.pairwise_map]; exact hbs.imp (fun hab => by simpa [natLe] using lexLe_fst _ _ hab)
  obtain ⟨m1, -, m3, m4⟩ := mergeD_spec natLe natLe_lin _ _ none hdvs hbk (fun _ _ => rfl)
  rw [hM] at m1 m3 m4
  -- every emitted key is in the merged table
  have hOkeys : ∀ e ∈ O, e.1.1 ∈ M ∧ h.min ≤ e.1.1 := by
    intro e he
    have hx : e.1 ∈ mergeD lexLe L batch none := by rw [← hOfst]; exact List.mem_map_of_mem he
    have hLB : ∀ x, x ∈ mergeD lexLe L batch none → x ∈ L ∨ x ∈ batch :=
      (mergeD_spec lexLe lexLe_lin L batch none hLs hbs (fun _ _ => rfl)).1
    rcases hLB _ hx with hl | hb
    · obtain ⟨⟨s, hs, hk⟩, -⟩ := hLmem _ hl
      have hs' : s < dv.length := by rw [ok.dvlen]; exact ok.slots s hs
      refine ⟨?_, by rw [hk]; omega⟩
      rcases m3 e.1.1 (Or.inl (by rw [hk, getD_of_lt _ _ _ hs']; exact List.mem_map_of_mem (List.getElem_mem hs'))) with h' | h'
      · exact h'
      · cases h'
    · refine ⟨?_, (hfit _ hb).1⟩
      rcases m3 e.1.1 (Or.inr (List.mem_map_of_mem hb)) with h' | h'
      · exact h'
      · cases h'
  have hLB : ∀ x, x ∈ mergeD lexLe L batch none → x ∈ L ∨ x ∈ batch :=
    (mergeD_spec lexLe lexLe_lin L batch none hLs hbs (fun _ _ => rfl)).1
  have hOin : ∀ e ∈ O, e.1 ∈ L ∨ e.1 ∈ batch := fun e he =>
    hLB _ (by rw [← hOfst]; exact List.mem_map_of_mem he)
  have hLlen : L.length = h.n := by rw [← hL]; simp [ciDecode, ok.idxlen, ok.doclen]
  have hLdoc : ∀ i (hi : i < L.length), L[i].2 = docs[i]'(by rw [ok.doclen, ← hLlen]; exact hi) := by
    intro i hi; simp only [← hL, ciDecode, List.getElem_map, List.getElem_zip]
  -- the docs column
  have hdocs := flatMap_raw O L (fun (x : P) => le h.dw x.2)
    (fun vi => slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + h.n + vi * h.dw)
      (14 + h.dc * h.vw + h.n + vi * h.dw + h.dw))
    (fun i hi => by
      have := sD i (i + 1) (by omega) (by omega)
      simp only [Nat.succ_mul] at this
      have hi' : i < docs.length := by rw [ok.doclen, ← hLlen]; exact hi
      rw [show 14 + h.dc * h.vw + h.n + i * h.dw + h.dw = 14 + h.dc * h.vw + h.n + (i * h.dw + h.dw) by omega,
        this, show i + 1 - i = 1 by omega, hLdoc i hi]
      have hdrop : docs.drop i = docs[i] :: docs.drop (i + 1) := List.drop_eq_getElem_cons hi'
      rw [hdrop, List.take_succ_cons, List.take_zero, List.flatMap_cons, List.flatMap_nil, List.append_nil])
    (fun x i hx => by
      obtain ⟨h1, -, h3⟩ := mergeWI_idx lexLe L 0 batch none x i (by rw [hO]; exact hx)
      simp at h1 h3; exact ⟨h1, h3⟩)
    (fun (x : P) => (le 8 x.2).take h.dw) (fun x => le8_take _ _ (W_le8 _ ok.hok.2.2.2.2))
  rw [hdocs]
  have hvals : M.flatMap (fun v => (le 8 (v - h.min)).take h.vw) = (M.map (· - h.min)).flatMap (le h.vw) := by
    exact flatMap_take8 M h.min h.vw ok.hok.2.2.2.1
  rw [hvals]
  have hslots : ∀ e ∈ O, ∃ hs : slotOf M e.1.1 < M.length, M[slotOf M e.1.1] = e.1.1 := by
    intro e he
    exact slotOf_spec M m4 e.1.1 (hOkeys e he).1
  have ok' : CIOk ⟨(O.map (fun e => slotOf M e.1.1)).length, h.min, h.vw, M.length, h.dw⟩ (M.map (· - h.min))
      (O.map (fun e => slotOf M e.1.1)) (O.map (fun e => e.1.2)) := by
    refine ⟨⟨by simpa using hlen, hdl, ok.hok.2.2.1,
      ok.hok.2.2.2.1, ok.hok.2.2.2.2⟩, by simp, rfl, by simp, ?_, ?_, ?_⟩
    · intro sl hsl
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hsl
      exact (hslots e he).1
    · intro d hd
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hd
      rcases m1 v hv with h' | h'
      · obtain ⟨x, hx, rfl⟩ := List.mem_map.mp h'
        show h.min + x - h.min < 256 ^ h.vw
        rw [Nat.add_sub_cancel_left]; exact ok.dvfit x hx
      · obtain ⟨p, hp, rfl⟩ := List.mem_map.mp h'
        exact (hfit p hp).2.1
    · intro d hd
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
      rcases hOin e he with h' | h'
      · exact ok.docfit _ (hLmem _ h').2
      · exact (hfit _ h').2.2
  have e : header (O.map (fun e => slotOf M e.1.1)).length h.min h.vw M.length h.dw ++
      (M.map (· - h.min)).flatMap (le h.vw) ++ O.map (fun e => slotOf M e.1.1) ++
      O.flatMap (fun e => le h.dw e.1.2) =
      ciBytes ⟨(O.map (fun e => slotOf M e.1.1)).length, h.min, h.vw, M.length, h.dw⟩ (M.map (· - h.min))
        (O.map (fun e => slotOf M e.1.1)) (O.map (fun e => e.1.2)) := by
    simp [ciBytes, List.flatMap_map]
  rw [e, (ciRead _ _ _ _ ok').2.2, ← hOfst]
  unfold ciDecode
  rw [List.zip_map', List.map_map]
  apply List.map_congr_left
  intro x hx
  obtain ⟨hs, hv⟩ := hslots x hx
  simp only [Function.comp]
  rw [getD_of_lt _ _ _ (by simpa using hs), List.getElem_map, hv]
  have := (hOkeys x hx).2
  apply Prod.ext <;> simp <;> omega

end IndexLayer.Leaf
