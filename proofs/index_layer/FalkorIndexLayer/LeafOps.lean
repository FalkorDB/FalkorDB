import FalkorIndexLayer.LeafBlockCopy2
/-
# `Leaf::merge_batch` (`cow_btree/leaf/mod.rs:471`, origin/main 8743953a8):
every fast path agrees with the slow path (`merge_sorted` + `chunks` + `from_pairs`).
-/
namespace IndexLayer.Leaf

/-- `slice::chunks(n)`. -/
def chunks (n : Nat) (l : List P) : List (List P) :=
  if h : n = 0 then [l] else
  if l = [] then [] else l.take n :: chunks n (l.drop n)
termination_by l.length
decreasing_by
  simp_wf
  have h1 : l.length ≠ 0 := by simpa using ‹¬l = []›
  omega

theorem chunks_spec (n : Nat) (hn : 0 < n) : ∀ (l : List P),
    (chunks n l).flatten = l ∧ ∀ c ∈ chunks n l, c.length ≤ n ∧ c.Sublist l := by
  intro l
  induction h : l.length using Nat.strongRecOn generalizing l with
  | _ m ih =>
    rw [chunks]
    simp only [show n ≠ 0 by omega, dite_false]
    split
    · next he => subst he; simp
    · next he =>
      have hlt : (l.drop n).length < m := by
        have : l.length ≠ 0 := by simpa using he
        simp; omega
      obtain ⟨i1, i2⟩ := ih _ hlt (l.drop n) rfl
      refine ⟨by simp [i1], fun c hc => ?_⟩
      rcases List.mem_cons.mp hc with rfl | hc
      · exact ⟨by simp; omega, List.take_sublist _ _⟩
      · obtain ⟨a, b⟩ := i2 c hc
        exact ⟨a, b.trans (List.drop_sublist _ _)⟩

/-- Well-formed pages, as every constructor produces them. -/
def WFLeaf : LeafV → Prop
  | .aos b => ∃ ps, b = aosBuild ps ∧ AllU64 ps
  | .compact b => ∃ ps min vw dw, b = cBuild ps min vw dw ∧ ps.length < 2 ^ 16 ∧ U64 min ∧ W vw ∧ W dw ∧
      CFits min vw dw ps
  | .indexed b => ∃ h dv idx docs, b = ciBytes h dv idx docs ∧ CIOk h dv idx docs ∧ Sorted dv ∧ h.dc < h.n

def allFits (b : List Nat) (batch : List P) : Bool := batch.all (fun p => packingFits b p.1 p.2)
def allPresent (b : List Nat) (batch : List P) : Bool :=
  batch.all (fun p => match distinctSlot b p.1 with | .ok _ => true | .err _ => false)

/-- `Leaf::merge_batch`. -/
def mergeBatch (leafMax : Nat) (l : LeafV) (batch : List P) : List LeafV :=
  let slow := (chunks leafMax (mergeD lexLe l.toPairs batch none)).map fromPairs
  if l.count + batch.length ≤ leafMax then
    match l with
    | .aos b => [.aos (aosMerge b batch)]
    | .compact b => if allFits b batch then [.compact (cMerge b batch)] else slow
    | .indexed b =>
      if allFits b batch then
        (if allPresent b batch then [.indexed (ciBcm b batch)] else [.indexed (ciMerge b batch)])
      else slow
  else slow

theorem length_mergeD {α : Type} [DecidableEq α] (le : α → α → Bool) :
    ∀ (l r : List α) (last : Option α), (mergeD le l r last).length ≤ l.length + r.length
  | [], [], _ => by simp [mergeD]
  | a :: as, [], l => by
    rw [mergeD]; unfold emit; have := length_mergeD le as [] (some a)
    split <;> simp only [List.length_cons, List.length_nil] at this ⊢ <;> omega
  | [], b :: bs, l => by
    rw [mergeD]; unfold emit; have := length_mergeD le [] bs (some b)
    split <;> simp only [List.length_cons, List.length_nil] at this ⊢ <;> omega
  | a :: as, b :: bs, l => by
    rw [mergeD]
    split
    · unfold emit; have := length_mergeD le as (b :: bs) (some a)
      split <;> simp only [List.length_cons, List.length_nil] at this ⊢ <;> omega
    · unfold emit; have := length_mergeD le (a :: as) bs (some b)
      split <;> simp only [List.length_cons, List.length_nil] at this ⊢ <;> omega
termination_by l r => l.length + r.length

theorem mergeWI_length {α : Type} [DecidableEq α] (le : α → α → Bool) (l : List α) (k : Nat) (r : List α)
    (last : Option α) : (mergeWI le l k r last).length = (mergeD le l r last).length := by
  rw [← mergeWI_fst le l k r last, List.length_map]

/-- **PROVEN** (`Leaf::merge_batch`): whichever path runs — AoS byte merge,
compact merge, indexed block-copy or rebuild merge, or the slow
`merge_sorted` + `chunks(LEAF_MAX)` + `from_pairs` path — the resulting pages,
read back in order, hold exactly `merge_sorted(leaf, batch)`. -/
theorem mergeBatch_spec (leafMax : Nat) (hm1 : 0 < leafMax) (hm2 : leafMax ≤ 256) (l : LeafV) (hwf : WFLeaf l)
    (hLs : l.toPairs.Pairwise (lt lexLe)) (hLu : AllU64 l.toPairs) (hcount : l.count = l.toPairs.length)
    (batch : List P) (hbs : batch.Pairwise (fun a b => lexLe a b = true)) (hbu : AllU64 batch) :
    ((mergeBatch leafMax l batch).map LeafV.toPairs).flatten = mergeD lexLe l.toPairs batch none := by
  have hL' : l.toPairs.Pairwise (fun a b => lexLe a b = true) := hLs.imp (fun h => h.1)
  obtain ⟨m1, -, -, m4⟩ := mergeD_spec lexLe lexLe_lin _ _ none hL' hbs (fun _ _ => rfl)
  have hmu : AllU64 (mergeD lexLe l.toPairs batch none) := fun p hp =>
    (m1 p hp).elim (fun h => hLu p h) (fun h => hbu p h)
  have slow : ((chunks leafMax (mergeD lexLe l.toPairs batch none)).map fromPairs).map LeafV.toPairs =
      chunks leafMax (mergeD lexLe l.toPairs batch none) := by
    rw [List.map_map]
    conv => rhs; rw [← List.map_id (chunks leafMax (mergeD lexLe l.toPairs batch none))]
    apply List.map_congr_left
    intro c hc
    obtain ⟨-, hc2⟩ := chunks_spec leafMax hm1 _
    obtain ⟨hlen, hsub⟩ := hc2 c hc
    exact fromPairs_roundtrip c (m4.sublist hsub) (fun p hp => hmu p (hsub.subset hp)) (by omega)
  have hslow : ((chunks leafMax (mergeD lexLe l.toPairs batch none)).map fromPairs |>.map LeafV.toPairs).flatten =
      mergeD lexLe l.toPairs batch none := by
    rw [slow]; exact (chunks_spec leafMax hm1 _).1
  have hml := length_mergeD lexLe l.toPairs batch none
  unfold mergeBatch
  simp only
  split
  · next hfit =>
    cases l with
    | aos b =>
      obtain ⟨ps, rfl, hpu⟩ := hwf
      dsimp only
      have e := (aos_roundtrip ps hpu).2.2
      simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
      rw [show (LeafV.aos (aosBuild ps)).toPairs = ps from e] at hmu ⊢
      rw [aosMerge_spec ps batch hpu]
      exact (aos_roundtrip _ hmu).2.2
    | compact b =>
      obtain ⟨ps, min, vw, dw, rfl, hn, hmin, hvw, hdw, hf⟩ := hwf
      dsimp only
      have e : (LeafV.compact (cBuild ps min vw dw)).toPairs = ps := cBuild_roundtrip ps min vw dw hn hmin hvw hdw hf
      rw [e] at hmu hml hcount m1 hslow ⊢
      split
      · next hall =>
        obtain ⟨r1, r2, r3, -, r5⟩ := cBytes_hdr ⟨ps.length, min, vw, ps.length, dw⟩ ⟨hn, hn, hmin, hvw, hdw⟩
          (ps.map (fun p => p.1 - min)) (ps.map (·.2))
        have hbf : CFits min vw dw batch := by
          intro p hp
          have := List.all_eq_true.mp hall p hp
          have := packingFits_spec (cBuild ps min vw dw) p.1 p.2 (hbu p hp).1 (hbu p hp).2
            (by simp only [cBuild, r3]; exact hvw) (by simp only [cBuild, r5]; exact hdw) this
          simp only [cBuild, r2, r3, r5] at this
          exact this
        simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
        rw [cMerge_spec ps batch min vw dw hn hmin hvw hdw hf hbf]
        have hcf : CFits min vw dw (mergeD lexLe ps batch none) := fun p hp => by
          rcases m1 p hp with h | h
          · exact hf p h
          · exact hbf p h
        exact cBuild_roundtrip _ min vw dw (by rw [hcount] at hfit; omega) hmin hvw hdw hcf
      · exact hslow
    | indexed b =>
      obtain ⟨h, dv, idx, docs, rfl, ok, hsort, hdcn⟩ := hwf
      dsimp only
      have e : (LeafV.indexed (ciBytes h dv idx docs)).toPairs = ciDecode h.min dv idx docs := (ciRead h dv idx docs ok).2.2
      rw [e] at hmu hml hLs hL' hcount m1 hslow ⊢
      split
      · next hall =>
        obtain ⟨-, r2, r3, -, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
        have hbf : CFits h.min h.vw h.dw batch := by
          intro p hp
          have := List.all_eq_true.mp hall p hp
          have := packingFits_spec (ciBytes h dv idx docs) p.1 p.2 (hbu p hp).1 (hbu p hp).2
            (by rw [r3]; exact ok.hok.2.2.2.1) (by rw [r5]; exact ok.hok.2.2.2.2) this
          rw [r2, r3, r5] at this
          exact this
        have hcnt : (mergeD lexLe (ciDecode h.min dv idx docs) batch none).length < 2 ^ 16 := by
          rw [hcount] at hfit; omega
        split
        · next hpres =>
          simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
          have hp : ∀ x ∈ batch, ∃ s, ∃ hs : s < dv.length, h.min + dv[s] = x.1 := by
            intro x hx
            have := List.all_eq_true.mp hpres x hx
            obtain ⟨dsOk, -⟩ := distinctSlot_spec h dv idx docs ok hsort x.1
            cases hd : distinctSlot (ciBytes h dv idx docs) x.1 with
            | ok s => obtain ⟨hs, hv⟩ := dsOk s hd; exact ⟨s, hs, hv⟩
            | err s => rw [hd] at this; cases this
          show (ciPairs (ciBcm (ciBytes h dv idx docs) batch)) = _
          exact ciBcm_spec h dv idx docs ok hsort hdcn hLs batch hbs hbf hp
            (by rw [← List.length_map (f := (·.1)), bcmL_eq_merge _ _ hLs hbs]; exact hcnt)
        · simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
          show (ciPairs (ciMerge (ciBytes h dv idx docs) batch)) = _
          exact ciMerge_spec h dv idx docs ok hsort hdcn hL' batch hbs hbf
            (by rw [mergeWI_length]; exact hcnt)
            (by
              have := length_mergeD natLe (dv.map (h.min + ·)) (batch.map (·.1)) none
              have hcn : (LeafV.indexed (ciBytes h dv idx docs)).count = h.n := (ciRead h dv idx docs ok).1
              rw [hcn] at hfit
              have hdv : dv.length < h.n := by rw [ok.dvlen]; exact hdcn
              simp only [List.length_map] at this; omega)
      · exact hslow
  · exact hslow

end IndexLayer.Leaf
