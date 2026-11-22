import FalkorNativeIndex.CchSmall
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# `relax`'s sequential pair loop is the simultaneous update (`customize.rs:264-310`)

The Rust walks `up(x)` as `i`, then `j > i`, and for each pair `{y, z}` improves the upward arc `y -> z` through
`x` (`y -> x -> z`) and the downward arc `z -> y` (`z -> x -> y`), one pair after the other. `relaxA`/`relaxM`
(`CchMetric.lean`) apply every pair at once; this file shows the two agree, so the order of the pairs does not matter.
-/
namespace NI

/-- The pairs `(y, z)`, `y` before `z` in the list, in the loop's order (`for i { for j > i }`). -/
def pairsOf : List Nat → List (Nat × Nat)
  | [] => []
  | y :: r => r.map (fun z => (y, z)) ++ pairsOf r

/-- One iteration of the inner loop body for the pair `(y, z)` (reads `A y x`, `A x z`, `A z x`, `A x y`). -/
def seqPair (x : Nat) (st : (Nat → Nat → Wt) × (Nat → Nat → Option Nat)) (p : Nat × Nat) :
    (Nat → Nat → Wt) × (Nat → Nat → Option Nat) :=
  let y := p.1
  let z := p.2
  let A := st.1
  let M := st.2
  let c1 := wadd (A y x) (A x z)
  let c2 := wadd (A z x) (A x y)
  (fun a b => if a = y ∧ b = z then wmin (A y z) c1 else if a = z ∧ b = y then wmin (A z y) c2 else A a b,
   fun a b => if a = y ∧ b = z ∧ wlt c1 (A y z) = true then some x
     else if a = z ∧ b = y ∧ wlt c2 (A z y) = true then some x else M a b)

/-- Whether the ordered pair `(a, b)` is one of the pairs, in either order. -/
def covB (ps : List (Nat × Nat)) (a b : Nat) : Bool :=
  ps.any (fun p => (a == p.1 && b == p.2) || (a == p.2 && b == p.1))

theorem covB_cons (y z : Nat) (ps a b) : covB ((y, z) :: ps) a b =
    (((a == y && b == z) || (a == z && b == y)) || covB ps a b) := by
  simp [covB, List.any_cons]

/-- The fold over distinct pairs avoiding `x`: every covered ordered pair gets the detour, nothing else changes. -/
theorem seq_fold (x : Nat) : ∀ (ps : List (Nat × Nat)),
    ps.Pairwise (fun p q => p ≠ q ∧ p ≠ (q.2, q.1)) → (∀ p ∈ ps, p.1 ≠ p.2 ∧ p.1 ≠ x ∧ p.2 ≠ x) →
    ∀ (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat),
      (∀ a b, (ps.foldl (seqPair x) (A, M)).1 a b =
        if covB ps a b = true then wmin (A a b) (wadd (A a x) (A x b)) else A a b) ∧
      (∀ a b, (ps.foldl (seqPair x) (A, M)).2 a b =
        if covB ps a b = true ∧ wlt (wadd (A a x) (A x b)) (A a b) = true then some x else M a b) := by
  intro ps
  induction ps with
  | nil =>
    intro _ _ A M
    refine ⟨fun a b => ?_, fun a b => ?_⟩ <;> simp [covB]
  | cons p ps ih =>
    intro hp hne A M
    obtain ⟨y, z⟩ := p
    have hpw := List.pairwise_cons.1 hp
    obtain ⟨hyz, hyx, hzx⟩ := hne (y, z) (by simp)
    simp only at hyz hyx hzx
    obtain ⟨h1, h2⟩ := ih hpw.2 (fun q hq => hne q (List.mem_cons_of_mem _ hq)) (seqPair x (A, M) (y, z)).1
      (seqPair x (A, M) (y, z)).2
    simp only [List.foldl_cons]
    have hAx : ∀ a, (seqPair x (A, M) (y, z)).1 a x = A a x := by
      intro a; unfold seqPair; simp only
      rw [ifN (fun h => hzx h.2.symm), ifN (fun h => hyx h.2.symm)]
    have hxA : ∀ b, (seqPair x (A, M) (y, z)).1 x b = A x b := by
      intro b; unfold seqPair; simp only
      rw [ifN (fun h => hyx h.1.symm), ifN (fun h => hzx h.1.symm)]
    have hnot1 : covB ps y z = false := by
      cases hc : covB ps y z
      · rfl
      · exfalso
        obtain ⟨q, hq, hq2⟩ := List.any_eq_true.1 hc
        have := hpw.1 q hq
        simp at hq2
        rcases hq2 with ⟨e1, e2⟩ | ⟨e1, e2⟩
        · exact this.1 (by rw [e1, e2])
        · exact this.2 (by rw [e1, e2])
    have hnot2 : covB ps z y = false := by
      cases hc : covB ps z y
      · rfl
      · exfalso
        obtain ⟨q, hq, hq2⟩ := List.any_eq_true.1 hc
        have := hpw.1 q hq
        simp at hq2
        rcases hq2 with ⟨e1, e2⟩ | ⟨e1, e2⟩
        · exact this.2 (by rw [e1, e2])
        · exact this.1 (by rw [e1, e2])
    constructor
    · intro a b
      rw [h1 a b, hAx a, hxA b, covB_cons]
      by_cases hyz' : a = y ∧ b = z
      · obtain ⟨rfl, rfl⟩ := hyz'
        simp [hnot1, seqPair]
      · by_cases hzy' : a = z ∧ b = y
        · obtain ⟨rfl, rfl⟩ := hzy'
          simp [hnot2, seqPair, hyz.symm]
        · have e : (seqPair x (A, M) (y, z)).1 a b = A a b := by
            unfold seqPair; simp only; rw [ifN hyz', ifN hzy']
          have c : ((a == y && b == z) || (a == z && b == y)) = false := by
            simp only [Bool.or_eq_false_iff, Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
            constructor
            · by_cases h : a = y
              · right; intro hb; exact hyz' ⟨h, hb⟩
              · left; exact h
            · by_cases h : a = z
              · right; intro hb; exact hzy' ⟨h, hb⟩
              · left; exact h
          rw [c, Bool.false_or, e]
    · intro a b
      rw [h2 a b, hAx a, hxA b, covB_cons]
      have eA1 : (seqPair x (A, M) (y, z)).1 y z = wmin (A y z) (wadd (A y x) (A x z)) := by
        unfold seqPair; simp
      have eA2 : (seqPair x (A, M) (y, z)).1 z y = wmin (A z y) (wadd (A z x) (A x y)) := by
        unfold seqPair; simp [hyz.symm]
      have eM1 : (seqPair x (A, M) (y, z)).2 y z =
          if wlt (wadd (A y x) (A x z)) (A y z) = true then some x else M y z := by
        unfold seqPair; simp only
        by_cases hc : wlt (wadd (A y x) (A x z)) (A y z) = true
        · simp [hc]
        · simp [hc, hyz]
      have eM2 : (seqPair x (A, M) (y, z)).2 z y =
          if wlt (wadd (A z x) (A x y)) (A z y) = true then some x else M z y := by
        unfold seqPair; simp only
        by_cases hc : wlt (wadd (A z x) (A x y)) (A z y) = true
        · simp [hc, hyz.symm]
        · simp [hc, hyz.symm]
      by_cases hyz' : a = y ∧ b = z
      · obtain ⟨rfl, rfl⟩ := hyz'
        rw [hnot1, eA1, eM1]
        by_cases hc : wlt (wadd (A a x) (A x b)) (A a b) = true
        · simp [hc]
        · simp [hc]
      · by_cases hzy' : a = z ∧ b = y
        · obtain ⟨rfl, rfl⟩ := hzy'
          rw [hnot2, eA2, eM2]
          by_cases hc : wlt (wadd (A a x) (A x b)) (A a b) = true
          · simp [hc, hyz.symm]
          · simp [hc, hyz.symm]
        · have e : (seqPair x (A, M) (y, z)).1 a b = A a b := by
            unfold seqPair; simp only; rw [ifN hyz', ifN hzy']
          have eM : (seqPair x (A, M) (y, z)).2 a b = M a b := by
            unfold seqPair; simp only
            rw [ifN (fun h => hyz' ⟨h.1, h.2.1⟩), ifN (fun h => hzy' ⟨h.1, h.2.1⟩)]
          have c : ((a == y && b == z) || (a == z && b == y)) = false := by
            simp only [Bool.or_eq_false_iff, Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
            constructor
            · by_cases h : a = y
              · right; intro hb; exact hyz' ⟨h, hb⟩
              · left; exact h
            · by_cases h : a = z
              · right; intro hb; exact hzy' ⟨h, hb⟩
              · left; exact h
          rw [c, Bool.false_or, e, eM]

theorem pairsOf_mem : ∀ (l : List Nat) (p : Nat × Nat), p ∈ pairsOf l → p.1 ∈ l ∧ p.2 ∈ l
  | [], p, h => by simp [pairsOf] at h
  | y :: r, p, h => by
    simp only [pairsOf, List.mem_append, List.mem_map] at h
    rcases h with ⟨z, hz, rfl⟩ | h
    · simp [hz]
    · have := pairsOf_mem r p h
      exact ⟨List.mem_cons_of_mem _ this.1, List.mem_cons_of_mem _ this.2⟩

theorem pairsOf_pairwise : ∀ (l : List Nat), l.Nodup →
    (pairsOf l).Pairwise (fun p q => p ≠ q ∧ p ≠ (q.2, q.1))
  | [], _ => by simp [pairsOf]
  | y :: r, hn => by
    have hy := List.nodup_cons.1 hn
    unfold pairsOf
    rw [List.pairwise_append]
    refine ⟨?_, pairsOf_pairwise r hy.2, ?_⟩
    · rw [List.pairwise_map]
      exact List.Pairwise.imp (fun {a b} h => ⟨fun e => h.2.2 (by simpa using e), fun e => hy.1 (by
          have : y = b := (Prod.mk.inj e).1
          rw [this]; exact h.2.1)⟩) (List.Pairwise.and_mem.1 hy.2)
    · intro p hp q hq
      obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hp
      have := pairsOf_mem r q hq
      constructor
      · intro e; exact hy.1 ((Prod.mk.inj e).1 ▸ this.1)
      · intro e; exact hy.1 ((Prod.mk.inj e).1 ▸ this.2)

theorem pairsOf_ne : ∀ (l : List Nat), l.Nodup → ∀ p ∈ pairsOf l, p.1 ≠ p.2
  | [], _, p, h => by simp [pairsOf] at h
  | y :: r, hn, p, h => by
    have hy := List.nodup_cons.1 hn
    simp only [pairsOf, List.mem_append, List.mem_map] at h
    rcases h with ⟨z, hz, rfl⟩ | h
    · intro e; have e' : y = z := e; exact hy.1 (e' ▸ hz)
    · exact pairsOf_ne r hy.2 p h

theorem covB_of_mem : ∀ (l : List Nat) (a b : Nat), a ∈ l → b ∈ l → a ≠ b → covB (pairsOf l) a b = true
  | [], a, b, ha, _, _ => by simp at ha
  | y :: r, a, b, ha, hb, hab => by
    unfold pairsOf
    rw [covB]
    apply List.any_eq_true.2
    by_cases hay : a = y
    · subst hay
      have hbr : b ∈ r := by rcases List.mem_cons.1 hb with h | h; exact absurd h.symm hab; exact h
      exact ⟨(a, b), List.mem_append_left _ (List.mem_map.2 ⟨b, hbr, rfl⟩), by simp⟩
    · by_cases hby : b = y
      · subst hby
        have har : a ∈ r := by rcases List.mem_cons.1 ha with h | h; exact absurd h hay; exact h
        exact ⟨(b, a), List.mem_append_left _ (List.mem_map.2 ⟨a, har, rfl⟩), by simp⟩
      · have har : a ∈ r := by rcases List.mem_cons.1 ha with h | h; exact absurd h hay; exact h
        have hbr : b ∈ r := by rcases List.mem_cons.1 hb with h | h; exact absurd h hby; exact h
        obtain ⟨q, hq, h2⟩ := List.any_eq_true.1 (covB_of_mem r a b har hbr hab)
        exact ⟨q, List.mem_append_right _ hq, h2⟩

theorem covB_pairsOf (l : List Nat) (hn : l.Nodup) (a b : Nat) :
    covB (pairsOf l) a b = true ↔ a ∈ l ∧ b ∈ l ∧ a ≠ b := by
  constructor
  · intro h
    obtain ⟨q, hq, h2⟩ := List.any_eq_true.1 h
    have hm := pairsOf_mem l q hq
    have hne := pairsOf_ne l hn q hq
    simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at h2
    rcases h2 with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact ⟨hm.1, hm.2, hne⟩
    · exact ⟨hm.2, hm.1, fun e => hne e.symm⟩
  · rintro ⟨ha, hb, hab⟩
    exact covB_of_mem l a b ha hb hab

/-- **The sequential pair loop is the simultaneous update**: for `up(x)` strictly ascending (no repeats, `x` not in it),
    walking its pairs one after the other gives exactly `relaxA`/`relaxM`, so the order of the pairs is immaterial. -/
theorem relax_sequential (up : Nat → List Nat) (hgt : ∀ r w, w ∈ up r → r < w) (hs : ∀ r, (up r).Nodup) (x : Nat)
    (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat) :
    (∀ a b, ((pairsOf (up x)).foldl (seqPair x) (A, M)).1 a b = relaxA up x A a b) ∧
    (∀ a b, ((pairsOf (up x)).foldl (seqPair x) (A, M)).2 a b = relaxM up x A M a b) := by
  have hpw := pairsOf_pairwise (up x) (hs x)
  have hne : ∀ p ∈ pairsOf (up x), p.1 ≠ p.2 ∧ p.1 ≠ x ∧ p.2 ≠ x := by
    intro p hp
    have hm := pairsOf_mem (up x) p hp
    exact ⟨pairsOf_ne (up x) (hs x) p hp, fun e => by have := hgt x _ hm.1; omega,
      fun e => by have := hgt x _ hm.2; omega⟩
  obtain ⟨h1, h2⟩ := seq_fold x (pairsOf (up x)) hpw hne A M
  refine ⟨fun a b => ?_, fun a b => ?_⟩
  · rw [h1 a b]
    unfold relaxA
    by_cases hc : a ∈ up x ∧ b ∈ up x ∧ a ≠ b
    · rw [ifP ((covB_pairsOf _ (hs x) a b).2 hc), ifP hc]
    · rw [ifN (fun h => hc ((covB_pairsOf _ (hs x) a b).1 h)), ifN hc]
  · rw [h2 a b]
    unfold relaxM
    by_cases hc : a ∈ up x ∧ b ∈ up x ∧ a ≠ b ∧ wlt (wadd (A a x) (A x b)) (A a b) = true
    · rw [ifP ⟨(covB_pairsOf _ (hs x) a b).2 ⟨hc.1, hc.2.1, hc.2.2.1⟩, hc.2.2.2⟩, ifP hc]
    · rw [ifN, ifN hc]
      rintro ⟨h, h'⟩
      obtain ⟨h3, h4, h5⟩ := (covB_pairsOf _ (hs x) a b).1 h
      exact hc ⟨h3, h4, h5, h'⟩

/-! ## The hierarchy as a whole (`Cch`, `customize.rs:324-410`) -/

/-- `Cch`: the shared topology and the four per-arc arrays. -/
structure CchM where
  topo : Topo
  uw : List Wt
  dw : List Wt
  um : List (Option Nat)
  dm : List (Option Nat)

/-- `Cch::node_count`, `topology`, and the four accessors (`customize.rs:363-410`; an out-of-range slot is a panic,
    modelled by the default). -/
def CchM.nodeCount (c : CchM) : Nat := c.topo.nodeCount
def CchM.topology (c : CchM) : Topo := c.topo
def CchM.upwardWeight (c : CchM) (k : Nat) : Wt := c.uw.getD k none
def CchM.downwardWeight (c : CchM) (k : Nat) : Wt := c.dw.getD k none
def CchM.upwardMiddle (c : CchM) (k : Nat) : Option Nat := c.um.getD k none
def CchM.downwardMiddle (c : CchM) (k : Nat) : Option Nat := c.dm.getD k none

/-- `Cch::customized`: the same topology (shared), a new metric. -/
def CchM.customized (c : CchM) (arrs : List Wt × List Wt × List (Option Nat) × List (Option Nat)) : CchM :=
  ⟨c.topo, arrs.1, arrs.2.1, arrs.2.2.1, arrs.2.2.2⟩

theorem customized_shares (c : CchM) (arrs) :
    (c.customized arrs).topo = c.topo ∧ (c.customized arrs).nodeCount = c.nodeCount := ⟨rfl, rfl⟩

theorem accessor_in_range (c : CchM) (k : Nat) (e : Wt) (h : c.uw[k]? = some e) : c.upwardWeight k = e := by
  unfold CchM.upwardWeight; rw [List.getD_eq_getElem?_getD, h]; rfl

/-- `Cch::build`: the topology, then customization of the weights read in rank space
    (`w r r' = weight (node_of r) (node_of r')`), stored in the slot arrays. -/
def buildCch (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (skel : Cch.Csr) (wN : Nat → Nat → Wt) : Option CchM :=
  (topoBuild backend bs skel).bind (fun t =>
    (invert t.r2n skel.n).map (fun n2r =>
      let adj := adjR skel t.r2n n2r
      let r := customizeRun (rowOf adj) (fun r1 r2 => wN (t.nodeOf r1) (t.nodeOf r2)) skel.n
      let arrs := metricArrays (buildRows adj skel.n) skel.n r.1 r.2
      ⟨t, arrs.1, arrs.2.1, arrs.2.2.1, arrs.2.2.2⟩))

/-- **`Cch::build` succeeds only with a permutation order, and every arc's slot holds the customized weights**: for a
    successful build, `r2n` is a permutation of the nodes, `up(r)` the fill-in, and for each arc `a < b` of the chordal
    rows the slot `arc_slot` finds (from either order) holds `A a b` and `M a b` in the upward arrays and `A b a`, `M b a`
    in the downward ones, with `A`/`M` as `customize_correct` characterises. -/
theorem buildCch_correct (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (hbs : Cch.BsSpec bs) (bsi : List Nat → Nat → Option Nat) (hbsi : BsIdxSpec bsi)
    (skel : Cch.Csr) (wN : Nat → Nat → Wt) (c : CchM) (h : buildCch backend bs skel wN = some c) :
    ∃ n2r, c.topo.r2n.Perm (List.range skel.n) ∧ invert c.topo.r2n skel.n = some n2r ∧
      c.nodeCount = skel.n ∧
      let adj := adjR skel c.topo.r2n n2r
      let r := customizeRun (rowOf adj) (fun r1 r2 => wN (c.topo.nodeOf r1) (c.topo.nodeOf r2)) skel.n
      ∀ a b, a < b → a < skel.n → b ∈ rowOf adj a →
        ∃ k, c.topo.arcSlot bsi a b = some k ∧ c.topo.arcSlot bsi b a = some k ∧
          c.uw[k]? = some (r.1 a b) ∧ c.dw[k]? = some (r.1 b a) ∧
          c.um[k]? = some (r.2 a b) ∧ c.dm[k]? = some (r.2 b a) := by
  unfold buildCch at h
  cases ht : topoBuild backend bs skel with
  | none => rw [ht] at h; simp at h
  | some t =>
    rw [ht] at h
    obtain ⟨n2r0, hperm, hinv, hnc, hfill⟩ := topoBuild_correct backend bs hbs skel t ht
    simp only [Option.bind_some, hinv, Option.map_some, Option.some.injEq] at h
    subst h
    refine ⟨n2r0, hperm, hinv, hnc, ?_⟩
    intro adj r a b hab han hb
    -- the topology came from `fromOrder`
    unfold topoBuild at ht
    cases ho : ndOrder backend bs skel with
    | none => rw [ho] at ht; simp at ht
    | some o =>
      rw [ho] at ht
      simp only [Option.bind_some] at ht
      obtain ⟨hw, _, _⟩ := ndOrder_checks backend bs skel o ho
      have hne : fromOrder skel o ≠ none := by rw [ht]; simp
      rw [Ne, fromOrder_none_iff, Classical.not_not] at hne
      obtain ⟨_, hl, hnd, hlt⟩ := hne
      obtain ⟨n2r, hinv'⟩ : ∃ n2r, invert o skel.n = some n2r := by
        cases hi : invert o skel.n with
        | none => exact absurd hi (by rw [invert_none_iff, Classical.not_not]; exact ⟨hl, hnd, hlt⟩)
        | some n2r => exact ⟨n2r, rfl⟩
      have hfe := fromOrder_eq skel o n2r hw hinv'
      rw [ht] at hfe
      have htt := Option.some.inj hfe
      subst htt
      have hn2 : n2r = n2r0 := by
        have := hinv; simp only at this; rw [hinv'] at this; exact Option.some.inj this
      subst hn2
      have hsorted : ∀ r', r' < skel.n → ((buildRows (adjR skel o n2r) skel.n).getD r' []).Pairwise (· < ·) := by
        intro r' hr'
        rw [rows_getD _ _ _ hr']
        exact rowOf_sorted _ r'
      have hb' : b ∈ (buildRows (adjR skel o n2r) skel.n).getD a [] := by
        rw [rows_getD _ _ _ han]; exact hb
      exact arc_via_slots bsi hbsi (buildRows (adjR skel o n2r) skel.n) skel.n (buildRows_length _ _)
        r.1 r.2 _ rfl rfl hsorted a b hab han hb'

end NI
