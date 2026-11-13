import CchCoarsenReview.Hkey
/-!
# The matching rounds: the mutual rule, `usel`, `zmatch`, and the 2-hop pass

**Rust** (m1opt.patch @7df8e5887): `coarsen_level`, m1_coarsen_bench.rs:830-1026.

Each round (:851-937): `y = mxv(MAX_FIRST, K, u)` under `u`'s structure with replace (:855-858), so
`y(i)` is stored iff `i` is an unmatched proposer with an unmatched `K`-neighbour, and is then the
largest key in its row among those; `y = unkey(y)` (:866-869) turns it into that neighbour's column
(`Hkey.unkey_key`). The mutual test `m(i) = (pf(pf(i)) == i)` on `y`'s pattern, with
`pf = eWiseAdd(SECOND, idx, y)` (:891-908); `mate⟨m⟩ = y` (:909-914); the next proposers are `y`'s
pattern where `m` is not true (:915-929).

* **`usel`** (:915-921) builds the next proposers as `select(m == false)`, a `bool` vector, instead of
  `apply(identity, y)⟨¬m, replace⟩` (:922-929). `mxv` reads `u` only as a structural mask and through
  `FIRST`, which ignores `u`'s values, so only the pattern matters (`usel_same_pattern`).
* **`zmatch`** (:873-890) keeps no `mate` during the rounds: `Z = eWiseAdd(SECOND, Z, y)` (every
  proposal written, :874-880), the mutual test reads `Z(Z(i))` (:881-890); after the loop
  `mate = Z` except where `Z(Z(i)) ≠ i`, which go back to `i` (:938-959).
* **2-hop** (:963-1021): unmatched vertices grouped by their heaviest neighbour (the anchor), each
  group sorted, consecutive members paired, pairs over the cap dropped on the host (:1008-1015), and
  `mate⟨tmp⟩ = tmp` with both directions (:1016-1019).

The proposals are abstracted to a function `prop U` of the proposer set with the `mxv` contract
(`PropSpec`); `prop_of_mxv` derives it from the `MAX_FIRST` contract and `unkey_key`.
-/

namespace CchCoarsen

/-- The proposals of one round as a function of the proposer pattern `U` (after `unkey`). -/
abbrev Prop' := (Nat → Bool) → Nat → Option Nat

/-- The `mxv` + `unkey` contract: a proposal goes to an unmatched `K`-neighbour, and every unmatched
proposer with an unmatched `K`-neighbour proposes. -/
structure PropSpec (K : Nat → Nat → Prop) (prop : Prop') : Prop where
  target : ∀ U i j, prop U i = some j → U i = true ∧ U j = true ∧ K i j
  stored : ∀ U i, (prop U i).isSome = true ↔ (U i = true ∧ ∃ j, U j = true ∧ K i j)

/-- `K`'s pattern is symmetric (`A` is, and the cap test `w(i)+w(j) ≤ cap` is) and has no diagonal. -/
structure KSpec (K : Nat → Nat → Prop) : Prop where
  symm : ∀ i j, K i j → K j i
  irrefl : ∀ i, ¬ K i i

/-- The `MAX_FIRST` `mxv` contract on the key matrix `Kv` (C API: `MAX` over the stored products,
stored iff some product is), then `unkey`. -/
structure MxvSpec (Kv : Nat → Nat → Option (BitVec 64)) (mx : (Nat → Bool) → Nat → Option (BitVec 64)) : Prop where
  val : ∀ U i k, mx U i = some k → U i = true ∧ ∃ j, U j = true ∧ Kv i j = some k
  stored : ∀ U i, (mx U i).isSome = true ↔ (U i = true ∧ ∃ j, U j = true ∧ (Kv i j).isSome = true)

/-- **The proposal contract follows from `mxv`'s and the key encoding**: the `unkey` of the maximum key
is a column that holds it. -/
theorem prop_of_mxv {Kv : Nat → Nat → Option (BitVec 64)} {mx} (hm : MxvSpec Kv mx) (salt : Nat)
    (hk : ∀ i j k, Kv i j = some k → j < 2 ^ 32 ∧ ∃ v, k = keyOf salt v j) :
    PropSpec (fun i j => (Kv i j).isSome = true)
      (fun U i => (mx U i).map (fun k => (unkeyOf salt k).toNat)) where
  target := by
    intro U i j h
    simp only [Option.map_eq_some_iff] at h
    obtain ⟨k, hk1, rfl⟩ := h
    obtain ⟨hu, j', hj', hkv⟩ := hm.val U i k hk1
    obtain ⟨hjlt, v, rfl⟩ := hk i j' _ hkv
    rw [unkey_key]
    have : ((BitVec.ofNat 32 j').setWidth 64).toNat = j' := by
      simp [BitVec.toNat_setWidth, BitVec.toNat_ofNat]; omega
    rw [this]
    exact ⟨hu, hj', by simp [hkv]⟩
  stored := by
    intro U i
    simp only [Option.isSome_map]
    exact hm.stored U i

/-! ## The baseline rounds -/

/-- `pf = eWiseAdd(SECOND, idx, y)`: the proposal, or the vertex itself. -/
def pf (y : Nat → Option Nat) (k : Nat) : Nat := (y k).getD k

/-- `m(i) = (pf(pf(i)) == i)` on `y`'s pattern. -/
def mutB (y : Nat → Option Nat) (i : Nat) : Bool := (y i).isSome && (pf y (pf y i) == i)

structure BState where
  mate : Nat → Nat
  U : Nat → Bool

/-- One baseline round: `mate⟨m⟩ = y`; next proposers = `y`'s pattern where `m` is not true. -/
def stepB (prop : Prop') (s : BState) : BState :=
  let y := prop s.U
  { mate := fun i => if mutB y i = true then (y i).getD i else s.mate i
    U := fun i => (y i).isSome && !(mutB y i) }

/-- `r` rounds (the loop's exit depends only on `y` and the next `U`, which `zmatch` reproduces). -/
def iterB (prop : Prop') (U0 : Nat → Bool) : Nat → BState
  | 0 => ⟨id, U0⟩
  | r + 1 => stepB prop (iterB prop U0 r)

/-- The baseline invariant: `mate` an involution, matched pairs are `K` edges, proposers unmatched. -/
structure BInv (K : Nat → Nat → Prop) (s : BState) : Prop where
  invol : ∀ i, s.mate (s.mate i) = i
  edge : ∀ i, s.mate i ≠ i → K i (s.mate i)
  free : ∀ i, s.U i = true → s.mate i = i

variable {K : Nat → Nat → Prop} {prop : Prop'}

/-- A mutual proposal is mutual both ways, between two distinct vertices. -/
theorem mut_pair (hK : KSpec K) (hP : PropSpec K prop) (U : Nat → Bool) {i : Nat}
    (h : mutB (prop U) i = true) :
    ∃ j, prop U i = some j ∧ prop U j = some i ∧ mutB (prop U) j = true ∧ j ≠ i := by
  simp only [mutB, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨hs, hpf⟩ := h
  obtain ⟨j, hj⟩ := Option.isSome_iff_exists.1 hs
  have hKij := (hP.target U i j hj).2.2
  have hji : j ≠ i := fun e => hK.irrefl i (e ▸ hKij)
  simp only [pf, hj, Option.getD_some] at hpf
  cases hyj : prop U j with
  | none => rw [hyj] at hpf; simp at hpf; exact absurd hpf hji
  | some k =>
    rw [hyj] at hpf; simp only [Option.getD_some] at hpf; subst hpf
    refine ⟨j, hj, hyj, ?_, hji⟩
    simp [mutB, pf, hyj, hj]

theorem stepB_inv (hK : KSpec K) (hP : PropSpec K prop) {s : BState} (h : BInv K s) :
    BInv K (stepB prop s) := by
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_⟩
  · simp only [stepB]
    by_cases hm : mutB (prop s.U) i = true
    · obtain ⟨j, hj, hji, hmj, _⟩ := mut_pair hK hP s.U hm
      simp [hm, hj, hmj, hji]
    · simp only [hm, Bool.false_eq_true, ite_false]
      by_cases hk : s.mate i = i
      · rw [hk]; simp [hm, hk]
      · have hnm : mutB (prop s.U) (s.mate i) ≠ true := by
          intro hmk
          obtain ⟨_, hy, _, _, _⟩ := mut_pair hK hP s.U hmk
          have := h.free _ (hP.target s.U _ _ hy).1
          rw [h.invol] at this; exact hk this.symm
        simp [hnm, h.invol]
  · simp only [stepB] at hi ⊢
    by_cases hm : mutB (prop s.U) i = true
    · obtain ⟨j, hj, _, _, _⟩ := mut_pair hK hP s.U hm
      simp only [hm, ite_true, hj, Option.getD_some]
      exact (hP.target s.U i j hj).2.2
    · simp only [hm, Bool.false_eq_true, ite_false] at hi ⊢
      exact h.edge i hi
  · simp only [stepB, Bool.and_eq_true, Bool.not_eq_true'] at hi ⊢
    obtain ⟨hs, hm⟩ := hi
    simp only [hm, Bool.false_eq_true, ite_false]
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.1 hs
    exact h.free i (hP.target s.U i j hj).1

/-- **After any number of rounds, `mate` is an involution and every matched pair is a `K` edge** (an
edge of `A` within the cap: `keys_K`). -/
theorem iterB_inv (hK : KSpec K) (hP : PropSpec K prop) (U0 : Nat → Bool) (r : Nat) :
    BInv K (iterB prop U0 r) := by
  induction r with
  | zero => exact ⟨fun _ => rfl, fun i h => absurd rfl h, fun _ _ => rfl⟩
  | succ r ih => exact stepB_inv hK hP ih

/-! ## `usel`: the next proposers by `select(m == false)` -/

/-- `m` is stored on `y`'s pattern. `usel`'s `select(VALUEEQ_BOOL, m, false)` and the baseline's
`apply(identity, y)⟨¬m, GrB_DESC_RC⟩` have the same pattern, and the next `mxv` reads only that. -/
theorem usel_same_pattern (y : Nat → Option Nat) (m : Nat → Option Bool)
    (hdom : ∀ i, (m i).isSome = (y i).isSome) (i : Nat) :
    (m i = some false) ↔ ((y i).isSome = true ∧ m i ≠ some true) := by
  cases hm : m i with
  | none => have := hdom i; rw [hm] at this; simp [← this]
  | some b => have := hdom i; rw [hm] at this; cases b <;> simp [← this]

/-! ## `zmatch` -/

structure ZState where
  Z : Nat → Nat
  U : Nat → Bool

/-- One `zmatch` round: `Z = eWiseAdd(SECOND, Z, y)`; `m(i) = (Z(Z(i)) == i)` on `y`'s pattern. -/
def stepZ (prop : Prop') (s : ZState) : ZState :=
  let y := prop s.U
  let Z' := fun i => (y i).getD (s.Z i)
  { Z := Z', U := fun i => (y i).isSome && !(Z' (Z' i) == i) }

def iterZ (prop : Prop') (U0 : Nat → Bool) : Nat → ZState
  | 0 => ⟨id, U0⟩
  | r + 1 => stepZ prop (iterZ prop U0 r)

/-- `z_final` (:938-959): `mate = Z` except where `Z(Z(i)) ≠ i`. -/
def finalZ (Z : Nat → Nat) (i : Nat) : Nat := if Z (Z i) = i then Z i else i

/-- The relation between the two paths after the same rounds. -/
structure Rel (b : BState) (z : ZState) : Prop where
  U : b.U = z.U
  matched : ∀ i, b.mate i ≠ i → z.Z i = b.mate i
  stale : ∀ i, b.mate i = i → z.Z i = i ∨ z.Z (z.Z i) ≠ i

/-- A proposal's target also proposes (it is unmatched, and the proposer is its unmatched neighbour). -/
theorem target_proposes (hK : KSpec K) (hP : PropSpec K prop) (U : Nat → Bool) {i j : Nat}
    (h : prop U i = some j) : (prop U j).isSome = true := by
  obtain ⟨hu, hju, hk⟩ := hP.target U i j h
  exact (hP.stored U j).2 ⟨hju, i, hu, hK.symm i j hk⟩

theorem step_rel (hK : KSpec K) (hP : PropSpec K prop) {b : BState} {z : ZState} (hb : BInv K b)
    (h : Rel b z) : Rel (stepB prop b) (stepZ prop z) := by
  have hU := h.U
  -- the two mutual tests agree on `y`'s pattern
  have hm : ∀ i, (prop b.U i).isSome = true →
      (((prop b.U i).getD (z.Z i)) |> fun k => (prop b.U k).getD (z.Z k)) = pf (prop b.U) (pf (prop b.U) i) := by
    intro i hs
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.1 hs
    obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 (target_proposes hK hP b.U hj)
    simp [pf, hj, hl]
  refine ⟨?_, fun i hi => ?_, fun i hi => ?_⟩
  · funext i
    simp only [stepB, stepZ, ← hU, mutB]
    by_cases hs : (prop b.U i).isSome = true
    · have := hm i hs; simp only at this; simp [hs, this]
    · simp [hs]
  · simp only [stepB, stepZ, ← hU] at hi ⊢
    by_cases hmu : mutB (prop b.U) i = true
    · obtain ⟨j, hj, _, _, _⟩ := mut_pair hK hP b.U hmu
      simp [hmu, hj]
    · simp only [hmu, Bool.false_eq_true, ite_false] at hi ⊢
      have hns : prop b.U i = none := by
        cases hy : prop b.U i with
        | none => rfl
        | some j => exact absurd (hb.free i (hP.target b.U i j hy).1) hi
      simp [hns, h.matched i hi]
  · simp only [stepB, stepZ, ← hU] at hi ⊢
    by_cases hmu : mutB (prop b.U) i = true
    · obtain ⟨j, hj, _, _, hji⟩ := mut_pair hK hP b.U hmu
      simp [hmu, hj] at hi; exact absurd hi hji
    · simp only [hmu, Bool.false_eq_true, ite_false] at hi
      cases hy : prop b.U i with
      | some j =>
        -- proposed and not mutual: `Z'(Z'(i)) = pf(pf(i)) ≠ i`
        right
        have hs : (prop b.U i).isSome = true := by simp [hy]
        have := hm i hs
        simp only [hy] at this ⊢
        rw [this]
        simp only [mutB, hs, Bool.true_and, beq_iff_eq] at hmu
        exact hmu
      | none =>
        simp only [Option.getD_none]
        rcases h.stale i hi with hz | hz
        · left; exact hz
        · right
          cases hk : prop b.U (z.Z i) with
          | none => simp; exact hz
          | some l =>
            simp only [Option.getD_some]
            intro hl; subst hl
            exact absurd (target_proposes hK hP b.U hk) (by simp [hy])

theorem iter_rel (hK : KSpec K) (hP : PropSpec K prop) (U0 : Nat → Bool) (r : Nat) :
    Rel (iterB prop U0 r) (iterZ prop U0 r) := by
  induction r with
  | zero => exact ⟨rfl, fun i h => absurd rfl h, fun i _ => Or.inl rfl⟩
  | succ r ih => exact step_rel hK hP (iterB_inv hK hP U0 r) ih

/-- **`zmatch` = the baseline**: after the same rounds the proposer sets are equal (so every round's
`y`, counts and exit test agree), and `z_final` gives the baseline's `mate`. -/
theorem zmatch_eq (hK : KSpec K) (hP : PropSpec K prop) (U0 : Nat → Bool) (r : Nat) :
    (iterZ prop U0 r).U = (iterB prop U0 r).U ∧
    finalZ (iterZ prop U0 r).Z = (iterB prop U0 r).mate := by
  have h := iter_rel hK hP U0 r
  have hb := iterB_inv hK hP U0 r
  refine ⟨h.U.symm, funext fun i => ?_⟩
  unfold finalZ
  by_cases hmi : (iterB prop U0 r).mate i = i
  · rcases h.stale i hmi with hz | hz
    · simp [hz, hmi]
    · simp [hz, hmi]
  · have hz := h.matched i hmi
    have hj : (iterB prop U0 r).mate ((iterB prop U0 r).mate i) ≠ (iterB prop U0 r).mate i := by
      rw [hb.invol]; exact Ne.symm hmi
    have hz2 := h.matched _ hj
    rw [hb.invol] at hz2
    simp [hz, hz2]

/-! ## The 2-hop pass -/

/-- Consecutive members paired (`select` even and odd positions, shift, `eWiseMult`, :978-996). -/
def pairsOf : List Nat → List (Nat × Nat)
  | a :: b :: l => (a, b) :: pairsOf l
  | _ => []

def flatP (P : List (Nat × Nat)) : List Nat := P.flatMap (fun p => [p.1, p.2])

theorem flatP_pairsOf_sublist : ∀ l : List Nat, (flatP (pairsOf l)).Sublist l
  | [] => by simp [pairsOf, flatP]
  | [_] => by simp [pairsOf, flatP]
  | a :: b :: l => by
    have := flatP_pairsOf_sublist l
    simp only [pairsOf, flatP, List.flatMap_cons, List.cons_append, List.nil_append] at this ⊢
    exact (this.cons_cons b).cons_cons a

theorem flatP_append (P Q : List (Nat × Nat)) : flatP (P ++ Q) = flatP P ++ flatP Q := by
  simp [flatP, List.flatMap_append]

theorem flatP_groups_sublist (gs : List (List Nat)) : (flatP (gs.flatMap pairsOf)).Sublist gs.flatten := by
  induction gs with
  | nil => simp [flatP]
  | cons g gs ih =>
    simp only [List.flatMap_cons, flatP_append, List.flatten_cons]
    exact (flatP_pairsOf_sublist g).append ih

/-- The GraphBLAS part of the 2-hop pass (:967-996), as its contract: the groups are the rows of `g`
(`g(anchor(i), i) = i` built with no duplicates, so each unmatched vertex is in one group, sorted by
`GxB_Matrix_sort`, :975-977); members are unmatched (the `mate == idx` mask, :967-968) and adjacent
to their group's anchor (`anchor = unkey(MAX)` over `K`'s row, :969-971). -/
structure TwoHopGB (K : Nat → Nat → Prop) (mate : Nat → Nat) where
  groups : List (List Nat)
  nodup : groups.flatten.Nodup
  free : ∀ g ∈ groups, ∀ x ∈ g, mate x = x
  anchor : Nat → Nat
  adj : ∀ g ∈ groups, ∀ x ∈ g, K x (anchor x)
  same : ∀ g ∈ groups, ∀ x ∈ g, ∀ y ∈ g, anchor x = anchor y

/-- The host's cap filter (:1008-1015). -/
def capPairs (wh : Nat → Int) (cap : Int) (P : List (Nat × Nat)) : List (Nat × Nat) :=
  P.filter (fun p => decide (wh p.1 + wh p.2 ≤ cap))

/-- Both directions (`bi/bx`, :1010-1013). -/
def bothDir (Q : List (Nat × Nat)) : List (Nat × Nat) := Q.flatMap (fun p => [(p.1, p.2), (p.2, p.1)])

/-- `tmp = build(bi, bx)`; `mate⟨tmp, struct⟩ = tmp` (:1016-1019). -/
def twohop (Q : List (Nat × Nat)) (mate : Nat → Nat) (i : Nat) : Nat := ((bothDir Q).lookup i).getD (mate i)

theorem keys_bothDir (Q : List (Nat × Nat)) : (bothDir Q).map (·.1) = flatP Q := by
  induction Q with
  | nil => rfl
  | cons p Q ih => simp [bothDir, flatP] at ih ⊢; exact ih

theorem mem_flatP {Q : List (Nat × Nat)} {x : Nat} : x ∈ flatP Q ↔ ∃ p ∈ Q, x = p.1 ∨ x = p.2 := by
  simp [flatP, List.mem_flatMap]

theorem sublist_flatMap {α β : Type} (f : α → List β) {l₁ l₂ : List α} (h : l₁.Sublist l₂) :
    (l₁.flatMap f).Sublist (l₂.flatMap f) := by
  induction h with
  | slnil => exact List.Sublist.slnil
  | cons a _ ih => simp only [List.flatMap_cons]; exact ih.trans (List.sublist_append_right _ _)
  | cons_cons a _ ih => simp only [List.flatMap_cons]; exact (List.Sublist.refl _).append ih

/-- The partner of a paired vertex. -/
theorem twohop_pair {Q : List (Nat × Nat)} (hn : (flatP Q).Nodup) {x y : Nat} (h : (x, y) ∈ Q) (mate : Nat → Nat) :
    twohop Q mate x = y ∧ twohop Q mate y = x := by
  have hk : ((bothDir Q).map (·.1)).Nodup := by rw [keys_bothDir]; exact hn
  have h1 : (x, y) ∈ bothDir Q := by simp only [bothDir, List.mem_flatMap]; exact ⟨(x, y), h, by simp⟩
  have h2 : (y, x) ∈ bothDir Q := by simp only [bothDir, List.mem_flatMap]; exact ⟨(x, y), h, by simp⟩
  simp [twohop, lookup_of_mem hk h1, lookup_of_mem hk h2]

theorem twohop_out {Q : List (Nat × Nat)} {x : Nat} (h : x ∉ flatP Q) (mate : Nat → Nat) :
    twohop Q mate x = mate x := by
  have : (bothDir Q).lookup x = none := by
    apply lookup_eq_none_of
    intro e he hx
    apply h
    rw [← keys_bothDir]; exact List.mem_map.2 ⟨e, he, hx⟩
  simp [twohop, this]

theorem twohop_mem {Q : List (Nat × Nat)} (hn : (flatP Q).Nodup) {x : Nat} (h : x ∈ flatP Q) (mate : Nat → Nat) :
    ∃ p ∈ Q, (x = p.1 ∧ twohop Q mate x = p.2) ∨ (x = p.2 ∧ twohop Q mate x = p.1) := by
  obtain ⟨p, hp, hx⟩ := mem_flatP.1 h
  have := twohop_pair hn hp mate
  rcases hx with rfl | rfl
  · exact ⟨p, hp, Or.inl ⟨rfl, this.1⟩⟩
  · exact ⟨p, hp, Or.inr ⟨rfl, this.2⟩⟩

/-- **The 2-hop pass keeps `mate` an involution and every pair within the cap.** Its new pairs are not
edges of `A`: they share an anchor, a common `K`-neighbour. -/
theorem twohop_inv {mate : Nat → Nat} (hinv : ∀ i, mate (mate i) = i) (G : TwoHopGB K mate)
    (wh : Nat → Int) (cap : Int) (hcap : ∀ i, mate i ≠ i → wh i + wh (mate i) ≤ cap) :
    let Q := capPairs wh cap (G.groups.flatMap pairsOf)
    let m' := twohop Q mate
    (∀ i, m' (m' i) = i) ∧ (∀ i, m' i ≠ i → wh i + wh (m' i) ≤ cap) ∧
    (∀ i, m' i ≠ i → mate i = m' i ∨ (K i (G.anchor i) ∧ K (m' i) (G.anchor i))) := by
  intro Q m'
  have hs : (flatP Q).Sublist G.groups.flatten := by
    have h1 : (flatP Q).Sublist (flatP (G.groups.flatMap pairsOf)) :=
      sublist_flatMap (fun p : Nat × Nat => [p.1, p.2]) List.filter_sublist
    exact h1.trans (flatP_groups_sublist G.groups)
  have hnQ : (flatP Q).Nodup := G.nodup.sublist hs
  have hQfree : ∀ x ∈ flatP Q, mate x = x := by
    intro x hx
    obtain ⟨g, hg, hxg⟩ := List.mem_flatten.1 (hs.subset hx)
    exact G.free g hg x hxg
  have hQpair : ∀ p ∈ Q, ∃ g ∈ G.groups, p.1 ∈ g ∧ p.2 ∈ g := by
    intro p hp
    simp only [Q, capPairs, List.mem_filter, List.mem_flatMap] at hp
    obtain ⟨⟨g, hg, hpg⟩, _⟩ := hp
    refine ⟨g, hg, ?_⟩
    have hs := flatP_pairsOf_sublist g
    have : p.1 ∈ flatP (pairsOf g) ∧ p.2 ∈ flatP (pairsOf g) := by
      constructor <;> exact mem_flatP.2 ⟨p, hpg, by simp⟩
    exact ⟨hs.subset this.1, hs.subset this.2⟩
  have hQcap : ∀ p ∈ Q, wh p.1 + wh p.2 ≤ cap := by
    intro p hp
    simp only [Q, capPairs, List.mem_filter, decide_eq_true_eq] at hp
    exact hp.2
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_⟩
  · by_cases hx : i ∈ flatP Q
    · obtain ⟨p, hp, (⟨rfl, h⟩ | ⟨rfl, h⟩)⟩ := twohop_mem hnQ hx mate
      · show twohop Q mate (twohop Q mate p.1) = p.1
        rw [h]; exact (twohop_pair hnQ hp mate).2
      · show twohop Q mate (twohop Q mate p.2) = p.2
        rw [h]; exact (twohop_pair hnQ hp mate).1
    · show twohop Q mate (twohop Q mate i) = i
      rw [twohop_out hx]
      by_cases hk : mate i = i
      · rw [hk, twohop_out hx, hk]
      · have : mate i ∉ flatP Q := fun h' => by
          have := hQfree _ h'; rw [hinv] at this; exact hk this.symm
        rw [twohop_out this, hinv]
  · by_cases hx : i ∈ flatP Q
    · obtain ⟨p, hp, (⟨rfl, h⟩ | ⟨rfl, h⟩)⟩ := twohop_mem hnQ hx mate
      · show wh p.1 + wh (twohop Q mate p.1) ≤ cap; rw [h]; exact hQcap p hp
      · show wh p.2 + wh (twohop Q mate p.2) ≤ cap; rw [h]; have := hQcap p hp; omega
    · have hm : m' i = mate i := twohop_out hx mate
      rw [hm] at hi ⊢; exact hcap i hi
  · by_cases hx : i ∈ flatP Q
    · right
      obtain ⟨p, hp, (⟨rfl, h⟩ | ⟨rfl, h⟩)⟩ := twohop_mem hnQ hx mate
      · obtain ⟨g, hg, h1, h2⟩ := hQpair p hp
        show K p.1 (G.anchor p.1) ∧ K (twohop Q mate p.1) (G.anchor p.1)
        rw [h, G.same g hg p.1 h1 p.2 h2]
        exact ⟨G.same g hg p.1 h1 p.2 h2 ▸ G.adj g hg p.1 h1, G.adj g hg p.2 h2⟩
      · obtain ⟨g, hg, h1, h2⟩ := hQpair p hp
        show K p.2 (G.anchor p.2) ∧ K (twohop Q mate p.2) (G.anchor p.2)
        rw [h, G.same g hg p.2 h2 p.1 h1]
        exact ⟨G.same g hg p.2 h2 p.1 h1 ▸ G.adj g hg p.2 h2, G.adj g hg p.1 h1⟩
    · left; exact (twohop_out hx mate).symm

/-- What the numbering needs (`MatePre`) follows from an involution whose pairs join live vertices
below `n`. -/
theorem matePre_of {n : Nat} {wh : Nat → Int} {mate : Nat → Nat} (hinv : ∀ i, mate (mate i) = i)
    (hlive : ∀ i, mate i ≠ i → live wh i ∧ live wh (mate i) ∧ mate i < n) : MatePre n wh mate := by
  intro i hi hl
  by_cases h : mate i = i
  · rw [h]; exact ⟨hi, hl, h⟩
  · obtain ⟨_, h2, h3⟩ := hlive i h
    exact ⟨h3, h2, hinv i⟩

/-! ## `K` from `hkey`'s keys -/

/-- The pattern of `hkey`'s output: an entry of `A` that passes the cap test. -/
theorem hostKey_isSome {β : Type} {A : Rows Int} (hA : ∀ i, StrictRow (A i)) (wh : Nat → Int) (cap : Int)
    (capped : Bool) (key : Int → Nat → β) (i j : Nat) :
    (den (hostKeyRows A wh cap capped key) i j).isSome = true ↔
      ((den A i j).isSome = true ∧ ¬ (capped = true ∧ wh i + wh j > cap)) := by
  have := lookup_filterMap_key (hA i).keys_nodup (fun c => decide (capped = true ∧ wh i + wh c > cap)) key j
  have h2 : den (hostKeyRows A wh cap capped key) i j =
      ((A i).lookup j).bind (fun v => if capped = true ∧ wh i + wh j > cap then none else some (key v j)) := by
    simpa [den, hostKeyRows] using this
  rw [h2]
  cases hl : (A i).lookup j with
  | none => simp [den, hl]
  | some v =>
    by_cases hc : capped = true ∧ wh i + wh j > cap
    · simp [hc, den, hl]
    · simp [hc, den, hl]

/-- **`hkey`'s pattern meets the rounds' preconditions**: symmetric when `A` is, no diagonal when `A`
has none, every pair an edge of `A`, and within the cap when `capped`. -/
theorem keys_K {β : Type} {A : Rows Int} (hA : ∀ i, StrictRow (A i)) (wh : Nat → Int) (cap : Int)
    (capped : Bool) (key : Int → Nat → β)
    (hsym : ∀ i j, (den A i j).isSome = true → (den A j i).isSome = true)
    (hdiag : ∀ i, (den A i i).isSome = false) :
    let K := fun i j => (den (hostKeyRows A wh cap capped key) i j).isSome = true
    KSpec K ∧ (∀ i j, K i j → (den A i j).isSome = true) ∧
      (capped = true → ∀ i j, K i j → wh i + wh j ≤ cap) := by
  intro K
  refine ⟨⟨fun i j h => ?_, fun i h => ?_⟩, fun i j h => ((hostKey_isSome hA wh cap capped key i j).1 h).1,
    fun hc i j h => ?_⟩
  · obtain ⟨h1, h2⟩ := (hostKey_isSome hA wh cap capped key i j).1 h
    refine (hostKey_isSome hA wh cap capped key j i).2 ⟨hsym i j h1, fun ⟨a, b⟩ => h2 ⟨a, by omega⟩⟩
  · have := ((hostKey_isSome hA wh cap capped key i i).1 h).1
    rw [hdiag i] at this; exact Bool.noConfusion this
  · obtain ⟨_, h2⟩ := (hostKey_isSome hA wh cap capped key i j).1 h
    have : ¬ (wh i + wh j > cap) := fun hg => h2 ⟨hc, hg⟩
    omega

/-- **The matching, end to end**: after any number of rounds and the 2-hop pass, `mate` is an
involution; every pair is within the cap; a pair from the rounds is an edge of `A`, and a 2-hop pair
shares an anchor. `hKcap` holds when `capped` (`keys_K`) and, when not, because `2·maxw ≤ cap`
(`hkey_uncapped_vacuous`). -/
theorem matching_final (hK : KSpec K) (hP : PropSpec K prop) (U0 : Nat → Bool) (r : Nat)
    (wh : Nat → Int) (cap : Int) (hKcap : ∀ i j, K i j → wh i + wh j ≤ cap)
    (G : TwoHopGB K (iterB prop U0 r).mate) :
    let mate := (iterB prop U0 r).mate
    let m' := twohop (capPairs wh cap (G.groups.flatMap pairsOf)) mate
    (∀ i, m' (m' i) = i) ∧ (∀ i, m' i ≠ i → wh i + wh (m' i) ≤ cap) ∧
    (∀ i, m' i ≠ i → K i (m' i) ∨ (K i (G.anchor i) ∧ K (m' i) (G.anchor i))) := by
  intro mate m'
  have hb := iterB_inv hK hP U0 r
  obtain ⟨h1, h2, h3⟩ := twohop_inv hb.invol G wh cap (fun i hi => hKcap i _ (hb.edge i hi))
  refine ⟨h1, h2, fun i hi => ?_⟩
  rcases h3 i hi with h | h
  · left; have := hb.edge i (by rw [h]; exact hi); rw [h] at this; exact this
  · right; exact h

/-- **`MatePre` discharged end to end**: on an `A` whose entries join live vertices below `n` (symmetric,
no diagonal), the mate the rounds and the 2-hop pass leave on `hkey`'s key pattern is what `cpack`
assumes (`keys_K` + `matching_final` + `matePre_of`). The round and 2-hop contracts are `PropSpec`
(from `prop_of_mxv`) and `TwoHopGB`; `hKcap` is `keys_K`'s cap clause when capped, and
`hkey_uncapped_vacuous`'s bound when not. -/
theorem mate_pre_final {β : Type} {n : Nat} {A : Rows Int} {wh : Nat → Int} (hp : KeyPre n A wh)
    (cap : Int) (capped : Bool) (key : Int → Nat → β)
    (hsym : ∀ i j, (den A i j).isSome = true → (den A j i).isSome = true)
    (hdiag : ∀ i, (den A i i).isSome = false) {prop : Prop'}
    (hP : PropSpec (fun i j => (den (hostKeyRows A wh cap capped key) i j).isSome = true) prop)
    (U0 : Nat → Bool) (r : Nat)
    (hKcap : ∀ i j, (den (hostKeyRows A wh cap capped key) i j).isSome = true → wh i + wh j ≤ cap)
    (G : TwoHopGB (fun i j => (den (hostKeyRows A wh cap capped key) i j).isSome = true)
      (iterB prop U0 r).mate) :
    MatePre n wh (twohop (capPairs wh cap (G.groups.flatMap pairsOf)) (iterB prop U0 r).mate) := by
  obtain ⟨hKS, hKA, _⟩ := keys_K hp.rowStrict wh cap capped key hsym hdiag
  obtain ⟨h1, _, h3⟩ := matching_final hKS hP U0 r wh cap hKcap G
  have edge : ∀ a b, (den (hostKeyRows A wh cap capped key) a b).isSome = true →
      live wh a ∧ live wh b ∧ a < n ∧ b < n := by
    intro a b hab
    obtain ⟨v, hv⟩ := (lookup_isSome_iff _ _).1 (hKA a b hab)
    have hl := hp.liveEnds a (b, v) hv
    have hb := hp.colLt a (b, v) hv
    have ha : a < n := by
      by_cases h : n ≤ a
      · rw [hp.rowsBeyond a h] at hv; simp at hv
      · omega
    exact ⟨hl.1, hl.2, ha, hb⟩
  refine matePre_of h1 (fun i hi => ?_)
  rcases h3 i hi with h | ⟨ha, hb⟩
  · obtain ⟨l1, l2, _, l4⟩ := edge _ _ h; exact ⟨l1, l2, l4⟩
  · obtain ⟨l1, _, _, _⟩ := edge _ _ ha
    obtain ⟨m1, _, m3, _⟩ := edge _ _ hb
    exact ⟨l1, m1, m3⟩

end CchCoarsen
