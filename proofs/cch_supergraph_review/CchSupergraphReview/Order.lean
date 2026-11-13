import CchSupergraphReview.Game
import FalkorNativeIndex.Cch
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# `chordal_supergraph`, line by line (`order.rs:381-437` @main d6fcd5aac)

| Lean | Rust (`graph/src/index/falkordb/cch/order.rs` @main) |
| --- | --- |
| `St` | the four locals `offsets`, `neighbors`, `first_child`, `next_sibling` (`:388-394`) |
| `rowSlots`, `slice` | `row_slots` (`:324-329`) and `neighbors[..]` (`:414`) |
| `higherOf` | step 1 (`:401-408`): `skeleton.row(node)` mapped through `node_to_rank`, kept if `> rank` |
| `absorbKids` | step 2 (`:410-419`): the `while let` over the child list, each child's row minus `rank` |
| `NI.sortDedup` | `row.sort_unstable(); row.dedup()` (`:423-424`), native_index's model (`sortDedup_unique`) |
| `stepRank` | one iteration (`:397-431`): the row, the parent link (`:426-428`), `extend` and `push` (`:429-430`) |
| `runRanks`, `chordalSupergraph` | the `for` over `rank_to_node` (`:397`), and the result (`:433-436`) |

`Rank` is its index (`Rank::new`/`Rank::index`, `:51-62`, are `+1`/`-1` on a `NonZeroUsize`; `Ord` compares the
stored value, so it compares indices). Indexing is `getD`, `List.set` and slices, which do nothing out of range
where Rust panics. `chordal_in_range` proves that on a well-formed skeleton with a permutation every index the
loop uses, at the iteration that uses it, is below its array's length, so the model and the Rust agree and the
Rust never panics there. The `while let` is modelled with fuel `n`;
`chain_kids` shows it never runs out (a rank has fewer than `n` children).

**Premises** (`OrderOk`, `Mirrored`). `node_to_rank` is the inverse of `rank_to_node` (what `invert` returns;
`Invert.lean`, `invert_ok_spec`), the skeleton is well formed (`from_order`'s check, `order.rs:199`), and it is
mirrored, which is `CchInput.skeleton`'s contract (`cch/mod.rs:168`). `from_order` does not check the mirror
itself: `nd::nested_dissection_order` does (`nd.rs:54`, `Csr::is_symmetric_simple`), so `Topology::build`'s only
skeletons have it (`mirrored_of_symmetricSimple`, `ndChecks_symm`); 03's
skeleton has it by construction (`Skeleton.lean`, `skel_symm`). `asymmetric_misses_arc` shows it is needed.
-/
namespace CchSG

open NI.Cch (Csr)
open CowBTree (Sorted)

/-- The builder's four arrays (`order.rs:388-394`). -/
structure St where
  offsets : List Nat
  neighbors : List Nat
  firstChild : List (Option Nat)
  nextSibling : List (Option Nat)

/-- `row_slots(&offsets, rank)` (`order.rs:324-329`). -/
def rowSlots (offsets : List Nat) (r : Nat) : Nat × Nat := (offsets.getD r 0, offsets.getD (r + 1) 0)

/-- `neighbors[a..b]`. -/
def slice (neighbors : List Nat) (s : Nat × Nat) : List Nat := (neighbors.drop s.1).take (s.2 - s.1)

/-- Step 1 (`order.rs:401-408`): `rank`'s higher skeleton neighbours, as ranks. -/
def higherOf (sk : Csr) (n2r : List Nat) (node rank : Nat) : List Nat :=
  ((sk.rowD node).map (fun nb => n2r.getD nb 0)).filter (fun h => decide (h > rank))

/-- Step 2 (`order.rs:410-419`): `row.extend(child's row minus rank)` for each child, following `next_sibling`. -/
def absorbKids (st : St) (rank : Nat) : Nat → Option Nat → List Nat → List Nat
  | 0, _, row => row
  | _ + 1, none, row => row
  | f + 1, some c, row =>
    absorbKids st rank f (st.nextSibling.getD c none)
      (row ++ (slice st.neighbors (rowSlots st.offsets c)).filter (fun h => h != rank))

/-- One iteration of the `for` (`order.rs:397-431`). -/
def stepRank (sk : Csr) (n2r : List Nat) (fuel : Nat) (st : St) (index node : Nat) : St :=
  let row := NI.sortDedup
    (absorbKids st index fuel (st.firstChild.getD index none) (higherOf sk n2r node index))
  let st1 : St := match row.head? with
    | some parent =>
      { st with nextSibling := st.nextSibling.set index (st.firstChild.getD parent none),
                firstChild := st.firstChild.set parent (some index) }
    | none => st
  { st1 with neighbors := st1.neighbors ++ row, offsets := st1.offsets ++ [(st1.neighbors ++ row).length] }

/-- The `for (index, &node) in rank_to_node.iter().enumerate()` (`order.rs:397`). -/
def runRanks (sk : Csr) (n2r : List Nat) (fuel : Nat) : Nat → List Nat → St → St
  | _, [], st => st
  | i, node :: rest, st => runRanks sk n2r fuel (i + 1) rest (stepRank sk n2r fuel st i node)

def initSt (n : Nat) : St := ⟨[0], [], List.replicate n none, List.replicate n none⟩

/-- `chordal_supergraph` (`order.rs:381-437`): `(offsets, neighbors)` of the upward `RankRows`. -/
def chordalSupergraph (sk : Csr) (r2n n2r : List Nat) : List Nat × List Nat :=
  let st := runRanks sk n2r r2n.length 0 r2n (initSt r2n.length)
  (st.offsets, st.neighbors)

/-! ## The rank-space skeleton and the premises -/

/-- The skeleton in rank space: `b` is in row `a` once both ends are mapped through the order. -/
def rankAdj (sk : Csr) (r2n n2r : List Nat) (a b : Nat) : Bool :=
  decide (a < r2n.length) && (sk.rowD (r2n.getD a 0)).any (fun nb => n2r.getD nb 0 == b)

/-- `node_to_rank` inverts `rank_to_node` over the skeleton's `n` nodes, and the skeleton is well formed. -/
structure OrderOk (sk : Csr) (r2n n2r : List Nat) : Prop where
  wf : sk.isWellFormed = true
  len_r2n : r2n.length = sk.n
  len_n2r : n2r.length = sk.n
  r2n_lt : ∀ r, r < sk.n → r2n.getD r 0 < sk.n
  n2r_lt : ∀ v, v < sk.n → n2r.getD v 0 < sk.n
  inv1 : ∀ r, r < sk.n → n2r.getD (r2n.getD r 0) 0 = r
  inv2 : ∀ v, v < sk.n → r2n.getD (n2r.getD v 0) 0 = v

/-- Every entry is mirrored. -/
def Mirrored (sk : Csr) : Prop := ∀ u, u < sk.n → ∀ v ∈ sk.rowD u, u ∈ sk.rowD v

theorem mirrored_of_symmetricSimple (bs : List Nat → Nat → Bool) (hbs : NI.Cch.BsSpec bs) (sk : Csr)
    (h : sk.isSymmetricSimple bs = true) : Mirrored sk := fun u hu v hv =>
  (((NI.Cch.isSymmetricSimple_iff bs hbs sk).1 h).2 u hu).2 v hv |>.2

theorem row_lt {sk : Csr} (hwf : sk.isWellFormed = true) {u v : Nat} (h : v ∈ sk.rowD u) : v < sk.n :=
  ((NI.Cch.isWellFormed_iff sk).1 hwf).2.2.2.2 v (NI.Cch.rowD_subset sk u v h)

section
variable {sk : Csr} {r2n n2r : List Nat}

theorem rankAdj_iff (a b : Nat) : rankAdj sk r2n n2r a b = true ↔
    a < r2n.length ∧ ∃ nb ∈ sk.rowD (r2n.getD a 0), n2r.getD nb 0 = b := by
  simp [rankAdj]

theorem rankAdj_supp (h : OrderOk sk r2n n2r) : Supp sk.n (rankAdj sk r2n n2r) := by
  intro a b hab
  obtain ⟨ha, nb, hnb, rfl⟩ := (rankAdj_iff a b).1 hab
  rw [h.len_r2n] at ha
  exact ⟨ha, h.n2r_lt nb (row_lt h.wf hnb)⟩

theorem rankAdj_symm (h : OrderOk sk r2n n2r) (hm : Mirrored sk) {a b : Nat}
    (hab : rankAdj sk r2n n2r a b = true) : rankAdj sk r2n n2r b a = true := by
  obtain ⟨ha, nb, hnb, rfl⟩ := (rankAdj_iff a b).1 hab
  rw [h.len_r2n] at ha
  have hnbn : nb < sk.n := row_lt h.wf hnb
  refine (rankAdj_iff _ _).2 ⟨by rw [h.len_r2n]; exact h.n2r_lt nb hnbn, r2n.getD a 0, ?_, h.inv1 a ha⟩
  rw [h.inv2 nb hnbn]
  exact hm _ (h.r2n_lt a ha) nb hnb

theorem rankAdj_or (h : OrderOk sk r2n n2r) (hm : Mirrored sk) (a b : Nat) :
    (rankAdj sk r2n n2r a b || rankAdj sk r2n n2r b a) = rankAdj sk r2n n2r a b := by
  cases hab : rankAdj sk r2n n2r a b
  · cases hba : rankAdj sk r2n n2r b a
    · rfl
    · rw [rankAdj_symm h hm hba] at hab; cases hab
  · rfl

theorem mem_higherOf (h : OrderOk sk r2n n2r) {r w : Nat} (hr : r < sk.n) :
    w ∈ higherOf sk n2r (r2n.getD r 0) r ↔ r < w ∧ rankAdj sk r2n n2r r w = true := by
  rw [rankAdj_iff, h.len_r2n]
  simp only [higherOf, List.mem_filter, List.mem_map, decide_eq_true_eq, gt_iff_lt]
  constructor
  · rintro ⟨⟨nb, hnb, rfl⟩, hlt⟩; exact ⟨hlt, hr, nb, hnb, rfl⟩
  · rintro ⟨hlt, _, nb, hnb, rfl⟩; exact ⟨⟨nb, hnb, rfl⟩, hlt⟩
end

/-! ## Rows so far, children so far -/

/-- Rows `0 .. k-1` of a row function `R`. -/
def rowsOf (R : Nat → List Nat) (k : Nat) : List (List Nat) := (List.range k).map R

/-- Their CSR offsets: entry `i` is the length of rows `0 .. i-1` together. -/
def offsOf (R : Nat → List Nat) (k : Nat) : List Nat :=
  (List.range (k + 1)).map (fun i => (rowsOf R i).flatten.length)

/-- The game's rows of ranks `0 .. k-1`. -/
abbrev rowsUpTo (n : Nat) (G : Nat → Nat → Bool) (k : Nat) : List (List Nat) := rowsOf (gameRow n G) k

/-- The offsets of those rows (`offsets` after `k` iterations). -/
abbrev offsUpTo (n : Nat) (G : Nat → Nat → Bool) (k : Nat) : List Nat := offsOf (gameRow n G) k

/-- The etree children of `p` among ranks `0 .. k-1`, ascending. -/
def kids (n : Nat) (G : Nat → Nat → Bool) (k p : Nat) : List Nat :=
  (List.range k).filter (fun c => par n G c == some p)

/-- The child list walked from `cur` through `next_sibling`, with `fuel` steps. -/
def chain (ns : List (Option Nat)) : Nat → Option Nat → List Nat
  | 0, _ => []
  | _ + 1, none => []
  | f + 1, some c => c :: chain ns f (ns.getD c none)

section
variable {n : Nat} {G : Nat → Nat → Bool}

theorem rowsOf_succ (R : Nat → List Nat) (k : Nat) : rowsOf R (k + 1) = rowsOf R k ++ [R k] := by
  simp [rowsOf, List.range_succ]

theorem offsOf_succ (R : Nat → List Nat) (k : Nat) :
    offsOf R (k + 1) = offsOf R k ++ [(rowsOf R (k + 1)).flatten.length] := by
  simp only [offsOf, List.range_succ (n := k + 1), List.map_append, List.map_cons, List.map_nil]

theorem offsOf_getD (R : Nat → List Nat) {k i : Nat} (hi : i ≤ k) :
    (offsOf R k).getD i 0 = (rowsOf R i).flatten.length := by
  simp [offsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega : i < k + 1)]

theorem flatten_prefix (R : Nat → List Nat) {c : Nat} : ∀ k, c < k →
    ∃ t, (rowsOf R k).flatten = (rowsOf R c).flatten ++ R c ++ t
  | 0, h => absurd h (Nat.not_lt_zero _)
  | k + 1, h => by
    rw [rowsOf_succ, List.flatten_append]
    by_cases hc : c = k
    · subst hc; exact ⟨[], by simp⟩
    · obtain ⟨t, ht⟩ := flatten_prefix R (c := c) k (by omega)
      exact ⟨t ++ R k, by rw [ht]; simp⟩

theorem slice_rowOf (R : Nat → List Nat) {k c : Nat} (hc : c < k) :
    slice (rowsOf R k).flatten (rowSlots (offsOf R k) c) = R c := by
  obtain ⟨t, ht⟩ := flatten_prefix R k hc
  simp only [slice, rowSlots, offsOf_getD R (Nat.le_of_lt hc), offsOf_getD R (Nat.succ_le_of_lt hc),
    rowsOf_succ, List.flatten_append, ht]
  simp [List.drop_append]

theorem rowsUpTo_succ (k : Nat) : rowsUpTo n G (k + 1) = rowsUpTo n G k ++ [gameRow n G k] := rowsOf_succ _ k

theorem offsUpTo_succ (k : Nat) :
    offsUpTo n G (k + 1) = offsUpTo n G k ++ [(rowsUpTo n G (k + 1)).flatten.length] := offsOf_succ _ k

theorem slice_row {k c : Nat} (hc : c < k) :
    slice (rowsUpTo n G k).flatten (rowSlots (offsUpTo n G k) c) = gameRow n G c := slice_rowOf _ hc

theorem mem_kids {k p c : Nat} : c ∈ kids n G k p ↔ c < k ∧ par n G c = some p := by
  simp [kids]

theorem kids_succ (k p : Nat) :
    kids n G (k + 1) p = kids n G k p ++ (if par n G k = some p then [k] else []) := by
  simp only [kids, List.range_succ, List.filter_append, List.filter_cons, List.filter_nil, beq_iff_eq]

theorem kids_length (k p : Nat) : (kids n G k p).length ≤ k := by
  have := List.length_filter_le (fun c => par n G c == some p) (List.range k)
  simpa [kids] using this

theorem kids_last : ∀ {k p c : Nat}, (kids n G k p).getLast? = some c →
    kids n G k p = kids n G c p ++ [c] ∧ c < k ∧ par n G c = some p
  | 0, p, c, h => by simp [kids] at h
  | k + 1, p, c, h => by
    rw [kids_succ] at h ⊢
    by_cases hk : par n G k = some p
    · rw [ite_eq_left hk] at h ⊢
      simp only [List.getLast?_append, List.getLast?_singleton, Option.some_or] at h
      cases h
      exact ⟨rfl, by omega, hk⟩
    · rw [ite_eq_right hk] at h ⊢
      simp only [List.append_nil] at h ⊢
      obtain ⟨h1, h2, h3⟩ := kids_last h
      exact ⟨h1, by omega, h3⟩

/-- **The `while let` walks exactly the children, newest first, and never runs out of fuel**: the child list of
    `p` is `kids k p` reversed when `next_sibling` links each child to the previous one. -/
theorem chain_kids (ns : List (Option Nat)) (p : Nat) : ∀ k,
    (∀ c, c < k → par n G c = some p → ns.getD c none = (kids n G c p).getLast?) →
    ∀ F, (kids n G k p).length ≤ F → chain ns F (kids n G k p).getLast? = (kids n G k p).reverse := by
  intro k
  induction k using Nat.strongRecOn with
  | _ k ih =>
    intro hns F hF
    cases hl : (kids n G k p).getLast? with
    | none =>
      rw [List.getLast?_eq_none_iff] at hl
      rw [hl]; cases F <;> rfl
    | some c =>
      obtain ⟨hk, hck, hpc⟩ := kids_last hl
      rw [hk] at hF ⊢
      simp only [List.length_append, List.length_singleton] at hF
      obtain ⟨F', rfl⟩ : ∃ F', F = F' + 1 := ⟨F - 1, by omega⟩
      simp only [chain, hns c hck hpc, List.reverse_append, List.reverse_singleton, List.singleton_append]
      rw [ih c hck (fun c' hc' hp' => hns c' (by omega) hp') F' (by omega)]

theorem absorbKids_eq (st : St) (r : Nat) : ∀ F cur row, absorbKids st r F cur row =
    row ++ ((chain st.nextSibling F cur).map
      (fun c => (slice st.neighbors (rowSlots st.offsets c)).filter (fun h => h != r))).flatten
  | 0, cur, row => by simp [absorbKids, chain]
  | F + 1, none, row => by simp [absorbKids, chain]
  | F + 1, some c, row => by
    simp only [absorbKids, chain, absorbKids_eq st r F, List.map_cons, List.flatten_cons, List.append_assoc]
end

/-! ## The loop invariant -/

/-- After `k` iterations: the arrays hold the game's first `k` rows, and the child lists are those rows'
    parents. -/
structure Inv (n : Nat) (G : Nat → Nat → Bool) (k : Nat) (st : St) : Prop where
  nb : st.neighbors = (rowsUpTo n G k).flatten
  off : st.offsets = offsUpTo n G k
  fcLen : st.firstChild.length = n
  nsLen : st.nextSibling.length = n
  fc : ∀ p, p < n → st.firstChild.getD p none = (kids n G k p).getLast?
  ns : ∀ c, st.nextSibling.getD c none =
    if c < k then (match par n G c with | some p => (kids n G c p).getLast? | none => none) else none

theorem inv_init (n : Nat) (G : Nat → Nat → Bool) : Inv n G 0 (initSt n) := by
  refine ⟨rfl, rfl, by simp [initSt], by simp [initSt], fun p _ => ?_, fun c => ?_⟩
  · simp [initSt, kids, List.getD_eq_getElem?_getD, List.getElem?_replicate]; split <;> rfl
  · simp [initSt, List.getD_eq_getElem?_getD, List.getElem?_replicate]; split <;> rfl

theorem getD_set_self {α : Type} (l : List α) (i : Nat) (a d : α) (h : i < l.length) :
    (l.set i a).getD i d = a := by
  simp [List.getD_eq_getElem?_getD, h]

theorem getD_set_ne {α : Type} (l : List α) {i j : Nat} (a d : α) (h : i ≠ j) :
    (l.set i a).getD j d = l.getD j d := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_set_ne h]

/-- **One iteration builds the game's row** for rank `k`, and keeps the invariant. -/
theorem inv_step {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk)
    {k : Nat} {st : St} (hi : Inv sk.n (rankAdj sk r2n n2r) k st) (hk : k < sk.n) (fuel : Nat) (hf : sk.n ≤ fuel) :
    Inv sk.n (rankAdj sk r2n n2r) (k + 1) (stepRank sk n2r fuel st k (r2n.getD k 0)) ∧
      NI.sortDedup (absorbKids st k fuel (st.firstChild.getD k none) (higherOf sk n2r (r2n.getD k 0) k)) =
        gameRow sk.n (rankAdj sk r2n n2r) k := by
  -- abbreviations
  have hs := rankAdj_supp h
  -- the row
  have hrow : NI.sortDedup (absorbKids st k fuel (st.firstChild.getD k none) (higherOf sk n2r (r2n.getD k 0) k)) =
      gameRow sk.n (rankAdj sk r2n n2r) k := by
    symm
    apply NI.sortDedup_unique _ _ (gameRow_sorted _ _ _)
    intro w
    rw [mem_gameRow hs, absorbKids_eq, hi.fc k hk,
      chain_kids st.nextSibling k k (fun c hc hp => by rw [hi.ns c, ite_eq_left hc, hp]) fuel
        (Nat.le_trans (kids_length k k) (by omega)),
      absorb_iff hs, rankAdj_or h hm]
    simp only [List.mem_append, List.mem_flatten, List.mem_map, List.mem_reverse, mem_higherOf h hk, hi.nb,
      hi.off]
    constructor
    · rintro (h0 | ⟨c, hc, hpc, hcw, hne⟩)
      · exact Or.inl h0
      · refine Or.inr ⟨_, ⟨c, mem_kids.2 ⟨hc, hpc⟩, rfl⟩, ?_⟩
        rw [slice_row hc]
        simp only [List.mem_filter, bne_iff_ne, ne_eq]
        exact ⟨(mem_gameRow hs).2 hcw, hne⟩
    · rintro (h0 | ⟨_, ⟨c, hc, rfl⟩, hw⟩)
      · exact Or.inl h0
      · obtain ⟨hc, hpc⟩ := mem_kids.1 hc
        rw [slice_row hc] at hw
        simp only [List.mem_filter, bne_iff_ne, ne_eq] at hw
        exact Or.inr ⟨c, hc, hpc, (mem_gameRow hs).1 hw.1, hw.2⟩
  refine ⟨?_, hrow⟩
  have hpar : (gameRow sk.n (rankAdj sk r2n n2r) k).head? = par sk.n (rankAdj sk r2n n2r) k := rfl
  unfold stepRank
  simp only [hrow, hpar]
  cases hp : par sk.n (rankAdj sk r2n n2r) k with
  | none =>
    refine ⟨?_, ?_, hi.fcLen, hi.nsLen, fun q hq => ?_, fun c => ?_⟩
    · simp [hi.nb, rowsUpTo_succ]
    · simp [hi.off, hi.nb, offsUpTo_succ, rowsUpTo_succ]
    · rw [hi.fc q hq, kids_succ]; simp [hp]
    · rw [hi.ns c]
      by_cases hc : c < k
      · simp [hc, Nat.lt_succ_of_lt hc]
      · by_cases hck : c = k
        · subst hck; simp [hp]
        · simp [hc, show ¬ c < k + 1 by omega]
  | some p =>
    have hpn : p < sk.n := (up_supp hs (par_spec hs hp).1).2
    refine ⟨?_, ?_, by simp [hi.fcLen], by simp [hi.nsLen], fun q hq => ?_, fun c => ?_⟩
    · simp [hi.nb, rowsUpTo_succ]
    · simp [hi.off, hi.nb, offsUpTo_succ, rowsUpTo_succ]
    · by_cases hqp : q = p
      · subst hqp
        rw [getD_set_self _ _ _ _ (by rw [hi.fcLen]; exact hpn)]
        simp [kids_succ, hp]
      · rw [getD_set_ne _ _ _ (Ne.symm hqp), hi.fc q hq, kids_succ]
        have : ¬ par sk.n (rankAdj sk r2n n2r) k = some q := by rw [hp]; simp; exact Ne.symm hqp
        simp [this]
    · by_cases hck : c = k
      · subst hck
        rw [getD_set_self _ _ _ _ (by rw [hi.nsLen]; exact hk), hi.fc p hpn]
        simp [hp]
      · rw [getD_set_ne _ _ _ (Ne.symm hck), hi.ns c]
        by_cases hc : c < k
        · simp [hc, Nat.lt_succ_of_lt hc]
        · simp [hc, show ¬ c < k + 1 by omega]

theorem inv_run {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk) (fuel : Nat)
    (hf : sk.n ≤ fuel) : ∀ (rest : List Nat) (k : Nat) (st : St), Inv sk.n (rankAdj sk r2n n2r) k st →
      k + rest.length = sk.n → rest = r2n.drop k →
      Inv sk.n (rankAdj sk r2n n2r) sk.n (runRanks sk n2r fuel k rest st)
  | [], k, st, hi, hl, _ => by simp at hl; subst hl; exact hi
  | node :: rest, k, st, hi, hl, hr => by
    have hk : k < sk.n := by simp at hl; omega
    have hnode : node = r2n.getD k 0 := by
      have : (r2n.drop k)[0]? = some node := by rw [← hr]; rfl
      simp only [List.getElem?_drop, Nat.add_zero] at this
      simp [List.getD_eq_getElem?_getD, this]
    have hrest : rest = r2n.drop (k + 1) := by
      have := congrArg List.tail hr
      simpa [List.tail_drop] using this
    subst hnode
    exact inv_run h hm fuel hf rest (k + 1) _ (inv_step h hm hi hk fuel hf).1 (by simp at hl; omega) hrest

/-- **`chordal_supergraph` plays the elimination game exactly** (`order.rs:381-437`). On a well-formed,
    mirrored skeleton with `node_to_rank` the inverse of `rank_to_node`, the result's `offsets` and `neighbors`
    hold, for every rank `r`, exactly the game's upward row `gameRow r` (ascending, no duplicates), in rank order.
    Every index the loop reads or writes is in range: `chordal_in_range`. -/
theorem chordalSupergraph_spec {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk) :
    chordalSupergraph sk r2n n2r =
      (offsUpTo sk.n (rankAdj sk r2n n2r) sk.n, (rowsUpTo sk.n (rankAdj sk r2n n2r) sk.n).flatten) := by
  have hi := inv_run h hm r2n.length (by have := h.len_r2n; omega) r2n 0 (initSt r2n.length)
    (by rw [h.len_r2n]; exact inv_init _ _) (by rw [h.len_r2n]; simp) (by simp)
  simp only [chordalSupergraph, hi.nb, hi.off]

/-- Row `r` of the result, read through `row_slots` (what `Topology::up` returns), is the game's row. -/
theorem chordalSupergraph_row {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk)
    {r : Nat} (hr : r < sk.n) :
    slice (chordalSupergraph sk r2n n2r).2 (rowSlots (chordalSupergraph sk r2n n2r).1 r) =
      gameRow sk.n (rankAdj sk r2n n2r) r := by
  rw [chordalSupergraph_spec h hm]; exact slice_row hr

/-! ## Every index is in range (no panic) -/

theorem runRanks_append (sk : Csr) (n2r : List Nat) (fuel : Nat) : ∀ (l1 l2 : List Nat) (i : Nat) (st : St),
    runRanks sk n2r fuel i (l1 ++ l2) st = runRanks sk n2r fuel (i + l1.length) l2 (runRanks sk n2r fuel i l1 st)
  | [], l2, i, st => by simp [runRanks]
  | x :: l1, l2, i, st => by
    simp only [List.cons_append, runRanks, runRanks_append sk n2r fuel l1 l2 (i + 1), List.length_cons]
    congr 1; omega

/-- The builder's state at the start of iteration `k`: `runRanks` over the first `k` nodes. -/
def stateAt (sk : Csr) (r2n n2r : List Nat) (k : Nat) : St :=
  runRanks sk n2r r2n.length 0 (r2n.take k) (initSt r2n.length)

/-- The whole run passes through `stateAt k` and continues from it. -/
theorem run_split (sk : Csr) (r2n n2r : List Nat) (k : Nat) :
    chordalSupergraph sk r2n n2r =
      ((runRanks sk n2r r2n.length (r2n.take k).length (r2n.drop k) (stateAt sk r2n n2r k)).offsets,
       (runRanks sk n2r r2n.length (r2n.take k).length (r2n.drop k) (stateAt sk r2n n2r k)).neighbors) := by
  have e := runRanks_append sk n2r r2n.length (r2n.take k) (r2n.drop k) 0 (initSt r2n.length)
  rw [List.take_append_drop, Nat.zero_add] at e
  simp only [chordalSupergraph, stateAt, e]

theorem inv_prefix {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk) (fuel : Nat)
    (hf : sk.n ≤ fuel) : ∀ k, k ≤ sk.n →
      Inv sk.n (rankAdj sk r2n n2r) k (runRanks sk n2r fuel 0 (r2n.take k) (initSt sk.n))
  | 0, _ => by simp only [List.take_zero, runRanks]; exact inv_init _ _
  | k + 1, hk => by
    have hkl : k < r2n.length := by rw [h.len_r2n]; omega
    rw [List.take_add_one, List.getElem?_eq_getElem hkl, Option.toList_some, runRanks_append]
    simp only [runRanks, List.length_take, Nat.zero_add, Nat.min_eq_left (Nat.le_of_lt hkl)]
    have hnode : r2n[k] = r2n.getD k 0 := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkl]
    rw [hnode]
    exact (inv_step h hm (inv_prefix h hm fuel hf k (by omega)) (by omega) fuel hf).1

/-- **Every index iteration `k` uses is in range**, given the loop invariant: `skeleton.row(node)` (`:404`, a
    row that exists), `node_to_rank[neighbor]` (`:406`), `first_child[index]` (`:411`), each child's
    `row_slots` and slice (`:414`) and `next_sibling[child]` (`:418`) for every child the `while let` visits
    (`chain_kids`: exactly `kids k k`), and `first_child[parent]`, `next_sibling[index]` (`:427`). -/
theorem step_in_range {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) {k : Nat} {st : St}
    (hi : Inv sk.n (rankAdj sk r2n n2r) k st) (hk : k < sk.n) :
    r2n.getD k 0 < sk.n ∧ (sk.row? (r2n.getD k 0)).isSome = true ∧
    (∀ nb ∈ sk.rowD (r2n.getD k 0), nb < n2r.length) ∧
    k < st.firstChild.length ∧ k < st.nextSibling.length ∧
    (∀ c ∈ kids sk.n (rankAdj sk r2n n2r) k k, c + 1 < st.offsets.length ∧
      st.offsets.getD c 0 ≤ st.offsets.getD (c + 1) 0 ∧ st.offsets.getD (c + 1) 0 ≤ st.neighbors.length ∧
      c < st.nextSibling.length) ∧
    (∀ p, par sk.n (rankAdj sk r2n n2r) k = some p → p < st.firstChild.length) := by
  have hs := rankAdj_supp h
  have hnode := h.r2n_lt k hk
  obtain ⟨row, hrow⟩ := NI.Cch.«row?_isSome_of_wf» sk h.wf _ hnode
  refine ⟨hnode, by rw [hrow]; rfl, fun nb hnb => by rw [h.len_n2r]; exact row_lt h.wf hnb,
    by rw [hi.fcLen]; exact hk, by rw [hi.nsLen]; exact hk, fun c hc => ?_, fun p hp => ?_⟩
  · obtain ⟨hck, _⟩ := mem_kids.1 hc
    obtain ⟨t, ht⟩ := flatten_prefix (gameRow sk.n (rankAdj sk r2n n2r)) k hck
    have e1 := offsOf_getD (gameRow sk.n (rankAdj sk r2n n2r)) (k := k) (i := c) (by omega)
    have e2 := offsOf_getD (gameRow sk.n (rankAdj sk r2n n2r)) (k := k) (i := c + 1) (by omega)
    rw [rowsOf_succ, List.flatten_append] at e2
    refine ⟨by rw [hi.off]; simp [offsOf]; omega, ?_, ?_, by rw [hi.nsLen]; omega⟩
    · rw [hi.off, e1, e2]; simp
    · rw [hi.off, e2, hi.nb, ht]; simp
  · rw [hi.fcLen]; exact (up_supp hs (par_spec hs hp).1).2

/-- **`chordal_supergraph` never indexes out of range** on a well-formed, mirrored skeleton with a permutation:
    at the start of every iteration `k` (the state `run_split` names), `step_in_range` holds. -/
theorem chordal_in_range {sk : Csr} {r2n n2r : List Nat} (h : OrderOk sk r2n n2r) (hm : Mirrored sk)
    {k : Nat} (hk : k < sk.n) :
    let st := stateAt sk r2n n2r k
    r2n.getD k 0 < sk.n ∧ (sk.row? (r2n.getD k 0)).isSome = true ∧
    (∀ nb ∈ sk.rowD (r2n.getD k 0), nb < n2r.length) ∧
    k < st.firstChild.length ∧ k < st.nextSibling.length ∧
    (∀ c ∈ kids sk.n (rankAdj sk r2n n2r) k k, c + 1 < st.offsets.length ∧
      st.offsets.getD c 0 ≤ st.offsets.getD (c + 1) 0 ∧ st.offsets.getD (c + 1) 0 ≤ st.neighbors.length ∧
      c < st.nextSibling.length) ∧
    (∀ p, par sk.n (rankAdj sk r2n n2r) k = some p → p < st.firstChild.length) := by
  intro st
  have hi := inv_prefix h hm r2n.length (by have := h.len_r2n; omega) k (Nat.le_of_lt hk)
  have e : initSt sk.n = initSt r2n.length := by rw [h.len_r2n]
  rw [e] at hi
  exact step_in_range h hi hk

end CchSG
