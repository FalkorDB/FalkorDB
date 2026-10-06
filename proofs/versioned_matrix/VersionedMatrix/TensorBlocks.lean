import VersionedMatrix.TensorProofs
/-
# `Tensor`: the whole structure — `me` blocks, widening, multi-pair counting

The per-pair model (`Tensor.lean`) is lifted to a whole tensor `TT`: a finite
support `ps` of pairs, each with its `PairSt`, plus the list of `me` blocks
(name, column width, synced). A pair's `me` row lives in block
`blkOf p = (compound_key p).0`.

| here | there (`tensor.rs`) |
| --- | --- |
| `TT`            | `struct Tensor` (:270) |
| `newT`          | `Tensor::new` (:338) |
| `cloneT`        | `impl Clone` (:324) |
| `meBlock`       | `me_block` (:365) |
| `meBlockMut`    | `me_block_mut` (:378) |
| `widenMe`       | `widen_me_for_id` (:403) |
| `countRows`     | the row-counting loop of `multi_pairs_in` (:1376-1384) |
| `multiPairsIn`/`multiPairs` | :1357 / :1349 |
| `hasMultiEdge`  | :1386 |
| `block0`/`blocksAll` | `edge_versioned_block_0` (:1179) / `edge_versioned_all` (:1186) |
| `fwdM`/`fwdDp`/`fwdDm`/`matrixT` | accessors :1025-1041, :1168 |
-/
namespace VMTensorOps
open VMTensor

abbrev Pair := Nat × Nat
abbrev Blk := Nat × Nat

def ME_DIM : Nat := 2 ^ 60
def ME_NARROW_NCOLS : Nat := 2 ^ 31

/-- Delta bookkeeping (`count`, `tx_nvals`, `fold`, synced) of a forward delta. -/
structure Bk where
  count : Nat
  tx : Nat
  fold : Bool
  synced : Bool

structure MeB where
  blk : Blk
  ncols : Nat
  synced : Bool

structure TT where
  ps : List Pair
  st : Pair → PairSt
  dpB : Bk
  dmB : Bk
  mSynced : Bool
  mtSynced : Bool
  nf : Bool
  me : List MeB
  nrows : Nat
  ncols : Nat

def emptyP : PairSt := ⟨none, none, false, false, []⟩
def blkOf (p : Pair) : Blk := (compoundKey p.1 p.2).1

/-- `Tensor::new` (:338). -/
def newT (r c : Nat) : TT :=
  ⟨[], fun _ => emptyP, ⟨0, 0, false, true⟩, ⟨0, 0, false, true⟩, true, true, false,
   [⟨(0, 0), ME_NARROW_NCOLS, true⟩], r, c⟩

/-- `Clone` (:324): shares every handle — the same value. -/
def cloneT (t : TT) : TT := { t with }
theorem cloneT_eq (t : TT) : cloneT t = t := rfl

def names (t : TT) : List Blk := t.me.map (·.blk)

/-- `me_block` (:365). -/
def meBlock (t : TT) (b : Blk) : Option MeB := t.me.find? (·.blk == b)

/-- `me_block_mut` (:378): the block, appended (at `me[0]`'s width) if new. -/
def meBlockMut (t : TT) (b : Blk) : TT :=
  if b ∈ names t then t
  else { t with me := t.me ++ [⟨b, (t.me.headD ⟨(0,0), 0, true⟩).ncols, true⟩] }

/-- `widen_me_for_id` (:403). -/
def widenMe (t : TT) (maxId : Nat) : TT :=
  if maxId < (t.me.headD ⟨(0,0), 0, true⟩).ncols then t
  else { t with me := t.me.map (fun e => { e with ncols := ME_DIM }) }

/-- Block-list invariant: block 0 first, names distinct, one shared width
(narrow or `ME_DIM`), every pair with ids has its block, ids fit the width. -/
structure BlkInv (t : TT) : Prop where
  head : ∃ w s rest, t.me = ⟨(0, 0), w, s⟩ :: rest
  nd : (names t).Nodup
  width : ∀ e ∈ t.me, e.ncols = (t.me.headD ⟨(0,0), 0, true⟩).ncols
  wval : (t.me.headD ⟨(0,0), 0, true⟩).ncols = ME_NARROW_NCOLS ∨
         (t.me.headD ⟨(0,0), 0, true⟩).ncols = ME_DIM
  present : ∀ p, (t.st p).me ≠ [] → blkOf p ∈ names t
  fits : ∀ p, ∀ i ∈ (t.st p).me, i < (t.me.headD ⟨(0,0), 0, true⟩).ncols

theorem newT_blkInv (r c : Nat) : BlkInv (newT r c) := by
  refine ⟨⟨_, _, _, rfl⟩, by simp [names, newT], ?_, Or.inl rfl, ?_, ?_⟩
  · intro e he; simp [newT] at he; subst he; rfl
  · intro p h; exact absurd rfl h
  · intro p i h; simp [newT, emptyP] at h

theorem newT_empty (r c : Nat) (p : Pair) : edges ((newT r c).st p) = [] := rfl

theorem meBlock_some {t : TT} (h : BlkInv t) {b : Blk} (hb : b ∈ names t) :
    ∃ e ∈ t.me, meBlock t b = some e ∧ e.blk = b := by
  unfold meBlock
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hb
  cases hf : t.me.find? (·.blk == e.blk) with
  | none => rw [List.find?_eq_none] at hf; exact absurd (by simp) (hf e he)
  | some e' =>
    have h1 := List.find?_some hf; have h2 := List.mem_of_find?_eq_some hf
    exact ⟨e', h2, rfl, by simpa using h1⟩

theorem meBlock_none {t : TT} {b : Blk} (hb : b ∉ names t) : meBlock t b = none := by
  unfold meBlock; rw [List.find?_eq_none]
  intro e he h; exact hb (List.mem_map.2 ⟨e, he, by simpa using h⟩)

theorem meBlockMut_present (t : TT) (b : Blk) : b ∈ names (meBlockMut t b) := by
  unfold meBlockMut; split
  · assumption
  · simp [names]

theorem meBlockMut_inv {t : TT} (h : BlkInv t) (b : Blk) : BlkInv (meBlockMut t b) := by
  unfold meBlockMut; split
  · exact h
  · next hn =>
    obtain ⟨w, s, rest, hme⟩ := h.head
    refine ⟨⟨w, s, rest ++ [⟨b, w, true⟩], by simp [hme]⟩, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [names, List.map_append, List.map_cons, List.map_nil]
      exact List.nodup_append.2 ⟨h.nd, by simp, by
        intro a ha c hc hac; simp at hc; subst hc; subst hac; exact hn ha⟩
    · intro e he
      simp only [List.mem_append, List.mem_singleton] at he
      have hh : (t.me ++ [(⟨b, (t.me.headD ⟨(0,0), 0, true⟩).ncols, true⟩ : MeB)]).headD (⟨(0,0), 0, true⟩ : MeB) =
          t.me.headD (⟨(0,0), 0, true⟩ : MeB) := by simp [hme]
      rw [hh]; rcases he with he | rfl
      · exact h.width e he
      · rfl
    · simpa [hme] using h.wval
    · intro p hp; simp only [names, List.map_append, List.mem_append]; exact Or.inl (h.present p hp)
    · intro p i hi; simpa [hme] using h.fits p i hi

/-- Widening: after `widen_me_for_id(max)` every block has room for `max`
(for any real id, `max ≤ GrB_INDEX_MAX < ME_DIM`), and the invariant holds. -/
theorem widenMe_fits {t : TT} (h : BlkInv t) {maxId : Nat} (hm : maxId < ME_DIM) :
    ∀ e ∈ (widenMe t maxId).me, maxId < e.ncols := by
  unfold widenMe; split
  · next hlt => intro e he; rw [h.width e he]; exact hlt
  · intro e he; simp at he; obtain ⟨a, _, rfl⟩ := he; exact hm

theorem widenMe_inv {t : TT} (h : BlkInv t) (maxId : Nat) : BlkInv (widenMe t maxId) := by
  unfold widenMe; split
  · exact h
  · obtain ⟨w, s, rest, hme⟩ := h.head
    have hw := h.wval; simp [hme] at hw
    refine ⟨⟨ME_DIM, s, rest.map (fun e => { e with ncols := ME_DIM }), by simp [hme]⟩, ?_, ?_, Or.inr (by simp [hme]), ?_, ?_⟩
    · simpa [names, List.map_map, Function.comp_def] using h.nd
    · intro e he; simp [hme] at he ⊢; rcases he with rfl | ⟨a, _, rfl⟩ <;> rfl
    · intro p hp; simpa [names, List.map_map, Function.comp_def] using h.present p hp
    · intro p i hi
      have := h.fits p i hi; simp [hme] at this ⊢
      rcases hw with hw | hw <;> simp [hw, ME_NARROW_NCOLS, ME_DIM] at this ⊢ <;> omega

/-- Widening moves no id. -/
theorem widenMe_st (t : TT) (maxId : Nat) : (widenMe t maxId).st = t.st := by
  unfold widenMe; split <;> rfl

/-! ## `multi_pairs` -/

/-- The loop of `multi_pairs_in` (:1376-1384) over `me.iter(0, MAX)`. -/
def countRows : List (Nat × Nat) → Option Nat → Nat → Nat
  | [], _, n => n
  | (k, _) :: xs, last, n => if last = some k then countRows xs last n else countRows xs (some k) (n + 1)

/-- One pair's row: its key and its (non-empty) ids. -/
def rowEnts (k : Nat) (ids : List Nat) : List (Nat × Nat) := ids.map (fun i => (k, i))

theorem countRows_row (k : Nat) : ∀ (ids : List Nat) (rest : List (Nat × Nat)) (n : Nat),
    countRows (rowEnts k ids ++ rest) (some k) n = countRows rest (some k) n
  | [], _, _ => rfl
  | i :: is, rest, n => by
    simp only [rowEnts, List.map_cons, List.cons_append, countRows, if_pos]
    exact countRows_row k is rest n

theorem countRows_groups : ∀ (gs : List (Nat × List Nat)) (last : Option Nat) (n : Nat),
    (∀ g ∈ gs, g.2 ≠ []) → (gs.map (·.1)).Pairwise (· ≠ ·) → (∀ g ∈ gs, last ≠ some g.1) →
    countRows (gs.flatMap (fun g => rowEnts g.1 g.2)) last n = n + gs.length
  | [], _, n, _, _, _ => by simp [countRows]
  | (k, ids) :: gs, last, n, hne, hpw, hl => by
    obtain ⟨i, is, rfl⟩ := List.exists_cons_of_ne_nil (hne _ List.mem_cons_self)
    have := countRows_row k is (gs.flatMap (fun g => rowEnts g.1 g.2)) (n + 1)
    have e1 : (((k, i :: is) :: gs).flatMap (fun g => rowEnts g.1 g.2)) =
        (k, i) :: (rowEnts k is ++ gs.flatMap (fun g => rowEnts g.1 g.2)) := by
      simp [rowEnts]
    rw [e1]; simp only [countRows]
    rw [if_neg (hl _ List.mem_cons_self), this]
    simp only [List.map_cons, List.pairwise_cons] at hpw
    rw [countRows_groups gs (some k) (n + 1) (fun g hg => hne g (List.mem_cons_of_mem _ hg)) hpw.2
      (by intro g hg h; injection h with h; exact hpw.1 g.1 (List.mem_map_of_mem hg) h)]
    simp; omega

/-- `multi_pairs_in` (:1357): `0` on an empty block, the hypersparse vector
count (`GxB_Matrix_Iterator`-free `hyper_vector_count`, = number of non-empty
rows) when both deltas are empty, the counting loop otherwise. Both paths
are given the block's effective entries, grouped by row key. -/
def multiPairsIn (gs : List (Nat × List Nat)) (clean : Bool) : Nat :=
  if (gs.flatMap (fun g => rowEnts g.1 g.2)).length = 0 then 0
  else if clean then (gs.filter (·.2 ≠ [])).length
  else countRows (gs.flatMap (fun g => rowEnts g.1 g.2)) none 0

/-- Every path of `multi_pairs_in` counts the multi-edge pairs of the block. -/
theorem multiPairsIn_eq (gs : List (Nat × List Nat)) (clean : Bool)
    (hne : ∀ g ∈ gs, g.2 ≠ []) (hpw : (gs.map (·.1)).Pairwise (· ≠ ·)) :
    multiPairsIn gs clean = gs.length := by
  unfold multiPairsIn
  split
  · next h =>
    cases gs with
    | nil => rfl
    | cons g gs =>
      obtain ⟨i, is, hi⟩ := List.exists_cons_of_ne_nil (hne g List.mem_cons_self)
      simp [rowEnts, hi] at h
  · split
    · rw [List.filter_eq_self.2 (fun g hg => by simpa using hne g hg)]
    · rw [countRows_groups gs none 0 hne hpw (by simp)]; simp

def multiPairs (blocks : List (List (Nat × List Nat) × Bool)) : Nat :=
  (blocks.map (fun b => multiPairsIn b.1 b.2)).sum

theorem multiPairs_eq (blocks : List (List (Nat × List Nat) × Bool))
    (h : ∀ b ∈ blocks, (∀ g ∈ b.1, g.2 ≠ []) ∧ (b.1.map (·.1)).Pairwise (· ≠ ·)) :
    multiPairs blocks = (blocks.map (fun b => b.1.length)).sum := by
  unfold multiPairs; congr 1
  apply List.map_congr_left
  intro b hb; exact multiPairsIn_eq b.1 b.2 (h b hb).1 (h b hb).2

/-- The ids a block holds: the `me` rows of its pairs. -/
def blkIds (t : TT) (b : Blk) : List Nat := (t.ps.filter (blkOf · == b)).flatMap (fun p => (t.st p).me)

/-- `has_multi_edge` (:1386). -/
def hasMultiEdge (t : TT) : Bool := t.me.any (fun e => (blkIds t e.blk).length != 0)

theorem hasMultiEdge_iff {t : TT} (h : BlkInv t) :
    hasMultiEdge t = true ↔ ∃ p ∈ t.ps, (t.st p).me ≠ [] := by
  simp only [hasMultiEdge, List.any_eq_true, bne_iff_ne, ne_eq, List.length_eq_zero_iff, blkIds,
    List.flatMap_eq_nil_iff, List.mem_filter, beq_iff_eq]
  constructor
  · rintro ⟨e, _, hall⟩
    apply Classical.byContradiction; intro hn
    exact hall (fun p hp => Classical.byContradiction (fun h' => hn ⟨p, hp.1, h'⟩))
  · rintro ⟨p, hp, hme⟩
    obtain ⟨e, he, hb⟩ := List.mem_map.1 (h.present p hme)
    exact ⟨e, he, fun hall => hme (hall p ⟨hp, hb.symm⟩)⟩

/-- Under the pair invariant a non-empty row is exactly a multi-edge pair. -/
theorem me_nonempty_iff_multi (s : PairSt) (h : Inv s) : s.me ≠ [] ↔ effV s = some .multi := by
  constructor
  · intro hne; exact Classical.byContradiction (fun hc => hne (h.single_me hc))
  · intro hm he; have := h.multi_me hm; rw [he] at this; simp at this

/-! ## Accessors -/

def block0 (t : TT) : Option MeB := t.me.head?
def blocksAll (t : TT) : List (Blk × MeB) := t.me.map (fun e => (e.blk, e))
def fwdM (t : TT) (p : Pair) := (t.st p).m
def fwdDp (t : TT) (p : Pair) := (t.st p).dp
def fwdDm (t : TT) (p : Pair) := (t.st p).dm
def matrixT (t : TT) (p : Pair) := (t.st (p.2, p.1)).mt

theorem block0_is_block0 {t : TT} (h : BlkInv t) : ∃ e, block0 t = some e ∧ e.blk = (0, 0) := by
  obtain ⟨w, s, rest, hme⟩ := h.head; exact ⟨⟨(0, 0), w, s⟩, by simp [block0, hme], rfl⟩
theorem blocksAll_names (t : TT) : (blocksAll t).map (·.1) = names t := by
  simp [blocksAll, names, Function.comp_def]
theorem accessors (t : TT) (p : Pair) : fwdM t p = (t.st p).m ∧ fwdDp t p = (t.st p).dp ∧
    fwdDm t p = (t.st p).dm ∧ matrixT t (p.2, p.1) = (t.st p).mt := ⟨rfl, rfl, rfl, rfl⟩

end VMTensorOps
