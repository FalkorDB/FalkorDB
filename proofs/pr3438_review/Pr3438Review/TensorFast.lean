import VersionedMatrix.TensorResize
/-
# `Encode for Tensor`: the fast path (tensor.rs:1472-1570 on the PR, 2fabe1a8a)

```rust
self.wait_fwd();
let mut multi = Vec::new();
if self.dp.nvals() == 0 && self.dm.nvals() == 0 && !self.has_multi_edge() {
    let empty = Matrix::<u64>::new(nrows, ncols);
    self.m.encode(w); empty.encode(w); empty.encode(w);
} else {
    /* main's rebuild: (src, dst, inline) over fwd_iter(0, u64::MAX);
       inline == MULTI_EDGE ⇒ value ids.len() | MSB and push (src, dst, ids) to multi;
       build fm (or write `empty` when there are no triples); then empty, empty */
}
let total = self.edge_count(); /* … tensor section from `multi` … */
```

The model is versioned_matrix's whole-tensor model `TT` (support `ps`, per-pair
`PairSt` with `m`/`dp`/`dm`/`me`), its invariants `AllInv` (per-pair
module-doc invariants, incl. promotion completeness `multi_me`/`single_me`) and
`BlkInv`, `fwdIter` (= `fwd_iter`), `dpCount`/`dmCount` (= `dp.nvals()` /
`dm.nvals()`, exact: `Delta` derefs to `Matrix::nvals` = `GrB_Matrix_nvals`),
`hasMultiEdge` and `waitFwd`. A matrix is written as its *content*, the map
`(src, dst) ↦ value`:

| here | there |
| --- | --- |
| `mainTriples`/`mainBase` | main's `f_rows/f_cols/f_vals` and `fm.build` (or `empty` when no triple) (tensor.rs:1499-1530 PR / 9212add39 :1485-1515) |
| `mainMulti`  | the `multi` list main fills (:1505-1513) |
| `fastBase`   | the content of `self.m`: `id` inline, `MULTI_EDGE = u64::MAX` for a promoted pair |
| `fastCond`   | `dp.nvals() == 0 && dm.nvals() == 0 && !has_multi_edge()` (:1493) |
| `encMain`/`encPR` | the logical payload: base content, `edge_count()`, `multi` (the two deltas are `empty` in both) |

* **`fast_base_eq`** — under the invariants, when `fastCond` holds the committed
  base *is* the matrix main rebuilds: same pattern, same values (no pair of `m`
  holds `MULTI_EDGE`, because a promoted pair has ≥ 2 `me` ids and
  `has_multi_edge` would be true), and main's `multi` list is empty.
* **`encPR_eq_encMain`** — the PR's logical payload equals main's for every tensor
  satisfying the (versioned_matrix-maintained) invariants.
* **`encode_one_state`** — no TOCTOU inside `encode`: the condition, the written
  base, and `edge_count` all read `waitFwd t`, whose pairs and `me` blocks are
  `t`'s (`wait_fwd` only latches fold bookkeeping).
* **`decode_same`** — with the GraphBLAS container hypothesis `MatrixRT` (a
  matrix written by `Matrix::encode` decodes to a matrix of the same content,
  whatever its storage layout/slack) and `Tensor::decode` reading the base only
  through `fwd_m.iter()`, both payloads decode to the same tensor.

Checked live too (scratch copy, `lean3438_tensor_fast_path_vs_main_rebuild`):
60 random tensors (single/multi edges, removals, flushed or not, decoded
round-trips); on the 16 that take the fast path the PR's and main's payloads
decode to the same edges and `edge_count`; 15 of the 16 are byte-identical, one
differs only in the base matrix's layout.
-/
namespace Pr3438
open VMTensor VMTensorOps

def MULTI_EDGE : Nat := 2 ^ 64 - 1
def U64MAX : Nat := 2 ^ 64 - 1

/-- The u64 a committed-base entry stores. -/
def u64Of : Val → Nat
  | .id i => i
  | .multi => MULTI_EDGE

/-- A matrix built from distinct-pair triples, as a content map. -/
def buildC (trip : List (Pair × Nat)) (q : Pair) : Option Nat :=
  (trip.find? (fun x => x.1 == q)).map (·.2)

def emptyC : Pair → Option Nat := fun _ => none

/-- main's `f_rows`/`f_cols`/`f_vals`. -/
def mainTriples (t : TT) : List (Pair × Nat) :=
  (fwdIter t 0 U64MAX).map fun pv => (pv.1, match pv.2 with
    | .id i => i
    | .multi => (t.st pv.1).me.length + MSB)

def mainBase (t : TT) : Pair → Option Nat :=
  if mainTriples t = [] then emptyC else buildC (mainTriples t)

def mainMulti (t : TT) : List (Pair × List Nat) :=
  (fwdIter t 0 U64MAX).filterMap fun pv => if pv.2 = .multi then some (pv.1, (t.st pv.1).me) else none

def fastBase (t : TT) : Pair → Option Nat := fun q => if q ∈ t.ps then ((t.st q).m).map u64Of else none

def fastCond (t : TT) : Bool := dpCount t == 0 && dmCount t == 0 && !hasMultiEdge t

structure Payload where
  base : Pair → Option Nat
  total : Nat
  multi : List (Pair × List Nat)

/-- main's `encode`, logically. `ec` is `edge_count` (the same call in both). -/
def encMain (ec : TT → Nat) (t : TT) : Payload :=
  let t' := waitFwd t
  ⟨mainBase t', ec (waitFwd t'), mainMulti t'⟩

/-- The PR's `encode`, logically. -/
def encPR (ec : TT → Nat) (t : TT) : Payload :=
  let t' := waitFwd t
  if fastCond t' then ⟨fastBase t', ec (waitFwd t'), []⟩
  else ⟨mainBase t', ec (waitFwd t'), mainMulti t'⟩

theorem waitFwd_ps (t : TT) : (waitFwd t).ps = t.ps := by unfold waitFwd; split <;> rfl

/-- **No TOCTOU**: the state `encode` checks is the state it writes, and is the
caller's (pairs, values and `me` blocks). -/
theorem encode_one_state (t : TT) :
    (waitFwd t).ps = t.ps ∧ (waitFwd t).st = t.st ∧ (waitFwd t).me = t.me ∧
    (waitFwd (waitFwd t)).st = t.st ∧ (waitFwd (waitFwd t)).me = t.me ∧ (waitFwd (waitFwd t)).ps = t.ps :=
  ⟨waitFwd_ps t, (waitFwd_st t).1, (waitFwd_st t).2.2, by rw [(waitFwd_st _).1, (waitFwd_st t).1],
   by rw [(waitFwd_st _).2.2, (waitFwd_st t).2.2], by rw [waitFwd_ps, waitFwd_ps]⟩

theorem mainBase_eq (t : TT) : mainBase t = buildC (mainTriples t) := by
  unfold mainBase; split
  · rename_i h; rw [h]; funext q; rfl
  · rfl

/-- What main's rebuilt matrix holds at a pair. -/
theorem buildC_main (t : TT) (hrows : ∀ p ∈ t.ps, p.1 ≤ U64MAX) (q : Pair) :
    buildC (mainTriples t) q =
      if q ∈ t.ps then (effV (t.st q)).map (fun v => match v with
        | .id i => i
        | .multi => (t.st q).me.length + MSB) else none := by
  unfold buildC mainTriples
  rw [List.find?_map]
  cases hf : (fwdIter t 0 U64MAX).find? ((fun x : Pair × Nat => x.1 == q) ∘ fun pv => (pv.1, match pv.2 with
      | .id i => i
      | .multi => (t.st pv.1).me.length + MSB)) with
  | none =>
    simp only [Option.map_none]
    have hnone := List.find?_eq_none.1 hf
    by_cases hq : q ∈ t.ps
    · rw [if_pos hq]
      cases hv : effV (t.st q) with
      | none => rfl
      | some v =>
        exfalso
        have hm : (q, v) ∈ fwdIter t 0 U64MAX :=
          (mem_fwdIter t 0 U64MAX q v).2 ⟨hq, by simp [inRow, hrows q hq], hv⟩
        have := hnone _ hm
        simp at this
    · rw [if_neg hq]
  | some pv =>
    have hp := List.find?_some hf
    have hmem := List.mem_of_find?_eq_some hf
    obtain ⟨p, v⟩ := pv
    simp only [Function.comp, beq_iff_eq] at hp
    subst hp
    obtain ⟨hps, _, hv⟩ := (mem_fwdIter t 0 U64MAX p v).1 hmem
    simp only [Option.map_some, if_pos hps, hv]

theorem dpCount_zero (t : TT) (h : dpCount t = 0) : ∀ p ∈ t.ps, (t.st p).dp = none := by
  intro p hp
  have := List.eq_nil_of_length_eq_zero h
  have hn := List.filter_eq_nil_iff.1 this p hp
  cases hd : (t.st p).dp with
  | none => rfl
  | some _ => simp [hd] at hn

theorem dmCount_zero (t : TT) (h : dmCount t = 0) : ∀ p ∈ t.ps, (t.st p).dm = false := by
  intro p hp
  have := List.eq_nil_of_length_eq_zero h
  have hn := List.filter_eq_nil_iff.1 this p hp
  simpa using hn

/-- Under the fast-path condition no pair is effectively a multi-edge pair. -/
theorem no_multi (t : TT) (hi : AllInv t) (hb : BlkInv t) (hm : hasMultiEdge t = false) :
    ∀ p ∈ t.ps, effV (t.st p) ≠ some .multi := by
  intro p hp he
  have hne := (me_nonempty_iff_multi (t.st p) (hi p)).2 he
  have : hasMultiEdge t = true := (hasMultiEdge_iff hb).2 ⟨p, hp, hne⟩
  rw [hm] at this; cases this

/-- **`fast_base_eq`**: the committed base is main's rebuilt matrix, and main
would push no multi-edge pair. -/
theorem fast_base_eq (t : TT) (hi : AllInv t) (hb : BlkInv t) (hrows : ∀ p ∈ t.ps, p.1 ≤ U64MAX)
    (hc : fastCond t = true) : fastBase t = mainBase t ∧ mainMulti t = [] := by
  simp only [fastCond, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
  obtain ⟨⟨hdp, hdm⟩, hme⟩ := hc
  have hnm := no_multi t hi hb hme
  have heff : ∀ p ∈ t.ps, effV (t.st p) = (t.st p).m := by
    intro p hp; simp [effV, dpCount_zero t hdp p hp, dmCount_zero t hdm p hp]
  refine ⟨?_, ?_⟩
  · funext q
    rw [mainBase_eq, buildC_main t hrows q]
    unfold fastBase
    by_cases hq : q ∈ t.ps
    · rw [if_pos hq, if_pos hq, heff q hq]
      have := hnm q hq; rw [heff q hq] at this
      cases hmq : (t.st q).m with
      | none => simp
      | some v =>
        cases v with
        | id i => simp [u64Of]
        | multi => exact absurd hmq this
    · rw [if_neg hq, if_neg hq]
  · unfold mainMulti
    apply List.filterMap_eq_nil_iff.2
    intro pv hpv
    obtain ⟨p, v⟩ := pv
    obtain ⟨hps, _, hv⟩ := (mem_fwdIter t 0 U64MAX p v).1 hpv
    have := hnm p hps
    rw [hv] at this
    simp only
    rw [if_neg (fun h => this (by rw [h]))]

/-- **`encPR_eq_encMain`**: the PR's payload is main's, for every tensor that
satisfies versioned_matrix's invariants (rows are `u64`). -/
theorem encPR_eq_encMain (ec : TT → Nat) (t : TT) (hi : AllInv t) (hb : BlkInv t)
    (hrows : ∀ p ∈ t.ps, p.1 ≤ U64MAX) : encPR ec t = encMain ec t := by
  unfold encPR encMain
  have hs := (waitFwd_st t).1
  have hme := (waitFwd_st t).2.2
  have hps := waitFwd_ps t
  have hi' : AllInv (waitFwd t) := by intro p; rw [hs]; exact hi p
  have hb' : BlkInv (waitFwd t) := by
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hb
    exact ⟨by rw [hme]; exact h1, by simp only [names, hme] at h2 ⊢; exact h2, by rw [hme]; exact h3,
      by rw [hme]; exact h4, by intro p; rw [hs, names, hme]; exact h5 p, by intro p; rw [hs, hme]; exact h6 p⟩
  have hrows' : ∀ p ∈ (waitFwd t).ps, p.1 ≤ U64MAX := by rw [hps]; exact hrows
  dsimp only
  split
  · rename_i hc
    obtain ⟨h1, h2⟩ := fast_base_eq _ hi' hb' hrows' hc
    rw [h1, h2]
  · rfl

/-! ## Decoding: same tensor -/

/-- GraphBLAS container FFI hypothesis (GxB_unload/GxB_load, GxB_Vector_unload/load,
`GrB_Matrix_dup`): a matrix written by `Matrix::encode` decodes (`Matrix::decode`)
to a matrix with the same content (pattern + values), whatever the storage
layout, sparsity, iso flag or array slack the bytes carry. -/
structure MatrixRT (Mx Bytes : Type) where
  content : Mx → Pair → Option Nat
  enc : Mx → Bytes
  dec : Bytes → Option (Pair → Option Nat)
  rt : ∀ m, dec (enc m) = some (content m)

/-- **`decode_same`**: whatever matrices carry the two bases, `Tensor::decode`
(which reads the base through `fwd_m.iter()`, i.e. its content, then the
`edge_count` and the tensor section) yields the same tensor from both payloads. -/
theorem decode_same {Mx Bytes T : Type} (X : MatrixRT Mx Bytes)
    (decodeT : (Pair → Option Nat) → Nat → List (Pair × List Nat) → T)
    (ec : TT → Nat) (t : TT) (hi : AllInv t) (hb : BlkInv t) (hrows : ∀ p ∈ t.ps, p.1 ≤ U64MAX)
    (mPR mMain : Mx) (hPR : X.content mPR = (encPR ec t).base) (hMain : X.content mMain = (encMain ec t).base) :
    (X.dec (X.enc mPR)).map (fun c => decodeT c (encPR ec t).total (encPR ec t).multi) =
      (X.dec (X.enc mMain)).map (fun c => decodeT c (encMain ec t).total (encMain ec t).multi) := by
  rw [X.rt, X.rt, hPR, hMain, encPR_eq_encMain ec t hi hb hrows]

end Pr3438
