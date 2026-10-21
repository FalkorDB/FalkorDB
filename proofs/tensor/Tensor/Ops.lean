/-
# The operations of `tensor.rs`, transcribed

Every definition here follows the corresponding Rust function statement by
statement, using the `Layer`/`Finset` model of the GraphBLAS calls:

| Rust                                    | model                                                      |
| --------------------------------------- | ---------------------------------------------------------- |
| `GrB_Matrix_setElement`                 | `Layer.set`                                                |
| `GrB_Matrix_removeElement`              | `Layer.remove`                                             |
| `Matrix::remove_all(mask)`              | `Layer.removeAll` (`dom \ mask`)                           |
| `dm<mask> = mask ∩ m` (`eWiseMult`)     | `(dm \ mask) ∪ (mask ∩ m.dom)` — masked assign, no replace |
| `new_m = m ⊕ dp` (`eWiseAdd`, `SECOND`) | union with `dp` winning (`Layer.mergeSecond`)              |
| `new_m<!dm, replace> = m` (`select`)     | `Layer.removeAll` (`dom \ dm`)                             |
| `Matrix::grown` (`GxB_Matrix_concat`)   | unchanged entries at new bounds (`resize`)                  |
| `set_pattern`                           | pattern union into a `bool` matrix                         |
| `intersection_nvals`                    | `(dp.dom ∩ m.dom).card`                                    |
| `VersionedMatrix<bool>` (`mt`, `me`)    | its effective entry set (`Finset`)                         |
| `Delta::count` / fold latches           | *not modelled* — see `flush`, whose decision is a parameter |
-/
import Tensor.Key

namespace FalkorDB

/-! ## `Layer` combinators used by `flush` -/

namespace Layer

/-- `m.element_wise_add_second(dp)`: union of the patterns, `dp`'s value winning
on the overlap (GraphBLAS `GrB_SECOND`). -/
def mergeSecond (L R : Layer Nat) : Layer Nat :=
  { dom := L.dom ∪ R.dom, val := fun p => if p ∈ R.dom then R.val p else L.val p }

@[simp] theorem dom_mergeSecond {L R : Layer Nat} : (L.mergeSecond R).dom = L.dom ∪ R.dom := rfl

theorem get_mergeSecond_of_mem_right {L R : Layer Nat} {p : Pair} (h : p ∈ R.dom) :
    (L.mergeSecond R).get p = R.get p := by simp [mergeSecond, get, h]

theorem get_mergeSecond_of_not_mem_right {L R : Layer Nat} {p : Pair} (h : p ∉ R.dom) :
    (L.mergeSecond R).get p = L.get p := by
  by_cases hL : p ∈ L.dom <;> simp [mergeSecond, get, h, hL]

end Layer

namespace Tensor

/-! ## Construction -/

/-- `Tensor::new`. -/
def new (nrows ncols : Nat) : Tensor where
  m := { dom := ∅, val := fun _ => 0 }
  dp := { dom := ∅, val := fun _ => 0 }
  dm := ∅
  mt := ∅
  me := ∅
  nrows := nrows
  ncols := ncols

/-- `Tensor::dup` — a new MVCC version; and `Clone`, a handle copy. Both carry
every layer over unchanged, so on the abstract state they are the identity.

`dup` is also where the write path's fold decision is *made* (from the finished
transaction's contribution, `count - tx_nvals`) and latched for the next `flush`
to execute — but it folds nothing itself, and the decision is a `flush` parameter
here, so none of that reaches the abstract state. -/
def dup (t : Tensor) : Tensor := t

/-! ## Reads -/

/-- `Tensor::get`, i.e. the `EdgeIds` iterator, as the list it yields.
`Inline` yields the single inline id (or nothing); `Multi` streams the `me` row,
which GraphBLAS yields in ascending column order — modelled by `Finset.sort`. -/
def getIds (t : Tensor) (p : Pair) : List Nat :=
  match t.effGet p with
  | some v => if v = MULTI then (t.meRow (key p)).sort (· ≤ ·) else [v]
  | none => []

/-- `Tensor::has_multi_edge` — `me` non-empty. -/
def hasMultiEdge (t : Tensor) : Bool := t.me.Nonempty

/-- `Tensor::edge_count`, in the exact order the Rust computes it (`Nat`
subtraction stands in for `u64` wrapping; `Count.lean` proves no step
underflows). -/
def edgeCount (t : Tensor) : Nat :=
  t.m.nvals + t.dp.nvals - t.dm.card
    - (if t.dp.nvals = 0 then 0 else (t.dp.dom ∩ t.m.dom).card) - t.multiCount + t.me.card

/-- `Tensor::extract`: the effective forward pattern as a `bool` matrix,
built as `((pattern m) ∖ dm) ∪ (pattern dp)`. -/
def extract (t : Tensor) : Finset Pair :=
  let s := t.m.dom
  let s := if t.dm.card > 0 then s \ t.dm else s
  if t.dp.nvals > 0 then s ∪ t.dp.dom else s

/-- `Tensor::structural_iter`: the effective pattern restricted to a row range. -/
def structuralIter (t : Tensor) (minRow maxRow : Nat) : Finset Pair :=
  t.effDom.filter (fun p => minRow ≤ p.1 ∧ p.1 ≤ maxRow)

/-- The ids one pair contributes to an iteration: its inline value, or its whole
`me` row when the inline value is the `MULTI` sentinel.  This is the body of
`Iter::next` (`buf` is filled from `me` for a sentinel pair). -/
def iterAt (t : Tensor) (p : Pair) : Multiset (Nat × Nat × Nat) :=
  match t.effGet p with
  | some v =>
      if v = MULTI then (t.meRow (key p)).val.map (fun i => (p.1, p.2, i))
      else {(p.1, p.2, v)}
  | none => 0

/-- Rows selected by a `(min_row, max_row)` range. -/
def inRows (minRow maxRow : Nat) (p : Pair) : Prop := minRow ≤ p.1 ∧ p.1 ≤ maxRow

instance (minRow maxRow : Nat) (p : Pair) : Decidable (inRows minRow maxRow p) := by
  unfold inRows; infer_instance

/-- `Tensor::fwd_iter`: the effective `(src, dst, inline)` triples of a row range
(the raw UINT64 values, sentinels included). -/
def fwdIter (t : Tensor) (minRow maxRow : Nat) : Multiset (Nat × Nat × Nat) :=
  (t.effDom.filter (inRows minRow maxRow)).val.map
    (fun p => (p.1, p.2, (t.effGet p).getD 0))

/-- `Tensor::fwd_iter(0, u64::MAX)`: every effective triple, with no row filter.

The Rust passes `u64::MAX` as the upper bound, which is that type's way of
spelling "no bound at all". Modelling it as the numeral `2 ^ 64 - 1` would make
every statement about `iter_edges` depend on node ids being u64-representable —
a fact the old `Bounded` hypothesis silently supplied and the block-indexed key
neither needs nor provides — so the model says "every row" directly. -/
def fwdIterAll (t : Tensor) : Multiset (Nat × Nat × Nat) :=
  t.effDom.val.map (fun p => (p.1, p.2, (t.effGet p).getD 0))

/-- `Tensor::iter_edges`: the inline ids of all non-`MULTI` pairs, then every `me`
entry with its address turned back into `(src, dst)` by `compound_key_inverse`.

Since #2579 that address is a block and a row rather than one shifted key, and
the Rust walks `edge_versioned_all()` — every live block — reassembling each pair
from the block it was found in. `keyInverse` is total, so every entry yields a
pair. -/
def iterEdges (t : Tensor) : Multiset (Nat × Nat × Nat) :=
  (t.fwdIterAll.filter (fun x => x.2.2 ≠ MULTI)) +
    t.me.val.map (fun x =>
      let p := keyInverse x.1.1 x.1.2
      (p.1, p.2, x.2))

/-- `Tensor::iter` with `transpose = false`. -/
def iterFwd (t : Tensor) (minRow maxRow : Nat) : Multiset (Nat × Nat × Nat) :=
  (t.effDom.filter (inRows minRow maxRow)).val.bind t.iterAt

/-- `Tensor::iter` with `transpose = true`: walks `mt` (whose rows are `dst`) and
recovers each pair's ids through `eff_get`.  That lookup is total here because
`Iter.lean` proves it (`iterBwd_eff_get_isSome`); the Rust used to fall back on
`unwrap_or(0)`, which was worse than dead — 0 is a valid edge id, so a broken
`mt` would have emitted a fabricated edge — and now carries an `unreachable!()`
citing that theorem. -/
def iterBwd (t : Tensor) (minRow maxRow : Nat) : Multiset (Nat × Nat × Nat) :=
  (t.mt.filter (inRows minRow maxRow)).val.bind (fun r => t.iterAt (r.2, r.1))

/-! ## `set_all_from_slices`

The write phase of `set_all_from_slices`, for one queued inline value:

```rust
self.mt.set(d, s, true);
if let Some(committed) = m_masked[i] {
    self.dm.erase(s, d);
    if committed == id { self.dp.erase(s, d); continue; }
}
self.dp.insert(s, d, id);
```

`Delta::insert` / `Delta::erase` are the counted wrappers over
`Matrix::set` / `Matrix::remove`: they move the approximate counter that feeds the
fold policy alongside the layer write. The layer effect — all this model sees — is
the bare `set` / `remove`; see `flush` on why the counter is not modelled. -/
def writeInline (t : Tensor) (p : Pair) (id : Nat) (mMasked : Option Nat) : Tensor :=
  match mMasked with
  | some committed =>
      if committed = id then
        -- Cancel to clean: committed value restored, drop both deltas.
        { t with mt := insert (p.2, p.1) t.mt, dm := t.dm.erase p, dp := t.dp.remove p }
      else
        { t with mt := insert (p.2, p.1) t.mt, dm := t.dm.erase p, dp := t.dp.set p id }
  | none => { t with mt := insert (p.2, p.1) t.mt, dp := t.dp.set p id }

/-- One edge insertion: the read phase's `Entry::Vacant` decision followed by its
write-phase effect.  `set_all_from_slices` on a batch of distinct pairs is
exactly this, and `Batch.lean` proves the `batch`-map machinery makes repeated
pairs behave like repeated `addEdge`s.

The Rust local `cur` (`from_dp.or_else(|| if masked { None } else { m.get })`) is
literally `eff_get`, so the branch condition is `t.effGet p`. -/
def addEdge (t : Tensor) (p : Pair) (id : Nat) : Tensor :=
  match t.effGet p with
  -- Already multi-edge: just add the id to `me`.
  | some v =>
      if v = MULTI then { t with me := insert (key p, id) t.me }
      -- Present single edge: promote — the existing inline id joins the new one
      -- in `me`, and the sentinel is queued for the inline slot.
      else
        writeInline
          { t with me := insert (key p, v) (insert (key p, id) t.me) }
          p MULTI (if (t.dp.get p).isSome then t.m.get p else none)
  -- First edge for this pair: inline.
  | none => writeInline t p id (if p ∈ t.dm then t.m.get p else none)

/-- `set_all_from_slices` as repeated single insertion. -/
def setAll (t : Tensor) (l : List (Pair × Nat)) : Tensor :=
  l.foldl (fun t e => addEdge t e.1 e.2) t

/-! ## `remove_all` -/

/-- The fast path of `remove_all`, taken when no pair has multiple edges:
`dm<mask> = mask ∩ m`, `dp &= ¬mask`, `mt.remove_mask(maskᵗ)`. -/
def removeFast (t : Tensor) (mask : Finset Pair) : Tensor :=
  { t with
    dm := (t.dm \ mask) ∪ (mask ∩ t.m.dom)
    dp := t.dp.removeAll mask
    mt := t.mt.filter (fun q => (q.2, q.1) ∉ mask) }

/-- `me` row of `p` after `me.remove(key, id)`. -/
def rowAfterErase (t : Tensor) (p : Pair) (id : Nat) : Finset Nat :=
  meRowOf (t.me.erase (key p, id)) (key p)

/-- Delete the pair: drop any pending add, mask the committed entry, and drop the
backward entry.

```rust
self.dp.remove(src, dst);
if self.m.contains(src, dst) { self.dm.set(src, dst, true); }
self.mt.remove(dst, src);
```
-/
def deletePair (t : Tensor) (p : Pair) : Tensor :=
  { t with dp := t.dp.remove p,
           dm := if p ∈ t.m.dom then insert p t.dm else t.dm,
           mt := t.mt.erase (p.2, p.1) }

/-- One iteration of the slow path of `remove_all`: remove edge `id` from pair
`p`, returning the new tensor and `some p` when the pair became empty. -/
def removeOne (t : Tensor) (id : Nat) (p : Pair) : Tensor × Option Pair :=
  match t.effGet p with
  | some v =>
      if v = MULTI then
        if 2 ≤ (rowAfterErase t p id).card then
          -- still multi
          ({ t with me := t.me.erase (key p, id) }, none)
        else
          match (rowAfterErase t p id).min with
          | some last =>
              -- Demote: the surviving id returns inline (cancelling if it is the
              -- committed value).  `mt` already has `(dst, src)`.
              let t1 : Tensor :=
                { t with me := (t.me.erase (key p, id)).erase (key p, last) }
              if t.m.get p = some last then ({ t1 with dp := t1.dp.remove p }, none)
              else ({ t1 with dp := t1.dp.set p last }, none)
          | none =>
              -- A `MULTI` pair with no surviving id.  `Remove.lean` proves this
              -- is unreachable (`removeOne_survivor`), and the Rust now agrees
              -- by construction: it carries an `unreachable!()` here rather than
              -- the delete-the-pair branch it used to.  The model keeps a total
              -- definition, so this arm stays — it is simply never taken.
              (deletePair { t with me := t.me.erase (key p, id) } p, some p)
      else if v = id then (deletePair t p, some p)
      else (t, none)
  | none => (t, none)

/-- The slow path of `remove_all`: `removeOne` per edge, collecting the pairs
that became empty (in encounter order, as the Rust `Vec` does). -/
def removeSlow (t : Tensor) (rels : List (Nat × Pair)) : Tensor × List Pair :=
  match rels with
  | [] => (t, [])
  | r :: rest =>
      let (t', e) := removeOne t r.1 r.2
      let (t'', es) := removeSlow t' rest
      (t'', e.toList ++ es)

/-- `Tensor::remove_all`. -/
def removeAll (t : Tensor) (rels : List (Nat × Pair)) : Tensor × List Pair :=
  if rels.isEmpty then (t, [])
  else if ¬ hasMultiEdge t then
    (removeFast t (rels.map (fun r => r.2)).toFinset, rels.map (fun r => r.2))
  else removeSlow t rels

/-! ## Maintenance -/

/-- `Tensor::resize`, capacity growth only — the shrink branch drops entries and
is out of scope (the callers `resize` upwards).

The Rust re-emits each layer at the target dimensions with `Matrix::grown`
(`GxB_Matrix_concat`, blocking the layer into the top-left of a fresh matrix)
instead of resizing through the `Cow`. Both place every entry at its original
coordinate, so the denotation is just the new bounds. -/
def resize (t : Tensor) (nrows ncols : Nat) : Tensor :=
  { t with nrows := nrows, ncols := ncols }

/-- One half of the fold: `dp` merged into the base with `SECOND` — so a `dp`
value *shadowing* a committed pair wins — then cleared.

In the Rust this is `new_m.element_wise_add(.., Some(&m), Some(&dp), ..)` built
into a fresh matrix that replaces `m`, rather than an in-place update. That is a
copy-on-write choice (the base is shared with the committed snapshot, so an
in-place fold would deep-copy it first) with no denotational content. -/
def foldDp (t : Tensor) : Tensor := { t with m := t.m.mergeSecond t.dp, dp := t.dp.clear }

/-- The other half: the tombstones applied to the base, then cleared. In the Rust
this is the complemented `dm` mask with `REPLACE` — `new_m<!dm, replace> = m`,
i.e. `Matrix::select` — which on a pattern is exactly `m.dom \ dm`. -/
def foldDm (t : Tensor) : Tensor := { t with m := t.m.removeAll t.dm, dm := ∅ }

/-- `Tensor::flush`.

`fdp`/`fdm` are the *latched fold decisions*, taken as parameters rather than
computed. The policy behind them — `should_fold` / `should_fold_read` /
`delta_dominates_base` in `versioned_matrix.rs`, latched in `dup` / `wait_fwd` /
`fold_oversized` / `fold_latched` and executed here — is a cost heuristic
evaluated on deliberately *approximate* counters (`Delta::count`, which
overcounts a shadowing `insert` and saturates on `erase`). Pinning a formula into
the model would freeze one tuning decision and make the proofs stale on the next
measurement. Quantifying over the decision instead proves the stronger and more
useful statement: *every* decision preserves the denotation, so no change to the
constants — and no drift in the counters that feed them — can affect what the
tensor means. See `edgesAt_flush_decision_irrelevant`.

The Rust decides both layers first and emits one matrix for the pair: `(true,
true)` is a single masked `eWiseAdd`, not two passes. `foldDm (foldDp t)` denotes
the same thing — `(m ⊕ dp) \ dm` either way, because `dp ∩ dm = ∅` — which is
what `flush_effGet` checks. `(false, false)` is guarded out in the Rust by
`if fold_dp || fold_dm`; here it is the identity, the only total completion.

`fold_latched` (end of a `GRAPH.BULK` command) and `fold_oversized` (MVCC commit)
differ from the `dup`-driven path only in when they fire and which policy
computes the decision, then run this same fold — so they need no separate model,
and the theorems below cover them. `wait_fwd` / `wait_base` / `is_synced` remain
outside the model: they only settle GraphBLAS pending work and latch decisions. -/
def flush (t : Tensor) (fdp fdm : Bool) : Tensor :=
  match fdp, fdm with
  | true,  true  => foldDm (foldDp t)
  | true,  false => foldDp t
  | false, true  => foldDm t
  | false, false => t

/-- `Tensor::rebuild_backward`: `mt := transpose (extract t)`. -/
def rebuildBackward (t : Tensor) : Tensor :=
  { t with mt := (extract t).image (fun p => (p.2, p.1)) }

end Tensor
end FalkorDB
