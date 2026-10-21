/-
# The operations of `versioned_matrix.rs`, transcribed

Each definition follows its Rust function statement by statement, with the
GraphBLAS calls read as `Finset` operations:

| Rust                                         | model                                  |
| -------------------------------------------- | -------------------------------------- |
| `Matrix::set` / `Delta::insert`              | `insert`                               |
| `Matrix::remove` / `Delta::erase`            | `Finset.erase`                         |
| `Matrix::remove_all(mask)`                   | `· \ mask`                             |
| `dm<mask> = mask ∩ m` (`eWiseMult`, no repl.) | `(dm \ mask) ∪ (mask ∩ m)`              |
| `new_m = m ⊕ dp` (`eWiseAdd`, `bool`)        | `m ∪ dp` (pattern union)               |
| `new_m<!dm, replace> = m` (`select`)          | `m \ dm`                               |
| `Matrix::grown` (`GxB_Matrix_concat`)         | unchanged set at new bounds            |
| `Matrix::transpose`                          | `Finset.image Prod.swap`               |
| `Delta::count`, fold latches                 | *not modelled* — `flush` takes a param |

`Delta::insert` / `Delta::erase` are the counted wrappers over `Matrix::set` /
`Matrix::remove`: they move the approximate counter feeding the fold policy
alongside the layer write. The layer effect is all this model sees.
-/
import VersionedMatrix.Model

namespace FalkorDB
namespace VersionedMatrix

variable {v : VersionedMatrix}

/-! ## Construction -/

/-- `VersionedMatrix::new`. -/
def new (nrows ncols : Nat) : VersionedMatrix where
  m := ∅
  dp := ∅
  dm := ∅
  nrows := nrows
  ncols := ncols

/-- `VersionedMatrix::from_matrix`: wrap an owned base with empty deltas. -/
def fromMatrix (s : Finset Coord) (nrows ncols : Nat) : VersionedMatrix where
  m := s
  dp := ∅
  dm := ∅
  nrows := nrows
  ncols := ncols

/-- `Dup::dup` (a new MVCC version) and `Clone` (a handle copy). Both carry every
layer over unchanged, so on the abstract state they are the identity.

`dup` is also where the write path's fold decision is made and latched for the
next `flush` — but it folds nothing itself, and that decision is a `flush`
parameter here, so none of it reaches the abstract state. -/
def dup (v : VersionedMatrix) : VersionedMatrix := v

/-! ## Reads -/

/-- `get(i, j)`: the base decides which delta is consulted — `dm` when the pair is
committed, `dp` when it is not. Reads only `m` and one delta, never both. -/
def get (v : VersionedMatrix) (p : Coord) : Option Bool :=
  if p ∈ v.m then (if p ∈ v.dm then none else some true)
  else if p ∈ v.dp then some true else none

/-- `nvals`: `|m| + |dp| − |dm|`, on `Nat` — so the subtraction truncates exactly
where the Rust's `u64` would wrap. `Count.lean` proves it cannot. -/
def nvals (v : VersionedMatrix) : Nat := v.m.card + v.dp.card - v.dm.card

/-- `extract`: materialize the effective structure as a fresh `bool` matrix. -/
def extract (v : VersionedMatrix) : Finset Coord := eff v

/-- `iter(min_row, max_row)`: the effective entries whose row falls in the range.

Modelled by its *result*, not as the algorithm. The Rust streams a three-way
sorted merge with a `dm` lookahead — `m` entries the tombstone stream covers are
dropped, `dp` entries interleave in order — precisely so that no merged matrix is
materialized. That the merge computes this set is the boundary this development
draws, the same one the `Tensor` proofs draw for their iterators. -/
def iter (v : VersionedMatrix) (minRow maxRow : Nat) : Finset Coord :=
  (eff v).filter (fun p => minRow ≤ p.1 ∧ p.1 ≤ maxRow)

/-! ## Writes -/

/-- `set(i, j)`: add the pair, or undo a pending delete. Branches on the committed
base alone, which `Inv.dp_disj_m` and `Inv.dm_sub_m` are what license. -/
def set (v : VersionedMatrix) (p : Coord) : VersionedMatrix :=
  if p ∈ v.m then { v with dm := v.dm.erase p } else { v with dp := insert p v.dp }

/-- `remove(i, j)`: mark the pair deleted, or undo a pending add. -/
def remove (v : VersionedMatrix) (p : Coord) : VersionedMatrix :=
  if p ∈ v.m then { v with dm := insert p v.dm } else { v with dp := v.dp.erase p }

/-- `remove_mask(mask)`: the two-GraphBLAS-op bulk delete.

`dm<mask> = mask ∩ m` is a *masked assign without replace*, so `dm` entries
outside the mask survive; inside it, `dm ⊆ m` means the old entries are in the
intersection anyway. `dp &= ¬mask` drops the pending adds. -/
def removeMask (v : VersionedMatrix) (mask : Finset Coord) : VersionedMatrix :=
  { v with dm := (v.dm \ mask) ∪ (mask ∩ v.m), dp := v.dp \ mask }

/-! ### `set_all` and its two arms

`set_all_inner::<PROBE_BASE>` checks `dm` emptiness once, then takes one of two
paths. Both are transcribed, because the whole point of `Write.lean` is that they
agree. -/

/-- The `dm`-empty fast path: skip anything already committed, so `dp ∩ m = ∅`
survives, and never touch `dm`. -/
def setAllFast (v : VersionedMatrix) (l : List Coord) : VersionedMatrix :=
  l.foldl (fun a p => if p ∈ a.m then a else { a with dp := insert p a.dp }) v

/-- The general path, taken when `dm` is non-empty: per-entry `set`. -/
def setAllSlow (v : VersionedMatrix) (l : List Coord) : VersionedMatrix :=
  l.foldl set v

/-- `set_all` (`PROBE_BASE = true`). -/
def setAll (v : VersionedMatrix) (l : List Coord) : VersionedMatrix :=
  if v.dm = ∅ then setAllFast v l else setAllSlow v l

/-- `set_all_new`'s fast path (`PROBE_BASE = false`): the per-entry base probe is
skipped, on the caller's guarantee that no entry is committed. -/
def setAllNewFast (v : VersionedMatrix) (l : List Coord) : VersionedMatrix :=
  l.foldl (fun a p => { a with dp := insert p a.dp }) v

/-- `set_all_new`. -/
def setAllNew (v : VersionedMatrix) (l : List Coord) : VersionedMatrix :=
  if v.dm = ∅ then setAllNewFast v l else setAllSlow v l

/-- The caller guarantee behind `set_all_new`: fresh entity ids, so no entry is
live in the committed base. A reclaimed id's stale base entry always carries a
`dm` tombstone, which makes `dm` non-empty and routes to the checked path — so
this is only ever assumed on the fast arm. -/
def FreshEntries (v : VersionedMatrix) (l : List Coord) : Prop := ∀ p ∈ l, p ∉ v.m

/-! ## Maintenance -/

/-- One half of the fold: `dp` merged into the base (pattern union for `bool`),
then cleared. Built into a fresh matrix and swapped in by the Rust — a
copy-on-write choice with no denotational content. -/
def foldDp (v : VersionedMatrix) : VersionedMatrix := { v with m := v.m ∪ v.dp, dp := ∅ }

/-- The other half: the tombstones applied to the base, then cleared. In the Rust
this is `new_m<!dm, replace> = m`, i.e. `Matrix::select`. -/
def foldDm (v : VersionedMatrix) : VersionedMatrix := { v with m := v.m \ v.dm, dm := ∅ }

/-- `VersionedMatrix::flush`.

`fdp`/`fdm` are the *latched fold decisions*, taken as parameters rather than
computed. The policy behind them — `should_fold` / `should_fold_read` /
`delta_dominates_base`, latched in `dup` / `wait` / `fold_oversized` /
`fold_latched` and executed here — is a cost heuristic over deliberately
approximate counters. Quantifying over the decision proves the statement worth
having: *every* decision preserves the denotation, so no retuning of the
constants and no drift in the counters can change what the matrix means. See
`eff_flush_decision_irrelevant`.

The Rust decides both layers first and emits one matrix for the pair — `(true,
true)` is a single masked `eWiseAdd`, not two passes — and `foldDm ∘ foldDp`
denotes the same thing. `(false, false)` is guarded out there by
`if fold_dp || fold_dm`; here it is the identity, the only total completion.

`fold_latched` (end of a `GRAPH.BULK` command) and `fold_oversized` (MVCC commit)
differ only in when they fire and which policy latches the decision, then run this
same fold, so they need no separate model. -/
def flush (v : VersionedMatrix) (fdp fdm : Bool) : VersionedMatrix :=
  match fdp, fdm with
  | true,  true  => foldDm (foldDp v)
  | true,  false => foldDp v
  | false, true  => foldDm v
  | false, false => v

/-- `VersionedMatrix<bool>::resize`, growth only.

Unlike `Tensor::resize`, the `bool` grow path *folds*: the base is rebuilt as the
streamed two-way merge `(m ∖ dm) ∪ dp` at the new bounds and both deltas are
swapped out for fresh empty ones. (That merge is what needs `dp ∩ m = ∅` — with
shadowing allowed it would double-count, which is why the `u64` tensor keeps its
own grow path.) When both deltas are already empty the Rust skips the merge and
just `grown`s the base, which is this same function on that input.

Shrinking is a separate branch in the Rust and out of scope here: it drops
entries, and the callers only ever grow. -/
def resize (v : VersionedMatrix) (nrows ncols : Nat) : VersionedMatrix where
  m := eff v
  dp := ∅
  dm := ∅
  nrows := nrows
  ncols := ncols

/-- `transpose`: every layer transposed, bounds swapped. -/
def transpose (v : VersionedMatrix) : VersionedMatrix where
  m := v.m.image Prod.swap
  dp := v.dp.image Prod.swap
  dm := v.dm.image Prod.swap
  nrows := v.ncols
  ncols := v.nrows

end VersionedMatrix
end FalkorDB
