import FalkorNativeIndex.Agree
import FalkorCowBTree.Basic
set_option linter.unusedSimpArgs false

/-!
# The `CowBTree` contract, as its contents

The native index calls six tree operations. Their behaviour on contents is **proven in
`proofs/cow_btree`** (and `proofs/index_layer` for `insert_batch`); here the tree *is* its sorted
entry list and each operation *is* its proven content spec. `enc`/`keyOf`/`docOf`, `Sorted`,
`sorted_ext`, `sins`, `mem_sins`, `sorted_sins`, `mem_erase_sorted` are imported from
`proofs/cow_btree/FalkorCowBTree/{Model,Basic}.lean` (Lake path dependency):

| here | Rust | proven contents |
| --- | --- | --- |
| `tInsert` | `CowBTree::insert` (`cow_btree/mod.rs:223`) | cow_btree `insert_spec` (= `sins`) |
| `tRemove` | `CowBTree::remove` (`:260`) | cow_btree `remove_spec` (= `List.erase`) |
| `tInsertBatch` | `CowBTree::insert_batch` (`:241`) | cow_btree `insertBatch_wf`, index_layer `Leaf.insertBatch_contents` (= union) |
| `tRemoveBatch` | `CowBTree::remove_batch` (`:286`) | cow_btree `removeBatchTree_wf` (= `List.filter`) |
| `tRange` | `CowBTree::range` / `point` (`:314`, `:324`) | cow_btree `range_spec`, `rangeDocs_spec` |
| `tFromSorted` | `CowBTree::from_sorted` (`:190`) | cow_btree `fromSorted_wf`, index_layer (contents = input) |

**The hypothesis these specs need, discharged.** They are proven under cow_btree's `TreeWF`. Before
#3283 `remove_batch` did not preserve it (cow_btree Bug 3 = FINDINGS W7-btree-1, now
`pre3283_rb_remove_batch_not_wf`), so the numeric-index rows that read through the tree were
CONDITIONAL. #3283 (799968a8b) fixed it, and cow_btree now proves every tree the API can build is
well-formed (`Reachable_wf_asserted`, from `removeBatchTree_wf`, `insertBatch_wf`, `fromSorted_wf`,
`insert_spec`, `remove_spec`; `BRANCH_MAX >= 4` is the const assert). `Contract.lean` proves each
operation equals its model above on every such tree, so those rows are PROVEN.
-/

namespace NI

/-! The entry encoding and the sorted-list lemmas are proofs/cow_btree's (`Model.lean`,
`Basic.lean`), imported via a Lake path dependency. Its `W` is this project's `W` (`2^64`). -/
export CowBTree (enc keyOf docOf keyOf_enc docOf_enc Sorted sorted_cons sorted_ext sins mem_sins
  sorted_sins mem_erase_sorted sorted_erase)

theorem W_eq : CowBTree.W = W := rfl

theorem enc_keyOf_docOf (e : Nat) : enc (keyOf e) (docOf e) = e := by
  unfold enc keyOf docOf; have := Nat.div_add_mod e CowBTree.W; rw [Nat.mul_comm] at this; omega

/-! ## The tree, as its contents -/

def tInsert (t : List Nat) (x : Nat) : List Nat := sins x t
def tRemove (t : List Nat) (x : Nat) : List Nat := t.erase x
def tInsertBatch (t : List Nat) (b : List Nat) : List Nat := b.foldl (fun t x => sins x t) t
def tRemoveBatch (t : List Nat) (b : List Nat) : List Nat := t.filter (fun e => !b.contains e)
def tRange (t : List Nat) (lo hi : Nat) : List Nat :=
  (t.filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))).map docOf
def tFromSorted (pairs : List Nat) : List Nat := pairs
def tIsEmpty (t : List Nat) : Bool := t.isEmpty

theorem mem_insertBatch : ∀ (b t : List Nat) (e : Nat), e ∈ tInsertBatch t b ↔ e ∈ t ∨ e ∈ b
  | [], t, e => by simp [tInsertBatch]
  | x :: xs, t, e => by
    simp only [tInsertBatch, List.foldl_cons] at *
    have := mem_insertBatch xs (sins x t) e
    simp only [tInsertBatch] at this
    rw [this, mem_sins]; simp only [List.mem_cons, or_assoc]

theorem sorted_insertBatch : ∀ (b t : List Nat), Sorted t → Sorted (tInsertBatch t b)
  | [], t, h => h
  | x :: xs, t, h => by
    simp only [tInsertBatch, List.foldl_cons]
    exact sorted_insertBatch xs _ (sorted_sins x t h)

theorem mem_removeBatch (t b : List Nat) (e : Nat) : e ∈ tRemoveBatch t b ↔ e ∈ t ∧ e ∉ b := by
  simp [tRemoveBatch]

theorem sorted_removeBatch (t b : List Nat) (h : Sorted t) : Sorted (tRemoveBatch t b) :=
  List.Pairwise.sublist List.filter_sublist h

/-- A sorted, deduplicated copy (`sort_unstable` + `dedup`): the unique strictly sorted list with
    the same members. -/
def sortDedup (l : List Nat) : List Nat := tInsertBatch [] l

theorem sortDedup_spec (l : List Nat) : Sorted (sortDedup l) ∧ ∀ e, e ∈ sortDedup l ↔ e ∈ l := by
  refine ⟨sorted_insertBatch l [] List.Pairwise.nil, fun e => ?_⟩
  simp [sortDedup, mem_insertBatch]

/-- Any strictly sorted list with `l`'s members *is* `sortDedup l`: the model of
    `sort_unstable(); dedup()` is determined. -/
theorem sortDedup_unique (l s : List Nat) (hs : Sorted s) (hm : ∀ e, e ∈ s ↔ e ∈ l) : s = sortDedup l :=
  sorted_ext hs (sortDedup_spec l).1 (fun e => by rw [hm, (sortDedup_spec l).2])

/-- `lo > hi` scans nothing (`NumericIndex::empty`, `numeric.rs:252-254`). -/
theorem tRange_empty (t : List Nat) : tRange t 1 0 = [] := by
  simp only [tRange]
  rw [List.filter_eq_nil_iff.2 (fun a _ => by simp; omega)]
  rfl

theorem mem_tRange (t : List Nat) (lo hi d : Nat) (hd : d < W) :
    d ∈ tRange t lo hi ↔ ∃ k, enc k d ∈ t ∧ lo ≤ k ∧ k ≤ hi := by
  simp only [tRange, List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨e, ⟨he, h1, h2⟩, rfl⟩
    exact ⟨keyOf e, by rw [enc_keyOf_docOf]; exact he, h1, h2⟩
  · rintro ⟨k, hk, h1, h2⟩
    exact ⟨enc k d, ⟨hk, by rw [keyOf_enc k d hd]; exact ⟨h1, h2⟩⟩, docOf_enc k d hd⟩

end NI
