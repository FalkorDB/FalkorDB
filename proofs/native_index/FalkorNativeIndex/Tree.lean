import FalkorNativeIndex.Agree
set_option linter.unusedSimpArgs false

/-!
# The `CowBTree` contract, as its contents

The native index calls six tree operations. Their behaviour on contents is **proven in
`proofs/cow_btree`** (and `proofs/index_layer` for `insert_batch`); here the tree *is* its sorted
entry list and each operation *is* its proven content spec. Copied from
`proofs/cow_btree/FalkorCowBTree/{Model,Basic}.lean` (`enc`/`keyOf`/`docOf`, `Sorted`, `sorted_ext`,
`sins`, `mem_sins`, `sorted_sins`, `mem_erase_sorted`):

| here | Rust | proven contents |
| --- | --- | --- |
| `tInsert` | `CowBTree::insert` (`cow_btree/mod.rs:223`) | cow_btree `insert_spec` (= `sins`) |
| `tRemove` | `CowBTree::remove` (`:260`) | cow_btree `remove_spec` (= `List.erase`) |
| `tInsertBatch` | `CowBTree::insert_batch` (`:241`) | index_layer `Leaf.insertBatch_contents` (= union) |
| `tRemoveBatch` | `CowBTree::remove_batch` (`:286`) | cow_btree `removeBatchTree_spec` (= `List.filter`) |
| `tRange` | `CowBTree::range` / `point` (`:314`, `:324`) | cow_btree `range_spec`, `rangeDocs_spec` |
| `tFromSorted` | `CowBTree::from_sorted` (`:190`) | index_layer (contents = input) |

**The hypothesis this rests on** (so the numeric-index rows that read through the tree are
CONDITIONAL): those specs are proven under cow_btree's `TreeWF`, which `remove_batch` does *not*
preserve (cow_btree Bug 3 = FINDINGS W7-btree-1, `rb_remove_batch_not_wf`; fix pending in PR #3283,
still open at cab3cf192). `remove_spec`'s other premise, `BRANCH_MAX >= 4`, is discharged since #2897
(7b60f28db) by the const assert (cow_btree `remove_spec_asserted`). Since #3307 `remove_batch` has a production
caller (`FalkorDbIndex::apply_writes` → `remove_encoded`). A bounded search over the cow_btree model
(`LEAF_MAX = 2, BRANCH_MAX = 4` and `3, 5`; 300+200 seeds × 30-40 alternating `remove_batch` /
`insert_batch` steps on 40-130 entries) found no content, range or empty-page divergence on the
batch-only paths production uses; Bug 3's panic needs a single-key `remove`, which only tests call.
-/

namespace NI

/-- A `(key, doc)` entry as one number (lexicographic order = `Nat` order). -/
def enc (k d : Nat) : Nat := k * W + d
def keyOf (e : Nat) : Nat := e / W
def docOf (e : Nat) : Nat := e % W

theorem keyOf_enc (k d : Nat) (hd : d < W) : keyOf (enc k d) = k := by
  unfold keyOf enc; rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide : 0 < W)]
  rw [Nat.div_eq_of_lt hd]; omega

theorem docOf_enc (k d : Nat) (hd : d < W) : docOf (enc k d) = d := by
  unfold docOf enc; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]

theorem enc_keyOf_docOf (e : Nat) : enc (keyOf e) (docOf e) = e := by
  unfold enc keyOf docOf; have := Nat.div_add_mod e W; rw [Nat.mul_comm] at this; omega

def Sorted (l : List Nat) : Prop := l.Pairwise (· < ·)

theorem sorted_cons {a : Nat} {l : List Nat} : Sorted (a :: l) ↔ (∀ b ∈ l, a < b) ∧ Sorted l :=
  List.pairwise_cons

theorem sorted_ext : ∀ {l₁ l₂ : List Nat}, Sorted l₁ → Sorted l₂ → (∀ e, e ∈ l₁ ↔ e ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 (List.mem_cons_self)) (List.not_mem_nil)
  | a :: _, [], _, _, h => absurd ((h a).1 (List.mem_cons_self)) (List.not_mem_nil)
  | a :: as, b :: bs, h₁, h₂, h => by
    rw [sorted_cons] at h₁ h₂
    have hab : a = b := by
      have ha := (h a).1 List.mem_cons_self
      have hb := (h b).2 List.mem_cons_self
      rcases List.mem_cons.1 ha with ha | ha
      · exact ha
      rcases List.mem_cons.1 hb with hb | hb
      · exact hb.symm
      have := h₂.1 a ha; have := h₁.1 b hb; omega
    subst hab
    congr 1
    apply sorted_ext h₁.2 h₂.2
    intro e
    constructor
    · intro he
      have := (h e).1 (List.mem_cons_of_mem _ he)
      rcases List.mem_cons.1 this with h' | h'
      · subst h'; have := h₁.1 e he; omega
      · exact h'
    · intro he
      have := (h e).2 (List.mem_cons_of_mem _ he)
      rcases List.mem_cons.1 this with h' | h'
      · subst h'; have := h₂.1 e he; omega
      · exact h'

/-- Sorted-set insert (cow_btree `sins`). -/
def sins (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x < y then x :: y :: ys else if x = y then y :: ys else y :: sins x ys

theorem mem_sins (x : Nat) : ∀ (l : List Nat) (e : Nat), e ∈ sins x l ↔ e ∈ l ∨ e = x
  | [], e => by simp [sins]
  | y :: ys, e => by
    unfold sins
    split
    · simp only [List.mem_cons]
      constructor
      · rintro (h | h | h) <;> simp [h]
      · rintro ((h | h) | h) <;> simp [h]
    · split
      · subst_vars; simp only [List.mem_cons]
        constructor
        · rintro (h | h) <;> simp [h]
        · rintro ((h | h) | h) <;> simp [h]
      · simp only [List.mem_cons, mem_sins x ys e]
        constructor
        · rintro (h | h | h) <;> simp [h]
        · rintro ((h | h) | h) <;> simp [h]

theorem sorted_sins (x : Nat) : ∀ (l : List Nat), Sorted l → Sorted (sins x l)
  | [], _ => by simp [sins, Sorted]
  | y :: ys, h => by
    unfold sins
    rw [sorted_cons] at h
    split
    · rw [sorted_cons]; refine ⟨?_, sorted_cons.2 h⟩
      intro b hb; rcases List.mem_cons.1 hb with hb | hb
      · omega
      · have := h.1 b hb; omega
    · split
      · exact sorted_cons.2 h
      · rw [sorted_cons]; refine ⟨?_, sorted_sins x ys h.2⟩
        intro b hb; rcases (mem_sins x ys b).1 hb with hb | hb
        · exact h.1 b hb
        · omega

theorem mem_erase_sorted (x : Nat) : ∀ (l : List Nat), Sorted l → ∀ e, e ∈ l.erase x ↔ e ∈ l ∧ e ≠ x
  | [], _, e => by simp
  | y :: ys, h, e => by
    rw [sorted_cons] at h
    by_cases hxy : y = x
    · subst hxy
      simp only [List.erase_cons_head]
      constructor
      · intro he; exact ⟨List.mem_cons_of_mem _ he, by have := h.1 e he; omega⟩
      · rintro ⟨he, hne⟩; rcases List.mem_cons.1 he with he | he
        · exact absurd he hne
        · exact he
    · rw [List.erase_cons_tail (by simpa using hxy)]
      simp only [List.mem_cons, mem_erase_sorted x ys h.2 e]
      constructor
      · rintro (he | ⟨he, hne⟩)
        · subst he; exact ⟨Or.inl rfl, hxy⟩
        · exact ⟨Or.inr he, hne⟩
      · rintro ⟨he | he, hne⟩
        · exact Or.inl he
        · exact Or.inr ⟨he, hne⟩

theorem sorted_erase (x : Nat) (l : List Nat) (h : Sorted l) : Sorted (l.erase x) :=
  List.Pairwise.sublist (List.erase_sublist) h

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
