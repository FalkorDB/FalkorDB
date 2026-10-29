/-
# `AttrNameMap` — the graph's one attribute-name dictionary

attribute_store.rs:131-252. `vec` gives id → name, `index` name → id. Ids are
minted only by `insert` (:189) / `get_or_create` (:224) as `vec.len() as u16`;
the cap `MAX_ATTRIBUTES = 65535` (:128) keeps that cast exact and reserves
`ATTRIBUTE_ID_NONE = 65535` (:124).

* `WF` — `index n = some i` iff `vec[i] = n`, and `|vec| ≤ 65535`.
* `insert_wf`, `getOrCreate_wf` — both keep `WF`; `getOrCreate_spec` — the
  returned id names `n` (or is `NONE` exactly when the table is full and `n` is
  new), ids already minted keep their names (`getOrCreate_prefix`), and a
  minted id is `< 65535`, so the `as u16` never wraps (`mint_cast_exact`).
* `uncapped_aliases_zero` — without the cap the 65,537th name would get id 0.
* accessors: `len`, `is_empty`, `get`, `iter`/`into_iter`, `get_index_of`,
  `Index::index` (a panic is `none`).
-/
namespace PendingCommit.NameMap

def MAX : Nat := 65535
def NONE : Nat := 65535

structure M where
  vec : List String
  index : String → Option Nat

/-- `x as u16`. -/
def asU16 (x : Nat) : Nat := x % 65536

def len (m : M) : Nat := m.vec.length                      -- :155
def isEmpty (m : M) : Bool := m.vec.length == 0             -- :160
def get (m : M) (i : Nat) : Option String := m.vec[i]?      -- :165
def iter (m : M) : List String := m.vec                     -- :172
def intoIter (m : M) : List String := iter m                -- :148
def getIndexOf (m : M) (n : String) : Option Nat := m.index n  -- :177
/-- `impl Index<usize>` (:244): `&self.vec[idx]`; `none` = panic. -/
def idx (m : M) (i : Nat) : Option String := m.vec[i]?

def push (m : M) (n : String) : M × Nat :=
  let i := asU16 m.vec.length
  ({ vec := m.vec ++ [n], index := fun x => if x = n then some i else m.index x }, i)

/-- `insert` (:189). -/
def insert (m : M) (n : String) : M :=
  if (m.index n).isSome then m
  else if m.vec.length ≥ MAX then m
  else (push m n).1

/-- `get_or_create` (:224). -/
def getOrCreate (m : M) (n : String) : M × Nat :=
  match m.index n with
  | some i => (m, i)
  | none => if m.vec.length ≥ MAX then (m, NONE) else push m n

structure WF (m : M) : Prop where
  cap : m.vec.length ≤ MAX
  iff : ∀ n i, m.index n = some i ↔ m.vec[i]? = some n

theorem accessors (m : M) (i : Nat) (n : String) :
    len m = m.vec.length ∧ (isEmpty m = true ↔ m.vec = []) ∧ get m i = m.vec[i]? ∧
      intoIter m = m.vec ∧ idx m i = m.vec[i]? ∧ getIndexOf m n = m.index n := by
  refine ⟨rfl, ?_, rfl, rfl, rfl, rfl⟩
  simp [isEmpty]

theorem mint_cast_exact (m : M) (h : m.vec.length < MAX) : asU16 m.vec.length = m.vec.length := by
  unfold asU16 MAX at *; omega

theorem push_wf (m : M) (h : WF m) (n : String) (hn : m.index n = none) (hc : m.vec.length < MAX) :
    WF (push m n).1 := by
  have hcast := mint_cast_exact m hc
  refine ⟨by simp [push]; unfold MAX at *; omega, ?_⟩
  intro x i
  simp only [push, hcast]
  by_cases hx : x = n
  · subst hx
    simp only [ite_true, Option.some.injEq]
    constructor
    · rintro rfl; simp
    · intro hi
      rw [List.getElem?_append] at hi
      split at hi
      · exact absurd ((h.iff x i).2 hi) (by rw [hn]; simp)
      · rename_i hge
        by_cases he : i - m.vec.length = 0
        · omega
        · simp [List.getElem?_singleton, he] at hi
  · simp only [hx, ite_false]
    rw [h.iff x i, List.getElem?_append]
    split
    · rfl
    · rename_i hge
      simp only [List.getElem?_singleton]
      constructor
      · intro h'; exact absurd ((h.iff x i).2 (by
          have : m.vec[i]? = none := List.getElem?_eq_none (by omega)
          rw [this] at h'; cases h')) (by intro; simp_all)
      · intro h'; split at h' <;> simp_all

theorem insert_wf (m : M) (h : WF m) (n : String) : WF (insert m n) := by
  unfold insert
  split
  · exact h
  · split
    · exact h
    · rename_i h1 h2
      exact push_wf m h n (by simpa using h1) (by omega)

theorem getOrCreate_wf (m : M) (h : WF m) (n : String) : WF (getOrCreate m n).1 := by
  unfold getOrCreate
  split
  · exact h
  · rename_i hn; split
    · exact h
    · exact push_wf m h n hn (by omega)

/-- **`getOrCreate_spec`**: the id names `n`, or is `NONE` exactly when `n` is
new and the table is full; a minted id is below the sentinel. -/
theorem getOrCreate_spec (m : M) (h : WF m) (n : String) :
    let r := getOrCreate m n
    (r.2 = NONE ∧ m.index n = none ∧ m.vec.length = MAX ∧ r.1 = m) ∨
      (r.2 < MAX ∧ r.1.vec[r.2]? = some n ∧ r.1.index n = some r.2) := by
  simp only
  unfold getOrCreate
  split
  · rename_i i hi
    right
    have := (h.iff n i).1 hi
    have hlt : i < m.vec.length := by
      rcases Nat.lt_or_ge i m.vec.length with h1 | h1
      · exact h1
      · rw [List.getElem?_eq_none h1] at this; cases this
    exact ⟨by have := h.cap; omega, this, hi⟩
  · rename_i hn
    split
    · left; exact ⟨rfl, hn, by have := h.cap; omega, rfl⟩
    · rename_i hc
      right
      have hcast := mint_cast_exact m (by omega)
      refine ⟨by simp only [push, hcast]; omega, ?_, ?_⟩
      · simp [push, hcast]
      · simp [push, hcast]

/-- Minting never renames an existing id. -/
theorem getOrCreate_prefix (m : M) (n : String) (i : Nat) (hi : i < m.vec.length) :
    (getOrCreate m n).1.vec[i]? = m.vec[i]? := by
  unfold getOrCreate
  split
  · rfl
  · split
    · rfl
    · simp [push, List.getElem?_append_left hi]

/-- Without the `MAX_ATTRIBUTES` cap the 65,537th distinct name is minted as
`65536 as u16 = 0` and aliases attribute 0 (the case the cap exists for). -/
theorem uncapped_aliases_zero (m : M) (h : m.vec.length = 65536) (n : String) :
    (push m n).2 = 0 := by simp [push, asU16, h]

end PendingCommit.NameMap
