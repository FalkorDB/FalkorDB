/-
# `NameMap` — a schema name list; `AttrNameMap` — the graph's attribute-name dictionary

`graph/src/graph/schema.rs`: a `NameMap` is one of `Labels`,
`RelationshipTypes`, `PropertyKeys`, each a `NameList` of `names` (id → name), `ids`
(name → id) and the schema-hash `chain`. `vec`/`index` here are `names`/`ids`. Ids are
minted only by `get_or_insert` (:168) as `names.len()`, refused once `len() ≥
capacity()` (:105: `MAX_ATTRIBUTES = 65535` for property keys, else `usize::MAX`).
`AttrNameMap` (attribute_store.rs) is the `PropertyKeys` map: `insert` (:155) is
`get_or_insert`, `get_or_create` (:182) its id through `u16::try_from`, else
`ATTRIBUTE_ID_NONE = 65535` (:128); the cap keeps every minted id `< 65535`.

* `WF` — `index n = some i` iff `vec[i] = n`, and `|vec| ≤ 65535`.
* `insert_wf`, `getOrCreate_wf` — both keep `WF`; `getOrCreate_spec` — the
  returned id names `n` (or is `NONE` exactly when the table is full and `n` is
  new), ids already minted keep their names (`getOrCreate_prefix`), and a
  minted id is `< 65535`, so it fits a `u16` exactly (`mint_cast_exact`).
* `uncapped_aliases_zero` — with an `as u16` cast and no cap the 65,537th
  name would get id 0; `uncapped_try_from_none` — through `u16::try_from` it is `NONE`.
* accessors: `len`, `is_empty`, `names`, `get`, `iter`/`into_iter`, `id_of`
  (= `get_index_of`), `Index::index` (a panic is `none`).
* `FirstInv` — `index n` is `n`'s first position. `from_names` (:76) builds it
  (`fromNames_first`, also on a malformed list naming a name twice) and keeps every
  position up to the capacity (`fromNames_vec`); `getOrInsert_first` keeps it; `WF`
  implies it (`wf_first`).
-/
namespace PendingCommit.NameMap

def MAX : Nat := 65535
def NONE : Nat := 65535

structure M where
  vec : List String
  index : String → Option Nat

/-- `x as u16`. -/
def asU16 (x : Nat) : Nat := x % 65536

/-- `u16::try_from` (attribute_store.rs). -/
def u16TryFrom (x : Nat) : Option Nat := if x < 65536 then some x else none

def len (m : M) : Nat := m.vec.length                      -- schema.rs
def isEmpty (m : M) : Bool := m.vec.length == 0             -- :134
def names (m : M) : List String := m.vec                    -- :140
def get (m : M) (i : Nat) : Option String := m.vec[i]?      -- :146
def iter (m : M) : List String := m.vec                     -- :153
def intoIter (m : M) : List String := iter m                -- :209
def idOf (m : M) (n : String) : Option Nat := m.index n     -- :159
def getIndexOf (m : M) (n : String) : Option Nat := idOf m n  -- attribute_store.rs
/-- `impl Index<usize>` (schema.rs): `&self.inner().names[id]`; `none` = panic. -/
def idx (m : M) (i : Nat) : Option String := m.vec[i]?

/-- The append of `get_or_insert` (schema.rs): `id = names.len()`, no cast. -/
def push (m : M) (n : String) : M × Nat :=
  let i := m.vec.length
  ({ vec := m.vec ++ [n], index := fun x => if x = n then some i else m.index x }, i)

/-- `get_or_insert` (schema.rs) on `names`/`ids`; `cap` is `capacity()`. -/
def getOrInsert (cap : Nat) (m : M) (n : String) : M × Option Nat :=
  match m.index n with
  | some i => (m, some i)
  | none => if m.vec.length ≥ cap then (m, none) else let r := push m n; (r.1, some r.2)

/-- `AttrNameMap::insert` (attribute_store.rs): `get_or_insert`, result dropped. -/
def insert (m : M) (n : String) : M := (getOrInsert MAX m n).1

/-- `AttrNameMap::get_or_create` (attribute_store.rs). -/
def getOrCreate (m : M) (n : String) : M × Nat :=
  let r := getOrInsert MAX m n
  (r.1, (r.2.bind u16TryFrom).getD NONE)

structure WF (m : M) : Prop where
  cap : m.vec.length ≤ MAX
  iff : ∀ n i, m.index n = some i ↔ m.vec[i]? = some n

theorem accessors (m : M) (i : Nat) (n : String) :
    len m = m.vec.length ∧ (isEmpty m = true ↔ m.vec = []) ∧ get m i = m.vec[i]? ∧
      intoIter m = m.vec ∧ idx m i = m.vec[i]? ∧ getIndexOf m n = m.index n ∧
      names m = m.vec ∧ idOf m n = m.index n := by
  refine ⟨rfl, ?_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  simp [isEmpty]

theorem mint_cast_exact (m : M) (h : m.vec.length < MAX) : asU16 m.vec.length = m.vec.length := by
  unfold asU16 MAX at *; omega

theorem push_wf (m : M) (h : WF m) (n : String) (hn : m.index n = none) (hc : m.vec.length < MAX) :
    WF (push m n).1 := by
  refine ⟨by simp [push]; unfold MAX at *; omega, ?_⟩
  intro x i
  simp only [push]
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
        · simp [he] at hi
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

theorem getOrInsert_wf (m : M) (h : WF m) (n : String) : WF (getOrInsert MAX m n).1 := by
  unfold getOrInsert
  split
  · exact h
  · rename_i hn; split
    · exact h
    · exact push_wf m h n hn (by omega)

theorem insert_wf (m : M) (h : WF m) (n : String) : WF (insert m n) := getOrInsert_wf m h n

theorem getOrCreate_wf (m : M) (h : WF m) (n : String) : WF (getOrCreate m n).1 :=
  getOrInsert_wf m h n

/-- **`getOrCreate_spec`**: the id names `n`, or is `NONE` exactly when `n` is
new and the table is full; a minted id is below the sentinel. -/
theorem getOrCreate_spec (m : M) (h : WF m) (n : String) :
    let r := getOrCreate m n
    (r.2 = NONE ∧ m.index n = none ∧ m.vec.length = MAX ∧ r.1 = m) ∨
      (r.2 < MAX ∧ r.1.vec[r.2]? = some n ∧ r.1.index n = some r.2) := by
  simp only
  unfold getOrCreate getOrInsert
  split
  · rename_i i hi
    right
    have := (h.iff n i).1 hi
    have hlt : i < m.vec.length := by
      rcases Nat.lt_or_ge i m.vec.length with h1 | h1
      · exact h1
      · rw [List.getElem?_eq_none h1] at this; cases this
    have hc := h.cap
    have ht : u16TryFrom i = some i := by unfold u16TryFrom MAX at *; rw [ite_eq_left_iff.mpr (fun h => absurd (by omega) h)]
    simp only [Option.bind_some, ht, Option.getD_some]
    exact ⟨by omega, this, hi⟩
  · rename_i hn
    split
    · left
      exact ⟨rfl, hn, by have := h.cap; omega, rfl⟩
    · rename_i hc
      right
      have ht : u16TryFrom m.vec.length = some m.vec.length := by
        unfold u16TryFrom MAX at *; rw [ite_eq_left_iff.mpr (fun h => absurd (by omega) h)]
      simp only [push, Option.bind_some, ht, Option.getD_some]
      refine ⟨by omega, ?_, ?_⟩
      · simp
      · simp

/-- Minting never renames an existing id. -/
theorem getOrCreate_prefix (m : M) (n : String) (i : Nat) (hi : i < m.vec.length) :
    (getOrCreate m n).1.vec[i]? = m.vec[i]? := by
  unfold getOrCreate getOrInsert
  split
  · rfl
  · split
    · rfl
    · simp [push, List.getElem?_append_left hi]

/-- The append as minted before #3757 (`vec.len() as u16`, attribute_store.rs before
`7ceec2081`). -/
def pushU16 (m : M) (n : String) : M × Nat :=
  let i := asU16 m.vec.length
  ({ vec := m.vec ++ [n], index := fun x => if x = n then some i else m.index x }, i)

/-- Without the `MAX_ATTRIBUTES` cap the 65,537th distinct name is minted as
`65536 as u16 = 0` and aliases attribute 0 (the case the cap exists for). -/
theorem uncapped_aliases_zero (m : M) (h : m.vec.length = 65536) (n : String) :
    (pushU16 m n).2 = 0 := by simp [pushU16, asU16, h]

/-- Since #3757 the id goes through `u16::try_from`: even without the cap the 65,537th
name would be `NONE`, never an alias of attribute 0. -/
theorem uncapped_try_from_none (m : M) (h : m.vec.length = 65536) (n : String)
    (hn : m.index n = none) :
    ((getOrInsert (2 ^ 64 - 1) m n).2.bind u16TryFrom).getD NONE = NONE := by
  simp [getOrInsert, hn, h, push, u16TryFrom]

/-- `push` mints exactly what `pushU16` did below the cap. -/
theorem push_eq_pushU16 (m : M) (h : m.vec.length < MAX) (n : String) : push m n = pushU16 m n := by
  simp [push, pushU16, mint_cast_exact m h]

/-! ## First positions (`from_names`, schema.rs) -/

/-- `n`'s first position in `l`. -/
def first (l : List String) (n : String) : Option Nat := l.findIdx? (fun x => decide (x = n))

/-- `index` is the first-position map of `vec`. -/
def FirstInv (m : M) : Prop := ∀ n, m.index n = first m.vec n

theorem first_snoc (l : List String) (x n : String) :
    first (l ++ [x]) n = (first l n).or (if x = n then some l.length else none) := by
  unfold first
  rw [List.findIdx?_append]
  by_cases h : x = n <;> simp [h]

theorem first_nil (n : String) : first [] n = none := rfl

/-- `ids.entry(name).or_insert(id)` over `names.iter().enumerate()` (schema.rs). -/
def fromIds : List String → Nat → (String → Option Nat) → (String → Option Nat)
  | [], _, f => f
  | x :: xs, i, f =>
    fromIds xs (i + 1) (if (f x).isSome then f else fun y => if y = x then some i else f y)

theorem orInsert_first (p : List String) (f : String → Option Nat) (x : String)
    (hf : ∀ n, f n = first p n) :
    ∀ n, (if (f x).isSome then f else fun y => if y = x then some p.length else f y) n =
      first (p ++ [x]) n := by
  intro n
  rw [first_snoc, ← hf n]
  by_cases hs : (f x).isSome = true
  · simp only [hs, ↓reduceIte]
    by_cases hx : x = n
    · subst hx; cases hfx : f x with
      | none => simp [hfx] at hs
      | some k => simp
    · simp [hx]
  · simp only [hs, Bool.false_eq_true, ↓reduceIte]
    by_cases hx : x = n
    · subst hx
      have : f x = none := by simpa using hs
      simp [this]
    · have : n ≠ x := fun e => hx e.symm
      simp [hx, this]

theorem fromIds_first : ∀ (xs p : List String) (f : String → Option Nat),
    (∀ n, f n = first p n) → ∀ n, fromIds xs p.length f n = first (p ++ xs) n
  | [], p, f, hf, n => by simpa [fromIds] using hf n
  | x :: xs, p, f, hf, n => by
    have := fromIds_first xs (p ++ [x]) _ (orInsert_first p f x hf) n
    simpa [fromIds] using this

/-- **`WF` implies `FirstInv`**: in a well-formed map every name has one position. -/
theorem wf_first (m : M) (h : WF m) : FirstInv m := by
  intro n
  cases hf : first m.vec n with
  | some j =>
    unfold first at hf
    obtain ⟨hj, hx, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
    have hx' : m.vec[j] = n := by simpa using hx
    exact (h.iff n j).2 (by rw [List.getElem?_eq_getElem hj, hx'])
  | none =>
    cases hi : m.index n with
    | none => rfl
    | some i =>
      have := (h.iff n i).1 hi
      unfold first at hf
      have hm := List.findIdx?_eq_none_iff.1 hf n (List.mem_of_getElem? this)
      simp at hm

/-- `get_or_insert` keeps `FirstInv`, whatever the capacity. -/
theorem getOrInsert_first (cap : Nat) (m : M) (h : FirstInv m) (n : String) :
    FirstInv (getOrInsert cap m n).1 := by
  unfold getOrInsert
  split
  · exact h
  · rename_i hn
    split
    · exact h
    · intro y
      simp only [push]
      rw [first_snoc, ← h y]
      by_cases hy : y = n
      · subst hy; simp [hn]
      · have : n ≠ y := fun e => hy e.symm
        simp [hy, this]

/-- **`get_or_insert`'s result**: `n`'s id, or `none` exactly when `n` is new and the
list is at its capacity; a list below capacity always answers. -/
theorem getOrInsert_spec (cap : Nat) (m : M) (h : FirstInv m) (n : String) :
    let r := getOrInsert cap m n
    (r.2 = none ∧ m.index n = none ∧ cap ≤ m.vec.length ∧ r.1 = m) ∨
      (∃ i, r.2 = some i ∧ r.1.vec[i]? = some n ∧ r.1.index n = some i ∧
        ∀ j < r.1.vec.length, j < m.vec.length → r.1.vec[j]? = m.vec[j]?) := by
  simp only
  unfold getOrInsert
  split
  · rename_i i hi
    right
    refine ⟨i, rfl, ?_, hi, fun _ _ _ => rfl⟩
    rw [h n] at hi
    unfold first at hi
    obtain ⟨hj, hx, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hi
    have hx' : m.vec[i] = n := by simpa using hx
    rw [List.getElem?_eq_getElem hj, hx']
  · rename_i hn
    split
    · left; exact ⟨rfl, hn, by omega, rfl⟩
    · right
      refine ⟨m.vec.length, rfl, by simp [push], by simp [push], ?_⟩
      intro j _ hj
      simp [push, List.getElem?_append_left hj]

/-- Below its capacity `get_or_insert` never refuses (labels and relationship types:
`usize::MAX`). -/
theorem getOrInsert_some (cap : Nat) (m : M) (n : String) (hc : m.vec.length < cap) :
    (getOrInsert cap m n).2.isSome := by
  unfold getOrInsert; split
  · rfl
  · rw [ite_eq_right_iff.mpr (fun h => absurd h (by omega))]; rfl

/-! ## The schema-hash chain (schema.rs, schema_hash.rs)

Each list's `chain` is extended (`extend_chain(tag, chain, name)`, abstract `xc` here —
xxh3, see graph_queries `GQ.SchemaHash.push`) exactly where `names` is pushed
(`get_or_insert`, schema.rs), and rebuilt as `chain_of(tag, names)` = `foldl
(extend_chain tag) 0` by `from_names` (:89). `tag` (:94) and `capacity` (:105) come
from the variant. `ChainInvT`: the chain is the fold of the names in id order;
`getOrInsertC_inv` keeps it and `fromNames_chainInv` builds it. `AttrNameMap` is the
`PropertyKeys` map: with `cp = xc 2`, `insertC`/`getOrCreateC` keep `ChainInv` and do to
`vec`/`index` exactly what `insert`/`getOrCreate` do (`*_model`). -/

/-- The `NameMap` variant. -/
inductive Kind where
  | labels
  | relTypes
  | propKeys
  deriving DecidableEq

/-- `NameMap::tag` (schema.rs). -/
def Kind.tag : Kind → Nat
  | .labels => 0
  | .relTypes => 1
  | .propKeys => 2

def USIZE_MAX : Nat := 2 ^ 64 - 1

/-- `NameMap::capacity` (schema.rs). -/
def Kind.capacity : Kind → Nat
  | .propKeys => MAX
  | .labels | .relTypes => USIZE_MAX

structure MC where
  m : M
  chain : Nat

def ChainInvT (xc : Nat → Nat → String → Nat) (tag : Nat) (c : MC) : Prop :=
  c.chain = c.m.vec.foldl (xc tag) 0

def ChainInv (cp : Nat → String → Nat) (c : MC) : Prop := c.chain = c.m.vec.foldl cp 0

/-- `NameMap::get_or_insert` (schema.rs) with its chain. -/
def getOrInsertC (xc : Nat → Nat → String → Nat) (k : Kind) (c : MC) (n : String) :
    MC × Option Nat :=
  match c.m.index n with
  | some i => (c, some i)
  | none => if c.m.vec.length ≥ k.capacity then (c, none) else
      let r := push c.m n; (⟨r.1, xc k.tag c.chain n⟩, some r.2)

/-- `NameMap::from_names(list, names)` (schema.rs): truncate to the capacity,
`ids` by `entry().or_insert` (first position wins), `chain = chain_of(tag, names)`. -/
def fromNames (xc : Nat → Nat → String → Nat) (k : Kind) (names : List String) : MC :=
  let ns := names.take k.capacity
  ⟨⟨ns, fromIds ns 0 (fun _ => none)⟩, ns.foldl (xc k.tag) 0⟩

/-- `NameMap::chain` (schema.rs). -/
def chain (c : MC) : Nat := c.chain

/-- `AttrNameMap::insert` (attribute_store.rs) with its chain. -/
def insertC (cp : Nat → String → Nat) (c : MC) (n : String) : MC :=
  (getOrInsertC (fun _ => cp) .propKeys c n).1

/-- `AttrNameMap::get_or_create` (attribute_store.rs) with its chain. -/
def getOrCreateC (cp : Nat → String → Nat) (c : MC) (n : String) : MC × Nat :=
  let r := getOrInsertC (fun _ => cp) .propKeys c n
  (r.1, (r.2.bind u16TryFrom).getD NONE)

theorem getOrInsertC_model (xc : Nat → Nat → String → Nat) (k : Kind) (c : MC) (n : String) :
    (getOrInsertC xc k c n).1.m = (getOrInsert k.capacity c.m n).1 ∧
      (getOrInsertC xc k c n).2 = (getOrInsert k.capacity c.m n).2 := by
  unfold getOrInsertC getOrInsert; split <;> (try split) <;> simp_all

theorem getOrInsertC_inv (xc : Nat → Nat → String → Nat) (k : Kind) (c : MC) (n : String)
    (h : ChainInvT xc k.tag c) : ChainInvT xc k.tag (getOrInsertC xc k c n).1 := by
  unfold getOrInsertC; split
  · exact h
  · split
    · exact h
    · simp [ChainInvT, push, List.foldl_append]; rw [h]

theorem fromNames_vec (xc : Nat → Nat → String → Nat) (k : Kind) (names : List String) :
    (fromNames xc k names).m.vec = names.take k.capacity := rfl

/-- **`from_names` builds the chain `get_or_insert` would have** (`chain_of`). -/
theorem fromNames_chainInv (xc : Nat → Nat → String → Nat) (k : Kind) (names : List String) :
    ChainInvT xc k.tag (fromNames xc k names) := rfl

/-- **`from_names` finds a name at its first position** (`FirstInv`), duplicates included. -/
theorem fromNames_first (xc : Nat → Nat → String → Nat) (k : Kind) (names : List String) :
    FirstInv (fromNames xc k names).m := by
  intro n
  have := fromIds_first (names.take k.capacity) [] (fun _ => none) (fun n => (first_nil n).symm) n
  simpa [fromNames] using this

/-- Without duplicates (and within the capacity) `from_names` is what inserting the names
in turn builds: the same `vec`, `index` and chain. -/
theorem fromNames_eq_inserts (xc : Nat → Nat → String → Nat) (k : Kind) (names : List String)
    (hc : names.length ≤ k.capacity) :
    (fromNames xc k names).m.vec = names ∧ ChainInvT xc k.tag (fromNames xc k names) :=
  ⟨by simp [fromNames, List.take_of_length_le hc], rfl⟩

theorem insertC_model (cp : Nat → String → Nat) (c : MC) (n : String) :
    (insertC cp c n).m = insert c.m n :=
  (getOrInsertC_model (fun _ => cp) .propKeys c n).1

theorem getOrCreateC_model (cp : Nat → String → Nat) (c : MC) (n : String) :
    (getOrCreateC cp c n).1.m = (getOrCreate c.m n).1 ∧
      (getOrCreateC cp c n).2 = (getOrCreate c.m n).2 := by
  have := getOrInsertC_model (fun _ => cp) .propKeys c n
  simp only [getOrCreateC, getOrCreate]
  exact ⟨this.1, by rw [this.2]; rfl⟩

theorem default_chainInv (cp : Nat → String → Nat) (idx : String → Option Nat) :
    ChainInv cp ⟨⟨[], idx⟩, 0⟩ := rfl

theorem insertC_inv (cp : Nat → String → Nat) (c : MC) (n : String) (h : ChainInv cp c) :
    ChainInv cp (insertC cp c n) :=
  getOrInsertC_inv (fun _ => cp) .propKeys c n h

theorem getOrCreateC_inv (cp : Nat → String → Nat) (c : MC) (n : String) (h : ChainInv cp c) :
    ChainInv cp (getOrCreateC cp c n).1 :=
  getOrInsertC_inv (fun _ => cp) .propKeys c n h

end PendingCommit.NameMap
