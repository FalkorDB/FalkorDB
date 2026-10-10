/-
# `OrderMap` (`graph/src/runtime/ordermap.rs`) is an insertion-ordered finite map

Keys are compared with an arbitrary `BEq` (Rust `PartialEq`); the laws we need
are stated per theorem (`EquivBEq` = reflexive/symmetric/transitive, what a
sane `PartialEq` for `Arc<String>` keys satisfies). A replaced entry keeps its
*old* key and gets the new value, exactly like `std::mem::replace(v, value)`.

| here | there |
| --- | --- |
| `OrderMap`        | `struct OrderMap<K, V> { vec: ThinVec<(K, V)> }` (ordermap.rs:28) |
| `insertL`         | `OrderMap::insert` (ordermap.rs:76) |
| `posL`, `removeL` | `OrderMap::remove` (ordermap.rs:91) — `position` then `ThinVec::remove(pos)` |
| `getL`            | `OrderMap::get` (ordermap.rs:102), `get_str` (ordermap.rs:140) |
| `fromList`        | `FromIterator` (ordermap.rs:183), `from_vec` (ordermap.rs:49), `Extend` (ordermap.rs:217) |
| `fromUniqueKeys`  | `OrderMap::from_unique_keys` (ordermap.rs:57) — `debug_assert!` only |
| `eqM`             | `PartialEq for OrderMap` (ordermap.rs:169) |
-/
namespace FalkorRuntimeDS.OrderMapModel

variable {K V : Type}

section Defs
variable [BEq K]

/-- ordermap.rs:76 — scan for `*k == key`; replace value in place, else push. -/
def insertL : List (K × V) → K → V → List (K × V) × Option V
  | [], k, v => ([(k, v)], none)
  | (k', v') :: t, k, v =>
    if k' == k then ((k', v) :: t, some v')
    else let r := insertL t k v; ((k', v') :: r.1, r.2)

/-- `self.vec.iter().position(|(k, _)| k == key)` -/
def posL : List (K × V) → K → Option Nat
  | [], _ => none
  | (k', _) :: t, k => if k' == k then some 0 else (posL t k).map (· + 1)

/-- ordermap.rs:91 — first match removed, returned. -/
def removeL : List (K × V) → K → List (K × V) × Option V
  | [], _ => ([], none)
  | (k', v') :: t, k =>
    if k' == k then (t, some v')
    else let r := removeL t k; ((k', v') :: r.1, r.2)

/-- ordermap.rs:102 — first `k == key`. -/
def getL : List (K × V) → K → Option V
  | [], _ => none
  | (k', v') :: t, k => if k' == k then some v' else getL t k

/-- `FromIterator` / `from_vec` / `extend`: fold `insert` from empty. -/
def fromList (l : List (K × V)) : List (K × V) :=
  l.foldl (fun acc p => (insertL acc p.1 p.2).1) []

/-- `from_unique_keys`: the pairs verbatim (the uniqueness check is a `debug_assert!`). -/
def fromUniqueKeys (l : List (K × V)) : List (K × V) := l

/-- ordermap.rs:169 — same length, and every `(k, v)` of `a` has `b.get(k) == Some(v')`
with `v' == v`. -/
def eqM [BEq V] (a b : List (K × V)) : Bool :=
  a.length == b.length &&
    a.all (fun p => match getL b p.1 with | some ov => ov == p.2 | none => false)

/-- No two entries have `==` keys. -/
def KeysNodup (l : List (K × V)) : Prop :=
  (l.map Prod.fst).Pairwise (fun a b => (a == b) = false)

end Defs

/-! ## Structure of `insert` / `remove` -/

section Structure
variable [BEq K]

/-- `remove` is exactly "find first position, `Vec::remove` it". -/
theorem removeL_eq_eraseIdx (l : List (K × V)) (k : K) :
    (removeL l k).1 = match posL l k with | none => l | some p => l.eraseIdx p := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [removeL, posL]
    by_cases h : (k' == k) = true
    · simp [h]
    · simp only [h, Bool.false_eq_true, ite_false]
      cases hp : posL t k with
      | none => simp [hp] at ih; simp [ih]
      | some p => simp [hp] at ih; simp [ih]

/-- Insertion order: a new key is appended at the end; an existing key keeps its slot. -/
theorem keys_insertL (l : List (K × V)) (k : K) (v : V) :
    (insertL l k v).1.map Prod.fst =
      if l.any (fun p => p.1 == k) then l.map Prod.fst else l.map Prod.fst ++ [k] := by
  induction l with
  | nil => simp [insertL]
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [insertL]
    by_cases h : (k' == k) = true
    · simp [h]
    · simp only [h, Bool.false_eq_true, ite_false, List.map_cons, ih, List.any_cons]
      split <;> simp_all <;> exact ‹_›

/-- `remove` keeps the relative order of the survivors (result is a sublist). -/
theorem removeL_sublist (l : List (K × V)) (k : K) : (removeL l k).1.Sublist l := by
  induction l with
  | nil => simp [removeL]
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [removeL]
    split
    · exact List.Sublist.cons _ (List.Sublist.refl _)
    · exact List.Sublist.cons_cons _ ih

theorem removeL_returns (l : List (K × V)) (k : K) : (removeL l k).2 = getL l k := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [removeL, getL]; split
    · rfl
    · exact ih

theorem insertL_returns (l : List (K × V)) (k : K) (v : V) : (insertL l k v).2 = getL l k := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [insertL, getL]; split
    · rfl
    · exact ih

/-- `insert` never creates a duplicate key. -/
theorem insertL_nodup {l : List (K × V)} (h : KeysNodup l) (k : K) (v : V) :
    KeysNodup (insertL l k v).1 := by
  unfold KeysNodup; rw [keys_insertL]
  split
  · exact h
  · rename_i hany
    rw [List.pairwise_append]
    refine ⟨h, by simp, ?_⟩
    intro a ha b hb
    simp at hb; subst hb
    simp only [List.mem_map] at ha
    obtain ⟨⟨a', va⟩, hmem, rfl⟩ := ha
    simp only [List.any_eq_true, not_exists, not_and] at hany
    simpa using hany _ hmem

theorem removeL_nodup {l : List (K × V)} (h : KeysNodup l) (k : K) :
    KeysNodup (removeL l k).1 :=
  List.Pairwise.sublist ((removeL_sublist l k).map _) h

theorem fromList_nodup (l : List (K × V)) : KeysNodup (fromList l) := by
  unfold fromList
  suffices ∀ acc, KeysNodup acc →
      KeysNodup (l.foldl (fun acc p => (insertL acc p.1 p.2).1) acc) from this [] (by simp [KeysNodup])
  induction l with
  | nil => intro acc h; exact h
  | cons p t ih => intro acc h; exact ih _ (insertL_nodup h _ _)

end Structure

/-! ## Lookup after insert / remove -/

section Lookup
variable [BEq K] [EquivBEq K]

/-- `get` after `insert`: the new value for `k`, everything else untouched.
Holds for *any* vector, duplicates or not. -/
theorem getL_insertL (l : List (K × V)) (k q : K) (v : V) :
    getL (insertL l k v).1 q = if k == q then some v else getL l q := by
  induction l with
  | nil => simp [insertL, getL]
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [insertL]
    by_cases h : (k' == k) = true
    · simp only [h, ite_true, getL]
      rw [BEq.congr_left h]
      by_cases hkq : (k == q) = true <;> simp [hkq]
    · simp only [h, Bool.false_eq_true, ite_false, getL, ih]
      by_cases hq : (k' == q) = true
      · have : (k == q) = false :=
          BEq.symm_false (BEq.neq_of_beq_of_neq (BEq.symm hq) (by simpa using h))
        simp [hq, this]
      · simp [hq]

theorem getL_none_of {l : List (K × V)} {q : K} (h : ∀ p ∈ l, (p.1 == q) = false) :
    getL l q = none := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [getL]
    have := h _ (List.mem_cons_self ..)
    simp only at this
    simp only [this, Bool.false_eq_true, ite_false]
    exact ih (fun p hp => h p (List.mem_cons_of_mem _ hp))

/-- `get` after `remove` (unique keys): `k` is gone, everything else untouched. -/
theorem getL_removeL {l : List (K × V)} (hn : KeysNodup l) (k q : K) :
    getL (removeL l k).1 q = if k == q then none else getL l q := by
  induction l with
  | nil => simp [removeL, getL]
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    unfold KeysNodup at hn
    simp only [List.map_cons, List.pairwise_cons, List.mem_map] at hn
    obtain ⟨hhd, htl⟩ := hn
    simp only [removeL]
    by_cases h : (k' == k) = true
    · simp only [h, ite_true, getL]
      by_cases hkq : (k == q) = true
      · have hk'q : (k' == q) = true := BEq.trans h hkq
        simp only [hkq, ite_true]
        apply getL_none_of
        intro p hp
        have := hhd p.1 ⟨p, hp, rfl⟩
        exact BEq.neq_of_neq_of_beq (BEq.symm_false this) hk'q
      · have : (k' == q) = false := BEq.neq_of_beq_of_neq h (by simpa using hkq)
        simp [hkq, this]
    · simp only [h, Bool.false_eq_true, ite_false, getL, ih htl]
      by_cases hq : (k' == q) = true
      · have : (k == q) = false :=
          BEq.symm_false (BEq.neq_of_beq_of_neq (BEq.symm hq) (by simpa using h))
        simp [hq, this]
      · simp [hq]

/-- `from_iter` / `from_vec`: last write wins, per key. -/
theorem getL_fromList_snoc (l : List (K × V)) (k q : K) (v : V) :
    getL (fromList (l ++ [(k, v)])) q = if k == q then some v else getL (fromList l) q := by
  simp only [fromList, List.foldl_append, List.foldl_cons, List.foldl_nil]
  exact getL_insertL _ _ _ _

end Lookup

/-! ## Equality is order-insensitive (for unique keys) -/

section Eq
variable [DecidableEq K] [DecidableEq V]

theorem mem_of_getL {l : List (K × V)} {k : K} {v : V} (h : getL l k = some v) :
    ∃ k', (k', v) ∈ l ∧ k' = k := by
  induction l with
  | nil => simp [getL] at h
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    simp only [getL] at h
    split at h
    · rename_i hk; simp at h hk; subst h hk; exact ⟨k', by simp, rfl⟩
    · obtain ⟨k'', hm, he⟩ := ih h; exact ⟨k'', List.mem_cons_of_mem _ hm, he⟩

theorem getL_of_mem {l : List (K × V)} (hn : KeysNodup l) {k : K} {v : V} (h : (k, v) ∈ l) :
    getL l k = some v := by
  induction l with
  | nil => simp at h
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    unfold KeysNodup at hn
    simp only [List.map_cons, List.pairwise_cons, List.mem_map] at hn
    simp only [getL]
    rcases List.mem_cons.mp h with h | h
    · simp at h; obtain ⟨rfl, rfl⟩ := h; simp
    · have hne := hn.1 k ⟨(k, v), h, rfl⟩
      simp only [beq_eq_false_iff_ne] at hne
      simp only [beq_iff_eq, show k' ≠ k from hne, ite_false]
      exact ih hn.2 h

theorem nodup_of_keysNodup {l : List (K × V)} (hn : KeysNodup l) : l.Nodup := by
  induction l with
  | nil => simp
  | cons hd t ih =>
    unfold KeysNodup at hn
    simp only [List.map_cons, List.pairwise_cons, List.mem_map] at hn
    refine List.nodup_cons.mpr ⟨?_, ih hn.2⟩
    intro hm
    have := hn.1 hd.1 ⟨hd, hm, rfl⟩
    simp at this

theorem perm_of_subset {α} [DecidableEq α] :
    ∀ {l₁ l₂ : List α}, l₁.Nodup → (∀ a ∈ l₁, a ∈ l₂) → l₂.length ≤ l₁.length → l₁.Perm l₂
  | [], l₂, _, _, hl => by
    have : l₂ = [] := by cases l₂ with | nil => rfl | cons => simp at hl
    subst this; exact List.Perm.refl _
  | a :: t, l₂, hn, hs, hl => by
    have ha : a ∈ l₂ := hs a (List.mem_cons_self ..)
    have hn' := List.nodup_cons.mp hn
    have ht : t.Perm (l₂.erase a) := by
      apply perm_of_subset hn'.2
      · intro b hb
        have hba : b ≠ a := fun e => hn'.1 (e ▸ hb)
        exact (List.mem_erase_of_ne hba).mpr (hs b (List.mem_cons_of_mem _ hb))
      · rw [List.length_erase_of_mem ha]; simp at hl; omega
    exact (ht.cons a).trans (List.perm_cons_erase ha).symm

/-- For unique-key maps, Rust `==` is exactly "same entries, any order". -/
theorem eqM_iff_perm {a b : List (K × V)} (ha : KeysNodup a) (hb : KeysNodup b) :
    eqM a b = true ↔ a.Perm b := by
  constructor
  · intro h
    simp only [eqM, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
    obtain ⟨hlen, hall⟩ := h
    apply perm_of_subset (nodup_of_keysNodup ha) _ (by omega)
    intro ⟨k, v⟩ hm
    have := hall _ hm
    simp only at this
    split at this
    · rename_i ov hg
      simp at this; subst this
      obtain ⟨k', hm', rfl⟩ := mem_of_getL hg
      exact hm'
    · simp at this
  · intro hp
    simp only [eqM, Bool.and_eq_true, beq_iff_eq, List.all_eq_true]
    refine ⟨hp.length_eq, ?_⟩
    intro ⟨k, v⟩ hm
    have := getL_of_mem hb (hp.mem_iff.mp hm)
    simp [this]

/-- …hence symmetric (for unique-key maps). -/
theorem eqM_symm {a b : List (K × V)} (ha : KeysNodup a) (hb : KeysNodup b) :
    eqM a b = eqM b a := by
  apply Bool.eq_iff_iff.mpr
  rw [eqM_iff_perm ha hb, eqM_iff_perm hb ha]
  exact ⟨List.Perm.symm, List.Perm.symm⟩

/-- …and insertion order does not matter. -/
example : eqM (fromList [(1, "a"), (2, "b")]) (fromList [(2, "b"), (1, "a")]) = true := by decide

end Eq

/-! ## Counterexamples: what breaks when `from_unique_keys` is handed duplicates

`from_unique_keys` only checks uniqueness under `debug_assert!`; in a release
build a duplicate key produces a map for which `==` is not symmetric and
`remove` leaves the key reachable. (Reproduced against Rust in
`graph/tests/lean_runtime_ds.rs::ordermap_from_unique_keys_duplicates`.) -/

/-- `a == b` but `b != a`. -/
example :
    eqM (fromUniqueKeys [(0, 1), (0, 1)]) (fromUniqueKeys [(0, 1), (1, 2)]) = true ∧
    eqM (fromUniqueKeys [(0, 1), (1, 2)]) (fromUniqueKeys [(0, 1), (0, 1)]) = false := by decide

/-- `remove(k)` then `get(k)` is still `Some`. -/
example : getL (removeL (fromUniqueKeys [(0, 1), (0, 2)]) 0).1 0 = some 2 := by decide

end FalkorRuntimeDS.OrderMapModel
