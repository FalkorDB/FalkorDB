import FalkorRuntimeDS.OrderMap
/-
# `OrderMap` (`graph/src/runtime/ordermap.rs`) — the remaining methods

Constructors, accessors, `Hash` and `Index`, on the same list model as
`OrderMap.lean` (the `ThinVec<(K, V)>` *is* the list; capacity is invisible).

| here | there |
| --- | --- |
| `emptyM`        | `Default::default` (ordermap.rs:33) |
| `withCapacity`  | `OrderMap::with_capacity` (ordermap.rs:42) |
| `fromUniqueKeys`| `OrderMap::from_unique_keys` (ordermap.rs:57) |
| `reserveExact`  | `OrderMap::reserve_exact` (ordermap.rs:69) |
| `iterM`         | `OrderMap::iter` (ordermap.rs:114) |
| `lenM`          | `OrderMap::len` (ordermap.rs:119) |
| `isEmptyM`      | `OrderMap::is_empty` (ordermap.rs:124) |
| `keysM`         | `OrderMap::keys` (ordermap.rs:128) |
| `valuesM`       | `OrderMap::values` (ordermap.rs:132) |
| `getStr`        | `OrderMap::get_str` (ordermap.rs:140) |
| `hashInput`     | `Hash for OrderMap` (ordermap.rs:158) — `len`, then pairs `sorted_by_key` |
| `indexM`        | `Index<&K>` (ordermap.rs:200) — `.error` = the `expect` panic |
| `intoIterM`     | `IntoIterator` (ordermap.rs:212) |
-/
set_option linter.unusedSectionVars false
namespace FalkorRuntimeDS.OrderMapModel

variable {K V : Type}

/-- ordermap.rs:33 -/
def emptyM : List (K × V) := []
/-- ordermap.rs:42 — `ThinVec::with_capacity`: no entries. -/
def withCapacity (_capacity : Nat) : List (K × V) := []
/-- ordermap.rs:69 — allocation only. -/
def reserveExact (l : List (K × V)) (_additional : Nat) : List (K × V) := l
/-- ordermap.rs:114 -/
def iterM (l : List (K × V)) : List (K × V) := l.map (fun p => (p.1, p.2))
/-- ordermap.rs:119 -/
def lenM (l : List (K × V)) : Nat := l.length
/-- ordermap.rs:124 -/
def isEmptyM (l : List (K × V)) : Bool := l.isEmpty
/-- ordermap.rs:128 -/
def keysM (l : List (K × V)) : List K := l.map (fun p => p.1)
/-- ordermap.rs:132 -/
def valuesM (l : List (K × V)) : List V := l.map (fun p => p.2)
/-- ordermap.rs:212 -/
def intoIterM (l : List (K × V)) : List (K × V) := l

/-- ordermap.rs:140 — scan comparing `k.as_ref() == key_str`. -/
def getStr : List (String × V) → String → Option V
  | [], _ => none
  | (k, v) :: t, s => if k == s then some v else getStr t s

/-- ordermap.rs:200 — `self.get(index).expect("no entry found for key")`. -/
def indexM [BEq K] (l : List (K × V)) (k : K) : Except String V :=
  match getL l k with
  | some v => .ok v
  | none => .error "no entry found for key"

/-- ordermap.rs:158 — what is fed to the hasher: `len`, then every pair in key
order (`sorted_by_key` is a stable sort; `mergeSort` is stable too, and for
unique keys any sort gives the same list, see `hash_consistent`). -/
def hashInput (le : K → K → Bool) (l : List (K × V)) : Nat × List (K × V) :=
  (l.length, l.mergeSort (fun p q => le p.1 q.1))

/-! ## Constructors -/

section Ctors
variable [BEq K]

theorem getL_emptyM (q : K) : getL (emptyM : List (K × V)) q = none := rfl
theorem emptyM_nodup : KeysNodup (emptyM : List (K × V)) := by simp [KeysNodup, emptyM]
theorem withCapacity_eq (n : Nat) : (withCapacity n : List (K × V)) = emptyM := rfl
theorem reserveExact_eq (l : List (K × V)) (n : Nat) : reserveExact l n = l := rfl
theorem getL_reserveExact (l : List (K × V)) (n : Nat) (q : K) :
    getL (reserveExact l n) q = getL l q := rfl

/-- Inserting a key not present appends it and returns `None`. -/
theorem insertL_fresh (l : List (K × V)) (k : K) (v : V) (h : ∀ p ∈ l, (p.1 == k) = false) :
    insertL l k v = (l ++ [(k, v)], none) := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    have hk := h _ (List.mem_cons_self ..)
    simp only at hk
    simp only [insertL, hk, Bool.false_eq_true, ite_false]
    rw [ih (fun p hp => h p (List.mem_cons_of_mem _ hp))]
    rfl

theorem foldl_insert_unique (acc l : List (K × V)) (h : KeysNodup (acc ++ l)) :
    l.foldl (fun acc p => (insertL acc p.1 p.2).1) acc = acc ++ l := by
  induction l generalizing acc with
  | nil => simp
  | cons p t ih =>
    simp only [List.foldl_cons]
    have hp : ∀ q ∈ acc, (q.1 == p.1) = false := by
      intro q hq
      unfold KeysNodup at h
      rw [List.map_append, List.pairwise_append] at h
      exact h.2.2 q.1 (List.mem_map_of_mem hq) p.1 (by simp)
    rw [insertL_fresh _ _ _ hp]
    simp only
    rw [ih (acc ++ [p]) (by simpa using h)]
    simp

/-- `from_unique_keys` honours its contract: on unique keys it builds exactly the map
`from_iter` would (and keeps the pairs verbatim, in order). -/
theorem fromUniqueKeys_eq_fromList (l : List (K × V)) (h : KeysNodup l) :
    fromUniqueKeys l = fromList l := by
  unfold fromUniqueKeys fromList
  rw [foldl_insert_unique [] l (by simpa using h)]
  simp

end Ctors

/-! ## Accessors -/

section Access
variable [BEq K]

theorem iterM_eq (l : List (K × V)) : iterM l = l := by simp [iterM]
theorem intoIterM_eq (l : List (K × V)) : intoIterM l = l := rfl

/-- Iteration order is insertion order: a fresh insert is appended at the end. -/
theorem iterM_insert_fresh (l : List (K × V)) (k : K) (v : V) (h : ∀ p ∈ l, (p.1 == k) = false) :
    iterM (insertL l k v).1 = iterM l ++ [(k, v)] := by
  rw [insertL_fresh _ _ _ h]; simp [iterM_eq]

theorem lenM_insertL (l : List (K × V)) (k : K) (v : V) :
    lenM (insertL l k v).1 = if l.any (fun p => p.1 == k) then lenM l else lenM l + 1 := by
  have := congrArg List.length (keys_insertL l k v)
  simp only [List.length_map] at this
  unfold lenM; rw [this]; split <;> simp

theorem lenM_removeL (l : List (K × V)) (k : K) :
    lenM (removeL l k).1 = if (getL l k).isSome then lenM l - 1 else lenM l := by
  induction l with
  | nil => rfl
  | cons hd t ih =>
    obtain ⟨k', v'⟩ := hd
    unfold lenM at *
    simp only [removeL, getL]
    split
    · simp
    · simp only [List.length_cons, ih]
      split
      · rename_i hs
        have : 0 < t.length := by
          cases t with
          | nil => simp [getL] at hs
          | cons => simp
        omega
      · rfl

theorem isEmptyM_iff (l : List (K × V)) : isEmptyM l = true ↔ lenM l = 0 := by
  cases l <;> simp [isEmptyM, lenM]

theorem isEmptyM_emptyM : isEmptyM (emptyM : List (K × V)) = true := rfl

theorem keysM_eq (l : List (K × V)) : keysM l = l.map Prod.fst := rfl

theorem keysM_insertL (l : List (K × V)) (k : K) (v : V) :
    keysM (insertL l k v).1 = if l.any (fun p => p.1 == k) then keysM l else keysM l ++ [k] :=
  keys_insertL l k v

/-- `keys()` and `values()` walk the same vector in lockstep: zipped, they are `iter()`. -/
theorem keysM_zip_valuesM (l : List (K × V)) : (keysM l).zip (valuesM l) = iterM l := by
  induction l with
  | nil => rfl
  | cons hd t ih => simp [keysM, valuesM, iterM] at *; exact ih

theorem length_valuesM (l : List (K × V)) : (valuesM l).length = lenM l := by simp [valuesM, lenM]

/-- `values()` after an insert on an existing key: that slot's value is replaced in place. -/
theorem valuesM_insertL_fresh (l : List (K × V)) (k : K) (v : V) (h : ∀ p ∈ l, (p.1 == k) = false) :
    valuesM (insertL l k v).1 = valuesM l ++ [v] := by
  rw [insertL_fresh _ _ _ h]; simp [valuesM]

/-- `get_str` is `get` on `Arc<String>` keys (`Arc<String>: PartialEq` compares contents). -/
theorem getStr_eq_getL (l : List (String × V)) (s : String) : getStr l s = getL l s := by
  induction l with
  | nil => rfl
  | cons hd t ih => obtain ⟨k, v⟩ := hd; simp only [getStr, getL, ih]

/-- `m[k]` returns `get(k)` when present and panics exactly when the key is absent. -/
theorem indexM_eq (l : List (K × V)) (k : K) :
    indexM l k = match getL l k with | some v => .ok v | none => .error "no entry found for key" :=
  rfl

theorem indexM_ok_iff (l : List (K × V)) (k : K) (v : V) : indexM l k = .ok v ↔ getL l k = some v := by
  unfold indexM; split <;> simp_all

theorem indexM_panics_iff (l : List (K × V)) (k : K) :
    (∃ e, indexM l k = .error e) ↔ getL l k = none := by
  unfold indexM; split <;> simp_all

end Access

/-! ## `Hash` is consistent with `==` -/

section Hash
variable [DecidableEq K] [DecidableEq V]

theorem eq_of_perm_pairwise {α} (R : α → α → Bool) :
    ∀ {l₁ l₂ : List α}, l₁.Perm l₂ → l₁.Pairwise (fun a b => R a b) → l₂.Pairwise (fun a b => R a b) →
      (∀ x ∈ l₁, ∀ y ∈ l₁, R x y → R y x → x = y) → l₁ = l₂
  | [], l₂, hp, _, _, _ => List.Perm.nil_eq hp
  | x :: t, [], hp, _, _, _ => absurd hp.length_eq (by simp)
  | x :: t, y :: t₂, hp, h₁, h₂, ha => by
    rw [List.pairwise_cons] at h₁ h₂
    have hxy : x = y := by
      by_cases e : x = y
      · exact e
      · have hx : x ∈ t₂ := by
          have := hp.mem_iff.mp (List.mem_cons_self ..)
          simpa [e] using this
        have hy : y ∈ t := by
          have := hp.mem_iff.mpr (List.mem_cons_self ..)
          simpa [Ne.symm e] using this
        exact ha x (List.mem_cons_self ..) y (List.mem_cons_of_mem _ hy) (h₁.1 y hy) (h₂.1 x hx)
    subst hxy
    have ht := (List.perm_cons x).mp hp
    rw [eq_of_perm_pairwise R ht h₁.2 h₂.2
      (fun a ha' b hb => ha a (List.mem_cons_of_mem _ ha') b (List.mem_cons_of_mem _ hb))]

/-- **Hash/Eq contract.** For unique-key maps and a total order on keys, `a == b`
(order-insensitive) implies `hash(a) == hash(b)`: both feed the same length and the
same key-sorted pair sequence to the hasher. -/
theorem hash_consistent (le : K → K → Bool)
    (trans : ∀ a b c, le a b → le b c → le a c) (total : ∀ a b, le a b || le b a)
    (antisymm : ∀ a b, le a b → le b a → a = b)
    {a b : List (K × V)} (ha : KeysNodup a) (hb : KeysNodup b) (h : eqM a b = true) :
    hashInput le a = hashInput le b := by
  have hp := (eqM_iff_perm ha hb).mp h
  unfold hashInput
  refine Prod.ext hp.length_eq ?_
  let R := fun (p q : K × V) => le p.1 q.1
  have tr : ∀ p q r : K × V, R p q → R q r → R p r := fun p q r => trans p.1 q.1 r.1
  have to : ∀ p q : K × V, R p q || R q p := fun p q => total p.1 q.1
  apply eq_of_perm_pairwise R
  · exact ((List.mergeSort_perm a R).trans hp).trans (List.mergeSort_perm b R).symm
  · exact List.pairwise_mergeSort tr to a
  · exact List.pairwise_mergeSort tr to b
  · intro x hx y hy hxy hyx
    have hx' := (List.mergeSort_perm a R).mem_iff.mp hx
    have hy' := (List.mergeSort_perm a R).mem_iff.mp hy
    have hk : x.1 = y.1 := antisymm _ _ hxy hyx
    have e1 := getL_of_mem ha hx'
    have e2 := getL_of_mem ha hy'
    rw [hk, e2] at e1
    cases x; cases y; simp_all

#guard hashInput (fun a b => decide (a ≤ b)) [(2, "b"), (1, "a")] ==
    hashInput (fun a b => decide (a ≤ b)) [(1, "a"), (2, "b")]

end Hash

end FalkorRuntimeDS.OrderMapModel
