/-
CREATE helpers: the attribute template and map-attribute resolution, and label
resolution (graph/src/runtime/ops/create.rs).

| here | there |
| --- | --- |
| `tplSet`, `tplBuild`, `rank`    | `build_attr_template` (create.rs:55) — the `(id, pos, value)` triples |
| `resolveMap`                    | `resolve_map_attrs` (create.rs:84) |
| `resolveLabels`, `resolvePat`   | `Runtime::resolve_pattern` (create.rs:353) |

`resolve` (`get_or_create_*_attr_id`) is a pure name→id map here: within one call the
attribute table only grows and existing names keep their ids (`GocOk` in runtime_core).
-/
namespace CreateAttrs

variable {K E : Type}

/-! ## `build_attr_template` (create.rs:55-80) -/

/-- create.rs:72-76: `template.iter_mut().find(|(k,..)| *k == id)` replaces the first entry
with that id, else pushes. (Shape guard create.rs:61-70: a non-`Map` root or a non-string
key returns `None` — the caller's slow path.) -/
def tplSet : List (Nat × E) → Nat × E → List (Nat × E)
  | [], p => [p]
  | q :: qs, p => if q.1 = p.1 then (q.1, p.2) :: qs else q :: tplSet qs p

def tplBuild (resolve : K → Nat) (entries : List (K × E)) : List (Nat × E) :=
  entries.foldl (fun t ke => tplSet t (resolve ke.1, ke.2)) []

def tplIds (t : List (Nat × E)) : List Nat := t.map (·.1)

def lookup (t : List (Nat × E)) (i : Nat) : Option E := (t.find? (·.1 == i)).map (·.2)

theorem tplSet_ids (t : List (Nat × E)) (p : Nat × E) :
    tplIds (tplSet t p) = if p.1 ∈ tplIds t then tplIds t else tplIds t ++ [p.1] := by
  induction t with
  | nil => simp [tplSet, tplIds]
  | cons q qs ih =>
    unfold tplSet
    by_cases h : q.1 = p.1
    · simp [h, tplIds]
    · simp only [h, ite_false]
      simp only [tplIds, List.map_cons, List.mem_cons] at ih ⊢
      rw [ih]
      by_cases hm : p.1 ∈ qs.map (·.1)
      · simp [hm]
      · have : ¬ (p.1 = q.1 ∨ p.1 ∈ qs.map (·.1)) := by
          rintro (h' | h'); exact h h'.symm; exact hm h'
        simp [hm]; exact fun h' => h h'.symm

theorem tplSet_nodup (t : List (Nat × E)) (p : Nat × E) (h : (tplIds t).Nodup) :
    (tplIds (tplSet t p)).Nodup := by
  rw [tplSet_ids]
  split
  · exact h
  · rename_i hm
    rw [List.nodup_append]
    refine ⟨h, by simp, ?_⟩
    intro a ha b hb; simp at hb; subst hb; intro he; subst he; exact hm ha

theorem tplSet_lookup (t : List (Nat × E)) (p : Nat × E) (i : Nat) :
    lookup (tplSet t p) i = if p.1 = i then some p.2 else lookup t i := by
  induction t with
  | nil => unfold lookup tplSet; by_cases h : p.1 = i <;> simp [h]
  | cons q qs ih =>
    unfold tplSet
    by_cases hq : q.1 = p.1
    · simp only [hq, ite_true]
      by_cases hi : p.1 = i
      · simp [lookup, hi]
      · simp [lookup, hi, List.find?_cons, hq]
    · simp only [hq, ite_false]
      unfold lookup at ih ⊢
      by_cases hqi : q.1 = i
      · have : p.1 ≠ i := fun h => hq (hqi.trans h.symm)
        simp [List.find?_cons, hqi, this]
      · have hb : (q.1 == i) = false := by simp [hqi]
        simp only [List.find?_cons, hb]
        exact ih

/-- The template ids are distinct (so `pos`, the rank in id order, is a permutation). -/
theorem tplBuild_nodup (resolve : K → Nat) (entries : List (K × E)) :
    (tplIds (tplBuild resolve entries)).Nodup := by
  unfold tplBuild
  suffices H : ∀ t : List (Nat × E), (tplIds t).Nodup →
      (tplIds (entries.foldl (fun t ke => tplSet t (resolve ke.1, ke.2)) t)).Nodup from
    H [] List.nodup_nil
  induction entries with
  | nil => intro t h; exact h
  | cons x xs ih => intro t h; exact ih _ (tplSet_nodup _ _ h)

/-- The value kept for a key id is that of its LAST occurrence in the map literal. -/
def lastVal (resolve : K → Nat) (entries : List (K × E)) (i : Nat) : Option E :=
  entries.foldl (fun acc ke => if resolve ke.1 = i then some ke.2 else acc) none

theorem tplBuild_lookup (resolve : K → Nat) (entries : List (K × E)) (i : Nat) :
    lookup (tplBuild resolve entries) i = lastVal resolve entries i := by
  unfold tplBuild lastVal
  suffices H : ∀ t : List (Nat × E),
      lookup (entries.foldl (fun t ke => tplSet t (resolve ke.1, ke.2)) t) i =
      entries.foldl (fun acc ke => if resolve ke.1 = i then some ke.2 else acc) (lookup t i) by
    exact H []
  induction entries with
  | nil => intro t; rfl
  | cons x xs ih => intro t; simp only [List.foldl_cons]; rw [ih, tplSet_lookup]

/-- create.rs:77-80: `pos` = index in the id-sorted order = number of smaller ids. -/
def rank (ids : List Nat) (id : Nat) : Nat := ids.countP (· < id)

theorem rank_cons (x : Nat) (xs : List Nat) (a : Nat) :
    rank (x :: xs) a = rank xs a + (if x < a then 1 else 0) := by
  simp [rank, List.countP_cons]

theorem rank_mono (l : List Nat) (a b : Nat) (h : a ≤ b) : rank l a ≤ rank l b := by
  induction l with
  | nil => simp [rank]
  | cons x xs ih =>
    rw [rank_cons, rank_cons]
    by_cases hx : x < a
    · have : x < b := by omega
      simp only [hx, this, ite_true]; omega
    · by_cases hxb : x < b <;> simp only [hx, hxb, ite_true, ite_false] <;> omega

/-- `pos` is strictly increasing in the id: `out[pos]` is sorted by attribute id. -/
theorem rank_strict (l : List Nat) (a b : Nat) (ha : a ∈ l) (h : a < b) : rank l a < rank l b := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    rw [rank_cons, rank_cons]
    rcases List.mem_cons.mp ha with rfl | ha'
    · have := rank_mono xs a b (by omega)
      simp only [Nat.lt_irrefl, h, ite_true, ite_false]; omega
    · have := ih ha'
      by_cases hx : x < a
      · have : x < b := by omega
        simp only [hx, this, ite_true]; omega
      · by_cases hxb : x < b <;> simp only [hx, hxb, ite_true, ite_false] <;> omega

theorem rank_lt (l : List Nat) (a : Nat) (ha : a ∈ l) : rank l a < l.length := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    rw [rank_cons, List.length_cons]
    rcases List.mem_cons.mp ha with rfl | ha'
    · have : rank xs a ≤ xs.length := List.countP_le_length
      simp only [Nat.lt_irrefl, ite_false]; omega
    · have := ih ha'
      by_cases hx : x < a <;> simp only [hx, ite_true, ite_false] <;> omega

/-! ## `resolve_map_attrs` (create.rs:84) -/

/-- `(resolve(k), v)` per entry, then `sort_unstable_by_key(id)`; ids of a map's distinct keys
are distinct, so the sorted order is unique and equals the stable merge sort. -/
def resolveMap (resolve : K → Nat) (m : List (K × E)) : List (Nat × E) :=
  (m.map fun kv => (resolve kv.1, kv.2)).mergeSort (fun a b => a.1 ≤ b.1)

theorem resolveMap_spec (resolve : K → Nat) (m : List (K × E)) :
    (resolveMap resolve m).Perm (m.map fun kv => (resolve kv.1, kv.2)) ∧
    (resolveMap resolve m).Pairwise (fun a b => a.1 ≤ b.1) := by
  refine ⟨List.mergeSort_perm _ _, ?_⟩
  have := List.pairwise_mergeSort (le := fun (a b : Nat × E) => decide (a.1 ≤ b.1))
    (fun a b c hab hbc => by simp at *; omega) (fun a b => by simp; omega)
    (m.map fun kv => (resolve kv.1, kv.2))
  simp only [decide_eq_true_eq] at this
  exact this

/-! ## `Runtime::resolve_pattern` (create.rs:353) -/

/-- `labels.iter().map(|l| g.get_label_id_mut(l)).collect()`, threading the label table. -/
def resolveLabels {Tb : Type} (goc : Tb → String → Nat × Tb) : Tb → List String → List Nat × Tb
  | t, [] => ([], t)
  | t, l :: ls =>
    let (i, t1) := goc t l
    let (is, t2) := resolveLabels goc t1 ls
    (i :: is, t2)

/-- `get_label_id_mut` contract: returns the name's id in the updated table; existing ids
are kept. -/
def GocOk {Tb : Type} (lk : Tb → String → Option Nat) (goc : Tb → String → Nat × Tb) : Prop :=
  ∀ t n, lk (goc t n).2 n = some (goc t n).1 ∧ ∀ n' i, lk t n' = some i → lk (goc t n).2 n' = some i

/-- Every label is replaced by its id in the final table, positionally. -/
theorem resolveLabels_spec {Tb : Type} (lk : Tb → String → Option Nat) (goc : Tb → String → Nat × Tb)
    (hg : GocOk lk goc) (t : Tb) (ls : List String) :
    (resolveLabels goc t ls).1.length = ls.length ∧
    (∀ n i, lk t n = some i → lk (resolveLabels goc t ls).2 n = some i) ∧
    ∀ k (h1 : k < ls.length) (h2 : k < (resolveLabels goc t ls).1.length),
      lk (resolveLabels goc t ls).2 ls[k] = some (resolveLabels goc t ls).1[k] := by
  induction ls generalizing t with
  | nil => simp [resolveLabels]
  | cons l ls ih =>
    obtain ⟨hl, hm, hk⟩ := ih (goc t l).2
    obtain ⟨g1, g2⟩ := hg t l
    simp only [resolveLabels]
    refine ⟨by simp [hl], fun n i h => hm n i (g2 n i h), fun k h1 h2 => ?_⟩
    cases k with
    | zero => simpa using hm l _ g1
    | succ k => simpa using hk k (by simpa using h1) (by simpa using h2)

structure QNode (L A : Type) where
  alias : Nat
  labels : List L
  attrs : A

/-- Nodes are rebuilt with resolved labels; aliases, attrs (and relationships, paths, which
are copied with their endpoints resolved the same way) keep their shape and order. -/
def resolveNodes {Tb A : Type} (goc : Tb → String → Nat × Tb) : Tb → List (QNode String A) → List (QNode Nat A) × Tb
  | t, [] => ([], t)
  | t, n :: ns =>
    let (ls, t1) := resolveLabels goc t n.labels
    let (rs, t2) := resolveNodes goc t1 ns
    (⟨n.alias, ls, n.attrs⟩ :: rs, t2)

theorem resolveLabels_length {Tb : Type} (goc : Tb → String → Nat × Tb) (t : Tb) (ls : List String) :
    (resolveLabels goc t ls).1.length = ls.length := by
  induction ls generalizing t with
  | nil => rfl
  | cons l ls ih => simp [resolveLabels, ih]

theorem resolveNodes_shape {Tb A : Type} (goc : Tb → String → Nat × Tb) (t : Tb) (ns : List (QNode String A)) :
    (resolveNodes goc t ns).1.map (fun n => (n.alias, n.labels.length)) =
      ns.map (fun n => (n.alias, n.labels.length)) ∧
    (resolveNodes goc t ns).1.map (·.attrs) = ns.map (·.attrs) := by
  induction ns generalizing t with
  | nil => simp [resolveNodes]
  | cons n ns ih =>
    obtain ⟨h1, h2⟩ := ih (resolveLabels goc t n.labels).2
    simp only [resolveNodes, List.map_cons, h1, h2, resolveLabels_length]
    simp

end CreateAttrs
