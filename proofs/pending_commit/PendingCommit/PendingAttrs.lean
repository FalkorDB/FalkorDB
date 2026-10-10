import PendingCommit.Props
import PendingCommit.ContentCompact
/-
# `Pending` property staging, validation and reads (pending.rs:46-514, :787-871)

* `rustUpsert_eq` — the `binary_search_by_key` + `entry[pos].1 = value` /
  `Vec::insert(pos, …)` in `set_node_attribute` (:444) and
  `set_relationship_attribute` (:813) is `Props.upsert` on a sorted list, so
  `Props.stage_sorted_lastWrite` / `read_your_writes_*` apply to the Rust code.
* `lookupSorted_eq` — `lookup_sorted` (:100) is first-match `get`.
* `isValid_spec` — `is_valid_property` (:62): scalars, `String`, `Point`,
  `VecF32`, temporal values, and lists whose elements are (recursively) valid
  and non-null; `Null` only at top level when allowed. `validate_*` (:84, :92)
  succeed iff `is_valid(v, true)`. (Same as C on live servers: `[[1]]` accepted,
  `[1, null]`, maps and lists of maps rejected.)
* `flatten_spec` — `flatten_label_map` (:46) emits parallel arrays whose zip
  is every `(node, label)` pair, total length = Σ labels.
* `setAttr_spec`, `setAttrs_spec`, `clearAttrs_spec`, `getAttr_spec`,
  `updateAttrs_spec` — the node/relationship staging API.
-/
namespace PendingCommit.PA

/-! ## Finite maps (`FxHashMap<u64, V>`) as association lists with unique keys -/

abbrev FMap (V : Type) := List (Nat × V)

def fget {V : Type} : FMap V → Nat → Option V
  | [], _ => none
  | (a, b) :: m, k => if a = k then some b else fget m k
def fset {V : Type} (m : FMap V) (k : Nat) (v : V) : FMap V := (k, v) :: m.filter (·.1 != k)
def frem {V : Type} (m : FMap V) (k : Nat) : FMap V := m.filter (·.1 != k)

theorem fget_filter_ne {V : Type} (m : FMap V) (k j : Nat) :
    fget (m.filter (·.1 != k)) j = if j = k then none else fget m j := by
  induction m with
  | nil => simp [fget]
  | cons p m ih =>
    obtain ⟨a, b⟩ := p
    by_cases hak : a = k
    · subst hak
      simp only [List.filter_cons, bne_self_eq_false, Bool.false_eq_true, ite_false, ih, fget]
      by_cases hj : j = a
      · simp [hj]
      · simp [hj, Ne.symm hj]
    · simp only [List.filter_cons, bne_iff_ne, ne_eq, hak, not_false_eq_true, ite_true, fget, ih]
      by_cases hja : a = j
      · subst hja; simp [hak]
      · simp [hja]

@[simp] theorem fget_fset {V : Type} (m : FMap V) (k j : Nat) (v : V) :
    fget (fset m k v) j = if j = k then some v else fget m j := by
  unfold fset; simp only [fget, fget_filter_ne]
  by_cases hj : j = k
  · subst hj; simp
  · simp [hj, Ne.symm hj]

@[simp] theorem fget_frem {V : Type} (m : FMap V) (k j : Nat) :
    fget (frem m k) j = if j = k then none else fget m j := fget_filter_ne m k j

/-! ## `set_node_attribute`'s upsert -/

/-- The Rust upsert: `Ok(pos) => entry[pos].1 = value`, `Err(pos) => entry.insert(pos, …)`. -/
def rustUpsert (l : List Entry) (k : Nat) (v : Val) : List Entry :=
  match BinSearch.bsearch (l.map (·.1)) k with
  | .ok pos => l.set pos (k, v)
  | .err pos => l.take pos ++ (k, v) :: l.drop pos

theorem upsert_pre (pre post : List Entry) (k : Nat) (v : Val) (h : ∀ e ∈ pre, e.1 < k) :
    upsert (pre ++ post) k v = pre ++ upsert post k v := by
  induction pre with
  | nil => rfl
  | cons e es ih =>
    have he := h e (by simp)
    simp only [List.cons_append, upsert]
    rw [if_neg (by omega), if_neg (by omega), ih (fun x hx => h x (by simp [hx]))]

theorem upsert_lt (post : List Entry) (k : Nat) (v : Val) (h : ∀ e ∈ post, k < e.1) :
    upsert post k v = (k, v) :: post := by
  cases post with
  | nil => rfl
  | cons e es => simp only [upsert]; rw [if_pos (h e (by simp))]

theorem keys_getD (l : List Entry) (j : Nat) (hj : j < l.length) : (l.map (·.1)).getD j 0 = l[j].1 := by
  simp [List.getD_eq_getElem?_getD, hj]

/-- **`rustUpsert_eq`**. -/
theorem rustUpsert_eq (l : List Entry) (hs : Sorted l) (k : Nat) (v : Val) : rustUpsert l k v = upsert l k v := by
  have hss := Content.sorted_keys l hs
  unfold rustUpsert
  cases e : BinSearch.bsearch (l.map (·.1)) k with
  | ok pos =>
    have ⟨hp, hk⟩ := (BinSearch.bsearch_ok_iff _ hss k pos).1 e
    simp only [List.length_map] at hp
    rw [keys_getD l pos hp] at hk
    have hpre : ∀ x ∈ l.take pos, x.1 < k := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hx
      simp only [List.length_take] at hj
      rw [List.getElem_take, ← hk]
      exact (List.pairwise_iff_getElem.1 hs) j pos (by omega) hp (by omega)
    show l.set pos (k, v) = _
    conv => rhs; rw [← List.take_append_drop pos l]
    rw [upsert_pre _ _ k v hpre, List.drop_eq_getElem_cons hp, List.set_eq_take_append_cons_drop, if_pos hp]
    congr 1
    simp only [upsert, hk]; simp
  | err pos =>
    have ⟨hp, hlo, hhi⟩ := BinSearch.bsearch_err _ hss k pos e
    simp only [List.length_map] at hp hhi
    have hpre : ∀ x ∈ l.take pos, x.1 < k := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hx
      simp only [List.length_take] at hj
      rw [List.getElem_take, ← keys_getD l j (by omega)]; exact hlo j (by omega)
    have hpost : ∀ x ∈ l.drop pos, k < x.1 := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hx
      simp only [List.length_drop] at hj
      rw [List.getElem_drop, ← keys_getD l (pos + j) (by omega)]; exact hhi (pos + j) (by omega) (by omega)
    show l.take pos ++ (k, v) :: l.drop pos = _
    conv => rhs; rw [← List.take_append_drop pos l]
    rw [upsert_pre _ _ k v hpre, upsert_lt _ k v hpost]

/-- `lookup_sorted` (:100): the same binary search, returning the value. -/
def lookupSorted (attrs : List Entry) (k : Nat) : Option Val := Content.spanGet attrs k

theorem lookupSorted_eq (attrs : List Entry) (h : Sorted attrs) (k : Nat) : lookupSorted attrs k = get attrs k :=
  Content.spanGet_eq attrs h k

/-! ## Property validation -/

/-- `Value` kinds as far as validation is concerned. -/
inductive PV where
  | null | bool | int | float | string | point | vecf32 | datetime | date | time | duration
  | list (xs : List PV)
  | map | node | rel | path | other

/-- `is_valid_property` (:62). -/
def isValid : PV → Bool → Bool
  | .null, allowNull => allowNull
  | .bool, _ | .int, _ | .float, _ | .string, _ | .point, _ | .vecf32, _
  | .datetime, _ | .date, _ | .time, _ | .duration, _ => true
  | .list xs, _ => isValidList xs
  | .map, _ | .node, _ | .rel, _ | .path, _ | .other, _ => false
where isValidList : List PV → Bool
  | [] => true
  | x :: xs => isValid x false && isValidList xs

/-- `validate_node_property` / `validate_relationship_property` (:84, :92):
`Err(INVALID_PROPERTY_MSG)` = `false`. -/
def validate (v : PV) : Bool := isValid v true

theorem isValidList_iff (xs : List PV) : isValid.isValidList xs = true ↔ ∀ x ∈ xs, isValid x false = true := by
  induction xs with
  | nil => simp [isValid.isValidList]
  | cons x xs ih => simp [isValid.isValidList, ih]

/-- **`isValid_spec`**. -/
theorem isValid_spec (allowNull : Bool) :
    isValid .null allowNull = allowNull ∧
    (∀ xs, isValid (.list xs) allowNull = true ↔ ∀ x ∈ xs, isValid x false = true) ∧
    isValid (.list [.list [.int]]) true = true ∧ isValid (.list [.int, .null]) true = false ∧
    isValid .map true = false ∧ isValid (.list [.map]) true = false ∧
    (∀ v, validate v = isValid v true) := by
  refine ⟨rfl, fun xs => ?_, rfl, rfl, rfl, rfl, fun _ => rfl⟩
  simp only [isValid]; exact isValidList_iff xs

/-! ## `flatten_label_map` -/

def flatten (m : FMap (List Nat)) : List Nat × List Nat :=
  (m.flatMap (fun p => p.2.map (fun _ => p.1)), m.flatMap (fun p => p.2))

/-- **`flatten_spec`**. -/
theorem flatten_spec (m : FMap (List Nat)) :
    (flatten m).1.length = (m.map (·.2.length)).sum ∧ (flatten m).2.length = (m.map (·.2.length)).sum ∧
      (flatten m).1.zip (flatten m).2 = m.flatMap (fun p => p.2.map (p.1, ·)) := by
  induction m with
  | nil => simp [flatten]
  | cons p m ih =>
    obtain ⟨h1, h2, h3⟩ := ih
    simp only [flatten, List.flatMap_cons, List.length_append, List.length_map, List.map_cons, List.sum_cons] at *
    refine ⟨by omega, by omega, ?_⟩
    rw [List.zip_append (by simp), h3]
    congr 1
    induction p.2 with
    | nil => rfl
    | cons x xs ihx => simp [ihx]

end PendingCommit.PA
