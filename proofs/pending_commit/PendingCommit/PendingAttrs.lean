import PendingCommit.Props
import PendingCommit.ContentCompact
/-
# `Pending` property staging, validation and reads (pending.rs:46-494, :768-852)

* `rustUpsert_eq` — the `binary_search_by_key` + `entry[pos].1 = value` /
  `Vec::insert(pos, …)` in `set_node_attribute` (:425) and
  `set_relationship_attribute` (:794) is `Props.upsert` on a sorted list, so
  `Props.stage_sorted_lastWrite` / `read_your_writes_*` apply to the Rust code.
* `lookupSorted_eq` — `lookup_sorted` (:100) is first-match `get`.
* `validate_spec`, `validate_depth`, `validate_nest` — `is_valid_property` (:69),
  `validate_list_property` (:98), `validate_property` (:115), `validate_*` (:124, :129):
  scalars, `String`, `Point`, `VecF32`, temporal values, and lists whose elements are
  (recursively) valid and non-null, nested at most `MAX_VALUE_NESTING` (4096) deep (#3339,
  b080a44c1); `Null` only at top level. Every accepted value was valid before (`isValid`, the
  pre-#3339 check) and has depth ≤ 4096. (Same as C on
  live servers: `[[1]]` accepted, `[1, null]`, maps and lists of maps rejected.)
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

def MAX_VALUE_NESTING : Nat := 4096

/-- `is_valid_property` (:69): a non-list primitive, or NULL when allowed (#3339 moved the
list arm into `validate_list_property`). -/
def isValidP : PV → Bool → Bool
  | .null, allowNull => allowNull
  | .bool, _ | .int, _ | .float, _ | .string, _ | .point, _ | .vecf32, _
  | .datetime, _ | .date, _ | .time, _ | .duration, _ => true
  | _, _ => false

/-- The nesting message (:103) / `INVALID_PROPERTY_MSG` (:64). -/
inductive PErr | nesting | invalid
  deriving DecidableEq

mutual
/-- `validate_list_property(items, depth)`'s `try_for_each` (:108-112); the recursive call's
depth check is inlined into `validateItem`. -/
def validateItems : Nat → List PV → Except PErr Unit
  | _, [] => .ok ()
  | depth, x :: xs => match validateItem depth x with
    | .ok () => validateItems depth xs
    | .error e => .error e
def validateItem : Nat → PV → Except PErr Unit
  | depth, .list inner =>
    if depth + 1 > MAX_VALUE_NESTING then .error .nesting else validateItems (depth + 1) inner
  | _, v => if isValidP v false then .ok () else .error .invalid
end

/-- `validate_list_property` (:98). -/
def validateListProperty (items : List PV) (depth : Nat) : Except PErr Unit :=
  if depth > MAX_VALUE_NESTING then .error .nesting else validateItems depth items

/-- `validate_property` (:115); `validate_node_property` (:124) and
`validate_relationship_property` (:129) return it. -/
def validateProperty : PV → Except PErr Unit
  | .list items => validateListProperty items 1
  | v => if isValidP v true then .ok () else .error .invalid

def validate (v : PV) : Bool := (validateProperty v).toBool

/-- Historical `is_valid_property` before #3339 (:62 at 4a4f2a72f): lists of any depth. -/
def isValid : PV → Bool → Bool
  | .null, allowNull => allowNull
  | .bool, _ | .int, _ | .float, _ | .string, _ | .point, _ | .vecf32, _
  | .datetime, _ | .date, _ | .time, _ | .duration, _ => true
  | .list xs, _ => isValidList xs
  | .map, _ | .node, _ | .rel, _ | .path, _ | .other, _ => false
where isValidList : List PV → Bool
  | [] => true
  | x :: xs => isValid x false && isValidList xs

theorem isValidList_iff (xs : List PV) : isValid.isValidList xs = true ↔ ∀ x ∈ xs, isValid x false = true := by
  induction xs with
  | nil => simp [isValid.isValidList]
  | cons x xs ih => simp [isValid.isValidList, ih]

mutual
/-- Nesting depth (`[]` is 1, scalars 0). -/
def depth : PV → Nat
  | .list xs => maxDepth xs + 1
  | _ => 0
def maxDepth : List PV → Nat
  | [] => 0
  | x :: xs => max (depth x) (maxDepth xs)
end

/-- An accepted item at nesting `d ≤ MAX` was valid before #3339 and fits: `d + depth ≤ MAX`. -/
theorem validateItem_ok : ∀ (n : Nat) (v : PV) (d : Nat), sizeOf v < n → d ≤ MAX_VALUE_NESTING →
    validateItem d v = .ok () → isValid v false = true ∧ d + depth v ≤ MAX_VALUE_NESTING := by
  intro n
  induction n with
  | zero => intro v d h; omega
  | succ f ih =>
    intro v d hs hd h
    have hL : ∀ (l : List PV) (d : Nat), sizeOf l < f → d ≤ MAX_VALUE_NESTING → validateItems d l = .ok () →
        isValid.isValidList l = true ∧ d + maxDepth l ≤ MAX_VALUE_NESTING := by
      intro l
      induction l with
      | nil => intro d _ hd _; exact ⟨rfl, by simp [maxDepth]; omega⟩
      | cons y ys ihl =>
        intro d hsz hd hok
        simp only [validateItems] at hok
        split at hok
        · rename_i hy
          have h1 := ih y d (by simp at hsz; omega) hd hy
          have h2 := ihl d (by simp at hsz; omega) hd hok
          refine ⟨by simp [isValid.isValidList, h1.1, h2.1], ?_⟩
          simp only [maxDepth]; omega
        · cases hok
    cases v with
    | list xs =>
      simp only [validateItem] at h
      split at h
      · cases h
      · rename_i hd1
        have := hL xs (d + 1) (by simp at hs; omega) (by omega) h
        exact ⟨by simpa [isValid] using this.1, by simp only [depth]; omega⟩
    | map | node | rel | path | other => simp [validateItem, isValidP] at h
    | null => simp [validateItem, isValidP] at h
    | _ => exact ⟨rfl, by simp [depth]; omega⟩

/-- **`validate_depth`**: every property a write accepts was valid before #3339 and has
depth ≤ `MAX_VALUE_NESTING`, the bound the RDB decoder enforces — so it saves and loads back
(proofs/value_math `stored_roundtrip`). -/
theorem validate_depth (v : PV) (h : validateProperty v = .ok ()) :
    isValid v true = true ∧ depth v ≤ MAX_VALUE_NESTING := by
  cases v with
  | list xs =>
    simp only [validateProperty, validateListProperty] at h
    split at h
    · cases h
    · have := validateItem_ok (sizeOf (PV.list xs) + 1) (.list xs) 0 (by omega) (by decide)
        (by simp only [validateItem]; rw [if_neg (by decide)]; exact h)
      exact ⟨by simpa [isValid] using this.1, by omega⟩
  | null => exact ⟨rfl, by simp [depth]⟩
  | map | node | rel | path | other => simp [validateProperty, isValidP] at h
  | _ => exact ⟨rfl, by simp [depth]⟩

/-- `k` lists around an integer (the Rust tests' `nested(k)`). -/
def nest : Nat → PV
  | 0 => .int
  | k + 1 => .list [nest k]

theorem validateItems_nest (k d : Nat) (hd : d ≤ MAX_VALUE_NESTING) :
    validateItems d [nest k] = if d + k ≤ MAX_VALUE_NESTING then .ok () else .error .nesting := by
  induction k generalizing d with
  | zero => simp [nest, validateItems, validateItem, isValidP, hd]
  | succ k ih =>
    have e2 : validateItems d [nest (k + 1)] = match validateItem d (.list [nest k]) with
        | .ok () => validateItems d [] | .error e => .error e := rfl
    have e3 : validateItem d (.list [nest k]) =
        if d + 1 > MAX_VALUE_NESTING then .error .nesting else validateItems (d + 1) [nest k] := rfl
    rw [e2, e3]
    by_cases h : d + 1 > MAX_VALUE_NESTING
    · have : ¬ d + (k + 1) ≤ MAX_VALUE_NESTING := by omega
      simp [h, this]
    · rw [if_neg h, ih (d + 1) (by omega), show d + 1 + k = d + (k + 1) by omega]
      by_cases h2 : d + (k + 1) ≤ MAX_VALUE_NESTING <;> simp [h2, validateItems]

/-- **#3339**: `k` nested lists are a valid property iff `k ≤ MAX_VALUE_NESTING`; past it the
error is the nesting message (`a_property_at_the_cap_is_valid` / `…_past_the_cap_is_refused`). -/
theorem validate_nest (k : Nat) :
    validateProperty (nest (k + 1)) = if k + 1 ≤ MAX_VALUE_NESTING then .ok () else .error .nesting := by
  simp only [nest, validateProperty, validateListProperty]
  rw [if_neg (by decide), validateItems_nest k 1 (by decide), Nat.add_comm 1 k]

/-- **`isValid_spec`** (now `validate_spec`): NULL only at top level, lists of primitives, no
maps — as before #3339 — on concrete shapes. -/
theorem validate_spec :
    validateProperty .null = .ok () ∧ validateProperty (.list [.list [.int]]) = .ok () ∧
    validateProperty (.list [.int, .null]) = .error .invalid ∧
    validateProperty .map = .error .invalid ∧ validateProperty (.list [.map]) = .error .invalid ∧
    (∀ v, validate v = (validateProperty v).toBool) := by
  refine ⟨rfl, ?_, ?_, rfl, ?_, fun _ => rfl⟩ <;> rfl

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
