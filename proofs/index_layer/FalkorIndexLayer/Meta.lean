/-
# `Field` and the `Index` metadata (`graph/src/index/mod.rs`)

Line numbers are from origin/main (49f698d22).

| Lean | Rust |
| --- | --- |
| `IType` | `IndexType` `mod.rs:96` |
| `Field`, `makeArrNames`, `Field.new`, `Field.newVec` | `mod.rs:107-168` |
| `Field.beqF`, `Field.hashF` | `PartialEq::eq` `mod.rs:192`, `Hash::hash` `mod.rs:203` |
| `AMap`, `AMap.upd`, `AMap.erase` | the `HashMap<Arc<String>, Vec<Arc<Field>>>` in `Index` (`mod.rs:887`) |
| `Idx` | `Index` `mod.rs:875` (RediSearch spec abstracted to `spec : Option Nat` + registered fields `rs`) |
| `Idx.*` accessors / mutators | `mod.rs:1039-2239` (each cited below) |

A Rust `HashMap` is modelled as an association list with unique keys; its
iteration order is unspecified in Rust, so every theorem below is either
order-independent or about `field_order`, which is the order-carrying list.
-/
namespace IndexLayer.Meta

inductive IType | range | fulltext | vector
  deriving DecidableEq, Repr

/-- `TextIndexOptions` (only the fields the index layer reads). -/
structure TextOpts where
  weight : Option Nat := none
  nostem : Option Bool := none
  phonetic : Option String := none
  language : Option String := none
  stopwords : Option (List String) := none
  deriving DecidableEq, Repr

/-- `VectorIndexOptions`. -/
structure VecOpts where
  dimension : Nat
  sim : Option String
  deriving DecidableEq, Repr

structure Field where
  name : String
  ty : IType
  options : Option TextOpts
  vopts : Option VecOpts
  numArr : Option String
  strArr : Option String
  deriving DecidableEq, Repr

/-- `Field::make_arr_names` (`mod.rs:121`). The `name` is a `CString` built from
a Rust `String`, so `to_str()` succeeds and `format!` adds no NUL: both
`CString::new(..).ok()` are `Some`. -/
def makeArrNames (name : String) (ty : IType) : Option String × Option String :=
  if ty = .range then (some (name ++ ":numeric:arr"), some (name ++ ":string:arr")) else (none, none)

/-- `Field::new` (`mod.rs:137`). -/
def Field.new (name : String) (ty : IType) (o : Option TextOpts) : Field :=
  { name, ty, options := o, vopts := none,
    numArr := (makeArrNames name ty).1, strArr := (makeArrNames name ty).2 }

/-- `Field::new_with_vector_options` (`mod.rs:154`). -/
def Field.newVec (name : String) (ty : IType) (v : VecOpts) : Field :=
  { name, ty, options := none, vopts := some v,
    numArr := (makeArrNames name ty).1, strArr := (makeArrNames name ty).2 }

/-- `PartialEq for Field` (`mod.rs:192`): name and type only. -/
def Field.beqF (a b : Field) : Bool := a.name == b.name && a.ty == b.ty
/-- `Hash for Field` (`mod.rs:203`): hashes the name only (`H` = any hasher). -/
def Field.hashF (H : String → Nat) (f : Field) : Nat := H f.name

theorem makeArrNames_range (n : String) :
    makeArrNames n .range = (some (n ++ ":numeric:arr"), some (n ++ ":string:arr")) := rfl
theorem makeArrNames_other (n : String) (t : IType) (h : t ≠ .range) :
    makeArrNames n t = (none, none) := by simp [makeArrNames, h]

/-- `Field::new`: name/type/options kept, no vector options, array sub-field
names present exactly for Range fields. -/
theorem Field.new_spec (n : String) (t : IType) (o : Option TextOpts) :
    (Field.new n t o).name = n ∧ (Field.new n t o).ty = t ∧ (Field.new n t o).options = o ∧
    (Field.new n t o).vopts = none ∧
    ((Field.new n t o).numArr.isSome ↔ t = .range) ∧ ((Field.new n t o).strArr.isSome ↔ t = .range) := by
  by_cases h : t = .range <;> simp [Field.new, makeArrNames, h]

theorem Field.newVec_spec (n : String) (t : IType) (v : VecOpts) :
    (Field.newVec n t v).name = n ∧ (Field.newVec n t v).ty = t ∧ (Field.newVec n t v).options = none ∧
    (Field.newVec n t v).vopts = some v ∧
    ((Field.newVec n t v).numArr.isSome ↔ t = .range) ∧ ((Field.newVec n t v).strArr.isSome ↔ t = .range) := by
  by_cases h : t = .range <;> simp [Field.newVec, makeArrNames, h]

/-- Accessors `options`/`vector_options`/`numeric_arr_name`/`string_arr_name`
(`mod.rs:171-188`) return the stored components. -/
theorem Field.accessors (n : String) (t : IType) (o : Option TextOpts) (v : Option VecOpts)
    (na sa : Option String) :
    let f : Field := ⟨n, t, o, v, na, sa⟩
    f.options = o ∧ f.vopts = v ∧ f.numArr = na ∧ f.strArr = sa := ⟨rfl, rfl, rfl, rfl⟩

/-- `Eq`/`Hash` consistency (the `HashMap`/`HashSet` contract): equal fields hash equal. -/
theorem Field.eq_hash (H : String → Nat) (a b : Field) (h : a.beqF b = true) :
    a.hashF H = b.hashF H := by
  simp [Field.beqF] at h; simp [Field.hashF, h.1]

theorem Field.beqF_refl (a : Field) : a.beqF a = true := by simp [Field.beqF]
theorem Field.beqF_symm (a b : Field) : a.beqF b = b.beqF a := by
  unfold Field.beqF
  have e1 : (a.name == b.name) = (b.name == a.name) := by
    rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact ⟨Eq.symm, Eq.symm⟩
  have e2 : (a.ty == b.ty) = (b.ty == a.ty) := by
    rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact ⟨Eq.symm, Eq.symm⟩
  rw [e1, e2]

/-! ## Association maps (the `HashMap`) -/

abbrev AMap (β : Type) := List (String × β)

namespace AMap
variable {β : Type}

def keys (m : AMap β) : List String := m.map (·.1)
def get (m : AMap β) (a : String) : Option β := (m.find? (·.1 == a)).map (·.2)
def upd (m : AMap β) (a : String) (v : β) : AMap β :=
  if a ∈ keys m then m.map (fun p => if p.1 = a then (a, v) else p) else m ++ [(a, v)]
def modify (m : AMap β) (a : String) (f : β → β) : AMap β :=
  m.map (fun p => if p.1 = a then (p.1, f p.2) else p)
def erase (m : AMap β) (a : String) : AMap β := m.filter (fun p => p.1 != a)

theorem keys_map_upd (m : AMap β) (a : String) (v : β) :
    keys (m.map (fun p => if p.1 = a then (a, v) else p)) = keys m := by
  induction m with
  | nil => rfl
  | cons p m ih =>
    simp only [List.map_cons, keys] at *
    by_cases h : p.1 = a <;> simp [h, ih]

theorem keys_upd (m : AMap β) (a : String) (v : β) :
    keys (upd m a v) = if a ∈ keys m then keys m else keys m ++ [a] := by
  unfold upd; split
  · exact keys_map_upd m a v
  · simp [keys]

theorem keys_modify (m : AMap β) (a : String) (f : β → β) : keys (modify m a f) = keys m := by
  induction m with
  | nil => rfl
  | cons p m ih =>
    simp only [modify, List.map_cons, keys] at *
    by_cases h : p.1 = a <;> simp [h, ih]

theorem keys_erase (m : AMap β) (a : String) : keys (erase m a) = (keys m).filter (· != a) := by
  induction m with
  | nil => rfl
  | cons p m ih =>
    simp only [erase, keys, List.filter_cons, List.map_cons] at *
    by_cases h : p.1 = a <;> simp [h, ih]

theorem get_cons (p : String × β) (m : AMap β) (a : String) :
    get (p :: m) a = if p.1 = a then some p.2 else get m a := by
  by_cases h : p.1 = a <;> simp [get, List.find?_cons, h]

theorem get_isSome (m : AMap β) (a : String) : (get m a).isSome ↔ a ∈ keys m := by
  induction m with
  | nil => simp [get, keys]
  | cons p m ih =>
    rw [get_cons]; simp only [keys, List.map_cons, List.mem_cons] at *
    by_cases h : p.1 = a
    · simp [h]
    · simp [h, ih, Ne.symm h]

theorem get_mapupd (m : AMap β) (a b : String) (v : β) :
    get (m.map (fun p => if p.1 = a then (a, v) else p)) b =
      if b = a ∧ a ∈ keys m then some v else get m b := by
  induction m with
  | nil => simp [get, keys]
  | cons p m ih =>
    simp only [List.map_cons]
    by_cases h1 : p.1 = a
    · simp only [h1, ite_true, get_cons, ih, keys, List.map_cons, List.mem_cons, true_or, and_true]
      by_cases h2 : a = b
      · subst h2; simp
      · simp [h2, Ne.symm h2]
    · have hk : (a ∈ keys (p :: m)) ↔ a ∈ keys m := by
        simp only [keys, List.map_cons, List.mem_cons]; exact ⟨fun h => h.resolve_left (fun e => h1 e.symm), Or.inr⟩
      simp only [h1, ite_false, get_cons, ih]
      by_cases h2 : p.1 = b
      · have : ¬ b = a := fun e => h1 (h2.trans e)
        simp [h2, this]
      · simp only [h2, ite_false]
        by_cases h3 : b = a
        · simp [h3, hk]
        · simp [h3]

theorem get_append (m : AMap β) (a b : String) (v : β) :
    get (m ++ [(a, v)]) b = match get m b with | some x => some x | none => if a = b then some v else none := by
  induction m with
  | nil => by_cases h : a = b <;> simp [get, h]
  | cons p m ih =>
    simp only [List.cons_append, get_cons, ih]
    by_cases h : p.1 = b <;> simp [h]

theorem get_upd (m : AMap β) (a b : String) (v : β) :
    get (upd m a v) b = if b = a then some v else get m b := by
  unfold upd; split
  · next h => rw [get_mapupd]; by_cases hb : b = a <;> simp [hb, h]
  · next h =>
    rw [get_append]
    by_cases hb : b = a
    · subst hb
      have : get m b = none := by
        cases hg : get m b
        · rfl
        · exact absurd ((get_isSome m b).mp (by simp [hg])) h
      simp [this]
    · cases get m b <;> simp [hb, Ne.symm hb]

theorem get_modify (m : AMap β) (a b : String) (f : β → β) :
    get (modify m a f) b = if b = a then (get m b).map f else get m b := by
  induction m with
  | nil => simp [modify, get]
  | cons p m ih =>
    simp only [modify, List.map_cons] at *
    by_cases h1 : p.1 = a
    · simp only [h1, ite_true, get_cons, ih]
      by_cases h2 : a = b
      · subst h2; simp
      · simp [h2, Ne.symm h2]
    · simp only [h1, ite_false, get_cons, ih]
      by_cases h2 : p.1 = b
      · subst h2; simp [h1]
      · simp [h2]

theorem get_erase (m : AMap β) (a b : String) :
    get (erase m a) b = if b = a then none else get m b := by
  induction m with
  | nil => simp [erase, get]
  | cons p m ih =>
    by_cases h1 : p.1 = a
    · have e : erase (p :: m) a = erase m a := by simp [erase, List.filter_cons, h1]
      rw [e, ih, get_cons]
      by_cases h2 : b = a
      · simp [h2]
      · have : ¬ p.1 = b := fun h => h2 (h.symm.trans h1)
        simp [h2, this]
    · have e : erase (p :: m) a = p :: erase m a := by simp [erase, List.filter_cons, h1]
      rw [e, get_cons, ih, get_cons]
      by_cases h2 : p.1 = b
      · have : ¬ b = a := fun h => h1 (h2.trans h)
        simp [h2, this]
      · simp [h2]

theorem nodup_keys_upd (m : AMap β) (a : String) (v : β) (h : (keys m).Nodup) :
    (keys (upd m a v)).Nodup := by
  rw [keys_upd]; split
  · exact h
  · next ha => exact List.nodup_append.mpr ⟨h, by simp, by simp; intro x hx hxa; subst hxa; exact ha hx⟩

theorem nodup_keys_erase (m : AMap β) (a : String) (h : (keys m).Nodup) : (keys (erase m a)).Nodup := by
  rw [keys_erase]; exact h.filter _

end AMap
end IndexLayer.Meta
