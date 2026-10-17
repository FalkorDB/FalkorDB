/-!
# Value glue: constructors, `Clone`, trait declarations, type names (value.rs)

Small, exact theorems for the `value.rs` functions that carry no algorithmic content of
their own but must still be pinned down: the `DeletedNode`/`DeletedRelationship`
snapshots, `Clone for Value`, the four trait *declarations* (`OrderedEnum::order`,
`CompareValue::compare_value`, `ValueTypeOf::value_of_type`, `ValueGetType::get_type`,
`Contains::contains`), `Value::name`, and `ValuesDeduper::with_capacity`.

The value type `W` here has all 16 variants of `enum Value` (value.rs:180). `f64` and `f32`
are *abstract* type parameters `F`, `F32`: nothing about them is assumed in this file.

| here | there |
| --- | --- |
| `W` | `enum Value` value.rs:180 (16 variants) |
| `DeletedNode`, `DeletedNode.new` | value.rs:62-75 |
| `DeletedRelationship`, `DeletedRelationship.new` | value.rs:77-99 |
| `W.clone` | `impl Clone for Value` value.rs:219-240 |
| `W.order`, `OrderedEnum` | trait value.rs:1173, impl :1177 |
| `CompareValue` | trait value.rs:1208 |
| `Ty`, `W.getType`, `ValueGetType` | trait value.rs:1323, impl :1327 |
| `W.valueOfType`, `ValueTypeOf` | trait value.rs:1274, impl :1281 |
| `W.name` | value.rs:1352 |
| `Deduper`, `withCapacity`, `checkAndInsert` | value.rs:1854-1896 |
-/

namespace ValueOrder.Glue

/-- All 16 variants of `enum Value` (value.rs:180). Ids are `Nat`, maps are association
lists (`OrderMap` keeps insertion order), `f64`/`f32` are abstract. -/
inductive W (F F32 : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : F)
  | str (s : String)
  | list (vs : List (W F F32))
  | map (kvs : List (String × W F F32))
  | node (id : Nat)
  | rel (id : Nat)
  | path (vs : List (W F F32))
  | vecf32 (v : List F32)
  | point (lat lon : F32)
  | datetime (t : Int)
  | date (t : Int)
  | time (t : Int)
  | duration (t : Int)

variable {F F32 : Type}

/-! ## Deleted-entity snapshots (value.rs:62-99) -/

/-- `struct DeletedNode { labels, attrs }` (value.rs:62). Labels are a set in Rust
(`HashSet<LabelId>`); a list up to permutation here. -/
structure DeletedNode (V : Type) where
  labels : List Nat
  attrs : List (String × V)

/-- `DeletedNode::new` (value.rs:69). -/
def DeletedNode.new {V} (labels : List Nat) (attrs : List (String × V)) : DeletedNode V :=
  { labels, attrs }

theorem DeletedNode.new_fields {V} (l : List Nat) (a : List (String × V)) :
    (DeletedNode.new l a).labels = l ∧ (DeletedNode.new l a).attrs = a := ⟨rfl, rfl⟩

/-- `struct DeletedRelationship { src, dst, type_name, attrs }` (value.rs:77). -/
structure DeletedRelationship (V : Type) where
  src : Nat
  dst : Nat
  typeName : String
  attrs : List (String × V)

/-- `DeletedRelationship::new` (value.rs:87). -/
def DeletedRelationship.new {V} (src dst : Nat) (typeName : String) (attrs : List (String × V)) :
    DeletedRelationship V := { src, dst, typeName, attrs }

theorem DeletedRelationship.new_fields {V} (s d : Nat) (t : String) (a : List (String × V)) :
    let r := DeletedRelationship.new s d t a
    r.src = s ∧ r.dst = d ∧ r.typeName = t ∧ r.attrs = a := ⟨rfl, rfl, rfl, rfl⟩

/-- The snapshot keeps *every* detail it was given: two snapshots are equal iff all four
fields are (so nothing a later `type(r)`/`startNode(r)`/`properties(r)` could ask for is
dropped at construction). -/
theorem DeletedRelationship.new_injective {V} (s d s' d' : Nat) (t t' : String)
    (a a' : List (String × V)) :
    DeletedRelationship.new s d t a = DeletedRelationship.new s' d' t' a' ↔
      s = s' ∧ d = d' ∧ t = t' ∧ a = a' := by
  constructor
  · intro h; cases h; exact ⟨rfl, rfl, rfl, rfl⟩
  · rintro ⟨rfl, rfl, rfl, rfl⟩; rfl

theorem DeletedNode.new_injective {V} (l l' : List Nat) (a a' : List (String × V)) :
    DeletedNode.new l a = DeletedNode.new l' a' ↔ l = l' ∧ a = a' := by
  constructor
  · intro h; cases h; exact ⟨rfl, rfl⟩
  · rintro ⟨rfl, rfl⟩; rfl

/-! ## `Clone for Value` (value.rs:219-240)

Every arm is a plain copy or an `Arc::clone` (refcount bump, same pointee). As a value
the clone is therefore the same value; the model states each arm explicitly. -/

def W.clone : W F F32 → W F F32
  | .null => .null
  | .bool b => .bool b
  | .int i => .int i
  | .float f => .float f
  | .str s => .str s            -- Arc::clone
  | .list l => .list l          -- Arc::clone
  | .map m => .map m            -- Arc::clone
  | .node n => .node n
  | .rel r => .rel r
  | .path p => .path p          -- Arc::clone
  | .vecf32 v => .vecf32 v      -- Arc::clone
  | .point a b => .point a b    -- Point::clone (two f32 copies)
  | .datetime t => .datetime t
  | .date t => .date t
  | .time t => .time t
  | .duration t => .duration t

theorem W.clone_eq (v : W F F32) : v.clone = v := by cases v <;> rfl

/-! ## `OrderedEnum` (trait value.rs:1173, impl :1177) -/

class OrderedEnum (α : Type) where
  order : α → Nat

/-- `impl OrderedEnum for Value::order` (value.rs:1178), all 16 arms. -/
def W.order : W F F32 → Nat
  | .null => 2^15 | .bool _ => 2^12 | .int _ => 2^13 | .float _ => 2^14
  | .str _ => 2^11 | .list _ => 2^3 | .map _ => 2^0 | .node _ => 2^1 | .rel _ => 2^2
  | .path _ => 2^4 | .point _ _ => 2^5 | .datetime _ => 2^6 | .date _ => 2^7
  | .time _ => 2^8 | .duration _ => 2^10 | .vecf32 _ => 2^18

instance : OrderedEnum (W F F32) := ⟨W.order⟩

/-- The trait method dispatches to the `Value` impl (the only impl of the private trait). -/
theorem orderedEnum_order_eq (v : W F F32) : OrderedEnum.order v = v.order := rfl

/-- The order ranks are pairwise distinct powers of two, and `VecF32` > `Null` > every
other variant (so `Null` sorts last among everything except vectors). -/
theorem order_null_gt (v : W F F32) (h1 : ∀ x, v ≠ .vecf32 x) (h2 : v ≠ .null) :
    v.order < (W.null : W F F32).order := by
  cases v <;> simp_all [W.order]

/-! ## `CompareValue` (trait value.rs:1208) -/

/-- The result flag `DisjointOrNull` (value.rs:1200). -/
inductive DisjointOrNull | disjoint | comparedNull | nan | none
  deriving DecidableEq, Repr

class CompareValue (α : Type) where
  compareValue : α → α → Ordering × DisjointOrNull

/-- `impl PartialEq for Value` (value.rs:1215) is generic in the trait: equality is
`compare_value(..).0 == Equal`, whatever the impl. -/
def eqVia {α} [CompareValue α] (a b : α) : Bool := (CompareValue.compareValue a b).1 == .eq

/-- `impl PartialOrd for Value` (value.rs:1788): `None` exactly on `ComparedNull`. -/
def partialCmpVia {α} [CompareValue α] (a b : α) : Option Ordering :=
  if (CompareValue.compareValue a b).2 = .comparedNull then none
  else some (CompareValue.compareValue a b).1

theorem partialCmpVia_none_iff {α} [CompareValue α] (a b : α) :
    partialCmpVia a b = none ↔ (CompareValue.compareValue a b).2 = .comparedNull := by
  unfold partialCmpVia; split <;> simp_all

theorem partialCmpVia_some {α} [CompareValue α] (a b : α)
    (h : (CompareValue.compareValue a b).2 ≠ .comparedNull) :
    partialCmpVia a b = some (CompareValue.compareValue a b).1 := by
  unfold partialCmpVia; simp [h]

/-! ## Types: `get_type` and `value_of_type` (value.rs:1274-1350) -/

/-- `enum Type` (functions/mod.rs), the constructors `value_of_type` inspects. -/
inductive Ty where
  | any | null | bool | int | float | string | list (t : Ty) | map | node | rel | path
  | vecf32 | point | datetime | date | time | duration
  | optional (t : Ty) | union (ts : List Ty)
  deriving Repr

class ValueGetType (α : Type) where
  getType : α → Ty

/-- `impl ValueGetType for Value::get_type` (value.rs:1328). -/
def W.getType : W F F32 → Ty
  | .null => .null | .bool _ => .bool | .int _ => .int | .float _ => .float
  | .str _ => .string | .list _ => .list .any | .map _ => .map | .node _ => .node
  | .rel _ => .rel | .path _ => .path | .vecf32 _ => .vecf32 | .point _ _ => .point
  | .datetime _ => .datetime | .date _ => .date | .time _ => .time | .duration _ => .duration

instance : ValueGetType (W F F32) := ⟨W.getType⟩

theorem valueGetType_eq (v : W F F32) : ValueGetType.getType v = v.getType := rfl

/-- `get_type` never produces `Any`, `Optional` or `Union`. -/
theorem getType_concrete (v : W F F32) :
    (∀ t, v.getType ≠ .optional t) ∧ (∀ ts, v.getType ≠ .union ts) ∧ v.getType ≠ .any := by
  cases v <;> simp [W.getType]

/-- `Value::name` (value.rs:1352). -/
def W.name : W F F32 → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .float _ => "Float"
  | .str _ => "String" | .list _ => "List" | .map _ => "Map" | .node _ => "Node"
  | .rel _ => "Relationship" | .path _ => "Path" | .vecf32 _ => "VecF32"
  | .point _ _ => "Point" | .datetime _ => "Datetime" | .date _ => "Date"
  | .time _ => "Time" | .duration _ => "Duration"

/-- The name depends only on the variant, and different variants have different names:
`name` is exactly a function of `order` (and vice versa). -/
theorem name_eq_iff_order_eq (a b : W F F32) : a.name = b.name ↔ a.order = b.order := by
  cases a <;> cases b <;> simp [W.name, W.order]

/-- `impl ValueTypeOf for Value::value_of_type` (value.rs:1282), arm by arm. `none` means
"accepted"; `some (actual, expected)` is the mismatch reported. -/
def W.valueOfType : W F F32 → Ty → Option (Ty × Ty)
  | .list vs, .list ty => listFirst vs ty
  | .null, .null | .bool _, .bool | .int _, .int | .float _, .float | .str _, .string
  | .point _ _, .point | .vecf32 _, .vecf32 | .map _, .map | .node _, .node | .rel _, .rel
  | .path _, .path | .datetime _, .datetime | .date _, .date | .time _, .time
  | .duration _, .duration => none
  | _, .any => none
  | v, .optional ty => v.valueOfType ty
  | v, .union tys => unionAll v tys tys
  | v, e => some (v.getType, e)
where
  /-- `for v in vs { if let Some(res) = v.value_of_type(ty) { return Some(res) } } None` -/
  listFirst : List (W F F32) → Ty → Option (Ty × Ty)
    | [], _ => none
    | v :: vs, ty => match v.valueOfType ty with
      | some r => some r
      | none => listFirst vs ty
  /-- `for ty in tys { v.value_of_type(ty)?; } Some((v.get_type(), Union(tys)))`:
  the `?` on an `Option` returns `None` at the FIRST accepting member. -/
  unionAll : W F F32 → List Ty → List Ty → Option (Ty × Ty)
    | v, [], all => some (v.getType, .union all)
    | v, t :: ts, all => match v.valueOfType t with
      | none => none
      | some _ => unionAll v ts all

class ValueTypeOf (α : Type) where
  valueOfType : α → Ty → Option (Ty × Ty)

instance : ValueTypeOf (W F F32) := ⟨W.valueOfType⟩

theorem valueTypeOf_eq (v : W F F32) (t : Ty) : ValueTypeOf.valueOfType v t = v.valueOfType t :=
  rfl

/-- `Any` accepts everything. -/
theorem valueOfType_any (v : W F F32) : v.valueOfType .any = none := by
  cases v <;> simp [W.valueOfType]

/-- Every value is accepted by its own `get_type` (lists: `List(Any)`). -/
theorem valueOfType_self (v : W F F32) : v.valueOfType v.getType = none := by
  cases v with
  | list vs =>
    simp only [W.getType, W.valueOfType]
    induction vs with
    | nil => simp [W.valueOfType.listFirst]
    | cons x xs ih => simp [W.valueOfType.listFirst, valueOfType_any, ih]
  | _ => simp [W.getType, W.valueOfType]

/-! ## `Contains` (trait value.rs:1756, impl for `ThinVec<Value>` :1762) -/

class Contains (C V : Type) where
  contains : C → V → V

/-- `impl Contains for ThinVec<Value>` (value.rs:1763), generic in the comparator. -/
def containsImpl {V} (cmp : V → V → Ordering × DisjointOrNull) (mkNull : V) (mkBool : Bool → V) :
    List V → V → V :=
  fun items value => go value items false
where
  go (value : V) : List V → Bool → V
    | [], isNull => if isNull then mkNull else mkBool false
    | item :: rest, isNull =>
      let (res, dis) := cmp value item
      let isNull := isNull || dis == .comparedNull
      if res = .eq then (if dis = .comparedNull then mkNull else mkBool true)
      else go value rest isNull

/-- With no `ComparedNull` flags and no equal element, the answer is `false`; an equal
element with a clean flag gives `true` at the first such element. -/
theorem containsImpl_false {V} (cmp : V → V → Ordering × DisjointOrNull) (n : V) (b : Bool → V)
    (items : List V) (x : V) (h : ∀ i ∈ items, (cmp x i).1 ≠ .eq ∧ (cmp x i).2 ≠ .comparedNull) :
    containsImpl cmp n b items x = b false := by
  unfold containsImpl
  suffices ∀ acc, acc = false → containsImpl.go cmp n b x items acc = b false by exact this _ rfl
  induction items with
  | nil => intro acc h; simp [containsImpl.go, h]
  | cons i is ih =>
    intro acc hacc
    have ⟨h1, h2⟩ := h i (by simp)
    simp only [containsImpl.go]
    have : ((cmp x i).2 == .comparedNull) = false := by simp [h2]
    simp only [h1, ite_false, hacc, this, Bool.or_self]
    exact ih (fun j hj => h j (by simp [hj])) _ rfl

/-! ## `ValuesDeduper::with_capacity` (value.rs:1859) and the insert protocol -/

/-- The deduper state: the set of hashes seen (a list; capacity is only a hint). -/
structure Deduper where
  seen : List UInt64

/-- `with_capacity(capacity)` (value.rs:1859): an empty set, whatever the capacity. -/
def withCapacity (_capacity : Nat) : Deduper := ⟨[]⟩

/-- `check_and_insert_hash` (value.rs:1887): `!seen.insert(h)`. -/
def checkAndInsert (d : Deduper) (h : UInt64) : Bool × Deduper :=
  if h ∈ d.seen then (true, d) else (false, ⟨h :: d.seen⟩)

theorem withCapacity_empty (c : Nat) (h : UInt64) : h ∉ (withCapacity c).seen := by
  simp [withCapacity]

/-- A fresh deduper reports the first hash as unseen and the same hash again as seen. -/
theorem withCapacity_first_then_seen (c : Nat) (h : UInt64) :
    (checkAndInsert (withCapacity c) h).1 = false ∧
    (checkAndInsert (checkAndInsert (withCapacity c) h).2 h).1 = true := by
  simp [checkAndInsert, withCapacity]

/-- After any sequence of calls, the answer for `h` is `true` iff `h` was offered before. -/
def runAll : Deduper → List UInt64 → Deduper
  | d, [] => d
  | d, h :: hs => runAll (checkAndInsert d h).2 hs

theorem checkAndInsert_mem (d : Deduper) (h x : UInt64) :
    x ∈ (checkAndInsert d h).2.seen ↔ x = h ∨ x ∈ d.seen := by
  unfold checkAndInsert; split
  · constructor
    · intro hx; exact Or.inr hx
    · rintro (rfl | hx) <;> assumption
  · simp

theorem runAll_mem (d : Deduper) (hs : List UInt64) (x : UInt64) :
    x ∈ (runAll d hs).seen ↔ x ∈ hs ∨ x ∈ d.seen := by
  induction hs generalizing d with
  | nil => simp [runAll]
  | cons h hs ih =>
    simp only [runAll, ih, checkAndInsert_mem, List.mem_cons]
    constructor
    · rintro (h1 | h2 | h3)
      · exact Or.inl (Or.inr h1)
      · exact Or.inl (Or.inl h2)
      · exact Or.inr h3
    · rintro ((h1 | h1) | h1)
      · exact Or.inr (Or.inl h1)
      · exact Or.inl h1
      · exact Or.inr (Or.inr h1)

theorem withCapacity_seen_iff (c : Nat) (hs : List UInt64) (h : UInt64) :
    (checkAndInsert (runAll (withCapacity c) hs) h).1 = true ↔ h ∈ hs := by
  have := runAll_mem (withCapacity c) hs h
  simp only [withCapacity, List.not_mem_nil, or_false] at this
  unfold checkAndInsert; simp only [withCapacity] at *
  by_cases hm : h ∈ (runAll { seen := [] } hs).seen <;> simp_all

end ValueOrder.Glue
