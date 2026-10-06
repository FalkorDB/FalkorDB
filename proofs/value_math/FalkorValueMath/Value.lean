import FalkorValueMath.Int64
/-!
# `Value` and its arithmetic (`graph/src/runtime/value.rs:180`, `:903-1171`)

`V F` is `enum Value` with the `f64` payload abstracted to a type `F` whose operations
come from an `FOps F` record (IEEE semantics is *not* assumed unless a theorem says so).
`Arc`s are transparent (`Arc::try_unwrap` success and failure branches are the same
function in the model — `add_slow`'s fast/slow copies are proved to agree by being one
definition). Temporal helpers (`decompose_duration`, `construct_duration_secs`,
`add_duration_to_timestamp`, `sub_duration_from_timestamp`) are parameters: they are
modelled and proved in `proofs/functions_temporal`.
-/

namespace ValueMath

inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : F)
  | str (s : String)
  | list (l : List (V F))
  | map (m : List (String × V F))
  | node (id : Nat)
  | rel (id : Nat)
  | path (l : List (V F))
  | vec (v : List F)
  | point (lat lon : F)
  | datetime (t : Int)
  | date (t : Int)
  | time (t : Int)
  | duration (t : Int)
  deriving Inhabited

variable {F : Type}

/-- `Value::name` (value.rs:1352). -/
def V.name : V F → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .float _ => "Float"
  | .str _ => "String" | .list _ => "List" | .map _ => "Map" | .node _ => "Node"
  | .rel _ => "Relationship" | .path _ => "Path" | .vec _ => "VecF32" | .point .. => "Point"
  | .datetime _ => "Datetime" | .date _ => "Date" | .time _ => "Time"
  | .duration _ => "Duration"

def V.isNull : V F → Bool
  | .null => true
  | _ => false

/-- The float operations the arithmetic uses; `fmt6` is Rust's `{:.6}`. -/
structure FOps (F : Type) where
  ofInt : Int → F
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  rem : F → F → F
  fmt6 : F → String

/-- The temporal helpers of value.rs:696-750 and temporal.rs (proved elsewhere). -/
structure TOps where
  decompose : Int → Except String (Int × Int × Int)
  construct : Int → Int → Int → Except String Int
  addTs : Int → Int → Except String Int
  subTs : Int → Int → Except String Int

/-- `OrderMap::insert` (ordermap.rs:76): replace in place, else push. -/
def omInsert (m : List (String × V F)) (k : String) (v : V F) : List (String × V F) :=
  match m with
  | [] => [(k, v)]
  | (k', v') :: rest => if k' = k then (k, v) :: rest else (k', v') :: omInsert rest k v

/-- `Extend for OrderMap` (ordermap.rs:218): insert each pair in order. -/
def omExtend (m : List (String × V F)) : List (String × V F) → List (String × V F)
  | [] => m
  | (k, v) :: rest => omExtend (omInsert m k v) rest

def lookup (m : List (String × V F)) (k : String) : Option (V F) :=
  match m with
  | [] => none
  | (k', v) :: rest => if k' = k then some v else lookup rest k

def boolStr (b : Bool) : String := if b then "true" else "false"

/-- `Duration + Duration` (value.rs:1008-1019), `ya + yb` etc. in `i32` in Rust; the
year/month sums are exact here (overflow only for year-wrapped durations, see the
functions_temporal header). -/
def durAdd (t : TOps) (a b : Int) (sign : Int) : Except String (V F) := do
  let (ya, ma, sa) ← t.decompose a
  let (yb, mb, sb) ← t.decompose b
  let totalMonths := (ya + sign * yb) * 12 + (ma + sign * mb)
  let ts ← t.construct (Int.tdiv totalMonths 12) (Int.tmod totalMonths 12) (sa + sign * sb)
  pure (.duration ts)

/-- `Value::add_slow` (value.rs:925-1036), arms in source order. -/
def addSlow (o : FOps F) (t : TOps) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .list a, .list b => .ok (.list (a ++ b))
  | .list l, rhs => .ok (.list (l ++ [rhs]))
  | lhs, .list l => .ok (.list (lhs :: l))
  | .map a, .map b => .ok (.map (omExtend a b))
  | .str a, .str b => .ok (.str (a ++ b))
  | .str s, .int i => .ok (.str (s ++ toString i))
  | .str s, .float f => .ok (.str (s ++ o.fmt6 f))
  | .str s, .bool b => .ok (.str (s ++ boolStr b))
  | .int i, .str s => .ok (.str (toString i ++ s))
  | .float f, .str s => .ok (.str (o.fmt6 f ++ s))
  | .bool b, .str s => .ok (.str (boolStr b ++ s))
  | .map _, _ => .error "Cannot merge a map with a non-map value"
  | _, .map _ => .error "Cannot merge a map with a non-map value"
  | .duration a, .duration b => durAdd t a b 1
  | .date d, .duration dur => .date <$> t.addTs d dur
  | .duration dur, .date d => .date <$> t.addTs d dur
  | .datetime d, .duration dur => .datetime <$> t.addTs d dur
  | .duration dur, .datetime d => .datetime <$> t.addTs d dur
  | .time d, .duration dur => .time <$> t.addTs d dur
  | .duration dur, .time d => .time <$> t.addTs d dur
  | a, b => .error s!"Unexpected types for add operator ({a.name}, {b.name})"

/-- `impl Add for Value` (value.rs:910): scalar fast path, else `add_slow`. -/
def add (o : FOps F) (t : TOps) : V F → V F → Except String (V F)
  | .int a, .int b => .ok (.int (wAdd a b))
  | .float a, .float b => .ok (.float (o.add a b))
  | .float a, .int b => .ok (.float (o.add a (o.ofInt b)))
  | .int a, .float b => .ok (.float (o.add (o.ofInt a) b))
  | a, b => addSlow o t a b

/-- `impl Sub for Value` (value.rs:1042). -/
def sub (o : FOps F) (t : TOps) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => .ok (.int (wSub a b))
  | .float a, .float b => .ok (.float (o.sub a b))
  | .float a, .int b => .ok (.float (o.sub a (o.ofInt b)))
  | .int a, .float b => .ok (.float (o.sub (o.ofInt a) b))
  | .duration a, .duration b => durAdd t a b (-1)
  | .date d, .duration dur => .date <$> t.subTs d dur
  | .datetime d, .duration dur => .datetime <$> t.subTs d dur
  | .time d, .duration dur => .time <$> t.subTs d dur
  | .duration _, .date _ => .error "Type mismatch: cannot subtract a temporal value from a duration"
  | .duration _, .datetime _ => .error "Type mismatch: cannot subtract a temporal value from a duration"
  | .duration _, .time _ => .error "Type mismatch: cannot subtract a temporal value from a duration"
  | a, b => .error s!"Unexpected types for sub operator ({a.name}, {b.name})"

def mulMsg (x : V F) : String := s!"Type mismatch: expected Integer, Float, or Null but was {x.name}"

/-- `impl Mul for Value` (value.rs:1091). -/
def mul (o : FOps F) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => .ok (.int (wMul a b))
  | .float a, .float b => .ok (.float (o.mul a b))
  | .float a, .int b => .ok (.float (o.mul a (o.ofInt b)))
  | .int a, .float b => .ok (.float (o.mul (o.ofInt a) b))
  | a, .int _ => .error (mulMsg a)
  | a, .float _ => .error (mulMsg a)
  | .int _, b => .error (mulMsg b)
  | .float _, b => .error (mulMsg b)
  | a, _ => .error (mulMsg a)

def divMsg (a b : V F) : String :=
  s!"Type mismatch: expected Integer, Float, or Null but was ({a.name}, {b.name})"

/-- `impl Div for Value` (value.rs:1120). -/
def div (o : FOps F) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => if b = 0 then .error "Division by zero" else .ok (.int (wDiv a b))
  | .float a, .float b => .ok (.float (o.div a b))
  | .float a, .int b => .ok (.float (o.div a (o.ofInt b)))
  | .int a, .float b => .ok (.float (o.div (o.ofInt a) b))
  | a, b => .error (divMsg a b)

/-- `impl Rem for Value` (value.rs:1148). -/
def rem (o : FOps F) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => if b = 0 then .error "Division by zero" else .ok (.int (wRem a b))
  | .float a, .float b => .ok (.float (o.rem a b))
  | .float a, .int b => .ok (.float (o.rem a (o.ofInt b)))
  | .int a, .float b => .ok (.float (o.rem (o.ofInt a) b))
  | a, b => .error (divMsg a b)

/-! ## Null propagation -/

section
variable (o : FOps F) (t : TOps)

theorem add_null_left (x : V F) : add o t .null x = .ok .null := by
  cases x <;> rfl

theorem add_null_right (x : V F) : add o t x .null = .ok .null := by
  cases x <;> rfl

theorem sub_null_left (x : V F) : sub o t .null x = .ok .null := by cases x <;> rfl
theorem sub_null_right (x : V F) : sub o t x .null = .ok .null := by cases x <;> rfl
theorem mul_null_left (x : V F) : mul o .null x = .ok .null := by cases x <;> rfl
theorem mul_null_right (x : V F) : mul o x .null = .ok .null := by cases x <;> rfl
theorem div_null_left (x : V F) : div o .null x = .ok .null := by cases x <;> rfl
theorem div_null_right (x : V F) : div o x .null = .ok .null := by cases x <;> rfl
theorem rem_null_left (x : V F) : rem o .null x = .ok .null := by cases x <;> rfl
theorem rem_null_right (x : V F) : rem o x .null = .ok .null := by cases x <;> rfl

/-! ## Integer lanes -/

theorem add_int (a b : Int) : add o t (.int a) (.int b) = .ok (.int (wAdd a b)) := rfl
theorem sub_int (a b : Int) : sub o t (.int a) (.int b) = .ok (.int (wSub a b)) := rfl
theorem mul_int (a b : Int) : mul o (.int a) (.int b) = .ok (.int (wMul a b)) := rfl

theorem div_int_zero (a : Int) : div o (.int a) (.int 0) = .error "Division by zero" := rfl
theorem rem_int_zero (a : Int) : rem o (.int a) (.int 0) = .error "Division by zero" := rfl

/-- Integer `+` is commutative and associative (wrapping), so `sum`-style folds do not
depend on evaluation order. -/
theorem add_int_comm (a b : Int) : add o t (.int a) (.int b) = add o t (.int b) (.int a) := by
  simp [add_int, wAdd_comm]

theorem mul_int_comm (a b : Int) : mul o (.int a) (.int b) = mul o (.int b) (.int a) := by
  simp [mul_int, wMul_comm]

/-- `i64::MAX + 1` is `i64::MIN`; C agrees (live: both -9223372036854775808). -/
theorem add_overflow_wraps : add o t (.int I64MAX) (.int 1) = .ok (.int I64MIN) := by
  rw [add_int, add_max_one]

/-! ## Mixed numeric lanes -/

theorem add_mixed (a : F) (b : Int) :
    add o t (.float a) (.int b) = .ok (.float (o.add a (o.ofInt b))) ∧
    add o t (.int b) (.float a) = .ok (.float (o.add (o.ofInt b) a)) := ⟨rfl, rfl⟩

/-! ## Lists (value.rs:931-953) -/

theorem add_list_list (a b : List (V F)) : add o t (.list a) (.list b) = .ok (.list (a ++ b)) := by
  rfl

/-- Appending a non-null, non-list value. -/
theorem add_list_scalar (l : List (V F)) (x : V F) (hn : x ≠ .null) (hl : ∀ m, x ≠ .list m) :
    add o t (.list l) x = .ok (.list (l ++ [x])) := by
  cases x <;> first | rfl | (exact absurd rfl hn) | (exact absurd rfl (hl _))

/-- Prepending a non-null, non-list value. -/
theorem add_scalar_list (l : List (V F)) (x : V F) (hn : x ≠ .null) (hl : ∀ m, x ≠ .list m) :
    add o t x (.list l) = .ok (.list (x :: l)) := by
  cases x <;> first | rfl | (exact absurd rfl hn) | (exact absurd rfl (hl _))

/-- `[1] + null = null` — null wins over list append (C agrees, live). -/
theorem add_list_null (l : List (V F)) : add o t (.list l) .null = .ok .null := rfl

/-! ## Maps (value.rs:954-968): right-biased merge that keeps first-seen key order -/

theorem lookup_omInsert (m : List (String × V F)) (k k' : String) (v : V F) :
    lookup (omInsert m k v) k' = if k = k' then some v else lookup m k' := by
  induction m with
  | nil => simp [omInsert, lookup]
  | cons p rest ih =>
    obtain ⟨k0, v0⟩ := p
    simp only [omInsert]
    by_cases h : k0 = k
    · subst h; by_cases h2 : k0 = k' <;> simp [lookup, h2]
    · simp only [h, if_false, lookup, ih]
      by_cases h2 : k0 = k'
      · subst h2; simp [Ne.symm h]
      · simp [h2]

theorem lookup_append (x y : List (String × V F)) (k : String) :
    lookup (x ++ y) k = (lookup x k).or (lookup y k) := by
  induction x with
  | nil => simp [lookup]
  | cons p rest ih =>
    obtain ⟨k0, v0⟩ := p
    simp only [List.cons_append, lookup]
    split <;> simp_all

/-- Looking a key up in `a + b` finds `b`'s last binding, else `a`'s: right-biased merge. -/
theorem lookup_omExtend (a b : List (String × V F)) (k : String) :
    lookup (omExtend a b) k = (lookup b.reverse k).or (lookup a k) := by
  induction b generalizing a with
  | nil => simp [omExtend, lookup]
  | cons p rest ih =>
    obtain ⟨k0, v0⟩ := p
    simp only [omExtend, ih, lookup_omInsert, List.reverse_cons, lookup_append, lookup]
    cases h : lookup rest.reverse k with
    | some w => simp
    | none => by_cases hk : k0 = k <;> simp [hk]

theorem add_map_lookup (a b : List (String × V F)) (k : String) :
    ∃ m, add o t (.map a) (.map b) = .ok (.map m) ∧
      lookup m k = (lookup b.reverse k).or (lookup a k) :=
  ⟨omExtend a b, rfl, lookup_omExtend a b k⟩

/-- Keys of `omInsert`: an existing key keeps its position, a new key goes last. -/
theorem keys_omInsert (m : List (String × V F)) (k : String) (v : V F) :
    (omInsert m k v).map Prod.fst =
      if k ∈ m.map Prod.fst then m.map Prod.fst else m.map Prod.fst ++ [k] := by
  induction m with
  | nil => simp [omInsert]
  | cons p rest ih =>
    obtain ⟨k0, v0⟩ := p
    simp only [omInsert]
    by_cases h : k0 = k
    · subst h; simp
    · simp only [h, if_false, List.map_cons, ih]
      by_cases hm : k ∈ rest.map Prod.fst
      · have : k ∈ k0 :: rest.map Prod.fst := List.mem_cons_of_mem _ hm
        rw [if_pos hm, if_pos this]
      · have : k ∉ k0 :: rest.map Prod.fst := by
          intro hc
          rcases List.mem_cons.1 hc with e | e
          · exact h e.symm
          · exact hm e
        rw [if_neg hm, if_neg this]; rfl

/-- Map merge never duplicates a key. -/
theorem nodup_omInsert (m : List (String × V F)) (k : String) (v : V F)
    (h : (m.map Prod.fst).Nodup) : ((omInsert m k v).map Prod.fst).Nodup := by
  rw [keys_omInsert]
  split
  · exact h
  · rename_i hk
    rw [List.nodup_append]
    refine ⟨h, by simp, ?_⟩
    intro a ha b hb
    rw [List.mem_singleton] at hb
    subst hb
    intro e; subst e; exact hk ha

theorem nodup_omExtend (a b : List (String × V F)) (h : (a.map Prod.fst).Nodup) :
    ((omExtend a b).map Prod.fst).Nodup := by
  induction b generalizing a with
  | nil => exact h
  | cons p rest ih => exact ih _ (nodup_omInsert _ _ _ h)

/-! ## Strings (value.rs:969-1002) -/

theorem add_str_str (a b : String) : add o t (.str a) (.str b) = .ok (.str (a ++ b)) := rfl
theorem add_str_int (a : String) (i : Int) : add o t (.str a) (.int i) = .ok (.str (a ++ toString i)) := rfl
theorem add_str_float (a : String) (f : F) : add o t (.str a) (.float f) = .ok (.str (a ++ o.fmt6 f)) := rfl

/-- `'a' + {b:1}` is a map error in Rust; C concatenates (`"a{b: 1}"`, live). -/
theorem add_str_map (a : String) (m : List (String × V F)) :
    add o t (.str a) (.map m) = .error "Cannot merge a map with a non-map value" := rfl

/-- `'a' + date(...)` is a type error in Rust; C concatenates (`"a2020-01-01"`, live). -/
theorem add_str_date (a : String) (d : Int) :
    add o t (.str a) (.date d) = .error "Unexpected types for add operator (String, Date)" := rfl

/-- `true + 1` is a type error in Rust; C returns the Float 2 (live). -/
theorem add_bool_int (b : Bool) (i : Int) :
    add o t (.bool b) (.int i) = .error "Unexpected types for add operator (Boolean, Integer)" := rfl

/-! ## Temporal: `+ Duration` is symmetric, `Duration - temporal` rejected -/

theorem add_date_dur_comm (d dur : Int) : add o t (.date d) (.duration dur) = add o t (.duration dur) (.date d) := rfl
theorem add_datetime_dur_comm (d dur : Int) : add o t (.datetime d) (.duration dur) = add o t (.duration dur) (.datetime d) := rfl
theorem add_time_dur_comm (d dur : Int) : add o t (.time d) (.duration dur) = add o t (.duration dur) (.time d) := rfl

theorem sub_dur_date (a d : Int) : sub o t (.duration a) (.date d) =
    .error "Type mismatch: cannot subtract a temporal value from a duration" := rfl

/-! ## `*`: the result is an error exactly when a non-null operand is not a number -/

def V.isNum : V F → Bool
  | .int _ | .float _ => true
  | _ => false

theorem mul_ok_iff (x y : V F) :
    (∃ r, mul o x y = .ok r) ↔ (x.isNull ∨ y.isNull ∨ (x.isNum ∧ y.isNum)) := by
  cases x <;> cases y <;> simp [mul, V.isNull, V.isNum]

theorem div_ok_iff (x y : V F) :
    (∃ r, div o x y = .ok r) ↔
      (x.isNull ∨ y.isNull ∨ (x.isNum ∧ y.isNum ∧ ¬ (∃ a, x = .int a) ∨
        x.isNum ∧ y.isNum ∧ y ≠ .int 0)) := by
  cases x <;> cases y <;> simp [div, V.isNull, V.isNum] <;> split <;> simp_all

theorem rem_ok_iff (x y : V F) :
    (∃ r, rem o x y = .ok r) ↔
      (x.isNull ∨ y.isNull ∨ (x.isNum ∧ y.isNum ∧ ¬ (∃ a, x = .int a) ∨
        x.isNum ∧ y.isNum ∧ y ≠ .int 0)) := by
  cases x <;> cases y <;> simp [rem, V.isNull, V.isNum] <;> split <;> simp_all

end

end ValueMath
