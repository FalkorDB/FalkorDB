import Columnar.VectorExpr

/-
# Arithmetic lanes (`vector_expr.rs:880-994`) against `Value` arithmetic

| here | there |
| --- | --- |
| `wrap` | two's-complement `i64` wrap-around for `+ - *` and `%` (`wrapping_add/sub/mul/rem`); `Int / Int` does not wrap: it is `checked_div`, so a zero divisor and `i64::MIN / -1` are errors (`value.rs:1163-1169`, `minDivNegOne`, `divOverflow`, #3603), and the lane's `wrapping_div` (`vector_expr.rs:910`) runs only when no live lane is `MIN / -1` (guard `:898-901`), so it never wraps either |
| `AOp`, `valueArith` | `impl Add/Sub/Mul/Div/Rem for Value` (`value.rs:904-1170`), numeric + null arms; every other shape is an abstract `other` (the generic lane calls it too) |
| `intLane`, `floatLane`, `floatOperand` | `int_lane`, `float_lane`, `float_operand` (`vector_expr.rs:962-994`) |
| `arithmetic` | `arithmetic` (`vector_expr.rs:880-959`) |
-/

namespace Columnar
open FalkorCore (V)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

def wrap (x : Int) : Int := (x + 2 ^ 63) % 2 ^ 64 - 2 ^ 63

inductive AOp where
  | add | sub | mul | div | rem
  deriving DecidableEq

def intOp : AOp → Int → Int → Int
  | .add, a, b => wrap (a + b)
  | .sub, a, b => wrap (a - b)
  | .mul, a, b => wrap (a * b)
  | .div, a, b => wrap (Int.tdiv a b)
  | .rem, a, b => wrap (Int.tmod a b)

def floatOp : AOp → F → F → F
  | .add => FloatModel.add | .sub => FloatModel.sub | .mul => FloatModel.mul
  | .div => FloatModel.div | .rem => FloatModel.rem

/-- The error `Value / Value` raises for `i64::MIN / -1` (`value.rs:1163-1167`, `checked_div`, #3603). -/
def divOverflow : String := "ArgumentError: integer overflow in division"

/-- The one `Int / Int` quotient that is not an `i64`. -/
def minDivNegOne (a b : Int) : Bool := a == -2 ^ 63 && b == -1

/-- `Value` arithmetic: nulls propagate, `Int + - *` wrap, `Int / Int` is `checked_div` (zero divisor
and `i64::MIN / -1` error, #3603), `Int % Int` is `wrapping_rem` (zero divisor errors,
`i64::MIN % -1 = 0`), mixed numerics promote the int with `as f64`. -/
def valueArith (other : AOp → V F → V F → Except String (V F)) (op : AOp) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b =>
    if (op = .div ∨ op = .rem) ∧ b = 0 then .error "Division by zero"
    else if op = .div ∧ minDivNegOne a b = true then .error divOverflow
    else .ok (.int (intOp op a b))
  | .float a, .float b => .ok (.float (floatOp op a b))
  | .float a, .int b => .ok (.float (floatOp op a (FloatModel.ofInt b)))
  | .int a, .float b => .ok (.float (floatOp op (FloatModel.ofInt a) b))
  | a, b => other op a b

def intLane (c : EC F) (len : Nat) : Option (List Int) :=
  match c with
  | .ints d _ => some d
  | .scalar (.int v) => some (List.replicate len v)
  | _ => none

def floatLane (c : EC F) (len : Nat) : Option (List F) :=
  match c with
  | .floats d _ => some d
  | .ints d _ => some (d.map FloatModel.ofInt)
  | .scalar (.float v) => some (List.replicate len v)
  | .scalar (.int v) => some (List.replicate len (FloatModel.ofInt v))
  | _ => none

def floatOperand (c : EC F) : Bool :=
  match c with
  | .floats .. => true
  | .scalar (.float _) => true
  | _ => false

/-- The int lane (`vector_expr.rs:887-923`). `/` and `%` take it only when no divisor is zero, and `/`
also only when no non-null row is `i64::MIN / -1` (#3603: that row is left to the generic lane,
which raises the overflow error). -/
def intStage (op : AOp) (lhs rhs : EC F) (len : Nat) : Option (EC F) :=
  let nulls := unionNulls lhs rhs len
  match intLane lhs len, intLane rhs len with
  | some a, some b =>
    if op = .div ∨ op = .rem then
      if (List.range len).all (fun i => b[i]?.getD 0 != 0 || nulls.isNull i) &&
          (op != .div || (List.range len).all (fun i =>
            !minDivNegOne (a[i]?.getD 0) (b[i]?.getD 0) || nulls.isNull i)) then
        some (.ints ((List.range len).map (fun i =>
          if nulls.isNull i then 0 else intOp op (a[i]?.getD 0) (b[i]?.getD 0))) nulls)
      else none
    else some (.ints (List.zipWith (intOp op) a b) nulls)
  | _, _ => none

/-- The float lane (`vector_expr.rs:925-942`). -/
def floatStage (op : AOp) (lhs rhs : EC F) (len : Nat) : Option (EC F) :=
  if floatOperand lhs || floatOperand rhs then
    match floatLane lhs len, floatLane rhs len with
    | some a, some b => some (.floats (List.zipWith (floatOp op) a b) (unionNulls lhs rhs len))
    | _, _ => none
  else none

/-- The generic lane (`vector_expr.rs:944-958`). -/
def genericStage (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs : EC F) (len : Nat) :
    Except String (EC F) :=
  ((List.range len).mapM (fun i => valueArith other op (lhs.get i) (rhs.get i))).map EC.values

/-- `arithmetic` (`vector_expr.rs:880-959`). -/
def arithmetic (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs : EC F) (len : Nat) :
    Except String (EC F) :=
  match intStage op lhs rhs len with
  | some c => .ok c
  | none =>
    match floatStage op lhs rhs len with
    | some c => .ok c
    | none => genericStage other op lhs rhs len

/-- What each row is, as seen by the lanes. -/
theorem intLane_spec {c : EC F} {len : Nat} {a : List Int} (hw : c.WF len) (h : intLane c len = some a)
    (i : Nat) (hi : i < len) :
    ∃ x, a[i]? = some x ∧ c.get i = (if c.isNull i then .null else .int x) ∧ c.isValues = false := by
  cases c with
  | ints d n =>
    simp only [intLane, Option.some.injEq] at h
    subst h
    simp only [EC.WF] at hw
    have hi' : i < d.length := by omega
    exact ⟨d[i], List.getElem?_eq_getElem hi', by cases hn : Bits.isNull n i <;> simp [EC.get, EC.isNull, hn, List.getElem?_eq_getElem hi'], rfl⟩
  | scalar v =>
    cases v <;> simp only [intLane, reduceCtorEq] at h
    rename_i x
    cases h
    exact ⟨x, by simp [hi], by simp [EC.get, EC.isNull, V.isNull], rfl⟩
  | _ => simp [intLane] at h

/-- A float-lane entry is the `as f64` image of the row's numeric value. -/
def NumRel (v : V F) (f : F) : Prop := v = .float f ∨ ∃ x, v = .int x ∧ f = FloatModel.ofInt x

theorem floatLane_spec {c : EC F} {len : Nat} {a : List F} (hw : c.WF len) (h : floatLane c len = some a)
    (i : Nat) (hi : i < len) :
    ∃ f, a[i]? = some f ∧ (c.isNull i = true → c.get i = .null) ∧ (c.isNull i = false → NumRel (c.get i) f) ∧
      c.isValues = false ∧ (floatOperand c = true → c.isNull i = false → c.get i = .float f) := by
  cases c with
  | ints d n =>
    simp only [floatLane, Option.some.injEq] at h
    subst h
    simp only [EC.WF] at hw
    have hi' : i < d.length := by omega
    refine ⟨FloatModel.ofInt d[i], by simp [List.getElem?_eq_getElem hi'], ?_, ?_, rfl, by simp [floatOperand]⟩
    · intro hn; simp only [EC.isNull] at hn; simp [EC.get, hn]
    · intro hn; simp only [EC.isNull] at hn
      simp [EC.get, hn, List.getElem?_eq_getElem hi', NumRel]
  | floats d n =>
    simp only [floatLane, Option.some.injEq] at h
    subst h
    simp only [EC.WF] at hw
    have hi' : i < d.length := by omega
    refine ⟨_, List.getElem?_eq_getElem hi', ?_, ?_, rfl, ?_⟩
    · intro hn; simp only [EC.isNull] at hn; simp [EC.get, hn]
    · intro hn; simp only [EC.isNull] at hn
      simp [EC.get, hn, List.getElem?_eq_getElem hi', NumRel]
    · intro _ hn; simp only [EC.isNull] at hn; simp [EC.get, hn, List.getElem?_eq_getElem hi']
  | scalar v =>
    cases v <;> simp only [floatLane, reduceCtorEq] at h
    · rename_i x
      cases h
      exact ⟨_, by simp [hi], by simp [EC.isNull, V.isNull], fun _ => Or.inr ⟨x, rfl, rfl⟩, rfl,
        by simp [floatOperand]⟩
    · rename_i x
      cases h
      exact ⟨_, by simp [hi], by simp [EC.isNull, V.isNull], fun _ => Or.inl rfl, rfl, fun _ _ => rfl⟩
  | _ => simp [floatLane] at h

theorem valueArith_null_l (other : AOp → V F → V F → Except String (V F)) (op : AOp) (b : V F) :
    valueArith other op .null b = .ok .null := by cases b <;> rfl

theorem valueArith_null_r (other : AOp → V F → V F → Except String (V F)) (op : AOp) (a : V F) :
    valueArith other op a .null = .ok .null := by cases a <;> rfl

theorem valueArith_float (other : AOp → V F → V F → Except String (V F)) (op : AOp) {v w : V F} {f g : F}
    (hv : NumRel v f) (hw : NumRel w g) (hfl : v = .float f ∨ w = .float g) :
    valueArith other op v w = .ok (.float (floatOp op f g)) := by
  rcases hv with rfl | ⟨x, rfl, rfl⟩ <;> rcases hw with rfl | ⟨y, rfl, rfl⟩ <;>
    first | rfl | (rcases hfl with h | h <;> cases h)

theorem mapM_ok {α β : Type} {f : α → Except String β} {g : α → β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = .ok (g a)) → l.mapM f = .ok (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (by simp), mapM_ok (fun x hx => h x (by simp [hx]))]
    rfl

theorem mapM_length {α β : Type} {f : α → Except String β} :
    ∀ {l : List α} {vs : List β}, l.mapM f = .ok vs → vs.length = l.length
  | [], vs, h => by simp [List.mapM_nil] at h; cases h; rfl
  | a :: l, vs, h => by
    rw [List.mapM_cons] at h
    cases hf : f a with
    | error e => rw [hf] at h; cases h
    | ok b =>
      rw [hf] at h
      cases hm : l.mapM f with
      | error e => rw [hm] at h; cases h
      | ok bs =>
        rw [hm] at h
        cases h
        simp [mapM_length hm]

theorem range_map_values_get (vs : List (V F)) :
    (List.range vs.length).map (EC.values vs).get = vs := by
  apply List.ext_getElem?
  intro k
  rcases Nat.lt_or_ge k vs.length with h | h
  · simp [List.getElem?_range h, EC.get, List.getElem?_eq_getElem h]
  · rw [List.getElem?_eq_none (by simpa using h), List.getElem?_eq_none h]

theorem zipWith_getElem? {α β γ : Type} (f : α → β → γ) (a : List α) (b : List β) (i : Nat)
    {x : α} {y : β} (ha : a[i]? = some x) (hb : b[i]? = some y) : (List.zipWith f a b)[i]? = some (f x y) := by
  rw [List.getElem?_zipWith, ha, hb]

theorem ints_get' (d : List Int) (n : Bits) (i : Nat) :
    (EC.ints d n : EC F).get i = if n.isNull i then V.null else V.int (d[i]?.getD 0) := rfl

theorem floats_get' (d : List F) (n : Bits) (i : Nat) :
    (EC.floats d n).get i = if n.isNull i then V.null else (d[i]?.map V.float).getD V.null := rfl

theorem genericStage_agree (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs : EC F)
    (len : Nat) :
    (genericStage other op lhs rhs len).map (fun c => (List.range len).map c.get) =
      (List.range len).mapM (fun i => valueArith other op (lhs.get i) (rhs.get i)) := by
  unfold genericStage
  cases hm : (List.range len).mapM (fun i => valueArith other op (lhs.get i) (rhs.get i)) with
  | error e => rfl
  | ok vs =>
    have hl := mapM_length hm
    simp only [List.length_range] at hl
    show Except.ok ((List.range len).map (EC.values vs).get) = Except.ok vs
    rw [← hl, range_map_values_get]

/-- Typed stages: each row of the result is the `Value` operation. -/
theorem ok_agree (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs c : EC F) (len : Nat)
    (h : ∀ i, i < len → valueArith other op (lhs.get i) (rhs.get i) = .ok (c.get i)) :
    (Except.ok c : Except String (EC F)).map (fun c => (List.range len).map c.get) =
      (List.range len).mapM (fun i => valueArith other op (lhs.get i) (rhs.get i)) := by
  rw [mapM_ok (g := c.get) (fun i hi => h i (by simpa using hi))]
  rfl

theorem intStage_agree (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs c : EC F)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) (hs : intStage op lhs rhs len = some c) :
    ∀ i, i < len → valueArith other op (lhs.get i) (rhs.get i) = .ok (c.get i) := by
  intro i hi
  unfold intStage at hs
  cases hil : intLane lhs len with
  | none => rw [hil] at hs; simp at hs
  | some a =>
    cases hir : intLane rhs len with
    | none => rw [hil, hir] at hs; simp at hs
    | some b =>
      rw [hil, hir] at hs
      simp only at hs
      obtain ⟨x, hx, gx, vx⟩ := intLane_spec hl hil i hi
      obtain ⟨y, hy, gy, vy⟩ := intLane_spec hr hir i hi
      have hn : (unionNulls lhs rhs len).isNull i = (lhs.isNull i || rhs.isNull i) :=
        unionNulls_exact lhs rhs len hl hr i hi (by simp [vy]) (by simp [vx])
      by_cases hdr : op = .div ∨ op = .rem
      · rw [if_pos hdr] at hs
        split at hs
        · rename_i hall2
          simp only [Bool.and_eq_true] at hall2
          obtain ⟨hall, hdv⟩ := hall2
          simp only [Option.some.injEq] at hs
          subst hs
          have hb := List.all_eq_true.mp hall i (by simp [hi])
          rw [ints_get', hn, gx, gy]
          simp only [List.getElem?_map, List.getElem?_range hi, Option.map_some, Option.getD_some, hn]
          by_cases h1 : lhs.isNull i
          · simp [h1, valueArith_null_l]
          · by_cases h2 : rhs.isNull i
            · simp [h1, h2, valueArith_null_r]
            · rw [hn, hy] at hb
              simp [h1, h2] at hb
              have hm : op = .div → minDivNegOne x y = false := by
                intro hop; subst hop
                simp only [bne_self_eq_false, Bool.false_or] at hdv
                have := List.all_eq_true.mp hdv i (by simp [hi])
                simpa [hx, hy, hn, h1, h2] using this
              by_cases hdv' : op = .div
              · simp [h1, h2, valueArith, hb, hx, hy, hm hdv', hdv']
              · simp [h1, h2, valueArith, hb, hx, hy, hdv']
        · simp at hs
      · rw [if_neg hdr] at hs
        simp only [Option.some.injEq] at hs
        subst hs
        rw [ints_get', hn, gx, gy, zipWith_getElem? _ _ _ i hx hy]
        by_cases h1 : lhs.isNull i
        · simp [h1, valueArith_null_l]
        · by_cases h2 : rhs.isNull i
          · simp [h1, h2, valueArith_null_r]
          · have hnd : op ≠ .div := fun h => hdr (Or.inl h)
            have hnr : op ≠ .rem := fun h => hdr (Or.inr h)
            simp [h1, h2, valueArith, hnd, hnr]

theorem floatStage_agree (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs c : EC F)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) (hs : floatStage op lhs rhs len = some c) :
    ∀ i, i < len → valueArith other op (lhs.get i) (rhs.get i) = .ok (c.get i) := by
  intro i hi
  unfold floatStage at hs
  split at hs
  · rename_i hfo
    cases hil : floatLane lhs len with
    | none => rw [hil] at hs; simp at hs
    | some a =>
      cases hir : floatLane rhs len with
      | none => rw [hil, hir] at hs; simp at hs
      | some b =>
        rw [hil, hir] at hs
        simp only [Option.some.injEq] at hs
        subst hs
        obtain ⟨f, hf, nf, rf, vf, of⟩ := floatLane_spec hl hil i hi
        obtain ⟨g, hg, ng, rg, vg, og⟩ := floatLane_spec hr hir i hi
        have hn : (unionNulls lhs rhs len).isNull i = (lhs.isNull i || rhs.isNull i) :=
          unionNulls_exact lhs rhs len hl hr i hi (by simp [vg]) (by simp [vf])
        rw [floats_get', hn, zipWith_getElem? _ _ _ i hf hg]
        cases h1 : lhs.isNull i
        · cases h2 : rhs.isNull i
          · simp only [Bool.or_false, Bool.false_eq_true, ↓reduceIte, Option.map_some, Option.getD_some]
            apply valueArith_float other op (rf h1) (rg h2)
            simp only [Bool.or_eq_true] at hfo
            rcases hfo with h | h
            · exact Or.inl (of h h1)
            · exact Or.inr (og h h2)
          · simp [ng h2, valueArith_null_r]
        · simp [nf h1, valueArith_null_l]
  · simp at hs

/-- **The arithmetic lanes agree with `Value` arithmetic, row by row** — same
values and, on the generic lane, the same first error; the typed lanes are
entered only where no row can error (a zero divisor sends `/`,`%` to the
generic lane, `Int ⊕ Int` never takes the float lane). -/
theorem arithmetic_agree (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs : EC F)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) :
    (arithmetic other op lhs rhs len).map (fun c => (List.range len).map c.get) =
      (List.range len).mapM (fun i => valueArith other op (lhs.get i) (rhs.get i)) := by
  unfold arithmetic
  cases hi : intStage op lhs rhs len with
  | some c => exact ok_agree other op lhs rhs c len (intStage_agree other op lhs rhs c len hl hr hi)
  | none =>
    cases hf : floatStage op lhs rhs len with
    | some c => exact ok_agree other op lhs rhs c len (floatStage_agree other op lhs rhs c len hl hr hf)
    | none => exact genericStage_agree other op lhs rhs len

/-! ### #3603: `i64::MIN / -1` is an error, `i64::MIN % -1` stays 0 -/

/-- **PROVEN** (`value.rs:1163-1167`): `Int / Int` is `checked_div`, so the one unrepresentable
quotient raises the overflow error instead of wrapping (formerly `wrap (tdiv MIN -1) = MIN`). -/
theorem div_min_neg_one (other : AOp → V F → V F → Except String (V F)) :
    valueArith other .div (.int (-2 ^ 63)) (.int (-1)) = .error divOverflow := by
  simp [valueArith, minDivNegOne]

/-- `i64::MIN % -1` stays `0` (`wrapping_rem`, `value.rs:1195`). -/
theorem rem_min_neg_one (other : AOp → V F → V F → Except String (V F)) :
    valueArith other .rem (.int (-2 ^ 63)) (.int (-1)) = .ok (.int 0) := by
  simp [valueArith, minDivNegOne, intOp, wrap]

/-- **PROVEN**: the int lane refuses a `/` batch holding `i64::MIN / -1` (the generic lane then
raises the error), but keeps `%` on it. -/
theorem intStage_min_div_neg_one :
    intStage (F := F) .div (.scalar (.int (-2 ^ 63))) (.scalar (.int (-1))) 1 = none := by
  simp [intStage, intLane, minDivNegOne]
  rfl

/-- **PROVEN**: end to end, the columnar `/` of `i64::MIN` by `-1` errors exactly as the scalar one. -/
theorem arithmetic_min_div_neg_one (other : AOp → V F → V F → Except String (V F)) :
    arithmetic other .div (.scalar (.int (-2 ^ 63))) (.scalar (.int (-1))) 1 = .error divOverflow := by
  simp [arithmetic, intStage_min_div_neg_one, floatStage, floatOperand, genericStage, valueArith,
    minDivNegOne, EC.get, divOverflow, List.range, List.range.loop]
  rfl

end Columnar
