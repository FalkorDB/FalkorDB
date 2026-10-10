import Columnar.Column

/-
# Columnar expression evaluation (`runtime/vector_expr.rs`, `runtime/vectorized.rs`)

Every lane is compared against the per-row definition: the `Value` operator
(`compare_value`-based comparison, `Value` `+ - * / %`, 3-valued `NOT`, lazy
`CASE`) applied row by row.

| here | there |
| --- | --- |
| `CmpOp`, `CmpOp.flip` | `CmpOp`, `CmpOp::flip` (`vectorized.rs:58-94`) |
| `CmpOp.ofIR` | `CmpOp::from_expr_ir` (`vectorized.rs:70-80`) |
| `Flag`, `compareFloats`, `compareValue` | `DisjointOrNull`, `compare_floats` (`value.rs:1801`), `compare_value` numeric/null arms (`value.rs:1225-1270`); other shapes abstract (`other`) |
| `compareValues` | `compare_values` (`vectorized.rs:239-271`) |
| `icmp`, `fcmp` | the primitive `==,!=,<,<=,>,>=` on `i64` / `f64` |
| `kernelI`, `kernelF` | `compare_i64_column` / `compare_f64_column` (`vectorized.rs:105-210`), `compare_i64_pairs` / `compare_f64_pairs` + `mask_out_nulls` (`vector_expr.rs:795-851`) |
| `kernelV` | `compare_value_column_3vl` / `compare_value_column` (`vectorized.rs:229-293`) |
| `EC` | `ExprColumn` (`vector_expr.rs:66-75`); `EC.get` = `ExprColumn::get` (`:80-109`) |
| `EC.isNull`, `EC.constNull`, `EC.nullsOrNone`, `unionNulls` | `is_null`, `constant_nullness`, `nulls_or_none`, `union_nulls` (`vector_expr.rs:127-189`) |
| `EC.intoValues` | `ExprColumn::into_values` (`vector_expr.rs:114-123`) |
| `compareColumns` | `compare_columns` (`vector_expr.rs:731-792`) |
| `wrap`, `valueArith` | `i64::wrapping_*` and `impl Add/Sub/Mul/Div/Rem for Value` (`value.rs:904-1170`) |
| `intLane`, `floatLane`, `floatOperand`, `arithmetic` | `int_lane`, `float_lane`, `float_operand`, `arithmetic` (`vector_expr.rs:878-985`) |
| `negate`, `notRow` | `negate_bools` (`vector_expr.rs:854-873`) / per-row `ExprIR::Not` (`eval.rs:699-708`) |
| `caseCol`, `caseRow` | `VectorEval::eval_case` (`vector_expr.rs:475-549`) / `ExprEval::eval_case` (`eval.rs:485-518`) |
| `foldCall` | `eval_function` scalar folding (`vector_expr.rs:601-619`) |
| `evalVariable` | `VectorEval::eval_variable` (`vector_expr.rs:310-329`) |
| `filterPass` | `FilterOp` mask read (`filter.rs:69-86`) |
-/

namespace Columnar

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

inductive CmpOp where
  | eq | neq | lt | le | gt | ge
  deriving DecidableEq

def CmpOp.flip : CmpOp → CmpOp
  | .eq => .eq | .neq => .neq | .lt => .gt | .le => .ge | .gt => .lt | .ge => .le

inductive Flag where
  | none | nan | disjoint | comparedNull
  deriving DecidableEq

def compareFloats (a b : F) : Ordering × Flag :=
  match FloatModel.pcmp a b with
  | some o => (o, .none)
  | none => (.lt, .nan)

/-- `compare_value`: faithful on `Int`/`Float`/`Null`; `other` stands for every
remaining arm (strings, lists, maps, points, disjoint types, …). -/
def compareValue (other : V F → V F → Ordering × Flag) : V F → V F → Ordering × Flag
  | .int a, .int b => (compare a b, .none)
  | .float a, .float b => compareFloats a b
  | .int i, .float f => compareFloats (FloatModel.ofInt i) f
  | .float f, .int i => compareFloats f (FloatModel.ofInt i)
  | .null, _ => (.lt, .comparedNull)
  | _, .null => (.gt, .comparedNull)
  | a, b => other a b

/-- `compare_values` (`vectorized.rs:239-271`). -/
def compareValues (other : V F → V F → Ordering × Flag) (a b : V F) (op : CmpOp) : Option Bool :=
  let (ord, flag) := compareValue other a b
  match op with
  | .eq => match flag with
    | .comparedNull => none
    | .nan | .disjoint => some false
    | .none => some (ord == .eq)
  | .neq => match flag with
    | .comparedNull => none
    | _ => some (ord != .eq)
  | _ => match flag with
    | .comparedNull | .disjoint => none
    | .nan => some false
    | .none => some (match op with
      | .lt => ord == .lt
      | .le => ord != .gt
      | .gt => ord == .gt
      | _ => ord != .lt)

/-- The per-row comparison result as a `Value` (the scalar evaluator's answer). -/
def rowCmp (other : V F → V F → Ordering × Flag) (a b : V F) (op : CmpOp) : V F :=
  match compareValues other a b op with
  | some b => .bool b
  | none => .null

/-- `i64` comparison operators. -/
def icmp (op : CmpOp) (a b : Int) : Bool :=
  match op with
  | .eq => a == b | .neq => a != b | .lt => decide (a < b) | .le => decide (a ≤ b)
  | .gt => decide (a > b) | .ge => decide (a ≥ b)

/-- `f64` comparison operators, from `partial_cmp` (`PartialOrd for f64`). -/
def fcmp (op : CmpOp) (a b : F) : Bool :=
  match op, FloatModel.pcmp a b with
  | .eq, some .eq => true
  | .eq, _ => false
  | .neq, some .eq => false
  | .neq, _ => true
  | .lt, some .lt => true
  | .le, some .lt => true
  | .le, some .eq => true
  | .gt, some .gt => true
  | .ge, some .gt => true
  | .ge, some .eq => true
  | _, _ => false

/-! ## Scalar kernels = `compare_values` on the corresponding values -/

theorem icmp_agree (other : V F → V F → Ordering × Flag) (op : CmpOp) (a b : Int) :
    compareValues other (.int a) (.int b) op = some (icmp op a b) := by
  have hc : compare a b = (if a < b then Ordering.lt else if a = b then Ordering.eq else Ordering.gt) := by
    simp [compare, compareOfLessAndEq]
  rcases Int.lt_trichotomy a b with h | h | h
  · have e : compare a b = .lt := by rw [hc, if_pos h]
    have h1 : a ≠ b := Int.ne_of_lt h
    have h2 : a ≤ b := Int.le_of_lt h
    have h3 : ¬ b ≤ a := Int.not_le.mpr h
    have h5 : (a != b) = true := bne_iff_ne.mpr h1
    cases op <;> simp [compareValues, compareValue, icmp, e, h, h1, h2, h3, h5] <;> rfl
  · subst h
    have e : compare a a = .eq := by rw [hc, if_neg (Int.lt_irrefl a), if_pos rfl]
    cases op <;> simp [compareValues, compareValue, icmp, e]
  · have e : compare a b = .gt := by rw [hc, if_neg (by omega), if_neg (by omega)]
    have h1 : a ≠ b := (Int.ne_of_lt h).symm
    have h2 : b ≤ a := Int.le_of_lt h
    have h3 : ¬ a ≤ b := Int.not_le.mpr h
    have h4 : ¬ a < b := Int.not_lt.mpr h2
    have h5 : (a != b) = true := bne_iff_ne.mpr h1
    cases op <;> simp [compareValues, compareValue, icmp, e, h, h1, h2, h3, h4, h5] <;> rfl

theorem fcmp_agree_ff (other : V F → V F → Ordering × Flag) (op : CmpOp) (a b : F) :
    compareValues other (.float a) (.float b) op = some (fcmp op a b) := by
  simp only [compareValues, compareValue, compareFloats, fcmp]
  cases h : FloatModel.pcmp a b with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

theorem fcmp_agree_if (other : V F → V F → Ordering × Flag) (op : CmpOp) (i : Int) (b : F) :
    compareValues other (.int i) (.float b) op = some (fcmp op (FloatModel.ofInt i) b) := by
  simp only [compareValues, compareValue, compareFloats, fcmp]
  cases h : FloatModel.pcmp (FloatModel.ofInt i) b with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

theorem fcmp_agree_fi (other : V F → V F → Ordering × Flag) (op : CmpOp) (a : F) (i : Int) :
    compareValues other (.float a) (.int i) op = some (fcmp op a (FloatModel.ofInt i)) := by
  simp only [compareValues, compareValue, compareFloats, fcmp]
  cases h : FloatModel.pcmp a (FloatModel.ofInt i) with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

/-- `CmpOp::flip` is sound on the int lane: `t op d` = `d op.flip t`. -/
theorem icmp_flip (op : CmpOp) (a b : Int) : icmp op.flip b a = icmp op a b := by
  cases op <;> simp only [icmp, CmpOp.flip] <;> apply Bool.eq_iff_iff.mpr <;> simp <;>
    constructor <;> intro <;> omega

/-- … and on the float lane (by IEEE antisymmetry of `partial_cmp`). -/
theorem fcmp_flip (op : CmpOp) (a b : F) : fcmp op.flip b a = fcmp op a b := by
  have hs := FloatModel.pcmp_swap a b
  simp only [fcmp, CmpOp.flip]
  rw [hs]
  cases FloatModel.pcmp a b with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

/-- NaN: every ordering is false and `<>` is true (both lanes). -/
theorem fcmp_nan (op : CmpOp) (a b : F) (h : FloatModel.pcmp a b = none) :
    fcmp op a b = (op == .neq) := by
  cases op <;> simp [fcmp, h] <;> rfl

/-! ## Column kernels -/

/-- `compare_*_column` / `compare_*_pairs`: elementwise, then `false` under a null bit. -/
def maskOut (res : List Bool) (nulls : Bits) : List Bool :=
  res.mapIdx (fun i r => if nulls.isNull i then false else r)

def kernelI (op : CmpOp) (a b : List Int) (nulls : Bits) : List Bool :=
  maskOut (List.zipWith (icmp op) a b) nulls

def kernelF (op : CmpOp) (a b : List F) (nulls : Bits) : List Bool :=
  maskOut (List.zipWith (fcmp op) a b) nulls

/-- `compare_value_column_3vl` against a constant: `(mask, nulls)`. -/
def kernelV (other : V F → V F → Ordering × Flag) (op : CmpOp) (data : List (V F)) (c : V F) :
    List Bool × Bits :=
  (data.map (fun v => (compareValues other v c op).getD false),
   data.map (fun v => (compareValues other v c op).isNone))

/-- `ExprColumn`. -/
inductive EC (F : Type) where
  | scalar (v : V F)
  | ints (d : List Int) (n : Bits)
  | floats (d : List F) (n : Bits)
  | bools (d : List Bool) (n : Bits)
  | values (d : List (V F))

namespace EC

/-- `ExprColumn::get` (in-range reads; defaults never observed under `WF`). -/
def get : EC F → Nat → V F
  | .scalar v, _ => v
  | .ints d n, i => if n.isNull i then .null else .int (d[i]?.getD 0)
  | .floats d n, i => if n.isNull i then .null else (d[i]?.map V.float).getD .null
  | .bools d n, i => if n.isNull i then .null else .bool (d[i]?.getD false)
  | .values d, i => d[i]?.getD .null

def isNull : EC F → Nat → Bool
  | .scalar v, _ => v.isNull
  | .ints _ n, i | .floats _ n, i | .bools _ n, i => n.isNull i
  | .values d, i => (d[i]?.getD .null).isNull

def constNull : EC F → Option Bool
  | .scalar v => some v.isNull
  | .ints _ n | .floats _ n | .bools _ n => if n.any id then none else some false
  | .values _ => none

def nullsOrNone : EC F → Nat → Bits
  | .ints _ n, _ | .floats _ n, _ | .bools _ n, _ => n
  | _, len => Bits.none len

def isValues : EC F → Bool
  | .values _ => true
  | _ => false

/-- Every lane has `len` entries. -/
def WF (len : Nat) : EC F → Prop
  | .scalar _ => True
  | .ints d n => d.length = len ∧ n.length = len
  | .floats d n => d.length = len ∧ n.length = len
  | .bools d n => d.length = len ∧ n.length = len
  | .values d => d.length = len

def intoValues (c : EC F) (len : Nat) : List (V F) :=
  match c with
  | .values d => d
  | .scalar v => List.replicate len v
  | other => (List.range len).map other.get

theorem isNull_eq (c : EC F) (i : Nat) (len : Nat) (hw : c.WF len) (hi : i < len) :
    c.isNull i = (c.get i).isNull := by
  cases c with
  | scalar v => rfl
  | values d => rfl
  | ints d n =>
    simp only [isNull, get]; split <;> simp_all [V.isNull]
  | floats d n =>
    simp only [isNull, get, WF] at hw ⊢
    split
    · simp_all [V.isNull]
    · rename_i h
      rw [List.getElem?_eq_getElem (by omega)]; simp_all [V.isNull]
  | bools d n =>
    simp only [isNull, get]; split <;> simp_all [V.isNull]

theorem intoValues_get (c : EC F) (len : Nat) (hw : c.WF len) (i : Nat) (hi : i < len) :
    (c.intoValues len)[i]? = some (c.get i) := by
  cases c with
  | values d =>
    simp only [intoValues, get, WF] at hw ⊢
    rw [List.getElem?_eq_getElem (by omega)]; rfl
  | scalar v => simp [intoValues, get, hi]
  | ints d n => simp [intoValues, hi]
  | floats d n => simp [intoValues, hi]
  | bools d n => simp [intoValues, hi]

theorem intoValues_length (c : EC F) (len : Nat) (hw : c.WF len) : (c.intoValues len).length = len := by
  cases c <;> simp_all [intoValues, WF]

end EC

theorem Bits.isNull_none (n i : Nat) : (Bits.none n).isNull i = false := by
  simp [Bits.none, Bits.isNull, List.getElem?_replicate]; split <;> rfl

theorem Bits.isNull_all (n i : Nat) (h : i < n) : (Bits.all n).isNull i = true := by
  simp [Bits.all, Bits.isNull, List.getElem?_replicate, h]

theorem Bits.isNull_ofFn (n : Nat) (p : Nat → Bool) (i : Nat) (h : i < n) :
    Bits.isNull ((List.range n).map p) i = p i := by
  simp [Bits.isNull, List.getElem?_range h]

/-- `union_nulls` (`vector_expr.rs:168-189`). -/
def unionNulls (l r : EC F) (len : Nat) : Bits :=
  match l.constNull, r.constNull with
  | some true, _ => Bits.all len
  | _, some true => Bits.all len
  | some false, some false => Bits.none len
  | some false, none => r.nullsOrNone len
  | none, some false => l.nullsOrNone len
  | none, none => (List.range len).map (fun i => l.isNull i || r.isNull i)

theorem constNull_spec (c : EC F) (len : Nat) (hw : c.WF len) (b : Bool) (h : c.constNull = some b) :
    ∀ i < len, c.isNull i = b := by
  intro i hi
  cases c with
  | scalar v => simp [EC.constNull] at h; simp [EC.isNull, h]
  | values d => simp [EC.constNull] at h
  | ints d n | floats d n | bools d n =>
    simp only [EC.constNull] at h
    split at h
    · simp at h
    · rename_i hany
      simp at h; subst h
      simp only [EC.isNull, Bits.isNull]
      cases hn : n[i]? with
      | none => rfl
      | some x =>
        simp only [Option.getD_some]
        cases x
        · rfl
        · exact absurd (List.any_eq_true.mpr ⟨true, List.mem_of_getElem? hn, rfl⟩) hany

theorem nullsOrNone_spec (c : EC F) (len : Nat) (hv : c.isValues = false) (hc : c.constNull = none)
    (i : Nat) : (c.nullsOrNone len).isNull i = c.isNull i := by
  cases c <;> simp [EC.isValues, EC.constNull] at hv hc <;> rfl

/-- **`union_nulls` is exact** except when one side is a `Values` column and the
other a non-null constant (it then answers "no nulls"); the typed lanes never
see a `Values` operand, and the generic lanes ignore the union. -/
theorem unionNulls_exact (l r : EC F) (len : Nat) (hl : l.WF len) (hr : r.WF len) (i : Nat) (hi : i < len)
    (h1 : ¬(r.isValues = true ∧ l.constNull = some false))
    (h2 : ¬(l.isValues = true ∧ r.constNull = some false)) :
    (unionNulls l r len).isNull i = (l.isNull i || r.isNull i) := by
  unfold unionNulls
  cases hlc : l.constNull with
  | some bl =>
    have el := constNull_spec l len hl bl hlc i hi
    cases bl with
    | true => simp [Bits.isNull_all len i hi, el]
    | false =>
      cases hrc : r.constNull with
      | some br =>
        have er := constNull_spec r len hr br hrc i hi
        cases br with
        | true => simp [Bits.isNull_all len i hi, er]
        | false => simp [Bits.isNull_none, el, er]
      | none =>
        simp only
        have hv : r.isValues = false := by
          cases hh : r.isValues
          · rfl
          · exact absurd ⟨hh, hlc⟩ h1
        rw [nullsOrNone_spec r len hv hrc i, el]; simp
  | none =>
    cases hrc : r.constNull with
    | some br =>
      have er := constNull_spec r len hr br hrc i hi
      cases br with
      | true => simp [Bits.isNull_all len i hi, er]
      | false =>
        simp only
        have hv : l.isValues = false := by
          cases hh : l.isValues
          · rfl
          · exact absurd ⟨hh, hrc⟩ h2
        rw [nullsOrNone_spec l len hv hlc i, er]; simp
    | none => simp only; exact Bits.isNull_ofFn len _ i hi

/-- The inexact corner, for the record: `1 + [null]` gets an empty union. -/
theorem unionNulls_values_inexact :
    (unionNulls (EC.scalar (V.int 1 : V F)) (EC.values [V.null]) 1).isNull 0 = false ∧
      (EC.values [(V.null : V F)]).isNull 0 = true := by
  simp [unionNulls, EC.constNull, EC.nullsOrNone, Bits.isNull_none, EC.isNull, V.isNull]

theorem unionNulls_length (l r : EC F) (len : Nat) (hl : l.WF len) (hr : r.WF len) :
    (unionNulls l r len).length = len := by
  unfold unionNulls
  cases hlc : l.constNull with
  | some bl =>
    cases bl <;> cases hrc : r.constNull <;> (try rename_i br; cases br) <;>
      simp [Bits.all, Bits.none] <;>
      (cases r <;> simp_all [EC.nullsOrNone, EC.WF, EC.constNull, Bits.none])
  | none =>
    cases hrc : r.constNull with
    | some br =>
      cases br <;> simp [Bits.all, Bits.none] <;>
      (cases l <;> simp_all [EC.nullsOrNone, EC.WF, EC.constNull, Bits.none])
    | none => simp

/-! ## `compare_columns` -/

/-- The typed lanes of `compare_columns` (`vector_expr.rs:738-767`). -/
def typedLane (lhs rhs : EC F) (op : CmpOp) (nulls : Bits) : Option (List Bool) :=
  match lhs, rhs with
  | .ints d _, .scalar (.int t) => some (kernelI op d (List.replicate d.length t) nulls)
  | .scalar (.int t), .ints d _ => some (kernelI op.flip d (List.replicate d.length t) nulls)
  | .floats d _, .scalar (.float t) => some (kernelF op d (List.replicate d.length t) nulls)
  | .scalar (.float t), .floats d _ => some (kernelF op.flip d (List.replicate d.length t) nulls)
  | .ints d _, .scalar (.float t) =>
    some (kernelF op (d.map FloatModel.ofInt) (List.replicate d.length t) nulls)
  | .floats d _, .scalar (.int t) =>
    some (kernelF op d (List.replicate d.length (FloatModel.ofInt t)) nulls)
  | .ints a _, .ints b _ => some (kernelI op a b nulls)
  | .floats a _, .floats b _ => some (kernelF op a b nulls)
  | _, _ => none

/-- The generic lanes (`vector_expr.rs:775-791`). -/
def genericLane (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp) (len : Nat) : EC F :=
  match rhs, lhs with
  | .scalar c, .values d => let r := kernelV other op d c; .bools r.1 r.2
  | _, _ =>
    .bools ((List.range len).map (fun i => (compareValues other (lhs.get i) (rhs.get i) op).getD false))
      ((List.range len).map (fun i => (compareValues other (lhs.get i) (rhs.get i) op).isNone))

/-- `compare_columns` (`vector_expr.rs:731-792`). -/
def compareColumns (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp) (len : Nat) : EC F :=
  match typedLane lhs rhs op (unionNulls lhs rhs len) with
  | some mask => .bools mask (unionNulls lhs rhs len)
  | none => genericLane other lhs rhs op len

theorem maskOut_get (res : List Bool) (nulls : Bits) (i : Nat) (hi : i < res.length) :
    (maskOut res nulls)[i]?.getD false = (if nulls.isNull i then false else res[i]) := by
  simp [maskOut, List.getElem?_mapIdx, List.getElem?_eq_getElem hi]

theorem maskOut_length (res : List Bool) (nulls : Bits) : (maskOut res nulls).length = res.length := by
  simp [maskOut]

/-- The typed-lane result read back: null where the union is null, else the kernel bit. -/
theorem bools_get (mask : List Bool) (nulls : Bits) (i : Nat) :
    (EC.bools mask nulls : EC F).get i = if nulls.isNull i then (V.null : V F) else .bool (mask[i]?.getD false) := rfl

/-- Null operands make every comparison null. -/
theorem rowCmp_null_l (other : V F → V F → Ordering × Flag) (b : V F) (op : CmpOp) :
    rowCmp other .null b op = .null := by
  cases op <;> rfl

theorem rowCmp_null_r (other : V F → V F → Ordering × Flag) (a : V F) (op : CmpOp) :
    rowCmp other a .null op = .null := by
  cases a <;> cases op <;> rfl

theorem kernelI_get (op : CmpOp) (a b : List Int) (nulls : Bits) (i : Nat) (ha : i < a.length)
    (hb : i < b.length) :
    (kernelI op a b nulls)[i]?.getD false = (if nulls.isNull i then false else icmp op a[i] b[i]) := by
  unfold kernelI
  rw [maskOut_get _ _ _ (by simp; omega)]
  simp

theorem kernelF_get (op : CmpOp) (a b : List F) (nulls : Bits) (i : Nat) (ha : i < a.length)
    (hb : i < b.length) :
    (kernelF op a b nulls)[i]?.getD false = (if nulls.isNull i then false else fcmp op a[i] b[i]) := by
  unfold kernelF
  rw [maskOut_get _ _ _ (by simp; omega)]
  simp

theorem ints_get (d : List Int) (n : Bits) (i : Nat) (hi : i < d.length) :
    (EC.ints d n : EC F).get i = if n.isNull i then .null else .int d[i] := by
  simp [EC.get, List.getElem?_eq_getElem hi]

theorem floats_get (d : List F) (n : Bits) (i : Nat) (hi : i < d.length) :
    (EC.floats d n).get i = if n.isNull i then .null else .float d[i] := by
  simp [EC.get, List.getElem?_eq_getElem hi]

theorem rowCmp_some (other : V F → V F → Ordering × Flag) {a b : V F} {op : CmpOp} {x : Bool}
    (h : compareValues other a b op = some x) : rowCmp other a b op = .bool x := by
  simp [rowCmp, h]

theorem generic_get (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp) (len i : Nat)
    (hi : i < len) :
    (EC.bools ((List.range len).map (fun i => (compareValues other (lhs.get i) (rhs.get i) op).getD false))
      ((List.range len).map (fun i => (compareValues other (lhs.get i) (rhs.get i) op).isNone))).get i =
      rowCmp other (lhs.get i) (rhs.get i) op := by
  rw [bools_get, Bits.isNull_ofFn len _ i hi]
  simp only [List.getElem?_map, List.getElem?_range hi, Option.map_some, Option.getD_some]
  unfold rowCmp
  cases compareValues other (lhs.get i) (rhs.get i) op <;> rfl

theorem kernelV_get (other : V F → V F → Ordering × Flag) (op : CmpOp) (d : List (V F)) (c : V F)
    (i : Nat) (hi : i < d.length) :
    (EC.bools (kernelV other op d c).1 (kernelV other op d c).2).get i = rowCmp other d[i] c op := by
  rw [bools_get]
  simp only [kernelV, Bits.isNull, List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some,
    Option.getD_some]
  unfold rowCmp
  cases compareValues other d[i] c op <;> rfl

theorem typedLane_agree (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) (i : Nat) (hi : i < len) {mask : List Bool}
    (ht : typedLane lhs rhs op (unionNulls lhs rhs len) = some mask) :
    (EC.bools mask (unionNulls lhs rhs len)).get i = rowCmp other (lhs.get i) (rhs.get i) op := by
  have hu := unionNulls_exact lhs rhs len hl hr i hi
  unfold typedLane at ht
  split at ht <;> simp only [Option.some.injEq, reduceCtorEq] at ht <;> subst ht
  · rename_i d n t
    simp only [EC.WF] at hl
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.or_false]
    rw [ints_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · rw [kernelI_get _ _ _ _ _ (by omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (icmp_agree other op _ _)]
  · rename_i t d n
    simp only [EC.WF] at hr
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.false_or]
    rw [ints_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_r]
    · rw [kernelI_get _ _ _ _ _ (by omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (icmp_agree other op _ _), icmp_flip]
  · rename_i d n t
    simp only [EC.WF] at hl
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.or_false]
    rw [floats_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · rw [kernelF_get _ _ _ _ _ (by omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (fcmp_agree_ff other op _ _)]
  · rename_i t d n
    simp only [EC.WF] at hr
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.false_or]
    rw [floats_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_r]
    · rw [kernelF_get _ _ _ _ _ (by omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (fcmp_agree_ff other op _ _), fcmp_flip]
  · rename_i d n t
    simp only [EC.WF] at hl
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.or_false]
    rw [ints_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · rw [kernelF_get _ _ _ _ _ (by simp; omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (fcmp_agree_if other op _ _)]
  · rename_i d n t
    simp only [EC.WF] at hl
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull, V.isNull, Bool.or_false]
    rw [floats_get d n i (by omega)]
    by_cases hn : n.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · rw [kernelF_get _ _ _ _ _ (by omega) (by simp; omega)]
      simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_some other (fcmp_agree_fi other op _ _)]
  · rename_i a na b nb
    simp only [EC.WF] at hl hr
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull]
    rw [ints_get a na i (by omega), ints_get b nb i (by omega)]
    by_cases hn : na.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · by_cases hm : nb.isNull i
      · simp [hu', EC.isNull, V.isNull, EC.get, hn, hm, rowCmp_null_r]
      · rw [kernelI_get _ _ _ _ _ (by omega) (by omega)]
        simp [hu', EC.isNull, V.isNull, EC.get, hn, hm, rowCmp_some other (icmp_agree other op _ _)]
  · rename_i a na b nb
    simp only [EC.WF] at hl hr
    have hu' := hu (by simp [EC.isValues]) (by simp [EC.isValues])
    rw [bools_get, hu']
    simp only [EC.isNull]
    rw [floats_get a na i (by omega), floats_get b nb i (by omega)]
    by_cases hn : na.isNull i
    · simp [hu', EC.isNull, V.isNull, EC.get, hn, rowCmp_null_l]
    · by_cases hm : nb.isNull i
      · simp [hu', EC.isNull, V.isNull, EC.get, hn, hm, rowCmp_null_r]
      · rw [kernelF_get _ _ _ _ _ (by omega) (by omega)]
        simp [hu', EC.isNull, V.isNull, EC.get, hn, hm, rowCmp_some other (fcmp_agree_ff other op _ _)]

/-- **`compare_columns` = the per-row comparison, at every row**, whichever lane
(typed int, typed float, promoted int→float, flipped constant, pairwise, or the
generic `compare_value` loop) is taken. -/
theorem compareColumns_agree (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) (i : Nat) (hi : i < len) :
    (compareColumns other lhs rhs op len).get i = rowCmp other (lhs.get i) (rhs.get i) op := by
  unfold compareColumns
  cases ht : typedLane lhs rhs op (unionNulls lhs rhs len) with
  | some mask => exact typedLane_agree other lhs rhs op len hl hr i hi ht
  | none =>
    simp only
    unfold genericLane
    split
    · rename_i c d
      simp only [EC.WF] at hl
      rw [kernelV_get other op d c i (by omega)]
      simp [EC.get, List.getElem?_eq_getElem (show i < d.length by omega)]
    · exact generic_get other lhs rhs op len i hi

end Columnar
