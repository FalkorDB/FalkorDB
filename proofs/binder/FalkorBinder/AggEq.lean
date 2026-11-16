/-
# ORDER BY aggregate matching: `replace_agg_subtrees` and `raw_exprs_structurally_equal`

Re-checked against PR #3254 (67a65dc6e), which replaced the `Constant` arms of
`expr_ir_eq` by `constants_equal` and compares the payloads of `ListComprehension`,
`Case`, `Quantifier`, `Reduce`, `ShortestPath` and `CompiledRegex`.

| here | there |
| --- | --- |
| `RV`, `RD`, `RE` | raw (pre-bind) `ExprIR<Arc<String>>` payloads and trees |
| `RF` | an `f64`: NaN, ±inf, or the finite dyadic `m * 2^e` |
| `disc` | `std::mem::discriminant` of the `ExprIR` variant |
| `isAggPlaceholder` | binder.rs:2665-2667 |
| `intFloatEq` | the `(Int, Float)` arm of `constants_equal` binder.rs:2650-2655 |
| `constEq` | `constants_equal` binder.rs:2643-2656 |
| `exprIrEq` | `expr_ir_eq` binder.rs:2699-2743 |
| `eqS` | `nodes_eq` binder.rs:2669-2697 |
| `structEq` | `raw_exprs_structurally_equal` binder.rs:2661-2746 |
| `replaceAgg` | `replace_agg_subtrees` binder.rs:2613-2638 |

Float–float equality is a parameter `feq` (f64 `==`, so NaN ≠ NaN), and equality of
constants of the other `Value` types (lists, maps, temporals …, which the parser never
produces in a raw tree) a parameter `oeq` (`Value::eq`).

* `nodesEq_refl` / `nodesEq_symm`: an equivalence when `feq`/`oeq` are.
* `exprIrEq_exact`: equal payloads compared by value are equal.
* `intFloatEq_iff`: an Int and a Float constant match iff the float is exactly that integer.
* `const_kinds_distinguished`, `order_by_other_aggregate_rejected`: the W4-plan-3 bug
  (`pre3254_const_kinds_conflated`, `pre3254_order_by_wrong_aggregate`) is fixed.
* `replaceAgg_id`: with no matching aggregate the ORDER BY tree is unchanged;
  `matchItem_spec`: every replacement is justified by `structEq`.
-/
namespace FalkorBinder.Agg

/-- An `f64`: NaN (any payload), ±inf, or the finite value `m * 2^e`. -/
inductive RF
  | nan (payload : Nat) | inf (neg : Bool) | fin (m : Int) (e : Int)
  deriving DecidableEq, Repr

/-- `Value`s that occur as constants in raw expressions. -/
inductive RV
  | null | bool (b : Bool) | int (i : Int) | float (f : RF) | str (s : String) | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RD
  | var (n : String)
  | func (name : String) (agg : Bool)
  | const (v : RV)
  | prop (k : String)
  | param (p : String)
  | quant (kind : Nat) (v : String)
  | listComp (v : String)
  | caseE (hasSubject : Bool)
  | reduce (acc iter : String)
  | shortest (types : List String) (minH : Nat) (maxH : Option Nat) (directed : Bool)
  | regex (kind : Nat) (src : String)
  /-- `Pattern` / `PatternComprehension` / `NestedPlan`: payload never compared. -/
  | pattern (tag : Nat) (payload : Nat)
  /-- payload-free variants (operators, `List`, `Map`, …). -/
  | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RE
  | node (d : RD) (cs : List RE)
  deriving Repr

def RE.d : RE → RD | .node d _ => d

/-- Variant discriminant: payloads are ignored. -/
def disc : RD → Nat
  | .var _ => 0 | .func _ _ => 1 | .const _ => 2 | .prop _ => 3 | .param _ => 4
  | .quant _ _ => 5 | .listComp _ => 6 | .caseE _ => 7 | .reduce _ _ => 8
  | .shortest _ _ _ _ => 9 | .regex _ _ => 10 | .pattern t _ => 11 + 2 * t | .other t => 12 + 2 * t

def placeholder : String := "__agg_order_by_placeholder__"

def isAggPlaceholder : RD → Bool
  | .var v => v == placeholder
  | _ => false

theorem isAggPlaceholder_iff (d : RD) : isAggPlaceholder d = true ↔ d = .var placeholder := by
  cases d <;> simp [isAggPlaceholder]

/-! ## `constants_equal` -/

def i64Lo : Int := -9223372036854775808
def i64Hi : Int := 9223372036854775808

/-- `f.fract() == 0.0` (false for NaN and ±inf, whose `fract` is NaN). -/
def fractZero : RF → Bool
  | .fin m e => decide (0 ≤ e) || m % (2 ^ (-e).toNat : Int) == 0
  | _ => false

/-- `f >= -2^63 && f < 2^63` (false for NaN; ±inf fails one side). -/
def inI64 : RF → Bool
  | .fin m e =>
    if 0 ≤ e then decide (i64Lo ≤ m * 2 ^ e.toNat) && decide (m * 2 ^ e.toNat < i64Hi)
    else decide (i64Lo * 2 ^ (-e).toNat ≤ m) && decide (m < i64Hi * 2 ^ (-e).toNat)
  | _ => false

/-- `f as i64` on an in-range float: truncation toward zero. -/
def asI64 : RF → Int
  | .fin m e => if 0 ≤ e then m * 2 ^ e.toNat else m.tdiv (2 ^ (-e).toNat)
  | .inf true => i64Lo | .inf false => i64Hi - 1 | .nan _ => 0

/-- The `(Int(i), Float(f))` arm of `constants_equal`, in Rust's `&&` order. -/
def intFloatEq (i : Int) (f : RF) : Bool :=
  fractZero f && inI64 f && asI64 f == i

/-- The float `m * 2^e` is exactly the integer `i` (cross-multiplied). -/
def denotes (m e i : Int) : Prop :=
  if 0 ≤ e then m * 2 ^ e.toNat = i else m = i * 2 ^ (-e).toNat

variable (feq : RF → RF → Bool) (oeq : Nat → Nat → Bool)

/-- `constants_equal`: mixed Int/Float exactly, otherwise `Value::eq`, which is
`false` for two values of different types (`Null` equals only `Null`). -/
def constEq : RV → RV → Bool
  | .int i, .float f => intFloatEq i f
  | .float f, .int i => intFloatEq i f
  | .null, .null => true
  | .bool a, .bool b => a == b
  | .int a, .int b => a == b
  | .float a, .float b => feq a b
  | .str a, .str b => a == b
  | .other a, .other b => oeq a b
  | _, _ => false

theorem intFloatEq_nan (i : Int) (p : Nat) : intFloatEq i (.nan p) = false := rfl
theorem intFloatEq_inf (i : Int) (n : Bool) : intFloatEq i (.inf n) = false := rfl

private theorem pow_pos' (k : Nat) : (0 : Int) < 2 ^ k := Int.pow_pos (by decide)

/-- An Int and a finite Float match exactly when the float is that integer
(for every `i64` `i`): no rounding of `i` to `f64`. -/
theorem intFloatEq_iff (i m e : Int) (hi : i64Lo ≤ i ∧ i < i64Hi) :
    intFloatEq i (.fin m e) = true ↔ denotes m e i := by
  unfold intFloatEq fractZero inI64 asI64 denotes
  by_cases he : 0 ≤ e
  · simp only [he, decide_true, Bool.true_or, ite_true, Bool.true_and, Bool.and_eq_true,
      decide_eq_true_eq, beq_iff_eq]
    constructor
    · intro h; exact h.2
    · intro h; subst h; exact ⟨hi, rfl⟩
  · simp only [he, decide_false, Bool.false_or, ite_false, Bool.and_eq_true, decide_eq_true_eq,
      beq_iff_eq]
    have hk := pow_pos' (-e).toNat
    generalize (2 : Int) ^ (-e).toNat = k at hk ⊢
    constructor
    · rintro ⟨⟨hm, -⟩, hq⟩
      have hd : k ∣ m := Int.dvd_of_emod_eq_zero hm
      rw [← hq, Int.tdiv_mul_cancel hd]
    · intro h
      subst h
      refine ⟨⟨Int.mul_emod_left _ _, ?_, ?_⟩, Int.mul_tdiv_cancel _ (by omega)⟩
      · exact Int.mul_le_mul_of_nonneg_right hi.1 (by omega)
      · exact Int.mul_lt_mul_of_pos_right hi.2 hk

/-- `Value`'s own equality would round `2^53 + 1` to `2^53`; `constants_equal` does not. -/
theorem intFloat_past_2p53 :
    intFloatEq (2 ^ 53 + 1) (.fin 1 53) = false ∧ intFloatEq (2 ^ 53) (.fin 1 53) = true ∧
    intFloatEq 3 (.fin 3 0) = true ∧ intFloatEq 3 (.fin 7 (-1)) = false ∧
    intFloatEq 0 (.fin 0 0) = true := by decide

theorem constEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ x, oeq x x = true) (v : RV) :
    constEq feq oeq v v = true := by
  cases v <;> simp [constEq, hf, ho]

theorem constEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ x y, oeq x y = oeq y x) (a b : RV) :
    constEq feq oeq a b = constEq feq oeq b a := by
  cases a <;> cases b <;> simp [constEq, hf, ho, Bool.beq_comm]

/-! ## `expr_ir_eq` -/

def exprIrEq : RD → RD → Bool
  | .const a, .const b => constEq feq oeq a b
  | .var a, .var b => a == b
  | .listComp a, .listComp b => a == b
  | .func a _, .func b _ => a == b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
  | .caseE a, .caseE b => a == b
  | .quant k a, .quant l b => k == l && a == b
  | .reduce a i, .reduce b j => a == b && i == j
  | .shortest t mn mx d, .shortest u mn' mx' d' => t == u && mn == mn' && mx == mx' && d == d'
  | .regex k s, .regex l t => k == l && s == t
  | a, b => disc a == disc b

/- Drop the `__agg_order_by_placeholder__` children everywhere (the index
filter of `nodes_eq`, applied at every level it recurses to). -/
mutual
def strip : RE → RE
  | .node d cs => .node d (stripL cs)
def stripL : List RE → List RE
  | [] => []
  | c :: cs => if isAggPlaceholder c.d then stripL cs else strip c :: stripL cs
end

/- `nodes_eq` on the stripped trees: same number of children, the payload
comparison `deq` (`expr_ir_eq`), children pairwise. -/
mutual
def eqSWith (deq : RD → RD → Bool) : RE → RE → Bool
  | .node da ca, .node db cb => ca.length == cb.length && deq da db && eqSLWith deq ca cb
def eqSLWith (deq : RD → RD → Bool) : List RE → List RE → Bool
  | a :: as, b :: bs => eqSWith deq a b && eqSLWith deq as bs
  | _, _ => true
end

def eqS : RE → RE → Bool := eqSWith (exprIrEq feq oeq)

def structEq (a b : RE) : Bool := eqS feq oeq (strip a) (strip b)

/-! ## Equivalence -/

theorem exprIrEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ x, oeq x x = true) (d : RD) :
    exprIrEq feq oeq d d = true := by
  cases d with
  | const v => simp [exprIrEq, constEq_refl feq oeq hf ho]
  | _ => simp [exprIrEq, disc]

theorem exprIrEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ x y, oeq x y = oeq y x) (a b : RD) :
    exprIrEq feq oeq a b = exprIrEq feq oeq b a := by
  cases a <;> cases b <;>
    simp [exprIrEq, disc, constEq_symm feq oeq hf ho, Bool.beq_comm, Bool.and_comm,
      Bool.and_left_comm] <;> omega

mutual
theorem eqSWith_refl (deq : RD → RD → Bool) (hd : ∀ x, deq x x = true) : ∀ a, eqSWith deq a a = true
  | .node d cs => by simp [eqSWith, hd, eqSLWith_refl deq hd cs]
theorem eqSLWith_refl (deq : RD → RD → Bool) (hd : ∀ x, deq x x = true) : ∀ l, eqSLWith deq l l = true
  | [] => by simp [eqSLWith]
  | c :: cs => by simp [eqSLWith, eqSWith_refl deq hd c, eqSLWith_refl deq hd cs]
end

theorem nodesEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ x, oeq x x = true) (a : RE) :
    structEq feq oeq a a = true := eqSWith_refl _ (exprIrEq_refl feq oeq hf ho) _

mutual
theorem eqSWith_symm (deq : RD → RD → Bool) (hd : ∀ x y, deq x y = deq y x) :
    ∀ a b, eqSWith deq a b = eqSWith deq b a
  | .node da ca, .node db cb => by
    simp only [eqSWith, hd da db, eqSLWith_symm deq hd ca cb]
    rw [Bool.beq_comm (a := ca.length)]
theorem eqSLWith_symm (deq : RD → RD → Bool) (hd : ∀ x y, deq x y = deq y x) :
    ∀ a b, eqSLWith deq a b = eqSLWith deq b a
  | [], [] => rfl
  | [], _ :: _ => by simp [eqSLWith]
  | _ :: _, [] => by simp [eqSLWith]
  | a :: as, b :: bs => by simp only [eqSLWith, eqSWith_symm deq hd a b, eqSLWith_symm deq hd as bs]
end

theorem nodesEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ x y, oeq x y = oeq y x) (a b : RE) :
    structEq feq oeq a b = structEq feq oeq b a := eqSWith_symm _ (exprIrEq_symm feq oeq hf ho) _ _

/-! ## What equality means -/

/-- Payloads compared by value (not just by variant). -/
def exact : RD → Bool
  | .var _ | .prop _ | .param _ | .listComp _ | .caseE _ | .quant _ _ | .reduce _ _
  | .shortest _ _ _ _ | .regex _ _ => true
  | .const .null | .const (.str _) | .const (.int _) | .const (.bool _) => true
  | _ => false

/-- Fine-grained kind: the variant, and for constants the value's type. -/
def kind : RD → Nat × Nat
  | .const .null => (2, 0) | .const (.bool _) => (2, 1) | .const (.int _) => (2, 2)
  | .const (.float _) => (2, 3) | .const (.str _) => (2, 4) | .const (.other t) => (2, 5 + t)
  | d => (disc d, 0)

/-- Sound on value-compared payloads of the same kind. -/
theorem exprIrEq_exact (a b : RD) (ha : exact a = true) (hb : exact b = true) (hk : kind a = kind b)
    (h : exprIrEq feq oeq a b = true) : a = b := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp_all [exact, exprIrEq, constEq, disc, kind]

/-- An Int and a Float constant match iff the float is exactly that integer. -/
theorem exprIrEq_int_float (i m e : Int) (hi : i64Lo ≤ i ∧ i < i64Hi) :
    (exprIrEq feq oeq (.const (.int i)) (.const (.float (.fin m e))) = true ↔ denotes m e i) ∧
    (exprIrEq feq oeq (.const (.float (.fin m e))) (.const (.int i)) = true ↔ denotes m e i) := by
  simp only [exprIrEq, constEq]; exact ⟨intFloatEq_iff i m e hi, intFloatEq_iff i m e hi⟩

/-- The W4-plan-3 cases are now told apart: an Int and a Float that is not that
integer, Null and Int, Bool and Int, String and Int, two quantifier kinds. -/
theorem const_kinds_distinguished (i : Int) (f : RF) (h : intFloatEq i f = false) :
    exprIrEq feq oeq (.const (.int i)) (.const (.float f)) = false ∧
    exprIrEq feq oeq (.const .null) (.const (.int i)) = false ∧
    exprIrEq feq oeq (.const (.bool true)) (.const (.int i)) = false ∧
    exprIrEq feq oeq (.const (.str "1")) (.const (.int 1)) = false ∧
    exprIrEq feq oeq (.quant 0 "x") (.quant 1 "x") = false := by
  simp [exprIrEq, constEq, h]

/-- Graph patterns are still compared by variant only (their anonymous elements
get fresh names per occurrence). -/
theorem patterns_by_variant (t p q : Nat) : exprIrEq feq oeq (.pattern t p) (.pattern t q) = true := by
  simp [exprIrEq, disc]

/-- `max(n.v % 3)` (with the parser's placeholder) and `max(n.v % k)`. -/
def aggOf (k : RV) : RE :=
  .node (.func "max" true)
    [.node (.other 20) [.node (.prop "v") [.node (.var "n") []], .node (.const k) []],
     .node (.var placeholder) []]

/-- W4-plan-3 fixed: `max(n.v % 3)` no longer matches `max(n.v % 4.0)`. -/
theorem order_by_other_aggregate_rejected :
    structEq feq oeq (aggOf (.int 3)) (aggOf (.float (.fin 1 2))) = false := by
  simp [structEq, eqS, aggOf, strip, stripL, isAggPlaceholder, RE.d, eqSWith, eqSLWith, exprIrEq,
    constEq, intFloatEq, fractZero, inI64, asI64, placeholder]

/-- …while `max(n.v % 3)` still matches `max(n.v % 3.0)` (as in C). -/
theorem order_by_int_float_same :
    structEq feq oeq (aggOf (.int 3)) (aggOf (.float (.fin 3 0))) = true := by
  simp [structEq, eqS, aggOf, strip, stripL, isAggPlaceholder, RE.d, eqSWith, eqSLWith, exprIrEq,
    constEq, intFloatEq, fractZero, inI64, asI64, placeholder, i64Lo, i64Hi]

/-- …and the same aggregate with another Int constant is told apart. -/
theorem order_by_int_distinguished : structEq feq oeq (aggOf (.int 3)) (aggOf (.int 4)) = false := by
  simp [structEq, eqS, aggOf, strip, stripL, isAggPlaceholder, RE.d, eqSWith, eqSLWith, exprIrEq,
    constEq, placeholder]

/-! ## Before PR #3254 (history: the W4-plan-3 bug) -/

/-- `expr_ir_eq` at 343e077f2: only the listed `Constant` pairs compared by value,
everything else by `discriminant`. -/
def exprIrEqPre3254 (feqN : RF → RF → Bool) : RD → RD → Bool
  | .var a, .var b => a == b
  | .func a _, .func b _ => a == b
  | .const (.str a), .const (.str b) => a == b
  | .const (.int a), .const (.int b) => a == b
  | .const (.float a), .const (.float b) => feqN a b
  | .const (.bool a), .const (.bool b) => a == b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
  | a, b => disc a == disc b

theorem pre3254_const_kinds_conflated (i : Int) (f : RF) :
    exprIrEqPre3254 feq (.const (.int i)) (.const (.float f)) = true ∧
    exprIrEqPre3254 feq (.const .null) (.const (.int i)) = true ∧
    exprIrEqPre3254 feq (.const (.bool true)) (.const (.int i)) = true ∧
    exprIrEqPre3254 feq (.const (.str "1")) (.const (.int 1)) = true ∧
    exprIrEqPre3254 feq (.quant 0 "x") (.quant 1 "x") = true := by
  simp [exprIrEqPre3254, disc]

/-- The CONFIRMED bug at 343e077f2: `max(n.v % 3)` matched `max(n.v % 4.0)`. -/
theorem pre3254_order_by_wrong_aggregate (f : RF) :
    eqSWith (exprIrEqPre3254 feq) (strip (aggOf (.int 3))) (strip (aggOf (.float f))) = true := by
  simp [aggOf, strip, stripL, isAggPlaceholder, RE.d, eqSWith, eqSLWith, exprIrEqPre3254, disc,
    placeholder]

/-! ## `replace_agg_subtrees` -/

/-- Projected items: alias, raw expression, `is_aggregation()`. -/
abbrev Item := String × RE × Bool

def matchItem (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => it.2.2 && structEq feq oeq e it.2.1).map Prod.fst

mutual
def replaceAgg (items : List Item) : RE → RE
  | .node d cs =>
    match d with
    | .func n true =>
      match matchItem feq oeq items (.node (.func n true) cs) with
      | some nm => .node (.var nm) []
      | none => .node d (replaceAggL items cs)
    | _ => .node d (replaceAggL items cs)
def replaceAggL (items : List Item) : List RE → List RE
  | [] => []
  | c :: cs => replaceAgg items c :: replaceAggL items cs
end

theorem matchItem_none (items : List Item) (e : RE) (h : ∀ it ∈ items, it.2.2 = false) :
    matchItem feq oeq items e = none := by
  unfold matchItem
  rw [List.find?_eq_none.2]; · rfl
  intro it hit; simp [h it hit]

mutual
/-- No aggregation among the projections: ORDER BY is left as written. -/
theorem replaceAgg_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ e, replaceAgg feq oeq items e = e
  | .node d cs => by
    have ih := replaceAggL_id items h cs
    cases d with
    | func n a => cases a <;> simp [replaceAgg, ih, matchItem_none feq oeq items _ h]
    | _ => simp [replaceAgg, ih]
theorem replaceAggL_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ l, replaceAggL feq oeq items l = l
  | [] => rfl
  | c :: cs => by simp [replaceAggL, replaceAgg_id items h c, replaceAggL_id items h cs]
end

/-- An aggregate call matching an item becomes that item's alias (first match). -/
theorem replaceAgg_root (items : List Item) (n : String) (cs : List RE) (nm : String)
    (h : matchItem feq oeq items (.node (.func n true) cs) = some nm) :
    replaceAgg feq oeq items (.node (.func n true) cs) = .node (.var nm) [] := by
  simp [replaceAgg, h]

/-- …and every match is justified by `structEq` against an aggregation item. -/
theorem matchItem_spec (items : List Item) (e : RE) (nm : String) (h : matchItem feq oeq items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ it.2.2 = true ∧ structEq feq oeq e it.2.1 = true := by
  unfold matchItem at h
  cases hf : items.find? (fun it => it.2.2 && structEq feq oeq e it.2.1) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

end FalkorBinder.Agg
