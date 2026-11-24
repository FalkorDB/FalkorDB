/-
# ORDER BY aggregate matching: `replace_agg_subtrees` and `raw_exprs_structurally_equal`

Floats are opaque codes compared by a parameter `feq` (f64 `==`, so NaN ≠ NaN);
`fint f = some i` says the float `f` is exactly the `i64` `i` (`fract() == 0`,
in `[-2^63, 2^63)`, `f as i64 == i`). Other constants (lists, maps, temporals, …)
are `other ty x`, equal per `oeq ty` within one type and never across types
(`Value::order` is injective on types).

* `nodesEq_refl` / `nodesEq_symm`: an equivalence when `feq`/`oeq` are.
* `exprIrEq_exact`, `exprIrEq_exact_any_kind`: exact on every value-compared payload.
* `const_kinds_distinguished`, `order_by_float_rejected`, `order_by_float_equal`:
  `max(n.v % 3)` does not match `max(n.v % 4.0)`, still matches `max(n.v % 3.0)`
  (W4-plan-3).
* `patterns_by_variant`: graph patterns are still compared by variant only.
* `replaceAgg_id`: with no matching aggregate the ORDER BY tree is unchanged;
  `matchItem_spec`: every replacement is justified by `structEq`.
-/
namespace FalkorBinder.Agg

/-- `Value`s that occur as constants in raw expressions. -/
inductive RV
  | null | bool (b : Bool) | int (i : Int) | float (f : Nat) | str (s : String) | other (ty : Nat) (x : Nat)
  deriving DecidableEq, Repr

inductive RD
  | var (n : String)
  | func (name : String) (agg : Bool)
  | const (v : RV)
  | prop (k : String)
  | param (p : String)
  | quant (kind : Nat) (v : String)
  | listComp (v : String)
  | case (hasSubject : Bool)
  | reduce (acc it : String)
  | shortest (rels : List String) (minHops : Nat) (maxHops : Option Nat) (directed : Bool)
  | regex (kind : Nat) (src : String)
  /-- `ExprIR::AggOrderPlaceholder` (#3016): the parser's ORDER BY slot of an aggregate -/
  | aggPh
  | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RE
  | node (d : RD) (cs : List RE)
  deriving Repr

def RE.d : RE → RD | .node d _ => d

/-- Variant discriminant: payloads are ignored. -/
def disc : RD → Nat
  | .var _ => 0 | .func _ _ => 1 | .const _ => 2 | .prop _ => 3 | .param _ => 4
  | .quant _ _ => 5 | .listComp _ => 6 | .case _ => 7 | .reduce _ _ => 8
  | .shortest _ _ _ _ => 9 | .regex _ _ => 10 | .aggPh => 11 | .other t => 12 + t

/-- `is_agg_placeholder` (binder.rs): since #3016 the placeholder is its own variant,
`matches!(node, ExprIR::AggOrderPlaceholder)` (before, `Variable("__agg_order_by_placeholder__")`). -/
def isAggPlaceholder : RD → Bool
  | .aggPh => true
  | _ => false

theorem isAggPlaceholder_iff (d : RD) : isAggPlaceholder d = true ↔ d = .aggPh := by
  cases d <;> simp [isAggPlaceholder]

/-- A variable a query writes, `` `__agg_order_by_placeholder__` `` included, is never taken for the
placeholder, so it is compared, not skipped (#3016). -/
theorem isAggPlaceholder_var (n : String) : isAggPlaceholder (.var n) = false := rfl

variable (feq : Nat → Nat → Bool) (fint : Nat → Option Int) (oeq : Nat → Nat → Nat → Bool)

/-- `constants_equal`: Int/Float exact, otherwise `Value` equality. -/
def constEq : RV → RV → Bool
  | .int i, .float f => fint f == some i
  | .float f, .int i => fint f == some i
  | .null, .null => true
  | .bool a, .bool b => a == b
  | .int a, .int b => a == b
  | .float a, .float b => feq a b
  | .str a, .str b => a == b
  | .other t a, .other u b => t == u && oeq t a b
  | _, _ => false

def exprIrEq : RD → RD → Bool
  | .const a, .const b => constEq feq fint oeq a b
  | .var a, .var b => a == b
  | .listComp a, .listComp b => a == b
  | .func a _, .func b _ => a == b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
  | .case a, .case b => a == b
  | .quant k a, .quant l b => k == l && a == b
  | .reduce a i, .reduce b j => a == b && i == j
  | .shortest r mn mx d, .shortest r' mn' mx' d' => r == r' && mn == mn' && mx == mx' && d == d'
  | .regex k s, .regex l t => k == l && s == t
  | a, b => disc a == disc b

/- Drop the `AggOrderPlaceholder` children everywhere (the index
filter of `nodes_eq`, applied at every level it recurses to). -/
mutual
def strip : RE → RE
  | .node d cs => .node d (stripL cs)
def stripL : List RE → List RE
  | [] => []
  | c :: cs => if isAggPlaceholder c.d then stripL cs else strip c :: stripL cs
end

/- `nodes_eq` on the stripped trees: same number of children, the payload
comparison `peq` (`expr_ir_eq`), children pairwise. -/
mutual
def eqS (peq : RD → RD → Bool) : RE → RE → Bool
  | .node da ca, .node db cb => ca.length == cb.length && peq da db && eqSL peq ca cb
def eqSL (peq : RD → RD → Bool) : List RE → List RE → Bool
  | a :: as, b :: bs => eqS peq a b && eqSL peq as bs
  | _, _ => true
end

def structEq (a b : RE) : Bool := eqS (exprIrEq feq fint oeq) (strip a) (strip b)

/-! ## Equivalence -/

theorem constEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (v : RV) :
    constEq feq fint oeq v v = true := by
  cases v <;> simp [constEq, hf, ho]

theorem exprIrEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (d : RD) :
    exprIrEq feq fint oeq d d = true := by
  cases d with
  | const v => simp [exprIrEq, constEq_refl feq fint oeq hf ho]
  | _ => simp [exprIrEq, disc]

theorem constEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ t x y, oeq t x y = oeq t y x) (a b : RV) :
    constEq feq fint oeq a b = constEq feq fint oeq b a := by
  cases a <;> cases b <;> simp [constEq, hf, Bool.beq_comm]
  rename_i t x u y
  rw [ho]
  by_cases htu : t = u
  · subst htu; rfl
  · have : (t == u) = false := by simp [htu]
    simp [this]

theorem exprIrEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ t x y, oeq t x y = oeq t y x) (a b : RD) :
    exprIrEq feq fint oeq a b = exprIrEq feq fint oeq b a := by
  cases a <;> cases b <;>
    simp [exprIrEq, disc, constEq_symm feq fint oeq hf ho, Bool.beq_comm, Bool.and_comm] <;> omega

mutual
theorem eqS_refl (peq : RD → RD → Bool) (hp : ∀ d, peq d d = true) : ∀ a, eqS peq a a = true
  | .node d cs => by simp [eqS, hp, eqSL_refl peq hp cs]
theorem eqSL_refl (peq : RD → RD → Bool) (hp : ∀ d, peq d d = true) : ∀ l, eqSL peq l l = true
  | [] => rfl
  | c :: cs => by simp [eqSL, eqS_refl peq hp c, eqSL_refl peq hp cs]
end

theorem nodesEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (a : RE) :
    structEq feq fint oeq a a = true := eqS_refl _ (exprIrEq_refl feq fint oeq hf ho) _

mutual
theorem eqS_symm (peq : RD → RD → Bool) (hp : ∀ a b, peq a b = peq b a) : ∀ a b, eqS peq a b = eqS peq b a
  | .node da ca, .node db cb => by
    simp only [eqS, hp da db, eqSL_symm peq hp ca cb]
    rw [Bool.beq_comm (a := ca.length)]
theorem eqSL_symm (peq : RD → RD → Bool) (hp : ∀ a b, peq a b = peq b a) : ∀ a b, eqSL peq a b = eqSL peq b a
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by simp only [eqSL, eqS_symm peq hp a b, eqSL_symm peq hp as bs]
end

theorem nodesEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ t x y, oeq t x y = oeq t y x) (a b : RE) :
    structEq feq fint oeq a b = structEq feq fint oeq b a := eqS_symm _ (exprIrEq_symm feq fint oeq hf ho) _ _

/-! ## What equality means -/

/-- Payloads compared by value (not just by variant). -/
def exact : RD → Bool
  | .var _ | .prop _ | .param _ => true
  | .const .null | .const (.str _) | .const (.int _) | .const (.bool _) => true
  | .quant _ _ | .listComp _ | .case _ | .reduce _ _ | .shortest _ _ _ _ | .regex _ _ => true
  | _ => false

/-- Fine-grained kind: the variant, and for constants the value's type. -/
def kind : RD → Nat
  | .const .null => 100 | .const (.bool _) => 101 | .const (.int _) => 102 | .const (.float _) => 103
  | .const (.str _) => 104 | .const (.other t _) => 200 + t
  | d => disc d

/-- Sound on value-compared payloads of the same kind. -/
theorem exprIrEq_exact (a b : RD) (ha : exact a = true) (hb : exact b = true) (_hk : kind a = kind b)
    (h : exprIrEq feq fint oeq a b = true) : a = b := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp_all [exact, exprIrEq, constEq, disc]

/-- Since #3254 the kind premise is not needed: different kinds never match. -/
theorem exprIrEq_exact_any_kind (a b : RD) (ha : exact a = true) (hb : exact b = true)
    (h : exprIrEq feq fint oeq a b = true) : a = b :=
  exprIrEq_exact feq fint oeq a b ha hb (by
    cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
      simp_all [exact, exprIrEq, constEq, disc, kind]) h

/-- An Int and a Float constant match exactly when the float is that integer. -/
theorem int_float_iff (i : Int) (f : Nat) :
    exprIrEq feq fint oeq (.const (.int i)) (.const (.float f)) = true ↔ fint f = some i := by
  simp [exprIrEq, constEq]

/-- #3254: constants of different kinds, and quantifiers of different kinds, are told apart. -/
theorem const_kinds_distinguished (i : Int) (f : Nat) (hf : fint f ≠ some i) :
    exprIrEq feq fint oeq (.const (.int i)) (.const (.float f)) = false ∧
    exprIrEq feq fint oeq (.const .null) (.const (.int i)) = false ∧
    exprIrEq feq fint oeq (.const (.bool true)) (.const (.int i)) = false ∧
    exprIrEq feq fint oeq (.const (.str "1")) (.const (.int 1)) = false ∧
    exprIrEq feq fint oeq (.quant 0 "x") (.quant 1 "x") = false := by
  simp [exprIrEq, constEq, hf]

/-- Graph patterns (`Pattern`, `PatternComprehension`, `NestedPlan`, here `other`)
are still compared by variant only (deliberately: fresh anonymous names). -/
theorem patterns_by_variant (t : Nat) : exprIrEq feq fint oeq (.other t) (.other t) = true := by
  simp [exprIrEq, disc]

/-- `max(n.v % 3)` (with the parser's placeholder) and `max(n.v % 4.0)`. -/
def aggOf (k : RV) : RE :=
  .node (.func "max" true)
    [.node (.other 20) [.node (.prop "v") [.node (.var "n") []], .node (.const k) []],
     .node .aggPh []]

/-- #3254: `max(n.v % 3)` no longer matches `max(n.v % 4.0)` (any float that is not 3). -/
theorem order_by_float_rejected (f : Nat) (hf : fint f ≠ some 3) :
    structEq feq fint oeq (aggOf (.int 3)) (aggOf (.float f)) = false := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, eqS, eqSL,
    exprIrEq, constEq, disc, hf]

/-- …but still matches `max(n.v % 3.0)`, as in C. -/
theorem order_by_float_equal (f : Nat) (hf : fint f = some 3) :
    structEq feq fint oeq (aggOf (.int 3)) (aggOf (.float f)) = true := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, eqS, eqSL,
    exprIrEq, constEq, disc, hf]

/-- …while the same aggregate with another Int constant is told apart. -/
theorem order_by_int_distinguished : structEq feq fint oeq (aggOf (.int 3)) (aggOf (.int 4)) = false := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, eqS, eqSL, exprIrEq, constEq, disc]

/-! ## `replace_agg_subtrees` -/

/-- Projected items: alias, raw expression, `is_aggregation()`. -/
abbrev Item := String × RE × Bool

def matchItem (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => it.2.2 && structEq feq fint oeq e it.2.1).map Prod.fst

mutual
def replaceAgg (items : List Item) : RE → RE
  | .node d cs =>
    match d with
    | .func n true =>
      match matchItem feq fint oeq items (.node (.func n true) cs) with
      | some nm => .node (.var nm) []
      | none => .node d (replaceAggL items cs)
    | _ => .node d (replaceAggL items cs)
def replaceAggL (items : List Item) : List RE → List RE
  | [] => []
  | c :: cs => replaceAgg items c :: replaceAggL items cs
end

theorem matchItem_none (items : List Item) (e : RE) (h : ∀ it ∈ items, it.2.2 = false) :
    matchItem feq fint oeq items e = none := by
  unfold matchItem
  rw [List.find?_eq_none.2]; · rfl
  intro it hit; simp [h it hit]

mutual
/-- No aggregation among the projections: ORDER BY is left as written. -/
theorem replaceAgg_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ e, replaceAgg feq fint oeq items e = e
  | .node d cs => by
    have ih := replaceAggL_id items h cs
    cases d with
    | func n a => cases a <;> simp [replaceAgg, ih, matchItem_none feq fint oeq items _ h]
    | _ => simp [replaceAgg, ih]
theorem replaceAggL_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ l, replaceAggL feq fint oeq items l = l
  | [] => rfl
  | c :: cs => by simp [replaceAggL, replaceAgg_id items h c, replaceAggL_id items h cs]
end

/-- An aggregate call matching an item becomes that item's alias (first match). -/
theorem replaceAgg_root (items : List Item) (n : String) (cs : List RE) (nm : String)
    (h : matchItem feq fint oeq items (.node (.func n true) cs) = some nm) :
    replaceAgg feq fint oeq items (.node (.func n true) cs) = .node (.var nm) [] := by
  simp [replaceAgg, h]

/-- …and every match is justified by `structEq` against an aggregation item. -/
theorem matchItem_spec (items : List Item) (e : RE) (nm : String) (h : matchItem feq fint oeq items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ it.2.2 = true ∧ structEq feq fint oeq e it.2.1 = true := by
  unfold matchItem at h
  cases hf : items.find? (fun it => it.2.2 && structEq feq fint oeq e it.2.1) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

end FalkorBinder.Agg
