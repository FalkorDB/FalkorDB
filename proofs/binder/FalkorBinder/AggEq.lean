/-
# ORDER BY rewriting: `replace_projected_subtrees` and `raw_exprs_structurally_equal`

Floats are opaque codes compared by a parameter `feq` (f64 `==`, so NaN ≠ NaN);
`fint f = some i` says the float `f` is exactly the `i64` `i` (`fract() == 0`,
in `[-2^63, 2^63)`, `f as i64 == i`). Other constants (lists, maps, temporals, …)
are `other ty x`, equal per `oeq ty` within one type and never across types
(`Value::order` is injective on types).

* `nodesEq_refl` / `nodesEq_symm`: an equivalence when `feq`/`oeq` are (either `same_type`).
* `exprIrEq_exact`, `exprIrEq_exact_any_kind`: exact on every value-compared payload.
* `const_kinds_distinguished`, `order_by_float_rejected`, `order_by_float_equal`:
  `max(n.v % 3)` does not match `max(n.v % 4.0)`, still matches `max(n.v % 3.0)`
  (W4-plan-3).
* `same_type_int_float_rejected`, `order_by_div_float_rejected`: with `same_type`
  `2` never matches `2.0`, so `n.v / 2` is not `n.v / 2.0`.
* `slot_skipped_only_under_agg`, `user_placeholder_name_kept`: only the parser's
  `AggOrderByPlaceholder` child of an aggregate call is skipped.
* `patterns_by_variant`: graph patterns are still compared by variant only.
* `replaceProjected_id`: with no aggregation and no rewritable projection the
  ORDER BY tree is unchanged; `matchItem_spec` / `matchRw_spec`: every replacement
  is justified by `structEq` (aggregates without, other sub-trees with `same_type`);
  `replaceProjected_agg_root`, `rep_rw_root`: a matching sub-tree becomes the alias.
* `order_by_projected`, `order_by_shadowed_kept`, `order_by_div_float_kept`,
  `order_by_local_kept`: `RETURN n.v AS k ORDER BY n.v` sorts by `k`; a shadowed
  alias, a constant of another type and a comprehension body are left as written.
* `order_by_agg_projected`, `order_by_local_agg_kept`: a projected aggregate is
  read from its alias, except inside a comprehension body (kept, then rejected).
-/
namespace FalkorBinder.Agg

/-- `Value`s that occur as constants in raw expressions. -/
inductive RV
  | null | bool (b : Bool) | int (i : Int) | float (f : Nat) | str (s : String) | other (ty : Nat) (x : Nat)
  deriving DecidableEq, Repr

/-- Raw `ExprIR` payloads; `slot` is `ExprIR::AggOrderByPlaceholder`. -/
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
  | slot
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
  | .shortest _ _ _ _ => 9 | .regex _ _ => 10 | .slot => 11 | .other t => 12 + t

/-- `is_agg_placeholder`: `matches!(node, ExprIR::AggOrderByPlaceholder)`. -/
def isAggPlaceholder : RD → Bool
  | .slot => true
  | _ => false

theorem isAggPlaceholder_iff (d : RD) : isAggPlaceholder d = true ↔ d = .slot := by
  cases d <;> simp [isAggPlaceholder]

/-- `FuncInvocation(f) if f.is_aggregate()`. -/
def isAggFunc : RD → Bool
  | .func _ a => a
  | _ => false

/-- `discriminant` of a constant `Value`. -/
def rvKind : RV → Nat
  | .null => 0 | .bool _ => 1 | .int _ => 2 | .float _ => 3 | .str _ => 4 | .other t _ => 5 + t

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

/-- `expr_ir_eq a b same_type`. -/
def exprIrEq (sameTy : Bool) : RD → RD → Bool
  | .const a, .const b => (!sameTy || rvKind a == rvKind b) && constEq feq fint oeq a b
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

/- Drop the `AggOrderByPlaceholder` children of aggregate calls (the index
filter `skip_placeholder` of `nodes_eq`, applied at every level it recurses to). -/
mutual
def strip : RE → RE
  | .node d cs => .node d (stripL (isAggFunc d) cs)
def stripL (agg : Bool) : List RE → List RE
  | [] => []
  | c :: cs => if agg && isAggPlaceholder c.d then stripL agg cs else strip c :: stripL agg cs
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

/-- `raw_exprs_structurally_equal a b same_type`. -/
def structEq (sameTy : Bool) (a b : RE) : Bool := eqS (exprIrEq feq fint oeq sameTy) (strip a) (strip b)

/-! ## Equivalence -/

theorem constEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (v : RV) :
    constEq feq fint oeq v v = true := by
  cases v <;> simp [constEq, hf, ho]

theorem exprIrEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (s : Bool) (d : RD) :
    exprIrEq feq fint oeq s d d = true := by
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

theorem exprIrEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ t x y, oeq t x y = oeq t y x) (s : Bool) (a b : RD) :
    exprIrEq feq fint oeq s a b = exprIrEq feq fint oeq s b a := by
  cases a <;> cases b <;>
    simp [exprIrEq, disc, constEq_symm feq fint oeq hf ho, Bool.beq_comm, Bool.and_comm] <;> omega

mutual
theorem eqS_refl (peq : RD → RD → Bool) (hp : ∀ d, peq d d = true) : ∀ a, eqS peq a a = true
  | .node d cs => by simp [eqS, hp, eqSL_refl peq hp cs]
theorem eqSL_refl (peq : RD → RD → Bool) (hp : ∀ d, peq d d = true) : ∀ l, eqSL peq l l = true
  | [] => rfl
  | c :: cs => by simp [eqSL, eqS_refl peq hp c, eqSL_refl peq hp cs]
end

theorem nodesEq_refl (hf : ∀ x, feq x x = true) (ho : ∀ t x, oeq t x x = true) (s : Bool) (a : RE) :
    structEq feq fint oeq s a a = true := eqS_refl _ (exprIrEq_refl feq fint oeq hf ho s) _

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

theorem nodesEq_symm (hf : ∀ x y, feq x y = feq y x) (ho : ∀ t x y, oeq t x y = oeq t y x) (s : Bool) (a b : RE) :
    structEq feq fint oeq s a b = structEq feq fint oeq s b a :=
  eqS_symm _ (exprIrEq_symm feq fint oeq hf ho s) _ _

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
theorem exprIrEq_exact (s : Bool) (a b : RD) (ha : exact a = true) (hb : exact b = true) (_hk : kind a = kind b)
    (h : exprIrEq feq fint oeq s a b = true) : a = b := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp_all [exact, exprIrEq, constEq, disc, rvKind]

/-- Since #3254 the kind premise is not needed: different kinds never match. -/
theorem exprIrEq_exact_any_kind (s : Bool) (a b : RD) (ha : exact a = true) (hb : exact b = true)
    (h : exprIrEq feq fint oeq s a b = true) : a = b :=
  exprIrEq_exact feq fint oeq s a b ha hb (by
    cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
      simp_all [exact, exprIrEq, constEq, disc, kind, rvKind]) h

/-- An Int and a Float constant match (without `same_type`) exactly when the float is that integer. -/
theorem int_float_iff (i : Int) (f : Nat) :
    exprIrEq feq fint oeq false (.const (.int i)) (.const (.float f)) = true ↔ fint f = some i := by
  simp [exprIrEq, constEq]

/-- With `same_type` an Int never matches a Float, whatever the float. -/
theorem same_type_int_float_rejected (i : Int) (f : Nat) :
    exprIrEq feq fint oeq true (.const (.int i)) (.const (.float f)) = false := by
  simp [exprIrEq, rvKind]

/-- #3254: constants of different kinds, and quantifiers of different kinds, are told apart. -/
theorem const_kinds_distinguished (s : Bool) (i : Int) (f : Nat) (hf : fint f ≠ some i) :
    exprIrEq feq fint oeq s (.const (.int i)) (.const (.float f)) = false ∧
    exprIrEq feq fint oeq s (.const .null) (.const (.int i)) = false ∧
    exprIrEq feq fint oeq s (.const (.bool true)) (.const (.int i)) = false ∧
    exprIrEq feq fint oeq s (.const (.str "1")) (.const (.int 1)) = false ∧
    exprIrEq feq fint oeq s (.quant 0 "x") (.quant 1 "x") = false := by
  simp [exprIrEq, constEq, hf]

/-- Graph patterns (`Pattern`, `PatternComprehension`, `NestedPlan`, here `other`)
are still compared by variant only (deliberately: fresh anonymous names). -/
theorem patterns_by_variant (s : Bool) (t : Nat) : exprIrEq feq fint oeq s (.other t) (.other t) = true := by
  simp [exprIrEq, disc]

/-- The parser's slot is skipped under an aggregate call: `max(x)` with its
slot equals `max(x)` without one. -/
theorem slot_skipped_only_under_agg (x : RE) :
    strip (.node (.func "max" true) [x, .node .slot []]) = strip (.node (.func "max" true) [x]) := by
  simp [strip, stripL, RE.d, isAggPlaceholder, isAggFunc]

/-- A user variable spelled like the old placeholder is no longer dropped. -/
theorem user_placeholder_name_kept :
    strip (.node (.func "max" true) [.node (.var "__agg_order_by_placeholder__") []]) =
      .node (.func "max" true) [.node (.var "__agg_order_by_placeholder__") []] := by
  simp [strip, stripL, RE.d, isAggPlaceholder, isAggFunc]

/-- `max(n.v % 3)` (with the parser's slot) and `max(n.v % 4.0)`. -/
def aggOf (k : RV) : RE :=
  .node (.func "max" true)
    [.node (.other 20) [.node (.prop "v") [.node (.var "n") []], .node (.const k) []],
     .node .slot []]

/-- #3254: `max(n.v % 3)` no longer matches `max(n.v % 4.0)` (any float that is not 3). -/
theorem order_by_float_rejected (f : Nat) (hf : fint f ≠ some 3) :
    structEq feq fint oeq false (aggOf (.int 3)) (aggOf (.float f)) = false := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, isAggFunc, eqS, eqSL,
    exprIrEq, constEq, disc, hf]

/-- …but still matches `max(n.v % 3.0)`, as in C. -/
theorem order_by_float_equal (f : Nat) (hf : fint f = some 3) :
    structEq feq fint oeq false (aggOf (.int 3)) (aggOf (.float f)) = true := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, isAggFunc, eqS, eqSL,
    exprIrEq, constEq, disc, hf]

/-- …while the same aggregate with another Int constant is told apart. -/
theorem order_by_int_distinguished : structEq feq fint oeq false (aggOf (.int 3)) (aggOf (.int 4)) = false := by
  simp [structEq, aggOf, strip, stripL, RE.d, isAggPlaceholder, isAggFunc, eqS, eqSL, exprIrEq, constEq, disc]

/-- `n.v / k` (division is some operator tag, here 21). -/
def divOf (k : RV) : RE :=
  .node (.other 21) [.node (.prop "v") [.node (.var "n") []], .node (.const k) []]

/-- With `same_type`, `n.v / 2` does not match `n.v / 2.0`, even when the float is 2. -/
theorem order_by_div_float_rejected (f : Nat) :
    structEq feq fint oeq true (divOf (.int 2)) (divOf (.float f)) = false := by
  simp [structEq, divOf, strip, stripL, RE.d, isAggPlaceholder, isAggFunc, eqS, eqSL,
    exprIrEq, constEq, disc, rvKind]

/-! ## `replace_projected_subtrees` -/

/-- The parameters of the rewrite: float/other equality, and the function and
operator flags `is_alias_rewritable` reads. `detFn f`: `f` is not
non-deterministic, not a write and not a UDF; `okOp t`: the operator `other t`
is in the allowed list (List, Length, GetElement(s), IsNode/IsRelationship,
Or/Xor/And/Not/Negate, comparisons, In, arithmetic, Pow, Modulo, Paren);
`pcTag`: the tag of `PatternComprehension`. -/
structure Ctx where
  feq : Nat → Nat → Bool
  fint : Nat → Option Int
  oeq : Nat → Nat → Nat → Bool
  detFn : String → Bool
  okOp : Nat → Bool
  pcTag : Nat

/-- Projected items: alias, raw expression, `is_aggregation()`. -/
abbrev Item := String × RE × Bool

/-- The per-node test of `is_alias_rewritable`'s walk. -/
def nodeOk (c : Ctx) (sh : List String) : RD → Bool
  | .var v => !sh.contains v
  | .func n a => !a && c.detFn n
  | .const (.other _ _) => false
  | .const _ => true
  | .prop _ | .param _ => true
  | .other t => c.okOp t
  | _ => false

mutual
def allOk (c : Ctx) (sh : List String) : RE → Bool
  | .node d cs => nodeOk c sh d && allOkL c sh cs
def allOkL (c : Ctx) (sh : List String) : List RE → Bool
  | [] => true
  | e :: es => allOk c sh e && allOkL c sh es
end

def isVarD : RD → Bool
  | .var _ => true
  | _ => false

mutual
def hasVar : RE → Bool
  | .node d cs => isVarD d || hasVarL cs
def hasVarL : List RE → Bool
  | [] => false
  | e :: es => hasVar e || hasVarL es
end

/-- `is_alias_rewritable(node, shadowed)`: every node passes, and some node is a variable. -/
def isAliasRewritable (c : Ctx) (sh : List String) (e : RE) : Bool :=
  allOk c sh e && hasVar e

/-- `matches!(expr.root().data(), ExprIR::Variable(v) if v == name)`. -/
def isVarOf (nm : String) : RE → Bool
  | .node (.var v) _ => v == nm
  | _ => false

/-- `shadowed`: the aliases that do not project the variable of the same name. -/
def shadowedOf (items : List Item) : List String :=
  (items.filter fun it => !isVarOf it.1 it.2.1).map (·.1)

/-- `rewritable`: the projections an alias may stand for. -/
def rewritableOf (c : Ctx) (items : List Item) : List Item :=
  items.filter fun it => isAliasRewritable c [] it.2.1

/-- First aggregation item matching (constants compared by value). -/
def matchItem (c : Ctx) (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => it.2.2 && structEq c.feq c.fint c.oeq false e it.2.1).map Prod.fst

/-- First rewritable item matching (constants of the same type only). -/
def matchRw (c : Ctx) (rw : List Item) (e : RE) : Option String :=
  (rw.find? fun it => structEq c.feq c.fint c.oeq true e it.2.1).map Prod.fst

/-- `local_from`: children from this index on are left as written. -/
def localFrom (c : Ctx) : RD → Option Nat
  | .quant _ _ | .listComp _ => some 1
  | .reduce _ _ => some 2
  | .other t => if t == c.pcTag then some 0 else none
  | _ => none

mutual
/-- `replace_projected_subtrees_rec`. -/
def rep (c : Ctx) (items rw : List Item) (sh : List String) : RE → RE
  | .node d cs =>
    if isAggFunc d then
      match matchItem c items (.node d cs) with
      | some nm => .node (.var nm) []
      | none => .node d (repL c items rw sh (localFrom c d) 0 cs)
    else if isAliasRewritable c sh (.node d cs) then
      match matchRw c rw (.node d cs) with
      | some nm => .node (.var nm) []
      | none => .node d (repL c items rw sh (localFrom c d) 0 cs)
    else .node d (repL c items rw sh (localFrom c d) 0 cs)
def repL (c : Ctx) (items rw : List Item) (sh : List String) (lf : Option Nat) (i : Nat) : List RE → List RE
  | [] => []
  | e :: es => (if lf.any (· ≤ i) then e else rep c items rw sh e) :: repL c items rw sh lf (i + 1) es
end

/-- `replace_projected_subtrees(node, exprs)`. -/
def replaceProjected (c : Ctx) (items : List Item) (e : RE) : RE :=
  rep c items (rewritableOf c items) (shadowedOf items) e

theorem matchItem_none (c : Ctx) (items : List Item) (e : RE) (h : ∀ it ∈ items, it.2.2 = false) :
    matchItem c items e = none := by
  unfold matchItem
  rw [List.find?_eq_none.2]; · rfl
  intro it hit; simp [h it hit]

theorem matchRw_nil (c : Ctx) (e : RE) : matchRw c [] e = none := rfl

mutual
theorem rep_id (c : Ctx) (items : List Item) (sh : List String) (h : ∀ it ∈ items, it.2.2 = false) :
    ∀ e, rep c items [] sh e = e
  | .node d cs => by
    have ih := repL_id c items sh h (localFrom c d) 0 cs
    simp only [rep, ih, matchItem_none c items _ h, matchRw_nil]
    split <;> (try split) <;> rfl
theorem repL_id (c : Ctx) (items : List Item) (sh : List String) (h : ∀ it ∈ items, it.2.2 = false) :
    ∀ lf i l, repL c items [] sh lf i l = l
  | _, _, [] => rfl
  | lf, i, e :: es => by
    simp only [repL, rep_id c items sh h e, repL_id c items sh h lf (i + 1) es, ite_self]
end

/-- No aggregation and no rewritable expression among the projections:
ORDER BY is left as written. -/
theorem replaceProjected_id (c : Ctx) (items : List Item)
    (h : ∀ it ∈ items, it.2.2 = false) (hr : ∀ it ∈ items, isAliasRewritable c [] it.2.1 = false) (e : RE) :
    replaceProjected c items e = e := by
  have : rewritableOf c items = [] := by
    unfold rewritableOf; rw [List.filter_eq_nil_iff]; intro it hit; simp [hr it hit]
  simp only [replaceProjected, this]
  exact rep_id c items _ h e

/-- An aggregate call matching an aggregation item becomes that item's alias (first match). -/
theorem replaceProjected_agg_root (c : Ctx) (items : List Item) (n : String) (cs : List RE) (nm : String)
    (h : matchItem c items (.node (.func n true) cs) = some nm) :
    replaceProjected c items (.node (.func n true) cs) = .node (.var nm) [] := by
  simp [replaceProjected, rep, isAggFunc, h]

/-- A non-aggregate rewritable sub-tree matching a rewritable item becomes its alias. -/
theorem rep_rw_root (c : Ctx) (items rw : List Item) (sh : List String) (d : RD) (cs : List RE) (nm : String)
    (hd : isAggFunc d = false) (hr : isAliasRewritable c sh (.node d cs) = true)
    (h : matchRw c rw (.node d cs) = some nm) :
    rep c items rw sh (.node d cs) = .node (.var nm) [] := by
  simp [rep, hd, hr, h]

/-- Every aggregate replacement is justified by `structEq` against an aggregation item. -/
theorem matchItem_spec (c : Ctx) (items : List Item) (e : RE) (nm : String) (h : matchItem c items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ it.2.2 = true ∧ structEq c.feq c.fint c.oeq false e it.2.1 = true := by
  unfold matchItem at h
  cases hf : items.find? (fun it => it.2.2 && structEq c.feq c.fint c.oeq false e it.2.1) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

/-- Every alias replacement is justified by `structEq` with `same_type` against
a rewritable item. -/
theorem matchRw_spec (c : Ctx) (items : List Item) (e : RE) (nm : String)
    (h : matchRw c (rewritableOf c items) e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ isAliasRewritable c [] it.2.1 = true ∧
      structEq c.feq c.fint c.oeq true e it.2.1 = true := by
  unfold matchRw at h
  cases hf : (rewritableOf c items).find? (fun it => structEq c.feq c.fint c.oeq true e it.2.1) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [rewritableOf, List.mem_filter] at hm
    exact ⟨it, hm.1, h, hm.2, hp⟩

/-! ### Examples (every function deterministic, every operator allowed, a float is 2) -/

def c0 : Ctx :=
  { feq := fun a b => a == b, fint := fun _ => some 2, oeq := fun _ a b => a == b,
    detFn := fun _ => true, okOp := fun _ => true, pcTag := 99 }

def nv : RE := .node (.prop "v") [.node (.var "n") []]

/-- `RETURN n.v AS k ORDER BY n.v` sorts by the column `k`. -/
theorem order_by_projected : replaceProjected c0 [("k", nv, false)] nv = .node (.var "k") [] := by
  rfl

/-- `RETURN n.v AS n ORDER BY n.v`: the ORDER BY `n` is the projected `n`, so it is kept. -/
theorem order_by_shadowed_kept : replaceProjected c0 [("n", nv, false)] nv = nv := by
  rfl

/-- `RETURN n.v / 2 AS k ORDER BY n.v / 2.0` is kept, though `2.0` is the integer 2. -/
theorem order_by_div_float_kept :
    replaceProjected c0 [("k", divOf (.int 2), false)] (divOf (.float 0)) = divOf (.float 0) := by
  rfl

/-- `RETURN n.v AS k ORDER BY [n IN l | n.v]`: the body is in the comprehension's scope, kept. -/
theorem order_by_local_kept :
    replaceProjected c0 [("k", nv, false)] (.node (.listComp "n") [.node (.var "l") [], nv]) =
      .node (.listComp "n") [.node (.var "l") [], nv] := by
  rfl

/-- `count(n)` with the parser's slot. -/
def cntN : RE := .node (.func "count" true) [.node (.var "n") [], .node .slot []]

/-- `RETURN count(n) AS c ORDER BY count(n)` sorts by `c`… -/
theorem order_by_agg_projected : replaceProjected c0 [("c", cntN, true)] cntN = .node (.var "c") [] := by
  rfl

/-- …but `ORDER BY [x IN l | count(n)]` keeps the aggregate in the body as written (#3011: the
body is a local scope, left alone even when nothing in it shadows `c`; before #3011 it was
rewritten to `[x IN l | c]`). Binding then rejects it: "failed to map aggregation expression". -/
theorem order_by_local_agg_kept :
    replaceProjected c0 [("c", cntN, true)] (.node (.listComp "x") [.node (.var "l") [], cntN]) =
      .node (.listComp "x") [.node (.var "l") [], cntN] := by
  rfl

end FalkorBinder.Agg
