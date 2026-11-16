/-
# ORDER BY rewriting: `replace_projected_subtrees` and `raw_exprs_structurally_equal`

| here | there |
| --- | --- |
| `RV`, `RD`, `RE` | raw (pre-bind) `ExprIR<Arc<String>>` payloads and trees |
| `disc` | `std::mem::discriminant` of the `ExprIR` variant |
| `RD.aggSlot`, `isAggPlaceholder` | `ExprIR::AggOrderByPlaceholder` (ast.rs:298), `is_agg_placeholder` binder.rs:2774-2776 |
| `constEq` | `constants_equal` binder.rs:2753-2766 |
| `exprIrEq` | `expr_ir_eq` binder.rs:2814-2830 |
| `strip`, `eqS` | `nodes_eq` binder.rs:2778-2812 (the `skip_placeholder` filter) |
| `structEq` | `raw_exprs_structurally_equal` binder.rs:2770-2833 |
| `rewritable` | `is_alias_rewritable` binder.rs:2690-2748 |
| `shadowedOf` | the `shadowed` set of `replace_projected_subtrees` binder.rs:2622-2632 |
| `replaceProj` | `replace_projected_subtrees_rec` binder.rs:2634-2684 |

Floats are opaque codes. `K.feq` is f64 `==` (NaN ≠ NaN), `K.fint f = some i` says the
float is exactly the integer `i` in `i64` range (`fract() == 0`, the range test and
`as i64`), and `K.oeq` is `Value` equality on the other constant kinds (lists, maps, …;
different kinds are never equal).

* `nodesEq_refl` / `nodesEq_symm`: an equivalence when `feq`/`oeq` are.
* `exprIrEq_exact`: payloads compared by value are equal when `expr_ir_eq` says so;
  `const_kinds_distinguished`: constants of different kinds never match, except an
  `Int` and a `Float` holding exactly that integer (bug 9 fixed by #3011;
  `pre3011_const_kinds_conflated` keeps the old witness on `exprIrEqPre`).
* `slot_skipped_only_under_aggregate`, `alias_named_like_slot_kept`: only the parser's
  slot under an aggregate call is skipped.
* `replaceProj_root_agg` / `replaceProj_root_proj` / `matchAgg_spec` / `matchProj_spec`:
  every replacement is the alias of a structurally equal item;
  `replaceProj_id`: no aggregate or rewritable item → ORDER BY unchanged;
  `replaceProj_shadowed`, `replaceProj_local_kept`: shadowed and comprehension-local
  subtrees are not replaced.
* **`pr3011_orderBy_div_int_float`** (counterexample, #3011): `constants_equal` lets
  `v / 2.0` match the projected `v / 2`, so `ORDER BY v / 2.0` sorts by the integer
  quotient; before #3011 a non-aggregate ORDER BY was never rewritten (`replaceAgg_id`).
-/
namespace FalkorBinder.Agg

/-- `Value`s that occur as constants in raw expressions. -/
inductive RV
  | null | bool (b : Bool) | int (i : Int) | float (f : Nat) | str (s : String) | other (tag : Nat)
  deriving DecidableEq, Repr

/-- `ExprIR<Arc<String>>` payloads. `func name agg safe`: `safe` = not
`non_deterministic`, not `write`, not a UDF (`is_alias_rewritable`). `op t`: the
payload-free variants `is_alias_rewritable` accepts (`List`, `Length`, `GetElement(s)`,
`IsNode`, `IsRelationship`, `Or`, `Xor`, `And`, `Not`, `Negate`, comparisons, `In`,
arithmetic, `Paren`); `other t`: every other variant. -/
inductive RD
  | var (n : String)
  | func (name : String) (agg : Bool) (safe : Bool)
  | const (v : RV)
  | prop (k : String)
  | param (p : String)
  | quant (kind : Nat) (v : String)
  | listComp (v : String)
  | reduce (acc it : String)
  | patComp
  | aggSlot
  | op (tag : Nat)
  | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RE
  | node (d : RD) (cs : List RE)
  deriving Repr

def RE.d : RE → RD | .node d _ => d

/-- Variant discriminant: payloads are ignored. -/
def disc : RD → Nat
  | .var _ => 0 | .func _ _ _ => 1 | .const _ => 2 | .prop _ => 3 | .param _ => 4
  | .quant _ _ => 5 | .listComp _ => 6 | .reduce _ _ => 7 | .patComp => 8 | .aggSlot => 9
  | .op t => 10 + 2 * t | .other t => 11 + 2 * t

/-- `is_agg_placeholder`: the slot is a variant of its own, not a name. -/
def isAggPlaceholder : RD → Bool
  | .aggSlot => true
  | _ => false

theorem isAggPlaceholder_iff (d : RD) : isAggPlaceholder d = true ↔ d = .aggSlot := by
  cases d <;> simp [isAggPlaceholder]

/-- `matches!(node.data(), ExprIR::FuncInvocation(f) if f.is_aggregate())` -/
def isAggFunc : RD → Bool
  | .func _ agg _ => agg
  | _ => false

structure K where
  feq : Nat → Nat → Bool
  fint : Nat → Option Int
  oeq : Nat → Nat → Bool

variable (k : K)

/-- `constants_equal`: an `Int` and a `Float` match iff the float is exactly that
integer; otherwise `Value` equality (`compare_value` = `Equal`), which never equates
different kinds and equates `Null` with `Null`. -/
def constEq : RV → RV → Bool
  | .int i, .float f => k.fint f == some i
  | .float f, .int i => k.fint f == some i
  | .null, .null => true
  | .bool a, .bool b => a == b
  | .int a, .int b => a == b
  | .float a, .float b => k.feq a b
  | .str a, .str b => a == b
  | .other a, .other b => k.oeq a b
  | _, _ => false

def exprIrEq : RD → RD → Bool
  | .var a, .var b => a == b
  | .func a _ _, .func b _ _ => a == b
  | .const a, .const b => constEq k a b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
  | a, b => disc a == disc b

/- `nodes_eq` skips the slot child of an aggregate call only (`skip_placeholder`). -/
mutual
def strip : RE → RE
  | .node d cs => .node d (stripL (isAggFunc d) cs)
def stripL (agg : Bool) : List RE → List RE
  | [] => []
  | c :: cs => if agg && isAggPlaceholder c.d then stripL agg cs else strip c :: stripL agg cs
end

/- `nodes_eq` on the stripped trees: same number of children, `expr_ir_eq`
on the payloads, children pairwise. -/
mutual
def eqS : RE → RE → Bool
  | .node da ca, .node db cb => ca.length == cb.length && exprIrEq k da db && eqSL ca cb
def eqSL : List RE → List RE → Bool
  | a :: as, b :: bs => eqS a b && eqSL as bs
  | _, _ => true
end

def structEq (a b : RE) : Bool := eqS k (strip a) (strip b)

/-! ## Equivalence -/

theorem constEq_refl (hf : ∀ x, k.feq x x = true) (ho : ∀ x, k.oeq x x = true) (v : RV) :
    constEq k v v = true := by
  cases v <;> simp [constEq, hf, ho]

theorem constEq_symm (hf : ∀ x y, k.feq x y = k.feq y x) (ho : ∀ x y, k.oeq x y = k.oeq y x) (a b : RV) :
    constEq k a b = constEq k b a := by
  cases a <;> cases b <;> simp [constEq, hf, ho, Bool.beq_comm]

theorem exprIrEq_refl (hf : ∀ x, k.feq x x = true) (ho : ∀ x, k.oeq x x = true) (d : RD) :
    exprIrEq k d d = true := by
  cases d with
  | const v => simp [exprIrEq, constEq_refl k hf ho]
  | _ => simp [exprIrEq, disc]

theorem exprIrEq_symm (hf : ∀ x y, k.feq x y = k.feq y x) (ho : ∀ x y, k.oeq x y = k.oeq y x) (a b : RD) :
    exprIrEq k a b = exprIrEq k b a := by
  cases a <;> cases b <;>
    simp [exprIrEq, disc, constEq_symm k hf ho, Bool.beq_comm] <;> omega

mutual
theorem eqS_refl (hf : ∀ x, k.feq x x = true) (ho : ∀ x, k.oeq x x = true) : ∀ a, eqS k a a = true
  | .node d cs => by simp [eqS, exprIrEq_refl k hf ho, eqSL_refl hf ho cs]
theorem eqSL_refl (hf : ∀ x, k.feq x x = true) (ho : ∀ x, k.oeq x x = true) : ∀ l, eqSL k l l = true
  | [] => rfl
  | c :: cs => by simp [eqSL, eqS_refl hf ho c, eqSL_refl hf ho cs]
end

theorem nodesEq_refl (hf : ∀ x, k.feq x x = true) (ho : ∀ x, k.oeq x x = true) (a : RE) :
    structEq k a a = true := eqS_refl k hf ho _

mutual
theorem eqS_symm (hf : ∀ x y, k.feq x y = k.feq y x) (ho : ∀ x y, k.oeq x y = k.oeq y x) :
    ∀ a b, eqS k a b = eqS k b a
  | .node da ca, .node db cb => by
    simp only [eqS, exprIrEq_symm k hf ho da db, eqSL_symm hf ho ca cb]
    rw [Bool.beq_comm (a := ca.length)]
theorem eqSL_symm (hf : ∀ x y, k.feq x y = k.feq y x) (ho : ∀ x y, k.oeq x y = k.oeq y x) :
    ∀ a b, eqSL k a b = eqSL k b a
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by simp only [eqSL, eqS_symm hf ho a b, eqSL_symm hf ho as bs]
end

theorem nodesEq_symm (hf : ∀ x y, k.feq x y = k.feq y x) (ho : ∀ x y, k.oeq x y = k.oeq y x) (a b : RE) :
    structEq k a b = structEq k b a := eqS_symm k hf ho _ _

/-! ## What equality means -/

/-- Payloads compared by value (not just by variant). -/
def exact : RD → Bool
  | .var _ | .prop _ | .param _ => true
  | .const (.str _) | .const (.int _) | .const (.bool _) => true
  | _ => false

/-- Fine-grained kind: the variant, and for constants the value's type. -/
def kind : RD → Nat
  | .const .null => 1000 | .const (.bool _) => 1001 | .const (.int _) => 1002 | .const (.float _) => 1003
  | .const (.str _) => 1004 | .const (.other t) => 2000 + t
  | d => disc d

/-- Sound on value-compared payloads of the same kind. -/
theorem exprIrEq_exact (a b : RD) (ha : exact a = true) (hb : exact b = true) (hk : kind a = kind b)
    (h : exprIrEq k a b = true) : a = b := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp_all [exact, exprIrEq, constEq, disc, kind]

/-- Bug 9 fixed (#3011): constants of different kinds never match — except an `Int`
and a `Float` that holds exactly that integer. (Quantifier kinds are still compared
by variant only; `is_alias_rewritable` never admits a quantifier.) -/
theorem const_kinds_distinguished (i : Int) (f : Nat) :
    exprIrEq k (.const (.int i)) (.const (.float f)) = (k.fint f == some i) ∧
    exprIrEq k (.const .null) (.const (.int i)) = false ∧
    exprIrEq k (.const (.bool true)) (.const (.int i)) = false ∧
    exprIrEq k (.const (.str "1")) (.const (.int 1)) = false ∧
    exprIrEq k (.quant 0 "x") (.quant 1 "x") = true := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The pre-#3011 `expr_ir_eq` (discriminant fallback for constants), kept for the
historical witness below. -/
def exprIrEqPre : RD → RD → Bool
  | .var a, .var b => a == b
  | .func a _ _, .func b _ _ => a == b
  | .const (.str a), .const (.str b) => a == b
  | .const (.int a), .const (.int b) => a == b
  | .const (.float a), .const (.float b) => k.feq a b
  | .const (.bool a), .const (.bool b) => a == b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
  | a, b => disc a == disc b

/-- Historical (bug 9, fixed by #3011): before it, constants of different kinds —
and any two `Null`s / lists / maps / quantifiers of different kind — were "equal". -/
theorem pre3011_const_kinds_conflated (i : Int) (f : Nat) :
    exprIrEqPre k (.const (.int i)) (.const (.float f)) = true ∧
    exprIrEqPre k (.const .null) (.const (.int i)) = true ∧
    exprIrEqPre k (.const (.bool true)) (.const (.int i)) = true ∧
    exprIrEqPre k (.const (.str "1")) (.const (.int 1)) = true ∧
    exprIrEqPre k (.quant 0 "x") (.quant 1 "x") = true := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `max(n.v % c)` with the parser's slot (`.op 0` = `%`). -/
def aggOf (c : RV) : RE :=
  .node (.func "max" true true)
    [.node (.op 0) [.node (.prop "v") [.node (.var "n") []], .node (.const c) []],
     .node .aggSlot []]

/-- Bug 9 fixed: `max(n.v % 3)` and `max(n.v % 4.0)` match only if the float is
exactly `3` — so `4.0` (`fint f = some 4`) is told apart. -/
theorem order_by_wrong_aggregate_fixed (f : Nat) :
    structEq k (aggOf (.int 3)) (aggOf (.float f)) = (k.fint f == some 3) := by
  simp [structEq, aggOf, strip, stripL, isAggFunc, isAggPlaceholder, RE.d, eqS, eqSL,
    exprIrEq, constEq, disc]

/-- …and the same aggregate with another Int constant is told apart. -/
theorem order_by_int_distinguished : structEq k (aggOf (.int 3)) (aggOf (.int 4)) = false := by
  simp [structEq, aggOf, strip, stripL, isAggFunc, isAggPlaceholder, RE.d, eqS, eqSL,
    exprIrEq, constEq, disc]

/-- The slot under an aggregate is skipped: `max(n.v)` with and without it match. -/
theorem slot_skipped_only_under_aggregate (x : RE) :
    strip (.node (.func "max" true true) [x, .node .aggSlot []]) =
      strip (.node (.func "max" true true) [x]) ∧
    strip (.node (.func "f" false true) [x, .node .aggSlot []]) =
      .node (.func "f" false true) [strip x, .node .aggSlot []] := by
  simp [strip, stripL, isAggFunc, isAggPlaceholder, RE.d]

/-- A user variable spelled like the old reserved name is an ordinary child, even
under an aggregate (`count(__agg_order_by_placeholder__)` keeps its argument). -/
theorem alias_named_like_slot_kept :
    strip (.node (.func "count" true true) [.node (.var "__agg_order_by_placeholder__") []]) =
      .node (.func "count" true true) [.node (.var "__agg_order_by_placeholder__") []] := by
  simp [strip, stripL, isAggFunc, isAggPlaceholder, RE.d]

/-! ## `is_alias_rewritable` -/

/-- One node of the walk: `false` = return false. -/
def nodeOk (sh : List String) : RD → Bool
  | .var v => !sh.contains v
  | .func _ agg safe => !agg && safe
  | .const (.bool _) | .const (.int _) | .const (.float _) | .const (.str _) => true
  | .prop _ | .param _ | .op _ => true
  | _ => false

mutual
def allOk (sh : List String) : RE → Bool
  | .node d cs => nodeOk sh d && allOkL sh cs
def allOkL (sh : List String) : List RE → Bool
  | [] => true
  | c :: cs => allOk sh c && allOkL sh cs
end

mutual
def hasVar : RE → Bool
  | .node (.var _) _ => true
  | .node _ cs => hasVarL cs
def hasVarL : List RE → Bool
  | [] => false
  | c :: cs => hasVar c || hasVarL cs
end

/-- `is_alias_rewritable(node, shadowed)`: every node accepted and some variable. -/
def rewritable (sh : List String) (e : RE) : Bool := allOk sh e && hasVar e

/-! ## `replace_projected_subtrees` -/

/-- Projected items: alias and raw expression. -/
abbrev Item := String × RE

mutual
/-- `QueryExpr::is_aggregation` (ast.rs:519-526): some node is an aggregate call. -/
def isAggregation : RE → Bool
  | .node d cs => isAggFunc d || isAggregationL cs
def isAggregationL : List RE → Bool
  | [] => false
  | c :: cs => isAggregation c || isAggregationL cs
end

/-- The item's root is `Variable(alias)` itself (`n AS n`, or `RETURN n`). -/
def isSelf (it : Item) : Bool :=
  match it.2 with
  | .node (.var v) _ => v == it.1
  | _ => false

/-- `shadowed`: every alias that is not a bare re-projection of its own name. -/
def shadowedOf (items : List Item) : List String := (items.filter (fun it => !isSelf it)).map (·.1)

def matchAgg (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => isAggregation it.2 && structEq k e it.2).map Prod.fst

def matchProj (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => rewritable [] it.2 && structEq k e it.2).map Prod.fst

/-- Children from `local_from` on are evaluated with names the node binds. -/
def isLocal : RD → Nat → Bool
  | .quant _ _, i | .listComp _, i => 1 ≤ i
  | .reduce _ _, i => 2 ≤ i
  | .patComp, _ => true
  | _, _ => false

mutual
def replaceProj (items : List Item) (sh : List String) : RE → RE
  | .node d cs =>
    if isAggFunc d then
      match matchAgg k items (.node d cs) with
      | some nm => .node (.var nm) []
      | none => .node d (replaceProjL items sh d 0 cs)
    else if rewritable sh (.node d cs) then
      match matchProj k items (.node d cs) with
      | some nm => .node (.var nm) []
      | none => .node d (replaceProjL items sh d 0 cs)
    else .node d (replaceProjL items sh d 0 cs)
def replaceProjL (items : List Item) (sh : List String) (p : RD) : Nat → List RE → List RE
  | _, [] => []
  | i, c :: cs => (if isLocal p i then c else replaceProj items sh c) :: replaceProjL items sh p (i + 1) cs
end

/-- `replace_projected_subtrees` -/
def replaceTop (items : List Item) (e : RE) : RE := replaceProj k items (shadowedOf items) e

theorem find_none_of {α} (l : List α) (p : α → Bool) (h : ∀ a ∈ l, p a = false) : l.find? p = none :=
  List.find?_eq_none.2 (fun a ha => by simp [h a ha])

theorem matchAgg_none (items : List Item) (e : RE) (h : ∀ it ∈ items, isAggregation it.2 = false) :
    matchAgg k items e = none := by
  unfold matchAgg; rw [find_none_of _ _ (fun it hit => by simp [h it hit])]; rfl

theorem matchProj_none (items : List Item) (e : RE) (h : ∀ it ∈ items, rewritable [] it.2 = false) :
    matchProj k items e = none := by
  unfold matchProj; rw [find_none_of _ _ (fun it hit => by simp [h it hit])]; rfl

mutual
/-- No aggregate and no rewritable item: ORDER BY is left as written. -/
theorem replaceProj_id (items : List Item) (sh : List String)
    (ha : ∀ it ∈ items, isAggregation it.2 = false) (hr : ∀ it ∈ items, rewritable [] it.2 = false) :
    ∀ e, replaceProj k items sh e = e
  | .node d cs => by
    have ih := replaceProjL_id items sh ha hr d 0 cs
    simp only [replaceProj, matchAgg_none k items _ ha, matchProj_none k items _ hr, ih]
    split <;> (try split) <;> rfl
theorem replaceProjL_id (items : List Item) (sh : List String)
    (ha : ∀ it ∈ items, isAggregation it.2 = false) (hr : ∀ it ∈ items, rewritable [] it.2 = false) :
    ∀ p i l, replaceProjL k items sh p i l = l
  | _, _, [] => rfl
  | p, i, c :: cs => by
    simp only [replaceProjL, replaceProj_id items sh ha hr c, replaceProjL_id items sh ha hr p (i + 1) cs]
    split <;> rfl
end

/-- An aggregate call matching an aggregation item becomes that item's alias. -/
theorem replaceProj_root_agg (items : List Item) (sh : List String) (d : RD) (cs : List RE) (nm : String)
    (hd : isAggFunc d = true) (h : matchAgg k items (.node d cs) = some nm) :
    replaceProj k items sh (.node d cs) = .node (.var nm) [] := by
  simp [replaceProj, hd, h]

/-- A rewritable non-aggregate subtree matching a rewritable item becomes its alias. -/
theorem replaceProj_root_proj (items : List Item) (sh : List String) (d : RD) (cs : List RE) (nm : String)
    (hd : isAggFunc d = false) (hw : rewritable sh (.node d cs) = true)
    (h : matchProj k items (.node d cs) = some nm) :
    replaceProj k items sh (.node d cs) = .node (.var nm) [] := by
  simp [replaceProj, hd, hw, h]

/-- A subtree mentioning a shadowed alias is never replaced at its root (when not an aggregate). -/
theorem replaceProj_shadowed (items : List Item) (sh : List String) (d : RD) (cs : List RE)
    (hd : isAggFunc d = false) (hw : rewritable sh (.node d cs) = false) :
    replaceProj k items sh (.node d cs) = .node d (replaceProjL k items sh d 0 cs) := by
  simp [replaceProj, hd, hw]

/-- The children a comprehension evaluates with its own names are kept as written. -/
theorem replaceProj_local_kept (items : List Item) (sh : List String) (p : RD) (i : Nat) (c : RE) (cs : List RE)
    (h : isLocal p i = true) :
    replaceProjL k items sh p i (c :: cs) = c :: replaceProjL k items sh p (i + 1) cs := by
  simp [replaceProjL, h]

theorem find_spec {α} (l : List α) (p : α → Bool) (a : α) (h : l.find? p = some a) : a ∈ l ∧ p a = true :=
  ⟨List.mem_of_find?_eq_some h, List.find?_some h⟩

/-- Every aggregate match is justified by `structEq` against an aggregation item. -/
theorem matchAgg_spec (items : List Item) (e : RE) (nm : String) (h : matchAgg k items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ isAggregation it.2 = true ∧ structEq k e it.2 = true := by
  unfold matchAgg at h
  cases hf : items.find? (fun it => isAggregation it.2 && structEq k e it.2) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have ⟨hm, hp⟩ := find_spec _ _ _ hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

/-- Every non-aggregate match is justified by `structEq` against a rewritable item. -/
theorem matchProj_spec (items : List Item) (e : RE) (nm : String) (h : matchProj k items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ rewritable [] it.2 = true ∧ structEq k e it.2 = true := by
  unfold matchProj at h
  cases hf : items.find? (fun it => rewritable [] it.2 && structEq k e it.2) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have ⟨hm, hp⟩ := find_spec _ _ _ hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

/-- #2925: `RETURN n.i AS x ORDER BY n.i` sorts by `x`; `RETURN n.x AS n ORDER BY n.x`
does not (the ORDER BY `n` is the projected `n`). -/
theorem orderBy_repeat_reads_alias :
    let ni : RE := .node (.prop "i") [.node (.var "n") []]
    replaceTop k [("x", ni)] ni = .node (.var "x") [] ∧
    replaceTop k [("n", ni)] ni = ni := by
  intro ni
  have hs : structEq k ni ni = true := by
    simp [ni, structEq, strip, stripL, isAggFunc, isAggPlaceholder, RE.d, eqS, eqSL, exprIrEq]
  have hw : rewritable [] ni = true := rfl
  have hm : matchProj k [("x", ni)] ni = some "x" := by
    simp only [matchProj, List.find?_cons, hw, hs, Bool.and_self]; rfl
  constructor
  · have hsh : shadowedOf [("x", ni)] = ["x"] := rfl
    have hw' : rewritable ["x"] ni = true := rfl
    unfold replaceTop; rw [hsh]
    exact replaceProj_root_proj k _ _ _ _ _ rfl hw' hm
  · have hsh : shadowedOf [("n", ni)] = ["n"] := rfl
    have hw' : rewritable ["n"] ni = false := rfl
    unfold replaceTop; rw [hsh, replaceProj_shadowed k _ _ _ _ rfl hw']
    rfl

/-! ## Counterexample (#3011): an `Int` and an exactly-equal `Float` are not the same expression -/

/-- `v / c` (`.op 1` = `Div`). -/
def divBy (c : RV) : RE := .node (.op 1) [.node (.var "v") [], .node (.const c) []]

/-- The ORDER BY key of `v / c` for an integer `v`, as a fraction `num / den`
(`Value::div`, value.rs:1159-1180): `Int / Int` is the truncated integer quotient,
`Int / Float` the real quotient. `fint f = some d` means the float is exactly `d`. -/
def divKey (v : Int) : RV → Option (Int × Int)
  | .int d => if d = 0 then none else some (v.tdiv d, 1)
  | .float f => (k.fint f).bind fun d => if d = 0 then none else some (v, d)
  | _ => none

/-- `a < b` on positive-denominator fractions. -/
def fracLt (a b : Int × Int) : Bool := decide (a.1 * b.2 < b.1 * a.2)

/-- **Counterexample.** `UNWIND [2, 3] AS v RETURN v / 2 AS k, v ORDER BY v / 2.0 DESC, v`:
no item is an aggregation, so before #3011 the ORDER BY was bound as written
(`replaceAgg_id`); now `v / 2.0` is replaced by `k`, the alias of `v / 2`. Its key
ties `v = 2` and `v = 3` (both `1`), where the written key orders `3` (1.5) above `2`
(1.0) — the sort follows the tie-breaker `v` instead. -/
theorem pr3011_orderBy_div_int_float (f : Nat) (hf : k.fint f = some 2) :
    let items : List Item := [("k", divBy (.int 2)), ("v", .node (.var "v") [])]
    (∀ it ∈ items, isAggregation it.2 = false) ∧
    replaceTop k items (divBy (.float f)) = .node (.var "k") [] ∧
    divKey k 2 (.int 2) = divKey k 3 (.int 2) ∧
    (∃ a b, divKey k 2 (.float f) = some a ∧ divKey k 3 (.float f) = some b ∧ fracLt a b = true) := by
  intro items
  have hs : structEq k (divBy (.float f)) (divBy (.int 2)) = true := by
    simp [structEq, divBy, strip, stripL, isAggFunc, isAggPlaceholder, RE.d, eqS, eqSL, exprIrEq,
      constEq, hf]
  have hw : rewritable [] (divBy (.int 2)) = true := rfl
  have hm : matchProj k items (divBy (.float f)) = some "k" := by
    simp only [items, matchProj, List.find?_cons, hw, hs, Bool.and_self]; rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro it hit
    simp only [items, List.mem_cons, List.not_mem_nil, or_false] at hit
    rcases hit with rfl | rfl <;> rfl
  · have hsh : shadowedOf items = ["k"] := rfl
    have hw' : rewritable ["k"] (divBy (.float f)) = true := rfl
    unfold replaceTop; rw [hsh]
    exact replaceProj_root_proj k _ _ _ _ _ rfl hw' hm
  · rfl
  · exact ⟨(2, 2), (3, 2), by simp [divKey, hf], by simp [divKey, hf], rfl⟩

end FalkorBinder.Agg
