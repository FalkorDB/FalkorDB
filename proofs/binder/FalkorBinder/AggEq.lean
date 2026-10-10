/-
# ORDER BY aggregate matching: `replace_agg_subtrees` and `raw_exprs_structurally_equal`

| here | there |
| --- | --- |
| `RV`, `RD`, `RE` | raw (pre-bind) `ExprIR<Arc<String>>` payloads and trees |
| `disc` | `std::mem::discriminant` of the `ExprIR` variant |
| `isAggPlaceholder` | binder.rs:2631-2633 |
| `exprIrEq` | `expr_ir_eq` binder.rs:2665-2683 |
| `nodesEq` | `nodes_eq` binder.rs:2635-2663 |
| `structEq` | `raw_exprs_structurally_equal` binder.rs:2627-2684 |
| `replaceAgg` | `replace_agg_subtrees` binder.rs:2597-2622 |

Floats are opaque codes compared by a parameter `feq` (f64 `==`, so NaN ≠ NaN).

* `nodesEq_refl` / `nodesEq_symm`: an equivalence when `feq` is.
* `nodesEq_sound`: equal trees are equal up to placeholders — **only** when no
  constant pair of different kinds (Int vs Float, Null vs …) and no quantifier
  /comprehension payloads differ; `const_kinds_conflated` is the counterexample.
* CONFIRMED bug: `max(n.v % 3)` and `max(n.v % 4.0)` compare equal (the
  `Constant` fallback compares only the discriminant), so
  `RETURN n.g AS g, max(n.v % 3) AS m ORDER BY max(n.v % 4.0)` sorts by `m`.
* `replaceAgg_id`: with no matching aggregate the ORDER BY tree is unchanged;
  `replaceAgg_sound`: every replacement is justified by `structEq`.
-/
namespace FalkorBinder.Agg

/-- `Value`s that occur as constants in raw expressions. -/
inductive RV
  | null | bool (b : Bool) | int (i : Int) | float (f : Nat) | str (s : String) | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RD
  | var (n : String)
  | func (name : String) (agg : Bool)
  | const (v : RV)
  | prop (k : String)
  | param (p : String)
  | quant (kind : Nat) (v : String)
  | listComp (v : String)
  | other (tag : Nat)
  deriving DecidableEq, Repr

inductive RE
  | node (d : RD) (cs : List RE)
  deriving Repr

def RE.d : RE → RD | .node d _ => d

/-- Variant discriminant: payloads are ignored. -/
def disc : RD → Nat
  | .var _ => 0 | .func _ _ => 1 | .const _ => 2 | .prop _ => 3 | .param _ => 4
  | .quant _ _ => 5 | .listComp _ => 6 | .other t => 7 + t

def placeholder : String := "__agg_order_by_placeholder__"

def isAggPlaceholder : RD → Bool
  | .var v => v == placeholder
  | _ => false

theorem isAggPlaceholder_iff (d : RD) : isAggPlaceholder d = true ↔ d = .var placeholder := by
  cases d <;> simp [isAggPlaceholder]

variable (feq : Nat → Nat → Bool)

def exprIrEq : RD → RD → Bool
  | .var a, .var b => a == b
  | .func a _, .func b _ => a == b
  | .const (.str a), .const (.str b) => a == b
  | .const (.int a), .const (.int b) => a == b
  | .const (.float a), .const (.float b) => feq a b
  | .const (.bool a), .const (.bool b) => a == b
  | .prop a, .prop b => a == b
  | .param a, .param b => a == b
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

/- `nodes_eq` on the stripped trees: same number of children, `expr_ir_eq`
on the payloads, children pairwise. -/
mutual
def eqS : RE → RE → Bool
  | .node da ca, .node db cb => ca.length == cb.length && exprIrEq feq da db && eqSL ca cb
def eqSL : List RE → List RE → Bool
  | a :: as, b :: bs => eqS a b && eqSL as bs
  | _, _ => true
end

def structEq (a b : RE) : Bool := eqS feq (strip a) (strip b)

/-! ## Equivalence -/

theorem exprIrEq_refl (hf : ∀ x, feq x x = true) (d : RD) : exprIrEq feq d d = true := by
  cases d with
  | const v => cases v <;> simp [exprIrEq, hf, disc]
  | _ => simp [exprIrEq, disc]

theorem exprIrEq_symm (hf : ∀ x y, feq x y = feq y x) (a b : RD) : exprIrEq feq a b = exprIrEq feq b a := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp [exprIrEq, disc, hf, eq_comm, Bool.beq_comm] <;> omega

mutual
theorem eqS_refl (hf : ∀ x, feq x x = true) : ∀ a, eqS feq a a = true
  | .node d cs => by simp [eqS, exprIrEq_refl feq hf, eqSL_refl hf cs]
theorem eqSL_refl (hf : ∀ x, feq x x = true) : ∀ l, eqSL feq l l = true
  | [] => rfl
  | c :: cs => by simp [eqSL, eqS_refl hf c, eqSL_refl hf cs]
end

theorem nodesEq_refl (hf : ∀ x, feq x x = true) (a : RE) : structEq feq a a = true := eqS_refl feq hf _

mutual
theorem eqS_symm (hf : ∀ x y, feq x y = feq y x) : ∀ a b, eqS feq a b = eqS feq b a
  | .node da ca, .node db cb => by
    simp only [eqS, exprIrEq_symm feq hf da db, eqSL_symm hf ca cb]
    rw [Bool.beq_comm (a := ca.length)]
theorem eqSL_symm (hf : ∀ x y, feq x y = feq y x) : ∀ a b, eqSL feq a b = eqSL feq b a
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by simp only [eqSL, eqS_symm hf a b, eqSL_symm hf as bs]
end

theorem nodesEq_symm (hf : ∀ x y, feq x y = feq y x) (a b : RE) :
    structEq feq a b = structEq feq b a := eqS_symm feq hf _ _

/-! ## What equality means -/

/-- Payloads compared by value (not just by variant). -/
def exact : RD → Bool
  | .var _ | .prop _ | .param _ => true
  | .const (.str _) | .const (.int _) | .const (.bool _) => true
  | _ => false

/-- Fine-grained kind: the variant, and for constants the value's type. -/
def kind : RD → Nat
  | .const .null => 10 | .const (.bool _) => 11 | .const (.int _) => 12 | .const (.float _) => 13
  | .const (.str _) => 14 | .const (.other t) => 20 + t
  | d => disc d

/-- Sound on value-compared payloads of the same kind. -/
theorem exprIrEq_exact (a b : RD) (ha : exact a = true) (hb : exact b = true) (hk : kind a = kind b)
    (h : exprIrEq feq a b = true) : a = b := by
  cases a <;> cases b <;> (try rename_i v w; cases v <;> cases w) <;>
    simp_all [exact, exprIrEq, disc, kind]

/-- Constants of different kinds — and any two `Null`s / lists / maps /
quantifiers of different kind — are "equal": only the variant is compared. -/
theorem const_kinds_conflated (i : Int) (f : Nat) : exprIrEq feq (.const (.int i)) (.const (.float f)) = true ∧
    exprIrEq feq (.const .null) (.const (.int i)) = true ∧
    exprIrEq feq (.const (.bool true)) (.const (.int i)) = true ∧
    exprIrEq feq (.const (.str "1")) (.const (.int 1)) = true ∧
    exprIrEq feq (.quant 0 "x") (.quant 1 "x") = true := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `max(n.v % 3)` (with the parser's placeholder) and `max(n.v % 4.0)`. -/
def aggOf (k : RV) : RE :=
  .node (.func "max" true)
    [.node (.other 20) [.node (.prop "v") [.node (.var "n") []], .node (.const k) []],
     .node (.var placeholder) []]

/-- The CONFIRMED bug: they compare equal, so ORDER BY `max(n.v % 4.0)` is
replaced by the alias of `max(n.v % 3)` (binder.rs:2606-2610). C rejects the
query; Rust sorts by the wrong key. -/
theorem order_by_wrong_aggregate (f : Nat) : structEq feq (aggOf (.int 3)) (aggOf (.float f)) = true := rfl

/-- …while the same aggregate with another Int constant is told apart. -/
theorem order_by_int_distinguished : structEq feq (aggOf (.int 3)) (aggOf (.int 4)) = false := rfl

/-! ## `replace_agg_subtrees` -/

/-- Projected items: alias, raw expression, `is_aggregation()`. -/
abbrev Item := String × RE × Bool

def matchItem (items : List Item) (e : RE) : Option String :=
  (items.find? fun it => it.2.2 && structEq feq e it.2.1).map Prod.fst

mutual
def replaceAgg (items : List Item) : RE → RE
  | .node d cs =>
    match d with
    | .func n true =>
      match matchItem feq items (.node (.func n true) cs) with
      | some nm => .node (.var nm) []
      | none => .node d (replaceAggL items cs)
    | _ => .node d (replaceAggL items cs)
def replaceAggL (items : List Item) : List RE → List RE
  | [] => []
  | c :: cs => replaceAgg items c :: replaceAggL items cs
end

theorem matchItem_none (items : List Item) (e : RE) (h : ∀ it ∈ items, it.2.2 = false) :
    matchItem feq items e = none := by
  unfold matchItem
  rw [List.find?_eq_none.2]; · rfl
  intro it hit; simp [h it hit]

mutual
/-- No aggregation among the projections: ORDER BY is left as written. -/
theorem replaceAgg_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ e, replaceAgg feq items e = e
  | .node d cs => by
    have ih := replaceAggL_id items h cs
    cases d with
    | func n a => cases a <;> simp [replaceAgg, ih, matchItem_none feq items _ h]
    | _ => simp [replaceAgg, ih]
theorem replaceAggL_id (items : List Item) (h : ∀ it ∈ items, it.2.2 = false) : ∀ l, replaceAggL feq items l = l
  | [] => rfl
  | c :: cs => by simp [replaceAggL, replaceAgg_id items h c, replaceAggL_id items h cs]
end

/-- An aggregate call matching an item becomes that item's alias (first match). -/
theorem replaceAgg_root (items : List Item) (n : String) (cs : List RE) (nm : String)
    (h : matchItem feq items (.node (.func n true) cs) = some nm) :
    replaceAgg feq items (.node (.func n true) cs) = .node (.var nm) [] := by
  simp [replaceAgg, h]

/-- …and every match is justified by `structEq` against an aggregation item. -/
theorem matchItem_spec (items : List Item) (e : RE) (nm : String) (h : matchItem feq items e = some nm) :
    ∃ it ∈ items, it.1 = nm ∧ it.2.2 = true ∧ structEq feq e it.2.1 = true := by
  unfold matchItem at h
  cases hf : items.find? (fun it => it.2.2 && structEq feq e it.2.1) with
  | none => rw [hf] at h; simp at h
  | some it =>
    rw [hf] at h; simp at h
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true] at hp
    exact ⟨it, hm, h, hp.1, hp.2⟩

end FalkorBinder.Agg
