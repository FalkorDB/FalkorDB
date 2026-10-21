import Pr3437Review.ExactProp
import Columnar.CoreBridge

/-!
# PR #3437's `VectorEval` (a739b990f) as columnar arms, and its `Lawful` instance

The PR changes two arms of `VectorEval` (vector_expr.rs at a739b990f):
* `eval_property` (:336-353): a `NodeIds` child is fetched with
  `materialize_node_property_exact` (`ExactProp.exactPR`), anything else as before;
* the function arm's guard gains `!counts_comprehension(node)` (:255-257), so
  `size(<list comprehension>)` is evaluated per row.

`armsPR` is columnar's `CoreBridge.armsOf` with those two arms replaced, over a runtime whose
bulk node read is the attribute store `st` (`withStore`). **`evalPropertyPR_eq`**: the PR's
property column *is* main's (`ExactProp.exact_eq_main`, and the two models of
`classify_exact_column` agree, `toECx_eq`). **`lawfulPR`**: `VecEval.Lawful` holds for the
PR's arms — so `VecEval.eval_sound` holds for the PR's `VectorEval::eval`.
-/

namespace Pr3437.PRArms
open FalkorCore (V)
open FalkorExpr.Eval (E Ev Compound Fn evalNode)
open Columnar (Batch Column EC FloatModel)
open Columnar.CoreOps (RT coreOps)
open Columnar.VecEval (Arms Lawful Sound SoundEv)
open Columnar.CoreBridge (armsOf rowSem lawful PropReads perRow evalProperty propertyValues toEC)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- The node attribute store behind the bulk reads, per attribute name (`ExactProp`). -/
structure NodeStore (F : Type) where
  entry : String → Nat → Option ExactProp.Packed
  unpackOther : ExactProp.Packed → V F
  attrIdx : String → Option Nat
  deletedEmpty : Bool
  noPendingAttrs : Bool
  /-- the per-id value of the deleted / pending path (runtime.rs:1611-1626) -/
  slowAt : String → Nat → V F

/-- `materialize_node_property_values` (runtime.rs:1595). -/
def NodeStore.matValues (st : NodeStore F) (ids : List Nat) (a : String) : List (V F) :=
  ExactProp.matValues (st.entry a) st.unpackOther (st.attrIdx a) st.deletedEmpty st.noPendingAttrs
    (fun keys => keys.map (st.slowAt a)) ids

/-- `materialize_node_property_exact` (runtime.rs:1566); `0.0` is `ofInt 0`. -/
def NodeStore.exact (st : NodeStore F) (ids : List Nat) (a : String) : ExactProp.Column F × ExactProp.Bits :=
  ExactProp.exactPR (st.entry a) st.unpackOther (FloatModel.ofInt 0) (st.attrIdx a) st.deletedEmpty
    st.noPendingAttrs (fun keys => keys.map (st.slowAt a)) ids

/-- The runtime whose bulk node read is `st`. -/
def withStore (rt : RT F) (st : NodeStore F) : RT F := { rt with matNode := st.matValues }

/-- The `match col` of `eval_property` on the exact fetch's `(Column, NullBitmap)`. -/
def toECx (len : Nat) : ExactProp.Column F × ExactProp.Bits → EC F
  | (.ints xs, bm) => .ints xs ((List.range len).map bm)
  | (.floats fs, bm) => .floats fs ((List.range len).map bm)
  | (.values vs, _) => .values vs

theorem mapM_toIntN : ∀ vs : List (V F), vs.mapM ExactProp.toIntN = Columnar.intsOf true vs
  | [] => rfl
  | v :: vs => by
    cases v <;> simp [ExactProp.toIntN, Columnar.intsOf, mapM_toIntN vs, Option.map]
    all_goals cases Columnar.intsOf true vs <;> rfl

theorem mapM_toFloatN : ∀ vs : List (V F),
    vs.mapM (ExactProp.toFloatN (FloatModel.ofInt 0)) = Columnar.floatsOf true .pure vs
  | [] => rfl
  | v :: vs => by
    cases v <;> simp [ExactProp.toFloatN, Columnar.floatsOf, mapM_toFloatN vs, Option.map]
    all_goals cases Columnar.floatsOf true .pure vs <;> rfl

theorem bm_ofValues (vs : List (V F)) :
    (List.range vs.length).map (fun j => match vs[j]? with | some v => ExactProp.isNullB v | none => false) =
      Columnar.Bits.ofValues vs := by
  apply List.ext_getElem
  · simp [Columnar.Bits.ofValues]
  · intro i h1 h2
    simp at h1
    simp only [List.getElem_map, List.getElem_range, Columnar.Bits.ofValues, List.getElem?_eq_getElem h1]
    cases vs[i] <;> rfl

/-- The two models of `classify_exact_column` give the same `ExprColumn`. -/
theorem toECx_eq (vs : List (V F)) :
    toECx vs.length (ExactProp.classifyExact (FloatModel.ofInt 0) vs) = toEC vs.length (Columnar.classifyExact vs) := by
  simp only [ExactProp.classifyExact, Columnar.classifyExact, Columnar.classifyNumeric, mapM_toIntN,
    mapM_toFloatN]
  cases Columnar.intsOf true vs with
  | some is => simp only [toECx, toEC]; exact congrArg _ (bm_ofValues vs)
  | none =>
    simp only [ne_eq, reduceCtorEq, not_false_eq_true, ite_true]
    cases Columnar.floatsOf true .pure vs with
    | some fs => simp only [toECx, toEC]; exact congrArg _ (bm_ofValues vs)
    | none => simp [toECx, toEC]

/-- The PR's `eval_property` (vector_expr.rs:336-353 at a739b990f). -/
def evalPropertyPR (rt : RT F) (st : NodeStore F) (col : Nat → Column F) (attr : String) (ch : E (V F))
    (rows : List Nat) : Option (EC F) :=
  match ch with
  | .var id _ =>
    match col id with
    | .nodeIds ids => some (toECx rows.length (st.exact (rows.map fun r => ids.getD r 0) attr))
    | _ => evalProperty rt col attr ch rows
  | _ => evalProperty rt col attr ch rows

/-- **The PR's property column is main's**, for every child, column and store. -/
theorem evalPropertyPR_eq (rt : RT F) (st : NodeStore F) (col : Nat → Column F) (attr : String)
    (ch : E (V F)) (rows : List Nat) :
    evalPropertyPR (withStore rt st) st col attr ch rows = evalProperty (withStore rt st) col attr ch rows := by
  unfold evalPropertyPR
  split
  · rename_i id n
    split
    · rename_i ids hcol
      simp only [evalProperty, propertyValues, hcol, Option.map_some, withStore, NodeStore.exact,
        NodeStore.matValues]
      rw [ExactProp.exact_eq_main]
      have hl : (ExactProp.matValues (st.entry attr) st.unpackOther (st.attrIdx attr) st.deletedEmpty
          st.noPendingAttrs (fun keys => keys.map (st.slowAt attr)) (rows.map fun r => ids.getD r 0)).length
            = rows.length := by
        simp only [ExactProp.matValues, ExactProp.batchValues]
        split
        · split <;> simp
        · simp
      rw [← hl, toECx_eq]
    · rfl
  · rfl

/-- `counts_comprehension(node)` (eval.rs:315-320): `size` over a single `ListComprehension`
child (`isLC t`: catch-all tag `t` is `ListComprehension`). -/
def countsC (isLC : Nat → Bool) (f : Fn (V F)) : List (E (V F)) → Bool
  | [.compound t _] => isLC t && f.name == "size"
  | _ => false

section
variable (rt : RT F) (c : Compound (V F)) (b : Batch F) (st : NodeStore F) (isLC : Nat → Bool)

/-- **The PR's `VectorEval` over batch `b`** (vector_expr.rs:211-283 at a739b990f): a counted
comprehension fails the function arm's guard and is evaluated per row (a `size` call never
takes the `hasLabels` branch), and `NodeIds` properties are fetched exactly. -/
def armsPR : Arms (V F) (EC F) :=
  { armsOf (withStore rt st) c b with
    evalProperty := evalPropertyPR (withStore rt st) st b.column
    evalFunction := fun ev f cs rows =>
      if countsC isLC f cs then perRow (withStore rt st) c b (.func f cs) rows
      else Columnar.FuncLane.evalFunction ev f cs rows }

/-- **`Lawful` for the PR's arms** (so `eval_sound` holds for the PR's `VectorEval::eval`). -/
theorem lawfulPR (hwf : b.WF) (hp : PropReads (withStore rt st)) :
    Lawful (armsPR rt c b st isLC) (rowSem (withStore rt st) c b) (fun id => id < b.cols.length)
      (fun r => r ∈ b.active) (Columnar.FuncLane.HL (withStore rt st)) :=
  let L := lawful (withStore rt st) c b hwf hp
  { L with
    prop := fun attr ch rows col hs hr h => by
      simp only [armsPR, evalPropertyPR_eq] at h
      exact L.prop attr ch rows col hs hr h
    func := fun ev f cs rows col hg hk hr h => by
      simp only [armsPR] at h
      split at h
      · exact L.perRow _ rows col h
      · exact L.func ev f cs rows col hg hk hr h }

/-- `size(<comprehension>)` goes per row on the PR's `VectorEval`. -/
theorem counted_goes_per_row (f : Fn (V F)) (t : Nat) (cs : List (E (V F))) (rows : List Nat)
    (hk : isLC t = true) (hn : f.name = "size") (hg : Columnar.VecEval.fnGuard f [.compound t cs] = true) :
    Columnar.VecEval.eval (armsPR rt c b st isLC) (.func f [.compound t cs]) rows =
      perRow (withStore rt st) c b (.func f [.compound t cs]) rows := by
  unfold Columnar.VecEval.eval
  simp [armsPR, countsC, hk, hn, hg]
end

end Pr3437.PRArms
