import Pr3437Review.PRArms
import Columnar.RuntimeReads

/-!
# The runtime state behind the PR's exact node fetch

proofs/columnar `RuntimeReads` models the runtime's attribute reads (per row and bulk) over a
state `St` and proves the bulk fetch is the per-row read (`matNode_eq`, `matRel_eq`,
`propReads`). The PR's `materialize_node_property_exact` (`ExactProp.exactPR`) reads the node
store as packed entries (`PackedAttr`, attribute_store.rs:257-278); `PSt` is that state with
the node store given packed, `toSt` the same state as `RuntimeReads` sees it
(`get_attr_by_idx` = `data.get(key)?.get(idx)` unpacked, `Block::unpack`), and `nodeStore` the
`PRArms.NodeStore` the exact fetch reads. **`nodeStore_matValues`**: its
`materialize_node_property_values` is the state's; **`withStore_rtOf`**: so the runtime the
PR's arms see is `RuntimeReads.rtOf` of the state, for which `PropReads` is proved.
-/

namespace Pr3437.Runtime
open FalkorCore (V)
open Columnar (FloatModel)
open Columnar.CoreOps (RT)
open Columnar.RuntimeReads (St matNode hasNodeAttrs pendGetN byIdxN rtOf)
open Pr3437.PRArms (NodeStore withStore)

variable {F : Type}

/-- The state with the committed node store as packed entries. -/
structure PSt (F : Type) where
  st : St F
  packedN : Nat → Nat → Option ExactProp.Packed
  unpackOther : ExactProp.Packed → V F

/-- The state as `RuntimeReads` reads it: `get_attr_by_idx` unpacks the packed entry. -/
def PSt.toSt (p : PSt F) : St F :=
  { p.st with byIdxN := fun k i => (p.packedN k i).map (ExactProp.unpack p.unpackOther) }

variable (p : PSt F)

/-- What `materialize_node_property_exact` reads (runtime.rs:1566-1587). -/
def nodeStore : NodeStore F where
  entry a k := match p.st.attrId a with
    | some i => p.packedN k i
    | none => none
  unpackOther := p.unpackOther
  attrIdx := p.st.attrId
  deletedEmpty := p.st.delNodes.isEmpty
  noPendingAttrs := !hasNodeAttrs p.toSt
  slowAt a id :=
    match p.st.delNodes.lookup id with
    | none => ((p.st.attrId a).bind fun i => match pendGetN p.toSt id i with
        | some v => some v
        | none => byIdxN p.toSt id i).getD .null
    | some attrs => (attrs.lookup a).getD .null

/-- The exact fetch's `materialize_node_property_values` is the state's. -/
theorem nodeStore_matValues (ids : List Nat) (a : String) :
    (nodeStore p).matValues ids a = matNode p.toSt ids a := by
  simp only [NodeStore.matValues, ExactProp.matValues, nodeStore, matNode]
  by_cases hc : (p.st.delNodes.isEmpty && !hasNodeAttrs p.toSt) = true
  · have hc' : (p.toSt.delNodes.isEmpty && !hasNodeAttrs p.toSt) = true := hc
    simp only [hc, hc', ite_true]
    cases hi : p.st.attrId a with
    | some i =>
      have hi' : p.toSt.attrId a = some i := hi
      simp [ExactProp.batchValues, hi, byIdxN, PSt.toSt]
    | none =>
      have hi' : p.toSt.attrId a = none := hi
      simp only [hi']
      apply List.ext_getElem
      · simp
      · intro i h1 h2; simp
  · have hc' : ¬ (p.toSt.delNodes.isEmpty && !hasNodeAttrs p.toSt) = true := hc
    simp only [hc, hc', Bool.false_eq_true, ite_false]
    rfl

/-- `withStore` over the state's node store is the state's runtime. -/
theorem withStore_rtOf (rt : RT F) : withStore (rtOf p.toSt rt) (nodeStore p) = rtOf p.toSt rt := by
  simp only [withStore, rtOf]
  congr 1
  funext ids a
  exact nodeStore_matValues p ids a

theorem propReads_withStore (rt : RT F) : Columnar.CoreBridge.PropReads (withStore (rtOf p.toSt rt) (nodeStore p)) := by
  rw [withStore_rtOf]; exact Columnar.RuntimeReads.propReads p.toSt rt

end Pr3437.Runtime
