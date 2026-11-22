import PendingCommit.AttrInv
import RuntimeCore.Discharge
/-!
# runtime_core's four CONDITIONAL accessors, discharged against `Pending`

proofs/runtime_core proves `Runtime::get_node_attribute`, `get_node_attribute_no_delete_check`,
`get_node_attrs` (`Accessors.attrs_agree`, `RuntimeCore.Discharge.attrs_agree_discharged`) and
`get_relationship_attrs` (`RelAccessors.relAttrs_agree`, `RuntimeCore.Discharge.relAttrs_agree_discharged`)
under two facts about `Pending` it cannot prove (it does not import this project):

* `hdisj` — an id is in at most one of `new_*_attrs` / `existing_*_attrs`;
* `huniq` — the keys of every staged list are unique.

Here the world `w : RuntimeCore.Accessors.W` is tied to the pending maps by `Corr`: its `newAttrs`/`exAttrs`
are the maps' lists with the values mapped through `f` (runtime_core's `Val` is a view of the
store's). Under `PA.PairInv` (kept by every staging operation, `AttrInv.lean`) both facts hold
(`corr_hdisj`, `corr_huniq`), so the runtime_core theorems apply with no extra hypothesis about
`Pending` (`*_pending`). What `PairInv` itself assumes of its callers is stated in `AttrInv.lean`
(`created_inv`'s premise, `stageAll_inv`'s premise) and discharged in `AttrCallers.lean`.
-/
namespace PendingCommit.PA
open PendingCommit

/-- `w`'s pending attribute maps are the pending state's, values viewed through `f`. -/
def Corr (w : RuntimeCore.Accessors.W) (f : Val → RuntimeCore.Accessors.Val) (nw ex : FMap (List Entry)) : Prop :=
  ∀ id, w.newAttrs id = (fget nw id).map (List.map (fun e => (e.1, f e.2))) ∧
        w.exAttrs id = (fget ex id).map (List.map (fun e => (e.1, f e.2)))

theorem corr_hdisj {w : RuntimeCore.Accessors.W} {f : Val → RuntimeCore.Accessors.Val} {c : List Nat}
    {nw ex : FMap (List Entry)} (hc : Corr w f nw ex) (h : PairInv c nw ex) :
    ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none := by
  intro id
  rcases h.disj id with h1 | h1
  · left; rw [(hc id).1, h1]; rfl
  · right; rw [(hc id).2, h1]; rfl

theorem corr_huniq {w : RuntimeCore.Accessors.W} {f : Val → RuntimeCore.Accessors.Val} {c : List Nat}
    {nw ex : FMap (List Entry)} (hc : Corr w f nw ex) (h : PairInv c nw ex) :
    ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup := by
  intro id l hl
  have key : ∀ l0 : List Entry, l = l0.map (fun e => (e.1, f e.2)) →
      (fget nw id = some l0 ∨ fget ex id = some l0) → (l.map Prod.fst).Nodup := by
    intro l0 hl0 hh
    have := inv_huniq h id l0 hh
    rw [hl0]; simpa [List.map_map, Function.comp_def] using this
  rcases hl with hn | he
  · rw [(hc id).1] at hn
    cases hf : fget nw id with
    | none => simp [hf] at hn
    | some l0 =>
      simp [hf] at hn
      exact key l0 hn.symm (Or.inl hf)
  · rw [(hc id).2] at he
    cases hf : fget ex id with
    | none => simp [hf] at he
    | some l0 =>
      simp [hf] at he
      exact key l0 he.symm (Or.inr hf)

open RuntimeCore.Accessors RuntimeCore.RelAccessors in
/-- `get_node_attrs` / `get_node_attribute` / `_no_delete_check` (runtime_core `Accessors.attrs_agree`)
with `hdisj`/`huniq` discharged by the pending invariant. -/
theorem attrs_agree_pending (w : RuntimeCore.Accessors.W) (f : Val → RuntimeCore.Accessors.Val) (c : List Nat) (nw ex : FMap (List Entry))
    (hcorr : Corr w f nw ex) (hinv : PairInv c nw ex)
    (hn : w.nameOk) (hd : w.deletedOk) (ha : w.attrsOk)
    (hc : ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id))
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getNodeAttrs w id)) = Val.norm (getNodeAttribute w id name) :=
  attrs_agree w hn hd ha hc (corr_hdisj hcorr hinv) (corr_huniq hcorr hinv) id name

open RuntimeCore.Accessors RuntimeCore.RelAccessors in
/-- `get_relationship_attrs` (runtime_core `RelAccessors.relAttrs_agree`). -/
theorem relAttrs_agree_pending (w : RuntimeCore.Accessors.W) (f : Val → RuntimeCore.Accessors.Val) (c : List Nat) (nw ex : FMap (List Entry))
    (hcorr : Corr w f nw ex) (hinv : PairInv c nw ex) (byName : Nat → String → Option RuntimeCore.Accessors.Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (hn : w.nameOk) (hd : w.deletedOk) (ha : w.attrsOk)
    (hc : ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id))
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getRelAttrs w id)) = Val.norm (getRelAttribute w byName id name) :=
  relAttrs_agree w byName hName hn hd ha hc (corr_hdisj hcorr hinv) (corr_huniq hcorr hinv) id name

open RuntimeCore.Accessors RuntimeCore.RelAccessors in
/-- The fully discharged node form: `nameOk`/`hc` come from the name-table and store models. -/
theorem attrs_agree_discharged_pending (w : RuntimeCore.Accessors.W) (f : Val → RuntimeCore.Accessors.Val) (c : List Nat)
    (nw ex : FMap (List Entry)) (hcorr : Corr w f nw ex) (hinv : PairInv c nw ex)
    {nm : List String} {st : RuntimeCore.Discharge.Store} (h : RuntimeCore.Discharge.Reach nm) (hb : RuntimeCore.Discharge.Backed w nm st) (hd : w.deletedOk)
    (ha : w.attrsOk) (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getNodeAttrs w id)) = Val.norm (getNodeAttribute w id name) :=
  RuntimeCore.Discharge.attrs_agree_discharged w h hb hd ha (corr_hdisj hcorr hinv) (corr_huniq hcorr hinv) id name

open RuntimeCore.Accessors RuntimeCore.RelAccessors in
/-- The fully discharged relationship form. -/
theorem relAttrs_agree_discharged_pending (w : RuntimeCore.Accessors.W) (f : Val → RuntimeCore.Accessors.Val) (c : List Nat)
    (nw ex : FMap (List Entry)) (hcorr : Corr w f nw ex) (hinv : PairInv c nw ex)
    {nm : List String} {st : RuntimeCore.Discharge.Store} (h : RuntimeCore.Discharge.Reach nm) (hb : RuntimeCore.Discharge.Backed w nm st) (hd : w.deletedOk)
    (ha : w.attrsOk) (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getRelAttrs w id)) =
      Val.norm (getRelAttribute w (RuntimeCore.Discharge.attrByName nm st) id name) :=
  RuntimeCore.Discharge.relAttrs_agree_discharged w h hb hd ha (corr_hdisj hcorr hinv) (corr_huniq hcorr hinv) id name

end PendingCommit.PA
