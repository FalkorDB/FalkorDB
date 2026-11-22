import FalkorNativeIndex.Rows
import PendingCommit.AttrInv
set_option linter.unusedSimpArgs false

/-!
# The staging functions' caller premise, discharged from proofs/pending_commit

`node_attrs_set`, `new_node_attrs` and `edge_attrs_set_with` (`graph_writes.rs`) take an
`FxHashMap<u64, Vec<(u16, Value)>>`. Their `*_hit` theorems (`Writes.lean`) need each vector's attribute ids
unique (`hw`) and the map's entity ids unique (`hkeys`, which a hash map's type gives). The vectors are
`Pending`'s staged attribute lists, whose keys `PendingCommit.PA.inv_huniq` proves unique (they are kept
sorted by `set_node_attribute`, and `CreateOp`'s lists are sorted by construction, `AttrCallers.placed_sorted`).
-/

namespace NI
open PendingCommit PendingCommit.PA

/-- The pending map as the argument of the staging functions (values translated by `f`; keys unchanged). -/
def toAttrs (f : Val → Value) (m : FMap (List Entry)) : List (Nat × List (Nat × Value)) :=
  m.map (fun p => (p.1, p.2.map (fun e => (e.1, f e.2))))

theorem fget_of_mem_nodup {V : Type} : ∀ (m : FMap V), (m.map (·.1)).Nodup → ∀ p ∈ m, fget m p.1 = some p.2
  | [], _, p, hp => by simp at hp
  | q :: m, hk, p, hp => by
    simp only [List.map_cons, List.nodup_cons] at hk
    rcases List.mem_cons.1 hp with rfl | hp
    · simp [fget]
    · have hne : q.1 ≠ p.1 := fun h => hk.1 (h ▸ List.mem_map_of_mem hp)
      simp [fget, hne, fget_of_mem_nodup m hk.2 p hp]

/-- **The premises of the `*_hit` theorems hold for a pending attribute map**: its entity ids are unique (a
    hash map's keys) and every staged list's attribute ids are unique (`inv_huniq`). -/
theorem pending_attrs_premises (f : Val → Value) (m : FMap (List Entry))
    (huniq : ∀ id l, fget m id = some l → (l.map Prod.fst).Nodup) (hk : (m.map (·.1)).Nodup) :
    ((toAttrs f m).map (·.1)).Nodup ∧ ∀ iw ∈ toAttrs f m, (iw.2.map (·.1)).Nodup := by
  refine ⟨by simpa [toAttrs, Function.comp_def] using hk, ?_⟩
  intro iw hiw
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hiw
  have := huniq p.1 p.2 (fget_of_mem_nodup m hk p hp)
  simpa [Function.comp_def] using this

theorem new_attrs_unique (f : Val → Value) {c : List Nat} {nw ex : FMap (List Entry)} (h : PairInv c nw ex)
    (hk : (nw.map (·.1)).Nodup) :
    ((toAttrs f nw).map (·.1)).Nodup ∧ ∀ iw ∈ toAttrs f nw, (iw.2.map (·.1)).Nodup :=
  pending_attrs_premises f nw (fun id l hl => inv_huniq h id l (Or.inl hl)) hk

theorem existing_attrs_unique (f : Val → Value) {c : List Nat} {nw ex : FMap (List Entry)} (h : PairInv c nw ex)
    (hk : (ex.map (·.1)).Nodup) :
    ((toAttrs f ex).map (·.1)).Nodup ∧ ∀ iw ∈ toAttrs f ex, (iw.2.map (·.1)).Nodup :=
  pending_attrs_premises f ex (fun id l hl => inv_huniq h id l (Or.inr hl)) hk

/-- `node_attrs_set` on `Pending`'s existing-node attributes: the SET replaces the entry. -/
theorem nodeAttrsSet_hit_pending (f : Val → Value) {c : List Nat} {nw ex : FMap (List Entry)}
    (h : PairInv c nw ex) (hk : (ex.map (·.1)).Nodup) (g : G) (hn : g.attrNames.Nodup)
    (hids : ∀ iw ∈ toAttrs f ex, iw.1 < W) (L A : String) (hcol : g.idx.hasColumn .node L A = true)
    (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value)) (aid : Nat) (nv : Value)
    (hmem : (id0, written) ∈ toAttrs f ex) (hamem : (aid, nv) ∈ written) (hA : g.attrName aid = some A)
    (hL : L ∈ (g.nodeLabels id0).map g.labelName) :
    lastOpS (colLog (L, A) (nodeAttrsSet g (toAttrs f ex)).changes) (enc kk id0) =
      outcome (g.nodeAttr id0 aid) (some nv) kk :=
  nodeAttrsSet_hit g hn _ (existing_attrs_unique f h hk).1 (existing_attrs_unique f h hk).2 hids L A hcol kk id0
    hid0 written aid nv hmem hamem hA hL

/-- `new_node_attrs` on `Pending`'s new-node attributes. -/
theorem newNodeAttrs_hit_pending (f : Val → Value) {c : List Nat} {nw ex : FMap (List Entry)}
    (h : PairInv c nw ex) (hk : (nw.map (·.1)).Nodup) (g : G) (hn : g.attrNames.Nodup)
    (newLabels : List (Nat × List Nat)) (hids : ∀ iw ∈ toAttrs f nw, iw.1 < W) (L A : String)
    (hcol : g.idx.hasColumn .node L A = true) (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value))
    (aid : Nat) (nv : Value) (lids : List Nat) (hmem : (id0, written) ∈ toAttrs f nw) (hamem : (aid, nv) ∈ written)
    (hA : g.attrName aid = some A) (hlab : hmGet newLabels id0 = some lids) (hL : L ∈ lids.map g.labelName) :
    lastOpS (colLog (L, A) (newNodeAttrs g (toAttrs f nw) newLabels).changes) (enc kk id0) =
      outcome none (some nv) kk :=
  newNodeAttrs_hit g hn _ newLabels (new_attrs_unique f h hk).1 (new_attrs_unique f h hk).2 hids L A hcol kk id0
    hid0 written aid nv lids hmem hamem hA hlab hL

/-- `edge_attrs_set_with` on `Pending`'s existing-relationship attributes. -/
theorem edgeAttrsSetWith_hit_pending (f : Val → Value) {c : List Nat} {nw ex : FMap (List Entry)}
    (h : PairInv c nw ex) (hk : (ex.map (·.1)).Nodup) (g : G) (hn : g.attrNames.Nodup) (typeId : Option Nat)
    (hids : ∀ iw ∈ toAttrs f ex, iw.1 < W) (T A : String) (hcol : g.idx.hasColumn .relationship T A = true)
    (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value)) (aid : Nat) (nv : Value)
    (hmem : (id0, written) ∈ toAttrs f ex) (hamem : (aid, nv) ∈ written) (hA : g.attrName aid = some A)
    (hT : edgeTypeOf g typeId id0 = T) :
    lastOpS (colLog (T, A) (edgeAttrsSetWith g (toAttrs f ex) typeId).changes) (enc kk id0) =
      outcome (g.relAttr id0 aid) (some nv) kk := by
  have := edgeAttrsSetWith_hit g hn _ typeId (existing_attrs_unique f h hk).1 (existing_attrs_unique f h hk).2 hids T A
    hcol kk id0 hid0 written aid nv hmem hamem hA hT
  exact this

end NI
