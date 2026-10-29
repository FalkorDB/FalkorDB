import FalkorEffectsEmitApply.Emit
import FalkorEffectsEmitApply.Liveness
/-!
# Emit then apply, with cancellation: two open replica divergences (#3447, #3448)

`Emit.forEachRecord` (`for_each_record`, emit.rs:323-375 on b84473a7f) and
`Live.applyAll` (the checked arms of `apply_record`) were proven separately;
nothing related what the master emits to what a replica *at the master's
pre-segment state* accepts. Two properties, both refuted on main:

* **`RelIdsCovered`** (#3447) — every relationship id the master allocated in
  the segment (created or cancelled) is named by some `CREATE_EDGE`, so the
  replica's relationship `IdSpace` ends with the same contiguous range
  (`IdSpace::verify`, id_space.rs:737-790, refuses a hole above the entry
  bound). `digest_cancelled` (emit.rs:455) skips a cancelled edge whose type has
  no id, and a type whose every edge was cascaded away is never registered
  (`Pending::commit` registers only `created_rels_by_type` keys via
  `create_relationships_bulk` → `get_type_id_mut`, pending.rs:1236-1249,
  graph.rs:2610/:1265; proofs/pending_commit `bug3447_cascade_drops_type_key`).
  `bug3447_cancelled_edge_id_gap`: live repro `CREATE ()-[:EXISTING]->()` then
  `CREATE (a)-[r1:NEW_TYPE]->(b), (c)-[r2:EXISTING]->(d) DETACH DELETE a`; the
  payload carries `CreateEdge{ids:[2]}` and nothing for 1, the replica's verify
  refuses (`verifyRange`), the master's passes. `relIdsCovered_of_registered`:
  with every created and cancelled type registered, the property holds for every
  `Pending`.
* **`ReplayApplies`** (#3448) — the emitted stream, translated to apply records
  (`toLive`), applies on the replica. `digest_labels(&p.remove_labels, None, …)`
  (emit.rs:370) runs before `digest_cancelled` (:376) and does not skip created
  or cancelled nodes; `delete_pending_node` keeps the node's `remove_labels`
  (proofs/pending_commit `bug3448_cancelled_node_keeps_remL`); the
  `RemoveLabels` arm calls `require_live` (apply.rs:359-360).
  `bug3448_removeLabels_before_create_refused`: `CREATE (n:Foo) REMOVE n:Foo
  DELETE n` emits `AddLabel Foo, RemoveLabels [0], CreateNode [0], DeleteNode [0]`,
  refused with `NotLive node 0`; without the stale entry the same segment applies
  (`agree3448_without_stale_remL`).
-/
namespace FalkorEA.CancelGap
open FalkorEA

/-! ## #3447: relationship ids -/

/-- `IdSpace::verify` (id_space.rs:737-790), the hole half: the ids taken at or
above `entry` are exactly `[entry, entry + k)`. -/
def verifyRange (entry : Nat) (taken : List Nat) : Bool :=
  let c := taken.filter (entry ≤ ·)
  c.isEmpty || (c.min? == some entry && c.max? == some (entry + c.length - 1))

/-- The relationship ids the stream's `CREATE_EDGE` records name (what the
replica's `create_relationships_bulk` takes). -/
def edgeIds : List Rec → List Nat
  | [] => []
  | .createEdge ids _ :: rs => ids ++ edgeIds rs
  | _ :: rs => edgeIds rs

/-- Relationship ids the master allocated this segment: created and cancelled. -/
def allocated (p : Pending) : List Nat := p.createdRelTypes.flatMap (·.2) ++ p.cancelledRels.map (·.2)

/-- **Property `RelIdsCovered`**. -/
def RelIdsCovered (p : Pending) (g : Schema) : Prop := ∀ i ∈ allocated p, i ∈ edgeIds (forEachRecord p g)

/-- The relationship-type dictionary after `Pending::commit`: each
`created_rels_by_type` key is get-or-created (pending.rs:1236-1249); cancelled
types are not looked at. -/
def typesAfterCommit (base : List Name) (p : Pending) : List Name :=
  p.createdRelTypes.foldl (fun ts q => if q.1 ∈ ts then ts else ts ++ [q.1]) base

/-- `CREATE (a)-[r1:NEW_TYPE]->(b), (c)-[r2:EXISTING]->(d) DETACH DELETE a` after
`CREATE ()-[:EXISTING]->()` (nodes 0,1 and edge 0 committed): `a` = 2 is
cancelled, `b, c, d` = 3, 4, 5 created, `r1` = 1 cancelled with `a`, `r2` = 2
created. -/
def pGap : Pending :=
  { createdNodes := [3, 4, 5], createdRelTypes := [("EXISTING", [2])], deletedNodes := [], deletedRels := [],
    newNodesAttrs := [], existingNodesAttrs := [], newRelsAttrs := [], existingRelsAttrs := [],
    setLabels := [], removeLabels := [], cancelledNodes := [2], cancelledRels := [("NEW_TYPE", 1)],
    baseline := ⟨[], ["EXISTING"], []⟩ }

def gGap : Schema := ⟨[], typesAfterCommit ["EXISTING"] pGap, []⟩

/-- **#3447**: `NEW_TYPE` is never registered, so the cancelled edge 1 is not
in the payload; the master's relationship id space `{0, 1, 2}` (entry bound 1)
verifies, the replica's `{0} ∪ {2}` has a hole and is refused
(live: "effects buffer allocated relationship ids 1..=2 but created only 1 of
them … forced full resync"; C stays in sync). -/
theorem bug3447_cancelled_edge_id_gap :
    gGap.types = ["EXISTING"] ∧
    forEachRecord pGap gGap = [.createNode [3, 4, 5] [], .createEdge [2] 0, .createNode [2] [], .deleteNode [2] []] ∧
    ¬ RelIdsCovered pGap gGap ∧
    verifyRange 1 ([0] ++ allocated pGap) = true ∧
    verifyRange 1 ([0] ++ edgeIds (forEachRecord pGap gGap)) = false := by
  refine ⟨by decide, by decide, fun h => ?_, by decide, by decide⟩
  have := h 1 (by decide)
  revert this; decide

theorem mem_edgeIds_append (a b : List Rec) (i : Nat) : i ∈ edgeIds (a ++ b) ↔ i ∈ edgeIds a ∨ i ∈ edgeIds b := by
  induction a with
  | nil => simp [edgeIds]
  | cons r rs ih =>
    cases r <;> simp [edgeIds, ih, or_assoc]

theorem mem_edgeIds_of (rs : List Rec) (ids : List Nat) (r : Nat) (h : Rec.createEdge ids r ∈ rs) (i : Nat)
    (hi : i ∈ ids) : i ∈ edgeIds rs := by
  induction rs with
  | nil => cases h
  | cons x xs ih =>
    rcases List.mem_cons.mp h with rfl | h
    · simp [edgeIds, hi]
    · have := ih h
      cases x <;> simp [edgeIds, this]

/-- **`relIdsCovered_of_registered`**: when every created and every cancelled
relationship's type has an id at emit time, every allocated id reaches the
replica. A fix that registers cascaded types (or `CreateOp::resolve_pattern`
registering types as it does labels) discharges the hypotheses. -/
theorem relIdsCovered_of_registered (p : Pending) (g : Schema)
    (hc : ∀ q ∈ p.createdRelTypes, (idxOf g.types q.1).isSome)
    (hx : ∀ q ∈ p.cancelledRels, (idxOf g.types q.1).isSome) : RelIdsCovered p g := by
  intro i hi
  rcases List.mem_append.mp hi with hi | hi
  · -- a created edge: its group is non-empty, so `digest_created_edges` emits it
    obtain ⟨q, hq, hiq⟩ := List.mem_flatMap.mp hi
    obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp (hc q hq)
    have hmem : Rec.createEdge q.2 r ∈ (p.createdRelTypes.filter (fun q => !q.2.isEmpty)).filterMap
        (fun (t, ids) => (idxOf g.types t).map (fun r => Rec.createEdge ids r)) := by
      refine List.mem_filterMap.mpr ⟨q, List.mem_filter.mpr ⟨hq, ?_⟩, ?_⟩
      · cases h2 : q.2 with
        | nil => rw [h2] at hiq; cases hiq
        | cons _ _ => rfl
      · obtain ⟨t, ids⟩ := q; simp only at hr ⊢; rw [hr]; rfl
    refine mem_edgeIds_of _ q.2 r ?_ i hiq
    unfold forEachRecord
    simp only [List.mem_append]
    simp [hmem]
  · -- a cancelled edge: `digest_cancelled` pairs it
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hi
    obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp (hx q hq)
    have hne : (p.cancelledNodes.isEmpty && p.cancelledRels.isEmpty) = false := by
      cases h : p.cancelledRels with
      | nil => rw [h] at hq; cases hq
      | cons _ _ => simp
    have hmem : Rec.createEdge [q.2] r ∈ digestCancelled p g := by
      unfold digestCancelled
      rw [hne]
      simp only [Bool.false_eq_true, ite_false, List.mem_append]
      have : Rec.createEdge [q.2] r ∈ (p.cancelledRels.filterMap
          (fun (t, i) => (idxOf g.types t).map (fun r => (r, i)))).map (fun (r, i) => Rec.createEdge [i] r) := by
        refine List.mem_map.mpr ⟨(r, q.2), List.mem_filterMap.mpr ⟨q, hq, ?_⟩, rfl⟩
        obtain ⟨t, j⟩ := q; simp only at hr ⊢; rw [hr]; rfl
      simp [this]
    refine mem_edgeIds_of _ [q.2] r ?_ q.2 (by simp)
    unfold forEachRecord
    simp only [List.mem_append]
    simp [hmem]

/-! ## #3448: a label removal on a cancelled node -/

/-- Opcode records as apply records (`apply_record`, apply.rs); schema records
touch no entity (their own checks are `applyAddSchema_eq_applyAddName`); an
edge's endpoints are `ep`. -/
def toLive (ep : Nat → Nat × Nat) : Rec → List Live.Rec
  | .addLabel .. | .addRelType .. | .addAttribute .. => []
  | .createNode ids ls => [.createNode ids ls]
  | .createEdge ids _ => [.createEdge (ids.map fun i => ⟨i, (ep i).1, (ep i).2⟩)]
  | .updateNode ids => [.updateNode ids []]
  | .updateEdge ids => [.updateEdge ids []]
  | .setLabels ids ls => [.setLabels ids ls]
  | .removeLabels ids ls => [.removeLabels ids ls]
  | .deleteEdge ids => [.deleteEdge ids]
  | .deleteNode ids _ => [.deleteNode ids]

/-- **Property `ReplayApplies`**: the replica at the master's pre-segment state
`r0` accepts the emitted stream. -/
def ReplayApplies (ep : Nat → Nat × Nat) (r0 : Live.G) (p : Pending) (g : Schema) : Prop :=
  ∃ r, Live.applyAll r0 ((forEachRecord p g).flatMap (toLive ep)) = .ok r

/-- An empty replica (the master before `CREATE (n:Foo) …` on a new graph). -/
def rEmpty : Live.G :=
  { nlive := fun _ => false, labels := fun _ => [], props := fun _ => [], rels := [],
    eprops := fun _ => [], released := [], taken := [] }

/-- `CREATE (n:Foo) REMOVE n:Foo DELETE n`: node 0 cancelled, `Foo` (label 0)
left in `remove_labels`. -/
def p3448 : Pending :=
  { createdNodes := [], createdRelTypes := [], deletedNodes := [], deletedRels := [],
    newNodesAttrs := [], existingNodesAttrs := [], newRelsAttrs := [], existingRelsAttrs := [],
    setLabels := [], removeLabels := [(0, [0])], cancelledNodes := [0], cancelledRels := [],
    baseline := ⟨[], [], []⟩ }

def g3448 : Schema := ⟨["Foo"], [], []⟩

/-- **#3448**: the stream names node 0 in `REMOVE_LABELS` before its
`CREATE_NODE`, and the replica refuses it as not live (live: "effects buffer
names node 0, which is not live on this replica … forced full resync"; C stays
in sync). -/
theorem bug3448_removeLabels_before_create_refused :
    forEachRecord p3448 g3448 = [.addLabel 0 "Foo", .removeLabels [0] [0], .createNode [0] [], .deleteNode [0] []] ∧
    Live.errOf (Live.applyAll rEmpty ((forEachRecord p3448 g3448).flatMap (toLive fun _ => (0, 0)))) =
      some (.notLive .node 0 "") ∧
    ¬ ReplayApplies (fun _ => (0, 0)) rEmpty p3448 g3448 := by
  refine ⟨by decide, by decide, fun ⟨r, h⟩ => ?_⟩
  have : Live.errOf (Live.applyAll rEmpty ((forEachRecord p3448 g3448).flatMap (toLive fun _ => (0, 0)))) = none := by
    rw [h]; rfl
  revert this; decide

/-- Control: the same segment without the stale `remove_labels` entry (what
C ships, and what a `delete_pending_node` that also clears `remove_labels` would
give) replays cleanly. -/
theorem agree3448_without_stale_remL :
    Live.okDangling (Live.applyAll rEmpty
      ((forEachRecord { p3448 with removeLabels := [] } g3448).flatMap (toLive fun _ => (0, 0)))) = some none := by
  decide

end FalkorEA.CancelGap
