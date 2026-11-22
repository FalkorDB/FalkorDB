import FalkorEffectsEmitApply.Emit
import FalkorEffectsEmitApply.Liveness
/-!
# Emit then apply, with cancellation: replica divergences

`Emit.forEachRecord` (`for_each_record`, emit.rs:323-375 on 5e475e6c9) and
`Live.applyAll` (the checked arms of `apply_record`) were proven separately;
nothing related what the master emits to what a replica *at the master's
pre-segment state* accepts. Two properties:

* **`RelIdsCovered`** — every relationship id the master allocated in
  the segment (created or cancelled) is named by some `CREATE_EDGE`, so the
  replica's relationship `IdSpace` ends with the same contiguous range
  (`IdSpace::verify`, id_space.rs:773-821, refuses a hole above the entry
  bound). `digest_cancelled` (emit.rs:451) skips a cancelled edge whose type has
  no id, so `Pending::commit` (pending.rs:1367) must register every cancelled type
  (`relIdsCovered_after_commit`; proofs/pending_commit `bug3447_cascade_drops_type_key`).
  `pre3461_bug3447_cancelled_edge_id_gap` is the historical counterexample over the
  commit that registered only `created_rels_by_type` keys: live repro `CREATE ()-[:EXISTING]->()` then
  `CREATE (a)-[r1:NEW_TYPE]->(b), (c)-[r2:EXISTING]->(d) DETACH DELETE a`; the
  payload carries `CreateEdge{ids:[2]}` and nothing for 1, the replica's verify
  refuses (`verifyRange`), the master's passes. `relIdsCovered_of_registered`:
  with every created and cancelled type registered, the property holds for every
  `Pending`.
* **`ReplayApplies`** — the emitted stream, translated to apply records
  (`toLive`), applies on the replica. `digest_labels(&p.remove_labels, None, …)`
  (emit.rs:370) runs before `digest_cancelled` (:374) and does not skip created
  or cancelled nodes, so `delete_pending_node` must clear the node's `remove_labels`
  (proofs/pending_commit `fixed3448_cancelled_node_clears_remL`); the
  `RemoveLabels` arm calls `require_live` (apply.rs:387-388).
  `bug3448_removeLabels_before_create_refused`: `CREATE (n:Foo) REMOVE n:Foo
  DELETE n` with the stale entry emits `AddLabel Foo, RemoveLabels [0], CreateNode [0], DeleteNode [0]`,
  refused with `NotLive node 0`; without it the same segment applies
  (`agree3448_without_stale_remL`). `remove_node_labels` does not
  stage a removal of a label that was only staged, and `delete_pending_node`
  clears `remove_labels`; the segment is `p3459` and replays
  (`fixed3448_replay_applies`). `bug3448_removeLabels_before_create_refused` stays as
  the emitter's behaviour on the old `Pending`, a historical counterexample.
-/
namespace FalkorEA.CancelGap
open FalkorEA

/-! ## Relationship ids -/

/-- `IdSpace::verify` (id_space.rs:773-821), the hole half: the ids taken at or
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

/-- Get-or-create of one name (`get_type_id_mut`, graph.rs). -/
def regName (ts : List Name) (n : Name) : List Name := if n ∈ ts then ts else ts ++ [n]

/-- The relationship-type dictionary after `Pending::commit` before #3461:
each `created_rels_by_type` key is get-or-created (pending.rs:1236-1249 on
343e077f2); cancelled types are not looked at. -/
def typesAfterCommitPre3461 (base : List Name) (p : Pending) : List Name :=
  p.createdRelTypes.foldl (fun ts q => regName ts q.1) base

/-- The relationship-type dictionary after `Pending::commit` (pending.rs:1367-1413,
#3461): each `created_rels_by_type` key is get-or-created by
`create_relationships_bulk`, then the type of every `cancelled_relationships`
entry by `get_type_id_mut` (the `FxHashSet` only skips repeats, which
get-or-create ignores anyway). -/
def typesAfterCommit (base : List Name) (p : Pending) : List Name :=
  p.cancelledRels.foldl (fun ts q => regName ts q.1) (typesAfterCommitPre3461 base p)

/-- `CREATE (a)-[r1:NEW_TYPE]->(b), (c)-[r2:EXISTING]->(d) DETACH DELETE a` after
`CREATE ()-[:EXISTING]->()` (nodes 0,1 and edge 0 committed): `a` = 2 is
cancelled, `b, c, d` = 3, 4, 5 created, `r1` = 1 cancelled with `a`, `r2` = 2
created. -/
def pGap : Pending :=
  { createdNodes := [3, 4, 5], createdRelTypes := [("EXISTING", [2])], deletedNodes := [], deletedRels := [],
    newNodesAttrs := [], existingNodesAttrs := [], newRelsAttrs := [], existingRelsAttrs := [],
    setLabels := [], removeLabels := [], cancelledNodes := [2], cancelledRels := [("NEW_TYPE", 1)],
    baseline := ⟨[], ["EXISTING"], []⟩ }

def gGap : Schema := ⟨[], typesAfterCommitPre3461 ["EXISTING"] pGap, []⟩

/-- History (fixed by #3461, f203972c5; now `relIdsCovered_after_commit`).
**#3447**: `NEW_TYPE` is never registered, so the cancelled edge 1 is not
in the payload; the master's relationship id space `{0, 1, 2}` (entry bound 1)
verifies, the replica's `{0} ∪ {2}` has a hole and is refused
(live: "effects buffer allocated relationship ids 1..=2 but created only 1 of
them … forced full resync"; C stays in sync). -/
theorem pre3461_bug3447_cancelled_edge_id_gap :
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

theorem regName_mem (ts : List Name) (n : Name) : n ∈ regName ts n := by
  unfold regName; split
  · assumption
  · simp

theorem regName_mono (ts : List Name) (n m : Name) (h : m ∈ ts) : m ∈ regName ts n := by
  unfold regName; split
  · exact h
  · simp [h]

theorem foldl_regName_mono {α} (f : α → Name) (l : List α) (ts : List Name) (m : Name) (h : m ∈ ts) :
    m ∈ l.foldl (fun ts q => regName ts (f q)) ts := by
  induction l generalizing ts with
  | nil => exact h
  | cons x xs ih => exact ih _ (regName_mono _ _ _ h)

theorem foldl_regName_mem {α} (f : α → Name) (l : List α) (ts : List Name) (q : α) (hq : q ∈ l) :
    f q ∈ l.foldl (fun ts q => regName ts (f q)) ts := by
  induction l generalizing ts with
  | nil => cases hq
  | cons x xs ih =>
    rcases List.mem_cons.mp hq with rfl | hq
    · exact foldl_regName_mono f xs _ _ (regName_mem _ _)
    · exact ih _ hq

theorem idxOf_isSome_of_mem (l : List Name) (n : Name) (h : n ∈ l) : (idxOf l n).isSome := by
  unfold idxOf
  have : (l.findIdx? (· == n)).isSome := by
    rw [List.findIdx?_isSome, List.any_eq_true]; exact ⟨n, h, by simp⟩
  revert this; cases l.findIdx? (· == n) <;> simp

/-- **#3461 fix of #3447 (`RelIdsCovered`)**: with the dictionary `Pending::commit`
now leaves, every relationship id the segment allocated (created or cancelled)
reaches a `CREATE_EDGE`, whatever the graph's types were before. -/
theorem relIdsCovered_after_commit (p : Pending) (ls base as : List Name) :
    RelIdsCovered p ⟨ls, typesAfterCommit base p, as⟩ := by
  apply relIdsCovered_of_registered
  · intro q hq
    exact idxOf_isSome_of_mem _ _
      (foldl_regName_mono _ _ _ _ (foldl_regName_mem (·.1) _ base q hq))
  · intro q hq
    exact idxOf_isSome_of_mem _ _ (foldl_regName_mem (·.1) _ _ q hq)

/-- The #3447 repro on the #3461 commit: `NEW_TYPE` is registered (id 1), the
cancelled edge 1 is paired, and the replica's id space verifies. -/
theorem fixed3447_cancelled_edge_id_gap :
    typesAfterCommit ["EXISTING"] pGap = ["EXISTING", "NEW_TYPE"] ∧
    RelIdsCovered pGap ⟨[], typesAfterCommit ["EXISTING"] pGap, []⟩ ∧
    verifyRange 1 ([0] ++ edgeIds (forEachRecord pGap ⟨[], typesAfterCommit ["EXISTING"] pGap, []⟩)) = true :=
  ⟨by decide, relIdsCovered_after_commit _ _ _ _, by decide⟩

/-! ## A label removal on a cancelled node -/

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

/-- `CREATE (n:Foo) REMOVE n:Foo DELETE n` as staged **since #3459**: the removal only cancelled
the staged add (`staged_labels_removed = 1`, proofs/pending_commit `fixed3448_remove_of_staged_label`)
and `delete_pending_node` clears `remove_labels` (`fixed3448_cancelled_node_clears_remL`). -/
def p3459 : Pending := { p3448 with removeLabels := [], stagedLabelsRemoved := 1 }

/-- **#3448, fixed by #3459**: the gate stays open (`effects_count` = 2 for the cancelled node + 1 for
the cancelled add), the
commit ships `AddLabel Foo, CreateNode [0], DeleteNode [0]` with no `RemoveLabels`, and the replica
applies it (`ReplayApplies`; before #3459: `bug3448_removeLabels_before_create_refused`). -/
theorem fixed3448_replay_applies :
    effectsCount p3459 = 3 ∧
    commitShips p3459 g3448 = [.addLabel 0 "Foo", .createNode [0] [], .deleteNode [0] []] ∧
    ReplayApplies (fun _ => (0, 0)) rEmpty p3459 g3448 := by
  refine ⟨rfl, by decide, ?_⟩
  have h : Live.okDangling (Live.applyAll rEmpty
      ((forEachRecord p3459 g3448).flatMap (toLive fun _ => (0, 0)))) = some none := by decide
  unfold ReplayApplies
  generalize Live.applyAll rEmpty ((forEachRecord p3459 g3448).flatMap (toLive fun _ => (0, 0))) = x at h ⊢
  cases x with
  | error e => cases h
  | ok r => exact ⟨r, rfl⟩

end FalkorEA.CancelGap
