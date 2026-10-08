import FalkorEffectsEmitApply.ApplyHelpers
import FalkorEffectsEmitApply.Apply
/-!
# The apply side after #3022: every record that acts on an entity checks it is live

Fixed by #3022 (`18fc277b9`; issues #2924, #2978; FINDINGS W2-effects-2/3). The
node- and edge-facing arms of `apply_record` (`graph/src/effects/v3/apply.rs:157`)
now refuse a record naming an entity this replica does not hold live:

| here | there (`apply.rs`) |
| --- | --- |
| `needLive` | `require_live` (`:648`) over `IdSpace::refuse_not_live` (accepts exactly the live ids: `proofs/id_space` `refuseNotLive_live`) |
| `firstWithRels` | `Graph::first_node_with_relationships` (`graph.rs:2245`; its Rust is proved against this spec in `proofs/graph_queries`) |
| `applyRec .createNode` | `CreateNode` arm (`:203`) — `create_nodes` refuses a live id |
| `applyRec .createEdge` | `CreateEdge` arm (`:254`) — `require_live((src ∪ dst) − released)` (`:279-280`), then `create_relationships_bulk` refuses a live or repeated id |
| `applyRec .updateNode` | `UpdateNode` arm (`:308`) — `require_live` (`:314`) |
| `applyRec .updateEdge` | `UpdateEdge` arm (`:328`) — `require_live` (`:334`) |
| `applyRec .setLabels/.removeLabels` | `:348` / `:360` — `require_live` (`:349`, `:361`) |
| `applyRec .deleteNode` | `DeleteNode` arm (`:380`) — `require_live` (`:386`), `NodeHasRelationships` (`:390-392`), then `delete_nodes` (records `released`) |
| `applyRec .deleteEdge` | `DeleteEdge` arm (`:403`) — `delete_relationships` refuses a non-live id |
| `danglingRel` | `Graph::validate`'s `verify_created_relationships` (`graph.rs:1524`), run by `apply_effects` (`:86`) |

The replica is abstract (as in `Apply.lean`): node liveness, labels, properties,
live relationships `(id, src, dst)` with properties, and the batch's `released`
node ids and `taken` relationship ids (`IdSpace`'s per-batch fields). Error
payload strings are not modelled.

Headline: **`apply_wf`** — from a well-formed replica at a batch boundary, any
sequence of records that applies, followed by a `validate` that passes, leaves a
well-formed replica: no dead id carries a label, property or relationship, and
every relationship's endpoints are live. The pre-#3022 counterexamples
(`Apply.lean`, `pre3022_*`) are each refused here (`*_refused`).
-/

namespace FalkorEA.Live
open FalkorEA

structure Edge where
  id : Nat
  src : Nat
  dst : Nat
deriving DecidableEq, Repr

structure G where
  nlive    : Nat → Bool
  labels   : Nat → List Nat
  props    : Nat → List (Nat × Val)
  rels     : List Edge
  eprops   : Nat → List (Nat × Val)
  released : List Nat
  taken    : List Nat

def G.elive (g : G) (r : Nat) : Bool := g.rels.any (·.id == r)
def G.hasRel (g : G) (n : Nat) : Bool := g.rels.any (fun e => e.src == n || e.dst == n)

inductive Rec where
  | createNode (ids lbls : List Nat)
  | createEdge (es : List Edge)
  | updateNode (ids : List Nat) (row : List (Nat × Val))
  | updateEdge (ids : List Nat) (row : List (Nat × Val))
  | setLabels (ids lbls : List Nat)
  | removeLabels (ids lbls : List Nat)
  | deleteNode (ids : List Nat)
  | deleteEdge (ids : List Nat)

/-- `require_live`: refuse an id that is not live. -/
def needLive (live : Nat → Bool) (k : Kind) (ids : List Nat) : Except AErr Unit :=
  match ids.find? (fun i => !live i) with
  | some i => .error (.notLive k i "")
  | none => .ok ()

/-- `IdSpace::create`'s `AlreadyLive` refusal, as the create arms meet it. -/
def needDead (live : Nat → Bool) (k : Kind) (ids : List Nat) : Except AErr Unit :=
  match ids.find? live with
  | some i => .error (.alreadyLive k i 0)
  | none => .ok ()

/-- `first_node_with_relationships` (spec): the first of `ids` with a live
relationship in either direction. -/
def firstWithRels (g : G) (ids : List Nat) : Option Nat := ids.find? g.hasRel

def upd {α} (ids : List Nat) (f : Nat → α) (v : Nat → α) : Nat → α :=
  fun i => if ids.contains i then v i else f i

def applyRec (g : G) : Rec → Except AErr G
  | .createNode ids lbls => do
    needDead g.nlive .node ids
    pure { g with nlive := upd ids g.nlive (fun _ => true), labels := upd ids g.labels (fun _ => lbls) }
  | .createEdge es => do
    needLive g.nlive .node ((es.flatMap fun e => [e.src, e.dst]).filter (fun n => !g.released.contains n))
    if !(es.map (·.id)).Nodup then .error (.graph "a relationship id repeats") else
    needDead g.elive .relationship (es.map (·.id))
    pure { g with rels := g.rels ++ es, taken := g.taken ++ es.map (·.id) }
  | .updateNode ids row => do
    needLive g.nlive .node ids
    pure { g with props := upd ids g.props (fun i => row ++ g.props i) }
  | .updateEdge ids row => do
    needLive g.elive .relationship ids
    pure { g with eprops := upd ids g.eprops (fun i => row ++ g.eprops i) }
  | .setLabels ids lbls => do
    needLive g.nlive .node ids
    pure { g with labels := upd ids g.labels (fun i => lbls ++ g.labels i) }
  | .removeLabels ids lbls => do
    needLive g.nlive .node ids
    pure { g with labels := upd ids g.labels (fun i => (g.labels i).filter (fun l => !lbls.contains l)) }
  | .deleteNode ids => do
    needLive g.nlive .node ids
    match firstWithRels g ids with
    | some id => .error (.nodeHasRelationships id)
    | none =>
      pure { g with nlive := upd ids g.nlive (fun _ => false), labels := upd ids g.labels (fun _ => []),
                    props := upd ids g.props (fun _ => []), released := g.released ++ ids }
  | .deleteEdge ids => do
    needLive g.elive .relationship ids
    pure { g with rels := g.rels.filter (fun e => !ids.contains e.id),
                  eprops := upd ids g.eprops (fun _ => []) }

def applyAll (g : G) (recs : List Rec) : Except AErr G := recs.foldlM applyRec g

/-- `verify_created_relationships`: the first relationship this batch took that is
still live and does not end at two live nodes. -/
def danglingRel (g : G) : Option Nat :=
  g.taken.find? fun r => match g.rels.find? (·.id == r) with
    | some e => !(g.nlive e.src && g.nlive e.dst)
    | none => false

/-- Well formed: a dead node carries nothing, a dead relationship carries no
properties, relationship ids are distinct, and every relationship ends at two
live nodes. -/
structure WF (g : G) : Prop where
  dead  : ∀ i, g.nlive i = false → g.labels i = [] ∧ g.props i = []
  edead : ∀ r, g.elive r = false → g.eprops r = []
  nodup : (g.rels.map (·.id)).Nodup
  ends  : ∀ e ∈ g.rels, g.nlive e.src = true ∧ g.nlive e.dst = true

/-- Mid-batch: as `WF`, except an endpoint may be a node *this batch released*,
provided the relationship is one this batch created (the cancelled-pair exemption). -/
structure BInv (g : G) : Prop where
  dead  : ∀ i, g.nlive i = false → g.labels i = [] ∧ g.props i = []
  edead : ∀ r, g.elive r = false → g.eprops r = []
  nodup : (g.rels.map (·.id)).Nodup
  ends  : ∀ e ∈ g.rels,
    (g.nlive e.src = true ∨ (e.id ∈ g.taken ∧ e.src ∈ g.released)) ∧
    (g.nlive e.dst = true ∨ (e.id ∈ g.taken ∧ e.dst ∈ g.released))

/-! ## The checks -/

theorem needLive_ok {live : Nat → Bool} {k ids} :
    needLive live k ids = .ok () ↔ ∀ i ∈ ids, live i = true := by
  unfold needLive
  cases h : ids.find? (fun i => !live i) with
  | some i =>
    simp only [reduceCtorEq, false_iff, Classical.not_forall]
    have := List.find?_some h; have hm := List.mem_of_find?_eq_some h
    exact ⟨i, hm, by simpa using this⟩
  | none =>
    simp only [true_iff]
    intro i hi; have := List.find?_eq_none.mp h i hi; simpa using this

theorem needDead_ok {live : Nat → Bool} {k ids} :
    needDead live k ids = .ok () ↔ ∀ i ∈ ids, live i = false := by
  unfold needDead
  cases h : ids.find? live with
  | some i =>
    simp only [reduceCtorEq, false_iff, Classical.not_forall]
    exact ⟨i, List.mem_of_find?_eq_some h, by simp [List.find?_some h]⟩
  | none =>
    simp only [true_iff]
    intro i hi; have := List.find?_eq_none.mp h i hi; simpa using this

theorem firstWithRels_none {g : G} {ids} :
    firstWithRels g ids = none ↔ ∀ i ∈ ids, g.hasRel i = false := by
  unfold firstWithRels; rw [List.find?_eq_none]; simp

theorem firstWithRels_some {g : G} {ids id} (h : firstWithRels g ids = some id) :
    id ∈ ids ∧ g.hasRel id = true := by
  unfold firstWithRels at h
  exact ⟨List.mem_of_find?_eq_some h, List.find?_some h⟩

theorem hasRel_false {g : G} {n} (h : g.hasRel n = false) : ∀ e ∈ g.rels, e.src ≠ n ∧ e.dst ≠ n := by
  intro e he
  have := List.any_eq_false.mp h e he
  simp at this; exact this

theorem upd_in {α} {ids : List Nat} {f v : Nat → α} {i} (h : i ∈ ids) : upd ids f v i = v i := by
  simp [upd, h]
theorem upd_out {α} {ids : List Nat} {f v : Nat → α} {i} (h : i ∉ ids) : upd ids f v i = f i := by
  simp [upd, h]

/-! ## Each arm preserves `BInv` -/

theorem binv_of_wf (g : G) (h : WF g) : BInv g :=
  ⟨h.dead, h.edead, h.nodup, fun e he => ⟨.inl (h.ends e he).1, .inl (h.ends e he).2⟩⟩

theorem elive_iff {g : G} {r} : g.elive r = true ↔ ∃ e ∈ g.rels, e.id = r := by
  simp [G.elive]

theorem applyRec_binv (g g' : G) (r : Rec) (hi : BInv g) (h : applyRec g r = .ok g') : BInv g' := by
  cases r with
  | createNode ids lbls =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hd; cases h
    refine ⟨fun i hi' => ?_, hi.edead, hi.nodup, fun e he => ?_⟩
    · by_cases m : i ∈ ids
      · simp [upd_in m] at hi'
      · simp only [upd_out m] at hi' ⊢; exact hi.dead i hi'
    · have := hi.ends e he
      constructor
      · rcases this.1 with a | a
        · left; by_cases m : e.src ∈ ids <;> simp [upd, m, a]
        · exact .inr a
      · rcases this.2 with a | a
        · left; by_cases m : e.dst ∈ ids <;> simp [upd, m, a]
        · exact .inr a
  | createEdge es =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl
    split at h; · cases h
    rename_i hnd
    split at h; · cases h
    rename_i hdd; cases h
    have hl' := needLive_ok.mp hl
    have hdd' := needDead_ok.mp hdd
    simp only [Bool.not_eq_true', decide_eq_false_iff_not, Classical.not_not] at hnd
    refine ⟨hi.dead, fun r hr => ?_, ?_, fun e he => ?_⟩
    · apply hi.edead
      simp only [G.elive, List.any_append, Bool.or_eq_false_iff] at hr ⊢; exact hr.1
    · simp only [List.map_append]
      refine List.nodup_append.mpr ⟨hi.nodup, hnd, fun a ha b hb hab => ?_⟩
      subst hab
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp ha
      have h1 := hdd' e.id hb
      simp [G.elive] at h1
      exact h1 e he rfl
    · simp only [List.mem_append] at he
      rcases he with he | he
      · have := hi.ends e he
        exact ⟨this.1.imp id (fun ⟨a, b⟩ => ⟨List.mem_append_left _ a, b⟩),
               this.2.imp id (fun ⟨a, b⟩ => ⟨List.mem_append_left _ a, b⟩)⟩
      · have hid : e.id ∈ g.taken ++ es.map (·.id) := List.mem_append_right _ (List.mem_map_of_mem he)
        have side : ∀ n, n = e.src ∨ n = e.dst → g.nlive n = true ∨ (e.id ∈ g.taken ++ es.map (·.id) ∧ n ∈ g.released) := by
          intro n hn
          by_cases hr : g.released.contains n = true
          · exact .inr ⟨hid, by simpa using hr⟩
          · left; apply hl' n
            simp only [List.mem_filter, List.mem_flatMap]
            refine ⟨⟨e, he, by rcases hn with rfl | rfl <;> simp⟩, by simpa using hr⟩
        exact ⟨side _ (.inl rfl), side _ (.inr rfl)⟩
  | updateNode ids row =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl; cases h
    have hl' := needLive_ok.mp hl
    refine ⟨fun i hi' => ?_, hi.edead, hi.nodup, hi.ends⟩
    have hm : i ∉ ids := fun m => by rw [hl' i m] at hi'; cases hi'
    simp only [upd_out hm]; exact hi.dead i hi'
  | updateEdge ids row =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl; cases h
    have hl' := needLive_ok.mp hl
    refine ⟨hi.dead, fun r hr => ?_, hi.nodup, hi.ends⟩
    have hr : g.elive r = false := hr
    have hm : r ∉ ids := fun m => by rw [hl' r m] at hr; cases hr
    simp only [upd_out hm]; exact hi.edead r hr
  | setLabels ids lbls =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl; cases h
    have hl' := needLive_ok.mp hl
    refine ⟨fun i hi' => ?_, hi.edead, hi.nodup, hi.ends⟩
    have hm : i ∉ ids := fun m => by rw [hl' i m] at hi'; cases hi'
    simp only [upd_out hm]; exact hi.dead i hi'
  | removeLabels ids lbls =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl; cases h
    have hl' := needLive_ok.mp hl
    refine ⟨fun i hi' => ?_, hi.edead, hi.nodup, hi.ends⟩
    have hm : i ∉ ids := fun m => by rw [hl' i m] at hi'; cases hi'
    simp only [upd_out hm]; exact hi.dead i hi'
  | deleteNode ids =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl
    split at h; · cases h
    rename_i hf; cases h
    have hno := firstWithRels_none.mp hf
    refine ⟨fun i hi' => ?_, hi.edead, hi.nodup, fun e he => ?_⟩
    · by_cases m : i ∈ ids
      · simp [upd_in m]
      · simp only [upd_out m] at hi' ⊢; exact hi.dead i hi'
    · have hs : e.src ∉ ids := fun m => (hasRel_false (hno _ m) e he).1 rfl
      have hd : e.dst ∉ ids := fun m => (hasRel_false (hno _ m) e he).2 rfl
      have := hi.ends e he
      simp only [upd_out hs, upd_out hd]
      exact ⟨this.1.imp id (fun ⟨a, b⟩ => ⟨a, List.mem_append_left _ b⟩),
             this.2.imp id (fun ⟨a, b⟩ => ⟨a, List.mem_append_left _ b⟩)⟩
  | deleteEdge ids =>
    simp only [applyRec, bind, Except.bind] at h
    split at h; · cases h
    rename_i hl; cases h
    refine ⟨hi.dead, fun r hr => ?_, ?_, fun e he => ?_⟩
    · by_cases m : r ∈ ids
      · simp [upd_in m]
      · simp only [upd_out m]; apply hi.edead
        simp only [G.elive, List.any_eq_false, List.mem_filter] at hr ⊢
        intro e he' heq
        exact hr e ⟨he', by simp at heq; subst heq; simpa using m⟩ heq
    · exact List.Nodup.sublist (List.filter_sublist.map _) hi.nodup
    · exact hi.ends e (List.mem_filter.mp he).1

/-! ## A whole buffer -/

theorem applyAll_binv : ∀ (recs : List Rec) (g g' : G), BInv g → applyAll g recs = .ok g' → BInv g'
  | [], g, g', hi, h => by simp [applyAll, pure, Except.pure] at h; subst h; exact hi
  | r :: rs, g, g', hi, h => by
    simp only [applyAll, List.foldlM_cons, bind, Except.bind] at h
    split at h; · cases h
    rename_i g1 h1
    exact applyAll_binv rs g1 g' (applyRec_binv g g1 r hi h1) h

theorem eq_of_id_eq {l : List Edge} (hn : (l.map (·.id)).Nodup) {a b : Edge}
    (ha : a ∈ l) (hb : b ∈ l) (h : a.id = b.id) : a = b := by
  induction l with
  | nil => cases ha
  | cons x t ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    simp only [List.mem_cons] at ha hb
    rcases ha with rfl | ha <;> rcases hb with rfl | hb
    · rfl
    · exact absurd ⟨b, hb, h.symm⟩ hn.1
    · exact absurd ⟨a, ha, h⟩ hn.1
    · exact ih hn.2 ha hb

/-- **A passing `validate` turns the batch invariant into well-formedness**: the
only relationships `BInv` lets end at a non-live node are ones this batch took,
and `verify_created_relationships` checks exactly those. -/
theorem wf_of_validate (g : G) (hi : BInv g) (hv : danglingRel g = none) : WF g := by
  refine ⟨hi.dead, hi.edead, hi.nodup, fun e he => ?_⟩
  by_cases ht : e.id ∈ g.taken
  · have := List.find?_eq_none.mp hv e.id ht
    have hf : g.rels.find? (·.id == e.id) = some e := by
      cases hfe : g.rels.find? (·.id == e.id) with
      | none => have := List.find?_eq_none.mp hfe e he; simp at this
      | some e' =>
        have h1 := List.mem_of_find?_eq_some hfe
        have h2 := List.find?_some hfe; simp at h2
        rw [eq_of_id_eq hi.nodup h1 he h2]
    rw [hf] at this; simpa using this
  · have := hi.ends e he
    exact ⟨this.1.resolve_right (fun h => ht h.1), this.2.resolve_right (fun h => ht h.1)⟩

/-- **#3022's correctness theorem.** From a well-formed replica, any buffer whose
records all apply and whose end-of-buffer `validate` passes leaves a well-formed
replica: no recycled id carries a label, property or relationship (W2-effects-2:
`UPDATE_NODE`/`UPDATE_EDGE`/`SET_LABELS`/`REMOVE_LABELS` now `require_live`;
W2-effects-3: `DELETE_NODE` refuses a node with relationships), and every
relationship ends at two live nodes (#2924: `CREATE_EDGE` endpoints are live or
released by this batch, and `validate` refuses any such edge left standing). -/
theorem apply_wf (g : G) (hw : WF g) (recs : List Rec) (g' : G)
    (h : applyAll g recs = .ok g') (hv : danglingRel g' = none) : WF g' :=
  wf_of_validate g' (applyAll_binv recs g g' (binv_of_wf g hw) h) hv

/-! ## The Rust tests (`apply.rs` `mod tests`, #3022), replayed

`liveness_setup`: nodes 0, 1, 2 created, edge 0 from 0 to 1, node 2 deleted, then
a new version (so `released` is empty again). -/

def s0 : G :=
  { nlive := fun i => i < 2, labels := fun _ => [], props := fun _ => [], rels := [⟨0, 0, 1⟩],
    eprops := fun _ => [], released := [], taken := [] }

def errOf : Except AErr G → Option AErr
  | .error e => some e
  | .ok _ => none

def okDangling : Except AErr G → Option (Option Nat)
  | .error _ => none
  | .ok g => some (danglingRel g)

theorem s0_wf : WF s0 := by
  refine ⟨fun i h => ⟨rfl, rfl⟩, fun _ _ => rfl, by decide, fun e he => ?_⟩
  simp [s0] at he; subst he; decide

/-- `an_edge_to_a_node_that_is_not_live_is_refused` (#2924). -/
theorem edge_to_dead_refused :
    [(0, 2, 2), (2, 0, 2), (0, 7, 7), (7, 7, 7)].all (fun (t : Nat × Nat × Nat) =>
      errOf (applyRec s0 (.createEdge [⟨1, t.1, t.2.1⟩])) == some (.notLive .node t.2.2 "")) := by
  decide

/-- `an_edge_to_a_node_created_earlier_in_the_buffer_applies`. -/
theorem edge_to_created_applies :
    okDangling (applyAll s0 [.createNode [2, 3] [], .createEdge [⟨1, 2, 3⟩]]) = some none := by decide

/-- `an_edge_onto_a_node_deleted_earlier_in_the_buffer_must_not_survive_it`: the
cancelled pair (create, delete, edge onto it, delete the edge) nets out; without
the `DELETE_EDGE` the records apply but `validate` names relationship 1. -/
theorem cancelled_pair :
    okDangling (applyAll s0 [.createNode [2] [], .deleteNode [2], .createEdge [⟨1, 0, 2⟩],
      .deleteEdge [1]]) = some none ∧
    okDangling (applyAll s0 [.createNode [2] [], .deleteNode [2], .createEdge [⟨1, 0, 2⟩]])
      = some (some 1) := by decide

/-- `updating_a_node_that_is_not_live_is_refused` / `relabelling_…` /
`updating_an_edge_that_is_not_live_is_refused` (W2-effects-2). -/
theorem updates_on_dead_refused :
    [2, 50].all (fun id =>
      errOf (applyRec s0 (.updateNode [id] [(0, .int 666)])) == some (.notLive .node id "") &&
      errOf (applyRec s0 (.setLabels [id] [0])) == some (.notLive .node id "") &&
      errOf (applyRec s0 (.removeLabels [id] [0])) == some (.notLive .node id "")) ∧
    errOf (applyRec s0 (.updateEdge [5] [(0, .int 888)])) = some (.notLive .relationship 5 "") := by
  decide

/-- `updates_and_labels_on_live_entities_still_apply`. -/
theorem updates_on_live_apply :
    okDangling (applyAll s0 [.updateNode [0, 1] [(0, .int 1)], .setLabels [0] [0],
      .removeLabels [0] [0], .updateEdge [0] [(0, .int 3)]]) = some none := by decide

/-- `deleting_a_node_that_still_has_edges_is_refused` (W2-effects-3), either end. -/
theorem delete_with_edges_refused :
    errOf (applyRec s0 (.deleteNode [0])) = some (.nodeHasRelationships 0) ∧
    errOf (applyRec s0 (.deleteNode [1])) = some (.nodeHasRelationships 1) := by decide

/-- `deleting_the_edges_first_lets_the_node_go` (what the primary ships for `DETACH DELETE`). -/
theorem detach_delete_applies :
    okDangling (applyAll s0 [.deleteEdge [0], .deleteNode [0, 1]]) = some none := by decide

end FalkorEA.Live
