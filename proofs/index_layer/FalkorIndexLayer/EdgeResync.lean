import FalkorIndexLayer.Iter
/-
# Gap closed 2026-10-20 (#2781): rolling back edge-index documents

Edge documents are keyed `(src, dst, edge_id)` (`Document::new_edge` `mod.rs:685`,
`ETable` here), but the rollback of a failed multi-`Commit` query decides each
published edge's fate from the edge id alone. `proofs/pending_commit` models the
rollback with documents keyed by id, so it proves the buggy rule faithfully
(`PD.resync_spec`) and cannot state the mismatch. This file keys them by
`(src, dst, id)`.

| Lean | Rust (fb3d516d0) |
| --- | --- |
| `Ep` | `endpoints_for_edge` of one graph version (`committed` or the query's discarded `private`) |
| `resyncEdgeKeys` | `Pending::resync_published_indexes`, edge arm (`runtime/pending.rs:1647-1659`): id live in committed ⇒ re-add, else if `private` knows it ⇒ delete under the private endpoints |
| re-add key | `Graph::commit_edge_index_locked` (`graph/graph.rs:3863-3896`): re-resolves `(src, dst)` from the **committed** tensor; an id with no endpoints is skipped |
| `resyncEdges` | the above then `Indexer::commit_edge` (`indexer.rs:760`, `Iter.commitEdge`: adds, then deletes) |
| `EdgeIndexMatches`, `RollbackRestoresEdgeIndex` | the property: after rollback the edge index is exactly the committed graph's documents |
| `resyncEdgesFixed` | a fix: also delete the private key when the endpoints changed |
-/
namespace IndexLayer.Iter
open IndexLayer

/-- `endpoints_for_edge` of one graph version: edge id ↦ `(src, dst)`. -/
abbrev Ep := Nat → Option (Nat × Nat)

/-- The add keys and the delete keys `resync_published_indexes` produces for the
published edge ids `pub` (one relationship type). -/
def resyncEdgeKeys (committed priv : Ep) (pub : List Nat) :
    List (Nat × Nat × Nat) × List (Nat × Nat × Nat) :=
  (pub.filterMap (fun e => (committed e).map (fun p => (p.1, p.2, e))),
   pub.filterMap (fun e => match committed e with
     | some _ => none
     | none => (priv e).map (fun p => (p.1, p.2, e))))

/-- Main: the rollback applied to the RS edge table `t1` the failed query left. -/
def resyncEdges (build : Nat × Nat × Nat → Doc) (committed priv : Ep) (pub : List Nat)
    (t1 : ETable) : ETable :=
  commitEdge t1 build (resyncEdgeKeys committed priv pub).1 (resyncEdgeKeys committed priv pub).2

/-- The edge index agrees with a graph version: there is a document under
`(s, d, e)` iff edge `e` runs `s → d` there. -/
def EdgeIndexMatches (g : Ep) (t : ETable) : Prop :=
  ∀ s d e, (t (s, d, e)).isSome ↔ g e = some (s, d)

/-- What the failed query's `Commit`s can have done to the shared index: ids it did
not publish are untouched, and a document under a published id is keyed by that
edge's committed or private endpoints (its own `Commit`s add under the private
endpoints and delete what they replace). -/
def PublishedBy (committed priv : Ep) (pub : List Nat) (t0 t1 : ETable) : Prop :=
  (∀ s d e, e ∉ pub → t1 (s, d, e) = t0 (s, d, e)) ∧
  (∀ s d e, e ∈ pub → (t1 (s, d, e)).isSome → committed e = some (s, d) ∨ priv e = some (s, d))

/-- **Property (#2781).** Rolling back a failed query restores the edge index to
the committed graph's documents: C's undo-log guarantee, which the doc comment of
`resync_published_indexes` (`pending.rs:1607-1620`) claims. -/
def RollbackRestoresEdgeIndex (resync : Ep → Ep → List Nat → ETable → ETable) : Prop :=
  ∀ committed priv pub t0 t1, EdgeIndexMatches committed t0 →
    PublishedBy committed priv pub t0 t1 → EdgeIndexMatches committed (resync committed priv pub t1)

/-! ## The live repro -/

/-- `CREATE (:A)-[:R {w:1}]->(:B), (:C), (:D)`: nodes a=0, b=1, c=2, d=3, edge 0 = a→b. -/
def committed2781 : Ep := fun e => if e = 0 then some (0, 1) else none
/-- The failed query's private version: `MATCH ()-[r:R]->() DELETE r WITH 1 AS x
MATCH (c:C), (d:D) CREATE (c)-[:R {w:1}]->(d) WITH 1 AS x RETURN 1/0` reuses id 0 for c→d. -/
def private2781 : Ep := fun e => if e = 0 then some (2, 3) else none
def doc0 : Doc := fun _ _ => []
/-- Before the query: the index holds the committed edge's document. -/
def t0_2781 : ETable := fun k => if k = (0, 1, 0) then some doc0 else none
/-- After its two `Commit`s: the first deleted `(0,1,0)`, the second added `(2,3,0)`. -/
def t1_2781 : ETable := fun k => if k = (2, 3, 0) then some doc0 else none

theorem t0_2781_matches : EdgeIndexMatches committed2781 t0_2781 := by
  intro s d e
  by_cases he : e = 0
  · subst he; by_cases h : (s, d) = (0, 1)
    · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; decide
    · have h' : ¬ ((s, d, 0) = ((0 : Nat), (1 : Nat), (0 : Nat))) := by
        intro hh; apply h; simp only [Prod.mk.injEq] at hh ⊢; exact ⟨hh.1, hh.2.1⟩
      simp [t0_2781, committed2781, h', Ne.symm h]
  · have h' : ¬ ((s, d, e) = ((0 : Nat), (1 : Nat), (0 : Nat))) := by
      intro hh; simp only [Prod.mk.injEq] at hh; exact he hh.2.2
    simp [t0_2781, committed2781, h', he]

theorem t1_2781_published : PublishedBy committed2781 private2781 [0] t0_2781 t1_2781 := by
  refine ⟨fun s d e he => ?_, fun s d e he hs => ?_⟩
  · have he' : e ≠ 0 := by simpa using he
    have h1 : ¬ ((s, d, e) = ((2 : Nat), (3 : Nat), (0 : Nat))) := by
      intro hh; simp only [Prod.mk.injEq] at hh; exact he' hh.2.2
    have h0 : ¬ ((s, d, e) = ((0 : Nat), (1 : Nat), (0 : Nat))) := by
      intro hh; simp only [Prod.mk.injEq] at hh; exact he' hh.2.2
    simp [t1_2781, t0_2781, h1, h0]
  · have he' : e = 0 := by simpa using he
    subst he'
    by_cases h : (s, d, 0) = ((2 : Nat), (3 : Nat), (0 : Nat))
    · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl, -⟩ := h; right; rfl
    · simp [t1_2781, h] at hs

/-- **BUG #2781.** After main's rollback the index holds a document under the
committed key `(a, b, 0)` **and** the orphan `(c, d, 0)` published under the rolled-back
endpoints; edge 0 is live in committed, so the rollback only re-adds it and never
deletes the private key. Live (port 6640 Rust main 9d3869aa8):
`MATCH (s)-[r:R]->(t) WHERE r.w = 1` through the index → `(c,0,d)` and `(a,0,b)`,
a full scan → `(a,0,b)` only; C (6641) → `(a,0,b)` both ways. -/
theorem bug2781_orphan_edge_doc (build : Nat × Nat × Nat → Doc) :
    (resyncEdges build committed2781 private2781 [0] t1_2781 (2, 3, 0)).isSome = true ∧
    (resyncEdges build committed2781 private2781 [0] t1_2781 (0, 1, 0)).isSome = true ∧
    committed2781 0 = some (0, 1) := by
  refine ⟨?_, ?_, rfl⟩ <;>
    simp [resyncEdges, commitEdge_lookup, resyncEdgeKeys, committed2781, private2781, t1_2781]

theorem bug2781_rollback_not_restored (build : Nat × Nat × Nat → Doc) :
    ¬ RollbackRestoresEdgeIndex (resyncEdges build) := by
  intro hP
  have := (hP committed2781 private2781 [0] t0_2781 t1_2781 t0_2781_matches t1_2781_published 2 3 0).mp
    (bug2781_orphan_edge_doc build).1
  simp [committed2781] at this

/-! ## What a fix can prove -/

/-- A fix: when the private endpoints of a published id differ from the committed
ones (or the id is gone from committed), delete the private key; re-add under the
committed endpoints as before. -/
def resyncEdgeKeysFixed (committed priv : Ep) (pub : List Nat) :
    List (Nat × Nat × Nat) × List (Nat × Nat × Nat) :=
  (pub.filterMap (fun e => (committed e).map (fun p => (p.1, p.2, e))),
   pub.filterMap (fun e => match priv e with
     | some p => if committed e = some p then none else some (p.1, p.2, e)
     | none => none))

def resyncEdgesFixed (build : Nat × Nat × Nat → Doc) (committed priv : Ep) (pub : List Nat)
    (t1 : ETable) : ETable :=
  commitEdge t1 build (resyncEdgeKeysFixed committed priv pub).1 (resyncEdgeKeysFixed committed priv pub).2

theorem resyncEdgesFixed_restores (build : Nat × Nat × Nat → Doc) :
    RollbackRestoresEdgeIndex (resyncEdgesFixed build) := by
  intro committed priv pub t0 t1 h0 ⟨hout, hin⟩ s d e
  have memA : (s, d, e) ∈ (resyncEdgeKeysFixed committed priv pub).1 ↔
      e ∈ pub ∧ committed e = some (s, d) := by
    simp only [resyncEdgeKeysFixed, List.mem_filterMap, Option.map_eq_some_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨e', he', p, hp, rfl, rfl, rfl⟩; exact ⟨he', hp⟩
    · rintro ⟨he, hc⟩; exact ⟨e, he, (s, d), hc, rfl, rfl, rfl⟩
  have memR : (s, d, e) ∈ (resyncEdgeKeysFixed committed priv pub).2 ↔
      e ∈ pub ∧ priv e = some (s, d) ∧ committed e ≠ some (s, d) := by
    simp only [resyncEdgeKeysFixed, List.mem_filterMap]
    constructor
    · rintro ⟨e', he', hm⟩
      split at hm
      · next p hp =>
        split at hm
        · cases hm
        · next hc =>
          simp only [Option.some.injEq, Prod.mk.injEq] at hm
          obtain ⟨rfl, rfl, rfl⟩ := hm
          exact ⟨he', hp, hc⟩
      · cases hm
    · rintro ⟨he, hp, hc⟩
      exact ⟨e, he, by simp [hp, hc]⟩
  simp only [resyncEdgesFixed, commitEdge_lookup]
  by_cases hR : (s, d, e) ∈ (resyncEdgeKeysFixed committed priv pub).2
  · simp only [hR, ite_true, Option.isSome_none, Bool.false_eq_true, false_iff]
    exact ((memR.mp hR).2.2)
  · simp only [hR, ite_false]
    by_cases hA : (s, d, e) ∈ (resyncEdgeKeysFixed committed priv pub).1
    · simp only [hA, ite_true, Option.isSome_some, true_iff]; exact (memA.mp hA).2
    · simp only [hA, ite_false]
      by_cases he : e ∈ pub
      · have hc : committed e ≠ some (s, d) := fun hc => hA (memA.mpr ⟨he, hc⟩)
        constructor
        · intro hs
          rcases hin s d e he hs with h | h
          · exact absurd h hc
          · exact absurd (memR.mpr ⟨he, h, hc⟩) hR
        · intro h; exact absurd h hc
      · rw [hout s d e he]; exact h0 s d e

end IndexLayer.Iter
