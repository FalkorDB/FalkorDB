import EndpointGraph
import FalkorIdSpace.Callers
/-!
# `Graph::delete_implicit_edges`: the fallible `release`

`delete_implicit_edges` (`graph.rs`) mutates the tensors, attributes and reverse index, then asks

    self.relationship_ids.release(&freed, &freed).map_err(|e| e.to_string())?;

and only then clears the adjacency bits and returns `all_implicit`. `EndpointGraph` models the
success path. Here the `?` is modelled: the outcome is `Err` exactly when `IdSpace::release`
refuses (`AlreadyRecycled`, `NeverCreated`, or the `checked` verify), with no adjacency cleared and
no edges returned; `Pending::commit` propagates the error and the private MVCC version is dropped.
`release`'s own model is `proofs/id_space` (`release_spec`, `refuseNotLive_ok`).

Whether `ReleaseOk` holds is a property of the graph (every id found in a tensor must be live),
which the engine checks rather than assumes: that is why the refusal exists.
-/
namespace Falkor.EndpointIndex
open IdSpaceModel

/-- `freed`: the collected ids as one set (`all_implicit.iter().map(|e| e.id).collect()`). -/
def freedSet (N X : List Nat) (T : List (List Edge)) : IdSet :=
  fun i => ((deletedEdges N X T).map DE.id).contains i

/-- `delete_implicit_edges` with the `release(...)?` call: `n` is the id bound the id-space model
ranges over, `sp` the relationship `IdSpace`. `Ok` carries the returned edges and the cleared
adjacency pairs; `Err` carries `release`'s error. -/
def implicitOutcome (n : Nat) (sp : IdSpace) (N X : List Nat) (T : List (List Edge)) :
    Except IdSpaceError (List DE × List (Nat × Nat)) :=
  if T.isEmpty then .ok ([], [])
  else match (sp.release n (freedSet N X T) (freedSet N X T)).2 with
    | .error e => .error e
    | .ok () => .ok (deletedEdges N X T, clearedAdj N X T)

/-- **Success exactly when `release` accepts**: with tensors present, the call returns the edges and
the cleared pairs of `EndpointGraph` iff the id-space refusals pass (`ReleaseOk` on the collected
ids) and its `checked` verify holds on the freed state. -/
theorem implicitOutcome_ok (n : Nat) (sp : IdSpace) (N X : List Nat) (T : List (List Edge))
    (hT : T.isEmpty = false) (r : List DE × List (Nat × Nat)) :
    implicitOutcome n sp N X T = .ok r ↔
      ReleaseOk n sp (freedSet N X T) ∧
      ((sp.afterRelease n (freedSet N X T)).checked n = .ok ()) ∧
      r = (deletedEdges N X T, clearedAdj N X T) := by
  unfold implicitOutcome
  simp only [hT, Bool.false_eq_true, ite_false]
  by_cases hk : ReleaseOk n sp (freedSet N X T)
  · rw [(release_spec n sp _ (freedSet N X T)).1 hk]
    cases hc : (sp.afterRelease n (freedSet N X T)).checked n <;> simp [hk, eq_comm]
  · obtain ⟨_, e, he, _⟩ := (release_spec n sp _ (freedSet N X T)).2 hk
    simp [he, hk]

/-- **On a refusal nothing is returned and no adjacency is cleared**: if `release` errs (say a
collected edge id is already recycled), `delete_implicit_edges` returns that error, and the id space
itself is untouched. -/
theorem implicitOutcome_err (n : Nat) (sp : IdSpace) (N X : List Nat) (T : List (List Edge))
    (hT : T.isEmpty = false) (hk : ¬ ReleaseOk n sp (freedSet N X T)) :
    (∃ e, implicitOutcome n sp N X T = .error e) ∧
      (sp.release n (freedSet N X T) (freedSet N X T)).1 = sp := by
  refine ⟨?_, ((release_spec n sp _ (freedSet N X T)).2 hk).1⟩
  obtain ⟨_, e, he, _⟩ := (release_spec n sp _ (freedSet N X T)).2 hk
  exact ⟨e, by simp [implicitOutcome, hT, he]⟩

/-- The reviewer's inconsistent-graph case: a collected edge id that the id space already holds as
recycled makes the whole call `Err`. -/
theorem implicitOutcome_recycled (n : Nat) (sp : IdSpace) (N X : List Nat) (T : List (List Edge))
    (hT : T.isEmpty = false) (d : DE) (hd : d ∈ deletedEdges N X T) (hlt : d.id < n)
    (hrec : sp.recycled d.id = true) : ∃ e, implicitOutcome n sp N X T = .error e := by
  refine (implicitOutcome_err n sp N X T hT fun ⟨h1, _⟩ => h1 d.id hlt ⟨?_, hrec⟩).1
  simp [freedSet]
  exact ⟨d, hd, rfl⟩

#print axioms implicitOutcome_ok
#print axioms implicitOutcome_err
#print axioms implicitOutcome_recycled

end Falkor.EndpointIndex
