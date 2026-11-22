import FalkorIdSpace.Quarantine
/-! # `release`'s caller contract, discharged

`release(requested, freed)` carries `debug_assert!(freed ⊆ requested)` and `live -= freed.len()`;
`release_no_underflow`/`release_succeeds` take that as `hsub`. The engine has exactly three call
sites (`grep '\.release(' graph/src`), and each builds its two sets so that the premise holds:

* `Graph::delete_nodes` (`graph.rs`): `.release(deleted_nodes, deleted_nodes)`.
* the implicit-edge sweep (`graph.rs`): `.release(&freed, &freed)`, one set built once.
* `Graph::delete_relationships` (`graph.rs`): `.release(rels, &resolved)` where phase 1 is
  `for edge_id in rels { … resolved.insert(edge_id) }` — `resolved` is `rels` thinned by whether the
  edge resolved to endpoints and a type (`resolveFrom`), so it is a subset by construction.

So the hypothesis is not an undischarged caller contract: for every shape the engine uses it holds,
and `release` is total and keeps `Inv`.
-/
namespace IdSpaceModel

/-- `graph.rs`'s `resolved`: the members of `rels` the phase-1 walk resolved (`resolves`
is the `endpoints_for_edge` / type-matrix lookup, whatever it answers). -/
def resolveFrom (rels : IdSet) (resolves : Nat → Bool) : IdSet := fun i => rels i && resolves i

theorem hsub_same_set (N : Nat) (s : IdSet) : ∀ i, i < N → s i = true → s i = true :=
  fun _ _ h => h

theorem hsub_resolved (N : Nat) (rels : IdSet) (resolves : Nat → Bool) :
    ∀ i, i < N → resolveFrom rels resolves i = true → rels i = true := by
  intro i _ h
  simp only [resolveFrom, Bool.and_eq_true] at h
  exact h.1

/-- **Every engine call of `release` is total and reachable-preserving**: whichever of the three
shapes it uses, once the refusals pass, `live -= freed.len()` cannot wrap and the result satisfies
`Inv`. -/
theorem release_call_sites (N : Nat) (sp : IdSpace) (h : Inv N sp) (rels : IdSet)
    (resolves : Nat → Bool) (hok : ReleaseOk N sp rels) :
    -- `release(s, s)` (nodes :2320, implicit edges :3249)
    (sp.release N rels rels = (sp.afterRelease N rels, .ok ()) ∧ Inv N (sp.afterRelease N rels) ∧
      sLen N rels ≤ sp.live) ∧
    -- `release(rels, resolved)` (:3006)
    (sp.release N rels (resolveFrom rels resolves) =
        (sp.afterRelease N (resolveFrom rels resolves), .ok ()) ∧
      Inv N (sp.afterRelease N (resolveFrom rels resolves)) ∧
      sLen N (resolveFrom rels resolves) ≤ sp.live) :=
  ⟨⟨(release_succeeds N sp rels rels h hok (hsub_same_set N rels)).1,
    (release_succeeds N sp rels rels h hok (hsub_same_set N rels)).2,
    release_no_underflow N sp rels rels h hok (hsub_same_set N rels)⟩,
   ⟨(release_succeeds N sp rels _ h hok (hsub_resolved N rels resolves)).1,
    (release_succeeds N sp rels _ h hok (hsub_resolved N rels resolves)).2,
    release_no_underflow N sp rels _ h hok (hsub_resolved N rels resolves)⟩⟩

end IdSpaceModel
