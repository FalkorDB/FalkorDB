import OpsTraverse.BidirBfs
import OpsTraverse.EvalSP
import OpsTraverse.NeighborIter
/-
# `bfs_shortest_path` as `eval_shortest_path`'s search

`EvalSP.evalSP` takes the search as a parameter; `bfsSV` is `bfs_shortest_path` over the two
`NeighborIter`s (`NeighborIter.neighbors` of `newNI` / `newRev`) returning the node list as `SV.path`.
The `min_hops` filter (a shortest path with fewer than `min_hops` edges is `Null`) is applied here.
-/
namespace OpsTraverse.BspWire
open OpsTraverse.Search OpsTraverse.BidirBfs OpsTraverse.EvalSP OpsTraverse.NeighborIter

/-- The forward and backward neighbour functions `eval_shortest_path` builds
(`new(g, rel_types, directed)`; `new_reversed` for the backward side when directed, else `new` again). -/
def fwdN (rt : List String) (directed : Bool) (adj : Row) (allT : List Row) (typed : String → Option (Row × Row)) :
    Nat → List Nat := neighbors (newNI rt directed adj allT typed)

def bwdN (rt : List String) (directed : Bool) (adj : Row) (allT : List Row) (typed : String → Option (Row × Row)) :
    Nat → List Nat :=
  if directed then neighbors (newRev rt allT typed) else neighbors (newNI rt false adj allT typed)

/-- `bfs_shortest_path` up to the returned `Value`, as `EvalSP`'s `bfs` parameter. -/
def bfsSV (nf nb : Nat → List Nat) (fuel : Nat) (s d maxLevel minHops : Nat) : SV :=
  match bspNodes nf nb s d maxLevel fuel with
  | some (some P) => if P.length - 1 < minHops then .null else .path P
  | _ => .null

/-- **PROVEN** (`bfs_shortest_path`, end to end for the value): given that the backward neighbour rows
are the converse of the forward ones (`hrv1`/`hrv2`, a property of the matrices `NeighborIter` reads,
not of this function), a returned path is the node sequence of a walk `s → d` along the forward rows that
is a shortest one (so it visits distinct nodes), at most `max_level` long and at least `min_hops` long;
`Null` only when `s = d`, no walk within `max_level` exists, or the shortest walk is shorter than
`min_hops`. -/
theorem bfsSV_spec (nf nb : Nat → List Nat) (hrv1 : ∀ u v, v ∈ nf u → u ∈ nb v)
    (hrv2 : ∀ u v, v ∈ nb u → u ∈ nf v) (fuel s d maxLevel minHops : Nat) (r : Option (List Nat))
    (hb : bspNodes nf nb s d maxLevel fuel = some r) :
    (∀ P, bfsSV nf nb fuel s d maxLevel minHops = .path P → ∃ es, WalkTo (gs nf) s es d ∧ nodesOf s es = P ∧
        Dst (gs nf) s d es.length ∧ es.length ≤ maxLevel ∧ minHops ≤ es.length ∧ P.Nodup) ∧
    (bfsSV nf nb fuel s d maxLevel minHops = .null → s = d ∨ (∀ es, WalkTo (gs nf) s es d → maxLevel < es.length) ∨
        ∃ es, WalkTo (gs nf) s es d ∧ Dst (gs nf) s d es.length ∧ es.length < minHops) := by
  obtain ⟨h0, h1⟩ := bspNodes_spec nf nb hrv1 hrv2 s d maxLevel fuel r hb
  unfold bfsSV
  rw [hb]
  by_cases hsd : s = d
  · refine ⟨fun P hP => ?_, fun _ => Or.inl hsd⟩
    have := h0 hsd; subst this; simp at hP
  · obtain ⟨hn, hs⟩ := h1 hsd
    cases r with
    | none => exact ⟨fun P hP => by simp at hP, fun _ => Or.inr (Or.inl (hn rfl))⟩
    | some P =>
      obtain ⟨es, hw, hnodes, hD, hle⟩ := hs P rfl
      have hlen : P.length = es.length + 1 := by rw [← hnodes]; simp [nodesOf]
      by_cases hm : P.length - 1 < minHops
      · simp only [hm, ↓reduceIte]
        exact ⟨fun P' h' => by simp at h', fun _ => Or.inr (Or.inr ⟨es, hw, hD, by omega⟩)⟩
      · simp only [hm, ↓reduceIte]
        refine ⟨fun P' h' => ?_, fun h' => by simp at h'⟩
        simp only [SV.path.injEq] at h'
        subst h'
        refine ⟨es, hw, hnodes, hD, hle, by omega, ?_⟩
        rw [← hnodes]
        exact (shortest_nodup (gs nf) s es d hw hD).1

end OpsTraverse.BspWire
