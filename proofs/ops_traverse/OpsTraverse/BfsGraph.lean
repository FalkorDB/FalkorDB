import OpsTraverse.AlgoSearch
import GraphQueries.Queries
/-
# `bfs_single_path` over the real relationship lists

`AlgoSearch.bspR` takes the forward and backward relationship lists as two functions, `rels` and
`relsRev`, and its theorems assume they agree (`hg` / `hg'`). Here both are instantiated with
graph_queries' model of the one Rust function that supplies them, so the agreement is proved.

| here | there |
| --- | --- |
| `relsG g types dir u` | `g.get_node_relationships_by_type(u, types, dir)` (graph.rs), modelled by `GQ.nodeRelsByType`: per selected tensor, `Tensor::iter(u, u, false)` (the row of `u`, outgoing) and/or `Tensor::iter(u, u, true)` (the column of `u`, incoming; with both halves, minus `src == u`). Both halves read the same `(src, dst, id)` entries of one tensor (`GQ.Ten`), so an edge reported at one endpoint in one direction is reported at the other endpoint in the reverse direction. |
| `gqDir` | `EdgeDirection` (the same three cases) |

`bfs_single_path` calls `get_node_relationships_by_type` with `config.rel_direction` on the forward
side and with its `reverse()` on the backward side (`revDir`), both with `config.rel_types`.
-/
namespace OpsTraverse.BfsGraph
open OpsTraverse.Search OpsTraverse.AlgoPaths OpsTraverse.RunAlgo OpsTraverse.AlgoSearch OpsTraverse.Enum

variable {F V : Type}

def gqDir : Dir → GQ.Dir
  | .outgoing => .outgoing
  | .incoming => .incoming
  | .both => .both

/-- `g.get_node_relationships_by_type(u, types, dir)`. -/
def relsG (g : GQ.G V) (types : List String) (dir : Dir) : Nat → List (Nat × Nat × Nat) :=
  fun u => GQ.nodeRelsByType g u types (gqDir dir)

theorem mem_relsG (g : GQ.G V) (types : List String) (dir : Dir) (u s d id : Nat) :
    (s, d, id) ∈ relsG g types dir u ↔ ∃ t ∈ GQ.selMats g types, (s, d, id) ∈ t ∧
      match dir with
      | .outgoing => s = u
      | .incoming => d = u
      | .both => s = u ∨ (d = u ∧ s ≠ u) := by
  cases dir <;>
    simp only [relsG, gqDir, GQ.nodeRelsByType, GQ.oneByType, GQ.Ten.out, GQ.Ten.inc, List.mem_flatMap,
      List.mem_filter, List.mem_append, beq_iff_eq, bne_iff_ne, ne_eq] <;>
    grind

/-- **PROVEN** (the two calls of `get_node_relationships_by_type` in `bfs_single_path` agree): a
relationship reported at `u` in direction `dir` whose far end (`far_endpoint`) is `v` is reported
at `v` in the reversed direction. -/
theorem relsG_converse (g : GQ.G V) (types : List String) (dir : Dir) (u v s d id : Nat)
    (h : (s, d, id) ∈ relsG g types dir u) (hf : farEndpoint u s d dir = some v) :
    (s, d, id) ∈ relsG g types (revDir dir) v := by
  obtain ⟨t, ht, hm, hc⟩ := (mem_relsG g types dir u s d id).mp h
  refine (mem_relsG g types (revDir dir) v s d id).mpr ⟨t, ht, hm, ?_⟩
  cases dir <;> simp only [revDir, farEndpoint] at hc hf ⊢ <;> grind

/-- `hg` of `AlgoSearch.bspR_valid`, at the real lists. -/
theorem relsG_hg (g : GQ.G V) (types : List String) (dir : Dir) :
    ∀ u v s d id, (s, d, id) ∈ relsG g types (revDir dir) u → farEndpoint u s d (revDir dir) = some v →
      (s, d, id) ∈ relsG g types dir v := by
  intro u v s d id h hf
  have := relsG_converse g types (revDir dir) u v s d id h hf
  rwa [revDir_revDir] at this

/-- `hg'` of `AlgoSearch.bspR_opt`, at the real lists. -/
theorem relsG_hg' (g : GQ.G V) (types : List String) (dir : Dir) :
    ∀ u v s d id, (s, d, id) ∈ relsG g types dir u → farEndpoint u s d dir = some v →
      (s, d, id) ∈ relsG g types (revDir dir) v :=
  relsG_converse g types dir

/-- The two lists `bfs_single_path` reads: forward and backward, both with `config.rel_types`. -/
abbrev bspG (o : Fl F) (cfg : Cfg F) (a : Attrs F) (g : GQ.G V) (t fuel : Nat) :=
  bspR o cfg a (relsG g cfg.relTypes cfg.dir) (relsG g cfg.relTypes (revDir cfg.dir)) t fuel

/-- **PROVEN** (`bfs_single_path`, soundness, nothing assumed about the graph): for a target other
than the source, a finished run's path is a non-empty walk over `node_successors` edges from the
source to the target with distinct nodes, weight its hop count and cost the left fold of `costProp`. -/
theorem bspG_valid (o : Fl F) (cfg : Cfg F) (a : Attrs F) (g : GQ.G V) (t fuel : Nat) (ht : t ≠ cfg.source)
    (es : List (Nat × Edge)) (w c : F) (h : bspG o cfg a g t fuel = some (some (es, w, c))) :
    WalkTo (stepC cfg (relsG g cfg.relTypes cfg.dir)) cfg.source es t ∧ (nodesOf cfg.source es).Nodup ∧
      es ≠ [] ∧ w = o.ofInt es.length ∧ c = fsum o.add o.zero (cattrD o cfg a) es :=
  bspR_valid o cfg a _ _ (relsG_hg g cfg.relTypes cfg.dir) t fuel ht es w c h

/-- **PROVEN** (`bfs_single_path`, optimality and completeness, nothing assumed about the graph): for
a target other than the source, a finished run's path has the fewest edges of any walk, and `Ok(None)`
comes back only when there is no walk. -/
theorem bspG_opt (o : Fl F) (cfg : Cfg F) (a : Attrs F) (g : GQ.G V) (t fuel : Nat) (ht : t ≠ cfg.source) :
    (∀ es w c, bspG o cfg a g t fuel = some (some (es, w, c)) →
      ∀ es', WalkTo (stepC cfg (relsG g cfg.relTypes cfg.dir)) cfg.source es' t → es.length ≤ es'.length) ∧
    (bspG o cfg a g t fuel = some none →
      ∀ es', ¬ WalkTo (stepC cfg (relsG g cfg.relTypes cfg.dir)) cfg.source es' t) :=
  bspR_opt o cfg a _ _ (relsG_hg' g cfg.relTypes cfg.dir) t fuel ht

theorem getRelMat_mem (g : GQ.G V) (n : String) (t : GQ.Ten) (h : GQ.getRelMat g n = some t) :
    t ∈ g.relMs := by
  unfold GQ.getRelMat at h
  split at h
  · obtain ⟨i, _, hi⟩ := Option.bind_eq_some_iff.mp h
    exact List.mem_of_getElem? hi
  · cases h

/-- **PROVEN** (`run_path_algo` end to end over the real relationship lists): `AlgoSearch.runAlgoR_good`
with `hg` discharged. `hfin`: `usize as f64` is finite (every `u64` is below `f64::MAX`); `hb`: node
ids are `u64` (a `NodeId` is a `u64`, which `GQ.Ten`'s `Nat` entries do not record). -/
theorem runAlgoG_good (o : Fl F) (cfg : Cfg F) (a : Attrs F) (g : GQ.G V)
    (hfin : ∀ n : Nat, n < 2 ^ 64 → o.fin (o.ofInt n) = true)
    (hb : ∀ t ∈ g.relMs, ∀ x ∈ t, x.1 < 2 ^ 64 ∧ x.2.1 < 2 ^ 64) (hsrc : cfg.source < 2 ^ 64)
    (fuel : Nat) {R : Type} (build : List (Fnd Edge F) → R) (out : R)
    (h : runAlgo cfg o.inf o.le (dijE o cfg a (relsG g cfg.relTypes cfg.dir) fuel)
      (bspE o cfg a (relsG g cfg.relTypes cfg.dir) (relsG g cfg.relTypes (revDir cfg.dir)) fuel)
      (bfsE o cfg a (relsG g cfg.relTypes cfg.dir)) (enumE o cfg a (relsG g cfg.relTypes cfg.dir) fuel)
      (sortR o) build = .ok out) :
    ∃ L, out = build L ∧ ∀ x ∈ L, Good o cfg a (relsG g cfg.relTypes cfg.dir) x := by
  refine runAlgoR_good o cfg a _ _ (relsG_hg g cfg.relTypes cfg.dir) hfin ?_ hsrc fuel build out h
  intro u x hx
  obtain ⟨s, d, id⟩ := x
  obtain ⟨t, ht, hm, _⟩ := (mem_relsG g cfg.relTypes cfg.dir u s d id).mp hx
  have ht' : t ∈ g.relMs := by
    unfold GQ.selMats at ht
    split at ht
    · exact ht
    · obtain ⟨n, _, hn⟩ := List.mem_filterMap.mp ht
      exact getRelMat_mem g n t hn
  exact hb t ht' _ hm

end OpsTraverse.BfsGraph
