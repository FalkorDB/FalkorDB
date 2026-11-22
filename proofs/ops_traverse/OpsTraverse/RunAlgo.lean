import OpsTraverse.AlgoPaths
/-
# `run_path_algo` (algo_procedures.rs:2710): which search runs, and what it is seeded with

Glue over the three searches. `dij` is the result of `dijkstra_single_path`, `bfsb` of
`bfs_find_bound`, `enum bound` of `enumerate_paths bound`, `sortF` is `sort_by(cmp_found_path)` and
`build` is `build_path_batch`; the theorem says exactly what is returned in terms of them and assumes
nothing about any of them (their own correctness is separate).
-/
namespace OpsTraverse.RunAlgo
open OpsTraverse.AlgoPaths

variable {F R P : Type}

/-- `run_path_algo`. `inf` is `f64::INFINITY`, `leF` is `<=`. -/
def runAlgo (cfg : Cfg F) (inf : F) (leF : F → F → Bool)
    (dij : Nat → Except String (Option P)) (bfsb : Nat → Except String (Option (F × F)))
    (enum : F → Except String (List P)) (sortF : List P → List P) (build : List P → R) :
    Except String R :=
  match cfg.target with
  | some t =>
    if cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t then do
      let found ← dij t
      pure (build found.toList)
    else do
      match ← bfsb t with
      | none => pure (build [])
      | some (bw, bc) =>
        let bound := if (cfg.pathCount == 0 || cfg.pathCount == 1) &&
            (cfg.maxCost.all (fun m => leF bc m)) then bw else inf
        let rs ← enum bound
        pure (build (sortF rs))
  | none => do
    let rs ← enum inf
    pure (build (sortF rs))

/-- **PROVEN** (`run_path_algo`, glue; the searches are parameters): with a target, one cheapest
path, no `maxCost`, no `maxLen` and `source != target` it is Dijkstra alone; otherwise with a target
it first asks `bfs_find_bound` (`None` answers an empty batch with no enumeration), seeds the
enumeration bound with the BFS path's weight only for `pathCount` 0 or 1 and when its cost respects
`maxCost` (else infinity), and sorts the enumerated paths; without a target (SSpaths) it enumerates
from an infinite bound and sorts. -/
theorem runAlgo_spec (cfg : Cfg F) (inf : F) (leF : F → F → Bool)
    (dij : Nat → Except String (Option P)) (bfsb : Nat → Except String (Option (F × F)))
    (enum : F → Except String (List P)) (sortF : List P → List P) (build : List P → R) :
    (∀ t, cfg.target = some t → cfg.pathCount = 1 → cfg.maxCost.isNone = true →
        cfg.maxLen = u32max → cfg.source ≠ t →
        runAlgo cfg inf leF dij bfsb enum sortF build = (dij t).bind (fun f => .ok (build f.toList))) ∧
    (∀ t, cfg.target = some t → ¬ (cfg.pathCount = 1 ∧ cfg.maxCost.isNone = true ∧
          cfg.maxLen = u32max ∧ cfg.source ≠ t) →
        runAlgo cfg inf leF dij bfsb enum sortF build = (bfsb t).bind (fun b => match b with
          | none => .ok (build [])
          | some (bw, bc) =>
            (enum (if (cfg.pathCount == 0 || cfg.pathCount == 1) &&
                (cfg.maxCost.all (fun m => leF bc m)) then bw else inf)).bind
              (fun rs => .ok (build (sortF rs))))) ∧
    (cfg.target = none →
        runAlgo cfg inf leF dij bfsb enum sortF build = (enum inf).bind (fun rs => .ok (build (sortF rs)))) := by
  refine ⟨fun t ht h1 h2 h3 h4 => ?_, fun t ht hn => ?_, fun ht => ?_⟩
  · have : (cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t) = true := by
      simp [h1, h2, h3, h4]
    simp [runAlgo, ht, this, bind, Except.bind, pure, Except.pure]
  · have : (cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t) = false := by
      cases h : (cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t)
      · rfl
      · exfalso; apply hn
        simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
        exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩
    simp only [runAlgo, ht, this, Bool.false_eq_true, ↓reduceIte]
    cases bfsb t with
    | error e => rfl
    | ok b => cases b <;> rfl
  · simp [runAlgo, ht, bind, Except.bind, pure, Except.pure]

end OpsTraverse.RunAlgo
