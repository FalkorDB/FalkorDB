import OpsTraverse.AlgoPaths
/-
# `run_path_algo` (algo_procedures.rs): which search runs, and what it is seeded with

Glue over the four searches. `dij` is the result of `dijkstra_single_path`, `bsp` of `bfs_single_path`
(the unit-weight fast path: no `weightProp`), `bfsb` of
`bfs_find_bound`, `enum bound` of `enumerate_paths bound`, `sortF` is `sort_by(cmp_found_path)` and
`build` is `build_path_batch`; the theorem says exactly what is returned in terms of them and assumes
nothing about any of them (their own correctness is separate).
-/
namespace OpsTraverse.RunAlgo
open OpsTraverse.AlgoPaths

variable {F R P : Type}

/-- `run_path_algo`. `inf` is `f64::INFINITY`, `leF` is `<=`. -/
def runAlgo (cfg : Cfg F) (inf : F) (leF : F → F → Bool)
    (dij : Nat → Except String (Option P)) (bsp : Nat → Except String (Option P))
    (bfsb : Nat → Except String (Option (F × F)))
    (enum : F → Except String (List P)) (sortF : List P → List P) (build : List P → R) :
    Except String R :=
  match cfg.target with
  | some t =>
    if cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t then do
      let found ← if cfg.weightProp.isNone then bsp t else dij t
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
path, no `maxCost`, no `maxLen` and `source != target` it is `bfs_single_path` alone without
`weightProp` and Dijkstra alone with it; otherwise with a target
it first asks `bfs_find_bound` (`None` answers an empty batch with no enumeration), seeds the
enumeration bound with the BFS path's weight only for `pathCount` 0 or 1 and when its cost respects
`maxCost` (else infinity), and sorts the enumerated paths; without a target (SSpaths) it enumerates
from an infinite bound and sorts. -/
theorem runAlgo_spec (cfg : Cfg F) (inf : F) (leF : F → F → Bool)
    (dij : Nat → Except String (Option P)) (bsp : Nat → Except String (Option P))
    (bfsb : Nat → Except String (Option (F × F)))
    (enum : F → Except String (List P)) (sortF : List P → List P) (build : List P → R) :
    (∀ t, cfg.target = some t → cfg.pathCount = 1 → cfg.maxCost.isNone = true →
        cfg.maxLen = u32max → cfg.source ≠ t → cfg.weightProp.isSome = true →
        runAlgo cfg inf leF dij bsp bfsb enum sortF build = (dij t).bind (fun f => .ok (build f.toList))) ∧
    (∀ t, cfg.target = some t → cfg.pathCount = 1 → cfg.maxCost.isNone = true →
        cfg.maxLen = u32max → cfg.source ≠ t → cfg.weightProp = none →
        runAlgo cfg inf leF dij bsp bfsb enum sortF build = (bsp t).bind (fun f => .ok (build f.toList))) ∧
    (∀ t, cfg.target = some t → ¬ (cfg.pathCount = 1 ∧ cfg.maxCost.isNone = true ∧
          cfg.maxLen = u32max ∧ cfg.source ≠ t) →
        runAlgo cfg inf leF dij bsp bfsb enum sortF build = (bfsb t).bind (fun b => match b with
          | none => .ok (build [])
          | some (bw, bc) =>
            (enum (if (cfg.pathCount == 0 || cfg.pathCount == 1) &&
                (cfg.maxCost.all (fun m => leF bc m)) then bw else inf)).bind
              (fun rs => .ok (build (sortF rs))))) ∧
    (cfg.target = none →
        runAlgo cfg inf leF dij bsp bfsb enum sortF build = (enum inf).bind (fun rs => .ok (build (sortF rs)))) := by
  refine ⟨fun t ht h1 h2 h3 h4 hw => ?_, fun t ht h1 h2 h3 h4 hw => ?_, fun t ht hn => ?_, fun ht => ?_⟩
  · have : (cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t) = true := by
      simp [h1, h2, h3, h4]
    have hw' : cfg.weightProp.isNone = false := by cases h : cfg.weightProp <;> simp_all
    simp [runAlgo, ht, this, hw', bind, Except.bind, pure, Except.pure]
  · have : (cfg.pathCount == 1 && cfg.maxCost.isNone && cfg.maxLen == u32max && cfg.source != t) = true := by
      simp [h1, h2, h3, h4]
    simp [runAlgo, ht, this, hw, bind, Except.bind, pure, Except.pure]
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

/-- **PROVEN** (`run_path_algo`, what reaches `build_path_batch`): whatever the four searches are, if
they only ever produce `Good` paths (`hdij`: the Dijkstra path, `hbsp`: the unit-weight BFS path, which `run_path_algo` only asks for a target other than the source,
`henum`: every enumerated path) and
`sort_by` only reorders (`hsort`), then every batch `run_path_algo` returns is `build` of a list of
`Good` paths. With the searches instantiated by `AlgoSearch` this is the end-to-end statement. -/
theorem runAlgo_ok_good (cfg : Cfg F) (inf : F) (leF : F → F → Bool)
    (dij : Nat → Except String (Option P)) (bsp : Nat → Except String (Option P))
    (bfsb : Nat → Except String (Option (F × F)))
    (enum : F → Except String (List P)) (sortF : List P → List P) (build : List P → R) (Good : P → Prop)
    (hdij : ∀ t f, dij t = .ok (some f) → Good f)
    (hbsp : ∀ t f, cfg.source ≠ t → bsp t = .ok (some f) → Good f)
    (henum : ∀ b rs, enum b = .ok rs → ∀ x ∈ rs, Good x)
    (hsort : ∀ l x, x ∈ sortF l → x ∈ l) (out : R)
    (h : runAlgo cfg inf leF dij bsp bfsb enum sortF build = .ok out) :
    ∃ L, out = build L ∧ ∀ x ∈ L, Good x := by
  obtain ⟨h0, h1, h2, h3⟩ := runAlgo_spec cfg inf leF dij bsp bfsb enum sortF build
  cases ht : cfg.target with
  | none =>
    rw [h3 ht] at h
    cases he : enum inf with
    | error e => rw [he] at h; simp [bind, Except.bind] at h
    | ok rs =>
      rw [he] at h
      simp only [bind, Except.bind, Except.ok.injEq] at h
      exact ⟨sortF rs, h.symm, fun x hx => henum _ _ he x (hsort _ x hx)⟩
  | some t =>
    by_cases hg : cfg.pathCount = 1 ∧ cfg.maxCost.isNone = true ∧ cfg.maxLen = u32max ∧ cfg.source ≠ t
    · have one : ∀ (s : Nat → Except String (Option P)), (∀ f, s t = .ok (some f) → Good f) →
          runAlgo cfg inf leF dij bsp bfsb enum sortF build = (s t).bind (fun f => .ok (build f.toList)) →
          ∃ L, out = build L ∧ ∀ x ∈ L, Good x := by
        intro s hs he
        rw [he] at h
        cases hd : s t with
        | error e => rw [hd] at h; simp [bind, Except.bind] at h
        | ok f =>
          rw [hd] at h
          simp only [bind, Except.bind, Except.ok.injEq] at h
          refine ⟨f.toList, h.symm, fun x hx => ?_⟩
          cases f with
          | none => simp at hx
          | some f => simp at hx; subst hx; exact hs _ hd
      cases hw : cfg.weightProp with
      | none => exact one bsp (fun f => hbsp t f hg.2.2.2) (h1 t ht hg.1 hg.2.1 hg.2.2.1 hg.2.2.2 hw)
      | some _ => exact one dij (fun f => hdij t f) (h0 t ht hg.1 hg.2.1 hg.2.2.1 hg.2.2.2 (by simp [hw]))
    · rw [h2 t ht hg] at h
      cases hb : bfsb t with
      | error e => rw [hb] at h; simp [bind, Except.bind] at h
      | ok b =>
        rw [hb] at h
        cases b with
        | none =>
          simp only [bind, Except.bind, Except.ok.injEq] at h
          exact ⟨[], h.symm, fun x hx => by simp at hx⟩
        | some bc =>
          obtain ⟨bw, bcost⟩ := bc
          simp only [bind, Except.bind] at h
          cases he : enum (if (cfg.pathCount == 0 || cfg.pathCount == 1) && (cfg.maxCost.all (fun m => leF bcost m)) then bw else inf) with
          | error e => rw [he] at h; simp at h
          | ok rs =>
            rw [he] at h
            simp only [Except.ok.injEq] at h
            exact ⟨sortF rs, h.symm, fun x hx => henum _ _ he x (hsort _ x hx)⟩

end OpsTraverse.RunAlgo
