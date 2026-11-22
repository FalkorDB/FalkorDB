/-
# `select_scan_node.rs` helpers (origin/main 8743953a8, after #2390 d2c42e032)

The plan is a rose tree; an orx-tree `NodeIdx` is modelled by its structural
path (child indices from the root), which is exactly what `node_path` computes.

| here | there |
| --- | --- |
| `IRk`, `PT`, `getAt` | `IR`, `DynTree<IR>`, `plan.node(idx)` |
| `scoreEndpoint`, `startKeyLt` | `score_endpoint` 110, the start-point comparator in `select_scan_node` 911-931 |
| `Ex`, `exprVars`, `propertyVar`, `collectEqualities` | `collect_equalities` 170 (#3821) |
| `collectFilteredVars`, `spineVars`, `eqLoop`, `spineEqs`, `filteredVars` | `collect_filtered_vars` 210 (`above` / `all` / `equality`), its `collect` 214 (= `filterVars`, `filterEqOf`) |
| `makeScanSubtree`, `filtersOf` | `make_scan_subtree` 276 (argument a whole `DynTree` since #3594), `filters_of` 312 |
| `argumentLeafOf` | `argument_leaf_of` 338 |
| `inMergeMatchBranch` | `in_merge_match_branch` 357 |
| `scanDepth`, `isPlannerScan`, `plannerScanAlias`, `plannerScanIdx` | `is_planner_scan_subtree` 377, `planner_scan_alias` 396, `planner_scan_idx` 417 |
| `projectedOutputs`, `copiesOf`, `collectOutputPairs`, `collectOutputAliases` | `projected_outputs` 459, `collect_output_pairs` 484, `collect_output_aliases` 502 (#3594) |
| `nodePath`, `resolvePath` | `node_path` 542, `resolve_path` 559 |
| `childSubtreeBinds` | `child_subtree_binds` 575 |
| `vlNonLeaf`, `vlLeaf`, `reach`, `vl_reverse_sound` | `select_var_len_scan_node` 632; `subtreeMayBind`: `subtree_may_bind` 606 |

Table lines re-mapped to PR #3821 (f71981fa3), which adds `collect_equalities` and the
`has_equality` tie-break; line ranges inside older docstrings cite the revision they were written for.
-/
namespace OptimizerScan.ScanTree

abbrev Var := Nat × Nat   -- (id, scope_id)

inductive IRk
  | allNodeScan (alias : Var)
  | labelScan (alias : Var) (labels : List Nat)
  | filter (vars : List Var)
  | includePending
  | argument (vars : Option (List Var))
  | condTraverse
  | condVarLen
  | pathBuilder
  | merge
  | project (exprs : List Var) (copies : List (Var × Var))   -- a copy is `(old, new)`
  | aggregate (names : List Var) (projections : List (Var × Var))
  | unwind (v : Var)
  | indexScan (alias : Var)
  | fulltextScan (node : Var) (score : Option Var)
  | other
  deriving DecidableEq, Repr

inductive PT where
  | node (ir : IRk) (cs : List PT)

def PT.ir : PT → IRk | .node ir _ => ir
def PT.cs : PT → List PT | .node _ cs => cs

def getAt : PT → List Nat → Option PT
  | t, [] => some t
  | .node _ cs, i :: p => match cs[i]? with
    | some c => getAt c p
    | none => none

/-! ## `score_endpoint` -/

/-- An endpoint as `score_endpoint` reads it. Since #2390 the inline attributes are no longer read
    (the planner lowers them to `Filter`s and strips them from the pattern). -/
structure EP where
  alias : Var
  labels : List Nat

/-- `FilteredVars` (`select_scan_node.rs`): variables of the Filters *above* the chain,
    `above` plus those of the Filters on the chain's own single-child spine, and (#3821) `equality`:
    the variables some of those Filters pin with a top-level `v.prop = <variable-free expr>`. -/
structure FV where
  above : List Var
  all : List Var
  equality : List Var

/-- `score_endpoint` (`select_scan_node.rs`):
    `(score, has_equality, filter_runs_late, cardinality)`. -/
def scoreEndpoint (n : EP) (fv : FV) (bound : List Nat) (count : Nat → Nat) : Nat × Bool × Bool × Nat :=
  ((if n.alias.1 ∈ bound then 3 else 0) + (if n.alias ∈ fv.all then 2 else 0) +
   (if n.labels.isEmpty then 0 else 1),
   decide (n.alias ∈ fv.equality),
   decide (n.alias ∈ fv.above),
   match n.labels with
   | [] => 2 ^ 64 - 1
   | l :: ls => (ls.map count).foldl min (count l))

/-- **PROVEN** (score formula): bound +3, filtered anywhere around the chain +2, labelled +1; the
first tie-breaker flag says the endpoint is pinned by an equality, the second that its Filter sits
above the chain; cardinality is `u64::MAX` for an unlabelled node, else the minimum label count. -/
theorem scoreEndpoint_spec (n : EP) (fv : FV) (b : List Nat) (c : Nat → Nat) :
    (scoreEndpoint n fv b c).1 ≤ 6 ∧
    (n.alias.1 ∈ b → (scoreEndpoint n fv b c).1 ≥ 3) ∧
    (n.alias ∈ fv.all → (scoreEndpoint n fv b c).1 ≥ 2) ∧
    (n.labels ≠ [] → (scoreEndpoint n fv b c).1 ≥ 1) ∧
    (scoreEndpoint n fv b c).2.1 = decide (n.alias ∈ fv.equality) ∧
    (scoreEndpoint n fv b c).2.2.1 = decide (n.alias ∈ fv.above) ∧
    (n.labels = [] → (scoreEndpoint n fv b c).2.2.2 = 2 ^ 64 - 1) := by
  unfold scoreEndpoint
  refine ⟨by split <;> split <;> split <;> omega, fun h => ?_, fun h => ?_, fun h => ?_, rfl, rfl,
    fun h => by simp [h]⟩
  · simp only [h, ite_true]; omega
  · simp only [h, ite_true]; omega
  · have : n.labels.isEmpty = false := by cases hl : n.labels <;> simp_all
    simp only [this]; simp

/-- **PROVEN**: since #2390 a bound endpoint is never outscored by an unbound one (an unbound
endpoint scores at most 2 + 1 = 3), as the doc comment says. -/
theorem score_bound_dominant (n m : EP) (fv : FV) (b : List Nat) (c : Nat → Nat)
    (hn : n.alias.1 ∈ b) (hm : m.alias.1 ∉ b) :
    (scoreEndpoint m fv b c).1 ≤ (scoreEndpoint n fv b c).1 := by
  unfold scoreEndpoint
  simp only [hn, hm, ite_true, ite_false]
  split <;> split <;> split <;> split <;> omega

/-- **PROVEN** (#3821): when `equality ⊆ all` (which `collect_filtered_vars` guarantees,
`filteredVars_equality_sub_all`), an equality-pinned endpoint always gets the +2 of a filtered one,
so the new flag only breaks ties among filtered endpoints and never outranks the score. -/
theorem score_equality_filtered (n : EP) (fv : FV) (b : List Nat) (c : Nat → Nat)
    (hsub : ∀ v, v ∈ fv.equality → v ∈ fv.all) (h : (scoreEndpoint n fv b c).2.1 = true) :
    (scoreEndpoint n fv b c).1 ≥ 2 := by
  have he : n.alias ∈ fv.equality := by simpa [scoreEndpoint] using h
  exact (scoreEndpoint_spec n fv b c).2.2.1 (hsub _ he)

/-- The start-point comparison key of `select_scan_node` (`select_scan_node.rs`), compared
    lexicographically, larger wins: score, then (#3821) the equality flag, then the late-filter flag,
    then the lower cardinality (the remaining chain-position / `from` tie-breaks are not modelled). -/
def startKeyLt (a b : Nat × Bool × Bool × Nat) : Prop :=
  a.1 < b.1 ∨ (a.1 = b.1 ∧ (a.2.1 = false ∧ b.2.1 = true ∨ (a.2.1 = b.2.1 ∧
    (a.2.2.1 = false ∧ b.2.2.1 = true ∨ (a.2.2.1 = b.2.2.1 ∧ b.2.2.2 < a.2.2.2)))))

/-- **PROVEN** (#3821, the PR's motivating case): of two equally scored endpoints, the one pinned by
    an equality beats one that is not, whatever their late-filter flags and cardinalities. -/
theorem equality_beats_late (x y : EP) (fv : FV) (b : List Nat) (c : Nat → Nat)
    (hs : (scoreEndpoint x fv b c).1 = (scoreEndpoint y fv b c).1)
    (hx : x.alias ∈ fv.equality) (hy : y.alias ∉ fv.equality) :
    startKeyLt (scoreEndpoint y fv b c) (scoreEndpoint x fv b c) := by
  refine Or.inr ⟨hs.symm, Or.inl ⟨?_, ?_⟩⟩ <;> simp [scoreEndpoint, hx, hy]

/-! ## `collect_equalities` (#3821)

A filter expression as `collect_equalities` reads it: `And`, `Eq`, `Property` and `Variable` nodes,
everything else (`Or`, `Not`, literals, parameters, functions, …) as `other`. -/
inductive Ex where
  | and (cs : List Ex)
  | eq (cs : List Ex)
  | prop (cs : List Ex)
  | var (v : Var)
  | other (cs : List Ex)

/-- The `Variable`s of an expression (the BFS of `collect`, `select_scan_node.rs`). -/
def exprVars : Ex → List Var
  | .and cs | .eq cs | .prop cs | .other cs => exprVarsL cs
  | .var v => [v]
where exprVarsL : List Ex → List Var
  | [] => []
  | e :: es => exprVars e ++ exprVarsL es

/-- `property_var`: a `Property` with exactly one child, that child a `Variable`. -/
def propertyVar : Ex → Option Var
  | .prop [.var v] => some v
  | _ => none

/-- `collect_equalities` (`select_scan_node.rs`): descend through `And` only; an `Eq` with
    two children records `v` when one side is `v.prop` and the other references no variable
    (left side tried first). -/
def collectEqualities : Ex → List Var
  | .and cs => collectEqualitiesL cs
  | .eq [l, r] =>
    match propertyVar l with
    | some v => if exprVars r = [] then [v] else
        match propertyVar r with
        | some w => if exprVars l = [] then [w] else []
        | none => []
    | none =>
        match propertyVar r with
        | some w => if exprVars l = [] then [w] else []
        | none => []
  | _ => []
where collectEqualitiesL : List Ex → List Var
  | [] => []
  | e :: es => collectEqualities e ++ collectEqualitiesL es

theorem propertyVar_mem : ∀ (e : Ex) (v : Var), propertyVar e = some v → v ∈ exprVars e
  | .prop [.var w], v, h => by
    simp [propertyVar] at h; subst h; simp [exprVars, exprVars.exprVarsL]
  | .and _, _, h | .eq _, _, h | .var _, _, h | .other _, _, h => by simp [propertyVar] at h
  | .prop [], _, h => by simp [propertyVar] at h
  | .prop (.and _ :: _), _, h | .prop (.eq _ :: _), _, h | .prop (.prop _ :: _), _, h
  | .prop (.other _ :: _), _, h | .prop (.var _ :: _ :: _), _, h => by simp [propertyVar] at h

mutual
/-- **PROVEN**: every variable `collect_equalities` records is a variable of the filter, so the
    `equality` set is a subset of what `collect` adds to `above`/`all`. -/
theorem collectEqualities_sub : ∀ (e : Ex) (v : Var), v ∈ collectEqualities e → v ∈ exprVars e
  | .and cs, v, h => by
    simp only [collectEqualities, exprVars] at h ⊢; exact collectEqualitiesL_sub cs v h
  | .eq [l, r], v, h => by
    have hl := propertyVar_mem l; have hr := propertyVar_mem r
    simp only [exprVars, exprVars.exprVarsL, List.append_nil, List.mem_append]
    simp only [collectEqualities] at h
    split at h
    · rename_i w hw
      split at h
      · simp at h; subst h; exact Or.inl (hl _ hw)
      · split at h
        · rename_i u hu; split at h
          · simp at h; subst h; exact Or.inr (hr _ hu)
          · simp at h
        · simp at h
    · split at h
      · rename_i u hu; split at h
        · simp at h; subst h; exact Or.inr (hr _ hu)
        · simp at h
      · simp at h
  | .eq [], _, h | .eq [_], _, h | .eq (_ :: _ :: _ :: _), _, h | .prop _, _, h | .var _, _, h
  | .other _, _, h => by simp [collectEqualities] at h
theorem collectEqualitiesL_sub : ∀ (cs : List Ex) (v : Var),
    v ∈ collectEqualities.collectEqualitiesL cs → v ∈ exprVars.exprVarsL cs
  | [], _, h => by simp [collectEqualities.collectEqualitiesL] at h
  | e :: es, v, h => by
    simp only [collectEqualities.collectEqualitiesL, List.mem_append] at h
    simp only [exprVars.exprVarsL, List.mem_append]
    rcases h with h | h
    · exact Or.inl (collectEqualities_sub e v h)
    · exact Or.inr (collectEqualitiesL_sub es v h)
end

/-- **PROVEN**: an equality under anything but `And` (e.g. `OR`/`NOT`) pins nothing. -/
theorem collectEqualities_other (cs : List Ex) : collectEqualities (.other cs) = [] := by
  simp [collectEqualities]

/-- **PROVEN**: `v.p = $param` / `$param = v.p` / `c.p = 1 AND v.p = 2` are recorded; `v.p = w.q`
    (both sides read a variable) is not. -/
theorem collectEqualities_examples (v w : Var) (k : List Ex) (hk : exprVars.exprVarsL k = []) :
    collectEqualities (.eq [.prop [.var v], .other k]) = [v] ∧
    collectEqualities (.eq [.other k, .prop [.var v]]) = [v] ∧
    collectEqualities (.and [.eq [.prop [.var w], .other k], .eq [.prop [.var v], .other k]]) = [w, v] ∧
    collectEqualities (.eq [.prop [.var v], .prop [.var w]]) = [] := by
  simp [collectEqualities, collectEqualities.collectEqualitiesL, propertyVar, exprVars,
    exprVars.exprVarsL, hk]

/-! ## Paths: `node_path` / `resolve_path` -/

/-- `node_path`: push `sibling_idx` while a parent exists, then reverse. A node's
parent is the path minus its last step. -/
def nodePathAux (p acc : List Nat) : List Nat :=
  if h : p = [] then acc else nodePathAux p.dropLast (acc ++ [p.getLast h])
termination_by p.length
decreasing_by
  have : p.length > 0 := List.length_pos_iff.mpr h
  simp [List.length_dropLast]; omega

def nodePath (p : List Nat) : List Nat := (nodePathAux p []).reverse

theorem nodePathAux_spec : ∀ (n : Nat) (p acc : List Nat), p.length = n → nodePathAux p acc = acc ++ p.reverse
  | 0, p, acc, h => by
    have : p = [] := List.length_eq_zero_iff.mp h
    subst this; simp [nodePathAux]
  | n + 1, p, acc, h => by
    have hne : p ≠ [] := by intro e; subst e; simp at h
    obtain ⟨q, i, rfl⟩ := List.eq_nil_or_concat p |>.resolve_left hne
    rw [nodePathAux, dite_eq_right (by simp)]
    rw [nodePathAux_spec n (q.concat i).dropLast _ (by simp at h ⊢; omega)]
    simp

/-- **PROVEN**: `node_path` computes the structural path. -/
theorem nodePath_eq (p : List Nat) : nodePath p = p := by
  simp [nodePath, nodePathAux_spec p.length p [] rfl]

/-- `resolve_path`: walk `get_child` from the root. -/
def resolvePath (t : PT) (p : List Nat) : Option (List Nat) := (getAt t p).map (fun _ => p)

/-- **PROVEN** (round trip): resolving the path of an existing node gives that
node back; a path into a removed subtree gives `None`. -/
theorem resolve_nodePath (t : PT) (p : List Nat) :
    resolvePath t (nodePath p) = if (getAt t p).isSome then some p else none := by
  rw [nodePath_eq]; unfold resolvePath; cases getAt t p <;> simp

/-- Removing child `i` of the node at `p` (`prune`). -/
def pruneAt : PT → List Nat → Nat → PT
  | .node ir cs, [], i => .node ir (cs.eraseIdx i)
  | .node ir cs, j :: p, i => .node ir (cs.modify j (fun c => pruneAt c p i))

/-- "pruning a child never changes the parent's path" (`select_scan_node.rs`). -/
theorem pruneAt_parent : ∀ (t : PT) (p : List Nat) (i : Nat),
    (getAt t p).isSome → (getAt (pruneAt t p i) p).isSome
  | .node _ _, [], _, _ => by simp [getAt, pruneAt]
  | .node ir cs, j :: p, i, h => by
    simp only [getAt, pruneAt] at h ⊢
    cases hc : cs[j]? with
    | none => simp [hc] at h
    | some c =>
      rw [hc] at h
      have : (cs.modify j (fun c => pruneAt c p i))[j]? = some (pruneAt c p i) := by
        rw [List.getElem?_modify]; simp [hc]
      rw [this]; exact pruneAt_parent c p i h

/-! ## Single-child chains -/

def isWrapper : IRk → Bool
  | .filter _ | .includePending => true
  | _ => false
def scanAlias : IRk → Option Var
  | .allNodeScan a | .labelScan a _ => some a
  | _ => none

/-- Depth of the scan at the bottom of a `Filter`/`IncludePending` chain
(`None` if the shape does not match). Shared by the three walkers. -/
def scanDepth : PT → Option Nat
  | .node ir cs =>
    match scanAlias ir with
    | some _ => some 0
    | none => if isWrapper ir then (match cs with | [c] => (scanDepth c).map (· + 1) | _ => none) else none

def isPlannerScan (t : PT) : Bool := (scanDepth t).isSome
def plannerScanIdx (t : PT) : Option (List Nat) := (scanDepth t).map (fun k => List.replicate k 0)
def plannerScanAlias (t : PT) : Option Var :=
  (plannerScanIdx t).bind (fun p => (getAt t p).bind (fun s => scanAlias s.ir))

theorem scanDepth_spec : ∀ (t : PT) (k : Nat), scanDepth t = some k →
    ∃ s, getAt t (List.replicate k 0) = some s ∧ (scanAlias s.ir).isSome
  | .node ir cs, k, h => by
    unfold scanDepth at h
    cases ha : scanAlias ir with
    | some a => simp [ha] at h; subst h; exact ⟨_, rfl, by simp [PT.ir, ha]⟩
    | none =>
      simp only [ha] at h
      split at h
      · split at h
        · next c =>
          cases hd : scanDepth c with
          | none => simp [hd] at h
          | some k' =>
            simp [hd] at h; subst h
            obtain ⟨s, hs, hsa⟩ := scanDepth_spec c k' hd
            exact ⟨s, by simp [List.replicate_succ, getAt, hs], hsa⟩
        · cases h
      · cases h

/-- **PROVEN**: the three walkers agree — a planner scan subtree has a scan at
`planner_scan_idx`, whose alias is `planner_scan_alias`. -/
theorem plannerScan_agree (t : PT) :
    (isPlannerScan t = (plannerScanIdx t).isSome) ∧ (isPlannerScan t = (plannerScanAlias t).isSome) ∧
    ∀ p, plannerScanIdx t = some p → ∃ s, getAt t p = some s ∧ plannerScanAlias t = scanAlias s.ir := by
  refine ⟨by simp [isPlannerScan, plannerScanIdx], ?_, ?_⟩
  · unfold isPlannerScan plannerScanAlias plannerScanIdx
    cases hd : scanDepth t with
    | none => simp
    | some k =>
      obtain ⟨s, hs, hsa⟩ := scanDepth_spec t k hd
      simp [hs, Option.isSome_iff_exists] at hsa ⊢
      exact hsa
  · intro p hp
    unfold plannerScanIdx at hp
    cases hd : scanDepth t with
    | none => simp [hd] at hp
    | some k =>
      simp [hd] at hp; subst hp
      obtain ⟨s, hs, -⟩ := scanDepth_spec t k hd
      exact ⟨s, hs, by simp [plannerScanAlias, plannerScanIdx, hd, hs]⟩

/-- `argument_leaf_of`: follow single children until an `Argument`. -/
def argumentLeafOf : PT → Option IRk
  | .node ir cs =>
    match ir with
    | .argument v => some (.argument v)
    | _ => match cs with | [c] => argumentLeafOf c | _ => none

theorem argumentLeafOf_spec : ∀ (t : PT) (a : IRk), argumentLeafOf t = some a →
    ∃ k v, a = .argument v ∧ (getAt t (List.replicate k 0)).map PT.ir = some (.argument v)
  | .node ir cs, a, h => by
    unfold argumentLeafOf at h
    split at h
    · next v => cases h; exact ⟨0, v, rfl, rfl⟩
    · split at h
      · next c =>
        obtain ⟨k, v, rfl, hk⟩ := argumentLeafOf_spec c a h
        exact ⟨k + 1, v, rfl, by simpa [List.replicate_succ, getAt] using hk⟩
      · cases h

/-! ## `make_scan_subtree` and `filters_of` -/

/-- `make_scan_subtree` (`select_scan_node.rs`): scan, optional subtree beneath it (since
    #3594 a whole `DynTree`: an `Argument` leaf, or an outer `Unwind`/`Project` subtree, attached with
    `push_child_tree`), optional `IncludePending`, then the salvaged `filters` wrapped innermost-first
    (`filters[0]` outermost). -/
def makeScanSubtree (alias : Var) (labels : List Nat) (pending : Bool)
    (argument : Option PT) (filters : List (List Var)) : PT :=
  let scan := if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels
  let s0 := PT.node scan argument.toList
  let s1 := if pending then PT.node .includePending [s0] else s0
  filters.foldr (fun vs t => PT.node (.filter vs) [t]) s1

/-- `filters_of` (`select_scan_node.rs`): the `Filter`s on a subtree's single-child spine,
    outermost first. -/
def filtersOf : PT → List (List Var)
  | .node ir cs =>
    (match ir with | .filter vs => [vs] | _ => []) ++ (match cs with | [c] => filtersOf c | _ => [])

theorem plannerScanAlias_filter (alias : Var) (vs : List Var) (t : PT) (ht : plannerScanAlias t = some alias) :
    plannerScanAlias (PT.node (.filter vs) [t]) = some alias := by
  simp only [plannerScanAlias, plannerScanIdx] at ht
  cases hd : scanDepth t with
  | none => simp [hd] at ht
  | some k =>
    simp only [hd, Option.map_some, Option.bind_some] at ht
    have e : scanDepth (PT.node (.filter vs) [t]) = some (k + 1) := by
      simp [scanDepth, scanAlias, isWrapper, hd]
    simp only [plannerScanAlias, plannerScanIdx, e, Option.map_some, Option.bind_some, List.replicate_succ]
    exact ht

/-- **PROVEN**: `make_scan_subtree` builds exactly the shape the planner-scan walkers accept, with the
node's alias at the bottom, whatever filters are salvaged onto it. -/
theorem makeScanSubtree_planner (alias : Var) (labels : List Nat) (pend : Bool)
    (arg : Option PT) (fs : List (List Var)) :
    plannerScanAlias (makeScanSubtree alias labels pend arg fs) = some alias := by
  unfold makeScanSubtree
  have hs : ∀ cs, plannerScanAlias (PT.node (if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels) cs) = some alias := by
    intro cs
    split <;> rfl
  have hp : ∀ t, plannerScanAlias t = some alias → plannerScanAlias (PT.node .includePending [t]) = some alias := by
    intro t ht
    simp only [plannerScanAlias, plannerScanIdx] at ht
    cases hd : scanDepth t with
    | none => simp [hd] at ht
    | some k =>
      simp only [hd, Option.map_some, Option.bind_some] at ht
      have e : scanDepth (PT.node .includePending [t]) = some (k + 1) := by
        simp [scanDepth, scanAlias, isWrapper, hd]
      simp only [plannerScanAlias, plannerScanIdx, e, Option.map_some, Option.bind_some, List.replicate_succ]
      exact ht
  have h1 : plannerScanAlias (if pend then PT.node .includePending
      [PT.node (if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels)
        arg.toList] else
      PT.node (if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels)
        arg.toList) = some alias := by
    split
    · exact hp _ (hs _)
    · exact hs _
  induction fs with
  | nil => exact h1
  | cons vs fs ih => exact plannerScanAlias_filter alias vs _ ih

/-- **PROVEN** (salvaging loses nothing): re-wrapping salvaged filters around any subtree puts them back
    on its spine, outermost first, in their original order. -/
theorem filtersOf_wrap (fs : List (List Var)) (t : PT) :
    filtersOf (fs.foldr (fun vs t => PT.node (.filter vs) [t]) t) = fs ++ filtersOf t := by
  induction fs with
  | nil => rfl
  | cons vs fs ih => simp [filtersOf, ih]

/-- ...so the rebuilt scan subtree carries exactly the Filters `filters_of` took off the old one
    (the scan itself and `IncludePending` add none). -/
theorem filtersOf_makeScanSubtree (alias : Var) (labels : List Nat) (pend : Bool) (v : Option (List Var))
    (fs : List (List Var)) :
    filtersOf (makeScanSubtree alias labels pend (some (.node (.argument v) [])) fs) = fs ∧
    filtersOf (makeScanSubtree alias labels pend none fs) = fs := by
  unfold makeScanSubtree
  rw [filtersOf_wrap, filtersOf_wrap]
  constructor <;> (cases pend <;> cases labels.isEmpty <;> simp [filtersOf])

/-- **PROVEN** (#3594, the argument is a whole subtree): `filters_of` walks through the new scan into
    the subtree beneath it (the scan has one child), so the rebuilt subtree's spine carries the
    salvaged filters followed by the Filters on that subtree's own spine; the scan and
    `IncludePending` add none. -/
theorem filtersOf_makeScanSubtree_tree (alias : Var) (labels : List Nat) (pend : Bool) (t : PT)
    (fs : List (List Var)) :
    filtersOf (makeScanSubtree alias labels pend (some t) fs) = fs ++ filtersOf t := by
  unfold makeScanSubtree
  rw [filtersOf_wrap]
  cases pend <;> cases labels.isEmpty <;> simp [filtersOf]

/-! ## Ancestor walks -/

/-- Ancestors of `p`, nearest first (`parent()` repeatedly). -/
def ancestors (p : List Nat) : List (List Nat) :=
  (List.range p.length).reverse.map (fun k => p.take k)

def transparent : IRk → Bool
  | .condTraverse | .condVarLen | .pathBuilder => true
  | _ => false

/-- The `collect_filtered_vars` loop over the ancestors' IR, nearest first. -/
def collectLoop : List IRk → List Var
  | [] => []
  | .filter vs :: rest => vs ++ collectLoop rest
  | ir :: rest => if transparent ir then collectLoop rest else []

/-- The upward walk of `collect_filtered_vars` (`select_scan_node.rs`): `above`. -/
def collectFilteredVars (t : PT) (p : List Nat) : List Var :=
  collectLoop ((ancestors p).filterMap (fun q => (getAt t q).map PT.ir))

def contOK : IRk → Bool
  | .filter _ => true
  | ir => transparent ir

/-- The nested `collect` (`select_scan_node.rs`): every `Variable` of the filter expression,
    which is how a `Filter` is modelled here (`IRk.filter vars`). -/
def filterVars : IRk → List Var
  | .filter vs => vs
  | _ => []

/-- **PROVEN**: the variables collected upwards are those of the `Filter`s in the
maximal run of Filter / CondTraverse / CondVarLenTraverse / PathBuilder
ancestors directly above the node. -/
theorem collectLoop_eq (l : List IRk) : collectLoop l = (l.takeWhile contOK).flatMap filterVars := by
  induction l with
  | nil => rfl
  | cons ir rest ih =>
    cases ir <;> simp [collectLoop, contOK, transparent, filterVars, List.takeWhile_cons, ih]

/-- The downward walk (`select_scan_node.rs`): from `start` itself, through
    Filter / CondTraverse / CondVarLenTraverse / PathBuilder nodes while they have one child. -/
def spineVars : PT → List Var
  | .node ir cs =>
    if contOK ir then filterVars ir ++ (match cs with | [c] => spineVars c | _ => []) else []

/-- What `collect_equalities` records for one IR node: `eqOf` of a `Filter` (its expression's
    pinned variables, `filterEqOf`), nothing for any other operator. -/
def filterEqs (eqOf : List Var → List Var) : IRk → List Var
  | .filter vs => eqOf vs
  | _ => []

/-- The upward loop's `equality` collection (`select_scan_node.rs`): same walk as `collectLoop`. -/
def eqLoop (eqOf : List Var → List Var) : List IRk → List Var
  | [] => []
  | .filter vs :: rest => eqOf vs ++ eqLoop eqOf rest
  | ir :: rest => if transparent ir then eqLoop eqOf rest else []

/-- The downward loop's `equality` collection (`select_scan_node.rs`): same walk as `spineVars`. -/
def spineEqs (eqOf : List Var → List Var) : PT → List Var
  | .node ir cs =>
    if contOK ir then filterEqs eqOf ir ++ (match cs with | [c] => spineEqs eqOf c | _ => []) else []

/-- `collect_filtered_vars` (`select_scan_node.rs`). A `Filter` is modelled by its variable
    list `vs`; `eqOf vs` is what `collect_equalities` records for its expression (instantiated by
    `filterEqOf`, an expression per filter). -/
def filteredVars (eqOf : List Var → List Var) (t : PT) (p : List Nat) : FV :=
  let above := collectFilteredVars t p
  let anc := (ancestors p).filterMap (fun q => (getAt t q).map PT.ir)
  { above := above, all := above ++ ((getAt t p).map spineVars).getD [],
    equality := eqLoop eqOf anc ++ ((getAt t p).map (spineEqs eqOf)).getD [] }

/-- `eqOf` is honest when each Filter's recorded equalities are among its variables — true of
    `collect_equalities` on the filter's expression (`filterEqOf_honest`). -/
def EqHonest (eqOf : List Var → List Var) : Prop := ∀ vs v, v ∈ eqOf vs → v ∈ vs

/-- A Filter whose expression is `E vs` (so `vs = exprVars (E vs)`, what `collect` gathers). -/
def filterEqOf (E : List Var → Ex) : List Var → List Var := fun vs => collectEqualities (E vs)

theorem filterEqOf_honest (E : List Var → Ex) (hE : ∀ vs, exprVars (E vs) = vs) :
    EqHonest (filterEqOf E) := by
  intro vs v h
  have := collectEqualities_sub (E vs) v h
  rwa [hE] at this

theorem eqLoop_sub (eqOf : List Var → List Var) (hE : EqHonest eqOf) :
    ∀ (l : List IRk) (v : Var), v ∈ eqLoop eqOf l → v ∈ collectLoop l
  | [], _, h => by simp [eqLoop] at h
  | ir :: rest, v, h => by
    cases ir with
    | filter vs =>
      simp only [eqLoop, List.mem_append] at h
      simp only [collectLoop, List.mem_append]
      rcases h with h | h
      · exact Or.inl (hE _ _ h)
      · exact Or.inr (eqLoop_sub eqOf hE rest v h)
    | _ =>
      simp only [eqLoop] at h; simp only [collectLoop]
      split at h
      · rename_i ht; simp only [ht, ite_true]; exact eqLoop_sub eqOf hE rest v h
      · simp at h

theorem spineEqs_sub (eqOf : List Var → List Var) (hE : EqHonest eqOf) :
    ∀ (t : PT) (v : Var), v ∈ spineEqs eqOf t → v ∈ spineVars t
  | .node ir cs, v, h => by
    unfold spineEqs at h; unfold spineVars
    split at h
    · rename_i hc
      simp only [hc, ite_true, List.mem_append] at h ⊢
      rcases h with h | h
      · left; cases ir <;> simp_all [filterEqs, filterVars]; exact hE _ _ h
      · right
        match cs, h with
        | [c], h => exact spineEqs_sub eqOf hE c v h
    · simp at h

/-- **PROVEN** (#3821): every variable in `equality` is in `all` — an equality-pinned endpoint is a
    filtered one (so `score_equality_filtered` applies to `collect_filtered_vars`' output). -/
theorem filteredVars_equality_sub_all (eqOf : List Var → List Var) (hE : EqHonest eqOf)
    (t : PT) (p : List Nat) (v : Var) (h : v ∈ (filteredVars eqOf t p).equality) :
    v ∈ (filteredVars eqOf t p).all := by
  simp only [filteredVars, collectFilteredVars, List.mem_append] at h ⊢
  rcases h with h | h
  · exact Or.inl (eqLoop_sub eqOf hE _ v h)
  · right
    cases hg : getAt t p with
    | none => simp [hg] at h
    | some u => simp only [hg, Option.map_some, Option.getD_some] at h ⊢; exact spineEqs_sub eqOf hE u v h

/-- The spine the downward walk visits: the start node and its single-child descendants, while
    each is a Filter or a transparent traverse. -/
def spine : PT → List IRk
  | .node ir cs => if contOK ir then ir :: (match cs with | [c] => spine c | _ => []) else []

/-- **PROVEN**: the downward walk collects exactly the variables of the Filters on that spine, and
    `all` is `above` plus them (so `above ⊆ all`: a filter above the chain still counts as filtering). -/
theorem spineVars_eq : ∀ (t : PT), spineVars t = (spine t).flatMap filterVars
  | .node ir cs => by
    unfold spineVars spine
    split
    · cases cs with
      | nil => simp
      | cons c cs => cases cs with
        | nil => simp [spineVars_eq c]
        | cons _ _ => simp
    · rfl

theorem filteredVars_above_sub (eqOf : List Var → List Var) (t : PT) (p : List Nat) (v : Var)
    (h : v ∈ (filteredVars eqOf t p).above) :
    v ∈ (filteredVars eqOf t p).all := by
  simp only [filteredVars] at h ⊢; exact List.mem_append_left _ h

/-- `in_merge_match_branch`: the child of the nearest `Merge` ancestor on the
path is that `Merge`'s last child. `chain` = (ancestor IR, its child count,
sibling index of the step taken from it), nearest first. -/
def inMergeLoop : List (IRk × Nat × Nat) → Bool
  | [] => false
  | (ir, n, i) :: rest => if ir = .merge then i == n - 1 else inMergeLoop rest

theorem inMergeLoop_spec (l : List (IRk × Nat × Nat)) :
    inMergeLoop l = true ↔ ∃ pre n i rest, l = pre ++ (.merge, n, i) :: rest ∧
      (∀ x ∈ pre, x.1 ≠ .merge) ∧ i = n - 1 := by
  induction l with
  | nil => simp [inMergeLoop]
  | cons x rest ih =>
    obtain ⟨ir, n, i⟩ := x
    simp only [inMergeLoop]
    split
    · next h =>
      subst h
      simp only [beq_iff_eq]
      constructor
      · intro h; exact ⟨[], n, i, rest, by simp, by simp, h⟩
      · rintro ⟨pre, n', i', rest', he, hpre, hi⟩
        cases pre with
        | nil => simp at he; obtain ⟨⟨rfl, rfl⟩, rfl⟩ := he; exact hi
        | cons y pre => simp at he; exact absurd he.1 (by intro h'; exact hpre y (by simp) (h' ▸ rfl))
    · next h =>
      rw [ih]
      constructor
      · rintro ⟨pre, n', i', rest', rfl, hpre, hi⟩
        exact ⟨(ir, n, i) :: pre, n', i', rest', by simp, by
          intro x hx; rcases List.mem_cons.mp hx with rfl | hx
          · exact h
          · exact hpre x hx, hi⟩
      · rintro ⟨pre, n', i', rest', he, hpre, hi⟩
        cases pre with
        | nil => simp at he; exact absurd he.1.1 h
        | cons y pre =>
          simp at he; obtain ⟨rfl, rfl⟩ := he
          exact ⟨pre, n', i', rest', rfl, fun x hx => hpre x (by simp [hx]), hi⟩

/-! ## `collect_output_aliases` and `child_subtree_binds` -/

/-- `projected_outputs` (`select_scan_node.rs`, #3594): the `(id, scope)` a projection or a
    known `Argument` writes — a `Project`'s expression aliases and each copy's `new` end, an
    `Aggregate`'s names and each projection's `new` end, a known `Argument`'s variables. -/
def projectedOutputs : IRk → List Var
  | .project es cs => es ++ cs.map Prod.snd
  | .aggregate ns ps => ns ++ ps.map Prod.snd
  | .argument (some vs) => vs
  | _ => []

/-- `collect_output_pairs` (`select_scan_node.rs`, #3594): `projected_outputs` plus each
    copy's source `old`, carried with its scope. -/
def copiesOf : IRk → List (Var × Var)
  | .project _ cs => cs
  | .aggregate _ ps => ps
  | _ => []

def collectOutputPairs (ir : IRk) : List Var :=
  projectedOutputs ir ++ (copiesOf ir).map Prod.fst

/-- `collect_output_aliases` (`select_scan_node.rs`); since #3594 a `Project`, `Aggregate` or
    known `Argument` contributes the ids of `projected_outputs`. -/
def collectOutputAliases : IRk → List Nat
  | .allNodeScan a | .labelScan a _ | .indexScan a => [a.1]
  | .fulltextScan n s => n.1 :: s.toList.map (·.1)
  | .project es cs => (projectedOutputs (.project es cs)).map (·.1)
  | .aggregate ns ps => (projectedOutputs (.aggregate ns ps)).map (·.1)
  | .unwind v => [v.1]
  | .argument (some vs) => (projectedOutputs (.argument (some vs))).map (·.1)
  | _ => []

theorem collectOutputAliases_spec (a : Var) (vs : List Var) (s : Option Var) :
    collectOutputAliases (.labelScan a []) = [a.1] ∧ collectOutputAliases (.indexScan a) = [a.1] ∧
    collectOutputAliases (.project vs []) = vs.map (·.1) ∧ collectOutputAliases (.argument none) = [] ∧
    collectOutputAliases (.argument (some vs)) = vs.map (·.1) ∧
    collectOutputAliases (.fulltextScan a s) = a.1 :: s.toList.map (·.1) ∧
    collectOutputAliases .other = [] := by
  simp [collectOutputAliases, projectedOutputs]

/-- **PROVEN** (#3594): a copy `(old, new)` contributes its `new` end and never its source; an
    aggregation contributes its names and its copies' `new` ends. -/
theorem collectOutputAliases_aggregate (es ns : List Var) (cs : List (Var × Var)) :
    collectOutputAliases (.project es cs) = es.map (·.1) ++ cs.map (·.2.1) ∧
    collectOutputAliases (.aggregate ns cs) = ns.map (·.1) ++ cs.map (·.2.1) := by
  simp [collectOutputAliases, projectedOutputs]

/-- **PROVEN** (#3594): on every operator that projects (`Project`, `Aggregate`, known `Argument`)
    `collect_output_aliases` is exactly the ids of `projected_outputs`, so the id helper and the
    scoped helper cannot disagree about what such an operator outputs. -/
theorem collectOutputAliases_projected (ir : IRk) (h : projectedOutputs ir ≠ []) :
    collectOutputAliases ir = (projectedOutputs ir).map (·.1) := by
  cases ir with
  | argument v => cases v with
    | none => simp [projectedOutputs] at h
    | some vs => rfl
  | project | aggregate => rfl
  | _ => simp [projectedOutputs] at h

/-- **PROVEN** (#3594): `collect_output_pairs` is `projected_outputs` plus the copies' sources, and
    a pair it lists either is a projected output (whose id `collect_output_aliases` reports) or is the
    source of a copy, with its scope. The Rust unit test
    `output_aliases_cover_copies_of_projections_and_aggregations` checks one instance. -/
theorem collectOutputPairs_spec (ir : IRk) (v : Var) :
    (∀ w ∈ projectedOutputs ir, w ∈ collectOutputPairs ir) ∧
    (v ∈ collectOutputPairs ir →
      (v ∈ projectedOutputs ir ∧ (projectedOutputs ir ≠ [] → v.1 ∈ collectOutputAliases ir)) ∨
      ∃ new : Var, (v, new) ∈ copiesOf ir) := by
  refine ⟨fun w hw => List.mem_append_left _ hw, fun hv => ?_⟩
  simp only [collectOutputPairs, List.mem_append] at hv
  rcases hv with hv | hv
  · refine Or.inl ⟨hv, fun hne => ?_⟩
    rw [collectOutputAliases_projected ir hne]
    exact List.mem_map.mpr ⟨v, hv, rfl⟩
  · right
    obtain ⟨⟨o, n⟩, hm, rfl⟩ := List.mem_map.mp hv
    exact ⟨n, hm⟩

mutual
def PT.walk : PT → List IRk
  | .node ir cs => ir :: PT.walkL cs
def PT.walkL : List PT → List IRk
  | [] => []
  | c :: cs => c.walk ++ PT.walkL cs
end

/-- `child_subtree_binds`; `getVars c` = the `GetVariables` of a child subtree. -/
def childBindsLoop (getVars : PT → List Var) (alias : Var) : List PT → Bool → Bool
  | [], b => b
  | c :: cs, b =>
    if c.walk.contains (.argument none) then false
    else childBindsLoop getVars alias cs (b || (getVars c).contains alias)

def childSubtreeBinds (getVars : PT → List Var) (t : PT) (alias : Var) : Bool :=
  childBindsLoop getVars alias t.cs false

theorem childBindsLoop_spec (gv : PT → List Var) (a : Var) : ∀ (cs : List PT) (b : Bool),
    childBindsLoop gv a cs b = true ↔
      (∀ c ∈ cs, IRk.argument none ∉ c.walk) ∧ (b = true ∨ ∃ c ∈ cs, a ∈ gv c)
  | [], b => by simp [childBindsLoop]
  | c :: cs, b => by
    simp only [childBindsLoop]
    split
    · next h => simp at h; simp [h]
    · next h =>
      rw [childBindsLoop_spec gv a cs]
      simp at h
      constructor
      · rintro ⟨h1, h2⟩
        refine ⟨fun c' hc' => ?_, ?_⟩
        · rcases List.mem_cons.mp hc' with rfl | hc'
          · exact h
          · exact h1 c' hc'
        · rcases h2 with h2 | ⟨c', hc', ha⟩
          · simp at h2; rcases h2 with h2 | h2
            · exact Or.inl h2
            · exact Or.inr ⟨c, by simp, h2⟩
          · exact Or.inr ⟨c', by simp [hc'], ha⟩
      · rintro ⟨h1, h2⟩
        refine ⟨fun c' hc' => h1 c' (by simp [hc']), ?_⟩
        rcases h2 with h2 | ⟨c', hc', ha⟩
        · simp [h2]
        · rcases List.mem_cons.mp hc' with rfl | hc'
          · simp [ha]
          · exact Or.inr ⟨c', hc', ha⟩

/-- **PROVEN**: `child_subtree_binds` is true iff no child subtree contains an
opaque `Argument(None)` and some child binds the alias (full `(id, scope)` pair). -/
theorem childSubtreeBinds_spec (gv : PT → List Var) (t : PT) (a : Var) :
    childSubtreeBinds gv t a = true ↔ (∀ c ∈ t.cs, IRk.argument none ∉ c.walk) ∧ ∃ c ∈ t.cs, a ∈ gv c := by
  unfold childSubtreeBinds; rw [childBindsLoop_spec]; simp

/-! ## `select_var_len_scan_node`: reversing a var-length leaf -/

/-- `k`-hop reachability in relation `R` (`R a b`: an edge of the pattern's
type from `a` to `b`). -/
inductive Reach (R : Nat → Nat → Prop) : Nat → Nat → Nat → Prop
  | zero (a : Nat) : Reach R 0 a a
  | step {k a m b : Nat} : R a m → Reach R k m b → Reach R (k + 1) a b

def tr (R : Nat → Nat → Prop) : Nat → Nat → Prop := fun a b => R b a

theorem Reach.snoc {R : Nat → Nat → Prop} : ∀ {k a m b : Nat}, Reach R k a m → R m b → Reach R (k + 1) a b
  | _, _, _, _, .zero _, h => .step h (.zero _)
  | _, _, _, _, .step h1 h2, h => .step h1 (Reach.snoc h2 h)

theorem Reach.reverse {R : Nat → Nat → Prop} : ∀ {k a b : Nat}, Reach R k a b → Reach (tr R) k b a
  | _, _, _, .zero a => .zero a
  | _, _, _, .step h1 h2 => Reach.snoc (Reach.reverse h2) h1

theorem tr_tr (R : Nat → Nat → Prop) : tr (tr R) = R := rfl

/-- **PROVEN**: walking the relationship backwards from `to` finds exactly the
`k`-hop paths walking forwards from `from` finds. -/
theorem reach_reverse_iff (R : Nat → Nat → Prop) (k a b : Nat) : Reach R k a b ↔ Reach (tr R) k b a :=
  ⟨Reach.reverse, fun h => by have := Reach.reverse h; rwa [tr_tr] at this⟩

/-- **PROVEN** (`select_var_len_scan_node` is sound): scanning `to` and walking
backwards (`CondVarLenTraverseOp` with a bound `to`, enforcing `from`'s labels as
the destination filter) binds exactly the `(from, to)` pairs of the original
plan, for any hop range. -/
theorem vl_reverse_sound (R : Nat → Nat → Prop) (Lf Lt : Nat → Prop) (lo hi : Nat) (a b : Nat) :
    (Lf a ∧ Lt b ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach R k a b) ↔
    (Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach (tr R) k b a) := by
  constructor
  · rintro ⟨h1, h2, k, hk1, hk2, hr⟩; exact ⟨h2, h1, k, hk1, hk2, Reach.reverse hr⟩
  · rintro ⟨h1, h2, k, hk1, hk2, hr⟩; exact ⟨h2, h1, k, hk1, hk2, (reach_reverse_iff R k a b).mpr hr⟩

/-- Undirected walks (#3593): the pattern relation is symmetric, so walking it backwards is
walking it forwards — `CondVarLenTraverseOp` reverses an undirected walk too (proofs/ops_traverse
`VarLen.varLen_bidir_reverse_iff`), and the rewrite is sound for it as for a directed one. -/
theorem vl_reverse_sound_symm (R : Nat → Nat → Prop) (hs : ∀ a b, R a b → R b a)
    (Lf Lt : Nat → Prop) (lo hi : Nat) (a b : Nat) :
    (Lf a ∧ Lt b ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach R k a b) ↔
    (Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach R k b a) := by
  have htr : tr R = R := by
    funext x y; apply propext; exact ⟨hs y x, hs x y⟩
  have := vl_reverse_sound R Lf Lt lo hi a b
  rwa [htr] at this

/-! ### `subtree_may_bind` (#3593) -/

/-- `subtree_may_bind` (`select_scan_node.rs`): some child of the scan holds an opaque
`Argument(None)` or binds `alias` (its `GetVariables`, full `(id, scope)` pair). -/
def subtreeMayBind (getVars : PT → List Var) (t : PT) (alias : Var) : Bool :=
  t.cs.any fun c => c.walk.contains (.argument none) || (getVars c).contains alias

/-- **PROVEN**: `subtree_may_bind` is true iff a child holds an `Argument(None)` or binds `alias`. -/
theorem subtreeMayBind_spec (gv : PT → List Var) (t : PT) (a : Var) :
    subtreeMayBind gv t a = true ↔ ∃ c ∈ t.cs, IRk.argument none ∈ c.walk ∨ a ∈ gv c := by
  simp [subtreeMayBind]

/-- **PROVEN**: whatever `child_subtree_binds` proves binding, `subtree_may_bind` counts too; and an
opaque `Argument` counts for `subtree_may_bind` where `child_subtree_binds` gives up. -/
theorem childSubtreeBinds_mayBind (gv : PT → List Var) (t : PT) (a : Var)
    (h : childSubtreeBinds gv t a = true) : subtreeMayBind gv t a = true := by
  obtain ⟨_, c, hc, ha⟩ := (childSubtreeBinds_spec gv t a).1 h
  exact (subtreeMayBind_spec gv t a).2 ⟨c, hc, Or.inr ha⟩

/-! ### The rewrite decision -/

/-- What `select_var_len_scan_node` (`select_scan_node.rs`) inspects. `bidirectional` and
`scanOnlyArgBelow` are read only by the history (`vlGuards3264`, `vlLeaf3264`): #3593 dropped both
refusals. -/
structure VLCase where
  expandInto : Bool
  bidirectional : Bool
  allShortest : Bool
  sameEnds : Bool
  oneChild : Bool
  childIsFromScan : Bool      -- `planner_scan_alias(child) == from` (looks through Filter/IncludePending)
  wrapperHasPending : Bool    -- an `IncludePending` between the child and the scan
  wrapperHasFilter : Bool     -- a `Filter` between the child and the scan
  childBindsTo : Bool         -- `child_subtree_binds(scan, to)`
  scanHasChildren : Bool      -- `kept` non-empty
  scanOnlyArgBelow : Bool     -- the scan is a leaf or has exactly one childless `Argument` child (#3264)
  argBindsTo : Bool
  argOpaque : Bool
  fromScore : Nat
  toScore : Nat
  attrsReadFrom : Bool := false   -- the edge attributes read `from` (#3593, :634-639)
  attrsReadTo : Bool := false     -- the edge attributes read `to`
  belowMayBindTo : Bool := false  -- `subtree_may_bind(scan, to)` (#3593, :749)

/-- Since #3593 an undirected walk is no longer refused (:624-630). -/
def vlGuards (c : VLCase) : Bool :=
  !c.expandInto && !c.allShortest && !c.sameEnds && c.oneChild && c.childIsFromScan

/-- The non-leaf rewrite (`select_scan_node.rs`): drop the planner's scan *and its wrapper
    chain*, keeping what is under the scan. A `Filter` in the wrapper refuses it (#2390), and since
    #3593 so do edge attributes reading `from` (which the drop unbinds). -/
def vlNonLeaf (c : VLCase) : Bool :=
  vlGuards c && !c.wrapperHasPending && !c.wrapperHasFilter && !c.attrsReadFrom && c.childBindsTo &&
    c.scanHasChildren

/-- The leaf rewrite (`select_scan_node.rs`): scan `to` instead, re-hanging *every* child of
    the planner's scan under the new scan (`kept`, :755-759 and :782-785). Refused when the edge
    attributes read an endpoint, the `Argument` leaf is opaque or binds `to`, the wrapper chain holds
    a `Filter`, or anything below may bind `to` (`subtree_may_bind`). -/
def vlLeaf (c : VLCase) : Bool :=
  vlGuards c && !vlNonLeaf c && !c.attrsReadFrom && !c.attrsReadTo && !c.argOpaque && !c.argBindsTo &&
    !c.wrapperHasFilter && !c.belowMayBindTo && decide (c.fromScore < c.toScore)

/-- The scan the leaf rewrite builds (:776-788): `to`'s scan over the old scan's children, wrapped in
`IncludePending` inside MERGE. -/
def leafScan (to : Var) (labels : List Nat) (kept : List PT) (inMerge : Bool) : PT :=
  let scan := if labels.isEmpty then PT.node (.allNodeScan to) kept else PT.node (.labelScan to labels) kept
  if inMerge then .node .includePending [scan] else scan

/-- The scan under the (optional) `IncludePending`. -/
def leafScanCore (t : PT) : PT :=
  match t with
  | .node .includePending [s] => s
  | t => t

/-- **PROVEN** (#3593): the new scan keeps every child of the planner's scan, in order — nothing that
fed the old scan (an earlier clause, an `UNWIND`, an `Argument`) is pruned. -/
theorem leafScan_keeps_below (to : Var) (labels : List Nat) (kept : List PT) (m : Bool) :
    (leafScanCore (leafScan to labels kept m)).cs = kept := by
  cases m <;> cases labels <;> simp [leafScan, leafScanCore, PT.cs]

/-! #### History: the earlier guard set (#3264), kept for `vlLeaf_eq_3264` -/

def vlGuards3264 (c : VLCase) : Bool :=
  !c.expandInto && !c.bidirectional && !c.allShortest && !c.sameEnds && c.oneChild && c.childIsFromScan

def vlNonLeaf3264 (c : VLCase) : Bool :=
  vlGuards3264 c && !c.wrapperHasPending && !c.wrapperHasFilter && c.childBindsTo && c.scanHasChildren

/-- The leaf rewrite as of #3264: carried over only a childless `Argument`. -/
def vlLeaf3264 (c : VLCase) : Bool :=
  vlGuards3264 c && !vlNonLeaf3264 c && !c.wrapperHasFilter && c.scanOnlyArgBelow &&
    !c.argOpaque && !c.argBindsTo && decide (c.fromScore < c.toScore)

/-- The leaf rewrite before #3264 (history, base 343e077f2): no `Filter` / below-scan check. -/
def vlLeafPre (c : VLCase) : Bool :=
  vlGuards3264 c && !vlNonLeaf3264 c && !c.argOpaque && !c.argBindsTo && decide (c.fromScore < c.toScore)

/-- **PROVEN** (#2390): the non-leaf rewrite never drops a `Filter` (nor an `IncludePending`). -/
theorem vlNonLeaf_keeps_wrappers (c : VLCase) (h : vlNonLeaf c = true) :
    c.wrapperHasFilter = false ∧ c.wrapperHasPending = false := by
  simp [vlNonLeaf] at h; exact ⟨h.1.1.1.2, h.1.1.1.1.2⟩

/-- **PROVEN** (#3593): the non-leaf rewrite unbinds `from`, so it never fires when the edge attributes
read `from`. -/
theorem vlNonLeaf_attrs (c : VLCase) (h : vlNonLeaf c = true) : c.attrsReadFrom = false := by
  simp [vlNonLeaf] at h; exact h.1.1.2

/-- The leaf rewrite fires only when every guard holds (ties keep the direction). Since #3593 an
undirected walk is no longer a refusal (`vlLeaf_bidir`, sound by `vl_reverse_sound_symm`); the
history statement is `vlLeaf3264_guards`. -/
theorem vlLeaf_guards (c : VLCase) (h : vlLeaf c = true) :
    c.expandInto = false ∧ c.allShortest = false ∧ c.sameEnds = false ∧
    c.childIsFromScan = true ∧ c.argBindsTo = false ∧ c.argOpaque = false ∧ c.fromScore < c.toScore ∧
    c.attrsReadFrom = false ∧ c.attrsReadTo = false ∧ c.belowMayBindTo = false := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp_all [vlLeaf, vlGuards]

/-- History (#3264): the leaf rewrite as it stood refused an undirected walk. -/
theorem vlLeaf3264_guards (c : VLCase) (h : vlLeaf3264 c = true) :
    c.expandInto = false ∧ c.bidirectional = false ∧ c.allShortest = false ∧ c.sameEnds = false ∧
    c.childIsFromScan = true ∧ c.argBindsTo = false ∧ c.argOpaque = false ∧ c.fromScore < c.toScore := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp_all [vlLeaf3264, vlGuards3264]

/-- **PROVEN** (#3264, #3593): the leaf rewrite never prunes a `Filter` wrapper and never rescans `to`
over a subtree that may bind it; everything under the old scan is kept (`leafScan_keeps_below`). -/
theorem vlLeaf_keeps_filter (c : VLCase) (h : vlLeaf c = true) :
    c.wrapperHasFilter = false ∧ c.belowMayBindTo = false := by
  cases hf : c.wrapperHasFilter <;> cases ho : c.belowMayBindTo <;> simp_all [vlLeaf]

/-- **PROVEN** (#3264): neither rewrite of `select_var_len_scan_node` drops a `Filter` wrapper. -/
theorem vl_never_drops_filter (c : VLCase) (h : vlNonLeaf c = true ∨ vlLeaf c = true) :
    c.wrapperHasFilter = false := by
  rcases h with h | h
  · exact (vlNonLeaf_keeps_wrappers c h).1
  · exact (vlLeaf_keeps_filter c h).1

/-- History (#3264): that rewrite was the pre-#3264 one plus the two #3264 refusals. -/
theorem vlLeaf_eq_pre (c : VLCase) :
    vlLeaf3264 c = (vlLeafPre c && !c.wrapperHasFilter && c.scanOnlyArgBelow) := by
  simp only [vlLeaf3264, vlLeafPre]; cases vlGuards3264 c <;> cases vlNonLeaf3264 c <;>
    cases c.wrapperHasFilter <;> cases c.scanOnlyArgBelow <;> cases c.argOpaque <;> cases c.argBindsTo <;>
    simp

/-- **PROVEN** (#3593 conservative on what #3264 rewrote): on a directed walk whose edge attributes
read neither endpoint, with only a childless `Argument` under the scan (so `subtree_may_bind` is
that `Argument` being opaque or binding `to`), the rewrite decides exactly as #3264's did. -/
theorem vlLeaf_eq_3264 (c : VLCase) (hb : c.bidirectional = false) (hf : c.attrsReadFrom = false)
    (ht : c.attrsReadTo = false) (ho : c.scanOnlyArgBelow = true)
    (hm : c.belowMayBindTo = (c.argOpaque || c.argBindsTo)) :
    vlLeaf c = vlLeaf3264 c := by
  simp only [vlLeaf, vlLeaf3264, vlNonLeaf, vlNonLeaf3264, vlGuards, vlGuards3264, hb, hf, ht, ho, hm]
  cases c.expandInto <;> cases c.allShortest <;> cases c.sameEnds <;> cases c.oneChild <;>
    cases c.childIsFromScan <;> cases c.wrapperHasPending <;> cases c.wrapperHasFilter <;>
    cases c.childBindsTo <;> cases c.scanHasChildren <;> cases c.argOpaque <;> cases c.argBindsTo <;> simp

/-- **PROVEN** (#3593): the leaf rewrite now reverses an undirected walk, which #3264 refused. -/
def vlBidirCase : VLCase :=
  { expandInto := false, bidirectional := true, allShortest := false, sameEnds := false,
    oneChild := true, childIsFromScan := true, wrapperHasPending := false, wrapperHasFilter := false,
    childBindsTo := false, scanHasChildren := false, scanOnlyArgBelow := true, argBindsTo := false,
    argOpaque := false, fromScore := 0, toScore := 1 }

theorem vlLeaf_bidir : vlLeaf vlBidirCase = true ∧ vlLeaf3264 vlBidirCase = false := by
  decide

/-- **PROVEN** (#3593): an operator under the scan (an earlier clause, an `UNWIND`) no longer
refuses the leaf rewrite; it is kept under the new scan (`leafScan_keeps_below`). -/
def vlInputCase : VLCase :=
  { expandInto := false, bidirectional := false, allShortest := false, sameEnds := false,
    oneChild := true, childIsFromScan := true, wrapperHasPending := false, wrapperHasFilter := false,
    childBindsTo := false, scanHasChildren := true, scanOnlyArgBelow := false, argBindsTo := false,
    argOpaque := false, fromScore := 0, toScore := 1 }

theorem vlLeaf_over_input : vlLeaf vlInputCase = true ∧ vlLeaf3264 vlInputCase = false := by
  decide

/-- **PROVEN** (#3593): `MATCH (b:B) WHERE (b)-[:X]->() MATCH (a)-[:R*1..2]->(b:B)`: the pattern
predicate's sub-plan under the scan binds `b` but holds an opaque `Argument`, so
`child_subtree_binds` cannot prove it and the non-leaf rewrite is out; `subtree_may_bind` counts it,
so the leaf rewrite does not rescan `b` over it (which would lose the correlation). -/
theorem vl_leaf_never_rebinds_to (gv : PT → List Var) (t : PT) (to : Var)
    (c : VLCase) (hm : c.belowMayBindTo = subtreeMayBind gv t to)
    (hb : ∃ ch ∈ t.cs, IRk.argument none ∈ ch.walk ∨ to ∈ gv ch) : vlLeaf c = false := by
  have : c.belowMayBindTo = true := by rw [hm]; exact (subtreeMayBind_spec gv t to).2 hb
  simp [vlLeaf, this]

/-! ### The leaf rewrite and a wrapper `Filter`

`MATCH (a) WHERE a.v = 1 MATCH (a)-[*1..2]->(b:B {w:2}) RETURN a.v, b.w`: clause 1's
`Filter(a.v = 1) → AllNodeScan(a)` is stitched in as the var-length traverse's child, which
`planner_scan_alias` accepts as the planner's scan of `from = a` (it looks through `Filter`). `a`
scores 2 (filtered, via the downward spine of `collect_filtered_vars`), `b` scores 3 (labelled, its
lowered `{w:2}` Filter above the traverse), so a leaf rewrite that prunes the child scans `b` and walks
backwards, and the pruned child takes `a.v = 1` with it: graph
`(:A {v:1})-[:R]->(:B {w:2}), (:A {v:5})-[:R]->(:B {w:2})` would return `[1, 2], [5, 2]` instead of `[1, 2]`.
The current rewrite refuses the leaf rewrite when the wrapper chain holds a `Filter`. -/

def vlBugCase : VLCase :=
  { expandInto := false, bidirectional := false, allShortest := false, sameEnds := false, oneChild := true,
    childIsFromScan := true, wrapperHasPending := false, wrapperHasFilter := true, childBindsTo := false,
    scanHasChildren := false, scanOnlyArgBelow := true, argBindsTo := false, argOpaque := false,
    fromScore := (scoreEndpoint ⟨(0, 0), []⟩ ⟨[(1, 0)], [(1, 0), (0, 0)], []⟩ [] (fun _ => 1)).1,
    toScore := (scoreEndpoint ⟨(1, 0), [7]⟩ ⟨[(1, 0)], [(1, 0), (0, 0)], []⟩ [] (fun _ => 1)).1 }

/-- History: the earlier rewrite (`vlLeafPre`) fired on the repro although the wrapper holds `a.v = 1`. -/
theorem vl_leaf_fires_over_filter : vlLeafPre vlBugCase = true ∧ vlBugCase.wrapperHasFilter = true := by
  decide

/-- **PROVEN** (#3264 fixes it, still so after #3593): the leaf rewrite refuses the repro and keeps
`a.v = 1`. -/
theorem vl_leaf_bug_case_fixed : vlLeaf vlBugCase = false := by
  decide

/-- What the rewritten plan selects (`Lt b`, `Lf a`, the backward walk) differs from the original
    (which also tests the dropped `F a`) exactly on the rows where `F a` is false — `(a.v = 5, b)`. -/
theorem vl_leaf_drops_filter (R : Nat → Nat → Prop) (Lf Lt F : Nat → Prop) (lo hi a b : Nat)
    (hF : ¬ F a) (hsel : Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach (tr R) k b a) :
    (Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach (tr R) k b a) ∧
    ¬ (Lf a ∧ F a ∧ Lt b ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach R k a b) := ⟨hsel, fun h => hF h.2.1⟩

end OptimizerScan.ScanTree
