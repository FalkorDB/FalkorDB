/-
# `select_scan_node.rs` helpers (origin/main 8743953a8, after #2390 d2c42e032; re-checked against PR #3099 8a40e2f60)

The plan is a rose tree; an orx-tree `NodeIdx` is modelled by its structural
path (child indices from the root), which is exactly what `node_path` computes.

| here | there |
| --- | --- |
| `IRk`, `PT`, `getAt` | `IR`, `DynTree<IR>`, `plan.node(idx)` |
| `scoreEndpoint` | `score_endpoint` 98 |
| `collectFilteredVars`, `spineVars`, `filteredVars` | `collect_filtered_vars` 156 (`above` / `all`), its `collect` 160 (= `filterVars`) |
| `makeScanSubtree` | `make_scan_subtree` 212 (#3099: no salvaged filters) |
| `filtersOf` | the spine's `Filter`s; was `filters_of` (removed by #3099), kept for `Salvage` (pre-#3099 history) |
| `argumentLeafOf` | `argument_leaf_of` 241 |
| `inMergeMatchBranch` | `in_merge_match_branch` 260 |
| `outerChildBoundVars` | `outer_child_bound_vars` 283 (new in #3099) |
| `scanDepth`, `isPlannerScan`, `plannerScanAlias`, `plannerScanIdx` | `planner_scan_alias` 348, `planner_scan_idx` 367 (`isPlannerScan` was `is_planner_scan_subtree`, removed by #3099) |
| `collectOutputAliases` | `collect_output_aliases` 405 |
| `nodePath`, `resolvePath` | `node_path` 464, `resolve_path` 481 |
| `childSubtreeBinds` | `child_subtree_binds` 497 |
| `vlNonLeaf`, `vlLeaf`, `reach`, `vl_reverse_sound` | `select_var_len_scan_node` 532 |
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
  | project (outs : List Var)
  | aggregate (names : List Var)
  | unwind (v : Var)
  | indexScan (alias : Var)          -- NodeByIndexScan / NodeByLabelAndIdScan / NodeByIdSeek
  | fulltextScan (node : Var) (score : Option Var)
  | vectorScan (node : Var) (score : Option Var)
  | procedureCall (yields : List Var)
  | optional (vars : List Var)
  | limit
  | skip
  | sort
  | distinct
  | commit
  | cartesianProduct
  | valueHashJoin
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

/-- `FilteredVars` (`select_scan_node.rs:145-154`): variables of the Filters *above* the chain, and
    `above` plus those of the Filters on the chain's own single-child spine. -/
structure FV where
  above : List Var
  all : List Var

/-- `score_endpoint` (`select_scan_node.rs:98-132`): `(score, filter_runs_late, cardinality)`. -/
def scoreEndpoint (n : EP) (fv : FV) (bound : List Nat) (count : Nat → Nat) : Nat × Bool × Nat :=
  ((if n.alias.1 ∈ bound then 3 else 0) + (if n.alias ∈ fv.all then 2 else 0) +
   (if n.labels.isEmpty then 0 else 1),
   decide (n.alias ∈ fv.above),
   match n.labels with
   | [] => 2 ^ 64 - 1
   | l :: ls => (ls.map count).foldl min (count l))

/-- **PROVEN** (score formula): bound +3, filtered anywhere around the chain +2, labelled +1; the
tie-breaker flag says the endpoint's Filter sits above the chain; cardinality is `u64::MAX` for an
unlabelled node, else the minimum label count. -/
theorem scoreEndpoint_spec (n : EP) (fv : FV) (b : List Nat) (c : Nat → Nat) :
    (scoreEndpoint n fv b c).1 ≤ 6 ∧
    (n.alias.1 ∈ b → (scoreEndpoint n fv b c).1 ≥ 3) ∧
    (n.alias ∈ fv.all → (scoreEndpoint n fv b c).1 ≥ 2) ∧
    (n.labels ≠ [] → (scoreEndpoint n fv b c).1 ≥ 1) ∧
    (scoreEndpoint n fv b c).2.1 = decide (n.alias ∈ fv.above) ∧
    (n.labels = [] → (scoreEndpoint n fv b c).2.2 = 2 ^ 64 - 1) := by
  unfold scoreEndpoint
  refine ⟨by split <;> split <;> split <;> omega, fun h => ?_, fun h => ?_, fun h => ?_, rfl,
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
    rw [nodePathAux, dif_neg (by simp)]
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

/-- "pruning a child never changes the parent's path" (`select_scan_node.rs:612,663`). -/
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

/-! ## `make_scan_subtree`; the spine's `Filter`s -/

/-- `make_scan_subtree` (`select_scan_node.rs:212-230`, #3099): scan, optional `Argument` leaf,
    optional `IncludePending`. Since #3099 it takes no salvaged filters (nothing is salvaged: a
    non-leaf child is kept whole unless it is a transparent `Argument`). -/
def makeScanSubtree (alias : Var) (labels : List Nat) (pending : Bool)
    (argument : Option IRk) : PT :=
  let scan := if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels
  let s0 := PT.node scan (argument.toList.map (fun a => PT.node a []))
  if pending then PT.node .includePending [s0] else s0

/-- The `Filter`s on a subtree's single-child spine, outermost first (what `filters_of` returned
    before #3099 removed it; `Salvage.lean` keeps that history). -/
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
node's alias at the bottom, with or without `IncludePending` and an `Argument` leaf. -/
theorem makeScanSubtree_planner (alias : Var) (labels : List Nat) (pend : Bool)
    (arg : Option IRk) :
    plannerScanAlias (makeScanSubtree alias labels pend arg) = some alias := by
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
        (arg.toList.map (fun a => PT.node a []))] else
      PT.node (if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels)
        (arg.toList.map (fun a => PT.node a []))) = some alias := by
    split
    · exact hp _ (hs _)
    · exact hs _
  exact h1

/-- **PROVEN** (salvaging loses nothing): re-wrapping salvaged filters around any subtree puts them back
    on its spine, outermost first, in their original order. -/
theorem filtersOf_wrap (fs : List (List Var)) (t : PT) :
    filtersOf (fs.foldr (fun vs t => PT.node (.filter vs) [t]) t) = fs ++ filtersOf t := by
  induction fs with
  | nil => rfl
  | cons vs fs ih => simp [filtersOf, ih]

/-- **PROVEN**: the scan subtree `make_scan_subtree` builds carries no `Filter` (the scan, its
    `Argument` leaf and `IncludePending` add none); since #3099 no Filter is moved onto it, so a
    predicate is never relocated into a rebuilt scan. -/
theorem filtersOf_makeScanSubtree (alias : Var) (labels : List Nat) (pend : Bool) (v : Option (List Var)) :
    filtersOf (makeScanSubtree alias labels pend (some (.argument v))) = [] ∧
    filtersOf (makeScanSubtree alias labels pend none) = [] := by
  unfold makeScanSubtree
  constructor <;> (cases pend <;> cases labels.isEmpty <;> simp [filtersOf])

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

/-- The upward walk of `collect_filtered_vars` (`select_scan_node.rs:171-181`): `above`. -/
def collectFilteredVars (t : PT) (p : List Nat) : List Var :=
  collectLoop ((ancestors p).filterMap (fun q => (getAt t q).map PT.ir))

def contOK : IRk → Bool
  | .filter _ => true
  | ir => transparent ir

/-- The nested `collect` (`select_scan_node.rs:160-169`): every `Variable` of the filter expression,
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

/-- The downward walk (`select_scan_node.rs:183-199`): from `start` itself, through
    Filter / CondTraverse / CondVarLenTraverse / PathBuilder nodes while they have one child. -/
def spineVars : PT → List Var
  | .node ir cs =>
    if contOK ir then filterVars ir ++ (match cs with | [c] => spineVars c | _ => []) else []

/-- `collect_filtered_vars` (`select_scan_node.rs:156-201`). -/
def filteredVars (t : PT) (p : List Nat) : FV :=
  let above := collectFilteredVars t p
  { above := above, all := above ++ ((getAt t p).map spineVars).getD [] }

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

theorem filteredVars_above_sub (t : PT) (p : List Nat) (v : Var) (h : v ∈ (filteredVars t p).above) :
    v ∈ (filteredVars t p).all := by
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

/-- `collect_output_aliases` (`select_scan_node.rs:405-462`; #3099 adds the `ProcedureCall` YIELDs
    and the `Optional` variables). A vector scan has no arm (`_ => {}`). -/
def collectOutputAliases : IRk → List Nat
  | .allNodeScan a | .labelScan a _ | .indexScan a => [a.1]
  | .fulltextScan n s => n.1 :: s.toList.map (·.1)
  | .project outs => outs.map (·.1)
  | .aggregate names => names.map (·.1)
  | .unwind v => [v.1]
  | .procedureCall ys => ys.map (·.1)
  | .optional vs => vs.map (·.1)
  | .argument (some vs) => vs.map (·.1)
  | _ => []

theorem collectOutputAliases_spec (a : Var) (vs : List Var) (s : Option Var) :
    collectOutputAliases (.labelScan a []) = [a.1] ∧ collectOutputAliases (.indexScan a) = [a.1] ∧
    collectOutputAliases (.project vs) = vs.map (·.1) ∧ collectOutputAliases (.argument none) = [] ∧
    collectOutputAliases (.argument (some vs)) = vs.map (·.1) ∧
    collectOutputAliases (.fulltextScan a s) = a.1 :: s.toList.map (·.1) ∧
    collectOutputAliases (.procedureCall vs) = vs.map (·.1) ∧
    collectOutputAliases (.optional vs) = vs.map (·.1) ∧
    collectOutputAliases (.vectorScan a s) = [] ∧
    collectOutputAliases .other = [] := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

mutual
def PT.walk : PT → List IRk
  | .node ir cs => ir :: PT.walkL cs
def PT.walkL : List PT → List IRk
  | [] => []
  | c :: cs => c.walk ++ PT.walkL cs
end

/-! ## `outer_child_bound_vars` (new in #3099) -/

/-- Operators the walk passes through when they have one child (`Filter`, `IncludePending`, `Limit`,
    `Skip`, `Sort`, `Distinct`, `Commit`). -/
def passThrough : IRk → Bool
  | .filter _ | .includePending | .limit | .skip | .sort | .distinct | .commit => true
  | _ => false

def isProduct : IRk → Bool
  | .cartesianProduct | .valueHashJoin => true
  | _ => false

/-- Operators that add their own outputs and keep their single child's (node scans, `Unwind`,
    `ProcedureCall`, and the separate `NodeByVectorScan` arm, which binds node and score). -/
def descends : IRk → Bool
  | .allNodeScan _ | .labelScan _ _ | .indexScan _ | .fulltextScan _ _ | .unwind _ | .procedureCall _
  | .vectorScan _ _ => true
  | _ => false

/-- What one operator adds in the walk: `collect_output_aliases`, and node + score for a vector scan. -/
def ownVars : IRk → List Nat
  | .vectorScan n s => n.1 :: s.toList.map (·.1)
  | ir => collectOutputAliases ir

mutual
/-- `outer_child_bound_vars` (`select_scan_node.rs:283-342`). The Rust walks with an explicit stack of
    branch roots; as a set the result is this recursion (the order of the stack is irrelevant). -/
def outerChildBoundVars : PT → List Nat
  | .node ir cs =>
    if passThrough ir then
      (match cs with
       | [c] => outerChildBoundVars c
       | _ => collectOutputAliases ir)
    else if isProduct ir then outerChildBoundVarsL cs
    else if descends ir then
      ownVars ir ++ (match cs with
        | [c] => outerChildBoundVars c
        | _ => [])
    else collectOutputAliases ir
def outerChildBoundVarsL : List PT → List Nat
  | [] => []
  | c :: cs => outerChildBoundVars c ++ outerChildBoundVarsL cs
end

/-- Every variable an operator of the subtree outputs. -/
def subtreeOwnVars (t : PT) : List Nat := t.walk.flatMap ownVars

theorem ownVars_collect (ir : IRk) (v : Nat) (h : v ∈ collectOutputAliases ir) : v ∈ ownVars ir := by
  cases ir <;> simp_all [ownVars, collectOutputAliases]

theorem mem_walkL (c : PT) : ∀ (cs : List PT), c ∈ cs → ∀ x ∈ c.walk, x ∈ PT.walkL cs
  | [], h => by simp at h
  | d :: ds, h => by
    intro x hx
    simp only [PT.walkL, List.mem_append]
    rcases List.mem_cons.mp h with rfl | h
    · exact Or.inl hx
    · exact Or.inr (mem_walkL c ds h x hx)

theorem walk_child (ir : IRk) (c : PT) (x : IRk) (hx : x ∈ c.walk) : x ∈ (PT.node ir [c]).walk := by
  simp [PT.walk, PT.walkL, hx]

mutual
/-- **PROVEN** (sound): every variable `outer_child_bound_vars` reports is output by some operator of
    the child subtree — it never invents a binding. -/
theorem outerChildBoundVars_sound : ∀ (t : PT) (v : Nat), v ∈ outerChildBoundVars t →
    ∃ ir ∈ t.walk, v ∈ ownVars ir
  | .node ir [], v, h => by
    have hm : ir ∈ (PT.node ir []).walk := by simp [PT.walk]
    simp only [outerChildBoundVars, outerChildBoundVarsL] at h
    by_cases hp : passThrough ir
    · simp only [hp, if_true] at h; exact ⟨ir, hm, ownVars_collect ir v h⟩
    by_cases hq : isProduct ir
    · simp [hp, hq] at h
    by_cases hd : descends ir
    · simp only [hp, hq, hd, if_true, if_false, Bool.false_eq_true, List.append_nil] at h
      exact ⟨ir, hm, h⟩
    · simp only [hp, hq, hd, if_false, Bool.false_eq_true] at h
      exact ⟨ir, hm, ownVars_collect ir v h⟩
  | .node ir [c], v, h => by
    have hm : ir ∈ (PT.node ir [c]).walk := by simp [PT.walk]
    simp only [outerChildBoundVars] at h
    by_cases hp : passThrough ir
    · simp only [hp, if_true] at h
      obtain ⟨i, hi, hv⟩ := outerChildBoundVars_sound c v h
      exact ⟨i, walk_child ir c i hi, hv⟩
    by_cases hq : isProduct ir
    · simp only [hp, hq, if_true, if_false, Bool.false_eq_true, outerChildBoundVarsL,
        List.append_nil] at h
      obtain ⟨i, hi, hv⟩ := outerChildBoundVars_sound c v h
      exact ⟨i, walk_child ir c i hi, hv⟩
    by_cases hd : descends ir
    · simp only [hp, hq, hd, if_true, if_false, Bool.false_eq_true] at h
      rcases List.mem_append.mp h with h | h
      · exact ⟨ir, hm, h⟩
      · obtain ⟨i, hi, hv⟩ := outerChildBoundVars_sound c v h
        exact ⟨i, walk_child ir c i hi, hv⟩
    · simp only [hp, hq, hd, if_false, Bool.false_eq_true] at h
      exact ⟨ir, hm, ownVars_collect ir v h⟩
  | .node ir (c₁ :: c₂ :: cs), v, h => by
    have hm : ir ∈ (PT.node ir (c₁ :: c₂ :: cs)).walk := by simp [PT.walk]
    simp only [outerChildBoundVars] at h
    by_cases hp : passThrough ir
    · simp only [hp, if_true] at h; exact ⟨ir, hm, ownVars_collect ir v h⟩
    by_cases hq : isProduct ir
    · simp only [hp, hq, if_true, if_false, Bool.false_eq_true] at h
      obtain ⟨i, hi, hv⟩ := outerChildBoundVarsL_sound (c₁ :: c₂ :: cs) v h
      exact ⟨i, by simp only [PT.walk]; exact List.mem_cons_of_mem _ hi, hv⟩
    by_cases hd : descends ir
    · simp only [hp, hq, hd, if_true, if_false, Bool.false_eq_true, List.append_nil] at h
      exact ⟨ir, hm, h⟩
    · simp only [hp, hq, hd, if_false, Bool.false_eq_true] at h
      exact ⟨ir, hm, ownVars_collect ir v h⟩
theorem outerChildBoundVarsL_sound : ∀ (cs : List PT) (v : Nat), v ∈ outerChildBoundVarsL cs →
    ∃ ir ∈ PT.walkL cs, v ∈ ownVars ir
  | [], v, h => by simp [outerChildBoundVarsL] at h
  | c :: cs, v, h => by
    simp only [outerChildBoundVarsL, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨ir', hm, hv⟩ := outerChildBoundVars_sound c v h
      exact ⟨ir', by simp only [PT.walkL]; exact List.mem_append_left _ hm, hv⟩
    · obtain ⟨ir', hm, hv⟩ := outerChildBoundVarsL_sound cs v h
      exact ⟨ir', by simp only [PT.walkL]; exact List.mem_append_right _ hm, hv⟩
end

theorem scanAlias_bound (ir : IRk) (cs : List PT) (a : Var) (h : scanAlias ir = some a) :
    a.1 ∈ outerChildBoundVars (.node ir cs) := by
  cases ir <;> (try simp only [scanAlias, reduceCtorEq, Option.some.injEq] at h)
  all_goals
    subst h
    unfold outerChildBoundVars
    simp only [passThrough, isProduct, descends, ownVars, collectOutputAliases, if_true, if_false,
      Bool.false_eq_true]
    exact List.mem_append_left _ (List.mem_singleton_self _)

theorem outerChildBoundVars_chain : ∀ (t : PT) (k : Nat), scanDepth t = some k →
    ∀ s, getAt t (List.replicate k 0) = some s → ∀ a, scanAlias s.ir = some a →
    a.1 ∈ outerChildBoundVars t
  | .node ir cs, k, h, s, hs, a, ha => by
    unfold scanDepth at h
    cases hsa : scanAlias ir with
    | some a' =>
      simp [hsa] at h; subst h
      simp [getAt] at hs; subst hs
      exact scanAlias_bound ir cs a ha
    | none =>
      simp only [hsa] at h
      split at h
      · next hw =>
        split at h
        · next c =>
          cases hd : scanDepth c with
          | none => simp [hd] at h
          | some k' =>
            simp [hd] at h; subst h
            have hs' : getAt c (List.replicate k' 0) = some s := by
              simpa [List.replicate_succ, getAt] using hs
            have ih := outerChildBoundVars_chain c k' hd s hs' a ha
            have hp : passThrough ir = true := by
              cases ir <;> simp_all [isWrapper, passThrough]
            simp [outerChildBoundVars, hp, ih]
        · cases h
      · cases h

/-- **PROVEN**: an earlier clause's `[Filter →] [IncludePending →] Scan` binds its scanned node, so a
    stitched `MATCH (a) WHERE … MATCH (a)-->(b)` counts `a` as bound. -/
theorem outerChildBoundVars_plannerScan (t : PT) (a : Var) (h : plannerScanAlias t = some a) :
    a.1 ∈ outerChildBoundVars t := by
  unfold plannerScanAlias plannerScanIdx at h
  cases hd : scanDepth t with
  | none => simp [hd] at h
  | some k =>
    simp [hd] at h
    cases hs : getAt t (List.replicate k 0) with
    | none => simp [hs] at h
    | some s =>
      simp [hs] at h
      exact outerChildBoundVars_chain t k hd s hs a h

/-- **PROVEN**: the branches of a product each bind their own variables (`CartesianProduct`,
    `ValueHashJoin`); an `Unwind` keeps its input's and adds its own. -/
theorem outerChildBoundVars_product (cs : List PT) :
    outerChildBoundVars (.node .cartesianProduct cs) = outerChildBoundVarsL cs ∧
    outerChildBoundVars (.node .valueHashJoin cs) = outerChildBoundVarsL cs := by
  constructor <;> (unfold outerChildBoundVars; rfl)

theorem outerChildBoundVars_unwind (k : Var) (c : PT) :
    outerChildBoundVars (.node (.unwind k) [c]) = k.1 :: outerChildBoundVars c := by
  simp [outerChildBoundVars, passThrough, isProduct, descends, ownVars, collectOutputAliases]

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

/-- What `select_var_len_scan_node` (`select_scan_node.rs:532-726`) inspects. -/
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
  argBindsTo : Bool
  argOpaque : Bool
  fromScore : Nat
  toScore : Nat

def vlGuards (c : VLCase) : Bool :=
  !c.expandInto && !c.bidirectional && !c.allShortest && !c.sameEnds && c.oneChild && c.childIsFromScan

/-- The non-leaf rewrite (`select_scan_node.rs:584-640`): drop the planner's scan *and its wrapper
    chain*, keeping what is under the scan. Since #2390 a `Filter` in the wrapper refuses it. -/
def vlNonLeaf (c : VLCase) : Bool :=
  vlGuards c && !c.wrapperHasPending && !c.wrapperHasFilter && c.childBindsTo && c.scanHasChildren

/-- The leaf rewrite (`select_scan_node.rs:642-725`, #3099): scan `to` instead, replacing the child's
    wrapper chain (`node_mut(child_idx).prune()`, line 721) but re-attaching everything under the old
    scan (`kept`, lines 699-714). Since #3099 a `Filter` in the wrapper chain refuses it
    (lines 662-675). -/
def vlLeaf (c : VLCase) : Bool :=
  vlGuards c && !vlNonLeaf c && !c.argOpaque && !c.argBindsTo && !c.wrapperHasFilter &&
    decide (c.fromScore < c.toScore)

/-- The pre-#3099 leaf rewrite (no wrapper check), kept for the historical bug below. -/
def pre3099_vlLeaf (c : VLCase) : Bool :=
  vlGuards c && !vlNonLeaf c && !c.argOpaque && !c.argBindsTo && decide (c.fromScore < c.toScore)

/-- The scan the leaf rewrite builds (#3099): `AllNodeScan`/`NodeByLabelScan(to)` over all of the old
    scan's children, under `IncludePending` in a MERGE match branch. -/
def vlLeafScan (to : Var) (labels : List Nat) (pending : Bool) (kept : List PT) : PT :=
  let scan := PT.node (if labels.isEmpty then .allNodeScan to else .labelScan to labels) kept
  if pending then .node .includePending [scan] else scan

/-- **PROVEN** (#3099): the rebuilt scan is a planner scan of `to` that keeps every child the old scan
    had (an earlier clause, an `UNWIND`, an `Argument`), so their rows and bindings survive. -/
theorem vlLeafScan_spec (to : Var) (labels : List Nat) (pending : Bool) (kept : List PT) :
    plannerScanAlias (vlLeafScan to labels pending kept) = some to ∧
    ∃ s, getAt (vlLeafScan to labels pending kept) (if pending then [0] else []) = some s ∧
      s.cs = kept := by
  have hs : scanAlias (if labels.isEmpty then .allNodeScan to else .labelScan to labels) = some to := by
    split <;> rfl
  unfold vlLeafScan
  generalize (if labels.isEmpty then IRk.allNodeScan to else .labelScan to labels) = ir at hs ⊢
  have hd0 : scanDepth (.node ir kept) = some 0 := by unfold scanDepth; simp [hs]
  cases pending
  · simp [plannerScanAlias, plannerScanIdx, hd0, getAt, PT.ir, PT.cs, hs]
  · have hd1 : scanDepth (.node .includePending [.node ir kept]) = some 1 := by
      unfold scanDepth; simp [scanAlias, isWrapper, hd0]
    simp [plannerScanAlias, plannerScanIdx, hd1, getAt, PT.ir, PT.cs, hs]

/-- **PROVEN** (#2390): the non-leaf rewrite never drops a `Filter` (nor an `IncludePending`). -/
theorem vlNonLeaf_keeps_wrappers (c : VLCase) (h : vlNonLeaf c = true) :
    c.wrapperHasFilter = false ∧ c.wrapperHasPending = false := by
  simp [vlNonLeaf] at h; exact ⟨h.1.1.2, h.1.1.1.2⟩

/-- The leaf rewrite fires only when every guard holds (ties keep the direction). -/
theorem vlLeaf_guards (c : VLCase) (h : vlLeaf c = true) :
    c.expandInto = false ∧ c.bidirectional = false ∧ c.allShortest = false ∧ c.sameEnds = false ∧
    c.childIsFromScan = true ∧ c.argBindsTo = false ∧ c.argOpaque = false ∧ c.fromScore < c.toScore := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp_all [vlLeaf, vlGuards]

/-- **PROVEN** (#3099): the leaf rewrite never drops a wrapper `Filter` either. -/
theorem vlLeaf_keeps_filter (c : VLCase) (h : vlLeaf c = true) : c.wrapperHasFilter = false := by
  simp_all [vlLeaf]

/-! ### CONFIRMED BUG (pre-#3099; FIXED by #3099): the leaf rewrite dropped a wrapper `Filter`

`MATCH (a) WHERE a.v = 1 MATCH (a)-[*1..2]->(b:B {w:2}) RETURN a.v, b.w`: clause 1's
`Filter(a.v = 1) → AllNodeScan(a)` is stitched in as the var-length traverse's child, which
`planner_scan_alias` accepts as the planner's scan of `from = a` (it looks through `Filter`). `a`
scores 2 (filtered, via the downward spine of `collect_filtered_vars`), `b` scores 3 (labelled, its
lowered `{w:2}` Filter above the traverse), so the leaf rewrite scans `b` and walks backwards — and the
pruned child took `a.v = 1` with it. Live (Rust built from main's planner, b3582e34a; also on the
pre-#2390 build 88e92ae25): graph `(:A {v:1})-[:R]->(:B {w:2}), (:A {v:5})-[:R]->(:B {w:2})` returns
`[1, 2], [5, 2]`; the right answer (and the fixed-length `-[:R]->` form, which salvages the Filter via
`filters_of`) is `[1, 2]`. #3099 refuses the leaf rewrite when the wrapper chain holds a `Filter`
(`vl_leaf_refuses_filter`). -/

def vlBugCase : VLCase :=
  { expandInto := false, bidirectional := false, allShortest := false, sameEnds := false, oneChild := true,
    childIsFromScan := true, wrapperHasPending := false, wrapperHasFilter := true, childBindsTo := false,
    scanHasChildren := false, argBindsTo := false, argOpaque := false,
    fromScore := (scoreEndpoint ⟨(0, 0), []⟩ ⟨[(1, 0)], [(1, 0), (0, 0)]⟩ [] (fun _ => 1)).1,
    toScore := (scoreEndpoint ⟨(1, 0), [7]⟩ ⟨[(1, 0)], [(1, 0), (0, 0)]⟩ [] (fun _ => 1)).1 }

/-- Before #3099 the rewrite fired on the repro although the wrapper holds `a.v = 1`. -/
theorem pre3099_vl_leaf_fires_over_filter :
    pre3099_vlLeaf vlBugCase = true ∧ vlBugCase.wrapperHasFilter = true := by
  decide

/-- **PROVEN** (#3099 fixes it): the rewrite now refuses the repro; every other guard still passes. -/
theorem vl_leaf_refuses_filter : vlLeaf vlBugCase = false ∧ pre3099_vlLeaf vlBugCase = true := by
  decide

/-- What the pre-#3099 rewritten plan selected (`Lt b`, `Lf a`, the backward walk) differs from the original
    (which also tests the dropped `F a`) exactly on the rows where `F a` is false — `(a.v = 5, b)`. -/
theorem vl_leaf_drops_filter (R : Nat → Nat → Prop) (Lf Lt F : Nat → Prop) (lo hi a b : Nat)
    (hF : ¬ F a) (hsel : Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach (tr R) k b a) :
    (Lt b ∧ Lf a ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach (tr R) k b a) ∧
    ¬ (Lf a ∧ F a ∧ Lt b ∧ ∃ k, lo ≤ k ∧ k ≤ hi ∧ Reach R k a b) := ⟨hsel, fun h => hF h.2.1⟩

end OptimizerScan.ScanTree
