/-
# `select_scan_node.rs` helpers (origin/main 3fec7d7c9)

The plan is a rose tree; an orx-tree `NodeIdx` is modelled by its structural
path (child indices from the root), which is exactly what `node_path` computes.

| here | there |
| --- | --- |
| `IRk`, `PT`, `getAt` | `IR`, `DynTree<IR>`, `plan.node(idx)` |
| `scoreEndpoint` | `score_endpoint` 89 |
| `collectFilteredVars` | `collect_filtered_vars` 134 |
| `makeScanSubtree` | `make_scan_subtree` 167 |
| `argumentLeafOf` | `argument_leaf_of` 200 |
| `inMergeMatchBranch` | `in_merge_match_branch` 219 |
| `scanDepth`, `isPlannerScan`, `plannerScanAlias`, `plannerScanIdx` | `is_planner_scan_subtree` 239, `planner_scan_alias` 258, `planner_scan_idx` 277 |
| `collectOutputAliases` | `collect_output_aliases` 315 |
| `nodePath`, `resolvePath` | `node_path` 366, `resolve_path` 383 |
| `childSubtreeBinds` | `child_subtree_binds` 399 |
| `vlRewrite`, `reach`, `vl_reverse_sound` | `select_var_len_scan_node` 434 |
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

structure EP where
  alias : Var
  labels : List Nat
  nattrs : Nat

def scoreEndpoint (n : EP) (filtered : List Var) (bound : List Nat) (count : Nat → Nat) : Nat × Nat :=
  ((if n.alias.1 ∈ bound then 3 else 0) + (if n.alias ∈ filtered then 2 else 0) +
   (if n.nattrs > 0 then 2 else 0) + (if n.labels.isEmpty then 0 else 1),
   match n.labels with
   | [] => 2 ^ 64 - 1
   | l :: ls => (ls.map count).foldl min (count l))

/-- **PROVEN** (score formula and its two halves): bound +3, filtered +2,
inline attributes +2, labelled +1; cardinality is `u64::MAX` for an unlabelled
node, else the minimum label count. -/
theorem scoreEndpoint_spec (n : EP) (f : List Var) (b : List Nat) (c : Nat → Nat) :
    (scoreEndpoint n f b c).1 ≤ 8 ∧
    (n.alias.1 ∈ b → (scoreEndpoint n f b c).1 ≥ 3) ∧
    (n.alias ∈ f → (scoreEndpoint n f b c).1 ≥ 2) ∧ (n.nattrs > 0 → (scoreEndpoint n f b c).1 ≥ 2) ∧
    (n.labels ≠ [] → (scoreEndpoint n f b c).1 ≥ 1) ∧
    (n.labels = [] → (scoreEndpoint n f b c).2 = 2 ^ 64 - 1) := by
  unfold scoreEndpoint
  refine ⟨by split <;> split <;> split <;> split <;> omega, fun h => ?_, fun h => ?_, fun h => ?_,
    fun h => ?_, fun h => by simp [h]⟩
  · simp only [h, ite_true]; omega
  · simp only [h, ite_true]; omega
  · simp only [h, ite_true]; omega
  · have : n.labels.isEmpty = false := by cases hl : n.labels <;> simp_all
    simp only [this]; simp

/-- The documented "bound has highest priority" is a heuristic only: a filtered,
attributed, labelled endpoint (5) outranks a bare bound one (3). Cost, not
correctness (soundness is `ScanOrder.order_sound`). -/
theorem score_bound_not_dominant :
    (scoreEndpoint ⟨(1, 0), [], 0⟩ [] [1] (fun _ => 0)).1 <
    (scoreEndpoint ⟨(2, 0), [7], 1⟩ [(2, 0)] [1] (fun _ => 0)).1 := by decide

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

/-- "pruning a child never changes the parent's path" (`select_scan_node.rs:532,587`). -/
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

/-! ## `make_scan_subtree` -/

def makeScanSubtree (alias : Var) (labels : List Nat) (attrFilter : Option (List Var)) (pending : Bool)
    (argument : Option IRk) : PT :=
  let scan := if labels.isEmpty then IRk.allNodeScan alias else .labelScan alias labels
  let s0 := PT.node scan (argument.toList.map (fun a => PT.node a []))
  let s1 := if pending then PT.node .includePending [s0] else s0
  match attrFilter with
  | some vs => PT.node (.filter vs) [s1]
  | none => s1

/-- **PROVEN**: `make_scan_subtree` builds exactly the shape the planner-scan
walkers accept, with the node's alias at the bottom. -/
theorem makeScanSubtree_planner (alias : Var) (labels : List Nat) (af : Option (List Var)) (pend : Bool)
    (arg : Option IRk) :
    plannerScanAlias (makeScanSubtree alias labels af pend arg) = some alias := by
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
  have hf : ∀ vs t, plannerScanAlias t = some alias → plannerScanAlias (PT.node (.filter vs) [t]) = some alias := by
    intro vs t ht
    simp only [plannerScanAlias, plannerScanIdx] at ht
    cases hd : scanDepth t with
    | none => simp [hd] at ht
    | some k =>
      simp only [hd, Option.map_some, Option.bind_some] at ht
      have e : scanDepth (PT.node (.filter vs) [t]) = some (k + 1) := by
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
  cases af with
  | none => exact h1
  | some vs => exact hf vs _ h1

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

def collectFilteredVars (t : PT) (p : List Nat) : List Var :=
  collectLoop ((ancestors p).filterMap (fun q => (getAt t q).map PT.ir))

def contOK : IRk → Bool
  | .filter _ => true
  | ir => transparent ir

def filterVars : IRk → List Var
  | .filter vs => vs
  | _ => []

/-- **PROVEN**: the variables collected are those of the `Filter`s in the
maximal run of Filter / CondTraverse / CondVarLenTraverse / PathBuilder
ancestors directly above the node. -/
theorem collectLoop_eq (l : List IRk) : collectLoop l = (l.takeWhile contOK).flatMap filterVars := by
  induction l with
  | nil => rfl
  | cons ir rest ih =>
    cases ir <;> simp [collectLoop, contOK, transparent, filterVars, List.takeWhile_cons, ih]

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

def collectOutputAliases : IRk → List Nat
  | .allNodeScan a | .labelScan a _ | .indexScan a => [a.1]
  | .fulltextScan n s => n.1 :: s.toList.map (·.1)
  | .project outs => outs.map (·.1)
  | .aggregate names => names.map (·.1)
  | .unwind v => [v.1]
  | .argument (some vs) => vs.map (·.1)
  | _ => []

theorem collectOutputAliases_spec (a : Var) (vs : List Var) (s : Option Var) :
    collectOutputAliases (.labelScan a []) = [a.1] ∧ collectOutputAliases (.indexScan a) = [a.1] ∧
    collectOutputAliases (.project vs) = vs.map (·.1) ∧ collectOutputAliases (.argument none) = [] ∧
    collectOutputAliases (.argument (some vs)) = vs.map (·.1) ∧
    collectOutputAliases (.fulltextScan a s) = a.1 :: s.toList.map (·.1) ∧
    collectOutputAliases .other = [] := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

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

/-- The rewrite's decision (`select_var_len_scan_node.rs:467-570`): every guard
must pass, and the `to` endpoint must score strictly higher. -/
structure VLCase where
  expandInto : Bool
  bidirectional : Bool
  allShortest : Bool
  sameEnds : Bool
  oneChild : Bool
  childIsFromScan : Bool
  wrapperHasPending : Bool
  argBindsTo : Bool
  argOpaque : Bool
  attrsUnfiltered : Bool
  fromScore : Nat
  toScore : Nat

def vlRewrite (c : VLCase) : Bool :=
  !c.expandInto && !c.bidirectional && !c.allShortest && !c.sameEnds && c.oneChild && c.childIsFromScan &&
  !c.argOpaque && !c.argBindsTo && !c.attrsUnfiltered && decide (c.fromScore < c.toScore)

/-- The rewrite fires only when every guard holds (ties keep the direction). -/
theorem vlRewrite_guards (c : VLCase) (h : vlRewrite c = true) :
    c.expandInto = false ∧ c.bidirectional = false ∧ c.allShortest = false ∧ c.sameEnds = false ∧
    c.childIsFromScan = true ∧ c.argBindsTo = false ∧ c.argOpaque = false ∧ c.attrsUnfiltered = false ∧
    c.fromScore < c.toScore := by
  simp [vlRewrite] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, -⟩, h6⟩, h7⟩, h8⟩, h9⟩, h10⟩ := h
  exact ⟨h1, h2, h3, h4, h6, h8, h7, h9, h10⟩

end OptimizerScan.ScanTree
