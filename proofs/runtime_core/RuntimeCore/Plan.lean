/-
Plan-shape helpers and constructors of `Runtime` (graph/src/runtime/runtime.rs).

| here | there |
| --- | --- |
| `IRK`, `nodeVars`, `getVariables` | `GetVariables` trait (:209) and its `DynNode` impl (:213-330) |
| `PT`, `returnNames`  | `ReturnNames` trait (:335) and its impl (:339-380) |
| `inspectBatch`       | `Runtime::inspect_batch` (:384) |
| `RtInit`, `rtNew`    | `Runtime::new` (:406-463) |
| `writeEscalation`, `resyncPublished`, `commitDeferred` | `write_escalation` (:494), `resync_published_indexes` (:500), `commit_deferred_indexes` (:510) |
| `defaultBatch`       | `default_batch` (:564) |
| `childrenSpec`       | `children_to_recurse` (:652), on `RunBatch.childrenToRecurse` |
| `runNested`          | `run_nested_plan` (:678) |
| `BK`, `buildOp`      | `build_batch_op` (:743-1306): child popping and the per-arm checks |
-/
namespace RuntimeCore.Plan

/-! ## `get_variables` (:213) -/

/-- The variable-bearing payload of each IR variant, as `get_variables` reads it. -/
inductive IRK where
  | optional (vs : List Nat)
  | procCall (yields : List Nat)
  | unwind (v : Nat)
  | createOrMerge (nodes rels paths : List Nat)
  | forEachOrLoadCsv (v : Nat)
  | silent                     -- Delete, Argument, Set, Remove, Filter, products, applies, Sort, Skip, …
  | scan (node : Nat)          -- label/all/index/label-and-id scans, id seek
  | scored (ent : Nat) (score : Option Nat)  -- fulltext / vector scans
  | rel (r f t : Nat)          -- CondTraverse, EdgeByIndexScan, AllShortestPaths, ExpandInto
  | varLen (r f t : Nat) (path : Option Nat)
  | pathBuilder (vars : List Nat)
  | aggregate (names : List Nat)
  | project (names : List Nat)

def nodeVars : IRK → List Nat
  | .optional vs | .procCall vs | .pathBuilder vs | .aggregate vs | .project vs => vs
  | .unwind v | .forEachOrLoadCsv v | .scan v => [v]
  | .createOrMerge ns rs ps => ns ++ rs ++ ps
  | .silent => []
  | .scored e s => e :: s.toList
  | .rel r f t => [r, f, t]
  | .varLen r f t p => [r, f, t] ++ p.toList

/-- The BFS walk (:214-327): collect each node's variables; the first `Project` ends the
walk after contributing its names (`break`). `bfs` = the subtree in `walk::<Bfs>` order. -/
def getVariables : List IRK → List Nat
  | [] => []
  | k :: ks => match k with
    | .project names => names
    | _ => nodeVars k ++ getVariables ks

def isProject : IRK → Bool | .project _ => true | _ => false

theorem getVariables_spec (bfs : List IRK) :
    getVariables bfs = ((bfs.takeWhile (!isProject ·)).flatMap nodeVars) ++
      (match bfs.dropWhile (!isProject ·) with
       | .project ns :: _ => ns
       | _ => []) := by
  induction bfs with
  | nil => rfl
  | cons k ks ih =>
    cases k <;> simp [getVariables, isProject, ih, nodeVars, List.takeWhile_cons, List.dropWhile_cons]

/-- Nothing after the first `Project` (in BFS order) is collected. -/
theorem getVariables_stops (pre post : List IRK) (ns : List Nat) (hp : ∀ k ∈ pre, isProject k = false) :
    getVariables (pre ++ .project ns :: post) = pre.flatMap nodeVars ++ ns := by
  induction pre with
  | nil => rfl
  | cons k ks ih =>
    have hk := hp k (List.mem_cons_self ..)
    cases k <;> simp_all [getVariables, isProject]

/-! ## `get_return_names` (:339) -/

inductive RK where
  | project (names : List Nat)
  | commit
  | procCall (yields : List Nat)
  | scored (ent : Nat) (score : Option Nat)
  | sort
  | apply
  | passThrough    -- Skip, Limit, Distinct, Union, NestedPlans: child 0's names
  | aggregate (names : List Nat)
  | other

inductive PT where
  | node (k : RK) (kids : List PT)

/-- `none` = the `child(0)` panic of a malformed plan. -/
def returnNames : PT → Option (List Nat)
  | .node (.project ns) _ => some ns
  | .node .commit kids => match kids with
    | c :: _ => returnNames c
    | [] => some []
  | .node (.procCall ys) _ => some ys
  | .node (.scored e s) _ => some (e :: s.toList)
  | .node .sort kids => match kids with
    | c :: _ => skipApply c
    | [] => none
  | .node .passThrough kids => match kids with
    | c :: _ => returnNames c
    | [] => none
  | .node (.aggregate ns) _ => some ns
  | .node _ _ => some []
where
  /-- `while matches!(child, Apply) { child = child.child(0) }` then the child's names. -/
  skipApply : PT → Option (List Nat)
  | .node .apply kids => match kids with
    | c :: _ => skipApply c
    | [] => none
  | t => returnNames t

/-- A `Project` root names the result columns; `Skip/Limit/Distinct/Union/NestedPlans` and
`Commit` are transparent; a `Sort` sees through any `Apply` chain below it; an aggregate names
its outputs; a scored scan names its entity then its score. -/
theorem returnNames_spec (ns : List Nat) (t : PT) (rest : List PT) (k : RK) :
    returnNames (.node (.project ns) rest) = some ns ∧
    returnNames (.node .passThrough (t :: rest)) = returnNames t ∧
    returnNames (.node .commit (t :: rest)) = returnNames t ∧
    returnNames (.node .commit []) = some [] ∧
    returnNames (.node .sort [.node .apply [t]]) = returnNames.skipApply t ∧
    returnNames (.node (.aggregate ns) rest) = some ns ∧
    (∀ e s, returnNames (.node (.scored e s) rest) = some (e :: s.toList)) := by
  refine ⟨by simp [returnNames], by simp [returnNames], by simp [returnNames], by simp [returnNames], ?_, by simp [returnNames], fun _ _ => by simp [returnNames]⟩
  simp [returnNames, returnNames.skipApply]

/-! ## `inspect_batch` (:384) -/

/-- Inspect mode records one owned row per active row, or the error; otherwise nothing. -/
def inspectBatch {R : Type} (inspect : Bool) (record : List (Nat × Except String R)) (idx : Nat)
    (res : Except String (List R)) : List (Nat × Except String R) :=
  if inspect then
    match res with
    | .ok rows => record ++ rows.map fun r => (idx, .ok r)
    | .error e => record ++ [(idx, .error e)]
  else record

theorem inspectBatch_spec {R : Type} (rec : List (Nat × Except String R)) (idx : Nat) (rows : List R) (e : String)
    (res : Except String (List R)) :
    inspectBatch true rec idx (.ok rows) = rec ++ rows.map (fun r => (idx, .ok r)) ∧
    inspectBatch true rec idx (.error e) = rec ++ [(idx, .error e)] ∧
    inspectBatch false rec idx res = rec := ⟨rfl, rfl, rfl⟩

/-! ## `Runtime::new` (:406) -/

structure RtInit (P B : Type) where
  returnNames : Option (List Nat)
  nestedPlans : List Nat
  pending : P
  write : Bool
  deadline : Option Nat
  record : List Nat
  dedupers : List Nat
  mergeCache : List Nat
  memoN : List Nat
  memoR : List Nat
  effects : Option B
  effectsCount : Nat
  buildEffects : Bool

/-- `nested_plans` = the root's children after the first when the root is `NestedPlans`;
the pending store opens its schema baseline only for a write query (`openWrite` =
`set_schema_baseline`, runtime.rs:428). Since #2846 it opens no id boundaries: the
graph's write version carries its own id batch (`IdSpace::new_version`). -/
def rtNew {P B : Type} (root : PT) (rootIsNested : Bool) (rootKids : List Nat) (write : Bool)
    (pendingNew : P) (openWrite : P → P) (now : Nat) (timeoutMs : Option Nat) : RtInit P B :=
  { returnNames := returnNames root
    nestedPlans := if rootIsNested then rootKids.drop 1 else []
    pending := if write then openWrite pendingNew else pendingNew
    write
    deadline := timeoutMs.map (now + ·)
    record := [], dedupers := [], mergeCache := [], memoN := [], memoR := []
    effects := none, effectsCount := 0, buildEffects := true }

theorem rtNew_spec {P B : Type} (root : PT) (nested : Bool) (kids : List Nat) (w : Bool) (p : P)
    (ow : P → P) (now : Nat) (t : Option Nat) :
    let r : RtInit P B := rtNew root nested kids w p ow now t
    r.returnNames = returnNames root ∧ (nested = true → r.nestedPlans = kids.drop 1) ∧
    (nested = false → r.nestedPlans = []) ∧ (w = true → r.pending = ow p) ∧ (w = false → r.pending = p) ∧
    r.deadline = t.map (now + ·) ∧ r.record = [] ∧ r.mergeCache = [] ∧ r.effects = none ∧
    r.buildEffects = true := by
  refine ⟨rfl, fun h => by simp [rtNew, h], fun h => by simp [rtNew, h], fun h => by simp [rtNew, h],
    fun h => by simp [rtNew, h], rfl, rfl, rfl, rfl, rfl⟩

/-! ## Delegating accessors (:494, :500, :510) -/

def writeEscalation {W : Type} (w : W) : W := w

/-- `resync_published_indexes` / `commit_deferred_indexes` hand the pending store and the
graph to `Pending`'s routines (proved in proofs/pending_commit; the RediSearch writes are FFI). -/
def resyncPublished {P G C : Type} (resync : P → C → G → P) (p : P) (committed : C) (g : G) : P :=
  resync p committed g
def commitDeferred {P G : Type} (commit : P → G → P) (p : P) (g : G) : P := commit p g

theorem delegates_spec {P G C W : Type} (w : W) (rs : P → C → G → P) (cm : P → G → P) (p : P) (c : C) (g : G) :
    writeEscalation w = w ∧ resyncPublished rs p c g = rs p c g ∧ commitDeferred cm p g = cm p g :=
  ⟨rfl, rfl, rfl⟩

/-! ## `default_batch` (:564) -/

/-- One row, no bindings: `BatchBuilder::new()`, `push_row(&Row::new())`, `finish()`. -/
def defaultBatch : List (List (Option Nat)) := [[]]

theorem defaultBatch_spec : defaultBatch.length = 1 ∧ ∀ r ∈ defaultBatch, r = [] := by
  simp [defaultBatch]

/-! ## `run_nested_plan` (:678) -/

/-- Find the plan, build it, install the argument row, take the first batch's first active
row's `result` slot; any missing piece is "nested plan #id produced no result". -/
def runNested {B V : Type} (plans : List Nat) (id : Nat) (build : Nat → Except String B)
    (firstBatch : B → Option (Except String (List (Nat → Option V)))) (result : Nat) :
    Except String V := do
  let idx ← match plans[id]? with
    | some i => pure i
    | none => throw s!"nested plan #{id} not found"
  let op ← build idx
  let missing := s!"nested plan #{id} produced no result"
  let rows ← match firstBatch op with
    | none => throw missing
    | some r => r
  match rows with
  | [] => throw missing
  | row :: _ => match row result with
    | some v => pure v
    | none => throw missing

theorem runNested_spec {B V : Type} (plans : List Nat) (id : Nat) (build : Nat → Except String B)
    (fb : B → Option (Except String (List (Nat → Option V)))) (res : Nat) (i : Nat) (b : B)
    (row : Nat → Option V) (rows : List (Nat → Option V)) (v : V)
    (hi : plans[id]? = some i) (hb : build i = .ok b) (hf : fb b = some (.ok (row :: rows)))
    (hv : row res = some v) :
    runNested plans id build fb res = .ok v ∧
    (plans[id]? = none → runNested plans id build fb res = .error s!"nested plan #{id} not found") := by
  refine ⟨by simp [runNested, hi, hb, hf, hv, bind, Except.bind, pure, Except.pure], fun h => ?_⟩
  simp [runNested, h, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-! ## `build_batch_op` (:743) -/

/-- How each IR arm sources its input: `pop_or_once` (a missing child becomes a one-row
`Once` leaf), `pop_or_argument` (MERGE / FOREACH: a one-row `Argument` leaf, so the body sees
its outer row), or no child. -/
inductive BK where
  | once           -- every scan/stream/write arm and NestedPlans
  | argumentLeaf   -- Merge, ForEach
  | skip (v : Option Int)   -- `Some n` = the SKIP expression evaluated to `Int(n)`
  | limit (v : Option Int)
  | vhj
  | union
  | argument
  | ddl

inductive Op (B : Type) where
  | once (b : List (List (Option Nat)))
  | argument (b : List (List (Option Nat)))
  | built (child : B)
  | skip (child : Op B) (n : Nat)
  | limit (child : Op B) (n : Nat)
  | wrap (child : Op B)
  | join (left right : Op B)
  | union
  | ddl

def popOrOnce {B : Type} : List (Op B) → Op B × List (Op B)
  | c :: cs => (c, cs)
  | [] => (.once defaultBatch, [])

def popOrArgument {B : Type} : List (Op B) → Op B × List (Op B)
  | c :: cs => (c, cs)
  | [] => (.argument defaultBatch, [])

def buildOp {B : Type} (k : BK) (kids : List (Op B)) : Except String (Op B) :=
  match k with
  | .once => .ok (.wrap (popOrOnce kids).1)
  | .argumentLeaf => .ok (.wrap (popOrArgument kids).1)
  | .skip v => match v with
    | none => .error "Skip operator requires an integer argument"
    | some n => if n < 0 then .error s!"SKIP must be a non-negative integer, got {n}"
                else .ok (.skip (popOrOnce kids).1 n.toNat)
  | .limit v => match v with
    | none => .error "Limit operator requires an integer argument"
    | some n => if n < 0 then .error s!"LIMIT must be a non-negative integer, got {n}"
                else .ok (.limit (popOrOnce kids).1 n.toNat)
  | .vhj =>
    let (l, rest) := popOrOnce kids
    match rest with
    | r :: _ => .ok (.join l r)
    | [] => .error "ValueHashJoin missing right child"
  | .union => .ok .union
  | .argument => .ok (.argument defaultBatch)
  | .ddl => .ok .ddl

/-- A leaf arm with no pre-built child reads one empty row (`Once`), MERGE/FOREACH an
`Argument` row; SKIP/LIMIT accept exactly non-negative integers; a join needs both inputs. -/
theorem buildOp_spec {B : Type} (c : Op B) (cs : List (Op B)) (n : Int) :
    buildOp (B := B) .once [] = .ok (.wrap (.once defaultBatch)) ∧
    buildOp (B := B) .argumentLeaf [] = .ok (.wrap (.argument defaultBatch)) ∧
    buildOp .once (c :: cs) = .ok (.wrap c) ∧
    buildOp (B := B) (.skip none) cs = .error "Skip operator requires an integer argument" ∧
    (n < 0 → buildOp (B := B) (.limit (some n)) cs = .error s!"LIMIT must be a non-negative integer, got {n}") ∧
    (0 ≤ n → buildOp (.skip (some n)) (c :: cs) = .ok (.skip c n.toNat)) ∧
    buildOp (B := B) .vhj [c] = .error "ValueHashJoin missing right child" := by
  refine ⟨rfl, rfl, rfl, rfl, fun h => by simp [buildOp, h], fun h => ?_, rfl⟩
  have : ¬ n < 0 := by omega
  simp [buildOp, this, popOrOnce]

end RuntimeCore.Plan
