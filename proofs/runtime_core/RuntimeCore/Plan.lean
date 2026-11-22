/-
Plan-shape helpers and constructors of `Runtime` (graph/src/runtime/runtime.rs).

| here | there |
| --- | --- |
| `IRK`, `VT`, `nodeVars`, `expand`, `getVariables`, `visible` | `GetVariables` trait (:257) and its `DynNode` impl (:260-395) |
| `PT`, `returnNames`  | `ReturnNames` trait (:399) and its impl (:402-) |
| `inspectBatch`       | `Runtime::inspect_batch` (:449) |
| `RtInit`, `rtNew`    | `Runtime::new` (:471-525; the graph fields are `QueryGraph.new`) |
| `resyncPublished`, `commitDeferred` | `resync_published_indexes` (:633), `commit_deferred_indexes` (:643) |
| `defaultBatch`       | `default_batch` (:699) |
| `childrenSpec`       | `children_to_recurse` (:787), on `RunBatch.childrenToRecurse` |
| `runNested`          | `run_nested_plan` (:813) |
| `BK`, `buildOp`      | `build_batch_op` (:878-1442): child popping and the per-arm checks |
-/
namespace RuntimeCore.Plan

/-! ## `get_variables` (:257-395) -/

/-- The variable-bearing payload of each IR variant, as `get_variables` reads it. -/
inductive IRK where
  | optional (vs : List Nat)
  | procCall (yields : List Nat)
  | unwind (v : Nat)
  | createOrMerge (nodes rels paths : List Nat)
  | forEachOrLoadCsv (v : Nat)
  | silent                     -- Delete, Argument, Set, Remove, Filter, products, Apply, Sort, Skip, …
  | semi                       -- SemiApply, AntiSemiApply, OrApplyMultiplexer (:320-323)
  | scan (node : Nat)          -- label/all/index/label-and-id scans, id seek
  | scored (ent : Nat) (score : Option Nat)  -- fulltext / vector scans
  | rel (r f t : Nat)          -- CondTraverse, EdgeByIndexScan, AllShortestPaths, ExpandInto
  | varLen (r f t : Nat) (path : Option Nat)
  | pathBuilder (vars : List Nat)
  | aggregate (names projOuts : List Nat)  -- `names`, then each projection's output (:379-385)
  | project (exprs copyOuts : List Nat)    -- each expr's variable, then each copy's output (:386-390)

def nodeVars : IRK → List Nat
  | .optional vs | .procCall vs | .pathBuilder vs => vs
  | .aggregate ns ps | .project ns ps => ns ++ ps
  | .unwind v | .forEachOrLoadCsv v | .scan v => [v]
  | .createOrMerge ns rs ps => ns ++ rs ++ ps
  | .silent | .semi => []
  | .scored e s => e :: s.toList
  | .rel r f t => [r, f, t]
  | .varLen r f t p => [r, f, t] ++ p.toList

/-- A plan subtree (`DynNode<IR>`): its IR payload and its children, in order. -/
inductive VT where
  | node (k : IRK) (kids : List VT)

/-- The children the walk enqueues after a node: none below `Project`/`Aggregate`
(`continue`, :384, :389), only child 0 (the input) of an existence check (:321), all
children otherwise (:392). -/
def expand : IRK → List VT → List VT
  | .project .., _ | .aggregate .., _ => []
  | .semi, kids => kids.take 1
  | _, kids => kids

theorem sizeOf_append (a b : List VT) : sizeOf (a ++ b) + 1 = sizeOf a + sizeOf b := by
  induction a with
  | nil => simp; omega
  | cons x a ih => simp [List.cons_append]; omega

theorem one_le_sizeOf (l : List VT) : 1 ≤ sizeOf l := by
  cases l <;> simp <;> omega

theorem sizeOf_expand (k : IRK) (kids : List VT) : sizeOf (expand k kids) ≤ sizeOf kids := by
  cases k <;> cases kids <;> simp [expand] <;> first | omega | exact one_le_sizeOf _

/-- The BFS queue (:270-394): pop the front, collect its variables, enqueue `expand`.
`get_variables` on a node `t` is `getVariables [t]`. -/
def getVariables : List VT → List Nat
  | [] => []
  | .node k kids :: q => nodeVars k ++ getVariables (q ++ expand k kids)
termination_by q => sizeOf q
decreasing_by
  have h1 := sizeOf_append q (expand k kids)
  have h2 := sizeOf_expand k kids
  simp
  omega

/-- The node kinds whose variables reach the rows leaving `t`: `t` itself, then (depth
first) what is visible in the children `expand` keeps. -/
def visible : VT → List IRK
  | .node k kids => k :: match k with
    | .project .. | .aggregate .. => []
    | .semi => match kids with
      | [] => []
      | c :: _ => visible c
    | _ => kids.flatMap visible

theorem visible_node (k : IRK) (kids : List VT) :
    visible (.node k kids) = k :: (expand k kids).flatMap visible := by
  cases k <;> cases kids <;> simp [visible, expand]

/-- One step of the walk, and what `expand` keeps. -/
theorem getVariables_spec (k : IRK) (kids q : List VT) :
    getVariables [] = [] ∧
    getVariables (.node k kids :: q) = nodeVars k ++ getVariables (q ++ expand k kids) ∧
    (∀ ns ps, expand (.project ns ps) kids = [] ∧ expand (.aggregate ns ps) kids = []) ∧
    expand .semi kids = kids.take 1 ∧
    (k ≠ .semi → (∀ ns ps, k ≠ .project ns ps) → (∀ ns ps, k ≠ .aggregate ns ps) →
      expand k kids = kids) := by
  refine ⟨by simp [getVariables], by rw [getVariables], fun _ _ => ⟨rfl, rfl⟩, rfl, ?_⟩
  intro h1 h2 h3
  cases k <;> simp_all [expand]

/-- A variable is collected iff some visible node of some queued subtree carries it. -/
theorem getVariables_mem (q : List VT) (v : Nat) :
    v ∈ getVariables q ↔ ∃ t ∈ q, ∃ k ∈ visible t, v ∈ nodeVars k := by
  induction q using getVariables.induct with
  | case1 => simp [getVariables]
  | case2 k kids q ih =>
    rw [getVariables, List.mem_append, ih]
    constructor
    · rintro (h | ⟨t, ht, k', hk', hv⟩)
      · exact ⟨.node k kids, List.mem_cons_self .., k, by simp [visible_node], h⟩
      · rcases List.mem_append.1 ht with ht | ht
        · exact ⟨t, List.mem_cons_of_mem _ ht, k', hk', hv⟩
        · exact ⟨.node k kids, List.mem_cons_self .., k', by
            rw [visible_node]; exact List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨t, ht, hk'⟩), hv⟩
    · rintro ⟨t, ht, k', hk', hv⟩
      rcases List.mem_cons.1 ht with rfl | ht
      · rw [visible_node] at hk'
        rcases List.mem_cons.1 hk' with rfl | hk'
        · exact Or.inl hv
        · obtain ⟨t', ht', hk''⟩ := List.mem_flatMap.1 hk'
          exact Or.inr ⟨t', List.mem_append.2 (Or.inr ht'), k', hk'', hv⟩
      · exact Or.inr ⟨t, List.mem_append.2 (Or.inl ht), k', hk', hv⟩

/-- Nothing below a `Project`/`Aggregate` is collected: the subtree's variables are
exactly the names it outputs, whatever its children bind. An existence check reports
only its input (child 0); its branches are never collected. -/
theorem getVariables_stops (ns ps : List Nat) (kids : List VT) (c : VT) (rest : List VT) :
    getVariables [.node (.project ns ps) kids] = ns ++ ps ∧
    getVariables [.node (.aggregate ns ps) kids] = ns ++ ps ∧
    getVariables [.node .semi (c :: rest)] = getVariables [c] := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [getVariables, expand, nodeVars]

/-- A scope boundary hides only its own subtree: a sibling branch queued after it is
still collected (the walk no longer `break`s). -/
theorem getVariables_sibling (ns ps : List Nat) (kids : List VT) (x : Nat) :
    getVariables [.node .silent [.node (.project ns ps) kids, .node (.scan x) []]] =
      ns ++ ps ++ [x] := by
  simp [getVariables, expand, nodeVars]

/-! ## `get_return_names` (:399-) -/

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

/-! ## `inspect_batch` (:449) -/

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

/-! ## `Runtime::new` (:471) -/

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
`set_schema_baseline`, runtime.rs). It opens no id boundaries: the
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

/-! ## Delegating accessors (:633, :643) -/

/-- `resync_published_indexes` / `commit_deferred_indexes` hand the pending store and the
graph to `Pending`'s routines (proved in proofs/pending_commit; the RediSearch writes are FFI). -/
def resyncPublished {P G C : Type} (resync : P → C → G → P) (p : P) (committed : C) (g : G) : P :=
  resync p committed g
def commitDeferred {P G : Type} (commit : P → G → P) (p : P) (g : G) : P := commit p g

theorem delegates_spec {P G C : Type} (rs : P → C → G → P) (cm : P → G → P) (p : P) (c : C) (g : G) :
    resyncPublished rs p c g = rs p c g ∧ commitDeferred cm p g = cm p g :=
  ⟨rfl, rfl⟩

/-! ## `default_batch` (:699) -/

/-- One row, no bindings: `BatchBuilder::new()`, `push_row(&Row::new())`, `finish()`. -/
def defaultBatch : List (List (Option Nat)) := [[]]

theorem defaultBatch_spec : defaultBatch.length = 1 ∧ ∀ r ∈ defaultBatch, r = [] := by
  simp [defaultBatch]

/-! ## `run_nested_plan` (:813) -/

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

/-! ## `build_batch_op` (:878) -/

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
