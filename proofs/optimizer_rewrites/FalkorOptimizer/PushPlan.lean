/-
# push_filters_down on a plan tree with write effects and scope-blind ids

Gap closed 2026-10-20 (#748, #997, #3352; #2556 in the last section,
`select_scan_node`'s chain rebuild, property `KeepsFiltersBound`). The push-down theorems in `Basic`
compare result rows only, on list operators with no graph state, and route by
`(id, scope)` (`Passes.routesScoped`). Main routes by bare ids collected from a
whole subtree, and pushes a variable-free conjunct into every child, including
the left branch of an `Apply` whose right branch writes. This file gives the
plan a tree, a graph state that writes change, and the routing exactly as Rust
computes it, so both families are visible.

| here | there (Rust at fb3d516d0) |
| --- | --- |
| `G` | the graph a query sees and writes: node count, labels, edges |
| `Op` | `IR` (`graph/src/planner/mod.rs:200-300`), the operators these repros use |
| `run` | the runtime ops, eager: `apply.rs:102-340` (right branch re-run per left row with `Argument` = that row), `optional.rs:91-292` (one null row when the branch is empty), `aggregate.rs:288` (no keys → one row even on empty input), `create.rs:101-160`, `merge.rs:67-300` (match else create), `filter.rs:48-80` (`Basic.filterRows`), `unwind.rs`, `node_by_label_scan.rs`, `cond_traverse.rs` (bound destination) |
| `gv` | `GetVariables::get_variables` (`runtime/runtime.rs:212-330`): `Aggregate` adds `names` and keeps walking, `Project` adds its outputs and `break`s |
| `branchOut` | `branch_output_variables` (`optimizer/push_filters_down.rs:59-79`) |
| `excluded` | the child exclusion list (`push_filters_down.rs:161-175`) |
| `pushAt` | one Filter's routing (`push_filters_down.rs:138-364`): grandchild variable sets (:176-185), Case 1 (:196-201), Case 2 inherited (`inh`, :210-244), augmentation of `Argument` subtrees (:247-257), the `all` test over ids (:264-276) |
| `childInh` | Case 2's ancestor walk, top-down: below an `Apply`'s right child the left subtree's ids, below its left child nothing, below `Merge` nothing (:212-243) |
| `sweep`, `pushFilters` | the restart loop (:103-370), as a top-down sweep that follows each pushed filter |
| `Cfg.main` | ids only (`collect_expr_variables`/`collect_subtree_variables`, `optimizer/mod.rs:92-110`), no write barrier |

The property each bug violates is `EffectSound`: a rewrite keeps the rows
**and** the graph the query leaves behind. `RowSound` is the old criterion,
rows only; `bug748_rows_only_blind` / `bug997_rows_only_blind` show it cannot see
the write that is skipped. `RoutesBound` is the syntactic property behind
#3352: a pushed conjunct reads only variables, by `(id, scope)`, that are
available where it lands.

Not modelled: conjunct splitting (the repros have one conjunct), filter
merging and the CP split (`Basic`, `Sequence`), laziness (operators are
eager), property access (`r.status = 'revoked'` is `r = 1`, the revoked
rule's node id; `r.decided_at` is `r`), and `Project`'s BFS `break` ending the
walk of later siblings too (here only its own subtree is skipped; no plan
below has a sibling after a `Project`).
-/
import FalkorOptimizer.Basic

namespace Falkor.Opt.PushPlan

open Falkor.Opt

/-! ## Graph state and plans -/

structure G where
  n : Nat
  lbl : List (Nat × Nat)
  /-- `(edge id, src, dst)` -/
  edges : List (Nat × Nat × Nat)
deriving DecidableEq, Repr

/-- A pattern node: its variable and (at most) one label. -/
structure Node where
  v : Var
  lbl : Option Nat
deriving DecidableEq, Repr

inductive Op where
  | argument
  /-- No child (a childless `Project` reads one empty row). -/
  | unit
  | project (bs : List (Var × Expr)) (c : Op)
  | filter (p : Expr) (c : Op)
  | unwind (v : Var) (xs : List Val) (c : Op)
  | allNodeScan (v : Var) (c : Op)
  | labelScan (v : Var) (l : Nat) (c : Op)
  /-- `(fr:l)-[e]->(to)` with `to` bound by the input row. -/
  | condTraverse (e fr to : Var) (l : Nat) (c : Op)
  /-- `IR::CondTraverse` as it carries its relationship (`fr`)-[`e`]->(`to`)
  and `transposed` (`planner/mod.rs:200-300`, `runtime/ops/cond_traverse.rs`):
  the walk starts from `fr`, read from the input row, and binds `e` and `to`
  (which must carry `to.lbl`). `transposed` reads each stored edge backwards,
  so a pattern edge `a → b` started from `b` is `hop e b a true`. -/
  | hop (e : Var) (fr to : Node) (transposed : Bool) (c : Op)
  | optional (vs : List Var) (c : Op)
  | aggregate (keys : List (Var × Expr)) (cnt : Option (Var × Expr)) (c : Op)
  | commit (c : Op)
  | create (v : Var) (c : Op)
  | merge (v : Var) (c : Op)
  | apply (l r : Op)
deriving Repr

/-! ## Semantics -/

abbrev Res := Except Err (List Row × G)

def forRows (f : Row → G → Res) : List Row → G → Res
  | [], g => .ok ([], g)
  | x :: xs, g => do
      let (ys, g1) ← f x g
      let (zs, g2) ← forRows f xs g1
      pure (ys ++ zs, g2)

def evalBs (bs : List (Var × Expr)) (r : Row) : Except Err Row :=
  bs.mapM (fun ve => do let v ← eval r ve.2; pure (ve.1, v))

def aggNames (keys : List (Var × Expr)) (cnt : Option (Var × Expr)) : List Var :=
  keys.map Prod.fst ++ (cnt.map Prod.fst).toList

/-- Grouping by the key values; `count(e)` counts non-null values. -/
def aggRows (keys : List (Var × Expr)) (cnt : Option (Var × Expr)) (rs : List Row) :
    Except Err (List Row) := do
  let kv ← rs.mapM (fun r => do
    let ks ← keys.mapM (fun ke => eval r ke.2)
    let c ← match cnt with
      | none => pure Val.null
      | some ce => eval r ce.2
    pure (ks, c))
  let groups : List (List Val) := if keys.isEmpty then [[]] else (kv.map Prod.fst).eraseDups
  pure (groups.map fun ks =>
    (keys.map Prod.fst).zip ks ++
      (match cnt with
       | none => []
       | some ce => [(ce.1, Val.int (kv.filter (fun p => p.1 == ks && p.2 != Val.null)).length)]))

def newNode (v : Var) (r : Row) (g : G) : Res :=
  .ok ([r ++ [(v, .int g.n)]], { g with n := g.n + 1 })

def run (arg : Row) : Op → G → Res
  | .argument, g => .ok ([arg], g)
  | .unit, g => .ok ([[]], g)
  | .project bs c, g => do
      let (rs, g) ← run arg c g
      let out ← rs.mapM (evalBs bs)
      pure (out, g)
  | .filter p c, g => do
      let (rs, g) ← run arg c g
      let out ← filterRows p rs
      pure (out, g)
  | .unwind v xs c, g => do
      let (rs, g) ← run arg c g
      pure (rs.flatMap (fun r => xs.map (fun x => r ++ [(v, x)])), g)
  | .allNodeScan v c, g => do
      let (rs, g) ← run arg c g
      pure (rs.flatMap (fun r => (List.range g.n).map (fun i => r ++ [(v, .int i)])), g)
  | .labelScan v l c, g => do
      let (rs, g) ← run arg c g
      pure (rs.flatMap (fun r =>
        ((List.range g.n).filter (fun i => g.lbl.contains (i, l))).map (fun i => r ++ [(v, .int i)])), g)
  | .condTraverse e fr to l c, g => do
      let (rs, g) ← run arg c g
      pure (rs.flatMap (fun r =>
        (g.edges.filter (fun t => r.get to == .int t.2.2 && g.lbl.contains (t.2.1, l))).map
          (fun t => r ++ [(e, .int t.1), (fr, .int t.2.1)])), g)
  | .optional vs c, g => do
      let (rs, g) ← run arg c g
      pure (if rs.isEmpty then [arg ++ vs.map (fun v => (v, .null))] else rs, g)
  | .aggregate keys cnt c, g => do
      let (rs, g) ← run arg c g
      let out ← aggRows keys cnt rs
      pure (out, g)
  | .hop e fr to tr c, g => do
      let (rs, g) ← run arg c g
      pure (rs.flatMap (fun r =>
        (g.edges.filter (fun t => r.get fr.v == .int (Int.ofNat (if tr then t.2.2 else t.2.1)) &&
            to.lbl.all (fun l => g.lbl.contains (if tr then t.2.1 else t.2.2, l)))).map
          (fun t => r ++ [(e, .int t.1), (to.v, .int (Int.ofNat (if tr then t.2.1 else t.2.2)))])), g)
  | .commit c, g => run arg c g
  | .create v c, g => do
      let (rs, g) ← run arg c g
      forRows (newNode v) rs g
  | .merge v c, g => do
      let (rs, g) ← run arg c g
      forRows (fun r g => if g.n = 0 then newNode v r g
        else .ok ((List.range g.n).map (fun i => r ++ [(v, .int i)]), g)) rs g
  | .apply l r, g => do
      let (xs, g) ← run arg l g
      forRows (fun x g => do
        let (ys, g) ← run x r g
        pure (ys.map (fun y => x ++ y), g)) xs g

/-! ## The correctness criteria -/

/-- The old criterion (every theorem in `Basic`): a rewrite keeps the result
rows, or both fail. It ignores what the query wrote. -/
def RowSound (pass : Op → Op) : Prop :=
  ∀ op g, Same ((run [] (pass op) g).map Prod.fst) ((run [] op g).map Prod.fst)

/-- **The property a plan rewrite must have**: it keeps the result rows *and*
the graph the query leaves behind (every write still happens, exactly as
often), or both fail. A fix of #748/#997/#3352 proves this for
`pushFilters` (with `Cfg.main` replaced by the fixed routing). -/
def EffectSound (pass : Op → Op) : Prop :=
  ∀ op g, Same (run [] (pass op) g) (run [] op g)

theorem EffectSound.rowSound {pass : Op → Op} (h : EffectSound pass) : RowSound pass := by
  intro op g
  have := h op g
  revert this
  cases run [] (pass op) g <;> cases run [] op g <;> simp_all [Same, Except.map]

/-! ## push_filters_down -/

/-- Routing configuration: how a conjunct's variable is matched against a
child's variable set, and whether a write blocks the push. -/
structure Cfg where
  eqv : Var → Var → Bool
  writeBarrier : Bool

/-- Rust main: ids only, no barrier. -/
def Cfg.main : Cfg := ⟨fun a b => a.id == b.id, false⟩

/-- A candidate fix: `(id, scope)` matching and no push below an operator whose
subtree writes (`Merge`/`Create`). -/
def Cfg.fixed : Cfg := ⟨fun a b => a == b, true⟩

/-- `get_variables` (runtime.rs:212-330). -/
def gv : Op → List Var
  | .argument | .unit => []
  | .project bs _ => bs.map Prod.fst
  | .filter _ c | .commit c => gv c
  | .unwind v _ c | .allNodeScan v c | .labelScan v _ c | .create v c | .merge v c => v :: gv c
  | .condTraverse e fr to _ c => [e, fr, to] ++ gv c
  | .hop e fr to _ c => [e, fr.v, to.v] ++ gv c
  | .optional vs c => vs ++ gv c
  | .aggregate keys cnt c => aggNames keys cnt ++ gv c
  | .apply l r => gv l ++ gv r

/-- `branch_output_variables` (push_filters_down.rs:59-79). -/
def branchOut : Op → Option (List Var)
  | .project bs _ => some (bs.map Prod.fst)
  | .aggregate keys cnt _ => some (aggNames keys cnt)
  | _ => none

/-- `subtree_contains(…, Argument)` (planner/mod.rs:333). -/
def hasArg : Op → Bool
  | .argument => true
  | .unit => false
  | .project _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c | .labelScan _ _ c
  | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c | .commit c
  | .create _ c | .merge _ c => hasArg c
  | .apply l r => hasArg l || hasArg r

def hasWrite : Op → Bool
  | .argument | .unit => false
  | .create _ _ | .merge _ _ => true
  | .project _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c | .labelScan _ _ c
  | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c | .commit c =>
    hasWrite c
  | .apply l r => hasWrite l || hasWrite r

/-- push_filters_down.rs:161-175 (the kinds that occur here). -/
def excluded : Op → Bool
  | .project _ _ | .aggregate _ _ _ | .merge _ _ | .argument | .optional _ _ => true
  | _ => false

def kid (c : Op) : List Op := match c with | .unit => [] | c => [c]

/-- `children()` of an operator. -/
def kids : Op → List Op
  | .argument | .unit => []
  | .project _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c | .labelScan _ _ c
  | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c | .commit c
  | .create _ c | .merge _ c => kid c
  | .apply l r => [l, r]

/-- Replace the children (same shape). -/
def setKids (op : Op) (ks : List Op) : Op :=
  match op, ks with
  | .project bs _, [c] => .project bs c
  | .filter p _, [c] => .filter p c
  | .unwind v xs _, [c] => .unwind v xs c
  | .allNodeScan v _, [c] => .allNodeScan v c
  | .labelScan v l _, [c] => .labelScan v l c
  | .condTraverse e fr to l _, [c] => .condTraverse e fr to l c
  | .hop e fr to tr _, [c] => .hop e fr to tr c
  | .optional vs _, [c] => .optional vs c
  | .aggregate k n _, [c] => .aggregate k n c
  | .commit _, [c] => .commit c
  | .create v _, [c] => .create v c
  | .merge v _, [c] => .merge v c
  | .apply _ _, [l, r] => .apply l r
  | op, _ => op

def isApply : Op → Bool
  | .apply _ _ => true
  | _ => false

/-- A grandchild's variable set and whether it resets the env (:176-185). -/
def info (k : Op) : List Var × Bool :=
  match branchOut k with
  | some o => (o, true)
  | none => (gv k, false)

/-- Does the conjunct route into a child with variable set `vs` (:264-276)? -/
def routes (cfg : Cfg) (p : Expr) (vs : List Var) : Bool :=
  p.vars.all (fun v => vs.any (cfg.eqv v))

/-- The variable set the routing test sees for child `k` (:247-257). -/
def avail (inh : List Var) (k : Op) : List Var :=
  let (vs, reset) := info k
  if !inh.isEmpty && !reset && hasArg k then vs ++ inh else vs

/-- One Filter `p` over `c`, with Case-2 inherited variables `inh`. -/
def pushAt (cfg : Cfg) (inh : List Var) (p : Expr) (c : Op) : Op :=
  let ks := kids c
  if excluded c || ks.isEmpty || (cfg.writeBarrier && hasWrite c) then .filter p c else
  let inh := (if isApply c then ((ks.head?.map info).map Prod.fst).getD [] else []) ++ inh
  let rs := ks.map (fun k => routes cfg p (avail inh k))
  if rs.any id then setKids c ((ks.zip rs).map (fun kb => if kb.2 then .filter p kb.1 else kb.1))
  else .filter p c

/-- Case 2, top-down (:212-243). -/
def childInh (op : Op) (i : Nat) (inh : List Var) : List Var :=
  match op with
  | .apply l _ => if i = 0 then [] else gv l
  | .merge _ _ => []
  | _ => inh

def mapKids (f : Nat → Op → Op) (op : Op) : Op :=
  setKids op ((kids op).zipIdx.map (fun ki => f ki.2 ki.1))

def sweep (cfg : Cfg) : Nat → List Var → Op → Op
  | 0, _, op => op
  | fuel + 1, inh, op =>
    match op with
    | .filter p c =>
      match pushAt cfg inh p c with
      | .filter p' c' => .filter p' (sweep cfg fuel inh c')
      | t => mapKids (fun i k => sweep cfg fuel (childInh t i inh) k) t
    | op => mapKids (fun i k => sweep cfg fuel (childInh op i inh) k) op

def pushFilters (cfg : Cfg) (op : Op) : Op := sweep cfg 64 [] op

/-- The syntactic property behind #3352: a conjunct may land on a child only
if every variable it reads, by `(id, scope)`, is in the variables actually
available there. Main routes with `Cfg.main` (ids) against `avail`, which for
an `Argument` subtree adds every id of the Apply's left subtree. -/
def RoutesBound (p : Expr) (available : List Var) : Prop := ∀ v ∈ p.vars, v ∈ available

/-! ## #748: `WITH 0 AS n0 CALL { MERGE () RETURN 0 AS n1 } MATCH () WHERE false RETURN 0` -/

def g0 : G := ⟨0, [], []⟩
def fls : Expr := .lit (.bool false)

def n0 : Var := ⟨0, 0⟩
def a0 : Var := ⟨0, 1⟩
def n1 : Var := ⟨1, 1⟩
def a1 : Var := ⟨1, 0⟩
def col : Var := ⟨2, 0⟩

def body748 : Op := .project [(n1, .lit (.int 0))] (.commit (.merge a0 (.project [] .argument)))
def apply748 : Op := .apply (.project [(n0, .lit (.int 0))] .unit) body748

/-- Before the optimizer: `Project(Filter(false, AllNodeScan(Apply(…))))`. -/
def plan748 : Op := .project [(col, .lit (.int 0))] (.filter fls (.allNodeScan a1 apply748))

/-- The live Rust EXPLAIN: the filter sits on both Apply branches. -/
def plan748opt : Op :=
  .project [(col, .lit (.int 0))]
    (.allNodeScan a1 (.apply (.filter fls (.project [(n0, .lit (.int 0))] .unit)) (.filter fls body748)))

theorem bug748_plan : pushFilters .main plan748 = plan748opt := rfl

/-- **#748.** Before: one node is merged, no rows. After push-down: no node,
no rows. Rust main creates 0 nodes, C 1 (live, 9d3869aa8). -/
theorem bug748_merge_skipped :
    run [] plan748 g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .main plan748) g0 = .ok ([], g0) ∧
    ¬ EffectSound (pushFilters .main) := by
  refine ⟨rfl, rfl, fun h => ?_⟩
  have := h plan748 g0
  rw [show run [] (pushFilters .main plan748) g0 = .ok ([], g0) from rfl,
      show run [] plan748 g0 = .ok ([], ⟨1, [], []⟩) from rfl] at this
  simp [Same, g0] at this

/-- The rows-only criterion holds on this instance: it cannot see the bug. -/
theorem bug748_rows_only_blind :
    Same ((run [] (pushFilters .main plan748) g0).map Prod.fst) ((run [] plan748 g0).map Prod.fst) := rfl

/-- `… CALL {…} WITH * MATCH () WHERE false …`: the `Project` stops the push,
the MERGE runs (live: 1 node on Rust and C). -/
def plan748star : Op :=
  .project [(col, .lit (.int 0))]
    (.filter fls (.allNodeScan a1 (.project [(n0, .var n0), (n1, .var n1)] apply748)))

theorem bug748_with_star_control :
    pushFilters .main plan748star =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.filter fls (.project [(n0, .var n0), (n1, .var n1)] apply748))) ∧
    run [] (pushFilters .main plan748star) g0 = .ok ([], ⟨1, [], []⟩) := ⟨rfl, rfl⟩

theorem bug748_fixed :
    run [] (pushFilters .fixed plan748) g0 = run [] plan748 g0 := rfl

/-! ## #997: `CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE false RETURN 0` -/

def x : Var := ⟨0, 0⟩
def c0 : Var := ⟨0, 1⟩
def apply997 : Op := .apply .argument (.aggregate [] none (.commit (.create c0 (.project [] .argument))))

def plan997With (p : Expr) : Op :=
  .project [(col, .lit (.int 0))] (.filter p (.allNodeScan a1 (.unwind x [.int 0] apply997)))

def plan997 : Op := plan997With fls

/-- The live Rust EXPLAIN: `Apply(Filter(Argument), Filter(Aggregate(…)))`. -/
theorem bug997_plan :
    pushFilters .main plan997 =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.unwind x [.int 0]
          (.apply (.filter fls .argument)
            (.filter fls (.aggregate [] none (.commit (.create c0 (.project [] .argument)))))))) := rfl

/-- **#997.** Before: one node created. After: none. Rust main 0 nodes, C 1 (live). -/
theorem bug997_create_skipped :
    run [] plan997 g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .main plan997) g0 = .ok ([], g0) ∧
    ¬ EffectSound (pushFilters .main) := by
  refine ⟨rfl, rfl, fun h => ?_⟩
  have := h plan997 g0
  rw [show run [] (pushFilters .main plan997) g0 = .ok ([], g0) from rfl,
      show run [] plan997 g0 = .ok ([], ⟨1, [], []⟩) from rfl] at this
  simp [Same, g0] at this

theorem bug997_rows_only_blind :
    Same ((run [] (pushFilters .main plan997) g0).map Prod.fst) ((run [] plan997 g0).map Prod.fst) := rfl

/-- `WHERE (CASE WHEN true THEN false ELSE x END)`: the conjunct reads `x`
(here `false AND x`, the same verdict). It is pushed below `Unwind` (id 0 of
`x` collides with the body's `CREATE ()` variable, id 0 of scope 1, as in the
live EXPLAIN `Unwind → Filter → Apply`), but not into the Apply, so the
CREATE runs: 1 node on Rust and C (live). -/
def caseX : Expr := .and fls (.var x)

theorem bug997_case_control :
    pushFilters .main (plan997With caseX) =
      .project [(col, .lit (.int 0))] (.allNodeScan a1 (.unwind x [.int 0] (.filter caseX apply997))) ∧
    run [] (pushFilters .main (plan997With caseX)) g0 = .ok ([], ⟨1, [], []⟩) := ⟨rfl, rfl⟩

theorem bug997_fixed :
    run [] (pushFilters .fixed plan997) g0 = run [] plan997 g0 := rfl

/-! ## #3352: OPTIONAL MATCH … WHERE after an aggregating WITH

```
MATCH (p:Person) OPTIONAL MATCH (r0:Rule)-[:SCOPED_TO]->(p)
WITH p, count(r0) AS n
OPTIONAL MATCH (r:Rule)-[:SCOPED_TO]->(p) WHERE r.status = 'revoked'
RETURN n, r.decided_at AS e
```
Ids are scope-local (`binder.rs:2257-2268`, id = env length): before the WITH
`p, r0, _anon` are 0, 1, 2 of scope 0; after it `p, n, r, _anon` are 0, 1, 2, 3
of scope 1. `get_variables` walks through the `Aggregate` into the pre-WITH
subtree, so Case 2 inherits id 2 and believes `r` is bound at the `Argument`. -/

def PERSON : Nat := 0
def RULE : Nat := 1
/-- Person 0, Rule 1 (revoked), edge 0: 1 → 0. -/
def g3352 : G := ⟨2, [(0, PERSON), (1, RULE)], [(0, 1, 0)]⟩

def p : Var := ⟨0, 0⟩
def r0 : Var := ⟨1, 0⟩
def e0 : Var := ⟨2, 0⟩
def p' : Var := ⟨0, 1⟩
def nV : Var := ⟨1, 1⟩
def r : Var := ⟨2, 1⟩
def e1 : Var := ⟨3, 1⟩
def nOut : Var := ⟨0, 2⟩
def eOut : Var := ⟨1, 2⟩

def revoked : Expr := .eq (.var r) (.lit (.int 1))

def agg3352 : Op :=
  .aggregate [(p', .var p)] (some (nV, .var r0))
    (.apply (.labelScan p PERSON .unit) (.optional [r0, e0] (.condTraverse e0 r0 p RULE .argument)))

def plan3352 : Op :=
  .project [(nOut, .var nV), (eOut, .var r)]
    (.apply agg3352 (.optional [r, e1] (.filter revoked (.condTraverse e1 r p' RULE .argument))))

/-- The live Rust EXPLAIN: `Optional → Conditional Traverse → Filter → Argument`. -/
theorem bug3352_plan :
    pushFilters .main plan3352 =
      .project [(nOut, .var nV), (eOut, .var r)]
        (.apply agg3352 (.optional [r, e1] (.condTraverse e1 r p' RULE (.filter revoked .argument)))) := rfl

/-- Why: what the routing test sees at the `Argument` contains id 2 (the
pre-WITH `_anon` edge), while by `(id, scope)` only `p, n` of scope 1 arrive
there (the `Aggregate`'s outputs). `RoutesBound` fails. -/
theorem bug3352_routes_unbound :
    avail (gv agg3352) .argument = gv agg3352 ∧ e0 ∈ gv agg3352 ∧
    routes .main revoked (gv agg3352) = true ∧
    ¬ RoutesBound revoked (aggNames [(p', .var p)] (some (nV, .var r0)) ++ gv .argument) := by
  refine ⟨rfl, by decide, rfl, ?_⟩
  intro h
  have := h r (by simp [revoked, Expr.vars])
  simp [aggNames, gv, r, p', nV] at this

/-- **#3352.** Before: `n = 1, e = 1` (the revoked rule; C: `e = 123`). After:
`n = 1, e = null` (Rust main, live). The graph is unchanged, so this is a
row difference: `pushFilters .main` is not even `RowSound`. -/
theorem bug3352_slot_collision_drops_match :
    run [] plan3352 g3352 = .ok ([[(nOut, .int 1), (eOut, .int 1)]], g3352) ∧
    run [] (pushFilters .main plan3352) g3352 = .ok ([[(nOut, .int 1), (eOut, .null)]], g3352) ∧
    ¬ RowSound (pushFilters .main) := by
  refine ⟨rfl, rfl, fun h => ?_⟩
  have := h plan3352 g3352
  rw [show run [] (pushFilters .main plan3352) g3352 = .ok ([[(nOut, .int 1), (eOut, .null)]], g3352) from rfl,
      show run [] plan3352 g3352 = .ok ([[(nOut, .int 1), (eOut, .int 1)]], g3352) from rfl] at this
  simp [Same, Except.map] at this

/-- Routing by `(id, scope)` keeps the filter above the traverse. -/
theorem bug3352_fixed :
    pushFilters .fixed plan3352 = plan3352 ∧
    run [] (pushFilters .fixed plan3352) g3352 = run [] plan3352 g3352 := ⟨rfl, rfl⟩

/-! ## A general fact the fix can build on -/

/-- `forRows` of a step that never changes the graph does not change it. -/
theorem forRows_keeps (f : Row → G → Res) (hf : ∀ x g rs g', f x g = .ok (rs, g') → g' = g) :
    ∀ xs g rs g', forRows f xs g = .ok (rs, g') → g' = g
  | [], g, rs, g', h => by simp [forRows] at h; exact h.2.symm
  | x :: xs, g, rs, g', h => by
    simp only [forRows, bind, Except.bind] at h
    rcases hfx : f x g with e | ⟨ys, g1⟩ <;> rw [hfx] at h
    · cases h
    · try simp only at h
      rcases hr : forRows f xs g1 with e | ⟨zs, g2⟩ <;> rw [hr] at h
      · cases h
      · simp [pure, Except.pure] at h
        rw [← h.2, forRows_keeps f hf xs g1 zs g2 hr, hf x g ys g1 hfx]

/-- A plan with no `Create`/`Merge` leaves the graph unchanged, so between
write-free plans `EffectSound` reduces to the rows criterion: the push-down
theorems of `Basic` carry over to read-only plans, and the write barrier of
`Cfg.fixed` only matters where a write is below. -/
theorem run_readonly : ∀ (op : Op) (arg : Row) (g : G) rs g',
    hasWrite op = false → run arg op g = .ok (rs, g') → g' = g := by
  intro op
  induction op with
  | argument | unit =>
    intro arg g rs g' _ h; simp [run] at h; exact h.2.symm
  | create | merge => intro _ _ _ _ hw; simp [hasWrite] at hw
  | apply l r ihl ihr =>
    intro arg g rs g' hw h
    simp only [hasWrite, Bool.or_eq_false_iff] at hw
    simp only [run, bind, Except.bind] at h
    rcases hl : run arg l g with e | ⟨xs, g1⟩ <;> rw [hl] at h
    · cases h
    · try simp only at h
      rw [forRows_keeps _ (fun x g ys g' hx => by
          try simp only [bind, Except.bind] at hx
          rcases hr : run x r g with e | ⟨zs, g2⟩ <;> rw [hr] at hx
          · cases hx
          · simp [pure, Except.pure] at hx; rw [← hx.2]; exact ihr x g zs g2 hw.2 hr) xs g1 rs g' h]
      exact ihl arg g xs g1 hw.1 hl
  | commit c ih => intro arg g rs g' hw h; exact ih arg g rs g' hw h
  | project _ c ih | filter _ c ih | unwind _ _ c ih | allNodeScan _ c ih | labelScan _ _ c ih
  | condTraverse _ _ _ _ c ih | hop _ _ _ _ c ih | optional _ c ih | aggregate _ _ c ih =>
    intro arg g rs g' hw h
    simp only [run, bind, Except.bind] at h
    rcases hc : run arg c g with e | ⟨xs, g1⟩ <;> rw [hc] at h
    · cases h
    · try simp only at h
      first
      | (rcases hm : List.mapM (evalBs _) xs with e | ys <;> rw [hm] at h
         · cases h
         · simp [pure, Except.pure] at h; rw [← h.2]; exact ih arg g xs g1 hw hc)
      | (rcases hm : filterRows _ xs with e | ys <;> rw [hm] at h
         · cases h
         · simp [pure, Except.pure] at h; rw [← h.2]; exact ih arg g xs g1 hw hc)
      | (rcases hm : aggRows _ _ xs with e | ys <;> rw [hm] at h
         · cases h
         · simp [pure, Except.pure] at h; rw [← h.2]; exact ih arg g xs g1 hw hc)
      | (simp [pure, Except.pure] at h; rw [← h.2]; exact ih arg g xs g1 hw hc)


/-! ## #2556: select_scan_node re-orders a chain and lands an inline filter early

```
MATCH (x:Person {id:2}) WITH collect(x) AS fs UNWIND fs AS f
MATCH (f)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id:100}),
      (post)-[:HAS_TAG]->(tag:Tag)
RETURN count(post)
```
The planner emits the chain bottom-up `(post)->(f)`, `(post)->(t:Tag)`,
`Filter(t.id = 100)`, `(post)->(tag:Tag)` over `Unwind` (the inline-attr
Filter directly above the hop that binds `t`, `planner/mod.rs:2178-2186`).
`select_scan_node` then re-orders the chain (`select_scan_node.rs:681-1260`):

* the best endpoint is `t`: `f` scores 3 (bound by `Unwind`), `t` 3
  (filtered 2 + labelled 1), and `t` wins on label cardinality (`f` has no
  label, so `u64::MAX`) (:849-873, `score_endpoint` :98-132);
* `best_pos = 1`, so the chain is rebuilt (:1179), on top of the kept outer
  child (`Unwind`, not a planner scan, :947-970);
* `initial_bound` is `{best} ∪ ids(child)` (:979-990). With the child kept,
  no scan for `t` is ever inserted, but `t` counts as bound;
* each inter-hop Filter is placed as soon as its ids are in `placed_bound`
  (:1202-1230): `t.id = 100` reads only `t`, "bound" from the start, so it
  lands right after the first hop, before the hop that binds `t`.

Ids are scope-local (live, a planner dump at fb3d516d0): `fs, f, post, t, tag`
are 0..4 of scope 1, the edges 5..7; `x` is 0 of scope 0. No id collides with
`t`, so the scope-blind ids of #3352 are not the cause here; the planted best
endpoint is. (In the issue's own query a pre-WITH `_anon_0` also has id 2 =
`post`, which additionally makes the first hop start from the unbound `post`;
that gives the "was List" error and is not modelled.)

| here | there (Rust at fb3d516d0) |
| --- | --- |
| `Hop` | a chain hop's `relationship` (alias, from, to), `rels` (:900-913) |
| `peelFilters`, `spine` | the chain walk through single-child Filters (:742-768) and the inter-CT Filters (:916-935) |
| `outAliases` | `collect_output_aliases` (:391-440) → `bound_vars` (:779-795) |
| `scoreE`, `cardE` | `score_endpoint` (:98-132) |
| `candidates`, `bestKey`, `maxByKey` | the endpoint candidates deduplicated by id (:810-833) and `max_by` (:849-873; Rust's `max_by` returns the last maximum) |
| `orderHops` | the greedy hop ordering (:996-1056): tier 1 both ends bound, else best-scoring new endpoint, lowest index on ties |
| `place` | the rebuild (:1196-1235): stack each hop, then every pending filter whose ids are all in `placed_bound` |
| `selectChain` | the `best_pos > 0` rebuild branch (:1179); other branches (bottom swap, leaf/scan child) return the plan unchanged |
| `ssCfg.main` / `ssCfg.fixed` | `initial_bound` with / without the best endpoint when the outer child is kept |
-/

abbrev Hop := Var × Node × Node

def BIG : Nat := 2 ^ 64

/-- `collect_output_aliases` (select_scan_node.rs:391-440), the kinds used
here; scans and `Filter` children give no bound vars (:779-795). -/
def outAliases : Op → List Nat
  | .unwind v _ _ => [v.id]
  | .aggregate keys cnt _ => (aggNames keys cnt).map Var.id
  | .project bs _ => bs.map (·.1.id)
  | _ => []

/-- `score_endpoint` (:98-132). `bound_vars` by bare id, `filtered` by
`(id, scope)`. -/
def scoreE (boundVars : List Nat) (filtered : List Var) (n : Node) : Nat :=
  (if boundVars.contains n.v.id then 3 else 0) + (if filtered.contains n.v then 2 else 0) +
    (if n.lbl.isSome then 1 else 0)

def cardE (cnt : Nat → Nat) (n : Node) : Nat :=
  match n.lbl with
  | some l => cnt l
  | none => BIG

/-- Lexicographic `≥` on keys. -/
def lexGe : List Nat → List Nat → Bool
  | a :: as, b :: bs => if a = b then lexGe as bs else decide (a > b)
  | _, _ => true

/-- `Iterator::max_by`: on ties the last maximal element. -/
def maxByKey {α : Type} (key : α → List Nat) : List α → Option α
  | [] => none
  | x :: xs => some (xs.foldl (fun best y => if lexGe (key y) (key best) then y else best) x)

def peelFilters : Op → List Expr × Op
  | .filter p c => let r := peelFilters c; (p :: r.1, r.2)
  | op => ([], op)

/-- The chain from its top hop: hops bottom-up, inter-hop filters, the child
of the bottom hop. -/
def spine : Nat → Op → List Hop × List Expr × Op
  | fuel + 1, .hop e fr to _ c =>
    match peelFilters c with
    | (ps, r@(.hop ..)) =>
      let (hs, fs, ch) := spine fuel r
      (hs ++ [(e, fr, to)], fs ++ ps, ch)
    | _ => ([(e, fr, to)], [], c)
  | _, op => ([], [], op)

/-- Endpoint candidates `(node, chain position, is_from)`, first occurrence of
each id (:810-833). -/
def candidates (hops : List Hop) : List (Node × Nat × Bool) :=
  let all := (hops.zipIdx).flatMap (fun hi => [(hi.1.2.1, hi.2, true), (hi.1.2.2, hi.2, false)])
  (all.foldl (fun (acc : List (Node × Nat × Bool) × List Nat) c =>
    if acc.2.contains c.1.v.id then acc else (acc.1 ++ [c], c.1.v.id :: acc.2)) ([], [])).1

/-- The `max_by` key of :849-873: score, runs-late, lower cardinality, lower
position, `from` side. -/
def bestKey (cnt : Nat → Nat) (bv : List Nat) (above all : List Var) (c : Node × Nat × Bool) :
    List Nat :=
  [scoreE bv all c.1, if above.contains c.1.v then 1 else 0, BIG - cardE cnt c.1, BIG - c.2.1,
   if c.2.2 then 1 else 0]

/-- The greedy ordering (:996-1056). Returns the oriented hops bottom-up with
their `transposed` flag, or `none` when no hop can start (`orderable = false`). -/
def orderHops (cnt : Nat → Nat) (bv : List Nat) (all : List Var) :
    Nat → List Nat → List Hop → Option (List (Hop × Bool))
  | _, _, [] => some []
  | 0, _, _ => none
  | fuel + 1, bound, pending =>
    let cands := (pending.zipIdx).filterMap (fun hi =>
      let fb := bound.contains hi.1.2.1.v.id
      let tb := bound.contains hi.1.2.2.v.id
      if !fb && !tb then none
      else if fb && tb then some (hi.2, [1, BIG, 0, BIG - hi.2])
      else
        let dest := if fb then hi.1.2.2 else hi.1.2.1
        some (hi.2, [0, scoreE bv all dest, BIG - cardE cnt dest, BIG - hi.2]))
    match maxByKey Prod.snd cands with
    | none => none
    | some (i, _) =>
      match pending[i]? with
      | none => none
      | some h =>
        let (h', tr) : Hop × Bool :=
          if bound.contains h.2.1.v.id then (h, false) else ((h.1, h.2.2, h.2.1), true)
        ((h', tr) :: ·) <$> orderHops cnt bv all fuel (h'.2.2.v.id :: h'.1.id :: bound) (pending.eraseIdx i)

/-- The rebuild (:1196-1235): stack the hops; after each, every pending filter
whose ids are all in `placed_bound` goes on top. A filter left over abandons
the rewrite (`none`). -/
def place : List (Hop × Bool) → Op → List Nat → List Expr → Option Op
  | [], sub, _, fs => if fs.isEmpty then some sub else none
  | (h, tr) :: hs, sub, bound, fs =>
    let sub := Op.hop h.1 h.2.1 h.2.2 tr sub
    let bound := h.2.2.v.id :: h.1.id :: bound
    let now := fs.filter (fun p => p.vars.all (fun v => bound.contains v.id))
    let later := fs.filter (fun p => !p.vars.all (fun v => bound.contains v.id))
    place hs (now.foldl (fun s p => .filter p s) sub) bound later

/-- Is the best endpoint seeded into `initial_bound` when the outer child is
kept? Main: yes (:979-990). A fix: only when a scan for it is inserted. -/
structure SsCfg where
  seedBest : Bool

def SsCfg.main : SsCfg := ⟨true⟩
def SsCfg.fixed : SsCfg := ⟨false⟩

/-- One chain whose bottom hop keeps an outer (non-scan) child: the
`best_pos > 0` rebuild branch. -/
def selectChain (cfg : SsCfg) (cnt : Nat → Nat) (above : List Var) (op : Op) : Op :=
  let (hops, fs, child) := spine 16 op
  let bv := outAliases child
  let all := above ++ fs.flatMap Expr.vars
  match maxByKey (bestKey cnt bv above all) (candidates hops) with
  | none => op
  | some (bn, pos, _) =>
    if pos = 0 then op else
    let ib := (if cfg.seedBest then [bn.v.id] else []) ++ (gv child).map Var.id
    match orderHops cnt bv all hops.length ib hops with
    | none => op
    | some ord => (place ord child ib fs).getD op

/-- The pass over a plan: the first chain met from the root, with the
variables of the Filters directly above it (`collect_filtered_vars` `above`,
:156-203). -/
def selectScan (cfg : SsCfg) (cnt : Nat → Nat) : Nat → List Var → Op → Op
  | 0, _, op => op
  | fuel + 1, above, op =>
    match op with
    | .hop .. => selectChain cfg cnt above op
    | .filter p c => .filter p (selectScan cfg cnt fuel (above ++ p.vars) c)
    | op => mapKids (fun _ k => selectScan cfg cnt fuel [] k) op

/-- What a Filter may read: the variables, by `(id, scope)`, that the rows
reaching it actually carry. `Project`/`Aggregate` reset the row. -/
def rowVars : Op → List Var
  | .argument | .unit => []
  | .project bs _ => bs.map Prod.fst
  | .aggregate keys cnt _ => aggNames keys cnt
  | .filter _ c | .commit c => rowVars c
  | .unwind v _ c | .allNodeScan v c | .labelScan v _ c | .create v c | .merge v c => v :: rowVars c
  | .condTraverse e fr _ _ c => [e, fr] ++ rowVars c
  | .hop e _ to _ c => [e, to.v] ++ rowVars c
  | .optional vs c => vs ++ rowVars c
  | .apply l r => rowVars l ++ rowVars r

/-- Every `Filter` reads only variables bound below it. -/
def filtersBound : Op → Bool
  | .argument | .unit => true
  | .filter p c => p.vars.all (fun v => (rowVars c).contains v) && filtersBound c
  | .project _ c | .unwind _ _ c | .allNodeScan _ c | .labelScan _ _ c | .condTraverse _ _ _ _ c
  | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c | .commit c | .create _ c | .merge _ c =>
    filtersBound c
  | .apply l r => filtersBound l && filtersBound r

/-- **The property #2556 violates.** A plan rewrite never moves a Filter below
the operator that binds a variable it reads: every Filter of a plan whose
Filters all read bound variables still does after the rewrite. A fix proves it
for `selectScan` (and `pushFilters`). -/
def KeepsFiltersBound (pass : Op → Op) : Prop :=
  ∀ op, filtersBound op = true → filtersBound (pass op) = true

/-- Labels and graph of the issue (ids are node indices; `t.id = 100` is
`t = 4`, the one Tag with id 100). Nodes: 0 person, 1 f (both Person), 2 m1,
3 m2 (Post), 4 t1, 5 t2 (Tag). Edges: KNOWS 0→1; HAS_CREATOR 2→1, 3→1;
HAS_TAG 2→4, 2→5, 3→4. -/
def PERSON' : Nat := 0
def POST : Nat := 1
def TAG : Nat := 2
def g2556 : G :=
  ⟨6, [(0, PERSON'), (1, PERSON'), (2, POST), (3, POST), (4, TAG), (5, TAG)],
   [(0, 0, 1), (1, 2, 1), (2, 3, 1), (3, 2, 4), (4, 2, 5), (5, 3, 4)]⟩
/-- `label_node_count`: 2 Person, 2 Post, 2 Tag. -/
def cnt2556 : Nat → Nat := fun _ => 2

def xV : Var := ⟨0, 0⟩
def fsV : Var := ⟨0, 1⟩
def fV : Var := ⟨1, 1⟩
def postV : Var := ⟨2, 1⟩
def tV : Var := ⟨3, 1⟩
def tagV : Var := ⟨4, 1⟩
def e5 : Var := ⟨5, 1⟩
def e6 : Var := ⟨6, 1⟩
def e7 : Var := ⟨7, 1⟩
def cntV : Var := ⟨0, 2⟩

def fN : Node := ⟨fV, none⟩
def postN : Node := ⟨postV, some POST⟩
def tN : Node := ⟨tV, some TAG⟩
def tagN : Node := ⟨tagV, some TAG⟩

/-- `t.id = 100`. -/
def tFilter : Expr := .eq (.var tV) (.lit (.int 4))

/-- `… WITH collect(x) AS fs UNWIND fs AS f`: the one `x` is node 1, so `f`
ranges over `[1]` (collect/unwind of node lists is not modelled; `count(x)`
stands in for `collect(x)`, the same names and ids). -/
def child2556 : Op :=
  .unwind fV [.int 1]
    (.aggregate [] (some (fsV, .var xV)) (.filter (.eq (.var xV) (.lit (.int 1))) (.labelScan xV PERSON' .unit)))

/-- The planner's plan (the pattern's relationships, not transposed; the
Filter directly above the hop binding `t`). -/
def plan2556 : Op :=
  .aggregate [] (some (cntV, .var postV))
    (.hop e7 postN tagN false (.filter tFilter (.hop e6 postN tN false (.hop e5 postN fN false child2556))))

/-- The plan C runs (live C EXPLAIN shape): each hop from its bound end, the
filter above the hop that binds `t`. -/
def spec2556 : Op :=
  .aggregate [] (some (cntV, .var postV))
    (.hop e7 postN tagN false (.filter tFilter (.hop e6 postN tN false (.hop e5 fN postN true child2556))))

/-- The live Rust EXPLAIN: `Aggregate → (post)->(tag:Tag) → (post)->(t:Tag) →
Filter → (f)<-(post:Post) → Unwind`. -/
def opt2556 : Op :=
  .aggregate [] (some (cntV, .var postV))
    (.hop e7 postN tagN false (.hop e6 postN tN false (.filter tFilter (.hop e5 fN postN true child2556))))

theorem bug2556_plan : selectScan .main cnt2556 16 [] plan2556 = opt2556 := rfl

/-- Why: `t` is the best endpoint (score 3 like `f`, lower cardinality), and
it is seeded into `initial_bound` although no scan for it is built. -/
theorem bug2556_t_seeded :
    maxByKey (bestKey cnt2556 [fV.id] [] [tV]) (candidates (spine 16 (.hop e7 postN tagN false
      (.filter tFilter (.hop e6 postN tN false (.hop e5 postN fN false child2556))))).1) =
      some (tN, 1, false) ∧
    outAliases child2556 = [fV.id] ∧ tV.id ∉ (gv child2556).map Var.id := by decide

/-- **#2556.** The planner's plan has every Filter over a bound variable; after
`select_scan_node` `t.id = 100` sits below the hop that binds `t`. It reads
NULL, every row is dropped: `count(post)` = 0 where the C-shaped plan gives 3.
Live (fb3d516d0, this query): Rust 0, C 3. -/
theorem bug2556_filter_below_binding :
    filtersBound plan2556 = true ∧
    filtersBound (selectScan .main cnt2556 16 [] plan2556) = false ∧
    run [] spec2556 g2556 = .ok ([[(cntV, .int 3)]], g2556) ∧
    run [] (selectScan .main cnt2556 16 [] plan2556) g2556 = .ok ([[(cntV, .int 0)]], g2556) ∧
    ¬ KeepsFiltersBound (selectScan .main cnt2556 16 []) := by
  refine ⟨rfl, rfl, rfl, rfl, fun h => ?_⟩
  have := h plan2556 rfl
  rw [show filtersBound (selectScan .main cnt2556 16 [] plan2556) = false from rfl] at this
  cases this

/-- Not seeding the unscanned best endpoint orders the hops from `f`, then `t`
(filtered and labelled) before `tag`, and lands the filter right above `t`'s
hop: exactly the C plan. -/
theorem bug2556_fixed :
    selectScan .fixed cnt2556 16 [] plan2556 = spec2556 ∧
    filtersBound (selectScan .fixed cnt2556 16 [] plan2556) = true ∧
    run [] (selectScan .fixed cnt2556 16 [] plan2556) g2556 = .ok ([[(cntV, .int 3)]], g2556) :=
  ⟨rfl, rfl, rfl⟩

/-- `push_filters_down` is not the cause: on the planner's plan it leaves
`t.id = 100` above `t`'s hop (the only child of that hop binds no `t`, by id). -/
theorem bug2556_push_not_cause : pushFilters .main plan2556 = plan2556 := rfl

end Falkor.Opt.PushPlan
