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

| here | there (Rust at fb3d516d0; current cites at #3933) |
| --- | --- |
| `G` | the graph a query sees and writes: node count, labels, edges |
| `Op` | `IR` (`graph/src/planner/mod.rs:200-300`), the operators these repros use |
| `run` | the runtime ops, eager: `apply.rs:102-340` (right branch re-run per left row with `Argument` = that row), `optional.rs:91-292` (one null row when the branch is empty), `aggregate.rs:288` (no keys → one row even on empty input), `create.rs:101-160`, `merge.rs:67-300` (match else create), `filter.rs:48-80` (`Basic.filterRows`), `unwind.rs`, `node_by_label_scan.rs`, `cond_traverse.rs` (bound destination) |
| `gvC`, `gv` | `GetVariables::get_variables` (`runtime/runtime.rs:286-392`): `Project` adds its outputs (names and copy outputs) and does not descend; `gvC true` also stops at `Aggregate` (#3521, current); `gv = gvC false` is the historical walk through `Aggregate` |
| `branchOut` | `branch_output_variables` (`optimizer/push_filters_down.rs:60-80`); with `copyOut` a copy contributes its output (#3933, fix of #3869) |
| `projectCp`, `distinct` | `IR::Project { exprs, copies }` and `IR::Distinct` (the #3869 shape) |
| `excluded` | the child exclusion list (`push_filters_down.rs:215-229`; before #3582 without the write kinds) |
| `isWriteOp`, `locked` | `is_write` (:110-124) and `holds_write`/`locked` (:247-254, used at :336), #3582 (`Cfg.pr3582`, `Cfg.pr3933`) |
| `pushAt` | one Filter's routing (`push_filters_down.rs:146-410`): grandchild variable sets, Case 1 (:262-270), Case 2 inherited (`inh`, :271-312), augmentation of `Argument` subtrees (:314-324), the `all` test (:336; by `(id, scope)` since #3933) |
| `childInh` | Case 2's ancestor walk, top-down: below an `Apply`'s right child the left subtree's visible variables (`bv`, `branch_visible_variables` :87-102, #2981), below its left child nothing, below `Merge` nothing |
| `sweep`, `pushFilters` | the restart loop (:146-410), as a top-down sweep that follows each pushed filter |
| `Cfg.main` | before #3582: ids only (`collect_expr_variables`/`collect_subtree_variables` then returned ids), no write barrier (historical) |
| `Cfg.pr3582` | Rust at 7c04fa598 (#3582): ids only, write kinds excluded, siblings of a writing branch locked; the CP split's `!holds_write` guard (:352) is not modelled (no CP split here) (historical) |
| `Cfg.pr3933` | the current Rust (#3933): `(id, scope)` keys (`optimizer/mod.rs:92`, `:167`), #3582's exclusion and locking, copy outputs in `branchOut`, `get_variables` stopping at `Aggregate` |

The property each bug violates is `EffectSound`: a rewrite keeps the rows
**and** the graph the query leaves behind. `RowSound` is the old criterion,
rows only; `bug748_rows_only_blind` / `bug997_rows_only_blind` show it cannot see
the write that is skipped. `RoutesBound` is the syntactic property behind
#3352: a pushed conjunct reads only variables, by `(id, scope)`, that are
available where it lands.

Write scope: the only write the model performs is creating a bare node (`newNode`,
from `CREATE ()`/`MERGE ()`), which bumps the node count `G.n`. `G.lbl` and `G.edges` are
read by the scans and traversals but are never written, so label and relationship writes
(and attribute writes) are NOT modelled: `EffectSound` compares the node count only.

Not modelled: conjunct splitting (the repros have one conjunct), filter
merging and the CP split (`Basic`, `Sequence`), laziness (operators are
eager), property access (`r.status = 'revoked'` is `r = 1`, the revoked
rule's node id; `r.decided_at` is `r`). `get_variables` skips a `Project`'s
subtree with `continue` (runtime.rs:322), as `gvC` does.
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
  /-- `IR::Project { exprs, copies }` with copies `(source, output)` (binder.rs:2447-2448):
  each output column carries its source's value. -/
  | projectCp (bs : List (Var × Expr)) (cs : List (Var × Var)) (c : Op)
  /-- `IR::Distinct` deduplicating on the columns `vs` (C: the projected columns of the WITH). -/
  | distinct (vs : List Var) (c : Op)
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

/-- `DISTINCT` on the columns `vs`: the first row of each key is kept. -/
def distinctOn (vs : List Var) (rs : List Row) : List Row :=
  (rs.foldl (fun (acc : List Row × List (List Val)) r =>
    let k := vs.map r.get
    if acc.2.contains k then acc else (acc.1 ++ [r], k :: acc.2)) ([], [])).1

def newNode (v : Var) (r : Row) (g : G) : Res :=
  .ok ([r ++ [(v, .int g.n)]], { g with n := g.n + 1 })

def run (arg : Row) : Op → G → Res
  | .argument, g => .ok ([arg], g)
  | .unit, g => .ok ([[]], g)
  | .project bs c, g => do
      let (rs, g) ← run arg c g
      let out ← rs.mapM (evalBs bs)
      pure (out, g)
  | .projectCp bs cs c, g => do
      let (rs, g) ← run arg c g
      let out ← rs.mapM (fun r => do
        let b ← evalBs bs r
        pure (b ++ cs.map (fun so => (so.2, r.get so.1))))
      pure (out, g)
  | .distinct vs c, g => do
      let (rs, g) ← run arg c g
      pure (distinctOn vs rs, g)
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
  /-- #3582: write operators (`Create`, `Commit`, …) join the exclusion list,
  and a grandchild is locked when a sibling subtree writes (`is_write`,
  `holds_write`/`locked`, push_filters_down.rs:110-124, :247-254, :336). -/
  lockSiblings : Bool
  /-- #3933 (fix of #3869): `branch_output_variables` lists each copy's *output*
  (push_filters_down.rs:60-80); before, its source. -/
  copyOut : Bool
  /-- #3521: `get_variables` stops at an `Aggregate` (runtime.rs:378-385) as it does at a
  `Project`; before, it walked on into the pre-WITH plan. -/
  aggStops : Bool

/-- Historical: Rust before #3582: ids only, no barrier. -/
def Cfg.main : Cfg := ⟨fun a b => a.id == b.id, false, false, false, false⟩

/-- A candidate fix: `(id, scope)` matching and no push below an operator whose
subtree writes (`Merge`/`Create`). -/
def Cfg.fixed : Cfg := ⟨fun a b => a == b, true, false, false, false⟩

/-- Historical: Rust with #3582 (7c04fa598): ids only, write operators excluded, siblings
of a writing branch locked. -/
def Cfg.pr3582 : Cfg := ⟨fun a b => a.id == b.id, false, true, false, false⟩

/-- **The current Rust** (#3933, 8d80077a4): `VarKey` (`(id, scope)`) equality in every
variable set, #3582's write exclusion and sibling lock, copy outputs in
`branch_output_variables` (#3869), and `get_variables` stopping at `Aggregate` (#3521). -/
def Cfg.pr3933 : Cfg := ⟨fun a b => a == b, false, true, true, true⟩

/-- `get_variables` (runtime.rs:286-392): `Project` adds its outputs (expression names and
copy outputs) and stops; `Aggregate` adds its names and, since #3521 (`stop`), stops too. -/
def gvC (stop : Bool) : Op → List Var
  | .argument | .unit => []
  | .project bs _ => bs.map Prod.fst
  | .projectCp bs cs _ => bs.map Prod.fst ++ cs.map Prod.snd
  | .filter _ c | .commit c | .distinct _ c => gvC stop c
  | .unwind v _ c | .allNodeScan v c | .labelScan v _ c | .create v c | .merge v c => v :: gvC stop c
  | .condTraverse e fr to _ c => [e, fr, to] ++ gvC stop c
  | .hop e fr to _ c => [e, fr.v, to.v] ++ gvC stop c
  | .optional vs c => vs ++ gvC stop c
  | .aggregate keys cnt c => aggNames keys cnt ++ (if stop then [] else gvC stop c)
  | .apply l r => gvC stop l ++ gvC stop r

/-- Historical `get_variables` (before #3521): it walks through `Aggregate`. -/
abbrev gv : Op → List Var := gvC false

/-- `branch_output_variables` (push_filters_down.rs:60-80). A copy contributes its output
since #3933 (`cfg.copyOut`), its source before. -/
def branchOut (cfg : Cfg) : Op → Option (List Var)
  | .project bs _ => some (bs.map Prod.fst)
  | .projectCp bs cs _ => some (bs.map Prod.fst ++ cs.map (if cfg.copyOut then Prod.snd else Prod.fst))
  | .aggregate keys cnt _ => some (aggNames keys cnt)
  | _ => none

/-- The elements of `xs` with no `eqv`-match in `ys` (`HashSet::difference`): by key
(`(id, scope)`) since #3933, by id before (`Cfg.eqv`). -/
def varsMinus (eqv : Var → Var → Bool) (xs ys : List Var) : List Var := xs.filter fun v => !ys.any (eqv · v)

/-- `branch_visible_variables` (push_filters_down.rs:87-102, #2981): an env-resetting
operator's outputs (`branchOut`); otherwise the variables the operator itself adds (its
`get_variables` minus its children's) and, recursively, what its children's rows carry. -/
def bv (cfg : Cfg) : Op → List Var
  | .argument | .unit => []
  | .project bs _ => bs.map Prod.fst
  | .projectCp bs cs _ => bs.map Prod.fst ++ cs.map (if cfg.copyOut then Prod.snd else Prod.fst)
  | .aggregate keys cnt _ => aggNames keys cnt
  | .filter _ c | .commit c | .distinct _ c => bv cfg c
  | .unwind v _ c | .allNodeScan v c | .labelScan v _ c | .create v c | .merge v c =>
    varsMinus cfg.eqv [v] (gvC cfg.aggStops c) ++ bv cfg c
  | .condTraverse e fr to _ c => varsMinus cfg.eqv [e, fr, to] (gvC cfg.aggStops c) ++ bv cfg c
  | .hop e fr to _ c => varsMinus cfg.eqv [e, fr.v, to.v] (gvC cfg.aggStops c) ++ bv cfg c
  | .optional vs c => varsMinus cfg.eqv vs (gvC cfg.aggStops c) ++ bv cfg c
  | .apply l r => bv cfg l ++ bv cfg r

/-- Every variable `bv` reports is one `get_variables` reports: Case 2 inherits no more than
`collect_subtree_variables` would. Needs copies to contribute their outputs (`copyOut`, #3933):
before, a copy's source was reported, which `get_variables` does not list
(`pre3933_bv_copy_not_gv`). -/
theorem bv_sub_gv (cfg : Cfg) (hc : cfg.copyOut = true) :
    ∀ (op : Op), ∀ v ∈ bv cfg op, v ∈ gvC cfg.aggStops op := by
  intro op
  induction op with
  | argument | unit => simp [bv]
  | project bs c _ => simp [bv, gvC]
  | projectCp bs cs c _ => simp [bv, gvC, hc]
  | aggregate k n c _ => intro x hx; simp only [bv] at hx; simp [gvC, hx]
  | filter p c ih => simpa [bv, gvC] using ih
  | commit c ih => simpa [bv, gvC] using ih
  | distinct vs c ih => simpa [bv, gvC] using ih
  | unwind v xs c ih | allNodeScan v c ih | labelScan v l c ih | create v c ih | merge v c ih =>
    intro x hx
    simp only [bv, List.mem_append, varsMinus, List.mem_filter] at hx
    rcases hx with ⟨h, _⟩ | h
    · simp at h; simp [gvC, h]
    · simp [gvC, ih x h]
  | condTraverse e fr to l c ih =>
    intro x hx
    simp only [bv, List.mem_append, varsMinus, List.mem_filter] at hx
    rcases hx with ⟨h, _⟩ | h
    · simp only [gvC, List.mem_append]; exact Or.inl h
    · simp [gvC, ih x h]
  | hop e fr to tr c ih =>
    intro x hx
    simp only [bv, List.mem_append, varsMinus, List.mem_filter] at hx
    rcases hx with ⟨h, _⟩ | h
    · simp only [gvC, List.mem_append]; exact Or.inl h
    · simp [gvC, ih x h]
  | optional vs c ih =>
    intro x hx
    simp only [bv, List.mem_append, varsMinus, List.mem_filter] at hx
    rcases hx with ⟨h, _⟩ | h
    · simp only [gvC, List.mem_append]; exact Or.inl h
    · simp [gvC, ih x h]
  | apply l r ihl ihr =>
    intro x hx
    simp only [bv, List.mem_append] at hx
    rcases hx with h | h
    · simp [gvC, ihl x h]
    · simp [gvC, ihr x h]

/-- Historical (before #3933): a copy `(a, a')` made `bv` report the source `a`, a variable
of the scope before the WITH that the projection's rows do not carry. -/
theorem pre3933_bv_copy_not_gv :
    bv .pr3582 (.projectCp [] [(⟨0, 0⟩, ⟨1, 1⟩)] .unit) = [⟨0, 0⟩] ∧
    ⟨0, 0⟩ ∉ gvC Cfg.pr3582.aggStops (.projectCp [] [(⟨0, 0⟩, ⟨1, 1⟩)] .unit) := by decide

/-- On the current code, an env-resetting operator's `branch_output_variables` is exactly its
`get_variables` (`collect_subtree_variables`): copies and aggregate names alike. -/
theorem pr3933_branchOut_eq_gv (k : Op) (o : List Var) (h : branchOut .pr3933 k = some o) :
    o = gvC Cfg.pr3933.aggStops k := by
  cases k <;> simp [branchOut, Cfg.pr3933] at h <;> subst h <;> simp [gvC, Cfg.pr3933]

/-- `subtree_contains(…, Argument)` (planner/mod.rs:333). -/
def hasArg : Op → Bool
  | .argument => true
  | .unit => false
  | .project _ c | .projectCp _ _ c | .distinct _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c
  | .labelScan _ _ c | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c
  | .commit c | .create _ c | .merge _ c => hasArg c
  | .apply l r => hasArg l || hasArg r

def hasWrite : Op → Bool
  | .argument | .unit => false
  | .create _ _ | .merge _ _ => true
  | .project _ c | .projectCp _ _ c | .distinct _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c
  | .labelScan _ _ c | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c
  | .commit c => hasWrite c
  | .apply l r => hasWrite l || hasWrite r

/-- push_filters_down.rs:215-229 before #3582 (the kinds that occur here). -/
def excluded : Op → Bool
  | .project _ _ | .projectCp _ _ _ | .aggregate _ _ _ | .merge _ _ | .argument | .optional _ _ => true
  | _ => false

/-- `is_write` (push_filters_down.rs:110-124), the kinds that occur here; #3582
adds `Create`/`Commit` (and `Delete`/`Set`/`Remove`/`ForEach`, not modelled)
to the exclusion list (:188-193). -/
def isWriteOp : Op → Bool
  | .create _ _ | .merge _ _ | .commit _ => true
  | _ => false

def kid (c : Op) : List Op := match c with | .unit => [] | c => [c]

/-- `children()` of an operator. -/
def kids : Op → List Op
  | .argument | .unit => []
  | .project _ c | .projectCp _ _ c | .distinct _ c | .filter _ c | .unwind _ _ c | .allNodeScan _ c
  | .labelScan _ _ c | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c
  | .commit c | .create _ c | .merge _ c => kid c
  | .apply l r => [l, r]

/-- Replace the children (same shape). -/
def setKids (op : Op) (ks : List Op) : Op :=
  match op, ks with
  | .project bs _, [c] => .project bs c
  | .projectCp bs cs _, [c] => .projectCp bs cs c
  | .distinct vs _, [c] => .distinct vs c
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
def info (cfg : Cfg) (k : Op) : List Var × Bool :=
  match branchOut cfg k with
  | some o => (o, true)
  | none => (gvC cfg.aggStops k, false)

/-- Does the conjunct route into a child with variable set `vs` (:264-276)? -/
def routes (cfg : Cfg) (p : Expr) (vs : List Var) : Bool :=
  p.vars.all (fun v => vs.any (cfg.eqv v))

/-- The variable set the routing test sees for child `k` (:247-257). -/
def avail (cfg : Cfg) (inh : List Var) (k : Op) : List Var :=
  let (vs, reset) := info cfg k
  if !inh.isEmpty && !reset && hasArg k then vs ++ inh else vs

/-- `locked[i]` (push_filters_down.rs:251-254): some other grandchild's subtree
writes (`subtree_contains(…, is_write)`; in the model a `Commit` always wraps a
`Create`/`Merge`, so `hasWrite` is that test). -/
def locked (cfg : Cfg) (ks : List Op) (i : Nat) : Bool :=
  cfg.lockSiblings && (ks.zipIdx.any (fun kj => hasWrite kj.1 && kj.2 != i))

/-- One Filter `p` over `c`, with Case-2 inherited variables `inh`. -/
def pushAt (cfg : Cfg) (inh : List Var) (p : Expr) (c : Op) : Op :=
  let ks := kids c
  if excluded c || (cfg.lockSiblings && isWriteOp c) || ks.isEmpty || (cfg.writeBarrier && hasWrite c) then
    .filter p c else
  let inh := (if isApply c then ((ks.head?.map (info cfg)).map Prod.fst).getD [] else []) ++ inh
  let rs := ks.zipIdx.map (fun ki => !locked cfg ks ki.2 && routes cfg p (avail cfg inh ki.1))
  if rs.any id then setKids c ((ks.zip rs).map (fun kb => if kb.2 then .filter p kb.1 else kb.1))
  else .filter p c

/-- Case 2, top-down (:237-268); since #2981 the Apply's left side contributes
`branch_visible_variables` (`bv`), before it `collect_subtree_variables` (`gv`). -/
def childInh (cfg : Cfg) (op : Op) (i : Nat) (inh : List Var) : List Var :=
  match op with
  | .apply l _ => if i = 0 then [] else bv cfg l
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
      | t => mapKids (fun i k => sweep cfg fuel (childInh cfg t i inh) k) t
    | op => mapKids (fun i k => sweep cfg fuel (childInh cfg op i inh) k) op

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

/-- **#3352 fixed by #2981.** Before #2981 the live Rust EXPLAIN was `Optional →
Conditional Traverse → Filter → Argument` (Case 2 inherited `gv agg3352`, which walks
through the `Aggregate`). Case 2 now inherits `branch_visible_variables` (`bv`), the
`Aggregate`'s outputs `p, n` only, so the filter on `r` stays above the traverse. -/
theorem fixed3352_plan : pushFilters .main plan3352 = plan3352 := rfl

/-- What Case 2 inherits now: the outputs of the `Aggregate`, no pre-WITH id. -/
theorem fixed3352_inherits_outputs :
    bv .main agg3352 = [p', nV] ∧ routes .main revoked (bv .main agg3352) = false := ⟨rfl, rfl⟩

/-- Why: what the routing test sees at the `Argument` contains id 2 (the
pre-WITH `_anon` edge), while by `(id, scope)` only `p, n` of scope 1 arrive
there (the `Aggregate`'s outputs). `RoutesBound` fails. -/
theorem bug3352_routes_unbound :
    avail .main (gv agg3352) .argument = gv agg3352 ∧ e0 ∈ gv agg3352 ∧
    routes .main revoked (gv agg3352) = true ∧
    ¬ RoutesBound revoked (aggNames [(p', .var p)] (some (nV, .var r0)) ++ gv .argument) := by
  refine ⟨rfl, by decide, rfl, ?_⟩
  intro h
  have := h r (by simp [revoked, Expr.vars])
  simp [aggNames, gvC, r, p', nV] at this

/-- **#3352.** `n = 1, e = 1` (the revoked rule; C: `e = 123`). Before #2981 Rust
returned `n = 1, e = null`; now the rows are kept. -/
theorem fixed3352_rows_kept :
    run [] plan3352 g3352 = .ok ([[(nOut, .int 1), (eOut, .int 1)]], g3352) ∧
    run [] (pushFilters .main plan3352) g3352 = run [] plan3352 g3352 := ⟨rfl, rfl⟩

/-- Routing by `(id, scope)` keeps the filter above the traverse. -/
theorem bug3352_fixed :
    pushFilters .fixed plan3352 = plan3352 ∧
    run [] (pushFilters .fixed plan3352) g3352 = run [] plan3352 g3352 := ⟨rfl, rfl⟩

/-! ## #3582: write operators excluded, siblings of a write locked

The PR keeps main's id-only routing (`Cfg.pr3582`) and adds the write lock.
#748 and #997 are fixed: the filter lands on the write-bearing branch's output,
never on the rows that feed it. #3352 is a routing bug with no write, untouched by the lock;
#2981 fixes it (`pr3582_3352_fixed`). -/

/-- At an `Apply` whose right branch writes, the left branch is never filtered:
the Filter either stays or goes onto the right branch only. -/
theorem pr3582_left_untouched (inh : List Var) (p : Expr) (l r : Op) (hr : hasWrite r = true) :
    pushAt .pr3582 inh p (.apply l r) = .filter p (.apply l r) ∨
    pushAt .pr3582 inh p (.apply l r) = .apply l (.filter p r) := by
  unfold pushAt
  simp only [kids, excluded, isWriteOp, Cfg.pr3582, locked, List.zipIdx_cons, List.zipIdx_nil]
  by_cases h : routes Cfg.pr3582 p (avail Cfg.pr3582 ((if isApply (.apply l r) then
      (([l, r].head?.map (info Cfg.pr3582)).map Prod.fst).getD [] else []) ++ inh) r) = true <;>
    simp_all [Cfg.pr3582, setKids]

/-- The filter over a write operator itself stays put (`Create`, `Commit`, `Merge`). -/
theorem pr3582_write_excluded (inh : List Var) (p : Expr) (c : Op) (hc : isWriteOp c = true) :
    pushAt .pr3582 inh p c = .filter p c := by
  unfold pushAt; simp [hc, Cfg.pr3582]

/-- #748 under #3582: `false` lands above the MERGE body only; the MERGE runs. -/
theorem pr3582_748_plan :
    pushFilters .pr3582 plan748 =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.apply (.project [(n0, .lit (.int 0))] .unit) (.filter fls body748))) := rfl

theorem pr3582_748_fixed :
    run [] (pushFilters .pr3582 plan748) g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .pr3582 plan748) g0 = run [] plan748 g0 := ⟨rfl, rfl⟩

/-- #997 under #3582: the `Argument` stays unfiltered; the CREATE runs. -/
theorem pr3582_997_plan :
    pushFilters .pr3582 plan997 =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.unwind x [.int 0]
          (.apply .argument
            (.filter fls (.aggregate [] none (.commit (.create c0 (.project [] .argument)))))))) := rfl

theorem pr3582_997_fixed :
    run [] (pushFilters .pr3582 plan997) g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .pr3582 plan997) g0 = run [] plan997 g0 := ⟨rfl, rfl⟩

/-- The controls are unchanged by #3582. -/
theorem pr3582_controls :
    run [] (pushFilters .pr3582 plan748star) g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .pr3582 (plan997With caseX)) g0 = .ok ([], ⟨1, [], []⟩) := ⟨rfl, rfl⟩

/-- #3352 under #3582 and #2981 (the Rust of this branch): no write below, so the lock never
fires, and Case 2 inherits `branch_visible_variables`, so the filter on `r` stays above the
traverse and the row is kept. (On main before #2981 this was `pr3582_3352_open`: `e = null`,
not `RowSound`.) -/
theorem pr3582_3352_fixed :
    pushFilters .pr3582 plan3352 = pushFilters .main plan3352 ∧
    pushFilters .pr3582 plan3352 = plan3352 ∧
    run [] (pushFilters .pr3582 plan3352) g3352 = .ok ([[(nOut, .int 1), (eOut, .int 1)]], g3352) :=
  ⟨rfl, rfl, rfl⟩

/-! ## #3933: `VarKey` routing, the current Rust (`Cfg.pr3933`)

Every variable set of the pass is a `HashSet<VarKey>` and every comparison is by
`(id, scope)`; #3582's write exclusion and sibling lock are kept; a copy contributes its
output to `branch_output_variables` (#3869); `get_variables` stops at `Aggregate` (#3521).
Each #3582 correctness fact is re-checked on this configuration. -/

/-- On the current code a conjunct only lands where every variable it reads is in the
child's set, by `(id, scope)` (`RoutesBound`). -/
theorem pr3933_routes_bound (p : Expr) (vs : List Var) (h : routes .pr3933 p vs = true) :
    RoutesBound p vs := by
  intro v hv
  simp only [routes, Cfg.pr3933, List.all_eq_true, List.any_eq_true, beq_iff_eq] at h
  obtain ⟨w, hw, rfl⟩ := h v hv
  exact hw

/-- At an `Apply` whose right branch writes, the left branch is never filtered. -/
theorem pr3933_left_untouched (inh : List Var) (p : Expr) (l r : Op) (hr : hasWrite r = true) :
    pushAt .pr3933 inh p (.apply l r) = .filter p (.apply l r) ∨
    pushAt .pr3933 inh p (.apply l r) = .apply l (.filter p r) := by
  unfold pushAt
  simp only [kids, excluded, isWriteOp, Cfg.pr3933, locked, List.zipIdx_cons, List.zipIdx_nil]
  by_cases h : routes Cfg.pr3933 p (avail Cfg.pr3933 ((if isApply (.apply l r) then
      (([l, r].head?.map (info Cfg.pr3933)).map Prod.fst).getD [] else []) ++ inh) r) = true <;>
    simp_all [Cfg.pr3933, setKids]

/-- The filter over a write operator itself stays put (`Create`, `Commit`, `Merge`). -/
theorem pr3933_write_excluded (inh : List Var) (p : Expr) (c : Op) (hc : isWriteOp c = true) :
    pushAt .pr3933 inh p c = .filter p c := by
  unfold pushAt; simp [hc, Cfg.pr3933]

/-- #748 on the current code: `false` lands above the MERGE body only; the MERGE runs. -/
theorem pr3933_748_plan :
    pushFilters .pr3933 plan748 =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.apply (.project [(n0, .lit (.int 0))] .unit) (.filter fls body748))) := rfl

theorem pr3933_748_fixed :
    run [] (pushFilters .pr3933 plan748) g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .pr3933 plan748) g0 = run [] plan748 g0 := ⟨rfl, rfl⟩

/-- #997 on the current code: the `Argument` stays unfiltered; the CREATE runs. -/
theorem pr3933_997_plan :
    pushFilters .pr3933 plan997 =
      .project [(col, .lit (.int 0))]
        (.allNodeScan a1 (.unwind x [.int 0]
          (.apply .argument
            (.filter fls (.aggregate [] none (.commit (.create c0 (.project [] .argument)))))))) := rfl

theorem pr3933_997_fixed :
    run [] (pushFilters .pr3933 plan997) g0 = .ok ([], ⟨1, [], []⟩) ∧
    run [] (pushFilters .pr3933 plan997) g0 = run [] plan997 g0 := ⟨rfl, rfl⟩

/-- The controls: the CREATE/MERGE still runs. In the `CASE` control `x` (id 0, scope 0) no
longer matches the body's `CREATE ()` variable (id 0, scope 1), and `get_variables` stops at the
body's `Aggregate`, so the conjunct now stays above the `Unwind` (it was pushed below it). -/
theorem pr3933_controls :
    run [] (pushFilters .pr3933 plan748star) g0 = .ok ([], ⟨1, [], []⟩) ∧
    pushFilters .pr3933 (plan997With caseX) =
      .project [(col, .lit (.int 0))] (.allNodeScan a1 (.filter caseX (.unwind x [.int 0] apply997))) ∧
    run [] (pushFilters .pr3933 (plan997With caseX)) g0 = .ok ([], ⟨1, [], []⟩) := ⟨rfl, rfl, rfl⟩

/-- #3352 on the current code: Case 2 inherits the `Aggregate`'s outputs only, the filter on
`r` stays above the traverse, the row is kept. -/
theorem pr3933_3352_fixed :
    bv .pr3933 agg3352 = [p', nV] ∧
    pushFilters .pr3933 plan3352 = plan3352 ∧
    run [] (pushFilters .pr3933 plan3352) g3352 = .ok ([[(nOut, .int 1), (eOut, .int 1)]], g3352) :=
  ⟨rfl, rfl, rfl⟩


/-! ### #3869: `MATCH (a:X) WITH DISTINCT a.x AS x WHERE a.y = 2 RETURN x`

Graph `(:X {x:1, y:1}), (:X {x:1, y:2}), (:X {x:2, y:2})` (nodes 0, 1, 2). The WITH's
`WHERE` reads `a`, which the WITH does not project: the binder copies it into the WITH's
scope (`(a, a')`, binder.rs:2433-2448) and the planner puts the Filter above the `Distinct`
above the `Project` (planner/mod.rs:2797-2866). C runs the WHERE before the DISTINCT:
`[1, 2]`. Ids (scope-local): `a` = (0, 0); `x` = (0, 1), `a'` = (1, 1); the RETURN's `x` = (0, 2).
`a.x` is `a / 2 + 1`, `a.y = 2` is `a <> 0`. -/

def X : Nat := 0
def g3869 : G := ⟨3, [(0, X), (1, X), (2, X)], []⟩

def aS : Var := ⟨0, 0⟩
def xW : Var := ⟨0, 1⟩
def aW : Var := ⟨1, 1⟩
def xR : Var := ⟨0, 2⟩

/-- `a.x` -/
def ax : Expr := .sub (.div (.var aS) (.lit (.int 2))) (.lit (.int (-1)))
/-- `a.y = 2`, on the copy `a'` -/
def ay2 : Expr := .neq (.var aW) (.lit (.int 0))

def with3869 : Op := .projectCp [(xW, ax)] [(aS, aW)] (.labelScan aS X .unit)

/-- The planner's plan. -/
def plan3869 : Op := .project [(xR, .var xW)] (.filter ay2 (.distinct [xW] with3869))

/-- C's plan: the WHERE below the DISTINCT. -/
def spec3869 : Op := .project [(xR, .var xW)] (.distinct [xW] (.filter ay2 with3869))

/-- Why: what a filter directly above the WITH's `Project` may read. Before #3933 the copy's
source `a` (scope 0, id 0) was listed, by id: `a'` (id 1) was not covered and the filter stayed
above the DISTINCT. Now the copy's output `a'` is listed, by key. -/
theorem fixed3869_branchOut :
    branchOut .pr3582 with3869 = some [xW, aS] ∧ routes .pr3582 ay2 [xW, aS] = false ∧
    branchOut .pr3933 with3869 = some [xW, aW] ∧ routes .pr3933 ay2 [xW, aW] = true := by
  decide

/-- **#3869 fixed.** On the current code the WHERE is pushed below the DISTINCT (C's plan)
and the query returns `[1, 2]`, as C does. -/
theorem pr3933_3869_fixed :
    pushFilters .pr3933 plan3869 = spec3869 ∧
    run [] (pushFilters .pr3933 plan3869) g3869 = .ok ([[(xR, .int 1)], [(xR, .int 2)]], g3869) := ⟨rfl, rfl⟩

/-- **#3869 before #3933** (`Cfg.pr3582`, the Rust of the PR's base): the filter stays above
the DISTINCT, which keeps `a = 0` (`y = 1`) for `x = 1`, then drops it: `[2]`. -/
theorem pre3933_3869_bug :
    pushFilters .pr3582 plan3869 = plan3869 ∧
    run [] (pushFilters .pr3582 plan3869) g3869 = .ok ([[(xR, .int 2)]], g3869) := ⟨rfl, rfl⟩

/-- The planner's own plan for a `WITH DISTINCT … WHERE` reading a carried variable runs the
WHERE after the DISTINCT; the push is what gives it C's order. So this push is *not* a
rows-preserving rewrite of that plan (`RowSound` fails), it is the step that makes the plan
match C: the specification here is `spec3869`, not the planner's intermediate plan. -/
theorem pr3933_3869_push_changes_rows :
    run [] plan3869 g3869 = .ok ([[(xR, .int 2)]], g3869) ∧ ¬ RowSound (pushFilters .pr3933) := by
  refine ⟨rfl, fun h => ?_⟩
  have := h plan3869 g3869
  rw [show run [] (pushFilters .pr3933 plan3869) g3869 = .ok ([[(xR, .int 1)], [(xR, .int 2)]], g3869) from rfl,
      show run [] plan3869 g3869 = .ok ([[(xR, .int 2)]], g3869) from rfl] at this
  simp [Same, Except.map] at this

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
  | project _ c ih | projectCp _ _ c ih | distinct _ c ih | filter _ c ih | unwind _ _ c ih
  | allNodeScan _ c ih | labelScan _ _ c ih | condTraverse _ _ _ _ c ih | hop _ _ _ _ c ih
  | optional _ c ih | aggregate _ _ c ih =>
    intro arg g rs g' hw h
    simp only [run, bind, Except.bind] at h
    rcases hc : run arg c g with e | ⟨xs, g1⟩ <;> rw [hc] at h
    · cases h
    · try simp only at h
      first
      | (rcases hm : List.mapM (evalBs _) xs with e | ys <;> rw [hm] at h
         · cases h
         · simp [pure, Except.pure] at h; rw [← h.2]; exact ih arg g xs g1 hw hc)
      | (split at h
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
  | .projectCp bs cs _ => bs.map Prod.fst ++ cs.map Prod.snd
  | .aggregate keys cnt _ => aggNames keys cnt
  | .filter _ c | .commit c | .distinct _ c => rowVars c
  | .unwind v _ c | .allNodeScan v c | .labelScan v _ c | .create v c | .merge v c => v :: rowVars c
  | .condTraverse e fr _ _ c => [e, fr] ++ rowVars c
  | .hop e _ to _ c => [e, to.v] ++ rowVars c
  | .optional vs c => vs ++ rowVars c
  | .apply l r => rowVars l ++ rowVars r

/-- Every `Filter` reads only variables bound below it. -/
def filtersBound : Op → Bool
  | .argument | .unit => true
  | .filter p c => p.vars.all (fun v => (rowVars c).contains v) && filtersBound c
  | .project _ c | .projectCp _ _ c | .distinct _ c | .unwind _ _ c | .allNodeScan _ c
  | .labelScan _ _ c | .condTraverse _ _ _ _ c | .hop _ _ _ _ c | .optional _ c | .aggregate _ _ c
  | .commit c | .create _ c | .merge _ c => filtersBound c
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

/-- #2556's control is unchanged too. -/
theorem pr3582_2556_push_not_cause : pushFilters .pr3582 plan2556 = plan2556 := rfl

/-- …and on the current code (#3933, `Cfg.pr3933`). -/
theorem pr3933_2556_push_not_cause : pushFilters .pr3933 plan2556 = plan2556 := rfl

end Falkor.Opt.PushPlan
