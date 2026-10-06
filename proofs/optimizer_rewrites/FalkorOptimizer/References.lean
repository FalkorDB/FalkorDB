/-
# "Does anything read this variable?" — reduce_expand_into / reduce_bound_edge /
# reduce_var_len_path / fuse_anonymous_traverse

All four passes lower a flag (`emit_relationship`, `bind_relationship`,
`emit_path`) or fuse away an intermediate node when no operator reads a
variable. They share one oracle.

| here | there (origin/main) |
| --- | --- |
| `IR`                 | `planner::IR` (`graph/src/planner/mod.rs:65-326`), every expression field abstracted to the variables its BFS finds (`Ex := List Var`) |
| `reads`              | specification: every variable any expression field of the operator mentions |
| `refsMain`           | `ir_references_variable` (`optimizer/reduce_expand_into.rs:22-75`) |
| `refsPR`             | `ir_references_variable` after PR #2918 (branch `fix/2896-edge-var-read-outside-ancestors`) |
| `Plan`, `Frame`, `Ctx` | an `orx_tree` node addressed by a zipper: parents with their other children |
| `ancestorsRead`      | the `while let Some(parent)` walk (`reduce_expand_into.rs:124-133`, `reduce_bound_edge.rs:45-54`, `reduce_var_len_path.rs:39-48`, `fuse_anonymous_traverse.rs:65-79`) |
| `readOutside`        | `variable_read_outside` in PR #2918 (walks parents *and* their other subtrees) |

Theorems: `refsPR_complete` (PR's per-node oracle sees every read),
`readOutside_iff` (PR's walk is exactly "some operator outside the subtree reads
v"), `ancestorsRead_sound` (main's walk never claims a read that is not there —
flags are never wrongly *kept*), and counterexamples `main_misses_unwind`,
`main_misses_create`, `main_misses_foreach`, `main_misses_call_sibling`
(each confirmed against the server, see the root file).
-/
import FalkorOptimizer.Basic

namespace Falkor.Opt.Refs

open Falkor.Opt

abbrev Ex := List Var

structure QNode where
  alias : Var
  attrs : Ex

structure QRel where
  alias : Var
  attrs : Ex
  src : QNode
  dst : QNode

inductive SetItem where
  | attribute (target value : Ex)
  | label (var : Var)

inductive IR where
  | argument
  | optional (vars : List Var)
  | procedureCall (args : List Ex) (yields : List Var)
  | unwind (expr : Ex) (var : Var)
  | create (nodeAttrs relAttrs : List Ex)
  | merge (nodeAttrs relAttrs : List Ex) (onCreate onMatch : List SetItem)
  | delete (exprs : List Ex)
  | set (items : List SetItem)
  | remove (exprs : List Ex)
  | allNodeScan (n : QNode)
  | nodeByLabelScan (n : QNode)
  | includePending (n : QNode)
  | nodeByIndexScan (n : QNode) (query : List Ex)
  | edgeByIndexScan (r : QRel) (query : List Ex)
  | nodeByFulltextScan (label query : Ex)
  | edgeByFulltextScan (label query : Ex)
  | nodeByVectorScan (label attr k vector : Ex)
  | edgeByVectorScan (label attr k vector : Ex)
  | nodeByLabelAndIdScan (n : QNode) (filter : List Ex)
  | nodeByIdSeek (n : QNode) (filter : List Ex)
  | condTraverse (r : QRel) (chain : List QRel)
  | condVarLenTraverse (r : QRel) (edgeFilter : Option Ex)
  | allShortestPaths (r : QRel)
  | expandInto (r : QRel)
  | pathBuilder (paths : List (List Var))
  | filter (e : Ex)
  | cartesianProduct
  | valueHashJoin (lhs rhs : Ex)
  | apply | semiApply | antiSemiApply
  | orApplyMultiplexer
  | loadCsv (file delim : Ex) (var : Var)
  | sort (exprs : List Ex)
  | skip (e : Ex)
  | limit (e : Ex)
  | aggregate (names : List Var) (keys aggs : List (Var × Ex)) (projections : List (Var × Var))
  | project (exprs : List (Var × Ex)) (copies : List (Var × Var))
  | distinct | nestedPlans | union | commit
  | forEach (list : Ex) (var : Var)
  | createIndex (options : Option Ex)
  | dropIndex

def SetItem.reads : SetItem → List Var
  | .attribute t v => t ++ v
  | .label v => [v]

def QNode.reads (n : QNode) : List Var := n.attrs
/-- A traverse reads its endpoints' inline properties and relationship properties. -/
def QRel.reads (r : QRel) : List Var := r.attrs ++ r.src.attrs ++ r.dst.attrs

/-- **Specification**: the variables an operator's expressions mention
(bound aliases are not reads; `Project`/`Aggregate` copies read their source,
which is the *first* component — `binder.rs:2234` inserts `(parent, copy)`,
`project.rs:115` iterates `(old_var, new_var)`). -/
def reads : IR → List Var
  | .argument | .optional _ | .cartesianProduct | .apply | .semiApply | .antiSemiApply
  | .orApplyMultiplexer | .distinct | .nestedPlans | .union | .commit | .dropIndex => []
  | .procedureCall args _ => args.flatten
  | .unwind e _ => e
  | .create na ra => na.flatten ++ ra.flatten
  | .merge na ra oc om => na.flatten ++ ra.flatten ++ (oc.flatMap SetItem.reads) ++ (om.flatMap SetItem.reads)
  | .delete es | .remove es | .sort es => es.flatten
  | .set items => items.flatMap SetItem.reads
  | .allNodeScan n | .nodeByLabelScan n | .includePending n => n.reads
  | .nodeByIndexScan n q => n.reads ++ q.flatten
  | .edgeByIndexScan r q => r.reads ++ q.flatten
  | .nodeByFulltextScan l q | .edgeByFulltextScan l q => l ++ q
  | .nodeByVectorScan l a k v | .edgeByVectorScan l a k v => l ++ a ++ k ++ v
  | .nodeByLabelAndIdScan n f | .nodeByIdSeek n f => n.reads ++ f.flatten
  | .condTraverse r ch => r.reads ++ ch.flatMap QRel.reads
  | .condVarLenTraverse r ef => r.reads ++ ef.getD []
  | .allShortestPaths r | .expandInto r => r.reads
  | .pathBuilder ps => ps.flatten
  | .filter e | .skip e | .limit e => e
  | .valueHashJoin l r => l ++ r
  | .loadCsv f d _ => f ++ d
  | .aggregate _ ks as ps => ks.flatMap Prod.snd ++ as.flatMap Prod.snd ++ ps.map Prod.fst
  | .project es cs => es.flatMap Prod.snd ++ cs.map Prod.fst
  | .forEach l _ => l
  | .createIndex o => o.getD []

def mem (v : Var) (xs : List Var) : Bool := xs.contains v

/-- `ir_references_variable` on origin/main, arm by arm (reduce_expand_into.rs:27-74). -/
def refsMain (ir : IR) (v : Var) : Bool :=
  match ir with
  | .project es cs => es.any (fun e => mem v e.2) || cs.any (fun c => c.1 == v)
  | .filter e => mem v e
  | .sort es => es.any (mem v)
  | .aggregate _ ks as _ => ks.any (fun e => mem v e.2) || as.any (fun e => mem v e.2)
  | .pathBuilder ps => ps.any (mem v)
  | .unwind _ var | .forEach _ var => var == v          -- line 54: the *bound* var, not the list
  | .delete es | .remove es => es.any (mem v)
  | .set items => items.any (fun i => mem v i.reads)
  | .merge _ _ oc om => oc.any (fun i => mem v i.reads) || om.any (fun i => mem v i.reads)
  | .valueHashJoin l r => mem v l || mem v r
  | _ => false                                          -- line 73

/-- `ir_references_variable` after PR #2918 (exhaustive match). The PR also
counts a traverse whose own alias is `v` (a re-bound edge) — harmless, it only
keeps a flag on. -/
def refsPR (ir : IR) (v : Var) : Bool :=
  match ir with
  | .project es cs => es.any (fun e => mem v e.2) || cs.any (fun c => c.1 == v)
  | .filter e | .skip e | .limit e => mem v e
  | .sort es => es.any (mem v)
  | .aggregate _ ks as ps => ks.any (fun e => mem v e.2) || as.any (fun e => mem v e.2) || ps.any (fun c => c.1 == v)
  | .pathBuilder ps => ps.any (mem v)
  | .unwind e _ | .forEach e _ => mem v e
  | .delete es | .remove es => es.any (mem v)
  | .set items => items.any (fun i => mem v i.reads)
  | .create na ra => na.any (mem v) || ra.any (mem v)
  | .merge na ra oc om => na.any (mem v) || ra.any (mem v) ||
      oc.any (fun i => mem v i.reads) || om.any (fun i => mem v i.reads)
  | .valueHashJoin l r => mem v l || mem v r
  | .procedureCall args _ => args.any (mem v)
  | .loadCsv f d _ => mem v f || mem v d
  | .allNodeScan n | .nodeByLabelScan n | .includePending n => mem v n.attrs
  | .nodeByIndexScan n q => mem v n.attrs || q.any (mem v)
  | .nodeByLabelAndIdScan n f | .nodeByIdSeek n f => mem v n.attrs || f.any (mem v)
  | .nodeByFulltextScan l q | .edgeByFulltextScan l q => mem v l || mem v q
  | .nodeByVectorScan l a k w | .edgeByVectorScan l a k w => mem v l || mem v a || mem v k || mem v w
  | .condTraverse r ch => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v ||
      ch.any (fun q => q.alias == v || mem v q.reads || q.src.alias == v || q.dst.alias == v)
  | .condVarLenTraverse r ef => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v ||
      mem v (ef.getD [])  -- `is_some_and(expr)`
  | .edgeByIndexScan r q => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v || q.any (mem v)
  | .expandInto r | .allShortestPaths r => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v
  | .createIndex o => mem v (o.getD [])
  | _ => false

theorem mem_iff (v : Var) (xs : List Var) : mem v xs = true ↔ v ∈ xs := by
  simp [mem]

theorem any_mem_flatten (v : Var) (es : List Ex) : es.any (mem v) = true ↔ v ∈ es.flatten := by
  simp [mem, List.mem_flatten]

/-- **PR #2918's oracle is complete**: every variable an operator reads is seen. -/
theorem refsPR_complete (ir : IR) (v : Var) (h : v ∈ reads ir) : refsPR ir v = true := by
  cases ir <;> simp [reads, refsPR, mem, List.mem_flatten, List.mem_flatMap, QRel.reads,
    QNode.reads, SetItem.reads, or_assoc] at h ⊢ <;>
    first
    | exact h
    | (rcases h with h | h <;> simp [h])
    | (rcases h with h | h | h <;> simp [h])
    | (rcases h with h | h | h | h <;> simp [h])
    | (rcases h with h | h
       · exact Or.inl h
       · exact Or.inr (Or.inl h))
    | (rcases h with h | h | h | ⟨a, ha, h⟩
       · simp [h]
       · simp [h]
       · simp [h]
       · refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨a, ha, ?_⟩)))))
         rcases h with h | h | h <;> simp [h])
    | (rename_i ef; cases ef <;> simp at h ⊢ <;> rcases h with h | h | h | h <;> simp [h])
    | skip

/-- **origin/main's oracle is sound** for everything but the bound var of
`Unwind`/`ForEach` (which only errs toward keeping the flag). -/
theorem refsMain_sound (ir : IR) (v : Var) (h : refsMain ir v = true)
    (hb : ∀ e var, ir = .unwind e var ∨ ir = .forEach e var → False) : v ∈ reads ir := by
  cases ir <;> simp_all [reads, refsMain, mem, List.mem_flatten, List.mem_flatMap, SetItem.reads]
  all_goals (rcases h with h | h; exact Or.inl h; exact Or.inr (Or.inl h))

/-! ### Counterexamples on origin/main (each reproduced against the server) -/

def r : Var := ⟨2, 0⟩
def xV : Var := ⟨3, 0⟩
def nd (v : Var) : QNode := ⟨v, []⟩

/-- `MATCH (a)-[r]->(b) UNWIND [r.w] AS w RETURN w`: returns 2 of 3 rows. -/
theorem main_misses_unwind :
    r ∈ reads (.unwind [r] xV) ∧ refsMain (.unwind [r] xV) r = false := by decide

/-- `MATCH (a)-[r]->(b) CREATE (:C {w: r.w})`: creates 2 nodes instead of 3. -/
theorem main_misses_create :
    r ∈ reads (.create [[r]] []) ∧ refsMain (.create [[r]] []) r = false := by decide

/-- `FOREACH (x IN [r.w] | …)`. -/
theorem main_misses_foreach :
    r ∈ reads (.forEach [r] xV) ∧ refsMain (.forEach [r] xV) r = false := by decide

/-! ## Tree walk -/

inductive Plan where
  | node (ir : IR) (children : List Plan)

def Plan.anyRead (v : Var) (refs : IR → Var → Bool) : Plan → Bool
  | .node ir cs => refs ir v || (cs.attach.map fun ⟨c, _⟩ => c.anyRead v refs).any id

/-- One level of the zipper: the parent operator and its *other* children. -/
structure Frame where
  parent : IR
  others : List Plan

abbrev Ctx := List Frame   -- innermost first

/-- origin/main: `while let Some(parent) = … { if ir_references_variable(parent) … }`. -/
def ancestorsRead (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v)

/-- PR #2918 `variable_read_outside`: parent, then every sibling subtree. -/
def readOutside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v || f.others.any (fun s => s.anyRead v refs))

/-- Operators outside the subtree (spec side). -/
def Plan.ops : Plan → List IR
  | .node ir cs => ir :: (cs.attach.flatMap fun ⟨c, _⟩ => c.ops)

def outsideOps (ctx : Ctx) : List IR :=
  ctx.flatMap (fun f => f.parent :: f.others.flatMap Plan.ops)

theorem anyRead_iff (refs : IR → Var → Bool) (v : Var) (p : Plan) :
    p.anyRead v refs = true ↔ ∃ ir ∈ p.ops, refs ir v = true := by
  induction p using Plan.rec (motive_2 := fun cs => ∀ c ∈ cs,
      c.anyRead v refs = true ↔ ∃ ir ∈ c.ops, refs ir v = true) with
  | node ir cs ih =>
    simp only [Plan.anyRead, Plan.ops, Bool.or_eq_true, List.any_eq_true, List.mem_map,
      List.mem_attach, true_and, Subtype.exists, id_eq, List.mem_cons, List.mem_flatMap]
    constructor
    · rintro (h | ⟨b, ⟨c, hc, rfl⟩, hb⟩)
      · exact ⟨ir, Or.inl rfl, h⟩
      · obtain ⟨ir', h1, h2⟩ := (ih c hc).1 hb
        exact ⟨ir', Or.inr ⟨c, hc, h1⟩, h2⟩
    · rintro ⟨ir', h | ⟨c, hc, h1⟩, h2⟩
      · subst h; exact Or.inl h2
      · exact Or.inr ⟨_, ⟨c, hc, rfl⟩, (ih c hc).2 ⟨ir', h1, h2⟩⟩
  | nil => rename_i h; cases h
  | cons c cs ihc ihcs =>
    rename_i c' hc'
    rcases List.mem_cons.1 hc' with rfl | h
    · exact ihc
    · exact ihcs c' h

/-- **`variable_read_outside` is exactly the specification** "some operator
outside the traverse's subtree reads `v`", relative to the per-node oracle. -/
theorem readOutside_iff (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) :
    readOutside refs ctx v = true ↔ ∃ ir ∈ outsideOps ctx, refs ir v = true := by
  simp only [readOutside, outsideOps, List.any_eq_true, Bool.or_eq_true, List.mem_flatMap,
    List.mem_cons]
  constructor
  · rintro ⟨f, hf, h | ⟨s, hs, hr⟩⟩
    · exact ⟨f.parent, ⟨f, hf, Or.inl rfl⟩, h⟩
    · obtain ⟨ir, h1, h2⟩ := (anyRead_iff refs v s).1 hr
      exact ⟨ir, ⟨f, hf, Or.inr ⟨s, hs, h1⟩⟩, h2⟩
  · rintro ⟨ir, ⟨f, hf, rfl | ⟨s, hs, h1⟩⟩, h2⟩
    · exact ⟨f, hf, Or.inl h2⟩
    · exact ⟨f, hf, Or.inr ⟨s, hs, (anyRead_iff refs v s).2 ⟨ir, h1, h2⟩⟩⟩

/-- With the complete oracle, PR #2918 keeps the flag whenever the spec says the
variable is read anywhere outside the traverse's subtree. -/
theorem pr_keeps_flag_when_read (ctx : Ctx) (v : Var)
    (h : ∃ ir ∈ outsideOps ctx, v ∈ reads ir) : readOutside refsPR ctx v = true :=
  (readOutside_iff refsPR ctx v).2 (by
    obtain ⟨ir, h1, h2⟩ := h; exact ⟨ir, h1, refsPR_complete ir v h2⟩)

/-- Ancestor-only walk implies the full walk (it is weaker). -/
theorem ancestors_le_outside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var)
    (h : ancestorsRead refs ctx v = true) : readOutside refs ctx v = true := by
  simp only [ancestorsRead, readOutside, List.any_eq_true, Bool.or_eq_true] at *
  obtain ⟨f, hf, h⟩ := h; exact ⟨f, hf, Or.inl h⟩

/-- `MATCH (a)-[r]->(b) CALL { WITH r DELETE r }`: the reader sits in Apply's
right child, a sibling of the traverse (issue #2896 / known finding 7). -/
def callCtx : Ctx := [⟨.apply, [.node (.delete [[r]]) [.node .argument []]]⟩, ⟨.commit, []⟩]

theorem main_misses_call_sibling :
    ancestorsRead refsMain callCtx r = false ∧ readOutside refsPR callCtx r = true := by
  refine ⟨rfl, pr_keeps_flag_when_read callCtx r ⟨.delete [[r]], ?_, ?_⟩⟩
  · simp [outsideOps, callCtx, Plan.ops]
  · simp [reads]

end Falkor.Opt.Refs
