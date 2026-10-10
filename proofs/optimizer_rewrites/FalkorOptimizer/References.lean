/-
# "Does this operator read this variable?" — `optimizer/references.rs`

reduce_expand_into / reduce_bound_edge / reduce_var_len_path / fuse_anonymous_traverse
lower a flag (`emit_relationship`, `bind_relationship`, `emit_path`) or fuse away an
intermediate node when no operator reads a variable. Since #2390 (d2c42e032) they share
one per-operator oracle, `optimizer/references.rs`.

| here | there (origin/main 8743953a8) |
| --- | --- |
| `IR`                 | `planner::IR` (`graph/src/planner/mod.rs:65-326`), every expression field abstracted to the variables its BFS finds (`Ex := List Var`) |
| `IQ`, `IQ.vars`      | `IndexQuery<QueryExpr<Variable>>` (`index/indexer.rs`), its operand expressions |
| `reads`              | specification: every variable any expression field of the operator mentions |
| `binds`              | variables an operator binds that the oracle also counts (harmless: only keeps a flag) |
| `PlanInv`            | the planner's invariant after #2390: scans / fixed-length operators carry no inline map, walks only their edge's (proofs/planner_build `E.planMatch_stripped`) |
| `refsNew`            | `ir_references_variable` (`references.rs:41-198`) |
| `iqLoop`, `iqRefs`   | `index_query_references_variable` (`references.rs:204-233`), its explicit stack |
| `qgRefs`, `setItemsRefs`, `exprRefs`, `subtreeRefs` | `query_graph_references_variable` (:287-300), `set_items_reference_variable` (:302-314), `expr_references_variable` (:317-323), `subtree_references_variable` (:327-334) |
| `refsPre2390`        | `ir_references_variable` at a9377c636 (`reduce_expand_into.rs:22-75`, `_ => false`) |
| `refsPR`             | `ir_references_variable` of PR #2918 (superseded by #2390's oracle; its walk is not merged) |

Theorems: `iqLoop_iff` (the stack loop = "some operand mentions v", at any And/Or depth),
`refsNew_complete` (under `PlanInv`, every read is seen), `refsNew_sound` (only reads or
bound variables are reported), `refsPR_complete`, `pre2390_refs_sound`; fixed counterexamples
`pre2390_misses_unwind/create/foreach` with `refsNew_sees_*`; still open:
`refsNew_misses_sibling_edges` (W2-traverse-1). The tree walks are in References2.lean.
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

/-- An index query: operands are expressions; `And`/`Or` nest. -/
inductive IQ where
  | eq (value : Ex)
  | arrayContains (value : Ex)
  | inList (list : Ex)
  | range (min max : Option Ex)
  | point (point radius : Ex)
  | and (qs : List IQ)
  | or (qs : List IQ)

mutual
def IQ.vars : IQ → List Var
  | .eq e | .arrayContains e | .inList e => e
  | .range mn mx => mn.getD [] ++ mx.getD []
  | .point p r => p ++ r
  | .and qs | .or qs => IQ.varsL qs
def IQ.varsL : List IQ → List Var
  | [] => []
  | q :: qs => q.vars ++ IQ.varsL qs
end

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
  | nodeByIndexScan (n : QNode) (query : IQ)
  | edgeByIndexScan (r : QRel) (query : IQ)
  | nodeByFulltextScan (label query : Ex)
  | edgeByFulltextScan (label query : Ex)
  | nodeByVectorScan (label attr k vector : Ex)
  | edgeByVectorScan (label attr k vector : Ex)
  | nodeByLabelAndIdScan (n : QNode) (filter : List Ex)
  | nodeByIdSeek (n : QNode) (filter : List Ex)
  /-- `sibling_edges`: the ids (only) of the other named edges of the pattern. -/
  | condTraverse (r : QRel) (chain : List QRel) (siblings : List Nat)
  | condVarLenTraverse (r : QRel) (edgeFilter : Option Ex) (pathVar : Option Var)
  | allShortestPaths (r : QRel)
  | expandInto (r : QRel) (siblings : List Nat)
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
the *first* component — `binder.rs:2234` inserts `(parent, copy)`,
`project.rs:115` iterates `(old_var, new_var)`). `sibling_edges` hold ids, not
expressions; see `runtimeReadsIds`. -/
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
  | .nodeByIndexScan n q => n.reads ++ q.vars
  | .edgeByIndexScan r q => r.reads ++ q.vars
  | .nodeByFulltextScan l q | .edgeByFulltextScan l q => l ++ q
  | .nodeByVectorScan l a k v | .edgeByVectorScan l a k v => l ++ a ++ k ++ v
  | .nodeByLabelAndIdScan n f | .nodeByIdSeek n f => n.reads ++ f.flatten
  | .condTraverse r ch _ => r.reads ++ ch.flatMap QRel.reads
  | .condVarLenTraverse r ef _ => r.reads ++ ef.getD []
  | .allShortestPaths r | .expandInto r _ => r.reads
  | .pathBuilder ps => ps.flatten
  | .filter e | .skip e | .limit e => e
  | .valueHashJoin l r => l ++ r
  | .loadCsv f d _ => f ++ d
  | .aggregate _ ks as ps => ks.flatMap Prod.snd ++ as.flatMap Prod.snd ++ ps.map Prod.fst
  | .project es cs => es.flatMap Prod.snd ++ cs.map Prod.fst
  | .forEach l _ => l
  | .createIndex o => o.getD []

/-- Variables the oracle also reports although the operator *binds* them. -/
def binds : IR → List Var
  | .unwind _ x | .forEach _ x => [x]
  | .aggregate _ _ _ ps => ps.map Prod.snd
  | .condVarLenTraverse _ _ pv => pv.toList
  | _ => []

/-- **The planner's invariant after #2390** (proofs/planner_build `E.planMatch_stripped`):
a MATCH pattern's inline maps are Filters, so scans and fixed-length operators carry none and a
walk only its edge's own (it prunes per edge). Index DDL options are constants (a CREATE INDEX
query has no MATCH to bind a variable). -/
def PlanInv : IR → Prop
  | .allNodeScan n | .nodeByLabelScan n | .includePending n
  | .nodeByIndexScan n _ | .nodeByLabelAndIdScan n _ | .nodeByIdSeek n _ => n.attrs = []
  | .edgeByIndexScan r _ | .expandInto r _ => r.attrs = [] ∧ r.src.attrs = [] ∧ r.dst.attrs = []
  | .condTraverse r ch _ => (r.attrs = [] ∧ r.src.attrs = [] ∧ r.dst.attrs = []) ∧
      ∀ q ∈ ch, q.attrs = [] ∧ q.src.attrs = [] ∧ q.dst.attrs = []
  | .condVarLenTraverse r _ _ | .allShortestPaths r => r.src.attrs = [] ∧ r.dst.attrs = []
  | .createIndex o => o.getD [] = []
  | _ => True

def mem (v : Var) (xs : List Var) : Bool := xs.contains v

theorem mem_iff (v : Var) (xs : List Var) : mem v xs = true ↔ v ∈ xs := by
  simp [mem]

theorem any_mem_flatten (v : Var) (es : List Ex) : es.any (mem v) = true ↔ v ∈ es.flatten := by
  simp [mem, List.mem_flatten]

/-! ## `references.rs` -/

/-- `expr_references_variable` (:317-323): a BFS for `Variable(v)` with `v.id == id &&
v.scope_id == scope` — by `(id, scope)`, never `==` (ids only, ast.rs). With an expression
abstracted to the variables it mentions, that is membership (Helpers.refsVar_iff proves
the BFS on the tree model). -/
def exprRefs (e : Ex) (v : Var) : Bool := mem v e
/-- `subtree_references_variable` (:327-334): the same question of one subtree. -/
def subtreeRefs (e : Ex) (v : Var) : Bool := mem v e

theorem exprRefs_iff (e : Ex) (v : Var) : exprRefs e v = true ↔ v ∈ e := mem_iff v e
theorem subtreeRefs_iff (e : Ex) (v : Var) : subtreeRefs e v = true ↔ v ∈ e := mem_iff v e

/-- `set_items_reference_variable` (:302-314). -/
def setItemsRefs (items : List SetItem) (v : Var) : Bool :=
  items.any fun i => match i with
    | .attribute t x => exprRefs t v || exprRefs x v
    | .label w => w.id == v.id && w.scope == v.scope

theorem setItemsRefs_iff (items : List SetItem) (v : Var) :
    setItemsRefs items v = true ↔ v ∈ items.flatMap SetItem.reads := by
  simp only [setItemsRefs, List.any_eq_true, List.mem_flatMap]
  constructor
  · rintro ⟨i, hi, h⟩; refine ⟨i, hi, ?_⟩
    rcases i with ⟨t, x⟩ | w
    · simp_all [SetItem.reads, exprRefs, mem]
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      have : w = v := by cases w; cases v; simp_all
      simp [SetItem.reads, this]
  · rintro ⟨i, hi, h⟩; refine ⟨i, hi, ?_⟩
    rcases i with ⟨t, x⟩ | w
    · simp_all [SetItem.reads, exprRefs, mem]
    · simp_all [SetItem.reads]

/-- `query_graph_references_variable` (:287-300): node attrs, then relationship attrs. -/
def qgRefs (na ra : List Ex) (v : Var) : Bool := na.any (exprRefs · v) || ra.any (exprRefs · v)

/-! ### `index_query_references_variable` (:204-233): an explicit stack -/

/-- The per-query arm of the `match` (:214-227); `And`/`Or` push their children and miss. -/
def iqHit (v : Var) : IQ → Bool
  | .eq e | .arrayContains e | .inList e => exprRefs e v
  | .range mn mx => mn.any (exprRefs · v) || mx.any (exprRefs · v)
  | .point p r => exprRefs p v || exprRefs r v
  | .and _ | .or _ => false

mutual
def IQ.size : IQ → Nat
  | .and qs | .or qs => 1 + IQ.sizeL qs
  | _ => 0
def IQ.sizeL : List IQ → Nat
  | [] => 0
  | q :: qs => 1 + q.size + IQ.sizeL qs
end

theorem IQ.sizeL_append (a b : List IQ) : IQ.sizeL (a ++ b) = IQ.sizeL a + IQ.sizeL b := by
  induction a with
  | nil => simp [IQ.sizeL]
  | cons q a ih => simp [IQ.sizeL, ih]; omega

theorem IQ.sizeL_reverse (a : List IQ) : IQ.sizeL a.reverse = IQ.sizeL a := by
  induction a with
  | nil => rfl
  | cons q a ih => simp [IQ.sizeL_append, IQ.sizeL, ih]; omega

/-- `let mut pending = vec![query]; while let Some(q) = pending.pop() { … }` (:212-231). The
stack is a list with its top first, so `pending.extend(qs)` (push in order, last on top) is
`qs.reverse ++ rest`; a leaf returns on a hit and otherwise continues. -/
def iqLoop (v : Var) : List IQ → Bool
  | [] => false
  | .and qs :: rest => iqLoop v (qs.reverse ++ rest)
  | .or qs :: rest => iqLoop v (qs.reverse ++ rest)
  | q :: rest => iqHit v q || iqLoop v rest
termination_by l => IQ.sizeL l
decreasing_by
  all_goals simp only [IQ.sizeL, IQ.size, IQ.sizeL_append, List.unattach_reverse,
    List.unattach_attach, IQ.sizeL_reverse]
  all_goals omega

theorem iqHit_imp (v : Var) (q : IQ) (h : iqHit v q = true) : v ∈ q.vars := by
  cases q with
  | range mn mx => cases mn <;> cases mx <;> simp_all [iqHit, IQ.vars, exprRefs, mem]
  | _ => simp_all [iqHit, IQ.vars, exprRefs, mem]

theorem mem_varsL (v : Var) (qs : List IQ) : v ∈ IQ.varsL qs ↔ ∃ q ∈ qs, v ∈ q.vars := by
  induction qs with
  | nil => simp [IQ.varsL]
  | cons q qs ih => simp [IQ.varsL, ih]

/-- **The stack loop is exactly "some operand of some pending query mentions `v`"**, at any
nesting depth of `And`/`Or` and in any sibling position. -/
theorem iqLoop_iff (v : Var) : ∀ (n : Nat) (l : List IQ), IQ.sizeL l ≤ n →
    (iqLoop v l = true ↔ ∃ q ∈ l, v ∈ q.vars) := by
  intro n
  induction n with
  | zero =>
    intro l hl
    cases l with
    | nil => simp [iqLoop]
    | cons q rest => simp [IQ.sizeL] at hl
  | succ n ih =>
    intro l hl
    cases l with
    | nil => simp [iqLoop]
    | cons q rest =>
      have hcons : ∀ (P : IQ → Prop), (∃ q' ∈ q :: rest, P q') ↔ P q ∨ ∃ q' ∈ rest, P q' := by
        intro P; simp
      rw [hcons]
      cases q with
      | and qs | or qs =>
        simp only [IQ.sizeL, IQ.size] at hl
        rw [iqLoop, ih _ (by rw [IQ.sizeL_append, IQ.sizeL_reverse]; omega)]
        simp only [List.mem_append, List.mem_reverse, IQ.vars, mem_varsL]
        constructor
        · rintro ⟨q', hq' | hq', hv⟩
          · exact Or.inl ⟨q', hq', hv⟩
          · exact Or.inr ⟨q', hq', hv⟩
        · rintro (⟨q', hq', hv⟩ | ⟨q', hq', hv⟩)
          · exact ⟨q', Or.inl hq', hv⟩
          · exact ⟨q', Or.inr hq', hv⟩
      | eq e | arrayContains e | inList e | point p r =>
        simp only [IQ.sizeL, IQ.size] at hl
        simp only [iqLoop, Bool.or_eq_true, ih rest (by omega), iqHit, IQ.vars, exprRefs, mem_iff,
          List.mem_append]
      | range mn mx =>
        simp only [IQ.sizeL, IQ.size] at hl
        simp only [iqLoop, Bool.or_eq_true, ih rest (by omega)]
        cases mn <;> cases mx <;> simp [iqHit, IQ.vars, exprRefs, mem]

def iqRefs (q : IQ) (v : Var) : Bool := iqLoop v [q]

theorem iqRefs_iff (q : IQ) (v : Var) : iqRefs q v = true ↔ v ∈ q.vars := by
  rw [iqRefs, iqLoop_iff v _ [q] (Nat.le_refl _)]; simp

/-- `ir_references_variable` (references.rs:41-198): an exhaustive match (no `_` arm). -/
def refsNew (ir : IR) (v : Var) : Bool :=
  match ir with
  | .project es cs => es.any (fun e => exprRefs e.2 v) || cs.any (fun c => c.1 == v)
  | .filter e => exprRefs e v
  | .sort es => es.any (exprRefs · v)
  | .aggregate _ ks as ps => ks.any (fun e => exprRefs e.2 v) || as.any (fun e => exprRefs e.2 v) ||
      ps.any (fun c => c.1 == v || c.2 == v)
  | .pathBuilder ps => ps.any (mem v)
  | .unwind e x | .forEach e x => exprRefs e v || x == v
  | .delete es | .remove es => es.any (exprRefs · v)
  | .set items => setItemsRefs items v
  | .merge na ra oc om => qgRefs na ra v || setItemsRefs oc v || setItemsRefs om v
  | .valueHashJoin l r => exprRefs l v || exprRefs r v
  | .procedureCall args _ => args.any (exprRefs · v)
  | .nodeByIndexScan _ q | .edgeByIndexScan _ q => iqRefs q v
  | .nodeByLabelAndIdScan _ f | .nodeByIdSeek _ f => f.any (exprRefs · v)
  | .condVarLenTraverse r ef pv => exprRefs r.attrs v || ef.any (exprRefs · v) || pv.any (· == v)
  | .allShortestPaths r => exprRefs r.attrs v
  | .nodeByFulltextScan l q | .edgeByFulltextScan l q => exprRefs l v || exprRefs q v
  | .nodeByVectorScan l a k w | .edgeByVectorScan l a k w =>
      exprRefs l v || exprRefs a v || exprRefs k v || exprRefs w v
  | .loadCsv f d _ => exprRefs f v || exprRefs d v
  | .skip e | .limit e => exprRefs e v
  | .create na ra => qgRefs na ra v
  | .argument | .optional _ | .allNodeScan _ | .nodeByLabelScan _ | .includePending _
  | .expandInto _ _ | .condTraverse _ _ _ | .cartesianProduct | .apply | .semiApply | .antiSemiApply
  | .orApplyMultiplexer | .distinct | .union | .commit | .createIndex _ | .dropIndex => false
  | .nestedPlans => false

/-- **#2390's oracle is complete** for the plans the planner builds: every variable an
operator's expressions read is reported. -/
theorem refsNew_complete (ir : IR) (v : Var) (hinv : PlanInv ir) (h : v ∈ reads ir) :
    refsNew ir v = true := by
  cases ir
  case condTraverse r ch sib =>
    obtain ⟨⟨h1, h2, h3⟩, hch⟩ := hinv
    simp only [reads, QRel.reads, h1, h2, h3, List.append_nil, List.nil_append, List.mem_flatMap] at h
    obtain ⟨a, ha, h⟩ := h
    obtain ⟨g1, g2, g3⟩ := hch a ha
    simp [g1, g2, g3] at h
  case aggregate ns ks as ps =>
    simp only [reads, List.mem_append, List.mem_flatMap, List.mem_map] at h
    simp only [refsNew, Bool.or_eq_true, List.any_eq_true, exprRefs_iff, beq_iff_eq]
    rcases h with (⟨⟨a, e⟩, he, hv⟩ | ⟨⟨a, e⟩, he, hv⟩) | ⟨⟨a, b⟩, he, rfl⟩
    · exact Or.inl (Or.inl ⟨_, he, hv⟩)
    · exact Or.inl (Or.inr ⟨_, he, hv⟩)
    · exact Or.inr ⟨_, he, Or.inl rfl⟩
  case condVarLenTraverse r ef pv =>
    obtain ⟨h1, h2⟩ := hinv
    simp only [reads, QRel.reads, h1, h2, List.append_nil, List.mem_append] at h
    simp only [refsNew, Bool.or_eq_true, exprRefs_iff]
    rcases h with h | h
    · exact Or.inl (Or.inl h)
    · cases ef <;> simp_all [exprRefs, mem]
  all_goals
    simp_all [reads, refsNew, PlanInv, exprRefs, mem, qgRefs, setItemsRefs_iff, iqRefs_iff,
      List.mem_flatten, List.mem_flatMap, QRel.reads, QNode.reads, SetItem.reads]
  all_goals first
    | (rcases h with h | h | h | h <;> simp_all)
    | (rcases h with h | h | h <;> simp_all)
    | (rcases h with h | h <;> simp_all)
    | skip

/-- …and **sound up to bound variables**: whatever it reports is read, or is a variable the
operator binds (UNWIND/FOREACH's own variable, an Aggregate's output, a CVLT's path
variable) — which only ever keeps a flag on. -/
theorem refsNew_sound (ir : IR) (v : Var) (h : refsNew ir v = true) : v ∈ reads ir ∨ v ∈ binds ir := by
  cases ir <;> simp only [refsNew, Bool.or_eq_true, List.any_eq_true, exprRefs_iff, beq_iff_eq,
    mem_iff, qgRefs, setItemsRefs_iff, iqRefs_iff, Bool.false_eq_true, Option.any_eq_true] at h <;>
    simp only [reads, binds, List.mem_append, List.mem_flatten, List.mem_flatMap, List.mem_map,
      QRel.reads, QNode.reads, List.mem_cons, List.not_mem_nil, Option.mem_toList, Option.getD]
  all_goals first
    | (simp_all [List.mem_flatMap]; done)
    | (rcases h with h | rfl
       · exact Or.inl h
       · exact Or.inr (Or.inl rfl))
    | (rcases h with h | ⟨x, hx, h | h⟩
       · exact Or.inl (Or.inl h)
       · exact Or.inl (Or.inr ⟨x, hx, h⟩)
       · exact Or.inr ⟨x, hx, h⟩)
    | (rcases h with (h | ⟨y, rfl, h⟩) | ⟨y, rfl, rfl⟩
       · exact Or.inl (Or.inl (Or.inl (Or.inl h)))
       · exact Or.inl (Or.inr h)
       · exact Or.inr rfl)
    | skip
  all_goals done

/-! ## `runtimeReadsIds`: the sibling-edge columns (W2-traverse-1, still open) -/

/-- What the operator reads *at runtime*, by id: also the columns of `sibling_edges`
(`edge_already_used`, `runtime/ops/mod.rs:155`). -/
def runtimeReadsIds (ir : IR) : List Nat :=
  (reads ir).map Var.id ++ match ir with
    | .condTraverse _ _ sib | .expandInto _ sib => sib
    | _ => []

def r : Var := ⟨2, 0⟩
def xV : Var := ⟨3, 0⟩
def nd (v : Var) : QNode := ⟨v, []⟩
def sRel : QRel := ⟨⟨4, 0⟩, [], nd ⟨5, 0⟩, nd ⟨6, 0⟩⟩

/-- `MATCH (a)-[r]->(x)<-[s]-(c) RETURN count(*)`: the second hop reads `r`'s column for
relationship uniqueness, but the oracle says nothing reads `r`, so `reduce_expand_into`
collapses it (live on 8743953a8: Rust 1, C 2; proofs/ops_traverse `twoHop_collapse_changes_count`). -/
theorem refsNew_misses_sibling_edges :
    r.id ∈ runtimeReadsIds (.condTraverse sRel [] [r.id]) ∧
    refsNew (.condTraverse sRel [] [r.id]) r = false ∧
    PlanInv (.condTraverse sRel [] [r.id]) := by
  refine ⟨by simp [runtimeReadsIds], rfl, ?_⟩
  simp [PlanInv, sRel, nd]

/-! ## Historical: the oracle at a9377c636 (before #2390) and PR #2918's -/

/-- `ir_references_variable` at a9377c636, arm by arm (reduce_expand_into.rs:27-74). -/
def refsPre2390 (ir : IR) (v : Var) : Bool :=
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

/-- `ir_references_variable` of PR #2918 (exhaustive). It also counts a traverse whose own
alias is `v` (a re-bound edge) — harmless, it only keeps a flag on. -/
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
  | .nodeByIndexScan n q => mem v n.attrs || mem v q.vars
  | .nodeByLabelAndIdScan n f | .nodeByIdSeek n f => mem v n.attrs || f.any (mem v)
  | .nodeByFulltextScan l q | .edgeByFulltextScan l q => mem v l || mem v q
  | .nodeByVectorScan l a k w | .edgeByVectorScan l a k w => mem v l || mem v a || mem v k || mem v w
  | .condTraverse r ch _ => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v ||
      ch.any (fun q => q.alias == v || mem v q.reads || q.src.alias == v || q.dst.alias == v)
  | .condVarLenTraverse r ef _ => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v ||
      mem v (ef.getD [])
  | .edgeByIndexScan r q => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v || mem v q.vars
  | .expandInto r _ | .allShortestPaths r => r.alias == v || mem v r.reads || r.src.alias == v || r.dst.alias == v
  | .createIndex o => mem v (o.getD [])
  | _ => false

/-- **PR #2918's oracle is complete** without any planner invariant (it reads the patterns). -/
theorem refsPR_complete (ir : IR) (v : Var) (h : v ∈ reads ir) : refsPR ir v = true := by
  cases ir
  case condTraverse r ch sib =>
    simp only [reads, QRel.reads, List.mem_append, List.mem_flatMap] at h
    simp only [refsPR, Bool.or_eq_true, beq_iff_eq, mem_iff, QRel.reads, List.mem_append,
      List.any_eq_true]
    rcases h with ((h | h) | h) | ⟨a, ha, (h | h) | h⟩
    · exact Or.inl (Or.inl (Or.inl (Or.inr (Or.inl (Or.inl h)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inr (Or.inl (Or.inr h)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inr (Or.inr h))))
    · exact Or.inr ⟨a, ha, Or.inl (Or.inl (Or.inr (Or.inl (Or.inl h))))⟩
    · exact Or.inr ⟨a, ha, Or.inl (Or.inl (Or.inr (Or.inl (Or.inr h))))⟩
    · exact Or.inr ⟨a, ha, Or.inl (Or.inl (Or.inr (Or.inr h)))⟩
  all_goals simp_all [reads, refsPR, mem, List.mem_flatten, List.mem_flatMap, QRel.reads,
    QNode.reads, SetItem.reads]
  all_goals first
    | (rcases h with h | h | h | h <;> simp_all)
    | (rcases h with h | h | h <;> simp_all)
    | (rcases h with h | h <;> simp_all)
    | skip

/-- The a9377c636 oracle was sound for everything but the bound var of `Unwind`/`ForEach`. -/
theorem pre2390_refs_sound (ir : IR) (v : Var) (h : refsPre2390 ir v = true)
    (hb : ∀ e var, ir = .unwind e var ∨ ir = .forEach e var → False) : v ∈ reads ir := by
  cases ir <;> simp_all [reads, refsPre2390, mem, List.mem_flatten, List.mem_flatMap, SetItem.reads]
  all_goals (rcases h with h | h; exact Or.inl h; exact Or.inr (Or.inl h))

/-! ### Counterexamples on a9377c636, **fixed by #2390 (d2c42e032)** (live on 8743953a8 = C) -/

/-- `MATCH (a)-[r]->(b) UNWIND [r.w] AS w RETURN w`: returned 2 of 3 rows; now 3. -/
theorem pre2390_misses_unwind :
    r ∈ reads (.unwind [r] xV) ∧ refsPre2390 (.unwind [r] xV) r = false := by decide

/-- `MATCH (a)-[r]->(b) CREATE (:C {w: r.w})`: created 2 nodes instead of 3; now 3. -/
theorem pre2390_misses_create :
    r ∈ reads (.create [[r]] []) ∧ refsPre2390 (.create [[r]] []) r = false := by decide

/-- `FOREACH (x IN [r.w] | …)`: now 3 nodes, as C. -/
theorem pre2390_misses_foreach :
    r ∈ reads (.forEach [r] xV) ∧ refsPre2390 (.forEach [r] xV) r = false := by decide

theorem refsNew_sees_unwind : refsNew (.unwind [r] xV) r = true := by decide
theorem refsNew_sees_create : refsNew (.create [[r]] []) r = true := by decide
theorem refsNew_sees_foreach : refsNew (.forEach [r] xV) r = true := by decide

end Falkor.Opt.Refs
