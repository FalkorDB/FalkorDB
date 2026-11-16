/-
# Where the oracle is asked: the tree walks of the four passes

| here | there (PR #2918 head 46e276b1f) |
| --- | --- |
| `Plan`, `Frame`, `Ctx`, `plug` | an `orx_tree` node addressed by a zipper: parents with their other children; `Frame.input` = `parent.child(0).idx() == cur.idx()` |
| `rowReachesSiblings` | `row_reaches_siblings` (`references.rs:214-222`) |
| `bindsVar`, `Plan.anyBinds` | `ir_binds_variable` (`references.rs:225-243`), `child.walk::<Bfs>().any(..)` (:264-267) |
| `srLoop`, `subtreeReads` | `subtree_reads_variable` (`references.rs:249-274`), its explicit stack |
| `varReadOutside`     | `variable_read_outside` (`references.rs:289-312`), used by `reduce_expand_into` (`reduce_expand_into.rs:39`), `reduce_bound_edge` (`reduce_bound_edge.rs:45`), `reduce_var_len_path` (`reduce_var_len_path.rs:41`) |
| `ancestorsRead`      | the `while let Some(parent)` ancestor walk the three passes used before PR #2918 |
| `wholeRead`          | `intermediate_unreferenced` (`fuse_anonymous_traverse.rs:70-79`, since #2390: BFS of the *whole* plan) |
| `readOutside`        | an earlier revision of PR #2918's walk: parents *and all* their other subtrees |

Theorems: `srLoop_iff` (the stack loop = "some operator the subtree hands the row to reads v",
skipping a nested `Union` branch that binds v itself), `varReadOutside_iff` (the walk is exactly
"some operator in `reachedOps` reads v"), `pr2918_keeps_flag_when_read` (with `refsNew_complete`),
`ancestors_le_varReadOutside` (the new walk never collapses where the ancestor walk kept the flag),
`varReadOutside_le_readOutside` (it reports only operators outside the subtree),
`pr2918_sees_call_sibling` (#2896's sibling case fixed), `pr2918_input_not_seen_from_body`,
`pr2918_cartesian_sibling_not_seen`, `pr2918_union_own_branch_skipped`; `anyRead_iff`,
`readOutside_iff`, `anyRead_plug`, `wholeRead_covers`, `fuse_unread_sound`; historical:
`pre2918_misses_call_sibling` (the ancestor walk, fixed by #2918).
-/
import FalkorOptimizer.References

namespace Falkor.Opt.Refs

open Falkor.Opt

inductive Plan where
  | node (ir : IR) (children : List Plan)

def Plan.anyRead (v : Var) (refs : IR → Var → Bool) : Plan → Bool
  | .node ir cs => refs ir v || (cs.attach.map fun ⟨c, _⟩ => c.anyRead v refs).any id

/-- One level of the zipper: the parent operator, its *other* children, and whether the
subtree is the parent's input (first) child. -/
structure Frame where
  parent : IR
  others : List Plan
  input : Bool

abbrev Ctx := List Frame   -- innermost first

/-- The parent's children: the subtree first when it is the input child, else after the others. -/
def Frame.children (f : Frame) (t : Plan) : List Plan :=
  if f.input then t :: f.others else f.others ++ [t]

/-- Rebuild the whole plan from a subtree and its context. -/
def plug : Ctx → Plan → Plan
  | [], t => t
  | f :: ctx, t => plug ctx (.node f.parent (f.children t))

/-- The three passes before PR #2918:
`while let Some(parent) = … { if ir_references_variable(parent) … }`. -/
def ancestorsRead (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v)

/-- An earlier revision of PR #2918's walk: parent, then every sibling subtree. -/
def readOutside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v || f.others.any (fun s => s.anyRead v refs))

/-- `intermediate_unreferenced` since #2390: `!plan.root().indices::<Bfs>().any(|idx|
ir_references_variable(..))` — every operator of the plan. -/
def wholeRead (refs : IR → Var → Bool) (plan : Plan) (v : Var) : Bool := plan.anyRead v refs

/-- Operators of a plan (spec side). -/
def Plan.ops : Plan → List IR
  | .node ir cs => ir :: (cs.attach.flatMap fun ⟨c, _⟩ => c.ops)

def outsideOps (ctx : Ctx) : List IR :=
  ctx.flatMap (fun f => f.parent :: f.others.flatMap Plan.ops)

theorem anyRead_node (refs : IR → Var → Bool) (v : Var) (ir : IR) (cs : List Plan) :
    (Plan.node ir cs).anyRead v refs = (refs ir v || cs.any (fun c => c.anyRead v refs)) := by
  simp only [Plan.anyRead]
  congr 1
  induction cs with
  | nil => rfl
  | cons c cs ih => simp [List.attach_cons, List.any_cons]

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

/-- **The earlier walk is exactly** "some operator outside the traverse's subtree reads `v`". -/
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

/-- **The whole plan is the subtree plus everything outside it.** -/
theorem anyRead_plug (refs : IR → Var → Bool) (v : Var) (ctx : Ctx) (t : Plan) :
    (plug ctx t).anyRead v refs = (t.anyRead v refs || readOutside refs ctx v) := by
  induction ctx generalizing t with
  | nil => simp [plug, readOutside]
  | cons f ctx ih =>
    rw [plug, ih, anyRead_node]
    have hc : (f.children t).any (fun c => c.anyRead v refs) =
        (t.anyRead v refs || f.others.any (fun s => s.anyRead v refs)) := by
      unfold Frame.children
      cases f.input
      · simp only [Bool.false_eq_true, ↓reduceIte, List.any_append, List.any_cons, List.any_nil,
          Bool.or_false]
        cases t.anyRead v refs <;> simp
      · simp
    rw [hc]
    simp only [readOutside, List.any_cons]
    cases refs f.parent v <;> cases t.anyRead v refs <;>
      cases f.others.any (fun s => s.anyRead v refs) <;> simp

/-- With the complete oracle, the earlier walk keeps the flag whenever the variable is read
anywhere outside the traverse's subtree. -/
theorem pr_keeps_flag_when_read (ctx : Ctx) (v : Var)
    (h : ∃ ir ∈ outsideOps ctx, v ∈ reads ir) : readOutside refsPR ctx v = true :=
  (readOutside_iff refsPR ctx v).2 (by
    obtain ⟨ir, h1, h2⟩ := h; exact ⟨ir, h1, refsPR_complete ir v h2⟩)

/-- Ancestor-only walk implies the full walk (it is weaker). -/
theorem ancestors_le_outside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var)
    (h : ancestorsRead refs ctx v = true) : readOutside refs ctx v = true := by
  simp only [ancestorsRead, readOutside, List.any_eq_true, Bool.or_eq_true] at *
  obtain ⟨f, hf, h⟩ := h; exact ⟨f, hf, Or.inl h⟩

/-- **`fuse_anonymous_traverse` since #2390**: the whole-plan check sees every reader outside
the traversal *and* inside it. -/
theorem wholeRead_covers (refs : IR → Var → Bool) (ctx : Ctx) (t : Plan) (v : Var)
    (h : readOutside refs ctx v = true ∨ t.anyRead v refs = true) :
    wholeRead refs (plug ctx t) v = true := by
  rw [wholeRead, anyRead_plug]; rcases h with h | h <;> simp [h]

/-- …so when it fuses (the intermediate is unreferenced), no operator of the plan that obeys
the planner's invariant reads the intermediate at all. -/
theorem fuse_unread_sound (plan : Plan) (v : Var) (hu : wholeRead refsNew plan v = false)
    (hinv : ∀ ir ∈ plan.ops, PlanInv ir) : ∀ ir ∈ plan.ops, v ∉ reads ir := by
  intro ir hir hv
  have := (anyRead_iff refsNew v plan).2 ⟨ir, hir, refsNew_complete ir v (hinv ir hir) hv⟩
  simp_all [wholeRead]

/-! ## PR #2918: `variable_read_outside` -/

/-- `row_reaches_siblings` (references.rs:214-222): an `Apply`-like parent hands the rows of its
input child to its other children; `Union`, `CartesianProduct` and the hash join never do. -/
def rowReachesSiblings (parent : IR) (isInput : Bool) : Bool :=
  match parent with
  | .union | .cartesianProduct | .valueHashJoin _ _ => false
  | _ => isInput

/-- `ir_binds_variable` (references.rs:225-243): a traverse's edge or path. -/
def bindsVar (ir : IR) (v : Var) : Bool :=
  match ir with
  | .condTraverse q _ _ | .expandInto q _ => q.alias == v
  | .condVarLenTraverse q _ pv => q.alias == v || pv.any (· == v)
  | .allShortestPaths q => q.alias == v
  | _ => false

def isUnion : IR → Bool
  | .union => true
  | _ => false

mutual
/-- `child.walk::<Bfs>().any(|ir| ir_binds_variable(ir, ..))`: the node itself and every descendant. -/
def Plan.anyBinds (v : Var) : Plan → Bool
  | .node ir cs => bindsVar ir v || Plan.anyBindsL v cs
def Plan.anyBindsL (v : Var) : List Plan → Bool
  | [] => false
  | c :: cs => c.anyBinds v || Plan.anyBindsL v cs
end

/-- The children `subtree_reads_variable` pushes: all of them, except, under a `Union`, a branch
that binds the variable itself (references.rs:262-271). -/
def pushed (v : Var) (ir : IR) (cs : List Plan) : List Plan :=
  cs.filter (fun c => !(isUnion ir && c.anyBinds v))

mutual
/-- Spec: the operators of a subtree that receive the variable — every operator, skipping a
nested `Union` branch that binds it itself. -/
def Plan.reachOps (v : Var) : Plan → List IR
  | .node ir cs => ir :: Plan.reachOpsL v ir cs
def Plan.reachOpsL (v : Var) (p : IR) : List Plan → List IR
  | [] => []
  | c :: cs => (if isUnion p && c.anyBinds v then [] else c.reachOps v) ++ Plan.reachOpsL v p cs
end

theorem reachOpsL_eq (v : Var) (p : IR) (cs : List Plan) :
    Plan.reachOpsL v p cs = (pushed v p cs).flatMap (Plan.reachOps v) := by
  induction cs with
  | nil => simp [Plan.reachOpsL, pushed]
  | cons c cs ih =>
    simp only [Plan.reachOpsL, ih, pushed, List.filter_cons]
    cases h : (isUnion p && c.anyBinds v) <;> simp

mutual
def Plan.size : Plan → Nat
  | .node _ cs => 1 + Plan.sizeL cs
def Plan.sizeL : List Plan → Nat
  | [] => 0
  | c :: cs => c.size + Plan.sizeL cs
end

theorem Plan.sizeL_append (a b : List Plan) : Plan.sizeL (a ++ b) = Plan.sizeL a + Plan.sizeL b := by
  induction a with
  | nil => simp [Plan.sizeL]
  | cons c a ih => simp [Plan.sizeL, ih]; omega

theorem Plan.sizeL_reverse (a : List Plan) : Plan.sizeL a.reverse = Plan.sizeL a := by
  induction a with
  | nil => rfl
  | cons c a ih => simp [Plan.sizeL_append, Plan.sizeL, ih]; omega

theorem Plan.sizeL_filter (q : Plan → Bool) (a : List Plan) : Plan.sizeL (a.filter q) ≤ Plan.sizeL a := by
  induction a with
  | nil => simp [Plan.sizeL]
  | cons c a ih =>
    rw [List.filter_cons]
    split <;> simp only [Plan.sizeL] <;> omega

/-- `subtree_reads_variable` (references.rs:249-274): `let mut pending = vec![idx]; while let
Some(idx) = pending.pop() { if ir_references_variable(..) { return true } … pending.push(child) }`.
The stack is a list with its top first, so pushing the children in order is
`(pushed ..).reverse ++ rest`. -/
def srLoop (refs : IR → Var → Bool) (v : Var) : List Plan → Bool
  | [] => false
  | .node ir cs :: rest => refs ir v || srLoop refs v ((pushed v ir cs).reverse ++ rest)
termination_by l => Plan.sizeL l
decreasing_by
  simp only [Plan.sizeL, Plan.size, Plan.sizeL_append, Plan.sizeL_reverse]
  have := Plan.sizeL_filter (fun c => !(isUnion ir && c.anyBinds v)) cs
  simp only [pushed]
  omega

def subtreeReads (refs : IR → Var → Bool) (v : Var) (s : Plan) : Bool := srLoop refs v [s]

/-- **The stack loop is exactly** "some operator the pending subtrees hand the variable to reads it",
in any order and at any depth. -/
theorem srLoop_iff (refs : IR → Var → Bool) (v : Var) : ∀ (n : Nat) (l : List Plan),
    Plan.sizeL l ≤ n → (srLoop refs v l = true ↔ ∃ ir ∈ l.flatMap (Plan.reachOps v), refs ir v = true) := by
  intro n
  induction n with
  | zero =>
    intro l hl
    cases l with
    | nil => simp [srLoop]
    | cons p rest => cases p; simp [Plan.sizeL, Plan.size] at hl
  | succ n ih =>
    intro l hl
    cases l with
    | nil => simp [srLoop]
    | cons p rest =>
      cases p with
      | node ir cs =>
        have hf := Plan.sizeL_filter (fun c => !(isUnion ir && c.anyBinds v)) cs
        simp only [Plan.sizeL, Plan.size] at hl
        rw [srLoop, Bool.or_eq_true, ih _ (by
          simp only [Plan.sizeL_append, Plan.sizeL_reverse, pushed]; omega)]
        simp only [List.flatMap_cons, Plan.reachOps, reachOpsL_eq, List.mem_append, List.mem_cons,
          List.flatMap_append, List.mem_flatMap, List.mem_reverse]
        constructor
        · rintro (h | ⟨x, (⟨c, hc, hx⟩ | ⟨c, hc, hx⟩), h⟩)
          · exact ⟨ir, Or.inl (Or.inl rfl), h⟩
          · exact ⟨x, Or.inl (Or.inr ⟨c, hc, hx⟩), h⟩
          · exact ⟨x, Or.inr ⟨c, hc, hx⟩, h⟩
        · rintro ⟨x, (rfl | ⟨c, hc, hx⟩) | ⟨c, hc, hx⟩, h⟩
          · exact Or.inl h
          · exact Or.inr ⟨x, Or.inl ⟨c, hc, hx⟩, h⟩
          · exact Or.inr ⟨x, Or.inr ⟨c, hc, hx⟩, h⟩

theorem subtreeReads_iff (refs : IR → Var → Bool) (v : Var) (s : Plan) :
    subtreeReads refs v s = true ↔ ∃ ir ∈ s.reachOps v, refs ir v = true := by
  rw [subtreeReads, srLoop_iff refs v _ [s] (Nat.le_refl _)]; simp

/-- `variable_read_outside` (references.rs:289-312): each parent, then — when the row reaches
them — every other child's subtree. -/
def varReadOutside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v ||
    (rowReachesSiblings f.parent f.input && f.others.any (fun s => subtreeReads refs v s)))

/-- Spec: the operators outside the subtree that receive its rows. -/
def reachedOps (v : Var) (ctx : Ctx) : List IR :=
  ctx.flatMap (fun f => f.parent ::
    (if rowReachesSiblings f.parent f.input then f.others.flatMap (Plan.reachOps v) else []))

/-- **PR #2918's walk is exactly** "some operator that receives the traverse's rows reads `v`". -/
theorem varReadOutside_iff (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) :
    varReadOutside refs ctx v = true ↔ ∃ ir ∈ reachedOps v ctx, refs ir v = true := by
  simp only [varReadOutside, reachedOps, List.any_eq_true, Bool.or_eq_true, Bool.and_eq_true,
    List.mem_flatMap, List.mem_cons]
  constructor
  · rintro ⟨f, hf, h | ⟨hr, s, hs, h⟩⟩
    · exact ⟨f.parent, ⟨f, hf, Or.inl rfl⟩, h⟩
    · obtain ⟨ir, h1, h2⟩ := (subtreeReads_iff refs v s).1 h
      exact ⟨ir, ⟨f, hf, Or.inr (by simp only [hr, ↓reduceIte, List.mem_flatMap]; exact ⟨s, hs, h1⟩)⟩, h2⟩
  · rintro ⟨ir, ⟨f, hf, rfl | h1⟩, h2⟩
    · exact ⟨f, hf, Or.inl h2⟩
    · cases hr : rowReachesSiblings f.parent f.input
      · simp [hr] at h1
      · simp only [hr, ↓reduceIte, List.mem_flatMap] at h1
        obtain ⟨s, hs, h1⟩ := h1
        exact ⟨f, hf, Or.inr ⟨hr, s, hs, (subtreeReads_iff refs v s).2 ⟨ir, h1, h2⟩⟩⟩

/-- **PR #2918 keeps the flag** whenever an operator that receives the rows (and obeys the
planner's invariant) reads the variable. -/
theorem pr2918_keeps_flag_when_read (ctx : Ctx) (v : Var)
    (h : ∃ ir ∈ reachedOps v ctx, PlanInv ir ∧ v ∈ reads ir) : varReadOutside refsNew ctx v = true :=
  (varReadOutside_iff refsNew ctx v).2 (by
    obtain ⟨ir, h1, hi, h2⟩ := h; exact ⟨ir, h1, refsNew_complete ir v hi h2⟩)

/-- The new walk never collapses where the ancestor walk kept the flag. -/
theorem ancestors_le_varReadOutside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var)
    (h : ancestorsRead refs ctx v = true) : varReadOutside refs ctx v = true := by
  simp only [ancestorsRead, varReadOutside, List.any_eq_true, Bool.or_eq_true] at *
  obtain ⟨f, hf, h⟩ := h; exact ⟨f, hf, Or.inl h⟩

theorem reachOps_sub (v : Var) (p : Plan) : ∀ ir ∈ p.reachOps v, ir ∈ p.ops := by
  induction p using Plan.rec (motive_2 := fun cs => ∀ c ∈ cs, ∀ ir ∈ c.reachOps v, ir ∈ c.ops) with
  | node ir cs ih =>
    intro x hx
    simp only [Plan.reachOps, reachOpsL_eq, List.mem_cons, List.mem_flatMap, pushed,
      List.mem_filter] at hx
    simp only [Plan.ops, List.mem_cons, List.mem_flatMap, List.mem_attach, true_and,
      Subtype.exists]
    rcases hx with rfl | ⟨c, ⟨hc, _⟩, hx⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨c, hc, ih c hc x hx⟩
  | nil => rename_i h _ _; cases h
  | cons c cs ihc ihcs =>
    rename_i c' hc' x hx
    rcases List.mem_cons.1 hc' with rfl | h
    · exact ihc x hx
    · exact ihcs c' h x hx

/-- …and reports only operators outside the traverse's subtree. -/
theorem varReadOutside_le_readOutside (refs : IR → Var → Bool) (ctx : Ctx) (v : Var)
    (h : varReadOutside refs ctx v = true) : readOutside refs ctx v = true := by
  obtain ⟨ir, h1, h2⟩ := (varReadOutside_iff refs ctx v).1 h
  refine (readOutside_iff refs ctx v).2 ⟨ir, ?_, h2⟩
  simp only [reachedOps, outsideOps, List.mem_flatMap, List.mem_cons] at h1 ⊢
  obtain ⟨f, hf, h1 | h1⟩ := h1
  · exact ⟨f, hf, Or.inl h1⟩
  · split at h1
    · simp only [List.mem_flatMap] at h1
      obtain ⟨s, hs, h1⟩ := h1
      exact ⟨f, hf, Or.inr ⟨s, hs, reachOps_sub v s ir h1⟩⟩
    · cases h1

/-! ### Examples -/

/-- `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w } RETURN w`: the reader is the CALL
body's projection, in Apply's right child — a *sibling* of the traverse. -/
def wV : Var := ⟨7, 0⟩
def callBody : Plan := .node (.project [(wV, [r])] []) [.node .argument []]
def callCtx : Ctx := [⟨.apply, [callBody], true⟩, ⟨.commit, [], true⟩]

/-- **#2896, sibling case — before PR #2918.** The ancestor walk misses the reader in the
sibling subtree, so `r` collapsed (live on 8743953a8: Rust `w = 1` only, C 1, 2, 3). -/
theorem pre2918_misses_call_sibling :
    ancestorsRead refsNew callCtx r = false ∧ readOutside refsNew callCtx r = true := by
  refine ⟨rfl, (readOutside_iff refsNew callCtx r).2 ⟨.project [(wV, [r])] [], ?_, ?_⟩⟩
  · simp [outsideOps, callCtx, callBody, Plan.ops]
  · decide

/-- **Fixed by PR #2918**: the CALL body receives the traverse's rows, so its read is seen. -/
theorem pr2918_sees_call_sibling : varReadOutside refsNew callCtx r = true :=
  (varReadOutside_iff refsNew callCtx r).2 ⟨.project [(wV, [r])] [], by
    simp [reachedOps, callCtx, callBody, rowReachesSiblings, Plan.reachOps, reachOpsL_eq, pushed,
      isUnion], by decide⟩

/-- A traverse in the CALL body: the Apply's input never sees the body's rows. -/
theorem pr2918_input_not_seen_from_body :
    varReadOutside refsNew [⟨.apply, [callBody], false⟩] r = false := by
  decide

/-- A `CartesianProduct` branch never sees the other branch's rows. -/
theorem pr2918_cartesian_sibling_not_seen :
    varReadOutside refsNew [⟨.cartesianProduct, [callBody], true⟩] r = false := by
  decide

/-- A nested `Union` branch that binds `r` itself is skipped: its `r` is its own variable
(the binder numbers each branch afresh). -/
def ownBranch : Plan := .node (.project [(wV, [r])] [])
  [.node (.condTraverse ⟨r, [], nd xV, nd wV⟩ [] []) [.node .argument []]]

theorem pr2918_union_own_branch_skipped :
    varReadOutside refsNew [⟨.apply, [.node .union [ownBranch]], true⟩] r = false ∧
    varReadOutside refsNew [⟨.apply, [.node .union [callBody]], true⟩] r = true := by
  constructor
  · rw [Bool.eq_false_iff]
    intro h
    obtain ⟨ir, h1, h2⟩ := (varReadOutside_iff refsNew _ r).1 h
    simp [reachedOps, rowReachesSiblings, Plan.reachOps, reachOpsL_eq, pushed, isUnion, ownBranch,
      Plan.anyBinds, Plan.anyBindsL, bindsVar] at h1
    rcases h1 with rfl | rfl <;> revert h2 <;> decide
  · refine (varReadOutside_iff refsNew _ r).2 ⟨.project [(wV, [r])] [], ?_, by decide⟩
    simp [reachedOps, rowReachesSiblings, Plan.reachOps, reachOpsL_eq, pushed, isUnion, callBody,
      Plan.anyBinds, Plan.anyBindsL, bindsVar]

end Falkor.Opt.Refs
