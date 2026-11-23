/-
# Where the oracle is asked: the tree walks of the four passes

Theorems: `anyRead_iff`, `readOutside_iff` (the all-outside walk is exactly "some operator
outside the subtree reads v"), `anyRead_plug` (whole plan = subtree ∪ outside), `wholeRead_covers`
(the fusion check sees every reader anywhere, hence `fuse_unread_sound` with `refsNew_complete`),
`ancestors_le_outside`. The three `reduce_*` passes use `variable_read_outside` (#2918),
modelled as `variableReadOutside`: `stackLoop_iff`, `variableReadOutside_iff` (ancestors, plus
the visible operators of every branch that receives the traverse's rows), `_sound` (only
operators outside the subtree), `_complete`, `ancestors_le_variableReadOutside`,
`call_sibling_read_seen` (#2896's sibling case, fixed), `union_branch_not_read`,
`union_body_own_branch_skipped`, `apply_input_not_reached`.
-/
import FalkorOptimizer.References

namespace Falkor.Opt.Refs

open Falkor.Opt

inductive Plan where
  | node (ir : IR) (children : List Plan)

def Plan.anyRead (v : Var) (refs : IR → Var → Bool) : Plan → Bool
  | .node ir cs => refs ir v || (cs.attach.map fun ⟨c, _⟩ => c.anyRead v refs).any id

/-- One level of the zipper: the parent operator and its *other* children. -/
structure Frame where
  parent : IR
  others : List Plan

abbrev Ctx := List Frame   -- innermost first

/-- Rebuild the whole plan from a subtree and its context. -/
def plug : Ctx → Plan → Plan
  | [], t => t
  | f :: ctx, t => plug ctx (.node f.parent (t :: f.others))

/-- The ancestor-only walk the three `reduce_*` passes used before #2918:
`while let Some(parent) = … { if ir_references_variable(parent) … }`. -/
def ancestorsRead (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v)

/-- Every operator outside the subtree: each parent, then every sibling subtree (the spec the
receiving-branch walk `variableReadOutside` refines). -/
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

/-- **The whole plan is the subtree plus everything outside it.** -/
theorem anyRead_plug (refs : IR → Var → Bool) (v : Var) (ctx : Ctx) (t : Plan) :
    (plug ctx t).anyRead v refs = (t.anyRead v refs || readOutside refs ctx v) := by
  induction ctx generalizing t with
  | nil => simp [plug, readOutside]
  | cons f ctx ih =>
    rw [plug, ih]
    have : (Plan.node f.parent (t :: f.others)).anyRead v refs =
        (refs f.parent v || t.anyRead v refs || f.others.any (fun s => s.anyRead v refs)) := by
      simp only [Plan.anyRead]
      cases h1 : refs f.parent v <;> simp [List.attach_cons]
    rw [this]
    simp only [readOutside, List.any_cons]
    cases refs f.parent v <;> cases t.anyRead v refs <;>
      cases f.others.any (fun s => s.anyRead v refs) <;> simp

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

/-- **`fuse_anonymous_traverse` since #2390**: the whole-plan check sees every reader outside
the traversal *and* inside it — strictly more than PR #2918's walk. -/
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

/-! ## `variable_read_outside` (#2918): the walk the three `reduce_*` passes use -/

/-- `ir_binds_variable` (references.rs): a traverse's own edge or path alias. -/
def irBinds (ir : IR) (v : Var) : Bool :=
  match ir with
  | .condTraverse r _ _ | .expandInto r _ | .allShortestPaths r => r.alias == v
  | .condVarLenTraverse r _ pv => r.alias == v || pv.any (· == v)
  | _ => false

def isUnion : IR → Bool
  | .union => true
  | _ => false

/-- `row_reaches_siblings`: an Apply-like parent hands the rows of its input (first) child to
its other children; `Union`, `CartesianProduct` and the hash join run independent children. -/
def rowReachesSiblings (parent : IR) (isInput : Bool) : Bool :=
  match parent with
  | .union | .cartesianProduct | .valueHashJoin _ _ => false
  | _ => isInput

/-- The children `subtree_reads_variable` pushes: all of them, except, under a `Union`, a
branch some operator of which binds `v` itself (`child.walk::<Bfs>().any(ir_binds_variable)`). -/
def keptChildren (v : Var) (ir : IR) (cs : List Plan) : List Plan :=
  cs.filter fun c => !(isUnion ir && c.anyRead v (fun ir v => irBinds ir v))

def Plan.size : Plan → Nat
  | .node _ cs => 1 + (cs.attach.map fun ⟨c, _⟩ => c.size).sum

def sizeL (l : List Plan) : Nat := (l.map Plan.size).sum

theorem sizeL_append (a b : List Plan) : sizeL (a ++ b) = sizeL a + sizeL b := by
  simp [sizeL]

theorem sizeL_reverse (a : List Plan) : sizeL a.reverse = sizeL a := by
  simp [sizeL, List.sum_reverse]

theorem sizeL_filter (p : Plan → Bool) (a : List Plan) : sizeL (a.filter p) ≤ sizeL a := by
  induction a with
  | nil => simp [sizeL]
  | cons c a ih =>
    simp only [sizeL] at *
    by_cases h : p c <;> simp [h] <;> omega

theorem node_size (ir : IR) (cs : List Plan) : (Plan.node ir cs).size = 1 + sizeL cs := by
  simp only [Plan.size, sizeL]
  congr 1
  induction cs with
  | nil => rfl
  | cons c cs ih => simp [List.attach_cons]

/-- `subtree_reads_variable`: `let mut pending = vec![idx]; while let Some(idx) = pending.pop()`.
The stack is a list with its top first; pushing the kept children in order puts the last on
top, so they are `(keptChildren …).reverse ++ rest`. -/
def stackLoop (refs : IR → Var → Bool) (v : Var) : List Plan → Bool
  | [] => false
  | .node ir cs :: rest => refs ir v || stackLoop refs v ((keptChildren v ir cs).reverse ++ rest)
termination_by l => sizeL l
decreasing_by
  have h1 := sizeL_filter (fun c => !(isUnion ir && c.anyRead v (fun ir v => irBinds ir v))) cs
  simp only [sizeL_append, sizeL_reverse, keptChildren]
  have h2 : sizeL (Plan.node ir cs :: rest) = (Plan.node ir cs).size + sizeL rest := by
    simp [sizeL]
  rw [h2, node_size]; omega

/-- **Specification** of `subtree_reads_variable`: the operators of a subtree, minus the
`Union` branches that bind `v` themselves (independent plans whose `v` is their own). -/
def Plan.visibleOps (v : Var) : Plan → List IR
  | .node ir cs => ir :: (cs.attach.flatMap fun ⟨c, _⟩ =>
      if isUnion ir && c.anyRead v (fun ir v => irBinds ir v) then [] else c.visibleOps v)

theorem flatMap_attach_val {β : Type} (l : List Plan) (g : Plan → List β) :
    (l.attach.flatMap fun x => g x.val) = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons c l ih => simp [List.attach_cons, List.flatMap_map]

theorem visibleOps_node (v : Var) (ir : IR) (cs : List Plan) :
    (Plan.node ir cs).visibleOps v = ir :: (keptChildren v ir cs).flatMap (Plan.visibleOps v) := by
  simp only [Plan.visibleOps, keptChildren]
  rw [flatMap_attach_val cs (fun c =>
    if (isUnion ir && c.anyRead v (fun ir v => irBinds ir v)) = true then [] else c.visibleOps v)]
  congr 1
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.flatMap_cons, List.filter_cons]
    cases hb : (isUnion ir && c.anyRead v (fun ir v => irBinds ir v)) <;>
      simp only [Bool.not_false, Bool.not_true, Bool.false_eq_true, ↓reduceIte, List.flatMap_cons,
        List.nil_append, ih]

/-- **The stack loop is exactly "some visible operator of some pending subtree reads `v`"**. -/
theorem stackLoop_iff (refs : IR → Var → Bool) (v : Var) : ∀ (n : Nat) (l : List Plan),
    sizeL l ≤ n → (stackLoop refs v l = true ↔ ∃ p ∈ l, ∃ ir ∈ p.visibleOps v, refs ir v = true) := by
  intro n
  induction n with
  | zero =>
    intro l hl
    cases l with
    | nil => simp [stackLoop]
    | cons p rest =>
      cases p with
      | node ir cs =>
        have : sizeL (Plan.node ir cs :: rest) = (Plan.node ir cs).size + sizeL rest := by
          simp [sizeL]
        rw [this, node_size] at hl; omega
  | succ n ih =>
    intro l hl
    cases l with
    | nil => simp [stackLoop]
    | cons p rest =>
      cases p with
      | node ir cs =>
        have hs : sizeL (Plan.node ir cs :: rest) = (Plan.node ir cs).size + sizeL rest := by
          simp [sizeL]
        have hk := sizeL_filter (fun c => !(isUnion ir && c.anyRead v (fun ir v => irBinds ir v))) cs
        rw [hs, node_size] at hl
        rw [stackLoop, Bool.or_eq_true, ih _ (by
          simp only [sizeL_append, sizeL_reverse, keptChildren]; omega)]
        simp only [List.mem_append, List.mem_reverse, List.mem_cons]
        constructor
        · rintro (h | ⟨p, hp | hp, hr⟩)
          · exact ⟨_, Or.inl rfl, ir, by rw [visibleOps_node]; exact List.mem_cons_self .., h⟩
          · obtain ⟨ir', h1, h2⟩ := hr
            exact ⟨_, Or.inl rfl, ir', by
              rw [visibleOps_node]; exact List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨p, hp, h1⟩), h2⟩
          · exact ⟨p, Or.inr hp, hr⟩
        · rintro ⟨p, rfl | hp, ir', h1, h2⟩
          · rw [visibleOps_node, List.mem_cons, List.mem_flatMap] at h1
            rcases h1 with rfl | ⟨c, hc, h1⟩
            · exact Or.inl h2
            · exact Or.inr ⟨c, Or.inl hc, ir', h1, h2⟩
          · exact Or.inr ⟨p, Or.inr hp, ir', h1, h2⟩

def subtreeReads (refs : IR → Var → Bool) (v : Var) (p : Plan) : Bool := stackLoop refs v [p]

theorem subtreeReads_iff (refs : IR → Var → Bool) (v : Var) (p : Plan) :
    subtreeReads refs v p = true ↔ ∃ ir ∈ p.visibleOps v, refs ir v = true := by
  rw [subtreeReads, stackLoop_iff refs v _ [p] (Nat.le_refl _)]; simp

theorem visibleOps_sub (v : Var) (p : Plan) : ∀ ir ∈ p.visibleOps v, ir ∈ p.ops := by
  induction p using Plan.rec (motive_2 := fun cs => ∀ c ∈ cs, ∀ ir ∈ c.visibleOps v, ir ∈ c.ops) with
  | node ir cs ih =>
    intro x hx
    simp only [Plan.visibleOps, Plan.ops, List.mem_cons, List.mem_flatMap, List.mem_attach,
      true_and, Subtype.exists] at hx ⊢
    rcases hx with rfl | ⟨c, hc, hx⟩
    · exact Or.inl rfl
    · split at hx
      · simp at hx
      · exact Or.inr ⟨c, hc, ih c hc x hx⟩
  | nil => simp_all
  | cons c cs ihc ihcs =>
    rename_i c' hc' x hx
    rcases List.mem_cons.1 hc' with rfl | h
    · exact ihc x hx
    · exact ihcs c' h x hx

/-- One level of the walk: the parent, and the children before and after the subtree `cur`
(`is_input_child` is `before = []`). -/
structure PFrame where
  parent : IR
  before : List Plan
  after : List Plan

def PFrame.toFrame (f : PFrame) : Frame := ⟨f.parent, f.before ++ f.after⟩

/-- **PR #2918 `variable_read_outside`** (references.rs): for each ancestor, its own
expressions, then — if the rows reach its other children — every other child's subtree. -/
def variableReadOutside (refs : IR → Var → Bool) (ctx : List PFrame) (v : Var) : Bool :=
  ctx.any fun f => refs f.parent v ||
    (rowReachesSiblings f.parent f.before.isEmpty &&
      (f.before ++ f.after).any (subtreeReads refs v))

/-- **The walk is exactly its specification**: some ancestor reads `v`, or a visible operator
of a branch that receives the traverse's rows does. -/
theorem variableReadOutside_iff (refs : IR → Var → Bool) (ctx : List PFrame) (v : Var) :
    variableReadOutside refs ctx v = true ↔ ∃ f ∈ ctx, refs f.parent v = true ∨
      (rowReachesSiblings f.parent f.before.isEmpty = true ∧
        ∃ s ∈ f.before ++ f.after, ∃ ir ∈ s.visibleOps v, refs ir v = true) := by
  simp only [variableReadOutside, List.any_eq_true, Bool.or_eq_true, Bool.and_eq_true,
    subtreeReads_iff]

/-- **Sound**: whatever the walk reports is an operator outside the traverse's subtree
(`readOutside_iff`), so with `refsNew_sound` a real read (or a bound variable). -/
theorem variableReadOutside_sound (refs : IR → Var → Bool) (ctx : List PFrame) (v : Var)
    (h : variableReadOutside refs ctx v = true) : readOutside refs (ctx.map PFrame.toFrame) v = true := by
  rw [variableReadOutside_iff] at h
  rw [readOutside_iff]
  obtain ⟨f, hf, h | ⟨_, s, hs, ir, h1, h2⟩⟩ := h
  · exact ⟨f.parent, by
      simp only [outsideOps, List.mem_flatMap, List.mem_map, List.mem_cons]
      exact ⟨f.toFrame, ⟨f, hf, rfl⟩, Or.inl rfl⟩, h⟩
  · exact ⟨ir, by
      simp only [outsideOps, List.mem_flatMap, List.mem_map, List.mem_cons]
      exact ⟨f.toFrame, ⟨f, hf, rfl⟩, Or.inr ⟨s, hs, visibleOps_sub v s ir h1⟩⟩, h2⟩

/-- The ancestor-only walk of main before #2918 implies the new walk (the new walk sees more). -/
theorem ancestors_le_variableReadOutside (refs : IR → Var → Bool) (ctx : List PFrame) (v : Var)
    (h : ancestorsRead refs (ctx.map PFrame.toFrame) v = true) :
    variableReadOutside refs ctx v = true := by
  simp only [ancestorsRead, List.any_map, List.any_eq_true, Function.comp_def] at h
  obtain ⟨f, hf, h⟩ := h
  exact (variableReadOutside_iff refs ctx v).2 ⟨f, hf, Or.inl h⟩

/-- **Complete**, with the complete oracle: a read (`reads`, any IR variant, under `PlanInv`)
by an ancestor or by a visible operator of a receiving branch keeps the flag on. -/
theorem variableReadOutside_complete (ctx : List PFrame) (v : Var)
    (h : ∃ f ∈ ctx, (PlanInv f.parent ∧ v ∈ reads f.parent) ∨
      (rowReachesSiblings f.parent f.before.isEmpty = true ∧
        ∃ s ∈ f.before ++ f.after, ∃ ir ∈ s.visibleOps v, PlanInv ir ∧ v ∈ reads ir)) :
    variableReadOutside refsNew ctx v = true := by
  rw [variableReadOutside_iff]
  obtain ⟨f, hf, ⟨hi, hr⟩ | ⟨hrs, s, hs, ir, h1, hi, hr⟩⟩ := h
  · exact ⟨f, hf, Or.inl (refsNew_complete _ v hi hr)⟩
  · exact ⟨f, hf, Or.inr ⟨hrs, s, hs, ir, h1, refsNew_complete _ v hi hr⟩⟩

/-- `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w } RETURN w`: the reader is the CALL
body's projection, in Apply's right child — a *sibling* of the traverse, whose subtree is
Apply's input (first) child. -/
def wV : Var := ⟨7, 0⟩
def callBody : Plan := .node (.project [(wV, [r])] []) [.node .argument []]
def callCtx : List PFrame := [⟨.apply, [], [callBody]⟩, ⟨.commit, [], []⟩]

/-- **#2896, sibling case — fixed by #2918.** The ancestor walk of main before #2918 missed
the reader in the CALL body (Rust 1 row, C 3); `variable_read_outside` sees it, so
`reduce_expand_into` / `reduce_bound_edge` / `reduce_var_len_path` keep `r` per-edge. -/
theorem call_sibling_read_seen :
    ancestorsRead refsNew (callCtx.map PFrame.toFrame) r = false ∧
    variableReadOutside refsNew callCtx r = true := by
  refine ⟨rfl, (variableReadOutside_iff refsNew callCtx r).2 ⟨⟨.apply, [], [callBody]⟩,
    List.mem_cons_self .., Or.inr ⟨rfl, callBody, List.mem_cons_self .., .project [(wV, [r])] [],
      ?_, by decide⟩⟩⟩
  simp [callBody, Plan.visibleOps]

/-- `MATCH (a)-[r]->(b) RETURN count(*) UNION ALL MATCH (x)-[r]->(y) RETURN id(r)`: the other
`Union` branch's `r` (same `VarKey` from a fresh binder) is not a read of this one. -/
def unionOther : Plan := .node (.project [(wV, [r])] []) [.node (.condTraverse ⟨r, [], nd xV, nd wV⟩ [] []) []]

theorem union_branch_not_read :
    variableReadOutside refsNew [⟨.union, [], [unionOther]⟩] r = false := by
  decide

/-- `CALL { WITH a RETURN 1 AS x UNION ALL MATCH (x)-[r]->(y) RETURN id(r) AS x }` after
`MATCH (a)-[r]->(b)`: the body's `Union` branch that binds its own `r` is skipped, while a
branch that only imports and reads `r` is still a read. -/
def importBranch : Plan := .node (.project [(wV, [r])] []) [.node .argument []]

theorem union_body_own_branch_skipped :
    variableReadOutside refsNew [⟨.apply, [], [.node .union [unionOther]]⟩] r = false ∧
    variableReadOutside refsNew [⟨.apply, [], [.node .union [unionOther, importBranch]]⟩] r = true := by
  refine ⟨?_, (variableReadOutside_iff _ _ _).2 ⟨_, List.mem_cons_self .., Or.inr ⟨rfl,
    _, List.mem_cons_self .., .project [(wV, [r])] [], ?_, by decide⟩⟩⟩
  · rw [Bool.eq_false_iff]; intro h
    rw [variableReadOutside_iff] at h
    simp [visibleOps_node, keptChildren, unionOther, isUnion, Plan.anyRead, irBinds,
      rowReachesSiblings] at h
    rcases h with h | h <;> exact absurd h (by decide)
  · simp [visibleOps_node, keptChildren, importBranch, unionOther, isUnion, Plan.anyRead, irBinds]

/-- A traverse in an Apply's body (not its input child) does not feed the input: the input's
reads of a same-keyed variable are not reads of the body's edge. -/
theorem apply_input_not_reached :
    variableReadOutside refsNew [⟨.apply, [importBranch], []⟩] r = false := by
  decide

end Falkor.Opt.Refs
