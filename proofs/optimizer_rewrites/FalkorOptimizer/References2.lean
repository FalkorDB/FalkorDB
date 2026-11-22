/-
# Where the oracle is asked: the tree walks of the four passes

Theorems: `anyRead_iff`, `readOutside_iff` (the PR's walk is exactly "some operator outside the
subtree reads v"), `anyRead_plug` (whole plan = subtree ∪ outside), `wholeRead_covers` (the
fusion check sees every reader anywhere, hence `fuse_unread_sound` with `refsNew_complete`),
`ancestors_le_outside`; still open: `main_misses_call_sibling` (#2896's sibling case: the three
`reduce_*` passes still walk ancestors only — live `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w }
RETURN w` → Rust 1 row, C 3).
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

/-- reduce_expand_into / reduce_bound_edge / reduce_var_len_path:
`while let Some(parent) = … { if ir_references_variable(parent) … }`. -/
def ancestorsRead (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) : Bool :=
  ctx.any (fun f => refs f.parent v)

/-- PR #2918 `variable_read_outside`: parent, then every sibling subtree. -/
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

/-- `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w } RETURN w`: the reader is the CALL
body's projection, in Apply's right child — a *sibling* of the traverse. -/
def wV : Var := ⟨7, 0⟩
def callCtx : Ctx := [⟨.apply, [.node (.project [(wV, [r])] []) [.node .argument []]]⟩, ⟨.commit, []⟩]

/-- **#2896, sibling case — still present on 8743953a8.** The per-operator oracle is now
complete, but `reduce_expand_into` / `reduce_bound_edge` / `reduce_var_len_path` still walk
ancestors only, so the reader in the sibling subtree is missed and `r` collapses (live: Rust
`w = 1` only, C 1, 2, 3; pattern comprehension `[(a)-->(b) | r.w]` likewise). -/
theorem main_misses_call_sibling :
    ancestorsRead refsNew callCtx r = false ∧ readOutside refsNew callCtx r = true := by
  refine ⟨rfl, (readOutside_iff refsNew callCtx r).2 ⟨.project [(wV, [r])] [], ?_, ?_⟩⟩
  · simp [outsideOps, callCtx, Plan.ops]
  · decide

/-! ## `reduce_expand_into` since #3054: which edges collapse, and their sibling flag -/

/-- The flag a collapsed edge gets in its siblings' lists (`COLLAPSED_SIBLING` /
`COLLAPSED_SIBLING_ANY_TYPE`, `runtime/ops/mod.rs`). -/
inductive SibFlag where
  | single
  | anyType
deriving DecidableEq

/-- The `match relationship.types.len()` of `reduce_expand_into`: `none` = the edge is
skipped (`continue`: never collapsed), `some f` = it may collapse, flagged `f` (`none`:
unflagged, a multi-type hop among at most two siblings). -/
def collapseFlag (ntypes nsibs : Nat) : Option (Option SibFlag) :=
  match ntypes with
  | 0 => some (some .anyType)
  | 1 => some (some .single)
  | _ => if nsibs > 2 then none else some none

/-- `reduce_expand_into`: the edge `v` (with `ntypes` types and `nsibs` sibling entries)
is collapsed iff the type guard lets it through and no ancestor reads it. -/
def collapses (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) (ntypes nsibs : Nat) : Bool :=
  (collapseFlag ntypes nsibs).isSome && !ancestorsRead refs ctx v

/-- **PROVEN**: #3054 only removes collapses: an edge is still collapsed only when the
ancestor walk finds no reader, so every theorem about `ancestorsRead` carries over. -/
theorem collapses_imp_unread (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) (nt ns : Nat)
    (h : collapses refs ctx v nt ns = true) : ancestorsRead refs ctx v = false := by
  simp only [collapses, Bool.and_eq_true, Bool.not_eq_true'] at h; exact h.2

/-- **PROVEN**: an untyped or single-type edge collapses exactly as before #3054. -/
theorem collapses_le_one_type (refs : IR → Var → Bool) (ctx : Ctx) (v : Var) (nt ns : Nat)
    (hn : nt ≤ 1) : collapses refs ctx v nt ns = !ancestorsRead refs ctx v := by
  rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hn with rfl | rfl <;> simp [collapses, collapseFlag]

/-- **PROVEN**: a hop over several types among three or more siblings is never collapsed
(its types cannot be told apart at run time). -/
theorem multi_type_many_siblings_emits (refs : IR → Var → Bool) (ctx : Ctx) (v : Var)
    (nt ns : Nat) (hn : 2 ≤ nt) (hs : 2 < ns) : collapses refs ctx v nt ns = false := by
  obtain ⟨k, rfl⟩ : ∃ k, nt = k + 2 := ⟨nt - 2, by omega⟩
  simp [collapses, collapseFlag, hs]

/-- The untyped `r` of `main_misses_call_sibling` still collapses after #3054 (bug 7 open). -/
theorem main_misses_call_sibling_collapses : collapses refsNew callCtx r 0 0 = true := by
  rw [collapses_le_one_type _ _ _ _ _ (by decide), main_misses_call_sibling.1]; rfl

end Falkor.Opt.Refs
