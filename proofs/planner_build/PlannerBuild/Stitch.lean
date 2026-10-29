import PlannerBuild.Core
/-
# Clause stitching: `Planner::plan_query` (planner/mod.rs:2824-2971)

Each clause is planned on its own; the plans are then stitched in reverse: the
last clause's plan is the root, and each earlier plan is inserted at an
"insertion point" found by walking down the plan built so far. This file
models the walks and the insertion exactly and proves that the Rust loop
computes the nested fill `pₙ[pₙ₋₁[…p₁]]` (`stitch_eq_nest`), where `p[x]` puts
`x` at `p`'s own insertion point. Whether `p[x]` means "run `x`, then clause
`p` on its output" is then a per-clause question (Clauses.lean, Match.lean).

The one place where the two walks disagree on a plan a clause can end with is
`PathBuilder`: the first walk (mod.rs:2743-2749, and the FOREACH body walk
mod.rs:3450) does not step over it, the in-loop walk (mod.rs:2817-2830) does
(`walkFirst_pathBuilder`, `walkLoop_pathBuilder`). MERGE with a named path is
planned as `PathBuilder(Merge(match))` (mod.rs:3128-3132), so when it is the
last clause the previous clause lands beside `Merge`, under `PathBuilder`,
which only ever runs its child 0: `merge_path_last_misplaced`.
-/
namespace PlannerBuild

abbrev Path := List Nat

/-- Subtree at a path (`tree.node(idx)`). -/
def Plan.get : Plan → Path → Option Plan
  | p, [] => some p
  | .node _ cs, i :: is => match cs[i]? with
    | some c => c.get is
    | none => none

/-- Rewrite the subtree at a path. -/
def Plan.modify (f : Plan → Plan) : Plan → Path → Plan
  | p, [] => f p
  | .node op cs, i :: is => .node op (cs.modify i (fun c => Plan.modify f c is))

/-- Rust's insertion at `idx` (mod.rs:2809-2816): if the node has children,
`child_mut(0).push_sibling_tree(Side::Left, n)`, else `push_child_tree(n)`.
Either way `n` becomes the new child 0. -/
def Plan.push0 : Plan → Plan → Plan
  | .node op cs, x => .node op (x :: cs)

theorem push0_both (x : Plan) (op : Op) (cs : List Plan) :
    (Plan.node op cs).push0 x = .node op (x :: cs) := rfl

def insertAt (p : Plan) (π : Path) (x : Plan) : Plan := p.modify (fun q => q.push0 x) π

/-! ## The walks -/

/-- Operators the first walk steps over (mod.rs:2743-2749). -/
def firstPass : Op → Bool
  | .sort _ | .skip _ | .limit _ | .distinct | .filter _ | .semiApply | .antiSemiApply
  | .orMux => true
  | _ => false

/-- Operators the in-loop walk steps over (mod.rs:2818-2830): the same, plus
the traversals and `PathBuilder` (and `EdgeByIndexScan`, not built by the
planner). -/
def loopPass : Op → Bool
  | .sort _ | .skip _ | .limit _ | .distinct | .filter _ | .semiApply | .antiSemiApply
  | .orMux | .hop _ _ | .pathBuilder _ => true
  | _ => false

def isApply : Op → Bool | .apply => true | _ => false

/-- `is_saturated_apply` (mod.rs:958-963). -/
def saturated : Plan → Bool
  | .node .apply (_ :: _ :: _) => true
  | _ => false

/-- mod.rs:2743-2753. (A childless `Sort`… would make Rust panic on
`child(0)`; the planner never builds one, the model stops.) -/
def walkFirst : Plan → Path
  | .node op (c :: cs) =>
    if firstPass op || (isApply op && !cs.isEmpty) then 0 :: walkFirst c else []
  | .node _ [] => []

/-- mod.rs:2817-2834. -/
def walkLoop : Plan → Path
  | .node op (c :: cs) =>
    if loopPass op || (isApply op && !cs.isEmpty) then 0 :: walkLoop c else []
  | .node _ [] => []

/-- `while child(0) is Apply { idx = child(0) }` (mod.rs:2764-2768). -/
def applyChain : Plan → Path
  | .node _ (c :: _) => if isApply c.op then 0 :: applyChain c else []
  | .node _ [] => []

/-- mod.rs:2756-2769: past `Commit` and the Apply chain below a projection. -/
def projDescend : Plan → Path
  | .node (.project p) (c :: cs) =>
    if c.op == .commit then 0 :: applyChain c else applyChain (.node (.project p) (c :: cs))
  | .node (.aggregate p) (c :: cs) =>
    if c.op == .commit then 0 :: applyChain c else applyChain (.node (.aggregate p) (c :: cs))
  | _ => []

/-- Minimal children of clause operators whose child 0 may be an Apply chain
(`descend_one_clause_expr_chain`, mod.rs:994-1007). -/
def clauseMin : Op → Option Nat
  | .forEach _ _ => some 2
  | .unwind _ _ | .set _ | .remove _ | .delete _ | .procCall _ => some 1
  | _ => none

/-- Descend saturated Applies (mod.rs:1012-1014). -/
def satChain : Plan → Path
  | .node .apply (c :: _ :: _) => 0 :: satChain c
  | _ => []

/-- `descend_one_clause_expr_chain` (mod.rs:990-1017). -/
def descendOne : Plan → Path
  | .node op (c :: cs) =>
    match clauseMin op with
    | some m => if m ≤ (c :: cs).length && isApply c.op then 0 :: satChain c else []
    | none => []
  | .node _ [] => []

/-- `descend_clause_expr_applies` (mod.rs:974-988), with fuel = plan depth. -/
def descendClause : Nat → Plan → Path
  | 0, _ => []
  | fuel + 1, p =>
    match descendOne p with
    | [] => []
    | π => match p.get π with
      | some q => if isApply q.op then π else π ++ descendClause fuel q
      | none => π

def Plan.depth : Plan → Nat
  | .node _ cs => 1 + (cs.map Plan.depth).foldr max 0

/-- The whole walk from a subtree root (first or loop variant). -/
def slotWith (w : Plan → Path) (p : Plan) : Path :=
  let π1 := w p
  match p.get π1 with
  | none => π1
  | some q =>
    let π2 := π1 ++ projDescend q
    match p.get π2 with
    | none => π2
    | some q2 => π2 ++ descendClause q2.depth q2

def slotFirst := slotWith walkFirst
def slotLoop := slotWith walkLoop

/-! ## `needs_apply_wrapping` and `add_argument_to_leaves` -/

/-- mod.rs:2874-2907. -/
def needsApplyWrapping : Plan → Bool
  | .node (.scan _) _ | .node (.hop _ _) _ | .node .cartesian _ | .node .argument _
  | .node (.pathBuilder _) _ => false
  | .node (.filter _) (c :: _) | .node .semiApply (c :: _) | .node .antiSemiApply (c :: _)
  | .node .orMux (c :: _) => needsApplyWrapping c
  | .node (.filter _) [] | .node .semiApply [] | .node .antiSemiApply [] | .node .orMux [] => false
  | _ => true

/-- `add_argument_to_leaves` (mod.rs:870-904): every leaf that is not an
`Argument` gets an `Argument` child; MERGE's match branch (last child) is
skipped, its input (child 0, if it has two children) is descended. -/
def addArgs : Plan → Plan
  | .node (.merge p) (i :: m :: rest) => .node (.merge p) (addArgs i :: m :: rest)
  | .node (.merge p) cs => .node (.merge p) cs
  | .node .argument [] => .node .argument []
  | .node op [] => .node op [.node .argument []]
  | .node op cs => .node op (cs.attach.map fun ⟨c, _⟩ => addArgs c)

/-! ## The stitching loop -/

/-- One insertion (mod.rs:2775-2816): returns the new tree and the path of the
inserted plan's root. The CartesianProduct branch wraps it in `Apply` with the
argument taps added (mod.rs:2785-2808). -/
def insertStep (res : Plan) (idx : Path) (n : Plan) : Plan × Path :=
  match res.get idx with
  | some (.node .cartesian cs) =>
    if needsApplyWrapping n then
      (res.modify (fun _ => .node .apply [n, .node .cartesian (cs.map addArgs)]) idx, idx ++ [0])
    else (insertAt res idx n, idx ++ [0])
  | _ => (insertAt res idx n, idx ++ [0])

/-- The `for n in iter` loop of `plan_query` (mod.rs:2774-2850). -/
def stitchLoop : Plan → Path → List Plan → Plan
  | res, _, [] => res
  | res, idx, n :: rest =>
    let (res', idx') := insertStep res idx n
    let idx'' := match res'.get idx' with
      | some q => idx' ++ slotLoop q
      | none => idx'
    stitchLoop res' idx'' rest

/-- `plan_query` minus the `Commit` wrapper and `ensure_apply_has_input`:
`plans` in clause order. -/
def rustStitch (plans : List Plan) : Plan :=
  match plans.reverse with
  | [] => .node .argument []
  | last :: rest => stitchLoop last (slotFirst last) rest

theorem insertStep_nonCP (res : Plan) (idx : Path) (n : Plan) (op : Op) (cs : List Plan)
    (h : res.get idx = some (.node op cs)) (hc : op ≠ .cartesian) :
    insertStep res idx n = (insertAt res idx n, idx ++ [0]) := by
  unfold insertStep; rw [h]; cases op <;> simp_all

theorem insertStep_CP (res : Plan) (idx : Path) (n : Plan) (cs : List Plan)
    (h : res.get idx = some (.node .cartesian cs)) :
    insertStep res idx n =
      if needsApplyWrapping n then
        (res.modify (fun _ => .node .apply [n, .node .cartesian (cs.map addArgs)]) idx, idx ++ [0])
      else (insertAt res idx n, idx ++ [0]) := by
  unfold insertStep; rw [h]

/-- The same insertion as one plan-level operation: `p[x]`. -/
def fillAt (p : Plan) (π : Path) (x : Plan) : Plan := (insertStep p π x).1

/-- Rust decides the CartesianProduct wrapping on the plan being inserted
*before* its own input is stitched into it; `refill p π x y` makes that
decision on `x` and then puts the finished `y` where `x` went. -/
def refill (p : Plan) (π : Path) (x y : Plan) : Plan :=
  (insertStep p π x).1.modify (fun _ => y) (insertStep p π x).2

/-- Nested fill `pₙ[pₙ₋₁[…p₁]]` (reverse clause order), each earlier plan at
the loop-walk slot of the next one. -/
def nestLoop : List Plan → Plan
  | [] => .node .argument []
  | [p] => p
  | p :: q :: rest => refill p (slotLoop p) q (nestLoop (q :: rest))

/-- The whole stitch, the last clause's slot found by the first walk. -/
def nestStitch (plans : List Plan) : Plan :=
  match plans.reverse with
  | [] => .node .argument []
  | [p] => p
  | p :: q :: rest => refill p (slotFirst p) q (nestLoop (q :: rest))

/-! ### Locality of the tree surgery -/

theorem modify_get_append (f : Plan → Plan) (p : Plan) (π ρ : Path) (q : Plan)
    (h : p.get π = some q) :
    (p.modify f π).get (π ++ ρ) = (f q).get ρ := by
  induction π generalizing p with
  | nil => simp [Plan.get] at h; subst h; simp [Plan.modify]
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.get] at h
      split at h
      · rename_i c hc
        simp only [Plan.modify, List.cons_append, Plan.get]
        rw [List.getElem?_modify]
        simp [hc, ih c h]
      · exact absurd h (by simp)

theorem modify_append (f : Plan → Plan) (p : Plan) (π ρ : Path) :
    p.modify f (π ++ ρ) = p.modify (fun q => q.modify f ρ) π := by
  induction π generalizing p with
  | nil => simp [Plan.modify]
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.modify, List.cons_append]
      congr 1
      apply List.ext_getElem?
      intro k
      rw [List.getElem?_modify, List.getElem?_modify]
      by_cases hk : i = k
      · subst hk; cases cs[i]? <;> simp [ih]
      · simp [hk]

theorem modify_congr (f f' : Plan → Plan) (p : Plan) (π : Path) (q : Plan)
    (h : p.get π = some q) (hf : f q = f' q) : p.modify f π = p.modify f' π := by
  induction π generalizing p with
  | nil => simp [Plan.get] at h; subst h; simp [Plan.modify, hf]
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.get] at h
      split at h
      · rename_i c hc
        simp only [Plan.modify]
        congr 1
        apply List.ext_getElem?
        intro k
        rw [List.getElem?_modify, List.getElem?_modify]
        by_cases hk : i = k
        · subst hk; simp [hc, ih c h]
        · simp [hk]
      · exact absurd h (by simp)

theorem get_modify_self (f : Plan → Plan) (p : Plan) (π : Path) (q : Plan)
    (h : p.get π = some q) : (p.modify f π).get π = some (f q) := by
  have := modify_get_append f p π [] q h
  simpa [Plan.get] using this

theorem get_append (p : Plan) (π ρ : Path) (q : Plan) (h : p.get π = some q) :
    p.get (π ++ ρ) = q.get ρ := by
  induction π generalizing p with
  | nil => simp [Plan.get] at h; subst h; rfl
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.get] at h
      split at h
      · rename_i c hc
        simp [Plan.get, hc, ih c h]
      · exact absurd h (by simp)

/-- `insertStep` inside an embedded subtree = embedding the insertion. -/
theorem insertStep_embed (res : Plan) (π ρ : Path) (sub n : Plan)
    (h : res.get π = some sub) :
    insertStep res (π ++ ρ) n =
      (res.modify (fun _ => (insertStep sub ρ n).1) π, π ++ (insertStep sub ρ n).2) := by
  have hg := get_append res π ρ sub h
  unfold insertStep
  rw [hg]
  split
  · split
    · simp only [List.append_assoc, Prod.mk.injEq, and_true]
      rw [modify_append]
      exact modify_congr _ _ res π sub h rfl
    · simp only [insertAt, List.append_assoc, Prod.mk.injEq, and_true]
      rw [modify_append]
      exact modify_congr _ _ res π sub h rfl
  · simp only [insertAt, List.append_assoc, Prod.mk.injEq, and_true]
    rw [modify_append]
    exact modify_congr _ _ res π sub h rfl

/-! ### Walks return real nodes -/

def Valid (p : Plan) (π : Path) : Prop := ∃ q, p.get π = some q

theorem walkFirst_valid (p : Plan) : Valid p (walkFirst p) := by
  induction p using walkFirst.induct with
  | case1 op c cs h => simp only [walkFirst, h, ite_true]; obtain ⟨q, hq⟩ := ‹Valid c (walkFirst c)›; exact ⟨q, by simp [Plan.get, hq]⟩
  | case2 op c cs h => simp only [walkFirst, h]; exact ⟨_, rfl⟩
  | case3 op => exact ⟨_, rfl⟩

theorem walkLoop_valid (p : Plan) : Valid p (walkLoop p) := by
  induction p using walkLoop.induct with
  | case1 op c cs h => simp only [walkLoop, h, ite_true]; obtain ⟨q, hq⟩ := ‹Valid c (walkLoop c)›; exact ⟨q, by simp [Plan.get, hq]⟩
  | case2 op c cs h => simp only [walkLoop, h]; exact ⟨_, rfl⟩
  | case3 op => exact ⟨_, rfl⟩

theorem valid_cons0 (op : Op) (c : Plan) (cs : List Plan) (π : Path) (h : Valid c π) :
    Valid (.node op (c :: cs)) (0 :: π) := by
  obtain ⟨q, hq⟩ := h; exact ⟨q, by simp [Plan.get, hq]⟩

theorem valid_nil (p : Plan) : Valid p [] := ⟨p, rfl⟩

theorem applyChain_valid (p : Plan) : Valid p (applyChain p) := by
  induction p using applyChain.induct with
  | case1 op c cs h ih => simp only [applyChain, h, ite_true]; exact valid_cons0 _ _ _ _ ih
  | case2 op c cs h => simp only [applyChain, h]; exact valid_nil _
  | case3 op => exact valid_nil _

theorem projDescend_valid (p : Plan) : Valid p (projDescend p) := by
  match p with
  | .node (.project q) (c :: cs) =>
    simp only [projDescend]; split
    · exact valid_cons0 _ _ _ _ (applyChain_valid c)
    · exact applyChain_valid _
  | .node (.aggregate q) (c :: cs) =>
    simp only [projDescend]; split
    · exact valid_cons0 _ _ _ _ (applyChain_valid c)
    · exact applyChain_valid _
  | .node (.project _) [] | .node (.aggregate _) [] => exact valid_nil _
  | .node .argument _ | .node (.filter _) _ | .node .distinct _ | .node (.sort _) _
  | .node (.skip _) _ | .node (.limit _) _ | .node (.unwind _ _) _ | .node (.scan _) _
  | .node (.hop _ _) _ | .node .cartesian _ | .node .apply _ | .node .semiApply _
  | .node .antiSemiApply _ | .node .orMux _ | .node (.optional _) _ | .node (.pathBuilder _) _
  | .node (.create _) _ | .node (.merge _) _ | .node (.set _) _ | .node (.remove _) _
  | .node (.delete _) _ | .node .commit _ | .node (.forEach _ _) _ | .node .union _
  | .node (.procCall _) _ => exact valid_nil _

theorem satChain_valid (p : Plan) : Valid p (satChain p) := by
  induction p using satChain.induct with
  | case1 c d ds ih => simp only [satChain]; exact valid_cons0 _ _ _ _ ih
  | case2 p h => rw [satChain.eq_2 p h]; exact valid_nil _

theorem descendOne_valid (p : Plan) : Valid p (descendOne p) := by
  match p with
  | .node op (c :: cs) =>
    simp only [descendOne]
    split
    · split
      · exact valid_cons0 _ _ _ _ (satChain_valid c)
      · exact valid_nil _
    · exact valid_nil _
  | .node _ [] => exact valid_nil _

theorem descendClause_valid (fuel : Nat) (p : Plan) : Valid p (descendClause fuel p) := by
  induction fuel generalizing p with
  | zero => exact valid_nil _
  | succ n ih =>
    simp only [descendClause]
    split
    · exact valid_nil _
    · rename_i π hπ
      split
      · rename_i q hq
        split
        · exact ⟨q, hq⟩
        · obtain ⟨r, hr⟩ := ih q
          exact ⟨r, by rw [get_append p _ _ q hq, hr]⟩
      · rename_i hn
        obtain ⟨q, hq⟩ := descendOne_valid p
        simp_all

theorem slotWith_valid (w : Plan → Path) (hw : ∀ p, Valid p (w p)) (p : Plan) :
    Valid p (slotWith w p) := by
  unfold slotWith
  obtain ⟨q, hq⟩ := hw p
  simp only [hq]
  obtain ⟨q2, hq2⟩ := projDescend_valid q
  have h2 : p.get (w p ++ projDescend q) = some q2 := by rw [get_append p _ _ q hq, hq2]
  simp only [h2]
  obtain ⟨r, hr⟩ := descendClause_valid q2.depth q2
  exact ⟨r, by rw [get_append p _ _ q2 h2, hr]⟩

theorem slotFirst_valid (p : Plan) : Valid p (slotFirst p) := slotWith_valid _ walkFirst_valid p
theorem slotLoop_valid (p : Plan) : Valid p (slotLoop p) := slotWith_valid _ walkLoop_valid p

/-- After an insertion at a real node, the inserted plan sits at the returned path. -/
theorem insertStep_get (res : Plan) (idx : Path) (n : Plan) (h : Valid res idx) :
    (insertStep res idx n).1.get (insertStep res idx n).2 = some n := by
  obtain ⟨q, hq⟩ := h
  unfold insertStep
  rw [hq]
  split
  · split
    · rw [modify_get_append _ res idx [0] _ hq]; rfl
    · simp only [insertAt]; rw [modify_get_append _ res idx [0] _ hq]; cases q; rfl
  · simp only [insertAt]; rw [modify_get_append _ res idx [0] _ hq]
    cases q; rfl

theorem modify_same (f h : Plan → Plan) (p : Plan) (π : Path) :
    (p.modify f π).modify h π = p.modify (fun q => h (f q)) π := by
  induction π generalizing p with
  | nil => simp [Plan.modify]
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.modify]
      congr 1
      rw [List.modify_modify_eq]
      apply List.ext_getElem?
      intro k
      rw [List.getElem?_modify, List.getElem?_modify]
      by_cases hk : i = k
      · subst hk; cases cs[i]? <;> simp [ih]
      · simp [hk]

theorem modify_const_self (p : Plan) (π : Path) (q : Plan) (h : p.get π = some q) :
    p.modify (fun _ => q) π = p := by
  induction π generalizing p with
  | nil => simp [Plan.get] at h; subst h; rfl
  | cons i is ih =>
    cases p with
    | node op cs =>
      simp only [Plan.get] at h
      split at h
      · rename_i c hc
        simp only [Plan.modify]
        congr 1
        apply List.ext_getElem?
        intro k
        rw [List.getElem?_modify]
        by_cases hk : i = k
        · subst hk; simp [hc, ih c h]
        · simp [hk]
      · exact absurd h (by simp)

/-- One loop iteration, with the inserted plan's own walk. -/
theorem stitchLoop_cons (res : Plan) (idx : Path) (n : Plan) (more : List Plan)
    (h : Valid res idx) :
    stitchLoop res idx (n :: more) =
      stitchLoop (insertStep res idx n).1 ((insertStep res idx n).2 ++ slotLoop n) more := by
  simp only [stitchLoop, insertStep_get res idx n h]

/-- The loop only ever touches the subtree it is working in. -/
theorem stitchLoop_embed (rest : List Plan) (res sub : Plan) (π ρ : Path)
    (h : res.get π = some sub) (hv : Valid sub ρ) :
    stitchLoop res (π ++ ρ) rest = res.modify (fun _ => stitchLoop sub ρ rest) π := by
  induction rest generalizing res sub ρ with
  | nil => simp only [stitchLoop]; rw [modify_const_self res π sub h]
  | cons n more ih =>
    have hv' : Valid res (π ++ ρ) := by
      obtain ⟨q, hq⟩ := hv; exact ⟨q, by rw [get_append res π ρ sub h, hq]⟩
    rw [stitchLoop_cons res _ n more hv', stitchLoop_cons sub ρ n more hv,
      insertStep_embed res π ρ sub n h]
    simp only [List.append_assoc]
    have hA : (res.modify (fun _ => (insertStep sub ρ n).1) π).get π = some (insertStep sub ρ n).1 :=
      get_modify_self _ res π sub h
    have hvA : Valid (insertStep sub ρ n).1 ((insertStep sub ρ n).2 ++ slotLoop n) := by
      obtain ⟨q, hq⟩ := slotLoop_valid n
      exact ⟨q, by rw [get_append _ _ _ n (insertStep_get sub ρ n hv), hq]⟩
    rw [ih _ _ _ hA hvA, modify_same]

/-- **Stitching = nested fill.** The Rust loop builds `pₙ[pₙ₋₁[…p₁]]`, each
earlier plan inserted at the loop-walk slot of the plan after it. -/
theorem stitchLoop_eq_nest (q : Plan) (more : List Plan) (res : Plan) (idx : Path)
    (h : Valid res idx) :
    stitchLoop res idx (q :: more) = refill res idx q (nestLoop (q :: more)) := by
  induction more generalizing q res idx with
  | nil =>
    rw [stitchLoop_cons res idx q [] h]
    simp only [stitchLoop, nestLoop, refill]
    rw [modify_const_self _ _ q (insertStep_get res idx q h)]
  | cons m ms ih =>
    rw [stitchLoop_cons res idx q _ h]
    have hq := insertStep_get res idx q h
    have hsl := slotLoop_valid q
    rw [stitchLoop_embed _ _ q _ _ hq hsl]
    rw [ih m q (slotLoop q) hsl]
    simp only [nestLoop, refill]

theorem rustStitch_eq_nestStitch (plans : List Plan) : rustStitch plans = nestStitch plans := by
  unfold rustStitch nestStitch
  split
  · rename_i h; simp [h]
  · rename_i last rest h
    rw [h]
    cases rest with
    | nil => rfl
    | cons q more =>
      simp only
      exact stitchLoop_eq_nest q more last _ (slotFirst_valid last)

end PlannerBuild
