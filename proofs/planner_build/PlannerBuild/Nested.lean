import PlannerBuild.Expr
/-
# Pattern comprehensions: extraction, hoisting vs nested plans, their sub-plans

| here | there |
| --- | --- |
| `PSt` | `Planner` fields `scope_vars` (as lengths), `visited`, `loop_vars`, `pattern_vars`, `nested_plans` (mod.rs:719-752); `minted` is a ghost log of `fresh_var` results |
| `fresh` | `Planner::fresh_var` mod.rs:843-857 |
| `EC` | `ExtractedComprehension` mod.rs:756-773 |
| `CP` | the IR built for a comprehension (Aggregate / Filter / Apply / PathBuilder / plan_match) |
| `build`, `buildL` | `build_pattern_comprehension_plan` mod.rs:1406-1488 |
| `interleave` | path components, mod.rs:1123-1129 |
| `hoist` | `hoist_or_nest` mod.rs:1200-1245 |
| `extract`, `extractL` | `extract_pattern_comprehensions` mod.rs:1064-1193 |
| `chainOf`, `extractClause` | `extract_clause_expr_comprehensions` mod.rs:1309-1340 / `extract_list_expr_comprehensions` mod.rs:1292-1298 |
| `extractFilter` | `extract_filter_comprehensions` mod.rs:1364-1402 |
| `finish` | `Planner::finish` mod.rs:828-840 |

`plan_match`'s own plan is abstract (`CP.matchG g visited`); it marks the
pattern's variables visited (here: `g.vars`); verified-label bookkeeping is in
PlanMatch.lean.
-/
namespace PlannerBuild.E

structure PSt where
  lens : Nat → Nat
  visited : List V
  loopVars : List V
  patVars : List V
  nestedN : Nat
  minted : List V

def fresh (st : PSt) (s : Nat) : V × PSt :=
  (⟨st.lens s, s⟩, { st with lens := fun t => if t = s then st.lens s + 1 else st.lens t,
                               minted := st.minted ++ [⟨st.lens s, s⟩] })

inductive EC
  | mk (var : V) (g : QG) (wh : Option Ex) (res : Ex) (path : Option (V × List V)) (nested : List EC)

def EC.var : EC → V | .mk v .. => v

inductive CP
  | matchG (g : QG) (visited : List V)
  | path (pv : V) (comps : List V) (c : CP)
  | apply (x s : CP)
  | filter (e : Ex) (c : CP)
  | agg (var acc : V) (res : Ex) (c : CP)
  | applyIn (s : CP)
  | nestedPlans (main : CP) (ns : List CP)

def withPath : Option (V × List V) → CP → CP
  | some (pv, cs), c => .path pv cs c
  | none, c => c

def withFilter : Option Ex → CP → CP
  | some e, c => .filter e c
  | none, c => c

mutual
def build (st : PSt) : EC → CP × PSt
  | .mk var g wh res path nested =>
    let saved := st.visited
    let sub := withPath path (.matchG g st.visited)
    let x := buildL { st with visited := st.visited ++ g.vars } sub nested
    let st3 := { x.2 with visited := saved }
    let a := fresh st3 var.scope
    (.agg var a.1 res (withFilter wh x.1), a.2)
def buildL (st : PSt) (sub : CP) : List EC → CP × PSt
  | [] => (sub, st)
  | i :: is =>
    let p := build st i
    buildL p.2 (.apply sub p.1) is
end

/-- Path components: node, rel, node, rel, …, node (mod.rs:1123-1129). -/
def interleave : List V → List V → List V
  | [], _ => []
  | n :: ns, r :: rs => n :: r :: interleave ns rs
  | n :: ns, [] => n :: interleave ns []

def isLCQ : D → Bool
  | .listComp _ | .quant _ => true
  | _ => false

/-- Loop variables a node binds, and the first child they scope over. -/
def binds : D → List V × Option Nat
  | .listComp v | .quant v => ([v], some 1)
  | .reduce acc it => ([acc, it], some 2)
  | _ => ([], none)

def inScope (first : Option Nat) (i : Nat) : Bool :=
  match first with
  | some f => decide (f ≤ i)
  | none => false

def gt0 (l : Ex) : Ex := .node .gt [.node .length [l], .node (.const (.int 0)) []]

def hoist (st : PSt) (acc : List EC) (node : Ex) (comp : EC) : Ex × PSt × List EC :=
  let reads := patternExprVariables node
  if !(reads.any fun v => st.loopVars.contains v) then (Ex.v comp.var, st, acc ++ [comp])
  else
    let saved := st.visited
    let st1 := { st with visited := st.visited ++ st.loopVars ++ st.patVars }
    let b := build st1 comp
    let free := reads.filter fun v => b.2.visited.contains v
    let id := b.2.nestedN
    (.node (.nested id comp.var) (free.map Ex.v),
     { b.2 with visited := saved, nestedN := b.2.nestedN + 1 }, acc)

mutual
def extract (sc : Nat) (st : PSt) (acc : List EC) (m : Mode) : Ex → Ex × PSt × List EC
  | .node d cs => match d with
    | .patComp g => match cs with
      | w :: r :: _ =>
        let a := fresh st sc
        let st1 := { a.2 with patVars := a.2.patVars ++ g.vars }
        let x := extract sc st1 [] .exists_ w
        let wh := if isTT x.1 then none else some x.1
        let y := extract sc x.2.1 x.2.2 .collect r
        hoist { y.2.1 with patVars := st.patVars } acc (.node d cs) (.mk a.1 g wh y.1 none y.2.2)
      | _ => (.node d cs, st, acc)
    | .pat g =>
      if m = .semiApply then
        let x := extractL sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs
        (.node d x.1, x.2.1, x.2.2)
      else
        let a := fresh st sc
        let b := fresh a.2 sc
        let h := hoist b.2 acc (.node d cs)
          (.mk a.1 g none (Ex.v b.1) (some (b.1, interleave g.nodes g.rels)) [])
        (if m = .exists_ then gt0 h.1 else h.1, h.2.1, h.2.2)
    | _ =>
      let x := extractL sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs
      (.node d x.1, x.2.1, x.2.2)
def extractL (sc : Nat) (st : PSt) (acc : List EC) (d : D) (bound : List V) (first : Option Nat)
    (cm : Mode) : Nat → List Ex → List Ex × PSt × List EC
  | _, [] => ([], st, acc)
  | i, c :: cs =>
    let st1 := if inScope first i then { st with loopVars := st.loopVars ++ bound } else st
    let m' := if i = 1 ∧ isLCQ d = true then Mode.exists_ else cm
    let x := extract sc st1 acc m' c
    let y := extractL sc { x.2.1 with loopVars := st.loopVars } x.2.2 d bound first cm (i + 1) cs
    (x.1 :: y.1, y.2.1, y.2.2)
end

/-- `Planner::finish`: root at `NestedPlans` only when nested plans exist. -/
def finish (main : CP) (ns : List CP) : CP := if ns.isEmpty then main else .nestedPlans main ns

theorem finish_empty (main : CP) : finish main [] = main := rfl
theorem finish_nonempty (main : CP) (n : CP) (ns : List CP) :
    finish main (n :: ns) = .nestedPlans main (n :: ns) := rfl

/-! ## Frame: what extraction leaves alone -/

def Frame (st st' : PSt) : Prop :=
  st'.visited = st.visited ∧ st'.loopVars = st.loopVars ∧ st'.patVars = st.patVars

mutual
theorem build_frame : ∀ (st : PSt) (c : EC),
    (build st c).2.visited = st.visited ∧ (build st c).2.loopVars = st.loopVars ∧
      (build st c).2.patVars = st.patVars
  | st, .mk var g wh res path nested => by
    have := buildL_frame { st with visited := st.visited ++ g.vars } (withPath path (.matchG g st.visited)) nested
    simp only [build, fresh]
    exact ⟨by trivial, this.1, this.2⟩
theorem buildL_frame : ∀ (st : PSt) (sub : CP) (l : List EC),
    (buildL st sub l).2.loopVars = st.loopVars ∧ (buildL st sub l).2.patVars = st.patVars
  | st, _, [] => ⟨rfl, rfl⟩
  | st, sub, i :: is => by
    have h1 := build_frame st i
    have h2 := buildL_frame (build st i).2 (.apply sub (build st i).1) is
    simp only [buildL]
    exact ⟨h2.1.trans h1.2.1, h2.2.trans h1.2.2⟩
end

theorem hoist_frame (st : PSt) (acc : List EC) (node : Ex) (comp : EC) :
    Frame st (hoist st acc node comp).2.1 := by
  unfold hoist
  dsimp only
  split
  · exact ⟨rfl, rfl, rfl⟩
  · have := build_frame { st with visited := st.visited ++ st.loopVars ++ st.patVars } comp
    exact ⟨rfl, this.2.1, this.2.2⟩

theorem Frame.trans {a b c : PSt} (h1 : Frame a b) (h2 : Frame b c) : Frame a c :=
  ⟨h2.1.trans h1.1, h2.2.1.trans h1.2.1, h2.2.2.trans h1.2.2⟩

theorem fresh_frame (st : PSt) (s : Nat) : Frame st (fresh st s).2 := ⟨rfl, rfl, rfl⟩

mutual
/-- Extraction restores `visited`, `loop_vars` and `pattern_vars` (the
truncations at mod.rs:1098 and 1063; the visited save/restore at mod.rs:1223/1110). -/
theorem extract_frame (sc : Nat) : ∀ (st : PSt) acc m (e : Ex), Frame st (extract sc st acc m e).2.1
  | st, acc, m, .node d cs => by
    have hL := extractL_frame sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs
    cases d <;> (try simp only [extract]) <;> (try exact hL)
    case patComp g =>
      match cs with
      | w :: r :: rest =>
        let a := fresh st sc
        let st1 : PSt := { a.2 with patVars := a.2.patVars ++ g.vars }
        have hx := extract_frame sc st1 [] .exists_ w
        let x := extract sc st1 [] .exists_ w
        have hy := extract_frame sc x.2.1 x.2.2 .collect r
        let y := extract sc x.2.1 x.2.2 .collect r
        have hmid : Frame st { y.2.1 with patVars := st.patVars } :=
          ⟨hy.1.trans hx.1, hy.2.1.trans hx.2.1, rfl⟩
        exact hmid.trans (hoist_frame _ _ _ _)
      | [] => exact ⟨rfl, rfl, rfl⟩
      | [_] => exact ⟨rfl, rfl, rfl⟩
    case pat g =>
      split
      · exact hL
      · exact ((fresh_frame _ _).trans (fresh_frame _ _)).trans (hoist_frame _ _ _ _)
theorem extractL_frame (sc : Nat) : ∀ (st : PSt) acc d bound first cm i (l : List Ex),
    Frame st (extractL sc st acc d bound first cm i l).2.1
  | st, acc, d, bound, first, cm, i, [] => ⟨rfl, rfl, rfl⟩
  | st, acc, d, bound, first, cm, i, c :: cs => by
    simp only [extractL]
    let st1 := if inScope first i then { st with loopVars := st.loopVars ++ bound } else st
    let m' := if i = 1 ∧ isLCQ d = true then Mode.exists_ else cm
    have hx := extract_frame sc st1 acc m' c
    let x := extract sc st1 acc m' c
    have hy := extractL_frame sc { x.2.1 with loopVars := st.loopVars } x.2.2 d bound first cm (i + 1) cs
    have h1 : st1.visited = st.visited ∧ st1.patVars = st.patVars := by
      show (if inScope first i then _ else _ : PSt).visited = _ ∧ (if inScope first i then _ else _ : PSt).patVars = _
      split <;> exact ⟨rfl, rfl⟩
    have hmid : Frame st { x.2.1 with loopVars := st.loopVars } :=
      ⟨hx.1.trans h1.1, rfl, hx.2.2.trans h1.2⟩
    exact hmid.trans hy
end

/-! ## Every minted variable is new -/

/-- Minted variables are pairwise distinct and lie in `[lens0 s, lens s)` of their scope. -/
def MInv (lens0 : Nat → Nat) (st : PSt) : Prop :=
  st.minted.Nodup ∧ (∀ v ∈ st.minted, lens0 v.scope ≤ v.id ∧ v.id < st.lens v.scope) ∧
    ∀ s, lens0 s ≤ st.lens s

theorem fresh_minv (lens0 : Nat → Nat) (st : PSt) (s : Nat) (h : MInv lens0 st) :
    MInv lens0 (fresh st s).2 := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨?_, ?_, ?_⟩
  · simp only [fresh]
    rw [List.nodup_append]
    refine ⟨h1, by simp, ?_⟩
    intro a ha b hb e
    simp only [List.mem_singleton] at hb; subst hb; subst e
    have := (h2 _ ha).2; simp at this
  · intro v hv
    simp only [fresh, List.mem_append, List.mem_singleton] at hv ⊢
    rcases hv with hv | rfl
    · have := h2 v hv
      by_cases hs : v.scope = s
      · simp only [hs, ↓reduceIte]; rw [hs] at this; omega
      · simp only [hs, ↓reduceIte]; exact this
    · simp only [↓reduceIte]; exact ⟨h3 s, Nat.lt_succ_self _⟩
  · intro t
    simp only [fresh]
    split
    · subst t; have := h3 s; omega
    · exact h3 t

/-- A state change that only touches non-id fields. -/
theorem minv_congr (lens0 : Nat → Nat) (st st' : PSt) (h1 : st'.lens = st.lens) (h2 : st'.minted = st.minted)
    (h : MInv lens0 st) : MInv lens0 st' := by
  obtain ⟨a, b, c⟩ := h
  exact ⟨h2 ▸ a, fun v hv => h1 ▸ b v (h2 ▸ hv), fun s => h1 ▸ c s⟩

mutual
theorem build_minv (lens0 : Nat → Nat) : ∀ (st : PSt) (c : EC), MInv lens0 st → MInv lens0 (build st c).2
  | st, .mk var g wh res path nested, h => by
    simp only [build]
    apply fresh_minv
    apply minv_congr lens0 _ _ rfl rfl
    exact buildL_minv lens0 _ _ nested (minv_congr lens0 st _ rfl rfl h)
theorem buildL_minv (lens0 : Nat → Nat) : ∀ (st : PSt) (sub : CP) (l : List EC), MInv lens0 st →
    MInv lens0 (buildL st sub l).2
  | st, _, [], h => h
  | st, sub, i :: is, h => by
    simp only [buildL]
    exact buildL_minv lens0 _ _ is (build_minv lens0 st i h)
end

theorem hoist_minv (lens0 : Nat → Nat) (st : PSt) (acc : List EC) (node : Ex) (comp : EC)
    (h : MInv lens0 st) : MInv lens0 (hoist st acc node comp).2.1 := by
  unfold hoist
  dsimp only
  split
  · exact h
  · apply minv_congr lens0 _ _ rfl rfl
    exact build_minv lens0 _ comp (minv_congr lens0 st _ rfl rfl h)

mutual
theorem extract_minv (lens0 : Nat → Nat) (sc : Nat) : ∀ (st : PSt) acc m (e : Ex), MInv lens0 st →
    MInv lens0 (extract sc st acc m e).2.1
  | st, acc, m, .node d cs, h => by
    have hL := extractL_minv lens0 sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs h
    cases d <;> (try simp only [extract]) <;> (try exact hL)
    case patComp g =>
      match cs with
      | w :: r :: rest =>
        let a := fresh st sc
        let st1 : PSt := { a.2 with patVars := a.2.patVars ++ g.vars }
        have h1 : MInv lens0 st1 := minv_congr lens0 a.2 _ rfl rfl (fresh_minv lens0 st sc h)
        have hx := extract_minv lens0 sc st1 [] .exists_ w h1
        let x := extract sc st1 [] .exists_ w
        have hy := extract_minv lens0 sc x.2.1 x.2.2 .collect r hx
        let y := extract sc x.2.1 x.2.2 .collect r
        exact hoist_minv lens0 _ _ _ _ (minv_congr lens0 y.2.1 _ rfl rfl hy)
      | [] => exact h
      | [_] => exact h
    case pat g =>
      split
      · exact hL
      · exact hoist_minv lens0 _ _ _ _ (fresh_minv lens0 _ _ (fresh_minv lens0 _ _ h))
theorem extractL_minv (lens0 : Nat → Nat) (sc : Nat) : ∀ (st : PSt) acc d bound first cm i (l : List Ex),
    MInv lens0 st → MInv lens0 (extractL sc st acc d bound first cm i l).2.1
  | st, acc, d, bound, first, cm, i, [], h => h
  | st, acc, d, bound, first, cm, i, c :: cs, h => by
    simp only [extractL]
    let st1 := if inScope first i then { st with loopVars := st.loopVars ++ bound } else st
    have h1 : MInv lens0 st1 := by
      show MInv lens0 (if inScope first i then _ else _)
      split
      · exact minv_congr lens0 st _ rfl rfl h
      · exact h
    let m' := if i = 1 ∧ isLCQ d = true then Mode.exists_ else cm
    have hx := extract_minv lens0 sc st1 acc m' c h1
    let x := extract sc st1 acc m' c
    exact extractL_minv lens0 sc { x.2.1 with loopVars := st.loopVars } x.2.2 d bound first cm (i + 1) cs
      (minv_congr lens0 x.2.1 _ rfl rfl hx)
end

/-- **Freshness.** Every variable the extraction mints (comprehension lists,
path variables, `collect` accumulators — inside nested comprehensions too) is
distinct from every other one, and from every variable the binder handed over
(`id < lens0 scope`, see proofs/binder `mint_fresh_iff`). -/
theorem extract_fresh (sc : Nat) (st : PSt) (acc : List EC) (m : Mode) (e : Ex) (h0 : st.minted = []) :
    (extract sc st acc m e).2.1.minted.Nodup ∧
    ∀ v ∈ (extract sc st acc m e).2.1.minted, ∀ w : V, w.id < st.lens w.scope → v ≠ w := by
  have h := extract_minv st.lens sc st acc m e ⟨by simp [h0], by simp [h0], fun _ => Nat.le_refl _⟩
  refine ⟨h.1, fun v hv w hw e => ?_⟩
  subst e
  have := (h.2.1 v hv).1; omega

/-! ## After extraction nothing is left to extract -/

mutual
/-- `PatternComprehension` nodes carry their WHERE and result children (the parser
always builds both, mod.rs:1082/968 read `child(0)`/`child(1)`). -/
def pcOK : Ex → Bool
  | .node (.patComp _) cs => decide (2 ≤ cs.length) && pcOKL cs
  | .node _ cs => pcOKL cs
def pcOKL : List Ex → Bool
  | [] => true
  | c :: cs => pcOK c && pcOKL cs
end

mutual
theorem needs_le_exists : ∀ (e : Ex) (m : Mode), needsExtraction e m = true → needsExtraction e .exists_ = true
  | .node d cs, m, h => by
    rw [needsExtraction_exists]
    cases d <;> simp only [needsExtraction] at h <;> simp only [hasPatternExpr] <;>
      (try exact (needsExtractionL_exists cs) ▸ needsL_le_exists cs _ h) <;> rfl
theorem needsL_le_exists : ∀ (l : List Ex) (m : Mode), needsExtractionL l m = true →
    needsExtractionL l .exists_ = true
  | [], _, h => by simp [needsExtractionL] at h
  | c :: cs, m, h => by
    simp only [needsExtractionL, Bool.or_eq_true] at h ⊢
    rcases h with h | h
    · exact Or.inl (needs_le_exists c m h)
    · exact Or.inr (needsL_le_exists cs m h)
end

theorem needs_of_noPat (e : Ex) (m : Mode) (h : hasPatternExpr e = false) : needsExtraction e m = false := by
  cases hn : needsExtraction e m
  · rfl
  · have := needs_le_exists e m hn; rw [needsExtraction_exists, h] at this; exact absurd this (by simp)

theorem noPat_vars (vs : List V) : hasPatternExprL (vs.map Ex.v) = false := by
  induction vs with
  | nil => rfl
  | cons v vs ih => simp [hasPatternExprL, hasPatternExpr, Ex.v, ih]

theorem hoist_noPat (st : PSt) (acc : List EC) (node : Ex) (comp : EC) :
    hasPatternExpr (hoist st acc node comp).1 = false := by
  unfold hoist; dsimp only
  split
  · rfl
  · simp only [hasPatternExpr]; exact noPat_vars _

theorem gt0_noPat (l : Ex) (h : hasPatternExpr l = false) : hasPatternExpr (gt0 l) = false := by
  simp [gt0, hasPatternExpr, hasPatternExprL, h]

mutual
/-- **Completeness.** In whatever mode, the rewritten expression needs no more
extraction: no pattern comprehension is left, and an existential pattern is
left only where `collect_patterns_and_rebuild` will turn it into a
semi-join (SemiApply mode, under AND/OR/NOT/paren). -/
theorem extract_clean (sc : Nat) : ∀ (st : PSt) acc m (e : Ex), pcOK e = true →
    needsExtraction (extract sc st acc m e).1 m = false
  | st, acc, m, .node d cs, hp => by
    have hpl : pcOKL cs = true := by cases d <;> simp_all [pcOK]
    have hL := extractL_clean sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs hpl
    cases d
    case patComp g =>
      match cs, hp with
      | w :: r :: rest, _ =>
        simp only [extract]
        exact needs_of_noPat _ _ (hoist_noPat _ _ _ _)
      | [], hp => simp [pcOK] at hp
      | [_], hp => simp [pcOK] at hp
    case pat g =>
      simp only [extract]
      split
      · rename_i hm; subst hm; simp [needsExtraction]
      · split
        · exact needs_of_noPat _ _ (gt0_noPat _ (hoist_noPat _ _ _ _))
        · exact needs_of_noPat _ _ (hoist_noPat _ _ _ _)
    all_goals (simp only [extract, needsExtraction]; exact hL)
theorem extractL_clean (sc : Nat) : ∀ (st : PSt) acc d bound first cm i (l : List Ex), pcOKL l = true →
    needsExtractionL (extractL sc st acc d bound first cm i l).1 cm = false
  | st, acc, d, bound, first, cm, i, [], _ => rfl
  | st, acc, d, bound, first, cm, i, c :: cs, hp => by
    simp only [pcOKL, Bool.and_eq_true] at hp
    simp only [extractL, needsExtractionL, Bool.or_eq_false_iff]
    refine ⟨?_, extractL_clean sc _ _ d bound first cm (i + 1) cs hp.2⟩
    by_cases hm : i = 1 ∧ isLCQ d = true
    · rw [if_pos hm]
      have := extract_clean sc (if inScope first i then { st with loopVars := st.loopVars ++ bound } else st)
        acc .exists_ c hp.1
      cases hn : needsExtraction (extract sc (if inScope first i then { st with loopVars := st.loopVars ++ bound } else st)
        acc .exists_ c).1 cm
      · rfl
      · rw [needs_le_exists _ cm hn] at this; exact absurd this (by simp)
    · rw [if_neg hm]; exact extract_clean sc _ acc cm c hp.1
end

/-! ## `hoist_or_nest`, the comprehension sub-plan, the clause/filter wrappers -/

mutual
theorem build_nestedN : ∀ (st : PSt) (c : EC), (build st c).2.nestedN = st.nestedN
  | st, .mk var g wh res path nested => by
    simp only [build, fresh]
    exact buildL_nestedN _ _ nested
theorem buildL_nestedN : ∀ (st : PSt) (sub : CP) (l : List EC), (buildL st sub l).2.nestedN = st.nestedN
  | st, _, [] => rfl
  | st, sub, i :: is => by
    simp only [buildL]
    rw [buildL_nestedN _ _ is, build_nestedN st i]
end

/-- Hoisted (queued for an Apply below the clause) exactly when the
comprehension reads no enclosing loop variable; then it is replaced by its list
variable. Otherwise it becomes nested plan number `nestedN`, whose children are
the variables it reads that are bound in the row (outer stream, loop variables,
enclosing comprehension's pattern variables). -/
theorem hoist_spec (st : PSt) (acc : List EC) (node : Ex) (comp : EC) :
    (∀ v ∈ patternExprVariables node, v ∉ st.loopVars) →
      hoist st acc node comp = (Ex.v comp.var, st, acc ++ [comp]) := by
  intro h
  unfold hoist; dsimp only
  have : (patternExprVariables node).any (fun v => st.loopVars.contains v) = false := by
    rw [List.any_eq_false]; intro v hv; simpa using h v hv
  rw [this]; rfl

theorem hoist_nested (st : PSt) (acc : List EC) (node : Ex) (comp : EC)
    (h : ∃ v ∈ patternExprVariables node, v ∈ st.loopVars) :
    hoist st acc node comp =
      (.node (.nested st.nestedN comp.var)
          (((patternExprVariables node).filter fun v => (st.visited ++ st.loopVars ++ st.patVars).contains v).map Ex.v),
       { (build { st with visited := st.visited ++ st.loopVars ++ st.patVars } comp).2 with
          visited := st.visited, nestedN := st.nestedN + 1 }, acc) := by
  obtain ⟨v, hv, hl⟩ := h
  have hany : (patternExprVariables node).any (fun v => st.loopVars.contains v) = true :=
    List.any_eq_true.2 ⟨v, hv, by simpa using hl⟩
  have hb := build_frame { st with visited := st.visited ++ st.loopVars ++ st.patVars } comp
  have hn := build_nestedN { st with visited := st.visited ++ st.loopVars ++ st.patVars } comp
  unfold hoist; dsimp only
  simp only [hany, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hn, hb.1]

/-- A hoisted comprehension's value cannot depend on the loop: if its meaning
depends only on the variables it reads, re-binding loop variables leaves it
unchanged. (Soundness of computing it once per row, below the clause.) -/
theorem hoisted_loop_invariant {Val : Type} (reads loop : List V) (h : ∀ v ∈ reads, v ∉ loop)
    (f : (V → Val) → Val) (hf : ∀ ρ ρ', (∀ v ∈ reads, ρ v = ρ' v) → f ρ = f ρ')
    (ρ : V → Val) (upd : V → Val) :
    f (fun v => if v ∈ loop then upd v else ρ v) = f ρ := by
  apply hf; intro v hv; simp [h v hv]

end PlannerBuild.E
