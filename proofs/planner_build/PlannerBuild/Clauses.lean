import PlannerBuild.Stitch
/-
# Clause plans vs openCypher clause semantics

For each clause kind, `planX` is the plan `Planner::plan` builds for it
(mod.rs:2829-3366; `plan_project` mod.rs:2310-2481), and `specX` is the
openCypher meaning of the clause as a map from the incoming driving table to
the outgoing one (with the graph state threaded, clause at a time). The
theorems say: filling the clause plan's insertion point with *any* input plan
`x` evaluates to `specX` applied to what `x` evaluates to.

Together with `stitch_eq_nest` (Stitch.lean) and `nest_correct` below this
gives: the stitched plan of `c₁ … cₙ` evaluates to `specₙ ∘ … ∘ spec₂` of the
first clause's result — the openCypher clause-sequence semantics.

What a clause *reads* (pattern matching, expression values) is a parameter
(`Sem`), and so are the sub-plans the clause is handed (MERGE's match branch,
FOREACH's and CALL's bodies): each theorem assumes the sub-plan implements
its sub-query, which is the same statement one level down.
-/
namespace PlannerBuild

variable {V G : Type} (S : Sem V G)

/-! ## WITH / RETURN: `plan_project` -/

/-- The projection part of a WITH/RETURN clause (`QueryIR::With`/`Return`),
pattern comprehensions excluded (see the gaps list). -/
structure ProjC where
  p : Nat
  agg : Bool
  distinct : Bool
  order : Option Nat
  skip : Option Nat
  limit : Option Nat
  filter : Option Nat
  write : Bool

def wrapOpt (mk : Nat → Op) : Option Nat → Plan → Plan
  | some k, p => .node (mk k) [p]
  | none, p => p

/-- mod.rs:2392-2479, in the same order: Project/Aggregate [Commit], then
Distinct (only without aggregation, mod.rs:2452), Sort, Skip, Limit, Filter. -/
def planProject (c : ProjC) : Plan :=
  let base := Plan.node (if c.agg then .aggregate c.p else .project c.p)
    (if c.write then [.node .commit []] else [])
  let d := if c.distinct && !c.agg then Plan.node .distinct [base] else base
  wrapOpt .filter c.filter (wrapOpt .limit c.limit (wrapOpt .skip c.skip (wrapOpt .sort c.order d)))

def optApply {A : Type} (f : Nat → A → A) : Option Nat → A → A
  | some k, a => f k a
  | none, a => a

/-- openCypher WITH/RETURN: project (or group), DISTINCT, ORDER BY, SKIP,
LIMIT, then WITH's WHERE. -/
def specProject (c : ProjC) (r : Rec V) (T : Res V G) : Res V G :=
  let rows1 := if c.agg then S.agg c.p r T.1 else T.1.map fun row => S.proj c.p row T.2
  let rows2 := if c.distinct then S.dedup rows1 else rows1
  let rows3 := optApply S.sort c.order rows2
  let rows4 := optApply (fun n rs => rs.drop n) c.skip rows3
  let rows5 := optApply (fun n rs => rs.take n) c.limit rows4
  (optApply (fun φ rs => rs.filter fun row => S.test φ row T.2) c.filter rows5, T.2)

/-- `DISTINCT` over an aggregation is the identity (groups have distinct keys);
the planner relies on it to drop the `Distinct` (mod.rs:2452). -/
def AggDistinct (c : ProjC) : Prop := ∀ r rows, S.dedup (S.agg c.p r rows) = S.agg c.p r rows

/-! ## Generic machinery: implementing a clause at the insertion point -/

/-- `p` implements the table transformer `F` at the slot found by walk `w`:
whatever plan `y` is stitched there (the CartesianProduct decision being taken
on `q`, see `refill`), the result evaluates to `F` of what `y` evaluates to,
provided `q` satisfies `ok`. -/
def Implements (w : Plan → Path) (ok : Plan → Prop) (p : Plan)
    (F : Rec V → Res V G → Res V G) : Prop :=
  ∀ q y r g, ok q → ev S (refill p (slotWith w p) q y) r g = F r (ev S y r g)

/-- A walk that steps over `W`. -/
def Steps (w : Plan → Path) (W : Op) : Prop := ∀ p, w (.node W [p]) = 0 :: w p

theorem slotWith_step (w : Plan → Path) (W : Op) (hw : Steps w W) (p : Plan) :
    slotWith w (.node W [p]) = 0 :: slotWith w p := by
  have hproj : ∀ q : Plan, projDescend q = projDescend q := fun _ => rfl
  unfold slotWith
  rw [hw p]
  simp only [Plan.get, List.getElem?_cons_zero]
  split <;> simp_all [Plan.get]
  split <;> simp_all

theorem refill_step (W : Op) (p : Plan) (π : Path) (q y : Plan) (hv : Valid p π) :
    refill (.node W [p]) (0 :: π) q y = .node W [refill p π q y] := by
  unfold refill
  have h : (Plan.node W [p]).get [0] = some p := rfl
  have := insertStep_embed (.node W [p]) [0] π p q h
  simp only [List.singleton_append] at this
  rw [this]
  simp only
  rw [show (0 :: (insertStep p π q).2) = [0] ++ (insertStep p π q).2 from rfl, modify_append,
    modify_same]
  rfl

theorem steps_loop (W : Op) (h : loopPass W = true) : Steps walkLoop W := by
  intro p; simp [walkLoop, h]

theorem steps_first (W : Op) (h : firstPass W = true) : Steps walkFirst W := by
  intro p; simp [walkFirst, h]

/-- `W`'s meaning on its (single) input. -/
def UnarySem (W : Op) (WS : Rec V → Res V G → Res V G) : Prop :=
  ∀ z r g, ev S (.node W [z]) r g = WS r (ev S z r g)

theorem implements_step (w : Plan → Path) (hw : ∀ p, Valid p (w p)) (W : Op) (hs : Steps w W)
    (WS : Rec V → Res V G → Res V G) (hW : UnarySem S W WS) (ok : Plan → Prop) (p : Plan)
    (F : Rec V → Res V G → Res V G) (hp : Implements S w ok p F) :
    Implements S w ok (.node W [p]) (fun r T => WS r (F r T)) := by
  intro q y r g hq
  rw [slotWith_step w W hs p, refill_step W p _ q y (slotWith_valid w hw p), hW, hp q y r g hq]

/-- The per-operator meanings used below (all by unfolding `ev`). -/
def rowSem (W : Op) : Rec V → Res V G → Res V G := fun _ T => forRows (S.step W) T.1 T.2
def bagSem (W : Op) : Rec V → Res V G → Res V G := fun r T => S.bag W r T.1 T.2

theorem unary_row (W : Op) (h1 : W.isBag = false)
    (h2 : W ≠ .argument ∧ W ≠ .union ∧ W ≠ .cartesian ∧ W ≠ .orMux ∧ W.isCorr = false) :
    UnarySem S W (rowSem S W) := by
  intro z r g
  obtain ⟨a, b, c, d, e⟩ := h2
  cases W <;> simp_all [Op.isBag, Op.isCorr] <;> rfl

theorem unary_bag (W : Op) (h1 : W.isBag = true) : UnarySem S W (bagSem S W) := by
  intro z r g
  cases W <;> simp_all [Op.isBag] <;> rfl

/-- A leaf clause operator: the input goes in as its only child. -/
theorem implements_leaf (w : Plan → Path) (hw0 : ∀ op, w (.node op []) = []) (W : Op)
    (hc : W ≠ .cartesian)
    (hproj : projDescend (.node W []) = []) (hd : descendClause 1 (.node W []) = [])
    (WS : Rec V → Res V G → Res V G) (hW : UnarySem S W WS) (ok : Plan → Prop) :
    Implements S w ok (.node W []) WS := by
  intro q y r g _
  have hslot : slotWith w (.node W []) = [] := by
    unfold slotWith; rw [hw0]; simp only [Plan.get, List.nil_append, hproj]
    show descendClause (Plan.node W []).depth (.node W []) = []
    have : (Plan.node W []).depth = 1 := by simp [Plan.depth]
    rw [this, hd]
  rw [hslot]
  rw [refill, insertStep_nonCP _ [] q W [] rfl hc]
  show ev S (Plan.node W [y]) r g = _
  exact hW y r g

theorem depth_succ (p : Plan) : ∃ k, p.depth = k + 1 := by
  cases p with | node op cs => exact ⟨(cs.map Plan.depth).foldr max 0, by simp [Plan.depth]; omega⟩

theorem descendClause_nil (p : Plan) (h : descendOne p = []) : descendClause p.depth p = [] := by
  obtain ⟨k, hk⟩ := depth_succ p
  rw [hk]; simp [descendClause, h]

/-- Walk properties shared by `walkLoop` and `walkFirst`. -/
structure GoodWalk (w : Plan → Path) : Prop where
  valid : ∀ p, Valid p (w p)
  leaf : ∀ op, w (.node op []) = []
  one : ∀ op s, loopPass op = false → w (.node op [s]) = []
  filter : Steps w (.filter 0) ∧ ∀ k, Steps w (.filter k)
  sort : ∀ k, Steps w (.sort k)
  skip : ∀ k, Steps w (.skip k)
  limit : ∀ k, Steps w (.limit k)
  distinct : Steps w .distinct

theorem goodWalk_loop : GoodWalk walkLoop where
  valid := walkLoop_valid
  leaf := fun _ => rfl
  one := fun op s h => by simp [walkLoop, h]
  filter := ⟨steps_loop _ rfl, fun _ => steps_loop _ rfl⟩
  sort := fun _ => steps_loop _ rfl
  skip := fun _ => steps_loop _ rfl
  limit := fun _ => steps_loop _ rfl
  distinct := steps_loop _ rfl

theorem goodWalk_first : GoodWalk walkFirst where
  valid := walkFirst_valid
  leaf := fun _ => rfl
  one := fun op s h => by
    simp [walkFirst]; cases op <;> simp_all [firstPass, loopPass]
  filter := ⟨steps_first _ rfl, fun _ => steps_first _ rfl⟩
  sort := fun _ => steps_first _ rfl
  skip := fun _ => steps_first _ rfl
  limit := fun _ => steps_first _ rfl
  distinct := steps_first _ rfl

/-- The input of a correlated operator with only its sub-plan goes in as child 0. -/
theorem refill_corr (w : Plan → Path) (hw : GoodWalk w) (W : Op) (s q y : Plan)
    (hl : loopPass W = false) (hc : W ≠ .cartesian)
    (hproj : projDescend (.node W [s]) = []) (hd : descendOne (.node W [s]) = []) :
    refill (.node W [s]) (slotWith w (.node W [s])) q y = .node W [y, s] := by
  have hslot : slotWith w (.node W [s]) = [] := by
    unfold slotWith; rw [hw.one W s hl]; simp only [Plan.get, List.nil_append, hproj]
    exact descendClause_nil _ hd
  rw [hslot, refill, insertStep_nonCP _ [] q W [s] rfl hc]
  rfl

/-! ## Clause by clause -/

theorem forRows_filter (k : Nat) (rows : List (Rec V)) (g : G) :
    forRows (S.step (.filter k)) rows g = (rows.filter fun row => S.test k row g, g) := by
  rw [forRows_ro (S.step (.filter k)) (fun row g => if S.test k row g then [row] else []) (fun _ _ => rfl)]
  congr 1
  induction rows with
  | nil => rfl
  | cons x xs ih => by_cases h : S.test k x g <;> simp_all [List.filter_cons]

theorem forRows_project (k : Nat) (rows : List (Rec V)) (g : G) :
    forRows (S.step (.project k)) rows g = (rows.map fun row => S.proj k row g, g) := by
  rw [forRows_ro (S.step (.project k)) (fun row g => [S.proj k row g]) (fun _ _ => rfl)]
  congr 1
  induction rows with
  | nil => rfl
  | cons x xs ih => simp [ih]

theorem forRows_id (op : Op) (h : ∀ r g, S.step op r g = ([r], g)) (rows : List (Rec V)) (g : G) :
    forRows (S.step op) rows g = (rows, g) := by
  rw [forRows_ro (S.step op) (fun row _ => [row]) h]; simp

theorem descendClause_leaf (op : Op) : descendClause 1 (.node op []) = [] := by
  simp [descendClause, descendOne]

theorem unary_rowSem (W : Op) (h1 : W.isBag = false)
    (h2 : W ≠ .argument ∧ W ≠ .union ∧ W ≠ .cartesian ∧ W ≠ .orMux ∧ W.isCorr = false) :
    UnarySem S W (rowSem S W) := unary_row S W h1 h2

/-- A per-row leaf clause (UNWIND, CREATE, SET, REMOVE, DELETE, a procedure
call): its plan is the operator alone (mod.rs:2979-2991, 3030-3038, 3042-3104,
2912-2916), and it implements "run the step on every incoming row". -/
theorem implements_rowLeaf (w : Plan → Path) (hw : GoodWalk w) (W : Op)
    (h1 : W.isBag = false)
    (h2 : W ≠ .argument ∧ W ≠ .union ∧ W ≠ .cartesian ∧ W ≠ .orMux ∧ W.isCorr = false)
    (hproj : projDescend (.node W []) = []) (ok : Plan → Prop) :
    Implements S w ok (.node W []) (rowSem S W) :=
  implements_leaf S w hw.leaf W h2.2.2.1 hproj (descendClause_leaf W) _ (unary_row S W h1 h2) ok

/-- openCypher UNWIND: each record, once per list element, extended with it. -/
def specUnwind (e : Nat) (x : Var) (T : Res V G) : Res V G :=
  (T.1.flatMap fun row => (S.items e row T.2).map (row.set x), T.2)

theorem unwind_correct (w : Plan → Path) (hw : GoodWalk w) (e : Nat) (x : Var) (ok : Plan → Prop)
    (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (.node (.unwind e x) []) (slotWith w (.node (.unwind e x) [])) q y) r g =
      specUnwind S e x (ev S y r g) := by
  rw [implements_rowLeaf S w hw (.unwind e x) rfl (by simp [Op.isCorr]) rfl ok q y r g hq]
  exact forRows_ro _ (fun row g => (S.items e row g).map (row.set x)) (fun _ _ => rfl) _ _

/-- openCypher CREATE / SET / REMOVE / DELETE: per record, in order, threading
the graph. -/
def specCreate (p : Nat) (T : Res V G) : Res V G :=
  forRows (fun row g => ([(S.create p row g).1], (S.create p row g).2)) T.1 T.2
def specWrite (p : Nat) (T : Res V G) : Res V G :=
  forRows (fun row g => ([row], S.write p row g)) T.1 T.2

theorem create_correct (w : Plan → Path) (hw : GoodWalk w) (p : Nat) (ok : Plan → Prop)
    (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (.node (.create p) []) (slotWith w (.node (.create p) [])) q y) r g =
      specCreate S p (ev S y r g) :=
  implements_rowLeaf S w hw (.create p) rfl (by simp [Op.isCorr]) rfl ok q y r g hq

theorem set_correct (w : Plan → Path) (hw : GoodWalk w) (p : Nat) (ok : Plan → Prop)
    (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (.node (.set p) []) (slotWith w (.node (.set p) [])) q y) r g =
      specWrite S p (ev S y r g) :=
  implements_rowLeaf S w hw (.set p) rfl (by simp [Op.isCorr]) rfl ok q y r g hq

theorem remove_correct (w : Plan → Path) (hw : GoodWalk w) (p : Nat) (ok : Plan → Prop)
    (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (.node (.remove p) []) (slotWith w (.node (.remove p) [])) q y) r g =
      specWrite S p (ev S y r g) :=
  implements_rowLeaf S w hw (.remove p) rfl (by simp [Op.isCorr]) rfl ok q y r g hq

theorem delete_correct (w : Plan → Path) (hw : GoodWalk w) (p : Nat) (ok : Plan → Prop)
    (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (.node (.delete p) []) (slotWith w (.node (.delete p) [])) q y) r g =
      specWrite S p (ev S y r g) :=
  implements_rowLeaf S w hw (.delete p) rfl (by simp [Op.isCorr]) rfl ok q y r g hq

/-- `CALL proc(...) YIELD ... [WHERE φ]` (mod.rs:2836-2927, non-index procedures). -/
def planProc (p : Nat) (φ : Option Nat) : Plan := wrapOpt .filter φ (.node (.procCall p) [])

def specProc (p : Nat) (φ : Option Nat) (T : Res V G) : Res V G :=
  let rows := T.1.flatMap fun row => S.proc p row T.2
  (optApply (fun k rs => rs.filter fun row => S.test k row T.2) φ rows, T.2)

theorem proc_correct (w : Plan → Path) (hw : GoodWalk w) (p : Nat) (φ : Option Nat)
    (ok : Plan → Prop) (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (planProc p φ) (slotWith w (planProc p φ)) q y) r g =
      specProc S p φ (ev S y r g) := by
  have base := implements_rowLeaf S w hw (.procCall p) rfl (by simp [Op.isCorr]) rfl ok
  cases φ with
  | none =>
    rw [show planProc p none = .node (.procCall p) [] from rfl, base q y r g hq]
    exact forRows_ro _ (fun row g => S.proc p row g) (fun _ _ => rfl) _ _
  | some k =>
    have h := implements_step S w hw.valid (.filter k) (hw.filter.2 k) _
      (unary_row S (.filter k) rfl (by simp [Op.isCorr])) ok _ _ base
    rw [show planProc p (some k) = .node (.filter k) [.node (.procCall p) []] from rfl,
      h q y r g hq]
    simp only [rowSem, specProc, optApply]
    rw [show forRows (S.step (.procCall p)) (ev S y r g).1 (ev S y r g).2 =
      ((ev S y r g).1.flatMap fun row => S.proc p row (ev S y r g).2, (ev S y r g).2) from
      forRows_ro _ (fun row g => S.proc p row g) (fun _ _ => rfl) _ _]
    exact forRows_filter S k _ _

/-! ### Correlated clauses: MERGE, FOREACH, CALL {}, OPTIONAL MATCH -/

/-- openCypher MERGE, per record: the pattern's matches (`mt`) if any, else
create it; optionally bind the named path afterwards. -/
def specMerge (p : Nat) (mt : Rec V → G → List (Rec V)) (path : Option Nat)
    (T : Res V G) : Res V G :=
  let merged := forRows (fun row g =>
    if (mt row g).isEmpty then ([(S.create p row g).1], (S.create p row g).2)
    else (mt row g, g)) T.1 T.2
  match path with
  | none => merged
  | some q => (merged.1.map (S.path q), merged.2)

/-- mod.rs:2995-3028: `Merge(match_branch)`, under a `PathBuilder` when the
pattern has named paths. -/
def planMerge (p : Nat) (m : Plan) (path : Option Nat) : Plan :=
  match path with
  | none => .node (.merge p) [m]
  | some q => .node (.pathBuilder q) [.node (.merge p) [m]]

/-- The match branch implements the pattern: read-only, returns `mt`. -/
def MatchBranch (m : Plan) (mt : Rec V → G → List (Rec V)) : Prop :=
  ∀ row g, ev S m row g = (mt row g, g)

theorem ev_merge_two (p : Nat) (m y : Plan) (mt : Rec V → G → List (Rec V))
    (hm : MatchBranch S m mt) (r : Rec V) (g : G) :
    ev S (.node (.merge p) [y, m]) r g = specMerge S p mt none (ev S y r g) := by
  show forRows _ _ _ = _
  simp only [specMerge]
  apply forRows_congr
  intro row g'
  rw [hm row g']
  split <;> rfl

theorem forRows_path (q : Nat) (T : Res V G) :
    forRows (S.step (.pathBuilder q)) T.1 T.2 = (T.1.map (S.path q), T.2) := by
  rw [forRows_ro (S.step (.pathBuilder q)) (fun row _ => [S.path q row]) (fun _ _ => rfl)]
  congr 1
  induction T.1 with
  | nil => rfl
  | cons x xs ih => simp [ih]

/-- **MERGE is correct at the in-loop insertion point**, with or without a
named path: the walk steps over `PathBuilder` (mod.rs:2725) and the input lands
as `Merge`'s child 0. -/
theorem merge_loop_correct (p : Nat) (m : Plan) (path : Option Nat) (mt : Rec V → G → List (Rec V))
    (hm : MatchBranch S m mt) (ok : Plan → Prop) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (planMerge p m path) (slotLoop (planMerge p m path)) q y) r g =
      specMerge S p mt path (ev S y r g) := by
  cases path with
  | none =>
    rw [show slotLoop (planMerge p m none) = slotWith walkLoop (.node (.merge p) [m]) from rfl,
      show planMerge p m none = .node (.merge p) [m] from rfl,
      refill_corr walkLoop goodWalk_loop (.merge p) m q y rfl (by simp) rfl rfl]
    exact ev_merge_two S p m y mt hm r g
  | some k =>
    have hs := slotWith_step walkLoop (.pathBuilder k) (steps_loop _ rfl) (.node (.merge p) [m])
    rw [show slotLoop (planMerge p m (some k)) =
        slotWith walkLoop (.node (.pathBuilder k) [.node (.merge p) [m]]) from rfl, hs,
      show planMerge p m (some k) = .node (.pathBuilder k) [.node (.merge p) [m]] from rfl,
      refill_step _ _ _ q y (slotWith_valid _ walkLoop_valid _),
      refill_corr walkLoop goodWalk_loop (.merge p) m q y rfl (by simp) rfl rfl]
    show forRows (S.step (.pathBuilder k)) (ev S (.node (.merge p) [y, m]) r g).1
      (ev S (.node (.merge p) [y, m]) r g).2 = _
    rw [forRows_path, ev_merge_two S p m y mt hm r g]
    rfl

/-- MERGE without a path is also correct as the last clause. -/
theorem merge_first_correct (p : Nat) (m : Plan) (mt : Rec V → G → List (Rec V))
    (hm : MatchBranch S m mt) (ok : Plan → Prop) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (planMerge p m none) (slotFirst (planMerge p m none)) q y) r g =
      specMerge S p mt none (ev S y r g) := by
  rw [show slotFirst (planMerge p m none) = slotWith walkFirst (.node (.merge p) [m]) from rfl,
    show planMerge p m none = .node (.merge p) [m] from rfl,
    refill_corr walkFirst goodWalk_first (.merge p) m q y rfl (by simp) rfl rfl]
  exact ev_merge_two S p m y mt hm r g

/-- **Bug (confirmed): MERGE with a named path as the last clause.** The first
walk (mod.rs:2638-2644) stops at `PathBuilder`, so the previous clause becomes
`PathBuilder`'s child 0 beside `Merge`: -/
theorem merge_path_last_misplaced (p k : Nat) (m q y : Plan) :
    refill (planMerge p m (some k)) (slotFirst (planMerge p m (some k))) q y =
      .node (.pathBuilder k) [y, .node (.merge p) [m]] := by
  have hslot : slotFirst (planMerge p m (some k)) = [] := by
    show slotWith walkFirst (.node (.pathBuilder k) [.node (.merge p) [m]]) = []
    unfold slotWith
    simp only [walkFirst, firstPass, isApply, Bool.false_or, Bool.and_false, Bool.false_eq_true,
      ite_false, Plan.get, List.nil_append, projDescend]
    exact descendClause_nil _ rfl
  rw [hslot, show planMerge p m (some k) = .node (.pathBuilder k) [.node (.merge p) [m]] from rfl,
    refill, insertStep_nonCP _ [] q _ _ rfl (by simp)]
  rfl

/-- …and `PathBuilder` only runs child 0 (runtime.rs:1111, `children_to_recurse`
`_` arm): the `Merge` never executes; the paths are built from the previous
clause's rows, which do not bind the pattern (Rust: "Variable _anon_0 not
found"). -/
theorem merge_path_last_skips_merge (p k : Nat) (m q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (planMerge p m (some k)) (slotFirst (planMerge p m (some k))) q y) r g =
      ((ev S y r g).1.map (S.path k), (ev S y r g).2) := by
  rw [merge_path_last_misplaced]
  show forRows (S.step (.pathBuilder k)) (ev S y r g).1 (ev S y r g).2 = _
  exact forRows_path S k _

/-- A sub-plan implementing a sub-query on one argument row. -/
def Body (b : Plan) (sb : Rec V → G → Res V G) : Prop := ∀ row g, ev S b row g = sb row g

/-- openCypher FOREACH: per record, run the body once per list element (for
its effects only); the record passes through unchanged. -/
def specForEach (e : Nat) (x : Var) (sb : Rec V → G → Res V G) (T : Res V G) : Res V G :=
  forRows (fun row g => ([row], forItems (fun v g => (sb (row.set x v) g).2) (S.items e row g) g))
    T.1 T.2

theorem forItems_congr {A : Type} (f f' : A → G → G) (h : ∀ a g, f a g = f' a g) (l : List A) (g : G) :
    forItems f l g = forItems f' l g := by
  induction l generalizing g with
  | nil => rfl
  | cons a as ih => simp [forItems, h, ih]

/-- FOREACH (mod.rs:3301-3364, no comprehension in the list): `ForEach(body)`,
the input becomes child 0. -/
theorem foreach_correct (w : Plan → Path) (hw : GoodWalk w) (e : Nat) (x : Var) (b : Plan)
    (sb : Rec V → G → Res V G) (hb : Body S b sb) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node (.forEach e x) [b]) (slotWith w (.node (.forEach e x) [b])) q y) r g =
      specForEach S e x sb (ev S y r g) := by
  rw [refill_corr w hw (.forEach e x) b q y rfl (by simp) rfl (by simp [descendOne, clauseMin])]
  show forRows _ _ _ = _
  apply forRows_congr
  intro row g'
  congr 1
  exact forItems_congr _ _ (fun v g => by rw [hb]) _ _

/-- openCypher `CALL { … RETURN … }`: per record, the body's records (which
extend it). -/
def specCall (sb : Rec V → G → Res V G) (T : Res V G) : Res V G := forRows sb T.1 T.2

/-- Returning CALL {} (mod.rs:3215-3281): `Apply(body)` — with the body's root
Project renamed in place, or a remapping Project on top: either way a plan
`b` implementing the body. -/
theorem call_correct (w : Plan → Path) (hw : GoodWalk w) (b : Plan) (sb : Rec V → G → Res V G)
    (hb : Body S b sb) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node .apply [b]) (slotWith w (.node .apply [b])) q y) r g =
      specCall sb (ev S y r g) := by
  rw [refill_corr w hw .apply b q y rfl (by simp) rfl rfl]
  show forRows _ _ _ = _
  exact forRows_congr _ _ (fun row g => hb row g) _ _

/-- openCypher unit subquery `CALL { … }` (no RETURN): per record, run the
body for its effects; the record passes through once. -/
def specUnitCall (sb : Rec V → G → Res V G) (T : Res V G) : Res V G :=
  forRows (fun row g => ([row], (sb row g).2)) T.1 T.2

/-- The keyless, aggregation-free `Aggregate` the planner puts over a unit
body (mod.rs:3287-3298) yields exactly one row, which Apply merges over the
input row: the input row. -/
def KeylessOne (p0 : Nat) : Prop := ∀ r rows, S.agg p0 r rows = [r]

theorem unit_call_correct (w : Plan → Path) (hw : GoodWalk w) (p0 : Nat) (hk : KeylessOne S p0)
    (b : Plan) (sb : Rec V → G → Res V G) (hb : Body S b sb) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node .apply [.node (.aggregate p0) [b]])
      (slotWith w (.node .apply [.node (.aggregate p0) [b]])) q y) r g =
      specUnitCall sb (ev S y r g) := by
  rw [refill_corr w hw .apply _ q y rfl (by simp) rfl rfl]
  show forRows _ _ _ = _
  apply forRows_congr
  intro row g'
  show S.bag (.aggregate p0) row (ev S b row g').1 (ev S b row g').2 = _
  simp only [Sem.bag]; rw [hk, hb row g']

/-- openCypher OPTIONAL MATCH: per record, the matches, or the record padded
with nulls for the pattern's new variables. -/
def specOptional (vs : List Var) (mt : Rec V → G → List (Rec V)) (T : Res V G) : Res V G :=
  forRows (fun row g => (if (mt row g).isEmpty then [S.pad vs row] else mt row g, g)) T.1 T.2

/-- OPTIONAL MATCH with no bound pattern variable (mod.rs:2959): `Optional(m)`. -/
theorem optional_correct (w : Plan → Path) (hw : GoodWalk w) (vs : List Var) (m : Plan)
    (mt : Rec V → G → List (Rec V)) (hm : MatchBranch S m mt) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node (.optional vs) [m]) (slotWith w (.node (.optional vs) [m])) q y) r g =
      specOptional S vs mt (ev S y r g) := by
  rw [refill_corr w hw (.optional vs) m q y rfl (by simp) rfl rfl]
  show forRows _ _ _ = _
  apply forRows_congr
  intro row g'
  show (let b := ev S m row g'; (if b.1.isEmpty then [S.pad vs row] else b.1, b.2)) = _
  simp only; rw [hm row g']

/-- OPTIONAL MATCH with a bound pattern variable (mod.rs:2957):
`Apply(Optional(m))`. Same meaning. -/
theorem optional_apply_correct (w : Plan → Path) (hw : GoodWalk w) (vs : List Var) (m : Plan)
    (mt : Rec V → G → List (Rec V)) (hm : MatchBranch S m mt) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node .apply [.node (.optional vs) [m]])
      (slotWith w (.node .apply [.node (.optional vs) [m]])) q y) r g =
      specOptional S vs mt (ev S y r g) := by
  rw [refill_corr w hw .apply _ q y rfl (by simp) rfl rfl]
  show forRows _ _ _ = _
  apply forRows_congr
  intro row g'
  show (let b := ev S m row g'; (if b.1.isEmpty then [S.pad vs row] else b.1, b.2)) = _
  simp only; rw [hm row g']

/-- openCypher MATCH over bound variables only: per record, its matches. -/
def specMatch (mt : Rec V → G → List (Rec V)) (T : Res V G) : Res V G :=
  (T.1.flatMap fun row => mt row T.2, T.2)

/-- MATCH whose variables are all bound (mod.rs:2970-2973): `Apply(m)`. -/
theorem match_bound_correct (w : Plan → Path) (hw : GoodWalk w) (m : Plan)
    (mt : Rec V → G → List (Rec V)) (hm : MatchBranch S m mt) (q y : Plan) (r : Rec V) (g : G) :
    ev S (refill (.node .apply [m]) (slotWith w (.node .apply [m])) q y) r g =
      specMatch mt (ev S y r g) := by
  rw [refill_corr w hw .apply m q y rfl (by simp) rfl rfl]
  show forRows _ _ _ = _
  exact forRows_ro _ mt hm _ _

/-- UNION / UNION ALL (mod.rs:3197-3214): every branch on the argument row,
concatenated; `Distinct` on top for UNION. -/
def planUnion (bs : List Plan) (all : Bool) : Plan :=
  if all then .node .union bs else .node .distinct [.node .union bs]

def specUnion (sbs : List (Rec V → G → Res V G)) (all : Bool) (r : Rec V) (g : G) : Res V G :=
  let cat := sbs.foldl (fun (acc : Res V G) sb => ((acc.1 ++ (sb r acc.2).1), (sb r acc.2).2)) ([], g)
  if all then cat else (S.dedup cat.1, cat.2)

theorem evUnion_eq (bs : List (Plan × (Rec V → G → Res V G)))
    (h : ∀ b ∈ bs, Body S b.1 b.2) (r : Rec V) (acc : Res V G) :
    ((acc.1 ++ (evUnion S (bs.map Prod.fst) r acc.2).1), (evUnion S (bs.map Prod.fst) r acc.2).2) =
      (bs.map Prod.snd).foldl (fun (acc : Res V G) sb => ((acc.1 ++ (sb r acc.2).1), (sb r acc.2).2)) acc := by
  induction bs generalizing acc with
  | nil => simp [evUnion]
  | cons b bs ih =>
    simp only [List.map_cons, evUnion, List.foldl_cons]
    rw [← ih (fun b hb => h b (List.mem_cons_of_mem _ hb))]
    have hb := h b List.mem_cons_self
    simp only [Body] at hb
    simp [hb, List.append_assoc]

theorem union_correct (bs : List (Plan × (Rec V → G → Res V G)))
    (h : ∀ b ∈ bs, Body S b.1 b.2) (all : Bool) (r : Rec V) (g : G) :
    ev S (planUnion (bs.map Prod.fst) all) r g = specUnion S (bs.map Prod.snd) all r g := by
  have e := evUnion_eq S bs h r ([], g)
  simp only [List.nil_append] at e
  cases all
  · show S.bag .distinct r (evUnion S _ r g).1 (evUnion S _ r g).2 = _
    simp only [specUnion, Sem.bag]
    rw [← e]; simp
  · show evUnion S _ r g = _
    simp only [specUnion]
    rw [← e]; simp

/-! ### WITH / RETURN -/

def projOp (c : ProjC) : Op := if c.agg then .aggregate c.p else .project c.p

def projBase (c : ProjC) : Plan := .node (projOp c) (if c.write then [.node .commit []] else [])

def projSem (c : ProjC) (r : Rec V) (T : Res V G) : Res V G :=
  if c.agg then (S.agg c.p r T.1, T.2) else (T.1.map fun row => S.proj c.p row T.2, T.2)

theorem projOp_ne (c : ProjC) : projOp c ≠ .cartesian := by
  unfold projOp; split <;> simp

theorem ev_projOp (c : ProjC) (z : Plan) (r : Rec V) (g : G) :
    ev S (.node (projOp c) [z]) r g = projSem S c r (ev S z r g) := by
  unfold projOp projSem
  cases c.agg
  · show forRows (S.step (.project c.p)) _ _ = _
    simpa using forRows_project S c.p _ _
  · rfl

theorem implements_projBase (w : Plan → Path) (hw : GoodWalk w) (c : ProjC) (ok : Plan → Prop) :
    Implements S w ok (projBase c) (projSem S c) := by
  intro q y r g _
  unfold projBase
  cases hwr : c.write
  · simp only [Bool.false_eq_true, ite_false]
    have hproj : projDescend (.node (projOp c) []) = [] := by
      unfold projOp; split <;> rfl
    rw [implements_leaf S w hw.leaf (projOp c) (projOp_ne c) hproj (descendClause_leaf _) _
      (fun z r g => ev_projOp S c z r g) ok q y r g ‹_›]
  · simp only [ite_true]
    have hslot : slotWith w (.node (projOp c) [.node .commit []]) = [0] := by
      unfold slotWith
      rw [hw.one _ _ (by unfold projOp; split <;> rfl)]
      have hpd : projDescend (.node (projOp c) [.node .commit []]) = [0] := by
        unfold projOp; split <;> rfl
      simp only [Plan.get, List.nil_append, hpd, List.getElem?_cons_zero]
      rw [descendClause_nil _ rfl]; rfl
    rw [hslot, refill, insertStep_nonCP _ [0] q .commit [] rfl (by simp)]
    show ev S (.node (projOp c) [.node .commit [y]]) r g = _
    rw [ev_projOp]
    congr 1
    show forRows (S.step .commit) _ _ = _
    exact forRows_id S .commit (fun _ _ => rfl) _ _

theorem implements_wrapOpt (w : Plan → Path) (hw : GoodWalk w) (mk : Nat → Op)
    (hs : ∀ k, Steps w (mk k)) (WS : Nat → Rec V → Res V G → Res V G)
    (hsem : ∀ k, UnarySem S (mk k) (WS k)) (ok : Plan → Prop) (o : Option Nat) (p : Plan)
    (F : Rec V → Res V G → Res V G) (hp : Implements S w ok p F) :
    Implements S w ok (wrapOpt mk o p)
      (fun r T => match o with | some k => WS k r (F r T) | none => F r T) := by
  cases o with
  | none => exact hp
  | some k => exact implements_step S w hw.valid (mk k) (hs k) (WS k) (hsem k) ok p F hp

theorem forRows_filter' (k : Nat) (T : Res V G) :
    forRows (S.step (.filter k)) T.1 T.2 = (T.1.filter fun row => S.test k row T.2, T.2) :=
  forRows_filter S k T.1 T.2

/-- **WITH / RETURN are correct** (either walk), given that DISTINCT over an
aggregation is the identity. -/
theorem project_correct (w : Plan → Path) (hw : GoodWalk w) (c : ProjC) (hc : AggDistinct S c)
    (ok : Plan → Prop) (q y : Plan) (r : Rec V) (g : G) (hq : ok q) :
    ev S (refill (planProject c) (slotWith w (planProject c)) q y) r g =
      specProject S c r (ev S y r g) := by
  have h0 := implements_projBase S w hw c ok
  have h1 : Implements S w ok (if c.distinct && !c.agg then Plan.node .distinct [projBase c] else projBase c)
      (fun r T => if c.distinct && !c.agg then bagSem S .distinct r (projSem S c r T) else projSem S c r T) := by
    split
    · exact implements_step S w hw.valid .distinct hw.distinct _ (unary_bag S _ rfl) ok _ _ h0
    · exact h0
  have h2 := implements_wrapOpt S w hw .sort hw.sort (fun k => bagSem S (.sort k))
    (fun k => unary_bag S _ rfl) ok c.order _ _ h1
  have h3 := implements_wrapOpt S w hw .skip hw.skip (fun k => bagSem S (.skip k))
    (fun k => unary_bag S _ rfl) ok c.skip _ _ h2
  have h4 := implements_wrapOpt S w hw .limit hw.limit (fun k => bagSem S (.limit k))
    (fun k => unary_bag S _ rfl) ok c.limit _ _ h3
  have h5 := implements_wrapOpt S w hw .filter hw.filter.2 (fun k => rowSem S (.filter k))
    (fun k => unary_row S _ rfl (by simp [Op.isCorr])) ok c.filter _ _ h4
  have e : planProject c = wrapOpt .filter c.filter (wrapOpt .limit c.limit (wrapOpt .skip c.skip
      (wrapOpt .sort c.order (if c.distinct && !c.agg then Plan.node .distinct [projBase c]
        else projBase c)))) := by
    unfold planProject projBase projOp; rfl
  rw [e, h5 q y r g hq]
  have sSort : ∀ (X : Res V G), (match c.order with | some k => bagSem S (.sort k) r X | none => X) =
      (optApply S.sort c.order X.1, X.2) := by intro X; cases c.order <;> rfl
  have sSkip : ∀ (X : Res V G), (match c.skip with | some k => bagSem S (.skip k) r X | none => X) =
      (optApply (fun n rs => rs.drop n) c.skip X.1, X.2) := by intro X; cases c.skip <;> rfl
  have sLimit : ∀ (X : Res V G), (match c.limit with | some k => bagSem S (.limit k) r X | none => X) =
      (optApply (fun n rs => rs.take n) c.limit X.1, X.2) := by intro X; cases c.limit <;> rfl
  have sFilter : ∀ (X : Res V G), (match c.filter with | some k => rowSem S (.filter k) r X | none => X) =
      (optApply (fun φ rs => rs.filter fun row => S.test φ row X.2) c.filter X.1, X.2) := by
    intro X; cases c.filter
    · rfl
    · exact forRows_filter' S _ X
  have sBase : (if c.distinct && !c.agg then bagSem S .distinct r (projSem S c r (ev S y r g))
      else projSem S c r (ev S y r g)) =
      (let rows1 := if c.agg then S.agg c.p r (ev S y r g).1
          else (ev S y r g).1.map fun row => S.proj c.p row (ev S y r g).2
       if c.distinct then S.dedup rows1 else rows1, (ev S y r g).2) := by
    unfold projSem bagSem
    cases ha : c.agg <;> cases hd : c.distinct <;> simp [Sem.bag, hc r]
  simp only [sSort, sSkip, sLimit, sFilter, sBase]
  rfl

end PlannerBuild