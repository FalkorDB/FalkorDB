import Columnar.Arith
import Columnar.Row

/-
# Remaining `vector_expr.rs` lanes: NOT, CASE, function folding, variables, filter mask

| here | there |
| --- | --- |
| `notRow`, `negate` | per-row `ExprIR::Not` (`eval.rs:699-708`) / `negate_bools` (`vector_expr.rs:854-873`) |
| `caseMatch`, `caseRow` | `ExprEval::eval_case` (`eval.rs:485-518`) |
| `caseArm`, `caseArms`, `caseCol` | `VectorEval::eval_case` (`vector_expr.rs:475-549`): arm-by-arm narrowing of `unclaimed`, scatter into `out` |
| `foldCall` | `VectorEval::eval_function` (`vector_expr.rs:555-620`) |
| `evalVariable` | `VectorEval::eval_variable` (`vector_expr.rs:310-329`) |
| `filterPass` | `FilterOp` verdict → passing rows (`filter.rs:69-108`) |
-/

namespace Columnar

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-! ## NOT -/

def notRow : V F → Except String (V F)
  | .bool b => .ok (.bool !b)
  | .null => .ok .null
  | _ => .error "Type mismatch: expected Boolean or Null"

/-- `negate_bools`: one pass, first bad row errors; mask/null bit per row. -/
def negate (child : EC F) (len : Nat) : Except String (EC F) :=
  ((List.range len).mapM (fun i => (match child.get i with
    | .bool b => .ok (!b, false)
    | .null => .ok (false, true)
    | _ => .error "Type mismatch: expected Boolean or Null" : Except String (Bool × Bool)))).map
    (fun ps => .bools (ps.map Prod.fst) (ps.map Prod.snd))

theorem mapM_map_ok {α β γ : Type} (f : α → Except String β) (g : β → γ) :
    ∀ (l : List α), l.mapM (fun a => (f a).map g) = (l.mapM f).map (List.map g)
  | [] => rfl
  | a :: l => by
    rw [List.mapM_cons, List.mapM_cons, mapM_map_ok f g l]
    cases f a <;> cases l.mapM f <;> rfl

/-- **`NOT` column = per-row `NOT`**, values and first error alike. -/
theorem negate_agree (child : EC F) (len : Nat) :
    (negate child len).map (fun c => (List.range len).map c.get) =
      (List.range len).mapM (fun i => notRow (child.get i)) := by
  let step : Nat → Except String (Bool × Bool) := fun i => match child.get i with
    | .bool b => .ok (!b, false)
    | .null => .ok (false, true)
    | _ => .error "Type mismatch: expected Boolean or Null"
  let toV : Bool × Bool → V F := fun p => if p.2 then .null else .bool p.1
  have hstep : ∀ i, notRow (child.get i) = (step i).map toV := by
    intro i; simp only [step, toV]; cases child.get i <;> rfl
  simp only [hstep]
  rw [mapM_map_ok step toV]
  unfold negate
  cases hm : (List.range len).mapM step with
  | error e => rfl
  | ok ps =>
    have hl := mapM_length hm
    simp only [List.length_range] at hl
    show Except.ok ((List.range len).map (EC.bools (ps.map Prod.fst) (ps.map Prod.snd)).get) =
      Except.ok (ps.map toV)
    congr 1
    apply List.ext_getElem?
    intro k
    rcases Nat.lt_or_ge k len with h | h
    · have hk : k < ps.length := by omega
      simp [List.getElem?_range h, EC.get, Bits.isNull, List.getElem?_eq_getElem hk, toV]
    · rw [List.getElem?_eq_none (by simpa using h), List.getElem?_eq_none (by simp; omega)]

/-! ## CASE -/

/-- The arm test, shared by both paths: value form `compare_values(when, subject, Eq) == Some(true)`
(columnar) — per-row `when.compare_value(subject) == (Equal, None)`; searched form: not
`false`/`null`. -/
def caseMatch (other : V F → V F → Ordering × Flag) (subject : Option (V F)) (w : V F) : Bool :=
  match subject with
  | some s => compareValues other w s .eq == some true
  | none => match w with
    | .bool false | .null => false
    | _ => true

/-- The two value-form tests are the same predicate. -/
theorem caseMatch_eq_compare_value (other : V F → V F → Ordering × Flag) (s w : V F) :
    caseMatch other (some s) w = (compareValue other w s == (.eq, .none)) := by
  simp only [caseMatch, compareValues]
  cases compareValue other w s with
  | mk o fl => cases fl <;> cases o <;> rfl

/-- Per-row CASE with pure (row-indexed) sub-expressions. -/
def caseRow (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F))
    (arms : List ((Nat → V F) × (Nat → V F))) (els : Nat → V F) (r : Nat) : V F :=
  match arms with
  | [] => els r
  | (w, t) :: rest =>
    if caseMatch other (subject.map (· r)) (w r) then t r else caseRow other subject rest els r

/-- One columnar arm: over the `unclaimed` positions, claim the matching ones
and scatter their `THEN` values into `out`. -/
def caseArm (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F)) (rows : List Nat)
    (arm : (Nat → V F) × (Nat → V F)) (st : (Nat → V F) × List Nat) : (Nat → V F) × List Nat :=
  let (out, unclaimed) := st
  let row := fun pos => rows[pos]?.getD 0
  let hit := fun pos => caseMatch other (subject.map (· (row pos))) (arm.1 (row pos))
  (fun pos => if pos ∈ unclaimed.filter hit then arm.2 (row pos) else out pos,
   unclaimed.filter (fun pos => !hit pos))

def caseArms (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F)) (rows : List Nat) :
    List ((Nat → V F) × (Nat → V F)) → (Nat → V F) × List Nat → (Nat → V F) × List Nat
  | [], st => st
  | a :: as, st => caseArms other subject rows as (caseArm other subject rows a st)

/-- `eval_case`: all arms, then ELSE over what is still unclaimed. -/
def caseCol (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F))
    (arms : List ((Nat → V F) × (Nat → V F))) (els : Nat → V F) (rows : List Nat) : List (V F) :=
  let st := caseArms other subject rows arms (fun _ => .null, List.range rows.length)
  (List.range rows.length).map (fun pos =>
    if pos ∈ st.2 then els (rows[pos]?.getD 0) else st.1 pos)

theorem caseArms_spec (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F))
    (rows : List Nat) (els : Nat → V F) :
    ∀ (arms : List ((Nat → V F) × (Nat → V F))) (out : Nat → V F) (un : List Nat) (pos : Nat),
      let st := caseArms other subject rows arms (out, un)
      (if pos ∈ st.2 then els (rows[pos]?.getD 0) else st.1 pos) =
        (if pos ∈ un then caseRow other subject arms els (rows[pos]?.getD 0) else out pos)
  | [], out, un, pos => by
    by_cases hu : pos ∈ un <;> simp [caseArms, caseRow, hu]
  | (w, t) :: rest, out, un, pos => by
    simp only [caseArms]
    refine Eq.trans (caseArms_spec other subject rows els rest _ _ pos) ?_
    simp only [caseArm, List.mem_filter, caseRow]
    by_cases hu : pos ∈ un <;>
      by_cases hm : caseMatch other (subject.map (· (rows[pos]?.getD 0))) (w (rows[pos]?.getD 0)) <;>
      simp [hu, hm]

/-- **Columnar CASE = per-row CASE** (pure sub-expressions): every row gets the
`THEN` of its first matching arm, or the `ELSE`. -/
theorem caseCol_agree (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F))
    (arms : List ((Nat → V F) × (Nat → V F))) (els : Nat → V F) (rows : List Nat) :
    caseCol other subject arms els rows = rows.map (caseRow other subject arms els) := by
  apply List.ext_getElem?
  intro k
  unfold caseCol
  rcases Nat.lt_or_ge k rows.length with h | h
  · simp only [List.getElem?_map, List.getElem?_range h, Option.map_some]
    have := caseArms_spec other subject rows els arms (fun _ => .null) (List.range rows.length) k
    simp only at this
    rw [this]
    simp [h, List.getElem?_eq_getElem h]
  · rw [List.getElem?_eq_none (by simpa using h), List.getElem?_eq_none (by simpa using h)]

/-- Laziness: a `THEN` is only ever read at a row whose `WHEN` matched (so
`CASE WHEN n.d <> 0 THEN 1 / n.d ELSE 0 END` cannot divide by zero). -/
theorem caseArm_then_only_on_hits (other : V F → V F → Ordering × Flag) (subject : Option (Nat → V F))
    (rows : List Nat) (arm : (Nat → V F) × (Nat → V F)) (out : Nat → V F) (un : List Nat) (pos : Nat)
    (hne : (caseArm other subject rows arm (out, un)).1 pos ≠ out pos) :
    pos ∈ un ∧ caseMatch other (subject.map (· (rows[pos]?.getD 0))) (arm.1 (rows[pos]?.getD 0)) = true := by
  simp only [caseArm] at hne
  split at hne
  · rename_i h; simp only [List.mem_filter] at h; exact h
  · exact absurd rfl hne

/-! ## Scalar function folding -/

/-- `eval_function`: all-`Scalar` arguments of a deterministic call on a
non-empty batch are folded to one call. -/
def foldCall (f : List (V F) → Except String (V F)) (deterministic : Bool) (args : List (EC F)) (len : Nat) :
    Except String (EC F) :=
  if len > 0 ∧ deterministic ∧ args ≠ [] ∧ args.all (fun a => match a with | .scalar _ => true | _ => false) then
    (f (args.map (·.get 0))).map EC.scalar
  else
    ((List.range len).mapM (fun i => f (args.map (·.get i)))).map EC.values

theorem scalar_get_const (a : EC F) (h : (match a with | .scalar _ => true | _ => false) = true) (i j : Nat) :
    a.get i = a.get j := by
  cases a <;> simp at h; rfl

/-- **Folding is invisible** when the function is a pure function of its
arguments: same values at every row, and the same (first-row) error. -/
theorem foldCall_agree (f : List (V F) → Except String (V F)) (det : Bool) (args : List (EC F)) (len : Nat) :
    (foldCall f det args len).map (fun c => (List.range len).map c.get) =
      (List.range len).mapM (fun i => f (args.map (·.get i))) := by
  unfold foldCall
  split
  · rename_i h
    obtain ⟨hl, -, -, hall⟩ := h
    have hconst : ∀ i, args.map (·.get i) = args.map (·.get 0) := by
      intro i
      apply List.map_congr_left
      intro a ha
      exact scalar_get_const a (List.all_eq_true.mp hall a ha) i 0
    simp only [hconst]
    obtain ⟨n, rfl⟩ : ∃ n, len = n + 1 := ⟨len - 1, by omega⟩
    cases hf : f (args.map (·.get 0)) with
    | error e =>
      rw [List.range_succ_eq_map, List.mapM_cons]; rfl
    | ok v =>
      rw [mapM_ok (f := fun _ => Except.ok v) (g := fun _ => v) (fun _ _ => rfl)]
      rfl
  · cases hm : (List.range len).mapM (fun i => f (args.map (·.get i))) with
    | error e => rfl
    | ok vs =>
      have hl := mapM_length hm
      simp only [List.length_range] at hl
      show Except.ok ((List.range len).map (EC.values vs).get) = Except.ok vs
      rw [← hl, range_map_values_get]

/-! ## Variables and the filter mask -/

/-- `eval_variable`: typed lanes gathered with an empty bitmap, anything else as
values (`none` = the gather panics). -/
def evalVariable (b : Batch F) (i : Nat) (rows : List Nat) : Option (EC F) :=
  match b.column i with
  | .ints d => (gatherL d rows).map (fun g => .ints g (Bits.none rows.length))
  | .floats d => (gatherL d rows).map (fun g => .floats g (Bits.none rows.length))
  | c => (mapO (fun r => c.get? r) rows).map .values

/-- **`eval_variable` reads what the per-row view reads**, never panics, for
every active row and every slot in the batch's column space. Beyond the column
space the per-row view answers `None` ("Variable not found") while the columnar
lane reads `Null` (`evalVariable_beyond`). -/
theorem evalVariable_agree {b : Batch F} (hw : b.WF) (i : Nat) (rows : List Nat)
    (hr : ∀ r ∈ rows, r ∈ b.active) (hi : i < b.cols.length) :
    ∃ c, evalVariable b i rows = some c ∧ ∀ k (hk : k < rows.length), some (c.get k) = viewAt b rows[k] i := by
  have hlt : ∀ r ∈ rows, r < b.len := fun r h => Batch.active_lt hw r (hr r h)
  have hview : ∀ r, r < b.len → (b.column i).isUnbound = false →
      viewAt b r i = (b.column i).get? r := by
    intro r hr' hu
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp
      (Column.get?_isSome hu (by rw [Batch.column_len hw hu]; exact hr'))
    simp [viewAt, Batch.valueAt, hu, hv]
  unfold evalVariable
  cases hc : b.column i with
  | nodeIds d =>
      have hu : (b.column i).isUnbound = false := by rw [hc]; rfl
      have hsome : ∀ r ∈ rows, ((b.column i).get? r).isSome := fun r h =>
        Column.get?_isSome hu (by rw [Batch.column_len hw hu]; exact hlt r h)
      rw [hc] at hsome
      obtain ⟨vs, hvs⟩ := Option.isSome_iff_exists.mp (mapO_isSome hsome)
      refine ⟨_, by show Option.map EC.values _ = _; rw [hvs]; rfl, fun k hk => ?_⟩
      have := mapO_getElem? hvs k
      rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
      rw [hview _ (hlt _ (List.getElem_mem hk)) hu, hc]
      have hvk : k < vs.length := by rw [mapO_length hvs]; exact hk
      rw [List.getElem?_eq_getElem hvk] at this
      simp only [EC.get, List.getElem?_eq_getElem hvk, Option.getD_some]; exact this
  | relIds d =>
      have hu : (b.column i).isUnbound = false := by rw [hc]; rfl
      have hsome : ∀ r ∈ rows, ((b.column i).get? r).isSome := fun r h =>
        Column.get?_isSome hu (by rw [Batch.column_len hw hu]; exact hlt r h)
      rw [hc] at hsome
      obtain ⟨vs, hvs⟩ := Option.isSome_iff_exists.mp (mapO_isSome hsome)
      refine ⟨_, by show Option.map EC.values _ = _; rw [hvs]; rfl, fun k hk => ?_⟩
      have := mapO_getElem? hvs k
      rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
      rw [hview _ (hlt _ (List.getElem_mem hk)) hu, hc]
      have hvk : k < vs.length := by rw [mapO_length hvs]; exact hk
      rw [List.getElem?_eq_getElem hvk] at this
      simp only [EC.get, List.getElem?_eq_getElem hvk, Option.getD_some]; exact this
  | ints d =>
      have hl : d.length = b.len := by
        have := Batch.column_len hw (i := i) (by rw [hc]; rfl); rw [hc] at this; exact this
      obtain ⟨g, hg⟩ := Option.isSome_iff_exists.mp (gatherL_isSome_iff.mpr (fun r h => hl ▸ hlt r h))
      refine ⟨_, by show Option.map _ (gatherL d rows) = _; rw [hg]; rfl, fun k hk => ?_⟩
      rw [hview _ (hlt _ (List.getElem_mem hk)) (by rw [hc]; rfl), hc]
      have := gatherL_getElem? hg k
      rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
      have hgk : k < g.length := by rw [gatherL_length hg]; exact hk
      rw [List.getElem?_eq_getElem hgk] at this
      simp [EC.get, Bits.isNull_none, List.getElem?_eq_getElem hgk, Column.get?, ← this]
  | floats d =>
      have hl : d.length = b.len := by
        have := Batch.column_len hw (i := i) (by rw [hc]; rfl); rw [hc] at this; exact this
      obtain ⟨g, hg⟩ := Option.isSome_iff_exists.mp (gatherL_isSome_iff.mpr (fun r h => hl ▸ hlt r h))
      refine ⟨_, by show Option.map _ (gatherL d rows) = _; rw [hg]; rfl, fun k hk => ?_⟩
      rw [hview _ (hlt _ (List.getElem_mem hk)) (by rw [hc]; rfl), hc]
      have := gatherL_getElem? hg k
      rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
      have hgk : k < g.length := by rw [gatherL_length hg]; exact hk
      rw [List.getElem?_eq_getElem hgk] at this
      simp [EC.get, Bits.isNull_none, List.getElem?_eq_getElem hgk, Column.get?, ← this]
  | values d =>
      have hu : (b.column i).isUnbound = false := by rw [hc]; rfl
      have hsome : ∀ r ∈ rows, ((b.column i).get? r).isSome := fun r h =>
        Column.get?_isSome hu (by rw [Batch.column_len hw hu]; exact hlt r h)
      rw [hc] at hsome
      obtain ⟨vs, hvs⟩ := Option.isSome_iff_exists.mp (mapO_isSome hsome)
      refine ⟨_, by show Option.map EC.values _ = _; rw [hvs]; rfl, fun k hk => ?_⟩
      have := mapO_getElem? hvs k
      rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
      rw [hview _ (hlt _ (List.getElem_mem hk)) hu, hc]
      have hvk : k < vs.length := by rw [mapO_length hvs]; exact hk
      rw [List.getElem?_eq_getElem hvk] at this
      simp only [EC.get, List.getElem?_eq_getElem hvk, Option.getD_some]; exact this
  | unbound =>
    have hsome : ∀ r ∈ rows, ((Column.unbound : Column F).get? r).isSome := fun _ _ => rfl
    obtain ⟨vs, hvs⟩ := Option.isSome_iff_exists.mp (mapO_isSome hsome)
    refine ⟨_, by show Option.map EC.values _ = _; rw [hvs]; rfl, fun k hk => ?_⟩
    have := mapO_getElem? hvs k
    rw [List.getElem?_eq_getElem hk, Option.bind_some] at this
    have hvk : k < vs.length := by rw [mapO_length hvs]; exact hk
    rw [List.getElem?_eq_getElem hvk] at this
    simp [EC.get, List.getElem?_eq_getElem hvk, viewAt, Batch.valueAt, hc, Column.isUnbound, hi]
    simpa [Column.get?] using this

theorem evalVariable_beyond (b : Batch F) (i : Nat) (hi : b.cols.length ≤ i) (r : Nat) :
    evalVariable b i [r] = some (.values [.null]) ∧ viewAt b r i = none := by
  have hc : b.column i = .unbound := by simp [Batch.column, List.getElem?_eq_none hi]
  simp [evalVariable, hc, mapO, Column.get?, viewAt, Batch.valueAt, Column.isUnbound, Nat.not_lt.mpr hi]

/-- The Filter verdict for a `Bools` column: keep row `rows[k]` iff the mask bit
is set and the row is not null. -/
def filterPass (rows : List Nat) (mask : List Bool) (nulls : Bits) : List Nat :=
  ((List.range rows.length).filter (fun k => mask[k]?.getD false && !nulls.isNull k)).map
    (fun k => rows[k]?.getD 0)

/-- **Filter keeps exactly the rows whose predicate is `true`** (`false` and
`null` drop), matching the per-row filter on `ExprColumn::get`. -/
theorem filterPass_agree (rows : List Nat) (mask : List Bool) (nulls : Bits) :
    filterPass rows mask nulls =
      ((List.range rows.length).filter (fun k => match (EC.bools mask nulls : EC F).get k with
        | .bool true => true
        | _ => false)).map (fun k => rows[k]?.getD 0) := by
  unfold filterPass
  congr 1
  apply List.filter_congr
  intro k _
  simp only [EC.get]
  cases nulls.isNull k <;> cases mask[k]?.getD false <;> rfl

end Columnar
