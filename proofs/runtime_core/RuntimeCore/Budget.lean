/-
# Row-budget hints: `row_budget`, `record_cap`, `hard_record_cap` (PR #3071)

| here | there |
| --- | --- |
| `Anc`                 | the `IR` variants the walk distinguishes, `runtime/runtime.rs:695-722` |
| `rowBudgetAux`        | `Runtime::row_budget`, `runtime/runtime.rs:695-722` (`skip` accumulator `:704`, `Limit` `:711`, `Skip` `:712`, `Project` `:713`, traverse iff `through_traverse` `:714`, barrier `:715`) |
| `recordCap`           | `Runtime::record_cap` = `row_budget(idx, true)`, `runtime/runtime.rs:728-733` (the emitters' pack-ceiling *hint*) |
| `hardRecordCap`       | `Runtime::hard_record_cap` = `row_budget(idx, false)`, `runtime/runtime.rs:740-745` (the hard stop of CondTraverse/ExpandInto and Sort's top-k, `build_batch_op` `:962`, `:1017`, `:1040`) |
| `Anc.sem`, `run`      | what each ancestor does to the row stream it pulls (LimitOp / SkipOp / ProjectOp / CondTraverseOp / ExpandIntoOp) |
| `hardCap`             | the `produced >= cap` stop + trim of `CondTraverseOp` (`ops/cond_traverse.rs`: `next`'s stop `:1410-1414`, `trim_to_cap` `:1349`) and `ExpandIntoOp` (`ops/expand_into.rs`: stop `:253`, trim `:294-300`), and SortOp's top-`record_cap` heap (`ops/sort.rs:504-506`, `build_top_k`) |
| `need`                | the smallest sound budget (reference) |
| `preEffectiveLimit`, `preEffectiveSkip`, `preRecordCap` | the walks before #3071 (`effective_limit`, `effective_skip`, `record_cap` at 343e077f2 `runtime.rs:694`, `:729`, `:761`), kept for bugs B1/B2 |

The walk starts at the *parent* of the operator (`self.plan.node(cur).parent()`), so an
ancestor list here is "nearest parent first".

`Limit`/`Skip` values: the walk evaluates the expression with no row; a non-`Int` or a
negative value makes the walk return `None`, and makes `build_batch_op` fail the
whole query. So on every plan that executes, each `Limit`/`Skip`
ancestor carries a `Nat`, which is what `Anc` stores.

`saturating_add`: each operand is `< 2^63` (a non-negative `i64`); a saturated sum
(`usize::MAX`) is larger than any row count a stream can reach, so `Nat` addition gives
the same cut.

PR #3071 fixes B1 and B2: the hard-cap consumers now get `hardRecordCap`, which is
exactly `need` (`hardRecordCap_eq_need`), so the hard cap is sound on every ancestor
chain (`hardRecordCap_sound`). `recordCap` (still walking through traverses) only sizes
emitter batches, where it cannot change rows (`Emitter.drive_flatten`).
-/
namespace RuntimeCore.Budget

/-- The ancestors the walks distinguish. `project g` is any 1:1 operator the walks treat
as transparent (`Project`); `traverse f` is `CondTraverse`/`ExpandInto`, which map one
input row to an arbitrary list of rows (possibly empty); `other h` is any barrier. -/
inductive Anc (α : Type) where
  | limit (n : Nat)
  | skip (n : Nat)
  | project (g : α → α)
  | condTraverse (f : α → List α)
  | expandInto (f : α → List α)
  | other (h : List α → List α)

variable {α : Type}

/-- `Runtime::row_budget` (`runtime.rs:695-722`) with its `skip` accumulator: a `Limit n`
returns `n + skip`, a `Skip k` adds `k`, a `Project` passes, `CondTraverse`/`ExpandInto`
pass iff `through`, anything else is a barrier. -/
def rowBudgetAux (through : Bool) : Nat → List (Anc α) → Option Nat
  | _, [] => none
  | s, .limit n :: _ => some (n + s)
  | s, .skip k :: r => rowBudgetAux through (s + k) r
  | s, .project _ :: r => rowBudgetAux through s r
  | s, .condTraverse _ :: r | s, .expandInto _ :: r =>
    if through then rowBudgetAux through s r else none
  | _, .other _ :: _ => none

/-- `Runtime::record_cap` (`runtime.rs:728-733`): the pack-ceiling hint. -/
def recordCap (as : List (Anc α)) : Option Nat := rowBudgetAux true 0 as

/-- `Runtime::hard_record_cap` (`runtime.rs:740-745`): the hard stop. -/
def hardRecordCap (as : List (Anc α)) : Option Nat := rowBudgetAux false 0 as

/-- Pre-#3071 `Runtime::effective_limit` (343e077f2 `runtime.rs:694`). -/
def preEffectiveLimit : List (Anc α) → Option Nat
  | [] => none
  | .limit n :: _ => some n
  | .project _ :: r | .skip _ :: r | .condTraverse _ :: r | .expandInto _ :: r => preEffectiveLimit r
  | .other _ :: _ => none

/-- Pre-#3071 `Runtime::effective_skip` (343e077f2 `runtime.rs:729`): the *first* `Skip`
reached through `Project`/`Limit`, else 0. -/
def preEffectiveSkip : List (Anc α) → Nat
  | [] => 0
  | .skip n :: _ => n
  | .project _ :: r | .limit _ :: r => preEffectiveSkip r
  | _ :: _ => 0

/-- Pre-#3071 `Runtime::record_cap` (343e077f2 `runtime.rs:761`), then also the hard cap. -/
def preRecordCap (as : List (Anc α)) : Option Nat :=
  (preEffectiveLimit as).map (· + preEffectiveSkip as)

/-- The row-stream semantics of one ancestor. -/
def Anc.sem : Anc α → List α → List α
  | .limit n, xs => xs.take n
  | .skip n, xs => xs.drop n
  | .project g, xs => xs.map g
  | .condTraverse f, xs => xs.flatMap f
  | .expandInto f, xs => xs.flatMap f
  | .other h, xs => h xs

/-- The rows reaching the top of the chain. -/
def run : List (Anc α) → List α → List α
  | [], xs => xs
  | a :: as, xs => run as (a.sem xs)

/-- A hard cap: the operator emits at most `cap` of its rows and then stops. -/
def hardCap (cap : Option Nat) (ys : List α) : List α :=
  match cap with
  | none => ys
  | some c => ys.take c

/-- Reference: the smallest budget that is sound for every row stream — a `Limit`
bounds it, a `Skip` adds to it, a 1:1 `Project` passes it, anything else is a barrier. -/
def need : List (Anc α) → Option Nat
  | [] => none
  | .limit n :: _ => some n
  | .skip s :: r => (need r).map (· + s)
  | .project _ :: r => need r
  | _ :: _ => none

theorem take_take_le (xs : List α) {c c' : Nat} (h : c ≤ c') :
    (xs.take c').take c = xs.take c := by
  rw [List.take_take]; congr 1; omega

/-- **PROVEN** (the reference budget is sound): cutting the stream to `need` rows never
changes what reaches the top. -/
theorem need_sound : ∀ (as : List (Anc α)) (c : Nat), need as = some c →
    ∀ xs, run as (xs.take c) = run as xs
  | [], _, h, _ => by simp [need] at h
  | .limit n :: r, c, h, xs => by
    simp [need] at h; subst h
    simp only [run, Anc.sem, List.take_take, Nat.min_self]
  | .skip s :: r, c, h, xs => by
    simp only [need, Option.map_eq_some_iff] at h
    obtain ⟨c', hc', rfl⟩ := h
    simp only [run, Anc.sem]
    have : (xs.take (c' + s)).drop s = (xs.drop s).take c' := by
      rw [List.drop_take]; congr 1; omega
    rw [this, need_sound r c' hc']
  | .project g :: r, c, h, xs => by
    simp only [need] at h
    simp only [run, Anc.sem]
    rw [List.map_take]
    exact need_sound r c h _
  | .condTraverse _ :: _, _, h, _ | .expandInto _ :: _, _, h, _ | .other _ :: _, _, h, _ => by
    simp [need] at h

/-- **PROVEN**: any budget at least `need` is sound too. -/
theorem hardCap_sound_of_need_le (as : List (Anc α)) (c c' : Nat) (h : need as = some c)
    (hle : c ≤ c') (xs : List α) : run as (hardCap (some c') xs) = run as xs := by
  simp only [hardCap]
  rw [← need_sound as c h (xs.take c'), take_take_le xs hle, need_sound as c h xs]

/-- Chains up to the first `Limit` made of `Project`s only. -/
inductive NoSkip : List (Anc α) → Prop
  | limit (n : Nat) (r : List (Anc α)) : NoSkip (.limit n :: r)
  | project (g : α → α) (r : List (Anc α)) : NoSkip r → NoSkip (.project g :: r)

/-- Chains up to the first `Limit` made of `Project`s and at most one `Skip`. -/
inductive OneSkip : List (Anc α) → Prop
  | none (r : List (Anc α)) : NoSkip r → OneSkip r
  | skip (s : Nat) (r : List (Anc α)) : NoSkip r → OneSkip (.skip s :: r)
  | project (g : α → α) (r : List (Anc α)) : OneSkip r → OneSkip (.project g :: r)

theorem noSkip_need {as : List (Anc α)} (h : NoSkip as) :
    ∃ n, need as = some n ∧ preEffectiveLimit as = some n := by
  induction h with
  | limit n r => exact ⟨n, rfl, rfl⟩
  | project g r _ ih => simpa [need, preEffectiveLimit] using ih

/-- Whenever `need` gives a budget, the walk gives the same one plus its accumulated
skip, whether or not it walks through traverses. -/
theorem rowBudgetAux_of_need (b : Bool) : ∀ (as : List (Anc α)) (c s : Nat),
    need as = some c → rowBudgetAux b s as = some (c + s)
  | [], _, _, h => by simp [need] at h
  | .limit n :: _, c, s, h => by simp [need] at h; subst h; rfl
  | .skip k :: r, c, s, h => by
    simp only [need, Option.map_eq_some_iff] at h
    obtain ⟨c', hc', rfl⟩ := h
    simp only [rowBudgetAux]
    rw [rowBudgetAux_of_need b r c' (s + k) hc']; congr 1; omega
  | .project _ :: r, c, s, h => by
    simp only [need] at h; simp only [rowBudgetAux]; exact rowBudgetAux_of_need b r c s h
  | .condTraverse _ :: _, _, _, h | .expandInto _ :: _, _, _, h | .other _ :: _, _, _, h => by
    simp [need] at h

/-- The hard walk with skip `s` is `need` shifted by `s`. -/
theorem rowBudgetAux_false (s : Nat) : ∀ (as : List (Anc α)),
    rowBudgetAux false s as = (need as).map (· + s)
  | [] => rfl
  | .limit n :: _ => rfl
  | .skip k :: r => by
    simp only [rowBudgetAux, need, rowBudgetAux_false (s + k) r, Option.map_map]
    congr 1; funext x; simp only [Function.comp]; omega
  | .project _ :: r => by simp only [rowBudgetAux, need]; exact rowBudgetAux_false s r
  | .condTraverse _ :: _ | .expandInto _ :: _ | .other _ :: _ => by simp [rowBudgetAux, need]

/-- **PROVEN** (#3071): `hard_record_cap` is exactly the reference budget `need`. -/
theorem hardRecordCap_eq_need (as : List (Anc α)) : hardRecordCap as = need as := by
  simp [hardRecordCap, rowBudgetAux_false]

/-- **PROVEN** (#3071, headline): the hard stop that `CondTraverse`, `ExpandInto` and
Sort's top-k apply with `hard_record_cap` never changes what reaches the top, on every
ancestor chain (stacked `SKIP`s and traverses included). -/
theorem hardRecordCap_sound (as : List (Anc α)) (xs : List α) :
    run as (hardCap (hardRecordCap as) xs) = run as xs := by
  rw [hardRecordCap_eq_need]
  cases h : need as with
  | none => rfl
  | some c => exact hardCap_sound_of_need_le as c c h (Nat.le_refl _) xs

/-- **PROVEN**: where the hard cap exists, the hint `record_cap` equals it. -/
theorem recordCap_eq_hard (as : List (Anc α)) (c : Nat) (h : hardRecordCap as = some c) :
    recordCap as = some c := by
  rw [hardRecordCap_eq_need] at h
  simpa [recordCap] using rowBudgetAux_of_need true as c 0 h

/-- **PROVEN**: for a chain of `Project`s and at most one `Skip` below the `Limit`,
`record_cap ≥ need`, so the hard cap is sound. -/
theorem recordCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) :
    ∃ c c', need as = some c ∧ recordCap as = some c' ∧ c ≤ c' := by
  have hn : ∃ c, need as = some c := by
    induction h with
    | none r hr => obtain ⟨n, h1, -⟩ := noSkip_need hr; exact ⟨n, h1⟩
    | skip s r hr => obtain ⟨n, h1, -⟩ := noSkip_need hr; exact ⟨n + s, by simp [need, h1]⟩
    | project g r _ ih => simpa [need] using ih
  obtain ⟨c, hc⟩ := hn
  exact ⟨c, c, hc, by simpa [recordCap] using rowBudgetAux_of_need true as c 0 hc, Nat.le_refl _⟩

theorem recordCap_hardCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) (xs : List α) :
    run as (hardCap (recordCap as) xs) = run as xs := by
  obtain ⟨c, c', h1, h2, h3⟩ := recordCap_sound_oneSkip h
  rw [h2]; exact hardCap_sound_of_need_le as c c' h1 h3 xs

/-! ### Why `record_cap` is only a hint (bug B1 before #3071)

`record_cap` walks through `CondTraverse`/`ExpandInto`. Before #3071 it was also the hard
stop of those operators (B1, CONFIRMED against Rust); since #3071 they stop at
`hard_record_cap`, and the theorems below show what goes wrong if the hint were used as a
hard stop, next to what `hard_record_cap` gives. -/

/-- The lower `CondTraverse` of `MATCH (a:A)-[:R]->(b)-[:R]->(a) RETURN a.id LIMIT 1`:
ancestors `ExpandInto` (keeps only the pair that closes the cycle), `Project`, `Limit 1`.
Rows are `a.id`; pair `1` has no edge back, pair `2` does. -/
def cycleChain : List (Anc Nat) :=
  [.expandInto (fun a => if a = 2 then [a] else []), .project id, .limit 1]

/-- **PROVEN** (bug B1, pre-#3071 hard cap): the hint as a hard cap drops the only answer:
`[]` instead of `[2]`. The pre-#3071 `record_cap` is the same here. -/
theorem hard_cap_unsound_expandInto :
    recordCap cycleChain = some 1 ∧
    run cycleChain (hardCap (recordCap cycleChain) [1, 2]) = [] ∧
    run cycleChain [1, 2] = [2] := by decide

/-- The lower `CondTraverse` of `MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1`:
the upper traverse maps `b1 ↦ []` (dead end) and `b2 ↦ [c]`. -/
def twoHopChain : List (Anc Nat) :=
  [.condTraverse (fun b => if b = 20 then [30] else []), .project id, .limit 1]

theorem hard_cap_unsound_condTraverse :
    recordCap twoHopChain = some 1 ∧
    run twoHopChain (hardCap (recordCap twoHopChain) [10, 20]) = [] ∧
    run twoHopChain [10, 20] = [30] := by decide

theorem pre3071_record_cap_was_hard :
    preRecordCap cycleChain = some 1 ∧ preRecordCap twoHopChain = some 1 := by decide

/-- **PROVEN** (#3071 fixes B1): `hard_record_cap` stops at the traverse, so both chains
keep their answer. -/
theorem hardRecordCap_fixes_traverse :
    hardRecordCap cycleChain = none ∧ run cycleChain (hardCap (hardRecordCap cycleChain) [1, 2]) = [2] ∧
    hardRecordCap twoHopChain = none ∧
      run twoHopChain (hardCap (hardRecordCap twoHopChain) [10, 20]) = [30] := by decide

/-- `MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1`: the traverse's
ancestors are `Skip 0`, `Project`, `Skip 1`, `Limit 1`. -/
def stackedSkips : List (Anc Nat) := [.skip 0, .project id, .skip 1, .limit 1]

/-- **PROVEN** (bug B2 before #3071, fixed by #3071): pre-#3071 `effective_skip` stopped at
the first `Skip` (0), so the cap was 1 while `need = 2`. -/
theorem pre3071_hard_cap_unsound_stacked_skip :
    preRecordCap stackedSkips = some 1 ∧ need stackedSkips = some 2 ∧
    run stackedSkips (hardCap (preRecordCap stackedSkips) [10, 20, 30]) = [] ∧
    run stackedSkips [10, 20, 30] = [20] := by decide

/-- **PROVEN** (#3071 fixes B2): every `Skip` on the way is added. -/
theorem stacked_skip_fixed :
    hardRecordCap stackedSkips = some 2 ∧ recordCap stackedSkips = some 2 ∧
    run stackedSkips (hardCap (hardRecordCap stackedSkips) [10, 20, 30]) = [20] := by decide

/-- **PROVEN** (general form of B1): whenever a `CondTraverse` sits between an operator
and the `Limit`, using the hint `record_cap` as a hard cap drops rows for some expansion. -/
theorem hard_cap_unsound_through_traverse (n : Nat) :
    ∃ (f : Nat → List Nat) (xs : List Nat),
      let as : List (Anc Nat) := [.condTraverse f, .limit (n + 1)]
      recordCap as = some (n + 1) ∧ run as (hardCap (recordCap as) xs) ≠ run as xs := by
  refine ⟨fun x => if x = n + 1 then [x] else [], List.range (n + 2), ?_⟩
  simp only [recordCap, rowBudgetAux, ite_true, Nat.add_zero, true_and]
  simp only [hardCap, run, Anc.sem]
  intro h
  have h1 : ((List.range (n + 2)).take (n + 1)) = List.range (n + 1) := by
    rw [List.take_range, Nat.min_eq_left (by omega)]
  rw [h1] at h
  have hl : ((List.range (n + 1)).flatMap (fun x => if x = n + 1 then [x] else [])) = [] := by
    rw [List.flatMap_eq_nil_iff]
    intro x hx; simp at hx; simp; omega
  have hr : ((List.range (n + 2)).flatMap (fun x => if x = n + 1 then [x] else [])) ≠ [] := by
    intro h'; rw [List.flatMap_eq_nil_iff] at h'
    have := h' (n + 1) (by simp); simp at this
  rw [hl] at h
  simp only [List.take_nil] at h
  rcases List.take_eq_nil_iff.mp h.symm with h0 | h0
  · omega
  · exact hr h0

theorem need_is_barrier_at_traverse (f : α → List α) (r : List (Anc α)) :
    need (.condTraverse f :: r) = none ∧ need (.expandInto f :: r) = none := ⟨rfl, rfl⟩

end RuntimeCore.Budget
