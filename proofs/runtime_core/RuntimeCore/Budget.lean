/-
# Row budgets: `row_budget`, `record_cap` (soft hint), `hard_record_cap` (hard stop)

| here | there |
| --- | --- |
| `Anc`                 | the `IR` variants the walk distinguishes, `runtime/runtime.rs` |
| `rowBudgetGo`, `rowBudget` | `Runtime::row_budget`, `runtime/runtime.rs` (`skip` is the accumulator, `thru` is `through_traverse`) |
| `recordCap`           | `Runtime::record_cap` = `row_budget(idx, true)`: the emitters' pack-ceiling hint |
| `hardRecordCap`       | `Runtime::hard_record_cap` = `row_budget(idx, false)`: the hard stop of `CondTraverseOp`, `ExpandIntoOp` and Sort's top-k |
| `Anc.sem`, `run`      | what each ancestor does to the row stream it pulls (LimitOp / SkipOp / ProjectOp / CondTraverseOp / ExpandIntoOp) |
| `hardCap`             | the `produced >= cap` stop + trim of `CondTraverseOp` (`ops/cond_traverse.rs`: `next`'s stop and `trim_to_cap`) and `ExpandIntoOp` (`ops/expand_into.rs`: stop, trim), and SortOp's top-`record_cap` heap (`ops/sort.rs`, `build_top_k`) |
| `need`                | the smallest sound budget (reference) |

The walk starts at the *parent* of the operator (`self.plan.node(cur).parent()`), so an
ancestor list here is "nearest parent first".

`Limit`/`Skip` values: the walk evaluates the expression with no row (`count`); a non-`Int`
or a negative value makes it return `None`, and makes `build_batch_op` fail the whole query
(`runtime.rs`). So on every plan that executes, each `Limit`/`Skip` ancestor carries a
non-negative `i64`, i.e. a `Nat` below `2^63`, which is what `Anc` stores.

`saturating_add` (`runtime.rs`): with several `Skip`s the sum can exceed `usize::MAX`, so it
is modelled exactly by `sat` (`min (a + b) USIZE_MAX`).
-/
namespace RuntimeCore.Budget

def USIZE_MAX : Nat := 2 ^ 64 - 1

/-- `usize::saturating_add`. -/
def sat (a b : Nat) : Nat := min (a + b) USIZE_MAX

/-- The ancestors the walk distinguishes. `project g` is any 1:1 operator the walk treats
as transparent (`Project`); `condTraverse`/`expandInto` map one input row to an arbitrary
list of rows (possibly empty); `other h` is any barrier. `Limit`/`Skip` values are
non-negative `i64`s (see header). -/
inductive Anc (α : Type) where
  | limit (n : Nat) (hn : n < 2 ^ 63 := by decide)
  | skip (n : Nat) (hn : n < 2 ^ 63 := by decide)
  | project (g : α → α)
  | condTraverse (f : α → List α)
  | expandInto (f : α → List α)
  | other (h : List α → List α)

variable {α : Type}

/-- `Runtime::row_budget` (`runtime.rs`), walk for walk: `skip` accumulates every `Skip`
(saturating), a `Limit n` returns `n.saturating_add(skip)`, `Project` passes,
`CondTraverse`/`ExpandInto` pass only when `thru` (`through_traverse`), else barrier. -/
def rowBudgetGo (thru : Bool) : Nat → List (Anc α) → Option Nat
  | _, [] => none
  | skip, .limit n _ :: _ => some (sat n skip)
  | skip, .skip s _ :: r => rowBudgetGo thru (sat skip s) r
  | skip, .project _ :: r => rowBudgetGo thru skip r
  | skip, .condTraverse _ :: r => if thru then rowBudgetGo thru skip r else none
  | skip, .expandInto _ :: r => if thru then rowBudgetGo thru skip r else none
  | _, .other _ :: _ => none

def rowBudget (as : List (Anc α)) (thru : Bool) : Option Nat := rowBudgetGo thru 0 as

/-- `Runtime::record_cap` (`runtime.rs`): the pack-ceiling hint. -/
def recordCap (as : List (Anc α)) : Option Nat := rowBudget as true

/-- `Runtime::hard_record_cap` (`runtime.rs`): the hard stop. -/
def hardRecordCap (as : List (Anc α)) : Option Nat := rowBudget as false

/-- The row-stream semantics of one ancestor. -/
def Anc.sem : Anc α → List α → List α
  | .limit n _, xs => xs.take n
  | .skip n _, xs => xs.drop n
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
  | .limit n _ :: _ => some n
  | .skip s _ :: r => (need r).map (· + s)
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
  | .limit n _ :: r, c, h, xs => by
    simp [need] at h; subst h
    simp only [run, Anc.sem, List.take_take, Nat.min_self]
  | .skip s _ :: r, c, h, xs => by
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

/-! ### `hard_record_cap` is `need` (saturated) -/

theorem rowBudgetGo_false (k : Nat) (hk : k ≤ USIZE_MAX) : ∀ as : List (Anc α),
    rowBudgetGo false k as = (need as).map (fun c => min (c + k) USIZE_MAX)
  | [] => rfl
  | .limit n _ :: _ => by simp [rowBudgetGo, need, sat]
  | .skip s _ :: r => by
    simp only [rowBudgetGo, need]
    rw [rowBudgetGo_false (sat k s) (by simp only [sat]; omega) r, Option.map_map]
    congr 1; funext c; simp only [Function.comp, sat]; omega
  | .project _ :: r => by simp only [rowBudgetGo, need]; exact rowBudgetGo_false k hk r
  | .condTraverse _ :: _ | .expandInto _ :: _ | .other _ :: _ => by simp [rowBudgetGo, need]

/-- **PROVEN** (`hard_record_cap` = the reference budget): it is `need`, saturated at
`usize::MAX`. -/
theorem hardRecordCap_eq (as : List (Anc α)) :
    hardRecordCap as = (need as).map (fun c => min c USIZE_MAX) := by
  simp only [hardRecordCap, rowBudget]
  rw [rowBudgetGo_false 0 (by decide) as]; simp

/-- **PROVEN** (#3071 fixes B1/B2/B3): the hard stop `hard_record_cap` gives `CondTraverse`,
`ExpandInto` and Sort's top-k never changes what reaches the top, for **every** ancestor
chain (traversals, stacked `SKIP`s, anything). The only premise: the operator emits at most
`usize::MAX` rows (its `produced` counter is a `usize`). -/
theorem hardRecordCap_sound (as : List (Anc α)) (xs : List α) (hx : xs.length ≤ USIZE_MAX) :
    run as (hardCap (hardRecordCap as) xs) = run as xs := by
  rw [hardRecordCap_eq]
  cases hn : need as with
  | none => rfl
  | some c =>
    simp only [Option.map_some, hardCap]
    by_cases hc : c ≤ USIZE_MAX
    · rw [Nat.min_eq_left hc]; exact need_sound as c hn xs
    · rw [Nat.min_eq_right (by omega), List.take_of_length_le hx]

/-- When the budget does not saturate, `hard_record_cap` is exactly `need`. -/
theorem hardRecordCap_eq_need (as : List (Anc α)) (c : Nat) (h : need as = some c)
    (hc : c ≤ USIZE_MAX) : hardRecordCap as = some c := by
  rw [hardRecordCap_eq, h]; simp [Nat.min_eq_left hc]

/-- A traversal between the operator and the `Limit` is a barrier for the hard stop. -/
theorem hardRecordCap_traverse (f : α → List α) (r : List (Anc α)) :
    hardRecordCap (.condTraverse f :: r) = none ∧ hardRecordCap (.expandInto f :: r) = none :=
  ⟨rfl, rfl⟩

/-! ### `record_cap` (the soft hint) -/

theorem rowBudgetGo_thru_of_need (k : Nat) : ∀ (as : List (Anc α)) (c : Nat), need as = some c →
    rowBudgetGo true k as = rowBudgetGo false k as
  | [], _, h => by simp [need] at h
  | .limit _ _ :: _, _, _ => rfl
  | .skip s _ :: r, c, h => by
    simp only [need, Option.map_eq_some_iff] at h
    obtain ⟨c', hc', _⟩ := h
    exact rowBudgetGo_thru_of_need (sat k s) r c' hc'
  | .project _ :: r, c, h => rowBudgetGo_thru_of_need k r c h
  | .condTraverse _ :: _, _, h | .expandInto _ :: _, _, h | .other _ :: _, _, h => by
    simp [need] at h

/-- Wherever the hard stop applies, the hint is the same value: `pack_hint` and
`record_cap` agree for an operator with only `Skip`/`Project` up to the `Limit`. -/
theorem recordCap_eq_hard_of_need (as : List (Anc α)) (c : Nat) (h : need as = some c) :
    recordCap as = hardRecordCap as :=
  rowBudgetGo_thru_of_need 0 as c h

/-- Chains up to the first `Limit` made of `Project`s only. -/
inductive NoSkip : List (Anc α) → Prop
  | limit (n : Nat) (hn : n < 2 ^ 63) (r : List (Anc α)) : NoSkip (.limit n hn :: r)
  | project (g : α → α) (r : List (Anc α)) : NoSkip r → NoSkip (.project g :: r)

/-- Chains up to the first `Limit` made of `Project`s and at most one `Skip`. -/
inductive OneSkip : List (Anc α) → Prop
  | none (r : List (Anc α)) : NoSkip r → OneSkip r
  | skip (s : Nat) (hs : s < 2 ^ 63) (r : List (Anc α)) : NoSkip r → OneSkip (.skip s hs :: r)
  | project (g : α → α) (r : List (Anc α)) : OneSkip r → OneSkip (.project g :: r)

theorem noSkip_need {as : List (Anc α)} (h : NoSkip as) :
    ∃ n, need as = some n ∧ n < 2 ^ 63 := by
  induction h with
  | limit n hn r => exact ⟨n, rfl, hn⟩
  | project g r _ ih => simpa [need] using ih

theorem oneSkip_need {as : List (Anc α)} (h : OneSkip as) :
    ∃ n, need as = some n ∧ n ≤ USIZE_MAX := by
  induction h with
  | none r hr =>
    obtain ⟨n, h1, h2⟩ := noSkip_need hr
    exact ⟨n, h1, by simp only [USIZE_MAX]; omega⟩
  | skip s hs r hr =>
    obtain ⟨n, h1, h2⟩ := noSkip_need hr
    exact ⟨n + s, by simp [need, h1], by simp only [USIZE_MAX]; omega⟩
  | project g r _ ih => simpa [need] using ih

/-- **PROVEN** (`record_cap` on the chains where it was always used as a hard stop): for a
chain of `Project`s and at most one `Skip` below the `Limit`, `record_cap ≥ need`, so the
hard cap is sound. -/
theorem recordCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) :
    ∃ c c', need as = some c ∧ recordCap as = some c' ∧ c ≤ c' := by
  obtain ⟨n, h1, h2⟩ := oneSkip_need h
  exact ⟨n, n, h1, by rw [recordCap_eq_hard_of_need as n h1, hardRecordCap_eq_need as n h1 h2],
    Nat.le_refl _⟩

theorem recordCap_hardCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) (xs : List α) :
    run as (hardCap (recordCap as) xs) = run as xs := by
  obtain ⟨c, c', h1, h2, h3⟩ := recordCap_sound_oneSkip h
  rw [h2]; exact hardCap_sound_of_need_le as c c' h1 h3 xs

/-! ### The former counterexamples (B1, B2), now answered right

The chains of the queries that lost rows before #3071 (`MATCH (a:A)-[:R]->(b)-[:R]->(a)
RETURN a.id LIMIT 1`, `MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1`,
`MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1`). -/

def cycleChain : List (Anc Nat) :=
  [.expandInto (fun a => if a = 2 then [a] else []), .project id, .limit 1]

def twoHopChain : List (Anc Nat) :=
  [.condTraverse (fun b => if b = 20 then [30] else []), .project id, .limit 1]

def stackedSkips : List (Anc Nat) := [.skip 0, .project id, .skip 1, .limit 1]

/-- The lower traverse under an `ExpandInto` gets no hard stop and the answer `[2]` survives. -/
theorem hardRecordCap_cycleChain :
    hardRecordCap cycleChain = none ∧
    run cycleChain (hardCap (hardRecordCap cycleChain) [1, 2]) = [2] := by decide

theorem hardRecordCap_twoHopChain :
    hardRecordCap twoHopChain = none ∧
    run twoHopChain (hardCap (hardRecordCap twoHopChain) [10, 20]) = [30] := by decide

/-- Stacked `SKIP`s add up: the budget is 2 and the answer `[20]` survives. -/
theorem hardRecordCap_stackedSkips :
    hardRecordCap stackedSkips = some 2 ∧ need stackedSkips = some 2 ∧
    run stackedSkips (hardCap (hardRecordCap stackedSkips) [10, 20, 30]) = [20] := by decide

/-- The soft hint still walks through traversals (it only sizes emitter batches:
`Emitter.drive_flatten` holds for any ceiling). -/
theorem recordCap_twoHopChain : recordCap twoHopChain = some 1 := by decide

end RuntimeCore.Budget
