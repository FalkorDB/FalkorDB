/-
# Row-budget hints: `effective_limit`, `effective_skip`, `record_cap`

| here | there |
| --- | --- |
| `Anc`                 | the `IR` variants the walks distinguish, `runtime/runtime.rs:581-600`, `:616-631` |
| `effectiveLimit`      | `Runtime::effective_limit`, `runtime/runtime.rs:575-605` |
| `effectiveSkip`       | `Runtime::effective_skip`, `runtime/runtime.rs:610-636` |
| `recordCap`           | `Runtime::record_cap`, `runtime/runtime.rs:642-648` |
| `Anc.sem`, `run`      | what each ancestor does to the row stream it pulls (LimitOp / SkipOp / ProjectOp / CondTraverseOp / ExpandIntoOp) |
| `hardCap`             | the `produced >= cap` stop + trim of `CondTraverseOp` (`ops/cond_traverse.rs:1030-1049,1179`) and `ExpandIntoOp` (`ops/expand_into.rs:246-293`), and SortOp's top-`limit+skip` heap (`ops/sort.rs:491-493`) |
| `need`                | the smallest sound budget (reference) |

The walks start at the *parent* of the operator (`self.plan.node(cur).parent()`), so an
ancestor list here is "nearest parent first".

`Limit`/`Skip` values: the walks evaluate the expression with no row; a non-`Int` or a
negative value makes the walk return `None`/`0`, and makes `build_batch_op` fail the
whole query (`runtime.rs:818-848`). So on every plan that executes, each `Limit`/`Skip`
ancestor carries a `Nat`, which is what `Anc` stores.

`saturating_add` (`runtime.rs:647`): both operands are `< 2^63` (they come from a
non-negative `i64`), so the sum never saturates in `usize`; `Nat` addition is exact.
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

/-- `Runtime::effective_limit` (`runtime.rs:575-605`). -/
def effectiveLimit : List (Anc α) → Option Nat
  | [] => none
  | .limit n :: _ => some n
  | .project _ :: r | .skip _ :: r | .condTraverse _ :: r | .expandInto _ :: r => effectiveLimit r
  | .other _ :: _ => none

/-- `Runtime::effective_skip` (`runtime.rs:610-636`): the *first* `Skip` reached through
`Project`/`Limit`, else 0. -/
def effectiveSkip : List (Anc α) → Nat
  | [] => 0
  | .skip n :: _ => n
  | .project _ :: r | .limit _ :: r => effectiveSkip r
  | _ :: _ => 0

/-- `Runtime::record_cap` (`runtime.rs:642-648`). -/
def recordCap (as : List (Anc α)) : Option Nat :=
  (effectiveLimit as).map (· + effectiveSkip as)

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
    ∃ n, need as = some n ∧ effectiveLimit as = some n := by
  induction h with
  | limit n r => exact ⟨n, rfl, rfl⟩
  | project g r _ ih => simpa [need, effectiveLimit] using ih

/-- **PROVEN** (when `record_cap` is right): for a chain of `Project`s and at most one
`Skip` below the `Limit`, `record_cap ≥ need`, so the hard cap is sound. -/
theorem recordCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) :
    ∃ c c', need as = some c ∧ recordCap as = some c' ∧ c ≤ c' := by
  induction h with
  | none r hr =>
    obtain ⟨n, h1, h2⟩ := noSkip_need hr
    exact ⟨n, n + effectiveSkip r, h1, by simp [recordCap, h2], by omega⟩
  | skip s r hr =>
    obtain ⟨n, h1, h2⟩ := noSkip_need hr
    exact ⟨n + s, n + s, by simp [need, h1], by simp [recordCap, effectiveLimit, effectiveSkip, h2], Nat.le_refl _⟩
  | project g r _ ih =>
    obtain ⟨c, c', h1, h2, h3⟩ := ih
    exact ⟨c, c', by simpa [need] using h1, by simpa [recordCap, effectiveLimit, effectiveSkip] using h2, h3⟩

theorem recordCap_hardCap_sound_oneSkip {as : List (Anc α)} (h : OneSkip as) (xs : List α) :
    run as (hardCap (recordCap as) xs) = run as xs := by
  obtain ⟨c, c', h1, h2, h3⟩ := recordCap_sound_oneSkip h
  rw [h2]; exact hardCap_sound_of_need_le as c c' h1 h3 xs

/-! ### Where `record_cap` is wrong for a hard cap (CONFIRMED against Rust, see header) -/

/-- The lower `CondTraverse` of `MATCH (a:A)-[:R]->(b)-[:R]->(a) RETURN a.id LIMIT 1`:
ancestors `ExpandInto` (keeps only the pair that closes the cycle), `Project`, `Limit 1`.
Rows are `a.id`; pair `1` has no edge back, pair `2` does. -/
def cycleChain : List (Anc Nat) :=
  [.expandInto (fun a => if a = 2 then [a] else []), .project id, .limit 1]

/-- **PROVEN** (bug B1): `record_cap` walks through `ExpandInto`, and the hard cap it
gives the traverse beneath drops the only answer: `[]` instead of `[2]`. -/
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

/-- `MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1`: the traverse's
ancestors are `Skip 0`, `Project`, `Skip 1`, `Limit 1`. `effective_skip` stops at the
first `Skip` (0), so `record_cap = 1` while `need = 2`. -/
def stackedSkips : List (Anc Nat) := [.skip 0, .project id, .skip 1, .limit 1]

/-- **PROVEN** (bug B2): stacked `Skip`s make the hard cap too small. -/
theorem hard_cap_unsound_stacked_skip :
    recordCap stackedSkips = some 1 ∧ need stackedSkips = some 2 ∧
    run stackedSkips (hardCap (recordCap stackedSkips) [10, 20, 30]) = [] ∧
    run stackedSkips [10, 20, 30] = [20] := by decide

/-- **PROVEN** (general form of B1): whenever a `CondTraverse`/`ExpandInto` sits between
a hard-capped operator and the `Limit`, some expansion makes the cap drop rows. -/
theorem hard_cap_unsound_through_traverse (n : Nat) :
    ∃ (f : Nat → List Nat) (xs : List Nat),
      let as : List (Anc Nat) := [.condTraverse f, .limit (n + 1)]
      recordCap as = some (n + 1) ∧ run as (hardCap (recordCap as) xs) ≠ run as xs := by
  refine ⟨fun x => if x = n + 1 then [x] else [], List.range (n + 2), ?_⟩
  simp only [recordCap, effectiveLimit, effectiveSkip, Option.map_some, Nat.add_zero, true_and]
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

/-- Suggested fix: give the hard-cap consumers `need` (Skip adds, Project passes,
traverse is a barrier) instead of `effective_limit + effective_skip`. `need_sound` is
the correctness proof of that fix. On the soft consumers (`BatchedResultEmitter`) the
current value is harmless — see `Emitter.emitAll_flatten`. -/
theorem need_is_barrier_at_traverse (f : α → List α) (r : List (Anc α)) :
    need (.condTraverse f :: r) = none ∧ need (.expandInto f :: r) = none := ⟨rfl, rfl⟩

end RuntimeCore.Budget
