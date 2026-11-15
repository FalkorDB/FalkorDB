import FalkorCore.Int64

/-!
# Lazy `UNWIND range(..)`: `RangeIter` after #2899 (`graph/src/runtime/eval.rs:112-133`)

One model of what `functions_str_list` (eager `range`, `math.rs:251`) and `expr_semantics`
(`eval_iter_expr`, `RangeIter::next`) each used to carry. `eval_iter_expr` (`eval.rs:1227-1270`)
builds `RangeIter { current: start, step, remaining: length }` with the signed `step` and
`length = |end - start| / |step| + 1` (`rangeLen`, computed in `i128`, so exact). `next` yields
`current`, decrements `remaining` and advances `current += step` only while `remaining > 0`.

| here | there |
| --- | --- |
| `rangeLen`     | the `length` of `eval_iter_expr` (`eval.rs:1252-1263`) and of eager `range` (`math.rs:251-308`), when they get that far (see `rangeLen`) |
| `rangeList`    | the elements `start + k*step`, `k < n` (`(start..=end).step_by(step)`) |
| `rangeIter`    | `RangeIter::next`, debug build (checked `+=`) |
| `rangeIterRel` | `RangeIter::next`, release build (wrapping `+=`) |
-/
namespace FalkorCore

/-- `length` as `eval_iter_expr` computes it (in `i128`/`u128`, so exactly), **for the arguments that reach
that computation**: `step ≠ 0` (a zero step is an error, `eval.rs:1245`), a compatible direction (not
`start > end` with `step > 0`, nor `start < end` with `step < 0`: those yield no rows, `:1249`), and a length of at
most `u32::MAX` (a longer one is the error `Range too large`, `:1261-1263`). Outside those conditions the Rust
returns no rows or an error where `rangeLen` still returns a positive count: this is the formula, not the
iterator's result. -/
def rangeLen (a e s : Int) : Nat := (e - a).natAbs / s.natAbs + 1

/-- Equivalent form with the distance written as a case split on the direction. -/
theorem rangeLen_eq_cases (a e s : Int) :
    rangeLen a e s = ((if e ≥ a then e - a else a - e).natAbs / s.natAbs) + 1 := by
  unfold rangeLen; split <;> (congr 2 <;> omega)

/-- The eager `range()` list: `start + k*step` for `k < n`. -/
def rangeList (a s : Int) (n : Nat) : List Int :=
  (List.range n).map fun (k : Nat) => a + (k : Int) * s

/-- `RangeIter::next` after #2899, debug build (checked `+=`): `remaining` counts the elements
still due and `current` advances only while another one is. -/
def rangeIter (step : Int) : Nat → Int → Except String (List Int)
  | 0, _ => .ok []
  | r + 1, cur =>
    if r = 0 then .ok [cur]
    else if I64MIN ≤ cur + step ∧ cur + step ≤ I64MAX then
      (rangeIter step r (cur + step)).map (cur :: ·)
    else .error "attempt to add with overflow"

/-- The same in a release build (wrapping `+=`). -/
def rangeIterRel (step : Int) : Nat → Int → List Int
  | 0, _ => []
  | r + 1, cur => if r = 0 then [cur] else cur :: rangeIterRel step r (wrap (cur + step))

/-- The lazy iterator is the element list (debug and release) as long as every element it is due
to yield is an `i64` (it never computes the one past the last). -/
theorem rangeIter_of_inRange (step : Int) :
    ∀ (r : Nat) (cur : Int), (∀ k : Nat, k < r → InI64 (cur + k * step)) →
      rangeIter step r cur = .ok (rangeList cur step r) ∧
      rangeIterRel step r cur = rangeList cur step r
  | 0, _, _ => ⟨rfl, rfl⟩
  | r + 1, cur, h => by
    by_cases hr : r = 0
    · subst hr; simp [rangeIter, rangeIterRel, rangeList]
    · have h1 : InI64 (cur + step) := by have := h 1 (by omega); simpa using this
      have ih := rangeIter_of_inRange step r (cur + step) (fun k hk => by
        have := h (k + 1) (by omega)
        have e : cur + step + (k : Int) * step = cur + ((k + 1 : Nat) : Int) * step := by
          push_cast; rw [Int.add_mul]; omega
        rwa [e])
      have hl : rangeList cur step (r + 1) = cur :: rangeList (cur + step) step r := by
        simp only [rangeList, List.range_succ_eq_map, List.map_cons, List.map_map]
        congr 1
        · simp
        · apply List.map_congr_left; intro k _
          simp only [Function.comp]; push_cast; rw [Int.add_mul]; omega
      have hw : wrap (cur + step) = cur + step := wrap_id h1
      have h1' : I64MIN ≤ cur + step ∧ cur + step ≤ I64MAX := h1
      refine ⟨?_, ?_⟩
      · simp only [rangeIter, hr, ↓reduceIte, h1', and_self, ih.1, hl]; rfl
      · simp only [rangeIterRel, hr, ↓reduceIte, hw, ih.2, hl]

end FalkorCore
