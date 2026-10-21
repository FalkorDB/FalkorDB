import FalkorCore.Value
import FalkorStrList

/-!
# PR #3437: a list comprehension / quantifier body, chunked (`runtime/eval.rs`)

Cites are to the PR branch (`perf/list-comprehension`, 6efd69cd6) unless marked *main*
(origin/main 9212add39).

| Lean | Rust |
| --- | --- |
| `isTrue` | `matches!(v, Value::Bool(true))` (eval.rs:1528, :1537, :1631) |
| `kOf`, `boolMismatch` | the `Bool(true)/Bool(false)/Null/_ => Err` match (eval.rs:1545-1554, :1588-1597; *main* :1013-1023) |
| `Sink` | `enum LoopSink` (eval.rs:304-311) |
| `refStep`, `ref` | *main*'s per-element loops: comprehension eval.rs:1072-1080, quantifier :1010-1025 (`mainCompLoop`, `mainQuantLoop`) |
| `Body` | `struct ListLoop` (eval.rs:289-300) after `ListLoop::new` (:323-338): `hasPred = false` iff the predicate was the literal `true`, `hasProj = false` iff the projection was the loop variable |
| `prStep`, `bulk`, `perElem` | the per-element half of `list_loop_chunk` (eval.rs:1497-1549), with the `predicate.is_none() && projection.is_none()` shortcut (:1504-1511) |
| `EC`, `EC.vals` | `ExprColumn` read back through `ExprColumn::get`; `bools` is the `Bools(mask, nulls)` lane |
| `Col`, `Col.Sound` | `VectorEval::eval` / `eval_values` over the chunk batch (eval.rs:1569, :1630, :1636) and the contract they meet (see `Bridge`) |
| `countLocal`, `tallyNoProj`, `passedOf`, `columnar` | `list_loop_columnar` (eval.rs:1553-1655) |
| `chunkStep` | `list_loop_chunk` (eval.rs:1459-1550): columnar first, re-run per element on any error |
| `prRun` | the `for chunk in …` loops (eval.rs:1350-1359, :1411-1421, :1428-1430) |
-/

namespace Pr3437.ListLoop
open FalkorCore FalkorStrList

variable {F : Type}

/-- `matches!(v, Value::Bool(true))`. -/
def isTrue : V F → Bool
  | .bool true => true
  | _ => false

/-- The quantifier's three outcomes; `none` is the type error. -/
def kOf : V F → Option K
  | .bool true => some .t
  | .bool false => some .f
  | .null => some .n
  | _ => none

def boolMismatch (v : V F) : String := "Type mismatch: expected Boolean but was " ++ v.name

/-- `enum LoopSink` (eval.rs:304). -/
inductive Sink (F : Type) where
  | collect (acc : List (V F))
  | count (t f n : Nat)
  | tally (n : Nat)

def bump : K → Nat → Nat → Nat → Sink F
  | .t, t, f, n => .count (t + 1) f n
  | .f, t, f, n => .count t (f + 1) n
  | .n, t, f, n => .count t f (n + 1)

/-! ## The reference: main's per-element loops -/

section
variable (pe pj : V F → Except String (V F))

/-- One element of main's loops: predicate first, projection only if it passed. -/
def refStep : Sink F → V F → Except String (Sink F)
  | .collect acc, v => do
      let p ← pe v
      if isTrue p then do let y ← pj v; pure (.collect (acc ++ [y])) else pure (.collect acc)
  | .tally k, v => do
      let p ← pe v
      if isTrue p then do let _ ← pj v; pure (.tally (k + 1)) else pure (.tally k)
  | .count t f n, v => do
      let p ← pe v
      match kOf p with
      | some k => pure (bump k t f n)
      | none => throw (boolMismatch p)

def ref (s : Sink F) (l : List (V F)) : Except String (Sink F) := l.foldlM (refStep pe pj) s

/-- *main* eval.rs:1072-1080: `row.insert(var, value); match eval(child 1) { Bool(true) => {},
_ => continue }; acc.push(eval(child 2)?)`. -/
def mainCompLoop : List (V F) → List (V F) → Except String (List (V F))
  | acc, [] => .ok acc
  | acc, v :: vs =>
    match pe v with
    | .error e => .error e
    | .ok p =>
      if isTrue p then
        match pj v with
        | .error e => .error e
        | .ok y => mainCompLoop (acc ++ [y]) vs
      else mainCompLoop acc vs

/-- *main* eval.rs:1010-1025 (and the PR's short-list loop, eval.rs:1325-1340). -/
def mainQuantLoop : Nat → Nat → Nat → List (V F) → Except String (Nat × Nat × Nat)
  | t, f, n, [] => .ok (t, f, n)
  | t, f, n, v :: vs =>
    match pe v with
    | .error e => .error e
    | .ok (.bool true) => mainQuantLoop (t + 1) f n vs
    | .ok (.bool false) => mainQuantLoop t (f + 1) n vs
    | .ok .null => mainQuantLoop t f (n + 1) vs
    | .ok w => .error (boolMismatch w)

theorem ref_nil (s : Sink F) : ref pe pj s [] = .ok s := rfl

theorem ref_cons (s : Sink F) (v : V F) (l : List (V F)) :
    ref pe pj s (v :: l) = (refStep pe pj s v >>= fun s' => ref pe pj s' l) := by
  simp [ref, List.foldlM_cons]

theorem ref_append (s : Sink F) (a b : List (V F)) :
    ref pe pj s (a ++ b) = (ref pe pj s a >>= fun s' => ref pe pj s' b) := by
  simp [ref, List.foldlM_append]

theorem ref_collect_eq_main : ∀ (l acc : List (V F)),
    ref pe pj (.collect acc) l = (mainCompLoop pe pj acc l).map .collect
  | [], acc => rfl
  | v :: vs, acc => by
    rw [ref_cons]; simp only [refStep, mainCompLoop]
    cases hp : pe v with
    | error e => rfl
    | ok p =>
      simp only [bind, Except.bind]
      by_cases ht : isTrue p = true
      · simp only [ht, ite_true]
        cases hj : pj v with
        | error e => rfl
        | ok y => simp only [pure, Except.pure]; exact ref_collect_eq_main vs _
      · simp only [ht, Bool.false_eq_true, ite_false, pure, Except.pure]
        exact ref_collect_eq_main vs _

theorem ref_count_eq_main : ∀ (l : List (V F)) (t f n : Nat),
    ref pe pj (.count t f n) l = (mainQuantLoop pe t f n l).map fun (t, f, n) => .count t f n
  | [], t, f, n => rfl
  | v :: vs, t, f, n => by
    rw [ref_cons]; simp only [refStep, mainQuantLoop]
    cases hp : pe v with
    | error e => rfl
    | ok p =>
      simp only [bind, Except.bind]
      rcases p with _ | ⟨_ | _⟩ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ <;>
        simp only [kOf, bump, pure, Except.pure, throw, throwThe, MonadExceptOf.throw,
          Except.map] <;> exact ref_count_eq_main vs _ _ _

/-- The size main reports is the length of the list main builds: counting with `tally`
evaluates exactly what collecting does. -/
theorem ref_tally_eq_collect : ∀ (l acc : List (V F)) (k : Nat),
    ref pe pj (.tally k) l =
      (mainCompLoop pe pj acc l).map fun ys => .tally (k + (ys.length - acc.length))
  | [], acc, k => by simp [ref_nil, mainCompLoop, Except.map]
  | v :: vs, acc, k => by
    rw [ref_cons]; simp only [refStep, mainCompLoop]
    cases hp : pe v with
    | error e => rfl
    | ok p =>
      simp only [bind, Except.bind]
      by_cases ht : isTrue p = true
      · simp only [ht, ite_true]
        cases hj : pj v with
        | error e => rfl
        | ok y =>
          simp only [pure, Except.pure]
          rw [ref_tally_eq_collect vs (acc ++ [y]) (k + 1)]
          cases hm : mainCompLoop pe pj (acc ++ [y]) vs with
          | error e => rfl
          | ok ys =>
            have h1 := mainCompLoop_len vs (acc ++ [y]) ys hm
            simp only [List.length_append, List.length_singleton] at h1
            simp only [Except.map, Except.ok.injEq, Sink.tally.injEq, List.length_append,
              List.length_singleton]; omega
      · simp only [ht, Bool.false_eq_true, ite_false, pure, Except.pure]
        exact ref_tally_eq_collect vs acc k
where
  mainCompLoop_len : ∀ (l acc ys : List (V F)), mainCompLoop pe pj acc l = .ok ys →
      acc.length ≤ ys.length
    | [], acc, ys, h => by simp [mainCompLoop] at h; subst h; omega
    | v :: vs, acc, ys, h => by
      simp only [mainCompLoop] at h
      cases hp : pe v with
      | error e => simp [hp] at h
      | ok p =>
        simp only [hp] at h
        by_cases ht : isTrue p = true
        · simp only [ht, ite_true] at h
          cases hj : pj v with
          | error e => simp [hj] at h
          | ok y =>
            simp only [hj] at h
            have := mainCompLoop_len vs _ ys h; simp at this; omega
        · simp only [ht, Bool.false_eq_true, ite_false] at h
          exact mainCompLoop_len vs acc ys h

/-- **First error in element order.** If main's loop fails, it fails at the first element
whose step fails, every earlier element having succeeded. -/
theorem ref_error_first : ∀ (l : List (V F)) (s : Sink F) (e : String),
    ref pe pj s l = .error e →
      ∃ pre x post s0, l = pre ++ x :: post ∧ ref pe pj s pre = .ok s0 ∧ refStep pe pj s0 x = .error e
  | [], s, e, h => by simp [ref_nil] at h
  | x :: xs, s, e, h => by
    rw [ref_cons] at h
    cases hx : refStep pe pj s x with
    | error e' =>
      rw [hx] at h; simp only [bind, Except.bind, Except.error.injEq] at h; subst h
      exact ⟨[], x, xs, s, rfl, rfl, hx⟩
    | ok s1 =>
      rw [hx] at h; simp only [bind, Except.bind] at h
      obtain ⟨pre, y, post, s0, hl, hp, hy⟩ := ref_error_first xs s1 e h
      refine ⟨x :: pre, y, post, s0, by simp [hl], ?_, hy⟩
      rw [ref_cons, hx]; exact hp

/-- Within one element of a comprehension: the predicate's error, or — only when the
predicate passed — the projection's. -/
theorem refStep_collect_error (acc : List (V F)) (v : V F) (e : String) :
    refStep pe pj (.collect acc) v = .error e ↔
      pe v = .error e ∨ ∃ p, pe v = .ok p ∧ isTrue p = true ∧ pj v = .error e := by
  simp only [refStep]
  cases hp : pe v with
  | error e' => simp [bind, Except.bind]
  | ok p =>
    simp only [bind, Except.bind]
    by_cases ht : isTrue p = true
    · simp only [ht, ite_true]
      cases pj v <;> simp [pure, Except.pure, ht]
    · simp [ht, pure, Except.pure]

end

/-! ## The PR's per-element path -/

/-- `ListLoop` after `ListLoop::new`; the evaluators are the per-element evaluation of the
predicate / projection *nodes* (`eval_node` with the loop variable bound). -/
structure Body (F : Type) where
  hasPred : Bool
  hasProj : Bool
  pe : V F → Except String (V F)
  pj : V F → Except String (V F)

/-- What `ListLoop::new` relies on when it drops a node (eval.rs:328-331): the literal
`true` evaluates to `true`, and the loop variable evaluates to the element. These are
`ExprEval::eval_node` on `ExprIR::Constant` / `ExprIR::Variable` (an `ExprEval` fact;
part of the `RowAgree` hypothesis of the root report). -/
structure Body.Drops (b : Body F) : Prop where
  predTrue : b.hasPred = false → ∀ v, b.pe v = .ok (.bool true)
  projId : b.hasProj = false → ∀ v, b.pj v = .ok v

/-- The loop body of eval.rs:1513-1549. -/
def prStep (b : Body F) (s : Sink F) (v : V F) : Except String (Sink F) :=
  (if b.hasPred then b.pe v else pure (.bool true)) >>= fun pass =>
  match s with
  | .collect acc =>
    if isTrue pass then
      (if b.hasProj then b.pj v else pure v) >>= fun y => pure (.collect (acc ++ [y]))
    else pure (.collect acc)
  | .tally k =>
    if isTrue pass then
      (if b.hasProj then b.pj v else pure v) >>= fun _ => pure (.tally (k + 1))
    else pure (.tally k)
  | .count t f n =>
    match kOf pass with
    | some k => pure (bump k t f n)
    | none => throw (boolMismatch pass)

/-- eval.rs:1504-1511: no predicate and no projection. -/
def bulk : Sink F → List (V F) → Sink F
  | .collect acc, c => .collect (acc ++ c)
  | .count t f n, c => .count (t + c.length) f n
  | .tally k, c => .tally (k + c.length)

def perElem (b : Body F) (s : Sink F) (chunk : List (V F)) : Except String (Sink F) :=
  if !b.hasPred && !b.hasProj then pure (bulk s chunk) else chunk.foldlM (prStep b) s

variable (b : Body F)

theorem prStep_eq (hd : b.Drops) (s : Sink F) (v : V F) : prStep b s v = refStep b.pe b.pj s v := by
  have hp : (if b.hasPred then b.pe v else pure (.bool true)) = b.pe v := by
    cases h : b.hasPred
    · simp [hd.predTrue h, pure, Except.pure]
    · rfl
  have hj : (if b.hasProj then b.pj v else pure v) = b.pj v := by
    cases h : b.hasProj
    · simp [hd.projId h, pure, Except.pure]
    · rfl
  unfold prStep; rw [hp, hj]
  cases s <;> rfl

theorem bulk_eq (hd : b.Drops) (hp : b.hasPred = false) (hj : b.hasProj = false) :
    ∀ (c : List (V F)) (s : Sink F), ref b.pe b.pj s c = .ok (bulk s c)
  | [], s => by cases s <;> simp [ref_nil, bulk]
  | v :: vs, s => by
    rw [ref_cons]
    cases s with
    | collect acc =>
      simp only [refStep, hd.predTrue hp v, hd.projId hj v, isTrue, bind, Except.bind, ite_true,
        pure, Except.pure]
      rw [bulk_eq hd hp hj vs]; simp [bulk]
    | count t f n =>
      simp only [refStep, hd.predTrue hp v, kOf, bump, bind, Except.bind, pure, Except.pure]
      rw [bulk_eq hd hp hj vs]; simp [bulk]; omega
    | tally k =>
      simp only [refStep, hd.predTrue hp v, hd.projId hj v, isTrue, bind, Except.bind, ite_true,
        pure, Except.pure]
      rw [bulk_eq hd hp hj vs]; simp [bulk]; omega

/-- **The per-element path is main's loop.** -/
theorem perElem_eq (hd : b.Drops) (s : Sink F) (c : List (V F)) :
    perElem b s c = ref b.pe b.pj s c := by
  unfold perElem
  split
  · rename_i h
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
    rw [bulk_eq b hd h.1 h.2]; rfl
  · have : (fun s v => prStep b s v) = refStep b.pe b.pj := by
      funext s v; exact prStep_eq b hd s v
    simp only [ref]; rw [show prStep b = refStep b.pe b.pj from this]

end Pr3437.ListLoop
