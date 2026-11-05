/-
# UNION / UNION ALL — `graph/src/runtime/ops/union.rs`

`UnionOp` concatenates its children's streams: each branch is instantiated
lazily (`run_batch`) once the previous one is exhausted.  `UNION` (distinct) is
planned as `Distinct` over this operator (`planner/mod.rs:3302`), so the operator
itself is always the `ALL` form.

State refinement: Rust keeps `current_child : usize` and indexes
`plan.node(idx).child(current_child)`; the model keeps the *suffix* of children
not yet started (`rest`), i.e. `children.drop (current_child + 1)` while a branch
is being drained and `children.drop current_child` otherwise.  `run_batch` is a
child that is either `.error e` (plan instantiation failed) or `.ok items` (the
iterator's full item sequence; items are themselves `Result`s and are passed
through untouched, `union.rs:71-73`).
-/

namespace Union

variable {ε β : Type}

/-- One child as `run_batch` sees it. -/
abbrev Child (ε β : Type) := Except ε (List (Except ε β))

structure St (ε β : Type) where
  current : Option (List (Except ε β))
  rest : List (Child ε β)

/-- `UnionOp::next` (`union.rs:68-96`), the `loop` unrolled by well-founded
recursion. -/
def next : St ε β → Option (Except ε β) × St ε β
  | ⟨some (x :: xs), rest⟩ => (some x, ⟨some xs, rest⟩)                 -- 71-72
  | ⟨some [], rest⟩ => next ⟨none, rest⟩                                -- 74-75
  | ⟨none, []⟩ => (none, ⟨none, []⟩)                                     -- 78-79
  | ⟨none, .error e :: _⟩ => (some (.error e), ⟨none, []⟩)             -- 90-92
  | ⟨none, .ok items :: cs⟩ => next ⟨some items, cs⟩                    -- 83-88
termination_by st => 2 * st.rest.length + (if st.current.isSome then 1 else 0)

/-- What the operator will still emit: the current branch's remaining items,
then each later branch in order, stopping after the first failed `run_batch`. -/
def stream : List (Child ε β) → List (Except ε β)
  | [] => []
  | .ok items :: cs => items ++ stream cs
  | .error e :: _ => [.error e]

def pendingItems (st : St ε β) : List (Except ε β) :=
  st.current.getD [] ++ stream st.rest

/-- `next` returns exactly the head of the pending stream and leaves its tail. -/
theorem next_spec : ∀ st : St ε β,
    match pendingItems st with
    | [] => (next st).1 = none ∧ pendingItems (next st).2 = []
    | x :: xs => (next st).1 = some x ∧ pendingItems (next st).2 = xs
  | ⟨some (x :: xs), rest⟩ => by simp [next, pendingItems]
  | ⟨some [], rest⟩ => by
    have ih := next_spec (⟨none, rest⟩ : St ε β)
    simp only [pendingItems, Option.getD_some, List.nil_append] at ih ⊢
    rw [next]; simpa [pendingItems] using ih
  | ⟨none, []⟩ => by simp [next, pendingItems, stream]
  | ⟨none, .error e :: _⟩ => by simp [next, pendingItems, stream]
  | ⟨none, .ok items :: cs⟩ => by
    have ih := next_spec (⟨some items, cs⟩ : St ε β)
    simp only [pendingItems, Option.getD_some, Option.getD_none, List.nil_append, stream] at ih ⊢
    rw [next]; simpa [pendingItems] using ih
termination_by st => 2 * st.rest.length + (if st.current.isSome then 1 else 0)

/-- Draining the iterator with enough fuel (`fuel > |pending|`). -/
def drain : Nat → St ε β → List (Except ε β)
  | 0, _ => []
  | fuel + 1, st => match next st with
    | (none, _) => []
    | (some x, st') => x :: drain fuel st'

theorem drain_eq (st : St ε β) (fuel : Nat) (h : (pendingItems st).length < fuel) :
    drain fuel st = pendingItems st := by
  induction fuel generalizing st with
  | zero => simp at h
  | succ n ih =>
    have hs := next_spec st
    unfold drain
    generalize hp : pendingItems st = p at hs h
    cases p with
    | nil =>
      obtain ⟨h1, _⟩ := hs
      rcases hn : next st with ⟨o, st'⟩
      rw [hn] at h1; simp only at h1; subst h1; rfl
    | cons x xs =>
      obtain ⟨h1, h2⟩ := hs
      rcases hn : next st with ⟨o, st'⟩
      rw [hn] at h1 h2; simp only [Option.some.injEq] at h1; subst h1; simp only
      rw [ih _ (by rw [h2]; simp at h; omega), h2]

/-- **UNION ALL = concatenation.**  Starting state of a fresh `UnionOp`
(`union.rs:43-54`: `current = None`, `current_child = 0`): the operator emits
every branch's items in branch order, and a failing `run_batch` emits its error
and ends the stream (`union.rs:90-92`). -/
theorem union_drain (children : List (Child ε β)) (fuel : Nat)
    (h : (stream children).length < fuel) :
    drain fuel ⟨none, children⟩ = stream children := by
  have := drain_eq ⟨none, children⟩ fuel (by simpa [pendingItems] using h)
  simpa [pendingItems] using this

/-- With every branch instantiable, the output is exactly the concatenation. -/
theorem stream_all_ok (bs : List (List (Except ε β))) :
    stream (bs.map Except.ok) = bs.flatten := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [stream, ih]

/-- Once `None` is returned the operator stays exhausted (fused), as the
`current_child >= num_children` check (`union.rs:78`) guarantees. -/
theorem next_none_stays (st : St ε β) (h : (next st).1 = none) :
    (next (next st).2).1 = none := by
  have hs := next_spec st
  generalize hp : pendingItems st = p at hs
  cases p with
  | nil =>
    have h2 := next_spec (next st).2
    rw [hs.2] at h2; exact h2.1
  | cons x xs => rw [hs.1] at h; cases h

end Union
