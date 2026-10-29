/-
# `CommitOp::next` (graph/src/runtime/ops/commit.rs:69), after #2846

The first call drains the child (:71-82), then runs, each fallible step
short-circuiting with `Some(Err(..))`: `Pending::commit` (:85-92), the writer
upgrade (:96-98), the deferred-index publish (:103), the effects block
(:105-131), and — new in #2846 — `Pending::end_segment` (:135-143), which both
resets the segment's mutation log and rolls the graph's id batches (the old
`clear()` + `open_id_boundaries()` pair). Then the schema baseline (:146-149)
and `results.reverse()` (:151). Every call ends with `results.pop()` (:155).

The runtime is abstract (`S`, with the six steps as functions), batches are `B`.

* `commitNext_ok`: on success the state is exactly
  `baseline ∘ endSegment ∘ effects ∘ publish ∘ upgrade ∘ commit` — `end_segment`
  runs once, after the commit and the effects buffer, before the baseline.
* `commitNext_commit_err`: a failed `Pending::commit` returns its error and
  runs nothing after it (in particular no `end_segment`).
* `commitNext_endSegment_err`: an `end_segment` refusal is returned as the
  operator's error and the baseline is not reset.
* `pops_in_order`: non-root, the drained batches are yielded in child order.
-/
namespace PendingCommit.CommitOpNext

structure Env (S : Type) where
  commit : S → Except String S
  upgrade : S → Except String S
  publish : S → S
  effects : S → Except String S
  endSegment : S → Except String S
  baseline : S → S

/-- The drain loop (:71-82): batches kept unless root, first error returned. -/
def drain {B : Type} (isRoot : Bool) : List (Except String B) → List B → Except String (List B)
  | [], acc => .ok acc
  | .error e :: _, _ => .error e
  | .ok b :: xs, acc => drain isRoot xs (if isRoot then acc else acc ++ [b])

/-- The first-call body (:70-152): `.error e` is the `return Some(Err(e))`
exits; `.ok (results, s)` is the state before the final `pop`. -/
def firstCall {S B : Type} (E : Env S) (isRoot : Bool) (child : List (Except String B))
    (results : List B) (s : S) : Except String (List B × S) := do
  let rs ← drain isRoot child results
  let s1 ← E.commit s
  let s2 ← E.upgrade s1
  let s3 := E.publish s2
  let s4 ← E.effects s3
  let s5 ← E.endSegment s4
  let s6 := E.baseline s5
  pure (rs.reverse, s6)

/-- `next` (:69): first call when the child is still there, then `pop`. -/
def commitNext {S B : Type} (E : Env S) (isRoot : Bool) (child : Option (List (Except String B)))
    (results : List B) (s : S) : Option (Except String B) × List B × S :=
  match child with
  | some xs =>
    match firstCall E isRoot xs results s with
    | .error e => (some (.error e), results, s)
    | .ok (rs, s') => match rs.getLast? with
      | some b => (some (.ok b), rs.dropLast, s')
      | none => (none, [], s')
  | none => match results.getLast? with
    | some b => (some (.ok b), results.dropLast, s)
    | none => (none, [], s)

theorem drain_ok_nonroot {B : Type} : ∀ (xs : List B) (acc : List B),
    drain false (xs.map .ok) acc = .ok (acc ++ xs)
  | [], acc => by simp [drain]
  | x :: xs, acc => by simp [drain, drain_ok_nonroot xs]

theorem commitNext_ok {S B : Type} (E : Env S) (isRoot : Bool) (xs : List (Except String B))
    (results rs : List B) (s s' : S) (h : firstCall E isRoot xs results s = .ok (rs, s')) :
    ∃ s1 s2 s4 s5, E.commit s = .ok s1 ∧ E.upgrade s1 = .ok s2 ∧ E.effects (E.publish s2) = .ok s4 ∧
      E.endSegment s4 = .ok s5 ∧ s' = E.baseline s5 := by
  unfold firstCall at h
  cases hd : drain isRoot xs results with
  | error e => simp [hd, bind, Except.bind] at h
  | ok r0 =>
    cases h1 : E.commit s with
    | error e => simp [hd, h1, bind, Except.bind] at h
    | ok s1 =>
      cases h2 : E.upgrade s1 with
      | error e => simp [hd, h1, h2, bind, Except.bind] at h
      | ok s2 =>
        cases h4 : E.effects (E.publish s2) with
        | error e => simp [hd, h1, h2, h4, bind, Except.bind] at h
        | ok s4 =>
          cases h5 : E.endSegment s4 with
          | error e => simp [hd, h1, h2, h4, h5, bind, Except.bind] at h
          | ok s5 =>
            simp [hd, h1, h2, h4, h5, bind, Except.bind, pure, Except.pure] at h
            exact ⟨s1, s2, s4, s5, by first | rfl | assumption, by first | rfl | assumption, by first | rfl | assumption, by first | rfl | assumption, h.2.symm⟩

theorem commitNext_commit_err {S B : Type} (E : Env S) (isRoot : Bool) (xs : List (Except String B))
    (results r0 : List B) (s : S) (e : String) (hd : drain isRoot xs results = .ok r0)
    (h : E.commit s = .error e) :
    commitNext E isRoot (some xs) results s = (some (.error e), results, s) := by
  simp [commitNext, firstCall, hd, h, bind, Except.bind]

theorem commitNext_endSegment_err {S B : Type} (E : Env S) (isRoot : Bool) (xs : List (Except String B))
    (results r0 : List B) (s s1 s2 s4 : S) (e : String) (hd : drain isRoot xs results = .ok r0)
    (h1 : E.commit s = .ok s1) (h2 : E.upgrade s1 = .ok s2) (h4 : E.effects (E.publish s2) = .ok s4)
    (h5 : E.endSegment s4 = .error e) :
    commitNext E isRoot (some xs) results s = (some (.error e), results, s) := by
  simp [commitNext, firstCall, hd, h1, h2, h4, h5, bind, Except.bind]

/-- Popping a reversed list yields the original order. -/
def popAll {B : Type} : Nat → List B → List B
  | 0, _ => []
  | n + 1, l => match l.getLast? with
    | some b => b :: popAll n l.dropLast
    | none => []

theorem popAll_reverse {B : Type} : ∀ (xs : List B), popAll xs.length xs.reverse = xs
  | [] => rfl
  | x :: xs => by
    simp only [List.length_cons, popAll, List.reverse_cons, List.getLast?_append,
      List.getLast?_singleton, Option.some_or, List.dropLast_concat]
    rw [popAll_reverse xs]

/-- **`pops_in_order`**: non-root, every drained batch comes out in child order. -/
theorem pops_in_order {S B : Type} (E : Env S) (bs : List B) (s s' : S) (rs : List B)
    (h : firstCall E false (bs.map .ok) [] s = .ok (rs, s')) :
    rs = bs.reverse ∧ popAll bs.length rs = bs := by
  unfold firstCall at h
  rw [drain_ok_nonroot] at h
  cases h1 : E.commit s <;> simp [h1, bind, Except.bind] at h
  rename_i s1
  cases h2 : E.upgrade s1 <;> simp [h2] at h
  rename_i s2
  cases h4 : E.effects (E.publish s2) <;> simp [h4] at h
  rename_i s4
  cases h5 : E.endSegment s4 <;> simp [h5, pure, Except.pure] at h
  obtain ⟨rfl, -⟩ := h
  exact ⟨by simp, by simpa using popAll_reverse bs⟩

end PendingCommit.CommitOpNext
