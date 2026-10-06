/-
# `Runtime::run_batch`: the iterative post-order operator-tree build

| here | there |
| --- | --- |
| `T`                 | the IR subtree, as the children `children_to_recurse` returns |
| `K`, `childrenToRecurse` | `Runtime::children_to_recurse`, `runtime/runtime.rs:652-674` |
| `Frame`, `runM`     | the `work`/`built` stack machine of `run_batch`, `runtime/runtime.rs:710-739` |
| `runBatch`          | `Runtime::run_batch` incl. the final `built.pop().ok_or_else(..)` |
| `recT`, `recL`      | the recursive build it replaced (reference): build the children left to right, then `build_batch_op(cur, kids)` |
| `mk`                | `build_batch_op` (`runtime.rs:743-1307`), abstract: any `Except`-valued constructor |

`built` is a `Vec` (push at the end); `built.drain(len - n..)` takes the last `n` in push
order. `mk` is arbitrary, so the theorem holds for every operator and error.
-/
namespace RuntimeCore.RunBatch

inductive T where
  | node (k : Nat) (kids : List T)

variable {B : Type}

mutual
/-- Recursive reference build. -/
def recT (mk : Nat → List B → Except String B) : T → Except String B
  | .node k ks =>
    match recL mk ks with
    | .ok bs => mk k bs
    | .error e => .error e
def recL (mk : Nat → List B → Except String B) : List T → Except String (List B)
  | [] => .ok []
  | t :: ts =>
    match recT mk t with
    | .error e => .error e
    | .ok b =>
      match recL mk ts with
      | .ok bs => .ok (b :: bs)
      | .error e => .error e
end

inductive Frame where
  | pre (t : T)
  | post (k : Nat) (n : Nat)

/-- The `while let Some(f) = work.pop()` loop, with fuel. -/
def runM (mk : Nat → List B → Except String B) : Nat → List Frame → List B → Except String (List B)
  | 0, _, built => .ok built
  | _ + 1, [], built => .ok built
  | fuel + 1, .pre (.node k ks) :: w, built =>
    runM mk fuel (ks.map .pre ++ .post k ks.length :: w) built
  | fuel + 1, .post k n :: w, built =>
    let start := built.length - n
    match mk k (built.drop start) with
    | .ok op => runM mk fuel w (built.take start ++ [op])
    | .error e => .error e

mutual
def steps : T → Nat
  | .node _ ks => stepsL ks + 2
def stepsL : List T → Nat
  | [] => 0
  | t :: ts => steps t + stepsL ts
end

/-- `run_batch` with enough fuel (`steps t`), then `built.pop()`. -/
def runBatch (mk : Nat → List B → Except String B) (t : T) : Except String B :=
  match runM mk (steps t) [.pre t] [] with
  | .ok built =>
    match built.getLast? with
    | some b => .ok b
    | none => .error "empty plan tree"
  | .error e => .error e

theorem recL_length (mk : Nat → List B → Except String B) :
    ∀ (ts : List T) (bs : List B), recL mk ts = .ok bs → bs.length = ts.length
  | [], bs, h => by simp [recL] at h; subst h; rfl
  | t :: ts, bs, h => by
    simp only [recL] at h
    split at h
    · simp at h
    · next b _ =>
      split at h
      · next bs' h' =>
        simp at h; subst h
        simp [recL_length mk ts bs' h']
      · simp at h

mutual
theorem runM_T (mk : Nat → List B → Except String B) :
    ∀ (t : T) (w : List Frame) (built : List B) (k : Nat),
      runM mk (steps t + k) (.pre t :: w) built =
        match recT mk t with
        | .ok b => runM mk k w (built ++ [b])
        | .error e => .error e
  | .node kk ks, w, built, k => by
    have e : steps (.node kk ks) + k = (stepsL ks + (k + 1)) + 1 := by simp [steps]; omega
    rw [e]
    simp only [runM]
    rw [runM_L mk ks (.post kk ks.length :: w) built (k + 1)]
    simp only [recT]
    split
    · next bs hbs =>
      have hl := recL_length mk ks bs hbs
      simp only [runM, List.length_append, hl, Nat.add_sub_cancel, List.drop_left, List.take_left]
      try (split <;> simp_all)
    · rfl
theorem runM_L (mk : Nat → List B → Except String B) :
    ∀ (ts : List T) (w : List Frame) (built : List B) (k : Nat),
      runM mk (stepsL ts + k) (ts.map .pre ++ w) built =
        match recL mk ts with
        | .ok bs => runM mk k w (built ++ bs)
        | .error e => .error e
  | [], w, built, k => by simp [stepsL, recL]
  | t :: ts, w, built, k => by
    have e : stepsL (t :: ts) + k = steps t + (stepsL ts + k) := by simp [stepsL]; omega
    rw [e, List.map_cons, List.cons_append, runM_T mk t (List.map Frame.pre ts ++ w) built (stepsL ts + k)]
    simp only [recL]
    cases recT mk t with
    | error e => rfl
    | ok b =>
      simp only
      rw [runM_L mk ts w (built ++ [b]) k]
      cases recL mk ts <;> simp
end

/-- **PROVEN** (headline): the iterative post-order `run_batch` builds exactly what the
recursive build would — same operator tree, same first error — for every plan tree and
every `build_batch_op`. The heap-allocated work stack is a pure stack-safety change. -/
theorem runBatch_eq_rec (mk : Nat → List B → Except String B) (t : T) :
    runBatch mk t = recT mk t := by
  simp only [runBatch]
  have := runM_T mk t [] [] 0
  simp only [Nat.add_zero] at this
  rw [this]
  cases recT mk t <;> simp [runM]

/-- **PROVEN**: `built.pop()` never hits `"empty plan tree"` — the machine always leaves
exactly one operator when the build succeeds. -/
theorem runBatch_ok_single (mk : Nat → List B → Except String B) (t : T) (b : B)
    (h : recT mk t = .ok b) : runM mk (steps t) [.pre t] [] = .ok [b] := by
  have := runM_T mk t [] [] 0
  simp only [Nat.add_zero, h] at this
  rw [this]; simp [runM]

/-! ### `children_to_recurse` (`runtime.rs:652-674`) -/

inductive K where
  | union | argument | createIndex | dropIndex | cartesianProduct | valueHashJoin
  | optional | merge | forEach | other

/-- Which IR children `run_batch` builds before the node itself. `ValueHashJoin` takes
children 0 and 1 (orx `child(i)` panics if absent; the planner always gives it two).
`Optional`/`Merge`/`ForEach` with a single child treat that child as their body (built
on demand), so nothing is pre-built. -/
def childrenToRecurse {α : Type} : K → List α → List α
  | .union, _ | .argument, _ | .createIndex, _ | .dropIndex, _ => []
  | .cartesianProduct, cs => cs
  | .valueHashJoin, cs => cs.take 2
  | .optional, cs | .merge, cs | .forEach, cs => if cs.length > 1 then cs.take 1 else []
  | .other, cs => cs.take 1

example : childrenToRecurse .forEach [1] = ([] : List Nat) ∧
    childrenToRecurse .forEach [1, 2] = [1] ∧ childrenToRecurse .other ([] : List Nat) = [] ∧
    childrenToRecurse .cartesianProduct [1, 2, 3] = [1, 2, 3] := by decide

/-- **`children_to_recurse`**: no pre-built children for `Union`/`Argument`/index DDL; all of
them for `CartesianProduct`; the two inputs of `ValueHashJoin`; the input (child 0) of
`Optional`/`Merge`/`ForEach` only when a separate body child exists; else child 0 if any. -/
theorem childrenToRecurse_spec {α : Type} (cs : List α) (a b : α) (rest : List α) :
    childrenToRecurse .union cs = [] ∧ childrenToRecurse .argument cs = [] ∧
    childrenToRecurse .cartesianProduct cs = cs ∧
    childrenToRecurse .valueHashJoin (a :: b :: rest) = [a, b] ∧
    childrenToRecurse .optional [a] = [] ∧ childrenToRecurse .merge (a :: b :: rest) = [a] ∧
    childrenToRecurse .other (a :: rest) = [a] ∧ childrenToRecurse .other ([] : List α) = [] := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_, rfl, rfl⟩
  simp [childrenToRecurse]

end RuntimeCore.RunBatch
