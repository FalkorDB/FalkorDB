import FalkorOptimizer.Basic
/-
# `ExprIR::and_of` / `or_of` keep the runtime value (ast.rs:342-380, #3335)

Every filter rebuilt by the optimizer passes since #3335 goes through
`ExprIR::and_of` (absorb_edge_filters_into_vlt.rs:139/145, eliminate_true_filters.rs:135,
push_filters_down.rs:130/347/360/423, replace_cartesian_with_hash_join.rs:208,
utilize_index.rs:1008), which splices an AND operand into the new AND. This file
shows that is invisible at run time, errors included.

| here | there |
| --- | --- |
| `scL sc` | the `ExprIR::And` (`sc = false`) / `ExprIR::Or` (`sc = true`) loops eval.rs:629-699, over the children's results left to right: an error or a non-boolean stops it, `sc` short-circuits, a null is remembered |
| `T`, `evalT` | a bound expression as `flat` sees it: AND/OR nodes, everything else by its value on the row |
| `flatT` | `ExprIR::flat` (ast.rs:355-380): early return of a lone other operand, splice, identity / single / node |

* `andLoop_eq`: Basic's `andLoop` (the model every pass theorem uses) is `scL false`.
* `scL_splice`: splicing a nested run of the same connective never changes the loop's result
  (associativity, with errors, nulls and short-circuit).
* `flat_eval`: `flat` evaluates like the unflattened node whenever more than one operand is left;
  `flat_single`, `flat_single_same`: the lone-operand cases return the operand itself (the
  pre-#3335 `remaining.len() == 1` branch).
* `flat_passes`: as a Filter predicate, `and_of xs` keeps / drops / errors on a row exactly as
  `AND(xs)` does — unconditionally (a lone non-boolean operand is a type error either way).
-/
namespace Falkor.Opt.AndOf
open Falkor.Opt

/-- The AND (`sc = false`) / OR (`sc = true`) loop over the children's results. -/
def scL (sc : Bool) : List (Except Err Val) → Bool → Except Err Val
  | [], isNull => .ok (if isNull then .null else .bool (!sc))
  | .error e :: _, _ => .error e
  | .ok (.bool b) :: rs, n => if b = sc then .ok (.bool sc) else scL sc rs n
  | .ok .null :: rs, _ => scL sc rs true
  | .ok _ :: _, _ => .error .typeMismatch

theorem andLoop_eq (r : Row) (cs : List Expr) (n : Bool) : andLoop r cs n = scL false (cs.map (eval r)) n := by
  induction cs generalizing n with
  | nil => rfl
  | cons c cs ih =>
    simp only [andLoop, List.map_cons]
    rcases eval r c with e | (_ | b | i | s)
    · rfl
    · exact ih true
    · cases b
      · rfl
      · exact ih n
    · rfl
    · rfl

/-- What the rest of the loop does with the result of a prefix. -/
def cont (sc : Bool) (l2 : List (Except Err Val)) : Except Err Val → Except Err Val
  | .error e => .error e
  | .ok (.bool b) => if b = sc then .ok (.bool sc) else scL sc l2 false
  | .ok .null => scL sc l2 true
  | .ok _ => .error .typeMismatch

theorem scL_append (sc : Bool) (l2 : List (Except Err Val)) :
    ∀ (l1 : List (Except Err Val)) (n : Bool), scL sc (l1 ++ l2) n = cont sc l2 (scL sc l1 n)
  | [], n => by cases n <;> cases sc <;> simp [scL, cont]
  | .error e :: l1, n => rfl
  | .ok (.bool b) :: l1, n => by
    simp only [List.cons_append, scL]
    split
    · simp [cont]
    · exact scL_append sc l2 l1 n
  | .ok .null :: l1, n => scL_append sc l2 l1 true
  | .ok (.int _) :: l1, n => rfl
  | .ok (.str _) :: l1, n => rfl

/-- A remembered null only turns the neutral result into null. -/
theorem scL_null (sc : Bool) : ∀ (l : List (Except Err Val)),
    scL sc l true = match scL sc l false with
      | .ok (.bool b) => if b = sc then .ok (.bool sc) else .ok .null
      | x => x
  | [] => by cases sc <;> simp [scL]
  | .error e :: l => rfl
  | .ok (.bool b) :: l => by
    simp only [scL]
    split
    · simp_all
    · exact scL_null sc l
  | .ok .null :: l => by
    simp only [scL]
    rw [scL_null sc l]
    rcases scL sc l false with e | (_ | b | i | s)
    · rfl
    · rfl
    · by_cases h : b = sc <;> simp [h]
    · rfl
    · rfl
  | .ok (.int _) :: l => rfl
  | .ok (.str _) :: l => rfl

/-- **Associativity**: a nested run of the same connective, evaluated as one
operand, gives the same result as its operands spliced in place. -/
theorem scL_splice (sc : Bool) (l1 l2 : List (Except Err Val)) (n : Bool) :
    scL sc (scL sc l1 false :: l2) n = scL sc (l1 ++ l2) n := by
  rw [scL_append]
  cases n
  · rcases h : scL sc l1 false with e | (_ | b | i | s)
    · rfl
    · rfl
    · simp only [scL, cont]
    · rfl
    · rfl
  · rw [scL_null sc l1]
    rcases scL sc l1 false with e | (_ | b | i | s)
    · rfl
    · rfl
    · by_cases hb : b = sc
      · simp [scL, cont, hb]
      · simp [scL, cont, hb]
    · rfl
    · rfl

/-! ## Trees and `flat` -/

inductive T
  /-- any operand that is not an AND/OR node, by its value on the row -/
  | leaf (v : Except Err Val)
  /-- `And` (`isOr = false`) / `Or` (`isOr = true`) -/
  | conn (isOr : Bool) (cs : List T)

mutual
def evalT : T → Except Err Val
  | .leaf v => v
  | .conn o cs => scL o (evalTL cs) false
def evalTL : List T → List (Except Err Val)
  | [] => []
  | c :: cs => evalT c :: evalTL cs
end

theorem evalTL_map (l : List T) : evalTL l = l.map evalT := by
  induction l with
  | nil => rfl
  | cons c cs ih => simp [evalTL, ih]

def splice (o : Bool) : T → List T
  | .conn o' cs => if o' = o then cs else [.conn o' cs]
  | t => [t]

def body (o : Bool) (ts : List T) : T :=
  match ts.flatMap (splice o) with
  | [] => .leaf (.ok (.bool (!o)))
  | [c] => c
  | cs => .conn o cs

/-- `ExprIR::flat`; `and_of = flatT false`, `or_of = flatT true`. -/
def flatT (o : Bool) (ts : List T) : T :=
  match ts with
  | [.conn o' cs] => if o' = o then body o ts else .conn o' cs
  | [t] => t
  | ts => body o ts

theorem flatT_eq_body (o : Bool) (ts : List T) : flatT o ts = body o ts := by
  unfold flatT
  split
  · rename_i o' cs
    split
    · rfl
    · rename_i h; simp [body, splice, h]
  · rename_i t h; cases t with
    | leaf v => rfl
    | conn o' cs => exact absurd rfl (h o' cs)
  · rfl

/-- The lone-operand case: an operand of another kind is returned as it is. -/
theorem flat_single (o : Bool) (v : Except Err Val) : flatT o [.leaf v] = .leaf v := rfl

theorem flat_single_same (o : Bool) (cs : List T) (h : 2 ≤ cs.length) :
    flatT o [.conn o cs] = .conn o cs := by
  rw [flatT_eq_body]; unfold body
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, splice, ↓reduceIte]
  match cs, h with
  | _ :: _ :: _, _ => rfl

/-- AND under OR (OR under AND) is one operand: never spliced. -/
theorem splice_other (o : Bool) (cs : List T) : splice o (.conn (!o) cs) = [.conn (!o) cs] := by
  cases o <;> simp [splice]

theorem scL_cons_congr (o : Bool) (x : Except Err Val) (l1 l2 : List (Except Err Val))
    (h : ∀ n, scL o l1 n = scL o l2 n) (n : Bool) : scL o (x :: l1) n = scL o (x :: l2) n := by
  rcases x with e | (_ | b | i | s)
  · rfl
  · exact h true
  · simp only [scL]; split
    · rfl
    · exact h n
  · rfl
  · rfl

theorem splice_eval (o : Bool) (ts : List T) (n : Bool) :
    scL o ((ts.flatMap (splice o)).map evalT) n = scL o (ts.map evalT) n := by
  induction ts generalizing n with
  | nil => rfl
  | cons t ts ih =>
    rw [List.flatMap_cons, List.map_append, List.map_cons]
    cases t with
    | leaf v =>
      simp only [splice, List.map_cons, List.map_nil, List.cons_append, List.nil_append]
      exact scL_cons_congr o _ _ _ ih n
    | conn o' cs =>
      by_cases h : o' = o
      · subst h
        simp only [splice, ↓reduceIte, evalT, evalTL_map]
        rw [← scL_splice]
        exact scL_cons_congr o' _ _ _ ih n
      · simp only [splice, h, ↓reduceIte, List.map_cons, List.map_nil, List.cons_append,
          List.nil_append]
        exact scL_cons_congr o _ _ _ ih n

/-- **`flat` evaluates like the unflattened node** whenever more than one operand is
left after splicing (the only shapes `and_of` / `or_of` build from parser trees). -/
theorem flat_eval (o : Bool) (ts : List T) (h : (ts.flatMap (splice o)).length ≠ 1) :
    evalT (flatT o ts) = evalT (.conn o ts) := by
  rw [flatT_eq_body]
  have he := splice_eval o ts false
  simp only [evalT, evalTL_map]
  rw [← he]; unfold body
  generalize ts.flatMap (splice o) = l at h
  match l, h with
  | [], _ => cases o <;> rfl
  | _ :: _ :: _, _ => simp only [evalT, evalTL_map]

/-- `FilterOp`'s per-row verdict on a value (filter.rs:48-80). -/
def passesV : Except Err Val → Except Err Bool
  | .ok (.bool b) => .ok b
  | .ok .null => .ok false
  | .ok _ => .error .typeMismatch
  | .error e => .error e

/-- **As a Filter, `and_of xs` behaves exactly like `AND(xs)`** on every row,
for every operand list. -/
theorem flat_passes (ts : List T) : passesV (evalT (flatT false ts)) = passesV (evalT (.conn false ts)) := by
  rw [flatT_eq_body]
  have he := splice_eval false ts false
  simp only [evalT, evalTL_map]
  rw [← he]; unfold body
  generalize ts.flatMap (splice false) = l
  match l with
  | [] => rfl
  | [c] =>
    simp only [List.map_cons, List.map_nil]
    rcases evalT c with e | (_ | b | i | s)
    · rfl
    · rfl
    · cases b <;> rfl
    · rfl
    · rfl
  | _ :: _ :: _ => simp only [evalT, evalTL_map]

end Falkor.Opt.AndOf
