/-
The emitter-driven `next` loop shared by the expanding operators, with and without a
record cap.

| here | there |
| --- | --- |
| `drive`    | `next` of `NodeByLabelScanOp` (node_by_label_scan.rs:77), `AllShortestPathsOp` (all_shortest_paths.rs:311), `CondVarLenTraverseOp` (cond_var_len_traverse.rs:621, error-free case): emit the seeded batch's packed outputs, then pull and seed the next child batch |
| `capDrive` | `CondTraverseOp::next` (cond_traverse.rs:1086) with `trim_to_cap` (:1121); `ExpandIntoOp::next` (expand_into.rs:244) with its `set_selection(0..remaining)` trim |
| `driveE`   | the same loop with errors: a failed child pull or expansion ends the stream with that error |

`pack c` = the list of output batches the emitter produces for child batch `c`
(`BatchedResultEmitter::emit_lazy` repeatedly until `Ok(None)`); its row-level contents
(`flatten`) are the per-row expansions in order — the emitter conservation theorem of
proofs/columnar (`emit_lazy` conserves the `(row, item)` stream).
-/
namespace OpsTraverse.Drive

variable {C R : Type}

def drive (pack : C → List (List R)) : List (List R) → List C → List (List R)
  | b :: bs, cs => b :: drive pack bs cs
  | [], c :: cs => drive pack (pack c) cs
  | [], [] => []
termination_by q cs => (cs.length, q.length)

/-- **No loss, no duplication, no reordering**: the rows emitted are the queued rows followed by
every child batch's expansion, in order. -/
theorem drive_flatten (pack : C → List (List R)) :
    ∀ (q : List (List R)) (cs : List C),
      (drive pack q cs).flatten = q.flatten ++ (cs.flatMap pack).flatten
  | b :: bs, cs => by
      rw [drive]; simp only [List.flatten_cons, drive_flatten pack bs cs, List.append_assoc]
  | [], c :: cs => by
      rw [drive]; rw [drive_flatten pack (pack c) cs]; simp
  | [], [] => by simp [drive]
termination_by q cs => (cs.length, q.length)

/-- With a cap: each emitted batch is trimmed to the remaining budget, nothing after the cap. -/
def capDrive (pack : C → List (List R)) (cap : Nat) : Nat → List (List R) → List C → List (List R)
  | produced, b :: bs, cs =>
    if cap ≤ produced then []
    else
      let t := b.take (cap - produced)
      t :: capDrive pack cap (produced + t.length) bs cs
  | produced, [], c :: cs => if cap ≤ produced then [] else capDrive pack cap produced (pack c) cs
  | _, [], [] => []
termination_by _ q cs => (cs.length, q.length)

theorem take_append_take {α : Type} (n : Nat) (a b : List α) :
    (a ++ b).take n = a.take n ++ b.take (n - a.length) := by
  exact List.take_append

/-- **LIMIT propagation is exact**: the capped stream's rows are the first `cap - produced`
rows of the uncapped stream. -/
theorem capDrive_flatten (pack : C → List (List R)) (cap : Nat) :
    ∀ (produced : Nat) (q : List (List R)) (cs : List C),
      (capDrive pack cap produced q cs).flatten =
        (q.flatten ++ (cs.flatMap pack).flatten).take (cap - produced)
  | produced, b :: bs, cs => by
      rw [capDrive]
      split
      · rename_i h; simp [Nat.sub_eq_zero_of_le h]
      · rename_i h
        simp only [List.flatten_cons, List.append_assoc]
        rw [capDrive_flatten pack cap _ bs cs]
        conv => rhs; rw [take_append_take]
        rw [List.length_take]
        have e : cap - (produced + min (cap - produced) b.length) = cap - produced - b.length := by
          simp only [Nat.min_def]; split <;> omega
        rw [e]
  | produced, [], c :: cs => by
      rw [capDrive]
      split
      · rename_i h; simp [Nat.sub_eq_zero_of_le h]
      · rw [capDrive_flatten pack cap produced (pack c) cs]; simp
  | produced, [], [] => by simp [capDrive]
termination_by _ q cs => (cs.length, q.length)

/-- With no cap reached the capped loop equals the plain one. -/
theorem capDrive_big (pack : C → List (List R)) (cap : Nat) (q : List (List R)) (cs : List C)
    (h : (q.flatten ++ (cs.flatMap pack).flatten).length ≤ cap) :
    (capDrive pack cap 0 q cs).flatten = (drive pack q cs).flatten := by
  rw [capDrive_flatten, drive_flatten, Nat.sub_zero, List.take_of_length_le h]

/-! ## Errors -/

/-- Child pulls and expansions may fail; the first failure is the last item. -/
def driveE (pack : C → Except String (List (List R))) : List (List R) → List (Except String C) →
    List (Except String (List R))
  | b :: bs, cs => .ok b :: driveE pack bs cs
  | [], [] => []
  | [], .error e :: _ => [.error e]
  | [], .ok c :: cs => match pack c with
    | .error e => [.error e]
    | .ok q => driveE pack q cs
termination_by q cs => (cs.length, q.length)

theorem driveE_ok (pack : C → Except String (List (List R))) (f : C → List (List R))
    (hf : ∀ c, pack c = .ok (f c)) :
    ∀ (q : List (List R)) (cs : List C),
      driveE pack q (cs.map .ok) = (drive f q cs).map .ok
  | b :: bs, cs => by rw [driveE, drive, driveE_ok pack f hf bs cs]; rfl
  | [], c :: cs => by
      simp only [List.map_cons]
      rw [driveE, hf c, drive]
      exact driveE_ok pack f hf (f c) cs
  | [], [] => by simp [driveE, drive]
termination_by q cs => (cs.length, q.length)

theorem driveE_child_err (pack : C → Except String (List (List R))) (e : String) (cs : List (Except String C)) :
    driveE pack [] (.error e :: cs) = [.error e] := by rw [driveE]

end OpsTraverse.Drive
