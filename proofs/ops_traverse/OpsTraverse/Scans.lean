import OpsTraverse.Basic
/-
# Node scans

| here | there |
| --- | --- |
| `labelScan`      | `NodeByLabelScanOp::next` (`runtime/ops/node_by_label_scan.rs:77-97`) / `AllNodeScan` (same op with no labels, `runtime/runtime.rs:759-764`): every input row × every node of `get_nodes(labels, 0)` |
| `idSeek`         | `NodeByIdSeekOp::next` (`runtime/ops/node_by_id_seek.rs:55-84`): `range -= deleted_nodes()` |
| `labelIdScan`    | `NodeByLabelAndIdScanOp::next` (`runtime/ops/node_by_label_and_id_scan.rs:59-94`): `get_nodes(labels, min).take_while(<= max).filter(range.contains)` |
| `getNodes`       | `Graph::get_nodes(labels, min)` — **boundary**: the label-matrix intersection iterator (GraphBLAS); assumed to enumerate the labelled live nodes `≥ min` in strictly increasing id order (`Sorted` hypothesis) |

`evaluate_id_filter` (the id range computed from the WHERE predicate) is
treated as an input; its correctness belongs to the optimizer.
-/

namespace OpsTraverse.Scans

def labelScan {α : Type} (rows : List α) (nodes : List Nat) : List (α × Nat) :=
  rows.flatMap fun r => nodes.map fun n => (r, n)

theorem mem_labelScan {α : Type} {rows : List α} {nodes : List Nat} {r : α} {n : Nat} :
    (r, n) ∈ labelScan rows nodes ↔ r ∈ rows ∧ n ∈ nodes := by
  simp [labelScan]

theorem length_labelScan {α : Type} (rows : List α) (nodes : List Nat) :
    (labelScan rows nodes).length = rows.length * nodes.length := by
  induction rows with
  | nil => simp [labelScan]
  | cons r rs ih =>
    simp only [labelScan, List.flatMap_cons, List.length_append, List.length_map] at ih ⊢
    rw [ih, List.length_cons, Nat.succ_mul, Nat.add_comm]

def idSeek (range deleted : List Nat) : List Nat := range.filter fun i => !(deleted.contains i)

theorem mem_idSeek {range deleted : List Nat} {i : Nat} :
    i ∈ idSeek range deleted ↔ i ∈ range ∧ i ∉ deleted := by
  simp [idSeek]

def getNodes (labelled : List Nat) (min : Nat) : List Nat := labelled.filter (min ≤ ·)

def labelIdScan (labelled range : List Nat) (min max : Nat) : List Nat :=
  ((getNodes labelled min).takeWhile (· ≤ max)).filter (range.contains ·)

theorem filter_takeWhile_sorted (max : Nat) (P : Nat → Bool) (hP : ∀ x, P x → x ≤ max) :
    ∀ l : List Nat, l.Pairwise (· < ·) →
      (l.takeWhile (· ≤ max)).filter P = l.filter P
  | [], _ => rfl
  | a :: l, h => by
    have ih := filter_takeWhile_sorted max P hP l (List.pairwise_cons.1 h).2
    by_cases ha : a ≤ max
    · simp [List.takeWhile_cons, ha, List.filter_cons, ih]
    · have hnone : ∀ y ∈ a :: l, P y = false := by
        intro y hy
        rcases List.mem_cons.1 hy with rfl | hy
        · cases h' : P y
          · rfl
          · exact absurd (hP y h') ha
        · have hay := (List.pairwise_cons.1 h).1 y hy
          cases h' : P y
          · rfl
          · have := hP y h'; omega
      have hnil : (a :: l).filter P = [] := List.filter_eq_nil_iff.2 (by simpa using hnone)
      simp [List.takeWhile_cons, ha, hnil]

/-- **The id-range label scan returns exactly the labelled nodes in the range**,
given a sorted `get_nodes` and a range inside `[min, max]`. -/
theorem labelIdScan_correct (labelled range : List Nat) (min max : Nat)
    (hs : labelled.Pairwise (· < ·)) (hr : ∀ i ∈ range, min ≤ i ∧ i ≤ max) :
    labelIdScan labelled range min max = labelled.filter (range.contains ·) := by
  unfold labelIdScan getNodes
  rw [filter_takeWhile_sorted max _ (fun x hx => (hr x (by simpa using hx)).2) _
    (hs.sublist List.filter_sublist)]
  rw [List.filter_filter]
  congr 1
  funext x
  by_cases hx : x ∈ range
  · simp [hx, (hr x hx).1]
  · simp [hx]

end OpsTraverse.Scans
