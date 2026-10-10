/-! # algo.SPpaths / algo.SSpaths host-side logic

| here | there (`graph/src/runtime/functions/algo_procedures.rs`) |
| --- | --- |
| `Dir`, `farEndpoint`  | `EdgeDirection`, `far_endpoint` :2121 |
| `Found`, `cmpFound`   | `FoundPath`, `cmp_found_path` :2092 (weights modelled as `Int`: finite weights only) |
| `record`              | `record_found_path` :2405 |
| `pathCostRust`        | cost accumulation `edge_numeric_attr(.., cost_prop, 0.0)` :2308, :2397, :2518 |
| `pathCostC`           | C `_get_value_or_default(.., SI_LongVal(1))` (proc_sp_paths.c:587) |
| `nextNode`            | `build_path_batch` :2572 node reconstruction |
| `useDijkstra`         | `run_path_algo` fast-path guard :2616-2626 |
-/
namespace AlgoUdf.Paths

inductive Dir | outgoing | incoming | both deriving DecidableEq

def farEndpoint (frm src dst : Nat) : Dir → Option Nat
  | .outgoing => if src = frm then some dst else none
  | .incoming => if dst = frm then some src else none
  | .both => if src = frm then some dst else if dst = frm then some src else none

theorem far_outgoing (f s d x : Nat) : farEndpoint f s d .outgoing = some x ↔ s = f ∧ x = d := by
  unfold farEndpoint; split <;> simp_all [eq_comm]

theorem far_incoming (f s d x : Nat) : farEndpoint f s d .incoming = some x ↔ d = f ∧ x = s := by
  unfold farEndpoint; split <;> simp_all [eq_comm]

/-- `both` is the union of the two directions (outgoing preferred on a self-loop). -/
theorem far_both (f s d : Nat) :
    farEndpoint f s d .both =
      (farEndpoint f s d .outgoing).or (farEndpoint f s d .incoming) := by
  unfold farEndpoint; by_cases h : s = f <;> by_cases h' : d = f <;> simp [h, h']

/-- A found path: weight, cost, hop count. -/
structure Found where
  w : Int
  c : Int
  len : Nat
  deriving DecidableEq, Repr, Inhabited

/-- `cmp_found_path`: weight, then cost, then hops. `lt a b` = `Ordering::Less`. -/
def lt (a b : Found) : Bool :=
  a.w < b.w || (a.w == b.w && (a.c < b.c || (a.c == b.c && a.len < b.len)))

/-- `record_found_path`, with the k-case's sorted insert at `partition_point`. -/
def insertSorted (f : Found) : List Found → List Found
  | [] => [f]
  | x :: xs => if lt x f then x :: insertSorted f xs else f :: x :: xs

def record1 (res : List Found) (f : Found) : List Found :=
  match res with
  | [] => [f]
  | best :: _ => if lt f best then [f] else res

def record0 (res : List Found) (f : Found) : List Found :=
  match res with
  | [] => [f]
  | best :: _ => if f.w > best.w then res else if f.w < best.w then [f] else res ++ [f]

def recordK (res : List Found) (f : Found) (k : Nat) : List Found :=
  if res.length = k then
    (if lt f (res.getLast!) then insertSorted f res.dropLast else res)
  else insertSorted f res

def record (res : List Found) (f : Found) (k : Nat) : List Found :=
  if k = 1 then record1 res f else if k = 0 then record0 res f else recordK res f k

theorem insertSorted_length (f : Found) (l : List Found) :
    (insertSorted f l).length = l.length + 1 := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [insertSorted]; split <;> simp [ih]

/-- The kept set never exceeds `pathCount` (k ≥ 2 branch; k = 1 keeps ≤ 1). -/
theorem recordK_length (res : List Found) (f : Found) (k : Nat) (hk : 1 ≤ k)
    (h : res.length ≤ k) : (recordK res f k).length ≤ k := by
  unfold recordK
  by_cases he : res.length = k
  · simp only [he, if_true]
    split
    · rw [insertSorted_length, List.length_dropLast]; omega
    · omega
  · simp only [he, ite_false]; rw [insertSorted_length]; omega

theorem record1_length (res : List Found) (f : Found) (h : res.length ≤ 1) :
    (record1 res f).length ≤ 1 := by
  unfold record1; split
  · simp
  · split <;> simp_all

/-- pathCount = 1 only ever keeps the new path or the previous one. -/
theorem record1_min (res : List Found) (f : Found) :
    ∀ g ∈ record1 res f, g = f ∨ g ∈ res := by
  intro g hg; unfold record1 at hg
  split at hg
  · simp at hg; exact Or.inl hg
  · split at hg
    · simp at hg; exact Or.inl hg
    · exact Or.inr hg

/-- pathCount = 0 keeps only paths tied at the minimum weight. -/
theorem record0_ties (res : List Found) (f : Found)
    (h : ∀ a ∈ res, ∀ b ∈ res, a.w = b.w) :
    ∀ a ∈ record0 res f, ∀ b ∈ record0 res f, a.w = b.w := by
  unfold record0
  split
  · simp
  · rename_i best rest
    split
    · exact h
    · split
      · simp
      · intro a ha b hb
        have hfb : f.w = best.w := by omega
        simp only [List.mem_append, List.mem_singleton] at ha hb
        rcases ha with ha | rfl <;> rcases hb with hb | rfl
        · exact h a ha b hb
        · rw [hfb]; exact h a ha best (by simp)
        · rw [hfb]; exact h best (by simp) b hb
        · rfl

/-! ## Missing cost: 0 in Rust, 1 in C -/

/-- Per-edge cost as Rust reads it: the attribute if numeric, else 0. -/
def pathCostRust (attrs : List (Option Int)) : Int := (attrs.map (·.getD 0)).foldl (· + ·) 0
/-- As C's enumeration reads it: the attribute if numeric, else 1. -/
def pathCostC (attrs : List (Option Int)) : Int := (attrs.map (·.getD 1)).foldl (· + ·) 0

theorem cost_agree_when_present (xs : List Int) :
    pathCostRust (xs.map some) = pathCostC (xs.map some) := by
  simp [pathCostRust, pathCostC, List.map_map, Function.comp_def]

/-- BUG (confirmed live and by `bug_sppaths_missing_cost_defaults_to_zero`): a
2-hop path whose edges carry no cost attribute (or no costProp at all) costs 2 in
C and 0 in Rust; with `maxCost: 1` C prunes it and Rust keeps it
(`bug_sppaths_max_cost_ignores_hops`). -/
theorem cost_default_diverges :
    pathCostRust [none, none] = 0 ∧ pathCostC [none, none] = 2 := by decide

/-! ## Path reconstruction -/

/-- `build_path_batch`: the node after an edge is the endpoint that is not `prev`. -/
def nextNode (prev s d : Nat) : Nat := if prev = s then d else s

theorem nextNode_far (prev s d : Nat) (dir : Dir) (x : Nat)
    (h : farEndpoint prev s d dir = some x) : nextNode prev s d = x := by
  cases dir <;> simp only [farEndpoint] at h <;>
    (unfold nextNode; split at h <;> (try split at h) <;> simp_all)

/-- Fast-path guard: Dijkstra only for one path, no maxCost, unbounded maxLen,
`source ≠ target`. Dijkstra ignores the cost/hops tie-break of `cmp_found_path`
(first relaxation wins on equal weight), so a weight tie can report the costlier
path; C's `Dijkstra_ShortestPath` does the same (checked live: both return cost
10 where `maxLen:10` returns cost 1) — shared with C, not a divergence. -/
def useDijkstra (target : Option Nat) (src : Nat) (k : Int) (maxCost : Option Int)
    (maxLen : Int) : Bool :=
  target.isSome && k == 1 && maxCost.isNone && maxLen == 4294967295 && target != some src

theorem dijkstra_never_src_eq_dst (src : Nat) (k : Int) (mc : Option Int) (ml : Int) :
    useDijkstra (some src) src k mc ml = false := by simp [useDijkstra]

end AlgoUdf.Paths
