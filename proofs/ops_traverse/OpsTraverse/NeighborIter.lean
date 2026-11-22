/-
# `NeighborIter` (graph/src/runtime/eval.rs): the `shortestPath()` neighbour source

| here | there |
| --- | --- |
| `Row` | what one `versioned_matrix::Iter` yields after `seek(node, node)`: the columns of row `node` |
| `NI`, `neighbors` | `NeighborIter`, `NeighborIter::neighbors` (:232) |
| `dedupAdj` | `Vec::sort_unstable` + `Vec::dedup` |
| `newNI` | `NeighborIter::new` (:164) |
| `newRev` | `NeighborIter::new_reversed` (:204) |

PROVEN relative to the matrices it reads: each iterator is a function `Row` from a node to the
columns of that row of the matrix it is bound to (the `seek(node,node)`-then-drain contract of
`proofs/versioned_matrix` `VMIterSeek.seek_abs` + `VMIter.drain_eq`, applied to the effective matrix).
That each such matrix is the adjacency / relationship matrix the graph means is a separate,
graph-wide fact (`AdjExact`), not assumed or proved here.
-/
namespace OpsTraverse.NeighborIter

abbrev Row := Nat → List Nat

structure NI where
  fwd : List Row
  bwd : List Row
  dedup : Bool

/-- `sort_unstable` then `dedup` of consecutive equals. -/
def dedupAdj : List Nat → List Nat
  | a :: b :: t => if a = b then dedupAdj (b :: t) else a :: dedupAdj (b :: t)
  | l => l

theorem mem_dedupAdj : ∀ (l : List Nat) (x : Nat), x ∈ dedupAdj l ↔ x ∈ l
  | [], _ => by simp [dedupAdj]
  | [a], _ => by simp [dedupAdj]
  | a :: b :: t, x => by
    simp only [dedupAdj]
    by_cases h : a = b
    · subst h; simp [mem_dedupAdj (a :: t) x]
    · simp [h, mem_dedupAdj (b :: t) x]

theorem dedupAdj_sorted : ∀ (l : List Nat), l.Pairwise (· ≤ ·) → (dedupAdj l).Pairwise (· < ·)
  | [], _ => by simp [dedupAdj]
  | [a], _ => by simp [dedupAdj]
  | a :: b :: t, h => by
    have h1 := List.pairwise_cons.mp h
    have h2 := dedupAdj_sorted (b :: t) h1.2
    simp only [dedupAdj]
    by_cases hab : a = b
    · subst hab; simp [h2]
    · simp only [hab, ↓reduceIte, List.pairwise_cons]
      refine ⟨fun y hy => ?_, h2⟩
      have hy' := (mem_dedupAdj (b :: t) y).mp hy
      have := h1.1 y hy'
      have hne : a ≠ y := by
        intro e
        subst e
        -- y = a ∈ b :: t with a ≤ b ≤ ... : only possible when equal to b
        rcases List.mem_cons.mp hy' with rfl | hm
        · exact hab rfl
        · have hb := (List.pairwise_cons.mp h1.2).1 a hm
          have := h1.1 b (by simp)
          omega
      omega

/-- `NeighborIter::neighbors` (:232). -/
def neighbors (ni : NI) (node : Nat) : List Nat :=
  let buf := ni.fwd.flatMap (fun r => r node) ++ ni.bwd.flatMap (fun r => r node)
  if ni.dedup then dedupAdj (buf.mergeSort (fun a b => decide (a ≤ b))) else buf

theorem le_trans' : ∀ a b c : Nat, decide (a ≤ b) = true → decide (b ≤ c) = true → decide (a ≤ c) = true := by
  intro a b c h1 h2; simp at *; omega

/-- **PROVEN** (`NeighborIter::neighbors`): the neighbours of `node` are exactly the columns of row
`node` in any of its iterators (forward ones, then backward ones); with `dedup` they come back sorted
ascending without repeats, without it as the plain concatenation. -/
theorem neighbors_spec (ni : NI) (node : Nat) :
    (∀ x, x ∈ neighbors ni node ↔ ∃ r ∈ ni.fwd ++ ni.bwd, x ∈ r node) ∧
    (ni.dedup = true → (neighbors ni node).Pairwise (· < ·)) ∧
    (ni.dedup = false → neighbors ni node =
        ni.fwd.flatMap (fun r => r node) ++ ni.bwd.flatMap (fun r => r node)) := by
  have hmem : ∀ x, x ∈ ni.fwd.flatMap (fun r => r node) ++ ni.bwd.flatMap (fun r => r node) ↔
      ∃ r ∈ ni.fwd ++ ni.bwd, x ∈ r node := by
    intro x
    simp only [List.mem_append, List.mem_flatMap]
    constructor
    · rintro (⟨r, hr, hx⟩ | ⟨r, hr, hx⟩)
      · exact ⟨r, Or.inl hr, hx⟩
      · exact ⟨r, Or.inr hr, hx⟩
    · rintro ⟨r, hr | hr, hx⟩
      · exact Or.inl ⟨r, hr, hx⟩
      · exact Or.inr ⟨r, hr, hx⟩
  refine ⟨fun x => ?_, fun hd => ?_, fun hd => ?_⟩
  · unfold neighbors
    cases hd : ni.dedup
    · simp only [Bool.false_eq_true, ↓reduceIte]; exact hmem x
    · simp only [↓reduceIte, mem_dedupAdj, List.mem_mergeSort]; exact hmem x
  · unfold neighbors
    simp only [hd, ↓reduceIte]
    apply dedupAdj_sorted
    have := List.pairwise_mergeSort (le := fun a b : Nat => decide (a ≤ b)) le_trans'
      (fun a b => by simp; omega) (ni.fwd.flatMap (fun r => r node) ++ ni.bwd.flatMap (fun r => r node))
    exact this.imp (fun h => by simpa using h)
  · unfold neighbors; simp [hd]

/-- `NeighborIter::new` (:164): `relTypes` empty reads the adjacency matrix (one forward iterator,
plus every relationship matrix transposed when undirected); otherwise one structural iterator per
existing type (plus its transpose when undirected). `typed t` is `none` for a type with no matrix. -/
def newNI (relTypes : List String) (directed : Bool) (adj : Row) (allT : List Row)
    (typed : String → Option (Row × Row)) : NI :=
  let fb : List Row × List Row :=
    if relTypes.isEmpty then ([adj], if directed then [] else allT)
    else
      let ms := relTypes.filterMap typed
      (ms.map (·.1), if directed then [] else ms.map (·.2))
  ⟨fb.1, fb.2, decide (fb.1.length + fb.2.length > 1)⟩

/-- `NeighborIter::new_reversed` (:204): the transposed matrices only, as the forward side. -/
def newRev (relTypes : List String) (allT : List Row) (typed : String → Option (Row × Row)) : NI :=
  let f : List Row := if relTypes.isEmpty then allT else (relTypes.filterMap typed).map (·.2)
  ⟨f, [], decide (f.length > 1)⟩

/-- **PROVEN** (`NeighborIter::new`): the iterator set per case, and `dedup` exactly when there is
more than one iterator. -/
theorem newNI_spec (rt : List String) (dir : Bool) (adj : Row) (allT : List Row)
    (typed : String → Option (Row × Row)) :
    (rt = [] → dir = true → (newNI rt dir adj allT typed).fwd = [adj] ∧
        (newNI rt dir adj allT typed).bwd = [] ∧ (newNI rt dir adj allT typed).dedup = false) ∧
    (rt = [] → dir = false → (newNI rt dir adj allT typed).fwd = [adj] ∧
        (newNI rt dir adj allT typed).bwd = allT) ∧
    (rt ≠ [] → (newNI rt dir adj allT typed).fwd = (rt.filterMap typed).map (·.1) ∧
        (newNI rt dir adj allT typed).bwd =
          (if dir then [] else (rt.filterMap typed).map (·.2))) ∧
    ((newNI rt dir adj allT typed).dedup =
      decide ((newNI rt dir adj allT typed).fwd.length + (newNI rt dir adj allT typed).bwd.length > 1)) := by
  refine ⟨fun h1 h2 => ?_, fun h1 h2 => ?_, fun h => ?_, rfl⟩
  · subst h1; subst h2; simp [newNI]
  · subst h1; subst h2; simp [newNI]
  · have : rt.isEmpty = false := by cases rt <;> simp_all
    simp [newNI, this]

/-- **PROVEN** (`NeighborIter::new_reversed`): predecessors, from the transposed matrices; never a
backward side. -/
theorem newRev_spec (rt : List String) (allT : List Row) (typed : String → Option (Row × Row)) :
    (newRev rt allT typed).bwd = [] ∧
    (rt = [] → (newRev rt allT typed).fwd = allT) ∧
    (rt ≠ [] → (newRev rt allT typed).fwd = (rt.filterMap typed).map (·.2)) ∧
    (newRev rt allT typed).dedup = decide ((newRev rt allT typed).fwd.length > 1) := by
  refine ⟨rfl, fun h => ?_, fun h => ?_, rfl⟩
  · subst h; simp [newRev]
  · have : rt.isEmpty = false := by cases rt <;> simp_all
    simp [newRev, this]

end OpsTraverse.NeighborIter
