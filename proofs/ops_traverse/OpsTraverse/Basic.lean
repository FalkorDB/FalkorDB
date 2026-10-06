/-
# Graph model shared by every traverse proof

A relationship tensor stores every parallel edge; what the operators see of it
is (a) the *pair* level — the boolean pattern of the adjacency / relationship
matrix, one entry per `(src, dst)` — and (b) the *edge* level —
`Tensor::get(src, dst)`, the ids of every edge on that pair, in id order.

| here | there |
| --- | --- |
| `Edge`, `Graph`         | an entry of a relationship `Tensor` (`graph/src/graph/graphblas/tensor.rs`); relationship types are modelled as a pre-filtered edge list |
| `pairs`                 | the structural iterator of the (merged) relationship matrix: `build_unrestricted_iter` (`runtime/ops/cond_traverse.rs:195`) |
| `edgesBetween`          | `relationship_tensors()[t].get(src, dst)` (`cond_traverse.rs:1067,1086`, `expand_into.rs:207,224`) |
| `step`                  | `Graph::get_node_relationships_by_type` (`graph/src/graph/graph.rs:2079`) — a self-loop is reported once under `Both` (`graph.rs:2105-2110`) |
-/

namespace OpsTraverse

/-- Order-preserving de-duplication (keeps the last occurrence). -/
def dedup {α : Type} [DecidableEq α] : List α → List α
  | [] => []
  | a :: l => if a ∈ l then dedup l else a :: dedup l

theorem mem_dedup {α : Type} [DecidableEq α] {a : α} :
    ∀ {l : List α}, a ∈ dedup l ↔ a ∈ l
  | [] => by simp [dedup]
  | b :: l => by
    unfold dedup
    by_cases h : b ∈ l
    · rw [if_pos h, mem_dedup, List.mem_cons]
      constructor
      · intro h'; exact Or.inr h'
      · rintro (rfl | h'); exact h; exact h'
    · simp [h, mem_dedup]

theorem nodup_dedup {α : Type} [DecidableEq α] : ∀ (l : List α), (dedup l).Nodup
  | [] => by simp [dedup]
  | b :: l => by
    unfold dedup
    by_cases h : b ∈ l
    · rw [if_pos h]; exact nodup_dedup l
    · rw [if_neg h]
      exact List.nodup_cons.2 ⟨fun hm => h (mem_dedup.1 hm), nodup_dedup l⟩

structure Edge where
  id  : Nat
  src : Nat
  dst : Nat
deriving DecidableEq, Repr

abbrev Graph := List Edge

/-- Edge ids are unique (the tensor is keyed by edge id). -/
def Graph.WF (g : Graph) : Prop := (g.map Edge.id).Nodup

/-- The pair-level pattern: one entry per `(src, dst)` that carries an edge. -/
def pairs (g : Graph) : List (Nat × Nat) := dedup (g.map fun e => (e.src, e.dst))

theorem mem_pairs {g : Graph} {s d : Nat} :
    (s, d) ∈ pairs g ↔ ∃ e ∈ g, e.src = s ∧ e.dst = d := by
  simp [pairs, mem_dedup]

theorem nodup_pairs (g : Graph) : (pairs g).Nodup := nodup_dedup _

/-- `Tensor::get(s, d)`: every edge id on the pair. -/
def edgesBetween (g : Graph) (s d : Nat) : List Nat :=
  (g.filter fun e => e.src == s && e.dst == d).map Edge.id

theorem mem_edgesBetween {g : Graph} {s d e : Nat} :
    e ∈ edgesBetween g s d ↔ ∃ x ∈ g, x.id = e ∧ x.src = s ∧ x.dst = d := by
  simp only [edgesBetween, List.mem_map, List.mem_filter, Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨x, ⟨hx, h1, h2⟩, rfl⟩; exact ⟨x, hx, rfl, h1, h2⟩
  · rintro ⟨x, hx, rfl, h1, h2⟩; exact ⟨x, ⟨hx, h1, h2⟩, rfl⟩

/-- Adjacency of `cur` as `(edge id, neighbour)`: outgoing edges, plus
incoming ones when `bidir` (a self-loop only once: the `if` takes the
outgoing branch first, mirroring the `*src != id` filter at `graph.rs:2110`). -/
def step (g : Graph) (bidir : Bool) (cur : Nat) : List (Nat × Nat) :=
  g.filterMap fun e =>
    if e.src = cur then some (e.id, e.dst)
    else if bidir && e.dst = cur then some (e.id, e.src) else none

theorem mem_step {g : Graph} {b : Bool} {cur e n : Nat} :
    (e, n) ∈ step g b cur ↔
      ∃ x ∈ g, x.id = e ∧
        ((x.src = cur ∧ x.dst = n) ∨ (b = true ∧ x.src ≠ cur ∧ x.dst = cur ∧ x.src = n)) := by
  simp only [step, List.mem_filterMap]
  constructor
  · rintro ⟨x, hx, h⟩
    refine ⟨x, hx, ?_⟩
    by_cases h1 : x.src = cur
    · simp [h1] at h; exact ⟨h.1, Or.inl ⟨h1, h.2⟩⟩
    · by_cases h2 : (b && x.dst == cur) = true
      · simp only [Bool.and_eq_true, beq_iff_eq] at h2
        simp [h1, h2.1, h2.2] at h
        exact ⟨h.1, Or.inr ⟨h2.1, h1, h2.2, h.2⟩⟩
      · simp only [Bool.and_eq_true, beq_iff_eq, not_and] at h2
        by_cases hb : b = true
        · have := h2 hb; simp [h1, hb, this] at h
        · simp [h1, hb] at h
  · rintro ⟨x, hx, rfl, (⟨h1, rfl⟩ | ⟨hb, h1, h2, rfl⟩)⟩
    · exact ⟨x, hx, by simp [h1]⟩
    · exact ⟨x, hx, by simp [h1, hb, h2]⟩

end OpsTraverse
