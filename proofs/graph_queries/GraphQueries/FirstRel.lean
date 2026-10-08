import GraphQueries.Accessors
/-
# `Graph::first_node_with_relationships` (graph.rs:2245, #3022)

The `DELETE_NODE` arm of `apply_record` refuses a node that still has a
relationship. The Rust answers it in bulk: one `Aᵀ·x` over the adjacency matrix
(outgoing) and one per relationship type over the backward adjacency `mt`
(incoming), and only when one of them is non-zero re-seeks an iterator per
matrix to find the lowest such node.

The GraphBLAS side is a hypothesis (`BulkOps`): what `count_in_rows`,
`count_pairs_into` and a seek-then-`next` return, stated as the GraphBLAS `mxv`
/ iterator spec. `proofs/versioned_matrix` proves `VersionedMatrix::count_in_rows`
and `Tensor::count_pairs_into` reduce to the per-layer `Matrix::count_in_rows`,
and `proofs/graphblas_wrappers` reduces that to `GrB_mxv` + `GrB_Vector_reduce`.
The graph-level facts it relies on are `AdjExact`: the adjacency matrix holds
exactly the `(src, dst)` pairs some tensor holds (it loses a pair with its last
edge of any type), and every edge's endpoints are rows of it.

| here | there (graph.rs) |
| --- | --- |
| `firstNodeWithRels` | `first_node_with_relationships` :2245-2285 |
| `BulkOps.countInRows` | `adj.count_in_rows(&x)` (`VersionedMatrix::count_in_rows`) over `x = Vector::from_sorted_indices(n, &ids)` |
| `BulkOps.pairsInto` | `t.count_pairs_into(&x)` (`Tensor::count_pairs_into`) |
| `BulkOps.outNext` / `incNext` | `out.seek(id, id); out.next().is_some()` / `it.seek(id, id); it.next().is_some()` over `adj.iter(0, 0)` / `t.iter(0, 0, true)` |

Headline: `firstNodeWithRels_spec` — for ascending `nodes` (a `RoaringTreemap`),
the result is exactly the lowest node of `nodes` with a relationship in either
direction (`List.find? hasRel nodes`).
-/
namespace GQ
variable {V : Type}

/-- A node has a relationship: some tensor holds an edge out of it or into it. -/
def hasRel (g : G V) (n : Nat) : Bool := g.relMs.any (fun t => t.any (fun e => e.1 == n || e.2.1 == n))

/-- The adjacency matrix is exact and sized: what `first_node_with_relationships`'
doc relies on ("a pair leaves it when its last edge of any type goes"). -/
structure AdjExact (g : G V) : Prop where
  exact : ∀ s d, (s, d) ∈ g.adj.ents ↔ ∃ t ∈ g.relMs, ∃ e ∈ t, e.1 = s ∧ e.2.1 = d
  rows  : ∀ t ∈ g.relMs, ∀ e ∈ t, e.1 < g.adj.nr ∧ e.2.1 < g.adj.nr

/-- **Hypothesis** (GraphBLAS `GrB_mxv` over `PLUS_PAIR`, `GrB_Vector_reduce`, the
row iterator's seek): the four bulk queries over the indicator of `S`. -/
structure BulkOps (g : G V) where
  countInRows : List Nat → Nat
  countInRows_spec : ∀ S, countInRows S = (g.adj.ents.filter (fun p => decide (p.1 ∈ S))).length
  pairsInto : Ten → List Nat → Nat
  pairsInto_pos : ∀ t S, 0 < pairsInto t S ↔ ∃ e ∈ t, e.2.1 ∈ S
  outNext : Nat → Bool
  outNext_spec : ∀ id, outNext id = true ↔ g.adj.row id ≠ []
  incNext : Ten → Nat → Bool
  incNext_spec : ∀ t id, incNext t id = true ↔ Ten.inc t id ≠ []

/-- `first_node_with_relationships` (:2245). -/
def firstNodeWithRels (g : G V) (B : BulkOps g) (nodes : List Nat) : Option Nat :=
  if nodes.isEmpty || g.adj.nvals == 0 then none else
  let n := g.adj.nr
  let ids := nodes.takeWhile (· < n)
  if ids.isEmpty then none else
  let touched := decide (B.countInRows ids > 0) || g.relMs.any (fun t => decide (B.pairsInto t ids > 0))
  if !touched then none else
  ids.find? (fun id => B.outNext id || g.relMs.any (fun t => B.incNext t id))

theorem out_iff (g : G V) (hx : AdjExact g) (id : Nat) :
    g.adj.row id ≠ [] ↔ ∃ t ∈ g.relMs, ∃ e ∈ t, e.1 = id := by
  unfold Mat.row
  constructor
  · intro h
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ h
    rw [List.mem_filter] at hp
    obtain ⟨t, ht, e, he, h1, _⟩ := (hx.exact p.1 p.2).mp hp.1
    exact ⟨t, ht, e, he, by rw [h1]; simpa using hp.2⟩
  · rintro ⟨t, ht, e, he, rfl⟩ hnil
    have := (hx.exact e.1 e.2.1).mpr ⟨t, ht, e, he, rfl, rfl⟩
    have : (e.1, e.2.1) ∈ g.adj.ents.filter (·.1 == e.1) := List.mem_filter.mpr ⟨this, by simp⟩
    rw [hnil] at this; cases this

theorem inc_iff (t : Ten) (id : Nat) : Ten.inc t id ≠ [] ↔ ∃ e ∈ t, e.2.1 = id := by
  unfold Ten.inc
  constructor
  · intro h
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ h
    rw [List.mem_filter] at he; exact ⟨e, he.1, by simpa using he.2⟩
  · rintro ⟨e, he, rfl⟩ hnil
    have : e ∈ t.filter (·.2.1 == e.2.1) := List.mem_filter.mpr ⟨he, by simp⟩
    rw [hnil] at this; cases this

theorem hasRel_iff (g : G V) (n : Nat) :
    hasRel g n = true ↔ (∃ t ∈ g.relMs, ∃ e ∈ t, e.1 = n) ∨ ∃ t ∈ g.relMs, ∃ e ∈ t, e.2.1 = n := by
  unfold hasRel
  simp only [List.any_eq_true, Bool.or_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨t, ht, e, he, h | h⟩
    · exact .inl ⟨t, ht, e, he, h⟩
    · exact .inr ⟨t, ht, e, he, h⟩
  · rintro (⟨t, ht, e, he, h⟩ | ⟨t, ht, e, he, h⟩)
    · exact ⟨t, ht, e, he, .inl h⟩
    · exact ⟨t, ht, e, he, .inr h⟩

/-- The per-id predicate the `find` uses is `hasRel`. -/
theorem probe_iff (g : G V) (hx : AdjExact g) (B : BulkOps g) (id : Nat) :
    (B.outNext id || g.relMs.any (fun t => B.incNext t id)) = hasRel g id := by
  apply Bool.eq_iff_iff.mpr
  rw [hasRel_iff, Bool.or_eq_true, B.outNext_spec, out_iff g hx, List.any_eq_true]
  apply or_congr Iff.rfl
  constructor
  · rintro ⟨t, ht, h⟩; exact ⟨t, ht, (inc_iff t id).mp ((B.incNext_spec t id).mp h)⟩
  · rintro ⟨t, ht, h⟩; exact ⟨t, ht, (B.incNext_spec t id).mpr ((inc_iff t id).mpr h)⟩

/-- An id past the matrices has no relationship. -/
theorem hasRel_lt (g : G V) (hx : AdjExact g) (n : Nat) (h : hasRel g n = true) : n < g.adj.nr := by
  rcases (hasRel_iff g n).mp h with ⟨t, ht, e, he, rfl⟩ | ⟨t, ht, e, he, rfl⟩
  · exact (hx.rows t ht e he).1
  · exact (hx.rows t ht e he).2

/-- On an ascending list, the `take_while(id < n)` prefix loses no `hasRel` id. -/
theorem find_takeWhile (g : G V) (hx : AdjExact g) :
    ∀ (nodes : List Nat), nodes.Pairwise (· < ·) →
      (nodes.takeWhile (· < g.adj.nr)).find? (hasRel g) = nodes.find? (hasRel g)
  | [], _ => rfl
  | x :: xs, hs => by
    rw [List.pairwise_cons] at hs
    by_cases hx' : x < g.adj.nr
    · rw [List.takeWhile_cons_of_pos (by simpa using hx'), List.find?_cons, List.find?_cons,
        find_takeWhile g hx xs hs.2]
    · rw [List.takeWhile_cons_of_neg (by simpa using hx'), List.find?_nil]
      symm; apply List.find?_eq_none.mpr
      intro y hy
      have hy' : ¬ y < g.adj.nr := by
        rcases List.mem_cons.mp hy with rfl | hy
        · exact hx'
        · have := hs.1 y hy; omega
      intro hr; exact hy' (hasRel_lt g hx y (by simpa using hr))

/-- If nothing among `ids` has a relationship, `find?` says so. -/
theorem find_none_of (g : G V) (ids : List Nat) (h : ∀ i ∈ ids, hasRel g i = false) :
    ids.find? (hasRel g) = none :=
  List.find?_eq_none.mpr (fun i hi => by simp [h i hi])

/-- **`first_node_with_relationships` returns exactly the lowest node of `nodes`
with a relationship in either direction** (given the GraphBLAS bulk-query spec,
an exact adjacency matrix and ascending `nodes`). Each early `None` is sound: an
empty set, an empty adjacency matrix (hence no edge at all), ids all past the
matrices, or both bulk counts zero. -/
theorem firstNodeWithRels_spec (g : G V) (hx : AdjExact g) (B : BulkOps g) (nodes : List Nat)
    (hs : nodes.Pairwise (· < ·)) :
    firstNodeWithRels g B nodes = nodes.find? (hasRel g) := by
  rw [← find_takeWhile g hx nodes hs]
  generalize hid : nodes.takeWhile (· < g.adj.nr) = ids
  unfold firstNodeWithRels
  simp only [hid]
  by_cases h0 : (nodes.isEmpty || g.adj.nvals == 0) = true
  · simp only [h0, ↓reduceIte]; symm
    rcases Bool.or_eq_true _ _ |>.mp h0 with h | h
    · have : nodes = [] := List.isEmpty_iff.mp h
      subst this; simp at hid; subst hid; rfl
    · apply find_none_of; intro i _
      cases hr : hasRel g i
      · rfl
      · exfalso
        rcases (hasRel_iff g i).mp hr with ⟨t, ht, e, he, _⟩ | ⟨t, ht, e, he, _⟩ <;>
        · have := (hx.exact e.1 e.2.1).mpr ⟨t, ht, e, he, rfl, rfl⟩
          simp [Mat.nvals] at h; rw [h] at this; cases this
  simp only [h0, Bool.false_eq_true, ↓reduceIte]
  by_cases h1 : ids.isEmpty = true
  · simp only [h1, ↓reduceIte]; rw [List.isEmpty_iff.mp h1]; rfl
  simp only [h1, Bool.false_eq_true, ↓reduceIte]
  by_cases h2 : (decide (B.countInRows ids > 0) || g.relMs.any (fun t => decide (B.pairsInto t ids > 0))) = true
  · rw [h2]; simp only [Bool.not_true, Bool.false_eq_true, if_false]
    congr 1; funext id; exact probe_iff g hx B id
  · simp only [Bool.not_eq_true] at h2
    rw [h2]; simp only [Bool.not_false, if_true]
    symm; apply find_none_of
    intro i hi
    rw [Bool.or_eq_false_iff] at h2
    obtain ⟨hc, hp⟩ := h2
    cases hr : hasRel g i
    · rfl
    · exfalso
      rcases (hasRel_iff g i).mp hr with ⟨t, ht, e, he, rfl⟩ | ⟨t, ht, e, he, rfl⟩
      · have hm := (hx.exact e.1 e.2.1).mpr ⟨t, ht, e, he, rfl, rfl⟩
        have : 0 < B.countInRows ids := by
          rw [B.countInRows_spec]
          exact List.length_pos_of_mem (List.mem_filter.mpr ⟨hm, by simpa using hi⟩)
        simp at hc; omega
      · have := (B.pairsInto_pos t ids).mpr ⟨e, he, hi⟩
        have := List.any_eq_false.mp hp t ht
        simp at this; omega

end GQ
