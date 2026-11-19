import GraphQueries.Ids
/-
# `Graph::restore`'s refusals (#3643, graph.rs:903-1053)

`restore` now returns `Result`: before it builds anything it holds every id the
payload carries to the counts it declares.

| here | there (graph.rs) |
| --- | --- |
| `idBound` | `Graph::id_bound` (:1037): `live.checked_add(deleted.len())`, then `deleted.max() < bound` |
| `restoreCheck` | `restore` (:922-972): both id bounds, the first edge scan's `live_edges`, `max_edge_id`, `max_endpoint`, the two refusals, then `len` |

The deleted ids are a `RoaringTreemap`, i.e. a duplicate-free set: a `List Nat`
with `Nodup` where it matters. An edge is `(src, dst, edge_id)` as
`Tensor::iter_edges` yields it. `none` is the `Err`.

* `idBound_some_iff` — the bound is `live + |deleted|`, it fits a `u64`, and
  every deleted id is below it.
* `restoreCheck_some_iff` — `restore` passes its checks iff both bounds exist,
  the tensors carry exactly `relationship_count` edges, and every edge's id is
  below the edge bound and both endpoints below the node bound; then `len` is
  `max_edge_id + 1` (0 with no edge).
* `restoreCheck_len_covers` — on success every edge id is below `len`, and `len`
  is at most the edge bound: the endpoint index is sized by the payload's
  counts, never by an id it merely names.
* `restoreCheck_accepts_dense` — a graph whose ids are dense (every id below
  `live + deleted` is live or deleted, the `IdSpace` invariant) passes.
* `restore_refuses_*` — the PR's unit test cases, refused.
-/
namespace GQ

/-- `Graph::id_bound` (:1037). -/
def idBound (live : Nat) (deleted : List Nat) : Option Nat :=
  let bound := live + deleted.length
  if 2 ^ 64 ≤ bound then none
  else match deleted.max? with
    | some m => if bound ≤ m then none else some bound
    | none => some bound

theorem max?_lt_iff (l : List Nat) (b : Nat) :
    (∀ m, l.max? = some m → m < b) ↔ ∀ x ∈ l, x < b := by
  constructor
  · intro h x hx
    cases hm : l.max? with
    | none => simp [List.max?_eq_none_iff] at hm; subst hm; simp at hx
    | some m =>
      have := h m hm
      have hle := (List.max?_eq_some_iff.1 hm).2 x hx
      omega
  · intro h m hm
    exact h m (List.max?_mem hm)

/-- **`idBound_some_iff`**. -/
theorem idBound_some_iff (live : Nat) (deleted : List Nat) (b : Nat) :
    idBound live deleted = some b ↔
      b = live + deleted.length ∧ b < 2 ^ 64 ∧ ∀ x ∈ deleted, x < b := by
  unfold idBound
  simp only
  by_cases hov : 2 ^ 64 ≤ live + deleted.length
  · simp only [hov, ite_true]
    constructor
    · intro h; cases h
    · rintro ⟨rfl, h, -⟩; omega
  · simp only [hov, ite_false]
    rw [← max?_lt_iff]
    cases hm : deleted.max? with
    | none => simp; omega
    | some m =>
      by_cases hb : live + deleted.length ≤ m
      · simp only [hb, ite_true]
        constructor
        · intro h; cases h
        · rintro ⟨rfl, -, h⟩; have := h m rfl; omega
      · simp only [hb, ite_false, Option.some.injEq]
        constructor
        · rintro rfl; refine ⟨rfl, by omega, ?_⟩
          intro m' hm'; cases hm'; omega
        · rintro ⟨rfl, -, -⟩; rfl

/-- The first scan's three maxima and count (:941-956). -/
def maxEdgeId (edges : List (Nat × Nat × Nat)) : Nat := edges.foldl (fun a e => max a e.2.2) 0
def maxEndpoint (edges : List (Nat × Nat × Nat)) : Nat := edges.foldl (fun a e => max (max a e.1) e.2.1) 0

theorem foldl_max_lt (f : Nat × Nat × Nat → Nat) : ∀ (l : List (Nat × Nat × Nat)) (a b : Nat),
    l.foldl (fun a e => max a (f e)) a < b ↔ a < b ∧ ∀ e ∈ l, f e < b
  | [], a, b => by simp
  | e :: l, a, b => by
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp]
    rw [foldl_max_lt f l, Nat.max_lt, and_assoc]

theorem maxEdgeId_lt (edges : List (Nat × Nat × Nat)) (b : Nat) :
    maxEdgeId edges < b ↔ 0 < b ∧ ∀ e ∈ edges, e.2.2 < b := by
  unfold maxEdgeId; rw [foldl_max_lt (fun e => e.2.2)]

theorem maxEndpoint_lt (edges : List (Nat × Nat × Nat)) (b : Nat) :
    maxEndpoint edges < b ↔ 0 < b ∧ ∀ e ∈ edges, e.1 < b ∧ e.2.1 < b := by
  unfold maxEndpoint
  have := foldl_max_lt (fun e => max e.1 e.2.1) edges 0 b
  have heq : edges.foldl (fun a e => max (max a e.1) e.2.1) 0 =
      edges.foldl (fun a e => max a (max e.1 e.2.1)) 0 := by
    congr 1; funext a e; omega
  rw [heq, this]
  constructor
  · rintro ⟨h0, h⟩; exact ⟨h0, fun e he => by have := h e he; omega⟩
  · rintro ⟨h0, h⟩; exact ⟨h0, fun e he => by have := h e he; omega⟩

/-- `restore`'s checks (:922-972): `some len` (the endpoint index's length) or `none` (`Err`). -/
def restoreCheck (nodeCount relCount : Nat) (delN delE : List Nat) (edges : List (Nat × Nat × Nat)) :
    Option Nat := do
  let nb ← idBound nodeCount delN
  let eb ← idBound relCount delE
  let live := edges.length
  if live ≠ relCount then none
  else if 0 < live ∧ (eb ≤ maxEdgeId edges ∨ nb ≤ maxEndpoint edges) then none
  else some (if 0 < live then maxEdgeId edges + 1 else 0)

/-- **`restoreCheck_some_iff`**. -/
theorem restoreCheck_some_iff (nodeCount relCount : Nat) (delN delE : List Nat)
    (edges : List (Nat × Nat × Nat)) :
    (restoreCheck nodeCount relCount delN delE edges).isSome ↔
      (∃ nb eb, idBound nodeCount delN = some nb ∧ idBound relCount delE = some eb ∧
        edges.length = relCount ∧ ∀ e ∈ edges, e.2.2 < eb ∧ e.1 < nb ∧ e.2.1 < nb) := by
  unfold restoreCheck
  cases hn : idBound nodeCount delN with
  | none => simp
  | some nb =>
    cases he : idBound relCount delE with
    | none => simp
    | some eb =>
      simp only [Option.bind_eq_bind, Option.bind_some]
      constructor
      · intro h
        refine ⟨nb, eb, rfl, rfl, ?_⟩
        by_cases hl : edges.length = relCount
        · refine ⟨hl, ?_⟩
          by_cases hz : 0 < edges.length
          · by_cases hbad : eb ≤ maxEdgeId edges ∨ nb ≤ maxEndpoint edges
            · simp [hl, hbad] at h; omega
            · intro e hm
              have h1 := (maxEdgeId_lt edges eb).1 (by omega)
              have h2 := (maxEndpoint_lt edges nb).1 (by omega)
              exact ⟨h1.2 e hm, (h2.2 e hm).1, (h2.2 e hm).2⟩
          · have : edges = [] := List.eq_nil_of_length_eq_zero (by omega)
            subst this; simp
        · simp [hl] at h
      · rintro ⟨nb', eb', hn', he', hl, h⟩
        cases hn'; cases he'
        subst hl
        by_cases hz : 0 < edges.length
        · have hne : edges ≠ [] := fun h => by simp [h] at hz
          obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil edges hne
          have h1 := (maxEdgeId_lt edges eb).2 ⟨by have := (h x hx).1; omega, fun e he => (h e he).1⟩
          have h2 := (maxEndpoint_lt edges nb).2
            ⟨by have := (h x hx).2.1; omega, fun e he => ⟨(h e he).2.1, (h e he).2.2⟩⟩
          have hbad : ¬(eb ≤ maxEdgeId edges ∨ nb ≤ maxEndpoint edges) := by omega
          simp [hbad]
        · simp [hz]

/-- **`restoreCheck_len_covers`**: on success every edge id is below `len` and
`len` is at most the edge bound (live + deleted edges) — the endpoint index
(`prepare_tiers(_, len)`, :982) is sized by the payload's counts. -/
theorem restoreCheck_len_covers (nodeCount relCount : Nat) (delN delE : List Nat)
    (edges : List (Nat × Nat × Nat)) (len : Nat)
    (h : restoreCheck nodeCount relCount delN delE edges = some len) :
    (∀ e ∈ edges, e.2.2 < len) ∧ len ≤ relCount + delE.length := by
  unfold restoreCheck at h
  cases hn : idBound nodeCount delN with
  | none => simp [hn] at h
  | some nb =>
    cases he : idBound relCount delE with
    | none => simp [hn, he] at h
    | some eb =>
      have heb := ((idBound_some_iff _ _ _).1 he).1
      simp only [hn, he, Option.bind_eq_bind, Option.bind_some] at h
      by_cases hl : edges.length = relCount
      · simp only [hl, ne_eq, not_true_eq_false, ite_false] at h
        by_cases hz : 0 < relCount
        · have hz' : 0 < edges.length := by omega
          by_cases hbad : eb ≤ maxEdgeId edges ∨ nb ≤ maxEndpoint edges
          · simp [hz, hbad] at h
          · simp only [hz, hbad, and_false, ite_false, ite_true, Option.some.injEq] at h
            subst h
            refine ⟨fun e hm => ?_, by omega⟩
            have := (maxEdgeId_lt edges (maxEdgeId edges + 1)).1 (by omega)
            exact this.2 e hm
        · have : edges = [] := List.eq_nil_of_length_eq_zero (by omega)
          subst this
          simp [hz] at h; subst h; simp
      · simp [hl] at h

/-- **`restoreCheck_accepts_dense`**: a saved graph passes. Its header counts
are the id spaces' `live`, its deleted lists their `recycled` sets, so the ids in
use and the deleted ids all lie below `live + |recycled|` (the `IdSpace`
invariant, id_space.rs), and the tensors hold its `relationship_count` edges. -/
theorem restoreCheck_accepts_dense (nodeCount relCount : Nat) (delN delE : List Nat)
    (edges : List (Nat × Nat × Nat))
    (hfitN : nodeCount + delN.length < 2 ^ 64) (hfitE : relCount + delE.length < 2 ^ 64)
    (hdN : ∀ x ∈ delN, x < nodeCount + delN.length) (hdE : ∀ x ∈ delE, x < relCount + delE.length)
    (hlen : edges.length = relCount)
    (hids : ∀ e ∈ edges, e.2.2 < relCount + delE.length ∧ e.1 < nodeCount + delN.length ∧
      e.2.1 < nodeCount + delN.length) :
    (restoreCheck nodeCount relCount delN delE edges).isSome :=
  (restoreCheck_some_iff _ _ _ _ _).2
    ⟨_, _, (idBound_some_iff _ _ _).2 ⟨rfl, hfitN, hdN⟩, (idBound_some_iff _ _ _).2 ⟨rfl, hfitE, hdE⟩,
      hlen, hids⟩

/-! The PR's `restore_refuses_ids_the_counts_do_not_account_for` cases. -/

theorem restore_accepts_test : restoreCheck 2 2 [] [] [(0, 1, 0), (1, 0, 1)] = some 2 := by decide
theorem restore_refuses_edge_id : restoreCheck 2 1 [] [] [(0, 1, 2 ^ 40)] = none := by decide
theorem restore_refuses_endpoint : restoreCheck 2 1 [] [] [(0, 9, 0)] = none := by decide
theorem restore_refuses_edge_count : restoreCheck 2 2 [] [] [(0, 1, 0)] = none := by decide
theorem restore_refuses_deleted_id : restoreCheck 2 0 [99] [] [] = none := by decide
theorem restore_refuses_overflow : restoreCheck (2 ^ 64 - 1) 0 [0] [] [] = none := by decide

end GQ
