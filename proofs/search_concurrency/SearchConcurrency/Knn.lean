import SearchConcurrency.Search
/-
# KNN vector queries: `Graph::vector_query_nodes` / `vector_query_edges` @ 89d68334a (#3088)

| here | there |
| --- | --- |
| `vectorQuery` (generic over the id type) | the shared body of both fns below |
| `vectorQueryNodes`  | `graph/src/graph/graph.rs:3978-4032` `vector_query_nodes`, `count = node_count()` (`:1040`) |
| `vectorQueryEdges`  | `graph.rs:4038-4088` `vector_query_edges`, `count = relationship_count()` (`:1057`) |
| `clampK`            | `graph.rs:3998` / `:4057` `k.min(count.max(1) as usize)` |
| `KnnIndex.query`    | `node_indexer.vector_query` / `edge_indexer.vector_query_edges` (RediSearch FFI) |
| `dimCheck`          | `graph.rs:3986-3993` / `:4046-4053` dimension-mismatch `Err` |
| `attrKnown = false` | `graph.rs:4004-4006` / `:4063-4065` `return Ok(Vec::new().into_iter())` |
| `outCap`            | `graph.rs:4014-4015` / `:4073-4074` `Vec::with_capacity(node_ids.len())` / `(triples.len())` |
| `filterMap step`    | `graph.rs:4016-4025` / `:4075-4083` `let Value::VecF32 .. else continue` + `vec_distance::distance` |
| `mergeSort`         | `graph.rs:4030` / `:4086` stable `sort_by(partial_cmp .. unwrap_or(Equal))` |

RediSearch is a hypothesis structure (`KnnIndex`), not an axiom: it holds `ranked`, the
index's documents in its own (approximate-distance) order, at most `count` of them
because every indexed document is a live entity, and `query k = ranked.take k`.
Floats are abstract (`FloatOps`).

HISTORICAL (W3-conc-4 / #3085, fixed by #3088 at 89d68334a): before the fix both fns did
`Vec::with_capacity(k)` with the raw user `k` (`Search.pre3088_huge_k_kills_server`). Now
every buffer is bounded by the live entity count: `alloc_bounded`, `no_abort_any_k`.
-/

namespace SC.Knn
open SC.Search

/-- Abstract f64: only the comparison used by `sort_by` matters. -/
structure FloatOps (D : Type) where
  le : D → D → Bool

/-- RediSearch KNN over one index (FFI boundary as a hypothesis structure). -/
structure KnnIndex (α : Type) (count : Nat) where
  ranked : List α
  /-- every indexed document is a live entity. -/
  ranked_le : ranked.length ≤ count
  query : Nat → List α
  query_spec : ∀ k, query k = ranked.take k

/-- `k.min(count.max(1) as usize)` (graph.rs:3998 / :4057). -/
def clampK (k count : Nat) : Nat := min k (max count 1)

/-- One loop iteration: `let Value::VecF32(v) = entity else continue;` then
`if let Some(d) = vec_distance::distance(..) { out.push((id, d)) }`. -/
def score {α V D : Type} (vecOf : α → Option V) (dist : V → Option D) (i : α) : Option (α × D) :=
  match (vecOf i).bind dist with
  | some d => some (i, d)
  | none => none

theorem filterMap_length_all {α β} (f : α → Option β) :
    ∀ l : List α, (∀ a ∈ l, (f a).isSome) → (l.filterMap f).length = l.length
  | [], _ => rfl
  | a :: l, h => by
    have ha := h a (List.mem_cons_self ..)
    have ih := filterMap_length_all f l (fun b hb => h b (List.mem_cons_of_mem _ hb))
    cases hfa : f a with
    | none => simp [hfa] at ha
    | some b => simp [hfa, ih]

structure Out (α D : Type) where
  /-- `k` handed to RediSearch. -/
  rsK : Nat
  /-- capacity of `out` (`Vec::with_capacity`). -/
  outCap : Nat
  /-- capacity of `vecs`. -/
  vecsCap : Nat
  rows : List (α × D)

/-- The shared body of `vector_query_nodes` / `vector_query_edges`, branch for branch.
`dimOk` is the `get_vector_dimension` check; `attrKnown` the attribute-id lookup;
`vecOf` the batched attribute fetch (`None` = not a `Value::VecF32`); `dist` is
`vec_distance::distance` against the query vector. -/
def vectorQuery {α V D : Type} (F : FloatOps D) (count : Nat) (idx : KnnIndex α count)
    (dimOk attrKnown : Bool) (vecOf : α → Option V) (dist : V → Option D) (k : Nat) :
    Except String (Out α D) :=
  if !dimOk then .error "Vector dimension mismatch"
  else
    let k := clampK k count
    let raw := idx.query k
    if !attrKnown then .ok ⟨k, 0, 0, []⟩
    else
      let ids := raw
      let out := ids.filterMap (score vecOf dist)
      .ok ⟨k, ids.length, ids.length, out.mergeSort (fun a b => F.le a.2 b.2)⟩

/-- Node ids. -/
abbrev vectorQueryNodes {V D : Type} (F : FloatOps D) (nodeCount : Nat) :=
  @vectorQuery Nat V D F nodeCount
/-- `(src, dst, edge_id)` triples. -/
abbrev vectorQueryEdges {V D : Type} (F : FloatOps D) (relCount : Nat) :=
  @vectorQuery (Nat × Nat × Nat) V D F relCount

theorem clampK_le_k (k c : Nat) : clampK k c ≤ k := Nat.min_le_left _ _
theorem clampK_le_count (k c : Nat) : clampK k c ≤ max c 1 := Nat.min_le_right _ _

/-- **Allocation bounded by the live entity count, for every user `k`.** -/
theorem alloc_bounded {α V D} (F : FloatOps D) (c : Nat) (idx : KnnIndex α c) dimOk attrKnown
    (vecOf : α → Option V) dist k o (h : vectorQuery F c idx dimOk attrKnown vecOf dist k = .ok o) :
    o.rsK ≤ max c 1 ∧ o.outCap ≤ c ∧ o.vecsCap ≤ c ∧ o.rows.length ≤ c := by
  unfold vectorQuery at h
  have hb := clampK_le_count k c
  have hr := idx.ranked_le
  cases dimOk <;> cases attrKnown <;> simp at h <;> subst h
  · simp; omega
  · simp only [idx.query_spec, List.length_take, List.length_mergeSort]
    have := List.length_filterMap_le (score vecOf dist) (idx.ranked.take (clampK k c))
    simp only [List.length_take] at this
    refine ⟨hb, ?_, ?_, ?_⟩ <;> omega

/-- Hence no `capacity overflow` panic and no OOM abort, whatever `k` is, provided the
allocator can hold one 16-byte row per live entity (`(NodeId, f64)`). -/
theorem no_abort_any_k {α V D} (F : FloatOps D) (c : Nat) (idx : KnnIndex α c) dimOk attrKnown
    (vecOf : α → Option V) dist k o mem
    (h : vectorQuery F c idx dimOk attrKnown vecOf dist k = .ok o)
    (hm : c * 16 ≤ mem) (hi : c * 16 ≤ isizeMax) :
    withCapacity 16 o.outCap mem = .ok (o.outCap * 16) := by
  have := (alloc_bounded F c idx dimOk attrKnown vecOf dist k o h).2.1
  have h16 : o.outCap * 16 ≤ c * 16 := Nat.mul_le_mul_right _ this
  unfold withCapacity
  simp only [show ¬ (o.outCap * 16 > isizeMax) by omega, show ¬ (o.outCap * 16 > mem) by omega,
    ite_false]

/-- **Result = top-`min(k, count)`**: the rows are exactly RediSearch's first `min k |index|`
candidates (scored and re-sorted), and the clamp never loses one — the same rows as the
unclamped query `ranked.take k`. -/
theorem candidates_topk {α} (c : Nat) (idx : KnnIndex α c) (k : Nat) :
    idx.query (clampK k c) = idx.ranked.take k ∧
    (idx.query (clampK k c)).length = min k idx.ranked.length := by
  have hr := idx.ranked_le
  rw [idx.query_spec]
  constructor
  · apply List.ext_getElem
    · simp only [List.length_take, clampK]; omega
    · intro n _ _; simp
  · simp only [List.length_take, clampK]; omega

/-- When every candidate has a vector and a defined distance, the result has exactly
`min k |index|` rows, each a candidate. -/
theorem rows_length_topk {α V D} (F : FloatOps D) (c : Nat) (idx : KnnIndex α c)
    (vecOf : α → Option V) dist k o
    (h : vectorQuery F c idx true true vecOf dist k = .ok o)
    (hall : ∀ i ∈ idx.ranked, ∃ d, (vecOf i).bind dist = some d) :
    o.rows.length = min k idx.ranked.length ∧
    ∀ r ∈ o.rows, r.1 ∈ idx.ranked.take k := by
  unfold vectorQuery at h
  simp at h; subst h
  have ⟨heq, hlen⟩ := candidates_topk c idx k
  simp only [List.length_mergeSort]
  constructor
  · rw [← hlen]
    apply filterMap_length_all
    intro i hi
    rw [heq] at hi
    obtain ⟨d, hd⟩ := hall i (List.mem_of_mem_take hi)
    simp [score, hd]
  · intro r hr
    rw [List.mem_mergeSort, List.mem_filterMap] at hr
    obtain ⟨i, hi, hm⟩ := hr
    rw [← heq]
    unfold score at hm
    split at hm
    · cases hm; exact hi
    · cases hm

/-- The #3085 repro, now: one live node, `k = 10^15` → RediSearch is asked for 1, one
16-byte row is reserved, and the one row comes back (C returns the same 1 row). -/
def oneNode : KnnIndex Nat 1 := ⟨[0], by decide, fun k => [0].take k, fun _ => rfl⟩

theorem huge_k_one_row :
    (vectorQueryNodes (V := Unit) (D := Nat) ⟨Nat.ble⟩ 1 oneNode true true (fun _ => some ())
      (fun _ => some 0) 1000000000000000).map (fun o => (o.rsK, o.outCap, o.rows)) =
      .ok (1, 1, [(0, 0)]) := by
  simp [vectorQuery, clampK, oneNode, score, Except.map]

/-- Empty graph: `count.max(1)` keeps `k ≥ 1` for RediSearch (a `k = 0` query is not issued)
and nothing is reserved. -/
theorem empty_graph {α V D} (F : FloatOps D) (idx : KnnIndex α 0) (vecOf : α → Option V) dist k
    (hk : 0 < k) :
    vectorQuery F 0 idx true true vecOf dist k = .ok ⟨1, 0, 0, []⟩ := by
  have hr := idx.ranked_le
  have hnil : idx.ranked = [] := List.eq_nil_of_length_eq_zero (by omega)
  have hc : clampK k 0 = 1 := by simp [clampK]; omega
  simp [vectorQuery, hc, idx.query_spec, hnil]

end SC.Knn
