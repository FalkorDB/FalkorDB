import FalkorEndpointIndex

/-!
# `Graph::restore` and `Graph::delete_implicit_edges`, whole

`FalkorEndpointIndex` proves the index; this file models the two `Graph` functions that drive it
in bulk, end to end: what they compute from their arguments, and what the index reads afterwards.
What each function does to stores other projects prove is stated as what it passes them
(`IdSpace::restored/release` is `id_space`, `AttributeStore::remove_all` is `attribute_store`,
`Tensor::remove_all` and the GraphBLAS masks are `versioned_matrix`/`graphblas_wrappers`).

| Lean | Rust (origin/main 256ab4d3f) |
| --- | --- |
| `Scan`, `scanStep`            | the first pass of `Graph::restore` (`graph.rs:900`): `max_edge_id` and `starts` over `tensor.iter_edges()` (`:932-942`) |
| `restoreLen`, `restoreStarts` | `len` (`:944-948`), the monotone clamp loop (`:951-955`) |
| `restoreIndex`                | `EndpointIndex::default()` then `prepare_tiers` (`:958`) then `set` per edge (`:959-962`) |
| `capOf`, `Restored`           | `node_cap`/`relationship_cap` `next_multiple_of(chunk).max(64)`, `schema_version` (`:970`), `version: 0` (`:1002`), empty indexers/constraints (`:996-1001`), (`:973-974`) |
| `deletedEdges`, `DE`          | the per-type collection and `all_implicit` of `Graph::delete_implicit_edges` (`:3102`) |
| `stagedDocs`                  | `index_remove_edge_docs.entry(type_id).or_default().insert(edge_id, (src, dst))` for an indexed type (`:3191`) |
| `tensorsAfter`, `emptiedPairs`, `clearedPairs` | `Tensor::remove_all` (`:3210`) and the adjacency clearing (`dead_adj_pairs` `:3223`, `check_adj_pairs` `:3225`, the bulk clear `:3267-3279`) |
-/
namespace Falkor.EndpointIndex

/-! ## `Graph::restore` -/

/-- The first pass: the largest edge id, and `starts[r-1]`, the smallest edge id whose own
endpoints need at least rank `r` (initially `u64::MAX`). Edges are `(edge_id, src, dst)`. -/
structure Scan where
  maxId : Nat
  s1 : Nat
  s2 : Nat
  s3 : Nat
deriving DecidableEq, Repr

def M64 : Nat := 2 ^ 64 - 1

def Scan.init : Scan := ⟨0, M64, M64, M64⟩

/-- One edge: `rank = rank_for(src.max(dst))`, `for r in 1..=rank { starts[r-1] = starts[r-1].min(edge_id) }`. -/
def scanStep (a : Scan) (e : Nat × Nat × Nat) : Scan :=
  let rank := rankFor (max e.2.1 e.2.2)
  { maxId := max a.maxId e.1
    s1 := if 1 ≤ rank then min a.s1 e.1 else a.s1
    s2 := if 2 ≤ rank then min a.s2 e.1 else a.s2
    s3 := if 3 ≤ rank then min a.s3 e.1 else a.s3 }

def scanAll (edges : List (Nat × Nat × Nat)) : Scan := edges.foldl scanStep Scan.init

/-- `len`: `max_edge_id + 1` if any tensor holds an edge, else 0. -/
def restoreLen (edges : List (Nat × Nat × Nat)) : Nat :=
  if edges.isEmpty then 0 else (scanAll edges).maxId + 1

/-- The clamp loop: `acc = len; for slot in starts.iter_mut().rev() { acc = acc.min(*slot); *slot = acc }`. -/
def restoreStarts (edges : List (Nat × Nat × Nat)) : Nat × Nat × Nat :=
  let sc := scanAll edges
  let len := restoreLen edges
  let a3 := min len sc.s3
  let a2 := min a3 sc.s2
  let a1 := min a2 sc.s1
  (a1, a2, a3)

/-- The restored index: `prepare_tiers(starts, len)` on an empty index, then `set` for every edge. -/
def restoreIndex (edges : List (Nat × Nat × Nat)) : Ix :=
  let st := restoreStarts edges
  run (({} : Ix).prepareTiers st.1 st.2.1 st.2.2 (restoreLen edges))
    (edges.map fun x => Op.set x.1 x.2.1 x.2.2)

/-- **The starts are ordered and inside `len`**: no tier size `s2 - s1`, `s3 - s2`, `len - s3` underflows
and the four tiers sum to `len`; `prepare_tiers`' own clamp is then a no-op. -/
theorem restoreStarts_ordered (edges : List (Nat × Nat × Nat)) :
    (restoreStarts edges).1 ≤ (restoreStarts edges).2.1 ∧
    (restoreStarts edges).2.1 ≤ (restoreStarts edges).2.2 ∧
    (restoreStarts edges).2.2 ≤ restoreLen edges := by
  simp only [restoreStarts]; omega

theorem restore_clamp_noop (edges : List (Nat × Nat × Nat)) :
    Ix.clampStarts (restoreStarts edges).1 (restoreStarts edges).2.1 (restoreStarts edges).2.2
      (restoreLen edges) = restoreStarts edges :=
  Ix.clampStarts_ordered (restoreStarts_ordered edges)

/-- **The restored index answers exactly the restored edge set**, for the starts the first pass
computed (`restore_agrees` for any starts; here they are `restoreStarts`). -/
theorem restoreIndex_get (edges : List (Nat × Nat × Nat)) (hv : ∀ x ∈ edges, Valid x.2.1 x.2.2) (i : Nat) :
    (restoreIndex edges).get i =
      refRun (fun _ => none) (edges.map fun x => Op.set x.1 x.2.1 x.2.2) i :=
  restore_agrees _ _ _ _ edges hv i

/-- With nothing to restore the index is empty: no slot is allocated (`len = 0`, every tier empty). -/
theorem restoreIndex_nil : restoreLen [] = 0 ∧ restoreStarts [] = (0, 0, 0) := by
  simp [restoreLen, restoreStarts, scanAll, Scan.init]

theorem scanStep_maxId (a : Scan) (e : Nat × Nat × Nat) : (scanStep a e).maxId = max a.maxId e.1 := rfl

theorem scanStep_s1 (a : Scan) (e : Nat × Nat × Nat) :
    (scanStep a e).s1 = if 1 ≤ rankFor (max e.2.1 e.2.2) then min a.s1 e.1 else a.s1 := rfl

theorem scan_maxId_mono : ∀ (r : List (Nat × Nat × Nat)) (a : Scan),
    a.maxId ≤ (r.foldl scanStep a).maxId
  | [], a => Nat.le_refl _
  | x :: r, a => by
    simp only [List.foldl_cons]
    have h1 := scan_maxId_mono r (scanStep a x)
    rw [scanStep_maxId] at h1
    omega

theorem scan_maxId_ge : ∀ (edges : List (Nat × Nat × Nat)) (a : Scan),
    ∀ e ∈ edges, e.1 ≤ (edges.foldl scanStep a).maxId
  | [], _, e, he => by simp at he
  | x :: r, a, e, he => by
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 he with rfl | he
    · have h1 := scan_maxId_mono r (scanStep a e)
      rw [scanStep_maxId] at h1
      omega
    · exact scan_maxId_ge r (scanStep a x) e he

/-- **No edge id is past the allocated length**: every restored edge id is below `len`, so `set`
never has to grow a tier the preparation did not size. -/
theorem restore_ids_lt_len (edges : List (Nat × Nat × Nat)) (e : Nat × Nat × Nat) (he : e ∈ edges) :
    e.1 < restoreLen edges := by
  have hne : edges.isEmpty = false := by
    cases edges with
    | nil => simp at he
    | cons _ _ => rfl
  simp only [restoreLen, hne, Bool.false_eq_true, ite_false, scanAll]
  have := scan_maxId_ge edges Scan.init e he
  omega

/-! ### The tier layout is what incremental growth would produce

Tier `r` (r = 1, 2, 3) begins at `starts[r-1]`. An edge whose endpoints need rank `r` has an id at or
past the start of tier `r`, so the layout puts it in a tier that is wide enough: restore never needs
a promotion, and the narrow tiers keep the early edges (6 bytes per edge instead of 8 or more). -/

theorem scan_s1_mono : ∀ (r : List (Nat × Nat × Nat)) (a : Scan), (r.foldl scanStep a).s1 ≤ a.s1
  | [], a => Nat.le_refl _
  | x :: r, a => by
    simp only [List.foldl_cons]
    have h1 := scan_s1_mono r (scanStep a x)
    rw [scanStep_s1] at h1
    split at h1 <;> omega

theorem scan_s1_le : ∀ (edges : List (Nat × Nat × Nat)) (a : Scan),
    ∀ e ∈ edges, 1 ≤ rankFor (max e.2.1 e.2.2) → (edges.foldl scanStep a).s1 ≤ e.1
  | [], _, e, he, _ => by simp at he
  | x :: r, a, e, he, hr => by
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 he with rfl | he
    · have h1 := scan_s1_mono r (scanStep a e)
      rw [scanStep_s1] at h1
      simp only [hr, ite_true] at h1
      omega
    · exact scan_s1_le r (scanStep a x) e he hr

/-- The first boundary: an edge needing rank 1 or more has an id at or past `starts[0]`. -/
theorem restore_tier1_start (edges : List (Nat × Nat × Nat)) (e : Nat × Nat × Nat) (he : e ∈ edges)
    (hr : 1 ≤ rankFor (max e.2.1 e.2.2)) : (restoreStarts edges).1 ≤ e.1 := by
  have h := scan_s1_le edges Scan.init e he hr
  simp only [restoreStarts, scanAll]
  omega

/-! ### The rest of the struct -/

/-- `u64::next_multiple_of`: the smallest multiple of `c` that is `≥ n`; `c = 0` panics (`none`). -/
def nextMultipleOf (n c : Nat) : Option Nat :=
  if c = 0 then none else some ((n + c - 1) / c * c)

/-- `bound.next_multiple_of(chunk).max(64)`: a capacity of at least 64. -/
def capOf (bound chunk : Nat) : Option Nat := (nextMultipleOf bound chunk).map (max 64)

/-- What `restore` builds from its arguments, besides the matrices and stores it is handed. -/
structure Restored where
  nodeCap : Nat
  relCap : Nat
  schemaVersion : Nat
  version : Nat
  indexersEmpty : Bool
  constraintsEmpty : Bool
deriving DecidableEq, Repr

/-- The scalar fields of `Self { .. }` (`graph.rs:969-1010`): `bound`s from `IdSpace::restored(count,
deleted).bound()`, `chunk` is `NODE_CREATION_BUFFER`. `none` is the `next_multiple_of(0)` panic. -/
def restoredFields (nodeBound relBound chunk labels types : Nat) : Option Restored := do
  let nc ← capOf nodeBound chunk
  let rc ← capOf relBound chunk
  pure { nodeCap := nc, relCap := rc, schemaVersion := labels + types, version := 0,
         indexersEmpty := true, constraintsEmpty := true }

theorem nextMultipleOf_spec (n c : Nat) (hc : 0 < c) :
    ∃ m, nextMultipleOf n c = some m ∧ n ≤ m ∧ m < n + c ∧ m % c = 0 := by
  refine ⟨(n + c - 1) / c * c, by simp [nextMultipleOf, Nat.pos_iff_ne_zero.1 hc], ?_⟩
  have h1 := Nat.div_add_mod (n + c - 1) c
  have h2 := Nat.mod_lt (n + c - 1) hc
  have h3 : (n + c - 1) / c * c % c = 0 := Nat.mul_mod_left _ _
  refine ⟨?_, ?_, h3⟩
  · have : (n + c - 1) / c * c + (n + c - 1) % c = n + c - 1 := by
      rw [Nat.mul_comm]; exact h1
    omega
  · have : (n + c - 1) / c * c ≤ n + c - 1 := Nat.div_mul_le_self _ _
    omega

/-- **The capacities** are at least the id-space bound and at least 64, and a zero `chunk` is the
`next_multiple_of` panic. -/
theorem capOf_spec (bound chunk : Nat) :
    (chunk = 0 → capOf bound chunk = none) ∧
    (0 < chunk → ∃ r, capOf bound chunk = some r ∧ bound ≤ r ∧ 64 ≤ r ∧
      (r = 64 ∨ r % chunk = 0)) := by
  refine ⟨fun h => by simp [capOf, nextMultipleOf, h], fun hc => ?_⟩
  obtain ⟨m, hm, h1, _, h3⟩ := nextMultipleOf_spec bound chunk hc
  refine ⟨max 64 m, by simp [capOf, hm], by omega, by omega, ?_⟩
  by_cases h : 64 ≤ m
  · right; rw [Nat.max_eq_right h]; exact h3
  · left; omega

/-- The other scalar fields: `schema_version` is the number of labels plus relationship types, the
version is 0, no index or constraint is carried over (the RDB carries none). -/
theorem restoredFields_spec (nb rb chunk labels types : Nat) (hc : 0 < chunk) :
    ∃ r, restoredFields nb rb chunk labels types = some r ∧ r.schemaVersion = labels + types ∧
      r.version = 0 ∧ r.indexersEmpty = true ∧ r.constraintsEmpty = true ∧
      nb ≤ r.nodeCap ∧ rb ≤ r.relCap ∧ 64 ≤ r.nodeCap ∧ 64 ≤ r.relCap := by
  obtain ⟨a, ha, h1, h2, _⟩ := (capOf_spec nb chunk).2 hc
  obtain ⟨b, hb, h3, h4, _⟩ := (capOf_spec rb chunk).2 hc
  exact ⟨⟨a, b, labels + types, 0, true, true⟩, by simp [restoredFields, ha, hb], rfl, rfl, rfl, rfl, h1, h3, h2, h4⟩

theorem restoredFields_zero_chunk (nb rb labels types : Nat) :
    restoredFields nb rb 0 labels types = none := by
  simp [restoredFields, capOf, nextMultipleOf]


/-! ## `Graph::delete_implicit_edges` -/

/-- One implicitly deleted edge (`DeletedEdge`): id, endpoints, relationship type. -/
structure DE where
  id : Nat
  src : Nat
  dst : Nat
  typ : Nat
deriving DecidableEq, Repr

/-- `all_implicit` (`graph.rs:3102`): per relationship type in order, the edges `collect` finds, as
`DeletedEdge`s (`:3180`). `N` = `deleted_nodes`, `X` = `explicit_rels`, `T` = the tensors, edges
`(edge_id, src, dst)`. No tensors: the early `Ok(Vec::new())`. -/
def deletedEdges (N X : List Nat) (T : List (List Edge)) : List DE :=
  if T.isEmpty then []
  else T.zipIdx.flatMap fun p => (collect N X p.1).map fun e => ⟨e.1, e.2.1, e.2.2, p.2⟩

/-- **What is returned**: exactly the edges of some type's tensor that touch a deleted node and are not
explicit, each with its type. -/
theorem deletedEdges_mem (N X : List Nat) (T : List (List Edge)) (d : DE) :
    d ∈ deletedEdges N X T ↔
      ∃ E, T[d.typ]? = some E ∧ (d.id, d.src, d.dst) ∈ E ∧ (d.src ∈ N ∨ d.dst ∈ N) ∧ d.id ∉ X := by
  unfold deletedEdges
  by_cases hT : T.isEmpty = true
  · have : T = [] := List.isEmpty_iff.1 hT
    subst this; simp
  · simp only [hT, Bool.false_eq_true, ite_false, List.mem_flatMap, List.mem_map, mem_collect]
    constructor
    · rintro ⟨⟨E, t⟩, hm, e, ⟨he, hN, hx⟩, rfl⟩
      have := List.mem_zipIdx_iff_getElem?.1 hm
      exact ⟨E, this, he, hN, hx⟩
    · rintro ⟨E, hE, he, hN, hx⟩
      exact ⟨⟨E, d.typ⟩, List.mem_zipIdx_iff_getElem?.2 hE, (d.id, d.src, d.dst), ⟨he, hN, hx⟩, by cases d; rfl⟩

/-- `remove_all(&rels)`: the tensor without the collected edges. -/
def tensorAfter (N X : List Nat) (E : List Edge) : List Edge :=
  E.filter fun e => decide (e ∉ collect N X E)

theorem tensorAfter_mem (N X : List Nat) (E : List Edge) (e : Edge) :
    e ∈ tensorAfter N X E ↔ e ∈ E ∧ ¬ ((e.2.1 ∈ N ∨ e.2.2 ∈ N) ∧ e.1 ∉ X) := by
  simp only [tensorAfter, List.mem_filter, decide_eq_true_eq, mem_collect]
  constructor
  · rintro ⟨he, hn⟩; exact ⟨he, fun h => hn ⟨he, h⟩⟩
  · rintro ⟨he, hn⟩; exact ⟨he, fun h => hn h.2⟩

/-- The pairs that lost their last edge in this tensor (`Tensor::remove_all`'s return, each pair once). -/
def emptiedPairs (N X : List Nat) (E : List Edge) : List (Nat × Nat) :=
  (((collect N X E).filter fun e => !(tensorAfter N X E).any fun f => f.2 == e.2).map (·.2)).eraseDups

theorem emptiedPairs_mem (N X : List Nat) (E : List Edge) (p : Nat × Nat) :
    p ∈ emptiedPairs N X E ↔
      (∃ e ∈ collect N X E, e.2 = p) ∧ ∀ f ∈ tensorAfter N X E, f.2 ≠ p := by
  simp only [emptiedPairs, List.mem_eraseDups, List.mem_map, List.mem_filter, Bool.not_eq_true',
    List.any_eq_false, beq_iff_eq]
  constructor
  · rintro ⟨e, ⟨he, hn⟩, rfl⟩
    exact ⟨⟨e, he, rfl⟩, fun f hf => by simpa using hn f hf⟩
  · rintro ⟨⟨e, he, rfl⟩, hn⟩
    exact ⟨e, ⟨he, fun f hf => by simpa using hn f hf⟩, rfl⟩

/-- The adjacency pairs cleared (`dead_adj_pairs` plus the `check_adj_pairs` that have no edge left in
any tensor, asked after every type was processed). -/
def clearedAdj (N X : List Nat) (T : List (List Edge)) : List (Nat × Nat) :=
  let em := T.flatMap (emptiedPairs N X)
  em.filter (fun p => decide (p.1 ∈ N ∧ p.2 ∈ N)) ++
    (em.filter fun p => !decide (p.1 ∈ N ∧ p.2 ∈ N)).filter fun p =>
      !(T.map (tensorAfter N X)).any fun E' => E'.any fun f => f.2 == p

/-- **The adjacency rule**: a pair is cleared iff some tensor lost its last edge on it and either both
its endpoints are deleted (no edge between them can survive the commit) or no tensor has an edge left on
it. -/
theorem clearedAdj_mem (N X : List Nat) (T : List (List Edge)) (p : Nat × Nat) :
    p ∈ clearedAdj N X T ↔
      (∃ E ∈ T, p ∈ emptiedPairs N X E) ∧
      ((p.1 ∈ N ∧ p.2 ∈ N) ∨ ∀ E ∈ T, ∀ f ∈ tensorAfter N X E, f.2 ≠ p) := by
  simp only [clearedAdj, List.mem_append, List.mem_filter, List.mem_flatMap, decide_eq_true_eq,
    Bool.not_eq_true', decide_eq_false_iff_not, List.any_eq_false, List.mem_map,
    Bool.not_eq_true']
  constructor
  · rintro (⟨⟨E, hE, hp⟩, hd⟩ | ⟨⟨⟨E, hE, hp⟩, hd⟩, hn⟩)
    · exact ⟨⟨E, hE, hp⟩, Or.inl hd⟩
    · refine ⟨⟨E, hE, hp⟩, Or.inr fun E' hE' f hf hfp => ?_⟩
      have h0 := hn (tensorAfter N X E') ⟨E', hE', rfl⟩
      exact h0 (List.any_eq_true.2 ⟨f, hf, by simp [hfp]⟩)
  · rintro ⟨⟨E, hE, hp⟩, hd | hn⟩
    · exact Or.inl ⟨⟨E, hE, hp⟩, hd⟩
    · by_cases hd : p.1 ∈ N ∧ p.2 ∈ N
      · exact Or.inl ⟨⟨E, hE, hp⟩, hd⟩
      · refine Or.inr ⟨⟨⟨E, hE, hp⟩, hd⟩, ?_⟩
        intro L hL
        obtain ⟨E', hE', rfl⟩ := hL
        have h1 : ((tensorAfter N X E').any fun f => f.2 == p) = false := by
          simp only [List.any_eq_false, beq_iff_eq]
          exact fun f hf hfp => hn E' hE' f hf hfp
        simp [h1]

/-- A pair with an edge still standing (and an endpoint that survives) keeps its adjacency bit: the
survivor may still hold an edge of another type. -/
theorem adj_kept_when_edge_remains (N X : List Nat) (T : List (List Edge)) (p : Nat × Nat)
    (E : List Edge) (hE : E ∈ T) (f : Edge) (hf : f ∈ tensorAfter N X E) (hfp : f.2 = p)
    (hnd : ¬ (p.1 ∈ N ∧ p.2 ∈ N)) : p ∉ clearedAdj N X T := by
  intro h
  rcases (clearedAdj_mem N X T p).1 h with ⟨_, hd | hn⟩
  · exact hnd hd
  · exact hn E hE f hf hfp

/-- A pair whose last edge was collected, with no edge left anywhere, is cleared. -/
theorem adj_cleared_when_last_gone (N X : List Nat) (T : List (List Edge)) (p : Nat × Nat)
    (E : List Edge) (hE : E ∈ T) (e : Edge) (he : e ∈ collect N X E) (hep : e.2 = p)
    (hgone : ∀ E' ∈ T, ∀ f ∈ tensorAfter N X E', f.2 ≠ p) : p ∈ clearedAdj N X T := by
  refine (clearedAdj_mem N X T p).2 ⟨⟨E, hE, (emptiedPairs_mem N X E p).2 ⟨⟨e, he, hep⟩, hgone E hE⟩⟩, Or.inr hgone⟩

/-- The staged edge-index document removals (`index_remove_edge_docs`): for an indexed type, one
`(edge_id ↦ (src, dst))` per collected edge. -/
def stagedDocs (isIdx : Nat → Bool) (N X : List Nat) (T : List (List Edge)) : List (Nat × Nat × Nat × Nat) :=
  ((deletedEdges N X T).filter fun d => isIdx d.typ).map fun d => (d.typ, d.id, d.src, d.dst)

theorem stagedDocs_mem (isIdx : Nat → Bool) (N X : List Nat) (T : List (List Edge)) (t i s d : Nat) :
    (t, i, s, d) ∈ stagedDocs isIdx N X T ↔
      isIdx t = true ∧ ∃ E, T[t]? = some E ∧ (i, s, d) ∈ E ∧ (s ∈ N ∨ d ∈ N) ∧ i ∉ X := by
  simp only [stagedDocs, List.mem_map, List.mem_filter, Prod.mk.injEq]
  constructor
  · rintro ⟨x, ⟨hx, hi⟩, rfl, rfl, rfl, rfl⟩
    obtain ⟨E, h1, h2, h3, h4⟩ := (deletedEdges_mem N X T x).1 hx
    exact ⟨hi, E, h1, h2, h3, h4⟩
  · rintro ⟨hi, E, h1, h2, h3, h4⟩
    exact ⟨⟨i, s, d, t⟩, ⟨(deletedEdges_mem N X T ⟨i, s, d, t⟩).2 ⟨E, h1, h2, h3, h4⟩, hi⟩, rfl, rfl, rfl, rfl⟩

/-- The endpoint index after the clear loop (`clear_edge_endpoint` per collected id, `graph.rs:3209`). -/
def clearImplicit (ix : Ix) (N X : List Nat) (T : List (List Edge)) : Ix :=
  run ix ((deletedEdges N X T).map fun d => Op.clear d.id)

/-- **The index after `delete_implicit_edges`**: every collected id reads as absent, every other id as
before (a well-formed index; `detach_delete_index` per type, here for all types at once). -/
theorem clearImplicit_get (ix : Ix) (hwf : WF ix) (N X : List Nat) (T : List (List Edge)) (i : Nat) :
    (clearImplicit ix N X T).get i =
      if i ∈ (deletedEdges N X T).map DE.id then none else ix.get i := by
  have := (run_refines ((deletedEdges N X T).map fun d => Op.clear d.id) ix ix.get hwf (fun _ => rfl)
    (by intro op hop; obtain ⟨_, _, rfl⟩ := List.mem_map.1 hop; trivial)).2 i
  unfold clearImplicit
  rw [this]
  have hm : (deletedEdges N X T).map (fun d => Op.clear d.id) =
      ((deletedEdges N X T).map DE.id).map Op.clear := by simp [List.map_map, Function.comp_def]
  rw [hm, refRun_clears]

#print axioms restoreIndex_get
#print axioms restoreStarts_ordered
#print axioms restoredFields_spec
#print axioms deletedEdges_mem
#print axioms clearedAdj_mem
#print axioms clearImplicit_get

end Falkor.EndpointIndex
