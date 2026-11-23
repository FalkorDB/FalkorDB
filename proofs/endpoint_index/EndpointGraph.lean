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
| `Scan`, `scanStep`            | the first pass of `Graph::restore` (`graph.rs`): `max_edge_id` and `starts` over `tensor.iter_edges()` |
| `restoreLen`, `restoreStarts` | `len`, the monotone clamp loop |
| `restoreIndex`                | `EndpointIndex::default()` then `prepare_tiers` then `set` per edge |
| `idBound`, `maxEndpoint`, `restoreCheck` | `Graph::id_bound` and the live-edge count / `max_edge_id` / `max_endpoint` checks that make `restore` an `Err` (#3643) |
| `capOf`, `boundOf`, `Restored` | `node_cap`/`relationship_cap` `next_multiple_of(chunk).max(64)` and `IdSpace::bound` (release `u64` wrap modelled), `schema_version`, `version: 0`, empty indexers/constraints, |
| `deletedEdges`, `DE`          | the per-type collection and `all_implicit` of `Graph::delete_implicit_edges` |
| `stagedDocs`                  | `index_remove_edge_docs.entry(type_id).or_default().insert(edge_id, (src, dst))` for an indexed type |
| `tensorsAfter`, `emptiedPairs`, `clearedPairs` | `Tensor::remove_all` and the adjacency clearing (`dead_adj_pairs`, `check_adj_pairs`, the bulk clear) |
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

/-- `2^64`: the `u64` modulus. The release profile has no `overflow-checks`, so `u64` `+` wraps
(a debug build panics instead). -/
def W : Nat := 2 ^ 64

/-- `u64::next_multiple_of` (`match self % rhs { 0 => self, r => self + (rhs - r) }`) as the release
build runs it: `c = 0` panics (`none`), and the addition wraps mod `2^64`. -/
def nextMultipleOf (n c : Nat) : Option Nat :=
  if c = 0 then none else some (if n % c = 0 then n else (n + (c - n % c)) % W)

/-- `bound.next_multiple_of(chunk).max(64)`: a capacity of at least 64 (barring wrap-around). -/
def capOf (bound chunk : Nat) : Option Nat := (nextMultipleOf bound chunk).map (max 64)

/-- `IdSpace::bound`: `live + recycled.len()`, a wrapping `u64` add in release. -/
def boundOf (live deleted : Nat) : Nat := (live + deleted) % W

/-- What `restore` builds from its arguments, besides the matrices and stores it is handed. -/
structure Restored where
  nodeCap : Nat
  relCap : Nat
  schemaVersion : Nat
  version : Nat
  indexersEmpty : Bool
  constraintsEmpty : Bool
deriving DecidableEq, Repr

/-- The scalar fields of `Self { .. }` (`graph.rs`): `bound`s from `IdSpace::restored(count,
deleted).bound()` (`count`, `deleted.len()` per space), `chunk` is `NODE_CREATION_BUFFER`. `none`
is the `next_multiple_of(0)` panic. -/
def restoredFields (nodeCount nodeDel relCount relDel chunk labels types : Nat) : Option Restored := do
  let nc ← capOf (boundOf nodeCount nodeDel) chunk
  let rc ← capOf (boundOf relCount relDel) chunk
  pure { nodeCap := nc, relCap := rc, schemaVersion := labels + types, version := 0,
         indexersEmpty := true, constraintsEmpty := true }

/-- Without wrap-around (`bound + chunk <= 2^64`, any real graph) the result is the smallest
multiple of `c` at or above `n`. -/
theorem nextMultipleOf_spec (n c : Nat) (hc : 0 < c) (hw : n + c ≤ W) :
    ∃ m, nextMultipleOf n c = some m ∧ n ≤ m ∧ m < n + c ∧ m % c = 0 := by
  have hr := Nat.mod_lt n hc
  by_cases h0 : n % c = 0
  · exact ⟨n, by simp [nextMultipleOf, Nat.pos_iff_ne_zero.1 hc, h0], Nat.le_refl _, by omega, h0⟩
  · have hdiv := Nat.div_add_mod n c
    have hm : (n + (c - n % c)) % W = n + (c - n % c) := Nat.mod_eq_of_lt (by omega)
    refine ⟨n + (c - n % c), by simp [nextMultipleOf, Nat.pos_iff_ne_zero.1 hc, h0, hm], by omega, by omega, ?_⟩
    have : n + (c - n % c) = c * (n / c + 1) := by
      rw [Nat.mul_add, Nat.mul_one]; omega
    rw [this]; exact Nat.mul_mod_right _ _

/-- **The capacities**, absent wrap-around, are at least the id-space bound and at least 64, and a
zero `chunk` is the `next_multiple_of` panic. -/
theorem capOf_spec (bound chunk : Nat) :
    (chunk = 0 → capOf bound chunk = none) ∧
    (0 < chunk → bound + chunk ≤ W → ∃ r, capOf bound chunk = some r ∧ bound ≤ r ∧ 64 ≤ r ∧
      (r = 64 ∨ r % chunk = 0)) := by
  refine ⟨fun h => by simp [capOf, nextMultipleOf, h], fun hc hw => ?_⟩
  obtain ⟨m, hm, h1, _, h3⟩ := nextMultipleOf_spec bound chunk hc hw
  refine ⟨max 64 m, by simp [capOf, hm], by omega, by omega, ?_⟩
  by_cases h : 64 ≤ m
  · right; rw [Nat.max_eq_right h]; exact h3
  · left; omega

/-- **Overflow is modelled, not excluded**: a bound within one chunk of `2^64` makes the release
build's `next_multiple_of` wrap, and the capacity comes out *below* the bound (here 64 for
`bound = 2^64 - 1`, `chunk = 1024`; a debug build panics). `restoreCheck` still lets such a bound
through (`id_bound` only refuses an overflowing sum); the loaders' `check_header` caps it at `2^34`
first, so no payload reaches it. -/
theorem capOf_wraps : capOf (W - 1) 1024 = some 64 := by decide

/-- The other scalar fields: `schema_version` is the number of labels plus relationship types, the
version is 0, no index or constraint is carried over (the RDB carries none). The capacity bounds
hold when neither id-space bound is within a chunk of `2^64` (`capOf_wraps` is the other case). -/
theorem restoredFields_spec (nc nd rc rd chunk labels types : Nat) (hc : 0 < chunk)
    (hn : boundOf nc nd + chunk ≤ W) (hr : boundOf rc rd + chunk ≤ W) :
    ∃ r, restoredFields nc nd rc rd chunk labels types = some r ∧
      r.schemaVersion = labels + types ∧
      r.version = 0 ∧ r.indexersEmpty = true ∧ r.constraintsEmpty = true ∧
      boundOf nc nd ≤ r.nodeCap ∧ boundOf rc rd ≤ r.relCap ∧ 64 ≤ r.nodeCap ∧ 64 ≤ r.relCap := by
  obtain ⟨a, ha, h1, h2, _⟩ := (capOf_spec (boundOf nc nd) chunk).2 hc hn
  obtain ⟨b, hb, h3, h4, _⟩ := (capOf_spec (boundOf rc rd) chunk).2 hc hr
  exact ⟨⟨a, b, labels + types, 0, true, true⟩, by simp [restoredFields, ha, hb], rfl, rfl, rfl, rfl, h1, h3, h2, h4⟩

theorem restoredFields_zero_chunk (nc nd rc rd labels types : Nat) :
    restoredFields nc nd rc rd 0 labels types = none := by
  simp [restoredFields, capOf, nextMultipleOf]

/-! ### The checks `restore` makes before building anything (#3643)

`Graph::restore` returns `Err` (`none` here) unless the payload's counts account for its ids. -/

/-- `Graph::id_bound(kind, live, deleted)`: `live.checked_add(deleted.len())`, and the largest
deleted id below it. `deleted` is the `RoaringTreemap`, listed without repeats. -/
def idBound (live : Nat) (deleted : List Nat) : Option Nat :=
  if live + deleted.length < W ∧ ∀ d ∈ deleted, d < live + deleted.length then
    some (live + deleted.length) else none

/-- `max_endpoint`: the largest `src.max(dst)` over the edges (0 when there are none). -/
def maxEndpoint (edges : List (Nat × Nat × Nat)) : Nat :=
  edges.foldl (fun m e => max m (max e.2.1 e.2.2)) 0

/-- The checks of `restore`: both id bounds, then the live-edge count equal to
`relationship_count`, then (with an edge) `max_edge_id < edge_bound` and
`max_endpoint < node_bound`. `some (node_bound, edge_bound)` is the `Ok` path, which goes on to
`restoreLen`/`restoreIndex`/`restoredFields` above. -/
def restoreCheck (nc : Nat) (nd : List Nat) (rc : Nat) (rd : List Nat)
    (edges : List (Nat × Nat × Nat)) : Option (Nat × Nat) :=
  match idBound nc nd, idBound rc rd with
  | some nb, some eb =>
    if edges.length ≠ rc then none
    else if edges ≠ [] ∧ ((scanAll edges).maxId ≥ eb ∨ maxEndpoint edges ≥ nb) then none
    else some (nb, eb)
  | _, _ => none

theorem maxEndpoint_ge_aux : ∀ (edges : List (Nat × Nat × Nat)) (m : Nat),
    m ≤ edges.foldl (fun m e => max m (max e.2.1 e.2.2)) m ∧
    ∀ e ∈ edges, max e.2.1 e.2.2 ≤ edges.foldl (fun m e => max m (max e.2.1 e.2.2)) m
  | [], m => ⟨Nat.le_refl _, by simp⟩
  | x :: r, m => by
    have ih := maxEndpoint_ge_aux r (max m (max x.2.1 x.2.2))
    refine ⟨by simp only [List.foldl_cons]; omega, fun e he => ?_⟩
    rcases List.mem_cons.1 he with rfl | he
    · simp only [List.foldl_cons]; omega
    · simpa using ih.2 e he

theorem maxEndpoint_lt_aux : ∀ (edges : List (Nat × Nat × Nat)) (m b : Nat), m < b →
    (∀ e ∈ edges, e.2.1 < b ∧ e.2.2 < b) →
    edges.foldl (fun m e => max m (max e.2.1 e.2.2)) m < b
  | [], _, _, hm, _ => hm
  | x :: r, m, b, hm, h => by
    have hx := h x (by simp)
    simp only [List.foldl_cons]
    exact maxEndpoint_lt_aux r _ b (by omega) (fun e he => h e (by simp [he]))

theorem scan_maxId_lt_aux : ∀ (edges : List (Nat × Nat × Nat)) (a : Scan) (b : Nat), a.maxId < b →
    (∀ e ∈ edges, e.1 < b) → (edges.foldl scanStep a).maxId < b
  | [], _, _, ha, _ => ha
  | x :: r, a, b, ha, h => by
    have hx := h x (by simp)
    simp only [List.foldl_cons]
    exact scan_maxId_lt_aux r _ b (by rw [scanStep_maxId]; omega) (fun e he => h e (by simp [he]))

/-- **What an accepted payload guarantees**: the bounds are the exact `live + deleted` counts
(no wrap), the payload carries exactly `relationship_count` edges, and every edge id is below the
edge bound and every endpoint below the node bound; so the index length is at most the edge
bound. Before #3643 an edge numbered `2^40` sized an index of `2^40 + 1` slots. -/
theorem restoreCheck_sound (nc rc : Nat) (nd rd : List Nat) (edges : List (Nat × Nat × Nat))
    (nb eb : Nat) (h : restoreCheck nc nd rc rd edges = some (nb, eb)) :
    nb = nc + nd.length ∧ eb = rc + rd.length ∧ nb < W ∧ eb < W ∧ edges.length = rc ∧
    (∀ e ∈ edges, e.1 < eb ∧ e.2.1 < nb ∧ e.2.2 < nb) ∧ restoreLen edges ≤ eb := by
  unfold restoreCheck at h
  cases hn : idBound nc nd with
  | none => simp [hn] at h
  | some nb' =>
  cases he : idBound rc rd with
  | none => simp [hn, he] at h
  | some eb' =>
  simp only [hn, he] at h
  unfold idBound at hn he
  split at hn <;> simp at hn
  split at he <;> simp at he
  rename_i hn1 he1
  subst hn he
  by_cases hl : edges.length = rc
  · by_cases hc : edges ≠ [] ∧ ((scanAll edges).maxId ≥ rc + rd.length ∨
        maxEndpoint edges ≥ nc + nd.length)
    · simp [hl, hc] at h
    · simp only [hl, ne_eq, not_true_eq_false, ite_false, hc, Option.some.injEq,
        Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      have hall : ∀ e ∈ edges, e.1 < rc + rd.length ∧ e.2.1 < nc + nd.length ∧
          e.2.2 < nc + nd.length := by
        intro e hmem
        have hne : edges ≠ [] := List.ne_nil_of_mem hmem
        have h1 := scan_maxId_ge edges Scan.init e hmem
        have h2 := (maxEndpoint_ge_aux edges 0).2 e hmem
        simp only [scanAll, maxEndpoint] at hc
        have : ¬ ((edges.foldl scanStep Scan.init).maxId ≥ rc + rd.length ∨
            edges.foldl (fun m e => max m (max e.2.1 e.2.2)) 0 ≥ nc + nd.length) :=
          fun hx => hc ⟨hne, hx⟩
        omega
      refine ⟨rfl, rfl, hn1.1, he1.1, hl, hall, ?_⟩
      unfold restoreLen
      cases edges with
      | nil => simp
      | cons x r =>
        have := scan_maxId_lt_aux (x :: r) Scan.init (rc + rd.length)
          (by have := hall x (by simp); simp [Scan.init]; omega) (fun e he => (hall e he).1)
        simp only [List.isEmpty_cons, Bool.false_eq_true, ite_false, scanAll]
        omega
  · simp [hl] at h

/-- **Every saved graph passes**: with the id spaces dense (deleted ids distinct and below
`live + deleted`, the bound below `2^64`), the payload's edge count, and every edge id and
endpoint inside the bounds, the checks accept with the bounds `IdSpace::bound` would give. -/
theorem restoreCheck_complete (nc rc : Nat) (nd rd : List Nat) (edges : List (Nat × Nat × Nat))
    (hnw : nc + nd.length < W) (hew : rc + rd.length < W)
    (hnd : ∀ d ∈ nd, d < nc + nd.length) (hrd : ∀ d ∈ rd, d < rc + rd.length)
    (hl : edges.length = rc)
    (he : ∀ e ∈ edges, e.1 < rc + rd.length ∧ e.2.1 < nc + nd.length ∧ e.2.2 < nc + nd.length) :
    restoreCheck nc nd rc rd edges = some (nc + nd.length, rc + rd.length) := by
  have hn : idBound nc nd = some (nc + nd.length) := by
    unfold idBound; split
    · rfl
    · rename_i h; exact absurd ⟨hnw, hnd⟩ h
  have hr : idBound rc rd = some (rc + rd.length) := by
    unfold idBound; split
    · rfl
    · rename_i h; exact absurd ⟨hew, hrd⟩ h
  have hc : ¬ (edges ≠ [] ∧ ((scanAll edges).maxId ≥ rc + rd.length ∨
      maxEndpoint edges ≥ nc + nd.length)) := by
    cases edges with
    | nil => simp
    | cons x r =>
      have hx := he x (by simp)
      have h1 := scan_maxId_lt_aux (x :: r) Scan.init _
        (by simp [Scan.init]; omega) (fun e h => (he e h).1)
      have h2 := maxEndpoint_lt_aux (x :: r) 0 (nc + nd.length) (by omega)
        (fun e h => ⟨(he e h).2.1, (he e h).2.2⟩)
      simp only [scanAll, maxEndpoint]
      omega
  simp only [restoreCheck, hn, hr, hl, ne_eq, not_true_eq_false, ite_false]
  split
  · rename_i h; exact absurd (by simpa using h) hc
  · rfl

/-- An edge numbered past the edge bound (the `2^40` of `restore_refuses_ids_the_counts_do_not_
account_for`) is refused, as is a deleted id outside the id space and a count that overflows. -/
theorem restoreCheck_refuses_examples :
    restoreCheck 2 [] 1 [] [(2 ^ 40, 0, 1)] = none ∧
    restoreCheck 2 [] 1 [] [(0, 0, 9)] = none ∧
    restoreCheck 2 [] 2 [] [(0, 0, 1)] = none ∧
    restoreCheck 2 [99] 0 [] [] = none ∧
    restoreCheck (W - 1) [0] 0 [] [] = none := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> decide


/-! ## `Graph::delete_implicit_edges` -/

/-- One implicitly deleted edge (`DeletedEdge`): id, endpoints, relationship type. -/
structure DE where
  id : Nat
  src : Nat
  dst : Nat
  typ : Nat
deriving DecidableEq, Repr

/-- `all_implicit` (`graph.rs`): per relationship type in order, the edges `collect` finds, as
`DeletedEdge`s. `N` = `deleted_nodes`, `X` = `explicit_rels`, `T` = the tensors, edges
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

/-- The endpoint index after the clear loop (`clear_edge_endpoint` per collected id, `graph.rs`). -/
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
#print axioms restoreCheck_sound
#print axioms restoreCheck_complete
#print axioms deletedEdges_mem
#print axioms clearedAdj_mem
#print axioms clearImplicit_get

end Falkor.EndpointIndex
