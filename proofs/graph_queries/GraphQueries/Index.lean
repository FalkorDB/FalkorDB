import GraphQueries.Labels
/-
# Index glue, memory report, payload dispatch, plan cache

The `Indexer` (RediSearch-backed, proofs/index_layer) is a parameter:
`IX` records the answers the graph code consumes. Every theorem below holds
for *any* indexer behaviour.

| here | there (graph.rs) |
| --- | --- |
| `populateStart`   | `populate_index` :494 |
| `dropBg`          | `drop_index_bg` :741 |
| `createIndex`     | `create_index` :3471 / `create_index_sync` :3509 |
| `syncDocs`        | `populate_indexes_sync` :3549 (one label/type) |
| `commitDocs`      | `commit_index_kind` :3761 (via `commit_index` :3650, `commit_index_locked` :3663) |
| `commitEdgeDocs`  | `commit_edge_index_locked` :3685 (via `commit_edge_index` :3673) |
| `dropIndex`       | `drop_index` :3805 |
| `indexInfo`       | `index_info` :4083 |
| `memEstimate`, `labelPass` | `memory_usage_report` :4534 sampling arithmetic and seen-dedup |
| `encodeOp`        | `encode_payload` :4708 dispatch |
| `getPlan`         | `get_plan` :1251 cache decision |
-/
namespace GQ
variable {V : Type}

/-- `populate_index`: no snapshot → nothing; else batch 0 at the default cursor. -/
def populateStart (snap : Option Nat) : Option (Nat × Nat × Option Nat × Option Nat) :=
  snap.map (fun _ => (0, 0, none, none))
theorem populateStart_spec (t : Nat) : populateStart (some t) = some (0, 0, none, none) ∧
    populateStart none = none := ⟨rfl, rfl⟩

/-- `drop_index_bg`: the label leaves the indexer's map (under its lock). -/
def dropBg (idxLabels : List String) (l : String) : List String := idxLabels.filter (· != l)
theorem dropBg_spec (idxLabels : List String) (l : String) : l ∉ dropBg idxLabels l := by
  simp [dropBg]

/-- `create_index` / `create_index_sync`: register the label (or type),
ask the indexer (`res`), then on success register the attribute names.
The label is registered **before** the indexer can refuse. -/
def createIndex (chunk : Nat) (hc : 0 < chunk) (g : G V) (isNode : Bool) (label : String)
    (attrs : List String) (res : Bool) : G V × Bool :=
  let g1 := if isNode then (getLabelMatMut chunk hc g label).1 else (getRelMatMut chunk hc g label).1
  if res then ({ g1 with attrs := attrs.foldl nmInsert g1.attrs }, true) else (g1, false)

theorem foldl_nmInsert_mem (nm attrs : List String) (s : String) (h : s ∈ nm) :
    s ∈ attrs.foldl nmInsert nm := by
  induction attrs generalizing nm with
  | nil => exact h
  | cons a l ih =>
    apply ih; unfold nmInsert; split
    · exact h
    · split
      · exact h
      · simp [h]

theorem foldl_nmInsert_has (nm attrs : List String) (s : String) (hs : s ∈ attrs)
    (hcap : nm.length + attrs.length ≤ MAXA) : s ∈ attrs.foldl nmInsert nm := by
  induction attrs generalizing nm with
  | nil => cases hs
  | cons a l ih =>
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 hs with rfl | hl
    · apply foldl_nmInsert_mem; unfold nmInsert; split
      · assumption
      · split
        · simp at hcap; omega
        · simp
    · apply ih _ hl; unfold nmInsert; split
      · simp at hcap ⊢; omega
      · split
        · simp at hcap ⊢; omega
        · simp at hcap ⊢; omega

theorem resize_attrs (chunk : Nat) (hc : 0 < chunk) (g : G V) : (resize chunk hc g).attrs = g.attrs := by
  simp only [resize]; (repeat' split) <;> rfl
theorem internLabel_attrs (g : G V) (s : String) : (internLabel g s).1.attrs = g.attrs := by
  unfold internLabel; split <;> rfl
theorem getLabelMatMut_attrs (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) :
    (getLabelMatMut chunk hc g s).1.attrs = g.attrs := by
  have h1 := internLabel_attrs g s
  unfold getLabelMatMut
  generalize internLabel g s = p at h1 ⊢
  obtain ⟨g1, id⟩ := p
  simp only at h1 ⊢
  split
  · rw [resize_attrs]; exact h1
  · exact h1

/-- The label is registered whether or not the indexer accepts the index
(Rust replies `Labels added: 1` to `CREATE INDEX` on a new label; C does not). -/
theorem createIndex_node_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (label : String)
    (attrs : List String) (res : Bool) (h : LInv g)
    (hcap : g.attrs.length + attrs.length ≤ MAXA) :
    let r := createIndex chunk hc g true label attrs res
    (getLabelId r.1 label).isSome ∧ r.2 = res ∧ (res = true → ∀ a ∈ attrs, a ∈ r.1.attrs) := by
  obtain ⟨-, hl2⟩ := getLabelMatMut_spec chunk hc g label h
  have hk := getLabelMatMut_attrs chunk hc g label
  cases res with
  | false =>
    refine ⟨?_, rfl, fun h => by cases h⟩
    show (getLabelId (getLabelMatMut chunk hc g label).1 label).isSome = true
    rw [hl2]; rfl
  | true =>
    refine ⟨?_, rfl, fun _ a ha => ?_⟩
    · show (getLabelId (getLabelMatMut chunk hc g label).1 label).isSome = true
      rw [hl2]; rfl
    · show a ∈ attrs.foldl nmInsert (getLabelMatMut chunk hc g label).1.attrs
      apply foldl_nmInsert_has _ _ _ ha; rw [hk]; exact hcap

/-- `populate_indexes_sync`: a document per entity carrying ≥ 1 indexed
attribute, holding exactly the present fields. -/
def syncDocs (ents : List Nat) (attrs : List Nat) (present : Nat → Nat → Bool) : List (Nat × List Nat) :=
  ents.filterMap (fun n => let fs := attrs.filter (present n); if fs = [] then none else some (n, fs))

theorem syncDocs_mem (ents attrs : List Nat) (present : Nat → Nat → Bool) (n : Nat) (hn : n ∈ ents)
    (hnd : ents.Nodup) :
    (∃ fs, (n, fs) ∈ syncDocs ents attrs present) ↔ ∃ a ∈ attrs, present n a = true := by
  simp only [syncDocs, List.mem_filterMap]
  constructor
  · rintro ⟨fs, m, -, hm⟩
    split at hm
    · cases hm
    · rename_i hne; cases hm
      obtain ⟨a, ha⟩ := List.exists_mem_of_ne_nil _ hne
      exact ⟨a, (List.mem_filter.1 ha).1, (List.mem_filter.1 ha).2⟩
  · rintro ⟨a, ha, hp⟩
    have hne : attrs.filter (present n) ≠ [] := List.ne_nil_of_mem (List.mem_filter.2 ⟨ha, hp⟩)
    exact ⟨attrs.filter (present n), n, hn, by simp [hne]⟩

/-- `commit_index_kind`: one document per queued id (fields filled when
present), removals passed through; nothing queued → no indexer call. -/
def commitDocs (adds : List (Nat × List Nat)) (rems : List (Nat × List Nat)) (names : List String)
    (fields : String → List Nat) (val : Nat → Nat → Option Nat) :
    Option (List (String × List (Nat × List (Nat × Nat))) × List (String × List Nat)) :=
  if adds = [] ∧ rems = [] then none else
  some (adds.map (fun (slot, ids) => let nm := (names[slot]?).getD ""
          (nm, ids.map (fun id => (id, (fields nm).filterMap (fun k => (val id k).map (k, ·)))))),
        rems.map (fun (slot, ids) => ((names[slot]?).getD "", ids)))

theorem commitDocs_noop (names : List String) (fields : String → List Nat) (val : Nat → Nat → Option Nat) :
    commitDocs [] [] names fields val = none := rfl

theorem commitDocs_every_id (adds rems : List (Nat × List Nat)) (names : List String)
    (fields : String → List Nat) (val : Nat → Nat → Option Nat) (r : _)
    (h : commitDocs adds rems names fields val = some r) (slot id : Nat) (ids : List Nat)
    (hs : (slot, ids) ∈ adds) (hi : id ∈ ids) :
    ∃ nm docs, (nm, docs) ∈ r.1 ∧ ∃ fs, (id, fs) ∈ docs := by
  unfold commitDocs at h
  split at h
  · cases h
  · cases h
    exact ⟨_, _, List.mem_map.2 ⟨_, hs, rfl⟩, _, List.mem_map.2 ⟨id, hi, rfl⟩⟩

/-- `commit_edge_index_locked`: an edge's document key is the `(src, dst, id)`
found by scanning the type's tensor; ids no longer in the tensor are skipped. -/
def commitEdgeDocs (tn : Ten) (ids : List Nat) : List (Nat × Nat × Nat) :=
  ids.filterMap (fun id => (tn.find? (·.2.2 == id)).map (fun x => (x.1, x.2.1, id)))

theorem commitEdgeDocs_mem (tn : Ten) (ids : List Nat) (s d e : Nat) (hnd : (tn.map (·.2.2)).Nodup) :
    (s, d, e) ∈ commitEdgeDocs tn ids ↔ e ∈ ids ∧ (s, d, e) ∈ tn := by
  simp only [commitEdgeDocs, List.mem_filterMap, Option.map_eq_some_iff]
  constructor
  · rintro ⟨id, hid, x, hx, he⟩
    obtain ⟨xs, xd, xe⟩ := x
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    have := List.find?_some hx; simp at this; subst this
    exact ⟨hid, List.mem_of_find?_eq_some hx⟩
  · rintro ⟨hid, hm⟩
    refine ⟨e, hid, (s, d, e), ?_, rfl⟩
    induction tn with
    | nil => cases hm
    | cons y tn ih =>
      simp only [List.map_cons, List.nodup_cons] at hnd
      rcases List.mem_cons.1 hm with rfl | hm'
      · simp
      · have : y.2.2 ≠ e := fun h => hnd.1 (h ▸ List.mem_map_of_mem hm')
        simp [this, ih hnd.2 hm']

/-- `drop_index`: empty attrs expand to the index type's fields; a UNIQUE
constraint over any of them blocks the drop; otherwise the indexer decides. -/
def dropIndex (dependsOn : String → Bool) (fieldsOfType : List String) (attrs : List String)
    (dropped : Nat) : Except String Nat :=
  let eff := if attrs = [] then fieldsOfType else attrs
  if eff.any dependsOn then .error "Index supports constraint"
  else if dropped > 0 then .ok dropped else .error "no such index"

theorem dropIndex_protects (dependsOn : String → Bool) (fieldsOfType attrs : List String) (dropped : Nat)
    (a : String) (ha : a ∈ (if attrs = [] then fieldsOfType else attrs)) (hd : dependsOn a = true) :
    dropIndex dependsOn fieldsOfType attrs dropped = .error "Index supports constraint" := by
  unfold dropIndex
  rw [if_pos (List.any_eq_true.2 ⟨a, ha, hd⟩)]

/-- `index_info`: node infos tagged `NODE`, then edge infos tagged `RELATIONSHIP`. -/
def indexInfo {I} (nodeInfos edgeInfos : List I) : List (String × I) :=
  nodeInfos.map ("NODE", ·) ++ edgeInfos.map ("RELATIONSHIP", ·)
theorem indexInfo_length {I} (a b : List I) : (indexInfo a b).length = a.length + b.length := by
  simp [indexInfo]

/-! ## `memory_usage_report` -/

/-- `(sampled_mem * count / sampled_count)`, 0 when nothing was sampled. -/
def memEstimate (sampled cnt sc : Nat) : Nat := if sc > 0 then sampled * cnt / sc else 0

theorem memEstimate_exact (sampled cnt : Nat) (h : cnt > 0) : memEstimate sampled cnt cnt = sampled := by
  simp [memEstimate, h, Nat.mul_div_cancel _ h]

/-- The `seen` pass over label matrices: a node is attributed to its first
label only; returns (per-label unprocessed counts, final seen set). -/
def labelPass : List (List Nat) → List Nat → List Nat × List Nat
  | [], seen => ([], seen)
  | rows :: rest, seen =>
    let fresh := (rows.filter (· ∉ seen))
    let r := labelPass rest (seen ++ fresh)
    (fresh.length :: r.1, r.2)

/-- Every labeled node is counted under exactly one label: the counts sum to
the number of distinct labeled nodes (`total_labeled`). -/
theorem labelPass_total : ∀ (ls : List (List Nat)) (seen : List Nat),
    (∀ r ∈ ls, r.Nodup) →
    (labelPass ls seen).1.sum + seen.length = (labelPass ls seen).2.length
  | [], seen, _ => by simp [labelPass]
  | rows :: rest, seen, h => by
    simp only [labelPass, List.sum_cons]
    have := labelPass_total rest (seen ++ rows.filter (· ∉ seen)) (fun r hr => h r (by simp [hr]))
    simp only [List.length_append] at this; omega

/-- `unlabeled_count = node_count.saturating_sub(total_labeled)`. -/
def unlabeled (nodeCount total : Nat) : Nat := nodeCount - total
theorem unlabeled_le (a b : Nat) : unlabeled a b ≤ a := Nat.sub_le a b

/-! ## `encode_payload` dispatch -/

inductive EState | nodes | deletedNodes | edges | deletedEdges | labelsMatrices | relationMatrices
  | adjMatrix | lblsMatrix | other deriving DecidableEq

/-- What each payload state writes: (store, upper id bound) or the matrix it encodes. -/
def encodeOp (g : G V) : EState → String × Nat
  | .nodes => ("node_attrs", maxNodeId g)
  | .deletedNodes => ("deleted_nodes", g.delNodes.length)
  | .edges => ("relationship_attrs", maxRelId g)
  | .deletedEdges => ("deleted_relationships", g.delRels.length)
  | .labelsMatrices => ("label_matrices", g.labelMs.length)
  | .relationMatrices => ("relation_matrices", g.relMs.length)
  | .adjMatrix => ("adjacency", 0)
  | .lblsMatrix => ("node_labels", 0)
  | .other => ("", 0)

theorem encodeOp_nodes (g : G V) (h : g.nodeCount ≠ 0) : (encodeOp g .nodes).2 + 1 = nodeBound g :=
  maxNodeId_lt g h

/-! ## `get_plan` cache decision -/

/-- Hit iff a cached plan exists for the stripped query text under the same
UDF version; a miss re-plans and caches only if the UDF version did not move
while planning. Returns (cached?, new cache). -/
def getPlan (cache : List (String × Nat)) (q : String) (udf udfAfter : Nat) : Bool × List (String × Nat) :=
  if (q, udf) ∈ cache then (true, cache)
  else (false, if udfAfter = udf then (q, udf) :: cache.filter (·.1 != q) else cache)

theorem getPlan_hit (cache : List (String × Nat)) (q : String) (u u' : Nat) (h : (q, u) ∈ cache) :
    getPlan cache q u u' = (true, cache) := by simp [getPlan, h]

theorem getPlan_stale (cache : List (String × Nat)) (q : String) (u u' : Nat) (h : (q, u) ∉ cache) :
    (getPlan cache q u u').1 = false := by simp [getPlan, h]

theorem getPlan_then_hit (cache : List (String × Nat)) (q : String) (u : Nat) :
    (getPlan (getPlan cache q u u).2 q u u).1 = true := by
  unfold getPlan; by_cases h : (q, u) ∈ cache <;> simp [h]

/-- `is_synced`: conjunction over every matrix and tensor. -/
def isSynced (flags : List Bool) : Bool := flags.all id
theorem isSynced_iff (flags : List Bool) : isSynced flags = true ↔ ∀ b ∈ flags, b = true := by
  simp [isSynced]

end GQ
