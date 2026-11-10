import FalkorNativeIndex.Ddl
set_option linter.unusedSimpArgs false

/-!
# The scan that stands in for a building column, and who feeds RediSearch (#3547, #2386)

* `Graph::node_matches` / `relationship_matches` (`graph.rs:4230`, `:4244`) and the filtered scans
  the two scan ops run on `Some(Err(scan_filter))` (`node_by_index_scan.rs:~330`,
  `edge_by_index_scan.rs:~370`): the scan keeps **exactly what the built column would answer**
  (`nodeFilteredScan_agrees`, `edgeFilteredScan_agrees`). The planner removes a predicate it hands
  to an index scan, so this filter is the only check the scanned rows get.
* `redisearch_fields` (`graph.rs:144`, `:151`): the identity without `index-falkordb`; with it, the
  fields that still go to RediSearch, i.e. all but the Range fields (`redisearchFields_mem` and its
  corollaries).
-/

namespace NI

/-! ## `node_matches`, `relationship_matches` -/

/-- `Graph::node_matches` (`graph.rs:4230-4240`): the committed value of attribute `attr` of node `id`
    matches the filter (`get_node_attribute_by_idx(id, attr).is_some_and(|v| scan_filter.matches(&v))`). -/
def G.nodeMatches (g : G) (id attr : Nat) (f : ScanFilter) : Bool :=
  (g.nodeAttr id attr).any f.matches

/-- `Graph::relationship_matches` (`graph.rs:4244-4254`): the same for an edge. -/
def G.relationshipMatches (g : G) (id attr : Nat) (f : ScanFilter) : Bool :=
  (g.relAttr id attr).any f.matches

theorem nodeMatches_spec (g : G) (id attr : Nat) (f : ScanFilter) :
    g.nodeMatches id attr f = true ↔ ∃ v, g.nodeAttr id attr = some v ∧ f.matches v = true := by
  unfold G.nodeMatches
  cases h : g.nodeAttr id attr <;> simp

theorem relationshipMatches_spec (g : G) (id attr : Nat) (f : ScanFilter) :
    g.relationshipMatches id attr f = true ↔ ∃ v, g.relAttr id attr = some v ∧ f.matches v = true := by
  unfold G.relationshipMatches
  cases h : g.relAttr id attr <;> simp

/-- A node without the attribute does not match (`is_some_and`): it holds no value, so the column
    never indexed it. -/
theorem nodeMatches_none (g : G) (id attr : Nat) (f : ScanFilter) (h : g.nodeAttr id attr = none) :
    g.nodeMatches id attr f = false := by simp [G.nodeMatches, h]

/-! ## The filtered scans -/

/-- The scan `NodeByIndexScanOp::next` runs for `Some(Err(scan_filter))`: the attribute is resolved once
    (`get_node_attribute_id`; a name the graph never saw is held by no node, so the scan is empty), then
    the label's nodes are filtered by `node_matches`. (The Rust scans the *pattern's* labels, a subset
    of the index label's nodes; the op's `extra_labels` filter then removes the same nodes from the
    index route, so the rows agree: `ops_traverse`.) -/
def nodeFilteredScan (g : G) (label : String) (f : ScanFilter) : List Nat :=
  match g.attrId f.attr with
  | none => []
  | some attr => ((g.labelNodes label).getD []).filter (fun id => g.nodeMatches id attr f)

/-- The same for `EdgeByIndexScanOp::next`: `get_all_edges(type)` filtered by `relationship_matches`. -/
def edgeFilteredScan (g : G) (typeName : String) (f : ScanFilter) : List (Nat × Nat × Nat) :=
  match g.attrId f.attr with
  | none => []
  | some attr => ((g.typeEdges typeName).getD []).filter (fun e => g.relationshipMatches e.2.2 attr f)

/-- **The scan keeps exactly what the built column answers.** For a `ScanFilter` whose attribute is
    `f.attr`, a node is scanned iff the column `CREATE INDEX` would have built from the live graph
    (`fromEntries (collectNodeEntries ..)`) returns it for the filter's keys. -/
theorem nodeFilteredScan_agrees (g : G) (label : String) (f : ScanFilter)
    (hlt : ∀ lm, g.labelNodes label = some lm → ∀ n ∈ lm, n < W) (id : Nat) (hid : id < W) :
    id ∈ nodeFilteredScan g label f ↔
      id ∈ (NumericIndex.fromEntries (collectNodeEntries g label f.attr)).lookup f.keys := by
  have hids : ∀ p ∈ collectNodeEntries g label f.attr, p.2 < W := by
    rintro ⟨v, n⟩ hp
    obtain ⟨lm, idx, hl, _, hn, _⟩ := (collectNodeEntries_spec g label f.attr v n).1 hp
    exact hlt lm hl n hn
  rw [lookup_fromEntries _ hids f.keys id hid]
  unfold nodeFilteredScan
  cases hat : g.attrId f.attr with
  | none =>
    simp only [List.not_mem_nil, false_iff, not_exists, not_and]
    intro v hv
    obtain ⟨lm, idx, _, h2, _⟩ := (collectNodeEntries_spec g label f.attr v id).1 hv
    rw [hat] at h2; cases h2
  | some attr =>
    simp only [List.mem_filter, nodeMatches_spec]
    constructor
    · rintro ⟨hm, v, hv, hf⟩
      cases hl : g.labelNodes label with
      | none => simp [hl] at hm
      | some lm =>
        simp only [hl, Option.getD_some] at hm
        exact ⟨v, (collectNodeEntries_spec g label f.attr v id).2 ⟨lm, attr, hl, hat, hm, hv⟩, hf⟩
    · rintro ⟨v, hv, hf⟩
      obtain ⟨lm, idx, hl, h2, hm, hv'⟩ := (collectNodeEntries_spec g label f.attr v id).1 hv
      rw [hat] at h2; cases h2
      simp only [hl, Option.getD_some]
      exact ⟨hm, v, hv', hf⟩

/-- The edge scan: an edge id is scanned iff the built column returns it. -/
theorem edgeFilteredScan_agrees (g : G) (typeName : String) (f : ScanFilter)
    (hlt : ∀ t, g.typeEdges typeName = some t → ∀ e ∈ t, e.2.2 < W) (id : Nat) (hid : id < W) :
    (∃ s d, (s, d, id) ∈ edgeFilteredScan g typeName f) ↔
      id ∈ (NumericIndex.fromEntries (collectEdgeEntries g typeName f.attr)).lookup f.keys := by
  have hids : ∀ p ∈ collectEdgeEntries g typeName f.attr, p.2 < W := by
    rintro ⟨v, n⟩ hp
    obtain ⟨idx, _, hn, _⟩ := (collectEdgeEntries_spec g typeName f.attr v n).1 hp
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hn
    cases ht : g.typeEdges typeName with
    | none => simp [ht] at he
    | some t => simp only [ht, Option.getD_some] at he; exact hlt t ht e he
  rw [lookup_fromEntries _ hids f.keys id hid]
  unfold edgeFilteredScan
  cases hat : g.attrId f.attr with
  | none =>
    simp only [List.not_mem_nil, exists_false, false_iff, not_exists, not_and]
    intro v hv
    obtain ⟨idx, h2, _⟩ := (collectEdgeEntries_spec g typeName f.attr v id).1 hv
    rw [hat] at h2; cases h2
  | some attr =>
    simp only [List.mem_filter, relationshipMatches_spec]
    constructor
    · rintro ⟨s, d, hm, v, hv, hf⟩
      exact ⟨v, (collectEdgeEntries_spec g typeName f.attr v id).2
        ⟨attr, hat, List.mem_map.2 ⟨(s, d, id), hm, rfl⟩, hv⟩, hf⟩
    · rintro ⟨v, hv, hf⟩
      obtain ⟨idx, h2, hn, hv'⟩ := (collectEdgeEntries_spec g typeName f.attr v id).1 hv
      rw [hat] at h2; cases h2
      obtain ⟨⟨s, d, e⟩, he, hee⟩ := List.mem_map.1 hn
      simp only at hee; subst hee
      exact ⟨s, d, he, v, hv', hf⟩

/-- **A building column's read, end to end (nodes)**: if `query_numeric` answers a query with a
    `ScanFilter` (the column is still building), the scan the op runs returns exactly the ids the
    column `CREATE INDEX` would build answers the same query with once it is built. -/
theorem nodeScan_building_agrees (g : G) (label : String) (q : IndexQuery) (f : ScanFilter)
    (hq : queryIndexNumericNodes g label q = some (.error f))
    (hlt : ∀ lm, g.labelNodes label = some lm → ∀ n ∈ lm, n < W) (id : Nat) (hid : id < W) :
    ∃ r, (NumericIndex.fromEntries (collectNodeEntries g label f.attr)).query q = some r ∧
      (id ∈ r ↔ id ∈ nodeFilteredScan g label f) := by
  rw [queryIndexNumericNodes_spec] at hq
  obtain ⟨_, _, _, _, _, _, hks⟩ := queryNumeric_error _ _ _ _ _ hq
  refine ⟨_, ?_, (nodeFilteredScan_agrees g label f hlt id hid).symm⟩
  simp only [NumericIndex.query, hks, Option.map_some]

/-- The same for edges. -/
theorem edgeScan_building_agrees (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery)
    (f : ScanFilter) (hq : queryIndexNumericEdges g ep t q = some (.error f))
    (hlt : ∀ tt, g.typeEdges t = some tt → ∀ e ∈ tt, e.2.2 < W) (id : Nat) (hid : id < W) :
    ∃ r, (NumericIndex.fromEntries (collectEdgeEntries g t f.attr)).query q = some r ∧
      (id ∈ r ↔ ∃ s d, (s, d, id) ∈ edgeFilteredScan g t f) := by
  rw [queryIndexNumericEdges_scan] at hq
  obtain ⟨_, _, _, _, _, _, hks⟩ := queryNumeric_error _ _ _ _ _ hq
  refine ⟨_, ?_, (edgeFilteredScan_agrees g t f hlt id hid).symm⟩
  simp only [NumericIndex.query, hks, Option.map_some]

/-! ## The scan ops under `index-falkordb` -/

/-- `NodeByIndexScanOp::next`'s source of nodes for one row (`node_by_index_scan.rs:302-370`):
    `can_utilize_index` false → the label scan; else rows, the filtered scan, or the error (`none`). -/
def nodeScanOp (g : G) (utilize : Bool) (label : String) (q : IndexQuery) : Option (List Nat) :=
  scanOp utilize (queryIndexNumericNodes g label q) ((g.labelNodes label).getD []) (nodeFilteredScan g label)

/-- `EdgeByIndexScanOp::next`'s source of edges (`edge_by_index_scan.rs:318-400`). -/
def edgeScanOp (g : G) (ep : Nat → Option (Nat × Nat)) (utilize : Bool) (t : String) (q : IndexQuery) :
    Option (List (Nat × Nat × Nat)) :=
  scanOp utilize (queryIndexNumericEdges g ep t q) ((g.typeEdges t).getD []) (edgeFilteredScan g t)

/-- A `Range` on an existing column is never an error and never read from RediSearch (there is none):
    the node op returns rows or the filtered scan whatever the column's build state. -/
theorem nodeScanOp_range_served (g : G) (label key : String) (mn mx : Option Value) (a b : Bool)
    (en : IndexEntry) (hen : g.idx.entry .node label key = some en)
    (hmn : ∀ v, mn = some v → v = .null ∨ (encodeNumeric v).isSome = true)
    (hmx : ∀ v, mx = some v → v = .null ∨ (encodeNumeric v).isSome = true) :
    (nodeScanOp g true label (.range key mn mx a b)).isSome = true := by
  unfold nodeScanOp
  have h := queryNumeric_range_served g.idx .node label key mn mx a b en hen hmn hmx
  rw [← queryIndexNumericNodes_spec] at h
  cases hq : queryIndexNumericNodes g label (.range key mn mx a b) with
  | none => simp [hq] at h
  | some a' => cases a' <;> simp [scanOp]

theorem edgeScanOp_range_served (g : G) (ep : Nat → Option (Nat × Nat)) (t key : String) (mn mx : Option Value)
    (a b : Bool) (en : IndexEntry) (hen : g.idx.entry .relationship t key = some en)
    (hmn : ∀ v, mn = some v → v = .null ∨ (encodeNumeric v).isSome = true)
    (hmx : ∀ v, mx = some v → v = .null ∨ (encodeNumeric v).isSome = true) :
    (edgeScanOp g ep true t (.range key mn mx a b)).isSome = true := by
  unfold edgeScanOp
  have h := queryNumeric_range_served g.idx .relationship t key mn mx a b en hen hmn hmx
  have hn : queryIndexNumericEdges g ep t (.range key mn mx a b) ≠ none := by
    rw [Ne, queryIndexNumericEdges_none]
    intro h0; rw [h0] at h; simp at h
  cases hq : queryIndexNumericEdges g ep t (.range key mn mx a b) with
  | none => exact absurd hq hn
  | some a' => cases a' <;> simp [scanOp]

/-- The op errors (`unsupported_by_native_index`) exactly when the native index declines the predicate
    (no column, or a shape the column does not serve): never because a column is building. -/
theorem nodeScanOp_error_iff (g : G) (label : String) (q : IndexQuery) :
    nodeScanOp g true label q = none ↔ g.idx.queryNumeric .node label q = none := by
  unfold nodeScanOp
  rw [scanOp_none_iff, queryIndexNumericNodes_spec]

theorem edgeScanOp_error_iff (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery) :
    edgeScanOp g ep true t q = none ↔ g.idx.queryNumeric .relationship t q = none := by
  unfold edgeScanOp
  rw [scanOp_none_iff, queryIndexNumericEdges_none]

/-! ## `redisearch_fields` (`graph.rs:144`, `:151`) -/

/-- `redisearch_fields` without `index-falkordb` (`graph.rs:144-148`): the identity, RediSearch owns every
    kind. -/
def redisearchFieldsAll (fields : List (String × List Field)) : List (String × List Field) := fields

/-- `redisearch_fields` with `index-falkordb` (`graph.rs:151-165`): per key, drop the Range fields
    (`!matches!(f.ty, IndexType::Range)`), and drop the key when nothing is left. The `HashMap` is an
    association list with unique keys; the statements are per entry. -/
def redisearchFields (fields : List (String × List Field)) : List (String × List Field) :=
  fields.filterMap (fun kf =>
    let kept := kf.2.filter (fun f => !(f.ty == .range))
    if kept.isEmpty then none else some (kf.1, kept))

theorem redisearchFieldsAll_id (fields : List (String × List Field)) : redisearchFieldsAll fields = fields := rfl

/-- **What RediSearch is still fed**: an entry `(k, kept)` is there iff some input entry `(k, l)` has
    `kept = l` without its Range fields, and that is non-empty. -/
theorem redisearchFields_mem (fields : List (String × List Field)) (k : String) (kept : List Field) :
    (k, kept) ∈ redisearchFields fields ↔
      ∃ l, (k, l) ∈ fields ∧ kept = l.filter (fun f => !(f.ty == .range)) ∧ kept ≠ [] := by
  unfold redisearchFields
  simp only [List.mem_filterMap]
  constructor
  · rintro ⟨⟨k', l⟩, hm, h⟩
    by_cases he : (l.filter (fun f => !(f.ty == .range))).isEmpty = true
    · simp [he] at h
    · simp only [he, Bool.false_eq_true, if_false, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨l, hm, rfl, by intro h0; simp [h0] at he⟩
  · rintro ⟨l, hm, rfl, hne⟩
    refine ⟨(k, l), hm, ?_⟩
    have : ¬ (l.filter (fun f => !(f.ty == .range))).isEmpty = true := by
      simpa [List.isEmpty_iff] using hne
    simp [this]

/-- No Range field is fed to RediSearch. -/
theorem redisearchFields_no_range (fields : List (String × List Field)) :
    ∀ k l, (k, l) ∈ redisearchFields fields → ∀ f ∈ l, f.ty ≠ .range := by
  intro k l h f hf
  obtain ⟨l0, _, rfl, _⟩ := (redisearchFields_mem fields k l).1 h
  simp only [List.mem_filter, Bool.not_eq_true', beq_eq_false_iff_ne] at hf
  exact hf.2

/-- Fulltext and Vector fields are untouched: every non-Range field of every input entry is still fed,
    including on a label that also carries a Range field. -/
theorem redisearchFields_keeps_nonrange (fields : List (String × List Field)) (k : String) (l : List Field)
    (hm : (k, l) ∈ fields) (f : Field) (hf : f ∈ l) (hnr : f.ty ≠ .range) :
    ∃ kept, (k, kept) ∈ redisearchFields fields ∧ f ∈ kept := by
  refine ⟨l.filter (fun f => !(f.ty == .range)), (redisearchFields_mem fields k _).2 ⟨l, hm, rfl, ?_⟩, ?_⟩
  · intro h0
    have : f ∈ l.filter (fun f => !(f.ty == .range)) := by simp [hf, hnr]
    rw [h0] at this; simp at this
  · simp [hf, hnr]

/-- A key with no field left is dropped, never fed as an empty field list. -/
theorem redisearchFields_nonempty (fields : List (String × List Field)) :
    ∀ k l, (k, l) ∈ redisearchFields fields → l ≠ [] := by
  intro k l h; exact ((redisearchFields_mem fields k l).1 h).choose_spec.2.2

/-- A key whose every entry is only Range fields is not fed at all. -/
theorem redisearchFields_drops_range_only (fields : List (String × List Field)) (k : String)
    (h : ∀ l, (k, l) ∈ fields → ∀ f ∈ l, f.ty = .range) : ∀ kept, (k, kept) ∉ redisearchFields fields := by
  intro kept hm
  obtain ⟨l, hl, rfl, hne⟩ := (redisearchFields_mem fields k kept).1 hm
  apply hne
  rw [List.filter_eq_nil_iff]
  intro f hf
  simp [h l hl f hf]

/-- Filtering twice changes nothing. -/
theorem redisearchFields_idem (fields : List (String × List Field)) :
    redisearchFields (redisearchFields fields) = redisearchFields fields := by
  unfold redisearchFields
  induction fields with
  | nil => rfl
  | cons kf rest ih =>
    obtain ⟨k, l⟩ := kf
    simp only [List.filterMap_cons]
    by_cases he : (l.filter (fun f => !(f.ty == .range))).isEmpty = true
    · simp only [he, if_true]; exact ih
    · simp only [he, Bool.false_eq_true, if_false, List.filterMap_cons]
      have hf : ((l.filter (fun f => !(f.ty == .range))).filter (fun f => !(f.ty == .range))) =
          l.filter (fun f => !(f.ty == .range)) := by
        rw [List.filter_filter]; congr 1; funext f; simp
      simp only [hf, he, Bool.false_eq_true, if_false, ih]

/-- **Range is the native index's alone**: an attribute with a Range field is built natively by
    `populate` (from every field, `native_attrs`), and the Range fields are never fed to RediSearch (an attribute
    that also has a Fulltext or Vector field still has those fed: `redisearchFields_keeps_nonrange`). -/
theorem native_owns_range (g : G) (e : EntityType) (label : String) (a : String) (fs : List Field)
    (fields : List (String × List Field)) (hr : fs.any (fun f => f.ty == .range) = true) :
    populate g e label [(a, fs)] = buildInOneBatch g e label a ∧
      ∀ l, (a, l) ∈ redisearchFields fields → ∀ f ∈ l, f.ty ≠ .range :=
  ⟨populate_builds g e label (a, fs) hr, fun l h => redisearchFields_no_range fields a l h⟩

end NI
