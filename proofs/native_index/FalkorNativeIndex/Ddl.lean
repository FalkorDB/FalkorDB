import FalkorNativeIndex.Build
set_option linter.unusedSimpArgs false

/-!
# Index DDL (`graph_writes.rs:395-480`, `:510-552`), the graph's index entry points
# (`graph.rs:3780-3809`, `:4087-4122`), the scan routing (`node_by_index_scan.rs:306-323`,
# `edge_by_index_scan.rs:329-349`), and folded-roots MVCC (`graph.rs:441-451`, `:1062-1065`)
-/

namespace NI

/-- **Specification** (no Rust counterpart since #2351, whose `build_column`/`collect_*_entries` became
    batches, `Build.lean`): the `(value, id)` pairs of every node of the label matrix that has `attr`. -/
def collectNodeEntries (g : G) (label attr : String) : List (Value × Nat) :=
  match g.labelNodes label, g.attrId attr with
  | some lm, some idx => lm.filterMap (fun n => (g.nodeAttr n idx).map (fun v => (v, n)))
  | _, _ => []

/-- **Specification**: the `(value, edge_id)` pairs of every edge of the type that has `attr`. -/
def collectEdgeEntries (g : G) (typeName attr : String) : List (Value × Nat) :=
  match g.attrId attr with
  | none => []
  | some idx =>
    let edgeIds := ((g.typeEdges typeName).getD []).map (·.2.2)
    edgeIds.filterMap (fun eid => (g.relAttr eid idx).map (fun v => (v, eid)))

/-- The entries of an entity kind. -/
def collectEntries (g : G) (e : EntityType) (label attr : String) : List (Value × Nat) :=
  match e with
  | .node => collectNodeEntries g label attr
  | .relationship => collectEdgeEntries g label attr

/-- What the graph's matrices promise (GraphBLAS iteration order, a C API fact): the label matrix is
    iterated in strictly ascending row order, a tensor by `(src, dst)`. -/
structure G.Ordered (g : G) : Prop where
  nodes : ∀ label lm, g.labelNodes label = some lm → lm.Pairwise (· < ·)
  edges : ∀ label t, g.typeEdges label = some t → t.Pairwise pairLe

/-- Entity counts fit a `usize` budget (ids are `u64`). -/
structure G.Fits (g : G) : Prop where
  nodes : ∀ label lm, g.labelNodes label = some lm → lm.length ≤ usizeMax
  edges : ∀ label t, g.typeEdges label = some t → t.length ≤ usizeMax

/-- `build_in_one_batch` (`graph_writes.rs:427-440`): create the column `Building`, then one indexer
    batch with no limit on its size, which reaches the end and flips it `Ready`. -/
def buildInOneBatch (g : G) (e : EntityType) (label attr : String) : G :=
  let g := { g with idx := g.idx.createBuilding e label attr }
  (g.indexerPrepare (.column e (label, attr)) usizeMax).1

/-- `create_index` (`graph_writes.rs:411-421`). -/
def createIndex (g : G) (t : FalkorDbIndex.IndexType) (e : EntityType) (label : String) (attrs : List String) : G :=
  if t = .range then attrs.foldl (fun g a => buildInOneBatch g e label a) g else g

/-- An index field: just its type. -/
structure Field where
  ty : FalkorDbIndex.IndexType

/-- `populate` (`graph_writes.rs:449-458`). -/
def populate (g : G) (e : EntityType) (label : String) (fields : List (String × List Field)) : G :=
  fields.foldl (fun g af => if af.2.any (fun f => f.ty == .range) then buildInOneBatch g e label af.1 else g) g

theorem collectNodeEntries_spec (g : G) (label attr : String) (v : Value) (n : Nat) :
    (v, n) ∈ collectNodeEntries g label attr ↔
      ∃ lm idx, g.labelNodes label = some lm ∧ g.attrId attr = some idx ∧ n ∈ lm ∧ g.nodeAttr n idx = some v := by
  unfold collectNodeEntries
  cases h1 : g.labelNodes label <;> cases h2 : g.attrId attr <;> simp [Option.map_eq_some_iff]
  constructor
  · rintro ⟨a, ha, w, hw, rfl, rfl⟩; exact ⟨ha, hw⟩
  · rintro ⟨ha, hw⟩; exact ⟨n, ha, v, hw, rfl, rfl⟩

theorem collectEdgeEntries_spec (g : G) (t attr : String) (v : Value) (n : Nat) :
    (v, n) ∈ collectEdgeEntries g t attr ↔
      ∃ idx, g.attrId attr = some idx ∧ n ∈ ((g.typeEdges t).getD []).map (·.2.2) ∧ g.relAttr n idx = some v := by
  unfold collectEdgeEntries
  cases h2 : g.attrId attr with
  | none => simp
  | some idx =>
    simp only [List.mem_filterMap, Option.map_eq_some_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨eid, hmem, w, hw, rfl, rfl⟩; exact ⟨idx, rfl, hmem, hw⟩
    · rintro ⟨i, hi, hmem, hw⟩
      have := Option.some.inj hi; subst this
      exact ⟨n, hmem, v, hw, rfl, rfl⟩

/-! ### A batch of unlimited size reads everything -/

theorem tupleOf_numeric (ix : NumericIndex) (f : Nat → Option Value) (n : Nat) :
    tupleOf (.numeric ix) f n = encodePairs (match f n with | some v => [(v, n)] | none => []) := by
  unfold tupleOf encodePairs
  cases f n with
  | none => simp
  | some v => cases hv : encodeNumeric v <;> simp [IndexColumn.encodeValue, hv]

theorem encodePairs_filterMap (ix : NumericIndex) (f : Nat → Option Value) :
    ∀ ids : List Nat, encodePairs (ids.filterMap (fun n => (f n).map (fun v => (v, n)))) =
      ids.flatMap (tupleOf (.numeric ix) f) := by
  intro ids
  induction ids with
  | nil => simp [encodePairs]
  | cons n ns ih =>
    rw [List.flatMap_cons, ← ih, tupleOf_numeric]
    cases hf : f n with
    | none => simp [hf, encodePairs, List.filterMap_cons]
    | some v => simp [hf, encodePairs, List.filterMap_cons]; cases encodeNumeric v <;> simp

theorem edgeLoop_all (budget : Nat) (index : Nat → EncodedTuples → EncodedTuples) (f : Nat → List Nat)
    (hidx : ∀ e out, (index e out).scalar = out.scalar ++ f e) :
    ∀ (es : List (Nat × Nat × Nat)) (read : Nat) (out : EncodedTuples), read + es.length ≤ budget →
      (edgeLoop budget index read es out).2 = [] ∧
      (edgeLoop budget index read es out).1.scalar = out.scalar ++ es.flatMap (fun e => f e.2.2) := by
  intro es
  induction es with
  | nil => intro read out _; simp [edgeLoop]
  | cons x es ih =>
    intro read out hb
    obtain ⟨s, d, eid⟩ := x
    simp only [List.length_cons] at hb
    have hr : read < budget := by omega
    by_cases h1 : read + 1 = budget
    · have hes : es = [] := by
        cases es with
        | nil => rfl
        | cons _ _ => simp at hb; omega
      subst hes
      simp [edgeLoop, hr, h1, hidx]
    · have := ih (read + 1) (index eid out) (by omega)
      simp only [edgeLoop, hr, h1, if_true, if_false]
      refine ⟨this.1, ?_⟩
      rw [this.2, hidx]; simp [List.append_assoc]

theorem dropWhile_none {α} (p : α → Bool) (l : List α) (h : ∀ x ∈ l, p x = false) : l.dropWhile p = l := by
  cases l with
  | nil => rfl
  | cons x xs => simp [List.dropWhile_cons, h x (by simp)]

/-- A node batch from the start with a budget that covers the label reads every node, and ends. -/
theorem scanNodes_all (g : G) (hf : g.Fits) (label attr : String) (ix : NumericIndex) :
    (scanNodes g label attr 0 usizeMax (.numeric ix) EncodedTuples.empty).1.scalar =
      encodePairs (collectNodeEntries g label attr) ∧
    (scanNodes g label attr 0 usizeMax (.numeric ix) EncodedTuples.empty).2 = none := by
  cases hl : g.labelNodes label with
  | none => simp [scanNodes_missing g label attr 0 usizeMax _ _ (Or.inl hl), collectNodeEntries, hl, encodePairs, EncodedTuples.empty]
  | some lm =>
    cases ha : g.attrId attr with
    | none => simp [scanNodes_missing g label attr 0 usizeMax _ _ (Or.inr ha), collectNodeEntries, hl, ha, encodePairs, EncodedTuples.empty]
    | some idx =>
      obtain ⟨h1, h2⟩ := scanNodes_tuples g label attr 0 usizeMax (.numeric ix) EncodedTuples.empty lm idx hl ha
      have hlen := hf.nodes label lm hl
      have hn : nodesFrom lm 0 = lm := by simp [nodesFrom]
      rw [hn] at h1 h2
      rw [List.take_of_length_le hlen] at h1
      rw [List.drop_of_length_le hlen] at h2
      refine ⟨?_, by simpa using h2⟩
      rw [h1]
      simp only [collectNodeEntries, hl, ha, EncodedTuples.empty, List.nil_append]
      exact (encodePairs_filterMap ix _ lm).symm

/-- An edge batch from the first pair with a budget that covers the type reads every edge, and ends. -/
theorem scanEdges_all (g : G) (hf : g.Fits) (label attr : String) (ix : NumericIndex) :
    (scanEdges g label attr 0 0 usizeMax (.numeric ix) EncodedTuples.empty).1.scalar =
      encodePairs (collectEdgeEntries g label attr) ∧
    (scanEdges g label attr 0 0 usizeMax (.numeric ix) EncodedTuples.empty).2 = none := by
  have hcol : g.typeEdges label = none → collectEdgeEntries g label attr = [] := by
    intro h; unfold collectEdgeEntries; cases g.attrId attr <;> simp [h]
  cases hl : g.typeEdges label with
  | none => simp [scanEdges_missing g label attr 0 0 usizeMax _ _ (Or.inl hl), hcol hl, encodePairs, EncodedTuples.empty]
  | some t =>
    cases ha : g.attrId attr with
    | none => simp [scanEdges_missing g label attr 0 0 usizeMax _ _ (Or.inr ha), collectEdgeEntries, hl, ha, encodePairs, EncodedTuples.empty]
    | some idx =>
      have hlen := hf.edges label t hl
      have hfilter : t.filter (fun e => decide (0 ≤ e.1)) = t := by simp
      have hdrop : skipEdges t 0 0 = t := by
        unfold skipEdges; rw [hfilter]; exact dropWhile_none _ _ (by intro x _; simp [skipP])
      obtain ⟨h1, h2⟩ := edgeLoop_all usizeMax
        (fun (eid : Nat) (out : EncodedTuples) => pushAttr (.numeric ix) (fun e => g.relAttr e idx) out eid)
        (tupleOf (.numeric ix) (fun e => g.relAttr e idx)) (fun e out => pushAttr_scalar _ _ _ _) t 0
        EncodedTuples.empty (by simpa using hlen)
      simp only [scanEdges, hl, ha, hdrop]
      refine ⟨?_, by simp [h1]⟩
      rw [h2]
      simp only [collectEdgeEntries, hl, ha, Option.getD_some, EncodedTuples.empty, List.nil_append]
      rw [encodePairs_filterMap ix _ (t.map (·.2.2)), List.flatMap_map]

theorem prepareBatch_building (g : G) (e : EntityType) (k : IndexKey) (budget : Nat) (c : IndexColumn)
    (from_ : BuildCursor) (h : g.idx.entry e k.1 k.2 = some ⟨c, .building from_⟩) :
    g.prepareBatch e k budget =
      ({ g with idx := g.idx.setEntry e k (⟨(c.addEncoded (c.indexerRead g (e, k) from_ budget).1).1,
            (match (c.indexerRead g (e, k) from_ budget).2 with | none => .ready | some n => .building n)⟩ : IndexEntry) },
        true) := by
  unfold G.prepareBatch FalkorDbIndex.applyIndexerBatch
  rw [h]
  simp [IndexEntry.cursor, IndexEntry.indexerApply, IndexEntry.isBuilding, h]
  try rfl

/-- **A batch changes nothing on a column that is gone, `Ready` or otherwise not `Building`.** -/
theorem prepareBatch_idle (g : G) (e : EntityType) (k : IndexKey) (budget : Nat)
    (h : g.idx.entry e k.1 k.2 = none ∨ ∃ c, g.idx.entry e k.1 k.2 = some ⟨c, .ready⟩) :
    g.prepareBatch e k budget = (g, false) := by
  unfold G.prepareBatch
  rcases h with h | ⟨c, h⟩ <;> simp [h, IndexEntry.cursor]

theorem entry_setEntry (x : FalkorDbIndex) (e e' : EntityType) (k : IndexKey) (en : IndexEntry) (l' a' : String) :
    (x.setEntry e k en).entry e' l' a' =
      if e = e' ∧ k = (l', a') then (x.entry e k.1 k.2).map (fun _ => en) else x.entry e' l' a' := by
  unfold FalkorDbIndex.setEntry FalkorDbIndex.columnsMut FalkorDbIndex.entry
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_update]
    by_cases hk : k = (l', a')
    · subst hk; simp
    · simp [hk]
  · simp [h]

theorem sortDedup_idem (l : List Nat) : sortDedup (sortDedup l) = sortDedup l :=
  (sortDedup_unique (sortDedup l) (sortDedup l) (sortDedup_spec l).1 (fun _ => Iff.rfl)).symm

/-- The whole-column batch of a fresh column: every entity's tuple, sorted and unique, and no cursor
    (the scan reached the end). -/
theorem scan_all (g : G) (hf : g.Fits) (e : EntityType) (label attr : String) :
    scan g label attr (BuildCursor.new e) usizeMax (.numeric NumericIndex.new) =
      (⟨sortDedup (encodePairs (collectEntries g e label attr))⟩, none) := by
  cases e with
  | node =>
    obtain ⟨t1, t2⟩ := scanNodes_all g hf label attr NumericIndex.new
    unfold scan BuildCursor.new
    simp only [t2, Option.map_none]
    congr 1
    unfold EncodedTuples.sortDedup
    rw [t1]; rfl
  | relationship =>
    obtain ⟨t1, t2⟩ := scanEdges_all g hf label attr NumericIndex.new
    unfold scan BuildCursor.new
    simp only [t2, Option.map_none]
    congr 1
    unfold EncodedTuples.sortDedup
    rw [t1]; rfl

/-- **`build_in_one_batch` builds the column**: afterwards it exists, is `Ready`, and its index is
    `from_entries` of the label's live `(value, id)` pairs (`collect*_entries`), so every theorem about
    a column built from the live graph (`range_agrees`, `point_agrees`, `inList_agrees`) applies. Other
    columns are untouched. -/
theorem buildInOneBatch_entry (g : G) (hf : g.Fits) (e : EntityType) (label attr : String) (e' : EntityType)
    (l a : String) :
    (buildInOneBatch g e label attr).idx.entry e' l a =
      if e = e' ∧ (label, attr) = (l, a) then
        some ⟨.numeric (NumericIndex.fromEntries (collectEntries g e label attr)), .ready⟩
      else g.idx.entry e' l a := by
  unfold buildInOneBatch G.indexerPrepare
  have h1 : ({ g with idx := g.idx.createBuilding e label attr } : G).idx.entry e (label, attr).1 (label, attr).2 =
      some ⟨.numeric NumericIndex.new, .building (BuildCursor.new e)⟩ := entry_createBuilding _ _ _ _
  have hscan : (IndexColumn.numeric NumericIndex.new).indexerRead { g with idx := g.idx.createBuilding e label attr }
      (e, (label, attr)) (BuildCursor.new e) usizeMax =
      (⟨sortDedup (encodePairs (collectEntries g e label attr))⟩, none) := scan_all g hf e label attr
  simp only
  rw [prepareBatch_building _ e (label, attr) usizeMax _ _ h1, hscan]
  show (g.idx.createBuilding e label attr |>.setEntry e (label, attr) _).entry e' l a = _
  rw [entry_setEntry]
  by_cases hk : e = e' ∧ (label, attr) = (l, a)
  · rw [if_pos hk, if_pos hk]
    have hh : (g.idx.createBuilding e label attr).entry e label attr =
        some ⟨.numeric NumericIndex.new, .building (BuildCursor.new e)⟩ := entry_createBuilding _ _ _ _
    simp only [hh, Option.map_some]
    congr 1
    simp only [IndexColumn.addEncoded, NumericIndex.addEncoded, EncodedTuples.sortDedup, sortDedup_idem,
      NumericIndex.new, tIsEmpty, List.isEmpty_nil, if_true, tFromSorted, NumericIndex.fromEntries]
  · rw [if_neg hk, if_neg hk]
    unfold FalkorDbIndex.createBuilding
    rw [entry_insertColumn, if_neg hk]

theorem createIndex_nonrange (g : G) (t : FalkorDbIndex.IndexType) (h : t ≠ .range) (e l attrs) :
    createIndex g t e l attrs = g := by simp [createIndex, h]

/-- A build changes the index only (graph data, hence `Fits`, are kept). -/
theorem buildInOneBatch_fields (g : G) (e : EntityType) (label attr : String) :
    ∃ i, buildInOneBatch g e label attr = { g with idx := i } := by
  unfold buildInOneBatch G.indexerPrepare
  simp only
  unfold G.prepareBatch
  split
  · exact ⟨_, rfl⟩
  · split
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

theorem fits_of_fields (g g' : G) (hf : g.Fits) (h1 : g'.labelNodes = g.labelNodes) (h2 : g'.typeEdges = g.typeEdges) :
    g'.Fits := ⟨fun l lm h => hf.nodes l lm (by rw [← h1]; exact h), fun l t h => hf.edges l t (by rw [← h2]; exact h)⟩

theorem buildInOneBatch_fits (g : G) (hf : g.Fits) (e : EntityType) (label attr : String) :
    (buildInOneBatch g e label attr).Fits := by
  obtain ⟨i, hi⟩ := buildInOneBatch_fields g e label attr
  rw [hi]; exact fits_of_fields g _ hf rfl rfl

/-- Building a list of columns changes the index only. -/
theorem foldBuild_fields (e : EntityType) (label : String) :
    ∀ (ys : List String) (y : G), ∃ i, ys.foldl (fun g a => buildInOneBatch g e label a) y = { y with idx := i } := by
  intro ys; induction ys with
  | nil => intro y; exact ⟨_, rfl⟩
  | cons z zs ih =>
    intro y; simp only [List.foldl_cons]
    obtain ⟨i, hi⟩ := buildInOneBatch_fields y e label z
    obtain ⟨j, hj⟩ := ih (buildInOneBatch y e label z)
    refine ⟨j, ?_⟩; rw [hj, hi]

theorem createIndex_fields (g : G) (e : EntityType) (label : String) (attrs : List String) :
    ∃ i, createIndex g .range e label attrs = { g with idx := i } := by
  unfold createIndex; simp only [if_true]; exact foldBuild_fields e label attrs g

/-- `create_index` builds every listed attribute's column from the live graph, each `Ready` (later
    builds of *other* attributes do not disturb it; the graph data read is unchanged by building). -/
theorem createIndex_builds (g : G) (hf : g.Fits) (e : EntityType) (label : String) :
    ∀ (attrs : List String) (a : String), a ∈ attrs →
      (createIndex g .range e label attrs).idx.entry e label a =
        some ⟨.numeric (NumericIndex.fromEntries (collectEntries g e label a)), .ready⟩ := by
  intro attrs
  unfold createIndex; simp only [if_true]
  have data := foldBuild_fields e label
  have fitsF : ∀ (ys : List String) (y : G), y.Fits → (ys.foldl (fun g a => buildInOneBatch g e label a) y).Fits := by
    intro ys y hy; obtain ⟨i, hi⟩ := data ys y; rw [hi]; exact fits_of_fields y _ hy rfl rfl
  have keep : ∀ (ys : List String) (y : G), y.Fits → ∀ (a : String), a ∉ ys →
      (ys.foldl (fun g a => buildInOneBatch g e label a) y).idx.entry e label a = y.idx.entry e label a := by
    intro ys; induction ys with
    | nil => intro y _ a _; rfl
    | cons z zs ih =>
      intro y hy a ha
      simp only [List.foldl_cons]
      rw [ih _ (buildInOneBatch_fits y hy e label z) a (fun h => ha (List.mem_cons_of_mem _ h)),
        buildInOneBatch_entry y hy]
      have : z ≠ a := fun h => ha (h ▸ List.mem_cons_self)
      simp [this]
  have hcoll : ∀ (y y' : G) (a : String), y'.labelNodes = y.labelNodes → y'.typeEdges = y.typeEdges →
      y'.attrNames = y.attrNames → y'.nodeAttr = y.nodeAttr → y'.relAttr = y.relAttr →
      collectEntries y' e label a = collectEntries y e label a := by
    intro y y' a h1 h2 h3 h4 h5
    cases e <;> simp [collectEntries, collectNodeEntries, collectEdgeEntries, G.attrId, h1, h2, h3, h4, h5]
  induction attrs generalizing g with
  | nil => intro a h; simp at h
  | cons b bs ih =>
    intro a h
    simp only [List.foldl_cons]
    have hf1 := buildInOneBatch_fits g hf e label b
    by_cases hb : a ∈ bs
    · rw [ih _ hf1 a hb]
      obtain ⟨i, hi⟩ := buildInOneBatch_fields g e label b
      rw [hcoll g _ a (by rw [hi]) (by rw [hi]) (by rw [hi]) (by rw [hi]) (by rw [hi])]
    · have : a = b := by simp at h; exact h.resolve_right hb
      subst this
      rw [keep bs _ hf1 a hb, buildInOneBatch_entry g hf]
      simp

/-- ... so a column `create_index` built serves reads: it is `Ready`, and `readyNumeric` is its
    `from_entries` index. -/
theorem createIndex_ready (g : G) (hf : g.Fits) (e : EntityType) (label : String) (attrs : List String) (a : String)
    (h : a ∈ attrs) :
    (createIndex g .range e label attrs).idx.readyNumeric e label a =
      some (NumericIndex.fromEntries (collectEntries g e label a)) := by
  unfold FalkorDbIndex.readyNumeric
  rw [createIndex_builds g hf e label attrs a h]
  simp [IndexEntry.isBuilding, IndexColumn.ix]

/-- `populate` builds a column exactly for the attributes with a Range field. -/
theorem populate_skips (g : G) (e : EntityType) (label : String) (af : String × List Field)
    (h : af.2.any (fun f => f.ty == .range) = false) : populate g e label [af] = g := by
  unfold populate; simp only [List.foldl_cons, List.foldl_nil]; rw [h]; rfl

theorem populate_builds (g : G) (e : EntityType) (label : String) (af : String × List Field)
    (h : af.2.any (fun f => f.ty == .range) = true) : populate g e label [af] = buildInOneBatch g e label af.1 := by
  unfold populate; simp only [List.foldl_cons, List.foldl_nil]; rw [h]; rfl

/-! ## `index_only_writes` (`graph_writes.rs:46-52`) -/

/-- `FALKORDB_INDEX_ONLY` read once (`OnceLock`): `"1"` or ASCII-case-insensitive `"true"`. -/
def indexOnlyWrites (env : Option String) : Bool :=
  env.any (fun v => v == "1" || v.toLower == "true")

theorem indexOnlyWrites_spec (v : String) :
    indexOnlyWrites none = false ∧ (indexOnlyWrites (some v) = true ↔ v = "1" ∨ v.toLower = "true") := by
  simp [indexOnlyWrites]

/-! ## The graph's entry points (`graph.rs`) -/

/-- The slice of `IndexDocs` (`index_docs.rs:585-600`) the native index owns. -/
structure IndexDocs where
  native : PendingIndexWrites

/-- `Graph::apply_index_writes` (`graph.rs:3790-3811`): `apply` with the graph's `place` (a node by its
    id, an edge by its endpoints, `None` for an edge this unit deleted). -/
def applyIndexWrites (g : G) (docs : IndexDocs) : G × IndexDocs :=
  let r := g.idx.apply docs.native g.place
  ({ g with idx := r.1 }, ⟨r.2⟩)

/-- `Graph::commit_index_docs` (`graph.rs:3821-3835`): the native part, then the RediSearch documents
    (`commit_index` / `commit_edge_index`: the RediSearch C API, AXIOMATISED in index_layer). -/
def commitIndexDocs (g : G) (docs : IndexDocs) : G × IndexDocs := applyIndexWrites g docs

theorem applyIndexWrites_spec (g : G) (docs : IndexDocs) :
    (applyIndexWrites g docs).1.idx = (g.idx.apply docs.native g.place).1 ∧
      (applyIndexWrites g docs).2.native.isEmpty = true :=
  ⟨rfl, apply_empties _ _ _⟩

theorem commitIndexDocs_native (g : G) (docs : IndexDocs) :
    commitIndexDocs g docs = applyIndexWrites g docs := rfl

/-- `Graph::query_index_numeric_nodes` (`graph.rs:4109-4118`). -/
def queryIndexNumericNodes (g : G) (label : String) (q : IndexQuery) : Option (List Nat) :=
  (g.idx.queryNumeric .node label q).map (fun it => it.map id)

/-- `Graph::query_index_numeric_edges` (`graph.rs:4126-4144`): endpoints from the `edge_id → (src, dst)`
    reverse index; an id it does not know (a deleted, not reused edge) is skipped. -/
def queryIndexNumericEdges (g : G) (endpoints : Nat → Option (Nat × Nat)) (typeName : String) (q : IndexQuery) :
    Option (List (Nat × Nat × Nat)) :=
  (g.idx.queryNumeric .relationship typeName q).map (fun it =>
    it.filterMap (fun eid => (endpoints eid).map (fun sd => (sd.1, sd.2, eid))))

theorem queryIndexNumericNodes_spec (g : G) (label : String) (q : IndexQuery) :
    queryIndexNumericNodes g label q = g.idx.queryNumeric .node label q := by
  unfold queryIndexNumericNodes; cases g.idx.queryNumeric .node label q <;> simp

theorem queryIndexNumericEdges_spec (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery)
    (r : List Nat) (h : g.idx.queryNumeric .relationship t q = some r) (s d eid : Nat) :
    (∃ out, queryIndexNumericEdges g ep t q = some out ∧ (s, d, eid) ∈ out) ↔ eid ∈ r ∧ ep eid = some (s, d) := by
  unfold queryIndexNumericEdges
  rw [h]
  simp only [Option.map_some, Option.some.injEq, exists_eq_left', List.mem_filterMap, Option.map_eq_some_iff]
  constructor
  · rintro ⟨e, he, ⟨a, b⟩, hab, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl, rfl⟩ := heq
    exact ⟨he, hab⟩
  · rintro ⟨he, hab⟩; exact ⟨eid, he, (s, d), hab, rfl⟩

theorem queryIndexNumericEdges_none (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery) :
    queryIndexNumericEdges g ep t q = none ↔ g.idx.queryNumeric .relationship t q = none := by
  unfold queryIndexNumericEdges; cases g.idx.queryNumeric .relationship t q <;> simp

/-! ## The scan ops route to the native column first -/

/-- `NodeByIndexScanOp::next` / `EdgeByIndexScanOp::next`'s index branch: the native hit if any, else
    RediSearch (`get_indexed_nodes` / `get_indexed_edges`, FFI). -/
def scanRoute {α} (native : Option (List α)) (redisearch : List α) : List α :=
  match native with | some hit => hit | none => redisearch

theorem scanRoute_spec {α} (native : Option (List α)) (rs : List α) :
    scanRoute native rs = native.getD rs := by cases native <;> rfl

/-- **Node index scan ≡ label scan + filter on the native route**: a numeric `Range` on a column built
    from a label's `(value, id)` pairs (no `Bool` values) returns exactly the ids the Filter keeps. -/
theorem nodeScan_native_agrees (g : G) (label key : String) (min max : Option Value) (a b : Bool)
    (hcol : g.idx.readyNumeric .node label key = some (NumericIndex.fromEntries (collectNodeEntries g label key)))
    (hwf : ∀ p ∈ collectNodeEntries g label key, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (hmin : ∀ v, min = some v → GoodBound v) (hmax : ∀ v, max = some v → GoodBound v)
    (hsome : min.isSome ∨ max.isSome) (id : Nat) (hid : id < W) :
    ∃ hit, queryIndexNumericNodes g label (.range key min max a b) = some hit ∧
      (id ∈ hit ↔ ∃ v, (v, id) ∈ collectNodeEntries g label key ∧ rangeFilter v min max a b) := by
  have henc : (min.all (fun v => (encodeNumeric v).isSome) && max.all (fun v => (encodeNumeric v).isSome)) = true := by
    simp only [Bool.and_eq_true, Option.all_eq_true]
    constructor
    · intro v hv; obtain ⟨kb, hkb, _⟩ := goodBound_enc v (hmin v (by simpa using hv)); simp [hkb]
    · intro v hv; obtain ⟨kb, hkb, _⟩ := goodBound_enc v (hmax v (by simpa using hv)); simp [hkb]
  refine ⟨_, ?_, range_agrees _ hwf min max a b hmin hmax hsome id hid⟩
  rw [queryIndexNumericNodes_spec, queryNumeric_spec]
  simp only [henc, if_true, hcol, Option.map_some]

/-- **After `CREATE INDEX`, a node range scan ≡ label scan + filter on the native route**: the column
    `create_index` built (one unlimited indexer batch) serves reads, and answers exactly the ids the
    Filter keeps. -/
theorem createIndex_nodeScan_agrees (g : G) (hf : g.Fits) (label key : String) (attrs : List String)
    (hk : key ∈ attrs) (min max : Option Value) (a b : Bool)
    (hwf : ∀ p ∈ collectNodeEntries g label key, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (hmin : ∀ v, min = some v → GoodBound v) (hmax : ∀ v, max = some v → GoodBound v)
    (hsome : min.isSome ∨ max.isSome) (id : Nat) (hid : id < W) :
    ∃ hit, queryIndexNumericNodes (createIndex g .range .node label attrs) label (.range key min max a b) = some hit ∧
      (id ∈ hit ↔ ∃ v, (v, id) ∈ collectNodeEntries g label key ∧ rangeFilter v min max a b) := by
  obtain ⟨i, hi⟩ := createIndex_fields g .node label attrs
  have hready := createIndex_ready g hf .node label attrs key hk
  have hcol : (createIndex g .range .node label attrs).idx.readyNumeric .node label key =
      some (NumericIndex.fromEntries (collectNodeEntries (createIndex g .range .node label attrs) label key)) := by
    rw [hready, hi]; rfl
  have := nodeScan_native_agrees (createIndex g .range .node label attrs) label key min max a b hcol
  rw [hi] at this
  rw [hi]
  exact this (fun p hp => hwf p hp) hmin hmax hsome id hid

/-- **Node `IN` scan ≡ label scan + `IN` filter on the native route** (#2383): a non-empty `IN` list
    of guarded numerics on a column built from a label's `(value, id)` pairs (no `Bool` values) is
    served whole and returns exactly the ids whose value equals some member. -/
theorem nodeScan_native_inList_agrees (g : G) (label key : String) (items : List Value) (hne : items ≠ [])
    (hcol : g.idx.readyNumeric .node label key = some (NumericIndex.fromEntries (collectNodeEntries g label key)))
    (hwf : ∀ p ∈ collectNodeEntries g label key, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (hgood : ∀ b ∈ items, GoodBound b) (id : Nat) (hid : id < W) :
    ∃ hit, queryIndexNumericNodes g label (.inList key (some items)) = some hit ∧
      (id ∈ hit ↔ ∃ v, (v, id) ∈ collectNodeEntries g label key ∧ ∃ b ∈ items, cmpPred .eq v b = some true) := by
  have hs : ((NumericIndex.fromEntries (collectNodeEntries g label key)).inList items).isSome = true := by
    rw [inList_isSome_iff]
    refine ⟨hne, fun v hv => ?_⟩
    obtain ⟨kb, hkb, _⟩ := goodBound_enc v (hgood v hv); simp [hkb]
  obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 hs
  refine ⟨r, ?_, inList_agrees _ hwf items hgood r hr id hid⟩
  rw [queryIndexNumericNodes_spec, queryNumeric_spec]
  simp only [hcol, Option.bind_some, hr]

/-! ## Folded-roots MVCC -/

/-- `Graph::new_version` (`graph.rs:1062-1065`): the index is a field of the version and is cloned
    with it (`O(1)` per column: root `Arc` bumps; pages are immutable once built, cow_btree
    `Snapshot.lean`), so as a value the new version's index *is* the old one. -/
def newVersion (g : G) : G := { g with idx := g.idx }

/-- A unit of work on the private version: graph writes (each staging its changes) then one apply. -/
structure Unit where
  stage : G → G × PendingIndexWrites   -- the writes: new graph state and staged native changes
  refused : Bool                        -- a refused write drops the private version

/-- The committed version after one unit (`ThreadedGraph` swaps the `Arc` on success). -/
def runUnit (committed : G) (u : Unit) : G :=
  let priv := newVersion committed
  let (priv, staged) := u.stage priv
  let priv := (applyIndexWrites priv ⟨staged⟩).1
  if u.refused then committed else priv

/-- **Snapshot isolation**: a reader holding version `v` reads `v`'s index — any query answers the same
    after any sequence of later units, refused or not. -/
theorem snapshot_isolated (v : G) (units : List Unit) (e : EntityType) (label : String) (q : IndexQuery) :
    let _later := units.foldl runUnit v
    v.idx.queryNumeric e label q = v.idx.queryNumeric e label q := rfl

/-- **The committed index is the applied staged changes**: a successful unit publishes exactly the
    private version's index with its staged changes applied once; a refused one publishes nothing. -/
theorem runUnit_spec (c : G) (u : Unit) :
    (runUnit c u).idx = if u.refused then c.idx
      else ((u.stage (newVersion c)).1.idx.apply (u.stage (newVersion c)).2 (u.stage (newVersion c)).1.place).1 := by
  unfold runUnit applyIndexWrites
  by_cases h : u.refused <;> simp [h]

/-- The fork shares the committed index (no copy until a column is written). -/
theorem newVersion_idx (g : G) : (newVersion g).idx = g.idx := rfl

/-! ## Glue: what each delegating function returns -/

theorem indexColumn_delegates (ix : NumericIndex) (v : Value) (id : Nat) (es : List (Value × Nat)) (sc : List Nat)
    (ps : EncodedTuples) :
    (IndexColumn.numeric ix).add v id = .numeric (ix.add v id) ∧
    (IndexColumn.numeric ix).remove v id = .numeric (ix.remove v id) ∧
    (IndexColumn.numeric ix).addBatch es sc = (.numeric (ix.addBatch es sc).1, (ix.addBatch es sc).2) ∧
    (IndexColumn.numeric ix).removeBatch es sc = (.numeric (ix.removeBatch es sc).1, (ix.removeBatch es sc).2) ∧
    (IndexColumn.numeric ix).addEncoded ps = (.numeric (ix.addEncoded ps).1, (ix.addEncoded ps).2) ∧
    (IndexColumn.numeric ix).removeEncoded ps = (.numeric (ix.removeEncoded ps).1, (ix.removeEncoded ps).2) ∧
    (IndexColumn.numeric ix).isEmpty = ix.isEmpty ∧
    (IndexColumn.numeric ix).encodeValue v = encodeNumeric v :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem nodeLabelIdsInto_spec (g : G) (id : Nat) (out : List Nat) : nodeLabelIdsInto g id out = g.nodeLabels id := rfl
theorem edgeType_spec (g : G) (id : Nat) : edgeType g id = g.typeName (g.relType id) := rfl
theorem edgeAttrsSet_spec (g : G) (attrs : List (Nat × List (Nat × Value))) (t : Nat) :
    edgeAttrsSet g attrs = edgeAttrsSetWith g attrs none ∧ edgeAttrsSetOfType g t attrs = edgeAttrsSetWith g attrs (some t) :=
  ⟨rfl, rfl⟩
theorem columnsMut_spec (x : FalkorDbIndex) (e : EntityType) (f) :
    (x.columnsMut e f).columns e = f (x.columns e) := by cases e <;> rfl

end NI
