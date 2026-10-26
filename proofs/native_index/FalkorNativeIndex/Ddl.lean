import FalkorNativeIndex.Writes
set_option linter.unusedSimpArgs false

/-!
# Index DDL (`graph_writes.rs:395-480`, `:510-552`), the graph's index entry points
# (`graph.rs:3780-3809`, `:4087-4122`), the scan routing (`node_by_index_scan.rs:306-323`,
# `edge_by_index_scan.rs:329-349`), and folded-roots MVCC (`graph.rs:441-451`, `:1062-1065`)
-/

namespace NI

/-- `collect_node_entries` (`graph_writes.rs:512-527`): every node of the label matrix that has `attr`. -/
def collectNodeEntries (g : G) (label attr : String) : List (Value × Nat) :=
  match g.labelNodes label, g.attrId attr with
  | some lm, some idx => lm.filterMap (fun n => (g.nodeAttr n idx).map (fun v => (v, n)))
  | _, _ => []

/-- `collect_edge_entries` (`graph_writes.rs:531-552`). -/
def collectEdgeEntries (g : G) (typeName attr : String) : List (Value × Nat) :=
  match g.attrId attr with
  | none => []
  | some idx =>
    let edgeIds := (g.typeEdges typeName).getD []
    edgeIds.filterMap (fun eid => (g.relAttr eid idx).map (fun v => (v, eid)))

/-- `match entity { Node => collect_node_entries, Relationship => collect_edge_entries }` (`:474-477`). -/
def collectEntries (g : G) (e : EntityType) (label attr : String) : List (Value × Nat) :=
  match e with
  | .node => collectNodeEntries g label attr
  | .relationship => collectEdgeEntries g label attr

/-- `build_column` (`graph_writes.rs:468-480`). -/
def buildColumn (g : G) (e : EntityType) (label attr : String) : G :=
  let pairs := collectEntries g e label attr
  { g with idx := g.idx.buildNumeric e label attr pairs }

/-- `create_index` (`graph_writes.rs:404-416`). -/
def createIndex (g : G) (t : FalkorDbIndex.IndexType) (e : EntityType) (label : String) (attrs : List String) : G :=
  if t = .range then attrs.foldl (fun g a => buildColumn g e label a) g else g

/-- An index field: just its type. -/
structure Field where
  ty : FalkorDbIndex.IndexType

/-- `populate` (`graph_writes.rs:421-432`). -/
def populate (g : G) (e : EntityType) (label : String) (fields : List (String × List Field)) : G :=
  fields.foldl (fun g af => if af.2.any (fun f => f.ty == .range) then buildColumn g e label af.1 else g) g

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
      ∃ idx, g.attrId attr = some idx ∧ n ∈ (g.typeEdges t).getD [] ∧ g.relAttr n idx = some v := by
  unfold collectEdgeEntries
  cases h2 : g.attrId attr <;> simp [Option.map_eq_some_iff]
  constructor
  · rintro ⟨a, ha, w, hw, rfl, rfl⟩; exact ⟨ha, hw⟩
  · rintro ⟨ha, hw⟩; exact ⟨n, ha, v, hw, rfl, rfl⟩

/-- **A built column is exactly the label's live `(value, id)` pairs**: its numeric index is
    `from_entries` of `collect_*_entries`, so `range_agrees` / `point_agrees` make a scan of it equal a
    label scan + filter. Other columns are untouched. -/
theorem buildColumn_numeric (g : G) (e : EntityType) (label attr : String) (e' : EntityType) (l a : String) :
    (buildColumn g e label attr).idx.numeric e' l a =
      if e = e' ∧ (label, attr) = (l, a) then
        some (NumericIndex.fromEntries (collectEntries g e label attr))
      else g.idx.numeric e' l a := by
  unfold buildColumn; simp only; rw [numeric_buildNumeric]

theorem createIndex_nonrange (g : G) (t : FalkorDbIndex.IndexType) (h : t ≠ .range) (e l attrs) :
    createIndex g t e l attrs = g := by simp [createIndex, h]

/-- `create_index` builds every listed attribute's column from the live graph (later builds of
    *other* attributes do not disturb it; the graph data read is unchanged by building). -/
theorem createIndex_builds (g : G) (e : EntityType) (label : String) :
    ∀ (attrs : List String) (a : String), a ∈ attrs →
      (createIndex g .range e label attrs).idx.numeric e label a =
        some (NumericIndex.fromEntries (collectEntries g e label a)) := by
  intro attrs
  unfold createIndex; simp only [if_true]
  -- building never changes the graph data, only the index
  have data : ∀ (ys : List String) (y : G), (ys.foldl (fun g a => buildColumn g e label a) y).labelNodes = y.labelNodes ∧
      (ys.foldl (fun g a => buildColumn g e label a) y).typeEdges = y.typeEdges ∧
      (ys.foldl (fun g a => buildColumn g e label a) y).attrNames = y.attrNames ∧
      (ys.foldl (fun g a => buildColumn g e label a) y).nodeAttr = y.nodeAttr ∧
      (ys.foldl (fun g a => buildColumn g e label a) y).relAttr = y.relAttr := by
    intro ys; induction ys with
    | nil => intro y; simp
    | cons z zs ih => intro y; simp only [List.foldl_cons]; rw [(ih _).1, (ih _).2.1, (ih _).2.2.1, (ih _).2.2.2.1, (ih _).2.2.2.2]; simp [buildColumn]
  have keep : ∀ (ys : List String) (y : G) (a : String), a ∉ ys →
      (ys.foldl (fun g a => buildColumn g e label a) y).idx.numeric e label a = y.idx.numeric e label a := by
    intro ys; induction ys with
    | nil => intro y a _; rfl
    | cons z zs ih =>
      intro y a ha
      simp only [List.foldl_cons]
      rw [ih _ a (fun h => ha (List.mem_cons_of_mem _ h)), buildColumn_numeric]
      have : z ≠ a := fun h => ha (h ▸ List.mem_cons_self)
      simp [this]
  have hcoll : ∀ (y y' : G) (a : String), y'.labelNodes = y.labelNodes → y'.typeEdges = y.typeEdges → y'.attrNames = y.attrNames →
      y'.nodeAttr = y.nodeAttr → y'.relAttr = y.relAttr →
      collectEntries y' e label a = collectEntries y e label a := by
    intro y y' a h1 h2 h3 h4 h5
    cases e <;> simp [collectEntries, collectNodeEntries, collectEdgeEntries, G.attrId, h1, h2, h3, h4, h5]
  induction attrs generalizing g with
  | nil => intro a h; simp at h
  | cons b bs ih =>
    intro a h
    simp only [List.foldl_cons]
    by_cases hb : a ∈ bs
    · rw [ih _ a hb, hcoll g (buildColumn g e label b) a rfl rfl rfl rfl rfl]
    · have : a = b := by simp at h; exact h.resolve_right hb
      subst this
      rw [keep bs _ a hb, buildColumn_numeric]
      simp

/-- `populate` builds a column exactly for the attributes with a Range field. -/
theorem populate_skips (g : G) (e : EntityType) (label : String) (af : String × List Field)
    (h : af.2.any (fun f => f.ty == .range) = false) : populate g e label [af] = g := by
  unfold populate; simp only [List.foldl_cons, List.foldl_nil]; rw [h]; rfl

theorem populate_builds (g : G) (e : EntityType) (label : String) (af : String × List Field)
    (h : af.2.any (fun f => f.ty == .range) = true) : populate g e label [af] = buildColumn g e label af.1 := by
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

/-- `Graph::apply_index_writes` (`graph.rs:3780-3788`). -/
def applyIndexWrites (g : G) (docs : IndexDocs) : G × IndexDocs :=
  let r := g.idx.apply docs.native
  ({ g with idx := r.1 }, ⟨r.2⟩)

/-- `Graph::commit_index_docs` (`graph.rs:3797-3808`): the native part, then the RediSearch documents
    (`commit_index` / `commit_edge_index`: the RediSearch C API, AXIOMATISED in index_layer). -/
def commitIndexDocs (g : G) (docs : IndexDocs) : G × IndexDocs := applyIndexWrites g docs

theorem applyIndexWrites_spec (g : G) (docs : IndexDocs) :
    (applyIndexWrites g docs).1.idx = (g.idx.apply docs.native).1 ∧ (applyIndexWrites g docs).2.native.isEmpty = true :=
  ⟨rfl, apply_empties _ _⟩

theorem commitIndexDocs_native (g : G) (docs : IndexDocs) :
    commitIndexDocs g docs = applyIndexWrites g docs := rfl

/-- `Graph::query_index_numeric_nodes` (`graph.rs:4087-4096`). -/
def queryIndexNumericNodes (g : G) (label : String) (q : IndexQuery) : Option (List Nat) :=
  (g.idx.queryNumeric .node label q).map (fun it => it.map id)

/-- `Graph::query_index_numeric_edges` (`graph.rs:4104-4122`): endpoints from the `edge_id → (src, dst)`
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
    (hcol : g.idx.numeric .node label key = some (NumericIndex.fromEntries (collectNodeEntries g label key)))
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

/-- **Node `IN` scan ≡ label scan + `IN` filter on the native route** (#2383): a non-empty `IN` list
    of guarded numerics on a column built from a label's `(value, id)` pairs (no `Bool` values) is
    served whole and returns exactly the ids whose value equals some member. -/
theorem nodeScan_native_inList_agrees (g : G) (label key : String) (items : List Value) (hne : items ≠ [])
    (hcol : g.idx.numeric .node label key = some (NumericIndex.fromEntries (collectNodeEntries g label key)))
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
    (runUnit c u).idx = if u.refused then c.idx else ((u.stage (newVersion c)).1.idx.apply (u.stage (newVersion c)).2).1 := by
  unfold runUnit applyIndexWrites
  by_cases h : u.refused <;> simp [h]

/-- The fork shares the committed index (no copy until a column is written). -/
theorem newVersion_idx (g : G) : (newVersion g).idx = g.idx := rfl

/-! ## Glue: what each delegating function returns -/

theorem indexColumn_delegates (ix : NumericIndex) (v : Value) (id : Nat) (es : List (Value × Nat)) (sc ps : List Nat) :
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
