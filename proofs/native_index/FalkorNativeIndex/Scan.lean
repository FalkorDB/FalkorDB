import FalkorNativeIndex.Ddl
set_option linter.unusedSimpArgs false

/-!
# The scan that stands in for a building column, and who feeds RediSearch (#3547, #2386)

* `Graph::node_matches` / `relationship_matches` (`graph.rs`) and the filtered scans
  the two scan ops run on `Some(Err(scan_filter))` (`node_by_index_scan.rs`,
  `edge_by_index_scan.rs`): the scan keeps **exactly what the built column would answer**
  (`nodeFilteredScan_agrees`, `edgeFilteredScan_agrees`). The planner removes a predicate it hands
  to an index scan, so this filter is the only check the scanned rows get.
* `redisearch_fields` (`graph.rs`): the identity without `index-falkordb`; with it, the
  fields that still go to RediSearch, i.e. all but the Range fields (`redisearchFields_mem` and its
  corollaries).
-/

namespace NI

/-! ## `ScanFilter::resolve` and `matches` -/

theorem resolveGo_all {A B : Type} (r : A → Option B) : ∀ (cs : List (A × Selection)) (l : List (B × Selection)),
    resolveGo true r cs = some l ↔
      (∀ c ∈ cs, (r c.1).isSome = true) ∧ l = cs.filterMap (fun c => (r c.1).map (fun b => (b, c.2))) := by
  intro cs
  induction cs with
  | nil => intro l; simp [resolveGo]
  | cons c cs ih =>
    intro l
    obtain ⟨a, sel⟩ := c
    cases hr : r a with
    | none => simp [resolveGo, hr]
    | some b =>
      simp only [resolveGo, hr, Option.map_eq_some_iff, List.mem_cons, forall_eq_or_imp, Option.isSome_some,
        List.filterMap_cons, Option.map_some, true_and]
      constructor
      · rintro ⟨l', hl', rfl⟩
        obtain ⟨h1, h2⟩ := (ih l').1 hl'
        exact ⟨h1, by rw [h2]⟩
      · rintro ⟨h1, rfl⟩
        exact ⟨_, (ih _).2 ⟨h1, rfl⟩, rfl⟩

theorem resolveGo_any {A B : Type} (r : A → Option B) : ∀ (cs : List (A × Selection)),
    resolveGo false r cs = some (cs.filterMap (fun c => (r c.1).map (fun b => (b, c.2)))) := by
  intro cs
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    obtain ⟨a, sel⟩ := c
    cases hr : r a <;> simp [resolveGo, hr, ih]

theorem all_congr' {α} (l : List α) (f g : α → Bool) (h : ∀ x ∈ l, f x = g x) : l.all f = l.all g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.all_cons, h x List.mem_cons_self, ih (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem any_congr' {α} (l : List α) (f g : α → Bool) (h : ∀ x ∈ l, f x = g x) : l.any f = l.any g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.any_cons, h x List.mem_cons_self, ih (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem any_filterMap {α β} (g : α → Option β) (p : β → Bool) (l : List α) :
    (l.filterMap g).any p = l.any (fun x => match g x with | some y => p y | none => false) := by
  induction l with
  | nil => rfl
  | cons x xs ih => cases hg : g x <;> simp [List.filterMap_cons, hg, ih]

theorem all_filterMap_some {α β} (g : α → Option β) (p : β → Bool) (l : List α)
    (h : ∀ x ∈ l, (g x).isSome = true) :
    (l.filterMap g).all p = l.all (fun x => match g x with | some y => p y | none => false) := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    have hx := h x List.mem_cons_self
    have ih' := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    cases hg : g x with
    | none => simp [hg] at hx
    | some y => simp [List.filterMap_cons, hg, ih']

theorem resolve_all_eq {A B : Type} (r : A → Option B) (cs : List (A × Selection)) (L : List (B × Selection))
    (h : resolveGo true r cs = some L) :
    (ScanFilter.all cs).resolve r = if L = [] then none else some (.all L) := by
  simp only [ScanFilter.resolve, h]; cases L <;> simp

theorem resolve_any_eq {A B : Type} (r : A → Option B) (cs : List (A × Selection)) (L : List (B × Selection))
    (h : resolveGo false r cs = some L) :
    (ScanFilter.any cs).resolve r = if L = [] then none else some (.any L) := by
  simp only [ScanFilter.resolve, h]; cases L <;> simp

/-- **`resolve` keeps the verdict**: a filter resolved by `r` matches an entity (whose values `vo` reads by
    resolved name) iff the original matches it reading each name through `r`; `None` is "matches nothing".
    (A conjunction names at least one attribute.) -/
theorem matches_resolve {A B : Type} (f : ScanFilter A) (r : A → Option B) (vo : B → Option Value)
    (hne : ∀ cs, f = .all cs → cs ≠ []) :
    (match f.resolve r with | none => false | some f' => f'.matches vo) =
      f.matches (fun a => (r a).bind vo) := by
  cases f with
  | all cs =>
    have hne' := hne cs rfl
    by_cases hall : ∀ c ∈ cs, (r c.1).isSome = true
    · have hg := (resolveGo_all r cs _).2 ⟨hall, rfl⟩
      have hl : cs.filterMap (fun c => (r c.1).map (fun b => (b, c.2))) ≠ [] := by
        intro h
        have := List.filterMap_eq_nil_iff.1 h
        obtain ⟨c, hc⟩ := List.exists_mem_of_ne_nil _ hne'
        have h1 := this c hc
        have h2 := hall c hc
        cases hr : r c.1 <;> simp [hr] at h1 h2
      rw [resolve_all_eq r cs _ hg, if_neg hl]
      simp only [ScanFilter.matches]
      rw [all_filterMap_some _ _ _ (fun c hc => by
        obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (hall c hc); simp [hb])]
      refine all_congr' _ _ _ (fun c hc => ?_)
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (hall c hc)
      simp [hb]
    · have hnone : resolveGo true r cs = none := by
        cases h : resolveGo true r cs with
        | none => rfl
        | some l => exact absurd ((resolveGo_all r cs l).1 h).1 hall
      simp only [ScanFilter.resolve, hnone]
      simp only [ScanFilter.matches]
      symm
      rw [Bool.eq_false_iff]
      intro hh
      apply hall
      intro c hc
      have := List.all_eq_true.1 hh c hc
      cases hr : r c.1 <;> simp [hr] at this ⊢
  | any cs =>
    have hg := resolveGo_any r cs
    rw [resolve_any_eq r cs _ hg]
    by_cases hl : cs.filterMap (fun c => (r c.1).map (fun b => (b, c.2))) = []
    · rw [if_pos hl]
      simp only [ScanFilter.matches]
      symm
      rw [Bool.eq_false_iff]
      intro hh
      obtain ⟨c, hc, hv⟩ := List.any_eq_true.1 hh
      cases hr : r c.1 with
      | none => simp [hr] at hv
      | some b =>
        have := List.filterMap_eq_nil_iff.1 hl c hc
        simp [hr] at this
    · rw [if_neg hl]
      simp only [ScanFilter.matches]
      rw [any_filterMap]
      refine any_congr' _ _ _ (fun c hc => ?_)
      cases hr : r c.1 <;> simp [hr]

/-! ## `node_matches`, `relationship_matches` -/

/-- `Graph::node_matches` (`graph.rs`): `scan_filter.matches(|&attr| get_node_attribute_by_idx(id, attr))`
    on a filter whose attributes are the graph's ids. -/
def G.nodeMatches (g : G) (id : Nat) (f : ScanFilter Nat) : Bool :=
  f.matches (fun attr => g.nodeAttr id attr)

/-- `Graph::relationship_matches` (`graph.rs`). -/
def G.relationshipMatches (g : G) (id : Nat) (f : ScanFilter Nat) : Bool :=
  f.matches (fun attr => g.relAttr id attr)

theorem nodeMatches_spec (g : G) (id : Nat) (f : ScanFilter Nat) :
    g.nodeMatches id f = f.matches (fun attr => g.nodeAttr id attr) := rfl

theorem relationshipMatches_spec (g : G) (id : Nat) (f : ScanFilter Nat) :
    g.relationshipMatches id f = f.matches (fun attr => g.relAttr id attr) := rfl

/-- A node holding none of the filter's attributes matches a union of nothing: an entity missing an
    attribute does not match its condition (no column holds it). -/
theorem nodeMatches_none (g : G) (id : Nat) (cs : List (Nat × Selection))
    (h : ∀ c ∈ cs, g.nodeAttr id c.1 = none) : g.nodeMatches id (.any cs) = false := by
  simp only [G.nodeMatches, ScanFilter.matches, List.any_eq_false]
  intro c hc; simp [h c hc]

/-! ## The filtered scans -/

/-- The scan `NodeByIndexScanOp::next` runs for `Some(Err(scan_filter))`: the filter's names are resolved
    once (`get_node_attribute_id`; a name the graph never saw is held by no node, so a condition on it
    matches nothing and `resolve` is `None` when the whole filter does), then the label's nodes are
    filtered by `node_matches`. (The Rust scans the *pattern's* labels, a subset of the index label's
    nodes; the op's `extra_labels` filter then removes the same nodes from the index route, so the rows
    agree: `ops_traverse`.) -/
def nodeFilteredScan (g : G) (label : String) (f : ScanFilter String) : List Nat :=
  match f.resolve g.attrId with
  | none => []
  | some rf => ((g.labelNodes label).getD []).filter (fun id => g.nodeMatches id rf)

/-- The same for `EdgeByIndexScanOp::next`: `get_all_edges(type)` filtered by `relationship_matches`. -/
def edgeFilteredScan (g : G) (typeName : String) (f : ScanFilter String) : List (Nat × Nat × Nat) :=
  match f.resolve g.attrId with
  | none => []
  | some rf => ((g.typeEdges typeName).getD []).filter (fun e => g.relationshipMatches e.2.2 rf)

/-- One condition answered by the column `CREATE INDEX` would build: `id` is among the rows of the column
    for the condition's attribute, looked up for its selection. -/
def NodeCond (g : G) (label : String) (c : String × Selection) (id : Nat) : Prop :=
  id ∈ (NumericIndex.fromEntries (collectNodeEntries g label c.1)).lookupSel c.2

def EdgeCond (g : G) (typeName : String) (c : String × Selection) (id : Nat) : Prop :=
  id ∈ (NumericIndex.fromEntries (collectEdgeEntries g typeName c.1)).lookupSel c.2

/-- What every condition of a filter says together: all of a conjunction, any of a union (and a
    conjunction names at least one attribute). -/
def FilterHolds (f : ScanFilter String) (cond : String × Selection → Prop) : Prop :=
  match f with
  | .all cs => cs ≠ [] ∧ ∀ c ∈ cs, cond c
  | .any cs => ∃ c ∈ cs, cond c

theorem collectNodeEntries_fun (g : G) (label attr : String) :
    ∀ p ∈ collectNodeEntries g label attr, ∀ q ∈ collectNodeEntries g label attr, p.2 = q.2 → p.1 = q.1 := by
  rintro ⟨v, n⟩ hp ⟨w, m⟩ hq h
  obtain ⟨lm, idx, hl, ha, _, hv⟩ := (collectNodeEntries_spec g label attr v n).1 hp
  obtain ⟨lm', idx', hl', ha', _, hw⟩ := (collectNodeEntries_spec g label attr w m).1 hq
  simp only at h; subst h
  rw [hl] at hl'; rw [ha] at ha'
  cases hl'; cases ha'
  rw [hv] at hw; exact Option.some.inj hw

theorem collectEdgeEntries_fun (g : G) (t attr : String) :
    ∀ p ∈ collectEdgeEntries g t attr, ∀ q ∈ collectEdgeEntries g t attr, p.2 = q.2 → p.1 = q.1 := by
  rintro ⟨v, n⟩ hp ⟨w, m⟩ hq h
  obtain ⟨idx, ha, _, hv⟩ := (collectEdgeEntries_spec g t attr v n).1 hp
  obtain ⟨idx', ha', _, hw⟩ := (collectEdgeEntries_spec g t attr w m).1 hq
  simp only at h; subst h
  rw [ha] at ha'; cases ha'
  rw [hv] at hw; exact Option.some.inj hw

/-- A condition's column answers a node iff the node is in the label, the attribute exists, and the node's
    value is one the selection matches: the very check `ScanFilter::matches` makes. -/
theorem nodeCond_iff (g : G) (label : String) (c : String × Selection)
    (hlt : ∀ lm, g.labelNodes label = some lm → ∀ n ∈ lm, n < W) (id : Nat) (hid : id < W) :
    NodeCond g label c id ↔
      ∃ lm idx, g.labelNodes label = some lm ∧ g.attrId c.1 = some idx ∧ id ∈ lm ∧
        ∃ v, g.nodeAttr id idx = some v ∧ c.2.matches v = true := by
  have hids : ∀ p ∈ collectNodeEntries g label c.1, p.2 < W := by
    rintro ⟨v, n⟩ hp
    obtain ⟨lm, idx, hl, _, hn, _⟩ := (collectNodeEntries_spec g label c.1 v n).1 hp
    exact hlt lm hl n hn
  unfold NodeCond
  rw [lookupSel_fromEntries _ hids (collectNodeEntries_fun g label c.1) c.2 id hid]
  constructor
  · rintro ⟨v, hv, hm⟩
    obtain ⟨lm, idx, hl, ha, hn, hv'⟩ := (collectNodeEntries_spec g label c.1 v id).1 hv
    exact ⟨lm, idx, hl, ha, hn, v, hv', hm⟩
  · rintro ⟨lm, idx, hl, ha, hn, v, hv, hm⟩
    exact ⟨v, (collectNodeEntries_spec g label c.1 v id).2 ⟨lm, idx, hl, ha, hn, hv⟩, hm⟩

theorem edgeCond_iff (g : G) (t : String) (c : String × Selection)
    (hlt : ∀ tt, g.typeEdges t = some tt → ∀ e ∈ tt, e.2.2 < W) (id : Nat) (hid : id < W) :
    EdgeCond g t c id ↔
      ∃ idx, g.attrId c.1 = some idx ∧ id ∈ ((g.typeEdges t).getD []).map (·.2.2) ∧
        ∃ v, g.relAttr id idx = some v ∧ c.2.matches v = true := by
  have hids : ∀ p ∈ collectEdgeEntries g t c.1, p.2 < W := by
    rintro ⟨v, n⟩ hp
    obtain ⟨idx, _, hn, _⟩ := (collectEdgeEntries_spec g t c.1 v n).1 hp
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hn
    cases ht : g.typeEdges t with
    | none => simp [ht] at he
    | some tt => simp only [ht, Option.getD_some] at he; exact hlt tt ht e he
  unfold EdgeCond
  rw [lookupSel_fromEntries _ hids (collectEdgeEntries_fun g t c.1) c.2 id hid]
  constructor
  · rintro ⟨v, hv, hm⟩
    obtain ⟨idx, ha, hn, hv'⟩ := (collectEdgeEntries_spec g t c.1 v id).1 hv
    exact ⟨idx, ha, hn, v, hv', hm⟩
  · rintro ⟨idx, ha, hn, v, hv, hm⟩
    exact ⟨v, (collectEdgeEntries_spec g t c.1 v id).2 ⟨idx, ha, hn, hv⟩, hm⟩

theorem holds_iff (ao : Option Nat) (attrOf : Nat → Option Value) (sel : Selection) :
    sel.matchesOpt (ao.bind attrOf) = true ↔
      ∃ idx v, ao = some idx ∧ attrOf idx = some v ∧ sel.matches v = true := by
  cases ao with
  | none => simp
  | some i => cases h : attrOf i <;> simp [h, Selection.matchesOpt]

theorem mem_getD_iff (o : Option (List Nat)) (id : Nat) :
    id ∈ o.getD [] ↔ ∃ lm, o = some lm ∧ id ∈ lm := by
  cases o <;> simp

/-- **The scan keeps exactly what the built columns answer** (nodes). A node is scanned for a filter iff
    every condition (a conjunction) or some condition (a union) of the filter is answered by the column
    `CREATE INDEX` would have built for that attribute from the live graph: `nodeFilteredScan` is the only
    check the scanned rows get, and it equals the built columns' answer. -/
theorem nodeFilteredScan_agrees (g : G) (label : String) (f : ScanFilter String)
    (hne : ∀ cs, f = .all cs → cs ≠ [])
    (hlt : ∀ lm, g.labelNodes label = some lm → ∀ n ∈ lm, n < W) (id : Nat) (hid : id < W) :
    id ∈ nodeFilteredScan g label f ↔ FilterHolds f (fun c => NodeCond g label c id) := by
  have hP := matches_resolve f g.attrId (fun attr => g.nodeAttr id attr) hne
  have hmem : id ∈ nodeFilteredScan g label f ↔
      id ∈ (g.labelNodes label).getD [] ∧ f.matches (fun a => (g.attrId a).bind (fun attr => g.nodeAttr id attr)) = true := by
    unfold nodeFilteredScan
    rw [← hP]
    cases hr : f.resolve g.attrId with
    | none => simp
    | some rf => simp [List.mem_filter, G.nodeMatches]
  rw [hmem]
  cases f with
  | all cs =>
    have hne' := hne cs rfl
    simp only [FilterHolds, ScanFilter.matches, List.all_eq_true, mem_getD_iff]
    constructor
    · rintro ⟨⟨lm, hl, hn⟩, hall⟩
      refine ⟨hne', fun c hc => ?_⟩
      have := hall c hc
      have h2 := (holds_iff (g.attrId c.1) (fun attr => g.nodeAttr id attr) c.2).1 (by simpa using this)
      obtain ⟨idx, v, hi, hv, hm⟩ := h2
      exact (nodeCond_iff g label c hlt id hid).2 ⟨lm, idx, hl, hi, hn, v, hv, hm⟩
    · rintro ⟨_, hall⟩
      obtain ⟨c0, hc0⟩ := List.exists_mem_of_ne_nil _ hne'
      obtain ⟨lm, idx, hl, _, hn, _⟩ := (nodeCond_iff g label c0 hlt id hid).1 (hall c0 hc0)
      refine ⟨⟨lm, hl, hn⟩, fun c hc => ?_⟩
      obtain ⟨lm', idx', hl', hi, _, v, hv, hm⟩ := (nodeCond_iff g label c hlt id hid).1 (hall c hc)
      have := (holds_iff (g.attrId c.1) (fun attr => g.nodeAttr id attr) c.2).2 ⟨idx', v, hi, hv, hm⟩
      simpa using this
  | any cs =>
    simp only [FilterHolds, ScanFilter.matches, List.any_eq_true, mem_getD_iff]
    constructor
    · rintro ⟨⟨lm, hl, hn⟩, c, hc, hh⟩
      have h2 := (holds_iff (g.attrId c.1) (fun attr => g.nodeAttr id attr) c.2).1 (by simpa using hh)
      obtain ⟨idx, v, hi, hv, hm⟩ := h2
      exact ⟨c, hc, (nodeCond_iff g label c hlt id hid).2 ⟨lm, idx, hl, hi, hn, v, hv, hm⟩⟩
    · rintro ⟨c, hc, hcond⟩
      obtain ⟨lm, idx, hl, hi, hn, v, hv, hm⟩ := (nodeCond_iff g label c hlt id hid).1 hcond
      refine ⟨⟨lm, hl, hn⟩, c, hc, ?_⟩
      have := (holds_iff (g.attrId c.1) (fun attr => g.nodeAttr id attr) c.2).2 ⟨idx, v, hi, hv, hm⟩
      simpa using this

/-- The edge scan: an edge id is scanned iff the built columns answer it (same statement, edges). -/
theorem edgeFilteredScan_agrees (g : G) (typeName : String) (f : ScanFilter String)
    (hne : ∀ cs, f = .all cs → cs ≠ [])
    (hlt : ∀ t, g.typeEdges typeName = some t → ∀ e ∈ t, e.2.2 < W) (id : Nat) (hid : id < W) :
    (∃ s d, (s, d, id) ∈ edgeFilteredScan g typeName f) ↔ FilterHolds f (fun c => EdgeCond g typeName c id) := by
  have hP := matches_resolve f g.attrId (fun attr => g.relAttr id attr) hne
  have hmem : (∃ s d, (s, d, id) ∈ edgeFilteredScan g typeName f) ↔
      id ∈ ((g.typeEdges typeName).getD []).map (·.2.2) ∧
        f.matches (fun a => (g.attrId a).bind (fun attr => g.relAttr id attr)) = true := by
    unfold edgeFilteredScan
    rw [← hP]
    cases hr : f.resolve g.attrId with
    | none => simp
    | some rf =>
      simp only [List.mem_filter, G.relationshipMatches, List.mem_map]
      constructor
      · rintro ⟨s, d, hm, hh⟩; exact ⟨⟨(s, d, id), hm, rfl⟩, hh⟩
      · rintro ⟨⟨⟨s, d, e⟩, he, hee⟩, hh⟩
        simp only at hee; subst hee
        exact ⟨s, d, he, hh⟩
  rw [hmem]
  cases f with
  | all cs =>
    have hne' := hne cs rfl
    simp only [FilterHolds, ScanFilter.matches, List.all_eq_true]
    constructor
    · rintro ⟨hn, hall⟩
      refine ⟨hne', fun c hc => ?_⟩
      have h2 := (holds_iff (g.attrId c.1) (fun attr => g.relAttr id attr) c.2).1 (by simpa using hall c hc)
      obtain ⟨idx, v, hi, hv, hm⟩ := h2
      exact (edgeCond_iff g typeName c hlt id hid).2 ⟨idx, hi, hn, v, hv, hm⟩
    · rintro ⟨_, hall⟩
      obtain ⟨c0, hc0⟩ := List.exists_mem_of_ne_nil _ hne'
      obtain ⟨_, _, hn, _⟩ := (edgeCond_iff g typeName c0 hlt id hid).1 (hall c0 hc0)
      refine ⟨hn, fun c hc => ?_⟩
      obtain ⟨idx', hi, _, v, hv, hm⟩ := (edgeCond_iff g typeName c hlt id hid).1 (hall c hc)
      have := (holds_iff (g.attrId c.1) (fun attr => g.relAttr id attr) c.2).2 ⟨idx', v, hi, hv, hm⟩
      simpa using this
  | any cs =>
    simp only [FilterHolds, ScanFilter.matches, List.any_eq_true]
    constructor
    · rintro ⟨hn, c, hc, hh⟩
      obtain ⟨idx, v, hi, hv, hm⟩ := (holds_iff (g.attrId c.1) (fun attr => g.relAttr id attr) c.2).1 (by simpa using hh)
      exact ⟨c, hc, (edgeCond_iff g typeName c hlt id hid).2 ⟨idx, hi, hn, v, hv, hm⟩⟩
    · rintro ⟨c, hc, hcond⟩
      obtain ⟨idx, hi, hn, v, hv, hm⟩ := (edgeCond_iff g typeName c hlt id hid).1 hcond
      refine ⟨hn, c, hc, ?_⟩
      have := (holds_iff (g.attrId c.1) (fun attr => g.relAttr id attr) c.2).2 ⟨idx, v, hi, hv, hm⟩
      simpa using this

/-- **A building column's read, end to end (nodes, one attribute)**: if `query_column` answers a query on
    attribute `a` with a `ScanFilter` (the column is still building), the scan the op runs returns exactly
    the ids the column `CREATE INDEX` would build answers the query's selection with once it is built. -/
theorem nodeScan_building_agrees (g : G) (label : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (sel : Selection) (hs : Selection.of q = some sel)
    (f : ScanFilter String) (hq : queryIndexNumericNodes g label q = some (.error f))
    (hlt : ∀ lm, g.labelNodes label = some lm → ∀ n ∈ lm, n < W) (id : Nat) (hid : id < W) :
    id ∈ nodeFilteredScan g label f ↔
      id ∈ (NumericIndex.fromEntries (collectNodeEntries g label a)).lookupSel sel := by
  rw [queryIndexNumericNodes_spec] at hq
  obtain ⟨_, _, _, hf⟩ := queryColumn_error_single _ _ _ _ a hat sel hs f hq
  subst hf
  rw [nodeFilteredScan_agrees g label _ (by intro cs h; cases h; simp) hlt id hid]
  simp [FilterHolds, ScanFilter.one, NodeCond]

/-- The same for edges. -/
theorem edgeScan_building_agrees (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery)
    (a : String) (hat : attrsOf q = some [a]) (sel : Selection) (hs : Selection.of q = some sel)
    (f : ScanFilter String) (hq : queryIndexNumericEdges g ep t q = some (.error f))
    (hlt : ∀ tt, g.typeEdges t = some tt → ∀ e ∈ tt, e.2.2 < W) (id : Nat) (hid : id < W) :
    (∃ s d, (s, d, id) ∈ edgeFilteredScan g t f) ↔
      id ∈ (NumericIndex.fromEntries (collectEdgeEntries g t a)).lookupSel sel := by
  rw [queryIndexNumericEdges_scan] at hq
  obtain ⟨_, _, _, hf⟩ := queryColumn_error_single _ _ _ _ a hat sel hs f hq
  subst hf
  rw [edgeFilteredScan_agrees g t _ (by intro cs h; cases h; simp) hlt id hid]
  simp [FilterHolds, ScanFilter.one, EdgeCond]

/-! ## The scan ops under `index-falkordb` -/

/-- `NodeByIndexScanOp::next`'s source of nodes for one row (`node_by_index_scan.rs`):
    `can_utilize_index` false → the label scan; else rows, the filtered scan, or the error (`none`). -/
def nodeScanOp (g : G) (utilize : Bool) (label : String) (q : IndexQuery) : Option (List Nat) :=
  scanOp utilize (queryIndexNumericNodes g label q) ((g.labelNodes label).getD []) (nodeFilteredScan g label)

/-- `EdgeByIndexScanOp::next`'s source of edges (`edge_by_index_scan.rs`). -/
def edgeScanOp (g : G) (ep : Nat → Option (Nat × Nat)) (utilize : Bool) (t : String) (q : IndexQuery) :
    Option (List (Nat × Nat × Nat)) :=
  scanOp utilize (queryIndexNumericEdges g ep t q) ((g.typeEdges t).getD []) (edgeFilteredScan g t)

/-- A `Range` on an existing column is never an error and never read from RediSearch (there is none):
    the node op returns rows or the filtered scan whatever the column's build state. -/
theorem nodeScanOp_range_served (g : G) (label key : String) (mn mx : Option Value) (a b : Bool)
    (en : IndexEntry) (hen : g.idx.entry .node label key = some en)
    (hmn : ∀ v, mn = some v → isNumericKind v = true ∨ v = .null)
    (hmx : ∀ v, mx = some v → isNumericKind v = true ∨ v = .null) :
    (nodeScanOp g true label (.range key mn mx a b)).isSome = true := by
  unfold nodeScanOp
  have h := queryColumn_range_served g.idx .node label key mn mx a b en hen hmn hmx
  rw [← queryIndexNumericNodes_spec] at h
  cases hq : queryIndexNumericNodes g label (.range key mn mx a b) with
  | none => simp [hq] at h
  | some a' => cases a' <;> simp [scanOp]

theorem edgeScanOp_range_served (g : G) (ep : Nat → Option (Nat × Nat)) (t key : String) (mn mx : Option Value)
    (a b : Bool) (en : IndexEntry) (hen : g.idx.entry .relationship t key = some en)
    (hmn : ∀ v, mn = some v → isNumericKind v = true ∨ v = .null)
    (hmx : ∀ v, mx = some v → isNumericKind v = true ∨ v = .null) :
    (edgeScanOp g ep true t (.range key mn mx a b)).isSome = true := by
  unfold edgeScanOp
  have h := queryColumn_range_served g.idx .relationship t key mn mx a b en hen hmn hmx
  have hn : queryIndexNumericEdges g ep t (.range key mn mx a b) ≠ none := by
    rw [Ne, queryIndexNumericEdges_none]
    intro h0; rw [h0] at h; simp at h
  cases hq : queryIndexNumericEdges g ep t (.range key mn mx a b) with
  | none => exact absurd hq hn
  | some a' => cases a' <;> simp [scanOp]

/-- The op errors (`unsupported_by_native_index`) exactly when the native index declines the predicate
    (no column, or a shape no kind serves): never because a column is building. -/
theorem nodeScanOp_error_iff (g : G) (label : String) (q : IndexQuery) :
    nodeScanOp g true label q = none ↔ g.idx.queryColumn .node label q = none := by
  unfold nodeScanOp
  rw [scanOp_none_iff, queryIndexNumericNodes_spec]

theorem edgeScanOp_error_iff (g : G) (ep : Nat → Option (Nat × Nat)) (t : String) (q : IndexQuery) :
    edgeScanOp g ep true t q = none ↔ g.idx.queryColumn .relationship t q = none := by
  unfold edgeScanOp
  rw [scanOp_none_iff, queryIndexNumericEdges_none]

/-! ## `redisearch_fields` (`graph.rs`) -/

/-- `redisearch_fields` without `index-falkordb` (`graph.rs`): the identity, RediSearch owns every
    kind. -/
def redisearchFieldsAll (fields : List (String × List Field)) : List (String × List Field) := fields

/-- `redisearch_fields` with `index-falkordb` (`graph.rs`): per key, drop the Range fields
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
    · simp only [he, Bool.false_eq_true, ite_false, Option.some.injEq, Prod.mk.injEq] at h
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
    · simp only [he, ite_true]; exact ih
    · simp only [he, Bool.false_eq_true, ite_false, List.filterMap_cons]
      have hf : ((l.filter (fun f => !(f.ty == .range))).filter (fun f => !(f.ty == .range))) =
          l.filter (fun f => !(f.ty == .range)) := by
        rw [List.filter_filter]; congr 1; funext f; simp
      simp only [hf, he, Bool.false_eq_true, ite_false, ih]

/-- **Range is the native index's alone**: an attribute with a Range field is built natively by
    `populate` (from every field, `native_attrs`), and the Range fields are never fed to RediSearch (an attribute
    that also has a Fulltext or Vector field still has those fed: `redisearchFields_keeps_nonrange`). -/
theorem native_owns_range (g : G) (e : EntityType) (label : String) (a : String) (fs : List Field)
    (fields : List (String × List Field)) (hr : fs.any (fun f => f.ty == .range) = true) :
    populate g e label [(a, fs)] = buildInOneBatch g e label a ∧
      ∀ l, (a, l) ∈ redisearchFields fields → ∀ f ∈ l, f.ty ≠ .range :=
  ⟨populate_builds g e label (a, fs) hr, fun l h => redisearchFields_no_range fields a l h⟩

end NI
