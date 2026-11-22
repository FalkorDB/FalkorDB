import GraphQueries.Labels
/-
# Index glue, memory report, payload dispatch, plan cache

The `Indexer` (RediSearch-backed, proofs/index_layer) is a parameter:
`IX` records the answers the graph code consumes. Every theorem below holds
for *any* indexer behaviour.

| here | there (graph.rs) |
| --- | --- |
| `populateStart`   | `populate_index` :560 |
| `dropBg`          | `drop_index_bg` :795 (`Indexer::remove_if_empty`, #3856) |
| `createIndex`     | `create_index` :3670 / `create_index_sync` :3717 |
| `syncDocs`        | `populate_indexes_sync` :3764 (one label/type) |
| `commitDocs`      | `commit_index_kind` :4043 (via `commit_index` :3927, `commit_index_locked` :3940) |
| `commitEdgeDocs`  | `commit_edge_index_locked` :3962 (via `commit_edge_index` :3950) |
| `commitIndexDocs` | `commit_index_docs` :3913 (native apply, then `commit_index`/`commit_edge_index` per non-empty kind) |
| `dropIndex`, `noSuchIndexMsg` | `drop_index` :4090 (message :4160-4177) |
| `indexInfo`       | `index_info` :4468 |
| `memEstimate`, `labelPass` | `memory_usage_report` :4919 sampling arithmetic and seen-dedup |
| `encodeOp`        | `encode_payload` :5093 dispatch |
| `getPlan`         | `get_plan` :1383 cache decision |
-/
namespace GQ
variable {V : Type}

/-- `populate_index`: no snapshot → nothing; else batch 0 at the default cursor. -/
def populateStart (snap : Option Nat) : Option (Nat × Nat × Option Nat × Option Nat) :=
  snap.map (fun _ => (0, 0, none, none))
theorem populateStart_spec (t : Nat) : populateStart (some t) = some (0, 0, none, none) ∧
    populateStart none = none := ⟨rfl, rfl⟩

/-- `drop_index_bg` (under the indexer lock): `remove_if_empty` drops the
label's entry only while it has no fields (#3856). The indexer map is
`(label, index_count)` pairs. -/
def dropBg (idx : List (String × Nat)) (l : String) : List (String × Nat) :=
  idx.filter (fun e => !(e.1 == l && e.2 == 0))

/-- **PROVEN**: no fieldless entry for the label is left. -/
theorem dropBg_spec (idx : List (String × Nat)) (l : String) :
    ∀ e ∈ dropBg idx l, e.1 = l → e.2 ≠ 0 := by
  intro e he h1 h2; simp [dropBg, h1, h2] at he

/-- **PROVEN**: on the drop path with no racing `CREATE INDEX` (`drop_index`
reported `remaining = 0`), the label leaves the indexer's map. -/
theorem dropBg_after_drop (idx : List (String × Nat)) (l : String)
    (h : ∀ n, (l, n) ∈ idx → n = 0) : l ∉ (dropBg idx l).map (·.1) := by
  simp only [dropBg, List.mem_map, List.mem_filter, not_exists, not_and]
  rintro ⟨l', n⟩ ⟨hm, hb⟩ rfl
  simp [h n hm] at hb

/-- **PROVEN** (#3856): an entry a `CREATE INDEX` re-filled before the
background drop ran is kept. -/
theorem dropBg_keeps_filled (idx : List (String × Nat)) (l : String) (n : Nat)
    (hm : (l, n) ∈ idx) (hn : n ≠ 0) : (l, n) ∈ dropBg idx l := by
  simp [dropBg, hm, hn]

/-- **PROVEN**: other labels are untouched. -/
theorem dropBg_other (idx : List (String × Nat)) (l l' : String) (n : Nat) (hl : l' ≠ l) :
    (l', n) ∈ dropBg idx l ↔ (l', n) ∈ idx := by
  simp [dropBg, hl]

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

/-- `commit_index_docs` (graph.rs): first `apply_index_writes` (the native index's
staged `docs.native`, a no-op without `index-falkordb`; its semantics belong to
proofs/index_layer), then the RediSearch node documents iff `node_adds`/`node_removes` is
non-empty, then the edge documents iff `edge_adds`/`edge_removes` is non-empty. The ops
are the calls made, in order. -/
def commitIndexDocs (nodeAddsEmpty nodeRemsEmpty edgeAddsEmpty edgeRemsEmpty : Bool) : List String :=
  ["apply_index_writes"] ++
  (if nodeAddsEmpty && nodeRemsEmpty then [] else ["commit_index"]) ++
  (if edgeAddsEmpty && edgeRemsEmpty then [] else ["commit_edge_index"])

/-- A kind with no documents is skipped before its indexer lock is taken. -/
theorem commitIndexDocs_skip (a b c d : Bool) :
    ("commit_index" ∈ commitIndexDocs a b c d ↔ ¬(a && b)) ∧
    ("commit_edge_index" ∈ commitIndexDocs a b c d ↔ ¬(c && d)) := by
  cases a <;> cases b <;> cases c <;> cases d <;> decide

/-- The native part is applied first, before either RediSearch publish. -/
theorem commitIndexDocs_native_first (a b c d : Bool) :
    (commitIndexDocs a b c d).head? = some "apply_index_writes" := by
  simp [commitIndexDocs]

/-- Nothing queued: only the (no-op on empty) native apply runs, no indexer is called. -/
theorem commitIndexDocs_empty : commitIndexDocs true true true true = ["apply_index_writes"] := rfl

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

/-- The "no such index" text (graph.rs): `:L` when no attribute is named
(`db.idx.fulltext.drop('L')`, as C), else `:L(a, b)`. -/
def noSuchIndexMsg (label : String) (attrs : List String) : String :=
  let attrList := ", ".intercalate attrs
  let a := if attrList = "" then "" else "(" ++ attrList ++ ")"
  "Unable to drop index on :" ++ label ++ a ++ ": no such index."

/-- `drop_index`: empty attrs expand to the index type's fields; a UNIQUE
constraint over any of them blocks the drop; otherwise the indexer decides. -/
def dropIndex (dependsOn : String → Bool) (fieldsOfType : List String) (label : String)
    (attrs : List String) (dropped : Nat) : Except String Nat :=
  let eff := if attrs = [] then fieldsOfType else attrs
  if eff.any dependsOn then .error "Index supports constraint"
  else if dropped > 0 then .ok dropped else .error (noSuchIndexMsg label attrs)

theorem dropIndex_protects (dependsOn : String → Bool) (fieldsOfType : List String) (label : String)
    (attrs : List String) (dropped : Nat)
    (a : String) (ha : a ∈ (if attrs = [] then fieldsOfType else attrs)) (hd : dependsOn a = true) :
    dropIndex dependsOn fieldsOfType label attrs dropped = .error "Index supports constraint" := by
  unfold dropIndex
  rw [ite_eq_left (List.any_eq_true.2 ⟨a, ha, hd⟩)]

/-- Dropping an index that does not exist (no constraint in the way)
reports `Unable to drop index on :L: no such index.` when no attribute was named, as C. -/
theorem dropIndex_none_noattr (dependsOn : String → Bool) (fieldsOfType : List String)
    (label : String) (h : fieldsOfType.any dependsOn = false) :
    dropIndex dependsOn fieldsOfType label [] 0 =
      .error ("Unable to drop index on :" ++ label ++ ": no such index.") := by
  simp [dropIndex, h, noSuchIndexMsg]

/-- With attributes named, the message lists them all: `:L(a, b)`. -/
example : noSuchIndexMsg "L" ["a", "b"] = "Unable to drop index on :L(a, b): no such index." := by
  decide +kernel

/-- `index_info`: node infos tagged `NODE`, then edge infos tagged `RELATIONSHIP`. -/
def indexInfo {I} (nodeInfos edgeInfos : List I) : List (String × I) :=
  nodeInfos.map ("NODE", ·) ++ edgeInfos.map ("RELATIONSHIP", ·)
theorem indexInfo_length {I} (a b : List I) : (indexInfo a b).length = a.length + b.length := by
  simp [indexInfo]

/-! ## `add_native_build` / `native_build_total` (`index-falkordb` only, #3585)

Both are `#[cfg(feature = "index-falkordb")]` and run only from `index_info`: under the feature a
label with a `Building` native Range column reports that build's progress. The column's own state
is `FalkorDbIndex::building_read` (proofs/native_index), taken here as the per-attribute answer
`Option Nat` (`none` = not a `Building` column). -/

/-- The `IndexInfo` fields `add_native_build` reads and writes: `pending: i32`,
`progress: u64`, `total: u64` (graph/src/index/mod.rs). -/
structure BuildInfo where
  pending : Int
  progress : Nat
  total : Nat

/-- `native_build_total` (graph.rs): the label's `nvals` / the type's `edge_count`; a name
without a matrix counts 0. -/
def nativeBuildTotal (g : G V) (isNode : Bool) (label : String) : Nat :=
  if isNode then ((getLabelMat g label).map (·.ents.length)).getD 0
  else ((getRelMat g label).map Ten.edgeCount).getD 0

/-- One turn of `add_native_build`'s loop over `field_order`: the pair is
`(is a Range field, building_read)`; the accumulator is `(columns, least read)`. -/
def nbStep (acc : Option (Int × Nat)) (a : Bool × Option Nat) : Option (Int × Nat) :=
  if a.1 then
    match a.2 with
    | some read => some (match acc with | none => (1, read) | some (columns, least) => (columns + 1, min least read))
    | none => acc
  else acc

def nbFold (attrs : List (Bool × Option Nat)) : Option (Int × Nat) := attrs.foldl nbStep none

/-- `add_native_build` (graph.rs). -/
def addNativeBuild (g : G V) (isNode : Bool) (label : String) (attrs : List (Bool × Option Nat))
    (info : BuildInfo) : BuildInfo :=
  match nbFold attrs with
  | none => info
  | some (columns, least) =>
    let total := nativeBuildTotal g isNode label
    let done := min least total
    { pending := info.pending + columns,
      progress := if info.pending > 0 then min done info.progress else done,
      total := total }

/-- Fields that are building Range columns. -/
def nbCount (attrs : List (Bool × Option Nat)) : Nat :=
  (attrs.filter (fun a => a.1 && a.2.isSome)).length

theorem nbCount_cons (r : Bool) (rd : Option Nat) (l : List (Bool × Option Nat)) :
    nbCount ((r, rd) :: l) = nbCount l + (if r && rd.isSome then 1 else 0) := by
  simp only [nbCount, List.filter_cons]
  by_cases h : (r && rd.isSome) = true <;> simp [h] <;> omega

theorem nbFold_cols : ∀ (attrs : List (Bool × Option Nat)) (acc : Option (Int × Nat)),
    (attrs.foldl nbStep acc).map (·.1) =
      (match acc with
       | none => if nbCount attrs = 0 then none else some (nbCount attrs : Int)
       | some (c, _) => some (c + nbCount attrs))
  | [], acc => by cases acc <;> simp [nbCount]
  | a :: l, acc => by
    obtain ⟨r, rd⟩ := a
    have ih := nbFold_cols l (nbStep acc (r, rd))
    simp only [List.foldl_cons, ih, nbCount_cons]
    cases r <;> cases rd <;> cases acc <;> simp [nbStep] <;> omega

/-- No building Range column: the info is untouched (the label is not under native construction). -/
theorem addNativeBuild_none (g : G V) (isNode : Bool) (label : String)
    (attrs : List (Bool × Option Nat)) (info : BuildInfo) (h : nbCount attrs = 0) :
    addNativeBuild g isNode label attrs info = info := by
  have := nbFold_cols attrs none
  have hn : nbFold attrs = none := by
    unfold nbFold
    cases hf : attrs.foldl nbStep none with
    | none => rfl
    | some x => rw [hf] at this; simp [h] at this
  simp [addNativeBuild, hn]

/-- Each building Range column counts as one pending index, and the label's total is reported once. -/
theorem addNativeBuild_pending (g : G V) (isNode : Bool) (label : String)
    (attrs : List (Bool × Option Nat)) (info : BuildInfo) (h : 0 < nbCount attrs) :
    (addNativeBuild g isNode label attrs info).pending = info.pending + nbCount attrs ∧
    (addNativeBuild g isNode label attrs info).total = nativeBuildTotal g isNode label := by
  have := nbFold_cols attrs none
  have hne : ¬ nbCount attrs = 0 := by omega
  cases hf : attrs.foldl nbStep none with
  | none => rw [hf] at this; simp [hne] at this
  | some x =>
    obtain ⟨c, l⟩ := x
    rw [hf] at this
    simp [hne] at this
    simp only [addNativeBuild, nbFold, hf]
    simp [this]

/-- The reported progress never exceeds the label's entities (so `m/n` shows `m ≤ n`). -/
theorem addNativeBuild_progress_le (g : G V) (isNode : Bool) (label : String)
    (attrs : List (Bool × Option Nat)) (info : BuildInfo) (h : 0 < nbCount attrs) :
    (addNativeBuild g isNode label attrs info).progress ≤ nativeBuildTotal g isNode label := by
  have := nbFold_cols attrs none
  have hne : ¬ nbCount attrs = 0 := by omega
  cases hf : attrs.foldl nbStep none with
  | none => rw [hf] at this; simp [hne] at this
  | some x =>
    obtain ⟨c, l⟩ := x
    simp only [addNativeBuild, nbFold, hf]
    split <;> omega

/-- While RediSearch has a field pending, the progress shown is the smaller of the two. -/
theorem addNativeBuild_progress_min (g : G V) (isNode : Bool) (label : String)
    (attrs : List (Bool × Option Nat)) (info : BuildInfo) (h : 0 < nbCount attrs)
    (hp : 0 < info.pending) :
    (addNativeBuild g isNode label attrs info).progress ≤ info.progress := by
  have := nbFold_cols attrs none
  have hne : ¬ nbCount attrs = 0 := by omega
  cases hf : attrs.foldl nbStep none with
  | none => rw [hf] at this; simp [hne] at this
  | some x =>
    obtain ⟨c, l⟩ := x
    simp only [addNativeBuild, nbFold, hf]
    simp [hp]
    omega

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
