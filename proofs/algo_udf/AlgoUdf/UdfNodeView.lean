import AlgoUdf.UdfClasses
/-! # Where a UDF node object reads its labels and attributes from (#3454) — FIXED by #3464

`UdfClasses.jsNode` takes the labels and attributes as arguments, so the question
*which view of the graph* `create_js_node` reads them from is modelled here. Before #3464
`create_js_node` read only the committed graph, so a node deleted (or relabelled) earlier
in the same query looked different to a UDF than to `labels(n)` / `n.prop`. Now
`create_js_node(ctx, id, graph, runtime)` (`udf/js_classes.rs:134`) reads through the
runtime when `Runtime::node_differs_from_committed` says the answer can differ, and from
the committed graph otherwise. Rust source of truth: origin/main 1c0e6c8de.

| here | there |
| --- | --- |
| `NodeView`        | a node's labels/attributes as some reader sees them |
| `Pending`, `overlayLabels`, `overlayAttrs`, `Pending.has` | `Pending::update_node_labels` `runtime/pending.rs:778-793`, `update_node_attrs` :638-660, `node_has_changes` :755-775 (a `None` entry is an absent map key) |
| `RT`, `RT.labels`, `RT.attrs` | `Runtime::get_node_labels` `runtime/runtime.rs:1787-1799` and `get_node_attrs` :1866-1882: a node this query deleted answers from the snapshot taken at the delete (`deleted_nodes`), else committed + pending overlay |
| `RT.differs`      | `Runtime::node_differs_from_committed` :1772-1785: deleted this query, or `Pending::node_has_changes` |
| `setNodeLabels`, `setNodeAttrs` | `set_node_labels` `udf/js_classes.rs:203`, `set_node_attrs` :220 (shared by both branches; `jsNode_eq_helpers`) |
| `rsJsNode`        | `create_js_node` :134-199: `runtime.filter(differs)` ⇒ runtime view (:157-166), else the committed graph (:167-170) |
| `rsJsEdge`        | `create_js_edge` :246: `.source`/`.target` via `create_js_node(.., runtime)` (:289-290); `create_js_path` :322 passes the runtime to its nodes (:341) |
| `type_convert` call | `value_to_js_level` `udf/type_convert.rs:184`: `Value::Node` → `create_js_node(ctx, id, graph, runtime)` |
| `specJsNode`      | the contract: read the runtime when one is given, as `labels(n)` / `n.prop` do |
| `preRsJsNode`, `pre3454_*` | historical: the code before #3464, which never read the runtime |

Entry points with no runtime (`getNeighbors` :518, `getNodeById` :576, `iterateNodes` :630,
`traverse` :863) call `create_js_node(.., None)`: the committed graph, unchanged by #3464.

Live (before the fix; Rust vs C), `CREATE (:P {name:'A'})`, UDF `inspect_node(n) = [n.labels, n.attributes.name]`:

| query | Rust (pre-#3464) | C |
| --- | --- | --- |
| `MATCH (n:P) DELETE n RETURN labels(n), n.name, test.inspect_node(n)` | `[P], A, [[], NULL]` | `[], A, [[], A]` |
| `MATCH (n:P)-[r]->() DETACH DELETE n RETURN test2.src(r)` (`r.source`) | `[[], NULL]` | error (undefined attribute) |

C also loses deleted-node data (differently). The property stated is Rust's own
contract, the one `create_js_edge` already follows for the type: a UDF sees the node as
the query's runtime does, i.e. as `labels(n)` / `n.name` in the same row do.
-/
namespace AlgoUdf.UdfNodeView
open AlgoUdf.UdfClasses

/-- A view of node labels / attributes. -/
structure NodeView where
  labels : Nat → List String
  attrs : Nat → List (String × JV)

/-- This query's staged node changes (`None` = no entry for the node in the map).
`added` is `new_nodes_attrs.get(id).or_else(existing_nodes_attrs.get(id))`. -/
structure Pending where
  setL : Nat → Option (List String)
  removeL : Nat → Option (List String)
  added : Nat → Option (List (String × JV))

/-- `Pending::update_node_labels`: adds, then removals. -/
def overlayLabels (p : Pending) (id : Nat) (ls : List String) : List String :=
  let a := match p.setL id with | some s => ls ++ s.filter (· ∉ ls) | none => ls
  match p.removeL id with | some r => a.filter (· ∉ r) | none => a

/-- `Pending::update_node_attrs`: a `Null` removes the key, anything else sets it
(the position of a re-set key is not modelled). -/
def overlayAttrs (p : Pending) (id : Nat) (as : List (String × JV)) : List (String × JV) :=
  match p.added id with
  | some ad => ad.foldl (fun acc kv => match kv.2 with
      | .null => acc.filter (·.1 != kv.1)
      | v => acc.filter (·.1 != kv.1) ++ [(kv.1, v)]) as
  | none => as

/-- `Pending::node_has_changes`. -/
def Pending.has (p : Pending) (id : Nat) : Bool :=
  (p.setL id).isSome || (p.removeL id).isSome || (p.added id).isSome

/-- The query runtime over the committed graph `g`: delete-time snapshots and pending changes. -/
structure RT where
  g : NodeView
  deleted : Nat → Option (List String × List (String × JV))
  p : Pending

/-- `Runtime::get_node_labels` (runtime.rs:1787). -/
def RT.labels (r : RT) (id : Nat) : List String :=
  match r.deleted id with | some s => s.1 | none => overlayLabels r.p id (r.g.labels id)

/-- `Runtime::get_node_attrs` (runtime.rs:1866). -/
def RT.attrs (r : RT) (id : Nat) : List (String × JV) :=
  match r.deleted id with | some s => s.2 | none => overlayAttrs r.p id (r.g.attrs id)

/-- `Runtime::node_differs_from_committed` (runtime.rs:1772). -/
def RT.differs (r : RT) (id : Nat) : Bool := (r.deleted id).isSome || r.p.has id

/-- When the runtime reports no difference, the runtime view IS the committed graph:
the fast path reads exactly what `get_node_labels` / `get_node_attrs` would build. -/
theorem RT.not_differs (r : RT) (id : Nat) (h : r.differs id = false) :
    r.labels id = r.g.labels id ∧ r.attrs id = r.g.attrs id := by
  have hd : r.deleted id = none := by
    cases hh : r.deleted id <;> simp_all [RT.differs]
  have hs : r.p.setL id = none := by
    cases hh : r.p.setL id <;> simp_all [RT.differs, Pending.has]
  have hr : r.p.removeL id = none := by
    cases hh : r.p.removeL id <;> simp_all [RT.differs, Pending.has]
  have ha : r.p.added id = none := by
    cases hh : r.p.added id <;> simp_all [RT.differs, Pending.has]
  simp [RT.labels, RT.attrs, overlayLabels, overlayAttrs, hd, hs, hr, ha]

/-- `set_node_labels` (js_classes.rs:203): the `labels` array. -/
def setNodeLabels (o : Obj) (ls : List String) : Obj := set o "labels" (.arr (ls.map .str))

/-- `set_node_attrs` (js_classes.rs:220): attributes copied onto the node (unless reserved)
and collected in `attributes`. -/
def setNodeAttrs (o : Obj) (attrs : List (String × JV)) : Obj :=
  let r := attrLoop (o, []) attrs
  set r.1 "attributes" (.obj r.2)

/-- `create_js_node` = markers, `set_node_labels`, `set_node_attrs`, `getNeighbors`: the
refactor into helpers builds the same object as `UdfClasses.jsNode`. -/
theorem jsNode_eq_helpers (id : Nat) (ls : List String) (attrs : List (String × JV)) :
    jsNode id ls attrs =
      set (setNodeAttrs (setNodeLabels
        (set (set (set [] "__falkor_type" (.str "node")) "__falkor_node_id" (.num id)) "id" (.num id))
          ls) attrs) "getNeighbors" .fn := rfl

/-- `create_js_node` on main (#3464): through the runtime iff it is given and
`node_differs_from_committed`, else the committed graph (js_classes.rs:157-170). -/
def rsJsNode (rt : Option RT) (g : NodeView) (id : Nat) : Obj :=
  match rt with
  | some r =>
      if r.differs id then jsNode id (r.labels id) (r.attrs id)
      else jsNode id (g.labels id) (g.attrs id)
  | none => jsNode id (g.labels id) (g.attrs id)

/-- The contract: read the runtime when one is given, committed graph only without one. -/
def specJsNode (rt : Option RT) (g : NodeView) (id : Nat) : Obj :=
  match rt with
  | some r => jsNode id (r.labels id) (r.attrs id)
  | none => jsNode id (g.labels id) (g.attrs id)

/-- `create_js_edge` on main: `.source` / `.target` from `create_js_node(.., runtime)` (:289-290). -/
def rsJsEdge (rt : Option RT) (g : NodeView) (id src dst : Nat) (ty : String)
    (attrs : List (String × JV)) : Obj :=
  jsEdge id src dst ty (rsJsNode rt g src) (rsJsNode rt g dst) attrs

def specJsEdge (rt : Option RT) (g : NodeView) (id src dst : Nat) (ty : String)
    (attrs : List (String × JV)) : Obj :=
  jsEdge id src dst ty (specJsNode rt g src) (specJsNode rt g dst) attrs

/-- **Property `RuntimeFirstNode`**: when the query's runtime is available (over the
committed graph the node is built from), the JS node carries exactly the labels and
attributes the runtime reports (what `labels(n)` and `n.prop` evaluate to in the same row). -/
def RuntimeFirstNode (mk : Option RT → NodeView → Nat → Obj) : Prop :=
  ∀ r id, mk (some r) r.g id = jsNode id (r.labels id) (r.attrs id)

/-- Edge endpoints follow the same rule. -/
def RuntimeFirstEdgeEnds
    (mk : Option RT → NodeView → Nat → Nat → Nat → String → List (String × JV) → Obj) : Prop :=
  ∀ r id s d ty attrs,
    get (mk (some r) r.g id s d ty attrs) "source" = some (.obj (jsNode s (r.labels s) (r.attrs s)))

theorem spec_runtime_first : RuntimeFirstNode specJsNode := fun _ _ => rfl

/-- **#3464 fixes #3454**: `create_js_node` on main satisfies `RuntimeFirstNode`,
including through the `node_differs_from_committed` fast path. -/
theorem rs_runtime_first : RuntimeFirstNode rsJsNode := by
  intro r id
  unfold rsJsNode
  by_cases h : r.differs id = true
  · simp [h]
  · have h' : r.differs id = false := by simpa using h
    obtain ⟨hl, ha⟩ := r.not_differs id h'
    simp [h', hl, ha]

/-- Without a runtime the two agree (the committed graph is all there is). -/
theorem rs_eq_spec_no_runtime (g : NodeView) (id : Nat) : rsJsNode none g id = specJsNode none g id := rfl

/-- The fixed node builder equals the contract whenever the runtime is over `g`. -/
theorem rs_eq_spec (r : RT) (id : Nat) : rsJsNode (some r) r.g id = specJsNode (some r) r.g id :=
  rs_runtime_first r id

theorem jsNode_labels (id : Nat) (ls : List String) (attrs : List (String × JV)) :
    get (jsNode id ls attrs) "labels" = some (.arr (ls.map .str)) := by
  unfold jsNode
  simp only
  rw [get_set_ne _ _ _ _ (by decide), get_set_ne _ _ _ _ (by decide), attrLoop_keeps _ (by decide)]
  rw [get_set_same _ _ _ (by decide)]

theorem jsEdge_source (id s d : Nat) (ty : String) (so dob : Obj) (attrs : List (String × JV)) :
    get (jsEdge id s d ty so dob attrs) "source" = some (.obj so) := by
  unfold jsEdge
  simp only
  rw [get_set_ne _ _ _ _ (by decide), get_set_ne _ _ _ _ (by decide), get_set_same _ _ _ (by decide)]

theorem spec_edge_ok : RuntimeFirstEdgeEnds specJsEdge := by
  intro rt id s d ty attrs
  exact jsEdge_source _ _ _ _ _ _ _

/-- **#3464**: edge endpoints on main satisfy `RuntimeFirstEdgeEnds`
(`r.source` after `DETACH DELETE` sees the snapshot). -/
theorem rs_edge_ok : RuntimeFirstEdgeEnds rsJsEdge := by
  intro r id s d ty attrs
  unfold rsJsEdge
  rw [jsEdge_source, rs_runtime_first]

/-! ## The scenario: node 0 `(:P {name:'A'})` deleted by this query -/

/-- The committed graph after the delete (the `Commit` before the projection): node 0 is gone. -/
def committedAfterDelete : NodeView := ⟨fun _ => [], fun _ => []⟩

/-- The runtime: node 0 answers from its delete-time snapshot. -/
def rtAfterDelete : RT where
  g := committedAfterDelete
  deleted := fun id => if id = 0 then some (["P"], [("name", .attr 0)]) else none
  p := ⟨fun _ => none, fun _ => none, fun _ => none⟩

/-- `labels(n)` in the same row (runtime): `[P]`. -/
theorem builtin_sees_snapshot : rtAfterDelete.labels 0 = ["P"] := rfl

/-- **#3454 fixed** (`MATCH (n:P) DELETE n RETURN labels(n), test.inspect_node(n)`): the UDF
node object now has `labels = [P]`, as `labels(n)` does. -/
theorem fixed3454_deleted_node :
    get (rsJsNode (some rtAfterDelete) committedAfterDelete 0) "labels" = some (.arr [.str "P"]) := by
  rw [show committedAfterDelete = rtAfterDelete.g from rfl, rs_runtime_first]
  exact jsNode_labels _ _ _

/-- **#3454 fixed** (`... DETACH DELETE n RETURN test2.src(r)`): `r.source.labels = [P]`. -/
theorem fixed3454_edge_source :
    get (rsJsEdge (some rtAfterDelete) committedAfterDelete 5 0 1 "R" []) "source" =
      some (.obj (jsNode 0 ["P"] [("name", .attr 0)])) := by
  rw [show committedAfterDelete = rtAfterDelete.g from rfl, rs_edge_ok]
  rfl

/-! ## Historical: `create_js_node` before #3464 (fixed by #3464) -/

/-- Pre-#3464: the runtime argument did not exist; the committed graph `g` was read. -/
def preRsJsNode (_rt : Option RT) (g : NodeView) (id : Nat) : Obj :=
  jsNode id (g.labels id) (g.attrs id)

def preRsJsEdge (rt : Option RT) (g : NodeView) (id src dst : Nat) (ty : String)
    (attrs : List (String × JV)) : Obj :=
  jsEdge id src dst ty (preRsJsNode rt g src) (preRsJsNode rt g dst) attrs

/-- Pre-#3464 BUG #3454: the UDF node object had `labels = []` while `labels(n)` was `[P]`;
`create_js_node` violated `RuntimeFirstNode`. -/
theorem pre3454_deleted_node_empty :
    get (preRsJsNode (some rtAfterDelete) committedAfterDelete 0) "labels" = some (.arr []) ∧
    get (specJsNode (some rtAfterDelete) committedAfterDelete 0) "labels" = some (.arr [.str "P"]) ∧
    ¬ RuntimeFirstNode preRsJsNode := by
  refine ⟨jsNode_labels _ _ _, jsNode_labels _ _ _, fun h => ?_⟩
  have h1 := congrArg (get · "labels") (h rtAfterDelete 0)
  simp only [preRsJsNode, jsNode_labels] at h1
  injection h1 with h1
  injection h1 with h1
  cases h1

/-- Pre-#3464 BUG #3454: `r.source.labels = []` after `DETACH DELETE`. -/
theorem pre3454_edge_source_empty :
    get (preRsJsEdge (some rtAfterDelete) committedAfterDelete 5 0 1 "R" []) "source" =
      some (.obj (jsNode 0 [] [])) ∧
    ¬ RuntimeFirstEdgeEnds preRsJsEdge := by
  refine ⟨jsEdge_source _ _ _ _ _ _ _, fun h => ?_⟩
  have h1 := h rtAfterDelete 5 0 1 "R" []
  rw [preRsJsEdge, jsEdge_source] at h1
  injection h1 with h1
  injection h1 with h1
  have h2 := congrArg (get · "labels") h1
  simp only [preRsJsNode, jsNode_labels] at h2
  injection h2 with h2
  injection h2 with h2
  cases h2

end AlgoUdf.UdfNodeView
