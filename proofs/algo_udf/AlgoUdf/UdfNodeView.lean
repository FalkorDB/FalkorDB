import AlgoUdf.UdfClasses
/-! # Where a UDF node object reads its labels and attributes from (#3454) — gap closed 2026-10-22

`UdfClasses.jsNode` takes the labels and attributes as arguments, so the question
*which view of the graph* `create_js_node` reads them from was abstracted away. Only the
edge type had a runtime-first model (`edgeType_runtime_first`). This file adds the
source view. Rust source of truth: origin/main b84473a7f.

| here | there |
| --- | --- |
| `NodeView`        | a node's labels/attributes as some reader sees them |
| `runtimeView`     | `Runtime::get_node_labels` `runtime/runtime.rs:1666-1678` and `get_node_attrs` :1745-1760: a node this query deleted answers from the snapshot taken at the delete (`deleted_nodes`, :1671 / :1749), else committed + pending |
| `rsJsNode`        | `create_js_node` `udf/js_classes.rs:129-210`: takes only the committed `graph` (:132) and reads `g.get_node_labels` :151 / `g.get_node_all_attrs` :161; `rt` is never passed |
| `rsJsEdge`        | `create_js_edge` :213-286: type and attributes runtime-first (:242-252, :266), but `.source`/`.target` built by `create_js_node(ctx, src_id, graph)` :257-258 (no runtime) |
| `type_convert` call | `value_to_js_level` `udf/type_convert.rs:183-185`: `Value::Node` → `create_js_node(ctx, id, graph)`, dropping the `runtime` it has (forwarded at :191-198 / :201 for edges and paths) |
| `specJsNode`      | the fixed behaviour: read the runtime when one is given, as `labels(n)` / `n.prop` do |

Live (Rust main vs C), `CREATE (:P {name:'A'})`, UDF `inspect_node(n) = [n.labels, n.attributes.name]`:

| query | Rust | C |
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

/-- `Runtime::get_node_labels` / `get_node_attrs`: the delete-time snapshot for a node
this query deleted (runtime.rs:1671, :1749), else the live view (committed + pending). -/
def runtimeView (deleted : Nat → Option (List String × List (String × JV))) (live : NodeView) :
    NodeView where
  labels id := match deleted id with | some s => s.1 | none => live.labels id
  attrs id := match deleted id with | some s => s.2 | none => live.attrs id

/-- `create_js_node` on main: the runtime argument does not exist; the committed graph `g`
is read (js_classes.rs:151, :161). -/
def rsJsNode (_rt : Option NodeView) (g : NodeView) (id : Nat) : Obj :=
  jsNode id (g.labels id) (g.attrs id)

/-- The intended behaviour: runtime first, committed graph only without a runtime. -/
def specJsNode (rt : Option NodeView) (g : NodeView) (id : Nat) : Obj :=
  match rt with
  | some r => jsNode id (r.labels id) (r.attrs id)
  | none => jsNode id (g.labels id) (g.attrs id)

/-- `create_js_edge` on main: `.source` / `.target` from `create_js_node` (:257-258). -/
def rsJsEdge (rt : Option NodeView) (g : NodeView) (id src dst : Nat) (ty : String)
    (attrs : List (String × JV)) : Obj :=
  jsEdge id src dst ty (rsJsNode rt g src) (rsJsNode rt g dst) attrs

def specJsEdge (rt : Option NodeView) (g : NodeView) (id src dst : Nat) (ty : String)
    (attrs : List (String × JV)) : Obj :=
  jsEdge id src dst ty (specJsNode rt g src) (specJsNode rt g dst) attrs

/-- **Property `RuntimeFirstNode`**: when the query's runtime is available, the JS node
carries exactly the labels and attributes the runtime reports (what `labels(n)` and
`n.prop` evaluate to in the same row). -/
def RuntimeFirstNode (mk : Option NodeView → NodeView → Nat → Obj) : Prop :=
  ∀ rt g id, mk (some rt) g id = jsNode id (rt.labels id) (rt.attrs id)

/-- Edge endpoints follow the same rule. -/
def RuntimeFirstEdgeEnds
    (mk : Option NodeView → NodeView → Nat → Nat → Nat → String → List (String × JV) → Obj) : Prop :=
  ∀ rt g id s d ty attrs,
    get (mk (some rt) g id s d ty attrs) "source" = some (.obj (jsNode s (rt.labels s) (rt.attrs s)))

theorem spec_runtime_first : RuntimeFirstNode specJsNode := fun _ _ _ => rfl

/-- Without a runtime the two agree (the committed graph is all there is). -/
theorem rs_eq_spec_no_runtime (g : NodeView) (id : Nat) : rsJsNode none g id = specJsNode none g id := rfl

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
  intro rt g id s d ty attrs
  exact jsEdge_source _ _ _ _ _ _ _

/-! ## The counterexample: node 0 `(:P {name:'A'})` deleted by this query -/

/-- The committed graph after the delete (the `Commit` before the projection): node 0 is gone. -/
def committedAfterDelete : NodeView := ⟨fun _ => [], fun _ => []⟩

/-- The runtime: node 0 answers from its delete-time snapshot. -/
def rtAfterDelete : NodeView :=
  runtimeView (fun id => if id = 0 then some (["P"], [("name", .attr 0)]) else none) committedAfterDelete

/-- `labels(n)` in the same row (runtime): `[P]`. -/
theorem builtin_sees_snapshot : rtAfterDelete.labels 0 = ["P"] := rfl

/-- **BUG #3454** (`MATCH (n:P) DELETE n RETURN labels(n), test.inspect_node(n)`): the UDF
node object has `labels = []` while `labels(n)` is `[P]`; `create_js_node` violates
`RuntimeFirstNode`. -/
theorem bug3454_deleted_node_empty :
    get (rsJsNode (some rtAfterDelete) committedAfterDelete 0) "labels" = some (.arr []) ∧
    get (specJsNode (some rtAfterDelete) committedAfterDelete 0) "labels" = some (.arr [.str "P"]) ∧
    ¬ RuntimeFirstNode rsJsNode := by
  refine ⟨jsNode_labels _ _ _, jsNode_labels _ _ _, fun h => ?_⟩
  have h1 := congrArg (get · "labels") (h rtAfterDelete committedAfterDelete 0)
  simp only [rsJsNode, jsNode_labels] at h1
  injection h1 with h1
  injection h1 with h1
  cases h1

/-- **BUG #3454** (`... DETACH DELETE n RETURN test2.src(r)`): the edge is built with the
runtime (its type and endpoints come from the snapshot), but its `.source` node object is
built from the committed graph, so `r.source.labels = []`. -/
theorem bug3454_edge_source_empty :
    get (rsJsEdge (some rtAfterDelete) committedAfterDelete 5 0 1 "R" []) "source" =
      some (.obj (jsNode 0 [] [])) ∧
    ¬ RuntimeFirstEdgeEnds rsJsEdge := by
  refine ⟨jsEdge_source _ _ _ _ _ _ _, fun h => ?_⟩
  have h1 := h rtAfterDelete committedAfterDelete 5 0 1 "R" []
  rw [rsJsEdge, jsEdge_source] at h1
  injection h1 with h1
  injection h1 with h1
  have h2 := congrArg (get · "labels") h1
  simp only [rsJsNode, jsNode_labels] at h2
  injection h2 with h2
  injection h2 with h2
  cases h2

end AlgoUdf.UdfNodeView
