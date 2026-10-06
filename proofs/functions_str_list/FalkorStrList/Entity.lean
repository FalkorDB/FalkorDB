/-!
# Graph-entity functions (functions/entity.rs) over the runtime accessor model

The runtime accessors entity.rs calls (`runtime.rs:1661-1880`) are modelled with the order
they use: **deleted-entity snapshot > pending (this query's writes) > committed graph**.
`deleteNode` / `deleteRel` model the snapshot the DELETE operator takes
(`ops/delete.rs:226-290, :415-500`).

House decision (2026-09-25): a deleted entity keeps returning *all* its details for the rest
of the query. `deleteRel_preserves` and `deleteNode_preserves` prove it for every accessor
except node labels; `deleteNode_labels_iff` shows labels are preserved exactly when the
query staged no label change on the node before deleting it — the snapshot reads the
*committed* label ids (`g.get_node_label_ids(id)`, ops/delete.rs:282 and :461), not the
pending view. Confirmed live and in `lean_functions_str_list::bug_deleted_node_labels_*`.
-/
namespace FalkorStrList.Entity

/-- Values seen by entity.rs (all 16 variants; payloads only where read). -/
inductive EV where
  | null | bool (b : Bool) | int (i : Int) | float | str (s : String) | list (vs : List EV)
  | map (kvs : List (String × EV)) | node (id : Nat) | rel (id : Nat) | path (vs : List EV)
  | vecf32 | point | datetime | date | time | duration

structure DelNode where
  labels : List String
  attrs : List (String × EV)

structure DelRel where
  src : Nat
  dst : Nat
  typeName : String
  attrs : List (String × EV)

/-- The runtime state the accessors consult. -/
structure RT where
  committedLabels : Nat → List String
  /-- `pending.update_node_labels`: the staged label set, if this query changed it. -/
  pendingLabels : Nat → Option (List String)
  /-- node attrs with pending writes applied (`update_node_attrs`) -/
  nodeAttrs : Nat → List (String × EV)
  relAttrs : Nat → List (String × EV)
  relType : Nat → Option String
  relEnds : Nat → Nat × Nat
  /-- committed + pending − pending-deleted degree (`get_node_indegree` arithmetic) -/
  indeg : Nat → List String → Nat
  outdeg : Nat → List String → Nat
  pendingDeleted : Nat → Bool
  delNodes : Nat → Option DelNode
  delRels : Nat → Option DelRel

/-- `get_node_labels` (runtime.rs:1661). -/
def RT.labels (r : RT) (id : Nat) : List String :=
  match r.delNodes id with
  | some dn => dn.labels
  | none => (r.pendingLabels id).getD (r.committedLabels id)

/-- `get_node_attrs` (runtime.rs:1740). -/
def RT.getNodeAttrs (r : RT) (id : Nat) : List (String × EV) :=
  match r.delNodes id with
  | some dn => dn.attrs
  | none => r.nodeAttrs id

/-- `get_relationship_attrs` (runtime.rs:1758). -/
def RT.getRelAttrs (r : RT) (id : Nat) : List (String × EV) :=
  match r.delRels id with
  | some d => d.attrs
  | none => r.relAttrs id

/-- `get_relationship_endpoints` (runtime.rs:1778). -/
def RT.getRelEnds (r : RT) (id : Nat) : Nat × Nat :=
  match r.delRels id with
  | some d => (d.src, d.dst)
  | none => r.relEnds id

/-- `get_relationship_type` (runtime.rs:1791). -/
def RT.getRelType (r : RT) (id : Nat) : Option String :=
  match r.delRels id with
  | some d => some d.typeName
  | none => r.relType id

/-- `node_has_label` (runtime.rs:1691) — same three-way order as `get_node_labels`. -/
def RT.hasLabel (r : RT) (id : Nat) (l : String) : Bool := (r.labels id).contains l

/-- `get_node_in/outdegree[_by_type]` (runtime.rs:1805-1875): 0 for a deleted node. -/
def RT.inDegree (r : RT) (id : Nat) (ts : List String) : Nat :=
  if (r.delNodes id).isSome || r.pendingDeleted id then 0 else r.indeg id ts
def RT.outDegree (r : RT) (id : Nat) (ts : List String) : Nat :=
  if (r.delNodes id).isSome || r.pendingDeleted id then 0 else r.outdeg id ts

/-- DELETE of a node with `snapshot` (ops/delete.rs:279-288 / :460-465): labels from the
COMMITTED graph, attrs through `get_node_attrs` (pending applied). -/
def deleteNode (r : RT) (id : Nat) : RT :=
  { r with
    delNodes := fun x => if x = id then some ⟨r.committedLabels id, r.getNodeAttrs id⟩ else r.delNodes x
    pendingDeleted := fun x => if x = id then true else r.pendingDeleted x }

/-- DELETE of a relationship (ops/delete.rs:485-500): type, attrs, endpoints through the
accessors, before marking it deleted. -/
def deleteRel (r : RT) (id : Nat) : RT :=
  { r with delRels := fun x =>
      if x = id then
        some { src := (r.getRelEnds id).1, dst := (r.getRelEnds id).2,
               typeName := (r.getRelType id).getD "", attrs := r.getRelAttrs id }
      else r.delRels x }

/-! ## The entity.rs functions -/

inductive Res where
  | ok (v : EV) | err (s : String) | unreachable

/-- `labels` (entity.rs:40). -/
def labels (r : RT) : List EV → Res
  | .node id :: _ => .ok (.list ((r.labels id).map .str))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `typeOf` (entity.rs:56). Note `Edge`/`Vectorf32`, unlike `Value::name`. -/
def typeName : EV → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .float => "Float"
  | .str _ => "String" | .list _ => "List" | .map _ => "Map" | .node _ => "Node"
  | .rel _ => "Edge" | .path _ => "Path" | .vecf32 => "Vectorf32" | .point => "Point"
  | .datetime => "Datetime" | .date => "Date" | .time => "Time" | .duration => "Duration"

def typeOf : List EV → Res
  | v :: _ => .ok (.str (typeName v))
  | [] => .unreachable

/-- `hasLabels` (entity.rs:86): the whole list is type-checked even once settled. -/
def hasLabelsLoop (r : RT) (id : Nat) : List EV → Bool → Res
  | [], acc => .ok (.bool acc)
  | .str n :: ls, acc => hasLabelsLoop r id ls (acc && r.hasLabel id n)
  | .int _ :: _, _ => .err "Type mismatch: expected String but was Integer"
  | .float :: _, _ => .err "Type mismatch: expected String but was Float"
  | .bool _ :: _, _ => .err "Type mismatch: expected String but was Boolean"
  | _ :: _, _ => .err "Type mismatch: expected String"

def hasLabels (r : RT) : List EV → Res
  | [.node id, .list ls] => hasLabelsLoop r id ls true
  | [.null, _] | [_, .null] => .ok .null
  | _ => .unreachable

/-- `u64 as i64` -/
def asI64 (n : Nat) : Int := if n < 2^63 then n else (n : Int) - 2^64

/-- `id` (entity.rs:130). -/
def id : List EV → Res
  | .node n :: _ | .rel n :: _ => .ok (.int (asI64 n))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `properties` (entity.rs:149). -/
def properties (r : RT) : List EV → Res
  | .map m :: _ => .ok (.map m)
  | .node n :: _ => .ok (.map (r.getNodeAttrs n))
  | .rel e :: _ => .ok (.map (r.getRelAttrs e))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `startNode` / `endNode` (entity.rs:166, :181). -/
def startNode (r : RT) : List EV → Res
  | .rel e :: _ => .ok (.node (r.getRelEnds e).1)
  | _ => .unreachable
def endNode (r : RT) : List EV → Res
  | .rel e :: _ => .ok (.node (r.getRelEnds e).2)
  | _ => .unreachable

/-- `length` (entity.rs:196): `path.len() / 2`. -/
def length : List EV → Res
  | .path p :: _ => .ok (.int (p.length / 2 : Nat))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `keys` (entity.rs:214). -/
def keys (r : RT) : List EV → Res
  | .map m :: _ => .ok (.list (m.map fun kv => .str kv.1))
  | .node n :: _ => .ok (.list ((r.getNodeAttrs n).map fun kv => .str kv.1))
  | .rel e :: _ => .ok (.list ((r.getRelAttrs e).map fun kv => .str kv.1))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `type` (entity.rs:243). -/
def relationshipType (r : RT) : List EV → Res
  | .rel e :: _ => match r.getRelType e with
    | some t => .ok (.str t)
    | none => .ok .null
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `exists` (entity.rs:257). -/
def «exists» : List EV → Res
  | .null :: _ => .ok (.bool false)
  | _ :: _ => .ok (.bool true)
  | [] => .ok (.bool true)

/-- Collect distinct type names, rejecting non-strings (`parse_degree_args` loops). -/
def collectTypes (tyDbg : EV → String) : List EV → List String → Except String (List String)
  | [], acc => .ok acc
  | .str s :: vs, acc => collectTypes tyDbg vs (if acc.contains s then acc else acc ++ [s])
  | v :: _, _ => .error s!"Type mismatch: expected String but was {tyDbg v}"

/-- `parse_degree_args` (entity.rs:304). `tyDbg` is `{:?}` of `get_type()`. -/
def parseDegreeArgs (tyDbg : EV → String) (fn : String) (args : List EV) :
    Except String (Option Nat × List String) :=
  match args with
  | [] => .error s!"Received 0 arguments to function '{fn}', expected at least 1"
  | a0 :: rest =>
    let idr : Except String (Option Nat) := match a0 with
      | .node n => .ok (some n)
      | .null => .ok none
      | other => .error s!"Type mismatch: expected Node but was {tyDbg other}"
    match idr with
    | .error e => .error e
    | .ok i =>
      match rest with
      | [] => .ok (i, [])
      | [.list l] => (collectTypes tyDbg l []).map fun ts => (i, ts)
      | .list _ :: _ => .error s!"Received {rest.length + 1} arguments to function '{fn}', expected at most 2"
      | _ => (collectTypes tyDbg rest []).map fun ts => (i, ts)

/-- `indegree` / `outdegree` (entity.rs:268, :284). -/
def degree (deg : RT → Nat → List String → Nat) (tyDbg : EV → String) (fn : String) (r : RT)
    (args : List EV) : Res :=
  match parseDegreeArgs tyDbg fn args with
  | .error e => .err e
  | .ok (none, _) => .ok .null
  | .ok (some n, ts) => .ok (.int (deg r n ts))   -- `count as i64`; counts ≪ 2^63

def indegree := degree RT.inDegree
def outdegree := degree RT.outDegree

/-! ## Theorems -/

/-- **House decision, relationships**: after `DELETE r`, `type(r)`, `startNode(r)`,
`endNode(r)`, `properties(r)`, `keys(r)` and `id(r)` return exactly what they returned before
(when the relationship had a type, as every real edge does). -/
theorem deleteRel_preserves (r : RT) (e : Nat) (ht : (r.getRelType e).isSome) :
    relationshipType (deleteRel r e) [.rel e] = relationshipType r [.rel e] ∧
    startNode (deleteRel r e) [.rel e] = startNode r [.rel e] ∧
    endNode (deleteRel r e) [.rel e] = endNode r [.rel e] ∧
    properties (deleteRel r e) [.rel e] = properties r [.rel e] ∧
    keys (deleteRel r e) [.rel e] = keys r [.rel e] := by
  cases h : r.getRelType e with
  | none => simp [h] at ht
  | some t =>
    simp [relationshipType, startNode, endNode, properties, keys, deleteRel, RT.getRelType,
      RT.getRelEnds, RT.getRelAttrs] at h ⊢
    simp [h]

/-- **House decision, nodes**: after `DELETE n`, `properties(n)`, `keys(n)`, `id(n)` are
unchanged, and degrees become 0. -/
theorem deleteNode_preserves (r : RT) (n : Nat) :
    properties (deleteNode r n) [.node n] = properties r [.node n] ∧
    keys (deleteNode r n) [.node n] = keys r [.node n] ∧
    id [.node n] = .ok (.int (asI64 n)) ∧
    (deleteNode r n).inDegree n [] = 0 ∧ (deleteNode r n).outDegree n [] = 0 := by
  simp [properties, keys, id, deleteNode, RT.getNodeAttrs, RT.inDegree, RT.outDegree]

/-- Labels survive the DELETE exactly when the snapshot's committed labels equal what
`labels(n)` showed before: i.e. iff this query staged no (net) label change on `n`. -/
theorem deleteNode_labels_iff (r : RT) (n : Nat) (hnd : r.delNodes n = none) :
    (deleteNode r n).labels n = r.labels n ↔
      (r.pendingLabels n).getD (r.committedLabels n) = r.committedLabels n := by
  simp [RT.labels, deleteNode, hnd, eq_comm]

/-- Concrete counterexample: `MATCH (n:A) SET n:C DELETE n RETURN labels(n)` — before the
DELETE `labels(n) = [A, C]`, after it `[A]`. -/
def exRT : RT :=
  { committedLabels := fun _ => ["A"], pendingLabels := fun _ => some ["A", "C"],
    nodeAttrs := fun _ => [], relAttrs := fun _ => [], relType := fun _ => none,
    relEnds := fun _ => (0, 0), indeg := fun _ _ => 0, outdeg := fun _ _ => 0,
    pendingDeleted := fun _ => false, delNodes := fun _ => none, delRels := fun _ => none }

theorem labels_lost_after_delete :
    exRT.labels 0 = ["A", "C"] ∧ (deleteNode exRT 0).labels 0 = ["A"] := by
  simp [exRT, RT.labels, deleteNode]

/-- With no staged label change, every node accessor survives the DELETE. -/
theorem deleteNode_preserves_all (r : RT) (n : Nat) (hnd : r.delNodes n = none)
    (hp : r.pendingLabels n = none) :
    labels (deleteNode r n) [.node n] = labels r [.node n] ∧
    (∀ l, (deleteNode r n).hasLabel n l = r.hasLabel n l) := by
  have h := (deleteNode_labels_iff r n hnd).mpr (by simp [hp])
  simp [labels, RT.hasLabel, h]

/-- `hasLabels` with all-string labels is "every label is in `labels(n)`". -/
theorem hasLabels_all (r : RT) (n : Nat) (ls : List String) :
    hasLabels r [.node n, .list (ls.map .str)] = .ok (.bool (ls.all fun l => r.hasLabel n l)) := by
  simp only [hasLabels]
  suffices ∀ acc, hasLabelsLoop r n (ls.map .str) acc = .ok (.bool (acc && ls.all fun l => r.hasLabel n l)) by
    simpa using this true
  induction ls with
  | nil => intro acc; simp [hasLabelsLoop]
  | cons l ls ih => intro acc; simp [hasLabelsLoop, ih, Bool.and_assoc]

/-- A non-string label is an error even when the answer is already `false`. -/
theorem hasLabels_typecheck (r : RT) (n : Nat) (i : Int) :
    hasLabels r [.node n, .list [.str "Nope", .int i]] = .err "Type mismatch: expected String but was Integer" := by
  rfl

/-- `typeOf` never hits `unreachable!()` (its argument type is `Any`, arity 1). -/
theorem typeOf_total (v : EV) (rest : List EV) : typeOf (v :: rest) = .ok (.str (typeName v)) := rfl

theorem exists_spec (v : EV) : «exists» [v] = .ok (.bool (match v with | .null => false | _ => true)) := by
  cases v <;> rfl

/-- `id` is the identity on ids below 2^63 (all real ids). -/
theorem id_small (n : Nat) (h : n < 2^63) : id [.node n] = .ok (.int n) ∧ id [.rel n] = .ok (.int n) := by
  simp [id, asI64, h]

/-- `length(p)` counts relationships: a path of k+1 nodes and k edges has length k. -/
theorem length_path (k : Nat) (vs : List EV) (h : vs.length = 2 * k + 1) :
    length [.path vs] = .ok (.int k) := by
  simp only [length, h]; congr; omega

/-- Null propagation and the dead `unreachable!()` arms under the declared types. -/
theorem entity_null (r : RT) :
    labels r [.null] = .ok .null ∧ id [.null] = .ok .null ∧ properties r [.null] = .ok .null ∧
    length [.null] = .ok .null ∧ keys r [.null] = .ok .null ∧
    relationshipType r [.null] = .ok .null ∧ hasLabels r [.null, .list []] = .ok .null := by
  simp [labels, id, properties, length, keys, relationshipType, hasLabels]

theorem keys_eq_props_keys (r : RT) (n : Nat) :
    keys r [.node n] = .ok (.list ((r.getNodeAttrs n).map fun kv => .str kv.1)) ∧
    properties r [.node n] = .ok (.map (r.getNodeAttrs n)) := ⟨rfl, rfl⟩

/-- `parse_degree_args`: zero args is an error; node alone means "all types"; the list form
and the varargs form give the same distinct type list. -/
theorem parseDegreeArgs_forms (d : EV → String) (fn : String) (n : Nat) (ts : List String) :
    parseDegreeArgs d fn [] = .error s!"Received 0 arguments to function '{fn}', expected at least 1" ∧
    parseDegreeArgs d fn [.node n] = .ok (some n, []) ∧
    parseDegreeArgs d fn [.node n, .list (ts.map .str)] =
      (collectTypes d (ts.map .str) []).map (fun x => (some n, x)) := by
  refine ⟨rfl, rfl, rfl⟩

theorem collectTypes_strs (d : EV → String) (ts acc : List String) :
    collectTypes d (ts.map .str) acc = .ok (ts.foldl (fun a s => if a.contains s then a else a ++ [s]) acc) := by
  induction ts generalizing acc with
  | nil => rfl
  | cons t ts ih => simp [collectTypes, ih]

theorem degree_null (deg : RT → Nat → List String → Nat) (d : EV → String) (fn : String) (r : RT) :
    degree deg d fn r [.null] = .ok .null := rfl

theorem degree_deleted (d : EV → String) (r : RT) (n : Nat) :
    indegree d "indegree" (deleteNode r n) [.node n] = .ok (.int 0) ∧
    outdegree d "outdegree" (deleteNode r n) [.node n] = .ok (.int 0) := by
  simp [indegree, outdegree, degree, parseDegreeArgs, RT.inDegree, RT.outDegree, deleteNode]

/-- `register` (entity.rs:36): the names, in order; all distinct. -/
def registered : List String :=
  ["labels", "typeOf", "hasLabels", "id", "properties", "startnode", "endnode", "length", "keys",
   "type", "exists", "indegree", "outdegree"]

theorem register_nodup : registered.Nodup := by decide

end FalkorStrList.Entity
