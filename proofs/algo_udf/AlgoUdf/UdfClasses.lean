import AlgoUdf.Marshal
import AlgoUdf.AlgoGlue
/-! # JS object builders and native entry points (`graph/src/udf/js_classes.rs`)

| here | there |
| --- | --- |
| `Slot`, `setGraph`, `clearGraph`, `withGraph` | `CURRENT_GRAPH`, `set_current_graph` :72, `clear_current_graph` :78, `with_current_graph` :84 |
| `JV`, `Obj`, `set`, `get` | QuickJS values / ordinary objects (AXIOMATISED: `obj.set(k, v)` replaces own key `k`; `"__proto__"` never becomes an own key, cf. `Marshal.objSet`) |
| `Ent`, `newEntity`, `entityValue`, `nodeIdOf`, `nextEpoch` | `JsEntity` :83-86 (`trace` :93, `constructor` :105), `new_entity_object` :110, `entity_value` :129, `node_id_of` :135, the `CALL_EPOCH` bump in `with_graph` :162-165 (#2976) |
| `jsNode`                 | `create_js_node` :200 (a `JsEntity` instance holding `Value::Node`; its visible properties below) |
| `jsEdge`                 | `create_js_edge` :311 (type: runtime, else committed graph, else `""`) |
| `jsPath`                 | `create_js_path` :385 |
| `optArg`                 | rquickjs `Opt<Object>`: `None` only for a *missing* argument |
| `getNeighborsCall`       | the shared method `function(...args) { return h(this, ...args); }` :251 → `js_get_neighbors_entry` :448 |
| `Exc`, `throwJs`, `entryErr` | `throw_js_error` :599, the `Err` arms of `js_get_neighbors_entry` :456-466 / `js_traverse` |
| `nodeById`               | `js_get_node_by_id` :609, `js_get_node_by_id_impl` :616 |
| `iterNodes`, `iterEdges` | `js_iterate_nodes(_impl)` :653, :660, `js_iterate_edges(_impl)` :708, :715 (label matrix / tensor scans AXIOMATISED) |
| `traverseEntry`          | `js_traverse` :760 (body: `UdfTraverse`) |
-/
namespace AlgoUdf.UdfClasses
open AlgoUdf

/-! ## Current graph slot -/

abbrev Slot (γ : Type) := Option γ

def setGraph {γ : Type} (_ : Slot γ) (g : γ) : Slot γ := some g
def clearGraph {γ : Type} (_ : Slot γ) : Slot γ := none
def withGraph {γ ρ : Type} (s : Slot γ) (f : γ → Except String ρ) : Except String ρ :=
  match s with
  | none => .error "No graph context available"
  | some g => f g

theorem withGraph_set {γ ρ : Type} (s : Slot γ) (g : γ) (f : γ → Except String ρ) :
    withGraph (setGraph s g) f = f g := rfl
theorem withGraph_clear {γ ρ : Type} (s : Slot γ) (f : γ → Except String ρ) :
    withGraph (clearGraph s) f = .error "No graph context available" := rfl

/-! ## Objects -/

inductive JV where
  | undef | null
  | str (s : String)
  | num (n : Nat)
  | arr (xs : List JV)
  | obj (ps : List (String × JV))
  | fn
  | attr (tag : Nat)          -- an attribute value as converted by `value_to_js`
  deriving Inhabited

abbrev Obj := List (String × JV)

def set (o : Obj) (k : String) (v : JV) : Obj :=
  if k = "__proto__" then o else o.filter (·.1 != k) ++ [(k, v)]

def get (o : Obj) (k : String) : Option JV := (o.find? (·.1 == k)).map (·.2)

theorem get_set_same (o : Obj) (k : String) (v : JV) (hk : k ≠ "__proto__") :
    get (set o k v) k = some v := by
  unfold get set; rw [ite_eq_right hk]
  rw [List.find?_append]
  have : (o.filter (·.1 != k)).find? (·.1 == k) = none := by
    rw [List.find?_eq_none]; intro x hx; simp at hx ⊢; exact hx.2
  rw [this]; simp

theorem find_filter_ne (k k' : String) (hk : k' ≠ k) :
    ∀ l : Obj, (l.filter (·.1 != k)).find? (·.1 == k') = l.find? (·.1 == k') := by
  intro l
  induction l with
  | nil => rfl
  | cons p t ih =>
    by_cases hp : p.1 = k
    · have hp' : (p.1 == k') = false := by
        rw [hp]; simp only [beq_eq_false_iff_ne]; exact fun h => hk h.symm
      rw [List.filter_cons_of_neg (by simp [hp]), ih, List.find?_cons_of_neg (by simp [hp'])]
    · rw [List.filter_cons_of_pos (by simp [hp]), List.find?_cons, List.find?_cons, ih]

theorem get_set_ne (o : Obj) (k k' : String) (v : JV) (hk : k' ≠ k) : get (set o k v) k' = get o k' := by
  unfold get set
  split
  · rfl
  · rw [List.find?_append, find_filter_ne k k' hk]
    cases o.find? (·.1 == k') with
    | some _ => rfl
    | none => simp; intro h; exact hk h.symm

/-- Keys `create_js_node` never copies onto the node itself (:155-156). -/
def reserved (k : String) : Bool :=
  k == "id" || k == "labels" || k == "attributes" || k == "getNeighbors" ||
    "__falkor".toList.isPrefixOf k.toList

/-- The attribute loop :150-164: `(node object, attributes object)`. -/
def attrLoop : Obj × Obj → List (String × JV) → Obj × Obj
  | acc, [] => acc
  | (o, a), (k, v) :: t => attrLoop ((if reserved k then o else set o k v), set a k v) t

def jsNode (id : Nat) (labels : List String) (attrs : List (String × JV)) : Obj :=
  let o0 := set (set (set (set [] "__falkor_type" (.str "node")) "__falkor_node_id" (.num id)) "id" (.num id))
    "labels" (.arr (labels.map .str))
  let (o, a) := attrLoop (o0, []) attrs
  set (set o "attributes" (.obj a)) "getNeighbors" .fn

theorem attrLoop_keeps (k : String) (hk : reserved k = true) :
    ∀ (attrs : List (String × JV)) (o a : Obj), get (attrLoop (o, a) attrs).1 k = get o k := by
  intro attrs
  induction attrs with
  | nil => intro o a; rfl
  | cons p t ih =>
    obtain ⟨k', v⟩ := p
    intro o a
    simp only [attrLoop]
    rw [ih]
    split
    · rfl
    · rename_i h
      apply get_set_ne
      intro he; subst he; rw [hk] at h; exact h rfl

/-- **Markers survive any attribute set**: a node's `__falkor_type`,
`__falkor_node_id`, `id` and `labels` are exactly what `create_js_node` wrote,
whatever attributes the node has. -/
theorem jsNode_markers (id : Nat) (labels : List String) (attrs : List (String × JV)) :
    get (jsNode id labels attrs) "__falkor_type" = some (.str "node") ∧
    get (jsNode id labels attrs) "__falkor_node_id" = some (.num id) ∧
    get (jsNode id labels attrs) "id" = some (.num id) ∧
    get (jsNode id labels attrs) "labels" = some (.arr (labels.map .str)) := by
  unfold jsNode
  simp only
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
  · rw [get_set_ne _ _ _ _ (by decide), get_set_ne _ _ _ _ (by decide), attrLoop_keeps _ (by decide)]
    repeat (first | rw [get_set_same _ _ _ (by decide)] | rw [get_set_ne _ _ _ _ (by decide)])

/-! ## Entity objects (`JsEntity`, #2976) -/

/-- The class data of a `JsEntity` instance (:83-86): the entity and the epoch it was made in. -/
structure Ent where
  value : Marshal.RV
  epoch : Nat

/-- `JsEntity::trace` (:93): the class data holds no JS value, so there is nothing to mark. -/
def entTrace : List Unit := []

/-- `JsClass::constructor` (:105): `None`, so JS has no constructor to make an instance. -/
def entConstructor : Option Unit := none

/-- `new_entity_object` (:110-123): an instance holding `v` and the current epoch
(`Class::instance` failing is the `JS object error` arm). -/
def newEntity (epochNow : Nat) (v : Marshal.RV) : Ent := ⟨v, epochNow⟩

/-- `entity_value` (:129-133): `None` for an object that is not a `JsEntity`, or whose epoch is not
the current one. -/
def entityValue (epochNow : Nat) : Option Ent → Option Marshal.RV
  | none => none
  | some e => if e.epoch = epochNow then some e.value else none

/-- `node_id_of` (:135-140). -/
def nodeIdOf (epochNow : Nat) (o : Option Ent) : Option Nat :=
  match entityValue epochNow o with
  | some (.node id) => some id
  | _ => none

/-- `with_graph`'s epoch bump (:162-165): only an outermost call (no graph lent before) starts a
new epoch (`wrapping_add`: the wrap after 2^64 calls is not modelled). -/
def nextEpoch (prevLent : Bool) (epoch : Nat) : Nat := if prevLent then epoch else epoch + 1

theorem entTrace_empty : entTrace = [] ∧ entConstructor = none := ⟨rfl, rfl⟩

/-- **An entity made in this call reads back as itself.** -/
theorem entityValue_new (e : Nat) (v : Marshal.RV) : entityValue e (some (newEntity e v)) = some v := by
  simp [entityValue, newEntity]

/-- **An entity from another call is not honoured**: the next outermost call has another epoch. -/
theorem entityValue_stale (e : Nat) (v : Marshal.RV) :
    entityValue (nextEpoch false e) (some (newEntity e v)) = none := by
  simp [entityValue, newEntity, nextEpoch]

/-- Nested `with_graph` keeps the epoch, so an entity stays valid inside the same UDF call. -/
theorem entityValue_nested (e : Nat) (v : Marshal.RV) :
    entityValue (nextEpoch true e) (some (newEntity e v)) = some v := entityValue_new e v

/-- A non-entity object (a plain object copying the markers) has no entity. -/
theorem entityValue_plain (e : Nat) : entityValue e none = none := rfl

theorem nodeIdOf_spec (e id : Nat) (v : Marshal.RV) :
    nodeIdOf e (some (newEntity e (.node id))) = some id ∧
    nodeIdOf e (some (newEntity e (.rel id))) = none ∧
    nodeIdOf (nextEpoch false e) (some (newEntity e v)) = none ∧ nodeIdOf e none = none := by
  refine ⟨?_, ?_, ?_, rfl⟩ <;> simp [nodeIdOf, entityValue, newEntity, nextEpoch]

/-- The Marshal model's `JsV.ent ps v current` is the object `entity_value` reads: `current` is
`entityValue` returning `some v`. -/
theorem ent_current_iff (now : Nat) (x : Ent) :
    (entityValue now (some x)).isSome = decide (x.epoch = now) := by
  simp only [entityValue]; split <;> simp_all

/-! ## Edges and paths -/

/-- `create_js_edge`'s `type`: the runtime's view, else the committed graph's, else `""`. -/
def edgeType (rt : Option (Option String)) (committed : Option String) : String :=
  match rt with
  | some t => t.getD ""
  | none => committed.getD ""

def jsEdge (id src dst : Nat) (ty : String) (srcObj dstObj : Obj) (attrs : List (String × JV)) : Obj :=
  let o := set (set (set (set (set [] "__falkor_type" (.str "edge")) "__falkor_edge_id" (.num id))
    "__falkor_edge_src" (.num src)) "__falkor_edge_dst" (.num dst)) "id" (.num id)
  let o := set (set (set o "type" (.str ty)) "source" (.obj srcObj)) "target" (.obj dstObj)
  set o "attributes" (.obj (attrs.foldl (fun a kv => set a kv.1 kv.2) []))

theorem jsEdge_markers (id s d : Nat) (ty : String) (so dob : Obj) (attrs : List (String × JV)) :
    get (jsEdge id s d ty so dob attrs) "__falkor_type" = some (.str "edge") ∧
    get (jsEdge id s d ty so dob attrs) "__falkor_edge_id" = some (.num id) ∧
    get (jsEdge id s d ty so dob attrs) "type" = some (.str ty) := by
  unfold jsEdge
  simp only
  refine ⟨?_, ?_, ?_⟩ <;>
  repeat (first | rw [get_set_same _ _ _ (by decide)] | rw [get_set_ne _ _ _ _ (by decide)])

theorem edgeType_runtime_first (t : Option String) (c : Option String) :
    edgeType (some t) c = t.getD "" ∧ edgeType none c = c.getD "" := ⟨rfl, rfl⟩

/-- Path elements as `create_js_path` sees them. -/
inductive PE | node (id : Nat) | rel (id : Nat) | other

/-- `(nodes, relationships, length)` :305-343. -/
def jsPath (pv : List PE) : List Nat × List Nat × Nat :=
  let ns := pv.filterMap (fun e => match e with | .node n => some n | _ => none)
  let rs := pv.filterMap (fun e => match e with | .rel r => some r | _ => none)
  (ns, rs, rs.length)

theorem jsPath_length (pv : List PE) : (jsPath pv).2.2 = (jsPath pv).2.1.length := rfl

/-! ## Native entry points -/

/-- rquickjs `Opt<Object>`: a missing argument is `None`; a present
non-object (including `undefined`) fails conversion. -/
def optArg (args : List JV) (i : Nat) : Except String (Option Obj) :=
  match args[i]? with
  | none => .ok none
  | some (.obj o) => .ok (some o)
  | some .undef => .error "Error converting from js 'undefined' into type 'object'"
  | some _ => .error "Error converting from js value into type 'object'"

/-- `node.getNeighbors(...)` goes through `(h) => function(...args) { return h(this, ...args); }`
(:251, #2976), so the user's arguments reach `js_get_neighbors_entry` :448 as given, after `this`:
`config` is the user's first argument (`optArg userArgs 0`), missing when omitted. rquickjs
converts the parameters before the body runs, then the body needs `this` to be a current node
entity (`thisNode = nodeIdOf …`, :453-455). Returns `(node id, config)`. -/
def getNeighborsCall (thisNode : Option Nat) (userArgs : List JV) : Except String (Nat × Option Obj) :=
  match optArg userArgs 0 with
  | .error e => .error e
  | .ok cfg => match thisNode with
    | none => .error "getNeighbors must be called on a node"
    | some n => .ok (n, cfg)

/-- Before #2976: `function(config) { return h(this.__falkor_node_id, config); }` always passed
two arguments, so an omitted config arrived as `undefined`. -/
def getNeighborsCallPre2976 (nodeId : Nat) (userArgs : List JV) : Except String (Option Obj) :=
  let config := (userArgs[0]?).getD .undef
  optArg [.num nodeId, config] 1

/-- Historical (BUG, confirmed live and by `bug_udf_get_neighbors_without_config_throws`; fixed
by #2976): `node.getNeighbors()` with no argument failed; C returns the neighbours. -/
theorem pre2976_getNeighbors_no_arg_errors (n : Nat) :
    getNeighborsCallPre2976 n [] = .error "Error converting from js 'undefined' into type 'object'" := rfl

/-- **Fixed by #2976**: `node.getNeighbors()` with no argument runs with no config (as in C). -/
theorem getNeighbors_no_arg_ok (n : Nat) : getNeighborsCall (some n) [] = .ok (n, none) := rfl

theorem getNeighbors_with_config (n : Nat) (o : Obj) (rest : List JV) :
    getNeighborsCall (some n) (.obj o :: rest) = .ok (n, some o) := rfl

/-- An explicit `undefined` config is still a conversion error (unchanged by #2976). -/
theorem getNeighbors_undefined_errors (n : Nat) :
    getNeighborsCall (some n) [.undef] = .error "Error converting from js 'undefined' into type 'object'" ∧
    getNeighborsCallPre2976 n [.undef] = .error "Error converting from js 'undefined' into type 'object'" :=
  ⟨rfl, rfl⟩

/-- Called on anything but a current node entity (`getNeighbors.call({__falkor_node_id: 0})`,
an edge, a node kept from an earlier call) it is an error, never another node's neighbours. -/
theorem getNeighbors_not_node (args : List JV) (h : optArg args 0 = .ok none ∨ ∃ o, optArg args 0 = .ok (some o)) :
    getNeighborsCall none args = .error "getNeighbors must be called on a node" := by
  rcases h with h | ⟨o, h⟩ <;> simp [getNeighborsCall, h]

/-- `graph.traverse` is bound directly (`Function::new(ctx, js_traverse)`), so an
omitted config really is missing. -/
theorem traverse_no_config_ok (arr : JV) : optArg [arr] 1 = .ok none := rfl

inductive Exc | msg (s : String) | bare deriving DecidableEq, Repr

/-- `throw_js_error`: throw the message as a JS string, or a bare exception if
the string cannot be allocated. -/
def throwJs (allocOk : Bool) (e : String) : Exc := if allocOk then .msg e else .bare

/-- The `Err` arm of `js_get_neighbors_entry` / `js_traverse`: fall back to the
string `"internal error"` before a bare exception. -/
def entryErr (allocMsg allocFallback : Bool) (e : String) : Exc :=
  if allocMsg then .msg e else if allocFallback then .msg "internal error" else .bare

theorem entryErr_spec (e : String) :
    entryErr true true e = .msg e ∧ entryErr false true e = .msg "internal error" ∧
    entryErr false false e = .bare ∧ throwJs true e = .msg e ∧ throwJs false e = .bare :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- A JS number argument as rquickjs sees it. -/
inductive JNum | int (i : Int) | float (f : Marshal.F64) | other

def nodeByIdErr : String := "getNodeById: expected a non-negative integer node id"

/-- Argument conversion of `js_get_node_by_id_impl` :521-536. -/
def nodeIdArg : Option JNum → Except String Nat
  | none => .error nodeByIdErr
  | some (.int i) => if 0 ≤ i then .ok i.toNat else .error nodeByIdErr
  | some (.float f) =>
    match f with
    | .zero _ => .ok 0
    | .integ k => if 0 ≤ k then .ok (min k.toNat (2 ^ 64 - 1)) else .error nodeByIdErr
    | _ => .error nodeByIdErr
  | some .other => .error nodeByIdErr

/-- `graph.getNodeById`: `some id` = a node object, `none` = `null`. -/
def nodeById (nodeCount maxId : Nat) (deleted : Nat → Bool) (arg : Option JNum) :
    Except String (Option Nat) :=
  match nodeIdArg arg with
  | .error e => .error e
  | .ok id => .ok (if nodeCount > 0 ∧ id ≤ maxId ∧ deleted id = false then some id else none)

/-- A node object is returned exactly for live node ids; everything else is `null`. -/
theorem nodeById_spec (nodeCount maxId : Nat) (deleted : Nat → Bool) (id : Nat) :
    nodeById nodeCount maxId deleted (some (.int id)) =
      .ok (if id ∈ AlgoGlue.activeNodes nodeCount maxId deleted then some id else none) := by
  simp only [nodeById, nodeIdArg, Int.natCast_nonneg, ite_true, Int.toNat_natCast]
  congr 2
  rw [AlgoGlue.mem_activeNodes]; simp only [eq_iff_iff]
  constructor <;> rintro ⟨a, b, c⟩ <;> exact ⟨by omega, b, c⟩

theorem nodeById_rejects (nodeCount maxId : Nat) (deleted : Nat → Bool) :
    nodeById nodeCount maxId deleted (some (.int (-1))) = .error nodeByIdErr ∧
    nodeById nodeCount maxId deleted (some (.float (.frac 0))) = .error nodeByIdErr ∧
    nodeById nodeCount maxId deleted (some (.float .nan)) = .error nodeByIdErr ∧
    nodeById nodeCount maxId deleted none = .error nodeByIdErr := by
  refine ⟨rfl, rfl, rfl, rfl⟩

/-- Label / type argument of `iterateNodes` / `iterateEdges`. -/
inductive LArg | absent | nullish | str (s : String) | other

def iterNodes (nodeCount maxId : Nat) (deleted : Nat → Bool) (labelScan : String → List Nat) :
    LArg → Except String (List Nat)
  | .absent | .nullish => .ok (AlgoGlue.activeNodes nodeCount maxId deleted)
  | .str l => .ok (labelScan l)
  | .other => .error "iterateNodes: label must be a string"

def iterEdges (allEdges : List (Nat × Nat × Nat)) (typeScan : String → List (Nat × Nat × Nat)) :
    LArg → Except String (List (Nat × Nat × Nat))
  | .absent | .nullish => .ok allEdges
  | .str t => .ok (typeScan t)
  | .other => .error "iterateEdges: relationship type must be a string"

/-- `iterateNodes()` returns exactly the live nodes, each once, ascending. -/
theorem iterNodes_all (nodeCount maxId : Nat) (deleted : Nat → Bool) (scan : String → List Nat) :
    ∀ x, (∃ r, iterNodes nodeCount maxId deleted scan .absent = .ok r ∧ x ∈ r) ↔
      nodeCount ≠ 0 ∧ x ≤ maxId ∧ deleted x = false := by
  intro x
  simp only [iterNodes, Except.ok.injEq, exists_eq_left']
  exact AlgoGlue.mem_activeNodes nodeCount maxId deleted x

theorem iter_errors (nc m : Nat) (d : Nat → Bool) (s : String → List Nat)
    (ae : List (Nat × Nat × Nat)) (ts : String → List (Nat × Nat × Nat)) :
    iterNodes nc m d s .other = .error "iterateNodes: label must be a string" ∧
    iterEdges ae ts .other = .error "iterateEdges: relationship type must be a string" ∧
    iterEdges ae ts .nullish = .ok ae := ⟨rfl, rfl, rfl⟩

/-- `js_traverse`: the impl's error becomes a JS exception (see `entryErr`). -/
def traverseEntry {ρ : Type} (impl : Except String ρ) (allocMsg allocFallback : Bool) : Except Exc ρ :=
  match impl with
  | .ok v => .ok v
  | .error e => .error (entryErr allocMsg allocFallback e)

theorem traverseEntry_spec {ρ : Type} (v : ρ) (e : String) :
    traverseEntry (.ok v) true true = .ok v ∧ traverseEntry (ρ := ρ) (.error e) true true = .error (.msg e) :=
  ⟨rfl, rfl⟩

end AlgoUdf.UdfClasses
