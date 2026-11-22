/-
# `algo.SPpaths` / `algo.SSpaths`: config parsing, ordering, result assembly
(`graph/src/runtime/functions/algo_procedures.rs`)

| here | there |
| --- | --- |
| `CV`, `Cfg`, `pRelTypes` … `pPathCount`, `parseCommon` | `PathAlgoConfig`, `parse_common_path_config` (:2030) |
| `parseSp`, `parseSs` | `parse_sp_config` (:2116), `parse_ss_config` (:2152) |
| `toNumeric` | `to_numeric_value` (:2171) |
| `cmpFound` | `cmp_found_path` (:2193) |
| `edgeAttr` | `edge_numeric_attr` (:2205) |
| `farEndpoint`, `successors` | `far_endpoint` (:2222), `node_successors` (:2254) |
| `itemCmp` | `DijkstraItem::cmp` / `partial_cmp` (:2289, :2304) |
| `buildBatch` | `build_path_batch` (:2679) |
| `Registered`, `spReg`, `ssReg` | `register_sp_paths` (:2761), `register_ss_paths` (:2773) |

`f64` is the type parameter `F` with only the operations the code uses (`n as f64`, `partial_cmp`,
`is_finite`, `fract`, `abs`, `as i64`). `parse_config` (map from the argument) is an input.

The three searches are modelled in `BfsFind` / `Dijkstra` / `Enum` (see `AlgoSearch`); the `f64` optimality
of `dijkstra_single_path` is not provable (`+` is not associative), its exact-domain twin is `DijkstraNat`.
-/
namespace OpsTraverse.AlgoPaths

/-! ## The config map -/

/-- The `Value` shapes the parser looks at. -/
inductive CV (F : Type) where
  | null
  | int (n : Int)
  | float (f : F)
  | str (s : String)
  | list (l : List (CV F))
  | node (id : Nat)
  | other

inductive Dir | outgoing | incoming | both
  deriving DecidableEq

/-- `EdgeDirection::from_str` (graph.rs). -/
def Dir.parse : String → Option Dir
  | "outgoing" => some .outgoing
  | "incoming" => some .incoming
  | "both" => some .both
  | _ => none

structure Cfg (F : Type) where
  source : Nat
  target : Option Nat
  relTypes : List String
  dir : Dir
  maxLen : Nat           -- a `u32`
  weightProp : Option String
  costProp : Option String
  maxCost : Option F
  pathCount : Int

variable {F : Type}

/-- `u32::MAX` -/
def u32max : Nat := 4294967295

def pRelTypes : Option (CV F) → Except String (List String)
  | none | some .null => .ok []
  | some (.list l) =>
    l.mapM (fun v => match v with
      | .str s => .ok s
      | _ => .error "relTypes must be array of strings")
  | _ => .error "relTypes must be array of strings"

def pDir : Option (CV F) → Except String Dir
  | none | some .null => .ok .outgoing
  | some (.str s) => match Dir.parse s with
    | some d => .ok d
    | none => .error "relDirection values must be 'incoming', 'outgoing' or 'both'"
  | _ => .error "relDirection values must be 'incoming', 'outgoing' or 'both'"

/-- `maxLen`: `n as u32` wraps modulo 2^32 (so `2^32` means `0`, and `0` finds nothing). -/
def pMaxLen : Option (CV F) → Except String Nat
  | none | some .null => .ok u32max
  | some (.int n) => if n < 0 then .error "maxLen must be non-negative integer" else .ok (n.toNat % 4294967296)
  | _ => .error "maxLen must be integer"

def pOptStr (msg : String) : Option (CV F) → Except String (Option String)
  | none | some .null => .ok none
  | some (.str s) => .ok (some s)
  | _ => .error msg

/-- `maxCost`: an `Int` is converted with `as f64` (`ofInt`). -/
def pMaxCost (ofInt : Int → F) : Option (CV F) → Except String (Option F)
  | none | some .null => .ok none
  | some (.int n) => .ok (some (ofInt n))
  | some (.float f) => .ok (some f)
  | _ => .error "maxCost must be numeric"

def pPathCount : Option (CV F) → Except String Int
  | none | some .null => .ok 1
  | some (.int n) => if n < 0 then .error "pathCount must be a non-negative integer" else .ok n
  | _ => .error "pathCount must be integer"

/-- `parse_common_path_config` (:2030): the seven fields in order; the first failing one is the
error. -/
def parseCommon (ofInt : Int → F) (get : String → Option (CV F)) (source : Nat) (target : Option Nat) :
    Except String (Cfg F) := do
  let relTypes ← pRelTypes (get "relTypes")
  let dir ← pDir (get "relDirection")
  let maxLen ← pMaxLen (get "maxLen")
  let weightProp ← pOptStr "weightProp must be string" (get "weightProp")
  let costProp ← pOptStr "costProp must be string" (get "costProp")
  let maxCost ← pMaxCost ofInt (get "maxCost")
  let pathCount ← pPathCount (get "pathCount")
  pure { source, target, relTypes, dir, maxLen, weightProp, costProp, maxCost, pathCount }

/-- **PROVEN** (`parse_common_path_config`): success exactly when each of the seven fields parses,
and then the config holds the source/target it was given and each parsed field. -/
theorem parseCommon_ok_iff (ofInt : Int → F) (get : String → Option (CV F)) (s : Nat) (t : Option Nat)
    (c : Cfg F) :
    parseCommon ofInt get s t = .ok c ↔
      ∃ rt d ml wp cp mc pc,
        pRelTypes (get "relTypes") = .ok rt ∧ pDir (get "relDirection") = .ok d ∧
        pMaxLen (get "maxLen") = .ok ml ∧ pOptStr "weightProp must be string" (get "weightProp") = .ok wp ∧
        pOptStr "costProp must be string" (get "costProp") = .ok cp ∧
        pMaxCost ofInt (get "maxCost") = .ok mc ∧ pPathCount (get "pathCount") = .ok pc ∧
        c = ⟨s, t, rt, d, ml, wp, cp, mc, pc⟩ := by
  unfold parseCommon
  cases h1 : pRelTypes (get "relTypes") <;> cases h2 : pDir (get "relDirection") <;>
  cases h3 : pMaxLen (get "maxLen") <;>
  cases h4 : pOptStr "weightProp must be string" (get "weightProp") <;>
  cases h5 : pOptStr "costProp must be string" (get "costProp") <;>
  cases h6 : pMaxCost ofInt (get "maxCost") <;> cases h7 : pPathCount (get "pathCount") <;>
  simp [bind, Except.bind, pure, Except.pure, h1, h2, h3, h4, h5, h6, h7, eq_comm]

/-- **PROVEN**: the first failing field's message is the error (a field after it is never read). -/
theorem parseCommon_first_error (ofInt : Int → F) (get : String → Option (CV F)) (s : Nat)
    (t : Option Nat) (e : String)
    (h1 : pRelTypes (get "relTypes") = .error e) : parseCommon ofInt get s t = .error e := by
  simp [parseCommon, bind, Except.bind, h1]

theorem parseCommon_dir_error (ofInt : Int → F) (get : String → Option (CV F)) (s : Nat)
    (t : Option Nat) (rt : List String) (e : String)
    (h1 : pRelTypes (get "relTypes") = .ok rt) (h2 : pDir (get "relDirection") = .error e) :
    parseCommon ofInt get s t = .error e := by
  simp [parseCommon, bind, Except.bind, h1, h2]

/-- Field specs: `relTypes`/`relDirection`/`maxLen`/`pathCount` defaults and rejections. -/
theorem pRelTypes_default : pRelTypes (F := F) none = .ok [] ∧ pRelTypes (F := F) (some .null) = .ok [] :=
  ⟨rfl, rfl⟩

theorem pRelTypes_nonstring (l : List (CV F)) (hn : ∃ x ∈ l, ∀ s, x ≠ .str s) :
    pRelTypes (some (.list l)) = .error "relTypes must be array of strings" := by
  simp only [pRelTypes]
  induction l with
  | nil => obtain ⟨x, hx, _⟩ := hn; simp at hx
  | cons y ys ih =>
    obtain ⟨x, hx, hxs⟩ := hn
    cases y with
    | str s0 =>
      have hx' : x ∈ ys := by
        rcases List.mem_cons.mp hx with rfl | h
        · exact absurd rfl (hxs s0)
        · exact h
      have := ih ⟨x, hx', hxs⟩
      simp [List.mapM_cons, bind, Except.bind, this]
    | _ => simp [List.mapM_cons, bind, Except.bind]

/-- A list of strings is accepted as is. -/
theorem pRelTypes_strings (ss : List String) : pRelTypes (F := F) (some (.list (ss.map CV.str))) = .ok ss := by
  simp only [pRelTypes]
  induction ss with
  | nil => rfl
  | cons s ss ih => simp [List.mapM_cons, bind, Except.bind, ih, pure, Except.pure]

theorem pDir_spec (s : String) :
    (s = "outgoing" → pDir (F := F) (some (.str s)) = .ok .outgoing) ∧
    (s = "incoming" → pDir (F := F) (some (.str s)) = .ok .incoming) ∧
    (s = "both" → pDir (F := F) (some (.str s)) = .ok .both) ∧
    (s ≠ "outgoing" → s ≠ "incoming" → s ≠ "both" →
      pDir (F := F) (some (.str s)) = .error "relDirection values must be 'incoming', 'outgoing' or 'both'") := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro rfl; rfl
  · rintro rfl; rfl
  · rintro rfl; rfl
  · intro h1 h2 h3
    simp [pDir, Dir.parse, h1, h2, h3]

theorem pDir_default : pDir (F := F) none = .ok .outgoing ∧ pDir (F := F) (some .null) = .ok .outgoing :=
  ⟨rfl, rfl⟩

theorem pMaxLen_spec (n : Int) :
    (0 ≤ n → pMaxLen (F := F) (some (.int n)) = .ok (n.toNat % 4294967296)) ∧
    (n < 0 → pMaxLen (F := F) (some (.int n)) = .error "maxLen must be non-negative integer") := by
  refine ⟨fun h => ?_, fun h => ?_⟩ <;> simp [pMaxLen, h]

/-- `maxLen` wraps: `2^32` is `0` (no path of at least one edge fits), shared with C. -/
theorem pMaxLen_wraps : pMaxLen (F := F) (some (.int 4294967296)) = .ok 0 := by simp [pMaxLen]

theorem pMaxLen_default : pMaxLen (F := F) none = .ok u32max := rfl

theorem pPathCount_spec (n : Int) :
    (0 ≤ n → pPathCount (F := F) (some (.int n)) = .ok n) ∧
    (n < 0 → pPathCount (F := F) (some (.int n)) = .error "pathCount must be a non-negative integer") := by
  refine ⟨fun h => ?_, fun h => ?_⟩ <;> simp [pPathCount, h]

theorem pPathCount_default : pPathCount (F := F) none = .ok 1 := rfl

theorem pMaxCost_spec (ofInt : Int → F) (n : Int) (f : F) :
    pMaxCost ofInt (some (.int n)) = .ok (some (ofInt n)) ∧
    pMaxCost ofInt (some (.float f)) = .ok (some f) ∧ pMaxCost ofInt none = .ok none := ⟨rfl, rfl, rfl⟩

/-! ## `parse_sp_config` / `parse_ss_config` -/

/-- `matches!(v, Some(v) if !matches!(v, Value::Null))`. -/
def present : Option (CV F) → Bool
  | some .null | none => false
  | some _ => true

/-- `parse_sp_config` (:2116) on the result of `parse_config(args)`. -/
def parseSp (ofInt : Int → F) (pc : Except String (String → Option (CV F))) : Except String (Cfg F) := do
  let get ← pc
  if !present (get "sourceNode") || !present (get "targetNode") then
    throw "sourceNode and targetNode are required"
  let source ← match get "sourceNode" with
    | some (.node id) => pure id
    | _ => throw "sourceNode and targetNode must be of type Node"
  let target ← match get "targetNode" with
    | some (.node id) => pure id
    | _ => throw "sourceNode and targetNode must be of type Node"
  parseCommon ofInt get source (some target)

/-- `parse_ss_config` (:2152). -/
def parseSs (ofInt : Int → F) (pc : Except String (String → Option (CV F))) : Except String (Cfg F) := do
  let get ← pc
  if !present (get "sourceNode") then throw "sourceNode is required"
  let source ← match get "sourceNode" with
    | some (.node id) => pure id
    | _ => throw "sourceNode must be of type Node"
  parseCommon ofInt get source none

/-- **PROVEN** (`parse_sp_config`): it succeeds exactly when the config argument parses, both
endpoints are `Node`s, and the common fields parse; the config carries those two nodes. -/
theorem parseSp_ok_iff (ofInt : Int → F) (pc : Except String (String → Option (CV F))) (c : Cfg F) :
    parseSp ofInt pc = .ok c ↔
      ∃ get s t, pc = .ok get ∧ get "sourceNode" = some (.node s) ∧ get "targetNode" = some (.node t) ∧
        parseCommon ofInt get s (some t) = .ok c := by
  rcases pc with e | get
  · simp [parseSp, bind, Except.bind]
  · simp only [parseSp, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure]
    constructor
    · intro h
      by_cases hp : (!present (get "sourceNode") || !present (get "targetNode")) = true
      · simp [hp] at h
      · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
        rcases hs : get "sourceNode" with _ | v
        · simp [hs, present] at hp
        · cases v <;> simp [hs] at h <;> rcases ht : get "targetNode" with _ | w <;>
            (try (simp [ht, present] at hp)) <;> (try cases w) <;> simp_all
    · rintro ⟨g, s, t, hg, hs, ht, h⟩
      cases hg
      simp [hs, ht, present, h]

theorem parseSp_missing (ofInt : Int → F) (get : String → Option (CV F))
    (h : present (get "sourceNode") = false ∨ present (get "targetNode") = false) :
    parseSp ofInt (.ok get) = .error "sourceNode and targetNode are required" := by
  rcases h with h | h <;> simp [parseSp, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, h]

theorem parseSs_missing (ofInt : Int → F) (get : String → Option (CV F))
    (h : present (get "sourceNode") = false) :
    parseSs ofInt (.ok get) = .error "sourceNode is required" := by
  simp [parseSs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, h]

/-- **PROVEN** (`parse_ss_config`): success exactly when the config parses, the source is a `Node`,
and the common fields parse; there is no target. -/
theorem parseSs_ok_iff (ofInt : Int → F) (pc : Except String (String → Option (CV F))) (c : Cfg F) :
    parseSs ofInt pc = .ok c ↔
      ∃ get s, pc = .ok get ∧ get "sourceNode" = some (.node s) ∧
        parseCommon ofInt get s none = .ok c := by
  rcases pc with e | get
  · simp [parseSs, bind, Except.bind]
  · simp only [parseSs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure]
    constructor
    · intro h
      by_cases hp : (!present (get "sourceNode")) = true
      · simp [hp] at h
      · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
        rcases hs : get "sourceNode" with _ | v
        · simp [hs, present] at hp
        · cases v <;> simp_all
    · rintro ⟨g, s, hg, hs, h⟩
      cases hg
      simp [hs, present, h]

/-! ## `to_numeric_value` -/

/-- The `f64` operations `to_numeric_value` uses. -/
structure FloatOps (F : Type) where
  isFinite : F → Bool
  fractZero : F → Bool     -- `v.fract() == 0.0`
  absLtMax : F → Bool      -- `v.abs() < (i64::MAX as f64)`
  toI64 : F → Int          -- `v as i64`

inductive Num (F : Type) where
  | int (n : Int)
  | float (f : F)

/-- `to_numeric_value` (:2171): an integral, finite, in-range float is reported as an `Integer`. -/
def toNumeric (o : FloatOps F) (v : F) : Num F :=
  if o.isFinite v && o.fractZero v && o.absLtMax v then .int (o.toI64 v) else .float v

/-- **PROVEN** (`to_numeric_value`): `Int (v as i64)` exactly when `v` is finite, integral and below
`i64::MAX as f64` in magnitude; every other float, `NaN` and `inf` included, stays a `Float`. -/
theorem toNumeric_spec (o : FloatOps F) (v : F) :
    (o.isFinite v = true → o.fractZero v = true → o.absLtMax v = true → toNumeric o v = .int (o.toI64 v)) ∧
    (¬ (o.isFinite v = true ∧ o.fractZero v = true ∧ o.absLtMax v = true) →
      ∃ f, toNumeric o v = .float f ∧ f = v) := by
  refine ⟨fun h1 h2 h3 => by simp [toNumeric, h1, h2, h3], fun h => ?_⟩
  refine ⟨v, ?_, rfl⟩
  have : ¬ (o.isFinite v && o.fractZero v && o.absLtMax v) = true := by
    simpa [Bool.and_eq_true, and_assoc] using h
  simp [toNumeric, this]

/-- A non-finite value never becomes an `Integer` (so `nan`/`inf` are reported as they are). -/
theorem toNumeric_nonfinite (o : FloatOps F) (v : F) (h : o.isFinite v = false) :
    toNumeric o v = .float v := by simp [toNumeric, h]

/-! ## `cmp_found_path` -/

/-- A found path: its edges (kept in original direction), weight and cost. -/
structure Found (E F : Type) where
  edges : List (E × Nat × Nat)
  weight : F
  cost : F

/-- `a.partial_cmp(b).unwrap_or(Equal)`. -/
def pc {F : Type} (pcmp : F → F → Option Ordering) (a b : F) : Ordering := (pcmp a b).getD .eq

/-- `cmp_found_path` (:2193): ascending weight, then cost, then hop count; an unordered
(`NaN`) comparison counts as equal. -/
def cmpFound {E : Type} (pcmp : F → F → Option Ordering) (a b : Found E F) : Ordering :=
  (pc pcmp a.weight b.weight).then ((pc pcmp a.cost b.cost).then (compare a.edges.length b.edges.length))

theorem ord_then_swap (x y : Ordering) : (x.then y).swap = x.swap.then y.swap := by
  cases x <;> cases y <;> rfl

/-- **PROVEN** (`cmp_found_path`): lexicographic on (weight, cost, hops). -/
theorem cmpFound_lex {E : Type} (pcmp : F → F → Option Ordering) (a b : Found E F) :
    (pcmp a.weight b.weight = some .lt → cmpFound pcmp a b = .lt) ∧
    (pcmp a.weight b.weight = some .gt → cmpFound pcmp a b = .gt) ∧
    ((pcmp a.weight b.weight = some .eq ∨ pcmp a.weight b.weight = none) →
      pcmp a.cost b.cost = some .lt → cmpFound pcmp a b = .lt) ∧
    ((pcmp a.weight b.weight = some .eq ∨ pcmp a.weight b.weight = none) →
      (pcmp a.cost b.cost = some .eq ∨ pcmp a.cost b.cost = none) →
      cmpFound pcmp a b = compare a.edges.length b.edges.length) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h1 h2 => ?_, fun h1 h2 => ?_⟩
  · simp [cmpFound, pc, h, Ordering.then]
  · simp [cmpFound, pc, h, Ordering.then]
  · rcases h1 with h1 | h1 <;> simp [cmpFound, pc, h1, h2, Ordering.then]
  · rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2 <;> simp [cmpFound, pc, h1, h2, Ordering.then]

/-- **PROVEN**: when `partial_cmp` is antisymmetric the comparator is (`sort_by` needs it). -/
theorem cmpFound_swap {E : Type} (pcmp : F → F → Option Ordering)
    (hanti : ∀ x y, pcmp y x = (pcmp x y).map Ordering.swap) (a b : Found E F) :
    cmpFound pcmp b a = (cmpFound pcmp a b).swap := by
  have hp : ∀ x y, pc pcmp y x = (pc pcmp x y).swap := by
    intro x y
    simp only [pc, hanti x y]
    cases pcmp x y with
    | none => rfl
    | some o => cases o <;> rfl
  have hn : ∀ m n : Nat, compare n m = (compare m n).swap := by
    intro m n
    simp only [compare, compareOfLessAndEq]
    by_cases h1 : m < n <;> by_cases h2 : n < m <;> by_cases h3 : m = n <;> simp [h1, h2, h3, Ordering.swap]
    all_goals omega
  have e1 := hp a.weight b.weight
  have e2 := hp a.cost b.cost
  have e3 := hn a.edges.length b.edges.length
  simp only [cmpFound, e1, e2, e3, ord_then_swap, Ordering.swap_swap]

/-! ## `edge_numeric_attr`, `far_endpoint`, `node_successors` -/

inductive AttrV (F : Type) where
  | float (f : F)
  | int (i : Int)
  | other

/-- `edge_numeric_attr` (:2205): `default` unless the property is configured, present and numeric. -/
def edgeAttr (ofInt : Int → F) (prop : Option String) (lookup : String → Option (AttrV F)) (default : F) : F :=
  match prop with
  | none => default
  | some p => match lookup p with
    | some (.float f) => f
    | some (.int i) => ofInt i
    | _ => default

/-- **PROVEN** (`edge_numeric_attr`): no property -> the default; a `Float` -> itself; an `Int` ->
`as f64`; absent or any other type -> the default. (Since #3121 every call site passes the cost
default explicitly: `1.0`, or `0.0` only in `dijkstra_single_path` when no `costProp` is set.) -/
theorem edgeAttr_spec (ofInt : Int → F) (lookup : String → Option (AttrV F)) (d : F) (p : String) :
    edgeAttr ofInt none lookup d = d ∧
    (∀ f, lookup p = some (.float f) → edgeAttr ofInt (some p) lookup d = f) ∧
    (∀ i, lookup p = some (.int i) → edgeAttr ofInt (some p) lookup d = ofInt i) ∧
    (lookup p = none → edgeAttr ofInt (some p) lookup d = d) ∧
    (lookup p = some .other → edgeAttr ofInt (some p) lookup d = d) := by
  refine ⟨rfl, fun f h => ?_, fun i h => ?_, fun h => ?_, fun h => ?_⟩ <;> simp [edgeAttr, h]

/-- `far_endpoint` (:2222). -/
def farEndpoint (from_ src dst : Nat) : Dir → Option Nat
  | .outgoing => if src = from_ then some dst else none
  | .incoming => if dst = from_ then some src else none
  | .both => if src = from_ then some dst else if dst = from_ then some src else none

/-- **PROVEN** (`far_endpoint`): the node an edge leads to when leaving `from_` in the requested
direction; `None` when the edge does not leave `from_` that way. `Both` prefers the source side, so a
self-loop yields its own endpoint. -/
theorem farEndpoint_spec (f s d : Nat) :
    (farEndpoint f s d .outgoing = some d ↔ s = f) ∧ (farEndpoint f s d .outgoing = none ↔ s ≠ f) ∧
    (farEndpoint f s d .incoming = some s ↔ d = f) ∧ (farEndpoint f s d .incoming = none ↔ d ≠ f) ∧
    (s = f → farEndpoint f s d .both = some d) ∧
    (s ≠ f → d = f → farEndpoint f s d .both = some s) ∧
    (s ≠ f → d ≠ f → farEndpoint f s d .both = none) := by
  refine ⟨?_, ?_, ?_, ?_, fun h => ?_, fun h1 h2 => ?_, fun h1 h2 => ?_⟩ <;>
    simp [farEndpoint, *] <;> omega

/-- Every endpoint it returns lies on the edge, and the edge touches `from_`. -/
theorem farEndpoint_incident (f s d far : Nat) (dir : Dir) (h : farEndpoint f s d dir = some far) :
    (s = f ∧ d = far) ∨ (d = f ∧ s = far) := by
  cases dir <;> simp only [farEndpoint] at h <;> split at h <;> (try split at h) <;> simp_all

/-- `node_successors` (:2254): the edges the neighbour iterator yields, with their far endpoint,
those that do not leave `node` in the direction dropped. -/
def successors {E : Type} (dir : Dir) (node : Nat) (rels : List (Nat × Nat × E)) :
    List (E × Nat × Nat × Nat) :=
  rels.filterMap (fun r => (farEndpoint node r.1 r.2.1 dir).map (fun far => (r.2.2, r.1, r.2.1, far)))

/-- **PROVEN** (`node_successors`): `(id, src, dst, far)` is returned exactly when the edge is among
the node's relationships and `far_endpoint` accepts it; order is the iterator's. -/
theorem mem_successors {E : Type} (dir : Dir) (node : Nat) (rels : List (Nat × Nat × E))
    (id : E) (src dst far : Nat) :
    (id, src, dst, far) ∈ successors dir node rels ↔
      (src, dst, id) ∈ rels ∧ farEndpoint node src dst dir = some far := by
  simp only [successors, List.mem_filterMap, Option.map_eq_some_iff]
  constructor
  · rintro ⟨⟨s', d', i'⟩, hm, f', hf, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl, rfl⟩ := he
    exact ⟨hm, hf⟩
  · rintro ⟨hm, hf⟩
    exact ⟨(src, dst, id), hm, far, hf, rfl⟩

theorem successors_length_le {E : Type} (dir : Dir) (node : Nat) (rels : List (Nat × Nat × E)) :
    (successors dir node rels).length ≤ rels.length := List.length_filterMap_le _ _

/-! ## `DijkstraItem` ordering -/

structure Item (F : Type) where
  weight : F
  node : Nat

/-- `DijkstraItem::cmp` (:2289): the max-heap pops the greatest, so the weight comparison is
reversed (lightest first), ties by reversed node id (lowest id first). -/
def itemCmp (pcmp : F → F → Option Ordering) (a b : Item F) : Ordering :=
  (pc pcmp b.weight a.weight).then (compare b.node a.node)

/-- `DijkstraItem::partial_cmp` (:2304) is `Some(cmp)`. -/
def itemPartialCmp (pcmp : F → F → Option Ordering) (a b : Item F) : Option Ordering := some (itemCmp pcmp a b)

/-- **PROVEN** (`DijkstraItem::cmp`/`partial_cmp`): a strictly lighter item is the *greater* one (so it
pops first), equal weights fall back to the smaller node id being greater, and `partial_cmp` is
`cmp` wrapped, as `PartialOrd` requires. -/
theorem itemCmp_spec (pcmp : F → F → Option Ordering) (a b : Item F) :
    (pcmp a.weight b.weight = some .lt → pcmp b.weight a.weight = some .gt → itemCmp pcmp a b = .gt) ∧
    ((pcmp b.weight a.weight = some .eq ∨ pcmp b.weight a.weight = none) →
      itemCmp pcmp a b = compare b.node a.node) ∧
    itemPartialCmp pcmp a b = some (itemCmp pcmp a b) := by
  refine ⟨fun _ h => ?_, fun h => ?_, rfl⟩
  · simp [itemCmp, pc, h, Ordering.then]
  · rcases h with h | h <;> simp [itemCmp, pc, h, Ordering.then]

theorem itemCmp_swap (pcmp : F → F → Option Ordering)
    (hanti : ∀ x y, pcmp y x = (pcmp x y).map Ordering.swap) (a b : Item F) :
    itemCmp pcmp b a = (itemCmp pcmp a b).swap := by
  have hp : ∀ x y, pc pcmp y x = (pc pcmp x y).swap := by
    intro x y
    simp only [pc, hanti x y]
    cases pcmp x y with
    | none => rfl
    | some o => cases o <;> rfl
  have hn : ∀ m n : Nat, compare n m = (compare m n).swap := by
    intro m n
    simp only [compare, compareOfLessAndEq]
    by_cases h1 : m < n <;> by_cases h2 : n < m <;> by_cases h3 : m = n <;> simp [h1, h2, h3, Ordering.swap]
    all_goals omega
  have e1 := hp a.weight b.weight
  have e3 := hn a.node b.node
  simp only [itemCmp, e1, e3, ord_then_swap, Ordering.swap_swap]

/-! ## `build_path_batch` -/

inductive PE (E : Type) where
  | node (n : Nat)
  | rel (e : E)

/-- The alternating node/relationship elements of one path (`build_path_batch`, :2679): the source,
then each edge followed by the node on its other side of the previous one. -/
def pathElems {E : Type} (source : Nat) (edges : List (E × Nat × Nat)) : List (PE E) :=
  .node source :: (go source edges)
where
  go {E : Type} (prev : Nat) : List (E × Nat × Nat) → List (PE E)
  | [] => []
  | (e, src, dst) :: es =>
    let next := if prev = src then dst else src
    .rel e :: .node next :: go next es

/-- The three procedure columns: `path`, `pathWeight`, `pathCost`. -/
def buildBatch {E : Type} (o : FloatOps F) (source : Nat) (results : List (Found E F)) :
    List (List (PE E)) × List (Num F) × List (Num F) :=
  (results.map (fun r => pathElems source r.edges), results.map (fun r => toNumeric o r.weight),
   results.map (fun r => toNumeric o r.cost))

/-- **PROVEN** (`build_path_batch`): one row per result, in order; each path has `2·hops + 1`
elements, starts at the source, and weight/cost columns are `to_numeric_value` of the result's. -/
theorem buildBatch_spec {E : Type} (o : FloatOps F) (source : Nat) (results : List (Found E F)) :
    (buildBatch o source results).1.length = results.length ∧
    (buildBatch o source results).2.1.length = results.length ∧
    (buildBatch o source results).2.2.length = results.length ∧
    (∀ r ∈ results, (pathElems source r.edges).length = 2 * r.edges.length + 1 ∧
        (pathElems source r.edges).head? = some (.node source)) ∧
    (buildBatch o source results).2.1 = results.map (fun r => toNumeric o r.weight) ∧
    (buildBatch o source results).2.2 = results.map (fun r => toNumeric o r.cost) := by
  have hlen : ∀ {E : Type} (prev : Nat) (es : List (E × Nat × Nat)),
      (pathElems.go prev es).length = 2 * es.length := by
    intro E prev es
    induction es generalizing prev with
    | nil => rfl
    | cons e es ih => simp [pathElems.go, ih]; omega
  refine ⟨by simp [buildBatch], by simp [buildBatch], by simp [buildBatch], fun r _ => ⟨?_, rfl⟩, rfl, rfl⟩
  simp [pathElems, hlen]

/-! ## Registration -/

/-- A registered procedure: name, arguments, output columns, and body. -/
structure Registered (R : Type) where
  name : String
  nargs : Nat
  cols : List String
  body : List (CV F) → Except String R

/-- `register_sp_paths` (:2761): `algo.SPpaths`, one `Any` argument, columns
`path, pathWeight, pathCost`; body = `parse_sp_config(args)?` then `run_path_algo`. -/
def spReg {R : Type} (ofInt : Int → F) (parseConfig : List (CV F) → Except String (String → Option (CV F)))
    (run : Cfg F → Except String R) : Registered (F := F) R :=
  ⟨"algo.SPpaths", 1, ["path", "pathWeight", "pathCost"],
   fun args => (parseSp ofInt (parseConfig args)).bind run⟩

/-- `register_ss_paths` (:2773). -/
def ssReg {R : Type} (ofInt : Int → F) (parseConfig : List (CV F) → Except String (String → Option (CV F)))
    (run : Cfg F → Except String R) : Registered (F := F) R :=
  ⟨"algo.SSpaths", 1, ["path", "pathWeight", "pathCost"],
   fun args => (parseSs ofInt (parseConfig args)).bind run⟩

/-- **PROVEN** (`register_sp_paths`, `register_ss_paths`): the two registrations differ only in
name and config parser; both take one argument and yield the same three columns. -/
theorem reg_spec {R : Type} (ofInt : Int → F)
    (parseConfig : List (CV F) → Except String (String → Option (CV F))) (run : Cfg F → Except String R) :
    (spReg ofInt parseConfig run).name = "algo.SPpaths" ∧ (ssReg ofInt parseConfig run).name = "algo.SSpaths" ∧
    (spReg ofInt parseConfig run).nargs = 1 ∧ (ssReg ofInt parseConfig run).nargs = 1 ∧
    (spReg ofInt parseConfig run).cols = ["path", "pathWeight", "pathCost"] ∧
    (ssReg ofInt parseConfig run).cols = (spReg ofInt parseConfig run).cols ∧
    (∀ args, (spReg ofInt parseConfig run).body args = (parseSp ofInt (parseConfig args)).bind run) ∧
    (∀ args, (ssReg ofInt parseConfig run).body args = (parseSs ofInt (parseConfig args)).bind run) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, fun _ => rfl, fun _ => rfl⟩

end OpsTraverse.AlgoPaths
