import AlgoUdf.Nesting
/-! # UDF value marshalling (`graph/src/udf/type_convert.rs`) — model, @ origin/main cab3cf192

| here | there |
| --- | --- |
| `F64`                 | f64 as `==`/`floor`/`abs`/`is_finite` see it |
| `JsV`, `Heap`, `deref` | the QuickJS values `js_to_value` distinguishes; an array/object may also be a `ref` into the QuickJS heap, so shared and *cyclic* values (`a.push(a)`) are representable |
| `objSet`, `jsKeys`    | AXIOMATISED QuickJS object semantics (ECMA-262 §10.1.9 [[Set]] through the `Object.prototype.__proto__` accessor, §10.1.11 OrdinaryOwnPropertyKeys: array-index keys first, ascending) |
| `JsKind`              | the object's internal class, as `JS_IsDate` / `JS_IsRegExp` test it (QuickJS FFI, AXIOMATISED by construction) |
| `ctorName`            | historical only: `obj.get("constructor").name`, the pre-#3074 test |
| `RV`                  | `runtime::value::Value` (all variants; Path as node/rel id lists) |
| `MAX`, `nestErr`, `chk` (`Nesting.lean`) | `MAX_UDF_VALUE_NESTING` :67, `udf_nesting_error` :87, `depth >= MAX_UDF_VALUE_NESTING` |
| `toLevel`, `toStep`, `toF` | `value_to_js_level` :124-249, `value_to_js_at` :110-122 (`f` = stack frames) |
| `toJs`                | `value_to_js` :97 (= `value_to_js_at(.., 0)`) |
| `classify`, `runShape`, `pathArm`, `pathLoop`, `fromLevel`, `fromStep`, `fromF` | `js_to_value_level` :270-470 (dispatch / recursive arms; array :319, path :359, plain object :443), `js_to_value_at` :262-267 |
| `fromJs`              | `js_to_value` :252 (= `js_to_value_at(.., 0)`) |
| `esc`, `unesc`        | key escaping :171-177 / :458-462 |
| `pre3336_fromF`       | historical: `js_to_value` before #3336 (`33a956360`): no depth check (`cap = none`) |

`with_stack` (:80, `stacker::maybe_grow`) only switches the machine stack; `f` counts conversion
levels, which is what the depth cap bounds (`MarshalCap.lean`).
-/
namespace AlgoUdf.Marshal

/-- f64 abstraction: `integ k` is a nonzero integral finite value, `frac` a
non-integral finite one (opaque), zeros and infinities carry their sign. -/
inductive F64 | nan | inf (neg : Bool) | zero (neg : Bool) | integ (k : Int) | frac (tag : Nat)
  deriving DecidableEq, Repr

def two53 : Int := 9007199254740992

/-- `f == f.floor()` -/
def F64.integral : F64 → Bool
  | .nan | .frac _ => false
  | _ => true
/-- `f.abs() < 2^53` -/
def F64.small : F64 → Bool
  | .zero _ => true
  | .integ k => decide (k.natAbs < two53.toNat)
  | _ => false
/-- `f == 0.0 && f.is_sign_negative()` -/
def F64.negZero : F64 → Bool
  | .zero true => true
  | _ => false
def F64.finite : F64 → Bool
  | .nan | .inf _ => false
  | _ => true
/-- `f as i64` on an integral small value. -/
def F64.toInt : F64 → Int
  | .integ k => k
  | _ => 0
/-- `i as f64` for `|i| < 2^53` (exact). -/
def ofInt (i : Int) : F64 := if i = 0 then .zero false else .integ i

inductive JsKind | plain | date (ms : F64) | regexp (src : String)
  deriving DecidableEq, Repr

inductive JsV where
  | null | undef
  | bool (b : Bool)
  | num (f : F64)
  | big (i : Int)
  | str (s : String)
  | sym
  | arr (items : List JsV) (props : List (List Char × JsV))
  | obj (props : List (List Char × JsV)) (kind : JsKind)
  /-- a reference to an array / object in the QuickJS heap -/
  | ref (a : Nat)
  deriving Repr, Inhabited

/-- The QuickJS heap: what each object reference points to. -/
abbrev Heap := Nat → JsV
/-- Follow one reference (a heap cell holds an array or object, never another reference). -/
def deref (H : Heap) : JsV → JsV
  | .ref a => H a
  | v => v
/-- The heap of a value with no references (a tree). -/
def H0 : Heap := fun _ => .undef

def get (ps : List (List Char × JsV)) (k : List Char) : Option JsV := (ps.find? (·.1 == k)).map (·.2)

/-- QuickJS `obj.set(k, v)` on an ordinary object: `"__proto__"` goes through the
inherited accessor (sets the prototype or is ignored), never an own key. -/
def objSet (ps : List (List Char × JsV)) (k : List Char) (v : JsV) : List (List Char × JsV) :=
  if k = "__proto__".toList then ps else ps ++ [(k, v)]

def isIndexKey (s : List Char) : Bool := !s.isEmpty && s.all Char.isDigit

/-- `obj.keys()`: array-index keys first, then the rest in insertion order
(the ascending sort of index keys is not modelled). -/
def jsKeys (ps : List (List Char × JsV)) : List (List Char) :=
  let ks := ps.map (·.1)
  ks.filter isIndexKey ++ ks.filter (!isIndexKey ·)

def JsKind.ctorName : JsKind → String
  | .plain => "Object" | .date _ => "Date" | .regexp _ => "RegExp"

/-- `obj.get::<Object>("constructor")` then `.get::<String>("name")`. -/
def ctorName (ps : List (List Char × JsV)) (kind : JsKind) : Option String :=
  match get ps "constructor".toList with
  | none => some kind.ctorName
  | some (.obj cps _) => match get cps "name".toList with
    | some (.str n) => some n
    | _ => none
  | some _ => none

def pre : List Char := "__falkor_".toList
def escPre : List Char := "__falkor_esc_".toList
def esc (k : List Char) : List Char := if pre.isPrefixOf k then escPre ++ k else k
def unesc (k : List Char) : List Char := if escPre.isPrefixOf k then k.drop 13 else k
def keep (k : List Char) : Bool := !pre.isPrefixOf k || escPre.isPrefixOf k

inductive RV where
  | null | bool (b : Bool) | int (i : Int) | float (f : F64) | str (s : String)
  | list (xs : List RV) | map (kv : List (List Char × RV))
  | node (id : Nat) | rel (id : Nat) | path (nodes rels : List Nat)
  | point (lat lon : F64) | datetime (ts : Int) | date (ts : Int)
  | vecf32 (xs : List F64) | time | duration
  deriving Repr, Inhabited

def i64max : Int := 9223372036854775807
def jsDateMax : Int := 8640000000000000

/-- `new Date(ms)`: time value is NaN outside ±8.64e15 ms (ECMA-262 TimeClip). -/
def mkDate (ms : Int) : F64 := if ms.natAbs ≤ jsDateMax.toNat then ofInt ms else .nan

def nodeObj (id : Nat) : JsV := .obj [("__falkor_type".toList, .str "node"), ("__falkor_node_id".toList, .num (ofInt id))] .plain
def relObj (id : Nat) : JsV := .obj [("__falkor_type".toList, .str "edge"), ("__falkor_edge_id".toList, .num (ofInt id))] .plain

/-- `obj.set(esc(key), js_val)` for each converted map entry (:171-180), in order. -/
def assembleObj : List (List Char × RV) → List JsV → List (List Char × JsV)
  | (k, _) :: kvs, j :: js => if esc k = "__proto__".toList then assembleObj kvs js else (esc k, j) :: assembleObj kvs js
  | _, _ => []

/-- A loop converting each element in order, stopping at the first error or overflow. -/
def listLoop {α β} (conv : α → R β) : List α → R (List β)
  | [] => R.ok []
  | x :: xs => R.bind (conv x) fun v => R.bind (listLoop conv xs) fun vs => R.ok (v :: vs)

/-- `value_to_js_level` (:124-249); `conv` converts a child (`value_to_js_at(.., depth + 1)`). -/
def toLevel (conv : RV → R JsV) : RV → R JsV
  | .null => R.ok .null
  | .bool b => R.ok (.bool b)
  | .int i => R.ok (if i.natAbs < two53.toNat then .num (ofInt i) else .big i)
  | .float f => R.ok (.num f)
  | .str s => R.ok (.str s)
  | .list xs => R.bind (listLoop conv xs) fun js => R.ok (.arr js [])
  | .map kv => R.bind (listLoop conv (kv.map (·.2))) fun js => R.ok (.obj (assembleObj kv js) .plain)
  | .node id => R.ok (nodeObj id)
  | .rel id => R.ok (relObj id)
  | .path ns rs => R.ok (.obj [("__falkor_type".toList, .str "path"),
      ("nodes".toList, .arr (ns.map nodeObj) []), ("relationships".toList, .arr (rs.map relObj) [])] .plain)
  | .point a b => R.ok (.obj [("__falkor_type".toList, .str "point"), ("latitude".toList, .num a), ("longitude".toList, .num b)] .plain)
  | .datetime ts => if (ts * 1000).natAbs > i64max.toNat then R.err "i64 overflow (debug panic)"
      else R.ok (.obj [("__falkor_temporal_type".toList, .str "datetime")] (.date (mkDate (ts * 1000))))
  | .date ts => if (ts * 1000).natAbs > i64max.toNat then R.err "i64 overflow (debug panic)"
      else R.ok (.obj [("__falkor_temporal_type".toList, .str "date")] (.date (mkDate (ts * 1000))))
  | .vecf32 xs => R.ok (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")])
  | .time | .duration => R.err "UDF Exception: unsupported type for JS conversion"

/-- `matches!(value, Value::List(_) | Value::Map(_))` (:117). -/
def needsV : RV → Bool
  | .list _ | .map _ => true
  | _ => false

/-- The children `value_to_js_level` converts at `depth + 1`. -/
def kidsV : RV → List RV
  | .list xs => xs
  | .map kv => kv.map (·.2)
  | _ => []

/-- `value_to_js_at` at depth `d` with `f` frames left (:110-122). -/
def toF (cap : Option Nat) : Nat → Nat → RV → R JsV
  | 0, _, _ => none
  | f + 1, d, v => if needsV v && chk cap d then R.err nestErr else toLevel (toF cap f (d + 1)) v

/-- `value_to_js` (:97): `value_to_js_at(.., 0)`; `MarshalCap.toJs_frames` shows `MAX + 1` frames
    always suffice, so the `none` arm is unreachable. -/
def toJs (v : RV) : Except String JsV := Rget (toF (some MAX) (MAX + 1) 0 v)

/-- u64 read of a marker id (`obj.get::<u64>`): a non-negative integral number. -/
def asU64 : JsV → Option Nat
  | .num (.zero _) => some 0
  | .num (.integ k) => if 0 ≤ k then some k.toNat else none
  | _ => none

def numOf : JsV → Option F64
  | .num f => some f
  | _ => none

/-- The non-marker object branch (:403-440): Date / RegExp by internal class
(`JS_IsDate`, `JS_IsRegExp`), never by a `constructor` property; anything else is a
map. `props` is `fromProps ps` (only used on the Map path). -/
def fromObjWith (ps : List (List Char × JsV)) (kind : JsKind) (props : Except String (List (List Char × RV))) : Except String RV :=
  match kind with
  | .date ms => if ms.finite then
      (let secs := ms.toInt.tdiv 1000  -- `(ms / 1000.0) as i64` on integral ms, |ms| ≤ 8.64e15
       match get ps "__falkor_temporal_type".toList with
       | some (.str "date") => .ok (.date secs)
       | _ => .ok (.datetime secs))
    else .error "Invalid Date value"
  | .regexp s => .ok (.str s)
  | .plain => do let kv ← props; pure (.map kv)

/-- vecf32 arm (:324-336): every element converts (`arr.get::<f64>`, then `as f32`;
the abstract `F64` already stands for the f32 value), inf / NaN included since #3074. -/
def fromVec : List JsV → Except String RV
  | [] => .ok (.vecf32 [])
  | .num f :: xs => do
      let r ← fromVec xs
      match r with | .vecf32 t => pure (.vecf32 (f :: t)) | _ => .error "unreachable"
  | _ :: _ => .error "VecF32 item error"

/-- `obj.get::<_, String>("__falkor_type")`. -/
def tag (ps : List (List Char × JsV)) : Option String :=
  match get ps "__falkor_type".toList with
  | some (.str t) => some t
  | _ => none

/-- How `js_to_value_level` treats a (dereferenced) value: `done` = no recursion and no depth check;
    the other four are the arms that check the depth and then recurse. -/
inductive Shape where
  | done (r : Except String RV)
  | list (items : List JsV)
  | vec (items : List JsV)
  | path (ps : List (List Char × JsV))
  | map (ps : List (List Char × JsV))

/-- The branch structure of `js_to_value_level` (:270-470) before any recursion. -/
def classify : JsV → Shape
  | .null | .undef => .done (.ok .null)
  | .bool b => .done (.ok (.bool b))
  | .num f => .done (.ok (if f.integral && f.small && !f.negZero then .int f.toInt else .float f))
  | .big i => .done (if i.natAbs ≤ i64max.toNat then .ok (.int i) else .error "BigInt out of i64 range")
  | .str s => .done (.ok (.str s))
  | .sym => .done (.error "Symbol values are not supported")
  | .arr items ps => if tag ps = some "vecf32" then .vec items else .list items
  | .obj ps kind =>
      if tag ps = some "node" then
        .done (match (get ps "__falkor_node_id".toList).bind asU64 with
          | some id => .ok (.node id) | none => .error "bad node id")
      else if tag ps = some "edge" then
        .done (match (get ps "__falkor_edge_id".toList).bind asU64 with
          | some id => .ok (.rel id) | none => .error "bad edge id")
      else if tag ps = some "path" then .path ps
      else if tag ps = some "point" then
        .done (match (get ps "latitude".toList).bind numOf, (get ps "longitude".toList).bind numOf with
          | some a, some b => .ok (.point a b) | _, _ => .error "bad point")
      else match kind with
        | .plain => .map ps
        | kind => .done (fromObjWith ps kind (.ok []))
  | .ref _ => .done (.error "Unsupported JS value type")

def asNode : RV → Option Nat
  | .node id => some id
  | _ => none
def asRel : RV → Option Nat
  | .rel id => some id
  | _ => none

/-- The path loop (:377-396): node `i` (must be a `Node`), then relationship `i` (must be a
    `Relationship`) while `i < n_rels`. -/
def pathLoop (conv : JsV → R RV) : List JsV → List JsV → R (List Nat × List Nat)
  | [], _ => R.ok ([], [])
  | n :: ns, rs => R.bind (conv n) fun v =>
      match asNode v with
      | none => R.err "Invalid path: element at node position is not a Node"
      | some id => match rs with
        | [] => R.bind (pathLoop conv ns []) fun p => R.ok (id :: p.1, p.2)
        | r :: rs' => R.bind (conv r) fun w =>
          match asRel w with
          | none => R.err "Invalid path: element at relationship position is not a Relationship"
          | some rid => R.bind (pathLoop conv ns rs') fun p => R.ok (id :: p.1, rid :: p.2)

/-- `obj.get::<Array>(k)`: the array's items. -/
def getArr (H : Heap) (ps : List (List Char × JsV)) (k : String) : Option (List JsV) :=
  match (get ps k.toList).map (deref H) with
  | some (.arr items _) => some items
  | _ => none

/-- The `"path"` arm after its depth check (:363-397). -/
def pathArm (H : Heap) (conv : JsV → R RV) (ps : List (List Char × JsV)) : R RV :=
  match getArr H ps "nodes", getArr H ps "relationships" with
  | some ns, some rs =>
      if rs.length = ns.length - 1 then R.bind (pathLoop conv ns rs) fun p => R.ok (.path p.1 p.2)
      else R.err "Invalid path: expected n_nodes - 1 relationships"
  | none, _ => R.err "path nodes error"
  | some _, none => R.err "path relationships error"

/-- The kept keys of a plain object (:447-451). -/
def keptProps (ps : List (List Char × JsV)) : List (List Char × JsV) := ps.filter (keep ·.1)

/-- The recursive arms, each behind its depth check `b` (:320, :361, :443). -/
def runShape (H : Heap) (b : Bool) (conv : JsV → R RV) : Shape → R RV
  | .done r => some r
  | .list items => if b then R.err nestErr else R.bind (listLoop conv items) fun xs => R.ok (.list xs)
  | .vec items => if b then R.err nestErr else some (fromVec items)
  | .path ps => if b then R.err nestErr else pathArm H conv ps
  | .map ps => if b then R.err nestErr else
      R.bind (listLoop conv ((keptProps ps).map (·.2))) fun xs =>
        R.ok (.map (((keptProps ps).map (unesc ·.1)).zip xs))

/-- `js_to_value_level(val, depth)` with `b = depth >= MAX_UDF_VALUE_NESTING`. -/
def fromLevel (H : Heap) (b : Bool) (conv : JsV → R RV) (v : JsV) : R RV :=
  runShape H b conv (classify (deref H v))

/-- `js_to_value_at` at depth `d` with `f` frames left (:262-267). -/
def fromF (H : Heap) (cap : Option Nat) : Nat → Nat → JsV → R RV
  | 0, _, _ => none
  | f + 1, d, v => fromLevel H (chk cap d) (fromF H cap f (d + 1)) v

/-- `js_to_value` (:252): `js_to_value_at(val, 0)`; `MarshalCap.fromJs_frames` shows `MAX + 1`
    frames always suffice (for any heap, cyclic ones included), so the `none` arm is unreachable. -/
def fromJs (H : Heap) (v : JsV) : Except String RV := Rget (fromF H (some MAX) (MAX + 1) 0 v)

/-- HISTORICAL (before #3336, `33a956360`): `js_to_value` recursed with no depth check. -/
def pre3336_fromF (H : Heap) (f : Nat) (v : JsV) : R RV := fromF H none f 0 v

end AlgoUdf.Marshal
