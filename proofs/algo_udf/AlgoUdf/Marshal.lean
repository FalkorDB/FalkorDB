/-! # UDF value marshalling (`graph/src/udf/type_convert.rs`) — model

| here | there |
| --- | --- |
| `F64`                 | f64 as `==`/`floor`/`abs`/`is_finite` see it |
| `JsV`                 | the QuickJS values `js_to_value` distinguishes |
| `objSet`, `jsKeys`    | AXIOMATISED QuickJS object semantics (ECMA-262 §10.1.9 [[Set]] through the `Object.prototype.__proto__` accessor, §10.1.11 OrdinaryOwnPropertyKeys: array-index keys first, ascending) |
| `JsKind`              | the object's internal class, as `JS_IsDate` / `JS_IsRegExp` test it (QuickJS FFI, AXIOMATISED by construction) |
| `ctorName`            | historical only: `obj.get("constructor").name`, the pre-#3074 test |
| `RV`                  | `runtime::value::Value` (all variants; Path as node/rel id lists) |
| `toJs`                | `value_to_js` :58 |
| `fromJs`              | `js_to_value` :185-371 @ 8743953a8 (-0.0 :203, vecf32 :230-244, Date/RegExp by class :313-345) |
| `esc`, `unesc`        | key escaping :104-109 / :348-366 |
| `pre3074_*`           | historical: `js_to_value` before #3074 (`7a81c83b0`) |
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
  deriving Repr, Inhabited

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

mutual
def toJs : RV → Except String JsV
  | .null => .ok .null
  | .bool b => .ok (.bool b)
  | .int i => .ok (if i.natAbs < two53.toNat then .num (ofInt i) else .big i)
  | .float f => .ok (.num f)
  | .str s => .ok (.str s)
  | .list xs => do let js ← toJsList xs; pure (.arr js [])
  | .map kv => do let ps ← toJsMap kv; pure (.obj ps .plain)
  | .node id => .ok (nodeObj id)
  | .rel id => .ok (relObj id)
  | .path ns rs => .ok (.obj [("__falkor_type".toList, .str "path"),
      ("nodes".toList, .arr (ns.map nodeObj) []), ("relationships".toList, .arr (rs.map relObj) [])] .plain)
  | .point a b => .ok (.obj [("__falkor_type".toList, .str "point"), ("latitude".toList, .num a), ("longitude".toList, .num b)] .plain)
  | .datetime ts => if (ts * 1000).natAbs > i64max.toNat then .error "i64 overflow (debug panic)"
      else .ok (.obj [("__falkor_temporal_type".toList, .str "datetime")] (.date (mkDate (ts * 1000))))
  | .date ts => if (ts * 1000).natAbs > i64max.toNat then .error "i64 overflow (debug panic)"
      else .ok (.obj [("__falkor_temporal_type".toList, .str "date")] (.date (mkDate (ts * 1000))))
  | .vecf32 xs => .ok (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")])
  | .time | .duration => .error "UDF Exception: unsupported type for JS conversion"
def toJsList : List RV → Except String (List JsV)
  | [] => .ok []
  | x :: xs => do let j ← toJs x; let js ← toJsList xs; pure (j :: js)
def toJsMap : List (List Char × RV) → Except String (List (List Char × JsV))
  | [] => .ok []
  | (k, v) :: kvs => do
      let j ← toJs v; let ps ← toJsMap kvs
      pure (if esc k = "__proto__".toList then ps else (esc k, j) :: ps)
end

/-- u64 read of a marker id (`obj.get::<u64>`): a non-negative integral number. -/
def asU64 : JsV → Option Nat
  | .num (.zero _) => some 0
  | .num (.integ k) => if 0 ≤ k then some k.toNat else none
  | _ => none

def numOf : JsV → Option F64
  | .num f => some f
  | _ => none

/-- The non-marker object branch (:313-345): Date / RegExp by internal class
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

/-- HISTORICAL (before #3074, `7a81c83b0`): Date / RegExp chosen by
`obj.get("constructor").name`, which a plain object can carry as an own key. -/
def pre3074_fromObjWith (ps : List (List Char × JsV)) (kind : JsKind) (props : Except String (List (List Char × RV))) : Except String RV :=
  match ctorName ps kind with
  | some "Date" => match kind with
    | .date ms => if ms.finite then
        (let secs := ms.toInt.tdiv 1000
         match get ps "__falkor_temporal_type".toList with
         | some (.str "date") => .ok (.date secs)
         | _ => .ok (.datetime secs))
      else .error "Invalid Date value"
    | _ => .error "Date getTime error"
  | some "RegExp" => match kind with
    | .regexp s => .ok (.str s)
    | _ => .ok (.str "[object Object]")  -- Object.prototype.toString
  | _ => do let kv ← props; pure (.map kv)

/-- HISTORICAL (before #3074): the number arm turned -0.0 into Int 0. -/
def pre3074_num (f : F64) : RV := if f.integral && f.small then .int f.toInt else .float f

/-- HISTORICAL (before #3074): vecf32 elements had to be finite. -/
def pre3074_fromVec : List JsV → Except String RV
  | [] => .ok (.vecf32 [])
  | .num f :: xs => if f.finite then (do
        let r ← pre3074_fromVec xs
        match r with | .vecf32 t => pure (.vecf32 (f :: t)) | _ => .error "unreachable")
      else .error "VecF32 element is not finite"
  | _ :: _ => .error "VecF32 item error"

/-- vecf32 arm (:236-244): every element converts (`arr.get::<f64>`, then `as f32`;
the abstract `F64` already stands for the f32 value), inf / NaN included since #3074. -/
def fromVec : List JsV → Except String RV
  | [] => .ok (.vecf32 [])
  | .num f :: xs => do
      let r ← fromVec xs
      match r with | .vecf32 t => pure (.vecf32 (f :: t)) | _ => .error "unreachable"
  | _ :: _ => .error "VecF32 item error"

mutual
def fromJs : JsV → Except String RV
  | .null | .undef => .ok .null
  | .bool b => .ok (.bool b)
  | .num f => .ok (if f.integral && f.small && !f.negZero then .int f.toInt else .float f)
  | .big i => if i.natAbs ≤ i64max.toNat then .ok (.int i) else .error "BigInt out of i64 range"
  | .str s => .ok (.str s)
  | .sym => .error "Symbol values are not supported"
  | .arr items ps =>
      match get ps "__falkor_type".toList with
      | some (.str "vecf32") => fromVec items
      | _ => do let xs ← fromJsList items; pure (.list xs)
  | .obj ps kind =>
      match get ps "__falkor_type".toList with
      | some (.str "node") => match (get ps "__falkor_node_id".toList).bind asU64 with
        | some id => .ok (.node id) | none => .error "bad node id"
      | some (.str "edge") => match (get ps "__falkor_edge_id".toList).bind asU64 with
        | some id => .ok (.rel id) | none => .error "bad edge id"
      | some (.str "point") => match (get ps "latitude".toList).bind numOf, (get ps "longitude".toList).bind numOf with
        | some a, some b => .ok (.point a b) | _, _ => .error "bad point"
      | _ => fromObjWith ps kind (fromProps ps)
def fromJsList : List JsV → Except String (List RV)
  | [] => .ok []
  | x :: xs => do let v ← fromJs x; let vs ← fromJsList xs; pure (v :: vs)
def fromProps : List (List Char × JsV) → Except String (List (List Char × RV))
  | [] => .ok []
  | (k, v) :: ps => if keep k then (do
        let x ← fromJs v; let r ← fromProps ps; pure ((unesc k, x) :: r))
      else fromProps ps
end

end AlgoUdf.Marshal
