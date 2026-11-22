import AlgoUdf.Nesting
/-! # UDF value marshalling (`graph/src/udf/type_convert.rs`) — model, @ origin/main cab3cf192

| here | there |
| --- | --- |
| `F64`                 | f64 as `==`/`floor`/`abs`/`is_finite` see it |
| `JsV`, `Heap`, `deref` | the QuickJS values `js_to_value` distinguishes; an array/object may also be a `ref` into the QuickJS heap, so shared and *cyclic* values (`a.push(a)`) are representable |
| `objSet`, `jsKeys`    | AXIOMATISED QuickJS object semantics (ECMA-262 §10.1.9 [[Set]] through the `Object.prototype.__proto__` accessor, §10.1.11 OrdinaryOwnPropertyKeys: array-index keys first, ascending) |
| `JsKind`              | the object's internal class, as `JS_IsDate` / `JS_IsRegExp` test it (QuickJS FFI, AXIOMATISED by construction). `date ms` / `regexp s` carry the value Rust then obtains by *calling the object's `getTime` / `toString` properties* (`obj.get("getTime")`, :412-438): those calls are an ORACLE here, so a Date or RegExp whose `getTime` / `toString` the script overrides (an own property) is outside the model: Rust returns whatever the override returns (or an `Err` if it is not a function / not a number), the model returns the internal time value / source |
| `ctorName`            | historical only: `obj.get("constructor").name`, the pre-#3074 test |
| `RV`                  | `runtime::value::Value` (all variants; Path as node/rel id lists) |
| `MAX`, `nestErr`, `chk` (`Nesting.lean`) | `MAX_UDF_VALUE_NESTING` :68, `udf_nesting_error` :88, `depth >= MAX_UDF_VALUE_NESTING` |
| `toLevel`, `toStep`, `toF` | `value_to_js_level` :125-250, `value_to_js_at` :111-122 (`f` = stack frames) |
| `toJs`                | `value_to_js` :98 (= `value_to_js_at(.., 0)`) |
| `classify`, `classifyObj`, `runShape`, `fromLevel`, `fromStep`, `fromF` | `js_to_value_level` :271-425 (dispatch / recursive arms; array :321, entity class :352, plain object :399), `js_to_value_at` :263-268 |
| `JsV.ent`             | an instance of `js_classes::JsEntity` (#2976): `entity_value` reads the entity back, only in the call that made it |
| `fromJs`              | `js_to_value` :253 (= `js_to_value_at(.., 0)`) |
| `esc`, `unesc`        | key escaping :172-178 / :413-417 |

`with_stack` (:81, `stacker::maybe_grow`) only switches the machine stack; `f` counts conversion
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

/-- `date ms` / `regexp src`: `ms` is what `getTime()` returns and `src` what `toString()` returns
(Rust calls the object's own methods; see the header: an overridden method is not modelled). -/
inductive JsKind | plain | date (ms : F64) | regexp (src : String)
  deriving DecidableEq, Repr

inductive RV where
  | null | bool (b : Bool) | int (i : Int) | float (f : F64) | str (s : String)
  | list (xs : List RV) | map (kv : List (List Char × RV))
  | node (id : Nat) | rel (id : Nat) | path (nodes rels : List Nat)
  | point (lat lon : F64) | datetime (ts : Int) | date (ts : Int)
  | vecf32 (xs : List F64)  -- the f32 elements widened by `v as f64`; `item as f32` narrows them back exactly
  | time | duration
  deriving Repr, Inhabited

inductive JsV where
  | null | undef
  | bool (b : Bool)
  | num (f : F64)
  | big (i : Int)
  | str (s : String)
  | sym
  | arr (items : List JsV) (props : List (List Char × JsV))
  | obj (props : List (List Char × JsV)) (kind : JsKind)
  /-- an instance of the Rust class `JsEntity` (js_classes.rs :84-136, #2976): what
  `create_js_node` / `create_js_edge` / `create_js_path` return. `props` are its visible own
  properties, `v` the entity in its class data, `current` whether its epoch is the current
  call's (`entity_value` :129 returns `Some v` exactly then) -/
  | ent (props : List (List Char × JsV)) (v : RV) (current : Bool)
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

def i64max : Int := 9223372036854775807
def i64min : Int := -9223372036854775808

/-- Two's-complement wrap of an unbounded integer into `i64` (what `*ts * 1000` does when the
workspace release profile, which sets no `overflow-checks`, overflows; a debug build panics). -/
def wrapI64 (x : Int) : Int := (x + 9223372036854775808) % 18446744073709551616 - 9223372036854775808
def jsDateMax : Int := 8640000000000000

/-- `new Date(ms)`: time value is NaN outside ±8.64e15 ms (ECMA-262 TimeClip). -/
def mkDate (ms : Int) : F64 := if ms.natAbs ≤ jsDateMax.toNat then ofInt ms else .nan

/-- `create_js_node` / `create_js_edge` (inside the call that converts it back, so `current`). -/
def nodeObj (id : Nat) : JsV := .ent [("__falkor_type".toList, .str "node"), ("__falkor_node_id".toList, .num (ofInt id))] (.node id) true
def relObj (id : Nat) : JsV := .ent [("__falkor_type".toList, .str "edge"), ("__falkor_edge_id".toList, .num (ofInt id))] (.rel id) true

/-- `obj.set(esc(key), js_val)` for each converted map entry (:172-181), in order. -/
def assembleObj : List (List Char × RV) → List JsV → List (List Char × JsV)
  | (k, _) :: kvs, j :: js => if esc k = "__proto__".toList then assembleObj kvs js else (esc k, j) :: assembleObj kvs js
  | _, _ => []

/-- A loop converting each element in order, stopping at the first error or overflow. -/
def listLoop {α β} (conv : α → R β) : List α → R (List β)
  | [] => R.ok []
  | x :: xs => R.bind (conv x) fun v => R.bind (listLoop conv xs) fun vs => R.ok (v :: vs)

/-- `value_to_js_level` (:125-250); `conv` converts a child (`value_to_js_at(.., depth + 1)`). -/
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
  | .path ns rs => R.ok (.ent [("__falkor_type".toList, .str "path"),
      ("nodes".toList, .arr (ns.map nodeObj) []), ("relationships".toList, .arr (rs.map relObj) [])] (.path ns rs) true)
  | .point a b => R.ok (.obj [("__falkor_type".toList, .str "point"), ("latitude".toList, .num a), ("longitude".toList, .num b)] .plain)
  | .datetime ts => R.ok (.obj [("__falkor_temporal_type".toList, .str "datetime")] (.date (mkDate (wrapI64 (ts * 1000)))))
  | .date ts => R.ok (.obj [("__falkor_temporal_type".toList, .str "date")] (.date (mkDate (wrapI64 (ts * 1000)))))
  | .vecf32 xs => R.ok (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")])
  | .time | .duration => R.err "UDF Exception: unsupported type for JS conversion"

/-- `matches!(value, Value::List(_) | Value::Map(_))` (:118). -/
def needsV : RV → Bool
  | .list _ | .map _ => true
  | _ => false

/-- The children `value_to_js_level` converts at `depth + 1`. -/
def kidsV : RV → List RV
  | .list xs => xs
  | .map kv => kv.map (·.2)
  | _ => []

/-- `value_to_js_at` at depth `d` with `f` frames left (:111-122). -/
def toF (cap : Option Nat) : Nat → Nat → RV → R JsV
  | 0, _, _ => none
  | f + 1, d, v => if needsV v && chk cap d then R.err nestErr else toLevel (toF cap f (d + 1)) v

/-- `value_to_js` (:98): `value_to_js_at(.., 0)`; `MarshalCap.toJs_frames` shows `MAX + 1` frames
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

/-- The non-marker object branch (:365-397): Date / RegExp by internal class
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

/-- vecf32 arm (:325-337): every element converts (`arr.get::<f64>`, then `as f32`;
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
    the other three are the arms that check the depth and then recurse. -/
inductive Shape where
  | done (r : Except String RV)
  | list (items : List JsV)
  | vec (items : List JsV)
  | map (ps : List (List Char × JsV))

/-- A non-array object that is not a current entity (:356-424): a `"point"` marker, else Date /
    RegExp by internal class, else a map. Since #2976 a `"node"` / `"edge"` / `"path"` marker on a
    plain object no longer makes it an entity. -/
def classifyObj (ps : List (List Char × JsV)) (kind : JsKind) : Shape :=
  if tag ps = some "point" then
    .done (match (get ps "latitude".toList).bind numOf, (get ps "longitude".toList).bind numOf with
      | some a, some b => .ok (.point a b) | _, _ => .error "bad point")
  else match kind with
    | .plain => .map ps
    | kind => .done (fromObjWith ps kind (.ok []))

/-- The branch structure of `js_to_value_level` (:271-421) before any recursion. -/
def classify : JsV → Shape
  | .null | .undef => .done (.ok .null)
  | .bool b => .done (.ok (.bool b))
  | .num f => .done (.ok (if f.integral && f.small && !f.negZero then .int f.toInt else .float f))
  | .big i => .done (if i64min ≤ i ∧ i ≤ i64max then .ok (.int i) else .error "BigInt out of i64 range")
  | .str s => .done (.ok (.str s))
  | .sym => .done (.error "Symbol values are not supported")
  | .arr items ps => if tag ps = some "vecf32" then .vec items else .list items
  | .obj ps kind => classifyObj ps kind
  | .ent ps v current => if current then .done (.ok v) else classifyObj ps .plain
  | .ref _ => .done (.error "Unsupported JS value type")

/-- The kept keys of a plain object (:447-451). -/
def keptProps (ps : List (List Char × JsV)) : List (List Char × JsV) := ps.filter (keep ·.1)

/-- The recursive arms, each behind its depth check `b` (:321, :399). -/
def runShape (_H : Heap) (b : Bool) (conv : JsV → R RV) : Shape → R RV
  | .done r => some r
  | .list items => if b then R.err nestErr else R.bind (listLoop conv items) fun xs => R.ok (.list xs)
  | .vec items => if b then R.err nestErr else some (fromVec items)
  | .map ps => if b then R.err nestErr else
      R.bind (listLoop conv ((keptProps ps).map (·.2))) fun xs =>
        R.ok (.map (((keptProps ps).map (unesc ·.1)).zip xs))

/-- `js_to_value_level(val, depth)` with `b = depth >= MAX_UDF_VALUE_NESTING`. -/
def fromLevel (H : Heap) (b : Bool) (conv : JsV → R RV) (v : JsV) : R RV :=
  runShape H b conv (classify (deref H v))

/-- `js_to_value_at` at depth `d` with `f` frames left (:263-268). -/
def fromF (H : Heap) (cap : Option Nat) : Nat → Nat → JsV → R RV
  | 0, _, _ => none
  | f + 1, d, v => fromLevel H (chk cap d) (fromF H cap f (d + 1)) v

/-- `js_to_value` (:253): `js_to_value_at(val, 0)`; `MarshalCap.fromJs_frames` shows `MAX + 1`
    frames always suffice (for any heap, cyclic ones included), so the `none` arm is unreachable. -/
def fromJs (H : Heap) (v : JsV) : Except String RV := Rget (fromF H (some MAX) (MAX + 1) 0 v)

end AlgoUdf.Marshal
