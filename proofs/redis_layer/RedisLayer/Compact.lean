/-!
# Compact result encoding vs the falkordb-py decoder

* `encPair` / `encV` — `reply_compact_value` (`src/reply.rs:134-344`): a value is emitted
  as its type tag followed by its payload; every caller wraps the two in a 2-array
  (`reply_result`, `reply.rs:619-666`; list elements `reply.rs:185`; map values
  `reply.rs:197`), and entity properties are 3-arrays `[attr_id, tag, payload]`.
* `dec` — `falkordb/query_result.py:parse_scalar` and its `__parse_*` helpers
  (falkordb-py 1.x, the version pinned in `tests/requirements.txt`), which look labels,
  relationship types and property names up by id in the client's schema cache.

RESP arrays whose length is sent postponed (`REDISMODULE_POSTPONED_LEN` then
`RM_ReplySetArrayLength`, `reply.rs:222-230`) are modelled as ordinary arrays: the
Redis module API guarantees the length set later is the one the client sees
(AXIOMATISED in the coverage table — it is a Redis guarantee, not code here).

Floats are abstract: `fmt` is `format_g(x, 15)` (`reply.rs:112`, libc `%.15g`) and
`parseF` is Python's `float()`. Vector elements are sent with `RedisModule_ReplyWithDouble`
(`reply.rs:330`, same as C), i.e. Redis's own double formatting `vfmt`, not `%.15g`. The round trip is exact *modulo* `parseF ∘ fmt`, which
is not the identity on `f64` (`0.1 + 0.2` → `"0.3"`); C emits the same `%.15g`
(`_ResultSet_ReplyWithRoundedDouble`), so this is shared behaviour, not a divergence.
-/

namespace RedisLayer.Compact

/-- A RESP reply tree. `bulk` carries a byte string, `dbl` a RESP double. -/
inductive R where
  | int (i : Int)
  | bulk (s : String)
  | nil
  | arr (xs : List R)
  deriving Repr, Inhabited

variable {F : Type}

/-- Runtime values that can reach a result set (`graph::runtime::value::Value`). -/
inductive Val (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (x : F)
  | str (s : String)
  | datetime (t : Int) | date (t : Int) | time (t : Int) | duration (t : Int)
  | list (xs : List (Val F))
  | map (kvs : List (String × Val F))
  | node (id : Nat) (labels : List Nat) (props : List (Nat × Val F))
  | edge (id : Nat) (ty : Nat) (src dst : Nat) (props : List (Nat × Val F))
  | path (nodes : List (Val F)) (edges : List (Val F))
  | vec (xs : List F)
  | point (lat lon : F)

/-- What the Python client hands the application. -/
inductive Py (F : Type) where
  | none
  | bool (b : Bool)
  | int (i : Int)
  | float (x : F)
  | str (s : String)
  | datetime (t : Int) | date (t : Int) | time (t : Int) | duration (t : Int)
  | list (xs : List (Py F))
  | dict (kvs : List (String × Py F))
  | node (id : Nat) (labels : Option (List String)) (props : List (String × Py F))
  | edge (id : Nat) (ty : String) (src dst : Nat) (props : List (String × Py F))
  | path (nodes edges : Py F)
  | vec (xs : List F)
  | point (lat lon : F)

structure Env (F : Type) where
  fmt : F → String                -- `format_g(x, 15)`
  vfmt : F → String               -- `RedisModule_ReplyWithDouble` (RESP2 bulk), vector elements
  parseF : String → F             -- Python `float(s)`
  label : Nat → String            -- `graph.schema.get_label`
  rel : Nat → String              -- `graph.schema.get_relation`
  prop : Nat → String             -- `graph.schema.get_property`

variable (E : Env F)

mutual
/-- `reply_compact_value`: the tag and the payload, before the caller's 2-array. -/
def encPair : Val F → List R
  | .null => [.int 1, .nil]
  | .bool x => [.int 4, .bulk (if x then "true" else "false")]
  | .int x => [.int 3, .int x]
  | .float x => [.int 5, .bulk (E.fmt x)]
  | .str s => [.int 2, .bulk s]
  | .datetime t => [.int 13, .int t]
  | .date t => [.int 14, .int t]
  | .time t => [.int 15, .int t]
  | .duration t => [.int 16, .int t]
  | .list xs => [.int 6, .arr (encList xs)]
  | .map kvs => [.int 10, .arr (encMap kvs)]
  | .node id ls ps => [.int 8, .arr [.int id, .arr (ls.map fun (l : Nat) => .int (l : Int)), .arr (encProps ps)]]
  | .edge id ty s d ps => [.int 7, .arr [.int id, .int ty, .int s, .int d, .arr (encProps ps)]]
  | .path ns es => [.int 9, .arr [.arr [.int 6, .arr (encList ns)], .arr [.int 6, .arr (encList es)]]]
  | .vec xs => [.int 12, .arr (xs.map fun x => .bulk (E.vfmt x))]
  | .point a o => [.int 11, .arr [.bulk (E.fmt a), .bulk (E.fmt o)]]
def encList : List (Val F) → List R
  | [] => []
  | v :: vs => .arr (encPair v) :: encList vs
def encMap : List (String × Val F) → List R
  | [] => []
  | (k, v) :: kvs => .bulk k :: .arr (encPair v) :: encMap kvs
def encProps : List (Nat × Val F) → List R
  | [] => []
  | (k, v) :: ps => .arr (.int k :: encPair v) :: encProps ps
end

def encV (v : Val F) : R := .arr (encPair E v)

def asNat : R → Option Nat
  | .int i => if 0 ≤ i then some i.toNat else none
  | _ => none

mutual
/-- `parse_scalar(value)`: `value[0]` is the type, `value[1]` the payload. -/
def dec : R → Option (Py F)
  | .arr [.int 1, _] => some .none
  | .arr [.int 2, .bulk s] => some (.str s)
  | .arr [.int 3, .int i] => some (.int i)
  | .arr [.int 4, .bulk s] => some (.bool (s == "true"))
  | .arr [.int 5, .bulk s] => some (.float (E.parseF s))
  | .arr [.int 6, .arr xs] => (decList xs).map .list
  | .arr [.int 7, .arr [.int id, .int ty, .int s, .int d, .arr ps]] =>
    match decProps ps with
    | some pp => some (.edge id.toNat (E.rel ty.toNat) s.toNat d.toNat pp)
    | none => none
  | .arr [.int 8, .arr [.int id, .arr ls, .arr ps]] =>
    match decLabels ls, decProps ps with
    | some l, some pp => some (.node id.toNat (if l = [] then none else some l) pp)
    | _, _ => none
  | .arr [.int 9, .arr [a, c]] =>
    match dec a, dec c with
    | some x, some y => some (.path x y)
    | _, _ => none
  | .arr [.int 10, .arr kvs] => (decMap kvs).map .dict
  | .arr [.int 11, .arr [.bulk a, .bulk o]] => some (.point (E.parseF a) (E.parseF o))
  | .arr [.int 12, .arr xs] => (decFloats xs).map .vec
  | .arr [.int 13, .int t] => some (.datetime t)
  | .arr [.int 14, .int t] => some (.date t)
  | .arr [.int 15, .int t] => some (.time t)
  | .arr [.int 16, .int t] => some (.duration t)
  | _ => none
def decList : List R → Option (List (Py F))
  | [] => some []
  | x :: xs =>
    match dec x, decList xs with
    | some a, some b => some (a :: b)
    | _, _ => none
def decMap : List R → Option (List (String × Py F))
  | [] => some []
  | .bulk k :: v :: rest =>
    match dec v, decMap rest with
    | some a, some b => some ((k, a) :: b)
    | _, _ => none
  | _ => none
def decProps : List R → Option (List (String × Py F))
  | [] => some []
  | .arr (.int k :: rest) :: ps =>
    match dec (.arr rest), decProps ps with
    | some a, some b => some ((E.prop k.toNat, a) :: b)
    | _, _ => none
  | _ => none
def decLabels : List R → Option (List String)
  | [] => some []
  | .int l :: ls => (decLabels ls).map (E.label l.toNat :: ·)
  | _ => none
def decFloats : List R → Option (List F)
  | [] => some []
  | .bulk s :: xs => (decFloats xs).map (E.parseF s :: ·)
  | _ => none
end

mutual
/-- The meaning the client is supposed to recover (the "decoder spec"). -/
def spec : Val F → Py F
  | .null => .none
  | .bool x => .bool x
  | .int x => .int x
  | .float x => .float (E.parseF (E.fmt x))
  | .str s => .str s
  | .datetime t => .datetime t
  | .date t => .date t
  | .time t => .time t
  | .duration t => .duration t
  | .list xs => .list (specList xs)
  | .map kvs => .dict (specMap kvs)
  | .node id ls ps => .node id (if ls = [] then none else some (ls.map E.label)) (specProps ps)
  | .edge id ty s d ps => .edge id (E.rel ty) s d (specProps ps)
  | .path ns es => .path (.list (specList ns)) (.list (specList es))
  | .vec xs => .vec (xs.map fun x => E.parseF (E.vfmt x))
  | .point a o => .point (E.parseF (E.fmt a)) (E.parseF (E.fmt o))
def specList : List (Val F) → List (Py F)
  | [] => []
  | v :: vs => spec v :: specList vs
def specMap : List (String × Val F) → List (String × Py F)
  | [] => []
  | (k, v) :: kvs => (k, spec v) :: specMap kvs
def specProps : List (Nat × Val F) → List (String × Py F)
  | [] => []
  | (k, v) :: ps => (E.prop k, spec v) :: specProps ps
end

theorem decLabels_enc (ls : List Nat) :
    decLabels E (ls.map fun (l : Nat) => R.int (l : Int)) = some (ls.map E.label) := by
  induction ls with
  | nil => simp [decLabels]
  | cons l ls ih => simp [decLabels, ih]

theorem decFloats_enc (xs : List F) :
    decFloats E (xs.map fun x => R.bulk (E.vfmt x)) = some (xs.map fun x => E.parseF (E.vfmt x)) := by
  induction xs with
  | nil => simp [decFloats]
  | cons x xs ih => simp [decFloats, ih]

theorem map_eq_nil_iff' {α β} (f : α → β) (l : List α) : (l.map f = []) ↔ l = [] := by
  cases l <;> simp

mutual
/-- **Round trip.** The client decodes every compact-encoded value to its spec. -/
theorem dec_enc : ∀ v : Val F, dec E (encV E v) = some (spec E v)
  | .null => by simp [encV, encPair, dec, spec]
  | .bool x => by cases x <;> simp [encV, encPair, dec, spec]
  | .int x => by simp [encV, encPair, dec, spec]
  | .float x => by simp [encV, encPair, dec, spec]
  | .str s => by simp [encV, encPair, dec, spec]
  | .datetime t => by simp [encV, encPair, dec, spec]
  | .date t => by simp [encV, encPair, dec, spec]
  | .time t => by simp [encV, encPair, dec, spec]
  | .duration t => by simp [encV, encPair, dec, spec]
  | .list xs => by simp [encV, encPair, dec, spec, decList_enc xs]
  | .map kvs => by simp [encV, encPair, dec, spec, decMap_enc kvs]
  | .node id ls ps => by
    simp only [encV, encPair, dec, decLabels_enc, decProps_enc ps, spec]
    simp [map_eq_nil_iff']
  | .edge id ty s d ps => by simp [encV, encPair, dec, spec, decProps_enc ps]
  | .path ns es => by
    have h1 := dec_enc_list ns
    have h2 := dec_enc_list es
    simp only [encV, encPair, dec, h1, h2, spec]
  | .vec xs => by simp [encV, encPair, dec, spec, decFloats_enc]
  | .point a o => by simp [encV, encPair, dec, spec]
theorem dec_enc_list (xs : List (Val F)) :
    dec E (.arr [.int 6, .arr (encList E xs)]) = some (.list (specList E xs)) := by
  simp [dec, decList_enc xs]
theorem decList_enc : ∀ xs : List (Val F), decList E (encList E xs) = some (specList E xs)
  | [] => by simp [encList, decList, specList]
  | v :: vs => by
    have h := dec_enc v
    simp only [encV] at h
    simp [encList, decList, specList, h, decList_enc vs]
theorem decMap_enc : ∀ kvs : List (String × Val F), decMap E (encMap E kvs) = some (specMap E kvs)
  | [] => by simp [encMap, decMap, specMap]
  | (k, v) :: kvs => by
    have h := dec_enc v
    simp only [encV] at h
    simp [encMap, decMap, specMap, h, decMap_enc kvs]
theorem decProps_enc : ∀ ps : List (Nat × Val F), decProps E (encProps E ps) = some (specProps E ps)
  | [] => by simp [encProps, decProps, specProps]
  | (k, v) :: ps => by
    have h := dec_enc v
    simp only [encV] at h
    simp [encProps, decProps, specProps, h, decProps_enc ps]
end

/-- The whole compact reply (`reply_result::<true>`, `reply.rs:619-666`): a header of
`[1, name]` pairs, rows of 2-arrays, statistics strings. -/
def encReply (names : List String) (rows : List (List (Val F))) (stats : List String) : R :=
  .arr [.arr (names.map fun n => .arr [.int 1, .bulk n]),
        .arr (rows.map fun row => .arr (row.map (encV E))),
        .arr (stats.map .bulk)]

/-- `QueryResult.__parse_records` / `__parse_header`. -/
def decReply : R → Option (List String × List (List (Py F)))
  | .arr [.arr hdr, .arr rows, _] => do
    let names ← hdr.mapM fun | .arr [.int 1, .bulk n] => some n | _ => none
    let rs ← rows.mapM fun | .arr cells => cells.mapM (dec E) | _ => none
    pure (names, rs)
  | _ => none

theorem decReply_enc (names : List String) (rows : List (List (Val F))) (stats : List String) :
    decReply E (encReply E names rows stats) = some (names, rows.map (·.map (spec E))) := by
  simp only [encReply, decReply]
  have h1 : (names.map fun n => R.arr [.int 1, .bulk n]).mapM
      (fun | .arr [.int 1, .bulk n] => some n | _ => none) = some names := by
    induction names with
    | nil => rfl
    | cons n ns ih => simp [List.mapM_cons, ih]
  have h2 : (rows.map fun row => R.arr (row.map (encV E))).mapM
      (fun | .arr cells => cells.mapM (dec E) | _ => none) = some (rows.map (·.map (spec E))) := by
    induction rows with
    | nil => rfl
    | cons r rs ih =>
      have hr : (r.map (encV E)).mapM (dec E) = some (r.map (spec E)) := by
        induction r with
        | nil => rfl
        | cons v vs ihv => simp [List.mapM_cons, dec_enc, ihv]
      simp [List.mapM_cons, hr, ih]
  simp [h1, h2]

/-! ## Where the round trip loses information on purpose -/

/-- A node whose span holds the same attribute id twice (possible through `GRAPH.BULK`
with a repeated property name, confirmed live — see `Bulk.lean`) is encoded with both
entries; Python's `properties[prop_name] = …` keeps only the last. -/
theorem duplicate_prop_ids_visible (E : Env F) (v w : Val F) :
    specProps E [(0, v), (0, w)] = [(E.prop 0, spec E v), (E.prop 0, spec E w)] := rfl

end RedisLayer.Compact
