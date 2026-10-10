import FalkorValueOrder.Glue
/-!
# JSON rendering of values (value.rs:49-55, :1374-1390, :1579-1753)

`DisplayJson::fmt_json` (the trait at value.rs:50 and the `Value` impl at :1582),
`Value::to_json_string` (:1374) and its `Display` wrapper `fmt` (:1384),
`write_json_string` (:1704) and `write_node_json` (:1716), modelled as functions that
return the `String` the `Formatter` receives (every `write!` is `++`; a `fmt::Error`
never arises when writing into a `String`).

Assumptions are a *structure* (`JsonEnv`), not axioms: the float classifiers and
`Display` of `f64`/`f32`, `{:.6}` of `f64::from(f32)`, `json_escape::escape_str`, the
temporal formatters, and the runtime accessors the impl calls
(`get_node_labels`, `get_node_attrs`, `get_relationship_attrs`, `get_relationship_type`,
`get_relationship_endpoints`).
-/

namespace ValueOrder.Json
open ValueOrder.Glue

/-- Everything `fmt_json` consults that is not its own logic. -/
structure JsonEnv (F F32 : Type) where
  isNaN : F → Bool
  isInf : F → Bool
  showF : F → String            -- `{fl}` (f64 Display)
  isNaN32 : F32 → Bool
  isInf32 : F32 → Bool
  showF32 : F32 → String        -- `{fl}` (f32 Display)
  fix6 : F32 → String           -- `{:.6}` of `f64::from(f32)`
  escape : String → String      -- concatenation of `escape_str(s)` chunks
  fmtDatetime : Int → String
  fmtDate : Int → String
  fmtTime : Int → String
  fmtDuration : Int → String
  nodeLabels : Nat → List String
  nodeAttrs : Nat → List (String × W F F32)
  relAttrs : Nat → List (String × W F F32)
  relType : Nat → Option String
  relEnds : Nat → Nat × Nat

variable {F F32 : Type} (E : JsonEnv F F32)

/-- `write_json_string` (value.rs:1704): `"` ++ escaped chunks ++ `"`. -/
def writeJsonString (s : String) : String := "\"" ++ E.escape s ++ "\""

/-- The `for (i, x) in xs.enumerate() { if i > 0 { "," } x }` loop used five times. -/
def sepJoin : List String → String
  | [] => ""
  | [x] => x
  | x :: y :: ys => x ++ "," ++ sepJoin (y :: ys)

theorem sepJoin_eq_intercalate (xs : List String) : sepJoin xs = ",".intercalate xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    cases xs with
    | nil => simp [sepJoin]
    | cons y ys =>
      simp only [sepJoin, ih, String.intercalate_cons_cons]

/-- Renderers for the two entity arms, supplied from outside so that the value recursion
below is structural (an entity's attributes come from the runtime, not from the value). -/
structure EntityR where
  node : Nat → Bool → String    -- `write_node_json(f, runtime, id, include_type)`
  rel : Nat → String            -- the `Relationship` arm

mutual
/-- `impl DisplayJson for Value::fmt_json` (value.rs:1582), arm by arm; the entity arms
call `R` (instantiated below with the real `write_node_json` / relationship arm). -/
def fmtS (R : EntityR) : W F F32 → String
  | .null => "null"
  | .bool b => if b then "true" else "false"
  | .int i => toString i
  | .float fl => if E.isNaN fl || E.isInf fl then "null" else E.showF fl
  | .str s => writeJsonString E s
  | .list l => "[" ++ sepJoin (fmtL R l) ++ "]"
  | .map m => "{" ++ sepJoin (fmtM R m) ++ "}"
  | .node id => R.node id true
  | .rel r => R.rel r
  | .path p => "[" ++ sepJoin (fmtL R p) ++ "]"
  | .vecf32 v => "[" ++ sepJoin (v.map fun fl =>
      if E.isNaN32 fl || E.isInf32 fl then "null" else E.showF32 fl) ++ "]"
  | .point lat lon => "{\"crs\":\"wgs-84\",\"latitude\":" ++ E.fix6 lat ++
      ",\"longitude\":" ++ E.fix6 lon ++ ",\"height\": null}"
  | .datetime t => writeJsonString E (E.fmtDatetime t)
  | .date t => writeJsonString E (E.fmtDate t)
  | .time t => writeJsonString E (E.fmtTime t)
  | .duration t => writeJsonString E (E.fmtDuration t)
def fmtL (R : EntityR) : List (W F F32) → List String
  | [] => []
  | v :: vs => fmtS R v :: fmtL R vs
/-- `write_json_string(k)?; ":"; v.fmt_json(..)` -/
def fmtM (R : EntityR) : List (String × W F F32) → List String
  | [] => []
  | (k, v) :: kvs => (writeJsonString E k ++ ":" ++ fmtS R v) :: fmtM R kvs
end

/-- `write_node_json` (value.rs:1716) given the renderer for attribute values. -/
def writeNodeJson (R : EntityR) (id : Nat) (includeType : Bool) : String :=
  "{" ++ (if includeType then "\"type\":\"node\"," else "") ++
  "\"id\":" ++ toString id ++ ",\"labels\":[" ++
  sepJoin ((E.nodeLabels id).map (writeJsonString E)) ++ "],\"properties\":{" ++
  sepJoin (fmtM E R (E.nodeAttrs id)) ++ "}}"

/-- The `Relationship` arm of `fmt_json` (value.rs:1622-1649). -/
def relJson (R : EntityR) (r : Nat) : String :=
  let tn := (E.relType r).getD ""
  "{\"type\":\"relationship\",\"id\":" ++ toString r ++ ",\"relationship\":" ++
    writeJsonString E tn ++ ",\"properties\":{" ++ sepJoin (fmtM E R (E.relAttrs r)) ++
    "},\"start\":" ++ writeNodeJson E R (E.relEnds r).1 false ++
    ",\"end\":" ++ writeNodeJson E R (E.relEnds r).2 false ++ "}"

/-- Entity renderers at nesting depth `n`. Entity attributes are stored values, which never
contain entities (`entityFree`), so depth 1 already renders every real graph exactly
(`fmtS_entityFree`); depth 0 is the (unreachable) bottom. -/
def renderers : Nat → EntityR
  | 0 => ⟨fun _ _ => "", fun _ => ""⟩
  | n + 1 => ⟨writeNodeJson E (renderers n), relJson E (renderers n)⟩

/-- `fmt_json` itself (value.rs:1582) at depth 2 (value → entity → attribute values). -/
def fmtJson (v : W F F32) : String := fmtS E (renderers E 2) v

/-- `DisplayJson` (trait value.rs:50): the single impl is `fmtJson`. -/
class DisplayJson (α : Type) where
  fmtJson : α → String


/-- `Value::to_json_string` (value.rs:1374) wraps the value in `JsonWrapper` whose
`Display::fmt` (:1384) calls `fmt_json`; `to_string()` collects that output. -/
def jsonWrapperFmt (v : W F F32) : String := DisplayJson.fmtJson (self := ⟨fmtJson E⟩) v
def toJsonString (v : W F F32) : String := jsonWrapperFmt E v

/-! ## Theorems -/

theorem displayJson_eq (v : W F F32) :
    (DisplayJson.fmtJson (self := ⟨fmtJson E⟩) v) = fmtJson E v := rfl

theorem toJsonString_eq (v : W F F32) : toJsonString E v = fmtJson E v := rfl

theorem jsonWrapperFmt_eq (v : W F F32) : jsonWrapperFmt E v = fmtJson E v := rfl

theorem writeJsonString_plain (s : String) (h : E.escape s = s) :
    writeJsonString E s = "\"" ++ s ++ "\"" := by simp [writeJsonString, h]

/-- NaN and ±∞ are rendered as JSON `null` (not as `NaN`/`inf`, which are not JSON). -/
theorem fmt_float_nonfinite (fl : F) (h : E.isNaN fl = true ∨ E.isInf fl = true) :
    fmtJson E (.float fl) = "null" := by
  rcases h with h | h <;> simp [fmtJson, fmtS, h]

theorem fmt_float_finite (fl : F) (h1 : E.isNaN fl = false) (h2 : E.isInf fl = false) :
    fmtJson E (.float fl) = E.showF fl := by simp [fmtJson, fmtS, h1, h2]

theorem fmtL_eq_map (R : EntityR) (l : List (W F F32)) : fmtL E R l = l.map (fmtS E R) := by
  induction l with
  | nil => simp [fmtL]
  | cons x xs ih => simp [fmtL, ih]

/-- A list renders as `[` + its elements' JSON joined by `,` + `]`; a path likewise. -/
theorem fmt_list (l : List (W F F32)) :
    fmtJson E (.list l) = "[" ++ ",".intercalate (l.map (fmtJson E)) ++ "]" ∧
    fmtJson E (.path l) = "[" ++ ",".intercalate (l.map (fmtJson E)) ++ "]" := by
  simp [fmtJson, fmtS, fmtL_eq_map, sepJoin_eq_intercalate]; rfl

/-- `include_type` only inserts `"type":"node",` right after the opening brace. -/
theorem writeNodeJson_includeType (R : EntityR) (id : Nat) :
    ∃ rest, writeNodeJson E R id true = "{" ++ ("\"type\":\"node\"," ++ rest) ∧
      writeNodeJson E R id false = "{" ++ rest :=
  ⟨_, by simp only [writeNodeJson, ite_true, String.append_assoc]; exact ⟨rfl, rfl⟩⟩

/-- A node renders with `"type":"node"`; a relationship's endpoints render without it. -/
theorem fmt_node (id : Nat) : fmtJson E (.node id) = writeNodeJson E (renderers E 1) id true := rfl

theorem fmt_rel (r : Nat) : fmtJson E (.rel r) = relJson E (renderers E 1) r := rfl

-- A value with no Node / Relationship inside it.
mutual
def entityFree : W F F32 → Bool
  | .node _ | .rel _ => false
  | .list l | .path l => entityFreeL l
  | .map m => entityFreeM m
  | _ => true
def entityFreeL : List (W F F32) → Bool
  | [] => true
  | v :: vs => entityFree v && entityFreeL vs
def entityFreeM : List (String × W F F32) → Bool
  | [] => true
  | (_, v) :: kvs => entityFree v && entityFreeM kvs
end

mutual
/-- Entity-free values render independently of the entity renderers, so the depth cut-off
in `renderers` is invisible on everything the graph can store. -/
theorem fmtS_entityFree (R R' : EntityR) :
    (v : W F F32) → entityFree v = true → fmtS E R v = fmtS E R' v
  | .null, _ | .bool _, _ | .int _, _ | .float _, _ | .str _, _ | .vecf32 _, _
  | .point _ _, _ | .datetime _, _ | .date _, _ | .time _, _ | .duration _, _ => rfl
  | .node _, h | .rel _, h => by simp [entityFree] at h
  | .list l, h => by
    simp only [entityFree] at h; simp only [fmtS, fmtL_entityFree R R' l h]
  | .path l, h => by
    simp only [entityFree] at h; simp only [fmtS, fmtL_entityFree R R' l h]
  | .map m, h => by
    simp only [entityFree] at h; simp only [fmtS, fmtM_entityFree R R' m h]
theorem fmtL_entityFree (R R' : EntityR) :
    (l : List (W F F32)) → entityFreeL l = true → fmtL E R l = fmtL E R' l
  | [], _ => rfl
  | v :: vs, h => by
    simp only [entityFreeL, Bool.and_eq_true] at h
    simp only [fmtL, fmtS_entityFree R R' v h.1, fmtL_entityFree R R' vs h.2]
theorem fmtM_entityFree (R R' : EntityR) :
    (m : List (String × W F F32)) → entityFreeM m = true → fmtM E R m = fmtM E R' m
  | [], _ => rfl
  | (k, v) :: kvs, h => by
    simp only [entityFreeM, Bool.and_eq_true] at h
    simp only [fmtM, fmtS_entityFree R R' v h.1, fmtM_entityFree R R' kvs h.2]
end

/-- Hence: if every stored attribute is entity-free (always true for graph properties),
rendering at any depth ≥ 1 for the entity arms gives the same text — the model's depth-2
`fmtJson` is exactly the unbounded recursion of the Rust impl. -/
theorem writeNodeJson_depth (R R' : EntityR) (id : Nat) (b : Bool)
    (h : ∀ kv ∈ E.nodeAttrs id, entityFree kv.2 = true) :
    writeNodeJson E R id b = writeNodeJson E R' id b := by
  have key : ∀ l : List (String × W F F32), (∀ kv ∈ l, entityFree kv.2 = true) →
      entityFreeM l = true := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons kv kvs ih =>
      intro h
      simp only [entityFreeM, Bool.and_eq_true]
      exact ⟨h kv (by simp), ih (fun x hx => h x (by simp [hx]))⟩
  have : fmtM E R (E.nodeAttrs id) = fmtM E R' (E.nodeAttrs id) :=
    fmtM_entityFree E R R' _ (key _ h)
  simp [writeNodeJson, this]

end ValueOrder.Json
