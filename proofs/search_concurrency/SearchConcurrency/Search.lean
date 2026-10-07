/-
# Search layer: index options, document building, KNN arguments

Faithful models of the Rust functions that turn Cypher index DDL and entity
properties into RediSearch calls. RediSearch itself is the FFI boundary and is
**axiomatised as a hypothesis structure** (`RediSearch`), never as a Lean `axiom`.

| here | there |
| --- | --- |
| `IndexType`, `Field`            | `graph/src/index/mod.rs:96` `IndexType`, `:107` `Field`, `:137` `Field::new`, `:154` `new_with_vector_options` |
| `arrNames`                      | `mod.rs:121` `Field::make_arr_names` |
| `docSet`                        | `mod.rs:716-871` `Document::set` @ 8743953a8 (branch for branch; vector arm `:730-744`, #3087; temporal arm `:803`, #3076) |
| `docSetPre3076`                 | historical: the range arm before #3076 (`e20300436`) wrote a temporal as a number |
| `docSetPre3087`                 | historical: the vector arm at fe619ac5f (no dimension check) |
| `docSetPanics`                  | the `unreachable!()` arm, `mod.rs:864-868` |
| `buildDoc`                      | the per-attribute loop in `graph.rs:3664` `commit_index_kind` / `graph.rs:625` population `build_doc` |
| `RediSearch`                    | `RediSearch_IndexAddDocument(.., REDISEARCH_ADD_REPLACE)` (`mod.rs:1902` `Index::add_document`) |
| `fulltextUnknown`               | `graph/src/runtime/runtime.rs:1894-1903` fulltext unknown-key refusal (#3094) |
| `parseDimension` … `parseNatOpt`| `graph/src/runtime/runtime.rs:1972-2050` `map_to_index_options`, vector branch |
| `parsePhonetic`                 | `runtime.rs:1924` phonetic arm |
| `metricOf`                      | `mod.rs:1273-1287` similarity match in `Index::register_fields` |
| `evalK`                         | `graph/src/runtime/ops/node_by_vector_scan.rs:143` `eval_vector_args` (k arm) |
| `withCapacity`                  | Rust std `Vec::with_capacity`; historical use: `graph.rs:3883` / `:3938` `Vec::with_capacity(k)` @ 49f698d22 (removed by #3088; current model in `Knn`) |
-/

namespace SC.Search

inductive IndexType where
  | range | fulltext | vector
deriving DecidableEq, Repr

/-- Values reaching `Document::set`. Numeric payloads are abstract: they never
decide *which* RediSearch field is written, only its content. -/
inductive Val where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float
  | str (s : String)
  | temporal
  | vecf32 (dim : Nat)
  | point
  | list (xs : List Val)
  | map
  | node
  | rel
  | path
deriving Repr

/-- One `RediSearch_DocumentAddField*` call made by `Document::set`. -/
inductive RSAdd where
  | vector (field : String) (nbytes : Nat)
  | text (field : String)
  | num (field : String)
  | tag (field : String)
  | geo (field : String)
  | numArr (field : String) (n : Nat)
  | strArr (field : String) (n : Nat)
deriving Repr, DecidableEq

structure Field where
  name : String
  ty : IndexType
  /-- `field.vector_options.map(|o| o.dimension)`: `none` when the field has no
  vector options (`Field::new`, mod.rs:137), `some d` from `new_with_vector_options`
  (mod.rs:154). -/
  vdim : Option Nat := none
deriving Repr

/-- `Field::make_arr_names` (mod.rs:121): only range fields get array sub-fields. -/
def arrNames (f : Field) : Option (String × String) :=
  if f.ty = .range then some (f.name ++ ":numeric:arr", f.name ++ ":string:arr") else none

def isNumeric : Val → Bool
  | .bool _ | .int _ | .float => true
  | _ => false

def isStr : Val → Bool
  | .str _ => true
  | _ => false

/-- `Document::set` (mod.rs:716-871 @ 8743953a8), branch for branch. The vector arm
(mod.rs:730-744, #3087) adds the blob only when
`field.vector_options.as_ref().is_some_and(|o| o.dimension != 0 && o.dimension == vec.len())`,
mirrored literally by `Option.any`. -/
def docSet (f : Field) (v : Val) : List RSAdd :=
  match f.ty with
  | .vector => match v with
    | .vecf32 d => if f.vdim.any (fun k => k != 0 && k == d) then [.vector f.name (d * 4)] else []
    | _ => []
  | .fulltext => match v with
    | .str _ => [.text f.name]
    | _ => []
  | .range => match v with
    | .bool _ | .int _ | .float => [.num f.name]
    | .str _ => [.tag f.name]
    -- #3076 (`e20300436`): temporals are not indexed (mod.rs:796-803)
    | .temporal => []
    | .list xs =>
      let n := (xs.filter isNumeric).length
      let s := (xs.filter isStr).length
      (if n > 0 then [.numArr (f.name ++ ":numeric:arr") n] else []) ++
      (if s > 0 then [.strArr (f.name ++ ":string:arr") s] else [])
    | .vecf32 _ => []
    | .point => [.geo f.name]
    | .null | .map | .node | .rel | .path => []

/-- HISTORICAL (fe619ac5f, before #3087): the vector arm wrote `vec.len() * 4` bytes
with no dimension check (`if let Value::VecF32(vec) = value { AddFieldVector(..) }`);
every other arm is as in `docSet` (whose temporal arm is the post-#3076 one; the
#3087 theorems never involve temporals). Kept only to state what #3087 fixed. -/
def docSetPre3087 (f : Field) (v : Val) : List RSAdd :=
  match f.ty, v with
  | .vector, .vecf32 d => [.vector f.name (d * 4)]
  | _, _ => docSet f v

/-- HISTORICAL (before #3076, `e20300436`): the range arm wrote a temporal's raw
number into the numeric field (`RediSearch_DocumentAddFieldNumber(.., *ts as f64, ..)`),
so `n.v > 0` index scans matched dates. Every other arm as in `docSet`. -/
def docSetPre3076 (f : Field) (v : Val) : List RSAdd :=
  match f.ty, v with
  | .range, .temporal => [.num f.name]
  | _, _ => docSet f v

/-- **#3076: a temporal value is never written to any RediSearch field**, on any
index type, so no numeric index query can return it. -/
theorem docSet_temporal_not_indexed (f : Field) : docSet f .temporal = [] := by
  unfold docSet; cases f.ty <;> rfl

/-- What #3076 fixed: before it, a temporal on a range field landed in the numeric field. -/
theorem pre3076_temporal_in_numeric (f : Field) (h : f.ty = .range) :
    docSetPre3076 f .temporal = [.num f.name] := by
  unfold docSetPre3076; rw [h]

/-- #3076 changed only the temporal arm. -/
theorem docSet_eq_pre3076_off_temporal (f : Field) (v : Val) (hv : v ≠ .temporal) :
    docSet f v = docSetPre3076 f v := by
  unfold docSetPre3076
  cases hf : f.ty <;> cases v <;> simp_all

/-- The `unreachable!()` arm: `Null`, `Map`, `Node`, `Relationship`, `Path` on a range field. -/
def docSetPanics (f : Field) (v : Val) : Bool :=
  match f.ty, v with
  | .range, .null | .range, .map | .range, .node | .range, .rel | .range, .path => true
  | _, _ => false

/-- The document for one entity: every (field, value) pair of the label's index. -/
def buildDoc (fv : List (Field × Val)) : List RSAdd :=
  fv.flatMap fun p => docSet p.1 p.2

def buildDocPre3087 (fv : List (Field × Val)) : List RSAdd :=
  fv.flatMap fun p => docSetPre3087 p.1 p.2

/-! ## RediSearch boundary (AXIOMATISED, as a hypothesis) -/

/-- What we rely on from `RediSearch_IndexAddDocument(idx, doc, REDISEARCH_ADD_REPLACE, NULL)`.

RediSearch (`src/document.c` `AddDocumentCtx_Submit`, vector preprocess in
`src/vector_index.c`, "Could not add vector with blob size N (expected M)") fails the
**whole** document when a vector field's blob cannot be indexed: either its size is not
`dim * sizeof(float32)`, or the field was created without vector params (no
`dimOf`, the #3087 comment at mod.rs:722-729). Under `REPLACE` the previous document
under the same key is removed first. `dimOf` is the dimension
`RediSearch_VectorFieldSetParams` registered (`register_fields`, mod.rs:1270-1329,
only when `dimension > 0`). -/
structure RediSearch where
  dimOf : String → Option Nat
  accepts : List RSAdd → Bool
  accepts_iff : ∀ d, accepts d = true ↔
    ∀ fld nb, RSAdd.vector fld nb ∈ d → ∃ k, dimOf fld = some k ∧ nb = k * 4

/-- The index entry left under the entity's key after `add_document`: the document
if accepted, nothing otherwise (REPLACE already deleted the old one). -/
def RediSearch.stored (rs : RediSearch) (d : List RSAdd) : List RSAdd :=
  if rs.accepts d then d else []

/-- `register_fields` (mod.rs:1270-1272): a vector field gets params, of its own
dimension, exactly when `vector_options` is present with `dimension > 0`. Only the
"if" half is needed for the correctness theorems. -/
def Registered (rs : RediSearch) (fv : List (Field × Val)) : Prop :=
  ∀ p ∈ fv, p.1.ty = .vector → ∀ k, p.1.vdim = some k → k ≠ 0 → rs.dimOf p.1.name = some k

/-! ### Key property (#3087): `set` never adds an unindexable vector -/

/-- For **any** field options and **any** value, a vector add made by `Document::set`
names the field itself and carries exactly `k * 4` bytes for the field's own, non-zero
dimension `k`. A wrong-dimension vector, a dimension-0 field or a field without vector
options never yields a vector add. -/
theorem docSet_vector_sound {f : Field} {v : Val} {fld : String} {nb : Nat}
    (hm : RSAdd.vector fld nb ∈ docSet f v) :
    f.ty = .vector ∧ fld = f.name ∧ ∃ k, f.vdim = some k ∧ k ≠ 0 ∧ nb = k * 4 := by
  cases hty : f.ty <;> cases v <;> simp [docSet, hty] at hm
  case vector.vecf32 d =>
    obtain ⟨hany, rfl, rfl⟩ := hm
    cases hk : f.vdim with
    | none => simp [hk] at hany
    | some k =>
      simp [hk] at hany
      obtain ⟨hk0, rfl⟩ := hany
      exact ⟨rfl, rfl, k, rfl, hk0, rfl⟩
  all_goals (split at hm <;> split at hm <;> simp_all)

/-- A vector of the index's dimension is still indexed: the guard drops nothing good. -/
theorem docSet_good_vector {f : Field} {d : Nat} (hty : f.ty = .vector)
    (hv : f.vdim = some d) (hd : d ≠ 0) :
    docSet f (.vecf32 d) = [.vector f.name (d * 4)] := by
  simp [docSet, hty, hv, hd]

/-- Wrong dimension, dimension 0, or no vector options: nothing is added. -/
theorem docSet_bad_vector_skipped {f : Field} {d : Nat} (hty : f.ty = .vector)
    (hbad : ∀ k, f.vdim = some k → k = 0 ∨ k ≠ d) :
    docSet f (.vecf32 d) = [] := by
  cases hk : f.vdim with
  | none => simp [docSet, hty, hk]
  | some k =>
    rcases hbad k hk with h | h <;> simp [docSet, hty, hk, h] <;> omega

/-- Therefore every document is accepted, whatever the values: no property value can
make an entity disappear from its label's other indexes. -/
theorem doc_always_accepted (rs : RediSearch) (fv : List (Field × Val))
    (hreg : Registered rs fv) : rs.accepts (buildDoc fv) = true := by
  apply (rs.accepts_iff _).mpr
  intro fld nb hm
  simp only [buildDoc, List.mem_flatMap] at hm
  obtain ⟨p, hp, hm⟩ := hm
  obtain ⟨hty, rfl, k, hk, hk0, rfl⟩ := docSet_vector_sound hm
  exact ⟨k, hreg p hp hty k hk hk0, rfl⟩

/-- … and every add of every field — in particular each range / fulltext entry — is in
the stored entry. -/
theorem other_fields_never_dropped (rs : RediSearch) (fv : List (Field × Val))
    (hreg : Registered rs fv) :
    ∀ p ∈ fv, ∀ a ∈ docSet p.1 p.2, a ∈ rs.stored (buildDoc fv) := by
  intro p hp a ha
  simp only [RediSearch.stored, doc_always_accepted rs fv hreg, ↓reduceIte]
  exact List.mem_flatMap.mpr ⟨p, hp, ha⟩

/-- #3087 changed only the vector arm. -/
theorem docSet_nonvector_eq_pre (f : Field) (v : Val) (h : f.ty ≠ .vector) :
    docSet f v = docSetPre3087 f v := by
  cases hty : f.ty <;> cases v <;> simp_all [docSetPre3087]

/-! ### The former counterexamples, now correctness theorems

`CREATE INDEX FOR (n:L) ON (n.name)`, a vector index on `n.v`, then
`CREATE (:L {name:'a', v:vecf32([1,2,3])})` and `MATCH (n:L) WHERE n.name = 'a'`
(an index scan). Three index shapes: `{dimension:2}`, `{dimension:0}`, no options
(W5-idx-2's half-created index). -/

def nameF : Field := { name := "range:name", ty := .range }
def vecF : Field := { name := "vector:v", ty := .vector, vdim := some 2 }
def vecF0 : Field := { name := "vector:v", ty := .vector, vdim := some 0 }
def vecFNone : Field := { name := "vector:v", ty := .vector }

/-- Was `vector_dim_mismatch_drops_range_entry` (BUG 3 / W3-conc-3): fixed by #3087
(49f698d22). Holds for **every** RediSearch satisfying the contract. -/
theorem vector_dim_mismatch_keeps_range_entry (rs : RediSearch)
    (h : rs.dimOf "vector:v" = some 2) :
    RSAdd.tag "range:name" ∈ rs.stored (buildDoc [(nameF, .str "a"), (vecF, .vecf32 3)]) := by
  apply other_fields_never_dropped rs _ ?_ (nameF, .str "a") (by simp) _ (by simp [docSet, nameF])
  intro p hp hty k hk hk0
  simp at hp
  rcases hp with rfl | rfl
  · simp [nameF] at hty
  · simp [vecF] at hk; subst hk; exact h

/-- `{dimension:0}` (Rust still creates it, divergence (a)): the name entry survives,
for any RediSearch — it does not even need to know the field. -/
theorem dim0_keeps_range_entry (rs : RediSearch) :
    RSAdd.tag "range:name" ∈ rs.stored (buildDoc [(nameF, .str "a"), (vecF0, .vecf32 3)]) := by
  apply other_fields_never_dropped rs _ ?_ (nameF, .str "a") (by simp) _ (by simp [docSet, nameF])
  intro p hp hty k hk hk0
  simp at hp
  rcases hp with rfl | rfl
  · simp [nameF] at hty
  · simp [vecF0] at hk; omega

/-- No vector options (the W5-idx-2 shape): the name entry survives. -/
theorem noopts_keeps_range_entry (rs : RediSearch) :
    RSAdd.tag "range:name" ∈ rs.stored (buildDoc [(nameF, .str "a"), (vecFNone, .vecf32 3)]) := by
  apply other_fields_never_dropped rs _ ?_ (nameF, .str "a") (by simp) _ (by simp [docSet, nameF])
  intro p hp hty k hk hk0
  simp at hp
  rcases hp with rfl | rfl
  · simp [nameF] at hty
  · simp [vecFNone] at hk

/-- The good vector is kept and still indexed. -/
theorem good_vector_keeps_range_entry (rs : RediSearch)
    (h : rs.dimOf "vector:v" = some 2) :
    RSAdd.tag "range:name" ∈ rs.stored (buildDoc [(nameF, .str "a"), (vecF, .vecf32 2)]) ∧
    RSAdd.vector "vector:v" 8 ∈ rs.stored (buildDoc [(nameF, .str "a"), (vecF, .vecf32 2)]) := by
  have hreg : Registered rs [(nameF, .str "a"), (vecF, .vecf32 2)] := by
    intro p hp hty k hk hk0
    simp at hp
    rcases hp with rfl | rfl
    · simp [nameF] at hty
    · simp [vecF] at hk; subst hk; exact h
  exact ⟨other_fields_never_dropped rs _ hreg (nameF, .str "a") (by simp) _ (by simp [docSet, nameF]),
    other_fields_never_dropped rs _ hreg (vecF, .vecf32 2) (by simp) _ (by simp [docSet, vecF])⟩

/-! ### HISTORICAL (fe619ac5f): what the unguarded arm cost, fixed by #3087 (49f698d22) -/

theorem pre3087_dim_mismatch_dropped_range_entry (rs : RediSearch)
    (h : rs.dimOf "vector:v" = some 2) :
    RSAdd.tag "range:name" ∉
      rs.stored (buildDocPre3087 [(nameF, .str "a"), (vecF, .vecf32 3)]) := by
  have hna : rs.accepts (buildDocPre3087 [(nameF, .str "a"), (vecF, .vecf32 3)]) = false := by
    cases hc : rs.accepts (buildDocPre3087 [(nameF, .str "a"), (vecF, .vecf32 3)]) with
    | false => rfl
    | true =>
      obtain ⟨k, hk, hnb⟩ := (rs.accepts_iff _).mp hc "vector:v" 12
        (by simp [buildDocPre3087, docSetPre3087, docSet, nameF, vecF])
      rw [h] at hk; cases hk; omega
  simp [RediSearch.stored, hna]

/-- Dimension 0 / no options: the field has no vector params (`dimOf = none`), so the
old arm's vector add made RediSearch reject the whole document. -/
theorem pre3087_noparams_dropped_range_entry (rs : RediSearch) (f : Field)
    (hf : f.ty = .vector) (hn : rs.dimOf f.name = none) :
    RSAdd.tag "range:name" ∉
      rs.stored (buildDocPre3087 [(nameF, .str "a"), (f, .vecf32 3)]) := by
  have hna : rs.accepts (buildDocPre3087 [(nameF, .str "a"), (f, .vecf32 3)]) = false := by
    cases hc : rs.accepts (buildDocPre3087 [(nameF, .str "a"), (f, .vecf32 3)]) with
    | false => rfl
    | true =>
      obtain ⟨k, hk, _⟩ := (rs.accepts_iff _).mp hc f.name 12
        (by simp [buildDocPre3087, docSetPre3087, docSet, nameF, hf])
      rw [hn] at hk; cases hk
  simp [RediSearch.stored, hna]

/-- The fixed `docSet` on those same no-params shapes adds no vector at all. -/
theorem docSet_noparams_no_vector (f : Field) (hf : f.ty = .vector)
    (h : f.vdim = none ∨ f.vdim = some 0) (d : Nat) : docSet f (.vecf32 d) = [] := by
  apply docSet_bad_vector_skipped hf
  intro k hk
  rcases h with h | h <;> rw [h] at hk <;> cases hk; exact Or.inl rfl

/-- The fulltext arm indexes strings and nothing else (mod.rs:747-759). -/
theorem fulltext_only_strings (f : Field) (v : Val) (h : f.ty = .fulltext) :
    docSet f v = (if isStr v then [.text f.name] else []) := by
  unfold docSet; rw [h]; cases v <;> rfl

def RSAdd.isVecOrText : RSAdd → Bool
  | .vector .. | .text .. => true
  | _ => false

/-- A range field never produces a vector or text add. -/
theorem range_never_vector_or_text (f : Field) (v : Val) (h : f.ty = .range) :
    ∀ a ∈ docSet f v, a.isVecOrText = false := by
  cases v <;> simp [docSet, h, RSAdd.isVecOrText]
  all_goals (intro a ha; rcases ha with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> rfl)

/-! ## Index options (`map_to_index_options`, runtime.rs:1881-2075, main @ fe619ac5f)

#3094 (`fe619ac5f`'s parent series, fixes #3091) made the vector branch require
`dimension` and `similarityFunction` (lower-casing the latter) and the fulltext
branch refuse unknown keys — all three as C does. -/

/-- An option value as it arrives in the `OPTIONS {…}` map. -/
inductive OptVal where
  | int (i : Int)
  | str (s : String)
  | float
  | bool (b : Bool)
  | list
deriving Repr, DecidableEq

abbrev Opts := List (String × OptVal)

/-- `get` (runtime.rs:1885): first entry with that key. -/
def get (m : Opts) (k : String) : Option OptVal := (m.find? (·.1 == k)).map (·.2)

/-- `FULLTEXT_OPTIONS` (runtime.rs:1894-1895). -/
def fulltextOptions : List String := ["weight", "nostem", "phonetic", "language", "stopwords"]

/-- The fulltext unknown-key refusal (runtime.rs:1896-1903): the first key, in map
order, that is not one of `FULLTEXT_OPTIONS`. -/
def fulltextUnknown (m : Opts) : Option String :=
  (m.find? (fun kv => !fulltextOptions.contains kv.1)).map (·.1)

/-- `dimension` arm (runtime.rs:1972-1989): **required** since #3094 (it used to default to `0`). -/
def parseDimension (m : Opts) : Except String Nat :=
  match get m "dimension" with
  | some (.int n) => if n < 0 then .error "dimension must be a non-negative integer" else .ok n.toNat
  | none => .error "dimension is required"
  | some _ => .error "dimension must be an integer"

/-- `M` / `efConstruction` / `efRuntime` arms (runtime.rs:2007-2050). -/
def parseNatOpt (m : Opts) (k : String) : Except String (Option Nat) :=
  match get m k with
  | some (.int n) => if n < 0 then .error (k ++ " must be a non-negative integer") else .ok (some n.toNat)
  | none => .ok none
  | some _ => .error (k ++ " must be an integer")

/-- `similarityFunction` arm (runtime.rs:1990-2006): **required**, and stored as
`s.to_ascii_lowercase()` (Lean's `String.toLower` is ASCII-only, so it is that). -/
def parseSim (m : Opts) : Except String (Option String) :=
  match get m "similarityFunction" with
  | some (.str s) => .ok (some s.toLower)
  | none => .error "similarityFunction is required"
  | some _ => .error "similarityFunction must be a string"

inductive Metric where
  | l2 | ip | cosine
deriving DecidableEq, Repr

/-- `register_fields`' similarity match (mod.rs:1273-1287): default `"euclidean"`,
exact string compare — on the lower-cased name `parseSim` stored. -/
def metricOf (s : Option String) : Except String Metric :=
  match s.getD "euclidean" with
  | "euclidean" => .ok .l2
  | "ip" => .ok .ip
  | "cosine" => .ok .cosine
  | other => .error ("Unknown similarity function '" ++ other ++ "'")

structure VecOpts where
  dimension : Nat
  sim : Option String
  m : Option Nat
  efC : Option Nat
  efR : Option Nat
deriving Repr

def parseVector (m : Opts) : Except String VecOpts := do
  let d ← parseDimension m
  let s ← parseSim m
  let mm ← parseNatOpt m "M"
  let c ← parseNatOpt m "efConstruction"
  let r ← parseNatOpt m "efRuntime"
  return { dimension := d, sim := s, m := mm, efC := c, efR := r }

/-- Whether CREATE VECTOR INDEX succeeds end-to-end in Rust: parse, then the metric
check in `register_fields` (only reached when `dimension > 0`, mod.rs:1270-1272).
With no `OPTIONS` map at all, `create_index` now refuses (index_ddl.rs:61-65); that
is `parseVector []`, an error, here. -/
def rustAccepts (m : Opts) : Bool :=
  match parseVector m with
  | .error _ => false
  | .ok o => if o.dimension > 0 then (metricOf o.sim).toBool else true

/-- The C reference (observed on `falkordb.so`, `src/index/indexer.c` option
parsing): `dimension` and `similarityFunction` are both required, and the
similarity name is compared case-insensitively. -/
def cAccepts (m : Opts) : Bool :=
  match get m "dimension", get m "similarityFunction" with
  | some (.int n), some (.str s) =>
    decide (n ≥ 0) && ["euclidean", "cosine", "ip"].contains s.toLower
  | _, _ => false

/-- Parsed values are exactly the given integers: no truncation, no sign flip. -/
theorem parseDimension_int (m : Opts) (n : Int) (h : get m "dimension" = some (.int n)) (hn : 0 ≤ n) :
    parseDimension m = .ok n.toNat := by
  simp [parseDimension, h]; omega

theorem parseDimension_neg (m : Opts) (n : Int) (h : get m "dimension" = some (.int n)) (hn : n < 0) :
    ∃ e, parseDimension m = .error e := by
  simp [parseDimension, h, hn]

/-- **`dimension` is required** (#3094). -/
theorem parseDimension_missing (m : Opts) (h : get m "dimension" = none) :
    parseDimension m = .error "dimension is required" := by
  simp [parseDimension, h]

/-- **`similarityFunction` is required** (#3094), and stored lower-cased. -/
theorem parseSim_missing (m : Opts) (h : get m "similarityFunction" = none) :
    parseSim m = .error "similarityFunction is required" := by
  simp [parseSim, h]
theorem parseSim_lower (m : Opts) (s : String) (h : get m "similarityFunction" = some (.str s)) :
    parseSim m = .ok (some s.toLower) := by
  simp [parseSim, h]

/-- Without either required key no vector index is created (the old half-created
index of W5-idx-2 needed `dimension` absent ⇒ `0` ⇒ `register_fields` skipped). -/
theorem rust_refuses_missing (m : Opts)
    (h : get m "dimension" = none ∨ get m "similarityFunction" = none) : rustAccepts m = false := by
  unfold rustAccepts parseVector
  rcases h with h | h
  · simp [parseDimension_missing m h, bind, Except.bind]
  · cases hd : parseDimension m <;> simp [hd, parseSim_missing m h, bind, Except.bind]

/-- The metric match is total on the three documented names and rejects all else. -/
theorem metricOf_ok_iff (s : String) :
    (∃ mt, metricOf (some s) = .ok mt) ↔ s = "euclidean" ∨ s = "ip" ∨ s = "cosine" := by
  constructor
  · intro ⟨mt, h⟩
    simp only [metricOf, Option.getD_some] at h
    split at h <;> simp_all
  · rintro (rfl | rfl | rfl) <;> exact ⟨_, rfl⟩

theorem metricOf_toBool (s : String) :
    (metricOf (some s)).toBool = ["euclidean", "cosine", "ip"].contains s := by
  by_cases h : s = "euclidean" ∨ s = "ip" ∨ s = "cosine"
  · obtain ⟨mt, hm⟩ := (metricOf_ok_iff s).mpr h
    rw [hm]; rcases h with rfl | rfl | rfl <;> rfl
  · have hne : ∀ mt, metricOf (some s) ≠ .ok mt := fun mt hm => h ((metricOf_ok_iff s).mp ⟨mt, hm⟩)
    cases hm : metricOf (some s) with
    | ok mt => exact absurd hm (hne mt)
    | error e =>
      simp only [Except.toBool]
      simp only [not_or] at h
      simp [h.1, h.2.1, h.2.2]

/-- **Rust now agrees with C** (#3091 fixed by #3094) on every options map with a
positive `dimension` and no malformed HNSW key: same required keys, same
case-insensitive name. -/
theorem rust_eq_c (m : Opts) (hd : ∀ n, get m "dimension" = some (.int n) → n ≠ 0)
    (hM : ∀ v, get m "M" = some v → ∃ n, v = .int n ∧ 0 ≤ n)
    (hC : ∀ v, get m "efConstruction" = some v → ∃ n, v = .int n ∧ 0 ≤ n)
    (hR : ∀ v, get m "efRuntime" = some v → ∃ n, v = .int n ∧ 0 ≤ n) :
    rustAccepts m = cAccepts m := by
  have nat_ok : ∀ k, (∀ v, get m k = some v → ∃ n, v = .int n ∧ 0 ≤ n) →
      ∃ o, parseNatOpt m k = .ok o := by
    intro k hk
    cases hg : get m k with
    | none => exact ⟨none, by simp [parseNatOpt, hg]⟩
    | some v =>
      obtain ⟨n, rfl, hn⟩ := hk v hg
      exact ⟨some n.toNat, by simp [parseNatOpt, hg]; omega⟩
  obtain ⟨o1, h1⟩ := nat_ok "M" hM
  obtain ⟨o2, h2⟩ := nat_ok "efConstruction" hC
  obtain ⟨o3, h3⟩ := nat_ok "efRuntime" hR
  unfold rustAccepts cAccepts parseVector
  cases hg : get m "dimension" with
  | none => simp [parseDimension, hg, bind, Except.bind]
  | some v =>
    cases v with
    | int n =>
      have hn0 := hd n hg
      by_cases hneg : n < 0
      · have : ¬ (0 ≤ n) := by omega
        cases hs : get m "similarityFunction" with
        | none => simp [parseDimension, hg, hs, hneg, bind, Except.bind]
        | some w => cases w <;> simp [parseDimension, hg, hs, hneg, this, bind, Except.bind]
      · cases hs : get m "similarityFunction" with
        | none => simp [parseDimension, parseSim, hg, hs, hneg, bind, Except.bind]
        | some w =>
          cases w with
          | str s =>
            simp only [parseDimension, parseSim, hg, hs, hneg, h1, h2, h3, bind, Except.bind,
              pure, Except.pure, ite_false]
            have hpos : n.toNat > 0 := by omega
            simp only [hpos, ite_true, metricOf_toBool]
            simp; omega
          | _ => simp [parseDimension, parseSim, hg, hs, hneg, bind, Except.bind]
    | _ => simp [parseDimension, hg, bind, Except.bind]

-- Executable checks, re-run against fe619ac5f. Historical (2c874022a, #3091): missing `dimension`
-- or `similarityFunction` was accepted by Rust and `'Euclidean'` refused; C the reverse. Now both agree:
#guard rustAccepts [("similarityFunction", .str "euclidean")] = false
#guard cAccepts [("similarityFunction", .str "euclidean")] = false
#guard rustAccepts [("dimension", .int 2)] = false
#guard cAccepts [("dimension", .int 2)] = false
#guard rustAccepts [("dimension", .int 2), ("similarityFunction", .str "Euclidean")] = true
#guard cAccepts [("dimension", .int 2), ("similarityFunction", .str "Euclidean")] = true
#guard rustAccepts [] = false
-- Fulltext: a misspelt key is refused (runtime.rs:1896-1903), first unknown key in map order.
#guard fulltextUnknown [("weight", .float), ("nostemm", .bool true), ("lang", .str "x")] = some "nostemm"
#guard fulltextUnknown [("weight", .float), ("phonetic", .bool true)] = none

/-- C's fulltext key check as observed live (fe619ac5f vs `falkordb.so`, 2026-10-05):
`{}` accepted, `{foo:1}` refused, `{foo:1, weight:2.0}` accepted — a non-empty map
is refused only when it names *no* known key. -/
def cFulltextKeysOk (m : Opts) : Bool := m.isEmpty || m.any (fun kv => fulltextOptions.contains kv.1)

/-- Rust refuses an unknown key exactly when `fulltextUnknown` finds one. -/
theorem fulltextUnknown_none_iff (m : Opts) :
    fulltextUnknown m = none ↔ ∀ kv ∈ m, fulltextOptions.contains kv.1 = true := by
  unfold fulltextUnknown
  simp [List.find?_eq_none]

/-- Wherever Rust accepts the keys, C does too (Rust is the stricter of the two). -/
theorem rust_keys_ok_imp_c (m : Opts) (h : fulltextUnknown m = none) : cFulltextKeysOk m = true := by
  have hall := (fulltextUnknown_none_iff m).mp h
  unfold cFulltextKeysOk
  cases m with
  | nil => rfl
  | cons kv rest =>
    simp only [List.isEmpty_cons, Bool.false_or, List.any_eq_true]
    exact ⟨kv, List.mem_cons_self .., hall kv (List.mem_cons_self ..)⟩

-- Remaining differences, both reproduced live (Rust fe619ac5f vs C):
-- (a) `dimension: 0` skips the metric check in Rust (`register_fields` only runs it for `> 0`):
--     `OPTIONS {dimension:0, similarityFunction:'bogus'}` Rust creates the index, C refuses.
#guard rustAccepts [("dimension", .int 0), ("similarityFunction", .str "bogus")] = true
#guard cAccepts [("dimension", .int 0), ("similarityFunction", .str "bogus")] = false
-- (b) a known key beside an unknown one: `OPTIONS {weight:1.0, foo:true}` C creates, Rust refuses.
#guard fulltextUnknown [("weight", .float), ("foo", .bool true)] = some "foo"
#guard cFulltextKeysOk [("weight", .float), ("foo", .bool true)] = true
#guard cFulltextKeysOk [("foo", .int 1)] = false

/-- `phonetic` arm (runtime.rs:1924-1941). -/
def parsePhonetic : Option OptVal → Except String (Option String)
  | some (.bool b) => .ok (some (if b then "dm:en" else ""))
  | some (.str s) => if s.toLower = "dm:en" then .ok (some "dm:en") else .error "Unsupported phonetic algorithm"
  | none => .ok none
  | some _ => .error "Phonetic must be bool or string"

/-- `register_fields` sets `RSFLDOPT_TXTPHONETIC` iff the code is non-empty (mod.rs:1248). -/
def phoneticFlag (p : Option String) : Bool := p.any (· ≠ "")

/-- `phonetic:false` is stored as `""` and therefore does *not* set the flag. -/
theorem phonetic_false_no_flag : (parsePhonetic (some (.bool false))).map phoneticFlag = .ok false := rfl
theorem phonetic_true_flag : (parsePhonetic (some (.bool true))).map phoneticFlag = .ok true := rfl

/-! ## KNN `k` and the result buffer -/

/-- `eval_vector_args`, k arm (node_by_vector_scan.rs:143-147): any positive Int. -/
def evalK : OptVal → Except String Nat
  | .int n => if n > 0 then .ok n.toNat else .error "Invalid arguments"
  | _ => .error "Invalid arguments"

inductive Alloc where
  | ok (bytes : Nat)
  /-- `capacity overflow` panic → FalkorDB's panic hook → `process::exit(1)`. -/
  | panicOverflow
  /-- allocator failure → `handle_alloc_error` / Redis `zmalloc` OOM → abort. -/
  | abort
deriving DecidableEq, Repr

def isizeMax : Nat := 2 ^ 63 - 1

/-- `Vec::<T>::with_capacity(k)` with `size_of::<T>() = elem`, on an allocator
that can hand out at most `mem` bytes (Rust std `RawVec::try_allocate_in`). -/
def withCapacity (elem k mem : Nat) : Alloc :=
  if k * elem > isizeMax then .panicOverflow
  else if k * elem > mem then .abort
  else .ok (k * elem)

/-- HISTORICAL — W3-conc-4 / #3085, **fixed by #3088 (89d68334a)**. At 49f698d22
`vector_query_nodes`/`_edges` did `Vec::with_capacity(k)` (`graph.rs:3883` / `:3938`) and
`(NodeId, f64)` is 16 bytes, so the user's `k` reached `with_capacity` verbatim and any
`k` above `mem / 16` killed the server before RediSearch returned a single candidate.
`CALL db.idx.vector.queryNodes('L','v',1000000000000000,vecf32([1,2]))` aborted the live
server (SIGSEGV in the OOM path). Current code: `Knn.alloc_bounded`, `Knn.no_abort_any_k`. -/
theorem pre3088_huge_k_kills_server (k mem : Nat) (_hk : evalK (.int k) = .ok k) (hm : mem < k * 16) :
    withCapacity 16 k mem ≠ .ok (k * 16) := by
  unfold withCapacity
  split
  · simp
  · simp

theorem evalK_accepts_huge : evalK (.int 1000000000000000) = .ok 1000000000000000 := rfl

/-- The shape of the #3088 fix: reserve for at most the candidates RediSearch yielded. -/
theorem capped_capacity_ok (k cand mem : Nat) (hm : cand * 16 ≤ mem) (hi : cand * 16 ≤ isizeMax) :
    withCapacity 16 (min k cand) mem = .ok (min k cand * 16) := by
  unfold withCapacity
  have : min k cand * 16 ≤ cand * 16 := Nat.mul_le_mul_right _ (Nat.min_le_right _ _)
  simp only [show ¬ (min k cand * 16 > isizeMax) by omega, show ¬ (min k cand * 16 > mem) by omega, ite_false]

end SC.Search
