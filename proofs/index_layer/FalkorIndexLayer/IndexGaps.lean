import FalkorIndexLayer.IndexerM
import FalkorIndexLayer.Iter
/-
# Gaps closed 2026-10-20: #3166 (stopwords) and #3382 (vector query before OPERATIONAL)

Both bugs were invisible to the model: `IndexerM` recorded the label's stopwords
but not the ones its RediSearch spec was created with, and the query entry points
were proven sound against the RS table without ever asking whether that table was
fully populated.

| Lean | Rust (fb3d516d0) |
| --- | --- |
| `Idx.rsStop` (`IndexM.lean`) | the stopword list `create_rs_index` hands to `RediSearch_IndexOptionsSetStopwords` (`mod.rs:1119-1133`); fixed for the spec's lifetime |
| `StopwordsAgree`, `CreateIndexKeepsStopwords` | the property: what `db.indexes()` / the RDB report (`Index::stopwords` `mod.rs:2189`) is what RediSearch applies |
| `scenario3166` | `create_index` `indexer.rs:300-313` (existing spec: `register_fields` only) and `:341-343` (`set_stopwords` on the metadata) |
| `incTicket` | `populate_index` taking its ticket: `try_increment_pending` (`mod.rs:2110`) on the published entry (shared `Arc` counters) |
| `vectorQuery` | `Indexer::vector_query` `indexer.rs:541-553` → `Index::vector_query` `mod.rs:1798-1853` (no `is_operational` `mod.rs:2084` check) |
| `VectorQueryRequiresOperational`, `VectorQueryComplete` | the property: a KNN query refuses until OPERATIONAL (C: "Invalid arguments for procedure") |
-/
namespace IndexLayer.Meta
namespace Ixr
open IndexLayer.Iter (RIter Key scoredIds KnnSpec)

/-! ## #3166: stopwords recorded but never applied -/

/-- **Property (#3166).** An index that has a live RediSearch spec reports the
stopword list that spec was created with: `Index::stopwords` (shown by
`db.indexes()` and written to the RDB) equals the list RediSearch applies. -/
def StopwordsAgree (x : Idx) : Prop := x.hasRsIndex = true → x.rsStop = x.stopwords

instance (x : Idx) : Decidable (StopwordsAgree x) := by unfold StopwordsAgree; infer_instance

/-- **Property (#3166), lifted to `create_index`.** A successful `create_index`
leaves the label's entry in agreement whenever it was in agreement before. -/
def CreateIndexKeepsStopwords (E : Env) : Prop :=
  ∀ (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat) (o : Option IdxOptions)
    (I' : Ixr) (x : Idx),
    (∀ y, I.map.get l = some y → StopwordsAgree y) →
    createIndex E I t l as tot o = .ok I' → I'.map.get l = some x → StopwordsAgree x

theorem addAttrs_stop (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) :
    ∀ (as : List String) (x : Idx) (nf : AMap (List Field)) (y : Idx) (nf' : AMap (List Field)),
      addAttrs t fo vo x nf as = .ok (y, nf') →
        y.stopwords = x.stopwords ∧ y.rsStop = x.rsStop ∧ y.spec = x.spec
  | [], x, nf, y, nf', h => by simp [addAttrs] at h; obtain ⟨rfl, -⟩ := h; exact ⟨rfl, rfl, rfl⟩
  | a :: as, x, nf, y, nf', h => by
    simp only [addAttrs] at h
    split at h; · cases h
    have ih := addAttrs_stop t fo vo as _ _ y nf' h
    refine ⟨ih.1.trans ?_, ih.2.1.trans ?_, ih.2.2.trans ?_⟩ <;>
      (split <;> simp [Idx.addFieldToExisting, Idx.insertField] <;> split <;> rfl)

theorem createRsIndex_stop (x : Idx) (label : String) (stop : Option (List String))
    (lang : Option String) (langOk : String → Bool) (fresh : Nat) (y : Idx)
    (h : x.createRsIndex label stop lang langOk fresh = .ok y) :
    y.rsStop = stop ∧ y.stopwords = x.stopwords := by
  unfold Idx.createRsIndex at h
  split at h; · cases h
  split at h
  · split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    cases h; exact ⟨rfl, rfl⟩
  · split at h; · cases h
    cases h; exact ⟨rfl, rfl⟩

theorem registerFields_stop (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tk : Bool)
    (y : Idx) (h : x.registerFields m fo tk = .ok y) :
    y.rsStop = x.rsStop ∧ y.stopwords = x.stopwords := by
  unfold Idx.registerFields at h; split at h <;> cases h; exact ⟨rfl, rfl⟩

theorem finish_stop (t : IType) (l : Option String) (s : Option (List String)) (n : Nat) (x : Idx) :
    (finish t l s n x).rsStop = x.rsStop ∧
    (finish t l s n x).stopwords = if s.isSome && x.stopwords.isNone then s else x.stopwords := by
  simp only [finish]; repeat' split
  all_goals simp_all [Idx.setLanguage, Idx.setStopwords, Idx.setProgress]

/-- What a fix can keep: on a label with no entry yet, `create_index` creates the
spec with the same stopwords it records, so the two agree. The bug below is the
other path: a spec that already exists. -/
theorem createIndex_fresh_stopwords_agree (E : Env) (I : Ixr) (t : IType) (l : String)
    (as : List String) (tot : Nat) (o : Option IdxOptions) (I' : Ixr) (x : Idx)
    (hnew : I.map.get l = none) (h : createIndex E I t l as tot o = .ok I')
    (hx : I'.map.get l = some x) : StopwordsAgree x := by
  intro _
  unfold createIndex at h
  simp only at h
  split at h; · cases h
  split at h; · cases h
  unfold build at h
  simp only [hnew] at h
  split at h; · cases h
  next x1 nf hr1 =>
  obtain ⟨st1, rs1, sp1⟩ := addAttrs_stop t _ _ as _ [] x1 nf hr1
  split at h; · cases h
  next x2 hx2 =>
  cases h
  simp only [AMap.get_upd, ite_true, Option.some.injEq] at hx
  subst hx
  have hd : (Idx.default E.freshId).stopwords = none ∧ (Idx.default E.freshId).spec = none := ⟨rfl, rfl⟩
  have hrs : x1.hasRsIndex = false := by simp [Idx.hasRsIndex, sp1, hd.2]
  unfold specStep at hx2
  simp only [hrs, Bool.false_eq_true, ite_false] at hx2
  split at hx2; · cases hx2
  next x' hx' =>
  obtain ⟨c1, c2⟩ := createRsIndex_stop _ _ _ _ _ _ _ hx'
  obtain ⟨r1, r2⟩ := registerFields_stop _ _ _ _ _ hx2
  obtain ⟨f1, f2⟩ := finish_stop t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2
  simp only [regen, Option.isSome_none, Bool.false_eq_true, ite_false, f1, f2, r1, r2, c1, c2, st1, hd.1]
  cases (optParts E.lower o).2.1 <;> simp

/-- The live repro: `CREATE INDEX FOR (n:L) ON (n.id)`, then
`CREATE FULLTEXT INDEX FOR (n:L) ON (n.name) OPTIONS {stopwords: ['router']}`.
Observed: (has spec, stopwords the spec applies, stopwords the label reports). -/
def obs3166 (r : Except String Ixr) : Option (Bool × Option (List String) × Option (List String)) :=
  match r with
  | .ok I => (I.map.get "L").map (fun y => (y.hasRsIndex, y.rsStop, y.stopwords))
  | .error _ => none

def opt3166 : Option IdxOptions := some (.text { stopwords := some ["router"] })

def step1_3166 : Except String Ixr := createIndex E0 I0 .range "L" ["id"] 0 none
def step2_3166 : Except String Ixr :=
  match step1_3166 with
  | .ok I1 => createIndex E0 I1 .fulltext "L" ["name"] 0 opt3166
  | .error e => .error e

/-- **BUG #3166.** Both `CREATE`s succeed; the label reports stopwords `['router']`
but its RediSearch spec was created (by the range index) with RediSearch's default
list, and the second call only registers the new field on it (`indexer.rs:313`).
Live (port 6640 Rust main 9d3869aa8 / 6641 C): `db.idx.fulltext.queryNodes('L', 'router')`
→ Rust 1 row, C 0; `db.indexes()` reports `[router]` on both. -/
theorem bug3166_stopwords_not_applied :
    obs3166 step1_3166 = some (true, none, none) ∧
    obs3166 step2_3166 = some (true, none, some ["router"]) := by decide

theorem bug3166_violates : ¬ CreateIndexKeepsStopwords E0 := by
  intro hP
  obtain ⟨e1, e2⟩ := bug3166_stopwords_not_applied
  unfold step2_3166 at e2
  cases h1 : step1_3166 with
  | error e => rw [h1] at e1; cases e1
  | ok I1 =>
    rw [h1] at e1 e2
    simp only at e2
    cases h2 : createIndex E0 I1 .fulltext "L" ["name"] 0 opt3166 with
    | error e => rw [h2] at e2; cases e2
    | ok I2 =>
      rw [h2] at e2
      simp only [obs3166, Option.map_eq_some_iff] at e1 e2
      obtain ⟨y, hy, hy'⟩ := e1
      obtain ⟨x, hx, hx'⟩ := e2
      simp only [Prod.mk.injEq] at hy' hx'
      have hpre : ∀ y', I1.map.get "L" = some y' → StopwordsAgree y' := by
        intro y' hy2; rw [hy] at hy2; cases hy2; intro _; rw [hy'.2.1, hy'.2.2]
      have := hP I1 .fulltext "L" ["name"] 0 opt3166 I2 x hpre h2 hx hx'.1
      rw [hx'.2.1, hx'.2.2] at this; cases this

/-! ## #3382: a vector query answers while the index is UNDER CONSTRUCTION -/

/-- `populate_index` taking its ticket for generation `g` on the published entry
(`try_increment_pending` `mod.rs:2110`; the counters are a shared `Arc`, so the
published index itself changes, as in `updateProgress`). -/
def incTicket (I : Ixr) (l : String) (g : Nat) : Ixr :=
  match I.map.get l with
  | some x => { I with map := I.map.upd l { x with slots := x.slots.inc g } }
  | none => I

/-- `Indexer::vector_query` (`indexer.rs:541-553`) → `Index::vector_query`
(`mod.rs:1798-1853`): route by label (no index ⇒ `Ok(empty)`), `CString::new` of
`vector:{field}` (NUL ⇒ `Err`), then `Iter.vectorQ` with the FFI outcomes
`cloneOk` (`try_clone_ref`), `nodeOk` (`CreateVecSimNode`), `iterOk`
(`GetResultsIterator`) and RediSearch's answer `res`. Nothing consults
`is_operational`. -/
def vectorQuery (I : Ixr) (l a : String) (cloneOk nodeOk iterOk : Bool) (res : List (Key × Nat)) :
    Except String RIter :=
  match I.route l with
  | none => .ok RIter.empty
  | some x =>
    if Idx.hasNul (Iter.vectorField a) then .error "nul byte"
    else .ok (Iter.vectorQ x.spec cloneOk nodeOk iterOk res)

/-- **Property (#3382), the gate.** A KNN query on a label whose index is not
OPERATIONAL (a population ticket of the current generation is pending) is refused,
as in C, instead of answered from the documents populated so far. `q I l a` is the
query entry point with every FFI outcome fixed. -/
def VectorQueryRequiresOperational (q : Ixr → String → String → Except String RIter) : Prop :=
  ∀ I l a x, I.map.get l = some x → operational x = false → ∃ e, q I l a = .error e

/-- **Property (#3382), the result.** An `Ok` answer with `k` at least the number
of labelled nodes carrying a vector is all of them (the true top-`k`). -/
def VectorQueryComplete (nodes : List Nat) (k : Nat) (r : Except String RIter) : Prop :=
  k ≥ nodes.length → ∀ it, r = .ok it → ∀ n ∈ nodes, n ∈ (scoredIds it).map (·.1)

/-- The shape a fix can prove: check `is_operational` before querying. -/
def vectorQueryGuarded (I : Ixr) (l a : String) (c n i : Bool) (res : List (Key × Nat)) :
    Except String RIter :=
  match I.route l with
  | some x => if operational x then vectorQuery I l a c n i res
              else .error "Invalid arguments for procedure"
  | none => vectorQuery I l a c n i res

theorem vectorQueryGuarded_requires_operational (c n i : Bool) (res : List (Key × Nat)) :
    VectorQueryRequiresOperational (fun I l a => vectorQueryGuarded I l a c n i res) := by
  intro I l a x hx hop
  refine ⟨"Invalid arguments for procedure", ?_⟩
  simp [vectorQueryGuarded, route, hx, hop]

/-- `CREATE VECTOR INDEX FOR (n:V) ON (n.e) OPTIONS {dimension:2, similarityFunction:'euclidean'}`,
then `populate_index` takes its ticket for the entry's generation. -/
def I3382 : Ixr :=
  match createIndex E0 I0 .vector "V" ["e"] 3 (some (.vector ⟨2, some "euclidean"⟩)) with
  | .ok I1 => incTicket I1 "V" ((I1.map.get "V").map (·.slots.gen) |>.getD 0)
  | .error _ => I0

/-- The state of the live repro: the index exists, has a spec, and is not
OPERATIONAL (one ticket pending; `db.indexes()` says UNDER CONSTRUCTION). -/
theorem i3382_under_construction :
    (I3382.map.get "V").map (fun x => (x.hasRsIndex, operational x)) = some (true, false) := by
  decide

/-- **BUG #3382 (gate).** On main the query answers `Ok` while the index is under
construction, whatever RediSearch returns. -/
theorem bug3382_vector_query_ignores_operational (res : List (Key × Nat)) :
    ¬ VectorQueryRequiresOperational (fun I l a => vectorQuery I l a true true true res) := by
  intro hP
  have e := i3382_under_construction
  simp only [Option.map_eq_some_iff, Prod.mk.injEq] at e
  obtain ⟨x, hx, -, hop⟩ := e
  obtain ⟨err, he⟩ := hP I3382 "V" "e" x hx hop
  simp [vectorQuery, route, hx, Iter.vectorField, Idx.hasNul] at he

/-- **BUG #3382 (result).** Three labelled nodes carry a vector and the populate job
has written none of them yet (the RS table is empty). The empty answer is exactly
what the RediSearch KNN contract allows for that table (`KnnSpec`), and main returns
it as `Ok` with 0 rows. Live (port 6640, 200k `:V` nodes): `queryNodes('V','e',3,…)`
right after `CREATE VECTOR INDEX` → count 0, then 3, while `db.indexes()` reports
`[Indexing] 0/200000: UNDER CONSTRUCTION`; C (6641) errors
`Invalid arguments for procedure 'db.idx.vector.queryNodes'` until OPERATIONAL. -/
theorem bug3382_partial_answer :
    KnnSpec (fun _ => none) (fun _ => True) 3 [] ∧
    ¬ VectorQueryComplete [0, 1, 2] 3 (vectorQuery I3382 "V" "e" true true true []) := by
  refine ⟨⟨by simp, by simp⟩, fun h => ?_⟩
  have e := i3382_under_construction
  simp only [Option.map_eq_some_iff, Prod.mk.injEq] at e
  obtain ⟨x, hx, -, -⟩ := e
  have hq : vectorQuery I3382 "V" "e" true true true [] = .ok (Iter.vectorQ x.spec true true true []) := by
    simp [vectorQuery, route, hx, Iter.vectorField, Idx.hasNul]
  have := h (by decide) _ hq 0 (by simp)
  revert this
  unfold Iter.vectorQ
  split <;> simp [scoredIds, RIter.empty, RIter.new, RIter.items, Iter.drainN]

end Ixr
end IndexLayer.Meta
