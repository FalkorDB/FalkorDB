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
| `step1_3166`, `step2_3166` | `create_index` (`indexer.rs:262`); fixed by #3680: a stopword list means a fresh spec built with it (`shareSpec`) |
| `I3382` | `create_index(.., owes_population = true)`: the owed ticket (`owe_population`, `mod.rs:1172`) |
| `vectorQuery` | `Indexer::vector_query` `indexer.rs:541-553` → `Index::vector_query` `mod.rs:1798-1853` (no `is_operational` `mod.rs:2084` check) |
| `VectorQueryRequiresOperational`, `VectorQueryComplete` | the property: a KNN query refuses until OPERATIONAL (C: "Invalid arguments for procedure") |
-/
namespace IndexLayer.Meta
namespace Ixr
open IndexLayer.Iter (RIter Key scoredIds KnnSpec)

/-! ## #3166: stopwords recorded but never applied (fixed by #3680) -/

/-- **Property (#3166).** An index that has a live RediSearch spec reports the
stopword list that spec was created with: `Index::stopwords` (shown by
`db.indexes()` and written to the RDB) equals the list RediSearch applies. -/
def StopwordsAgree (x : Idx) : Prop := x.hasRsIndex = true → x.rsStop = x.stopwords

instance (x : Idx) : Decidable (StopwordsAgree x) := by unfold StopwordsAgree; infer_instance

/-- **Property (#3166), lifted to `create_index`.** A successful `create_index`
leaves the label's entry in agreement whenever it was in agreement before. -/
def CreateIndexKeepsStopwords (E : Env) : Prop :=
  ∀ (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat) (o : Option IdxOptions)
    (owes : Bool) (I' : Ixr) (x : Idx),
    (∀ y, I.map.get l = some y → StopwordsAgree y) →
    createIndex E I t l as tot o owes = .ok I' → I'.map.get l = some x → StopwordsAgree x

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

theorem registerAllFields_stop (x : Idx) (tk : Bool) (y : Idx) (h : x.registerAllFields tk = .ok y) :
    y.rsStop = x.rsStop ∧ y.stopwords = x.stopwords := by
  unfold Idx.registerAllFields at h; split at h <;> cases h; exact ⟨rfl, rfl⟩

/-- #3680: an explicit stopword list is always recorded (the spec was just built with it). -/
theorem finish_stop (t : IType) (l : Option String) (s : Option (List String)) (n : Nat) (x : Idx) :
    (finish t l s n x).rsStop = x.rsStop ∧
    (finish t l s n x).stopwords = if s.isSome then s else x.stopwords := by
  simp only [finish]; repeat' split
  all_goals simp_all [Idx.setLanguage, Idx.setStopwords, Idx.setProgress]

theorem x0Of_stop (E : Env) (ex : Option Idx) (share : Bool) :
    (x0Of E ex share).stopwords = (ex.bind (·.stopwords)) ∧
    (share = true → ∀ y, ex = some y → (x0Of E ex share).rsStop = y.rsStop) := by
  cases ex with
  | none => simp [x0Of, Idx.default]
  | some y => cases share <;> simp [x0Of, Idx.forkForDdl, Idx.bumpId]

/-- **#3166 fixed by #3680.** `create_index` keeps the stopwords the label reports equal
to the ones its spec applies: a statement with a stopword list builds a fresh spec with
it (`shareSpec` is false), and a statement that shares the spec records no new list. -/
theorem createIndex_keeps_stopwords (E : Env) : CreateIndexKeepsStopwords E := by
  intro I t l as tot o owes I' x hpre h hx _
  unfold createIndex at h
  simp only at h
  split at h; · cases h
  split at h; · cases h
  unfold build at h
  simp only at h
  split at h; · cases h
  next x1 nf hr1 =>
  obtain ⟨st1, rs1, -⟩ := addAttrs_stop t _ _ as _ [] x1 nf hr1
  split at h; · cases h
  next x2 hx2 =>
  cases h
  simp only [AMap.get_upd, ite_true, Option.some.injEq] at hx
  subst hx
  obtain ⟨-, -, -, -, -, os, ors⟩ := owe_fields owes (finish t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2)
  obtain ⟨f1, f2⟩ := finish_stop t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2
  rw [os, ors, f1, f2]
  generalize hsh : shareSpec owes (I.map.get l) t (optParts E.lower o).1 (optParts E.lower o).2.1 = sh at hr1 hx2 st1 rs1
  unfold specStep at hx2
  cases sh with
  | true =>
    obtain ⟨-, -, -, hst, hr⟩ := shareSpec_rs _ _ _ _ _ hsh
    simp only [ite_true] at hx2; cases hx2
    cases hex : I.map.get l with
    | none => rw [hex] at hr; cases hr
    | some y =>
      have hy := hpre y hex
      rw [hex] at hr st1
      have hrs : y.hasRsIndex = true := by simpa [Idx.hasRsIndex] using hr
      obtain ⟨-, hr0⟩ := x0Of_stop E (some y) true
      have := hr0 rfl y rfl
      rw [hex] at rs1
      simp only [Idx.deferFields, hst, Option.isSome_none, Bool.false_eq_true, ite_false]
      rw [rs1, this, st1, (x0Of_stop E (some y) true).1]
      exact hy hrs
  | false =>
    simp only [Bool.false_eq_true, ite_false] at hx2
    split at hx2; · cases hx2
    next x' hx' =>
    obtain ⟨c1, c2⟩ := createRsIndex_stop _ _ _ _ _ _ _ hx'
    obtain ⟨r1, r2⟩ := registerAllFields_stop _ _ _ hx2
    rw [r1, r2, c1, c2]
    cases (optParts E.lower o).2.1 <;> simp

/-- On a label with no entry yet, `create_index` creates the spec with the same
stopwords it records, so the two agree. -/
theorem createIndex_fresh_stopwords_agree (E : Env) (I : Ixr) (t : IType) (l : String)
    (as : List String) (tot : Nat) (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (x : Idx)
    (hnew : I.map.get l = none) (h : createIndex E I t l as tot o owes = .ok I')
    (hx : I'.map.get l = some x) : StopwordsAgree x :=
  createIndex_keeps_stopwords E I t l as tot o owes I' x (fun y hy => by rw [hnew] at hy; cases hy) h hx

/-- The live repro: `CREATE INDEX FOR (n:L) ON (n.id)`, then
`CREATE FULLTEXT INDEX FOR (n:L) ON (n.name) OPTIONS {stopwords: ['router']}`.
Observed: (has spec, stopwords the spec applies, stopwords the label reports). -/
def obs3166 (r : Except String Ixr) : Option (Bool × Option (List String) × Option (List String)) :=
  match r with
  | .ok I => (I.map.get "L").map (fun y => (y.hasRsIndex, y.rsStop, y.stopwords))
  | .error _ => none

def opt3166 : Option IdxOptions := some (.text { stopwords := some ["router"] })

def step1_3166 : Except String Ixr := createIndex E0 I0 .range "L" ["id"] 0 none true
def step2_3166 : Except String Ixr :=
  match step1_3166 with
  | .ok I1 => createIndex E0 I1 .fulltext "L" ["name"] 0 opt3166 true
  | .error e => .error e

/-- **#3166 fixed (#3680).** Before #3680 the second call only registered the new field
on the range index's spec (created with RediSearch's default list) and the label
reported `['router']` while 'router' still matched (Rust 1 row, C 0). Now the second
call builds a fresh spec with `['router']`, and the label reports the same list. -/
theorem fix3166_stopwords_applied :
    obs3166 step1_3166 = some (true, none, none) ∧
    obs3166 step2_3166 = some (true, some ["router"], some ["router"]) := by decide

/-! ## #3382: a vector query answers while the index is UNDER CONSTRUCTION -/

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

/-- `Indexer::is_operational` (`indexer.rs:536-544`, PR #3581): no index on the label ⇒
`true` (`is_none_or`), else `Index::is_operational` (`mod.rs:2088`, `operational`). -/
def isOperational (I : Ixr) (l : String) : Bool :=
  match I.route l with
  | some x => operational x
  | none => true

theorem isOperational_none (I : Ixr) (l : String) (h : I.map.get l = none) :
    isOperational I l = true := by
  simp [isOperational, route, h]

theorem isOperational_some (I : Ixr) (l : String) (x : Idx) (h : I.map.get l = some x) :
    isOperational I l = operational x := by
  simp [isOperational, route, h]

/-- PR #3581's gate in `Graph::vector_query_nodes`/`_edges` (`graph.rs:4381,4447`):
`if !indexer.is_operational(label) { Err } else indexer.vector_query ..` is exactly the
fix shape `vectorQueryGuarded`, so it satisfies `VectorQueryRequiresOperational`. -/
theorem vectorQueryGuarded_eq_isOperational (I : Ixr) (l a : String) (c n i : Bool)
    (res : List (Key × Nat)) :
    vectorQueryGuarded I l a c n i res =
      (if isOperational I l then vectorQuery I l a c n i res
       else .error "Invalid arguments for procedure") := by
  unfold vectorQueryGuarded isOperational
  cases route I l <;> simp

/-- `CREATE VECTOR INDEX FOR (n:V) ON (n.e) OPTIONS {dimension:2, similarityFunction:'euclidean'}`:
since #3680 the statement itself leaves the entry owing its population (one ticket of its
generation, taken over by the job the publisher starts). -/
def I3382 : Ixr :=
  match createIndex E0 I0 .vector "V" ["e"] 3 (some (.vector ⟨2, some "euclidean"⟩)) true with
  | .ok I1 => I1
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
