import FalkorIndexLayer.Proofs
import FalkorIndexLayer.Maintenance
/-
# Result iterators, RS handle refcounts, and the query entry points
(`graph/src/index/mod.rs`, origin/main 3fec7d7c9)

| Lean | Rust |
| --- | --- |
| `nextN`, `drainN` | `IndexResultsIter::next` `mod.rs:311`, `EdgeTripleIter::next` 381, `ScoredEdgeTripleIter::next` 445 (skip keys of the wrong length, decode) |
| `RIter`, `RIter.empty`, `RIter.next` | `IndexResultsIter::{new,empty,empty_scored}` 277/288/299, `EdgeTripleIter::{new,empty}` 362/370, `ScoredEdgeTripleIter::{new,empty}` 426/434, vector wrappers 498-537 |
| `decTriple` | `decode_triple` 612 |
| `RC`, `RC.clone/release/dropSpec` | `OwnedIndex::{from_owned,as_ptr,try_clone,into_raw,drop}` 926-964, `SpecHandle::{new,as_ptr,try_clone_ref,drop}` 980-1010 |
| `DocLife` | `Document::drop` 641, `add_document` 1893 (`consumed`) |
| `queryN`, `queryE` | `Index::query` 1677, `Index::query_edges` 1700 |
| `fulltextQ`, `vectorQ` | `Index::fulltext_query(_edges)` 1718/1747, `vector_query(_edges)` 1784/1846 |
| `delKey`, `delEdgeKey`, `commitEdge` | `delete_document` 1915, `delete_edge_document` 1932, `Indexer::commit_edge` `indexer.rs:744` |

RediSearch is a *hypothesis*: `run n` is the list of `(key, score)` pairs the
results iterator returns for query node `n`, and `RunSpec` states the documented
LLAPI contract (it returns the keys of exactly the stored documents that match).
-/
namespace IndexLayer.Iter
open IndexLayer

abbrev Key := List Nat

/-- One `next()` step: skip results whose key length is not `len`
(`debug_assert!` + `continue`), decode the first good one. -/
def nextN {β : Type} (len : Nat) (dec : Key → β) : List (Key × Nat) → Option ((β × Nat) × List (Key × Nat))
  | [] => none
  | (k, s) :: r => if k.length = len then some ((dec k, s), r) else nextN len dec r

def drainN {β : Type} (len : Nat) (dec : Key → β) (l : List (Key × Nat)) : List (β × Nat) :=
  l.filterMap (fun p => if p.1.length = len then some (dec p.1, p.2) else none)

/-- **Iterator law**: draining with `next()` yields exactly the decoded results
whose key has the right length, in RediSearch order. -/
theorem drain_next {β : Type} (len : Nat) (dec : Key → β) (l : List (Key × Nat)) :
    drainN len dec l = match nextN len dec l with
      | none => []
      | some (x, r) => x :: drainN len dec r := by
  induction l with
  | nil => rfl
  | cons p r ih =>
    obtain ⟨k, s⟩ := p
    by_cases h : k.length = len
    · simp [drainN, nextN, h]
    · simp only [nextN, h, ite_false]; rw [← ih]; simp [drainN, h]

/-- An `IndexResultsIter` / `EdgeTripleIter`: `none` is the null C iterator. -/
structure RIter where
  res : Option (List (Key × Nat))

def RIter.empty : RIter := ⟨none⟩
def RIter.new (l : List (Key × Nat)) : RIter := ⟨some l⟩
def RIter.next {β : Type} (len : Nat) (dec : Key → β) (it : RIter) : Option ((β × Nat) × RIter) :=
  match it.res with
  | none => none
  | some l => (nextN len dec l).map (fun p => (p.1, ⟨some p.2⟩))
def RIter.items {β : Type} (len : Nat) (dec : Key → β) (it : RIter) : List (β × Nat) :=
  match it.res with | none => [] | some l => drainN len dec l
/-- `Drop`: `RediSearch_ResultsIteratorFree` runs iff the iterator is non-null. -/
def RIter.frees (it : RIter) : Bool := it.res.isSome

theorem RIter.empty_spec {β : Type} (len : Nat) (dec : Key → β) :
    RIter.empty.next len dec = none ∧ RIter.empty.items len dec = [] ∧ RIter.empty.frees = false :=
  ⟨rfl, rfl, rfl⟩

theorem RIter.items_next {β : Type} (len : Nat) (dec : Key → β) (it : RIter) :
    it.items len dec = match it.next len dec with
      | none => []
      | some (x, it') => x :: it'.items len dec := by
  cases it with
  | mk res =>
    cases res with
    | none => rfl
    | some l =>
      simp only [RIter.items, RIter.next]
      rw [drain_next]
      cases nextN len dec l <;> rfl

theorem RIter.new_frees (l : List (Key × Nat)) : (RIter.new l).frees = true := rfl

def NODE_LEN : Nat := 16
def EDGE_LEN : Nat := 48
def decTriple (k : Key) : Nat × Nat × Nat :=
  (decodeId (k.take 16), decodeId ((k.drop 16).take 16), decodeId (k.drop 32))

/-- The id iterators (`IdIter`: map `|_, id| id`; `ScoredIdIter` and
`VectorScoredIdIter`: `(id, score)`; the vector wrappers just delegate). -/
def ids (it : RIter) : List Nat := (it.items NODE_LEN decodeId).map (·.1)
def scoredIds (it : RIter) : List (Nat × Nat) := it.items NODE_LEN decodeId
def triples (it : RIter) : List (Nat × Nat × Nat) := (it.items EDGE_LEN decTriple).map (·.1)
def scoredTriples (it : RIter) : List ((Nat × Nat × Nat) × Nat) := it.items EDGE_LEN decTriple

theorem mem_drainN {β : Type} (len : Nat) (dec : Key → β) (l : List (Key × Nat)) (x : β × Nat) :
    x ∈ drainN len dec l ↔ ∃ k, (k, x.2) ∈ l ∧ k.length = len ∧ dec k = x.1 := by
  obtain ⟨b, s⟩ := x
  simp only [drainN, List.mem_filterMap]
  constructor
  · rintro ⟨⟨k, s'⟩, h1, h2⟩
    split at h2
    · next hl => cases h2; exact ⟨k, h1, hl, rfl⟩
    · cases h2
  · rintro ⟨k, h1, h2, h3⟩
    exact ⟨(k, s), h1, by simp [h2, h3]⟩

/-! ## RS spec handle reference counting -/

/-- `rc`: strong refs on the RS spec; `owned`: live `OwnedIndex` clones (held by
iterators); `specAlive`: the shared `SpecHandle` (creation ref); `valid`: not yet
`DropIndex`ed; `freed`: `IndexSpec_Free` ran. -/
structure RC where
  rc : Nat
  owned : Nat
  specAlive : Bool
  valid : Bool
  freed : Bool
  deriving DecidableEq, Repr

/-- `RediSearch_CreateIndex` + `OwnedIndex::from_owned` + `SpecHandle::new`. -/
def RC.create : RC := ⟨1, 0, true, true, false⟩
/-- `SpecHandle::try_clone_ref` → `OwnedIndex::try_clone` → `RediSearch_IndexClone`
(NULL once invalidated ⇒ `from_owned` gives `None`). -/
def RC.clone (r : RC) : Option RC :=
  if r.valid ∧ ¬ r.freed then some { r with rc := r.rc + 1, owned := r.owned + 1 } else none
/-- `OwnedIndex::drop` → `RediSearch_IndexRelease`. -/
def RC.release (r : RC) : RC :=
  { r with rc := r.rc - 1, owned := r.owned - 1, freed := r.rc - 1 == 0 }
/-- `SpecHandle::drop`: `ManuallyDrop::take` + `into_raw` (no release by
`OwnedIndex`) + `RediSearch_DropIndex` (invalidate and release the creation ref). -/
def RC.dropSpec (r : RC) : RC :=
  { r with specAlive := false, valid := false, rc := r.rc - 1, freed := r.rc - 1 == 0 }
/-- What a buggy `SpecHandle::drop` without `into_raw` would do: the inner
`OwnedIndex` also releases. -/
def RC.dropSpecNoIntoRaw (r : RC) : RC := (r.dropSpec).release

def RC.Inv (r : RC) : Prop :=
  r.rc = r.owned + (if r.specAlive then 1 else 0) ∧ (r.freed = true ↔ r.rc = 0) ∧
  (r.specAlive = true → r.valid = true)

theorem RC.create_inv : RC.create.Inv := by simp [RC.create, RC.Inv]

theorem RC.clone_inv (r r' : RC) (h : r.Inv) (hc : r.clone = some r') : r'.Inv := by
  unfold RC.clone at hc; split at hc
  · next hv =>
    cases hc; obtain ⟨h1, h2, h3⟩ := h
    refine ⟨by cases hsa : r.specAlive <;> simp [hsa] at h1 ⊢ <;> omega, ?_, h3⟩
    simp [hv.2]
  · cases hc

theorem RC.release_inv (r : RC) (h : r.Inv) (ho : r.owned > 0) : r.release.Inv := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨by cases hsa : r.specAlive <;> simp [RC.release, hsa] at h1 ⊢ <;> omega, by simp [RC.release], h3⟩

theorem RC.dropSpec_inv (r : RC) (h : r.Inv) (hs : r.specAlive = true) : r.dropSpec.Inv := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨by simp [RC.dropSpec]; simp [hs] at h1; omega, by simp [RC.dropSpec], by simp [RC.dropSpec]⟩

/-- **No use-after-free**: while an iterator holds a clone, or the spec handle
is alive, the spec is not freed. -/
theorem RC.not_freed (r : RC) (h : r.Inv) (hl : r.owned > 0 ∨ r.specAlive = true) : r.freed = false := by
  obtain ⟨h1, h2, -⟩ := h
  cases hf : r.freed
  · rfl
  · have := h2.mp hf
    rcases hl with hl | hl
    · omega
    · simp [hl] at h1; omega

/-- After `DROP INDEX` a new query cannot clone: `Index::query` returns the empty iterator. -/
theorem RC.clone_after_drop (r : RC) : r.dropSpec.clone = none := by
  simp [RC.dropSpec, RC.clone]

/-- `into_raw` is load-bearing: without it, dropping the spec while one
iterator holds a clone frees the spec under that iterator. -/
theorem RC.into_raw_needed :
    let r := (RC.create.clone).getD RC.create
    r.owned = 1 ∧ r.dropSpec.freed = false ∧ r.dropSpecNoIntoRaw.freed = true := by decide

/-- `Document` ownership: `RediSearch_FreeDocument` runs in `Drop` iff the doc
was never handed to `add_document` (which sets `consumed`); RS frees consumed
docs itself. Exactly one free either way. -/
structure DocLife where
  added : Bool
def DocLife.rustFrees (d : DocLife) : Bool := !d.added   -- `!consumed && !null`
def DocLife.rsFrees (d : DocLife) : Bool := d.added
theorem DocLife.freed_once (d : DocLife) :
    (if d.rustFrees then 1 else 0) + (if d.rsFrees then 1 else 0) = 1 := by
  cases d with | mk a => cases a <;> rfl

/-! ## Query entry points -/

/-- `Index::query` (`mod.rs:1677`): no spec or a dead spec ⇒ empty; a null
query node ⇒ empty; else the results iterator over `run n`. -/
def queryN (spec : Option Nat) (cloneOk : Bool) (node : Option QN) (run : QN → List (Key × Nat)) : RIter :=
  match spec with
  | none => RIter.empty
  | some _ => if cloneOk then (match node with | none => RIter.empty | some n => RIter.new (run n)) else RIter.empty

/-- The RediSearch LLAPI contract for a node-index table `t` (doc keys are
`nodeKey id`, `mod.rs:657`): the iterator returns exactly the keys of the
stored documents that match `n`. -/
def RunSpec (t : Table) (run : QN → List (Key × Nat)) : Prop :=
  ∀ n k, (∃ s, (k, s) ∈ run n) ↔ ∃ id d, id < 2 ^ 64 ∧ t id = some d ∧ rsMatch d n ∧ k = nodeKey id

theorem queryN_mem (t : Table) (run : QN → List (Key × Nat)) (hr : RunSpec t run) (s : Nat) (n : QN)
    (id : Nat) (hid : id < 2 ^ 64) :
    id ∈ ids (queryN (some s) true (some n) run) ↔ ∃ d, t id = some d ∧ rsMatch d n := by
  simp only [ids, queryN, RIter.new, RIter.items, List.mem_map]
  constructor
  · rintro ⟨⟨b, sc⟩, hm, rfl⟩
    obtain ⟨k, hk, -, hdec⟩ := (mem_drainN _ _ _ _).mp hm
    obtain ⟨id', d, hid', ht, hmt, rfl⟩ := (hr n k).mp ⟨sc, hk⟩
    simp only at hdec
    rw [decodeId_nodeKey id' hid'] at hdec
    subst hdec; exact ⟨d, ht, hmt⟩
  · rintro ⟨d, ht, hmt⟩
    obtain ⟨sc, hk⟩ := (hr n (nodeKey id)).mpr ⟨id, d, hid, ht, hmt, rfl⟩
    exact ⟨(id, sc), (mem_drainN _ _ _ _).mpr ⟨nodeKey id, hk, nodeKey_length id, decodeId_nodeKey id hid⟩, rfl⟩

/-- **Index scan = label scan + filter** (the bridge from `Index::query` to the
Cypher predicate). If the label's RS table holds `docOf fields (P id)` for
exactly the ids carrying the label (the maintenance invariant, `Maintenance.commit_preserves_inv`)
and the query is one the exactness theorems cover (`equal_exact`, `inList_exact`,
`numRange_exact`, `strRange_exact` give `hexact`), then an id is returned iff it
has the label and satisfies the predicate. -/
theorem query_eq_scan_filter (fields : Attr → Bool) (P : Nat → Props) (lab : Nat → Bool)
    (t : Table) (run : QN → List (Key × Nat)) (hr : RunSpec t run) (s : Nat) (q : IQ)
    (ht : ∀ id, t id = if lab id then some (docOf fields (P id)) else none)
    (hexact : ∀ id, lab id = true → indexHit fields (P id) q = holds (P id) q)
    (id : Nat) (hid : id < 2 ^ 64) :
    id ∈ ids (queryN (some s) true (buildQ fields q) run) ↔ lab id = true ∧ holds (P id) q = true := by
  cases hb : buildQ fields q with
  | none =>
    have hn : ∀ hl : lab id = true, holds (P id) q = false := fun hl => by
      have := hexact id hl; simp [indexHit, hb] at this; exact this
    simp only [ids]
    simp [queryN, RIter.empty, RIter.items]
    intro hl; simp [hn hl]
  | some n =>
    rw [queryN_mem t run hr s n id hid, ht]
    constructor
    · rintro ⟨d, hd, hm⟩
      split at hd
      · next hl =>
        cases hd
        refine ⟨hl, ?_⟩
        rw [← hexact id hl]; simp [indexHit, hb, hm]
      · cases hd
    · rintro ⟨hl, hh⟩
      refine ⟨docOf fields (P id), by simp [hl], ?_⟩
      rw [← hexact id hl] at hh; simpa [indexHit, hb] using hh

/-! ### Edge indexes (`Document::new_edge`, `query_edges`) -/

/-- An edge-index table, keyed by `(src, dst, edge_id)`. -/
abbrev ETable := Nat × Nat × Nat → Option Doc

def ERunSpec (t : ETable) (run : QN → List (Key × Nat)) : Prop :=
  ∀ n k, (∃ s, (k, s) ∈ run n) ↔ ∃ s d e doc, s < 2 ^ 64 ∧ d < 2 ^ 64 ∧ e < 2 ^ 64 ∧
    t (s, d, e) = some doc ∧ rsMatch doc n ∧ k = edgeKey s d e

theorem edgeKey_length (s d e : Nat) : (edgeKey s d e).length = 48 := by
  simp [edgeKey, nodeKey_length]

theorem decTriple_edgeKey (s d e : Nat) (hs : s < 2 ^ 64) (hd : d < 2 ^ 64) (he : e < 2 ^ 64) :
    decTriple (edgeKey s d e) = (s, d, e) := by
  obtain ⟨h1, h2, h3⟩ := decode_edgeKey s d e hs hd he
  simp [decTriple, h1, h2, h3]

/-- `Index::query_edges` returns exactly the `(src, dst, edge_id)` triples whose
stored edge document matches (the triple is read back from the key, so no
tensor scan is needed). -/
theorem queryE_mem (t : ETable) (run : QN → List (Key × Nat)) (hr : ERunSpec t run) (sp : Nat) (n : QN)
    (s d e : Nat) (hs : s < 2 ^ 64) (hd : d < 2 ^ 64) (he : e < 2 ^ 64) :
    (s, d, e) ∈ triples (queryN (some sp) true (some n) run) ↔ ∃ doc, t (s, d, e) = some doc ∧ rsMatch doc n := by
  simp only [triples, queryN, RIter.new, RIter.items, List.mem_map]
  constructor
  · rintro ⟨⟨b, sc⟩, hm, rfl⟩
    obtain ⟨k, hk, -, hdec⟩ := (mem_drainN _ _ _ _).mp hm
    obtain ⟨s', d', e', doc, hs', hd', he', ht, hmt, rfl⟩ := (hr n k).mp ⟨sc, hk⟩
    simp only at hdec
    rw [decTriple_edgeKey _ _ _ hs' hd' he'] at hdec
    simp only [Prod.mk.injEq] at hdec; obtain ⟨rfl, rfl, rfl⟩ := hdec; exact ⟨doc, ht, hmt⟩
  · rintro ⟨doc, ht, hmt⟩
    obtain ⟨sc, hk⟩ := (hr n (edgeKey s d e)).mpr ⟨s, d, e, doc, hs, hd, he, ht, hmt, rfl⟩
    exact ⟨((s, d, e), sc), (mem_drainN _ _ _ _).mpr
      ⟨edgeKey s d e, hk, edgeKey_length s d e, decTriple_edgeKey s d e hs hd he⟩, rfl⟩

/-- Edge version of the headline: an edge-index scan returns exactly the edges of
the type that satisfy the predicate. -/
theorem queryE_eq_scan_filter (fields : Attr → Bool) (P : Nat × Nat × Nat → Props)
    (has : Nat × Nat × Nat → Bool) (t : ETable) (run : QN → List (Key × Nat)) (hr : ERunSpec t run)
    (sp : Nat) (q : IQ) (ht : ∀ x, t x = if has x then some (docOf fields (P x)) else none)
    (hexact : ∀ x, has x = true → indexHit fields (P x) q = holds (P x) q)
    (s d e : Nat) (hs : s < 2 ^ 64) (hd : d < 2 ^ 64) (he : e < 2 ^ 64) :
    (s, d, e) ∈ triples (queryN (some sp) true (buildQ fields q) run) ↔
      has (s, d, e) = true ∧ holds (P (s, d, e)) q = true := by
  cases hb : buildQ fields q with
  | none =>
    have hn : ∀ hl : has (s, d, e) = true, holds (P (s, d, e)) q = false := fun hl => by
      have := hexact _ hl; simp [indexHit, hb] at this; exact this
    simp only [triples]
    simp [queryN, RIter.empty, RIter.items]
    intro hl; simp [hn hl]
  | some n =>
    rw [queryE_mem t run hr sp n s d e hs hd he, ht]
    constructor
    · rintro ⟨doc, hdoc, hm⟩
      split at hdoc
      · next hl => cases hdoc; exact ⟨hl, by rw [← hexact _ hl]; simp [indexHit, hb, hm]⟩
      · cases hdoc
    · rintro ⟨hl, hh⟩
      refine ⟨docOf fields (P (s, d, e)), by simp [hl], ?_⟩
      rw [← hexact _ hl] at hh; simpa [indexHit, hb] using hh

/-! ### Full-text and vector queries (abstract RediSearch semantics) -/

/-- `fulltext_query` (`mod.rs:1718`): `CString::new` fails on a NUL ⇒ `Err`;
no / dead spec ⇒ `Ok(empty)`; `RediSearch_IterateQuery` error ⇒ `Err(msg)`;
otherwise the scored iterator. `ft s str` is RediSearch's answer (FFI). -/
def fulltextQ (hasNul : Bool) (spec : Option Nat) (cloneOk : Bool)
    (ft : Nat → Except String (List (Key × Nat))) : Except String RIter :=
  if hasNul then .error "nul byte" else
  match spec with
  | none => .ok RIter.empty
  | some s => if cloneOk then (match ft s with | .error e => .error e | .ok l => .ok (RIter.new l)) else .ok RIter.empty

/-- Full-text spec: RS returns the keys of the stored documents its query
language matches (`M` is the RS full-text match relation, left abstract). -/
def FtSpec (t : Table) (M : Doc → Prop) (l : List (Key × Nat)) : Prop :=
  ∀ k, (∃ s, (k, s) ∈ l) ↔ ∃ id d, id < 2 ^ 64 ∧ t id = some d ∧ M d ∧ k = nodeKey id

theorem fulltextQ_exact (t : Table) (M : Doc → Prop) (sp : Nat) (ft : Nat → Except String (List (Key × Nat)))
    (l : List (Key × Nat)) (hft : ft sp = .ok l) (hs : FtSpec t M l) (id : Nat) (hid : id < 2 ^ 64) :
    ∃ it, fulltextQ false (some sp) true ft = .ok it ∧
      (id ∈ (scoredIds it).map (·.1) ↔ ∃ d, t id = some d ∧ M d) := by
  refine ⟨RIter.new l, by simp [fulltextQ, hft], ?_⟩
  simp only [scoredIds, RIter.new, RIter.items, List.mem_map]
  constructor
  · rintro ⟨⟨b, sc⟩, hm, rfl⟩
    obtain ⟨k, hk, -, hdec⟩ := (mem_drainN _ _ _ _).mp hm
    obtain ⟨id', d, hid', ht, hmt, rfl⟩ := (hs k).mp ⟨sc, hk⟩
    simp only at hdec; rw [decodeId_nodeKey id' hid'] at hdec; subst hdec; exact ⟨d, ht, hmt⟩
  · rintro ⟨d, ht, hmt⟩
    obtain ⟨sc, hk⟩ := (hs (nodeKey id)).mpr ⟨id, d, hid, ht, hmt, rfl⟩
    exact ⟨(id, sc), (mem_drainN _ _ _ _).mpr ⟨nodeKey id, hk, nodeKey_length id, decodeId_nodeKey id hid⟩, rfl⟩

theorem fulltextQ_errors (spec : Option Nat) (c : Bool) (ft : Nat → Except String (List (Key × Nat))) :
    fulltextQ true spec c ft = .error "nul byte" ∧ fulltextQ false none c ft = .ok RIter.empty ∧
    (∀ s e, ft s = .error e → fulltextQ false (some s) true ft = .error e) := by
  refine ⟨rfl, rfl, fun s e h => by simp [fulltextQ, h]⟩

/-- `vector_query` (`mod.rs:1784`): the field is `vector:{attr}`; a null node or
a null results iterator ⇒ empty; else the scored iterator. `node`/`iterOk` are
`CreateVecSimNode`/`GetResultsIterator` (FFI). -/
def vectorQ (spec : Option Nat) (cloneOk nodeOk iterOk : Bool) (l : List (Key × Nat)) : RIter :=
  match spec with
  | none => RIter.empty
  | some _ => if cloneOk && nodeOk && iterOk then RIter.new l else RIter.empty

def vectorField (attr : String) : String := "vector:" ++ attr

/-- KNN spec (HNSW is approximate, so only this is promised): at most `k`
results, each a stored document carrying a vector. -/
def KnnSpec (t : Table) (hasVec : Doc → Prop) (k : Nat) (l : List (Key × Nat)) : Prop :=
  l.length ≤ k ∧ ∀ key s, (key, s) ∈ l → ∃ id d, id < 2 ^ 64 ∧ t id = some d ∧ hasVec d ∧ key = nodeKey id

theorem vectorQ_sound (t : Table) (hasVec : Doc → Prop) (k : Nat) (l : List (Key × Nat))
    (h : KnnSpec t hasVec k l) (spec : Option Nat) (c n i : Bool) :
    (scoredIds (vectorQ spec c n i l)).length ≤ k ∧
    ∀ id ∈ (scoredIds (vectorQ spec c n i l)).map (·.1), ∃ d, t id = some d ∧ hasVec d := by
  have key : (scoredIds (vectorQ spec c n i l)).length ≤ k ∧
      ∀ x ∈ scoredIds (vectorQ spec c n i l), ∃ d, t x.1 = some d ∧ hasVec d := by
    unfold vectorQ; split
    · simp [scoredIds, RIter.empty, RIter.items]
    · split
      · refine ⟨?_, fun x hx => ?_⟩
        · simp only [scoredIds, RIter.new, RIter.items, drainN]
          exact Nat.le_trans (List.length_filterMap_le _ _) h.1
        · obtain ⟨kk, hk, -, hdec⟩ := (mem_drainN _ _ _ _).mp (by simpa [scoredIds, RIter.new, RIter.items] using hx)
          obtain ⟨id, d, hid, ht, hv, rfl⟩ := h.2 _ _ hk
          rw [decodeId_nodeKey id hid] at hdec
          exact ⟨d, by rw [← hdec]; exact ht, hv⟩
      · simp [scoredIds, RIter.empty, RIter.items]
  refine ⟨key.1, fun id hid => ?_⟩
  obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hid
  exact key.2 x hx

theorem vectorField_spec (a : String) : vectorField a = "vector:" ++ a := rfl

/-! ### Document deletion keys and `commit_edge` -/

/-- `delete_document` deletes the key `Document::new` wrote. -/
def delKey (id : Nat) : Key := nodeKey id
/-- `delete_edge_document` deletes the key `Document::new_edge` wrote. -/
def delEdgeKey (s d e : Nat) : Key := edgeKey s d e

theorem delKey_spec (id : Nat) : delKey id = nodeKey id ∧ (delKey id).length = 16 :=
  ⟨rfl, nodeKey_length id⟩
theorem delEdgeKey_spec (s d e : Nat) : delEdgeKey s d e = edgeKey s d e ∧ (delEdgeKey s d e).length = 48 :=
  ⟨rfl, edgeKey_length s d e⟩

def ETable.add (t : ETable) (x : Nat × Nat × Nat) (d : Doc) : ETable := fun y => if y = x then some d else t y
def ETable.del (t : ETable) (x : Nat × Nat × Nat) : ETable := fun y => if y = x then none else t y

/-- `Indexer::commit_edge` for one relationship type: adds, then deletes by
`(src, dst, edge_id)` (the map is `edge_id → (src, dst)`). -/
def commitEdge (t : ETable) (build : Nat × Nat × Nat → Doc) (adds : List (Nat × Nat × Nat))
    (removes : List (Nat × Nat × Nat)) : ETable :=
  removes.foldl ETable.del (adds.foldl (fun t x => t.add x (build x)) t)

theorem commitEdge_lookup (t : ETable) (build : Nat × Nat × Nat → Doc) :
    ∀ (adds removes : List (Nat × Nat × Nat)) (y : Nat × Nat × Nat),
    commitEdge t build adds removes y = if y ∈ removes then none else if y ∈ adds then some (build y) else t y := by
  intro adds removes y
  have hadd : ∀ (as : List (Nat × Nat × Nat)) (t : ETable),
      (as.foldl (fun t x => t.add x (build x)) t) y = if y ∈ as then some (build y) else t y := by
    intro as
    induction as with
    | nil => simp
    | cons a as ih =>
      intro t; rw [List.foldl_cons, ih]
      by_cases h1 : y ∈ as
      · simp [h1]
      · by_cases h2 : y = a
        · subst h2; simp [ETable.add, h1]
        · simp [ETable.add, h1, h2]
  have hdel : ∀ (rs : List (Nat × Nat × Nat)) (t : ETable),
      (rs.foldl ETable.del t) y = if y ∈ rs then none else t y := by
    intro rs
    induction rs with
    | nil => simp
    | cons a as ih =>
      intro t; rw [List.foldl_cons, ih]
      by_cases h1 : y ∈ as
      · simp [h1]
      · by_cases h2 : y = a
        · subst h2; simp [ETable.del, h1]
        · simp [ETable.del, h1, h2]
  simp [commitEdge, hdel, hadd]

end IndexLayer.Iter
