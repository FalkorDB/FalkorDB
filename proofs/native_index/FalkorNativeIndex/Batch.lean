import FalkorNativeIndex.Build
/-!
# An indexer batch commits as a version of its own (#3566, 29e85ffc3), and catches up on writers (#3567, b18530683)

An indexer batch is built on a *build version* of the published graph with no lock held
(`Graph::new_build_version`, `graph.rs:1139`; `Graph::indexer_prepare`, `versioned.rs:366`), then
`MvccGraph::commit_indexer_batch` (`mvcc_graph.rs:235`) fits it to whatever was published meanwhile
(`Graph::indexer_fit`, `versioned.rs:391`). The proofs below model that logic on the one index the batch
is about (`V.entry`); everything else in the graph is the opaque `rest : R`, which a `Swap` takes from the
published version.

| Lean | Rust |
| --- | --- |
| `SE` (`en`, `generation`, `commits`) | `IndexEntry` (`index`, `state`, `generation`, `indexer_commits`; `versioned.rs:54-65`) |
| `SE.new` | `IndexEntry::new` (`:173-185`) |
| `SE.indexerApply` | `IndexEntry::indexer_apply` (`:209-221`, now bumps `indexer_commits`) |
| `discardedAgainst` | `IndexEntry::discarded_against` (`:226-241`) |
| `V` | the parts of a `Graph` the fit reads: `version`, `falkordb_index.{indexes_created, batches_published}`, the batch's entry |
| `V.newVersion`, `V.newBuildVersion` | `Graph::new_version` (`graph.rs:1084`), `Graph::new_build_version` (`:1139`) |
| `V.createEntry` | `FalkorDbIndex::insert_column` (`falkordb_index.rs:450`): `indexes_created += 1`, that is the generation |
| `V.setBatchesPublished`, `V.batchPublishedSince` | `falkordb_index.rs:565`, `:575` |
| `V.applyBatch` | `FalkorDbIndex::apply_indexer_batch` (`versioned.rs:353`) |
| `indexerFit` | `Graph::indexer_fit` (`versioned.rs:391`) |
| `V.withBatchIndex` | `Graph::with_batch_index` (`:316`) with `take_index` (`:329`) |
| `commitIndexerBatch` | `MvccGraph::commit_indexer_batch` (`mvcc_graph.rs:235`) and `commit_version(_, false)` (`:261`) |

`Graph::validate` is the abstract predicate `valid`; schema-version bump, `trim_attr_stores`,
`fold_oversized_deltas` and `set_indexer_graph` do not touch the version, the batch's entry or the
counters, and are not modelled.

**Counters are unbounded `Nat`.** The Rust generation, `indexer_commits`, data-version and batch counters
are `u64`, each bumped by one per create / commit / publish; the model treats them as unbounded naturals,
so no theorem here covers a wrap at 2^64. That is an explicit assumption (2^64 commits are not reachable),
not a proven bound.

Theorems: `fit_fresh` (nothing committed since: `Same`), `fit_after_other_batch` (`Swap`),
`fit_after_writer` (`CatchUp`, #3567: it was `Discarded Writers`), `fit_after_same_batch` (`AnotherBatch`), `fit_after_drop` /
`fit_after_recreate` (`Gone`: a dropped, or dropped and created again, index has another generation),
`commit_discarded` (nothing published), `commit_same`, `commit_swap` (only the batch's index comes from the
batch; all else from the published version), `commit_catchUp` (the same, with the batch's index caught up: `Rebase.lean`), `commit_keeps_version` / `commit_not_stale` (a batch commit
never makes a writer's fork stale), `commit_batches` (the batch counter rises by one on each publish).
-/
namespace NI

/-- `IndexEntry<IndexColumn>` with its two counters. -/
structure SE where
  en : IndexEntry
  generation : Nat
  commits : Nat

namespace SE
def new (index : IndexColumn) (state : BuildState) (generation : Nat) : SE :=
  ⟨IndexEntry.new index state, generation, 0⟩

/-- `IndexEntry::indexer_apply`: as `IndexEntry.indexerApply`, and `indexer_commits += 1`. -/
def indexerApply (s : SE) (work : EncodedTuples) (next : Option BuildCursor) : SE × EncodedTuples × Bool :=
  if s.en.isBuilding then
    let r := s.en.indexerApply work next
    (⟨r.1, s.generation, s.commits + 1⟩, r.2.1, true)
  else (s, work, false)

theorem indexerApply_idle (s : SE) (work : EncodedTuples) (next : Option BuildCursor)
    (h : s.en.isBuilding = false) : s.indexerApply work next = (s, work, false) := by
  simp [indexerApply, h]

theorem indexerApply_building (s : SE) (work : EncodedTuples) (next : Option BuildCursor)
    (h : s.en.isBuilding = true) :
    (s.indexerApply work next).2.2 = true ∧
    (s.indexerApply work next).1.en = (s.en.indexerApply work next).1 ∧
    (s.indexerApply work next).1.generation = s.generation ∧
    (s.indexerApply work next).1.commits = s.commits + 1 := by
  simp [indexerApply, h]

theorem new_commits (i : IndexColumn) (st : BuildState) (g : Nat) :
    (new i st g).commits = 0 ∧ (new i st g).generation = g := ⟨rfl, rfl⟩
end SE

/-- `Discarded` (`versioned.rs:157-164`); `Writers` is gone since #3567: a batch catches up on writers. -/
inductive Discarded where
  | gone | anotherBatch
deriving DecidableEq, Repr

/-- `IndexerCommit` (`versioned.rs:128-143`); `CatchUp(CatchUp::Refresh)` is `catchUp`. -/
inductive IndexerCommit where
  | same | swap | catchUp | discarded (why : Discarded)
deriving DecidableEq, Repr

/-- `IndexEntry::discarded_against`: `ours` is a copy moved on by a batch, `pub` the published copy. -/
def discardedAgainst (ours : SE) (pub : Option SE) : Option Discarded :=
  match pub with
  | some p => if p.generation = ours.generation then
                (if p.commits + 1 ≠ ours.commits then some .anotherBatch else none)
              else some .gone
  | none => some .gone

theorem discardedAgainst_none_iff (o : SE) (p : Option SE) :
    discardedAgainst o p = none ↔ ∃ q, p = some q ∧ q.generation = o.generation ∧ q.commits + 1 = o.commits := by
  unfold discardedAgainst
  cases p with
  | none => simp
  | some q => by_cases h : q.generation = o.generation <;> by_cases h2 : q.commits + 1 = o.commits <;> simp [h, h2]

/-- What the fit reads of a graph version. -/
structure V (R : Type) where
  rest : R
  version : Nat
  indexesCreated : Nat
  batches : Nat
  entry : Option SE

namespace V
variable {R : Type}

/-- `Graph::new_version`: the data version moves on; counters and indexes are cloned. -/
def newVersion (v : V R) : V R := { v with version := v.version + 1 }
/-- `Graph::new_build_version`: `new_version`, back at the same data version. -/
def newBuildVersion (v : V R) : V R := { v.newVersion with version := v.version }

theorem newBuildVersion_spec (v : V R) : v.newBuildVersion = v := rfl
theorem newVersion_version (v : V R) : v.newVersion.version = v.version + 1 := rfl

/-- `insert_column` for the batch's index: the next generation, no batches yet. -/
def createEntry (v : V R) (index : IndexColumn) (state : BuildState) : V R :=
  { v with indexesCreated := v.indexesCreated + 1,
           entry := some (SE.new index state (v.indexesCreated + 1)) }
/-- `drop_column`. -/
def dropEntry (v : V R) : V R := { v with entry := none }

theorem createEntry_generation (v : V R) (i : IndexColumn) (st : BuildState) :
    (v.createEntry i st).entry.map (·.generation) = some (v.indexesCreated + 1) ∧
    (v.createEntry i st).indexesCreated = v.indexesCreated + 1 := ⟨rfl, rfl⟩

/-- `set_batches_published`: the published graph's count, plus one for a batch's version. -/
def setBatchesPublished (v : V R) (pub : V R) (indexerBatch : Bool) : V R :=
  { v with batches := pub.batches + if indexerBatch then 1 else 0 }

/-- `batch_published_since`: `pub.batch_published_since(older)`. -/
def batchPublishedSince (pub older : V R) : Bool := pub.batches != older.batches

theorem setBatchesPublished_writer (v pub : V R) : (v.setBatchesPublished pub false).batches = pub.batches := rfl
theorem setBatchesPublished_batch (v pub : V R) : (v.setBatchesPublished pub true).batches = pub.batches + 1 := rfl

/-- A batch published after `older` (a version whose count `pub` extends) moves the count. -/
theorem batchPublishedSince_after_batch (pub older : V R) (h : pub.batches = older.batches + 1) :
    batchPublishedSince pub older = true := by simp [batchPublishedSince, h]
theorem batchPublishedSince_same (pub : V R) : batchPublishedSince pub pub = false := by simp [batchPublishedSince]

/-- `apply_indexer_batch`: the batch's index, if still `Building`, takes the work and moves on. -/
def applyBatch (v : V R) (work : EncodedTuples) (next : Option BuildCursor) : V R × Bool :=
  match v.entry with
  | none => (v, false)
  | some s => let r := s.indexerApply work next; ({ v with entry := some r.1 }, r.2.2)

/-- `with_batch_index`: a new build version of `pub` with the batch's index (`ours`) in place of its own,
if both have one (`take_index`). -/
def withBatchIndex (ours pub : V R) : V R :=
  match ours.entry, pub.newBuildVersion.entry with
  | some t, some _ => { pub.newBuildVersion with entry := some t }
  | _, _ => pub.newBuildVersion

/-- `Graph::caught_up_onto` with `catch_up_index` (`versioned.rs:438-473`): a new build version of `pub`, the
batch's entry in its slot with its index caught up on the writers' changes. `f ours pub` is the caught-up index
(`Rebase.lean`: `IndexColumn.refresh`); the entry's state, generation and commit count are the batch's
(`caught = ours.clone()`). -/
def caughtUpOnto (ours pub : V R) (f : SE → SE → IndexColumn) : V R :=
  match ours.entry, pub.newBuildVersion.entry with
  | some o, some p => { pub.newBuildVersion with entry := some { o with en := { o.en with index := f o p } } }
  | _, _ => pub.newBuildVersion
end V

/-- `Graph::indexer_fit`. `ours` is the batch's version, `pub` the published one. -/
def indexerFit {R : Type} (ours pub : V R) : IndexerCommit :=
  match ours.entry with
  | none => .discarded .gone
  | some o =>
    match discardedAgainst o pub.entry with
    | some why => .discarded why
    | none =>
      if pub.version ≠ ours.version then .catchUp
      else if V.batchPublishedSince pub ours then .swap else .same

/-- The version a batch commit publishes, for each fit (`commit_indexer_batch`, `mvcc_graph.rs:235-260`). -/
def commitVersion {R : Type} (pub prepared : V R) (f : SE → SE → IndexColumn) : IndexerCommit → V R
  | .same => prepared
  | .swap => V.withBatchIndex prepared pub
  | .catchUp => V.caughtUpOnto prepared pub f
  | .discarded _ => pub

/-- `MvccGraph::commit_indexer_batch` (`valid` is `Graph::validate`): the new published version and what
went in; `none` is the `Invalid` refusal, nothing published. `f` is the catch-up of the batch's index. -/
def commitIndexerBatch {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn) :
    V R × Option IndexerCommit :=
  let fit := indexerFit prepared pub
  match fit with
  | .discarded _ => (pub, some fit)
  | _ =>
    let ver := commitVersion pub prepared f fit
    if valid ver then (ver.setBatchesPublished pub true, some fit) else (pub, none)

/-! ## The fit -/

theorem fit_gone_of_no_entry {R : Type} (ours pub : V R) (h : ours.entry = none) :
    indexerFit ours pub = .discarded .gone := by simp [indexerFit, h]

/-- Nothing committed since the version was made: a batch applied to a fresh build version of the published
graph fits as `Same`. -/
theorem fit_fresh {R : Type} (pub : V R) (p : SE) (hp : pub.entry = some p) (hb : p.en.isBuilding = true)
    (work : EncodedTuples) (next : Option BuildCursor) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub = .same := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst]
  simp [ha.2.2.1, ha.2.2.2, V.batchPublishedSince]

/-- Only another index's batch committed meanwhile (the count rose, nothing of this index or the data
changed): the fit is `Swap`. -/
theorem fit_after_other_batch {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version = pub.version) (hcnt : pub'.batches = pub.batches + 1) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub' = .swap := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent, hver]
  simp [ha.2.2.1, ha.2.2.2, V.batchPublishedSince, hcnt]

/-- A writer committed meanwhile (the data version moved, this index untouched): `CatchUp` (#3567; before it,
`Discarded Writers`). -/
theorem fit_after_writer {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version ≠ pub.version) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub' = .catchUp := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hver]

/-- Another batch committed on this index meanwhile: `AnotherBatch`, whatever else happened. -/
theorem fit_after_same_batch {R : Type} (pub pub' : V R) (p p' : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = some p') (hg : p'.generation = p.generation) (hc : p'.commits = p.commits + 1) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .anotherBatch := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hg, hc]

/-- The index was dropped meanwhile: `Gone`. -/
theorem fit_after_drop {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = none) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .gone := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]

/-- The index was dropped and created again meanwhile (the published version's count of created indexes
only rises, and the new entry takes the next one): another generation, `Gone`. -/
theorem fit_after_recreate {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hle : p.generation ≤ pub.indexesCreated) (hmono : pub.indexesCreated ≤ pub'.indexesCreated)
    (i : IndexColumn) (st : BuildState) (hent : pub'.entry = some (SE.new i st (pub'.indexesCreated + 1))) :
    indexerFit (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .gone := by
  have ha := SE.indexerApply_building p work next hb
  have hne : pub'.indexesCreated + 1 ≠ p.generation := by omega
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent, SE.new]
  simp [ha.2.2.1, hne]

/-! ## The commit -/

theorem commit_discarded {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (why : Discarded) (h : indexerFit prepared pub = .discarded why) :
    commitIndexerBatch pub prepared valid f = (pub, some (.discarded why)) := by
  simp [commitIndexerBatch, h]

theorem commit_same {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (h : indexerFit prepared pub = .same) (hv : valid prepared = true) :
    commitIndexerBatch pub prepared valid f = (prepared.setBatchesPublished pub true, some .same) := by
  simp [commitIndexerBatch, h, commitVersion, hv]

/-- `Swap`: the published version, at its own data version, with exactly the batch's index replaced. -/
theorem commit_swap {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (h : indexerFit prepared pub = .swap) (hv : valid (V.withBatchIndex prepared pub) = true) :
    commitIndexerBatch pub prepared valid f =
      ((V.withBatchIndex prepared pub).setBatchesPublished pub true, some .swap) ∧
    (V.withBatchIndex prepared pub).rest = pub.rest ∧
    (V.withBatchIndex prepared pub).version = pub.version ∧
    (V.withBatchIndex prepared pub).indexesCreated = pub.indexesCreated ∧
    (V.withBatchIndex prepared pub).entry = (match prepared.entry, pub.entry with
        | some t, some _ => some t | _, e => e) := by
  refine ⟨by simp [commitIndexerBatch, h, commitVersion, hv], ?_, ?_, ?_, ?_⟩
  all_goals
    simp only [V.withBatchIndex, V.newBuildVersion, V.newVersion]
    cases prepared.entry <;> cases pub.entry <;> simp

/-- `CatchUp` (#3567): the published version, at its own data version (the writers' data), with exactly the
batch's index replaced by the batch's index caught up on what the writers changed: its state, generation and
commit count are the batch's. -/
theorem commit_catchUp {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (h : indexerFit prepared pub = .catchUp) (hv : valid (V.caughtUpOnto prepared pub f) = true) :
    commitIndexerBatch pub prepared valid f =
      ((V.caughtUpOnto prepared pub f).setBatchesPublished pub true, some .catchUp) ∧
    (V.caughtUpOnto prepared pub f).rest = pub.rest ∧
    (V.caughtUpOnto prepared pub f).version = pub.version ∧
    (V.caughtUpOnto prepared pub f).indexesCreated = pub.indexesCreated ∧
    (V.caughtUpOnto prepared pub f).entry = (match prepared.entry, pub.entry with
        | some o, some p => some { o with en := { o.en with index := f o p } } | _, e => e) := by
  refine ⟨by simp [commitIndexerBatch, h, commitVersion, hv], ?_, ?_, ?_, ?_⟩
  all_goals
    simp only [V.caughtUpOnto, V.newBuildVersion, V.newVersion]
    cases prepared.entry <;> cases pub.entry <;> simp

/-- A refused version (`validate` fails) publishes nothing. -/
theorem commit_invalid {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (c : IndexerCommit) (h : indexerFit prepared pub = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = false) :
    commitIndexerBatch pub prepared valid f = (pub, none) := by
  cases c with
  | discarded w => exact absurd rfl (hc w)
  | same => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]
  | swap => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]
  | catchUp => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]

/-- The data version never moves with a batch commit, so a writer's fork (made at `pub.version + 1`) is
not made stale by one (`MvccGraph::is_stale`: `published.version + 1 != fork.version`). -/
def isStale {R : Type} (pub fork : V R) : Bool := pub.version + 1 != fork.version

/-- `CatchUp` and `Swap` are built on `pub.new_build_version()`, so keep its data version; `Same` is at it. -/
theorem commit_keeps_version {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (hsame : indexerFit prepared pub = .same → prepared.version = pub.version) :
    (commitIndexerBatch pub prepared valid f).1.version = pub.version := by
  unfold commitIndexerBatch
  generalize hf : indexerFit prepared pub = fit
  cases fit with
  | discarded w => rfl
  | same =>
    have := hsame hf
    simp only [commitVersion]
    by_cases hv : valid prepared = true <;> simp [hv, V.setBatchesPublished, this]
  | swap =>
    simp only [commitVersion]
    by_cases hv : valid (V.withBatchIndex prepared pub) = true
    · simp only [hv, ↓reduceIte]
      simp only [V.setBatchesPublished, V.withBatchIndex, V.newBuildVersion, V.newVersion]
      cases prepared.entry <;> cases pub.entry <;> simp
    · simp [hv]
  | catchUp =>
    simp only [commitVersion]
    by_cases hv : valid (V.caughtUpOnto prepared pub f) = true
    · simp only [hv, ↓reduceIte]
      simp only [V.setBatchesPublished, V.caughtUpOnto, V.newBuildVersion, V.newVersion]
      cases prepared.entry <;> cases pub.entry <;> simp
    · simp [hv]

/-- `Same` implies the batch's version is at the published data version (what `commit_version`'s
`debug_assert!` checks). -/
theorem fit_same_version {R : Type} (ours pub : V R) (h : indexerFit ours pub = .same) : ours.version = pub.version := by
  unfold indexerFit at h
  split at h
  · cases h
  · split at h
    · cases h
    · by_cases hv : pub.version = ours.version
      · exact hv.symm
      · simp [hv] at h

theorem commit_not_stale {R : Type} (pub prepared fork : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn) :
    isStale (commitIndexerBatch pub prepared valid f).1 fork = isStale pub fork := by
  have := commit_keeps_version pub prepared valid f (fun h => fit_same_version prepared pub h)
  simp [isStale, this]

/-- Each publish counts one batch. -/
theorem commit_batches {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : SE → SE → IndexColumn)
    (c : IndexerCommit) (hf : indexerFit prepared pub = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = true) :
    (commitIndexerBatch pub prepared valid f).1.batches = pub.batches + 1 := by
  cases c with
  | discarded w => exact absurd rfl (hc w)
  | same =>
    simp only [commitVersion] at hv
    simp only [commitIndexerBatch, hf, commitVersion]
    simp only [hv, ↓reduceIte]
    rfl
  | swap =>
    simp only [commitVersion] at hv
    simp only [commitIndexerBatch, hf, commitVersion]
    simp only [hv, ↓reduceIte]
    rfl
  | catchUp =>
    simp only [commitVersion] at hv
    simp only [commitIndexerBatch, hf, commitVersion]
    simp only [hv, ↓reduceIte]
    rfl

end NI
