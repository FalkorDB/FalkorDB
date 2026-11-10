import FalkorNativeIndex.Build
/-!
# An indexer batch commits as a version of its own (#3566, 29e85ffc3), and catches up on writers (#3567, b18530683)

An indexer batch is built on a *build version* of the published graph with no lock held
(`Graph::new_build_version`, `graph.rs:1139`; `Graph::indexer_prepare`, `versioned.rs:366`), then
`MvccGraph::commit_indexer_batch` (`mvcc_graph.rs:264`) fits it to whatever was published meanwhile
(`Graph::indexer_fit`, `versioned.rs:504`). The proofs below model that logic on the one index the batch
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
| `indexerFit`, `catchUpBy`, `catchUpWork`, `CatchUp` | `Graph::indexer_fit` (`versioned.rs:504`), `catch_up_by` (`:209`), `CatchUp` (`:149`): since #3568 the fit picks Refresh or Insert (`fit_catchUp_how`); since #2355 a catch-up past the batch's limit is discarded (`fit_after_writer_too_large`); the fit reads `SE.written` (the published copy's `since_last_indexer_step.len()`, `SE.since`) and `batchLen work` (`batch_len(&work)`) |
| `V.caughtUp`, `V.insertedOnto` | `Graph::caught_up_onto` (`:519`), `Graph::insert_index` (`:538`) |
| `V.withBatchIndex` | `Graph::with_batch_index` (`:316`) with `take_index` (`:329`) |
| `commitIndexerBatch` | `MvccGraph::commit_indexer_batch` (`mvcc_graph.rs:264`) and `commit_version(_, false)` (`:293-388`), up to the swap |
| `commitIndexerBatchE`, `CommitError` | the same with its `compare_and_swap` (#3545, `mvcc_graph.rs:372-377`) and `Err(Invalid)` / `Err(Stale)` |

Since #3545 a commit publishes by compare-and-swap against the version it loaded (`CommitError::Stale` if
another publisher slipped in): `commitIndexerBatchE` models it (`commit_cas_stale`, `commit_cas_current`).

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
  /-- `Written::since_last_indexer_step.len()`: the one thing of `Written` the fit reads (`Rebase.lean`'s `EN.proj`
  takes it from the entry's `Written`; `0` when the index is not `Building`, see `SE.written`). -/
  since : Nat

namespace SE
def new (index : IndexColumn) (state : BuildState) (generation : Nat) : SE :=
  ⟨IndexEntry.new index state, generation, 0, 0⟩

/-- What `Graph::indexer_fit` reads of the published copy: `since_last_indexer_step.len()` while `Building`,
`0` for a `Ready` index (`versioned.rs:518-521`). -/
def written (s : SE) : Nat := if s.en.isBuilding then s.since else 0

/-- `IndexEntry::indexer_apply`: as `IndexEntry.indexerApply`, and `indexer_commits += 1`. -/
def indexerApply (s : SE) (work : EncodedTuples) (next : Option BuildCursor) : SE × EncodedTuples × Bool :=
  if s.en.isBuilding then
    let r := s.en.indexerApply work next
    (⟨r.1, s.generation, s.commits + 1, 0⟩, r.2.1, true)
  else (s, work, false)

theorem indexerApply_idle (s : SE) (work : EncodedTuples) (next : Option BuildCursor)
    (h : s.en.isBuilding = false) : s.indexerApply work next = (s, work, false) := by
  simp [indexerApply, h]

theorem indexerApply_building (s : SE) (work : EncodedTuples) (next : Option BuildCursor)
    (h : s.en.isBuilding = true) :
    (s.indexerApply work next).2.2 = true ∧
    (s.indexerApply work next).1.en = (s.en.indexerApply work next).1 ∧
    (s.indexerApply work next).1.generation = s.generation ∧
    (s.indexerApply work next).1.commits = s.commits + 1 ∧
    (s.indexerApply work next).1.since = 0 := by
  simp [indexerApply, h]

theorem new_commits (i : IndexColumn) (st : BuildState) (g : Nat) :
    (new i st g).commits = 0 ∧ (new i st g).generation = g ∧ (new i st g).since = 0 := ⟨rfl, rfl, rfl⟩

/-- A `Ready` index counts no writes, whatever its `since` field says. -/
theorem written_ready (s : SE) (h : s.en.isBuilding = false) : s.written = 0 := by simp [written, h]
end SE

/-- `Discarded` (`versioned.rs:224-234`); `Writers` is gone since #3567: a batch catches up on writers. -/
inductive Discarded where
  | gone | anotherBatch
  /-- `CatchUpTooLarge` (`versioned.rs:233`, the online build, #2355): writers committed, and catching up would
  do more than the batch's limit allows (`PreparedBatch::limit_catch_up`). -/
  | catchUpTooLarge
deriving DecidableEq, Repr

/-- `CatchUp` (`versioned.rs:183-190`): how a batch that writers overtook catches up. -/
inductive CatchUp where
  | refresh | insert
deriving DecidableEq, Repr

/-- `REFRESH_COST` (`versioned.rs:171`): refreshing one entity costs as much as inserting ten tuples. -/
def REFRESH_COST : Nat := 10

/-- `catch_up_by` (`versioned.rs:209`): refresh iff `written * REFRESH_COST < batch` (`u64::saturating_mul`,
see `Rebase.catchUpBy_saturating`). -/
def catchUpBy (written batch : Nat) : CatchUp :=
  if written * REFRESH_COST < batch then .refresh else .insert

/-- `IndexerCommit` (`versioned.rs:128-143`). `catchUp how` is `CatchUp(how)`: since #3568 the fit picks
`Refresh` or `Insert` (`catchUpBy`, from the published copy's `since` count and the batch's length). -/
inductive IndexerCommit where
  | same | swap | catchUp (how : CatchUp) | discarded (why : Discarded)
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

/-- `Graph::insert_index` (`versioned.rs:538-575`) on a `V`: `published`'s copy of the index (generation
included) with the batch inserted (`g o p` is the column `indexer_insert` leaves), carrying the batch's state
and commit count. An index that is no longer `Building` in `published` is left as it is. -/
def insertedOnto (ours pub : V R) (g : SE → SE → IndexColumn) : V R :=
  match ours.entry, pub.newBuildVersion.entry with
  | some o, some p =>
    if p.en.isBuilding then
      { pub.newBuildVersion with entry := some ⟨⟨g o p, o.en.state⟩, p.generation, o.commits, o.since⟩ }
    else pub.newBuildVersion
  | _, _ => pub.newBuildVersion

/-- `Graph::caught_up_onto` with `catch_up_index` (`versioned.rs:438-473`): a new build version of `pub`, the
batch's entry in its slot with its index caught up on the writers' changes. `f ours pub` is the caught-up index
(`Rebase.lean`: `IndexColumn.refresh`); the entry's state, generation and commit count are the batch's
(`caught = ours.clone()`). -/
def caughtUpOnto (ours pub : V R) (f : SE → SE → IndexColumn) : V R :=
  match ours.entry, pub.newBuildVersion.entry with
  | some o, some p => { pub.newBuildVersion with entry := some { o with en := { o.en with index := f o p } } }
  | _, _ => pub.newBuildVersion
/-- `Graph::caught_up_onto` (`versioned.rs:519`) for the way `how` says: refresh the batch's copy, or insert the
batch into the published one. `f how` is the column that way builds (`Rebase.lean`: `IndexColumn.refresh`,
`IndexColumn.indexerInsert`). -/
def caughtUp (how : CatchUp) (ours pub : V R) (f : CatchUp → SE → SE → IndexColumn) : V R :=
  match how with
  | .refresh => caughtUpOnto ours pub (f .refresh)
  | .insert => insertedOnto ours pub (f .insert)
end V

/-- `VersionedIndex::batch_len` for `IndexColumn` (`falkordb_index.rs:382`) = `EncodedTuples::len`. -/
def IndexColumn.batchLen (b : EncodedTuples) : Nat := b.len

/-- What a catch-up the way `how` does, as `indexer_fit` counts it (`versioned.rs:527-530`): a refresh reads
again the `written` entities; an insert adds the batch's tuples and reads again the written entities in the
batch's range, at most the `read` it was given. -/
def catchUpWork (how : CatchUp) (written batchLen read : Nat) : Nat :=
  match how with
  | .refresh => written
  | .insert => batchLen + min written read

/-- `Graph::indexer_fit` (`versioned.rs:504`). `ours` is the batch's version, `pub` the published one, `work` the
batch's tuples (`column.work`), `limit` its catch-up limit (`catch_up_limit`, `none` for a batch at the floor) and
`read` what it read (`column.read`). The two numbers `catch_up_by` weighs when writers committed are read off them
as Rust reads them: the published copy's `since_last_indexer_step.len()` (`SE.written`, `0` for a `Ready` index
or no entry) and `batch_len(&work)`. A catch-up that would do more than `limit` is discarded (`:531`). -/
def indexerFit {R : Type} (work : EncodedTuples) (ours pub : V R) (limit : Option Nat := none) (read : Nat := 0) :
    IndexerCommit :=
  match ours.entry with
  | none => .discarded .gone
  | some o =>
    match discardedAgainst o pub.entry with
    | some why => .discarded why
    | none =>
      if pub.version ≠ ours.version then
        if limit.any (fun l => l < catchUpWork (catchUpBy ((pub.entry.map SE.written).getD 0) (IndexColumn.batchLen work))
            ((pub.entry.map SE.written).getD 0) (IndexColumn.batchLen work) read) then .discarded .catchUpTooLarge
        else .catchUp (catchUpBy ((pub.entry.map SE.written).getD 0) (IndexColumn.batchLen work))
      else if V.batchPublishedSince pub ours then .swap else .same

/-- The version a batch commit publishes, for each fit (`commit_indexer_batch`, `mvcc_graph.rs:264-287`). -/
def commitVersion {R : Type} (pub prepared : V R) (f : CatchUp → SE → SE → IndexColumn) : IndexerCommit → V R
  | .same => prepared
  | .swap => V.withBatchIndex prepared pub
  | .catchUp how => V.caughtUp how prepared pub f
  | .discarded _ => pub

variable (work : EncodedTuples)

/-- `MvccGraph::commit_indexer_batch` (`valid` is `Graph::validate`): the new published version and what
went in; `none` is the `Invalid` refusal, nothing published. `f` is the catch-up of the batch's index. -/
def commitIndexerBatch {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (limit : Option Nat := none) (read : Nat := 0) : V R × Option IndexerCommit :=
  let fit := indexerFit work prepared pub limit read
  match fit with
  | .discarded _ => (pub, some fit)
  | _ =>
    let ver := commitVersion pub prepared f fit
    if valid ver then (ver.setBatchesPublished pub true, some fit) else (pub, none)

/-- The two refusals of `MvccGraph::commit_indexer_batch`: `Graph::validate` (`Invalid`) and the
compare-and-swap (`Stale`, #3545). -/
inductive CommitError where
  | invalid
  | stale
  deriving DecidableEq, Repr

/-- `MvccGraph::commit_indexer_batch` with its compare-and-swap (`commit_version`, `mvcc_graph.rs:293-388`, the swap at `:372-377`).
`pub` is the published version the commit loaded and fitted against; `now` is what is published when the swap
runs, and `unchanged` is `Arc::ptr_eq(&previous, &published)`: nothing else was published in between (always so
under the host's global lock; a publisher without it breaks that). A `Discarded` fit returns before the commit,
`validate` refuses before the swap, and a refused swap leaves `now` published. Returns the published version
afterwards and the outcome. -/
def commitIndexerBatchE {R : Type} (pub now prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (unchanged : Bool) (limit : Option Nat := none) (read : Nat := 0) : V R × Except CommitError IndexerCommit :=
  let fit := indexerFit work prepared pub limit read
  match fit with
  | .discarded _ => (now, .ok fit)
  | _ =>
    let ver := commitVersion pub prepared f fit
    if !valid ver then (now, .error .invalid)
    else if unchanged then (ver.setBatchesPublished pub true, .ok fit)
    else (now, .error .stale)

/-! ## The fit -/

theorem fit_gone_of_no_entry {R : Type} (ours pub : V R) (h : ours.entry = none) :
    indexerFit work ours pub = .discarded .gone := by simp [indexerFit, h]

/-- Nothing committed since the version was made: a batch applied to a fresh build version of the published
graph fits as `Same`. -/
theorem fit_fresh {R : Type} (pub : V R) (p : SE) (hp : pub.entry = some p) (hb : p.en.isBuilding = true)
    (work : EncodedTuples) (next : Option BuildCursor) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub = .same := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst]
  simp [ha.2.2.1, ha.2.2.2, V.batchPublishedSince]

/-- Only another index's batch committed meanwhile (the count rose, nothing of this index or the data
changed): the fit is `Swap`. -/
theorem fit_after_other_batch {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version = pub.version) (hcnt : pub'.batches = pub.batches + 1) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' = .swap := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent, hver]
  simp [ha.2.2.1, ha.2.2.2, V.batchPublishedSince, hcnt]

/-- A writer committed meanwhile (the data version moved, this index untouched), for a batch with no catch-up limit
(one at the floor): `CatchUp` (#3567; before it, `Discarded Writers`), by whichever way `catch_up_by` picks (#3568)
from the published copy's `since` count and the batch's length. A batch with a limit: `fit_after_writer_within`,
`fit_after_writer_too_large`. -/
theorem fit_after_writer {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version ≠ pub.version) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' =
      .catchUp (catchUpBy ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work)) := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hver]

/-- A writer committed meanwhile, and the catch-up the fit picks stays within the batch's limit, or the batch has
none: `CatchUp`, as `fit_after_writer`. -/
theorem fit_after_writer_within {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version ≠ pub.version) (limit : Option Nat) (read : Nat)
    (hlim : ∀ l, limit = some l → catchUpWork (catchUpBy ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work))
      ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work) read ≤ l) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' limit read =
      .catchUp (catchUpBy ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work)) := by
  have ha := SE.indexerApply_building p work next hb
  have hno : limit.any (fun l => l < catchUpWork (catchUpBy ((pub'.entry.map SE.written).getD 0)
      (IndexColumn.batchLen work)) ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work) read) = false := by
    cases limit with
    | none => rfl
    | some l => simpa using hlim l rfl
  have hw : (pub'.entry.map SE.written).getD 0 = p.written := by rw [hent, hp]; rfl
  rw [hw] at hno
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hver, hno]

/-- A writer committed meanwhile, and the catch-up the fit picks would do more than the batch's limit:
`CatchUpTooLarge`, so nothing is published (the online build, #2355). -/
theorem fit_after_writer_too_large {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = pub.entry) (hver : pub'.version ≠ pub.version) (l read : Nat)
    (hl : l < catchUpWork (catchUpBy ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work))
      ((pub'.entry.map SE.written).getD 0) (IndexColumn.batchLen work) read) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' (some l) read = .discarded .catchUpTooLarge := by
  have ha := SE.indexerApply_building p work next hb
  have hw : (pub'.entry.map SE.written).getD 0 = p.written := by rw [hent, hp]; rfl
  rw [hw] at hl
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hver, hl]

/-- Another batch committed on this index meanwhile: `AnotherBatch`, whatever else happened. -/
theorem fit_after_same_batch {R : Type} (pub pub' : V R) (p p' : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = some p') (hg : p'.generation = p.generation) (hc : p'.commits = p.commits + 1) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .anotherBatch := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]
  simp [ha.2.2.1, ha.2.2.2, hg, hc]

/-- The index was dropped meanwhile: `Gone`. -/
theorem fit_after_drop {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hent : pub'.entry = none) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .gone := by
  have ha := SE.indexerApply_building p work next hb
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent]

/-- The index was dropped and created again meanwhile (the published version's count of created indexes
only rises, and the new entry takes the next one): another generation, `Gone`. -/
theorem fit_after_recreate {R : Type} (pub pub' : V R) (p : SE) (hp : pub.entry = some p)
    (hb : p.en.isBuilding = true) (work : EncodedTuples) (next : Option BuildCursor)
    (hle : p.generation ≤ pub.indexesCreated) (hmono : pub.indexesCreated ≤ pub'.indexesCreated)
    (i : IndexColumn) (st : BuildState) (hent : pub'.entry = some (SE.new i st (pub'.indexesCreated + 1))) :
    indexerFit work (pub.newBuildVersion.applyBatch work next).1 pub' = .discarded .gone := by
  have ha := SE.indexerApply_building p work next hb
  have hne : pub'.indexesCreated + 1 ≠ p.generation := by omega
  simp only [V.newBuildVersion, V.newVersion, V.applyBatch, hp, indexerFit, discardedAgainst, hent, SE.new]
  simp [ha.2.2.1, hne]

/-! ## The commit -/

variable (limit : Option Nat) (read : Nat)

theorem commit_discarded {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (why : Discarded) (h : indexerFit work prepared pub limit read = .discarded why) :
    commitIndexerBatch work pub prepared valid f limit read = (pub, some (.discarded why)) := by
  simp [commitIndexerBatch, h]

theorem commit_same {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (h : indexerFit work prepared pub limit read = .same) (hv : valid prepared = true) :
    commitIndexerBatch work pub prepared valid f limit read = (prepared.setBatchesPublished pub true, some .same) := by
  simp [commitIndexerBatch, h, commitVersion, hv]

/-- `Swap`: the published version, at its own data version, with exactly the batch's index replaced. -/
theorem commit_swap {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (h : indexerFit work prepared pub limit read = .swap) (hv : valid (V.withBatchIndex prepared pub) = true) :
    commitIndexerBatch work pub prepared valid f limit read =
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

/-- `CatchUp` (#3567, #3568), either way: the published version, at its own data version (the writers' data),
with only the batch's index slot changed, by the way `how` says; the commit counts one batch. -/
theorem commit_catchUp {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (how : CatchUp) (h : indexerFit work prepared pub limit read = .catchUp how)
    (hv : valid (V.caughtUp how prepared pub f) = true) :
    commitIndexerBatch work pub prepared valid f limit read =
      ((V.caughtUp how prepared pub f).setBatchesPublished pub true, some (.catchUp how)) ∧
    (V.caughtUp how prepared pub f).rest = pub.rest ∧
    (V.caughtUp how prepared pub f).version = pub.version ∧
    (V.caughtUp how prepared pub f).indexesCreated = pub.indexesCreated := by
  refine ⟨by simp [commitIndexerBatch, h, commitVersion, hv], ?_, ?_, ?_⟩
  all_goals
    cases how <;>
    simp only [V.caughtUp, V.caughtUpOnto, V.insertedOnto, V.newBuildVersion, V.newVersion] <;>
    cases prepared.entry <;> cases pub.entry <;> simp <;> split <;> simp

/-- `CatchUp::Refresh`: the batch's index, caught up on what the writers changed (`f .refresh`), in the
published version; its state, generation and commit count are the batch's. -/
theorem commit_catchUp_refresh {R : Type} (pub prepared : V R) (f : CatchUp → SE → SE → IndexColumn) :
    (V.caughtUp .refresh prepared pub f).entry = (match prepared.entry, pub.entry with
        | some o, some p => some { o with en := { o.en with index := f .refresh o p } } | _, e => e) := by
  simp only [V.caughtUp, V.caughtUpOnto, V.newBuildVersion, V.newVersion]
  cases prepared.entry <;> cases pub.entry <;> simp

/-- `CatchUp::Insert` (#3568): the published copy of the index with the batch inserted (`f .insert`), carrying
the batch's state and commit count; an index no longer `Building` in the published version is left alone. -/
theorem commit_catchUp_insert {R : Type} (pub prepared : V R) (f : CatchUp → SE → SE → IndexColumn) :
    (V.caughtUp .insert prepared pub f).entry = (match prepared.entry, pub.entry with
        | some o, some p => if p.en.isBuilding then some ⟨⟨f .insert o p, o.en.state⟩, p.generation, o.commits, o.since⟩
                            else some p
        | _, e => e) := by
  simp only [V.caughtUp, V.insertedOnto, V.newBuildVersion, V.newVersion]
  cases prepared.entry <;> cases pub.entry <;> simp
  split <;> simp

/-- **#3545, a current commit publishes**: when nothing was published since the load (`now = pub`, the
swap's pointer check passes) `commitIndexerBatchE` is exactly the commit above, with `Invalid` for its refusal. -/
theorem commit_cas_current {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn) :
    commitIndexerBatchE work pub pub prepared valid f true limit read =
      ((commitIndexerBatch work pub prepared valid f limit read).1,
        match (commitIndexerBatch work pub prepared valid f limit read).2 with
        | some fit => .ok fit
        | none => .error .invalid) := by
  unfold commitIndexerBatchE commitIndexerBatch
  generalize indexerFit work prepared pub limit read = fit
  cases fit with
  | discarded w => simp
  | same => by_cases hv : valid prepared = true <;> simp [commitVersion, hv]
  | swap => by_cases hv : valid (V.withBatchIndex prepared pub) = true <;> simp [commitVersion, hv]
  | catchUp how => by_cases hv : valid (V.caughtUp how prepared pub f) = true <;> simp [commitVersion, hv]

/-- **#3545, a stale commit publishes nothing**: another version was published between the load and the swap
(`unchanged = false`), `validate` accepts, and the fit is not `Discarded`: the commit is refused with `Stale`, and
what is published is still `now`, the other publisher's version, untouched. -/
theorem commit_cas_stale {R : Type} (pub now prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (c : IndexerCommit) (h : indexerFit work prepared pub limit read = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = true) :
    commitIndexerBatchE work pub now prepared valid f false limit read = (now, .error .stale) := by
  cases c with
  | discarded w => exact absurd rfl (hc w)
  | same => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]
  | swap => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]
  | catchUp how => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]

/-- A refused version (`validate` fails) publishes nothing, whether or not another publisher slipped in:
the validation refusal comes before the swap. -/
theorem commit_cas_invalid {R : Type} (pub now prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (c : IndexerCommit) (h : indexerFit work prepared pub limit read = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = false) (u : Bool) :
    commitIndexerBatchE work pub now prepared valid f u limit read = (now, .error .invalid) := by
  cases c with
  | discarded w => exact absurd rfl (hc w)
  | same => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]
  | swap => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]
  | catchUp how => simp [commitIndexerBatchE, h, commitVersion] at hv ⊢; simp [hv]

/-- **The fit makes the Refresh / Insert choice** (#3568): when it says `CatchUp how`, `how` is `catch_up_by` of
the published copy's `since` count and the batch's length. -/
theorem fit_catchUp_how {R : Type} (ours pub : V R) (how : CatchUp)
    (h : indexerFit work ours pub limit read = .catchUp how) :
    how = catchUpBy ((pub.entry.map SE.written).getD 0) (IndexColumn.batchLen work) := by
  unfold indexerFit at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · split at h
        · cases h
        · cases h; rfl
      · split at h <;> cases h

/-- A refused version (`validate` fails) publishes nothing. -/
theorem commit_invalid {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (c : IndexerCommit) (h : indexerFit work prepared pub limit read = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = false) :
    commitIndexerBatch work pub prepared valid f limit read = (pub, none) := by
  cases c with
  | discarded w => exact absurd rfl (hc w)
  | same => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]
  | swap => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]
  | catchUp how => simp [commitIndexerBatch, h, commitVersion] at hv ⊢; simp [hv]

/-- The data version never moves with a batch commit, so a writer's fork (made at `pub.version + 1`) is
not made stale by one (`MvccGraph::is_stale`: `published.version + 1 != fork.version`). -/
def isStale {R : Type} (pub fork : V R) : Bool := pub.version + 1 != fork.version

/-- `CatchUp` and `Swap` are built on `pub.new_build_version()`, so keep its data version; `Same` is at it. -/
theorem commit_keeps_version {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (hsame : indexerFit work prepared pub limit read = .same → prepared.version = pub.version) :
    (commitIndexerBatch work pub prepared valid f limit read).1.version = pub.version := by
  unfold commitIndexerBatch
  generalize hf : indexerFit work prepared pub limit read = fit
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
  | catchUp how =>
    simp only [commitVersion]
    by_cases hv : valid (V.caughtUp how prepared pub f) = true
    · simp only [hv, ↓reduceIte]
      cases how <;>
      simp only [V.setBatchesPublished, V.caughtUp, V.caughtUpOnto, V.insertedOnto, V.newBuildVersion, V.newVersion] <;>
      cases prepared.entry <;> cases pub.entry <;> simp <;> split <;> simp
    · simp [hv]

/-- `Same` implies the batch's version is at the published data version (what `commit_version`'s
`debug_assert!` checks). -/
theorem fit_same_version {R : Type} (ours pub : V R) (h : indexerFit work ours pub limit read = .same) : ours.version = pub.version := by
  unfold indexerFit at h
  split at h
  · cases h
  · split at h
    · cases h
    · by_cases hv : pub.version = ours.version
      · exact hv.symm
      · rw [ite_eq_left hv] at h
        split at h <;> cases h

theorem commit_not_stale {R : Type} (pub prepared fork : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn) :
    isStale (commitIndexerBatch work pub prepared valid f limit read).1 fork = isStale pub fork := by
  have := commit_keeps_version work limit read pub prepared valid f
    (fun h => fit_same_version work limit read prepared pub h)
  simp [isStale, this]

/-- Each publish counts one batch. -/
theorem commit_batches {R : Type} (pub prepared : V R) (valid : V R → Bool) (f : CatchUp → SE → SE → IndexColumn)
    (c : IndexerCommit) (hf : indexerFit work prepared pub limit read = c) (hc : ∀ w, c ≠ .discarded w)
    (hv : valid (commitVersion pub prepared f c) = true) :
    (commitIndexerBatch work pub prepared valid f limit read).1.batches = pub.batches + 1 := by
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
  | catchUp how =>
    simp only [commitVersion] at hv
    simp only [commitIndexerBatch, hf, commitVersion]
    simp only [hv, ↓reduceIte]
    rfl

end NI
