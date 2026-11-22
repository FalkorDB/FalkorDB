/-!
# `MvccGraph`: what a load publishes and what a commit does with a fork

(graph/src/graph/mvcc_graph.rs, origin/main `256ab4d3f`)

`MvccGraph` is one `ArcSwap<Graph>`: the published version. An RDB load builds the graph, wraps
it with `from_graph` and registers it; every query then `read`s it, a writer `fork`s it
(`Graph::new_version`, data unchanged, version + 1) and `commit`s the fork back. This models
exactly those entry points over an abstract graph value `D`; the index-falkordb halves
(`rebase_onto_indexer_batches`, the build queue, `commit_indexer_batch`) are proofs/native_index.
Concurrency (several writers racing the swap) is proofs/search_concurrency's; here one
`compare_and_swap` is a test of pointer identity (`Gv.id`), which is what `Arc::ptr_eq` is.

| here | there |
| --- | --- |
| `Gv` | an `Arc<Graph>`: pointer identity, `version`, `schema_version`, the four schema counts, `validate()`'s verdict, the logical data |
| `M.fromGraph` / `M.new` | `from_graph`  / `new`  |
| `M.read` | `read`  |
| `Gv.newVersion`, `M.fork` | `Graph::new_version` (graph_queries `newVersion_spec`), `fork`  |
| `M.isStale` | `is_stale`  |
| `prepare` | `prepare`  (`fold_oversized_deltas` and `finish_version` change no logical data) |
| `M.publish` | `publish`  |
| `M.commit` | `commit`  (feature-gated `to_build`/`online_build::enqueue` omitted) |
| `M.dropEvents` | `Drop`  |

`debug_assert!(writer || published.version == new_graph.version)` of `prepare` is a debug-build
check on batch commits (`writer = false`, only `commit_indexer_batch` passes it); the release
build has no such arm and this model follows the release build.

Theorems: `fromGraph_read` (what a load publishes is what is read), `fork_not_stale`,
`commit_ok` (a success publishes exactly the validated fork as version + 1, data unchanged),
`commit_refused` (a refusal changes nothing), `commit_stale_second` (of two forks of one version
only the first publishes), `prepare_stale`/`prepare_invalid`, `commit_schema_bump`
(a changed schema moves `schema_version`), `drop_cancels_published`.
-/
namespace GraphPersist.Mvcc

/-- One published or forked `Graph`, as `MvccGraph` sees it. -/
structure Gv (D : Type) where
  id : Nat
  version : Nat
  schemaVersion : Nat
  /-- `(labels, types, node attribute names, relationship attribute names)` lengths. -/
  sig : Nat × Nat × Nat × Nat
  /-- `Graph::validate()` is `Ok`. -/
  valid : Bool
  data : D

/-- `MvccGraph`: the `ArcSwap` content. -/
structure M (D : Type) where
  cur : Gv D

inductive CommitError | stale | invalid
  deriving DecidableEq, Repr

variable {D : Type}

/-- `from_graph` (:118) / `new` (:106, which is `from_graph(Graph::new(..))`). -/
def M.fromGraph (g : Gv D) : M D := ⟨g⟩

/-- `read` (:129): `load_full()`. -/
def M.read (m : M D) : Gv D := m.cur

/-- `Graph::new_version`: the same data, one version later, a new allocation. -/
def Gv.newVersion (g : Gv D) (fresh : Nat) : Gv D := { g with id := fresh, version := g.version + 1 }

/-- `fork` (:141): `load().new_version()`; takes nothing, changes nothing. -/
def M.fork (m : M D) (fresh : Nat) : Gv D := m.cur.newVersion fresh

/-- `is_stale` (:155): `load().version + 1 != fork.version`. -/
def M.isStale (m : M D) (fork : Gv D) : Bool := m.cur.version + 1 != fork.version

/-- `prepare` (:293) for a writer's version (`writer = true`). `published` is the version the
caller loaded. -/
def prepare (published new : Gv D) : Except CommitError (Gv D) :=
  if published.version + 1 != new.version then .error .stale
  else if !new.valid then .error .invalid
  else
    .ok (if published.sig != new.sig && new.schemaVersion == published.schemaVersion
         then { new with schemaVersion := new.schemaVersion + 1 } else new)

/-- `publish` (:354): `compare_and_swap(published, version)`; the swap happens only if the slot
still holds `published` (pointer identity), else the later commit is refused. -/
def M.publish (m : M D) (published version : Gv D) : M D × Except CommitError (Gv D) :=
  if m.cur.id == published.id then (⟨version⟩, .ok version) else (m, .error .stale)

/-- `commit` (:205): load, `prepare`, `publish`. -/
def M.commit (m : M D) (fork : Gv D) : M D × Except CommitError (Gv D) :=
  let published := m.read
  match prepare published fork with
  | .error e => (m, .error e)
  | .ok v => m.publish published v

/-- `Drop` (:370): `self.graph.load().cancel_indexing()`, on the version published at the time. -/
def M.dropEvents (m : M D) : List Nat := [m.cur.id]

/-- What a load publishes is what every later read returns. -/
theorem fromGraph_read (g : Gv D) : (M.fromGraph g).read = g := rfl

/-- A fork of the published version is not stale (nothing was published since). -/
theorem fork_not_stale (m : M D) (fresh : Nat) : m.isStale (m.fork fresh) = false := by
  simp [M.isStale, M.fork, Gv.newVersion]

/-- A fork carries the published data unchanged, one version later. -/
theorem fork_data (m : M D) (fresh : Nat) :
    (m.fork fresh).data = m.cur.data ∧ (m.fork fresh).version = m.cur.version + 1 := ⟨rfl, rfl⟩

theorem prepare_stale (p n : Gv D) (h : (p.version + 1 != n.version) = true) :
    prepare p n = .error .stale := by simp [prepare, h]

theorem prepare_invalid (p n : Gv D) (h1 : (p.version + 1 != n.version) = false) (h2 : n.valid = false) :
    prepare p n = .error .invalid := by simp [prepare, h1, h2]

/-- A refusal changes nothing: the published graph is as it was (and no version is handed back). -/
theorem commit_refused (m : M D) (fork : Gv D) (e : CommitError)
    (h : (m.commit fork).2 = .error e) : (m.commit fork).1 = m := by
  unfold M.commit at *
  cases hp : prepare m.read fork with
  | error e' => simp [hp]
  | ok v =>
    simp only [hp] at h ⊢
    unfold M.publish at h ⊢
    split at h <;> simp_all

/-- **A successful commit publishes exactly the validated fork**: the slot now holds the version
returned, it is the fork's data at version + 1, and the fork was valid; the schema version moves
iff the schema counts changed (see `commit_schema_bump`). -/
theorem commit_ok (m : M D) (fork v : Gv D) (m' : M D)
    (h : m.commit fork = (m', .ok v)) :
    m'.cur = v ∧ v.data = fork.data ∧ v.version = m.cur.version + 1 ∧ fork.valid = true := by
  unfold M.commit M.read at h
  cases hp : prepare m.cur fork with
  | error e => simp [hp] at h
  | ok w =>
    simp only [hp] at h
    unfold M.publish at h
    split at h
    · simp only [Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [prepare] at hp
      split at hp
      · cases hp
      · split at hp
        · cases hp
        · rename_i h1 h2
          simp at h1 h2
          have hv : w.version = fork.version ∧ w.data = fork.data := by
            split at hp <;> (cases hp; simp)
          refine ⟨rfl, hv.2, by omega, by simpa using h2⟩
    · simp at h

/-- A changed schema moves `schema_version` (so clients re-read the names) unless the fork had
already moved it. -/
theorem commit_schema_bump (m : M D) (fork v : Gv D) (m' : M D)
    (h : m.commit fork = (m', .ok v)) (hs : m.cur.sig ≠ fork.sig)
    (hv : fork.schemaVersion = m.cur.schemaVersion) : v.schemaVersion = m.cur.schemaVersion + 1 := by
  unfold M.commit M.read at h
  cases hp : prepare m.cur fork with
  | error e => simp [hp] at h
  | ok w =>
    simp only [hp] at h
    unfold M.publish at h
    split at h
    · simp only [Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨-, rfl⟩ := h
      simp only [prepare] at hp
      split at hp
      · cases hp
      · split at hp
        · cases hp
        · have hne : (m.cur.sig != fork.sig) = true := by simpa using hs
          simp [hne] at hp
          cases hp; simp [hv]
    · simp at h

/-- Of two forks of one published version only the first publishes: after it, the second is
`Stale` (its `version` is no longer `published.version + 1`), and the slot is not touched. -/
theorem commit_stale_second (m : M D) (f1 f2 v : Gv D) (m1 : M D)
    (h1 : m.commit f1 = (m1, .ok v)) (hv : f2.version = m.cur.version + 1) :
    (m1.commit f2) = (m1, .error .stale) := by
  have hc := commit_ok m f1 v m1 h1
  obtain ⟨hcur, -, hver, -⟩ := hc
  unfold M.commit M.read
  rw [hcur]
  have : (v.version + 1 != f2.version) = true := by simp; omega
  simp [prepare, this]

theorem drop_cancels_published (m : M D) : m.dropEvents = [m.cur.id] := rfl

end GraphPersist.Mvcc
