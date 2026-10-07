# Online (background) index build — design

**Status:** proposed. **Revision 2**, after four independent adversarial reviews
(correctness proof / scan-vs-hook divergence / assumption fact-check / concurrency).
Supersedes the prototype on `rs/native-numeric-index`, which **must not ship** (§11).
**Scope:** native FalkorDB index (`index-falkordb`), numeric column kind, node + edge.
**Base:** `FalkorDB/FalkorDB@main` (post-#726 two-phase write path).

> **What changed in r2.** The mechanism is unchanged and survived attack. The *specification*
> did not: K-SOUND was false as written (§5), DELTA's defining property was never stated (§5),
> the locking section was wrong in both its premise and its conclusion (§7), and the TOMB-vs-`dirty`
> change was mis-sold as a pure win (§11). Two hard-failure bugs were found in the prototype and
> two live correctness bugs in `main` (§13).

---

## 1. Problem

`CREATE INDEX ON :L(a)` on a graph that already holds data must scan every matching entity,
read its attribute, and build the index — O(|L|) reads + sort + bulk load. Hundreds of ms per
million rows for numeric; far worse for vector, where embedding is CPU-bound.

1. `CREATE INDEX` returns promptly; it does not block writes for the scan's duration.
2. Concurrent writes are neither lost nor left stale.
3. No query may observe a partially-built index.
4. No deadlock, and no use of L1 (§7).

---

## 2. Model

Committed versions `V0, V1, …` are immutable; a write forks the current version, mutates the
fork, publishes it. **The index is folded into the version** (`falkordb_index` is a `Graph`
field, CoW-cloned by `new_version()`), so publishing an index change means publishing a version.

A **column** `c = (entity-type, label/type L, attr a)`. For version `V`, entity `x`:

- `match_V(x)` — live, carries `L`, has `a`, and the value is indexable.
- `tuples_V(x)` — the **encoded** tuples `x` contributes: `{(encode(value), x)}`. A set, so
  array kinds generalize — but see I-W8 and §13(B) for what that requires.
- `T[V]` — tuples physically stored in column `c` at `V`.

**EXACT(V)** ⟺ `T[V] == ⋃ₓ tuples_V(x)`.

> **Caveat on EXACT.** It is defined over *encoded* tuples, so it is satisfied by an index that
> answers queries wrongly whenever the encoder is many-to-one — which it is today (§13 B).
> EXACT is the right target for *this* design; it is not a complete correctness statement for
> the index as a whole.

---

## 3. Three artifacts

| Artifact | Lives in | Size | Written by |
|---|---|---|---|
| **BASE** | the build job's private memory — unreachable from any version | O(N) | the scan |
| **DELTA** | the column's own tree, inside the version | O(writes during build) | client writes |
| **TOMB** | a second tree in `Building` state, inside the version | O(writes during build) | client writes |

BASE is deliberately stale; TOMB + DELTA are the catch-up log that makes installing a stale
artifact safe.

**Writers write directly into the column** — DELTA *is* the column's tree during a build; there
is no parallel write path. The sole exception is a *remove* whose target row lives in BASE,
which no version can name yet; that remove is deferred into TOMB.

```
ColumnState::Building { tomb, epoch: u64 }   // tomb: same CoW tuple tree as the column
ColumnState::Ready
```

**TOMB stores encoded `(key, doc)` tuples**, not raw `Value`s — its dedup must use the same
equivalence relation as the column, or §13(B)'s collisions turn into wrong removals.

---

## 4. Flows

### 4.1 CREATE INDEX (client path)
One ordinary write commit: column ← `Building { tomb: empty, epoch: fresh }`; record `N` = the
version produced; return. `create_index_sync` (RDB load, replica effect) stays synchronous → `Ready`.

### 4.2 Scan — no locks
```
snapshot := Arc to version N        (lock-free clone; immutable)
BASE     := scan(snapshot)          (minutes are fine; parallelisable; chunk for bounded memory)
```
Writes committing as N+1, N+2, … are invisible to the scan by construction.

### 4.3 Client write while Building
```
fork current version                 (CoW: index roots bump, O(1))
stage rem/add from live graph state  (existing hooks)
  add    → column tree (DELTA)
  remove → column tree (usually a no-op) AND append to TOMB
commit
```

### 4.4 Install — one commit
```
try-acquire writer slot             (NON-blocking; on failure back off and retry — §7)
  fork the current committed version V_M
  read DELTA and TOMB *from this fork*   (never from a value captured earlier)
  new := BASE
  new.remove_batch(TOMB)            (removes strictly first)
  new.merge(DELTA)                  (then adds)
  column := Ready { tree: new }     (TOMB dropped)
  acquire GIL
  commit V_M+1                      (Arc swap under the GIL — #452 fork-safety)
release GIL, release slot
```

### 4.5 Read
`Building` → must not use the column → fall back. `Ready` → exact for that snapshot → use it.
A query holding a `Building` snapshot keeps falling back for its whole life; no mid-query switch.

---

## 5. Invariants

**Reads**
- **I-R1** A query consults `c` only if its snapshot says `Ready`. Obligation at every read site.
- **I-R2** Index and graph resolve from the same snapshot. (Free: folded roots.)
- **I-R3** EXACT holds wherever `c` is `Ready`.
- **I-R4** A `Building` column has a correct fallback at plan/run time.

**Writes**
- **I-W1** One writer at a time; **fork and commit inside the same exclusive window**.
- **I-W3** Every mutation changing whether/how an entity is indexed goes through the hooks.
- **I-W4** While Building: new tuples → DELTA, destroyed tuples → TOMB.
- **I-W4′** `DELTA == ⋃ tuples_V_M(x)` over touched `x` — DELTA holds each touched entity's
  **final** state, not an append-only add-log. *Requires removes to hit the column tree, not
  only TOMB.* Without this, `v0→v1→v2` installs two rows.
- **I-W5** Writes never wait for the scan; only for the single install commit.
- **I-W6** A **pure-add hook** may be used only where `tuples_before(x) ⊆ tuples_after(x)`.
  Three hooks stage no removes — `import_node_attrs`, `import_relationship_attrs`,
  `set_nodes_labels_bulk` — and every new hook must be audited against this.
- **I-W7** Each hook recomputes the entity's tuples from **live state at its own point in the
  commit order**, never from a value captured earlier. (`pending::commit` fixes an intra-commit
  order: labels → label-removes → imports → sets → deletes; correctness of combinations such as
  `REMOVE n:L SET n.a=9` depends on it.)
- **I-W8** Hooks stage **encoded tuples**, never raw values. Any diffing (arrays) happens
  post-encode — see §13(B).

**Build**
- **I-B1** `BASE == ⋃ₓ tuples_V_N(x)`.
- **I-B2** BASE is unreachable from any committed version until install; the scan holds no locks.
- **I-B3** Install is one commit; reads DELTA/TOMB from the version it forks; **removes strictly
  before adds**; publishes `Ready` atomically with the merged tree.
- **I-B4** Epoch: install/finish are no-ops unless the column's epoch matches. In-flight
  bookkeeping is epoch-keyed.
- **I-B5** **Liveness:** every `Building` column is eventually dispatched, by a mechanism that
  does not depend on a subsequent client write (§8).
- **I-B6** The scan **materialises the snapshot under the Rust mutex** (`Matrix::wait`) before
  iterating, so that no later `GxB_rowIterator_attach` — the scan's or a concurrent query's —
  triggers materialisation outside that mutex. See §10.5.

**Resources**
- **I-X1** Every per-build artifact carried in the version forks in **O(1)**. (`RoaringTreemap`
  does not — see §11.)
- **I-X2** The writer slot is released on **every** exit path including unwind — RAII, not a
  convention. See §13(C).

**Locking** — §7.

**Derived**
- **K-SOUND′** For every `t ∈ tuples_V_N(x) ∖ tuples_V_M(x)`, `t ∈ TOMB`.
  *Holds because* every tuple-**destroying** hook recomputes the destroyed tuple from live state
  when it runs, and the first such recomputation for `x` yields its snapshot-era tuple.
  *(The r1 formulation — "the first remove staged enters TOMB" — was **false**: pure-add hooks
  mean no remove need be staged at all. Counterexample: `SET n:L` on a node already carrying `:L`.)*
- **K-SAFE** Any tuple in TOMB still valid at `V_M` is also in DELTA — by I-W4′. Hence removes
  must precede adds.

---

## 6. Correctness argument

**Claim.** I-B1 ∧ I-W3 ∧ I-W4′ ∧ I-W6 ∧ I-B3 ⇒ EXACT(V_M+1).

| Case | BASE | TOMB | DELTA | Result |
|---|---|---|---|---|
| untouched, matched at N | row | — | — | correct ✓ |
| untouched, unmatched at N | — | — | — | absent ✓ |
| updated v0→v1→v2 | (v0,x) | (v0,x),(v1,x) | (v2,x) | (v2,x) ✓ |
| updated v0→v1→v0 | (v0,x) | (v0,x),(v1,x) | (v0,x) | (v0,x) ✓ *order-critical* |
| deleted | (v0,x) | (v0,x) | — | absent ✓ |
| created after N | — | — | (v1,x) | (v1,x) ✓ |
| created and deleted after N | — | no-op | — | absent ✓ |
| id reused, same value | (v,x) | (v,x) | (v,x) | (v,x) ✓ *order-critical* |
| id reused repeatedly (n cycles) | (v0,x) | all intermediates | final | final ✓ — TOMB is subtracted from BASE **only**, never from DELTA |
| label added during build | — | — | (v,x) | (v,x) ✓ |
| label removed during build | (v0,x) | (v0,x) | — | absent ✓ |
| **no-op label re-add** (`SET n:L`, already `:L`) | (v,x) | **—** (pure-add hook) | (v,x) | (v,x) ✓ — correct by *set collapse*, not by K-SOUND |
| numeric → non-numeric | (v0,x) | (v0,x) | — (encoder drops) | absent ✓ |
| **attribute deleted** (`SET a=NULL` / `REMOVE`) | (v0,x) | (v0,x) | — (`Null` dropped by encoder) | absent ✓ — relies on every kind rejecting `Null` |
| **multi-label node** | per-label row | per-label | per-label | ✓ — `stage_index_node` fans one attr write to **every** label column; the whole argument is per-column |
| **cascade edge delete** | (v0,e) | (v0,e) | — | ✓ — `delete_implicit_edges` is a *second, independent* remove path, correct only because it stages before `relationship_attrs.remove_all` |
| **`GRAPH.BULK` row** | maybe | — | — | ✗ **BROKEN** — §13(A) |

Untouched entities follow from the contrapositive of I-W3; touched entities from K-SOUND′ and
induction on I-W4′.

**Scan-vs-write races do not exist.** The scan reads a frozen snapshot, so whether a write
touches an entity the scan has passed or not yet reached is irrelevant. Carol deleted before the
scan reaches her: the scan still emits her row; TOMB removes it at install.

**Remove-before-add is necessary and sufficient.** Install starts *from* BASE, so all
snapshot-era tuples are present; the only ordering is TOMB-vs-DELTA. Applying DELTA first would
delete exactly `TOMB ∩ DELTA` — every tuple destroyed and later recreated. By I-W4′ no DELTA
tuple is invalid at `V_M`, so TOMB never needs to fire after DELTA. No case requires the opposite.

---

## 7. Locking — rewritten in r2

Both the premise and the conclusion of r1 were wrong.

**Premise (wrong in r1).** r1 said "L1 is the de-facto writer lock; the CAS slot is a backstop."
False: the main client writer CASes the slot while holding L1-**read** (`graph_core.rs:592`,
`Mode::Reader`; the comment there says so). **The writer slot is already the real exclusion
primitive.** L1 exists only to serialize the non-MVCC RediSearch index.

**Conclusion (wrong in r1, and again in the first r2 draft).** r1 required the slot to become
*blocking*. That creates a cycle: the client path holds the slot and *then* acquires the GIL during
escalation, so a blocking slot deadlocks the moment the main thread parks on it — and **dropping L1
does not help**, because the main thread's GIL hold is implicit and non-releasable. **The
blocking-slot requirement is withdrawn.**

The first r2 draft then said the installer takes "writer-slot → GIL, never L1". **That is also
wrong**, for a blunt reason found during implementation: `MvccGraph::commit` takes **`&mut self`**
(`read`/`write`/`rollback` take `&self`, only `commit` does not). Publishing a version therefore
requires exclusive access to the `ThreadedGraph`, i.e. **L1-write is unavoidable**. "The native
index is MVCC so it needs no L1" is true of the *index*, false of the *commit*.

**Actual orders in the code:**

```
inline main-thread cmd   GIL (implicit) → L1 → try-slot (None ⇒ retryable error)
pooled client write      L1-read → try-slot (None ⇒ error) → [escalate: drop L1-read, GIL, L1-write]
installer                GIL → L1-write → try-slot → install → commit
```

**Why this is cycle-free:**

- **I-L1 (load-bearing)** The **main thread must never block on the writer slot** — try-acquire
  plus a retryable error. It cannot release its implicit GIL, so any main-thread park is a global
  stall.
- **I-L2** The installer acquires **GIL → L1-write → try-slot**, the same GIL-before-L1 order every
  other writer uses. The slot is *try*-acquired: a client can hold the slot while waiting for the
  GIL during escalation, so blocking on it would deadlock. On contention the installer releases
  everything and is re-dispatched by the sweep (§8) — a background job can retry; a client cannot.
- **I-L3** Never block (channel send, slot park, GIL acquire) while holding L1.
- **I-L4** Slot release survives an unwind (I-X2): `MvccGraph::write` latches a flag cleared only by
  `commit`/`rollback`, so a panic between them wedges *every* subsequent write in the process. The
  installer catches around `install` and rolls back.

The escalating client holds the slot but no L1 while it waits for the GIL, so the installer's
`GIL → L1-write` never waits on a thread that is waiting for the GIL.

The GIL covers only the Arc swap (#452 BGSAVE fork-exclusion). **It is not free:
`MvccGraph::commit` calls `trim_attr_stores()`, an O(attribute-store) walk. That is a pre-existing
cost on every commit; it is an argument for one install commit rather than N/1024 of them.**

---

## 8. Liveness

Dispatch on "a client write committed" is insufficient — it is wired into **one of five** write
paths, and it leaks its bookkeeping. A single **periodic sweep** over the graph registry running
`dispatch(collect_pending_builds(g))` subsumes all of it: MULTI/EXEC and replica `CREATE INDEX`,
a dropped `spawn`, a panicked job, a fork-contention bail, and "CREATE INDEX was the last write".
Dispatch is idempotent under the in-flight set, so the sweep is cheap and needs no per-path wiring.

**The in-flight marker must be inserted by the job, not the dispatcher** (or handed to the job so
it is released if `spawn` drops it) — otherwise a dropped job leaks the key and suppresses every
future re-spawn of that column.

---

## 9. Cost

| Phase | Cost | Locks |
|---|---|---|
| scan + BASE build | O(N) reads + sort + bulk load | **none** |
| client write during build | one batched insert into DELTA + one into TOMB | ordinary write locks |
| install | O(1) adopt BASE + O(\|TOMB\|·log N) + O(\|DELTA\|·log N) | slot + GIL, briefly |

The O(N) work never runs under a lock, and nothing is quadratic — provided I-X1 holds.

---

## 10. Verified assumptions

| # | Assumption | Verdict |
|---|---|---|
| 1 | Read gating enforced at every read site | **TRUE** — two sites, both funnel through `query_numeric`, which cannot return a `Building` column. *But* `numeric()`/`numeric_mut()` are `pub` state-agnostic hatches, and the fallback goes to **RediSearch, not a scan** — a hole at P7 |
| 2 | Aborted writes discard index mutations with the fork | **TRUE** — `rollback()` never touches the graph; the fork drops. *This is why TOMB must be versioned, not a shared log* |
| 3 | `new_version()` is O(1) per column | **PARTIAL** — trees are Arc-rooted ✓, but `dirty: RoaringTreemap` deep-clones ⇒ O(W²) per build. Fixed by TOMB (§11) |
| 4 | L1 guarantees the writer wins the CAS | **FALSE** — see §7 |
| 5 | Off-thread snapshot scan is safe | **SETTLED — safe, with a required mitigation.** `Iter::new` does not call `Matrix::wait`; it calls `GxB_rowIterator_attach`, and SuiteSparse finishes pending work on attach (corroborated by the comment at `graph.rs:1478`). So the scan **does** mutate the snapshot's matrix, and that materialization happens inside GraphBLAS, *not* under the Rust `self.lock`. **This is identical to what concurrent read queries already do** — two `MATCH`es on the same committed snapshot both attach — so the scan adds no new hazard class. **Mitigation (adopted, I-B6):** the scan calls `Matrix::wait()` on the snapshot first, which materializes once *under* the Rust mutex (correct double-checked lock: Acquire fast path → mutex → re-check → `GrB_MATERIALIZE` → Release). Afterwards `has_pending` is false and every later attach, scan's or query's, is a pure read — the scan warms the snapshot instead of racing on it. Residual: the pre-existing attach-materialises-outside-the-mutex race for *first* readers deserves its own investigation, but it does not block this design |

---

## 11. Difference from the prototype — with the honest trade

| Prototype | This design | Note |
|---|---|---|
| Chunked install: N/1024 commits | One install commit; BASE adopted wholesale | chunking multiplies the O(N) `trim_attr_stores` per commit |
| `dirty` RoaringTreemap of ids | TOMB tuple tree | **not a pure win — see below** |
| `dirty` deep-clones per fork (O(W²)) | TOMB forks O(1) | strict improvement |
| L1 → slot → GIL (**inverted** ⇒ deadlock) | slot → GIL, never L1 | strict improvement |
| Dispatch on client write only | Periodic sweep (§8) | strict improvement |
| Slot leaked on panic | RAII guard (I-X2) | strict improvement |

> **The TOMB trade-off, stated plainly.** `dirty` is *touch*-keyed: any touch of an id suppresses
> **all** of that id's BASE rows, so it is immune to a hook computing the wrong value. TOMB is
> *remove*-keyed: a stale row survives unless the hook stages the exact matching encoded tuple.
> This is a **narrowing of the safety property**, bought in exchange for O(1) BASE adoption.
> It is acceptable because the steady-state (`Ready`) path *already* depends on exact value
> precision — a hook staging a wrong old value corrupts the index today, build or no build — so
> TOMB extends an existing dependency rather than creating a new class. But it makes **I-W6 an
> audit obligation**, not a note.

---

## 12. Edge cases

| # | Case | Resolution |
|---|---|---|
| C1 | DROP during build | Epoch ⇒ install no-ops; needs a cancel signal for liveness |
| C2 | DROP + re-CREATE during build | New epoch ⇒ stale install inert; in-flight key must include the epoch |
| C3 | Graph deleted during build | Job holds an Arc (no UAF); install no-ops on teardown |
| C4 | Crash / restart mid-build | Index not serialised; RDB load rebuilds synchronously ⇒ `Ready` |
| C5 | BGSAVE fork mid-build | BASE unpublished ⇒ child sees no partial index; commit under GIL |
| C6 | Replica | Effect → `create_index_sync`. **Open:** a long sync build stalls the replica link |
| C7 | Build panics / OOMs | Sweep (§8) re-dispatches; RAII (I-X2) prevents a wedged slot |
| C8 | Many columns building at once | Independent epochs; needs a concurrency cap |
| C9 | **`GRAPH.BULK` during build** | **Unsound — §13(A)** |
| C10 | Long build, write-heavy | \|DELTA\|,\|TOMB\| = O(writes) ⇒ install cost grows; needs a cap for vector scale |
| C11 | Aborted write | Fork discarded ⇒ DELTA/TOMB entries vanish — the reason TOMB is versioned |
| C12 | Id reuse after an unhooked delete | `deleted_*` is a **free list**, cleared on reuse, so the backstop is defeated; TOMB is the *only* real defence, and it depends on hook completeness — which §13(A) breaks |

---

## 13. Defects found in review that are **not** this design's to fix

**(A) `GRAPH.BULK` bypasses the index hooks — live on `main`.**
`import_node_attrs_resolved` / `import_relationship_attrs_resolved` have no hook at all;
`bulk_insert.rs:294` / `:367` call them. The comment's stated mitigation is fictional —
`populate_indexes_sync` is never called from `bulk_insert.rs`. Bulk-loading into an indexed graph
leaves rows permanently unindexed; `bulk_insert.rs:290` runs the one hooked call *before* the
attributes exist, so it no-ops. **During a build it is worse than a hole:** those ids are never
recorded, so install writes the *snapshot* value for a row the bulk has overwritten.
*This design cannot be sound until (A) is fixed or `GRAPH.BULK` is excluded by precondition.*

**(B) The encoder is many-to-one across types — live on `main`.**
```rust
Value::Int(i)  => *i as f64,        // lossy at >= 2^53
Value::Bool(b) => f64::from(*b),    // true -> 1.0
Value::Datetime(t) | Date(t) | Time(t) | Duration(t) => *t as f64,
```
No type tag, so `WHERE n.a = 1` matches a node with `a = true`, `a = duration(1)`, or
`a = datetime(epoch 1)`; and `Int >= 2^53` is coerced (`9007199254740993` → `…992`). EXACT can
hold while answers are wrong. Fix: widen the key with the `Value` discriminant, or re-check the
raw attribute on the read path. Also the reason arrays must diff post-encode (I-W8).

**(C) The writer slot has no RAII release — latent on `main`, reachable via the prototype.**
Released only by `commit`/`rollback`; unwinding never releases it. Every other slot-taker pairs
with a rollback *by convention*. A panic in a slot-holder wedges **all** writes for the life of
the process.

**(D) `delete_relationships` drops removes for unresolvable edges**, leaving orphan tuples that
resurface on edge-id reuse.

**(E) Latent coupling:** the scan reads `labels_matices`; the hooks read `node_labels_matrix`.
They agree today only because three sites update both.
