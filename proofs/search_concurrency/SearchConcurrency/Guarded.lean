/-
# Locksets: which locks every access to shared state holds (#3483)

`Locks` checks lock *order* (deadlock freedom). It never asks what a thread touches
while it holds, or does not hold, a lock, and `Locks.dropIndexBgProg` models
`RediSearch_DropIndex` as touching nothing. So a data race behind an axiomatised FFI
call was invisible to it. This module adds the missing check, Eraser-style: for each
shared resource, the locks held at *every* access must have a common member. It needs
no claim about which lock "should" guard what; the per-access locksets come from the
RediSearch LLAPI audit (2026-10-26: C side at `deps/RediSearch@a272834a6`, compared with
`ea1a6f40c`; Rust call sites at main 322e57a98).

| here | there |
| --- | --- |
| `gil`                    | the Redis GIL: main-thread callbacks, `QuerySession` writer mode, `RM_ThreadSafeContextLock` |
| `rsGlobal`, `rsSpec`     | RediSearch's global `RWLOCK` (`src/rwlock.c:59-102`) and per-spec `rwlock` (`src/spec.c:2263-2310`), write mode |
| `indexer`, `graphW`      | `Indexer::write_lock`; the per-graph `RwLock` in write mode |
| `naRef`                  | refcount of RediSearch's global `NA_rstr` (`src/info/index_error.c:29`), a plain `int` |
| `idxMem`                 | an index's inverted-index / numeric-tree memory, which ForkGC's child scans unlocked (`src/fork_gc.c:443-456`) |

**Accesses to `naRef` at ea1a6f40c** (all reach `IndexError_Init` `:42` or `IndexError_Clear` `:91`):
* `RediSearch_CreateIndex` (`api:47-78`, takes no RediSearch lock) from `Index::create_rs_index`
  (`graph/src/index/mod.rs:1101`), GIL held on every path: `{gil, indexer}`.
* `RediSearch_CreateField` (`api:167-242`, RWLOCK W): `{gil, indexer, rsGlobal}`.
* `RediSearch_DropIndex` (`api:80-92`, RWLOCK W) from `SpecHandle::drop` on `graph_free`'s
  spawned thread (`src/graph_core.rs:1794`): `{rsGlobal}`.
* `RediSearch_IndexRelease` dropping the last ref (`api:99-101`, no lock) from
  `OwnedIndex::drop` on a reader after a concurrent drop: `{}`.
At a272834a6 (FalkorDB/RediSearch#29) errors borrow `NA_rstr`: none of these touch it.

**Accesses to `idxMem`** (both commits):
* ForkGC: `RM_Fork` under the GIL only (`fork_gc.c:1050-1065`): `{gil}`.
* Writer commit `Indexer::commit` → `IndexAddDocument` (`api:648-688`, RWLOCK W + spec W):
  `{gil, graphW, indexer, rsGlobal, rsSpec}`.
* Background population `populate_index_batch` (`graph/src/graph/graph.rs:543-563`,
  "Deliberately no global lock") → `IndexAddDocument`: `{indexer, rsGlobal, rsSpec}`.
-/

namespace SC.Guarded

def gil : Nat := 0
def rsGlobal : Nat := 1
def rsSpec : Nat := 2
def indexer : Nat := 3
def graphW : Nat := 4

/-- One access to a shared resource with the locks its thread holds. -/
structure Access where
  site : String
  held : List Nat
deriving Repr, Inhabited

/-- Locks held at every access (Eraser's candidate set). -/
def lockset : List Access → List Nat
  | [] => [gil, rsGlobal, rsSpec, indexer, graphW]
  | a :: as => (lockset as).filter (a.held.contains ·)

/-- Race-free: some lock is common to every access. -/
def consistent (as : List Access) : Bool := !(lockset as).isEmpty

/-! ## #3483: `NA_rstr`'s refcount -/

def naRef_ea1a6f4 : List Access :=
  [⟨"CreateIndex ← Index::create_rs_index", [gil, indexer]⟩,
   ⟨"CreateField ← Index::create_rs_index/register_fields", [gil, indexer, rsGlobal]⟩,
   ⟨"DropIndex ← SpecHandle::drop on graph_free's thread", [rsGlobal]⟩,
   ⟨"IndexRelease (last ref) ← OwnedIndex::drop on a reader", []⟩]

/-- With RediSearch#29 no LLAPI path touches `NA_rstr`'s refcount. -/
def naRef_rs29 : List Access := []

/-- **#3483.** No lock is common to the accesses: the main thread's create holds only the
GIL among them, the drop thread only RediSearch's lock. -/
theorem bug3483_naRef_race : consistent naRef_ea1a6f4 = false := by decide
/-- Even the GIL-free drop alone would race with create: the two share nothing. -/
theorem bug3483_create_vs_drop :
    consistent [naRef_ea1a6f4[0]!, naRef_ea1a6f4[2]!] = false := by decide
theorem rs29_naRef_free : consistent naRef_rs29 = true := by decide

/-! ## H1 (open): ForkGC forks while background population writes the index

The ForkGC takes only the GIL around `RM_Fork`; background population writes under
RediSearch's locks but not the GIL. The child can snapshot a half-written block or tree
and send the parent a delta computed from it. -/

def idxMem : List Access :=
  [⟨"ForkGC RM_Fork (fork_gc.c:1050-1065)", [gil]⟩,
   ⟨"writer commit → IndexAddDocument", [gil, graphW, indexer, rsGlobal, rsSpec]⟩,
   ⟨"populate_index_batch → IndexAddDocument", [indexer, rsGlobal, rsSpec]⟩]

theorem h1_fork_vs_populate_race : consistent idxMem = false := by decide

/-- Writers alone are consistent (RediSearch's locks); so are fork and writer commit
(the GIL). It is the GIL-free population that breaks it. -/
theorem h1_without_populate : consistent (idxMem.take 2) = true := by decide

/-- Either fix restores a common lock: population under the GIL, or the GC holding the
spec lock across the fork. -/
theorem h1_fix_populate_gil :
    consistent (idxMem.take 2 ++ [⟨"populate under GIL", [gil, indexer, rsGlobal, rsSpec]⟩])
      = true := by decide
theorem h1_fix_gc_spec_lock :
    consistent (⟨"ForkGC fork under spec R", [gil, rsSpec]⟩ :: idxMem.drop 1) = true := by
  decide

/-! ## The check is the lockset property -/

theorem mem_lockset {as : List Access} {l : Nat} (h : l ∈ lockset as) (a : Access)
    (ha : a ∈ as) : l ∈ a.held := by
  induction as with
  | nil => simp at ha
  | cons b bs ih =>
    simp only [lockset, List.mem_filter] at h
    rcases List.mem_cons.mp ha with rfl | hb
    · simpa using h.2
    · exact ih h.1 hb

/-- If the accesses are consistent, one lock is held at every access. -/
theorem consistent_common_lock (as : List Access) (h : consistent as = true) :
    ∃ l, ∀ a ∈ as, l ∈ a.held := by
  simp only [consistent, Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
  obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ h
  exact ⟨l, fun a ha => mem_lockset hl a ha⟩

end SC.Guarded
