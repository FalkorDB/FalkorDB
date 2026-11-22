/-
# Locksets: which locks every access to shared state holds

`Locks` checks lock *order* (deadlock freedom), not what a thread touches while it holds, or
does not hold, a lock, so a data race behind an axiomatised FFI call is invisible to it. This
module adds the missing check, Eraser-style: for each shared resource, the locks held at *every*
access must have a common member (per-access locksets from the RediSearch LLAPI audit).

| here | there |
| --- | --- |
| `gil`                    | the Redis GIL: main-thread callbacks, `QuerySession` writer mode, `RM_ThreadSafeContextLock` |
| `rsGlobal`, `rsSpec`     | RediSearch's global `RWLOCK` (`src/rwlock.c`) and per-spec `rwlock` (`src/spec.c`), write mode; a read-mode hold (`h1_fix_gc_spec_lock`) is modelled as exclusive, sound because every other access to `idxMem` holds `rsSpec` in write mode |
| `indexer`, `graphW`      | `Indexer::write_lock`; the per-graph `RwLock` in write mode |
| `idxMem`                 | an index's inverted-index / numeric-tree memory, which ForkGC's child scans unlocked (`src/fork_gc.c`) |

**Accesses to `idxMem`** (both commits):
* ForkGC: `RM_Fork` under the GIL only (`fork_gc.c:1050-1065`): `{gil}`.
* Writer commit `Indexer::commit` → `IndexAddDocument` (`api:648-688`, RWLOCK W + spec W):
  `{gil, graphW, indexer, rsGlobal, rsSpec}`.
* Background population `populate_index_batch` (`graph/src/graph/graph.rs` `populate_index_batch`,
  the "Deliberately no global lock" comment ) → `IndexAddDocument`: `{indexer, rsGlobal, rsSpec}`.
-/

namespace SC.Guarded

def gil : Nat := 0
def rsGlobal : Nat := 1
def rsSpec : Nat := 2
def indexer : Nat := 3
def graphW : Nat := 4

/-- One access to a shared resource with the locks its thread holds. There is no read/write
mode: every held lock is treated as exclusive, which is exact when the accesses hold it in
write mode (all RediSearch accesses here) and sound for a single read-mode holder against
write-mode holders (`h1_fix_gc_spec_lock`). It would be unsound if two accesses held their only
common lock in read mode and one of them wrote. -/
structure Access where
  site : String
  held : List Nat
deriving Repr, Inhabited

/-- Locks held at every access (Eraser's candidate set). Derived from the accesses, not
from a fixed universe of lock ids, so a lock added later is never dropped. The empty
access list has no candidates (it is handled in `consistent`). -/
def lockset : List Access → List Nat
  | [] => []
  | [a] => a.held
  | a :: b :: as => (lockset (b :: as)).filter (a.held.contains ·)

/-- Race-free: no access at all, or some lock is common to every access. -/
def consistent (as : List Access) : Bool := as.isEmpty || !(lockset as).isEmpty

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
spec lock across the fork. For the second, the GC holds `rsSpec` in *read* mode while the
writer and population accesses hold it in *write* mode, so the shared lock does exclude
them even though `Access` has no modes (see `Access`). -/
theorem h1_fix_populate_gil :
    consistent (idxMem.take 2 ++ [⟨"populate under GIL", [gil, indexer, rsGlobal, rsSpec]⟩])
      = true := by decide
theorem h1_fix_gc_spec_lock :
    consistent (⟨"ForkGC fork under spec R", [gil, rsSpec]⟩ :: idxMem.drop 1) = true := by
  decide

/-! ## The check is the lockset property -/

theorem mem_lockset : ∀ (as : List Access) {l : Nat}, l ∈ lockset as →
    ∀ a ∈ as, l ∈ a.held
  | [], _, _, a, ha => by simp at ha
  | [b], l, h, a, ha => by
    simp only [lockset] at h
    rcases List.mem_singleton.mp ha with rfl
    exact h
  | b :: c :: cs, l, h, a, ha => by
    simp only [lockset, List.mem_filter] at h
    rcases List.mem_cons.mp ha with rfl | hb
    · simpa using h.2
    · exact mem_lockset (c :: cs) h.1 a hb

/-- If the accesses are consistent, one lock is held at every access (vacuous for no accesses). -/
theorem consistent_common_lock (as : List Access) (h : consistent as = true) :
    ∃ l, ∀ a ∈ as, l ∈ a.held := by
  by_cases he : as.isEmpty = true
  · exact ⟨0, fun a ha => by simp [List.isEmpty_iff.mp he] at ha⟩
  · simp only [consistent, he, Bool.false_or, Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
    obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ h
    exact ⟨l, fun a ha => mem_lockset as hl a ha⟩

end SC.Guarded
