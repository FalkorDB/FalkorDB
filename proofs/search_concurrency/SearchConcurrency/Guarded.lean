/-
# Guarded accesses: which lock an FFI call needs held (#3483)

`Locks` checks lock *order* (deadlock freedom). It never asks what a thread touches
while it holds, or does not hold, a lock, and `Locks.dropIndexBgProg` models
`RediSearch_DropIndex` as touching nothing. So a data race behind an axiomatised FFI
call was invisible to it. This module adds the missing obligation, Eraser-style: every
access to a guarded resource happens with its guard held.

| here | there |
| --- | --- |
| `naRef`                 | refcount of RediSearch's process-global `NA_rstr` (`deps/RediSearch@ea1a6f4 src/info/index_error.c:29`). A `RedisModuleString` refcount is a plain `int`: guarded by the GIL |
| `createProg`            | `Index::create_rs_index` (`graph/src/index/mod.rs:1101`) → `RediSearch_CreateIndex`/`CreateField` → `IndexError_Init` → `RM_HoldString(NA_rstr)` (`index_error.c:42`); callers hold the GIL |
| `dropProg_ea1a6f4`      | `graph_free` (`src/graph_core.rs:1777`) drops on `std::thread::spawn` (`:1797`) → `SpecHandle::drop` (`mod.rs:996`, no GIL by design, #726) → `RediSearch_DropIndex` → `IndexSpec_Free` → `IndexError_Clear` → `RM_FreeString(NA_rstr)` (`index_error.c:91`) |
| `dropProg_rs29`         | FalkorDB/RediSearch#29 (a272834a6): errors borrow `NA_rstr` (`IndexError_FreeKey` skips it); the GC stop runs on the main thread via `RM_EventLoopAddOneShot` |

Live: issue #3483 (AOF replay, 6.0.3 crashes in `RM_FreeString`, `illegal decrRefCount
… type 0, encoding 8, refcount 0` = the embstr `"N/A"`). C v4.22.2 pins RediSearch
311f90126, which has no `IndexError`/`NA_rstr`, so the C engine cannot hit it.
-/

namespace SC.Guarded

/-- A thread step: take or release a lock, or touch a resource. -/
inductive GOp where
  | acq (l : Nat)
  | rel (l : Nat)
  | touch (r : Nat)
deriving DecidableEq, Repr

def gil : Nat := 0
def naRef : Nat := 0

/-- The lock a resource needs held; `none` = thread-safe (atomic or thread-local). -/
def guard : Nat → Option Nat
  | 0 => some gil  -- naRef: non-atomic `robj.refcount`
  | _ => none

/-- Every guarded touch happens with its guard in `held`. -/
def guardedOk : List Nat → List GOp → Bool
  | _, [] => true
  | held, .acq l :: ops => guardedOk (l :: held) ops
  | held, .rel l :: ops => guardedOk (held.erase l) ops
  | held, .touch r :: ops => (match guard r with | some l => held.contains l | none => true)
                             && guardedOk held ops

/-- Resources a program touches without the guard held. -/
def unguarded : List Nat → List GOp → List Nat
  | _, [] => []
  | held, .acq l :: ops => unguarded (l :: held) ops
  | held, .rel l :: ops => unguarded (held.erase l) ops
  | held, .touch r :: ops =>
      (match guard r with | some l => if held.contains l then [] else [r] | none => [])
      ++ unguarded held ops

/-- Writer / RDB-load / replica `GRAPH.EFFECT`: create an index under the GIL. -/
def createProg : List GOp := [.acq gil, .touch naRef, .rel gil]
/-- `graph_free`'s drop thread on RediSearch ea1a6f4: `IndexError_Clear` frees `NA_rstr`. -/
def dropProg_ea1a6f4 : List GOp := [.touch naRef]
/-- The same drop with RediSearch#29: the error borrows `NA_rstr`; nothing guarded. -/
def dropProg_rs29 : List GOp := []

/-- Two threads race on `r` if both touch it and at least one does so unguarded. -/
def races (p q : List GOp) (r : Nat) : Bool :=
  (unguarded [] p).contains r && (q.contains (.touch r) || p == q)
  || (unguarded [] q).contains r && p.contains (.touch r)

/-- **#3483.** The drop thread touches `NA_rstr`'s refcount without the GIL while the
main thread may be creating an index: a lost update, then a premature free. -/
theorem bug3483_drop_unguarded : guardedOk [] dropProg_ea1a6f4 = false := by decide
theorem bug3483_races_create : races createProg dropProg_ea1a6f4 naRef = true := by decide

/-- With RediSearch#29 every program is guarded and no pair races on `NA_rstr`. -/
theorem rs29_guarded : guardedOk [] createProg ∧ guardedOk [] dropProg_rs29 := by decide
theorem rs29_no_race : races createProg dropProg_rs29 naRef = false := by decide

/-- The guard check is the lockset property: no unguarded touches. -/
theorem guardedOk_iff (held : List Nat) (p : List GOp) :
    guardedOk held p = true ↔ unguarded held p = [] := by
  induction p generalizing held with
  | nil => simp [guardedOk, unguarded]
  | cons op ops ih =>
    cases op with
    | acq l => simp [guardedOk, unguarded, ih]
    | rel l => simp [guardedOk, unguarded, ih]
    | touch r =>
      simp only [guardedOk, unguarded, Bool.and_eq_true, List.append_eq_nil_iff, ih]
      cases guard r with
      | none => simp
      | some l => by_cases h : held.contains l <;> simp_all

end SC.Guarded
