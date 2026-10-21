/-
# Claim 3: the `'static` guard never outlives the cell it borrows

| here | there |
| --- | --- |
| `Alloc`, `H`           | an `Arc<AtomicRefCell<Graph>>` heap allocation: strong count, borrow word (`Cell.B`); `none` = freed |
| `Field.guard a`        | `Runtime.read_guard = Some(AtomicRef)` into the cell at `a` (`runtime.rs:147`); dropping it is `AtomicBorrowRef::drop` — `fetch_sub(1)` on the cell's borrow word (atomic_refcell `lib.rs:273-283`), a write into the allocation |
| `Field.arc a`          | `Runtime.g: Arc<…>` (`runtime.rs:150`); dropping it decrements the strong count and frees the allocation at 0 |
| `Field.other`          | every other field (no access to the cell) |
| `runtimeFields`        | `struct Runtime` field order (`runtime.rs:142-…`): `parameters`, `read_guard`, `g`, `write`, … |
| `dropFields`           | Rust's drop glue for a struct without `Drop` impl: fields in declaration order (Reference, "Destructors": *the fields of a struct are dropped in declaration order*). `Runtime` has no `impl Drop` (grep) |

Facts checked in the tree at d7d166e82, not by Lean: `Runtime` has no `Drop` impl; there is
no `&mut Runtime` and no `let mut runtime`/`mut rt` binding anywhere (`grep -rn "&mut
Runtime\|let mut runtime\|let mut rt\b"` is empty), so the `pub g` field is never
reassigned after `new`, and `read_guard` is private, so the `AtomicRef<'static>` never
leaves `Runtime` except as `GraphRef::Held(&'_ Graph)`, bounded by `&self`. A move of
`Runtime` copies the guard's pointer, which points into the `Arc`'s heap allocation, not
into `Runtime` — so moves are irrelevant (`move_irrelevant`).
-/
import Pr3436Review.Cell

namespace Pr3436.Drop
open Pr3436.Cell

structure Alloc where
  strong : Nat
  b : B
  deriving DecidableEq, Repr

/-- The heap: `none` = freed (or never allocated). -/
abbrev H := Addr → Option Alloc

def H.set (h : H) (a : Addr) (v : Option Alloc) : H := fun x => if x = a then v else h x

inductive Field where
  | guard (a : Addr)
  | arc (a : Addr)
  | other
  deriving DecidableEq, Repr

/-- `none` = undefined behaviour (a write into a freed allocation, or a double free). -/
def dropField (h : H) : Field → Option H
  | .guard a => match h a with
    | some al => some (h.set a (some { al with b := release al.b }))
    | none => none                                  -- use after free
  | .arc a => match h a with
    | some al => some (h.set a (if al.strong ≤ 1 then none else some { al with strong := al.strong - 1 }))
    | none => none
  | .other => some h

def dropFields (h : H) : List Field → Option H
  | [] => some h
  | f :: fs => match dropField h f with
    | some h' => dropFields h' fs
    | none => none

/-- The PR's field order, for a runtime `rt` (`parameters`, `read_guard`, `g`, the rest). -/
def runtimeFields (rt : Rt) : List Field :=
  [.other] ++ (match rt.guard with | some a => [.guard a] | none => []) ++ [.arc rt.g, .other]

/-- **Claim 3.** Dropping a `Runtime` whose `g` is live never touches freed memory — even
when `g` is the *last* strong reference — because the guard (on `g`'s own cell, `Rt.ok`)
is released before `g`. -/
theorem drop_sound (rt : Rt) (hr : rt.ok) (h : H) (al : Alloc) (ha : h rt.g = some al) :
    ∃ h', dropFields h (runtimeFields rt) = some h' := by
  unfold Rt.ok at hr
  cases hw : rt.write with
  | true =>
    simp [hw] at hr
    simp [runtimeFields, hr, dropFields, dropField, ha]
  | false =>
    simp [hw] at hr
    simp [runtimeFields, hr, dropFields, dropField, ha, H.set]

/-- The order is load-bearing: had `g` been declared before `read_guard`, dropping a
read runtime holding the last strong reference would write into freed memory. -/
def swappedFields (rt : Rt) : List Field :=
  [.other, .arc rt.g] ++ (match rt.guard with | some a => [.guard a] | none => []) ++ [.other]

theorem swapped_unsound :
    let rt : Rt := { g := 0, write := false, guard := some 0 }
    let h : H := fun x => if x = 0 then some { strong := 1, b := .shared 1 } else none
    dropFields h (swappedFields rt) = none := by
  decide

/-- With another strong reference alive (a read runtime's version is also held by
`MvccGraph.graph` while its session holds the read lock, since `commit` needs the write
lock), either order is safe — so even an unwind out of the middle of `Runtime::new`'s
struct literal, which drops the already-built operands, cannot dangle. -/
theorem swapped_ok_if_shared (rt : Rt) (hr : rt.ok) (h : H) (al : Alloc)
    (ha : h rt.g = some al) (hs : 2 ≤ al.strong) :
    ∃ h', dropFields h (swappedFields rt) = some h' := by
  unfold Rt.ok at hr
  have h1 : ¬ al.strong ≤ 1 := by omega
  cases hw : rt.write with
  | true =>
    simp [hw] at hr
    simp [swappedFields, hr, dropFields, dropField, ha, h1]
  | false =>
    simp [hw] at hr
    simp [swappedFields, hr, dropFields, dropField, ha, h1, H.set]

/-- After the drop, the cell's borrow word is back where it was before `new` (the guard
gave back exactly its one shared borrow), when `g` was not the last reference. -/
theorem drop_restores_borrow (rt : Rt) (hr : rt.ok) (hw : rt.write = false) (h : H) (n s : Nat)
    (ha : h rt.g = some { strong := s + 2, b := .shared (n + 1) }) :
    (dropFields h (runtimeFields rt)).bind (fun h' => h' rt.g) =
      some { strong := s + 1, b := .shared n } := by
  unfold Rt.ok at hr; simp [hw] at hr
  simp [runtimeFields, hr, dropFields, dropField, ha, H.set, release]

/-- A `Runtime` value holds addresses into the heap, never into itself: moving it (a
bitwise copy to a new location) leaves every field — and so the drop sequence — unchanged. -/
theorem move_irrelevant (rt : Rt) (moved : Rt) (hm : moved = rt) :
    runtimeFields moved = runtimeFields rt := by rw [hm]

end Pr3436.Drop
