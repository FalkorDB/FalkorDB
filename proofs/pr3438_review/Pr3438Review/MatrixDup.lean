import GraphblasWrappers.Glue
/-
# `Encode<19> for Matrix<T>` unloads a private `dup()` (matrix.rs:520-575, PR 2fabe1a8a)

```rust
let copy = self.dup();                                     // GrB_Matrix_dup (+ hyper-hash opt-out)
GxB_Container_new(&container);
GxB_unload_Matrix_into_Container(copy.inner(), container); // copy left empty
/* write scrubbed header bytes, then x, h, p, i, b through ManuallyDrop<Vector> views */
GxB_Container_free(&c);                                    // frees the arrays unloaded from copy
/* end of fn: `copy` dropped → Handle::drop → GrB_Matrix_free(copy) */
```

main (9212add39 :519-571) unloaded `self.inner()` and loaded it back at the end.

**(a) Payload.** graphblas_wrappers models main's encoder as `GBWGlue.encodeM unload load`
with the C-spec hypotheses `UnloadLoad` (load ∘ unload = id) and `PtrBlind`
(load ignores header pointer fields). The PR's tokens are main's tokens for
`dup m` (`encodePR_eq_main_of_dup`). With the FFI hypothesis `DupSpec`
(`GrB_Matrix_dup`: "make an exact copy" — same type, dims, pattern and values;
GB_dup.c / GB_dup_worker.c also keep the sparsity format, iso, int widths and
pending work, but allocate the arrays at their exact size), **`decode_pr`**: the
PR's payload decodes to a matrix with `m`'s content. The *bytes* need not be
main's: `Vector<bool>::encode` writes `n_bytes` = the array's allocated size
(GxB_Vector_unload's `n_bytes`), and `dup` drops slack. `pr3438_bytes_can_differ`
is a model satisfying every hypothesis where they differ, and it happens for
real: scratch test `lean3438_matrix_encode_dup_vs_main` — a 300-entry
`Matrix<u64>` built by `set` writes an `x` array of 4096 bytes on main, 2400 on
the PR (both decode to the same 300 entries); after removals both write 1600.
So claim 5(a) holds as **decode-equivalence**, not byte-equality
(`encodePR_eq_main_of_dup` gives byte-equality exactly when `unload (dup m) = unload m`).

**(b), (c) Handles.** `Heap` tracks live matrix handles (full with a content, or
emptied by an unload) and live containers (holding unloaded arrays or not);
`step` is the C contract of each call and fails (`none`) on a use-after-free,
double free, unload of an emptied matrix, or allocating over a live handle.
* **`pr_self_never_unloaded`** — every state along the PR's encode has `self`
  full with its original content: a lock-free reader of `self` sees the same
  matrix before, during and after.
* **main empties `self`** (counterexample removed; in 17ccc5412) — main's sequence passes through a state where
  `self` is emptied (the #3434 window), then restores it. Confirmed live:
  `lean3438_matrix_encode_dup_vs_main` reads `self.nvals()` between main's
  unload and load — 0 for a 300-entry matrix; after the PR's encode `self`
  still iterates the same 300 entries.
* **`pr_no_leak_no_double_free`** — the PR's sequence runs (every free hits a
  live handle exactly once) and ends in the starting heap: the copy and the
  container are both freed; `GrB_Matrix_free` on the emptied copy is valid
  (GB_unload_into_container.c leaves `A->magic = GB_MAGIC` with no arrays).
  The written arrays are the copy's (`pr_writes_self_content`).
  `pr_panic_after_dup` — if `GxB_Container_new` fails (the `assert_eq!`
  panics), unwinding drops `copy` and the heap is again the starting one.
  The five `ManuallyDrop<Vector>` views free nothing (modelled as `write`).
-/
namespace Pr3438
open GBWGlue

/-! ## (a) the payload -/

/-- The PR's tokens: main's encoder applied to the private copy. -/
def encodePR {M α : Type} (dup : M → M) (unload : M → Container α) (m : M) : List (Tok α) :=
  let c := unload (dup m)
  [.bytes (scrub c.hdr), .vec c.x, .vec c.h, .vec c.p, .vec c.i, .vec c.b]

theorem encodePR_eq_main_of_dup {M α : Type} (dup : M → M) (unload : M → Container α)
    (load : Container α → M) (m : M) : encodePR dup unload m = (encodeM unload load (dup m)).1 := rfl

/-- Byte-equality with main exactly when the copy unloads to the same container. -/
theorem encodePR_bytes_eq {M α : Type} (dup : M → M) (unload : M → Container α)
    (load : Container α → M) (m : M) (h : unload (dup m) = unload m) :
    encodePR dup unload m = (encodeM unload load m).1 := by
  simp [encodePR, encodeM, h]

/-- FFI hypothesis on `GrB_Matrix_dup`: the copy has the same content. -/
def DupSpec {M C : Type} (content : M → C) (dup : M → M) : Prop := ∀ m, content (dup m) = content m

/-- **`decode_pr`**: the PR's payload decodes to `m`'s content. -/
theorem decode_pr {M α C : Type} (dup : M → M) (unload : M → Container α) (load : Container α → M)
    (content : M → C) (hul : UnloadLoad unload load) (hb : PtrBlind load) (hd : DupSpec content dup) (m : M) :
    (decodeC (encodePR dup unload m)).map (fun c => content (load c)) = some (content m) := by
  rw [encodePR_eq_main_of_dup dup unload load]
  have := congrArg (Option.map content) (decodeC_encodeM unload load hul hb (dup m))
  simp only [Option.map_map, Option.map_some, hd m] at this
  exact this

/-- #3097 still holds on the PR: no pointer field of the written header carries
a heap address. -/
theorem pr_no_heap_addresses {M α : Type} (dup : M → M) (unload : M → Container α)
    (load : Container α → M) (m : M) :
    ∀ hd, Tok.bytes hd ∈ encodePR dup unload m → ∀ f ∈ hd, f.ptr = true → f.val = 0 := by
  rw [encodePR_eq_main_of_dup dup unload load]
  exact encodeM_no_heap_addresses unload load (dup m)

/-- A model where every hypothesis holds and the PR's bytes are not main's: a
matrix is `(content, slack)`, the container's `x` carries the slack, `dup`
drops it. -/
theorem pr3438_bytes_can_differ :
    let unload : Nat × Nat → Container Nat := fun m => ⟨[], m.1, 0, 0, 0, m.2⟩
    let load : Container Nat → Nat × Nat := fun c => (c.x, c.b)
    let dup : Nat × Nat → Nat × Nat := fun m => (m.1, 0)
    UnloadLoad unload load ∧ PtrBlind load ∧ DupSpec Prod.fst dup ∧
      encodePR dup unload (5, 3) ≠ (encodeM unload load (5, 3)).1 := by
  refine ⟨fun m => rfl, fun c => rfl, fun m => rfl, ?_⟩
  simp [encodePR, encodeM]

/-! ## (b), (c) handles -/

inductive MS (C : Type) where
  | full (c : C)
  | empty

structure Heap (C : Type) where
  mats : Nat → Option (MS C)
  conts : Nat → Option (Option C)

def upd {β : Type} (f : Nat → β) (k : Nat) (v : β) : Nat → β := fun x => if x = k then v else f x

inductive Op where
  | dup (src dst : Nat)
  | cnew (k : Nat)
  | unload (h k : Nat)
  | load (h k : Nat)
  | write (k : Nat)
  | cfree (k : Nat)
  | mfree (h : Nat)

/-- The C contract of each call; `none` = undefined behaviour (use-after-free,
double free, unload of an emptied matrix, a handle allocated twice). -/
def step {C : Type} (s : Heap C) : Op → Option (Heap C)
  | .dup src dst => match s.mats src, s.mats dst with
    | some (.full c), none => some { s with mats := upd s.mats dst (some (.full c)) }
    | _, _ => none
  | .cnew k => match s.conts k with
    | none => some { s with conts := upd s.conts k (some none) }
    | some _ => none
  | .unload h k => match s.mats h, s.conts k with
    | some (.full c), some none =>
      some { mats := upd s.mats h (some .empty), conts := upd s.conts k (some (some c)) }
    | _, _ => none
  | .load h k => match s.mats h, s.conts k with
    | some _, some (some c) =>
      some { mats := upd s.mats h (some (.full c)), conts := upd s.conts k (some none) }
    | _, _ => none
  | .write k => match s.conts k with
    | some (some _) => some s
    | _ => none
  | .cfree k => match s.conts k with
    | some _ => some { s with conts := upd s.conts k none }
    | none => none
  | .mfree h => match s.mats h with
    | some _ => some { s with mats := upd s.mats h none }
    | none => none

/-- Run a sequence, returning every state reached after each call. -/
def trace {C : Type} (s : Heap C) : List Op → Option (List (Heap C))
  | [] => some []
  | o :: os => match step s o with
    | none => none
    | some s' => (trace s' os).map (s' :: ·)

def finalOf {C : Type} (s : Heap C) (tr : List (Heap C)) : Heap C := tr.getLastD s

/-- main's `encode` (9212add39). -/
def mainOps (sf k : Nat) : List Op := [.cnew k, .unload sf k, .write k, .load sf k, .cfree k]

/-- The PR's `encode`. -/
def prOps (sf cp k : Nat) : List Op := [.dup sf cp, .cnew k, .unload cp k, .write k, .cfree k, .mfree cp]

structure Fresh {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) : Prop where
  sf_full : s.mats sf = some (.full c)
  cp_free : s.mats cp = none
  k_free : s.conts k = none
  ne : cp ≠ sf

theorem pr_trace {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) (h : Fresh s sf cp k c) :
    trace s (prOps sf cp k) = some
      (let s1 : Heap C := { s with mats := upd s.mats cp (some (.full c)) }
       let s2 : Heap C := { s1 with conts := upd s.conts k (some none) }
       let s3 : Heap C := { mats := upd s1.mats cp (some .empty), conts := upd s2.conts k (some (some c)) }
       let s5 : Heap C := { s3 with conts := upd s3.conts k none }
       let s6 : Heap C := { s5 with mats := upd s5.mats cp none }
       [s1, s2, s3, s3, s5, s6]) := by
  obtain ⟨h1, h2, h3, hne⟩ := h
  simp [prOps, trace, step, h1, h2, h3, upd]

/-- **(b)** Along the PR's encode `sf` is never touched. -/
theorem pr_self_never_unloaded {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) (h : Fresh s sf cp k c) :
    ∃ tr, trace s (prOps sf cp k) = some tr ∧ ∀ s' ∈ tr, s'.mats sf = some (.full c) := by
  refine ⟨_, pr_trace s sf cp k c h, ?_⟩
  have hne := h.ne
  simp [upd, Ne.symm hne, h.sf_full]

/-- **(c)** It runs without a use-after-free or double free and ends in the
starting heap: no leak. -/
theorem pr_no_leak_no_double_free {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) (h : Fresh s sf cp k c) :
    ∃ tr, trace s (prOps sf cp k) = some tr ∧ (finalOf s tr).mats = s.mats ∧ (finalOf s tr).conts = s.conts := by
  refine ⟨_, pr_trace s sf cp k c h, ?_, ?_⟩
  · funext x; simp only [finalOf, List.getLastD, List.getLast, upd]
    by_cases hx : x = cp
    · subst hx; simp [h.cp_free]
    · simp [hx]
  · funext x; simp only [finalOf, List.getLastD, List.getLast, upd]
    by_cases hx : x = k
    · subst hx; simp [h.k_free]
    · simp [hx]

/-- What is written is the copy's content, i.e. `sf`'s. -/
theorem pr_writes_self_content {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) (h : Fresh s sf cp k c) :
    ∃ tr, trace s (prOps sf cp k) = some tr ∧ ∃ s', tr[3]? = some s' ∧ s'.conts k = some (some c) := by
  refine ⟨_, pr_trace s sf cp k c h, _, rfl, ?_⟩
  simp [upd]

/-- If `GxB_Container_new` fails, the panic drops `copy`: back to the start. -/
theorem pr_panic_after_dup {C : Type} (s : Heap C) (sf cp k : Nat) (c : C) (h : Fresh s sf cp k c) :
    (trace s [.dup sf cp, .mfree cp]).map (fun tr => (finalOf s tr).mats) = some s.mats := by
  have hu : upd s.mats cp (some (MS.full c)) cp = some (.full c) := by simp [upd]
  have hf : upd (upd s.mats cp (some (MS.full c))) cp none = s.mats := by
    funext x; by_cases hx : x = cp
    · subst hx; simp [upd, h.cp_free]
    · simp [upd, hx]
  simp [trace, step, h.sf_full, h.cp_free, hu, finalOf, hf]

end Pr3438
