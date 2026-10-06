/-
# PR #2845, columnar side: `rows_only`, `has_origins`, the empty-projection
# fast path and the emitter's gather guard

Model of the merged (now origin/main a9377c636) graph/src/runtime/batch.rs (`rows_only` :853,
`gather` :897-918, `concat` :979-1053, `origin_row` :1298, `has_origins`
:1312, `BatchBuilder::push_row_with` :577, `BatchBuilder::finish` :750-797),
runtime/ops/project.rs `eval` :54-147, and
runtime/ops/batched_result_emitter.rs `start_batch` :602, `finish_batch` :648.
Columns are abstracted to their bound/unbound shape plus values (`Option
(List V)`, `none` = `Column::Unbound`) — the only thing these functions
inspect, the value-level laws being proofs/columnar's.
-/
namespace Pr2845

structure Batch (V : Type) where
  len : Nat
  sel : Option (List Nat)
  cols : List (Option (List V))
  origins : Option (List Nat)
  deriving DecidableEq, Repr

namespace Batch
variable {V : Type}

def active (b : Batch V) : List Nat := b.sel.getD (List.range b.len)

/-- `origin_row` :1298. -/
def originRow (b : Batch V) (r : Nat) : Nat := match b.origins with
  | none => 0
  | some o => o.getD r 0

/-- `has_origins` :1312. -/
def hasOrigins (b : Batch V) : Bool := b.origins.isSome

/-- `rows_only` :853. -/
def rowsOnly (n : Nat) : Batch V := ⟨n, none, [], none⟩

def setOriginRows (b : Batch V) (o : List Nat) : Batch V := { b with origins := some o }

/-- `gather` :897: origins kept only when some gathered one is non-zero. -/
def gather [Inhabited V] (b : Batch V) (idx : List Nat) : Batch V :=
  ⟨idx.length, none, b.cols.map (Option.map fun c => idx.map (c.getD · default)),
    b.origins.bind fun o =>
      let g := idx.map (o.getD · 0)
      if g.any (· != 0) then some g else none⟩

end Batch

open Batch

theorem hasOrigins_iff {V : Type} (b : Batch V) : b.hasOrigins = true ↔ b.origins ≠ none := by
  cases h : b.origins <;> simp [hasOrigins, h]

/-- A batch without the sidecar reads origin 0 everywhere. -/
theorem originRow_none {V : Type} (b : Batch V) (h : b.hasOrigins = false) (r : Nat) :
    b.originRow r = 0 := by
  cases hb : b.origins with
  | none => simp [originRow, hb]
  | some _ => simp [hasOrigins, hb] at h

/-- `gather` reads each picked row's origin (proofs/columnar `gather_originRow`). -/
theorem gather_originRow {V : Type} [Inhabited V] (b : Batch V) (idx : List Nat) (k : Nat)
    (hk : k < idx.length) : (b.gather idx).originRow k = b.originRow idx[k] := by
  cases hb : b.origins with
  | none => simp [gather, originRow, hb]
  | some o =>
    simp only [gather, originRow, hb, Option.bind_some]
    by_cases hany : (idx.map (o.getD · 0)).any (· != 0) = true
    · simp only [hany, ite_true]
      simp [List.getD_eq_getElem?_getD, hk]
    · simp only [hany, Bool.false_eq_true, ite_false]
      simp only [List.any_eq_true, List.mem_map, bne_iff_ne, ne_eq, not_exists, not_and,
        Decidable.not_not, forall_exists_index, and_imp] at hany
      exact (hany _ idx[k] (List.getElem_mem hk) rfl).symm

/-! ## `ProjectOp::eval` on a projection naming nothing -/

/-- `BatchBuilder::finish` (:750) after pushing rows that bind no slot
(`push_row` of an empty `Row`: `base.len() = 0`, no extra, so no column is
ever opened, :585-595): `rows = 0` gives the empty batch; otherwise `rows`
rows, no columns, origins iff some stamped origin is non-zero
(`any_origin`). -/
def builderFinishEmptyRows {V : Type} (origins : List Nat) : Batch V :=
  if origins.length = 0 then ⟨0, none, [], none⟩
  else ⟨origins.length, none, [], if origins.any (· != 0) then some origins else none⟩

/-- origin/main's path for `trees = [] ∧ copy_from_parent = []`: one empty
`Row` per active row with `origin_row = batch.origin_row(row)` (project.rs:111-129). -/
def projectEmptyGeneral {V : Type} (b : Batch V) : Batch V :=
  builderFinishEmptyRows (b.active.map b.originRow)

/-- The PR's fast path (project.rs:98-105). -/
def projectEmptyFast {V : Type} (b : Batch V) : Batch V :=
  let os := b.active.map b.originRow
  let out : Batch V := rowsOnly b.active.length
  if os.any (· != 0) then out.setOriginRows os else out

/-- **The fast path is the general path, exactly** (same length, no columns,
same origins sidecar) — so the PR changes no observable of `ProjectOp`. -/
theorem projectEmpty_fast_eq_general {V : Type} (b : Batch V) :
    projectEmptyFast b = projectEmptyGeneral b := by
  unfold projectEmptyFast projectEmptyGeneral builderFinishEmptyRows rowsOnly setOriginRows
  by_cases h0 : b.active.length = 0
  · have : b.active = [] := List.eq_nil_of_length_eq_zero h0
    simp [this]
  · simp only [List.length_map, h0, ite_false]
    split <;> simp_all

/-- `ProjectOp::eval` dispatch. `fastCols`/`general` stand for the two
pre-existing paths (column fast path :80-91, row path :108-129; the latter's
value semantics is proofs/columnar `eval_sound`/`evalValues_sound`). -/
def projectMain {T C V : Type} (fastCols general : List T → List C → Batch V → Batch V)
    (trees : List T) (copies : List C) (b : Batch V) : Batch V :=
  if b.active ≠ [] ∧ trees ≠ [] ∧ copies = [] then fastCols trees copies b else general trees copies b

def projectMerged {T C V : Type} (fastCols general : List T → List C → Batch V → Batch V)
    (trees : List T) (copies : List C) (b : Batch V) : Batch V :=
  if b.active ≠ [] ∧ trees ≠ [] ∧ copies = [] then fastCols trees copies b
  else if trees = [] ∧ copies = [] then projectEmptyFast b
  else general trees copies b

/-- **Project semantics preserved.** If the row path with no projections is
`projectEmptyGeneral` (it is: :111-129 with both loops empty), the merged
operator equals origin/main's on every input, so every theorem about
origin/main's `ProjectOp` (incl. `eval_sound` for its columns) carries over. -/
theorem projectMerged_eq_main {T C V : Type} (fastCols general : List T → List C → Batch V → Batch V)
    (hgen : ∀ b, general [] [] b = projectEmptyGeneral b)
    (trees : List T) (copies : List C) (b : Batch V) :
    projectMerged fastCols general trees copies b = projectMain fastCols general trees copies b := by
  unfold projectMerged projectMain
  split
  · rfl
  · split
    · rename_i _ h; obtain ⟨rfl, rfl⟩ := h; rw [hgen, projectEmpty_fast_eq_general]
    · rfl

/-! ## `BatchedResultEmitter`: the gather guard -/

/-- `start_batch` before (origin/main :590) and after (merged :602-606). -/
def shouldExpandOld {V : Type} (b : Batch V) : Bool := b.cols.length > 0
def shouldExpandNew {V : Type} (b : Batch V) : Bool := b.cols.length > 0 || b.hasOrigins

/-- `finish_batch` (:648-664) for `count = idx.length > 0` packed results
with parent rows `idx`: gather the parent, or a fresh `Batch::new(0)`; then
`I::finish` installs the result lane via `set_column`, whose first column
sets `len` (:1238) — when the binding has one. -/
def finishBatch {V : Type} [Inhabited V] (expand : Bool) (parent : Batch V)
    (idx : List Nat) (lane : Option (List V)) : Batch V :=
  let out : Batch V := if expand then parent.gather idx else ⟨0, none, [], none⟩
  match lane with
  | none => out
  | some vs => { out with len := if out.len = 0 then vs.length else out.len,
                          cols := out.cols ++ [some vs] }

theorem finishBatch_originRow {V : Type} [Inhabited V] (e : Bool) (p : Batch V) (idx : List Nat)
    (lane : Option (List V)) (k : Nat) :
    (finishBatch e p idx lane).originRow k =
      (if e then (p.gather idx).originRow k else 0) := by
  unfold finishBatch
  cases lane <;> cases e <;> simp [originRow]

/-- **Correlation preserved (merged).** Every packed result carries the
origin of the parent row it was produced from — whatever the parent's
shape, columnless entry projections included. -/
theorem emit_origin_sound {V : Type} [Inhabited V] (p : Batch V) (idx : List Nat)
    (lane : Option (List V)) (k : Nat) (hk : k < idx.length) :
    (finishBatch (shouldExpandNew p) p idx lane).originRow k = p.originRow idx[k] := by
  rw [finishBatch_originRow]
  split
  · exact gather_originRow p idx k hk
  · rename_i h
    simp only [shouldExpandNew, Bool.or_eq_true, decide_eq_true_eq, not_or] at h
    have : p.hasOrigins = false := by simpa using h.2
    rw [originRow_none p this]

/-- origin/main lost it: a columnless parent with origins `[0,1,2]` (the
entry projection of an import-less body fed three outer rows), results from
rows 1 and 2 — both come out tagged 0, i.e. attached to the first outer row
(the PR's "3 x 3 collapses onto the first x"). -/
def colless : Batch Nat := ⟨3, none, [], some [0, 1, 2]⟩

theorem pre2845_emit_origin_lost_old :
    (finishBatch (shouldExpandOld colless) colless [1, 2] (some [7, 8])).originRow 0 = 0 ∧
    colless.originRow 1 = 1 := by decide

/-- Row count: `count` whenever the parent has a column, an origin, or the
binding installs a lane. -/
theorem emit_len {V : Type} [Inhabited V] (p : Batch V) (idx : List Nat) (lane : Option (List V))
    (hl : ∀ vs, lane = some vs → vs.length = idx.length) (hne : idx ≠ [])
    (h : shouldExpandNew p = true ∨ lane.isSome) :
    (finishBatch (shouldExpandNew p) p idx lane).len = idx.length := by
  have hlen : idx.length ≠ 0 := by intro h0; exact hne (List.eq_nil_of_length_eq_zero h0)
  unfold finishBatch
  cases hs : shouldExpandNew p <;> cases lane <;> simp_all [gather]

/-- Residual (unchanged from proofs/columnar `finishLen_drops`): a no-alias
emitter over a parent with no column AND no origin yields 0 rows. Its one
user, ExpandInto (expand_into.rs:96), always has both endpoint columns. -/
theorem emit_len_residual :
    (finishBatch (V := Nat) (shouldExpandNew (rowsOnly 3 : Batch Nat)) (rowsOnly 3) [0, 1, 2] none).len = 0 := by
  decide

/-! ## `Batch::concat` over columnless batches -/

/-- `concat` :979-1053, restricted to batches with no columns (what the
empty entry projection and its consumers produce): `total` rows, no
columns, origins concatenated over active rows, kept iff some is non-zero. -/
def concatColless {V : Type} (bs : List (Batch V)) : Batch V :=
  let total := (bs.map fun b => b.active.length).sum
  if total = 0 then ⟨0, none, [], none⟩ else
  let os := bs.flatMap fun b => b.active.map b.originRow
  ⟨total, none, [], if os.any (· != 0) then some os else none⟩

/-- Per-row definition (pushing every active row of every batch into one
`BatchBuilder`) = the same batch: concat keeps row count and correlation. -/
theorem concatColless_perRow {V : Type} (bs : List (Batch V)) :
    concatColless bs = builderFinishEmptyRows (bs.flatMap fun b => b.active.map b.originRow) := by
  unfold concatColless builderFinishEmptyRows
  have hl : (bs.flatMap fun b => b.active.map b.originRow).length =
      (bs.map fun b => b.active.length).sum := by
    simp [List.length_flatMap]
  rw [hl]

end Pr2845
