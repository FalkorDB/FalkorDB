import FalkorOpsApply.Correlated
import FalkorOpsApply.CreateBatch
import FalkorOpsApply.Delete
/-
# Per-row write visibility through Apply (gap closed 2026-10-20)

Correlated.lean models a sub-plan as a *pure* function of its argument rows
(`SubPlan`), so a CALL subquery that writes, and the order in which its writes
become visible to later rows, was outside every theorem.  openCypher (and the C
engine's per-record `Argument` tap with eager write operators) runs a correlated
subquery to completion for one outer row before the next: every write of row `i`
is visible to rows `i+1, …`, and every reference a row holds names one entity.

Before #2968 Rust batched writing CALL subqueries (#2738) and streamed CREATE → DELETE
batch by batch (#2729); both are fixed and proved here (`applyW_write_eq_ref`,
`pipeline_partitionInvariant`).  Stale node references after id recycling (#3164) remain.
The sufficient condition under which batching would be sound for a writing sub-plan
(`applyBatchedW_eq_ref_of_readIndep`, property `WriteSeparable`) is kept.
-/
namespace RowWrites

open Correlated

/-! ## A writing sub-plan, and Apply over it -/

/-- A correlated sub-plan that reads the graph and queues writes.
* `read g x` — the rows the sub-plan's reading operators (scans, traversals, filters,
  projections: the row-local part, `rowLocal`) produce for argument row `x` against
  the committed graph `g`;
* `writes y` — what `Create`/`Delete`/`Set` queue in `pending` for output row `y`;
* `commit g ws` — `CommitOp::next` → `Pending::commit` (`commit.rs`, `pending.rs`). -/
structure WSub (G α β W : Type) where
  read : G → α → List β
  writes : β → List W
  commit : G → List W → G

/-- Batched Apply over a writing sub-plan (`apply.rs` + `commit.rs`): ONE
sub-plan instance gets the whole argument batch; its reads all see `g`; the blocking
`Commit` drains every row's writes, then commits them at once. -/
def applyBatchedW {G α β W : Type} (s : WSub G α β W) (g : G) (xs : List α) :
    List (α × β) × G :=
  let outs := xs.flatMap (fun x => (s.read g x).map (fun y => (x, y)))
  (outs, s.commit g (outs.flatMap (fun p => s.writes p.2)))

/-- Reference (openCypher; C's CALL {} with eager write ops): the subquery runs to
completion, commit included, for each outer row in turn, threading the graph.  This is
also exactly Rust's per-row mode (`next_per_row`, `apply.rs`: one sub-plan
instance, hence one `Commit`, per row). -/
def applyRefW {G α β W : Type} (s : WSub G α β W) : G → List α → List (α × β) × G
  | g, [] => ([], g)
  | g, x :: xs =>
    let ys := (s.read g x).map (fun y => (x, y))
    let g1 := s.commit g (ys.flatMap (fun p => s.writes p.2))
    let r := applyRefW s g1 xs
    (ys ++ r.1, r.2)

/-- Batched Apply over a writing sub-plan returns the same rows and the same final graph as
the per-row reference, for every graph and every input batch (what #2738 violated; since
#2968 `can_batch` admits no writing sub-plan). -/
def WriteSeparable {G α β W : Type} (s : WSub G α β W) : Prop :=
  ∀ g xs, applyBatchedW s g xs = applyRefW s g xs

/-- The rows batched Apply returns are exactly Correlated's `applyBatched` over the
row-local read part frozen at `g`: this file extends that model, it does not replace it. -/
theorem applyBatchedW_rows {G α β W : Type} (s : WSub G α β W) (g : G) (xs : List α) :
    (applyBatchedW s g xs).1 = applyBatched Prod.mk (rowLocal (s.read g)) xs := by
  rw [applyBatched_eq_ref _ _ (rowLocal_separable _)]
  unfold applyBatchedW applyRef single rowLocal
  simp [Function.comp_def]

/-- Sufficient condition (what a fix can aim for): the sub-plan's reads do not observe its
own writes, committing nothing is a no-op, and committing in one go equals committing row
by row. -/
def ReadIndep {G α β W : Type} (s : WSub G α β W) : Prop :=
  (∀ g ws x, s.read (s.commit g ws) x = s.read g x) ∧
  (∀ g, s.commit g [] = g) ∧
  (∀ g w1 w2, s.commit (s.commit g w1) w2 = s.commit g (w1 ++ w2))

theorem applyBatchedW_eq_ref_of_readIndep {G α β W : Type} (s : WSub G α β W)
    (h : ReadIndep s) : WriteSeparable s := by
  obtain ⟨hr, h0, hc⟩ := h
  have hread : ∀ g ws, s.read (s.commit g ws) = s.read g := fun g ws => funext (hr g ws)
  intro g xs
  induction xs generalizing g with
  | nil => simp [applyBatchedW, applyRefW, h0]
  | cons x xs ih =>
    have ih' := ih (s.commit g (((s.read g x).map (fun y => (x, y))).flatMap (fun p => s.writes p.2)))
    simp only [applyBatchedW, hread] at ih'
    simp only [applyBatchedW, applyRefW, ← ih', hc, List.flatMap_cons, List.flatMap_append]

/-! ## Writing sub-plans run per row (#2968, fixes #2738)

`ApplyOp::new` (`apply.rs`) blacklists every writing operator in `can_batch`, so a
writing CALL subquery takes `next_per_row`: one sub-plan instance, hence one `Commit`, per
outer row, in order.  When the body deletes (`sub_plan_deletes`), a row any of whose
imported entities (`imported_ids`: the variables the body's leading `WITH` reads) an
earlier row deleted runs nothing (`row_has_deleted_entity`, as C): a non-returning
CALL {} (`passthrough`) passes the input row through, a returning one yields no row. -/

/-- `next_per_row` over a writing sub-plan (`apply.rs`), flattened over output batches
(the `acc`/`overflow` re-batching keeps row order).  `skip g x` = `sub_plan_deletes &&
row_has_deleted_entity` against the graph the row runs on; `pass` = the passthrough row of a
non-returning CALL {} (`none` for a returning one; the OPTIONAL NULL fallback is
`Correlated.optionalPerRow`). -/
def applyPerRowW {G α β W : Type} (s : WSub G α β W) (skip : G → α → Bool) (pass : Option (α → β)) :
    G → List α → List (α × β) × G
  | g, [] => ([], g)
  | g, x :: xs =>
    if skip g x then
      let r := applyPerRowW s skip pass g xs
      ((match pass with | some p => [(x, p x)] | none => []) ++ r.1, r.2)
    else
      let ys := (s.read g x).map (fun y => (x, y))
      let g1 := s.commit g (ys.flatMap (fun p => s.writes p.2))
      let r := applyPerRowW s skip pass g1 xs
      (ys ++ r.1, r.2)

/-- A body that does not delete never skips (`sub_plan_deletes = false`): per-row mode is
exactly the reference, one commit per row. -/
theorem applyPerRowW_eq_ref {G α β W : Type} (s : WSub G α β W) (pass : Option (α → β))
    (g : G) (xs : List α) : applyPerRowW s (fun _ _ => false) pass g xs = applyRefW s g xs := by
  induction xs generalizing g with
  | nil => rfl
  | cons x xs ih => simp only [applyPerRowW, Bool.false_eq_true, ite_false, applyRefW, ih]

/-- A skipped row runs no part of the body: no writes, the graph is unchanged for the next row,
and it yields only the passthrough row (if any). -/
theorem applyPerRowW_skip {G α β W : Type} (s : WSub G α β W) (skip : G → α → Bool)
    (pass : Option (α → β)) (g : G) (x : α) (xs : List α) (h : skip g x = true) :
    applyPerRowW s skip pass g (x :: xs) =
      ((match pass with | some p => [(x, p x)] | none => []) ++ (applyPerRowW s skip pass g xs).1,
        (applyPerRowW s skip pass g xs).2) := by
  simp [applyPerRowW, h]

/-- `ApplyOp::next` over a writing sub-plan whose plan kinds are `sub`: batched iff
`can_batch`. -/
def applyW {G α β W : Type} (sub : List IRKind) (s : WSub G α β W) (skip : G → α → Bool)
    (pass : Option (α → β)) (g : G) (xs : List α) : List (α × β) × G :=
  if canBatchApply sub then applyBatchedW s g xs else applyPerRowW s skip pass g xs

/-- The guard rejects every sub-plan holding a writing operator (#2968). -/
theorem canBatchApply_rejects_write (sub : List IRKind) (k : IRKind) (hk : k ∈ sub)
    (hw : k ∈ writeKinds) : canBatchApply sub = false := by
  unfold canBatchApply
  simp only [Bool.not_eq_false', List.any_eq_true, decide_eq_true_eq]
  exact ⟨k, hk, List.mem_append_left _ hw⟩

/-- **#2738 fixed** (property `WriteSeparable` no longer needed): a sub-plan with a writing
operator runs per row, so Apply over it is the per-row reference (each row sees every earlier
row's committed writes) whenever no row is skipped; with skips it is `applyPerRowW`. -/
theorem applyW_write_eq_ref {G α β W : Type} (sub : List IRKind) (s : WSub G α β W)
    (pass : Option (α → β)) (hw : ∃ k ∈ sub, k ∈ writeKinds) (g : G) (xs : List α) :
    applyW sub s (fun _ _ => false) pass g xs = applyRefW s g xs := by
  obtain ⟨k, hk, hkw⟩ := hw
  simp only [applyW, canBatchApply_rejects_write sub k hk hkw, Bool.false_eq_true, ite_false]
  exact applyPerRowW_eq_ref s pass g xs

theorem applyW_write_perRow {G α β W : Type} (sub : List IRKind) (s : WSub G α β W)
    (skip : G → α → Bool) (pass : Option (α → β)) (hw : ∃ k ∈ sub, k ∈ writeKinds) (g : G)
    (xs : List α) : applyW sub s skip pass g xs = applyPerRowW s skip pass g xs := by
  obtain ⟨k, hk, hkw⟩ := hw
  simp only [applyW, canBatchApply_rejects_write sub k hk hkw, Bool.false_eq_true, ite_false]

/-- `MATCH (a:N) CREATE (:N)`: one row per `:N` in the committed graph, one node created
per row. -/
def matchCreateN : WSub Nat Nat Unit Unit where
  read g _ := List.replicate g ()
  writes _ := [()]
  commit g ws := g + ws.length

/-- **#2738's query, fixed**: `UNWIND [1,2,3] AS x CALL { MATCH (a:N) CREATE (:N) RETURN 1 AS
one } RETURN count(*)` over one `:N` (plan `Apply(Unwind, Project(Commit(Create(
NodeByLabelScan(Project(Argument))))))`) is no longer batched and returns 1+2+4 = 7 rows,
leaving 8 `:N`, as C. -/
theorem fixed2738_call_sees_writes :
    canBatchApply [.project, .commit, .create, .scan] = false ∧
    canBatchApply [.project, .commit, .delete, .project] = false ∧
    canBatchApply [.project, .commit, .setProps, .scan] = false ∧
    ((applyW [.project, .commit, .create, .scan] matchCreateN (fun _ _ => false) none 1
        [1, 2, 3]).1.length,
      (applyW [.project, .commit, .create, .scan] matchCreateN (fun _ _ => false) none 1
        [1, 2, 3]).2) = (7, 8) := by decide

/-- One `:A` node, alive (`true`) or deleted; `CALL { WITH n CREATE (n)-[:R]->(n) DELETE n
RETURN 1 AS z }`: one row, and its commit deletes `n`. -/
def createDeleteN : WSub Bool Nat Unit Unit where
  read _ _ := [()]
  writes _ := [()]
  commit g ws := if ws.isEmpty then g else false

/-- The skip (`row_has_deleted_entity`): `MATCH (n:A {id: 1}) UNWIND [1, 2] AS i CALL { WITH n
CREATE (n)-[:R]->(n) DELETE n RETURN 1 AS z } RETURN count(*)` → 1 (row 2's `n` was deleted
by row 1, so its body does not run); the non-returning form passes both rows through. -/
theorem skip_deleted_import :
    (applyW [.project, .commit, .delete, .create, .project] createDeleteN (fun g _ => !g) none
        true [1, 2]).1.length = 1 ∧
    (applyW [.project, .commit, .delete, .create, .project] createDeleteN (fun g _ => !g)
        (some (fun _ => ())) true [1, 2]).1.length = 2 := by decide

/-! ### Helpers of `next_per_row` (#2968) -/

/-- `row_has_deleted_entity` (`apply.rs`): some imported variable bound in the row holds a node
or relationship that is deleted (committed: id free; or pending-deleted).  `val id` = the
row's value (`none` = unbound), `delN`/`delR` = `is_*_deleted` of graph or pending. -/
def rowHasDeleted (imported : List Nat) (val : Nat → Option (CreateBatch.Val Unit))
    (delN delR : Nat → Bool) : Bool :=
  imported.any fun id => match val id with
    | some (.node n) => delN n
    | some (.rel r) => delR r
    | _ => false

theorem rowHasDeleted_iff (imported : List Nat) (val : Nat → Option (CreateBatch.Val Unit))
    (delN delR : Nat → Bool) :
    rowHasDeleted imported val delN delR = true ↔
      ∃ id ∈ imported, (∃ n, val id = some (.node n) ∧ delN n = true) ∨
        (∃ r, val id = some (.rel r) ∧ delR r = true) := by
  unfold rowHasDeleted
  rw [List.any_eq_true]
  apply exists_congr; intro id; apply and_congr_right; intro _
  rcases val id with _ | ((n | r) | w) <;> simp

/-- The output accumulator: `acc` batches (`acc_rows` rows in all) and the `overflow`. -/
structure Acc (ρ : Type) where
  acc : List (List ρ)
  accRows : Nat
  overflow : Option (List ρ)

/-- `push_acc` (`apply.rs`) with `BATCH_SIZE = cap`: append the batch if it fits, else
its first `room` rows; the rest waits in `overflow`. -/
def pushAcc {ρ : Type} (cap : Nat) (a : Acc ρ) (b : List ρ) : Acc ρ :=
  let room := cap - a.accRows
  if b.length ≤ room then { a with accRows := a.accRows + b.length, acc := a.acc ++ [b] }
  else { a with acc := if room > 0 then a.acc ++ [b.take room] else a.acc,
                accRows := if room > 0 then a.accRows + room else a.accRows,
                overflow := some (b.drop room) }

/-- `flush_acc`: the concatenation of `acc` (`Batch::concat`, or the single batch). -/
def flushAcc {ρ : Type} (a : Acc ρ) : List ρ × Acc ρ := (a.acc.flatten, { a with acc := [], accRows := 0 })

/-- `push_acc` loses and reorders no row, and keeps the output within `BATCH_SIZE`. -/
theorem pushAcc_spec {ρ : Type} (cap : Nat) (a : Acc ρ) (b : List ρ) (hc : a.accRows ≤ cap)
    (hr : a.accRows = a.acc.flatten.length) :
    let a' := pushAcc cap a b
    a'.acc.flatten ++ (if b.length ≤ cap - a.accRows then [] else (a'.overflow.getD [])) =
      a.acc.flatten ++ b ∧
    a'.accRows = a'.acc.flatten.length ∧ a'.accRows ≤ cap := by
  show (pushAcc cap a b).acc.flatten ++ _ = _ ∧ _
  unfold pushAcc
  by_cases h : b.length ≤ cap - a.accRows
  · simp only [if_pos h]
    refine ⟨by simp, by simp [hr], by omega⟩
  · simp only [if_neg h]
    by_cases hp : cap - a.accRows > 0
    · simp only [if_pos hp]
      refine ⟨by simp, ?_, by omega⟩
      simp only [List.flatten_append, List.flatten_singleton, List.length_append, List.length_take, ← hr]
      omega
    · simp only [if_neg hp]
      have h0 : cap - a.accRows = 0 := by omega
      refine ⟨by simp [h0], hr, hc⟩

theorem flushAcc_spec {ρ : Type} (a : Acc ρ) :
    (flushAcc a).1 = a.acc.flatten ∧ (flushAcc a).2.acc = [] ∧ (flushAcc a).2.accRows = 0 :=
  ⟨rfl, rfl, rfl⟩

/-! ## #2729 — CREATE → DELETE no longer depends on the batch split (#2968)

`CreateOp::next` processes ONE batch and passes it on; both it and `DeleteOp` write the
query's shared `pending`.  Before #2968 `DeleteOp` deleted each batch as it arrived, so the
create of batch `k+1` ran after the delete of batch `k` and `create_batch`'s endpoint check
(`create.rs`, `Delete.goneN`) rejected a node an earlier batch pending-deleted:
`UNWIND range(1,1100) AS x MATCH (n:A {id:1}) CREATE (n)-[:R]->(n) DETACH DELETE n` failed
(1024 + 76 rows, `BATCH_SIZE`) where C succeeds.  Since #2968 a Delete with a writing
operator below it is eager (`Delete.eagerOf`, `Delete.deleteEager`): the child (CREATE) is
drained first, then every batch is deleted, as C does. -/

/-- `create_batch`'s context with the endpoint check reading the live pending-delete
state (`create.rs` = `Delete.goneN`). Ids and attributes do not matter here. -/
def ctxOf (st : Delete.St) : CreateBatch.Ctx Unit Unit Unit where
  reserveN _ n := .ok (List.range n)
  reserveR log n := .ok (List.range' log.length n)
  evAttrs _ _ := .ok none
  valid _ := .ok ()
  gone _ a := Delete.goneN st a

/-- `value_at(var, row)` as `delete_batch` reads it (`Null` when absent). -/
def dvOf : Option (CreateBatch.Val Unit) → Delete.DV
  | some (.node n) => .node n
  | some (.rel n) => .rel n
  | some (.other _) => .other
  | none => .null

/-- `CreateOp::next` on one batch, pulled by the eager Delete: create against the pending
deletes from before the Delete runs (`st`), appending to the log. -/
def createStep (r : CreateBatch.PRel Unit) (st : Delete.St)
    (acc : List (CreateBatch.Ev Unit Unit) × List (List (CreateBatch.Row Unit)))
    (rows : List (CreateBatch.Row Unit)) :
    Except String (List (CreateBatch.Ev Unit Unit) × List (List (CreateBatch.Row Unit))) := do
  let (log', rows') ← CreateBatch.createRel (ctxOf st) false acc.1 rows r
  pure (log', acc.2 ++ [rows'])

/-- `delete_batch` on one replayed batch (`Delete.deleteBatch`). -/
def delStep (v : Nat) (st : Delete.St) (rows : List (CreateBatch.Row Unit)) : Except String Delete.St :=
  Delete.deleteBatch st (rows.map fun row => dvOf (row v)) []

/-- `Create(rel)` then an eager `Delete(var)` over a sequence of batches (#2968):
`Delete.deleteEager` over the stream `CreateOp` produces. -/
def pipeline (r : CreateBatch.PRel Unit) (v : Nat) (s : Delete.St × List (CreateBatch.Ev Unit Unit))
    (bs : List (List (CreateBatch.Row Unit))) :
    Except String (Delete.St × List (CreateBatch.Ev Unit Unit)) := do
  let (log', outs) ← bs.foldlM (createStep r s.1) (s.2, [])
  let (st', _) ← Delete.deleteEager (delStep v) s.1 (outs.map .ok)
  pure (st', log')

/-- **The property #2729 violated.**  A write pipeline's outcome depends only on the rows,
not on how they are cut into batches (what `CALL { … }` or `BATCH_SIZE` changes). -/
def PartitionInvariant {σ ρ : Type} (run : List (List ρ) → Except String σ) : Prop :=
  ∀ bs bs' : List (List ρ), bs.flatten = bs'.flatten → run bs = run bs'

/-! ### `delete_batch` on flat values is a function of the id sets -/

namespace DelClosed
open Delete

theorem St.ext' (a b : St) (h1 : ∀ j, a.pc j = b.pc j) (h2 : ∀ j, a.pd j = b.pd j)
    (h3 : ∀ j, a.gd j = b.gd j) (h4 : ∀ j, a.rc j = b.rc j) (h5 : ∀ j, a.rd j = b.rd j)
    (h6 : ∀ j, a.grd j = b.grd j) : a = b := by
  cases a; cases b
  simp only [St.mk.injEq]
  exact ⟨funext h1, funext h2, funext h3, funext h4, funext h5, funext h6⟩

/-- `delete_nodes_bulk`, closed form. -/
def nb (st : St) (ids : List Nat) : St :=
  { st with pd := fun j => st.pd j || (ids.contains j && !st.pc j && !st.gd j),
            pc := fun j => st.pc j && !(ids.contains j && !st.pd j),
            gd := fun j => st.gd j || (ids.contains j && !st.pd j && st.pc j) }

/-- `delete_relationships_bulk`, closed form. -/
def rb (st : St) (ids : List Nat) : St :=
  { st with rd := fun j => st.rd j || (ids.contains j && (st.rc j || !st.grd j)) }

theorem bulkScan_closed (ids : List Nat) (st : St) (com : List Nat) :
    (bulkScan ids (st, com)).2 = com ++ ids.filter (fun j => !st.pd j && !st.pc j && !st.gd j) ∧
    (∀ j, (bulkScan ids (st, com)).1.pd j = st.pd j) ∧
    (∀ j, (bulkScan ids (st, com)).1.pc j = (st.pc j && !(ids.contains j && !st.pd j))) ∧
    (∀ j, (bulkScan ids (st, com)).1.gd j = (st.gd j || (ids.contains j && !st.pd j && st.pc j))) ∧
    (bulkScan ids (st, com)).1.rc = st.rc ∧ (bulkScan ids (st, com)).1.rd = st.rd ∧
    (bulkScan ids (st, com)).1.grd = st.grd := by
  induction ids generalizing st com with
  | nil => simp [bulkScan]
  | cons id ids ih =>
    simp only [bulkScan]
    by_cases h1 : st.pd id
    · simp only [h1, ite_true]
      obtain ⟨a, b, c, d, e, f, g⟩ := ih st com
      refine ⟨by rw [a]; simp [List.filter_cons, h1], b, fun j => ?_, fun j => ?_, e, f, g⟩
      · rw [c j]; by_cases hj : j = id
        · subst hj; simp [h1]
        · simp [List.contains_cons, hj]
      · rw [d j]; by_cases hj : j = id
        · subst hj; simp [h1]
        · simp [List.contains_cons, hj]
    · by_cases h2 : st.pc id
      · simp only [h1, h2, Bool.false_eq_true, ite_false, ite_true]
        have hd : delNode st id = { st with pc := clrS st.pc id, gd := setS st.gd id } := by
          simp [delNode, h1, h2]
        rw [hd]
        obtain ⟨a, b, c, d, e, f, g⟩ := ih { st with pc := clrS st.pc id, gd := setS st.gd id } com
        refine ⟨?_, b, fun j => ?_, fun j => ?_, e, f, g⟩
        · rw [a]; simp only [List.filter_cons, h1, h2, Bool.not_false, Bool.not_true,
            Bool.true_and, Bool.false_and, Bool.false_eq_true, ite_false]
          congr 1; apply List.filter_congr; intro j _
          by_cases hj : j = id
          · subst hj; simp [clrS, setS, h2]
          · simp [clrS, setS, hj]
        · rw [c j]; by_cases hj : j = id
          · subst hj; simp [clrS, h1, h2]
          · simp [clrS, List.contains_cons, hj]
        · rw [d j]; by_cases hj : j = id
          · subst hj; simp [setS, h1, h2]
          · simp [setS, clrS, List.contains_cons, hj]
      · by_cases h3 : st.gd id
        · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_true]
          obtain ⟨a, b, c, d, e, f, g⟩ := ih st com
          refine ⟨by rw [a]; simp [List.filter_cons, h3], b, fun j => ?_, fun j => ?_, e, f, g⟩
          · rw [c j]; by_cases hj : j = id
            · subst hj; simp [h2]
            · simp [List.contains_cons, hj]
          · rw [d j]; by_cases hj : j = id
            · subst hj; simp [h3]
            · simp [List.contains_cons, hj]
        · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_false, ite_true]
          obtain ⟨a, b, c, d, e, f, g⟩ := ih st (com ++ [id])
          refine ⟨by rw [a]; simp [List.filter_cons, h1, h2, h3], b, fun j => ?_, fun j => ?_, e, f, g⟩
          · rw [c j]; by_cases hj : j = id
            · subst hj; simp [h2]
            · simp [List.contains_cons, hj]
          · rw [d j]; by_cases hj : j = id
            · subst hj; simp [h3, h2]
            · simp [List.contains_cons, hj]

theorem delNodesBulk_closed (st : St) (ids : List Nat) : delNodesBulk st ids = nb st ids := by
  obtain ⟨a, b, c, d, e, f, g⟩ := bulkScan_closed ids st []
  unfold delNodesBulk
  generalize hr : bulkScan ids (st, []) = r at a b c d e f g
  obtain ⟨st1, com⟩ := r
  simp only at a b c d e f g ⊢
  apply St.ext' <;> intro j <;> simp only [nb]
  · exact c j
  · rw [a, b j]; simp only [List.nil_append, List.contains_iff_mem, List.mem_filter]
    cases hm : decide (j ∈ ids) <;> simp_all <;> cases st.pd j <;> simp
  · exact d j
  · rw [e]
  · rw [f]
  · rw [g]

theorem relScan_mem (st : St) (ids seen pcr com : List Nat) (j : Nat) :
    (j ∈ (relScan st ids seen (pcr, com)).1 ∨ j ∈ (relScan st ids seen (pcr, com)).2) ↔
      (j ∈ pcr ∨ j ∈ com) ∨
      (j ∈ ids ∧ j ∉ seen ∧ (!st.rd j && (st.rc j || !st.grd j)) = true) := by
  induction ids generalizing seen pcr com with
  | nil => simp [relScan]
  | cons id ids ih =>
    simp only [relScan]
    by_cases h0 : seen.contains id
    · simp only [h0, ite_true, ih]
      have : id ∈ seen := List.contains_iff_mem.mp h0
      constructor
      · rintro (h | ⟨h1, h2, h3⟩)
        · exact Or.inl h
        · exact Or.inr ⟨List.mem_cons_of_mem _ h1, h2, h3⟩
      · rintro (h | ⟨h1, h2, h3⟩)
        · exact Or.inl h
        · rcases List.mem_cons.mp h1 with rfl | h1
          · exact absurd this h2
          · exact Or.inr ⟨h1, h2, h3⟩
    · simp only [h0, Bool.false_eq_true, ite_false]
      have hs : id ∉ seen := fun h => h0 (List.contains_iff_mem.mpr h)
      by_cases h1 : st.rd id
      · simp only [h1, ite_true, ih]
        by_cases hj : j = id
        · subst hj; simp [h1, hs]
        · simp [hj]
      · by_cases h2 : st.rc id
        · simp only [h1, h2, Bool.false_eq_true, ite_false, ite_true, ih]
          by_cases hj : j = id
          · subst hj; simp [h1, h2, hs]
          · simp [hj]
        · by_cases h3 : st.grd id
          · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_true, ih]
            by_cases hj : j = id
            · subst hj; simp [h1, h2, h3, hs]
            · simp [hj]
          · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_false, ite_true, ih]
            by_cases hj : j = id
            · subst hj; simp [h1, h2, h3, hs]
            · simp [hj]

theorem foldl_delRel_closed (ps : List Nat) (st : St) :
    ps.foldl delRel st = { st with rd := fun j => st.rd j || (ps.contains j && (st.rc j || !st.grd j)) } := by
  induction ps generalizing st with
  | nil => apply St.ext' <;> intro j <;> simp
  | cons p ps ih =>
    simp only [List.foldl_cons, ih]
    have k := delRel_keep st p 0
    apply St.ext' <;> intro j
    · simp [delRel]; split <;> (try split) <;> rfl
    · simp [delRel]; split <;> (try split) <;> rfl
    · simp [delRel]; split <;> (try split) <;> rfl
    · simp only [k.2.2]
    · simp only [k.2.1, k.2.2, List.contains_cons]
      unfold delRel
      cases hjp : (j == p)
      · have hj : j ≠ p := by simpa using hjp
        split
        · simp
        · split
          · simp [setS, hj]
          · simp
      · have hj : j = p := by simpa using hjp
        subst hj
        split
        · next h => simp [h]
        · split
          · simp only [setS]; cases hc : st.rc j <;> cases hg : st.grd j <;> simp_all
          · next h1 h2 => cases hr : st.rd j <;> cases hc : st.rc j <;> cases hg : st.grd j <;> simp_all
    · simp only [k.2.1]

theorem delRelsBulk_closed (st : St) (rels : List Nat) : delRelsBulk st rels = rb st rels := by
  unfold delRelsBulk
  have hm := relScan_mem st rels [] [] []
  generalize relScan st rels [] ([], []) = r at hm
  obtain ⟨pcr, com⟩ := r
  simp only [foldl_delRel_closed]
  apply St.ext' <;> intro j <;> simp only [rb]
  have hj := hm j
  clear hm
  simp only [List.not_mem_nil, or_self, false_or, not_false_eq_true, true_and] at hj
  by_cases hr : st.rd j
  · simp [hr]
  · simp only [hr, Bool.false_or, Bool.not_false, Bool.true_and] at hj ⊢
    simp only [← List.contains_iff_mem] at hj
    cases hp : pcr.contains j <;> cases hc : com.contains j <;> cases hl : rels.contains j <;>
      cases hq : (st.rc j || !st.grd j) <;> simp_all

theorem nb_nil (st : St) : nb st [] = st := by
  apply St.ext' <;> intro j <;> simp [nb]

theorem rb_nil (st : St) : rb st [] = st := by
  apply St.ext' <;> intro j <;> simp [rb]

theorem nb_append (st : St) (a b : List Nat) : nb (nb st a) b = nb st (a ++ b) := by
  apply St.ext' <;> intro j <;> simp only [nb, List.contains_append] <;>
    cases a.contains j <;> cases b.contains j <;> cases st.pd j <;> cases st.pc j <;>
    cases st.gd j <;> rfl

theorem rb_append (st : St) (a b : List Nat) : rb (rb st a) b = rb st (a ++ b) := by
  apply St.ext' <;> intro j <;> simp only [rb, List.contains_append] <;>
    cases a.contains j <;> cases b.contains j <;> cases st.rd j <;> cases st.rc j <;>
    cases st.grd j <;> rfl

theorem rb_nb (st : St) (a b : List Nat) : rb (nb st a) b = nb (rb st b) a := by
  apply St.ext' <;> intro j <;> rfl

/-- A `DV` with no path (all `dvOf` produces). -/
def flat : DV → Bool
  | .path _ => false
  | _ => true

def isOther : DV → Bool
  | .other => true
  | _ => false

def nodesOf : List DV → List Nat
  | [] => []
  | .node n :: vs => n :: nodesOf vs
  | _ :: vs => nodesOf vs

def relsOf : List DV → List Nat
  | [] => []
  | .rel n :: vs => n :: relsOf vs
  | _ :: vs => relsOf vs

theorem nodesOf_append (a b : List DV) : nodesOf (a ++ b) = nodesOf a ++ nodesOf b := by
  induction a with
  | nil => rfl
  | cons v vs ih => cases v <;> simp [nodesOf, ih]

theorem relsOf_append (a b : List DV) : relsOf (a ++ b) = relsOf a ++ relsOf b := by
  induction a with
  | nil => rfl
  | cons v vs ih => cases v <;> simp [relsOf, ih]

theorem scan_flat (st : St) (vs : List DV) (h : ∀ v ∈ vs, flat v = true) :
    scan st vs = if vs.any isOther then .error mismatch else .ok (st, nodesOf vs, relsOf vs) := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    have ih := ih (fun w hw => h w (List.mem_cons_of_mem _ hw))
    have hv := h v (List.mem_cons_self ..)
    cases v with
    | node n =>
      simp only [scan, ih, List.any_cons, isOther, Bool.false_or, nodesOf]
      split <;> rfl
    | rel n =>
      simp only [scan, ih, List.any_cons, isOther, Bool.false_or, relsOf]
      split <;> rfl
    | path _ => simp [flat] at hv
    | null =>
      simp only [scan, delEntity, List.any_cons, isOther, Bool.false_or, nodesOf, relsOf]
      exact ih
    | other => simp [scan, delEntity, isOther]; rfl

/-- `delete_batch` over flat values, closed form. -/
def dB (st : St) (vs : List DV) : Except String St :=
  if vs.any isOther then .error mismatch else .ok (rb (nb st (nodesOf vs)) (relsOf vs))

theorem deleteBatch_flat (st : St) (vs : List DV) (h : ∀ v ∈ vs, flat v = true) :
    deleteBatch st vs [] = dB st vs := by
  unfold deleteBatch dB
  rw [scan_flat st vs h]
  split
  · rfl
  · simp only [bind, Except.bind, List.foldlM_nil]
    show Except.ok _ = _
    congr 1
    by_cases hn : (nodesOf vs).isEmpty <;> by_cases hr : (relsOf vs).isEmpty <;>
      simp only [hn, hr, ite_true, ite_false, Bool.false_eq_true, delNodesBulk_closed,
        delRelsBulk_closed] <;>
      simp only [List.isEmpty_iff] at hn hr <;> simp only [hn, hr, nb_nil, rb_nil]

theorem dB_append (st : St) (a b : List DV) :
    dB st (a ++ b) = (dB st a).bind (fun st' => dB st' b) := by
  unfold dB
  simp only [List.any_append, nodesOf_append, relsOf_append]
  cases ha : a.any isOther
  · cases hb : b.any isOther
    · simp only [Bool.false_or, Bool.false_eq_true, ite_false, Except.bind]
      congr 1
      apply St.ext' <;> intro j <;> simp only [nb, rb, List.contains_append] <;>
        cases (nodesOf a).contains j <;> cases (nodesOf b).contains j <;>
        cases (relsOf a).contains j <;> cases (relsOf b).contains j <;> cases st.pd j <;>
        cases st.pc j <;> cases st.gd j <;> cases st.rd j <;> cases st.rc j <;> cases st.grd j <;> rfl
    · simp [Except.bind]
  · simp [Except.bind]

/-- **Eager DELETE is batch-split invariant**: deleting every batch in order equals one
`delete_batch` over all rows. -/
theorem foldlM_dB (st : St) (bs : List (List DV)) :
    bs.foldlM dB st = dB st bs.flatten := by
  induction bs generalizing st with
  | nil => simp [dB, nodesOf, relsOf, nb_nil, rb_nil]; rfl
  | cons b bs ih =>
    simp only [List.foldlM_cons, List.flatten_cons, dB_append]
    cases dB st b with
    | error e => rfl
    | ok st' => exact ih st'

end DelClosed

/-! ### `create_batch` over a relationship pattern is a per-row fold -/

namespace CrClosed
open CreateBatch

/-- The per-row endpoint check of `createRel` against fixed pending deletes. -/
def chk (st : Delete.St) (r : PRel Unit) (row : Row Unit) : Except String (Nat × Nat) := do
  let (a, b) ← endpoint row r.src r.dst
  if !false && (Delete.goneN st a || Delete.goneN st b) then
    throw "Failed to create relationship; endpoint was not found."
  pure (a, b)

/-- `createRel (ctxOf st)` row by row: check, reserve the next id, log, bind. -/
def crRows (st : Delete.St) (r : PRel Unit) :
    List (Ev Unit Unit) → List (Row Unit) → Except String (List (Ev Unit Unit) × List (Row Unit))
  | log, [] => .ok (log, [])
  | log, row :: rows =>
    match chk st r row with
    | .error e => .error e
    | .ok (a, b) =>
      (crRows st r (log ++ [.createdRel log.length a b r.ty]) rows).map
        fun q => (q.1, upd row r.alias (.rel log.length) :: q.2)

theorem mapM_none (rows : List (Row Unit)) :
    rows.mapM (fun _ => (Except.ok none : Except String (Option (List (Nat × Val Unit))))) =
      .ok (rows.map fun _ => none) := by
  induction rows with
  | nil => rfl
  | cons x xs ih => simp only [List.mapM_cons, ih, List.map_cons]; rfl

theorem filterMap_none {α β γ : Type} (ids : List Nat) (l : List α) (f : Nat → γ → β) :
    ((ids.zip (l.map fun _ => (none : Option γ))).filterMap fun p => p.2.map (f p.1)) = [] := by
  induction ids generalizing l with
  | nil => rfl
  | cons i is ih => cases l with
    | nil => rfl
    | cons x xs => simpa [List.filterMap_cons] using ih xs

def cf (log : List (Ev Unit Unit)) (rows : List (Row Unit)) (r : PRel Unit) (eps : List (Nat × Nat)) :
    List (Ev Unit Unit) × List (Row Unit) :=
  (log ++ ((List.range' log.length eps.length).zip eps).map (fun p => .createdRel p.1 p.2.1 p.2.2 r.ty),
   (rows.zip (List.range' log.length eps.length)).map fun p => upd p.1 r.alias (.rel p.2))

theorem createRel_cf (st : Delete.St) (log : List (Ev Unit Unit)) (rows : List (Row Unit)) (r : PRel Unit) :
    createRel (ctxOf st) false log rows r = (rows.mapM (chk st r)).map (cf log rows r) := by
  unfold createRel
  show (do let eps ← rows.mapM (chk st r); _) = _
  cases rows.mapM (chk st r) with
  | error e => rfl
  | ok eps =>
    simp only [ctxOf, bind, Except.bind, mapM_none, Except.map, cf]
    rw [filterMap_none (f := fun i a => (i, a))]
    rfl

theorem createRel_eq (st : Delete.St) (r : PRel Unit) (rows : List (Row Unit)) (log : List (Ev Unit Unit)) :
    createRel (ctxOf st) false log rows r = crRows st r log rows := by
  rw [createRel_cf]
  induction rows generalizing log with
  | nil => simp [cf, Except.map, List.mapM_nil, pure, Except.pure, crRows]
  | cons row rows ih =>
    rw [List.mapM_cons]
    simp only [crRows]
    cases chk st r row with
    | error e => rfl
    | ok ab =>
      obtain ⟨a, b⟩ := ab
      show _ = Except.map _ (crRows st r _ rows)
      rw [← ih]
      cases hm : rows.mapM (chk st r) with
      | error e => simp [bind, Except.bind, pure, Except.pure, Except.map]
      | ok eps =>
        simp only [bind, Except.bind, pure, Except.pure, Except.map, cf, List.length_cons,
          List.range'_succ, List.zip_cons_cons, List.map_cons, List.length_append, List.length_singleton,
          List.append_assoc, List.singleton_append, List.length_nil, Nat.zero_add]

theorem crRows_append (st : Delete.St) (r : PRel Unit) (xs ys : List (Row Unit)) (log : List (Ev Unit Unit)) :
    crRows st r log (xs ++ ys) =
      (crRows st r log xs).bind fun p => (crRows st r p.1 ys).map fun q => (q.1, p.2 ++ q.2) := by
  induction xs generalizing log with
  | nil =>
    simp only [List.nil_append, crRows, Except.bind]
    cases crRows st r log ys <;> rfl
  | cons x xs ih =>
    simp only [List.cons_append, crRows]
    cases chk st r x with
    | error e => rfl
    | ok ab =>
      simp only [ih]
      cases crRows st r _ xs with
      | error e => rfl
      | ok p =>
        simp only [Except.bind, Except.map]
        cases crRows st r p.1 ys <;> rfl

end CrClosed

theorem createStep_fold (r : CreateBatch.PRel Unit) (st : Delete.St) (bs : List (List (CreateBatch.Row Unit)))
    (log : List (CreateBatch.Ev Unit Unit)) (outs : List (List (CreateBatch.Row Unit))) :
    (bs.foldlM (createStep r st) (log, outs)).map (fun p => (p.1, p.2.flatten)) =
      (CrClosed.crRows st r log bs.flatten).map (fun p => (p.1, outs.flatten ++ p.2)) := by
  induction bs generalizing log outs with
  | nil =>
    simp [CrClosed.crRows, Except.map, pure, Except.pure]
  | cons b bs ih =>
    simp only [List.foldlM_cons, List.flatten_cons, CrClosed.crRows_append, createStep,
      CrClosed.createRel_eq]
    cases CrClosed.crRows st r log b with
    | error e => rfl
    | ok p =>
      obtain ⟨l1, o1⟩ := p
      show (List.foldlM (createStep r st) (l1, outs ++ [o1]) bs).map _ = _
      rw [ih]
      simp only [Except.bind, Except.map]
      cases CrClosed.crRows st r l1 bs.flatten <;> simp [List.flatten_append]

theorem dvOf_flat (x : Option (CreateBatch.Val Unit)) : DelClosed.flat (dvOf x) = true := by
  rcases x with _ | ((_ | _) | _) <;> rfl

theorem delStep_fold (v : Nat) (st : Delete.St) (outs : List (List (CreateBatch.Row Unit))) :
    outs.foldlM (delStep v) st = DelClosed.dB st (outs.flatten.map fun row => dvOf (row v)) := by
  induction outs generalizing st with
  | nil => simp [DelClosed.dB, DelClosed.nodesOf, DelClosed.relsOf, DelClosed.nb_nil, DelClosed.rb_nil]; rfl
  | cons b bs ih =>
    simp only [List.foldlM_cons, List.flatten_cons, List.map_append, DelClosed.dB_append, delStep]
    rw [DelClosed.deleteBatch_flat _ _ (fun w hw => by
      obtain ⟨row, _, rfl⟩ := List.mem_map.mp hw; exact dvOf_flat _)]
    show (DelClosed.dB st _).bind _ = _
    cases DelClosed.dB st (b.map fun row => dvOf (row v)) with
    | error e => rfl
    | ok st' => exact ih st'

/-- The whole eager pipeline is a function of the flattened rows. -/
theorem pipeline_flat (r : CreateBatch.PRel Unit) (v : Nat) (s : Delete.St × List (CreateBatch.Ev Unit Unit))
    (bs : List (List (CreateBatch.Row Unit))) :
    pipeline r v s bs =
      (CrClosed.crRows s.1 r s.2 bs.flatten).bind fun p =>
        (DelClosed.dB s.1 (p.2.map fun row => dvOf (row v))).map (·, p.1) := by
  have h := createStep_fold r s.1 bs s.2 []
  simp only [List.flatten_nil, List.nil_append] at h
  unfold pipeline
  revert h
  cases bs.foldlM (createStep r s.1) (s.2, []) with
  | error e =>
    intro h
    cases hc : CrClosed.crRows s.1 r s.2 bs.flatten with
    | error e' => rw [hc] at h; cases h; rfl
    | ok p => rw [hc] at h; cases h
  | ok q =>
    intro h
    cases hc : CrClosed.crRows s.1 r s.2 bs.flatten with
    | error e' => rw [hc] at h; cases h
    | ok p =>
      rw [hc] at h
      obtain ⟨l, outs⟩ := q
      simp only [Except.map, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, hf⟩ := h
      simp only [bind, Except.bind, Delete.deleteEager_spec, delStep_fold, hf, Except.map]
      cases DelClosed.dB s.1 (p.2.map fun row => dvOf (row v)) <;> rfl

/-- **#2729 fixed**: CREATE then an eager DELETE gives the same result (pending deletes and
event log, or the same error) however the rows are split into batches. -/
theorem pipeline_partitionInvariant (r : CreateBatch.PRel Unit) (v : Nat)
    (s : Delete.St × List (CreateBatch.Ev Unit Unit)) : PartitionInvariant (pipeline r v s) := by
  intro bs bs' h
  rw [pipeline_flat, pipeline_flat, h]

/-- The node `n` (id 0) in variable 0 of every row; `CREATE (n)-[:R]->(n)` aliases the new
relationship as variable 1. -/
def row15 : CreateBatch.Row Unit := fun j => if j = 0 then some (.node 0) else none
def selfLoop : CreateBatch.PRel Unit := { alias := 1, ty := "R", src := 0, dst := 0, attrs := () }
def st0 : Delete.St := { pc := fun _ => false, pd := fun _ => false, gd := fun _ => false,
                         rc := fun _ => false, rd := fun _ => false, grd := fun _ => false }

/-- #2729's rows, split over two batches (as 1024 + 76 rows reach CREATE), now succeed and
pending-delete `n`, exactly like one batch. -/
theorem fixed2729_two_batches :
    (match pipeline selfLoop 0 (st0, []) [[row15], [row15]] with
      | .ok (st, log) => (st.pd 0, log.length)
      | .error _ => (false, 0)) = (true, 2) := by
  rw [pipeline_flat]; decide

/-! ## #3164 — per-row commits recycle ids under stale row references

`UNWIND [1,2] AS x MATCH (n:A) CALL { WITH n DETACH DELETE n RETURN 1 AS k UNION
CREATE (:B) RETURN 2 AS k } RETURN x, id(n), k` on `(:A),(:C)`.  `Union`/`Distinct`
force per-row mode; each branch has its own `Commit` (live `GRAPH.EXPLAIN`).  Row 1
deletes A (id 0) and commits, releasing id 0; its CREATE branch reclaims id 0 for a
`:B`.  Row 2 still holds `Value::Node(0)` and its DETACH DELETE removes that `:B`.

Graph model: live nodes as `(id, serial)`.  The serial is the entity's identity —
never reused — which Rust does not store: a row value is just the id. -/

structure NG where
  live : List (Nat × Nat)
  serial : Nat
  /-- entities removed so far in this query (the `Nodes deleted` statistic, by entity) -/
  removed : List (Nat × Nat)
  deriving DecidableEq

/-- `IdSpace::reserve` reclaims a released id before extending the space
(`id_space.rs`): the smallest id no live node holds. -/
def freshId (g : NG) : Nat :=
  ((List.range (g.live.length + 1)).find? (fun i => !(g.live.map Prod.fst).contains i)).getD
    g.live.length

/-- The pending writes a row queues. -/
inductive Op where
  /-- Rust `DETACH DELETE n` with `n = Value::Node(id)`: whatever node now has that id. -/
  | delId (id : Nat)
  /-- Reference `DETACH DELETE n`: the entity the row was bound to (a no-op if gone). -/
  | delEnt (e : Nat × Nat)
  | create
  deriving DecidableEq

def applyOp (g : NG) : Op → NG
  | .delId id => { g with live := g.live.filter (·.1 != id),
                          removed := g.removed ++ g.live.filter (·.1 == id) }
  | .delEnt e => { g with live := g.live.filter (· != e),
                          removed := g.removed ++ g.live.filter (· == e) }
  | .create => { g with live := g.live ++ [(freshId g, g.serial)], serial := g.serial + 1 }

/-- `Pending::commit` (`pending.rs`; deletes via `Graph::delete_nodes`, `graph.rs`,
which releases the id).  The Union's two branch Commits run in branch order
(`union.rs`), so the row's ops apply in sequence. -/
def commitOps (g : NG) (ws : List Op) : NG := ws.foldl applyOp g

/-- Rust: rows carry the node id; k=1 is the DELETE branch, k=2 the CREATE branch. -/
def rustSub : WSub NG Nat (Nat × Nat) Op where
  read _ n := [(n, 1), (n, 2)]
  writes | (n, 1) => [.delId n] | _ => [.create]
  commit := commitOps

/-- Reference: rows carry the bound entity. -/
def refSub : WSub NG (Nat × Nat) ((Nat × Nat) × Nat) Op where
  read _ e := [(e, 1), (e, 2)]
  writes | (e, 1) => [.delEnt e] | _ => [.create]
  commit := commitOps

/-- Binding an id to the entity it names in the graph the row was produced from. -/
def bindIn (g : NG) (id : Nat) : Nat × Nat :=
  ((g.live.find? (·.1 == id)).getD (id, 0))

/-- Rust's per-row Apply (`next_per_row`, `apply.rs`): one commit per row; since #2968 a row
whose imported node id is free (`row_has_deleted_entity` = `is_node_deleted` =
`node_ids.is_free`) is skipped.  A reclaimed id is not free, so row 2 still runs. -/
def rustPerRow (g : NG) (ids : List Nat) : NG :=
  (applyPerRowW rustSub (fun g n => !(g.live.map Prod.fst).contains n) none g ids).2

def refPerRow (g : NG) (ids : List Nat) : NG := (applyRefW refSub g (ids.map (bindIn g))).2

/-- **The property #3164 violates.**  When every delete in the subquery goes through an
outer row's reference, the query removes only entities those rows were bound to in the
graph they were produced from: a reference never resolves to an entity created later
that reused a released id. -/
def NoStaleAlias (run : NG → List Nat → NG) : Prop :=
  ∀ g0 ids, ∀ e ∈ (run g0 ids).removed, e ∈ g0.removed ∨ (e ∈ g0.live ∧ e.1 ∈ ids)

/-- `(:A {id:1})` = id 0, `(:C)` = id 1. -/
def g3164 : NG := { live := [(0, 0), (1, 1)], serial := 2, removed := [] }

/-- **#3164**: Rust deletes 2 nodes (A, then the `:B` that reclaimed id 0) and ends with
one `:B`; the reference deletes 1 and ends with two `:B` at ids 0 and 2.  Live on main
9d3869aa8: Rust `Nodes deleted: 2`, final ids `0:[B] 1:[C]`; C `Nodes deleted: 1`, final
ids `0:[B] 1:[C] 2:[B]`.  (C also drops the outer row `x=2,k=1` whose node is gone;
rows are not modelled here, only effects.) -/
theorem bug3164_stale_ref_deletes_recycled_id :
    rustPerRow g3164 [0, 0] = { live := [(1, 1), (0, 3)], serial := 4, removed := [(0, 0), (0, 2)] } ∧
    refPerRow g3164 [0, 0] = { live := [(1, 1), (0, 2), (2, 3)], serial := 4, removed := [(0, 0)] } ∧
    ¬ NoStaleAlias rustPerRow := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h g3164 [0, 0] (0, 2) (by decide)
  revert this; decide

end RowWrites
