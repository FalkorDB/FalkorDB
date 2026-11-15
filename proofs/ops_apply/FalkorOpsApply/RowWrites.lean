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

Three Rust behaviours break that, each from one modelled mechanism:

| issue | mechanism (fb3d516d0) | here |
| --- | --- | --- |
| #2738 | `ApplyOp::next_batched` (`apply.rs:152`) runs ONE sub-plan over all rows of the input batch; its `Commit` (`commit.rs:69`) drains every row before `pending.commit()`, so every row reads the pre-batch graph.  `can_batch` (`apply.rs:119-133`) does not blacklist `Create`/`Delete`/`Set`/`Commit`. | `applyBatchedW`, `canBatchApply_admits_write`, `bug2738_batched_call_hides_writes` |
| #2729 | `CreateOp::next` (`create.rs:121`) and `DeleteOp::next` (`delete.rs:115`) stream one batch at a time against the shared `pending`; `create_batch`'s endpoint check (`create.rs:259-266`) reads pending deletes made by `delete_batch` for EARLIER batches.  Effects depend on how rows are cut into batches, and the plan inside `CALL {}` cuts them differently from top level. | `pipeline`, `PartitionInvariant`, `bug2729_batch_split_flips_endpoint_check` |
| #3164 | `next_per_row` (`apply.rs:238`) commits each row's sub-plan (each `Union` branch has its own `Commit`, `union.rs:65-96`); the commit releases a deleted node's id (`Graph::delete_nodes`, `graph.rs:2216`) and the next `create_batch` reclaims it (`IdSpace::reserve`, `id_space.rs:483-498`, via `create.rs:158`).  A row holds only `Value::Node(NodeId)` (`value.rs:220`), so a later row's stale reference resolves to the new entity. | `NG`, `rustPerRow`, `NoStaleAlias`, `bug3164_stale_ref_deletes_recycled_id` |

The sufficient condition under which batching IS sound for a writing sub-plan
(`applyBatchedW_eq_ref_of_readIndep`) is stated too, so a fix can prove itself
against `WriteSeparable`.
-/
namespace RowWrites

open Correlated

/-! ## A writing sub-plan, and Apply over it -/

/-- A correlated sub-plan that reads the graph and queues writes.
* `read g x` — the rows the sub-plan's reading operators (scans, traversals, filters,
  projections: the row-local part, `rowLocal`) produce for argument row `x` against
  the committed graph `g`;
* `writes y` — what `Create`/`Delete`/`Set` queue in `pending` for output row `y`;
* `commit g ws` — `CommitOp::next` → `Pending::commit` (`commit.rs:69`, `pending.rs:1228`). -/
structure WSub (G α β W : Type) where
  read : G → α → List β
  writes : β → List W
  commit : G → List W → G

/-- Batched Apply over a writing sub-plan (`apply.rs:152-232` + `commit.rs:69-90`): ONE
sub-plan instance gets the whole argument batch; its reads all see `g`; the blocking
`Commit` drains every row's writes, then commits them at once. -/
def applyBatchedW {G α β W : Type} (s : WSub G α β W) (g : G) (xs : List α) :
    List (α × β) × G :=
  let outs := xs.flatMap (fun x => (s.read g x).map (fun y => (x, y)))
  (outs, s.commit g (outs.flatMap (fun p => s.writes p.2)))

/-- Reference (openCypher; C's CALL {} with eager write ops): the subquery runs to
completion, commit included, for each outer row in turn, threading the graph.  This is
also exactly Rust's per-row mode (`next_per_row`, `apply.rs:238-337`: one sub-plan
instance, hence one `Commit`, per row). -/
def applyRefW {G α β W : Type} (s : WSub G α β W) : G → List α → List (α × β) × G
  | g, [] => ([], g)
  | g, x :: xs =>
    let ys := (s.read g x).map (fun y => (x, y))
    let g1 := s.commit g (ys.flatMap (fun p => s.writes p.2))
    let r := applyRefW s g1 xs
    (ys ++ r.1, r.2)

/-- **The property #2738 violates.**  Batched Apply over a writing sub-plan returns the
same rows and the same final graph as the per-row reference, for every graph and every
input batch.  A sound `can_batch` must only admit sub-plans with this property. -/
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

/-! ## #2738 — a writing CALL subquery is batched

`UNWIND [1,2,3] AS x CALL { MATCH (a:N) CREATE (:N) RETURN 1 AS one } RETURN count(*)`
on a graph with one `:N`.  Plan (live `GRAPH.EXPLAIN`): `Apply(Unwind, Project(Commit(
Create(NodeByLabelScan(Project(Argument))))))` — no blacklisted operator, so batched.
State = the number of `:N` nodes. -/

/-- The `can_batch` guard (`apply.rs:119-133`) admits the writing sub-plan above. -/
theorem canBatchApply_admits_write :
    canBatchApply [.project, .commit, .create, .scan] = true ∧
    canBatchApply [.project, .commit, .delete, .project] = true ∧
    canBatchApply [.project, .commit, .setProps, .scan] = true := by decide

/-- `MATCH (a:N) CREATE (:N)`: one row per `:N` in the committed graph, one node created
per row. -/
def matchCreateN : WSub Nat Nat Unit Unit where
  read g _ := List.replicate g ()
  writes _ := [()]
  commit g ws := g + ws.length

/-- **#2738**: batched Apply returns 3 rows and leaves 4 nodes; per-row (C, openCypher)
returns 1+2+4 = 7 rows and leaves 8.  Live on main 9d3869aa8: Rust `count(*)=3`, 4 `:N`;
C `count(*)=7`, 8 `:N`. -/
theorem bug2738_batched_call_hides_writes :
    ((applyBatchedW matchCreateN 1 [1, 2, 3]).1.length, (applyBatchedW matchCreateN 1 [1, 2, 3]).2)
      = (3, 4) ∧
    ((applyRefW matchCreateN 1 [1, 2, 3]).1.length, (applyRefW matchCreateN 1 [1, 2, 3]).2)
      = (7, 8) ∧
    ¬ WriteSeparable matchCreateN := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := congrArg Prod.snd (h 1 [1, 2, 3])
  revert this; decide

/-! ## #2729 — streaming CREATE → DELETE depends on the batch split

`CreateOp::next` and `DeleteOp::next` each process ONE batch and pass it on; both write
the query's shared `pending`.  So for `CREATE (n0)-[:R]->(n0) DETACH DELETE n0` the
create of batch `k+1` runs after the delete of batch `k`, and `create_batch`'s
endpoint check (`create.rs:259-266`, `Delete.goneN`) rejects a node an earlier batch
pending-deleted.  C's Create and Delete are eager (all records, then the next op), so
for C the split never matters.  The CALL {} form of #2729 plans the MATCH differently
(`All Node Scan → var-len traverse → Filter` instead of `Filter(All Node Scan) →
var-len traverse`), which hands the same 25 rows to `Create` in more than one batch.
The same mechanism, no CALL: `UNWIND range(1,1100) AS x MATCH (n:A {id:1})
CREATE (n)-[:R]->(n) DETACH DELETE n` errors on Rust (1024 + 76 rows, `BATCH_SIZE`,
`batch.rs:81`) and succeeds with 1000 rows; C succeeds for both. -/

/-- `create_batch`'s context with the endpoint check reading the live pending-delete
state (`create.rs:259-262` = `Delete.goneN`). Ids and attributes do not matter here. -/
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

/-- One batch through `Create(rel)` then `Delete(var)` (`create.rs:121-136`,
`delete.rs:115-127`): create against the current pending deletes, then delete. -/
def step (r : CreateBatch.PRel Unit) (v : Nat) (s : Delete.St × List (CreateBatch.Ev Unit Unit))
    (rows : List (CreateBatch.Row Unit)) :
    Except String (Delete.St × List (CreateBatch.Ev Unit Unit)) := do
  let (log', rows') ← CreateBatch.createRel (ctxOf s.1) false s.2 rows r
  let st' ← Delete.deleteBatch s.1 (rows'.map fun row => dvOf (row v)) []
  pure (st', log')

/-- The streaming pipeline over a sequence of batches. -/
def pipeline (r : CreateBatch.PRel Unit) (v : Nat) (s : Delete.St × List (CreateBatch.Ev Unit Unit))
    (bs : List (List (CreateBatch.Row Unit))) :
    Except String (Delete.St × List (CreateBatch.Ev Unit Unit)) :=
  bs.foldlM (step r v) s

/-- **The property #2729 violates.**  A write pipeline's outcome depends only on the rows,
not on how they are cut into batches (what `CALL { … }` or `BATCH_SIZE` changes).  A fix
(eager Create/Delete, or deferring the endpoint check to the whole stream) proves this. -/
def PartitionInvariant {σ ρ : Type} (run : List (List ρ) → Except String σ) : Prop :=
  ∀ bs bs' : List (List ρ), bs.flatten = bs'.flatten → run bs = run bs'

/-- `Except.toBool` is a function of the result, so `PartitionInvariant` gives equal
success/failure (used to refute it from a decidable fact). -/
theorem PartitionInvariant.toBool {σ ρ : Type} {run : List (List ρ) → Except String σ}
    (h : PartitionInvariant run) (bs bs' : List (List ρ)) (e : bs.flatten = bs'.flatten) :
    (run bs).toBool = (run bs').toBool := by rw [h bs bs' e]

def errMsg {σ : Type} : Except String σ → Option String
  | .error e => some e
  | .ok _ => none

/-- `(n0)-[:rt8]->(n0)` with `n0` in slot 0, bound before the CREATE (not created by it). -/
def rt8 : CreateBatch.PRel Unit := { alias := 1, ty := "rt8", src := 0, dst := 0, attrs := () }

/-- A row with `n0` = node 15 (live: `n0: (15)` in all 25 rows of #2729). -/
def row15 : CreateBatch.Row Unit := fun j => if j = 0 then some (.node 15) else none

def st0 : Delete.St := { pc := fun _ => false, pd := fun _ => false, gd := fun _ => false,
                         rc := fun _ => false, rd := fun _ => false, grd := fun _ => false }

/-- **#2729**: two rows holding the same `n0`.  One batch (top level): both
relationships are created, then `n0` is deleted — success.  Two batches (inside
`CALL {}`): the second batch's create sees the first batch's pending delete and fails
with `endpoint was not found`.  Live: Rust LEFT succeeds, RIGHT fails with exactly this
message; C succeeds on both (25 rows, 25 relationships created). -/
theorem bug2729_batch_split_flips_endpoint_check :
    (pipeline rt8 0 (st0, []) [[row15, row15]]).toBool = true ∧
    errMsg (pipeline rt8 0 (st0, []) [[row15], [row15]])
      = some "Failed to create relationship; endpoint was not found." ∧
    ¬ PartitionInvariant (pipeline rt8 0 (st0, [])) := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h.toBool [[row15, row15]] [[row15], [row15]] rfl
  revert this; decide

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
(`id_space.rs:483-498`): the smallest id no live node holds. -/
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

/-- `Pending::commit` (`pending.rs:1228`; deletes via `Graph::delete_nodes`, `graph.rs:2216`,
which releases the id).  The Union's two branch Commits run in branch order
(`union.rs:65-96`), so the row's ops apply in sequence. -/
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

/-- Rust's per-row Apply (`next_per_row`, `apply.rs:238-337`): one commit per row. -/
def rustPerRow (g : NG) (ids : List Nat) : NG := (applyRefW rustSub g ids).2

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
