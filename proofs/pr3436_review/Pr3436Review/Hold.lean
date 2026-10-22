/-
# e30861ea3: one borrow held across a call = main's sequential borrows

e30861ea3 changed the commit and effects functions from taking the cell
(`&AtomicRefCell<Graph>`) and borrowing inside to taking the graph; the caller holds one
`graph_mut()` / `graph()` across the call (cites: PR branch @ e30861ea3, "main" =
origin/main b97da5a1c/b84473a7f, identical `.rs`):

| function | main's borrows (inside) | e30861ea3 (caller holds) | model |
| --- | --- | --- | --- |
| `Pending::commit` (`pending.rs:1226-1367`) | `borrow_mut` per mutation group, each *conditional* on non-empty pending state (`create_nodes`, rel creation, `set_nodes_labels_bulk`, `remove_nodes_labels`, node/rel attrs, deletes, `delete_implicit_edges`), then **unconditional** `g.borrow_mut().apply_index_writes` (main `:1366`), then `enforce_constraints` (`let g = g.borrow()`, main `:1389`) | `&mut self.runtime.graph_mut()` (`ops/commit.rs:89`) | `commitSteps`: optional `mut`s ++ [`mut` apply_index_writes, `rd` enforce_constraints] |
| `IndexDocs::commit` (`pending.rs:473`), `Pending::commit_deferred_indexes` (`:1676`) | one `let mut g = g.borrow_mut()` held for the body | `&mut self.graph_mut()` (`runtime.rs:613`) | single `mut` |
| `Pending::end_segment` (`:1717`) | one `g.borrow_mut().roll_id_batches()` (main `:1799`) | `&mut self.runtime.graph_mut()` (`ops/commit.rs:139`) | single `mut` |
| `set_schema_baseline` (`:523`), `enforce_constraints` (`:1372`), `resync_published_indexes`' `private` (`:1621`), `EffectsBuffer::build{,_index}` (`effects/mod.rs:483,496`), `for_each_record` (`effects/v3/emit.rs:323`), `build_index_buffer` (`:226`) | one `g.borrow()` held for the body | `&self.runtime.graph()` (`ops/commit.rs:116,148`, `runtime.rs:606`, `index_ddl.rs:130`), `&g.borrow()` (`runtime.rs:462`) | single `rd` |

Within a call no step re-borrows the cell: the functions now take `&mut Graph`/`&Graph`
and call `Graph` methods (`&mut self`), and the only graph-cell borrows anywhere in
`graph/src/graph` + `graph/src/index` are the population thread's, on a *published*
version (`Protocol` table). Other threads cannot borrow the private version at all
(`Protocol.priv_borrows_owner_only`), and the owner thread is inside the call, so the
borrow word the call meets — `E`, the owner's own outstanding borrows from its call stack
— is constant for the call, and the same on main and the PR (the callers are unchanged).

Result: for every `E` and every step list, main and e30861ea3 agree on panic / error /
final graph (`commit_same`, `single_mut_same`, `single_rd_same`). The one shape where a
held borrow would differ — a call whose main path takes *no* borrow (or only shared ones
before its first exclusive one) while `E` is non-zero — does not occur: `Pending::commit`
always reaches the unconditional `apply_index_writes` `borrow_mut` before any shared
borrow or graph-free return (`commitSteps_head_mut`), and every other function borrows
first thing. **No refutation**: nothing inside `Pending::commit` borrows the same cell
through another handle (the indexer's `Arc` is only ever set to a published version,
`apply_index_writes`/`commit_index` act on `&mut self`). Live, e30861ea3 build: 90 s of
2 indexed CREATE writers, SET/REMOVE, DETACH DELETE, failing writes (resync), `MULTI`
failing writes, range-index CREATE/DROP (population threads) and 4 reader threads —
no panic.
-/
import Pr3436Review.Cell

namespace Pr3436.Hold
open Pr3436.Cell

/-- One step of a function body: a mutation group (needs the exclusive borrow) or a read
(needs a shared one). Either can fail with an error (`?`), which ends the call. -/
inductive St (G : Type) where
  | mut (f : G → Except String G)
  | rd (f : G → Except String Unit)

inductive Res (G : Type) where
  | panic
  | err (e : String)
  | ok (v : G)

/-- Run the steps with no borrow checks (inside a held borrow). -/
def runFree {G} (v : G) : List (St G) → Res G
  | [] => .ok v
  | .mut f :: ss => match f v with
    | .ok v' => runFree v' ss
    | .error e => .err e
  | .rd f :: ss => match f v with
    | .ok () => runFree v ss
    | .error e => .err e

/-- main: each step borrows the cell (word `E` = everyone else's borrows, constant for the
call) and releases it at the end of the step. -/
def runMain {G} (E : B) (v : G) : List (St G) → Res G
  | [] => .ok v
  | .mut f :: ss => match borrowMut E with
    | .panic => .panic
    | .ok _ => match f v with
      | .ok v' => runMain E v' ss
      | .error e => .err e
  | .rd f :: ss => match borrow E with
    | .panic => .panic
    | .ok _ => match f v with
      | .ok () => runMain E v ss
      | .error e => .err e

/-- e30861ea3, `&mut runtime.graph_mut()`: one exclusive borrow, then the body. -/
def runHoldMut {G} (E : B) (v : G) (ss : List (St G)) : Res G :=
  match borrowMut E with
  | .panic => .panic
  | .ok _ => runFree v ss

/-- e30861ea3, `&runtime.graph()` (write runtime: one fresh shared borrow). -/
def runHoldRd {G} (E : B) (v : G) (ss : List (St G)) : Res G :=
  match borrow E with
  | .panic => .panic
  | .ok _ => runFree v ss

/-- With nobody else borrowing, main's per-step borrows all succeed. -/
theorem runMain_free {G} : ∀ (v : G) (ss : List (St G)), runMain (.shared 0) v ss = runFree v ss
  | _, [] => rfl
  | v, .mut f :: ss => by
    simp only [runMain, runFree, borrowMut]
    cases f v with
    | ok v' => exact runMain_free v' ss
    | error e => rfl
  | v, .rd f :: ss => by
    simp only [runMain, runFree, borrow]
    cases f v with
    | ok u => exact runMain_free v ss
    | error e => rfl

/-- Shared borrows succeed on any non-exclusive word; a body of reads runs unchecked. -/
def AllRd {G} : List (St G) → Prop
  | [] => True
  | .rd _ :: ss => AllRd ss
  | .mut _ :: _ => False

theorem runMain_shared {G} (n : Nat) : ∀ (v : G) (ss : List (St G)), AllRd ss →
    runMain (.shared n) v ss = runFree v ss
  | _, [], _ => rfl
  | _, .mut _ :: _, h => by simp [AllRd] at h
  | v, .rd f :: ss, h => by
    simp only [runMain, runFree, borrow]
    cases f v with
    | ok u => exact runMain_shared n v ss h
    | error e => rfl

def HeadMut {G} : List (St G) → Prop
  | .mut _ :: _ => True
  | _ => False

/-- **Exclusive hold.** When the body's first step is a mutation, holding one
`graph_mut()` across the call gives main's panic / error / result for every external
borrow state. -/
theorem hold_mut_same {G} (E : B) (v : G) (ss : List (St G)) (hh : HeadMut ss) :
    runHoldMut E v ss = runMain E v ss := by
  match ss, hh with
  | .mut f :: rest, _ =>
    cases E with
    | shared n =>
      cases n with
      | zero => simp [runHoldMut, borrowMut]; exact (runMain_free v (.mut f :: rest)).symm
      | succ k => simp [runHoldMut, runMain, borrowMut]
    | excl => simp [runHoldMut, runMain, borrowMut]

/-- **Shared hold.** A non-empty body of reads under one `graph()` gives main's outcome. -/
theorem hold_rd_same {G} (E : B) (v : G) (ss : List (St G)) (hne : ss ≠ []) (hr : AllRd ss) :
    runHoldRd E v ss = runMain E v ss := by
  cases E with
  | shared n => simp only [runHoldRd, borrow]; exact (runMain_shared n v ss hr).symm
  | excl =>
    match ss, hne, hr with
    | .rd f :: _, _, _ => simp [runHoldRd, runMain, borrow]

/-- `Pending::commit`'s body: whichever optional mutation groups run (`opt`, the ones
whose pending state is non-empty, in source order), then the unconditional
`apply_index_writes` and `enforce_constraints`. -/
def commitSteps {G} (opt : List (G → Except String G)) (aiw : G → Except String G)
    (ec : G → Except String Unit) : List (St G) :=
  opt.map .mut ++ [.mut aiw, .rd ec]

theorem commitSteps_head_mut {G} (opt : List (G → Except String G)) (aiw : G → Except String G)
    (ec : G → Except String Unit) : HeadMut (commitSteps opt aiw ec) := by
  cases opt <;> simp [commitSteps, HeadMut]

/-- **`Pending::commit` (e30861ea3) = main.** Whatever the pending state and whatever the
external borrows, one `&mut graph_mut()` held across the call panics, fails or commits
exactly as main's sequence of borrows did. -/
theorem commit_same {G} (E : B) (v : G) (opt : List (G → Except String G))
    (aiw : G → Except String G) (ec : G → Except String Unit) :
    runHoldMut E v (commitSteps opt aiw ec) = runMain E v (commitSteps opt aiw ec) :=
  hold_mut_same E v _ (commitSteps_head_mut opt aiw ec)

/-- `IndexDocs::commit`, `commit_deferred_indexes`, `end_segment`: main held / took one
`borrow_mut` for the graph part of the body. -/
theorem single_mut_same {G} (E : B) (v : G) (f : G → Except String G) :
    runHoldMut E v [.mut f] = runMain E v [.mut f] :=
  hold_mut_same E v _ trivial

/-- `set_schema_baseline`, `enforce_constraints`, resync's `private`, the effects
builders: main held one `borrow()` for the body. -/
theorem single_rd_same {G} (E : B) (v : G) (f : G → Except String Unit) :
    runHoldRd E v [.rd f] = runMain E v [.rd f] :=
  hold_rd_same E v _ (by simp) trivial

/-- The shape that *would* differ, for the record: a body with no borrow at all (main
never touches the cell) under a held `graph_mut()` panics on a busy cell where main did
not. `Pending::commit` is never of this shape (`commitSteps_head_mut`). -/
theorem empty_body_differs :
    runHoldMut (G := Nat) (.shared 1) 0 [] = .panic ∧ runMain (G := Nat) (.shared 1) 0 [] = .ok 0 :=
  ⟨rfl, rfl⟩

end Pr3436.Hold
