/-
# `Hold`: borrows around `Pending::commit` (main 256ab4d3f)

There is no `Arc<AtomicRefCell<Graph>>`: a read query holds a published `Arc<Graph>`
(no borrow at all), a write query owns its fork in `Runtime.fork : RefCell<Option<Graph>>`
(`runtime.rs`, `graph_mut` = `fork.borrow_mut()`, `graph()` = `fork.borrow()`), and the
commit functions below are still called as `&mut self.runtime.graph_mut()` / `&self.runtime.graph()`
(`ops/commit.rs`, `runtime.rs`, `index_ddl.rs`). So the theorems below hold with
the borrow word `E` read as the **fork cell's** word (`RuntimeCore.QG.Rt.forkB`): a `RefCell` is
`!Sync`, so no other thread can have a borrow out; `E` is only the owner's own outstanding borrows
from its call stack. The index population thread borrows nothing, since it reads a published
`Arc<Graph>` (`published_read_no_borrow`). Tying the model to `RuntimeCore.QueryGraph`:

* `holdMut_panics_iff_graphMut` — the held `graph_mut()` panics exactly when
  `Runtime::graph_mut` does (a borrow out on the fork cell, or no fork);
* `published_read_no_borrow` — reading a published version takes no borrow, whatever the fork cell's state;
* `hold_after_publish_panics` — after `publish` the fork is gone, so a held `graph_mut()` panics.
-/
import RuntimeCore.Cell
import RuntimeCore.QueryGraph

namespace PendingCommit.Hold
open RuntimeCore.Cell

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

/-- e30861ea3, `&mut runtime.graph_mut()`: one exclusive borrow, then the body. -/
def runHoldMut {G} (E : B) (v : G) (ss : List (St G)) : Res G :=
  match borrowMut E with
  | .panic => .panic
  | .ok _ => runFree v ss

/-! ## Against `RuntimeCore.QG` (the published-`Arc<Graph>` runtime) -/

theorem runFree_not_panic {G : Type} : ∀ (v : G) (ss : List (St G)), runFree v ss ≠ .panic
  | _, [] => by simp [runFree]
  | v, .mut f :: ss => by
    simp only [runFree]; cases f v with
    | ok v' => exact runFree_not_panic v' ss
    | error e => simp
  | v, .rd f :: ss => by
    simp only [runFree]; cases f v with
    | ok u => exact runFree_not_panic v ss
    | error e => simp

/-- The held `graph_mut()` panics exactly when the exclusive borrow of the fork cell is refused
(the body never panics inside a held borrow) — which is `Runtime::graph_mut` on a runtime that has
its fork (`QG.graphMut_panics_iff`: a borrow out, or no fork). -/
theorem holdMut_panics_iff_graphMut {G : Type} (rt : RuntimeCore.QG.Rt G) (f : G) (hf : rt.fork = some f)
    (v : G) (ss : List (St G)) :
    runHoldMut rt.forkB v ss = .panic ↔ RuntimeCore.QG.graphMut rt = .panic := by
  rw [RuntimeCore.QG.graphMut_panics_iff]
  unfold runHoldMut
  cases hb : rt.forkB with
  | excl => simp [borrowMut]
  | shared n =>
    cases n with
    | zero => simp [borrowMut, hf]; exact runFree_not_panic v ss
    | succ k => simp [borrowMut]

/-- Reading a published version takes no borrow whatever the fork cell's word is. -/
theorem published_read_no_borrow {G : Type} (rt : RuntimeCore.QG.Rt G) (g : G) (h : rt.published = some g) :
    RuntimeCore.QG.graph rt = .ok (.published g, rt) :=
  RuntimeCore.QG.graph_published rt g h

/-- After `publish` there is no fork: a held `graph_mut()` panics. -/
theorem hold_after_publish_panics {G : Type} (rt : RuntimeCore.QG.Rt G) (h : rt.fork = none) :
    RuntimeCore.QG.graphMut rt = .panic := by
  unfold RuntimeCore.QG.graphMut
  cases borrowMut rt.forkB <;> simp [h]

end PendingCommit.Hold
