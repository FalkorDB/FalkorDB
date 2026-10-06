import FalkorCowBTree.Cursor

/-!
# Snapshot isolation (`CowBTree::clone`, `RangeIter::_root`, `make_private`)

In Rust a snapshot is `CowBTree::clone` / the `_root` a `RangeIter` owns: an `Arc` bump that shares
every page. Isolation rests on two code facts, which is what licenses modelling pages as immutable
values:

1. `make_private` (`node.rs:391`) **always** replaces the `Arc<Branch>` by a fresh clone before
   handing out `&mut` (never `Arc::make_mut`'s in-place path), and the only `&mut` into a tree is
   through the writer's own `root` field (`insert` / `remove` / `insert_batch` take `&mut self`).
2. Leaf pages are `Arc<[u8]>` built fresh by every edit (`Arc::from(buf)`, `leaf/*.rs`) and never
   written after construction (`grep get_mut` finds only `make_private`).

So every write is a function *old root value → new root value*; a holder of the old root observes
the old value forever. In the model that is literally true of `insert`/`remove`/`insertBatch`, and
the theorems below record it together with the sharing discipline (only the routed child changes).
-/

namespace CowBTree

/-- A writer operation. -/
inductive Op where
  | ins (x : E)
  | rem (x : E)
  | batch (xs : List E)

def applyOp (c : Cfg) (t : Node × Nat) : Op → Node × Nat
  | .ins x => insert c t.2 t.1 x
  | .rem x => remove c t.2 t.1 x
  | .batch xs => insertBatch c t.2 t.1 xs

def applyOps (c : Cfg) (t : Node × Nat) (ops : List Op) : Node × Nat := ops.foldl (applyOp c) t

/-- **Snapshot isolation**: take a snapshot (clone), run any sequence of writes on the writer, and the
    snapshot's range scans still return exactly the snapshot-time contents. -/
theorem snapshot_isolated (c : Cfg) (t : Node × Nat) (ops : List Op) (lo hi n : Nat) :
    let snapshot := t
    let _writer := applyOps c t ops
    range snapshot.2 snapshot.1 lo hi n = range t.2 t.1 lo hi n := rfl

/-- ...and, on a well-formed tree, that is exactly the reference filter of the snapshot's contents,
    independent of the writer. -/
theorem snapshot_reads_committed (c : Cfg) (t : Node × Nat) (ops : List Op) (lo hi n : Nat)
    (hw : TreeWF c t.2 t.1) (hn : (toList t.2 t.1).length ≤ n) :
    let snapshot := t
    let _writer := applyOps c t ops
    range snapshot.2 snapshot.1 lo hi n = (toList t.2 t.1).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi)) :=
  range_spec c t.2 t.1 lo hi n hw hn

/-- **Path copying** (`node.rs:244-248`): an insert that does not split a branch replaces exactly
    the routed child; every other child is the *same* node (shared, never copied). -/
theorem insertOne_shares_siblings (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) (x : E) (c' : Node)
    (hc : (insertOne c h (cs.getD (childIndex seps x) default) x).2 = none)
    (hc' : (insertOne c h (cs.getD (childIndex seps x) default) x).1 = c') :
    insertOne c (h + 1) (.branch seps cs) x = (.branch seps (setAt cs (childIndex seps x) c'), none) := by
  simp only [insertOne]
  generalize insertOne c h (cs.getD (childIndex seps x) default) x = res at hc hc'
  obtain ⟨a, b⟩ := res
  simp only at hc hc'
  subst hc hc'
  rfl

end CowBTree
