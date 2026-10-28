/-
# Claims 1 and 4: `Runtime::graph()` reads the same `Graph` that `g.borrow()` would

**Merged: PR #3436 (25babfe4f); folded from the retired `pr3436_review`.** The PR branch @ e30861ea3 is the same code as main (`.rs` identical apart from line numbers: `runtime.rs` cites below are current main, ca36bfa15). Branch history: PR #3436, `perf/read-graph-once` @ e30861ea3 (d7d166e82, 896ba73ef, 653f467a9,
merge of origin/main b97da5a1c = 30f8d7cff, then e30861ea3 "graph()/graph_mut() instead of
a cell accessor"). Cites are to e30861ea3 unless marked; `pre_e30861e_*` names keep the
653f467a9 `graph_cell()` theorems as history.

| here | there |
| --- | --- |
| `B`, `borrow`, `release`, `borrowMut`, `releaseMut` | `AtomicRefCell`'s borrow word, atomic_refcell-0.1.14 `src/lib.rs`: `borrow` `:126-134` → `AtomicBorrowRef::try_new` `:209-228` (`fetch_add(1)`, panic iff HIGH_BIT set); drop `:273-283` (`fetch_sub(1)`); `borrow_mut` `:151-160` → `AtomicBorrowRefMut::try_new` `:298-320` (CAS `0 → HIGH_BIT`, else panic "already immutably/mutably borrowed"); drop `:289-294` (`store(0)`) |
| `Heap`, `Addr`            | the `Arc<AtomicRefCell<Graph>>` allocations: an `Arc` is an address of a cell (value + borrow word) |
| `Rt`                      | `Runtime`'s private `g` (`runtime.rs:161`), `write` and `read_guard: Option<HeldRead>` (`:157`; `HeldRead { guard, _owner }` `:115-118`). `guard = some a`: the `HeldRead`'s `_owner` is an `Arc` of cell `a` and its `guard` borrows that same cell |
| `new`                     | `Runtime::new` (`runtime.rs:444-…`): `read_guard = (!write).then(‖ { owner = Arc::clone(&g); guard = owner.borrow(); HeldRead { guard, _owner: owner } })` `:476-491` — `owner` and `g` are the same allocation, so the guard borrows `g`'s cell |
| `GraphRef`, `deref`       | `GraphRef::{Held, Borrowed}` + `Deref` (`runtime.rs:121-137`) |
| `graph`, `dropRef`        | `Runtime::graph` (`runtime.rs:559-564`: `Held(&held.guard)` / `Borrowed(self.g.borrow())`); dropping a `GraphRef` |
| `graphMut`                | `Runtime::graph_mut` (`runtime.rs:576-579`): `assert!(self.write, …)`, then `self.g.borrow_mut()` |
| `graphArc`                | `Runtime::graph_arc` (`runtime.rs:587-589`): `Arc::clone(&self.g)` (model in `DropOrder`) |
| `borrowMain`              | origin/main's `self.g.borrow()` at every rewritten site |

**The rewrite.** Every hunk outside `runtime.rs` is one of these swaps inside the same
expression (every temporary and `let` keeps its scope; `&X.g.borrow()` / `&X.graph()` both
coerce to `&Graph` by `Deref`):

* `X.g.borrow()` → `X.graph()`: the 21 files of d7d166e82, plus nine sites 653f467a9 found
  once `g` went private: `functions/procedures.rs:49,65,81`, `ops/node_by_index_scan.rs:325`,
  `ops/node_by_label_and_id_scan.rs:78`, `ops/remove.rs:130`, `ops/set.rs:287,322,352`.
  Covered by `graph_same_result` / `read_program_same` / `sites_same` (read runtime) and
  `graph_write_is_borrow` / `write_program_same` (write runtime).
* `X.g.borrow_mut()` → `X.graph_mut()` (e30861ea3; 653f467a9 had `graph_cell().borrow_mut()`):
  `index_ddl.rs:71,102`, `ops/create.rs:176,212,299,331,363,378,387`, `ops/delete.rs:259,413,454`,
  `ops/remove.rs:138`, `ops/set.rs:107`, `runtime.rs:1491,1510`. `graph_mut_write_same`:
  on a write runtime it is main's `g.borrow_mut()`; `graph_mut_read_asserts`: on a read
  runtime the `assert!` fires before the cell is touched; `graph_mut_same_cell`: it
  borrows the cell `graph()` reads.
* functions that took the cell and borrowed inside now take `&mut Graph` / `&Graph`, the
  caller holding one `graph_mut()` / `graph()` across the call (`ops/commit.rs:89,113,139,148`,
  `runtime.rs:611,617`, `index_ddl.rs:≈130`): proven equivalent in `Hold`.
* `X.g` kept past the call → `X.graph_arc()` (`udf/js_context.rs:314`, `src/graph_core.rs:263`):
  an owned clone, `DropOrder.graph_arc_keeps_g`.
-/
namespace RuntimeCore.Cell

/-- The borrow word: `shared n` = `n` live `AtomicRef`s; `excl` = HIGH_BIT (an `AtomicRefMut`). -/
inductive B where
  | shared (n : Nat)
  | excl
  deriving DecidableEq, Repr

/-- A step result: `panic` = `panic!` inside atomic_refcell. -/
inductive R (α : Type) where
  | ok (a : α)
  | panic
  deriving Repr

/-- `AtomicBorrowRef::try_new` (`lib.rs:209-228`): `fetch_add(1)`, panic iff HIGH_BIT. -/
def borrow : B → R B
  | .shared n => .ok (.shared (n + 1))
  | .excl => .panic

/-- `AtomicBorrowRef::drop` (`lib.rs:273-283`): `fetch_sub(1)`. -/
def release : B → B
  | .shared (n + 1) => .shared n
  | b => b

/-- `AtomicBorrowRefMut::try_new` (`lib.rs:298-320`): CAS `0 → HIGH_BIT`. -/
def borrowMut : B → R B
  | .shared 0 => .ok .excl
  | _ => .panic

/-- `AtomicBorrowRefMut::drop` (`lib.rs:289-294`): `store(0)`. -/
def releaseMut : B → B
  | .excl => .shared 0
  | b => b

abbrev Addr := Nat

structure Cell (G : Type) where
  val : G
  b : B

/-- All live `Arc<AtomicRefCell<Graph>>` allocations. -/
abbrev Heap (G : Type) := Addr → Cell G

def Heap.set {G} (h : Heap G) (a : Addr) (c : Cell G) : Heap G :=
  fun x => if x = a then c else h x

/-- The fields of `Runtime` this PR touches. `guard = some a` means `read_guard` holds a
`HeldRead` whose `_owner` is an `Arc` of cell `a` and whose `guard` borrows cell `a`. -/
structure Rt where
  g : Addr
  write : Bool
  guard : Option Addr
  deriving DecidableEq, Repr

/-- `Runtime::new`'s new lines (`runtime.rs:475-490`): a read runtime clones `g` into
`_owner` (same allocation, so same address) and borrows that cell once; a write runtime
takes nothing. The strong-count side of the clone is in `DropOrder`. -/
def new {G} (h : Heap G) (g : Addr) (write : Bool) : R (Rt × Heap G) :=
  if write then .ok ({ g, write, guard := none }, h)
  else match borrow (h g).b with
    | .ok b' => .ok ({ g, write, guard := some g }, h.set g { h g with b := b' })
    | .panic => .panic

/-- `GraphRef` (`runtime.rs:121-126`). `held a` is the `&Graph` the guard derefs to;
`borrowed a` owns a fresh `AtomicRef` into the cell at `a`. -/
inductive GraphRef where
  | held (a : Addr)
  | borrowed (a : Addr)
  deriving DecidableEq, Repr

/-- `impl Deref for GraphRef` (`runtime.rs:128-137`): both arms deref to the cell's value
(`AtomicRef::deref` is `self.value.as_ref()`, `lib.rs:394`). -/
def deref {G} (h : Heap G) : GraphRef → G
  | .held a => (h a).val
  | .borrowed a => (h a).val

/-- `Runtime::graph` (`runtime.rs:558-563`). -/
def graph {G} (h : Heap G) (rt : Rt) : R (GraphRef × Heap G) :=
  match rt.guard with
  | some a => .ok (.held a, h)
  | none => match borrow (h rt.g).b with
    | .ok b' => .ok (.borrowed rt.g, h.set rt.g { h rt.g with b := b' })
    | .panic => .panic

/-- Dropping the `GraphRef` at the end of its scope. -/
def dropRef {G} (h : Heap G) : GraphRef → Heap G
  | .held _ => h
  | .borrowed a => h.set a { h a with b := release (h a).b }

/-- origin/main: `self.g.borrow()` — always a fresh `AtomicRef` into `g`'s cell. -/
def borrowMain {G} (h : Heap G) (rt : Rt) : R (GraphRef × Heap G) :=
  match borrow (h rt.g).b with
  | .ok b' => .ok (.borrowed rt.g, h.set rt.g { h rt.g with b := b' })
  | .panic => .panic

/-- The runtime invariant `new` establishes and nothing breaks, since neither `g` nor
`read_guard` is ever reassigned (both private since 653f467a9; `DropOrder.g_never_reassigned`):
a guard, if any, borrows `g`'s own cell, and only read runtimes have one. -/
def Rt.ok (rt : Rt) : Prop := rt.guard = (if rt.write then none else some rt.g)

theorem new_ok {G} (h : Heap G) (g : Addr) (w : Bool) (rt : Rt) (h' : Heap G)
    (hn : new h g w = .ok (rt, h')) : rt.ok ∧ rt.g = g ∧ rt.write = w := by
  unfold new at hn
  cases w with
  | true => simp at hn; obtain ⟨rfl, rfl⟩ := hn; simp [Rt.ok]
  | false =>
    simp at hn
    split at hn
    · simp at hn; obtain ⟨rfl, rfl⟩ := hn; simp [Rt.ok]
    · cases hn

theorem heap_set_val {G} (h : Heap G) (a x : Addr) (b : B) :
    (h.set a { h a with b := b } x).val = (h x).val := by
  unfold Heap.set; split <;> simp_all

/-- `new` never changes a cell's *value*: it only moves the borrow word. -/
theorem new_vals {G} (h : Heap G) (g : Addr) (w : Bool) (rt : Rt) (h' : Heap G)
    (hn : new h g w = .ok (rt, h')) (x : Addr) : (h' x).val = (h x).val := by
  unfold new at hn
  cases w with
  | true => simp at hn; obtain ⟨rfl, rfl⟩ := hn; rfl
  | false =>
    simp at hn
    split at hn
    · simp at hn; obtain ⟨rfl, rfl⟩ := hn; exact heap_set_val _ _ _ _
    · cases hn

/-! ### Claim 1: same `Graph` -/

/-- **Claim 1 (deref).** Whatever `graph()` returns derefs to the value of `g`'s cell —
exactly what `g.borrow()` derefs to. -/
theorem graph_deref {G} (h : Heap G) (rt : Rt) (hr : rt.ok) (r : GraphRef) (h' : Heap G)
    (hg : graph h rt = .ok (r, h')) : deref h' r = (h rt.g).val := by
  unfold graph at hg
  unfold Rt.ok at hr
  split at hg
  · rename_i a ha
    simp at hg; obtain ⟨rfl, rfl⟩ := hg
    cases hw : rt.write <;> simp [hw] at hr <;> rw [hr] at ha
    · cases ha; rfl
    · cases ha
  · split at hg
    · simp at hg; obtain ⟨rfl, rfl⟩ := hg
      simp [deref, Heap.set]
    · cases hg

theorem borrowMain_deref {G} (h : Heap G) (rt : Rt) (r : GraphRef) (h' : Heap G)
    (hg : borrowMain h rt = .ok (r, h')) : deref h' r = (h rt.g).val := by
  unfold borrowMain at hg
  split at hg
  · simp at hg; obtain ⟨rfl, rfl⟩ := hg; simp [deref, Heap.set]
  · cases hg

/-- **Claim 1 (headline).** For any function `f` of the graph (an accessor, a scan, the
reply's `version`), a site that reads through `graph()` computes the same answer as the
same site reading through `g.borrow()` on main, whenever both borrows succeed. -/
theorem graph_same_result {G α} (f : G → α) (h : Heap G) (rt : Rt) (hr : rt.ok)
    (r₁ r₂ : GraphRef) (h₁ h₂ : Heap G)
    (hpr : graph h rt = .ok (r₁, h₁)) (hmain : borrowMain h rt = .ok (r₂, h₂)) :
    f (deref h₁ r₁) = f (deref h₂ r₂) := by
  rw [graph_deref h rt hr r₁ h₁ hpr, borrowMain_deref h rt r₂ h₂ hmain]

/-- A read runtime's `graph()` never panics and never touches the borrow word. -/
theorem graph_read_held {G} (h : Heap G) (rt : Rt) (hr : rt.ok) (hw : rt.write = false) :
    graph h rt = .ok (.held rt.g, h) := by
  unfold Rt.ok at hr; simp [hw] at hr
  simp [graph, hr]

/-- Main's `g.borrow()` succeeds exactly when the cell is not mutably borrowed; so on
every state where no `borrow_mut` is live on `g`'s cell (proven for every published
version a reader holds, `Proto.guard_never_excl`), main and the PR both succeed. -/
theorem borrowMain_ok_iff {G} (h : Heap G) (rt : Rt) :
    (∃ r h', borrowMain h rt = .ok (r, h')) ↔ (h rt.g).b ≠ .excl := by
  unfold borrowMain
  cases hb : (h rt.g).b with
  | shared n => simp [borrow]
  | excl => simp [borrow]

/-! ### Claim 4: write runtimes still borrow per access -/

/-- **Claim 4.** A write runtime's `graph()` is *definitionally* main's `g.borrow()`:
same success/panic, same borrow-word change, same `GraphRef` (an owned `AtomicRef`),
released at the same scope end. So the write path's `borrow_mut` calls interleave with
`graph()` exactly as they did with `g.borrow()`. -/
theorem graph_write_is_borrow {G} (h : Heap G) (rt : Rt) (hr : rt.ok) (hw : rt.write = true) :
    graph h rt = borrowMain h rt := by
  unfold Rt.ok at hr; simp [hw] at hr
  simp [graph, borrowMain, hr]

/-- A program a runtime runs against its cell: reads through `graph()` / `g.borrow()`
(`read f` takes the borrow, computes `f`, drops it at the end of the statement), and its
own `borrow_mut` sites (`mutate u`: `borrow_mut`, apply `u`, drop). -/
inductive Op (G : Type) (α : Type) where
  | read (f : G → α)
  | mutate (u : G → G)

/-- Run a program on the PR: `graph()` per read. -/
def runPR {G α} (h : Heap G) (rt : Rt) : List (Op G α) → R (List α × Heap G)
  | [] => .ok ([], h)
  | .read f :: ops =>
    match graph h rt with
    | .ok (r, h₁) =>
      match runPR (dropRef h₁ r) rt ops with
      | .ok (xs, h₂) => .ok (f (deref h₁ r) :: xs, h₂)
      | .panic => .panic
    | .panic => .panic
  | .mutate u :: ops =>
    match borrowMut (h rt.g).b with
    | .ok _ => runPR (h.set rt.g { val := u (h rt.g).val, b := (h rt.g).b }) rt ops
    | .panic => .panic

/-- Run the same program on main: `g.borrow()` per read. -/
def runMain {G α} (h : Heap G) (rt : Rt) : List (Op G α) → R (List α × Heap G)
  | [] => .ok ([], h)
  | .read f :: ops =>
    match borrowMain h rt with
    | .ok (r, h₁) =>
      match runMain (dropRef h₁ r) rt ops with
      | .ok (xs, h₂) => .ok (f (deref h₁ r) :: xs, h₂)
      | .panic => .panic
    | .panic => .panic
  | .mutate u :: ops =>
    match borrowMut (h rt.g).b with
    | .ok _ => runMain (h.set rt.g { val := u (h rt.g).val, b := (h rt.g).b }) rt ops
    | .panic => .panic

/-- **Claim 4 (programs).** For a write runtime, any interleaving of reads and its own
`borrow_mut`s gives the same results, the same panics and the same final heap on the PR
as on main. -/
theorem write_program_same {G α} (rt : Rt) (hr : rt.ok) (hw : rt.write = true) :
    ∀ (ops : List (Op G α)) (h : Heap G), runPR h rt ops = runMain h rt ops
  | [], _ => rfl
  | .read f :: ops, h => by
    simp only [runPR, runMain, graph_write_is_borrow h rt hr hw]
    split <;> simp [write_program_same rt hr hw ops]
  | .mutate u :: ops, h => by
    simp only [runPR, runMain]
    split <;> simp [write_program_same rt hr hw ops]

/-- A read-only program (no `mutate`: a read runtime runs no `borrow_mut` site, see
`Reuse.read_plan_no_update`). -/
def readOnly {G α} : List (Op G α) → Prop
  | [] => True
  | .read _ :: ops => readOnly ops
  | .mutate _ :: _ => False

/-- The answers a read-only program computes against graph value `v`. -/
def ans {G α} (v : G) : List (Op G α) → List α
  | [] => []
  | .read f :: ops => f v :: ans v ops
  | .mutate _ :: ops => ans v ops

/-- On a read runtime, the PR runs a read-only program without ever panicking or
touching the borrow word, and returns `f (value of g's cell)` for every read. -/
theorem read_program_pr {G α} (rt : Rt) (hr : rt.ok) (hw : rt.write = false) :
    ∀ (ops : List (Op G α)) (h : Heap G), readOnly ops →
      runPR h rt ops = .ok (ans (h rt.g).val ops, h)
  | [], _, _ => rfl
  | .read f :: ops, h, hro => by
    simp only [runPR, graph_read_held h rt hr hw, dropRef]
    rw [read_program_pr rt hr hw ops h hro]
    simp [deref, ans]
  | .mutate _ :: _, _, hro => by simp [readOnly] at hro

theorem dropRef_borrowMain {G} (h : Heap G) (rt : Rt) (r : GraphRef) (h₁ : Heap G)
    (hm : borrowMain h rt = .ok (r, h₁)) :
    (∀ x, (dropRef h₁ r x).val = (h x).val) ∧ (dropRef h₁ r rt.g).b = (h rt.g).b := by
  unfold borrowMain at hm
  split at hm
  · rename_i b' hb
    simp at hm; obtain ⟨rfl, rfl⟩ := hm
    refine ⟨?_, ?_⟩
    · intro x; simp only [dropRef, Heap.set]; split <;> (try split) <;> simp_all
    · simp only [dropRef, Heap.set]; simp
      cases hh : (h rt.g).b with
      | shared n => rw [hh] at hb; simp [borrow] at hb; subst hb; simp [release]
      | excl => rw [hh] at hb; simp [borrow] at hb
  · cases hm

/-- Main on a read runtime: as long as `g`'s cell is not mutably borrowed, every
`g.borrow()` succeeds and the program returns the same answers. -/
theorem read_program_main {G α} (rt : Rt) :
    ∀ (ops : List (Op G α)) (h : Heap G), readOnly ops → (h rt.g).b ≠ .excl →
      ∃ h', runMain h rt ops = .ok (ans (h rt.g).val ops, h')
  | [], h, _, _ => ⟨h, rfl⟩
  | .read f :: ops, h, hro, hne => by
    obtain ⟨r, h₁, hm⟩ := (borrowMain_ok_iff h rt).2 hne
    have hv := dropRef_borrowMain h rt r h₁ hm
    have hne' : (dropRef h₁ r rt.g).b ≠ .excl := by rw [hv.2]; exact hne
    obtain ⟨h₂, hrun⟩ := read_program_main rt ops (dropRef h₁ r) hro hne'
    refine ⟨h₂, ?_⟩
    simp only [runMain, hm, hrun, borrowMain_deref h rt r h₁ hm, hv.1 rt.g, ans]
  | .mutate _ :: _, _, hro, _ => by simp [readOnly] at hro

/-- **Claim 1 (programs).** On a read runtime whose cell is not mutably borrowed — the
protocol proves this for every version a reader runs on (`Proto.guard_never_excl`) —
main and the PR compute the same answers for every read-only program. -/
theorem read_program_same {G α} (rt : Rt) (hr : rt.ok) (hw : rt.write = false)
    (ops : List (Op G α)) (h : Heap G) (hro : readOnly ops) (hne : (h rt.g).b ≠ .excl) :
    ∃ h', runMain h rt ops = .ok (ans (h rt.g).val ops, h') ∧
      runPR h rt ops = .ok (ans (h rt.g).val ops, h) :=
  let ⟨h', hm⟩ := read_program_main rt ops h hro hne
  ⟨h', hm, read_program_pr rt hr hw ops h hro⟩

/-! ### `graph_mut()` (e30861ea3) -/

/-- What `graph_mut()` can do: the `assert!` fires (a read runtime), the cell's
`borrow_mut` panics, or it hands out the `AtomicRefMut`. -/
inductive GM (G : Type) where
  | assertFail
  | panic
  | ok (h : Heap G)

/-- `Runtime::graph_mut` (`runtime.rs:576-579`). -/
def graphMut {G} (h : Heap G) (rt : Rt) : GM G :=
  if rt.write then
    match borrowMut (h rt.g).b with
    | .ok b' => .ok (h.set rt.g { h rt.g with b := b' })
    | .panic => .panic
  else .assertFail

/-- main (and d7d166e82/653f467a9): `X.g.borrow_mut()` / `X.graph_cell().borrow_mut()`. -/
def borrowMutMain {G} (h : Heap G) (rt : Rt) : GM G :=
  match borrowMut (h rt.g).b with
  | .ok b' => .ok (h.set rt.g { h rt.g with b := b' })
  | .panic => .panic

/-- On a write runtime, `graph_mut()` is main's `g.borrow_mut()`: same cell, same panic,
same borrow word. -/
theorem graph_mut_write_same {G} (h : Heap G) (rt : Rt) (hw : rt.write = true) :
    graphMut h rt = borrowMutMain h rt := by
  simp [graphMut, borrowMutMain, hw]

/-- On a read runtime the `assert!` fires before any borrow is attempted: the result does
not depend on the borrow word at all. (Unreachable: a read plan has no site that calls
it, `Reuse.read_plan_no_update`; main would instead have panicked inside the cell with
"already immutably borrowed" — or, before the PR, raced.) -/
theorem graph_mut_read_asserts {G} (h : Heap G) (rt : Rt) (hw : rt.write = false) :
    graphMut h rt = .assertFail := by
  simp [graphMut, hw]

theorem graph_mut_read_ignores_word {G} (h h' : Heap G) (rt : Rt) (hw : rt.write = false) :
    graphMut h rt = .assertFail ∧ graphMut h' rt = .assertFail :=
  ⟨graph_mut_read_asserts h rt hw, graph_mut_read_asserts h' rt hw⟩

/-- `graph_mut()` borrows exactly the cell `graph()` reads (`g`'s): it changes only that
cell's borrow word, and the value `graph()` derefs to is that cell's. -/
theorem graph_mut_same_cell {G} (h h' : Heap G) (rt : Rt) (hm : graphMut h rt = .ok h') :
    ∀ x, x ≠ rt.g → h' x = h x := by
  intro x hx
  unfold graphMut at hm
  split at hm
  · split at hm
    · cases hm; simp [Heap.set, hx]
    · cases hm
  · cases hm

/-! ### History: `graph_cell()` (653f467a9, removed by e30861ea3) -/

/-- 653f467a9's `Runtime::graph_cell` (`runtime.rs:565-567` there): `&self.g`. -/
def pre_e30861e_graphCell (rt : Rt) : Addr := rt.g

/-- main's `&X.g` at a cell site. -/
def pre_e30861e_cellMain (rt : Rt) : Addr := rt.g

theorem pre_e30861e_graph_cell_is_g (rt : Rt) : pre_e30861e_graphCell rt = pre_e30861e_cellMain rt := rfl

theorem pre_e30861e_graph_cell_borrowMut_same {G} (h : Heap G) (rt : Rt) :
    borrowMut (h (pre_e30861e_graphCell rt)).b = borrowMut (h (pre_e30861e_cellMain rt)).b := rfl

theorem pre_e30861e_graph_cell_borrow_same {G} (h : Heap G) (rt : Rt) :
    borrow (h (pre_e30861e_graphCell rt)).b = borrow (h (pre_e30861e_cellMain rt)).b := rfl

theorem pre_e30861e_graph_cell_same {G} (h : Heap G) (rt : Rt) (hr : rt.ok) (r : GraphRef) (h' : Heap G)
    (hg : graph h rt = .ok (r, h')) : deref h' r = (h (pre_e30861e_graphCell rt)).val :=
  graph_deref h rt hr r h' hg

/-- **Claim 1 over all rewritten sites.** Whatever list of `X.g.borrow()` sites is
rewritten to `X.graph()` — the 21 files of d7d166e82 and the nine of 653f467a9 alike — a
read runtime computes the same answers at every one of them as main does. -/
theorem sites_same {G α} (sites : List (G → α)) (rt : Rt) (hr : rt.ok) (hw : rt.write = false)
    (h : Heap G) (hne : (h rt.g).b ≠ .excl) :
    ∃ h', runMain h rt (sites.map .read) = .ok (sites.map (· (h rt.g).val), h') ∧
      runPR h rt (sites.map .read) = .ok (sites.map (· (h rt.g).val), h) := by
  have hro : readOnly (sites.map (Op.read (G := G) (α := α))) := by
    induction sites with
    | nil => trivial
    | cons f fs ih => exact ih
  have hans : ∀ fs : List (G → α), ans (h rt.g).val (fs.map Op.read) = fs.map (· (h rt.g).val) := by
    intro fs
    induction fs with
    | nil => rfl
    | cons f fs ih => simp [ans, ih]
  rw [← hans sites]
  exact read_program_same rt hr hw _ h hro hne

end RuntimeCore.Cell
