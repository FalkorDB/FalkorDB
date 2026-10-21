/-
# Claims 1 and 4: `Runtime::graph()` reads the same `Graph` that `g.borrow()` would

PR #3436, branch `perf/read-graph-once` @ 653f467a9 (d7d166e82 + 896ba73ef docs +
653f467a9 "the held read borrow owns its cell; `g` is private", on origin/main
9212add39). Cites are to 653f467a9 unless marked.

| here | there |
| --- | --- |
| `B`, `borrow`, `release`, `borrowMut`, `releaseMut` | `AtomicRefCell`'s borrow word, atomic_refcell-0.1.14 `src/lib.rs`: `borrow` `:126-134` → `AtomicBorrowRef::try_new` `:209-228` (`fetch_add(1)`, panic iff HIGH_BIT set); drop `:273-283` (`fetch_sub(1)`); `borrow_mut` `:151-160` → `AtomicBorrowRefMut::try_new` `:298-320` (CAS `0 → HIGH_BIT`, else panic "already immutably/mutably borrowed"); drop `:289-294` (`store(0)`) |
| `Heap`, `Addr`            | the `Arc<AtomicRefCell<Graph>>` allocations: an `Arc` is an address of a cell (value + borrow word) |
| `Rt`                      | `Runtime`'s private `g` (`runtime.rs:160`), `write` (`:162`) and `read_guard: Option<HeldRead>` (`:157`; `HeldRead { guard, _owner }` `:115-118`). `guard = some a`: the `HeldRead`'s `_owner` is an `Arc` of cell `a` and its `guard` borrows that same cell |
| `new`                     | `Runtime::new` (`runtime.rs:439-…`): `read_guard = (!write).then(‖ { owner = Arc::clone(&g); guard = owner.borrow(); HeldRead { guard, _owner: owner } })` `:471-486` — `owner` and `g` are the same allocation, so the guard borrows `g`'s cell |
| `GraphRef`, `deref`       | `GraphRef::{Held, Borrowed}` + `Deref` (`runtime.rs:121-137`) |
| `graph`, `dropRef`        | `Runtime::graph` (`runtime.rs:554-559`: `Held(&held.guard)` / `Borrowed(self.g.borrow())`); dropping a `GraphRef` |
| `graphCell`               | `Runtime::graph_cell` (`runtime.rs:565-567`): `&self.g` |
| `borrowMain`              | origin/main's `self.g.borrow()` at every rewritten site |

**The rewrite.** Every hunk outside `runtime.rs` is one of two textual swaps inside the
same expression (so every temporary and `let` keeps its scope, and `&X.g.borrow()` /
`&X.graph()` both coerce to `&Graph` by `Deref`):

* `X.g.borrow()` → `X.graph()`: the 21 files of d7d166e82, plus nine sites 653f467a9 found
  once `g` went private (the chain was split across lines): `functions/procedures.rs:49,65,81`
  (`db.labels`, `db.relationshipTypes`, `db.propertyKeys`), `ops/node_by_index_scan.rs:325`
  (fallback scan), `ops/node_by_label_and_id_scan.rs:78`, `ops/remove.rs:130`,
  `ops/set.rs:287,322,352`. All are covered by `graph_same_result` /
  `read_program_same` (read runtime) and `graph_write_is_borrow` / `write_program_same`
  (write runtime) — none depends on which site it is (`sites_same`).
* `X.g` (the cell) → `X.graph_cell()`: `index_ddl.rs:70,107,136`,
  `ops/commit.rs:89,116,139,148`, `ops/create.rs:176,212,299,331,363,378,387`,
  `ops/delete.rs:259,414,457`, `ops/remove.rs:138`, `ops/set.rs:107`,
  `udf/js_context.rs:314,319`, `src/graph_core.rs:263`. `graph_cell()` *is* `&self.g`
  (`graph_cell_is_g`), so a `borrow_mut()`, a `borrow()` or an `Arc::clone` through it acts
  on the same cell with the same outcome as before (`graph_cell_borrowMut_same`,
  `graph_cell_borrow_same`), and it names the cell `graph()` reads (`graph_cell_same`).
-/
namespace Pr3436.Cell

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

/-- `Runtime::new`'s new lines (`runtime.rs:471-486`): a read runtime clones `g` into
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

/-- `Runtime::graph` (`runtime.rs:554-559`). -/
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
version a reader holds, `Protocol.guard_never_excl`), main and the PR both succeed. -/
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
protocol proves this for every version a reader runs on (`Protocol.guard_never_excl`) —
main and the PR compute the same answers for every read-only program. -/
theorem read_program_same {G α} (rt : Rt) (hr : rt.ok) (hw : rt.write = false)
    (ops : List (Op G α)) (h : Heap G) (hro : readOnly ops) (hne : (h rt.g).b ≠ .excl) :
    ∃ h', runMain h rt ops = .ok (ans (h rt.g).val ops, h') ∧
      runPR h rt ops = .ok (ans (h rt.g).val ops, h) :=
  let ⟨h', hm⟩ := read_program_main rt ops h hro hne
  ⟨h', hm, read_program_pr rt hr hw ops h hro⟩

/-! ### `graph_cell()` (653f467a9) -/

/-- `Runtime::graph_cell` (`runtime.rs:565-567`): `&self.g`. -/
def graphCell (rt : Rt) : Addr := rt.g

/-- main's `&X.g` at a cell site. -/
def cellMain (rt : Rt) : Addr := rt.g

theorem graph_cell_is_g (rt : Rt) : graphCell rt = cellMain rt := rfl

/-- A `borrow_mut()` through `graph_cell()` has main's outcome on main's cell. -/
theorem graph_cell_borrowMut_same {G} (h : Heap G) (rt : Rt) :
    borrowMut (h (graphCell rt)).b = borrowMut (h (cellMain rt)).b := rfl

/-- So does a `borrow()` (e.g. `set_schema_baseline`, the effects builder). -/
theorem graph_cell_borrow_same {G} (h : Heap G) (rt : Rt) :
    borrow (h (graphCell rt)).b = borrow (h (cellMain rt)).b := rfl

/-- `graph()` and `graph_cell()` cannot disagree: what `graph()` hands out derefs to the
value of the cell `graph_cell()` names. -/
theorem graph_cell_same {G} (h : Heap G) (rt : Rt) (hr : rt.ok) (r : GraphRef) (h' : Heap G)
    (hg : graph h rt = .ok (r, h')) : deref h' r = (h (graphCell rt)).val :=
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

end Pr3436.Cell
