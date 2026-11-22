import RuntimeCore.Cell

/-!
# `Runtime`'s graph since #3683: a published `Arc<Graph>` or an owned fork

Re-target of the `HeldRead` model (`Cell`, `DropOrder`, `Protocol`, kept as history: they model
`Arc<AtomicRefCell<Graph>>`, which no longer exists). Since `08616ec29` (#3683, "published versions
are `Arc<Graph>`; a fork is a plain `Graph`") `Runtime` holds

* `published : OnceCell<Arc<Graph>>` — a read query's version from the start, a write query's once
  `publish` ran. Nothing can change a published version, so reading it takes no borrow;
* `fork : RefCell<Option<Graph>>` — a write query's private graph until `publish` takes it.

| here | there |
| --- | --- |
| `QueryGraph`, `From<Arc<Graph>>`, `From<Graph>` | `runtime.rs:109-128` |
| `GraphRef`, `GraphRef.deref` | `GraphRef` `:130-138`, `impl Deref` `:140-147` (`Published(&Graph)` / `Fork(Ref<Graph>)`) |
| `Rt`, `new` | the graph fields of `Runtime` (`published`, `fork`, `write`) and `Runtime::new` `:471-500` (`write = fork.is_some()`, `set_schema_baseline(fork)` iff a fork) |
| `graph`, `dropRef` | `Runtime::graph` `:563-575` (`published.get().map_or_else(.. Ref::map(fork.borrow(), as_ref().expect(..)) .., ..)`) |
| `graphMut`, `dropMut` | `Runtime::graph_mut` `:583-587` (`RefMut::map(fork.borrow_mut(), as_mut().expect(..))`) |
| `publish` | `Runtime::publish` `:600-613`: `fork.borrow_mut().take().expect(..)`, `commit(fork)?`, `published.set(p)` asserted `is_ok` |
| `upgradeToWrite` | `Runtime::upgrade_to_write` `:626-629`: `write_escalation.upgrade_to_write(&mut self.graph_mut())` |

The borrow word is `std::cell::RefCell`'s, the same shape as the atomic one in `Cell` (shared count
or exclusive; `borrow` panics only on an exclusive borrow, `borrow_mut` only unless the count is 0),
so `Cell.B`/`borrow`/`borrowMut` are reused.

Proven: `new` gives a read runtime a published graph and no fork, a write runtime the reverse, and
never both (`new_inv`, `publish_inv`); reading a published graph never panics and touches no state
(`graph_published`, `read_graph_total` — the point of #3683); a fork is read through a borrow
(`graph_fork`) and `graph()` panics exactly when the version is unpublished and the fork is
exclusively borrowed or refused (`graph_panics_iff`); `graph_mut` panics exactly without a fork or
with any borrow out (`graphMut_panics_iff`), so always on a read runtime (`graphMut_read_panics`);
`publish` leaves the version `commit` returned and no fork (`publish_ok`), keeps `write`
(`publish_write`), and a refusal leaves neither (`publish_refused`, after which `graph()` panics:
`graph_after_refusal`); a second publish and a publish on a read runtime panic
(`publish_twice_panics`, `publish_read_panics`); after a publish `graph()` reads the published
version without a borrow and `graph_mut` panics (`graph_after_publish`,
`graphMut_after_publish`); `upgrade_to_write` hands the host exactly the fork, releases the borrow,
and returns the host's answer (`upgrade_spec`, `upgrade_release`).
-/
namespace RuntimeCore.QG

open RuntimeCore.Cell (B R borrow release borrowMut releaseMut)

/-- `QueryGraph` (`runtime.rs:109-114`). -/
inductive QueryGraph (G : Type) where
  | published (g : G)
  | fork (g : G)

/-- The graph fields of `Runtime`. `published` is the `OnceCell<Arc<Graph>>`'s content, `fork` the
`RefCell<Option<Graph>>`'s, `forkB` that cell's borrow word, `baseline` whether
`pending.set_schema_baseline(fork)` ran. -/
structure Rt (G : Type) where
  published : Option G
  fork : Option G
  forkB : B
  write : Bool
  baseline : Bool

/-- `Runtime::new` (`:471-500`): `(published, fork)` from the `QueryGraph`, `write = fork.is_some()`,
the schema baseline only for a fork. -/
def new {G : Type} : QueryGraph G → Rt G
  | .published g => { published := some g, fork := none, forkB := .shared 0, write := false, baseline := false }
  | .fork f => { published := none, fork := some f, forkB := .shared 0, write := true, baseline := true }

/-- `GraphRef` (`:130-138`): the arm says how it was got; the payload is what it derefs to. -/
inductive GraphRef (G : Type) where
  | published (g : G)
  | fork (g : G)

/-- `impl Deref for GraphRef` (`:140-147`): both arms are the graph they hold. -/
def GraphRef.deref {G : Type} : GraphRef G → G
  | .published g => g
  | .fork g => g

/-- `Runtime::graph` (`:563-575`). A published version is returned as is. Otherwise the fork cell is
borrowed shared first (`fork.borrow()`, which panics iff it is exclusively borrowed), and only then
`expect` panics if the fork is gone. -/
def graph {G : Type} (rt : Rt G) : R (GraphRef G × Rt G) :=
  match rt.published with
  | some g => .ok (.published g, rt)
  | none =>
    match borrow rt.forkB with
    | .panic => .panic
    | .ok b' =>
      match rt.fork with
      | none => .panic
      | some f => .ok (.fork f, { rt with forkB := b' })

/-- Dropping a `GraphRef`: a `Ref` releases its shared borrow, a published reference holds none. -/
def dropRef {G : Type} (rt : Rt G) : GraphRef G → Rt G
  | .published _ => rt
  | .fork _ => { rt with forkB := release rt.forkB }

/-- `Runtime::graph_mut` (`:583-587`): `fork.borrow_mut()` (panics unless no borrow is out), then
`expect` (panics without a fork). The `RefMut` derefs to the fork. -/
def graphMut {G : Type} (rt : Rt G) : R (G × Rt G) :=
  match borrowMut rt.forkB with
  | .panic => .panic
  | .ok b' =>
    match rt.fork with
    | none => .panic
    | some f => .ok (f, { rt with forkB := b' })

/-- Dropping the `RefMut` returned by `graph_mut`. -/
def dropMut {G : Type} (rt : Rt G) : Rt G := { rt with forkB := releaseMut rt.forkB }

/-- `Runtime::publish` (`:600-613`). `fork.borrow_mut()` is a temporary of the first statement, so
the borrow word is back to what it was after `take`; `take().expect(..)` panics without a fork;
`commit` consumes the fork and either refuses (`Err`: the fork stays gone, nothing is published) or
returns the published version, which `OnceCell::set` stores (it fails, and the `assert!` panics, if a
version was already published). -/
def publish {G E : Type} (commit : G → Except E G) (rt : Rt G) : R (Rt G × Except E Unit) :=
  match borrowMut rt.forkB with
  | .panic => .panic
  | .ok _ =>
    match rt.fork with
    | none => .panic
    | some f =>
      let rt1 : Rt G := { rt with fork := none }
      match commit f with
      | .error e => .ok (rt1, .error e)
      | .ok p =>
        match rt1.published with
        | some _ => .panic
        | none => .ok ({ rt1 with published := some p }, .ok ())

/-- `Runtime::upgrade_to_write` (`:626-629`): the host (`esc`, `WriteEscalation::upgrade_to_write`)
receives `&mut` the fork borrowed by `graph_mut()` and may change it; the `RefMut` temporary is
dropped at the end of the statement. -/
def upgradeToWrite {G : Type} (esc : G → Except String Unit × G) (rt : Rt G) :
    R (Except String Unit × Rt G) :=
  match graphMut rt with
  | .panic => .panic
  | .ok (f, rt1) =>
    let (r, f') := esc f
    .ok (r, dropMut { rt1 with fork := some f' })

/-- No published version and a fork, or the reverse, or (after a refused commit) neither: never both. -/
def Rt.inv {G : Type} (rt : Rt G) : Prop := rt.published = none ∨ rt.fork = none

/-- The version the runtime reads: the published one if any, else the fork. -/
def Rt.cur {G : Type} (rt : Rt G) : Option G := rt.published.or rt.fork

/-! ### `QueryGraph` / `new` -/

theorem new_published {G : Type} (g : G) :
    (new (QueryGraph.published g)).published = some g ∧ (new (QueryGraph.published g)).fork = none ∧
    (new (QueryGraph.published g)).write = false ∧ (new (QueryGraph.published g)).baseline = false ∧
    (new (QueryGraph.published g)).forkB = .shared 0 := ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem new_fork {G : Type} (f : G) :
    (new (QueryGraph.fork f)).published = none ∧ (new (QueryGraph.fork f)).fork = some f ∧
    (new (QueryGraph.fork f)).write = true ∧ (new (QueryGraph.fork f)).baseline = true ∧
    (new (QueryGraph.fork f)).forkB = .shared 0 := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `write = fork.is_some()`: a runtime is a write runtime iff it was given a fork. -/
theorem new_write_iff {G : Type} (q : QueryGraph G) :
    (new q).write = true ↔ (new q).fork.isSome := by
  cases q <;> simp [new]

theorem new_inv {G : Type} (q : QueryGraph G) : (new q).inv := by
  cases q <;> simp [new, Rt.inv]

/-- Exactly one of the two holds a graph at construction. -/
theorem new_cur {G : Type} (g : G) :
    (new (QueryGraph.published g)).cur = some g ∧ (new (QueryGraph.fork g)).cur = some g :=
  ⟨rfl, rfl⟩

/-! ### `deref` -/

theorem deref_spec {G : Type} (g : G) :
    (GraphRef.published g).deref = g ∧ (GraphRef.fork g).deref = g := ⟨rfl, rfl⟩

/-! ### `graph` -/

/-- A published version is read with no borrow and no state change, whatever the fork cell's state:
the borrow counter that every concurrent reader used to write (#3436's motivation) is gone. -/
theorem graph_published {G : Type} (rt : Rt G) (g : G) (h : rt.published = some g) :
    graph rt = .ok (.published g, rt) := by
  simp [graph, h]

/-- A read runtime's `graph()` never panics. -/
theorem read_graph_total {G : Type} (g : G) :
    graph (new (QueryGraph.published g)) = .ok (.published g, new (QueryGraph.published g)) :=
  graph_published _ g rfl

/-- Reading an unpublished fork takes a shared borrow of the cell. -/
theorem graph_fork {G : Type} (rt : Rt G) (f : G) (n : Nat) (hp : rt.published = none)
    (hf : rt.fork = some f) (hb : rt.forkB = .shared n) :
    graph rt = .ok (.fork f, { rt with forkB := .shared (n + 1) }) := by
  simp [graph, hp, hf, hb, borrow]

theorem graph_fork_release {G : Type} (rt : Rt G) (f : G) (n : Nat) (hp : rt.published = none)
    (hf : rt.fork = some f) (hb : rt.forkB = .shared n) :
    (match graph rt with | .ok (r, rt') => some (dropRef rt' r) | .panic => none) = some rt := by
  rw [graph_fork rt f n hp hf hb]
  cases rt
  simp_all [dropRef, release]

/-- `graph()` panics exactly when nothing is published and the fork is exclusively borrowed or gone
(refused at commit). -/
theorem graph_panics_iff {G : Type} (rt : Rt G) :
    graph rt = .panic ↔ rt.published = none ∧ (rt.forkB = .excl ∨ rt.fork = none) := by
  unfold graph
  cases hp : rt.published with
  | some g => simp
  | none =>
    cases hb : rt.forkB with
    | excl => simp [borrow]
    | shared n =>
      cases hf : rt.fork <;> simp [borrow]

/-- Whatever `graph()` returns derefs to the version the runtime reads (`Rt.cur`: the published
one if any, else the fork). -/
theorem graph_deref {G : Type} (rt : Rt G) (r : GraphRef G) (rt' : Rt G)
    (h : graph rt = .ok (r, rt')) : rt.cur = some r.deref := by
  unfold graph at h
  cases hp : rt.published with
  | some g =>
    simp [hp] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [Rt.cur, hp, GraphRef.deref]
  | none =>
    simp only [hp] at h
    cases hb : rt.forkB with
    | excl => simp [hb, borrow] at h
    | shared n =>
      cases hf : rt.fork with
      | none => simp [hb, hf, borrow] at h
      | some f =>
        simp [hb, hf, borrow] at h
        obtain ⟨rfl, rfl⟩ := h
        simp [Rt.cur, hp, hf, GraphRef.deref]

/-- Every site that reads through `graph()` computes `f` of the version the runtime reads, for any
function `f` of the graph (an accessor, a scan, the reply's `version`): the accessor theorems of
`Accessors`/`RelAccessors` transfer through `graph()` unchanged. -/
theorem graph_same_result {G α : Type} (f : G → α) (rt : Rt G) (r : GraphRef G) (rt' : Rt G)
    (h : graph rt = .ok (r, rt')) : rt.cur.map f = some (f r.deref) := by
  rw [graph_deref rt r rt' h]
  rfl

/-! ### `graph_mut` -/

theorem graphMut_panics_iff {G : Type} (rt : Rt G) :
    graphMut rt = .panic ↔ rt.forkB ≠ .shared 0 ∨ rt.fork = none := by
  unfold graphMut
  cases hb : rt.forkB with
  | excl => simp [borrowMut]
  | shared n =>
    cases n with
    | zero => cases hf : rt.fork <;> simp [borrowMut]
    | succ n => simp [borrowMut]

/-- On a read runtime `graph_mut` panics ("graph_mut on a published version"): there is no fork. -/
theorem graphMut_read_panics {G : Type} (g : G) :
    graphMut (new (QueryGraph.published g)) = .panic := by
  simp [graphMut, new, borrowMut]

/-- On a fresh write runtime it hands out the fork and leaves the cell exclusively borrowed. -/
theorem graphMut_write {G : Type} (f : G) :
    graphMut (new (QueryGraph.fork f)) =
      .ok (f, { (new (QueryGraph.fork f)) with forkB := .excl }) := by
  simp [graphMut, new, borrowMut]

theorem graphMut_release {G : Type} (rt : Rt G) (f : G) (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) :
    (match graphMut rt with | .ok (_, rt') => some (dropMut rt') | .panic => none) = some rt := by
  cases rt
  simp_all [graphMut, borrowMut, dropMut, releaseMut]

/-- A live `graph()` borrow makes `graph_mut` panic (RefCell's "already borrowed"). -/
theorem graphMut_while_reading {G : Type} (rt : Rt G) (f : G) (n : Nat) (hp : rt.published = none)
    (hf : rt.fork = some f) (hb : rt.forkB = .shared n) :
    (match graph rt with | .ok (_, rt') => graphMut rt' | .panic => .panic) = .panic := by
  rw [graph_fork rt f n hp hf hb]
  simp [graphMut, borrowMut]

/-- A live `graph_mut()` makes `graph()` panic on an unpublished fork. -/
theorem graph_while_writing {G : Type} (f : G) :
    (match graphMut (new (QueryGraph.fork f)) with | .ok (_, rt') => graph rt' | .panic => .panic) = .panic := by
  simp [graphMut, new, borrowMut, graph, borrow]

/-! ### `publish` -/

theorem publish_ok {G E : Type} (commit : G → Except E G) (rt : Rt G) (f p : G)
    (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) (hp : rt.published = none)
    (hc : commit f = .ok p) :
    publish commit rt = .ok ({ rt with fork := none, published := some p }, .ok ()) := by
  simp [publish, hb, hf, hp, hc, borrowMut]

theorem publish_refused {G E : Type} (commit : G → Except E G) (rt : Rt G) (f : G) (e : E)
    (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) (hc : commit f = .error e) :
    publish commit rt = .ok ({ rt with fork := none }, .error e) := by
  simp [publish, hb, hf, hc, borrowMut]

/-- `publish` keeps the `write` flag: a published write runtime is still a write runtime. -/
theorem publish_write {G E : Type} (commit : G → Except E G) (rt rt' : Rt G) (r : Except E Unit)
    (h : publish commit rt = .ok (rt', r)) : rt'.write = rt.write ∧ rt'.forkB = rt.forkB := by
  unfold publish at h
  cases hb : rt.forkB <;> simp [hb, borrowMut] at h
  rename_i n
  cases n <;> simp at h
  cases hf : rt.fork <;> simp [hf] at h
  rename_i f
  cases hc : commit f <;> simp [hc] at h
  · obtain ⟨rfl, rfl⟩ := h; simp
  · cases hp : rt.published <;> simp [hp] at h
    obtain ⟨rfl, rfl⟩ := h; simp

theorem publish_inv {G E : Type} (commit : G → Except E G) (rt rt' : Rt G) (r : Except E Unit)
    (h : publish commit rt = .ok (rt', r)) : rt'.inv := by
  unfold publish at h
  cases hb : rt.forkB <;> simp [hb, borrowMut] at h
  rename_i n
  cases n <;> simp at h
  cases hf : rt.fork <;> simp [hf] at h
  rename_i f
  cases hc : commit f <;> simp [hc] at h
  · obtain ⟨rfl, rfl⟩ := h; simp [Rt.inv]
  · cases hp : rt.published <;> simp [hp] at h
    obtain ⟨rfl, rfl⟩ := h; simp [Rt.inv]

/-- A second publish panics (the fork is gone), as does one on a read runtime. -/
theorem publish_twice_panics {G E : Type} (commit : G → Except E G) (rt rt' : Rt G) (r : Except E Unit)
    (h : publish commit rt = .ok (rt', r)) : publish commit rt' = .panic := by
  have hf : rt'.fork = none := by
    unfold publish at h
    cases hb : rt.forkB <;> simp [hb, borrowMut] at h
    rename_i n
    cases n <;> simp at h
    cases hf : rt.fork <;> simp [hf] at h
    rename_i f
    cases hc : commit f <;> simp [hc] at h
    · obtain ⟨rfl, rfl⟩ := h; rfl
    · cases hp : rt.published <;> simp [hp] at h
      obtain ⟨rfl, rfl⟩ := h; rfl
  obtain ⟨_, hb⟩ := publish_write commit rt rt' r h
  unfold publish
  cases hb' : rt'.forkB with
  | excl => simp [borrowMut]
  | shared n => cases n <;> simp [hf, borrowMut]

theorem publish_read_panics {G E : Type} (commit : G → Except E G) (g : G) :
    publish commit (new (QueryGraph.published g)) = .panic := by
  simp [publish, new, borrowMut]

/-- After a successful publish `graph()` reads the published version, with no borrow. -/
theorem graph_after_publish {G E : Type} (commit : G → Except E G) (rt : Rt G) (f p : G)
    (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) (hp : rt.published = none)
    (hc : commit f = .ok p) :
    (match publish commit rt with | .ok (rt', _) => graph rt' | .panic => .panic) =
      .ok (.published p, { rt with fork := none, published := some p }) := by
  rw [publish_ok commit rt f p hb hf hp hc]
  simp [graph]

/-- ... and `graph_mut` panics: a published version is never changed. -/
theorem graphMut_after_publish {G E : Type} (commit : G → Except E G) (rt : Rt G) (f p : G)
    (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) (hp : rt.published = none)
    (hc : commit f = .ok p) :
    (match publish commit rt with | .ok (rt', _) => graphMut rt' | .panic => .panic) = .panic := by
  rw [publish_ok commit rt f p hb hf hp hc]
  simp [graphMut, borrowMut, hb]

/-- After a refused commit nothing is published and the fork is gone: `graph()` panics
("a write query's fork was refused at commit"). -/
theorem graph_after_refusal {G E : Type} (commit : G → Except E G) (rt : Rt G) (f : G) (e : E)
    (hb : rt.forkB = .shared 0) (hf : rt.fork = some f) (hp : rt.published = none)
    (hc : commit f = .error e) :
    (match publish commit rt with | .ok (rt', _) => graph rt' | .panic => .panic) = .panic := by
  rw [publish_refused commit rt f e hb hf hc]
  simp [graph, hp, hb, borrow]

/-! ### `upgrade_to_write` -/

/-- The host sees exactly the fork, its answer is the result, and what it did to the fork is kept. -/
theorem upgrade_spec {G : Type} (esc : G → Except String Unit × G) (f : G) :
    upgradeToWrite esc (new (QueryGraph.fork f)) =
      .ok ((esc f).1, { (new (QueryGraph.fork f)) with fork := some (esc f).2 }) := by
  simp [upgradeToWrite, graphMut, new, borrowMut, dropMut, releaseMut]

/-- It leaves no borrow behind (the `RefMut` was a temporary). -/
theorem upgrade_release {G : Type} (esc : G → Except String Unit × G) (rt rt' : Rt G)
    (r : Except String Unit) (h : upgradeToWrite esc rt = .ok (r, rt')) : rt'.forkB = .shared 0 := by
  unfold upgradeToWrite graphMut at h
  cases hb : rt.forkB <;> simp [hb, borrowMut] at h
  rename_i n
  cases n <;> simp at h
  cases hf : rt.fork <;> simp [hf] at h
  obtain ⟨rfl, rfl⟩ := h
  simp [dropMut, releaseMut]

/-- On a read runtime it panics (there is no fork to hand over). -/
theorem upgrade_read_panics {G : Type} (esc : G → Except String Unit × G) (g : G) :
    upgradeToWrite esc (new (QueryGraph.published g)) = .panic := by
  simp [upgradeToWrite, graphMut, new, borrowMut]

end RuntimeCore.QG
