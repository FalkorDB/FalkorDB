import RedisLayer.QuerySession

/-!
# Module-load environment repair (#3494) and the native index builder's host (#3585)

| here | there |
| --- | --- |
| `scanLen`, `entriesOf`, `leakDeepCopy`, `view` | `graph::env_snapshot::leak_deep_copy`, `graph/src/env_snapshot.rs` |
| `Ld`, `syncDeepbind` | `sync_deepbind_environ`, `src/module_init.rs` (Linux only; called from `graph_init` after the `KMP_*` `set_var` calls) |
| `publish`, `warn` | `RedisBuildHost::publish` / `warn`, `src/graph_core.rs` (`impl BuildHost`, feature `index-falkordb`) |

**`leak_deep_copy`.** A C `environ` is a pointer to a NULL-terminated array of pointers to
NUL-terminated strings. `src : Option (List (Option Bytes))` is that pointer (`none` = NULL) and the
array's cells up to and beyond its terminator; the `while` loop counts the cells before the first
NULL (`scanLen`; a missing terminator is the caller's contract violation, as in the Rust `# Safety`).
The copy allocates one fresh cell per entry (`entry.to_owned().into_raw()`: the string's bytes, now
owned by the copy) and a terminating NULL, in a fresh array. Cells are `(address, bytes)` with
addresses from a fresh counter `next`, so no copied string aliases anything allocated before.

* `copy_terminated` — `len + 1` cells, the last NULL, and no NULL before it;
* `copy_read_back` — reading the copy back gives exactly the source's entries, in order;
* `copy_null_src` — a NULL source gives the empty environment (a lone NULL);
* `copy_fresh` — every string cell is allocated at or after `next`;
* `copy_indep` — the result is a function of the entries' bytes only, so a later `setenv`
  `realloc`ing the source array, or a `putenv` caller freeing its buffer, cannot change it.

**`sync_deepbind_environ`.** `dlopen(NULL, RTLD_NOW|RTLD_NOLOAD)` and `dlsym(main, "environ")` are
libc; their results are the inputs `Ld.main` (non-NULL handle) and `Ld.live` (address of the main
executable's `environ` variable, `none` = NULL). `ours` is `&raw mut environ`, this module's own.
`syncDeepbind` returns what this module's `environ` holds afterwards: unchanged when the handle is
NULL, when the symbol is NULL, or when it is this module's own variable (the path CI reaches);
otherwise an owned snapshot of the live array (`sync_unchanged`, `sync_installs`).

**`RedisBuildHost`.** `publish(f)` takes the GIL through the same guard queries use (`hold_gil`),
runs `f` once, and drops the guard: `publish_holds` (the GIL is held while `f` runs),
`publish_nested` (a thread that already holds it, the main thread, takes no lock events),
`publish_log` (otherwise the events are `gilLock, f's…, gilUnlock` and the thread ends without a
context). `warn` appends exactly the message to the server log (`warn_appends`).
-/
namespace RedisLayer.EnvBuild

open RedisLayer.QuerySession

abbrev Bytes := List Nat

/-! ### `leak_deep_copy` -/

/-- The `while` loop's count: cells before the first NULL (`src.add(len)` read until NULL). -/
def scanLen : List (Option Bytes) → Nat
  | [] => 0
  | none :: _ => 0
  | some _ :: t => scanLen t + 1

/-- The entries the loop copies: a NULL source has none. -/
def entriesOf : Option (List (Option Bytes)) → List Bytes
  | none => []
  | some a => (a.take (scanLen a)).filterMap id

/-- `leak_deep_copy`: one fresh `(address, bytes)` cell per entry, then the terminating NULL. -/
def leakDeepCopy (src : Option (List (Option Bytes))) (next : Nat) : List (Option (Nat × Bytes)) :=
  (entriesOf src).mapIdx (fun i e => some (next + i, e)) ++ [none]

/-- Reading a `char **` back: the strings before the first NULL. -/
def readBack (a : List (Option (Nat × Bytes))) : List Bytes :=
  (a.takeWhile Option.isSome).filterMap (fun (o : Option (Nat × Bytes)) => o.map Prod.snd)

/-- The copy as the C array a reader sees (addresses forgotten). -/
def view (a : List (Option (Nat × Bytes))) : List (Option Bytes) :=
  a.map (fun (o : Option (Nat × Bytes)) => o.map Prod.snd)

theorem view_mapIdx (l : List Bytes) (k : Nat) :
    view (l.mapIdx (fun i e => some (k + i, e))) = l.map some := by
  apply List.ext_getElem <;> simp [view]

theorem copy_null_src (n : Nat) : leakDeepCopy none n = [none] := by
  simp [leakDeepCopy, entriesOf]

theorem copy_length (src : Option (List (Option Bytes))) (n : Nat) :
    (leakDeepCopy src n).length = (entriesOf src).length + 1 := by
  simp [leakDeepCopy]

theorem copy_terminated (src : Option (List (Option Bytes))) (n : Nat) :
    (leakDeepCopy src n).getLast? = some none ∧
    ∀ c ∈ (leakDeepCopy src n).dropLast, c.isSome = true := by
  constructor
  · simp [leakDeepCopy]
  · intro c hc
    simp only [leakDeepCopy, List.dropLast_concat, List.mem_mapIdx] at hc
    obtain ⟨i, _, rfl⟩ := hc
    rfl

theorem takeWhile_cells (cs : List (Nat × Bytes)) :
    ((cs.map some) ++ [none]).takeWhile Option.isSome = cs.map some := by
  induction cs with
  | nil => simp
  | cons c cs ih => simp [ih]

theorem copy_read_back (src : Option (List (Option Bytes))) (n : Nat) :
    readBack (leakDeepCopy src n) = entriesOf src := by
  unfold readBack leakDeepCopy
  have hcs : ((entriesOf src).mapIdx (fun i e => some (n + i, e))) =
      ((entriesOf src).mapIdx (fun i e => (n + i, e))).map some := by
    apply List.ext_getElem <;> simp
  rw [hcs, takeWhile_cells]
  have hsnd : ((entriesOf src).mapIdx (fun i e => (n + i, e))).map Prod.snd = entriesOf src := by
    apply List.ext_getElem <;> simp
  rw [List.filterMap_map]
  simpa [Function.comp_def] using hsnd

theorem copy_fresh (src : Option (List (Option Bytes))) (n : Nat) :
    ∀ c ∈ leakDeepCopy src n, ∀ a b, c = some (a, b) → n ≤ a := by
  intro c hc a b h
  simp only [leakDeepCopy, List.mem_append, List.mem_mapIdx, List.mem_singleton] at hc
  rcases hc with ⟨i, _, rfl⟩ | rfl
  · simp at h; omega
  · simp at h

/-- The copy depends only on the entries' bytes: two sources with the same entries copy alike. -/
theorem copy_indep (s₁ s₂ : Option (List (Option Bytes))) (n : Nat) (h : entriesOf s₁ = entriesOf s₂) :
    leakDeepCopy s₁ n = leakDeepCopy s₂ n := by
  simp [leakDeepCopy, h]

/-! ### `sync_deepbind_environ` -/

/-- The libc answers the function branches on. -/
structure Ld where
  /-- `dlopen(NULL, RTLD_NOW | RTLD_NOLOAD)` is non-NULL. -/
  main : Bool
  /-- `dlsym(main, "environ")`: the address of the executable's `environ`; `none` is NULL. -/
  live : Option Nat
  /-- `*live`: the array the executable's `environ` points at. -/
  liveVal : Option (List (Option Bytes))
  /-- `&raw mut environ`: this module's own `environ` variable. -/
  ours : Nat

/-- What this module's `environ` holds after the call, given what it held before. -/
def syncDeepbind (ld : Ld) (oursVal : Option (List (Option Bytes))) (next : Nat) :
    Option (List (Option Bytes)) :=
  if ld.main = false then oursVal
  else match ld.live with
    | none => oursVal
    | some l => if l = ld.ours then oursVal else some (view (leakDeepCopy ld.liveVal next))

theorem sync_unchanged (ld : Ld) (o : Option (List (Option Bytes))) (n : Nat)
    (h : ld.main = false ∨ ld.live = none ∨ ld.live = some ld.ours) : syncDeepbind ld o n = o := by
  unfold syncDeepbind
  rcases h with h | h | h <;> simp [h]

/-- Otherwise it installs a NULL-terminated snapshot of the live array's strings. -/
theorem sync_installs (ld : Ld) (o : Option (List (Option Bytes))) (n l : Nat)
    (hm : ld.main = true) (hl : ld.live = some l) (hne : l ≠ ld.ours) :
    syncDeepbind ld o n = some ((entriesOf ld.liveVal).map some ++ [none]) := by
  unfold syncDeepbind
  simp only [hm, hl, hne, ite_false, view, leakDeepCopy, List.map_append]
  have := view_mapIdx (entriesOf ld.liveVal) n
  simp only [view] at this
  simp [this]

/-! ### `RedisBuildHost` -/

/-- `RedisBuildHost::publish`: `let _gil = hold_gil(); f();` — the guard drops after `f`. -/
def publish (f : Th → Th) (t : Th) : Th :=
  let (nr, t1) := gilAcquire t
  gilDrop nr (f t1)

theorem publish_holds (t : Th) : holdsGil (gilAcquire t).2 = true := by
  unfold gilAcquire
  split <;> simp_all [holdsGil]

/-- A thread that already holds the GIL (the main thread) runs `f` with no lock events. -/
theorem publish_nested (f : Th → Th) (t : Th) (h : holdsGil t = true) : publish f t = f t := by
  simp [publish, gilAcquire, h, gilDrop]

/-- Otherwise: lock, `f`'s own events, unlock, and the thread holds no context afterwards.
`f` is any step that keeps the thread's role and context and only appends events. -/
theorem publish_log (f : Th → Th) (extra : List Ev) (t : Th) (hm : t.main = false) (hc : t.ctx = false)
    (hf : ∀ u : Th, (f u).main = u.main ∧ (f u).ctx = u.ctx ∧ (f u).log = u.log ++ extra) :
    (publish f t).ctx = false ∧ (publish f t).log = t.log ++ [Ev.gilLock] ++ extra ++ [Ev.gilUnlock] := by
  obtain ⟨_, h2, h3⟩ := hf ⟨false, true, t.log ++ [Ev.gilLock]⟩
  simp [publish, gilAcquire, holdsGil, hm, hc, gilDrop, h2, h3]

/-- `RedisBuildHost::warn`: `log_warning(message)` appends the message to the server log. -/
def warn (serverLog : List String) (message : String) : List String := serverLog ++ [message]

theorem warn_appends (l : List String) (m : String) : warn l m = l ++ [m] := rfl

end RedisLayer.EnvBuild
