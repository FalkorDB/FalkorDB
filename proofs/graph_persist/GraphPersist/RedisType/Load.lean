/-! # Loading keys into the keyspace (redis_type.rs:77-176, :722-780, :851-870)

| here | there |
| --- | --- |
| `ioKeyName`           | `io_key_name` (:77-88): NULL name → `""`, else the lossy UTF-8 rendering |
| `MS`, `alloc`         | `GRAPH_REGISTRY`, `DECODE_STATE.placeholders`, and the `Arc<RwLock<ThreadedGraph>>` values (`some none` = a placeholder `ThreadedGraph`, `some (some v)` = a loaded graph) |
| `loadWhole`           | `graph_rdb_load`, `LoadedKey::Graph` arm (:98-108) |
| `loadPartial`         | `graph_rdb_load`, `LoadedKey::Partial` arm (:109-162) |
| `finalizePending`     | `finalize_pending_graphs` (:722-754) + `install_graph` (:757-780) |
| `metaLoad`            | `graphmeta_rdb_load` (:851-867) |
| `cacheOnLoad`         | the `DEFAULT_CACHE_SIZE` (:66) every RDB load passes to `rdb_load_graph` |

`fin` is `DECODE_STATE.finalized[graph_name]` as the decoder (`Persist.Pending`) leaves it:
`none` after every key but the last of a multi-key graph, the finished graph after the last.

* `load_main_last` / `load_main_first` — whichever position the graph's own key has in
  the load order, after `finalize_pending_graphs` the registry entry for the graph and the
  value Redis holds under its key are the **same** `Arc`, holding the finished graph; no
  virtual key is registered; all placeholders are dropped.
* `cache_size_ignored` — a loaded graph's query cache is 25 whatever `CACHE_SIZE` says.
-/
namespace GraphPersist.RedisType

variable {Nm V : Type} [DecidableEq Nm]

/-- `io_key_name`: `RedisModule_GetKeyNameFromIO` may return NULL. `lossy` is
`String::from_utf8_lossy`. -/
def ioKeyName (lossy : List UInt8 → String) (raw : Option (List UInt8)) : String :=
  match raw with
  | none => ""
  | some b => lossy b

theorem ioKeyName_spec (lossy : List UInt8 → String) :
    ioKeyName lossy none = "" ∧ ∀ b, ioKeyName lossy (some b) = lossy b := ⟨rfl, fun _ => rfl⟩

def fupd {α β : Type} [DecidableEq α] (f : α → β) (a : α) (x : β) : α → β := fun b => if b = a then x else f b

structure MS (Nm V : Type) where
  reg : Nm → Option Nat
  ph : Nm → Option Nat
  heap : Nat → Option (Option V)
  next : Nat

/-- `Arc::new(RwLock::new(..))`. -/
def alloc (ms : MS Nm V) (c : Option V) : MS Nm V × Nat :=
  ({ ms with heap := fupd ms.heap ms.next (some c), next := ms.next + 1 }, ms.next)

/-- `LoadedKey::Graph`: a fresh `Arc`, registered under the key, returned to Redis. -/
def loadWhole (ms : MS Nm V) (kn : Nm) (v : V) : MS Nm V × Nat :=
  let (ms1, a) := alloc ms (some v)
  ({ ms1 with reg := fupd ms1.reg kn (some a) }, a)

/-- `LoadedKey::Partial { is_virtual }` (:109-162). Returns the state, what is left in
`finalized[graph_name]`, and the `Arc` handed to Redis. -/
def loadPartial (ms : MS Nm V) (fin : Option V) (gname kn : Nm) (isVirtual : Bool) : MS Nm V × Option V × Nat :=
  match (if kn = gname then fin else none) with
  | some v =>
    match ms.ph kn with
    | some p => ({ ms with heap := fupd ms.heap p (some (some v)), ph := fupd ms.ph kn none }, none, p)
    | none =>
      let (ms1, a) := alloc ms (some v)
      ({ ms1 with reg := fupd ms1.reg kn (some a) }, none, a)
  | none =>
    let (ms1, p) := alloc ms none
    let ms2 := { ms1 with ph := fupd ms1.ph kn (some p) }
    (if isVirtual then ms2 else { ms2 with reg := fupd ms2.reg kn (some p) }, fin, p)

/-- `finalize_pending_graphs` for one graph, `pending` already empty (the decoder never
leaves a pending graph with `keys_remaining == 0`: it finalizes inline). -/
def finalizePending (ms : MS Nm V) (fin : Option V) (gname : Nm) : MS Nm V :=
  let ms1 := match fin with
    | some v =>
      match ms.ph gname with
      | some p => { ms with heap := fupd ms.heap p (some (some v)), ph := fupd ms.ph gname none }
      | none => ms   -- "no placeholder pointer … graph data will be lost"
    | none => ms
  { ms1 with ph := fun _ => none }

/-- The non-final keys: each gets a placeholder (registered iff it is the graph's own key). -/
def loadPrefix (ms : MS Nm V) (gname : Nm) : List Nm → MS Nm V
  | [] => ms
  | kn :: ks => loadPrefix (loadPartial ms none gname kn (decide (kn ≠ gname))).1 gname ks

/-- Fresh: nothing is registered or placed under these names, and the heap is empty from
`next` on. -/
structure Fresh (ms : MS Nm V) (names : List Nm) : Prop where
  reg : ∀ n ∈ names, ms.reg n = none
  ph : ∀ n ∈ names, ms.ph n = none
  heap : ∀ i, ms.next ≤ i → ms.heap i = none

theorem loadPrefix_virtual (ms : MS Nm V) (gname : Nm) (ks : List Nm) (hk : ∀ k ∈ ks, k ≠ gname) :
    (loadPrefix ms gname ks).reg = ms.reg ∧ (loadPrefix ms gname ks).ph gname = ms.ph gname ∧
    ms.next ≤ (loadPrefix ms gname ks).next ∧
    ∀ i, i < ms.next → (loadPrefix ms gname ks).heap i = ms.heap i := by
  induction ks generalizing ms with
  | nil => simp [loadPrefix]
  | cons k ks ih =>
    have hk0 : k ≠ gname := hk k (by simp)
    obtain ⟨i1, i2, i3, i4⟩ := ih (loadPartial ms none gname k (decide (k ≠ gname))).1 (fun x hx => hk x (by simp [hx]))
    have e : (loadPartial ms none gname k (decide (k ≠ gname))).1 =
        ⟨ms.reg, fupd ms.ph k (some ms.next), fupd ms.heap ms.next (some none), ms.next + 1⟩ := by
      simp [loadPartial, alloc, hk0]
    simp only [loadPrefix]
    rw [e] at i1 i2 i3 i4 ⊢
    refine ⟨i1, by rw [i2]; simp [fupd, Ne.symm hk0], by simp at i3; omega, fun i hi => ?_⟩
    rw [i4 i (by simp; omega)]; simp [fupd]; omega

theorem loadPrefix_append (ms : MS Nm V) (gname : Nm) (a b : List Nm) :
    loadPrefix ms gname (a ++ b) = loadPrefix (loadPrefix ms gname a) gname b := by
  induction a generalizing ms with
  | nil => rfl
  | cons k ks ih => simp only [List.cons_append, loadPrefix]; exact ih _

/-- **The graph's own key is the last to load.** It finds the finished graph in `finalized`,
gets a fresh `Arc` with it, and registers it. -/
theorem load_main_last (ms : MS Nm V) (gname : Nm) (vkeys : List Nm) (R : V)
    (hv : ∀ k ∈ vkeys, k ≠ gname) (hf : Fresh ms (gname :: vkeys)) :
    let ms1 := loadPrefix ms gname vkeys
    let r := loadPartial ms1 (some R) gname gname false
    let ms3 := finalizePending r.1 r.2.1 gname
    ms3.reg gname = some r.2.2 ∧ ms3.heap r.2.2 = some (some R) ∧ (∀ k ∈ vkeys, ms3.reg k = none) ∧
    (∀ n, ms3.ph n = none) ∧ r.2.1 = none := by
  intro ms1 r ms3
  obtain ⟨i1, i2, i3, i4⟩ := loadPrefix_virtual ms gname vkeys hv
  have hph : ms1.ph gname = none := by simp only [ms1]; rw [i2]; exact hf.ph gname (by simp)
  have hr : r = (⟨fupd ms1.reg gname (some ms1.next), ms1.ph, fupd ms1.heap ms1.next (some (some R)),
      ms1.next + 1⟩, none, ms1.next) := by
    simp [r, loadPartial, hph, alloc]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [ms3, finalizePending, hr, fupd]
  · simp [ms3, finalizePending, hr, fupd]
  · intro k hk
    have hk0 : k ≠ gname := hv k hk
    simp [ms3, finalizePending, hr, fupd, hk0]
    show ms1.reg k = none
    simp only [ms1]; rw [i1]; exact hf.reg k (by simp [hk])
  · intro n; simp [ms3, finalizePending, hr]
  · simp [hr]

/-- **The graph's own key loads before the last key.** It gets a placeholder `Arc`,
registered under the graph's name; the last (virtual) key finishes the graph, which waits
in `finalized` until `finalize_pending_graphs` writes it into that same placeholder. -/
theorem load_main_first (ms : MS Nm V) (gname : Nm) (pre post : List Nm) (last : Nm) (R : V)
    (hpre : ∀ k ∈ pre, k ≠ gname) (hpost : ∀ k ∈ post, k ≠ gname) (hlast : last ≠ gname)
    (hf : Fresh ms (gname :: (pre ++ post ++ [last]))) :
    let ms1 := loadPrefix ms gname (pre ++ gname :: post)
    let r := loadPartial ms1 (some R) gname last true
    let ms3 := finalizePending r.1 r.2.1 gname
    ∃ p, ms3.reg gname = some p ∧ ms3.heap p = some (some R) ∧
      (∀ k ∈ pre ++ post ++ [last], ms3.reg k = none) ∧ (∀ n, ms3.ph n = none) := by
  intro ms1 r ms3
  -- after `pre`
  obtain ⟨a1, a2, a3, a4⟩ := loadPrefix_virtual ms gname pre hpre
  let msA := loadPrefix ms gname pre
  -- the main key: a placeholder at `msA.next`, registered and placed under `gname`
  have hmain : (loadPartial msA none gname gname (decide (gname ≠ gname))).1 =
      ⟨fupd msA.reg gname (some msA.next), fupd msA.ph gname (some msA.next),
        fupd msA.heap msA.next (some none), msA.next + 1⟩ := by
    simp [loadPartial, alloc]
  let msB := (loadPartial msA none gname gname (decide (gname ≠ gname))).1
  obtain ⟨b1, b2, b3, b4⟩ := loadPrefix_virtual msB gname post hpost
  have hsplit : ms1 = loadPrefix msB gname post := by
    simp only [ms1, msB, msA, loadPrefix_append, loadPrefix]
  have hphg : ms1.ph gname = some msA.next := by
    rw [hsplit, b2]; simp only [msB, hmain, fupd]; simp
  have hregg : ms1.reg gname = some msA.next := by
    rw [hsplit, b1]; simp only [msB, hmain, fupd]; simp
  have hheap : ms1.heap msA.next = some none := by
    rw [hsplit, b4 _ (by simp only [msB, hmain]; omega)]; simp only [msB, hmain, fupd]; simp
  have hB : msB.next = msA.next + 1 := by simp only [msB, hmain]
  have hnext : msA.next < ms1.next := by
    rw [hsplit]; have := b3; omega
  -- the last key: `finalized` is keyed by `gname`, so it stays; the key gets a placeholder
  have hr : r = (⟨ms1.reg, fupd ms1.ph last (some ms1.next), fupd ms1.heap ms1.next (some none),
      ms1.next + 1⟩, some R, ms1.next) := by
    simp [r, loadPartial, alloc, hlast]
  refine ⟨msA.next, ?_, ?_, ?_, ?_⟩
  · simp [ms3, finalizePending, hr, fupd, Ne.symm hlast, hphg, hregg]
  · simp [ms3, finalizePending, hr, fupd, Ne.symm hlast, hphg]
  · intro k hk
    have hk0 : k ≠ gname := by
      simp only [List.mem_append, List.mem_singleton] at hk
      rcases hk with (hk | hk) | rfl
      · exact hpre k hk
      · exact hpost k hk
      · exact hlast
    simp only [ms3, finalizePending, hr, fupd, Ne.symm hlast, hphg]
    show ms1.reg k = none
    rw [hsplit, b1]
    simp only [msB, hmain, fupd, hk0, ite_false, msA]
    rw [a1]; exact hf.reg k (List.mem_cons_of_mem _ hk)
  · intro n; simp [ms3, finalizePending]

/-- `graphmeta_rdb_load` (:851): decode like a graph key, keep a dummy byte on success,
NULL (load failure) on error. -/
def metaLoad {E : Type} (res : Except E Unit) : Option Unit :=
  match res with
  | .ok _ => some ()
  | .error _ => none

theorem metaLoad_spec {E : Type} (e : E) : metaLoad (.ok () : Except E Unit) = some () ∧ metaLoad (.error e : Except E Unit) = none :=
  ⟨rfl, rfl⟩

/-! ## Cache size on load -/

/-- `DEFAULT_CACHE_SIZE` (:66), passed by `graph_rdb_load` (:97) and `graphmeta_rdb_load`
(:856); unchanged by #3161, which only dropped the placeholder graph `create_virtual_keys` built with it. -/
def DEFAULT_CACHE_SIZE : Nat := 25

/-- The plan-cache size an RDB-loaded graph gets: the constant, not `CACHE_SIZE`. Graphs
created by `GRAPH.QUERY` / `GRAPH.COPY` / effects get `CONFIGURATION_CACHE_SIZE`
(`commands/query.rs:129`, `copy.rs`, `effect.rs:47`). -/
def cacheOnLoad (_configured : Nat) : Nat := DEFAULT_CACHE_SIZE
def cacheOnCreate (configured : Nat) : Nat := configured

/-- **After a restart (or `DEBUG RELOAD`, or a replica's full sync) every graph's plan cache
holds 25 entries whatever `CACHE_SIZE` was set to** — C reads the config for loaded graphs
too. Same graph, two cache sizes, depending on whether it was loaded or created. -/
theorem cache_size_ignored (configured : Nat) (h : configured ≠ 25) :
    cacheOnLoad configured ≠ cacheOnCreate configured := by
  simp [cacheOnLoad, cacheOnCreate, DEFAULT_CACHE_SIZE]; omega

end GraphPersist.RedisType
