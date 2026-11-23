import GraphPersist.Persist.Payload
import GraphPersist.RedisType.Db
/-! # Virtual keys on save (redis_type.rs)

| here | there |
| --- | --- |
| `RegEntry`, `regGet`      | a `GRAPH_REGISTRY` entry (graph_core.rs `RegisteredGraph`, keyed by `graph_id`, #3424): the graph, its id and its location `(db, name)`; `regGet` = `registry.get(&id)` |
| `KeySlice`, `VS`, `lookup` | `KeySlice` and `VirtualKeyState { vkey_map, graph_vkeys }` (serializers/mod.rs), both keyed by `(db, name)` (`DbKey`); `HashMap::insert` overwrites, so a lookup sees the last insert |
| `createGraphVKeys`        | `create_graph_virtual_keys`: one graph, `d` = the database `ctx` has selected; returns the state and the Redis keys it opened |
| `createVKeys`             | `create_virtual_keys`: each registered graph through `in_db(ctx, db, …)` (`Db.inDb`); `vname g i` stands for the `i`-th name `{g}g_<uuid>` (see `Uuid`) |
| `saveKeySlice`            | `save_key_slice(rdb, db, key_name, owner)` |
| `rdbSave`, `SaveOut`      | `graph_rdb_save`: owner = `Some(graph_id)` of the value it saves |
| `metaSave`, `vkeyType`    | `graphmeta_rdb_save`: owner = `None`; virtual keys typed `graphmeta` (#3161) |
| `deleteVKeys`             | `delete_virtual_keys`: each graph's names deleted through `in_db(ctx, db, …)` |

`key_holds_graph` (`RedisLayer.Telemetry.keyHolds` in proofs/redis_layer) is the input
`holds d name graph` (asked with database `d` selected), and `delete_key` /
`RedisModule_OpenKey` are `RedisModule_*` effects, recorded as the `(db, name)` they touch;
`build_multi_key_payloads` is the proved `buildMulti`, abstracted here as `layout`.
`create_virtual_keys` first sweeps stale graphmeta keys (`Sweep.deleteStaleGraphmeta`).
Every `in_db` restores the selection (`Db.inDb_spec`), so each graph's call starts from the
same `sel`.

* `save_main` / `save_vkey` / `save_single` — with distinct `(db, name)` locations, a
  graph needing `K ≥ 2` keys is saved as `K` keys of its own database: its own key holds
  slice 0 and the `i`-th virtual key slice `i`, each stamped `key_count = K`; any other key
  is saved whole. `save_same_name_two_dbs` — two graphs of one name in different databases
  each get their own slices (#2782).
* `rdbSave_owner_mismatch` — a graph key whose `(db, name)` carries another graph's slice
  (a lossy non-UTF-8 name) is saved whole, not as that slice.
* `metaSave_of_slice` / `metaSave_rdbSave` — `graphmeta_rdb_save` writes exactly the slice
  `graph_rdb_save` would, for the graph that owns the slice; `metaSave_rdbSave_fails` — not
  for any other owner; `metaSave_stale` — a leftover graphmeta key writes nothing.
* `deleteVKeys_spec`, `createdKeys_spec` — the end of a save deletes exactly the virtual
  keys it created, each in its graph's database.
-/
namespace GraphPersist.RedisType
open GraphPersist Persist

variable {Nm Gr : Type} [DecidableEq Nm]

/-- A `GRAPH_REGISTRY` entry: `graph_id` (the `Arc`'s address), and `RegisteredGraph { graph,
db, name }`. -/
structure RegEntry (Nm Gr : Type) where
  id : Nat
  db : Int
  name : Nm
  graph : Gr

/-- `KeySlice { graph_id, graph_name, payloads }`. -/
structure KeySlice (Nm : Type) where
  graphId : Nat
  graphName : Nm
  payloads : List Entry

structure VS (Nm : Type) where
  map : List ((Int × Nm) × KeySlice Nm)
  gv : List ((Int × Nm) × List Nm)

/-- `HashMap::get` after a sequence of inserts: the last one for the key. -/
def lookup {α β : Type} [DecidableEq α] (l : List (α × β)) (k : α) : Option β :=
  (l.reverse.find? (fun p => p.1 = k)).map Prod.snd

/-- `GRAPH_REGISTRY.get(&id).graph`. -/
def regGet (registry : List (RegEntry Nm Gr)) (id : Nat) : Option Gr :=
  lookup (registry.map fun e => (e.id, e.graph)) id

/-- The virtual names of graph `e`'s keys `1..K`. -/
def vkNamesOf (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (e : RegEntry Nm Gr) : List Nm :=
  (List.range ((layout e.graph).length - 1)).map fun i => vname e.name (i + 1)

/-- The `vkey_map` inserts of `create_graph_virtual_keys` with `d` selected: nothing unless
the key holds the graph and it needs `K ≥ 2` keys; then slice 0 under the graph's own
`(db, name)` and slices `1..K` under `(db, virtual name)`. -/
def vkInserts (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (d : Int) (e : RegEntry Nm Gr) : List ((Int × Nm) × KeySlice Nm) :=
  if holds d e.name e.graph = true ∧ 2 ≤ (layout e.graph).length then
    ((e.db, e.name), ⟨e.id, e.name, (layout e.graph).getD 0 []⟩) ::
      ((List.range ((layout e.graph).length - 1)).map fun i =>
        ((e.db, vname e.name (i + 1)), ⟨e.id, e.name, (layout e.graph).getD (i + 1) []⟩))
  else []

/-- Its `graph_vkeys` insert. -/
def vkNames (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (d : Int) (e : RegEntry Nm Gr) : List ((Int × Nm) × List Nm) :=
  if holds d e.name e.graph = true ∧ 2 ≤ (layout e.graph).length then [((e.db, e.name), vkNamesOf layout vname e)]
  else []

/-- The Redis keys it opens and fills (`OpenKey` + `ModuleTypeSetValue`), in the selected
database `d`. -/
def vkCreated (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (d : Int) (e : RegEntry Nm Gr) : List (Int × Nm) :=
  if holds d e.name e.graph = true ∧ 2 ≤ (layout e.graph).length then (vkNamesOf layout vname e).map (d, ·)
  else []

/-- `create_graph_virtual_keys(ctx, vkey_state, vkey_max, (graph_id, db, graph_name, graph_ref))`,
`d` = the database `ctx` has selected. -/
def createGraphVKeys (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (acc : VS Nm × List (Int × Nm)) (d : Int) (e : RegEntry Nm Gr) : VS Nm × List (Int × Nm) :=
  (⟨acc.1.map ++ vkInserts holds layout vname d e, acc.1.gv ++ vkNames holds layout vname d e⟩,
   acc.2 ++ vkCreated holds layout vname d e)

/-- `create_virtual_keys`: `vkey_state.clear()`, then `let _ = in_db(ctx, db, ||
create_graph_virtual_keys(..))` for each registry entry. Returns the state and the keys
created. -/
def createVKeys (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) : VS Nm × List (Int × Nm) :=
  registry.foldl (fun acc e => ((inDb ok sel e.db fun d => createGraphVKeys holds layout vname acc d e).1).getD acc)
    (⟨[], []⟩, [])

/-- One graph's part of `create_virtual_keys`: `create_graph_virtual_keys` in its own database,
or nothing when that database cannot be selected. -/
def insOf (ok : Int → Bool) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (e : RegEntry Nm Gr) := if ok e.db then vkInserts holds layout vname e.db e else []
def nmsOf (ok : Int → Bool) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (e : RegEntry Nm Gr) := if ok e.db then vkNames holds layout vname e.db e else []
def crOf (ok : Int → Bool) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (e : RegEntry Nm Gr) := if ok e.db then vkCreated holds layout vname e.db e else []

omit [DecidableEq Nm] in
theorem createVKeys_foldl (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool)
    (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) :
    ∀ acc : VS Nm × List (Int × Nm),
      registry.foldl (fun acc e => ((inDb ok sel e.db fun d => createGraphVKeys holds layout vname acc d e).1).getD acc) acc =
        (⟨acc.1.map ++ registry.flatMap (insOf ok holds layout vname),
          acc.1.gv ++ registry.flatMap (nmsOf ok holds layout vname)⟩,
         acc.2 ++ registry.flatMap (crOf ok holds layout vname)) := by
  induction registry with
  | nil => intro acc; simp
  | cons e es ih =>
    intro acc
    simp only [List.foldl_cons]
    rw [ih, inDb_fst]
    by_cases h : ok e.db <;> simp [h, createGraphVKeys, insOf, nmsOf, crOf]

omit [DecidableEq Nm] in
theorem createVKeys_eq (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool)
    (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) :
    createVKeys ok sel holds layout vname registry =
      (⟨registry.flatMap (insOf ok holds layout vname), registry.flatMap (nmsOf ok holds layout vname)⟩,
       registry.flatMap (crOf ok holds layout vname)) := by
  simp only [createVKeys]; rw [createVKeys_foldl]; simp

/-- What `graph_rdb_save` writes for one key. -/
inductive SaveOut (Nm Gr : Type) where
  /-- `rdb_save_graph_key(rdb, graph, graph_name, payloads, key_count)` -/
  | slice (g : Nm) (gr : Gr) (payloads : List Entry) (keyCount : Nat)
  /-- `rdb_save_graph(rdb, graph, key_name)`: the key's own value, whole -/
  | whole (key : Nm)

/-- `save_key_slice(rdb, db, key_name, owner)`: the slice `create_virtual_keys` assigned key
`key_name` of database `db`, unless there is none, `owner` is another graph than the slice's,
or the slice's graph is no longer registered (`false` = `none`). `reg` is
`GRAPH_REGISTRY.get(&id)`. -/
def saveKeySlice (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (owner : Option Nat) :
    Option (SaveOut Nm Gr) :=
  if vs.map.isEmpty then none else
  match lookup vs.map (db, key) with
  | none => none
  | some s =>
    if owner.any (fun id => decide (id ≠ s.graphId)) then none else
    match reg s.graphId with
    | some gr => some (.slice s.graphName gr s.payloads (1 + ((lookup vs.gv (db, s.graphName)).map List.length).getD 0))
    | none => none

/-- `graph_rdb_save`: the key's slice if it has one for the graph `id` the key holds, else
the whole graph. -/
def rdbSave (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (id : Nat) : SaveOut Nm Gr :=
  (saveKeySlice vs reg db key (some id)).getD (.whole key)

/-- `graphmeta_rdb_save`: a virtual key — typed `graphmeta` since #3161, as C types its own —
encodes its slice; a graphmeta key this save did not create writes nothing (`none`). -/
def metaSave (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) : Option (SaveOut Nm Gr) :=
  saveKeySlice vs reg db key none

/-- The type `create_graph_virtual_keys` gives a virtual key: `graphmeta`, holding a dummy
byte, not a `graphdata` placeholder graph — C frees a `graphdata` key by dropping its graph
from `GRAPH.LIST`, which made a Rust-saved multi-key graph vanish from C's list (#3160). -/
def vkeyType : String := "graphmeta"

theorem lookup_nil {α β : Type} [DecidableEq α] (k : α) : lookup ([] : List (α × β)) k = none := rfl

theorem saveKeySlice_some (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (owner : Option Nat)
    (s : KeySlice Nm) (gr : Gr) (hs : lookup vs.map (db, key) = some s) (ho : ∀ id, owner = some id → id = s.graphId)
    (hr : reg s.graphId = some gr) :
    saveKeySlice vs reg db key owner =
      some (.slice s.graphName gr s.payloads (1 + ((lookup vs.gv (db, s.graphName)).map List.length).getD 0)) := by
  have hne : vs.map.isEmpty = false := by
    cases hm : vs.map with
    | nil => rw [hm, lookup_nil] at hs; cases hs
    | cons _ _ => rfl
  have hany : owner.any (fun id => decide (id ≠ s.graphId)) = false := by
    cases owner with
    | none => rfl
    | some id => simp [ho id rfl]
  simp only [saveKeySlice, hne, hs, hany, hr]; rfl

theorem saveKeySlice_none (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (owner : Option Nat)
    (hs : lookup vs.map (db, key) = none) : saveKeySlice vs reg db key owner = none := by
  unfold saveKeySlice; split
  · rfl
  · rw [hs]

/-- **A graph key never takes another graph's slice.** When the key's `(db, name)` was
assigned a slice of a graph other than the one the key holds — a graph under a non-UTF-8
key whose lossy name is another, split, graph's key — `graph_rdb_save` writes the key's own
graph whole. -/
theorem rdbSave_owner_mismatch (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (id : Nat)
    (s : KeySlice Nm) (hs : lookup vs.map (db, key) = some s) (hid : id ≠ s.graphId) :
    rdbSave vs reg db key id = .whole key := by
  have hne : vs.map.isEmpty = false := by
    cases hm : vs.map with
    | nil => rw [hm, lookup_nil] at hs; cases hs
    | cons _ _ => rfl
  simp [rdbSave, saveKeySlice, hne, hs, hid]

/-- Both save callbacks agree on a key that has a slice, for the graph that slice belongs to. -/
theorem metaSave_rdbSave (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm) (id : Nat) (o : SaveOut Nm Gr)
    (h : metaSave vs reg db key = some o) (hown : ∀ s, lookup vs.map (db, key) = some s → s.graphId = id) :
    rdbSave vs reg db key id = o := by
  unfold metaSave at h; unfold rdbSave
  cases hs : lookup vs.map (db, key) with
  | none => rw [saveKeySlice_none _ _ _ _ _ hs] at h; cases h
  | some s =>
    cases hr : reg s.graphId with
    | none =>
      have hne : vs.map.isEmpty = false := by
        cases hm : vs.map with
        | nil => rw [hm, lookup_nil] at hs; cases hs
        | cons _ _ => rfl
      simp [saveKeySlice, hne, hs, hr] at h
    | some gr =>
      rw [saveKeySlice_some _ _ _ _ _ s gr hs (fun _ h => by cases h) hr] at h
      rw [saveKeySlice_some _ _ _ _ _ s gr hs (fun i hi => by cases hi; exact (hown s hs).symm) hr, ← Option.some.inj h]
      rfl

/-- **Counterexample to `metaSave_rdbSave` without the owner premise** (#3424): the two
callbacks no longer agree on every key with a slice — `graph_rdb_save` on a key holding
another graph than the slice's writes that graph whole (`rdbSave_owner_mismatch`). -/
theorem metaSave_rdbSave_fails :
    ¬ ∀ (vs : VS String) (reg : Nat → Option Nat) (db : Int) (key : String) (id : Nat) (o : SaveOut String Nat),
      metaSave vs reg db key = some o → rdbSave vs reg db key id = o := by
  intro hall
  have h := hall ⟨[((0, "g"), ⟨1, "g", []⟩)], []⟩ (fun _ => some 0) 0 "g" 2 (.slice "g" 0 [] 1)
    (by simp [metaSave, saveKeySlice, lookup])
  simp [rdbSave, saveKeySlice, lookup] at h

/-- Conversely: whatever slice `graph_rdb_save` would write for a key, the `graphmeta`
callback writes too — so `save_main`/`save_vkey` hold for a virtual key saved through
`graphmeta_rdb_save`, which is how Redis saves it since #3161. -/
theorem metaSave_of_slice (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key g : Nm) (id : Nat) (gr : Gr)
    (pl : List Entry) (k : Nat) (h : rdbSave vs reg db key id = .slice g gr pl k) :
    metaSave vs reg db key = some (.slice g gr pl k) := by
  unfold rdbSave at h; unfold metaSave
  cases hs : lookup vs.map (db, key) with
  | none => rw [saveKeySlice_none _ _ _ _ _ hs] at h; cases h
  | some s =>
    have hne : vs.map.isEmpty = false := by
      cases hm : vs.map with
      | nil => rw [hm, lookup_nil] at hs; cases hs
      | cons _ _ => rfl
    cases hr : reg s.graphId with
    | none => simp [saveKeySlice, hne, hs, hr] at h
    | some gr' =>
      by_cases hid : id = s.graphId
      · rw [saveKeySlice_some _ _ _ _ _ s gr' hs (fun i hi => by cases hi; exact hid) hr] at h
        rw [saveKeySlice_some _ _ _ _ _ s gr' hs (fun _ h => by cases h) hr]
        simp only [Option.getD_some] at h; rw [h]
      · simp [saveKeySlice, hne, hs, hid] at h

/-- A graphmeta key with no slice assigned writes nothing. -/
theorem metaSave_stale (vs : VS Nm) (reg : Nat → Option Gr) (db : Int) (key : Nm)
    (h : lookup vs.map (db, key) = none) : metaSave vs reg db key = none :=
  saveKeySlice_none _ _ _ _ _ h

/-! ## Lookups under distinct keys -/

theorem lookup_unique {α β : Type} [DecidableEq α] (l : List (α × β)) (k : α) (v : β) (hmem : (k, v) ∈ l)
    (huniq : ∀ p ∈ l, p.1 = k → p = (k, v)) : lookup l k = some v := by
  unfold lookup
  have hex : ∃ p ∈ l.reverse, (fun p : α × β => decide (p.1 = k)) p = true := ⟨(k, v), by simpa using hmem, by simp⟩
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 (List.find?_isSome.2 hex)
  have hpm := List.mem_of_find?_eq_some hp
  have hpk := List.find?_some hp
  simp at hpk
  rw [hp, huniq p (by simpa using hpm) hpk]; rfl

theorem lookup_none {α β : Type} [DecidableEq α] (l : List (α × β)) (k : α) (h : ∀ p ∈ l, p.1 ≠ k) :
    lookup l k = none := by
  unfold lookup
  rw [List.find?_eq_none.2 (by intro p hp; simpa using h p (by simpa using hp))]; rfl

theorem nodup_map_inj {α β : Type} {f : α → β} {l : List α} (h : (l.map f).Nodup) {a b : α}
    (ha : a ∈ l) (hb : b ∈ l) (he : f a = f b) : a = b := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at h
    rcases List.mem_cons.1 ha with ha1 | ha1 <;> rcases List.mem_cons.1 hb with hb1 | hb1
    · rw [ha1, hb1]
    · exact absurd ⟨b, hb1, by rw [← he, ha1]⟩ h.1
    · exact absurd ⟨a, ha1, by rw [he, hb1]⟩ h.1
    · exact ih h.2 ha1 hb1

/-- The naming assumptions: registry ids distinct (a `HashMap` key), registry locations
`(db, name)` distinct (one key holds one value; the same name in two databases is allowed),
virtual names injective, and no virtual name equal to a graph name. -/
structure Names (registry : List (RegEntry Nm Gr)) (vname : Nm → Nat → Nm) : Prop where
  id_nodup : (registry.map (·.id)).Nodup
  loc_nodup : (registry.map fun e => (e.db, e.name)).Nodup
  vname_inj : ∀ g i g' j, vname g i = vname g' j → g = g' ∧ i = j
  vname_fresh : ∀ g i, ∀ e ∈ registry, vname g i ≠ e.name

omit [DecidableEq Nm] in
theorem mem_insOf {ok : Int → Bool} {holds : Int → Nm → Gr → Bool} {layout : Gr → List (List Entry)}
    {vname : Nm → Nat → Nm} {e : RegEntry Nm Gr} {p : (Int × Nm) × KeySlice Nm}
    (h : p ∈ insOf ok holds layout vname e) :
    p = ((e.db, e.name), ⟨e.id, e.name, (layout e.graph).getD 0 []⟩) ∨
      ∃ i, i + 1 < (layout e.graph).length ∧
        p = ((e.db, vname e.name (i + 1)), ⟨e.id, e.name, (layout e.graph).getD (i + 1) []⟩) := by
  unfold insOf vkInserts at h
  split at h
  · split at h
    · simp only [List.mem_cons, List.mem_map, List.mem_range] at h
      rcases h with rfl | ⟨i, hi, rfl⟩
      · exact Or.inl rfl
      · exact Or.inr ⟨i, by omega, rfl⟩
    · simp at h
  · simp at h

omit [DecidableEq Nm] in
theorem regGet_mem (registry : List (RegEntry Nm Gr)) (hid : (registry.map (·.id)).Nodup)
    (e : RegEntry Nm Gr) (he : e ∈ registry) : regGet registry e.id = some e.graph := by
  apply lookup_unique
  · exact List.mem_map.2 ⟨e, he, rfl⟩
  · intro q hq hqe
    obtain ⟨e', he', rfl⟩ := List.mem_map.1 hq
    rw [nodup_map_inj hid he' he hqe]

theorem gv_lookup (ok : Int → Bool) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) :
    lookup (registry.flatMap (nmsOf ok holds layout vname)) (e.db, e.name) = some (vkNamesOf layout vname e) := by
  apply lookup_unique
  · exact List.mem_flatMap.2 ⟨e, he, by simp [nmsOf, vkNames, hok, hh, hK]⟩
  · intro q hq hqe
    obtain ⟨e', he', hq⟩ := List.mem_flatMap.1 hq
    simp only [nmsOf, vkNames] at hq
    split at hq
    · split at hq
      · simp only [List.mem_singleton] at hq; subst hq
        rw [nodup_map_inj hn.loc_nodup he' he hqe]
      · simp at hq
    · simp at hq

/-- The slice at `(e.db, key)` and the save of it, for any owner that is `e` or none. -/
theorem save_at (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) (key : Nm) (i : Nat)
    (hslot : lookup (createVKeys ok sel holds layout vname registry).1.map (e.db, key) =
      some ⟨e.id, e.name, (layout e.graph).getD i []⟩)
    (owner : Option Nat) (ho : ∀ id, owner = some id → id = e.id) :
    saveKeySlice (createVKeys ok sel holds layout vname registry).1 (regGet registry) e.db key owner =
      some (.slice e.name e.graph ((layout e.graph).getD i []) (layout e.graph).length) := by
  rw [saveKeySlice_some _ _ _ _ _ _ e.graph hslot ho (regGet_mem registry hn.id_nodup e he)]
  rw [createVKeys_eq] at hslot ⊢
  simp only [gv_lookup ok holds layout vname registry hn e he hok hh hK, Option.map_some, Option.getD_some,
    vkNamesOf, List.length_map, List.length_range]
  congr 2; omega

/-- **The graph's own key holds slice 0.** -/
theorem save_main (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) :
    rdbSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) e.db e.name e.id =
      .slice e.name e.graph ((layout e.graph).getD 0 []) (layout e.graph).length := by
  have hmap : lookup (createVKeys ok sel holds layout vname registry).1.map (e.db, e.name) =
      some ⟨e.id, e.name, (layout e.graph).getD 0 []⟩ := by
    rw [createVKeys_eq]
    apply lookup_unique
    · exact List.mem_flatMap.2 ⟨e, he, by simp [insOf, vkInserts, hok, hh, hK]⟩
    · intro p hp hpk
      obtain ⟨e', he', hp⟩ := List.mem_flatMap.1 hp
      rcases mem_insOf hp with rfl | ⟨i, _, rfl⟩
      · simp only at hpk; rw [nodup_map_inj hn.loc_nodup he' he hpk]
      · simp only [Prod.mk.injEq] at hpk; exact absurd hpk.2 (hn.vname_fresh _ _ e he)
  simp only [rdbSave, save_at ok sel holds layout vname registry hn e he hok hh hK e.name 0 hmap
    (some e.id) (fun _ h => (Option.some.inj h).symm), Option.getD_some]

/-- The `i`-th virtual key's slot. -/
theorem vkey_slot (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout e.graph).length) :
    lookup (createVKeys ok sel holds layout vname registry).1.map (e.db, vname e.name i) =
      some ⟨e.id, e.name, (layout e.graph).getD i []⟩ := by
  rw [createVKeys_eq]
  apply lookup_unique
  · refine List.mem_flatMap.2 ⟨e, he, ?_⟩
    simp only [insOf, vkInserts, hok, hh, hK, true_and, ite_true, List.mem_cons, List.mem_map,
      List.mem_range]
    exact Or.inr ⟨i - 1, by omega, by rw [Nat.sub_add_cancel hi]⟩
  · intro p hp hpk
    obtain ⟨e', he', hp⟩ := List.mem_flatMap.1 hp
    rcases mem_insOf hp with rfl | ⟨j, _, rfl⟩
    · simp only [Prod.mk.injEq] at hpk; exact absurd hpk.2.symm (hn.vname_fresh _ _ e' he')
    · simp only [Prod.mk.injEq] at hpk
      obtain ⟨hdb, hv⟩ := hpk
      obtain ⟨hnm, hji⟩ := hn.vname_inj _ _ _ _ hv
      have := nodup_map_inj hn.loc_nodup he' he (by simp [hdb, hnm])
      subst this; subst hji; rfl

/-- **The `i`-th virtual key holds slice `i`** — as `graph_rdb_save` would write it for the
graph's id … -/
theorem save_vkey (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout e.graph).length) :
    rdbSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) e.db (vname e.name i) e.id =
      .slice e.name e.graph ((layout e.graph).getD i []) (layout e.graph).length := by
  simp only [rdbSave, save_at ok sel holds layout vname registry hn e he hok hh hK _ i
    (vkey_slot ok sel holds layout vname registry hn e he hok hh hK i hi hiK)
    (some e.id) (fun _ h => (Option.some.inj h).symm), Option.getD_some]

/-- … and as `graphmeta_rdb_save`, the callback that saves a `graphmeta` virtual key, writes it. -/
theorem save_vkey_meta (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (hn : Names registry vname)
    (e : RegEntry Nm Gr) (he : e ∈ registry) (hok : ok e.db = true) (hh : holds e.db e.name e.graph = true)
    (hK : 2 ≤ (layout e.graph).length) (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout e.graph).length) :
    metaSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) e.db (vname e.name i) =
      some (.slice e.name e.graph ((layout e.graph).getD i []) (layout e.graph).length) :=
  metaSave_of_slice _ _ _ _ _ _ _ _ _ (save_vkey ok sel holds layout vname registry hn e he hok hh hK i hi hiK)

/-- **Everything else is saved whole.** A key that is, in its database, neither a split
graph's own key nor one of its virtual keys is written by `rdb_save_graph` under its own name. -/
theorem save_single (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool) (layout : Gr → List (List Entry))
    (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (db : Int) (key : Nm) (id : Nat)
    (hk : ∀ e ∈ registry, ok e.db = true → holds e.db e.name e.graph = true → 2 ≤ (layout e.graph).length →
      db ≠ e.db ∨ (key ≠ e.name ∧ ∀ i, key ≠ vname e.name i)) :
    rdbSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) db key id = .whole key := by
  have : lookup (createVKeys ok sel holds layout vname registry).1.map (db, key) = none := by
    rw [createVKeys_eq]
    apply lookup_none
    intro p hp hpk
    obtain ⟨e, he, hp⟩ := List.mem_flatMap.1 hp
    have hc : ok e.db = true ∧ holds e.db e.name e.graph = true ∧ 2 ≤ (layout e.graph).length := by
      simp only [insOf, vkInserts] at hp
      split at hp
      · split at hp
        · rename_i h1 h2; exact ⟨h1, h2.1, h2.2⟩
        · simp at hp
      · simp at hp
    rcases mem_insOf hp with rfl | ⟨i, _, rfl⟩ <;> simp only [Prod.mk.injEq] at hpk
    · rcases hk e he hc.1 hc.2.1 hc.2.2 with h | h
      · exact h hpk.1.symm
      · exact h.1 hpk.2.symm
    · rcases hk e he hc.1 hc.2.1 hc.2.2 with h | h
      · exact h hpk.1.symm
      · exact h.2 _ hpk.2.symm
  simp [rdbSave, saveKeySlice_none _ _ _ _ _ this]

/-- **Two graphs of one name in two databases are each saved as their own slices** (#2782:
the registry and `vkey_map` were keyed by name, so one displaced the other). -/
theorem save_same_name_two_dbs (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool)
    (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr))
    (hn : Names registry vname) (a b : RegEntry Nm Gr) (ha : a ∈ registry) (hb : b ∈ registry)
    (hoka : ok a.db = true) (hokb : ok b.db = true)
    (hha : holds a.db a.name a.graph = true) (hhb : holds b.db b.name b.graph = true)
    (hKa : 2 ≤ (layout a.graph).length) (hKb : 2 ≤ (layout b.graph).length) (_hname : a.name = b.name) :
    rdbSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) a.db a.name a.id =
      .slice a.name a.graph ((layout a.graph).getD 0 []) (layout a.graph).length ∧
    rdbSave (createVKeys ok sel holds layout vname registry).1 (regGet registry) b.db b.name b.id =
      .slice b.name b.graph ((layout b.graph).getD 0 []) (layout b.graph).length :=
  ⟨save_main ok sel holds layout vname registry hn a ha hoka hha hKa,
   save_main ok sel holds layout vname registry hn b hb hokb hhb hKb⟩

/-! ## End of save: delete what was created -/

/-- `delete_virtual_keys`: for each `((db, graph_name), names)`, `in_db(ctx, db, ||
delete_key(..) for each name)`, then clear the state. Returns the keys deleted. -/
def deleteVKeys (ok : Int → Bool) (sel : Int) (vs : VS Nm) : List (Int × Nm) × VS Nm :=
  (vs.gv.flatMap (fun p => ((inDb ok sel p.1.1 fun d => p.2.map (d, ·)).1).getD []), ⟨[], []⟩)

omit [DecidableEq Nm] in
/-- **The end of a save deletes exactly the keys `create_virtual_keys` created**, each in the
database it was created in, and clears the state. -/
theorem deleteVKeys_spec (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool)
    (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) :
    (deleteVKeys ok sel (createVKeys ok sel holds layout vname registry).1).1 =
        (createVKeys ok sel holds layout vname registry).2 ∧
    (deleteVKeys ok sel (createVKeys ok sel holds layout vname registry).1).2 = ⟨[], []⟩ := by
  refine ⟨?_, rfl⟩
  rw [createVKeys_eq]
  simp only [deleteVKeys, List.flatMap_assoc]
  congr 1; funext e
  by_cases hok : ok e.db <;> by_cases hh : holds e.db e.name e.graph <;>
    by_cases hK : 2 ≤ (layout e.graph).length <;> simp [hok, hh, hK, nmsOf, crOf, vkNames, vkCreated, inDb_fst]

omit [DecidableEq Nm] in
/-- **Exactly the virtual names of every split graph, in its own database.** -/
theorem createdKeys_spec (ok : Int → Bool) (sel : Int) (holds : Int → Nm → Gr → Bool)
    (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm) (registry : List (RegEntry Nm Gr)) (d : Int) (k : Nm) :
    (d, k) ∈ (createVKeys ok sel holds layout vname registry).2 ↔
      ∃ e ∈ registry, ok e.db = true ∧ holds e.db e.name e.graph = true ∧ 2 ≤ (layout e.graph).length ∧
        d = e.db ∧ ∃ i, 1 ≤ i ∧ i < (layout e.graph).length ∧ k = vname e.name i := by
  rw [createVKeys_eq]
  simp only [List.mem_flatMap]
  constructor
  · rintro ⟨e, he, hk⟩
    simp only [crOf, vkCreated] at hk
    split at hk
    · split at hk
      · rename_i hok hc
        simp only [vkNamesOf, List.map_map, List.mem_map, List.mem_range, Function.comp_def, Prod.mk.injEq] at hk
        obtain ⟨i, hi, rfl, rfl⟩ := hk
        exact ⟨e, he, hok, hc.1, hc.2, rfl, i + 1, by omega, by omega, rfl⟩
      · simp at hk
    · simp at hk
  · rintro ⟨e, he, hok, hh, hK, rfl, i, hi1, hi2, rfl⟩
    refine ⟨e, he, ?_⟩
    simp only [crOf, vkCreated, hok, hh, hK, true_and, ite_true, vkNamesOf, List.map_map,
      List.mem_map, List.mem_range, Function.comp_def, Prod.mk.injEq]
    exact ⟨i - 1, by omega, by rw [Nat.sub_add_cancel hi1]⟩

end GraphPersist.RedisType
