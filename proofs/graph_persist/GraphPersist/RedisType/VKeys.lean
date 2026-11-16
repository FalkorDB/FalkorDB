import GraphPersist.Persist.Payload
/-! # Virtual keys on save (redis_type.rs:245-300, :494-670, :702-756, :1089-1099)

Since #3424 (#2782) every name here is a `DbKey = (db, name)` (serializers/mod.rs:37): `Nm`
stands for it, and `vname (db, g) i` is the `i`-th virtual key `(db, {g}g_<uuid>)`, created
in the graph's own database (`in_db`). Graphs are identified by `graph_id` (`Id`).

| here | there |
| --- | --- |
| `VS`, `lookup`            | `VirtualKeyState { vkey_map, graph_vkeys }` (serializers/mod.rs:42-50), `KeySlice { graph_id, graph_name, payloads }`; `HashMap::insert` overwrites, so a lookup sees the last insert |
| `createVKeys`             | `create_virtual_keys` (:499) over the registry's `(id, (db, name), graph)`, each through `create_graph_virtual_keys` (:530) |
| `saveKeySlice`            | `save_key_slice` (:272, #3161; `owner` check #3424) |
| `rdbSave`, `SaveOut`      | `graph_rdb_save` (:245), `owner = Some(graph_id)` of the key's value |
| `metaSave`, `vkeyType`    | `graphmeta_rdb_save` (:1089, `owner = None`); virtual keys typed `graphmeta` (#3161) |
| `deleteVKeys`             | `delete_virtual_keys` (:638) |
| `scanKeys`, `Reply`       | `scan_keys_by_type` (:759), one `SCAN … TYPE` reply per step |
| `deleteStaleGraphmeta`    | `delete_stale_graphmeta_keys` (:702), one scan per non-empty db (`for_each_db`) |
| `deleteStaleVirtual`      | `delete_stale_virtual_keys` (:738), `meta_keys : Vec<DbKey>` |

`key_holds_graph` and `delete_key` are `RedisModule_*` calls (inputs / effects);
`build_multi_key_payloads` is the proved `buildMulti`, abstracted here as `layout`.

* `save_main` / `save_vkey` / `save_single` — with distinct names and ids, a graph needing
  `K ≥ 2` keys is saved as `K` keys: its own key holds slice 0 and the `i`-th virtual key
  slice `i`, each stamped `key_count = K`; any other graph is saved whole under its key.
* `save_other_owner` — a key whose value is another graph than the slice's (same lossy
  name) is saved whole, never as that slice (#3424).
* `metaSave_of_slice` / `metaSave_rdbSave` — a virtual key is written by
  `graphmeta_rdb_save`, which writes exactly the slice `graph_rdb_save` would;
  `metaSave_stale` — a leftover graphmeta key writes nothing.
* `deleteVKeys_spec` — the end of a save deletes exactly the virtual keys it created.
* `scanKeys_spec`, `deleteStaleGraphmeta_spec`, `deleteStaleVirtual_spec`.
-/
namespace GraphPersist.RedisType
open GraphPersist Persist

variable {Nm Id Gr : Type} [DecidableEq Nm] [DecidableEq Id]
set_option linter.unusedSectionVars false

/-- `vkey_map` values are `(graph_id, graph name, payloads)`. -/
structure VS (Id Nm : Type) where
  map : List (Nm × (Id × Nm × List Entry))
  gv : List (Nm × List Nm)

/-- `HashMap::get` after a sequence of inserts: the last one for the key. -/
def lookup {κ β : Type} [DecidableEq κ] (l : List (κ × β)) (k : κ) : Option β :=
  (l.reverse.find? (fun p => p.1 = k)).map Prod.snd

/-- What one graph contributes (`create_graph_virtual_keys`, :530): nothing unless its key
holds it and it needs `K ≥ 2` keys; then key 0 under the graph's own name and keys `1..K`
under virtual names. -/
def vkInserts (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (id : Id) (g : Nm) (gr : Gr) : List (Nm × (Id × Nm × List Entry)) :=
  if holds g gr ∧ 2 ≤ (layout gr).length then
    (g, (id, g, (layout gr).getD 0 [])) ::
      ((List.range ((layout gr).length - 1)).map fun i => (vname g (i + 1), (id, g, (layout gr).getD (i + 1) [])))
  else []

def vkNames (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (g : Nm) (gr : Gr) : List (Nm × List Nm) :=
  if holds g gr ∧ 2 ≤ (layout gr).length then [(g, (List.range ((layout gr).length - 1)).map fun i => vname g (i + 1))]
  else []

/-- `create_virtual_keys`: `vkey_state.clear()`, then each registered `(id, (db, name), graph)`
in turn. -/
def createVKeys (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) : VS Id Nm :=
  ⟨registry.flatMap (fun p => vkInserts holds layout vname p.1 p.2.1 p.2.2),
   registry.flatMap (fun p => vkNames holds layout vname p.2.1 p.2.2)⟩

/-- The Redis keys `create_virtual_keys` opens and fills with a placeholder value. -/
def createdKeys (vs : VS Id Nm) : List Nm := vs.gv.flatMap Prod.snd

/-- What `graph_rdb_save` writes for one key. -/
inductive SaveOut (Nm Gr : Type) where
  /-- `rdb_save_graph_key(rdb, graph, graph_name, payloads, key_count)` -/
  | slice (g : Nm) (gr : Gr) (payloads : List Entry) (keyCount : Nat)
  /-- `rdb_save_graph(rdb, graph, key_name)`: the key's own value, whole -/
  | whole (key : Nm)

/-- `save_key_slice` (:272): the slice `create_virtual_keys` assigned `key`, if this save
assigned it one, `owner` (when given) is the slice's graph, and that graph is still
registered (`false` = `none`). `reg` is `GRAPH_REGISTRY.get` by id. -/
def saveKeySlice (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) (owner : Option Id) :
    Option (SaveOut Nm Gr) :=
  match lookup vs.map key with
  | some (id, g, payloads) =>
    if owner.any (· ≠ id) then none else
    match reg id with
    | some gr => some (.slice g gr payloads (1 + ((lookup vs.gv g).map List.length).getD 0))
    | none => none
  | none => none

/-- `graph_rdb_save` (:245): the key's slice if it has one for the graph `owner` the key
holds, else the whole graph. -/
def rdbSave (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) (owner : Id) : SaveOut Nm Gr :=
  (saveKeySlice vs reg key (some owner)).getD (.whole key)

/-- `graphmeta_rdb_save` (:1089): a virtual key — typed `graphmeta`, as C types its own —
encodes its slice (no owner: its value is a dummy byte); a graphmeta key this save did not
create writes nothing (`none`). -/
def metaSave (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) : Option (SaveOut Nm Gr) :=
  saveKeySlice vs reg key none

/-- The type `create_virtual_keys` gives a virtual key: `graphmeta`, holding a dummy byte,
not a `graphdata` placeholder graph (#3160). -/
def vkeyType : String := "graphmeta"

/-- The slice's graph id assigned to `key`, if any. -/
def sliceOwner (vs : VS Id Nm) (key : Nm) : Option Id := (lookup vs.map key).map (·.1)

/-- Both save callbacks agree on a key that has a slice, when `graph_rdb_save` is called
for the key's own graph. -/
theorem metaSave_rdbSave (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) (o : SaveOut Nm Gr)
    (h : metaSave vs reg key = some o) (id : Id) (hid : sliceOwner vs key = some id) :
    rdbSave vs reg key id = o := by
  simp only [metaSave, saveKeySlice] at h
  simp only [sliceOwner] at hid
  simp only [rdbSave, saveKeySlice]
  cases hl : lookup vs.map key with
  | none => rw [hl] at h; cases h
  | some v =>
    obtain ⟨i, g, pl⟩ := v
    rw [hl] at h hid; simp at hid; subst hid
    revert h; cases hr : reg i <;> simp [hr]

/-- Conversely: whatever slice `graph_rdb_save` would write for a key, the `graphmeta`
callback writes too — so `save_main`/`save_vkey` hold for a virtual key saved through
`graphmeta_rdb_save`. -/
theorem metaSave_of_slice (vs : VS Id Nm) (reg : Id → Option Gr) (key g : Nm) (id : Id) (gr : Gr)
    (pl : List Entry) (k : Nat) (h : rdbSave vs reg key id = .slice g gr pl k) :
    metaSave vs reg key = some (.slice g gr pl k) := by
  unfold rdbSave saveKeySlice at h; unfold metaSave saveKeySlice
  cases hl : lookup vs.map key with
  | none => rw [hl] at h; simp at h
  | some v =>
    obtain ⟨i, g', pl'⟩ := v
    rw [hl] at h; simp only [Option.any_some, Option.any_none, Bool.false_eq_true, if_false]
    revert h; by_cases hi : id = i <;> cases hr : reg i <;> simp [hi, hr]

/-- A graphmeta key with no slice assigned writes nothing. -/
theorem metaSave_stale (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) (h : lookup vs.map key = none) :
    metaSave vs reg key = none := by simp [metaSave, saveKeySlice, h]

/-- **A key holding another graph than its slice's is saved whole** (#3424): two keys of
different bytes can share a lossy name, and the slice is only ever written for its own graph. -/
theorem save_other_owner (vs : VS Id Nm) (reg : Id → Option Gr) (key : Nm) (owner id : Id)
    (h : sliceOwner vs key = some id) (hne : owner ≠ id) :
    rdbSave vs reg key owner = .whole key := by
  simp only [sliceOwner] at h
  simp only [rdbSave, saveKeySlice]
  cases hl : lookup vs.map key with
  | none => rw [hl] at h; cases h
  | some v =>
    obtain ⟨i, g, pl⟩ := v
    rw [hl] at h; simp at h; subst h
    simp [hne]

/-! ## Lookups under distinct keys -/

theorem lookup_unique {κ β : Type} [DecidableEq κ] (l : List (κ × β)) (k : κ) (v : β) (hmem : (k, v) ∈ l)
    (huniq : ∀ p ∈ l, p.1 = k → p = (k, v)) : lookup l k = some v := by
  unfold lookup
  have hex : ∃ p ∈ l.reverse, (fun p : κ × β => decide (p.1 = k)) p = true := ⟨(k, v), by simpa using hmem, by simp⟩
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 (List.find?_isSome.2 hex)
  have hpm := List.mem_of_find?_eq_some hp
  have hpk := List.find?_some hp
  simp at hpk
  rw [hp, huniq p (by simpa using hpm) hpk]; rfl

theorem lookup_none {κ β : Type} [DecidableEq κ] (l : List (κ × β)) (k : κ) (h : ∀ p ∈ l, p.1 ≠ k) :
    lookup l k = none := by
  unfold lookup
  rw [List.find?_eq_none.2 (by intro p hp; simpa using h p (by simpa using hp))]; rfl

theorem nodup_key_eq {α κ : Type} (f : α → κ) {l : List α} (h : (l.map f).Nodup) {a b : α}
    (ha : a ∈ l) (hb : b ∈ l) (he : f a = f b) : a = b := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at h
    have ha' := List.mem_cons.1 ha
    have hb' := List.mem_cons.1 hb
    rcases ha' with ha1 | ha1 <;> rcases hb' with hb1 | hb1
    · rw [ha1, hb1]
    · exact absurd ⟨b, hb1, by rw [← he, ha1]⟩ h.1
    · exact absurd ⟨a, ha1, by rw [he, hb1]⟩ h.1
    · exact ih h.2 ha1 hb1

/-- `GRAPH_REGISTRY.get` by id over the registry's entries. -/
def regGet (registry : List (Id × Nm × Gr)) : Id → Option Gr :=
  lookup (registry.map fun p => (p.1, p.2.2))

/-- The naming assumptions: registry ids distinct (a `HashMap` key), `(db, name)`s distinct
(one graph per key of a database), virtual names injective, and no virtual name equal to a
graph's key. -/
structure Names (registry : List (Id × Nm × Gr)) (vname : Nm → Nat → Nm) : Prop where
  id_nodup : (registry.map (·.1)).Nodup
  reg_nodup : (registry.map (·.2.1)).Nodup
  vname_inj : ∀ g i g' j, vname g i = vname g' j → g = g' ∧ i = j
  vname_fresh : ∀ g i, ∀ p ∈ registry, vname g i ≠ p.2.1

theorem regGet_mem (registry : List (Id × Nm × Gr)) (hid : (registry.map (·.1)).Nodup)
    (id : Id) (g : Nm) (gr : Gr) (hm : (id, g, gr) ∈ registry) : regGet registry id = some gr := by
  apply lookup_unique
  · exact List.mem_map.2 ⟨_, hm, rfl⟩
  · intro p hp hpk
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
    have := nodup_key_eq (·.1) hid hq hm hpk
    subst this; rfl

theorem mem_vkInserts {holds : Nm → Gr → Bool} {layout : Gr → List (List Entry)} {vname : Nm → Nat → Nm}
    {id : Id} {g : Nm} {gr : Gr} {p : Nm × (Id × Nm × List Entry)} (h : p ∈ vkInserts holds layout vname id g gr) :
    p.2.1 = id ∧ p.2.2.1 = g ∧ (p.1 = g ∧ p.2.2.2 = (layout gr).getD 0 [] ∨
      ∃ i, i + 1 < (layout gr).length ∧ p.1 = vname g (i + 1) ∧ p.2.2.2 = (layout gr).getD (i + 1) []) := by
  unfold vkInserts at h
  split at h
  · simp only [List.mem_cons, List.mem_map, List.mem_range] at h
    rcases h with rfl | ⟨i, hi, rfl⟩
    · exact ⟨rfl, rfl, Or.inl ⟨rfl, rfl⟩⟩
    · exact ⟨rfl, rfl, Or.inr ⟨i, by omega, rfl, rfl⟩⟩
  · simp at h

theorem gv_lookup (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (hn : Names registry vname) (id : Id) (g : Nm) (gr : Gr)
    (hg : (id, g, gr) ∈ registry) (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length) :
    lookup (createVKeys holds layout vname registry).gv g =
      some ((List.range ((layout gr).length - 1)).map fun i => vname g (i + 1)) := by
  apply lookup_unique
  · simp only [createVKeys, List.mem_flatMap]
    exact ⟨(id, g, gr), hg, by simp [vkNames, hh, hK]⟩
  · intro p hp hpg
    simp only [createVKeys, List.mem_flatMap] at hp
    obtain ⟨⟨id', g', gr'⟩, hreg, hpi⟩ := hp
    simp only [vkNames] at hpi
    split at hpi
    · simp at hpi; subst hpi
      simp only at hpg; subst hpg
      have e := nodup_key_eq (·.2.1) hn.reg_nodup hreg hg rfl
      simp only [Prod.mk.injEq] at e; obtain ⟨_, _, rfl⟩ := e
      rfl
    · simp at hpi

/-- **The graph's own key holds slice 0.** -/
theorem save_main (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (hn : Names registry vname) (id : Id) (g : Nm) (gr : Gr)
    (hg : (id, g, gr) ∈ registry) (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length) :
    rdbSave (createVKeys holds layout vname registry) (regGet registry) g id =
      .slice g gr ((layout gr).getD 0 []) (layout gr).length := by
  have hmap : lookup (createVKeys holds layout vname registry).map g = some (id, g, (layout gr).getD 0 []) := by
    apply lookup_unique
    · simp only [createVKeys, List.mem_flatMap]
      exact ⟨(id, g, gr), hg, by simp [vkInserts, hh, hK]⟩
    · intro p hp hpg
      simp only [createVKeys, List.mem_flatMap] at hp
      obtain ⟨⟨id', g', gr'⟩, hreg, hpi⟩ := hp
      obtain ⟨h1, h2, h3⟩ := mem_vkInserts hpi
      simp only at h1 h2 h3
      rcases h3 with ⟨h4, h5⟩ | ⟨i, _, h4, _⟩
      · have : g' = g := by rw [← h4, hpg]
        subst this
        have e := nodup_key_eq (·.2.1) hn.reg_nodup hreg hg rfl
        simp only [Prod.mk.injEq] at e; obtain ⟨rfl, _, rfl⟩ := e
        obtain ⟨p1, p2, p3, p4⟩ := p
        simp only at hpg h1 h2 h5; subst hpg h1 h2 h5; rfl
      · rw [hpg] at h4; exact absurd h4.symm (hn.vname_fresh _ _ _ hg)
  simp only [rdbSave, saveKeySlice, hmap, regGet_mem registry hn.id_nodup id g gr hg,
    gv_lookup holds layout vname registry hn id g gr hg hh hK, Option.map_some, Option.getD_some,
    List.length_map, List.length_range, Option.any_some, ne_eq, not_true_eq_false, decide_false,
    Bool.false_eq_true, if_false]
  congr 1; omega

theorem vkey_lookup (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (hn : Names registry vname) (id : Id) (g : Nm) (gr : Gr)
    (hg : (id, g, gr) ∈ registry) (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length)
    (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout gr).length) :
    lookup (createVKeys holds layout vname registry).map (vname g i) =
    some (id, g, (layout gr).getD i []) := by
  apply lookup_unique
  · simp only [createVKeys, List.mem_flatMap]
    refine ⟨(id, g, gr), hg, ?_⟩
    simp only [vkInserts, hh, hK, true_and, decide_true, ite_true, List.mem_cons, List.mem_map, List.mem_range]
    right
    exact ⟨i - 1, by omega, by rw [Nat.sub_add_cancel hi]⟩
  · intro p hp hpg
    simp only [createVKeys, List.mem_flatMap] at hp
    obtain ⟨⟨id', g', gr'⟩, hreg, hpi⟩ := hp
    obtain ⟨h1, h2, h3⟩ := mem_vkInserts hpi
    simp only at h1 h2 h3
    rcases h3 with ⟨h4, _⟩ | ⟨j, _, h4, h5⟩
    · rw [hpg] at h4; exact absurd h4 (hn.vname_fresh _ _ _ hreg)
    · rw [hpg] at h4
      obtain ⟨rfl, hij⟩ := hn.vname_inj _ _ _ _ h4
      have e := nodup_key_eq (·.2.1) hn.reg_nodup hreg hg rfl
      simp only [Prod.mk.injEq] at e; obtain ⟨rfl, _, rfl⟩ := e
      obtain ⟨p1, p2, p3, p4⟩ := p
      simp only at hpg h1 h2 h5; subst hpg h1 h2 h5; rw [← hij]

/-- **The `i`-th virtual key holds slice `i`** — written by `graphmeta_rdb_save`, the
callback Redis calls for a `graphmeta` key. -/
theorem save_vkey_meta (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (hn : Names registry vname) (id : Id) (g : Nm) (gr : Gr)
    (hg : (id, g, gr) ∈ registry) (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length)
    (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout gr).length) :
    metaSave (createVKeys holds layout vname registry) (regGet registry) (vname g i) =
      some (.slice g gr ((layout gr).getD i []) (layout gr).length) := by
  have hmap := vkey_lookup holds layout vname registry hn id g gr hg hh hK i hi hiK
  simp only [metaSave, saveKeySlice, hmap, regGet_mem registry hn.id_nodup id g gr hg,
    gv_lookup holds layout vname registry hn id g gr hg hh hK, Option.map_some, Option.getD_some,
    List.length_map, List.length_range, Option.any_none, Bool.false_eq_true, if_false]
  congr 2; omega

/-- **The `i`-th virtual key holds slice `i`** (as `graph_rdb_save` would encode it for the
slice's own graph). -/
theorem save_vkey (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (hn : Names registry vname) (id : Id) (g : Nm) (gr : Gr)
    (hg : (id, g, gr) ∈ registry) (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length)
    (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout gr).length) :
    rdbSave (createVKeys holds layout vname registry) (regGet registry) (vname g i) id =
      .slice g gr ((layout gr).getD i []) (layout gr).length := by
  apply metaSave_rdbSave _ _ _ _ (save_vkey_meta holds layout vname registry hn id g gr hg hh hK i hi hiK)
  simp [sliceOwner, vkey_lookup holds layout vname registry hn id g gr hg hh hK i hi hiK]

/-- **Everything else is saved whole.** A key that is neither a split graph's own key nor
one of its virtual keys is written by `rdb_save_graph` under its own name. -/
theorem save_single (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (key : Nm) (owner : Id)
    (hk : ∀ id g gr, (id, g, gr) ∈ registry → holds g gr = true → 2 ≤ (layout gr).length →
      key ≠ g ∧ ∀ i, key ≠ vname g i) :
    rdbSave (createVKeys holds layout vname registry) (regGet registry) key owner = .whole key := by
  have : lookup (createVKeys holds layout vname registry).map key = none := by
    apply lookup_none
    intro p hp hpk
    simp only [createVKeys, List.mem_flatMap] at hp
    obtain ⟨⟨id, g, gr⟩, hreg, hpi⟩ := hp
    simp only [vkInserts] at hpi
    split at hpi
    · rename_i hc
      obtain ⟨hh, hK⟩ := hc
      obtain ⟨n1, n2⟩ := hk id g gr hreg (by simpa using hh) hK
      simp only [List.mem_cons, List.mem_map, List.mem_range] at hpi
      rcases hpi with rfl | ⟨i, _, rfl⟩
      · exact n1 hpk.symm
      · exact n2 _ hpk.symm
    · simp at hpi
  simp [rdbSave, saveKeySlice, this]

/-! ## End of save: delete what was created -/

/-- `delete_virtual_keys` (:638): `delete_key` each recorded name (in its db), then clear
the state. -/
def deleteVKeys (vs : VS Id Nm) : List Nm × VS Id Nm := (vs.gv.flatMap Prod.snd, ⟨[], []⟩)

theorem deleteVKeys_spec (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) :
    (deleteVKeys (createVKeys holds layout vname registry)).1 = createdKeys (createVKeys holds layout vname registry) ∧
    (deleteVKeys (createVKeys holds layout vname registry)).2 = (⟨[], []⟩ : VS Id Nm) := ⟨rfl, rfl⟩

/-- Exactly the virtual names of every split graph. -/
theorem createdKeys_spec (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Id × Nm × Gr)) (k : Nm) :
    k ∈ createdKeys (createVKeys holds layout vname registry) ↔
      ∃ id g gr, (id, g, gr) ∈ registry ∧ holds g gr = true ∧ 2 ≤ (layout gr).length ∧
        ∃ i, 1 ≤ i ∧ i < (layout gr).length ∧ k = vname g i := by
  simp only [createdKeys, createVKeys, List.flatMap_assoc, List.mem_flatMap]
  constructor
  · rintro ⟨⟨id, g, gr⟩, hreg, p, hp, hk⟩
    simp only [vkNames] at hp
    split at hp
    · rename_i hc
      simp at hp; subst hp
      simp only [List.mem_map, List.mem_range] at hk
      obtain ⟨i, hi, rfl⟩ := hk
      exact ⟨id, g, gr, hreg, by simpa using hc.1, hc.2, i + 1, by omega, by omega, rfl⟩
    · simp at hp
  · rintro ⟨id, g, gr, hreg, hh, hK, i, hi1, hi2, rfl⟩
    refine ⟨(id, g, gr), hreg, (g, (List.range ((layout gr).length - 1)).map fun i => vname g (i + 1)),
      by simp [vkNames, hh, hK], ?_⟩
    simp only [List.mem_map, List.mem_range]
    exact ⟨i - 1, by omega, by rw [Nat.sub_add_cancel hi1]⟩

/-! ## `SCAN … TYPE` and the stale-key sweeps -/

/-- One `RedisModule_Call("SCAN", cursor, "TYPE", t)` reply. -/
inductive Reply (Nm : Type) where
  | null
  | notArray
  | short                                        -- an array of length < 2
  | page (last : Bool) (keys : List (Option Nm)) -- `last` = the new cursor is "0"; `none` = non-UTF-8 name

/-- `scan_keys_by_type` (:642) over the successive replies. -/
def scanKeys : List (Reply Nm) → List Nm
  | [] => []
  | .null :: _ | .notArray :: _ | .short :: _ => []
  | .page last ks :: rest => ks.filterMap id ++ if last then [] else scanKeys rest

/-- **The scan collects every UTF-8 key name of every page up to the one returning cursor
"0", skips non-UTF-8 names, and stops early on a null, non-array or short reply.** -/
theorem scanKeys_spec (pages : List (List (Option Nm))) (lastKs : List (Option Nm)) (rest : List (Reply Nm)) :
    scanKeys ((pages.map fun ks => Reply.page false ks) ++ Reply.page true lastKs :: rest) =
      (pages ++ [lastKs]).flatMap fun ks => ks.filterMap id := by
  induction pages with
  | nil => simp [scanKeys]
  | cons ks pages ih => simp [scanKeys, ih]

theorem scanKeys_stops (r : Reply Nm) (rest : List (Reply Nm)) (h : r = .null ∨ r = .notArray ∨ r = .short) :
    scanKeys (r :: rest) = [] := by
  rcases h with rfl | rfl | rfl <;> rfl

/-- `delete_stale_graphmeta_keys` (:702): in every non-empty database `d` (`for_each_db`),
delete every key its type scan returns — as `(d, key)`. -/
def deleteStaleGraphmeta {Db : Type} (replies : List (Db × List (Reply Nm))) : List (Db × Nm) :=
  replies.flatMap fun (d, r) => (scanKeys r).map (d, ·)

theorem deleteStaleGraphmeta_spec {Db : Type} (replies : List (Db × List (Reply Nm))) (d : Db) (k : Nm) :
    (d, k) ∈ deleteStaleGraphmeta replies ↔ ∃ r, (d, r) ∈ replies ∧ k ∈ scanKeys r := by
  simp only [deleteStaleGraphmeta, List.mem_flatMap, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨⟨d', r⟩, hm, k', hk, rfl, rfl⟩; exact ⟨r, hm, hk⟩
  · rintro ⟨r, hm, hk⟩; exact ⟨(d, r), hm, k, hk, rfl, rfl⟩

/-- `delete_stale_virtual_keys` (:738): the graphmeta sweep, then `mem::take(meta_keys)` —
each `(db, key)` deleted in its db; `meta_keys` is left empty. -/
def deleteStaleVirtual {Db : Type} (replies : List (Db × List (Reply Nm))) (metaKeys : List (Db × Nm)) :
    List (Db × Nm) × List (Db × Nm) :=
  (deleteStaleGraphmeta replies ++ metaKeys, [])

theorem deleteStaleVirtual_spec {Db : Type} (replies : List (Db × List (Reply Nm))) (metaKeys : List (Db × Nm))
    (d : Db) (k : Nm) :
    ((d, k) ∈ (deleteStaleVirtual replies metaKeys).1 ↔
      (∃ r, (d, r) ∈ replies ∧ k ∈ scanKeys r) ∨ (d, k) ∈ metaKeys) ∧
    (deleteStaleVirtual replies metaKeys).2 = [] := by
  simp [deleteStaleVirtual, deleteStaleGraphmeta_spec]

end GraphPersist.RedisType
