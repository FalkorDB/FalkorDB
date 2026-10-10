import GraphPersist.Persist.Payload
/-! # Virtual keys on save (redis_type.rs:178-227, :410-541, :584-719, :874-882)

| here | there |
| --- | --- |
| `VS`, `lookup`            | `VirtualKeyState { vkey_map, graph_vkeys }` (serializers/mod.rs:30-49); `HashMap::insert` overwrites, so a lookup sees the last insert |
| `createVKeys`             | `create_virtual_keys` (:410-517); `vname g i` stands for the `i`-th name `{g}g_<uuid>` (see `Uuid`) |
| `saveKeySlice`            | `save_key_slice` (:203-227, #3161) |
| `rdbSave`, `SaveOut`      | `graph_rdb_save` (:178-198) |
| `metaSave`, `vkeyType`    | `graphmeta_rdb_save` (:874-882); virtual keys typed `graphmeta` (:492-500, #3161) |
| `deleteVKeys`             | `delete_virtual_keys` (:523-539) |
| `scanKeys`, `Reply`       | `scan_keys_by_type` (:635-719), one `SCAN … TYPE` reply per step |
| `deleteStaleGraphmeta`    | `delete_stale_graphmeta_keys` (:584-610) |
| `deleteStaleVirtual`      | `delete_stale_virtual_keys` (:618-633) |

`key_holds_graph` (:562) and `delete_key` (:542) are `RedisModule_*` calls (inputs / effects);
`build_multi_key_payloads` is the proved `buildMulti`, abstracted here as `layout`.

* `save_main` / `save_vkey` / `save_single` — with distinct names, a graph needing `K ≥ 2`
  keys is saved as `K` keys: its own key holds slice 0 and the `i`-th virtual key slice `i`,
  each stamped `key_count = K`; any other graph is saved whole under its key.
* `metaSave_of_slice` / `metaSave_rdbSave` — since #3161 a virtual key is a `graphmeta` key
  and is written by `graphmeta_rdb_save`, which writes exactly the slice `graph_rdb_save`
  would; `metaSave_stale` — a leftover graphmeta key writes nothing.
* `deleteVKeys_spec` — the end of a save deletes exactly the virtual keys it created.
* `scanKeys_spec`, `deleteStaleVirtual_spec`.
-/
namespace GraphPersist.RedisType
open GraphPersist Persist

variable {Nm Gr : Type} [DecidableEq Nm]

structure VS (Nm : Type) where
  map : List (Nm × (Nm × Nat × List Entry))
  gv : List (Nm × List Nm)

/-- `HashMap::get` after a sequence of inserts: the last one for the key. -/
def lookup {β : Type} (l : List (Nm × β)) (k : Nm) : Option β :=
  (l.reverse.find? (fun p => p.1 = k)).map Prod.snd

/-- What one graph contributes (`:433-512`): nothing unless its key holds it and it needs
`K ≥ 2` keys; then key 0 under the graph's own name and keys `1..K` under virtual names. -/
def vkInserts (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (g : Nm) (gr : Gr) : List (Nm × (Nm × Nat × List Entry)) :=
  if holds g gr ∧ 2 ≤ (layout gr).length then
    (g, (g, 0, (layout gr).getD 0 [])) ::
      ((List.range ((layout gr).length - 1)).map fun i => (vname g (i + 1), (g, i + 1, (layout gr).getD (i + 1) [])))
  else []

def vkNames (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (g : Nm) (gr : Gr) : List (Nm × List Nm) :=
  if holds g gr ∧ 2 ≤ (layout gr).length then [(g, (List.range ((layout gr).length - 1)).map fun i => vname g (i + 1))]
  else []

/-- `create_virtual_keys`: `vkey_state.clear()`, then each registered graph in turn. -/
def createVKeys (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) : VS Nm :=
  ⟨registry.flatMap (fun p => vkInserts holds layout vname p.1 p.2),
   registry.flatMap (fun p => vkNames holds layout vname p.1 p.2)⟩

/-- The Redis keys `create_virtual_keys` opens and fills with a placeholder value. -/
def createdKeys (vs : VS Nm) : List Nm := vs.gv.flatMap Prod.snd

/-- What `graph_rdb_save` writes for one key. -/
inductive SaveOut (Nm Gr : Type) where
  /-- `rdb_save_graph_key(rdb, graph, graph_name, payloads, key_count)` -/
  | slice (g : Nm) (gr : Gr) (payloads : List Entry) (keyCount : Nat)
  /-- `rdb_save_graph(rdb, graph, key_name)`: the key's own value, whole -/
  | whole (key : Nm)

/-- `save_key_slice` (:203, #3161): the slice `create_virtual_keys` assigned `key`,
if this save assigned it one and its graph is still registered (`false` = `none`). -/
def saveKeySlice (vs : VS Nm) (reg : Nm → Option Gr) (key : Nm) : Option (SaveOut Nm Gr) :=
  match lookup vs.map key with
  | some (g, _, payloads) =>
    match reg g with
    | some gr => some (.slice g gr payloads (1 + ((lookup vs.gv g).map List.length).getD 0))
    | none => none
  | none => none

/-- `graph_rdb_save` (:177): the key's slice if it has one, else the whole graph. `reg` is
`GRAPH_REGISTRY.get`. -/
def rdbSave (vs : VS Nm) (reg : Nm → Option Gr) (key : Nm) : SaveOut Nm Gr :=
  (saveKeySlice vs reg key).getD (.whole key)

/-- `graphmeta_rdb_save` (:874, #3161): a virtual key — typed `graphmeta` since #3161, as C
types its own — encodes its slice; a graphmeta key this save did not create writes
nothing (`none`). -/
def metaSave (vs : VS Nm) (reg : Nm → Option Gr) (key : Nm) : Option (SaveOut Nm Gr) :=
  saveKeySlice vs reg key

/-- The type `create_virtual_keys` gives a virtual key (:496-500): `graphmeta`, holding a
dummy byte, not a `graphdata` placeholder graph — C frees a `graphdata` key by dropping its
graph from `GRAPH.LIST`, which made a Rust-saved multi-key graph vanish from C's list (#3160). -/
def vkeyType : String := "graphmeta"

/-- Both save callbacks agree on a key that has a slice. -/
theorem metaSave_rdbSave (vs : VS Nm) (reg : Nm → Option Gr) (key : Nm) (o : SaveOut Nm Gr)
    (h : metaSave vs reg key = some o) : rdbSave vs reg key = o := by
  simp only [metaSave] at h; simp [rdbSave, h]

/-- Conversely: whatever slice `graph_rdb_save` would write for a key, the `graphmeta`
callback writes too — so `save_main`/`save_vkey` hold for a virtual key saved through
`graphmeta_rdb_save`, which is how Redis saves it since #3161. -/
theorem metaSave_of_slice (vs : VS Nm) (reg : Nm → Option Gr) (key g : Nm) (gr : Gr) (pl : List Entry)
    (k : Nat) (h : rdbSave vs reg key = .slice g gr pl k) : metaSave vs reg key = some (.slice g gr pl k) := by
  unfold rdbSave at h; unfold metaSave
  cases hs : saveKeySlice vs reg key with
  | none => rw [hs] at h; cases h
  | some o => rw [hs] at h; simp at h; rw [h]

/-- A graphmeta key with no slice assigned writes nothing. -/
theorem metaSave_stale (vs : VS Nm) (reg : Nm → Option Gr) (key : Nm) (h : lookup vs.map key = none) :
    metaSave vs reg key = none := by simp [metaSave, saveKeySlice, h]

/-! ## Lookups under distinct keys -/

theorem lookup_unique {β : Type} (l : List (Nm × β)) (k : Nm) (v : β) (hmem : (k, v) ∈ l)
    (huniq : ∀ p ∈ l, p.1 = k → p = (k, v)) : lookup l k = some v := by
  unfold lookup
  have hex : ∃ p ∈ l.reverse, (fun p : Nm × β => decide (p.1 = k)) p = true := ⟨(k, v), by simpa using hmem, by simp⟩
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 (List.find?_isSome.2 hex)
  have hpm := List.mem_of_find?_eq_some hp
  have hpk := List.find?_some hp
  simp at hpk
  rw [hp, huniq p (by simpa using hpm) hpk]; rfl

theorem lookup_none {β : Type} (l : List (Nm × β)) (k : Nm) (h : ∀ p ∈ l, p.1 ≠ k) : lookup l k = none := by
  unfold lookup
  rw [List.find?_eq_none.2 (by intro p hp; simpa using h p (by simpa using hp))]; rfl

theorem nodup_fst_eq {l : List (Nm × Gr)} (h : (l.map Prod.fst).Nodup) {a b : Nm × Gr}
    (ha : a ∈ l) (hb : b ∈ l) (he : a.1 = b.1) : a = b := by
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

/-- The naming assumptions the uuid scheme is meant to deliver: registry names distinct,
virtual names injective, and no virtual name equal to a graph name. -/
structure Names (registry : List (Nm × Gr)) (vname : Nm → Nat → Nm) : Prop where
  reg_nodup : (registry.map Prod.fst).Nodup
  vname_inj : ∀ g i g' j, vname g i = vname g' j → g = g' ∧ i = j
  vname_fresh : ∀ g i, ∀ p ∈ registry, vname g i ≠ p.1

theorem mem_vkInserts {holds : Nm → Gr → Bool} {layout : Gr → List (List Entry)} {vname : Nm → Nat → Nm}
    {g : Nm} {gr : Gr} {p : Nm × (Nm × Nat × List Entry)} (h : p ∈ vkInserts holds layout vname g gr) :
    p.2.1 = g ∧ (p.1 = g ∧ p.2.2.1 = 0 ∨ ∃ i, p.1 = vname g (i + 1) ∧ p.2.2.1 = i + 1) := by
  unfold vkInserts at h
  split at h
  · simp only [List.mem_cons, List.mem_map, List.mem_range] at h
    rcases h with rfl | ⟨i, _, rfl⟩
    · exact ⟨rfl, Or.inl ⟨rfl, rfl⟩⟩
    · exact ⟨rfl, Or.inr ⟨i, rfl, rfl⟩⟩
  · simp at h

/-- **The graph's own key holds slice 0.** -/
theorem save_main (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) (hn : Names registry vname) (g : Nm) (gr : Gr) (hg : (g, gr) ∈ registry)
    (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length) :
    rdbSave (createVKeys holds layout vname registry) (lookup registry) g =
      .slice g gr ((layout gr).getD 0 []) (layout gr).length := by
  have hreg_uniq : ∀ p ∈ registry, p.1 = g → p = (g, gr) := by
    intro p hp hpg
    obtain ⟨g', gr'⟩ := p; simp only at hpg; subst hpg
    exact nodup_fst_eq hn.reg_nodup hp hg rfl
  have hmap : lookup (createVKeys holds layout vname registry).map g = some (g, 0, (layout gr).getD 0 []) := by
    apply lookup_unique
    · simp only [createVKeys, List.mem_flatMap]
      exact ⟨(g, gr), hg, by simp [vkInserts, hh, hK]⟩
    · intro p hp hpg
      simp only [createVKeys, List.mem_flatMap] at hp
      obtain ⟨⟨g', gr'⟩, hreg, hpi⟩ := hp
      obtain ⟨h1, h2⟩ := mem_vkInserts hpi
      simp only at h1 h2
      rcases h2 with ⟨h3, h4⟩ | ⟨i, h3, _⟩
      · have : g' = g := by rw [← h3, hpg]
        subst this
        have e := hreg_uniq _ hreg rfl
        simp only [Prod.mk.injEq] at e; obtain ⟨_, rfl⟩ := e
        simp only [vkInserts, hh, hK, true_and, decide_true, ite_true, List.mem_cons, List.mem_map,
          List.mem_range] at hpi
        rcases hpi with hpi | ⟨i, _, hpi⟩
        · exact hpi
        · rw [← hpi] at hpg; exact absurd hpg (hn.vname_fresh _ _ _ hg)
      · rw [hpg] at h3; exact absurd h3.symm (hn.vname_fresh _ _ _ hg)
  have hgv : lookup (createVKeys holds layout vname registry).gv g =
      some ((List.range ((layout gr).length - 1)).map fun i => vname g (i + 1)) := by
    apply lookup_unique
    · simp only [createVKeys, List.mem_flatMap]
      exact ⟨(g, gr), hg, by simp [vkNames, hh, hK]⟩
    · intro p hp hpg
      simp only [createVKeys, List.mem_flatMap] at hp
      obtain ⟨⟨g', gr'⟩, hreg, hpi⟩ := hp
      simp only [vkNames] at hpi
      split at hpi
      · simp at hpi; subst hpi
        simp only at hpg; subst hpg
        have e := hreg_uniq _ hreg rfl
        simp only [Prod.mk.injEq] at e; obtain ⟨_, rfl⟩ := e
        rfl
      · simp at hpi
  simp only [rdbSave, saveKeySlice, hmap, lookup_unique registry g gr hg hreg_uniq, hgv, Option.map_some, Option.getD_some,
    List.length_map, List.length_range]
  congr 1; omega

/-- **The `i`-th virtual key holds slice `i`.** -/
theorem save_vkey (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) (hn : Names registry vname) (g : Nm) (gr : Gr) (hg : (g, gr) ∈ registry)
    (hh : holds g gr = true) (hK : 2 ≤ (layout gr).length) (i : Nat) (hi : 1 ≤ i) (hiK : i < (layout gr).length) :
    rdbSave (createVKeys holds layout vname registry) (lookup registry) (vname g i) =
      .slice g gr ((layout gr).getD i []) (layout gr).length := by
  have hreg_uniq : ∀ p ∈ registry, p.1 = g → p = (g, gr) := by
    intro p hp hpg
    obtain ⟨g', gr'⟩ := p; simp only at hpg; subst hpg
    exact nodup_fst_eq hn.reg_nodup hp hg rfl
  have hmap : lookup (createVKeys holds layout vname registry).map (vname g i) =
      some (g, i, (layout gr).getD i []) := by
    apply lookup_unique
    · simp only [createVKeys, List.mem_flatMap]
      refine ⟨(g, gr), hg, ?_⟩
      simp only [vkInserts, hh, hK, true_and, decide_true, ite_true, List.mem_cons, List.mem_map, List.mem_range]
      right
      exact ⟨i - 1, by omega, by rw [Nat.sub_add_cancel hi]⟩
    · intro p hp hpg
      simp only [createVKeys, List.mem_flatMap] at hp
      obtain ⟨⟨g', gr'⟩, hreg, hpi⟩ := hp
      obtain ⟨h1, h2⟩ := mem_vkInserts hpi
      simp only at h1 h2
      rcases h2 with ⟨h3, _⟩ | ⟨j, h3, h4⟩
      · rw [hpg] at h3; exact absurd h3 (hn.vname_fresh _ _ _ hreg)
      · rw [hpg] at h3
        obtain ⟨rfl, rfl⟩ := hn.vname_inj _ _ _ _ h3
        have e := hreg_uniq _ hreg rfl
        simp only [Prod.mk.injEq] at e; obtain ⟨_, rfl⟩ := e
        simp only [vkInserts, hh, hK, true_and, decide_true, ite_true, List.mem_cons, List.mem_map,
          List.mem_range] at hpi
        rcases hpi with hpi | ⟨k, _, hpi⟩
        · rw [hpi] at hpg; exact absurd hpg.symm (hn.vname_fresh _ _ _ hg)
        · rw [← hpi] at hpg ⊢
          obtain ⟨_, hk⟩ := hn.vname_inj _ _ _ _ hpg
          have hkj : k = j := by omega
          subst hkj; rfl
  have hgv : lookup (createVKeys holds layout vname registry).gv g =
      some ((List.range ((layout gr).length - 1)).map fun i => vname g (i + 1)) := by
    apply lookup_unique
    · simp only [createVKeys, List.mem_flatMap]
      exact ⟨(g, gr), hg, by simp [vkNames, hh, hK]⟩
    · intro p hp hpg
      simp only [createVKeys, List.mem_flatMap] at hp
      obtain ⟨⟨g', gr'⟩, hreg, hpi⟩ := hp
      simp only [vkNames] at hpi
      split at hpi
      · simp at hpi; subst hpi
        simp only at hpg; subst hpg
        have e := hreg_uniq _ hreg rfl
        simp only [Prod.mk.injEq] at e; obtain ⟨_, rfl⟩ := e
        rfl
      · simp at hpi
  simp only [rdbSave, saveKeySlice, hmap, lookup_unique registry g gr hg hreg_uniq, hgv, Option.map_some, Option.getD_some,
    List.length_map, List.length_range]
  congr 1; omega

/-- **Everything else is saved whole.** A key that is neither a split graph's own key nor
one of its virtual keys is written by `rdb_save_graph` under its own name. -/
theorem save_single (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) (key : Nm)
    (hk : ∀ g gr, (g, gr) ∈ registry → holds g gr = true → 2 ≤ (layout gr).length →
      key ≠ g ∧ ∀ i, key ≠ vname g i) :
    rdbSave (createVKeys holds layout vname registry) (lookup registry) key = .whole key := by
  have : lookup (createVKeys holds layout vname registry).map key = none := by
    apply lookup_none
    intro p hp hpk
    simp only [createVKeys, List.mem_flatMap] at hp
    obtain ⟨⟨g, gr⟩, hreg, hpi⟩ := hp
    simp only [vkInserts] at hpi
    split at hpi
    · rename_i hc
      obtain ⟨hh, hK⟩ := hc
      obtain ⟨n1, n2⟩ := hk g gr hreg (by simpa using hh) hK
      simp only [List.mem_cons, List.mem_map, List.mem_range] at hpi
      rcases hpi with rfl | ⟨i, _, rfl⟩
      · exact n1 hpk.symm
      · exact n2 _ hpk.symm
    · simp at hpi
  simp [rdbSave, saveKeySlice, this]

/-! ## End of save: delete what was created -/

/-- `delete_virtual_keys` (:523): `delete_key` each recorded name, then clear the state. -/
def deleteVKeys (vs : VS Nm) : List Nm × VS Nm := (vs.gv.flatMap Prod.snd, ⟨[], []⟩)

theorem deleteVKeys_spec (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) :
    (deleteVKeys (createVKeys holds layout vname registry)).1 = createdKeys (createVKeys holds layout vname registry) ∧
    (deleteVKeys (createVKeys holds layout vname registry)).2 = ⟨[], []⟩ := ⟨rfl, rfl⟩

/-- Exactly the virtual names of every split graph. -/
theorem createdKeys_spec (holds : Nm → Gr → Bool) (layout : Gr → List (List Entry)) (vname : Nm → Nat → Nm)
    (registry : List (Nm × Gr)) (k : Nm) :
    k ∈ createdKeys (createVKeys holds layout vname registry) ↔
      ∃ g gr, (g, gr) ∈ registry ∧ holds g gr = true ∧ 2 ≤ (layout gr).length ∧
        ∃ i, 1 ≤ i ∧ i < (layout gr).length ∧ k = vname g i := by
  simp only [createdKeys, createVKeys, List.flatMap_assoc, List.mem_flatMap]
  constructor
  · rintro ⟨⟨g, gr⟩, hreg, p, hp, hk⟩
    simp only [vkNames] at hp
    split at hp
    · rename_i hc
      simp at hp; subst hp
      simp only [List.mem_map, List.mem_range] at hk
      obtain ⟨i, hi, rfl⟩ := hk
      exact ⟨g, gr, hreg, by simpa using hc.1, hc.2, i + 1, by omega, by omega, rfl⟩
    · simp at hp
  · rintro ⟨g, gr, hreg, hh, hK, i, hi1, hi2, rfl⟩
    refine ⟨(g, gr), hreg, (g, (List.range ((layout gr).length - 1)).map fun i => vname g (i + 1)),
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

/-- `scan_keys_by_type` (:635) over the successive replies. -/
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

/-- `delete_stale_graphmeta_keys` (:584): delete every key the type scan returns. -/
def deleteStaleGraphmeta (replies : List (Reply Nm)) : List Nm := scanKeys replies

/-- `delete_stale_virtual_keys` (:618): the graphmeta sweep, then `mem::take(meta_keys)`
deleted one by one; `meta_keys` is left empty. -/
def deleteStaleVirtual (replies : List (Reply Nm)) (metaKeys : List Nm) : List Nm × List Nm :=
  (deleteStaleGraphmeta replies ++ metaKeys, [])

theorem deleteStaleVirtual_spec (replies : List (Reply Nm)) (metaKeys : List Nm) (k : Nm) :
    (k ∈ (deleteStaleVirtual replies metaKeys).1 ↔ k ∈ scanKeys replies ∨ k ∈ metaKeys) ∧
    (deleteStaleVirtual replies metaKeys).2 = [] := by
  simp [deleteStaleVirtual, deleteStaleGraphmeta]

end GraphPersist.RedisType
