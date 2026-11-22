import GraphPersist.Persist.MultiLoad3
/-! # `DECODE_STATE` across graphs and databases (decoder/mod.rs, #3424)

`rdb_load_graph(rdb, db, encver, key_name, cache_size)`, `key_count > 1` branch, on the
whole `DECODE_STATE` (serializers/mod.rs): `pending` and `finalized` are keyed by the
`DbKey` `(db, hdr.graph_name)`, and a virtual key is recorded in `meta_keys` as
`(db, key_name)`. `stepKeyDb` runs `Pending.stepKey` — the step of one graph's entry — on
the entry `(db, hdr.graph_name)` and leaves every other entry alone.

* `runKeysDb_sep` — an entry's fate depends only on the keys of that `(db, graph name)`:
  the keys of other graphs, interleaved in any way, do not touch it.
* `multi_load_db` — `multi_load` on the whole state: a graph of database `d`, its keys
  loaded in any order, finalizes at `(d, name)`, and `meta_keys` gets its virtual keys,
  each with `d`.
* `two_dbs_same_name` — two graphs of one name in two databases, their keys interleaved in
  any way, each finalize to themselves (#2782: keyed by name, their slices were folded into
  one pending graph).
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type} [DecidableEq Nm]

/-- `DECODE_STATE`'s `pending`, `finalized` and `meta_keys`. -/
structure DSM (M Tn Nm Ix Cn : Type) where
  pending : Int × Nm → Option (PG M Tn Nm Ix Cn)
  finalized : Int × Nm → Option (Restore M Tn Nm Ix Cn)
  metaKeys : List (Int × Nm)

def DSM.empty : DSM M Tn Nm Ix Cn := ⟨fun _ => none, fun _ => none, []⟩

/-- The entry `k`, as the one-graph state `stepKey` works on. -/
def DSM.at (dsm : DSM M Tn Nm Ix Cn) (k : Int × Nm) : DS M Tn Nm Ix Cn := ⟨dsm.pending k, dsm.finalized k, []⟩

/-- One key as `rdb_load_graph` sees it after its reads: its database (`io_db`), name, the
header and schema it carries and its payloads. -/
structure LKey (M Tn Nm Ix Cn : Type) where
  db : Int
  kn : Nm
  hdr : Hdr Nm
  sch : Sch Nm Ix Cn
  prs : List (PR M Tn)

/-- The `DbKey` of the graph a key belongs to: `(db, hdr.graph_name)`. -/
def LKey.gk (it : LKey M Tn Nm Ix Cn) : Int × Nm := (it.db, it.hdr.name)

/-- `rdb_load_graph`'s multi-key branch on the whole state; also returns `is_virtual`. -/
def stepKeyDb (C : Codecs M Tn Nm Ix Cn) (dsm : DSM M Tn Nm Ix Cn) (it : LKey M Tn Nm Ix Cn) :
    DSM M Tn Nm Ix Cn × Bool :=
  let r := stepKey C (dsm.at it.gk) it.kn it.hdr it.sch it.prs
  (⟨fun k => if k = it.gk then r.1.pending else dsm.pending k,
    fun k => if k = it.gk then r.1.finalized else dsm.finalized k,
    dsm.metaKeys ++ r.1.metaKeys.map (it.db, ·)⟩, r.2)

def runKeysDb (C : Codecs M Tn Nm Ix Cn) : DSM M Tn Nm Ix Cn → List (LKey M Tn Nm Ix Cn) → DSM M Tn Nm Ix Cn
  | dsm, [] => dsm
  | dsm, it :: rest => runKeysDb C (stepKeyDb C dsm it).1 rest

/-- Two states agree on entry `k`. -/
def Agree (k : Int × Nm) (a b : DSM M Tn Nm Ix Cn) : Prop := a.pending k = b.pending k ∧ a.finalized k = b.finalized k

theorem step_agree (C : Codecs M Tn Nm Ix Cn) (k : Int × Nm) (a b : DSM M Tn Nm Ix Cn) (it : LKey M Tn Nm Ix Cn)
    (h : Agree k a b) : Agree k (stepKeyDb C a it).1 (stepKeyDb C b it).1 := by
  by_cases hk : k = it.gk
  · subst hk; simp [Agree, stepKeyDb, DSM.at, h.1, h.2]
  · simp [Agree, stepKeyDb, hk, h.1, h.2]

theorem step_other (C : Codecs M Tn Nm Ix Cn) (k : Int × Nm) (a : DSM M Tn Nm Ix Cn) (it : LKey M Tn Nm Ix Cn)
    (hk : it.gk ≠ k) : Agree k (stepKeyDb C a it).1 a := by
  simp [Agree, stepKeyDb, Ne.symm hk]

theorem run_agree (C : Codecs M Tn Nm Ix Cn) (k : Int × Nm) :
    ∀ (items : List (LKey M Tn Nm Ix Cn)) (a b : DSM M Tn Nm Ix Cn), Agree k a b →
      Agree k (runKeysDb C a items) (runKeysDb C b items)
  | [], _, _, h => h
  | it :: rest, a, b, h => run_agree C k rest _ _ (step_agree C k a b it h)

/-- **Separation.** Entry `k` after any sequence of keys is what the keys of graph `k`
alone make of it. -/
theorem runKeysDb_sep (C : Codecs M Tn Nm Ix Cn) (k : Int × Nm) :
    ∀ (items : List (LKey M Tn Nm Ix Cn)) (dsm : DSM M Tn Nm Ix Cn),
      Agree k (runKeysDb C dsm items) (runKeysDb C dsm (items.filter fun it => it.gk = k))
  | [], _ => ⟨rfl, rfl⟩
  | it :: rest, dsm => by
    by_cases hk : it.gk = k
    · simp only [runKeysDb, List.filter_cons, hk, decide_true, ite_true]
      exact runKeysDb_sep C k rest _
    · simp only [runKeysDb, List.filter_cons, hk, decide_false]
      have h1 := runKeysDb_sep C k rest (stepKeyDb C dsm it).1
      have h2 := run_agree C k (rest.filter fun it => it.gk = k) _ _ (step_other C k dsm it hk)
      exact ⟨h1.1.trans h2.1, h1.2.trans h2.2⟩

/-- `stepKey`'s entry does not depend on the meta keys recorded so far. -/
theorem stepKey_meta (C : Codecs M Tn Nm Ix Cn) (ds : DS M Tn Nm Ix Cn) (kn : Nm) (h : Hdr Nm) (s : Sch Nm Ix Cn)
    (prs : List (PR M Tn)) :
    (stepKey C ds kn h s prs).1 =
      ⟨(stepKey C ⟨ds.pending, ds.finalized, []⟩ kn h s prs).1.pending,
       (stepKey C ⟨ds.pending, ds.finalized, []⟩ kn h s prs).1.finalized,
       ds.metaKeys ++ (if kn ≠ h.name then [kn] else [])⟩ := by
  obtain ⟨p, f, m⟩ := ds
  cases p <;> by_cases hk : kn = h.name <;> simp only [stepKey] <;> split <;> simp_all

theorem stepKeyDb_meta (C : Codecs M Tn Nm Ix Cn) (dsm : DSM M Tn Nm Ix Cn) (it : LKey M Tn Nm Ix Cn) :
    (stepKeyDb C dsm it).1.metaKeys = dsm.metaKeys ++ (if it.kn ≠ it.hdr.name then [(it.db, it.kn)] else []) := by
  simp only [stepKeyDb]
  rw [stepKey_meta]
  by_cases hk : it.kn = it.hdr.name <;> simp [DSM.at, hk]

theorem runKeys_meta (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) :
    ∀ (items : List (Nm × List (PR M Tn))) (ds : DS M Tn Nm Ix Cn),
      runKeys C h s ds items =
        ⟨(runKeys C h s ⟨ds.pending, ds.finalized, []⟩ items).pending,
         (runKeys C h s ⟨ds.pending, ds.finalized, []⟩ items).finalized, ds.metaKeys ++ metaOf h items⟩
  | [], ds => by simp [runKeys, metaOf]
  | (kn, prs) :: rest, ds => by
    simp only [runKeys]
    rw [runKeys_meta C h s rest, runKeys_meta C h s rest (stepKey C ⟨ds.pending, ds.finalized, []⟩ kn h s prs).1,
      stepKey_meta C ds kn h s prs, stepKey_meta C ⟨ds.pending, ds.finalized, []⟩ kn h s prs]
    simp [metaOf_cons]

/-- The keys of one graph, all at `(d, h.name)`, run on the whole state exactly as
`runKeys` runs on the entry, with `meta_keys` qualified by `d`. -/
theorem runKeysDb_one (C : Codecs M Tn Nm Ix Cn) (d : Int) (h : Hdr Nm) (s : Sch Nm Ix Cn) :
    ∀ (items : List (Nm × List (PR M Tn))) (dsm : DSM M Tn Nm Ix Cn),
      let r := runKeysDb C dsm (items.map fun p => ⟨d, p.1, h, s, p.2⟩)
      let e := runKeys C h s (dsm.at (d, h.name)) items
      r.pending (d, h.name) = e.pending ∧ r.finalized (d, h.name) = e.finalized ∧
        r.metaKeys = dsm.metaKeys ++ (metaOf h items).map (d, ·)
  | [], dsm => by simp [runKeysDb, runKeys, DSM.at, metaOf]
  | (kn, prs) :: rest, dsm => by
    intro r e
    have ih := runKeysDb_one C d h s rest (stepKeyDb C dsm ⟨d, kn, h, s, prs⟩).1
    have hat : (stepKeyDb C dsm ⟨d, kn, h, s, prs⟩).1.at (d, h.name) =
        ⟨(stepKey C (dsm.at (d, h.name)) kn h s prs).1.pending,
         (stepKey C (dsm.at (d, h.name)) kn h s prs).1.finalized, []⟩ := by
      simp [DSM.at, stepKeyDb, LKey.gk]
    have he : e = runKeys C h s (stepKey C (dsm.at (d, h.name)) kn h s prs).1 rest := rfl
    rw [runKeys_meta] at he
    have hst := stepKey_meta C (dsm.at (d, h.name)) kn h s prs
    simp only [DSM.at] at hst
    simp only [DSM.at] at he
    rw [hst] at he
    simp only at he
    simp only [hat] at ih
    refine ⟨?_, ?_, ?_⟩
    · rw [he]; exact ih.1
    · rw [he]; exact ih.2.1
    · show (runKeysDb C (stepKeyDb C dsm ⟨d, kn, h, s, prs⟩).1 (rest.map fun p => ⟨d, p.1, h, s, p.2⟩)).metaKeys = _
      rw [ih.2.2, stepKeyDb_meta, metaOf_cons]
      by_cases hk : kn = h.name <;> simp [hk]

/-- The keys of graph `g` of database `d`, loaded in order `ord` under names `names`. -/
def keysOf (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (vmax : Nat) (d : Int) (ord : List Nat) (names : Nat → Nm) :
    List (LKey M Tn Nm Ix Cn) :=
  let _ := C
  ord.map fun k => ⟨d, names k, hdrOf g g.name (buildMulti g vmax).length, g.sch,
    (keyAt (buildMulti g vmax) k).map (prOfE g)⟩

/-- **`multi_load` on the whole `DECODE_STATE`.** A graph of database `d` saved across
`K ≥ 2` keys, loaded in any order: no pending entry left at `(d, name)`, the finalized entry
there is the single-key restore of the graph, and `meta_keys` gets exactly the keys not
named after the graph, each with database `d`. -/
theorem multi_load_db (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 2 ≤ (buildMulti g vmax).length)
    (hH : C.H.good (hdrOf g g.name (buildMulti g vmax).length))
    (ord : List Nat) (hord : ord.Perm (List.range (buildMulti g vmax).length)) (names : Nat → Nm) (d : Int) :
    let r := runKeysDb C DSM.empty (keysOf C g vmax d ord names)
    r.pending (d, g.name) = none ∧ r.finalized (d, g.name) = some (restoreOf g g.name) ∧
      r.metaKeys = (metaOf (hdrOf g g.name (buildMulti g vmax).length)
        (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g)))).map (d, ·) := by
  intro r
  have h1 := (multi_load C g hw vmax ht hK hH ord hord names).2
  have h2 := runKeysDb_one C d (hdrOf g g.name (buildMulti g vmax).length) g.sch
    (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g))) DSM.empty
  have e : (DSM.empty : DSM M Tn Nm Ix Cn).at (d, (hdrOf g g.name (buildMulti g vmax).length).name) = DS.empty := rfl
  rw [e, h1] at h2
  have hm : keysOf C g vmax d ord names = (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g))).map
      fun p => (⟨d, p.1, hdrOf g g.name (buildMulti g vmax).length, g.sch, p.2⟩ : LKey M Tn Nm Ix Cn) := by
    simp [keysOf]
  rw [← hm] at h2
  exact ⟨h2.1, h2.2.1, by simpa [r, DSM.empty] using h2.2.2⟩

/-- **Two graphs of one name in two databases do not collide** (#2782). Whatever
interleaving `items` of their keys Redis loads — `items` restricted to each graph is that
graph's keys in some order — each finalizes to itself, in its own database. -/
theorem two_dbs_same_name (C : Codecs M Tn Nm Ix Cn) (g g' : G M Tn Nm Ix Cn) (hw : WF C g) (hw' : WF C g')
    (vmax : Nat) (ht : total g ≤ U64MAX) (ht' : total g' ≤ U64MAX)
    (hK : 2 ≤ (buildMulti g vmax).length) (hK' : 2 ≤ (buildMulti g' vmax).length)
    (hH : C.H.good (hdrOf g g.name (buildMulti g vmax).length))
    (hH' : C.H.good (hdrOf g' g'.name (buildMulti g' vmax).length))
    (d d' : Int) (_hd : d ≠ d') (_hn : g.name = g'.name)
    (ord ord' : List Nat) (hord : ord.Perm (List.range (buildMulti g vmax).length))
    (hord' : ord'.Perm (List.range (buildMulti g' vmax).length)) (names names' : Nat → Nm)
    (items : List (LKey M Tn Nm Ix Cn))
    (hi : items.filter (fun it => it.gk = (d, g.name)) = keysOf C g vmax d ord names)
    (hi' : items.filter (fun it => it.gk = (d', g'.name)) = keysOf C g' vmax d' ord' names') :
    (runKeysDb C DSM.empty items).finalized (d, g.name) = some (restoreOf g g.name) ∧
    (runKeysDb C DSM.empty items).finalized (d', g'.name) = some (restoreOf g' g'.name) := by
  have s1 := runKeysDb_sep C (d, g.name) items DSM.empty
  have s2 := runKeysDb_sep C (d', g'.name) items DSM.empty
  rw [hi] at s1; rw [hi'] at s2
  exact ⟨s1.2.trans (multi_load_db C g hw vmax ht hK hH ord hord names d).2.1,
    s2.2.trans (multi_load_db C g' hw' vmax ht' hK' hH' ord' hord' names' d').2.1⟩

end GraphPersist.Persist
