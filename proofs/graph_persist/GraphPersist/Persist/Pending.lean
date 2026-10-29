import GraphPersist.Persist.Fold
/-! # `DECODE_STATE` for one multi-key graph (decoder/mod.rs:69-174)

After a key's reads, `rdb_load_graph`'s `key_count > 1` branch: record a virtual key
(`key_name != hdr.graph_name`), create the pending graph on the first key
(`keys_remaining = key_count - 1`, attribute dictionary from *that* key's schema) or
decrement `keys_remaining`, fold the payloads in, and finalize as soon as
`keys_remaining == 0` (`finalize_pending_graph`, :310-342, into `finalized`).

`run_keys`: whatever order the `K` keys arrive in, nothing is finalized before the last
one, the last one finalizes exactly the fold of every key's payloads, and the meta-key
list is exactly the keys not named after the graph.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type} [DecidableEq Nm]

/-- `PendingGraph` (serializers/mod.rs): the counter, the first key's header and schema,
and the accumulated locals. -/
structure PG (M Tn Nm Ix Cn : Type) where
  remaining : Nat
  hdr : Hdr Nm
  sch : Sch Nm Ix Cn
  acc : Acc M Tn

/-- `DECODE_STATE` restricted to one graph name. -/
structure DS (M Tn Nm Ix Cn : Type) where
  pending : Option (PG M Tn Nm Ix Cn)
  finalized : Option (Restore M Tn Nm Ix Cn)
  metaKeys : List Nm

def DS.empty : DS M Tn Nm Ix Cn := ⟨none, none, []⟩

/-- `finalize_pending_graph`: `Graph::restore` from the pending graph's fields. -/
def finalizePG (pg : PG M Tn Nm Ix Cn) : Restore M Tn Nm Ix Cn :=
  ⟨pg.hdr.name, pg.hdr.nodeCount, pg.hdr.edgeCount, pg.acc, pg.sch.labels, pg.sch.types, pg.sch.attrs,
    pg.sch.indexes, pg.sch.constraints⟩

/-- One key of a multi-key graph, after its reads (`:69-173`). Returns the new state and
`is_virtual` (`:171-173`). -/
def stepKey (C : Codecs M Tn Nm Ix Cn) (ds : DS M Tn Nm Ix Cn) (kn : Nm) (h : Hdr Nm) (s : Sch Nm Ix Cn)
    (prs : List (PR M Tn)) : DS M Tn Nm Ix Cn × Bool :=
  let mk := if kn ≠ h.name then ds.metaKeys ++ [kn] else ds.metaKeys
  let pg0 : PG M Tn Nm Ix Cn := match ds.pending with
    | none => ⟨h.keyCount - 1, h, s, acc0 C⟩
    | some pg => { pg with remaining := pg.remaining - 1 }
  let pg1 := { pg0 with acc := prs.foldl (applyPR pg0.sch.attrs.length) pg0.acc }
  if pg1.remaining = 0 then (⟨none, some (finalizePG pg1), mk⟩, decide (kn ≠ h.name))
  else (⟨some pg1, ds.finalized, mk⟩, decide (kn ≠ h.name))

/-- Keys arriving one after another, all carrying the same header and schema. -/
def runKeys (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) :
    DS M Tn Nm Ix Cn → List (Nm × List (PR M Tn)) → DS M Tn Nm Ix Cn
  | ds, [] => ds
  | ds, (kn, prs) :: rest => runKeys C h s (stepKey C ds kn h s prs).1 rest

def metaOf (h : Hdr Nm) (items : List (Nm × List (PR M Tn))) : List Nm :=
  (items.filter fun it => decide (it.1 ≠ h.name)).map Prod.fst

theorem metaOf_cons (h : Hdr Nm) (kn : Nm) (prs : List (PR M Tn)) (rest : List (Nm × List (PR M Tn))) :
    metaOf h ((kn, prs) :: rest) = (if kn ≠ h.name then [kn] else []) ++ metaOf h rest := by
  by_cases hk : kn = h.name <;> simp [metaOf, List.filter_cons, hk]

/-- Later keys: each decrements, the last finalizes. -/
theorem run_later (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) :
    ∀ (rest : List (Nm × List (PR M Tn))) (ds : DS M Tn Nm Ix Cn) (pg : PG M Tn Nm Ix Cn),
    rest ≠ [] → ds.pending = some pg → pg.remaining = rest.length → ds.finalized = none →
    runKeys C h s ds rest =
      ⟨none, some (finalizePG ⟨0, pg.hdr, pg.sch,
        (rest.flatMap Prod.snd).foldl (applyPR pg.sch.attrs.length) pg.acc⟩),
       ds.metaKeys ++ metaOf h rest⟩
  | [], _, _, hne, _, _, _ => absurd rfl hne
  | (kn, prs) :: rest, ds, pg, _, hp, hr, hf => by
    simp only [runKeys, stepKey, hp]
    simp only [List.length_cons] at hr
    by_cases hz : rest = []
    · subst hz
      simp only [hr, Nat.add_sub_cancel, ite_true, List.flatMap_cons, List.flatMap_nil, List.append_nil,
        runKeys, metaOf_cons]
      congr 1
      by_cases hk : kn = h.name <;> simp [hk, metaOf]
    · have hlen : rest.length ≠ 0 := fun e => hz (List.length_eq_zero_iff.1 e)
      simp only [hr, Nat.add_sub_cancel, hlen, ite_false]
      rw [run_later C h s rest ⟨some ⟨rest.length, pg.hdr, pg.sch, prs.foldl (applyPR pg.sch.attrs.length) pg.acc⟩,
        ds.finalized, if kn ≠ h.name then ds.metaKeys ++ [kn] else ds.metaKeys⟩
        ⟨rest.length, pg.hdr, pg.sch, prs.foldl (applyPR pg.sch.attrs.length) pg.acc⟩ hz rfl rfl hf]
      simp only [List.flatMap_cons, List.foldl_append, metaOf_cons, DS.mk.injEq, true_and]
      by_cases hk : kn = h.name <;> simp [hk]

/-- **Any arrival order.** `K = items.length ≥ 2` keys of one graph, each carrying the
header (`key_count = K`) and schema: nothing finalizes early; the last key finalizes the
fold of every key's payloads (in arrival order — which `pending_any_order` shows does not
matter); `meta_keys` is exactly the keys not named after the graph. -/
theorem run_keys (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (items : List (Nm × List (PR M Tn)))
    (hK : h.keyCount = items.length) (h2 : 2 ≤ items.length) :
    runKeys C h s DS.empty items =
      ⟨none, some ⟨h.name, h.nodeCount, h.edgeCount,
        (items.flatMap Prod.snd).foldl (applyPR s.attrs.length) (acc0 C),
        s.labels, s.types, s.attrs, s.indexes, s.constraints⟩, metaOf h items⟩ := by
  match items, h2 with
  | (kn, prs) :: rest, h2 =>
    simp only [List.length_cons] at hK h2
    have hne : rest ≠ [] := by rintro rfl; simp at h2
    have hlen : rest.length ≠ 0 := fun e => hne (List.length_eq_zero_iff.1 e)
    simp only [runKeys, stepKey, DS.empty, hK, Nat.add_sub_cancel, hlen, ite_false]
    rw [run_later C h s rest ⟨some ⟨rest.length, h, s, prs.foldl (applyPR s.attrs.length) (acc0 C)⟩,
        none, if kn ≠ h.name then [] ++ [kn] else []⟩
        ⟨rest.length, h, s, prs.foldl (applyPR s.attrs.length) (acc0 C)⟩ hne rfl rfl rfl]
    simp only [finalizePG, List.flatMap_cons, List.foldl_append, metaOf_cons, DS.mk.injEq, true_and]
    by_cases hk : kn = h.name <;> simp [hk]

/-- No key before the last finalizes the graph. -/
theorem run_prefix_pending (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn)
    (items : List (Nm × List (PR M Tn))) (hK : h.keyCount = items.length) (j : Nat) (hj : 0 < j)
    (hjK : j < items.length) :
    (runKeys C h s DS.empty (items.take j)).finalized = none ∧
    ∃ pg, (runKeys C h s DS.empty (items.take j)).pending = some pg ∧ pg.remaining = items.length - j := by
  have key : ∀ (rest : List (Nm × List (PR M Tn))) (ds : DS M Tn Nm Ix Cn) (pg : PG M Tn Nm Ix Cn),
      ds.pending = some pg → rest.length < pg.remaining → ds.finalized = none →
      (runKeys C h s ds rest).finalized = none ∧
      ∃ pg', (runKeys C h s ds rest).pending = some pg' ∧ pg'.remaining = pg.remaining - rest.length := by
    intro rest
    induction rest with
    | nil => intro ds pg hp _ hf; exact ⟨hf, pg, hp, by simp⟩
    | cons it rest ih =>
      intro ds pg hp hl hf
      obtain ⟨kn, prs⟩ := it
      simp only [List.length_cons] at hl
      have hnz : pg.remaining - 1 ≠ 0 := by omega
      let pg1 : PG M Tn Nm Ix Cn := ⟨pg.remaining - 1, pg.hdr, pg.sch,
        prs.foldl (applyPR pg.sch.attrs.length) pg.acc⟩
      let mk := if kn ≠ h.name then ds.metaKeys ++ [kn] else ds.metaKeys
      have hstep : (stepKey C ds kn h s prs).1 = ⟨some pg1, ds.finalized, mk⟩ := by
        simp [stepKey, hp, hnz, pg1, mk]
      have e1 : runKeys C h s ds ((kn, prs) :: rest) = runKeys C h s ⟨some pg1, ds.finalized, mk⟩ rest := by
        simp only [runKeys, hstep]
      rw [e1]
      obtain ⟨i1, pg', i2, i3⟩ := ih ⟨some pg1, ds.finalized, mk⟩ pg1 rfl (by simp [pg1]; omega) hf
      exact ⟨i1, pg', i2, by rw [i3]; simp [pg1]; omega⟩
  obtain ⟨it, rest⟩ := items
  · simp at hjK
  rename_i it rest
  obtain ⟨kn, prs⟩ := it
  obtain ⟨j', rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
  simp only [List.take_succ_cons, runKeys, stepKey, DS.empty]
  simp only [List.length_cons] at hK hjK
  have hnz : h.keyCount - 1 ≠ 0 := by omega
  simp only [hnz, ite_false]
  obtain ⟨i1, pg', i2, i3⟩ := key (rest.take j') ⟨some ⟨h.keyCount - 1, h, s,
      prs.foldl (applyPR s.attrs.length) (acc0 C)⟩, none, if kn ≠ h.name then [] ++ [kn] else []⟩
    ⟨h.keyCount - 1, h, s, prs.foldl (applyPR s.attrs.length) (acc0 C)⟩ rfl (by simp; omega) rfl
  refine ⟨i1, pg', i2, ?_⟩
  rw [i3]; simp; omega

end GraphPersist.Persist
