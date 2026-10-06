import PendingCommit.PendingDeg
/-
# `IndexDocs`, the deferred/published document logs, `clear`, `effects_count`,
# constructors and `end_segment` (pending.rs:248-386, :1455-1696)

* `IndexDocs::is_empty`/`absorb`/`merge`/`ids_by_slot` (:267-338): per slot,
  `absorb` and `merge` make `self` the union (`absorb_spec`, `merge_spec`;
  for edge removals the incoming endpoints win, `HashMap::extend`), `absorb`
  empties `other`; `ids_by_slot` is adds ∪ removes per slot (edge removals by
  key) (`ids_spec`).
* `IndexDocs::commit` (:344), `commit_deferred_indexes` (:1572),
  `take_deferred_indexes` (:1499): the graph-side index writes are
  uninterpreted (`ci`, `ce`; RediSearch); proved: what is handed over, that the
  deferred set is drained, and that `published` grows by it.
* `resync_published_indexes` (:1517): every published `(slot, id)` is re-added
  iff the committed version still has it, otherwise removed (edges: removed
  with the private version's endpoints, dropped if neither version has the
  edge); `published` is emptied (`resync_spec`).
* `end_segment` (:1569, was `clear` before #2846): both branches (background
  drop or in place) give the same state — every per-commit field empty,
  `published`, `deferred_docs` and the schema baseline untouched
  (`clear_spec`) — and the graph's id batches are rolled in the same call
  (`endSegment_spec`).
* `effects_count` (:1680) is the documented sum; it does not count
  `cancelled_*` (`effects_ignores_cancelled`, the suspicion noted in the root
  file).
* `new`/`default` (:353, :346), `set_schema_baseline` (:381): field-level
  equations. (`node_id_space`, `rel_id_space`, `issued_*`,
  `open_id_boundaries` were removed by #2846.)
-/
namespace PendingCommit.PD
open PA PS

def memD (m : FMap (List Nat)) (s x : Nat) : Prop := x ∈ (fget m s).getD []

/-- `*self.map.entry(slot).or_default() |= ids`. -/
def uni (m : FMap (List Nat)) (s : Nat) (ids : List Nat) : FMap (List Nat) := fset m s ((fget m s).getD [] ++ ids)

def foldUni (m : FMap (List Nat)) (L : FMap (List Nat)) : FMap (List Nat) := L.foldl (fun m p => uni m p.1 p.2) m

theorem foldUni_mem : ∀ (L m : FMap (List Nat)) (s x : Nat),
    memD (foldUni m L) s x ↔ memD m s x ∨ ∃ ids, (s, ids) ∈ L ∧ x ∈ ids
  | [], m, s, x => by simp [foldUni]
  | (a, ids) :: L, m, s, x => by
    simp only [foldUni, List.foldl_cons] at *
    rw [show L.foldl (fun m p => uni m p.1 p.2) (uni m a ids) = foldUni (uni m a ids) L from rfl, foldUni_mem L]
    simp only [memD, uni, fget_fset]
    by_cases hs : s = a
    · subst hs; simp only [ite_true, Option.getD_some, List.mem_append, List.mem_cons]
      constructor
      · rintro ((h | h) | ⟨ids', h1, h2⟩)
        · exact .inl h
        · exact .inr ⟨ids, .inl rfl, h⟩
        · exact .inr ⟨ids', .inr h1, h2⟩
      · rintro (h | ⟨ids', hm, h2⟩)
        · exact .inl (.inl h)
        · rcases hm with he | h1
          · injection he with _ he2; subst he2; exact .inl (.inr h2)
          · exact .inr ⟨ids', h1, h2⟩
    · simp only [hs, ite_false, List.mem_cons]
      constructor
      · rintro (h | ⟨ids', h1, h2⟩)
        · exact .inl h
        · exact .inr ⟨ids', .inr h1, h2⟩
      · rintro (h | ⟨ids', hm, h2⟩)
        · exact .inl h
        · rcases hm with he | h1
          · injection he with he1; exact absurd he1 hs
          · exact .inr ⟨ids', h1, h2⟩

theorem memD_of_list (L : FMap (List Nat)) (hk : (L.map (·.1)).Nodup) (s x : Nat) :
    (∃ ids, (s, ids) ∈ L ∧ x ∈ ids) ↔ memD L s x := by
  unfold memD
  constructor
  · rintro ⟨ids, hm, hx⟩; rw [fget_of_mem L hk s ids hm]; exact hx
  · intro h
    cases e : fget L s with
    | none => rw [e] at h; simp at h
    | some ids => rw [e] at h; exact ⟨ids, mem_of_fget L s ids e, h⟩

/-- Inner-map `extend`: later entries win. -/
def fextend (m : FMap (Nat × Nat)) (L : FMap (Nat × Nat)) : FMap (Nat × Nat) := L.foldl (fun m p => fset m p.1 p.2) m

theorem fextend_get : ∀ (L m : FMap (Nat × Nat)), (L.map (·.1)).Nodup → ∀ (k : Nat),
    fget (fextend m L) k = match fget L k with | some v => some v | none => fget m k
  | [], m, _, k => by simp [fextend, fget]
  | (a, v) :: L, m, hnd, k => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    simp only [fextend, List.foldl_cons]
    rw [show L.foldl (fun m p => fset m p.1 p.2) (fset m a v) = fextend (fset m a v) L from rfl,
      fextend_get L _ hnd.2]
    simp only [fget]
    by_cases hk : a = k
    · subst hk
      have : fget L a = none := by
        cases e : fget L a with
        | none => rfl
        | some w => exact absurd (List.mem_map_of_mem (f := (·.1)) (mem_of_fget L a w e)) hnd.1
      simp [this]
    · simp only [hk, ite_false, fget_fset, Ne.symm hk]

structure Docs where
  nodeAdds : FMap (List Nat)
  nodeRemoves : FMap (List Nat)
  edgeAdds : FMap (List Nat)
  edgeRemoves : FMap (FMap (Nat × Nat))

def Docs.empty : Docs := ⟨[], [], [], []⟩

/-- `IndexDocs::is_empty` (:267). -/
def Docs.isEmpty (d : Docs) : Bool := d.nodeAdds.isEmpty && d.nodeRemoves.isEmpty && d.edgeAdds.isEmpty && d.edgeRemoves.isEmpty

theorem isEmpty_spec (d : Docs) : d.isEmpty = true ↔ d = Docs.empty := by
  cases d; simp [Docs.isEmpty, Docs.empty, and_assoc]

theorem fset_keys {V : Type} (m : FMap V) (hk : (m.map (·.1)).Nodup) (k : Nat) (v : V) :
    ((fset m k v).map (·.1)).Nodup := by
  unfold fset
  simp only [List.map_cons, List.nodup_cons, List.mem_map, List.mem_filter, bne_iff_ne, ne_eq]
  refine ⟨fun ⟨x, ⟨_, hx⟩, he⟩ => hx he, ?_⟩
  have := (List.filter_sublist (p := fun x : Nat × V => x.1 != k) (l := m)).map (·.1)
  exact hk.sublist this

theorem foldUni_keys : ∀ (L m : FMap (List Nat)), (m.map (·.1)).Nodup → ((foldUni m L).map (·.1)).Nodup
  | [], _, h => h
  | (a, ids) :: L, m, h => foldUni_keys L _ (fset_keys m h a _)

/-- `absorb`/`merge` for edge removals: inner maps extended. -/
def absorbE (m L : FMap (FMap (Nat × Nat))) : FMap (FMap (Nat × Nat)) :=
  L.foldl (fun m p => fset m p.1 (fextend ((fget m p.1).getD []) p.2)) m

def egetD (m : FMap (FMap (Nat × Nat))) (s e : Nat) : Option (Nat × Nat) := fget ((fget m s).getD []) e

theorem absorbE_get : ∀ (L m : FMap (FMap (Nat × Nat))), (L.map (·.1)).Nodup → (∀ p ∈ L, (p.2.map (·.1)).Nodup) →
    ∀ s e, egetD (absorbE m L) s e = match egetD L s e with | some v => some v | none => egetD m s e
  | [], m, _, _, s, e => by simp [absorbE, egetD, fget]
  | (a, inner) :: L, m, hk, hi, s, e => by
    simp only [List.map_cons, List.nodup_cons] at hk
    simp only [absorbE, List.foldl_cons]
    rw [show L.foldl (fun m p => fset m p.1 (fextend ((fget m p.1).getD []) p.2))
        (fset m a (fextend ((fget m a).getD []) inner)) = absorbE (fset m a (fextend ((fget m a).getD []) inner)) L
        from rfl, absorbE_get L _ hk.2 (fun p hp => hi p (by simp [hp]))]
    simp only [egetD, fget_fset, fget]
    by_cases hs : a = s
    · subst hs
      have hL : fget L a = none := by
        cases e' : fget L a with
        | none => rfl
        | some w => exact absurd (List.mem_map_of_mem (f := (·.1)) (mem_of_fget L a w e')) hk.1
      simp only [hL, ite_true, Option.getD_none, Option.getD_some]
      simp only [fget, Option.getD_some]
      rw [fextend_get inner _ (hi (a, inner) (by simp)) e]
    · simp only [hs, ite_false, Ne.symm hs]

/-- `IndexDocs::absorb` (:279): `(self ∪ other, empty)`. `merge` (:298) is `.1`. -/
def absorb (self other : Docs) : Docs × Docs :=
  (⟨foldUni self.nodeAdds other.nodeAdds, foldUni self.nodeRemoves other.nodeRemoves,
    foldUni self.edgeAdds other.edgeAdds, absorbE self.edgeRemoves other.edgeRemoves⟩, Docs.empty)

def merge (self other : Docs) : Docs := (absorb self other).1

/-- Keys unique at both levels (`FxHashMap`s). -/
structure DocsWF (d : Docs) : Prop where
  na : (d.nodeAdds.map (·.1)).Nodup
  nr : (d.nodeRemoves.map (·.1)).Nodup
  ea : (d.edgeAdds.map (·.1)).Nodup
  er : (d.edgeRemoves.map (·.1)).Nodup
  eri : ∀ p ∈ d.edgeRemoves, (p.2.map (·.1)).Nodup

/-- **`absorb_spec`** (and `merge_spec`, its first component). -/
theorem absorb_spec (self other : Docs) (ho : DocsWF other) (s x : Nat) :
    (memD (absorb self other).1.nodeAdds s x ↔ memD self.nodeAdds s x ∨ memD other.nodeAdds s x) ∧
    (memD (absorb self other).1.nodeRemoves s x ↔ memD self.nodeRemoves s x ∨ memD other.nodeRemoves s x) ∧
    (memD (absorb self other).1.edgeAdds s x ↔ memD self.edgeAdds s x ∨ memD other.edgeAdds s x) ∧
    (egetD (absorb self other).1.edgeRemoves s x =
      match egetD other.edgeRemoves s x with | some v => some v | none => egetD self.edgeRemoves s x) ∧
    (absorb self other).2 = Docs.empty := by
  refine ⟨?_, ?_, ?_, absorbE_get _ _ ho.er ho.eri s x, rfl⟩
  · simp only [absorb]; rw [foldUni_mem, memD_of_list _ ho.na]
  · simp only [absorb]; rw [foldUni_mem, memD_of_list _ ho.nr]
  · simp only [absorb]; rw [foldUni_mem, memD_of_list _ ho.ea]

/-- `ids_by_slot` (:318). -/
def idsBySlot (d : Docs) : FMap (List Nat) × FMap (List Nat) :=
  (foldUni (foldUni [] d.nodeAdds) d.nodeRemoves,
   foldUni (foldUni [] d.edgeAdds) (d.edgeRemoves.map (fun p => (p.1, p.2.map (·.1)))))

theorem ids_spec (d : Docs) (h : DocsWF d) (s x : Nat) :
    (memD (idsBySlot d).1 s x ↔ memD d.nodeAdds s x ∨ memD d.nodeRemoves s x) ∧
    (memD (idsBySlot d).2 s x ↔ memD d.edgeAdds s x ∨ (egetD d.edgeRemoves s x).isSome) := by
  have e0 : ∀ s x, ¬ memD ([] : FMap (List Nat)) s x := fun s x => by simp [memD, fget]
  constructor
  · simp only [idsBySlot]; rw [foldUni_mem, foldUni_mem, memD_of_list _ h.na, memD_of_list _ h.nr]
    simp [e0]
  · simp only [idsBySlot]; rw [foldUni_mem, foldUni_mem, memD_of_list _ h.ea]
    simp only [e0, false_or]
    apply or_congr Iff.rfl
    constructor
    · rintro ⟨ids, hm, hx⟩
      simp only [List.mem_map] at hm
      obtain ⟨⟨s', inner⟩, hm', he⟩ := hm
      injection he with h1 h2; subst h1; subst h2
      simp only [egetD, fget_of_mem _ h.er _ _ hm', Option.getD_some]
      obtain ⟨⟨k, v⟩, hk, rfl⟩ := List.mem_map.1 hx
      rw [fget_of_mem _ (h.eri _ hm') _ _ hk]; rfl
    · intro hs
      unfold egetD at hs
      cases e1 : fget d.edgeRemoves s with
      | none => rw [e1] at hs; simp [fget] at hs
      | some inner =>
        rw [e1] at hs; simp only [Option.getD_some] at hs
        cases e2 : fget inner x with
        | none => rw [e2] at hs; simp at hs
        | some v =>
          exact ⟨inner.map (·.1), List.mem_map.2 ⟨(s, inner), mem_of_fget _ _ _ e1, rfl⟩,
            List.mem_map.2 ⟨(x, v), mem_of_fget _ _ _ e2, rfl⟩⟩

/-- `IndexDocs::commit` (:344): node docs then edge docs, through the graph's
(uninterpreted, RediSearch-backed) index writers. -/
def commitDocs {G : Type} (ci : G → FMap (List Nat) → FMap (List Nat) → G)
    (ce : G → FMap (List Nat) → FMap (FMap (Nat × Nat)) → G) (g : G) (d : Docs) : G :=
  ce (ci g d.nodeAdds d.nodeRemoves) d.edgeAdds d.edgeRemoves

theorem commitDocs_spec {G : Type} (ci : G → FMap (List Nat) → FMap (List Nat) → G)
    (ce : G → FMap (List Nat) → FMap (FMap (Nat × Nat)) → G) (g : G) (d : Docs) :
    commitDocs ci ce g d = ce (ci g d.nodeAdds d.nodeRemoves) d.edgeAdds d.edgeRemoves := rfl

/-! ## Per-query document logs -/

structure Logs where
  docs : Docs
  deferred : Docs
  published : Docs

/-- `take_deferred_indexes` (:1499). -/
def takeDeferred (l : Logs) : Docs × Logs := (l.deferred, { l with deferred := Docs.empty })

/-- `commit_deferred_indexes` (:1572). -/
def commitDeferred {G : Type} (ci : G → FMap (List Nat) → FMap (List Nat) → G)
    (ce : G → FMap (List Nat) → FMap (FMap (Nat × Nat)) → G) (l : Logs) (g : G) : Logs × G :=
  let (d, l1) := takeDeferred l
  ({ l1 with published := merge l1.published d }, commitDocs ci ce g d)

theorem commitDeferred_spec {G : Type} (ci : G → FMap (List Nat) → FMap (List Nat) → G)
    (ce : G → FMap (List Nat) → FMap (FMap (Nat × Nat)) → G) (l : Logs) (g : G) (hw : DocsWF l.deferred) (s x : Nat) :
    let r := commitDeferred ci ce l g
    r.1.deferred = Docs.empty ∧ r.1.docs = l.docs ∧ r.2 = commitDocs ci ce g l.deferred ∧
      (memD r.1.published.nodeAdds s x ↔ memD l.published.nodeAdds s x ∨ memD l.deferred.nodeAdds s x) ∧
      (memD r.1.published.nodeRemoves s x ↔ memD l.published.nodeRemoves s x ∨ memD l.deferred.nodeRemoves s x) ∧
      (memD r.1.published.edgeAdds s x ↔ memD l.published.edgeAdds s x ∨ memD l.deferred.edgeAdds s x) ∧
      (egetD r.1.published.edgeRemoves s x = match egetD l.deferred.edgeRemoves s x with
        | some v => some v | none => egetD l.published.edgeRemoves s x) := by
  obtain ⟨a, b, c, d, -⟩ := absorb_spec l.published l.deferred hw s x
  exact ⟨rfl, rfl, rfl, a, b, c, d⟩

/-- `resync_published_indexes` (:1517), with the committed version's label
test `lab id slot`, and the committed / private versions' edge endpoints. -/
def keepSlots (m : FMap (List Nat)) (f : Nat → Nat → Bool) : FMap (List Nat) :=
  m.filterMap (fun p => let l := p.2.filter (f p.1); if l = [] then none else some (p.1, l))

def resync (l : Logs) (lab : Nat → Nat → Bool) (cEp pEp : Nat → Option (Nat × Nat)) :
    Logs × Option (FMap (List Nat) × FMap (List Nat) × FMap (List Nat) × FMap (FMap (Nat × Nat))) :=
  let nodes := (idsBySlot l.published).1
  let edges := (idsBySlot l.published).2
  let l' := { l with published := Docs.empty }
  if nodes = [] ∧ edges = [] then (l', none)
  else (l', some (keepSlots nodes (fun s x => lab x s), keepSlots nodes (fun s x => !lab x s),
    keepSlots edges (fun _ x => (cEp x).isSome),
    edges.filterMap (fun p => let l := p.2.filterMap (fun x => if (cEp x).isSome then none else (pEp x).map (x, ·))
                              if l = [] then none else some (p.1, l))))

theorem keepSlots_none (f : Nat → Nat → Bool) (s : Nat) : ∀ (m : FMap (List Nat)), fget m s = none →
    fget (keepSlots m f) s = none
  | [], _ => rfl
  | (a, ids) :: m, h => by
    simp only [fget] at h
    split at h
    · cases h
    · rename_i ha
      have ih := keepSlots_none f s m h
      unfold keepSlots at ih ⊢
      simp only [List.filterMap_cons]
      by_cases hl : ids.filter (f a) = []
      · simp only [hl, ite_true]; exact ih
      · simp only [hl, ite_false, fget, ha]; exact ih

theorem keepSlots_mem (f : Nat → Nat → Bool) (s x : Nat) : ∀ (m : FMap (List Nat)), (m.map (·.1)).Nodup →
    (memD (keepSlots m f) s x ↔ memD m s x ∧ f s x = true)
  | [], _ => by simp [keepSlots, memD, fget]
  | (a, ids) :: m, hk => by
    simp only [List.map_cons, List.nodup_cons] at hk
    have ih := keepSlots_mem f s x m hk.2
    have hnone : fget m a = none := by
      cases e : fget m a with
      | none => rfl
      | some w => exact absurd (List.mem_map_of_mem (f := (·.1)) (mem_of_fget m a w e)) hk.1
    unfold keepSlots at ih ⊢
    simp only [List.filterMap_cons]
    by_cases hs : a = s
    · subst hs
      have k0 := keepSlots_none f a m hnone
      unfold keepSlots at k0
      by_cases hl : ids.filter (f a) = []
      · simp only [hl, ite_true]
        simp only [memD, k0, Option.getD_none, List.not_mem_nil, false_iff, fget, ite_true, Option.getD_some]
        intro ⟨hx, hf⟩
        have : x ∈ ids.filter (f a) := List.mem_filter.2 ⟨hx, hf⟩
        rw [hl] at this; simp at this
      · simp only [hl, ite_false, memD, fget, ite_true, Option.getD_some, List.mem_filter]
    · by_cases hl : ids.filter (f a) = []
      · simp only [hl, ite_true]; simp only [memD, fget, hs, ite_false] at ih ⊢; exact ih
      · simp only [hl, ite_false]; simp only [memD, fget, hs, ite_false] at ih ⊢; exact ih

/-- **`resync_spec`**: `published` is emptied; each published node document
is re-added iff the committed version still has the label, removed otherwise;
each published edge document is re-added iff the committed version has the
edge. -/
theorem resync_spec (l : Logs) (lab : Nat → Nat → Bool) (cEp pEp : Nat → Option (Nat × Nat))
    (hw : DocsWF l.published) (s x : Nat) :
    let r := resync l lab cEp pEp
    r.1.published = Docs.empty ∧ r.1.deferred = l.deferred ∧ r.1.docs = l.docs ∧
      ∀ out, r.2 = some out →
        (memD out.1 s x ↔ (memD l.published.nodeAdds s x ∨ memD l.published.nodeRemoves s x) ∧ lab x s = true) ∧
        (memD out.2.1 s x ↔ (memD l.published.nodeAdds s x ∨ memD l.published.nodeRemoves s x) ∧ lab x s = false) ∧
        (memD out.2.2.1 s x ↔ (memD l.published.edgeAdds s x ∨ (egetD l.published.edgeRemoves s x).isSome) ∧
          (cEp x).isSome = true) := by
  have hn : ((idsBySlot l.published).1.map (·.1)).Nodup := foldUni_keys _ _ (foldUni_keys _ _ List.nodup_nil)
  have he : ((idsBySlot l.published).2.map (·.1)).Nodup := foldUni_keys _ _ (foldUni_keys _ _ List.nodup_nil)
  have hi := ids_spec l.published hw s x
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold resync; dsimp only; split <;> rfl
  · unfold resync; dsimp only; split <;> rfl
  · unfold resync; dsimp only; split <;> rfl
  · intro out hout
    unfold resync at hout
    dsimp only at hout
    split at hout
    · cases hout
    · cases hout
      refine ⟨?_, ?_, ?_⟩
      · rw [keepSlots_mem _ _ _ _ hn, hi.1]
      · rw [keepSlots_mem _ _ _ _ hn, hi.1]; simp
      · rw [keepSlots_mem _ _ _ _ he, hi.2]

/-! ## The whole `Pending`: constructors, `clear`, `effects_count`, id spaces -/

structure Full where
  st : PS.P
  logs : Logs
  endpoints : List (Nat × Nat × Nat × Nat)
  nodeLabels : List (Nat × Nat)
  schema : Nat × Nat × Nat × Nat

/-- `Pending::new` / `Default` (:353, :346). -/
def newFull : Full := ⟨PS.P0, ⟨Docs.empty, Docs.empty, Docs.empty⟩, [], [], (0, 0, 0, 0)⟩

theorem new_spec : newFull.st.created = [] ∧ newFull.st.relsByType = [] ∧ newFull.st.deletedN = [] ∧
    newFull.st.cancelledN = [] ∧ newFull.st.newN = [] ∧ newFull.st.setL = [] ∧ newFull.logs.published = Docs.empty ∧
    newFull.st.taken = [] ∧ newFull.schema = (0, 0, 0, 0) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `set_schema_baseline` (:381): the four dictionary sizes. -/
def setSchemaBaseline (f : Full) (labels types nattrs rattrs : Nat) : Full := { f with schema := (labels, types, nattrs, rattrs) }

theorem accessors_spec (f : Full) (a b c d : Nat) :
    (setSchemaBaseline f a b c d).schema = (a, b, c, d) ∧ (setSchemaBaseline f a b c d).st = f.st :=
  ⟨rfl, rfl⟩

/-- The per-segment reset half of `end_segment` (:1569-1647). `offload` is
the `big_entries >= OFFLOAD_THRESHOLD` branch (maps moved to a background
thread instead of cleared in place). -/
def clearMaps (p : PS.P) : PS.P :=
  { p with newN := [], existN := [], newR := [], existR := [], setL := [], remL := [], relsByType := [] }

def clearBranch (f : Full) (offload : Bool) : Full :=
  -- `std::mem::take` (offload) or `.clear()` (in place): both leave the maps empty
  let st1 : PS.P := if offload then clearMaps f.st else clearMaps f.st
  let st2 : PS.P := { st1 with created := [], relTypes := [], deletedN := [], deletedR := [], cancelledN := [] }
  let st3 : PS.P := { st2 with taken := [], cancelledR := [] }
  { f with st := st3, endpoints := [], nodeLabels := [], logs := { f.logs with docs := Docs.empty } }

def bigEntries (f : Full) : Nat :=
  f.st.newN.length + f.st.existN.length + f.st.newR.length + f.st.existR.length + f.st.setL.length +
    f.st.remL.length + (f.st.relsByType.map (·.2.length)).sum

def clear (f : Full) : Full := clearBranch f (decide (bigEntries f ≥ 4096))

/-- **`clear_spec`**: both branches agree; every per-commit field is empty
and the per-query state (`published`, `deferred_docs`, schema baseline) is
untouched. -/
theorem clear_spec (f : Full) :
    clearBranch f true = clearBranch f false ∧
    let c := clear f
    c.st.created = [] ∧ c.st.relsByType = [] ∧ c.st.relTypes = [] ∧ c.st.deletedN = [] ∧ c.st.deletedR = [] ∧
    c.st.cancelledN = [] ∧ c.st.cancelledR = [] ∧ c.st.taken = [] ∧ c.st.newN = [] ∧ c.st.existN = [] ∧
    c.st.newR = [] ∧ c.st.existR = [] ∧ c.st.setL = [] ∧ c.st.remL = [] ∧ c.endpoints = [] ∧
    c.nodeLabels = [] ∧ c.logs.docs = Docs.empty ∧ c.logs.published = f.logs.published ∧
    c.logs.deferred = f.logs.deferred ∧ c.schema = f.schema := by
  refine ⟨by simp [clearBranch], ?_⟩
  unfold clear clearBranch; cases decide (bigEntries f ≥ 4096) <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `end_segment` (:1569): the reset above, then `g.roll_id_batches()`
(graph.rs:1584; `roll` here, `none` = its `Err`). The `Pending` reset happens
whether or not the roll succeeds; the error is returned to `CommitOp`. -/
def endSegment {Gr : Type} (roll : Gr → Option Gr) (f : Full) (g : Gr) : Full × Option Gr :=
  (clear f, roll g)

/-- **`endSegment_spec`**: the two halves of the segment boundary move
together — the mutation log is empty and the graph's id batches are rolled
(or the error is reported) in one call, so the next segment cannot
allocate against a stale boundary while holding an empty log, nor keep a
stale log over a fresh boundary. -/
theorem endSegment_spec {Gr : Type} (roll : Gr → Option Gr) (f : Full) (g : Gr) :
    (endSegment roll f g).1 = clear f ∧ (endSegment roll f g).2 = roll g := ⟨rfl, rfl⟩

/-- `effects_count` (:1684). -/
def effectsCount (p : PS.P) : Nat :=
  p.created.length + p.relTypes.length + p.deletedN.length + p.deletedR.length + p.newN.length +
    p.existN.length + p.newR.length + p.existR.length + (p.setL.map (·.2.length)).sum +
    (p.remL.map (·.2.length)).sum

/-- `CREATE (a) DELETE a` leaves only `cancelled_nodes = {a}`: `effects_count`
is 0, although the id went through the master's recycle bin (the suspicion
recorded in the root file; not confirmed against a replica). -/
theorem effects_ignores_cancelled : effectsCount { PS.P0 with cancelledN := [0] } = 0 := rfl

end PendingCommit.PD
