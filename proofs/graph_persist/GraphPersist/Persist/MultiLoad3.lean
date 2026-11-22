import GraphPersist.Persist.MultiLoad2
/-! # Multi-key save → load in any order (`multi_load`) -/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

abbrev capOf (vmax : Nat) : Nat := if vmax = 0 then U64MAX else vmax
abbrev kf (g : G M Tn Nm Ix Cn) (vmax : Nat) : List (List Entry) :=
  keysFrom (capOf vmax) (keyCount (total g) vmax) (kindsOf g)

theorem kf_entity (g : G M Tn Nm Ix Cn) (vmax : Nat) : ∀ e ∈ (kf g vmax).flatten, IsEntity e.st := by
  have key : ∀ (cap n : Nat) (ts : List Kind), (∀ k ∈ ts, IsEntity k.1) →
      ∀ es ∈ keysFrom cap n ts, ∀ e ∈ es, IsEntity e.st := by
    intro cap n
    induction n with
    | zero => intro ts _ es hes; simp [keysFrom] at hes
    | succ n ih =>
      intro ts hts es hes e he
      have hf := fillKey_kinds IsEntity _ cap ts rfl hts
      simp only [keysFrom, List.mem_cons] at hes
      rcases hes with rfl | hes
      · exact hf.1 e he
      · exact ih _ hf.2 es hes e he
  intro e he
  obtain ⟨es, hes, he⟩ := List.mem_flatten.1 he
  refine key _ _ _ ?_ es hes e he
  intro k hk; simp only [kindsOf, List.mem_map, List.mem_filter] at hk
  obtain ⟨p, ⟨hp, _⟩, rfl⟩ := hk
  simp at hp; rcases hp with rfl | rfl | rfl | rfl <;> simp [IsEntity]

theorem kf_length (g : G M Tn Nm Ix Cn) (vmax : Nat) : (kf g vmax).length = keyCount (total g) vmax := by
  have : ∀ n ts, (keysFrom (capOf vmax) n ts).length = n := by
    intro n; induction n with
    | zero => intro ts; rfl
    | succ n ih => intro ts; simp [keysFrom, ih]
  exact this _ _

theorem buildMulti_length (g : G M Tn Nm Ix Cn) (vmax : Nat) : (buildMulti g vmax).length = (kf g vmax).length := by
  simp only [buildMulti]; split <;> rename_i heq <;> simp [heq]

theorem keyAt_zero (g : G M Tn Nm Ix Cn) (vmax : Nat) (h : 0 < (buildMulti g vmax).length) :
    ∃ k0 ks, kf g vmax = k0 :: ks ∧ keyAt (buildMulti g vmax) 0 = k0 ++ matrixEntries g := by
  rw [buildMulti_length] at h
  cases hk : kf g vmax with
  | nil => simp [hk] at h
  | cons k0 ks => exact ⟨k0, ks, rfl, by simp [keyAt, buildMulti, hk]⟩

/-! ## Contributions per key -/

theorem entity_contrib (g : G M Tn Nm Ix Cn) (es : List Entry) (h : ∀ e ∈ es, IsEntity e.st) (d : M) :
    lmsOf (es.map (prOfE g)) = [] ∧ tnsOf (es.map (prOfE g)) = [] ∧
    adjOf (es.map (prOfE g)) d = d ∧ lblsOf (es.map (prOfE g)) d = d := by
  induction es with
  | nil => simp [lmsOf, tnsOf, adjOf, lblsOf]
  | cons e es ih =>
    obtain ⟨i1, i2, i3, i4⟩ := ih (fun x hx => h x (by simp [hx]))
    have he := h e (by simp)
    obtain ⟨st, c, o⟩ := e
    rcases he with rfl | rfl | rfl | rfl <;>
      simp [prOfE, lmsOf, tnsOf, adjOf, lblsOf, i1, i2, i3, i4]

theorem matrix_contrib (g : G M Tn Nm Ix Cn) (d : M) :
    lmsOf ((matrixEntries g).map (prOfE g)) = g.labelMats ∧ tnsOf ((matrixEntries g).map (prOfE g)) = g.tensors ∧
    adjOf ((matrixEntries g).map (prOfE g)) d = g.adj ∧ lblsOf ((matrixEntries g).map (prOfE g)) d = g.lbls := by
  simp only [matrixEntries]
  by_cases h1 : g.labelMats.length > 0 <;> by_cases h2 : g.tensors.length > 0 <;>
    simp [h1, h2, prOfE, prOf, lmsOf, tnsOf, adjOf, lblsOf] <;>
    simp_all [List.length_eq_zero_iff]

theorem lmsOf_flatMap (f : Nat → List (PR M Tn)) : ∀ ord : List Nat, lmsOf (ord.flatMap f) = ord.flatMap (fun k => lmsOf (f k))
  | [] => rfl
  | k :: ks => by simp [List.flatMap_cons, lmsOf_append, lmsOf_flatMap f ks]
theorem tnsOf_flatMap (f : Nat → List (PR M Tn)) : ∀ ord : List Nat, tnsOf (ord.flatMap f) = ord.flatMap (fun k => tnsOf (f k))
  | [] => rfl
  | k :: ks => by simp [List.flatMap_cons, tnsOf_append, tnsOf_flatMap f ks]
theorem adjOf_flatMap (f : Nat → List (PR M Tn)) : ∀ (ord : List Nat) (d : M),
    adjOf (ord.flatMap f) d = ord.foldl (fun b k => adjOf (f k) b) d
  | [], _ => rfl
  | k :: ks, d => by simp [List.flatMap_cons, adjOf_append, adjOf_flatMap f ks]
theorem lblsOf_flatMap (f : Nat → List (PR M Tn)) : ∀ (ord : List Nat) (d : M),
    lblsOf (ord.flatMap f) d = ord.foldl (fun b k => lblsOf (f k) b) d
  | [], _ => rfl
  | k :: ks, d => by simp [List.flatMap_cons, lblsOf_append, lblsOf_flatMap f ks]

/-- Key `k ≠ 0` holds entity slices only. -/
theorem keyAt_entity (g : G M Tn Nm Ix Cn) (vmax k : Nat) (hk : k ≠ 0) :
    ∀ e ∈ keyAt (buildMulti g vmax) k, IsEntity e.st := by
  intro e he
  rcases keyAt_cases g vmax k e he with h | ⟨h, _⟩
  · exact kf_entity g vmax e h
  · exact absurd h hk

/-! ## The theorem -/

theorem multi_acc (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 0 < (buildMulti g vmax).length)
    (ord : List Nat) (hord : ord.Perm (List.range (buildMulti g vmax).length)) :
    (ord.flatMap fun k => (keyAt (buildMulti g vmax) k).map (prOfE g)).foldl (applyPR g.sch.attrs.length) (acc0 C) =
      (restoreOf g g.name).acc := by
  obtain ⟨P, hP⟩ : ∃ P, P = ord.flatMap fun k => (keyAt (buildMulti g vmax) k).map (prOfE g) := ⟨_, rfl⟩
  rw [← hP]
  have hfaith : ∀ p ∈ P, Faithful g p := by
    intro p hp
    rw [hP] at hp
    obtain ⟨k, _, hp⟩ := List.mem_flatMap.1 hp
    obtain ⟨e, _, rfl⟩ := List.mem_map.1 hp
    exact prOfE_faithful g e
  have hNg : ∀ i sp, g.nodes i = some sp → GoodSpan g.sch.attrs.length sp := fun i sp h => (hw.nk.2.2.2 i sp h).2
  have hEg : ∀ i sp, g.edges i = some sp → GoodSpan g.sch.attrs.length sp := fun i sp h => (hw.ek.2.2.2 i sp h).2
  rw [fold_char g _ hw.lim hNg hEg P (acc0 C) hfaith]
  have hnd : ord.Nodup := hord.nodup_iff.2 List.nodup_range
  have h0 : 0 ∈ ord := hord.mem_iff.2 (List.mem_range.2 hK)
  have hin : ∀ k, k < (buildMulti g vmax).length → k ∈ ord := fun k hk => hord.mem_iff.2 (List.mem_range.2 hk)
  -- an entry of some key, decoded, is in `P`
  have hPmem : ∀ e ∈ (kf g vmax).flatten, prOfE g e ∈ P := by
    intro e he
    obtain ⟨k, hk, hek⟩ := mem_of_keysFrom g vmax e he
    rw [hP]
    exact List.mem_flatMap.2 ⟨k, hin k hk, List.mem_map.2 ⟨e, hek, rfl⟩⟩
  -- the kinds
  have hlenN := live_length g.nodeCount g.delNodes hw.nk.2.1 hw.nk.2.2.1
  have hlenE := live_length g.edgeCount g.delEdges hw.ek.2.1 hw.ek.2.2.1
  have nodes_cov : ∀ j ∈ live g.nodeCount g.delNodes, j ∈ nodeIds P := by
    intro j hj
    obtain ⟨o, ho, rfl⟩ := List.mem_iff_getElem.1 hj
    obtain ⟨e, he, hst, h1, h2⟩ := covered_kind g vmax ht .nodes o (by simp [kindTotal]; omega) (by simp [IsEntity])
    rw [mem_nodeIds]
    obtain ⟨st, c, off⟩ := e
    simp only at hst; subst hst
    have hm := hPmem _ he
    simp only [prOfE] at hm
    refine ⟨_, hm, ?_⟩
    simp only [map_fst_rawOf]
    exact slice_mem _ _ _ _ h1 h2 ho
  have edges_cov : ∀ j ∈ live g.edgeCount g.delEdges, j ∈ edgeIds P := by
    intro j hj
    obtain ⟨o, ho, rfl⟩ := List.mem_iff_getElem.1 hj
    obtain ⟨e, he, hst, h1, h2⟩ := covered_kind g vmax ht .edges o (by simp [kindTotal]; omega) (by simp [IsEntity])
    rw [mem_edgeIds]
    obtain ⟨st, c, off⟩ := e
    simp only at hst; subst hst
    have hm := hPmem _ he
    simp only [prOfE] at hm
    refine ⟨_, hm, ?_⟩
    simp only [map_fst_rawOf]
    exact slice_mem _ _ _ _ h1 h2 ho
  have delN_iff : ∀ j, j ∈ delNIds P ↔ j ∈ g.delNodes := by
    intro j
    rw [mem_delNIds]
    constructor
    · rintro ⟨l, hl, hj⟩
      rw [hP] at hl
      obtain ⟨k, _, hk⟩ := List.mem_flatMap.1 hl
      obtain ⟨e, _, he⟩ := List.mem_map.1 hk
      obtain ⟨_, rfl⟩ := prOfE_delN g e l he
      exact List.mem_of_mem_drop (List.mem_of_mem_take hj)
    · intro hj
      obtain ⟨o, ho, rfl⟩ := List.mem_iff_getElem.1 hj
      obtain ⟨e, he, hst, h1, h2⟩ := covered_kind g vmax ht .delNodes o (by simp [kindTotal]; omega) (by simp [IsEntity])
      obtain ⟨st, c, off⟩ := e
      simp only at hst; subst hst
      have hm := hPmem _ he
      simp only [prOfE] at hm
      exact ⟨_, hm, slice_mem _ _ _ _ h1 h2 ho⟩
  have delE_iff : ∀ j, j ∈ delEIds P ↔ j ∈ g.delEdges := by
    intro j
    rw [mem_delEIds]
    constructor
    · rintro ⟨l, hl, hj⟩
      rw [hP] at hl
      obtain ⟨k, _, hk⟩ := List.mem_flatMap.1 hl
      obtain ⟨e, _, he⟩ := List.mem_map.1 hk
      obtain ⟨_, rfl⟩ := prOfE_delE g e l he
      exact List.mem_of_mem_drop (List.mem_of_mem_take hj)
    · intro hj
      obtain ⟨o, ho, rfl⟩ := List.mem_iff_getElem.1 hj
      obtain ⟨e, he, hst, h1, h2⟩ := covered_kind g vmax ht .delEdges o (by simp [kindTotal]; omega) (by simp [IsEntity])
      obtain ⟨st, c, off⟩ := e
      simp only at hst; subst hst
      have hm := hPmem _ he
      simp only [prOfE] at hm
      exact ⟨_, hm, slice_mem _ _ _ _ h1 h2 ho⟩
  -- the matrices: only key 0 contributes
  obtain ⟨k0, ks, hkf, hk0⟩ := keyAt_zero g vmax hK
  have hk0e : ∀ e ∈ k0, IsEntity e.st := fun e he => kf_entity g vmax e (by rw [hkf]; simp [he])
  have hlms : lmsOf P = g.labelMats := by
    rw [hP, lmsOf_flatMap, flatMap_single _ ord hnd h0 (fun k hk => (entity_contrib g _ (keyAt_entity g vmax k hk) C.m0).1)]
    rw [hk0, List.map_append, lmsOf_append, (entity_contrib g k0 hk0e C.m0).1, (matrix_contrib g C.m0).1]; rfl
  have htns : tnsOf P = g.tensors := by
    rw [hP, tnsOf_flatMap, flatMap_single _ ord hnd h0 (fun k hk => (entity_contrib g _ (keyAt_entity g vmax k hk) C.m0).2.1)]
    rw [hk0, List.map_append, tnsOf_append, (entity_contrib g k0 hk0e C.m0).2.1, (matrix_contrib g C.m0).2.1]; rfl
  have hadj : adjOf P C.m0 = g.adj := by
    rw [hP, adjOf_flatMap]
    rw [fold_single (fun k b => adjOf ((keyAt (buildMulti g vmax) k).map (prOfE g)) b) ord hnd h0
      (fun k hk d => (entity_contrib g _ (keyAt_entity g vmax k hk) d).2.2.1)
      (fun d d' => by
        rw [hk0, List.map_append, adjOf_append, adjOf_append, (entity_contrib g k0 hk0e d).2.2.1,
          (entity_contrib g k0 hk0e d').2.2.1, (matrix_contrib g d).2.2.1, (matrix_contrib g d').2.2.1])]
    rw [hk0, List.map_append, adjOf_append, (entity_contrib g k0 hk0e C.m0).2.2.1, (matrix_contrib g C.m0).2.2.1]
  have hlbls : lblsOf P C.m0 = g.lbls := by
    rw [hP, lblsOf_flatMap]
    rw [fold_single (fun k b => lblsOf ((keyAt (buildMulti g vmax) k).map (prOfE g)) b) ord hnd h0
      (fun k hk d => (entity_contrib g _ (keyAt_entity g vmax k hk) d).2.2.2)
      (fun d d' => by
        rw [hk0, List.map_append, lblsOf_append, lblsOf_append, (entity_contrib g k0 hk0e d).2.2.2,
          (entity_contrib g k0 hk0e d').2.2.2, (matrix_contrib g d).2.2.2, (matrix_contrib g d').2.2.2])]
    rw [hk0, List.map_append, lblsOf_append, (entity_contrib g k0 hk0e C.m0).2.2.2, (matrix_contrib g C.m0).2.2.2]
  simp only [restoreOf, acc0, hlms, htns, hadj, hlbls, List.nil_append, Bool.false_or]
  apply Acc.ext
  · funext j
    by_cases hs : g.nodes j = none
    · simp [hs]
    · obtain ⟨sp, hsp⟩ := Option.ne_none_iff_exists'.1 hs
      simp [hs, nodes_cov j (hw.nk.2.2.2 j sp hsp).1]
  · funext j
    by_cases hs : g.edges j = none
    · simp [hs]
    · obtain ⟨sp, hsp⟩ := Option.ne_none_iff_exists'.1 hs
      simp [hs, edges_cov j (hw.ek.2.2.2 j sp hsp).1]
  · funext j; simp [delN_iff]
  · funext j; simp [delE_iff]
  all_goals rfl

theorem keyAt_nonzero (g : G M Tn Nm Ix Cn) (vmax k : Nat) (hk : k ≠ 0) :
    keyAt (buildMulti g vmax) k = keyAt (kf g vmax) k := by
  simp only [keyAt, buildMulti, kf, capOf]
  split
  · rename_i heq; rw [heq]
  · rename_i k0 ks heq
    rw [heq]
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    simp

theorem kf_bounds (g : G M Tn Nm Ix Cn) (vmax : Nat) :
    ∀ es ∈ kf g vmax, (∀ e ∈ es, 0 < e.count) ∧ es.length ≤ 4 :=
  keysFrom_entries _ _ _ (by simp only [kindsOf, List.length_map]; exact Nat.le_trans (List.length_filter_le _ _) (by simp))

theorem keyAt_kf_length (g : G M Tn Nm Ix Cn) (vmax k : Nat) : (keyAt (kf g vmax) k).length ≤ 4 := by
  simp only [keyAt]
  cases h : (kf g vmax)[k]? with
  | none => simp
  | some l => simpa using (kf_bounds g vmax l (List.mem_of_getElem? h)).2

theorem kindTotal_le (g : G M Tn Nm Ix Cn) (st : St) : kindTotal g st ≤ total g := by
  cases st <;> simp [kindTotal, total] <;> omega

/-- Every key of the layout satisfies `key_rt`'s side conditions. -/
theorem layout_ok (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 0 < (buildMulti g vmax).length) (k : Nat) :
    (∀ e ∈ keyAt (buildMulti g vmax) k, EntryOK g e) ∧ (keyAt (buildMulti g vmax) k).length < 2 ^ 64 ∧
    ∀ e ∈ keyAt (buildMulti g vmax) k, e.count < 2 ^ 64 := by
  have hU : U64MAX < 2 ^ 64 := by decide
  have hent : ∀ e ∈ (kf g vmax).flatten, EntryOK g e ∧ e.count < 2 ^ 64 := by
    intro e he
    obtain ⟨es, hes, hein⟩ := List.mem_flatten.1 he
    have hc := (kf_bounds g vmax es hes).1 e hein
    have hs := slice_in_kind g vmax ht e he hc
    have := kindTotal_le g e.st
    exact ⟨Or.inl ⟨kf_entity g vmax e he, hc, hs⟩, by omega⟩
  have hmat : ∀ e ∈ matrixEntries g, e.count < 2 ^ 64 := by
    intro e he
    simp only [matrixEntries, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
    have := hw.lm.1; have := hw.tn.1
    rcases he with (he | he) | he | he <;> (try (split at he <;> simp at he)) <;> subst he <;> simp <;> omega
  refine ⟨fun e he => ?_, ?_, fun e he => ?_⟩
  · rcases keyAt_cases g vmax k e he with h | ⟨_, h⟩
    · exact (hent e h).1
    · exact Or.inr h
  · by_cases hk0 : k = 0
    · subst hk0
      obtain ⟨k0, ks, hkf, hk0⟩ := keyAt_zero g vmax hK
      rw [hk0, List.length_append]
      have h1 := (kf_bounds g vmax k0 (by rw [hkf]; simp)).2
      have h2 : (matrixEntries g).length ≤ 4 := by
        simp only [matrixEntries, List.length_append]; split <;> split <;> simp
      omega
    · rw [keyAt_nonzero g vmax k hk0]; have := keyAt_kf_length g vmax k; omega
  · rcases keyAt_cases g vmax k e he with h | ⟨_, h⟩
    · exact (hent e h).2
    · exact hmat e h

/-- **Multi-key save → load = the graph, in any key order.** Lay a graph out over
`K ≥ 2` keys with `build_multi_key_payloads`; whatever order Redis loads them in, (1) each
key reads back (with truncation detection, `RT`), and (2) the `DECODE_STATE` machine ends
with no pending graph, the finalized graph equal to the single-key restore of the graph,
and exactly the keys not named after the graph recorded as meta keys. -/
theorem multi_load [DecidableEq Nm] (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 2 ≤ (buildMulti g vmax).length)
    (hH : C.H.good (hdrOf g g.name (buildMulti g vmax).length))
    (ord : List Nat) (hord : ord.Perm (List.range (buildMulti g vmax).length)) (names : Nat → Nm) :
    (∀ k ∈ ord, RT (rdbKey C) (encodeGraph C g g.name (keyAt (buildMulti g vmax) k) (buildMulti g vmax).length)
      (hdrOf g g.name (buildMulti g vmax).length, g.sch, (keyAt (buildMulti g vmax) k).map (prOfE g))) ∧
    runKeys C (hdrOf g g.name (buildMulti g vmax).length) g.sch DS.empty
        (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g))) =
      ⟨none, some (restoreOf g g.name),
       metaOf (hdrOf g g.name (buildMulti g vmax).length)
        (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g)))⟩ := by
  refine ⟨fun k _ => ?_, ?_⟩
  · obtain ⟨h1, h2, h3⟩ := layout_ok C g hw vmax ht (by omega) k
    exact key_rt C g hw _ hH _ h1 h2 h3
  · have hlen : (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g))).length =
        (buildMulti g vmax).length := by simp [hord.length_eq]
    rw [run_keys C _ g.sch _ (by simp [hdrOf, hlen]) (by omega)]
    have hacc := multi_acc C g hw vmax ht (by omega) ord hord
    simp only [List.flatMap_map] at hacc ⊢
    rw [hacc]
    rfl

/-- With the graph's own key named after it and every virtual key not, the meta keys are
exactly the virtual keys, in load order (`delete_stale_virtual_keys` deletes these). -/
theorem multi_meta [DecidableEq Nm] (h : Hdr Nm) (ord : List Nat) (names : Nat → Nm) (f : Nat → List (PR M Tn))
    (h0 : names 0 = h.name) (hv : ∀ k, k ≠ 0 → names k ≠ h.name) :
    metaOf h (ord.map fun k => (names k, f k)) = (ord.filter fun k => k ≠ 0).map names := by
  induction ord with
  | nil => rfl
  | cons k ks ih =>
    simp only [List.map_cons, metaOf_cons, ih]
    by_cases hk : k = 0
    · subst hk; simp [h0]
    · simp [hv k hk, hk]

end GraphPersist.Persist
