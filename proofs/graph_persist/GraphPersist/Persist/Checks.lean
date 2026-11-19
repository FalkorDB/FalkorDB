import GraphPersist.Persist.MultiLoad3
/-! # #3643: a corrupted payload is refused, a saved one still loads

`GRAPH.RESTORE` and the RDB load read untrusted words. #3643 holds them to the header
(decoder/validate.rs): `check_header` (:26), `check_directory` (:50),
`label_matrix_count` (:72), `tensor_count` (:90), `check_assembled` (:115), the stores'
`with_id_bound` and `MAX_ATTRIBUTES` (attribute_store.rs), and a fallible
`Graph::restore` (graph.rs:903, `id_bound` :1037). The model carries them as `checkHeader`,
`checkDir`, `decPayload`, `decEnt` and `AsmOK`.

* Every payload the encoder writes still passes every check: `load_encode`, `key_rt`
  and `multi_load` are re-proved on the checked decoder; `encoder_passes_checks` and
  `multi_key_passes_checks` state the checks on their own.
* Each check refuses what it is meant to: `loadFromReader_refuses_header` /
  `rdbKey_refuses_header` (ids above `2^34`), `loadFromReader_refuses_dir` /
  `rdbKey_refuses_dir` (a section larger than the header says),
  `decPayload_refuses_labels` (more label matrices than labels), `decEnt_refuses_id` /
  `decEnt_refuses_attrs`, `kDir_refuses` and the `asm_refuses_*` witnesses
  (`check_assembled` / `Graph::restore`), `run_keys_refuses` (multi-key).
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

theorem bind_ok {α β : Type} {d : Dec α} {f : α → Dec β} {s r : Stream} {a : α} (h : d s = .ok a r) :
    d.bind f s = f a r := by simp [Dec.bind, h]

/-! ## The encoder's payloads pass -/

/-- The single-key payload of a well-formed graph passes `check_header`,
`check_directory` and (`check_assembled` + `Graph::restore`). -/
theorem encoder_passes_checks (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm) :
    checkHeader (hdrOf g g.name 1) = true ∧
    checkDir (hdrOf g g.name 1) ((buildPayloads g).map fun e => (e.st, e.count)) = true ∧
    AsmOK C (hdrOf g g.name 1) g.sch (restoreOf g dest).acc :=
  ⟨checkHeader_hdrOf C g hw _ _, checkDir_of _ _ (buildPayloads_dir g g.name 1),
   asmOK_restoreOf C g hw _ _ _⟩

/-- Every key `build_multi_key_payloads` writes passes `check_header` and
`check_directory` (a slice never exceeds its kind's total, which every key's header
carries), and the assembled graph passes `check_assembled` + `Graph::restore`. -/
theorem multi_key_passes_checks (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 0 < (buildMulti g vmax).length) (k : Nat) :
    checkHeader (hdrOf g g.name (buildMulti g vmax).length) = true ∧
    checkDir (hdrOf g g.name (buildMulti g vmax).length)
      ((keyAt (buildMulti g vmax) k).map fun e => (e.st, e.count)) = true ∧
    AsmOK C (hdrOf g g.name (buildMulti g vmax).length) g.sch (restoreOf g g.name).acc :=
  ⟨checkHeader_hdrOf C g hw _ _,
   checkDir_of _ _ (fun e he => EntryOK.dir g _ _ e ((layout_ok C g hw vmax ht hK k).1 e he)),
   asmOK_restoreOf C g hw _ _ _⟩

/-! ## `check_header` -/

theorem checkHeader_false_iff (h : Hdr Nm) :
    checkHeader h = false ↔ MAX_ID_BOUND < h.nodeBound ∨ MAX_ID_BOUND < h.edgeBound := by
  simp only [checkHeader, Bool.and_eq_false_iff, decide_eq_false_iff_not, Nat.not_le]

/-- **`check_header` refuses** a header whose node or edge id space exceeds `2^34`:
`load_graph_from_reader` (`GRAPH.RESTORE`) fails right after the header. -/
theorem loadFromReader_refuses_header (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (h : Hdr Nm) (hg : C.H.good h)
    (hb : MAX_ID_BOUND < h.nodeBound ∨ MAX_ID_BOUND < h.edgeBound) (rest : Stream) :
    loadFromReader C dest (C.H.enc h ++ rest) = .err := by
  rw [loadFromReader, bind_ok ((C.H.rt h hg).1 rest), kHdr, if_pos ((checkHeader_false_iff h).2 hb)]; rfl

/-- … and so does `rdb_load_graph`. -/
theorem rdbKey_refuses_header (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (hg : C.H.good h)
    (hb : MAX_ID_BOUND < h.nodeBound ∨ MAX_ID_BOUND < h.edgeBound) (rest : Stream) :
    rdbKey C (C.H.enc h ++ rest) = (.err : Out (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn))) := by
  rw [rdbKey, bind_ok ((C.H.rt h hg).1 rest), kKeyHdr, if_pos ((checkHeader_false_iff h).2 hb)]; rfl

/-! ## `check_directory` -/

theorem checkDir_false (h : Hdr Nm) (es : List Entry) (e : Entry) (he : e ∈ es) (m : Nat)
    (hm : dirCap h e.st = some m) (hgt : m < e.count) :
    checkDir h (es.map fun e => (e.st, e.count)) = false := by
  rw [Bool.eq_false_iff]
  intro ht
  simp only [checkDir, List.all_map, List.all_eq_true, Function.comp_apply] at ht
  have := ht e he
  rw [hm] at this
  simp at this; omega

/-- **`check_directory` refuses** a directory listing more nodes, edges or deleted ids
of a kind than the header declares (`GRAPH.RESTORE`). -/
theorem loadFromReader_refuses_dir (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (h : Hdr Nm) (hg : C.H.good h)
    (hc : checkHeader h = true) (hk : h.keyCount = 1) (s : Sch Nm Ix Cn) (hs : C.S.good s)
    (es : List Entry) (hn : es.length < 2 ^ 64) (hcnt : ∀ e ∈ es, e.count < 2 ^ 64)
    (hacc : ∀ e ∈ es, e.st.accepted = true) (e : Entry) (he : e ∈ es) (m : Nat)
    (hm : dirCap h e.st = some m) (hgt : m < e.count) (rest : Stream) :
    loadFromReader C dest (C.H.enc h ++ (C.S.enc s ++ (encDir es ++ rest))) = .err := by
  rw [loadFromReader, bind_ok ((C.H.rt h hg).1 _), kHdr, if_neg (by simp [hc]), if_neg (by simp [hk]),
    bind_ok ((C.S.rt s hs).1 _), kSch, bind_ok ((decDir_rt es hn hcnt hacc).1 rest),
    checkDir_false h es e he m hm hgt]
  rfl

/-- … and so does `rdb_load_graph` (any key of a multi-key graph). -/
theorem rdbKey_refuses_dir (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (hg : C.H.good h)
    (hc : checkHeader h = true) (s : Sch Nm Ix Cn) (hs : C.S.good s)
    (es : List Entry) (hn : es.length < 2 ^ 64) (hcnt : ∀ e ∈ es, e.count < 2 ^ 64)
    (hacc : ∀ e ∈ es, e.st.accepted = true) (e : Entry) (he : e ∈ es) (m : Nat)
    (hm : dirCap h e.st = some m) (hgt : m < e.count) (rest : Stream) :
    rdbKey C (C.H.enc h ++ (C.S.enc s ++ (encDir es ++ rest))) =
      (.err : Out (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn))) := by
  rw [rdbKey, bind_ok ((C.H.rt h hg).1 _), kKeyHdr, if_neg (by simp [hc]),
    bind_ok ((C.S.rt s hs).1 _), kKeySch, bind_ok ((decDir_rt es hn hcnt hacc).1 rest)]
  simp only [checkDir_false h es e he m hm hgt]
  rfl

/-! ## `label_matrix_count` and the stores' id bound -/

/-- **`label_matrix_count` refuses** a label-matrix section claiming more matrices than
the header has labels (before #3643 the count sized a `Vec` unchecked). -/
theorem decPayload_refuses_labels (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (c n : Nat) (hn : n < 2 ^ 64)
    (hgt : h.labelCount < n) (rest : Stream) :
    decPayload C h (.labelsM, c) (encU n ++ rest) = .err := by
  show (readU.bind fun n => if n ≤ h.labelCount then (decIdx C.Mx.dec n).bind fun ms => Dec.pure (PR.lm ms)
    else Dec.fail) (encU n ++ rest) = .err
  rw [bind_ok ((readU_rt n hn).1 rest), if_neg (by omega)]; rfl

/-- **`with_id_bound` refuses** an entity id outside the header's id space. -/
theorem decEnt_refuses_id (bound i : Nat) (hi : i < 2 ^ 64) (hb : bound ≤ i) (rest : Stream) :
    decEnt bound (encU i ++ rest) = .err := by
  rw [decEnt, bind_ok ((readU_rt i hi).1 rest), if_neg (by omega)]; rfl

/-- **`MAX_ATTRIBUTES` refuses** an entity claiming more than 65535 attributes. -/
theorem decEnt_refuses_attrs (bound i n : Nat) (hi : i < 2 ^ 64) (hb : i < bound) (hn : n < 2 ^ 64)
    (hgt : MAX_ATTRIBUTES < n) (rest : Stream) :
    decEnt bound (encU i ++ (encU n ++ rest)) = .err := by
  rw [decEnt, bind_ok ((readU_rt i hi).1 _), if_pos hb, bind_ok ((readU_rt n hn).1 rest), if_neg (by omega)]
  rfl

/-! ## `check_assembled` and `Graph::restore` -/

/-- **`check_assembled` / `Graph::restore` refuse** (single key): whatever the payloads
decoded to, if the assembly fails `AsmOK` the load is an `Err`, not a graph. -/
theorem kDir_refuses (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (dest : Nm) (s : Sch Nm Ix Cn)
    (dir : List (St × Nat)) (t rest : Stream) (prs : List (PR M Tn))
    (hd : seqD (decPayload C h) dir (t ++ rest) = .ok prs rest)
    (hA : ¬ AsmOK C h s (finish C dest h.nodeCount h.edgeCount s prs).acc) :
    kDir C h dest s dir (t ++ rest) = .err := by
  rw [kDir, bind_ok hd, if_neg hA]; rfl

/-- A deleted id at or beyond `count + deleted_count` (`id_bound`). -/
theorem asm_refuses_deleted_node (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (j : Nat) (hj : a.delN j = true) (hb : h.nodeBound ≤ j) : ¬ AsmOK C h s a := fun H => by
  have := H.dn.1 j hj; simp only [Hdr.nodeBound] at hb; omega

theorem asm_refuses_deleted_edge (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (j : Nat) (hj : a.delE j = true) (hb : h.edgeBound ≤ j) : ¬ AsmOK C h s a := fun H => by
  have := H.de.1 j hj; simp only [Hdr.edgeBound] at hb; omega

/-- A deleted set of the wrong size (`check_deleted`). -/
theorem asm_refuses_deleted_count (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (hc : ((List.range (h.nodeCount + h.delNodeCount)).filter a.delN).length ≠ h.delNodeCount) :
    ¬ AsmOK C h s a := fun H => hc H.dn.2

/-- Label matrices or tensors not matching the schema's labels / types. -/
theorem asm_refuses_labels (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (hc : a.lms.length ≠ s.labels.length) : ¬ AsmOK C h s a := fun H => hc H.lm

theorem asm_refuses_types (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (hc : a.tns.length ≠ s.types.length) : ¬ AsmOK C h s a := fun H => hc H.tn

/-- Tensors carrying a different number of edges than the header declares. -/
theorem asm_refuses_edge_count (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (hc : (a.tns.flatMap C.edges).length ≠ h.edgeCount) : ¬ AsmOK C h s a := fun H => hc H.ec

/-- An edge whose id or an endpoint lies outside the id spaces. -/
theorem asm_refuses_edge_bound (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (a : Acc M Tn)
    (x : Nat × Nat × Nat) (hx : x ∈ a.tns.flatMap C.edges)
    (hb : h.edgeBound ≤ x.2.2 ∨ h.nodeBound ≤ x.1 ∨ h.nodeBound ≤ x.2.1) : ¬ AsmOK C h s a := fun H => by
  have := H.eb x hx; omega

end GraphPersist.Persist
