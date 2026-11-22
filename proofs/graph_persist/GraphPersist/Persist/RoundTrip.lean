import GraphPersist.Persist.Payload
/-! # `load ∘ encode = id` and truncation -/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

/-! ### The id-space count -/

theorem length_filter_split (p : Nat → Bool) (l : List Nat) :
    (l.filter p).length + (l.filter fun x => !p x).length = l.length := by
  induction l with
  | nil => rfl
  | cons x xs ih => by_cases h : p x <;> simp [h] <;> omega

/-- `encode_with_range` walks exactly `count` live ids: the deleted ids are distinct and
all below `count + |deleted|`. -/
theorem live_length (n : Nat) (del : List Nat) (hnd : del.Nodup) (hlt : ∀ d ∈ del, d < n + del.length) :
    (live n del).length = n := by
  have hsplit := length_filter_split (fun i => decide (i ∉ del)) (List.range (n + del.length))
  have hperm : ((List.range (n + del.length)).filter fun x => !decide (x ∉ del)).Perm del := by
    rw [List.perm_ext_iff_of_nodup ((List.nodup_range).filter _) hnd]
    intro a
    simp only [List.mem_filter, List.mem_range, decide_not, Bool.not_not, decide_eq_true_eq]
    exact ⟨fun h => h.2, fun h => ⟨hlt a h, h⟩⟩
  rw [hperm.length_eq] at hsplit
  simp only [live, List.length_range] at hsplit ⊢
  omega

theorem live_lt (n : Nat) (del : List Nat) : ∀ i ∈ live n del, i < n + del.length := by
  intro i hi; simp [live] at hi; exact hi.1

/-! ### Each payload reads back -/

/-- What each directory entry of `build_payloads` decodes to. -/
def prOf (g : G M Tn Nm Ix Cn) (e : Entry) : PR M Tn :=
  match e.st with
  | .nodes => .nodes ((live g.nodeCount g.delNodes).map fun i =>
      (i, ((g.nodes i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2)))
  | .delNodes => .delN g.delNodes
  | .edges => .edges ((live g.edgeCount g.delEdges).map fun i =>
      (i, ((g.edges i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2)))
  | .delEdges => .delE g.delEdges
  | .labelsM => .lm g.labelMats
  | .relM => .rt g.tensors
  | .adj => .adj g.adj
  | .lblsM => .lbls g.lbls
  | _ => .skip

theorem kind_rt (limit n : Nat) (del : List Nat) (S : Store) (hl : limit ≤ 65536)
    (hk : KindWF limit n del S) (hn : 0 < n) :
    RT (rep decEnt n) (encRange S (live n del) n 0)
      ((live n del).map fun i => (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2))) := by
  obtain ⟨hb, hnd, hlt, hS⟩ := hk
  have hlen := live_length n del hnd hlt
  have hrep := RT_rep decEnt (fun e => encEnt S e.1)
    (fun e => e.1 < 2 ^ 64 ∧ e.2 = ((S e.1).getD []).map (fun kv => ((some kv.1 : Option Nat), kv.2)) ∧
      ∀ sp, S e.1 = some sp → GoodSpan limit sp)
    (by
      rintro ⟨i, raw⟩ ⟨hi, hr, hg⟩
      simp only at hi hr hg ⊢
      rw [hr]; exact decEnt_rt S i hi limit hl hg)
    ((live n del).map fun i => (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2)))
    (by
      intro e he
      obtain ⟨i, hi, rfl⟩ := List.mem_map.1 he
      exact ⟨by have := live_lt n del i hi; omega, rfl, fun sp h => (hS i sp h).2⟩)
  rw [List.length_map, hlen] at hrep
  have htake : (live n del).take (max n 1) = live n del := List.take_of_length_le (by omega)
  simpa [encRange, htake, List.flatMap_map] using hrep

theorem payload_rt (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) :
    ∀ e ∈ buildPayloads g, RT (decPayload C g.tensors.length (e.st, e.count)) (encPayload C g e) (prOf g e) := by
  intro e he
  simp only [buildPayloads, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
  rcases he with ((((((he | he) | he) | he) | he) | he) | he | he)
  all_goals (try (split at he <;> simp at he))
  all_goals (try subst he)
  · -- nodes
    have := RT_bind (kind_rt _ _ _ _ hw.lim hw.nk (by assumption)) (f := fun es => Dec.pure (PR.nodes (M := M) (Tn := Tn) es)) (RT_pure _)
    simpa [decPayload, encPayload, prOf] using this
  · -- deleted nodes
    have hb : ∀ i ∈ g.delNodes, i < 2 ^ 64 := fun i hi => by have := hw.nk.2.2.1 i hi; have := hw.nk.1; omega
    have := RT_bind (decRoar_rt g.delNodes hb) (f := fun l => Dec.pure (PR.delN (M := M) (Tn := Tn) l)) (RT_pure _)
    simpa [decPayload, encPayload, prOf, encRoar] using this
  · -- edges
    have := RT_bind (kind_rt _ _ _ _ hw.lim hw.ek (by assumption)) (f := fun es => Dec.pure (PR.edges (M := M) (Tn := Tn) es)) (RT_pure _)
    simpa [decPayload, encPayload, prOf] using this
  · -- deleted edges
    have hb : ∀ i ∈ g.delEdges, i < 2 ^ 64 := fun i hi => by have := hw.ek.2.2.1 i hi; have := hw.ek.1; omega
    have := RT_bind (decRoar_rt g.delEdges hb) (f := fun l => Dec.pure (PR.delE (M := M) (Tn := Tn) l)) (RT_pure _)
    simpa [decPayload, encPayload, prOf, encRoar] using this
  · -- label matrices
    have := RT_bind (readU_rt _ hw.lm.1)
      (f := fun n => (decIdx C.Mx.dec n).bind fun ms => Dec.pure (PR.lm (Tn := Tn) ms))
      (RT_bind (decIdx_rt C.Mx g.labelMats hw.lm.1 hw.lm.2) (f := fun ms => Dec.pure (PR.lm (Tn := Tn) ms)) (RT_pure _))
    simpa [decPayload, encPayload, prOf] using this
  · -- relation tensors
    have := RT_bind (decIdx_rt C.Tx g.tensors hw.tn.1 hw.tn.2) (f := fun ts => Dec.pure (PR.rt (M := M) ts)) (RT_pure _)
    simpa [decPayload, encPayload, prOf] using this
  · have := RT_bind (C.Mx.rt _ hw.mx.1) (f := fun m => Dec.pure (PR.adj (Tn := Tn) m)) (RT_pure _)
    simpa [decPayload, encPayload, prOf] using this
  · have := RT_bind (C.Mx.rt _ hw.mx.2) (f := fun m => Dec.pure (PR.lbls (Tn := Tn) m)) (RT_pure _)
    simpa [decPayload, encPayload, prOf] using this

/-! ### Folding the payloads gives the graph back -/

theorem insAll_spec (l : List Nat) : ∀ (s : Nat → Bool), insAll s l = fun j => s j || decide (j ∈ l) := by
  induction l with
  | nil => intro s; funext j; simp [insAll]
  | cons i is ih =>
    intro s
    simp only [insAll, List.foldl_cons] at ih ⊢
    rw [ih]
    funext j
    by_cases h : j = i
    · simp [h]
    · have : (j == i) = false := by simp [h]
      simp [this, h]

theorem kind_store (limit n : Nat) (del : List Nat) (S : Store) (hl : limit ≤ 65536)
    (hk : KindWF limit n del S) :
    ((live n del).map fun i => (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2))).foldl
      (applyEnt limit) (fun _ => none) = S := by
  rw [fold_ents limit hl S _ (fun i _ sp h => (hk.2.2.2 i sp h).2)]
  funext j
  by_cases h : j ∈ live n del ∧ S j ≠ none
  · simp [h]
  · rw [ite_eq_right h]
    cases hs : S j with
    | none => rfl
    | some sp => exact absurd ⟨(hk.2.2.2 j sp hs).1, by simp [hs]⟩ h

theorem kind_empty (limit : Nat) (del : List Nat) (S : Store) (hk : KindWF limit 0 del S) :
    S = fun _ => none := by
  have hlen := live_length 0 del hk.2.1 hk.2.2.1
  funext j
  cases hs : S j with
  | none => rfl
  | some sp =>
    have := (hk.2.2.2 j sp hs).1
    rw [List.length_eq_zero_iff.1 hlen] at this
    simp at this

theorem fold_build (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm) :
    ((buildPayloads g).map (prOf g)).foldl (applyPR g.sch.attrs.length) (acc0 C) = (restoreOf g dest).acc := by
  have hN := kind_store _ _ _ _ hw.lim hw.nk
  have hE := kind_store _ _ _ _ hw.lim hw.ek
  have hN0 : g.nodeCount = 0 → g.nodes = fun _ => none := fun h => kind_empty _ _ _ (h ▸ hw.nk)
  have hE0 : g.edgeCount = 0 → g.edges = fun _ => none := fun h => kind_empty _ _ _ (h ▸ hw.ek)
  simp only [buildPayloads, List.map_append, List.foldl_append]
  by_cases h1 : g.nodeCount > 0 <;> by_cases h2 : g.delNodes.length > 0 <;>
  by_cases h3 : g.edgeCount > 0 <;> by_cases h4 : g.delEdges.length > 0 <;>
  by_cases h5 : g.labelMats.length > 0 <;> by_cases h6 : g.tensors.length > 0 <;>
  simp only [h1, h2, h3, h4, h5, h6, ite_true, ite_false, List.map_cons, List.map_nil,
    List.foldl_cons, List.foldl_nil, prOf, applyPR, acc0, restoreOf, insAll_spec, hN, hE,
    Acc.mk.injEq, Bool.false_or, List.nil_append, true_and, and_true] <;>
  simp_all [List.length_eq_zero_iff]

theorem seqD_map {E E' β : Type} (f : E' → Dec β) (h : E → E') :
    ∀ l : List E, seqD f (l.map h) = seqD (fun e => f (h e)) l
  | [] => rfl
  | e :: es => by simp [seqD, seqD_map f h es]

theorem buildPayloads_bounds (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) :
    (buildPayloads g).length < 2 ^ 64 ∧ ∀ e ∈ buildPayloads g, e.count < 2 ^ 64 := by
  have h1 := hw.nk.1; have h2 := hw.ek.1; have h3 := hw.lm.1; have h4 := hw.tn.1
  refine ⟨?_, ?_⟩
  · simp only [buildPayloads, List.length_append, List.length_cons, List.length_nil]
    split <;> split <;> split <;> split <;> split <;> split <;> simp
  · intro e he
    simp only [buildPayloads, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
    rcases he with ((((((he | he) | he) | he) | he) | he) | he | he) <;>
      (try (split at he <;> simp at he)) <;> subst he <;> simp <;> omega

/-- `build_payloads` lists only states `decode_payload_directory` accepts. -/
theorem buildPayloads_accepted (g : G M Tn Nm Ix Cn) : ∀ e ∈ buildPayloads g, e.st.accepted = true := by
  intro e he
  simp only [buildPayloads, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
  rcases he with ((((((he | he) | he) | he) | he) | he) | he | he) <;>
    (try (split at he <;> simp at he)) <;> subst he <;> rfl

/-- **`load_graph_from_reader ∘ encode_graph = id` on the graph state**, with truncation:
the single-key payload of a well-formed graph decodes to exactly the `Graph::restore`
arguments of that graph (under any destination name), and every word-boundary truncation
of it makes the decoder ask for one more word. -/
theorem load_encode (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm) :
    RT (loadFromReader C dest) (encodeGraph C g g.name (buildPayloads g) 1) (restoreOf g dest) := by
  obtain ⟨hPn, hPc⟩ := buildPayloads_bounds C g hw
  have hseq := RT_seq (fun e : Entry => decPayload C g.tensors.length (e.st, e.count))
    (encPayload C g) (prOf g) (buildPayloads g) (payload_rt C g hw)
  rw [← seqD_map (decPayload C g.tensors.length) (fun e : Entry => (e.st, e.count))] at hseq
  have hfin := RT_bind hseq (f := fun prs => Dec.pure (finish C dest g.nodeCount g.edgeCount g.sch prs))
    (RT_pure _)
  have hf : finish C dest g.nodeCount g.edgeCount g.sch ((buildPayloads g).map (prOf g)) = restoreOf g dest := by
    simp only [finish, fold_build C g hw dest, restoreOf]
  rw [hf] at hfin
  have hdir : RT (kSch C g.tensors.length dest g.nodeCount g.edgeCount g.sch)
      (encDir (buildPayloads g) ++ ((buildPayloads g).flatMap (encPayload C g) ++ [])) (restoreOf g dest) :=
    RT_bind (decDir_rt (buildPayloads g) hPn hPc (buildPayloads_accepted g)) hfin
  have hsch : RT (C.S.dec.bind (kSch C g.tensors.length dest g.nodeCount g.edgeCount))
      (C.S.enc g.sch ++ (encDir (buildPayloads g) ++ ((buildPayloads g).flatMap (encPayload C g) ++ [])))
      (restoreOf g dest) :=
    RT_bind (C.S.rt _ hw.sch) hdir
  have hk : kHdr C dest (hdrOf g g.name 1) = C.S.dec.bind (kSch C g.tensors.length dest g.nodeCount g.edgeCount) := by
    simp [kHdr, hdrOf]
  have hhdr : RT (loadFromReader C dest)
      (C.H.enc (hdrOf g g.name 1) ++ (C.S.enc g.sch ++ (encDir (buildPayloads g) ++
        ((buildPayloads g).flatMap (encPayload C g) ++ [])))) (restoreOf g dest) :=
    RT_bind (C.H.rt _ hw.hdr) (hk ▸ hsch)
  have e : encodeGraph C g g.name (buildPayloads g) 1 = C.H.enc (hdrOf g g.name 1) ++
      (C.S.enc g.sch ++ (encDir (buildPayloads g) ++ ((buildPayloads g).flatMap (encPayload C g) ++ []))) := by
    simp only [encodeGraph, List.append_assoc, List.append_nil]
  rw [e]
  exact hhdr

/-- **#2537, generalised.** Any payload `encode_graph` produces, cut at any word
boundary short of its end, makes `load_graph_from_reader` ask for another word. Behind
`vec_load_graph` the reader is `BufferedReader::from_blob` (`rdb = NULL`), whose
`ensure_available → load_chunk` then calls `RedisModule_LoadStringBuffer(NULL)`
(`buffered_io.rs`; redis_layer `from_slice_past_end`): a crash, not an `Err`.
`vec_blob_eof` is the same for the whole (prefixed) blob. -/
theorem truncated_payload_eof (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm)
    (p : Stream) (hp : StrictPre p (encodeGraph C g g.name (buildPayloads g) 1)) :
    loadFromReader C dest p = .eof :=
  (load_encode C g hw dest).2 p hp

/-- The empty payload (`GRAPH.RESTORE k ""`) is the shortest such cut. -/
theorem empty_payload_eof (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm)
    (hne : encodeGraph C g g.name (buildPayloads g) 1 ≠ []) : loadFromReader C dest [] = .eof :=
  truncated_payload_eof C g hw dest [] ⟨_, hne, rfl⟩

/-! ## Versioned `GRAPH.RESTORE` blobs (#3610)

`vec_save_graph` (encoder/mod.rs) writes `write_unsigned(ENCODING_VERSION)` before
`encode_graph`; `vec_load_graph` (decoder/mod.rs) opens the blob with
`BufferedReader::from_blob` (buffered_io.rs), which consumes that prefix as the reader's
`encver`, and then runs `load_graph_from_reader`. -/

/-- `graphblas::serialization::ENCODING_VERSION` (serialization.rs). -/
def ENCODING_VERSION : Nat := 21

/-- `Reader::encver`'s default (serialization.rs): every reader but the RDB/blob one
(`PipeReader` for `GRAPH.COPY`, in-process copies) reports `ENCODING_VERSION`. -/
def readerEncverDefault : Nat := ENCODING_VERSION

theorem readerEncverDefault_eq : readerEncverDefault = 21 := rfl

/-- `vec_save_graph(graph)` as words. -/
def vecSave (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) : Stream :=
  encU ENCODING_VERSION ++ encodeGraph C g g.name (buildPayloads g) 1

/-- `BufferedReader::from_blob` on words: a leading unsigned is the version (refused above 21
and below 19); anything else is an unprefixed blob from an older build, read from its first
word at 19. -/
def fromBlob : Dec Nat
  | .u v :: r =>
    if ENCODING_VERSION < v.toNat then .err else if v.toNat < 19 then .err else .ok v.toNat r
  | s => .ok 19 s

/-- `vec_load_graph(data, dest)`. -/
def vecLoad (C : Codecs M Tn Nm Ix Cn) (dest : Nm) : Dec (Restore M Tn Nm Ix Cn) :=
  fromBlob.bind fun _ => loadFromReader C dest

theorem encodeGraph_ne_nil (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (es : List Entry) (K : Nat) :
    encodeGraph C g g.name es K ≠ [] := by
  simp [encodeGraph, encDir, encU]

theorem fromBlob_prefix (r : Stream) : fromBlob (encU ENCODING_VERSION ++ r) = .ok 21 r := by
  simp [fromBlob, encU, ENCODING_VERSION]

/-- **`vec_load_graph ∘ vec_save_graph = id`, with truncation.** The prefixed blob reads back
to `Graph::restore` of the graph (`from_blob` takes version 21 and stops right after the prefix),
and every word-boundary cut of it — including the empty blob and the bare prefix — asks for one
more word. -/
theorem vec_blob_rt (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm) :
    RT (vecLoad C dest) (vecSave C g) (restoreOf g dest) := by
  have hl := load_encode C g hw dest
  refine ⟨fun r => ?_, fun p hp => ?_⟩
  · simp only [vecSave, List.append_assoc, vecLoad, Dec.bind, fromBlob_prefix, hl.1 r]
  · obtain ⟨q, hq, hpq⟩ := hp
    cases p with
    | nil =>
      have := hl.2 [] ⟨_, encodeGraph_ne_nil C g _ 1, rfl⟩
      simp only [vecLoad, Dec.bind, fromBlob]; exact this
    | cons t p' =>
      simp only [vecSave, encU, List.cons_append, List.nil_append, List.cons.injEq] at hpq
      obtain ⟨rfl, hpq⟩ := hpq
      have := hl.2 p' ⟨q, hq, hpq⟩
      simp only [vecLoad, Dec.bind, fromBlob, ENCODING_VERSION]
      simpa using this

/-- Truncated blobs (#2537) still reach `load_chunk(NULL)` rather than an `Err`. -/
theorem vec_blob_eof (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (dest : Nm)
    (p : Stream) (hp : StrictPre p (vecSave C g)) : vecLoad C dest p = .eof :=
  (vec_blob_rt C g hw dest).2 p hp

/-- A prefix above 21 (a newer build's blob) is refused before any graph word is read. -/
theorem vec_blob_newer (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (v : UInt64)
    (hv : ENCODING_VERSION < v.toNat) (r : Stream) : vecLoad C dest (.u v :: r) = .err := by
  simp [vecLoad, Dec.bind, fromBlob, hv]

/-- A prefix below 19 (`UNVERSIONED_BLOB_ENCVER`) is refused before any graph word is read. -/
theorem vec_blob_older (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (v : UInt64)
    (hv : v.toNat < 19) (r : Stream) : vecLoad C dest (.u v :: r) = .err := by
  have : ¬ ENCODING_VERSION < v.toNat := by unfold ENCODING_VERSION; omega
  simp [vecLoad, Dec.bind, fromBlob, hv, this]

/-- An older build's blob (no prefix: its first word is the header's graph-name buffer) is
handed to `load_graph_from_reader` whole, as before #3610. -/
theorem vec_blob_old (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (b : List UInt8) (r : Stream) :
    vecLoad C dest (.bytes b :: r) = loadFromReader C dest (.bytes b :: r) := by
  simp [vecLoad, Dec.bind, fromBlob]

end GraphPersist.Persist
