import GraphPersist.Persist.MultiKey
/-! # One RDB key of a multi-key graph reads back its slices

`rdb_load_graph` (decoder/mod.rs:44-66) reads header, schema and directory, then — with
`key_count > 1` — `decode_payloads_into_pending` (:260-307) reads the payloads into the
pending graph. `rdbKey` is that read; `key_rt` says every key `build_multi_key_payloads`
lays out (entity slices `(offset, count)` and, on key 0, the matrices) reads back as the
slices it encodes.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

def rawOf (S : Store) (i : Nat) : Nat × List (Option Nat × Val) :=
  (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2))

/-- How many entities of a kind the graph has. -/
def kindTotal (g : G M Tn Nm Ix Cn) : St → Nat
  | .nodes => g.nodeCount
  | .delNodes => g.delNodes.length
  | .edges => g.edgeCount
  | .delEdges => g.delEdges.length
  | _ => 0

def IsEntity (st : St) : Prop := st = .nodes ∨ st = .delNodes ∨ st = .edges ∨ st = .delEdges

/-- What one directory entry of a key decodes to (slice-aware `prOf`). -/
def prOfE (g : G M Tn Nm Ix Cn) (e : Entry) : PR M Tn :=
  match e.st with
  | .nodes => .nodes ((((live g.nodeCount g.delNodes).drop e.offset).take e.count).map (rawOf g.nodes))
  | .delNodes => .delN ((g.delNodes.drop e.offset).take e.count)
  | .edges => .edges ((((live g.edgeCount g.delEdges).drop e.offset).take e.count).map (rawOf g.edges))
  | .delEdges => .delE ((g.delEdges.drop e.offset).take e.count)
  | _ => prOf g e

/-- A directory entry a multi-key layout can hold. -/
def EntryOK (g : G M Tn Nm Ix Cn) (e : Entry) : Prop :=
  (IsEntity e.st ∧ 0 < e.count ∧ e.offset + e.count ≤ kindTotal g e.st) ∨ e ∈ matrixEntries g

theorem slice_rt (limit n : Nat) (del : List Nat) (S : Store) (hl : limit ≤ 65536)
    (hk : KindWF limit n del S) (off cnt : Nat) (hc : 0 < cnt) (ho : off + cnt ≤ n) :
    RT (rep decEnt cnt) (encRange S (live n del) cnt off) ((((live n del).drop off).take cnt).map (rawOf S)) := by
  obtain ⟨hb, hnd, hlt, hS⟩ := hk
  have hlen := live_length n del hnd hlt
  have hsl : (((live n del).drop off).take cnt).length = cnt := by simp; omega
  have hrep := RT_rep decEnt (fun e => encEnt S e.1)
    (fun e => e.1 < 2 ^ 64 ∧ e = rawOf S e.1 ∧ ∀ sp, S e.1 = some sp → GoodSpan limit sp)
    (by
      rintro ⟨i, raw⟩ ⟨hi, hr, hg⟩
      simp only at hi hg ⊢
      rw [hr]; exact decEnt_rt S i hi limit hl hg)
    ((((live n del).drop off).take cnt).map (rawOf S))
    (by
      intro e he
      obtain ⟨i, hi, rfl⟩ := List.mem_map.1 he
      have hi' : i ∈ live n del := List.mem_of_mem_drop (List.mem_of_mem_take hi)
      exact ⟨by have := live_lt n del i hi'; simp [rawOf]; omega, rfl, fun sp h => (hS i sp h).2⟩)
  rw [List.length_map, hsl] at hrep
  have htake : ((live n del).drop off).take (max cnt 1) = ((live n del).drop off).take cnt := by
    rw [Nat.max_eq_left hc]
  have ht : (((((live n del).drop off).take cnt).map (rawOf S)).flatMap fun e => encEnt S e.1) =
      (((live n del).drop off).take cnt).flatMap (encEnt S) := by
    rw [List.flatMap_map]; rfl
  rw [ht] at hrep
  show RT _ ((((live n del).drop off).take (max cnt 1)).flatMap (encEnt S)) _
  rw [htake]; exact hrep

theorem roar_slice_rt (del : List Nat) (hb : ∀ i ∈ del, i < 2 ^ 64) (off cnt : Nat) (ho : off + cnt ≤ del.length) :
    RT (decRoar cnt) (encRoar del cnt off) ((del.drop off).take cnt) := by
  have hsl : ((del.drop off).take cnt).length = cnt := by simp; omega
  have := decRoar_rt ((del.drop off).take cnt)
    (fun i hi => hb i (List.mem_of_mem_drop (List.mem_of_mem_take hi)))
  rw [hsl] at this
  simpa [encRoar] using this

theorem entry_rt (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (e : Entry) (he : EntryOK g e) :
    RT (decPayload C g.tensors.length (e.st, e.count)) (encPayload C g e) (prOfE g e) := by
  rcases he with ⟨hst, hc, ho⟩ | he
  · rcases hst with h | h | h | h <;> obtain ⟨st, count, offset⟩ := e <;> simp only at h <;> subst h <;>
      simp only [kindTotal] at ho
    · have := RT_bind (slice_rt _ _ _ _ hw.lim hw.nk offset count hc ho)
        (f := fun es => Dec.pure (PR.nodes (M := M) (Tn := Tn) es)) (RT_pure _)
      simpa [decPayload, encPayload, prOfE] using this
    · have hb : ∀ i ∈ g.delNodes, i < 2 ^ 64 := fun i hi => by have := hw.nk.2.2.1 i hi; have := hw.nk.1; omega
      have := RT_bind (roar_slice_rt g.delNodes hb offset count ho)
        (f := fun l => Dec.pure (PR.delN (M := M) (Tn := Tn) l)) (RT_pure _)
      simpa [decPayload, encPayload, prOfE] using this
    · have := RT_bind (slice_rt _ _ _ _ hw.lim hw.ek offset count hc ho)
        (f := fun es => Dec.pure (PR.edges (M := M) (Tn := Tn) es)) (RT_pure _)
      simpa [decPayload, encPayload, prOfE] using this
    · have hb : ∀ i ∈ g.delEdges, i < 2 ^ 64 := fun i hi => by have := hw.ek.2.2.1 i hi; have := hw.ek.1; omega
      have := RT_bind (roar_slice_rt g.delEdges hb offset count ho)
        (f := fun l => Dec.pure (PR.delE (M := M) (Tn := Tn) l)) (RT_pure _)
      simpa [decPayload, encPayload, prOfE] using this
  · -- a matrix entry: exactly as in the single-key payload
    have hin : e ∈ buildPayloads g := by
      simp only [matrixEntries, List.mem_append] at he
      simp only [buildPayloads, List.mem_append]
      rcases he with (he | he) | he
      · exact Or.inl (Or.inl (Or.inr he))
      · exact Or.inl (Or.inr he)
      · exact Or.inr he
    have := payload_rt C g hw e hin
    have hp : prOfE g e = prOf g e := by
      simp only [matrixEntries, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
      rcases he with (he | he) | he | he
      · split at he <;> simp at he; subst he; rfl
      · split at he <;> simp at he; subst he; rfl
      · subst he; rfl
      · subst he; rfl
    rw [hp]; exact this

/-! ## The key read -/

/-- `rdb_load_graph`'s reads for one key: header, schema, directory, then the payloads
(`decode_payloads_into_pending` reads exactly the arms `decPayload` does). -/
def kKeyDir (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) (dir : List (St × Nat)) :
    Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) :=
  (seqD (decPayload C h.relCount) dir).bind fun prs => Dec.pure (h, s, prs)

def kKeySch (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) (s : Sch Nm Ix Cn) :
    Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) := decDir.bind (kKeyDir C h s)

def kKeyHdr (C : Codecs M Tn Nm Ix Cn) (h : Hdr Nm) : Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) :=
  C.S.dec.bind (kKeySch C h)

def rdbKey (C : Codecs M Tn Nm Ix Cn) : Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) := C.H.dec.bind (kKeyHdr C)

/-- **One key reads back.** A key holding `es` (each entry an in-range slice or a matrix)
under `key_count = K` decodes to its header, the schema, and each entry's slice. -/
theorem key_rt (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (K : Nat)
    (hH : C.H.good (hdrOf g g.name K)) (es : List Entry) (hes : ∀ e ∈ es, EntryOK g e)
    (hn : es.length < 2 ^ 64) (hc : ∀ e ∈ es, e.count < 2 ^ 64) :
    RT (rdbKey C) (encodeGraph C g g.name es K) (hdrOf g g.name K, g.sch, es.map (prOfE g)) := by
  have hseq := RT_seq (fun e : Entry => decPayload C g.tensors.length (e.st, e.count))
    (encPayload C g) (prOfE g) es (fun e he => entry_rt C g hw e (hes e he))
  rw [← seqD_map (decPayload C g.tensors.length) (fun e : Entry => (e.st, e.count))] at hseq
  have hfin : RT (kKeyDir C (hdrOf g g.name K) g.sch (es.map fun e => (e.st, e.count)))
      (es.flatMap (encPayload C g) ++ []) (hdrOf g g.name K, g.sch, es.map (prOfE g)) :=
    RT_bind hseq (RT_pure _)
  have hdir : RT (kKeySch C (hdrOf g g.name K) g.sch) (encDir es ++ (es.flatMap (encPayload C g) ++ []))
      (hdrOf g g.name K, g.sch, es.map (prOfE g)) := RT_bind (decDir_rt es hn hc) hfin
  have hsch : RT (kKeyHdr C (hdrOf g g.name K))
      (C.S.enc g.sch ++ (encDir es ++ (es.flatMap (encPayload C g) ++ [])))
      (hdrOf g g.name K, g.sch, es.map (prOfE g)) := RT_bind (C.S.rt _ hw.sch) hdir
  have hall := RT_bind (C.H.rt _ hH) hsch
  have e : encodeGraph C g g.name es K = C.H.enc (hdrOf g g.name K) ++
      (C.S.enc g.sch ++ (encDir es ++ (es.flatMap (encPayload C g) ++ []))) := by
    simp only [encodeGraph, List.append_assoc, List.append_nil]
  rw [e]; exact hall

end GraphPersist.Persist
