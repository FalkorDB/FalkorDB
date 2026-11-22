import GraphPersist.Persist.Store
/-! # Payload directory, matrix lists, the graph, the decoder -/
namespace GraphPersist.Persist
open GraphPersist

/-! ## Payload directory (`EncodeState`, serialization.rs:120-158) -/

inductive St where
  | init | nodes | delNodes | edges | delEdges | schema | labelsM | relM | adj | lblsM | cch | final
  deriving DecidableEq, Repr

def St.toU : St → Nat
  | .init => 0 | .nodes => 1 | .delNodes => 2 | .edges => 3 | .delEdges => 4 | .schema => 5
  | .labelsM => 6 | .relM => 7 | .adj => 8 | .lblsM => 9 | .cch => 10 | .final => 11

/-- `EncodeState::from_u64` (:150). -/
def St.ofU : Nat → Option St
  | 0 => some .init | 1 => some .nodes | 2 => some .delNodes | 3 => some .edges
  | 4 => some .delEdges | 5 => some .schema | 6 => some .labelsM | 7 => some .relM
  | 8 => some .adj | 9 => some .lblsM | 10 => some .cch | 11 => some .final | _ => none

theorem St.ofU_toU (s : St) : St.ofU s.toU = some s := by cases s <;> rfl
theorem St.toU_lt (s : St) : s.toU < 2 ^ 64 := by cases s <;> decide

/-- `from_u64` refuses every value past `Final = 11` (`_ => None`). -/
theorem St.ofU_none (v : Nat) (h : 12 ≤ v) : St.ofU v = none := by
  match v, h with
  | v + 12, _ => rfl

/-- `EncodeState::from_u64` is exactly the inverse of the `#[repr(u64)]` discriminant:
`from_u64 v = Some s` iff `s as u64 == v` (so every tag the encoder writes decodes to its state,
and nothing else decodes). -/
theorem St.ofU_iff (v : Nat) (s : St) : St.ofU v = some s ↔ s.toU = v := by
  constructor
  · intro h
    by_cases hv : v < 12
    · have : v = 0 ∨ v = 1 ∨ v = 2 ∨ v = 3 ∨ v = 4 ∨ v = 5 ∨ v = 6 ∨ v = 7 ∨ v = 8 ∨ v = 9 ∨ v = 10 ∨ v = 11 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        cases s <;> simp_all [St.ofU, St.toU]
    · rw [St.ofU_none v (by omega)] at h; cases h
  · rintro rfl; exact St.ofU_toU s

/-- The states `decode_payload_directory` (decoder/mod.rs:48) accepts: all but
`CchIndices` (no CCH index in this module), `Init` and `Final` (never saved). -/
def St.accepted : St → Bool
  | .init | .cch | .final => false
  | _ => true

/-- `PayloadEntry` (:160). -/
structure Entry where
  st : St
  count : Nat
  offset : Nat
  deriving DecidableEq, Repr

/-- `encode_graph` lines 79-83: count, then `(state, count)` per payload (offsets stay home). -/
def encDir (es : List Entry) : Stream :=
  encU es.length ++ es.flatMap fun e => encU e.st.toU ++ encU e.count

/-- The directory loop of `decode_payload_directory` (decoder/mod.rs:48-72), which
`rdb_load_graph` (:128) and `load_graph_from_reader` (:509) call: both words are read
before the state is checked, and an unknown, `CchIndices`, `Init` or `Final` state fails. -/
def decDirEntry : Dec (St × Nat) :=
  readU.bind fun s => readU.bind fun c =>
    match St.ofU s with
    | some st => if st.accepted then Dec.pure (st, c) else Dec.fail
    | none => Dec.fail

def decDir : Dec (List (St × Nat)) := readU.bind fun n => rep decDirEntry n

/-- Every directory the encoder writes reads back (`build_payloads` lists only accepted
states; `buildPayloads_accepted`, `EntryOK.accepted`). -/
theorem decDir_rt (es : List Entry) (hn : es.length < 2 ^ 64) (hc : ∀ e ∈ es, e.count < 2 ^ 64)
    (hs : ∀ e ∈ es, e.st.accepted = true) :
    RT decDir (encDir es) (es.map fun e => (e.st, e.count)) := by
  have hent : ∀ e : Entry, e.count < 2 ^ 64 → e.st.accepted = true →
      RT decDirEntry (encU e.st.toU ++ encU e.count) (e.st, e.count) := by
    intro e he ha
    have := RT_bind (readU_rt _ (St.toU_lt e.st))
      (f := fun s => readU.bind fun c => match St.ofU s with
        | some st => if st.accepted then Dec.pure (st, c) else Dec.fail | none => Dec.fail)
      (RT_bind (readU_rt _ he) (f := fun c => match St.ofU e.st.toU with
        | some st => if st.accepted then Dec.pure (st, c) else Dec.fail | none => Dec.fail)
        (by rw [St.ofU_toU]; simp only [ha, ite_true]; exact RT_pure _))
    simpa [decDirEntry] using this
  have hrep := RT_rep decDirEntry (fun p : St × Nat => encU p.1.toU ++ encU p.2)
    (fun p => p.2 < 2 ^ 64 ∧ p.1.accepted = true) (fun p hp => by simpa using hent ⟨p.1, p.2, 0⟩ hp.1 hp.2)
    (es.map fun e => (e.st, e.count))
    (by intro p hp; obtain ⟨e, he, rfl⟩ := List.mem_map.1 hp; exact ⟨hc e he, hs e he⟩)
  have := RT_bind (readU_rt es.length hn) (f := fun n => rep decDirEntry n) (by simpa using hrep)
  simpa [decDir, encDir, List.flatMap_map] using this

/-- #3793: a directory entry naming `CchIndices`, `Init` or `Final` is refused once its two
words are read, before any payload; before, 10 read as `Final` and its payload was skipped
unread, misaligning the stream. -/
theorem decDirEntry_refuses (st : St) (hs : st.accepted = false) (c : Nat) (hc : c < 2 ^ 64)
    (r : Stream) : decDirEntry (encU st.toU ++ encU c ++ r) = .err := by
  have h1 := (readU_rt _ (St.toU_lt st)).1 (encU c ++ r)
  have h2 := (readU_rt _ hc).1 r
  simp only [decDirEntry, Dec.bind, List.append_assoc, h1, h2, St.ofU_toU, hs]
  rfl

/-- …in particular a v20 key listing a CCH index payload (tag 10) fails at its directory. -/
theorem decDir_refuses_cch (c : Nat) (hc : c < 2 ^ 64) (r : Stream) :
    decDir (encU 1 ++ (encU St.cch.toU ++ encU c ++ r)) = .err := by
  have h1 := (readU_rt 1 (by decide)).1 (encU St.cch.toU ++ encU c ++ r)
  have h2 := decDirEntry_refuses .cch rfl c hc r
  simp only [decDir, Dec.bind, h1, rep, h2]

/-! ## Indexed matrix lists (`LabelsMatrices` / `RelationMatrices` arms of `encode_payload`) -/

/-- `for (i, x) in xs.iter().enumerate() { write_unsigned(i); x.encode(w) }`. -/
def encIdx {β : Type} (enc : β → Stream) (xs : List β) : Stream :=
  ((List.range xs.length).zip xs).flatMap fun (i, x) => encU i ++ enc x

/-- `for _ in 0..n { let _id = read_unsigned()?; push(decode()?) }` — the id is discarded. -/
def decIdx {β : Type} (d : Dec β) (n : Nat) : Dec (List β) :=
  (rep (readU.bind fun i => d.bind fun x => Dec.pure (i, x)) n).bind fun l => Dec.pure (l.map Prod.snd)

structure Codec (α : Type) where
  enc : α → Stream
  dec : Dec α
  good : α → Prop
  rt : ∀ a, good a → RT dec (enc a) a

theorem decIdx_rt {β : Type} (C : Codec β) (xs : List β) (hn : xs.length < 2 ^ 64) (hg : ∀ x ∈ xs, C.good x) :
    RT (decIdx C.dec xs.length) (encIdx C.enc xs) xs := by
  have hpair : ∀ p : Nat × β, p.1 < 2 ^ 64 ∧ C.good p.2 →
      RT (readU.bind fun i => C.dec.bind fun x => Dec.pure (i, x)) (encU p.1 ++ C.enc p.2) p := by
    intro p ⟨h1, h2⟩
    have := RT_bind (readU_rt _ h1) (f := fun i => C.dec.bind fun x => Dec.pure (i, x))
      (RT_bind (C.rt _ h2) (f := fun x => Dec.pure (p.1, x)) (RT_pure (p.1, p.2)))
    simpa using this
  have hz : ((List.range xs.length).zip xs).length = xs.length := by simp
  have hmem : ∀ p ∈ (List.range xs.length).zip xs, p.1 < 2 ^ 64 ∧ C.good p.2 := by
    intro p hp
    have h1 := List.of_mem_zip hp
    exact ⟨by have := List.mem_range.1 h1.1; omega, hg _ h1.2⟩
  have hrep := RT_rep _ (fun p : Nat × β => encU p.1 ++ C.enc p.2) _ hpair _ hmem
  rw [hz] at hrep
  have := RT_bind hrep (f := fun l => Dec.pure (l.map Prod.snd)) (RT_pure _)
  have hsnd : ((List.range xs.length).zip xs).map Prod.snd = xs := List.map_snd_zip (by simp)
  rw [hsnd] at this
  simpa [decIdx, encIdx] using this

/-! ## The graph -/

/-- `Header` (serializers/mod.rs:123). -/
structure Hdr (Nm : Type) where
  name : Nm
  nodeCount : Nat
  edgeCount : Nat
  delNodeCount : Nat
  delEdgeCount : Nat
  labelCount : Nat
  relCount : Nat
  multiEdge : List Bool
  keyCount : Nat

/-- `Schema` (serializers/mod.rs:237). -/
structure Sch (Nm Ix Cn : Type) where
  attrs : List Nm
  labels : List Nm
  types : List Nm
  indexes : List Ix
  constraints : List Cn

/-- The sub-codecs: header and schema (proved in `redis_layer`), GraphBLAS matrices and
tensors (trusted), and `VersionedMatrix::new(0, 0)`. -/
structure Codecs (M Tn Nm Ix Cn : Type) where
  H : Codec (Hdr Nm)
  S : Codec (Sch Nm Ix Cn)
  Mx : Codec M
  Tx : Codec Tn
  m0 : M

/-- What a `Graph` contributes to its payload. -/
structure G (M Tn Nm Ix Cn : Type) where
  name : Nm
  nodeCount : Nat
  edgeCount : Nat
  delNodes : List Nat
  delEdges : List Nat
  nodes : Store
  edges : Store
  labelMats : List M
  tensors : List Tn
  adj : M
  lbls : M
  multiEdge : List Bool
  sch : Sch Nm Ix Cn


variable {M Tn Nm Ix Cn : Type}

/-- The ids `encode_with_range` walks: `0..=max_id` (`max_id = count + |deleted| - 1`,
`Graph::max_node_id`, graph.rs:1783; the loop is `encode_with_range`, attribute_store.rs:1425-1437), skipping deleted ones. -/
def live (n : Nat) (del : List Nat) : List Nat := (List.range (n + del.length)).filter fun i => decide (i ∉ del)

/-- `Header::from_graph(graph, graph_name, key_count)` (serializers/mod.rs:213). -/
def hdrOf (g : G M Tn Nm Ix Cn) (name : Nm) (k : Nat) : Hdr Nm :=
  ⟨name, g.nodeCount, g.edgeCount, g.delNodes.length, g.delEdges.length, g.labelMats.length,
   g.tensors.length, g.multiEdge, k⟩

/-- `build_payloads` (encoder/mod.rs:192-255). -/
def buildPayloads (g : G M Tn Nm Ix Cn) : List Entry :=
  (if g.nodeCount > 0 then [⟨.nodes, g.nodeCount, 0⟩] else []) ++
  (if g.delNodes.length > 0 then [⟨.delNodes, g.delNodes.length, 0⟩] else []) ++
  (if g.edgeCount > 0 then [⟨.edges, g.edgeCount, 0⟩] else []) ++
  (if g.delEdges.length > 0 then [⟨.delEdges, g.delEdges.length, 0⟩] else []) ++
  (if g.labelMats.length > 0 then [⟨.labelsM, g.labelMats.length, 0⟩] else []) ++
  (if g.tensors.length > 0 then [⟨.relM, g.tensors.length, 0⟩] else []) ++
  [⟨.adj, 1, 0⟩, ⟨.lblsM, 1, 0⟩]

/-- `AttributeStore::encode_with_range` over `live`: skip `offset` live ids, then write
until `count` are written (one is written before the `>= count` test, so `count = 0`
still writes one — never reached: every payload entry has `count > 0`). -/
def encRange (S : Store) (ids : List Nat) (count off : Nat) : Stream :=
  ((ids.drop off).take (max count 1)).flatMap (encEnt S)

/-- `Graph::encode_payload` (graph.rs:5092-5145). -/
def encPayload (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (e : Entry) : Stream :=
  match e.st with
  | .nodes => encRange g.nodes (live g.nodeCount g.delNodes) e.count e.offset
  | .delNodes => encRoar g.delNodes e.count e.offset
  | .edges => encRange g.edges (live g.edgeCount g.delEdges) e.count e.offset
  | .delEdges => encRoar g.delEdges e.count e.offset
  | .labelsM => encU g.labelMats.length ++ encIdx C.Mx.enc g.labelMats
  | .relM => encIdx C.Tx.enc g.tensors
  | .adj => C.Mx.enc g.adj
  | .lblsM => C.Mx.enc g.lbls
  | _ => []

/-- `encode_graph` (encoder/mod.rs:68-88). -/
def encodeGraph (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (name : Nm) (ps : List Entry) (k : Nat) :
    Stream :=
  C.H.enc (hdrOf g name k) ++ C.S.enc g.sch ++ encDir ps ++ ps.flatMap (encPayload C g)

/-! ## Decoding -/

/-- What one payload yields. -/
inductive PR (M Tn : Type) where
  | nodes (es : List (Nat × List (Option Nat × Val)))
  | delN (l : List Nat)
  | edges (es : List (Nat × List (Option Nat × Val)))
  | delE (l : List Nat)
  | lm (ms : List M)
  | rt (ts : List Tn)
  | adj (m : M)
  | lbls (m : M)
  | skip

/-- The payload `match` of `rdb_load_graph` (:193-228), `decode_payloads_into_pending`
(:266-305) and `load_graph_from_reader` (:473-508) — the three are the same arms. -/
def decPayload (C : Codecs M Tn Nm Ix Cn) (relCount : Nat) : St × Nat → Dec (PR M Tn)
  | (.nodes, c) => (rep decEnt c).bind fun es => Dec.pure (.nodes es)
  | (.delNodes, c) => (decRoar c).bind fun l => Dec.pure (.delN l)
  | (.edges, c) => (rep decEnt c).bind fun es => Dec.pure (.edges es)
  | (.delEdges, c) => (decRoar c).bind fun l => Dec.pure (.delE l)
  | (.labelsM, _) => readU.bind fun n => (decIdx C.Mx.dec n).bind fun ms => Dec.pure (.lm ms)
  | (.relM, _) => (decIdx C.Tx.dec relCount).bind fun ts => Dec.pure (.rt ts)
  | (.adj, _) => C.Mx.dec.bind fun m => Dec.pure (.adj m)
  | (.lblsM, _) => C.Mx.dec.bind fun m => Dec.pure (.lbls m)
  | _ => Dec.pure .skip

/-- Read one payload per directory entry, in order. -/
def seqD {E β : Type} (f : E → Dec β) : List E → Dec (List β)
  | [] => Dec.pure []
  | e :: es => (f e).bind fun b => (seqD f es).bind fun bs => Dec.pure (b :: bs)

theorem RT_seq {E β : Type} (f : E → Dec β) (enc : E → Stream) (val : E → β) :
    ∀ es : List E, (∀ e ∈ es, RT (f e) (enc e) (val e)) → RT (seqD f es) (es.flatMap enc) (es.map val)
  | [], _ => RT_pure []
  | e :: es, h => by
    have h1 := h e (by simp)
    have h2 := RT_seq f enc val es (fun x hx => h x (by simp [hx]))
    have := RT_bind h1 (f := fun b => (seqD f es).bind fun bs => Dec.pure (b :: bs))
      (RT_bind h2 (f := fun bs => Dec.pure (val e :: bs)) (RT_pure _))
    simpa [seqD] using this

/-- The decoder's locals / the `PendingGraph` fields that feed `Graph::restore`. -/
structure Acc (M Tn : Type) where
  nodes : Store
  edges : Store
  delN : Nat → Bool
  delE : Nat → Bool
  lms : List M
  tns : List Tn
  adj : M
  lbls : M

def acc0 (C : Codecs M Tn Nm Ix Cn) : Acc M Tn :=
  ⟨fun _ => none, fun _ => none, fun _ => false, fun _ => false, [], [], C.m0, C.m0⟩

/-- `RoaringTreemap::insert` of each id. -/
def insAll (s : Nat → Bool) (l : List Nat) : Nat → Bool := l.foldl (fun s i => fun j => s j || j == i) s

/-- Fold one payload into the locals. `limit` is `attrs_name.len()`. -/
def applyPR (limit : Nat) (a : Acc M Tn) : PR M Tn → Acc M Tn
  | .nodes es => { a with nodes := es.foldl (applyEnt limit) a.nodes }
  | .delN l => { a with delN := insAll a.delN l }
  | .edges es => { a with edges := es.foldl (applyEnt limit) a.edges }
  | .delE l => { a with delE := insAll a.delE l }
  | .lm ms => { a with lms := a.lms ++ ms }
  | .rt ts => { a with tns := a.tns ++ ts }
  | .adj m => { a with adj := m }
  | .lbls m => { a with lbls := m }
  | .skip => a

/-- The arguments of `Graph::restore` (+ the indexes and constraints applied after it). -/
structure Restore (M Tn Nm Ix Cn : Type) where
  name : Nm
  nodeCount : Nat
  edgeCount : Nat
  acc : Acc M Tn
  labels : List Nm
  types : List Nm
  attrs : List Nm
  indexes : List Ix
  constraints : List Cn

/-- `load_graph_from_reader` (decoder/mod.rs:431-537), behind `pipe_load_graph` (:406)
and `vec_load_graph` (:421). `attrs_name.len()` is the schema's attribute count (the
names are distinct; `AttrNameMap` is an index set). -/
def finish (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (nc ec : Nat) (s : Sch Nm Ix Cn) (prs : List (PR M Tn)) :
    Restore M Tn Nm Ix Cn :=
  ⟨dest, nc, ec, prs.foldl (applyPR s.attrs.length) (acc0 C), s.labels, s.types, s.attrs, s.indexes,
    s.constraints⟩

/-- After the directory: one payload per entry, then `Graph::restore`. -/
def kDir (C : Codecs M Tn Nm Ix Cn) (rc : Nat) (dest : Nm) (nc ec : Nat) (s : Sch Nm Ix Cn)
    (dir : List (St × Nat)) : Dec (Restore M Tn Nm Ix Cn) :=
  (seqD (decPayload C rc) dir).bind fun prs => Dec.pure (finish C dest nc ec s prs)

def kSch (C : Codecs M Tn Nm Ix Cn) (rc : Nat) (dest : Nm) (nc ec : Nat) (s : Sch Nm Ix Cn) :
    Dec (Restore M Tn Nm Ix Cn) :=
  decDir.bind (kDir C rc dest nc ec s)

/-- `if hdr.key_count != 1 { return Err(..) }` (:438). -/
def kHdr (C : Codecs M Tn Nm Ix Cn) (dest : Nm) (h : Hdr Nm) : Dec (Restore M Tn Nm Ix Cn) :=
  if h.keyCount ≠ 1 then Dec.fail else C.S.dec.bind (kSch C h.relCount dest h.nodeCount h.edgeCount)

def loadFromReader (C : Codecs M Tn Nm Ix Cn) (dest : Nm) : Dec (Restore M Tn Nm Ix Cn) :=
  C.H.dec.bind (kHdr C dest)

/-- What a well-formed graph restores to. -/
def restoreOf (g : G M Tn Nm Ix Cn) (dest : Nm) : Restore M Tn Nm Ix Cn :=
  ⟨dest, g.nodeCount, g.edgeCount,
   ⟨g.nodes, g.edges, fun j => decide (j ∈ g.delNodes), fun j => decide (j ∈ g.delEdges),
    g.labelMats, g.tensors, g.adj, g.lbls⟩,
   g.sch.labels, g.sch.types, g.sch.attrs, g.sch.indexes, g.sch.constraints⟩

/-- One entity kind's well-formedness: the id space is `[0, count + |deleted|)`, the
deleted ids are distinct and inside it, ids fit `u64`, and only live ids carry a span,
each a `GoodSpan`. -/
def KindWF (limit count : Nat) (del : List Nat) (S : Store) : Prop :=
  count + del.length < 2 ^ 64 ∧ del.Nodup ∧ (∀ d ∈ del, d < count + del.length) ∧
  ∀ i sp, S i = some sp → i ∈ live count del ∧ GoodSpan limit sp

structure WF (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) : Prop where
  hdr : C.H.good (hdrOf g g.name 1)
  sch : C.S.good g.sch
  lim : g.sch.attrs.length ≤ 65536
  nk : KindWF g.sch.attrs.length g.nodeCount g.delNodes g.nodes
  ek : KindWF g.sch.attrs.length g.edgeCount g.delEdges g.edges
  lm : g.labelMats.length < 2 ^ 64 ∧ ∀ m ∈ g.labelMats, C.Mx.good m
  tn : g.tensors.length < 2 ^ 64 ∧ ∀ t ∈ g.tensors, C.Tx.good t
  mx : C.Mx.good g.adj ∧ C.Mx.good g.lbls

end GraphPersist.Persist
