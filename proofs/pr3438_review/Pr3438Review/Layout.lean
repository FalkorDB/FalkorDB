import Pr3438Review.Slice
import GraphPersist.Persist.MultiLoad3
import GraphPersist.Layout
/-
# The PR's encoders inside the RDB layout: same stream, same round trip

graph_persist models main's payload encoders on its graph `G` (entity ids
`0 .. nodeCount + |delNodes|`, i.e. `max_id + 1 = count + |deleted|`, as
`live`), and proves the whole save → load round trip: single key
(`load_encode`) and multi-key in any load order (`multi_load`). Its premises are
about the layout, not about how a slice is found, so the PR can only break them
by changing the bytes of a payload. It does not:

| here | there (PR 2fabe1a8a) |
| --- | --- |
| `encRangePR` | `AttributeStore::encode_with_range` (attribute_store.rs:1425) writing `encEnt` per entity |
| `encRoarPR`  | `Encode for RoaringTreemap::encode_with_range` (serialization.rs:180-197): `select(offset)`, `advance_to`, `take(count)` |
| `encPayloadPR` | `Graph::encode_payload` (graph.rs:4942-4975) with the two above |
| `encodeGraphPR` | `encode_graph` (encoder/mod.rs:66-88) |

* **`encRoarPR_eq`** (claim 3) — the deleted-ids seek writes main's bytes for
  every `count` and `offset` (main: `iter().skip(offset).take(count)` =
  graph_persist `encRoar`), including `offset` past the end.
* `encRangePR_eq` — for `count > 0` the attribute slice is main's.
* **`encoder_counts_pos`** — every entity entry `build_payloads` and
  `build_multi_key_payloads` emit has `count > 0`, so the `count == 0` early
  return is unreachable from `encode_payload` (its only caller,
  encoder/mod.rs:86; the payload lists come from those two builders only:
  encoder/mod.rs:16/48/60, redis_type.rs:459).
* **`encodeGraphPR_eq`** — on any layout of such entries the PR's whole stream
  *is* main's stream; hence **`load_encode_pr`** and **`multi_load_pr`**: the
  single- and multi-key round trips hold for the PR word for word.
* **`chunk_concat`** (claim 2, stated directly) — for the greedy slicing
  (`chunkType`, graph_persist `Layout.lean`) the concatenation of the per-slice
  encodes is one walk over the whole live range: slices of `vkey_max` live ids
  concatenate to the full encode.
-/
namespace Pr3438
open GraphPersist GraphPersist.Persist

variable {M Tn Nm Ix Cn R : Type}

/-! ## Deleted-ids encoder (claim 3) -/

/-- `Encode for RoaringTreemap::encode_with_range` on the PR. -/
def encRoarPR (H : RoaringContract R) (r : R) (count offset : Nat) : Stream :=
  [.bytes ((match H.select r offset with
    | some first => (H.iterFrom r first).take count
    | none => []).flatMap le8)]

/-- main's `iter().skip(offset).take(count)`, written straight from the crate's `iter()`. -/
def encRoarMain (H : RoaringContract R) (r : R) (count offset : Nat) : Stream :=
  [.bytes ((((H.elems r).drop offset).take count).flatMap le8)]

theorem encRoarMain_eq (H : RoaringContract R) (r : R) (count offset : Nat) :
    encRoarMain H r count offset = encRoar (H.elems r) count offset := rfl

/-- **Claim 3.** The seek writes main's bytes, for every `count` and `offset`. -/
theorem encRoarPR_eq (H : RoaringContract R) (r : R) (count offset : Nat) :
    encRoarPR H r count offset = encRoarMain H r count offset := by
  unfold encRoarPR encRoarMain
  rw [H.select_spec]
  cases h : (H.elems r)[offset]? with
  | none =>
    have hl : (H.elems r).length ≤ offset := List.getElem?_eq_none_iff.1 h
    simp [List.drop_eq_nil_of_le hl]
  | some a =>
    simp only
    rw [H.iterFrom_spec, filter_ge_select _ (H.sorted r) offset a h]

/-! ## Attribute slices -/

def encRangePR (H : RoaringContract R) (r : R) (S : Store) (maxId count offset : Nat) : Stream :=
  (prEncode H id r maxId count offset).flatMap (encEnt S)

theorem liveL_eq_live (n : Nat) (del : List Nat) (maxId : Nat) (h : maxId + 1 = n + del.length) :
    liveL del maxId = live n del := by
  simp only [liveL, live, h, decide_not]

theorem encRangePR_eq (H : RoaringContract R) (r : R) (S : Store) (n : Nat) (del : List Nat)
    (hd : H.elems r = del) (maxId : Nat) (hm : maxId + 1 = n + del.length) (count offset : Nat)
    (hc : 0 < count) :
    encRangePR H r S maxId count offset = encRange S (live n del) count offset := by
  unfold encRangePR encRange
  rw [prEncode_spec, if_neg (by omega), hd, liveL_eq_live n del maxId hm, Nat.max_eq_left hc]
  simp

/-! ## Payloads and the graph stream -/

/-- `Graph::encode_payload` on the PR; `max_node_id() = node_count + |deleted| - 1`
as graph_persist's `live` has it. -/
def encPayloadPR (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn)
    (e : Entry) : Stream :=
  match e.st with
  | .nodes => encRangePR H rn g.nodes (g.nodeCount + g.delNodes.length - 1) e.count e.offset
  | .delNodes => encRoarPR H rn e.count e.offset
  | .edges => encRangePR H re g.edges (g.edgeCount + g.delEdges.length - 1) e.count e.offset
  | .delEdges => encRoarPR H re e.count e.offset
  | _ => encPayload C g e

/-- The roaring sets behind the graph are its deleted-id lists. -/
structure Backs (H : RoaringContract R) (rn re : R) (g : G M Tn Nm Ix Cn) : Prop where
  nodes : H.elems rn = g.delNodes
  edges : H.elems re = g.delEdges

/-- An entity entry with a positive count inside its kind. -/
def SliceOK (g : G M Tn Nm Ix Cn) (e : Entry) : Prop :=
  IsEntity e.st → 0 < e.count ∧ e.offset + e.count ≤ kindTotal g e.st

theorem entryOK_slice (g : G M Tn Nm Ix Cn) (e : Entry) (h : EntryOK g e) : SliceOK g e := by
  intro hi
  rcases h with ⟨_, hc, ho⟩ | hm
  · exact ⟨hc, ho⟩
  · exfalso
    simp only [matrixEntries, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hm
    rcases hi with hs | hs | hs | hs <;>
      rcases hm with (hm | hm) | hm | hm <;> (try (split at hm <;> simp at hm)) <;>
      (try subst hm) <;> simp_all

theorem encPayloadPR_eq (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn)
    (hb : Backs H rn re g) (e : Entry) (he : SliceOK g e) :
    encPayloadPR H rn re C g e = encPayload C g e := by
  obtain ⟨st, count, offset⟩ := e
  cases st with
  | nodes =>
    obtain ⟨hc, ho⟩ := he (Or.inl rfl)
    simp only [kindTotal] at hc ho
    simp only [encPayloadPR, encPayload]
    exact encRangePR_eq H rn g.nodes g.nodeCount g.delNodes hb.nodes _ (by omega) count offset hc
  | delNodes =>
    simp only [encPayloadPR, encPayload]; rw [encRoarPR_eq, encRoarMain_eq, hb.nodes]
  | edges =>
    obtain ⟨hc, ho⟩ := he (Or.inr (Or.inr (Or.inl rfl)))
    simp only [kindTotal] at hc ho
    simp only [encPayloadPR, encPayload]
    exact encRangePR_eq H re g.edges g.edgeCount g.delEdges hb.edges _ (by omega) count offset hc
  | delEdges =>
    simp only [encPayloadPR, encPayload]; rw [encRoarPR_eq, encRoarMain_eq, hb.edges]
  | _ => rfl

/-- `encode_graph` on the PR. -/
def encodeGraphPR (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn)
    (name : Nm) (ps : List Entry) (k : Nat) : Stream :=
  C.H.enc (hdrOf g name k) ++ C.S.enc g.sch ++ encDir ps ++ ps.flatMap (encPayloadPR H rn re C g)

theorem encodeGraphPR_eq (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn)
    (hb : Backs H rn re g) (name : Nm) (ps : List Entry) (k : Nat) (hps : ∀ e ∈ ps, SliceOK g e) :
    encodeGraphPR H rn re C g name ps k = encodeGraph C g name ps k := by
  unfold encodeGraphPR encodeGraph
  congr 1
  induction ps with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons]
    rw [encPayloadPR_eq H rn re C g hb e (hps e (by simp)), ih (fun e' he' => hps e' (by simp [he']))]

/-! ## No caller passes `count = 0` -/

theorem buildPayloads_slice (g : G M Tn Nm Ix Cn) : ∀ e ∈ buildPayloads g, SliceOK g e := by
  intro e he hi
  simp only [buildPayloads, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at he
  rcases he with ((((((he | he) | he) | he) | he) | he) | he | he) <;>
    (try (split at he <;> simp at he)) <;> (try subst he) <;>
    simp_all [kindTotal, IsEntity]

/-- **`encoder_counts_pos`**: every entity payload either builder emits has a
positive count — `encode_with_range` is never called with `count = 0`. -/
theorem encoder_counts_pos (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (vmax : Nat)
    (ht : total g ≤ U64MAX) :
    (∀ e ∈ buildPayloads g, IsEntity e.st → 0 < e.count) ∧
    (0 < (buildMulti g vmax).length →
      ∀ k, ∀ e ∈ keyAt (buildMulti g vmax) k, IsEntity e.st → 0 < e.count) :=
  ⟨fun e he hi => (buildPayloads_slice g e he hi).1,
   fun hK k e he hi => (entryOK_slice g e ((layout_ok C g hw vmax ht hK k).1 e he) hi).1⟩

/-! ## The round trips, for the PR -/

/-- **Single key** (`vec_save_graph`/`pipe_save_graph`/one-key RDB): graph_persist's
`load_encode`, for the PR's encoder. -/
theorem load_encode_pr (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn)
    (hw : WF C g) (hb : Backs H rn re g) (dest : Nm) :
    RT (loadFromReader C dest) (encodeGraphPR H rn re C g g.name (buildPayloads g) 1) (restoreOf g dest) := by
  rw [encodeGraphPR_eq H rn re C g hb _ _ _ (buildPayloads_slice g)]
  exact load_encode C g hw dest

/-- **Multi key, any load order**: graph_persist's `multi_load`, for the PR's encoder. -/
theorem multi_load_pr [DecidableEq Nm] (H : RoaringContract R) (rn re : R) (C : Codecs M Tn Nm Ix Cn)
    (g : G M Tn Nm Ix Cn) (hw : WF C g) (hb : Backs H rn re g) (vmax : Nat)
    (ht : total g ≤ U64MAX) (hK : 2 ≤ (buildMulti g vmax).length)
    (hH : C.H.good (hdrOf g g.name (buildMulti g vmax).length))
    (ord : List Nat) (hord : ord.Perm (List.range (buildMulti g vmax).length)) (names : Nat → Nm) :
    (∀ k ∈ ord, RT (rdbKey C) (encodeGraphPR H rn re C g g.name (keyAt (buildMulti g vmax) k) (buildMulti g vmax).length)
      (hdrOf g g.name (buildMulti g vmax).length, g.sch, (keyAt (buildMulti g vmax) k).map (prOfE g))) ∧
    runKeys C (hdrOf g g.name (buildMulti g vmax).length) g.sch DS.empty
        (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g))) =
      ⟨none, some (restoreOf g g.name),
       metaOf (hdrOf g g.name (buildMulti g vmax).length)
        (ord.map fun k => (names k, (keyAt (buildMulti g vmax) k).map (prOfE g)))⟩ := by
  obtain ⟨h1, h2⟩ := multi_load C g hw vmax ht hK hH ord hord names
  refine ⟨fun k hk => ?_, h2⟩
  rw [encodeGraphPR_eq H rn re C g hb _ _ _
    (fun e he => entryOK_slice g e ((layout_ok C g hw vmax ht (by omega) k).1 e he))]
  exact h1 k hk

/-! ## Claim 2 directly: slices concatenate to the full walk -/

theorem take_drop_concat {α : Type} (L : List α) (off c t : Nat) :
    (L.drop off).take c ++ (L.drop (off + c)).take t = (L.drop off).take (c + t) := by
  rw [List.take_add, List.drop_drop]

/-- **`chunk_concat`**: slicing `total` live ids from `off` greedily into chunks of
`cap` (`chunkType`, the per-kind shape of `build_multi_key_payloads`) and
encoding each chunk with the PR's `encode_with_range` gives, concatenated, the
single walk over those `total` live ids. -/
theorem chunk_concat {E : Type} (H : RoaringContract R) (ent : Nat → E) (r : R) (maxId cap : Nat)
    (hcap : 0 < cap) : ∀ total off,
      (chunkType cap hcap off total).flatMap (fun s => prEncode H ent r maxId s.count s.offset) =
        ((((liveL (H.elems r) maxId).drop off).take total).map ent) := by
  intro total
  induction total using Nat.strongRecOn with
  | _ total ih =>
    intro off
    match total with
    | 0 => simp [chunkType]
    | n + 1 =>
      rw [chunkType]
      have hpos : 0 < min cap (n + 1) := Nat.lt_min.mpr ⟨hcap, Nat.succ_pos n⟩
      have hlt : n + 1 - min cap (n + 1) < n + 1 := by omega
      simp only [List.flatMap_cons]
      rw [ih _ hlt (off + min cap (n + 1)), prEncode_spec, if_neg (by omega), ← List.map_append,
        take_drop_concat, show min cap (n + 1) + (n + 1 - min cap (n + 1)) = n + 1 by omega]

/-- With `off = 0` and `total` = every live id, the concatenation is one full
`encode_with_range(count = total, offset = 0)`. -/
theorem chunk_concat_full {E : Type} (H : RoaringContract R) (ent : Nat → E) (r : R) (maxId cap : Nat)
    (hcap : 0 < cap) (htot : 0 < (liveL (H.elems r) maxId).length) :
    (chunkType cap hcap 0 (liveL (H.elems r) maxId).length).flatMap (fun s => prEncode H ent r maxId s.count s.offset) =
      prEncode H ent r maxId (liveL (H.elems r) maxId).length 0 := by
  rw [chunk_concat, prEncode_spec, if_neg (by omega)]

end Pr3438
