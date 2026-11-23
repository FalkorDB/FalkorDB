import GraphPersist.Persist.RoundTrip
import GraphPersist.Persist.KeyRT
/-! # The header's id cap (`check_header`, decoder/validate.rs, #3643)

Right after `Header::decode`, `rdb_load_graph` (decoder/mod.rs:130) and
`load_graph_from_reader` (:526) call `check_header`, which refuses a header whose
`node_count + deleted_node_count` or `edge_count + deleted_edge_count` exceeds
`MAX_ID_BOUND = 2^34`. `loadFromReader` / `rdbKey` / `rdbLoad` stay as the models of
the code before #3643; `loadFromReaderChecked` / `rdbKeyChecked` / `rdbLoadChecked`
model it after.

* `load_encode_checked`, `key_rt_ver_checked` — below the cap the checked decoders read
  back exactly what `load_encode` / `key_rt_ver` say.
* `load_encode_fails_past_id_bound`, `key_rt_ver_fails_past_id_bound` — **counterexamples**:
  a well-formed graph (`WF`) with `2^34 + 1` live nodes, which `load_encode` / `key_rt_ver`
  say reads back, is refused by the checked decoders. `WF` bounds counts only by `2^64`.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

/-- `MAX_ID_BOUND` (decoder/validate.rs:21). -/
def MAX_ID_BOUND : Nat := 2 ^ 34

/-- `check_header` (decoder/validate.rs:26) accepts. -/
def checkHeader (h : Hdr Nm) : Bool :=
  decide (h.nodeCount + h.delNodeCount ≤ MAX_ID_BOUND) &&
    decide (h.edgeCount + h.delEdgeCount ≤ MAX_ID_BOUND)

/-- `load_graph_from_reader` after #3643. -/
def loadFromReaderChecked (C : Codecs M Tn Nm Ix Cn) (dest : Nm) : Dec (Restore M Tn Nm Ix Cn) :=
  C.H.dec.bind fun h => if checkHeader h then kHdr C dest h else Dec.fail

/-- `rdb_load_graph`'s reads after #3643. -/
def rdbKeyChecked (C : Codecs M Tn Nm Ix Cn) : Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) :=
  C.H.dec.bind fun h => if checkHeader h then kKeyHdr C h else Dec.fail

def rdbLoadChecked (C : Codecs M Tn Nm Ix Cn) (encver : Int) : Dec (Hdr Nm × Sch Nm Ix Cn × List (PR M Tn)) :=
  if checkVer encver then rdbKeyChecked C else Dec.fail

/-- Swapping the continuation at the value the first decoder reads keeps `RT`. -/
theorem RT_bind_congr {α β} {d : Dec α} {f g : α → Dec β} {t1 t2 : Stream} {x : α} {b : β}
    (h1 : RT d t1 x) (hfg : f x = g x) (h : RT (d.bind f) (t1 ++ t2) b) : RT (d.bind g) (t1 ++ t2) b := by
  have key : ∀ p q, p ++ q = t1 ++ t2 → (d.bind g) p = (d.bind f) p := by
    intro p q hpq
    rcases List.append_eq_append_iff.1 hpq with ⟨as, h, _⟩ | ⟨bs, h, _⟩
    · by_cases has : as = []
      · subst has
        simp at h
        subst h
        have e := h1.1 []
        simp at e
        simp only [Dec.bind, e, hfg]
      · have := h1.2 p ⟨as, has, h.symm⟩
        simp only [Dec.bind, this]
    · subst h
      simp only [Dec.bind, h1.1, hfg]
  refine ⟨fun r => ?_, fun p hp => ?_⟩
  · have e := h.1 r
    simp only [Dec.bind, List.append_assoc, h1.1] at e ⊢
    rw [← hfg]; exact e
  · have ⟨q, _, hq⟩ := hp
    rw [key p q hq]; exact h.2 p hp

theorem encodeGraph_split (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (name : Nm) (ps : List Entry) (k : Nat) :
    encodeGraph C g name ps k =
      C.H.enc (hdrOf g name k) ++ (C.S.enc g.sch ++ encDir ps ++ ps.flatMap (encPayload C g)) := by
  simp only [encodeGraph, List.append_assoc]

/-- The graph's id spaces fit the cap. -/
def WithinCap (g : G M Tn Nm Ix Cn) : Prop :=
  g.nodeCount + g.delNodes.length ≤ MAX_ID_BOUND ∧ g.edgeCount + g.delEdges.length ≤ MAX_ID_BOUND

theorem checkHeader_hdrOf (g : G M Tn Nm Ix Cn) (name : Nm) (k : Nat) :
    checkHeader (hdrOf g name k) = true ↔ WithinCap g := by
  simp only [checkHeader, hdrOf, WithinCap, Bool.and_eq_true]
  exact ⟨fun ⟨a, b⟩ => ⟨of_decide_eq_true a, of_decide_eq_true b⟩,
    fun ⟨a, b⟩ => ⟨decide_eq_true a, decide_eq_true b⟩⟩

/-- **`load_encode` below the cap**: the checked `load_graph_from_reader` reads a graph
within `MAX_ID_BOUND` back exactly as `load_encode` says, truncation included. -/
theorem load_encode_checked (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g)
    (hcap : WithinCap g) (dest : Nm) :
    RT (loadFromReaderChecked C dest) (encodeGraph C g g.name (buildPayloads g) 1) (restoreOf g dest) := by
  have h := load_encode C g hw dest
  rw [encodeGraph_split] at h ⊢
  exact RT_bind_congr (C.H.rt _ hw.hdr)
    (by simp only [(checkHeader_hdrOf g g.name 1).2 hcap, ite_true]) h

/-- **`key_rt_ver` below the cap.** -/
theorem key_rt_ver_checked (C : Codecs M Tn Nm Ix Cn) (g : G M Tn Nm Ix Cn) (hw : WF C g) (K : Nat)
    (hH : C.H.good (hdrOf g g.name K)) (es : List Entry) (hes : ∀ e ∈ es, EntryOK g e)
    (hn : es.length < 2 ^ 64) (hc : ∀ e ∈ es, e.count < 2 ^ 64) (v : Int) (hv : v = 19 ∨ v = 20)
    (hcap : WithinCap g) :
    RT (rdbLoadChecked C v) (encodeGraph C g g.name es K) (hdrOf g g.name K, g.sch, es.map (prOfE g)) := by
  have hck : checkVer v = true := (checkVer_iff v).2 hv
  have h := key_rt C g hw K hH es hes hn hc
  simp only [rdbLoadChecked, hck, ite_true]
  rw [encodeGraph_split] at h ⊢
  exact RT_bind_congr (C.H.rt _ hH)
    (by simp only [(checkHeader_hdrOf g g.name K).2 hcap, ite_true]) h

/-! ## Counterexamples: a well-formed graph past the cap -/

/-- A codec for one value: no words, reads back `a`. -/
def kc {α : Type} (a : α) : Codec α := ⟨fun _ => [], Dec.pure a, fun x => x = a, by rintro x rfl; exact RT_pure _⟩

/-- `2^34 + 1` live nodes and nothing else (`UNWIND range(1, 17179869185) AS i CREATE ()`). -/
def gBig : G Unit Unit Unit Unit Unit :=
  { name := (), nodeCount := 2 ^ 34 + 1, edgeCount := 0, delNodes := [], delEdges := [],
    nodes := fun _ => none, edges := fun _ => none, labelMats := [], tensors := [], adj := (),
    lbls := (), multiEdge := [], sch := ⟨[], [], [], [], []⟩ }

def CBig : Codecs Unit Unit Unit Unit Unit :=
  ⟨kc (hdrOf gBig () 1), kc gBig.sch, kc (), kc (), ()⟩

theorem gBig_WF : WF CBig gBig where
  hdr := rfl
  sch := rfl
  lim := by simp [gBig]
  nk := by
    unfold KindWF
    refine ⟨by decide, List.nodup_nil, fun _ h => ?_, fun _ _ h => ?_⟩ <;> simp [gBig] at h
  ek := by
    unfold KindWF
    refine ⟨by decide, List.nodup_nil, fun _ h => ?_, fun _ _ h => ?_⟩ <;> simp [gBig] at h
  lm := ⟨by simp [gBig], by simp [gBig]⟩
  tn := ⟨by simp [gBig], by simp [gBig]⟩
  mx := ⟨rfl, rfl⟩

theorem gBig_past_cap : checkHeader (hdrOf gBig () 1) = false := by
  simp [checkHeader, hdrOf, gBig, MAX_ID_BOUND]

/-- **Counterexample to `load_encode` on the new code**: `gBig` is `WF`, so `load_encode`
holds for `loadFromReader` (the old code), but the checked `load_graph_from_reader`
refuses its own encoding (`GRAPH.COPY`, `GRAPH.RESTORE` of a `DUMP`). -/
theorem load_encode_fails_past_id_bound :
    WF CBig gBig ∧
    RT (loadFromReader CBig ()) (encodeGraph CBig gBig gBig.name (buildPayloads gBig) 1) (restoreOf gBig ()) ∧
    ¬ RT (loadFromReaderChecked CBig ()) (encodeGraph CBig gBig gBig.name (buildPayloads gBig) 1)
      (restoreOf gBig ()) := by
  refine ⟨gBig_WF, load_encode CBig gBig gBig_WF (), fun h => ?_⟩
  have e := h.1 []
  have hp : gBig.name = () := rfl
  simp only [loadFromReaderChecked, Dec.bind, CBig, kc, Dec.pure, hp, gBig_past_cap,
    Bool.false_eq_true, ite_false, Dec.fail] at e
  cases e

/-- **Counterexample to `key_rt_ver` on the new code**: the same graph, saved as one
version-20 RDB key, is refused on load. -/
theorem key_rt_ver_fails_past_id_bound :
    RT (rdbLoad CBig 20) (encodeGraph CBig gBig gBig.name [] 1) (hdrOf gBig gBig.name 1, gBig.sch, []) ∧
    ¬ RT (rdbLoadChecked CBig 20) (encodeGraph CBig gBig gBig.name [] 1)
      (hdrOf gBig gBig.name 1, gBig.sch, []) := by
  refine ⟨key_rt_ver CBig gBig gBig_WF 1 rfl [] (by simp) (by simp) (by simp) 20 (by simp),
    fun h => ?_⟩
  have e := h.1 []
  have hp : gBig.name = () := rfl
  have hv : checkVer 20 = true := by decide
  simp only [rdbLoadChecked, hv, ite_true, rdbKeyChecked, Dec.bind, CBig, kc, Dec.pure, hp,
    gBig_past_cap, Bool.false_eq_true, ite_false, Dec.fail] at e
  cases e

end GraphPersist.Persist
