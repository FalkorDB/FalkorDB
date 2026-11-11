import FalkorIndexLayer.LeafCompact
/-
# `CompactIndexedLeaf` (`cow_btree/leaf/compact_indexed.rs`, origin/main 8743953a8)

Page = header, `distinct_count` value deltas, a one-byte slot index per entry,
then the docs. Entry `i` is `(min + values[index[i]], docs[i])`.

| Lean | Rust |
| --- | --- |
| `ciBytes`, `ciDecode` | the page layout; `key`/`doc` |
| `ciIndexOffset`, `ciDistinct` | `index_offset`, `distinct_count` |
| `ciKey`, `ciDoc`, `ciDocLayout`, `ciCount` | `key`, `doc`, `doc_layout`, `count` |
| `runs`, `ciBuild` | `CompactIndexedLeaf::build` |
| `distinctSlot` | `distinct_slot` |
| `ciInsertOk`, `ciInsertErr` | `splice_insert` (`Ok` / `Err` arms) |
| `ciRemove` | `splice_remove` |
-/
namespace IndexLayer.Leaf

theorem getD_of_lt {α : Type} (l : List α) (i : Nat) (d : α) (h : i < l.length) : l.getD i d = l[i] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

def ciBytes (h : Hdr) (dv idx docs : List Nat) : List Nat :=
  header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw)

def ciDecode (min : Nat) (dv idx docs : List Nat) : List P :=
  (idx.zip docs).map (fun sd => (min + dv.getD sd.1 0, sd.2))

def ciCount (b : List Nat) : Nat := readU16 b 0
def ciDistinct (b : List Nat) : Nat := readU16 b 11
def ciIndexOffset (b : List Nat) : Nat := 14 + ciDistinct b * byteAt b 10
def ciKey (b : List Nat) (i : Nat) : Nat :=
  readU64 b 2 + readWidth b (14 + byteAt b (ciIndexOffset b + i) * byteAt b 10) (byteAt b 10)
def ciDocLayout (b : List Nat) : Nat × Nat × Nat := (ciIndexOffset b + ciCount b, byteAt b 13, byteAt b 13)
def ciDoc (b : List Nat) (i : Nat) : Nat :=
  readWidth b ((ciDocLayout b).1 + i * (ciDocLayout b).2.1) (ciDocLayout b).2.2
def ciPairs (b : List Nat) : List P := pairsOf (ciCount b) (ciKey b) (ciDoc b)

/-- Well-formed columns. -/
structure CIOk (h : Hdr) (dv idx docs : List Nat) : Prop where
  hok : h.ok
  dvlen : dv.length = h.dc
  idxlen : idx.length = h.n
  doclen : docs.length = h.n
  slots : ∀ s ∈ idx, s < h.dc
  dvfit : ∀ d ∈ dv, d < 256 ^ h.vw
  docfit : ∀ d ∈ docs, d < 256 ^ h.dw

theorem byteAt_mid (A B C : List Nat) (i : Nat) (hi : i < B.length) :
    byteAt (A ++ B ++ C) (A.length + i) = B[i] := by
  unfold byteAt
  rw [getD_of_lt _ _ _ (by simp; omega)]
  rw [List.getElem_append_left (by simp; omega), List.getElem_append_right (by omega)]
  simp

theorem ciBytes_hdr (h : Hdr) (hok : h.ok) (dv idx docs : List Nat) :
    readU16 (ciBytes h dv idx docs) 0 = h.n ∧ readU64 (ciBytes h dv idx docs) 2 = h.min ∧
    byteAt (ciBytes h dv idx docs) 10 = h.vw ∧ readU16 (ciBytes h dv idx docs) 11 = h.dc ∧
    byteAt (ciBytes h dv idx docs) 13 = h.dw := by
  have := header_read h hok (dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw))
  unfold ciBytes; simp only [List.append_assoc] at this ⊢; exact this

theorem ciPre_len (h : Hdr) (dv : List Nat) (hdv : dv.length = h.dc) :
    (header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw)).length = 14 + h.dc * h.vw := by
  simp [header_len, length_flatMap_const (le h.vw) h.vw (fun _ => le_length _ _), hdv]

/-- **PROVEN** (indexed page read-back): entry `i` is `(min + values[index[i]], docs[i])`. -/
theorem ciRead (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) :
    ciCount (ciBytes h dv idx docs) = h.n ∧ ciDistinct (ciBytes h dv idx docs) = h.dc ∧
    ciPairs (ciBytes h dv idx docs) = ciDecode h.min dv idx docs := by
  obtain ⟨r1, r2, r3, r4, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
  refine ⟨r1, r4, ?_⟩
  have hpre := ciPre_len h dv ok.dvlen
  have hio : ciIndexOffset (ciBytes h dv idx docs) = 14 + h.dc * h.vw := by
    simp only [ciIndexOffset, ciDistinct, r4, r3]
  unfold ciPairs ciCount
  rw [r1]
  have hlen : (ciDecode h.min dv idx docs).length = h.n := by
    simp [ciDecode, ok.idxlen, ok.doclen]
  rw [← hlen]
  apply pairsOf_eq
  intro i hi
  have hi' : i < h.n := by rw [hlen] at hi; exact hi
  have hidx : i < idx.length := by rw [ok.idxlen]; exact hi'
  have hdoc : i < docs.length := by rw [ok.doclen]; exact hi'
  have hs : idx[i] < dv.length := by rw [ok.dvlen]; exact ok.slots _ (List.getElem_mem _)
  simp only [ciDecode, List.getElem_map, List.getElem_zip]
  constructor
  · unfold ciKey
    rw [r2, r3, hio]
    have hb : byteAt (ciBytes h dv idx docs) (14 + h.dc * h.vw + i) = idx[i] := by
      unfold ciBytes
      rw [show header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw) =
        (header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw)) ++ idx ++ docs.flatMap (le h.dw) by rfl,
        ← hpre, byteAt_mid _ _ _ i hidx]
    rw [hb, readWidth_eq _ _ _ ok.hok.2.2.2.1]
    unfold ciBytes
    rw [show header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw) =
      header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ (idx ++ docs.flatMap (le h.dw)) by simp,
      rd_chunk (le h.vw) h.vw (fun _ => le_length _ _) _ _ dv idx[i] hs _ (by rw [header_len]),
      unle_le_of_lt _ _ (ok.dvfit _ (List.getElem_mem _)), getD_of_lt _ _ _ hs]
  · unfold ciDoc ciDocLayout ciCount
    rw [hio, r1, r5, readWidth_eq _ _ _ ok.hok.2.2.2.2, Nat.add_assoc 14]
    unfold ciBytes
    rw [show header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw) =
      (header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx) ++ docs.flatMap (le h.dw) ++ [] by simp,
      rd_chunk (le h.dw) h.dw (fun _ => le_length _ _) _ [] docs i hdoc _
        (by simp [header_len, length_flatMap_const (le h.vw) h.vw (fun _ => le_length _ _), ok.dvlen, ok.idxlen]),
      unle_le_of_lt _ _ (ok.docfit _ (List.getElem_mem _))]

/-! ## `build` -/

/-- The run-detection loop of `build`: a key opens a new distinct slot unless it
equals the last one. Returns the final distinct list and the index bytes. -/
def runs : List P → List Nat → List Nat × List Nat
  | [], d => (d, [])
  | p :: r, d =>
    let d' := if d.getLast? = some p.1 then d else d ++ [p.1]
    ((runs r d').1, (d'.length - 1) :: (runs r d').2)

theorem runs_spec : ∀ (ps : List P) (d : List Nat),
    (∃ ext, (runs ps d).1 = d ++ ext) ∧ (runs ps d).2.length = ps.length ∧
    (∀ i, i < ps.length → (runs ps d).2.getD i 0 < (runs ps d).1.length ∧
      (runs ps d).1.getD ((runs ps d).2.getD i 0) 0 = (ps.getD i (0, 0)).1) ∧
    (∀ x ∈ (runs ps d).1, x ∈ d ∨ ∃ p ∈ ps, p.1 = x) ∧ (runs ps d).1.length ≤ d.length + ps.length
  | [], d => ⟨⟨[], by simp [runs]⟩, by simp [runs], fun i h => by simp at h,
      fun x hx => Or.inl (by simpa [runs] using hx), by simp [runs]⟩
  | p :: r, d => by
    simp only [runs]
    generalize hd' : (if d.getLast? = some p.1 then d else d ++ [p.1]) = d'
    obtain ⟨⟨ext, he⟩, h2, h3, h4, h5⟩ := runs_spec r d'
    have hlast : d'.getLast? = some p.1 := by
      rw [← hd']; split
      · next h => exact h
      · simp
    have hne : d'.length ≠ 0 := by intro h0; rw [List.length_eq_zero_iff.mp h0] at hlast; simp at hlast
    have hd'mem : ∀ x ∈ d', x ∈ d ∨ x = p.1 := by
      intro x hx; rw [← hd'] at hx; split at hx
      · exact Or.inl hx
      · simp at hx; exact hx
    have hd'len : d'.length ≤ d.length + 1 := by rw [← hd']; split <;> simp
    refine ⟨?_, by simp [h2], fun i hi => ?_, fun x hx => ?_, ?_⟩
    · rw [he, ← hd']; split
      · exact ⟨ext, rfl⟩
      · exact ⟨[p.1] ++ ext, by simp⟩
    · cases i with
      | zero =>
        simp only [List.getD_cons_zero]
        rw [he]
        refine ⟨by simp; omega, ?_⟩
        rw [getD_of_lt _ _ _ (by simp; omega), List.getElem_append_left (by omega)]
        rw [List.getLast?_eq_getElem?] at hlast
        rw [List.getElem?_eq_getElem (by omega)] at hlast
        simpa using hlast
      | succ i =>
        simp only [List.getD_cons_succ]
        exact h3 i (by simpa using hi)
    · rcases h4 x hx with h | ⟨q, hq, rfl⟩
      · rcases hd'mem x h with h | rfl
        · exact Or.inl h
        · exact Or.inr ⟨p, by simp, rfl⟩
      · exact Or.inr ⟨q, by simp [hq], rfl⟩
    · simp; omega

/-- `CompactIndexedLeaf::build`. -/
def ciBuild (ps : List P) (min vw dw : Nat) : List Nat :=
  ciBytes ⟨ps.length, min, vw, (runs ps []).1.length, dw⟩ ((runs ps []).1.map (· - min)) (runs ps []).2 (ps.map (·.2))

theorem ciDecode_eq (min : Nat) (dv idx docs : List Nat) (ps : List P) (hl : idx.length = ps.length)
    (hd : docs = ps.map (·.2)) (h : ∀ i, i < ps.length → min + dv.getD (idx.getD i 0) 0 = (ps.getD i (0, 0)).1) :
    ciDecode min dv idx docs = ps := by
  apply List.ext_getElem (by simp [ciDecode, hl, hd])
  intro i h1 h2
  simp only [ciDecode, List.getElem_map, List.getElem_zip, hd]
  have := h i h2
  rw [getD_of_lt idx i 0 (by omega), getD_of_lt ps i (0, 0) h2] at this
  rw [this]

/-- **PROVEN** (`CompactIndexedLeaf::build` round trip): with at most 256 entries
(so every slot fits the `u8` index; `LEAF_MAX <= 256` is a const assert) and
the deltas fitting their widths, the built page reads back as `ps`. -/
theorem ciBuild_roundtrip (ps : List P) (min vw dw : Nat) (hn : ps.length ≤ 256) (hm : U64 min)
    (hvw : W vw) (hdw : W dw) (hf : CFits min vw dw ps) :
    ciPairs (ciBuild ps min vw dw) = ps := by
  obtain ⟨-, hl, hs, hmem, hdl⟩ := runs_spec ps []
  simp only [List.length_nil, Nat.zero_add] at hdl
  have hkey : ∀ x ∈ (runs ps []).1, min ≤ x ∧ x - min < 256 ^ vw := by
    intro x hx
    rcases hmem x hx with h | ⟨p, hp, rfl⟩
    · simp at h
    · exact ⟨(hf p hp).1, (hf p hp).2.1⟩
  have ok : CIOk ⟨ps.length, min, vw, (runs ps []).1.length, dw⟩ ((runs ps []).1.map (· - min)) (runs ps []).2
      (ps.map (·.2)) := by
    refine ⟨⟨show ps.length < 2 ^ 16 by omega, show (runs ps []).1.length < 2 ^ 16 by omega, hm, hvw, hdw⟩,
      by simp, hl, by simp, ?_, ?_, ?_⟩
    · intro s hs'
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hs'
      have := (hs i (by omega)).1
      rwa [getD_of_lt _ _ _ hi] at this
    · intro d hd
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hd
      exact (hkey k hk).2
    · intro d hd; obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hd; exact (hf p hp).2.2
  unfold ciBuild
  rw [(ciRead _ _ _ _ ok).2.2]
  apply ciDecode_eq _ _ _ _ _ hl rfl
  intro i hi
  obtain ⟨h1, h2⟩ := hs i hi
  rw [getD_of_lt _ _ _ (by simpa using h1), List.getElem_map]
  rw [getD_of_lt _ _ _ h1] at h2
  have := (hkey _ (List.getElem_mem h1)).1
  rw [h2] at this ⊢
  show min + ((ps.getD i (0, 0)).1 - min) = (ps.getD i (0, 0)).1
  omega

end IndexLayer.Leaf
