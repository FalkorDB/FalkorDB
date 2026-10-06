import FalkorIndexLayer.LeafAos
/-
# `CompactLeaf` (`cow_btree/leaf/compact.rs`, origin/main 3fec7d7c9)

Page = 14-byte header (`count:u16, min:u64, value_width:u8, distinct_count:u16,
doc_width:u8`), then the value deltas, then (indexed pages only) the index,
then the docs.

| Lean | Rust |
| --- | --- |
| `header` | `write_compact_header` 72 |
| `layout` | `CompactLayout::read` 49 |
| `isIndexed` | `is_indexed` 28 |
| `packingFits` | `packing_fits` 108 |
| `cBytes`, `cBuild` | `CompactLeaf::build` |
| `cCount`, `cKey`, `cDoc`, `cDocLayout` | `count` 126, `key` 131, `doc` 140, `doc_layout` |
| `cInsert`, `cRemove`, `cMerge` | `splice_insert`, `splice_remove`, `merge` (+ `append_spliced_docs` 91) |
-/
namespace IndexLayer.Leaf

def header (n min vw dc dw : Nat) : List Nat := le 2 n ++ (le 8 min ++ ([vw] ++ (le 2 dc ++ [dw])))

theorem header_len (n min vw dc dw : Nat) : (header n min vw dc dw).length = 14 := by
  simp [header, le_length]

structure Hdr where
  n : Nat
  min : Nat
  vw : Nat
  dc : Nat
  dw : Nat

def Hdr.ok (h : Hdr) : Prop := h.n < 2 ^ 16 ∧ h.dc < 2 ^ 16 ∧ U64 h.min ∧ W h.vw ∧ W h.dw

theorem header_read (h : Hdr) (hok : h.ok) (B : List Nat) :
    let b := header h.n h.min h.vw h.dc h.dw ++ B
    readU16 b 0 = h.n ∧ readU64 b 2 = h.min ∧ byteAt b 10 = h.vw ∧ readU16 b 11 = h.dc ∧
      byteAt b 13 = h.dw := by
  obtain ⟨hn, hd, hm, -, -⟩ := hok
  have e : header h.n h.min h.vw h.dc h.dw ++ B =
      le 2 h.n ++ (le 8 h.min ++ ([h.vw] ++ (le 2 h.dc ++ ([h.dw] ++ B)))) := by
    simp [header, List.append_assoc]
  simp only [e]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · unfold readU16
    rw [rd_app_left _ _ 0 2 (by simp [le_length]), rd_le 2 _ (by simpa using hn)]
  · unfold readU64
    rw [rd_app_right' _ _ _ _ (by simp [le_length]), rd_app_left _ _ _ 8 (by simp [le_length])]
    simp only [le_length]; exact rd_le 8 _ (u64_256 _ hm)
  · simp [byteAt, List.getD, le, le_length]
  · unfold readU16
    rw [rd_app_right' _ _ _ _ (by simp [le_length]), rd_app_right' _ _ _ _ (by simp [le_length]),
      rd_app_right' _ _ _ _ (by simp [le_length]), rd_app_left _ _ _ 2 (by simp [le_length])]
    simp only [le_length, List.length_singleton]; exact rd_le 2 _ (by simpa using hd)
  · simp [byteAt, List.getD, le, le_length]

/-! ## Non-indexed compact pages -/

/-- A compact page from its header and decoded columns. -/
def cBytes (h : Hdr) (deltas docs : List Nat) : List Nat :=
  header h.n h.min h.vw h.dc h.dw ++ deltas.flatMap (le h.vw) ++ docs.flatMap (le h.dw)

/-- `CompactLeaf::build` (distinct_count written as `count`). -/
def cBuild (ps : List P) (min vw dw : Nat) : List Nat :=
  cBytes ⟨ps.length, min, vw, ps.length, dw⟩ (ps.map (fun p => p.1 - min)) (ps.map (·.2))

def cCount (b : List Nat) : Nat := readU16 b 0
def cKey (b : List Nat) (i : Nat) : Nat := readU64 b 2 + readWidth b (14 + i * byteAt b 10) (byteAt b 10)
def cDocLayout (b : List Nat) : Nat × Nat × Nat := (14 + cCount b * byteAt b 10, byteAt b 13, byteAt b 13)
def cDoc (b : List Nat) (i : Nat) : Nat :=
  readWidth b ((cDocLayout b).1 + i * (cDocLayout b).2.1) (cDocLayout b).2.2
def cPairs (b : List Nat) : List P := pairsOf (cCount b) (cKey b) (cDoc b)

/-- Columns fit their widths. -/
def CFits (min vw dw : Nat) (ps : List P) : Prop :=
  ∀ p ∈ ps, min ≤ p.1 ∧ p.1 - min < 256 ^ vw ∧ p.2 < 256 ^ dw

theorem cBytes_read (h : Hdr) (hok : h.ok) (ds os : List Nat) (hlen : ds.length = h.n) (hlo : os.length = h.n)
    (hdl : ∀ d ∈ ds, d < 256 ^ h.vw) (hol : ∀ d ∈ os, d < 256 ^ h.dw) :
    cCount (cBytes h ds os) = h.n ∧
    ∀ i (hi : i < h.n), cKey (cBytes h ds os) i = h.min + ds[i]'(by omega) ∧
      cDoc (cBytes h ds os) i = os[i]'(by omega) := by
  have hr := header_read h hok (ds.flatMap (le h.vw) ++ os.flatMap (le h.dw))
  simp only [cBytes] at *
  rw [← List.append_assoc] at hr
  obtain ⟨r1, r2, r3, -, r5⟩ := hr
  refine ⟨r1, fun i hi => ⟨?_, ?_⟩⟩
  · unfold cKey; rw [r2, r3, readWidth_eq _ _ _ hok.2.2.2.1,
      rd_chunk (le h.vw) h.vw (fun _ => le_length _ _) (header h.n h.min h.vw h.dc h.dw) _ ds i (by omega) _
        (by rw [header_len]), unle_le_of_lt _ _ (hdl _ (List.getElem_mem _))]
  · unfold cDoc cDocLayout cCount
    rw [r1, r3, r5, readWidth_eq _ _ _ hok.2.2.2.2]
    rw [show header h.n h.min h.vw h.dc h.dw ++ ds.flatMap (le h.vw) ++ os.flatMap (le h.dw) =
      (header h.n h.min h.vw h.dc h.dw ++ ds.flatMap (le h.vw)) ++ os.flatMap (le h.dw) ++ [] by simp]
    rw [rd_chunk (le h.dw) h.dw (fun _ => le_length _ _) _ [] os i (by omega) _
        (by simp [header_len, length_flatMap_const (le h.vw) h.vw (fun _ => le_length _ _), hlen]),
      unle_le_of_lt _ _ (hol _ (List.getElem_mem _))]

/-- **PROVEN** (`CompactLeaf::build` round trip). -/
theorem cBuild_roundtrip (ps : List P) (min vw dw : Nat) (hn : ps.length < 2 ^ 16) (hm : U64 min)
    (hvw : W vw) (hdw : W dw) (hf : CFits min vw dw ps) : cPairs (cBuild ps min vw dw) = ps := by
  have hok : (Hdr.ok ⟨ps.length, min, vw, ps.length, dw⟩) := ⟨hn, hn, hm, hvw, hdw⟩
  obtain ⟨c1, c2⟩ := cBytes_read ⟨ps.length, min, vw, ps.length, dw⟩ hok (ps.map (fun p => p.1 - min))
    (ps.map (·.2)) (by simp) (by simp)
    (fun d hd => by obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hd; exact (hf p hp).2.1)
    (fun d hd => by obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hd; exact (hf p hp).2.2)
  unfold cPairs cBuild
  rw [c1]
  apply pairsOf_eq
  intro i hi
  obtain ⟨k1, k2⟩ := c2 i hi
  rw [k1, k2]
  have := (hf ps[i] (List.getElem_mem hi)).1
  simp; omega

/-! ### `splice_insert` / `splice_remove` -/

def insAt {α : Type} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop i
def delAt {α : Type} (l : List α) (i : Nat) : List α := l.take i ++ l.drop (i + 1)

/-- `CompactLayout::read` (`compact.rs:49`): offsets of the index and doc bodies. -/
def layout (b : List Nat) : Nat × Nat :=
  let n := readU16 b 0
  let dc := readU16 b 11
  let vw := byteAt b 10
  let io := 14 + dc * vw
  (io, io + if dc < n then n else 0)

/-- `is_indexed` (`compact.rs:28`). -/
def isIndexed (b : List Nat) : Bool := decide (readU16 b 11 < readU16 b 0)

/-- `packing_fits` (`compact.rs:108`). -/
def packingFits (b : List Nat) (key doc : Nat) : Bool :=
  decide (readU64 b 2 ≤ key) && decide (widthFor (key - readU64 b 2) ≤ byteAt b 10) &&
    decide (widthFor doc ≤ byteAt b 13)

theorem packingFits_spec (b : List Nat) (key doc : Nat) (hk : U64 key) (hd : U64 doc)
    (hw : W (byteAt b 10)) (hw' : W (byteAt b 13)) (h : packingFits b key doc = true) :
    readU64 b 2 ≤ key ∧ key - readU64 b 2 < 256 ^ byteAt b 10 ∧ doc < 256 ^ byteAt b 13 := by
  simp only [packingFits, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, lt_of_widthFor_le _ _ (by unfold U64 at hk; omega) hw h.1.2, lt_of_widthFor_le _ _ hd hw' h.2⟩

/-- `CompactLeaf::splice_insert`, byte for byte. -/
def cInsert (b : List Nat) (key doc pos : Nat) : List Nat :=
  let n := readU16 b 0
  let min := readU64 b 2
  let vw := byteAt b 10
  let dw := byteAt b 13
  let docsOff := (layout b).2
  header (n + 1) min vw (n + 1) dw ++ slice b 14 (14 + pos * vw) ++ (le 8 (key - min)).take vw ++
    slice b (14 + pos * vw) docsOff ++
    -- `append_spliced_docs`
    (slice b docsOff (docsOff + pos * dw) ++ (le 8 doc).take dw ++ slice b (docsOff + pos * dw) (docsOff + n * dw))

/-- `CompactLeaf::splice_remove`. -/
def cRemove (b : List Nat) (pos : Nat) : List Nat :=
  let n := readU16 b 0
  let min := readU64 b 2
  let vw := byteAt b 10
  let dw := byteAt b 13
  let docsOff := (layout b).2
  header (n - 1) min vw (n - 1) dw ++ slice b 14 (14 + pos * vw) ++ slice b (14 + (pos + 1) * vw) docsOff ++
    slice b docsOff (docsOff + pos * dw) ++ slice b (docsOff + (pos + 1) * dw) (docsOff + n * dw)

theorem W_le8 (w : Nat) (h : W w) : w ≤ 8 := by rcases h with rfl | rfl | rfl | rfl <;> omega
theorem W_pos (w : Nat) (h : W w) : 0 < w := by rcases h with rfl | rfl | rfl | rfl <;> omega

/-- The slices of a compact page are its column chunks. -/
theorem cSlices (h : Hdr) (ds os : List Nat) (hlen : ds.length = h.n) (hlo : os.length = h.n) (i j : Nat) (hij : i ≤ j) (hj : j ≤ h.n) :
    slice (cBytes h ds os) (14 + i * h.vw) (14 + j * h.vw) = ((ds.drop i).take (j - i)).flatMap (le h.vw) ∧
    slice (cBytes h ds os) (14 + h.n * h.vw + i * h.dw) (14 + h.n * h.vw + j * h.dw) =
      ((os.drop i).take (j - i)).flatMap (le h.dw) := by
  constructor
  · have := slice_chunks (le h.vw) h.vw (fun _ => le_length _ _) (header h.n h.min h.vw h.dc h.dw)
      (os.flatMap (le h.dw)) ds i j hij (by omega)
    rw [header_len] at this; unfold cBytes; exact this
  · have := slice_chunks (le h.dw) h.dw (fun _ => le_length _ _)
      (header h.n h.min h.vw h.dc h.dw ++ ds.flatMap (le h.vw)) [] os i j hij (by omega)
    simp only [List.append_nil, List.length_append, header_len,
      length_flatMap_const (le h.vw) h.vw (fun _ => le_length _ _), hlen] at this
    unfold cBytes; exact this

theorem flatMap_insAt {α : Type} (f : α → List Nat) (l : List α) (i : Nat) (x : α) :
    (insAt l i x).flatMap f = (l.take i).flatMap f ++ f x ++ (l.drop i).flatMap f := by
  simp [insAt, List.flatMap_append]

theorem flatMap_delAt {α : Type} (f : α → List Nat) (l : List α) (i : Nat) :
    (delAt l i).flatMap f = (l.take i).flatMap f ++ (l.drop (i + 1)).flatMap f := by
  simp [delAt, List.flatMap_append]

/-- Header fields of a compact page, read back. -/
theorem cBytes_hdr (h : Hdr) (hok : h.ok) (ds os : List Nat) :
    readU16 (cBytes h ds os) 0 = h.n ∧ readU64 (cBytes h ds os) 2 = h.min ∧ byteAt (cBytes h ds os) 10 = h.vw ∧
    readU16 (cBytes h ds os) 11 = h.dc ∧ byteAt (cBytes h ds os) 13 = h.dw := by
  have := header_read h hok (ds.flatMap (le h.vw) ++ os.flatMap (le h.dw))
  unfold cBytes; rw [List.append_assoc]; exact this

theorem cLayout (h : Hdr) (hok : h.ok) (ds os : List Nat) (hdc : h.dc = h.n) :
    (layout (cBytes h ds os)).2 = 14 + h.n * h.vw := by
  obtain ⟨r1, -, r3, r4, -⟩ := cBytes_hdr h hok ds os
  simp only [layout, r1, r3, r4, hdc]; simp

/-- **PROVEN** (`splice_insert` = insert into the columns): the spliced bytes are
exactly the page whose columns have `(key - min, doc)` inserted at `pos`. -/
theorem cInsert_spec (h : Hdr) (hok : h.ok) (ds os : List Nat) (hlen : ds.length = h.n) (hlo : os.length = h.n)
    (hdc : h.dc = h.n) (key doc pos : Nat) (hpos : pos ≤ h.n) :
    cInsert (cBytes h ds os) key doc pos =
      cBytes ⟨h.n + 1, h.min, h.vw, h.n + 1, h.dw⟩ (insAt ds pos (key - h.min)) (insAt os pos doc) := by
  obtain ⟨r1, r2, r3, -, r5⟩ := cBytes_hdr h hok ds os
  have hl := cLayout h hok ds os hdc
  have s1 := (cSlices h ds os hlen hlo 0 pos (by omega) hpos).1
  have s2 := (cSlices h ds os hlen hlo pos h.n hpos (Nat.le_refl _)).1
  have s3 := (cSlices h ds os hlen hlo 0 pos (by omega) hpos).2
  have s4 := (cSlices h ds os hlen hlo pos h.n hpos (Nat.le_refl _)).2
  simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at s1 s3
  rw [List.take_of_length_le (by simp [hlen])] at s2
  rw [List.take_of_length_le (by simp [hlo])] at s4
  unfold cInsert
  simp only [r1, r2, r3, r5, hl]
  rw [le8_take _ _ (W_le8 _ hok.2.2.2.1), le8_take _ _ (W_le8 _ hok.2.2.2.2), s1, s2, s3, s4]
  unfold cBytes
  rw [flatMap_insAt, flatMap_insAt]
  simp [List.append_assoc]

/-- **PROVEN** (`splice_remove` = delete from the columns). -/
theorem cRemove_spec (h : Hdr) (hok : h.ok) (ds os : List Nat) (hlen : ds.length = h.n) (hlo : os.length = h.n)
    (hdc : h.dc = h.n) (pos : Nat) (hpos : pos < h.n) :
    cRemove (cBytes h ds os) pos =
      cBytes ⟨h.n - 1, h.min, h.vw, h.n - 1, h.dw⟩ (delAt ds pos) (delAt os pos) := by
  obtain ⟨r1, r2, r3, -, r5⟩ := cBytes_hdr h hok ds os
  have hl := cLayout h hok ds os hdc
  have s1 := (cSlices h ds os hlen hlo 0 pos (by omega) (by omega)).1
  have s2 := (cSlices h ds os hlen hlo (pos + 1) h.n (by omega) (Nat.le_refl _)).1
  have s3 := (cSlices h ds os hlen hlo 0 pos (by omega) (by omega)).2
  have s4 := (cSlices h ds os hlen hlo (pos + 1) h.n (by omega) (Nat.le_refl _)).2
  simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at s1 s3
  rw [List.take_of_length_le (by simp [hlen])] at s2
  rw [List.take_of_length_le (by simp [hlo])] at s4
  unfold cRemove
  simp only [r1, r2, r3, r5, hl]
  rw [s1, s2, s3, s4]
  unfold cBytes
  rw [flatMap_delAt, flatMap_delAt]
  simp [List.append_assoc]

/-! ### `CompactLeaf::merge` -/

/-- `CompactLeaf::merge`, byte for byte: leaf entries' value/doc bytes are copied
raw from their old slots, batch entries are encoded. -/
def cMerge (b : List Nat) (batch : List P) : List Nat :=
  let min := readU64 b 2
  let vw := byteAt b 10
  let dw := byteAt b 13
  let docsOff := (layout b).2
  let out := mergeWI lexLe (cPairs b) 0 batch none
  let values := out.flatMap (pickEnc (fun vi => slice b (14 + vi * vw) (14 + vi * vw + vw))
    (fun x => (le 8 (x.1 - min)).take vw))
  let docs := out.flatMap (pickEnc (fun vi => slice b (docsOff + vi * dw) (docsOff + vi * dw + dw))
    (fun x => (le 8 x.2).take dw))
  header (values.length / vw) min vw (values.length / vw) dw ++ values ++ docs

theorem slice_one {α : Type} (f : α → List Nat) (l : List α) (i : Nat) (hi : i < l.length) :
    ((l.drop i).take (i + 1 - i)).flatMap f = f l[i] := by
  rw [show i + 1 - i = 1 by omega, List.drop_eq_getElem_cons hi, List.take_succ_cons, List.take_zero]; simp

/-- **PROVEN** (`CompactLeaf::merge` = merge of the entry lists): merging a batch
whose entries fit the page's widths into a compact page gives the compact page
of `merge_sorted(leaf, batch)`. -/
theorem cMerge_spec (ps batch : List P) (min vw dw : Nat) (hn : ps.length < 2 ^ 16) (hm : U64 min)
    (hvw : W vw) (hdw : W dw) (hf : CFits min vw dw ps) (hb : CFits min vw dw batch) :
    cMerge (cBuild ps min vw dw) batch =
      cBytes ⟨(mergeD lexLe ps batch none).length, min, vw, (mergeD lexLe ps batch none).length, dw⟩
        ((mergeD lexLe ps batch none).map (fun p => p.1 - min)) ((mergeD lexLe ps batch none).map (·.2)) := by
  have hok : (Hdr.ok ⟨ps.length, min, vw, ps.length, dw⟩) := ⟨hn, hn, hm, hvw, hdw⟩
  have hdec := cBuild_roundtrip ps min vw dw hn hm hvw hdw hf
  obtain ⟨r1, r2, r3, -, r5⟩ := cBytes_hdr _ hok (ps.map (fun p => p.1 - min)) (ps.map (·.2))
  have hl := cLayout _ hok (ps.map (fun p => p.1 - min)) (ps.map (·.2)) rfl
  have sl := cSlices ⟨ps.length, min, vw, ps.length, dw⟩ (ps.map (fun p => p.1 - min)) (ps.map (·.2))
    (by simp) (by simp)
  unfold cMerge
  simp only [cBuild] at hdec ⊢
  simp only [r2, r3, r5, hl, hdec]
  have hidx := fun x i h => mergeWI_idx lexLe ps 0 batch none x i h
  have hv := flatMap_raw (mergeWI lexLe ps 0 batch none) ps (fun (x : P) => le vw (x.1 - min))
    (fun vi => slice (cBytes ⟨ps.length, min, vw, ps.length, dw⟩ (ps.map (fun p => p.1 - min)) (ps.map (·.2)))
      (14 + vi * vw) (14 + vi * vw + vw))
    (fun i hi => by
      have := (sl i (i + 1) (by omega) (by simp; omega)).1
      simp only [Nat.succ_mul] at this
      rw [show 14 + i * vw + vw = 14 + (i * vw + vw) by omega, this, slice_one _ _ i (by simpa using hi)]
      simp)
    (fun x i h => by obtain ⟨h1, -, h3⟩ := hidx x i h; simp at h1 h3; exact ⟨h1, h3⟩)
    (fun (x : P) => (le 8 (x.1 - min)).take vw) (fun x => le8_take _ _ (W_le8 _ hvw))
  have hd := flatMap_raw (mergeWI lexLe ps 0 batch none) ps (fun (x : P) => le dw x.2)
    (fun vi => slice (cBytes ⟨ps.length, min, vw, ps.length, dw⟩ (ps.map (fun p => p.1 - min)) (ps.map (·.2)))
      (14 + ps.length * vw + vi * dw) (14 + ps.length * vw + vi * dw + dw))
    (fun i hi => by
      have := (sl i (i + 1) (by omega) (by simp; omega)).2
      simp only [Nat.succ_mul] at this
      rw [show 14 + ps.length * vw + i * dw + dw = 14 + ps.length * vw + (i * dw + dw) by omega, this,
        slice_one _ _ i (by simpa using hi)]
      simp)
    (fun x i h => by obtain ⟨h1, -, h3⟩ := hidx x i h; simp at h1 h3; exact ⟨h1, h3⟩)
    (fun (x : P) => (le 8 x.2).take dw) (fun x => le8_take _ _ (W_le8 _ hdw))
  rw [hv, hd]
  have hfst := mergeWI_fst lexLe ps 0 batch none
  have hlen : ((mergeWI lexLe ps 0 batch none).flatMap (fun e => le vw (e.1.1 - min))).length / vw =
      (mergeD lexLe ps batch none).length := by
    rw [length_flatMap_const _ vw (fun _ => le_length _ _), Nat.mul_div_cancel _ (W_pos _ hvw), ← hfst,
      List.length_map]
  rw [hlen]
  unfold cBytes
  rw [← hfst, List.map_map, List.map_map]
  simp [List.flatMap_map, Function.comp]

end IndexLayer.Leaf
