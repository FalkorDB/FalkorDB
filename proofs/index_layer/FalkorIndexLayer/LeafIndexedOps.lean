import FalkorIndexLayer.LeafIndexed
/-
# `CompactIndexedLeaf` splices (`compact_indexed.rs`, origin/main 8743953a8):
`distinct_slot`, `splice_insert` (both arms), `splice_remove`.
-/
namespace IndexLayer.Leaf

inductive SlotR | ok (s : Nat) | err (s : Nat)

def slotValue (b : List Nat) (s : Nat) : Nat := readU64 b 2 + readWidth b (14 + s * byteAt b 10) (byteAt b 10)
def slotPred (b : List Nat) (key : Nat) : Nat → Bool := fun s => decide (slotValue b s < key)

/-- `distinct_slot`: binary search of the distinct table. -/
def distinctSlot (b : List Nat) (key : Nat) : SlotR :=
  let dc := readU16 b 11
  let slot := partitionPoint (slotPred b key) 0 dc
  if slot < dc ∧ slotValue b slot = key then .ok slot else .err slot

/-- The distinct table is non-decreasing (an invariant of every constructor). -/
def Sorted (dv : List Nat) : Prop := dv.Pairwise (· ≤ ·)

theorem ci_value (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (s : Nat) (hs : s < h.dc) :
    slotValue (ciBytes h dv idx docs) s = h.min + dv[s]'(by rw [ok.dvlen]; exact hs) := by
  unfold slotValue
  obtain ⟨-, r2, r3, -, -⟩ := ciBytes_hdr h ok.hok dv idx docs
  rw [r2, r3, readWidth_eq _ _ _ ok.hok.2.2.2.1]
  unfold ciBytes
  rw [show header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx ++ docs.flatMap (le h.dw) =
      header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ (idx ++ docs.flatMap (le h.dw)) by simp,
    rd_chunk (le h.vw) h.vw (fun _ => le_length _ _) _ _ dv s (by rw [ok.dvlen]; exact hs) _ (by rw [header_len]),
    unle_le_of_lt _ _ (ok.dvfit _ (List.getElem_mem _))]

/-- **PROVEN** (`distinct_slot`): on a sorted distinct table, `Ok(s)` is the slot
holding `key`, and `Err(s)` is the insertion point keeping the table sorted. -/
theorem distinctSlot_spec (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hsort : Sorted dv)
    (key : Nat) :
    (∀ s, distinctSlot (ciBytes h dv idx docs) key = .ok s → ∃ hs : s < dv.length, h.min + dv[s] = key) ∧
    (∀ s, distinctSlot (ciBytes h dv idx docs) key = .err s → s ≤ dv.length ∧
      (∀ t (ht : t < dv.length), t < s → h.min + dv[t] < key) ∧
      (∀ t (ht : t < dv.length), s ≤ t → key ≤ h.min + dv[t]) ∧ ∀ t (ht : t < dv.length), h.min + dv[t] ≠ key) := by
  obtain ⟨-, -, -, r4, -⟩ := ciBytes_hdr h ok.hok dv idx docs
  have hv := ci_value h dv idx docs ok
  have hmono : Mono (slotPred (ciBytes h dv idx docs) key) 0 h.dc := by
    intro i j _ hij hj hp
    simp only [slotPred, decide_eq_true_eq] at hp ⊢
    rw [hv i (by omega)]; rw [hv j hj] at hp
    have : dv[i]'(by rw [ok.dvlen]; omega) ≤ dv[j]'(by rw [ok.dvlen]; omega) := by
      rcases Nat.lt_or_ge i j with hlt | hge
      · exact List.pairwise_iff_getElem.mp hsort i j (by rw [ok.dvlen]; omega) (by rw [ok.dvlen]; omega) hlt
      · have : i = j := by omega
        subst this; exact Nat.le_refl _
    omega
  obtain ⟨p1, p2, p3, p4⟩ := partitionPoint_spec _ 0 h.dc (Nat.zero_le _) hmono
  unfold distinctSlot
  simp only [r4]
  generalize partitionPoint (slotPred (ciBytes h dv idx docs) key) 0 h.dc = sl at p1 p2 p3 p4 ⊢
  refine ⟨fun s hs => ?_, fun s hs => ?_⟩
  · split at hs
    · next hc => cases hs; obtain ⟨c1, c2⟩ := hc; rw [hv _ c1] at c2; exact ⟨by rw [ok.dvlen]; exact c1, c2⟩
    · cases hs
  · split at hs
    · cases hs
    · next hc =>
      cases hs
      refine ⟨by rw [ok.dvlen]; exact p2, fun t ht hts => ?_, fun t ht hst => ?_, fun t ht he => ?_⟩
      · have := p3 t (Nat.zero_le _) hts
        simp only [slotPred, decide_eq_true_eq] at this; rw [hv t (by rw [← ok.dvlen]; exact ht)] at this; exact this
      · have := p4 t hst (by rw [← ok.dvlen]; exact ht)
        simp only [slotPred, decide_eq_false_iff_not, Nat.not_lt] at this; rw [hv t (by rw [← ok.dvlen]; exact ht)] at this
        exact this
      · -- the key is present: then the partition point lands on it
        rcases Nat.lt_or_ge t (sl) with hlt | hge
        · have := p3 t (Nat.zero_le _) hlt
          simp only [slotPred, decide_eq_true_eq] at this; rw [hv t (by rw [← ok.dvlen]; exact ht)] at this; omega
        · -- slot ≤ t < dc, value slot ≥ key, value slot ≤ value t = key
          have hslt : sl < h.dc := by rw [← ok.dvlen]; omega
          have h1 := p4 _ (Nat.le_refl _) hslt
          simp only [slotPred, decide_eq_false_iff_not, Nat.not_lt] at h1
          rw [hv _ hslt] at h1
          have h2 : dv[sl]'(by rw [ok.dvlen]; exact hslt) ≤ dv[t] := by
            rcases Nat.lt_or_ge (sl) t with hl | hg
            · exact List.pairwise_iff_getElem.mp hsort _ _ _ _ hl
            · have : sl = t := by omega
              simp only [this]; exact Nat.le_refl _
          apply hc
          refine ⟨hslt, ?_⟩
          rw [hv _ hslt]; omega

/-! ## Region slices of an indexed page -/

theorem flatMap_single (l : List Nat) : l.flatMap (fun x => [x]) = l := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

theorem ciSlices (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) :
    (∀ i j, i ≤ j → j ≤ h.dc → slice (ciBytes h dv idx docs) (14 + i * h.vw) (14 + j * h.vw) =
      ((dv.drop i).take (j - i)).flatMap (le h.vw)) ∧
    (∀ i j, i ≤ j → j ≤ h.n → slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + i) (14 + h.dc * h.vw + j) =
      (idx.drop i).take (j - i)) ∧
    (∀ i j, i ≤ j → j ≤ h.n → slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + h.n + i * h.dw)
        (14 + h.dc * h.vw + h.n + j * h.dw) = ((docs.drop i).take (j - i)).flatMap (le h.dw)) ∧
    (∀ k (hk : k < h.n), byteAt (ciBytes h dv idx docs) (14 + h.dc * h.vw + k) = idx[k]'(by rw [ok.idxlen]; exact hk)) := by
  have hpre := ciPre_len h dv ok.dvlen
  refine ⟨fun i j hij hj => ?_, fun i j hij hj => ?_, fun i j hij hj => ?_, fun k hk => ?_⟩
  · have := slice_chunks (le h.vw) h.vw (fun _ => le_length _ _) (header h.n h.min h.vw h.dc h.dw)
      (idx ++ docs.flatMap (le h.dw)) dv i j hij (by rw [ok.dvlen]; exact hj)
    rw [header_len] at this
    unfold ciBytes; rw [List.append_assoc (header _ _ _ _ _ ++ _)]; exact this
  · have := slice_chunks (fun x => [x]) 1 (fun _ => rfl) (header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw))
      (docs.flatMap (le h.dw)) idx i j hij (by rw [ok.idxlen]; exact hj)
    rw [hpre, flatMap_single, flatMap_single] at this
    simp only [Nat.mul_one] at this
    unfold ciBytes; exact this
  · have := slice_chunks (le h.dw) h.dw (fun _ => le_length _ _)
      (header h.n h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx) [] docs i j hij (by rw [ok.doclen]; exact hj)
    rw [List.length_append, hpre, ok.idxlen, List.append_nil] at this
    unfold ciBytes; exact this
  · unfold ciBytes; rw [← hpre, byteAt_mid _ _ _ k (by rw [ok.idxlen]; exact hk)]

/-! ## `splice_insert` -/

def remap (slot x : Nat) : Nat := if x ≥ slot then x + 1 else x

/-- `CompactIndexedLeaf::splice_insert`, both arms, byte for byte. -/
def ciInsert (b : List Nat) (key doc pos : Nat) : List Nat :=
  let n := readU16 b 0
  let min := readU64 b 2
  let vw := byteAt b 10
  let dc := readU16 b 11
  let dw := byteAt b 13
  let io := (layout b).1
  let docsOff := (layout b).2
  let docsSpliced := slice b docsOff (docsOff + pos * dw) ++ (le 8 doc).take dw ++
    slice b (docsOff + pos * dw) (docsOff + n * dw)
  match distinctSlot b key with
  | .ok slot =>
    header (n + 1) min vw dc dw ++ slice b 14 io ++ slice b io (io + pos) ++ [slot] ++ slice b (io + pos) (io + n) ++
      docsSpliced
  | .err slot =>
    header (n + 1) min vw (dc + 1) dw ++ slice b 14 (14 + slot * vw) ++ (le 8 (key - min)).take vw ++
      slice b (14 + slot * vw) io ++ (List.range pos).map (fun k => remap slot (byteAt b (io + k))) ++ [slot] ++
      (List.range (n - pos)).map (fun k => remap slot (byteAt b (io + pos + k))) ++ docsSpliced

theorem ciLayout (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hdc : h.dc < h.n) :
    layout (ciBytes h dv idx docs) = (14 + h.dc * h.vw, 14 + h.dc * h.vw + h.n) := by
  obtain ⟨r1, -, r3, r4, -⟩ := ciBytes_hdr h ok.hok dv idx docs
  simp only [layout, r1, r3, r4, hdc, ite_true]

theorem take_zip' {α β : Type} : ∀ (i : Nat) (a : List α) (b : List β),
    (a.take i).zip (b.take i) = (a.zip b).take i
  | 0, _, _ => by simp
  | _ + 1, [], _ => by simp
  | _ + 1, _ :: _, [] => by simp
  | i + 1, x :: a, y :: b => by simp [take_zip' i a b]

theorem drop_zip' {α β : Type} : ∀ (i : Nat) (a : List α) (b : List β),
    (a.drop i).zip (b.drop i) = (a.zip b).drop i
  | 0, _, _ => by simp
  | _ + 1, [], _ => by simp
  | _ + 1, _ :: _, [] => by simp
  | i + 1, x :: a, y :: b => by simp [drop_zip' i a b]

theorem zip_insAt {α β : Type} (a : List α) (b : List β) (i : Nat) (x : α) (y : β) (hl : a.length = b.length)
    (hi : i ≤ a.length) : (insAt a i x).zip (insAt b i y) = insAt (a.zip b) i (x, y) := by
  unfold insAt
  rw [List.zip_append (by simp; omega), List.zip_cons_cons, take_zip', drop_zip']

theorem zip_delAt {α β : Type} (a : List α) (b : List β) (i : Nat) (hl : a.length = b.length) :
    (delAt a i).zip (delAt b i) = delAt (a.zip b) i := by
  unfold delAt
  rw [List.zip_append (by simp; omega), take_zip', drop_zip']

theorem map_insAt {α β : Type} (f : α → β) (l : List α) (i : Nat) (x : α) :
    (insAt l i x).map f = insAt (l.map f) i (f x) := by simp [insAt, List.map_take, List.map_drop]

theorem map_delAt {α β : Type} (f : α → β) (l : List α) (i : Nat) :
    (delAt l i).map f = delAt (l.map f) i := by simp [delAt, List.map_take, List.map_drop]

theorem getD_insAt_remap (l : List Nat) (slot v x : Nat) (hs : slot ≤ l.length) (hx : x < l.length) :
    (insAt l slot v).getD (remap slot x) 0 = l.getD x 0 := by
  unfold insAt remap
  simp only [List.getD_eq_getElem?_getD]
  split
  · next hge =>
    rw [List.getElem?_append_right (by simp; omega)]
    simp only [List.length_take, Nat.min_eq_left hs]
    rw [show x + 1 - slot = (x - slot) + 1 by omega, List.getElem?_cons_succ, List.getElem?_drop]
    congr 2; omega
  · next hlt =>
    rw [List.getElem?_append_left (by simp; omega), List.getElem?_take]
    simp only [show x < slot by omega, ite_true]

theorem range_map_byte (b : List Nat) (off k : Nat) (g : Nat → Nat) (l : List Nat) (hk : k ≤ l.length)
    (hb : ∀ j (hj : j < l.length), byteAt b (off + j) = l[j]) :
    (List.range k).map (fun j => g (byteAt b (off + j))) = (l.take k).map g := by
  apply List.ext_getElem (by simp; omega)
  intro j h1 h2
  simp only [List.getElem_map, List.getElem_range, List.getElem_take]
  rw [hb j (by simp at h1; omega)]

/-- **PROVEN** (`splice_insert`): on a well-formed indexed page with a sorted
distinct table, inserting `(key, doc)` at `pos` yields a page that reads back as
the old entries with `(key, doc)` inserted at `pos` — through either arm: an
existing distinct value is reused (`Ok`), or a new one is spliced into the table
and every stored slot `>= slot` is shifted (`Err`). The `u8` slot bytes stay in
range because `dc + 1 < 256`. -/
theorem ciInsert_spec (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hsort : Sorted dv)
    (hdcn : h.dc < h.n) (hn : h.n + 1 < 2 ^ 16) (hdc : h.dc + 1 < 256)
    (key doc pos : Nat) (hpos : pos ≤ h.n) (hk : h.min ≤ key) (hkw : key - h.min < 256 ^ h.vw)
    (hdoc : doc < 256 ^ h.dw) :
    ciPairs (ciInsert (ciBytes h dv idx docs) key doc pos) = insAt (ciDecode h.min dv idx docs) pos (key, doc) := by
  obtain ⟨r1, r2, r3, r4, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
  obtain ⟨sV, sI, sD, sB⟩ := ciSlices h dv idx docs ok
  have hlay := ciLayout h dv idx docs ok hdcn
  obtain ⟨dsOk, dsErr⟩ := distinctSlot_spec h dv idx docs ok hsort key
  have ev : (le 8 (key - h.min)).take h.vw = le h.vw (key - h.min) := le8_take _ _ (W_le8 _ ok.hok.2.2.2.1)
  have ed : (le 8 doc).take h.dw = le h.dw doc := le8_take _ _ (W_le8 _ ok.hok.2.2.2.2)
  have dsp : slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + h.n) (14 + h.dc * h.vw + h.n + pos * h.dw) ++
      le h.dw doc ++ slice (ciBytes h dv idx docs) (14 + h.dc * h.vw + h.n + pos * h.dw)
        (14 + h.dc * h.vw + h.n + h.n * h.dw) = (insAt docs pos doc).flatMap (le h.dw) := by
    have a := sD 0 pos (Nat.zero_le _) hpos
    have b := sD pos h.n hpos (Nat.le_refl _)
    simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at a
    rw [a, b, List.take_of_length_le (l := docs.drop pos) (by simp [ok.doclen]), insAt, List.flatMap_append, List.flatMap_cons,
      List.append_assoc]
  have hdocs : (insAt docs pos doc).length = h.n + 1 := by simp [insAt, ok.doclen]; omega
  unfold ciInsert
  simp only [r1, r2, r3, r4, r5, hlay, ev, ed]
  split
  · next slot hs =>
    obtain ⟨hsl, hval⟩ := dsOk slot hs
    have v0 := sV 0 h.dc (Nat.zero_le _) (Nat.le_refl _)
    have i0 := sI 0 pos (Nat.zero_le _) hpos
    have i1 := sI pos h.n hpos (Nat.le_refl _)
    simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at v0 i0
    rw [v0, i0, i1, List.take_of_length_le (l := dv) (by simp [ok.dvlen]),
      List.take_of_length_le (l := idx.drop pos) (by simp [ok.idxlen]), dsp]
    have ok' : CIOk ⟨h.n + 1, h.min, h.vw, h.dc, h.dw⟩ dv (insAt idx pos slot) (insAt docs pos doc) :=
      ⟨⟨hn, by have := ok.hok.2.1; exact this, ok.hok.2.2.1, ok.hok.2.2.2.1, ok.hok.2.2.2.2⟩, ok.dvlen,
        by simp [insAt, ok.idxlen]; omega, hdocs,
        fun s hs' => by
          simp only [insAt, List.mem_append, List.mem_cons] at hs'
          rcases hs' with h' | rfl | h'
          · exact ok.slots s (List.mem_of_mem_take h')
          · rw [← ok.dvlen]; exact hsl
          · exact ok.slots s (List.mem_of_mem_drop h'),
        ok.dvfit,
        fun d hd => by
          simp only [insAt, List.mem_append, List.mem_cons] at hd
          rcases hd with h' | rfl | h'
          · exact ok.docfit d (List.mem_of_mem_take h')
          · exact hdoc
          · exact ok.docfit d (List.mem_of_mem_drop h')⟩
    have e : header (h.n + 1) h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx.take pos ++ [slot] ++
        idx.drop pos ++ (insAt docs pos doc).flatMap (le h.dw) =
        ciBytes ⟨h.n + 1, h.min, h.vw, h.dc, h.dw⟩ dv (insAt idx pos slot) (insAt docs pos doc) := by
      simp [ciBytes, insAt, List.append_assoc]
    rw [e, (ciRead _ _ _ _ ok').2.2]
    unfold ciDecode
    rw [zip_insAt _ _ _ _ _ (by rw [ok.idxlen, ok.doclen]) (by rw [ok.idxlen]; exact hpos), map_insAt]
    simp only [getD_of_lt dv slot 0 hsl, hval]
  · next slot hs =>
    obtain ⟨hsl, -, -, -⟩ := dsErr slot hs
    have hsl' : slot ≤ h.dc := by rw [← ok.dvlen]; exact hsl
    have v0 := sV 0 slot (Nat.zero_le _) hsl'
    have v1 := sV slot h.dc hsl' (Nat.le_refl _)
    simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at v0
    rw [v0, v1, List.take_of_length_le (l := dv.drop slot) (by simp [ok.dvlen])]
    have ib1 := range_map_byte (ciBytes h dv idx docs) (14 + h.dc * h.vw) pos (remap slot) idx
      (by rw [ok.idxlen]; exact hpos) (fun j hj => sB j (by rw [← ok.idxlen]; exact hj))
    have ib2 := range_map_byte (ciBytes h dv idx docs) (14 + h.dc * h.vw + pos) (h.n - pos) (remap slot)
      (idx.drop pos) (by simp [ok.idxlen])
      (fun j hj => by
        rw [Nat.add_assoc, sB (pos + j) (by simp [ok.idxlen] at hj; omega)]; simp)
    rw [ib1, ib2, List.take_of_length_le (l := idx.drop pos) (by simp [ok.idxlen]), dsp]
    have ok' : CIOk ⟨h.n + 1, h.min, h.vw, h.dc + 1, h.dw⟩ (insAt dv slot (key - h.min))
        (insAt (idx.map (remap slot)) pos slot) (insAt docs pos doc) :=
      ⟨⟨hn, show h.dc + 1 < 2 ^ 16 by omega, ok.hok.2.2.1, ok.hok.2.2.2.1, ok.hok.2.2.2.2⟩, by simp [insAt, ok.dvlen]; omega,
        by simp [insAt, ok.idxlen]; omega, hdocs,
        fun s' hs' => by
          simp only [insAt, List.mem_append, List.mem_cons] at hs'
          have hmap : ∀ y ∈ idx.map (remap slot), y < h.dc + 1 := by
            intro y hy; obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hy
            have := ok.slots x hx; show remap slot x < h.dc + 1; unfold remap; split <;> omega
          rcases hs' with h' | rfl | h'
          · exact hmap s' (List.mem_of_mem_take h')
          · show _ < h.dc + 1; omega
          · exact hmap s' (List.mem_of_mem_drop h'),
        fun d hd => by
          simp only [insAt, List.mem_append, List.mem_cons] at hd
          rcases hd with h' | rfl | h'
          · exact ok.dvfit d (List.mem_of_mem_take h')
          · exact hkw
          · exact ok.dvfit d (List.mem_of_mem_drop h'),
        fun d hd => by
          simp only [insAt, List.mem_append, List.mem_cons] at hd
          rcases hd with h' | rfl | h'
          · exact ok.docfit d (List.mem_of_mem_take h')
          · exact hdoc
          · exact ok.docfit d (List.mem_of_mem_drop h')⟩
    have e : header (h.n + 1) h.min h.vw (h.dc + 1) h.dw ++ (dv.take slot).flatMap (le h.vw) ++ le h.vw (key - h.min) ++
        (dv.drop slot).flatMap (le h.vw) ++ (idx.take pos).map (remap slot) ++ [slot] ++
        (idx.drop pos).map (remap slot) ++ (insAt docs pos doc).flatMap (le h.dw) =
        ciBytes ⟨h.n + 1, h.min, h.vw, h.dc + 1, h.dw⟩ (insAt dv slot (key - h.min))
          (insAt (idx.map (remap slot)) pos slot) (insAt docs pos doc) := by
      simp [ciBytes, insAt, List.append_assoc, List.map_take, List.map_drop]
    rw [e, (ciRead _ _ _ _ ok').2.2]
    unfold ciDecode
    rw [zip_insAt _ _ _ _ _ (by simp [ok.idxlen, ok.doclen]) (by simp [ok.idxlen]; exact hpos), map_insAt]
    have hslot : (insAt dv slot (key - h.min)).getD slot 0 = key - h.min := by
      rw [getD_of_lt _ slot 0 (by simp [insAt, ok.dvlen]; omega)]
      simp only [insAt]
      rw [List.getElem_append_right (by simp; omega)]
      simp [Nat.min_eq_left hsl]
    have hl : ((idx.map (remap slot)).zip docs).map
        (fun sd => (h.min + (insAt dv slot (key - h.min)).getD sd.1 0, sd.2)) =
        (idx.zip docs).map (fun sd => (h.min + dv.getD sd.1 0, sd.2)) := by
      rw [List.zip_map_left, List.map_map]
      apply List.map_congr_left
      intro sd hsd
      simp only [Function.comp, Prod.map, id]
      rw [getD_insAt_remap dv slot _ sd.1 hsl (by rw [ok.dvlen]; exact ok.slots _ (List.of_mem_zip hsd).1)]
    rw [hl, hslot, show h.min + (key - h.min) = key by omega]

/-! ## `splice_remove` -/

/-- `CompactIndexedLeaf::splice_remove`, byte for byte (the distinct table is kept,
possibly with a now-unused value). -/
def ciRemove (b : List Nat) (pos : Nat) : List Nat :=
  let n := readU16 b 0
  let min := readU64 b 2
  let vw := byteAt b 10
  let dc := readU16 b 11
  let dw := byteAt b 13
  let io := (layout b).1
  let docsOff := (layout b).2
  header (n - 1) min vw dc dw ++ slice b 14 io ++ slice b io (io + pos) ++ slice b (io + pos + 1) (io + n) ++
    slice b docsOff (docsOff + pos * dw) ++ slice b (docsOff + (pos + 1) * dw) (docsOff + n * dw)

/-- **PROVEN** (`splice_remove`): the page reads back as the old entries minus
entry `pos`. -/
theorem ciRemove_spec (h : Hdr) (dv idx docs : List Nat) (ok : CIOk h dv idx docs) (hdcn : h.dc < h.n)
    (pos : Nat) (hpos : pos < h.n) :
    ciPairs (ciRemove (ciBytes h dv idx docs) pos) = delAt (ciDecode h.min dv idx docs) pos := by
  obtain ⟨r1, r2, r3, r4, r5⟩ := ciBytes_hdr h ok.hok dv idx docs
  obtain ⟨sV, sI, sD, -⟩ := ciSlices h dv idx docs ok
  have hlay := ciLayout h dv idx docs ok hdcn
  unfold ciRemove
  simp only [r1, r2, r3, r4, r5, hlay]
  have v0 := sV 0 h.dc (Nat.zero_le _) (Nat.le_refl _)
  have i0 := sI 0 pos (Nat.zero_le _) (by omega)
  have i1 := sI (pos + 1) h.n (by omega) (Nat.le_refl _)
  have d0 := sD 0 pos (Nat.zero_le _) (by omega)
  have d1 := sD (pos + 1) h.n (by omega) (Nat.le_refl _)
  simp only [Nat.zero_mul, Nat.add_zero, List.drop_zero, Nat.sub_zero] at v0 i0 d0
  rw [show 14 + h.dc * h.vw + pos + 1 = 14 + h.dc * h.vw + (pos + 1) by omega, v0, i0, i1, d0, d1,
    List.take_of_length_le (l := dv) (by simp [ok.dvlen]),
    List.take_of_length_le (l := idx.drop (pos + 1)) (by simp [ok.idxlen]),
    List.take_of_length_le (l := docs.drop (pos + 1)) (by simp [ok.doclen])]
  have ok' : CIOk ⟨h.n - 1, h.min, h.vw, h.dc, h.dw⟩ dv (delAt idx pos) (delAt docs pos) :=
    ⟨⟨show h.n - 1 < 2 ^ 16 by have := ok.hok.1; omega, ok.hok.2.1, ok.hok.2.2.1, ok.hok.2.2.2.1, ok.hok.2.2.2.2⟩,
      ok.dvlen, by simp [delAt, ok.idxlen]; omega, by simp [delAt, ok.doclen]; omega,
      fun s' hs' => by
        simp only [delAt, List.mem_append] at hs'
        rcases hs' with h' | h'
        · exact ok.slots s' (List.mem_of_mem_take h')
        · exact ok.slots s' (List.mem_of_mem_drop h'),
      ok.dvfit,
      fun d hd => by
        simp only [delAt, List.mem_append] at hd
        rcases hd with h' | h'
        · exact ok.docfit d (List.mem_of_mem_take h')
        · exact ok.docfit d (List.mem_of_mem_drop h')⟩
  have e : header (h.n - 1) h.min h.vw h.dc h.dw ++ dv.flatMap (le h.vw) ++ idx.take pos ++ idx.drop (pos + 1) ++
      (docs.take pos).flatMap (le h.dw) ++ (docs.drop (pos + 1)).flatMap (le h.dw) =
      ciBytes ⟨h.n - 1, h.min, h.vw, h.dc, h.dw⟩ dv (delAt idx pos) (delAt docs pos) := by
    simp [ciBytes, delAt, List.append_assoc]
  rw [e, (ciRead _ _ _ _ ok').2.2]
  unfold ciDecode
  rw [zip_delAt _ _ _ (by rw [ok.idxlen, ok.doclen]), map_delAt]

end IndexLayer.Leaf
