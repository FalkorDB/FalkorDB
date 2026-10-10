import FalkorIndexLayer.LeafDispatch
/-
# The narrowable doc width `DOC_BYTES` (#2278, 3597b3a82)

`AosLeaf<DOC_BYTES>` (`cow_btree/leaf/aos.rs`) stores `key:8 + doc:DOC_BYTES` per entry; the doc is
written by `doc_le_bytes` (`cow_btree/mod.rs:65-74`, `assert!` that the dropped high bytes are zero)
and read back by `read_width(.., DOC_BYTES)` (`aos.rs:31-36`, `mod.rs:88-102`). `Leaf::doc_layout`
reports `(FIELD, FIELD + DOC_BYTES, DOC_BYTES)` (`leaf/mod.rs:213-219`) and `from_pairs` sizes AoS as
`count * (FIELD + DOC_BYTES)` (`leaf/mod.rs:287`).

| Lean | Rust |
| --- | --- |
| `docLeBytes` | `doc_le_bytes` `cow_btree/mod.rs:65` |
| `aosEncD`, `aosBuildD` | `AosLeaf::build` `aos.rs:39` (`None` = the `doc_le_bytes` panic) |
| `aosCountD`, `aosKeyD`, `aosDocD` | `AosLeaf::count` 18, `key` 23, `doc` 31 |
| `aosMergeD` | `AosLeaf::merge_batch` 51 |
| `docLayoutD` | `Leaf::doc_layout` AoS arm `leaf/mod.rs:215` |
| `fromPairsD`, `toPairsD` | `Leaf::from_pairs` `leaf/mod.rs:260` with `aos_size = count * (FIELD + DOC_BYTES)` |

Proven: `docLeBytes_spec` (the assert passes iff `doc < 256^DOC_BYTES`, and then the bytes are
`le DOC_BYTES doc`), `docLeBytes_lossless` (narrowing is lossless), `aos_roundtripD` and
`fromPairsD_roundtrip` for every `DOC_BYTES ∈ {1, 2, 4, 8}`, `aosMergeD_spec`, `docLayoutD_spec`,
`aosBuildD_length` (`count * (8 + DOC_BYTES)` bytes — the `heap_bytes` leaf term), and that
`DOC_BYTES = 8` is the old 16-byte model (`aosD8_*`).

BUG (latent, new in #2278): the const assert admits `DOC_BYTES ∈ 1..=8`, but `read_width` only knows
widths 1/2/4/8 and reads 8 bytes for anything else. With `DOC_BYTES = 3` (5, 6, 7) every doc read is
wrong: `docBytes3_reads_garbage` (the first entry's doc reads 8 bytes into the next key), and the
last entry's read runs past the page (`docBytes3_last_oob`: Rust panics "range end index 16 out of
range for slice of length 11" on the second `insert` of a `CowBTree::<256, 256, 3>`; debug builds trip
`debug_assert_eq!(width, 8)` first). Fix: assert `DOC_BYTES ∈ {1, 2, 4, 8}`.
-/
namespace IndexLayer.Leaf

/-! ## `doc_le_bytes` -/

/-- `doc_le_bytes::<D>(doc)` (`cow_btree/mod.rs:65-74`): `none` is the `assert!` panic. -/
def docLeBytes (D d : Nat) : Option (List Nat) :=
  if ((le 8 d).drop D).all (· == 0) then some ((le 8 d).take D) else none

theorem le_drop : ∀ (a w x : Nat), a ≤ w → (le w x).drop a = le (w - a) (x / 256 ^ a)
  | 0, _, _, _ => by simp
  | a + 1, w + 1, x, h => by
    simp only [le, List.drop_succ_cons]
    rw [le_drop a w (x / 256) (by omega), Nat.div_div_eq_div_mul, Nat.pow_succ, Nat.mul_comm (256 ^ a) 256]
    simp
  | _ + 1, 0, _, h => by omega

theorem le_all_zero : ∀ (w x : Nat), (le w x).all (· == 0) = true ↔ x % 256 ^ w = 0
  | 0, x => by simp [le, Nat.mod_one]
  | w + 1, x => by
    simp only [le, List.all_cons, Bool.and_eq_true, beq_iff_eq]
    rw [le_all_zero w (x / 256), Nat.pow_succ, Nat.mul_comm (256 ^ w) 256, Nat.mod_mul]
    omega

/-- **`doc_le_bytes`**: for a `u64` doc and `DOC_BYTES <= 8`, the assert passes exactly when the doc fits
    in `DOC_BYTES` bytes, and the bytes written are its low `DOC_BYTES` little-endian bytes. -/
theorem docLeBytes_spec (D d : Nat) (hD : D ≤ 8) (hd : d < 2 ^ 64) :
    docLeBytes D d = if d < 256 ^ D then some (le D d) else none := by
  unfold docLeBytes
  rw [le8_take D d hD, le_drop D 8 d hD]
  have hq : d / 256 ^ D < 256 ^ (8 - D) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.pow_pos (by omega)), ← Nat.pow_add, Nat.sub_add_cancel hD]
    simpa using hd
  by_cases hfit : d < 256 ^ D
  · have : d / 256 ^ D = 0 := Nat.div_eq_of_lt hfit
    have hz : (le (8 - D) (d / 256 ^ D)).all (· == 0) = true := by
      rw [le_all_zero, this]; simp
    simp [hz, hfit]
  · have hpos : 0 < d / 256 ^ D := Nat.div_pos (Nat.le_of_not_lt hfit) (Nat.pow_pos (by omega))
    have hz : ¬ (le (8 - D) (d / 256 ^ D)).all (· == 0) = true := by
      rw [le_all_zero, Nat.mod_eq_of_lt hq]; omega
    simp [hz, hfit]

/-- **Narrowing is lossless**: whatever `doc_le_bytes` writes reads back as the doc. -/
theorem docLeBytes_lossless (D d : Nat) (hD : D ≤ 8) (hd : d < 2 ^ 64) (bs : List Nat)
    (h : docLeBytes D d = some bs) : bs.length = D ∧ unle bs = d := by
  rw [docLeBytes_spec D d hD hd] at h
  split at h
  · next hfit => cases h; exact ⟨le_length D d, unle_le_of_lt D d hfit⟩
  · cases h

/-! ## `AosLeaf<D>` -/

def aosEncD (D : Nat) (p : P) : List Nat := le 8 p.1 ++ le D p.2
theorem aosEncD_len (D : Nat) (p : P) : (aosEncD D p).length = 8 + D := by simp [aosEncD, le_length]

/-- The docs fit the configured width (the precondition of every `doc_le_bytes`). -/
def DocsFit (D : Nat) (ps : List P) : Prop := ∀ p ∈ ps, p.2 < 256 ^ D

/-- `AosLeaf::<D>::build` (`aos.rs:39-46`): `none` is the `doc_le_bytes` panic. -/
def aosBuildD (D : Nat) (ps : List P) : Option (List Nat) :=
  (ps.mapM (fun p => (docLeBytes D p.2).map (fun db => le 8 p.1 ++ db))).map List.flatten

theorem mapM_some {α β} (f : α → Option β) (g : α → β) : ∀ (l : List α), (∀ a ∈ l, f a = some (g a)) →
    l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a List.mem_cons_self, mapM_some f g l (fun b hb => h b (List.mem_cons_of_mem _ hb))]
    rfl

theorem mapM_none {α β} (f : α → Option β) : ∀ (l : List α) (a : α), a ∈ l → f a = none → l.mapM f = none
  | [], _, h, _ => by simp at h
  | b :: l, a, h, hn => by
    rw [List.mapM_cons]
    rcases List.mem_cons.1 h with rfl | h
    · rw [hn]; rfl
    · cases hb : f b with
      | none => rfl
      | some y => simp [mapM_none f l a h hn]

/-- With fitting docs the build succeeds, and the page is the concatenation of `key:8 ++ doc:D`. -/
theorem aosBuildD_some (D : Nat) (hD : D ≤ 8) (ps : List P) (hu : AllU64 ps) (hf : DocsFit D ps) :
    aosBuildD D ps = some (ps.flatMap (aosEncD D)) := by
  unfold aosBuildD
  rw [mapM_some _ (fun p => aosEncD D p) ps (fun p hp => by
    rw [docLeBytes_spec D p.2 hD (hu p hp).2, if_pos (hf p hp)]; rfl)]
  simp [List.flatMap]

/-- A doc that does not fit panics (Rust: the `doc_le_bytes` assert). -/
theorem aosBuildD_none (D : Nat) (hD : D ≤ 8) (ps : List P) (p : P) (hp : p ∈ ps) (hu : U64 p.2)
    (hn : ¬ p.2 < 256 ^ D) : aosBuildD D ps = none := by
  unfold aosBuildD
  rw [mapM_none _ ps p hp (by rw [docLeBytes_spec D p.2 hD hu, if_neg hn]; rfl)]
  rfl

/-- The page holds `count * (8 + DOC_BYTES)` bytes (the `heap_bytes` leaf term, `proofs/cow_btree`
    `heapBytes_aos`): 12 B/entry at `DOC_BYTES = 4`, 16 at 8. -/
theorem aosBuildD_length (D : Nat) (ps : List P) : (ps.flatMap (aosEncD D)).length = ps.length * (8 + D) :=
  length_flatMap_const _ _ (aosEncD_len D) ps

/-- `AosLeaf::count` (`aos.rs:18`): `len / STRIDE`, `STRIDE = FIELD + DOC_BYTES`. -/
def aosCountD (D : Nat) (b : List Nat) : Nat := b.length / (8 + D)
/-- `AosLeaf::key` (`aos.rs:23`). -/
def aosKeyD (D : Nat) (b : List Nat) (i : Nat) : Nat := readU64 b ((8 + D) * i)
/-- `AosLeaf::doc` (`aos.rs:31`): `read_width(.., STRIDE * i + FIELD, DOC_BYTES)`. -/
def aosDocD (D : Nat) (b : List Nat) (i : Nat) : Nat := readWidth b ((8 + D) * i + 8) D
def aosPairsD (D : Nat) (b : List Nat) : List P := pairsOf (aosCountD D b) (aosKeyD D b) (aosDocD D b)

/-- **`AosLeaf<D>` round trip, for every `DOC_BYTES ∈ {1, 2, 4, 8}`.** -/
theorem aos_roundtripD (D : Nat) (hD : W D) (ps : List P) (hu : AllU64 ps) (hf : DocsFit D ps) :
    aosCountD D (ps.flatMap (aosEncD D)) = ps.length ∧
    (∀ i (hi : i < ps.length), aosKeyD D (ps.flatMap (aosEncD D)) i = ps[i].1 ∧
      aosDocD D (ps.flatMap (aosEncD D)) i = ps[i].2) ∧
    aosPairsD D (ps.flatMap (aosEncD D)) = ps := by
  have hD8 : D ≤ 8 := by rcases hD with rfl | rfl | rfl | rfl <;> omega
  have hc : aosCountD D (ps.flatMap (aosEncD D)) = ps.length := by
    simp only [aosCountD, aosBuildD_length]; exact Nat.mul_div_cancel _ (by omega)
  have hkd : ∀ i (hi : i < ps.length), aosKeyD D (ps.flatMap (aosEncD D)) i = ps[i].1 ∧
      aosDocD D (ps.flatMap (aosEncD D)) i = ps[i].2 := by
    intro i hi
    have hp := hu ps[i] (List.getElem_mem hi)
    have hdf := hf ps[i] (List.getElem_mem hi)
    have e : ps.flatMap (aosEncD D) = [] ++ ps.flatMap (aosEncD D) ++ [] := by simp
    constructor
    · unfold aosKeyD readU64
      rw [e, rd_in_chunk (aosEncD D) (8 + D) (aosEncD_len D) [] [] ps i hi 0 8 (by omega) _
        (by simp; rw [Nat.mul_comm])]
      simp only [aosEncD, List.drop_zero]
      rw [List.take_left' (le_length 8 _), unle_le_of_lt 8 _ (u64_256 _ hp.1)]
    · unfold aosDocD
      rw [readWidth_eq _ _ _ hD, e, rd_in_chunk (aosEncD D) (8 + D) (aosEncD_len D) [] [] ps i hi 8 D
        (by omega) _ (by simp; rw [Nat.mul_comm])]
      simp only [aosEncD]
      rw [List.drop_left' (le_length 8 _), List.take_of_length_le (by rw [le_length]; exact Nat.le_refl _),
        unle_le_of_lt D _ hdf]
  refine ⟨hc, hkd, ?_⟩
  unfold aosPairsD; rw [hc]; exact pairsOf_eq ps _ _ hkd

/-- `AosLeaf::merge_batch` (`aos.rs:51-66`): the `merge_walk` emissions, each re-encoded through
    `doc_le_bytes` (`none` = its panic). -/
def aosMergeD (D : Nat) (b : List Nat) (batch : List P) : Option (List Nat) :=
  aosBuildD D ((mergeW lexLe (aosPairsD D b) batch none).map (·.1))

/-- **Merging into a built `AosLeaf<D>` yields the page of the merged entry list.** -/
theorem aosMergeD_spec (D : Nat) (hD : W D) (ps batch : List P) (hu : AllU64 ps) (hf : DocsFit D ps) :
    aosMergeD D (ps.flatMap (aosEncD D)) batch = aosBuildD D (mergeD lexLe ps batch none) := by
  unfold aosMergeD
  rw [(aos_roundtripD D hD ps hu hf).2.2, mergeW_fst lexLe ps batch none]

/-- `Leaf::doc_layout`'s AoS arm (`leaf/mod.rs:215`): `(FIELD, FIELD + DOC_BYTES, DOC_BYTES)`. -/
def docLayoutD (D : Nat) : Nat × Nat × Nat := (8, 8 + D, D)

/-- The cursor's cached layout reads exactly `AosLeaf::doc` (`cursor.rs:197-200`). -/
theorem docLayoutD_spec (D : Nat) (b : List Nat) (i : Nat) :
    aosDocD D b i = readWidth b ((docLayoutD D).1 + i * (docLayoutD D).2.1) (docLayoutD D).2.2 := by
  simp only [aosDocD, docLayoutD]; congr 1; rw [Nat.mul_comm]; omega

/-! ## `DOC_BYTES = 8` is the 16-byte model of `LeafAos.lean` -/

theorem aosD8_enc (p : P) : aosEncD 8 p = aosEnc p := rfl
theorem aosD8_count (b : List Nat) : aosCountD 8 b = aosCount b := rfl
theorem aosD8_key (b : List Nat) (i : Nat) : aosKeyD 8 b i = aosKey b i := by
  simp only [aosKeyD, aosKey, aosRead]; rw [show (8 + 8) * i = 16 * i + 0 by omega]
theorem aosD8_doc (b : List Nat) (i : Nat) : aosDocD 8 b i = aosDoc b i := by
  simp only [aosDocD, aosDoc, aosRead, readWidth, readU64]
theorem aosD8_pairs (b : List Nat) : aosPairsD 8 b = aosPairs b := by
  unfold aosPairsD aosPairs; rw [aosD8_count, funext (aosD8_key b), funext (aosD8_doc b)]
theorem aosD8_build (ps : List P) (hu : AllU64 ps) : aosBuildD 8 ps = some (aosBuild ps) := by
  rw [aosBuildD_some 8 (by omega) ps hu (fun p hp => u64_256 _ (hu p hp).2)]; rfl
theorem aosD8_layout : docLayoutD 8 = (8, 16, 8) := rfl

/-! ## `from_pairs` with the AoS size `count * (8 + DOC_BYTES)` -/

/-- `Leaf::<_, D>::from_pairs` (`leaf/mod.rs:260-312`); `none` is a `doc_le_bytes` panic in the AoS arm. -/
def fromPairsD (D : Nat) (ps : List P) : Option LeafV :=
  match ps.getLast? , ps.head? with
  | some last, some first =>
    let count := ps.length
    let (distinct, md0) := scan ps 1 0
    let maxDoc := max md0 last.2
    let minValue := first.1
    let vw := pow2BytesFor (last.1 - minValue)
    let dw := pow2BytesFor maxDoc
    let dedup := decide (distinct < count)
    let compactSize := 14 + distinct * vw + (if dedup then count else 0) + count * dw
    let aosSize := count * (8 + D)                                       -- line 287
    if compactSize + 8 * count ≤ aosSize then
      if dedup then some (.indexed (ciBuild ps minValue vw dw)) else some (.compact (cBuild ps minValue vw dw))
    else (aosBuildD D ps).map .aos
  | _, _ => (aosBuildD D ps).map .aos

/-- Decoding a page of a `D`-wide tree (`Leaf::iter` / `to_pairs`). -/
def toPairsD (D : Nat) : LeafV → List P
  | .aos b => aosPairsD D b
  | l => l.toPairs

/-- `DOC_BYTES = 8` is the existing `from_pairs` model. -/
theorem fromPairsD8 (ps : List P) (hu : AllU64 ps) : fromPairsD 8 ps = some (fromPairs ps) := by
  unfold fromPairsD fromPairs
  have h16 : (8 : Nat) + 8 = 16 := rfl
  cases hl : ps.getLast? <;> cases hh : ps.head? <;>
    simp only [aosD8_build ps hu, Option.map_some, h16] <;>
    (split <;> (try split) <;> rfl)

/-- **`from_pairs` round trip for every `DOC_BYTES ∈ {1, 2, 4, 8}`**: on a strictly sorted list of
    `u64` entries whose docs fit, it never panics and the page it picks reads back as the input. -/
theorem fromPairsD_roundtrip (D : Nat) (hD : W D) (ps : List P) (hs : Sorted2 ps) (hu : AllU64 ps)
    (hf : DocsFit D ps) (hn : ps.length ≤ 256) :
    ∃ l, fromPairsD D ps = some l ∧ toPairsD D l = ps := by
  have hD8 : D ≤ 8 := by rcases hD with rfl | rfl | rfl | rfl <;> omega
  have haos : (aosBuildD D ps).map LeafV.aos = some (.aos (ps.flatMap (aosEncD D))) := by
    rw [aosBuildD_some D hD8 ps hu hf]; rfl
  have haosr : toPairsD D (.aos (ps.flatMap (aosEncD D))) = ps := (aos_roundtripD D hD ps hu hf).2.2
  unfold fromPairsD
  split
  · next last first hl hf' =>
    obtain ⟨hdoc, -⟩ := scan_doc ps 1 0 hs
    simp only [hl, Option.map_some, Option.getD_some] at hdoc
    have hb := sorted_bounds ps hs first last hf' hl
    have hlastin : last ∈ ps := List.mem_of_getLast? hl
    have hfirstin : first ∈ ps := List.mem_of_mem_head? hf'
    have hvw := pow2BytesFor_spec (last.1 - first.1) (by have := (hu last hlastin).1; unfold U64 at this; omega)
    have hmd : max (scan ps 1 0).2 last.2 < 2 ^ 64 := by
      have h1 := (hu last hlastin).2
      have h2 := scan_bound (2 ^ 64) ps 1 0 (fun p hp => (hu p hp).2) (by decide)
      unfold U64 at h1; omega
    have hdw := pow2BytesFor_spec _ hmd
    have hfit : CFits first.1 (pow2BytesFor (last.1 - first.1)) (pow2BytesFor (max (scan ps 1 0).2 last.2)) ps := by
      intro p hp
      obtain ⟨b1, b2⟩ := hb p hp
      refine ⟨b1, by have := hvw.2; omega, by have := hdoc p hp; have := hdw.2; omega⟩
    have hmin : U64 first.1 := (hu first hfirstin).1
    generalize hsc : scan ps 1 0 = sc at hfit hdw
    obtain ⟨dc, md⟩ := sc
    simp only at hfit hdw ⊢
    repeat' split
    all_goals first
      | exact ⟨_, rfl, by simp only [toPairsD]; rw [(dispatch_spec _).2.2.1]; exact ciBuild_roundtrip ps _ _ _ hn hmin hvw.1 hdw.1 hfit⟩
      | exact ⟨_, rfl, by simp only [toPairsD]; rw [(dispatch_spec _).2.1]; exact cBuild_roundtrip ps _ _ _ (by omega) hmin hvw.1 hdw.1 hfit⟩
      | exact ⟨_, haos, haosr⟩
  · exact ⟨_, haos, haosr⟩

/-! ## BUG: `DOC_BYTES ∉ {1, 2, 4, 8}` passes the const assert but cannot be read -/

/-- With `DOC_BYTES = 3` the stride is 11, but `read_width(.., 3)` reads 8 bytes (`mod.rs:98-101`):
    the first entry's doc comes back as 5 bytes of the next key glued on. -/
theorem docBytes3_reads_garbage :
    aosBuildD 3 [(1, 5), (2, 6)] = some ([(1, 5), (2, 6)].flatMap (aosEncD 3)) ∧
    aosDocD 3 ([(1, 5), (2, 6)].flatMap (aosEncD 3)) 0 ≠ 5 := by
  refine ⟨aosBuildD_some 3 (by omega) _ (by
    intro p hp; simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> exact ⟨by unfold U64; decide, by unfold U64; decide⟩) (by
    intro p hp; simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> decide), ?_⟩
  decide

/-- ...and the last entry's 8-byte read ends past the page: the Rust slice `b[off..off + 8]` panics. -/
theorem docBytes3_last_oob :
    ([(1, 5)].flatMap (aosEncD 3)).length < (8 + 3) * 0 + 8 + 8 := by decide

end IndexLayer.Leaf
