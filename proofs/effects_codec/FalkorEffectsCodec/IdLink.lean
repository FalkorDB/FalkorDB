import FalkorEffectsCodec.Payload
import FalkorIdList.Wire
/-!
# Linking the abstract `IdCodec` to the real `IdList` model

`Records.lean` states every record theorem for an abstract `IdCodec` (the ids block: `IdList`
encoder + `read_ids`). Its roundtrip used to be a hypothesis "owned by the IdList proof", which
made `Payload::records`/`Records::next` CONDITIONAL. This file discharges it: `idListCodec ρ hρ`
is an `IdCodec` built from `proofs/id_list`'s byte-exact model (`IdListWire.encodeIds` /
`readIds`, imported through Lake), with

| here | there |
| --- | --- |
| `IdT`        | an `IdList` the encoder accepts: segments well-formed, `sum len < 2^32` (the `u32` record count), at most one segment per id |
| `count`      | `IdList::count` = the total number of ids = the record count that `read_ids` is given |
| `idEnc`      | `impl EffectEncode<3> for IdList` (`id_list.rs`), bytes as `u8` |
| `idDec n`    | `read_ids(r, n: u32)` (`id_list.rs`); an `n ≥ 2^32` is not a `u32`, so it is refused |

The only assumption left is `RoaringOK ρ` of `id_list` (the `roaring` crate's serialize /
deserialize pair, an external library exactly as in `proofs/id_list`); the other two
`IdCodec` fields are proved: `idListCodec_roundtrip` is `IdListWire.readIds_encodeIds` through the
`u8`/`Nat` byte conversion, `idListCodec_good` is `readIds_never_panics` + `readIds_safe`
(`read_ids` reads a prefix of its input and never panics).
-/
namespace FalkorCodec

open IdListPush (Seg)

/-- `Bytes` (`List UInt8`) as the `Nat` byte lists of `id_list`. -/
def natBytes (b : Bytes) : List Nat := b.map UInt8.toNat

/-- The `Nat` byte lists of `id_list` as `Bytes`. -/
def u8Bytes (b : List Nat) : Bytes := b.map UInt8.ofNat

theorem natBytes_lt (b : Bytes) : ∀ x ∈ natBytes b, x < 256 := by
  intro x hx
  simp only [natBytes, List.mem_map] at hx
  obtain ⟨u, _, rfl⟩ := hx
  exact u.toNat_lt

theorem u8Bytes_natBytes (b : Bytes) : u8Bytes (natBytes b) = b := by
  simp [u8Bytes, natBytes, List.map_map, Function.comp_def]

theorem natBytes_u8Bytes (l : List Nat) (h : ∀ x ∈ l, x < 256) : natBytes (u8Bytes l) = l := by
  induction l with
  | nil => simp [natBytes, u8Bytes]
  | cons a t ih =>
    have ha : a < 256 := h a (by simp)
    have ht := ih (fun x hx => h x (by simp [hx]))
    simp only [natBytes, u8Bytes, List.map_cons] at ht ⊢
    rw [ht]
    simp [UInt8.toNat_ofNat, Nat.mod_eq_of_lt ha]

theorem natBytes_append (a b : Bytes) : natBytes (a ++ b) = natBytes a ++ natBytes b := by
  simp [natBytes]

theorem u8Bytes_append (a b : List Nat) : u8Bytes (a ++ b) = u8Bytes a ++ u8Bytes b := by
  simp [u8Bytes]

/-! ## Every byte the `IdList` encoder writes is a byte -/

theorem encodeSeg_bytes (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) (s : Seg)
    (bs : List Nat) (h : IdListWire.encodeSeg ρ s = .ok bs) : ∀ b ∈ bs, b < 256 := by
  have hpair : ∀ k ∈ [0, 0x40, 2], ∀ v c : Nat,
      ∀ b ∈ IdListWire.writePair k v c, b < 256 := by
    intro k hk v c b hb
    obtain ⟨_, _, _, _, _, hlt⟩ := IdListWire.pairHeader_fields k hk
      (IdListWire.widthCode (IdListWire.widthFor v)) (IdListWire.widthCode_lt v)
      (IdListWire.widthCode (IdListWire.widthFor c)) (IdListWire.widthCode_lt c)
    simp only [IdListWire.writePair, List.mem_cons, List.mem_append] at hb
    rcases hb with rfl | hb | hb
    · exact hlt
    · exact IdListWire.le_bytes _ _ b hb
    · exact IdListWire.le_bytes _ _ b hb
  have hbm : ∀ (d : Nat), (d = 0 ∨ d = 64) → ∀ bm bs',
      IdListWire.encodeSeg.bitmap ρ d bm = .ok bs' → ∀ b ∈ bs', b < 256 := by
    intro d hd bm bs' hb b hbm
    unfold IdListWire.encodeSeg.bitmap at hb
    by_cases hl : (ρ.ser bm).length ≠ ρ.size bm
    · simp [hl] at hb
    · simp only [hl, if_false] at hb
      cases hb
      simp only [List.mem_cons, List.mem_append] at hbm
      rcases hbm with rfl | hbm | hbm
      · rcases hd with rfl | rfl <;> decide
      · exact IdListWire.le_bytes _ _ b hbm
      · exact hρ.bytes _ b hbm
  cases s with
  | range b l =>
    simp only [IdListWire.encodeSeg] at h; cases h
    exact hpair _ (by simp [IdListWire.SEG_KIND_RANGE]) _ _
  | rdesc b l =>
    simp only [IdListWire.encodeSeg] at h; cases h
    exact hpair _ (by simp [IdListWire.SEG_KIND_RANGE, IdListWire.SEG_DESCENDING]) _ _
  | rep i c =>
    simp only [IdListWire.encodeSeg] at h; cases h
    exact hpair _ (by simp [IdListWire.SEG_KIND_REPEAT]) _ _
  | asc bm =>
    simp only [IdListWire.encodeSeg] at h
    exact hbm 0 (Or.inl rfl) bm bs h
  | dsc bm =>
    simp only [IdListWire.encodeSeg] at h
    exact hbm _ (Or.inr (by decide)) bm bs h

theorem encodeSegs_bytes (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) :
    ∀ (segs : List Seg) (bs : List Nat), IdListWire.encodeSegs ρ segs = .ok bs →
      ∀ b ∈ bs, b < 256
  | [], bs, h => by simp [IdListWire.encodeSegs] at h; subst h; simp
  | s :: t, bs, h => by
    simp only [IdListWire.encodeSegs] at h
    rcases ha : IdListWire.encodeSeg ρ s with e | a
    · simp [ha, bind, Except.bind] at h
    · rcases hc : IdListWire.encodeSegs ρ t with e | c
      · simp [ha, hc, bind, Except.bind] at h
      · simp [ha, hc, bind, Except.bind, pure, Except.pure] at h
        subst h
        intro b hb
        simp only [List.mem_append] at hb
        rcases hb with hb | hb
        · exact encodeSeg_bytes ρ hρ s a ha b hb
        · exact encodeSegs_bytes ρ hρ t c hc b hb

theorem encodeIds_bytes (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) (segs : List Seg)
    (bs : List Nat) (h : IdListWire.encodeIds ρ segs = .ok bs) : ∀ b ∈ bs, b < 256 := by
  unfold IdListWire.encodeIds at h
  rcases hbody : IdListWire.encodeSegs ρ segs with e | body
  · simp [hbody, bind, Except.bind] at h
  · simp [hbody, bind, Except.bind, pure, Except.pure] at h
    subst h
    intro b hb
    simp only [List.mem_append] at hb
    rcases hb with hb | hb
    · exact IdListWire.le_bytes _ _ b hb
    · exact encodeSegs_bytes ρ hρ segs body hbody b hb

/-! ## The codec -/

/-- An `IdList` that the encoder accepts and `read_ids` can return. -/
def IdT : Type :=
  {t : List Seg // (∀ s ∈ t, Seg.WF s) ∧ IdListWire.sumLen t < 4294967296 ∧
    t.length ≤ IdListWire.sumLen t}

/-- `impl EffectEncode<3> for IdList`, bytes as `u8`; an encoder error is `EErr.idList`. -/
def idEnc (ρ : IdListWire.Roaring) (t : IdT) : Except EErr Bytes :=
  match IdListWire.encodeIds ρ t.1 with
  | .ok bs => .ok (u8Bytes bs)
  | .error _ => .error .idList

/-- `read_ids(r, n)`: any error is `DErr.idList`; a count that is not a `u32` is refused. -/
def idDec (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) (n : Nat) : R IdT :=
  fun bs =>
    if hn : n < 4294967296 then
      match h : IdListWire.readIds ρ (natBytes bs) n with
      | .ok (segs, rest) =>
        have hs := IdListWire.readIds_safe ρ hρ (natBytes bs) n (natBytes_lt bs) hn h
        have hf := IdListWire.flat_length hs.1
        .ok (⟨segs, hs.1, by
          have := hs.2.1; have := hs.2.2.1; rw [hf] at *; omega, by
          have := hs.2.1; have := hs.2.2.1; rw [hf] at *; omega⟩, u8Bytes rest)
      | .error _ => .error .idList
    else .error .idList

theorem idDec_roundtrip (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) (t : IdT)
    (bs : Bytes) (rest : Bytes) (h : idEnc ρ t = .ok bs) :
    idDec ρ hρ (IdListWire.sumLen t.1) (bs ++ rest) = .ok (t, rest) := by
  obtain ⟨segs, hw, hc, hn⟩ := t
  simp only [idEnc] at h
  rcases henc : IdListWire.encodeIds ρ segs with e | enc
  · rw [henc] at h; cases h
  rw [henc] at h
  cases h
  obtain ⟨enc', henc', hread⟩ :=
    IdListWire.readIds_encodeIds ρ hρ segs hw (IdListWire.sumLen segs) rfl hc hn (natBytes rest)
  have hbytes := encodeIds_bytes ρ hρ segs enc henc
  rw [henc] at henc'; cases henc'
  have hnat : natBytes (u8Bytes enc ++ rest) = enc ++ natBytes rest := by
    rw [natBytes_append, natBytes_u8Bytes enc hbytes]
  unfold idDec
  simp only [hc, dite_true]
  split
  · rename_i segs' rest' hr
    rw [hnat, hread] at hr
    cases hr
    simp [u8Bytes_natBytes]
  · rename_i e he
    rw [hnat, hread] at he
    cases he

theorem idDec_good (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) (n : Nat) :
    Good (idDec ρ hρ n) := by
  refine ⟨?_, ?_, ?_⟩
  · intro bs a r' h
    unfold idDec at h
    by_cases hn : n < 4294967296
    · simp only [hn, dite_true] at h
      split at h
      · rename_i segs rest hr
        have hs := IdListWire.readIds_safe ρ hρ (natBytes bs) n (natBytes_lt bs) hn hr
        obtain ⟨pre, hpre, _⟩ := hs.2.2.2
        cases h
        refine ⟨u8Bytes pre, ?_⟩
        rw [← u8Bytes_append, ← hpre, u8Bytes_natBytes]
      · cases h
    · simp only [hn, dite_false] at h; cases h
  · intro bs h
    unfold idDec at h
    by_cases hn : n < 4294967296
    · simp only [hn, dite_true] at h
      split at h
      · cases h
      · cases h
    · simp only [hn, dite_false] at h; cases h
  · intro bs h
    unfold idDec at h
    by_cases hn : n < 4294967296
    · simp only [hn, dite_true] at h
      split at h
      · cases h
      · cases h
    · simp only [hn, dite_false] at h; cases h

/-- **The `IdCodec` the record theorems are about, instantiated with the real `IdList` model.** -/
def idListCodec (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ) : IdCodec where
  T := IdT
  count := fun t => IdListWire.sumLen t.1
  enc := idEnc ρ
  dec := idDec ρ hρ
  roundtrip := fun t bs rest h => idDec_roundtrip ρ hρ t bs rest h
  good := idDec_good ρ hρ

/-- `Payload::records` / `Records::next` over the real `IdList` codec: a payload built from
encoded records yields those records, in order (`records_concat` at `idListCodec`). -/
theorem records_concat_idList (ρ : IdListWire.Roaring) (hρ : IdListWire.RoaringOK ρ)
    (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool)
    (items : List (Bytes × Record (idListCodec ρ hρ).T))
    (h : ∀ p ∈ items, p.1 ≠ [] ∧ encRecord (idListCodec ρ hρ) utf8 lg I p.2 = .ok p.1 ∧
      RecOk (idListCodec ρ hρ) utf8 lg I p.2)
    (fuel : Nat) (hf : items.length < fuel) :
    readAllRecords (idListCodec ρ hρ) utf8 lg I fuel (items.map Prod.fst).flatten =
      .ok (items.map Prod.snd, []) :=
  records_concat (idListCodec ρ hρ) utf8 lg I items h fuel hf

end FalkorCodec
