import FalkorEffectsCodec.RecRT
/-!
# Payload framing: `open_payload`, `maybe_compress`, `Payload`, `Records`

`graph/src/effects/v3/records.rs:1211-1399`. zstd and crc32 are external C/
Rust libraries; what the framing needs from them is a hypothesis structure
(`Codec`), not an axiom: zstd's frame round-trips, crc32 is a 32-bit function.
-/
namespace FalkorCodec

def EFFECTS_VERSION : Nat := 3
def FLAG_COMPRESSED : Nat := 1
def KNOWN_FLAGS : Nat := FLAG_COMPRESSED
def COMPRESSED_PREFIX : Nat := 12

/-- zstd (`encode_all`, `bulk::decompress`) and `crc32fast::hash`. -/
structure Codec where
  compress : Bytes → Option Bytes
  decompress : Bytes → Nat → Option Bytes
  crc : Bytes → Nat

/-- What the framing relies on (zstd's documented contract; crc32 is `u32`). -/
structure CodecOK (Z : Codec) : Prop where
  roundtrip : ∀ b f, Z.compress b = some f → Z.decompress f b.length = some b
  crc_lt : ∀ b, Z.crc b < 2 ^ 32

/-- `open_payload` (`records.rs:1261`): the plain record bytes, or why not. -/
def openPayload (Z : Codec) : R Bytes := do
  let version ← u8
  if version ≠ EFFECTS_VERSION then fail (.unsupportedVersion version) else
  let flags ← u8
  if flags &&& (255 - KNOWN_FLAGS) ≠ 0 then fail (.unknownFlags flags) else
  if flags &&& FLAG_COMPRESSED = 0 then fun r => .ok (r, []) else
  let plainLen ← u32
  let compLen ← u32
  let checksum ← u32
  let frame ← take compLen
  fun r =>
    if r ≠ [] then .error (.trailingBytes r.length) else
    match Z.decompress frame plainLen with
    | none => .error .badCompression
    | some plain =>
      if plain.length ≠ plainLen then .error (.compressedLengthMismatch plainLen plain.length)
      else if Z.crc plain ≠ checksum then .error (.checksumMismatch checksum (Z.crc plain))
      else .ok (plain, [])

/-- `maybe_compress` (`records.rs:1361`): the new buffer and whether it compressed. -/
def maybeCompress (Z : Codec) (buf : Bytes) (minBytes : Nat) : Bytes × Bool :=
  if minBytes = 0 ∨ buf.length < 2 ∨ buf.length - 2 < minBytes then (buf, false) else
  if (buf[1]!.toNat &&& FLAG_COMPRESSED) ≠ 0 then (buf, false) else
  match Z.compress (buf.drop 2) with
  | none => (buf, false)
  | some frame =>
    if frame.length + COMPRESSED_PREFIX ≥ buf.length - 2 then (buf, false) else
    let plainLen := (buf.length - 2) % 2 ^ 32           -- `as u32`
    let compLen := frame.length % 2 ^ 32
    ([buf[0]!, buf[1]! ||| 1] ++ w32 plainLen ++ w32 compLen ++ w32 (Z.crc (buf.drop 2)) ++ frame, true)

/-- `new_buffer` (`v3/mod.rs`): `[EFFECTS_VERSION, 0]`. -/
def newBuffer : Bytes := [3, 0]

theorem u8_cons (b : UInt8) (r : Bytes) : u8 (b :: r) = .ok (b.toNat, r) := by
  simp [u8, rdU, FalkorCodec.take, unle]

/-- An uncompressed payload opens to its record bytes. -/
theorem openPayload_plain (Z : Codec) (body : Bytes) :
    openPayload Z (newBuffer ++ body) = .ok (body, []) := by
  simp [openPayload, newBuffer, bind_apply, u8_cons, EFFECTS_VERSION, KNOWN_FLAGS, FLAG_COMPRESSED]

/-- **Compression is transparent**: sealing a fresh buffer — compressed or
left alone — and opening it gives back exactly the record bytes, provided the
records are under 4 GiB (`plain_len` is written `as u32`). -/
theorem openPayload_maybeCompress (Z : Codec) (hZ : CodecOK Z) (body : Bytes) (m : Nat)
    (hlen : body.length < 2 ^ 32) :
    openPayload Z (maybeCompress Z (newBuffer ++ body) m).1 = .ok (body, []) := by
  have hl : (newBuffer ++ body).length - 2 = body.length := by simp [newBuffer]
  unfold maybeCompress
  by_cases h1 : m = 0 ∨ (newBuffer ++ body).length < 2 ∨ (newBuffer ++ body).length - 2 < m
  · simp only [h1, ite_true]; exact openPayload_plain Z body
  simp only [h1, ite_false]
  have hflag : ((newBuffer ++ body)[1]!.toNat &&& FLAG_COMPRESSED) = 0 := by
    simp [newBuffer, FLAG_COMPRESSED]
  simp only [hflag, ne_eq, not_true_eq_false, ite_false]
  have hd : (newBuffer ++ body).drop 2 = body := by simp [newBuffer]
  rw [hd]
  cases hc : Z.compress body with
  | none => exact openPayload_plain Z body
  | some frame =>
    simp only
    by_cases h2 : frame.length + COMPRESSED_PREFIX ≥ (newBuffer ++ body).length - 2
    · simp only [h2, ite_true]; exact openPayload_plain Z body
    simp only [h2, ite_false]
    rw [hl] at h2 ⊢
    rw [Nat.mod_eq_of_lt hlen]
    have hfl : frame.length < 2 ^ 32 := by simp [COMPRESSED_PREFIX] at h2; omega
    rw [Nat.mod_eq_of_lt hfl]
    have h0 : (newBuffer ++ body)[0]! = 3 := by simp [newBuffer]
    have h1' : (newBuffer ++ body)[1]! = 0 := by simp [newBuffer]
    rw [h0, h1']
    have e2 : ((0 : UInt8) ||| 1) = 1 := rfl
    rw [e2]
    have hfr : take frame.length frame = .ok (frame, []) := by simpa using take_append frame []
    simp [openPayload, u8_cons, EFFECTS_VERSION, KNOWN_FLAGS, FLAG_COMPRESSED, u32_w32 hlen,
      u32_w32 hfl, u32_w32 (hZ.crc_lt body), hfr, hZ.roundtrip body frame hc]

/-- Nothing is compressed twice, and compression only happens when it saves
more than the 12-byte prefix. -/
theorem maybeCompress_shrinks (Z : Codec) (buf : Bytes) (m : Nat) (h : (maybeCompress Z buf m).2 = true) :
    (maybeCompress Z buf m).1.length < buf.length := by
  unfold maybeCompress at h ⊢
  by_cases a : m = 0 ∨ buf.length < 2 ∨ buf.length - 2 < m
  · simp only [a, ite_true] at h; cases h
  simp only [a, ite_false] at h ⊢
  by_cases b : (buf[1]!.toNat &&& FLAG_COMPRESSED) ≠ 0
  · rw [if_pos b] at h; cases h
  rw [if_neg b] at h ⊢
  cases hc : Z.compress (buf.drop 2) with
  | none => simp only [hc] at h; cases h
  | some frame =>
    simp only [hc] at h ⊢
    by_cases d : frame.length + COMPRESSED_PREFIX ≥ buf.length - 2
    · simp only [d, ite_true] at h; cases h
    simp only [d, ite_false]; simp [w32, COMPRESSED_PREFIX] at d ⊢; omega

/-- `Records::next` (`records.rs:1236`): stop at the end, yield one record,
fuse after the first error. -/
def readAllRecords (C : IdCodec) (utf8 I : Bytes → Bool) : Nat → R (List (Record C.T))
  | 0 => fun _ => .error .outOfFuel
  | fuel + 1 => fun r =>
    if r = [] then .ok ([], [])
    else match readRecord C utf8 I r with
      | .error e => .error e
      | .ok (x, r') => match readAllRecords C utf8 I fuel r' with
        | .error e => .error e
        | .ok (xs, r'') => .ok (x :: xs, r'')

/-- `Payload::deref`: the plain bytes; `Payload::records`: a fresh `Records`
over them. A payload built from encoded records yields those records, in order. -/
theorem records_concat (C : IdCodec) (utf8 I : Bytes → Bool) :
    ∀ (items : List (Bytes × Record C.T)),
    (∀ p ∈ items, p.1 ≠ [] ∧ encRecord C utf8 I p.2 = .ok p.1 ∧ RecOk C utf8 I p.2) →
    ∀ fuel, items.length < fuel →
    readAllRecords C utf8 I fuel (items.map Prod.fst).flatten = .ok (items.map Prod.snd, [])
  | [], _, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    simp [readAllRecords]
  | (b, x) :: items, h, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have hb := h (b, x) (by simp)
    have ih := records_concat C utf8 I items (fun p hp => h p (by simp [hp])) f (by simp at hf; omega)
    simp only [List.map_cons, List.flatten_cons, readAllRecords]
    rw [if_neg (by simp [hb.1])]
    rw [readRecord_encRecord C utf8 I x b _ hb.2.1 hb.2.2]
    simp only [ih]

end FalkorCodec
