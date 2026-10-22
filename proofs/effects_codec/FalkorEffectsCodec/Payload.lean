import FalkorEffectsCodec.RecRT
/-!
# Payload framing: `open_payload`, `maybe_compress`, `frame_prefix`, `Payload`, `Records`

`graph/src/effects/v3/records.rs:1211-1423`.

**#3278 (`dcbd445d1`, issue #3276).** `maybe_compress` used to write both lengths
`as u32`; a payload of 4 GiB of records or more was framed with a wrapped
`plain_len`, which the reader takes as the decompress ceiling, so the replica
refused it and looped on forced resyncs (`pre3278_wrapped_frame_unreadable`). It
now declines to compress such a payload (`u32::try_from(..).is_err()`) and builds
the prefix through `frame_prefix`, which refuses a length that does not fit
(`framePrefix_none`). `openPayload_maybeCompress` therefore holds for every
payload, with no length hypothesis. zstd and crc32 are external C/
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

/-- `frame_prefix` (`records.rs:1413`): `u32 plain_len · u32 comp_len · u32
checksum`, or `None` when a length does not fit its `u32`. -/
def framePrefix (plainLen compLen checksum : Nat) : Option Bytes :=
  if plainLen < 2 ^ 32 ∧ compLen < 2 ^ 32 then
    some (w32 plainLen ++ w32 compLen ++ w32 checksum)
  else none

theorem framePrefix_some {p c k : Nat} (hp : p < 2 ^ 32) (hc : c < 2 ^ 32) :
    framePrefix p c k = some (w32 p ++ w32 c ++ w32 k) := by
  simp [framePrefix, hp, hc]

/-- **A length that does not fit is refused, not truncated.** -/
theorem framePrefix_none {p c k : Nat} (h : 2 ^ 32 ≤ p ∨ 2 ^ 32 ≤ c) : framePrefix p c k = none := by
  unfold framePrefix; rw [if_neg (by omega)]

/-- The prefix is `COMPRESSED_PREFIX` bytes whenever it exists (`[u8; 12]`). -/
theorem framePrefix_length {p c k : Nat} {b : Bytes} (h : framePrefix p c k = some b) :
    b.length = COMPRESSED_PREFIX := by
  unfold framePrefix at h; split at h
  · cases h; simp [w32, COMPRESSED_PREFIX]
  · cases h

/-- `maybe_compress` (`records.rs:1361`): the new buffer and whether it compressed. -/
def maybeCompress (Z : Codec) (buf : Bytes) (minBytes : Nat) : Bytes × Bool :=
  if minBytes = 0 ∨ buf.length < 2 ∨ buf.length - 2 < minBytes then (buf, false) else
  -- `u32::try_from(buf.len() - HEADER).is_err()` (#3278)
  if 2 ^ 32 ≤ buf.length - 2 then (buf, false) else
  if (buf[1]!.toNat &&& FLAG_COMPRESSED) ≠ 0 then (buf, false) else
  match Z.compress (buf.drop 2) with
  | none => (buf, false)
  | some frame =>
    if frame.length + COMPRESSED_PREFIX ≥ buf.length - 2 then (buf, false) else
    match framePrefix (buf.length - 2) frame.length (Z.crc (buf.drop 2)) with
    | none => (buf, false)
    | some pfx => ([buf[0]!, buf[1]! ||| 1] ++ pfx ++ frame, true)

/-- `maybe_compress` before #3278: both lengths written `as u32`. -/
def pre3278_maybeCompress (Z : Codec) (buf : Bytes) (minBytes : Nat) : Bytes × Bool :=
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

/-- A compressed frame only ever opens to exactly `plain_len` bytes: the reader
checks the decompressed length against the declared one. -/
theorem openPayload_compressed_len (Z : Codec) {p c k : Nat} (hp : p < 2 ^ 32) (hc : c < 2 ^ 32)
    (hk : k < 2 ^ 32) (fr b r : Bytes)
    (h : openPayload Z ([3, 1] ++ w32 p ++ w32 c ++ w32 k ++ fr) = .ok (b, r)) : b.length = p := by
  simp [openPayload, u8_cons, EFFECTS_VERSION, KNOWN_FLAGS, FLAG_COMPRESSED, u32_w32 hp, u32_w32 hc,
    u32_w32 hk] at h
  revert h
  cases take c fr with
  | error e => simp
  | ok q =>
    obtain ⟨frame, r'⟩ := q
    simp only
    split
    · cases Z.decompress frame p with
      | none => simp
      | some plain =>
        simp only
        split
        · split
          · intro h; cases h; assumption
          · simp
        · simp
    · simp

/-- An uncompressed payload opens to its record bytes. -/
theorem openPayload_plain (Z : Codec) (body : Bytes) :
    openPayload Z (newBuffer ++ body) = .ok (body, []) := by
  simp [openPayload, newBuffer, bind_apply, u8_cons, EFFECTS_VERSION, KNOWN_FLAGS, FLAG_COMPRESSED]

/-- **Compression is transparent**: sealing a fresh buffer — compressed or
left alone — and opening it gives back exactly the record bytes, for records of
any length (#3278: a payload too long for the `u32` lengths ships uncompressed). -/
theorem openPayload_maybeCompress (Z : Codec) (hZ : CodecOK Z) (body : Bytes) (m : Nat) :
    openPayload Z (maybeCompress Z (newBuffer ++ body) m).1 = .ok (body, []) := by
  have hl : (newBuffer ++ body).length - 2 = body.length := by simp [newBuffer]
  unfold maybeCompress
  by_cases h1 : m = 0 ∨ (newBuffer ++ body).length < 2 ∨ (newBuffer ++ body).length - 2 < m
  · simp only [h1, ite_true]; exact openPayload_plain Z body
  simp only [h1, ite_false]
  by_cases hbig : 2 ^ 32 ≤ (newBuffer ++ body).length - 2
  · simp only [hbig, ite_true]; exact openPayload_plain Z body
  simp only [hbig, ite_false]
  have hlen : body.length < 2 ^ 32 := by omega
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
    have hfl : frame.length < 2 ^ 32 := by simp [COMPRESSED_PREFIX] at h2; omega
    rw [framePrefix_some hlen hfl]
    simp only
    have h0 : (newBuffer ++ body)[0]! = 3 := by simp [newBuffer]
    have h1' : (newBuffer ++ body)[1]! = 0 := by simp [newBuffer]
    rw [h0, h1']
    have e2 : ((0 : UInt8) ||| 1) = 1 := rfl
    rw [e2]
    have hfr : take frame.length frame = .ok (frame, []) := by simpa using take_append frame []
    simp [openPayload, u8_cons, EFFECTS_VERSION, KNOWN_FLAGS, FLAG_COMPRESSED, u32_w32 hlen,
      u32_w32 hfl, u32_w32 (hZ.crc_lt body), hfr, hZ.roundtrip body frame hc]

/-- **#3278, the bug it fixed.** Before the fix, a payload of 4 GiB of records or
more that zstd shrank was written with its length wrapped, and *no* behaviour of
zstd can make it open to its records: the reader asks for `plain_len mod 2^32`
bytes and refuses anything else. -/
theorem pre3278_wrapped_frame_unreadable (Z : Codec) (hZ : CodecOK Z) (body : Bytes) (m : Nat)
    (hbig : 2 ^ 32 ≤ body.length) (hc : (pre3278_maybeCompress Z (newBuffer ++ body) m).2 = true) :
    openPayload Z (pre3278_maybeCompress Z (newBuffer ++ body) m).1 ≠ .ok (body, []) := by
  have hl : (newBuffer ++ body).length - 2 = body.length := by simp [newBuffer]
  unfold pre3278_maybeCompress at hc ⊢
  by_cases h1 : m = 0 ∨ (newBuffer ++ body).length < 2 ∨ (newBuffer ++ body).length - 2 < m
  · simp only [h1, ite_true] at hc; cases hc
  simp only [h1, ite_false] at hc ⊢
  have hflag : ((newBuffer ++ body)[1]!.toNat &&& FLAG_COMPRESSED) = 0 := by
    simp [newBuffer, FLAG_COMPRESSED]
  simp only [hflag, ne_eq, not_true_eq_false, ite_false] at hc ⊢
  have hd : (newBuffer ++ body).drop 2 = body := by simp [newBuffer]
  rw [hd] at hc ⊢
  cases hcz : Z.compress body with
  | none => simp only [hcz] at hc; cases hc
  | some frame =>
    simp only [hcz] at hc ⊢
    by_cases h2 : frame.length + COMPRESSED_PREFIX ≥ (newBuffer ++ body).length - 2
    · simp only [h2, ite_true] at hc; cases hc
    simp only [h2, ite_false]
    rw [hl]
    have h0 : (newBuffer ++ body)[0]! = 3 := by simp [newBuffer]
    have h1' : (newBuffer ++ body)[1]! = 0 := by simp [newBuffer]
    rw [h0, h1']
    have e2 : ((0 : UInt8) ||| 1) = 1 := rfl
    rw [e2]
    have hp : body.length % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by decide)
    have hq : frame.length % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by decide)
    intro h
    have := openPayload_compressed_len Z hp hq (hZ.crc_lt body) frame body [] h
    have : body.length % 2 ^ 32 < body.length := Nat.lt_of_lt_of_le hp hbig
    omega

/-- History (before #3278): transparent only for records under 4 GiB, because
`plain_len` was written `as u32`. -/
theorem pre3278_openPayload_maybeCompress (Z : Codec) (hZ : CodecOK Z) (body : Bytes) (m : Nat)
    (hlen : body.length < 2 ^ 32) :
    openPayload Z (pre3278_maybeCompress Z (newBuffer ++ body) m).1 = .ok (body, []) := by
  have hl : (newBuffer ++ body).length - 2 = body.length := by simp [newBuffer]
  unfold pre3278_maybeCompress
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
  by_cases a2 : 2 ^ 32 ≤ buf.length - 2
  · simp only [a2, ite_true] at h; cases h
  simp only [a2, ite_false] at h ⊢
  by_cases b : (buf[1]!.toNat &&& FLAG_COMPRESSED) ≠ 0
  · rw [if_pos b] at h; cases h
  rw [if_neg b] at h ⊢
  cases hc : Z.compress (buf.drop 2) with
  | none => simp only [hc] at h; cases h
  | some frame =>
    simp only [hc] at h ⊢
    by_cases d : frame.length + COMPRESSED_PREFIX ≥ buf.length - 2
    · simp only [d, ite_true] at h; cases h
    simp only [d, ite_false] at h ⊢
    cases hp : framePrefix (buf.length - 2) frame.length (Z.crc (buf.drop 2)) with
    | none => simp only [hp] at h; cases h
    | some pfx =>
      have := framePrefix_length hp
      simp only [List.length_append, List.length_cons, List.length_nil]
      simp [COMPRESSED_PREFIX] at d this; omega

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
