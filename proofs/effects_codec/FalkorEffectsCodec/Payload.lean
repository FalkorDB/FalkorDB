import FalkorEffectsCodec.RecRT
/-!
# Payload framing: `open_payload`, `maybe_compress`, `frame_prefix`, `Payload`, `Records`

`graph/src/effects/v3/records.rs:1211-1480`.

**#3610 (durations carry three components).** The writer's version is 4
(`EFFECTS_VERSION`, `v3/mod.rs:83`); 3 (`EFFECTS_VERSION_V3`, `:87`) is still
read. `open_payload` (`records.rs:1308`) accepts either and the `Payload` carries
the header's version (`openPayload_versions`); `Records::next` (`:1282`) decodes
every record in that version (`payloadRecords`: `lg := version = 3`, i.e.
`read_record::<3>` reads `T_DURATION` as one legacy `i64`). `records_concat` holds
in either version.

**#3278 (`dcbd445d1`, issue #3276).** `maybe_compress` used to write both lengths
`as u32`; a payload of 4 GiB of records or more was framed with a wrapped
`plain_len`, which the reader takes as the decompress ceiling, so the replica
refused it and looped on forced resyncs. It
now declines to compress such a payload (`u32::try_from(..).is_err()`) and builds
the prefix through `frame_prefix`, which refuses a length that does not fit
(`framePrefix_none`). `openPayload_maybeCompress` therefore holds for every
payload, with no length hypothesis. zstd and crc32 are external C/
Rust libraries; what the framing needs from them is a hypothesis structure
(`Codec`), not an axiom: zstd's frame round-trips, crc32 is a 32-bit function.
-/
namespace FalkorCodec

def EFFECTS_VERSION : Nat := 4
def EFFECTS_VERSION_V3 : Nat := 3
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

/-- `open_payload` (`records.rs:1308`): the plain record bytes and the header's
version (`Payload { bytes, version }`), or why not. -/
def openPayload (Z : Codec) : R (Bytes × Nat) := do
  let version ← u8
  if version ≠ EFFECTS_VERSION ∧ version ≠ EFFECTS_VERSION_V3 then fail (.unsupportedVersion version) else
  let flags ← u8
  if flags &&& (255 - KNOWN_FLAGS) ≠ 0 then fail (.unknownFlags flags) else
  if flags &&& FLAG_COMPRESSED = 0 then fun r => .ok ((r, version), []) else
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
      else .ok ((plain, version), [])

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
  unfold framePrefix; rw [ite_eq_right (by omega)]

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

/-- `new_buffer` (`v3/mod.rs`): `[EFFECTS_VERSION, 0]`. -/
def newBuffer : Bytes := [4, 0]

theorem u8_cons (b : UInt8) (r : Bytes) : u8 (b :: r) = .ok (b.toNat, r) := by
  simp [u8, rdU, FalkorCodec.take, unle]

/-- A compressed frame only ever opens to exactly `plain_len` bytes: the reader
checks the decompressed length against the declared one. -/
theorem openPayload_compressed_len (Z : Codec) {p c k : Nat} (hp : p < 2 ^ 32) (hc : c < 2 ^ 32)
    (hk : k < 2 ^ 32) (fr : Bytes) (b : Bytes × Nat) (r : Bytes)
    (v : UInt8) (hv : v.toNat = EFFECTS_VERSION ∨ v.toNat = EFFECTS_VERSION_V3)
    (h : openPayload Z ([v, 1] ++ w32 p ++ w32 c ++ w32 k ++ fr) = .ok (b, r)) : b.1.length = p := by
  have hv' : ¬(v.toNat ≠ EFFECTS_VERSION ∧ v.toNat ≠ EFFECTS_VERSION_V3) := by omega
  simp only [EFFECTS_VERSION, EFFECTS_VERSION_V3] at hv'
  simp [openPayload, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, hv', KNOWN_FLAGS, FLAG_COMPRESSED, u32_w32 hp, u32_w32 hc,
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
          · intro h; cases h; simpa
          · simp
        · simp
    · simp

/-- An uncompressed payload opens to its record bytes, in this build's version. -/
theorem openPayload_plain (Z : Codec) (body : Bytes) :
    openPayload Z (newBuffer ++ body) = .ok ((body, EFFECTS_VERSION), []) := by
  simp [openPayload, newBuffer, bind_apply, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, KNOWN_FLAGS,
    FLAG_COMPRESSED]

/-- **The two versions this build reads** (`records.rs:1311`): a header of 3 or 4
opens, and the `Payload` carries it; any other version is `UnsupportedVersion`
before a further byte is read. -/
theorem openPayload_versions (Z : Codec) :
    (∀ body, openPayload Z ([3, 0] ++ body) = .ok ((body, 3), [])) ∧
    (∀ body, openPayload Z ([4, 0] ++ body) = .ok ((body, 4), [])) ∧
    (∀ (v : UInt8) rest, v.toNat ≠ 3 → v.toNat ≠ 4 →
      openPayload Z (v :: rest) = .error (.unsupportedVersion v.toNat)) := by
  refine ⟨fun body => ?_, fun body => ?_, fun v rest h3 h4 => ?_⟩
  · simp [openPayload, bind_apply, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, KNOWN_FLAGS,
      FLAG_COMPRESSED]
  · simp [openPayload, bind_apply, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, KNOWN_FLAGS,
      FLAG_COMPRESSED]
  · simp [openPayload, bind_apply, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, h3, h4]

/-- **Compression is transparent**: sealing a fresh buffer — compressed or
left alone — and opening it gives back exactly the record bytes, for records of
any length (#3278: a payload too long for the `u32` lengths ships uncompressed). -/
theorem openPayload_maybeCompress (Z : Codec) (hZ : CodecOK Z) (body : Bytes) (m : Nat) :
    openPayload Z (maybeCompress Z (newBuffer ++ body) m).1 = .ok ((body, EFFECTS_VERSION), []) := by
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
    have h0 : (newBuffer ++ body)[0]! = 4 := by simp [newBuffer]
    have h1' : (newBuffer ++ body)[1]! = 0 := by simp [newBuffer]
    rw [h0, h1']
    have e2 : ((0 : UInt8) ||| 1) = 1 := rfl
    rw [e2]
    have hfr : take frame.length frame = .ok (frame, []) := by simpa using take_append frame []
    simp [openPayload, u8_cons, EFFECTS_VERSION, EFFECTS_VERSION_V3, KNOWN_FLAGS, FLAG_COMPRESSED, u32_w32 hlen,
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
  · rw [ite_eq_left b] at h; cases h
  rw [ite_eq_right b] at h ⊢
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

/-- `Records::next` (`records.rs:1282`): stop at the end, yield one record,
fuse after the first error. `lg` is the record decoder's version: `true` for
`EffectDecode<3>` (`read_record::<3>`, `:619`), `false` for `<4>`. -/
def readAllRecords (C : IdCodec) (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) : Nat → R (List (Record C.T))
  | 0 => fun _ => .error .outOfFuel
  | fuel + 1 => fun r =>
    if r = [] then .ok ([], [])
    else match readRecord C utf8 lg I r with
      | .error e => .error e
      | .ok (x, r') => match readAllRecords C utf8 lg I fuel r' with
        | .error e => .error e
        | .ok (xs, r'') => .ok (x :: xs, r'')

/-- `Payload::records` (`records.rs:1255`): a fresh `Records` over the plain
bytes (`Payload::deref`, `:1247`), carrying the header's version; `next`
dispatches on it (`:1282`) — version 3 reads with `EffectDecode<3>`, anything
else (`open_payload` accepts only 4) with `<4>`. -/
def payloadRecords (C : IdCodec) (utf8 I : Bytes → Bool) (p : Bytes × Nat) (fuel : Nat) :
    Except DErr (List (Record C.T) × Bytes) :=
  readAllRecords C utf8 (p.2 = EFFECTS_VERSION_V3) I fuel p.1

/-- A payload built from encoded records yields those records, in order — in
either record-decoder version (the encoder's bytes are version-independent; `RecOk`
under `lg = true` excludes durations, which v3 reads differently). -/
theorem records_concat (C : IdCodec) (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) :
    ∀ (items : List (Bytes × Record C.T)),
    (∀ p ∈ items, p.1 ≠ [] ∧ encRecord C utf8 lg I p.2 = .ok p.1 ∧ RecOk C utf8 lg I p.2) →
    ∀ fuel, items.length < fuel →
    readAllRecords C utf8 lg I fuel (items.map Prod.fst).flatten = .ok (items.map Prod.snd, [])
  | [], _, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    simp [readAllRecords]
  | (b, x) :: items, h, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have hb := h (b, x) (by simp)
    have ih := records_concat C utf8 lg I items (fun p hp => h p (by simp [hp])) f (by simp at hf; omega)
    simp only [List.map_cons, List.flatten_cons, readAllRecords]
    rw [ite_eq_right (by simp [hb.1])]
    rw [readRecord_encRecord C utf8 lg I x b _ hb.2.1 hb.2.2]
    simp only [ih]

/-- **End to end, v4**: what the master seals opens, in version 4, to exactly the
records it wrote. -/
theorem payloadRecords_maybeCompress (C : IdCodec) (utf8 I : Bytes → Bool) (Z : Codec) (hZ : CodecOK Z)
    (items : List (Bytes × Record C.T))
    (h : ∀ p ∈ items, p.1 ≠ [] ∧ encRecord C utf8 false I p.2 = .ok p.1 ∧ RecOk C utf8 false I p.2)
    (m fuel : Nat) (hf : items.length < fuel) :
    (openPayload Z (maybeCompress Z (newBuffer ++ (items.map Prod.fst).flatten) m).1).map
      (fun q => payloadRecords C utf8 I q.1 fuel) = .ok (.ok (items.map Prod.snd, [])) := by
  rw [openPayload_maybeCompress Z hZ]
  simp only [Except.map, payloadRecords, EFFECTS_VERSION, EFFECTS_VERSION_V3]
  simp only [show ((4 : Nat) = 3) = False from by decide, decide_false]
  rw [records_concat C utf8 false I items h fuel hf]

end FalkorCodec
