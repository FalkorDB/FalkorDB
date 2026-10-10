import FalkorEffectsCodec.Format
/-!
# `graph/src/effects/mod.rs`: version dispatch, the hex dump, `EffectsBuffer`
and `graph/src/effects/v3/mod.rs`'s small functions.
-/
namespace FalkorCodec

def DESCRIBE_BYTES_PER_LINE : Nat := 32
def DESCRIBE_BYTE_LIMIT : Nat := 2048

inductive DLine where
  | size (n : Nat)
  | v3 (l : Line)
  | unknownVersion (v : Nat)
  | hex (offset : Nat) (chunk : Bytes)
  | moreBytes (n : Nat)
deriving Repr

/-- `buf.chunks(32)`. -/
def chunks32 (buf : Bytes) : List Bytes :=
  if h : buf = [] then [] else buf.take 32 :: chunks32 (buf.drop 32)
termination_by buf.length
decreasing_by simp; have : buf.length ≠ 0 := by simpa using h
              omega

/-- `EffectsPayload::describe` (`effects/mod.rs:246`): size line, the format's
own lines for version 3 (`v3lines`), the first 2 KiB as hex, and a tail note. -/
def describeTop (v3lines : Bytes → List Line) (buf : Bytes) : List DLine :=
  let head := [DLine.size buf.length]
  match buf.head? with
  | none => head
  | some v =>
    let body := if v.toNat = 3 then (v3lines buf).map .v3 else [.unknownVersion v.toNat]
    let hex := ((chunks32 buf).take (DESCRIBE_BYTE_LIMIT / DESCRIBE_BYTES_PER_LINE)).zipIdx.map
      fun (c, i) => DLine.hex (i * DESCRIBE_BYTES_PER_LINE) c
    let tail := if buf.length > DESCRIBE_BYTE_LIMIT then [DLine.moreBytes (buf.length - DESCRIBE_BYTE_LIMIT)] else []
    head ++ body ++ hex ++ tail

theorem chunks32_flatten (buf : Bytes) : (chunks32 buf).flatten = buf := by
  induction buf using chunks32.induct with
  | case1 => simp [chunks32]
  | case2 buf h ih => rw [chunks32]; simp [h, ih]

/-- The hex lines are the buffer itself, 32 bytes per line, up to 2 KiB; an
empty buffer yields only its size. -/
theorem describeTop_spec (v3lines : Bytes → List Line) (buf : Bytes) :
    (buf = [] → describeTop v3lines buf = [.size 0]) ∧
    (describeTop v3lines buf).head? = some (.size buf.length) := by
  constructor
  · intro h; subst h; rfl
  · unfold describeTop; split <;> simp

/-- `EffectsPayload::apply` (`effects/mod.rs:292`): empty is a no-op, version 3
goes to the v3 format, any other version is `UnsupportedVersion`. -/
def applyTop {G E} (v3apply : G → Bytes → Except E G) (unsupported : Nat → E) (g : G) (buf : Bytes) : Except E G :=
  match buf.head? with
  | none => .ok g
  | some v => if v.toNat = 3 then v3apply g buf else .error (unsupported v.toNat)

theorem applyTop_spec {G E} (f : G → Bytes → Except E G) (u : Nat → E) (g : G) (buf : Bytes) :
    (buf = [] → applyTop f u g buf = .ok g) ∧
    (∀ rest, applyTop f u g (3 :: rest) = f g (3 :: rest)) ∧
    (∀ v rest, v.toNat ≠ 3 → applyTop f u g (v :: rest) = .error (u v.toNat)) := by
  refine ⟨fun h => by subst h; rfl, fun rest => rfl, fun v rest h => by simp [applyTop, h]⟩

/-! ### `EffectsBuffer` -/

/-- `std::io::Write for EffectsBuffer` (`:422`, `:430`): append all, report all. -/
def ebWrite (buf b : Bytes) : Bytes × Nat := (buf ++ b, b.length)
def ebFlush (buf : Bytes) : Bytes := buf
/-- `EffectWrite for EffectsBuffer` (`:436-450`): the same as `Vec<u8>`'s. -/
def ebEffectWrite : EffectWrite Bytes := vecWrite

theorem ebWrite_spec (buf b : Bytes) : ebWrite buf b = (buf ++ b, b.length) ∧ ebFlush buf = buf := ⟨rfl, rfl⟩
theorem eb_is_vec : ebEffectWrite = vecWrite := rfl

/-- `EffectsBuffer::new` / `default` (`:456`, `:464`) = `new_buffer()`;
`is_empty` (`:473`) = the format's; `build*`/`replicate` delegate (`:484-540`). -/
def ebNew : Bytes := fmtNewBuffer
theorem ebNew_isEmpty : fmtIsEmpty ebNew = true := rfl
theorem ebNew_eq : ebNew = [3, 0] := rfl

/-! ### `v3/mod.rs` -/

/-- `Opcode::is_batchable`. -/
theorem isBatchable_spec (op : Nat) : isBatchable op = true ↔ 1 ≤ op ∧ op ≤ 8 := by
  simp [isBatchable]

/-- The tag pairs are mutually inverse (`schema_tag`/`entity_from_schema_tag`, …). -/
theorem tags_roundtrip (e : Entity) (c : CType) (s : CStatus) :
    entityFromSchemaTag (schemaTag e) = .ok e ∧ entityFromTag (entityTag e) = .ok e ∧
    ctypeFromTag (ctypeTag c) = .ok c ∧ statusFromTag (statusTag s) = .ok s := by
  cases e <;> cases c <;> cases s <;> exact ⟨rfl, rfl, rfl, rfl⟩

/-- `update_opcode`. -/
def updateOpcode : Entity → Nat | .node => 1 | .rel => 2
theorem updateOpcode_batchable (e : Entity) : isBatchable (updateOpcode e) = true := by
  cases e <;> rfl

/-- `IndexFieldBit::index_type`. -/
theorem bitKind_spec : bitKind 1 = some .fulltext ∧ bitKind 2 = some .range ∧ bitKind 4 = some .range ∧
    bitKind 8 = some .range ∧ bitKind 16 = some .vector ∧ bitKind 32 = none := by decide

/-- `field_type_bits`: exactly the set bits below 2^32, ascending. -/
theorem setBits_spec (ft b : Nat) : b ∈ setBits ft ↔ ∃ i, i < 32 ∧ b = 2 ^ i ∧ ft / 2 ^ i % 2 = 1 := by
  simp only [setBits, List.mem_filterMap, List.mem_range]
  constructor
  · rintro ⟨i, hi, h⟩; split at h <;> simp_all; rename_i h'; subst h; exact ⟨i, hi, rfl, h'⟩
  · rintro ⟨i, hi, rfl, h⟩; exact ⟨i, hi, by simp [h]⟩

/-- `index_type_of`: a field type naming two kinds of index is refused. -/
theorem indexTypeOf_mixed : (indexTypeOf 0x11).toOption = none ∧ (indexTypeOf 0x12).toOption = none ∧
    (indexTypeOf 0x18).toOption = none := by decide

/-- `seal` (`v3/mod.rs`) — `sealBuf` (`seal` is a Lean keyword): `maybe_compress(buf, compression_min_bytes())`. -/
def sealBuf (Z : Codec) (setting : Int) (buf : Bytes) : Bytes := (maybeCompress Z buf (compressionMinBytes setting)).1
theorem seal_transparent (Z : Codec) (hZ : CodecOK Z) (v : Int) (body : Bytes) (h : body.length < 2 ^ 32) :
    openPayload Z (sealBuf Z v (newBuffer ++ body)) = .ok (body, []) :=
  openPayload_maybeCompress Z hZ body _ h

end FalkorCodec
