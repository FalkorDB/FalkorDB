import RedisLayer.RoundTrip
/-!
# The reader's encoding version and `GRAPH.RESTORE` blob versioning (#3610)

`src/serializers/buffered_io.rs`. A `BufferedReader` now carries the stream's encoding version
(`encver`, `:172`), which `Reader::encver` (`:179`) reports and `Value::decode` branches on for
`T_DURATION` (one legacy offset below 21, three components from 21). The byte layer itself is
unchanged: every read (`readW`) ignores `encver`, so all round trips of `RoundTrip.lean` hold
for a reader at any version.

| Lean | Rust |
| --- | --- |
| `ENCODING_VERSION`, `UNVERSIONED_BLOB_ENCVER` | `serialization.rs:38` (21), `buffered_io.rs:614` (19) |
| `BR`, `BR.encverOf` | `struct BufferedReader` `:170`, `Reader::encver` `:179` |
| `brNew`              | `BufferedReader::new(rdb, encver)` `:202` |
| `fromSliceAt`, `fromSlice` | `from_slice_at` `:244`, `from_slice` `:216` (`#[cfg(test)]`) |
| `fromBlob`           | `from_blob` `:228` |
| `rdbEncver`          | `rdb_load_graph`'s `u64::try_from(encver).unwrap_or(u64::MAX)` (decoder/mod.rs:122) |

* `vec_blob_roundtrip` — what `vec_save_graph` writes (`write_unsigned(ENCODING_VERSION)`, then
  the graph) is read by `from_blob` at `encver = 21`, positioned after the prefix, and then reads
  back every value written.
* `fromBlob_old_blob` — a blob from an older build (it opens with the graph name's `TYPE_BYTES`
  buffer) is read from its first byte at version 19, exactly as `from_slice` read it before.
* `prefixed_was_invalid` — every byte string `from_blob` treats as prefixed was refused by the
  old reader's very first read: no blob that loaded before is reinterpreted.
* `fromBlob_newer` — a prefix above 21 is refused; `fromBlob_prefix` — any prefix `≤ 21` is taken
  as the version (there is no lower bound: a prefix below 19 is read at that version).
* `fromBlob_short` — a truncated prefix is an `Err`, never the `load_chunk(NULL)` crash.
-/
namespace RedisLayer.BufferedIO

/-- `graphblas::serialization::ENCODING_VERSION` (serialization.rs:38). -/
def ENCODING_VERSION : Nat := 21
/-- `UNVERSIONED_BLOB_ENCVER` (buffered_io.rs:614). -/
def UNVERSIONED_BLOB_ENCVER : Nat := 19

variable {σ : Type}

/-- `BufferedReader` (`:170`): its encoding version and the byte reader. -/
structure BR (σ : Type) where
  encver : Nat
  rd : Rd σ

/-- `Reader::encver` for `BufferedReader` (`:179`): the field. -/
def BR.encverOf (r : BR σ) : Nat := r.encver

/-- `BufferedReader::new(rdb, encver)` (`:202`): no chunk loaded yet. -/
def brNew (chunks : List Bytes) (encver : Nat) : BR (List Bytes) := ⟨encver, ⟨[], chunks⟩⟩

/-- `BufferedReader::from_slice_at(data, encver)` (`:244`): `rdb = NULL`, `buf = data`. -/
def fromSliceAt (data : Bytes) (encver : Nat) : BR (List Bytes) := ⟨encver, ⟨data, []⟩⟩

/-- `from_slice` (`:216`, test-only now): `from_slice_at(data, ENCODING_VERSION)`. -/
def fromSlice (data : Bytes) : BR (List Bytes) := fromSliceAt data ENCODING_VERSION

/-- `BufferedReader::from_blob(data)` (`:228`). -/
def fromBlob (data : Bytes) : Res (BR (List Bytes)) :=
  if data.head? ≠ some TYPE_UNSIGNED then .ok (fromSliceAt data UNVERSIONED_BLOB_ENCVER)
  else
    (readW (listSrc true) (.u 0) (fromSliceAt data ENCODING_VERSION).rd).bind fun (w, r) =>
      match w with
      | .u v => if v > ENCODING_VERSION then .err else .ok ⟨v, r⟩
      | _ => .err   -- unreachable: `read_unsigned` returns a `u64`

/-- `rdb_load_graph`'s `u64::try_from(encver).unwrap_or(u64::MAX)` (decoder/mod.rs:122). -/
def rdbEncver (encver : Int) : Nat := if 0 ≤ encver then encver.toNat else 2 ^ 64 - 1

theorem encver_new (chunks : List Bytes) (v : Nat) : (brNew chunks v).encverOf = v := rfl
theorem encver_fromSliceAt (data : Bytes) (v : Nat) :
    (fromSliceAt data v).encverOf = v ∧ (fromSliceAt data v).rd = ⟨data, []⟩ := ⟨rfl, rfl⟩

/-- Past `check_encoding_version` (19..=21) the reader gets exactly the RDB's version. -/
theorem rdbEncver_gate (v : Int) (h1 : 19 ≤ v) (h2 : v ≤ 21) : (rdbEncver v : Int) = v := by
  unfold rdbEncver; split <;> omega

/-- The RDB round trip is unchanged: `BufferedReader::new(rdb, encver)` reads back what the
writer saved, for any `encver` (the byte layer never looks at it). -/
theorem rdb_roundtrip_ver (B : Nat) (h9 : 9 ≤ B) (hB : B < 256^8) (xs : List W)
    (hok : ∀ x ∈ xs, x.ok) (rest : List Bytes) (v : Nat) :
    readSeq (listSrc false) xs (brNew (writeAll B xs ++ rest) v).rd = .ok (xs, ⟨[], rest⟩) :=
  rdb_roundtrip B h9 hB xs hok rest

theorem readU_record (v : Nat) (hv : v < 256 ^ 8) (T : Bytes) :
    readW (listSrc true) (.u 0) ⟨record (.u v) ++ T, []⟩ = .ok (.u v, ⟨T, []⟩) := by
  simp [readW, readTag, ensure, record, TYPE_UNSIGNED, readBytes, Res.bind, fromLE_le 8 v hv]

/-- A prefix `≤ ENCODING_VERSION` is the reader's version, positioned right after it. -/
theorem fromBlob_prefix (v : Nat) (hv : v ≤ ENCODING_VERSION) (T : Bytes) :
    fromBlob (record (.u v) ++ T) = .ok ⟨v, ⟨T, []⟩⟩ := by
  have hlt : v < 256 ^ 8 := by unfold ENCODING_VERSION at hv; omega
  have hh : (record (.u v) ++ T).head? = some TYPE_UNSIGNED := by simp [record]
  simp only [fromBlob, hh, ne_eq, not_true_eq_false, ite_false, fromSliceAt, readU_record v hlt T,
    Res.bind]
  simp [Nat.not_lt.2 hv]

/-- A prefix above `ENCODING_VERSION` is refused. -/
theorem fromBlob_newer (v : Nat) (hv : ENCODING_VERSION < v) (hlt : v < 256 ^ 8) (T : Bytes) :
    fromBlob (record (.u v) ++ T) = .err := by
  have hh : (record (.u v) ++ T).head? = some TYPE_UNSIGNED := by simp [record]
  simp only [fromBlob, hh, ne_eq, not_true_eq_false, ite_false, fromSliceAt, readU_record v hlt T,
    Res.bind]
  simp [hv]

/-- **`vec_save_graph` → `vec_load_graph`.** The blob is `write_unsigned(ENCODING_VERSION)` then
the graph's writes (`VecWriter`); `from_blob` reads it at version 21 and the reader then returns
every value written, ending at the blob's end. -/
theorem vec_blob_roundtrip (xs : List W) (hok : ∀ x ∈ xs, x.ok) (T : Bytes) :
    fromBlob (vecWrite (.u ENCODING_VERSION :: xs) ++ T) = .ok ⟨ENCODING_VERSION, ⟨vecWrite xs ++ T, []⟩⟩ ∧
      readSeq (listSrc true) xs ⟨vecWrite xs ++ T, []⟩ = .ok (xs, ⟨T, []⟩) := by
  refine ⟨?_, vec_roundtrip xs hok T⟩
  rw [vecWrite_cons, List.append_assoc]
  exact fromBlob_prefix _ (Nat.le_refl _) _

/-- Any blob whose first byte is not `TYPE_UNSIGNED` (including the empty one) is read from its
first byte at version 19. -/
theorem fromBlob_unversioned (data : Bytes) (h : data.head? ≠ some TYPE_UNSIGNED) :
    fromBlob data = .ok (fromSliceAt data UNVERSIONED_BLOB_ENCVER) := by
  simp [fromBlob, h]

/-- **Older builds' blobs.** Their first write is the header's graph name (`write_buffer`,
`TYPE_BYTES`), so `from_blob` reads them exactly as `from_slice` did — at version 19, from byte 0. -/
theorem fromBlob_old_blob (b : Bytes) (xs : List W) (T : Bytes) :
    fromBlob (vecWrite (.buf b :: xs) ++ T) =
      .ok (fromSliceAt (vecWrite (.buf b :: xs) ++ T) UNVERSIONED_BLOB_ENCVER) := by
  apply fromBlob_unversioned
  simp [vecWrite, record, TYPE_BYTES, TYPE_UNSIGNED]

/-- **No reinterpretation.** A blob `from_blob` treats as prefixed (first byte `TYPE_UNSIGNED`)
failed the old reader's first read — `Header::decode`'s `read_buffer` of the graph name — so it
never loaded before #3610. -/
theorem prefixed_was_invalid (data : Bytes) (h : data.head? = some TYPE_UNSIGNED) (x : Bytes) :
    readW (listSrc true) (.buf x) ⟨data, []⟩ = .err := by
  cases data with
  | nil => simp at h
  | cons t ts =>
    simp at h; subst h
    simp [readW, ensure, Res.bind, TYPE_UNSIGNED, TYPE_BYTES, TYPE_BLOB]

/-- A prefix tag with fewer than 8 bytes after it is an `Err` (`read_bytes` bounds check), not a
read past the end. -/
theorem fromBlob_short (rest : Bytes) (h : rest.length < 8) :
    fromBlob (TYPE_UNSIGNED :: rest) = .err := by
  have h8 : ¬ 8 ≤ rest.length := by omega
  simp [fromBlob, fromSliceAt, readW, readTag, ensure, Res.bind, readBytes, h8]

end RedisLayer.BufferedIO
