/-!
# Error and simple-string reply text (`str_as_legal_resp_string`, #3428)

Every error the module replies with (`RedisError` → `Context::reply_error_string`) and every
simple string (`reply_simple_string`) passes through redismodule-rs
`Context::str_as_legal_resp_string` (`src/context/mod.rs`) before reaching
`RedisModule_ReplyWithError` / `RedisModule_ReplyWithSimpleString`, which take a C string and
frame it as `-<text>\r\n` / `+<text>\r\n` (`Resp.respT`).

| here | there |
| --- | --- |
| `scrub`, `legalResp` | `str_as_legal_resp_string` on the `fix/resp-string-bytes` fork branch (#3428, main b84473a7f): `s.bytes().map(\r \n NUL → space)`, then `CString::from_vec_unchecked` |

Before #3428 every char was truncated to its low byte (`c as u8`), so non-ASCII text was
garbled (`é` U+00E9 became the lone byte 0xE9, not UTF-8) and any char whose low byte is 0
(`一` U+4E00, `Ā` U+0100, …) produced a NUL, so `CString::new(..).unwrap()` panicked and took
the server down (#3185). `none` below is that panic.

What is proved for main:
* `bytesOf_toUTF8`: `bytesOf` (Rust `s.bytes()`) is Lean's UTF-8 encoding `String.toUTF8`.
* `legalResp_length`: the reply text has exactly the UTF-8 byte length of the message.
* `legalResp_safe`: no CR, LF or NUL survives — the `from_vec_unchecked` safety comment
  holds and the text cannot break RESP framing.
* `legalResp_high_bytes`: every byte ≥ 0x80 is kept at its position, so every multi-byte
  UTF-8 sequence (all of whose bytes are ≥ 0x80) is passed through unchanged.
* `legalResp_clean`: a message with no CR, LF or NUL byte is sent byte for byte.
* `legalResp_cjk`: `一` goes out as its three UTF-8 bytes `E4 B8 80`.
-/
namespace RedisLayer.ErrorText

/-- The UTF-8 bytes of a Rust `&str` (`s.bytes()`), char by char. -/
def bytesOf (s : String) : List UInt8 := s.toList.flatMap String.utf8EncodeChar

/-- `bytesOf` is Lean's own UTF-8 encoding of the string. -/
theorem bytesOf_toUTF8 (s : String) : (bytesOf s).toByteArray = s.toUTF8 := by
  have h := String.toByteArray_ofList (l := s.toList)
  rw [String.ofList_toList] at h
  rw [String.toUTF8_eq_toByteArray, h]; rfl

/-- One byte of `str_as_legal_resp_string` on main: `b'\r' | b'\n' | 0 => b' '`, else `b`. -/
def scrub (b : UInt8) : UInt8 := if b = 13 ∨ b = 10 ∨ b = 0 then 32 else b

/-- `str_as_legal_resp_string` on main (#3428): map the UTF-8 bytes of the message. -/
def legalResp (s : String) : List UInt8 := (bytesOf s).map scrub

/-- **PROVEN**: the reply text keeps the message's UTF-8 byte length. -/
theorem legalResp_length (s : String) : (legalResp s).length = (bytesOf s).length := by
  simp [legalResp]

theorem scrub_safe (b : UInt8) : scrub b ≠ 13 ∧ scrub b ≠ 10 ∧ scrub b ≠ 0 := by
  unfold scrub; split
  · decide
  · rename_i h; simp only [not_or] at h; exact h

/-- **PROVEN**: no CR, LF or NUL is left, so `CString::from_vec_unchecked` is sound (no interior
NUL) and the `-…\r\n` frame cannot be split. -/
theorem legalResp_safe (s : String) : ∀ b ∈ legalResp s, b ≠ 13 ∧ b ≠ 10 ∧ b ≠ 0 := by
  intro b hb
  simp only [legalResp, List.mem_map] at hb
  obtain ⟨a, _, rfl⟩ := hb
  exact scrub_safe a

theorem scrub_high (b : UInt8) (h : 128 ≤ b) : scrub b = b := by
  unfold scrub; split
  · rename_i h'; exfalso; rcases h' with rfl | rfl | rfl <;> revert h <;> decide
  · rfl

/-- **PROVEN** (byte preservation): position by position, every byte ≥ 0x80 of the message's
UTF-8 encoding is sent unchanged, so non-ASCII characters survive intact. -/
theorem legalResp_high_bytes (s : String) (i : Nat) (h : i < (bytesOf s).length)
    (hb : 128 ≤ (bytesOf s)[i]) :
    (legalResp s)[i]'(by simpa [legalResp] using h) = (bytesOf s)[i] := by
  simp only [legalResp, List.getElem_map]
  exact scrub_high _ hb

/-- **PROVEN**: a message with no CR, LF or NUL byte goes out byte for byte. -/
theorem legalResp_clean (s : String) (h : ∀ b ∈ bytesOf s, b ≠ 13 ∧ b ≠ 10 ∧ b ≠ 0) :
    legalResp s = bytesOf s := by
  unfold legalResp
  conv => rhs; rw [← List.map_id (bytesOf s)]
  apply List.map_congr_left
  intro b hb
  obtain ⟨h1, h2, h3⟩ := h b hb
  simp [scrub, h1, h2, h3]

/-! ### History: before #3428 (b6567cb97) -/

/-- **PROVEN** (#3428): main sends `一` as its UTF-8 bytes. -/
theorem legalResp_cjk : legalResp "一" = [0xE4, 0xB8, 0x80] := by decide

/-- **PROVEN** (#3428): main sends `é` as its UTF-8 bytes `C3 A9`. -/
theorem legalResp_latin1 : legalResp "é" = [0xC3, 0xA9] := by decide

/-- The ASCII rewrite is unchanged: `a\r\nb` still becomes `a  b`. -/
theorem legalResp_crlf : legalResp "a\r\nb" = [97, 32, 32, 98] := by decide

end RedisLayer.ErrorText
