/-
# `identifier_limits` (`graph/src/identifier_limits.rs`)

Rust `&str::len()` is the UTF-8 byte length, so a name is modelled as a
`List Char` and its length as `byteLen` = Σ `Char.utf8Size`.

| here | there |
| --- | --- |
| `MAX_IDENTIFIER_LEN` | identifier_limits.rs:10 |
| `TOO_LONG`           | identifier_limits.rs:15 |
| `validate`           | `validate_identifier_len` (identifier_limits.rs:26) |
| `isTooLong`          | `is_identifier_too_long` (identifier_limits.rs:42) — `ends_with` |
-/
namespace FalkorRuntimeDS.IdentifierLimitsModel

def MAX_IDENTIFIER_LEN : Nat := 512

def TOO_LONG : List Char := "exceeds maximum length of 512 characters".toList

def byteLen (s : List Char) : Nat := (s.map Char.utf8Size).sum

/-- identifier_limits.rs:26 — `format!("{entity} {TOO_LONG}")` on rejection. -/
def validate (name entity : List Char) : Except (List Char) Unit :=
  if byteLen name > MAX_IDENTIFIER_LEN then .error (entity ++ ' ' :: TOO_LONG) else .ok ()

/-- identifier_limits.rs:42 -/
def isTooLong (err : List Char) : Bool := TOO_LONG.isSuffixOf err

theorem validate_ok_iff (name entity : List Char) :
    validate name entity = .ok () ↔ byteLen name ≤ MAX_IDENTIFIER_LEN := by
  unfold validate; split <;> simp_all <;> omega

/-- Every rejection is recognised by `is_identifier_too_long`, whatever the entity text. -/
theorem isTooLong_of_validate {name entity e : List Char} (h : validate name entity = .error e) :
    isTooLong e = true := by
  unfold validate at h; split at h
  · cases h
    unfold isTooLong
    rw [List.isSuffixOf_iff_suffix]
    exact ⟨entity ++ [' '], by simp⟩
  · cases h

/-- The limit counts bytes, not characters (the message says "characters"; C
FalkorDB's `strnlen` counts bytes too, so the two engines agree). -/
theorem byteLen_replicate (n : Nat) (c : Char) : byteLen (List.replicate n c) = n * c.utf8Size := by
  induction n with
  | zero => simp [byteLen]
  | succ n ih =>
    simp only [byteLen, List.replicate_succ, List.map_cons, List.sum_cons] at *
    rw [ih, Nat.succ_mul]; omega

example : (List.replicate 257 'é').length = 257 ∧
    validate (List.replicate 257 'é') "Label name".toList ≠ .ok () := by
  refine ⟨List.length_replicate, ?_⟩
  rw [Ne, validate_ok_iff, byteLen_replicate]
  have : 'é'.utf8Size = 2 := by decide
  rw [this]; decide

end FalkorRuntimeDS.IdentifierLimitsModel
