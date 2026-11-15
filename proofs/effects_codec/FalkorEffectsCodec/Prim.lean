/-
# Primitives: `graph/src/effects/writer.rs` and `graph/src/effects/reader.rs`

| here | there |
| --- | --- |
| `Bytes`            | `&[u8]` / `Vec<u8>` |
| `le k n`           | `to_le_bytes` for a `k`-byte integer (`writer.rs:45-80`, `EffectWrite::u8..f64`, `schema_id`) |
| `unle`             | `from_le_bytes` (`reader.rs:109-135`) |
| `R α`              | a `&mut Reader` method returning `Result<α, DecodeError>`; the state is `buf[pos..]` (`reader.rs:12`) |
| `take`             | `Reader::take` (`reader.rs:42`) |
| `rdU k`            | `take_array::<k>` + `from_le_bytes` (`reader.rs:103-135`) |
| `guardCount`       | `Reader::guard_count` (`reader.rs:72`) |
| `wString`          | `EffectWrite::string` (`writer.rs:158`) |
| `rString`          | `Reader::string` (`reader.rs:165`) |

The reader state is the *unconsumed suffix* of the buffer. `pos` never appears:
`remaining() = buf.len() - pos` is `r.length`, and `rest()` is `r` itself.
Every way the Rust can panic on this path is modelled as `DErr.panic`, so
"never panics" is the statement `≠ .error .panic`.
-/

namespace FalkorCodec

abbrev Bytes := List UInt8

/-- Decode errors (`graph/src/effects/error.rs:118-240`), only the ones this
codec can raise, plus `panic` for any Rust panic and `outOfFuel` for the
modelling device in `Value.lean` (proved unreachable). -/
inductive DErr where
  | eof (want have_ : Nat)
  | implausible (count remaining : Nat)
  | badString
  | badValueType (t : Nat)
  | badOpcode (t : Nat)
  | emptyRecord (op : Nat)
  | badBool (v : Nat)
  | badSimilarity (v : Nat)
  | badSchemaType (v : Nat)
  | badConstraintType (v : Nat)
  | badEntityType (v : Nat)
  | badConstraintStatus (v : Nat)
  | badFieldType (v : Nat)
  | emptyIndexSchemaList
  | unsupportedVersion (v : Nat)
  | unknownFlags (v : Nat)
  | badCompression
  | compressedLengthMismatch (declared actual : Nat)
  | trailingBytes (n : Nat)
  | checksumMismatch (declared actual : Nat)
  | idList            -- whatever `read_ids` reports; owned by the IdList proof
  | panic
  | outOfFuel
deriving Repr, DecidableEq

/-- A reader action: state is the unconsumed suffix. -/
abbrev R (α : Type) := Bytes → Except DErr (α × Bytes)

instance : Monad R where
  pure a := fun r => .ok (a, r)
  bind m f := fun r => match m r with
    | .error e => .error e
    | .ok (a, r') => f a r'

@[simp] theorem pure_apply (a : α) (r : Bytes) : (pure a : R α) r = .ok (a, r) := rfl
@[simp] theorem bind_apply (m : R α) (f : α → R β) (r : Bytes) :
    (m >>= f) r = (match m r with
      | .error e => .error e
      | .ok (a, r') => f a r') := rfl

def fail (e : DErr) : R α := fun _ => .error e
@[simp] theorem fail_apply (e : DErr) (r : Bytes) : (fail e : R α) r = .error e := rfl

/-! ## Little-endian integers -/

/-- `k` little-endian bytes of `n`, i.e. `(n as uK).to_le_bytes()`. -/
def le : Nat → Nat → Bytes
  | 0, _ => []
  | k + 1, n => UInt8.ofNat (n % 256) :: le k (n / 256)

/-- `uK::from_le_bytes`. -/
def unle : Bytes → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * unle bs

@[simp] theorem le_length (k n : Nat) : (le k n).length = k := by
  induction k generalizing n with
  | zero => rfl
  | succ k ih => simp [le, ih]

theorem unle_le (k n : Nat) : unle (le k n) = n % 2 ^ (8 * k) := by
  induction k generalizing n with
  | zero => simp [le, unle, Nat.mod_one]
  | succ k ih =>
    simp only [le, unle, ih]
    have h1 : (UInt8.ofNat (n % 256)).toNat = n % 256 := by
      simp
    rw [h1]
    have h2 : 2 ^ (8 * (k + 1)) = 256 * 2 ^ (8 * k) := by
      rw [Nat.mul_succ, Nat.pow_add]; omega
    rw [h2, Nat.mod_mul]

theorem unle_le_of_lt {k n : Nat} (h : n < 2 ^ (8 * k)) : unle (le k n) = n := by
  rw [unle_le, Nat.mod_eq_of_lt h]

theorem unle_lt (bs : Bytes) : unle bs < 2 ^ (8 * bs.length) := by
  induction bs with
  | nil => simp [unle]
  | cons b bs ih =>
    simp only [unle, List.length_cons]
    have hb := b.toNat_lt
    have h2 : 2 ^ (8 * (bs.length + 1)) = 256 * 2 ^ (8 * bs.length) := by
      rw [Nat.mul_succ, Nat.pow_add]; omega
    rw [h2]
    have : 256 ≤ 2 ^ 8 := by decide
    omega

/-! ## The cursor -/

/-- `Reader::take` (`reader.rs:42-55`). The check precedes the slice
`&self.buf[self.pos..self.pos + n]`, which is the only way it could panic. -/
def take (n : Nat) : R Bytes := fun r =>
  if r.length < n then .error (.eof n r.length) else .ok (r.take n, r.drop n)

/-- `take_array::<k>` + `from_le_bytes`. -/
def rdU (k : Nat) : R Nat := do
  let bs ← take k
  pure (unle bs)

def u8 : R Nat := rdU 1
def u16 : R Nat := rdU 2
def u32 : R Nat := rdU 4
def u64 : R Nat := rdU 8

/-- `Reader::guard_count` (`reader.rs:72-85`). The Rust multiplies with
`saturating_mul` in `u64`; since `remaining() < 2^64`, `sat(count*min) >
remaining` iff `count*min > remaining` over `Nat`, so this is exact. -/
def guardCount (count minEach : Nat) : R Nat := fun r =>
  if count * minEach > r.length then .error (.implausible count r.length)
  else .ok (count, r)

@[simp] theorem take_append (bs rest : Bytes) :
    take bs.length (bs ++ rest) = .ok (bs, rest) := by
  simp [take]

theorem take_append' {n : Nat} (bs rest : Bytes) (h : bs.length = n) :
    take n (bs ++ rest) = .ok (bs, rest) := by subst h; simp

theorem rdU_le {k n : Nat} (h : n < 2 ^ (8 * k)) (rest : Bytes) :
    rdU k (le k n ++ rest) = .ok (n, rest) := by
  simp [rdU, take_append' (le k n) rest (le_length k n), unle_le_of_lt h]

/-- `writer.rs` widths: `buf.u8`, `u16`, `u32`/`schema_id`, `u64`/`i64`/`f64`. -/
def w8 (n : Nat) : Bytes := le 1 n
def w16 (n : Nat) : Bytes := le 2 n
def w32 (n : Nat) : Bytes := le 4 n
def w64 (n : Nat) : Bytes := le 8 n

theorem u8_w8 {n : Nat} (h : n < 2 ^ 8) (rest : Bytes) : u8 (w8 n ++ rest) = .ok (n, rest) :=
  rdU_le (k := 1) (by simpa using h) rest
theorem u16_w16 {n : Nat} (h : n < 2 ^ 16) (rest : Bytes) :
    u16 (w16 n ++ rest) = .ok (n, rest) := rdU_le (k := 2) (by simpa using h) rest
theorem u32_w32 {n : Nat} (h : n < 2 ^ 32) (rest : Bytes) :
    u32 (w32 n ++ rest) = .ok (n, rest) := rdU_le (k := 4) (by simpa using h) rest
theorem u64_w64 {n : Nat} (h : n < 2 ^ 64) (rest : Bytes) :
    u64 (w64 n ++ rest) = .ok (n, rest) := rdU_le (k := 8) (by simpa using h) rest

/-! ## Strings

`EffectWrite::string` (`writer.rs:158-165`): `u64 (len + 1)`, the bytes, a NUL.
`Reader::string` (`reader.rs:165-181`): reject `len == 0`, `take(len)`, split
off the last byte, which must be `0`, then `String::from_utf8`.

UTF-8 validity is a parameter: Rust `String`s are valid by construction, so the
encoder only ever feeds `from_utf8` a valid body; which validator is used does
not matter to any theorem here. -/

def wString (s : Bytes) : Bytes := w64 (s.length + 1) ++ s ++ [0]

def rString (utf8 : Bytes → Bool) : R Bytes := do
  let len ← u64
  if len = 0 then fail .badString else
  -- `usize::try_from(len)`: identity on a 64-bit target.
  let raw ← take len
  -- `raw.split_at(n - 1)`: `n ≥ 1` by the check above, so it cannot panic.
  let body := raw.take (len - 1)
  let nul := raw.drop (len - 1)
  if nul ≠ [0] then fail .badString else
  if utf8 body then pure body else fail .badString

theorem rString_wString (utf8 : Bytes → Bool) (s rest : Bytes)
    (hlen : s.length + 1 < 2 ^ 64) (hv : utf8 s = true) :
    rString utf8 (wString s ++ rest) = .ok (s, rest) := by
  have ht : take (s.length + 1) (s ++ ([0] ++ rest)) = .ok (s ++ [0], rest) := by
    have := take_append' (n := s.length + 1) (s ++ [0]) rest (by simp)
    simpa only [List.append_assoc] using this
  unfold rString wString
  simp only [List.append_assoc, bind_apply]
  rw [u64_w64 hlen]
  simp only [Nat.add_one_ne_zero, ite_false, bind_apply]
  rw [ht]
  simp [hv]

/-! ## Never over-reads: every action leaves a suffix of its input -/

/-- `m` only ever consumes: on success the new state is a suffix of the old. -/
def Consumes (m : R α) : Prop := ∀ r a r', m r = .ok (a, r') → ∃ pre, r = pre ++ r'

theorem take_consumes (n : Nat) : Consumes (take n) := by
  intro r a r' h
  unfold take at h
  split at h
  · cases h
  · cases h; exact ⟨r.take n, (List.take_append_drop n r).symm⟩

theorem pure_consumes (a : α) : Consumes (pure a : R α) := by
  intro r b r' h; cases h; exact ⟨[], rfl⟩

theorem fail_consumes (e : DErr) : Consumes (fail e : R α) := by
  intro r b r' h; cases h

theorem bind_consumes {m : R α} {f : α → R β} (hm : Consumes m) (hf : ∀ a, Consumes (f a)) :
    Consumes (m >>= f) := by
  intro r b r' h
  simp only [bind_apply] at h
  split at h
  · cases h
  · rename_i a r1 hr1
    obtain ⟨p1, rfl⟩ := hm _ _ _ hr1
    obtain ⟨p2, rfl⟩ := hf a _ _ _ h
    exact ⟨p1 ++ p2, by simp⟩

theorem ite_consumes {c : Prop} [Decidable c] {m₁ m₂ : R α} (h₁ : Consumes m₁)
    (h₂ : Consumes m₂) : Consumes (if c then m₁ else m₂) := by
  split <;> assumption

theorem rdU_consumes (k : Nat) : Consumes (rdU k) :=
  bind_consumes (take_consumes k) (fun _ => pure_consumes _)

theorem guardCount_consumes (c m : Nat) : Consumes (guardCount c m) := by
  intro r a r' h; unfold guardCount at h; split at h
  · cases h
  · cases h; exact ⟨[], rfl⟩

theorem rString_consumes (utf8 : Bytes → Bool) : Consumes (rString utf8) := by
  unfold rString
  refine bind_consumes (rdU_consumes 8) (fun len => ?_)
  refine ite_consumes (fail_consumes _) ?_
  refine bind_consumes (take_consumes _) (fun raw => ?_)
  refine ite_consumes (fail_consumes _) ?_
  exact ite_consumes (pure_consumes _) (fail_consumes _)

/-- No action here panics. -/
def NoPanic (m : R α) : Prop := ∀ r, m r ≠ .error .panic

theorem take_noPanic (n : Nat) : NoPanic (take n) := by
  intro r; unfold take; split <;> simp

theorem bind_noPanic {m : R α} {f : α → R β} (hm : NoPanic m) (hf : ∀ a, NoPanic (f a)) :
    NoPanic (m >>= f) := by
  intro r
  simp only [bind_apply]
  split
  · rename_i e he; intro h; cases h; exact hm r he
  · exact hf _ _

theorem pure_noPanic (a : α) : NoPanic (pure a : R α) := by intro r; simp
theorem fail_noPanic {e : DErr} (h : e ≠ .panic) : NoPanic (fail e : R α) := by
  intro r; simp [h]
theorem ite_noPanic {c : Prop} [Decidable c] {m₁ m₂ : R α} (h₁ : NoPanic m₁)
    (h₂ : NoPanic m₂) : NoPanic (if c then m₁ else m₂) := by
  split <;> assumption
theorem rdU_noPanic (k : Nat) : NoPanic (rdU k) :=
  bind_noPanic (take_noPanic k) (fun _ => pure_noPanic _)
theorem guardCount_noPanic (c m : Nat) : NoPanic (guardCount c m) := by
  intro r; unfold guardCount; split <;> simp

theorem rString_noPanic (utf8 : Bytes → Bool) : NoPanic (rString utf8) := by
  unfold rString
  refine bind_noPanic (rdU_noPanic 8) (fun len => ?_)
  refine ite_noPanic (fail_noPanic (by decide)) ?_
  refine bind_noPanic (take_noPanic _) (fun raw => ?_)
  refine ite_noPanic (fail_noPanic (by decide)) ?_
  exact ite_noPanic (pure_noPanic _) (fail_noPanic (by decide))

/-- `guard_count` is what bounds every `with_capacity`: on success the claimed
count times the item floor fits in what is left. -/
theorem guardCount_bound {c m r a r'} (h : guardCount c m r = .ok (a, r')) :
    a = c ∧ r' = r ∧ c * m ≤ r.length := by
  unfold guardCount at h; split at h
  · cases h
  · cases h; exact ⟨rfl, rfl, by omega⟩

end FalkorCodec
