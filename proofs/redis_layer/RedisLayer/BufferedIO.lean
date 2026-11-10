/-!
# `src/serializers/buffered_io.rs` — the v19 type-tagged byte layer

Line numbers refer to `origin/main` (`3fec7d7c9`).

| here | there |
| --- | --- |
| `le` / `fromLE`               | `u64::to_le_bytes` / `from_le_bytes` (8 bytes), `f32` (4 bytes) |
| `toU` / `ofU`                 | `i64` two's complement under `to_le_bytes` |
| `W`, `record`                    | the logical `Writer` calls and their inline record (`:68-103`, `:107-118`) |
| `W.enc`                       | the logical byte stream (inline record, or `[6] ++ data` for a blob) |
| `flush`/`accommodate`/`push`  | `BufferedWriter::flush` `:51`, `::accommodate` `:59` |
| `write`, `G`                  | `write_unsigned/signed/double/float/buffer` `:68-126`, `finish` `:129` |
| `Rd`, `ensure`, `readTag`, `readBytes` | `BufferedReader::ensure_available` `:221`, `read_tag` `:229`, `read_bytes` `:246` |
| `readW`                       | `read_unsigned/signed/double/float/buffer` `:272-330` (+ trait shims `:177-192`) |
| `listSrc`                     | `load_chunk` `:212` over `RedisModule_LoadStringBuffer` |
| `pipeSrc`, `frame`            | `PipeWriter::flush/finish` `:357-382`, `PipeReader::load_chunk` `:498` |
| `vecWrite`                    | `VecWriter` `:576-620` |

The Redis RDB is modelled as the ordered list of buffers `RedisModule_SaveStringBuffer`
stored; `RedisModule_LoadStringBuffer` hands them back in order (the Redis module API
contract — the only FFI fact used, and it enters as the definition of `listSrc`, not as an
axiom). `f64`/`f32`/`i64` payloads are carried as their bit patterns: `to_le_bytes` /
`from_le_bytes` are bit-exact, so NaN payloads and `-0.0` survive by construction.

`pos` is folded into `rest = buf[pos..]`: every read only inspects `buf[pos..]`; `pos`
itself appears only inside error strings, which are not modelled (errors are `.err`).
-/

namespace RedisLayer.BufferedIO

abbrev Bytes := List Nat

/-- `n.to_le_bytes()` truncated to `k` bytes. -/
def le : Nat → Nat → Bytes
  | 0, _ => []
  | k+1, n => n % 256 :: le k (n / 256)

def fromLE : Bytes → Nat
  | [] => 0
  | b :: bs => b + 256 * fromLE bs

@[simp] theorem le_length (k n : Nat) : (le k n).length = k := by
  induction k generalizing n <;> simp [le, *]

theorem fromLE_le (k n : Nat) (h : n < 256 ^ k) : fromLE (le k n) = n := by
  induction k generalizing n with
  | zero => simp [le, fromLE]; omega
  | succ k ih =>
    have : n / 256 < 256 ^ k := by
      rw [Nat.div_lt_iff_lt_mul (by decide)]; rw [Nat.pow_succ] at h; exact h
    simp [le, fromLE, ih _ this]; omega

/-- `i64 → u64` bit pattern. -/
def toU (i : Int) : Nat := if i < 0 then (i + 2^64).toNat else i.toNat
def ofU (n : Nat) : Int := if n < 2^63 then n else (n : Int) - 2^64
def i64ok (i : Int) : Prop := -2^63 ≤ i ∧ i < 2^63

theorem toU_lt (i : Int) (h : i64ok i) : toU i < 256 ^ 8 := by
  unfold toU i64ok at *; split <;> omega
theorem ofU_toU (i : Int) (h : i64ok i) : ofU (toU i) = i := by
  unfold toU ofU i64ok at *; split <;> split <;> omega

/-! ## Tags (`:22-29`) -/
def TYPE_BYTES := 0
def TYPE_FLOAT := 1
def TYPE_DOUBLE := 2
def TYPE_SIGNED := 3
def TYPE_UNSIGNED := 4
def TYPE_BLOB := 6

/-- One logical `Writer` call. -/
inductive W where
  | u (n : Nat)        -- write_unsigned
  | s (i : Int)        -- write_signed
  | d (bits : Nat)     -- write_double (f64 bits)
  | f (bits : Nat)     -- write_float  (f32 bits)
  | buf (data : Bytes) -- write_buffer
  deriving DecidableEq, Repr

def W.ok : W → Prop
  | .u n => n < 256^8
  | .s i => i64ok i
  | .d b => b < 256^8
  | .f b => b < 256^4
  | .buf ds => ds.length < 256^8

/-- The inline record a write pushes into the buffer. -/
def record : W → Bytes
  | .u n => TYPE_UNSIGNED :: le 8 n
  | .s i => TYPE_SIGNED :: le 8 (toU i)
  | .d b => TYPE_DOUBLE :: le 8 b
  | .f b => TYPE_FLOAT :: le 4 b
  | .buf ds => TYPE_BYTES :: le 8 ds.length ++ ds

theorem rec_ne (x : W) : record x ≠ [] := by cases x <;> simp [record]

section Writer
variable (B : Nat)   -- `BUFFER_SIZE` (256_000); every theorem holds for any `B ≥ 9`.

/-- Does `write_buffer` inline (`:111-112`)? Non-buffers always inline. -/
def inl : W → Bool
  | .buf ds => decide (1 + 8 + ds.length ≤ B)
  | _ => true

/-- The payload of a `write_buffer` (empty for the scalar writes). -/
def payload : W → Bytes
  | .buf ds => ds
  | _ => []

/-- The logical stream of one write. -/
def W.enc (x : W) : Bytes := if inl B x then record x else TYPE_BLOB :: payload x

structure BW where
  out : List Bytes   -- buffers handed to `RedisModule_SaveStringBuffer`, in order
  buf : Bytes
  deriving Repr

/-- `flush` (`:51-56`). -/
def flush (w : BW) : BW := if w.buf.isEmpty then w else ⟨w.out ++ [w.buf], []⟩
/-- `accommodate` (`:59-66`). -/
def accommodate (w : BW) (needed : Nat) : BW :=
  if w.buf.length + needed > B then flush w else w
def push (w : BW) (bs : Bytes) : BW := ⟨w.out, w.buf ++ bs⟩

/-- One writer call (`write_unsigned` … `write_buffer`, `:68-126`). -/
def write (w : BW) (x : W) : BW :=
  if inl B x then push (accommodate B w (record x).length) (record x)
  else
    let w := flush (push (accommodate B w 1) [TYPE_BLOB])
    ⟨w.out ++ [payload x], w.buf⟩

/-- `finish` (`:129`): flush the tail; the result is the RDB buffer list. -/
def finish (w : BW) : List Bytes := (flush w).out

/-- What a writer starting with pending buffer `buf` emits for `xs`. -/
def G (buf : Bytes) (xs : List W) : List Bytes := finish (xs.foldl (write B) ⟨[], buf⟩)

def writeAll (xs : List W) : List Bytes := G B [] xs

/-! ### Concatenation: chunking is invisible -/

def flat (w : BW) : Bytes := w.out.flatten ++ w.buf

theorem flat_flush (w : BW) : flat (flush w) = flat w := by
  unfold flush flat; split <;> simp_all

@[simp] theorem flush_buf (w : BW) : (flush w).buf = [] := by
  unfold flush; split <;> simp_all

theorem flat_accommodate (w : BW) (n : Nat) : flat (accommodate B w n) = flat w := by
  unfold accommodate; split <;> simp [flat_flush]

theorem flat_write (w : BW) (x : W) : flat (write B w x) = flat w ++ W.enc B x := by
  unfold write W.enc
  by_cases h : inl B x
  · simp only [h, ite_true]
    have := flat_accommodate B w (record x).length
    simp [push, flat] at *; rw [← List.append_assoc, this]; simp
  · simp only [h]
    have h1 := flat_flush (push (accommodate B w 1) [TYPE_BLOB])
    have h2 := flat_accommodate B w 1
    simp [flat, push] at h1 h2 ⊢
    rw [h1, ← List.append_assoc, h2]; simp

theorem flat_fold (w : BW) (xs : List W) :
    flat (xs.foldl (write B) w) = flat w ++ (xs.map (W.enc B)).flatten := by
  induction xs generalizing w with
  | nil => simp
  | cons x xs ih => simp [List.foldl, ih, flat_write]

/-- **Buffered writes = concatenation of the logical writes**, whatever `B` is, i.e.
wherever the flush points fall. -/
theorem writeAll_flatten (xs : List W) :
    (writeAll B xs).flatten = (xs.map (W.enc B)).flatten := by
  have := flat_fold B ⟨[], []⟩ xs
  have h2 := flat_flush (xs.foldl (write B) ⟨[], []⟩)
  simp [flat] at this
  unfold writeAll G finish
  have h3 : (flush (List.foldl (write B) ⟨[], []⟩ xs)).buf = [] := by
    unfold flush; split <;> simp_all
  simp [flat, h3] at h2; rw [h2, this]

/-- An extra flush anywhere changes nothing in the byte stream either. -/
theorem flat_extra_flush (w : BW) (xs ys : List W) :
    flat (ys.foldl (write B) (flush (xs.foldl (write B) w)))
      = flat (ys.foldl (write B) (xs.foldl (write B) w)) := by
  simp [flat_fold, flat_flush]

end Writer

end RedisLayer.BufferedIO
