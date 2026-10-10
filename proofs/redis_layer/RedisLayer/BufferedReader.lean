import RedisLayer.BufferedIO
/-!
# Readers of the v19 byte layer, and the write→read round trip

`BufferedReader` (`buffered_io.rs:170-331`) and `PipeReader` (`:437-559`) share their
read logic and differ only in where a chunk comes from: `Src.load` is `load_chunk`
(`:212` / `:498`) and `Src.blob` is the `TYPE_BLOB` arm of `read_buffer` (`:307-325` /
`:474-479`). `Res.crash` is a Rust panic or UB (indexing `buf[pos]` of an empty chunk,
`load_string_buffer(NULL)`).
-/
namespace RedisLayer.BufferedIO

inductive Res (α : Type) where
  | ok (a : α)
  | err
  | crash
  deriving Repr

def Res.bind {α β} : Res α → (α → Res β) → Res β
  | .ok a, f => f a
  | .err, _ => .err
  | .crash, _ => .crash

structure Src (σ : Type) where
  load : σ → Res (Bytes × σ)
  blob : σ → Res (Bytes × σ)

structure Rd (σ : Type) where
  rest : Bytes      -- `buf[pos..]`
  src : σ

variable {σ : Type} (S : Src σ)

/-- `ensure_available` (`:221` / `:515`). -/
def ensure (r : Rd σ) : Res (Rd σ) :=
  if r.rest = [] then (S.load r.src).bind fun (c, s) => .ok ⟨c, s⟩ else .ok r

/-- `read_tag` (`:229` / `:522`): `buf[pos]` panics on an empty chunk. -/
def readTag (e : Nat) (r : Rd σ) : Res (Rd σ) :=
  (ensure S r).bind fun r =>
    match r.rest with
    | [] => .crash
    | t :: rs => if t = e then .ok ⟨rs, r.src⟩ else .err

/-- `read_bytes` (`:246` / `:538`). -/
def readBytes (n : Nat) (r : Rd σ) : Res (Bytes × Rd σ) :=
  if n ≤ r.rest.length then .ok (r.rest.take n, ⟨r.rest.drop n, r.src⟩) else .err

/-- `read_unsigned/signed/double/float/buffer`; the `W` argument only says which. -/
def readW : W → Rd σ → Res (W × Rd σ)
  | .u _, r => (readTag S TYPE_UNSIGNED r).bind fun r =>
      (readBytes 8 r).bind fun (b, r) => .ok (.u (fromLE b), r)
  | .s _, r => (readTag S TYPE_SIGNED r).bind fun r =>
      (readBytes 8 r).bind fun (b, r) => .ok (.s (ofU (fromLE b)), r)
  | .d _, r => (readTag S TYPE_DOUBLE r).bind fun r =>
      (readBytes 8 r).bind fun (b, r) => .ok (.d (fromLE b), r)
  | .f _, r => (readTag S TYPE_FLOAT r).bind fun r =>
      (readBytes 4 r).bind fun (b, r) => .ok (.f (fromLE b), r)
  | .buf _, r => (ensure S r).bind fun r =>
      match r.rest with
      | [] => .crash
      | t :: rs =>
        if t = TYPE_BYTES then
          (readBytes 8 ⟨rs, r.src⟩).bind fun (lb, r) =>
            (readBytes (fromLE lb) r).bind fun (ds, r) => .ok (.buf ds, r)
        else if t = TYPE_BLOB then
          (S.blob r.src).bind fun (ds, s) => .ok (.buf ds, ⟨[], s⟩)
        else .err

def readSeq : List W → Rd σ → Res (List W × Rd σ)
  | [], r => .ok ([], r)
  | x :: xs, r => (readW S x r).bind fun (y, r) =>
      (readSeq xs r).bind fun (ys, r) => .ok (y :: ys, r)

/-! ## Sources -/

/-- `BufferedReader::new` (`rdb` live) / `from_slice` (`rdb` NULL): chunks are the RDB's
string buffers. `load_string_buffer(NULL)` is UB (`crash`); the blob arm checks NULL first
and errors (`:310-314`). -/
def listSrc (null : Bool) : Src (List Bytes) where
  load := fun l => if null then .crash else match l with
    | [] => .err
    | c :: cs => .ok (c, cs)
  blob := fun l => if null then .err else match l with
    | [] => .err
    | c :: cs => .ok (c, cs)

/-- `PipeReader::load_chunk` (`:498-513`) on the byte stream of the pipe: an 8-byte LE
length, then that many bytes; length 0 is end-of-stream (`Err`). The blob arm
(`:474-479`) is `load_chunk` and then the whole chunk. -/
def pipeLoad (s : Bytes) : Res (Bytes × Bytes) :=
  if s.length < 8 then .err else
  let len := fromLE (s.take 8)
  if len = 0 then .err else
  if (s.drop 8).length < len then .err else .ok ((s.drop 8).take len, (s.drop 8).drop len)

def pipeSrc : Src Bytes where
  load := pipeLoad
  blob := pipeLoad

/-- `PipeWriter`'s byte stream (`flush` `:357`, blob `:429-431`, `finish` `:376-382`):
each chunk the buffered writer would save, length-prefixed, then a zero length. -/
def frame (cs : List Bytes) : Bytes := (cs.map fun c => le 8 c.length ++ c).flatten

def pipeStream (B : Nat) (xs : List W) : Bytes := frame (writeAll B xs) ++ le 8 0

theorem pipeLoad_frame (c : Bytes) (cs : List Bytes) (t : Bytes)
    (hc : c ≠ []) (hl : c.length < 256^8) :
    pipeLoad (frame (c :: cs) ++ t) = .ok (c, frame cs ++ t) := by
  have hlen : fromLE (le 8 c.length) = c.length := fromLE_le 8 _ hl
  have hne : c.length ≠ 0 := by simpa using hc
  simp only [pipeLoad, frame, List.map_cons, List.flatten_cons, List.append_assoc]
  have e1 : (le 8 c.length ++ (c ++ ((cs.map fun c => le 8 c.length ++ c).flatten ++ t))).take 8
      = le 8 c.length := by simpa using List.take_left (l₁ := le 8 c.length) (l₂ := (c ++ ((cs.map fun c => le 8 c.length ++ c).flatten ++ t)))
  have e2 : (le 8 c.length ++ (c ++ ((cs.map fun c => le 8 c.length ++ c).flatten ++ t))).drop 8
      = c ++ ((cs.map fun c => le 8 c.length ++ c).flatten ++ t) := by
    simpa using List.drop_left (l₁ := le 8 c.length) (l₂ := (c ++ ((cs.map fun c => le 8 c.length ++ c).flatten ++ t)))
  rw [e1, e2, hlen]
  simp [hne]
  rw [if_neg (by omega), if_neg (by omega)]

/-- End of stream: the terminator makes the next `load_chunk` fail cleanly. -/
theorem pipeLoad_end (t : Bytes) : pipeLoad (le 8 0 ++ t) = .err := by
  simp [pipeLoad, le, fromLE]

end RedisLayer.BufferedIO
