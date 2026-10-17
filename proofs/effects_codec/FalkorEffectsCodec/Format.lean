import FalkorEffectsCodec.Payload
/-!
# `EffectsFormat for EffectsPayload` (`graph/src/effects/v3/format.rs`) and the
small glue around it (`payload.rs`, `announce.rs`, `error.rs`'s `LocalName`)

The graph-touching steps (`apply_effects`, `for_each_record`,
`build_index_buffer`, `build_constraint_buffer`) are proven in
proofs/effects_emit_apply; here they are parameters, and the theorems say what
`format.rs` does around them.
-/
namespace FalkorCodec

/-- `EffectsFormat::new_buffer` (`format.rs:22`) = `v3::new_buffer`. -/
def fmtNewBuffer : Bytes := newBuffer
/-- `EffectsFormat::is_empty` (`format.rs:27`): `buf.len() <= HEADER_LEN`. -/
def fmtIsEmpty (buf : Bytes) : Bool := buf.length ≤ 2

theorem fmtIsEmpty_iff (body : Bytes) : fmtIsEmpty (fmtNewBuffer ++ body) = true ↔ body = [] := by
  simp [fmtIsEmpty, fmtNewBuffer, newBuffer, List.length_eq_zero_iff]

/-- `EffectsFormat::replicate` (`format.rs:32`): `seal` then hand to the sink;
`seal` is `maybe_compress(buf, compression_min_bytes())` (`v3/mod.rs`). -/
def fmtReplicate (Z : Codec) (minBytes : Nat) (buf : Bytes) : Bytes := (maybeCompress Z buf minBytes).1

/-- What the replica receives opens to exactly the records the master built. -/
theorem fmtReplicate_transparent (Z : Codec) (hZ : CodecOK Z) (m : Nat) (body : Bytes)
    (h : body.length < 2 ^ 32) :
    openPayload Z (fmtReplicate Z m (fmtNewBuffer ++ body)) = .ok (body, []) :=
  openPayload_maybeCompress Z hZ body m h

/-- `compression_min_bytes` (`v3/mod.rs`): `usize::try_from(i64).unwrap_or(0)` —
a negative setting disables compression. -/
def compressionMinBytes (v : Int) : Nat := if v < 0 then 0 else v.toNat

theorem compressionMinBytes_neg (v : Int) (h : v < 0) : compressionMinBytes v = 0 := by
  simp [compressionMinBytes, h]

theorem seal_disabled (Z : Codec) (buf : Bytes) : maybeCompress Z buf 0 = (buf, false) := by
  simp [maybeCompress]

/-! ### `describe` (`format.rs:46`) -/

inductive Line where
  | unreadable (flags : Nat)
  | header (flags : Nat) (compressed : Bool) (decoded : Nat) (stopped : Bool)
  | record (i : Nat)
  | bad (i : Nat)
  | more (n : Nat)
deriving DecidableEq, Repr

def DESCRIBE_RECORD_LIMIT : Nat := 16

/-- The loop over `payload.records()` (fused after the first error). -/
def describeLoop {E A} : List (Except E A) → Nat → Bool → List Line → (Nat × Bool × List Line)
  | [], decoded, stopped, lines => (decoded, stopped, lines)
  | .ok _ :: rs, decoded, stopped, lines =>
    describeLoop rs (decoded + 1) stopped (if decoded < DESCRIBE_RECORD_LIMIT then lines ++ [.record decoded] else lines)
  | .error _ :: rs, decoded, _, lines => describeLoop rs decoded true (lines ++ [.bad decoded])

def describe {E A} (flags : Nat) (opened : Bool) (recs : List (Except E A)) : List Line :=
  if !opened then [.unreadable flags] else
  let (decoded, stopped, lines) := describeLoop recs 0 false []
  let lines := if decoded > DESCRIBE_RECORD_LIMIT then lines ++ [.more (decoded - DESCRIBE_RECORD_LIMIT)] else lines
  .header flags (flags % 2 = 1) decoded stopped :: lines

/-- `Records` fuses: what it yields is some `ok`s, then at most one error. -/
def Fused {E A} : List (Except E A) → Prop
  | [] => True
  | .ok _ :: rs => Fused rs
  | [.error _] => True
  | .error _ :: _ :: _ => False

def nOk {E A} : List (Except E A) → Nat
  | [] => 0
  | .ok _ :: rs => nOk rs + 1
  | .error _ :: rs => nOk rs

theorem describeLoop_spec {E A} : ∀ (rs : List (Except E A)) (d : Nat) (st : Bool) (ls : List Line),
    Fused rs →
    (describeLoop rs d st ls).1 = d + nOk rs ∧
    (describeLoop rs d st ls).2.2.length =
      ls.length + (min (d + nOk rs) DESCRIBE_RECORD_LIMIT - min d DESCRIBE_RECORD_LIMIT) +
        (if rs.any (fun r => match r with | .error _ => true | .ok _ => false) then 1 else 0)
  | [], d, st, ls, _ => by simp [describeLoop, nOk]
  | .ok _ :: rs, d, st, ls, hf => by
    have ih := describeLoop_spec rs (d + 1) st (if d < DESCRIBE_RECORD_LIMIT then ls ++ [.record d] else ls) hf
    simp only [describeLoop, nOk, List.any_cons]
    refine ⟨by rw [ih.1]; omega, ?_⟩
    rw [ih.2]; split <;> simp [DESCRIBE_RECORD_LIMIT] at * <;> omega
  | [.error _], d, st, ls, _ => by simp [describeLoop, nOk]
  | .error _ :: _ :: _, _, _, _, hf => by simp [Fused] at hf

/-- **`describe` reports every record it decoded and where it stopped**: the
header carries the true count; at most 16 record lines, one error line iff
decoding stopped, one "more" line iff over 16. -/
theorem describe_spec {E A} (flags : Nat) (recs : List (Except E A)) (hf : Fused recs) :
    ∃ stopped rest, describe flags true recs = .header flags (flags % 2 = 1) (nOk recs) stopped :: rest ∧
      rest.length = min (nOk recs) DESCRIBE_RECORD_LIMIT +
        (if recs.any (fun r => match r with | .error _ => true | .ok _ => false) then 1 else 0) +
        (if nOk recs > DESCRIBE_RECORD_LIMIT then 1 else 0) := by
  have h := describeLoop_spec recs 0 false [] hf
  simp only [describe, Bool.not_true, Bool.false_eq_true, ite_false]
  generalize describeLoop recs 0 false [] = D at h ⊢
  obtain ⟨dec, st, ls⟩ := D
  simp only at h ⊢
  rw [h.1]
  simp only [Nat.zero_add]
  refine ⟨st, _, rfl, ?_⟩
  have h2 := h.2
  simp only [Nat.zero_add, List.length_nil] at h2 ⊢
  generalize (recs.any fun r => match r with | .error _ => true | .ok _ => false) = he at h2 ⊢
  have hm : min 0 DESCRIBE_RECORD_LIMIT = 0 := by simp
  rw [hm] at h2
  by_cases hc : nOk recs > DESCRIBE_RECORD_LIMIT
  · rw [if_pos hc, if_pos hc]; simp only [List.length_append, List.length_cons, List.length_nil]
    cases he <;> simp at h2 ⊢ <;> omega
  · rw [if_neg hc, if_neg hc]
    cases he <;> simp at h2 ⊢ <;> omega

/-! ### `build` (`format.rs:115`): encode every record, count them, stop at the first failure -/

def buildLoop {Rec} (enc : Rec → Except EErr Bytes) : List Rec → Bytes → Nat → Option EErr → (Bytes × Nat × Option EErr)
  | [], buf, n, failed => (buf, n, failed)
  | r :: rs, buf, n, failed =>
    match failed with
    | some e => buildLoop enc rs buf n (some e)
    | none => match enc r with
      | .ok b => buildLoop enc rs (buf ++ b) (n + 1) none
      | .error e => buildLoop enc rs buf n (some e)

def build {Rec} (enc : Rec → Except EErr Bytes) (recs : List Rec) (buf : Bytes) : Bytes × Except EErr Nat :=
  let (buf, n, failed) := buildLoop enc recs buf 0 none
  (buf, match failed with | none => .ok n | some e => .error e)

/-- When every record encodes, `build` appends them all, in order, and returns
how many. -/
theorem build_ok {Rec} (enc : Rec → Except EErr Bytes) :
    ∀ (recs : List Rec) (bss : List Bytes) (buf : Bytes) (n : Nat),
    recs.length = bss.length → (∀ i (h1 : i < recs.length) (h2 : i < bss.length), enc recs[i] = .ok bss[i]) →
    buildLoop enc recs buf n none = (buf ++ bss.flatten, n + recs.length, none)
  | [], [], buf, n, _, _ => by simp [buildLoop]
  | r :: rs, b :: bs, buf, n, hl, h => by
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    simp only [buildLoop, h0]
    rw [build_ok enc rs bs (buf ++ b) (n + 1) (by simpa using hl)
      (fun i h1 h2 => by have := h (i + 1) (by simp; omega) (by simp; omega); simpa using this)]
    simp; omega
  | [], _ :: _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, hl, _ => by simp at hl

/-- Once a record fails, nothing after it is encoded (the buffer is frozen). -/
theorem buildLoop_after_fail {Rec} (enc : Rec → Except EErr Bytes) :
    ∀ (recs : List Rec) (buf : Bytes) (n : Nat) (e : EErr),
    buildLoop enc recs buf n (some e) = (buf, n, some e)
  | [], _, _, _ => rfl
  | _ :: rs, buf, n, e => by simp only [buildLoop]; exact buildLoop_after_fail enc rs buf n e

/-- `apply`, `build_index`, `build_constraint` (`format.rs:108/144/154`) delegate. -/
def fmtApply {G E} (applyEffects : G → Bytes → Except E G) (g : G) (buf : Bytes) := applyEffects g buf
theorem fmtApply_eq {G E} (f : G → Bytes → Except E G) (g : G) (b : Bytes) : fmtApply f g b = f g b := rfl
def fmtBuildIndex {P G X} (bib : P → G → Bool → X → Bytes → Except String Bytes) := bib
theorem fmtBuildIndex_eq {P G X} (f : P → G → Bool → X → Bytes → Except String Bytes) : fmtBuildIndex f = f := rfl
def fmtBuildConstraint {G X B} (bcb : G → Bool → X → B → Bytes → Except String Bytes) := bcb
theorem fmtBuildConstraint_eq {G X B} (f : G → Bool → X → B → Bytes → Except String Bytes) :
    fmtBuildConstraint f = f := rfl

/-! ### `payload.rs`, `announce.rs`, `error.rs` -/

/-- `take_effects_buffer` (`payload.rs`): `borrow_mut().take()?` then
`(!buf.is_empty()).then_some(buf)`. Returns (result, cell afterwards). -/
def takeEffectsBuffer (cell : Option Bytes) : Option Bytes × Option Bytes :=
  match cell with
  | none => (none, none)
  | some b => (if b ≠ [] then some b else none, none)

/-- The cell is always emptied, and only a non-empty buffer is handed on. -/
theorem takeEffectsBuffer_spec (cell : Option Bytes) :
    (takeEffectsBuffer cell).2 = none ∧
    ((takeEffectsBuffer cell).1 = some b ↔ cell = some b ∧ b ≠ []) := by
  cases cell with
  | none => simp [takeEffectsBuffer]
  | some c => by_cases h : c = [] <;> simp [takeEffectsBuffer, h] <;> (try constructor) <;> (try intro h') <;> simp_all

/-- `SchemaBaseline::of` (`announce.rs`): the three dictionary sizes. -/
structure Baseline where
  labels : Nat
  types : Nat
  attrs : Nat
def baselineOf {G} (labels types attrs : G → List Bytes) (g : G) : Baseline :=
  ⟨(labels g).length, (types g).length, (attrs g).length⟩
theorem baselineOf_spec {G} (l t a : G → List Bytes) (g : G) :
    baselineOf l t a g = ⟨(l g).length, (t g).length, (a g).length⟩ := rfl

/-- `Display for LocalName` (`error.rs:460`). -/
def localName (n : Option String) : String :=
  match n with
  | some n => " (that id is '" ++ n ++ "' here)"
  | none => ""
theorem localName_none : localName none = "" := rfl
theorem localName_some (n : String) : localName (some n) = " (that id is '" ++ n ++ "' here)" := rfl

end FalkorCodec
