import FalkorEffectsCodec.RecArms
/-!
# `IndexFieldOptions` (`records.rs:162` encode_sized, `:217` decode_sized)
-/
namespace FalkorCodec

/-- `nostem`: `u8::from(bool)`; decode `0|1`, else `BadBool`. -/
def bBool : Blk Bool where
  enc b := .ok (w8 (if b then 1 else 0))
  dec := do
    let v ← u8
    if v = 0 then pure false else if v = 1 then pure true else fail (.badBool v)
  ok _ := True
  rt := by
    intro b bs rest _ e; cases e
    cases b <;> (simp only [bind_apply]; rw [u8_w8 (by decide)]; rfl)

def asciiLower (c : UInt8) : UInt8 := if 65 ≤ c ∧ c ≤ 90 then c + 32 else c
/-- `eq_ignore_ascii_case`. -/
def eqIgnoreCase (a b : Bytes) : Bool := a.map asciiLower == b.map asciiLower

def IP : Bytes := [105, 112]
def COSINE : Bytes := [99, 111, 115, 105, 110, 101]
def EUCLIDEAN : Bytes := [101, 117, 99, 108, 105, 100, 101, 97, 110]

/-- The similarity-function code the encoder writes (`records.rs:192-200`). -/
def simCode (s : Bytes) : Nat := if eqIgnoreCase s IP then 1 else if eqIgnoreCase s COSINE then 2 else 0

def simOfCode (v : Nat) : Except DErr Bytes :=
  if v = 1 then .ok IP else if v = 2 then .ok COSINE else if v = 0 then .ok EUCLIDEAN
  else .error (.badSimilarity v)

/-- What a similarity name becomes after a trip over the wire. -/
def simCanon (s : Bytes) : Bytes := if eqIgnoreCase s IP then IP else if eqIgnoreCase s COSINE then COSINE else EUCLIDEAN

def bSim : Blk Bytes where
  enc s := .ok (w64 (simCode s))
  dec := do let v ← u64; match simOfCode v with | .ok s => pure s | .error e => fail e
  ok s := simCanon s = s
  rt := by
    intro s bs rest h e; cases e
    simp only [bind_apply]; rw [u64_w64 (by unfold simCode; split <;> (try split) <;> decide)]
    simp only
    rw [← h]; unfold simCode simCanon simOfCode
    by_cases h1 : eqIgnoreCase s IP <;> by_cases h2 : eqIgnoreCase s COSINE <;> simp [h1, h2] <;> rfl

/-- **The similarity function is normalised by the wire**: whatever the master
holds, the replica decodes `simCanon` of it — lowercase `ip`/`cosine`, and
*anything else* becomes `euclidean`. -/
theorem bSim_lossy (s : Bytes) (rest : Bytes) :
    ∃ bs, bSim.enc s = .ok bs ∧ bSim.dec (bs ++ rest) = .ok (simCanon s, rest) := by
  refine ⟨_, rfl, ?_⟩
  simp only [bSim, bind_apply]; rw [u64_w64 (by unfold simCode; split <;> (try split) <;> decide)]
  simp only; unfold simCode simCanon simOfCode
  by_cases h1 : eqIgnoreCase s IP <;> by_cases h2 : eqIgnoreCase s COSINE <;> simp [h1, h2] <;> rfl

-- `COSINE` (as typed) arrives as `cosine`; `L2` arrives as `euclidean`.
#guard simCanon [67, 79, 83, 73, 78, 69] == COSINE
#guard simCanon [76, 50] == EUCLIDEAN

/-- `VectorIndexOptions`: `dimension`, `M`, `efConstruction`, `efRuntime`,
`similarity_function` (usize ↔ u64 is the identity on a 64-bit target). -/
abbrev VecT := Nat × (Option Nat × (Option Nat × (Option Nat × Option Bytes)))

def bVec : Blk VecT :=
  bU64.seq fun _ => (bOpt bU64).seq fun _ => (bOpt bU64).seq fun _ => (bOpt bU64).seq fun _ => bOpt bSim

/-- `language`, `stopwords`, `weight` (f64 bits), `nostem`, `phonetic`. -/
abbrev TextT := Option Bytes × (Option (List Bytes) × (Option Nat × (Option Bool × Option Bytes)))

def bStopwords (utf8 : Bytes → Bool) : Blk (List Bytes) :=
  bItems 8 9 .panic (bStr utf8) (by intro s bs e; cases e; exact wString_len s)

def bText (utf8 : Bytes → Bool) : Blk TextT :=
  (bOpt (bStr utf8)).seq fun _ => (bOpt (bStopwords utf8)).seq fun _ => (bOpt bU64).seq fun _ =>
    (bOpt bBool).seq fun _ => bOpt (bStr utf8)

/-- The vector half is present exactly when the field type has the vector bit;
the encoder refuses a mismatch (`OptionsFieldTypeMismatch`). -/
def bVecPart (ft : Nat) : Blk (Option VecT) where
  enc
    | none => if hasVector ft then .error .optionsFieldTypeMismatch else .ok []
    | some v => if hasVector ft then bVec.enc v else .error .optionsFieldTypeMismatch
  dec := if hasVector ft then do let v ← bVec.dec; pure (some v) else pure none
  ok
    | none => hasVector ft = false
    | some v => hasVector ft = true ∧ bVec.ok v
  rt := by
    intro o bs rest h e
    cases o with
    | none => simp only at h; simp only [h] at e ⊢; cases e; rfl
    | some v =>
      obtain ⟨h1, h2⟩ := h
      simp only [h1, ite_true] at e ⊢
      simp only [bind_apply]; rw [bVec.rt v bs rest h2 e]; rfl

abbrev OptsT := TextT × Option VecT

/-- `IndexFieldOptions::encode_sized(buf, field_type)` / `decode_sized(r, field_type)`. -/
def bOpts (utf8 : Bytes → Bool) (ft : Nat) : Blk OptsT := (bText utf8).seq fun _ => bVecPart ft

end FalkorCodec
