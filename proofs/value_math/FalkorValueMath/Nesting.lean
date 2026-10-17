import FalkorValueMath.Misc
/-!
# Bounded value nesting (#3339, b080a44c1; GHSA-mwvh-8fq5-x2j8)

| Lean | Rust |
| --- | --- |
| `MAX_VALUE_NESTING`, `valueNestingError` | value.rs:62, `value_nesting_error` :67 |
| `DErr`, `decodeValue`, `decodeValue.decodeN` | `decode_value(r, depth)` value.rs:2166-2233 (the `for _ in 0..len` loop = `decodeN`) |
| `decode` | `Decode<19>::decode` value.rs:2154 (= `decode_value(r, 0)`) |
| `isValidProperty`, `validateListProperty`, `validateProperty` | pending.rs:69, :98, :115 (copied from proofs/pending_commit `PA`, here over `V F` so the codec theorems apply) |

`depth` is the doc-comment's measure (`[]` and `{}` are 1, `[[]]` is 2, a scalar 0).
`within d v` says that no `decode_value` call made while decoding `v` at depth `d` sees
`depth > MAX_VALUE_NESTING`: every sub-value sits inside at most `MAX - d` arrays.

* `decodeValue_refuses_past_cap` — past the cap the decoder returns the nesting error
  before reading a token, so the recursion never goes deeper than `MAX + 1` frames
  (`decodeValue_reach`).
* `decodeValue_agrees` — otherwise it decodes exactly as before #3339: its `Ok` results
  are the old decoder's results that are `within` the cap, and nothing else.
* `decode_encode` / `decode_encode_deep` — on an encoded value: round-trip iff `within`,
  else the nesting error. `within_of_depth` / `not_within_of_depth` place the cut between
  `depth ≤ MAX` (always loads) and `depth ≥ MAX + 2` (always refused).
* `validateProperty_depth`, **`stored_roundtrip`** — every value a property write accepts
  has `depth ≤ MAX` and round-trips through the RDB codec (the save/load guarantee).
* History: `pre3339_decode_any_depth` (every nesting decoded, one frame per level),
  `pre3339_validate_any_depth` (any depth was storable).
-/

namespace ValueMath
open FalkorCore

variable {F : Type}

def MAX_VALUE_NESTING : Nat := 4096

/-- `value_nesting_error()` (value.rs:67). -/
def valueNestingError : String :=
  "Value nesting exceeds the maximum depth of " ++ toString MAX_VALUE_NESTING

theorem valueNestingError_text :
    valueNestingError = "Value nesting exceeds the maximum depth of 4096" := rfl

/-- `Err(value_nesting_error())` vs every other decode error (reader errors, short vector
buffers, unknown tags). Running out of the model's fuel is `other` too. -/
inductive DErr | nesting | other
  deriving DecidableEq

variable (co : CodecOps F)

/-- `decode_value(r, depth)` (value.rs:2166): the depth check first, then the old
decoder with `depth + 1` for array elements. -/
def decodeValue : Nat → Nat → List (Tok F) → Except DErr (V F × List (Tok F))
  | 0, depth, _ => if depth > MAX_VALUE_NESTING then .error .nesting else .error .other
  | fuel + 1, depth, toks =>
    if depth > MAX_VALUE_NESTING then .error .nesting else
    match toks with
    | .u tag :: rest =>
      if tag = T_NULL then .ok (.null, rest)
      else if tag = T_BOOL then match rest with
        | .s i :: r => .ok (.bool (i ≠ 0), r) | _ => .error .other
      else if tag = T_INT64 then match rest with
        | .s i :: r => .ok (.int i, r) | _ => .error .other
      else if tag = T_DOUBLE then match rest with
        | .d f :: r => .ok (.float f, r) | _ => .error .other
      else if tag = T_STRING ∨ tag = T_INTERN + T_STRING then match rest with
        | .buf b :: r =>
          .ok (.str (co.fromUtf8Lossy (if b.getLast? = some 0 then b.dropLast else b)), r)
        | _ => .error .other
      else if tag = T_ARRAY then match rest with
        | .u len :: r => (decodeN fuel (depth + 1) len r).map (fun (l, r') => (.list l, r'))
        | _ => .error .other
      else if tag = T_POINT then match rest with
        | .d a :: .d b :: r => .ok (.point (co.narrow a) (co.narrow b), r) | _ => .error .other
      else if tag = T_VECTOR_F32 then match rest with
        | .buf b :: r => match readVec co b with
          | some v => .ok (.vec v, r)
          | none => .error .other
        | _ => .error .other
      else if tag = T_DATETIME then match rest with
        | .s i :: r => .ok (.datetime i, r) | _ => .error .other
      else if tag = T_DATE then match rest with
        | .s i :: r => .ok (.date i, r) | _ => .error .other
      else if tag = T_TIME then match rest with
        | .s i :: r => .ok (.time i, r) | _ => .error .other
      else if tag = T_DURATION then match rest with
        | .s i :: r => .ok (.duration i, r) | _ => .error .other
      else .error .other
    | _ => .error .other
where
  /-- `for _ in 0..len { items.push(decode_value(r, depth + 1)?) }`. -/
  decodeN : Nat → Nat → Nat → List (Tok F) → Except DErr (List (V F) × List (Tok F))
  | _, _, 0, r => .ok ([], r)
  | fuel, depth, n + 1, r => match decodeValue fuel depth r with
    | .ok (x, r') => (decodeN fuel depth n r').map (fun (xs, r'') => (x :: xs, r''))
    | .error e => .error e

/-- `Decode<19>::decode` (value.rs:2154). -/
def decode (fuel : Nat) (toks : List (Tok F)) : Except DErr (V F × List (Tok F)) :=
  decodeValue co fuel 0 toks

/-! ## The cap -/

/-- **Never past the cap**: a call at `depth > MAX_VALUE_NESTING` reads nothing and
returns `Err(value_nesting_error())`. -/
theorem decodeValue_refuses_past_cap (fuel d : Nat) (toks : List (Tok F)) (h : d > MAX_VALUE_NESTING) :
    decodeValue co fuel d toks = .error .nesting := by
  cases fuel <;> (unfold decodeValue; simp [h])

/-- The deepest `decode_value` call a decode makes (ghost: mirrors `decodeValue`'s
call tree, including the elements decoded before an error stops the loop). -/
def reach : Nat → Nat → List (Tok F) → Nat
  | 0, depth, _ => depth
  | fuel + 1, depth, toks =>
    if depth > MAX_VALUE_NESTING then depth else
    match toks with
    | .u tag :: .u len :: r => if tag = T_ARRAY then max depth (reachN fuel (depth + 1) len r) else depth
    | _ => depth
where
  reachN : Nat → Nat → Nat → List (Tok F) → Nat
  | _, depth, 0, _ => depth
  | fuel, depth, n + 1, r => max (reach fuel depth r) (match decodeValue co fuel depth r with
    | .ok (_, r') => reachN fuel depth n r'
    | .error _ => depth)

/-- **Bounded recursion**: from the top-level call (`depth = 0`), no call is ever made
at a depth above `MAX + 1` — whatever the input. -/
theorem decodeValue_reach : ∀ (fuel d : Nat) (toks : List (Tok F)), d ≤ MAX_VALUE_NESTING + 1 →
    reach co fuel d toks ≤ MAX_VALUE_NESTING + 1 := by
  intro fuel
  induction fuel with
  | zero => intro d toks h; simp [reach]; omega
  | succ f ih =>
    intro d toks h
    have hN : ∀ n d r, d ≤ MAX_VALUE_NESTING + 1 → reach.reachN co f d n r ≤ MAX_VALUE_NESTING + 1 := by
      intro n
      induction n with
      | zero => intro d r h; simp [reach.reachN]; omega
      | succ n ihn =>
        intro d r h
        simp only [reach.reachN]
        refine Nat.max_le.2 ⟨ih d r h, ?_⟩
        split
        · exact ihn _ _ h
        · exact h
    unfold reach
    split
    · exact h
    · split
      · split
        · exact Nat.max_le.2 ⟨h, hN _ _ _ (by omega)⟩
        · exact h
      · exact h

theorem decode_reach (fuel : Nat) (toks : List (Tok F)) :
    reach co fuel 0 toks ≤ MAX_VALUE_NESTING + 1 :=
  decodeValue_reach co fuel 0 toks (by decide)

/-! ## Agreement with the old decoder below the cap -/

mutual
/-- `within d v`: decoding `v` at depth `d` never trips the check. -/
def within : Nat → V F → Bool
  | d, .list l => decide (d ≤ MAX_VALUE_NESTING) && withinL (d + 1) l
  | d, _ => decide (d ≤ MAX_VALUE_NESTING)
def withinL : Nat → List (V F) → Bool
  | _, [] => true
  | d, x :: xs => within d x && withinL d xs
end

theorem withinL_iff (d : Nat) (l : List (V F)) : withinL d l = true ↔ ∀ x ∈ l, within d x = true := by
  induction l with
  | nil => simp [withinL]
  | cons x xs ih => simp [withinL, ih]

/-- The old result, kept only when `within` the cap. -/
def capped (d : Nat) : Option (V F × List (Tok F)) → Option (V F × List (Tok F))
  | some (v, r) => if within d v then some (v, r) else none
  | none => none

def cappedN (d : Nat) : Option (List (V F) × List (Tok F)) → Option (List (V F) × List (Tok F))
  | some (l, r) => if withinL d l then some (l, r) else none
  | none => none

/-- **Decodes exactly as before below the cap**: `decode_value(r, d)` succeeds with
`(v, rest)` iff the pre-#3339 decoder does and `v` is `within` the cap at `d`. -/
theorem decodeValue_agrees : ∀ (fuel d : Nat) (toks : List (Tok F)),
    (decodeValue co fuel d toks).toOption = capped d (pre3339_decode co fuel toks) := by
  intro fuel
  induction fuel with
  | zero =>
    intro d toks
    by_cases h : d > MAX_VALUE_NESTING <;> simp [decodeValue, pre3339_decode, capped, h, Except.toOption]
  | succ f ih =>
    intro d toks
    have hN : ∀ n d r, d ≤ MAX_VALUE_NESTING + 1 →
        (decodeValue.decodeN co f d n r).toOption = cappedN d (pre3339_decode.decodeN co f n r) := by
      intro n
      induction n with
      | zero => intro d r _; simp [decodeValue.decodeN, pre3339_decode.decodeN, cappedN, withinL,
          Except.toOption]
      | succ n ihn =>
        intro d r hd
        have hx := ih d r
        simp only [decodeValue.decodeN, pre3339_decode.decodeN]
        rcases hp : pre3339_decode co f r with _ | ⟨x, r'⟩
        · rw [hp] at hx
          rcases hdv : decodeValue co f d r with e | ⟨x, r'⟩
          · simp [cappedN, Except.toOption]
          · rw [hdv] at hx; simp [capped, Except.toOption] at hx
        · rw [hp] at hx
          by_cases hw : within d x = true
          · simp only [capped, hw, ite_true] at hx
            have hdv : decodeValue co f d r = .ok (x, r') := by
              revert hx; cases decodeValue co f d r <;> simp [Except.toOption]
            rw [hdv]
            have h2 := ihn d r' hd
            rcases hq' : pre3339_decode.decodeN co f n r' with _ | ⟨xs, r''⟩
            · rw [hq'] at h2
              rcases hq : decodeValue.decodeN co f d n r' with e | q
              · simp [hq, hq', cappedN, Except.toOption, Except.map]
              · rw [hq] at h2; simp [cappedN, Except.toOption] at h2
            · rw [hq'] at h2
              by_cases hwl : withinL d xs = true
              · simp only [cappedN, hwl, ite_true] at h2
                have hq : decodeValue.decodeN co f d n r' = .ok (xs, r'') := by
                  revert h2; cases decodeValue.decodeN co f d n r' <;> simp [Except.toOption]
                simp [hq, hq', cappedN, withinL, hw, hwl, Except.toOption, Except.map]
              · simp only [cappedN, hwl] at h2
                rcases hq : decodeValue.decodeN co f d n r' with e | q
                · simp [hq, hq', cappedN, withinL, hw, hwl, Except.toOption, Except.map]
                · rw [hq] at h2; simp [Except.toOption] at h2
          · have hdv : ∃ e, decodeValue co f d r = .error e := by
              simp only [capped, hw] at hx
              revert hx; cases decodeValue co f d r <;> simp [Except.toOption]
            obtain ⟨e, hdv⟩ := hdv
            rw [hdv]
            rcases hq' : pre3339_decode.decodeN co f n r' with _ | ⟨xs, r''⟩
            · simp [hq', cappedN, Except.toOption]
            · simp [hq', cappedN, withinL, hw, Except.toOption]
    by_cases hd : d > MAX_VALUE_NESTING
    · rw [decodeValue_refuses_past_cap co _ _ _ hd]
      cases hp : pre3339_decode co (f + 1) toks with
      | none => simp [capped, Except.toOption]
      | some p =>
        obtain ⟨v, r⟩ := p
        have hw : within d v = false := by
          cases v <;> simp [within, hd] <;> omega
        simp [capped, hw, Except.toOption]
    · have hd' : d ≤ MAX_VALUE_NESTING := by omega
      have hsc : ∀ (v : V F) (r : List (Tok F)), (∀ l, v ≠ .list l) →
          capped d (some (v, r)) = some (v, r) := by
        intro v r hv
        cases v with
        | list l => exact absurd rfl (hv l)
        | _ => simp [capped, within, hd']
      unfold decodeValue pre3339_decode
      rw [if_neg hd]
      split
      · rename_i tag rest
        by_cases t1 : tag = T_NULL
        · simp only [t1, ite_true]; simp [Except.toOption, capped, within, hd']
        simp only [t1, ite_false]
        by_cases t2 : tag = T_BOOL
        · simp only [t2, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t2, ite_false]
        by_cases t3 : tag = T_INT64
        · simp only [t3, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t3, ite_false]
        by_cases t4 : tag = T_DOUBLE
        · simp only [t4, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t4, ite_false]
        by_cases t5 : tag = T_STRING ∨ tag = T_INTERN + T_STRING
        · simp only [t5, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t5, ite_false]
        by_cases t6 : tag = T_ARRAY
        · simp only [t6, ite_true]
          split
          · rename_i len r
            have := hN len (d + 1) r (by omega)
            cases hq : decodeValue.decodeN co f (d + 1) len r with
            | error e =>
              rw [hq] at this
              cases hq' : pre3339_decode.decodeN co f len r with
              | none => simp [hq, hq', capped, Except.map, Except.toOption]
              | some q =>
                obtain ⟨l, r'⟩ := q
                rw [hq'] at this
                simp only [Except.toOption, cappedN] at this
                have hwl : withinL (d + 1) l = false := by
                  cases hwx : withinL (d + 1) l
                  · rfl
                  · simp [hwx] at this
                simp [hq, hq', capped, within, hwl, Except.map, Except.toOption]
            | ok q =>
              obtain ⟨l, r'⟩ := q
              rw [hq] at this
              cases hq' : pre3339_decode.decodeN co f len r with
              | none => rw [hq'] at this; simp [cappedN, Except.toOption] at this
              | some q' =>
                obtain ⟨l', r''⟩ := q'
                rw [hq'] at this
                simp only [Except.toOption, cappedN] at this
                by_cases hwl : withinL (d + 1) l' = true
                · simp only [hwl, ite_true, Option.some.injEq, Prod.mk.injEq] at this
                  obtain ⟨rfl, rfl⟩ := this
                  simp [hq, hq', capped, within, hwl, hd', Except.map, Except.toOption]
                · simp [hwl] at this
          · simp [Except.toOption, capped]
        simp only [t6, ite_false]
        by_cases t7 : tag = T_POINT
        · simp only [t7, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t7, ite_false]
        by_cases t8 : tag = T_VECTOR_F32
        · simp only [t8, ite_true]
          split
          · rename_i b r
            rcases hv : readVec co b with _ | v <;> simp [hv, Except.toOption, capped, within, hd']
          · simp [Except.toOption, capped]
        simp only [t8, ite_false]
        by_cases t9 : tag = T_DATETIME
        · simp only [t9, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t9, ite_false]
        by_cases t10 : tag = T_DATE
        · simp only [t10, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t10, ite_false]
        by_cases t11 : tag = T_TIME
        · simp only [t11, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t11, ite_false]
        by_cases t12 : tag = T_DURATION
        · simp only [t12, ite_true]; split <;> simp [Except.toOption, capped, within, hd']
        simp only [t12, ite_false]
        simp [Except.toOption, capped]
      · simp [Except.toOption, capped]

theorem toOption_some {α ε : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e <;> simp_all [Except.toOption]

/-! ## Encoded values -/

/-- **Round-trip below the cap**: a storable value `within` the cap decodes to itself. -/
theorem decodeValue_encode (fuel d : Nat) (v : V F) (rest : List (Tok F)) (hs : sizeOf v < fuel)
    (hst : Storable v) (hw : within d v = true) :
    decodeValue co fuel d (encode co v ++ rest) = .ok (v, rest) := by
  apply toOption_some
  rw [decodeValue_agrees, pre3339_decode_encode co fuel v rest hs hst]
  simp [capped, hw]

theorem decode_encode (fuel : Nat) (v : V F) (rest : List (Tok F)) (hs : sizeOf v < fuel)
    (hst : Storable v) (hw : within 0 v = true) :
    decode co fuel (encode co v ++ rest) = .ok (v, rest) :=
  decodeValue_encode co fuel 0 v rest hs hst hw

/-- **Refused past the cap**: a storable value that is not `within` the cap is the
nesting error, not some other failure. -/
theorem decodeValue_encode_deep : ∀ (fuel d : Nat) (v : V F) (rest : List (Tok F)), sizeOf v < fuel →
    Storable v → within d v = false → decodeValue co fuel d (encode co v ++ rest) = .error .nesting := by
  intro fuel
  induction fuel with
  | zero => intro d v rest h; omega
  | succ f ih =>
    intro d v rest hs hst hw
    by_cases hd : d > MAX_VALUE_NESTING
    · exact decodeValue_refuses_past_cap co _ _ _ hd
    have hd' : d ≤ MAX_VALUE_NESTING := by omega
    have hN : ∀ (d' : Nat) (l : List (V F)) (r : List (Tok F)), sizeOf l < f → StorableList l →
        withinL d' l = false → decodeValue.decodeN co f d' l.length (encodeList co l ++ r) = .error .nesting := by
      intro d' l
      induction l with
      | nil => intro r _ _ h; simp [withinL] at h
      | cons x xs ihl =>
        intro r hsz hsl hwl
        simp only [StorableList] at hsl
        simp only [encodeList, List.length_cons, List.append_assoc, decodeValue.decodeN]
        by_cases hx : within d' x = true
        · rw [decodeValue_encode co f d' x _ (by simp at hsz; omega) hsl.1 hx]
          have hxs : withinL d' xs = false := by simpa [withinL, hx] using hwl
          simp [ihl r (by simp at hsz; omega) hsl.2 hxs, Except.map]
        · rw [ih d' x _ (by simp at hsz; omega) hsl.1 (by simpa using hx)]
    cases v with
    | list l =>
      simp only [Storable] at hst
      have hwl : withinL (d + 1) l = false := by simpa [within, hd'] using hw
      simp only [encode, List.append_assoc, List.cons_append, List.nil_append]
      unfold decodeValue
      simp [hd, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY,
        hN (d + 1) l rest (by simp at hs; omega) hst hwl, Except.map]
    | _ => simp [within, hd'] at hw

/-! ## Depth -/

mutual
/-- The doc-comment's nesting measure on lists (`[]` is 1, `[[]]` is 2, scalars 0;
maps are not storable and are counted as 0 here). -/
def ldepth : V F → Nat
  | .list l => maxDepth l + 1
  | _ => 0
def maxDepth : List (V F) → Nat
  | [] => 0
  | x :: xs => max (ldepth x) (maxDepth xs)
end

theorem within_nonlist (d : Nat) (v : V F) (h : ∀ l, v ≠ .list l) : within d v = decide (d ≤ MAX_VALUE_NESTING) := by
  cases v <;> simp_all [within]

/-- `depth ≤ MAX - d` ⇒ `within d`: nothing at depth ≤ MAX is refused. -/
theorem within_of_depth : ∀ (n : Nat) (v : V F) (d : Nat), sizeOf v < n → d + ldepth v ≤ MAX_VALUE_NESTING →
    within d v = true := by
  intro n
  induction n with
  | zero => intro v d h; omega
  | succ f ih =>
    intro v d hs h
    have hL : ∀ (l : List (V F)) (d : Nat), sizeOf l < f → d + maxDepth l ≤ MAX_VALUE_NESTING → withinL d l = true := by
      intro l
      induction l with
      | nil => intro _ _ _; rfl
      | cons x xs ihl =>
        intro d hsz hd
        simp only [maxDepth] at hd
        simp only [withinL, Bool.and_eq_true]
        exact ⟨ih x d (by simp at hsz; omega) (by omega), ihl d (by simp at hsz; omega) (by omega)⟩
    cases v with
    | list l =>
      simp only [ldepth] at h
      simp only [within, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨by omega, hL l (d + 1) (by simp at hs; omega) (by omega)⟩
    | _ => simp [within, ldepth] at h ⊢; omega

/-- `depth ≥ MAX + 2 - d` ⇒ refused. (Exactly `MAX + 1` is the one ambiguous depth: an
innermost `[]` there is accepted, an innermost element is not — `within_nest`.) -/
theorem not_within_of_depth : ∀ (n : Nat) (v : V F) (d : Nat), sizeOf v < n →
    d + ldepth v ≥ MAX_VALUE_NESTING + 2 → within d v = false := by
  intro n
  induction n with
  | zero => intro v d h; omega
  | succ f ih =>
    intro v d hs h
    have hL : ∀ (l : List (V F)) (d : Nat), l ≠ [] → sizeOf l < f → d + maxDepth l ≥ MAX_VALUE_NESTING + 2 →
        withinL d l = false := by
      intro l
      induction l with
      | nil => intro d h; exact absurd rfl h
      | cons x xs ihl =>
        intro d _ hsz hd
        simp only [maxDepth] at hd
        simp only [withinL, Bool.and_eq_false_iff]
        by_cases hx : ldepth x ≥ maxDepth xs
        · exact Or.inl (ih x d (by simp at hsz; omega) (by omega))
        · have hne : xs ≠ [] := by rintro rfl; simp [maxDepth] at hx
          exact Or.inr (ihl d hne (by simp at hsz; omega) (by omega))
    cases v with
    | list l =>
      simp only [ldepth] at h
      by_cases hl : l = []
      · subst hl; simp [within, maxDepth] at h ⊢; omega
      · simp only [within, Bool.and_eq_false_iff, decide_eq_false_iff_not]
        by_cases hd : d ≤ MAX_VALUE_NESTING
        · exact Or.inr (hL l (d + 1) hl (by simp at hs; omega) (by omega))
        · exact Or.inl hd
    | _ => simp [ldepth, within] at h ⊢; omega

/-- `n` lists around `x` (the Rust tests' `nested(depth)` / `nested_tape(depth)`). -/
def nest : Nat → V F → V F
  | 0, x => x
  | k + 1, x => .list [nest k x]

theorem within_nest (k d : Nat) (x : V F) :
    within d (nest k x) = (decide (d + k ≤ MAX_VALUE_NESTING + 1) && within (d + k) x) := by
  induction k generalizing d with
  | zero =>
    simp only [nest, Nat.add_zero]
    by_cases h : d ≤ MAX_VALUE_NESTING + 1
    · simp [h]
    · cases x <;> simp [within, h] <;> omega
  | succ k ih =>
    simp only [nest, within, withinL, Bool.and_true, ih]
    rw [show d + 1 + k = d + (k + 1) by omega]
    by_cases h : d ≤ MAX_VALUE_NESTING
    · simp [h]
    · have : ¬ d + (k + 1) ≤ MAX_VALUE_NESTING + 1 := by omega
      simp [h, this]

/-- The Rust tests `rdb_decode_accepts_the_cap` / `rdb_decode_refuses_one_past_the_cap`:
`k` arrays around a scalar decode iff `k ≤ MAX`. -/
theorem within_nest_scalar (k : Nat) (x : V F) (hx : ∀ l, x ≠ .list l) :
    within 0 (nest k x) = decide (k ≤ MAX_VALUE_NESTING) := by
  rw [within_nest, within_nonlist _ _ hx]
  by_cases h : k ≤ MAX_VALUE_NESTING
  · simp [h]; omega
  · simp [h]

/-- The one leniency: `MAX` arrays around `[]` (depth `MAX + 1`) still decode. Harmless —
the recursion is still bounded and no write accepts it (`validateProperty_depth`). -/
theorem within_nest_empty : within 0 (nest MAX_VALUE_NESTING (.list ([] : List (V F)))) = true ∧
    ldepth (nest MAX_VALUE_NESTING (.list ([] : List (V F)))) = MAX_VALUE_NESTING + 1 := by
  have hd : ∀ k, ldepth (nest k (.list ([] : List (V F)))) = k + 1 := by
    intro k; induction k with
    | zero => rfl
    | succ k ih => simp [nest, ldepth, maxDepth, ih]
  refine ⟨?_, hd _⟩
  rw [within_nest]; simp [within, withinL]

/-! ## Property writes (pending.rs:69-131) -/

/-- `is_valid_property` (pending.rs:69): a non-list primitive, or NULL when allowed. -/
def isValidProperty : V F → Bool → Bool
  | .null, allowNull => allowNull
  | .bool _, _ | .int _, _ | .float _, _ | .str _, _ | .point _ _, _ | .vec _, _
  | .datetime _, _ | .date _, _ | .time _, _ | .duration _, _ => true
  | _, _ => false

/-- The two error strings: the nesting message (pending.rs:103) and `INVALID_PROPERTY_MSG`. -/
inductive PErr | nesting | invalid
  deriving DecidableEq

mutual
/-- The `try_for_each` body of `validate_list_property` (pending.rs:108-112) at nesting
`depth`; a nested list is `validate_list_property(inner, depth + 1)`, whose depth check is
inlined here (`validateListProperty` is the same check around `validateItems`). -/
def validateItems : Nat → List (V F) → Except PErr Unit
  | _, [] => .ok ()
  | depth, x :: xs => match validateItem depth x with
    | .ok () => validateItems depth xs
    | .error e => .error e
def validateItem : Nat → V F → Except PErr Unit
  | depth, .list inner =>
    if depth + 1 > MAX_VALUE_NESTING then .error .nesting else validateItems (depth + 1) inner
  | _, v => if isValidProperty v false then .ok () else .error .invalid
end

/-- `validate_list_property(items, depth)` (pending.rs:98). -/
def validateListProperty (items : List (V F)) (depth : Nat) : Except PErr Unit :=
  if depth > MAX_VALUE_NESTING then .error .nesting else validateItems depth items

/-- `validate_property` (pending.rs:115); `validate_node_property` (:124) and
`validate_relationship_property` (:129) are this function. -/
def validateProperty : V F → Except PErr Unit
  | .list items => validateListProperty items 1
  | v => if isValidProperty v true then .ok () else .error .invalid

def validateNodeProperty (v : V F) : Except PErr Unit := validateProperty v
def validateRelationshipProperty (v : V F) : Except PErr Unit := validateProperty v

theorem validateNode_eq (v : V F) : validateNodeProperty v = validateProperty v ∧
    validateRelationshipProperty v = validateProperty v := ⟨rfl, rfl⟩

mutual
/-- Every vector fits `Encode<19>`'s u32 dimension (needed by `Storable`; a property
write does not check it — a 2³²-element vector is 16 GiB). -/
def VecsFit : V F → Prop
  | .list l => VecsFitL l
  | .vec v => v.length < 2 ^ 32
  | _ => True
def VecsFitL : List (V F) → Prop
  | [] => True
  | x :: xs => VecsFit x ∧ VecsFitL xs
end

/-- What an accepted list element is: at depth `d` (`d ≤ MAX`), storable modulo vectors,
and `d + depth ≤ MAX`. -/
theorem validateItem_ok : ∀ (n : Nat) (v : V F) (d : Nat), sizeOf v < n → validateItem d v = .ok () →
    d + ldepth v ≤ MAX_VALUE_NESTING ∨ (∀ l, v ≠ .list l) := by
  intro n
  induction n with
  | zero => intro v d h; omega
  | succ f ih =>
    intro v d hs h
    have hL : ∀ (l : List (V F)) (d : Nat), sizeOf l < f → validateItems d l = .ok () →
        ∀ x ∈ l, d + ldepth x ≤ MAX_VALUE_NESTING ∨ (∀ l, x ≠ .list l) := by
      intro l
      induction l with
      | nil => intro _ _ _ x hx; simp at hx
      | cons y ys ihl =>
        intro d hsz hok x hx
        simp only [validateItems] at hok
        split at hok
        · rename_i hy
          rcases List.mem_cons.1 hx with rfl | hx
          · exact ih x d (by simp at hsz; omega) hy
          · exact ihl d (by simp at hsz; omega) hok x hx
        · cases hok
    cases v with
    | list l =>
      left
      simp only [validateItem] at h
      split at h
      · cases h
      · rename_i hd
        have hm : ∀ x ∈ l, d + 1 + ldepth x ≤ MAX_VALUE_NESTING := by
          intro x hx
          rcases hL l (d + 1) (by simp at hs; omega) h x hx with h1 | h1
          · exact h1
          · cases x with
            | list l' => exact absurd rfl (h1 l')
            | _ => simp [ldepth]; omega
        have : ∀ (l : List (V F)), (∀ x ∈ l, d + 1 + ldepth x ≤ MAX_VALUE_NESTING) →
            d + 1 + maxDepth l ≤ MAX_VALUE_NESTING := by
          intro l; induction l with
          | nil => intro _; simp [maxDepth]; omega
          | cons y ys ihy =>
            intro hy
            simp only [maxDepth]
            have := hy y (by simp); have := ihy (fun x hx => hy x (by simp [hx])); omega
        have := this l hm
        simp only [ldepth]; omega
    | _ => right; intro l h; cases h

theorem validateItems_storable : ∀ (n : Nat) (l : List (V F)) (d : Nat), sizeOf l < n →
    validateItems d l = .ok () → VecsFitL l → StorableList l := by
  intro n
  induction n with
  | zero => intro l d h; omega
  | succ f ih =>
    intro l d hs h hv
    cases l with
    | nil => trivial
    | cons x xs =>
      simp only [validateItems] at h
      simp only [VecsFitL] at hv
      split at h
      · rename_i hx
        refine ⟨?_, ih xs d (by simp at hs; omega) h hv.2⟩
        cases x with
        | list l' =>
          simp only [validateItem] at hx
          split at hx
          · cases hx
          · exact ih l' (d + 1) (by simp at hs; omega) hx hv.1
        | vec v => exact hv.1
        | map _ | node _ | rel _ | path _ => simp [validateItem, isValidProperty] at hx
        | _ => trivial
      · cases h

/-- **Accepted ⇒ within the cap**: every value a property write accepts has
`depth ≤ MAX_VALUE_NESTING`. -/
theorem validateProperty_depth (v : V F) (h : validateProperty v = .ok ()) :
    ldepth v ≤ MAX_VALUE_NESTING := by
  cases v with
  | list l =>
    simp only [validateProperty, validateListProperty] at h
    split at h
    · cases h
    · rename_i hd
      have := validateItem_ok (sizeOf (V.list l : V F) + 1) (.list l) 0 (by omega)
        (by simp only [validateItem]; rw [if_neg (by omega)]; exact h)
      rcases this with h1 | h1
      · omega
      · exact absurd rfl (h1 l)
  | _ => simp [ldepth]

theorem validateProperty_storable (v : V F) (h : validateProperty v = .ok ()) (hv : VecsFit v) :
    Storable v := by
  cases v with
  | list l =>
    simp only [validateProperty, validateListProperty] at h
    split at h
    · cases h
    · exact validateItems_storable (sizeOf l + 1) l 1 (by omega) h hv
  | vec v => exact hv
  | map _ | node _ | rel _ | path _ => simp [validateProperty, isValidProperty] at h
  | _ => trivial

/-- **The save/load guarantee**: every value a property write accepts (with vectors
that fit the u32 dimension) round-trips through `Encode<19>` / `Decode<19>`. -/
theorem stored_roundtrip (v : V F) (rest : List (Tok F)) (h : validateProperty v = .ok ())
    (hv : VecsFit v) : decode co (sizeOf v + 1) (encode co v ++ rest) = .ok (v, rest) :=
  decode_encode co _ v rest (by omega) (validateProperty_storable v h hv)
    (within_of_depth (sizeOf v + 1) v 0 (by omega) (by have := validateProperty_depth v h; omega))

/-- `validate_list_property`'s cut on `k` lists around a primitive: accepted iff
`k ≤ MAX` (the tests `a_property_at_the_cap_is_valid` / `…_past_the_cap_is_refused`). -/
theorem validateItems_nest (k d : Nat) (x : V F) (hx : ∀ l, x ≠ .list l) (hp : isValidProperty x false = true)
    (hd : d ≤ MAX_VALUE_NESTING) :
    validateItems d [nest k x] = if d + k ≤ MAX_VALUE_NESTING then .ok () else .error .nesting := by
  induction k generalizing d with
  | zero =>
    have : validateItem d x = .ok () := by
      cases x with
      | list l => exact absurd rfl (hx l)
      | _ => simp_all [validateItem]
    simp [nest, validateItems, this, hd]
  | succ k ih =>
    have e2 : validateItems d [nest (k + 1) x] = match validateItem d (.list [nest k x]) with
        | .ok () => validateItems d [] | .error e => .error e := rfl
    have e3 : validateItem d (.list [nest k x]) =
        if d + 1 > MAX_VALUE_NESTING then .error .nesting else validateItems (d + 1) [nest k x] := rfl
    rw [e2, e3]
    by_cases h : d + 1 > MAX_VALUE_NESTING
    · have : ¬ d + (k + 1) ≤ MAX_VALUE_NESTING := by omega
      simp [h, this]
    · rw [if_neg h, ih (d + 1) (by omega), show d + 1 + k = d + (k + 1) by omega]
      by_cases h2 : d + (k + 1) ≤ MAX_VALUE_NESTING <;> simp [h2, validateItems]

/-- A property of `k` lists around a primitive is accepted iff `k ≤ MAX`; past it the
error is the nesting message. -/
theorem validateProperty_nest (k : Nat) (x : V F) (hx : ∀ l, x ≠ .list l) (hp : isValidProperty x false = true) :
    validateProperty (nest (k + 1) x) = if k + 1 ≤ MAX_VALUE_NESTING then .ok () else .error .nesting := by
  simp only [nest, validateProperty, validateListProperty]
  rw [if_neg (by decide), validateItems_nest k 1 x hx hp (by decide), Nat.add_comm 1 k]

/-! ## History (before #3339) -/

/-- `is_valid_property` before #3339 (pending.rs:62 at 4a4f2a72f): lists of any depth. -/
def pre3339_isValidProperty : V F → Bool → Bool
  | .null, allowNull => allowNull
  | .list l, _ => l.attach.all (fun ⟨x, _⟩ => pre3339_isValidProperty x false)
  | v, _ => isValidProperty v false

/-- **Historical (#2722)**: any nesting was a valid property, so `reduce(a = 1, i IN
range(1, n) | [a])` could store a value whose drop / hash / RDB save recursed `n` frames;
and the old decoder took any nesting back (one stack frame per level, the advisory's
`GRAPH.RESTORE` payload). Fixed by #3339 (b080a44c1): `validateProperty_nest`,
`decodeValue_refuses_past_cap`, `decode_reach`. -/
theorem pre3339_isValid_any_depth (k : Nat) :
    pre3339_isValidProperty (nest k (.int 1 : V F)) true = true := by
  have : ∀ k b, pre3339_isValidProperty (nest k (.int 1 : V F)) b = true := by
    intro k; induction k with
    | zero => intro b; simp [nest, pre3339_isValidProperty, isValidProperty]
    | succ k ih => intro b; simp [nest, pre3339_isValidProperty, ih]
  exact this k true

theorem pre3339_decode_any_depth (k : Nat) :
    pre3339_decode co (sizeOf (nest k (.null : V F)) + 1) (encode co (nest k .null)) = some (nest k .null, []) := by
  have hs : ∀ k, Storable (nest k (.null : V F)) := by
    intro k; induction k with
    | zero => trivial
    | succ k ih => exact ⟨ih, trivial⟩
  simpa using pre3339_decode_encode co _ (nest k .null) [] (by omega) (hs k)

/-- …while the decoder now refuses the same input past the cap, with the nesting error. -/
theorem decode_nest_deep (k : Nat) (hk : k > MAX_VALUE_NESTING) :
    decode co (sizeOf (nest k (.null : V F)) + 1) (encode co (nest k .null)) = .error .nesting := by
  have hs : ∀ k, Storable (nest k (.null : V F)) := by
    intro k; induction k with
    | zero => trivial
    | succ k ih => exact ⟨ih, trivial⟩
  have := decodeValue_encode_deep co (sizeOf (nest k (.null : V F)) + 1) 0 (nest k .null) [] (by omega) (hs k)
    (by rw [within_nest_scalar k _ (by intro l h; cases h)]; simp; omega)
  simpa [decode] using this

end ValueMath
