import FalkorValueMath.Conversion
/-!
# `functions/conversion.rs`: `float_to_string` (:299) — PR #3609 (fee0066ab)

`toString(Float)` (#2937): plain `{f:?}` for `0` and `1e-3 <= |f| < 1e7`, Rust's
`{f}` for NaN/±inf, otherwise scientific with an upper-case `E`, the mantissa
replaced by the correctly rounded one at the same digit count when that still
round-trips, and a one-digit mantissa widened to two digits (`{f:.1e}`, kept when
it round-trips) or else padded with `.0`.

The std float formatters (`{:?}`, `{}`, `{:e}`, `{:.N$e}`) and `str::parse::<f64>()
== Ok(f)` are the parameters of `FmtOps` (like `ConvOps.display`); strings are
`List Char`, and `split_once`, the ASCII digit count and `contains('.')` are modelled
exactly. `value_to_string`'s Float arm is `ConvOps.display := floatToString`
(`toStr_float`).
-/

namespace ValueMath.FloatToString

open FalkorCore
open ValueMath

abbrev Str := List Char

/-- `str::split_once(c)`. -/
def splitOnce (c : Char) : Str → Option (Str × Str)
  | [] => none
  | x :: xs => if x = c then some ([], xs) else (splitOnce c xs).map (fun p => (x :: p.1, p.2))

/-- `m.bytes().filter(u8::is_ascii_digit).count()`. -/
def digitCount (s : Str) : Nat := (s.filter Char.isDigit).length

/-- `s.split_once('e').unwrap_or((&s, "0"))`. -/
def splitE (s : Str) : Str × Str := (splitOnce 'e' s).getD (s, ['0'])

structure FmtOps (F : Type) where
  plain : F → Bool                -- `abs == 0.0 || (1e-3..1e7).contains(&abs)`
  isFinite : F → Bool             -- `f.is_finite()`
  debug : F → Str                 -- `format!("{f:?}")`
  display : F → Str               -- `format!("{f}")`
  sci : F → Str                   -- `format!("{f:e}")`
  sciPrec : Nat → F → Str         -- `format!("{f:.prec$e}")`
  roundTrips : Str → F → Bool     -- `s.parse::<f64>() == Ok(f)`

variable {F : Type} (o : FmtOps F)

/-- `sci` after the tie fix (conversion.rs:309-322). -/
def sciFixed (f : F) : Str :=
  let sci0 := o.sci f
  let digits := match splitOnce 'e' sci0 with
    | some (m, _) => digitCount m
    | none => 0
  if digits > 1 && o.roundTrips (o.sciPrec (digits - 1) f) f then o.sciPrec (digits - 1) f
  else sci0

/-- `float_to_string` (conversion.rs:299). -/
def floatToString (f : F) : Str :=
  if o.plain f then o.debug f
  else if !o.isFinite f then o.display f
  else
    let sci := sciFixed o f
    let me := splitE sci
    if '.' ∈ me.1 then me.1 ++ ['E'] ++ me.2
    else
      let two := o.sciPrec 1 f
      if o.roundTrips two f then (splitE two).1 ++ ['E'] ++ (splitE two).2
      else me.1 ++ ['.', '0', 'E'] ++ me.2

theorem fts_plain (f : F) (h : o.plain f = true) : floatToString o f = o.debug f := by
  simp [floatToString, h]

theorem fts_nonfinite (f : F) (h : o.plain f = false) (hf : o.isFinite f = false) :
    floatToString o f = o.display f := by
  simp [floatToString, h, hf]

/-- Outside the plain range a finite float is printed `mantissa ++ "E" ++ exp` with the
mantissa taken from (in order) the fixed `{:e}` form when it has a fractional part, else
`{:.1e}` when it round-trips, else the one-digit mantissa padded with `.0`. -/
theorem fts_sci_spec (f : F) (h : o.plain f = false) (hf : o.isFinite f = true) :
    floatToString o f =
      if '.' ∈ (splitE (sciFixed o f)).1 then
        (splitE (sciFixed o f)).1 ++ ['E'] ++ (splitE (sciFixed o f)).2
      else if o.roundTrips (o.sciPrec 1 f) f then
        (splitE (o.sciPrec 1 f)).1 ++ ['E'] ++ (splitE (o.sciPrec 1 f)).2
      else (splitE (sciFixed o f)).1 ++ ['.', '0', 'E'] ++ (splitE (sciFixed o f)).2 := by
  simp [floatToString, h, hf]

/-- The tie fix: the correctly rounded mantissa at the shortest digit count replaces
the shortest one exactly when it has more than one digit and round-trips. -/
theorem sciFixed_spec (f : F) (m e : Str) (hs : splitOnce 'e' (o.sci f) = some (m, e)) :
    sciFixed o f = if digitCount m > 1 && o.roundTrips (o.sciPrec (digitCount m - 1) f) f
      then o.sciPrec (digitCount m - 1) f else o.sci f := by
  simp [sciFixed, hs]

/-- A finite float outside the plain range always shows a fractional part and an
upper-case `E`, given only that `{:.1e}`'s mantissa has a `.` (std prints `d.de…`). -/
theorem fts_sci_has_fraction (f : F) (h : o.plain f = false) (hf : o.isFinite f = true)
    (htwo : '.' ∈ (splitE (o.sciPrec 1 f)).1) :
    ∃ m e, floatToString o f = m ++ 'E' :: e ∧ '.' ∈ m := by
  rw [fts_sci_spec o f h hf]
  split
  · exact ⟨(splitE (sciFixed o f)).1, (splitE (sciFixed o f)).2, by simp, by assumption⟩
  · split
    · exact ⟨(splitE (o.sciPrec 1 f)).1, (splitE (o.sciPrec 1 f)).2, by simp, htwo⟩
    · exact ⟨(splitE (sciFixed o f)).1 ++ ['.', '0'], (splitE (sciFixed o f)).2, by simp, by simp⟩

/-- `splitOnce` splits at the first occurrence. -/
theorem splitOnce_append (c : Char) (m e : Str) (hm : c ∉ m) :
    splitOnce c (m ++ c :: e) = some (m, e) := by
  induction m with
  | nil => simp [splitOnce]
  | cons x xs ih =>
    simp only [List.mem_cons, not_or] at hm
    simp [splitOnce, Ne.symm hm.1, ih hm.2]

/-- `value_to_string`'s Float arm (conversion.rs:130) is `float_to_string`. -/
theorem toStr_float (c : ConvOps F) (hd : ∀ x, c.display x = String.ofList (floatToString o x))
    (f : F) : toStr c (.float f) = .str (String.ofList (floatToString o f)) := by
  simp [toStr, hd]

/-- The std outputs for `5e-324` (the smallest subnormal): `{:e}` = `5e-324`,
`{:.1e}` = `4.9e-324` (round-trips), so the result is Java's `4.9E-324`; and for `1e20`
the `.0` padding gives `1.0E20` (`{:.1e}` = `1.0e20` round-trips too). -/
def subnormalOps : FmtOps Unit :=
  { plain := fun _ => false, isFinite := fun _ => true, debug := fun _ => [], display := fun _ => []
    sci := fun _ => "5e-324".toList, sciPrec := fun _ _ => "4.9e-324".toList
    roundTrips := fun s _ => s == "4.9e-324".toList || s == "5e-324".toList }

example : floatToString subnormalOps () = "4.9E-324".toList := by decide

def e20Ops : FmtOps Unit :=
  { plain := fun _ => false, isFinite := fun _ => true, debug := fun _ => [], display := fun _ => []
    sci := fun _ => "1e20".toList, sciPrec := fun _ _ => "1.0e20".toList, roundTrips := fun _ _ => true }

example : floatToString e20Ops () = "1.0E20".toList := by decide

end ValueMath.FloatToString
