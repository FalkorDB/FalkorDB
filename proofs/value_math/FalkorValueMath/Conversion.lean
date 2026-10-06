import FalkorValueMath.MathFns
/-!
# `graph/src/runtime/functions/conversion.rs` against C `numeric_funcs.c`

Strings are `List Char`. Parsers modelled:

* `parseI64` — Rust `<i64 as FromStr>::from_str` (core `from_str_radix`, radix 10):
  optional single `+`/`-`, then ≥ 1 ASCII digit, nothing else, no whitespace; out of range
  is an error.
* `f64Grammar` — the set of strings Rust `<f64 as FromStr>` accepts (core
  `dec2flt/parse.rs`): `[+-]?` then `inf`/`infinity`/`nan` (ASCII case-insensitive) or
  `digits [. digits] | . digits` (≥ 1 digit) with optional `[eE][+-]?digits+`. For
  integer literals the parsed value is exact-rounded (`roundF`, dec2flt is correctly
  rounded); otherwise the value is abstract.
* `cStrtoll` — C `strtoll(s, &end, 10)` + `end[0] != '\0' || errno == ERANGE → null`:
  skips leading `isspace`, optional sign, digits; saturates and sets ERANGE on overflow.

C `AR_TOINTEGER` (numeric_funcs.c, master) takes the `strtoll` path when the string has
no `.`, and the `strtod` + `floor` path otherwise. C `AR_TOFLOAT` uses `strtof` —
**float32** (so C `toFloat('0.1')` is `0.100000001490116`; Rust is exact).
-/

namespace ValueMath

def isDigit (c : Char) : Bool := '0' ≤ c && c ≤ '9'
def digitVal (c : Char) : Nat := c.toNat - '0'.toNat

/-- Value of a digit string (most significant first). -/
def digitsVal (ds : List Char) : Nat := ds.foldl (fun acc c => acc * 10 + digitVal c) 0

def allDigits (ds : List Char) : Bool := !ds.isEmpty && ds.all isDigit

/-- Rust `i64::from_str`. -/
def parseI64 : List Char → Option Int
  | [] => none
  | c :: rest =>
    let (neg, ds) := if c = '-' then (true, rest) else if c = '+' then (false, rest) else (false, c :: rest)
    if allDigits ds then
      let v : Int := if neg then -(digitsVal ds : Int) else digitsVal ds
      if InI64 v then some v else none
    else none

/-- C `isspace` in the "C" locale. -/
def cIsSpace (c : Char) : Bool := c = ' ' || c = '\t' || c = '\n' || c = '\x0b' || c = '\x0c' || c = '\r'

/-- C `strtoll(s, &end, 10)` followed by C `AR_TOINTEGER`'s acceptance test. -/
def cStrtoll (s : List Char) : Option Int :=
  let s' := s.dropWhile cIsSpace
  match s' with
  | [] => none
  | c :: rest =>
    let (neg, ds) := if c = '-' then (true, rest) else if c = '+' then (false, rest) else (false, c :: rest)
    if allDigits ds then
      let v : Int := if neg then -(digitsVal ds : Int) else digitsVal ds
      if InI64 v then some v else none   -- ERANGE → null
    else none

/-! ## Rust `f64::from_str` acceptance -/

def lowerAscii (c : Char) : Char := if 'A' ≤ c && c ≤ 'Z' then Char.ofNat (c.toNat + 32) else c

def isSpecial (s : List Char) : Bool :=
  let l := s.map lowerAscii
  l = ['i', 'n', 'f'] || l = ['i', 'n', 'f', 'i', 'n', 'i', 't', 'y'] || l = ['n', 'a', 'n']

/-- `digits [. digits]` / `. digits` with at least one digit, then an optional exponent. -/
def decimalBody (s : List Char) : Bool :=
  let intPart := s.takeWhile isDigit
  let r1 := s.dropWhile isDigit
  let (fracPart, r2) := match r1 with
    | '.' :: r => (r.takeWhile isDigit, r.dropWhile isDigit)
    | r => ([], r)
  (intPart.length + fracPart.length > 0) &&
  (match r2 with
   | [] => true
   | e :: r => (e = 'e' || e = 'E') &&
      (match r with
       | '+' :: ds | '-' :: ds => allDigits ds
       | ds => allDigits ds)
   )

def f64Grammar (s : List Char) : Bool :=
  let body := match s with
    | '+' :: r | '-' :: r => r
    | r => r
  isSpecial body || decimalBody body

/-! ## `toInteger(String)` (conversion.rs:46-75) -/

/-- `s.strip_prefix(['-', '+']).unwrap_or(s)` (conversion.rs:56): drop one leading sign. -/
def stripSign : List Char → List Char
  | c :: rest => if c = '-' ∨ c = '+' then rest else c :: rest
  | [] => []

/-- Rust `value_to_integer` on a string. `f64Val` is `s.parse::<f64>()` (abstract),
`floorI` is `f.floor()` as an exact integer (abstract, `none` when non-finite). -/
def rustToIntegerStr (f64Val : List Char → Option Int) (s : List Char) : Option Int :=
  if s.isEmpty then none else                                     -- :48-50
  match parseI64 s with                                           -- :51
  | some i => some i
  | none =>
    if allDigits (stripSign s) then none else                     -- :56-59 (#2989)
    match f64Val s with                                           -- :60-73
    | some fl => if fl ≥ TWO63 ∨ fl < -TWO63 then none else some fl
    | none => none

/-- For signed integer literals Rust's `f64` parse is `roundF` of the exact value. -/
def intLitF64 (s : List Char) : Option Int :=
  match s with
  | [] => none
  | c :: rest =>
    let (neg, ds) := if c = '-' then (true, rest) else if c = '+' then (false, rest) else (false, c :: rest)
    if allDigits ds then some (roundF (if neg then -(digitsVal ds : Int) else digitsVal ds)) else none

/-- C `AR_TOINTEGER` on a string with no `.` (the `strtoll` branch). -/
def cToIntegerNoDot (s : List Char) : Option Int := if s.isEmpty then none else cStrtoll s

/-- On every string Rust's `i64` parser accepts, C's `strtoll` path gives the same value. -/
theorem parseI64_sub_cStrtoll (s : List Char) (v : Int) (h : parseI64 s = some v) :
    cStrtoll s = some v := by
  cases s with
  | nil => simp [parseI64] at h
  | cons c rest =>
    have hc : cIsSpace c = false := by
      cases hsp : cIsSpace c
      · rfl
      · simp only [cIsSpace, Bool.or_eq_true, decide_eq_true_eq] at hsp
        rcases hsp with (((((rfl|rfl)|rfl)|rfl)|rfl)|rfl) <;>
          simp [parseI64, allDigits, isDigit] at h
    simp only [cStrtoll, List.dropWhile, hc]
    simpa [parseI64] using h

/-- Hence Rust and C agree on every in-range integer literal. -/
theorem toInteger_int_literals_agree (f64Val : List Char → Option Int) (s : List Char) (v : Int)
    (h : parseI64 s = some v) :
    rustToIntegerStr f64Val s = some v ∧ cToIntegerNoDot s = some v := by
  have hne : s.isEmpty = false := by cases s <;> simp_all [parseI64]
  exact ⟨by simp [rustToIntegerStr, hne, h], by simp [cToIntegerNoDot, hne, parseI64_sub_cStrtoll s v h]⟩

/-- Leading whitespace: Rust rejects, C skips it (live: `toInteger(' 12')` Rust null,
C 12). Also seen in proofs/functions_temporal. -/
theorem leading_space : parseI64 [' ', '1', '2'] = none ∧ cStrtoll [' ', '1', '2'] = some 12 := by decide

/-- `'1e3'`: Rust's f64 fallback accepts it, C's no-`.` branch does not (also seen). -/
theorem exp_literal : f64Grammar ['1', 'e', '3'] = true ∧ cStrtoll ['1', 'e', '3'] = none := by decide

/-- `f64` grammar spot checks (all match the live `toFloat` results). -/
example : f64Grammar ['5', '.'] ∧ f64Grammar ['.', '5'] ∧ f64Grammar ['-', 'i', 'n', 'f', 'i', 'n', 'i', 't', 'y'] ∧
    f64Grammar ['N', 'a', 'N'] ∧ !f64Grammar ['.'] ∧ !f64Grammar ['1', 'e'] ∧
    !f64Grammar [' ', '1'] ∧ !f64Grammar ['0', 'x', '1', '0'] ∧ !f64Grammar ['1', '_', '0', '0', '0'] := by decide

/-! ### The negative-overflow window -/

theorem bitLen_of_range {n : Nat} (h1 : 2 ^ 63 ≤ n) (h2 : n < 2 ^ 64) : bitLen n = 64 := by
  cases n with
  | zero => omega
  | succ k =>
    simp only [bitLen]
    have a := (Nat.log2_lt (n := k + 1) (k := 64) (by omega)).2 h2
    have b : ¬ Nat.log2 (k + 1) < 63 := fun hc => by
      have := (Nat.log2_lt (n := k + 1) (k := 63) (by omega)).1 hc; omega
    omega

/-- Every integer in `[2^63, 2^63 + 1024]` rounds to the double `2^63` (ulp 2048, the
tie at +1024 goes to the even significand 2^52). -/
theorem roundNat_window {n : Nat} (h1 : 2 ^ 63 ≤ n) (h2 : n ≤ 2 ^ 63 + 1024) : roundNat n = 2 ^ 63 := by
  unfold roundNat
  rw [bitLen_of_range h1 (by omega)]
  simp only [show ¬ (64 ≤ 53) by omega, if_false, show 64 - 53 = 11 by rfl, show 11 - 1 = 10 by rfl]
  have hq : n / 2 ^ 11 = 2 ^ 52 ∨ (n = 2 ^ 63 + 1024 ∧ n / 2 ^ 11 = 2 ^ 52) := by left; omega
  have hq' : n / 2 ^ 11 = 2 ^ 52 := by omega
  rw [hq']
  have hr : n % 2 ^ 11 ≤ 2 ^ 10 := by omega
  have hev : (2 : Nat) ^ 52 % 2 = 0 := by decide
  simp only [hev]
  split
  · rename_i hc; omega
  · rfl

/-- **#2955 fixed** (`ac5c27b76`, PR #2989): the 1024 integer strings from
`-9223372036854775809` down to `-9223372036854776832` are out of `i64` range and Rust's
`toInteger` now returns null for them, as C does (strtoll ERANGE). Historical: before
`ac5c27b76` the failed `parse::<i64>` fell through to the `f64` path, which rounds every
one of them to exactly `-2^63` and passed the strict `floored < i64::MIN as f64` guard, so
Rust returned `i64::MIN` (`roundNat_window` is that rounding; the theorem was
`toInteger_neg_overflow_window`, now proved with the opposite conclusion). -/
theorem toInteger_neg_overflow_null (s : List Char) (ds : List Char) (hs : s = '-' :: ds)
    (hd : allDigits ds = true) (h1 : 2 ^ 63 + 1 ≤ digitsVal ds) (_h2 : digitsVal ds ≤ 2 ^ 63 + 1024) :
    rustToIntegerStr intLitF64 s = none ∧ cToIntegerNoDot s = none := by
  subst hs
  have hnot : ¬ InI64 (-(digitsVal ds : Int)) := by unfold InI64; omega
  have hp : parseI64 ('-' :: ds) = none := by simp [parseI64, hd, hnot]
  refine ⟨?_, ?_⟩
  · simp [rustToIntegerStr, hp, stripSign, hd]
  · simp [cToIntegerNoDot, cStrtoll, cIsSpace, hd, hnot]

/-- The concrete string from the #2955 report (live: Rust and C both null since `ac5c27b76`). -/
theorem toInteger_minus_2p63_minus_1 :
    rustToIntegerStr intLitF64 ['-', '9', '2', '2', '3', '3', '7', '2', '0', '3', '6', '8', '5', '4', '7', '7', '5', '8', '0', '9'] = none ∧
    cToIntegerNoDot ['-', '9', '2', '2', '3', '3', '7', '2', '0', '3', '6', '8', '5', '4', '7', '7', '5', '8', '0', '9'] = none := by decide

/-- A failed `i64` parse of a sign-digits string means the value is out of range. -/
theorem parseI64_none_out (c : Char) (rest : List Char) (hd : allDigits (stripSign (c :: rest)) = true)
    (hp : parseI64 (c :: rest) = none) :
    ¬ InI64 (if c = '-' then -(digitsVal rest : Int) else
      if c = '+' then (digitsVal rest : Int) else (digitsVal (c :: rest) : Int)) := by
  intro hin
  by_cases hm : c = '-'
  · subst hm; simp [stripSign] at hd; simp [parseI64, hd] at hp; simp at hin; exact hp hin
  · by_cases hpl : c = '+'
    · subst hpl; simp [stripSign] at hd; simp [parseI64, hd] at hp; simp at hin; exact hp hin
    · simp [stripSign, hm, hpl] at hd; simp [parseI64, hm, hpl, hd] at hp; simp [hm, hpl] at hin
      exact hp hin

/-- **Correctness (#2989)**: on every integer literal — an optional `+`/`-` followed by
≥ 1 ASCII digits, any length — Rust's `toInteger` equals C's `strtoll` branch, for every
`f64` parser: in range both give the value, out of range both give null. -/
theorem toInteger_int_literal_agrees_c (f64Val : List Char → Option Int) (s : List Char)
    (hd : allDigits (stripSign s) = true) :
    rustToIntegerStr f64Val s = cToIntegerNoDot s := by
  cases hp : parseI64 s with
  | some v => rw [(toInteger_int_literals_agree f64Val s v hp).1, (toInteger_int_literals_agree f64Val s v hp).2]
  | none =>
    cases s with
    | nil => simp [rustToIntegerStr, cToIntegerNoDot]
    | cons c rest =>
      have hout := parseI64_none_out c rest hd hp
      have hc : cIsSpace c = false := by
        cases hsp : cIsSpace c
        · rfl
        · exfalso
          simp only [cIsSpace, Bool.or_eq_true, decide_eq_true_eq] at hsp
          rcases hsp with (((((rfl|rfl)|rfl)|rfl)|rfl)|rfl) <;> simp [stripSign, allDigits, isDigit] at hd
      have hr : rustToIntegerStr f64Val (c :: rest) = none := by simp [rustToIntegerStr, hp, hd]
      rw [hr]
      by_cases hm : c = '-'
      · subst hm; simp [stripSign] at hd; simp at hout
        simp [cToIntegerNoDot, cStrtoll, hc, hd, hout]
      · by_cases hpl : c = '+'
        · subst hpl; simp [stripSign] at hd; simp at hout
          simp [cToIntegerNoDot, cStrtoll, hc, hd, hout]
        · simp [stripSign, hm, hpl] at hd; simp [hm, hpl] at hout
          simp [cToIntegerNoDot, cStrtoll, hc, hm, hpl, hd, hout]

/-- Positive overflow is null (live: `toInteger('9223372036854775808')` null in both).
Since `ac5c27b76` the digits guard (conversion.rs:56-59) returns null before the `f64`
path; before it, the same rounding sent the value to `2^63 ≥ 2^63`, also null. -/
theorem toInteger_pos_overflow (ds : List Char) (hd : allDigits ds = true)
    (h1 : 2 ^ 63 ≤ digitsVal ds) :
    rustToIntegerStr intLitF64 ds = none := by
  have hnot : ¬ InI64 (digitsVal ds : Int) := by unfold InI64; omega
  have hne : ds ≠ [] := by intro h; subst h; simp [allDigits] at hd
  obtain ⟨c, rest, rfl⟩ : ∃ c rest, ds = c :: rest := by cases ds <;> simp_all
  have hc : isDigit c = true := by simp [allDigits] at hd; exact hd.1
  have hcm : c ≠ '-' := by intro e; subst e; simp [isDigit] at hc
  have hcp : c ≠ '+' := by intro e; subst e; simp [isDigit] at hc
  have hp : parseI64 (c :: rest) = none := by simp [parseI64, hcm, hcp, hd, hnot]
  simp [rustToIntegerStr, hp, stripSign, hcm, hcp, hd]

/-! ## `toInteger(Float)` (conversion.rs:77-84) -/

/-- Rust: non-finite → null, else `floor as i64` (saturating). C: `(int64_t)floor(x)`
(arm64: NaN → 0, ±∞ saturate). At class level: -/
def rustToIntegerFloatNonFinite : FC → Option (Option Int)
  | .nan | .pinf | .ninf => some none
  | _ => none

def cToIntegerFloatNonFinite : FC → Option Int
  | .nan => some 0 | .pinf => some I64MAX | .ninf => some I64MIN
  | _ => none

/-- Divergence on non-finite floats (live: `toInteger(0.0/0)` Rust null, C 0;
`toInteger(1.0/0)` Rust null, C 9223372036854775807). -/
theorem toInteger_nonfinite_diverges :
    rustToIntegerFloatNonFinite .nan = some none ∧ cToIntegerFloatNonFinite .nan = some 0 ∧
    rustToIntegerFloatNonFinite .pinf = some none ∧ cToIntegerFloatNonFinite .pinf = some I64MAX := by
  decide

/-! ## `toBoolean` (conversion.rs:189) -/

def eqIgnoreAsciiCase (a b : List Char) : Bool := a.map lowerAscii == b.map lowerAscii

def toBooleanStr (s : List Char) : Option Bool :=
  if eqIgnoreAsciiCase s ['t', 'r', 'u', 'e'] then some true
  else if eqIgnoreAsciiCase s ['f', 'a', 'l', 's', 'e'] then some false
  else none

theorem toBoolean_roundtrip (b : Bool) : toBooleanStr (boolStr b).toList = some b := by
  cases b <;> decide

example : toBooleanStr ['T', 'R', 'U', 'E'] = some true ∧ toBooleanStr [' ', 't', 'r', 'u', 'e'] = none ∧
    toBooleanStr ['t'] = none := by decide

/-! ## Scalar conversions over `V` and their list forms -/

variable {F : Type}

/-- Everything a conversion needs from the float layer. -/
structure ConvOps (F : Type) where
  parseF : String → Option F           -- `s.parse::<f64>()`
  floorToInt : F → Option Int          -- `if !finite null else floor as i64`
  strToInt : String → Option Int       -- the String arm of `value_to_integer`
  ofInt : Int → F
  display : F → String                 -- `format!("{f}")`
  pointStr : F → F → String
  fmtDatetime : Int → String
  fmtDate : Int → String
  fmtTime : Int → String
  fmtDuration : Int → String

variable (c : ConvOps F)

def optV {α} (f : α → V F) : Option α → V F
  | some a => f a
  | none => .null

/-- `value_to_integer` (conversion.rs:44): never an `Err`. -/
def toInteger : V F → V F
  | .str s => optV .int (c.strToInt s)
  | .int i => .int i
  | .float f => optV .int (c.floorToInt f)
  | .bool b => .int (if b then 1 else 0)
  | _ => .null

/-- `value_to_float` (conversion.rs:103). -/
def toFloat : V F → V F
  | .str s => optV .float (c.parseF s)
  | .float f => .float f
  | .int i => .float (c.ofInt i)
  | _ => .null

/-- `value_to_string` (conversion.rs:125) on a non-empty argument list. -/
def toStr : V F → V F
  | .str s => .str s
  | .int i => .str (toString i)
  | .float f => .str (c.display f)
  | .bool b => .str (boolStr b)
  | .point la lo => .str (c.pointStr la lo)
  | .datetime t => .str (c.fmtDatetime t)
  | .date t => .str (c.fmtDate t)
  | .time t => .str (c.fmtTime t)
  | .duration t => .str (c.fmtDuration t)
  | _ => .null

def toBooleanV : V F → V F
  | .bool b => .bool b
  | .str s => optV .bool (toBooleanStr s.toList)
  | .int n => .bool (n ≠ 0)
  | _ => .null

/-- `toBooleanList` element map (conversion.rs:222-234). -/
def toBooleanElem : V F → V F
  | .bool b => .bool b
  | .str s => if eqIgnoreAsciiCase s.toList ['t', 'r', 'u', 'e'] then .bool true
              else if eqIgnoreAsciiCase s.toList ['f', 'a', 'l', 's', 'e'] then .bool false else .null
  | .int n => .bool (n ≠ 0)
  | _ => .null

/-- `toFloatList` element map (conversion.rs:249-254). -/
def toFloatElem : V F → V F
  | .float f => .float f
  | .int i => .float (c.ofInt i)
  | .str s => optV .float (c.parseF s)
  | _ => .null

def listFn (elem : V F → V F) : V F → V F
  | .list vs => .list (vs.map elem)
  | _ => .null

/-- The list conversions are exactly the element-wise `...OrNull` scalar conversions. -/
theorem toBooleanList_eq (v : V F) : listFn toBooleanElem v = listFn toBooleanV v := by
  cases v with
  | list vs =>
    simp only [listFn, V.list.injEq]
    apply List.map_congr_left
    intro x _
    cases x with
    | str s =>
      simp only [toBooleanElem, toBooleanV, toBooleanStr]
      by_cases ht : eqIgnoreAsciiCase s.toList ['t', 'r', 'u', 'e'] = true <;>
        by_cases hf : eqIgnoreAsciiCase s.toList ['f', 'a', 'l', 's', 'e'] = true <;> simp [ht, hf, optV]
    | _ => rfl
  | _ => rfl

theorem toFloatList_eq (v : V F) : listFn (toFloatElem c) v = listFn (toFloat c) v := by
  cases v with
  | list vs =>
    simp only [listFn, V.list.injEq]
    apply List.map_congr_left
    intro x _; cases x <;> rfl
  | _ => rfl

/-- `toIntegerList` calls `value_to_integer(..).unwrap_or(Null)`; since
`value_to_integer` never returns `Err`, that is the plain map (and inherits every scalar
divergence above, e.g. `toIntegerList([' 3'])` = `[null]`, C `[3]`, live). -/
theorem toIntegerList_eq (vs : List (V F)) :
    listFn (toInteger c) (.list vs) = .list (vs.map (toInteger c)) := rfl

theorem toStringList_eq (vs : List (V F)) :
    listFn (toStr c) (.list vs) = .list (vs.map (toStr c)) := rfl

/-- `toString` of an Integer is its decimal form, and of a String is the String:
`toInteger(toString(i)) = i` for every Integer whose decimal form Rust's parser accepts. -/
theorem toString_toInteger_int (i : Int) (h : c.strToInt (toString i) = some i) :
    toInteger c (toStr c (.int i)) = .int i := by
  simp only [toStr, toInteger]; rw [h]; rfl

/-- `isEmpty` (conversion.rs:175). -/
def isEmpty : V F → Option (V F)
  | .null => some .null
  | .str s => some (.bool s.isEmpty)
  | .list l => some (.bool l.isEmpty)
  | .map m => some (.bool m.isEmpty)
  | _ => none     -- unreachable!()

theorem isEmpty_no_panic (v : V F) (h : Accepts v (.union [.map, .list .any, .string, .null])) :
    (isEmpty v).isSome := by
  rw [← valueOfType_none_iff] at h
  cases v <;> simp_all [isEmpty, vot_union, unionLoop, valueOfType, tagMatch, Ty.isAny, listLoop_none, vot_any]

end ValueMath
