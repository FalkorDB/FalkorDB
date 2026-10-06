import FalkorTemporal.Fns
/-!
# String parsers of temporal.rs (parse_date_string :148, parse_time_string :272,
# parse_datetime_string :329, parse_duration_string :347)

Control flow followed line by line over Lean `String`; the integer parsers and the chrono
constructors are the abstract `Chrono` fields. `digits_only[..4]` etc. slice bytes of an
ASCII-digit string, i.e. characters: `String.take`/`drop` here.
-/
namespace FalkorTemporal.Fns

variable (C : Chrono)

def digitsOnly (s : String) : String := String.ofList (s.toList.filter Char.isDigit)

/-- `s[a..a+n]` on an ASCII string (bytes = chars). -/
def sub (s : String) (a n : Nat) : String := String.ofList ((s.toList.drop a).take n)

/-- `x.parse::<u32>().map_err(..)?` -/
def pU32 (x err : String) : Except String Nat :=
  match C.parseU32 x with | some n => .ok n | none => .error err
def pI32 (x err : String) : Except String Int :=
  match C.parseI32 x with | some n => .ok n | none => .error err

def ymd (y : Int) (m d : Nat) (s : String) : Except String Int :=
  match C.fromYmd y m d with | some z => .ok z | none => .error s!"Invalid date: {s}"

/-- `parse_date_string` (temporal.rs:148). -/
def parseDate (s : String) : Except String Int :=
  if s.contains 'W' then C.parseWeek s
  else if s.startsWith "-" then .error s!"Unsupported date string: {s}"
  else
    let digits := digitsOnly s
    if s.contains '-' then
      match s.splitOn "-" with
      | [p0] => do let y ← pI32 C p0 s!"Invalid year: {s}"; ymd C y 1 1 s
      | [p0, p1] => do
          let y ← pI32 C p0 s!"Invalid year: {s}"
          let m ← pU32 C p1 s!"Invalid month: {s}"
          ymd C y m 1 s
      | [p0, p1, p2] => do
          let y ← pI32 C p0 s!"Invalid year: {s}"
          let m ← pU32 C p1 s!"Invalid month: {s}"
          let d ← pU32 C p2 s!"Invalid day: {s}"
          ymd C y m d s
      | _ => .error s!"Invalid date string: {s}"
    else
      match digits.length with
      | 4 => do let y ← pI32 C digits s!"Invalid year: {s}"; ymd C y 1 1 s
      | 6 => do
          let y ← pI32 C (sub digits 0 4) s!"Invalid year: {s}"
          let m ← pU32 C ((sub digits 4 2)) s!"Invalid month: {s}"
          ymd C y m 1 s
      | 7 => do
          let y ← pI32 C (sub digits 0 4) s!"Invalid year: {s}"
          let o ← pU32 C ((sub digits 4 3)) s!"Invalid ordinal: {s}"
          match C.fromYo y o with
          | some z => .ok z
          | none => .error s!"Invalid ordinal date: {s}"
      | 8 => do
          let y ← pI32 C (sub digits 0 4) s!"Invalid year: {s}"
          let m ← pU32 C ((sub digits 4 2)) s!"Invalid month: {s}"
          let d ← pU32 C ((sub digits 6 2)) s!"Invalid day: {s}"
          ymd C y m d s
      | _ => .error s!"Invalid date string: {s}"

theorem ymd_ok (y : Int) (m d : Nat) (s : String) (z : Int) (h : ymd C y m d s = .ok z) :
    ValidDate y m d ∧ z = daysFromCivil y m d := by
  unfold ymd at h; split at h
  · rename_i z' hz; cases h; exact C.fromYmd_spec _ _ _ _ hz
  · cases h

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- Soundness of `parse_date_string`: a successful parse is a valid calendar date, or an
ordinal date chrono accepted, or the ISO-week path. -/
theorem parseDate_ok (s : String) (z : Int) (h : parseDate C s = .ok z) :
    (∃ y m d, ValidDate y m d ∧ z = daysFromCivil y m d) ∨ (∃ y o, C.fromYo y o = some z) ∨
    C.parseWeek s = .ok z := by
  unfold parseDate at h
  split at h
  · exact Or.inr (Or.inr h)
  split at h
  · cases h
  simp only at h
  split at h
  · split at h
    all_goals first
      | (cases h; done)
      | skip
    · obtain ⟨y, -, h⟩ := bind_ok h; exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
    · obtain ⟨y, -, h⟩ := bind_ok h; obtain ⟨m, -, h⟩ := bind_ok h
      exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
    · obtain ⟨y, -, h⟩ := bind_ok h; obtain ⟨m, -, h⟩ := bind_ok h
      obtain ⟨d, -, h⟩ := bind_ok h
      exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
  · split at h
    · obtain ⟨y, -, h⟩ := bind_ok h; exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
    · obtain ⟨y, -, h⟩ := bind_ok h; obtain ⟨m, -, h⟩ := bind_ok h
      exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
    · obtain ⟨y, -, h⟩ := bind_ok h; obtain ⟨o, -, h⟩ := bind_ok h
      split at h
      · rename_i z' hz; cases h; exact Or.inr (Or.inl ⟨_, _, hz⟩)
      · cases h
    · obtain ⟨y, -, h⟩ := bind_ok h; obtain ⟨m, -, h⟩ := bind_ok h
      obtain ⟨d, -, h⟩ := bind_ok h
      exact Or.inl ⟨_, _, _, ymd_ok C _ _ _ _ _ h⟩
    · cases h

/-- A leading `-` (negative year) is rejected with the documented message. -/
theorem parseDate_negative (s : String) (hW : s.contains 'W' = false) (h : s.startsWith "-" = true) :
    parseDate C s = .error s!"Unsupported date string: {s}" := by
  simp [parseDate, hW, h]

/-- `if parts.len() > i { parts[i].parse()? } else { 0 }` -/
def optPart (parts : List String) (i : Nat) (err : String) : Except String Nat :=
  if parts.length > i then pU32 C (parts.getD i "") err else .ok 0

/-- `parse_time_string` (temporal.rs:272). -/
def parseTime (s0 : String) : Except String Nat :=
  let s := (s0.splitOn ".").headD s0
  let hms (h m sec : Nat) : Except String Nat :=
    match fromHms h m sec with | some t => .ok t | none => .error s!"Invalid time: {s}"
  if s.contains ':' then
    let parts := s.splitOn ":"
    do
      let hour ← pU32 C (parts.headD "") s!"Invalid hour: {s}"
      let minute ← optPart C parts 1 s!"Invalid minute: {s}"
      let second ← optPart C parts 2 s!"Invalid second: {s}"
      hms hour minute second
  else
    let digits := digitsOnly s
    match digits.length with
    | 2 => do let h ← pU32 C digits s!"Invalid hour: {s}"; hms h 0 0
    | 4 => do
        let h ← pU32 C (sub digits 0 2) s!"Invalid hour: {s}"
        let m ← pU32 C ((sub digits 2 2)) s!"Invalid minute: {s}"
        hms h m 0
    | 6 => do
        let h ← pU32 C (sub digits 0 2) s!"Invalid hour: {s}"
        let m ← pU32 C ((sub digits 2 2)) s!"Invalid minute: {s}"
        let sec ← pU32 C ((sub digits 4 2)) s!"Invalid second: {s}"
        hms h m sec
    | _ => .error s!"Invalid time string: {s}"

theorem hms_ok (s : String) (h m sec t : Nat)
    (e : (match fromHms h m sec with | some t => Except.ok t | none => .error s!"Invalid time: {s}")
      = (Except.ok t : Except String Nat)) : t < 86400 := by
  split at e
  · rename_i t' ht; cases e; exact fromHms_lt _ _ _ _ ht
  · cases e

/-- Every successfully parsed time is a valid second of the day. -/
theorem parseTime_ok (s : String) (t : Nat) (h : parseTime C s = .ok t) : t < 86400 := by
  unfold parseTime at h; simp only at h
  split at h
  · obtain ⟨_, -, h⟩ := bind_ok h; obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h; exact hms_ok _ _ _ _ _ h
  · split at h
    · obtain ⟨_, -, h⟩ := bind_ok h; exact hms_ok _ _ _ _ _ h
    · obtain ⟨_, -, h⟩ := bind_ok h; obtain ⟨_, -, h⟩ := bind_ok h; exact hms_ok _ _ _ _ _ h
    · obtain ⟨_, -, h⟩ := bind_ok h; obtain ⟨_, -, h⟩ := bind_ok h
      obtain ⟨_, -, h⟩ := bind_ok h; exact hms_ok _ _ _ _ _ h
    · cases h

/-- `parse_datetime_string` (temporal.rs:329): split at the first `T`. -/
def parseDatetime (s : String) : Except String (Int × Nat) :=
  match s.splitOn "T" with
  | [] | [_] => do let d ← parseDate C s; pure (d, 0)
  | datePart :: rest => do
      let d ← parseDate C datePart
      let t ← parseTime C ("T".intercalate rest)
      pure (d, t)

theorem parseDatetime_ok (s : String) (d : Int) (t : Nat) (h : parseDatetime C s = .ok (d, t)) :
    t < 86400 := by
  unfold parseDatetime at h; split at h
  · obtain ⟨_, -, h⟩ := bind_ok h; cases h; decide
  · obtain ⟨_, -, h⟩ := bind_ok h; cases h; decide
  · obtain ⟨_, -, h⟩ := bind_ok h; obtain ⟨t', ht, h⟩ := bind_ok h; cases h
    exact parseTime_ok C _ _ ht

/-- Without a `T` the time is midnight and the date is `parse_date_string` of the whole. -/
theorem parseDatetime_noT (s : String) (h : s.splitOn "T" = [s]) :
    parseDatetime C s = (do let d ← parseDate C s; pure (d, 0)) := by
  simp [parseDatetime, h]

/-! ## `parse_duration_string` (temporal.rs:347) -/

/-- The seven accumulators `(years, months, 0, days, hours, minutes, seconds)`. -/
structure DurAcc where
  years : Int := 0
  months : Int := 0
  days : Int := 0
  hours : Int := 0
  minutes : Int := 0
  seconds : Int := 0

/-- `n * 7` on i64 in a release build wraps (`wrapMul` in Duration.lean). -/
def wrap64 (x : Int) : Int := (x + 2 ^ 63) % 2 ^ 64 - 2 ^ 63

/-- The date-part loop: digits and `-` accumulate, a letter commits the number. -/
def dateLoop : List Char → String → DurAcc → Except String DurAcc
  | [], _, a => .ok a
  | ch :: cs, buf, a =>
    if ch.isDigit || ch == '-' then dateLoop cs (buf.push ch) a
    else match C.parseI64 buf with
      | none => .error s!"Invalid number in duration: {buf}"
      | some n =>
        match ch with
        | 'Y' => dateLoop cs "" { a with years := n }
        | 'M' => dateLoop cs "" { a with months := n }
        | 'W' => dateLoop cs "" { a with days := wrap64 (a.days + wrap64 (n * 7)) }
        | 'D' => dateLoop cs "" { a with days := wrap64 (a.days + n) }
        | _ => .error s!"Unknown duration component: {ch}"

/-- The time-part loop: also accepts `.`, and drops the fraction before parsing. -/
def timeLoop : List Char → String → DurAcc → Except String DurAcc
  | [], _, a => .ok a
  | ch :: cs, buf, a =>
    if ch.isDigit || ch == '-' || ch == '.' then timeLoop cs (buf.push ch) a
    else match C.parseI64 ((buf.splitOn ".").headD buf) with
      | none => .error s!"Invalid number in duration: {buf}"
      | some n =>
        match ch with
        | 'H' => timeLoop cs "" { a with hours := n }
        | 'M' => timeLoop cs "" { a with minutes := n }
        | 'S' => timeLoop cs "" { a with seconds := n }
        | _ => .error s!"Unknown duration time component: {ch}"

/-- `if let Some(tp) = time_part { ..loop.. }` -/
def timeOpt (tp : Option String) (a : DurAcc) : Except String DurAcc :=
  match tp with
  | none => .ok a
  | some tp => timeLoop C tp.toList "" a

def parseDuration (s0 : String) : Except String (Int × Int × Int × Int × Int × Int × Int) :=
  if s0.startsWith "P" then
    let s := String.ofList (s0.toList.drop 1)
    let (datePart, timePart) := match s.splitOn "T" with
      | [] | [_] => (s, none)
      | d :: rest => (d, some ("T".intercalate rest))
    do
      let a ← dateLoop C datePart.toList "" {}
      let a ← timeOpt C timePart a
      pure (a.years, a.months, 0, a.days, a.hours, a.minutes, a.seconds)
  else .error s!"Duration string must start with 'P': {s0}"

/-- The weeks slot is always 0 (weeks are folded into days), and a string not starting
with `P` is rejected. -/
theorem parseDuration_weeks (s : String) (r : Int × Int × Int × Int × Int × Int × Int)
    (h : parseDuration C s = .ok r) : r.2.2.1 = 0 := by
  unfold parseDuration at h; split at h
  · simp only at h; obtain ⟨_, -, h⟩ := bind_ok h; obtain ⟨_, -, h⟩ := bind_ok h; cases h; rfl
  · cases h

theorem parseDuration_noP (s : String) (h : s.startsWith "P" = false) :
    parseDuration C s = .error s!"Duration string must start with 'P': {s}" := by
  simp [parseDuration, h]

end FalkorTemporal.Fns
