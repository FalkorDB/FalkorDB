/-!
# Startup banner (`src/banner/mod.rs`)

**`style`.** The environment is read through `var : String → Option String`, as in the Rust
(`std::env::var(name).ok()` in `print`). `setVar` is the Rust `set` closure: a variable set to the
empty string counts as unset. The locale is the first of `LC_ALL`, `LC_CTYPE`, `LANG` that is set
(`find_map`), ASCII-lowercased (`String.toLower` maps only `A`–`Z`, as `to_ascii_lowercase`).

* `style_none_iff` — no banner exactly when `TERM` is unset, empty or `dumb`;
* `style_textOnly` — a non-UTF-8 locale or `TERM=linux` gives the one-line ASCII banner;
* `style_logo_needs_utf8` — a logo (any colour) is printed only under a UTF-8 locale, not on the
  Linux console;
* `style_no_color` — with `NO_COLOR` set, no coloured style is chosen;
* `style_trueColor_iff` — 24-bit colour exactly when advertised by `COLORTERM`, with a capable
  `TERM`, a UTF-8 locale and no `NO_COLOR`;
* `locale_lc_all` — POSIX precedence: a set `LC_ALL` names the locale.

**`render`.** The logo files (`include_str!`) are parameters (`Logos`). `render_logo` — each logo
style's output contains its logo; `render_textOnly_no_esc`, `render_plain_no_esc` — the text-only
and `NO_COLOR` banners carry no escape byte (given an escape-free version and plain logo).

**`write_bounded`.** `text.split_inclusive('\n')` is the input `lines`; `ok i` is whether attempt
`i` succeeded (`write_all` and then `drained`, both terminal I/O). `writeBounded` returns the lines
attempted and the result of `.all`: `wb_true_iff` (true iff every line went out),
`wb_attempted_prefix` (lines are attempted in order), `wb_stops` (nothing after the first failed
line is attempted), `wb_true_all` (on success every line was attempted).

`print`, `drained` and `reopen_nonblocking` are terminal I/O (`isatty`, `ioctl(TIOCOUTQ)`,
`ttyname_r`, `open`) and are not modelled.
-/
namespace RedisLayer.Banner

inductive Style where
  | trueColor
  | ansi256
  | plain
  | textOnly
  deriving DecidableEq, Repr

abbrev Env := String → Option String

/-- The Rust `set` closure: `var(name).filter(|v| !v.is_empty())`. -/
def setVar (var : Env) (name : String) : Option String :=
  (var name).filter (fun v => !v.isEmpty)

/-- `haystack.contains(needle)` on character lists. -/
def hasInfix (needle : List Char) : List Char → Bool
  | [] => needle.isEmpty
  | c :: t => needle.isPrefixOf (c :: t) || hasInfix needle t

/-- `["LC_ALL", "LC_CTYPE", "LANG"].into_iter().find_map(set).unwrap_or_default()
    .to_ascii_lowercase()`. -/
def locale (var : Env) : String :=
  ((["LC_ALL", "LC_CTYPE", "LANG"].findSome? (setVar var)).getD "").toLower

def isUtf8 (loc : String) : Bool :=
  hasInfix "utf-8".toList loc.toList || hasInfix "utf8".toList loc.toList

def style (var : Env) : Option Style :=
  match setVar var "TERM" with
  | none => none
  | some term =>
    if term = "dumb" then none
    else if !isUtf8 (locale var) || term = "linux" then some .textOnly
    else if (setVar var "NO_COLOR").isSome then some .plain
    else if setVar var "COLORTERM" = some "truecolor" ∨ setVar var "COLORTERM" = some "24bit" then
      some .trueColor
    else some .ansi256

theorem style_none_iff (var : Env) :
    style var = none ↔ setVar var "TERM" = none ∨ setVar var "TERM" = some "dumb" := by
  unfold style
  split
  · simp_all
  · rename_i term h
    rw [h]
    by_cases hd : term = "dumb"
    · simp [hd]
    · simp only [hd, ite_false]
      constructor
      · intro h'
        split at h' <;> (try split at h') <;> (try split at h') <;> simp_all
      · intro h'; simp_all

theorem style_textOnly (var : Env) (term : String) (hT : setVar var "TERM" = some term)
    (hd : term ≠ "dumb") (h : isUtf8 (locale var) = false ∨ term = "linux") :
    style var = some .textOnly := by
  unfold style
  rw [hT]
  simp only [hd, ite_false]
  rcases h with h | h <;> simp [h]

theorem style_logo_needs_utf8 (var : Env) (s : Style) (hs : style var = some s)
    (hl : s ≠ .textOnly) :
    isUtf8 (locale var) = true ∧ setVar var "TERM" ≠ some "linux" := by
  unfold style at hs
  split at hs
  · simp at hs
  · rename_i term hT
    rw [hT]
    by_cases hd : term = "dumb"
    · simp [hd] at hs
    · simp only [hd, ite_false] at hs
      by_cases hc : (!isUtf8 (locale var) || term = "linux") = true
      · simp only [hc, ite_true, Option.some.injEq] at hs
        exact absurd hs.symm hl
      · simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq, not_or] at hc
        refine ⟨?_, ?_⟩
        · cases h : isUtf8 (locale var) <;> simp_all
        · intro h; apply hc.2; simpa using h

theorem style_no_color (var : Env) (s : Style) (hs : style var = some s)
    (hn : (setVar var "NO_COLOR").isSome = true) :
    s ≠ .trueColor ∧ s ≠ .ansi256 := by
  unfold style at hs
  split at hs
  · simp at hs
  · rename_i term _
    by_cases hd : term = "dumb"
    · simp [hd] at hs
    · simp only [hd, ite_false, hn, ite_true] at hs
      split at hs <;> (simp only [Option.some.injEq] at hs; subst hs; simp)

theorem style_trueColor_iff (var : Env) :
    style var = some .trueColor ↔
      ∃ term, setVar var "TERM" = some term ∧ term ≠ "dumb" ∧ term ≠ "linux" ∧
        isUtf8 (locale var) = true ∧ setVar var "NO_COLOR" = none ∧
        (setVar var "COLORTERM" = some "truecolor" ∨ setVar var "COLORTERM" = some "24bit") := by
  unfold style
  constructor
  · intro hs
    split at hs
    · simp at hs
    · rename_i term hT
      by_cases hd : term = "dumb"
      · simp [hd] at hs
      · simp only [hd, ite_false] at hs
        by_cases hc : (!isUtf8 (locale var) || term = "linux") = true
        · simp [hc] at hs
        · simp only [hc] at hs
          simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq, not_or] at hc
          have hu : isUtf8 (locale var) = true := by
            cases h : isUtf8 (locale var) <;> simp_all
          by_cases hn : (setVar var "NO_COLOR").isSome = true
          · simp [hn] at hs
          · simp only [hn] at hs
            refine ⟨term, hT, hd, hc.2, hu, ?_, ?_⟩
            · cases h : setVar var "NO_COLOR" <;> simp_all
            · by_cases hct : setVar var "COLORTERM" = some "truecolor" ∨
                  setVar var "COLORTERM" = some "24bit"
              · exact hct
              · simp [hct] at hs
  · rintro ⟨term, hT, hd, hl, hu, hn, hc⟩
    rw [hT]
    simp only [hd, hl, hu, hn, ite_false, Bool.not_true, Bool.false_or, decide_false,
      Option.isSome_none, Bool.false_eq_true]
    simp [hc]

theorem locale_lc_all (var : Env) (v : String) (h : setVar var "LC_ALL" = some v) :
    locale var = v.toLower := by
  simp [locale, List.findSome?, h]

/-! The Rust unit tests, on the model. -/

def envOf (vars : List (String × String)) : Env :=
  fun name => (vars.find? (·.1 == name)).map (·.2)

example : style (envOf [("LANG", "en_US.UTF-8")]) = none := by decide +kernel
example : style (envOf [("TERM", "dumb"), ("LANG", "en_US.UTF-8")]) = none := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LANG", "C")]) = some .textOnly := by
  decide +kernel
example : style (envOf [("TERM", "linux"), ("LANG", "en_US.UTF-8")]) = some .textOnly := by
  decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LC_ALL", "C"), ("LANG", "en_US.UTF-8")]) =
    some .textOnly := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LC_ALL", ""), ("LC_CTYPE", "C.utf8"),
    ("LANG", "C")]) = some .ansi256 := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LANG", "en_US.UTF-8"),
    ("COLORTERM", "truecolor"), ("NO_COLOR", "1")]) = some .plain := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LANG", "en_US.UTF-8"),
    ("COLORTERM", "truecolor"), ("NO_COLOR", "")]) = some .trueColor := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LANG", "en_US.UTF-8"),
    ("COLORTERM", "24bit")]) = some .trueColor := by decide +kernel
example : style (envOf [("TERM", "xterm-256color"), ("LANG", "en_US.UTF-8"),
    ("COLORTERM", "yes")]) = some .ansi256 := by decide +kernel

/-! ### `render` -/

/-- The `include_str!` logo files. -/
structure Logos where
  trueColor : String
  ansi256 : String
  plain : String

def logoOf (l : Logos) : Style → String
  | .trueColor => l.trueColor
  | .ansi256 => l.ansi256
  | .plain => l.plain
  | .textOnly => ""

def render (l : Logos) (s : Style) (version : String) : String :=
  match s with
  | .textOnly => "\nFalkorDB v" ++ version ++ " - https://falkordb.com\n\n"
  | .plain => "\n" ++ l.plain ++ "\n  FalkorDB v" ++ version ++ " \u00b7 https://falkordb.com\n\n"
  | .trueColor | .ansi256 =>
    let logo := if s = .trueColor then l.trueColor else l.ansi256
    "\n" ++ logo ++ "\n\x1b[2m  FalkorDB v" ++ version ++ " \u00b7 https://falkordb.com\x1b[0m\n\n"

theorem render_logo (l : Logos) (s : Style) (v : String) (hs : s ≠ .textOnly) :
    ∃ a b, render l s v = a ++ logoOf l s ++ b := by
  cases s with
  | textOnly => exact absurd rfl hs
  | plain => exact ⟨"\n", _, by simp only [render, logoOf, String.append_assoc]; rfl⟩
  | trueColor =>
    exact ⟨"\n", _, by simp only [render, logoOf, ite_true, String.append_assoc]; rfl⟩
  | ansi256 =>
    exact ⟨"\n", _, by
      simp only [render, logoOf, reduceCtorEq, ite_false, String.append_assoc]; rfl⟩

theorem render_textOnly_no_esc (l : Logos) (v : String) (hv : '\x1b' ∉ v.toList) :
    '\x1b' ∉ (render l .textOnly v).toList := by
  simp only [render, String.toList_append, List.mem_append, not_or]
  refine ⟨⟨by decide, hv⟩, by decide⟩

theorem render_plain_no_esc (l : Logos) (v : String) (hp : '\x1b' ∉ l.plain.toList)
    (hv : '\x1b' ∉ v.toList) :
    '\x1b' ∉ (render l .plain v).toList := by
  simp only [render, String.toList_append, List.mem_append, not_or]
  exact ⟨⟨⟨⟨by decide, hp⟩, by decide⟩, hv⟩, by decide⟩

/-! ### `write_bounded` -/

/-- `lines.all(|line| write_all(line).is_ok() && drained(..))`, from attempt `i`: the lines
    attempted and the result. -/
def writeBounded (ok : Nat → Bool) (i : Nat) : List String → List String × Bool
  | [] => ([], true)
  | l :: t =>
    if ok i then
      let r := writeBounded ok (i + 1) t
      (l :: r.1, r.2)
    else ([l], false)

theorem wb_true_iff (ok : Nat → Bool) (i : Nat) (ls : List String) :
    (writeBounded ok i ls).2 = true ↔ ∀ k < ls.length, ok (i + k) = true := by
  induction ls generalizing i with
  | nil => simp [writeBounded]
  | cons l t ih =>
    unfold writeBounded
    by_cases h : ok i = true
    · simp only [h, ite_true, ih]
      constructor
      · intro H k hk
        cases k with
        | zero => simpa using h
        | succ k =>
          have := H k (by simp at hk; omega)
          rwa [show i + 1 + k = i + (k + 1) by omega] at this
      · intro H k hk
        have := H (k + 1) (by simp; omega)
        rwa [show i + (k + 1) = i + 1 + k by omega] at this
    · simp only [h, ite_false, Bool.false_eq_true, false_iff]
      intro H
      exact h (by simpa using H 0 (by simp))

theorem wb_attempted_prefix (ok : Nat → Bool) (i : Nat) (ls : List String) :
    (writeBounded ok i ls).1 <+: ls := by
  induction ls generalizing i with
  | nil => simp [writeBounded]
  | cons l t ih =>
    unfold writeBounded
    by_cases h : ok i = true
    · simp only [h, ite_true]
      exact List.cons_prefix_cons.mpr ⟨rfl, ih (i + 1)⟩
    · simp only [h, ite_false, Bool.false_eq_true]
      exact List.cons_prefix_cons.mpr ⟨rfl, List.nil_prefix⟩

theorem wb_stops (ok : Nat → Bool) (i k : Nat) (ls : List String) (hk : ok (i + k) = false) :
    (writeBounded ok i ls).1.length ≤ k + 1 := by
  induction ls generalizing i k with
  | nil => simp [writeBounded]
  | cons l t ih =>
    unfold writeBounded
    by_cases h : ok i = true
    · simp only [h, ite_true, List.length_cons]
      cases k with
      | zero => simp_all
      | succ k =>
        have := ih (i + 1) k (by rwa [show i + 1 + k = i + (k + 1) by omega])
        omega
    · simp only [h, ite_false, Bool.false_eq_true, List.length_singleton]
      omega

theorem wb_true_all (ok : Nat → Bool) (i : Nat) (ls : List String)
    (h : (writeBounded ok i ls).2 = true) : (writeBounded ok i ls).1 = ls := by
  induction ls generalizing i with
  | nil => simp [writeBounded]
  | cons l t ih =>
    unfold writeBounded at h ⊢
    by_cases hi : ok i = true
    · simp only [hi, ite_true] at h ⊢
      rw [ih (i + 1) h]
    · simp [hi] at h

end RedisLayer.Banner
