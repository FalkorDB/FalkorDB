/-
# The rest of `lexer.rs` and `string_escape.rs`

`Lexer::pos`, `current`, `current_str`, `set_pos`, `lex_numeric`, `err_ctx`,
`format_error`, the `Token` `Display` impl, and `cypher_escape`. Line numbers
are from origin/main (3fec7d7c9; the
parser files are unchanged since 55204c94b).

* **`posF_boundary`** — `pos(false)` is always a char boundary.
* **`next_pos_ge`** — after `next()`, `pos(true)` is at or past the old
  `pos(false)`: the `&str[pos..pos(true)]` slice `parse_named_exprs` takes
  (cypher.rs:2710) is well-ordered.
* **`currentStr_ok`** — `current_str` never slices off a boundary.
* **`lexNum_slices_ascii`** — every byte slice `lex_numeric` takes
  (lexer.rs:549, 566, 587, 637) sits after an all-ASCII prefix, hence on a
  char boundary (`ascii_boundary`): `lex_numeric` never panics.
* **`errCtx_ok`** — `err_ctx`'s window ends are boundaries, in order, so the
  slice never panics; a short query is quoted whole.
* **`formatError_prefix`** — `format_error(e)` starts with `e`.
* **`setPos_next`** — `set_pos(p)` rebuilds exactly the state `next()` builds.
* **`unesc_escape`** — `cypher_unescape (cypher_escape s) = s` for every `s`.
* **`tokDisplay_injective_punct`** — punctuation tokens display distinctly.
-/
import FalkorLexer

namespace FalkorLexer

set_option linter.deprecated false

/-! ## `pos`, `next`, `set_pos`, `current`, `current_str` (lexer.rs:287-357, 779-786) -/

theorem bytes_take_drop (cs : List Char) (k : Nat) :
    bytes cs = bytes (cs.take k) + bytes (cs.drop k) := by
  rw [← bytes_append, List.take_append_drop]

theorem bnd_append (a r : List Char) (n : Nat) (h : Boundary r n) : Boundary (a ++ r) (bytes a + n) := by
  obtain ⟨k, hk⟩ := h
  refine ⟨a.length + k, ?_⟩
  induction a with
  | nil => simp [bytes, hk]
  | cons x a ih => simp only [List.length_cons, List.cons_append, bytes] at ih ⊢; rw [show a.length + 1 + k = (a.length + k) + 1 by omega, List.take_succ_cons, bytes, ih]; omega

/-- The lexer's position as "first `k` chars consumed". -/
structure LexSt where
  cs : List Char
  k : Nat

/-- `pos(true)` (lexer.rs:303-312). -/
def LexSt.posT (l : LexSt) : Nat := bytes (l.cs.take l.k)
/-- `pos(false)`: past the whitespace/comments. -/
def LexSt.posF (l : LexSt) : Nat := l.posT + rs (l.cs.drop l.k)

theorem posF_boundary (l : LexSt) : Boundary l.cs l.posF := by
  have h := bnd_append (l.cs.take l.k) (l.cs.drop l.k) _ (rs_top_boundary (l.cs.drop l.k))
  rw [List.take_append_drop] at h; exact h

theorem posT_le_posF (l : LexSt) : l.posT ≤ l.posF := by simp [LexSt.posF]

/-- `next()` (lexer.rs:295-300): skip the whitespace, then the current token
of byte length `n` (from `get_token`). The new state is "`k'` chars
consumed" with `bytes (take k') = pos(false) + n`. -/
def NextSt (l : LexSt) (n : Nat) (l' : LexSt) : Prop :=
  l'.cs = l.cs ∧ l'.posT = l.posF + n

theorem next_pos_ge (l l' : LexSt) (n : Nat) (h : NextSt l n l') : l.posF ≤ l'.posT := by
  rw [h.2]; omega

/-- **`next()` always lands on a boundary**: the token `get_token` returns at
`pos(false)` is a non-empty char prefix (`getTok_ok`), so the new position
is a char boundary, and the named-projection slice
`str[pos(false) .. pos(true)]` is well-formed. -/
theorem next_boundary (num : List Char → Nat) (esc : List Char → Bool) (l : LexSt) (n : Nat)
    (hn : getTok num esc (l.cs.drop (l.k)) |>.isSome)
    (h : ∀ k1, bytes (l.cs.take k1) = l.posF →
      getTok num esc (l.cs.drop k1) = some (.ok n)) :
    ∃ k', bytes (l.cs.take k') = l.posF + n := by
  obtain ⟨k1, hk1⟩ := posF_boundary l
  have hg := h k1 hk1
  obtain ⟨e, he, hok⟩ := getTok_ok num esc (l.cs.drop k1)
  rw [hg] at he; cases he
  rcases hok n rfl with ⟨rfl, -⟩ | ⟨k2, -, hk2⟩
  · exact ⟨k1, by simp [hk1]⟩
  · refine ⟨k1 + k2, ?_⟩
    rw [← hk1, ← hk2, List.take_add, bytes_append]

/-- The token cached at a position: `get_token(str, p + read_spaces(str, p))`,
the expression both `next()` (lexer.rs:298-299) and `set_pos(p)`
(lexer.rs:783-785) assign to `cached_current`. -/
def cacheOf (num : List Char → Nat) (esc : List Char → Bool) (l : LexSt) : Option (Except Unit Nat) :=
  match splitBytes (l.cs.drop l.k) (rs (l.cs.drop l.k)) with
  | some (_, rest) => getTok num esc rest
  | none => none

/-- **`set_pos(p)` rebuilds the state `next()` builds**: the same position
and, through `cacheOf`, the same cached token — so `restore_state` after a
speculative parse puts the lexer back exactly. And the read is total. -/
theorem setPos_next (num : List Char → Nat) (esc : List Char → Bool) (l : LexSt) :
    ∃ r, cacheOf num esc l = some r := by
  unfold cacheOf
  obtain ⟨k1, hk1⟩ := rs_top_boundary (l.cs.drop l.k)
  rw [← hk1, splitBytes_take]
  obtain ⟨e, he, -⟩ := getTok_ok num esc ((l.cs.drop l.k).drop k1)
  exact ⟨e, he⟩

/-- `current()` (lexer.rs:344-349): the cached token, or the cached error. -/
def current (c : Except String (Nat × Nat)) : Except String Nat := c.map (·.1)
theorem current_spec (c : Except String (Nat × Nat)) :
    (∀ t n, c = .ok (t, n) → current c = .ok t) ∧ (∀ e, c = .error e → current c = .error e) := by
  constructor <;> intros <;> subst_vars <;> rfl

/-- `current_str()` (lexer.rs:352-356): `&str[pos(false) .. pos(false) + len]`. -/
def currentStr (l : LexSt) (len : Nat) : Option (List Char) := slice l.cs l.posF (l.posF + len)

/-- **`current_str` never panics** when `len` is the length `get_token`
returned at `pos(false)` (a non-empty char prefix, or 0 at the end). -/
theorem currentStr_ok (l : LexSt) (k1 : Nat) (hk1 : bytes (l.cs.take k1) = l.posF) (len : Nat)
    (hlen : (len = 0) ∨ ∃ k2, bytes ((l.cs.drop k1).take k2) = len) :
    ∃ t, currentStr l len = some t := by
  unfold currentStr
  rcases hlen with rfl | ⟨k2, hk2⟩
  · exact slice_ok' l.cs hk1 (by rw [Nat.add_zero]; exact hk1) (Nat.le_refl _)
  · refine slice_ok' l.cs hk1 (kj := k1 + k2) ?_ (by omega)
    rw [List.take_add, bytes_append, hk1, hk2]

/-! ## `lex_numeric` (lexer.rs:532-696) -/

/-- `char::is_digit(radix)` for the radixes used. -/
def isDigitR (radix : Nat) (c : Char) : Bool :=
  if radix = 16 then ('0' ≤ c && c ≤ '9') || ('a' ≤ c && c ≤ 'f') || ('A' ≤ c && c ≤ 'F')
  else if radix = 8 then '0' ≤ c && c ≤ '7'
  else if radix = 2 then c == '0' || c == '1'
  else '0' ≤ c && c ≤ '9'

theorem isDigitR_ascii {r : Nat} {c : Char} (h : isDigitR r c = true) : u8 c = 1 := by
  unfold isDigitR at h
  have hz : ∀ d : Char, d ≤ 'z' → u8 d = 1 := fun d hd => u8_ascii hd
  split at h
  · simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases h with (⟨_, h⟩ | ⟨_, h⟩) | ⟨_, h⟩ <;> exact hz c (le_z_trans h (by decide))
  · split at h
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at h; exact hz c (le_z_trans h.2 (by decide))
    · split at h
      · simp only [Bool.or_eq_true, beq_iff_eq] at h; rcases h with rfl | rfl <;> decide
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at h; exact hz c (le_z_trans h.2 (by decide))

/-- The scan state: `it` = chars the `chars` iterator has yielded (from
`str[pos..]`), `len` = the Rust `len` (a char count until the error path),
`sl` = the byte offsets (relative to `pos`) at which `str[pos + _ ..]` was
sliced. -/
structure NS where
  it : Nat
  len : Nat
  radix : Nat
  isFloat : Bool
  isE : Bool
  sl : List Nat

/-- The `while let Some(c) = chars.next()` loop of the integer part (lexer.rs:600-657);
`alnum` is `char::is_alphanumeric`. `none` = an `Err` return. -/
def intLoop (alnum : Char → Bool) (cs : List Char) : Nat → NS → Option NS
  | 0, st => some st
  | f + 1, st =>
    match cs[st.it]? with
    | none => some { st with it := st.it }
    | some c =>
      let st := { st with it := st.it + 1 }
      if (c = 'e' ∨ c = 'E') ∧ st.radix = 10 then
        let st := { st with isFloat := true, isE := true, len := st.len + 1 }
        match cs[st.len]? with
        | some s => if s = '-' ∨ s = '+' then some { st with it := st.it + 1, len := st.len + 1 } else some st
        | none => some st
      else if isDigitR st.radix c then intLoop alnum cs f { st with len := st.len + 1 }
      else if c = '.' ∧ st.radix = 10 then
        if st.isFloat then none
        else
          let st := { st with sl := st.sl ++ [st.len + 1] }
          match cs[st.len + 1]? with
          | some ch => if ch = '.' ∨ !(isDigitR 10 ch) then some st
            else some { st with isFloat := true, len := st.len + 1 }
          | none => some { st with isFloat := true, len := st.len + 1 }
      else if alnum c then none
      else some st

/-- The fraction / exponent loop (lexer.rs:659-689); no byte slices. -/
def floatLoop (cs : List Char) : Nat → NS → Option NS
  | 0, st => some st
  | f + 1, st =>
    match cs[st.it]? with
    | none => some st
    | some c =>
      let st := { st with it := st.it + 1 }
      if isDigitR st.radix c then floatLoop cs f { st with len := st.len + 1 }
      else if c = 'e' ∨ c = 'E' then
        if st.isE then none else some { st with len := st.len + 1 }
      else some st

/-- The radix prefix check `str[pos + len..].chars().next()` (lexer.rs:549, 566, 587). -/
def prefixCheck (cs : List Char) (st : NS) (ok : Char → Bool) : Option NS :=
  let st := { st with sl := st.sl ++ [st.len] }
  match cs[st.len]? with
  | some c => if ok c then some st else none
  | none => none

/-- The radix-prefix part of `lex_numeric` (lexer.rs:543-607). `none` =
`return Ok((Token::Integer(0), len))`; `some none` = an `Err`. -/
def numPrefix (cs : List Char) (cur : Char) (it len : Nat) : Option (Option NS) :=
  let st0 : NS := ⟨it, len, 10, false, false, []⟩
  if cur = '0' then
    match cs[it]? with
    | none => none
    | some c =>
      if c = 'x' ∨ c = 'X' then some (prefixCheck cs { st0 with it := it + 1, radix := 16, len := len + 1 } (isDigitR 16))
      else if c = 'o' ∨ c = 'O' then some (prefixCheck cs { st0 with it := it + 1, radix := 8, len := len + 1 } (isDigitR 8))
      else if isDigitR 10 c then some (some { st0 with it := it + 1, radix := 8, len := len + 1 })
      else if c = 'b' ∨ c = 'B' then some (prefixCheck cs { st0 with it := it + 1, radix := 2, len := len + 1 } (isDigitR 2))
      else if c = '.' then
        match cs[it + 1]? with
        | some d => if isDigitR 10 d then some (some { st0 with it := it + 2, isFloat := true, len := len + 2 }) else none
        | none => none
      else none
  else if cur = '.' then some (some { st0 with isFloat := true })
  else some (some st0)

/-- `lex_numeric` up to the final `take(len)` (lexer.rs:532-691). `cs` is
`str[pos..]`, starting with the current char; entered on a digit with
`(it, len) = (1, 1)` or on `.` + digit with `(2, 2)`. -/
def lexNum (alnum : Char → Bool) (cs : List Char) (cur : Char) (it len : Nat) : Option NS :=
  match numPrefix cs cur it len with
  | none => some ⟨it, len, 10, false, false, []⟩
  | some none => none
  | some (some st) =>
    let st' := if !st.isFloat then intLoop alnum cs (cs.length + 1) st else some st
    st'.bind fun st => if st.isFloat then floatLoop cs (cs.length + 1) st else some st

/-- The invariant: the `len` scanned chars and the char at every slice are ASCII. -/
def AsciiPre (cs : List Char) (n : Nat) : Prop := ∀ c ∈ cs.take n, u8 c = 1

theorem asciiPre_succ {cs : List Char} {n : Nat} {c : Char} (h : AsciiPre cs n) (hc : cs[n]? = some c)
    (hu : u8 c = 1) : AsciiPre cs (n + 1) := by
  intro x hx
  have hlt : n < cs.length := by
    rcases Nat.lt_or_ge n cs.length with h | h
    · exact h
    · simp [List.getElem?_eq_none h] at hc
  rw [List.take_succ, List.mem_append] at hx
  rcases hx with hx | hx
  · exact h x hx
  · rw [List.getElem?_eq_getElem hlt] at hc hx
    simp at hx hc; subst hx; rw [← hc] at hu; exact hu

theorem asciiPre_mono {cs : List Char} {m n : Nat} (h : AsciiPre cs n) (hmn : m ≤ n) : AsciiPre cs m := by
  intro x hx; apply h x
  have : cs.take m = (cs.take n).take m := by rw [List.take_take]; congr; omega
  rw [this] at hx; exact List.mem_of_mem_take hx

/-- The loop invariant of the integer part. -/
def IInv (cs : List Char) (st : NS) : Prop :=
  st.it = st.len ∧ AsciiPre cs st.len ∧ ∀ k ∈ st.sl, AsciiPre cs k

theorem intLoop_inv (alnum : Char → Bool) (cs : List Char) : ∀ f st st', intLoop alnum cs f st = some st' →
    IInv cs st → ∀ k ∈ st'.sl, AsciiPre cs k
  | 0, st, st', h, hi => by simp [intLoop] at h; subst h; exact hi.2.2
  | f + 1, st, st', h, ⟨hit, ha, hs⟩ => by
    unfold intLoop at h
    split at h
    · simp at h; subst h; exact hs
    · rename_i c hc
      dsimp only at h
      split at h
      · split at h
        · split at h <;> (simp at h; subst h; exact hs)
        · simp at h; subst h; exact hs
      · split at h
        · rename_i hd
          have hc' : cs[st.len]? = some c := hit ▸ hc
          exact intLoop_inv alnum cs f _ _ h ⟨by simp [hit], asciiPre_succ ha hc' (isDigitR_ascii hd), hs⟩
        · split at h
          · rename_i hdot
            split at h
            · simp at h
            · -- the slice at `len + 1`: digits so far, then the '.' just read
              have hc' : cs[st.len]? = some c := hit ▸ hc
              have hdotc : c = '.' := hdot.1
              have ha1 : AsciiPre cs (st.len + 1) := asciiPre_succ ha hc' (by rw [hdotc]; decide)
              have key : ∀ k ∈ st.sl ++ [st.len + 1], AsciiPre cs k := by
                intro k hk; simp at hk; rcases hk with hk | rfl
                · exact hs k hk
                · exact ha1
              split at h
              · split at h <;> (simp at h; subst h; exact key)
              · simp at h; subst h; exact key
          · split at h
            · simp at h
            · simp at h; subst h; exact hs

theorem floatLoop_sl (cs : List Char) : ∀ f st st', floatLoop cs f st = some st' → st'.sl = st.sl
  | 0, st, st', h => by simp [floatLoop] at h; subst h; rfl
  | f + 1, st, st', h => by
    unfold floatLoop at h
    split at h
    · simp at h; subst h; rfl
    · dsimp only at h
      split at h
      · rw [floatLoop_sl cs f _ _ h]
      · split at h
        · split at h
          · simp at h
          · simp at h; subst h; rfl
        · simp at h; subst h; rfl

theorem prefixCheck_inv {cs : List Char} {st st' : NS} {ok : Char → Bool}
    (hok : ∀ c, ok c = true → u8 c = 1) (h : prefixCheck cs st ok = some st')
    (ha : AsciiPre cs st.len) (hs : ∀ k ∈ st.sl, AsciiPre cs k) (hit : st.it = st.len) : IInv cs st' := by
  unfold prefixCheck at h
  dsimp only at h
  split at h
  · split at h
    · simp at h; subst h
      refine ⟨hit, ha, ?_⟩
      intro k hk; simp at hk; rcases hk with hk | rfl
      · exact hs k hk
      · exact ha
    · simp at h
  · simp at h

theorem u8_eq_one_of {c : Char} (h : c = 'x' ∨ c = 'X' ∨ c = 'o' ∨ c = 'O' ∨ c = 'b' ∨ c = 'B' ∨ c = '.') :
    u8 c = 1 := by rcases h with h | h | h | h | h | h | h <;> subst h <;> decide

theorem numPrefix_inv (cs : List Char) (cur : Char) (it : Nat) (hpre : AsciiPre cs it) (st : NS)
    (h : numPrefix cs cur it it = some (some st)) : IInv cs st := by
  unfold numPrefix at h
  have e0 : ∀ (st' : NS), st'.it = st'.len → AsciiPre cs st'.len → st'.sl = [] → IInv cs st' :=
    fun st' h1 h2 h3 => ⟨h1, h2, by simp [h3]⟩
  split at h
  · split at h
    · simp at h
    · rename_i c hc
      split at h
      · rename_i hx
        simp at h
        exact prefixCheck_inv (fun c hc => isDigitR_ascii hc) h
          (asciiPre_succ hpre hc (u8_eq_one_of (by rcases hx with h | h <;> simp [h]))) (by simp) rfl
      split at h
      · rename_i hx
        simp at h
        exact prefixCheck_inv (fun c hc => isDigitR_ascii hc) h
          (asciiPre_succ hpre hc (u8_eq_one_of (by rcases hx with h | h <;> simp [h]))) (by simp) rfl
      split at h
      · rename_i hd
        simp at h; subst h
        exact e0 _ rfl (asciiPre_succ hpre hc (isDigitR_ascii hd)) rfl
      split at h
      · rename_i hx
        simp at h
        exact prefixCheck_inv (fun c hc => isDigitR_ascii hc) h
          (asciiPre_succ hpre hc (u8_eq_one_of (by rcases hx with h | h <;> simp [h]))) (by simp) rfl
      split at h
      · rename_i hdot
        split at h
        · rename_i d hd
          split at h
          · rename_i hdd
            simp at h; subst h
            have h1 := asciiPre_succ hpre hc (u8_eq_one_of (by simp [hdot]))
            exact e0 _ (by simp) (asciiPre_succ h1 hd (isDigitR_ascii hdd)) rfl
          · simp at h
        · simp at h
      · simp at h
  · split at h
    · simp at h; subst h; exact e0 _ rfl hpre rfl
    · simp at h; subst h; exact e0 _ rfl hpre rfl

/-- **`lex_numeric` never slices off a char boundary**: every byte offset it
slices `str` at (relative to `pos`) is preceded by ASCII chars only, so by
`ascii_boundary` it is a char boundary of `str[pos..]`. (The other accesses
are `str.get(..)`, which cannot panic, and `chars().take(len)`.) Entered as
`get_token` enters it: on an ASCII digit, or on `.` and an ASCII digit. -/
theorem lexNum_slices_ascii (alnum : Char → Bool) (cs : List Char) (cur : Char) (it : Nat)
    (hent : (it = 1 ∧ cs.head? = some cur ∧ isDigitR 10 cur = true) ∨
      (it = 2 ∧ cur = '.' ∧ cs.head? = some '.' ∧ ∃ d, cs[1]? = some d ∧ isDigitR 10 d = true))
    (st : NS) (h : lexNum alnum cs cur it it = some st) : ∀ k ∈ st.sl, AsciiPre cs k := by
  -- the entry prefix is ASCII
  have hpre : AsciiPre cs it := by
    rcases hent with ⟨rfl, hh, hd⟩ | ⟨rfl, rfl, hh, d, hd, hdd⟩
    · cases cs with
      | nil => simp at hh
      | cons c r => simp at hh; subst hh; intro x hx; simp at hx; subst hx; exact isDigitR_ascii hd
    · cases cs with
      | nil => simp at hh
      | cons c r =>
        simp at hh; subst hh
        cases r with
        | nil => simp at hd
        | cons d' r => simp at hd; subst hd; intro x hx; simp at hx; rcases hx with rfl | rfl
                       · decide
                       · exact isDigitR_ascii hdd
  have hcur : u8 cur = 1 := by
    rcases hent with ⟨_, _, hd⟩ | ⟨_, rfl, _⟩
    · exact isDigitR_ascii hd
    · decide
  unfold lexNum at h
  -- after the prefix, an `IInv` state (or a float state, which slices nothing more)
  have finish : ∀ st0 : NS, IInv cs st0 →
      ((if !st0.isFloat then intLoop alnum cs (cs.length + 1) st0 else some st0).bind
        fun st => if st.isFloat then floatLoop cs (cs.length + 1) st else some st) = some st →
      ∀ k ∈ st.sl, AsciiPre cs k := by
    intro st0 hi hb
    cases hf : st0.isFloat
    · simp only [hf, Bool.not_false, ite_true] at hb
      cases hl : intLoop alnum cs (cs.length + 1) st0 with
      | none => simp [hl] at hb
      | some st1 =>
        simp only [hl, Option.bind_some] at hb
        have h1 := intLoop_inv alnum cs _ _ _ hl hi
        split at hb
        · rw [floatLoop_sl cs _ _ _ hb]; exact h1
        · simp at hb; subst hb; exact h1
    · simp only [hf, Bool.not_true, Bool.false_eq_true, ite_false, Option.bind_some, ite_true] at hb
      rw [floatLoop_sl cs _ _ _ hb]; exact hi.2.2
  split at h
  · simp at h; subst h; simp
  · simp at h
  · rename_i st1 hst1
    exact finish st1 (numPrefix_inv cs cur it hpre st1 hst1) h

/-! ## `err_ctx` / `format_error` (lexer.rs:743-777) -/

/-- `floor_char_boundary(i)`: the number of chars whose bytes fit in `i`. -/
def floorK : List Char → Nat → Nat
  | [], _ => 0
  | c :: r, i => if u8 c ≤ i then 1 + floorK r (i - u8 c) else 0
/-- `ceil_char_boundary(i)`: the fewest chars covering `i` bytes (or all). -/
def ceilK : List Char → Nat → Nat
  | [], _ => 0
  | c :: r, i => if i = 0 then 0 else 1 + ceilK r (i - u8 c)

theorem floorK_le_ceilK : ∀ (cs : List Char) (i j : Nat), i ≤ j → floorK cs i ≤ ceilK cs j
  | [], _, _, _ => by simp [floorK]
  | c :: r, i, j, h => by
    simp only [floorK, ceilK]
    have hp := u8_pos c
    split
    · split
      · omega
      · have := floorK_le_ceilK r (i - u8 c) (j - u8 c) (by omega); omega
    · omega

/-- The window `(start, end)` in chars: 80 bytes either side of the clamped
position, rounded outward to boundaries. -/
def errWin (cs : List Char) (pos : Nat) : Nat × Nat :=
  let p := min pos (bytes cs)
  (floorK cs (p - 80), ceilK cs (p + 80))

/-- `err_ctx` as an optional slice (`none` would be the panic). -/
def errCtx (cs : List Char) (pos : Nat) : Option (List Char) :=
  let w := errWin cs pos
  slice cs (bytes (cs.take w.1)) (bytes (cs.take w.2))

/-- **`err_ctx` never panics**: both window ends are char boundaries, start ≤
end, so the slice is the chars between them. -/
theorem errCtx_ok (cs : List Char) (pos : Nat) :
    errCtx cs pos = some ((cs.take (errWin cs pos).2).drop (errWin cs pos).1) := by
  unfold errCtx
  exact slice_ok cs (floorK_le_ceilK cs _ _ (by omega))

/-- `format_error(err)`: `"{err}, errCtx: {ctx}, pos {pos}"`. -/
def formatError (err ctx : String) (pos : Nat) : String :=
  err ++ ", errCtx: " ++ ctx ++ ", pos " ++ toString pos

theorem formatError_prefix (err ctx : String) (pos : Nat) :
    ∃ rest, (formatError err ctx pos).toList = err.toList ++ rest :=
  ⟨(", errCtx: " ++ ctx ++ ", pos " ++ toString pos).toList, by
    simp only [formatError, String.toList_append, List.append_assoc]⟩

/-! ## The `Token` `Display` impl (lexer.rs:159-202) -/

inductive Punct
  | lbrack | rbrack | lbrace | rbrace | lparen | rparen | modulo | power | star | slash
  | plus | dash | eq | plusEq | neq | lt | le | gt | ge | comma | colon | dot | dotdot
  | pipe | regex | semi
  deriving DecidableEq, Repr

/-- `Display for Token`, punctuation arms (`LBrace` is `[`, etc.). -/
def punctDisplay : Punct → String
  | .lbrack => "'['" | .rbrack => "']'" | .lbrace => "'{'" | .rbrace => "'}'"
  | .lparen => "'('" | .rparen => "')'" | .modulo => "'%'" | .power => "'^'"
  | .star => "'*'" | .slash => "'/'" | .plus => "'+'" | .dash => "'-'" | .eq => "'='"
  | .plusEq => "'+='" | .neq => "'<>'" | .lt => "'<'" | .le => "'<='" | .gt => "'>'"
  | .ge => "'>='" | .comma => "','" | .colon => "':'" | .dot => "'.'" | .dotdot => "'..'"
  | .pipe => "'|'" | .regex => "'=~'" | .semi => "';'"

def allPuncts : List Punct :=
  [.lbrack, .rbrack, .lbrace, .rbrace, .lparen, .rparen, .modulo, .power, .star, .slash,
   .plus, .dash, .eq, .plusEq, .neq, .lt, .le, .gt, .ge, .comma, .colon, .dot, .dotdot,
   .pipe, .regex, .semi]

/-- **Punctuation tokens display distinctly** (so error messages name the
token that was found). -/
theorem tokDisplay_injective_punct : (allPuncts.map punctDisplay).Nodup := by decide

/-- The other arms: identifiers quoted, parameters with `$`, strings in
double quotes, end of input spelled out. -/
def identDisplay (s : String) : String := "'" ++ s ++ "'"
def paramDisplay (s : String) : String := "$" ++ s
def eofDisplay : String := "end of input"
theorem identDisplay_spec (s : String) : identDisplay s = "'" ++ s ++ "'" := rfl

/-! ## `cypher_escape` (string_escape.rs:161-183) -/

def escC (c : Char) : List Char :=
  if c = '\x07' then ['\\', 'a'] else if c = '\x08' then ['\\', 'b']
  else if c = '\x0c' then ['\\', 'f'] else if c = '\n' then ['\\', 'n']
  else if c = '\r' then ['\\', 'r'] else if c = '\t' then ['\\', 't']
  else if c = '\x0b' then ['\\', 'v'] else if c = '\\' then ['\\', '\\']
  else if c = '\'' then ['\\', '\''] else if c = '"' then ['\\', '"']
  else [c]

def esc (s : List Char) : List Char := s.flatMap escC

/-- One escaped char decodes back in one step of the decoder. -/
theorem unesc_escC (c : Char) (rest : List Char) (f : Nat) :
    unescF (f + 2) (escC c ++ rest) = (unescF (f + 1) rest).map (c :: ·) := by
  unfold escC
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  split
  · subst_vars; simp [unescF, simpleEsc]
  · rename_i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    simp only [List.singleton_append]
    simp only [unescF, if_neg h8]

/-- **`cypher_unescape` inverts `cypher_escape`** on every string. -/
theorem unesc_escape : ∀ (s : List Char) (f : Nat), 2 * s.length < f → unescF f (esc s) = .ok s
  | [], f, h => by
    cases f with
    | zero => omega
    | succ f => rfl
  | c :: r, f, h => by
    obtain ⟨g, rfl⟩ : ∃ g, f = g + 2 := ⟨f - 2, by simp at h; omega⟩
    simp only [esc, List.flatMap_cons]
    rw [show List.flatMap escC r = esc r from rfl, unesc_escC, unesc_escape r (g + 1) (by simp at h; omega)]
    rfl

end FalkorLexer
