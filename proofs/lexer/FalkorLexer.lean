/-
# The Cypher lexer: total, boundary-safe, and where it disagrees with the spec

A model of FalkorDB-rs's hand-written Cypher lexer (`graph/src/parser/lexer.rs`),
its string-escape decoder (`graph/src/parser/string_escape.rs`), the literal
handling in the parser (`graph/src/parser/cypher.rs`) and the expression-height
/ nesting guards added in `e7daadac8`, with machine-checked proofs of:

* **`rs_agrees_spec`** — `read_spaces` skips exactly the openCypher
  `WHITESPACE`/`Comment` run wherever the spec accepts one (since #2908,
  `dfa326583`); an unterminated `/*` is not skipped (`rs_unterminated_general`).
* **`lexAll_total_cover`** — on every input the lexer never slices a `&str` off
  a char boundary (so never panics), and the whitespace/comment runs and token
  texts it walks over concatenate back to exactly the input.
* **`unesc_agrees_spec`** — `cypher_unescape` equals the openCypher
  `EscapedChar` decoding on every string whose escapes are the spec's
  lower-case ones (`\\ \' \" \b \f \n \r \t \uXXXX`).
* **`int_literal_correct`** — a canonically spelled integer literal, negated
  or not, evaluates to its mathematical value when that is in `i64` range and
  is rejected otherwise — overflow is never wrapped.
* **`height_sound`** / **`guarded_height_le`** — every height estimate the
  parser keeps is an upper bound on the tree it describes, so a parse that
  passes `check_depth` builds no tree taller than `MAX_TREE_DEPTH`.
* **`nested_depth_bound`** — every `parse_expr` / `CALL {}` recursion is at
  most `MAX_NESTING` deep.

and machine-checked *counterexamples* for the places where a property is false
(each one reproduced against the real Rust in `graph/tests/lean_lexer.rs`):

| theorem | what is wrong |
| --- | --- |
| `pre2908_rs_block_comment_bug`  | (fixed by #2908 `dfa326583`) `/* ... */` ended at the first `/`, not at `*/` |
| `pre2908_rs_trailing_slash_bug` | (fixed by #2908) a lone trailing `/` was skipped as whitespace |
| `pre2908_rs_unterminated_bug`   | (fixed by #2908) an unterminated `/*` was accepted |
| `foreach_unbounded`         | `FOREACH` nesting bypasses `nested`, so its recursion is unbounded |
| `set_target_unbounded`      | `SET n.a.a…` / `REMOVE n.a.a…` targets bypass `check_depth` |
| `hop_truncation_bug`        | `*4294967298` is `*2` (`i as u32`) |
| `min_noncanonical_bug`      | `-0x08000000000000000` is rejected |
| `min_postfix_bug`           | `9223372036854775808.x` escapes the overflow check |
| `unesc_plus_bug`            | `\u+041` decodes to `A` |

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `u8`, `bytes`             | `char::len_utf8`, `str::len` |
| `splitBytes`              | `&s[..n]` / `&s[n..]` — `none` is the panic "byte index is not a char boundary" |
| `rs`, `rsSlash`, `rsLine`, `rsBlock` | `Lexer::read_spaces` (lexer.rs:319-342) |
| `strScan`                 | `Lexer::lex_string_literal` (lexer.rs:495-529) |
| `btScan`                  | backtick identifier arm of `get_token` (lexer.rs:466-487) |
| `btpScan`                 | backtick parameter arm (lexer.rs:410-426) |
| `identRun`                | ASCII identifier / plain parameter arms (lexer.rs:427-432, 439-465) |
| `getTok`                  | `Lexer::get_token` (lexer.rs:359-493) |
| `lexAll`                  | the `Lexer::new` / `Lexer::next` loop (lexer.rs:287-300) |
| `unesc`                   | `cypher_unescape` (string_escape.rs:71-129) |
| `fromStrRadix16`          | `u32::from_str_radix(_, 16)` (accepts a leading `+`) |
| `specUnesc`               | openCypher `EscapedChar` (graph/src/Cypher.g4:553-554) |
| `str2int`                 | `Lexer::str2number_token`, integer half (lexer.rs:698-739) |
| `evalLit`                 | level 9 of `parse_expr_inner` (cypher.rs:2522-2540) + `parse_literal` (cypher.rs:429-453) |
| `asU32`                   | `i as u32` in the var-length hop parser (cypher.rs:1788-1798, 2989-3009) |
| `Frame`, `Op`, `step`     | the `(level, tree, height)` frames of `parse_expr_inner` and `parse_expr_return!` / `parse_operators!` (macro.rs) |
| `Reach`                   | every frame update of `parse_expr_inner`, each with its `check_depth` |
| `setTarget`               | the property loop of `parse_set_items` / `parse_remove_items` (cypher.rs:3242-3245, 3322-3325) |
| `Call`, `runCall`         | `Parser::nested` (cypher.rs:328-338), `parse_expr`, `parse_foreach_clause` (cypher.rs:3348-3422) |

## What this does NOT cover

* (Wave 4: `FalkorLexerExtra.lean` models `lex_numeric`'s scan and proves
  its slices safe, `lexNum_slices_ascii`, plus `pos`, `current_str`,
  `set_pos`, `err_ctx`, `format_error`, `Display` and `cypher_escape`.)
* The numeric *scanner* (`lex_numeric`) is abstracted here to "some number of chars
  ≥ 1": its final length is `str[pos..].chars().take(len).collect().len()`,
  a char-prefix byte length by construction, and its inner slices
  (lexer.rs:549,566,587,637) sit after an ASCII-only prefix
  (`ascii_boundary`). Its *values* are covered by `str2int`.
* Floats (`str::parse::<f64>`) are not modelled.
* Keyword lookup (`phf`) does not affect lengths and is omitted.
* The height model is over an abstract tree; the "primary adds ≤ 2 levels"
  premise (`PRIMARY_LEVELS`) is checked constructor by constructor in
  `primary_levels`, against the tree shapes read off `parse_primary_expr`.
-/

namespace FalkorLexer

set_option linter.deprecated false

/-! ## 1. Bytes and char boundaries -/

/-- `char::len_utf8`. -/
abbrev u8 (c : Char) : Nat := c.utf8Size

/-- `str::len` of a char list. -/
def bytes : List Char → Nat
  | [] => 0
  | c :: r => u8 c + bytes r

/-- `n` is a char boundary of `cs`. -/
def Boundary (cs : List Char) (n : Nat) : Prop := ∃ k, bytes (cs.take k) = n

/-- Rust slicing at byte `n`: `none` is exactly the panic. -/
def splitBytes : List Char → Nat → Option (List Char × List Char)
  | cs, 0 => some ([], cs)
  | [], _ + 1 => none
  | c :: r, n + 1 =>
    if u8 c ≤ n + 1 then (splitBytes r (n + 1 - u8 c)).map (fun p => (c :: p.1, p.2))
    else none

theorem u8_pos (c : Char) : 0 < u8 c := Char.utf8Size_pos c

theorem bnd_zero (cs : List Char) : Boundary cs 0 := ⟨0, rfl⟩

theorem bnd_cons {c : Char} {r : List Char} {n : Nat} (h : Boundary r n) :
    Boundary (c :: r) (u8 c + n) := by
  obtain ⟨k, hk⟩ := h
  exact ⟨k + 1, by simp [bytes, hk]⟩

theorem bnd_cons1 {c : Char} {r : List Char} {n : Nat} (hc : u8 c = 1) (h : Boundary r n) :
    Boundary (c :: r) (1 + n) := by
  rw [← hc]; exact bnd_cons h

/-- Slicing at a boundary succeeds and cuts after `k` chars. -/
theorem splitBytes_take (cs : List Char) (k : Nat) :
    splitBytes cs (bytes (cs.take k)) = some (cs.take k, cs.drop k) := by
  induction cs generalizing k with
  | nil => cases k <;> rfl
  | cons c r ih =>
    cases k with
    | zero => rfl
    | succ k =>
      have hp := u8_pos c
      simp only [List.take_succ_cons, List.drop_succ_cons, bytes]
      obtain ⟨m, hm⟩ : ∃ m, u8 c + bytes (r.take k) = m + 1 := ⟨_, (Nat.succ_pred_eq_of_pos (by omega)).symm⟩
      rw [hm]
      simp only [splitBytes]
      have h1 : u8 c ≤ m + 1 := by omega
      have h2 : m + 1 - u8 c = bytes (r.take k) := by omega
      simp [h1, h2, ih]

/-- ... and a boundary is exactly where slicing does not panic. -/
theorem splitBytes_some {cs : List Char} {n : Nat} {a b : List Char}
    (h : splitBytes cs n = some (a, b)) : ∃ k, a = cs.take k ∧ b = cs.drop k ∧ bytes a = n := by
  induction cs generalizing n a b with
  | nil =>
    cases n with
    | zero => simp [splitBytes] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨0, rfl, rfl, rfl⟩
    | succ n => simp [splitBytes] at h
  | cons c r ih =>
    cases n with
    | zero => simp [splitBytes] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨0, rfl, rfl, rfl⟩
    | succ n =>
      simp only [splitBytes] at h
      split at h
      · rename_i hle
        cases hs : splitBytes r (n + 1 - u8 c) with
        | none => simp [hs] at h
        | some p =>
          simp [hs] at h
          obtain ⟨rfl, rfl⟩ := h
          obtain ⟨k, h1, h2, h3⟩ := ih hs
          refine ⟨k + 1, by simp [h1], by simp [h2], ?_⟩
          simp [bytes, h3]; omega
      · simp at h

theorem boundary_iff (cs : List Char) (n : Nat) :
    Boundary cs n ↔ ∃ p, splitBytes cs n = some p := by
  constructor
  · rintro ⟨k, rfl⟩; exact ⟨_, splitBytes_take cs k⟩
  · rintro ⟨⟨a, b⟩, h⟩
    obtain ⟨k, rfl, -, h3⟩ := splitBytes_some h
    exact ⟨k, h3⟩

/-- Every inner slice `str[pos + len ..]` that `lex_numeric` takes sits after
an ASCII-only prefix, where a byte count is a char count. -/
theorem ascii_boundary (cs : List Char) (k : Nat)
    (h : ∀ c ∈ cs.take k, u8 c = 1) : bytes (cs.take k) = (cs.take k).length := by
  generalize cs.take k = l at h
  induction l with
  | nil => rfl
  | cons c r ih =>
    simp only [bytes, List.length_cons]
    rw [h c (by simp), ih (fun x hx => h x (by simp [hx]))]; omega

/-! ## 2. `read_spaces` (lexer.rs:319-342)

Since #2908 (`dfa326583`) the Rust walks bytes with a cursor `end`:
whitespace advances it, `//` jumps past the next `'\n'` (or to `str.len()`),
`/*` jumps past the first `*/` after it, and anything else — a lone `/`, an
unterminated `/*` — `break`s with `end` on that `/`. `end` only ever moves
over whole ASCII chars or to just after a `'\n'` / `*/`, so it is always a
char boundary (`rs_boundary`) and `bytes.get(end) == Some(b'/')` is "the char
at `end` is `/`"; the model therefore works on chars. `rs` is the outer
`loop`, `rsSlash` the `Some(b'/')` arm, `rsLine` the `find('\n')` jump and
`rsBlock` the `find("*/")` jump (`none` = not found ⇒ `break`). -/

/-- `' ' | '\t' | '\n'`, lexer.rs:327. -/
def isWs (c : Char) : Bool := c == ' ' || c == '\t' || c == '\n'

mutual
/-- `read_spaces(str, pos)` on `str[pos..]`: the bytes from `pos` to the final `end`. -/
def rs : List Char → Nat
  | [] => 0                                         -- `_ => break` (end of input)
  | c :: r =>
    if isWs c then 1 + rs r                         -- `end += 1`
    else if c = '/' then rsSlash r                  -- `Some(b'/')`
    else 0                                          -- `_ => break`
/-- The `Some(b'/')` arm, with the `/` behind the cursor. -/
def rsSlash : List Char → Nat
  | '/' :: r => 2 + rsLine r                        -- `str[end..].find('\n')`
  | '*' :: r =>
    match rsBlock r with                            -- `str[end + 2..].find("*/")`
    | some n => 2 + n                               -- `end += 2 + n + 2`
    | none => 0                                     -- `None => break`
  | _ => 0                                          -- `_ => break`: division
/-- After `//`: `map_or(str.len(), |n| end + n + 1)` — through the first
`'\n'`, then back to the loop; to the end of input if there is none. -/
def rsLine : List Char → Nat
  | [] => 0
  | c :: r => u8 c + (if c = '\n' then rs r else rsLine r)
/-- After `/*`: through the first `*/`, then back to the loop. -/
def rsBlock : List Char → Option Nat
  | [] => none
  | '*' :: '/' :: r => some (2 + rs r)
  | c :: r => (rsBlock r).map (u8 c + ·)
end

theorem isWs_u8 {c : Char} (h : isWs c = true) : u8 c = 1 := by
  simp [isWs] at h; rcases h with (rfl | rfl) | rfl <;> rfl

theorem u8_slash : u8 '/' = 1 := by decide
theorem u8_star : u8 '*' = 1 := by decide

/-- What each piece of `read_spaces` returns is a char boundary, so the
slices `str[end..]`, `str[end + 2..]` and the caller's `&str[pos + len..]`
never panic. -/
def RsB (cs : List Char) : Prop :=
  Boundary cs (rs cs) ∧ Boundary ('/' :: cs) (rsSlash cs) ∧ Boundary cs (rsLine cs) ∧
    ∀ n, rsBlock cs = some n → Boundary cs n

theorem rsB_all : ∀ (n : Nat) (cs : List Char), cs.length ≤ n → RsB cs := by
  intro n
  induction n with
  | zero =>
    intro cs h
    match cs, h with
    | [], _ => exact ⟨bnd_zero _, by simp [rsSlash]; exact bnd_zero _, bnd_zero _,
        fun n h => by simp [rsBlock] at h⟩
  | succ n ih =>
    intro cs h
    match cs, h with
    | [], _ => exact ⟨bnd_zero _, by simp [rsSlash]; exact bnd_zero _, bnd_zero _,
        fun n h => by simp [rsBlock] at h⟩
    | c :: r, h =>
      have hr := ih r (by simp at h; omega)
      obtain ⟨h1, h2, h3, h4⟩ := hr
      refine ⟨?_, ?_, ?_, ?_⟩
      · -- rs
        simp only [rs]
        split
        · rename_i hw; exact bnd_cons1 (isWs_u8 hw) h1
        · split
          · subst_vars; exact h2
          · exact bnd_zero _
      · -- rsSlash (c :: r)
        by_cases hc : c = '/'
        · subst hc; simp only [rsSlash]
          rw [show 2 + rsLine r = u8 '/' + (u8 '/' + rsLine r) by simp only [u8_slash]; omega]
          exact bnd_cons (bnd_cons h3)
        · by_cases hs : c = '*'
          · subst hs; simp only [rsSlash]
            split
            · rename_i m hm
              rw [show 2 + m = u8 '/' + (u8 '*' + m) by simp only [u8_slash, u8_star]; omega]
              exact bnd_cons (bnd_cons (h4 m hm))
            · exact bnd_zero _
          · have : rsSlash (c :: r) = 0 :=
              rsSlash.eq_3 _ (by intro _ h; cases h; exact hc rfl) (by intro _ h; cases h; exact hs rfl)
            rw [this]; exact bnd_zero _
      · -- rsLine
        simp only [rsLine]; split
        · exact bnd_cons h1
        · exact bnd_cons h3
      · -- rsBlock
        intro m hm
        have hgen : rsBlock (c :: r) = (rsBlock r).map (u8 c + ·) → Boundary (c :: r) m := by
          intro he
          rw [he] at hm
          cases hb : rsBlock r with
          | none => simp [hb] at hm
          | some k => simp [hb] at hm; subst hm; exact bnd_cons (h4 k hb)
        by_cases hc : c = '*'
        · subst hc
          cases r with
          | nil => simp [rsBlock] at hm
          | cons d r' =>
            by_cases hd : d = '/'
            · subst hd
              simp only [rsBlock, Option.some.injEq] at hm; subst hm
              have := (ih r' (by simp at h; omega)).1
              rw [show 2 + rs r' = u8 '*' + (u8 '/' + rs r') by simp only [u8_slash, u8_star]; omega]
              exact bnd_cons (bnd_cons this)
            · exact hgen (rsBlock.eq_3 _ _ (by intro _ _ h; cases h; exact hd rfl))
        · exact hgen (rsBlock.eq_3 _ _ (by intro _ h _; exact hc h))

theorem rs_boundary (cs : List Char) : RsB cs := rsB_all _ cs (Nat.le_refl _)

theorem rs_top_boundary (cs : List Char) : Boundary cs (rs cs) := (rs_boundary cs).1

/-- The openCypher `Comment` / `WHITESPACE` rule (Cypher.g4:697-733), for
the whitespace chars the Rust accepts: a block comment ends at `*/` and an
unterminated comment or a lone `/` is not whitespace. `none` = the rest is
not a well-formed whitespace/comment run (or the fuel ran out). -/
def specSkip : Nat → List Char → Option Nat
  | 0, _ => none
  | _ + 1, [] => some 0
  | f + 1, c :: r =>
    if isWs c then (specSkip f r).map (1 + ·)
    else match c, r with
      | '/', '/' :: r' =>
        let body := r'.takeWhile (· != '\n')
        let rest := r'.drop body.length
        match rest with
        | [] => some (2 + bytes body)
        | _ :: rest' => (specSkip f rest').map (2 + bytes body + 1 + ·)
      | '/', '*' :: r' => blockSpec f r' 2
      | _, _ => some 0
where
  /-- inside `/*`: `acc` bytes consumed so far; needs `*/`. -/
  blockSpec : Nat → List Char → Nat → Option Nat
  | 0, _, _ => none
  | _ + 1, [], _ => none                      -- unterminated: syntax error
  | f + 1, '*' :: '/' :: r, acc => (specSkip f r).map (acc + 2 + ·)
  | f + 1, c :: r, acc => blockSpec f r (acc + u8 c)

/-- The `//` jump in spec terms: up to the first `'\n'`, and past it back to the loop. -/
theorem rsLine_spec : ∀ r : List Char,
    rsLine r = match r.drop (r.takeWhile (· != '\n')).length with
      | [] => bytes (r.takeWhile (· != '\n'))
      | _ :: rest' => bytes (r.takeWhile (· != '\n')) + 1 + rs rest'
  | [] => by simp [rsLine, bytes]
  | c :: r => by
    by_cases hc : c = '\n'
    · subst hc; simp [rsLine, bytes]; rfl
    · have ih := rsLine_spec r
      have hne : (c != '\n') = true := by simp [hc]
      simp only [rsLine, if_neg hc, List.takeWhile_cons, hne, ite_true, List.length_cons,
        List.drop_succ_cons, bytes]
      rw [ih]
      generalize List.drop (List.takeWhile (fun x => x != '\n') r).length r = d
      cases d <;> simp <;> omega

/-- **`read_spaces` is the spec's whitespace/comment rule** (fixed by #2908,
`dfa326583`): wherever the openCypher `WHITESPACE`/`Comment` rule accepts the
run, the Rust skips exactly that many bytes — a block comment ends at its
first `*/`, a line comment at its `'\n'`. -/
theorem rs_agrees_spec : ∀ f : Nat,
    (∀ cs n, specSkip f cs = some n → rs cs = n) ∧
    (∀ r acc n, specSkip.blockSpec f r acc = some n → ∃ m, rsBlock r = some m ∧ n = acc + m) := by
  intro f
  induction f with
  | zero => exact ⟨fun _ _ h => by simp [specSkip] at h,
      fun _ _ _ h => by simp [specSkip.blockSpec] at h⟩
  | succ f ih =>
    obtain ⟨ihS, ihB⟩ := ih
    constructor
    · intro cs n h
      match cs with
      | [] => simp [specSkip] at h; simp [rs, h]
      | c :: r =>
        simp only [specSkip] at h
        by_cases hw : isWs c = true
        · rw [if_pos hw] at h
          cases hs : specSkip f r with
          | none => simp [hs] at h
          | some k => simp [hs] at h; subst h; simp [rs, hw, ihS r k hs]
        · rw [if_neg hw] at h
          have hrs : rs (c :: r) = if c = '/' then rsSlash r else 0 := by simp [rs, hw]
          rw [hrs]
          split at h
          next _ _ r' =>
            simp only [ite_true, rsSlash]
            rw [rsLine_spec]
            revert h
            generalize List.drop (List.takeWhile (fun x => x != '\n') r').length r' = d
            cases d with
            | nil => intro h; simp at h; omega
            | cons x rest' =>
              intro h
              cases hs : specSkip f rest' with
              | none => simp [hs] at h
              | some k => simp [hs] at h; simp only [ihS rest' k hs]; omega
          · obtain ⟨m, hm, rfl⟩ := ihB _ 2 n h
            simp [rsSlash, hm]
          · simp at h; subst h
            split
            · subst_vars; unfold rsSlash; split <;> simp_all
            · rfl
    · intro r acc n h
      match r with
      | [] => simp [specSkip.blockSpec] at h
      | c :: r =>
        have hgen : specSkip.blockSpec (f + 1) (c :: r) acc =
            specSkip.blockSpec f r (acc + u8 c) → rsBlock (c :: r) = (rsBlock r).map (u8 c + ·) →
            ∃ m, rsBlock (c :: r) = some m ∧ n = acc + m := by
          intro he hr
          rw [he] at h
          obtain ⟨m, hm, rfl⟩ := ihB r _ n h
          exact ⟨u8 c + m, by simp [hr, hm], by omega⟩
        by_cases hc : c = '*'
        · subst hc
          cases r with
          | nil =>
            exact hgen (specSkip.blockSpec.eq_4 _ _ _ _ (by intro _ _ h; cases h))
              (rsBlock.eq_3 _ _ (by intro _ _ h; cases h))
          | cons d r' =>
            by_cases hd : d = '/'
            · subst hd
              rw [specSkip.blockSpec.eq_3] at h
              cases hs : specSkip f r' with
              | none => simp [hs] at h
              | some k =>
                simp [hs] at h; subst h
                exact ⟨2 + k, by simp [rsBlock, ihS r' k hs], by omega⟩
            · exact hgen (specSkip.blockSpec.eq_4 _ _ _ _ (by intro _ _ h; cases h; exact hd rfl))
                (rsBlock.eq_3 _ _ (by intro _ _ h; cases h; exact hd rfl))
        · exact hgen (specSkip.blockSpec.eq_4 _ _ _ _ (by intro _ h _; exact hc h))
            (rsBlock.eq_3 _ _ (by intro _ h _; exact hc h))

/-! ### Before #2908 (historical)

The pre-`dfa326583` `read_spaces` (lexer.rs:314-362 at `8743953a8`) kept a
one-char lookahead `next` in three states: the outer loop (`top`), inside
`// ...` (`line`) and inside `/* ...` (`block`). Kept with the three
counterexamples it had against `specSkip` (FINDINGS: #2901), all fixed by
#2908 (`dfa326583`). -/

inductive Pre2908Mode
  | top    -- the outer `while let Some(' ' | '\t' | '\n' | '/') = next`
  | slash  -- just read a `/`, deciding on the char after it
  | line   -- inside `// ...`
  | block  -- inside `/* ...`
  deriving DecidableEq

/-- Bytes `read_spaces` skips. The Rust keeps a running `len` and undoes the
`/` with `len -= 1 + c.len_utf8()` when it is not a comment; here each state
returns what it adds, so "undo" is "the `.slash` state adds 0". -/
def pre2908Rs : Pre2908Mode → List Char → Nat
  | .top, [] => 0
  | .top, c :: r =>
    if isWs c then 1 + pre2908Rs .top r                -- len += 1; next = chars.next()
    else if c = '/' then pre2908Rs .slash r
    else 0
  | .slash, [] => 1                             -- len += 1; next = None ⇒ break  (BUG: lone '/')
  | .slash, d :: r =>
    if d = '/' then 1 + (u8 d + pre2908Rs .line r)            -- `//`
    else if d = '*' then 1 + (u8 d + pre2908Rs .block r)      -- `/*`
    else 0                                      -- len -= 1 + c.len_utf8(); break
  | .line, [] => 0
  | .line, c :: r => u8 c + (if c = '\n' then pre2908Rs .top r else pre2908Rs .line r)
  | .block, [] => 0                             -- BUG: unterminated `/*` accepted
  | .block, c :: r =>
    if c = '*' then 1 + pre2908Rs .block r              -- len += 1; continue
    else u8 c + (if c = '/' then pre2908Rs .top r else pre2908Rs .block r)   -- BUG: any '/' ends it

/-- `/* a/ -1 //*/`: the old Rust stopped the comment after `/* a/` (plus the
space: 6 bytes) and lexed `-1` as code, so `RETURN 5 /* a/ -1 //*/` gave 4
(C FalkorDB: 5). Fixed by #2908 (`dfa326583`): `rs_block_comment_fixed`. -/
theorem pre2908_rs_block_comment_bug :
    pre2908Rs .top "/* a/ -1 //*/".toList = 6 ∧
    specSkip 20 "/* a/ -1 //*/".toList = some 13 := by decide

/-- `RETURN 1 /`: the dangling `/` was swallowed (C FalkorDB: syntax error).
Fixed by #2908 (`dfa326583`): `rs_trailing_slash_fixed`. -/
theorem pre2908_rs_trailing_slash_bug :
    pre2908Rs .top " /".toList = 2 ∧ specSkip 5 " /".toList = some 1 := by decide

/-- `RETURN 1 /* x`: an unterminated block comment was accepted (C: error).
Fixed by #2908 (`dfa326583`): `rs_unterminated_fixed`. -/
theorem pre2908_rs_unterminated_bug :
    pre2908Rs .top "/* x".toList = 4 ∧ specSkip 10 "/* x".toList = none := by decide

/-! ### Now: the three inputs, and the Rust regression tests of #2908 -/

/-- The whole block comment is skipped, `/` inside it included, so
`RETURN 5 /* a/ -1 //*/` is 5, as in C. -/
theorem rs_block_comment_fixed :
    rs "/* a/ -1 //*/".toList = 13 ∧ specSkip 20 "/* a/ -1 //*/".toList = some 13 :=
  ⟨by simp [rs, rsSlash, rsBlock, isWs] <;> decide, by decide⟩

/-- A lone trailing `/` is left for the parser (division, then a syntax error). -/
theorem rs_trailing_slash_fixed :
    rs " /".toList = 1 ∧ specSkip 5 " /".toList = some 1 :=
  ⟨by simp [rs, rsSlash, isWs], by decide⟩

/-- An unterminated `/*` is not a comment: `read_spaces` stops on its `/`, so
the parser sees `/ *` and rejects the query, as C does. -/
theorem rs_unterminated_fixed : rs "/* x".toList = 0 ∧ rs " /* x".toList = 1 := by
  constructor <;> simp [rs, rsSlash, rsBlock, isWs]

/-- In general: a `/*` with no `*/` after it is never skipped. -/
theorem rs_unterminated_general (r : List Char) (h : rsBlock r = none) :
    rs ('/' :: '*' :: r) = 0 := by
  simp [rs, isWs, rsSlash, h]

/-- `block_comment_ends_at_star_slash` / `lone_slash_is_division` (lexer.rs tests). -/
example : rs " /* a/b */ + 1".toList = 11 ∧ rs " /*/ ** / * */ 2 /**/".toList = 15 ∧
    rs " // a /* b\n2".toList = 11 ∧ rs " / 2".toList = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [rs, rsSlash, rsBlock, rsLine, isWs] <;> decide

/-! ## 3. Token scanners and `get_token` (lexer.rs:359-529) -/

theorem bytes_append (a b : List Char) : bytes (a ++ b) = bytes a + bytes b := by
  induction a with
  | nil => simp [bytes]
  | cons c r ih => simp [bytes, ih]; omega

theorem bytes_take_succ {l : List Char} {k : Nat} {x : Char} {t : List Char}
    (h : l.drop k = x :: t) : bytes (l.take (k + 1)) = bytes (l.take k) + u8 x := by
  induction l generalizing k with
  | nil => simp at h
  | cons c r ih =>
    cases k with
    | zero => simp at h; obtain ⟨rfl, rfl⟩ := h; simp [bytes]
    | succ k => simp at h; simp [bytes, ih h]; omega

/-- `&s[i..j]`: panics unless both ends are boundaries (and `i ≤ j`). -/
def slice (cs : List Char) (i j : Nat) : Option (List Char) :=
  match splitBytes cs j with
  | none => none
  | some (a, _) => (splitBytes a i).map (·.2)

theorem slice_ok (cs : List Char) {ki kj : Nat} (h : ki ≤ kj) :
    slice cs (bytes (cs.take ki)) (bytes (cs.take kj)) = some ((cs.take kj).drop ki) := by
  unfold slice
  rw [splitBytes_take]
  simp only
  have : cs.take ki = (cs.take kj).take ki := by rw [List.take_take]; congr; omega
  rw [this, splitBytes_take]; rfl

/-- The string-literal scan of `lex_string_literal`, on the chars after the
opening quote `q`. `some n`: the body is `n` bytes and a closing `q` follows.
`none`: unterminated, or a `\` at end of input — both `Err`. -/
def strScan (q : Char) : List Char → Option Nat
  | [] => none                                  -- !end ⇒ Err("Unterminated string")
  | c :: r =>
    if c = '\\' then
      match r with
      | [] => none                              -- chars.next() = None ⇒ Err
      | c2 :: r' => (strScan q r').map (u8 c2 + u8 c + ·)   -- len += c2; len += c
    else if c = q then some 0                   -- end = true; break
    else (strScan q r).map (u8 c + ·)           -- len += c.len_utf8()

theorem strScan_spec {q : Char} : ∀ {r : List Char} {n : Nat}, strScan q r = some n →
    ∃ k t, bytes (r.take k) = n ∧ r.drop k = q :: t
  | [], _, h => by simp [strScan] at h
  | [c], n, h => by
    simp only [strScan] at h
    split at h
    · simp at h
    · split at h
      · subst_vars; simp at h; subst h; exact ⟨0, [], rfl, rfl⟩
      · simp at h
  | c :: c2 :: r, n, h => by
    simp only [strScan] at h
    split at h
    · cases hs : strScan q r with
      | none => simp [hs] at h
      | some m =>
        simp [hs] at h
        obtain ⟨k, t, h1, h2⟩ := strScan_spec hs
        refine ⟨k + 2, t, ?_, by simpa using h2⟩
        simp [bytes, h1]; omega
    · split at h
      · subst_vars; simp at h; subst h; exact ⟨0, _, rfl, rfl⟩
      · cases hs : strScan q (c2 :: r) with
        | none => simp [hs] at h
        | some m =>
          simp [hs] at h
          obtain ⟨k, t, h1, h2⟩ := strScan_spec hs
          refine ⟨k + 1, t, ?_, by simpa using h2⟩
          simp [bytes, h1]; omega

/-- Backtick identifier scan (lexer.rs:467-475), after the opening backtick:
`some n` = `n` body bytes, then a closing backtick. -/
def btScan : List Char → Option Nat
  | [] => none                                  -- !end ⇒ Err(&str[pos..pos + len])
  | c :: r => if c = '`' then some 0 else (btScan r).map (u8 c + ·)

theorem btScan_spec : ∀ {r : List Char} {n : Nat}, btScan r = some n →
    ∃ k t, bytes (r.take k) = n ∧ r.drop k = '`' :: t
  | [], _, h => by simp [btScan] at h
  | c :: r, n, h => by
    simp only [btScan] at h
    split at h
    · subst_vars; simp at h; subst h; exact ⟨0, _, rfl, rfl⟩
    · cases hs : btScan r with
      | none => simp [hs] at h
      | some m =>
        simp [hs] at h
        obtain ⟨k, t, h1, h2⟩ := btScan_spec hs
        exact ⟨k + 1, t, by simp [bytes, h1]; omega, by simpa using h2⟩

/-- Backtick parameter scan (lexer.rs:413-419): the closing backtick is
counted (`len += ch.len_utf8()` before the `==` test). -/
def btpScan : List Char → Option Nat
  | [] => none
  | c :: r => if c = '`' then some (u8 c) else (btpScan r).map (u8 c + ·)

theorem btpScan_spec : ∀ {r : List Char} {m : Nat}, btpScan r = some m →
    ∃ k t, bytes (r.take k) + 1 = m ∧ r.drop k = '`' :: t
  | [], _, h => by simp [btpScan] at h
  | c :: r, n, h => by
    simp only [btpScan] at h
    split at h
    · subst_vars; simp at h; subst h; exact ⟨0, _, rfl, rfl⟩
    · cases hs : btpScan r with
      | none => simp [hs] at h
      | some m =>
        simp [hs] at h
        obtain ⟨k, t, h1, h2⟩ := btpScan_spec hs
        exact ⟨k + 1, t, by simp [bytes]; omega, by simpa using h2⟩

def isDigit (c : Char) : Bool := '0' ≤ c && c ≤ '9'
def isIdStart (c : Char) : Bool := ('a' ≤ c && c ≤ 'z') || ('A' ≤ c && c ≤ 'Z') || c == '_'
def isIdChar (c : Char) : Bool := isIdStart c || isDigit c

theorem u8_ascii {c : Char} (h : c ≤ 'z') : u8 c = 1 := by
  have h1 : c.val ≤ 127 := by
    have := h; simp [Char.le_def] at this
    exact UInt32.le_iff_toNat_le.mpr (by have := UInt32.le_iff_toNat_le.mp this; simp at this ⊢; omega)
  simp [u8, Char.utf8Size, h1]

theorem le_z_trans {a c : Char} (h : c ≤ a) (ha : a ≤ 'z') : c ≤ 'z' := by
  simp [Char.le_def] at *
  exact UInt32.le_iff_toNat_le.mpr (by
    have := UInt32.le_iff_toNat_le.mp h; have := UInt32.le_iff_toNat_le.mp ha; omega)

theorem isIdChar_u8 {c : Char} (h : isIdChar c = true) : u8 c = 1 := by
  simp [isIdChar, isIdStart, isDigit] at h
  rcases h with ((⟨-, h⟩ | ⟨-, h⟩) | rfl) | ⟨-, h⟩
  · exact u8_ascii h
  · exact u8_ascii (le_z_trans h (by decide))
  · rfl
  · exact u8_ascii (le_z_trans h (by decide))

/-- `while let Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_') = chars.next() { len += 1 }` -/
def identRun : List Char → Nat
  | [] => 0
  | c :: r => if isIdChar c then 1 + identRun r else 0

theorem identRun_bytes : ∀ r : List Char, bytes (r.take (identRun r)) = identRun r
  | [] => rfl
  | c :: r => by
    simp only [identRun]
    split
    · rename_i h
      rw [Nat.add_comm, List.take_succ_cons]
      simp only [bytes, isIdChar_u8 h, identRun_bytes r]; omega
    · rfl

/-- One-char tokens of `get_token` (lexer.rs:366-375, 380, 394-395, 401, 488). -/
def singles : List Char := ['[', ']', '{', '}', '(', ')', '%', '^', '*', '/', '-', ',', ':', '|', ';']

@[simp] theorem us_dollar : '$'.utf8Size = 1 := rfl
@[simp] theorem us_bt : '`'.utf8Size = 1 := rfl
@[simp] theorem us_sq : '\''.utf8Size = 1 := rfl
@[simp] theorem us_dq : '"'.utf8Size = 1 := rfl
@[simp] theorem us_bs : '\\'.utf8Size = 1 := rfl

theorem singles_u8 : ∀ c ∈ singles, u8 c = 1 := by decide

/-- `Lexer::get_token` (lexer.rs:359-493), reduced to what decides lengths
and slices. `none` = a panic (a slice off a char boundary); `.ok n` = a token
of `n` bytes (`0` only for `EndOfFile`); `.error` = an `Err` token, at which the
parser stops. `num cs` is the char count `lex_numeric` settles on (≥ 1, or
≥ 2 after `.`; its result is `str[pos..].chars().take(len).collect().len()`).
`esc body` is whether `cypher_unescape` accepts the string body. -/
def getTokC (num : List Char → Nat) (esc : List Char → Bool) (c : Char) (r : List Char) :
    Option (Except Unit Nat) :=
    if c ∈ singles then some (.ok 1)
    else if c = '+' then some (.ok (if r.head? = some '=' then 2 else 1))
    else if c = '=' then some (.ok (if r.head? = some '~' then 2 else 1))
    else if c = '<' then some (.ok (if r.head? = some '=' ∨ r.head? = some '>' then 2 else 1))
    else if c = '>' then some (.ok (if r.head? = some '=' then 2 else 1))
    else if c = '.' then
      if r.head? = some '.' then some (.ok 2)
      else if r.head?.map isDigit = some true then
        some (.ok (bytes ((c :: r).take (max 2 (num (c :: r))))))
      else some (.ok 1)
    else if c = '\'' ∨ c = '"' then
      match strScan c r with
      | none => some (.error ())
      | some n => (slice (c :: r) 1 (1 + n)).map fun body =>
          if esc body then .ok (n + 2) else .error ()
    else if isDigit c then some (.ok (bytes ((c :: r).take (max 1 (num (c :: r))))))
    else if c = '$' then
      match r with
      | [] => some (.error ())
      | f :: r' =>
        if f = '`' then
          match btpScan r' with
          | none => some (.error ())
          | some m => (slice (c :: r) 2 (2 + m - 1)).map fun _ => .ok (2 + m)
        else if isIdChar f then
          (slice (c :: r) 1 (2 + identRun r')).map fun _ => .ok (2 + identRun r')
        else some (.error ())
    else if isIdStart c then
      (slice (c :: r) 0 (1 + identRun r)).map fun _ => .ok (1 + identRun r)
    else if c = '`' then
      match btScan r with
      | none => (slice (c :: r) 0 (1 + bytes r)).map fun _ => .error ()
      | some n => (slice (c :: r) 1 (1 + n)).map fun _ => .ok (n + 2)
    else some (.error ())

def getTok (num : List Char → Nat) (esc : List Char → Bool) : List Char → Option (Except Unit Nat)
  | [] => some (.ok 0)
  | c :: r => getTokC num esc c r

/-- What `getTok` promises: no panic, and a non-EOF token is a non-empty
char prefix. -/
def TokOk (cs : List Char) (res : Option (Except Unit Nat)) : Prop :=
  ∃ e, res = some e ∧ ∀ n, e = .ok n → (n = 0 ∧ cs = []) ∨ ∃ k, 0 < k ∧ bytes (cs.take k) = n

theorem bytes_take1 (c : Char) (r : List Char) : bytes ((c :: r).take 1) = u8 c := by
  simp [bytes]

theorem slice_ok' (cs : List Char) {i j ki kj : Nat} (hi : bytes (cs.take ki) = i)
    (hj : bytes (cs.take kj) = j) (h : ki ≤ kj) : ∃ b, slice cs i j = some b := by
  subst hi hj; exact ⟨_, slice_ok cs h⟩

theorem tok_one {c : Char} {r : List Char} (hc : u8 c = 1) :
    ∃ k, 0 < k ∧ bytes ((c :: r).take k) = 1 := ⟨1, by omega, by simp [bytes, hc]⟩

theorem tok_two {c d : Char} {r t : List Char} (hc : u8 c = 1) (hr : r.head? = some d)
    (hd : u8 d = 1) : ∃ k, 0 < k ∧ bytes ((c :: r).take k) = 2 := by
  cases r with
  | nil => simp at hr
  | cons d' t => simp at hr; subst hr; exact ⟨2, by omega, by simp [bytes, hc, hd]⟩

theorem getTok_ok (num : List Char → Nat) (esc : List Char → Bool) (cs : List Char) :
    TokOk cs (getTok num esc cs) := by
  match cs with
  | [] => exact ⟨_, rfl, fun n h => by cases h; simp⟩
  | c :: r =>
    show TokOk (c :: r) (getTokC num esc c r)
    unfold getTokC
    by_cases h1 : c ∈ singles
    · rw [if_pos h1]; exact ⟨_, rfl, fun n e => by cases e; exact .inr (tok_one (singles_u8 c h1))⟩
    rw [if_neg h1]
    by_cases h2 : c = '+'
    · rw [if_pos h2]; subst h2; refine ⟨_, rfl, fun n e => ?_⟩; cases e
      by_cases h : r.head? = some '='
      · rw [if_pos h]; exact .inr (tok_two (t := []) rfl h rfl)
      · rw [if_neg h]; exact .inr (tok_one rfl)
    rw [if_neg h2]
    by_cases h3 : c = '='
    · rw [if_pos h3]; subst h3; refine ⟨_, rfl, fun n e => ?_⟩; cases e
      by_cases h : r.head? = some '~'
      · rw [if_pos h]; exact .inr (tok_two (t := []) rfl h rfl)
      · rw [if_neg h]; exact .inr (tok_one rfl)
    rw [if_neg h3]
    by_cases h4 : c = '<'
    · rw [if_pos h4]; subst h4; refine ⟨_, rfl, fun n e => ?_⟩; cases e
      by_cases h : r.head? = some '=' ∨ r.head? = some '>'
      · rw [if_pos h]; rcases h with h | h
        · exact .inr (tok_two (t := []) rfl h rfl)
        · exact .inr (tok_two (t := []) rfl h rfl)
      · rw [if_neg h]; exact .inr (tok_one rfl)
    rw [if_neg h4]
    by_cases h5 : c = '>'
    · rw [if_pos h5]; subst h5; refine ⟨_, rfl, fun n e => ?_⟩; cases e
      by_cases h : r.head? = some '='
      · rw [if_pos h]; exact .inr (tok_two (t := []) rfl h rfl)
      · rw [if_neg h]; exact .inr (tok_one rfl)
    rw [if_neg h5]
    by_cases h6 : c = '.'
    · rw [if_pos h6]; subst h6
      by_cases h : r.head? = some '.'
      · rw [if_pos h]; exact ⟨_, rfl, fun n e => by cases e; exact .inr (tok_two (t := []) rfl h rfl)⟩
      rw [if_neg h]
      by_cases h' : r.head?.map isDigit = some true
      · rw [if_pos h']
        exact ⟨_, rfl, fun n e => by cases e; exact .inr ⟨max 2 (num ('.' :: r)), by omega, rfl⟩⟩
      · rw [if_neg h']; exact ⟨_, rfl, fun n e => by cases e; exact .inr (tok_one rfl)⟩
    rw [if_neg h6]
    by_cases h7 : c = '\'' ∨ c = '"'
    · rw [if_pos h7]
      have hc : u8 c = 1 := by rcases h7 with rfl | rfl <;> rfl
      cases hs : strScan c r with
      | none => exact ⟨_, rfl, fun n e => by cases e⟩
      | some n =>
        simp only
        obtain ⟨k, t, h1, h2⟩ := strScan_spec hs
        obtain ⟨b, hb⟩ := slice_ok' (c :: r) (i := 1) (j := 1 + n) (ki := 1) (kj := k + 1)
          (by simp [bytes, hc]) (by simp [bytes, h1, hc] <;> omega) (by omega)
        rw [hb]
        refine ⟨_, rfl, fun m e => ?_⟩
        simp only at e
        by_cases he : esc b = true
        · rw [if_pos he] at e
          cases e
          refine .inr ⟨k + 2, by omega, ?_⟩
          have h3 : bytes (r.take (k + 1)) = bytes (r.take k) + u8 c := bytes_take_succ h2
          simp only [List.take_succ_cons, bytes, h3, h1, hc]; omega
        · rw [if_neg he] at e; cases e
    rw [if_neg h7]
    by_cases h8 : isDigit c = true
    · rw [if_pos h8]
      exact ⟨_, rfl, fun n e => by cases e; exact .inr ⟨max 1 (num (c :: r)), by omega, rfl⟩⟩
    rw [if_neg h8]
    by_cases h9 : c = '$'
    · rw [if_pos h9]; subst h9
      match r with
      | [] => exact ⟨_, rfl, fun n e => by cases e⟩
      | f :: r' =>
        simp only
        by_cases hf : f = '`'
        · rw [if_pos hf]; subst hf
          cases hs : btpScan r' with
          | none => exact ⟨_, rfl, fun n e => by cases e⟩
          | some m =>
            simp only
            obtain ⟨k, t, h1, h2⟩ := btpScan_spec hs
            obtain ⟨b, hb⟩ := slice_ok' ('$' :: '`' :: r') (i := 2) (j := 2 + m - 1) (ki := 2) (kj := k + 2)
              (by rfl) (by simp [bytes, ← h1]; omega) (by omega)
            rw [hb]
            refine ⟨_, rfl, fun n e => ?_⟩
            cases e
            refine .inr ⟨k + 3, by omega, ?_⟩
            have h3 := bytes_take_succ h2
            simp only [List.take_succ_cons, bytes, h3]
            simp [u8] at h1 ⊢; omega
        rw [if_neg hf]
        by_cases hf' : isIdChar f = true
        · rw [if_pos hf']
          have hb2 : bytes (('$' :: f :: r').take (2 + identRun r')) = 2 + identRun r' := by
            rw [Nat.add_comm, List.take_succ_cons, List.take_succ_cons]
            simp only [bytes, isIdChar_u8 hf', identRun_bytes]; simp [u8]; omega
          obtain ⟨b, hb⟩ := slice_ok' ('$' :: f :: r') (i := 1) (j := 2 + identRun r') (ki := 1) (kj := 2 + identRun r')
            (by rfl) hb2 (by omega)
          rw [hb]
          exact ⟨_, rfl, fun n e => by cases e; exact .inr ⟨_, by omega, hb2⟩⟩
        · rw [if_neg hf']; exact ⟨_, rfl, fun n e => by cases e⟩
    rw [if_neg h9]
    by_cases h10 : isIdStart c = true
    · rw [if_pos h10]
      have hc : u8 c = 1 := isIdChar_u8 (by simp [isIdChar, h10])
      have hb2 : bytes ((c :: r).take (1 + identRun r)) = 1 + identRun r := by
        rw [Nat.add_comm, List.take_succ_cons]
        simp only [bytes, hc, identRun_bytes]; omega
      obtain ⟨b, hb⟩ := slice_ok' (c :: r) (i := 0) (j := 1 + identRun r) (ki := 0) (kj := 1 + identRun r) rfl hb2 (by omega)
      rw [hb]
      exact ⟨_, rfl, fun n e => by cases e; exact .inr ⟨_, by omega, hb2⟩⟩
    rw [if_neg h10]
    by_cases h11 : c = '`'
    · rw [if_pos h11]; subst h11
      cases hs : btScan r with
      | none =>
        simp only
        obtain ⟨b, hb⟩ := slice_ok' ('`' :: r) (i := 0) (j := 1 + bytes r) (ki := 0) (kj := r.length + 1) rfl
          (by simp [bytes] <;> omega) (by omega)
        rw [hb]; exact ⟨_, rfl, fun n e => by cases e⟩
      | some n =>
        simp only
        obtain ⟨k, t, h1, h2⟩ := btScan_spec hs
        obtain ⟨b, hb⟩ := slice_ok' ('`' :: r) (i := 1) (j := 1 + n) (ki := 1) (kj := k + 1) rfl
          (by simp [bytes, h1] <;> omega) (by omega)
        rw [hb]
        refine ⟨_, rfl, fun m e => ?_⟩
        cases e
        refine .inr ⟨k + 2, by omega, ?_⟩
        have h3 := bytes_take_succ h2
        simp only [List.take_succ_cons, bytes, h3, h1]; simp [u8]; omega
    rw [if_neg h11]
    exact ⟨_, rfl, fun n e => by cases e⟩

/-! ## 4. The token loop: never panics, and covers the input exactly -/

/-- `Lexer::new` then `Lexer::next` until `EndOfFile` or an `Err` token
(lexer.rs:287-300): skip `read_spaces`, slice there, take `get_token`, slice
after it. Returns the whitespace runs and token texts in order, or `none` if
any slice would panic. -/
def lexAll (num : List Char → Nat) (esc : List Char → Bool) :
    Nat → List Char → Option (List (List Char))
  | 0, _ => some []
  | f + 1, cs =>
    match splitBytes cs (rs cs) with
    | none => none
    | some (sp, rest) =>
      match getTok num esc rest with
      | none => none
      | some (.error _) => some [sp, rest]      -- the parser stops at an Err token
      | some (.ok 0) => some [sp, rest]         -- EndOfFile
      | some (.ok (n + 1)) =>
        match splitBytes rest (n + 1) with
        | none => none
        | some (tok, rest') => (lexAll num esc f rest').map (sp :: tok :: ·)

/-- **The lexer is total and its segments cover the input exactly**: for
every input (any `&str`), no slice panics, and the whitespace/comment runs and
token texts concatenate back to the input. Holds for any numeric scanner and
any escape decoder, since neither can move a slice off a char boundary. -/
theorem lexAll_total_cover (num : List Char → Nat) (esc : List Char → Bool) :
    ∀ (f : Nat) (cs : List Char), cs.length < f →
      ∃ segs, lexAll num esc f cs = some segs ∧ segs.flatten = cs
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | f + 1, cs, hlen => by
    obtain ⟨k1, hk1⟩ := rs_top_boundary cs
    have hs1 : splitBytes cs (rs cs) = some (cs.take k1, cs.drop k1) := by
      rw [← hk1]; exact splitBytes_take cs k1
    obtain ⟨e, he, hok⟩ := getTok_ok num esc (cs.drop k1)
    simp only [lexAll, hs1, he]
    match e, hok with
    | .error _, _ => exact ⟨_, rfl, by simp⟩
    | .ok 0, _ => exact ⟨_, rfl, by simp⟩
    | .ok (n + 1), hok =>
      rcases hok (n + 1) rfl with ⟨h0, -⟩ | ⟨k2, hk2pos, hk2⟩
      · omega
      simp only
      rw [← hk2, splitBytes_take]
      simp only
      have hne : cs.drop k1 ≠ [] := by
        intro h; rw [h] at hk2; simp [bytes] at hk2
      have hlt : ((cs.drop k1).drop k2).length < f := by
        have : k1 < cs.length := by simpa using hne
        simp only [List.length_drop]; omega
      obtain ⟨segs, h1, h2⟩ := lexAll_total_cover num esc f _ hlt
      rw [h1]
      refine ⟨_, rfl, ?_⟩
      simp only [List.flatten_cons, h2, List.take_append_drop]

/-- Instantiated at the fuel the loop needs: every input. -/
theorem lexer_never_panics (num : List Char → Nat) (esc : List Char → Bool) (cs : List Char) :
    ∃ segs, lexAll num esc (cs.length + 1) cs = some segs ∧ segs.flatten = cs :=
  lexAll_total_cover num esc _ cs (Nat.lt_succ_self _)

/-! ## 5. `cypher_unescape` against openCypher `EscapedChar` -/

/-- One hex digit (`char::to_digit(16)` as `from_str_radix` uses it). -/
def hexDigit? (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - 48)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 87)
  else if 'A' ≤ c ∧ c ≤ 'F' then some (c.toNat - 55)
  else none

/-- All-digit value, most significant first; `none` on a non-digit. -/
def hexVal : List Char → Nat → Option Nat
  | [], acc => some acc
  | c :: r, acc => (hexDigit? c).bind fun d => hexVal r (acc * 16 + d)

/-- `u32::from_str_radix(s, 16)`: empty is `Err`; a leading `+` is skipped
when something follows it; then every char must be a digit and the value must
fit `u32`. -/
def fromStrRadix16 (s : List Char) : Option Nat :=
  let ds := match s with
    | '+' :: t => if t = [] then none else some t
    | [] => none
    | _ => some s
  (ds.bind (hexVal · 0)).bind fun v => if v < 2 ^ 32 then some v else none

/-- The one-char escapes of string_escape.rs:78-88. -/
def simpleEsc (e : Char) : Option Char :=
  if e = 'a' then some '\x07' else if e = 'b' then some '\x08'
  else if e = 'f' then some '\x0c' else if e = 'n' then some '\n'
  else if e = 'r' then some '\r' else if e = 't' then some '\t'
  else if e = 'v' then some '\x0b' else if e = '\\' then some '\\'
  else if e = '\'' then some '\'' else if e = '"' then some '"'
  else if e = '?' then some '?' else none

/-- `cypher_unescape` (string_escape.rs:71-129), with fuel (≥ input length is
enough). `take(w).collect()` then `hex.len() != w` compares *bytes*. -/
def unescF : Nat → List Char → Except String (List Char)
  | 0, _ => .error "fuel"
  | _ + 1, [] => .ok []
  | f + 1, c :: r =>
    if c = '\\' then
      match r with
      | [] => .error "Unterminated escape sequence at end of string"
      | e :: r' =>
        match simpleEsc e with
        | some d => (unescF f r').map (d :: ·)
        | none =>
          if e = 'u' ∨ e = 'U' then
            let w := if e = 'u' then 4 else 8
            let hex := r'.take w
            if bytes hex ≠ w then .error "Invalid unicode escape"
            else match fromStrRadix16 hex with
              | none => .error "Invalid unicode escape"
              | some v =>
                if v.isValidChar then (unescF f (r'.drop w)).map (Char.ofNat v :: ·)
                else .error "Invalid unicode code point"   -- char::from_u32 = None
          else (unescF f r').map (fun t => '\\' :: e :: t)  -- unknown: kept as-is
    else (unescF f r).map (c :: ·)

/-- The escapes Cypher.g4:553-554 gives a lower-case spelling to. -/
def specSimple (e : Char) : Option Char :=
  if e = '\\' then some '\\' else if e = '\'' then some '\''
  else if e = '"' then some '"' else if e = 'b' then some '\x08'
  else if e = 'f' then some '\x0c' else if e = 'n' then some '\n'
  else if e = 'r' then some '\r' else if e = 't' then some '\t'
  else none

/-- Exactly `w` hex digits. -/
def hexN (w : Nat) (l : List Char) : Option Nat :=
  if l.length = w then hexVal l 0 else none

/-- openCypher `StringLiteral` body decoding, lower-case escapes, with
`\uXXXX` (UTF-16 unit, 4 digits) and `\UXXXXXXXX` (8 digits) as Neo4j reads
the grammar's `('U'|'u')` alternatives. `none` = not a valid literal (an
unknown escape is a syntax error in the grammar), or a hex escape naming no
Unicode scalar value. -/
def specUnescF : Nat → List Char → Option (List Char)
  | 0, _ => none
  | _ + 1, [] => some []
  | f + 1, c :: r =>
    if c = '\\' then
      match r with
      | [] => none
      | e :: r' =>
        match specSimple e with
        | some d => (specUnescF f r').map (d :: ·)
        | none =>
          if e = 'u' ∨ e = 'U' then
            let w := if e = 'u' then 4 else 8
            match hexN w (r'.take w) with
            | none => none
            | some v => if v.isValidChar then (specUnescF f (r'.drop w)).map (Char.ofNat v :: ·)
                        else none
          else none
    else (specUnescF f r).map (c :: ·)

theorem specSimple_simpleEsc {e d : Char} (h : specSimple e = some d) : simpleEsc e = some d := by
  simp only [specSimple] at h
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  split at h; · subst_vars; cases h; rfl
  cases h

theorem specSimple_none_u : specSimple 'u' = none ∧ specSimple 'U' = none := by decide
theorem simpleEsc_none_u : simpleEsc 'u' = none ∧ simpleEsc 'U' = none := by decide

theorem hexDigit_ascii {c : Char} {d : Nat} (h : hexDigit? c = some d) : u8 c = 1 := by
  unfold hexDigit? at h
  split at h
  · rename_i h1; exact u8_ascii (le_z_trans h1.2 (by decide))
  split at h
  · rename_i h1; exact u8_ascii (le_z_trans h1.2 (by decide))
  split at h
  · rename_i h1; exact u8_ascii (le_z_trans h1.2 (by decide))
  · simp at h

theorem hexVal_ascii : ∀ {l : List Char} {acc v : Nat}, hexVal l acc = some v → bytes l = l.length
  | [], _, _, _ => rfl
  | c :: r, acc, v, h => by
    simp only [hexVal] at h
    cases hd : hexDigit? c with
    | none => simp [hd] at h
    | some d =>
      simp [hd] at h
      simp [bytes, hexDigit_ascii hd, hexVal_ascii h]; omega

theorem hexVal_not_plus {c : Char} {r : List Char} {acc v : Nat}
    (h : hexVal (c :: r) acc = some v) : c ≠ '+' := by
  rintro rfl; simp [hexVal, hexDigit?] at h

theorem toNat_le {c d : Char} (h : c ≤ d) : c.toNat ≤ d.toNat :=
  UInt32.le_iff_toNat_le.mp h

theorem hexVal_lt : ∀ {l : List Char} {acc v : Nat}, hexVal l acc = some v →
    v < (acc + 1) * 16 ^ l.length
  | [], acc, v, h => by simp [hexVal] at h; subst h; simp
  | c :: r, acc, v, h => by
    simp only [hexVal] at h
    cases hd : hexDigit? c with
    | none => simp [hd] at h
    | some d =>
      simp [hd] at h
      have h1 := hexVal_lt h
      have hd16 : d < 16 := by
        unfold hexDigit? at hd
        split at hd
        · rename_i hc; cases hd
          have h1 := toNat_le hc.1; have h2 := toNat_le hc.2
          have : ('9' : Char).toNat = 57 := rfl
          have : ('0' : Char).toNat = 48 := rfl
          omega
        split at hd
        · rename_i _ hc; cases hd
          have h1 := toNat_le hc.1; have h2 := toNat_le hc.2
          have : ('f' : Char).toNat = 102 := rfl
          have : ('a' : Char).toNat = 97 := rfl
          omega
        split at hd
        · rename_i _ _ hc; cases hd
          have h1 := toNat_le hc.1; have h2 := toNat_le hc.2
          have : ('F' : Char).toNat = 70 := rfl
          have : ('A' : Char).toNat = 65 := rfl
          omega
        · cases hd
      have : (acc * 16 + d + 1) ≤ (acc + 1) * 16 := by omega
      calc v < (acc * 16 + d + 1) * 16 ^ r.length := h1
        _ ≤ (acc + 1) * 16 * 16 ^ r.length := Nat.mul_le_mul_right _ this
        _ = (acc + 1) * 16 ^ (r.length + 1) := by rw [Nat.pow_succ]; ac_rfl

theorem hexN_rust {w : Nat} {l : List Char} {v : Nat} (hw0 : 0 < w) (hw : w ≤ 8) (h : hexN w l = some v) :
    bytes l = w ∧ fromStrRadix16 l = some v := by
  unfold hexN at h
  split at h
  · rename_i hl
    refine ⟨by rw [hexVal_ascii h, hl], ?_⟩
    have hlt : v < 2 ^ 32 := by
      have := hexVal_lt h; simp at this
      have h2 : 16 ^ l.length ≤ 16 ^ 8 := Nat.pow_le_pow_right (by omega) (by omega)
      have : (16 : Nat) ^ 8 = 2 ^ 32 := by decide
      omega
    unfold fromStrRadix16
    match l, h with
    | [], h => simp at hl; omega
    | c :: r, h =>
      have hc := hexVal_not_plus h
      have : (match c :: r with
          | '+' :: t => if t = [] then none else some t
          | [] => none
          | _ => some (c :: r)) = some (c :: r) := by
        split
        · rename_i heq; simp at heq; exact absurd heq.1 hc
        · simp at *
        · rfl
      simp only [this, Option.bind_some, h, hlt, if_true]
  · simp at h

/-- **`cypher_unescape` extends the openCypher decoding**: every string body
the spec accepts (lower-case escapes, `\uXXXX`, `\UXXXXXXXX`) is decoded by
the Rust to exactly the spec's string. -/
theorem unesc_agrees_spec : ∀ (f : Nat) (cs out : List Char),
    specUnescF f cs = some out → unescF f cs = .ok out
  | 0, _, _, h => by simp [specUnescF] at h
  | f + 1, [], out, h => by simp [specUnescF] at h; subst h; rfl
  | f + 1, c :: r, out, h => by
    simp only [specUnescF] at h
    simp only [unescF]
    by_cases hc : c = '\\'
    · rw [if_pos hc] at h ⊢
      match r, h with
      | [], h => simp at h
      | e :: r', h =>
        simp only at h ⊢
        cases hs : specSimple e with
        | some d =>
          rw [hs] at h
          rw [specSimple_simpleEsc hs]
          simp only at h ⊢
          cases h2 : specUnescF f r' with
          | none => simp [h2] at h
          | some t => simp [h2] at h; subst h; simp [unesc_agrees_spec f r' t h2, Except.map]
        | none =>
          rw [hs] at h
          simp only at h
          by_cases hu : e = 'u' ∨ e = 'U'
          · rw [if_pos hu] at h
            have hse : simpleEsc e = none := by
              rcases hu with rfl | rfl <;> decide
            rw [hse]; simp only; rw [if_pos hu]
            generalize hw : (if e = 'u' then 4 else 8) = w at h ⊢
            have hw8 : w ≤ 8 := by rw [← hw]; split <;> omega
            have hw0 : 0 < w := by rw [← hw]; split <;> omega
            cases hx : hexN w (r'.take w) with
            | none => simp [hx] at h
            | some v =>
              rw [hx] at h
              obtain ⟨hb, hf⟩ := hexN_rust hw0 hw8 hx
              simp only [hb, ne_eq, not_true_eq_false, if_false, hf]
              simp only at h
              by_cases hv : v.isValidChar
              · rw [if_pos hv] at h; rw [if_pos hv]
                cases h2 : specUnescF f (r'.drop w) with
                | none => simp [h2] at h
                | some t =>
                  simp [h2] at h; subst h
                  simp [unesc_agrees_spec f _ t h2, Except.map]
              · rw [if_neg hv] at h; simp at h
          · rw [if_neg hu] at h; simp at h
    · rw [if_neg hc] at h ⊢
      cases h2 : specUnescF f r with
      | none => simp [h2] at h
      | some t => simp [h2] at h; subst h; simp [unesc_agrees_spec f r t h2, Except.map]

/-- `\u+041`: `from_str_radix` skips the `+`, so three hex digits decode to
`A`. The spec wants four hex digits; C FalkorDB keeps the text `\u+041`. -/
theorem unesc_plus_bug :
    (unescF 10 "\\u+041".toList).toOption = some ['A'] ∧
    specUnescF 10 "\\u+041".toList = none := by decide

/-- The grammar's `\N`, `\B`, … (upper-case) are kept verbatim, as in C
FalkorDB; the grammar reads `\N` as a newline. -/
theorem unesc_upper_kept : (unescF 5 "\\N".toList).toOption = some ['\\', 'N'] := by decide

/-- `\uD800` (a lone surrogate) and `\U00110000` are rejected by both. -/
example : (unescF 10 "\\uD800".toList).toOption = none ∧ specUnescF 10 "\\uD800".toList = none ∧
    (unescF 12 "\\U00110000".toList).toOption = none := by decide

example : (unescF 10 "\\u00e9\\t".toList).toOption = some ['é', '\t'] := by decide

/-! ## 6. Integer literals: overflow is rejected, not wrapped -/

def I64MIN : Int := -2 ^ 63
def I64MAX : Int := 2 ^ 63 - 1

/-- Value of a digit string, most significant first (`from_str_radix`'s
accumulation, before its overflow check). -/
def digitsVal (radix : Nat) (ds : List Nat) : Nat := ds.foldl (fun a d => a * radix + d) 0

/-- Canonical digits of `n` (no leading zeros); fuel `f` > `n` suffices. -/
def canonF (radix : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n < radix then [n] else canonF radix f (n / radix) ++ [n % radix]

def canon (radix n : Nat) : List Nat := canonF radix (n + 1) n

/-- `Lexer::str2number_token`, integer half (lexer.rs:711-738), on the digits
after the radix prefix (`0x`, `0o`, `0b`, or the leading `0` of an octal).
`MIN_I64` (lexer.rs:271-277) holds exactly the canonical spellings of `2^63`,
which are returned as `i64::MIN` for the parser to sort out. -/
def str2int (radix : Nat) (ds : List Nat) : Except String Int :=
  if ds = canon radix (2 ^ 63) then .ok I64MIN
  else if ds = [] then .error "Invalid input"
  else if (digitsVal radix ds : Int) ≤ I64MAX then .ok (digitsVal radix ds)
  else .error "Integer overflow"                -- IntErrorKind::PosOverflow

/-- Level 9 of `parse_expr_inner` (cypher.rs:2199-2221, 2522-2540) — and
`parse_literal` (cypher.rs:429-453), which has the same three cases — applied
to one literal token, optionally preceded by `-`; a `Negate` node over a
constant then evaluates by `checked_neg`, which cannot fail on `0..=i64::MAX`. -/
def evalLit (neg : Bool) : Except String Int → Except String Int
  | .error e => .error e
  | .ok v =>
    if neg then (if v = I64MIN then .ok I64MIN else .ok (-v))
    else if v = I64MIN then .error "Integer overflow '9223372036854775808'"
    else .ok v

/-- openCypher: the literal's value, negated if asked, if it fits `i64`. -/
def specLit (neg : Bool) (radix : Nat) (ds : List Nat) : Option Int :=
  if ds = [] then none else
  let w : Int := if neg then -(digitsVal radix ds : Int) else digitsVal radix ds
  if I64MIN ≤ w ∧ w ≤ I64MAX then some w else none

@[simp] theorem toOption_ok {ε α : Type} (v : α) : (Except.ok v : Except ε α).toOption = some v := rfl
@[simp] theorem toOption_err {ε α : Type} (e : ε) : (Except.error e : Except ε α).toOption = none := rfl

theorem canon_min_val : ∀ r ∈ [2, 8, 10, 16], digitsVal r (canon r (2 ^ 63)) = 2 ^ 63 := by decide

/-- **Integer literals are exact**: for a canonically spelled literal in any
of the four radixes, negated or not, the parser produces the literal's value
when it fits `i64` and an error otherwise. -/
theorem int_literal_correct (neg : Bool) (radix : Nat) (hr : radix ∈ [2, 8, 10, 16])
    (ds : List Nat) (hne : ds ≠ []) (hcanon : ds = canon radix (digitsVal radix ds)) :
    (evalLit neg (str2int radix ds)).toOption = specLit neg radix ds := by
  have hmin := canon_min_val radix hr
  have hv' : ds = canon radix (2 ^ 63) ↔ digitsVal radix ds = 2 ^ 63 :=
    ⟨fun h => by rw [h]; exact hmin, fun h => by rw [hcanon, h]⟩
  unfold str2int specLit
  simp only [hne, if_false, hv']
  generalize digitsVal radix ds = v
  have e1 : I64MIN = -9223372036854775808 := rfl
  have e2 : I64MAX = 9223372036854775807 := rfl
  have e3 : (2 : Nat) ^ 63 = 9223372036854775808 := rfl
  rw [e3]
  by_cases hv : v = 9223372036854775808
  · subst hv
    cases neg
    · simp only [if_true, evalLit, Bool.false_eq_true, if_false]
      rw [if_neg (by rw [e1, e2]; omega)]; rfl
    · simp only [if_true, evalLit]
      rw [if_pos (by rw [e1, e2]; omega)]; rfl
  · simp only [hv, if_false]
    by_cases hle : (v : Int) ≤ I64MAX
    · rw [if_pos hle]
      have h0 : (v : Int) ≠ I64MIN := by rw [e1]; omega
      cases neg
      · simp only [evalLit, if_neg h0, Bool.false_eq_true, if_false, toOption_ok]
        rw [if_pos (by rw [e1, e2] at *; omega)]
      · simp only [evalLit, if_neg h0, if_true, toOption_ok]
        rw [if_pos (by rw [e1, e2] at *; omega)]
    · rw [if_neg hle]
      cases neg
      · simp only [evalLit, toOption_err, Bool.false_eq_true, if_false]
        rw [if_neg (by rw [e1, e2] at *; omega)]
      · simp only [evalLit, toOption_err, if_true]
        rw [if_neg (by rw [e1, e2] at *; omega)]

/-- `-0x08000000000000000` is `-2^63`, but only the canonical spelling is in
`MIN_I64`, so the Rust reports an overflow (C FalkorDB: -9223372036854775808). -/
theorem min_noncanonical_bug :
    (evalLit true (str2int 16 (0 :: canon 16 (2 ^ 63)))).toOption = none ∧
    specLit true 16 (0 :: canon 16 (2 ^ 63)) = some I64MIN := by decide

/-- The tree shapes level 9 inspects (cypher.rs:2524-2539). -/
inductive LitTree
  | const (v : Int)
  | negate (e : LitTree)
  | post (e : LitTree)          -- `.x`, `[i]`, `{..}`, `:L` wrapped at level 10

/-- The level-9 check: `Negate(Constant(MIN))` → `MIN`; a bare
`Constant(MIN)` → overflow; anything else passes. -/
def level9 : LitTree → Except String LitTree
  | .negate (.const v) => if v = I64MIN then .ok (.const I64MIN) else .ok (.negate (.const v))
  | .const v => if v = I64MIN then .error "Integer overflow" else .ok (.const v)
  | e => .ok e

/-- A postfix operator at level 10 wraps the constant before level 9 looks,
so `9223372036854775808.x` is accepted as `(-2^63).x`: the literal's overflow
is never reported (C FalkorDB: `Integer overflow '9223372036854775808'`). -/
theorem min_postfix_bug : level9 (.post (.const I64MIN)) = .ok (.post (.const I64MIN)) := rfl

/-- `i as u32` on an `i64` hop count (cypher.rs:1790, 1797, 2991, 2998). -/
def asU32 (i : Int) : Nat := (i % 2 ^ 32).toNat

/-- `*4294967298` is `*2` and `*4294967296` is `*0`; the i64::MIN token of
`*9223372036854775808` becomes 0. Silent truncation (C FalkorDB truncates the
same way; openCypher has no such wrap). -/
theorem hop_truncation_bug :
    asU32 4294967298 = 2 ∧ asU32 4294967296 = 0 ∧ asU32 I64MIN = 0 := by decide

/-! ## 7. The expression-height bound of `e7daadac8`

`parse_expr_inner` builds trees on an explicit stack of `(level, tree, height)`
frames. The recorded `height` is an *estimate*; `check_depth` compares it
with `MAX_TREE_DEPTH`. What has to be true for the bound to mean anything is
that the estimate never falls below the real height, on any path. `Reach t e`
lists every way the Rust gives a tree `t` the estimate `e`, each with the
check it performs. -/

/-- An expression tree; only the shape matters. -/
inductive T
  | mk (kids : List T)

mutual
/-- `Parser::tree_height` (cypher.rs:314-322): a lone root is 1. -/
def height : T → Nat
  | .mk ks => 1 + hl ks
def hl : List T → Nat
  | [] => 0
  | k :: ks => max (height k) (hl ks)
end

def MAX_TREE_DEPTH : Nat := 256
def PRIMARY_LEVELS : Nat := 2

theorem hl_append (a b : List T) : hl (a ++ b) = max (hl a) (hl b) := by
  induction a with
  | nil => simp [hl]
  | cons k ks ih => simp [hl, ih, Nat.max_assoc]

theorem le_hl {k : T} {ks : List T} (h : k ∈ ks) : height k ≤ hl ks := by
  induction ks with
  | nil => simp at h
  | cons k' ks ih =>
    simp only [hl]
    rcases List.mem_cons.mp h with rfl | h
    · exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

theorem hl_le {ks : List T} {n : Nat} (h : ∀ k ∈ ks, height k ≤ n) : hl ks ≤ n := by
  induction ks with
  | nil => simp [hl]
  | cons k ks ih =>
    simp only [hl]
    exact Nat.max_le.mpr ⟨h k (by simp), ih (fun x hx => h x (by simp [hx]))⟩

/-- Every way a frame of `parse_expr_inner` (and its helpers) comes to hold
tree `t` with recorded height `e`. Each constructor names the Rust site and
carries that site's `check_depth` as the premise `e ≤ MAX_TREE_DEPTH` — or
has none when the site performs none. -/
inductive Reach : T → Nat → Prop
  /-- A complete primary (cypher.rs:2222-2238): whatever `parse_primary_expr`
  built above the sub-expressions it parsed — each `parse_expr` result
  `(cᵢ, eᵢ)` — is charged `max eᵢ + PRIMARY_LEVELS` and checked. The premise
  `hshape` is the per-constructor fact `primary_levels` checks. -/
  | primary (cs : List (T × Nat)) (t : T) (m : Nat)
      (hc : ∀ p ∈ cs, Reach p.1 p.2) (hm : ∀ p ∈ cs, p.2 ≤ m)
      (hshape : height t ≤ hl (cs.map Prod.fst) + PRIMARY_LEVELS)
      (hck : m + PRIMARY_LEVELS ≤ MAX_TREE_DEPTH) : Reach t (m + PRIMARY_LEVELS)
  /-- A run of `NOT`s (one frame for an odd count, two for an even one since #3060,
  cypher.rs:2193-2203), unary `-`, a recursing `(`/`[` primary, `NOT IN`'s pushed `Not`
  (cypher.rs:2215, 2229, 2400ff): each a childless node at 1. -/
  | opener : Reach (.mk []) 1
  /-- `tree!(Op, res)` then `height + 1` and `check_depth`: `parse_operators!`
  (macro.rs), the comparison and predicate wraps (cypher.rs:2310-2321,
  2365-2405, 2504-2506). -/
  | wrap {t : T} {e : Nat} : Reach t e → e + 1 ≤ MAX_TREE_DEPTH → Reach (.mk [t]) (e + 1)
  /-- `parse_expr_return!`'s `Some(expr)` arm: `push_child_tree(res)`,
  `*h = max(*h, height + 1)`, `check_depth(*h)`. Also an n-ary operator's next
  operand (`parse_operators!` keeps the node, then this attaches). -/
  | attach {ks : List T} {ep : Nat} {c : T} {ec : Nat} : Reach (.mk ks) ep → Reach c ec →
      max ep (ec + 1) ≤ MAX_TREE_DEPTH → Reach (.mk (ks ++ [c])) (max ep (ec + 1))
  /-- `((x))` → `(x)` (cypher.rs:2596-2602): the inner `Paren` is taken out,
  `height.saturating_sub(1)`; no check (the estimate shrinks). -/
  | collapse {ks : List T} {e : Nat} : Reach (.mk [.mk ks]) e → Reach (.mk ks) (e - 1)
  /-- The chained comparison's cloned middle operand (cypher.rs:2301-2317):
  measured exactly by `tree_height`, plus one for the new comparison. -/
  | chainMid (m : T) : height m + 1 ≤ MAX_TREE_DEPTH → Reach (.mk [m]) (height m + 1)
  /-- Postfix `[i]` / `[a..b]` (cypher.rs:2555-2562): `GetElement(s)(lhs, idx…)`,
  `height = max(height + 1, index_height + 1)`, checked. -/
  | index {t e} {is : List T} {ei : Nat} : Reach t e → (∀ i ∈ is, height i ≤ ei) →
      max (e + 1) (ei + 1) ≤ MAX_TREE_DEPTH → Reach (.mk (t :: is)) (max (e + 1) (ei + 1))
  /-- Postfix `.prop` (cypher.rs:2563-2566): `height += 1`, checked at the
  end of the step. -/
  | prop {t : T} {e : Nat} : Reach t e → e + 1 ≤ MAX_TREE_DEPTH → Reach (.mk [t]) (e + 1)
  /-- Postfix map projection `{k: v, .p, .*}` (cypher.rs:2567-2573):
  `MapProjection(base, Constant(k) → v …)`, `max(height + 1, value_height + 2)`. -/
  | mapProj {t e} {vs : List T} {ev : Nat} : Reach t e → (∀ v ∈ vs, height v ≤ ev) →
      max (e + 1) (ev + 2) ≤ MAX_TREE_DEPTH →
      Reach (.mk (t :: vs.map (fun v => .mk [v]))) (max (e + 1) (ev + 2))
  /-- `x:L1:L2` (cypher.rs:2578-2588): `hasLabels(res, List(Constant…))`,
  `max(height + 1, 3)`. -/
  | labels {t : T} {e n : Nat} : Reach t e → max (e + 1) 3 ≤ MAX_TREE_DEPTH →
      Reach (.mk [t, .mk (List.replicate n (.mk []))]) (max (e + 1) 3)

theorem hl_replicate_leaf (n : Nat) : hl (List.replicate n (T.mk [])) ≤ 1 := by
  induction n with
  | zero => simp [hl]
  | succ n ih => simp [List.replicate, hl, height]; omega

/-- **The estimate is sound**: whatever path built it, a frame's recorded
height is at least the height of its tree. -/
theorem height_sound : ∀ {t e}, Reach t e → height t ≤ e
  | _, _, .primary cs t m hc hm hshape _ => by
    have : hl (cs.map Prod.fst) ≤ m := hl_le (by
      intro k hk
      obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hk
      exact Nat.le_trans (height_sound (hc p hp)) (hm p hp))
    simp [PRIMARY_LEVELS] at hshape ⊢; omega
  | _, _, .opener => by simp [height, hl]
  | _, _, .wrap h _ => by have := height_sound h; simp [height, hl]; omega
  | _, _, .attach (ks := ks) (c := c) hp hc _ => by
    have h1 := height_sound hp; have h2 := height_sound hc
    simp only [height, hl_append, hl] at h1 ⊢; omega
  | _, _, .collapse h => by have := height_sound h; simp [height, hl] at this ⊢; omega
  | _, _, .chainMid m _ => by simp [height, hl]; omega
  | _, _, .index (is := is) h hi _ => by
    have h1 := height_sound h
    have h2 : hl is ≤ _ := hl_le hi
    simp only [height, hl]; omega
  | _, _, .prop h _ => by have := height_sound h; simp [height, hl]; omega
  | _, _, .mapProj (vs := vs) (ev := ev) h hv _ => by
    have h1 := height_sound h
    have h2 : hl (vs.map (fun v => T.mk [v])) ≤ ev + 1 := hl_le (by
      intro k hk
      obtain ⟨v, hv', rfl⟩ := List.mem_map.mp hk
      have := hv v hv'; simp [height, hl]; omega)
    simp only [height, hl]; omega
  | _, _, .labels (n := n) h _ => by
    have h1 := height_sound h
    have := hl_replicate_leaf n
    simp only [height, hl]; omega

/-- ... and every recorded height passed its check. -/
theorem guarded : ∀ {t e}, Reach t e → e ≤ MAX_TREE_DEPTH
  | _, _, .primary _ _ _ _ _ _ hck => hck
  | _, _, .opener => by decide
  | _, _, .wrap _ h => h
  | _, _, .attach _ _ h => h
  | _, _, .collapse h => Nat.le_trans (Nat.sub_le _ _) (guarded h)
  | _, _, .chainMid _ h => h
  | _, _, .index _ _ h => h
  | _, _, .prop _ h => h
  | _, _, .mapProj _ _ h => h
  | _, _, .labels _ h => h

/-- **The bound holds on every path `Reach` models**: no expression tree
`parse_expr_inner` accepts is taller than `MAX_TREE_DEPTH`. -/
theorem guarded_height_le {t : T} {e : Nat} (h : Reach t e) : height t ≤ MAX_TREE_DEPTH :=
  Nat.le_trans (height_sound h) (guarded h)

/-- `PRIMARY_LEVELS = 2` covers each shape `parse_primary_expr` builds over its
sub-expressions `cs` (read off cypher.rs:1568-1598, 1926-2091, 2824-2913,
3111-3141): `FuncInvocation(Distinct(args…), placeholder)`,
`Map(Constant(k) → v…)`, `Case(subject, List(conds…), else)`,
`ListComprehension(l, c, e)`, `Quantifier(l, c)`, `Reduce(i, l, b)`,
`PatternComprehension(c, e)`, and `count(*)`'s two leaves. -/
theorem primary_levels (cs : List T) :
    height (.mk [.mk cs, .mk []]) ≤ hl cs + PRIMARY_LEVELS ∧              -- f(DISTINCT args…)
    height (.mk (cs.map fun v => .mk [v])) ≤ hl cs + PRIMARY_LEVELS ∧     -- Map(Constant(k) → v…)
    height (.mk [.mk cs]) ≤ hl cs + PRIMARY_LEVELS ∧                      -- Case(…, List(conds…), …)
    height (.mk cs) ≤ hl cs + PRIMARY_LEVELS ∧                            -- comprehensions, calls
    height (.mk [.mk [], .mk []]) ≤ hl [] + PRIMARY_LEVELS := by          -- count(*)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [height, hl, PRIMARY_LEVELS]; omega
  · have : hl (cs.map fun v => T.mk [v]) ≤ hl cs + 1 := hl_le (by
      intro k hk
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hk
      have := le_hl hv; simp [height, hl]; omega)
    simp only [height, PRIMARY_LEVELS]; omega
  · simp [height, hl, PRIMARY_LEVELS]; omega
  · simp [height, PRIMARY_LEVELS]; omega
  · simp [height, hl, PRIMARY_LEVELS]

/-- Property chain `n.a.a…a` (`k` steps). -/
def chain : Nat → T
  | 0 => .mk []
  | k + 1 => .mk [chain k]

theorem chain_height (k : Nat) : height (chain k) = k + 1 := by
  induction k with
  | zero => simp [chain, height, hl]
  | succ k ih => simp [chain, height, hl, ih]; omega

/-- Inside `parse_expr_inner` the chain is bounded: `Reach` cannot hold
`chain 256`… -/
theorem chain_rejected : ¬ ∃ e, Reach (chain 256) e := by
  rintro ⟨e, h⟩
  have := guarded_height_le h
  rw [chain_height] at this
  simp [MAX_TREE_DEPTH] at this

/-- …but `parse_set_items` / `parse_remove_items` (cypher.rs:3242-3245,
3322-3325) build the SET/REMOVE target with `parse_property_lookup` in a bare
`while` loop, outside `Reach`: no estimate, no check. Modelled literally, it
accepts every length, so the tree the binder then recurses over is unbounded
(repro: `MATCH (n) SET n.a.a…(5000) = 1` overflows the binder's stack;
C FalkorDB rejects it with an error). -/
def setTarget : Nat → Except String T
  | 0 => .ok (.mk [])
  | k + 1 => (setTarget k).map fun t => .mk [t]

theorem set_target_unbounded (k : Nat) :
    setTarget k = .ok (chain k) ∧ height (chain k) = k + 1 := by
  refine ⟨?_, chain_height k⟩
  induction k with
  | zero => rfl
  | succ k ih => simp [setTarget, ih, chain, Except.map]

/-! ## 8. Call-stack recursion: `MAX_NESTING` and `FOREACH` -/

/-- The parser's call-recursive descents. `expr` is a `parse_expr` (or
`parse_literal_list/map`, `CALL {}`) — all routed through `Parser::nested`;
`foreach` is `parse_foreach_clause`, whose body re-enters itself directly
(cypher.rs:3402-3405). -/
inductive Call
  | leaf
  | expr (kids : List Call)
  | foreach (body : Call)

def MAX_NESTING : Nat := 100

mutual
/-- `Parser::nested` (cypher.rs:328-338): refuse at `depth ≥ MAX_NESTING`,
else `depth + 1` for the callee. `parse_foreach_clause` passes `depth`
through unchanged. -/
def runCall : Nat → Call → Bool
  | _, .leaf => true
  | d, .expr ks => if d ≥ MAX_NESTING then false else runAll (d + 1) ks
  | d, .foreach b => runCall d b
def runAll : Nat → List Call → Bool
  | _, [] => true
  | d, k :: ks => runCall d k && runAll d ks
end

mutual
/-- Nesting of `nested` levels. -/
def exprDepth : Call → Nat
  | .leaf => 0
  | .expr ks => 1 + exprDepthL ks
  | .foreach b => exprDepth b
def exprDepthL : List Call → Nat
  | [] => 0
  | k :: ks => max (exprDepth k) (exprDepthL ks)
end

mutual
/-- Rust call-stack depth: every descent is a frame. -/
def stackDepth : Call → Nat
  | .leaf => 0
  | .expr ks => 1 + stackDepthL ks
  | .foreach b => 1 + stackDepth b
def stackDepthL : List Call → Nat
  | [] => 0
  | k :: ks => max (stackDepth k) (stackDepthL ks)
end

mutual
theorem nested_bound : ∀ (d : Nat) (c : Call), runCall d c = true → d ≤ MAX_NESTING →
    d + exprDepth c ≤ MAX_NESTING
  | d, .leaf, _, hd => by simp [exprDepth]; exact hd
  | d, .expr ks, h, _ => by
    simp only [runCall] at h
    split at h
    · simp at h
    · rename_i hlt
      have := nested_boundL (d + 1) ks h (by omega)
      simp only [exprDepth]; omega
  | d, .foreach b, h, hd => by
    simp only [runCall] at h
    simpa [exprDepth] using nested_bound d b h hd
theorem nested_boundL : ∀ (d : Nat) (ks : List Call), runAll d ks = true → d ≤ MAX_NESTING →
    d + exprDepthL ks ≤ MAX_NESTING
  | d, [], _, hd => by simp [exprDepthL]; exact hd
  | d, k :: ks, h, hd => by
    simp only [runAll, Bool.and_eq_true] at h
    have h1 := nested_bound d k h.1 hd
    have h2 := nested_boundL d ks h.2 hd
    simp only [exprDepthL]; omega
end

/-- **`MAX_NESTING` bounds every `nested` descent**: a parse that gets
through has at most 100 levels of `parse_expr` / literal / `CALL {}`
nesting. -/
theorem nested_depth_bound (c : Call) (h : runCall 0 c = true) : exprDepth c ≤ MAX_NESTING := by
  have := nested_bound 0 c h (by decide); omega

def foreachChain : Nat → Call
  | 0 => .expr [.leaf]                  -- the innermost `CREATE ()` clause's expression
  | n + 1 => .foreach (foreachChain n)

/-- **…but not `FOREACH`**: `parse_foreach_clause` never passes through
`nested`, so `FOREACH (x IN [1] | FOREACH (…| CREATE ()))` is accepted at
every depth, with a call stack as deep as the text. Repro: 100 000 levels
abort the process with a stack overflow (C FalkorDB crashes as well). -/
theorem foreach_unbounded (n : Nat) :
    runCall 0 (foreachChain n) = true ∧ stackDepth (foreachChain n) = n + 1 := by
  induction n with
  | zero => simp [foreachChain, runCall, runAll, stackDepth, stackDepthL, MAX_NESTING]
  | succ n ih =>
    obtain ⟨h1, h2⟩ := ih
    refine ⟨by simpa [foreachChain, runCall] using h1, ?_⟩
    simp [foreachChain, stackDepth, h2]; omega

end FalkorLexer
