/-
# Clause-level fragments: the shortestPath filter skipper, SET/REMOVE
# targets, and the pattern-predicate comma
-/
import FalkorParserGrammar.Model

namespace FalkorParserGrammar

/-! ## `parse_shortest_path_expr`'s filter skipper (cypher.rs:1819-1846)

```rust
let mut depth = 0i32;
loop {
    match self.lexer.current()? {
        Token::LBracket => { depth += 1; self.lexer.next(); }            // `{`
        Token::RBracket => { depth -= 1; self.lexer.next(); if depth <= 0 { break; } }
        Token::RBrace => break,                                          // `]`
        _ => self.lexer.next(),
    }
}
```
`none` is "did not stop within the budget". -/
def skipFilter : Nat → Int → List Tok → Option (List Tok)
  | 0, _, _ => none
  | n + 1, d, ts =>
    match cur ts with
    | .lbrace => skipFilter n (d + 1) (adv ts)
    | .rbrace => if d - 1 ≤ 0 then some (adv ts) else skipFilter n (d - 1) (adv ts)
    | .rbrack => some ts
    | _ => skipFilter n d (adv ts)

/-- At end of input the skipper never stops, whatever the budget:
`RETURN shortestPath((a)-[{` spins a worker thread forever (confirmed:
`bug_shortest_path_filter_skip_loops_forever_at_eof`). -/
theorem skipFilter_eof_diverges : ∀ (n : Nat) (d : Int), skipFilter n d [] = none
  | 0, _ => rfl
  | n + 1, d => by simp only [skipFilter, cur, adv]; exact skipFilter_eof_diverges n d

/-- ... and so does every input that runs out before a `}`/`]` closes it. -/
theorem skipFilter_prefix_diverges (n : Nat) (d : Int) :
    skipFilter n d [.lbrace, .ident 1, .colon, .int 1] = none := by
  match n with
  | 0 | 1 | 2 | 3 | 4 => rfl
  | k + 5 => simp only [skipFilter, cur, adv]; exact skipFilter_eof_diverges k _

/-! ## `parse_set_items` / `parse_remove_items` targets (cypher.rs:3235-3241, 3317-3322)

```rust
let (mut expr, recurse) = self.parse_primary_expr(false)?;
if recurse {
    self.reject_unparenthesized_target(&expr)?;     // #3060
    expr = self.parse_expr(false)?; match_token!(self.lexer, RParen);
}
```
`recurse` is also `true` for a list literal `[` (2807-2810); since #3060
(`fe619ac5f`) `reject_unparenthesized_target` (3289-3298) refuses any root but
`Paren` before the `)` is demanded. -/

/-- `reject_unparenthesized_target` (3289): `Ok` iff the root is `ExprIR::Paren`. -/
def rejectUnparen (e : RT) (ts : List Tok) : Res Unit :=
  if e.root = .paren then .ok () ts else .err ts

def setTarget (pe : List Tok → Out) (fuel : Nat) (ts : List Tok) : Res RT :=
  match primary pe fuel ts with
  | .ok (e, true) r =>
    match rejectUnparen e r with
    | .ok () r =>
      match pe r with
      | .ok e (.rparen :: r') => .ok e r'
      | .ok _ r' => .err r'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel
  | .ok (e, false) r => .ok e r
  | .err e => .err e
  | .panic => .panic
  | .fuel => .fuel

/-- **`SET [n).x = 5` is refused** (#3060; C: syntax error), while `SET (n).x = 5`
still reads the target `n`. Historical (2c874022a, `set_bracket_paren_accepted`): the
list node was dropped and the target read as `n` (live: `SET [n).x = 5` set `n.x`). -/
theorem set_bracket_paren_refused :
    (∃ e, setTarget (parseExpr 100) 100 [.lbrack, .ident 3, .rparen, .dot, .ident 4] = .err e) ∧
    setTarget (parseExpr 100) 100 [.lparen, .ident 3, .rparen, .dot, .ident 4] =
      .ok (lf (.var 3)) [.dot, .ident 4] := by
  refine ⟨⟨_, rfl⟩, rfl⟩

/-- Only a `(` opens a nested target: whenever `recurse` holds for a non-`Paren`
primary, the target is refused before anything after it is read. -/
theorem setTarget_only_paren (pe : List Tok → Out) (fuel : Nat) (ts r : List Tok) (e : RT)
    (h : primary pe fuel ts = .ok (e, true) r) (hp : e.root ≠ .paren) :
    setTarget pe fuel ts = .err r := by
  simp [setTarget, h, rejectUnparen, hp]

end FalkorParserGrammar
