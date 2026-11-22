/-
# Clause-level fragments: SET/REMOVE targets
-/
import FalkorParserGrammar.Model

namespace FalkorParserGrammar

/-! ## `parse_set_items` / `parse_remove_items` targets (cypher.rs)

```rust
let (mut expr, recurse) = self.parse_primary_expr(false)?;
if recurse {
    self.reject_unparenthesized_target(&expr)?;
    expr = self.parse_expr(false)?; match_token!(self.lexer, RParen);
    self.expect_property_after_paren()?;            // the `)` must be followed by `.`
}
```
`recurse` is also `true` for a list literal `[` (2880-2884); `reject_unparenthesized_target` (3395-3402) refuses any root but
`Paren` before the `)` is demanded. -/

/-- `reject_unparenthesized_target` (3395): `Ok` iff the root is `ExprIR::Paren`. -/
def rejectUnparen (e : RT) (ts : List Tok) : Res Unit :=
  if e.root = .paren then .ok () ts else .err ts

def setTarget (pe : List Tok → Out) (fuel : Nat) (ts : List Tok) : Res RT :=
  match primary pe fuel ts with
  | .ok (e, true) r =>
    match rejectUnparen e r with
    | .ok () r =>
      match pe r with
      | .ok e (.rparen :: r') => if cur r' = .dot then .ok e r' else .err r'
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

/-- **A parenthesised target must be followed by `.`** (#3300, `expect_property_after_paren`):
`SET (n).x = 5` reads the target, `SET (n) = {..}` / `SET (n):L` / `SET (n.x) = 5` are refused. -/
theorem set_paren_needs_dot :
    setTarget (parseExpr 100) 100 [.lparen, .ident 3, .rparen, .dot, .ident 4] =
      .ok (lf (.var 3)) [.dot, .ident 4] ∧
    (∃ e, setTarget (parseExpr 100) 100 [.lparen, .ident 3, .rparen, .eq, .int 5] = .err e) ∧
    (∃ e, setTarget (parseExpr 100) 100 [.lparen, .ident 3, .rparen, .colon, .ident 4] = .err e) ∧
    (∃ e, setTarget (parseExpr 100) 100
      [.lparen, .ident 3, .dot, .ident 4, .rparen, .eq, .int 5] = .err e) := by
  refine ⟨rfl, ⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩⟩

/-- Only a `(` opens a nested target: whenever `recurse` holds for a non-`Paren`
primary, the target is refused before anything after it is read. -/
theorem setTarget_only_paren (pe : List Tok → Out) (fuel : Nat) (ts r : List Tok) (e : RT)
    (h : primary pe fuel ts = .ok (e, true) r) (hp : e.root ≠ .paren) :
    setTarget pe fuel ts = .err r := by
  simp [setTarget, h, rejectUnparen, hp]

end FalkorParserGrammar
