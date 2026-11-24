/-
# A WHERE pattern predicate must end before the next MATCH clause (#2502)

Gap closed 2026-10-20. The models below were faithful (`parenArm`,
`patternP`, `sepP`), but no property said where a pattern predicate must end. Citations are at
main `fb3d516d0` (`graph/src/parser/cypher.rs`):

* `parse_where` (1214-1231) calls `parse_expr(true)`; its primary
  `parse_primary_expr` takes the `(` arm (2114-2128), which tries
  `self.parse_pattern(&Keyword::Match)` (2119) — modelled by `parenArm`
  (`C.Primary`).
* `parse_pattern`'s part separator continues only on `,` (the earlier separator also continued on the
  clause's own keyword, `sepP` / `C.Pattern`, which made a predicate eat a following `MATCH`
  keyword and fold the next clause's pattern into the predicate).
-/
import FalkorParserGrammar.C.Index
import FalkorParserGrammar.C.Primary
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- The keywords that start a clause (`parse_single_query`'s loops,
cypher.rs, and `parse_query`'s `UNION`, 771-789). -/
def isClauseKw : CK → Bool
  | .match_ | .optional | .unwind | .call | .load | .create | .merge | .delete | .detach
  | .set | .remove | .foreach | .with_ | .return_ | .union => true
  | _ => false

/-- **A WHERE pattern predicate ends before a clause keyword.** If the
pattern-predicate arm `arm` (the `(` arm of `parse_primary_expr(true)`) reads a
whole token list `p` as a pattern predicate, then on `p` followed by a
clause keyword `k` and anything `r`, it still reads a pattern predicate, stops
right before `k`, and leaves the anonymous counter where it was. A fix (e.g.
`parse_pattern` not folding on the keyword when called for a predicate) proves
this for `parenArm`. -/
def PredEndsBeforeClause (arm : P (ET × Bool)) : Prop :=
  ∀ (p r : List CT) (k : CK) (n n' : Nat) (t : ET) (b : Bool),
    isClauseKw k = true → t.root = .pattern →
    arm (p, n) = .ok (t, b) ([], n') →
    ∃ t', t'.root = .pattern ∧ arm (p ++ .kw k :: r, n) = .ok (t', b) (.kw k :: r, n')

/-- The state left after a successful parse. -/
def R.st? {α} : R α → Option PS | .ok _ s => some s | _ => none

/-- `(a)-[:Q]->()` with `a = 1`, `Q = 5`. -/
def predQ : List CT :=
  [.lparen, .ident 1, .rparen, .dash, .lbrack, .colon, .ident 5, .rbrack, .dash, .gt,
   .lparen, .rparen]

/-- `(a)-[:R]->(m) RETURN m` with `R = 6`, `m = 2`. -/
def restR : List CT :=
  [.lparen, .ident 1, .rparen, .dash, .lbrack, .colon, .ident 6, .rbrack, .dash, .gt,
   .lparen, .ident 2, .rparen, .kw .return_, .ident 2]

/-- **#2502 fixed by #2981** (`sepP` no longer continues on the clause keyword,
cypher.rs). `(a)-[:Q]->()` followed by `MATCH (a)-[:R]->(m) RETURN m`
is the same pattern predicate and stops right before `MATCH`: the instance of
`PredEndsBeforeClause` that main refuted before (`bug2502_pred_swallows_match`,
pre-#2981: the arm ate `MATCH` and stopped only at `RETURN`, state
`([RETURN, m], 3)`). -/
theorem fixed2502_pred_stops_at_match :
    ∃ t', t'.root = .pattern ∧
      parenArm o0 30 true (predQ ++ .kw .match_ :: restR, 0) = .ok (t', false) (.kw .match_ :: restR, 2) :=
  ⟨leaf .pattern, rfl, rfl⟩

/-- An oracle whose `parse_expr(allow)` is `parse_primary_expr(allow)` with
the real `(` arm (`primaryC`): enough for a WHERE that is one pattern
predicate followed by a clause keyword (no operator follows, so
`parse_expr` returns the primary). -/
def oPred : Oracle :=
  { o0 with pe := fun allow => do let r ← primaryC o0 (fun _ => none) 30 allow false; pure r.1 }

/-- The number of top-level clauses of a parsed query and whether the first
one is a MATCH with a filter. -/
def shape : R IR → Option (Nat × Bool)
  | .ok (.query cs _) _ => some (cs.length, match cs with
      | .match_ _ (some _) _ :: _ => true
      | _ => false)
  | _ => none

/-- **#2502, the whole query, fixed by #2981** (`Parser::parse`, `parseP`):
`MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` parses to its three
clauses, as with a non-pattern WHERE (`WHERE a`). Pre-#2981
(`bug2502_where_match_merged`) it parsed to TWO: the second MATCH vanished into
the WHERE predicate (live: Rust `'m' not defined`, C the two `m` rows). -/
theorem fixed2502_where_match_split :
    shape (run (parseP oPred 40 40)
      (.kw .match_ :: .lparen :: .ident 1 :: .rparen :: .kw .where_ :: predQ ++ .kw .match_ :: restR))
      = some (3, true) ∧
    shape (run (parseP oPred 40 40)
      (.kw .match_ :: .lparen :: .ident 1 :: .rparen :: .kw .where_ :: .ident 1 :: .kw .match_ :: restR))
      = some (3, true) := by
  decide

end FalkorParserGrammar.C
