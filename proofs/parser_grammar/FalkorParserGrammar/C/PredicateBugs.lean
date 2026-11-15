/-
# #2502: a WHERE pattern predicate swallows the next MATCH clause

Gap closed 2026-10-20. The models below were already faithful (`parenArm`,
`patternP`, `sepP`), but no property said where a pattern predicate must end,
so the bug went unnoticed. Citations are at main `fb3d516d0`
(`graph/src/parser/cypher.rs`):

* `parse_where` (1214-1231) calls `parse_expr(true)`; its primary
  `parse_primary_expr` takes the `(` arm (2114-2128), which tries
  `self.parse_pattern(&Keyword::Match)` (2119) — modelled by `parenArm`
  (`C.Primary`).
* `parse_pattern`'s part separator (1562-1579) continues on `,` **and on the
  clause's own keyword** (`if token == *clause { self.lexer.next(); continue; }`),
  modelled by `sepP` (`C.Pattern`). Called with `Keyword::Match` for a
  *predicate*, it therefore eats a following `MATCH` keyword and folds the next
  clause's pattern into the predicate.

Live (release, ports 6670/6671, on `CREATE (a:A)-[:Q]->(:B), (a)-[:R]->(:M {v:1}),
(a)-[:R]->(:M {v:2})`): `MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m.v`
gives Rust `'m' not defined`, C rows 1, 2; with `RETURN count(*)` and `(a:A)`,
Rust 1 (the R traversal sits inside the Semi Apply), C 2.
-/
import FalkorParserGrammar.C.Index
import FalkorParserGrammar.C.Primary
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- The keywords that start a clause (`parse_single_query`'s loops,
cypher.rs:796-927, and `parse_query`'s `UNION`, 771-789). -/
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

/-- **#2502 (main violates `PredEndsBeforeClause`).** `(a)-[:Q]->()` alone is a
pattern predicate, but followed by `MATCH (a)-[:R]->(m) RETURN m` the arm eats
`MATCH` and the next pattern and stops only at `RETURN`. -/
theorem bug2502_pred_swallows_match : ¬ PredEndsBeforeClause (parenArm o0 30 true) := by
  intro h
  have h1 : parenArm o0 30 true (predQ, 0) = .ok (leaf .pattern, false) ([], 2) := rfl
  obtain ⟨t', _, h2⟩ := h predQ restR .match_ 0 2 (leaf .pattern) false rfl rfl h1
  have h3 := congrArg R.st? h2
  have h4 : (parenArm o0 30 true (predQ ++ .kw .match_ :: restR, 0)).st? =
      some ([.kw .return_, .ident 2], 3) := by decide
  rw [h4] at h3
  simp [R.st?] at h3

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

/-- **#2502, the whole query** (`Parser::parse`, `parseP`): `MATCH (a) WHERE
(a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` parses to TWO clauses — the
second MATCH has vanished into the WHERE predicate, so `m` is never bound
(live: Rust `'m' not defined`, C returns the two `m` rows). With a
non-pattern WHERE (`WHERE a`) the same query has its three clauses. -/
theorem bug2502_where_match_merged :
    shape (run (parseP oPred 40 40)
      (.kw .match_ :: .lparen :: .ident 1 :: .rparen :: .kw .where_ :: predQ ++ .kw .match_ :: restR))
      = some (2, true) ∧
    shape (run (parseP oPred 40 40)
      (.kw .match_ :: .lparen :: .ident 1 :: .rparen :: .kw .where_ :: .ident 1 :: .kw .match_ :: restR))
      = some (3, true) := by
  decide

end FalkorParserGrammar.C
