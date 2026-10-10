/-
# Clause-level counterexamples: inputs the Rust accepts outside Cypher.g4
# (each CONFIRMED on a live server, Rust module vs C module)
-/
import FalkorParserGrammar.C.Index
import FalkorParserGrammar.C.Oracle0
import FalkorParserGrammar.C.PatternBugs

namespace FalkorParserGrammar.C

def q (ts : List CT) : R IR := run (parseP o0 40 40) ts

/-- **`MATCH MATCH (n) RETURN n` is refused** (#3060, `fe619ac5f`: the second
`optional_match_token!(Match)` is gone, cypher.rs:918-925), as in C. Historical
(2c874022a, `match_match_accepted`): it was accepted. -/
theorem match_match_refused :
    (q [.kw .match_, .kw .match_, .lparen, .ident 1, .rparen, .kw .return_, .ident 1]).isOk = false := by
  decide

theorem match_ok :
    (q [.kw .match_, .lparen, .ident 1, .rparen, .kw .return_, .ident 1]).isOk = true := by
  decide

/-- Every reading clause the Rust accepts after `MATCH` parses its pattern right after
that one keyword (no extension flag is left for it). -/
theorem reading_match_at_adv (o : Oracle) (s : PS) r s' (h1 : s.cur = .kw .match_) (g : GReading o s r s') :
    ∃ p s1 f, GPat o Ext.rust GExt.rust .match_ (QG.empty, []) s.adv p s1 ∧ GWhere o s1 f s' := by
  cases g with
  | optMatch hc => simp [h1] at hc
  | match_ hc hp => obtain ⟨p, s1, f, gp, gw, _⟩ := hp; exact ⟨p, s1, f, gp, gw⟩
  | unwind hc => simp [h1] at hc
  | load hc => simp [h1] at hc

/-- The pattern after the first MATCH would have to start at the second
MATCH keyword: no `oC_PatternPart` starts with a keyword followed by `(`. -/
theorem g4_pattern_part_not_kw_lparen (o : Oracle) (cl : CK) (gs : QG Nat × List Nat) (s : PS) e s'
    (h1 : s.cur = .kw .match_) (h2 : s.adv.cur = .lparen) (g : GPart o Ext.rust GExt.rust cl gs s e s') :
    False := by
  cases g with
  | anon gn _ => cases gn with | mk hc => simp [h1] at hc
  | named _ he _ _ => simp [h2] at he
  | asp _ _ he => simp [h2] at he

/-- **`LOAD CSV WITH FROM 'f' AS r` is refused** (#3060: `WITH` must be followed by
`HEADERS`, cypher.rs:939-942), as in C. Historical (2c874022a, `load_csv_with_from_accepted`). -/
theorem load_csv_with_from_refused :
    (q [.kw .load, .kw .csv, .kw .with_, .kw .from, .str 1, .kw .as_, .ident 2,
        .kw .return_, .ident 2]).isOk = false := by decide

theorem load_csv_with_headers_ok :
    (q [.kw .load, .kw .csv, .kw .with_, .kw .headers, .kw .from, .str 1, .kw .as_, .ident 2,
        .kw .return_, .ident 2]).isOk = true ∧
    (q [.kw .load, .kw .csv, .kw .from, .str 1, .kw .as_, .ident 2,
        .kw .return_, .ident 2]).isOk = true := by decide

/-- ... and no accepted LOAD CSV has `WITH` directly before `FROM`. -/
theorem no_load_with_from (o : Oracle) (s : PS) r s' (hw : s.adv.cur = .kw .with_)
    (hf : s.adv.adv.cur = .kw .from) (g : GLoadCsv o s r s') : False := by
  cases g with
  | mk _ hd _ =>
    rcases hd with ⟨_, _, hh, _⟩ | ⟨_, hnw, _⟩
    · rw [hf] at hh; cases hh
    · exact hnw hw

/-- **`SET [n).x = 5` / `REMOVE [n).x` are refused** at the clause level too (#3060,
`reject_unparenthesized_target`), while `SET (n).x = 5` is accepted. -/
theorem set_bracket_target_refused :
    (q [.kw .create, .lparen, .ident 1, .rparen, .kw .set, .lbrack, .ident 1, .rparen, .dot, .ident 2,
        .eq, .int 5]).isOk = false ∧
    (q [.kw .create, .lparen, .ident 1, .rparen, .kw .remove, .lbrack, .ident 1, .rparen, .dot,
        .ident 2]).isOk = false ∧
    (q [.kw .create, .lparen, .ident 1, .rparen, .kw .set, .lparen, .ident 1, .rparen, .dot, .ident 2,
        .eq, .int 5]).isOk = true := by decide

/-- **A trailing comma in a procedure call is refused** (#3060, `parse_expression_list`). -/
theorem call_trailing_comma_refused :
    (q [.kw .call, .ident 77, .lparen, .int 1, .comma, .rparen, .kw .yield_, .ident 5,
        .kw .return_, .ident 5]).isOk = false := by decide

/-- `SET (n.x) = 5` is accepted as a replace-assignment to a property
(cypher.rs:3261-3275: the `=` arm takes any primary as target). Live:
`CREATE (n {a:1}) SET (n.x) = 5 RETURN n` sets x; C: syntax error. -/
theorem set_paren_property_accepted :
    (q [.kw .create, .lparen, .ident 1, .rparen, .kw .set, .lparen, .ident 1, .dot, .ident 2,
        .rparen, .eq, .int 5]).isOk = true := by decide

/-- `DELETE a DETACH DELETE b` is folded into one clause with `detach`
(cypher.rs:1172-1187); the folded IR is a single DELETE. -/
def delShape : Option IR → Option (Nat × Nat × Bool)
  | some (.query cs _) => match cs with
    | [.delete es d] => some (cs.length, es.length, d)
    | _ => none
  | _ => none

theorem delete_detach_folded :
    delShape (q [.kw .delete, .ident 1, .kw .detach, .kw .delete, .ident 2]).get? =
      some (1, 2, true) := by decide

/-- Mixing UNION and UNION ALL is rejected (as in C). -/
theorem union_mix_rejected :
    (q [.kw .return_, .int 1, .kw .union, .kw .return_, .int 1, .kw .union, .kw .all,
        .kw .return_, .int 1]).isOk = false := by decide

/-- `CALL db.labels` without parentheses inside a query is accepted. -/
theorem call_implicit_accepted :
    (q [.kw .call, .ident 77, .kw .yield_, .ident 5, .kw .return_, .ident 5]).isOk = true := by decide

/-- A reading clause after an update needs WITH (cypher.rs:829-876), as in C. -/
theorem match_after_create_rejected :
    (q [.kw .create, .lparen, .rparen, .kw .match_, .lparen, .rparen, .kw .return_, .int 1]).isOk =
      false := by decide

end FalkorParserGrammar.C
