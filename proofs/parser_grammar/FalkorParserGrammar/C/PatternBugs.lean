/-
# Pattern-level bugs and the theorems that pin their fixes

No pattern-level bug is open any more. `anon_name_distinct` pins the fix of W4-parse-1 (#3528:
an unnamed entity's alias is `Alias.anon`, which no written name equals). The #3289 and #3300
theorems pin the fixed, correct behaviour (a repeated relationship alias and `[:A:B]` are now
refused, as in C); `create_named_path_distinct_rels_ok` and
`create_anon_path_dup_rel_rejected` are the unchanged controls.
-/
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- **`MATCH (_anon_0:A), (:B)` has two nodes** (#3528, W4-parse-1 fixed): the user's node
is `Alias.named _anon_0` and the unnamed one `Alias.anon 0` (`next_unnamed`, cypher.rs:3065), which
differ, so `add_pattern_node` keeps them apart: labels `[[A], [B]]`, aliases as written. As in C
(1 row on `CREATE (:A), (:B)`). Historical (before #3528, `anon_name_collides`): the unnamed node
was *named* `_anon_0` (`format!("_anon_{}", ..)`), the same as the user's, so the two were merged
into one node with labels A and B (Rust 0 rows, C 1). -/
theorem anon_name_distinct :
    ((run (patternP o0 20 .match_)
      [.lparen, .ident (anonName 0), .colon, .ident 1, .rparen, .comma,
       .lparen, .colon, .ident 2, .rparen]).get?.map
        (fun g => g.nodes.map (fun n => (n.alias, n.labels)))) =
      some [(.named (anonName 0), [1]), (.anon 0, [2])] := by decide

/-- In general, a pattern alias the query wrote never equals an unnamed one: every `GOptVar`
alias read from an identifier is `named`, every fresh one `anon` (#3528). -/
theorem optVar_named_ne_anon {s s' t t' : PS} {v : Nat} {a b : Alias}
    (ha : GOptVar s a s') (hb : GOptVar t b t') (hv : identOf s.cur = some v)
    (hk : identOf t.cur = none) : a ≠ b := by
  cases ha with
  | named _ => cases hb with
    | named h => simp [hk] at h
    | anon _ => exact Alias.noConfusion
  | anon h => simp [hv] at h

/-- **`CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d)` is rejected** (#3289, W4-parse-2): every
relationship goes through `add_pattern_relationship` (cypher.rs:1378), which refuses an alias the
pattern already holds in any clause ("The bound variable 'r' can't be redeclared in a CREATE
clause"), as C does. Before it only MATCH rejected a repeat in a named path: CREATE silently
dropped the second relationship (Rust created 4 nodes and 1 relationship). -/
theorem create_named_path_dup_rel_rejected :
    (run (patternP o0 30 .create)
      [.ident 10, .eq, .lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 2, .rparen, .comma,
       .ident 11, .eq, .lparen, .ident 3, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 4, .rparen]).isOk = false := by decide

/-- The same without path names was already rejected, as in C. -/
theorem create_anon_path_dup_rel_rejected :
    (run (patternP o0 30 .create)
      [.lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 2, .rparen, .comma,
       .lparen, .ident 3, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 4, .rparen]).isOk = false := by decide

/-- A named path with two distinct relationships still parses (the check is on a repeat only). -/
theorem create_named_path_distinct_rels_ok :
    (run (patternP o0 30 .create)
      [.ident 10, .eq, .lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 2, .rparen, .dash, .lbrack, .ident 6, .colon,
         .ident 9, .rbrack, .dash, .gt, .lparen, .ident 3, .rparen]).isOk = true := by decide

/-- **`MERGE (a)-[r:R]->(b)-[r:R]->(c)` is rejected** (#3289, W4-parse-3): "The bound variable 'r'
can't be redeclared in a MERGE clause", as in C. Before, MERGE silently dropped the second
relationship (Rust created 3 nodes and 1 relationship). -/
theorem merge_dup_rel_rejected :
    (run (patternP o0 30 .merge)
      [.lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9, .rbrack, .dash, .gt,
       .lparen, .ident 2, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9, .rbrack, .dash, .gt,
       .lparen, .ident 3, .rparen]).isOk = false := by
  decide

/-- **`[:A:B]` is refused** (#3300, W4-parse-4; C: syntax error): a bare `:` after a type name
no longer continues the list (`typesP_colon_refused`), so the detail's `]` is missing. `[:A|B]`
and the legacy `[:A|:B]` still parse. -/
theorem type_colon_refused :
    (run (patternP o0 20 .match_)
      [.lparen, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 1, .colon, .ident 2,
       .rbrack, .dash, .gt, .lparen, .rparen]).isOk = false ∧
    (run (patternP o0 20 .match_)
      [.lparen, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 1, .pipe, .ident 2,
       .rbrack, .dash, .gt, .lparen, .rparen]).isOk = true ∧
    (run (patternP o0 20 .match_)
      [.lparen, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 1, .pipe, .colon, .ident 2,
       .rbrack, .dash, .gt, .lparen, .rparen]).isOk = true := by decide

end FalkorParserGrammar.C
