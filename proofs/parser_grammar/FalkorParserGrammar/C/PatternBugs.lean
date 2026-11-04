/-
# Pattern-level bugs and the theorems that pin their fixes

Only `anon_name_collides` documents a bug that is still open (confirmed on a live server, Rust
vs C). The #3289 and #3300 theorems pin the fixed, correct behaviour (a repeated relationship
alias and `[:A:B]` are now refused, as in C); `create_named_path_distinct_rels_ok` and
`create_anon_path_dup_rel_rejected` are the unchanged controls.
-/
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- `MATCH (_anon_0:A), (:B)`: the anonymous node is named `_anon_0`
(cypher.rs:3037), the same as the user's node, so `add_pattern_node` merges
them into one node with labels A and B. Live: Rust returns 0 rows, C 1
(on `CREATE (:A), (:B)`). -/
theorem anon_name_collides :
    ((run (patternP o0 20 .match_)
      [.lparen, .ident (anonName 0), .colon, .ident 1, .rparen, .comma,
       .lparen, .colon, .ident 2, .rparen]).get?.map (fun g => g.nodes.map (·.labels))) =
      some [[1, 2]] := by decide

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
