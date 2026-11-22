/-
# Pattern-level bugs and the theorems that pin their fixes

No bug here is still open. `fixed3303_anon_name_distinct` pins the fix of W4-parse-1 (a written
`_anon_0` merged with an unnamed node, fixed by #3303's `Alias`). The #3289 and #3300 theorems pin
the fixed, correct behaviour (a repeated relationship
alias and `[:A:B]` are now refused, as in C); `create_named_path_distinct_rels_ok` and
`create_anon_path_dup_rel_rejected` are the unchanged controls.
-/
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- The identifier `_anon_0` as a name id (any id: no identifier is an `Alias::Anon`). -/
def nmAnon0 : Nat := 2000000

/-- **`MATCH (_anon_0:A), (:B)` has two nodes** (W4-parse-1, fixed by #3303). Before, the
unnamed node was named `_anon_0` (cypher.rs:3037), the same as the user's node, so
`add_pattern_node` merged them into one node with labels A and B: on `CREATE (:A), (:B)` Rust
returned 0 rows, C 1. Now the unnamed node is `Alias::Anon(0)`, which no written name equals. -/
theorem fixed3303_anon_name_distinct :
    ((run (patternP o0 20 .match_)
      [.lparen, .ident nmAnon0, .colon, .ident 1, .rparen, .comma,
       .lparen, .colon, .ident 2, .rparen]).get?.map (fun g => g.nodes.map (·.labels))) =
      some [[1], [2]] := by decide

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
