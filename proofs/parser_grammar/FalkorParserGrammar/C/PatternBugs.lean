/-
# Pattern-level counterexamples (all CONFIRMED on a live server, Rust vs C)
-/
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-- `MATCH (_anon_0:A), (:B)`: the anonymous node is named `_anon_0`
(cypher.rs:2935), the same as the user's node, so `add_pattern_node` merges
them into one node with labels A and B. Live: Rust returns 0 rows, C 1
(on `CREATE (:A), (:B)`). -/
theorem anon_name_collides :
    ((run (patternP o0 20 .match_)
      [.lparen, .ident (anonName 0), .colon, .ident 1, .rparen, .comma,
       .lparen, .colon, .ident 2, .rparen]).get?.map (fun g => g.nodes.map (·.labels))) =
      some [[1, 2]] := by decide

/-- `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d)`: in a named path the
repeated relationship variable is only rejected for MATCH (cypher.rs:1514-1522),
so CREATE silently drops the second relationship. Live: Rust creates 4
nodes and 1 relationship; C: "Variable `r` already declared". -/
theorem create_named_path_drops_dup_rel :
    ((run (patternP o0 30 .create)
      [.ident 10, .eq, .lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 2, .rparen, .comma,
       .ident 11, .eq, .lparen, .ident 3, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 4, .rparen]).get?.map
        (fun g => (g.nodes.length, g.rels.length))) = some (4, 1) := by decide

/-- The same without path names is rejected, as in C. -/
theorem create_anon_path_dup_rel_rejected :
    (run (patternP o0 30 .create)
      [.lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 2, .rparen, .comma,
       .lparen, .ident 3, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9,
         .rbrack, .dash, .gt, .lparen, .ident 4, .rparen]).isOk = false := by decide

/-- `MERGE (a)-[r:R]->(b)-[r:R]->(c)`: in an anonymous path the repeated
relationship variable is an error for MATCH and CREATE only
(cypher.rs:1535-1548), so MERGE silently drops the second relationship.
Live: Rust creates 3 nodes and 1 relationship; C: "The bound variable 'r'
can't be redeclared in a MERGE clause". -/
theorem merge_dup_rel_dropped :
    ((run (patternP o0 30 .merge)
      [.lparen, .ident 1, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9, .rbrack, .dash, .gt,
       .lparen, .ident 2, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 9, .rbrack, .dash, .gt,
       .lparen, .ident 3, .rparen]).get?.map (fun g => (g.nodes.length, g.rels.length))) = some (3, 1) := by
  decide

/-- `[:A:B]` is accepted (C: syntax error). -/
theorem type_colon_accepted :
    (run (patternP o0 20 .match_)
      [.lparen, .rparen, .dash, .lbrack, .ident 5, .colon, .ident 1, .colon, .ident 2,
       .rbrack, .dash, .gt, .lparen, .rparen]).isOk = true := by decide

end FalkorParserGrammar.C
