/-
# The Cypher expression parser: precedence, grammar agreement, no panics

A model of FalkorDB-rs's hand-written expression parser
(`graph/src/parser/cypher.rs` at main `fe619ac5f`, the explicit-stack `parse_expr_inner` and its
primaries), checked against a transcription of the openCypher grammar
(`graph/src/Cypher.g4:275-560`). The lexer, literal values and depth guards
are in `proofs/lexer`, which this project does not repeat.

## What is proved

* **`parseExpr_no_panic`**: the modelled `parse_expr_inner` never reaches
  one of its panic sites, whatever the input: the two
  `children().last().unwrap()` of the comparison chain (cypher.rs:2300,
  2305), `child(0)` of a `Negate` (2526) and of a `Paren` (2598), and the two
  `unreachable!()` (2614, 2617). The proof is a stack invariant (`Inv`):
  levels ≤ 11, the stack is not empty, and the top frame's tree is `Good`
  (every `Negate`/`Paren`/comparison has a child). It also shows every
  returned tree is `Good`.
* **`binary_pairs_agree`**: for all 400 ordered pairs of the 20 binary
  operator spellings (`OR XOR AND = <> < <= > >= IN STARTS WITH ENDS WITH
  CONTAINS =~ + - * / % ^`), `1 op1 2 op2 3` gives the grammar's tree.
* **`bounded_agree`**: on all 9,724 token lists of length ≤ 3 over a
  21-token alphabet, model and grammar accept the same inputs with the same
  trees, apart from the documented disagreement classes (`known`). The same
  check was run with `#eval` over all 4.3 million lists of length ≤ 5, with the
  same result.
* Precedence and associativity for arbitrary literals: unary minus binds
  tighter than `^` (`-2^2 = 4`), `^` is left-associative, `- + * / %` fold
  left, chained comparisons desugar to `AND` (`chained_comparison`), `NOT`
  binds looser than comparison, `IS NULL`/`STARTS WITH` sit between
  comparison and `+`, and postfix `.p` / `[i]` / `[i..j]` bind tightest.
* **`parse_deterministic`**: the parse is a function of the input
  (backtracking restores the saved position and anonymous counter).

## Disagreements (all CONFIRMED against the Rust)

Repros are in `graph/tests/lean_parser_grammar.rs`; the `bug_*` tests
fail while the bug is present. "Live" results compare the release Rust
module with the C module (`bin/macos-arm64v8-release/falkordb.so`).

| Lean | Rust | test |
| --- | --- | --- |
| `skipFilter_eof_diverges` | `shortestPath((a)-[{` at end of input: the filter-skip loop (cypher.rs:1820-1846) never ends; a worker thread spins forever, and 11 such queries stop all graph queries. C: syntax error. | `bug_shortest_path_filter_skip_loops_forever_at_eof` |
| `slice_bound_error_swallowed` | A failed slice bound becomes `0` / `i64::MAX` (cypher.rs:2124-2132). Live: `[1,2,3][abs()..2]` → `[1, 2]` (C: arity error), `[1,2,3][1+..2]` → `[1, 2]`, `[(1..2]` accepted. | `bug_slice_bound_error_is_swallowed` |
| — | Pattern predicate in a projection swallows the `,` (2103 → parse_pattern's comma loop 1546-1548). Live: `RETURN (a)-->(b), 1` is a syntax error; `RETURN (a)-->(b), (b)-->(a)` returns ONE column; C returns two. Same root cause as W2-binder-1. | `bug_pattern_predicate_swallows_projection_comma` |
| — | Backtracking in `parse_list_literal_or_comprehension` (2807-2814) is exponential: `[(x {k: [(x {k: ...})]})]`, 18 levels (206 bytes) took 49 s. C is exponential too (8 levels > 500 s). | `bug_pattern_comprehension_backtracking_is_exponential` |
| `sign_sequences` | `+-1` accepted, `- -1` / `-+1` rejected; the grammar allows one sign. `- -1` is #2309. | — |

### Fixed by #3060 (`fe619ac5f`, closes #3057) — now correctness theorems

| Lean (now) | was (2c874022a) | fix |
| --- | --- | --- |
| `not_not_kept` | `not_not_dropped`: `NOT NOT x` → `x`; live `RETURN NOT NOT 1` → 1 | a run of NOTs folds to one or two (cypher.rs:2193-2203); `not_run_folds` (≥ 3, value-equal) |
| `is_null_then_predicate` | `is_null_then_predicate_rejected`: `1 IS NULL IN [false]` rejected | level 5 is re-entered after `IS [NOT] NULL` (2410-2413) |
| `call_trailing_comma_refused`, `C.call_trailing_comma_refused` | `call_trailing_comma`: `abs(-1,)` accepted | `parse_expression_list` only ends at once when empty (2729-2739) |
| `set_bracket_paren_refused`, `setTarget_only_paren`, `C.set_bracket_target_refused` | `set_bracket_paren_accepted`: `SET [n).x = 5` ran as `n.x` | `reject_unparenthesized_target` (3289) |
| `C.match_match_refused`, `C.reading_match_at_adv` | `match_match_accepted` | the second `optional_match_token!(Match)` is gone (918-925) |
| `C.load_csv_with_from_refused`, `C.no_load_with_from` | `load_csv_with_from_accepted` | `WITH` must be followed by `HEADERS` (939-942) |

The bounded check (`bounded_agree`) no longer needs the `NOT NOT`, `IS NULL`-then-predicate or
`,)` exemptions, and the clause relations lost the `matchMatch` / `loadWith` / `trailingComma`
extension flags (`CExt`); `GArgsE`/`GArgs` now demand an expression after every `,`.

### Clause level (wave 4, `C/*`; CONFIRMED live, Rust module vs C module)

| Lean | Rust | live result |
| --- | --- | --- |
| `call_skips_rest_validation`, `match_last_after_call_accepted` | `inner_validate`'s CALL arm returns `Ok(())` without validating the clauses after it (ast.rs:1167-1201). | `CALL db.labels() YIELD label MATCH (n)` and `CALL db.labels() YIELD label CREATE ()-[r]->() RETURN label` **crash the server** (panic at planner/optimizer/utilize_node_by_id.rs:118). `... MERGE ()-[:R\|S]->()` creates an edge, `... UNWIND [1] AS x` is accepted. C rejects all four. |
| `anon_name_collides` | Anonymous entities are named `_anon_N` (cypher.rs:2935, 2972, 3043), a name the user can write; `add_pattern_node` then merges the two nodes. | `CREATE (:A), (:B)` then `MATCH (_anon_0:A), (:B) RETURN count(*)`: Rust 0, C 1. |
| `create_named_path_drops_dup_rel` | In a named path a repeated relationship variable is rejected only for MATCH (cypher.rs:1514-1522). | `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d)`: Rust creates 4 nodes, 1 relationship; C: "Variable `r` already declared". |
| `merge_dup_rel_dropped` | Same for MERGE in an anonymous path (cypher.rs:1535-1548). | `MERGE (a)-[r:R]->(b)-[r:R]->(c)`: Rust 3 nodes, 1 relationship; C: "can't be redeclared in a MERGE clause". |
| `set_paren_property_accepted` | The `=` arm of `parse_set_items` takes any primary as target (cypher.rs:3261-3275). | `CREATE (n {a:1}) SET (n.x) = 5 RETURN n` sets x; C: syntax error. |
| `type_colon_accepted` | `[:A:B]` (cypher.rs:2977-2987 continues on `:` alone). | Rust matches; C: syntax error. |

Out of this project's files but seen while probing: 2000 chained
`UNWIND [1] AS xN` clauses crash the Rust server (parse, validate and bind
succeed up to 8000; the crash is in planning/execution); C runs 5000.
`SET n.x = 1 SET n.y = n.x` gives Rust 1, C null (C evaluates the folded
SETs as one batch).

Known and not re-reported: #2309 (`- -1`), #2605 (`.1e-5`), W2-binder-1
(`MATCH p MATCH q` folded; `OPTIONAL MATCH p MATCH q`), #2895 (SET chains,
FOREACH nesting), and the lexer findings of #2909.

## Places where Rust follows the grammar and C does not

The C module is the reference implementation, but here it disagrees with
openCypher and the Rust matches the grammar. Each needs a decision:

* `2^3^2`: Rust 64 (left-assoc, grammar), C 512.
* Comparison chains: `1 = 1 = 1` gives Rust true (chained), C false;
  `1 < 2 < 3 < 0` gives Rust false, C **true**; `WHERE n.x = 0 = true` gives Rust 0 rows, C 1.
* `'a' + 'b' STARTS WITH 'ab'`: Rust true, C `'afalse'`. `1 + 2 IS NULL`: Rust false, C `1`.
  `- 1 IS NULL`: Rust false, C type error.
* `SKIP (1)`: Rust rejects (only a literal or a parameter at the root), C accepts.

Outside this project's scope, seen along the way: `true CONTAINS 'c'` and
`1 ENDS WITH 1` return null in Rust, where C raises a type error.

## Clause level: what is proved (`C/*`)

* **Soundness against Cypher.g4** (`C.PatternG*`, `C.ClausesG*`, `C.Index`,
  `C.ExprC*`, `C.PrimaryG`, `C.Params`): for every parser of the clause layer,
  an accepted input is derivable in a transcription of the Cypher.g4 rule
  (lines 30-264, 361-426) with its semantic action, and the IR / tree
  returned is the derivation's. Implementation extensions are explicit flags
  (`Ext`, `GExt`, `CExt`): `[:A:B]`, MATCH/CREATE keyword folding,
  allShortestPaths, dropped duplicate relationship variables,
  DELETE/SET/REMOVE folding, a SET/REMOVE target that is not a variable or property,
  implicit CALL, queries without RETURN; plus the
  FalkorDB-only constructs (index DDL, LOAD CSV, FOREACH, CALL {}, reduce,
  shortestPath). Top: `parseP_sound`, `queryP_sound`, `patternP_sound`,
  `primaryC_sound`, `shortestP_sound`, `listLitP_sound`, `lit_sound`.
* **No panic**: `parse_no_panic` (index DDL + every clause + end of input,
  including both `unreachable!()` dispatchers), `primaryC_np` (its quantifier
  `unreachable!()`), `np_paramsP`.
* **ast.rs**: `connectedComponents_spec` (components are node-disjoint, made
  of the graph's nodes or relationship endpoints, and cover every node whose
  id is not also a relationship/path id), the `QueryGraph` operations,
  `validate` (`vClause_simple`; the CALL bug), `find_aggregate_name_spec`,
  `treeHeight_spec`.

## Here | there

| here | there |
| --- | --- |
| `Tok`, `cur`, `adv` | `Token`, `Lexer::current`, `Lexer::next` (lexer.rs:295-300, 364) |
| `Frame`, `St`, `step`, `loop` | the `(level, Option<tree>, height)` stack of `parse_expr_inner` (2177-2618) |
| `ret` | `parse_expr_return!` (macro.rs:115-133) |
| `binStep`, `opTok` | `parse_operators!` (macro.rs:135-168), levels 0,1,2,6,7,8 |
| `cmpStep`, `lastCmpIsOrdering` | level 4, chained comparisons (2258-2329) |
| `predStep`, `isLoop` | level 5 (2330-2509) |
| `countNots`, `notFrames`, `signs` | level 3 / level 9 prefixes (2183-2221) |
| `postStep`, `postLoop` | level 10 (2543-2591) |
| `closeStep` | level 11 (2592-2613) |
| `primary`, `dotted`, `fnTable`, `exprList` | `parse_primary_expr` (1926-2117), `parse_dotted_ident`, `get_functions`, `parse_expression_list` |
| `listLit`, `listComp` | `parse_list_literal_or_comprehension`, `parse_list_comprehension` (2778-2859) |
| `listOp` | `parse_list_operator_expression` (2120-2140) |
| `parseMap`, `mapProj*`, `labels` | `parse_map` (3111), `parse_map_projection` (3143), `parse_labels` (3102) |
| `skipFilter` | `parse_shortest_path_expr`'s filter skipper (1819-1846) |
| `setTarget`, `rejectUnparen` | `parse_set_items` / `parse_remove_items` targets (3236, 3317), `reject_unparenthesized_target` (3289) |
| `G.*` | `oC_Expression` … `oC_Atom` (Cypher.g4:275-560) |
| `C.CT`, `C.CK`, `C.P`, `tok`/`opt`/`ident`/`tryIdent` | `Token`, `Keyword` (lexer.rs:63-156), the parser, `match_token!` / `optional_match_token!` (macro.rs:51-113), `parse_ident` / `try_parse_ident` |
| `C.QG`, `addNode`, `mergeNode`, `addRel`, `addPath`, `mergeAttrs` | `QueryGraph` and its methods, `merge_attr_maps` (ast.rs:622-790) |
| `C.dfs`, `C.dfsRels`, `C.ccLoop`, `C.filterVisited` | `QueryGraph::dfs`, `connected_components`, `filter_visited` (ast.rs:792-887) |
| `C.IR`, `C.vClause`, `C.returnCols` | `QueryIR`, `inner_validate`, `return_column_names` (ast.rs:928-1392) |
| `C.nodeP`, `C.relP`, `C.patternP`, `C.addPatNode` | `parse_node_pattern`, `parse_relationship_pattern`, `parse_pattern`, `add_pattern_node` |
| `C.queryP`, `C.singleP`, `C.segments`, `C.readingP`, `C.writingP`, `C.callP`, `C.foreachP` | `parse_query`, `parse_single_query`, its loops, `parse_reading_clasue`, `parse_writing_clause`, `parse_call_clause`, `parse_foreach_clause` |
| `C.projP`, `C.namedExprs`, `C.oslP`, `C.whereP`, `C.setItems`, `C.removeItems`, `C.deleteP`, `C.mergeP`, `C.loadCsvP` | the corresponding `parse_*` |
| `C.indexOps`, `C.parseP`, `C.endP` | `parse_index_ops`, `parse`, `expect_end_of_input` |
| `C.caseP`, `C.quantP`, `C.reduceP`, `C.listCompP`, `C.patCompP`, `C.listLitP`, `C.shortestP`, `C.primaryC` | `parse_case_expression`, `parse_quantifier_expr`, `parse_reduce_expr`, `parse_list_comprehension`, `parse_pattern_comprehension`, `parse_list_literal_or_comprehension`, `parse_shortest_path_expr`, `parse_primary_expr` |
| `C.litP`, `C.paramValueP`, `C.paramsP` | `parse_literal` … `parse_literal_map`, `parse_param_value`, `parse_parameters` |

## What this does NOT cover

* The token alphabet has no `CASE`, quantifiers, `reduce`, `shortestPath`,
  `DISTINCT`, `count(*)`, aggregates, `|`, pattern predicates or pattern
  comprehensions, and no floats. Pattern-comprehension attempts are modelled
  as failing, which is exact without `|`.
* Tree heights, `check_depth` and `nested` are abstracted away (see
  `proofs/lexer`). The step budget (`fuel`) is the model's; it is not
  proved that more fuel never changes a non-`fuel` result.
* Identifier length limits, error messages and error positions.
* Clause level (`C/*`): the expression parser is an *oracle* (`Oracle`:
  `parse_expr`, `parse_primary_expr(false)`, `parse_map`, the projection
  text). Soundness holds for every oracle; no-panic for every oracle that does
  not panic (`Oracle.Safe`, which `parseExpr_no_panic` gives for the
  expression model; the token types differ — `Tok` is the expression
  fragment, `CT` all of `Token` — and the bridge is by inspection). SET /
  REMOVE soundness also assumes `Oracle.PPRec` (`recurse` only after `(`/`[`).
* Soundness is "accepted ⇒ derivable, with the rule's tree"; completeness
  (every grammatical query is accepted) is not claimed — the Rust rejects
  grammatical inputs on purpose (aggregates in WHERE, `SKIP 1+1`, mixed
  UNION / UNION ALL, undirected CREATE, ...).
* Relationship types go through a `HashSet` (cypher.rs:2976): the model keeps
  first-occurrence order; only membership is observable.
* `lex_numeric` / float values, identifier-length limits, error messages and
  error positions; the `forbidden_pattern_comprehension` flag is a parameter
  (`forb`) rather than parser state; keyword case-folding of `shortestPath`
  / `CYPHER` is lexical.
* Partial correctness only for `connected_components` and the parsers: the
  model's budget (`fuel`) may run out; nothing is claimed then.
* `anon_counter += 1` could overflow `u32` only after about 4·10⁹ anonymous
  entities, which needs a query larger than Redis accepts.
-/
import FalkorParserGrammar.Model
import FalkorParserGrammar.NoPanic
import FalkorParserGrammar.Grammar
import FalkorParserGrammar.Theorems
import FalkorParserGrammar.Bounded
import FalkorParserGrammar.Clauses
import FalkorParserGrammar.C.Core
import FalkorParserGrammar.C.Graph
import FalkorParserGrammar.C.Components
import FalkorParserGrammar.C.IR
import FalkorParserGrammar.C.Pattern
import FalkorParserGrammar.C.PatternG
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0
import FalkorParserGrammar.C.PatternBugs
import FalkorParserGrammar.C.Clauses
import FalkorParserGrammar.C.ClausesNP
import FalkorParserGrammar.C.ClausesG
import FalkorParserGrammar.C.ClausesG2
import FalkorParserGrammar.C.ClausesG3
import FalkorParserGrammar.C.Index
import FalkorParserGrammar.C.ClauseBugs
import FalkorParserGrammar.C.ExprC
import FalkorParserGrammar.C.ExprC2
import FalkorParserGrammar.C.ExprC3
import FalkorParserGrammar.C.ExprC4
import FalkorParserGrammar.C.ExprC5
import FalkorParserGrammar.C.Primary
import FalkorParserGrammar.C.PrimaryG
import FalkorParserGrammar.C.Params
import FalkorParserGrammar.C.Fmt
