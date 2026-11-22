/-
# The Cypher expression parser: precedence, grammar agreement, no panics

A model of FalkorDB-rs's hand-written expression parser
(`graph/src/parser/cypher.rs` at main `737dfb5da`, `ast.rs` validation at `b84473a7f`, the explicit-stack `parse_expr_inner` and its
primaries), checked against a transcription of the openCypher grammar
(`graph/src/Cypher.g4:275-560`). The lexer, literal values and depth guards
are in `proofs/lexer`, which this project does not repeat.

## What is proved

* **`parseExpr_no_panic`**: the modelled `parse_expr_inner` never reaches
  one of its panic sites, whatever the input: the two
  `children().last().unwrap()` of the comparison chain (cypher.rs), `child(0)` of a `Negate` (2549) and of a `Paren` (2636), and the two
  `unreachable!()` (2652, 2655). The proof is a stack invariant (`Inv`):
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
  same result. The slices (`..`) are not an exception.
* **`bounded_bal`**: a successful `parse_expr` consumes a
  bracket-balanced run with no top-level `,` on every list of length ≤ 3
  (`#eval`: ≤ 5 over the 21 tokens, ≤ 6 over `x 1 + ( ) [ ] ..`). This is the
  fragment's `Oracle.Bal`, the premise of `listLitP_eq_ungated`.
* Precedence and associativity for arbitrary literals: unary minus binds
  tighter than `^` (`-2^2 = 4`), `^` is left-associative, `- + * / %` fold
  left, chained comparisons desugar to `AND` (`chained_comparison`), `NOT`
  binds looser than comparison, `IS NULL`/`STARTS WITH` sit between
  comparison and `+`, and postfix `.p` / `[i]` / `[i..j]` bind tightest.
* **`parse_deterministic`**: the parse is a function of the input
  (backtracking restores the saved position and anonymous counter).

## Sign runs, update targets and variables

* `parse_expr_inner` level 9 reads any run of `+` / `-` (cypher.rs)
  and folds it by the parity of the `-`: none pushes `(9, None)`, `n` of them push `2 - n % 2`
  `Negate` frames, as the `NOT` run does. Model: `countSigns`, `signFrames` (the earlier `signs` is kept as
  `pre3525_signs`); `signFrames_ok` proves the no-panic invariant (`parseExpr_no_panic` is unchanged);
  `sign_runs`, `countSigns_rest` (the run is consumed whole); `known` in
  `BoundedDefs` lists every pair of adjacent signs, and `bounded_agree` / `bounded_bal` hold on the
  same 9,724 lists. A parameter value parsed through the expression fallback sets `legacy_signs`
  (`parse_param_value`, cypher.rs): `C.legacySigns` models the loop (`legacySigns_runs`,
  `legacySigns_neg_le`: at most one `-`, `+`, `-` or `+-`). The Rust-side `Parser::new` field is false.
* `parse_update_target` (cypher.rs) is the `SET` / `REMOVE` target and `.prop` chain used by
  `parse_set_items` / `parse_remove_items`. `C.updateTargetP` models it, `setItems` and
  `removeItems` call it, `GUpdate` / `updateTargetP_sound` carry the grammar side; `setItems_sound`,
  `removeItems_sound`, `np_setItems`, `np_removeItems`. The depth
  charges (`check_depth` on the target and after every lookup, `nested` on a nested FOREACH) only add
  failures, so soundness is unaffected; they are in `proofs/lexer`.
* `Variable.alias: Option<Alias>`; `name`, `is_unnamed`, `as_str`, `fmt`
  are derived from it and `filter_visited` keys by `VarKey` (`id`, `scope_id`). `C.Var` is
  `{alias : Option Alias, id, scope}` with `Var.name`, `Var.isUnnamed`, `Var.fmt`, `Var.key`.

## Duplicate relationships and relationship types

* `add_pattern_relationship` (cypher.rs) replaces five copies of
  `if !query_graph.add_relationship(..) && clause == Match` (named path, anonymous path, both
  allShortestPaths arms) and the pattern-comprehension chain. It rejects a repeated alias in every
  clause (MATCH/MERGE/CREATE messages). `chainA`, `chainN` and `aspLoop` `fail` whenever
  `addRel` reports a repeat; `GExt.rust := ⟨false⟩` (no extension left). A pattern comprehension collects the
  first repeat and goes on (`repeated.get_or_insert`), the caller reports it after the speculative attempt
  succeeded, so the `Err`-to-fallback of `attempt` cannot swallow it: `pcChain` carries `(graph, repeated)`,
  `patCompP : P (ET × Bool)`, `listLitRest` fails on a `true` flag, `GPatComp` takes the flag and
  `GBracket.namedPC/anonPC` require it `false`.
  Theorems: `create_named_path_dup_rel_rejected`, `create_named_path_distinct_rels_ok`,
  `merge_dup_rel_rejected`, `pattern_comprehension_repeated_rel_refused`.
* `parse_relationship_types` (cypher.rs; used by
  `parse_relationship_pattern` and shortestPath) is `ident (| :? ident)*`: a bare `:` does not
  continue the list. `typesP` and `spTypes`; `GTypes` has no `colon` constructor
  and `Ext` no `typeColon` field (`typesP_sound`), `GSpTypes` has no `:`-only case.
  `expect_property_after_paren` (cypher.rs) makes a parenthesised SET/REMOVE target require a
  following `.`: `expectDotP` in `targetP`, `GTarget.paren` carries `s2.adv.cur = .dot`.
  Theorems: `typesP_colon_refused`, `type_colon_refused`, `set_paren_target_refused`,
  `set_paren_needs_dot`.

## Same-alias allShortestPaths arrow

`parse_pattern` records `is_incoming` (the leading `<`, cypher.rs) before
`parse_relationship_pattern`, because `(a)<-[*]-(a)` loses its direction once from/to are swapped
(both ends share alias `a`). `C.Pattern.aspRel` takes that flag (`asp := rev` when
`!bidir && (src.alias != la || inc)`), `C.Pattern.aspLoop` passes `c = .lt`, and `GAsp.varLen` carries
`s.cur = .lt`. `aspRel_same_alias`: with `src.alias = la`, `<-` gives `.rev` and `->` gives `.fwd`.
COVERAGE.tsv is the authoritative line list (docstring cites in this project may point at earlier targets).

## A WHERE pattern predicate stops before a clause keyword

`C.PredicateBugs`: the property `PredEndsBeforeClause` (a WHERE pattern
predicate stops before a clause keyword) and `fixed2502_pred_stops_at_match`,
`fixed2502_where_match_split` (`bug2502_pred_swallows_match`, `bug2502_where_match_merged` are the
earlier `parse_pattern(&Keyword::Match)` behaviour, kept as historical counterexamples):
`MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` parses to two clauses.

## Disagreements (all CONFIRMED against the Rust)

Repros are in `graph/tests/lean_parser_grammar.rs`; the `bug_*` tests
fail while the bug is present. "Live" results compare the release Rust
module with the C module (`bin/macos-arm64v8-release/falkordb.so`).

| Lean | Rust | test |
| --- | --- | --- |
| — | Pattern predicate in a projection swallows the `,` (2103 → parse_pattern's comma loop 1546-1548). Live: `RETURN (a)-->(b), 1` is a syntax error; `RETURN (a)-->(b), (b)-->(a)` returns ONE column; C returns two. Same root cause as W2-binder-1. | `bug_pattern_predicate_swallows_projection_comma` |
| `sign_runs` (was `sign_sequences`) | a run of signs parses and folds by the parity of its `-` (`- -1`, `-+1`, `+-1`, `--5`); the grammar (Cypher.g4:345) allows one sign, so this is now a documented extension (`known` in `BoundedDefs`). A `CYPHER` parameter value keeps the one-sign grammar (`legacy_signs`, `C.legacySigns`). Earlier shape: `pre3525_signs`. | `chained_unary_signs_parse`, `param_of` tests in cypher.rs |

### The pattern-comprehension gate and memo (`C.ListGate`)

Alternatives 2-3 of `parse_list_literal_or_comprehension` run only behind a gate
(`(` or a name, and `list_has_top_level_pipe`) and a memo of failed attempts (otherwise backtracking is
exponential: `[(x {k: [(x {k: ...})]})]`, 18 levels, 206 bytes, took 49 s; C is exponential too).
Live (release, port-local, separate stress runs): the four query families of the Rust
regression test (`nested_pattern_comprehension_attempts_are_not_exponential`, which runs them at
40 levels) at 80 levels, and a mixed family at 16 levels (1.9 MB), parse in ≤ 0.05 s.

| Lean | claim |
| --- | --- |
| `pipeScan_exact`, `pipeScan_exact0` | the one-counter lookahead equals a kind-matched bracket stack wherever that stack meets no mismatched closer; `pipeScan_mismatch`: `[([)] \| 1]` is where they differ (the parse still rejects it) |
| `patCompP_pipe` | given `Oracle.Bal` (expression and map parsers consume bracket-balanced runs with no top-level `\|`/`,`), a pattern comprehension that parses implies the gate is open — through `parse_node_pattern` / `parse_relationship_pattern` (`cn_nodeP`, `cn_relP`, …) |
| `listLitP_eq_ungated` | CONDITIONAL on `Oracle.Bal` and `MemoSound`: the merged parser returns exactly what the ungated one (`listLitPU`) returns, wherever that one does not run out of model fuel |
| `memoRun_sound` | if an attempt's outcome is a function of its key (`AttemptDet`), the memo only holds failing keys (= `MemoSound`). Not discharged: the key omits `anon_counter`, and an attempt can read anonymous aliases (`Alias::Anon(N)` in `parse_pattern`'s duplicate-relationship check); no input was found where the same key recurs with a different counter and outcome |
| `memoRun_nodup`, `failed_attempts_linear` | each key fails at most once, so any visit list of the abstract `memoRun` model with `pos ≤ n` and depth ≤ `MAX_NESTING` has at most `808·(n+1)` failed attempts (not linked to actual parser runs; the premises on a whole parse are by inspection). Successful attempts are not memoised and not bounded here |

### Slice bounds

`slice_bound_error_reported`, `listOp_from_err`, `listOp_to_err`, `slice_bounds_omitted`: a bound is left out
only when `..` / `]` comes where it would start; one that fails to parse is an error (cypher.rs);
`slice_unclosed_paren_refused` (`[1][(1..2]` is refused), so `bounded_agree` needs no slice exception (length ≤ 5 by `#eval`).

### Correctness theorems

* `not_not_kept`, `not_run_folds`: a run of NOTs folds to one or two (cypher.rs); ≥ 3 value-equal.
* `is_null_then_predicate`: level 5 is re-entered after `IS [NOT] NULL` (2435-2438).
* `call_trailing_comma_refused`, `C.call_trailing_comma_refused`: `parse_expression_list` only ends at once when empty (2769-2779).
* `set_bracket_paren_refused`, `setTarget_only_paren`, `C.set_bracket_target_refused`: `reject_unparenthesized_target` (3395).
* `C.match_match_refused`, `C.reading_match_at_adv`: no second `optional_match_token!(Match)` (934-941).
* `C.load_csv_with_from_refused`, `C.no_load_with_from`: `WITH` must be followed by `HEADERS` (955-958).

The bounded check (`bounded_agree`) needs no `NOT NOT`, `IS NULL`-then-predicate or
`,)` exemptions, and the clause relations have no `matchMatch` / `loadWith` / `trailingComma`
extension flags (`CExt`); `GArgsE`/`GArgs` demand an expression after every `,`.

### Clause level (wave 4, `C/*`)

* `call_validates_rest`, `call_transparent`, `match_last_after_call_rejected`: `inner_validate`'s CALL arm continues with
  `iter.next().map_or(Ok(()), …)` (ast.rs), so `CALL db.labels() YIELD label MATCH (n)` and the like are rejected, as in C.
* `fixed3303_anon_name_distinct`: anonymous entities are `Alias::Anon(N)` (cypher.rs), which no identifier equals
  (`CREATE (:A), (:B)` then `MATCH (_anon_0:A), (:B) RETURN count(*)` is 1, as in C).
* `create_named_path_dup_rel_rejected`, `merge_dup_rel_rejected`, `pattern_comprehension_repeated_rel_refused`: every
  relationship goes through `add_pattern_relationship` (cypher.rs), which rejects a repeated alias in every clause.
* `set_paren_target_refused`, `set_paren_needs_dot`: `expect_property_after_paren` (cypher.rs) lets a parenthesised
  SET/REMOVE target open only `(n).x`.
* `type_colon_refused`, `typesP_colon_refused`: `parse_relationship_types` (cypher.rs) continues only on `|` (with an optional `:` after it).

Out of this project's files but seen while probing: 2000 chained
`UNWIND [1] AS xN` clauses crash the Rust server (parse, validate and bind
succeed up to 8000; the crash is in planning/execution); C runs 5000.
`SET n.x = 1 SET n.y = n.x` gives Rust 1, C null (C evaluates the folded
SETs as one batch).

Known and not re-reported: #2605 (`.1e-5`), W2-binder-1
(`MATCH p MATCH q` folded; `OPTIONAL MATCH p MATCH q`), #2895 (SET chains,
FOREACH nesting; see `proofs/lexer`), and the lexer findings of #2909 (see `proofs/lexer`).

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
  (`Ext`, `GExt`, `CExt`): MATCH/CREATE keyword folding,
  allShortestPaths (`GExt.rust = GExt.g4`),
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
  `validate` (`vClause_simple`; `call_validates_rest`), `find_aggregate_name_spec`,
  `treeHeight_spec`.

## Here | there

| here | there |
| --- | --- |
| `Tok`, `cur`, `adv` | `Token`, `Lexer::current`, `Lexer::next` (lexer.rs) |
| `Frame`, `St`, `step`, `loop` | the `(level, Option<tree>, height)` stack of `parse_expr_inner` (2176-2661) |
| `ret` | `parse_expr_return!` (macro.rs) |
| `binStep`, `opTok` | `parse_operators!` (macro.rs), levels 0,1,2,6,7,8 |
| `cmpStep`, `lastCmpIsOrdering` | level 4, chained comparisons (2286-2357) |
| `predStep`, `isLoop` | level 5 (2358-2549) |
| `countNots`, `notFrames`, `countSigns`, `signFrames` | level 3 / level 9 prefixes (2189-2250; the level 9 prefix is a run) |
| `postStep`, `postLoop` | level 10 (2571-2634) |
| `closeStep` | level 11 (2635-2656) |
| `primary`, `dotted`, `fnTable`, `exprList` | `parse_primary_expr` (1923-2116), `parse_dotted_ident`, `get_functions`, `parse_expression_list` |
| `listLit`, `listComp` | `parse_list_literal_or_comprehension`, `parse_list_comprehension` (2818-2876, 2915-2950) |
| `C.pipeScan`, `C.gateOK`, `C.listLitP`, `C.listLitPU`, `Oracle.seen` | `list_has_top_level_pipe` (2892-2912), the gate (2845-2878), the merged / ungated `parse_list_literal_or_comprehension`, `not_pattern_comprehension` |
| `listOp` | `parse_list_operator_expression` (2117-2147) |
| `parseMap`, `mapProj*`, `labels` | `parse_map` (3220), `parse_map_projection` (3252), `parse_labels` (3211) |
| `skipFilter`, `C.skipP` | `parse_shortest_path_expr`'s earlier filter skipper (1833-1858 at 3f4ea8f40; `skipFilter_eof_diverges`, historical); `C.spFilterP` is the current filter (1833-1843) |
| `C.updateTargetP`, `targetP`, `dotChain`, `rejectUnparenP`, `expectDotP` | `parse_update_target` (3394), the target of `parse_set_items` / `parse_remove_items` (3340, 3457), `reject_unparenthesized_target` (3420), `expect_property_after_paren` (3434) |
| `G.*` | `oC_Expression` … `oC_Atom` (Cypher.g4:275-560) |
| `C.CT`, `C.CK`, `C.P`, `tok`/`opt`/`ident`/`tryIdent` | `Token`, `Keyword` (lexer.rs), the parser, `match_token!` / `optional_match_token!` (macro.rs), `parse_ident` / `try_parse_ident` |
| `C.QG`, `addNode`, `mergeNode`, `addRel`, `addPath`, `mergeAttrs` | `QueryGraph` and its methods, `merge_attr_maps` (ast.rs) |
| `C.dfs`, `C.dfsRels`, `C.ccLoop`, `C.filterVisited` | `QueryGraph::dfs`, `connected_components`, `filter_visited` (ast.rs) |
| `C.IR`, `C.vClause`, `C.returnCols` | `QueryIR`, `inner_validate`, `return_column_names` (ast.rs) |
| `C.nodeP`, `C.relP`, `C.patternP`, `C.addPatNode` | `parse_node_pattern`, `parse_relationship_pattern`, `parse_pattern`, `add_pattern_node` |
| `C.queryP`, `C.singleP`, `C.segments`, `C.readingP`, `C.writingP`, `C.callP`, `C.foreachP` | `parse_query`, `parse_single_query`, its loops, `parse_reading_clasue`, `parse_writing_clause`, `parse_call_clause`, `parse_foreach_clause` |
| `C.projP`, `C.namedExprs`, `C.oslP`, `C.whereP`, `C.setItems`, `C.removeItems`, `C.deleteP`, `C.mergeP`, `C.loadCsvP` | the corresponding `parse_*` |
| `C.indexOps`, `C.parseP`, `C.endP` | `parse_index_ops`, `parse`, `expect_end_of_input` |
| `C.caseP`, `C.quantP`, `C.reduceP`, `C.listCompP`, `C.patCompP`, `C.listLitP`, `C.shortestP`, `C.primaryC` | `parse_case_expression`, `parse_quantifier_expr`, `parse_reduce_expr`, `parse_list_comprehension`, `parse_pattern_comprehension`, `parse_list_literal_or_comprehension`, `parse_shortest_path_expr`, `parse_primary_expr` |
| `C.litP`, `C.paramValueP`, `C.paramsP` | `parse_literal` … `parse_literal_map`, `parse_param_value`, `parse_parameters` |

## What this does NOT cover

* The token alphabet has no `CASE`, quantifiers, `reduce`, `shortestPath`,
  `DISTINCT`, `count(*)`, aggregates, `|`, pattern predicates or pattern
  comprehensions, and no floats. Without `|` the gate is always shut,
  so pattern-comprehension attempts are never made (the clause level, `C/*`,
  models them, the gate and the memo).
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
* Relationship types go through a `HashSet` (cypher.rs): the model keeps
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
import FalkorParserGrammar.C.ListGate
import FalkorParserGrammar.C.PredicateBugs
