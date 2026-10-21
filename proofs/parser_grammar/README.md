# parser_grammar: does the parser read Cypher the way the grammar says?

This project proves that FalkorDB's hand-written Cypher parser never panics and builds the tree the openCypher grammar (`graph/src/Cypher.g4`) prescribes. It also gives concrete counterexamples where the parser accepts what the grammar rejects, or loops forever. Every query goes through it. User-facing behaviour that depends on it:

- operator precedence: `-2^2`, `1 + 2 * 3`, `NOT a = b`, `x IS NULL`, chained comparisons such as `1 < x < 3`;
- which queries are syntax errors and which are accepted;
- the pattern graph a MATCH / CREATE / MERGE builds, and the validation that runs on it;
- whether a malformed query can hang a worker thread or crash the server.

The Rust code is `graph/src/parser/cypher.rs` and `graph/src/parser/ast.rs`. The lexer, literal values and the depth guards are in [`../lexer`](../lexer), which this project does not repeat. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorParserGrammar.lean`](FalkorParserGrammar.lean). This README explains *what* is proven and *why it matters*.

The project has two layers:

- the **expression layer** (`Model.lean`, `NoPanic.lean`, `Theorems.lean`, `Bounded.lean`): a line-by-line model of `parse_expr_inner`, the explicit-stack expression parser;
- the **clause layer** (`C/*`): every clause parser, patterns, the full `parse_primary_expr`, query parameters and the `ast.rs` query graph and validation.

## The questions this project answers

**1. Can the expression parser panic?**

`parse_expr_inner` has six panic sites:

- two `children().last().unwrap()` in the comparison chain;
- `child(0)` of a `Negate` and of a `Paren`;
- two `unreachable!()`.

- **No.** `parseExpr_no_panic` proves the modelled parser never reaches any of them, on any input, and that every tree it returns is well formed (`Good`: every `Negate`, `Paren` and comparison has a child).
- **At the clause level, no.** `parse_no_panic` covers index DDL, every clause and end of input, including the two `unreachable!()` dispatchers. `primaryC_np` covers `parse_primary_expr`, and `np_paramsP` the query-parameter parser.

**2. Is operator precedence what the grammar says?**

- **Yes, for every pair of binary operators.** `binary_pairs_agree` checks all 400 ordered pairs of the 20 binary operators (`OR XOR AND = <> < <= > >= IN STARTS WITH ENDS WITH CONTAINS =~ + - * / % ^`): `1 op1 2 op2 3` gives the grammar's tree.
- **For short inputs, exhaustively.** `bounded_agree` proves that on all 9,724 token lists of length ≤ 3 over a 21-token alphabet, the parser and the grammar accept the same inputs with the same trees, apart from the documented classes in `known`. The same check was run with `#eval` (not as a proof) over all 4.3 million lists of length ≤ 5, with the same result.
- **For arbitrary operands:**
  - unary minus binds tighter than `^`, so `-2^2 = 4`;
  - `^` is left-associative;
  - `- + * / %` fold left;
  - chained comparisons desugar to `AND` (`chained_comparison`);
  - `NOT` binds looser than comparison;
  - `IS NULL` and `STARTS WITH` sit between comparison and `+`;
  - postfix `.p`, `[i]` and `[i..j]` bind tightest.
- `parse_deterministic`: the parse is a function of the input; backtracking restores the saved position and the anonymous-name counter.

**3. Is every accepted query grammatical, with the grammar's tree?**

- **Yes, apart from listed extensions.** For every clause parser, an accepted input is derivable in a transcription of the matching `Cypher.g4` rule, and the tree returned is that derivation's. The top-level theorems are `parseP_sound` and `queryP_sound`, with `patternP_sound`, `primaryC_sound`, `shortestP_sound`, `listLitP_sound` and `lit_sound` for the pieces.
- The extensions are explicit flags (`Ext`, `GExt`, `CExt`), not silent. They include `[:A:B]`, keyword folding (`MATCH … MATCH`), allShortestPaths, and FalkorDB-only constructs: index DDL, LOAD CSV, FOREACH, `CALL {}`, reduce and shortestPath.
- Completeness is **not** claimed: the parser rejects some grammatical inputs on purpose, for example aggregates in WHERE, `SKIP 1+1`, mixed UNION / UNION ALL, and undirected CREATE.

**4. Is the #3104 pattern-comprehension gate safe?**

Before #3104, `[(x {k: [(x {k: …})]})]` nested 18 levels deep (206 bytes) took 49 s to parse. #3104 runs the pattern-comprehension attempt only behind a gate (`list_has_top_level_pipe`) and memoises failed attempts.

- **The gate is exact on balanced brackets** (`pipeScan_exact`).
- **Failed attempts are linear**: at most `808·(n+1)` on `n` bytes (`failed_attempts_linear`, `memoRun_nodup`).
- **"The gate never changes the parse" is CONDITIONAL.** `listLitP_eq_ungated` holds under two premises, `Oracle.Bal` and `MemoSound`. `Oracle.Bal` was **false** for the Rust until #3041 because of the slice-bound bug (`pre3041_gate_changes_result`): `MATCH (a) RETURN [(a)-->(b) WHERE [1][(1..2] | b]` was accepted before #3104 and a syntax error after (C's answer). #3041 makes a failed slice bound an error, so `[1][(1..2]` is refused (`slice_unclosed_paren_refused`), and the expression fragment consumes balanced runs on every input checked (`bounded_bal`). `Oracle.Bal` is no longer refuted, but it is not proved for the whole expression parser (CASE, comprehensions, maps, … are outside the fragment). `MemoSound` is unproven: the memo key omits the anonymous counter, and no input was found where that matters (`memoRun_sound`).

**5. Does `connected_components` split a pattern correctly?**

- **Yes.** `connectedComponents_spec` proves the components are node-disjoint, consist of the graph's nodes or relationship endpoints, and cover every node.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `parseExpr_no_panic` | The expression parser never reaches a panic site, and returns well-formed trees | `NoPanic.lean` |
| `binary_pairs_agree` | All 400 binary-operator pairs parse to the grammar's tree | `Theorems.lean` |
| `bounded_agree` | Every token list of length ≤ 3 is parsed exactly as the grammar parses it, apart from `known` classes (no slice class since #3041) | `Bounded.lean` |
| `slice_bound_error_reported` | A slice bound that fails to parse is an error, as in the grammar and C (#3041) | `Theorems.lean` |
| `bounded_bal` | A successful expression parse consumes a balanced run (the fragment's `Oracle.Bal`) | `Bounded.lean` |
| `chained_comparison` | `a < b < c` becomes `a < b AND b < c` | `Theorems.lean` |
| `parse_deterministic` | The parse is a function of the input | `Theorems.lean` |
| `parseP_sound`, `queryP_sound` | Every accepted query is derivable in the grammar, with the derivation's tree | `C/Index.lean`, `C/ClausesG3.lean` |
| `patternP_sound`, `primaryC_sound`, `shortestP_sound`, `listLitP_sound` | The same for patterns, primaries, shortestPath and list literals | `C/PatternG2.lean`, `C/PrimaryG.lean`, `C/ExprC5.lean`, `C/ExprC3.lean` |
| `parse_no_panic` | The clause layer never panics | `C/Index.lean` (clause lemmas in `C/ClausesNP.lean`) |
| `pipeScan_exact`, `failed_attempts_linear` | #3104's gate is exact on balanced input; failed attempts are linear | `C/ListGate.lean` |
| `connectedComponents_spec` | Pattern components are disjoint and cover every node | `C/Components.lean` |
| `not_not_kept`, `is_null_then_predicate` | `NOT NOT x` keeps both NOTs; `1 IS NULL IN [false]` parses (since #3060) | `Theorems.lean` |

### How `parseExpr_no_panic` is proved

1. **Model the tokens and trees.** `Tok` is the expression fragment of the lexer's tokens; `RT` is the expression tree.
2. **Model the parser.** `parse_expr_inner` runs a loop over a stack of `(level, tree, height)` frames. `Frame` and `St` model the stack and state. `step` is one iteration, with one Lean function per level: `binStep` for the binary levels, `cmpStep` for comparison chains, `predStep` for `IS NULL`/`STARTS WITH`/…, `postStep` for postfix operators, and `closeStep` for parentheses. A state `.panic` marks each panic site.
3. **State the invariant.** `Inv` says:
   - the stack is not empty;
   - every frame's level is at most 11;
   - every frame's tree has well-formed children (`AlmostGood`);
   - the top frame's tree, when it has one, is itself well formed (`Good`).

   This holds because a `Negate`, `Paren` or comparison is only ever pushed *under* the frame that will produce its child.
4. **Show each step keeps the invariant.** `StOK` lifts `Inv` to the parser state and is false for `.panic`. Each step from an `StOK` state reaches an `StOK` state, so `.panic` is unreachable.
5. **Hypotheses.** None about other Rust code. The step budget (`fuel`) belongs to the model: if it runs out, the result is a non-panic error.

## Bugs found

Each bug was confirmed against the Rust (`graph/tests/lean_parser_grammar.rs`, not committed; the `bug_*` tests fail while the bug is present), and live against the C module (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W3-parser-1 | `skipFilter_eof_diverges` | `RETURN shortestPath((a)-[{` at end of input: the filter-skip loop never ends, a worker spins forever, and 11 such queries stop all graph queries. C: syntax error | **Open** |
| #3036 | `pre3041_slice_bound_error_swallowed` (history) | A slice bound that failed to parse became `0` or `i64::MAX`: `[1,2,3][abs()..2]` → `[1, 2]` (C: arity error); `[(1..2]` was accepted | Fixed by #3041: `slice_bound_error_reported`, `listOp_from_err`, `listOp_to_err` |
| — | (Rust test only) | A pattern predicate in a projection swallows the `,`: `RETURN (a)-->(b), (b)-->(a)` returns one column (C: two). Same root cause as W2-binder-1 | **Open** |
| — | `sign_sequences` | `+-1` is accepted, `- -1` and `-+1` are rejected; the grammar allows one sign (`- -1` is #2309) | **Open** |
| W4-plan-4 | `call_validates_rest`, `match_last_after_call_rejected`; history `pre3297_call_skips_rest_validation`, `pre3297_match_last_after_call_accepted` | Nothing after a procedure CALL was validated: `… MERGE ()-[:R\|S]->()` created an edge, `… UNWIND [1] AS x` was accepted; C rejects all | **Fixed by #3297** (b774fdfad, main b84473a7f); the `parent().unwrap()` in `utilize_node_by_id` remains as a hazard |
| W4-parse-1 | `anon_name_collides` | Anonymous entities are named `_anon_N`, a name users can write: `MATCH (_anon_0:A), (:B) RETURN count(*)` → Rust 0, C 1 | **Open** |
| W4-parse-2 | `create_named_path_drops_dup_rel` | `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d)` creates 1 relationship (C: "already declared"); since #2829 `RETURN p, q` also returns an inconsistent path | **Open** |
| W4-parse-3 | `merge_dup_rel_dropped` | `MERGE (a)-[r:R]->(b)-[r:R]->(c)` creates 1 relationship (C: error) | **Open** |
| W4-parse-4 | `set_paren_property_accepted`, `type_colon_accepted` | `SET (n.x) = 5` and `[:A:B]` are accepted (C: syntax error) | **Open** |
| #2502 | `bug2502_pred_swallows_match`, `bug2502_where_match_merged` (property `PredEndsBeforeClause`) | A WHERE pattern predicate is parsed by `parse_pattern(&Keyword::Match)` (cypher.rs:2119), whose separator (1562-1579) continues on `MATCH`, so the next MATCH clause is folded into the predicate: `MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` → Rust `'m' not defined`, C returns the rows; with `RETURN count(*)` Rust 1, C 2 | **Open** |
| W3-parser-9 | `pre3041_gate_changes_result` (history) | Exponential backtracking in nested `[(...)]` lists: 206 bytes took 49 s | Fixed by #3104 |
| #3057 | `not_not_kept`, `is_null_then_predicate`, `call_trailing_comma_refused`, `set_bracket_paren_refused`, `match_match_refused`, `load_csv_with_from_refused` | `NOT NOT x` was parsed as `x`; `1 IS NULL IN [false]` was rejected; `abs(-1,)`, `SET [n).x = 5`, `MATCH MATCH` and `LOAD CSV WITH FROM` were accepted | Fixed by #3060 (the theorems are now the correct behaviour) |

Out of this project's files but seen while probing: 2000 chained `UNWIND [1] AS xN` clauses crash the Rust server in planning or execution (W4-parse-5); C runs 5000.

**Rust follows the grammar and C does not** (each needs a decision, see FINDINGS "Decisions needed"):

- `2^3^2`: Rust 64 (left-associative, as the grammar says), C 512.
- Chained comparisons: `1 < 2 < 3 < 0` → Rust false, C true.
- `'a' + 'b' STARTS WITH 'ab'`: Rust true, C `'afalse'`.
- `SKIP (1)`: Rust rejects, C accepts.

**CONDITIONAL:** `listLitP_eq_ungated` rests on `Oracle.Bal` (refuted before #3041; since #3041 unproven but with no known counterexample, and checked on the fragment by `bounded_bal`, see question 4) and `MemoSound` (unproven). The `parse_list_literal_or_comprehension` row in `COVERAGE.tsv` is PROVEN through `listLitP_sound` and the no-panic theorem, which do not need those premises; only the "same as before #3104" claim is conditional.

## Files

| File | What's in it |
|---|---|
| `FalkorParserGrammar.lean` | The report header and imports |
| `Model.lean` | The expression parser: tokens `Tok`, trees, frames, `step`, one function per level, `primary`, `listLit`, `listOp` (and `pre3041_listOp`) |
| `Grammar.lean` | The reference: openCypher's expression rules (`G.*`) as a recursive-descent recogniser |
| `NoPanic.lean` | `Good`, `Inv`, `parseExpr_no_panic` |
| `Theorems.lean` | Precedence, associativity, `binary_pairs_agree`, determinism, the #3060 fixes, `sign_sequences`, the #3041 slice fix and `pre3041_slice_bound_error_swallowed` |
| `Bounded.lean` | `bounded_agree` and the `known` disagreement classes; `bounded_bal` (balanced consumption) |
| `Clauses.lean` | Clause-level fragments: the shortestPath filter skipper, SET/REMOVE targets, the projection-comma bug |
| `Tactic.lean` | Proof helpers |
| `C/Core.lean` | Clause-layer tokens, the parser monad and its primitives |
| `C/Clauses.lean` | The clause parsers, one model function per Rust function |
| `C/ClausesG.lean`, `C/ClausesG2.lean`, `C/ClausesG3.lean` | The clause grammar and soundness of each clause parser, up to `parse_query` |
| `C/ClausesNP.lean` | No panics at the clause level |
| `C/Pattern.lean`, `C/PatternG.lean`, `C/PatternG2.lean` | Node, relationship and path patterns, their grammar and soundness |
| `C/Primary.lean`, `C/PrimaryG.lean` | The full `parse_primary_expr` and its soundness |
| `C/ExprC.lean` … `C/ExprC5.lean` | CASE, quantifiers, reduce, list and pattern comprehensions, shortestPath |
| `C/ListGate.lean` | The #3104 gate and memo |
| `C/Index.lean` | Index DDL and the entry point `parse`: `parseP_sound`, `parse_no_panic` |
| `C/Params.lean` | Query parameters and small helpers |
| `C/Graph.lean`, `C/Components.lean`, `C/IR.lean`, `C/Fmt.lean` | `ast.rs`: `QueryGraph`, connected components, `QueryIR` validation, `Display` |
| `C/ClauseBugs.lean`, `C/PatternBugs.lean` | The clause- and pattern-level counterexamples |
| `C/PredicateBugs.lean` | #2502: `PredEndsBeforeClause` (a WHERE pattern predicate ends before a clause keyword) and its counterexamples |
| `C/Oracle0.lean` | A small concrete expression oracle for running the clause models on examples |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 112 rows: 74 in `parser/cypher.rs` and 38 in `parser/ast.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 112 |

Every function listed is PROVEN. For the functions involved in open bugs (for example `parse_shortest_path_expr`), the row cites both the soundness theorem and the counterexample.

## What the model does not cover

- The expression layer's token alphabet has no `CASE`, quantifiers, `reduce`, `shortestPath`, `DISTINCT`, `count(*)`, aggregates, `|`, pattern predicates or pattern comprehensions, and no floats. The clause layer models those.
- At the clause level the expression parser is an *oracle* (`Oracle`). Soundness holds for every oracle; no-panic holds for every oracle that does not panic (`Oracle.Safe`). `parseExpr_no_panic` gives that for the expression model, but the two layers use different token types, and the bridge between them is by inspection.
- Soundness is "accepted ⇒ grammatical, with the grammar's tree". Completeness is not claimed.
- Tree heights, `check_depth` and `nested` are abstracted away; see [`../lexer`](../lexer).
- The step budget (`fuel`) is the model's. It is not proven that more fuel never changes a result, so parsers and `connected_components` are proven partially correct: nothing is claimed when fuel runs out.
- Relationship types go through a `HashSet`; the model keeps first-occurrence order, and only membership is observable.
- Identifier length limits, error messages and error positions are not modelled.
- The header cites `cypher.rs` line numbers from main at `737dfb5da` (the `ast.rs` CALL arm at `b84473a7f`, after #3297); the theorems are re-checked against current main by `lean_ci.sh`, but the line numbers may have drifted.

## Build and check

```bash
cd proofs/parser_grammar && lake build   # this project
bash proofs/lean_ci.sh parser_grammar    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `parser/cypher.rs` or `parser/ast.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A new syntax extension should get a flag in `Ext`/`GExt`/`CExt` and a grammar rule, so that `parseP_sound` keeps saying exactly what is accepted beyond openCypher.
