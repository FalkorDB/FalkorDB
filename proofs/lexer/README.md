# lexer: how FalkorDB turns query text into tokens, literals and bounded trees

This project proves that the first stage of every query, turning the query string into tokens, never panics and reads the text the way openCypher says it should. It also proves the depth guards that stop a deeply nested query from overflowing the parser's stack, and shows with counterexamples where those guards are bypassed. Every query depends on it:

- whitespace and comments (`//`, `/* … */`) are skipped exactly where the grammar allows them;
- string literals decode escapes like `\n` and `A` as openCypher specifies;
- integer literals give their exact value or an overflow error, never a wrapped number;
- `MAX_TREE_DEPTH` and `MAX_NESTING` limit how deep an expression can be.

The Rust code is `graph/src/parser/lexer.rs`, `graph/src/parser/string_escape.rs`, and the literal handling and depth guards in `graph/src/parser/cypher.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorLexer.lean`](FalkorLexer.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can the lexer panic on some input?**

Rust panics when it slices a `&str` at a byte that is not a character boundary, for example in the middle of a multi-byte `é`. The lexer slices the query text everywhere: to skip spaces, to cut out a token, to quote an error.

- **No.** `lexAll_total_cover` proves that on every input, every slice the token loop takes is on a character boundary. `lexer_never_panics` instantiates it with enough fuel for any input.
- The same theorem proves that nothing is lost or invented: the whitespace runs and token texts the lexer walks over concatenate back to exactly the input.
- `lex_numeric`, `current_str` and the error-context window are covered too (`lexNum_slices_ascii`, `currentStr_ok`, `errCtx_ok`).

**2. Are comments and whitespace skipped like the grammar says?**

- **Before #2908, no.** A block comment ended at the first `/`, not at `*/`, so `RETURN 5 /* a/ -1 //*/` returned 4 (C: 5); a lone trailing `/` was skipped as if it were whitespace; and an unterminated `/*` was accepted.
- **Before #2910, no, for `\r`.** A query with Windows line endings failed to parse.
- **On main, yes.** `rs_agrees_spec` proves that `read_spaces` skips exactly the openCypher `WHITESPACE`/`Comment` run wherever the spec accepts one. `rs_unterminated_general` proves an unterminated `/*` is not skipped, and `rs_crlf` that `\r` is whitespace.

**3. Do string escapes decode correctly?**

- **Yes, for the escapes openCypher defines.** `unesc_agrees_spec` proves `cypher_unescape` equals the spec's `EscapedChar` decoding on every string whose escapes are the lower-case spec ones (`\\ \' \" \b \f \n \r \t \uXXXX`).
- **Before #2910**, `'\u+041'` decoded to `A`, because `u32::from_str_radix` accepts a leading `+`. Now a `\u`/`\U` run must be all hex digits (`unesc_plus_rejected`, `hexRun_spec`).
- Going the other way, `unesc_escape` proves that unescaping what `cypher_escape` produces gives back the original string.

**4. Do integer literals have the right value?**

`RETURN 9223372036854775807` must give `i64::MAX`, `RETURN 9223372036854775808` must be an overflow error, and `RETURN -9223372036854775808` must give `i64::MIN`.

- **On main, yes, in every spelling.** `int_literal_correct` proves that for any non-empty digit string, decimal, hex, octal or binary, with or without leading zeros, negated or not, the parser produces the literal's value when it fits `i64` and an error otherwise. Overflow is never wrapped.
- `litExpr_min` proves the literal `2^63` is accepted only when it is directly negated, never under a postfix operator such as `9223372036854775808.x`.
- **Before #2910**, `-0x08000000000000000` (a leading zero) was rejected, `9223372036854775808.x` escaped the overflow check, and `0e5` lexed as `0` followed by the identifier `e5`. `lexNum_zero_exp` proves `0e5` is now one float literal.

**5. Does the depth guard bound every tree the parser builds?**

The parser builds expressions on an explicit stack and keeps a *height estimate* for each partial tree; `check_depth` compares the estimate with `MAX_TREE_DEPTH` (256). The guard only means something if the estimate is never below the real height.

- **For expressions, yes.** `height_sound` proves every estimate is an upper bound, and `guarded_height_le` that a parse which passes `check_depth` builds no tree taller than `MAX_TREE_DEPTH`. `chain_rejected` shows a chain of 256 levels is refused.
- `nested_depth_bound` proves every `parse_expr` / literal / `CALL {}` recursion through `Parser::nested` is at most `MAX_NESTING` (100) deep.
- **But two paths bypass the guards** (open, see [Bugs found](#bugs-found)): `SET n.a.a.a… = 1` builds its target outside `check_depth` (`set_target_unbounded`), and nested `FOREACH` never passes through `nested` (`foreach_unbounded`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `lexAll_total_cover` | On every input the token loop slices only at character boundaries (never panics), and the pieces it walks over concatenate back to the input | `FalkorLexer.lean` |
| `getTok_ok` | One call of `get_token` never panics: it returns an error, end of input, or a non-empty token that ends on a character boundary | `FalkorLexer.lean` |
| `rs_agrees_spec` | `read_spaces` skips exactly the openCypher whitespace/comment run | `FalkorLexer.lean` |
| `unesc_agrees_spec` | `cypher_unescape` equals the spec's escape decoding | `FalkorLexer.lean` |
| `int_literal_correct` | Integer literals in any spelling evaluate to their value or to an overflow error | `FalkorLexer.lean` |
| `litExpr_min` | `2^63` is accepted only directly negated | `FalkorLexer.lean` |
| `height_sound`, `guarded_height_le` | Height estimates are upper bounds, so a parse that passes `check_depth` is at most `MAX_TREE_DEPTH` tall | `FalkorLexer.lean` |
| `nested_depth_bound` | Every `nested` descent is at most `MAX_NESTING` deep | `FalkorLexer.lean` |
| `lexNum_slices_ascii` | Every slice `lex_numeric` takes sits after an all-ASCII prefix, so it is on a boundary | `FalkorLexerExtra.lean` |
| `errCtx_ok` | The error-message window never slices off a boundary | `FalkorLexerExtra.lean` |
| `setPos_next` | `set_pos(p)` rebuilds exactly the state `next()` builds | `FalkorLexerExtra.lean` |
| `unesc_escape` | `cypher_unescape(cypher_escape(s)) = s` | `FalkorLexerExtra.lean` |

### How `lexAll_total_cover` is proved

1. **Model bytes and slicing.** The query is a list of `Char`. `bytes` is `str::len`, `u8` is `char::len_utf8`, and `splitBytes cs n` is `&s[..n]` / `&s[n..]`, returning `none` exactly where Rust panics with "byte index is not a char boundary". `boundary_iff` proves that "`n` is a boundary" and "slicing at `n` succeeds" are the same thing.
2. **Model each scanner.** `rs` is `read_spaces`; `strScan`, `btScan`, `btpScan` and `identRun` are the string, backtick and identifier arms of `get_token`; `getTok` is `get_token` itself. Each scanner has a `_spec` theorem saying the length it returns is the byte length of a whole-character prefix.
3. **Abstract what does not matter for slicing.** `getTok` takes the numeric scanner (`num`) and the escape check (`esc`) as parameters. The theorem holds for *any* such functions, so it does not depend on how numbers or escapes are read; those are proven separately (`lexNum_slices_ascii`, `unesc_agrees_spec`).
4. **Induct on the input.** `lexAll` is the `Lexer::new` / `Lexer::next` loop. Each step skips a whitespace run (a boundary, by `rs_top_boundary`) and then a token (a boundary, by `getTok_ok`), so the remaining text is a suffix at a boundary and the induction hypothesis applies.

There are no hypotheses about other Rust code: the result holds for every input string.

## Bugs found

Every bug was reproduced against the real Rust (`graph/tests/lean_lexer.rs`, not committed) before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | removed (17ccc5412) | A block comment ended at the first `/`: `RETURN 5 /* a/ -1 //*/` gave 4 (C: 5) | Fixed by #2908 (`rs_block_comment_fixed`, `rs_agrees_spec`) |
| 2 | removed (17ccc5412) | A lone trailing `/` was skipped as whitespace | Fixed by #2908 (`rs_trailing_slash_fixed`) |
| 3 | removed (17ccc5412) | An unterminated `/*` was accepted | Fixed by #2908 (`rs_unterminated_fixed`, `rs_unterminated_general`) |
| 4 | removed (17ccc5412) | `\r` was not whitespace, so CRLF queries failed to parse | Fixed by #2910 (`rs_crlf`) |
| 5 | removed (17ccc5412) | `0e5` lexed as `0` then `e5` | Fixed by #2910 (`lexNum_zero_exp`) |
| 6 | removed (17ccc5412) | `-0x08000000000000000` was rejected as an overflow (C: `-9223372036854775808`) | Fixed by #2910 (`min_noncanonical_fixed`, `int_literal_correct`) |
| 7 | removed (17ccc5412) | `9223372036854775808.x` escaped the overflow check | Fixed by #2910 (`litExpr_min`) |
| 8 | removed (17ccc5412) | `'\u+041'` decoded to `A` | Fixed by #2910 (`unesc_plus_rejected`) |
| 9 | `set_target_unbounded` | `MATCH (n) SET n.a.a…(5000) = 1` builds its target outside `check_depth`; the binder then overflows its stack. C rejects it | **Open** (#2895, PR #2900 not merged) |
| 10 | `foreach_unbounded` | Nested `FOREACH (x IN [1] \| FOREACH (…))` never goes through `nested`; 100 000 levels abort the process with a stack overflow (C crashes as well) | **Open** (#2895, PR #2900 not merged) |
| 11 | `hop_truncation_bug` | A hop bound is cast `i as u32`, so `*4294967298` means `*2` | Not filed: C truncates the same way |

There are no CONDITIONAL rows in this project.

## Files

| File | What's in it |
|---|---|
| `FalkorLexer.lean` | The report header; bytes and boundaries (`splitBytes`, `boundary_iff`); `read_spaces` (`rs`) against the spec (`specSkip`, `rs_agrees_spec`) and the pre-#2908/#2910 versions; the token scanners, `getTok` and the loop `lexAll`; `cypher_unescape` (`unescF`) against `specUnescF`; integer literals (`str2int`, `evalLit`, `litExpr`); the hop cast `asU32`; the height model (`T`, `Reach`, `height_sound`) and `SET` targets; the call-recursion model (`Call`, `runCall`, `foreach_unbounded`) |
| `FalkorLexerExtra.lean` | The rest of `lexer.rs` and `string_escape.rs`: `pos`, `next`, `set_pos`, `current`, `current_str` (`LexSt`), `lex_numeric` (`lexNum`, `intLoop`, `floatLoop`), `err_ctx`/`format_error`, the `Token` `Display` impl (`tokDisplay_injective_punct`), and `cypher_escape` (`esc`, `unesc_escape`) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 89 rows: all of `lexer.rs` and `string_escape.rs`, plus the functions of `cypher.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 21 |
| MODELLED | 9 |
| NOT COVERED | 59 |

- **PROVEN:** all 14 `lexer.rs` functions, `cypher_unescape` and `cypher_escape`, and in `cypher.rs` `check_depth`, `nested`, `parse_literal`, `parse_expr` and `parse_expr_inner`.
- **MODELLED:** `cypher.rs` functions of which this project models only a fragment: the hop-bound cast in `parse_shortest_path_expr` and `parse_relationship_pattern`, the property-target loop of `parse_set_items`/`parse_remove_items`, the recursion shape of `parse_foreach_clause`, the integer arm of `parse_unsigned_literal`, the per-constructor shape check of `parse_primary_expr` (`primary_levels`), and the height helpers `with_child_height` and `tree_height` (the iterative walk is not proved equal to `height`).
- **NOT COVERED:** the rest of the parser routines in `cypher.rs`, which are the grammar, not the lexer. They are covered by [`../parser_grammar`](../parser_grammar); `bash proofs/coverage.sh` counts each function once, in its best bucket.

## What the model does not cover

- The numeric scanner's *length* is abstract in `lexAll_total_cover` ("some whole-character prefix"); `FalkorLexerExtra.lean` proves `lex_numeric`'s slices safe separately. The integer *value* is covered by `str2int`.
- Floats (`str::parse::<f64>`) are not modelled.
- Keyword lookup (the `phf` table) is omitted; it does not affect lengths.
- The height model works on an abstract tree whose shape is all that matters. That a primary expression adds at most two levels (`PRIMARY_LEVELS`) is checked constructor by constructor in `primary_levels` against the tree shapes read off `parse_primary_expr`, not proved from the parsing code.
- `unesc_agrees_spec` is about the spec's lower-case escapes. The grammar also reads upper-case escapes such as `\N` as a newline; Rust keeps them verbatim, as C does (`unesc_upper_kept`). This is a known divergence from the grammar, not a bug against C.

## Build and check

```bash
cd proofs/lexer && lake build          # this project
bash proofs/lean_ci.sh lexer           # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `lexer.rs`, `string_escape.rs`, or the literal handling or depth guards in `cypher.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A fix for #2895 should turn `set_target_unbounded` and `foreach_unbounded` into `pre<PR>_` history and add theorems that the new guards bound both paths.
