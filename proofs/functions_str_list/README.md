# functions_str_list: string, list and entity functions never panic and match the spec

This project proves that FalkorDB's Cypher string and list functions are *total*: no `i64` argument can make them panic or overflow. It also proves that they return what the openCypher spec says and, where both have an answer, what C FalkorDB returns. It covers the list parts of expression evaluation too (indexing, slicing, `IN`, quantifiers, comprehensions, `reduce`), the `=~` regex match, and the graph-entity functions (`labels`, `type`, `properties`, `startNode`, …), including what they return for an entity deleted earlier in the same query.

User-facing behaviour that depends on it:

- `l[i]`, `l[a..b]`, `x IN l`;
- `all`, `any`, `none` and `single`;
- `[x IN l | …]` and `reduce(…)`;
- `size`, `head`, `last`, `tail`, `reverse`, `range`;
- `list.remove`, `list.insert`, `list.sort`, `list.dedup`;
- `substring`, `left`, `right`, `trim`, `split`, `replace`, `toLower`/`toUpper`, `string.join`;
- `=~`, `string.replaceRegEx` and `string.matchRegEx`;
- `STARTS WITH`, `ENDS WITH`, `CONTAINS`;
- `labels(n)`, `hasLabels`, `type(r)`, `keys`, `properties`, `startNode`/`endNode`, `indegree`/`outdegree`.

The Rust is in `graph/src/runtime/functions/{string,list,internal,entity}.rs`, the eager `range` in `math.rs`, and the list, regex and quantifier arms of `graph/src/runtime/eval.rs`. The C side is `src/arithmetic/{string,list}_funcs/*.c` on `master`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorStrList.lean`](FalkorStrList.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can a list index or slice panic or overflow?**

No:

- `l[i]` is total for every `i64`, including `i64::MIN`. `len + i` can't overflow (`getElement_noOverflow`), and the result is the spec: count from the front for `i ≥ 0`, from the back for `i < 0`, and `null` outside `[-len, len)` (`getElement_spec`).
- `l[a..b]` never panics: after the early return, `0 ≤ start ≤ end ≤ len` (`getElements_bounds`). It equals the clamped spec (`getElements_spec`).

C truncates these bounds to `int32`, so `[1,2,3][4294967296]` returns `1` in C (`cGetElement_truncates`).

**2. Do the list functions overflow on huge arguments?**

Not on main:

- `list.remove(l, idx, count)` equals the spec for every `i64` index and count, in debug and release (`listRemove_eq_spec`). Before #2952, `list.remove([1,2,3], 1, 9223372036854775807)` panicked in debug (`pre2952_listRemove_end_overflows`).
- `list.insert` normalisation can't overflow (`insertNorm_safe`).
- Every element of the eager `range(a, b, step)` lies between start and end, so it never overflows (`range_elems_between`, `range_noOverflow`).

The lazy `UNWIND range(10, 0, -9223372036854775808)` still panics in debug (`rangeIter_negate_overflows`), a variant of #2894; see `expr_semantics`.

**3. Are the quantifiers and `IN` three-valued logic?**

Yes:

- `all` is a Kleene AND fold and `any` a Kleene OR fold (`all_is_kand`, `any_is_kor`);
- `none` is `NOT any` (`none_is_not_any`), and `single` is characterised by `single_spec`;
- `x IN l` is `false` exactly when no element compares equal and none involves `null` (`contains_false_iff`). So `1 IN [2, null]` is `null`.

**4. Do the string functions agree with C?**

On every input, for these:

| Function | Theorem |
|---|---|
| `substring` | `substring3_agrees_C` |
| `left`, `right` | `left_agrees_C`, `right_agrees_C` |
| `reverse` | `reverse_agrees_C`: C's bytes are the UTF-8 of Rust's `chars().rev()` |
| `trim` | `trim_agrees_C`: Rust's `ltrim∘rtrim` equals C's `rtrim∘ltrim` |
| `split` | `splitAux_eq_cSplit` for every non-empty delimiter, and `split_agrees_C` for the empty-string branches |

`string.join` precomputes its length. `joinLen_correct` proves that the length is right, so its `debug_assert_eq!` can't fire.

Case mapping is a deliberate difference. Rust uses full Unicode mapping, and C uses simple mapping: `toUpper('ß')` is `'SS'` in Rust and `'ẞ'` in C. Neo4j matches Rust.

**5. What does a deleted entity return later in the same query?**

The team decided on 2026-09-25 that a deleted entity keeps returning *all* its details for the rest of the query. `deleteRel_preserves`, `deleteNode_preserves` and `deleteNode_preserves_all` prove this holds for type, endpoints, properties, keys and id. For node labels it holds only when the query didn't change the labels first (`deleteNode_labels_iff`), which is bug W4-fn-1 below.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `getElement_spec`, `getElement_noOverflow` | `l[i]` is total for every `i64` and matches the spec | `FalkorStrList.lean` |
| `getElements_bounds`, `getElements_spec` | `l[a..b]` never panics and is the clamped slice | `FalkorStrList.lean` |
| `listRemove_eq_spec` | `list.remove` equals the spec for all `i64` inputs, debug and release (#2952) | `FalkorStrList.lean` |
| `range_noOverflow` | The eager `range()` never overflows | `FalkorStrList.lean` |
| `all_is_kand`, `any_is_kor`, `none_is_not_any`, `single_spec` | The list quantifiers are three-valued logic | `FalkorStrList.lean` |
| `contains_false_iff` | When `IN` returns `false` rather than `null` | `FalkorStrList.lean` |
| `substring3_agrees_C`, `left_agrees_C`, `right_agrees_C`, `trim_agrees_C`, `reverse_agrees_C`, `split_agrees_C` | These string functions equal C on every input | `FalkorStrList.lean` |
| `joinLen_correct` | `string.join`'s precomputed length is exact | `FalkorStrList.lean` |
| `listSort_float_cmp_antisymmetric`, `listSort_perm` | `list.sort`'s float comparison is antisymmetric (#2912), and the sort returns a permutation | `FalkorStrList.lean`, `Misc.lean` |
| `deleteNode_preserves_all`, `deleteNode_labels_iff` | Deleted entities keep their details; labels only when unchanged in the query | `Entity.lean` |
| `startEndNode_null`, `startEndNode_total` | `startNode(null)`/`endNode(null)` is `null` (#3132) | `Entity.lean` |

### How `splitAux_eq_cSplit` is proved

1. **Model the type.** A string is `Str`, a list of code points, which is what `.chars()` yields. Byte length is `blen`, so the byte-length guards can be stated.
2. **Model the two functions.** `splitAux` is the contract of Rust's `str::split`: scan left to right and cut at each match. `cSplit` is C's `AR_SPLIT`: find the first occurrence with `findFirst`, emit the prefix, and recurse on the rest.
3. **Unfold one step.** `splitAux_step` shows that one step of the Rust scanner either reaches the end with no match or cuts at exactly the position `findFirst` returns. With that, the proof is an induction on the length of the remaining string.
4. **Hypothesis.** The delimiter must be non-empty (`hd : d ≠ []`). The empty-delimiter case takes a different branch in both implementations, and `split_agrees_C` covers it separately.

This proof relies on a gap stated in the header. C matches bytes with `strstr`, while the model matches code points. For valid UTF-8 the two coincide, because UTF-8 is self-synchronising, but only one direction is proven (`prefix_chars_to_bytes`).

## Bugs found

Each bug has a Rust repro test (`graph/tests/lean_functions_str_list.rs`, run locally and not committed) and was compared live against C. See [`../FINDINGS.md`](../FINDINGS.md).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| A (W2-strlist-1) | `pre2912_listSort_cmp_not_antisymmetric` | `list.sort` over a list with NaN panicked and killed the server | Fixed by #2912 (`listSort_float_cmp_antisymmetric`) |
| B (W2-strlist-2) | `pre2952_listRemove_end_overflows` | `list.remove([1,2,3], 1, i64::MAX)` panicked in debug; release returned `[1]` by accident (`listRemoveRelease_eq_spec`) | Fixed by #2952 (closes #2926; `listRemove_eq_spec`) |
| C (W2-strlist-3) | `regexMatches_unanchored` | `=~` matches substrings: `'abc' =~ 'b'` is `true`; openCypher requires a whole-string match. C rejects `=~` | **Open** |
| D (W2-strlist-4) | `toLower_rejects_valid` | `toLower('A�')` errors with "Invalid UTF8 string"; a Rust `String` is always valid UTF-8, so the check can only misfire. C returns `'a�'` | **Open** |
| E (W2-strlist-5) | `replaceRegex_dollar` | `string.replaceRegEx('abc', 'b', '$0x')` returns `'ac'` because `$` is expanded; C copies the replacement literally (`'a$0xc'`) | **Open**: listed under "decisions needed" (#2929) |
| F (W2-strlist-6) | `comprehension_diverges` | `[x IN null \| x]` returns `[]` (C: `null`); `[x IN 5 \| x]` returns `[5]` (C: type error) | **Open** |
| W4-fn-1 | `labels_lost_after_delete`, `deleteNode_labels_iff` | `MATCH (n:A) SET n:C DELETE n RETURN labels(n)` returns `['A']` instead of `['A','C']`: the DELETE snapshot copies the *committed* labels (`ops/delete.rs:282`, `:461`), not the pending ones | **Open** |
| — | `rangeIter_negate_overflows` | `UNWIND range(10, 0, i64::MIN)` panics in debug ("attempt to negate with overflow"); release is correct | Unconfirmed variant of #2894 |

C bugs recorded for comparison:

- C truncates indices to int32 (`cGetElement_truncates`, `cGetElements_truncates`);
- C's `replace(s, '', t)` inserts inside multibyte characters and produces invalid UTF-8 (`cReplace_empty_breaks_utf8`);
- `range(0, 5000000000)` crashed the C server.

This project has no CONDITIONAL rows. The string pool contract (`intern`) is discharged by `intern_id_rust`.

## Files

| File | What's in it |
|---|---|
| `FalkorStrList.lean` | The report header; indexing and slicing, `list.remove`/`insert`, `head`/`last`/`tail`/`size`/`reverse`, `range`, quantifiers, `IN`, comprehension and `reduce`, the string functions against C, regex match and `$` expansion, `string.join`; bugs A–F |
| `Entity.lean` | `functions/entity.rs` over the runtime accessor model (deleted snapshot > pending > committed): deleted-entity theorems, `hasLabels`, degree arguments, `startNode`/`endNode`; W4-fn-1 |
| `Misc.lean` | The remaining `string.rs`, `list.rs` and `internal.rs` functions: `STARTS WITH`/`ENDS WITH`/`CONTAINS`, `isNull`, string interning, `toStringList`, `matchRegEx`, `list.sort`, `list.dedup`, registration |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Shared foundations (`InI64`, `wrap`, `I64MIN`/`I64MAX`) come from [`../core`](../core) through a Lake path dependency.

## Coverage

`COVERAGE.tsv` has 70 rows across `string.rs`, `list.rs`, `internal.rs`, `entity.rs`, `math.rs`, `value.rs` and `eval.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 65 |
| MODELLED | 4 |
| NOT COVERED | 1 |

- **MODELLED:** these `eval.rs` functions are modelled to state a bug, not proven correct, because the bugs are still open:
  - `RangeIter::next` (`rangeIter_negate_overflows`);
  - the `ListComprehension` arm and `eval_iter_expr` (bug F);
  - `eval_compiled_regex` (bugs C and E).
- **NOT COVERED:** `check_string_or_null`, a bare type check.

Some rows give the file as a bare name (`eval.rs`, `list.rs`) and others as the full path (`graph/src/runtime/functions/list.rs`). `bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Strings.** Strings are lists of code points. The only UTF-8 byte fact used is that every code point takes at least one byte. That C's byte-wise matching coincides with code-point matching is assumed, not proven (one direction is `prefix_chars_to_bytes`).
- **External code.** std `str::split`, `str::replace` and `step_by`, the `regex` crate and the Unicode case tables are *specified*, not verified. The regex engine is an abstract span predicate, so the Rust `regex` versus C oniguruma dialect differences aren't modelled.
- **Comparison.** `compare_value` is abstract here; see [`../value_order`](../value_order).
- **Graph storage.** `keys`, `labels` and `properties` are proven over an accessor model, not over the real attribute storage.
- **`string.join` lengths.** These are modelled in code points; Rust uses bytes, and the identity is the same.

## Build and check

```bash
cd proofs/functions_str_list && lake build    # this project
bash proofs/lean_ci.sh functions_str_list     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                       # repo-wide coverage report
```

**When you change `functions/string.rs`, `list.rs`, `internal.rs`, `entity.rs`, or the list/regex arms of `eval.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

When you fix one of the open bugs (C–F, W4-fn-1), rename its counterexample to `pre<PR>_…`, add the correctness theorem, and move the row to PROVEN. Any change to `ops/delete.rs`'s snapshot also affects `Entity.lean`.
