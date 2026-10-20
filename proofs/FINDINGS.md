# Lean 4 verification of FalkorDB-rs: findings

Current state (2026-10-20): proofs cite `origin/main` @ 289c201fe.
There is one Lake project per area under `proofs/<area>/` (35 projects, including
`native_index` and `core` — the shared `FalkorCore` library (pilot: Value `V F`, float model
`F`, i64, orderings), imported by value_order, value_math, expr_semantics,
functions_str_list; the PR review projects `pr2845_review` and `pr2917_review` were
retired on 2026-10-15 after their PRs merged, their theorems folded into the owning
projects).
All projects pin Lean `leanprover/lean4:v4.35.0` (upgraded from v4.34.0 on 2026-10-15).
`bash proofs/lean_ci.sh` builds all of them and fails on any `sorry`, `admit`,
`native_decide` or an axiom that is not justified with a `-- AXIOM-OK:` comment.
Headline theorems depend only on `propext`, `Quot.sound` and `Classical.choice`.
Each project's root `.lean` file holds its report in the header: a table mapping
Lean definitions to Rust `file:line`, the theorems, the bugs and the gaps.
`bash proofs/coverage.sh` aggregates the per-function `COVERAGE.tsv` files: of
2602 non-FFI, non-test functions, 2565 are PROVEN, 36 CONDITIONAL (proven only
under an undischarged hypothesis about other Rust code, named in the row) and 1
NOT COVERED (a trait declaration); 1292 FFI functions are AXIOMATISED; 0 unlisted.
See `.claude/skills/proofs/SKILL.md` for how to work with them.

A bug is **confirmed** only if a repro against the real Rust shows it, and it is
compared against the C module where relevant. Each entry below gives the repro
query or test. The Rust repro tests (`graph/tests/lean_<area>.rs`) were run
locally and are not committed, because many `bug_*` tests fail on purpose. The
RediSearch-dependent repros are in `proofs/<area>/repro.py` and run against
live servers. Sections are kept in the order they were written; later "Merged"
and "Re-check" notes record which bugs main has since fixed.

## Status

| wave | areas | theorems | confirmed bugs |
| --- | --- | ---: | ---: |
| 1 | value_order, expr_semantics, lexer, id_space, effects_codec, id_list, cow_btree, versioned_matrix, endpoint_index, runtime_ds | 745 | 29 |
| 2 | binder, optimizer_scan, optimizer_rewrites, ops_aggregate, ops_apply, ops_traverse, pending_commit, functions_str_list, functions_temporal, effects_emit_apply, index_layer, aeneas_pilot | ~750 | ~65 |
| 3 | runtime_core, value_math, graph_persist, redis_layer, graphblas_wrappers, planner_build, algo_udf, search_concurrency, columnar, parser_grammar | ~870 | ~60 |

Coverage after **wave 9** (2026-10-11, honest buckets; Rust read from e8f8a3017): of 3773 non-test fns, **2452 PROVEN + 26 CONDITIONAL + 1295 AXIOMATISED (FFI), 0 unlisted**, i.e. 98.95% of non-FFI functions proven outright (it read 2478/2478 before the CONDITIONAL bucket existed).

**Wave 9 (2026-10-11): why the proofs missed the #2917 hash bug, and what changed.** CodeRabbit found that PR #2917 (`GroupKey::eq` → `Value::sort_cmp`) broke `HashMap<GroupKey,_>`'s contract at 203f2022c: VecF32/Point NaN components with different payloads, and `-0.0`/`0.0` components, were `sort_cmp`-equal but hashed apart (fixed in 14e341cf4). The proofs missed it for three reasons: (1) value_order's float model `F = nan | num Int` had no signed zero and one NaN, and `V` had VecF32/Point only as type tags; (2) ops_aggregate's `gkHash_consistent` took "Value's Hash agrees with Eq" as an undischarged hypothesis `hc` — which value_order itself refutes on main (`hash_not_consistent_with_eq`) — yet coverage counted `GroupKey::hash` as PROVEN; (3) #2917 was never modelled. Changes:
  - **CONDITIONAL bucket** (`coverage.sh`, `COVERAGE.tsv`): a theorem resting on an undischarged hypothesis about other Rust code is covered but not proven. An audit of the PROVEN rows' theorems (hypothesis structures and `∀`-premises about Rust fns; FFI specs excluded) found 46 such rows. **Discharged (19 rows, stay PROVEN):** graph_queries `IdSpaceContract` (16 graph.rs rows; `IdRust.rustIdOps_contract` models the id_space.rs mutators and proves the contract, `IdRustUse.*_rust` instantiate), pending_commit `CancelSpec` (`delete_pending_node`, `remove_pending_relationships_for_node`; `CancelRust.cancelN_spec`/`cancelR_spec`), functions_str_list `Pool.intern_eq` (`intern`; `rustPool`). **Re-bucketed CONDITIONAL (27 rows, 26 fns):** ops_aggregate `GroupKey::hash` (`hc`, false on main) and sort.rs `cmp_key`/`cmp`/`cmp` (`Lawful compare_value`, false at e8f8a3017; #2912's `sort_cmp` satisfies it); graph_queries `first_node_with_relationships` (`AdjExact`); versioned_matrix `count_in_rows` (`LayerCount`); effects_codec `Payload::records`, `Records::next` (`IdCodec`); expr_semantics `NeighborIter::new`/`new_reversed` (graph `G` laws), `eval_batch`, `try_eval_batch_property` (`BatchLaws`); optimizer_rewrites `is_constant_true`, `eliminate_true_filters` (`hconst`, false for `^`, #2902); runtime_core `get_node_attribute`, `get_node_attribute_no_delete_check`, `get_relationship_attribute`, `materialize_node_property`, `materialize_node_property_values`, `materialize_relationship_property(_values)`, `get_node_attrs`, `get_relationship_attrs` (Graph read-agreement `hc`/`hName`/`hbg`; `hdisj` fails under #2876); ops_apply `ApplyOp::next_batched`, `OptionalOp::next_batched`, `SemiApplyOp::next`, `OrApplyMultiplexerOp::next` (`Separable` sub-plan, false for ValueHashJoin).
  - **Extended float/value model** (value_order): `F` has NaN sign/payload, `-0.0`, fractions, ±inf; `V` has `vecf32`/`point` contents with `compare_vecf32`/the Point arm and their hash arms. New `SortCmp.lean` models #2912's `sort_cmp` and proves it a total preorder on every well-formed value (`SC.scMain_total`); on main `sort_cmp`-equal ⇒ hash-equal is **false** (`SC.pre2917_hash_not_consistent`, then named `main_hash_not_consistent`: Float `NaN`/`-NaN`, vector/point NaN payloads and signed zeros), and main's `Value::eq` also merges vector/point signed zeros that hash apart (`vec_zero_eq_but_hash_differs`).
  - **proofs/pr2917_review** (14e341cf4): `scPR_total`, `hashPR_consistent` (sort_cmp-equal ⇒ hash-equal for all values, nested lists/maps/vectors/points), `hc_pr`/`gk_hash_consistent` (ops_aggregate's `hc` discharged for the PR), `float_signed_zero_one_key`, `float_nans_one_key`, `vec_nans_one_key`, `point_nans_one_key`, `signed_zeros_apart`; history `pre14e341_vec_nan_hash_differs`, `pre14e341_vec_zero_eq_but_hash_differs`. Live (C master, main, PR builds): the PR matches C on Float/vector/point grouping. Retired 2026-10-15; now `value_order` `Main.scMain_total`, `Main.main_hash_consistent`, `Main.*_one_key`, `Main.signed_zeros_apart(_nested)`, `Main.pre14e341_*`, and `ops_aggregate` `Discharge.hc_main`/`gk_hash_consistent_main`/`gkEq_spec`.
  - **W9-hash-1 (new, confirmed live on main):** DISTINCT keeps `NaN` and `-NaN` apart. `UNWIND [0.0/0.0, -(0.0/0.0)] AS x RETURN count(DISTINCT x)` → Rust 2, C 1; same for `RETURN DISTINCT x`, `collect(DISTINCT x)`, and `[NaN]`, `{a:NaN}`, `vecf32([NaN])` keys. Cause: the Float hash arm writes `x.to_bits()` for NaN (`value.rs:836-847`) and DISTINCT dedups by hash only. Lean `SC.pre2917_float_nan_hash_differs` (then `main_float_nan_hash_differs`). PR #2917's Float NaN word (203f2022c) fixes it (PR build: 1). Point NaNs are unreachable in Rust: `point()` refuses a NaN coordinate (C accepts it).

Coverage after **wave 5** (2026-10-04; `lean_ci.sh`: 34 projects, 0 failing; Rust read from origin/main @ 3fec7d7c9). Of 3722 non-test fns in `graph/src` + `src`: **2427 PROVEN + 1295 AXIOMATISED (FFI), 0 MODELLED, 0 NOT COVERED, 0 unlisted**, i.e. 100% of non-FFI functions proven. 613 test/bench fns are excluded and counted separately.

Coverage after **wave 4** (`lean_ci.sh`: 34 projects, 0 failing): of 3807 fns, 2233 PROVEN, 1308 AXIOMATISED (FFI), 36 MODELLED, 126 NOT COVERED, 104 unlisted. That is **89.4% of non-FFI functions proven**, with 266 functions left.

Coverage after wave 3 (`coverage.sh`, all 33 projects; `lean_ci.sh`: 33 projects, 0 failing):

| scope | fns | PROVEN | AXIOMATISED | MODELLED | NOT COVERED | unlisted | covered | proven |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all of `graph/src` + `src` | 3805 | 689 | 1317 | 444 | 1079 | 276 | 64.4% | 18.1% |
| excluding FFI externs (AXIOMATISED) | 2488 | 689 | — | 444 | 1079 | 276 | 45.5% | 27.7% |

Most AXIOMATISED rows are bindgen `extern` declarations (GraphBLAS, LAGraph, RediSearch, Redis module API).

---

## Wave 1: issues and PRs (all PRs open and green, none merged)

| # | bug | issue | PR |
| --- | --- | --- | --- |
| 1 | ORDER BY panics: the sort comparator is not a total order (NaN, Int/Float near 2^53, maps) | #2891 | #2912 |
| 2 | Map `<>` with disjoint values, NaN grouping, `min`/`max` over NaN | #2913 | #2917 (stacked on #2912) |
| 3 | Client `GRAPH.EFFECT` with a huge node or edge id crashes (2^61) or hangs (`u64::MAX-1`) the server | #2892 | #2911 |
| 4 | The v3 encoder writes records its own decoder rejects | #2914 | #2916 |
| 5 | Pushing onto a decoded IdList reorders ids or trips an assert | #2919 | #2920 |
| 6 | The parser depth guard is bypassed by `SET`/`REMOVE` property chains and nested `FOREACH` (crash) | #2895 | #2900 |
| 7 | A block comment ends at any `/`; a trailing `/` is swallowed; an unterminated `/*` is accepted | #2901 | #2908 |
| 8 | Lexer: CRLF, `0e5`, non-canonical `-2^63`, an overflow hidden by a postfix, `\u+041` | #2909 | #2910 |
| 9 | `UNWIND range()` overflows near the i64 limits | #2894 | #2899 |
| 10 | `^` drops operand errors, so `WHERE x ^ true` has its filter removed; `'a' ^ 2` gives null | #2902 | #2904 |
| 11 | `sign()` of a Float returns a Float | #2906 | #2907 |
| 12 | Parallel edges collapse when the edge or path variable is read outside the traverse's ancestors | #2896 | #2918 |
| 13 | CowBTree `insert_batch` drops `(MAX,MAX)`; `BRANCH_MAX = 3` breaks `remove` | #2893 | #2897 |
| 14 | The `versioned_matrix::Iter` lifetime (needs a design choice) | #2898 | — |
| 15 | Unchecked contracts (`Cow`, `OrderSet`, `OrderMap`, `prepare_tiers`) | #2903 | #2905 (merged) |

Filed later, with no PR yet:

| issue | bug |
| --- | --- |
| #2922 | `cap_at_or_above_batch_size_is_ignored` fails on macOS arm64 because LLVM 21-23 miscompiles the `cap < BATCH_SIZE` guard |
| #2923 | An inline property map that references a variable from an earlier MATCH: "'r' not defined", or silently 0 rows for nodes |
| #2924 | A client `GRAPH.EFFECT` CREATE_EDGE with a non-live src/dst is applied, and the graph stays corrupt after an RDB reload |
| #2925 | `ORDER BY … LIMIT k` is about 15% slower than C (the top-k builder re-evaluates keys; the Project fast path is off) |

Not filed:
- The DISTINCT/MERGE hash-collision bug, because #2780 was closed as not planned.
- Hop counts truncated to u32, and the doubled-backtick escape, because C behaves the same.

---

## Wave 2: confirmed bugs (issues and PRs in progress)

IDs are `W2-<area>-<n>`. "C ✓" means C gives the correct answer, so the bug is Rust-only.

### Server crash / data damage
- **W2-temporal-1..3**: `date({year:2020, week:10000000000})`, `date({…quarter:2, dayOfQuarter:i64::MAX})` and `date('2020W1é')` panic at `functions/temporal.rs:83`, `:95` and `:248`. C returns errors. C ✓.
- **W2-strlist-1**: `list.sort` with NaN panics (`functions/list.rs:200`). Already fixed by PR #2912.
- **W2-agg-3**: `percentileDisc`/`percentileCont` with NaN panic (`aggregation.rs:615`, `:645`). Probably already fixed by PR #2912; to verify.
- **W2-pending-1**: #2876 is wider than filed. The id-keyed deleted-entity maps (`runtime.rs:159`, read by every accessor at `:1460-1790`) shadow any live entity that reuses the id, even when nothing can reach the deleted one. `MERGE` then creates a duplicate node, which is persistent damage. Reads of labels, properties, type and endpoints of relationships are wrong too. C ✓.
- **W2-scan-2**: an id seek on a graph that has never held a node returns a phantom node 0 (`graph.rs:1495` `max_node_id`, `node_by_id_seek.rs:67`). `SET` on it leaks into the next `CREATE`, and `DELETE` errors. C ✓.
- **W2-effects-1**: a write that only registers a name or only cancels entities ships no payload (`commit.rs:108`, `pending.rs:1688` `effects_count`). Examples: `CREATE (a:A {p:1}) DELETE a`, `OPTIONAL MATCH (n:Nope) SET n:L6`. The next payload is refused, which forces a full resync, and during AOF load the server exits. `CREATE (a)-[:T1 {p4:1}]->(b) DELETE a,b` makes the replica silently miss attribute `p4`. 134/200 random seeds diverge.
- **W2-effects-2**: UPDATE_NODE, UPDATE_EDGE, SET_LABELS and REMOVE_LABELS in `apply.rs:283,327,346,357` never check liveness. Sent through a client `GRAPH.EFFECT`, the next entity that reuses the id inherits the property or label.
- **W2-effects-3**: DELETE_NODE of a node that still has edges is accepted (`apply.rs:374`). The edge dangles and the next node inherits it. Related to #2924.
- **W2-effects-4**: a refused buffer's CREATE_INDEX is not rolled back, because the Indexer is shared by `Arc` across versions (`apply.rs:72`, `graph.rs:3214`).

### Wrong results: binder / parser (`proofs/binder`)
- **W2-binder-1**: the general cause of #2923. The parser folds `MATCH p MATCH q`, `CREATE … CREATE …` and `OPTIONAL MATCH p MATCH q` into one pattern (`cypher.rs:1540-1556`). The WHERE pattern-predicate loop (`cypher.rs:2102`) swallows the next MATCH. `bind_graph` binds all nodes before any relationship (`binder.rs:1322` vs `:1339`). Effects:
  - consecutive CREATEs can't see each other: "'a' not defined", C returns 1;
  - relationship uniqueness is enforced across separate MATCH clauses: 0 rows, C returns 1 (also seen in ops_traverse);
  - a mandatory MATCH after OPTIONAL MATCH becomes optional;
  - `(a)-[r]->(b {v: r.w-2})` is rejected.
- **W2-binder-2**: an inline property filter that references another component is placed on its own component's scan (`planner/mod.rs:1840`): [] vs C's 2 rows. This is the node case of #2923.
- **W2-binder-3**: deleted scope-table entries let ids be reused. WHERE/SKIP/LIMIT parent copies (`binder.rs:1231-1239`) and comprehension or pattern locals (`:1893`, `:2068`) are removed while their ids are still in use. The results are wrong rows, extra rows, and "Invalid node id for 'to'".
- **W2-binder-4**: UNION branches inside `CALL {}` start at scope 0, shared with the outer query (`binder.rs:868-876`).
- **W2-binder-5**: ORDER BY copies leak past WITH/RETURN (the snapshot at `binder.rs:1216`). This causes a spurious "already declared", and `RETURN *` also returns ORDER BY-only variables.
- **W2-binder-6**: labels from CREATE and from pattern comprehensions tighten an earlier MATCH scan (`binder.rs:1645`, `:1847-1904`): 0 or 1 rows where C returns 3.
- **W2-binder-7**: a pattern predicate in WITH … WHERE rebinds a parent variable as a fresh one (`define_name_in_scope`, `binder.rs:2121`): 3 rows where C returns 1.

### Wrong results: optimizer
- **W2-rewrites-1**: a filter merge evaluates the outer conjunct first (`push_filters_down.rs:122`). This gives "Division by zero" where C returns 15.
- **W2-rewrites-2 / W2-scan-***: id seek with a negative bound, `id(n) > -1`, gives 0 rows where C gives 5 (`runtime.rs:1319`); a non-integer bound (`id(n)=null`, `=1.0`, `>1.5`) raises an error where C gives 0 rows (`runtime.rs:1324`).
- **W2-rewrites-3**: the hash join drops Int/Float pairs near 2^53 (`value_hash_join.rs:153` compares keys exactly, while `=` goes through `as f64`).
- **W2-rewrites-4**: a filter inside an uncorrelated `CALL {}` is pushed onto the Argument, so values are swapped (`push_filters_down.rs:214-261`, `mod.rs:91-109`). This is a new way to hit #2557.
- **W2-scan-1**: `select_scan_node.rs:893-900` marks `best_node` as bound when no scan is built for it, so a filter runs before its variable is bound: 0 rows where C gives 1.
- **W2-scan-3**: a computed constant ≥ 2^52 in an index equality returns every node (`utilize_index.rs:910-928`).
- **W2-scan-4/5**: an `IN` with a computed left side is pushed as `n.a IN [...]`, and `2 IN [n.a, n.b]` becomes an array-contains query on `a` (`utilize_index.rs:630-655`).
- **W2-traverse-1**: an unreferenced named edge is collapsed even though a sibling traverse still reads it for uniqueness (`reduce_expand_into.rs:118-143`): 1 where C gives 2. Check against PR #2918.

### Wrong results: runtime operators
- **W2-apply-1**: a ValueHashJoin inside a batched Apply or Optional joins rows from different outer rows, because it is missing from the `can_batch` blacklists (`apply.rs:119-133`, `optional.rs:104-106`): 16 where C gives 8.
- **W2-apply-2**: `merge_pattern_cache` is shared by every MERGE clause (`runtime.rs:163`, `merge.rs:118-135`). This gives "Variable b not found" plus a rollback, and wrong UNION ALL results.
- **W2-apply-3**: MERGE results depend on BATCH_SIZE (`merge.rs:313-343`).
- **W2-apply-4**: the all-bound MERGE shortcut drops matches visible to a path variable (`merge.rs:368-400`): `count(p)` is 1 where C gives 2.
- **W2-apply-5** (low): batched Optional moves the NULL rows after the matched rows, which `collect()` exposes.
- **W2-traverse-2**: the bidirectional dedup (`cond_traverse.rs:273-299`, `947-968`, `1267`) ignores other columns, keys on an intermediate node at 3 or more hops, and resets per batch.
- **W2-traverse-3**: an undirected anonymous collapse gives one row per direction (`cond_traverse.rs:876-942`, `expand_into.rs:182-221`): 4 rows where C gives 2.
- **W2-traverse-4**: allShortestPaths on a directed cycle returns the path reversed (`all_shortest_paths.rs:284-288`). C is also wrong.
- **W2-traverse-5**: algo.SPpaths/SSpaths charge cost 0 per relationship without a `costProp` (`algo_procedures.rs:2308`, `2397`, `2518`). C charges 1.
- **W2-agg-1**: clearing the global dedupers after each CALL{} (`apply.rs:283`) makes an outer `count(DISTINCT)` over-count: 6 where C gives 2.
- **W2-agg-2**: per-group DISTINCT state is keyed by the FxHash of the group key (`aggregate.rs:715`, `:959`), so colliding groups share one seen-set.
- **W2-agg-4** (low): a per-row percentile argument is overwritten, so the last one wins (`aggregation.rs:539`, `:253`).
- **W2-agg-5**: an aggregate inside a list literal `return`s out of the whole stack machine (`eval.rs:877`). `[min(x),max(x)]` gives 2 where C gives `[1,2]`. **Fixed by #2951 (0912f4500; closes #2945)**: `evalStack_eq_evalRef`, `evalStack_list_of_aggs`, `main_min_max_list`; history `pre2951_stackEval_return_drops_frame`.

### Wrong results: index layer (`proofs/index_layer/repro.py`)
- **W2-index-1**: `n.v = date(..)` returns every node of the label (`utilize_index.rs:631` drops the filter and `node_by_index_scan.rs:252` falls back). C ✓.
- **W2-index-2**: `n.v = point(..)` returns [] (`index/mod.rs:1660`). C ✓.
- **W2-index-3**: `n.v IN [date(..), 2]` drops the date item (`node_by_index_scan.rs:222`). C ✓.
- **W2-index-4**: for multi-label `(n:A:B)`, conditions on B's indexed property are pushed into A's index (`utilize_index.rs:503`, `:521`; `mod.rs:1547`, `:1564`). C ✓.
- **W2-index-5**: `n.v > 0` returns dates, which are stored as numbers (`mod.rs:785`). C ✓.
- **W2-index-6**: `n.v > 'a' AND n.v < 'a'` returns `'a'`, because the equal-bounds shortcut ignores exclusivity (`mod.rs:1418`). C ✓.
- Shared with C:
  - the TAG encoding breaks string order for bytes ≤ 0x20, `\` and `_` (`mod.rs:565`);
  - Bool is indexed as 1/0;
  - an open bound becomes an exclusive ±inf, which misses a stored +inf;
  - `n.v = ''` returns [];
  - an undirected edge-index scan returns each edge once instead of twice;
  - a geo radius above ±85° returns nothing.

### Wrong results: functions
- **W2-strlist-2** (**fixed by #2952, 07e712944; closes #2926**): `list.remove([1,2,3],1,i64::MAX)` panicked on overflow in debug builds (pre-fix `list.rs:168`). Now `remove_span` saturates (`listRemove_eq_spec`; historical `pre2952_listRemove_end_overflows`).
- **W2-strlist-3**: `=~` matches substrings: `'abc' =~ 'b'` is true, where openCypher says false (`internal.rs:112`, `eval.rs:1274`).
- **W2-strlist-4**: `toLower`/`toUpper` reject valid strings that contain U+FFFD (`string.rs:148`, `:170`). C ✓. **Fixed by #2953 (e7eb853b1)**: `toLower_total`, `case_agrees_C_if_simple`; history `pre2953_toLower_rejects_valid`.
- **W2-strlist-5**: `string.replaceRegEx` expands `$` in the replacement (`string.rs:469`). C ✓. Fix: `NoExpand`.
- **W2-strlist-6**: a list comprehension over null gives [] (C: null); over a scalar it gives `[5]` (C: type error) (`eval.rs:1018`).
- **W2-temporal-4**: antipodal `distance()` gives NaN, because `(1-a).sqrt()` is taken with `a > 1` (`value.rs:137`). C ✓. **Fixed by #2983 (86dcd5a11; closes #2931)**: `a` clamped to [0,1]; `distance_finite` (under the stated `NanLaws` float facts); history `pre2983_distance_nan`.
- **W2-temporal-5**: `date('2020-03-31') - duration({months:1})` gives 02-29, while `+ duration({months:-1})` gives 03-02 (`value.rs:746`). C gives 03-02.
- **W2-temporal-6**: Time + duration is not reduced mod a day (`value.rs:1027`). C ✓. **Fixed by #2985 (289c201fe; closes #2933)**: `timeAddDur_eq_tod`, `time_normalised`, `add_time_dur_in_day`; history `pre2985_time_not_normalised`.
- **W2-temporal-7**: `toString(date({year:12345}))` gives "2345-01-01" (`value.rs:803`). C ✓.
- **W2-temporal-8**: `duration('P2635249153387078803W')` wraps to P5D in release (`temporal.rs:376`). **Fixed by #2987 (9d3869aa8; closes #2935)**: checked W/D arithmetic; `dateLoop_week_overflow`, `dateLoop_days_i64`; history `pre2987_week_string_wraps`.
- **W2-temporal-9**: the `date()` map form's dayOfWeek is 0-based with 0 = Sunday (`temporal.rs:75`). C accepts 7.
- Conversion divergences seen live, with no Rust test:
  - `toString(1.0)` gives "1" (C "1.000000");
  - `toInteger('1e3')` gives 1000 (C null);
  - `toInteger(' 12')` gives null (C 12);
  - `toFloat('0x10')` gives null (C 16);
  - `duration('P5')` gives PT0S (C null).

### Stats (write-statistics counters)
- **W2-pending-2**: entities created and deleted in the same segment vanish from the stats (`pending.rs:677`, counters at `:1141`, `:1147`, `:1226`).
- **W2-pending-3**: writes to a deleted relationship are applied and counted (`remove.rs`; `set.rs:230` misses a DETACH cascade).
- Policy question: Rust counts property changes net at commit, C counts per clause. Decide which to match.

### C-side bugs seen along the way (not Rust)
- The C master crashes on `OPTIONAL MATCH (n:Nope) SET n:L6` (Redis assert `blocked.c:346`).
- C SIGSEGVs in `RediSearch_ResultsIteratorNext` on `distance(..) <= 0` with a range index.
- C hangs on some `WHERE id(c) < 0` patterns.
- On macOS, C crashes on node DELETE in one agent's environment.
- In C, a MERGE inside CALL returns one row, and ON MATCH/ON CREATE SET values land on the wrong rows.

---

## Wave 3: confirmed bugs (issues and PRs in progress)

### Server crash / hang / data loss
- **W3-conc-1**: the server hangs permanently once more than about 1024 queries are queued. The main thread holds the GIL and does a blocking send into the full pool queue (`threadpool.rs:59,96`, `graph_core.rs:1052`). C finishes 3000/3000. Lean: `pool_deadlock_reachable`; the fix is proven in `fixed_never_stuck`.
- **W3-parser-1**: the shortestPath filter-skip loop never ends at EOF (`cypher.rs:1819-1845`). `RETURN shortestPath((a)-[{` spins a worker forever, and a handful of these block all graph queries.
- **W3-planner-1**: the previous MATCH's WHERE is dropped when the next MATCH traverses from its variable (stitching at `planner/mod.rs:2710`, pruned by `select_scan_node.rs:239`). `… DELETE b` then deletes data that C does not.
- **W3-gb-1**: a corrupt blob in GRAPH.RESTORE or an RDB panics the server via an `assert_eq!` on `GxB_Vector_deserialize` (`vector.rs:188`, `:219`; `tensor.rs:1586`). **Fixed by #2996 (d99861124)**: `decodeBlob_never_dies`, history `pre2996_decodeBlob_is_dos`.
- **W3-conc-4**: a huge `k` in `db.idx.vector.queryNodes` runs `Vec::with_capacity(k)` and SIGSEGVs (`graph.rs:3744`, `:3799`).
- **W3-algo-1**: forged `__falkor_type` entity markers in a UDF result are trusted (`type_convert.rs:229-236`). This panics the reply or creates a dangling edge.
- **W3-algo-2**: SHUTDOWN after a main-thread UDF aborts, because ThreadJsState drop order frees the QuickJS runtime first (`js_context.rs:129-136`).
- **W3-runtime-1/2**: the LIMIT/SKIP budget is pushed through hard-cap traversals (`runtime.rs:594-626`), so traverse→traverse under LIMIT, and stacked SKIPs, silently lose rows.
- **W3-parser-9**: exponential backtracking in `parse_list_literal_or_comprehension` (`cypher.rs:2796-2803`). A 206-byte query takes 49 s; C is worse.

### Wrong results
- **Planner** (`proofs/planner_build`):
  - a multi-component MATCH after `MATCH … WHERE` is cross-joined, not correlated;
  - MERGE with a named path as the last clause fails with "Variable _anon_0 not found";
  - a self-loop hop after other hops drops those hops.
- **Constraints**:
  - the UNIQUE key includes the value's type (`graph.rs:4141`, #2774);
  - a constraint created over violating data reports OPERATIONAL;
  - enforcement is quadratic (#2995).
- **Index / concurrency**:
  - a second index on a label is never populated (**fixed by #3086, 73077bb4a**: `main_operational_complete`; history `pre3086_second_index_never_populated`);
  - unconfirmed: between `Indexer::create_index` storing an entry and `populate_index` taking its ticket (`graph.rs:3598-3604`) the index reads OPERATIONAL with the new field unpopulated, so a concurrent reader could plan an index scan and miss rows (`store_window_operational_incomplete`); predates #3086, not seen live;
  - a vector of the wrong dimension drops the node from every index on its label;
  - a MULTI/EXEC write fails with "another write is in progress";
  - index-option validation differs from C.
- **GraphBLAS**:
  - `Tensor::decode` accepts an orphan `me` row (**fixed by #3033, 96375c238**: `decodeCheck_ok_iff`, `decodeCheck_accepts_encoded`; history `pre3033_orphan_row_panics`); unconfirmed follow-up: decode still accepts a multi-edge pair with exactly one id, or one edge id under two pairs (`edge_count` stays consistent; `PairPlan` assumes >= 2 ids);
  - `Matrix::decode` leaks 5 blocks per matrix;
  - concurrent Matrix drops leak (**fixed by #3090, e45be4965; closes #3089**: `arc_frees_once`, `arc_every_interleaving_frees_once`; history `pre3090_race_leaks`);
  - DUMP/COPY payloads may contain heap addresses (**fixed by #3097, cab41a8c5**: `encodeM_no_heap_addresses`). Unverified follow-up: `Matrix::decode` still copies the header's reserved `*_future` pointer arrays from the payload; safe only if GraphBLAS never reads or frees them.
- **Redis layer**:
  - writes ignore the per-query TIMEOUT (`graph_core.rs:911`);
  - loose TIMEOUT/version/arity parsing;
  - GRAPH.CONFIG accepts invalid values (**fixed by #3021, 30b4fb7dc; closes #3020**);
  - a GRAPH.BULK header can name a property twice (**fixed by #3063, 0f45a8685**: the first column wins; the parse itself keeps one entry per attribute since #3026, fb3d516d0: `keepFirst_nodup`, `keepFirst_find`);
  - several output-format differences;
  - likely root cause of #2539 (`attribute_store.rs:598` `as u16`) (**bulk side fixed by #3063, 0f45a8685**).
- **Parser**:
  - a slice bound that fails to parse is silently replaced by 0 or MAX;
  - `NOT NOT x` is parsed as `x`;
  - `SET [n).x` is accepted;
  - `1 IS NULL IN [...]` is rejected;
  - trailing commas are accepted in calls;
  - `MATCH MATCH` is accepted.
- **Algo / UDF**:
  - WCC and CDLP with nodeLabels return a compact index instead of a node id;
  - HarmonicCentrality sizes by node count instead of the id bound;
  - library-name collisions and case folding;
  - the validation context has no `graph` global;
  - -0.0 comes back as Int 0;
  - `getNeighbors` lists a self-loop twice;
  - a UDF map with a `constructor` key is taken for a Date;
  - a vecf32 holding inf is rejected;
  - unknown labels in betweenness/harmonic return nothing instead of an error.
- **Value / math**:
  - toInteger turns negatives below i64::MIN into MIN (#2955, PR #2989);
  - `coalesce()` accepts zero arguments (#2956, PR #2990).

### Latent
- `Batch::concat` binds a slot from an empty batch, and `to_owned_row` drops trailing Unbound columns (`batch.rs:990`, `:1403`).
- `vector::Iter` can outlive its vector (the #2898 class).

## Wave 4: confirmed bugs (documented only — no issues/PRs until the open PRs are handled)

- **W4-graph-1** (**fixed by #3277, 13e849935; closes #3244**; `pre3277_stale_type_after_registration`, now `type_cleared_after_registration`) — a deleted edge keeps its old type. `get_type_id_mut` (`graph.rs:1167`) adds a type without resizing `relationship_type_matrix`, so the removal in `delete_relationships` / `delete_implicit_edges` (`:2615`, `:2724`) silently does nothing in release (only a `debug_assert` catches the dimension mismatch). Repro: create an `A` edge, `GRAPH.CONSTRAINT CREATE g MANDATORY RELATIONSHIP B PROPERTIES 1 x`, delete the edge, create a `C` edge (reuses the id), `type(r)` → Rust `A`, C `C`. The replica's schema-apply (`effects/v3/apply.rs:536`) has the same path. Fix: `self.resize()` in `get_type_id_mut`. Test `lean_graph_queries::deleted_edge_type_survives_type_registration`.
- **W4-graph-2** — `get_node_relationships` (`graph.rs:2056`) reports a self-loop twice; seen through UDF `getNeighbors` (count 2 vs C 1). Probably the same as #3083 (PR #3084 fixes the UDF side) — check whether #3084 fixes it at this source.
- **W4-graph-3** — constraint matching depends on property order (`constraint.rs:79`): `UNIQUE NODE L (a,b)` then `(b,a)` creates both; C says "Constraint already exists".
- Minor: `CREATE INDEX` on a new label also reports "Labels added: 1" (C does not).
- **#2776, wider** — `SET e = <node|rel|map>` keeps attributes written earlier in the same query: only the Node←Map branch clears pending attrs (`set.rs:202-249`, `:279-376`). `CREATE (a {x:1}),(b {y:2}) SET a = b` → Rust `{x:1,y:2}`, C `{y:2}`; `SET r.w=5 SET r={y:2}` → Rust `{w:5,y:2}`, C `{y:2}`. Node←Node and Rel←Map are new shapes for the open #2776. Lean `SetRemove.noClear_keeps_pending`; tests in `lean_ops_apply.rs`.
- **W4-fn-1** — a deleted node loses labels staged earlier in the same query (violates the "deleted entity keeps all details" decision). The DELETE snapshot copies the committed `get_node_label_ids` (`ops/delete.rs:282`, `:461`) instead of the pending view. `MATCH (n:A) SET n:C DELETE n RETURN labels(n), n:C` → `['A']|false` (expected `['A','C']|true`); `REMOVE n:A … DELETE n` → `['A']` (expected `[]`). Fix: build the snapshot through `get_node_labels`. Lean `labels_lost_after_delete`; tests `lean_functions_str_list -- deleted`. Every other accessor (type, endpoints, properties, keys, id) is proven preserved after DELETE.
- Minor: `point({latitude:null, longitude:1})` gives a different error message from the same map passed as a value.
- **W4-plan-1** — `expr_to_plan` keeps null rows under `NOT` (`planner/mod.rs:1421-1427`): `NOT(complex)` is planned as AntiSemiApply, so rows where the inner predicate is null pass. `MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)` → Rust 2, C 1. Tests in `lean_planner_build.rs`.
- **W4-plan-2** — a subscript is never accepted as a boolean (`binder.rs:2304-2314`): `WHERE n.flags[0]` → "Expected boolean predicate", C 1. Same for `true AND l[0]` and `true AND -null`.
- **W4-plan-3** — ORDER BY aggregate matched to the wrong projection: `expr_ir_eq` (`binder.rs:2653`) falls back to discriminant equality, so all Int/Float/Null/Bool/String constants compare equal. `RETURN … max(n.v % 3) AS m ORDER BY max(n.v % 4.0)` sorts by `m`; C rejects the query. **PR #3011 widens this to non-aggregate ORDER BY, so fix `expr_ir_eq` before or with #3011.**
- **W4-plan-4 (crash)** — nothing after a procedure CALL is validated (`ast.rs:1166-1206`): `CALL db.labels() YIELD label MATCH (n)` panics the server at `utilize_node_by_id.rs:118` (`parent().unwrap()`); C rejects it. `… UNWIND` is accepted, and `… CREATE (a)-[:R|S]->(b)` creates an edge.
- **W4-plan-5** — a query may end with WITH (`ast.rs:1268`): `MATCH (n) WITH n` returns rows; C and openCypher reject it.
- Minor divergences: `false AND (1+2)` → false (C type error); `date({year:2020, month:null})` treats null as absent (C errors); `WITH r WHERE r.w = 1` on a var-length `r` is rejected (C 1).
- W4-plan-4 was independently confirmed by the parser agent: `inner_validate`'s CALL arm returns `Ok(())` (`ast.rs:1167-1201`). `CALL db.labels() YIELD label CREATE ()-[r]->() RETURN label` also crashes.
- **W4-parse-1** — `_anon_N` name collision (`cypher.rs:2924`): `MATCH (_anon_0:A), (:B) RETURN count(*)` → Rust 0, C 1.
- **W4-parse-2** — CREATE with named paths silently drops a duplicate relationship (`cypher.rs:1513-1521`): `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d)` creates 4 nodes and 1 relationship; C errors "Variable `r` already declared".
- **W4-parse-3** — MERGE drops a duplicate relationship (`cypher.rs:1534-1547`): `MERGE (a)-[r:R]->(b)-[r:R]->(c)` creates 3 nodes and 1 relationship; C errors.
- **W4-parse-4** (minor) — `SET (n.x) = 5` (`cypher.rs:3249-3263`) and `[:A:B]` (`:2966-2976`) are accepted; C gives a syntax error.
- **W4-parse-5 (crash)** — 2000 chained `UNWIND [1] AS xN` clauses crash the server in planning or execution (parse, validate and bind succeed up to 8000); C runs 5000.
- **#2769 re-confirmed (crash)** — `pending_deleted_*degree` (`pending.rs:1088`, `:1110`) asks the committed graph for the endpoints of an edge created and deleted in the same batch, which panics at `graph.rs:3009`. `CREATE (a)-[r:R]->(b) DELETE r SET b.d = indegree(b), a.o = outdegree(a) RETURN b.d, a.o` kills the server; C returns `0|0`. Lean `deg_panics_on_pending_deleted`.
- Lean-only, not confirmed live: `set_span` stores `n as u16` (`attribute_store.rs:597`, `:608`), so 65,536+ entries per entity wrap (GRAPH.BULK or a malformed RDB only; #3063 now filters the bulk side: `normResolved_fits`). A GRAPH.BULK header with a repeated column stored two entries for one id (fixed by #3063: `resolved_duplicate_fixed`, history `pre3063_resolved_duplicate`).
- **W4-redis-1 (crash)** — the slow log cuts the query at byte 2048 regardless of character boundaries (`src/slow_log.rs:49` `entry_hash`, and `truncate` at `:31`). A query over 2048 bytes that takes more than 10 ms, with a multi-byte character spanning byte 2048, kills the server; C answers normally. Lean `SlowLog.entry_hash_panics`.
- **W4-redis-2** (**fixed by #3252, cfe6e7945; closes #3239**; `pre3252_field_prefix_lost`, now `field_name_kept`) — an index on an attribute literally named `range:x` (or `vector:…`) comes back as an index on `x` after `DEBUG RELOAD` (`serializers/mod.rs:642-646`); C keeps `[range:x]`.
- **W4-redis-3** — module-load arguments differ from C (redismodule-rs: case-sensitive names, booleans compared to `== "yes"`). `CMD_INFO YES` → 0 (C 1); `CMD_INFO garbage` loads (C refuses); `cache_size 10` and `ASYNC_DELETE yes` are ignored.
- W4-redis-4 (minor) — GRAPH.BULK reply says "relations created"; C says "edges created".
- #2537, wider: any truncated GRAPH.RESTORE payload calls `LoadStringBuffer(NULL)`, not only an empty one (#2996 does not cover it: it changes no RDB string-buffer path, so this is still open).
- **W5-proc-1..3** (procedures, live vs C):
  - a lone `CALL db.idx.fulltext.createNodeIndex({label:'W'})` under GRAPH.QUERY fails with a misleading "graph.RO_QUERY is to be executed only on read-only queries" (`procedure_call.rs:67`);
  - `db.idx.fulltext.drop` on a missing index errors (C does nothing);
  - `db.idx.fulltext.queryNodes(1,2)` errors (C returns empty).
  - Output differences: `db.indexes` and `db.constraints` row order; `db.indexes` `info` holds only field names (C returns full RediSearch info and `_src_id`/`_dest_id` for edge indexes); `dbms.functions` union type order (`Point or Null` vs `Null or Point`).
- **W5-udf-1** — `graph.traverse([a],{returnType:'edges'})` returns a self-loop twice (`js_classes.rs:770`): Rust `[1,0,1]`, C `[1,0]`. PR #3084 may cover `collect_edges`; check.
- **W5-udf-2** — `graph.traverse([a])` leaves `a` out when it is its own neighbour, because the start node is pre-marked visited (`js_classes.rs:744`): Rust `[1]`, C `[0,1]`.
- **W5-udf-3** — `node.getNeighbors()` with no argument throws "Error converting from js 'undefined'" (`js_classes.rs:195`, `:352`); C returns the neighbours.
- **W5-udf-4** — `GRAPH.UDF LOAD ""` is accepted (`src/commands/udf.rs:117`), but its functions can never be called: the repository names them `.f` (`repository.rs:105`) while JS stores `f` (`js_globals.rs:125`). C rejects it with "empty lib name".
  Repros: a standalone Rust test against origin/main (not committed).
- **W5-rdb-1** — RDB-loaded graphs ignore `CACHE_SIZE`: `redis_type.rs:64` hard-codes `DEFAULT_CACHE_SIZE = 25` and passes it at `:94` and `:842`. With `CACHE_SIZE 1`, after `DEBUG RELOAD` a q1, q2, q1 sequence reports "Cached execution: 1" (C 0).
- #2537 refined: a GRAPH.RESTORE payload cut at a record boundary SIGSEGVs in `RM_LoadStringBuffer` (`decoder/mod.rs:426` → `buffered_io.rs:212`); a cut inside a record errors cleanly. C crashes on both.
- Theoretical: `uuid_v4` (`redis_type.rs:771`) uses `t ^ seq`, so virtual-key names can collide across calls (counters 2 and 3 at clocks 4 and 5 ns). A collision would lose a virtual key. Not reproduced. **Fixed by #3286 (bf11a2774)**: graph_persist `counter_field_bijective`, `uuid_counter_distinct`; history `pre3286_uuid_collision_example`.
- **W5-idx-2** (found while addressing review on PR #3087) — `CREATE VECTOR INDEX FOR (n:L) ON (n.emb)` with no OPTIONS returns an error to the client ("field type 0x0010 and the options block disagree about the vector half"), yet the index is still created, half-built. A node with a vecf32 then drops out of the other indexes on its label. #3087 now guards the drop; the half-created index on error is not fixed (PR #3094, which requires a dimension, may prevent it).
- **W6-opt-1** (found reviewing PR #2845, pre-existing on main 89d68334a) — filter push-down gives an OPTIONAL MATCH inside a `CALL {}` body the outer query's variable ids, so the filter on `n` is pushed onto a row where that slot is unbound (`push_filters_down.rs:217`, `:238`). `MATCH (x:A:B) CALL { OPTIONAL MATCH (n:A:B {id:3}) RETURN n.id AS nid } RETURN x.id, nid` → Rust "Variable n not found", C `[[1,3],[2,3],[3,3]]`; four variants fail the same way. Possibly related to #2728. Lean `pr2845_review: filter_pushed_through_boundary`. A trial fix (match `IR::Apply | IR::Optional(_)` and inherit `branch_output_variables(left)`) makes all five match C, and 166/166 related flow tests pass.
- **W4-index-1** — a range index on an attribute that already has a fulltext index is never queried correctly. `build_query_node` takes the attribute's first field whatever its type (`index/mod.rs:1386,1411,1459,1471,1529,1590`). After `CREATE FULLTEXT INDEX … ON (n.s)` then `CREATE INDEX … ON (n.s)`, `n.s='foo'`, `n.s > 'c'`, `{s:'bar'}` and `n.s IN [...]` all return `[]` (C returns the rows). Fix: pick the field whose type is `IndexType::Range`. Lean `IndexerM.bug_range_after_fulltext`.
- Unconfirmed: `create_rs_index` leaks index options on its NUL-byte error path.
- **W4-matrix-1** (latent, API-level only) — `Tensor::resize` shrinking (`tensor.rs:814-828`) never trims `me`. Orphan multi-edge rows survive: `iter_edges` yields phantom ids and `edge_count` is too high. After growing back and adding an edge, `get` and `iter_edges` disagree. Its only shrinking caller, `rebuild_derived_matrices`, covers every node id, so Cypher can't reach it. Lean `VMTensorOps.shrink_orphans_me`; test `lean_versioned_matrix::tensor_shrink_keeps_orphan_me_rows`. Fix: drop `me` rows whose pair falls outside the new bounds. **Fixed by #3250 (2eacf0ece; closes #3246)**: `Tensor::resize` never shrinks and the RDB load keeps the saved size; `resizeT_edges`, `rebuild_never_shrinks`; history `pre3250_shrink_orphans_me`.

### Gaps closed: open bugs the proofs had missed (2026-10-20)
A triage of every open `type:Bug label:rust` issue (live on Rust main 9d3869aa8 and C) found 56 the proofs had not flagged. The models were extended for the ones that are proof gaps (model too coarse, property not stated, code not modelled, FFI contract). Each now has a stated property a fix can prove and a `bug<issue>_*` counterexample on main that matches the live repro. All are Open. Root-cause families: slot ids reused across WITH/aggregation scopes (#3150 #3333 #3368 #3369 #3372 #3375), named paths over an already-bound node (#2311 #2725 #2727 #2737 #2743 #2744), rewrite criteria that compared rows but not writes (#748 #997), and per-row write visibility through CALL {} (#2729 #2738 #3164). The triage corrected two causes: #2502 is the parser (a WHERE pattern predicate swallows the next MATCH), and #2556 is select_scan_node seeding an unbound endpoint, not push-down.
- #748 (plan/optimizer): a constant-false WHERE after `CALL { MERGE … }` is pushed into both Apply branches by push_filters_down (the empty variable set routes everywhere), so the MERGE never runs: Rust 0 nodes, C 1, and the rows-only criterion cannot see it. Lean bug748_merge_skipped; property EffectSound.
- #997 (plan/optimizer): `CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE false` gets the false filter pushed onto the Apply's Argument, so the CREATE never runs: Rust 0 nodes, C 1, while the CASE-wrapped form creates 1. Lean bug997_create_skipped; property EffectSound.
- #1334 (binder): the Pattern arm of bind_expr_node mints unknown aliases fresh, so WHERE () -- (n1) with no n1 in scope is accepted (C errors). Lean bug1334_pattern_predicate_declares_var; property PatternPredDeclaresNothing.
- #2311 (plan): OPTIONAL MATCH p=(n) over a bound node: the bound node-only branch of plan_match (mod.rs:1915-1962) emits no PathBuilder, so p is null (Rust `1, null`, C `1, [(0)]`). Lean bug2311_optional_path_null; property BindsAllPaths.
- #2502 (parser): a WHERE pattern predicate is parsed by `parse_pattern(&Keyword::Match)` (cypher.rs:2119), whose separator continues on `MATCH` (1562-1579), so `MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` folds the second MATCH into the predicate (Rust `'m' not defined` / count 1, C rows / count 2). Lean bug2502_pred_swallows_match (+ bug2502_where_match_merged); property PredEndsBeforeClause.
- #2556 (plan/select_scan_node): with the bottom hop's outer child kept (UNWIND after an aggregating WITH), `initial_bound` seeds the best endpoint `t` (select_scan_node.rs:979-990), so the inline filter `t {id:100}` is placed after the first hop, before `t` is bound; Rust 0 / C 3 (the issue's own query: a "was List" type error); not push_filters_down. Lean bug2556_filter_below_binding; property KeepsFiltersBound.
- #2653 (store): a non-UTF-8 graph key is replicated and signalled through from_utf8_lossy (`\xc3(bin` → `\xef\xbf\xbd(bin`), so the replica writes a different key. Lean bug2653_nonutf8_key_mangled; property ReplicatedKeyIsStored.
- #2724 (fn): prev() is the only user-callable C built-in (of 104) missing from the Rust registry. Lean bug2724_prev_missing; property CoversC.
- #2725 (plan): `MATCH (n0) CALL … YIELD … MATCH (n0), p0=(n0)` is planned `Project(Argument(ProcedureCall(Scan)))` and gives one empty row, so `Variable n0 not found` (C answers). Lean bug2725_call_then_bound_path_loses_var; property Plan.argIsLeaf, BindsAllPaths.
- #2727 (plan): `WITH n0 AS alias4 MATCH (alias4), p1=(alias4)` returns 0 rows (C `'x'`); same two root causes as #2744. Lean bug2727_alias_path_drops_rows; property BindsAllPaths, Plan.argIsLeaf.
- #2729 (rt/create+delete): streaming CREATE→DELETE depends on the batch split, so the endpoint check rejects a node an earlier batch pending-deleted; the CALL {} form splits the rows differently, and top-level `UNWIND range(1,1100) … CREATE (n)-[:R]->(n) DETACH DELETE n` also fails while C succeeds. Lean RowWrites.bug2729_batch_split_flips_endpoint_check; property RowWrites.PartitionInvariant.
- #2737 (plan): `MATCH (n1:X) OPTIONAL MATCH p6=(n1) RETURN p6` gives null (C `[(0)]`); same root as #2311 (nodeOnly_bound_no_paths). Lean bug2737_optional_path_null; property BindsAllPaths.
- #2738 (rt/apply): a writing CALL subquery is batched (can_batch admits Create/Delete/Set/Commit), so later outer rows do not see earlier rows' writes; Rust count 3 / 4 nodes vs C 7 / 8. Lean RowWrites.bug2738_batched_call_hides_writes; property RowWrites.WriteSeparable.
- #2743 (plan): `MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)` gives 0 (C 1): no PathBuilder, and all_visited (mod.rs:3067) counts p0, so there is no Apply wrap and the earlier clauses land under an Argument that never runs them. Lean bug2743_bound_path_drops_rows; property BindsAllPaths, Plan.argIsLeaf.
- #2744 (plan): `MATCH (n0:A) WITH n0 MATCH p=(n0)` is planned `Project(ExpandInto(Argument(Project(Scan))))` and returns 0 rows (C 1); planMatchArm_path_unwrapped. Lean bug2744_bound_path_unfed_argument; property Plan.argIsLeaf, BindsAllPaths.
- #2781 (idx): rolling back a failed multi-Commit query that reused an edge id with new endpoints re-adds the committed key but never deletes the private-endpoint document, so an orphan (c,d,0) stays in the edge index. Lean IndexLayer.Iter.bug2781_rollback_not_restored; property RollbackRestoresEdgeIndex.
- #2782 (srv): GRAPH_REGISTRY is keyed by name, not (db, name), so CREATE `shared` in db0, db1, db0 → the third write aborts with "deleted or replaced" (C ok ×3). Lean bug2782_same_name_other_db_aborts; property KeyspaceRegistered.
- #3121 (udf): SPpaths' Dijkstra fast path (algo_procedures.rs:2308) charges 0 for a configured costProp the edge lacks; C charges 1 (pathCost 2 vs 3). Lean bug3121_fastpath_missing_cost_zero; property MissingCostIsOne.
- #3150 (aggregate/binder ids): the accumulator of count(1) shares id 0 with the null key x; unbinding it drops the null column and reply.rs:655 unwrap panics ('a' survives as value-only). Lean bug3150_null_key_unbound; property KeysBound.
- #3164 (rt/apply+id_space): per-row commits release a deleted node's id, a later CREATE reclaims it, and later outer rows' stale `Value::Node(id)` delete the new node; Rust deletes 2 vs C 1. Lean RowWrites.bug3164_stale_ref_deletes_recycled_id; property RowWrites.NoStaleAlias.
- #3166 (idx): CREATE FULLTEXT INDEX OPTIONS {stopwords} on a label that already has a range index records the stopwords but leaves the existing RediSearch spec on its default list (create_index only calls register_fields), so 'router' still matches (Rust 1 row, C 0). Lean IndexLayer.Meta.Ixr.bug3166_stopwords_not_applied; property StopwordsAgree.
- #3226 (idx): a write that commits while its own index scan is open (scan below LIMIT) re-takes the RediSearch spec lock as a writer while holding it as a reader, and the server aborts at spec.c:2304; the same path takes the GIL while holding RediSearch's global lock (stall with RESTORE). Lean SC.Locks.bug3226_open_scan_upgrade; property noReacquire.
- #3333 (aggregate/binder ids): the pre-projection alias of m (id 2) stays bound on the Aggregate output and the next MATCH gives b id 2, so b starts bound to m and the query returns []. Lean bug3333_alias_binds_next_var; property EmitsOnlyScope.
- #3349 (udf): WCC, pageRank and betweenness on a one-node nodeLabels projection fail with -1000: GraphBLAS stores the 1×1 compact matrix by column, LAGraph_CheckGraph rejects it, and Rust returns the error. Lean bug3349_one_node_projection_fails; property ProjectionTotal.
- #3352 (plan/optimizer): Case 2 push-down inherits bare ids from the whole pre-WITH subtree (get_variables walks through Aggregate), takes the pre-WITH id 2 for `r`, and puts `r.status = 'revoked'` on the Argument: Rust e=null, C 123. Lean bug3352_slot_collision_drops_match (bug3352_routes_unbound); property RowSound (RoutesBound).
- #3368 (rt): MERGE hashes a value-only leftover at its variable's slot (`get_by_id` ignores the bound bit, merge.rs:189), so two MERGEs with equal leftover counts share a `merge_pattern_cache` entry and `c` is bound to the earlier `B b1` node instead of creating `C c1`. Lean bug3368_merge_takes_leftover_as_node; property KeyFromBoundNodeOnly (aggregate side: bug3368_count_acc_in_merge_slot, property EmitsOnlyScope).
- #3369 (rt): the same cache hit re-inserts the cached pair `(2, b1)` into slot 2 (`r2`) and leaves `c` (id 3) as an unbound `Int 1`, so `c.id` fails with "Type mismatch … but was Integer". Lean bug3369_merge_leaves_int_in_node_var; property MergeBindsOwnNode (aggregate side: bug3369_count_acc_in_merge_slot, property EmitsOnlyScope).
- #3372 (rt/aggregate→traverse): Aggregate re-binds a grouping key at its slot from the previous scope (aggregate.rs:1187-1196), and the binder restarts slot ids at 0 in each scope (binder.rs:2261). A fresh variable in the next MATCH that gets the same slot is read by CondTraverse as already bound to that node (cond_traverse.rs:648-652,771-777), so `WITH c, count(*) AS k MATCH (c)-[:E]->(d)` returns 0 instead of 4. Lean bug3372_slot_reuse_drops_rows; property ScopedRow.
- #3375 (aggregate/binder ids): count(x)'s output id equals sum(x)'s accumulator id and is written into `combined` first, so sum returns the count (3 vs 6) and avg panics in finalize_avg. Lean bug3375_output_overwrites_next_acc, bug3375_avg_panics; property OutputsOwnAcc.
- #3380 (udf): algo.betweenness defaults to 16 sources (C 32), never randomises without a seed (C uses the clock), and drops colliding seeded draws (samplingSize 20, seed 7 on 40 nodes → 15 sources). Lean bug3380_default_size_16, bug3380_seedless_deterministic, bug3380_seeded_fewer_sources; property DefaultSizeMatchesC, SeedlessRunVaries, SourcesExact.
- #3382 (idx): db.idx.vector.queryNodes answers from a half-built index (0 rows, then 3, while UNDER CONSTRUCTION); Indexer::vector_query never checks is_operational, while C refuses until OPERATIONAL. Lean IndexLayer.Meta.Ixr.bug3382_vector_query_ignores_operational; property VectorQueryRequiresOperational.

### Merged
- 2026-10-20 on main (now 289c201fe): #2983 (antipodal distance; closes #2931), #2987 (duration string overflow; closes #2935), #3294 (AND/OR/XOR errors name the operand type; closes #3292), #2951 (list literal of aggregates; closes #2945), #2953 (toLower/toUpper with U+FFFD), #2985 (localtime ± duration; closes #2933). Proofs re-targeted: 35 projects green; 2595 PROVEN + 25 CONDITIONAL of 2620 non-FFI, non-test fns, so every non-FFI function is covered; 0 unlisted, 0 unmatched. About 270 rows that matched only by name were re-pinned to their exact line.
  - Fixed, counterexamples kept as `pre<PR>_` history: W2-agg-5, W2-strlist-4, W2-temporal-4 (the distance proof assumes the `NanLaws` float facts as hypotheses), W2-temporal-6, W2-temporal-8. #3294 is modelled (`and_or_mismatch_names_type`, `xor_mismatch_names_type`).
- 2026-10-19 on main (now fb3d516d0): #3090 (a shared GrB_Matrix is freed exactly once; closes #3089), #3026 (GRAPH.BULK keeps one entry per attribute; closes #3024). Proofs re-targeted: 35 projects green; 2594 PROVEN + 25 CONDITIONAL of 2620 non-FFI, non-test fns (99.0%); 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: concurrent-drop leak (`Own.lean`: for any number of owners and any interleaving of clones and drops the matrix is freed exactly once, when the count reaches zero, `arc_frees_once`; `Iter` frees its iterator before the matrix; std's `Arc` contract is assumed, not proven); repeated GRAPH.BULK header column (`PropColumns`: at most one entry per attribute, first non-null wins, column order kept, equal to the plain read when no name repeats; `pre3026_two_entries_for_one_attr`).
- 2026-10-19 on main (now 73077bb4a): #3033 (tensor decode refuses sections that break promotion completeness; closes #3031), #3086 (a field added while an index populates gets populated; closes #3069). Proofs re-targeted: 35 projects green; 2590 PROVEN + 25 CONDITIONAL of 2616 non-FFI, non-test fns (99.0%); 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: orphan `me` row on decode (`decodeCheck_refuses_incomplete`; every encoded tensor still decodes, `decodeCheck_accepts_encoded`), search_concurrency Bug 2 (#3069) lost population of a second index (`main_operational_complete`, by an invariant over every interleaving of creates and batches).
  - New PROVEN: `Matrix::count_value_eq` (`countValueEq_spec`, under the `GrB_select` VALUEEQ contract), `is_ticket_current` (was MODELLED).
- 2026-10-19 on main (now 1d15bf713): #3267 (nested list claims bounded; closes #3266), #3298 (repeated CREATE_NODE id refused; closes #3247), #3250 (tensors never shrink; closes #3246), #3278 (no u32-overflowing compressed frame; closes #3276), #3283 (remove_batch never leaves a one-child branch; closes #3237), #3097 (matrix headers without heap addresses), #2996 (corrupt RESTORE/RDB payloads refused, not panicked on), #3286 (virtual-key names cannot collide; closes #3284), #2905 (Cow/OrderSet/OrderMap/prepare_tiers contracts; closes #2903). Proofs re-targeted: 35 projects green; 2589 PROVEN + 25 CONDITIONAL of 2615 non-FFI, non-test fns (99.0% PROVEN); 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: W7-btree-1 (`removeBatchTree_wf`), W4-matrix-1 (`resizeT_edges`), W3-gb-1 decode DoS (`decodeBlob_never_dies`), heap addresses in payloads (`encodeM_no_heap_addresses`), quadratic effects reservation (`capTrace_linear`), u32 frame length (`pre3278_wrapped_frame_unreadable`), repeated CREATE_NODE id, uuid `t ^ seq` collision (`uuid_counter_distinct`), deleted-id count overflow on decode (`countCheck_in_bounds`), `OrderSet` duplicates (`fromVec_nodup`), `prepare_tiers` underflow (`clampStarts_mono`).
  - **Discharged: the 13 native_index CONDITIONAL rows are PROVEN.** cow_btree now proves `insert_batch`, `from_sorted` and `remove_batch` keep `TreeWF` (`insertBatch_wf`, `fromSorted_wf`, `removeBatchTree_wf`), so every tree the API can build is well-formed (`Reachable_wf_asserted`), and native_index `Contract.lean` shows each Rust tree operation equals its model on reachable trees.
  - Still present: the `Matrix::decode` component leak (#3080), `vector::Iter` lifetime (#2898), #2537 (string-buffer path), release `Cow` one-directional isolation and `OrderMap::from_unique_keys` (debug-only on purpose).
- 2026-10-17 on main (now 0f45a8685): #3252 (index on an attribute named `range:x` survives reload; closes #3239), #3335 (a parenthesised WHERE uses the index and NodeByIdSeek again; closes #3187, #3331), #3334 (UNIQUE enforced by index lookup; closes #3330), #3277 (deleted edge loses its type after a type registration; closes #3244), #3063 (bulk spans never overflow `Slot::len`; closes #2539, whose CREATE-error half, rejecting the query with C's error, remains open in substance). Proofs re-targeted: 36 projects green; 2569 PROVEN + 38 CONDITIONAL of 2608 non-FFI, non-test fns; 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: W4-graph-1 (`pre3277_*`; `getTypeIdMut_spec`, `type_cleared_after_registration`); W4-redis-2 (`pre3252_*`; `decField_encField` round-trips every name); bulk span overflow and repeated column (`pre3063_*`; `normResolved_fits`, `normResolved_get`, `resolved_duplicate_fixed`).
  - #3335: Paren is gone from bound trees. New PROVEN `ExprIR::and_of`/`or_of`/`flat` (`tv_andOf`, `scL_splice`, `flat_passes`); binder `bind_paren` (`(x)` binds exactly like `x`), `bind_andOr_paren`; planner `toPlan_correct` holds unchanged; columnar `eval_sound` holds without its Paren arm.
  - #3334 (`UniqueIndex.lean`): `uniqueViolated_spec`, and `uniqueViolated_agrees` (equals the old label-scan verdict). `check_node_constraint`/`check_edge_constraint` are now CONDITIONAL on index completeness (the index returns every live member not written by this commit). Worth a flow test: a value escaped by `tag_encode_lower` (space, `_`, `\`) committed earlier, then a duplicate CREATE must still fail.
  - To check live: the bulk caller drops `Null` values before dedup, so header `p,q,p` with values `(null,2,3)` keeps the second `p` (3); C's answer is unknown.
- 2026-10-15 on main (now cab3cf192): #2897 (CowBTree BRANCH_MAX >= 4; closes #2893), #3285 (DOC_BYTES limited to readable widths; closes #3238), #3336 (UDF value nesting cap; security advisory GHSA-frmf-ffvc-mxx5). Proofs re-targeted: 37 projects green; 2565 PROVEN + 36 CONDITIONAL of 2602 non-FFI, non-test fns; 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: W1 #13 bug 2 (`constAssert_rejects_B3`, `remove_spec_asserted`); W7-btree-2 (`readWidth_admitted`, `admitted_fromPairs_roundtrip`); UDF conversion recursion (`fromJs_frames`/`toJs_frames`: at most cap+1 levels for any input, cyclic values included; `fromJs_eq_pre3336`: unchanged at or below the cap; `pre3336_self_array_overflows`).
  - W7-btree-1 was still present here; fixed later by #3283 (see 2026-10-19).
  - New, model only, not confirmed live (`cap_asymmetry_vecf32`): the two UDF conversion directions count a VecF32 or Path leaf differently, so 1024 lists around a vecf32 pass `value_to_js` but are refused by `js_to_value` on the way back; an identity UDF returns a nesting error for that value (an error, not a crash).
- 2026-10-14 on main (now b080a44c1): #3014 (bounded queues; closes #3013), #3290 (inline write vs queued write; closes #3093), #3102 (self-loop hop order; closes #2999), #3328 (module-load arguments like C; closes #3268), #3339 (value nesting cap on RDB/RESTORE decode, property writes and bulk insert; security advisory GHSA-mwvh-8fq5-x2j8, also #2722 for stored values). Proofs re-targeted: 37 projects green; 2559 PROVEN + 36 CONDITIONAL of 2596 non-FFI, non-test fns; 0 unlisted, 0 unmatched.
  - Fixed, counterexamples kept as `pre<PR>_` history: W3-conc-1 deadlock (`acquire_bounded`, `release_exactly_once`, `main_never_blocks`); #3093 (`one_commit_per_version`, `claimFork_on_committed`); W4-redis-3 / #3268 (`rust_c_same_accept`, `error_applies_nothing`; checked live against C); planner bug 4 / #2999 (`selfLoops_perm`, `deferred_after_binder`, `eq_reference` to the old loop); unbounded nesting (`decodeValue_refuses_past_cap`, `stored_roundtrip`, bulk `deep_refused`).
  - New divergence, confirmed live: C's `strtoll` skips leading whitespace and Rust's integer parsing does not, so `--loadmodule … CACHE_SIZE " 10"` and `GRAPH.CONFIG SET RESULTSET_SIZE " 5"` succeed on C and are refused by Rust (`leading_space_divergence`). Intended extensions: Rust accepts `INDEX_WORKER_THREADS`, `BOLT_PORT`, `EFFECTS_COMPRESSION`, which C refuses (`rust_only_names`).
  - #3339 detail: the decoder accepts one level more than writes (MAX_VALUE_NESTING arrays around `[]`, `within_nest_empty`); harmless, since recursion stays bounded and no write can store such a value.
- 2026-10-12 on main (now 4a4f2a72f): #2350 (native numeric index), #3307 (native-index writes once per unit of work), #2912 (sort_cmp total order; closes #2891), #2917 (map <>, NaN grouping, min/max over NaN, vector/point keys; closes #2913), #2910 (lexer literals; closes #2909), #3104 (pattern-comprehension backtracking; closes #3103), #3000 (closes #2998), #3234 (closes #3233), #3262 (closes #3251), #3113 (closes #3109), plus #3181/#3192 (build/CI). Proofs re-targeted: 37 projects green; 2536 PROVEN + 36 CONDITIONAL of 2572 non-FFI, non-test fns; 0 unlisted, 0 unmatched.
  - New project `native_index` (#2350/#3307): key encoding preserves and reflects order and is injective except -0.0/+0.0 (`encodeF64_cmp`, `encodeF64_inj`); an index scan returns exactly what a label scan plus filter returns (`range_agrees`, `point_agrees`, `nodeScan_native_agrees`); the batched apply equals applying changes one by one (`applyColumn_eq_sequential`); updates and deletes remove the old entry; reader snapshots are isolated (`snapshot_isolated`). 13 rows are CONDITIONAL on CowBTree's tree invariant, which `remove_batch` still breaks (W7-btree-1, PR #3283). **Since #3307 `remove_batch` has a production caller** (`apply_writes` → `remove_encoded`), so W7-btree-1 is no longer latent; no content/range failure was found over batch-only sequences, and the panic still needs a single-key `remove`, which only tests call.
  - Discharged: ops_aggregate `GroupKey::hash` and sort.rs `cmp_key`/`cmp`/`cmp` are PROVEN again on main's `sort_cmp` (`Discharge.hc_main`, `lawful_main`, ...); value_order's `cmp_key` row no longer counts a refutation as PROVEN; pr2917_review rows are live (merged code is byte-identical to 14e341cf4).
  - Fixed, counterexamples kept as `pre<PR>_` history: W9-hash-1, W2-agg-3, W2-strlist-1, map `<>`, min/max over NaN (#2912/#2917); #2909 CRLF, `0e5`, non-canonical -2^63, postfix overflow, `\u+041` (#2910, confirmed live); W3-parser-9 exponential parse (#3104, confirmed live); W4-redis-1 slow-log crash (#3234); #2998 write TIMEOUT (#3000); W5-rdb-1 (#3262); W2-traverse-5 (#3113; the single-path shortcut still charges 0, as C does). Renamed: `sortCmp_not_antisymmetric`/`_not_transitive`, `floatLane_eq_cmpV`, `listSort_cmp_not_antisymmetric`, `partialCmpEq_not_transitive` → `pre2912_*`; `map_neq_both_false`, `min_nan_picks_larger`, `maxScalar_ne_batch_nan`, `SC.main_*_hash_differs` → `pre2917_*`.
  - #3104's gate is exact on balanced brackets (`pipeScan_exact`) and failed attempts are linear (`failed_attempts_linear`), but "the gate never changes the parse" holds only under two premises (`listLitP_eq_ungated`). One is false today because of the slice-bound bug: `MATCH (a) RETURN [(a)-->(b) WHERE [1][(1..2] | b]` (invalid, unclosed `(`) was wrongly accepted before #3104 and is now a syntax error, as in C. The other (`MemoSound`: the memo key omits `anon_counter`) is unproven; no input was found where it matters.
  - New, confirmed live, shared with C: an indexed `n.v > i64::MIN` drops matching rows because `int_loses_f64_precision` lets i64::MIN through while i64::MIN+1..+512 round to the same f64 (`i64min_hole`); C drops the same rows, and for `=` Rust is right where C returns extra rows. Suggested fix: reject when `unsigned_abs() >= 1<<52`. Bool and number share an index key (`bool_divergence`), pinned as C-compatible.
  - Still present: W2-effects-4 (PR #3235), #2537 (external PR #3167), the Fx DISTINCT collision (#2780, not planned), `Value::eq` NaN/2^53 quirks (now harmless: grouping and sorting use `sort_cmp`).
  - coverage.sh: `#[cfg(all(test, ...))]` modules count as test code.
- 2026-10-11 (nothing merged): honest-coverage pass after the #2917 review miss — CONDITIONAL bucket, extended float/vector/point model, proofs/pr2917_review, new bug W9-hash-1. See Status → "Wave 9". 36 projects green; 2452 PROVEN + 26 CONDITIONAL of 2478 non-FFI fns.
- 2026-10-08 on main (now e8f8a3017): #2907 (closes #2906), #2908 (closes #2901), #2920 (closes #2919), #3022 (closes #2924, #2978), #3081 (closes #2963 in part, #2100), plus #2829 (CREATE named paths), #3125 (compact date/time strings reject non-digits), #3127, #3132 (startNode/endNode accept null), #2489 (tests), #3179/#3182 (CI/deps). Proofs re-targeted: 35 projects green, 2478/2478 non-FFI, non-test fns PROVEN, 0 unlisted, 0 unmatched.
  - Fixed, with the counterexamples turned into correctness theorems (`pre<PR>_` history kept):
    - #2906 `sign(float)` returns an Integer (`sign_float_is_int`);
    - #2901 block comments, a lone `/` and an unterminated `/*` (`rs_agrees_spec`; re-checked live);
    - #2919 pushing onto a decoded IdList (`decoded_then_pushed_keeps_order`);
    - #2924, W2-effects-2 and W2-effects-3 GRAPH.EFFECT liveness (`apply_wf`, `firstNodeWithRels_spec`, `validate_ok_iff`);
    - from the index shared-with-C list, open bounds now include ±inf and undirected edge-index scans emit both orientations (`numRange_exact`, `orient_length`; re-checked live);
    - `CREATE p=(…) RETURN p` returns the path (#2829).
  - **Regression on main from #2829** (W8-plan-1): a CREATE that declares a named path and comes after another clause never runs. `MATCH (a:A) CREATE p=(a)-[:R]->(:B)` and `UNWIND [1,2] AS x CREATE p=(:N {x:x})` give "Variable _anon_0 not found" and create nothing; the build before #2829 and C both create them. The planner's first walk does not step over PathBuilder, the same shape as planner_build bug 1. Open PR #3101 (adds PathBuilder to that walk) should fix both. Lean `create_path_last_misplaced`, `create_path_last_skips_create`.
  - W4-parse-2 has a new shape after #2829: `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d) RETURN p,q` returns an inconsistent path for `q` (edge 0 joins (0)→(1)); C rejects the query.
  - Still present:
    - W2-effects-1 and W2-effects-4;
    - W2-temporal-1..9, including `date('2020W1é')`: week strings are parsed before the #3125 digit check;
    - the deleted-node labels bug (W4-fn-1);
    - the #2909 lexer bugs;
    - planner_build bug 1;
    - index_layer bugs 1-4, 7, 8, 10, 12 and 13.
- 2026-10-07 on main (now 8743953a8): #3072 (closes #2961, W2-index-6), #3074 (closes #3073), #3076 (closes #2959, W2-index-5), #3079 (closes #3077), #2390 (plans an inline property map once; adds `optimizer/references.rs`), #2278 (CowBTree point lookups, memory accounting, tuple cursor, narrowable doc width), plus #2487 and #2488 (tests only). Proofs re-targeted: 35 projects green, 2466/2466 non-FFI, non-test fns PROVEN, 0 unlisted, 0 unmatched.
  - Fixed and re-checked live: W2-index-5 (#3076), W2-index-6 (#3072); the UDF -0.0, constructor-key and vecf32-inf round trips (#3074); the `graph` global at LOAD (#3079); W1 #13 bug 1, `insert_batch` dropping `(MAX,MAX)` (#2278); and from #2390, the per-operator part of #2896 (UNWIND/CREATE/FOREACH/var-length reads) and the earlier `MATCH … WHERE` dropped under a traversal (#2972). W2-rewrites-4 now matches C (most likely fixed by #2845). W4-plan-4 no longer crashes on current builds, but the `parent().unwrap()` at `utilize_node_by_id.rs:106` is still there.
  - Still present: the sibling part of #2896 (CALL body, pattern comprehension), W2-traverse-1/2/3, W2-binder-2 / #2923, W6-opt-1, W4-plan-1/2/3/5, planner_build bugs 1, 2 and 4, W2-scan-1, W1 #13 bug 2 (BRANCH_MAX=3), W3-algo-1/2, W5-udf-1..4. The W3-conc-1 cite is now `graph_core.rs:1002`.
  - New bugs:
    - **W7-btree-1** (latent; CowBTree has no production caller): `remove_batch` leaves a one-child non-root branch at any BRANCH_MAX ≥ 4; one later `remove` then makes `insert_batch` panic in `Node::min`. Lean `BugsBatch.lean`. **Fixed by #3283 (799968a8b)**: `removeBatchTree_wf`; history `pre3283_rb_*`.
    - **W7-btree-2** (latent): DOC_BYTES ∈ {3,5,6,7} passes the const assert, but `read_width` handles only 1/2/4/8, so `CowBTree::<256,256,3>` panics on its second insert. Lean `LeafAosD.docBytes3_*`.
    - **W7-scan-1**: `select_var_len_scan_node`'s leaf rewrite prunes a Filter wrapper: `MATCH (a) WHERE a.v=1 MATCH (a)-[*1..2]->(b:B {w:2})` returns an extra row. Predates #2390. Lean `ScanTree.vl_leaf_*`.
- 2026-10-06 on main (now a9377c636): #2952 (closes #2926, W2-strlist-2), #3021 (closes #3020: GRAPH.CONFIG validation, arity, ASCII name folding, `ASYNC_DELETE` default 1, C's unknown-field text), #2845 (every `CALL {}` body gets an entry projection as a record boundary; closes #2601, #2602), plus #3170 and #2471 (build/tests only). Proofs re-targeted: 35 projects green, 2449/2449 non-FFI fns PROVEN, 0 unlisted. Counterexamples turned into correctness theorems with `pre<PR>_` historical notes: functions_str_list `listRemove_eq_spec` / `removeSpan_wf` (`pre2952_listRemove_end_overflows`); redis_layer `fixed3021_counterexamples`, `vkey_ok_nonneg`, `jsheap_ok_ge`, `cmdinfo_ok_iff`, `get_extra_wrong_arity`, `dotless_i_not_folded` (`pre3021_*`); pr2845_review now models main (`pre2845_old_aliases`, `pre2845_bodyOld_empty`, `pre2845_emit_origin_lost_old`), and columnar proves `rows_only` / `has_origins` and the new `start_batch` arm. Still open: W6-opt-1 (push-down through the CALL boundary) and the UNION-branch scope alias (#3027).
- 2026-10-05 on main (now da6f808c3): #2989 (closes #2955), #2990 (closes #2956), #3010 (closes #3009), #3087 (closes #3075), #3088 (closes #3085). Proofs re-targeted: 35 projects green, 2446/2446 non-FFI fns PROVEN. The #2955, #2956 and #3009 counterexamples are now correctness theorems (`toInteger_neg_overflow_null`, `coalesce_zero_args_rejected`, `rust_c_agree` / `four_commands_share` for the shared flag parser). Still open: write queries ignore the per-query TIMEOUT (#2998, PR #3000).
- 2026-10-04/05 on main (now fe619ac5f): #2911 (closes #2892), #3053 (closes #3052), #3060 (closes #3057), #3094 (closes #3091), plus #3161 and #3163 (not from this work).
- Proofs re-targeted to fe619ac5f (34 projects green; 2444 PROVEN + 1295 AXIOMATISED of 3739, 0 unlisted). Counterexamples for #2892, #3052, #3057 and #3091 are now correctness theorems.
- Fixed and re-checked live on fe619ac5f: #2892 (model only), #3052, #3057, #3091. W5-idx-2 is fixed by #3094: no half-created index.
- Still diverging after #3094:
  - `OPTIONS {dimension:0, similarityFunction:'bogus'}` creates the index (C refuses), and with `dimension:0` a node with a 3-dim vector drops out of its label's range index (= W3 conc-3; PR #3087 guards the drop);
  - fulltext `OPTIONS {weight:1.0, foo:true}` is refused (C creates it; C only refuses a non-empty map with no known key);
  - `SAMPLES` error text has an `ERR ` prefix (C does not).
- Still open: `SET (n.x) = 5` is accepted (W4-parse-4).
- 2026-10-05: #3087 merged (main 49f698d22), fixing W3-conc-3 / #3075; re-checked live for dimension mismatch, dimension 0 and no options. W5-idx-2 is fully fixed (#3094 plus the #3087 guard). Proofs re-targeted: `index/mod.rs` 94/94 PROVEN, 2444 PROVEN of 3739, 34 projects green. The counterexamples are now theorems: `docSet_vector_sound`, `other_fields_never_dropped`, `setVector_sound`. Still differs from C (DDL only): `{dimension:0, similarityFunction:'bogus'}` is accepted (C refuses), and it no longer drops nodes.
- #2916 (v3 encoder refuses records its own reader rejects; closes #2914) merged to main as 2c874022a (2026-10-04). The effects_codec counterexamples are now correctness theorems (`encRecord_ok_shape`, `readRecord_encRecord_wf`).

### Re-check against main 2c874022a (2026-10-04, live Rust release build)
- Still present: #2892 (fix in PR #2911, merge conflict resolved), #2924, W2-effects-1, W2-effects-2, W2-effects-3, W2-effects-4.
- Fixed on main by #2846 (157d42ec1): a duplicate edge id inside one CREATE_EDGE record is now refused. A duplicate node id inside one CREATE_NODE (`[0,0]`) was silently de-duplicated until #3298 (2bb51088b, #3247) made it refused (`createNode_repeat_refused`).
- id_space is re-proven for the #2846 design (21/21 fns PROVEN, invariant `Inv` preserved by create/cancel/release).
- Also still present on 2c874022a: #2876/W2-pending-1, W2-pending-2, W2-pending-3, W2-scan-2, W4-graph-1, W4-graph-2, W4-graph-3, W4-fn-1, #2769 (panic now at `graph.rs:3138`), #2776, W5-rdb-1 (#3161 did not touch the cache size), #2537.
- Fixed by #3161: an RDB saved by Rust across several virtual keys now loads in C with the graph listed in GRAPH.LIST.
- All projects now cite main 2c874022a: 34 projects, 0 failing; 2443/2443 non-FFI non-test fns PROVEN (3738 non-test fns, 1295 AXIOMATISED, 630 test). Gap: graph_queries and pending_commit state the IdSpace behaviour they rely on as a hypothesis (`IdSpaceContract`, `CancelSpec`). It is proven in proofs/id_space, but the standalone projects are not linked, so the hypothesis is not yet discharged by importing those theorems.

### Decisions made
- 2026-09-25 (Avi): a deleted entity keeps returning **all** its details (labels, properties, type, endpoints) for the rest of the query. Rust's current behaviour stays even where C returns `[]` for labels. PRs must not strip them; #3043 already leaves this alone. Open gap: labels staged by SET/REMOVE in the same query before the DELETE are not reflected in the deleted-node snapshot (pending_commit suspicion).
- Traversal fixes filed as #3106–#3109, with PRs #3110–#3113. All PRs from this work carry the `6.0` label.

### Decisions needed (Rust follows the grammar or TCK, C differs)
- `2^3^2`, chained comparisons, `+` against STARTS WITH / IS NULL, `SKIP (1)`.
- Integer division through double vs exact.
- toFloat through float32 in C.
- toString(float) format (#2937).
- replaceRegEx `$` expansion (#2929).
- Net vs per-clause property stats.
- UNIQUE constraints with `true` = 1 (C) vs openCypher.

---

## Wave 1: details

### Server crash / hang
1. **ORDER BY panics**, with "comparison function does not correctly implement a total order", once there are 21 or more rows. NaN compares Less both ways (`runtime/value.rs:1801`, `runtime/ops/sort.rs:348`); Int vs Float via `i as f64` is not transitive near 2^53; `compare_map` (`value.rs:1472`) returns Equal early. Trigger: a read-only `MATCH (n:N) RETURN n.v ORDER BY n.v` over stored NaNs. C returns the rows. → #2891 / PR #2912.
2. **A client `GRAPH.EFFECT` with node id 2^61 crashes the server**, and `u64::MAX-1` hangs it (`id_space.rs:292`, `graph.rs:1474`, `:734`). → #2892 / PR #2911.
3. **`SET`/`REMOVE n.a.a.a…` bypasses `check_depth`, and nested FOREACH has no depth guard.** → #2895 / PR #2900.
4. **`UNWIND range(MAX-1, MAX)` overflows** in `RangeIter::next` (`runtime/eval.rs:120`). → #2894 / PR #2899.

### Wrong results (C is correct)
5. **DISTINCT, count(DISTINCT) and MERGE drop rows on a 64-bit FxHash collision.** Closed as not planned (#2780).
6. **Parallel edges collapse when the edge variable is read outside the traverse's ancestors.** → #2896 / PR #2918.
7. **`^` drops operand errors**, and `WHERE x ^ true` has its filter removed; `'a' ^ 2` gives null. → #2902 / PR #2904.
8. **Lexer:** block comments (→ #2901 / PR #2908); CRLF, `0e5`, `-2^63` spellings, an overflow hidden by a postfix, `\u+041` (→ #2909 / PR #2910).
9. **Map `<>`, NaN grouping and min/max over NaN.** → #2913 / PR #2917.
10. **`sign(float)` returns a Float.** → #2906 / PR #2907.

### Latent (no current production caller)
11. **v3 encoder shape checks.** → #2914 / PR #2916.
12. **`IdList::from_segments`.** → #2919 / PR #2920.
13. **CowBTree.** → #2893 / PR #2897.
14. **The versioned-matrix iterator lifetime.** → #2898 (design needed).
15. **One-directional copy-on-write and unchecked contracts.** → #2903 / PR #2905 (merged 1d15bf713: `OrderSet` dedups in every build, `prepare_tiers` clamps, `Cow` asserts in debug across clones; release `Cow` and `OrderMap::from_unique_keys` stay unchecked on purpose).

### Unconfirmed / worth a look
- ~~A hostile nested `GRAPH.EFFECT` payload makes `Value::decode` reserve memory quadratic in the input.~~ Fixed by #3267 (eaa65eeba): `capTrace_linear`, `decode_reserves_linear`; history `pre3267_hostile_quadratic`.
- A duplicate roaring high key in an IdList bitmap: roaring-rs keeps the last partition, and CRoaring may keep the first.
- A duplicate edge id inside one `CreateEdge` record is only caught by the final count check; a duplicate node id is refused since #3298.
- ~~Compressed effect payloads of 4 GiB or more have their length truncated to u32.~~ Fixed by #3278 (dcbd445d1): they ship uncompressed; history `pre3278_wrapped_frame_unreadable`.
- ~~`Tensor::resize` never trims `me` on shrink.~~ Fixed by #3250: it never shrinks.

## Not covered / modelling limits
- f64 arithmetic is abstract (no IEEE reasoning).
- GraphBLAS, RediSearch, the Redis module API and QuickJS are hypotheses or axioms at the FFI boundary.
- `Arc`, concurrency and HashMap iteration order are not modelled.
- Aeneas extraction works only for small pure modules (`narrow_int::width_for` was proven on extracted code); larger modules are blocked by iterators, closures, `dyn`, `roaring` and `smallvec`. The pilot (`proofs/aeneas_pilot`) was removed on 2026-10-17: it added no coverage (`width_for` is PROVEN in id_list and runtime_ds), needed Charon/Aeneas on Lean v4.31, and committed generated Lean. Its last version is in 26cf24871.
