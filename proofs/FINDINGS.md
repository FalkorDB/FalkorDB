# Lean 4 verification of FalkorDB-rs: findings

Fixed bugs and the merge history were removed on 2026-11-16; see the git history of this file before 7f2d99c85 (`git show 5261abd4f:proofs/FINDINGS.md`). What is left is the open ledger: bugs that are not fixed on `main`, the decisions, the open gaps and the modelling limits.

## Current state

- Proofs last re-targeted to `origin/main` @ 43936bab9. This branch is based on `origin/main` @ 2e35cc1e1 (43 commits later, not yet re-targeted); the sha a re-target diffs from is 43936bab9. Each project's header carries its own revision pin where a section was last checked against an older one; the `COVERAGE.tsv` rows are the authoritative per-function list.
- 35 Lake projects, one per area under `proofs/<area>/` (including `native_index` and `core`, the shared `FalkorCore` library imported by value_order, value_math, expr_semantics, functions_str_list). All pin Lean `leanprover/lean4:v4.36.0`.
- Coverage (`bash proofs/coverage.sh`, 2026-11-16): of 2940 non-FFI, non-test functions, **2727 PROVEN, 25 CONDITIONAL, 2752 PROVEN + CONDITIONAL**, 2 MODELLED, 186 NOT COVERED, 0 unlisted, 0 unmatched. 1292 FFI functions are AXIOMATISED. CONDITIONAL means proven only under an undischarged hypothesis about other Rust code, named in the row. 1303 test/bench functions are excluded.
- `bash proofs/lean_ci.sh` builds all projects and fails on any `sorry`, `admit`, `native_decide` or an axiom that is not justified with a `-- AXIOM-OK:` comment. Headline theorems depend only on `propext`, `Quot.sound` and `Classical.choice`.
- Each project's root `.lean` file holds its report in the header: a table mapping Lean definitions to Rust `file:line`, the theorems, the bugs and the gaps. See `.claude/skills/proofs/SKILL.md` for how to work with them.
- The workflow `.github/workflows/lean-proofs.yml` (check name `Lean proofs`) runs on every pull request to `main` or a release branch (`N.M`) and on every push to `main`, with no `paths:` filter so that it can be a required check; inside the job a PR that touches none of `proofs/**`, `graph/src/**`, `src/**` or the workflow file reports success at once with a summary line saying it was skipped, and every other run executes both gates (`lean_ci.sh`, `coverage.sh --unmatched`). It is a separate check, not part of `ci-gate`.

A bug is **confirmed** only if a repro against the real Rust shows it, and it is compared against the C module where relevant. Each entry below gives the repro query or test. The Rust repro tests (`graph/tests/lean_<area>.rs`) were run locally and are not committed, because many `bug_*` tests fail on purpose. The RediSearch-dependent repros are in `proofs/<area>/repro.py` and run against live servers. Entries without a PR or issue number have no tracker yet. Sections keep the order they were written (waves); IDs such as `W2-binder-3` are the stable names.

---

## Filed issues and PRs still open (from wave 1)

| bug | issue | PR |
| --- | --- | --- |
| Parallel edges collapse when the edge or path variable is read outside the traverse's ancestors (the per-operator part was fixed by #2390; the CALL body and pattern comprehension remain) | #2896 | #2918 |
| The `versioned_matrix::Iter` lifetime (needs a design choice) | #2898 | none |

Filed later:

| issue | bug |
| --- | --- |
| #2923 | An inline property map that references a variable from an earlier MATCH: "'r' not defined", or silently 0 rows for nodes |
| #2925 | `ORDER BY … LIMIT k` is about 15% slower than C (the top-k builder re-evaluates keys; the Project fast path is off) |

Not filed:
- Hop counts truncated to u32, and the doubled-backtick escape, because C behaves the same.

---

## Wave 2: open bugs

IDs are `W2-<area>-<n>`. "C ✓" means C gives the correct answer, so the bug is Rust-only.

### Server crash / data damage
- **W2-pending-1**: #2876 is wider than filed. The id-keyed deleted-entity maps (`runtime.rs:159`, read by every accessor at `:1460-1790`) shadow any live entity that reuses the id, even when nothing can reach the deleted one. `MERGE` then creates a duplicate node, which is persistent damage. Reads of labels, properties, type and endpoints of relationships are wrong too. C ✓.
- **W2-effects-1**: a write that only registers a name or only cancels entities ships no payload (`commit.rs:108`, `pending.rs:1688` `effects_count`). Examples: `CREATE (a:A {p:1}) DELETE a`, `OPTIONAL MATCH (n:Nope) SET n:L6`. The next payload is refused, which forces a full resync, and during AOF load the server exits. `CREATE (a)-[:T1 {p4:1}]->(b) DELETE a,b` makes the replica silently miss attribute `p4`. 134/200 random seeds diverge.
- **W2-effects-4**: a refused buffer's CREATE_INDEX is not rolled back, because the Indexer is shared by `Arc` across versions (`apply.rs:72`, `graph.rs:3214`).

### Wrong results: binder / parser (`proofs/binder`)
- **W2-binder-1** (**fixed by #2981**: the pattern separator continues only on `,`, so no clause is folded; `bind_graph` defines relationship aliases first; Lean parser_grammar `PatternG.Ext.rust` fold off, binder `node_attr_on_rel_rust_accepts`, `q2923_fixed`, `match_rel_reused`; planner_build `q2923_lifted`, `disjoint_match_cp`): the general cause of #2923. The parser folds `MATCH p MATCH q`, `CREATE … CREATE …` and `OPTIONAL MATCH p MATCH q` into one pattern (`cypher.rs:1540-1556`). The WHERE pattern-predicate loop (`cypher.rs:2102`) swallows the next MATCH. `bind_graph` binds all nodes before any relationship (`binder.rs:1322` vs `:1339`). Effects:
  - consecutive CREATEs can't see each other: "'a' not defined", C returns 1;
  - relationship uniqueness is enforced across separate MATCH clauses: 0 rows, C returns 1 (also seen in ops_traverse);
  - a mandatory MATCH after OPTIONAL MATCH becomes optional;
  - `(a)-[r]->(b {v: r.w-2})` is rejected.
- **W2-binder-2** (**fixed by #2981**: `lift_cross_entity_attrs` lifts an entry that reads another variable of the pattern above the whole pattern, as WHERE; planner_build `q2923_lifted`, `disjoint_match_cp`): an inline property filter that references another component was placed on its own component's scan (`planner/mod.rs:1840` before the fix): [] vs C's 2 rows. This is the node case of #2923.
- **W2-binder-3**: deleted scope-table entries let ids be reused. WHERE/SKIP/LIMIT parent copies (`binder.rs:1231-1239`) and comprehension or pattern locals (`:1893`, `:2068`) are removed while their ids are still in use. The results are wrong rows, extra rows, and "Invalid node id for 'to'".
- **W2-binder-5**: ORDER BY copies leak past WITH/RETURN (the snapshot at `binder.rs:1216`). This causes a spurious "already declared", and `RETURN *` also returns ORDER BY-only variables.
- **W2-binder-6**: labels from CREATE and from pattern comprehensions tighten an earlier MATCH scan (`binder.rs:1645`, `:1847-1904`): 0 or 1 rows where C returns 3.
- **W2-binder-7**: a pattern predicate in WITH … WHERE rebinds a parent variable as a fresh one (`define_name_in_scope`, `binder.rs:2121`): 3 rows where C returns 1.

### Wrong results: optimizer
- **W2-rewrites-1**: a filter merge evaluates the outer conjunct first (`push_filters_down.rs:122`). This gives "Division by zero" where C returns 15.
- **W2-rewrites-2 / W2-scan-***: id seek with a negative bound, `id(n) > -1`, gives 0 rows where C gives 5 (`runtime.rs:1319`); a non-integer bound (`id(n)=null`, `=1.0`, `>1.5`) raises an error where C gives 0 rows (`runtime.rs:1324`).
- **W2-rewrites-3**: the hash join drops Int/Float pairs near 2^53 (`value_hash_join.rs:153` compares keys exactly, while `=` goes through `as f64`).
- **W2-scan-1**: `select_scan_node.rs:893-900` marks `best_node` as bound when no scan is built for it, so a filter runs before its variable is bound: 0 rows where C gives 1.
- **W2-scan-6** (confirmed live on main 43936bab9 and 65654bc23; this is #2972 (its `UNWIND` row), fixed by open PR #3099 (2b1e86053), checked on #3099's build: #2972's three repro queries give 1 / 2 / 2 there, as on C, where main gives 0 / 0 / 1; the two queries below are the `WITH` and `UNWIND` cases among them): `select_scan_node` prunes a planner scan whole but salvages only its `Filter`s and `Argument` leaf (`is_planner_scan_subtree` `:315` accepts a scan with any child; `argument_leaf_of` `:276`). A `WITH`/`UNWIND` the planner stitched below the scan is lost, and a salvaged filter reading its variable runs unbound. `WITH 1 AS k MATCH (b:B) WHERE b.v = k MATCH (b)-[:H]->(c:C) RETURN count(*)` → Rust 0, C 1; `UNWIND [1,1] AS k MATCH (b:B) WHERE b.v = 1 MATCH (b)-[:H]->(c:C) RETURN count(*)` → Rust 1, C 2 (no `Unwind` in EXPLAIN). Lean `optimizer_scan` `Salvage.premise_fails`, `salvage_loses_row`, `rebuilt_drops_unwind`; it refutes the premise `ScanOrder.salvage_wellScoped` takes, so that premise is now modelled instead of assumed (`Salvage.premise_of_spine` gives a sufficient condition under which it holds; the refutation is the one concrete `Filter → NodeByLabelScan → Unwind` tree, `exChild`).
- **W2-scan-3**: a computed constant ≥ 2^52 in an index equality returns every node (`utilize_index.rs:910-928`).
- **W2-scan-4/5**: an `IN` with a computed left side is pushed as `n.a IN [...]`, and `2 IN [n.a, n.b]` becomes an array-contains query on `a` (`utilize_index.rs:630-655`). *Fixed by #3068* (`bare_property_of`): Lean C7/C7'/C8 now agree, `trySingle_inn_bare`.
- **W2-traverse-1**: an unreferenced named edge is collapsed even though a sibling traverse still reads it for uniqueness (`reduce_expand_into.rs:118-143`): 1 where C gives 2. Check against PR #2918.

### Wrong results: runtime operators
- **W2-apply-1**: a ValueHashJoin inside a batched Apply or Optional joined rows from different outer rows (16 where C gives 8). **FIXED by #3035 (closes #3012)**: the join keeps one hash table per `origin_row` and probes with the left row's origin, so the join stays in the batched sub-plan. The Lean model of the old origin-blind join (`vhj_not_separable`, `canBatchApply_admits_vhj`) is kept as the historical counterexample; the proof bot maintains the Lean side.
- **W2-apply-4**: the all-bound MERGE shortcut drops matches visible to a path variable (`merge.rs:368-400`): `count(p)` is 1 where C gives 2.
- **W2-apply-5** (low): batched Optional moves the NULL rows after the matched rows, which `collect()` exposes.
- **W2-traverse-2**: the bidirectional dedup (`cond_traverse.rs:273-299`, `947-968`, `1267`) ignores other columns, keys on an intermediate node at 3 or more hops, and resets per batch.
- **W2-traverse-3**: an undirected anonymous collapse gives one row per direction (`cond_traverse.rs:876-942`, `expand_into.rs:182-221`): 4 rows where C gives 2.
- **W2-traverse-5** (partly fixed by #3580): `algo.SPpaths`/`SSpaths` with no `costProp` at all: the Dijkstra fast path charges 0 per relationship (`CostProp.fast_unconfigured_agree`) while the enumeration path's default is 1.0 per edge, so the two paths disagree; C master charges 1 there since ae9ef2a73 (the `edge-c` image charges 0). With a `costProp` that names an attribute no edge has, Rust charges 1 per edge, as C master (`CostProp.fast_unknown_name_hops`); on `(1)-[:R {w:5}]->(2)-[:R]->(3)`, `costProp:'nosuch'` gives pathCost Rust 2 and `edge-c` 0. Not filed.
- **W2-agg-1**: clearing the global dedupers after each CALL{} (`apply.rs:283`) makes an outer `count(DISTINCT)` over-count: 6 where C gives 2.
- **#2779** (confirmed live): `count(DISTINCT …)` above a per-row `CALL {}` counts duplicates once its input spans batches: `UNWIND range(1,5000) AS x CALL { WITH x RETURN x % 3 AS y LIMIT 1 } RETURN count(DISTINCT y)` → Rust 15, C 3. Fix open as #2968, with #2738 and #2729.

### Wrong results: index layer (`proofs/index_layer/repro.py`)
- **W2-index-1**: `n.v = date(..)` returns every node of the label (`utilize_index.rs:631` drops the filter and `node_by_index_scan.rs:252` falls back). C ✓.
- **W2-index-2**: `n.v = point(..)` returns [] (`index/mod.rs:1660`). C ✓.
- **W2-index-3**: `n.v IN [date(..), 2]` drops the date item (`node_by_index_scan.rs:221-236`). C ✓. Still open after #2383: the scan ops now build one `InList` but drop non-scalar members first (filter at about node_by_index_scan.rs:221-236). #2383 itself is behaviour-preserving (`canUse_inListV_eq_or`).
- **W2-index-4**: for multi-label `(n:A:B)`, conditions on B's indexed property are pushed into A's index (`utilize_index.rs:503`, `:521`; `mod.rs:1547`, `:1564`). C ✓.
- Shared with C:
  - the TAG encoding breaks string order for bytes ≤ 0x20, `\` and `_` (`mod.rs:565`);
  - Bool is indexed as 1/0 (`bool_divergence`, pinned as C-compatible);
  - `n.v = ''` returns [];
  - a geo radius above ±85° returns nothing;
  - an indexed `n.v > i64::MIN` drops matching rows because `int_loses_f64_precision` lets i64::MIN through while i64::MIN+1..+512 round to the same f64 (`i64min_hole`, #2538); C drops the same rows, and for `=` Rust is right where C returns extra rows. Suggested fix: reject when `unsigned_abs() >= 1<<52`.

### Wrong results: functions
- **W2-strlist-3**: `=~` matches substrings: `'abc' =~ 'b'` is true, where openCypher says false (`internal.rs:112`, `eval.rs:1274`).
- Conversion divergences seen live, with no Rust test:
  - `toString(1.0)` gives "1" (C "1.000000");
  - `toInteger('1e3')` gives 1000 (C null);
  - `toInteger(' 12')` gives null (C 12);
  - `toFloat('0x10')` gives null (C 16);
  - `duration('P5')` gives PT0S (C null).

### Stats (write-statistics counters)
- **W2-pending-2**: entities created and deleted in the same segment vanish from the stats (`pending.rs:677`, counters at `:1141`, `:1147`, `:1226`).
- **W2-pending-3**: writes to a deleted relationship are applied and counted (`remove.rs`; `set.rs:230` misses a DETACH cascade).

### C-side bugs seen along the way (not Rust)
- The C master crashes on `OPTIONAL MATCH (n:Nope) SET n:L6` (Redis assert `blocked.c:346`).
- C SIGSEGVs in `RediSearch_ResultsIteratorNext` on `distance(..) <= 0` with a range index.
- C hangs on some `WHERE id(c) < 0` patterns.
- On macOS, C crashes on node DELETE in one agent's environment.
- In C, a MERGE inside CALL returns one row, and ON MATCH/ON CREATE SET values land on the wrong rows.

---

## Wave 3: open bugs

### Server crash / hang / data loss
- **W3-parser-1**: the shortestPath filter-skip loop never ends at EOF (`cypher.rs:1819-1845`). `RETURN shortestPath((a)-[{` spins a worker forever, and a handful of these block all graph queries. **Fixed by #3019**: the filter is now parsed (`$p` one token, `{...}` by `parse_map`), so input ending inside it is a syntax error.
- **W3-conc-4**: a huge `k` in `db.idx.vector.queryNodes` runs `Vec::with_capacity(k)` and SIGSEGVs (`graph.rs:3744`, `:3799`).
- **W3-algo-1**: forged `__falkor_type` entity markers in a UDF result are trusted (`type_convert.rs:229-236`). This panics the reply or creates a dangling edge.
- **W3-algo-2**: SHUTDOWN after a main-thread UDF aborts, because ThreadJsState drop order frees the QuickJS runtime first (`js_context.rs:129-136`). **Fixed by #3015** (fields reordered, `js_context.rs:143-151`): `UdfContext.threadJsState_drop_safe`.
- **W3-runtime-1/2**: the LIMIT/SKIP budget is pushed through hard-cap traversals (`runtime.rs:594-626`), so traverse→traverse under LIMIT, and stacked SKIPs, silently lose rows.

### Wrong results
- **Planner** (`proofs/planner_build`):
  - a multi-component MATCH after `MATCH … WHERE` is cross-joined, not correlated;
  - MERGE with a named path as the last clause, or after another clause, never runs: `MATCH (a:A) CREATE p=(a)-[:R]->(:B)` and `UNWIND [1,2] AS x CREATE p=(:N {x:x})` give "Variable _anon_0 not found" and create nothing; C creates them (W8-plan-1, a regression from #2829: the planner's first walk does not step over PathBuilder; fix open as #3101; Lean `create_path_last_misplaced`, `create_path_last_skips_create`).
- **Constraints**:
  - the UNIQUE key includes the value's type (`graph.rs:4141`, #2774);
  - a constraint created over violating data reports OPERATIONAL.
- **Index / concurrency**:
  - unconfirmed: between `Indexer::create_index` storing an entry and `populate_index` taking its ticket (`graph.rs:3598-3604`) the index reads OPERATIONAL with the new field unpopulated, so a concurrent reader could plan an index scan and miss rows (`store_window_operational_incomplete`); not seen live;
  - index-option validation differs from C: `OPTIONS {dimension:0, similarityFunction:'bogus'}` creates the index (C refuses); fulltext `OPTIONS {weight:1.0, foo:true}` is refused (C creates it; C only refuses a non-empty map with no known key); the `SAMPLES` error text has an `ERR ` prefix (C does not).
- **GraphBLAS**:
  - unconfirmed follow-up to #3033: `Tensor::decode` still accepts a multi-edge pair with exactly one id, or one edge id under two pairs (`edge_count` stays consistent; `PairPlan` assumes >= 2 ids);
  - `Matrix::decode` leaks 5 blocks per matrix (fix open as #3080);
  - unverified follow-up to #3097: `Matrix::decode` still copies the header's reserved `*_future` pointer arrays from the payload; safe only if GraphBLAS never reads or frees them.
- **Redis layer**:
  - loose TIMEOUT/version/arity parsing;
  - several output-format differences;
  - module-load arguments: C's `strtoll` skips leading whitespace and Rust's integer parsing does not, so `--loadmodule … CACHE_SIZE " 10"` and `GRAPH.CONFIG SET RESULTSET_SIZE " 5"` succeed on C and are refused by Rust (`leading_space_divergence`, confirmed live). Rust accepts `INDEX_WORKER_THREADS`, `BOLT_PORT`, `EFFECTS_COMPRESSION`, which C refuses (`rust_only_names`, intended).

### Latent
- `vector::Iter` can outlive its vector (the #2898 class).

## Wave 4: open bugs

- **W4-graph-2** — `get_node_relationships` (`graph.rs:2056`) reports a self-loop twice; seen through UDF `getNeighbors` (count 2 vs C 1). Probably the same as #3083 (PR #3084 fixes the UDF side) — check whether #3084 fixes it at this source.
- **W4-graph-3** — constraint matching depends on property order (`constraint.rs:79`): `UNIQUE NODE L (a,b)` then `(b,a)` creates both; C says "Constraint already exists".
- Minor: `CREATE INDEX` on a new label also reports "Labels added: 1" (C does not).
- Minor: `point({latitude:null, longitude:1})` gives a different error message from the same map passed as a value.
- **W4-plan-1** — `expr_to_plan` keeps null rows under `NOT` (`planner/mod.rs:1421-1427`): `NOT(complex)` is planned as AntiSemiApply, so rows where the inner predicate is null pass. `MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)` → Rust 2, C 1. Tests in `lean_planner_build.rs`.
- **W4-plan-2** — a subscript is never accepted as a boolean (`binder.rs:2304-2314`): `WHERE n.flags[0]` → "Expected boolean predicate", C 1. Same for `true AND l[0]` and `true AND -null`.
- Minor divergences: `false AND (1+2)` → false (C type error); `date({year:2020, month:null})` treats null as absent (C errors); `WITH r WHERE r.w = 1` on a var-length `r` is rejected (C 1).
- **W4-parse-5 (crash)** — 2000 chained `UNWIND [1] AS xN` clauses crash the server in planning or execution (parse, validate and bind succeed up to 8000); C runs 5000.
- Lean-only, not confirmed live (the malformed-RDB path): `set_span` stores `n as u16` (`attribute_store.rs:597`, `:608`), so 65,536+ entries per entity wrap in a malformed RDB (the GRAPH.BULK side was fixed by #3063).
- **W5-proc-1..3** (procedures, live vs C):
  - a lone `CALL db.idx.fulltext.createNodeIndex({label:'W'})` under GRAPH.QUERY fails with a misleading "graph.RO_QUERY is to be executed only on read-only queries" (`procedure_call.rs:67`);
  - `db.idx.fulltext.drop` on a missing index errors (C does nothing);
  - `db.idx.fulltext.queryNodes(1,2)` errors (C returns empty). (*fixed by #3269, c23975b98; now `ops_traverse` `IndexScan.ftArgs_nonstring_empty`; old model `pre3269_ftArgs`*)
  - Output differences: `db.indexes` and `db.constraints` row order; `db.indexes` `info` holds only field names (C returns full RediSearch info and `_src_id`/`_dest_id` for edge indexes).
- **W5-udf-2** — `graph.traverse([a])` leaves `a` out when it is its own neighbour, because the start node is pre-marked visited (`js_classes.rs:744`): Rust `[1]`, C `[0,1]`.
- **W5-udf-3** — `node.getNeighbors()` with no argument throws "Error converting from js 'undefined'" (`js_classes.rs:195`, `:352`); C returns the neighbours.
- **W6-opt-1** (found reviewing PR #2845, pre-existing on main 89d68334a) — filter push-down gives an OPTIONAL MATCH inside a `CALL {}` body the outer query's variable ids, so the filter on `n` is pushed onto a row where that slot is unbound (`push_filters_down.rs:217`, `:238`). `MATCH (x:A:B) CALL { OPTIONAL MATCH (n:A:B {id:3}) RETURN n.id AS nid } RETURN x.id, nid` → Rust "Variable n not found", C `[[1,3],[2,3],[3,3]]`; four variants fail the same way. Possibly related to #2728. Lean `filter_pushed_through_boundary` (binder `CallBoundary.lean`). A trial fix (match `IR::Apply | IR::Optional(_)` and inherit `branch_output_variables(left)`) makes all five match C, and 166/166 related flow tests pass.
- **W4-index-1** — a range index on an attribute that already has a fulltext index is never queried correctly. `build_query_node` takes the attribute's first field whatever its type (`index/mod.rs:1386,1411,1459,1471,1529,1590`). After `CREATE FULLTEXT INDEX … ON (n.s)` then `CREATE INDEX … ON (n.s)`, `n.s='foo'`, `n.s > 'c'`, `{s:'bar'}` and `n.s IN [...]` all return `[]` (C returns the rows). Fix: pick the field whose type is `IndexType::Range`. Lean `IndexerM.bug_range_after_fulltext`.
- Unconfirmed: `create_rs_index` leaks index options on its NUL-byte error path.
- **W7-scan-1**: `select_var_len_scan_node`'s leaf rewrite prunes a Filter wrapper: `MATCH (a) WHERE a.v=1 MATCH (a)-[*1..2]->(b:B {w:2})` returns an extra row. Predates #2390. Lean `ScanTree.vl_leaf_*`. Fixed by PR #3264 (`ScanTree.vlLeaf_keeps_filter`, `vl_leaf_bug_case_fixed`).

### Open bugs the proofs had missed (triage of 2026-10-20)
A triage of every open `type:Bug label:rust` issue (live on Rust main 9d3869aa8 and C) found 56 the proofs had not flagged. Only the proof gaps needed a model change; the models were extended so each has a stated property a fix can prove and a `bug<issue>_*` counterexample on main that matches the live repro. The entries below are the ones still open. Root-cause families: slot ids reused across WITH/aggregation scopes (#3333 #3372 #3375), named paths over an already-bound node (#2311 #2725 #2727 #2737 #2744), rewrite criteria that compared rows but not writes (#748 #997), and per-row write visibility through CALL {} (#2729 #2738 #3164). #2502 is the parser (a WHERE pattern predicate swallows the next MATCH), and #2556 is select_scan_node seeding an unbound endpoint, not push-down.
- #748 (plan/optimizer): a constant-false WHERE after `CALL { MERGE … }` is pushed into both Apply branches by push_filters_down (the empty variable set routes everywhere), so the MERGE never runs: Rust 0 nodes, C 1, and the rows-only criterion cannot see it. Lean bug748_merge_skipped; property EffectSound.
- #997 (plan/optimizer): `CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE false` gets the false filter pushed onto the Apply's Argument, so the CREATE never runs: Rust 0 nodes, C 1, while the CASE-wrapped form creates 1. Lean bug997_create_skipped; property EffectSound.
- #1334 (binder): the Pattern arm of bind_expr_node mints unknown aliases fresh, so WHERE () -- (n1) with no n1 in scope is accepted (C errors). Lean bug1334_pattern_predicate_declares_var; property PatternPredDeclaresNothing.
- #2311 (plan): OPTIONAL MATCH p=(n) over a bound node: the bound node-only branch of plan_match (mod.rs:1915-1962) emits no PathBuilder, so p is null (Rust `1, null`, C `1, [(0)]`). Lean bug2311_optional_path_null; property BindsAllPaths.
- #2502 (parser): a WHERE pattern predicate is parsed by `parse_pattern(&Keyword::Match)` (cypher.rs:2119), whose separator continues on `MATCH` (1562-1579), so `MATCH (a) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` folds the second MATCH into the predicate (Rust `'m' not defined` / count 1, C rows / count 2). Lean bug2502_pred_swallows_match (+ bug2502_where_match_merged); property PredEndsBeforeClause. **Fixed by #2981** (the separator continues only on `,`): fixed2502_pred_stops_at_match, fixed2502_where_match_split.
- #2556 (plan/select_scan_node): with the bottom hop's outer child kept (UNWIND after an aggregating WITH), `initial_bound` seeds the best endpoint `t` (select_scan_node.rs:979-990), so the inline filter `t {id:100}` is placed after the first hop, before `t` is bound; Rust 0 / C 3 (the issue's own query: a "was List" type error); not push_filters_down. Lean bug2556_filter_below_binding; property KeepsFiltersBound. FIXED by #3523 (seed `best_node` only when no outer child is kept; optimizer_scan `ScanOrder.initialBound_wellScoped`, `initialBound_keeps_row`).
- #2653 (store): a non-UTF-8 graph key is replicated and signalled through from_utf8_lossy (`\xc3(bin` → `\xef\xbf\xbd(bin`), so the replica writes a different key. Lean bug2653_nonutf8_key_mangled; property ReplicatedKeyIsStored.
- #2725 (plan): `MATCH (n0) CALL … YIELD … MATCH (n0), p0=(n0)` is planned `Project(Argument(ProcedureCall(Scan)))` and gives one empty row, so `Variable n0 not found` (C answers). Lean bug2725_call_then_bound_path_loses_var; property Plan.argIsLeaf, BindsAllPaths.
- #2727 (plan): `WITH n0 AS alias4 MATCH (alias4), p1=(alias4)` returns 0 rows (C `'x'`); same two root causes as #2744. Lean bug2727_alias_path_drops_rows; property BindsAllPaths, Plan.argIsLeaf.
- #2729 (rt/create+delete): streaming CREATE→DELETE depends on the batch split, so the endpoint check rejects a node an earlier batch pending-deleted; the CALL {} form splits the rows differently, and top-level `UNWIND range(1,1100) … CREATE (n)-[:R]->(n) DETACH DELETE n` also fails while C succeeds. Lean RowWrites.bug2729_batch_split_flips_endpoint_check; property RowWrites.PartitionInvariant.
- #2737 (plan): `MATCH (n1:X) OPTIONAL MATCH p6=(n1) RETURN p6` gives null (C `[(0)]`); same root as #2311 (nodeOnly_bound_no_paths). Lean bug2737_optional_path_null; property BindsAllPaths.
- #2738 (rt/apply): a writing CALL subquery is batched (can_batch admits Create/Delete/Set/Commit), so later outer rows do not see earlier rows' writes; Rust count 3 / 4 nodes vs C 7 / 8. Lean RowWrites.bug2738_batched_call_hides_writes; property RowWrites.WriteSeparable.
- #2744 (plan): `MATCH (n0:A) WITH n0 MATCH p=(n0)` is planned `Project(ExpandInto(Argument(Project(Scan))))` and returns 0 rows (C 1); planMatchArm_path_unwrapped. Lean bug2744_bound_path_unfed_argument; property Plan.argIsLeaf, BindsAllPaths.
- #2781 (idx): rolling back a failed multi-Commit query that reused an edge id with new endpoints re-adds the committed key but never deletes the private-endpoint document, so an orphan (c,d,0) stays in the edge index. Lean IndexLayer.Iter.bug2781_rollback_not_restored; property RollbackRestoresEdgeIndex.
- #2782 (srv): GRAPH_REGISTRY is keyed by name, not (db, name), so CREATE `shared` in db0, db1, db0 → the third write aborts with "deleted or replaced" (C ok ×3). Lean bug2782_same_name_other_db_aborts; property KeyspaceRegistered.
- #3164 (rt/apply+id_space): per-row commits release a deleted node's id, a later CREATE reclaims it, and later outer rows' stale `Value::Node(id)` delete the new node; Rust deletes 2 vs C 1. Lean RowWrites.bug3164_stale_ref_deletes_recycled_id; property RowWrites.NoStaleAlias.
- #3166 (idx; closed as a duplicate of the open #2449): CREATE FULLTEXT INDEX OPTIONS {stopwords} on a label that already has a range index records the stopwords but leaves the existing RediSearch spec on its default list (create_index only calls register_fields), so 'router' still matches (Rust 1 row, C 0). Lean IndexLayer.Meta.Ixr.bug3166_stopwords_not_applied; property StopwordsAgree.
- #3226 (idx): a write that commits while its own index scan is open (scan below LIMIT) re-takes the RediSearch spec lock as a writer while holding it as a reader, and the server aborts at spec.c:2304; the same path takes the GIL while holding RediSearch's global lock (stall with RESTORE). Lean SC.Locks.bug3226_open_scan_upgrade; property noReacquire.
- #3333 (aggregate/binder ids): the pre-projection alias of m (id 2) stays bound on the Aggregate output and the next MATCH gives b id 2, so b starts bound to m and the query returns []. Lean bug3333_alias_binds_next_var; property EmitsOnlyScope.
- #3349 (udf): WCC, pageRank and betweenness on a one-node nodeLabels projection fail with -1000: GraphBLAS stores the 1×1 compact matrix by column, LAGraph_CheckGraph rejects it, and Rust returns the error. Lean pre3572_one_node_projection_fails; property ProjectionTotal. **Fixed by #3572** (pin_row_major on the compact matrix and G->AT): projection_total.
- #3352 (plan/optimizer): Case 2 push-down inherits bare ids from the whole pre-WITH subtree (get_variables walks through Aggregate), takes the pre-WITH id 2 for `r`, and puts `r.status = 'revoked'` on the Argument: Rust e=null, C 123. Lean bug3352_slot_collision_drops_match (bug3352_routes_unbound); property RowSound (RoutesBound). Since #3521 get_variables stops at Aggregate/Project (`RuntimeCore.Plan.getVariables_stops`, old walk `pre3521_hidden_reported`). **Fixed by #2981** (Case 2 inherits branch_visible_variables): fixed3352_plan, fixed3352_rows_kept.
- #3368, #3369 (rt, filed repros fixed by #3050): model only and not yet run live: equal leftovers in two rows of the SAME MERGE still share one `merge_pattern_cache` entry (`MergeSlot.same_op_leftover_shares_entry`).
- #3372 (rt/aggregate→traverse): Aggregate re-binds a grouping key at its slot from the previous scope (aggregate.rs:1187-1196), and the binder restarts slot ids at 0 in each scope (binder.rs:2261). A fresh variable in the next MATCH that gets the same slot is read by CondTraverse as already bound to that node (cond_traverse.rs:648-652,771-777), so `WITH c, count(*) AS k MATCH (c)-[:E]->(d)` returns 0 instead of 4. Lean bug3372_slot_reuse_drops_rows; property ScopedRow.
- #3375 (aggregate/binder ids): count(x)'s output id equals sum(x)'s accumulator id and is written into `combined` first, so sum returns the count (3 vs 6) and avg panics in finalize_avg. Lean bug3375_output_overwrites_next_acc, bug3375_avg_panics; property OutputsOwnAcc.
- #3380 (udf): algo.betweenness defaults to 16 sources (C 32), never randomises without a seed (C uses the clock), and drops colliding seeded draws (samplingSize 20, seed 7 on 40 nodes → 15 sources). Lean bug3380_default_size_16, bug3380_seedless_deterministic, bug3380_seeded_fewer_sources; property DefaultSizeMatchesC, SeedlessRunVaries, SourcesExact.
- #3382 (idx): db.idx.vector.queryNodes answers from a half-built index (0 rows, then 3, while UNDER CONSTRUCTION); Indexer::vector_query never checks is_operational, while C refuses until OPERATIONAL. Lean IndexLayer.Meta.Ixr.bug3382_vector_query_ignores_operational; property VectorQueryRequiresOperational. PR #3581 gates Graph::vector_query_nodes/_edges on Indexer::is_operational (graph.rs:4381,4447): SC.Knn.not_operational_refused, IndexLayer.Meta.Ixr.vectorQueryGuarded_eq_isOperational (Indexer::vector_query itself still has no gate).

### Open reports #3442–#3454 (gkorland, 2026-10-22)
Of the twelve Rust bugs filed on 2026-10-21 and reproduced against main b84473a7f and C, these are still open. #3446 (GRAPH.EFFECT expands a compressed id range eagerly) is a memory cost outside what the proofs can express. Common causes: operators modelled over the committed graph only (pending deletes and staged labels invisible to MERGE/ExpandInto/IncludePending/UDFs), label sets modelled as only growing (REMOVE), and replies or effect streams proven for shape only, with the panicking lookups taken as total.

- #3447 (store/effects→replica): `remove_pending_relationships_for_node` (pending.rs:839-896) drops a type's `created_rels_by_type` group once empty, so `Pending::commit` (pending.rs:1239-1253) never registers a type whose only edges were cascaded; `digest_cancelled` (emit.rs:457) skips that cancelled edge id and the replica's relationship id space has a hole that `IdSpace::verify` (id_space.rs:737-790) refuses. Lean bug3447_cancelled_edge_id_gap, bug3447_cascade_drops_type_key; property RelIdsCovered / CancelledTypeRegistered; fix relIdsCovered_of_registered. **Fixed by #3461** (f203972c5): `Pending::commit` registers every cancelled relationship's type (pending.rs:1360-1371); Lean relIdsCovered_after_commit, fixed3447_cancelled_edge_id_gap, pending_commit PC.stCancelTypes_registers; old counterexample kept as pre3461_bug3447_cancelled_edge_id_gap. #3461 also makes `effects_count` (pending.rs:1970) count cancelled entities, fixing the cancelled half of W2-effects-1 (gate_ships_of_cancelled; history pre3461_gate_drops_cancelled, pre3461_cancelled_edge_attr_dropped); schema-only writes are still dropped (gate_drops_schema_only).
- #3447 (srv/reply, compact part): a pending edge of a new type that a DETACH DELETE cascades away is snapshotted with its type name. Its type is never registered, because Pending::commit registers only surviving groups (pending.rs:872-873, :1242-1251). The compact reply's `get_type_id(..).unwrap()` (reply.rs:260) then panics; the verbose reply renders the edge. Lean bug3447_compact_cancelled_new_type_panics; property RegisteredTypes ⇒ ReplyTotal (replyTotal_of_registered).
- #3448 (store/effects): `remove_node_labels` (pending.rs:699-711) also stages labels that were only staged, and `delete_pending_node` (:788) never clears `remove_labels`; `emit_v3` digests `remove_labels` (emit.rs:372) before `digest_cancelled` (:376), so `CREATE (n:Foo) REMOVE n:Foo DELETE n` emits `RemoveLabels [0]` before `CreateNode [0]` and the replica's `require_live` (apply.rs:359-360) refuses it. Lean bug3448_removeLabels_before_create_refused, pre3459_bug3448_remove_of_staged_label; property ReplayApplies / RemLCommitted / RemLLive. **Fixed by #3459**: `remove_node_labels` counts a removal of an only-staged label in `staged_labels_removed` instead of staging it, and `delete_pending_node` clears `remove_labels`; Lean fixed3448_run_remL_committed, fixed3448_remove_of_staged_label, fixed3448_cancelled_node_clears_remL, effects_emit_apply fixed3448_replay_applies, gate_keeps_staged_removal.
- #3450 (binder REMOVE/WITH labels): `bind_ir` QueryIR::Remove (binder.rs:477-482) leaves `node_labels` alone; `bind_projection` copies the parent's labels to the WITH child (:1132-1141/:1159-1168) and `update_all_node_labels` (:122-134) stamps them on the next MATCH, so `MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b)` scans `(a:A)` (Rust 0 rows, C 1). Lean bug3450_remove_label_kept_across_with; property StampedLabelsHeld.
- #3450 (planner verified_labels): the QueryIR::Remove arm (mod.rs:3212-3218) never unmarks `verified_labels`, `plan_project` carries them across WITH (mod.rs:2479-2492), and `unverified_labels` (:1793) then gives the bound endpoint no hasLabels filter (:1840-1856), so `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b)` returns 1 row `[] [B]` (C 0). Lean bug3450_remove_keeps_verified_label; property RemoveKeepsVerifiedSound.
- #3451 (rt/path_builder): PathBuilder infers a var-length segment's direction by comparing the VLT path's first node with the previous path node (`path_builder.rs:102-110`); on a cycle (n = m) that says "forward", so `MATCH p=(a{id:0})<-[:R*3..3]-(a)` on `0→1→2→0` gives `[0,1,2,0]`/`[1,2,3]` instead of `[0,2,1,0]`/`[3,2,1]`. Main is correct on every non-cyclic walk (`vltMain_follows_of_ne`). Lean bug3451_main_violates, bug3451_cycle_not_reversed; property PathFollowsPattern; fix vltFix_follows.
- #3452 residual (rt/traverse; the ExpandInto half was fixed by #3463): `CondTraverseOp` never reads `Pending`, so `DETACH DELETE a MERGE (a)-[r:R]->(x:B)` still matches the doomed edge (`EIP.bug_condTraverse_main_violates`; from the code, not yet run live; needs an owner).

### Open: #3483 (2026-10-26)
- #3483 (srv/idx, crash): AOF replay of 152 `GRAPH.DELETE`s of indexed graphs crashes 6.0.3 in `RM_FreeString` (7/20 starts here; `illegal decrRefCount … type 0, encoding 8, refcount 0`, the embstr "N/A"). `graph_free` drops the graph on `std::thread::spawn` with no GIL (`src/graph_core.rs:1797`); `SpecHandle::drop` → `RediSearch_DropIndex` → `IndexError_Clear` frees RediSearch's process-global `NA_rstr` (`deps/RediSearch@ea1a6f4 src/info/index_error.c:91`), while the main thread's `create_rs_index` holds it (`:42`). The refcount is a plain int, so updates are lost and the key is freed early. C pins RediSearch 311f90126, which has no `NA_rstr`. Fix: FalkorDB/RediSearch#29 (errors borrow `NA_rstr`). Lean `SC.Guarded.bug3483_naRef_race` (lockset of create `{gil, indexer}`, drop `{rsGlobal}`, last `IndexRelease` `{}`: empty); fixed `rs29_naRef_free`.
- **Why the proofs missed it.** `RediSearch_DropIndex` and `RediSearch_IndexRelease` are AXIOMATISED (`index_layer/COVERAGE.tsv`), and their spec covers refcount and invalidation, but not which lock the caller must hold. `Locks.dropIndexBgProg` models the drop as touching nothing, and `Locks` checks lock order only, never what a thread touches without a lock. The Rust comment that the drop needs no GIL ("touches no Redis timer") had the same gap, and the property it relied on was never stated. New obligation: for each shared resource behind an FFI call, the locks held at every access (C side and Rust call site) must share a member; `SC.Guarded.consistent` checks it (Eraser lockset).
- **LLAPI thread audit (pilot, 2026-10-26).** All 45 `RediSearch_*` functions FalkorDB calls, C side at a272834a6 vs ea1a6f40c and every Rust call site with its thread and locks. GIL-free call sites: index scans on readers (clone, node build, iterate, next, free), unique-constraint lookups before escalation, background population (`CreateDocument2`, `DocumentAddField*`, `IndexAddDocument`, `FreeDocument`), `SpecHandle::drop` → `DropIndex` (drop_index_bg, readers, every teardown thread), `OwnedIndex::drop` → final `IndexRelease`, `GRAPH.MEMORY` → `MemUsage`. At a272834a6 the index free (`IndexSpec_Free`) touches no timer and no shared Redis object, so all of these are allowed; `CreateIndex` still needs the GIL (`RM_CreateTimer`), and every caller holds it. `RM_EventLoopAddOneShot` is non-null in the loaded module (checked with lldb), so #29's main-thread GC stop is live.
- **H1 (open, idx, corruption)**: ForkGC forks under the GIL only (`deps/RediSearch src/fork_gc.c:1050-1065`) while `populate_index_batch` (`graph/src/graph/graph.rs:543-563`) adds documents under RediSearch's locks without the GIL; the child can scan a half-written block or tree and the parent applies its delta. Lean `SC.Guarded.h1_fork_vs_populate_race`; either fix (population under the GIL, or the GC holding the spec lock across the fork) is `h1_fix_populate_gil` / `h1_fix_gc_spec_lock`. No live repro yet.
- **H2 (latent, RediSearch#29)**: `IndexError_GlobalCleanup` frees `NA_rstr` (`index_error.c:214-219`); a later LLAPI index free re-creates it and `FreeKey(old)` frees the old one again. Only after `RediSearch_CleanupModule` (`src/module_init.rs:463`, `RS_GLOBAL_DTORS` builds), whose teardown threads are not joined.
- Lower: lazy global init races (`NA_rstr`, default/empty stopwords) are safe only because the first `CreateIndex` holds the GIL; `gc->deletedDocsFromLastRun`, `FieldsGlobalStats` are non-atomic and written GIL-free (stat drift); a results iterator must be freed on its creating thread (RediSearch's per-thread lock count), not yet shown impossible from Rust.
- Comments whose lock claims the code contradicts (besides the two #3511 fixes): `src/telemetry.rs:797-800`, `src/commands/config_cmd.rs:25-27` ("Index::drop takes the GIL"); `graph/src/thread_id.rs:19-22`; `src/commands/memory.rs:74-75`, `src/commands/explain.rs:43` (a "GIL lock-order assertion" that does not exist); `graph/src/graph/graph.rs:556-560` (population never touches the spec lifecycle: it does when it holds the last Indexer); `graph/src/locks.rs:13-15`; `src/redis_type.rs:350-353`; `graph/src/graph/graph.rs:3648-3651`.

### Open: #3439 and PR reviews on open PRs
Performance PRs from reproducing the one.semurg.io graph benchmark (SNAP com-Orkut) were reviewed with the proofs; the two whose PRs are still open keep their review projects on the PR branches, not on `main`.
- **#3435** (`algo.BFS`: lend the clean tensor base instead of copying it, bitmap visited set, push/pull): theorems `Clean.cleanFwd_eq_extract`, `Bfs.bfsFull_correct`, `Algo.nodes_same_as_main` on the PR branch; the review's three defects were fixed in the PR (a76e66420).
- **#3437** (list-comprehension / quantifier bodies evaluated columnarly over chunks): `Pr3437.Final.pr_comprehension_eq_main`, `pr_quantifier_eq_main`, `pr_size_eq_main`, `pr_chunkStep_eq`, `pr_columnar_sound` on the PR branch (cites to a739b990f). Remaining premise: `Final.Registry`, a call named `hasLabels` is the registered `hasLabels` (VectorEval picks its fast path by name); treated as parse-tree shape.

Confirmed bug found by these reviews (pre-existing on main, not caused by any of the PRs):
- **#3439 (srv, crash)**: a queued write calls `release_locks()` (`graph_core.rs:858`) and then serializes its reply with per-access borrows of its now-published version (`reply.rs:428-470`). An inline write inside `MULTI` that fails after a Commit runs `resync_published_indexes` (`pending.rs:1684`), whose `borrow_mut` on that version panics on the main thread ("already immutably borrowed"), killing the server. Repro `proofs/runtime_core/repro_3439.py P 20 2 1 0` after `CREATE INDEX FOR (n:L) ON (n.v)`: Rust main 9212add39 and b6567cb97 die within ~1 s; C survives (494 failed writes in 20 s, `:L` count 0). Lean `RuntimeCore.Proto.writer_reply_race`, `writer_reply_race_rev` (runtime_core/RuntimeCore/Protocol.lean). Suggested fix: serialize the reply before `release_locks`, or have resync update the index without `borrow_mut` on the published cell.

### Other open findings (confirmed or model-only)
- **Config rewrite** (confirmed live, same class as the fixed #3413; not filed): other native configs accept values through the load-argument path or `GRAPH.CONFIG SET` that their native range rejects, so `CONFIG REWRITE` saves a line the next start refuses and the server aborts. Live on main + #3419: `loadmodule … CACHE_SIZE 2000` → `graph.CACHE_SIZE 2000` → restart "argument must be between 0 and 1000 inclusive … aborting"; `GRAPH.CONFIG SET JS_HEAP_SIZE 5000000000` → restart "between 0 and 4294967296 … aborting". Lean also shows `THREAD_COUNT` > 1024 and `NODE_CREATION_BUFFER` > 2^30 by load argument. C writes no `graph.*` lines on rewrite. Lean `ConfigRewrite.rewrite_reload_counterexamples`.
- **`shutdown_within`** (latent, confirmed live; #3502; no production caller; `search_concurrency` `PoolWithin`, repro `proofs/search_concurrency/repro/threadpool_within.rs` on the real `threadpool.rs`): (1) a second `shutdown_within` returns `true` while a worker still runs, because the first call drained `workers` (`second_call_true_despite_straggler`; live `first=false second=true`); (2) a `shutdown_within(50ms)` behind a concurrent blocking `shutdown()` waits on the `workers` mutex, which `shutdown` holds across `join` (`threadpool.rs:149-150`), past its limit and forever if a job never returns (`within_blocks_behind_shutdown`, `holdEnd_ge`; live: still blocked after 3 s).
- **UDF conversion caps** (model only, not confirmed live, `cap_asymmetry_vecf32`): the two UDF conversion directions count a VecF32 or Path leaf differently, so 1024 lists around a vecf32 pass `value_to_js` but are refused by `js_to_value` on the way back; an identity UDF returns a nesting error for that value (an error, not a crash).
- **GRAPH.BULK null dedup** (to check live): the bulk caller drops `Null` values before dedup, so header `p,q,p` with values `(null,2,3)` keeps the second `p` (3); C's answer is unknown.
- A duplicate roaring high key in an IdList bitmap: roaring-rs keeps the last partition, and CRoaring may keep the first (unconfirmed).
- #2537, wider: any truncated GRAPH.RESTORE payload calls `LoadStringBuffer(NULL)` (fix open as #3167), not only an empty one; a payload cut at a record boundary SIGSEGVs in `RM_LoadStringBuffer` (`decoder/mod.rs:426` → `buffered_io.rs:212`), a cut inside a record errors cleanly; C crashes on both.

### Open gaps (modelling)
- native_index: `refresh_inv`, `rebase_inv`, `catchUp_inv` and `insert_inv` assume `Frame` (outside the written set the two versions agree on what each entity contributes and where it sits); that every staged write names the entity it changes is proven per write in `Writes.lean` (`*_changes`) and `writerRecord_records` records every named id, but the two are not composed into one induction over interleaved writers and batches (the global "a `Building` column holds exactly the tuples of the entities below its cursor" stays unproven). The rebase's one `Err(Stale)` (a release build, a fork still borrowed) is assumed unreachable for an escalating writer (`mvccRebase_unborrowed`); a debug build panics on it. `overtakenAny` proves the gate (a batch published since and a column overtaken); that a column can only be overtaken if the batch count moved is not proven here. `SE`/`Batch.lean` still model the one index a batch is about.
- `civil_from_days_wide` is modelled over unbounded `Int` (no i64 overflow proof, magnitudes are about 1e14); `includePendingMain_exact` needs a non-empty pattern.
- `check_node_constraint` and `check_edge_constraint` are CONDITIONAL on the contract of `native_unique_query` + `ReadOnce::candidates` (`unique.rs`); `native_index` proves that contract on its own model (`nativeUniqueQuery_mem`, `equalCandidates_complete`), but the two projects are separate Lake packages, so the link is not mechanised.
- The #3104 gate: `bounded_bal` checks the `Oracle.Bal` premise for every expression input up to length 3, but it is not proven for the whole expression parser, so `listLitP_eq_ungated` stays conditional on it and on `MemoSound` (the memo key omits `anon_counter`; no input was found where it matters).
- `evalVariable_beyond`: the vectorised `eval_variable` reads Null past the column space.
- Stale comments on main: `Matrix::delta_lmxm`'s doc still states the pre-#3492 `(self * (m + dp))<!(self * dm)>`; `build_state` says a missing type returns `None` (only all-missing does); the comment in `Runtime::new` (`runtime.rs`) cites the retired `proofs/pr3436_review`.
- Release `Cow` one-directional isolation and `OrderMap::from_unique_keys` are unchecked on purpose (debug-only asserts).

### Decisions made
- 2026-09-25 (Avi): a deleted entity keeps returning **all** its details (labels, properties, type, endpoints) for the rest of the query. Rust's current behaviour stays even where C returns `[]` for labels. PRs must not strip them; #3043 already leaves this alone.
- Traversal fixes are filed as #3106–#3109 with PRs #3110–#3113; #3106 (PR #3110) is still open. All PRs from this work carry the `6.0` label.
- The DISTINCT/MERGE 64-bit FxHash collision (DISTINCT, `count(DISTINCT)` and MERGE drop rows on a collision) was closed as not planned (#2780); `Value::eq` NaN/2^53 quirks are harmless since grouping and sorting use `sort_cmp`.

### Decisions needed (Rust follows the grammar or TCK, C differs)
- `2^3^2`, chained comparisons, `+` against STARTS WITH / IS NULL, `SKIP (1)`.
- Integer division through double vs exact.
- toFloat through float32 in C.
- toString(float) format (#2937).
- Net vs per-clause property stats.
- UNIQUE constraints with `true` = 1 (C) vs openCypher.
- Short-circuit evaluation (found while closing the #3437 gap, live on Rust 9212add39 vs C): Rust short-circuits AND/OR/XOR and CASE as its source intends; C evaluates every operand/branch. `RETURN false AND 1/0 = 1` → Rust `false`, C `Division by zero`; likewise `true OR 1/0 = 1` (Rust `true`), `null XOR (1/0 = 1)` (Rust `null`), `CASE WHEN true THEN 1 ELSE 1/x END` with x = 0 (Rust `1`). Lean expr_semantics `and_false_short_circuits`, `or_true_short_circuits`, `xor_null_settles`, `case_hit_lazy`.

---

## Not covered / modelling limits
- f64 arithmetic is abstract (no IEEE reasoning).
- GraphBLAS, RediSearch, the Redis module API and QuickJS are hypotheses or axioms at the FFI boundary.
- `Arc`, concurrency and HashMap iteration order are not modelled. Data races are modelled only for resources listed in `SC.Guarded.guard` (#3483); FFI axioms do not yet state their thread context.
- Aeneas extraction works only for small pure modules (`narrow_int::width_for` was proven on extracted code); larger modules are blocked by iterators, closures, `dyn`, `roaring` and `smallvec`. The pilot (`proofs/aeneas_pilot`) was removed on 2026-10-17: it added no coverage (`width_for` is PROVEN in id_list and runtime_ds), needed Charon/Aeneas on Lean v4.31, and committed generated Lean. Its last version is in 26cf24871.
