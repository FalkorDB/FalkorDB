# CCH on GraphBLAS: handoff

**As of 2026-11-13.** This branch hands the CCH (Customizable Contraction Hierarchies) redesign from Dvir's sessions to Gabriel's agents. It holds no engine code changes, only:
- the design and its research;
- the prototype reports and their code;
- a walkthrough of the design on a 20-node example;
- five Lean projects that check the design, and repros of the two bugs they confirmed.

**State in one paragraph.** The redesign replaces today's CchInput CSR copy and feral with four things. Feral is a Rust METIS port; its crates are an MIT clean-room reimplementation by John Kitchin, and FalkorDB/feral is our fork. The replacements are:
- our own nested dissection on GraphBLAS;
- a topology that is a shared base plus a copy-on-write delta;
- a write log that lets the native index builder build in batches under MVCC;
- a scoped recompute of the arcs a commit changes.

The design is on paper at revision 3.1 and has been reviewed twice on paper. Prototypes have tested its two riskiest assumptions, the read path and the ordering speed, and five Lean projects check its write log, supergraph, recompute, topology and the prototype's optimized passes. Decision D5 (how the ordering meets its time bar) blocks PR 6. No PR of the redesign's plan has been opened.

## Code state

- **`main` has only stubs for CCH:** `unimplemented!` bodies in `graph/src/index/falkordb/cch/mod.rs`, and a stub `input.rs`.
- **The working CCH is an open, linear PR stack**, each branch on the one before:
  `cch/05-inputs` → `06-graph-api` → `07-ddl` → `07b-ddl-replication` → `08b-weight-check` → `07c-background-build` → `11-partial` → `12a-chunked-weights` → `08-maint` → `08c-build-catch-up` → `09-procedure` → `10-rdb-v20`. The tip is `origin/cch/10-rdb-v20` at `484e36d4e`.
- **03 cites the code at `@08c`,** that is `origin/cch/08c-build-catch-up` at `31ca1b802`. It was the stack's tip when 03 was written; it no longer is.
- **Base the redesign's PR 1 on the stack's tip at the time,** unless the stack has merged by then.

## Read in this order

1. `design/03-design.md`, revision 3.1: the design. The note at its top lists what 3.1 changed. A few terms:
   - S2 adds the arcs a new edge needs;
   - S6 rebuilds the hierarchy as a new lineage from the current order;
   - S7 re-orders as a new generation;
   - n_r is the rank count;
   - M1–M8 are the measurements in §9, and D1–D8 the decisions at the end;
   - SHEM is METIS's sorted heavy-edge matching, and FM is Fiduccia–Mattheyses refinement.
2. `prototypes/00-results.md`: the first three reports' verdicts, and a **Change** list of amendments that are **not merged into 03 yet**. Its time verdicts are superseded by the later reports; its top note says how.
3. The team doc, https://claude.ai/artifact/1Xs7y3EE3kBMJU8woAv6dw: the same design for the team, with a status dropdown per decision. As of 2026-11-13, D3 is settled by measurement, D5 blocks, and the rest have a recommendation that waits for agreement.
4. The walkthrough, https://claude.ai/artifact/KWVvHG9rNkurf4R1n1BovP. Its source is in `walkthrough/`. It shows:
   - the design's objects;
   - the matrix becoming the index;
   - the build under concurrent writers;
   - an MVCC grid of what each version shares or copies.

   The algorithm step by step is in Dvir's CCH explainer, https://claude.ai/artifact/SEivswkaiNdpjVejgzdyzd; `walkthrough/reference-style.html` is a local copy of it.
5. `prototypes/*.md`: one report per measurement (below).
6. `research/`: the briefs behind the design. u1–u4 cover today's code and C, and r1–r3 the literature; r3's checks are in `research/check/`. `design/01`, `design/02` and `design/design-*.md` are earlier rounds, kept for history. 03's `[FOLD]` source is `../cch-background-fold/discussion.md`.

The three claude.ai pages are Dvir's. Ask him to share them if a link doesn't open.

## What has been measured

All on SNAP roadNet-CA, PA and TX, on an Apple M5 laptop, unless a report says otherwise. None used the stack's 1.54M-node California.

| Question | Answer | Source |
|---|---|---|
| Can queries read the topology through GraphBLAS row iterators? | No: they fail the read gates | `prototypes/m2-read-paths.md` |
| Rust arrays loaded read-only into GraphBLAS (`GxB_IS_READONLY`)? | Works with no copy. Flat slices tie today's code: p99 524–531 µs against the 526 µs gate, with `:edge-c` not measured. So this is not a clean pass. Caveats: a build without LTO, the read-only load measured only on 10 threads, and concurrent readers untested | `m2-read-paths.md` |
| Mutual-max matching against feral's SHEM, on order quality | Better than SHEM in all 15 paired runs, but conditional: it fails without the weight cap, the 2-hop pass, or at least 3 rounds. Still +5.2–6.0 % arcs and +6.4–11.1 % triangles over C METIS | `m1-matching-quality.md`, `00-results.md` |
| GraphBLAS coarsening time, first prototype | 13.17 s at 1 thread against feral's whole ordering at 6.97 s: fails | `m1-graphblas-cost.md` |
| Reuse depth 0's levels at deeper depths (D5 option b) | Plain reuse fails on quality. Reusing until a cell's weight cap breaks (the "cap stop") keeps quality, under the quality harness's stop rule | `m1-reuse-quality.md`, `m1-reuse-cost.md`, `d5-b-verdict.md` |
| The cap stop after removing copies and rescans | Coarsening plus stand-ins for uncoarsening went from 7.22 to 2.30 s on CA at 1 thread: 0.33× feral, 0.45× C METIS. At 4 threads, 1.20 s. The 2.30 s is 1.23 s of Rust passes, 0.52 s of GraphBLAS kernels and 0.55 s of stand-ins. Output is identical to the unoptimized bench (57/57 hashes); an independent re-run reproduced it on CA only. **The order quality of this variant is unmeasured:** its stop rule is looser than the quality harness's | `m1-optimize.md`: **read its corrections note first** |

Most of the optimized prototype's work runs as Rust passes over GraphBLAS's own CSR arrays: unload in O(1), one host pass, load back in O(1). Only the matching rounds are still GraphBLAS kernels. With the quality harness's floor, the cap stop costs at least 6 % more; that rule's per-cell cap and its five fresh runs were not measured.

## Open work, in the order we would do it

1. **Fix 03's text.** These are cheap, and PR 4's spec depends on them.
   - Fold in the Lean findings (see Proofs): F1–F5, T2, T3, N1, N2, R-W1, and two wording fixes from the topology proof (P holds "for r < n_r′"; §2's "cleared by a fold" should match §6, which keeps the arcs with seq > F). F1 and T2 break the design if read literally; each fix is one sentence.
   - Merge `00-results.md`'s Change list:
     - the mandatory weight cap;
     - the anchor-grouping 2-hop pass;
     - the 8-round cap;
     - the larger level budget;
     - named user operators for the JIT;
     - building Cᵀ;
     - dropping "skip coarsening below a floor" as the cost fix.
   - Record the optimization pass's main result in §3: most of the ordering's work is Rust passes over GraphBLAS's unloaded arrays. 03 §1 says every cell is "processed together by GraphBLAS calls".
   - Settle the places where 03 reads two ways. The walkthrough lists them under "Where the design reads two ways":
     - n_r in §2 against §3/§4;
     - §6 against §7 on the node of the writer that sets upkeep;
     - what a writer does when a trigger is already met with no upkeep;
     - S6's undefined n and dead ids, and whether "arcs/8" is counted before or after the commit;
     - whether a build-time top rank counts toward S7's "top ranks used".
2. **Gather D5's data. The team decides.**
   - First, time feral by step. Its time has never been split, so it is unknown whether the host steps (first separator, FM, minimum degree, the tree) fit in what is left on CA at 1 thread: 4.58 s against feral, or 2.76 s against C METIS, which is the project's bar.
   - Measure the order quality of the optimized cap stop, under the quality harness's exact rule: its per-cell cap and best-of-5 fresh runs.
   - Consider moving matching to the host: about 0.32 s of depth 0 on CA.
   - Carry two caveats into the module work.
     - The O(1) hand-over of Rust buffers to GraphBLAS relies on both sides using one allocator. In the module, GraphBLAS allocates with RedisAlloc, so the arrays must come from it.
     - The Rust passes trust their input, and need debug assertions.
   - The team doc frames D5 as "how the ordering meets its time bar", with four options: (a) more builder threads, (b) reuse, (c) mutual matching in feral, (d) keep feral. 03's own D5 text still says "builder GraphBLAS threads". Align them.
3. **Fix R-F1 on the stack, at `cch/08-maint`** (the first branch with this `partial.rs`): compare `to_bits()`. The test and the fix are in `repro/r-f1-signed-zero/`; both patches apply from `cch/08-maint` through `cch/10-rdb-v20`. Check P-seed too: the full path takes a pair's minimum with GraphBLAS's MIN monoid and the scoped path with `cheapest_edge`'s `<`, so parallel +0.0 and −0.0 edges may give different zeros. That is R-F1's class on the seed side; it is unconfirmed, and `to_bits()` alone may not give bit identity.
4. **Start PR 1 (GraphBLAS wrappers) and PR 2 (`ChunkedArr`) alongside D5.** Neither depends on it. PR 3 (base + delta topology, still fed by feral) gates on M2 and M3; it must also handle T1 (sort S2's Down batch and the fold's arcs before `insert_batch` or `from_sorted`) and read relax's slots by id after S2 (`B2_positional_id_breaks_after_s2`). The full plan is 03 §9, PRs 1–10.
5. **The measurements 03 §9 lists that nobody has run:**
   - M1 on the 1.54M-node California, a 9M-edge graph, Linux and 16 threads, with query p99 against `:edge-c`, and memory;
   - M2's p99 against `:edge-c`;
   - M3 to M8.

## Prototype code

`prototypes/code/` holds every prototype's code (5.6 MB). Each prototype comes as a patch against its base commit, with its scripts and small results; raw logs are packed as `.tar.xz`. Each folder's README says how to rebuild and re-run it.
- Most patches apply to `7df8e5887`, an older main. `m2-read-paths` applies to `31ca1b802` (`cch/08c-build-catch-up`), which is not on main.
- `m1-optimize` was rebuilt from its patch, and matched its saved hashes 13/13 on PA.
- `m1-matching` is a source tree that builds as copied, but needs network access: it fetches FalkorDB/feral at `f127189` and `flate2`. It includes a copy of feral's crates, which are MIT-licensed.
- `cmetis` is the C METIS driver. METIS and GKlib are cloned at build time, pinned to master's submodule commits.

The SNAP data is not committed. The README gives its URLs, each file's sha256, and the converters. The scripts hard-code the original scratch path; each README names the variable to change. The originals live in temporary folders that are wiped on reboot, so treat this copy as the only one.

## The walkthrough

`walkthrough/walkthrough.py` is a stdlib Python model of 03. It runs the design on a 20-node example and checks 20 invariants. Each invariant has deliberate breaks that must fail, 33 breaks in all. It writes `example.json`, and `walkthrough/page-src/build.py` turns that into `index.html`. To rebuild and check:

```bash
cd .handover/cch-redesign/walkthrough
python3 walkthrough.py          # prints "Invariants: 20 of 20 pass; 33 deliberate breaks, all fail"
python3 page-src/build.py       # writes index.html
for c in checks/c*.py; do python3 "$c"; done   # independent checks; c1–c5 and c9 print "problems: 0"
```

`SIMPLIFICATIONS.md` lists every place the model does less than 03. `render-final/` keeps only the headless screenshot scripts (`*.mjs`, `probe.js`, `wrap.py`), not their output. The published page belongs to Dvir: to change it, ask him for edit access, or publish a new page.

## Proofs

Five Lean 4 projects check the design before its code exists. Where real code exists they model it line by line: today's `order.rs`, `relax` and `seed`'s allocation, the stack's `partial.rs` (its scoped path), and the prototype patch's kept passes, apart from `book` and `npar`. Each project's root `.lean` header is its full report: the code-line or 03-section table, the theorems, the readings it chose where 03 is ambiguous, its findings and its gaps. `proofs/FINDINGS.md`, section "CCH redesign review (2026-11-13)", lists every finding in one place, including the unconfirmed ones.

Each project was reviewed by an independent agent, the authors fixed every finding, and the reviewers then verified the fixes. `bash proofs/lean_ci.sh` passes for all five, and `bash proofs/coverage.sh` reports 0 unmatched rows. The reviewers ran `#print axioms` on every headline theorem: each uses only `propext`, `Classical.choice` and `Quot.sound`. A clean checkout of this commit builds all five.

| Project | What it proves |
|---|---|
| `cch_redesign_review` | The write log, publish rules, writer rebase, hand-off and Built-entry jobs (03 §5–§7), over every interleaving. No logged write is lost (`no_lost_write`), a Building entry serves only the fallback, and retention, stale jobs and the bound all hold. `fitted_job_only_writers_between`, `job_window_frozen` and `upkeep_writer_not_overtaken`: a fitted job's window keeps its generation and state, and only writers ran in it, none of them rebasing. Witnesses show the key transitions are reachable: a fold publish, writer rebase rows 2 (within and over budget) and 3, a CatchUp relink, and an adoption Restart. |
| `cch_supergraph_review` | The masked skeleton read is exactly the live pattern, and B′ = (B ∖ D) ∪ A is exactly the live pattern at V_sg, linked to `no_lost_write`. Today's `chordal_supergraph` is exactly the elimination game, chordal and panic-free. Ids are row-major, and I7 holds. **All 20 functions of `order.rs` are now PROVEN** (they were 0). |
| `cch_recompute_review` | Today's `relax` computes the triangle minimum with the strict-`<` middle, and is **PROVEN** through a bridge to the supergraph proof. The scoped recompute gives the full customization's weights and middles (`scoped_agrees`), and Lemma A holds. For the stack's `f64` code it gives the `==`-agreement form (`recustomize_seeded_agrees`). |
| `cch_topology_review` | Base + delta: merged rows strictly ascending (I3′); ids unique, below `arc_capacity` and never reused; I7; I1′ and P under rank allocation, node delete and id reuse. A write in one version leaves every other version's reads unchanged (`WeightArr::set`, the cow_btree delta). A fold changes nothing a reader sees. It feeds the recompute proof at every state of a lineage (`lineage_scoped_agrees`). |
| `cch_coarsen_review` | Each kept optimized pass of the ordering prototype equals the GraphBLAS formula it replaced, on every valid input: contraction, numbering and Cᵀ, the split, keys and cap, the reuse level, the gather and the threaded assembly. `book` is covered only at the level of the numbering's decisions, and `npar` is not modelled. Rows come out sorted, and matching gives a valid capped pairing. Ids stay under 32 bits up to (depth + 1)·n, given `nparts ≤ n`; values on contracted levels stay at most nnz(A_0); reuse-level values are bounded only relative to their parent. |

Lake dependencies: supergraph requires redesign; recompute requires supergraph; topology requires recompute (and through it the other two) and `cow_btree`; coarsen stands alone, requiring only `graphblas_wrappers`. There is no cycle.

Coverage of today's CCH code after this work: `order.rs` 20/20 PROVEN, `cch/mod.rs` 17/20, `customize.rs` 1/13 (`relax`; `seed`, `customize` and `validate` stay NOT COVERED), `query.rs` 0/8, `nd.rs` 0/3.

**Still open.** These premises are named in the headers and not proved:
- chordality after S2, that Davis–Hager's closure ends chordal;
- that writers log every pair whose liveness their commit changes (`Records`), which PR 4 must make true;
- the composition of the topology and write-log models. Their state types differ, so the step from the write-log model's writers to the topology model's `WLin` is argued. Everything else about a fold's window is proved: the generation stays fixed, only writers change this index's entry, and no writer rebases (`fitted_job_only_writer_steps`, `fitted_window_writer_no_rebase`);
- P-seed, that the seed function is pure and gives the same bits on both paths (recompute; see Open work item 3);
- FreshOk's injectivity of π and its held endpoints (topology), and `halloc` in `bprime_rank` (supergraph);
- `hres` from reachability;
- Lemma B (S5);
- the prototype's numbering facts behind `B2Pre`, `BookPre` and the gathered level's `gw > 0` (argued, not proved), the bounds on B2 offsets, `npar`, and the numbering line by line;
- liveness: convergence and D2.

**Confirmed bugs, with repros in `repro/`:**
- **R-F1, on the stack (`partial.rs:221-236`, from `cch/08-maint` through `cch/10-rdb-v20`).** The scoped recompute tests "changed" with `f64 !=`, so a seed that only flips the sign of a zero is never written or propagated. The result is `==`-equal to a full customization but not bit-identical. `repro/r-f1-signed-zero/` has a failing test and the `to_bits()` fix, with which all 7 `partial.rs` tests pass.
- **H2, in the prototype.** `book` works only with `pk` and `hb2` on (`repro/h2-switch-dependency/`); that `npar` has the same dependency is read from the code, not run. This matters if module code keeps any of the old paths.

**Findings in 03's text,** each with a Lean trace unless marked:
- **F1, a lost write if read literally.** Past the bound, the writer "clears upkeep and the log".
- **F2.** A segment that hands off before its writer escalates can revert a published S6.
- **F3, F4.** Wording: the CatchUp pair count, and when the bound is checked.
- **F5.** A fold or S6 publish must also move `batches_published`.
- **T2, I1′ broken if read literally.** S6's rank release must also clear `rank_of` and free the rank.
- **T3, found by reading** (the model's unbounded rows hide it, reading R16). The base is n_r wide and the delta n_r′, so a fold's new base needs a longer `Ap`; 03 is silent.
- **N1.** The full-seed pass's union equals the live value with no premise, though 03 cites the tensor invariants for it.
- **N2.** Slabs must be read at or before V_sg; 03 says "any" snapshot.
- **N3.** `Records` (writers log every pair they change) carries the weight of B′'s exactness: without it, a created edge that is not logged lacks an arc, and a deleted one keeps its arc.
- **R-W1.** The middle is the first x that attains the minimum, not "the first x to win a strict `<`".

**Notes on existing code,** not on 03:
- **T1, for PR 3.** S2's Down batch and the fold's arcs must be sorted before `insert_batch` or `from_sorted` (`cow_btree/mod.rs:204-207, 250-253`).
- **H1, H3, for the module, from the prototype.** A dead-column guard in the contraction pass; vertex weights ≥ 1 for the numbering patch.
- **N4, on main, unreachable today.** `Topology::from_order` does not check that the skeleton is mirrored; `nd.rs:54` checks it before every call.

**When the design lands.** `cch_redesign_review/STALE_WHEN_LANDED.md` lists the PROVEN rows each PR changes:
- the 23 rows this work made PROVEN (`order.rs`, `id_index`/`index_id`, `relax`): their lines move on the stack, and PR 3 changes `relax`'s slot reads and the `Topology` accessors;
- `writer_rebase` and `indexer_catch_up` (PR 4);
- `overtaken`, the fit and `indexer_apply` (PR 9);
- `Csr::*` (removed in PR 7).

The `order.rs` model then moves from `cch_supergraph_review` to an owning project, as its header says.

## Conventions this work followed

- Every number cites its source: a file, a log or a report. Anything not measured says "unmeasured" or "derived".
- Benchmarks go through the `bench` and `profile` skills. The bar is p99 at or below the C engine (`falkordb/falkordb-server:edge-c`, not `:edge`), with memory matching C.
- `graph` stays host-agnostic: no `RedisModule_*` calls under `graph/src/`.
- Full readable names. METIS-style names (`xadj`, `adjncy`, `perm`) are renamed at the port boundary.
- Every PR closes an issue, lists its new functions in `proofs/*/COVERAGE.tsv`, and gets a local review by an independent agent before each push.
- PR and issue text describes the change. Risks and caveats go in the conversation, not on GitHub.
