# CCH prototypes: results against the design's gates

**Superseded on time (2026-11-13).** This file covers the first three reports only. Its M1 time verdict ("Fail at 1 thread") and its "(b) reuse … unmeasured" were overtaken by later reports:
- `m1-reuse-quality.md` and `m1-reuse-cost.md` measured reuse;
- `d5-b-verdict.md` judged it;
- `m1-optimize.md` cut the cap stop's coarsening on CA at 1 thread to 2.30 s, 0.33× feral. Read its corrections note: that variant's order quality is unmeasured, and the host steps are not timed.

The quality verdicts and the Change list below still stand.

All three prototypes ran on one Apple M5 laptop under macOS, with other sessions running, on SNAP roadNet-CA, PA and TX. None of them used the stack's 1.54M-node California. The gates set on that graph (3.62 s, 219 ms) are therefore read as ratios to feral or today's code on the same graph. This is evidence, not proof. `03-design.md` was written before these reports, so none of the changes below are in it yet.

Reports:
- [Q] `m1-matching-quality.md`
- [C] `m1-graphblas-cost.md`
- [R] `m2-read-paths.md`

## Verdicts

| Gate (03 §9) | Verdict | Basis |
|---|---|---|
| M1 quality (arcs, triangles, height ≤ feral) | **Pass, conditional** | Mutual-max has fewer arcs and fewer triangles than SHEM in all 15 paired runs, and lower height in 4 of 5 per graph. It is still +5.2–6.0 % arcs and +6.4–11.1 % triangles over C METIS. It fails without the weight cap, without the parallel 2-hop pass, or with fewer than 3 rounds [Q] |
| M1 cost (time ≤ feral) | **Fail at 1 thread** | The GraphBLAS calls alone take 13.17 s on CA, against 6.97 s for feral's whole ordering (1.9×, the same on PA and TX). At 4 threads: 4.79 s, which leaves about 2.2 s for host work (derived) [C]. Mutual-max in plain Rust is also 21–27 % slower than feral [Q] |
| M1 identical order across threads | **Partial** | The coarsening hierarchy matched at 1 and 4 threads [C]. Full orders were not compared |
| M2 query p99 ≤ 526 µs | **Iterators fail** | GxB iterators take 897–1,274 µs. Flat arrays take 524–531 µs, at the line, like today's 524–534 µs [R] |
| M2 customize ≤ 219 ms | **Iterators fail** | Iterators take 1.8–2.4× today's time. Flat arrays match today [R] |
| M2 read-only load | **Works (v10.5.0)** | No copy, writes refused, transpose and masked mxm correct. Measured at 10 threads only; concurrent readers untested [R] |
| Unmeasured | — | M1: query p99 with the new order, peak memory, the 9M-edge graph, 16 threads. M2: `:edge-c` p99, scoped µs/arc, the full-seed pass, valued against iso `Down` |

## What it means for the design

**Keep.**
- **D3 option (b):** Rust-owned arrays, loaded read-only and read as slices. M2 rules out iterators. A naming trap: the report's "(a)" is these flat arrays, and its "(b1)/(b2)" are D3's option (a). PR 3's other M2 gates are still open.
- **Mutual-max matching**, on quality.

**Change.**
1. **§3:** make the weight cap mandatory: `vwgt[u]+vwgt[v] ≤ 1.5·tvwgt/CoarsenTo` (METIS `coarsen.c:38`, per [Q]). Without it, coarse vertices reach 195k in weight instead of about 29k.
2. **§3:** specify the 2-hop pass as anchor grouping. Feral's sequential pass fails.
3. **§3:** cap matching at 8 rounds. Caps of 1 and 2 fail. Measured on CA only.
4. **§3:** budget about twice as many coarsening levels (30 against 16 on CA's largest component).
5. **PR 1:** create user operators with `GxB_IndexUnaryOp_new(…, name, defn)`. As 03 creates them they are `jit: undefined`, so the planned PreJIT harvest can never specialize them. Generic kernels take 27 % of depth 0 [C].
6. **PR 1:** build `Cᵀ`, not `C` (−153 ms per depth, measured [C]).
7. **§3:** drop "skip coarsening below a floor" as the cost fix. The time is per entry, in the large levels of every depth, because each depth re-coarsens the whole graph. Deep levels cost under 0.5 ms [C].

**Decisions for the team.**
- **D5 against GraphBLAS ND.** At 1 thread, GraphBLAS ND fails the time gate. Options:
  - (a) more builder threads, gated by M7's read p99 during a build;
  - (b) reuse each depth's hierarchy at the next depth (the biggest lever [C] names; unmeasured);
  - (c) put mutual-max into feral on the host (better quality, 21–27 % slower);
  - (d) keep feral (03 §10.3).

  The team should choose which to measure before PR 6.
- **Quality against time.** Mutual-max gives 0.1–1.7 % fewer arcs and 0.2–10.9 % fewer triangles. In both forms it orders more slowly than feral, and ordering is paid on every build and, under D4, on every load. Whether that trade is worth it depends on query p99 and customization time with the new order, and neither has been measured.

## How round 2 changed the design

03 re-checked all 23 round-2 issues against @08c and found all 23 real. It replaced seven proposed fixes and rejected none (03 App. B).
- **Log:**
  - Overflow is a log node with its own seq, not a flag (V1.1, fatal).
  - Consumers free what they consumed.
  - A Built entry keeps no log unless a job is running.
  - Pairs are `Arc`, and the log counts in `memory_usage`.
- **Hooks and rebase:** the hooks read the log from the entries, not `Written`. Rebase has a rule for every (fork, published) state, which closes the lost write in a Building → Building rebase (V2.3).
- **Built-entry jobs (fold, S6):** they get a ticket, a catch-up and a rebase, and they count toward `overtaken` (D6). The fold keeps the published metric. S7 is a new generation.
- **Ranks:**
  - Pool ranks have no holder.
  - Rank allocation runs before classification in every resolver, including Supergraph, which now recomputes the etree and counts over B′.
  - A new node gets a pool rank only at degree ≤ 1.
- **Full-seed pass:** it reads `m⟨¬dm∧¬dp⟩ ∪ dp` with `me` merged, seeds both directions, and writes by arc id.
- **Inline hand-off and lock budget:** the hand-off takes a new generation, and B_gil is one budget per lock hold.

None of the prototypes tested any of this; they measured ordering and read paths only. The round-2 design holds on paper until the M5 cases in 03 §9 pass, each with a deliberate break. Its additions to the build path are unmeasured: the recomputation over B′ and the by-id seed scatter.

**Still needed before any gate closes:**
- `:edge-c` p99 on roadNet-CA;
- the stack's California and a 9M-edge graph;
- Linux and LTO builds;
- the read-only load at 1 thread, and the ordering at 16 threads;
- ordering peak memory.

## Prototype code

`code/` holds the code behind every report here: patches against their base commits, or source trees, with their scripts and small results. `code/README.md` indexes them. It also gives the SNAP data's URLs and checksums, the converters that turn the data into what each bench reads, and what was checked when the code was saved. Each subfolder's README says how to rebuild and re-run that prototype.
