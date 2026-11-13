# D5 option (b): judge's verdict

**Superseded on time by `m1-optimize.md` (2026-11-13).** After the optimization pass, the cap stop's coarsening on CA takes 2.30 s at 1 thread (0.33× feral, 0.45× C METIS) and 1.20 s at 4 threads, with identical output. So "no room for host steps at 1 thread" below no longer holds: 4.58 s are left against feral and 2.76 s against C METIS. Whether the host steps fit is unmeasured, and that variant's order quality is unmeasured; see `m1-optimize.md`'s corrections note.

**Update after this verdict (main session, 2026-11-13).** capstop was then measured in GraphBLAS, not derived. On CA, PA and TX it takes 0.98–1.10× feral at 1 thread and 0.45–0.58× at 4 threads, GraphBLAS calls only (`m1-reuse-cost.md`, addendum; logs `scratchpad/m1rcB/logs/main_*.log`). The conclusion stands: no room for host steps at 1 thread, comfortable at 4.

**Verdict: (b) does not answer D5.** Drop it as a way to keep one builder thread. Keep B2 with a cap stop ("capstop") only as a cost cut if the team picks (a), more threads.
- **(b) as specified fails the quality gate:** all three metrics at or below feral in 1/15 seeds, against 12/15 for R (J).
- **capstop keeps quality (13/15, J) but not time:** its GraphBLAS part alone is ≈ 1.16–1.19× feral at 1 thread (d).
- **Even B2 alone leaves almost nothing:** 0.92–0.94× feral (C) leaves 0.28–0.42 s (d) for separators, FM, minimum degree and the tree.
- **At 4 threads it helps:** capstop ≈ 0.53–0.55× feral, R 0.80–0.81× (d, C).

Sources: **Q** = `m1-reuse-quality.md` (`out/main.txt`, `out/diag.txt`); **C** = `m1-reuse-cost.md` (`logs/main_*_fine.log`); **J** = my recount of Q's logs (`scratchpad/d5judge/`); **d** = derived.

| Graph | Variant | GraphBLAS, 1 thr: s (× feral) | 4 thr | Plain-Rust order (s) | Arcs | Triangles | Height | Seeds, all 3 ≤ feral |
|---|---|---|---|---|---|---|---|---|
| CA | feral | 7.08, whole ordering (C) | — | 7.20 (Q) | 12.63M (Q) | 219.0M | 941 | — |
| CA | R | 14.88 (2.10) (C) | 5.69 (0.80) | 8.65 (Q) | 12.52M (Q) | 208.0M | 900 | 4/5 (J) |
| CA | (b), B2 | 6.65 (0.94) (C) | 3.16 (0.45) | 5.72 (Q) | 12.71M (Q) | 232.8M | 955 | 0/5 (J) |
| CA | capstop | ≈ 8.4 (1.19) (d) | ≈ 3.9 (0.54) (d) | 6.71 (Q, 1 run) | 12.60M (Q) | 215.7M | 877 (J) | 4/5 (J) |
| PA | feral | 3.60 (C) | — | 3.62 | 7.69M | 186.3M | 878 | — |
| PA | R | 7.36 (2.04) | 2.93 (0.81) | 4.59 | 7.60M | 174.3M | 854 | 4/5 |
| PA | (b), B2 | 3.32 (0.92) | 1.64 (0.46) | 3.08 | 7.67M | 186.8M | 882 | 1/5 |
| PA | capstop | ≈ 4.2 (1.16) (d) | ≈ 2.0 (0.55) (d) | 3.58 | 7.61M | 174.9M | 855 | 4/5 |
| TX | feral | 4.53 (C) | — | 4.69 | 8.46M | 113.0M | 702 | — |
| TX | R | 9.55 (2.11) | 3.64 (0.80) | 5.75 | 8.40M | 111.2M | 691 | 4/5 |
| TX | (b), B2 | 4.21 (0.93) | 1.97 (0.44) | 3.90 | 8.44M | 115.0M | 710 | 0/5 |
| TX | capstop | ≈ 5.3 (1.18) (d) | ≈ 2.4 (0.53) (d) | 4.43 | 8.39M | 109.3M | 656 | 5/5 |

Quality: median of 5 seeds. capstop cost = B2 + 14.7–14.8 % (Q) of R's coarsening calls (C).

## Attempts to refute

- **Built as specified: yes.** No matching after depth 0, and every check fired when broken (Q, C). B1 = B2 on 653,998 levels (Q) and 9,174 (C; recounted, J). SHEM and R reproduce the earlier report (Q).
- **Like with like: mostly.** One process and one binary per graph (same md5 in all six logs, J), JIT RUN, R re-run, load 1.6–5.6 (C). Two biases pull opposite ways: the bench's labeller needs 19–23 depths (C) where feral stops by 15–16 (Q), inflating every total; its R has no best-of-5, understating R and capstop.
- **Straddler share: irrelevant.** The bench's is 2–3.7× the quality stage's (C), but share-driven calls are 0.34 of 6.65 s; B2 ≥ 6.31 s, 0.89× feral (C, d).
- **Weight cause: plausible, not proven.** capstop also restores fresh best-of-5 matching (Q).

## What it means for D5

- Remove (b) as an alternative to (a). Under (a), use capstop in B2 form: ≈ 3.9 s against R's 5.69 s on CA at 4 threads (d, C).
- New for (c): in plain Rust, capstop orders in 0.93–0.99× SHEM's time (Q, 1 run; d) with near-R quality, removing R's 1.20–1.27× slowdown (Q, d).

## Unmeasured

- Host-step time (separators, FM, minimum degree, tree, peeling, components): feral's own time has never been split, so no variant's room is known to suffice.
- capstop in GraphBLAS with best-of-5, and which of its two changes carries quality.
- Single-stage hierarchy quality, round cap 8, the 1.54M-node California, a 9M-edge graph, Linux, 16 threads, query p99, memory.

Recount break: one raised TX capstop height turns 5/5 into 4/5 (`scratchpad/d5judge/diag_broken.txt`).
