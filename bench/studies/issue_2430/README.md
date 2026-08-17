# Issue #2430: the two-state multi-edge point read

A bound multi-edge point read lands in one of two cost states, selected by graph
size. Three hypotheses are refuted in the issue itself (scaling with `|me|`, more
ids walked, a latched-but-unflushed delta). This directory narrows what is left.

Two measurements, deliberately at different grains:

| | what it runs | result |
| --- | --- | --- |
| `engine_two_states.py` | the whole query, per pair | 8,725 → 9,943 instructions: a **step** of ~1,150 between 41k and 88k pairs |
| `graph/.../issue_2430_bench.rs` | the same read with no pipeline around it | 2,771 → 2,879: a **smooth drift** of ~108 across the same range |

**The fixture detail both depend on.** Node count is held at 1,000 while pairs
grow, so a bigger graph means *longer adjacency rows*, not more of them. An
earlier version of the Rust bench gave every pair its own row — holding row
length at 1 at every size — and so measured a perfectly flat cost and wrongly
cleared edge storage outright. Row length is exactly the variable a point read
can be sensitive to, since `GrB_Matrix_extractElement` searches within a row.

## What this establishes

Edge storage is **not** where the two states come from, on two independent
grounds:

1. **Magnitude.** The tensor's contribution across the whole range is ~108
   instructions; the engine-level effect is ~1,150. Edge storage accounts for at
   most a tenth of it.
2. **Shape.** The tensor's cost *drifts* smoothly and roughly logarithmically —
   which is what a binary search inside a lengthening row should do. The
   engine-level cost *steps*. A drift cannot produce a step.

And the storage format is constant throughout — `m` sparse, `dp`/`dm`/`me`
hypersparse at every size — so a GraphBLAS format switch is not the cause either.
That was the leading remaining hypothesis and it is now refuted.

## What is still open

Which pipeline stage steps, and why the selection is not monotonic in graph size
(the issue reports low/high/low/high/high; this fixture reproduces a single
crossing, so the non-monotonicity is fixture-sensitive and worth pinning down).
The tensor can be excluded from that search.

Run both:

    python3 bench/studies/issue_2430/engine_two_states.py target/release/libfalkordb.dylib 6650
    cargo test --release -p graph issue_2430 -- --ignored --nocapture --test-threads=1
