# Background CCH build: fold the relationship matrices, or keep the private copy? (DRAFT, not posted)

**The question.** CCH index builds will move to the native index's background builder (the `falkordb-index-build` pool). A build needs each selected relationship type's forward edges as one flat matrix: the stored matrix `m`, with the pending additions `dp` merged in and the pending deletions `dm` taken out. #3744 does this with `effective`, as C does with `Delta_Matrix_export`. When anything is pending, it builds a private merged copy and drops it after the build.

The alternative is a **fold**: the same merge, written back as the new stored matrix. The build then reads `m` directly, the result outlives the build, and `Effective` can go. A fold is the same logical data with different physical storage. So the builder would be changing graph storage, not only the index. Below is what that means under MVCC, with measurements. I'd like to raise it as its own PR after #3744, which keeps `Effective` for now.

## How the background builder works today (main eaa238efa)

- **Reading.** A batch reads a private fork of the published version (`new_build_version`, `online_build.rs:406`) and holds no lock while it reads (`online_build.rs:13-14`).
- **Publishing.** It publishes under the GIL (`src/graph_core.rs:1606-1614`) through `commit_indexer_batch` (`mvcc_graph.rs:262`), which picks one outcome:
  - **Same:** nothing committed since the fork, so the builder's own version is published.
  - **Swap or CatchUp:** a new fork of the published version gets the batch's index moved into it (`versioned.rs:551-584`). The builder's own storage is dropped.
  - **Discarded:** nothing is published.
- **Writer forks already in flight** take only the index across when they escalate (`rebase_onto_indexer_batches`, `versioned.rs:664`). Their storage stays their own.
- **The builder already changes storage a little.** `prepare` runs `fold_oversized_deltas` on the batch's version at publish, under the GIL (`mvcc_graph.rs:330-347`).

## Where a fold could happen

1. **On the published version: no.** Readers reach it without a lock, and its stored matrices must not change under them. See the comment at `mvcc_graph.rs:332-346`, and the copy-on-write rule at `cow.rs:48-59`.
2. **On the writer, before the build is scheduled.** This is correct, but it puts a rewrite of the whole stored matrix on the write path, for every write that triggers a rebuild. That costs about 14 ms for a 9M-edge relationship type (measured below). The fold rule (`versioned_matrix.rs:86-160`) exists to keep exactly that cost off writes.
3. **On the builder's own fork, during the build.** It must not run inside `prepare`, which holds the GIL. This is MVCC-safe: the fork belongs to the builder, and `Cow::replace` swaps the fork's own pointer without writing to the shared matrix. The fold survives only when the batch publishes as **Same**. Under Swap or CatchUp it's dropped. If a writer forked before the publish, its commit replaces the folded storage with its own unfolded storage. Correct, but the fold is lost.
4. **Keep the private copy (today).** The builder changes nothing, and the copy is dropped after the build.

## Measurements

Setup: a 1500 x 1500 grid, so 2.25M nodes and 8,994,000 edges of one type. One write is left pending: 1,000 deletions and 1,000 additions. Each variant ran 3 times in its own process, and times are medians. Memory is bytes in use across malloc (Rust and GraphBLAS together), on macOS with the optimized test build. The graph alone takes 710.5 MB. A "reader" is an older version of the tensor kept alive, as a query on the previous version would keep it.

| | copy (today) | fold on the fork | copy, reader holds old version | fold, reader holds old version |
|---|---|---|---|---|
| Fold | n/a | 14.3 ms | n/a | 14.6 ms |
| Input build | 76.5 ms | 61.3 ms | 76.9 ms | 61.2 ms |
| **Fold + build** | **76.5 ms** | **75.4 ms** | **76.9 ms** | **76.8 ms** |
| Second build of the same version | 75.2 ms | 64.3 ms | 72.0 ms | 76.1 ms |
| Peak during the fold | n/a | 1,112 MB | n/a | 1,112 MB |
| **Peak during the input build** | **1,130 MB** | **1,179 MB** | **1,130 MB** | **1,278 MB** |
| After everything is dropped | 710.5 MB | 741.6 MB | 710.5 MB | 741.6 MB |

What the table shows:

- **Time is a wash.** The fold costs about what the copy costs inside the build. Later builds of the same version skip the copy, which saved at most about 11 ms here; the reader runs were within noise.
- **The merge briefly needs about 400 MB above the graph, either way.** That's the fold's peak, and the copy runs the same merge inside the build.
- **A fold raises the build's peak.** The folded matrix lives through the whole build, while the copy is dropped early. A reader on the previous version keeps the old stored matrix (117 MB here) alive as well, so the peak rises by 148 MB.
- **A fold leaves 31 MB more in use afterwards.** It isn't the matrices: GraphBLAS reports the tensor at 171.0 MB before and 170.9 MB after. I haven't found where it goes.
- **`GRAPH.MEMORY USAGE` sees neither approach.** It counts only the published version (`src/commands/memory.rs:73-83`).

## Proposal

Keep the private copy (4) for the first background version. It needs no new MVCC rules: the builder changes nothing.

If we want to drop `Effective`, fold on the builder's fork (3) in a follow-up PR. It's correct, costs the same time, and costs up to one stored matrix of extra peak memory while readers hold the previous version. Rule out 1 and 2.

## Questions for the team

- Are we fine with the background builder publishing storage changes, beyond the `fold_oversized_deltas` it already runs at publish?
- On a graph that keeps writing, how often would a CCH build publish as Same? A build takes much longer than a range-index batch, and any write during it drops the fold.
- Does the catch-up story for a CCH index change the answer? That's rebuild from scratch (#3363) or scoped recustomization (#3366).
