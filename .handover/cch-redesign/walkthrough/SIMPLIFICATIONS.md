# Where the toy model is simpler than the design

`walkthrough.py` runs every step of `../design/03-design.md` (revision 3.1, with the prototype "Change" list in `../prototypes/00-results.md`) on one 20-id example. It is a model of the design's rules, not of FalkorDB's code, and nothing in it is measured. Below is every place it does less than the design says. Each item gives what the toy does, and why the step still shows what it is meant to show.

## Everywhere

1. **GraphBLAS calls are host loops.** Matrices are Python dicts and vectors are lists. Each step lists the call as the design writes it, and the toy computes the same result by hand. There is no JIT, no PreJIT harvest, no `GxB_Context`, no thread budget, and no read-only Container load or unload.
   *Why it is fine:* a call's output depends only on its inputs. The page shows the inputs and outputs, and they are the same.
2. **Integers, not 32-bit arrays.** Ranks, ids and weights are Python integers, and +∞ is a float. Weights are whole numbers, so every sum is exact and "bit for bit" comparisons are exact equality.
   *Why it is fine:* with whole-number weights, f64 sums are exact too, up to 2⁵³.
3. **One index.** The example has one CCH index and no other indexes.
   *Why it is fine:* a batch publish can never be `Swap`, which needs another index's batch, and the step says so. B_gil's "one budget shared by every index in a lock hold" holds trivially.

## A. Storage

4. **Only some tensor states occur.** The example reaches states A, B, D, G and H of `tensor.rs`. No pair has two edges of one type, so the multi-edge layer `me` is empty, and the backward matrix `mt` is not modelled.
   *Why it is fine:* the seed pass's `me` merge only matters for multi-edge pairs, and the brief asked for the multi-edge layer "if any".
5. **Deltas are never flushed.** `dp` and `dm` keep growing through the example, and nothing folds them into `m`.
   *Why it is fine:* the masked reads must be correct on unflushed deltas, so leaving them in place exercises exactly that.
6. **Relationship ids are never reused.** Node ids are reused, lowest first, as `graph/src/graph/id_space.rs:75` says.
   *Why it is fine:* no step depends on a reused relationship id.

## B. Ordering

7. **The matching salt is fixed.** Each level's salt is `fmix32(1 + 64·depth + level)`, a fixed stand-in for the per-level random draw in m1-matching-quality.
   *Why it is fine:* the rules (keys, synchronous rounds, the cap, mutual pairs) are unchanged, and a fixed salt makes the run repeatable.
8. **How a proposal's target is decoded.** The design gives `y⟨u⟩ = K MAX_FIRST u` and mutual pairs via `GxB_Vector_extract_Vector`, but does not say how the proposed column is recovered from `y`. The toy un-scrambles the key's low 32 bits, which works because `fmix32` is a bijection.
   *Why it is fine:* any way of recovering the argmax gives the same target.
9. **The cap mask is computed on the host.** The design says only that "a mask drops candidate pairs" over the cap.
   *Why it is fine:* the set of allowed pairs is the same.
10. **The 2-hop pass is checked per cell, and never fires on this example.** On this graph every matching shrinks its cell by at least 15 %, under all 600 salt bases the script tries. Forcing the pass on every level pairs nothing: the one shared anchor (depth 0, level 2) would break the weight cap. So step B5.h runs the same code on a side example, a 7-vertex star, which is not part of the order.
    *Why it is fine:* the side example uses the same function and the same rules. The page labels it as a side example.
11. **Coarsening stops early.** At toy scale the depth-0 cell stops at 4 vertices (the design's floor is 100), and those 4 coarse vertices all touch each other.
    *Why it is fine:* this is the toy floor, not a change to the rule. Its effect is that the first separator leaves one side empty and node FM restores the balance while uncoarsening. B6 and B7 both say so.
12. **The first separator is simpler than METIS's.** The design runs 7 greedy-growing trials with edge FM, then takes the lighter of METIS's boundary with two-sided and one-sided FM and a König cover. The toy does less:
    - It runs one trial per coarse vertex (4 here, at most 7), and grows side 0 breadth-first by edge-cut gain until it holds half the weight.
    - It does not refine the edge bisection.
    - It takes the lighter boundary as the separator and runs node FM.
    - It keeps the best trial by: balance kept, then separator weight, then imbalance.

    *Why it is fine:* the step's output is the same kind of thing, a label per coarse vertex from the best of several trials, and it feeds uncoarsening the same way.
13. **Node FM is a plain two-sided pass.**
    - Candidates are separator vertices only, as the design requires.
    - A moved vertex is locked.
    - A pass stops after 3 moves without improvement and rolls back to the best state.
    - Passes repeat until one keeps nothing.
    - Gains are recomputed after every move; there are no bucket queues.
    - The balance limit (max side ≤ 1.2 × cell weight / 2) and the 3-move stop are toy choices; the design fixes neither.
    - A move over the limit is allowed only if it shrinks the heavier side.
    - There is no best-of-5 at the top depths and no `mindiff` tie-break.

    *Why it is fine:* the step shows refinement seeded from the boundary, which is the design's point (against feral's full rescan).
14. **The order within a separator.** Separator nodes take the top ranks of their cell, lowest id first. The design does not fix the order inside a separator.
    *Why it is fine:* any order inside the separator is valid.
15. **Leaves use plain exact minimum degree.** Degrees are recomputed after each elimination, with ties to the lowest id. There are no 128-bit bitsets.
    *Why it is fine:* it gives the order an exact minimum degree defines.
16. **Peeling and components run on the host.** Peeling is a host loop that mirrors the masked `mxv`; with 2 rounds, the design's host pass after 64 rounds never applies. Components come from a host search, not `LAGr_ConnectedComponents`, labelled by the smallest id.
    *Why it is fine:* the rounds, ranks and labels are the same.
17. **The postorder relabel is applied to π.** The design relabels B's private arrays by q. The postorder takes roots in ascending rank, and children in ascending rank.
    *Why it is fine:* B = S(π,π) is the same matrix either way.

## C. Supergraph

18. **Column counts walk each row's subtree.** The design uses Gilbert–Ng–Peyton counts; the toy walks each row's subtree, which costs O(arcs). C1's call says so. C1's B is B11's one extract, relabelled by q; there is no second extract.
    *Why it is fine:* the counts are the same numbers. G13 checks that they equal the game's arc count, and that the check fails when the counts are not recomputed over B′.
19. **CSR layout.** `Ap` has one entry per rank plus one, and row i holds rank i + 1. Down.base is transposed on the host.
    *Why it is fine:* the ids, the transposition and the row-major numbering are what the step shows.

## D. Weights

20. **Relax runs once.** The toy runs the whole relax at once, then reports how its triangles split into two batches by apex rank.
    *Why it is fine:* the sweep is frozen at V_c, so where one batch ends cannot change a value.
21. **The seed pass's slabs.** It uses rows 0–9, rows 10–19, and a third slab for ids 20 and up, which exist because node 20 was created after the build started.
    *Why it is fine:* the design's seed pass covers every node id at V_c, so it needs that third slab too.
22. **Seeds are read per pair.** In the scoped recompute, each popped arc's seed is the smallest live weight per direction across the indexed types (`PairWeights`).
    *Why it is fine:* this is the design's rule. With no multi-edge pairs, the `me` merge has nothing to add.

## E. The batched build

23. **The timeline is a scripted sequence.** Batches and writers are a fixed list of events. Each records:
    - its fork version, its publish outcome, what it consumed, and the published log after it;
    - the log, kept as a Python list, newest first;
    - a relink as new node objects that share the pairs list.

    These paths are not exercised:
    - an Overflow node, because the bound is never reached;
    - writer rebase, because no writer's query spans a batch publish;
    - `Absorbed::Restart` at adoption, because the adoption fits B_gil.

    *Why it is fine:* the brief's timeline (v5 to v9) has none of these cases. Every rule it does use (Same, CatchUp, relink by seq, consumption, adoption within budget) is applied by seq exactly as §5 states.

## F. After the build

24. **Work units have no clock.** The toy counts:
    - one unit per apex-merge step, heap push and seed lookup (two per popped arc, one per direction);
    - plus S2's row reads and one unit per delta insert (Up and Down).

    There is no clock and no cost for copying chunks. The toy checks the running total after S2 and after every pop, and stops at the first check over B_gil; the design reads the clock every 256 pops against the hold's deadline. B_gil = 250 units was chosen so that every writer commit fits (the largest needs 201) except v14 (F5). That writer passes 250 at pop 15 of its recompute, with 257 units, and stops there. To report what finishing would have cost (35 pops, 506 units), the toy runs the same flush to the end on a scratch copy; that copy is never published.
    *Why it is fine:* the step shows the rule: one budget per lock hold, checked during the work, and over it the work hands off. The toy number only decides which commit is the example.
25. **The hand-off base.** Every commit here has one segment, so "the hierarchy at the abort" is the version before the commit. Multi-segment queries are not modelled.
    *Why it is fine:* with one segment, both readings give the same base.
26. **Valves.** The n/8 valve uses n = the rank dimension, and applies off the lock only (Seal and Repair). The second valve, the break-even of the new full path, is not modelled.
    *Why it is fine:* F5's Repair passes the n/8 valve, so it takes the full path, which is the case the design calls out for delta arcs. That path relaxes from the full-seed pass's arrays, written by arc id from Ax and from the delta's entries, as D2 relaxes from D1's. G9 checks those seeds against per-pair seeds, and G14 checks the result against a full customization whose seeds are read pair by pair; both fail when the delta is not scattered (03 App. B V3.4).
27. **The toy cow_btree is small and insert-only.**
    - Pages hold 4 entries per leaf and 4 children per branch (256 and 256 in the engine).
    - Inserts go one at a time (no `insert_batch`).
    - A full page splits in half.
    - Nothing is ever deleted, so there is no merge on underflow.
    - There is no compact leaf format.

    *Why it is fine:* path copying from root to leaf, and sharing every other page, is what F7 shows. The page counts depend on the toy page size.
28. **Metric chunks are 16 slots.** The engine uses 2,048. Copy-on-write is per chunk and per version, and an append copies a shared last chunk before extending it.
    *Why it is fine:* F7 shows which chunks a writer copied. The chunk size only changes how many.
29. **The fold.**
    - The trigger drops the 64k floor, leaving `√|base|` = √54 ≈ 7.35 for base 1.
    - Every writer's flush checks it (03 §6: "the writer whose flush crosses a trigger sets upkeep"). The v12 writer crosses it (Up.delta goes from 5 to 15 entries) and sets upkeep, ticket 2. That fold forks at v12 and is still running at v14, when the hand-off's new generation discards it as Gone (03 §6 Hand-off). The toy does not compute that job's result, because nothing ever publishes it.
    - The Repair's adoption is not a writer flush, so it does not check. The v15 writer finds the trigger met and no upkeep, and sets it again (ticket 4). 03 §6 says only that the writer whose flush *crosses* the trigger sets it, and that after an abandoned job "a later commit re-triggers it"; the toy reads both as "a writer whose flush ends with the trigger met and no upkeep sets it".
    - A writer that overtakes a fold, an S6 job and the §5 bound's abandonment are not modelled.

    *Why it is fine:* the steps show the fold's contract. Same ids, a metric equal to a full customization of the folded topology, and new deltas built from `added_since_fold` with seq > F are checked, each with a break. F5 shows a hand-off discarding an in-flight fold.
30. **S6 and S7 are evaluated, not run.** Every writer flush from v10 to v17 records each decay signal of 03 §6 (in `decay` and in each F step). With toy thresholds (the 1,024 and 64k floors dropped, as for the fold):
    - S7 "arcs ≥ 1.25 × arcs at build": met from v12 on (69 ≥ 67.5 at v12).
    - S7 "one commit adding > arcs/8": met at v12 (10 > 7.375) and at v14's attempt (16 > 8.625).
    - S7 "top ranks ≥ reserve": met from v12 on (2 ≥ 2; the toy reserve is 2).
    - S6 "deletions ≥ n/8": met from v13 on (3 ≥ 2.75).
    - S7 "pool-row fill > arcs/16" is never met. S6's "dead ids" is not defined in 03 and is not evaluated.

    Under the design, the v12 writer would start S7, a re-order as a new generation through the batched build, and v14 would be S7 too. So F4 to F8 show their rules on a hierarchy the design would already be replacing. Guard (a) on separator size is never reached.
    *Why it is fine:* the toy says so in the notes and in each F step where a signal is first met. Running S7 would repeat parts B to E, which the page already shows; suppressing it keeps the node delete, the hand-off, the fold and id reuse on one lineage.

## G. Invariants

31. **Queries are two upward Dijkstra searches.** The CCH query runs one search forward over upward weights and one backward over downward weights, and takes the best rank both reach. It has no stall-on-demand and no walk of the elimination tree. The reference answer is Dijkstra on the live directed graph, with the smallest weight across types. Deleted nodes are not queried. A node with no rank answers "no path" unless source and target are the same.
    *Why it is fine:* the invariant is about distances, and these are exact distances on both sides.
32. **"Every published version"** means the Built versions v9 to v17, plus the fold's publish. While the entry is Building (v5 to v8, and v14 before the Repair), readers get the Dijkstra fallback, which is Dijkstra by definition. Not every invariant runs on every published version: each one's `checked_at` says where it runs. Six do (coverage, closure, P, ids, distances, Down = Upᵀ). The others run where their step runs: the tree and the counts at Supergraph, the full-seed pass at D1 and F5's Repair, S2 at every commit that runs it, the fold at its publish.
33. **G4's break maps 02's pool onto this example.** 02 put isolated nodes and unused ids on pool ranks, in id order, and those ranks stayed in `free_ranks`. Its allocation popped the lowest free rank and made the holder rank-less. The break gives node 18 the free pool rank 2 at v11, adds road 18–17 (S2 gives rank 2 an arc while it stays free, so P fails), then replays F3's node 21, which pops rank 2 and strips node 18 (so coverage fails). That is 03 App. B V3.1.
    *Why it is fine:* under 02 the exact rank numbers would differ, but the mechanism, a pop that strips a node with arcs, is the same.

## H. Readings where the design is unclear

These are places where 03 says two things, or leaves a term open. The toy picks one reading, says which, and lists the question in example.json's `design_questions` for the design author.

34. **What n_r counts.** 03 §2 defines n_r = ranked + reserve, with "ranked" counting nodes with an edge in the skeleton *or the log at V_sg*. That gives 22 here. §3 step 0 ranks only the skeleton's nodes, and §4 step 1 ranks the log's nodes at Supergraph as a pool or top rank. The toy follows §3 and §4: n_r = 20 (B2), then n_r′ = 21 after node 20's top rank (C2).
35. **Which log nodes a Built entry keeps while upkeep is set.** 03 §6 says "every writer keeps its nodes"; §7 says a Built entry with upkeep = Some{since} "keeps its nodes with seq > since". They differ only on the node of the writer that set upkeep. The toy follows §7: the published log is empty after v12 and v15, [seq 13] after v13, and [seq 16] after v16. The fold forks at F ≥ since and needs only nodes with seq > F.
36. **Decay terms.** 03 §6 does not define S6's n or "dead ids", and "one commit adding > arcs/8" could count arcs before or after the commit. The toy uses n = the rank dimension, counts arcs before the commit, and does not evaluate dead ids.

## I. The design view: object identity across versions (example.json `mvcc`, `track`)

37. **Identity hangs off the containers the model already writes.** Bases (the `Base` object), delta pages (`PAGES`) and metric chunks (`CHUNKS`) already had ids. The model adds copy-on-write containers for `rank_of` and `node_of` (16-slot chunks with chunk ids, as `ChunkedArr` has), for `free_ranks`, `LineageStats` and `added_since_fold` (one id per whole object), and an id for each metric array's chunk table. A clone shares every id; the first write to a structure, or to one of its chunks, in a clone gives it a new id.
    *Why it is fine:* an id changes exactly where the model writes. Each published state is snapshotted at its publish, and G15 checks that two states sharing an id had equal contents when they published; `checks/c9_mvcc.py` checks the shown instances against each state's own dump.
38. **TopologyVersion and Cch ids are derived.** The model's `Hier` is one object, so there is no separate `TopologyVersion` or `Cch` to copy. At each publish their ids are derived: a new topology exactly when one of its fields has a new id (or `n` or `arc_capacity` changed), and a new `Cch` when the topology or a metric array is new.
    *Why it is fine:* that is the design's rule for an `Arc` struct whose fields are themselves shared; c9 checks the derivation both ways.
39. **Retention is shown as "held only by this state".** The grid lists, for each published state, the instances no other published state holds; with `Arc`, they are freed when that state's last holder (its entry, a reader, a job) lets go, whatever order states drop in. Readers and jobs that pin a state are shown as their own rows (the fold jobs and the Repair), not modelled as reference counts.
    *Why it is fine:* the sets do not depend on drop order, and c9 recomputes them from the instance ids.
40. **Job publishes are columns of their own.** A job's publish keeps the data version, so the grid has two states for v14 (the hand-off, then the Repair) and two for v16 (the writer, then the fold). The hand-off state holds v13's hierarchy as the Repair's base, and the model snapshots v13's hierarchy for it.
41. **LineageStats.** S2 now writes its pool-row counter only when it changes, and rank allocation counts the top ranks it takes. Before, the toy rewrote the counter on every S2 and never counted later top ranks. No output changed except that counter: the decay signal for top ranks reads `n − n_r`, which was already right.
42. **Split pages in the toy B-tree.** The toy tree only inserts and never merges. So the pages an insert creates are its root-to-leaf path plus, when a leaf splits, a new half off the path that holds only keys of one old leaf the inserts reached. G16 and c5/c9 use that rule; a tree that merges or rebalances would need a different one.
43. **The matrix track** recomputes each frame from the steps' records (the cells' pre-relabel ranks, q, the logged pairs) rather than copying C1–C5, so G19 can check it against them. Its last two frames read v16's and the fold's hierarchies from the MVCC data.
44. **Only a full-path Repair rebuilds its metric.** 03 §5 Phases says a Repair runs "as Seal, from base", and the Seal chunks the metric with `WeightArr::from_vec`. The toy does that on the full path: the v14 Repair builds flat arrays and chunks them, so every chunk is new (G17 checks it, with a break that writes into the base's chunks instead). A Repair that took the scoped path would, in the toy, write only the changed values into its base's chunks; the example's only Repair takes the full path, so that case never occurs.
