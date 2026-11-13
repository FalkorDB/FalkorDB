# U1: the current Rust CCH pipeline and the contract Phase 1 must keep

Revisions read: `origin/main` = 7df8e5887 (`mod.rs`, `nd.rs`, `order.rs`, `customize.rs`, `query.rs`, `testkit.rs`; `input.rs` there is a 3-line stub), `origin/cch/05-inputs` = d69afac42 (`input.rs`), `origin/cch/11-partial` = cb2a97519 (`partial.rs`), `origin/cch/08-maint` = 89e6c0cbe (`maint.rs`), `origin/cch/08c-build-catch-up` = 31ca1b802 (`mod.rs`, `customize.rs`), feral-metis checkout `f127189`, C `origin/master` = a61fbb67a. Paths are under `graph/src/index/falkordb/cch/` unless stated otherwise. "Computed" means arithmetic on measured counts, not a measurement.

## 1. Data flow and structures

```
tensors --GraphBLAS--> Matrix<f64> W, Matrix<bool> S --export--> CchInput{skeleton: Csr, weights: WeightCsr}
  -> nd.rs (i64 copy -> feral copy) -> rank_to_node
  -> order.rs Topology{rank_to_node, node_to_rank, upward, downward}
  -> customize.rs Metric (4 per-arc arrays) -> query.rs
```

**Input build** (`input.rs` @05 :65-127). `n = graph.node_cap()` (:79), so freed and spare ids are isolated nodes. Each type's effective forward matrix (`tensor.effective_fwd()`, `graph/src/graph/graphblas/tensor.rs:1060-1067` @main, which is `(m minus dm) union dp`) is scored into a `Matrix<f64>`. The types are merged with MIN (:91-103) and the diagonal is dropped (:108). The skeleton `S = pattern(W) union pattern(W transposed)` is built as a `Matrix<bool>` (:112-113). **Both are then exported to CSR** (:118-119): the export *copies* into Rust `Vec`s (`graphblas/csr.rs:67-109` @main) and the GraphBLAS matrix is freed afterwards.

| Structure | Fields, element type | Bytes | Read by |
|---|---|---|---|
| `Csr` skeleton (`mod.rs:82-88`) | `row_offsets: Vec<u64>` (n+1), `neighbors: Vec<u64>`, symmetric | 8/node + 8/entry (each undirected pair twice) | `nd.rs`, `Topology::from_order` only |
| `WeightCsr` (`mod.rs:159-163`) | a `Csr` plus `values: Vec<f64>`, directed | 8/node + 16/entry | `customize::validate`/`seed`; `partial.rs` `edge_entry` (@11 :35-46) |
| `nd.rs` copy (`nd.rs:91-119`) | `backend_id: Vec<i64>` (n), offsets `i64` (m+1), neighbors `i64` (nnz); m = connected nodes | 8n + 8(m+1) + 8 nnz, freed on return (`nd.rs:87-90`) | feral |
| feral `Graph` (`feral-metis/src/graph.rs:53-101`, `Idx = i64` per `feral-ordering-core/src/lib.rs:54`) | offsets, adjacency, vertex weights (all 1), edge weights (all 1) | 16/node + 16/entry, plus every coarsening level | feral ND (`node_nd.rs:35`) |
| `Rank` (`order.rs:44-63`) | `NonZeroUsize` holding index+1 | 8; `Option<Rank>` is also 8 | everywhere |
| `Topology` (`order.rs:129-139`) | `rank_to_node: Box<[NodeId]>` (u64), `node_to_rank: Box<[Rank]>`, `upward`/`downward: RankRows` | 32/node + 16/arc | Phases 2 and 3, maintenance |
| `RankRows` (`order.rs:142-148`) | `offsets: Box<[usize]>` (n+1), `neighbors: Box<[Rank]>` | 8/node + 8/arc each | as above |
| `Metric` (`customize.rs:94-99`) | `upward_weight`, `downward_weight: [f64]`, `upward_middle`, `downward_middle: [Option<Rank>]` | 32/arc | query, scoped recustomization |

On 08-maint the four metric arrays become `WeightArr`: copy-on-write chunks of 2,048 slots (`customize.rs:100-127` @08-maint). They are still indexed by slot.

California (1,540,096-node skeleton, 3,313,386 entries, 5,838,003 arcs; PR #3321 body):
- skeleton CSR: 38.8 MB (computed);
- Topology: 142.7 MB measured (PR #3321), which matches 32 x 1,540,096 + 16 x 5,838,003 = 142.69 MB;
- Metric: 186.8 MB (computed).

**Against C** (`src/algorithms/cch.h:69-140` @a61fbb67a), per arc: C keeps `up`, `down`, `up_w`, `dn_w`, `up_mid`, `dn_mid`, 8 bytes each, so 48 B/arc. Rust keeps 16 + 32 = 48 B/arc, the same. Per node: C keeps `perm`/`iperm` (16 B), six row pointers (48 B) and an `arr_t` header per row (size not checked). Rust keeps 32 B/node.

C frees its skeleton CSR and its `parent` array as "PHASE-1 SCRATCH" (`cch.h:76-93`). Rust keeps both CSRs alive through ordering, the elimination game and customization: `build_checked` borrows `&CchInput` for the whole build (`customize.rs:563-581` @08c, called from `mod.rs:686-688` @08c).

**Ordering** (`nd.rs:46-85`):
- checks `is_well_formed` and `is_symmetric_simple` (:51-58; the second costs one binary search per entry);
- ranks isolated nodes first, in id order (:62-64);
- copies the connected nodes into `i64` arrays (:91-113); feral then copies them again (`graph.rs:53-101`).

The output is `Vec<NodeId>` with `rank_to_node[rank] = node`.

**Elimination game** (`order.rs:195-212`):
- `invert` builds `node_to_rank` (:333-360);
- `chordal_supergraph` (:381-437), for each rank in increasing order:
  1. maps its skeleton row to ranks and keeps those above it (:401-408);
  2. appends each child's finished upward row, without the rank itself (:410-419);
  3. sorts and dedups the row (:423-424);
  4. links the row to its parent, which is `row.first()` (:426-428).

  Scratch: `first_child` and `next_sibling` (8 B/node each).
- `transpose` is a counting sort into `downward` (:441-467).

## 2. The Phase 1 output contract (what Phases 2 and 3 actually read)

API surface: `node_count`, `ranks()`, `arc_count`, `node_of`, `rank_of`, `up(r)`, `up_range(r)`, `down(r)`, `arc_slot(a, b)` (`order.rs:214-302`). Nothing reads `RankRows` directly.

| Invariant | Where stated | Code that relies on it |
|---|---|---|
| **I1. Two id spaces with a bijection over `0..n`**, n = node_cap | `order.rs:104-112`; checked by `invert` :333-360 | `seed` maps each edge `rank_of(u), rank_of(v)` (`customize.rs:226`). The query bounds-checks against `node_count` (`query.rs:236-240`) and maps back with `node_of` (:331-334). |
| **I2. Up rows strictly ascending, every entry above its own rank** | `order.rs:633-643` (test) | `arc_slot` binary search (`order.rs:297-301`); relax cursor (`customize.rs:282-294`); query `gallop_to` (`query.rs:110-114, 156-172`) |
| **I3. Arc slot = flat position**: `up(r)[i]` is slot `up_range(r).start + i`; `arc_count` sizes every weight array; **slots stable for the life of a Topology** | `order.rs:123-125, 262-268`; `customize.rs:36-37` | `x_slots_start + i` (`customize.rs:270-276`); `topology.up_range(rank).zip(topology.up(rank))` (`query.rs:113`); `up(lo)[slot - up_range(lo).start]` (`partial.rs:158` @11); arrays sized by `arc_count` (`customize.rs:208-214`) |
| **I4. Every skeleton edge is an arc** | `order.rs:667-680` | `seed`: no arc means `return Err(CchError::NeedsRebuild)` (`customize.rs:227-230`). Maintenance classifies a new pair with no arc as *structural*, which means a full rebuild (`maint.rs:21-26` @08-maint) |
| **I5. Chordal closure**: every pair in `up(x)` is adjacent. Stronger form actually used: for `y` in `up(x)`, `up(x)` above `y` is a subset of `up(y)` | `order.rs:682-697`; `customize.rs:247-249` | `panic!("chordal closure broken…")` in the cursor walk (`customize.rs:288-293`); `.expect("chordal closure: upper neighbours of lo are adjacent")` (`partial.rs:221-223` @11) |
| **I6. Elimination tree is implicit**: parent(r) = `up(r)[0]`, and every upper neighbour is an ancestor | `order.rs:375-380`; `query.rs:16-22` | chain climb `while let Some(&parent) = topology.up(rank).first()` (`query.rs:76-79`); `debug_assert_eq!(…, "an upper neighbour must be an ancestor")` (`query.rs:115-119`); different roots mean no path (`query.rs:254-265`) |
| **I7. `downward` is the exact transpose of `upward`, ascending** | `order.rs:119-121, 655-666` | the scoped recustomization merges `down(lo)` with `down(hi)` (`partial.rs:173-204` @11) and expects `"down/up are inverses"` (:187-188). Only that path reads `downward` (`order.rs:137`). |
| **I8. Increasing rank is a valid order to process arcs in** (an arc reads only arcs with lower lower-ends) | `customize.rs:77-85`; `partial.rs:8-15` @11 | `relax` loops `for x in topology.ranks()` (`customize.rs:268`); the heap pops by lower rank (`partial.rs:151-156`) |
| **I9. The metric stores ranks, not just slots**: a middle is a `Rank` strictly below both ends | `customize.rs:59-62` | `unpack` asserts `m < a.min(b)` (`query.rs:220`). Tie rules (strict `<`, lowest apex wins) depend on rank order (`customize.rs:66-75`). The scoped path is compared bit for bit with the full sweep. |

**Consequence for an incremental Phase 1:**
- Inserting one arc shifts every later slot in the flat `neighbors` array (I3), and each of the four metric arrays is keyed by slot.
- Changing any rank invalidates the `node_to_rank` entries and the middle ranks stored in the metric (I9). It also changes ancestor chains (I6).
- Today none of this is incremental. A new adjacency without an arc forces a full rebuild (I4, `maint.rs:21-26`). The background build is a single batch with `Cursor = ()` (`mod.rs:473-477, 604-612` @08c).

## 3. Lower-triangle enumeration

**Full sweep** (`relax`, `customize.rs:264-312`):
- For each `x` in increasing rank, every pair `i < j` of `up(x)` is a triangle `x < y < z`.
- The top arc `(y, z)` is found by a cursor that moves only forward through `up(y)` (:288-294), not by a search.
- One sweep therefore enumerates the sum over x of C(|up(x)|, 2) triangles (derived).

It needs from the supergraph:
- rows addressable by rank in O(1);
- **two rows open at once** (`up(x)` and an arbitrary higher `up(y)`);
- ascending rows (I2);
- slot positions parallel to row order (I3);
- I5.

**Scoped path** (`partial.rs:173-204` @11) enumerates one arc's lower triangles the other way round: the merge of `down(lo)` and `down(hi)` gives the common lower neighbours, then two `arc_slot` binary searches per apex. It needs I7.

**What a GraphBLAS version of this would face** (design input, not implemented or measured):
- Triangles `U(x,y) and U(x,z) and U(y,z)` are the pattern of the masked product `C<U> = U transposed · U` over strictly-upper U in rank space. That is the pull form: arc `(y,z)` takes the minimum over `x` in down(y) and down(z), which is what `partial.rs` does for one arc.
- Customization is not one masked (min,+) product. Row `y` may only be read after every row below it that feeds it is final (I8), so it is a triangular solve over (min,+).
- That solve could run one batch per elimination-tree level. The code notes that a parallel pull "has not been implemented or measured" (`customize.rs:251-259`).
- Keeping the tie rules (I9) needs a lexicographic minimum on (weight, apex rank), i.e. a user-defined type and monoid. Without it, results are no longer bit-identical to the scoped path.

## 4. Where the CSR is consumed, and what changes

**Consumers of `Csr`/`WeightCsr`:**
- `nd.rs:46-119`: checks, the isolated-node split, and the i64 copy;
- `order.rs:195-212, 401-408`: one skeleton row per node, read in rank order;
- `customize.rs:136-161, 163-178, 204-240`: validate, `entries()`, seed;
- `partial.rs:35-46, 98-149` @11: `edge_entry` per changed pair, and a full `customize` past the n/8 valve;
- 08-maint: `CchInput::weights_for` (full recustomization) and `PairWeights`, which seeds pair by pair from the graph without building a CSR (`maint.rs:51-55` @08-maint);
- `testkit.rs:34-69`: `to_input` builds both CSRs with `extract_tuples`.

**If Phase 1 took GraphBLAS matrices as input:**
- The skeleton already exists as a `Matrix<bool>` before export (`input.rs:112-113` @05). The pattern export, `Csr`, and the `From<CsrExport>` impls (`mod.rs` diff @05) would go.
- The shape and symmetry checks in `nd.rs:51-58` become guaranteed by construction.
- The isolated-node split becomes a row-degree reduce.
- Step 1 of the game (higher skeleton neighbours in rank space) maps to two whole-matrix operations: `GrB_extract` with `rank_to_node` as both index lists, then `GrB_select` TRIU with thunk 1.
- Step 2 (absorbing children's rows) is sequential by rank and reads rows of the result built so far. That needs either:
  - a row cursor over a GraphBLAS matrix: `degree_bench.rs:16-19` @main measured 1,654 instructions per `GxB_Iterator` attach against 298 for a re-seek, so the cursor must be reused; or
  - a level-by-level formulation (other slices).

**If Phase 1's output were a GraphBLAS matrix** (upward U, n x n, strictly upper, row-major, rank space):
- Rows come out sorted (`csr.rs:5-11`), so I2 holds.
- `downward` needs a second matrix holding the transpose. A matrix is stored in one orientation, and I7's row merges want rows of the transpose.
- **Slots (I3) have no public GraphBLAS equivalent.** The options:
  - (a) store the slot as U's `u64` value: +8 B/arc over a pattern-only matrix;
  - (b) store the weights as values of matrices with U's pattern, with no slots at all. This conflicts with 08-maint's chunked copy-on-write `WeightArr`, which needs a dense slot index (`customize.rs:100-127` @08-maint);
  - (c) unload into a `GxB_Container` to read the offsets. That empties the matrix, so a shared published matrix has to be copied first (`matrix.rs:568-573` @main). This is the inner-structure exposure the redesign wants to avoid.
- `arc_slot` (used by seed, unpack and maintenance classification) becomes `extractElement`.
- The query reads one row per chain rank, so it needs a reusable cursor. The bar is query p99 526 us on roadNet-CA (PR #3324). The effect of this change is unmeasured.
- Sharing across MVCC versions: the `Topology` is shared by `Arc` today (`mod.rs:246-258`). A shared `GrB_Matrix` has to be fully waited before it is published, because the export, a read, is serialized under the matrix lock (`matrix.rs:1283-1293`).
- GraphBLAS v10 has 32-bit index hints (`GxB_ROWINDEX_INTEGER_HINT`, `GraphBLAS.h:1557-1559` in the parking-lot-pin checkout), against 8-byte `Rank`/`usize` today. The possible memory saving is unmeasured.

## 5. Costs today

| Step | Complexity (from the code) | Measured (California unless noted) |
|---|---|---|
| Input weight passes | O(nnz) GraphBLAS passes per type | about 5 ms without the skeleton and export (`input.rs:114` @07c); export not measured separately |
| Skeleton checks | O(n + nnz), plus one binary search per entry (`nd.rs:31-33`) | unmeasured |
| ND (feral) | multilevel; no bound stated in the code; two input copies (§1) | 3.62 s against C METIS 3.07 s; peak 447 against 363 MiB over the input; 5,838,003 arcs (PR #3321) |
| Elimination game | each finished row is read once by its parent, then sorted: O(nnz + arcs) plus the sort | 97 ms; peak 185 MiB over the input; topology holds 142.7 MB (PR #3321) |
| Transpose | O(n + arcs) | part of the 97 ms (unmeasured alone) |
| Seed | O(directed edges x log row length) | unmeasured alone |
| Relax | sum over x of C(\|up(x)\|, 2) triangles plus cursor walks | 219 ms (PR #3323); about 0.09 us/arc (`maint.rs:85-94` @08-maint) |
| Full path (weights + customize) | | about 0.5 s (`maint.rs:85-94`) |
| Scoped recustomization | cone of the changed arcs | about 1.3 us per arc; 0.41 s for 512 random pairs (`maint.rs:85-94`) |
| Query | both ancestor chains, plus unpacking | p50 0.385 ms (PR #3321); p99 526 us on roadNet-CA (PR #3324) |
| `CREATE CCH INDEX` end to end | | 4.63 s (PR #3321) |

The ordering dominates (3.62 s of the 4.63 s; separate runs, PR #3321). It runs as one uninterruptible batch on the shared builder: "a multi-second CCH build delays column builds behind it, default 1 worker" (PR #3771 body).

Not measured anywhere I found: elimination-tree height and chain lengths, peak memory of the export step, and the size of the weight CSR on California.
