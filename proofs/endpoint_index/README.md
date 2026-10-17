# endpoint_index: the edge id → (source, destination) lookup

This project proves that `EndpointIndex` (`graph/src/graph/endpoint_index.rs`) answers exactly what the edge set says. `EndpointIndex` is the graph-wide reverse index from a relationship id to its `(src, dst)`. It does not record the type, which lives in `relationship_type_matrix`, or node → incident edges, which live in the per-type tensors.

Every time the engine materialises a relationship, it reads this index. That happens on `RETURN r`, `startNode(r)` / `endNode(r)`, path construction, and any traversal that hands back an edge. So these behaviours depend on it:

- a relationship always reports its own endpoints, never another edge's;
- after `DELETE r`, the id reads as absent until it is reused;
- when `CREATE` reuses a deleted id, it reports only the new endpoints;
- `DETACH DELETE` leaves no stale endpoints behind;
- after an RDB load (`Graph::restore`), every edge reports its endpoints exactly as before;
- a read query keeps its view while a write query changes the index.

To save memory the index keeps four tiers of slots, 16-, 24-, 32- and 64-bit. An edge whose endpoints fit a narrow tier is stored narrow. When a reused id gets endpoints too large for its tier, the tier is *promoted*: re-encoded into a wider one. A deleted edge is a tombstone, both fields set to the tier's all-ones value. Nothing ever shrinks.

The full Lean↔Rust mapping is in the header of [`FalkorEndpointIndex.lean`](FalkorEndpointIndex.lean) (flat model) and [`EndpointPaged.lean`](EndpointPaged.lean) (pages and `Arc`s). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does `get` return what a plain map would?**

Yes. `run_refines` proves the result for any sequence of the index operations the graph performs:

- `set`, one per created edge;
- `clear`, one per deleted edge;
- `prepare`, once per bulk create.

After any such sequence, `get` equals a flat map `edge id → Option (src, dst)` updated the same way, and the internal invariant holds. `fresh_refines` states it from an empty graph. Pointwise:

- `get_set`: after `set e s d`, id `e` reads `(s, d)` and every other id is unchanged;
- `get_clear`: after `clear e`, id `e` reads absent and every other id is unchanged;
- `get_prepare`: `prepare` changes nothing anyone can read.

**2. Can promotion to a wider tier lose or change anything?**

No. `raiseTo_spec` proves `raise_to` changes no answer: live pairs, tombstones and gaps all read the same. It keeps the width invariant, empties every tier below the target, so the target starts at id 0, and leaves the tiers above alone. `view_set` proves all three arms of `set` behave identically from the outside: overwrite in place, reuse a low id with a promotion, or append into the widest or a new tier.

**3. Does restore rebuild the same index?**

`restore_agrees` proves `Graph::restore`'s path (`prepare_tiers` with any tier boundaries, then `set` for every edge) gives exactly the restored edge set.

**4. Does `DETACH DELETE` leave stale endpoints?**

No. `mem_collect` proves `delete_implicit_edges`' per-type collection is exactly "incident to a deleted node and not deleted explicitly", and `collect_nodup` that it lists no id twice. `detach_delete_index` proves that after clearing them, those edges read as absent and every other edge reads as before.

**5. Do versions stay isolated?**

Tiers are split into pages held by `Arc`s, so a new MVCC version shares pages until it writes. `pageMut_spec` proves a write through one version changes only that version's page `p`. It changes no other live version, such as the snapshot a reader holds, and keeps every strong count exact. `cloneV_spec`, `pushFresh_spec` and `replace_spec` cover cloning a version, adding a fresh page, and `regrow`. In the flat model, `snapshot_isolated` holds by construction.

**6. Do paged writes stay in bounds?**

`TierOps::put` does not bounds-check, so a bad offset would panic or write into slack. `set_inplace_in_range`, `set_promote_in_range` and `set_append_in_range` prove every write the three `set` arms issue is in range. `put_spec`, `vacate_spec` and `growTo_spec` prove the paged operations equal the flat ones. `paged_at` and `ensurePages_spec` prove `pages[i / P][i % P]` addresses the right slot.

**7. What does it cost in memory?**

`bytes_per_edge_bounds` proves the index uses between 4 and 16 bytes per edge id.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `run_refines` | After any valid sequence of `set`/`clear`/`prepare`, `get` equals a flat reference map | `FalkorEndpointIndex.lean` |
| `get_set`, `get_clear`, `get_prepare` | The pointwise effect of each operation | `FalkorEndpointIndex.lean` |
| `view_set` | All three arms of `set` mean "pad, then overwrite slot `e`", and keep the invariant | `FalkorEndpointIndex.lean` |
| `raiseTo_spec` | Promotion changes no answer and leaves the target tier starting at id 0 | `FalkorEndpointIndex.lean` |
| `restore_agrees` | The restore path yields exactly the edge set, for any tier boundaries | `FalkorEndpointIndex.lean` |
| `detach_delete_index` | After `DETACH DELETE`'s clears, incident edges read absent and the rest are unchanged | `FalkorEndpointIndex.lean` |
| `pageMut_spec` | A page write through one version is invisible to every other version | `EndpointPaged/Arc.lean` |
| `set_inplace_in_range`, `set_promote_in_range`, `set_append_in_range` | Every write `set` issues is in range | `EndpointPaged.lean` |
| `get_put_contract` | Each width's `put`/`get` is a truncating cast, and the tombstone value always fits | `EndpointPaged/Layout.lean` |
| `bytes_per_edge_bounds` | 4 to 16 bytes per edge id | `EndpointPaged/Layout.lean` |

### How `run_refines` is proved

1. **Model the type.** `Ix` has four optional tiers, `w16`, `w24`, `w32` and `w64`. Each tier is a flat list of slots, where a slot is a pair `(src, dst)`; the paging is proved separately. `EMPTY r` is tier `r`'s all-ones value, and `decode` reads a slot whose two fields are both `EMPTY` as absent.
2. **Define the meaning.** `view` concatenates the four decoded tiers in id order. `get_view` proves `get`, with its tier-by-tier walk subtracting lengths, is lookup in `view`. `locate_some` and `locate_none` prove `locate` finds the tier and offset of an id or says it is past the end.
3. **State the invariant.** `WF` says every slot of a narrow tier is either the tombstone or strictly below `EMPTY` in both fields. That is what `rank_for` guarantees, and it is why a narrow slot can be widened without becoming a tombstone (`widen_ok`).
4. **One operation.** `view_set`, `view_clear` and `view_prepare` describe each operation's effect on `view` and prove it keeps `WF`. `set` splits into its three arms, and the promotion arm uses `raiseTo_spec`.
5. **Any sequence.** `step_refines` lifts one operation to the reference map, and `run_refines` is induction on the list.
6. **Hypothesis.** `Valid s d` says both endpoints fit in 64 bits and are not *both* `u64::MAX`. That pair is the 64-bit tier's tombstone, so it would read as deleted (`u64_max_pair_reads_as_absent`); one side at `u64::MAX` is fine (`u64_max_one_side_is_live`). The engine cannot produce such a pair: `IdSpace::create` refuses node ids `≥ GrB_INDEX_MAX = 2^60 − 1` (`proofs/id_space`, `create_out_of_range`).

## Bugs found

None. The one boundary the spec needs, an edge whose endpoints are both `u64::MAX`, is unreachable (above), and is documented in `endpoint_index.rs` only for the narrow tiers.

There are no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `FalkorEndpointIndex.lean` | The flat model (`Ix`, tiers, `get`, `locate`, `promote`, `raise_to`, `set`, `clear`, `prepare`, `prepare_tiers`), the refinement proof, restore, the paging lemma, `DETACH DELETE` collection, the `u64::MAX` boundary, and `#guard` replays of Rust unit tests |
| `EndpointPaged.lean` | Root of the second library: every write `set` issues is in range |
| `EndpointPaged/Layout.lean` | The `Endpoint` impls and their contract, `Paged<T>` shape, memory accounting, `from_slots` |
| `EndpointPaged/Writes.lean` | Page-level `put`/`vacate`, `grow_to`, `reserve_exact` |
| `EndpointPaged/Arc.lean` | `Arc` pages as a heap with strong counts: copy-on-write isolation |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile builds two libraries, `FalkorEndpointIndex` and `EndpointPaged`; both are default targets. There are no path dependencies.

## Coverage

`COVERAGE.tsv` has 69 rows: 66 in `endpoint_index.rs` and 3 in `graph.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 52 |
| MODELLED | 2 |
| NOT COVERED | 15 |

- **MODELLED:** `Graph::restore` and `Graph::delete_implicit_edges`. Only their endpoint-index parts are modelled here (`restore_agrees`; `mem_collect`, `collect_nodup`, `detach_delete_index`). The rest of those functions is covered in `proofs/graph_queries`.
- **NOT COVERED:** all 15 are `#[cfg(test)]` unit tests and test helpers in `endpoint_index.rs`, not engine code.
- The two `Endpoint` trait declarations have no body. They are PROVEN through `get_put_contract` and `u24_get_put`, which hold for all four impls.

## What the model does not cover

- The flat model treats a tier as one list of slots; `EndpointPaged` proves the page layout and page writes separately.
- Weak counts are not modelled: no `Weak` to a page or tier is ever created, so "strong = 1 and weak = 0" is "strong = 1".
- The allocator, and `reserve_exact`'s capacity, have no effect on answers and are not modelled.
- `bytes_per_edge` ends in an `f64` division. Its numerator and denominator are proved, and the quotient is plain IEEE division.
- MVCC isolation in the flat model is by construction (states are values). The real isolation argument is the `Arc` heap model, which covers one sequence of operations, not concurrent threads.

## Build and check

```bash
cd proofs/endpoint_index && lake build    # this project (both libraries)
bash proofs/lean_ci.sh endpoint_index     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                   # repo-wide coverage report
```

**When you change `endpoint_index.rs`, or the calls into it in `graph.rs` (`create_relationships_bulk`, the delete paths, `restore`):**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match. A change to tiers or promotion touches `FalkorEndpointIndex.lean`; a change to pages or `Arc` handling touches `EndpointPaged/`.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
