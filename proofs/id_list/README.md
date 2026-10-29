# id_list: the compact id lists inside replication effects

This project proves that `IdList` (`graph/src/effects/v3/id_list.rs`) keeps every id it is given, in order, and that its byte format decodes back to exactly what was encoded. It also shows, with concrete counterexamples, where it used to break.

`IdList` is how a v3 effect record carries a list of node or relationship ids, for example the ids a `CREATE` produced or a `DELETE` removed, when a write is replicated with `GRAPH.EFFECT`. It stores ids as compact segments:

- ascending or descending ranges;
- repeats;
- ascending or descending roaring bitmaps when a run of ranges is cheaper as a bitmap.

The order matters: the replica pairs the *n*-th id with the *n*-th row of attributes. If ids are reordered, properties land on the wrong entity (the class of #2842). If decoding is not total, a malformed payload can crash the replica.

The full Lean↔Rust mapping is in the header of [`FalkorIdList.lean`](FalkorIdList.lean), in the `file:line` table at the top of `Push.lean` and `Wire.lean`, and in [`REPORT.md`](REPORT.md). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does the builder keep every id, in push order?**

Yes. `pushAll_ok` proves that pushing any sequence of `u64` ids keeps every one of them in the order pushed: unsorted, with duplicates, `0` and `u64::MAX`, or empty. `len` counts every push, and the `assert!` in `maybe_collapse_run` never fires. This holds whatever the collapse arithmetic decides at each push. `collapse_ok` proves that turning a run into a bitmap reads back the same ids in the run's direction. `fromIter_iter` is the same statement for `IdList::from_iter`.

The collapse decision is first treated as a free choice, so correctness cannot depend on it. `PushT.lean` then threads the real tally through `push`, and `pushAllT_ok` proves the same result for the list the Rust actually builds.

**2. Does `decode(encode(…))` give back the ids?**

Yes. `push_encode_decode` proves that building a list, encoding it and decoding the bytes returns the same segments, and the ids as pushed. It holds for every id sequence whose count fits the `u32` record count. `readIds_encodeIds` proves decoding stops exactly where the list ends, so re-encoding a decoded list is byte-identical. `decode_encodeSeg` does the same per segment.

**3. Can a malformed buffer crash the decoder or make it over-read?**

No. `readIds_never_panics` proves no input reaches the debug-build `u64` underflow in `read_ids`, or any other modelled panic. `readIds_safe` proves that whatever `read_ids` accepts from arbitrary bytes has these properties:

- every segment is well-formed, so iterating cannot panic or wrap;
- it totals exactly `count` ids;
- it has at most `count` segments;
- it was read from a prefix of the input, so the decoder never over-reads.

The concrete refusals are each a theorem (`refuses_reserved_bit`, `refuses_wrap`, `refuses_truncated`, `refuses_too_many_segments`, …).

**4. Is it safe to push onto a decoded list?**

- **Before #2920, no.** `from_segments` started the run at the last decoded segment. A later collapse could read a descending range back ascending (decoded `[10,9,8]` plus `20,22` came back `[8,9,10,20,22]`) or hit the `assert!`.
- **On main, yes.** `decoded_then_pushed_keeps_order` proves pushing any ids onto any well-formed decoded list gives the decoded ids followed by the new ones. `readIds_then_pushed_keeps_order` proves the same end to end from the wire.

**5. Does collapsing to a bitmap actually save space?**

`collapse_shrinks` proves that when `prefers_bitmap` fires, the bitmap segment is strictly shorter than the ranges it replaces. That rests on roaring's `serialized_size()` matching the tally (`hsize`), which the Rust test `predicted_matches_roaring` checks. `no_collapse_below_floor` proves nothing collapses below roaring's minimum size. `bucketBody_spec` proves the per-bucket cost picks the cheapest container, as roaring's `optimize()` does.

**6. Is the encoding canonical?**

No, and that is allowed. `decode ∘ encode` is the identity, but the decoder accepts more than one byte string per list. `nonminimal_width_accepted` shows a wider-than-needed integer, and `lone_descending_accepted` shows a lone id marked descending. Honest buffers are never misread. This only matters if two engines' raw bytes are ever compared.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `pushAll_ok`, `push_ok` | Any push sequence keeps every id in order, for any collapse decisions; the `assert!` never fires | `Push.lean` |
| `pushAllT_ok` | The same with the real collapse arithmetic | `PushT.lean` |
| `collapse_ok` | Collapsing a run to a bitmap keeps its ids and direction | `Push.lean` |
| `push_encode_decode` | Build, encode, decode gives back the pushed ids | `Wire.lean` |
| `readIds_encodeIds`, `decode_encodeSeg` | Decoding an encoded list returns its segments and stops exactly at its end | `Wire.lean` |
| `readIds_never_panics`, `readIds_safe` | No input panics the decoder; what it accepts is well-formed, totals `count`, and was read in bounds | `Wire.lean` |
| `decoded_then_pushed_keeps_order` | Pushing onto a decoded list keeps every id in order (#2920) | `Decoded.lean` |
| `collapse_shrinks` | When the bitmap is chosen, it is smaller than the ranges it replaces | `CostProps.lean` |
| `toRoaring_spec`, `eqList_iff` | `to_roaring` is exactly the id set; `==` compares ids, not segmentation | `Misc.lean` |

### How `pushAll_ok` is proved

1. **Model the type.** `Seg` has one constructor per `Segment` variant: `range`, `rep`, `rdesc`, `asc`, `dsc`. A bitmap is held as its ascending list of ids. `St` is the list's segments, its `len`, and the run's start and claimed direction. `flat` is what `IdList::iter` yields.
2. **Model the function.** `push` follows `IdList::push` line by line, in the same branch order. It claims a direction, tries the in-place extension fast paths, then appends and maybe collapses. `collapse` returns `none` exactly where `maybe_collapse_run`'s `assert!` would fire.
3. **State the invariant.** `Inv` says:
   - the run start is within the list;
   - every segment is well-formed;
   - every segment in the run is range-like and reads in the run's direction (`RunOk`);
   - the last range-like segment is always inside the run.
4. **One push.** `push_ok` proves that from `Inv` a push succeeds and lands in `Inv`, with `flat` extended by the id and `len` by one. The proof is a case split on the last segment and the claimed direction.
5. **Any sequence.** `pushAll_ok` is induction on the ids, with one collapse decision per push.
6. **Hypotheses.** There are two:
   - every id is `< 2^64`, which Rust's `u64` guarantees;
   - the collapse decision is arbitrary, which only makes the result stronger.

## Bugs found

See [`../FINDINGS.md`](../FINDINGS.md) and [`REPORT.md`](REPORT.md) for the repros.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | removed (17ccc5412) | Pushing onto a decoded `IdList` reordered ids (decoded `[10,9,8]` + `20,22,…` → `[8,9,10,20,22,…]`) or panicked on a decoded `Repeat`. Latent: no production path pushed onto a decoded list | Fixed by #2920 (closes #2919; `decoded_then_pushed_keeps_order`) |

There are no open bugs and no CONDITIONAL rows.

Suspected, not confirmed: a roaring blob with the same high key twice keeps only the last partition in roaring-rs. CRoaring may keep the first, so the two engines could decode the same bytes differently. This has not been checked against the C module.

## Files

| File | What's in it |
|---|---|
| `Push.lean` | `Seg`, `St`, `push`, `collapse`, `from_segments`; the invariant `Inv`, `push_ok`, `pushAll_ok`, `fromIter_iter`; the pre-#2920 counterexamples |
| `Decoded.lean` | The decoded-list invariant `DInv` and `decoded_then_pushed_keeps_order` (#2920) |
| `Wire.lean` | The byte format: narrow integers, headers, `Segment::encode`/`decode`, `read_ids`; round trip, safety, refusals, non-canonical inputs |
| `Cost.lean` | The collapse arithmetic: `bucket_body`, `bitmap_header`, `Run` and its tally, `encoded_len` |
| `CostProps.lean` | Its theorems: `bucketBody_spec`, `addLoop_eq`, `encodedLen_eq`, `collapse_shrinks`, `no_collapse_below_floor` |
| `PushT.lean` | `push` with the real collapse decision: `pushT_st`, `pushAllT_ok` |
| `Misc.lean` | `is_empty`, `count`, `to_roaring`, the `PartialEq`s, the `From`s, `Debug` |
| `REPORT.md` | The long-form report; some counts and the "free Boolean" note predate `Cost`/`PushT`, and the root `.lean` header is current |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile has no path dependencies.

## Coverage

`COVERAGE.tsv` has 47 rows: 41 in `id_list.rs`, 5 in `effects/reader.rs` and 1 in `narrow_int.rs` (`width_for`).

| Bucket | Rows |
|---|---|
| PROVEN | 47 |

Every function is proven.

## What the model does not cover

- Roaring is abstract. `RoaringOK` states only what the format needs:
  - a serialized set deserializes back to itself;
  - `serialized_size` is the blob length and fits a `u32`;
  - whatever deserializes is a sorted set of `u64`s.

  The actual blob bytes and `optimize()` are not modelled. That `serialized_size()` equals the tally is a hypothesis of `collapse_shrinks`, checked by the Rust test `predicted_matches_roaring`.
- A segment's `u32` length overflowing inside `push` (2^32 pushes) is not modelled; the wire theorems assume fewer than 2^32 ids.
- The cached `len`/`min`/`max` of bitmap segments are derived from the set, so cache maintenance is not checked.
- `Reader` is a byte list. Allocation bounds inside roaring's deserializer were read, not modelled.

## Build and check

```bash
cd proofs/id_list && lake build    # this project
bash proofs/lean_ci.sh id_list     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh            # repo-wide coverage report
```

**When you change `effects/v3/id_list.rs`, `effects/reader.rs` or `narrow_int.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match. A change to the byte format touches `Wire.lean`; a change to `push` or the collapse touches `Push.lean`, `Cost.lean` and `PushT.lean`.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
