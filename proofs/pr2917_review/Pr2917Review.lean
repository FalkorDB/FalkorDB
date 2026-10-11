import Pr2917Review.Review
/-
# PR #2917 review (fix/2913-compare-mismatches @ 14e341cf4)

#2917 makes `GroupKey::eq` (ops/aggregate.rs:93) compare by `Value::sort_cmp`
(#2912's total order) instead of `Value::eq`, so every NaN key lands in one
group as in C. A `HashMap<GroupKey, _>` then needs: keys `sort_cmp`-equal ⇒
equal hashes. A CodeRabbit review found the PR at 203f2022c broke that for
VecF32/Point components (NaNs with different payloads were equal but hashed
apart; `-0.0`/`0.0` likewise). 14e341cf4 orders `-0.0` before `0.0` in
`sort_cmp_f32` and hashes components with `f32_hash_bits` (one NaN word).

The proofs had missed it: proofs/value_order's float model had no signed zero or
NaN payload and `V` had no vector/point contents, proofs/ops_aggregate's
`gkHash_consistent` took "Value's Hash agrees with Eq" as an undischarged
hypothesis `hc` (false on main), and #2917 was not modelled. This project fixes
the last two; value_order now has the extended model (copied here verbatim).

## Modules

| module | contents |
| --- | --- |
| Model    | copy of value_order `Model.lean`: `F` (NaN sign/payload, `-0.0`, integral, fractional, ±inf), `V` with `vecf32`/`point`, `compare_value`, main's hash `hwords` |
| Laws     | copy of value_order `Laws.lean`: comparator laws, `thn`, `lexA`, `cmp3` |
| SortCmp  | copy of value_order `SortCmp.lean`: `Value::sort_cmp` generic in `sort_cmp_f32` (`sc`), `sc_laws` |
| Review   | the PR: `scF32`, `f32HashBits`, `floatWordPR`, `hwordsPR`, `GroupKey` |

Line cites: 14e341cf4 — value.rs `Hash for Value` :822 (Float arm :839-850, VecF32/Point
:876-886), `sort_cmp` :1529, `sort_cmp_map` :1566, `sort_cmp_f64` :1891, `sort_cmp_f32`
:1905, `f32_hash_bits` :1919, `sort_cmp_int_float` :1931; ops/aggregate.rs `GroupKey::eq`
:93, `GroupKey::hash` :109.

## Theorems

* `scF32_key` — the new `sort_cmp_f32` is the order of a key with `-0.0 < 0.0` and every
  NaN equal (on top); `scF32_lw`, `scF32_refl`.
* **`scPR_total`** — `sort_cmp` is a total preorder (reflexive, antisymmetric,
  transitive) on every well-formed value, nested lists/maps/vectors/points included.
* **`hashPR_consistent`** — `sort_cmp`-equal ⇒ `hash`-equal, for every well-formed
  value (`wf`: Ints are `i64`, Float `num`s representable, one value per map key).
* **`hc_pr`, `gk_hash_consistent`** — ops_aggregate's `hc` discharged for the PR, and the
  `HashMap<GroupKey, _>` contract (`gkHash_consistent`, copied) instantiated;
  `gkEq_spec`: the generic `gkEq` is the PR's `GroupKey::eq`.
* `float_signed_zero_one_key`, `float_nans_one_key` — Float `-0.0`/`0.0` and all NaNs
  are one key; `vec_nans_one_key`, `point_nans_one_key` — vector/point NaN components
  are one key; `signed_zeros_apart` — vector/point signed zeros are not.
* History (the PR at 203f2022c): `pre14e341_vec_nan_hash_differs`,
  `pre14e341_vec_zero_eq_but_hash_differs`, `pre14e341_point_zero_eq_but_hash_differs`.

## Live check (2026-10-11; C `bin/macos-arm64v8-release/falkordb.so`, Rust main
release build, PR release build of the 14e341cf4 tree)

| query | C | main | PR |
| --- | --- | --- | --- |
| `UNWIND [0.0/0.0, -(0.0/0.0)] AS x WITH x RETURN x, count(*)` | `nan 2` | `nan 1, nan 1` | `nan 2` |
| `… RETURN count(DISTINCT x)` / `RETURN DISTINCT x` | 1 / 1 row | **2 / 2 rows** | 1 / 1 row |
| `UNWIND [-0.0, 0.0] AS x WITH x RETURN x, count(*)` | `-0 2` | `-0 2` | `-0 2` |
| `UNWIND [vecf32([0.0/0.0]), vecf32([-(0.0/0.0)])] AS v WITH v RETURN v, count(*)` | 1 group | 2 groups | 1 group |
| `UNWIND [vecf32([-0.0]), vecf32([0.0])] AS v WITH v RETURN v, count(*)` | 2 groups | 2 groups | 2 groups |
| same with `point({latitude:±0.0, longitude:0.0})` | 2 groups | 2 groups | 2 groups |

The PR matches C on every row. Main's DISTINCT split of `NaN`/`-NaN` (also
`count(DISTINCT [NaN])`, `{a:NaN}`, `vecf32([NaN])`) comes from the Float hash
writing `to_bits()` for NaN (`value_order` `main_float_nan_hash_differs`); the
PR's Float NaN word fixes it. Rust's `point()` refuses a NaN coordinate (C
accepts it), so `point_nans_one_key` is defensive.

## Gaps
* Non-integral floats are `frac lo r` (strictly between `lo` and `lo+1`, `r` ranks
  them); their `to_bits` is some function of `(lo, r)`, not the IEEE layout. Strings
  are `Nat`s; Path, Relationship and the temporals are not in `V` (Path shares the
  list arm, the others Int's `cmp` and hash shape).
* The hash is modelled as the word stream fed to the hasher; `ahash`'s mixing is not
  modelled (equal streams ⇒ equal hashes is all the contract needs).
* COVERAGE.tsv holds commented rows only: #2917 is not on main yet. When it merges,
  uncomment them and re-bucket ops_aggregate's `GroupKey::hash` row from CONDITIONAL.
-/
