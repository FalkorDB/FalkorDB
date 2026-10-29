# core: the shared foundations of the Lean models (`FalkorCore`)

`core` is a library, not a proof about one piece of Rust. It holds the few models that several projects used to copy into their own files:

- the engine `Value`;
- the IEEE float model;
- `i64` range and wrap-around arithmetic;
- integer orderings;
- the generic laws a comparator must satisfy to be a valid sort order.

Each of these now has one definition, so a fix to the float model, for example, reaches every project that compares floats.

No query feature depends on `core` directly. It matters because the headline theorems of other projects are stated in its terms. For example:

- `value_order`'s proof that `ORDER BY` is a total order uses `core`'s float model `F`, comparator laws `Lw`, `thn` and `lexA`, and `cmp3`;
- `value_math`'s argument type-checking theorems quantify over `core`'s `V F`, the engine `Value`;
- `expr_semantics` and `functions_str_list` use `core`'s `wrap` and `InI64` for every integer-overflow result.

If a definition in `core` doesn't match the Rust it stands for, every one of those theorems is about the wrong thing.

The module table (what each file contains and which project it came from) is in the header of [`FalkorCore.lean`](FalkorCore.lean). The header calls `core` a **pilot**: it collects what was shared when it was created, and other projects still carry their own copies of some of these definitions.

## The questions this library answers

**1. What is a FalkorDB `Value`, to a proof?**

`V F` is `enum Value` (`graph/src/runtime/value.rs:180`) with all 16 variants:

- `null`, `bool`, `int`, `float`;
- `str`, `list`, `map`;
- `node`, `rel`, `path`;
- `vec` (VecF32), `point`;
- `datetime`, `date`, `time`, `duration`.

The `f64` payload is a type parameter `F`, so a project can use the concrete float model below, Lean's `Float`, or an abstract structure of float operations. `Arc`s are transparent. `V.order` is `OrderedEnum::order`, the type rank that `ORDER BY` uses across types (`value.rs:1178`), and `V.name` is `Value::name`, the type name in error messages (`value.rs:1352`).

**2. Which floats does the model tell apart?**

Every class of IEEE value that ordering or hashing can distinguish. `F` has six constructors:

- `nan neg payload`: NaN with its sign bit and payload;
- `negZero`: `-0.0`, while `num 0` is `+0.0`;
- `num n`: an integral finite value;
- `frac lo r`: a non-integral finite value strictly between `lo` and `lo + 1`, where `r` orders the ones that share `lo`;
- `negInf` and `posInf`.

The same `F` models `f64` (Float) and `f32` (VecF32 and Point components); only the bit layout differs. The model was extended on 2026-10-11. Before that it was `nan | num`, which is why `value_order` missed the #2917 hash bug on `-0.0` and NaN payloads (see [`../value_order/README.md`](../value_order/README.md)).

On top of `F`:

| Definition | Rust |
|---|---|
| `fpcmp` | `partial_cmp`: `none` exactly when either side is NaN; `-0.0` and `+0.0` are equal |
| `cmpF` | `compare_floats` (`value.rs:1801`): `partial_cmp`, and NaN answers `(Less, NaN)` whichever side it is on |
| `roundF` | `i as f64`: round to 53 significant bits, ties to even |
| `bits64`, `bits32` | `f64::to_bits` / `f32::to_bits` |
| `asI64` | `x as i64`: saturating, NaN gives 0, truncation toward zero |

There is no axiom. A non-integral value has no concrete bit pattern: `bits64` maps it to *some* function of `(lo, r)`, which is all the hash proofs need.

**3. How is `i64` arithmetic represented?**

As a mathematical `Int` with a range predicate. `InI64 x` says `x` fits in an `i64`. `wrap` is two's-complement wrap-around onto `[-2⁶³, 2⁶³)`, which is what a release build does. `wAdd`, `wSub`, `wMul`, `wDiv` and `wRem` are the `wrapping_*` operations; `wDiv` wraps only `MIN / -1`, and `wRem` gives `0` for `MIN % -1`. `satI64` is the saturating `f as i64`, and `w` is the `i64 → u64` reinterpretation used for hash words.

Two laws come with it:

- `wrap_inRange`: `wrap` always lands in range;
- `wrap_id`: `wrap` is the identity on in-range results, so a wrapped operation equals exact arithmetic whenever the exact result fits.

**4. What makes a comparator a valid sort order?**

`FalkorCore.Laws` answers this once, for any type. `Lw c` holds when `c` is:

- antisymmetric: `c b a` is `c a b` reversed;
- transitive: `a ≤ b` and `b ≤ c` imply `a ≤ c`.

Together these make `c` a total preorder, which is what Rust's `sort_by` requires. Since Rust 1.81, the sort panics when it catches a comparator breaking them.

The library proves that the comparators Rust builds from smaller ones keep these laws:

- **`thn`** is `Ordering::then`/`then_with`. `thn_swap` and `thn_trans` show that a lawful key comparator followed by a second comparator is lawful, provided the second one is lawful on the pairs the key leaves equal. This is how a multi-key `ORDER BY a, b` is shown to be lawful.
- **`lexA`** is lexicographic order, shorter first: `[T]::cmp`, and the list arm of `sort_cmp`. `lexA_refl`, `lexA_swap` and `lexA_trans` show that it is lawful whenever the element comparator is lawful on the elements that actually occur. `lexA_lw` is the unconditional form. `zipNe_eq_lexA` shows that the `zip`/`find` "first unequal pair" form equals `lexA` on lists of equal length.
- **`cmp3`** is the lexicographic order on `Nat × Int × Nat` (the float position key). `cmp3_lw` shows that it is a total order, and `cmp3_eq_iff` shows that it is `eq` exactly on equal keys. `cmpI_lw` and `cmpN_lw` give the same for `Ord::cmp` on `i64` and `usize`.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `thn_swap`, `thn_trans` | `a.cmp(b).then(…)` is a valid sort order when its parts are | `Laws.lean` |
| `lexA_swap`, `lexA_trans`, `lexA_refl`, `lexA_lw` | Lexicographic list order is a valid sort order when the element order is | `Laws.lean` |
| `lexA_eq` | Two lists compare equal only if they have the same length and are equal element by element | `Laws.lean` |
| `zipNe_eq_lexA` | The "first unequal pair" loop equals lexicographic order on lists of one length | `Laws.lean` |
| `cmp3_lw`, `cmp3_eq_iff` | The float position key is a total order whose `eq` is equality | `Laws.lean` |
| `cmpI_swap`, `cmpI_le_trans`, `cmpI_lw` | `Ord::cmp` on integers is a total order | `Order.lean`, `Laws.lean` |
| `lt_le_lt'`, `le_lt_lt'`, `eq_eq_eq'` | Mixed `<`/`≤`/`=` chains follow from antisymmetry and transitivity, on any subset of values | `Laws.lean` |
| `wrap_inRange`, `wrap_id` | Wrapping always lands in `i64` range and changes nothing on in-range values | `Int64.lean` |
| `fpcmp_num`, `cmpF_num` | On two integral floats, `partial_cmp` and `compare_floats` are plain integer comparison | `Float.lean` |

### How `lexA_trans` is proved

1. **Model the function.** `lexA c` compares two lists head to head with `thn (c x y) (lexA c xs ys)`; an empty list is smaller than a non-empty one.
2. **Weaken the hypothesis.** The element comparator `c` only needs to be lawful on elements satisfying a predicate `P`, and every element of the three lists must satisfy `P`. This matters for `value_order`. There, `sort_cmp` is only lawful on *well-formed* values, and the list arm is proven lawful by induction on value size, so `P` is "smaller than the current value".
3. **Induct on the three lists.** The cases where some list is empty follow from the definition. In the cons case, compare the heads (`c x y`, `c y z`) and split on the nine outcomes. `lt_le_lt'`, `le_lt_lt'` and `eq_eq_eq'` settle the cases where the heads decide, and when both head comparisons are `eq` the induction hypothesis on the tails finishes.

None of these theorems has a hypothesis about Rust code. `#print axioms` for `cmp3_lw` and `lexA_trans` shows only Lean's standard axioms.

## Bugs found

None. `core` models no Rust function on its own, so it can't find a bug by itself. The bugs its definitions helped find are recorded in the projects that use them.

## Files

| File | What's in it |
|---|---|
| `FalkorCore.lean` | Imports every module; the header with the module table and where each piece came from |
| `Value.lean` | `V F` (the engine `Value`, all 16 variants), `V.order`, `V.name`, `isNull`/`isNum`/`isList` |
| `Float.lean` | `F`, `F.nkey`, `fpcmp`, `Flag`, `cmpF`, `roundF` (`i as f64`), `bits64`/`bits32`, `asI64`, `diffZero` |
| `Int64.lean` | `I64MIN`/`I64MAX`, `InI64`, `wrap`, `wAdd`/`wSub`/`wMul`/`wDiv`/`wRem`, `satI64`, `w` |
| `Order.lean` | `cmpI`, `cmpN`, `cmpB` (`Ord::cmp` on `i64`/`usize`/`bool`), `cmp3`, and their swap/transitivity lemmas |
| `Laws.lean` | `Lw`, `thn`, `lexA`, `zipNe` and their laws; `cmp3_lw` |
| `COVERAGE.tsv` | Header only: `core` owns no Rust functions |

### Who uses it

Projects pull it in through a Lake path dependency in their `lakefile.toml`:

```toml
[[require]]
name = "falkor-core"
path = "../core"
```

| Project | Modules it imports |
|---|---|
| [`value_order`](../value_order) | all of `FalkorCore` (`F`, `fpcmp`, i64 helpers, `Lw`/`thn`/`lexA`) |
| [`value_math`](../value_math) | `Value` (`V F`), `Int64` |
| [`expr_semantics`](../expr_semantics) | `Int64`, `Float` (the `Flag` type) |
| [`functions_str_list`](../functions_str_list) | `Int64` |
| [`native_index`](../native_index) | `Float` |

## Coverage

`COVERAGE.tsv` has no rows. `core` models no Rust function on its own: every `value.rs` row it touches stays with the project that proves something about it (`value_order`, `value_math`, `expr_semantics`, …), and that project imports `FalkorCore`. So `core` adds nothing to the repo-wide counts in `bash proofs/coverage.sh`.

## What the model does not cover

- **Fractional floats.** A non-integral float is `frac lo r`. It is ordered correctly against everything else, but its IEEE bit pattern is abstract. No float arithmetic (`+`, `*`, rounding of a sum) is modelled; projects that need it take it as a parameter.
- **`V F` contents.** Strings are Lean `String`s; nodes and relationships are bare ids; temporal values are their `i64` seconds.
- **Not yet shared.** Several projects (`columnar`, `functions_temporal`, `runtime_core`, among others) still define their own `wrap`, `BATCH_SIZE` or restricted `V`. Moving them onto `core` is not done yet.

## Build and check

```bash
cd proofs/core && lake build    # this library
bash proofs/lean_ci.sh core     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh         # repo-wide coverage report
```

**When you change `core`:** rebuild every project that requires it (`value_order`, `value_math`, `expr_semantics`, `functions_str_list`, `native_index`), or run `bash proofs/lean_ci.sh` with no argument to build all of them.

**When you change the `Value` enum, `OrderedEnum::order`, `Value::name`, `compare_floats`, or the integer arithmetic in `value.rs`:** update the matching definition here first, then fix whatever stops building downstream. A new `Value` variant, for example, means a new `V` constructor and a new arm in `V.order` and `V.name`, and then every `match` over `V` in the dependent projects.
