import FalkorStrList.Misc
import FalkorStrList.Entity
import FalkorCore.Int64
import FalkorCore.Range
/-
# String and list functions: totality and spec agreement

A Lean 4 model of FalkorDB-rs's Cypher string and list functions
(`graph/src/runtime/functions/{string,list,internal}.rs`, eager `range` in
`math.rs`, and the list/regex/quantifier arms of `graph/src/runtime/eval.rs`),
checked against the openCypher spec and the C implementation (`master`
branch, `src/arithmetic/{string,list}_funcs/*.c`, `src/util/strutil.c`).
Build: `cd proofs/functions_str_list && lake build` (Lean 4.36, core only).
No `sorry`/`admit`/`axiom`; headline theorems use only `propext`,
`Quot.sound`, `Classical.choice`. Repros: `graph/tests/lean_functions_str_list.rs`
(`cargo test -p graph --test lean_functions_str_list -- --nocapture`).
Coverage per Rust fn: `COVERAGE.tsv`.

## What is modelled

| here | there |
| --- | --- |
| `InI64`, `wrap` (proofs/core), `asUsize`, `trunc32` | i64 range (debug panics outside), release wrap, `as usize`, C `(int32_t)` |
| `getElementIdx`, `getElement` | `ExprIR::GetElement` list arm, `eval.rs` |
| `sliceStart`, `sliceEnd`, `getElements` | `get_elements`, `eval.rs` |
| `cGetElement`, `cGetElements` | C `AR_SUBSCRIPT`, `AR_SLICE` (int32 truncation) |
| `removeSpan`, `listRemove`, `listRemoveSpec` | `remove_span` `list.rs`, `list.remove` `list.rs` (`listRemoveDebug`/`Release`: historical) |
| `insertNorm` | `list.insert` / `list.insertListElements`, `list.rs`, `list.rs` |
| `head`, `last`, `tail`, `size`, reverse | `list.rs` |
| `rangeLen`, `rangeElems`, `range` | eager `range`, `math.rs` |
| `FalkorCore.rangeIter` (`core/FalkorCore/Range.lean`, shared with expr_semantics) | `RangeIter::next`, `eval.rs`, built at `eval.rs` (debug: checked `+=`) |
| `evalQuantifier`, `quant`, `kand`/`kor`/`knot` | `eval_quantifier`, `eval.rs`; loop `eval.rs` |
| `containsLoop`, `contains` | `Contains for ThinVec<Value>`, `value.rs` (via `list_contains`, `eval.rs`) |
| `comprehensionRust` / `comprehensionSpec` | `ExprIR::ListComprehension`, `eval.rs` (literal/`range` source via `eval_iter_expr`, else `eval_node`) |
| `pre2954_comprehensionRust` | the earlier arm (every source through `eval_iter_expr`'s UNWIND fallback) |
| `reduceList` | `ExprIR::Reduce`, `eval.rs` |
| `Str`, `blen` | `String` as code points (`.chars()`), `s.len()` / `strlen` bytes |
| `substring2`, `substring3`, `cSubstring3` | `substring`, `string.rs`; C `AR_SUBSTRING` |
| `left`, `right`, `cLeft`, `cRight` | `string.rs`; C `AR_LEFT`, `AR_RIGHT` |
| `cReverseBytes` | C `AR_REVERSE` (writes each code point's bytes leftwards) |
| `ltrim`, `rtrim`, `trimRust`, `trimC` | `string.rs`; C `AR_LTRIM`/`AR_RTRIM`/`AR_TRIM` |
| `splitAux` | std `str::split` contract, used by `split`, `string.rs` |
| `findFirst`, `cSplit`, `splitC` | C `AR_SPLIT` |
| `splitRust` | `split`, `string.rs` |
| `replaceRust`, `cReplaceEmptyBytes` | `replace`, `string.rs` (std `str::replace`); C `AR_REPLACE` |
| `caseRust`, `caseC` | `toLower`/`toUpper`, `string.rs`; C `str_tolower` (utf8proc simple map) |
| `isMatchRust`, `isMatchSpec` | `internal_regex_matches` `internal.rs`, `RegexFnKind::Matches` `eval.rs` |
| `spliceLit`, `spliceRepl`, `callSite` | `Regex::replace_all(_, NoExpand(&str))` literal replacement, `string.rs`, `eval.rs`; `Repl.expand` is the default `&str` replacer the call sites no longer use |
| `joinLen`, `joinStr` | `compute_join_length` / `join_with_preallocate`, `string.rs` |
| `startsWith` | `internal_starts_with`, `internal.rs` (ends/contains analogous) |
| `cmpFloatModel` | `compare_floats` `value.rs` as used by `list.sort` `list.rs` |

## Proven (91 theorems in this file; main ones)

* Indexing `l[i]` is total for every i64 (incl. `i64::MIN`), `len+i` cannot
  overflow, and equals the spec (negative from the end, else null)
  (`getElement_noOverflow`, `getElement_inBounds`, `getElement_spec`).
* Slicing `l[a..b]` never panics: after the early return, `0 ≤ s ≤ e ≤ len`
  (`getElements_bounds`); it equals the clamped spec (`getElements_spec`).
* `list.remove`: (`remove_span`, saturating end) it equals the spec
  for all i64 inputs in every build and never panics (`listRemove_eq_spec`,
  `removeSpan_wf`, `listRemove_huge_count`). Historical: the earlier inline arithmetic
  matched the spec by accident in release (`listRemoveRelease_eq_spec`) and overflowed in debug.
* `list.insert*` normalisation can't overflow and yields a valid split point (`insertNorm_safe`).
* Eager `range`: every element lies between start and end, so no overflow
  for any i64 arguments (`range_elems_between`, `range_noOverflow`).
* Quantifiers: `all` = Kleene AND fold, `any` = Kleene OR fold,
  `none = NOT any`, `single` characterised (`all_is_kand`, `any_is_kor`,
  `none_is_not_any`, `single_spec`).
* `IN`: false iff no element compares Equal and none touches NULL (`contains_false_iff`).
* `substring`: the byte-length guard is harmless (`substring2_spec`,
  `substring3_spec`); Rust = C on all inputs (`substring3_agrees_C`).
* `left`/`right`: Rust = C on all inputs (`left_agrees_C`, `right_agrees_C`), spec (`left_right_spec`).
* `reverse` on strings: C's byte output = UTF-8 of Rust's `chars().rev()` (`reverse_agrees_C`).
* `trim`: Rust (`ltrim∘rtrim`) = C (`rtrim∘ltrim`) (`trim_agrees_C`); `ltrim_idem`, `ltrim_head`.
* `toLower`/`toUpper` never error (`toLower_total`) and equal C when the
  case mapping is simple (`case_agrees_C_if_simple`).
* `split`: std's scanner = C's find-first algorithm for every non-empty
  delimiter (`splitAux_eq_cSplit`), and all empty-string branches agree (`split_agrees_C`).
* `string.join`: when `compute_join_length` succeeds it equals the joined
  length, so the `debug_assert_eq!` holds (`joinLen_correct`).
* `size` counts code points, `size ≤ bytes`, additive under `+`.

## CONFIRMED bugs (Rust repro fails; outputs recorded live vs C on ports 18270/18271)

C. **`=~` is unanchored** — `internal_regex_matches` `internal.rs`, `eval.rs`
   (`Regex::is_match`): `'abc' =~ 'b'` → `true`, `'xabcx' =~ 'abc'` → `true`.
   openCypher/Neo4j require a whole-string match (→ `false`). C rejects `=~`
   ("FalkorDB does not currently support =~"). Test `bug_regex_match_unanchored`.
   Fix: compile `^(?:pat)$` (binder `rewrite_compiled_regex` and runtime path).

## Divergences recorded, not bugs (or C is wrong)

* Case mapping: Rust uses full Unicode mapping, C simple: `toUpper('ß')` Rust
  `'SS'` (size 2) vs C `'ẞ'`; `toLower('İ')` `'i̇'` vs `'i'`; `toLower('ΑΣ')`
  final sigma `'ας'` vs `'ασ'`; `toUpper('ﬁ')` `'FI'` vs `'ﬁ'`. Neo4j matches Rust.
* `string.matchRegEx('ab','(a)|(b)')`: Rust `null` for non-participating
  groups, C `''` (deliberate per the comment on `regex_captures_list`, `string.rs`).
* `'a' + (0.0/0.0)`: Rust `'aNaN'`, C `'anan'`.
* `STARTS WITH`/`ENDS WITH`/`CONTAINS` with a non-string: Rust `null` (openCypher), C error.
* C bugs: index/slice bounds truncated to int32 (`[1,2,3][4294967296]` → C `1`,
  `[1,2,3][MIN..MAX]` → C `[1, 2]`) (`cGetElement_truncates`, `cGetElements_truncates`);
  `replace(s,'',t)` inserts inside multibyte chars → invalid UTF-8
  (`cReplace_empty_breaks_utf8`); `range(0, 5000000000)` crashed the C server (Rust: "Range too large").
* Regex dialect: Rust `regex` crate vs C oniguruma Java syntax (no
  backreferences/lookaround in Rust) — not modelled.

## Gaps / assumptions

* Strings are lists of code points; UTF-8 byte-level facts used are
  `utf8Size ≥ 1` only. C's byte-wise `strstr`/`strncmp` matching is assumed to
  coincide with code-point matching for valid UTF-8 (self-synchronisation),
  which is not proven; `prefix_chars_to_bytes` proves one direction.
* std `str::split`/`str::replace`/`step_by`, the `regex` crate, and Unicode case
  tables are specified, not verified; the regex engine is an abstract span predicate.
* `Value::compare_value` is abstract here (see `proofs/value_order`).
* `keys`, `labels`, `properties` read graph attribute storage: not covered.
* `string.join` lengths are modelled as code-point lengths; Rust uses bytes,
  and the identity is the same.


## Wave 5 (`FalkorStrList/Entity.lean`, `FalkorStrList/Misc.lean`)

entity.rs (all 15 fns) over an accessor model (deleted snapshot > pending > committed),
plus the remaining string.rs / list.rs / internal.rs fns: all PROVEN. House decision
(deleted entities keep all details): `deleteRel_preserves`, `deleteNode_preserves`,
`deleteNode_labels_preserved`, `deleteNode_preserves_all`. W4-fn-1 (the DELETE snapshot copied
the committed labels, so labels staged by SET/REMOVE earlier in the same query were lost) is
the DELETE snapshot takes `Runtime::staged_label_ids`
(`ops/delete.rs`), the committed ids with the staged adds and removes applied.
`deleteNode_labels_preserved` proves the snapshot's labels equal what `labels(n)` / `n:L` showed
before the DELETE, with `staged_set_kept_after_delete` (`MATCH (n:A) SET n:C DELETE n RETURN
labels(n), n:C` is `['A','C']|true`) and `staged_remove_kept_after_delete` as the two shapes.
proofs/pending_commit `stagedLabelIds_eq_overlay` shows `staged_label_ids` is the overlay `pendingLabels` models.
`DeleteOp`'s snapshot flag is `runtime.inspect || snapshot_required(..)` (`Entity.lean`'s `deleteNode` /
`deleteRel` are the `snapshot = true` case).

`hasLabels` / `parse_degree_args` errors name the received type (`hasLabels_mismatch_names`,
`parseDegreeArgs_mismatch_names`, `parseDegreeArgs_examples`); `startNode(null)`/`endNode(null)` = null
(`startEndNode_null`, `startEndNode_total`). `toLower`/`toUpper` do not reject U+FFFD (`toLower_total`,
`case_agrees_C_if_simple`). `RangeIter` (exact `remaining` count, signed `step`, advance only while another
element is due): `rangeIter`, `rangeIter_of_inRange`, `rangeIter_eq_range` (lazy = eager `range`,
no overflow, every `i64` input), `rangeIter_examples`. `ListComprehension` evaluates a non-literal,
non-`range` source with `eval_node` (null → null, scalar → type error): `comprehensionRust` over `CSrc`,
`comprehension_eq_spec`, `comprehension_null`, `comprehension_scalar_err`,
`comprehension_literal_range_unchanged`. `Value::name` spells `Edge`/`Vectorf32` and `typeOf` reuses it
(`Entity.valueName`, `typeName_eq_valueName`). A non-boolean list-comprehension WHERE predicate is a
type error ("expected Boolean but was <name>") — `comprehensionLoop`, `whereKeep`,
`comprehensionLoop_nonbool`, `comprehensionFull_noWhere` (historical `pre3270_nonbool_skipped`,
`pre3270_loop_agrees`). `evalIterExpr` models all three `eval_iter_expr` arms (`evalIterExpr_range`,
`_lit`, `_other`), `evalCompiledRegex` all three `eval_compiled_regex` kinds with `checkStringOrNull`
(`regex_*`, `evalCompiledRegex_not_unreachable`; `=~` stays unanchored, bug C).
Others: `hasLabels_all`, `parseDegreeArgs_forms`, `strReplace_self`, `toStringVec_err`,
`capturesList_shape`, `listSort_perm`, `listDedup_spec`, `internalIsNull_spec`.
-/

namespace FalkorStrList
open FalkorCore

/-! ## Machine integers

Rust `i64` values are modelled as `Int` together with the range predicate
`InI64`. A debug build panics exactly when an arithmetic result leaves that
range; a release build (the `[profile.release]` in `Cargo.toml` sets no
`overflow-checks`) wraps, which is `wrap`. `asUsize` is Rust's `x as usize`
on a 64-bit target. -/

def I32MAX : Int := 2147483647
-- `I64MIN`/`I64MAX`, `InI64` (range) and `wrap` (release wrap) are proofs/core's
-- (`FalkorCore/Int64.lean`).
def asUsize (x : Int) : Nat := (x % 18446744073709551616).toNat
/-- C's `(int32_t)x` truncation, as used by `AR_SUBSCRIPT` / `AR_SLICE`. -/
def trunc32 (x : Int) : Int := (x + 2147483648) % 4294967296 - 2147483648

theorem asUsize_wrap64 (x : Int) (h0 : 0 ≤ x) (h1 : x < 18446744073709551616) :
    asUsize (wrap x) = x.toNat := by
  unfold asUsize wrap; omega

/-! ## `[l][i]` — `ExprIR::GetElement` (`eval.rs`) -/

/-- `eval.rs`: `normalized_index = if i < 0 { len + i } else { i }`,
then `values[normalized_index as usize]` under the guard `0 ≤ n < len`. -/
def getElementIdx (len : Nat) (i : Int) : Int := if i < 0 then (len : Int) + i else i

def getElement (l : List α) (i : Int) : Option α :=
  let n := getElementIdx l.length i
  if 0 ≤ n ∧ n < l.length then l[n.toNat]? else none

/-- `len + i` never overflows: `len ≤ i64::MAX` and `i < 0`. -/
theorem getElement_noOverflow (len : Nat) (i : Int) (hl : (len : Int) ≤ I64MAX)
    (hi : InI64 i) : InI64 (getElementIdx len i) := by
  unfold getElementIdx InI64 I64MAX at *; split <;> omega

/-- The unchecked `values[n as usize]` is in bounds whenever the guard holds, so
indexing is total (no panic for any `i : i64`, including `i64::MIN`). -/
theorem getElement_inBounds (l : List α) (i : Int)
    (h : 0 ≤ getElementIdx l.length i ∧ getElementIdx l.length i < l.length) :
    (getElementIdx l.length i).toNat < l.length := by omega

/-- Spec: non-negative indices count from the front, negative from the back
(`l[-1]` is the last element); anything outside `[-len, len)` is null. -/
theorem getElement_spec (l : List α) (i : Int) :
    getElement l i =
      if 0 ≤ i then l[i.toNat]?
      else if -(l.length : Int) ≤ i then l[(l.length + i).toNat]? else none := by
  unfold getElement getElementIdx
  by_cases h : i < 0
  · simp only [h, ite_true]
    have : ¬ 0 ≤ i := by omega
    simp only [this, ite_false]
    by_cases h2 : -(l.length : Int) ≤ i
    · simp only [h2, ite_true]; rw [ite_eq_left (by omega)]
    · simp only [h2, ite_false]; rw [ite_eq_right (by omega)]
  · simp only [h, ite_false]
    have : 0 ≤ i := by omega
    simp only [this, ite_true]
    by_cases h2 : i < l.length
    · simp [h2]
    · simp only [h2, and_false, ite_false]
      rw [List.getElem?_eq_none]; omega

/-- C truncates the index to `int32_t` before normalising (`list_funcs.c`,
`AR_SUBSCRIPT`). -/
def cGetElement (l : List α) (i : Int) : Option α := getElement l (trunc32 i)

/-- C and Rust disagree at `2^32`: C reads element 0, Rust returns null
(openCypher: null). Live: `RETURN [1,2,3][4294967296]` → C `1`, Rust `null`. -/
theorem cGetElement_truncates :
    cGetElement [1, 2, 3] 4294967296 = some 1 ∧ getElement [1, 2, 3] 4294967296 = none := by
  decide

/-! ## `[l][a..b]` — `get_elements` (`eval.rs`) -/

/-- `eval.rs`, the two normalisations, line by line. -/
def sliceStart (len : Nat) (s : Int) : Int := if s < 0 then max ((len : Int) + s) 0 else s
def sliceEnd (len : Nat) (e : Int) : Int := if e < 0 then max ((len : Int) + e) 0 else min e len

def getElements (l : List α) (s e : Int) : List α :=
  let s' := sliceStart l.length s
  let e' := sliceEnd l.length e
  if s' > e' then [] else (l.drop s'.toNat).take (e' - s').toNat

/-- Totality of `values[start as usize..end as usize]` (`eval.rs`): when
the `start > end` early return is not taken, `0 ≤ start ≤ end ≤ len`, so the
slice never panics — for every pair of `i64` bounds. -/
theorem getElements_bounds (len : Nat) (s e : Int) (h : ¬ sliceStart len s > sliceEnd len e) :
    0 ≤ sliceStart len s ∧ sliceStart len s ≤ sliceEnd len e ∧ sliceEnd len e ≤ len := by
  unfold sliceStart sliceEnd at *
  split <;> split <;> omega

theorem getElements_noOverflow (len : Nat) (x : Int) (hl : (len : Int) ≤ I64MAX)
    (hx : InI64 x) (hneg : x < 0) : InI64 ((len : Int) + x) := by
  simp only [InI64, I64MAX] at *; omega

/-- The clamp of the openCypher spec: negative bounds count from the end,
then everything is clamped into `[0, len]`. -/
def clampIdx (len : Nat) (x : Int) : Nat :=
  (if x < 0 then max ((len : Int) + x) 0 else min x len).toNat

/-- **Spec agreement.** `l[a..b]` is exactly the elements at positions
`clamp a ≤ k < clamp b`. Rust does not clamp `start` from above; the lemma
shows that an over-long start is caught by the `start > end` early return. -/
theorem getElements_spec (l : List α) (s e : Int) :
    getElements l s e =
      (l.drop (clampIdx l.length s)).take (clampIdx l.length e - clampIdx l.length s) := by
  unfold getElements sliceStart sliceEnd clampIdx
  by_cases hs : s < 0 <;> by_cases he : e < 0 <;> simp only [hs, he, ite_true, ite_false]
  all_goals
    split
    · -- early return: the spec side is empty too
      rename_i hgt
      by_cases hd : l.length ≤ (clampIdx l.length s)
      all_goals first
        | (rw [List.drop_eq_nil_of_le (by unfold clampIdx at *; simp only [hs] at *; omega)]; simp)
        | (rw [show ∀ a b : Nat, a ≤ b → a - b = 0 from fun a b h => by omega]; simp; omega)
    · rename_i hle
      congr 2 <;> omega

/-- C slice: both bounds truncated to `int32_t` (`list_funcs.c`, `AR_SLICE`).
Live: `RETURN [1,2,3][-9223372036854775808..9223372036854775807]` → C `[1, 2]`
(the bounds became `0..-1`), Rust `[1, 2, 3]` (spec). -/
def cGetElements (l : List α) (s e : Int) : List α := getElements l (trunc32 s) (trunc32 e)

theorem cGetElements_truncates :
    cGetElements [1, 2, 3] I64MIN I64MAX = [1, 2] ∧
    getElements [1, 2, 3] I64MIN I64MAX = [1, 2, 3] := by decide

/-! ## `list.remove(l, idx, count)` (`list.rs`, `remove_span` `list.rs`)

The span is computed by `remove_span`, which clamps the
end with `saturating_add`: `listRemove` / `removeSpan` below model that code.
`listRemoveDebug` / `listRemoveRelease` are kept as the **historical** model of
the earlier inline arithmetic (`(normalized + count) as usize`), whose debug
build overflowed. -/

inductive Out (α : Type) where
  | ok (l : List α)
  | panic (msg : String)
  deriving DecidableEq, Repr

def removeNorm (len : Nat) (idx : Int) : Int := if idx < 0 then (len : Int) + idx else idx

/-- Historical (pre-#2952) debug build: `(normalized + count)` was a checked add. -/
def listRemoveDebug (l : List α) (idx count : Int) : Out α :=
  let len : Int := l.length
  let n := removeNorm l.length idx
  if n < 0 ∨ n ≥ len ∨ count ≤ 0 then .ok l
  else if ¬ InI64 (n + count) then .panic "attempt to add with overflow"
  else
    let e := min (n + count).toNat l.length
    .ok (l.take n.toNat ++ l.drop e)

/-- Historical (pre-#2952) release build: the add wraps, then `as usize`, then `.min(vs.len())`. -/
def listRemoveRelease (l : List α) (idx count : Int) : List α :=
  let len : Int := l.length
  let n := removeNorm l.length idx
  if n < 0 ∨ n ≥ len ∨ count ≤ 0 then l
  else
    let e := min (asUsize (wrap (n + count))) l.length
    l.take n.toNat ++ l.drop e

/-- Spec (and C): drop `count` elements starting at the normalised index. -/
def listRemoveSpec (l : List α) (idx count : Int) : List α :=
  let len : Int := l.length
  let n := removeNorm l.length idx
  if n < 0 ∨ n ≥ len ∨ count ≤ 0 then l
  else l.take n.toNat ++ l.drop (n + count).toNat

/-- …but the release build is right by accident: a wrapped (negative) sum
becomes `≥ 2^63` under `as usize`, and `.min(len)` clamps it. Proven for every
list shorter than `i64::MAX` and every `i64` pair. -/
theorem removeNorm_range (len : Nat) (idx : Int) (hl : (len : Int) ≤ I64MAX) (hi : InI64 idx) :
    InI64 (removeNorm len idx) := by
  unfold removeNorm at *; simp only [InI64, I64MAX] at *; split <;> omega

theorem listRemoveRelease_eq_spec (l : List α) (idx count : Int)
    (hl : (l.length : Int) ≤ I64MAX) (hi : InI64 idx) (hc : InI64 count) :
    listRemoveRelease l idx count = listRemoveSpec l idx count := by
  have hn := removeNorm_range l.length idx hl hi
  unfold listRemoveRelease listRemoveSpec
  simp only []
  generalize removeNorm l.length idx = n at *
  simp only [InI64, I64MAX] at *
  split
  · rfl
  · rename_i hg
    have h0 : 0 ≤ n + count := by omega
    have h1 : n + count < 18446744073709551616 := by omega
    rw [asUsize_wrap64 _ h0 h1]
    congr 1
    by_cases hle : (n + count).toNat ≤ l.length
    · rw [Nat.min_eq_left hle]
    · rw [Nat.min_eq_right (by omega), List.drop_eq_nil_of_le (Nat.le_refl _),
        List.drop_eq_nil_of_le (by omega)]

/-- `i64::saturating_add`. -/
def satAdd64 (a b : Int) : Int :=
  if a + b > I64MAX then I64MAX else if a + b < I64MIN then I64MIN else a + b

/-- `remove_span` (`list.rs`): `None` when nothing is removed, else the
`start..end` span; `end = (normalized.saturating_add(count) as usize).min(len)`.
`normalized ≥ 0` there, so the saturated sum is non-negative and `as usize` is
the identity (`toNat`). -/
def removeSpan (len : Nat) (idx count : Int) : Option (Nat × Nat) :=
  let n := removeNorm len idx
  if n < 0 ∨ n ≥ (len : Int) ∨ count ≤ 0 then none
  else some (n.toNat, min (satAdd64 n count).toNat len)

/-- `list_remove` (`list.rs`) on a list argument: the original list when
`remove_span` is `None`, else `vs[..start] ++ vs[end..]`. The same in debug and
release: no arithmetic in it can overflow (`removeSpan_wf`). -/
def listRemove (l : List α) (idx count : Int) : List α :=
  match removeSpan l.length idx count with
  | none => l
  | some (s, e) => l.take s ++ l.drop e

/-- The unit test `remove_span_clamps_huge_count` (`list.rs`), as a theorem. -/
theorem removeSpan_clamps_huge_count :
    removeSpan 3 1 I64MAX = some (1, 3) ∧ removeSpan 3 (-1) I64MAX = some (2, 3) ∧
    removeSpan 3 0 2 = some (0, 2) ∧ removeSpan 3 3 1 = none ∧
    removeSpan 3 I64MIN 1 = none ∧ removeSpan 3 1 0 = none := by decide

/-- The span is well formed: `start < end ≤ len`, so `vs.len() - (end - start)`
and both slices are in bounds (no panic). -/
theorem removeSpan_wf (len : Nat) (idx count : Int) (s e : Nat)
    (hl : (len : Int) ≤ I64MAX)
    (h : removeSpan len idx count = some (s, e)) : s < e ∧ e ≤ len := by
  unfold removeSpan satAdd64 at h
  generalize removeNorm len idx = n at *
  unfold I64MAX I64MIN at *
  by_cases hg : n < 0 ∨ n ≥ (len : Int) ∨ count ≤ 0
  · rw [ite_eq_left hg] at h; cases h
  · rw [ite_eq_right hg] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    split <;> (try split) <;> omega

/-- **Bug B fixed.** The debug overflow input now returns C's `[1]`. -/
theorem listRemove_huge_count : listRemove [1, 2, 3] 1 I64MAX = [1] := by decide

/-- `list.remove` equals the spec (and C) for every list shorter than
`i64::MAX` and every `i64` index and count, in debug and release alike. -/
theorem listRemove_eq_spec (l : List α) (idx count : Int)
    (hl : (l.length : Int) ≤ I64MAX) (hi : InI64 idx) (hc : InI64 count) :
    listRemove l idx count = listRemoveSpec l idx count := by
  have hn := removeNorm_range l.length idx hl hi
  unfold listRemove listRemoveSpec removeSpan
  generalize removeNorm l.length idx = n at *
  simp only [InI64, I64MAX] at *
  by_cases hg : n < 0 ∨ n ≥ (l.length : Int) ∨ count ≤ 0
  · simp only [ite_eq_left hg]
  · simp only [ite_eq_right hg]
    have hsat : (satAdd64 n count).toNat = min (n + count).toNat 9223372036854775807 := by
      unfold satAdd64; unfold I64MAX I64MIN
      split <;> (try split) <;> omega
    rw [hsat]
    congr 1
    have hM : (n + count).toNat ≤ 9223372036854775807 ∨ l.length ≤ min (n + count).toNat 9223372036854775807 := by omega
    rcases hM with hM | hM
    · rw [Nat.min_eq_left hM]
      by_cases hle : (n + count).toNat ≤ l.length
      · rw [Nat.min_eq_left hle]
      · rw [Nat.min_eq_right (by omega), List.drop_eq_nil_of_le (Nat.le_refl _),
          List.drop_eq_nil_of_le (by omega)]
    · rw [List.drop_eq_nil_of_le (as := l) (by omega), List.drop_eq_nil_of_le (as := l) (by omega)]

/-- The fix agrees with the old release build wherever that was defined: the
historical release model already matched the spec (`listRemoveRelease_eq_spec`). -/
theorem listRemove_eq_preRelease (l : List α) (idx count : Int)
    (hl : (l.length : Int) ≤ I64MAX) (hi : InI64 idx) (hc : InI64 count) :
    listRemove l idx count = listRemoveRelease l idx count := by
  rw [listRemove_eq_spec l idx count hl hi hc, listRemoveRelease_eq_spec l idx count hl hi hc]

/-! ## `list.insert` / `list.insertListElements` normalisation
(`list.rs`, `list.rs`) -/

def insertNorm (len : Nat) (idx : Int) : Int := if idx < 0 then (len : Int) + idx + 1 else idx

/-- `len + idx + 1` never overflows, and when the range guard passes the
position is a valid split point for `vs[..pos]` / `vs[pos..]`. -/
theorem insertNorm_safe (len : Nat) (idx : Int) (hl : (len : Int) < I64MAX) (hi : InI64 idx) :
    InI64 (insertNorm len idx) ∧
    (¬ (insertNorm len idx < 0 ∨ insertNorm len idx > len) → (insertNorm len idx).toNat ≤ len) := by
  unfold insertNorm at *; simp only [InI64, I64MAX] at *
  constructor
  · split <;> omega
  · intro h; split at * <;> omega

/-- `-1` inserts after the last element (C's `bounds_inclusive`). -/
theorem insertNorm_minus_one (len : Nat) : insertNorm len (-1) = len := by
  unfold insertNorm; simp only [show (-1 : Int) < 0 by decide, ite_true]; omega

/-! ## `head` / `last` / `tail` / `reverse` / `size` on lists (`list.rs`) -/

def head (l : List α) : Option α := if l.isEmpty then none else l[0]?
def last (l : List α) : Option α := l.getLast?
def tail (l : List α) : List α := if l.isEmpty then [] else l.drop 1

theorem head_eq (l : List α) : head l = l.head? := by cases l <;> rfl
theorem tail_eq (l : List α) : tail l = l.tail := by cases l <;> rfl
theorem tail_length (l : List α) : (tail l).length = l.length - 1 := by
  rw [tail_eq]; simp
theorem last_eq_head_reverse (l : List α) : last l = l.reverse.head? := by
  unfold last; simp [List.head?_reverse]
theorem reverse_involutive (l : List α) : l.reverse.reverse = l := List.reverse_reverse l
theorem size_concat (a b : List α) : (a ++ b).length = a.length + b.length := List.length_append

/-! ## Eager `range(start, end, step)` (`math.rs`) -/

/-! `rangeLen` (`math.rs`, the length `|end - start| / |step| + 1`, computed in `i128`, so no
overflow is possible) and `rangeList` (the `step_by` elements `start + k*step`) are the shared
definitions in `proofs/core` (`FalkorCore/Range.lean`). `(start..=end).step_by(step)` yields
`start + k*step` while it stays inside the range and never computes an element outside it, so the
eager list is modelled directly. -/
def rangeElems (start stop step : Int) : List Int := rangeList start step (rangeLen start stop step)

inductive ROut where
  | ok (l : List Int)
  | err (msg : String)
  deriving DecidableEq, Repr

def range (start stop step : Int) : ROut :=
  if step = 0 then .err "ArgumentError: step argument to range() can't be 0"
  else if (start > stop ∧ step > 0) ∨ (start < stop ∧ step < 0) then .ok []
  else if rangeLen start stop step > 4294967295 then .err "Range too large"
  else .ok (rangeElems start stop step)

/-- Every element of `range` lies between `start` and `end`, hence inside
`i64`: the eager `range` cannot overflow for any `i64` arguments (contrast
with the pre-#2899 lazy `RangeIter`, #2894; the current one is `rangeIter_eq_range`). -/
theorem range_elems_between (start stop step : Int) (hs : step ≠ 0)
    (hdir : ¬ ((start > stop ∧ step > 0) ∨ (start < stop ∧ step < 0)))
    (x : Int) (hx : x ∈ rangeElems start stop step) :
    min start stop ≤ x ∧ x ≤ max start stop := by
  unfold rangeElems rangeList at hx
  simp only [List.mem_map, List.mem_range] at hx
  obtain ⟨k, hk, rfl⟩ := hx
  rw [rangeLen_eq_cases] at hk
  have hk' : k ≤ (if stop ≥ start then stop - start else start - stop).natAbs / step.natAbs := by omega
  have hmul : k * step.natAbs ≤ (if stop ≥ start then stop - start else start - stop).natAbs :=
    (Nat.le_div_iff_mul_le (by omega)).mp hk'
  have hmulI : ((k * step.natAbs : Nat) : Int) ≤
      ((if stop ≥ start then stop - start else start - stop).natAbs : Int) := by exact_mod_cast hmul
  push_cast at hmulI
  by_cases hsg : step > 0
  · have hab : (step.natAbs : Int) = step := by omega
    have hup : stop ≥ start := by omega
    simp only [hup, ite_true] at hmulI
    rw [hab] at hmulI
    have : ((stop - start).natAbs : Int) = stop - start := by omega
    rw [this] at hmulI
    have h0 : 0 ≤ (k : Int) * step := Int.mul_nonneg (by omega) (by omega)
    constructor <;> omega
  · have hneg : step < 0 := by omega
    have hab : (step.natAbs : Int) = -step := by omega
    have hdn : start ≥ stop := by omega
    rw [hab] at hmulI
    by_cases heq : stop ≥ start
    · have : start = stop := by omega
      subst this
      have hk0 : k = 0 := by simp at hk'; omega
      subst hk0; simp
    · simp only [heq, ite_false] at hmulI
      have : ((start - stop).natAbs : Int) = start - stop := by omega
      rw [this] at hmulI
      have h0 : 0 ≤ (k : Int) * -step := Int.mul_nonneg (by omega) (by omega)
      rw [Int.mul_neg] at hmulI h0
      constructor <;> omega

theorem range_noOverflow (start stop step : Int) (h1 : InI64 start) (h2 : InI64 stop)
    (l : List Int) (hr : range start stop step = .ok l) : ∀ x ∈ l, InI64 x := by
  intro x hx
  unfold range at hr
  split at hr
  · cases hr
  · split at hr
    · cases hr; simp at hx
    · split at hr
      · cases hr
      · cases hr
        rename_i hs hdir _
        have := range_elems_between start stop step hs hdir x hx
        simp only [InI64] at *; omega

/-- The two C/TCK checks: first and last elements. -/
theorem range_examples :
    range 0 10 3 = .ok [0, 3, 6, 9] ∧ range 10 0 (-3) = .ok [10, 7, 4, 1] ∧
    range 10 0 I64MIN = .ok [10] ∧ range 1 0 1 = .ok [] := ⟨rfl, rfl, rfl, rfl⟩

/-! ## Lazy `RangeIter::next`, earlier constructor (history)

The earlier constructor stored `step: step.unsigned_abs() as i64`; for `step = i64::MIN` that is
`i64::MIN` again, and `next` then evaluated `-self.step`, which overflowed. -/

/-! ## Lazy `RangeIter::next` (`eval.rs`)

The iterator model `rangeIter` (debug build, checked `+=`) and `rangeIter_of_inRange` are in
`proofs/core` (`FalkorCore/Range.lean`), shared with `expr_semantics`. -/

/-- **#2894 / #2899, fixed**: the iterator `eval_iter_expr` builds yields exactly the list
the eager `range(a, b, step)` returns whenever that list is non-empty (the empty, `step = 0`
and "too large" cases return before the iterator exists; `rangeLazy_eq_range` below covers them
too), and never overflows — including `step = i64::MIN` and ranges ending on
`i64::MAX`/`i64::MIN`. -/
theorem rangeIter_eq_range (start stop step : Int) (h1 : InI64 start) (h2 : InI64 stop)
    (l : List Int) (hr : range start stop step = .ok l) (hne : l ≠ []) :
    rangeIter step (rangeLen start stop step) start = .ok l := by
  unfold range at hr
  split at hr
  · cases hr
  · split at hr
    · cases hr; exact absurd rfl hne
    · split at hr
      · cases hr
      · cases hr
        rename_i hs hdir _
        refine (rangeIter_of_inRange step _ start ?_).1
        intro k hk
        have hx : start + k * step ∈ rangeElems start stop step := by
          unfold rangeElems; exact List.mem_map.mpr ⟨k, List.mem_range.mpr hk, rfl⟩
        have := range_elems_between start stop step hs hdir _ hx
        simp only [InI64] at *; omega

/-- `eval_iter_expr`'s `range(..)` arm (`eval.rs`) for integer arguments: the early
returns first (`step = 0` is an error, a direction-incompatible range is empty, a length above
`u32::MAX` is "Range too large"), then the `RangeIter` of `rangeLen` elements. A null or
non-integer argument is rejected by `validate_args_type` before this point and is not
modelled. -/
def rangeLazy (start stop step : Int) : ROut :=
  if step = 0 then .err "ArgumentError: step argument to range() can't be 0"
  else if (start > stop ∧ step > 0) ∨ (start < stop ∧ step < 0) then .ok []
  else if rangeLen start stop step > 4294967295 then .err "Range too large"
  else match rangeIter step (rangeLen start stop step) start with
    | .ok l => .ok l
    | .error m => .err m

/-- **Every `i64` input**: `UNWIND range(a, b, step)` is exactly the eager `range(a, b, step)`,
empty ranges, `step = 0` and "Range too large" included, and the iterator never overflows (it
could only turn a `.ok` into the `"attempt to add with overflow"` error). -/
theorem rangeLazy_eq_range (start stop step : Int) (h1 : InI64 start) (h2 : InI64 stop) :
    rangeLazy start stop step = range start stop step := by
  unfold rangeLazy range
  by_cases hs : step = 0
  · simp [hs]
  by_cases hd : (start > stop ∧ step > 0) ∨ (start < stop ∧ step < 0)
  · simp [hs, hd]
  by_cases hl : rangeLen start stop step > 4294967295
  · simp [hs, hd, hl]
  have key : rangeIter step (rangeLen start stop step) start = .ok (rangeElems start stop step) := by
    refine (rangeIter_of_inRange step _ start ?_).1
    intro k hk
    have hx : start + k * step ∈ rangeElems start stop step := by
      unfold rangeElems; exact List.mem_map.mpr ⟨k, List.mem_range.mpr hk, rfl⟩
    have := range_elems_between start stop step hs hd _ hx
    simp only [InI64] at *; omega
  simp [hs, hd, hl, key]

/-- The examples of `range_iter_stops_at_i64_bounds` (`eval.rs` unit test) and
the former debug panic `UNWIND range(10, 0, i64::MIN)`. -/
theorem rangeIter_examples :
    rangeIter 1 (rangeLen (I64MAX - 1) I64MAX 1) (I64MAX - 1) = .ok [I64MAX - 1, I64MAX] ∧
    rangeIter (-1) (rangeLen (I64MIN + 1) I64MIN (-1)) (I64MIN + 1) = .ok [I64MIN + 1, I64MIN] ∧
    rangeIter I64MIN (rangeLen 10 0 I64MIN) 10 = .ok [10] ∧
    rangeIter I64MAX 2 (-1) = .ok [-1, I64MAX - 1] := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rfl

/-! ## Quantifiers `all/any/none/single` (`eval_quantifier`, `eval.rs`) -/

/-- The three predicate outcomes the loop at `eval.rs` counts. -/
inductive K where | t | f | n
  deriving DecidableEq, Repr

inductive QType where | all | any | none | single
  deriving DecidableEq, Repr

def cnt (xs : List K) (k : K) : Nat := (xs.filter (· == k)).length

/-- `eval_quantifier`, branch for branch. `.n` result is `Value::Null`. -/
def evalQuantifier : QType → Nat → Nat → Nat → K
  | .all, _, f, n => if f > 0 then .f else if n > 0 then .n else .t
  | .any, t, _, n => if t > 0 then .t else if n > 0 then .n else .f
  | .none, t, _, n => if t > 0 then .f else if n > 0 then .n else .t
  | .single, t, _, n =>
      if t = 1 ∧ n = 0 then .t else if t > 1 then .f else if n > 0 then .n else .f

/-- Kleene three-valued AND / OR / NOT (openCypher's logic). -/
def kand : K → K → K
  | .f, _ => .f | _, .f => .f | .n, _ => .n | _, .n => .n | .t, .t => .t
def kor : K → K → K
  | .t, _ => .t | _, .t => .t | .n, _ => .n | _, .n => .n | .f, .f => .f
def knot : K → K | .t => .f | .f => .t | .n => .n

def quant (q : QType) (xs : List K) : K := evalQuantifier q (cnt xs .t) (cnt xs .f) (cnt xs .n)

theorem cnt_cons (x : K) (xs : List K) (k : K) :
    cnt (x :: xs) k = cnt xs k + (if x = k then 1 else 0) := by
  unfold cnt; cases x <;> cases k <;> simp <;> omega

theorem fold_kand (xs : List K) :
    xs.foldr kand .t = (if cnt xs .f > 0 then .f else if cnt xs .n > 0 then .n else .t) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.foldr_cons, ih]; simp only [cnt_cons]
    cases x <;> by_cases hf : cnt xs .f > 0 <;> by_cases hn : cnt xs .n > 0 <;> simp [hf, hn, kand] <;> omega

theorem fold_kor (xs : List K) :
    xs.foldr kor .f = (if cnt xs .t > 0 then .t else if cnt xs .n > 0 then .n else .f) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.foldr_cons, ih]; simp only [cnt_cons]
    cases x <;> by_cases hf : cnt xs .t > 0 <;> by_cases hn : cnt xs .n > 0 <;> simp [hf, hn, kor] <;> omega

/-- `all(x IN l WHERE p)` is the Kleene conjunction of the predicates. -/
theorem all_is_kand (xs : List K) : quant .all xs = xs.foldr kand .t := by
  rw [fold_kand]; rfl

/-- `any(x IN l WHERE p)` is the Kleene disjunction. -/
theorem any_is_kor (xs : List K) : quant .any xs = xs.foldr kor .f := by
  rw [fold_kor]; rfl

/-- `none(...) = NOT any(...)`. -/
theorem none_is_not_any (xs : List K) : quant .none xs = knot (quant .any xs) := by
  unfold quant evalQuantifier
  by_cases ht : cnt xs .t > 0 <;> by_cases hn : cnt xs .n > 0 <;> simp [ht, hn, knot]

/-- `single`: true iff exactly one true and no unknowns; false as soon as two
trues are seen (whatever the nulls); otherwise unknown if some null. -/
theorem single_spec (xs : List K) :
    quant .single xs =
      if cnt xs .t > 1 then .f
      else if cnt xs .n > 0 then .n
      else if cnt xs .t = 1 then .t else .f := by
  unfold quant evalQuantifier
  by_cases h1 : cnt xs .t > 1
  · simp only [h1, ite_true]; rw [ite_eq_right (by omega)]
  · simp only [h1, ite_false]
    by_cases h2 : cnt xs .n > 0
    · simp only [h2, ite_true]; rw [ite_eq_right (by omega)]
    · simp only [h2, ite_false]
      by_cases h3 : cnt xs .t = 1 <;> simp [h3] <;> omega

/-! ## `x IN list` — `Contains for ThinVec<Value>` (`value.rs`)

Parametric in the element comparison `cmp : α → α → (Ordering × Dis)`; the
comparison itself is `value_order`'s subject. -/

inductive Dis where | none | null | nan | disjoint
  deriving DecidableEq, Repr

/-- `value.rs`, with `is_null` accumulated left to right. -/
def containsLoop (cmp : α → Ordering × Dis) : Bool → List α → K
  | isNull, [] => if isNull then .n else .f
  | isNull, x :: xs =>
    let (res, dis) := cmp x
    let isNull := isNull || dis == .null
    if res == .eq then (if dis == .null then .n else .t) else containsLoop cmp isNull xs

def contains (cmp : α → Ordering × Dis) (xs : List α) : K := containsLoop cmp false xs

/-- `IN` is true iff some element compares `Equal` *definitely* with no
earlier element comparing `Equal` under a NULL; false iff nothing compares
`Equal` and no comparison touched NULL. -/
theorem contains_false_iff (cmp : α → Ordering × Dis) (xs : List α) :
    contains cmp xs = .f ↔ ∀ x ∈ xs, (cmp x).1 ≠ .eq ∧ (cmp x).2 ≠ .null := by
  unfold contains
  suffices h : ∀ b, containsLoop cmp b xs = .f ↔ b = false ∧ ∀ x ∈ xs, (cmp x).1 ≠ .eq ∧ (cmp x).2 ≠ .null by
    simp [h]
  induction xs with
  | nil => intro b; cases b <;> simp [containsLoop]
  | cons x xs ih =>
    intro b
    simp only [containsLoop, List.forall_mem_cons]
    rcases hc : cmp x with ⟨r, d⟩
    by_cases hr : r = .eq
    · subst hr; cases d <;> simp
    · have : (r == Ordering.eq) = false := by cases r <;> simp_all
      simp only [this, Bool.false_eq_true, ite_false, ih]
      cases b <;> cases d <;> simp [hr]

/-- The empty list never contains anything, not even null. -/
theorem contains_nil (cmp : α → Ordering × Dis) : contains cmp [] = .f := rfl

/-! ## Comprehension / reduce / quantifier sources

`[x IN src ...]` (`eval.rs`) streams a list literal or
`range(..)` through `eval_iter_expr` (`eval.rs`) and evaluates any other source with
`eval_node`: `List` iterates, `Null` gives `null`, anything else is a type error (an
earlier version sent every source through `eval_iter_expr`, whose fallback arm is UNWIND's). Quantifiers (`eval.rs`) and `reduce` already returned
`Null` for a null source and raised for a scalar, as does C. -/

inductive Src where | null | scalar (i : Int) | list (l : List Int)
  deriving DecidableEq, Repr

/-- The syntactic shape of a comprehension source after #2954: a list literal or a
`range(..)` call with integer arguments (both yield a list: the only cases modelled; a null or
non-integer `range` argument fails `validate_args_type` inside `eval_iter_expr` with a type
error, which is not modelled here), or any other expression, given by the value `eval_node`
returns. -/
inductive CSrc where | literal (l : List Int) | range (l : List Int) | other (v : Src)
  deriving DecidableEq, Repr

/-- The value a source expression denotes. -/
def CSrc.val : CSrc → Src
  | .literal l | .range l => .list l
  | .other v => v

inductive COut where | null | list (l : List Int) | err
  deriving DecidableEq, Repr

/-- `ListComprehension` before #2954 (no WHERE, identity projection): every source
through `eval_iter_expr`'s UNWIND fallback. Historical. -/
def pre2954_comprehensionRust : Src → COut
  | .null => .list []
  | .scalar i => .list [i]
  | .list l => .list l

/-- `ListComprehension` on main (no WHERE, identity projection): literal / `range`
stream, any other source is matched on its evaluated value. -/
def comprehensionRust : CSrc → COut
  | .literal l | .range l => .list l
  | .other .null => .null
  | .other (.scalar _) => .err
  | .other (.list l) => .list l

/-- C / openCypher, and Rust's own `reduce`/quantifier arms. -/
def comprehensionSpec : Src → COut
  | .null => .null
  | .scalar _ => .err
  | .list l => .list l

/-- Main agrees with C / openCypher on every source. -/
theorem comprehension_eq_spec (s : CSrc) : comprehensionRust s = comprehensionSpec s.val := by
  cases s with
  | literal l => rfl
  | range l => rfl
  | other v => cases v <;> rfl

/-- `[x IN null | x]` is `null`. -/
theorem comprehension_null : comprehensionRust (.other .null) = .null := rfl

/-- `[x IN 5 | x]` is a type error. -/
theorem comprehension_scalar_err (i : Int) : comprehensionRust (.other (.scalar i)) = .err := rfl

/-- List-valued sources are unchanged by #2954: a literal, a `range`, or any expression
evaluating to a list gives what the pre-#2954 code gave. -/
theorem comprehension_literal_range_unchanged (l : List Int) :
    comprehensionRust (.literal l) = pre2954_comprehensionRust (.list l) ∧
    comprehensionRust (.range l) = pre2954_comprehensionRust (.list l) ∧
    comprehensionRust (.other (.list l)) = pre2954_comprehensionRust (.list l) :=
  ⟨rfl, rfl, rfl⟩

theorem comprehension_agrees_on_lists (l : List Int) :
    pre2954_comprehensionRust (.list l) = comprehensionSpec (.list l) := rfl

/-! ### The per-element loop of `ListComprehension` (`eval.rs`): WHERE and projection

For each element `row.insert(var, value)`, the predicate `eval_node(child(1))?` must be
`true` (keep), `false` or `null` (skip); any other value is a type error
(an earlier version skipped every non-`true` result by `_ => continue`). A kept
element is projected by `eval_node(child(2))?`, and the first `Err` ends the loop. -/

/-- What a WHERE predicate evaluates to: a boolean outcome, or any other value (`Value::name`). -/
inductive PV where | t | f | n | other (name : String)
  deriving DecidableEq, Repr

/-- The `match` on the predicate result (`eval.rs`): `Ok(true)` = keep. -/
def whereKeep : PV → Except String Bool
  | .t => .ok true
  | .f | .n => .ok false
  | .other nm => .error s!"Type mismatch: expected Boolean but was {nm}"

/-- The same `match` before #3270: `_ => continue`. Historical. -/
def pre3270WhereKeep : PV → Bool
  | .t => true
  | _ => false

/-- The `for value in result` loop, one element at a time, errors propagating with `?`.
`pred`/`proj` are the element's `eval_node(child(1))` / `eval_node(child(2))`. -/
def comprehensionLoop {α β : Type} (pred : α → Except String PV) (proj : α → Except String β) :
    List α → Except String (List β)
  | [] => .ok []
  | x :: xs =>
    match pred x with
    | .error e => .error e
    | .ok p =>
      match whereKeep p with
      | .error e => .error e
      | .ok false => comprehensionLoop pred proj xs
      | .ok true =>
        match proj x with
        | .error e => .error e
        | .ok y =>
          match comprehensionLoop pred proj xs with
          | .error e => .error e
          | .ok rest => .ok (y :: rest)

/-- The loop before #3270. Historical. -/
def pre3270ComprehensionLoop {α β : Type} (pred : α → Except String PV)
    (proj : α → Except String β) : List α → Except String (List β)
  | [] => .ok []
  | x :: xs =>
    match pred x with
    | .error e => .error e
    | .ok p =>
      if pre3270WhereKeep p then
        match proj x with
        | .error e => .error e
        | .ok y =>
          match pre3270ComprehensionLoop pred proj xs with
          | .error e => .error e
          | .ok rest => .ok (y :: rest)
      else pre3270ComprehensionLoop pred proj xs

section loop
variable {α β : Type} (pred : α → Except String PV) (proj : α → Except String β)

theorem comprehensionLoop_nil : comprehensionLoop pred proj [] = .ok [] := rfl

/-- A `false` or `null` predicate skips the element. -/
theorem comprehensionLoop_skip (x : α) (xs : List α) (h : pred x = .ok .f ∨ pred x = .ok .n) :
    comprehensionLoop pred proj (x :: xs) = comprehensionLoop pred proj xs := by
  rcases h with h | h <;> simp [comprehensionLoop, h, whereKeep]

/-- **Fixed by #3270.** A predicate that is neither boolean nor null is a type error. -/
theorem comprehensionLoop_nonbool (x : α) (xs : List α) (nm : String)
    (h : pred x = .ok (.other nm)) :
    comprehensionLoop pred proj (x :: xs) =
      .error s!"Type mismatch: expected Boolean but was {nm}" := by
  simp [comprehensionLoop, h, whereKeep]

/-- A predicate error ends the loop with that error. -/
theorem comprehensionLoop_pred_error (x : α) (xs : List α) (e : String)
    (h : pred x = .error e) : comprehensionLoop pred proj (x :: xs) = .error e := by
  simp [comprehensionLoop, h]

/-- A `true` predicate keeps the element: project it, then the rest. -/
theorem comprehensionLoop_keep (x : α) (xs : List α) (h : pred x = .ok .t) :
    comprehensionLoop pred proj (x :: xs) =
      (proj x).bind fun y => (comprehensionLoop pred proj xs).map (y :: ·) := by
  simp only [comprehensionLoop, h, whereKeep]
  cases proj x <;> cases comprehensionLoop pred proj xs <;> rfl

/-- No WHERE (`child(1)` is the constant `true`) and an identity projection: the list itself. -/
theorem comprehensionLoop_true_id (l : List α) :
    comprehensionLoop (fun _ => (.ok .t : Except String PV)) (fun x => (.ok x : Except String α)) l
      = .ok l := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [comprehensionLoop, whereKeep, ih]

/-- Before #3270 a non-boolean predicate was silently skipped; main raises. The two loops
agree whenever every predicate result is boolean or null. -/
theorem pre3270_loop_agrees (l : List α) (h : ∀ x ∈ l, ∀ nm, pred x ≠ .ok (.other nm)) :
    pre3270ComprehensionLoop pred proj l = comprehensionLoop pred proj l := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    have hx := h x (by simp)
    have ih' := ih (fun y hy => h y (by simp [hy]))
    simp only [pre3270ComprehensionLoop, comprehensionLoop]
    cases hp : pred x with
    | error e => rfl
    | ok p =>
      cases p with
      | t => simp [pre3270WhereKeep, whereKeep, ih']
      | f => simp [pre3270WhereKeep, whereKeep, ih']
      | n => simp [pre3270WhereKeep, whereKeep, ih']
      | other nm => exact absurd hp (hx nm)

end loop

/-- `[x IN [1] WHERE 5 | x]`: the pre-#3270 loop returned `[]`, main raises. -/
theorem pre3270_nonbool_skipped :
    pre3270ComprehensionLoop (fun _ => (.ok (.other "Integer") : Except String PV))
      (fun x => (.ok x : Except String Int)) [1] = .ok [] ∧
    comprehensionLoop (fun _ => (.ok (.other "Integer") : Except String PV))
      (fun x => (.ok x : Except String Int)) [1] =
      .error "Type mismatch: expected Boolean but was Integer" := ⟨rfl, rfl⟩

/-- The whole `ListComprehension` arm: the source dispatch of `comprehensionRust`
(`none` = the arm pushes `Value::Null`) then the element loop. -/
def comprehensionFull {β : Type} (pred : Int → Except String PV) (proj : Int → Except String β) :
    CSrc → Except String (Option (List β))
  | .literal l | .range l => (comprehensionLoop pred proj l).map some
  | .other .null => .ok none
  | .other (.scalar _) => .error "Type mismatch: expected List or Null but was Integer"
  | .other (.list l) => (comprehensionLoop pred proj l).map some

/-- With no WHERE and an identity projection the full arm is `comprehensionRust`. -/
theorem comprehensionFull_noWhere (s : CSrc) :
    comprehensionFull (fun _ => (.ok .t : Except String PV))
      (fun x => (.ok x : Except String Int)) s =
    match comprehensionRust s with
    | .list l => .ok (some l)
    | .null => .ok none
    | .err => .error "Type mismatch: expected List or Null but was Integer" := by
  cases s with
  | literal l => simp [comprehensionFull, comprehensionLoop_true_id, comprehensionRust, Except.map]
  | range l => simp [comprehensionFull, comprehensionLoop_true_id, comprehensionRust, Except.map]
  | other v =>
    cases v <;> simp [comprehensionFull, comprehensionLoop_true_id, comprehensionRust, Except.map]

/-! ### `eval_iter_expr` (`eval.rs`): the three source shapes

`range(..)` (arguments already `eval_node`d and `validate_args_type`d: an error from either is
the `Except`), a list literal (each child `eval_node`d in order, `?` on the first error) and any
other expression (the UNWIND fallback). `none` is `Ok(None)` ("the row produces no rows"). -/

inductive IVal where | null | list (l : List Int) | scalar (a : Int)
  deriving DecidableEq, Repr

inductive IterSrc where
  | range (start stop step : Except String Int)
  | lit (elems : List (Except String Int))
  | other (v : Except String IVal)

def evalIterExpr : IterSrc → Except String (Option (List Int))
  | .range start stop step =>
    match start, stop, step with
    | .error e, _, _ => .error e
    | _, .error e, _ => .error e
    | _, _, .error e => .error e
    | .ok a, .ok b, .ok c =>
      match rangeLazy a b c with
      | .err m => .error m
      | .ok [] => .ok none
      | .ok l => .ok (some l)
  | .lit elems => (elems.mapM id).map some
  | .other v =>
    match v with
    | .error e => .error e
    | .ok (.list l) => .ok (some l)
    | .ok .null => .ok none
    | .ok (.scalar a) => .ok (some [a])

/-- A list literal with all elements fine yields them, in order (an empty literal is `Some([])`). -/
theorem evalIterExpr_lit (l : List Int) :
    evalIterExpr (.lit (l.map Except.ok)) = .ok (some l) := by
  have : (l.map (Except.ok : Int → Except String Int)).mapM id = .ok l := by
    induction l with
    | nil => rfl
    | cons x xs ih =>
      simp [List.mapM_cons, List.map, ih, Bind.bind, Except.bind]
      rfl
  simp [evalIterExpr, this, Except.map]

/-- The first failing literal element is the error. -/
theorem evalIterExpr_lit_error (e : String) :
    evalIterExpr (.lit [.error e]) = .error e := rfl

/-- The UNWIND fallback: a list iterates, null gives no rows, a scalar one row. -/
theorem evalIterExpr_other (l : List Int) (a : Int) :
    evalIterExpr (.other (.ok (.list l))) = .ok (some l) ∧
    evalIterExpr (.other (.ok .null)) = .ok none ∧
    evalIterExpr (.other (.ok (.scalar a))) = .ok (some [a]) := ⟨rfl, rfl, rfl⟩

/-- Argument errors propagate left to right. -/
theorem evalIterExpr_range_arg_error (e : String) (b c : Except String Int) :
    evalIterExpr (.range (.error e) b c) = .error e := rfl

/-- `range(a, b, step)` over `i64` arguments: the streamed rows are the eager `range` list
(`rangeLazy_eq_range`); an empty `range` is `None`, "no rows". -/
theorem evalIterExpr_range (a b c : Int) (h1 : InI64 a) (h2 : InI64 b) :
    evalIterExpr (.range (.ok a) (.ok b) (.ok c)) =
      match range a b c with
      | .err m => .error m
      | .ok [] => .ok none
      | .ok l => .ok (some l) := by
  simp only [evalIterExpr]
  rw [rangeLazy_eq_range a b c h1 h2]

/-- `reduce(acc = init, x IN l | body)` (`eval.rs`) is `foldl`. -/
def reduceList (body : β → α → β) (init : β) (l : List α) : β := l.foldl body init
theorem reduce_nil (body : β → α → β) (init : β) : reduceList body init [] = init := rfl
theorem reduce_append (body : β → α → β) (init : β) (a b : List α) :
    reduceList body init (a ++ b) = reduceList body (reduceList body init a) b := List.foldl_append

/-! ## Strings

A Rust `String` is valid UTF-8, so it is modelled by its code points
(`Str := List Char`, what `.chars()` yields). Byte-level quantities — `s.len()`
in Rust, `strlen` in C — are `blen`, the sum of `Char.utf8Size`. The
bytes themselves are `String.utf8EncodeChar`. -/

abbrev Str := List Char

def blen (s : Str) : Nat := (s.map Char.utf8Size).sum

theorem blen_cons (c : Char) (s : Str) : blen (c :: s) = c.utf8Size + blen s := by
  simp [blen]

theorem length_le_blen (s : Str) : s.length ≤ blen s := by
  induction s with
  | nil => simp [blen]
  | cons c s ih => rw [blen_cons]; have := Char.utf8Size_pos c; simp; omega

/-- Dropping `k ≤ length` code points removes at least `k` bytes. -/
theorem blen_drop_le (s : Str) (k : Nat) : blen (s.drop k) + k ≤ blen s + (k - s.length) := by
  induction s generalizing k with
  | nil => simp [blen]
  | cons c s ih =>
    cases k with
    | zero => simp
    | succ k =>
      simp only [List.drop_succ_cons, blen_cons, List.length_cons]
      have := ih k; have := Char.utf8Size_pos c; omega

theorem length_drop_le_bytes (s : Str) (k : Nat) (hk : k < blen s) :
    (s.drop k).length ≤ blen s - k := by
  have h1 := length_le_blen (s.drop k)
  have h2 := blen_drop_le s k
  by_cases hkl : k ≤ s.length
  · omega
  · rw [List.drop_eq_nil_of_le (by omega)]; simp

inductive SOut where
  | ok (s : Str)
  | err (msg : String)
  deriving DecidableEq, Repr

/-! ### `substring` (`string.rs`) -/

/-- `string.rs`. Note the guard compares the *code point* index against
the *byte* length `s.len()`. -/
def substring2 (s : Str) (start : Int) : SOut :=
  if start < 0 then .err "start must be a non-negative integer"
  else if start ≥ blen s then .ok []
  else .ok (s.drop start.toNat)

/-- `string.rs`. -/
def substring3 (s : Str) (start length : Int) : SOut :=
  if start < 0 then .err "start must be a non-negative integer"
  else if start.toNat ≥ blen s then .ok []
  else if length < 0 then .err "length must be a non-negative integer"
  else .ok ((s.drop start.toNat).take length.toNat)

/-- C `AR_SUBSCRIPT`'s sibling `AR_SUBSTRING` (`string_funcs.c`): byte-length
guard, `length = MIN(length, suffix_len)` with `suffix_len` in *bytes*, then
code-point walks. -/
def cSubstring3 (s : Str) (start length : Int) : SOut :=
  if start < 0 then .err "start must be a non-negative integer"
  else if start ≥ blen s then .ok []
  else if length < 0 then .err "length must be a non-negative integer"
  else .ok ((s.drop start.toNat).take (min length.toNat (blen s - start.toNat)))

/-- **Spec** (openCypher, 0-based code points): the byte-length guard is
harmless — past the last code point `drop` is already empty. -/
theorem substring2_spec (s : Str) (start : Int) :
    substring2 s start =
      if start < 0 then .err "start must be a non-negative integer" else .ok (s.drop start.toNat) := by
  unfold substring2
  split
  · rfl
  · split
    · rw [List.drop_eq_nil_of_le]; have := length_le_blen s; omega
    · rfl

/-- Rust and C agree on every input: `take (min len suffix_bytes)` is
`take len` because a suffix has no more code points than bytes. -/
theorem substring3_agrees_C (s : Str) (start length : Int) :
    substring3 s start length = cSubstring3 s start length := by
  unfold substring3 cSubstring3
  split
  · rfl
  · rename_i h0
    have e : (start.toNat ≥ blen s) ↔ (start ≥ (blen s : Int)) := by omega
    by_cases hb : start ≥ (blen s : Int)
    · rw [ite_eq_left (e.mpr hb), ite_eq_left hb]
    · rw [ite_eq_right (fun h => hb (e.mp h)), ite_eq_right hb]
      split
      · rfl
      · congr 1
        have hk : start.toNat < blen s := by omega
        have := length_drop_le_bytes s start.toNat hk
        by_cases hm : length.toNat ≤ blen s - start.toNat
        · rw [Nat.min_eq_left hm]
        · rw [Nat.min_eq_right (by omega), List.take_of_length_le (by omega),
            List.take_of_length_le (by omega)]

theorem substring3_spec (s : Str) (start length : Int) (h0 : 0 ≤ start) (hl : 0 ≤ length) :
    substring3 s start length = .ok ((s.drop start.toNat).take length.toNat) := by
  unfold substring3
  rw [ite_eq_right (by omega)]
  split
  · rw [List.drop_eq_nil_of_le]; · simp
    have := length_le_blen s; omega
  · rw [ite_eq_right (by omega)]

/-- Shared divergence (Rust = C): a negative length is *not* rejected when the
start is already past the end — `substring('abc', 10, -1)` is `''`. -/
theorem substring3_negative_length_past_end :
    substring3 ['a', 'b', 'c'] 10 (-1) = .ok [] ∧
    substring3 ['a', 'b', 'c'] 0 (-1) = .err "length must be a non-negative integer" := by decide

/-- UTF-8: `substring('héllo', 1, 3)` is `'éll'` (code points, not bytes). -/
theorem substring_multibyte :
    substring3 "héllo".toList 1 3 = .ok "éll".toList ∧ substring2 "héllo".toList 5 = .ok [] := by
  decide

/-! ### `left` / `right` (`string.rs`) -/

def left (s : Str) (n : Int) : SOut :=
  if n < 0 then .err "length must be a non-negative integer" else .ok (s.take n.toNat)

/-- `string.rs`: `chars().count().saturating_sub(n as usize)`. -/
def right (s : Str) (n : Int) : SOut :=
  if n < 0 then .err "length must be a non-negative integer"
  else .ok (s.drop (s.length - n.toNat))

/-- C `AR_LEFT`: `strlen(s) <= newlen` (bytes) returns the whole string. -/
def cLeft (s : Str) (n : Int) : SOut :=
  if n < 0 then .err "length must be a non-negative integer"
  else if (blen s : Int) ≤ n then .ok s else .ok (s.take n.toNat)

/-- C `AR_RIGHT`: `start = str_length(s) - newlen` in `int64_t`. -/
def cRight (s : Str) (n : Int) : SOut :=
  if n < 0 then .err "length must be a non-negative integer"
  else if (s.length : Int) - n ≤ 0 then .ok s else .ok (s.drop ((s.length : Int) - n).toNat)

theorem left_agrees_C (s : Str) (n : Int) : left s n = cLeft s n := by
  unfold left cLeft
  split
  · rfl
  · split
    · rw [List.take_of_length_le]; have := length_le_blen s; omega
    · rfl

theorem right_agrees_C (s : Str) (n : Int) : right s n = cRight s n := by
  unfold right cRight
  split
  · rfl
  · split
    · rw [show s.length - n.toNat = 0 by omega]; rfl
    · congr 2; omega

theorem left_right_spec (s : Str) (n : Nat) :
    left s n = .ok (s.take n) ∧ right s n = .ok (s.drop (s.length - n)) ∧
    (s.drop (s.length - n)).length = min n s.length ∧
    s.take (s.length - n) ++ s.drop (s.length - n) = s := by
  refine ⟨?_, ?_, ?_, List.take_append_drop _ _⟩
  · unfold left; simp
  · unfold right; simp
  · simp; omega

/-! ### `size` / `reverse` (`list.rs`, `list.rs`) -/

/-- `size(s)` counts code points, like C's `str_length`. -/
def size (s : Str) : Nat := s.length

theorem str_size_concat (a b : Str) : size (a ++ b) = size a + size b := List.length_append
theorem size_reverse (s : Str) : size s.reverse = size s := List.length_reverse
theorem size_le_bytes (s : Str) : size s ≤ blen s := length_le_blen s

/-- C `AR_REVERSE` walks code points forward and writes each one's bytes
immediately to the left of what it already wrote. -/
def cReverseBytes (enc : Char → List UInt8) : Str → List UInt8 → List UInt8
  | [], acc => acc
  | c :: t, acc => cReverseBytes enc t (enc c ++ acc)

theorem cReverseBytes_eq (enc : Char → List UInt8) (s : Str) (acc : List UInt8) :
    cReverseBytes enc s acc = s.reverse.flatMap enc ++ acc := by
  induction s generalizing acc with
  | nil => rfl
  | cons c t ih => simp [cReverseBytes, ih, List.flatMap_append]

/-- **Rust and C reverse agree byte for byte**: C's output is the UTF-8
encoding of `s.chars().rev()`. -/
theorem reverse_agrees_C (s : Str) :
    cReverseBytes String.utf8EncodeChar s [] = s.reverse.flatMap String.utf8EncodeChar := by
  rw [cReverseBytes_eq]; simp

/-- Both reverse *code points*, not grapheme clusters: a combining accent
moves to the other letter (`reverse('éa')` = `'áe'`, live on both). -/
theorem reverse_moves_combining_mark :
    ['e', '́', 'a'].reverse = ['a', '́', 'e'] := by decide

/-! ### `trim` / `ltrim` / `rtrim` (`string.rs`) -/

def isSp (c : Char) : Bool := c == ' '
/-- `trim_start_matches(' ')` and C `AR_LTRIM`. Only U+0020 is trimmed. -/
def ltrim (s : Str) : Str := s.dropWhile isSp
/-- `trim_end_matches(' ')` and C `AR_RTRIM`. -/
def rtrim (s : Str) : Str := (s.reverse.dropWhile isSp).reverse
/-- Rust `trim_matches(' ')` strips the end first… -/
def trimRust (s : Str) : Str := ltrim (rtrim s)
/-- …C `AR_TRIM` is `AR_RTRIM(AR_LTRIM(s))`. -/
def trimC (s : Str) : Str := rtrim (ltrim s)

theorem rtrim_cons (c : Char) (t : Str) :
    rtrim (c :: t) = if isSp c ∧ rtrim t = [] then [] else c :: rtrim t := by
  unfold rtrim
  rw [List.reverse_cons, List.dropWhile_append]
  by_cases he : (List.dropWhile isSp t.reverse).isEmpty
  · have he' : List.dropWhile isSp t.reverse = [] := List.isEmpty_iff.mp he
    rw [ite_eq_left he, he']
    by_cases hc : isSp c
    · simp [List.dropWhile, hc]
    · simp [List.dropWhile, hc]
  · have he' : List.dropWhile isSp t.reverse ≠ [] := fun h => he (List.isEmpty_iff.mpr h)
    rw [ite_eq_right he]
    simp [he']

theorem trim_agrees_C (s : Str) : trimRust s = trimC s := by
  unfold trimRust trimC
  induction s with
  | nil => rfl
  | cons c t ih =>
    rw [rtrim_cons]
    by_cases hc : isSp c
    · have hl : ltrim (c :: t) = ltrim t := by simp [ltrim, List.dropWhile, hc]
      rw [hl, ← ih]
      by_cases hr : rtrim t = []
      · simp [hc, hr, ltrim]
      · simp [hc, hr, ltrim]
    · have hl : ltrim (c :: t) = c :: t := by simp [ltrim, List.dropWhile, hc]
      rw [hl, rtrim_cons]
      simp [hc, ltrim]

theorem ltrim_head (s : Str) : ∀ c, (ltrim s).head? = some c → c ≠ ' ' := by
  intro c h
  unfold ltrim at h
  have := List.head_dropWhile_not (p := isSp) (l := s)
  induction s with
  | nil => simp at h
  | cons x xs ih =>
    by_cases hx : isSp x
    · simp [List.dropWhile, hx] at h; exact ih h (by
        intro w; exact List.head_dropWhile_not (p := isSp) (l := xs) w)
    · simp [List.dropWhile, hx] at h; subst h; simpa [isSp] using hx

theorem ltrim_idem (s : Str) : ltrim (ltrim s) = ltrim s := by
  unfold ltrim
  induction s with
  | nil => rfl
  | cons x xs ih =>
    by_cases hx : isSp x
    · simp only [List.dropWhile, hx]; exact ih
    · simp [List.dropWhile, hx]

/-- Only spaces: tabs and newlines survive (both engines). -/
theorem trim_only_spaces : trimRust ['\t', ' ', 'a', '\n'] = ['\t', ' ', 'a', '\n'] := by decide

/-! ### `split(s, delim)` (`string.rs`)

Rust delegates to std `str::split`, whose contract is: the pieces between
non-overlapping matches found left to right. `splitAux` is that contract
written as a character-at-a-time scanner (`cur` holds the piece so far,
reversed). C `AR_SPLIT` (`string_funcs.c`) is a different algorithm — find
the first match, emit the prefix, jump past the delimiter — and it is
`cSplit`. Its final `if(rest_len > 0 || delimiter_found)` push always fires for
a non-empty input: the loop is left either by `break` (not found, so
`rest_len ≥ dlen > 0`), by the loop condition after a successful iteration
(`delimiter_found`), or never entered (`rest_len = str_len > 0`); `cSplit`
therefore always emits the rest. C matches bytes and Rust code points; for a
valid UTF-8 delimiter the two coincide because UTF-8 is self-synchronising —
that step is assumed, not proven (see gaps). -/

theorem isPrefixOf_length {d s : Str} (h : d.isPrefixOf s = true) : d.length ≤ s.length :=
  (List.isPrefixOf_iff_prefix.mp h).length_le

def splitAux (d : Str) (hd : d ≠ []) (s cur : Str) : List Str :=
  if h : d.isPrefixOf s then cur.reverse :: splitAux d hd (s.drop d.length) []
  else match s with
    | [] => [cur.reverse]
    | c :: t => splitAux d hd t (c :: cur)
termination_by s.length
decreasing_by
  · have := isPrefixOf_length h
    have : d.length ≠ 0 := fun h0 => hd (List.length_eq_zero_iff.mp h0)
    simp only [List.length_drop]; omega
  · simp

/-- First position where `d` occurs (`strncmp(start + len, d, dlen) == 0`). -/
def findFirst (d : Str) : Str → Option Nat
  | [] => if d.isPrefixOf [] then some 0 else none
  | c :: t => if d.isPrefixOf (c :: t) then some 0 else (findFirst d t).map (· + 1)

theorem findFirst_le (d s : Str) (i : Nat) (h : findFirst d s = some i) : i + d.length ≤ s.length := by
  induction s generalizing i with
  | nil =>
    simp only [findFirst] at h; split at h
    · rename_i hp; cases h; have := isPrefixOf_length hp; simpa using this
    · cases h
  | cons c t ih =>
    simp only [findFirst] at h; split at h
    · rename_i hp; cases h; have := isPrefixOf_length hp; simpa using this
    · cases hf : findFirst d t with
      | none => simp [hf] at h
      | some j => simp [hf] at h; subst h; have := ih j hf; simp; omega

def cSplit (d : Str) (hd : d ≠ []) (s : Str) : List Str :=
  match hf : findFirst d s with
  | none => [s]
  | some i => s.take i :: cSplit d hd (s.drop (i + d.length))
termination_by s.length
decreasing_by
  have := findFirst_le d s i hf
  have : d.length ≠ 0 := fun h0 => hd (List.length_eq_zero_iff.mp h0)
  simp only [List.length_drop]; omega

/-- One unfolding of the scanner, stated against `findFirst`. -/
theorem splitAux_step (d : Str) (hd : d ≠ []) (s cur : Str) :
    splitAux d hd s cur =
      match findFirst d s with
      | none => [cur.reverse ++ s]
      | some i => (cur.reverse ++ s.take i) :: splitAux d hd (s.drop (i + d.length)) [] := by
  induction s generalizing cur with
  | nil =>
    rw [splitAux]
    simp only [findFirst]
    split <;> rename_i h <;> simp
  | cons c t ih =>
    rw [splitAux]
    simp only [findFirst]
    by_cases h : d.isPrefixOf (c :: t) = true
    · simp [h]
    · simp only [h, dite_false, ite_false, Bool.false_eq_true]
      rw [ih]
      cases findFirst d t with
      | none => simp
      | some j =>
        simp only [Option.map_some, List.reverse_cons, List.append_assoc, List.singleton_append,
          List.take_succ_cons, List.cons.injEq]
        rw [show j + 1 + d.length = (j + d.length) + 1 by omega, List.drop_succ_cons]
        try simp

/-- **Rust's `split` and C's `AR_SPLIT` agree** for every non-empty delimiter. -/
theorem splitAux_eq_cSplit (d : Str) (hd : d ≠ []) (s : Str) :
    splitAux d hd s [] = cSplit d hd s := by
  rw [splitAux_step, cSplit]
  split
  · split <;> simp_all
  · rename_i i hf
    split
    · simp_all
    · rename_i j hj
      have hij : i = j := by simp_all
      subst hij
      simp only [List.reverse_nil, List.nil_append]
      rw [splitAux_eq_cSplit d hd (s.drop (i + d.length))]
termination_by s.length
decreasing_by
  have := findFirst_le d s _ (by assumption)
  have : d.length ≠ 0 := fun h0 => hd (List.length_eq_zero_iff.mp h0)
  simp only [List.length_drop]; omega

/-- `string.rs`. -/
def splitRust (s d : Str) : List Str :=
  if s = [] then [[]]
  else if hd : d = [] then s.map (fun c => [c])
  else splitAux d hd s []

/-- C `AR_SPLIT`, branch order as written (delimiter first). -/
def splitC (s d : Str) : List Str :=
  if hd : d = [] then (if s = [] then [[]] else s.map (fun c => [c]))
  else if s = [] then [[]]
  else cSplit d hd s

theorem split_agrees_C (s d : Str) : splitRust s d = splitC s d := by
  unfold splitRust splitC
  by_cases hs : s = [] <;> by_cases hd : d = [] <;> simp [hs, hd, splitAux_eq_cSplit]

theorem split_examples :
    splitRust "a,b,".toList [','] = ["a".toList, "b".toList, []] ∧
    splitRust "aaa".toList "aa".toList = [[], ['a']] ∧
    splitRust [] [','] = [[]] ∧
    splitRust "hé".toList [] = [['h'], ['é']] := by decide +kernel

/-! ### `replace(s, search, repl)` (`string.rs`)

std `str::replace` with a non-empty pattern is `split` then join; with an
empty pattern it inserts `repl` before every code point and at the end. C
(`AR_REPLACE`) advances one *byte* per empty match, so it inserts `repl`
inside multi-byte sequences and produces invalid UTF-8. -/

def replaceRust (s f t : Str) : Str :=
  if hf : f = [] then t ++ s.flatMap (fun c => c :: t) else t.intercalate (splitAux f hf s [])

/-- C on bytes for an empty search string: a match at every byte offset
`0..=strlen`. -/
def cReplaceEmptyBytes (b t : List UInt8) : List UInt8 := t ++ b.flatMap (fun x => x :: t)

def utf8 (s : Str) : List UInt8 := s.flatMap String.utf8EncodeChar

/-- Rust's empty-pattern output is always valid UTF-8 (it is a `String`);
C's is not: `replace('é', '', '-')` gives bytes `2d c3 2d a9 2d`. Live:
C `-h-\xc3-\xa9-l-l-o-`, Rust `-h-é-l-l-o-`. A C bug; Rust is correct. -/
theorem cReplace_empty_breaks_utf8 :
    cReplaceEmptyBytes (utf8 ['é']) (utf8 ['-']) = [0x2d, 0xc3, 0x2d, 0xa9, 0x2d] ∧
    ByteArray.validateUTF8 ⟨#[0x2d, 0xc3, 0x2d, 0xa9, 0x2d]⟩ = false ∧
    utf8 (replaceRust ['é'] [] ['-']) = [0x2d, 0xc3, 0xa9, 0x2d] := by decide

theorem replace_examples :
    replaceRust "aaa".toList "aa".toList ['b'] = "ba".toList ∧
    replaceRust [] [] ['x'] = ['x'] ∧
    replaceRust "abc".toList ['b'] [] = "ac".toList := by decide +kernel

/-! ### `toLower` / `toUpper` (`string.rs`)

`caseMap` abstracts Unicode case mapping. Rust uses std's *full* mapping
(`char::to_lowercase` may yield several code points, and `str::to_lowercase`
applies the final-sigma rule); C uses utf8proc's *simple* one-to-one mapping.
C errors only on invalid byte sequences, which a Rust `String` cannot contain.

Rust maps every string (`caseRust`; an earlier version errored whenever U+FFFD occurred). -/

/-- Main (`string.rs`): `s.to_lowercase()` / `s.to_uppercase()`. -/
def caseRust (full : Char → Str) (s : Str) : SOut := .ok (s.flatMap full)

def caseC (simple : Char → Char) (s : Str) : SOut := .ok (s.map simple)

/-- **Bug D fixed.** Main never errors: every string, including one holding
U+FFFD, is case-mapped (`toLower('A�')` → `'a�'`, as C). -/
theorem toLower_total (full : Char → Str) (s : Str) :
    caseRust full s = .ok (s.flatMap full) := rfl

theorem toLower_accepts_replacement (full : Char → Str) :
    caseRust full ['A', '�'] = .ok (full 'A' ++ full '�') := by
  simp [caseRust]

/-- Main agrees with C on every string whenever the full mapping is the simple
one (no multi-code-point or context-dependent mappings); in particular on
strings holding U+FFFD, where pre-#2953 Rust errored. -/
theorem case_agrees_C_if_simple (simple : Char → Char) (s : Str) :
    caseRust (fun c => [simple c]) s = caseC simple s := by
  simp only [caseRust, caseC]
  congr 1
  induction s with
  | nil => rfl
  | cons c t ih => simp [List.flatMap_cons, ih]

/-- If the full mapping were one-to-one, `size` would be preserved as in C;
it is not (`'ß' ↦ "SS"`), so `size(toUpper('ß'))` is 2 in Rust and 1 in C.
openCypher defers to the host (Neo4j/Java: full mapping, `"SS"`), so this is
a recorded divergence, not a bug. -/
theorem case_size_preserved_if_simple (full : Char → Str) (h : ∀ c, (full c).length = 1)
    (s : Str) : caseRust full s = .ok (s.flatMap full) ∧
    (s.flatMap full).length = s.length := by
  refine ⟨rfl, ?_⟩
  induction s with
  | nil => rfl
  | cons c t ih => simp only [List.flatMap_cons, List.length_append, h, List.length_cons]; rw [ih]; omega

/-! ### `=~` (`internal.rs`, `eval.rs`)

`Regex::is_match` is *unanchored*: it asks for a match anywhere. openCypher
(and Neo4j, following Java's `Pattern.matches`) requires the pattern to match
the whole string. Abstracting the regex engine as a span predicate
`m i j` ("the pattern matches `s[i..j]`"): -/

def isMatchRust (n : Nat) (m : Nat → Nat → Bool) : Bool :=
  (List.range (n + 1)).any fun i => (List.range (n + 1)).any fun j => i ≤ j && m i j
def isMatchSpec (n : Nat) (m : Nat → Nat → Bool) : Bool := m 0 n

/-- Rust is implied by the spec (never a false negative)… -/
theorem regex_spec_implies_rust (n : Nat) (m : Nat → Nat → Bool) (h : isMatchSpec n m = true) :
    isMatchRust n m = true := by
  unfold isMatchRust isMatchSpec at *
  simp only [List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨0, by omega, n, by omega, by omega, h⟩

/-- …but not conversely. Literal pattern `b` on `abc` (spans matching are
exactly `[1,2)`): **Bug C**, repro `bug_regex_match_unanchored`
(`'abc' =~ 'b'` → Rust `true`, spec `false`; C rejects `=~`). -/
theorem regexMatches_unanchored :
    let m : Nat → Nat → Bool := fun i j => i == 1 && j == 2
    isMatchRust 3 m = true ∧ isMatchSpec 3 m = false := by decide

/-! ### `string.replaceRegEx` replacement (`string.rs`, `eval.rs`)

Both call sites pass `NoExpand(repl)` to
`Regex::replace_all`, so the replacement is inserted verbatim (`$n`, `$name`, `${name}` are
not interpreted), like C. `NoExpand::replace_append` appends the string unchanged. Model:
each matched span `[i, j)` of the text is replaced by the replacement as it is. -/

/-- Replace the non-overlapping, ascending spans (`(start, stop)` offsets into the text)
by `r`, copying `r` literally (`NoExpand`). -/
def spliceLit (r : Str) : Nat → Str → List (Nat × Nat) → Str
  | _, t, [] => t
  | off, t, (i, j) :: rest =>
    t.take (i - off) ++ r ++ spliceLit r j (t.drop (j - off)) rest

/-- The replacement string reaches the output byte for byte: with no match the text is
unchanged and with one match at `[i, j)` the output is prefix ++ `r` ++ suffix, for every
`r` (in particular one containing `$`). -/
theorem spliceLit_single (r t : Str) (i j : Nat) :
    spliceLit r 0 t [(i, j)] = t.take i ++ r ++ t.drop j := by
  simp [spliceLit]

/-! The two replacers of `Regex::replace_all`. A plain `&str` replacement is *expanded*: `$name`
(the longest run of letters, digits and `_`) is the capture group of that name or number, `$$`
is `$`, and an unknown group is empty; `NoExpand` is appended unchanged. Passing a `&str` instead is the expanding behaviour. The model covers the `$name` and `$$` forms (not the
`${name}` form) with the capture lookup `cap` abstract. -/

inductive Repl where
  | expand (r : Str)
  | noExpand (r : Str)

def isNameChar (c : Char) : Bool := c.isAlphanum || c == '_'

/-- `$name` / `$$` expansion of a plain `&str` replacement. -/
def expandDollar (cap : Str → Str) : Nat → Str → Str
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, '$' :: '$' :: rest => '$' :: expandDollar cap n rest
  | n + 1, '$' :: rest =>
    let nm := rest.takeWhile isNameChar
    if nm.isEmpty then '$' :: expandDollar cap n rest
    else cap nm ++ expandDollar cap n (rest.drop nm.length)
  | n + 1, c :: rest => c :: expandDollar cap n rest

/-- What `Replacer::replace_append` adds for one match. -/
def appendRepl (cap : Str → Str) : Repl → Str
  | .noExpand r => r
  | .expand r => expandDollar cap r.length r

/-- `Regex::replace_all` over the ascending spans, with either replacer. -/
def spliceRepl (rep : Repl) (cap : Str → Str) : Nat → Str → List (Nat × Nat) → Str
  | _, t, [] => t
  | off, t, (i, j) :: rest =>
    t.take (i - off) ++ appendRepl cap rep ++ spliceRepl rep cap j (t.drop (j - off)) rest

/-- The call sites of `string.replaceRegEx` (`string.rs`, `eval.rs`) pass
`NoExpand(repl)`: that is the `callSite` replacer. -/
def callSite (r : Str) : Repl := .noExpand r

/-- `NoExpand` is the literal splice, whatever the capture groups are. -/
theorem spliceRepl_callSite (r : Str) (cap : Str → Str) (off : Nat) (t : Str)
    (sp : List (Nat × Nat)) : spliceRepl (callSite r) cap off t sp = spliceLit r off t sp := by
  induction sp generalizing off t with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨i, j⟩ := p
    simp only [callSite] at ih ⊢
    simp only [spliceRepl, spliceLit, appendRepl, ih]

/-- The default (`&str`) replacer is what bug E was: `'$0x'` names the group `0x`, which does not
exist, so `replaceRegEx('abc','b','$0x')` was `'ac'`. It differs from the call-site result
`'a$0xc'`, so the model does tell the two replacers apart. -/
theorem expand_differs_from_callSite :
    spliceRepl (.expand "$0x".toList) (fun _ => []) 0 "abc".toList [(1, 2)] = "ac".toList ∧
    spliceRepl (callSite "$0x".toList) (fun _ => []) 0 "abc".toList [(1, 2)] = "a$0xc".toList := by
  decide +kernel

/-- **Bug E, fixed (#3604).** `replaceRegEx('abc','b','$0x')` is `'a$0xc'` and
`replaceRegEx('abc','b','[$0]')` is `'a[$0]c'`, as in C (`$` is copied literally). The splice
itself copies the replacement by definition; what the fix changed is *which replacer* the call
sites pass, and that is `callSite = NoExpand` (`spliceRepl_callSite`; the other replacer
reproduces the bug, `expand_differs_from_callSite`). The call sites are pinned by the Rust
tests, not by this file. -/
theorem replaceRegex_literal :
    spliceLit "$0x".toList 0 "abc".toList [(1, 2)] = "a$0xc".toList ∧
    spliceLit "[$0]".toList 0 "abc".toList [(1, 2)] = "a[$0]c".toList := by decide +kernel

/-! ### `eval_compiled_regex` (`eval.rs`) and `check_string_or_null` (`eval.rs`)

A regex function whose constant pattern was compiled at bind time. The text (and, for
`replaceRegEx`, the replacement) are `eval_node`d and checked to be a String or Null (the
check is `value_of_type(String or Null)`, so the error names `Display for Type` of
`get_type`, `Entity.tyDisp`); a Null text then gives each kind's null result. The regex
engine (`Regex::is_match`, `regex_captures_list`, `replace_all`) is the `RegexOps` record. -/

/-- `check_string_or_null` (`eval.rs`). -/
def checkStringOrNull : Entity.EV → Except String Unit
  | .null => .ok ()
  | .str _ => .ok ()
  | v => .error s!"Type mismatch: expected String or Null but was {Entity.tyDisp v}"

theorem checkStringOrNull_ok_iff (v : Entity.EV) :
    checkStringOrNull v = .ok () ↔ (v = .null ∨ ∃ s, v = .str s) := by
  cases v <;> simp [checkStringOrNull]

theorem checkStringOrNull_err (v : Entity.EV) (h1 : v ≠ .null) (h2 : ∀ s, v ≠ .str s) :
    checkStringOrNull v =
      .error s!"Type mismatch: expected String or Null but was {Entity.tyDisp v}" := by
  cases v <;> simp_all [checkStringOrNull]

structure RegexOps where
  isMatch : String → Bool                  -- `rf.regex.is_match(s)`
  captures : String → Entity.EV            -- `regex_captures_list(&rf.regex, s)`
  replaceEmpty : String → String           -- `replace_all(text, "")`
  replaceNoExpand : String → String → String  -- `replace_all(text, NoExpand(repl))`

inductive RKind where | matches | matchList | replace
  deriving DecidableEq, Repr

/-- `eval_compiled_regex`, arm by arm. `text` / `repl` are the results of
`eval_node(child(0))` / `eval_node(child(1))` (`repl = none` when there is one child).
`Some(_) => unreachable!()` is the sentinel error `"unreachable"`
(`evalCompiledRegex_not_unreachable`). -/
def evalCompiledRegex (R : RegexOps) (kind : RKind) (text : Except String Entity.EV)
    (repl : Option (Except String Entity.EV)) : Except String Entity.EV :=
  match text with
  | .error e => .error e
  | .ok t =>
    match checkStringOrNull t with
    | .error e => .error e
    | .ok () =>
      match kind with
      | .matches =>
        match t with
        | .str s => .ok (.bool (R.isMatch s))
        | _ => .ok .null
      | .matchList =>
        match t with
        | .str s => .ok (R.captures s)
        | _ => .ok (.list [])
      | .replace =>
        let r : Except String (Option Entity.EV) :=
          match repl with
          | none => .ok none
          | some (.error e) => .error e
          | some (.ok r) =>
            match checkStringOrNull r with
            | .error e => .error e
            | .ok () => .ok (some r)
        match r with
        | .error e => .error e
        | .ok r =>
          match t with
          | .str s =>
            match r with
            | none => .ok (.str (R.replaceEmpty s))
            | some .null => .ok .null
            | some (.str rp) => .ok (.str (R.replaceNoExpand s rp))
            | some _ => .error "unreachable"
          | _ => .ok .null

section regexArm
variable (R : RegexOps)

theorem regex_matches_str (s : String) :
    evalCompiledRegex R .matches (.ok (.str s)) none = .ok (.bool (R.isMatch s)) := rfl

theorem regex_matches_null :
    evalCompiledRegex R .matches (.ok .null) none = .ok .null := rfl

theorem regex_matchList_str (s : String) :
    evalCompiledRegex R .matchList (.ok (.str s)) none = .ok (R.captures s) := rfl

/-- A null text gives the empty list (not null). -/
theorem regex_matchList_null :
    evalCompiledRegex R .matchList (.ok .null) none = .ok (.list []) := rfl

/-- A text that is neither String nor Null is a type error before any kind-specific work. -/
theorem regex_text_mismatch (k : RKind) (t : Entity.EV) (r : Option (Except String Entity.EV))
    (h1 : t ≠ .null) (h2 : ∀ s, t ≠ .str s) :
    evalCompiledRegex R k (.ok t) r =
      .error s!"Type mismatch: expected String or Null but was {Entity.tyDisp t}" := by
  have := checkStringOrNull_err t h1 h2
  cases k <;> simp [evalCompiledRegex, this]

/-- Evaluation errors of the text propagate. -/
theorem regex_text_error (k : RKind) (e : String) (r : Option (Except String Entity.EV)) :
    evalCompiledRegex R k (.error e) r = .error e := rfl

/-- `replaceRegEx(text, pattern)` with no replacement child: `replace_all(text, "")`. -/
theorem regex_replace_none (s : String) :
    evalCompiledRegex R .replace (.ok (.str s)) none = .ok (.str (R.replaceEmpty s)) := rfl

/-- A string replacement is spliced with `NoExpand` (#3604). -/
theorem regex_replace_str (s rp : String) :
    evalCompiledRegex R .replace (.ok (.str s)) (some (.ok (.str rp))) =
      .ok (.str (R.replaceNoExpand s rp)) := rfl

/-- A null replacement, or a null text, gives null. -/
theorem regex_replace_null (s : String) :
    evalCompiledRegex R .replace (.ok (.str s)) (some (.ok .null)) = .ok .null ∧
    evalCompiledRegex R .replace (.ok .null) (some (.ok (.str s))) = .ok .null := ⟨rfl, rfl⟩

/-- The replacement is checked even when the text is null (the check precedes the null
return): `replaceRegEx(null, p, 5)` is a type error. -/
theorem regex_replace_null_text_bad_repl :
    evalCompiledRegex R .replace (.ok .null) (some (.ok (.int 5))) =
      .error "Type mismatch: expected String or Null but was Integer" := rfl

theorem tm_ne_unreachable (b : String) :
    ("Type mismatch: expected String or Null but was " ++ b) ≠ "unreachable" := by
  intro h
  have := congrArg String.length h
  simp [String.length_append] at this
  have h1 : "Type mismatch: expected String or Null but was ".length = 47 := by decide
  have h2 : "unreachable".length = 11 := by decide
  omega

/-- `unreachable!()` is never hit: with evaluated arguments the result is never the sentinel. -/
theorem evalCompiledRegex_not_unreachable (k : RKind) (t : Entity.EV)
    (r : Option (Except String Entity.EV)) (hr : ∀ e, r ≠ some (.error e)) :
    evalCompiledRegex R k (.ok t) r ≠ .error "unreachable" := by
  cases t <;> cases k <;> cases r with
  | none => simp [evalCompiledRegex, checkStringOrNull] <;> exact tm_ne_unreachable _
  | some r =>
    cases r with
    | error e => exact absurd rfl (hr e)
    | ok r => cases r <;> simp [evalCompiledRegex, checkStringOrNull] <;> exact tm_ne_unreachable _

/-- **Bug C, still open.** `=~` calls `Regex::is_match` (unanchored): the compiled arm answers
with `isMatchRust`, true for every text that has *some* matching span, where the spec asks for a
match of the whole text (`regexMatches_unanchored`: `'abc' =~ 'b'`). -/
theorem regex_matches_isMatchRust (m : String → Nat → Nat → Bool) (s : String) :
    evalCompiledRegex
      { isMatch := fun s => isMatchRust s.length (m s), captures := fun _ => .null,
        replaceEmpty := id, replaceNoExpand := fun s _ => s }
      .matches (.ok (.str s)) none = .ok (.bool (isMatchRust s.length (m s))) := rfl

end regexArm

/-! ### `string.join` length pre-computation (`string.rs`) -/

/-- `compute_join_length`, `i32` checked arithmetic as `Option`. -/
def joinLen (ls : List Nat) (dl : Nat) : Option Nat :=
  if ls = [] then some 0
  else
    let n := ls.length
    let total := dl * (n - 1) + ls.sum
    if (dl : Int) > I32MAX ∨ (n : Int) > I32MAX ∨ (total + 1 : Int) > I32MAX then none else some total

def joinStr (d : Str) : List Str → Str
  | [] => []
  | [x] => x
  | x :: y :: t => x ++ d ++ joinStr d (y :: t)

theorem joinStr_eq_intercalate (d : Str) (l : List Str) : joinStr d l = d.intercalate l := by
  induction l with
  | nil => rfl
  | cons x t ih =>
    cases t with
    | nil => simp [joinStr]
    | cons y t => simp [joinStr, ih, List.intercalate_cons_cons]

theorem joinStr_length (d : Str) (l : List Str) :
    (joinStr d l).length = (if l = [] then 0 else d.length * (l.length - 1)) + (l.map List.length).sum := by
  induction l with
  | nil => rfl
  | cons x t ih =>
    cases t with
    | nil => simp [joinStr]
    | cons y t =>
      simp only [joinStr, List.length_append, ih]
      simp [Nat.mul_succ]; omega

/-- The `debug_assert_eq!(result.len(), capacity)` at `string.rs` holds:
whenever `compute_join_length` succeeds its value is the joined length (here
in code points of a byte-alphabet model: the lengths are byte lengths in Rust
and the identity is the same). -/
theorem joinLen_correct (d : Str) (l : List Str) (c : Nat)
    (h : joinLen (l.map List.length) d.length = some c) : (joinStr d l).length = c := by
  rw [joinStr_length]
  unfold joinLen at h
  by_cases hl : l = []
  · subst hl; simp at h; simp [h]
  · have : l.map List.length ≠ [] := by simp [hl]
    simp only [this, ite_false, List.length_map] at h
    split at h
    · cases h
    · cases h; simp [hl]

/-! ### String `+` (the String arms of `add_slow`, `value.rs`) -/

theorem str_concat_size (a b : Str) : size (a ++ b) = size a + size b := List.length_append

/-! ### `STARTS WITH` / `ENDS WITH` / `CONTAINS` (`internal.rs`) -/

inductive SV where | str (s : Str) | int (i : Int) | null
  deriving DecidableEq, Repr

/-- `internal_starts_with`: any non-(String,String) pair is `Null`. -/
def startsWith : SV → SV → K
  | .str s, .str p => if p.isPrefixOf s then .t else .f
  | _, _ => .n

/-- A code-point prefix is a byte prefix (C compares bytes). The converse
needs UTF-8's prefix-freeness and is not proven here. -/
theorem prefix_chars_to_bytes (p s : Str) (h : p <+: s) : utf8 p <+: utf8 s := by
  obtain ⟨t, rfl⟩ := h
  exact ⟨utf8 t, by simp [utf8, List.flatMap_append]⟩

/-- openCypher: a non-string operand gives null. Rust follows the spec; C
raises "Type mismatch" (`RETURN 'abc' STARTS WITH 1`). -/
theorem startsWith_nonString : startsWith (.str ['a']) (.int 1) = .n ∧
    startsWith .null (.str []) = .n := ⟨rfl, rfl⟩

theorem startsWith_empty (s : Str) : startsWith (.str s) (.str []) = .t := by
  simp [startsWith]

/-! ### `list.sort` (`list.rs`, comparator)

`sort_by(Value::sort_cmp)` (an earlier `sort_by(|a, b| a.compare_value(b).0)` used `compare_floats`,
which returns `Less` whenever `partial_cmp` is `None`), whose float arm `sort_cmp_f64` (`value.rs`) puts NaN
above every number and equal to itself. -/

def cmpFloatModel (aNaN bNaN : Bool) (o : Ordering) : Ordering :=
  if aNaN || bNaN then .lt else o

/-- `sort_cmp_f64` (value.rs): `partial_cmp` (`o`, meaningful when neither is NaN),
else `a.is_nan().cmp(&b.is_nan())`. -/
def sortCmpF64Model (aNaN bNaN : Bool) (o : Ordering) : Ordering :=
  if aNaN || bNaN then compare aNaN bNaN else o

/-- Main's float comparator is antisymmetric (swapping the operands,
whose `partial_cmp` swaps too, reverses the answer) and NaN is equal to NaN and above
every number. The whole of `sort_cmp` is a total preorder: proofs/value_order
`Main.scMain_total`. -/
theorem listSort_float_cmp_antisymmetric (aNaN bNaN : Bool) (o : Ordering) :
    sortCmpF64Model bNaN aNaN o.swap = (sortCmpF64Model aNaN bNaN o).swap ∧
    sortCmpF64Model true true o = .eq ∧ sortCmpF64Model true false o = .gt ∧
    sortCmpF64Model false true o = .lt := by
  cases aNaN <;> cases bNaN <;> simp [sortCmpF64Model] <;> decide

end FalkorStrList
