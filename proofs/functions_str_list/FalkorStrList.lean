import FalkorStrList.Misc
import FalkorStrList.Entity
/-
# String and list functions: totality and spec agreement

A Lean 4 model of FalkorDB-rs's Cypher string and list functions
(`graph/src/runtime/functions/{string,list,internal}.rs`, eager `range` in
`math.rs`, and the list/regex/quantifier arms of `graph/src/runtime/eval.rs`),
checked against the openCypher spec and the C implementation (`master`
branch, `src/arithmetic/{string,list}_funcs/*.c`, `src/util/strutil.c`).
Build: `cd proofs/functions_str_list && lake build` (Lean 4.34, core only).
No `sorry`/`admit`/`axiom`; headline theorems use only `propext`,
`Quot.sound`, `Classical.choice`. Repros: `graph/tests/lean_functions_str_list.rs`
(`cargo test -p graph --test lean_functions_str_list -- --nocapture`).
Coverage per Rust fn: `COVERAGE.tsv`.

## What is modelled

| here | there |
| --- | --- |
| `inI64`, `wrap64`, `asUsize`, `trunc32` | i64 range (debug panics outside), release wrap, `as usize`, C `(int32_t)` |
| `getElementIdx`, `getElement` | `ExprIR::GetElement` list arm, `eval.rs:573-584` |
| `sliceStart`, `sliceEnd`, `getElements` | `get_elements`, `eval.rs:1722-1750` |
| `cGetElement`, `cGetElements` | C `AR_SUBSCRIPT`, `AR_SLICE` (int32 truncation) |
| `removeSpan`, `listRemove`, `listRemoveSpec` | `remove_span` `list.rs:41-54`, `list.remove` `list.rs:165-189` (`listRemoveDebug`/`Release`: pre-#2952, historical) |
| `insertNorm` | `list.insert` / `list.insertListElements`, `list.rs:253-267,292-310` |
| `head`, `last`, `tail`, `size`, reverse | `list.rs:57-154` |
| `rangeLen`, `rangeElems`, `range` | eager `range`, `math.rs:251-310` |
| `rangeIterStepField` | `RangeIter` construction/`next`, `eval.rs:108-123,1211-1216` |
| `evalQuantifier`, `quant`, `kand`/`kor`/`knot` | `eval_quantifier`, `eval.rs:1616-1660`; loop `eval.rs:979-1016` |
| `containsLoop`, `contains` | `Contains for ThinVec<Value>`, `value.rs:1762-1785` (via `list_contains`, `eval.rs:1708`) |
| `comprehensionRust` / `comprehensionSpec` | `ListComprehension` via `eval_iter_expr` fallback, `eval.rs:1018-1033,1234-1243` |
| `reduceList` | `ExprIR::Reduce`, `eval.rs:1035-1061` |
| `Str`, `blen` | `String` as code points (`.chars()`), `s.len()` / `strlen` bytes |
| `substring2`, `substring3`, `cSubstring3` | `substring`, `string.rs:51-101`; C `AR_SUBSTRING` |
| `left`, `right`, `cLeft`, `cRight` | `string.rs:207-299`; C `AR_LEFT`, `AR_RIGHT` |
| `cReverseBytes` | C `AR_REVERSE` (writes each code point's bytes leftwards) |
| `ltrim`, `rtrim`, `trimRust`, `trimC` | `string.rs:233-274`; C `AR_LTRIM`/`AR_RTRIM`/`AR_TRIM` |
| `splitAux` | std `str::split` contract, used by `split`, `string.rs:125-129` |
| `findFirst`, `cSplit`, `splitC` | C `AR_SPLIT` |
| `splitRust` | `split`, `string.rs:103-137` |
| `replaceRust`, `cReplaceEmptyBytes` | `replace`, `string.rs:183-205` (std `str::replace`); C `AR_REPLACE` |
| `caseRust`, `caseC` | `tolower`/`toupper`, `string.rs:139-181`; C `str_tolower` (utf8proc simple map) |
| `isMatchRust`, `isMatchSpec` | `regex_matches` `internal.rs:100-120`, `RegexFnKind::Matches` `eval.rs:1274` |
| `expandRust` | `Regex::replace_all(_, &str)` `$`-expansion, `string.rs:462,469`, `eval.rs:1294-1302` |
| `joinLen`, `joinStr` | `compute_join_length` / `join_with_preallocate`, `string.rs:324-386` |
| `startsWith` | `internal_starts_with`, `internal.rs:32-48` (ends/contains analogous) |
| `cmpFloatModel` | `compare_floats` `value.rs:1801` as used by `list.sort` `list.rs:213` |

## Proven (86 theorems; main ones)

* Indexing `l[i]` is total for every i64 (incl. `i64::MIN`), `len+i` cannot
  overflow, and equals the spec (negative from the end, else null)
  (`getElement_noOverflow`, `getElement_inBounds`, `getElement_spec`).
* Slicing `l[a..b]` never panics: after the early return, `0 ≤ s ≤ e ≤ len`
  (`getElements_bounds`); it equals the clamped spec (`getElements_spec`).
* `list.remove`: since #2952 (`remove_span`, saturating end) it equals the spec
  for all i64 inputs in every build and never panics (`listRemove_eq_spec`,
  `removeSpan_wf`, `listRemove_huge_count`). Historical: before #2952 release
  matched the spec by accident (`listRemoveRelease_eq_spec`) and debug
  overflowed (`pre2952_listRemove_end_overflows`).
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
* `split`: std's scanner = C's find-first algorithm for every non-empty
  delimiter (`splitAux_eq_cSplit`), and all empty-string branches agree (`split_agrees_C`).
* `string.join`: when `compute_join_length` succeeds it equals the joined
  length, so the `debug_assert_eq!` holds (`joinLen_correct`).
* `size` counts code points, `size ≤ bytes`, additive under `+`.

## CONFIRMED bugs (Rust repro fails; outputs recorded live vs C on ports 18270/18271)

A. **`list.sort` panics on NaN, crashes server** — `list.rs:213`
   (`sort_by` with `compare_value`). Same root cause as #2891 but a separate
   call site not fixed by an ORDER BY fix. Query:
   `WITH [x IN range(1,40) | CASE WHEN x % 3 = 0 THEN 0.0/0.0 ELSE toFloat(x % 7) END] AS l RETURN size(list.sort(l))`
   → release redis-server dies ("user-provided comparison function does not correctly
   implement a total order"); C returns 40. Test `bug_list_sort_nan_panics`.
   Fix: sort with a total order (`total_cmp`-style NaN placement), as for #2891.
B. **`list.remove` count overflow — FIXED by #2952 (07e712944).** Was: the
   inline `(normalized + count)`: `list.remove([1,2,3], 1, 9223372036854775807)`
   panicked in debug ("attempt to add with overflow"); release returned `[1]`
   by accident (proved). C `[1]`. Test `bug_list_remove_count_overflow`.
   The fix (`remove_span`, `saturating_add`) is modelled and proved correct
   (`listRemove_eq_spec`); the counterexample is kept as
   `pre2952_listRemove_end_overflows`.
C. **`=~` is unanchored** — `internal.rs:112`, `eval.rs:1274`
   (`Regex::is_match`): `'abc' =~ 'b'` → `true`, `'xabcx' =~ 'abc'` → `true`.
   openCypher/Neo4j require a whole-string match (→ `false`). C rejects `=~`
   ("FalkorDB does not currently support =~"). Test `bug_regex_match_unanchored`.
   Fix: compile `^(?:pat)$` (binder `rewrite_compiled_regex` and runtime path).
D. **`toLower`/`toUpper` reject valid strings containing U+FFFD** —
   `string.rs:148,170`: `toLower('A�')` → error "Invalid UTF8 string";
   C → `'a�'`. A Rust `String` is always valid UTF-8, so the check can only
   misfire. Test `bug_case_map_rejects_replacement_char`. Fix: delete the check.
E. **`string.replaceRegEx` expands `$` references** — `string.rs:469`,
   `eval.rs:1302`: `replaceRegEx('abc','b','$0x')` → `'ac'`, `'[$0]'` → `'a[b]c'`;
   C copies literally: `'a$0xc'`, `'a[$0]c'`. Test `bug_replace_regex_dollar_expansion`.
   Fix: `regex::NoExpand(repl)`.
F. **List comprehension over null / scalar** — `eval.rs:1018-1033` via
   `eval_iter_expr`'s UNWIND fallback (`eval.rs:1234-1243`): `[x IN null | x]` → `[]`
   (C `null`), `[x IN 5 | x]` → `[5]` (C type error). Reduce and quantifiers
   already do it right. Test `bug_list_comprehension_null_and_scalar_source`.

## Divergences recorded, not bugs (or C is wrong)

* Case mapping: Rust uses full Unicode mapping, C simple: `toUpper('ß')` Rust
  `'SS'` (size 2) vs C `'ẞ'`; `toLower('İ')` `'i̇'` vs `'i'`; `toLower('ΑΣ')`
  final sigma `'ας'` vs `'ασ'`; `toUpper('ﬁ')` `'FI'` vs `'ﬁ'`. Neo4j matches Rust.
* `string.matchRegEx('ab','(a)|(b)')`: Rust `null` for non-participating
  groups, C `''` (deliberate per comment at `string.rs:487`).
* `'a' + (0.0/0.0)`: Rust `'aNaN'`, C `'anan'`.
* `STARTS WITH`/`ENDS WITH`/`CONTAINS` with a non-string: Rust `null` (openCypher), C error.
* C bugs: index/slice bounds truncated to int32 (`[1,2,3][4294967296]` → C `1`,
  `[1,2,3][MIN..MAX]` → C `[1, 2]`) (`cGetElement_truncates`, `cGetElements_truncates`);
  `replace(s,'',t)` inserts inside multibyte chars → invalid UTF-8
  (`cReplace_empty_breaks_utf8`); `range(0, 5000000000)` crashed the C server (Rust: "Range too large").
* Regex dialect: Rust `regex` crate vs C oniguruma Java syntax (no
  backreferences/lookaround in Rust) — not modelled.

## Unconfirmed / variants of known issues

* `RangeIter` with `step = i64::MIN`: `step.unsigned_abs() as i64` is `i64::MIN`
  and `-self.step` (`eval.rs:120`) overflows — debug panic "attempt to negate
  with overflow" for `UNWIND range(10, 0, -9223372036854775808)`; release is
  correct (`[10]`). Variant of #2894 (`rangeIter_negate_overflows`).

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
`deleteNode_preserves_all`; **CONFIRMED BUG** `deleteNode_labels_iff` /
`labels_lost_after_delete`: the DELETE snapshot copies committed labels
(ops/delete.rs:282, :461 `g.get_node_label_ids(id)`), so labels staged by SET/REMOVE earlier
in the same query are lost/resurrected. Repro: `lean_functions_str_list::
bug_deleted_node_labels_ignore_staged_set` (`MATCH (n:A) SET n:C DELETE n RETURN labels(n), n:C`
→ Rust `['A']|false`, expected `['A','C']|true`; C returns `[]`) and
`..._staged_remove` (`REMOVE n:A … DELETE n RETURN labels(n)` → Rust `['A']`, expected `[]`).
Others: `hasLabels_all`, `parseDegreeArgs_forms`, `strReplace_self`, `toStringVec_err`,
`capturesList_shape`, `listSort_perm`, `listDedup_spec`, `internalIsNull_spec`.
-/

namespace FalkorStrList

/-! ## Machine integers

Rust `i64` values are modelled as `Int` together with the range predicate
`inI64`. A debug build panics exactly when an arithmetic result leaves that
range; a release build (the `[profile.release]` in `Cargo.toml` sets no
`overflow-checks`) wraps, which is `wrap64`. `asUsize` is Rust's `x as usize`
on a 64-bit target. -/

def I64MIN : Int := -9223372036854775808
def I64MAX : Int := 9223372036854775807
def I32MAX : Int := 2147483647
def inI64 (x : Int) : Prop := I64MIN ≤ x ∧ x ≤ I64MAX
instance (x : Int) : Decidable (inI64 x) := by unfold inI64; infer_instance

def wrap64 (x : Int) : Int := (x + 9223372036854775808) % 18446744073709551616 - 9223372036854775808
def asUsize (x : Int) : Nat := (x % 18446744073709551616).toNat
/-- C's `(int32_t)x` truncation, as used by `AR_SUBSCRIPT` / `AR_SLICE`. -/
def trunc32 (x : Int) : Int := (x + 2147483648) % 4294967296 - 2147483648

theorem asUsize_wrap64 (x : Int) (h0 : 0 ≤ x) (h1 : x < 18446744073709551616) :
    asUsize (wrap64 x) = x.toNat := by
  unfold asUsize wrap64; omega

/-! ## `[l][i]` — `ExprIR::GetElement` (`eval.rs:573-584`) -/

/-- `eval.rs:578-584`: `normalized_index = if i < 0 { len + i } else { i }`,
then `values[normalized_index as usize]` under the guard `0 ≤ n < len`. -/
def getElementIdx (len : Nat) (i : Int) : Int := if i < 0 then (len : Int) + i else i

def getElement (l : List α) (i : Int) : Option α :=
  let n := getElementIdx l.length i
  if 0 ≤ n ∧ n < l.length then l[n.toNat]? else none

/-- `len + i` never overflows: `len ≤ i64::MAX` and `i < 0`. -/
theorem getElement_noOverflow (len : Nat) (i : Int) (hl : (len : Int) ≤ I64MAX)
    (hi : inI64 i) : inI64 (getElementIdx len i) := by
  unfold getElementIdx inI64 at *; unfold I64MIN I64MAX at *; split <;> omega

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
  · simp only [h, if_true]
    have : ¬ 0 ≤ i := by omega
    simp only [this, if_false]
    by_cases h2 : -(l.length : Int) ≤ i
    · simp only [h2, if_true]; rw [if_pos (by omega)]
    · simp only [h2, if_false]; rw [if_neg (by omega)]
  · simp only [h, if_false]
    have : 0 ≤ i := by omega
    simp only [this, if_true]
    by_cases h2 : i < l.length
    · simp [h2]
    · simp only [h2, and_false, if_false]
      rw [List.getElem?_eq_none]; omega

/-- C truncates the index to `int32_t` before normalising (`list_funcs.c`,
`AR_SUBSCRIPT`). -/
def cGetElement (l : List α) (i : Int) : Option α := getElement l (trunc32 i)

/-- C and Rust disagree at `2^32`: C reads element 0, Rust returns null
(openCypher: null). Live: `RETURN [1,2,3][4294967296]` → C `1`, Rust `null`. -/
theorem cGetElement_truncates :
    cGetElement [1, 2, 3] 4294967296 = some 1 ∧ getElement [1, 2, 3] 4294967296 = none := by
  decide

/-! ## `[l][a..b]` — `get_elements` (`eval.rs:1722-1750`) -/

/-- `eval.rs:1729-1737`, the two normalisations, line by line. -/
def sliceStart (len : Nat) (s : Int) : Int := if s < 0 then max ((len : Int) + s) 0 else s
def sliceEnd (len : Nat) (e : Int) : Int := if e < 0 then max ((len : Int) + e) 0 else min e len

def getElements (l : List α) (s e : Int) : List α :=
  let s' := sliceStart l.length s
  let e' := sliceEnd l.length e
  if s' > e' then [] else (l.drop s'.toNat).take (e' - s').toNat

/-- Totality of `values[start as usize..end as usize]` (`eval.rs:1742`): when
the `start > end` early return is not taken, `0 ≤ start ≤ end ≤ len`, so the
slice never panics — for every pair of `i64` bounds. -/
theorem getElements_bounds (len : Nat) (s e : Int) (h : ¬ sliceStart len s > sliceEnd len e) :
    0 ≤ sliceStart len s ∧ sliceStart len s ≤ sliceEnd len e ∧ sliceEnd len e ≤ len := by
  unfold sliceStart sliceEnd at *
  split <;> split <;> omega

theorem getElements_noOverflow (len : Nat) (x : Int) (hl : (len : Int) ≤ I64MAX)
    (hx : inI64 x) (hneg : x < 0) : inI64 ((len : Int) + x) := by
  unfold inI64 I64MIN I64MAX at *; omega

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
  by_cases hs : s < 0 <;> by_cases he : e < 0 <;> simp only [hs, he, if_true, if_false]
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

/-! ## `list.remove(l, idx, count)` (`list.rs:165-189`, `remove_span` `list.rs:41-54`)

Since #2952 (07e712944) the span is computed by `remove_span`, which clamps the
end with `saturating_add`: `listRemove` / `removeSpan` below model that code.
`listRemoveDebug` / `listRemoveRelease` are kept as the **historical** model of
the pre-#2952 inline arithmetic (`(normalized + count) as usize`), whose debug
build overflowed (`pre2952_listRemove_end_overflows`). -/

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
  else if ¬ inI64 (n + count) then .panic "attempt to add with overflow"
  else
    let e := min (n + count).toNat l.length
    .ok (l.take n.toNat ++ l.drop e)

/-- Historical (pre-#2952) release build: the add wraps, then `as usize`, then `.min(vs.len())`. -/
def listRemoveRelease (l : List α) (idx count : Int) : List α :=
  let len : Int := l.length
  let n := removeNorm l.length idx
  if n < 0 ∨ n ≥ len ∨ count ≤ 0 then l
  else
    let e := min (asUsize (wrap64 (n + count))) l.length
    l.take n.toNat ++ l.drop e

/-- Spec (and C): drop `count` elements starting at the normalised index. -/
def listRemoveSpec (l : List α) (idx count : Int) : List α :=
  let len : Int := l.length
  let n := removeNorm l.length idx
  if n < 0 ∨ n ≥ len ∨ count ≤ 0 then l
  else l.take n.toNat ++ l.drop (n + count).toNat

/-- **Bug B (historical; fixed by #2952, 07e712944).** Before #2952,
`list.remove([1,2,3], 1, i64::MAX)` overflowed the add in a debug build.
Rust repro: `lean_functions_str_list::bug_list_remove_count_overflow` panicked
with "attempt to add with overflow". The current code is `listRemove` below
(`listRemove_huge_count`, `listRemove_eq_spec`). -/
theorem pre2952_listRemove_end_overflows :
    listRemoveDebug [1, 2, 3] 1 I64MAX = .panic "attempt to add with overflow" := by decide

/-- …but the release build is right by accident: a wrapped (negative) sum
becomes `≥ 2^63` under `as usize`, and `.min(len)` clamps it. Proven for every
list shorter than `i64::MAX` and every `i64` pair. -/
theorem removeNorm_range (len : Nat) (idx : Int) (hl : (len : Int) ≤ I64MAX) (hi : inI64 idx) :
    inI64 (removeNorm len idx) := by
  unfold removeNorm inI64 I64MIN I64MAX at *; split <;> omega

theorem listRemoveRelease_eq_spec (l : List α) (idx count : Int)
    (hl : (l.length : Int) ≤ I64MAX) (hi : inI64 idx) (hc : inI64 count) :
    listRemoveRelease l idx count = listRemoveSpec l idx count := by
  have hn := removeNorm_range l.length idx hl hi
  unfold listRemoveRelease listRemoveSpec
  simp only []
  generalize removeNorm l.length idx = n at *
  unfold inI64 I64MIN I64MAX at *
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

/-- `remove_span` (`list.rs:41-54`): `None` when nothing is removed, else the
`start..end` span; `end = (normalized.saturating_add(count) as usize).min(len)`.
`normalized ≥ 0` there, so the saturated sum is non-negative and `as usize` is
the identity (`toNat`). -/
def removeSpan (len : Nat) (idx count : Int) : Option (Nat × Nat) :=
  let n := removeNorm len idx
  if n < 0 ∨ n ≥ (len : Int) ∨ count ≤ 0 then none
  else some (n.toNat, min (satAdd64 n count).toNat len)

/-- `list_remove` (`list.rs:165-189`) on a list argument: the original list when
`remove_span` is `None`, else `vs[..start] ++ vs[end..]`. The same in debug and
release: no arithmetic in it can overflow (`removeSpan_wf`). -/
def listRemove (l : List α) (idx count : Int) : List α :=
  match removeSpan l.length idx count with
  | none => l
  | some (s, e) => l.take s ++ l.drop e

/-- The unit test `remove_span_clamps_huge_count` (`list.rs:364`), as a theorem. -/
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
  · rw [if_pos hg] at h; cases h
  · rw [if_neg hg] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    split <;> (try split) <;> omega

/-- **Bug B fixed.** The debug overflow input now returns C's `[1]`. -/
theorem listRemove_huge_count : listRemove [1, 2, 3] 1 I64MAX = [1] := by decide

/-- `list.remove` equals the spec (and C) for every list shorter than
`i64::MAX` and every `i64` index and count, in debug and release alike. -/
theorem listRemove_eq_spec (l : List α) (idx count : Int)
    (hl : (l.length : Int) ≤ I64MAX) (hi : inI64 idx) (hc : inI64 count) :
    listRemove l idx count = listRemoveSpec l idx count := by
  have hn := removeNorm_range l.length idx hl hi
  unfold listRemove listRemoveSpec removeSpan
  generalize removeNorm l.length idx = n at *
  unfold inI64 I64MIN I64MAX at *
  by_cases hg : n < 0 ∨ n ≥ (l.length : Int) ∨ count ≤ 0
  · simp only [if_pos hg]
  · simp only [if_neg hg]
    have hsat : (satAdd64 n count).toNat = min (n + count).toNat 9223372036854775807 := by
      unfold satAdd64 I64MAX I64MIN
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
    (hl : (l.length : Int) ≤ I64MAX) (hi : inI64 idx) (hc : inI64 count) :
    listRemove l idx count = listRemoveRelease l idx count := by
  rw [listRemove_eq_spec l idx count hl hi hc, listRemoveRelease_eq_spec l idx count hl hi hc]

/-! ## `list.insert` / `list.insertListElements` normalisation
(`list.rs:253-267`, `list.rs:305-310`) -/

def insertNorm (len : Nat) (idx : Int) : Int := if idx < 0 then (len : Int) + idx + 1 else idx

/-- `len + idx + 1` never overflows, and when the range guard passes the
position is a valid split point for `vs[..pos]` / `vs[pos..]`. -/
theorem insertNorm_safe (len : Nat) (idx : Int) (hl : (len : Int) < I64MAX) (hi : inI64 idx) :
    inI64 (insertNorm len idx) ∧
    (¬ (insertNorm len idx < 0 ∨ insertNorm len idx > len) → (insertNorm len idx).toNat ≤ len) := by
  unfold insertNorm inI64 I64MIN I64MAX at *
  constructor
  · split <;> omega
  · intro h; split at * <;> omega

/-- `-1` inserts after the last element (C's `bounds_inclusive`). -/
theorem insertNorm_minus_one (len : Nat) : insertNorm len (-1) = len := by
  unfold insertNorm; simp only [show (-1 : Int) < 0 by decide, if_true]; omega

/-! ## `head` / `last` / `tail` / `reverse` / `size` on lists (`list.rs:57-154`) -/

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

/-! ## Eager `range(start, end, step)` (`math.rs:251-310`) -/

/-- `math.rs:274-287`: `|end - start| / |step| + 1`, computed in `i128`, so no
overflow is possible (it is plain `Int` arithmetic here). -/
def rangeLen (start stop step : Int) : Nat :=
  ((if stop ≥ start then stop - start else start - stop).natAbs / step.natAbs) + 1

/-- `(start..=end).step_by(step)` / `(end..=start).rev().step_by(|step|)`:
std's `step_by` on an inclusive range yields `start + k*step` while it stays
inside the range; it never computes an element outside it, so it cannot
overflow. That is modelled directly as the element list. -/
def rangeElems (start stop step : Int) : List Int :=
  (List.range (rangeLen start stop step)).map (fun (k : Nat) => start + (k : Int) * step)

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
with the lazy `RangeIter`, #2894). -/
theorem range_elems_between (start stop step : Int) (hs : step ≠ 0)
    (hdir : ¬ ((start > stop ∧ step > 0) ∨ (start < stop ∧ step < 0)))
    (x : Int) (hx : x ∈ rangeElems start stop step) :
    min start stop ≤ x ∧ x ≤ max start stop := by
  unfold rangeElems rangeLen at hx
  simp only [List.mem_map, List.mem_range] at hx
  obtain ⟨k, hk, rfl⟩ := hx
  have hk' : k ≤ (if stop ≥ start then stop - start else start - stop).natAbs / step.natAbs := by omega
  have hmul : k * step.natAbs ≤ (if stop ≥ start then stop - start else start - stop).natAbs :=
    (Nat.le_div_iff_mul_le (by omega)).mp hk'
  have hmulI : ((k * step.natAbs : Nat) : Int) ≤
      ((if stop ≥ start then stop - start else start - stop).natAbs : Int) := by exact_mod_cast hmul
  push_cast at hmulI
  by_cases hsg : step > 0
  · have hab : (step.natAbs : Int) = step := by omega
    have hup : stop ≥ start := by omega
    simp only [hup, if_true] at hmulI
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
    · simp only [heq, if_false] at hmulI
      have : ((start - stop).natAbs : Int) = start - stop := by omega
      rw [this] at hmulI
      have h0 : 0 ≤ (k : Int) * -step := Int.mul_nonneg (by omega) (by omega)
      rw [Int.mul_neg] at hmulI h0
      constructor <;> omega

theorem range_noOverflow (start stop step : Int) (h1 : inI64 start) (h2 : inI64 stop)
    (l : List Int) (hr : range start stop step = .ok l) : ∀ x ∈ l, inI64 x := by
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
        unfold inI64 at *; omega

/-- The two C/TCK checks: first and last elements. -/
theorem range_examples :
    range 0 10 3 = .ok [0, 3, 6, 9] ∧ range 10 0 (-3) = .ok [10, 7, 4, 1] ∧
    range 10 0 I64MIN = .ok [10] ∧ range 1 0 1 = .ok [] := ⟨rfl, rfl, rfl, rfl⟩

/-! ## Lazy `RangeIter::next` (`eval.rs:108-123`)

The constructor stores `step: step.unsigned_abs() as i64` (`eval.rs:1215`); for
`step = i64::MIN` that is `i64::MIN` again, and `next` then evaluates
`-self.step`, which overflows before the (known, #2894) `+=`. -/

def rangeIterStepField (step : Int) : Int := wrap64 step.natAbs

theorem rangeIter_negate_overflows :
    rangeIterStepField I64MIN = I64MIN ∧ ¬ inI64 (-(rangeIterStepField I64MIN)) := by decide

/-! ## Quantifiers `all/any/none/single` (`eval_quantifier`, `eval.rs:1616-1660`) -/

/-- The three predicate outcomes the loop at `eval.rs:988-1004` counts. -/
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
  unfold cnt; cases x <;> cases k <;> simp [List.filter_cons] <;> omega

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
  · simp only [h1, if_true]; rw [if_neg (by omega)]
  · simp only [h1, if_false]
    by_cases h2 : cnt xs .n > 0
    · simp only [h2, if_true]; rw [if_neg (by omega)]
    · simp only [h2, if_false]
      by_cases h3 : cnt xs .t = 1 <;> simp [h3] <;> omega

/-! ## `x IN list` — `Contains for ThinVec<Value>` (`value.rs:1762-1785`)

Parametric in the element comparison `cmp : α → α → (Ordering × Dis)`; the
comparison itself is `value_order`'s subject. -/

inductive Dis where | none | null | nan | disjoint
  deriving DecidableEq, Repr

/-- `value.rs:1767-1783`, with `is_null` accumulated left to right. -/
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
      simp only [this, Bool.false_eq_true, if_false, ih]
      cases b <;> cases d <;> simp [hr]

/-- The empty list never contains anything, not even null. -/
theorem contains_nil (cmp : α → Ordering × Dis) : contains cmp [] = .f := rfl

/-! ## Comprehension / reduce / quantifier sources

`[x IN src ...]` goes through `eval_iter_expr` (`eval.rs:1178`), whose fallback
arm (`eval.rs:1234-1243`) is UNWIND's: `Null` → no rows, scalar → one row.
Quantifiers (`eval.rs:1005-1012`) and `reduce` (`eval.rs:1053-1059`) instead
return `Null` for a null source and raise for a scalar, as does C. -/

inductive Src where | null | scalar (i : Int) | list (l : List Int)
  deriving DecidableEq, Repr

inductive COut where | null | list (l : List Int) | err
  deriving DecidableEq, Repr

/-- `ListComprehension` with no WHERE and identity projection. -/
def comprehensionRust : Src → COut
  | .null => .list []
  | .scalar i => .list [i]
  | .list l => .list l

/-- C / openCypher, and Rust's own `reduce`/quantifier arms. -/
def comprehensionSpec : Src → COut
  | .null => .null
  | .scalar _ => .err
  | .list l => .list l

/-- **Bug F.** Live: `[x IN null | x]` → Rust `[]`, C `null`;
`[x IN 5 | x]` → Rust `[5]`, C type error. Repro:
`bug_list_comprehension_null_and_scalar_source`. -/
theorem comprehension_diverges :
    comprehensionRust .null ≠ comprehensionSpec .null ∧
    comprehensionRust (.scalar 5) ≠ comprehensionSpec (.scalar 5) := by decide

theorem comprehension_agrees_on_lists (l : List Int) :
    comprehensionRust (.list l) = comprehensionSpec (.list l) := rfl

/-- `reduce(acc = init, x IN l | body)` (`eval.rs:1035-1061`) is `foldl`. -/
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

/-! ### `substring` (`string.rs:51-101`) -/

/-- `string.rs:64-74`. Note the guard compares the *code point* index against
the *byte* length `s.len()`. -/
def substring2 (s : Str) (start : Int) : SOut :=
  if start < 0 then .err "start must be a non-negative integer"
  else if start ≥ blen s then .ok []
  else .ok (s.drop start.toNat)

/-- `string.rs:77-96`. -/
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
    · rw [if_pos (e.mpr hb), if_pos hb]
    · rw [if_neg (fun h => hb (e.mp h)), if_neg hb]
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
  rw [if_neg (by omega)]
  split
  · rw [List.drop_eq_nil_of_le]; · simp
    have := length_le_blen s; omega
  · rw [if_neg (by omega)]

/-- Shared divergence (Rust = C): a negative length is *not* rejected when the
start is already past the end — `substring('abc', 10, -1)` is `''`. -/
theorem substring3_negative_length_past_end :
    substring3 ['a', 'b', 'c'] 10 (-1) = .ok [] ∧
    substring3 ['a', 'b', 'c'] 0 (-1) = .err "length must be a non-negative integer" := by decide

/-- UTF-8: `substring('héllo', 1, 3)` is `'éll'` (code points, not bytes). -/
theorem substring_multibyte :
    substring3 "héllo".toList 1 3 = .ok "éll".toList ∧ substring2 "héllo".toList 5 = .ok [] := by
  decide

/-! ### `left` / `right` (`string.rs:207-299`) -/

def left (s : Str) (n : Int) : SOut :=
  if n < 0 then .err "length must be a non-negative integer" else .ok (s.take n.toNat)

/-- `string.rs:289-290`: `chars().count().saturating_sub(n as usize)`. -/
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

/-! ### `size` / `reverse` (`list.rs:64-71`, `list.rs:149`) -/

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

/-! ### `trim` / `ltrim` / `rtrim` (`string.rs:233-274`) -/

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
    rw [if_pos he, he']
    by_cases hc : isSp c
    · simp [List.dropWhile, hc]
    · simp [List.dropWhile, hc]
  · have he' : List.dropWhile isSp t.reverse ≠ [] := fun h => he (List.isEmpty_iff.mpr h)
    rw [if_neg he]
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
      · simp [hc, hr, ltrim, List.dropWhile]
    · have hl : ltrim (c :: t) = c :: t := by simp [ltrim, List.dropWhile, hc]
      rw [hl, rtrim_cons]
      simp [hc, ltrim, List.dropWhile]

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

/-! ### `split(s, delim)` (`string.rs:103-137`)

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
    split <;> rename_i h <;> simp [h]
  | cons c t ih =>
    rw [splitAux]
    simp only [findFirst]
    by_cases h : d.isPrefixOf (c :: t) = true
    · simp [h]
    · simp only [h, dite_false, if_false, Bool.false_eq_true]
      rw [ih]
      cases findFirst d t with
      | none => simp
      | some j =>
        simp only [Option.map_some, List.reverse_cons, List.append_assoc, List.singleton_append,
          List.take_succ_cons, List.cons.injEq, and_true]
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

/-- `string.rs:112-131`. -/
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

/-! ### `replace(s, search, repl)` (`string.rs:183-205`)

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

/-! ### `toLower` / `toUpper` (`string.rs:139-181`)

`caseMap` abstracts Unicode case mapping. Rust uses std's *full* mapping
(`char::to_lowercase` may yield several code points, and `str::to_lowercase`
applies the final-sigma rule); C uses utf8proc's *simple* one-to-one mapping.
The model only needs: Rust errors whenever U+FFFD occurs; C errors only on
invalid byte sequences, which a Rust `String` cannot contain. -/

def caseRust (full : Char → Str) (s : Str) : SOut :=
  if '�' ∈ s then .err "Invalid UTF8 string" else .ok (s.flatMap full)

def caseC (simple : Char → Char) (s : Str) : SOut := .ok (s.map simple)

/-- **Bug D.** On the valid string `"A�"` Rust errors and C returns
`"a�"`. Repro `bug_case_map_rejects_replacement_char`. -/
theorem toLower_rejects_valid (full : Char → Str) (simple : Char → Char) :
    caseRust full ['A', '�'] = .err "Invalid UTF8 string" ∧
    caseC simple ['A', '�'] = .ok [simple 'A', simple '�'] := by
  constructor
  · unfold caseRust; simp
  · rfl

/-- If the full mapping were one-to-one, `size` would be preserved as in C;
it is not (`'ß' ↦ "SS"`), so `size(toUpper('ß'))` is 2 in Rust and 1 in C.
openCypher defers to the host (Neo4j/Java: full mapping, `"SS"`), so this is
a recorded divergence, not a bug. -/
theorem case_size_preserved_if_simple (full : Char → Str) (h : ∀ c, (full c).length = 1)
    (s : Str) (hs : '�' ∉ s) : caseRust full s = .ok (s.flatMap full) ∧
    (s.flatMap full).length = s.length := by
  refine ⟨by unfold caseRust; simp [hs], ?_⟩
  induction s with
  | nil => rfl
  | cons c t ih => simp only [List.flatMap_cons, List.length_append, h, List.length_cons]; simp at hs; rw [ih hs.2]; omega

/-! ### `=~` (`internal.rs:100-120`, `eval.rs:1274`)

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

/-! ### `string.replaceRegEx` replacement (`string.rs:469`, `eval.rs:1302`)

`Regex::replace_all(text, &str)` interprets `$n` / `$name` / `${name}` in the
replacement; C copies it verbatim. Model of the expansion on the `$0x` case:
`$0x` names the (absent) group `0x`, which expands to the empty string. -/

def expandRust (groupName : Str → Option Str) : Str → Str
  | '$' :: rest =>
    let name := rest.takeWhile (fun c => c.isAlphanum || c == '_')
    if name = [] then '$' :: expandRust groupName rest
    else (groupName name).getD [] ++ expandRust groupName (rest.drop name.length)
  | c :: rest => c :: expandRust groupName rest
  | [] => []
termination_by s => s.length
decreasing_by all_goals simp_wf <;> omega

/-- **Bug E.** `replaceRegEx('abc','b','$0x')`: Rust `'ac'`, C `'a$0xc'`. -/
theorem replaceRegex_dollar :
    let g : Str → Option Str := fun n => if n = ['0'] then some ['b'] else none
    expandRust g "$0x".toList = [] ∧ expandRust g "[$0]".toList = "[b]".toList := by decide +kernel

/-! ### `string.join` length pre-computation (`string.rs:324-361`) -/

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

/-- The `debug_assert_eq!(result.len(), capacity)` at `string.rs:384` holds:
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
    simp only [this, if_false, List.length_map] at h
    split at h
    · cases h
    · cases h; simp [hl]

/-! ### String `+` (`value.rs:969-975`) -/

theorem str_concat_size (a b : Str) : size (a ++ b) = size a + size b := List.length_append

/-! ### `STARTS WITH` / `ENDS WITH` / `CONTAINS` (`internal.rs:32-84`) -/

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

/-! ### `list.sort` (`list.rs:212-213`)

`sort_by(|a, b| a.compare_value(b).0)`; for floats `compare_floats`
(`value.rs:1801`) returns `Less` whenever `partial_cmp` is `None`. -/

def cmpFloatModel (aNaN bNaN : Bool) (o : Ordering) : Ordering :=
  if aNaN || bNaN then .lt else o

/-- **Bug A.** NaN is `Less` than 1.0 *and* 1.0 is `Less` than NaN, so the
comparator is not antisymmetric; std's sort detects this and panics
("user-provided comparison function does not correctly implement a total
order"), which kills the server. Same root cause as #2891 (ORDER BY), separate
call site. Repro: `bug_list_sort_nan_panics`. -/
theorem listSort_cmp_not_antisymmetric :
    cmpFloatModel true false .gt = .lt ∧ cmpFloatModel false true .lt = .lt := ⟨rfl, rfl⟩

end FalkorStrList
