import FalkorNativeIndex.Tag
import FalkorNativeIndex.Rows
set_option linter.unusedSimpArgs false

/-!
# `Selection` with the tag kind (#3738, `range.rs`, `tag.rs`)

Since #3738 a `Selection::Scalars` carries a `Scalars { numeric: KeySet, tag: StrSel }`, one share per
kind, and a `ListsHolding` probe is a `ColumnKey` (a number's encoding or a string). This file models the
new `Selection` (`SelT`) and its builders, and proves two things about them:

* **The numeric share is the old `Selection`, exactly.** `lift` embeds the numeric-only `Selection`
  (`Numeric.lean`) into `SelT`; on numeric and `NULL` operands every builder returns the lifted old
  selection (`equalT_numeric`, `rangeT_numeric`, `listsHoldingT_numeric`), `matchesT` of a lifted
  selection is the old `matches` (`matchesT_lift`), and `lookupT` of a lifted selection is the old
  `lookupSel` (`lookupT_lift`). So every theorem about the numeric path (`lookupSel_fromEntries`,
  `range_agrees`, `equalToAny_agrees`, ...) still describes what the new code does for numbers.
* **The tag share is Cypher's string semantics.** `n.v = 's'` selects exactly the string `'s'`
  (`equalT_string_matches`); a string range exactly the strings in the window, in codepoint order
  (`rangeT_string_matches`; Lean's `String.<` is codepoint-lexicographic, which is Rust's byte order on
  UTF-8); bounds of two kinds select nothing (`rangeT_mixed`); `'s' IN n.v` exactly the lists holding
  `'s'` (`listsHoldingT_string_matches`); an `IN` list or a union of equalities is the union of its
  equalities (`equalToAnyT_matches`, `disjunctionT_matches`); a conjunction is the meet of its conjuncts
  (`andT_matches`, `conjunctionT_matches`); and a decline is exactly a value no kind holds
  (`equalT_none_iff`, `equalToAnyT_none_iff`).

`Value.list` in this project keeps only its elements' numeric keys, so a list's *string* elements are a
parameter `strs : Value → List String` (what `TagIndex::for_each_stored_key` emits as `TagArray` keys),
empty on a non-list (`StrsOk`). The string trees themselves (`StrTree`, `StrRanges`) are not modelled:
`lookupT` takes the tag kind's lookups as functions.
-/

namespace NI
namespace Tag

/-! ## `StrSel` (`tag.rs`) -/

/-- One end of a string window: `Bound<Arc<String>>`. -/
inductive Bnd where
  | unb
  | incl (s : String)
  | excl (s : String)
deriving DecidableEq, Repr

/-- `s` is at or past a lower end. -/
def Bnd.loOk : Bnd → String → Bool
  | .unb, _ => true
  | .incl b, s => decide (b ≤ s)
  | .excl b, s => decide (b < s)

/-- `s` is at or before an upper end. -/
def Bnd.hiOk : Bnd → String → Bool
  | .unb, _ => true
  | .incl b, s => decide (s ≤ b)
  | .excl b, s => decide (s < b)

/-- `StrSel`: a set of strings (`Set`, sorted and distinct; `binary_search` on it is membership), or a
    window (`Window`). -/
inductive StrSel where
  | set (ss : List String)
  | window (lo hi : Bnd)
deriving DecidableEq, Repr

/-- `StrSel::none`. -/
def StrSel.none : StrSel := .set []

/-- `StrSel::window`: `None` leaves a side open. -/
def StrSel.ofBounds (mn mx : Option String) (incMin incMax : Bool) : StrSel :=
  let bnd := fun (v : Option String) (inc : Bool) => match v with
    | Option.none => Bnd.unb
    | Option.some v => if inc then Bnd.incl v else Bnd.excl v
  .window (bnd mn incMin) (bnd mx incMax)

/-- `StrSel::contains`. -/
def StrSel.contains : StrSel → String → Bool
  | .set ss, s => decide (s ∈ ss)
  | .window lo hi, s => lo.loOk s && hi.hiOk s

/-- `is_empty_window`. -/
def emptyWindow : Bnd → Bnd → Bool
  | .incl lo, .incl hi => decide (hi < lo)
  | .incl lo, .excl hi | .excl lo, .incl hi | .excl lo, .excl hi => decide (hi ≤ lo)
  | _, _ => false

/-- `StrSel::is_empty`. -/
def StrSel.isEmpty : StrSel → Bool
  | .set ss => ss.isEmpty
  | .window lo hi => emptyWindow lo hi

/-- `cmp_lo(a, b) == Greater`: `a` starts later than `b`. -/
def loGt : Bnd → Bnd → Bool
  | .unb, _ => false
  | _, .unb => true
  | .incl x, .incl y | .excl x, .excl y | .incl x, .excl y => decide (y < x)
  | .excl x, .incl y => decide (y ≤ x)

/-- `cmp_hi(a, b) == Less`: `a` ends earlier than `b`. -/
def hiLt : Bnd → Bnd → Bool
  | .unb, _ => false
  | _, .unb => true
  | .incl x, .incl y | .excl x, .excl y | .incl x, .excl y => decide (x < y)
  | .excl x, .incl y => decide (x ≤ y)

/-- `cmp::max_by(a, b, cmp_lo)` (the second on a tie). -/
def Bnd.maxLo (a b : Bnd) : Bnd := if loGt a b then a else b
/-- `cmp::min_by(a, b, cmp_hi)` (the first on a tie). -/
def Bnd.minHi (a b : Bnd) : Bnd := if hiLt b a then b else a

/-- `StrSel::intersect`. -/
def StrSel.intersect : StrSel → StrSel → StrSel
  | .set a, b => .set (a.filter b.contains)
  | b, .set a => .set (a.filter b.contains)
  | .window alo ahi, .window blo bhi => .window (alo.maxLo blo) (ahi.minHi bhi)

theorem loGt_true (a b : Bnd) (h : loGt a b = true) (s : String) (ha : a.loOk s = true) : b.loOk s = true := by
  cases a <;> cases b <;> simp_all [loGt, Bnd.loOk]
  · exact Std.le_trans (Std.le_of_lt h) ha
  · exact Std.lt_of_lt_of_le h ha
  · exact Std.le_trans h (Std.le_of_lt ha)
  · exact Std.lt_trans h ha

theorem loGt_false (a b : Bnd) (h : loGt a b = false) (s : String) (hb : b.loOk s = true) : a.loOk s = true := by
  cases a <;> cases b <;> simp_all [loGt, Bnd.loOk]
  · exact Std.le_trans (Std.not_lt.1 h) hb
  · exact Std.le_of_lt (Std.lt_of_le_of_lt (Std.not_lt.1 h) hb)
  · exact Std.lt_of_lt_of_le (Std.not_le.1 h) hb
  · exact Std.lt_of_le_of_lt (Std.not_lt.1 h) hb

theorem hiLt_true (a b : Bnd) (h : hiLt a b = true) (s : String) (ha : a.hiOk s = true) : b.hiOk s = true := by
  cases a <;> cases b <;> simp_all [hiLt, Bnd.hiOk]
  · exact Std.le_trans ha (Std.le_of_lt h)
  · exact Std.lt_of_le_of_lt ha h
  · exact Std.le_trans (Std.le_of_lt ha) h
  · exact Std.lt_trans ha h

theorem hiLt_false (a b : Bnd) (h : hiLt a b = false) (s : String) (hb : b.hiOk s = true) : a.hiOk s = true := by
  cases a <;> cases b <;> simp_all [hiLt, Bnd.hiOk]
  · exact Std.le_trans hb (Std.not_lt.1 h)
  · exact Std.le_of_lt (Std.lt_of_lt_of_le hb (Std.not_lt.1 h))
  · exact Std.lt_of_le_of_lt hb (Std.not_le.1 h)
  · exact Std.lt_of_lt_of_le hb (Std.not_lt.1 h)

/-- The later of two lower ends admits exactly what both admit. -/
theorem maxLo_loOk (a b : Bnd) (s : String) : (a.maxLo b).loOk s = (a.loOk s && b.loOk s) := by
  unfold Bnd.maxLo
  cases h : loGt a b
  · simp only [Bool.false_eq_true, ite_false]
    cases hb : b.loOk s
    · simp
    · simp [loGt_false a b h s hb]
  · simp only [ite_true]
    cases ha : a.loOk s
    · simp
    · simp [loGt_true a b h s ha]

/-- The earlier of two upper ends admits exactly what both admit. -/
theorem minHi_hiOk (a b : Bnd) (s : String) : (a.minHi b).hiOk s = (a.hiOk s && b.hiOk s) := by
  unfold Bnd.minHi
  cases h : hiLt b a
  · simp only [Bool.false_eq_true, ite_false]
    cases ha : a.hiOk s
    · simp
    · simp [hiLt_false b a h s ha]
  · simp only [ite_true]
    cases hb : b.hiOk s
    · simp
    · simp [hiLt_true b a h s hb]

/-- **`StrSel::intersect` selects what both select** (a conjunction of string predicates). -/
theorem intersect_contains (a b : StrSel) (s : String) :
    (a.intersect b).contains s = (a.contains s && b.contains s) := by
  cases a with
  | set ss =>
    simp only [StrSel.intersect, StrSel.contains, List.mem_filter]
    by_cases h : s ∈ ss <;> simp [h]
  | window alo ahi =>
    cases b with
    | set ss =>
      simp only [StrSel.intersect, StrSel.contains, List.mem_filter]
      by_cases h : s ∈ ss <;> simp [h, Bool.and_comm]
    | window blo bhi =>
      simp only [StrSel.intersect, StrSel.contains, maxLo_loOk, minHi_hiOk]
      cases alo.loOk s <;> cases ahi.hiOk s <;> cases blo.loOk s <;> cases bhi.hiOk s <;> rfl

/-- **`is_empty` is sound**: a window it calls empty holds no string, so answering it without a cursor
    (and emptying a conjunction with it) loses nothing. -/
theorem isEmpty_contains (t : StrSel) (h : t.isEmpty = true) (s : String) : t.contains s = false := by
  cases t with
  | set ss => cases ss <;> simp_all [StrSel.isEmpty, StrSel.contains]
  | window lo hi =>
    simp only [StrSel.contains]
    cases lo <;> cases hi <;> simp_all [StrSel.isEmpty, emptyWindow, Bnd.loOk, Bnd.hiOk]
    · intro h1 h2; exact Std.lt_irrefl (Std.lt_of_le_of_lt (Std.le_trans h1 h2) h)
    · intro h1 h2; exact Std.lt_irrefl (Std.lt_of_le_of_lt h1 (Std.lt_of_lt_of_le h2 h))
    · intro h1 h2; exact Std.lt_irrefl (Std.lt_of_lt_of_le h1 (Std.le_trans h2 h))
    · intro h1 h2; exact Std.lt_irrefl (Std.lt_trans h1 (Std.lt_of_lt_of_le h2 h))

/-! ## `Scalars` and `Selection` (`range.rs`) -/

/-- `Scalars`: the numeric keys, the strings and (#2527) the geo cells a scalar predicate selects. -/
structure Scalars where
  numeric : KeySet
  tag : StrSel
  geo : KeySet := .empty
deriving DecidableEq, Repr

/-- `Scalars::none`. -/
def Scalars.none : Scalars := ⟨.empty, StrSel.none, .empty⟩

/-- `Scalars::is_empty`. -/
def Scalars.isEmpty (sc : Scalars) : Bool := sc.numeric == .empty && sc.tag.isEmpty && sc.geo == .empty

/-- `GeoIndex::is_scalar_in` (`geo.rs`): a point whose Morton key the cells hold. -/
def geoIn (ks : KeySet) : Value → Bool
  | .point _ _ (some k) => ks.contains k
  | _ => false

/-- `TagIndex::is_scalar_in`: a string the selection holds. -/
def tagIn (t : StrSel) : Value → Bool
  | .string s => t.contains s
  | _ => false

/-- `Scalars::contains`. -/
def Scalars.contains (sc : Scalars) (v : Value) : Bool :=
  isScalarIn sc.numeric v || tagIn sc.tag v || geoIn sc.geo v

/-- `Scalars::and`: kind by kind. -/
def Scalars.and (a b : Scalars) : Scalars := ⟨a.numeric.and b.numeric, a.tag.intersect b.tag, a.geo.and b.geo⟩

/-- `Selection` since #3738. -/
inductive SelT where
  | scalars (sc : Scalars)
  | listsHolding (first : ColumnKey) (rest : List ColumnKey)
deriving DecidableEq, Repr

/-- `Selection::empty`. -/
def SelT.empty : SelT := .scalars Scalars.none

/-- `Selection::is_empty`. -/
def SelT.isEmpty : SelT → Bool
  | .scalars sc => sc.isEmpty
  | _ => false

/-- `is_held_by`: a list holding the element. `strs v` is the string elements of the list `v`. -/
def isHeldBy (strs : Value → List String) : ColumnKey → Value → Bool
  | .encoded k, v => isListHolding k v
  | .tag s, v => decide (s ∈ strs v)

/-- `Selection::matches`. -/
def SelT.matches (strs : Value → List String) : SelT → Value → Bool
  | .scalars sc, v => sc.contains v
  | .listsHolding f r, v => (f :: r).all (fun e => isHeldBy strs e v)

/-- What `strs` must be: a non-list has no string elements. -/
def StrsOk (strs : Value → List String) : Prop := ∀ v, (∃ es, v = .list es) ∨ strs v = []

/-- The numeric-only `Selection` (`Numeric.lean`) as a `SelT`: no string share. -/
def lift : Selection → SelT
  | .scalars ks => .scalars ⟨ks, StrSel.none, .empty⟩
  | .listsHolding f r => .listsHolding (.encoded f) (r.map .encoded)

/-- `Kind::of` returns `Tag` for a string. -/
def isStr : Value → Bool
  | .string _ => true
  | _ => false

/-- `Kind::of` returns `Geo` for a point (#2527). -/
def isPt : Value → Bool
  | .point .. => true
  | _ => false

/-- The geo cell of a point (`GeoIndex::key_of(value).map_or(KeySet::Empty, KeySet::Point)`). -/
def cellOf : Value → KeySet
  | .point _ _ (some k) => .point k
  | _ => .empty

/-- `Selection::equal`. A point selects its grid cell (#2527): a superset of the equal points, which
    the planner's retained filter narrows (`equalT_point_matches`). -/
def equalT (v : Value) : Option SelT :=
  match v with
  | .string s => some (.scalars ⟨.empty, .set [s], .empty⟩)
  | .point _ _ _ => some (.scalars ⟨.empty, StrSel.none, cellOf v⟩)
  | _ => if isNumericKind v then some (.scalars ⟨pointKeys v, StrSel.none, .empty⟩)
         else if v == .null then some SelT.empty else Option.none

/-- `EqualValues::push`: the numeric keys, the strings and the geo cells, or a decline at a value no
    kind holds. -/
def pushEq : List Value → List Nat × List String × List Nat → Option (List Nat × List String × List Nat)
  | [], acc => some acc
  | v :: vs, (ks, ss, cs) =>
    match v with
    | .string s => pushEq vs (ks, ss ++ [s], cs)
    | .point _ _ k => pushEq vs (ks, ss, cs ++ k.toList)
    | _ => if isNumericKind v then pushEq vs (ks ++ (encodeNumeric v).toList, ss, cs)
           else if v == .null then pushEq vs (ks, ss, cs) else Option.none

/-- `EqualValues::selection` (`StrSel::set` sorts and dedups; membership is what it keeps). -/
def eqSel (acc : List Nat × List String × List Nat) : SelT :=
  .scalars ⟨KeySet.ofKeys acc.1, .set acc.2.1, KeySet.ofKeys acc.2.2⟩

/-- `Selection::equal_to_any`. -/
def equalToAnyT (values : List Value) : Option SelT := (pushEq values ([], [], [])).map eqSel

/-- The string of a bound (`as_str`). -/
def strOf : Option Value → Option String
  | some (.string s) => some s
  | _ => Option.none

/-- The largest `u64`. -/
def u64Max : Nat := 2 ^ 64 - 1

/-- `Selection::range`: a bound no kind holds declines; a `NULL` bound, or bounds of two kinds, answer
    empty; string bounds a window of the tag kind; point bounds every point (#2527: the Morton key does
    not follow `compare_value`'s order, so the whole grid `KeySet::Range(0, u64::MAX)`, narrowed by the
    retained filter); else the numeric window. -/
def rangeT (mn mx : Option Value) (incMin incMax : Bool) : Option SelT :=
  let bounds := mn.toList ++ mx.toList
  if bounds.any (fun b => !isNumericKind b && !isStr b && !isPt b && b != Value.null) then Option.none
  else if bounds.any (fun b => b == Value.null) then some SelT.empty
  else if (bounds.any isNumericKind && bounds.any isStr) || (bounds.any isNumericKind && bounds.any isPt) ||
      (bounds.any isStr && bounds.any isPt) then some SelT.empty
  else if bounds.any isStr then some (.scalars ⟨.empty, StrSel.ofBounds (strOf mn) (strOf mx) incMin incMax, .empty⟩)
  else if bounds.any isPt then some (.scalars ⟨.empty, StrSel.none, .range 0 u64Max⟩)
  else some (.scalars ⟨rangeKeys mn mx incMin incMax, StrSel.none, .empty⟩)

/-- `cover_of` (`geo.rs`, NOT COVERED): the Morton windows covering the disk of `radius` (`f64` bits)
    around a point, sorted and disjoint. Left abstract. -/
opaque coverOf : Nat → Nat → Nat → List (Nat × Nat)

/-- `Selection::within` (#2527): `distance(n.v, centre) < radius`. A `NULL` operand, or a radius that is
    not a number, answers empty; a centre that is not a point declines; else the cells covering the disk. -/
def withinT (centre radius : Value) : Option SelT :=
  if centre == .null || radius == .null then some SelT.empty
  else match centre with
    | .point lat lon _ =>
      match radius with
      | .int i => some (.scalars ⟨.empty, StrSel.none, KeySet.ofWindows (coverOf lat lon (i64ToF64 i))⟩)
      | .float f => some (.scalars ⟨.empty, StrSel.none, KeySet.ofWindows (coverOf lat lon f)⟩)
      | _ => some SelT.empty
    | _ => Option.none

/-- `Selection::lists_holding`. -/
def listsHoldingT (v : Value) : Option SelT :=
  match v with
  | .string s => some (.listsHolding (.tag s) [])
  | _ => if isNumericKind v then
           some ((encodeNumeric v).elim SelT.empty (fun k => .listsHolding (.encoded k) []))
         else Option.none

/-- The loop of `Selection::disjunction`. -/
def disjGoT : List IndexQuery → List Nat × List String × List Nat → Option (List Nat × List String × List Nat)
  | [], acc => some acc
  | m :: ms, acc =>
    match disjMemberValues m with
    | Option.none => Option.none
    | some vs => (pushEq vs acc).bind (disjGoT ms)

/-- `Selection::disjunction`. -/
def disjunctionT (members : List IndexQuery) : Option SelT :=
  if !onOneAttribute members then Option.none else (disjGoT members ([], [], [])).map eqSel

/-- `Selection::and`. -/
def SelT.and : SelT → SelT → SelT
  | .scalars a, .scalars b => .scalars (a.and b)
  | .listsHolding f r, .listsHolding f' r' => .listsHolding f (r ++ [f'] ++ r')
  | _, _ => SelT.empty

/-- `Selection::of` on a leaf. -/
def ofLeafT : IndexQuery → Option SelT
  | .equal _ v => equalT v
  | .range _ mn mx a b => rangeT mn mx a b
  | .inList _ (some items) => if items.isEmpty then Option.none else equalToAnyT items
  | .inList _ Option.none => Option.none
  | .arrayContains _ v => listsHoldingT v
  | .point _ c r => withinT c r
  | .or _ => Option.none
  | .and _ => Option.none

/-- `Selection::of` on a conjunct. -/
def ofAtomT : IndexQuery → Option SelT
  | .or ms => (atoms (.or ms)).bind disjunctionT
  | .and _ => Option.none
  | q => ofLeafT q

/-- `Selection::conjunction`. -/
def conjunctionT (cs : List IndexQuery) : Option SelT :=
  if !onOneAttribute cs then Option.none
  else
    let sels := cs.map ofAtomT
    if sels.any (fun s => match s with | some s => s.isEmpty | Option.none => false) then some SelT.empty
    else
      match sels.mapM (fun s => s) with
      | Option.none => Option.none
      | some [] => Option.none
      | some (s :: ss) => some (ss.foldl SelT.and s)

/-- `Selection::of`. -/
def ofT (q : IndexQuery) : Option SelT :=
  match q with
  | .or _ => (atoms q).bind disjunctionT
  | .and _ => (atoms q).bind conjunctionT
  | _ => ofLeafT q

/-! ## The numeric share is the old `Selection` -/

theorem isScalarIn_empty (v : Value) : isScalarIn .empty v = false := by
  rw [isScalarIn_eq, KeySet.matchesValue]; cases encodeNumeric v <;> rfl

theorem tagIn_none (v : Value) : tagIn StrSel.none v = false := by
  cases v <;> simp [tagIn, StrSel.none, StrSel.contains]

theorem geoIn_empty (v : Value) : geoIn .empty v = false := by
  cases v with
  | point a o k => cases k <;> rfl
  | _ => rfl

/-- Only a point is ever in a geo share. -/
theorem geoIn_not_point (ks : KeySet) (v : Value) (h : isPt v = false) : geoIn ks v = false := by
  cases v <;> simp_all [geoIn, isPt]

/-- **A lifted selection matches what the old one did.** -/
theorem matchesT_lift (strs : Value → List String) (sel : Selection) (v : Value) :
    (lift sel).matches strs v = sel.matches v := by
  cases sel with
  | scalars ks => simp [lift, SelT.matches, Scalars.contains, tagIn_none, geoIn_empty, Selection.matches]
  | listsHolding f r =>
    simp only [lift, SelT.matches, Selection.matches, List.all_cons, List.all_map]
    rfl

theorem equalT_numeric (v : Value) (h : isNumericKind v = true) : equalT v = (Selection.equal v).map lift := by
  cases v <;> simp_all [equalT, Selection.equal, isNumericKind, lift, StrSel.none]

theorem equalT_null : equalT .null = some SelT.empty := rfl

/-- **`Equal` declines exactly at a value no kind holds** (neither a number nor a string nor `NULL`). -/
theorem equalT_none_iff (v : Value) : equalT v = Option.none ↔ indexable v = false ∧ v ≠ .null := by
  cases v <;> simp [equalT, indexable, isNumericKind]

theorem listsHoldingT_numeric (v : Value) (h : isNumericKind v = true) :
    listsHoldingT v = (Selection.listsHoldingOf v).map lift := by
  cases v <;> simp_all [listsHoldingT, Selection.listsHoldingOf, isNumericKind] <;>
    (cases encodeNumeric _ <;> rfl)

/-- A probe that is neither a number nor a string (`NULL` included) declines, as before. -/
theorem listsHoldingT_none (v : Value) (h : indexable v = false) : listsHoldingT v = Option.none := by
  cases v <;> simp_all [listsHoldingT, indexable, isNumericKind]

/-- **On bounds that are neither strings nor (#2527) points, `range` is the old numeric `range`.** -/
theorem rangeT_numeric (mn mx : Option Value) (a b : Bool)
    (h : ∀ x ∈ mn.toList ++ mx.toList, isStr x = false ∧ isPt x = false) :
    rangeT mn mx a b = (Selection.range mn mx a b).map lift := by
  have hs : (mn.toList ++ mx.toList).any isStr = false := by
    simp only [List.any_eq_false]; intro x hx; simp [(h x hx).1]
  have hp : (mn.toList ++ mx.toList).any isPt = false := by
    simp only [List.any_eq_false]; intro x hx; simp [(h x hx).2]
  have hL : ∀ L : List Value, (∀ x ∈ L, isStr x = false ∧ isPt x = false) →
      L.any (fun b => !isNumericKind b && !isStr b && !isPt b && b != Value.null) =
      L.any (fun b => !isNumericKind b && b != Value.null) := by
    intro L
    induction L with
    | nil => intro _; rfl
    | cons x xs ih =>
      intro hx
      simp only [List.any_cons]
      rw [ih (fun y hy => hx y (List.mem_cons_of_mem _ hy)), (hx x List.mem_cons_self).1,
        (hx x List.mem_cons_self).2]
      simp
  have he' := hL _ h
  unfold rangeT Selection.range
  dsimp only
  rw [he', hs, hp]
  simp only [Bool.and_false, Bool.or_false, Bool.false_eq_true, ite_false]
  split
  · rfl
  · split
    · rfl
    · simp [lift, StrSel.none]

/-- A `SelT` that `is_empty` says is empty matches nothing. -/
theorem isEmpty_matches (strs : Value → List String) (sel : SelT) (h : sel.isEmpty = true) (v : Value) :
    sel.matches strs v = false := by
  cases sel with
  | listsHolding => simp [SelT.isEmpty] at h
  | scalars sc =>
    obtain ⟨n, t, g⟩ := sc
    simp only [SelT.isEmpty, Scalars.isEmpty, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, ht⟩, rfl⟩ := h
    have h1 : isScalarIn .empty v = false := isScalarIn_empty v
    have h2 : tagIn t v = false := by cases v <;> simp [tagIn, isEmpty_contains t ht]
    simp [SelT.matches, Scalars.contains, h1, h2, geoIn_empty]

theorem empty_matches (strs : Value → List String) (v : Value) : SelT.empty.matches strs v = false :=
  isEmpty_matches strs _ (by decide) v

/-! ## The tag share is string semantics -/

/-- **`n.v = 's'` selects exactly the string `'s'`.** -/
theorem equalT_string_matches (strs : Value → List String) (s : String) (v : Value) :
    ∃ sel, equalT (.string s) = some sel ∧ (sel.matches strs v = true ↔ v = .string s) := by
  refine ⟨_, rfl, ?_⟩
  cases v <;> simp [SelT.matches, Scalars.contains, isScalarIn_empty, tagIn, StrSel.contains, geoIn_empty]

/-- A string in a window with these ends. -/
def inWindow (mn mx : Option String) (incMin incMax : Bool) (s : String) : Prop :=
  (∀ b, mn = some b → if incMin then b ≤ s else b < s) ∧ (∀ b, mx = some b → if incMax then s ≤ b else s < b)

/-- **A range with string bounds selects exactly the strings between them** (and no number). -/
theorem rangeT_string_matches (strs : Value → List String) (mn mx : Option String) (a b : Bool)
    (hsome : mn.isSome ∨ mx.isSome) (v : Value) :
    ∃ sel, rangeT (mn.map .string) (mx.map .string) a b = some sel ∧
      (sel.matches strs v = true ↔ ∃ s, v = .string s ∧ inWindow mn mx a b s) := by
  have hstr : ∀ x ∈ (mn.map Value.string).toList ++ (mx.map Value.string).toList, ∃ s, x = .string s := by
    intro x hx
    cases mn <;> cases mx <;> simp at hx <;>
      first | exact ⟨_, hx⟩ | (rcases hx with hx | hx <;> exact ⟨_, hx⟩)
  have h5 : ((mn.map Value.string).toList ++ (mx.map Value.string).toList).any isPt = false := by
    simp only [List.any_eq_false]; intro x hx; obtain ⟨s, rfl⟩ := hstr x hx; simp [isPt]
  have h1 : ((mn.map Value.string).toList ++ (mx.map Value.string).toList).any
      (fun b => !isNumericKind b && !isStr b && !isPt b && b != Value.null) = false := by
    simp only [List.any_eq_false]; intro x hx; obtain ⟨s, rfl⟩ := hstr x hx; simp [isStr]
  have h2 : ((mn.map Value.string).toList ++ (mx.map Value.string).toList).any (fun b => b == Value.null) = false := by
    simp only [List.any_eq_false]; intro x hx; obtain ⟨s, rfl⟩ := hstr x hx; simp
  have h3 : ((mn.map Value.string).toList ++ (mx.map Value.string).toList).any isNumericKind = false := by
    simp only [List.any_eq_false]; intro x hx; obtain ⟨s, rfl⟩ := hstr x hx; simp [isNumericKind]
  have h4 : ((mn.map Value.string).toList ++ (mx.map Value.string).toList).any isStr = true := by
    cases mn <;> cases mx <;> simp_all [isStr]
  refine ⟨_, by simp only [rangeT, h1, h2, h3, h4, h5]; rfl, ?_⟩
  have hso : ∀ o : Option String, strOf (o.map Value.string) = o := by intro o; cases o <;> rfl
  simp only [hso]
  cases v with
  | string s =>
    simp only [SelT.matches, Scalars.contains, isScalarIn_eq, KeySet.matchesValue, encodeNumeric,
      Option.any_none, Bool.false_or, tagIn, StrSel.ofBounds, StrSel.contains, Value.string.injEq,
      exists_eq_left', inWindow, Bool.and_eq_true, geoIn, Bool.or_false]
    constructor
    · rintro ⟨hl, hh⟩
      refine ⟨fun b' hb => ?_, fun b' hb => ?_⟩
      · subst hb; cases a <;> simpa [Bnd.loOk] using hl
      · subst hb; cases b <;> simpa [Bnd.hiOk] using hh
    · rintro ⟨hl, hh⟩
      refine ⟨?_, ?_⟩
      · cases mn with
        | none => rfl
        | some m => have := hl m rfl; cases a <;> simpa [Bnd.loOk] using this
      · cases mx with
        | none => rfl
        | some m => have := hh m rfl; cases b <;> simpa [Bnd.hiOk] using this
  | _ => simp [SelT.matches, Scalars.contains, isScalarIn_empty, tagIn, geoIn_empty]

/-- **Bounds of two kinds select nothing** (`n.v >= 1 AND n.v < 'z'`: Cypher orders a number only
    with numbers and a string only with strings). -/
theorem rangeT_mixed (x : Value) (s : String) (hx : isNumericKind x = true) (a b : Bool) :
    rangeT (some x) (some (.string s)) a b = some SelT.empty ∧
    rangeT (some (.string s)) (some x) a b = some SelT.empty := by
  cases x <;> simp_all [rangeT, isNumericKind, isStr, isPt]

/-- **`'s' IN n.v` selects exactly the lists holding `'s'`.** -/
theorem listsHoldingT_string_matches (strs : Value → List String) (s : String) (v : Value) :
    ∃ sel, listsHoldingT (.string s) = some sel ∧ (sel.matches strs v = true ↔ s ∈ strs v) := by
  refine ⟨_, rfl, ?_⟩
  simp [SelT.matches, isHeldBy]

/-! ## Unions and conjunctions -/

/-- What `n.v = b` selects for a value `b`, as `equal` builds it (nothing for a decline). -/
def eqMatches (strs : Value → List String) (b v : Value) : Bool := (equalT b).any (fun s => s.matches strs v)

theorem beq_dec_comm (a b : Nat) : (a == b) = decide (b = a) := by
  by_cases h : a = b
  · subst h; simp
  · simp [h, Ne.symm h]

theorem pushEq_spec (strs : Value → List String) :
    ∀ (vs : List Value) (acc acc' : List Nat × List String × List Nat),
    pushEq vs acc = some acc' → ∀ v,
      (eqSel acc').matches strs v = ((eqSel acc).matches strs v || vs.any (fun b => eqMatches strs b v))
  | [], acc, acc', h, v => by simp [pushEq] at h; subst h; simp
  | b :: vs, (ks, ss, cs), acc', h, v => by
    have key : ∀ ks' ss' cs', pushEq vs (ks', ss', cs') = some acc' →
        (eqSel (ks', ss', cs')).matches strs v = ((eqSel (ks, ss, cs)).matches strs v || eqMatches strs b v) →
        (eqSel acc').matches strs v =
          ((eqSel (ks, ss, cs)).matches strs v || (b :: vs).any (fun b => eqMatches strs b v)) := by
      intro ks' ss' cs' hp he
      rw [pushEq_spec strs vs _ _ hp v, he, List.any_cons, Bool.or_assoc]
    have hsc : ∀ (n : KeySet) (t : StrSel) (g : KeySet), (SelT.scalars ⟨n, t, g⟩).matches strs v =
        (isScalarIn n v || tagIn t v || geoIn g v) := by
      intro n t g; rfl
    cases b with
    | string s =>
      apply key ks (ss ++ [s]) cs (by simpa [pushEq] using h)
      simp only [eqSel, hsc, eqMatches, equalT, Option.any_some]
      cases v with
      | point a o k => cases k <;> simp [tagIn, StrSel.contains, isScalarIn_empty, geoIn_empty, geoIn, KeySet.contains]
      | _ => simp [tagIn, StrSel.contains, isScalarIn_empty, geoIn_empty, geoIn, Bool.or_assoc]
    | point la lo k =>
      apply key ks ss (cs ++ k.toList) (by simpa [pushEq] using h)
      simp only [eqSel, hsc, eqMatches, equalT, Option.any_some]
      cases v with
      | point la' lo' k' =>
        cases k' with
        | none => simp [geoIn, isScalarIn_eq, KeySet.matchesValue, encodeNumeric, tagIn, cellOf]
        | some k' =>
          simp only [geoIn, isScalarIn_empty, tagIn_none, ofKeys_contains, List.mem_append, cellOf]
          cases k <;> simp [KeySet.contains, tagIn, isScalarIn_eq, KeySet.matchesValue, encodeNumeric,
            beq_dec_comm, Bool.or_assoc] <;> (cases decide (k' ∈ cs) <;> simp)
      | _ => simp [geoIn, tagIn, tagIn_none, isScalarIn_empty, cellOf, StrSel.none, StrSel.contains]
    | null =>
      apply key ks ss cs (by simpa [pushEq, isNumericKind] using h)
      simp [eqMatches, equalT, isNumericKind, empty_matches]
    | int i | float x | bool c =>
      apply key (ks ++ (encodeNumeric _).toList) ss cs (by simpa [pushEq, isNumericKind] using h)
      simp only [eqSel, hsc, eqMatches, equalT, isNumericKind, ite_true, Option.any_some]
      simp only [isScalarIn_eq, KeySet.matchesValue, tagIn_none, geoIn_empty, Bool.or_false]
      cases hv : encodeNumeric v with
      | none => simp
      | some k =>
        have hg : geoIn (KeySet.ofKeys cs) v = false := by
          cases v <;> simp_all [encodeNumeric, geoIn]
        simp only [Option.any_some, ofKeys_contains, List.mem_append, pointKeys, hg, Bool.or_false]
        cases encodeNumeric _ <;> simp [KeySet.contains] <;>
          (cases decide (k ∈ ks) <;> cases tagIn (StrSel.set ss) v <;> simp [beq_dec_comm])
    | _ => simp [pushEq, isNumericKind] at h

theorem pushEq_none_iff : ∀ (vs : List Value) (acc : List Nat × List String × List Nat),
    pushEq vs acc = Option.none ↔ ∃ b ∈ vs, equalT b = Option.none
  | [], acc => by simp [pushEq]
  | b :: vs, (ks, ss, cs) => by
    cases b <;> simp [pushEq, isNumericKind, equalT, pushEq_none_iff vs]

theorem eqSel_nil_matches (strs : Value → List String) (v : Value) :
    (eqSel ([], [], [])).matches strs v = false := by
  have : isScalarIn (KeySet.ofKeys []) v = false := by
    simp [isScalarIn_eq, KeySet.matchesValue]; cases encodeNumeric v <;> simp [ofKeys_contains]
  have hg : geoIn (KeySet.ofKeys []) v = false := by
    cases v with
    | point a o k => cases k <;> simp [geoIn, ofKeys_contains]
    | _ => rfl
  simp [eqSel, SelT.matches, Scalars.contains, this, hg, tagIn]; cases v <;> simp [tagIn, StrSel.contains]

/-- **`n.v IN [b, ...]` selects what `n.v = b OR ...` does: numbers by key, strings by value** — and
    declines exactly when some member is a value no kind holds. -/
theorem equalToAnyT_matches (strs : Value → List String) (items : List Value) (sel : SelT)
    (h : equalToAnyT items = some sel) (v : Value) :
    sel.matches strs v = items.any (fun b => eqMatches strs b v) := by
  unfold equalToAnyT at h
  cases hp : pushEq items ([], [], []) with
  | none => simp [hp] at h
  | some acc =>
    simp [hp] at h; subst h
    rw [pushEq_spec strs items _ _ hp v, eqSel_nil_matches, Bool.false_or]

theorem equalToAnyT_none_iff (items : List Value) :
    equalToAnyT items = Option.none ↔ ∃ b ∈ items, indexable b = false ∧ b ≠ .null := by
  simp only [equalToAnyT, Option.map_eq_none_iff, pushEq_none_iff, equalT_none_iff]

/-- The values a union's members name. -/
def memberValuesT : List IndexQuery → List Value
  | [] => []
  | m :: ms => (disjMemberValues m).getD [] ++ memberValuesT ms

theorem disjGoT_spec (strs : Value → List String) :
    ∀ (ms : List IndexQuery) (acc acc' : List Nat × List String × List Nat),
    disjGoT ms acc = some acc' → ∀ v,
      (eqSel acc').matches strs v = ((eqSel acc).matches strs v || (memberValuesT ms).any (fun b => eqMatches strs b v))
  | [], acc, acc', h, v => by simp [disjGoT] at h; subst h; simp [memberValuesT]
  | m :: ms, acc, acc', h, v => by
    simp only [disjGoT] at h
    cases hm : disjMemberValues m with
    | none => simp [hm] at h
    | some vs =>
      simp only [hm, Option.bind_eq_some_iff] at h
      obtain ⟨mid, hp, hg⟩ := h
      rw [disjGoT_spec strs ms mid acc' hg v, pushEq_spec strs vs acc mid hp v]
      simp [memberValuesT, hm, List.any_append, Bool.or_assoc]

/-- **A union of `=`s and `IN` lists on one attribute selects the union of its equalities**, strings
    included (`n.v = 1 OR n.v = 'a'` reads both kinds). -/
theorem disjunctionT_matches (strs : Value → List String) (ms : List IndexQuery) (sel : SelT)
    (h : disjunctionT ms = some sel) (v : Value) :
    onOneAttribute ms = true ∧ sel.matches strs v = (memberValuesT ms).any (fun b => eqMatches strs b v) := by
  unfold disjunctionT at h
  by_cases ho : onOneAttribute ms = true
  · simp only [ho, Bool.not_true, Bool.false_eq_true, ite_false] at h
    cases hg : disjGoT ms ([], [], []) with
    | none => simp [hg] at h
    | some acc =>
      simp [hg] at h; subst h
      exact ⟨ho, by rw [disjGoT_spec strs ms _ _ hg v, eqSel_nil_matches, Bool.false_or]⟩
  · simp [ho] at h

/-- `isScalarIn` holds only for a non-list, non-string value. -/
theorem isScalarIn_string (ks : KeySet) (s : String) : isScalarIn ks (.string s) = false := by
  simp [isScalarIn_eq, KeySet.matchesValue, encodeNumeric]

theorem isScalarIn_list (ks : KeySet) (es : List (Option Nat)) : isScalarIn ks (.list es) = false := by
  simp [isScalarIn_eq, KeySet.matchesValue, encodeNumeric]

theorem isHeldBy_list (strs : Value → List String) (hs : StrsOk strs) (e : ColumnKey) (v : Value)
    (h : isHeldBy strs e v = true) : ∃ es, v = .list es := by
  cases e with
  | encoded k =>
    rw [isHeldBy, isListHolding_iff] at h
    cases v <;> simp [arrayKeys] at h ⊢
  | tag s =>
    rcases hs v with hv | hv
    · exact hv
    · simp [isHeldBy, hv] at h

theorem scalars_contains_not_list (sc : Scalars) (es : List (Option Nat)) : sc.contains (.list es) = false := by
  simp [Scalars.contains, isScalarIn_list, tagIn, geoIn]

/-- **`Selection::and` selects what both select** — kind by kind on scalars (a number is never a
    string), and nothing for a scalar and a list selection. -/
theorem andT_matches (strs : Value → List String) (hs : StrsOk strs) (a b : SelT) (v : Value) :
    (a.and b).matches strs v = (a.matches strs v && b.matches strs v) := by
  cases a with
  | scalars sa =>
    cases b with
    | scalars sb =>
      simp only [SelT.and, SelT.matches, Scalars.contains, Scalars.and]
      have hk : isScalarIn (sa.numeric.and sb.numeric) v = (isScalarIn sa.numeric v && isScalarIn sb.numeric v) := by
        have := selection_and_matches (.scalars sa.numeric) (.scalars sb.numeric) v
        simpa [Selection.and, Selection.matches] using this
      rw [hk]
      cases v with
      | string s =>
        simp only [isScalarIn_string, tagIn, intersect_contains, geoIn]; simp
      | point a o k =>
        cases k with
        | none => simp [isScalarIn_eq, KeySet.matchesValue, encodeNumeric, tagIn, geoIn]
        | some k => simp [isScalarIn_eq, KeySet.matchesValue, encodeNumeric, tagIn, geoIn, keyset_and_contains]
      | _ => simp [tagIn, geoIn]
    | listsHolding f r =>
      simp only [SelT.and, empty_matches]
      cases hm : (SelT.listsHolding f r).matches strs v
      · simp
      · have : isHeldBy strs f v = true := by simp [SelT.matches] at hm; exact hm.1
        obtain ⟨es, rfl⟩ := isHeldBy_list strs hs f v this
        simp [SelT.matches, scalars_contains_not_list]
  | listsHolding f r =>
    cases b with
    | scalars sb =>
      simp only [SelT.and, empty_matches]
      cases hm : (SelT.listsHolding f r).matches strs v
      · simp
      · have : isHeldBy strs f v = true := by simp [SelT.matches] at hm; exact hm.1
        obtain ⟨es, rfl⟩ := isHeldBy_list strs hs f v this
        simp [SelT.matches, scalars_contains_not_list]
    | listsHolding f' r' =>
      simp only [SelT.and, SelT.matches, List.all_cons, List.all_append, List.all_nil, Bool.and_true]
      simp only [Bool.and_assoc]

theorem foldl_andT_matches (strs : Value → List String) (hs : StrsOk strs) (v : Value) :
    ∀ (ss : List SelT) (s : SelT),
      (ss.foldl SelT.and s).matches strs v = (s.matches strs v && ss.all (fun x => x.matches strs v))
  | [], s => by simp
  | x :: xs, s => by
    rw [List.foldl_cons, foldl_andT_matches strs hs v xs, andT_matches strs hs, List.all_cons, Bool.and_assoc]

theorem mapM_id_mem {α : Type} : ∀ (l : List (Option α)) (r : List α), l.mapM (fun s => s) = some r →
    (∀ o ∈ l, ∃ x, o = some x ∧ x ∈ r) ∧ (∀ x ∈ r, some x ∈ l)
  | [], r, h => by simp at h; subst h; simp
  | o :: l, r, h => by
    cases o with
    | none => simp [List.mapM_cons] at h
    | some a =>
      cases hl : l.mapM (fun s => s) with
      | none => simp [List.mapM_cons, hl] at h
      | some r' =>
        simp [List.mapM_cons, hl] at h; subst h
        obtain ⟨ih1, ih2⟩ := mapM_id_mem l r' hl
        refine ⟨?_, ?_⟩
        · intro o ho
          rcases List.mem_cons.1 ho with rfl | ho
          · exact ⟨a, rfl, List.mem_cons_self⟩
          · obtain ⟨x, hx, hxr⟩ := ih1 o ho; exact ⟨x, hx, List.mem_cons_of_mem _ hxr⟩
        · intro x hx
          rcases List.mem_cons.1 hx with rfl | hx
          · exact List.mem_cons_self
          · exact List.mem_cons_of_mem _ (ih2 x hx)

/-- **A conjunction on one attribute selects exactly what every conjunct selects**, strings
    included; a conjunct selecting nothing by construction empties it (sound: it matches nothing). -/
theorem conjunctionT_matches (strs : Value → List String) (hs : StrsOk strs) (cs : List IndexQuery) (sel : SelT)
    (h : conjunctionT cs = some sel) (v : Value) :
    sel.matches strs v = true ↔ ∀ c ∈ cs, ∃ s, ofAtomT c = some s ∧ s.matches strs v = true := by
  unfold conjunctionT at h
  by_cases ho : onOneAttribute cs = true
  · simp only [ho, Bool.not_true, Bool.false_eq_true, ite_false] at h
    split at h
    · rename_i hany
      simp at h; subst h
      simp only [empty_matches, Bool.false_eq_true, false_iff]
      simp only [List.any_eq_true, List.mem_map] at hany
      obtain ⟨o, ⟨c, hc, rfl⟩, ho'⟩ := hany
      intro hall
      obtain ⟨s, hs', hm⟩ := hall c hc
      rw [hs'] at ho'
      simp [isEmpty_matches strs s ho' v] at hm
    · split at h
      · simp at h
      · simp at h
      · rename_i s ss hm
        simp at h; subst h
        obtain ⟨h1, h2⟩ := mapM_id_mem _ _ hm
        rw [foldl_andT_matches strs hs v,
          show (SelT.matches strs s v && ss.all fun x => SelT.matches strs x v) =
            (s :: ss).all (fun x => SelT.matches strs x v) from rfl, List.all_eq_true]
        constructor
        · intro hall c hc
          obtain ⟨x, hx, hxr⟩ := h1 (ofAtomT c) (List.mem_map_of_mem hc)
          exact ⟨x, hx, hall x hxr⟩
        · intro hall x hx
          obtain ⟨c, hc, hcx⟩ := List.mem_map.1 (h2 x hx)
          obtain ⟨y, hy, hym⟩ := hall c hc
          rw [hcx] at hy
          cases hy
          exact hym
  · simp [ho] at h

/-! ## `RangeIndex::lookup` -/

/-- `RangeIndex::lists_holding` (private): the lists holding one element, in the kind that holds it.
    `tagLists` is `TagIndex::lists_holding`. -/
def listsT (ix : NumericIndex) (tagLists : String → List Nat) : ColumnKey → List Nat
  | .encoded k => ix.listsHolding k
  | .tag s => tagLists s

/-- The geo key of a stored point (`GeoIndex::key_of`), `none` for anything else. -/
def geoKey : Value → Option Nat
  | .point _ _ k => k
  | _ => Option.none

theorem geoIn_eq (ks : KeySet) (v : Value) :
    geoIn ks v = ((geoKey v).map ks.contains).getD false := by
  cases v with
  | point a o k => cases k <;> rfl
  | _ => rfl

theorem scalars_geo_matches (strs : Value → List String) (g : KeySet) (v : Value) :
    SelT.matches strs (.scalars ⟨.empty, StrSel.none, g⟩) v = geoIn g v := by
  simp [SelT.matches, Scalars.contains, isScalarIn_empty, tagIn_none]

/-- **`n.v = p` on a point (#2527) selects exactly the points in `p`'s grid cell** (the same Morton key):
    `p` itself among them, so no equal point is lost; a point with a NaN coordinate (no key) selects
    nothing, as it equals nothing. The planner's retained filter drops the cell's other points. -/
theorem equalT_point_matches (strs : Value → List String) (a o : Nat) (k : Option Nat) (v : Value) :
    ∃ sel, equalT (.point a o k) = some sel ∧
      sel.matches strs v = (k.isSome && geoKey v == k) := by
  refine ⟨_, rfl, ?_⟩
  rw [scalars_geo_matches]
  cases k with
  | none => cases v with
    | point _ _ k' => cases k' <;> rfl
    | _ => rfl
  | some c =>
    cases v with
    | point _ _ k' =>
      cases k' with
      | none => rfl
      | some d =>
        simp only [cellOf, geoIn, geoKey, KeySet.contains, Option.isSome_some, Bool.true_and]
        rw [Bool.eq_iff_iff]; simp only [beq_iff_eq, Option.some.injEq]; exact ⟨fun h => h.symm, fun h => h.symm⟩
    | _ => rfl

/-- Hence an equal point (same coordinates, so the same key) is always selected. -/
theorem equalT_point_self (strs : Value → List String) (a o c : Nat) :
    ∃ sel, equalT (.point a o (some c)) = some sel ∧ sel.matches strs (.point a o (some c)) = true := by
  obtain ⟨sel, h, hm⟩ := equalT_point_matches strs a o (some c) (.point a o (some c))
  exact ⟨sel, h, by simp [hm, geoKey]⟩

/-- **A range on point bounds (#2527) selects every point with a key** (`KeySet::Range(0, u64::MAX)`:
    a Morton key is a `u64`), narrowed by the retained filter. -/
theorem rangeT_point (strs : Value → List String) (mn mx : Option Value) (a b : Bool)
    (hpt : ∀ x ∈ mn.toList ++ mx.toList, isPt x = true) (hne : mn.toList ++ mx.toList ≠ []) (v : Value) :
    ∃ sel, rangeT mn mx a b = some sel ∧
      sel.matches strs v = ((geoKey v).map (fun k => decide (k ≤ u64Max))).getD false := by
  have hx : ∀ x ∈ mn.toList ++ mx.toList,
      isNumericKind x = false ∧ isStr x = false ∧ x ≠ Value.null := by
    intro x hx'
    have := hpt x hx'
    cases x <;> simp_all [isPt, isNumericKind, isStr]
  obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hne
  have h1 : (mn.toList ++ mx.toList).any (fun b => !isNumericKind b && !isStr b && !isPt b && b != Value.null) = false := by
    rw [List.any_eq_false]; intro x hx'; simp [hpt x hx']
  have h2 : (mn.toList ++ mx.toList).any (fun b => b == Value.null) = false := by
    rw [List.any_eq_false]; intro x hx'; simp [(hx x hx').2.2]
  have h3 : (mn.toList ++ mx.toList).any isNumericKind = false := by
    rw [List.any_eq_false]; intro x hx'; simp [(hx x hx').1]
  have h4 : (mn.toList ++ mx.toList).any isStr = false := by
    rw [List.any_eq_false]; intro x hx'; simp [(hx x hx').2.1]
  have h5 : (mn.toList ++ mx.toList).any isPt = true := List.any_eq_true.mpr ⟨y, hy, hpt y hy⟩
  refine ⟨.scalars ⟨.empty, StrSel.none, .range 0 u64Max⟩, ?_, ?_⟩
  · simp [rangeT, h1, h2, h3, h4, h5]
  · rw [scalars_geo_matches, geoIn_eq]
    cases geoKey v <;> simp [KeySet.contains]

/-- `distance(n.v, c) < r` with a `NULL` operand answers empty. -/
theorem withinT_null (c r : Value) (h : c = .null ∨ r = .null) : withinT c r = some SelT.empty := by
  rcases h with rfl | rfl <;> simp [withinT]

/-- A centre that is neither `NULL` nor a point declines (the scan is not used). -/
theorem withinT_not_point (c r : Value) (hc : c ≠ .null) (hp : isPt c = false) (hr : r ≠ .null) :
    withinT c r = Option.none := by
  cases c <;> simp_all [withinT, isPt]

/-- **`distance(n.v, centre) < radius` (#2527) selects exactly the points whose key lies in a window of
    the cover of the disk** (`cover_of`, abstract: the proof that the cover holds every point of the disk
    is geometry, NOT COVERED). A radius that is not a number answers empty. -/
theorem withinT_point_matches (strs : Value → List String) (lat lon : Nat) (k : Option Nat) (r : Value)
    (f : Nat) (hr : r = .float f ∨ ∃ i, r = .int i ∧ f = i64ToF64 i) (v : Value) :
    ∃ sel, withinT (.point lat lon k) r = some sel ∧
      sel.matches strs v =
        ((geoKey v).map (fun key => (coverOf lat lon f).any (fun w => decide (w.1 ≤ key ∧ key ≤ w.2)))).getD false := by
  rcases hr with rfl | ⟨i, rfl, rfl⟩ <;>
  · refine ⟨_, by simp [withinT]; rfl, ?_⟩
    rw [scalars_geo_matches, geoIn_eq]
    cases geoKey v <;> simp [ofWindows_contains]

theorem withinT_other_radius (lat lon : Nat) (k : Option Nat) (r : Value) (hr : r ≠ .null)
    (hn : ∀ f, r ≠ .float f) (hi : ∀ i, r ≠ .int i) : withinT (.point lat lon k) r = some SelT.empty := by
  cases r <;> simp_all [withinT]

/-- `GeoIndex::scalars` (`geo.rs`): `docs_in` over the geo tree `gt` of `(Morton key, doc)` entries,
    the same cursor code as the numeric kind's (`NumericIndex.lookup` on a tree, `WindowIter` for
    `Windows`). -/
def geoLookup (gt : List Nat) (ks : KeySet) : List Nat := (NumericIndex.mk gt []).lookup ks

/-- `RangeIndex::rows` on scalars (#2527): one stream per kind whose share is not empty, in the order
    numeric, tag, geo; none is the numeric kind's empty read, one is that stream, more are chained
    (`DistinctIter::disjoint`: a scalar is of one kind, so they share no doc). -/
def scalarStreams (ix : NumericIndex) (gt : List Nat) (tagScalars : StrSel → List Nat) (sc : Scalars) :
    List (List Nat) :=
  (if sc.numeric == .empty then [] else [ix.lookup sc.numeric]) ++
    (if sc.tag.isEmpty then [] else [tagScalars sc.tag]) ++
    (if sc.geo == .empty then [] else [geoLookup gt sc.geo])

/-- `RangeIndex::lookup`. `tagScalars` is `TagIndex::scalars`; `gt` is the geo kind's tree. -/
def lookupT (ix : NumericIndex) (gt : List Nat) (tagScalars : StrSel → List Nat) (tagLists : String → List Nat) :
    SelT → List Nat
  | .scalars sc =>
    match scalarStreams ix gt tagScalars sc with
    | [] => ix.lookup .empty
    | [s] => s
    | ss => ss.flatten
  | .listsHolding first rest =>
    rest.foldl (fun found e =>
      if found.isEmpty then found else (listsT ix tagLists e).filter (fun d => found.contains d))
      (listsT ix tagLists first)

/-- **On a lifted (numeric) selection, `lookup` is the old `lookupSel`**, whatever the tag and geo kinds
    hold: every theorem about the numeric lookup (`lookupSel_fromEntries`, ...) describes the new code. -/
theorem lookupT_lift (ix : NumericIndex) (gt : List Nat) (ts : StrSel → List Nat) (tl : String → List Nat)
    (s : Selection) : lookupT ix gt ts tl (lift s) = ix.lookupSel s := by
  cases s with
  | scalars ks =>
    simp only [lift, lookupT, scalarStreams, NumericIndex.lookupSel, StrSel.none, StrSel.isEmpty,
      List.isEmpty_nil]
    cases h : ks == KeySet.empty
    · simp
    · simp only [beq_iff_eq] at h; subst h; simp
  | listsHolding f r =>
    simp only [lift, lookupT, NumericIndex.lookupSel, NumericIndex.lookupLists, List.foldl_map, listsT]

/-- **A scalar lookup returns the numeric, tag and geo shares' docs**, each read only when it can hold
    one (`tagScalars` of an empty `StrSel` is empty, and the empty `KeySet` reads nothing). -/
theorem lookupT_scalars_mem (ix : NumericIndex) (gt : List Nat) (ts : StrSel → List Nat) (tl : String → List Nat)
    (sc : Scalars) (hT : sc.tag.isEmpty = true → ts sc.tag = [])
    (hN : ix.lookup .empty = []) (d : Nat) :
    d ∈ lookupT ix gt ts tl (.scalars sc) ↔
      d ∈ ix.lookup sc.numeric ∨ d ∈ ts sc.tag ∨ d ∈ geoLookup gt sc.geo := by
  have hG : geoLookup gt .empty = [] := by
    simp [geoLookup, NumericIndex.lookup, NumericIndex.empty, tRange_empty]
  have hflat : d ∈ lookupT ix gt ts tl (.scalars sc) ↔ ∃ s ∈ scalarStreams ix gt ts sc, d ∈ s := by
    simp only [lookupT]
    split
    · rename_i h; simp [h, hN]
    · rename_i s h; simp [h]
    · simp [List.mem_flatten]
  rw [hflat]
  simp only [scalarStreams]
  by_cases hn : sc.numeric = KeySet.empty <;> by_cases hg : sc.geo = KeySet.empty <;>
    cases ht : sc.tag.isEmpty <;> simp [hn, hg, hT, ht, hN, hG]

/-- **A point column read with a cell set finds exactly the docs stored under a key the set holds** —
    `Windows` (a distance's cell cover, `WindowIter`) included. -/
theorem geoLookup_mem (gt : List Nat) (ks : KeySet) (id : Nat) (hid : id < W) :
    id ∈ geoLookup gt ks ↔ ∃ k, enc k id ∈ gt ∧ ks.contains k = true :=
  lookup_mem (NumericIndex.mk gt []) ks id hid

/-- **`rows_at(Geo, k)` (#2527): `geo.scalars(KeySet::Point(k))`, the docs stored under exactly `k`.** -/
theorem geoLookup_point_mem (gt : List Nat) (k id : Nat) (hid : id < W) :
    id ∈ geoLookup gt (.point k) ↔ enc k id ∈ gt := by
  rw [geoLookup_mem gt _ id hid]
  constructor
  · rintro ⟨k', h, hc⟩
    simp only [KeySet.contains, beq_iff_eq] at hc
    subst hc; exact h
  · intro h; exact ⟨k, h, by simp [KeySet.contains]⟩

end Tag
end NI
