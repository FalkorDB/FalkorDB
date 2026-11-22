import FalkorNativeIndex.Tree
set_option linter.unusedSimpArgs false

/-!
# `NumericIndex` (`graph/src/index/falkordb/numeric.rs`)

The index is its tree (`tree: Tree`, `numeric.rs:139-141`), here the tree's content list (`Tree.lean`).
Each function is modelled line by line. Theorems:
* `range_keys` — the inclusive key bounds `range` computes (`numeric.rs:269-316`) select exactly the
  keys satisfying the value bounds, including the exclusive `k ± 1` steps and their `checked_*`
  guards (no stored key is `0` or `u64::MAX`, `encodeF64_bounds`).
* `range_agrees`, `point_agrees` — **index scan ≡ label scan + filter**: a column built from
  `(value, id)` entries returns exactly the ids whose value passes the Filter, for `Bool`-free data and
  guarded numeric bounds.
* `add_spec`, `remove_spec`, `addBatch_spec`, `removeBatch_spec`, `addEncoded_spec`,
  `removeEncoded_spec`, `fromEntries_spec`: contents after each write.
-/

namespace NI

/-- `KeyKind` (`encode.rs:44-48`): which of a column's trees a key lives in. A scalar value's key goes in
    the scalar tree; each numeric element of a list value goes in the array tree. -/
inductive KeyKind where
  | scalar
  | array
deriving DecidableEq, Repr

/-- `for_each_stored_key` (`encode.rs:74-92`): a list is stored under each numeric element's key in the
    array tree (`items.iter().filter_map(encode_numeric)`), anything else under its own `encode_numeric`
    key in the scalar tree. -/
def forEachStoredKey : Value → List (KeyKind × Nat)
  | .list es => (es.filterMap id).map (fun k => (KeyKind.array, k))
  | v => match encodeNumeric v with
    | some k => [(KeyKind.scalar, k)]
    | none => []

/-- `EncodedTuples { by_kind }` (`numeric.rs:231-233`): the encoded `(key, doc)` tuples a column's
    changes move in, kept apart by `KeyKind` (`scalar` = `by_kind[Scalar]`, `array` = `by_kind[Array]`).
    A tuple `(key, doc)` is its tree entry `enc key doc` (`Tree.lean`). -/
structure EncodedTuples where
  scalar : List Nat
  array : List Nat
deriving Repr, DecidableEq

namespace EncodedTuples
/-- `Default` -/
def empty : EncodedTuples := ⟨[], []⟩
/-- `Index<KeyKind>` (`numeric.rs:235-243`). -/
def get (t : EncodedTuples) : KeyKind → List Nat
  | .scalar => t.scalar
  | .array => t.array
/-- `tuples[kind].push(x)` (`IndexMut<KeyKind>`, `numeric.rs:245-252`). -/
def push (t : EncodedTuples) : KeyKind → Nat → EncodedTuples
  | .scalar, x => { t with scalar := t.scalar ++ [x] }
  | .array, x => { t with array := t.array ++ [x] }
/-- `scalars` (`numeric.rs:269-273`). -/
def scalars (s : List Nat) : EncodedTuples := ⟨s, []⟩
/-- `is_empty` (`numeric.rs:277-279`): no tuple of either kind. -/
def isEmpty (t : EncodedTuples) : Bool := t.scalar.isEmpty && t.array.isEmpty
/-- `len` (`numeric.rs:285-287`): every kind's tuples. -/
def len (t : EncodedTuples) : Nat := t.scalar.length + t.array.length
/-- `clear` (`numeric.rs:290-292`). -/
def clear (_t : EncodedTuples) : EncodedTuples := ⟨[], []⟩
/-- `sort_dedup` (`numeric.rs:296-301`): each kind `sort_unstable()` then `dedup()`. -/
def sortDedup (t : EncodedTuples) : EncodedTuples := ⟨NI.sortDedup t.scalar, NI.sortDedup t.array⟩
end EncodedTuples

/-- `NumericIndex { tree, array_tree }` (`numeric.rs:316-327`). -/
structure NumericIndex where
  tree : List Nat
  arrayTree : List Nat
deriving Repr

/-- `NumericIndex::default` (`numeric.rs:316`). -/
def NumericIndex.new : NumericIndex := ⟨[], []⟩

/-- The scalar `(key, id)` entries of `(value, id)` pairs: the `filter_map` over `encode_numeric`. -/
def encodePairs (entries : List (Value × Nat)) : List Nat :=
  entries.filterMap (fun (v, id) => (encodeNumeric v).map (fun k => enc k id))

/-- The array-tree entries of `(value, id)` pairs: a list value's numeric elements (`for_each_stored_key`). -/
def encodeArrayPairs (entries : List (Value × Nat)) : List Nat :=
  entries.flatMap (fun (v, id) => match v with
    | .list es => (es.filterMap _root_.id).map (fun k => enc k id)
    | _ => [])

/-- `RangeIndex::push_encoded` (`range.rs:452-459`): `for_each_stored_key`, pushing `(key, id)` onto the
    tuples of the key's kind. -/
def pushStored (v : Value) (id : Nat) (out : EncodedTuples) : EncodedTuples :=
  (forEachStoredKey v).foldl (fun o kk => o.push kk.1 (enc kk.2 id)) out

/-- What the column built from `entries` holds (the specification of a built column; `from_entries`
    until #3736 was the test-only constructor): both trees, encoded, `sort_unstable`, `dedup`,
    `from_sorted`. -/
def NumericIndex.fromEntries (entries : List (Value × Nat)) : NumericIndex :=
  ⟨tFromSorted (sortDedup (encodePairs entries)), tFromSorted (sortDedup (encodeArrayPairs entries))⟩

/-- `RangeKind::add_encoded` for one tree (`numeric.rs:438-455`): skip an empty share; into an empty
    tree `Tree::from_sorted` (a bulk build), else `insert_batch`. -/
def addTree (t new : List Nat) : List Nat :=
  if new.isEmpty then t else if tIsEmpty t then tFromSorted new else tInsertBatch t new

/-- `RangeKind::remove_encoded` for one tree (`numeric.rs:456-463`). -/
def removeTree (t rm : List Nat) : List Nat :=
  if rm.isEmpty then t else tRemoveBatch t rm

/-- `NumericIndex::add_encoded` (`numeric.rs:438-455`, the `RangeKind` impl): takes tuples the column
    already sorted, one tree per kind. -/
def NumericIndex.addTrees (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex :=
  ⟨addTree ix.tree tuples.scalar, addTree ix.arrayTree tuples.array⟩

/-- `NumericIndex::remove_encoded` (`numeric.rs:456-463`). -/
def NumericIndex.removeTrees (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex :=
  ⟨removeTree ix.tree tuples.scalar, removeTree ix.arrayTree tuples.array⟩

/-- `RangeIndex::add_encoded` (`range.rs:516-523`): `tuples.sort_dedup()`, then the numeric kind's
    `add_encoded`. Returns the index and the (sorted) tuples. -/
def NumericIndex.addEncoded (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex × EncodedTuples :=
  let tuples := tuples.sortDedup
  (ix.addTrees tuples, tuples)

/-- `RangeIndex::remove_encoded` (`range.rs:526-533`). -/
def NumericIndex.removeEncoded (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex × EncodedTuples :=
  let tuples := tuples.sortDedup
  (ix.removeTrees tuples, tuples)

/-- `RangeIndex::add` (`range.rs:462-469`): the tuples of `(v, id)` in an empty buffer, `add_encoded`. -/
def NumericIndex.add (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  (ix.addEncoded (pushStored v id EncodedTuples.empty)).1

/-- `RangeIndex::remove` (`range.rs:474-481`). -/
def NumericIndex.remove (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  (ix.removeEncoded (pushStored v id EncodedTuples.empty)).1

/-- The tuples of a batch of `(value, id)` entries: `scratch.clear()`, then `push_encoded` each. -/
def pushAll (entries : List (Value × Nat)) : EncodedTuples :=
  entries.foldl (fun o p => pushStored p.1 p.2 o) EncodedTuples.empty

/-- `RangeIndex::add_batch` (`range.rs:489-496`); returns the index and the scratch buffer (sorted by
    `add_encoded`). -/
def NumericIndex.addBatch (ix : NumericIndex) (entries : List (Value × Nat)) :
    NumericIndex × EncodedTuples :=
  ix.addEncoded (pushAll entries)

/-- `RangeIndex::remove_batch` (`range.rs:502-509`). -/
def NumericIndex.removeBatch (ix : NumericIndex) (entries : List (Value × Nat)) :
    NumericIndex × EncodedTuples :=
  ix.removeEncoded (pushAll entries)

/-- `RangeKind::is_empty` / `RangeIndex::is_empty` (`numeric.rs:466-468`): no tuple in either tree. -/
def NumericIndex.isEmpty (ix : NumericIndex) : Bool := tIsEmpty ix.tree && tIsEmpty ix.arrayTree

/-- `NumericIndex::empty` (`numeric.rs`, `range(1, 0)`): nothing. -/
def NumericIndex.empty (ix : NumericIndex) : List Nat := tRange ix.tree 1 0

/-- `KeySet` (`numeric.rs:58-72`): the keys a numeric predicate selects. A column looks them up to
    answer; a scan standing in for a column still building checks each entity's key against them. -/
inductive KeySet where
  | empty
  | point (k : Nat)
  | range (lo hi : Nat)
  | keys (ks : List Nat)
deriving Repr, DecidableEq

/-- `KeySet::distinct` (`numeric.rs:77-81`): `sort_unstable(); dedup()`. -/
def KeySet.distinct (keys : List Nat) : KeySet := .keys (sortDedup keys)

/-- `KeySet::contains` (`numeric.rs:85-92`). `Keys` is a `binary_search`, modelled here as plain list
    membership, which equals the search exactly when the list is sorted. The model does not carry that as a
    type invariant: it is the separate theorem `keysOf_sorted` (every `Keys` the column builds comes from
    `distinct`, so it is sorted), and a `KeySet.keys` built any other way is outside what these specs
    claim about the Rust (`binary_search` on an unsorted slice is unspecified). -/
def KeySet.contains : KeySet → Nat → Bool
  | .empty, _ => false
  | .point k, key => k == key
  | .range lo hi, key => decide (lo ≤ key ∧ key ≤ hi)
  | .keys ks, key => decide (key ∈ ks)

/-- `KeySet::of_keys` (`numeric.rs:143-151`): the set of `keys`, in any order and possibly repeated;
    none is `Empty`, one is a `Point`, more is `Keys`. -/
def KeySet.ofKeys (keys : List Nat) : KeySet :=
  match KeySet.distinct keys with
  | .keys [] => .empty
  | .keys [k] => .point k
  | ks => ks

/-- `KeySet::and` (`numeric.rs:173-199`), arm by arm: two windows meet in one window, a point set keeps
    the points the other side holds. -/
def KeySet.and : KeySet → KeySet → KeySet
  | .range a b, .range c d => if max a c ≤ min b d then .range (max a c) (min b d) else .empty
  | .point k, o => if o.contains k then .point k else .empty
  | o, .point k => if o.contains k then .point k else .empty
  | .keys ks, o => KeySet.ofKeys (ks.filter (fun k => o.contains k))
  | o, .keys ks => KeySet.ofKeys (ks.filter (fun k => o.contains k))
  | .empty, _ => .empty
  | _, .empty => .empty

/-- `point_keys` (`numeric.rs:385-387`). -/
def pointKeys (value : Value) : KeySet :=
  match encodeNumeric value with
  | some k => .point k
  | none => .empty

/-- `range` (`numeric.rs:389-441`, `range_keys`), branch for branch (`none` = an early `return
    KeySet::Empty`). -/
def rangeBounds (min max : Option Value) (incMin incMax : Bool) : Option (Nat × Nat) := do
  let lo ← match min with
    | none => some 0
    | some v => match encodeNumeric v with
      | none => none
      | some k => if incMin then some k else (if k + 1 < W then some (k + 1) else none)  -- checked_add
  let hi ← match max with
    | none => some (W - 1)
    | some v => match encodeNumeric v with
      | none => none
      | some k => if incMax then some k else (if 1 ≤ k then some (k - 1) else none)      -- checked_sub
  if lo > hi then none else some (lo, hi)

def rangeKeys (min max : Option Value) (incMin incMax : Bool) : KeySet :=
  match rangeBounds min max incMin incMax with
  | none => .empty
  | some (lo, hi) => .range lo hi

/-- `lookup` (`numeric.rs:367-383`): the entity ids under `keys`. `Keys` is the lazy chain of point
    cursors (`UnionIter`); as a set of ids it is the concatenation of the point scans. The chain is
    deliberately *not* id-ordered (`DocIter` doc): every theorem is about membership. The tree
    snapshot is pinned by `tree.clone()`; here the tree is an immutable value, which is that
    property. -/
def NumericIndex.lookup (ix : NumericIndex) : KeySet → List Nat
  | .empty => ix.empty
  | .point k => tRange ix.tree k k
  | .range lo hi => tRange ix.tree lo hi
  | .keys ks => ks.flatMap (fun k => tRange ix.tree k k)

/-- `point` (`numeric.rs:295-301`) = `lookup (point_keys value)`. -/
def NumericIndex.point (ix : NumericIndex) (v : Value) : List Nat := ix.lookup (pointKeys v)

/-- `point_key` (`numeric.rs:304-310`): the ids stored under one encoded key. -/
def NumericIndex.pointKey (ix : NumericIndex) (key : Nat) : List Nat := tRange ix.tree key key

/-- `range` (`numeric.rs:315-327`) = `lookup (range_keys ..)`. -/
def NumericIndex.range (ix : NumericIndex) (min max : Option Value) (incMin incMax : Bool) : List Nat :=
  ix.lookup (rangeKeys min max incMin incMax)

theorem point_eq (ix : NumericIndex) (v : Value) :
    ix.point v = match encodeNumeric v with | some k => tRange ix.tree k k | none => ix.empty := by
  unfold NumericIndex.point pointKeys; cases encodeNumeric v <;> rfl

theorem range_eq (ix : NumericIndex) (min max : Option Value) (incMin incMax : Bool) :
    ix.range min max incMin incMax = match rangeBounds min max incMin incMax with
      | none => ix.empty | some (lo, hi) => tRange ix.tree lo hi := by
  unfold NumericIndex.range rangeKeys; cases rangeBounds min max incMin incMax <;> rfl

/-- `IndexQuery<Value>` (`index/mod.rs`): every variant a column is asked. `inList`'s payload is
    `some items` for `list: Value::List(items)` and `none` for any other `Value`; `or`/`and` carry their
    children. -/
inductive IndexQuery where
  | equal (key : String) (value : Value)
  | range (key : String) (min max : Option Value) (incMin incMax : Bool)
  | inList (key : String) (list : Option (List Value))
  | arrayContains (key : String) (value : Value)
  | or (children : List IndexQuery)
  | and (children : List IndexQuery)
  | point (key : String)

/-! ## `Selection` (`range.rs:113-340`): what a column answers a predicate on one attribute with -/

/-- `Selection` (`range.rs:121-129`): scalars with a key in the set, or lists holding an element under
    `first` and under each of `rest`. -/
inductive Selection where
  | scalars (keys : KeySet)
  | listsHolding (first : Nat) (rest : List Nat)
deriving Repr, DecidableEq

/-- `Selection::empty` (`range.rs:133-135`). -/
def Selection.empty : Selection := .scalars .empty

/-- `Selection::is_empty` (`range.rs:139-141`): `matches!(self, Scalars(KeySet::Empty))`. -/
def Selection.isEmpty : Selection → Bool
  | .scalars .empty => true
  | _ => false

/-- `Kind::of` (`range.rs:105-107`): the numeric kind holds `Int`, `Float` and `Bool`; nothing else
    (`NULL`, a string, a point, a temporal, a list, ...) is a scalar any kind routes by. -/
def isNumericKind : Value → Bool
  | .int _ | .float _ | .bool _ => true
  | _ => false

/-- `Selection::equal` (`range.rs:188-194`): a numeric value selects its key (`Empty` for NaN); `NULL`
    matches nothing (answered empty); any other value is declined. -/
def Selection.equal (v : Value) : Option Selection :=
  if isNumericKind v then some (.scalars (pointKeys v))
  else if v == Value.null then some Selection.empty
  else none

/-- `Selection::push_equal_keys` (`range.rs:211-217`): the key `equal` selects for each value, a decline
    at a value no kind holds. The `unreachable!` arm (an equality selecting anything but one key or
    none) is modelled as a decline; `pushEqualKeys_spec` shows it never fires. -/
def pushEqualKeys : List Value → List Nat → Option (List Nat)
  | [], keys => some keys
  | v :: vs, keys =>
    match Selection.equal v with
    | none => none
    | some (.scalars (.point k)) => pushEqualKeys vs (keys ++ [k])
    | some (.scalars .empty) => pushEqualKeys vs keys
    | some _ => none

/-- `Selection::equal_to_any` (`range.rs:203-209`). -/
def Selection.equalToAny (values : List Value) : Option Selection :=
  (pushEqualKeys values []).map (fun keys => .scalars (KeySet.ofKeys keys))

/-- `Selection::range` (`range.rs:231-237`): a bound of a type no kind holds declines; a `NULL` bound
    answers empty; else the window. -/
def Selection.range (min max : Option Value) (incMin incMax : Bool) : Option Selection :=
  let bounds := (min.toList ++ max.toList)
  if bounds.any (fun b => !isNumericKind b && b != Value.null) then none
  else if bounds.any (fun b => b == Value.null) then some Selection.empty
  else some (.scalars (rangeKeys min max incMin incMax))

/-- `Selection::lists_holding` (`range.rs:256-262`): the lists holding a numeric `value`; a probe of any
    other type (a `NULL` included) is declined. -/
def Selection.listsHoldingOf (v : Value) : Option Selection :=
  if isNumericKind v then
    some ((encodeNumeric v).elim Selection.empty (fun first => .listsHolding first []))
  else none

/-! ### Attributes and atoms (`range.rs:356-399`) -/

mutual
/-- `attributes_of` (`range.rs:392-399`): `some` of the attributes a predicate constrains, flattening
    nested composites; `none` when the shape names no attribute (an empty `And`/`Or`). -/
def attrsOf : IndexQuery → Option (List String)
  | .equal k _ => some [k]
  | .range k .. => some [k]
  | .inList k _ => some [k]
  | .arrayContains k _ => some [k]
  | .point k => some [k]
  | .or cs => if cs.isEmpty then none else attrsOfAll cs
  | .and cs => if cs.isEmpty then none else attrsOfAll cs
/-- The `children.iter().all(|c| attributes_of(c, out))`. -/
def attrsOfAll : List IndexQuery → Option (List String)
  | [] => some []
  | c :: cs => (attrsOf c).bind (fun a => (attrsOfAll cs).map (fun r => a ++ r))
end

/-- `pair[0] == pair[1]` for every adjacent pair (`attributes.windows(2).all(..)`). -/
def allSame : List String → Bool
  | a :: b :: r => a == b && allSame (b :: r)
  | _ => true

/-- `on_one_attribute` (`range.rs:379-386`): every query constrains the same single attribute. -/
def onOneAttribute (qs : List IndexQuery) : Bool :=
  match (qs.mapM attrsOf) with
  | none => false
  | some ls => allSame ls.flatten

/-- Whether `m` is the combinator `isAnd` selects (an `And` for an `And`, an `Or` for an `Or`). -/
def sameCombinator (isAnd : Bool) : IndexQuery → Bool
  | .and _ => isAnd
  | .or _ => !isAnd
  | _ => false

mutual
/-- `atoms` (`range.rs:356-363`): the members of an `AND` or `OR`, each member that is the same
    combinator opened into its own; any other predicate is its own only atom; `none` when the
    combinator or one opened in it has no members. -/
def atoms : IndexQuery → Option (List IndexQuery)
  | .and ms => if ms.isEmpty then none else atomsGo true ms
  | .or ms => if ms.isEmpty then none else atomsGo false ms
  | q => some [q]
/-- The loop of `atoms`. -/
def atomsGo (isAnd : Bool) : List IndexQuery → Option (List IndexQuery)
  | [] => some []
  | m :: ms =>
    if sameCombinator isAnd m then
      (atoms m).bind (fun a => (atomsGo isAnd ms).map (fun r => a ++ r))
    else (atomsGo isAnd ms).map (fun r => m :: r)
end

/-! ### Unions and conjunctions -/

/-- The values one member of a union contributes (`range.rs:276-283`): an `Equal`'s, or a **non-empty**
    `IN` list's; any other member declines the union. -/
def disjMemberValues : IndexQuery → Option (List Value)
  | .equal _ v => some [v]
  | .inList _ (some items) => if items.isEmpty then none else some items
  | _ => none

/-- The loop of `disjunction` (`range.rs:276-283`). -/
def disjGo : List IndexQuery → List Nat → Option (List Nat)
  | [], keys => some keys
  | m :: ms, keys =>
    match disjMemberValues m with
    | none => none
    | some vs => (pushEqualKeys vs keys).bind (disjGo ms)

/-- `Selection::disjunction` (`range.rs:276-283`): an `OR` of `=`s and `IN` lists, all on one attribute. -/
def Selection.disjunction (members : List IndexQuery) : Option Selection :=
  if !onOneAttribute members then none
  else (disjGo members []).map (fun keys => .scalars (KeySet.ofKeys keys))

/-- `Selection::and` (`range.rs:329-336`): what both select. A value is a scalar or a list, never
    both, so a scalar selection and a list selection select nothing together. -/
def Selection.and : Selection → Selection → Selection
  | .scalars a, .scalars b => .scalars (a.and b)
  | .listsHolding f r, .listsHolding f' r' => .listsHolding f (r ++ [f'] ++ r')
  | _, _ => Selection.empty

/-- The selection of one query that is not a combinator: `Selection::of` (`range.rs:160-181`) on a
    leaf. -/
def Selection.ofLeaf : IndexQuery → Option Selection
  | .equal _ v => Selection.equal v
  | .range _ min max a b => Selection.range min max a b
  -- An empty list is declined rather than answered as "no rows" (`range.rs:169-172`).
  | .inList _ (some items) => if items.isEmpty then none else Selection.equalToAny items
  | .inList _ none => none
  | .arrayContains _ v => Selection.listsHoldingOf v
  | .point _ => none
  | .or _ => none
  | .and _ => none

/-- `Selection::of` on a conjunct, as `conjunction` calls it. A conjunct is an atom — no conjunct is
    itself an `AND` (`debug_assert!`, `range.rs:307-312`; `atoms_and_no_and` shows it never is), so
    that arm is modelled as a decline. An `OR` conjunct is its own disjunction. -/
def Selection.ofAtom : IndexQuery → Option Selection
  | .or ms => (atoms (.or ms)).bind Selection.disjunction
  | .and _ => none
  | q => Selection.ofLeaf q

/-- `Selection::conjunction` (`range.rs:307-314`): on one attribute, what every conjunct selects. A
    conjunct selecting nothing empties the conjunction, even beside one no kind serves; otherwise a
    conjunct no kind serves declines it. -/
def Selection.conjunction (cs : List IndexQuery) : Option Selection :=
  if !onOneAttribute cs then none
  else
    let sels := cs.map Selection.ofAtom
    if sels.any (fun s => match s with | some s => s.isEmpty | none => false) then some Selection.empty
    else
      match sels.mapM (fun s => s) with
      | none => none
      | some [] => none
      | some (s :: ss) => some (ss.foldl Selection.and s)

/-- `Selection::of` (`range.rs:160-181`): the selection of a predicate on **one** attribute, or `none`
    when no kind serves it. -/
def Selection.of (q : IndexQuery) : Option Selection :=
  match q with
  | .or _ => (atoms q).bind Selection.disjunction
  | .and _ => (atoms q).bind Selection.conjunction
  | _ => Selection.ofLeaf q

/-- `Selection::matches` (`range.rs:146-148`) given the encoder's keys: the scan's check while the
    column builds. -/
def stores (v : Value) (kind : KeyKind) (wanted : Nat → Bool) : Bool :=
  (forEachStoredKey v).any (fun kk => decide (kk.1 = kind) && wanted kk.2)

/-- `NumericIndex::is_scalar_in` (`numeric.rs:507-512`). -/
def isScalarIn (keys : KeySet) (v : Value) : Bool := stores v .scalar keys.contains

/-- `NumericIndex::is_list_holding` (`numeric.rs:514-519`). -/
def isListHolding (key : Nat) (v : Value) : Bool := stores v .array (fun k => k == key)

/-- `Selection::matches`. -/
def Selection.matches : Selection → Value → Bool
  | .scalars ks, v => isScalarIn ks v
  | .listsHolding first rest, v => (first :: rest).all (fun k => isListHolding k v)

/-- `NumericIndex::lists_holding` (`numeric.rs:487-495`): a point lookup into the **array** tree; the
    scalar tree is deliberately not consulted. -/
def NumericIndex.listsHolding (ix : NumericIndex) (key : Nat) : List Nat := tRange ix.arrayTree key key

/-- `RangeIndex::lookup`, the `ListsHolding` arm (`range.rs:582-606`): the first probe's lists, each
    later probe keeping only those it also reaches (a set probe: a probe yields docs in key order). -/
def NumericIndex.lookupLists (ix : NumericIndex) (first : Nat) (rest : List Nat) : List Nat :=
  rest.foldl (fun found e =>
    if found.isEmpty then found else (ix.listsHolding e).filter (fun d => found.contains d))
    (ix.listsHolding first)

/-- `RangeIndex::lookup` (`range.rs:577-606`): the entities a selection selects. -/
def NumericIndex.lookupSel (ix : NumericIndex) : Selection → List Nat
  | .scalars ks => ix.lookup ks
  | .listsHolding first rest => ix.lookupLists first rest

/-! ## Keys -/

/-- The keys a non-NaN numeric value can take. -/
def StoredKey (k : Nat) : Prop := 0 < k ∧ k < W - 1

theorem encodeNumeric_some (v : Value) (hv : v.WF) (k : Nat) (h : encodeNumeric v = some k) : StoredKey k := by
  cases v <;> simp [encodeNumeric] at h
  · rename_i b
    obtain ⟨hn, rfl⟩ := h
    exact encodeF64_bounds _ (by cases b <;> decide) (by simpa using hn)
  · rename_i i
    obtain ⟨hn, rfl⟩ := h
    exact encodeF64_bounds _ (i64ToF64_lt i hv).1 (by simpa using hn)
  · rename_i x
    obtain ⟨hn, rfl⟩ := h
    exact encodeF64_bounds _ hv (by simpa using hn)

/-- The bound condition a range query imposes on a key. -/
def boundOK (kmin kmax : Option Nat) (incMin incMax : Bool) (k : Nat) : Bool :=
  (match kmin with | none => true | some m => if incMin then m ≤ k else m < k) &&
  (match kmax with | none => true | some m => if incMax then k ≤ m else k < m)

def loOf (kmin : Option Nat) (incMin : Bool) : Nat :=
  match kmin with | none => 0 | some a => if incMin then a else a + 1
def hiOf (kmax : Option Nat) (incMax : Bool) : Nat :=
  match kmax with | none => W - 1 | some b => if incMax then b else b - 1

theorem rangeBounds_eq (min max : Option Value) (incMin incMax : Bool) (kmin kmax : Option Nat)
    (hmin : min.map encodeNumeric = kmin.map some) (hmax : max.map encodeNumeric = kmax.map some)
    (hwmin : ∀ k, kmin = some k → StoredKey k) (hwmax : ∀ k, kmax = some k → StoredKey k) :
    rangeBounds min max incMin incMax =
      if loOf kmin incMin > hiOf kmax incMax then none else some (loOf kmin incMin, hiOf kmax incMax) := by
  rcases min with _ | vmin <;> rcases kmin with _ | a <;> simp at hmin <;>
  rcases max with _ | vmax <;> rcases kmax with _ | b <;> simp at hmax
  all_goals simp only [rangeBounds, hmin, hmax, loOf, hiOf]
  all_goals first
    | rfl
    | (have ha := hwmin _ rfl; have hb := hwmax _ rfl; unfold StoredKey at ha hb
       have e1 : a + 1 < W := by omega
       have e2 : 1 ≤ b := by omega
       cases incMin <;> cases incMax <;> simp [e1, e2])
    | (have ha := hwmin _ rfl; unfold StoredKey at ha
       have e1 : a + 1 < W := by omega
       cases incMin <;> simp [e1])
    | (have hb := hwmax _ rfl; unfold StoredKey at hb
       have e2 : 1 ≤ b := by omega
       cases incMax <;> simp [e2])

theorem boundOK_iff (kmin kmax : Option Nat) (incMin incMax : Bool) (k : Nat) (hk : StoredKey k) :
    boundOK kmin kmax incMin incMax k = decide (loOf kmin incMin ≤ k ∧ k ≤ hiOf kmax incMax) := by
  unfold StoredKey at hk
  rcases kmin with _ | a <;> rcases kmax with _ | b <;> (try cases incMin) <;> (try cases incMax) <;>
    simp only [boundOK, loOf, hiOf, Bool.true_and, Bool.and_true, ite_true, Bool.false_eq_true, ite_false] <;>
    (apply Bool.eq_iff_iff.2) <;> (try simp only [decide_eq_true_eq, Bool.and_eq_true]) <;> (try constructor) <;> (try intro) <;> first | rfl | trivial | omega | exact decide_eq_true (by omega)

/-- **The key bounds are exact**: for any stored key, `lo ≤ k ≤ hi` iff the value bounds hold;
    when `range` returns early, no stored key satisfies them. -/
theorem range_keys (min max : Option Value) (incMin incMax : Bool) (kmin kmax : Option Nat)
    (hmin : min.map encodeNumeric = kmin.map some) (hmax : max.map encodeNumeric = kmax.map some)
    (hwmin : ∀ k, kmin = some k → StoredKey k) (hwmax : ∀ k, kmax = some k → StoredKey k)
    (k : Nat) (hk : StoredKey k) :
    (match rangeBounds min max incMin incMax with
      | none => false
      | some (lo, hi) => decide (lo ≤ k ∧ k ≤ hi)) = boundOK kmin kmax incMin incMax k := by
  rw [rangeBounds_eq min max incMin incMax kmin kmax hmin hmax hwmin hwmax, boundOK_iff _ _ _ _ k hk]
  by_cases hc : loOf kmin incMin > hiOf kmax incMax
  · simp only [hc, ite_true]; symm; rw [decide_eq_false_iff_not]; omega
  · simp only [hc, ite_false]

/-! ## Index scan ≡ label scan + filter -/

theorem goodBound_enc (b : Value) (hb : GoodBound b) : ∃ kb, encodeNumeric b = some kb ∧ StoredKey kb := by
  cases b <;> simp [GoodBound] at hb
  · rename_i j
    have ⟨hW, hN⟩ := i64ToF64_lt j hb.1
    exact ⟨_, by simp [encodeNumeric_int, hN], encodeF64_bounds _ hW hN⟩
  · rename_i f
    exact ⟨_, by simp [encodeNumeric_float, hb.2], encodeF64_bounds _ hb.1 hb.2⟩

theorem mem_encodePairs (entries : List (Value × Nat)) (k id : Nat) (hid : id < W)
    (hids : ∀ p ∈ entries, p.2 < W) :
    enc k id ∈ encodePairs entries ↔ ∃ v, (v, id) ∈ entries ∧ encodeNumeric v = some k := by
  simp only [encodePairs, List.mem_filterMap, Option.map_eq_some_iff]
  constructor
  · rintro ⟨⟨v, id'⟩, hm, k', hk, he⟩
    have hid' := hids _ hm
    have h1 := congrArg keyOf he; have h2 := congrArg docOf he
    rw [keyOf_enc _ _ hid', keyOf_enc _ _ hid] at h1
    rw [docOf_enc _ _ hid', docOf_enc _ _ hid] at h2
    subst h1 h2; exact ⟨v, hm, hk⟩
  · rintro ⟨v, hm, hk⟩
    exact ⟨(v, id), hm, k, hk, rfl⟩

/-- The value-level condition of a `Range` query: each present bound passes the Filter. -/
def rangeFilter (v : Value) (min max : Option Value) (incMin incMax : Bool) : Prop :=
  (∀ b, min = some b → cmpPred (if incMin then .ge else .gt) v b = some true) ∧
  (∀ b, max = some b → cmpPred (if incMax then .le else .lt) v b = some true)

/-- Per value: the key bound check is the Filter. -/
theorem boundOK_filter (v : Value) (hv : v.WF) (hnb : NotBool v) (k : Nat) (hk : encodeNumeric v = some k)
    (min max : Option Value) (incMin incMax : Bool) (kmin kmax : Option Nat)
    (hmin : ∀ b, min = some b → GoodBound b) (hmax : ∀ b, max = some b → GoodBound b)
    (ekmin : min.map encodeNumeric = kmin.map some) (ekmax : max.map encodeNumeric = kmax.map some) :
    boundOK kmin kmax incMin incMax k = true ↔ rangeFilter v min max incMin incMax := by
  have side : ∀ (op : Op) (b : Value), GoodBound b → ∀ kb, encodeNumeric b = some kb →
      (cmpPred op v b = some true ↔ keyPred op k kb = true) := by
    intro op b hb kb hkb
    rw [filter_iff_key op v b hv hnb hb kb hkb, hk]; simp
  unfold rangeFilter boundOK
  rcases min with _ | bmin <;> rcases kmin with _ | a <;> simp at ekmin <;>
  rcases max with _ | bmax <;> rcases kmax with _ | c <;> simp at ekmax
  all_goals simp only [Bool.true_and, Bool.and_true, Bool.and_eq_true, reduceCtorEq, false_implies,
    implies_true, and_true, true_and, forall_eq', Option.some.injEq]
  all_goals first
    | (rw [side _ _ (hmin _ rfl) _ ekmin, side _ _ (hmax _ rfl) _ ekmax]
       cases incMin <;> cases incMax <;> simp [keyPred])
    | (rw [side _ _ (hmax _ rfl) _ ekmax]; cases incMax <;> simp [keyPred])
    | (rw [side _ _ (hmin _ rfl) _ ekmin]; cases incMin <;> simp [keyPred])
    | (constructor <;> intro <;> trivial)

/-- **Range scan ≡ label scan + filter.** A column built by `from_entries` over a label's `(value, id)`
    pairs (no `Bool` values) answers a `Range` with at least one guarded bound with exactly the ids
    whose value passes the Filter. -/
theorem range_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (min max : Option Value) (incMin incMax : Bool)
    (hmin : ∀ b, min = some b → GoodBound b) (hmax : ∀ b, max = some b → GoodBound b)
    (hsome : min.isSome ∨ max.isSome) (id : Nat) (hid : id < W) :
    id ∈ (NumericIndex.fromEntries entries).range min max incMin incMax ↔
      ∃ v, (v, id) ∈ entries ∧ rangeFilter v min max incMin incMax := by
  -- the bounds' keys
  let kmin : Option Nat := min.bind encodeNumeric
  let kmax : Option Nat := max.bind encodeNumeric
  have ekmin : min.map encodeNumeric = kmin.map some := by
    rcases min with _ | b
    · rfl
    · obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hmin b rfl); simp [kmin, hkb]
  have ekmax : max.map encodeNumeric = kmax.map some := by
    rcases max with _ | b
    · rfl
    · obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hmax b rfl); simp [kmax, hkb]
  have wmin : ∀ k, kmin = some k → StoredKey k := by
    intro k hk; rcases min with _ | b
    · simp [kmin] at hk
    · obtain ⟨kb, hkb, hs⟩ := goodBound_enc b (hmin b rfl); simp [kmin, hkb] at hk; subst hk; exact hs
  have wmax : ∀ k, kmax = some k → StoredKey k := by
    intro k hk; rcases max with _ | b
    · simp [kmax] at hk
    · obtain ⟨kb, hkb, hs⟩ := goodBound_enc b (hmax b rfl); simp [kmax, hkb] at hk; subst hk; exact hs
  have hids : ∀ p ∈ entries, p.2 < W := fun p hp => (hwf p hp).2.2
  -- the scan
  have scan : id ∈ (NumericIndex.fromEntries entries).range min max incMin incMax ↔
      ∃ k, enc k id ∈ encodePairs entries ∧ StoredKey k ∧ boundOK kmin kmax incMin incMax k = true := by
    simp only [range_eq, NumericIndex.fromEntries, NumericIndex.empty, tFromSorted]
    have keyEq := range_keys min max incMin incMax kmin kmax ekmin ekmax wmin wmax
    have memT : ∀ k, enc k id ∈ sortDedup (encodePairs entries) ↔ enc k id ∈ encodePairs entries :=
      fun k => (sortDedup_spec _).2 _
    have stored : ∀ k, enc k id ∈ encodePairs entries → StoredKey k := by
      intro k hk
      obtain ⟨v, hm, he⟩ := (mem_encodePairs entries k id hid hids).1 hk
      exact encodeNumeric_some v (hwf _ hm).1 k he
    split
    · rename_i hrb
      rw [tRange_empty]; simp only [List.not_mem_nil, false_iff, not_exists, not_and]
      intro k hk hs hb
      have := keyEq k hs; rw [hrb] at this; simp at this; rw [hb] at this; exact absurd this (by simp)
    · rename_i lo hi hrb
      rw [mem_tRange _ _ _ _ hid]
      constructor
      · rintro ⟨k, hk, h1, h2⟩
        have hk' := (memT k).1 hk
        have hs := stored k hk'
        refine ⟨k, hk', hs, ?_⟩
        rw [← keyEq k hs, hrb]; simp [h1, h2]
      · rintro ⟨k, hk, hs, hb⟩
        have := keyEq k hs; rw [hrb, hb] at this; simp at this
        exact ⟨k, (memT k).2 hk, this.1, this.2⟩
  rw [scan]
  constructor
  · rintro ⟨k, hk, _, hb⟩
    obtain ⟨v, hm, he⟩ := (mem_encodePairs entries k id hid hids).1 hk
    obtain ⟨hv, hnb, _⟩ := hwf _ hm
    exact ⟨v, hm, (boundOK_filter v hv hnb k he min max incMin incMax kmin kmax hmin hmax ekmin ekmax).1 hb⟩
  · rintro ⟨v, hm, hf⟩
    obtain ⟨hv, hnb, _⟩ := hwf _ hm
    -- a present bound forces `v` to be indexed
    have : ∃ k, encodeNumeric v = some k := by
      rcases hsome with h | h
      · rcases min with _ | b
        · simp at h
        · obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hmin b rfl)
          obtain ⟨k, hk, _⟩ := (filter_iff_key _ v b hv hnb (hmin b rfl) kb hkb).1 (hf.1 b rfl)
          exact ⟨k, hk⟩
      · rcases max with _ | b
        · simp at h
        · obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hmax b rfl)
          obtain ⟨k, hk, _⟩ := (filter_iff_key _ v b hv hnb (hmax b rfl) kb hkb).1 (hf.2 b rfl)
          exact ⟨k, hk⟩
    obtain ⟨k, hk⟩ := this
    exact ⟨k, (mem_encodePairs entries k id hid hids).2 ⟨v, hm, hk⟩, encodeNumeric_some v hv k hk,
      (boundOK_filter v hv hnb k hk min max incMin incMax kmin kmax hmin hmax ekmin ekmax).2 hf⟩

/-- **Point scan ≡ label scan + `=` filter.** -/
theorem point_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (b : Value) (hb : GoodBound b) (id : Nat) (hid : id < W) :
    id ∈ (NumericIndex.fromEntries entries).point b ↔
      ∃ v, (v, id) ∈ entries ∧ cmpPred .eq v b = some true := by
  obtain ⟨kb, hkb, _⟩ := goodBound_enc b hb
  have hids : ∀ p ∈ entries, p.2 < W := fun p hp => (hwf p hp).2.2
  simp only [point_eq, hkb, NumericIndex.fromEntries, tFromSorted]
  rw [mem_tRange _ _ _ _ hid]
  constructor
  · rintro ⟨k, hk, h1, h2⟩
    have hk' := ((sortDedup_spec _).2 _).1 hk
    obtain ⟨v, hm, he⟩ := (mem_encodePairs entries k id hid hids).1 hk'
    obtain ⟨hv, hnb, _⟩ := hwf _ hm
    refine ⟨v, hm, (filter_iff_key .eq v b hv hnb hb kb hkb).2 ⟨k, he, ?_⟩⟩
    simp [keyPred]; omega
  · rintro ⟨v, hm, hf⟩
    obtain ⟨hv, hnb, _⟩ := hwf _ hm
    obtain ⟨k, he, hkp⟩ := (filter_iff_key .eq v b hv hnb hb kb hkb).1 hf
    simp [keyPred] at hkp; subst hkp
    exact ⟨k, ((sortDedup_spec _).2 _).2 ((mem_encodePairs entries k id hid hids).2 ⟨v, hm, he⟩),
      Nat.le_refl _, Nat.le_refl _⟩

/-- A `Keys` lookup finds the ids under any of its keys. -/
theorem lookup_keys_mem (ix : NumericIndex) (ks : List Nat) (id : Nat) :
    id ∈ ix.lookup (.keys ks) ↔ ∃ k ∈ ks, id ∈ tRange ix.tree k k := by
  simp only [NumericIndex.lookup, List.mem_flatMap]

/-- `KeySet::distinct` keeps the keys' membership and sorts them (so `contains`' `binary_search` is
    membership). -/
theorem distinct_mem (keys : List Nat) (k : Nat) :
    (∃ ks, KeySet.distinct keys = .keys ks ∧ Sorted ks) ∧
    (KeySet.distinct keys).contains k = decide (k ∈ keys) := by
  refine ⟨⟨sortDedup keys, rfl, (sortDedup_spec _).1⟩, ?_⟩
  simp only [KeySet.distinct, KeySet.contains, decide_eq_decide, (sortDedup_spec _).2]

theorem lookup_distinct_mem (ix : NumericIndex) (keys : List Nat) (id : Nat) :
    id ∈ ix.lookup (KeySet.distinct keys) ↔ ∃ k ∈ keys, id ∈ tRange ix.tree k k := by
  simp only [KeySet.distinct, lookup_keys_mem, (sortDedup_spec _).2]

/-! ## `IN` lists, unions and conjunctions (`Selection`, `range.rs`)

`n.a IN [v1, .., vn]` reaches the column as one `InList`, `a = v1 OR a IN [..]` as an `Or`, a conjunction
on one attribute as an `And`, and `x IN n.a` as an `ArrayContains`. Each becomes a `Selection` built from
the query alone (`Selection::of`), which a `Ready` column looks up and a `Building` one hands to a scan.
Proven: a served `IN` / `OR` is exactly the ids a label scan plus the filter returns
(`equalToAny_agrees`, `disjunction_agrees`); the column *declines* rather than answers part of a query
(`equalToAny_isSome_iff`, `disjunction_attr_mismatch`, `disjunction_range_declines`, `of_or_nil`), because
the scan op does not re-check the predicate; and every set is the same whatever the order (all statements
are membership). -/

/-- `Selection::equal`, on a numeric value. -/
theorem equal_numeric (v : Value) (h : isNumericKind v = true) :
    Selection.equal v = some (.scalars (pointKeys v)) := by
  simp [Selection.equal, h]

theorem equal_null : Selection.equal .null = some Selection.empty := by
  simp [Selection.equal, isNumericKind]

theorem equal_other (v : Value) (h1 : isNumericKind v = false) (h2 : v ≠ .null) : Selection.equal v = none := by
  simp [Selection.equal, h1, h2]

/-- **`push_equal_keys`**: it declines exactly at a value no kind holds (neither numeric nor `NULL`); else
    it appends the key of each value that encodes (`NULL` and NaN add none), and the `unreachable!` arm
    never fires. -/
theorem pushEqualKeys_spec (vs : List Value) : ∀ (keys out : List Nat),
    pushEqualKeys vs keys = some out ↔
      (∀ v ∈ vs, isNumericKind v = true ∨ v = .null) ∧ out = keys ++ vs.filterMap encodeNumeric := by
  induction vs with
  | nil => intro keys out; simp [pushEqualKeys]; exact eq_comm
  | cons v vs ih =>
    intro keys out
    by_cases hn : isNumericKind v = true
    · rw [pushEqualKeys, equal_numeric v hn]
      cases he : encodeNumeric v with
      | none =>
        have : pointKeys v = .empty := by simp [pointKeys, he]
        simp only [this, ih, List.mem_cons, forall_eq_or_imp, List.filterMap_cons, he]
        simp [hn]
      | some k =>
        have : pointKeys v = .point k := by simp [pointKeys, he]
        simp only [this, ih, List.mem_cons, forall_eq_or_imp, List.filterMap_cons, he]
        simp [hn, List.append_assoc]
    · have hn' : isNumericKind v = false := by simpa using hn
      by_cases hv : v = .null
      · subst hv
        rw [pushEqualKeys, equal_null]
        simp only [Selection.empty, ih, List.mem_cons, forall_eq_or_imp, List.filterMap_cons]
        simp [encodeNumeric, isNumericKind]
      · rw [pushEqualKeys, equal_other v hn' hv]
        simp [hn, hv]

theorem equalToAny_isSome_iff (items : List Value) :
    (Selection.equalToAny items).isSome = true ↔ ∀ v ∈ items, isNumericKind v = true ∨ v = .null := by
  unfold Selection.equalToAny
  rw [Option.isSome_map]
  cases h : pushEqualKeys items [] with
  | none =>
    simp only [Option.isSome_none, Bool.false_eq_true, false_iff]
    intro hall
    have := (pushEqualKeys_spec items [] (items.filterMap encodeNumeric)).2 ⟨hall, by simp⟩
    rw [h] at this; cases this
  | some out =>
    simp only [Option.isSome_some, true_iff]
    exact ((pushEqualKeys_spec items [] out).1 h).1

theorem ofKeys_contains (keys : List Nat) (k : Nat) : (KeySet.ofKeys keys).contains k = decide (k ∈ keys) := by
  have h := (distinct_mem keys k).2
  unfold KeySet.ofKeys
  unfold KeySet.distinct at h ⊢
  generalize hl : sortDedup keys = l at h ⊢
  match l with
  | [] => simpa [KeySet.contains] using h
  | [a] =>
    have h' : k = a ↔ k ∈ keys := by simpa [KeySet.contains] using h
    simp only [KeySet.contains]
    by_cases hk : k ∈ keys
    · have := h'.2 hk; subst this; simp [hk]
    · have : k ≠ a := fun e => hk (h'.1 e)
      simp [hk, Ne.symm this]
  | _ :: _ :: _ => exact h

theorem lookup_ofKeys (ix : NumericIndex) (keys : List Nat) :
    ix.lookup (KeySet.ofKeys keys) = ix.lookup (KeySet.distinct keys) := by
  unfold KeySet.ofKeys KeySet.distinct
  generalize sortDedup keys = l
  match l with
  | [] => simp [NumericIndex.lookup, NumericIndex.empty, tRange_empty]
  | [a] => simp [NumericIndex.lookup]
  | _ :: _ :: _ => rfl

theorem lookup_ofKeys_mem (ix : NumericIndex) (keys : List Nat) (id : Nat) :
    id ∈ ix.lookup (KeySet.ofKeys keys) ↔ ∃ k ∈ keys, id ∈ tRange ix.tree k k := by
  rw [lookup_ofKeys, lookup_distinct_mem]

theorem cmpPred_eq_null (d : Value) : cmpPred .eq d .null = none := by
  cases d <;> simp [cmpPred, compareValue]

/-- **`IN` list ≡ label scan + `IN` filter**: a served `IN [b1, .., bn]` (every `bi` a guarded non-NaN
    numeric, or `NULL`, which matches nothing) returns exactly the ids whose value equals some member. -/
theorem equalToAny_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (items : List Value) (hgood : ∀ b ∈ items, GoodBound b ∨ b = .null) (sel : Selection)
    (hr : Selection.equalToAny items = some sel) (id : Nat) (hid : id < W) :
    id ∈ (NumericIndex.fromEntries entries).lookupSel sel ↔
      ∃ v, (v, id) ∈ entries ∧ ∃ b ∈ items, cmpPred .eq v b = some true := by
  unfold Selection.equalToAny at hr
  cases hp : pushEqualKeys items [] with
  | none => simp [hp] at hr
  | some keys =>
    simp only [hp, Option.map_some, Option.some.injEq] at hr
    subst hr
    obtain ⟨hk, hkeys⟩ := (pushEqualKeys_spec items [] keys).1 hp
    simp only [NumericIndex.lookupSel, lookup_ofKeys_mem]
    rw [hkeys]
    constructor
    · rintro ⟨k, hk', hmem⟩
      simp only [List.nil_append, List.mem_filterMap] at hk'
      obtain ⟨b, hb, hkb⟩ := hk'
      have hgb : GoodBound b := by
        rcases hgood b hb with h | h
        · exact h
        · subst h; simp [encodeNumeric] at hkb
      have hpt : id ∈ (NumericIndex.fromEntries entries).point b := by
        simp only [point_eq, hkb]; exact hmem
      obtain ⟨v, hv, hc⟩ := (point_agrees entries hwf b hgb id hid).1 hpt
      exact ⟨v, hv, b, hb, hc⟩
    · rintro ⟨v, hv, b, hb, hc⟩
      have hgb : GoodBound b := by
        rcases hgood b hb with h | h
        · exact h
        · subst h; rw [cmpPred_eq_null] at hc; cases hc
      have hp' := (point_agrees entries hwf b hgb id hid).2 ⟨v, hv, hc⟩
      obtain ⟨kb, hkb, _⟩ := goodBound_enc b hgb
      simp only [point_eq, hkb] at hp'
      exact ⟨kb, by simp only [List.nil_append, List.mem_filterMap]; exact ⟨b, hb, hkb⟩, hp'⟩

/-- The values a union's members contribute. -/
def memberValues (cs : List IndexQuery) : List Value :=
  cs.flatMap (fun c => (disjMemberValues c).getD [])

theorem disjGo_spec (ms : List IndexQuery) : ∀ (keys out : List Nat),
    disjGo ms keys = some out ↔
      (∀ m ∈ ms, (disjMemberValues m).isSome = true) ∧
      (∀ v ∈ memberValues ms, isNumericKind v = true ∨ v = .null) ∧
      out = keys ++ (memberValues ms).filterMap encodeNumeric := by
  induction ms with
  | nil => intro keys out; simp [disjGo, memberValues]; exact eq_comm
  | cons m ms ih =>
    intro keys out
    cases hm : disjMemberValues m with
    | none => simp [disjGo, hm]
    | some vs =>
      simp only [disjGo, hm, Option.bind_eq_bind]
      have e1 : (pushEqualKeys vs keys).bind (disjGo ms) = some out ↔
          ∃ k1, pushEqualKeys vs keys = some k1 ∧ disjGo ms k1 = some out := by
        cases pushEqualKeys vs keys <;> simp
      rw [e1]
      constructor
      · rintro ⟨k1, h1, h2⟩
        obtain ⟨hv1, hk1⟩ := (pushEqualKeys_spec vs keys k1).1 h1
        obtain ⟨a, b, c⟩ := (ih k1 out).1 h2
        refine ⟨?_, ?_, ?_⟩
        · intro m' hm'
          rcases List.mem_cons.1 hm' with rfl | hm'
          · simp [hm]
          · exact a m' hm'
        · intro v hv
          simp only [memberValues, List.flatMap_cons, hm, Option.getD_some, List.mem_append] at hv
          rcases hv with hv | hv
          · exact hv1 v hv
          · exact b v (by simpa [memberValues] using hv)
        · simp only [memberValues, List.flatMap_cons, hm, Option.getD_some, List.filterMap_append]
          rw [c, hk1]; simp [memberValues, List.append_assoc]
      · rintro ⟨a, b, c⟩
        simp only [memberValues, List.flatMap_cons, hm, Option.getD_some, List.mem_append] at b c
        have hv1 : ∀ v ∈ vs, isNumericKind v = true ∨ v = .null := fun v hv => b v (Or.inl hv)
        refine ⟨keys ++ vs.filterMap encodeNumeric, (pushEqualKeys_spec vs keys _).2 ⟨hv1, rfl⟩, ?_⟩
        refine (ih _ out).2 ⟨fun m' hm' => a m' (List.mem_cons_of_mem _ hm'), ?_, ?_⟩
        · intro v hv; exact b v (Or.inr hv)
        · rw [c]; simp [List.filterMap_append, List.append_assoc, memberValues]

/-- **`OR` of equalities / `IN` lists ≡ label scan + `OR` filter**: a served disjunction names one
    attribute and returns exactly the ids whose value equals some member of some child. -/
theorem disjunction_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (cs : List IndexQuery) (hgood : ∀ b ∈ memberValues cs, GoodBound b ∨ b = .null) (sel : Selection)
    (hr : Selection.disjunction cs = some sel) (id : Nat) (hid : id < W) :
    onOneAttribute cs = true ∧
    (id ∈ (NumericIndex.fromEntries entries).lookupSel sel ↔
      ∃ v, (v, id) ∈ entries ∧ ∃ b ∈ memberValues cs, cmpPred .eq v b = some true) := by
  unfold Selection.disjunction at hr
  by_cases hone : onOneAttribute cs = true
  · simp only [hone, Bool.not_true, Bool.false_eq_true, ite_false] at hr
    cases hg : disjGo cs [] with
    | none => simp [hg] at hr
    | some keys =>
      simp only [hg, Option.map_some, Option.some.injEq] at hr
      subst hr
      obtain ⟨_, hk, hkeys⟩ := (disjGo_spec cs [] keys).1 hg
      refine ⟨hone, ?_⟩
      simp only [NumericIndex.lookupSel, lookup_ofKeys_mem]
      rw [hkeys]
      constructor
      · rintro ⟨k, hk', hmem⟩
        simp only [List.nil_append, List.mem_filterMap] at hk'
        obtain ⟨b, hb, hkb⟩ := hk'
        have hgb : GoodBound b := by
          rcases hgood b hb with h | h
          · exact h
          · subst h; simp [encodeNumeric] at hkb
        have hpt : id ∈ (NumericIndex.fromEntries entries).point b := by
          simp only [point_eq, hkb]; exact hmem
        obtain ⟨v, hv, hc⟩ := (point_agrees entries hwf b hgb id hid).1 hpt
        exact ⟨v, hv, b, hb, hc⟩
      · rintro ⟨v, hv, b, hb, hc⟩
        have hgb : GoodBound b := by
          rcases hgood b hb with h | h
          · exact h
          · subst h; rw [cmpPred_eq_null] at hc; cases hc
        have hp' := (point_agrees entries hwf b hgb id hid).2 ⟨v, hv, hc⟩
        obtain ⟨kb, hkb, _⟩ := goodBound_enc b hgb
        simp only [point_eq, hkb] at hp'
        exact ⟨kb, by simp only [List.nil_append, List.mem_filterMap]; exact ⟨b, hb, hkb⟩, hp'⟩
  · simp [hone] at hr

/-- An `OR` spanning two attributes cannot be answered from one column: it is declined. -/
theorem disjunction_attr_mismatch (a b : String) (hab : a ≠ b) (v w : Value) :
    Selection.disjunction [.equal a v, .equal b w] = none := by
  have : onOneAttribute [.equal a v, .equal b w] = false := by
    simp [onOneAttribute, attrsOf, allSame, hab]
  simp [Selection.disjunction, this]

/-- A range (or any other child) is not a union member: the whole `OR` is declined. -/
theorem disjunction_range_declines (k : String) (mn mx : Option Value) (a b : Bool)
    (rest : List IndexQuery) : Selection.disjunction (.range k mn mx a b :: rest) = none := by
  unfold Selection.disjunction
  split <;> simp [disjGo, disjMemberValues]

/-- An empty `OR` or `AND` is not match-all: `Selection::of` declines it (`atoms` is `None`). -/
theorem of_or_nil : Selection.of (.or []) = none ∧ Selection.of (.and []) = none := by
  simp [Selection.of, atoms]

/-- `a = null` holds for nothing, so it adds no key (#2386): an `OR` of nulls is served, with no
    rows, rather than declined. -/
theorem disjunction_null_only (k : String) :
    Selection.disjunction [.equal k .null] = some Selection.empty := by
  simp [Selection.disjunction, onOneAttribute, attrsOf, allSame, disjGo, disjMemberValues, pushEqualKeys,
    equal_null, Selection.empty, KeySet.ofKeys, KeySet.distinct, sortDedup, tInsertBatch]

/-- **Every key set the column builds is sorted and distinct** (so `KeySet::contains`' `binary_search`
    is exact): `Keys` only comes from `distinct`, through `of_keys`. -/
theorem ofKeys_sorted (keys ks : List Nat) (h : KeySet.ofKeys keys = .keys ks) : Sorted ks := by
  unfold KeySet.ofKeys KeySet.distinct at h
  have hs := (sortDedup_spec keys).1
  generalize sortDedup keys = l at h hs
  match l with
  | [] => simp at h
  | [a] => simp at h
  | _ :: _ :: _ => simp only [KeySet.keys.injEq] at h; subst h; exact hs

theorem keyset_and_sorted (a b : KeySet) (ks : List Nat) (h : KeySet.and a b = .keys ks) : Sorted ks := by
  cases a <;> cases b <;> simp [KeySet.and] at h <;> first
    | exact ofKeys_sorted _ _ h
    | (split at h <;> simp_all)
    | skip

/-! ### The lazy chain (`UnionIter::next`, `numeric.rs:79-90`)

State `(keys, cur)`: the keys still to visit and the cursor over the current one (`None` until the
first `next`). `next` yields from `cur` if it has an item, else builds the cursor for the next key
(`self.tree.point(self.keys.next()?)`, one descent) and loops; `?` on an empty `keys` returns `None`.
A cursor is modelled by the ids it has left. -/

abbrev UState := List Nat × Option (List Nat)

/-- What the iterator still has to yield: the rest of the current cursor, then every later key's
    point scan in order. -/
def UState.remaining (t : List Nat) (s : UState) : List Nat :=
  s.2.getD [] ++ s.1.flatMap (fun k => tRange t k k)

/-- `UnionIter::next`: the `loop`, structural in `keys` (each non-yielding iteration consumes a key,
    "keys are distinct, so this advances — it cannot spin"). -/
def unionNext (t : List Nat) : List Nat → Option (List Nat) → Option (Nat × UState)
  | keys, some (d :: ds) => some (d, (keys, some ds))
  | [], _ => none
  | k :: ks, _ => unionNext t ks (some (tRange t k k))

/-- **`UnionIter::next` yields exactly the head of what remains**, leaves the tail, and returns
    `None` only when nothing remains. So draining the chain enumerates the point scans in key order
    and `lookup`'s eager model (`lookup (.keys ks)`) is exact. -/
theorem unionNext_spec (t : List Nat) : ∀ (keys : List Nat) (cur : Option (List Nat)),
    (∀ d s, unionNext t keys cur = some (d, s) →
      UState.remaining t (keys, cur) = d :: UState.remaining t s) ∧
    (unionNext t keys cur = none → UState.remaining t (keys, cur) = []) := by
  intro keys
  induction keys with
  | nil =>
    intro cur
    cases cur with
    | none => simp [unionNext, UState.remaining]
    | some l =>
      cases l with
      | nil => simp [unionNext, UState.remaining]
      | cons d ds => simp [unionNext, UState.remaining]
  | cons k ks ih =>
    intro cur
    cases cur with
    | none =>
      obtain ⟨h1, h2⟩ := ih (some (tRange t k k))
      refine ⟨fun d s h => ?_, fun h => ?_⟩
      · simp only [unionNext] at h
        have := h1 d s h
        simpa [UState.remaining] using this
      · simp only [unionNext] at h
        simpa [UState.remaining] using h2 h
    | some l =>
      cases l with
      | nil =>
        obtain ⟨h1, h2⟩ := ih (some (tRange t k k))
        refine ⟨fun d s h => ?_, fun h => ?_⟩
        · simp only [unionNext] at h
          simpa [UState.remaining] using h1 d s h
        · simp only [unionNext] at h
          simpa [UState.remaining] using h2 h
      | cons d ds =>
        refine ⟨fun d' s h => ?_, fun h => by simp [unionNext] at h⟩
        simp only [unionNext, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp [UState.remaining]

/-- The iterator `lookup` builds for `KeySet::Keys(keys)` starts at `(keys, None)` (no cursor until the
    first `next`) and its remaining output is the lookup. -/
theorem unionIter_initial (ix : NumericIndex) (keys : List Nat) :
    UState.remaining ix.tree (keys, none) = ix.lookup (.keys keys) := by
  simp [UState.remaining, NumericIndex.lookup]

/-! ## A key set looked up, and the scan that stands in for it (#2386)

While a column is `Building` the read is answered with a `ScanFilter` (`KeySet` + attribute) instead
of rows; the scan op keeps every entity whose value's key is in the set. Both are driven by the one
`KeySet`, which is why the scan "keeps exactly what the column will answer". -/

/-- `ScanFilter::matches` on a key set (`falkordb_index.rs:272-277`):
    `encode_numeric(value).is_some_and(|key| keys.contains(key))`. -/
def KeySet.matchesValue (ks : KeySet) (v : Value) : Bool := (encodeNumeric v).any ks.contains

/-- A lookup finds exactly the ids stored under a key the set contains. -/
theorem lookup_mem (ix : NumericIndex) (ks : KeySet) (id : Nat) (hid : id < W) :
    id ∈ ix.lookup ks ↔ ∃ k, enc k id ∈ ix.tree ∧ ks.contains k = true := by
  cases ks with
  | empty => simp [NumericIndex.lookup, NumericIndex.empty, tRange_empty, KeySet.contains]
  | point k0 =>
    simp only [NumericIndex.lookup, mem_tRange _ _ _ _ hid, KeySet.contains, beq_iff_eq]
    constructor
    · rintro ⟨k, hk, h1, h2⟩; exact ⟨k, hk, by omega⟩
    · rintro ⟨k, hk, h⟩; exact ⟨k, hk, by omega, by omega⟩
  | range lo hi =>
    simp only [NumericIndex.lookup, mem_tRange _ _ _ _ hid, KeySet.contains, decide_eq_true_eq]
  | keys ks =>
    rw [lookup_keys_mem]
    simp only [mem_tRange _ _ _ _ hid, KeySet.contains, decide_eq_true_eq]
    constructor
    · rintro ⟨k, hk, k', hk', h1, h2⟩; exact ⟨k', hk', by have : k' = k := by omega
                                                          subst this; exact hk⟩
    · rintro ⟨k, hk, h⟩; exact ⟨k, h, k, hk, Nat.le_refl _, Nat.le_refl _⟩

/-- **The scan standing in for a building column keeps exactly what the built column answers.** A
    column built from `entries` answers a key set with the ids whose value's key the set contains,
    which is the `ScanFilter` check on those values (`ScanFilter::matches`). Needs no value
    well-formedness: only that ids fit the entry encoding. -/
theorem lookup_fromEntries (entries : List (Value × Nat)) (hids : ∀ p ∈ entries, p.2 < W)
    (ks : KeySet) (id : Nat) (hid : id < W) :
    id ∈ (NumericIndex.fromEntries entries).lookup ks ↔
      ∃ v, (v, id) ∈ entries ∧ ks.matchesValue v = true := by
  rw [lookup_mem _ _ _ hid]
  simp only [NumericIndex.fromEntries, tFromSorted, (sortDedup_spec _).2]
  constructor
  · rintro ⟨k, hk, hc⟩
    obtain ⟨v, hm, he⟩ := (mem_encodePairs entries k id hid hids).1 hk
    exact ⟨v, hm, by simp [KeySet.matchesValue, he, hc]⟩
  · rintro ⟨v, hm, hf⟩
    cases he : encodeNumeric v with
    | none => simp [KeySet.matchesValue, he] at hf
    | some k =>
      simp only [KeySet.matchesValue, he, Option.any_some] at hf
      exact ⟨k, (mem_encodePairs entries k id hid hids).2 ⟨v, hm, he⟩, hf⟩

/-! ## Lists: the array tree -/

/-- The numeric elements' keys of a list value; none for anything else. -/
def arrayKeys : Value → List Nat
  | .list es => es.filterMap _root_.id
  | _ => []

theorem stores_scalar (v : Value) (p : Nat → Bool) : stores v .scalar p = (encodeNumeric v).any p := by
  cases v with
  | list es =>
    simp only [stores, forEachStoredKey, encodeNumeric, Option.any_none, List.any_map]
    apply Bool.eq_false_iff.2
    simp only [ne_eq, List.any_eq_true, Function.comp_apply]
    rintro ⟨x, _, hx⟩
    simp at hx
  | _ =>
    simp only [stores, forEachStoredKey]
    cases h : encodeNumeric _ <;> simp [h]

theorem stores_array (v : Value) (p : Nat → Bool) : stores v .array p = (arrayKeys v).any p := by
  cases v with
  | list es => simp [stores, forEachStoredKey, arrayKeys, List.any_map, Function.comp_def]
  | _ =>
    simp only [stores, forEachStoredKey, arrayKeys]
    cases h : encodeNumeric _ <;> simp [h]

theorem isScalarIn_eq (ks : KeySet) (v : Value) : isScalarIn ks v = ks.matchesValue v := by
  simp [isScalarIn, stores_scalar, KeySet.matchesValue]

theorem isListHolding_iff (key : Nat) (v : Value) : isListHolding key v = true ↔ key ∈ arrayKeys v := by
  simp [isListHolding, stores_array, List.any_eq_true]

theorem mem_listsHolding (ix : NumericIndex) (k id : Nat) (hid : id < W) :
    id ∈ ix.listsHolding k ↔ enc k id ∈ ix.arrayTree := by
  simp only [NumericIndex.listsHolding, mem_tRange _ _ _ _ hid]
  constructor
  · rintro ⟨k', hk', h1, h2⟩; have : k' = k := by omega
    subst this; exact hk'
  · intro h; exact ⟨k, h, Nat.le_refl _, Nat.le_refl _⟩

theorem lookupLists_foldl_mem (ix : NumericIndex) (rest : List Nat) : ∀ (found : List Nat) (d : Nat),
    d ∈ rest.foldl (fun found e =>
      if found.isEmpty then found else (ix.listsHolding e).filter (fun d => found.contains d)) found ↔
      d ∈ found ∧ ∀ e ∈ rest, d ∈ ix.listsHolding e := by
  induction rest with
  | nil => intro found d; simp
  | cons e rest ih =>
    intro found d
    simp only [List.foldl_cons]
    rw [ih]
    by_cases he : found.isEmpty = true
    · have : found = [] := List.isEmpty_iff.1 he
      subst this; simp
    · simp only [he, Bool.false_eq_true, ite_false, List.mem_filter, List.contains_eq_any_beq,
        List.any_eq_true, beq_iff_eq, List.mem_cons, forall_eq_or_imp]
      constructor
      · rintro ⟨⟨h1, ⟨x, hx, rfl⟩⟩, h3⟩; exact ⟨hx, h1, h3⟩
      · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h2, ⟨d, h1, rfl⟩⟩, h3⟩

/-- **A `ListsHolding` lookup returns the lists holding every probed element.** -/
theorem lookupLists_mem (ix : NumericIndex) (first : Nat) (rest : List Nat) (d : Nat) :
    d ∈ ix.lookupLists first rest ↔ d ∈ ix.listsHolding first ∧ ∀ e ∈ rest, d ∈ ix.listsHolding e :=
  lookupLists_foldl_mem ix rest _ d

theorem mem_encodeArrayPairs (entries : List (Value × Nat)) (k id : Nat) (hid : id < W)
    (hids : ∀ p ∈ entries, p.2 < W) :
    enc k id ∈ encodeArrayPairs entries ↔ ∃ v, (v, id) ∈ entries ∧ k ∈ arrayKeys v := by
  simp only [encodeArrayPairs, List.mem_flatMap]
  constructor
  · rintro ⟨⟨v, id'⟩, hm, he⟩
    have hid' := hids _ hm
    cases v with
    | list es =>
      simp only [List.mem_map] at he
      obtain ⟨k', hk', he'⟩ := he
      have h1 := congrArg keyOf he'; have h2 := congrArg docOf he'
      rw [keyOf_enc _ _ hid', keyOf_enc _ _ hid] at h1
      rw [docOf_enc _ _ hid', docOf_enc _ _ hid] at h2
      subst h1 h2; exact ⟨.list es, hm, by simpa [arrayKeys] using hk'⟩
    | _ => simp at he
  · rintro ⟨v, hm, hk⟩
    refine ⟨(v, id), hm, ?_⟩
    cases v with
    | list es =>
      show enc k id ∈ (es.filterMap _root_.id).map (fun k => enc k id)
      exact List.mem_map.2 ⟨k, by simpa [arrayKeys] using hk, rfl⟩
    | _ => simp [arrayKeys] at hk

/-- **The scan standing in for a building column keeps exactly what the built column answers, for every
    selection** (scalars and lists): a column built from `entries` (one value per entity) answers a
    selection with the ids whose value `Selection::matches`. -/
theorem lookupSel_fromEntries (entries : List (Value × Nat)) (hids : ∀ p ∈ entries, p.2 < W)
    (hfun : ∀ p ∈ entries, ∀ q ∈ entries, p.2 = q.2 → p.1 = q.1)
    (sel : Selection) (id : Nat) (hid : id < W) :
    id ∈ (NumericIndex.fromEntries entries).lookupSel sel ↔
      ∃ v, (v, id) ∈ entries ∧ sel.matches v = true := by
  cases sel with
  | scalars ks =>
    simp only [NumericIndex.lookupSel, Selection.matches, isScalarIn_eq]
    exact lookup_fromEntries entries hids ks id hid
  | listsHolding first rest =>
    simp only [NumericIndex.lookupSel, lookupLists_mem, Selection.matches, List.all_eq_true,
      isListHolding_iff, List.mem_cons, forall_eq_or_imp]
    have key : ∀ k, id ∈ (NumericIndex.fromEntries entries).listsHolding k ↔
        ∃ v, (v, id) ∈ entries ∧ k ∈ arrayKeys v := by
      intro k
      rw [mem_listsHolding _ _ _ hid]
      simp only [NumericIndex.fromEntries, tFromSorted, (sortDedup_spec _).2]
      exact mem_encodeArrayPairs entries k id hid hids
    rw [key]
    constructor
    · rintro ⟨⟨v, hv, hf⟩, h2⟩
      refine ⟨v, hv, hf, fun e he => ?_⟩
      obtain ⟨w, hw, hew⟩ := (key e).1 (h2 e he)
      have := hfun _ hv _ hw rfl
      simp only at this
      rw [this]; exact hew
    · rintro ⟨v, hv, hf, h2⟩
      exact ⟨⟨v, hv, hf⟩, fun e he => (key e).2 ⟨v, hv, h2 e he⟩⟩

/-! ## Writes -/

theorem addTree_spec (t new : List Nat) (e : Nat) : e ∈ addTree t new ↔ e ∈ t ∨ e ∈ new := by
  unfold addTree
  by_cases h : new.isEmpty = true
  · have : new = [] := List.isEmpty_iff.1 h
    subst this; simp
  · by_cases ht : tIsEmpty t = true
    · have : t = [] := by simpa [tIsEmpty] using ht
      subst this
      simp [h, tIsEmpty, tFromSorted]
    · simp [h, ht, mem_insertBatch]

theorem addTree_sorted (t new : List Nat) (hs : Sorted t) (hn : Sorted new) : Sorted (addTree t new) := by
  unfold addTree
  by_cases h : new.isEmpty = true
  · simp [h, hs]
  · by_cases ht : tIsEmpty t = true
    · simp [h, ht, tFromSorted, hn]
    · simp only [h, ht, Bool.false_eq_true, ite_false]; exact sorted_insertBatch _ _ hs

theorem removeTree_spec (t rm : List Nat) (e : Nat) : e ∈ removeTree t rm ↔ e ∈ t ∧ e ∉ rm := by
  unfold removeTree
  by_cases h : rm.isEmpty = true
  · have : rm = [] := List.isEmpty_iff.1 h
    subst this; simp
  · simp [h, mem_removeBatch]

theorem foldl_push_scalar (id : Nat) (l : List (KeyKind × Nat)) : ∀ out : EncodedTuples,
    (l.foldl (fun o kk => o.push kk.1 (enc kk.2 id)) out).scalar =
      out.scalar ++ (l.filter (fun kk => decide (kk.1 = .scalar))).map (fun kk => enc kk.2 id) := by
  induction l with
  | nil => intro out; simp
  | cons kk l ih =>
    intro out
    simp only [List.foldl_cons]
    rw [ih]
    obtain ⟨k, n⟩ := kk
    cases k <;> simp [EncodedTuples.push, List.filter_cons, List.append_assoc]

theorem foldl_push_array (id : Nat) (l : List (KeyKind × Nat)) : ∀ out : EncodedTuples,
    (l.foldl (fun o kk => o.push kk.1 (enc kk.2 id)) out).array =
      out.array ++ (l.filter (fun kk => decide (kk.1 = .array))).map (fun kk => enc kk.2 id) := by
  induction l with
  | nil => intro out; simp
  | cons kk l ih =>
    intro out
    simp only [List.foldl_cons]
    rw [ih]
    obtain ⟨k, n⟩ := kk
    cases k <;> simp [EncodedTuples.push, List.filter_cons, List.append_assoc]

/-- A value's scalar tuple: its `encode_numeric` key (a list has none). -/
theorem pushStored_scalar (v : Value) (id : Nat) (out : EncodedTuples) :
    (pushStored v id out).scalar =
      out.scalar ++ (match encodeNumeric v with | some k => [enc k id] | none => []) := by
  unfold pushStored
  rw [foldl_push_scalar]
  cases v with
  | list es => simp [forEachStoredKey, encodeNumeric, List.filter_map, Function.comp_def]
  | _ =>
    simp only [forEachStoredKey]
    cases h : encodeNumeric _ <;> simp [h]

/-- A list's array tuples: one per numeric element. -/
theorem pushStored_array (v : Value) (id : Nat) (out : EncodedTuples) :
    (pushStored v id out).array = out.array ++ (arrayKeys v).map (fun k => enc k id) := by
  unfold pushStored
  rw [foldl_push_array]
  cases v with
  | list es =>
    simp [forEachStoredKey, arrayKeys, List.filter_map, Function.comp_def]
    congr 1
    exact List.filter_eq_self.2 (fun _ _ => rfl)
  | _ =>
    simp only [forEachStoredKey, arrayKeys]
    cases h : encodeNumeric _ <;> simp [h]

theorem addEncoded_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.addEncoded tuples).1.tree ↔ e ∈ ix.tree ∨ e ∈ tuples.scalar := by
  simp only [NumericIndex.addEncoded, NumericIndex.addTrees, EncodedTuples.sortDedup, addTree_spec,
    (sortDedup_spec _).2]

theorem addEncoded_array_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.addEncoded tuples).1.arrayTree ↔ e ∈ ix.arrayTree ∨ e ∈ tuples.array := by
  simp only [NumericIndex.addEncoded, NumericIndex.addTrees, EncodedTuples.sortDedup, addTree_spec,
    (sortDedup_spec _).2]

/-- Both trees stay sorted: a bulk build into an empty tree or a merge into a non-empty one. -/
theorem addEncoded_sorted (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (tuples : EncodedTuples) :
    Sorted (ix.addEncoded tuples).1.tree ∧ Sorted (ix.addEncoded tuples).1.arrayTree :=
  ⟨addTree_sorted _ _ hs (sortDedup_spec _).1, addTree_sorted _ _ ha (sortDedup_spec _).1⟩

/-- The tuples a call hands back are its sorted, deduplicated input. -/
theorem addEncoded_tuples (ix : NumericIndex) (tuples : EncodedTuples) :
    (ix.addEncoded tuples).2 = tuples.sortDedup := rfl

theorem removeEncoded_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.removeEncoded tuples).1.tree ↔ e ∈ ix.tree ∧ e ∉ tuples.scalar := by
  simp only [NumericIndex.removeEncoded, NumericIndex.removeTrees, EncodedTuples.sortDedup, removeTree_spec,
    (sortDedup_spec _).2]

theorem removeEncoded_array_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.removeEncoded tuples).1.arrayTree ↔ e ∈ ix.arrayTree ∧ e ∉ tuples.array := by
  simp only [NumericIndex.removeEncoded, NumericIndex.removeTrees, EncodedTuples.sortDedup, removeTree_spec,
    (sortDedup_spec _).2]

/-- `EncodedTuples`' accessors (`numeric.rs:269-301`). -/
theorem encodedTuples_spec (s : List Nat) (t : EncodedTuples) :
    (EncodedTuples.scalars s).scalar = s ∧ (EncodedTuples.scalars s).array = [] ∧
    (t.isEmpty = true ↔ t.scalar = [] ∧ t.array = []) ∧ t.len = t.scalar.length + t.array.length ∧
    t.clear = EncodedTuples.empty ∧ Sorted t.sortDedup.scalar ∧ Sorted t.sortDedup.array ∧
    (∀ e, e ∈ t.sortDedup.scalar ↔ e ∈ t.scalar) ∧ (∀ e, e ∈ t.sortDedup.array ↔ e ∈ t.array) := by
  refine ⟨rfl, rfl, ?_, rfl, rfl, (sortDedup_spec _).1, (sortDedup_spec _).1,
    fun e => (sortDedup_spec _).2 e, fun e => (sortDedup_spec _).2 e⟩
  simp [EncodedTuples.isEmpty]

/-- `EncodedTuples::index` / `index_mut` (`numeric.rs:235-252`): `tuples[kind]`, and a push onto it. -/
theorem encodedTuples_get_push (t : EncodedTuples) (x : Nat) :
    t.get .scalar = t.scalar ∧ t.get .array = t.array ∧
    (t.push .scalar x).scalar = t.scalar ++ [x] ∧ (t.push .scalar x).array = t.array ∧
    (t.push .array x).array = t.array ++ [x] ∧ (t.push .array x).scalar = t.scalar :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `NumericIndex::tree_mut` (`numeric.rs:333-340`) picks the tree of the kind: the model's two fields. -/
theorem forEachStoredKey_scalar (v : Value) (k : Nat) :
    (KeyKind.scalar, k) ∈ forEachStoredKey v ↔ encodeNumeric v = some k ∧ ∀ es, v ≠ .list es := by
  cases v with
  | list es => simp [forEachStoredKey]
  | _ =>
    simp only [forEachStoredKey]
    cases h : encodeNumeric _ <;> simp [h, eq_comm]

theorem forEachStoredKey_array (v : Value) (k : Nat) :
    (KeyKind.array, k) ∈ forEachStoredKey v ↔ k ∈ arrayKeys v := by
  cases v with
  | list es => simp [forEachStoredKey, arrayKeys]
  | _ =>
    simp only [forEachStoredKey, arrayKeys]
    cases h : encodeNumeric _ <;> simp [h]

theorem add_spec (ix : NumericIndex) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.add v id).tree ↔ e ∈ ix.tree ∨ ∃ k, encodeNumeric v = some k ∧ e = enc k id := by
  unfold NumericIndex.add
  rw [addEncoded_spec, pushStored_scalar]
  cases h : encodeNumeric v <;> simp [EncodedTuples.empty, h]

theorem add_array_spec (ix : NumericIndex) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.add v id).arrayTree ↔ e ∈ ix.arrayTree ∨ ∃ k ∈ arrayKeys v, e = enc k id := by
  unfold NumericIndex.add
  rw [addEncoded_array_spec, pushStored_array]
  simp [EncodedTuples.empty, eq_comm]

theorem remove_spec (ix : NumericIndex) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.remove v id).tree ↔ e ∈ ix.tree ∧ ∀ k, encodeNumeric v = some k → e ≠ enc k id := by
  unfold NumericIndex.remove
  rw [removeEncoded_spec, pushStored_scalar]
  cases h : encodeNumeric v <;> simp [EncodedTuples.empty, h]

theorem remove_array_spec (ix : NumericIndex) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.remove v id).arrayTree ↔ e ∈ ix.arrayTree ∧ ∀ k ∈ arrayKeys v, e ≠ enc k id := by
  unfold NumericIndex.remove
  rw [removeEncoded_array_spec, pushStored_array]
  simp [EncodedTuples.empty]
  intro _; constructor
  · intro h k hk he; exact h k hk he.symm
  · intro h k hk he; exact h k hk he.symm

theorem foldl_pushStored_scalar (entries : List (Value × Nat)) : ∀ out : EncodedTuples,
    (entries.foldl (fun o p => pushStored p.1 p.2 o) out).scalar = out.scalar ++ encodePairs entries := by
  induction entries with
  | nil => intro out; simp [encodePairs]
  | cons p rest ih =>
    intro out
    simp only [List.foldl_cons]
    rw [ih, pushStored_scalar]
    obtain ⟨v, id⟩ := p
    simp only [encodePairs, List.filterMap_cons]
    cases h : encodeNumeric v <;> simp [h, List.append_assoc]

theorem foldl_pushStored_array (entries : List (Value × Nat)) : ∀ out : EncodedTuples,
    (entries.foldl (fun o p => pushStored p.1 p.2 o) out).array = out.array ++ encodeArrayPairs entries := by
  induction entries with
  | nil => intro out; simp [encodeArrayPairs]
  | cons p rest ih =>
    intro out
    simp only [List.foldl_cons]
    rw [ih, pushStored_array]
    obtain ⟨v, id⟩ := p
    simp only [encodeArrayPairs, List.flatMap_cons, List.append_assoc]
    cases v <;> simp [arrayKeys]

theorem pushAll_spec (entries : List (Value × Nat)) :
    (pushAll entries).scalar = encodePairs entries ∧ (pushAll entries).array = encodeArrayPairs entries := by
  unfold pushAll
  exact ⟨by rw [foldl_pushStored_scalar]; simp [EncodedTuples.empty],
         by rw [foldl_pushStored_array]; simp [EncodedTuples.empty]⟩

theorem addBatch_spec (ix : NumericIndex) (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (ix.addBatch entries).1.tree ↔ e ∈ ix.tree ∨ e ∈ encodePairs entries := by
  rw [NumericIndex.addBatch, addEncoded_spec, (pushAll_spec entries).1]

theorem addBatch_array_spec (ix : NumericIndex) (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (ix.addBatch entries).1.arrayTree ↔ e ∈ ix.arrayTree ∨ e ∈ encodeArrayPairs entries := by
  rw [NumericIndex.addBatch, addEncoded_array_spec, (pushAll_spec entries).2]

theorem removeBatch_spec (ix : NumericIndex) (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (ix.removeBatch entries).1.tree ↔ e ∈ ix.tree ∧ e ∉ encodePairs entries := by
  rw [NumericIndex.removeBatch, removeEncoded_spec, (pushAll_spec entries).1]

theorem removeBatch_array_spec (ix : NumericIndex) (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (ix.removeBatch entries).1.arrayTree ↔ e ∈ ix.arrayTree ∧ e ∉ encodeArrayPairs entries := by
  rw [NumericIndex.removeBatch, removeEncoded_array_spec, (pushAll_spec entries).2]

theorem fromEntries_spec (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (NumericIndex.fromEntries entries).tree ↔ e ∈ encodePairs entries := by
  simp [NumericIndex.fromEntries, tFromSorted, (sortDedup_spec _).2]

theorem fromEntries_array_spec (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (NumericIndex.fromEntries entries).arrayTree ↔ e ∈ encodeArrayPairs entries := by
  simp [NumericIndex.fromEntries, tFromSorted, (sortDedup_spec _).2]

theorem fromEntries_sorted (entries : List (Value × Nat)) :
    Sorted (NumericIndex.fromEntries entries).tree ∧ Sorted (NumericIndex.fromEntries entries).arrayTree :=
  ⟨(sortDedup_spec _).1, (sortDedup_spec _).1⟩

theorem isEmpty_spec (ix : NumericIndex) : ix.isEmpty = true ↔ ix.tree = [] ∧ ix.arrayTree = [] := by
  simp [NumericIndex.isEmpty, tIsEmpty]

theorem empty_spec (ix : NumericIndex) : ix.empty = [] := tRange_empty _

theorem new_spec : NumericIndex.new.tree = [] ∧ NumericIndex.new.arrayTree = [] := ⟨rfl, rfl⟩

/-- Temporal values never reach either tree (`encode.rs:34-38`). -/
theorem temporal_not_added (ix : NumericIndex) (i : Int) (id : Nat) :
    (ix.add (.date i) id).tree = ix.tree ∧ (ix.add (.date i) id).arrayTree = ix.arrayTree := by
  simp [NumericIndex.add, NumericIndex.addEncoded, NumericIndex.addTrees, pushStored, forEachStoredKey,
    encodeNumeric, EncodedTuples.empty, EncodedTuples.sortDedup, addTree, sortDedup, tInsertBatch]

end NI
