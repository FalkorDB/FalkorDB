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

/-- `EncodedTuples { scalar }` (`numeric.rs:92-94`): the encoded `(key, doc)` tuples a column's
    changes move in. A tuple `(key, doc)` is its tree entry `enc key doc` (`Tree.lean`). -/
structure EncodedTuples where
  scalar : List Nat
deriving Repr, DecidableEq

namespace EncodedTuples
/-- `Default` -/
def empty : EncodedTuples := ⟨[]⟩
/-- `scalars` (`numeric.rs:101-103`). -/
def scalars (s : List Nat) : EncodedTuples := ⟨s⟩
/-- `is_empty` (`numeric.rs:107-109`). -/
def isEmpty (t : EncodedTuples) : Bool := t.scalar.isEmpty
/-- `len` (`numeric.rs:113-115`). -/
def len (t : EncodedTuples) : Nat := t.scalar.length
/-- `clear` (`numeric.rs:118-120`). -/
def clear (_t : EncodedTuples) : EncodedTuples := ⟨[]⟩
/-- `sort_dedup` (`numeric.rs:124-127`): `sort_unstable()` then `dedup()`. -/
def sortDedup (t : EncodedTuples) : EncodedTuples := ⟨NI.sortDedup t.scalar⟩
end EncodedTuples

/-- `NumericIndex { tree }`. -/
structure NumericIndex where
  tree : List Nat
deriving Repr

/-- `NumericIndex::new` / `Default` (`numeric.rs:146-148`). -/
def NumericIndex.new : NumericIndex := ⟨[]⟩

/-- The `(key, id)` entries of `(value, id)` pairs: the `filter_map` of `numeric.rs:216-219`, `:439`. -/
def encodePairs (entries : List (Value × Nat)) : List Nat :=
  entries.filterMap (fun (v, id) => (encodeNumeric v).map (fun k => enc k id))

/-- `from_entries` (`numeric.rs:434-447`, `#[cfg(test)]` since #2351; still the specification of what a built column holds): encode, `sort_unstable`, `dedup`, `from_sorted`. -/
def NumericIndex.fromEntries (entries : List (Value × Nat)) : NumericIndex :=
  ⟨tFromSorted (sortDedup (encodePairs entries))⟩

/-- `add` (`numeric.rs:152-163`). -/
def NumericIndex.add (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  match encodeNumeric v with
  | some k => ⟨tInsert ix.tree (enc k id)⟩
  | none => ix

/-- `remove` (`numeric.rs:166-174`). -/
def NumericIndex.remove (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  match encodeNumeric v with
  | some k => ⟨tRemove ix.tree (enc k id)⟩
  | none => ix

/-- `encode_pairs_into` (`numeric.rs:212-220`): `out` is cleared, then filled. Returns the new `out`. -/
def encodePairsInto (entries : List (Value × Nat)) (_out : List Nat) : List Nat :=
  encodePairs entries

/-- `add_batch` (`numeric.rs:183-191`); returns the index and the scratch buffer. -/
def NumericIndex.addBatch (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) :
    NumericIndex × List Nat :=
  let scratch := sortDedup (encodePairsInto entries scratch)   -- `sort_unstable` (dups harmless: union)
  (⟨tInsertBatch ix.tree scratch⟩, scratch)

/-- `remove_batch` (`numeric.rs:197-205`). -/
def NumericIndex.removeBatch (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) :
    NumericIndex × List Nat :=
  let scratch := sortDedup (encodePairsInto entries scratch)
  (⟨tRemoveBatch ix.tree scratch⟩, scratch)

/-- `add_encoded` (`numeric.rs:226-236`): `tuples.sort_dedup()`, then into an empty tree
    `Tree::from_sorted` (a bulk build), else `insert_batch`. Returns the index and the (sorted) tuples. -/
def NumericIndex.addEncoded (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex × EncodedTuples :=
  let tuples := tuples.sortDedup
  (if tIsEmpty ix.tree then ⟨tFromSorted tuples.scalar⟩ else ⟨tInsertBatch ix.tree tuples.scalar⟩, tuples)

/-- `remove_encoded` (`numeric.rs:239-246`). -/
def NumericIndex.removeEncoded (ix : NumericIndex) (tuples : EncodedTuples) : NumericIndex × EncodedTuples :=
  let tuples := tuples.sortDedup
  (⟨tRemoveBatch ix.tree tuples.scalar⟩, tuples)

/-- `is_empty` (`numeric.rs:249-251`). -/
def NumericIndex.isEmpty (ix : NumericIndex) : Bool := tIsEmpty ix.tree

/-- `empty` (`numeric.rs:424-426`): `range(1, 0)`. -/
def NumericIndex.empty (ix : NumericIndex) : List Nat := tRange ix.tree 1 0

/-- `point` (`numeric.rs:255-263`). -/
def NumericIndex.point (ix : NumericIndex) (v : Value) : List Nat :=
  match encodeNumeric v with
  | some k => tRange ix.tree k k
  | none => ix.empty

/-- `range` (`numeric.rs:269-316`), branch for branch (`none` = an early `return self.empty()`). -/
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

def NumericIndex.range (ix : NumericIndex) (min max : Option Value) (incMin incMax : Bool) : List Nat :=
  match rangeBounds min max incMin incMax with
  | none => ix.empty
  | some (lo, hi) => tRange ix.tree lo hi

/-- `IndexQuery<Value>` (`index/mod.rs`), the variants the native index distinguishes. `inList`'s
    payload is `some items` for `list: Value::List(items)` and `none` for any other `Value` (#2383);
    `or` carries its children (`And`, `Point`, `ArrayContains` are `other`). -/
inductive IndexQuery where
  | equal (key : String) (value : Value)
  | range (key : String) (min max : Option Value) (incMin incMax : Bool)
  | inList (key : String) (list : Option (List Value))
  | or (children : List IndexQuery)
  | other

/-- `encode_numeric` over a whole list, `collect::<Option<Vec<u64>>>()`: `none` if any member does
    not encode (`numeric.rs` `in_list`, `union`). -/
def encodeAll (vs : List Value) : Option (List Nat) := vs.mapM encodeNumeric

/-- `union_of` (`numeric.rs:408-421`): `keys.sort_unstable(); keys.dedup()`, then one point cursor per
    distinct key, chained lazily. As a set of ids this is the concatenation of the point scans. The
    chain is deliberately *not* id-ordered (`DocIter` doc); the model is a list whose order is not
    claimed, every theorem below is about membership. The tree snapshot is pinned by `tree.clone()`,
    so every member is answered from one version; here the tree is an immutable value, which is
    that property. -/
def NumericIndex.unionOf (ix : NumericIndex) (keys : List Nat) : List Nat :=
  (sortDedup keys).flatMap (fun k => tRange ix.tree k k)

/-- `in_list` (`numeric.rs:356-368`): an empty list is declined, as is a list with any member that
    does not encode. -/
def NumericIndex.inList (ix : NumericIndex) (items : List Value) : Option (List Nat) :=
  if items.isEmpty then none else (encodeAll items).map ix.unionOf

/-- The `match child` of `union` (`numeric.rs:386-392`): an `Equal`, or a **non-empty** `IN` list. -/
def unionMember : IndexQuery → Option (String × List Value)
  | .equal k v => some (k, [v])
  | .inList k (some items) => if items.isEmpty then none else some (k, items)
  | _ => none

/-- The loop of `union` (`numeric.rs:382-399`). `attr` is the `get_or_insert` cell: the first member
    fixes it and every later member must name the same attribute. -/
def unionGo (attr : Option String) (keys : List Nat) : List IndexQuery → Option (List Nat)
  | [] => some keys
  | c :: cs =>
    match unionMember c with
    | none => none
    | some (k, vs) =>
      if attr.getD k ≠ k then none
      else match encodeAll vs with
        | none => none
        | some ks => unionGo (some k) (keys ++ ks) cs

/-- `union` (`numeric.rs:376-400`): `(!keys.is_empty()).then(|| self.union_of(keys))`. -/
def NumericIndex.union (ix : NumericIndex) (children : List IndexQuery) : Option (List Nat) :=
  (unionGo none [] children).bind (fun keys => if keys.isEmpty then none else some (ix.unionOf keys))

/-- `query` (`numeric.rs:323-343`). -/
def NumericIndex.query (ix : NumericIndex) : IndexQuery → Option (List Nat)
  | .equal _ v => some (ix.point v)
  | .range _ min max a b => some (ix.range min max a b)
  | .inList _ (some items) => ix.inList items
  | .inList _ none => none
  | .or cs => ix.union cs
  | .other => none

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
    simp only [NumericIndex.range, NumericIndex.fromEntries, NumericIndex.empty, tFromSorted]
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
  simp only [NumericIndex.point, hkb, NumericIndex.fromEntries, tFromSorted]
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

/-! ## `IN` lists and unions of point lookups (#2383)

`n.a IN [v1, .., vn]` reaches the column as one `InList`, and `a = v1 OR a IN [..]` as an `Or`.
Both are served as a union of point lookups over the distinct keys (`union_of`). The Rust claims,
and the theorems below prove, three things: the union is exactly the ids a label scan plus the
`IN` / `OR` filter returns (`inList_agrees`, `union_agrees`); the column *declines* (`None`) rather
than answer part of a query (`inList_isSome_iff`, `union_nil`, `union_attr_mismatch`), because the
scan op does not re-check the predicate; and the set is the same whatever the order, since the lazy
chain is deliberately not id-ordered (all statements are membership). -/

theorem encodeAll_nil : encodeAll [] = some [] := rfl

theorem encodeAll_cons (v : Value) (vs : List Value) :
    encodeAll (v :: vs) = (encodeNumeric v).bind (fun k => (encodeAll vs).map (k :: ·)) := by
  simp [encodeAll, List.mapM_cons]
  cases encodeNumeric v <;> simp
  rename_i k
  cases List.mapM encodeNumeric vs <;> simp

theorem encodeAll_isSome (vs : List Value) :
    (encodeAll vs).isSome = true ↔ ∀ v ∈ vs, (encodeNumeric v).isSome = true := by
  induction vs with
  | nil => simp [encodeAll_nil]
  | cons v vs ih =>
    rw [encodeAll_cons]
    cases h : encodeNumeric v <;> simp [h, ih]

theorem encodeAll_mem (vs : List Value) (ks : List Nat) (h : encodeAll vs = some ks) (k : Nat) :
    k ∈ ks ↔ ∃ v ∈ vs, encodeNumeric v = some k := by
  induction vs generalizing ks with
  | nil => simp [encodeAll_nil] at h; subst h; simp
  | cons v vs ih =>
    rw [encodeAll_cons] at h
    cases hv : encodeNumeric v with
    | none => simp [hv] at h
    | some k0 =>
      simp only [hv, Option.bind_some, Option.map_eq_some_iff] at h
      obtain ⟨ks', hks', rfl⟩ := h
      rw [List.mem_cons, ih ks' hks']
      constructor
      · rintro (rfl | ⟨w, hw, hk⟩)
        · exact ⟨v, List.mem_cons_self .., hv⟩
        · exact ⟨w, List.mem_cons_of_mem _ hw, hk⟩
      · rintro ⟨w, hw, hk⟩
        rcases List.mem_cons.1 hw with rfl | hw
        · left; rw [hv] at hk; exact (Option.some.inj hk).symm
        · right; exact ⟨w, hw, hk⟩

theorem encodeAll_append (as bs : List Value) (ka kb : List Nat) (ha : encodeAll as = some ka)
    (hb : encodeAll bs = some kb) : encodeAll (as ++ bs) = some (ka ++ kb) := by
  induction as generalizing ka with
  | nil => simp [encodeAll_nil] at ha; subst ha; simpa using hb
  | cons v vs ih =>
    rw [encodeAll_cons] at ha
    cases hv : encodeNumeric v with
    | none => simp [hv] at ha
    | some k0 =>
      simp only [hv, Option.bind_some, Option.map_eq_some_iff] at ha
      obtain ⟨ks', hks', rfl⟩ := ha
      rw [List.cons_append, encodeAll_cons, hv]
      simp [ih ks' hks']

theorem unionOf_mem (ix : NumericIndex) (keys : List Nat) (id : Nat) :
    id ∈ ix.unionOf keys ↔ ∃ k ∈ keys, id ∈ tRange ix.tree k k := by
  simp only [NumericIndex.unionOf, List.mem_flatMap, (sortDedup_spec _).2]

/-- `in_list` answers exactly when the list is non-empty and every member encodes (`numeric.rs`
    doc: serving a subset "would silently drop rows"). -/
theorem inList_isSome_iff (ix : NumericIndex) (items : List Value) :
    (ix.inList items).isSome = true ↔ items ≠ [] ∧ ∀ v ∈ items, (encodeNumeric v).isSome = true := by
  unfold NumericIndex.inList
  cases items with
  | nil => simp
  | cons v vs =>
    simp only [List.isEmpty_cons, Bool.false_eq_true, if_false, Option.isSome_map, encodeAll_isSome]
    simp

/-- **`IN` list ≡ label scan + `IN` filter**: a served `IN [b1, .., bn]` (every `bi` a guarded
    non-NaN numeric) returns exactly the ids whose value equals some member. -/
theorem inList_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (items : List Value) (hgood : ∀ b ∈ items, GoodBound b) (r : List Nat)
    (hr : (NumericIndex.fromEntries entries).inList items = some r) (id : Nat) (hid : id < W) :
    id ∈ r ↔ ∃ v, (v, id) ∈ entries ∧ ∃ b ∈ items, cmpPred .eq v b = some true := by
  unfold NumericIndex.inList at hr
  cases hne : items.isEmpty
  · simp only [hne, Bool.false_eq_true, if_false] at hr
    cases hall : encodeAll items with
    | none => simp [hall] at hr
    | some ks =>
      simp only [hall, Option.map_some, Option.some.injEq] at hr
      subst hr
      rw [unionOf_mem]
      constructor
      · rintro ⟨k, hk, hmem⟩
        obtain ⟨b, hb, hkb⟩ := (encodeAll_mem items ks hall k).1 hk
        have hp : id ∈ (NumericIndex.fromEntries entries).point b := by
          simp only [NumericIndex.point, hkb]; exact hmem
        obtain ⟨v, hv, hc⟩ := (point_agrees entries hwf b (hgood b hb) id hid).1 hp
        exact ⟨v, hv, b, hb, hc⟩
      · rintro ⟨v, hv, b, hb, hc⟩
        have hp := (point_agrees entries hwf b (hgood b hb) id hid).2 ⟨v, hv, hc⟩
        obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hgood b hb)
        simp only [NumericIndex.point, hkb] at hp
        exact ⟨kb, (encodeAll_mem items ks hall kb).2 ⟨b, hb, hkb⟩, hp⟩
  · simp [hne] at hr

/-- The values a union's members contribute. -/
def memberValues (cs : List IndexQuery) : List Value :=
  cs.flatMap (fun c => match unionMember c with | some (_, vs) => vs | none => [])

/-- Every member is an `Equal` or non-empty `IN` list on attribute `a`. -/
def SameAttr (a : String) (cs : List IndexQuery) : Prop :=
  ∀ c ∈ cs, ∃ vs, unionMember c = some (a, vs)

theorem unionGo_spec (cs : List IndexQuery) : ∀ (attr : Option String) (keys out : List Nat),
    unionGo attr keys cs = some out →
    ∃ a, (∀ b, attr = some b → a = b) ∧ SameAttr a cs ∧
      ∃ ks, encodeAll (memberValues cs) = some ks ∧ out = keys ++ ks := by
  induction cs with
  | nil =>
    intro attr keys out h
    simp only [unionGo, Option.some.injEq] at h
    subst h
    exact ⟨attr.getD "", fun b hb => by simp [hb], fun c hc => by simp at hc,
      [], by simp [memberValues, encodeAll_nil], by simp⟩
  | cons c cs ih =>
    intro attr keys out h
    simp only [unionGo] at h
    cases hm : unionMember c with
    | none => simp [hm] at h
    | some p =>
      obtain ⟨k, vs⟩ := p
      simp only [hm] at h
      by_cases hk : attr.getD k ≠ k
      · simp [hk] at h
      · simp only [hk, if_false] at h
        cases he : encodeAll vs with
        | none => simp [he] at h
        | some ks1 =>
          simp only [he] at h
          obtain ⟨a, ha, hs, ks2, hk2, hout⟩ := ih (some k) (keys ++ ks1) out h
          have hak : a = k := ha k rfl
          subst hak
          refine ⟨a, ?_, ?_, ks1 ++ ks2, ?_, ?_⟩
          · intro b hb
            rw [hb] at hk
            simp only [ne_eq, Decidable.not_not, Option.getD_some] at hk
            subst hk; rfl
          · intro c' hc'
            rcases List.mem_cons.1 hc' with rfl | hc'
            · exact ⟨vs, hm⟩
            · exact hs c' hc'
          · simp only [memberValues, List.flatMap_cons, hm] at hk2 ⊢
            exact encodeAll_append _ _ _ _ he hk2
          · rw [hout]; simp [List.append_assoc]

/-- An empty `OR` is not match-all: it is declined (`numeric.rs`, test `an empty OR`). -/
theorem union_nil (ix : NumericIndex) : ix.union [] = none := by
  simp [NumericIndex.union, unionGo]

/-- An `OR` spanning two attributes cannot be answered from one column: it is declined. -/
theorem union_attr_mismatch (ix : NumericIndex) (a b : String) (hab : a ≠ b) (v w : Value) :
    ix.union [.equal a v, .equal b w] = none := by
  simp only [NumericIndex.union, unionGo, unionMember]
  cases encodeAll [v] <;> simp [Option.getD_none, Option.getD_some, hab]

/-- A range (or any other child) is not a union member: the whole `OR` is declined. -/
theorem union_range_declines (ix : NumericIndex) (k : String) (mn mx : Option Value) (a b : Bool)
    (rest : List IndexQuery) : ix.union (.range k mn mx a b :: rest) = none := by
  simp [NumericIndex.union, unionGo, unionMember]

/-- **`OR` of equalities / `IN` lists ≡ label scan + `OR` filter**: a served union returns exactly
    the ids whose value equals some member of some child, and every child names one attribute. -/
theorem union_agrees (entries : List (Value × Nat))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W)
    (cs : List IndexQuery) (hgood : ∀ b ∈ memberValues cs, GoodBound b) (r : List Nat)
    (hr : (NumericIndex.fromEntries entries).union cs = some r) (id : Nat) (hid : id < W) :
    (∃ a, SameAttr a cs) ∧
    (id ∈ r ↔ ∃ v, (v, id) ∈ entries ∧ ∃ b ∈ memberValues cs, cmpPred .eq v b = some true) := by
  unfold NumericIndex.union at hr
  cases hg : unionGo none [] cs with
  | none => simp [hg] at hr
  | some keys =>
    simp only [hg, Option.bind_some] at hr
    obtain ⟨a, _, hs, ks, hks, hkeys⟩ := unionGo_spec cs none [] keys hg
    by_cases hemp : keys.isEmpty = true
    · simp [hemp] at hr
    · simp only [hemp, Bool.false_eq_true, if_false, Option.some.injEq] at hr
      subst hr
      refine ⟨⟨a, hs⟩, ?_⟩
      rw [unionOf_mem]
      have hkeys' : keys = ks := by simpa using hkeys
      subst hkeys'
      constructor
      · rintro ⟨k, hk, hmem⟩
        obtain ⟨b, hb, hkb⟩ := (encodeAll_mem _ _ hks k).1 hk
        have hp : id ∈ (NumericIndex.fromEntries entries).point b := by
          simp only [NumericIndex.point, hkb]; exact hmem
        obtain ⟨v, hv, hc⟩ := (point_agrees entries hwf b (hgood b hb) id hid).1 hp
        exact ⟨v, hv, b, hb, hc⟩
      · rintro ⟨v, hv, b, hb, hc⟩
        have hp := (point_agrees entries hwf b (hgood b hb) id hid).2 ⟨v, hv, hc⟩
        obtain ⟨kb, hkb, _⟩ := goodBound_enc b (hgood b hb)
        simp only [NumericIndex.point, hkb] at hp
        exact ⟨kb, (encodeAll_mem _ _ hks kb).2 ⟨b, hb, hkb⟩, hp⟩

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
    and `union_of`'s eager model (`unionOf`) is exact. -/
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

/-- The iterator `union_of` builds starts at `(sorted distinct keys, None)` and its remaining output
    is `unionOf`. -/
theorem unionIter_initial (ix : NumericIndex) (keys : List Nat) :
    UState.remaining ix.tree (sortDedup keys, none) = ix.unionOf keys := by
  simp [UState.remaining, NumericIndex.unionOf]

/-- `query` dispatches `Equal` to `point`, `Range` to `range`, an `IN` list to `in_list`, an `OR` to
    `union`, anything else to `None`. -/
theorem query_spec (ix : NumericIndex) (q : IndexQuery) :
    ix.query q = match q with
      | .equal _ v => some (ix.point v)
      | .range _ a b c d => some (ix.range a b c d)
      | .inList _ (some items) => ix.inList items
      | .inList _ none => none
      | .or cs => ix.union cs
      | .other => none := by
  cases q <;> (try rename_i l; cases l) <;> rfl

/-! ## Writes -/

theorem add_spec (ix : NumericIndex) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.add v id).tree ↔ e ∈ ix.tree ∨ ∃ k, encodeNumeric v = some k ∧ e = enc k id := by
  unfold NumericIndex.add
  cases h : encodeNumeric v <;> simp [tInsert, mem_sins]

theorem remove_spec (ix : NumericIndex) (hs : Sorted ix.tree) (v : Value) (id : Nat) (e : Nat) :
    e ∈ (ix.remove v id).tree ↔ e ∈ ix.tree ∧ ∀ k, encodeNumeric v = some k → e ≠ enc k id := by
  unfold NumericIndex.remove
  cases h : encodeNumeric v <;> simp [tRemove, mem_erase_sorted _ _ hs]

theorem addBatch_spec (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) (e : Nat) :
    e ∈ (ix.addBatch entries scratch).1.tree ↔ e ∈ ix.tree ∨ e ∈ encodePairs entries := by
  simp [NumericIndex.addBatch, encodePairsInto, mem_insertBatch, (sortDedup_spec _).2]

theorem removeBatch_spec (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) (e : Nat) :
    e ∈ (ix.removeBatch entries scratch).1.tree ↔ e ∈ ix.tree ∧ e ∉ encodePairs entries := by
  simp [NumericIndex.removeBatch, encodePairsInto, mem_removeBatch, (sortDedup_spec _).2]

theorem addEncoded_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.addEncoded tuples).1.tree ↔ e ∈ ix.tree ∨ e ∈ tuples.scalar := by
  unfold NumericIndex.addEncoded
  by_cases h : tIsEmpty ix.tree = true
  · have : ix.tree = [] := by simpa [tIsEmpty] using h
    simp only [h, if_true, tFromSorted, EncodedTuples.sortDedup]
    simp [this, (sortDedup_spec _).2]
  · simp only [h, if_false, EncodedTuples.sortDedup]
    simp [mem_insertBatch, (sortDedup_spec _).2]

/-- Both branches (a bulk build into an empty tree, a merge into a non-empty one) keep the tree
    sorted. -/
theorem addEncoded_sorted (ix : NumericIndex) (hs : Sorted ix.tree) (tuples : EncodedTuples) :
    Sorted (ix.addEncoded tuples).1.tree := by
  unfold NumericIndex.addEncoded
  by_cases h : tIsEmpty ix.tree = true
  · simp only [h, if_true, tFromSorted]; exact (sortDedup_spec _).1
  · simp only [h, if_false]; exact sorted_insertBatch _ _ hs

/-- The tuples a call hands back are its sorted, deduplicated input. -/
theorem addEncoded_tuples (ix : NumericIndex) (tuples : EncodedTuples) :
    (ix.addEncoded tuples).2 = tuples.sortDedup := rfl

theorem removeEncoded_spec (ix : NumericIndex) (tuples : EncodedTuples) (e : Nat) :
    e ∈ (ix.removeEncoded tuples).1.tree ↔ e ∈ ix.tree ∧ e ∉ tuples.scalar := by
  simp [NumericIndex.removeEncoded, mem_removeBatch, EncodedTuples.sortDedup, (sortDedup_spec _).2]

/-- `EncodedTuples`' accessors (`numeric.rs:101-127`). -/
theorem encodedTuples_spec (s : List Nat) (t : EncodedTuples) :
    (EncodedTuples.scalars s).scalar = s ∧ (t.isEmpty = true ↔ t.scalar = []) ∧ t.len = t.scalar.length ∧
    t.clear.scalar = [] ∧ Sorted t.sortDedup.scalar ∧ ∀ e, e ∈ t.sortDedup.scalar ↔ e ∈ t.scalar :=
  ⟨rfl, by simp [EncodedTuples.isEmpty], rfl, rfl, (sortDedup_spec _).1, (sortDedup_spec _).2⟩

theorem fromEntries_spec (entries : List (Value × Nat)) (e : Nat) :
    e ∈ (NumericIndex.fromEntries entries).tree ↔ e ∈ encodePairs entries := by
  simp [NumericIndex.fromEntries, tFromSorted, (sortDedup_spec _).2]

theorem fromEntries_sorted (entries : List (Value × Nat)) : Sorted (NumericIndex.fromEntries entries).tree :=
  (sortDedup_spec _).1

theorem isEmpty_spec (ix : NumericIndex) : ix.isEmpty = true ↔ ix.tree = [] := by
  simp [NumericIndex.isEmpty, tIsEmpty]

theorem empty_spec (ix : NumericIndex) : ix.empty = [] := tRange_empty _

theorem new_spec : NumericIndex.new.tree = [] := rfl

/-- `encode_pairs_into` clears `out` first: the old contents never leak. -/
theorem encodePairsInto_clears (entries : List (Value × Nat)) (out : List Nat) :
    encodePairsInto entries out = encodePairs entries := rfl

/-- Temporal values never reach the tree (`numeric.rs:152-163`, `encode.rs:29-41`). -/
theorem temporal_not_added (ix : NumericIndex) (i : Int) (id : Nat) :
    (ix.add (.date i) id).tree = ix.tree := by
  simp [NumericIndex.add, encodeNumeric]

end NI
