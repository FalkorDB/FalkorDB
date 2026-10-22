import FalkorNativeIndex.Tree
set_option linter.unusedSimpArgs false

/-!
# `NumericIndex` (`graph/src/index/falkordb/numeric.rs`)

The index is its tree (`tree: Tree`, `numeric.rs:39-41`), here the tree's content list (`Tree.lean`).
Each function is modelled line by line. Theorems:
* `range_keys` — the inclusive key bounds `range` computes (`numeric.rs:192-226`) select exactly the
  keys satisfying the value bounds, including the exclusive `k ± 1` steps and their `checked_*`
  guards (no stored key is `0` or `u64::MAX`, `encodeF64_bounds`).
* `range_agrees`, `point_agrees` — **index scan ≡ label scan + filter**: a column built from
  `(value, id)` entries returns exactly the ids whose value passes the Filter, for `Bool`-free data and
  guarded numeric bounds.
* `add_spec`, `remove_spec`, `addBatch_spec`, `removeBatch_spec`, `addEncoded_spec`,
  `removeEncoded_spec`, `fromEntries_spec`: contents after each write.
-/

namespace NI

/-- `NumericIndex { tree }`. -/
structure NumericIndex where
  tree : List Nat
deriving Repr

/-- `NumericIndex::new` / `Default` (`numeric.rs:46-48`). -/
def NumericIndex.new : NumericIndex := ⟨[]⟩

/-- The `(key, id)` entries of `(value, id)` pairs: the `filter_map` of `numeric.rs:58`, `:136`. -/
def encodePairs (entries : List (Value × Nat)) : List Nat :=
  entries.filterMap (fun (v, id) => (encodeNumeric v).map (fun k => enc k id))

/-- `from_entries` (`numeric.rs:55-65`): encode, `sort_unstable`, `dedup`, `from_sorted`. -/
def NumericIndex.fromEntries (entries : List (Value × Nat)) : NumericIndex :=
  ⟨tFromSorted (sortDedup (encodePairs entries))⟩

/-- `add` (`numeric.rs:69-80`). -/
def NumericIndex.add (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  match encodeNumeric v with
  | some k => ⟨tInsert ix.tree (enc k id)⟩
  | none => ix

/-- `remove` (`numeric.rs:83-91`). -/
def NumericIndex.remove (ix : NumericIndex) (v : Value) (id : Nat) : NumericIndex :=
  match encodeNumeric v with
  | some k => ⟨tRemove ix.tree (enc k id)⟩
  | none => ix

/-- `encode_pairs_into` (`numeric.rs:129-137`): `out` is cleared, then filled. Returns the new `out`. -/
def encodePairsInto (entries : List (Value × Nat)) (_out : List Nat) : List Nat :=
  encodePairs entries

/-- `add_batch` (`numeric.rs:100-108`); returns the index and the scratch buffer. -/
def NumericIndex.addBatch (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) :
    NumericIndex × List Nat :=
  let scratch := sortDedup (encodePairsInto entries scratch)   -- `sort_unstable` (dups harmless: union)
  (⟨tInsertBatch ix.tree scratch⟩, scratch)

/-- `remove_batch` (`numeric.rs:114-122`). -/
def NumericIndex.removeBatch (ix : NumericIndex) (entries : List (Value × Nat)) (scratch : List Nat) :
    NumericIndex × List Nat :=
  let scratch := sortDedup (encodePairsInto entries scratch)
  (⟨tRemoveBatch ix.tree scratch⟩, scratch)

/-- `add_encoded` (`numeric.rs:142-148`): `pairs.sort_unstable()` then `insert_batch`. -/
def NumericIndex.addEncoded (ix : NumericIndex) (pairs : List Nat) : NumericIndex × List Nat :=
  let pairs := sortDedup pairs
  (⟨tInsertBatch ix.tree pairs⟩, pairs)

/-- `remove_encoded` (`numeric.rs:151-157`). -/
def NumericIndex.removeEncoded (ix : NumericIndex) (pairs : List Nat) : NumericIndex × List Nat :=
  let pairs := sortDedup pairs
  (⟨tRemoveBatch ix.tree pairs⟩, pairs)

/-- `is_empty` (`numeric.rs:161-163`). -/
def NumericIndex.isEmpty (ix : NumericIndex) : Bool := tIsEmpty ix.tree

/-- `empty` (`numeric.rs:252-254`): `range(1, 0)`. -/
def NumericIndex.empty (ix : NumericIndex) : List Nat := tRange ix.tree 1 0

/-- `point` (`numeric.rs:167-175`). -/
def NumericIndex.point (ix : NumericIndex) (v : Value) : List Nat :=
  match encodeNumeric v with
  | some k => tRange ix.tree k k
  | none => ix.empty

/-- `range` (`numeric.rs:181-228`), branch for branch (`none` = an early `return self.empty()`). -/
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

/-- `IndexQuery<Value>` (`index/mod.rs`), the variants the native index distinguishes. -/
inductive IndexQuery where
  | equal (key : String) (value : Value)
  | range (key : String) (min max : Option Value) (incMin incMax : Bool)
  | other
deriving Repr

/-- `query` (`numeric.rs:234-249`). -/
def NumericIndex.query (ix : NumericIndex) : IndexQuery → Option (List Nat)
  | .equal _ v => some (ix.point v)
  | .range _ min max a b => some (ix.range min max a b)
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

/-- `query` dispatches `Equal` to `point`, `Range` to `range`, anything else to `None`. -/
theorem query_spec (ix : NumericIndex) (q : IndexQuery) :
    ix.query q = match q with
      | .equal _ v => some (ix.point v)
      | .range _ a b c d => some (ix.range a b c d)
      | .other => none := by
  cases q <;> rfl

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

theorem addEncoded_spec (ix : NumericIndex) (pairs : List Nat) (e : Nat) :
    e ∈ (ix.addEncoded pairs).1.tree ↔ e ∈ ix.tree ∨ e ∈ pairs := by
  simp [NumericIndex.addEncoded, mem_insertBatch, (sortDedup_spec _).2]

theorem removeEncoded_spec (ix : NumericIndex) (pairs : List Nat) (e : Nat) :
    e ∈ (ix.removeEncoded pairs).1.tree ↔ e ∈ ix.tree ∧ e ∉ pairs := by
  simp [NumericIndex.removeEncoded, mem_removeBatch, (sortDedup_spec _).2]

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

/-- Temporal values never reach the tree (`numeric.rs:349-377`). -/
theorem temporal_not_added (ix : NumericIndex) (i : Int) (id : Nat) :
    (ix.add (.date i) id).tree = ix.tree := by
  simp [NumericIndex.add, encodeNumeric]

end NI
