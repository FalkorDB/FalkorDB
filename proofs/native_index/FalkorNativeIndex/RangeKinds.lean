import FalkorNativeIndex.Tag
import FalkorNativeIndex.Memory

/-!
# `RangeIndex` over its kinds (#3738, #2527, `range.rs`)

Since #3738 `RangeIndex { numeric: NumericIndex, tag: TagIndex }`, and since #2527 also
`geo: GeoIndex`; each of its whole-column operations runs the numeric kind, then the tag kind, then
the geo kind (`let Self { numeric, tag, geo } = self`):
`new` (`Default`), `for_each_stored_key`, `push_encoded`, `add_encoded`, `remove_encoded`,
`is_empty`, `memory_usage`. `EncodedTuples` carries the numeric kinds' `u64` tuples (`by_kind`) and
the tag kinds' string tuples (`tags`); `sort_dedup` sorts only `by_kind`, which holds the geo
kind's tuples too (`KeyKind::Geo`, here the field `geo`).

The geo kind (`geo.rs`) is one sorted tree of `enc key doc` tuples, `key` a point's Morton key
(`Value.point (some key)`; a `NaN` point and every other value store nothing): its `add_encoded` /
`remove_encoded` are `add_sorted` / `remove_sorted`, the numeric kind's `addTree` / `removeTree`.

The numeric kind is the existing `NumericIndex` model. The tag trees (`StrTree`) are not modelled
(their `tag.rs` rows are NOT COVERED), so the tag kind's operations are parameters, as in
`TagSel.lookupT`. Proven: each operation is the old numeric model on the numeric share, so every
numeric theorem (`addEncoded_spec`, `removeEncoded_spec`, `isEmpty_spec`, `pushStored_scalar`, …)
still describes it, and the tag and geo kinds add exactly their own shares.
-/

namespace NI
namespace RangeKinds

/-- `EncodedTuples` (`numeric.rs`): the numeric kinds' tuples and the tag kinds' tuples. -/
structure Buf where
  num : EncodedTuples
  tag : List (String × Nat)
  tagArray : List (String × Nat)
  geo : List Nat := []

/-- `RangeIndex { numeric, tag }` (`range.rs`), the tag kind's state `T` left abstract. -/
structure RangeK (T : Type) where
  numeric : NumericIndex
  tag : T
  geo : List Nat := []

variable {T : Type}

/-- `RangeIndex::new` (`Default`): every kind empty. -/
def new (tagNew : T) : RangeK T := ⟨NumericIndex.new, tagNew, []⟩

/-- The tag kind's stored keys (`TagIndex::for_each_stored_key`): a parameter. -/
abbrev TagKeys := Value → List (Tag.KeyKind × String)

/-- A numeric key kind in the #3738 `KeyKind`. -/
def liftKind : KeyKind → Tag.KeyKind
  | .scalar => .scalar
  | .array => .array

/-- `GeoIndex::key_of`: a point's Morton key; `None` for a `NaN` point and any other value. -/
def geoKey : Value → Option Nat
  | .point k => k
  | _ => none

/-- `RangeIndex::for_each_stored_key`: the numeric kind's keys (`ColumnKey::Encoded`), then the
    tag kind's (`ColumnKey::Tag`), then the geo kind's (`KeyKind::Geo`, `ColumnKey::Encoded`). -/
def storedKeys (tagKeys : TagKeys) (v : Value) : List (Tag.KeyKind × Tag.ColumnKey) :=
  (forEachStoredKey v).map (fun kk => (liftKind kk.1, .encoded kk.2)) ++
    (tagKeys v).map (fun kk => (kk.1, .tag kk.2)) ++
    (geoKey v).toList.map (fun k => (.geo, .encoded k))

/-- `EncodedTuples::push` of one stored key. -/
def pushKey (b : Buf) (id : Nat) : Tag.KeyKind × Tag.ColumnKey → Buf
  | (.scalar, .encoded k) => { b with num := b.num.push .scalar (enc k id) }
  | (.array, .encoded k) => { b with num := b.num.push .array (enc k id) }
  | (.tag, .tag s) => { b with tag := b.tag ++ [(s, id)] }
  | (.tagArray, .tag s) => { b with tagArray := b.tagArray ++ [(s, id)] }
  | (.geo, .encoded k) => { b with geo := b.geo ++ [enc k id] }
  | _ => b

/-- `RangeIndex::push_encoded`: `for_each_stored_key`, each key pushed onto its kind. -/
def pushEncoded (tagKeys : TagKeys) (v : Value) (id : Nat) (b : Buf) : Buf :=
  (storedKeys tagKeys v).foldl (fun o kk => pushKey o id kk) b

/-- `RangeIndex::add_encoded`: `sort_dedup` (the numeric share), then each kind's `add_encoded`. -/
def addEncoded (tagAdd : T → Buf → T) (r : RangeK T) (b : Buf) : RangeK T × Buf :=
  let b := { b with num := b.num.sortDedup, geo := sortDedup b.geo }
  (⟨r.numeric.addTrees b.num, tagAdd r.tag b, addTree r.geo b.geo⟩, b)

/-- `RangeIndex::remove_encoded`. -/
def removeEncoded (tagRemove : T → Buf → T) (r : RangeK T) (b : Buf) : RangeK T × Buf :=
  let b := { b with num := b.num.sortDedup, geo := sortDedup b.geo }
  (⟨r.numeric.removeTrees b.num, tagRemove r.tag b, removeTree r.geo b.geo⟩, b)

/-- `RangeIndex::is_empty`: no tuple in any kind. -/
def isEmpty (tagEmpty : T → Bool) (r : RangeK T) : Bool :=
  r.numeric.isEmpty && tagEmpty r.tag && tIsEmpty r.geo

/-- `RangeIndex::memory_usage`: every kind's heap bytes, `ht` a tree's `heap_bytes`. -/
def memoryUsage (ht : List Nat → Nat) (tagMem : T → Nat) (r : RangeK T) : Nat :=
  r.numeric.memoryUsage (fun i => ht i.tree + ht i.arrayTree) + tagMem r.tag + ht r.geo

theorem new_spec (tagNew : T) :
    (new tagNew).numeric.tree = [] ∧ (new tagNew).numeric.arrayTree = [] ∧ (new tagNew).tag = tagNew ∧
      (new tagNew).geo = [] :=
  ⟨rfl, rfl, rfl, rfl⟩

private theorem foldl_pushKey_num (id : Nat) :
    ∀ (l : List (KeyKind × Nat)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (liftKind kk.1, .encoded kk.2)) b).num =
        l.foldl (fun o kk => o.push kk.1 (enc kk.2 id)) b.num
  | [], _ => rfl
  | (k, n) :: l, b => by
    simp only [List.foldl_cons]
    rw [foldl_pushKey_num id l]
    cases k <;> rfl

private theorem foldl_pushKey_tags_num (id : Nat) :
    ∀ (l : List (Tag.KeyKind × String)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (kk.1, .tag kk.2)) b).num = b.num
  | [], _ => rfl
  | (k, s) :: l, b => by
    simp only [List.foldl_cons]
    rw [foldl_pushKey_tags_num id l]
    cases k <;> rfl

private theorem foldl_pushKey_num_tag (id : Nat) :
    ∀ (l : List (KeyKind × Nat)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (liftKind kk.1, .encoded kk.2)) b).tag = b.tag ∧
      (l.foldl (fun o kk => pushKey o id (liftKind kk.1, .encoded kk.2)) b).tagArray = b.tagArray
  | [], _ => ⟨rfl, rfl⟩
  | (k, n) :: l, b => by
    simp only [List.foldl_cons]
    have := foldl_pushKey_num_tag id l (pushKey b id (liftKind k, .encoded n))
    rw [this.1, this.2]
    cases k <;> exact ⟨rfl, rfl⟩

private theorem foldl_pushKey_tags (id : Nat) :
    ∀ (l : List (Tag.KeyKind × String)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (kk.1, .tag kk.2)) b).tag =
          b.tag ++ (l.filter (·.1 == .tag)).map (fun kk => (kk.2, id)) ∧
      (l.foldl (fun o kk => pushKey o id (kk.1, .tag kk.2)) b).tagArray =
          b.tagArray ++ (l.filter (·.1 == .tagArray)).map (fun kk => (kk.2, id))
  | [], _ => by simp
  | (k, s) :: l, b => by
    simp only [List.foldl_cons]
    have := foldl_pushKey_tags id l (pushKey b id (k, .tag s))
    rw [this.1, this.2]
    cases k <;> simp [pushKey]

private theorem foldl_pushKey_num_geo (id : Nat) :
    ∀ (l : List (KeyKind × Nat)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (liftKind kk.1, .encoded kk.2)) b).geo = b.geo
  | [], _ => rfl
  | (k, n) :: l, b => by
    simp only [List.foldl_cons]
    rw [foldl_pushKey_num_geo id l]
    cases k <;> rfl

private theorem foldl_pushKey_tags_geo (id : Nat) :
    ∀ (l : List (Tag.KeyKind × String)) (b : Buf),
      (l.foldl (fun o kk => pushKey o id (kk.1, .tag kk.2)) b).geo = b.geo
  | [], _ => rfl
  | (k, s) :: l, b => by
    simp only [List.foldl_cons]
    rw [foldl_pushKey_tags_geo id l]
    cases k <;> rfl

private theorem foldl_pushKey_geo (id : Nat) :
    ∀ (l : List Nat) (b : Buf),
      (l.foldl (fun o k => pushKey o id (.geo, .encoded k)) b).geo = b.geo ++ l.map (enc · id) ∧
      (l.foldl (fun o k => pushKey o id (.geo, .encoded k)) b).num = b.num ∧
      (l.foldl (fun o k => pushKey o id (.geo, .encoded k)) b).tag = b.tag ∧
      (l.foldl (fun o k => pushKey o id (.geo, .encoded k)) b).tagArray = b.tagArray
  | [], _ => by simp
  | k :: l, b => by
    simp only [List.foldl_cons]
    have := foldl_pushKey_geo id l (pushKey b id (.geo, .encoded k))
    rw [this.1, this.2.1, this.2.2.1, this.2.2.2]
    simp [pushKey]

/-- **The numeric share of `push_encoded` is the old `pushStored`** (so `pushStored_scalar` /
    `pushStored_array` still describe it): the tag kind's keys never touch it. -/
theorem pushEncoded_num (tagKeys : TagKeys) (v : Value) (id : Nat) (b : Buf) :
    (pushEncoded tagKeys v id b).num = pushStored v id b.num := by
  simp only [pushEncoded, storedKeys, List.foldl_append, List.foldl_map]
  rw [(foldl_pushKey_geo id _ _).2.1, foldl_pushKey_tags_num, foldl_pushKey_num]
  rfl

/-- The tag share of `push_encoded`: one `(string, id)` per key the tag kind stores, by kind. -/
theorem pushEncoded_tag (tagKeys : TagKeys) (v : Value) (id : Nat) (b : Buf) :
    (pushEncoded tagKeys v id b).tag =
        b.tag ++ ((tagKeys v).filter (·.1 == .tag)).map (fun kk => (kk.2, id)) ∧
    (pushEncoded tagKeys v id b).tagArray =
        b.tagArray ++ ((tagKeys v).filter (·.1 == .tagArray)).map (fun kk => (kk.2, id)) := by
  simp only [pushEncoded, storedKeys, List.foldl_append, List.foldl_map]
  rw [(foldl_pushKey_geo id _ _).2.2.1, (foldl_pushKey_geo id _ _).2.2.2,
    (foldl_pushKey_tags id _ _).1, (foldl_pushKey_tags id _ _).2,
    (foldl_pushKey_num_tag id _ _).1, (foldl_pushKey_num_tag id _ _).2]
  exact ⟨rfl, rfl⟩

/-- **The geo share of `push_encoded`: one `enc key id` for a point's Morton key**, nothing for any
    other value (a `NaN` point, a list of points). -/
theorem pushEncoded_geo (tagKeys : TagKeys) (v : Value) (id : Nat) (b : Buf) :
    (pushEncoded tagKeys v id b).geo = b.geo ++ (geoKey v).toList.map (enc · id) := by
  simp only [pushEncoded, storedKeys, List.foldl_append, List.foldl_map]
  rw [(foldl_pushKey_geo id _ _).1, foldl_pushKey_tags_geo, foldl_pushKey_num_geo]

/-- `for_each_stored_key`'s geo key is the point's Morton key, and only a point's. -/
theorem storedKeys_geo (tagKeys : TagKeys) (v : Value) (n : Nat) :
    (Tag.KeyKind.geo, Tag.ColumnKey.encoded n) ∈ storedKeys tagKeys v ↔ v = .point (some n) := by
  simp only [storedKeys, List.mem_append, List.mem_map]
  constructor
  · rintro ((⟨⟨k', n'⟩, _, he⟩ | ⟨_, _, he⟩) | ⟨k, hk, he⟩)
    · cases k' <;> simp [liftKind] at he
    · simp at he
    · simp only [Prod.mk.injEq, Tag.ColumnKey.encoded.injEq, true_and] at he
      subst he
      cases v <;> simp_all [geoKey]
  · rintro rfl
    exact Or.inr ⟨n, by simp [geoKey], rfl⟩

/-- `for_each_stored_key`'s numeric keys are exactly the old ones (`forEachStoredKey_scalar/_array`). -/
theorem storedKeys_encoded (tagKeys : TagKeys) (v : Value) (k : KeyKind) (n : Nat) :
    (liftKind k, Tag.ColumnKey.encoded n) ∈ storedKeys tagKeys v ↔ (k, n) ∈ forEachStoredKey v := by
  simp only [storedKeys, List.mem_append, List.mem_map]
  constructor
  · rintro ((⟨⟨k', n'⟩, h, he⟩ | ⟨_, _, he⟩) | ⟨_, _, he⟩)
    · simp only [Prod.mk.injEq, Tag.ColumnKey.encoded.injEq] at he
      obtain ⟨hk, rfl⟩ := he
      cases k <;> cases k' <;> simp_all [liftKind]
    · simp at he
    · cases k <;> simp [liftKind] at he
  · intro h
    exact Or.inl (Or.inl ⟨(k, n), h, rfl⟩)

/-- **`add_encoded` is the old numeric `addEncoded` on the numeric share** (`addEncoded_spec`,
    `addEncoded_array_spec`, `addEncoded_sorted` carry over), and hands the tag kind the tag
    tuples unchanged. -/
theorem addEncoded_numeric (tagAdd : T → Buf → T) (r : RangeK T) (b : Buf) :
    (addEncoded tagAdd r b).1.numeric = (r.numeric.addEncoded b.num).1 ∧
    (addEncoded tagAdd r b).2.num = (r.numeric.addEncoded b.num).2 ∧
    (addEncoded tagAdd r b).1.tag = tagAdd r.tag (addEncoded tagAdd r b).2 ∧
    (addEncoded tagAdd r b).2.tag = b.tag ∧ (addEncoded tagAdd r b).2.tagArray = b.tagArray :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **`remove_encoded` is the old numeric `removeEncoded` on the numeric share.** -/
theorem removeEncoded_numeric (tagRemove : T → Buf → T) (r : RangeK T) (b : Buf) :
    (removeEncoded tagRemove r b).1.numeric = (r.numeric.removeEncoded b.num).1 ∧
    (removeEncoded tagRemove r b).2.num = (r.numeric.removeEncoded b.num).2 ∧
    (removeEncoded tagRemove r b).1.tag = tagRemove r.tag (removeEncoded tagRemove r b).2 ∧
    (removeEncoded tagRemove r b).2.tag = b.tag ∧ (removeEncoded tagRemove r b).2.tagArray = b.tagArray :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem addEncoded_mem (tagAdd : T → Buf → T) (r : RangeK T) (b : Buf) (e : Nat) :
    (e ∈ (addEncoded tagAdd r b).1.numeric.tree ↔ e ∈ r.numeric.tree ∨ e ∈ b.num.scalar) ∧
    (e ∈ (addEncoded tagAdd r b).1.numeric.arrayTree ↔ e ∈ r.numeric.arrayTree ∨ e ∈ b.num.array) :=
  ⟨addEncoded_spec r.numeric b.num e, addEncoded_array_spec r.numeric b.num e⟩

theorem removeEncoded_mem (tagRemove : T → Buf → T) (r : RangeK T) (b : Buf) (e : Nat) :
    (e ∈ (removeEncoded tagRemove r b).1.numeric.tree ↔ e ∈ r.numeric.tree ∧ e ∉ b.num.scalar) ∧
    (e ∈ (removeEncoded tagRemove r b).1.numeric.arrayTree ↔
      e ∈ r.numeric.arrayTree ∧ e ∉ b.num.array) :=
  ⟨removeEncoded_spec r.numeric b.num e, removeEncoded_array_spec r.numeric b.num e⟩

/-- **`add_encoded` adds the batch's geo tuples to the geo tree**, `remove_encoded` removes them. -/
theorem addEncoded_geo_mem (tagAdd : T → Buf → T) (r : RangeK T) (b : Buf) (e : Nat) :
    e ∈ (addEncoded tagAdd r b).1.geo ↔ e ∈ r.geo ∨ e ∈ b.geo := by
  simp only [addEncoded, addTree_spec, (sortDedup_spec _).2]

theorem removeEncoded_geo_mem (tagRemove : T → Buf → T) (r : RangeK T) (b : Buf) (e : Nat) :
    e ∈ (removeEncoded tagRemove r b).1.geo ↔ e ∈ r.geo ∧ e ∉ b.geo := by
  simp only [removeEncoded, removeTree_spec, (sortDedup_spec _).2]

/-- **`is_empty` holds iff both numeric trees are empty, the tag kind is, and the geo tree is.** -/
theorem isEmpty_iff (tagEmpty : T → Bool) (r : RangeK T) :
    isEmpty tagEmpty r = true ↔
      r.numeric.tree = [] ∧ r.numeric.arrayTree = [] ∧ tagEmpty r.tag = true ∧ r.geo = [] := by
  simp only [isEmpty, Bool.and_eq_true, isEmpty_spec, and_assoc, tIsEmpty, List.isEmpty_iff]

/-- A column holding a string is not empty, whatever its numeric trees: the old
    numeric-only `is_empty` would have said empty. -/
theorem isEmpty_tag_nonempty (tagEmpty : T → Bool) (r : RangeK T) (h : tagEmpty r.tag = false) :
    isEmpty tagEmpty r = false := by
  simp [isEmpty, h]

/-- **`memory_usage` is the numeric kind's two trees (`memoryUsage_two_trees`) plus the tag kind's
    plus the geo tree's.** -/
theorem memoryUsage_kinds (ht : List Nat → Nat) (tagMem : T → Nat) (r : RangeK T) :
    memoryUsage ht tagMem r = ht r.numeric.tree + ht r.numeric.arrayTree + tagMem r.tag + ht r.geo := by
  simp only [memoryUsage, memoryUsage_two_trees]

end RangeKinds
end NI
