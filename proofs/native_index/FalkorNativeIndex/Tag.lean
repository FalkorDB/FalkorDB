import FalkorNativeIndex.Decline
import FalkorNativeIndex.Pending
set_option linter.unusedSimpArgs false

/-!
# The tag (string) kind of a Range column (#3738): the parts that are not tag-tree internals

Since #3738 a Range column has a numeric and a tag kind (`range.rs`, `tag.rs`). A staged change, an
`EncodedTuples` and an equality lookup carry a `ColumnKey`: a number's encoding or the string itself.
The numeric path is unchanged (the older models, keyed on `Nat`, still describe it); this file models
what changed around it:

* `encodeValueK` = `RangeIndex::encode_value` (`range.rs`, called by
  `IndexColumn::encode_value`, `falkordb_index.rs`): a string is its own tag key; every other
  value is the old numeric encoding (`encodeValueK_numeric`, `encodeValueK_string`).
* `ET` = `EncodedTuples` (`numeric.rs`), `u64` tuples per numeric kind and string tuples per
  tag kind: `is_empty` / `len` / `clear` / `push` / `retain_docs` count every kind (`et_spec`).
* `lastWriteWinsT` = `last_write_wins` (`pending_writes.rs`) for any tuple type and any total
  order (`sort_by` on `(kind, &key, id)`, a derived `Ord`): each tuple keeps exactly its last staged
  change (`lastWriteWinsT_spec`, the generic `lastWriteWins_spec`).
* `describeK` = `describe_predicate` (`mod.rs`, `indexable`, `union_unindexable`): a decline names the operand whose type no kind holds (`describeK_equal`,
  `describeK_or`).
-/

namespace NI
namespace Tag

/-! ## Keys -/

/-- `KeyKind` (`encode.rs`): the tree a tuple lives in. -/
inductive KeyKind where
  | scalar | array | tag | tagArray
deriving DecidableEq, Repr

/-- `ColumnKey` (`encode.rs`): a number's encoding or the string a tag tree is keyed on. -/
inductive ColumnKey where
  | encoded (k : Nat)
  | tag (s : String)
deriving DecidableEq, Repr

/-- `indexable` (`mod.rs`): some kind holds a value of this type. -/
def indexable : Value → Bool
  | .int _ | .float _ | .bool _ | .string _ => true
  | _ => false

/-- `RangeIndex::encode_value` (`range.rs`) through `Kind::of` (`range.rs`). -/
def encodeValueK : Value → Option (KeyKind × ColumnKey)
  | .string s => some (.tag, .tag s)
  | v@(.int _) | v@(.float _) | v@(.bool _) => (encodeNumeric v).map (fun k => (.scalar, .encoded k))
  | _ => none

/-- A string is keyed by itself in the tag tree. -/
theorem encodeValueK_string (s : String) : encodeValueK (.string s) = some (.tag, .tag s) := rfl

/-- **Every non-string value takes exactly the key it took before #3738** (the old `encodeValue`,
    `encodeValue_spec`), in the scalar tree. -/
theorem encodeValueK_numeric (v : Value) (hv : ∀ s, v ≠ .string s) :
    encodeValueK v = (encodeNumeric v).map (fun k => (.scalar, .encoded k)) := by
  cases v with
  | string s => exact absurd rfl (hv s)
  | _ => simp [encodeValueK, encodeNumeric]

/-- What is not indexable has no key (`NULL`, temporals, the rest). -/
theorem encodeValueK_none (v : Value) (hv : indexable v = false) : encodeValueK v = none := by
  cases v <;> simp_all [indexable, encodeValueK]

/-! ## `EncodedTuples` with the tag kinds (`numeric.rs`) -/

/-- `by_kind` (scalar, array) and `tags` (tag, tag array). -/
structure ET where
  scalar : List (Nat × Nat)
  array : List (Nat × Nat)
  tag : List (String × Nat)
  tagArray : List (String × Nat)
deriving DecidableEq, Repr

namespace ET
def empty : ET := ⟨[], [], [], []⟩
/-- `scalars`. -/
def scalars (s : List (Nat × Nat)) : ET := { empty with scalar := s }
/-- `is_empty`. -/
def isEmpty (t : ET) : Bool := t.scalar.isEmpty && t.array.isEmpty && t.tag.isEmpty && t.tagArray.isEmpty
/-- `len`. -/
def len (t : ET) : Nat := t.scalar.length + t.array.length + (t.tag.length + t.tagArray.length)
/-- `clear`. -/
def clear (_t : ET) : ET := empty
/-- `push`; a key of the other kind's type is `unreachable!` (`none`). -/
def push (t : ET) : KeyKind → ColumnKey → Nat → Option ET
  | .scalar, .encoded k, d => some { t with scalar := t.scalar ++ [(k, d)] }
  | .array, .encoded k, d => some { t with array := t.array ++ [(k, d)] }
  | .tag, .tag s, d => some { t with tag := t.tag ++ [(s, d)] }
  | .tagArray, .tag s, d => some { t with tagArray := t.tagArray ++ [(s, d)] }
  | _, _, _ => none
/-- `retain_docs`. -/
def retainDocs (t : ET) (keep : Nat → Bool) : ET :=
  ⟨t.scalar.filter (keep ·.2), t.array.filter (keep ·.2), t.tag.filter (keep ·.2), t.tagArray.filter (keep ·.2)⟩
end ET

/-- **`is_empty`, `len`, `clear` count every kind, the tag kinds included; `push` files a tuple in its
    kind's list; `retain_docs` filters every kind by doc.** -/
theorem et_spec (t : ET) :
    (t.isEmpty = true ↔ t.scalar = [] ∧ t.array = [] ∧ t.tag = [] ∧ t.tagArray = []) ∧
    t.len = t.scalar.length + t.array.length + t.tag.length + t.tagArray.length ∧
    t.clear.isEmpty = true ∧ t.clear.len = 0 ∧
    (t.isEmpty = true ↔ t.len = 0) ∧
    (∀ s d, (t.push .tag (.tag s) d).map ET.len = some (t.len + 1)) ∧
    (∀ s d, (t.push .tagArray (.tag s) d).map ET.len = some (t.len + 1)) ∧
    (∀ k d, (t.push .scalar (.encoded k) d).map ET.len = some (t.len + 1)) ∧
    (∀ k d, (t.push .array (.encoded k) d).map ET.len = some (t.len + 1)) ∧
    (∀ keep, (∀ x ∈ (t.retainDocs keep).tag, keep x.2 = true) ∧ (∀ x ∈ (t.retainDocs keep).scalar, keep x.2 = true) ∧
      (∀ x ∈ (t.retainDocs keep).tagArray, keep x.2 = true) ∧ (∀ x ∈ (t.retainDocs keep).array, keep x.2 = true)) := by
  refine ⟨by simp [ET.isEmpty, and_assoc], by simp only [ET.len]; omega, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [ET.isEmpty, ET.len, Bool.and_eq_true, List.isEmpty_iff]
    constructor
    · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩; simp [h1, h2, h3, h4]
    · intro h
      refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> (apply List.eq_nil_of_length_eq_zero; omega)
  · intro s d; simp [ET.push, ET.len]; omega
  · intro s d; simp [ET.push, ET.len]; omega
  · intro k d; simp [ET.push, ET.len]; omega
  · intro k d; simp [ET.push, ET.len]; omega
  · intro keep; refine ⟨?_, ?_, ?_, ?_⟩ <;> intro x hx <;> simp [ET.retainDocs] at hx <;> exact hx.2

/-- `retain_docs` keeps exactly the tuples whose doc passes, in every kind. -/
theorem et_retainDocs_mem (t : ET) (keep : Nat → Bool) :
    (∀ x, x ∈ (t.retainDocs keep).scalar ↔ x ∈ t.scalar ∧ keep x.2 = true) ∧
    (∀ x, x ∈ (t.retainDocs keep).array ↔ x ∈ t.array ∧ keep x.2 = true) ∧
    (∀ x, x ∈ (t.retainDocs keep).tag ↔ x ∈ t.tag ∧ keep x.2 = true) ∧
    (∀ x, x ∈ (t.retainDocs keep).tagArray ↔ x ∈ t.tagArray ∧ keep x.2 = true) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intro x <;> simp [ET.retainDocs]

theorem et_scalars (s : List (Nat × Nat)) :
    (ET.scalars s).scalar = s ∧ (ET.scalars s).array = [] ∧ (ET.scalars s).tag = [] ∧ (ET.scalars s).tagArray = [] :=
  ⟨rfl, rfl, rfl, rfl⟩

/-! ## `last_write_wins` over any tuple and any total order (`pending_writes.rs`) -/

/-- A staged change reduced to its tuple `(kind, key, id)` and its side. -/
structure WriteT (T : Type) where
  tup : T
  add : Bool

section LWW
variable {T : Type} [DecidableEq T]

/-- `log.sort_by(|a, b| tuple(a).cmp(&tuple(b)))` — stable. -/
def sortLogT (le : T → T → Bool) (log : List (WriteT T)) : List (WriteT T) :=
  log.mergeSort (fun a b => le a.tup b.tup)

def lwwGoT : List (WriteT T) → List (WriteT T)
  | [] => []
  | [w] => [w]
  | w :: w' :: r => if w'.tup = w.tup then lwwGoT (w' :: r) else w :: lwwGoT (w' :: r)

/-- `(removes, adds)`. -/
def lastWriteWinsT (le : T → T → Bool) (log : List (WriteT T)) : List T × List T :=
  let kept := lwwGoT (sortLogT le log)
  ((kept.filter (fun w => !w.add)).map (·.tup), (kept.filter (·.add)).map (·.tup))

def lastOpT (log : List (WriteT T)) (e : T) : Option Bool :=
  ((log.filter (fun w => w.tup = e)).getLast?).map (·.add)

theorem lastOpT_cons (a : WriteT T) (l : List (WriteT T)) (e : T) :
    lastOpT (a :: l) e = match lastOpT l e with
      | some b => some b
      | none => if a.tup = e then some a.add else none := by
  unfold lastOpT
  by_cases h : a.tup = e
  · rw [List.filter_cons_of_pos (by simp [h]), List.getLast?_cons]
    cases (List.filter (fun w => decide (w.tup = e)) l).getLast? <;> simp [h]
  · rw [List.filter_cons_of_neg (by simp [h])]
    cases (List.filter (fun w => decide (w.tup = e)) l).getLast? <;> simp [h]

theorem lastOpT_none (l : List (WriteT T)) (e : T) (h : ∀ x ∈ l, x.tup ≠ e) : lastOpT l e = none := by
  unfold lastOpT
  rw [List.filter_eq_nil_iff.2 (by intro x hx; simpa using h x hx)]
  rfl

theorem lastOpT_head (a : WriteT T) (l : List (WriteT T)) : lastOpT (a :: l) a.tup ≠ none := by
  rw [lastOpT_cons]; cases lastOpT l a.tup <;> simp

variable (le : T → T → Bool)
  (htrans : ∀ a b c, le a b = true → le b c = true → le a c = true)
  (htotal : ∀ a b, (le a b || le b a) = true)
  (hanti : ∀ a b, le a b = true → le b a = true → a = b)
include htrans htotal

theorem sortLogT_filter (log : List (WriteT T)) (e : T) :
    (sortLogT le log).filter (fun w => w.tup = e) = log.filter (fun w => w.tup = e) := by
  have trans : ∀ a b c : WriteT T, le a.tup b.tup = true → le b.tup c.tup = true → le a.tup c.tup = true :=
    fun a b c => htrans a.tup b.tup c.tup
  have total : ∀ a b : WriteT T, (le a.tup b.tup || le b.tup a.tup) = true := fun a b => htotal a.tup b.tup
  have hrefl : ∀ x, le x x = true := fun x => by have := htotal x x; simpa using this
  have hpw : (log.filter (fun w => w.tup = e)).Pairwise (fun a b => le a.tup b.tup = true) := by
    apply List.Pairwise.imp_of_mem _ (List.pairwise_of_forall (R := fun _ _ => True) (fun _ _ => trivial))
    intro a b ha hb _
    simp [List.mem_filter] at ha hb; rw [ha.2, hb.2]; exact hrefl e
  have hsub := List.sublist_mergeSort trans total hpw (List.filter_sublist (l := log))
  have h1 := hsub.filter (fun w => decide (w.tup = e))
  rw [List.filter_filter] at h1
  simp only [Bool.and_self] at h1
  have hlen : ((sortLogT le log).filter (fun w => w.tup = e)).length = (log.filter (fun w => w.tup = e)).length :=
    ((List.mergeSort_perm log _).filter _).length_eq
  exact (h1.eq_of_length hlen.symm).symm

omit htrans htotal in
include hanti in
theorem lwwGoT_spec : ∀ (s : List (WriteT T)), s.Pairwise (fun a b => le a.tup b.tup = true) →
    ∀ (e : T) (b : Bool), (∃ w ∈ lwwGoT s, w.tup = e ∧ w.add = b) ↔ lastOpT s e = some b
  | [], _, e, b => by simp [lwwGoT, lastOpT]
  | [w], _, e, b => by
    rw [lastOpT_cons, lastOpT_none [] e (by simp)]
    simp only [lwwGoT, List.mem_singleton, exists_eq_left]
    by_cases h : w.tup = e <;> simp [h]
  | w :: w' :: r, hs, e, b => by
    have hs' : (w' :: r).Pairwise (fun a b => le a.tup b.tup = true) := hs.of_cons
    have ih := lwwGoT_spec (w' :: r) hs' e b
    have hle : ∀ x ∈ w' :: r, le w.tup x.tup = true := fun x hx => List.rel_of_pairwise_cons hs hx
    unfold lwwGoT
    rw [lastOpT_cons]
    by_cases heq : w'.tup = w.tup
    · simp only [heq, ite_true, ih]
      by_cases he : w.tup = e
      · have : lastOpT (w' :: r) e ≠ none := by rw [← he, ← heq]; exact lastOpT_head w' r
        cases h : lastOpT (w' :: r) e
        · exact absurd h this
        · simp
      · cases lastOpT (w' :: r) e <;> simp [he]
    · simp only [heq, ite_false, List.mem_cons, exists_eq_or_imp]
      have hne : ∀ x ∈ w' :: r, x.tup ≠ w.tup := by
        intro x hx hxe
        rcases List.mem_cons.1 hx with hx | hx
        · subst hx; exact heq hxe
        · have h1 : le w'.tup x.tup = true := List.rel_of_pairwise_cons hs' hx
          have h2 : le w.tup w'.tup = true := hle w' (by simp)
          rw [hxe] at h1
          exact heq (hanti _ _ h1 h2)
      by_cases he : w.tup = e
      · have hn : lastOpT (w' :: r) e = none :=
          lastOpT_none _ e (by intro x hx hxe; exact hne x hx (hxe.trans he.symm))
        rw [hn] at ih ⊢
        simp only [he, ite_true, true_and, Option.some.injEq]
        constructor
        · rintro (h | ⟨x, hx, h1, h2⟩)
          · exact h
          · exact absurd (ih.1 ⟨x, hx, h1, h2⟩) (by simp)
        · intro h; exact Or.inl h
      · simp only [he, false_and, false_or, ih]
        cases lastOpT (w' :: r) e <;> simp [he]

include hanti in
/-- **`last_write_wins` keeps exactly each tuple's last staged change**, whatever the tuple's type —
    a `(kind, ColumnKey, id)` with a string key included — given only that the sort's order is total
    (`Ord`'s contract; the derived `Ord` on `ColumnKey`/`Arc<String>` compares the strings). -/
theorem lastWriteWinsT_spec (log : List (WriteT T)) (e : T) :
    (e ∈ (lastWriteWinsT le log).2 ↔ lastOpT log e = some true) ∧
    (e ∈ (lastWriteWinsT le log).1 ↔ lastOpT log e = some false) := by
  have hs : (sortLogT le log).Pairwise (fun a b => le a.tup b.tup = true) :=
    List.pairwise_mergeSort (le := fun a b : WriteT T => le a.tup b.tup)
      (fun a b c => htrans a.tup b.tup c.tup) (fun a b => htotal a.tup b.tup) log
  have hlast : lastOpT (sortLogT le log) e = lastOpT log e := by
    unfold lastOpT; rw [sortLogT_filter le htrans htotal]
  rw [← hlast]
  constructor
  · rw [← lwwGoT_spec le hanti _ hs e true]
    simp only [lastWriteWinsT, List.mem_map, List.mem_filter]
    constructor
    · rintro ⟨w, ⟨h1, h2⟩, h3⟩; exact ⟨w, h1, h3, h2⟩
    · rintro ⟨w, h1, h3, h2⟩; exact ⟨w, ⟨h1, h2⟩, h3⟩
  · rw [← lwwGoT_spec le hanti _ hs e false]
    simp only [lastWriteWinsT, List.mem_map, List.mem_filter]
    constructor
    · rintro ⟨w, ⟨h1, h2⟩, h3⟩; exact ⟨w, h1, h3, by simpa using h2⟩
    · rintro ⟨w, h1, h3, h2⟩; exact ⟨w, ⟨h1, by simpa using h2⟩, h3⟩

end LWW

/-- The PR's test `the_last_change_to_a_string_wins` (`pending_writes.rs`): `+b`, `-a`, `-b`
    on doc 7 leave `b` and `a` removed and nothing added, for any total order. -/
theorem string_last_change_wins (le : (String × Nat) → (String × Nat) → Bool)
    (htrans : ∀ a b c, le a b = true → le b c = true → le a c = true)
    (htotal : ∀ a b, (le a b || le b a) = true)
    (hanti : ∀ a b, le a b = true → le b a = true → a = b) :
    let log : List (WriteT (String × Nat)) := [⟨("b", 7), true⟩, ⟨("a", 7), false⟩, ⟨("b", 7), false⟩]
    (∀ e, e ∉ (lastWriteWinsT le log).2) ∧
    (∀ e, e ∈ (lastWriteWinsT le log).1 ↔ e = ("a", 7) ∨ e = ("b", 7)) := by
  intro log
  have h := lastWriteWinsT_spec le htrans htotal hanti log
  refine ⟨fun e he => ?_, fun e => ?_⟩
  · have := (h e).1.1 he
    by_cases ha : e = ("a", 7)
    · subst ha; simp [lastOpT, log] at this
    · by_cases hb : e = ("b", 7)
      · subst hb; simp [lastOpT, log] at this
      · simp [lastOpT, log, Ne.symm ha, Ne.symm hb] at this
  · rw [(h e).2]
    by_cases ha : e = ("a", 7)
    · subst ha; simp [lastOpT, log]
    · by_cases hb : e = ("b", 7)
      · subst hb; simp [lastOpT, log]
      · simp [lastOpT, log, Ne.symm ha, Ne.symm hb, ha, hb]

/-! ## `describe_predicate` since #3738 (`mod.rs`) -/

/-- The operand a decline names: a value (`value.name()`), or an `IN` operand that is not a list. -/
inductive Operand where
  | value (v : Value)
  | nonList

/-- The shape, with the unindexable operand (`none` when there is none to name). -/
inductive DescK where
  | equality (key : String) (bad : Option Value)
  | range (key : String) (bad : Option Value)
  | orSpanning (n : Nat)
  | orOneKey (n : Nat) (key : String) (bad : Option Value)
  | orNonMember (n : Nat) (key : String)
  | and (children : List DescK)
  | inList (key : String) (bad : Option Operand)
  | geo (key : String)
  | arrayContains (key : String) (bad : Option Value)

/-- `!indexable(v) && !matches!(v, Null)`: a value no kind holds and that is not `NULL`. -/
def unindexable (v : Value) : Bool := !indexable v && v != .null

mutual
/-- `union_unindexable` (`mod.rs`). -/
def unionUnindexable : List Pred → Option Value
  | [] => none
  | c :: cs =>
    match (match c with
      | .equal _ v => if unindexable v then some v else none
      | .inList _ (some items) => items.find? unindexable
      | .or nested => unionUnindexable nested
      | _ => none) with
    | some v => some v
    | none => unionUnindexable cs
end

mutual
/-- `describe_predicate` (`mod.rs`). An `ArrayContains` names its probe's type unless it is a number or a
    string (`indexable`); a non-list `IN` operand is `nonList` (the model's `Pred` keeps no value for it). -/
def describeK : Pred → DescK
  | .equal k v => .equality k (if unindexable v then some v else none)
  | .range k mn mx _ _ => .range k ([mn, mx].filterMap id |>.find? unindexable)
  | .or cs =>
    let members := unionMembers true cs
    let n := cs.length
    match oneKey members with
    | none => .orSpanning n
    | some key => if members.all (·.2) then .orOneKey n key (unionUnindexable cs) else .orNonMember n key
  | .and cs => .and (describeAllK cs)
  | .inList k l =>
    .inList k (match l with
      | some items => (items.find? unindexable).map .value
      | none => some .nonList)
  | .point k => .geo k
  | .arrayContains k v => .arrayContains k (if indexable v then none else some v)
def describeAllK : List Pred → List DescK
  | [] => []
  | c :: cs => describeK c :: describeAllK cs
end

/-- A declined `Equal` names its value's type exactly when no kind holds it and it is not `NULL`: a
    string, a number, a NaN or a `NULL` is a plain "equality on `k`". -/
theorem describeK_equal (k : String) (v : Value) :
    describeK (.equal k v) = .equality k (if !indexable v && v != .null then some v else none) := rfl

theorem describeK_equal_string (k s : String) : describeK (.equal k (.string s)) = .equality k none := rfl

theorem describeK_inList (k : String) (items : List Value) :
    describeK (.inList k (some items)) = .inList k ((items.find? unindexable).map .value) ∧
    describeK (.inList k none) = .inList k (some .nonList) := ⟨rfl, rfl⟩

theorem describeK_range_named (k : String) (mn mx : Option Value) (a b : Bool) (v : Value) :
    describeK (.range k mn mx a b) = .range k (some v) → unindexable v = true ∧ (mn = some v ∨ mx = some v) := by
  intro h
  simp only [describeK, DescK.range.injEq, true_and] at h
  have hf := List.find?_some h
  have hm := List.mem_of_find?_eq_some h
  refine ⟨hf, ?_⟩
  cases mn <;> cases mx <;> simp at hm ⊢ <;> first | exact hm | exact hm.symm | exact hm.imp Eq.symm Eq.symm

theorem describeK_leaves (k : String) (v : Value) :
    describeK (.point k) = .geo k ∧
    describeK (.arrayContains k v) = .arrayContains k (if indexable v then none else some v) := ⟨rfl, rfl⟩

/-- A string probe of a list is served by the tag array tree, so its description names no type. -/
theorem describeK_arrayContains_string (k s : String) :
    describeK (.arrayContains k (.string s)) = .arrayContains k none := rfl

theorem describeK_and (cs : List Pred) : describeK (.and cs) = .and (describeAllK cs) := rfl

/-- **The three union reasons are decided by the flattened members exactly as before #3738**; the
    one-attribute, all-served case now names the unindexable value, if any. -/
theorem describeK_or (cs : List Pred) :
    ((∃ n, describeK (.or cs) = .orSpanning n) ↔ oneKey (unionMembers true cs) = none) ∧
    (∀ key, describeK (.or cs) = .orNonMember cs.length key ↔
      oneKey (unionMembers true cs) = some key ∧ ¬ (unionMembers true cs).all (·.2) = true) ∧
    (∀ key bad, describeK (.or cs) = .orOneKey cs.length key bad ↔
      oneKey (unionMembers true cs) = some key ∧ (unionMembers true cs).all (·.2) = true ∧
        bad = unionUnindexable cs) := by
  simp only [describeK]
  generalize unionMembers true cs = ms
  cases hk : oneKey ms with
  | none => simp
  | some k =>
    by_cases hs : ms.all (·.2) = true
    · simp only [hs, ite_true]
      refine ⟨by simp, by simp, fun key bad => ?_⟩
      simp only [DescK.orOneKey.injEq, true_and, Option.some.injEq]
      constructor
      · rintro ⟨rfl, rfl⟩; exact ⟨rfl, rfl⟩
      · rintro ⟨rfl, rfl⟩; exact ⟨rfl, rfl⟩
    · simp [hs]

/-- `a = 1 OR a IN []` (every value indexable) is a one-attribute union naming no value; before #3738
    it was called "not all-numeric". -/
theorem describeK_empty_in (v : Value) (hv : unindexable v = false) :
    describeK (.or [.equal "a" v, .inList "a" (some [])]) = .orOneKey 2 "a" none := by
  simp [describeK, unionMembers, oneKey, Option.filter, unionUnindexable, hv]

/-- `unsupported_by_native_index` (`mod.rs`): what the error names. -/
structure Unsupported where
  entity : String
  label : String
  shape : DescK

def unsupportedByNativeIndex (entity label : String) (q : Pred) : Unsupported := ⟨entity, label, describeK q⟩

/-- The error carries the entity kind, the label/type and the predicate's shape. -/
theorem unsupported_spec (entity label : String) (q : Pred) :
    (unsupportedByNativeIndex entity label q).entity = entity ∧
    (unsupportedByNativeIndex entity label q).label = label ∧
    (unsupportedByNativeIndex entity label q).shape = describeK q := ⟨rfl, rfl, rfl⟩

end Tag
end NI
