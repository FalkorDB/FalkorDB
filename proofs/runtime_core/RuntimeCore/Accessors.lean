/-
# Entity accessors: deleted snapshot, then this query's pending writes, then the committed graph

| here | there |
| --- | --- |
| `W`                       | the state a read sees: `Runtime.deleted_nodes`, `Pending.{new,existing}_nodes_attrs`, `Pending.{set,remove}_labels`, the committed `Graph` |
| `OM.insert/remove/lookup` | `OrderMap::{insert,remove,get}` (assoc list, unique keys) |
| `pendingGet`              | `Pending::get_node_attribute`, `runtime/pending.rs:485-501` |
| `memoLookup`, `memoInsert`, `attrIdM` | `Runtime::{memo_lookup,memo_insert,node_attr_id}`, `runtime/runtime.rs:1407-1443` |
| `getNodeAttribute`        | `Runtime::get_node_attribute` + `_no_delete_check`, `runtime.rs:1459-1490` |
| `materialize`             | `Runtime::materialize_node_property_values`, `runtime.rs:1556-1598` |
| `updateNodeAttrs`, `getNodeAttrs` | `Pending::update_node_attrs` (`pending.rs:503-525`), `Runtime::get_node_attrs` (`runtime.rs:1739-1755`) |
| `labelIds`, `hasLabelId`, `hasLabel` | `Runtime::get_node_labels` (`:1660-1672`), `node_has_label_id` (`:1725-1737`), `node_has_label` (`:1690-1703`); `Pending::{node_has_label,update_node_labels}` (`pending.rs:596-641`) |
| `stage`, `unstage`        | `Pending::stage_node_labels` / `remove_node_labels` (`pending.rs:533-584`) |
| `degree`                  | `Runtime::get_node_{in,out}degree{,_by_type}` (`runtime.rs:1804-1872`) |
| `relEndpoints`, `relType` | `Runtime::get_relationship_{endpoints,type}` (`runtime.rs:1777-1802`) |

The relationship attribute accessors (`:1493-1535`, `:1604-1658`, `:1757-1775`) have the
same shape over the relationship maps and are covered by the same theorems
(instantiate `W` with relationship ids; `get_relationship_attribute`'s committed fallback
is by name, which `hCommName` below equates with the by-index read).

Hypotheses (not axioms) are the contracts of the stores underneath: the attribute name
table is a bijection on registered names, the committed all-attrs listing agrees with the
by-index read, a node is not in both pending attribute maps, and `lookup_sorted`'s binary
search equals a linear lookup on its sorted unique-key vector.
-/
namespace RuntimeCore.Accessors

inductive Val where
  | null
  | int (i : Int)
  | str (s : String)
  deriving DecidableEq, Repr

/-! ### Association-list `OrderMap` -/

abbrev OM := List (String × Val)

def OM.lookup (k : String) : OM → Option Val
  | [] => none
  | p :: m => if p.1 = k then some p.2 else OM.lookup k m

def OM.insert (k : String) (v : Val) : OM → OM
  | [] => [(k, v)]
  | p :: m => if p.1 = k then (k, v) :: m else p :: OM.insert k v m

def OM.remove (k : String) : OM → OM
  | [] => []
  | p :: m => if p.1 = k then OM.remove k m else p :: OM.remove k m

theorem OM.lookup_insert (k k' : String) (v : Val) :
    ∀ m : OM, OM.lookup k (OM.insert k' v m) = if k' = k then some v else OM.lookup k m
  | [] => by simp [OM.insert, OM.lookup]
  | p :: m => by
    have ih := OM.lookup_insert k k' v m
    by_cases h1 : p.1 = k' <;> by_cases h2 : p.1 = k <;> by_cases h3 : k' = k <;>
      simp_all [OM.insert, OM.lookup]

theorem OM.lookup_remove (k k' : String) :
    ∀ m : OM, OM.lookup k (OM.remove k' m) = if k' = k then none else OM.lookup k m
  | [] => by simp [OM.remove, OM.lookup]
  | p :: m => by
    have ih := OM.lookup_remove k k' m
    by_cases h1 : p.1 = k' <;> by_cases h2 : p.1 = k <;> by_cases h3 : k' = k <;>
      simp_all [OM.remove, OM.lookup]

/-! ### The world a read sees -/

structure Snap where
  attrs : OM
  labels : List Nat

structure W where
  /-- `Runtime.deleted_nodes`. -/
  deleted : Nat → Option Snap
  deletedEmpty : Bool
  /-- `Pending.new_nodes_attrs` / `existing_nodes_attrs`: `(attr_id, value)` per node. -/
  newAttrs : Nat → Option (List (Nat × Val))
  exAttrs : Nat → Option (List (Nat × Val))
  /-- `Pending::has_node_attrs`. -/
  hasAttrs : Bool
  /-- `Graph::get_node_attribute_by_idx`. -/
  committed : Nat → Nat → Option Val
  /-- `Graph::get_node_all_attrs` (as an `OrderMap`). -/
  committedAll : Nat → OM
  /-- `Graph::get_node_attr_id` / `node_attr_name`. -/
  attrId : String → Option Nat
  attrName : Nat → Option String
  /-- Labels: committed label ids, pending `set_labels` / `remove_labels`. -/
  committedLabels : Nat → List Nat
  setL : Nat → List Nat
  remL : Nat → List Nat
  labelName : Nat → String
  labelId : String → Option Nat

/-- `lookup_sorted` (binary search on a sorted unique-key vector ≡ linear lookup). -/
def lookupA (a : Nat) : List (Nat × Val) → Option Val
  | [] => none
  | p :: m => if p.1 = a then some p.2 else lookupA a m

/-- `Pending::get_node_attribute` (`pending.rs:485-501`). -/
def pendingGet (w : W) (id a : Nat) : Option Val :=
  if !w.hasAttrs then none
  else
    match (w.newAttrs id).bind (lookupA a) with
    | some v => some v
    | none => (w.exAttrs id).bind (lookupA a)

/-- `get_node_attribute_no_delete_check` (`runtime.rs:1479-1490`), with the attr id
already resolved (see `attrIdM` for the memo). -/
def noDelete (w : W) (id : Nat) (name : String) : Option Val :=
  match w.attrId name with
  | none => none
  | some a =>
    match pendingGet w id a with
    | some v => some v
    | none => w.committed id a

/-- `Runtime::get_node_attribute` (`runtime.rs:1459-1475`). -/
def getNodeAttribute (w : W) (id : Nat) (name : String) : Option Val :=
  match (if !w.deletedEmpty then w.deleted id else none) with
  | some dn => dn.attrs.lookup name
  | none => noDelete w id name

/-- `materialize_node_property_values` (`runtime.rs:1556-1598`); `bg` is the batched
`get_node_attributes_by_idx`. -/
def materialize (w : W) (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) :
    List Val :=
  let idx := w.attrId name
  if w.deletedEmpty && !w.hasAttrs then
    match idx with
    | some i => bg ids i
    | none => ids.map (fun _ => .null)
  else
    ids.map (fun id =>
      match w.deleted id with
      | none =>
        (idx.bind (fun i =>
          match pendingGet w id i with
          | some v => some v
          | none => w.committed id i)).getD .null
      | some dn => (dn.attrs.lookup name).getD .null)

/-- Contract: `deleted_nodes.is_empty()` is what it says. -/
def W.deletedOk (w : W) : Prop := w.deletedEmpty = true → ∀ id, w.deleted id = none

/-- Contract: `has_node_attrs() == false` means both pending maps are empty. -/
def W.attrsOk (w : W) : Prop :=
  w.hasAttrs = false → ∀ id, w.newAttrs id = none ∧ w.exAttrs id = none

/-- **PROVEN**: the bulk column read agrees, row for row, with the per-row accessor —
on both the hot read-only path and the overlay path. -/
theorem materialize_eq (w : W) (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materialize w bg ids name = ids.map (fun id => (getNodeAttribute w id name).getD .null) := by
  simp only [materialize]
  split
  · next hh =>
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hh
    have hdel := hd hh.1
    split
    · next i hi =>
      rw [hbg]
      apply List.map_congr_left
      intro id _
      simp [getNodeAttribute, hh.1, noDelete, hi, pendingGet, hh.2]
    · next hi =>
      apply List.map_congr_left
      intro id _
      simp [getNodeAttribute, hh.1, noDelete, hi]
  · next hh =>
    apply List.map_congr_left
    intro id _
    simp only [getNodeAttribute]
    cases hde : w.deletedEmpty
    · simp only [Bool.not_false, ite_true]
      cases w.deleted id with
      | some dn => rfl
      | none =>
        simp only [noDelete]
        cases w.attrId name <;> simp
    · have := hd hde id
      simp only [Bool.not_true, ite_false, this]
      simp only [noDelete]
      cases w.attrId name <;> simp

/-! ### `get_node_attrs` agrees with `get_node_attribute` -/

/-- One step of `update_node_attrs`'s loop (`pending.rs:513-523`). -/
def attrStep (w : W) (m : OM) (p : Nat × Val) : OM :=
  match w.attrName p.1 with
  | none => m
  | some key => if p.2 = .null then OM.remove key m else OM.insert key p.2 m

/-- `Pending::update_node_attrs`: only ONE map is consulted (`new` else `existing`). -/
def updateNodeAttrs (w : W) (id : Nat) (m : OM) : OM :=
  match (match w.newAttrs id with | some l => some l | none => w.exAttrs id) with
  | none => m
  | some added => added.foldl (attrStep w) m

/-- `Runtime::get_node_attrs` (`runtime.rs:1739-1755`). -/
def getNodeAttrs (w : W) (id : Nat) : OM :=
  match w.deleted id with
  | some dn => dn.attrs
  | none => updateNodeAttrs w id (w.committedAll id)

def W.nameOk (w : W) : Prop :=
  ∀ n i, w.attrName i = some n ↔ w.attrId n = some i

theorem lookupA_none (a : Nat) : ∀ (l : List (Nat × Val)), a ∉ l.map Prod.fst → lookupA a l = none
  | [], _ => rfl
  | q :: qs, h => by
    simp only [List.map_cons, List.mem_cons, not_or] at h
    simp only [lookupA]
    rw [if_neg (fun e => h.1 e.symm)]
    exact lookupA_none a qs h.2

theorem fold_lookup (w : W) (hn : w.nameOk) (name : String) :
    ∀ (added : List (Nat × Val)) (m : OM), (added.map Prod.fst).Nodup →
      OM.lookup name (added.foldl (attrStep w) m) =
        match w.attrId name with
        | none => OM.lookup name m
        | some a =>
          match lookupA a added with
          | none => OM.lookup name m
          | some v => if v = .null then none else some v
  | [], m, _ => by simp [lookupA]; split <;> rfl
  | p :: rest, m, hnd => by
    simp only [List.foldl_cons]
    have hnd' : (rest.map Prod.fst).Nodup := (List.nodup_cons.mp hnd).2
    have hp : p.1 ∉ rest.map Prod.fst := (List.nodup_cons.mp hnd).1
    rw [fold_lookup w hn name rest _ hnd']
    cases ha : w.attrId name with
    | none =>
      simp only [attrStep]
      cases hk : w.attrName p.1 with
      | none => rfl
      | some key =>
        have hne : key ≠ name := by
          intro h; subst h; have := (hn key p.1).mp hk; rw [ha] at this; simp at this
        simp only
        split
        · rw [OM.lookup_remove]; simp [hne]
        · rw [OM.lookup_insert]; simp [hne]
    | some a =>
      simp only [lookupA]
      by_cases hpa : p.1 = a
      · -- the entry for `name`; nothing later can override it (unique ids)
        have hnone : lookupA a rest = none := by
          subst hpa; exact lookupA_none _ _ hp
        simp only [hnone, hpa, ite_true, attrStep]
        have hk : w.attrName a = some name := (hn name a).mpr ha
        simp only [hk]
        split
        · next hv => rw [OM.lookup_remove]; simp [hv]
        · next hv => rw [OM.lookup_insert]; simp [hv]
      · simp only [hpa, ite_false]
        split
        · simp only [attrStep]
          cases hk : w.attrName p.1 with
          | none => rfl
          | some key =>
            have hne : key ≠ name := by
              intro h; subst h; have := (hn key p.1).mp hk; rw [ha] at this; simp at this
              exact hpa this.symm
            simp only
            split
            · rw [OM.lookup_remove]; simp [hne]
            · rw [OM.lookup_insert]; simp [hne]
        · rfl

def Val.norm : Option Val → Val
  | none => .null
  | some v => v

/-- **PROVEN** (headline for attributes): `n.prop` and `properties(n).prop` agree, for
deleted, pending-written and committed nodes, where `null` stands for "absent" —
provided (1) the name table is a bijection, (2) the committed all-attrs listing agrees
with the by-index read, (3) a node's pending writes live in one pending map, (4) pending
attr ids are unique, and (5) `deleted_nodes.is_empty()` / `has_node_attrs()` are honest.

(3) is where #2876 (deleted-id reuse) bites: a reused id can be in both maps, and in
`deleted_nodes`, so the three reads diverge. -/
theorem attrs_agree (w : W) (hn : w.nameOk) (hd : w.deletedOk) (ha : w.attrsOk)
    (hc : ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id))
    (hdisj : ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none)
    (huniq : ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup)
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getNodeAttrs w id)) = Val.norm (getNodeAttribute w id name) := by
  simp only [getNodeAttrs, getNodeAttribute]
  cases hdi : w.deleted id with
  | some dn =>
    cases hde : w.deletedEmpty
    · simp [hdi]
    · simp [hd hde id] at hdi
  | none =>
    simp only [ite_self]
    simp only [updateNodeAttrs, noDelete]
    -- which pending list (if any) is consulted
    cases hN : w.newAttrs id with
    | some l =>
      have hEx : w.exAttrs id = none := by
        rcases hdisj id with h | h
        · rw [hN] at h; simp at h
        · exact h
      simp only
      rw [fold_lookup w hn name l _ (huniq id l (Or.inl hN))]
      have hA : w.hasAttrs = true := by
        cases hh : w.hasAttrs
        · have := (ha hh id).1; rw [hN] at this; simp at this
        · rfl
      cases hid : w.attrId name with
      | none => simp [hc, hid]
      | some a =>
        simp only [pendingGet, hA, Bool.not_true, hN, Option.bind_some, hEx, Option.bind_none]
        cases hla : lookupA a l with
        | none => simp [hc, hid]
        | some v => by_cases hv : v = .null <;> simp [hv, Val.norm]
    | none =>
      cases hE : w.exAttrs id with
      | none =>
        simp only
        rw [hc]
        cases hid : w.attrId name with
        | none => rfl
        | some a =>
          simp only [Option.bind_some, pendingGet, hN, hE, Option.bind_none]
          cases w.hasAttrs <;> simp
      | some l =>
        simp only
        rw [fold_lookup w hn name l _ (huniq id l (Or.inr hE))]
        have hA : w.hasAttrs = true := by
          cases hh : w.hasAttrs
          · have := (ha hh id).2; rw [hE] at this; simp at this
          · rfl
        cases hid : w.attrId name with
        | none => simp [hc, hid]
        | some a =>
          simp only [pendingGet, hA, Bool.not_true, hN, hE, Option.bind_none, Option.bind_some]
          cases hla : lookupA a l with
          | none => simp [hc, hid]
          | some v => by_cases hv : v = .null <;> simp [hv, Val.norm]

/-- The two reads DO diverge once a node id sits in both pending maps (id reuse,
#2876): `get_node_attribute` falls through `new` to `existing`, `get_node_attrs` reads
only `new`. Concrete witness. -/
def reuseWorld : W where
  deleted := fun _ => none
  deletedEmpty := true
  newAttrs := fun _ => some [(1, .int 7)]
  exAttrs := fun _ => some [(0, .int 5)]
  hasAttrs := true
  committed := fun _ _ => none
  committedAll := fun _ => []
  attrId := fun n => if n = "a" then some 0 else if n = "b" then some 1 else none
  attrName := fun i => if i = 0 then some "a" else if i = 1 then some "b" else none
  committedLabels := fun _ => []
  setL := fun _ => []
  remL := fun _ => []
  labelName := fun _ => ""
  labelId := fun _ => none

theorem reuse_diverges :
    getNodeAttribute reuseWorld 0 "a" = some (.int 5) ∧
    OM.lookup "a" (getNodeAttrs reuseWorld 0) = none := by decide

/-! ### Attribute-id memo (`runtime.rs:1407-1457`) -/

def ATTR_ID_MEMO_CAP : Nat := 32

/-- Memo entries are `(Arc<String>, id)`, looked up by pointer; `nameOf` maps a pointer
to its string (a live `Arc` is never recycled, since the memo holds a clone). -/
def memoLookup (nameOf : Nat → String) (memo : List (Nat × Nat)) (ptr : Nat) : Option Nat :=
  (memo.find? (fun e => e.1 = ptr)).map (·.2)

def memoInsert (memo : List (Nat × Nat)) (ptr id : Nat) : List (Nat × Nat) :=
  if memo.length < ATTR_ID_MEMO_CAP then memo ++ [(ptr, id)] else memo

/-- `node_attr_id`: memo hit, else table lookup and memoize the hit (never a miss). -/
def attrIdM (nameOf : Nat → String) (attrId : String → Option Nat) (memo : List (Nat × Nat))
    (ptr : Nat) : Option Nat × List (Nat × Nat) :=
  match memoLookup nameOf memo ptr with
  | some id => (some id, memo)
  | none =>
    match attrId (nameOf ptr) with
    | none => (none, memo)
    | some id => (some id, memoInsert memo ptr id)

def MemoOk (nameOf : Nat → String) (attrId : String → Option Nat) (memo : List (Nat × Nat)) :
    Prop := (∀ e ∈ memo, attrId (nameOf e.1) = some e.2) ∧ memo.length ≤ ATTR_ID_MEMO_CAP

theorem memoLookup_ok {nameOf attrId memo ptr id} (h : MemoOk nameOf attrId memo)
    (hl : memoLookup nameOf memo ptr = some id) : attrId (nameOf ptr) = some id := by
  simp only [memoLookup, Option.map_eq_some_iff] at hl
  obtain ⟨e, he, rfl⟩ := hl
  have hm := List.mem_of_find?_eq_some he
  have := List.find?_some he
  simp at this; rw [← this]; exact h.1 e hm

/-- **PROVEN**: the memoized id lookup returns exactly what the name table returns, and
keeps the memo valid and within its cap. -/
theorem attrIdM_correct (nameOf : Nat → String) (attrId : String → Option Nat)
    (memo : List (Nat × Nat)) (ptr : Nat) (h : MemoOk nameOf attrId memo) :
    (attrIdM nameOf attrId memo ptr).1 = attrId (nameOf ptr) ∧
    MemoOk nameOf attrId (attrIdM nameOf attrId memo ptr).2 := by
  simp only [attrIdM]
  cases hl : memoLookup nameOf memo ptr with
  | some id => exact ⟨(memoLookup_ok h hl).symm, h⟩
  | none =>
    cases ha : attrId (nameOf ptr) with
    | none => exact ⟨rfl, h⟩
    | some id =>
      refine ⟨rfl, ?_, ?_⟩
      · intro e he
        simp only [memoInsert] at he
        split at he
        · simp only [List.mem_append, List.mem_singleton] at he
          rcases he with he | rfl
          · exact h.1 e he
          · exact ha
        · exact h.1 e he
      · simp only [memoInsert]; split
        · simp; unfold ATTR_ID_MEMO_CAP at *; omega
        · exact h.2

/-- **PROVEN**: the memo stays valid when the name table grows (`get_or_create_*_attr_id`
only adds names; existing ids never change). -/
theorem memo_ok_mono {nameOf : Nat → String} {t t' : String → Option Nat} {memo}
    (h : MemoOk nameOf t memo) (hmono : ∀ n i, t n = some i → t' n = some i) :
    MemoOk nameOf t' memo := ⟨fun e he => hmono _ _ (h.1 e he), h.2⟩

/-! ### Labels -/

/-- `OrderSet` insert / remove / contains on a list. -/
def osInsert (s : List Nat) (x : Nat) : List Nat := if x ∈ s then s else s ++ [x]
def osRemove (s : List Nat) (x : Nat) : List Nat := s.filter (· ≠ x)

/-- `get_node_labels` for a node not deleted by this query: committed ids, then
`update_node_labels` (adds, then removals). Returns label ids (names via `labelName`). -/
def labelIds (w : W) (id : Nat) : List Nat :=
  (w.remL id).foldl osRemove ((w.setL id).foldl osInsert (w.committedLabels id))

/-- `Pending::node_has_label` (`pending.rs:596-617`). -/
def pendingHas (w : W) (id l : Nat) : Option Bool :=
  if l ∈ w.remL id then some false else if l ∈ w.setL id then some true else none

/-- `Runtime::node_has_label_id` (`runtime.rs:1725-1737`). -/
def hasLabelId (w : W) (id l : Nat) : Bool :=
  match w.deleted id with
  | some dn => decide (l ∈ dn.labels)
  | none => (pendingHas w id l).getD (decide (l ∈ w.committedLabels id))

/-- `Runtime::get_node_labels` (`runtime.rs:1660-1672`), as label ids. -/
def getLabelIds (w : W) (id : Nat) : List Nat :=
  match w.deleted id with
  | some dn => dn.labels
  | none => labelIds w id

theorem mem_foldl_insert : ∀ (xs s : List Nat) (l : Nat), l ∈ xs.foldl osInsert s ↔ l ∈ s ∨ l ∈ xs
  | [], s, l => by simp
  | x :: xs, s, l => by
    simp only [List.foldl_cons]
    rw [mem_foldl_insert xs]
    simp only [osInsert, List.mem_cons]
    split
    · next hx =>
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl h
        · subst h; exact Or.inl hx
        · exact Or.inr h
    · simp only [List.mem_append, List.mem_singleton]
      constructor
      · rintro ((h | h) | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr h

theorem mem_foldl_remove : ∀ (xs s : List Nat) (l : Nat), l ∈ xs.foldl osRemove s ↔ l ∈ s ∧ l ∉ xs
  | [], s, l => by simp
  | x :: xs, s, l => by
    simp only [List.foldl_cons]
    rw [mem_foldl_remove xs]
    simp only [osRemove, List.mem_filter, List.mem_cons, not_or, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, h2, h3⟩
    · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h1, h2⟩, h3⟩

/-- **PROVEN** (headline for labels): the one-bit label test `n:L` (`node_has_label_id`)
answers exactly "is `L` in `labels(n)`" — for deleted, pending-relabelled and committed
nodes — with no disjointness assumption on the staged add/remove sets. -/
theorem hasLabelId_iff (w : W) (id l : Nat) : hasLabelId w id l = true ↔ l ∈ getLabelIds w id := by
  unfold hasLabelId getLabelIds
  cases hdel : w.deleted id with
  | some dn => simp
  | none =>
    simp only [labelIds, pendingHas]
    have := mem_foldl_remove (w.remL id) ((w.setL id).foldl osInsert (w.committedLabels id)) l
    rw [mem_foldl_insert] at this
    rw [this]
    by_cases hr : l ∈ w.remL id
    · simp [hr]
    · by_cases hs : l ∈ w.setL id
      · simp [hr, hs]
      · simp [hr, hs]

/-- `node_has_label(id, name)` (`runtime.rs:1690-1703`). -/
def hasLabel (w : W) (id : Nat) (name : String) : Bool :=
  match w.labelId name with
  | none => false
  | some l => hasLabelId w id l

/-- **PROVEN**: testing by name equals "the name is among `labels(n)`", given that every
label id the node can carry is registered and the label table is a bijection. -/
theorem hasLabel_iff (w : W) (id : Nat) (name : String)
    (hinv : ∀ n l, w.labelId n = some l → w.labelName l = n)
    (hreg : ∀ l ∈ getLabelIds w id, w.labelId (w.labelName l) = some l) :
    hasLabel w id name = true ↔ name ∈ (getLabelIds w id).map w.labelName := by
  unfold hasLabel
  cases hl : w.labelId name with
  | none =>
    simp only [Bool.false_eq_true, false_iff, List.mem_map, not_exists, not_and]
    intro l hm he
    have := hreg l hm; rw [he, hl] at this; simp at this
  | some l =>
    simp only
    rw [hasLabelId_iff, List.mem_map]
    constructor
    · intro hm; exact ⟨l, hm, hinv name l hl⟩
    · rintro ⟨l', hm, he⟩
      have := hreg l' hm; rw [he, hl] at this; simp at this; subst this; exact hm

/-- `stage_node_labels` (`pending.rs:533-549`) and `remove_node_labels` (`:569-584`)
on one node's `(set, remove)` lists. -/
def stage (set rem : List Nat) (ls : List Nat) : List Nat × List Nat :=
  (set ++ ls, rem.filter (· ∉ ls))

def unstage (set rem : List Nat) (ls : List Nat) : List Nat × List Nat :=
  (set.filter (· ∉ ls), rem ++ ls)

def Disjoint (set rem : List Nat) : Prop := ∀ x, x ∈ set → x ∉ rem

/-- **PROVEN**: both staging operations keep the add/remove sets disjoint, and the last
clause wins: after `SET n:L` the staged answer for `L` is `true`, after `REMOVE n:L` it
is `false`. -/
theorem stage_spec (set rem ls : List Nat) (h : Disjoint set rem) :
    Disjoint (stage set rem ls).1 (stage set rem ls).2 ∧
    ∀ l ∈ ls, l ∉ (stage set rem ls).2 ∧ l ∈ (stage set rem ls).1 := by
  simp only [stage, Disjoint, List.mem_append, List.mem_filter, decide_eq_true_eq]
  refine ⟨?_, fun l hl => ⟨fun ⟨_, h2⟩ => h2 hl, Or.inr hl⟩⟩
  rintro x (hx | hx) ⟨hr, hn⟩
  · exact h x hx hr
  · exact hn hx

theorem unstage_spec (set rem ls : List Nat) (h : Disjoint set rem) :
    Disjoint (unstage set rem ls).1 (unstage set rem ls).2 ∧
    ∀ l ∈ ls, l ∈ (unstage set rem ls).2 := by
  simp only [unstage, Disjoint, List.mem_append, List.mem_filter, decide_eq_true_eq]
  refine ⟨?_, fun l hl => Or.inr hl⟩
  rintro x ⟨hs, hn⟩ (hr | hr)
  · exact h x hs hr
  · exact hn hr

/-! ### Degrees (`runtime.rs:1804-1872`) -/

structure Edge where
  id : Nat
  src : Nat
  dst : Nat
  ty : String
  deriving DecidableEq

/-- Degrees, with `types = []` meaning all types (the untyped variant). `E` = committed edges, `C` = this query's created edges, `D` = pending-deleted
flag on committed edges (`pending_deleted_indegree` iterates `deleted_relationships` and
resolves each id in the committed graph — `Graph::get_relationship_endpoints` panics on
an unknown id, so every deleted id is a committed edge; `D` models that set). -/
def tyOk (types : List String) (e : Edge) : Bool := types.isEmpty || types.contains e.ty

/-- `get_node_{in,out}degree{,_by_type}`: `endp` is `Edge.dst` for in-degree and
`Edge.src` for out-degree. -/
def degree (endp : Edge → Nat) (E C : List Edge) (D : Edge → Bool) (nodeGone : Bool)
    (types : List String) (v : Nat) : Nat :=
  if nodeGone then 0
  else
    let base := E.countP (fun e => endp e = v && tyOk types e)
    let added := C.countP (fun e => endp e = v && tyOk types e)
    let removed := E.countP (fun e => D e && endp e = v && tyOk types e)
    base + added - removed

/-- **PROVEN** (no `usize` underflow, exact count): `base + added - removed` never
underflows, and it is the number of live edges at `v` of the requested types —
committed ones not deleted by this query, plus the ones it created. -/
theorem degree_exact (endp : Edge → Nat) (E C : List Edge) (D : Edge → Bool)
    (types : List String) (v : Nat) :
    E.countP (fun e => D e && endp e = v && tyOk types e) ≤
      E.countP (fun e => endp e = v && tyOk types e) + C.countP (fun e => endp e = v && tyOk types e) ∧
    degree endp E C D false types v =
      (E.filter (fun e => !D e)).countP (fun e => endp e = v && tyOk types e) +
        C.countP (fun e => endp e = v && tyOk types e) := by
  have hsplit : ∀ (L : List Edge), L.countP (fun e => endp e = v && tyOk types e) =
      L.countP (fun e => D e && endp e = v && tyOk types e) +
        (L.filter (fun e => !D e)).countP (fun e => endp e = v && tyOk types e) := by
    intro L
    induction L with
    | nil => rfl
    | cons e L ih =>
      simp only [List.countP_cons, List.filter_cons]
      cases hD : D e <;> cases hp : (endp e = v && tyOk types e) <;>
        simp [hp, List.countP_cons, ih] <;> omega
  refine ⟨by rw [hsplit E]; omega, ?_⟩
  simp only [degree, Bool.false_eq_true, ite_false]
  rw [hsplit E]; omega

/-- Relationship endpoints/type: deleted snapshot, then pending-created, then committed. -/
def relLookup {X : Type} (del : Nat → Option X) (pend : Nat → Option X) (com : Nat → X) (id : Nat) : X :=
  match del id with
  | some x => x
  | none => match pend id with
    | some x => x
    | none => com id

theorem relLookup_deleted {X : Type} (del pend : Nat → Option X) (com : Nat → X) (id : Nat) (x : X)
    (h : del id = some x) : relLookup del pend com id = x := by simp [relLookup, h]

end RuntimeCore.Accessors
