import RuntimeCore.RelAccessors

/-!
# Discharging the store contracts of the entity accessors

`Accessors`/`RelAccessors` take the contracts of the stores underneath as hypotheses: the
attribute-name table is a bijection on registered names (`nameOk`), the committed all-attrs listing
agrees with the by-index read (`hc`), the batched read agrees with the per-id read (`hbg`), and
`get_relationship_attribute`'s by-name read is `attr_id` then by-index (`hName`). Here they are
**proved** from the models of the real `Graph` fns (copied from `proofs/graph_queries`
`Attrs.lean`, which owns them; a path dependency would change the lake manifest of every project
that imports this one):

| here | there |
| --- | --- |
| `idx`, `getOrCreate`, `nmInsert` | `AttrNameMap::get_or_create` (`graph.rs`, `get_or_create_node_attr_id` / `_rel_`), `AttrNameMap::insert` (`add_node_attribute_name`, `add_rel_attribute_name`): the only two ways the name table grows; both stop at `MAX_ATTRIBUTES = u16::MAX` |
| `Store`, `byIdx`, `attrByName`, `attrPairs`, `batchByIdx` | `Graph::get_node_attribute_by_idx` / `get_relationship_attribute` (by name, through `u16`) / `get_node_all_attrs` / `get_node_attributes_by_idx` (`graph.rs`) |
| `Reach` | the name tables a graph can have: empty, grown by `getOrCreate` or `nmInsert` |
| `Backed` | the `Accessors.W` a read sees over a graph with name table `nm` and store `st` |

Proved: every reachable table has unique names and at most `MAXA` entries (`reach_nodup`,
`reach_cap`), hence `nameOk` (`nameOk_of_reach`), `hc` (`hc_of_reach`: a name looked up in the
all-attrs listing is the by-index read of its id), `hName` (`hName_of_reach`) and `hbg`
(`hbg_rfl`). The accessor theorems then hold with those four premises gone
(`materialize_discharged`, `getRelAttribute_discharged`, `materializeRel_discharged`,
`materializeNodeProp_discharged`, `materializeRelProp_discharged`, `attrs_agree_discharged`).
What remains is about `Pending`, not the graph: one pending attribute map per entity (`hdisj`) and
unique keys in each pending list (`huniq`); see `COVERAGE.tsv`.
-/
namespace RuntimeCore.Discharge

open RuntimeCore.Accessors
open RuntimeCore.RelAccessors

def MAXA : Nat := 65535
def u16c (n : Nat) : Nat := n % 65536

/-- Index of a name in the table. -/
def idx : List String → String → Option Nat
  | [], _ => none
  | a :: l, s => if a = s then some 0 else (idx l s).map (· + 1)

theorem idx_get : ∀ (l : List String) s i, idx l s = some i → l[i]? = some s
  | [], _, _, h => by simp [idx] at h
  | a :: l, s, i, h => by
    simp only [idx] at h
    split at h
    · rename_i ha
      cases h; simp [ha]
    · cases hh : idx l s with
      | none => simp [hh] at h
      | some j =>
        simp [hh] at h
        subst h
        simpa using idx_get l s j hh

theorem idx_none_iff : ∀ (l : List String) s, idx l s = none ↔ s ∉ l
  | [], s => by simp [idx]
  | a :: l, s => by
    simp only [idx]
    split
    · rename_i h; simp [h]
    · rename_i h
      have := idx_none_iff l s
      simp [this, Ne.symm h]

theorem idx_lt : ∀ (l : List String) s i, idx l s = some i → i < l.length
  | [], _, _, h => by simp [idx] at h
  | a :: l, s, i, h => by
    simp only [idx] at h
    split at h
    · cases h; simp
    · cases hh : idx l s with
      | none => simp [hh] at h
      | some j =>
        simp [hh] at h
        subst h
        have := idx_lt l s j hh
        simp; omega

theorem idx_of_get : ∀ (l : List String) (i : Nat) (s : String), l.Nodup → l[i]? = some s →
    idx l s = some i
  | [], i, s, _, h => by simp at h
  | a :: l, 0, s, _, h => by
    simp at h; subst h; simp [idx]
  | a :: l, i + 1, s, hn, h => by
    simp only [List.getElem?_cons_succ] at h
    have hmem : s ∈ l := List.mem_of_getElem? h
    have hne : a ≠ s := by
      intro e; subst e; exact (List.nodup_cons.1 hn).1 hmem
    have := idx_of_get l i s (List.nodup_cons.1 hn).2 h
    simp [idx, hne, this]

/-- `get_or_create`: the id of `s`, appending it if absent and the table is not full. -/
def getOrCreate (nm : List String) (s : String) : List String × Nat :=
  match idx nm s with
  | some i => (nm, i)
  | none => if nm.length ≥ MAXA then (nm, MAXA) else (nm ++ [s], u16c nm.length)

/-- `AttrNameMap::insert`. -/
def nmInsert (nm : List String) (s : String) : List String :=
  if s ∈ nm then nm else if nm.length ≥ MAXA then nm else nm ++ [s]

/-- The tables a graph can have. -/
inductive Reach : List String → Prop
  | nil : Reach []
  | create (nm : List String) (s : String) : Reach nm → Reach (getOrCreate nm s).1
  | insert (nm : List String) (s : String) : Reach nm → Reach (nmInsert nm s)

theorem reach_cap {nm : List String} (h : Reach nm) : nm.length ≤ MAXA := by
  induction h with
  | nil => simp
  | create nm s _ ih =>
    unfold getOrCreate; split
    · exact ih
    · split
      · exact ih
      · rename_i hn; simp; unfold MAXA at *; omega
  | insert nm s _ ih =>
    unfold nmInsert; split
    · exact ih
    · split
      · exact ih
      · rename_i h1 h2; simp; unfold MAXA at *; omega

theorem reach_nodup {nm : List String} (h : Reach nm) : nm.Nodup := by
  induction h with
  | nil => simp
  | create nm s _ ih =>
    unfold getOrCreate
    cases hi : idx nm s with
    | some i => simpa using ih
    | none =>
      have hs : s ∉ nm := (idx_none_iff nm s).1 hi
      simp only
      split
      · exact ih
      · simp only [List.nodup_append, ih, List.nodup_cons, List.not_mem_nil, not_false_eq_true,
          List.nodup_nil, and_self, List.mem_singleton, true_and]
        intro a ha b hb; subst hb; exact fun e => hs (e ▸ ha)
  | insert nm s _ ih =>
    unfold nmInsert
    split
    · exact ih
    · rename_i hs
      split
      · exact ih
      · simp only [List.nodup_append, ih, List.nodup_cons, List.not_mem_nil, not_false_eq_true,
          List.nodup_nil, and_self, List.mem_singleton, true_and]
        intro a ha b hb; subst hb; exact fun e => hs (e ▸ ha)

/-! ### The stores -/

abbrev Store := Nat → List (Nat × Val)

def byIdx (st : Store) (key i : Nat) : Option Val := (st key).lookup i
/-- `get_relationship_attribute` by name: the id goes through `as u16`. -/
def attrByName (nm : List String) (st : Store) (key : Nat) (s : String) : Option Val :=
  (idx nm s).bind (fun i => byIdx st key (u16c i))
/-- `get_node_all_attrs`: the entity's `(attr id, value)` pairs named through the table. -/
def attrPairs (nm : List String) (st : Store) (key : Nat) : OM :=
  (st key).filterMap (fun p => nm[p.1]?.map (·, p.2))
/-- `get_node_attributes_by_idx`: appends one value per id (default `d`) to `out`. -/
def batchByIdx (st : Store) (ids : List Nat) (i : Nat) (d : Val) (out : List Val) : List Val :=
  out ++ ids.map (fun k => (byIdx st k i).getD d)

/-- The world a read sees over name table `nm` and store `st`. -/
def Backed (w : W) (nm : List String) (st : Store) : Prop :=
  w.committed = byIdx st ∧ w.committedAll = attrPairs nm st ∧ w.attrId = idx nm ∧
    w.attrName = (fun i => nm[i]?)

theorem nameOk_of_reach {nm : List String} (h : Reach nm) (w : W) (hb : Backed w nm st) : w.nameOk := by
  obtain ⟨_, _, h3, h4⟩ := hb
  intro n i
  rw [h3, h4]
  constructor
  · intro hi
    -- `nm[i] = n` and the names are unique, so the first occurrence is `i`
    have hnd := reach_nodup h
    exact idx_of_get nm i n hnd hi
  · intro hi
    exact idx_get nm n i hi

theorem hc_of_reach {nm : List String} (h : Reach nm) (w : W) (hb : Backed w nm st) :
    ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id) := by
  obtain ⟨h1, h2, h3, _⟩ := hb
  intro id name
  rw [h1, h2, h3]
  have hnd := reach_nodup h
  unfold attrPairs byIdx
  induction st id with
  | nil => simp [OM.lookup, List.lookup]
  | cons p l ih =>
    rcases p with ⟨a, v⟩
    simp only [List.filterMap_cons]
    cases hn : nm[a]? with
    | none =>
      -- an id outside the table names nothing, so it is not `idx nm name`
      simp only [Option.map_none]
      rw [ih]
      cases hx : idx nm name with
      | none => simp
      | some j =>
        have hj : a ≠ j := by
          intro e; subst e
          have := idx_get nm name a hx
          rw [hn] at this; cases this
        simp [List.lookup, show (j == a) = false from beq_eq_false_iff_ne.2 (Ne.symm hj)]
    | some nmA =>
      simp only [Option.map_some, OM.lookup]
      by_cases hname : nmA = name
      · subst hname
        have hidx : idx nm nmA = some a := by
          exact idx_of_get nm a nmA hnd hn
        simp [hidx, List.lookup]
      · rw [ite_eq_right (by simpa using hname)]
        rw [ih]
        cases hx : idx nm name with
        | none => simp
        | some j =>
          have hj : a ≠ j := by
            intro e; subst e
            have := idx_get nm name a hx
            rw [hn] at this
            exact hname (Option.some.inj this)
          simp [List.lookup, show (j == a) = false from beq_eq_false_iff_ne.2 (Ne.symm hj)]

theorem hName_of_reach {nm : List String} (h : Reach nm) (st : Store) :
    ∀ id n, attrByName nm st id n = (idx nm n).bind (byIdx st id) := by
  intro id n
  unfold attrByName
  cases hx : idx nm n with
  | none => rfl
  | some i =>
    have hl := idx_lt nm n i hx
    have hc := reach_cap h
    have : u16c i = i := by unfold u16c MAXA at *; omega
    simp [this]

theorem hbg_rfl (st : Store) : ∀ ids i, batchByIdx st ids i .null [] =
    ids.map (fun id => (byIdx st id i).getD .null) := by
  intro ids i; simp [batchByIdx]

/-! ### The accessor theorems with the store premises gone -/

theorem materialize_discharged (w : W) {nm : List String} {st : Store}
    (hb : Backed w nm st) (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materialize w (fun ids i => batchByIdx st ids i .null []) ids name =
      ids.map (fun id => (getNodeAttribute w id name).getD .null) :=
  materialize_eq w _ (fun ids i => by rw [hb.1]; exact hbg_rfl st ids i) hd ids name

theorem getRelAttribute_discharged (w : W) {nm : List String} {st : Store} (h : Reach nm)
    (hb : Backed w nm st) (id : Nat) (name : String) :
    getRelAttribute w (attrByName nm st) id name = getNodeAttribute w id name :=
  getRelAttribute_eq w _ (fun i n => by rw [hName_of_reach h st i n, hb.2.2.1, hb.1]) id name

theorem materializeRel_discharged (w : W) {nm : List String} {st : Store} (h : Reach nm)
    (hb : Backed w nm st) (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeRel w (fun ids i => batchByIdx st ids i .null []) ids name =
      ids.map (fun id => (getRelAttribute w (attrByName nm st) id name).getD .null) :=
  materializeRel_eq w _ (fun i n => by rw [hName_of_reach h st i n, hb.2.2.1, hb.1]) _
    (fun ids i => by rw [hb.1]; exact hbg_rfl st ids i) hd ids name

theorem materializeNodeProp_discharged {C : Type} (classify : List Val → C) (w : W)
    {nm : List String} {st : Store} (hb : Backed w nm st) (hd : w.deletedOk) (ids : List Nat)
    (name : String) :
    materializeNodeProp classify w (fun ids i => batchByIdx st ids i .null []) ids name =
      classify (ids.map (fun id => (getNodeAttribute w id name).getD .null)) :=
  materializeNodeProp_eq classify w _ (fun ids i => by rw [hb.1]; exact hbg_rfl st ids i) hd ids name

theorem materializeRelProp_discharged {C : Type} (classify : List Val → C) (w : W)
    {nm : List String} {st : Store} (h : Reach nm) (hb : Backed w nm st) (hd : w.deletedOk)
    (ids : List Nat) (name : String) :
    materializeRelProp classify w (fun ids i => batchByIdx st ids i .null []) ids name =
      classify (ids.map (fun id => (getRelAttribute w (attrByName nm st) id name).getD .null)) :=
  materializeRelProp_eq classify w _ (fun i n => by rw [hName_of_reach h st i n, hb.2.2.1, hb.1]) _
    (fun ids i => by rw [hb.1]; exact hbg_rfl st ids i) hd ids name

theorem attrs_agree_discharged (w : W) {nm : List String} {st : Store} (h : Reach nm)
    (hb : Backed w nm st) (hd : w.deletedOk) (ha : w.attrsOk)
    (hdisj : ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none)
    (huniq : ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup)
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getNodeAttrs w id)) = Val.norm (getNodeAttribute w id name) :=
  attrs_agree w (nameOk_of_reach h w hb) hd ha (hc_of_reach h w hb) hdisj huniq id name

theorem relAttrs_agree_discharged (w : W) {nm : List String} {st : Store} (h : Reach nm)
    (hb : Backed w nm st) (hd : w.deletedOk) (ha : w.attrsOk)
    (hdisj : ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none)
    (huniq : ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup)
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getRelAttrs w id)) =
      Val.norm (getRelAttribute w (attrByName nm st) id name) :=
  relAttrs_agree w (attrByName nm st) (fun i n => by rw [hName_of_reach h st i n, hb.2.2.1, hb.1])
    (nameOk_of_reach h w hb) hd ha (hc_of_reach h w hb) hdisj huniq id name

end RuntimeCore.Discharge
