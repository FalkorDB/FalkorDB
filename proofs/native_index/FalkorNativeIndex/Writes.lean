import FalkorNativeIndex.Pending
set_option linter.unusedSimpArgs false

/-!
# How graph writes reach the native index (`graph/src/index/falkordb/graph_writes.rs`)

Every staging function is a nest of loops that `push`es `remove (old, id)` / `add (new, id)` under
`(label, attr)` columns. Each is modelled loop for loop (`foldl` over the Rust iterators) and proven
equal to a flat list of **ops** (`StageOp`: id, label names, attr name, old value, new value), staged as
`for label: remove old; add new` (`opsChanges_*` theorems). The generic theorem
`lastOp_ops` then says what one unit of work does to any tuple `(key, id)` of any column: the last
op for that `(id, attr)` under that label decides — add if the new value encodes to `key`, remove if
only the old one does. With `applyColumn_eq_sequential` / `mem_seqTree` that is the column after the
commit.

The graph is the slice the staging functions read (`G`): label/type/attribute names by id
(`get_labels`, `get_types`, `node_attr_name` = `rel_attr_name` = `attrs_name[id]`), each node's label
ids (`get_node_label_ids`), attribute values by index (`get_node_attribute_by_idx`), an edge's type
(`get_relationship_type_id`), and the native index.
-/

namespace NI

structure G where
  labels : List String
  types : List String
  attrNames : List String
  nodeLabels : Nat → List Nat
  nodeAttr : Nat → Nat → Option Value
  relType : Nat → Nat
  relAttr : Nat → Nat → Option Value
  idx : FalkorDbIndex
  /-- label matrix column: the nodes carrying label `l` (`get_label_matrix(label).iter`) -/
  labelNodes : String → Option (List Nat)
  /-- relationship tensor of a type: its edge ids (`get_relationship_matrix(type).iter`) -/
  typeEdges : String → Option (List Nat)

namespace G
def labelName (g : G) (l : Nat) : String := g.labels.getD l ""
def typeName (g : G) (t : Nat) : String := g.types.getD t ""
def attrName (g : G) (a : Nat) : Option String := g.attrNames[a]?
/-- `get_node_all_attrs` (`graph.rs:3532`): `(name, value)` of every attribute the node has. -/
def nodeAllAttrs (g : G) (id : Nat) : List (String × Value) :=
  (List.range g.attrNames.length).filterMap (fun a =>
    match g.attrNames[a]?, g.nodeAttr id a with
    | some n, some v => some (n, v)
    | _, _ => none)
/-- `get_relationship_all_attrs` (`graph.rs:3563`). -/
def relAllAttrs (g : G) (id : Nat) : List (String × Value) :=
  (List.range g.attrNames.length).filterMap (fun a =>
    match g.attrNames[a]?, g.relAttr id a with
    | some n, some v => some (n, v)
    | _, _ => none)
/-- `get_node_attribute_id` / `get_relationship_attribute_id`: position of a name. -/
def attrId (g : G) (n : String) : Option Nat := g.attrNames.idxOf? n
end G

/-! ## Ops and their staged changes -/

/-- One staging op: under each label name, remove `old` then add `new`, for entity `id`, attr `attr`. -/
structure StageOp where
  id : Nat
  labels : List String
  attr : String
  old : Option Value
  new : Option Value

/-- What `push` appends (`pending_writes.rs:81-95`). -/
def pushL (ix : FalkorDbIndex) (ent : EntityType) (label attr : String) (v : Value) (id : Nat) (add : Bool) :
    List (IndexKey × IndexWrite) :=
  match ix.encodeValue ent label attr v with
  | none => []
  | some k => [((label, attr), ⟨k, id, add⟩)]

theorem push_eq (c : IndexChanges) (ix : FalkorDbIndex) (l a : String) (v : Value) (id : Nat) (add : Bool) :
    c.push ix l a v id add = { c with changes := c.changes ++ pushL ix c.entity l a v id add } := by
  unfold IndexChanges.push pushL
  cases ix.encodeValue c.entity l a v <;> simp

def opChanges (ix : FalkorDbIndex) (ent : EntityType) (op : StageOp) : List (IndexKey × IndexWrite) :=
  op.labels.flatMap (fun l =>
    (match op.old with | some o => pushL ix ent l op.attr o op.id false | none => []) ++
    (match op.new with | some v => pushL ix ent l op.attr v op.id true | none => []))

def opsChanges (ix : FalkorDbIndex) (ent : EntityType) (ops : List StageOp) : List (IndexKey × IndexWrite) :=
  ops.flatMap (opChanges ix ent)

/-- A column's log inside a change list (what `extend` files under that column, `extend_log`). -/
def colLog (k : IndexKey) (cs : List (IndexKey × IndexWrite)) : List IndexWrite :=
  (cs.filter (fun kw => kw.1 = k)).map (·.2)

/-! ## `lastOp` algebra -/

theorem lastOp_append (A B : List IndexWrite) (e : Nat) :
    lastOp (A ++ B) e = (lastOp B e).or (lastOp A e) := by
  unfold lastOp
  rw [List.filter_append, List.getLast?_append]
  cases (List.filter (fun w => decide (w.tup = e)) B).getLast? <;> simp

theorem colLog_append (k : IndexKey) (A B : List (IndexKey × IndexWrite)) :
    colLog k (A ++ B) = colLog k A ++ colLog k B := by
  simp [colLog, List.filter_append]

theorem colLog_flatMap {α} (k : IndexKey) (l : List α) (F : α → List (IndexKey × IndexWrite)) :
    colLog k (l.flatMap F) = l.flatMap (fun x => colLog k (F x)) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [List.flatMap_cons, colLog_append, ih]

/-- If every item's block either leaves `e` alone or ends it at `r`, the whole list ends `e` at `r`
    iff some item touches it. -/
theorem lastOp_flatMap_common {α} (l : List α) (F : α → List IndexWrite) (e : Nat) (r : Option Bool)
    (h : ∀ x ∈ l, lastOp (F x) e = none ∨ lastOp (F x) e = r) :
    lastOp (l.flatMap F) e = if ∃ x ∈ l, lastOp (F x) e ≠ none then r else none := by
  induction l with
  | nil => simp [lastOp]
  | cons x xs ih =>
    rw [List.flatMap_cons, lastOp_append, ih (fun y hy => h y (List.mem_cons_of_mem _ hy))]
    rcases h x List.mem_cons_self with hx | hx
    · rw [hx]
      by_cases hc : ∃ y ∈ xs, lastOp (F y) e ≠ none
      · have : ∃ y ∈ x :: xs, lastOp (F y) e ≠ none := by
          obtain ⟨y, hy, h'⟩ := hc; exact ⟨y, List.mem_cons_of_mem _ hy, h'⟩
        simp only [hc, this, if_true]; cases r <;> rfl
      · have : ¬ ∃ y ∈ x :: xs, lastOp (F y) e ≠ none := by
          rintro ⟨y, hy, h'⟩
          rcases List.mem_cons.1 hy with rfl | hy
          · exact h' hx
          · exact hc ⟨y, hy, h'⟩
        simp [hc, this, hx]
    · by_cases hr : r = none
      · subst hr
        have : ¬ ∃ y ∈ x :: xs, lastOp (F y) e ≠ none := by
          rintro ⟨y, hy, h'⟩
          rcases h y hy with h1 | h1 <;> exact h' h1
        have hc : ¬ ∃ y ∈ xs, lastOp (F y) e ≠ none := fun ⟨y, hy, h'⟩ => this ⟨y, List.mem_cons_of_mem _ hy, h'⟩
        simp [hc, this, hx]
      · have : ∃ y ∈ x :: xs, lastOp (F y) e ≠ none := ⟨x, List.mem_cons_self, by rw [hx]; exact hr⟩
        rw [if_pos this, hx]
        split <;> cases r <;> simp_all

/-! ## One op on one tuple -/

theorem colLog_pushL (ix : FalkorDbIndex) (ent : EntityType) (l a : String) (v : Value) (id : Nat) (add : Bool)
    (k : IndexKey) :
    colLog k (pushL ix ent l a v id add) =
      if (l, a) = k ∧ ix.hasColumn ent l a = true then
        (match encodeNumeric v with | some kk => [⟨kk, id, add⟩] | none => []) else [] := by
  unfold pushL colLog
  rw [encodeValue_spec]
  by_cases hc : ix.hasColumn ent l a = true
  · simp only [hc, if_true]
    cases encodeNumeric v <;> by_cases hk : (l, a) = k <;> simp [hk]
  · simp [hc]

theorem enc_inj (k d k' d' : Nat) (hd : d < W) (hd' : d' < W) : enc k d = enc k' d' ↔ k = k' ∧ d = d' := by
  constructor
  · intro h
    have h1 := congrArg keyOf h; have h2 := congrArg docOf h
    rw [keyOf_enc _ _ hd, keyOf_enc _ _ hd'] at h1
    rw [docOf_enc _ _ hd, docOf_enc _ _ hd'] at h2
    exact ⟨h1, h2⟩
  · rintro ⟨rfl, rfl⟩; rfl

/-- `encode` matches key `kk`. -/
def encIs (v : Option Value) (kk : Nat) : Bool :=
  match v with | some v => encodeNumeric v == some kk | none => false

/-- **One op on tuple `(kk, id0)` of column `(L, A)`**: untouched unless the op is for `id0`, `A`
    and carries label `L`; then the new value wins if it encodes to `kk`, else the old one is removed
    if it does. -/
theorem lastOp_op (ix : FalkorDbIndex) (ent : EntityType) (op : StageOp) (L A : String) (kk id0 : Nat)
    (hid : op.id < W) (hid0 : id0 < W) (hcol : ix.hasColumn ent L A = true) :
    lastOp (colLog (L, A) (opChanges ix ent op)) (enc kk id0) =
      if op.id = id0 ∧ op.attr = A ∧ L ∈ op.labels then
        (if encIs op.new kk then some true else if encIs op.old kk then some false else none)
      else none := by
  unfold opChanges
  rw [colLog_flatMap]
  -- every label block ends the same way
  let r : Option Bool :=
    if op.id = id0 ∧ op.attr = A then
      (if encIs op.new kk then some true else if encIs op.old kk then some false else none) else none
  have blk : ∀ l ∈ op.labels,
      lastOp (colLog (L, A)
        ((match op.old with | some o => pushL ix ent l op.attr o op.id false | none => []) ++
         (match op.new with | some v => pushL ix ent l op.attr v op.id true | none => []))) (enc kk id0) =
      if l = L then r else none := by
    intro l _
    rw [colLog_append, lastOp_append]
    have one : ∀ (ov : Option Value) (add : Bool),
        lastOp (colLog (L, A) (match ov with | some o => pushL ix ent l op.attr o op.id add | none => []))
          (enc kk id0) =
        if l = L ∧ op.attr = A ∧ op.id = id0 ∧ encIs ov kk = true then some add else none := by
      intro ov add
      cases ov with
      | none => simp [colLog, lastOp, encIs]
      | some o =>
        simp only [colLog_pushL]
        by_cases hk : (l, op.attr) = (L, A)
        · simp only [Prod.mk.injEq] at hk
          obtain ⟨rfl, rfl⟩ := hk
          simp only [hcol, and_self, if_true, true_and, encIs]
          cases hv : encodeNumeric o with
          | none => simp [lastOp]
          | some k' =>
            simp only [lastOp, IndexWrite.tup, List.filter_cons, List.filter_nil]
            by_cases hm : enc k' op.id = enc kk id0
            · rw [enc_inj _ _ _ _ hid hid0] at hm
              obtain ⟨rfl, rfl⟩ := hm
              simp
            · have : ¬ (k' = kk ∧ op.id = id0) := fun h => hm (by rw [h.1, h.2])
              simp only [hm, decide_false, Bool.false_eq_true, if_false, List.getLast?_nil, Option.map_none]
              split
              · rename_i hh; exfalso; apply this; simp_all
              · rfl
        · simp only [hk, false_and, if_false, lastOp, colLog, List.filter_nil, List.map_nil, List.getLast?_nil,
            Option.map_none]
          split
          · rename_i hh; exact absurd (by rw [hh.1, hh.2.1]) hk
          · rfl
    rw [one op.old false, one op.new true]
    by_cases hl : l = L
    · subst hl
      simp only [true_and, if_true, r]
      by_cases ha : op.attr = A <;> by_cases hi : op.id = id0 <;> simp [ha, hi] <;>
        cases encIs op.new kk <;> cases encIs op.old kk <;> simp
    · simp [hl]
  rw [lastOp_flatMap_common op.labels _ (enc kk id0) r (by
    intro l hl; rw [blk l hl]; by_cases h : l = L <;> simp [h])]
  by_cases hL : L ∈ op.labels
  · by_cases hr : r = none
    · rw [hr]
      have hl : (if (∃ x ∈ op.labels, lastOp (colLog (L, A)
          ((match op.old with | some o => pushL ix ent x op.attr o op.id false | none => []) ++
           (match op.new with | some v => pushL ix ent x op.attr v op.id true | none => []))) (enc kk id0) ≠ none)
          then (none : Option Bool) else none) = none := ite_self _
      rw [hl]
      simp only [r] at hr
      by_cases hc : op.id = id0 ∧ op.attr = A
      · rw [if_pos hc] at hr; rw [if_pos ⟨hc.1, hc.2, hL⟩, hr]
      · rw [if_neg (fun h => hc ⟨h.1, h.2.1⟩)]
    · have : ∃ x ∈ op.labels, lastOp (colLog (L, A)
          ((match op.old with | some o => pushL ix ent x op.attr o op.id false | none => []) ++
           (match op.new with | some v => pushL ix ent x op.attr v op.id true | none => []))) (enc kk id0) ≠ none :=
        ⟨L, hL, by rw [blk L hL]; simpa using hr⟩
      rw [if_pos this]
      simp only [r] at hr ⊢
      by_cases hc : op.id = id0 ∧ op.attr = A
      · simp [hc, hL]
      · simp [hc] at hr
  · have : ¬ ∃ x ∈ op.labels, lastOp (colLog (L, A)
          ((match op.old with | some o => pushL ix ent x op.attr o op.id false | none => []) ++
           (match op.new with | some v => pushL ix ent x op.attr v op.id true | none => []))) (enc kk id0) ≠ none := by
      rintro ⟨x, hx, h'⟩
      rw [blk x hx] at h'
      by_cases h : x = L
      · subst h; exact hL hx
      · simp [h] at h'
    rw [if_neg this]; simp [hL]

/-- **A list of ops on one tuple**: if all ops for `(id0, A)` under `L` agree (they are one op, or
    copies of it), the tuple ends as that op says; untouched otherwise. -/
theorem lastOp_ops (ix : FalkorDbIndex) (ent : EntityType) (ops : List StageOp) (L A : String) (kk id0 : Nat)
    (hids : ∀ op ∈ ops, op.id < W) (hid0 : id0 < W) (hcol : ix.hasColumn ent L A = true)
    (r : Option Bool)
    (huniq : ∀ op ∈ ops, op.id = id0 → op.attr = A → L ∈ op.labels →
      (if encIs op.new kk then some true else if encIs op.old kk then some false else none) = r) :
    lastOp (colLog (L, A) (opsChanges ix ent ops)) (enc kk id0) =
      if ∃ op ∈ ops, op.id = id0 ∧ op.attr = A ∧ L ∈ op.labels then r else none := by
  unfold opsChanges
  rw [colLog_flatMap]
  have hop := fun op (hop : op ∈ ops) => lastOp_op ix ent op L A kk id0 (hids op hop) hid0 hcol
  rw [lastOp_flatMap_common ops _ _ r (by
    intro op h; rw [hop op h]
    by_cases hc : op.id = id0 ∧ op.attr = A ∧ L ∈ op.labels
    · right; rw [if_pos hc]; exact huniq op h hc.1 hc.2.1 hc.2.2
    · left; rw [if_neg hc])]
  by_cases hex : ∃ op ∈ ops, op.id = id0 ∧ op.attr = A ∧ L ∈ op.labels
  · obtain ⟨op, hm, hc⟩ := hex
    by_cases hr : r = none
    · subst hr; split <;> split <;> rfl
    · have : ∃ x ∈ ops, lastOp (colLog (L, A) (opChanges ix ent x)) (enc kk id0) ≠ none :=
        ⟨op, hm, by rw [hop op hm, if_pos hc, huniq op hm hc.1 hc.2.1 hc.2.2]; exact hr⟩
      rw [if_pos this, if_pos ⟨op, hm, hc⟩]
  · have : ¬ ∃ x ∈ ops, lastOp (colLog (L, A) (opChanges ix ent x)) (enc kk id0) ≠ none := by
      rintro ⟨x, hx, h'⟩
      rw [hop x hx] at h'
      split at h'
      · rename_i hc; exact hex ⟨x, hx, hc⟩
      · exact h' rfl
    rw [if_neg this, if_neg hex]

/-! ## The staging functions, loop for loop -/

/-- `node_label_ids_into` (`graph_writes.rs:488-495`): clear, then the label matrix row. -/
def nodeLabelIdsInto (g : G) (id : Nat) (_out : List Nat) : List Nat := g.nodeLabels id

/-- `edge_type` (`graph_writes.rs:503-508`). -/
def edgeType (g : G) (id : Nat) : String := g.typeName (g.relType id)

/-- Generic: a fold whose step appends a block is a `flatMap`. -/
theorem foldl_appends {α} (f : IndexChanges → α → IndexChanges) (blk : α → List (IndexKey × IndexWrite))
    (hf : ∀ c x, f c x = { c with changes := c.changes ++ blk x }) :
    ∀ (l : List α) (c : IndexChanges), l.foldl f c = { c with changes := c.changes ++ l.flatMap blk } := by
  intro l
  induction l with
  | nil => intro c; simp
  | cons x xs ih => intro c; simp [hf, ih, List.append_assoc]

/-- A step that only appends when the entity is fixed. -/
theorem foldl_appends' {α} (ent : EntityType) (f : IndexChanges → α → IndexChanges)
    (blk : α → List (IndexKey × IndexWrite))
    (hf : ∀ c x, c.entity = ent → f c x = { c with changes := c.changes ++ blk x }) :
    ∀ (l : List α) (c : IndexChanges), c.entity = ent →
      l.foldl f c = { c with changes := c.changes ++ l.flatMap blk } := by
  intro l
  induction l with
  | nil => intro c _; simp
  | cons x xs ih =>
    intro c hc
    simp only [List.foldl_cons]
    rw [hf c x hc, ih _ (by simp [hc])]
    simp [List.append_assoc]

/-- The per-label body shared by the set paths: `remove old` (if any), then `add new`
    (`graph_writes.rs:79-85`, `:116-122`). -/
def setUnderLabel (g : G) (attr : String) (old : Option Value) (nv : Value) (id : Nat)
    (ch : IndexChanges) (label : String) : IndexChanges :=
  let ch := match old with | some o => ch.remove g.idx label attr o id | none => ch
  ch.add g.idx label attr nv id

theorem setUnderLabel_eq (g : G) (attr : String) (old : Option Value) (nv : Value) (id : Nat)
    (ch : IndexChanges) (label : String) (ent : EntityType) (h : ch.entity = ent) :
    setUnderLabel g attr old nv id ch label =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, [label], attr, old, some nv⟩ } := by
  unfold setUnderLabel
  cases old <;> simp [IndexChanges.add, IndexChanges.remove, push_eq, h, opChanges, List.append_assoc]

theorem setUnderLabels (g : G) (attr : String) (old : Option Value) (nv : Value) (id : Nat)
    (labels : List String) (ch : IndexChanges) (ent : EntityType) (h : ch.entity = ent) :
    labels.foldl (setUnderLabel g attr old nv id) ch =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, labels, attr, old, some nv⟩ } := by
  rw [foldl_appends' ent _ (fun l => opChanges g.idx ent ⟨id, [l], attr, old, some nv⟩)
    (fun c x hc => setUnderLabel_eq g attr old nv id c x ent hc) labels ch h]
  simp [opChanges]

theorem opsChanges_flatMap {α} (ix : FalkorDbIndex) (ent : EntityType) (l : List α) (F : α → List StageOp) :
    opsChanges ix ent (l.flatMap F) = l.flatMap (fun x => opsChanges ix ent (F x)) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [opsChanges, List.flatMap_cons, List.flatMap_append] at *; rw [ih]

theorem filterMap_split {α β} (f : α → Option β) (l : List α) :
    l.filterMap f = l.flatMap (fun x => [x].filterMap f) := by
  induction l with
  | nil => rfl
  | cons x xs ih => rw [List.filterMap_cons, List.flatMap_cons, ← ih]; cases h : f x <;> simp [h]

/-- `node_attrs_set`'s attribute loop (`graph_writes.rs:73-86`). -/
def naAttrLoop (g : G) (id : Nat) (labels : List Nat) (ch : IndexChanges) (an : Nat × Value) : IndexChanges :=
  match g.attrName an.1 with
  | none => ch
  | some attr =>
    let old := g.nodeAttr id an.1   -- hoisted out of the label loop
    (labels.map g.labelName).foldl (setUnderLabel g attr old an.2 id) ch

/-- `node_attrs_set` (`graph_writes.rs:62-89`). -/
def nodeAttrsSet (g : G) (attrs : List (Nat × List (Nat × Value))) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  (attrs.foldl (fun (st : IndexChanges × List Nat) (iw : Nat × List (Nat × Value)) =>
    let labels := nodeLabelIdsInto g iw.1 st.2
    (iw.2.foldl (naAttrLoop g iw.1 labels) st.1, labels)) (IndexChanges.new .node, [])).1

/-- Its ops. -/
def naOps (g : G) (id : Nat) (written : List (Nat × Value)) : List StageOp :=
  written.filterMap (fun an =>
    (g.attrName an.1).map (fun attr => ⟨id, (g.nodeLabels id).map g.labelName, attr, g.nodeAttr id an.1, some an.2⟩))
def nodeAttrsSetOps (g : G) (attrs : List (Nat × List (Nat × Value))) : List StageOp :=
  attrs.flatMap (fun iw => naOps g iw.1 iw.2)

theorem naAttrLoop_eq (g : G) (id : Nat) (ch : IndexChanges) (an : Nat × Value) (h : ch.entity = .node) :
    naAttrLoop g id (g.nodeLabels id) ch an =
      { ch with changes := ch.changes ++ opsChanges g.idx .node (naOps g id [an]) } := by
  unfold naAttrLoop naOps
  cases hn : g.attrName an.1 with
  | none => simp [opsChanges, hn]
  | some attr => simp only; rw [setUnderLabels _ _ _ _ _ _ _ .node h]; simp [opsChanges, hn]

/-- **`node_attrs_set` stages exactly its ops**: for each written attribute (with a name) of each
    node, under each of the node's labels, `remove old` (if any) then `add new`. -/
theorem nodeAttrsSet_changes (g : G) (attrs : List (Nat × List (Nat × Value))) :
    (nodeAttrsSet g attrs).changes =
      if g.idx.hasColumns .node then opsChanges g.idx .node (nodeAttrsSetOps g attrs) else [] := by
  unfold nodeAttrsSet
  by_cases hc : g.idx.hasColumns .node = true
  · simp only [hc, Bool.not_true, Bool.false_eq_true, if_false, if_true]
    have outer : ∀ (l : List (Nat × List (Nat × Value))) (c : IndexChanges) (buf : List Nat), c.entity = .node →
        (l.foldl (fun (st : IndexChanges × List Nat) (iw : Nat × List (Nat × Value)) =>
          let labels := nodeLabelIdsInto g iw.1 st.2
          (iw.2.foldl (naAttrLoop g iw.1 labels) st.1, labels)) (c, buf)).1 =
        { c with changes := c.changes ++ opsChanges g.idx .node (nodeAttrsSetOps g l) } := by
      intro l
      induction l with
      | nil => intro c buf _; simp [opsChanges, nodeAttrsSetOps]
      | cons iw rest ih =>
        intro c buf hent
        simp only [List.foldl_cons, nodeLabelIdsInto] at ih ⊢
        rw [foldl_appends' .node _ (fun an => opsChanges g.idx .node (naOps g iw.1 [an]))
          (fun c x hc => naAttrLoop_eq g iw.1 c x hc) iw.2 c hent]
        rw [ih _ _ (by simp [hent])]
        have e1 : naOps g iw.1 iw.2 = iw.2.flatMap (fun an => naOps g iw.1 [an]) := filterMap_split _ _
        simp only [nodeAttrsSetOps, List.flatMap_cons, opsChanges_flatMap, List.flatMap_append, e1,
          List.append_assoc]
        simp [opsChanges, List.flatMap_append, List.flatMap_assoc]
    rw [outer attrs _ [] rfl]
    rfl
  · simp only [Bool.not_eq_true] at hc
    simp [hc, IndexChanges.new]

/-! ## Semantics: what a unit of work does to one tuple -/

theorem nodup_keys_unique {α β} (l : List (α × β)) (h : (l.map (·.1)).Nodup) (k : α) (v v' : β)
    (h1 : (k, v) ∈ l) (h2 : (k, v') ∈ l) : v = v' := by
  induction l with
  | nil => simp at h1
  | cons p ps ih =>
    simp only [List.map_cons, List.nodup_cons] at h
    rcases List.mem_cons.1 h1 with h1 | h1 <;> rcases List.mem_cons.1 h2 with h2 | h2
    · rw [← h1] at h2; simp at h2; exact h2.symm ▸ rfl
    · exfalso; apply h.1; rw [← h1]; exact List.mem_map.2 ⟨_, h2, rfl⟩
    · exfalso; apply h.1; rw [← h2]; exact List.mem_map.2 ⟨_, h1, rfl⟩
    · exact ih h.2 h1 h2

theorem attrName_inj (g : G) (hn : g.attrNames.Nodup) (a b : Nat) (n : String)
    (ha : g.attrName a = some n) (hb : g.attrName b = some n) : a = b := by
  unfold G.attrName at ha hb
  have hlt : a < g.attrNames.length := by
    rcases Nat.lt_or_ge a g.attrNames.length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at ha; cases ha
  exact (hn.getElem?_inj hlt).1 (by rw [ha, hb])

/-- The outcome `lastOp` predicts for one op on key `kk`. -/
def outcome (old new : Option Value) (kk : Nat) : Option Bool :=
  if encIs new kk then some true else if encIs old kk then some false else none

/-- **`node_attrs_set`, the tuple of a written attribute.** For a node `id0` written in this call
    with attribute `aid` (name `A`) := `nv`, carrying a label named `L`, column `(L, A)` ends the
    tuple `(kk, id0)` as: present if `nv` encodes to `kk`; removed if only the old value does.
    So the old entry goes, the new one comes. Rests on the caller's uniqueness (`hkeys`: the map's
    node ids; `hw`: each node's attribute ids — `Pending::set_node_attribute` keeps them sorted and
    unique, `set_node_attributes` only `debug_assert!`s it). -/
theorem nodeAttrsSet_hit (g : G) (hn : g.attrNames.Nodup) (attrs : List (Nat × List (Nat × Value)))
    (hkeys : (attrs.map (·.1)).Nodup) (hw : ∀ iw ∈ attrs, (iw.2.map (·.1)).Nodup)
    (hids : ∀ iw ∈ attrs, iw.1 < W) (L A : String) (hcol : g.idx.hasColumn .node L A = true)
    (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value)) (aid : Nat) (nv : Value)
    (hmem : (id0, written) ∈ attrs) (hamem : (aid, nv) ∈ written) (hA : g.attrName aid = some A)
    (hL : L ∈ (g.nodeLabels id0).map g.labelName) :
    lastOp (colLog (L, A) (nodeAttrsSet g attrs).changes) (enc kk id0) =
      outcome (g.nodeAttr id0 aid) (some nv) kk := by
  have hcols : g.idx.hasColumns .node = true := by
    unfold FalkorDbIndex.hasColumn at hcol; unfold FalkorDbIndex.hasColumns
    cases h : g.idx.columns .node with
    | nil => simp [h, hmGet] at hcol
    | cons _ _ => simp
  rw [nodeAttrsSet_changes, if_pos hcols]
  have hops : ∀ op ∈ nodeAttrsSetOps g attrs, op.id < W := by
    intro op hop
    simp only [nodeAttrsSetOps, List.mem_flatMap, naOps, List.mem_filterMap, Option.map_eq_some_iff] at hop
    obtain ⟨iw, hiw, an, _, attr, _, rfl⟩ := hop
    exact hids iw hiw
  rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol (outcome (g.nodeAttr id0 aid) (some nv) kk)]
  · rw [if_pos]
    refine ⟨⟨id0, (g.nodeLabels id0).map g.labelName, A, g.nodeAttr id0 aid, some nv⟩, ?_, rfl, rfl, hL⟩
    simp only [nodeAttrsSetOps, List.mem_flatMap, naOps, List.mem_filterMap, Option.map_eq_some_iff]
    exact ⟨(id0, written), hmem, (aid, nv), hamem, A, hA, rfl⟩
  · intro op hop hid hattr _
    simp only [nodeAttrsSetOps, List.mem_flatMap, naOps, List.mem_filterMap, Option.map_eq_some_iff] at hop
    obtain ⟨⟨id', w'⟩, hiw, ⟨aid', nv'⟩, han, attr, hattr', rfl⟩ := hop
    simp only at hid hattr hattr' ⊢
    subst hid hattr
    have hw' : w' = written := nodup_keys_unique attrs hkeys _ _ _ hiw hmem
    subst hw'
    have ha : aid' = aid := attrName_inj g hn _ _ _ hattr' hA
    subst ha
    have hv : nv' = nv := nodup_keys_unique _ (hw _ hmem) _ _ _ han hamem
    subst hv
    rfl

/-- ...and every other tuple of every node column is untouched by `node_attrs_set`. -/
theorem nodeAttrsSet_miss (g : G) (attrs : List (Nat × List (Nat × Value)))
    (hids : ∀ iw ∈ attrs, iw.1 < W) (L A : String) (hcol : g.idx.hasColumn .node L A = true)
    (kk id0 : Nat) (hid0 : id0 < W)
    (hno : ¬ ∃ written aid nv, (id0, written) ∈ attrs ∧ (aid, nv) ∈ written ∧ g.attrName aid = some A ∧
      L ∈ (g.nodeLabels id0).map g.labelName) :
    lastOp (colLog (L, A) (nodeAttrsSet g attrs).changes) (enc kk id0) = none := by
  rw [nodeAttrsSet_changes]
  split
  · have hops : ∀ op ∈ nodeAttrsSetOps g attrs, op.id < W := by
      intro op hop
      simp only [nodeAttrsSetOps, List.mem_flatMap, naOps, List.mem_filterMap, Option.map_eq_some_iff] at hop
      obtain ⟨iw, hiw, an, _, attr, _, rfl⟩ := hop
      exact hids iw hiw
    rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol none (by
      intro op hop hid hattr hl; exfalso
      simp only [nodeAttrsSetOps, List.mem_flatMap, naOps, List.mem_filterMap, Option.map_eq_some_iff] at hop
      obtain ⟨⟨id', w'⟩, hiw, ⟨aid', nv'⟩, han, attr, hattr', rfl⟩ := hop
      simp only at hid hattr hl
      subst hid hattr
      exact hno ⟨w', aid', nv', hiw, han, hattr', hl⟩)]
    split <;> rfl
  · simp [colLog, lastOp]

/-! ## The other staging functions -/

theorem filterMap_toList {α β} (f : α → Option β) (l : List α) : l.filterMap f = l.flatMap (fun x => (f x).toList) := by
  induction l with
  | nil => rfl
  | cons x xs ih => rw [List.filterMap_cons, List.flatMap_cons, ← ih]; cases f x <;> rfl

theorem flatMap_single {α β} (f : α → β) (l : List α) : l.flatMap (fun x => [f x]) = l.map f := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [List.flatMap_cons, ih]

theorem flatMap_map' {α β γ} (f : α → β) (F : β → List γ) (l : List α) :
    (l.map f).flatMap F = l.flatMap (fun x => F (f x)) := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [List.flatMap_cons, ih]

/-- `remove (old, id)` if any, then `add (new, id)` if any, under one label — every staging body. -/
def stage (g : G) (attr : String) (old new : Option Value) (id : Nat) (ch : IndexChanges) (label : String) :
    IndexChanges :=
  let ch := match old with | some o => ch.remove g.idx label attr o id | none => ch
  match new with | some v => ch.add g.idx label attr v id | none => ch

theorem stage_eq (g : G) (attr : String) (old new : Option Value) (id : Nat) (ch : IndexChanges) (label : String)
    (ent : EntityType) (h : ch.entity = ent) :
    stage g attr old new id ch label =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, [label], attr, old, new⟩ } := by
  unfold stage
  subst h
  cases old <;> cases new <;> simp [IndexChanges.add, IndexChanges.remove, push_eq, opChanges, List.append_assoc]

theorem stage_labels (g : G) (attr : String) (old new : Option Value) (id : Nat) (labels : List String)
    (ch : IndexChanges) (ent : EntityType) (h : ch.entity = ent) :
    labels.foldl (stage g attr old new id) ch =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, labels, attr, old, new⟩ } := by
  rw [foldl_appends' ent _ (fun l => opChanges g.idx ent ⟨id, [l], attr, old, new⟩)
    (fun c x hc => stage_eq g attr old new id c x ent hc) labels ch h]
  simp [opChanges]

theorem add_is_stage (g : G) (ch : IndexChanges) (l a : String) (v : Value) (id : Nat) :
    ch.add g.idx l a v id = stage g a none (some v) id ch l := rfl
theorem remove_is_stage (g : G) (ch : IndexChanges) (l a : String) (v : Value) (id : Nat) :
    ch.remove g.idx l a v id = stage g a (some v) none id ch l := rfl

/-- Any fold whose step stages a list of ops is `opsChanges` of the concatenated ops. -/
theorem foldl_ops {α} (ent : EntityType) (ix : FalkorDbIndex) (f : IndexChanges → α → IndexChanges)
    (ops : α → List StageOp) (hf : ∀ c x, c.entity = ent → f c x = { c with changes := c.changes ++ opsChanges ix ent (ops x) })
    (l : List α) (c : IndexChanges) (h : c.entity = ent) :
    l.foldl f c = { c with changes := c.changes ++ opsChanges ix ent (l.flatMap ops) } := by
  rw [foldl_appends' ent f (fun x => opsChanges ix ent (ops x)) hf l c h, opsChanges_flatMap]

/-- `Vec::chunks_exact(n)` (`n > 0`): full chunks only (fuel = length; each chunk drops `n ≥ 1`). -/
def chunksGo (n : Nat) : Nat → List Value → List (List Value)
  | 0, _ => []
  | f + 1, l => if n ≤ l.length then l.take n :: chunksGo n f (l.drop n) else []
def chunksExact (n : Nat) (_hn : 0 < n) (l : List Value) : List (List Value) := chunksGo n l.length l

/-- `node_rows_set` (`graph_writes.rs:98-126`): node `ids[i]` takes `rows[i*n + j]` for `attr_ids[j]`,
    under the record's labels. -/
def nodeRowsSet (g : G) (ids labelIds attrIds : List Nat) (rows : List Value) : IndexChanges :=
  if h : !g.idx.hasColumns .node || attrIds.isEmpty then IndexChanges.new .node else
  have hn : 0 < attrIds.length := by
    cases attrIds with
    | nil => simp at h
    | cons _ _ => simp
  let attrs := attrIds.map g.attrName
  (ids.zip (chunksExact attrIds.length hn rows)).foldl (fun ch (idRow : Nat × List Value) =>
    ((attrIds.zip attrs).zip idRow.2).foldl (fun ch (x : (Nat × Option String) × Value) =>
      match x.1.2 with
      | none => ch
      | some attr =>
        let old := g.nodeAttr idRow.1 x.1.1
        (labelIds.map g.labelName).foldl (stage g attr old (some x.2) idRow.1) ch) ch) (IndexChanges.new .node)

def nodeRowsSetOps (g : G) (ids labelIds attrIds : List Nat) (rows : List Value) (hn : 0 < attrIds.length) :
    List StageOp :=
  (ids.zip (chunksExact attrIds.length hn rows)).flatMap (fun idRow =>
    ((attrIds.zip (attrIds.map g.attrName)).zip idRow.2).filterMap (fun x =>
      x.1.2.map (fun attr => ⟨idRow.1, labelIds.map g.labelName, attr, g.nodeAttr idRow.1 x.1.1, some x.2⟩)))

theorem nodeRowsSet_changes (g : G) (ids labelIds attrIds : List Nat) (rows : List Value)
    (hc : g.idx.hasColumns .node = true) (hn : 0 < attrIds.length) :
    (nodeRowsSet g ids labelIds attrIds rows).changes = opsChanges g.idx .node (nodeRowsSetOps g ids labelIds attrIds rows hn) := by
  have hne : attrIds.isEmpty = false := by cases attrIds <;> simp_all
  unfold nodeRowsSet
  rw [dif_neg (by simp [hc, hne])]
  rw [foldl_ops .node g.idx _ (fun idRow => ((attrIds.zip (attrIds.map g.attrName)).zip idRow.2).filterMap (fun x =>
      x.1.2.map (fun attr => (⟨idRow.1, labelIds.map g.labelName, attr, g.nodeAttr idRow.1 x.1.1, some x.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, nodeRowsSetOps]
  · intro c idRow hc'
    rw [foldl_ops .node g.idx _ (fun x => (x.1.2.map (fun attr =>
        (⟨idRow.1, labelIds.map g.labelName, attr, g.nodeAttr idRow.1 x.1.1, some x.2⟩ : StageOp))).toList) ?_ _ _ hc']
    · simp only [filterMap_toList]
    · intro c x hcx
      cases ha : x.1.2 with
      | none => simp [ha, opsChanges]
      | some attr => simp only [ha]; rw [stage_labels _ _ _ _ _ _ _ .node hcx]; simp [opsChanges]

/-- `new_node_attrs` (`graph_writes.rs:133-157`): adds under `new_labels[id]`. -/
def newNodeAttrs (g : G) (attrs : List (Nat × List (Nat × Value))) (newLabels : List (Nat × List Nat)) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  attrs.foldl (fun ch iw =>
    match hmGet newLabels iw.1 with
    | none => ch                          -- no labels this transaction
    | some labelIds =>
      iw.2.foldl (fun ch an =>
        match g.attrName an.1 with
        | none => ch
        | some attr => (labelIds.map g.labelName).foldl (fun ch l => ch.add g.idx l attr an.2 iw.1) ch) ch)
    (IndexChanges.new .node)

def newNodeAttrsOps (g : G) (attrs : List (Nat × List (Nat × Value))) (newLabels : List (Nat × List Nat)) : List StageOp :=
  attrs.flatMap (fun iw => match hmGet newLabels iw.1 with
    | none => []
    | some labelIds => iw.2.filterMap (fun an =>
        (g.attrName an.1).map (fun attr => ⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩)))

theorem addLabels (g : G) (attr : String) (v : Value) (id : Nat) (labels : List String) (ch : IndexChanges)
    (ent : EntityType) (h : ch.entity = ent) :
    labels.foldl (fun ch l => ch.add g.idx l attr v id) ch =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, labels, attr, none, some v⟩ } := by
  have : (fun ch l => IndexChanges.add ch g.idx l attr v id) = stage g attr none (some v) id := rfl
  rw [this, stage_labels _ _ _ _ _ _ _ ent h]

theorem removeLabels (g : G) (attr : String) (v : Value) (id : Nat) (labels : List String) (ch : IndexChanges)
    (ent : EntityType) (h : ch.entity = ent) :
    labels.foldl (fun ch l => ch.remove g.idx l attr v id) ch =
      { ch with changes := ch.changes ++ opChanges g.idx ent ⟨id, labels, attr, some v, none⟩ } := by
  have : (fun ch l => IndexChanges.remove ch g.idx l attr v id) = stage g attr (some v) none id := rfl
  rw [this, stage_labels _ _ _ _ _ _ _ ent h]

theorem newNodeAttrs_changes (g : G) (attrs : List (Nat × List (Nat × Value))) (newLabels : List (Nat × List Nat))
    (hc : g.idx.hasColumns .node = true) :
    (newNodeAttrs g attrs newLabels).changes = opsChanges g.idx .node (newNodeAttrsOps g attrs newLabels) := by
  unfold newNodeAttrs
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .node g.idx _ (fun iw => match hmGet newLabels iw.1 with
    | none => []
    | some labelIds => iw.2.filterMap (fun an =>
        (g.attrName an.1).map (fun attr => (⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, newNodeAttrsOps]
  · intro c iw hc'
    cases hl : hmGet newLabels iw.1 with
    | none => simp [opsChanges]
    | some labelIds =>
      simp only
      rw [foldl_ops .node g.idx _ (fun an => ((g.attrName an.1).map (fun attr =>
          (⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩ : StageOp))).toList) ?_ _ _ hc']
      · simp only [filterMap_toList]
      · intro c an hca
        cases ha : g.attrName an.1 with
        | none => simp [opsChanges]
        | some attr => simp only; rw [addLabels _ _ _ _ _ _ .node hca]; simp [opsChanges]

/-- `bulk_node_attrs` (`graph_writes.rs:161-182`). -/
def bulkNodeAttrs (g : G) (rows : List (Nat × List (Nat × Value))) (labelIds : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  rows.foldl (fun ch iw => iw.2.foldl (fun ch an =>
      match g.attrName an.1 with
      | none => ch
      | some attr => (labelIds.map g.labelName).foldl (fun ch l => ch.add g.idx l attr an.2 iw.1) ch) ch)
    (IndexChanges.new .node)

def bulkNodeAttrsOps (g : G) (rows : List (Nat × List (Nat × Value))) (labelIds : List Nat) : List StageOp :=
  rows.flatMap (fun iw => iw.2.filterMap (fun an =>
    (g.attrName an.1).map (fun attr => ⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩)))

theorem bulkNodeAttrs_changes (g : G) (rows : List (Nat × List (Nat × Value))) (labelIds : List Nat)
    (hc : g.idx.hasColumns .node = true) :
    (bulkNodeAttrs g rows labelIds).changes = opsChanges g.idx .node (bulkNodeAttrsOps g rows labelIds) := by
  unfold bulkNodeAttrs
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .node g.idx _ (fun iw => iw.2.filterMap (fun an =>
    (g.attrName an.1).map (fun attr => (⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, bulkNodeAttrsOps]
  · intro c iw hc'
    rw [foldl_ops .node g.idx _ (fun an => ((g.attrName an.1).map (fun attr =>
        (⟨iw.1, labelIds.map g.labelName, attr, none, some an.2⟩ : StageOp))).toList) ?_ _ _ hc']
    · simp only [filterMap_toList]
    · intro c an hca
      cases ha : g.attrName an.1 with
      | none => simp [opsChanges]
      | some attr => simp only; rw [addLabels _ _ _ _ _ _ .node hca]; simp [opsChanges]

/-- `add_node_under_label` (`graph_writes.rs:438-448`). -/
def addNodeUnderLabel (ch : IndexChanges) (g : G) (id lid : Nat) : IndexChanges :=
  let label := g.labelName lid
  (g.nodeAllAttrs id).foldl (fun ch av => ch.add g.idx label av.1 av.2 id) ch

/-- `remove_node_under_label` (`graph_writes.rs:451-461`). -/
def removeNodeUnderLabel (ch : IndexChanges) (g : G) (id lid : Nat) : IndexChanges :=
  let label := g.labelName lid
  (g.nodeAllAttrs id).foldl (fun ch av => ch.remove g.idx label av.1 av.2 id) ch

def underLabelOps (g : G) (id lid : Nat) (add : Bool) : List StageOp :=
  (g.nodeAllAttrs id).map (fun av =>
    if add then ⟨id, [g.labelName lid], av.1, none, some av.2⟩ else ⟨id, [g.labelName lid], av.1, some av.2, none⟩)

theorem addNodeUnderLabel_eq (ch : IndexChanges) (g : G) (id lid : Nat) (h : ch.entity = .node) :
    addNodeUnderLabel ch g id lid = { ch with changes := ch.changes ++ opsChanges g.idx .node (underLabelOps g id lid true) } := by
  unfold addNodeUnderLabel underLabelOps
  rw [foldl_ops .node g.idx _ (fun av => [(⟨id, [g.labelName lid], av.1, none, some av.2⟩ : StageOp)]) ?_ _ _ h]
  · simp [flatMap_single]
  · intro c av hc; rw [add_is_stage, stage_eq _ _ _ _ _ _ _ .node hc]; simp [opsChanges]

theorem removeNodeUnderLabel_eq (ch : IndexChanges) (g : G) (id lid : Nat) (h : ch.entity = .node) :
    removeNodeUnderLabel ch g id lid = { ch with changes := ch.changes ++ opsChanges g.idx .node (underLabelOps g id lid false) } := by
  unfold removeNodeUnderLabel underLabelOps
  rw [foldl_ops .node g.idx _ (fun av => [(⟨id, [g.labelName lid], av.1, some av.2, none⟩ : StageOp)]) ?_ _ _ h]
  · simp [flatMap_single]
  · intro c av hc; rw [remove_is_stage, stage_eq _ _ _ _ _ _ _ .node hc]; simp [opsChanges]

/-- `labels_added` (`graph_writes.rs:189-202`). -/
def labelsAdded (g : G) (rowsIds colsLabels : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  (rowsIds.zip colsLabels).foldl (fun ch p => addNodeUnderLabel ch g p.1 p.2) (IndexChanges.new .node)

/-- `labels_product_added` (`graph_writes.rs:206-221`). -/
def labelsProductAdded (g : G) (ids labelIds : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  labelIds.foldl (fun ch lid => ids.foldl (fun ch id => addNodeUnderLabel ch g id lid) ch) (IndexChanges.new .node)

/-- `labels_removed` (`graph_writes.rs:226-239`). -/
def labelsRemoved (g : G) (rowsIds colsLabels : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  (rowsIds.zip colsLabels).foldl (fun ch p => removeNodeUnderLabel ch g p.1 p.2) (IndexChanges.new .node)

theorem labelsAdded_changes (g : G) (r c : List Nat) (hc : g.idx.hasColumns .node = true) :
    (labelsAdded g r c).changes = opsChanges g.idx .node ((r.zip c).flatMap (fun p => underLabelOps g p.1 p.2 true)) := by
  unfold labelsAdded
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .node g.idx _ (fun p => underLabelOps g p.1 p.2 true) (fun c p h => addNodeUnderLabel_eq c g p.1 p.2 h) _ _ rfl]
  simp [IndexChanges.new]

theorem labelsRemoved_changes (g : G) (r c : List Nat) (hc : g.idx.hasColumns .node = true) :
    (labelsRemoved g r c).changes = opsChanges g.idx .node ((r.zip c).flatMap (fun p => underLabelOps g p.1 p.2 false)) := by
  unfold labelsRemoved
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .node g.idx _ (fun p => underLabelOps g p.1 p.2 false) (fun c p h => removeNodeUnderLabel_eq c g p.1 p.2 h) _ _ rfl]
  simp [IndexChanges.new]

theorem labelsProductAdded_changes (g : G) (ids labelIds : List Nat) (hc : g.idx.hasColumns .node = true) :
    (labelsProductAdded g ids labelIds).changes =
      opsChanges g.idx .node (labelIds.flatMap (fun lid => ids.flatMap (fun id => underLabelOps g id lid true))) := by
  unfold labelsProductAdded
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .node g.idx _ (fun lid => ids.flatMap (fun id => underLabelOps g id lid true)) ?_ _ _ rfl]
  · simp [IndexChanges.new]
  · intro c lid h
    rw [foldl_ops .node g.idx _ (fun id => underLabelOps g id lid true) (fun c id h => addNodeUnderLabel_eq c g id lid h) _ _ h]

/-- `nodes_deleted` (`graph_writes.rs:243-262`): every indexed value under every label. -/
def nodesDeleted (g : G) (ids : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .node then IndexChanges.new .node else
  (ids.foldl (fun (st : IndexChanges × List Nat) id =>
    let labels := nodeLabelIdsInto g id st.2
    ((g.nodeAllAttrs id).foldl (fun ch av =>
      (labels.map g.labelName).foldl (fun ch l => ch.remove g.idx l av.1 av.2 id) ch) st.1, labels))
    (IndexChanges.new .node, [])).1

def nodesDeletedOps (g : G) (ids : List Nat) : List StageOp :=
  ids.flatMap (fun id => (g.nodeAllAttrs id).map (fun av => ⟨id, (g.nodeLabels id).map g.labelName, av.1, some av.2, none⟩))

theorem nodesDeleted_changes (g : G) (ids : List Nat) (hc : g.idx.hasColumns .node = true) :
    (nodesDeleted g ids).changes = opsChanges g.idx .node (nodesDeletedOps g ids) := by
  unfold nodesDeleted
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  have outer : ∀ (l : List Nat) (c : IndexChanges) (buf : List Nat), c.entity = .node →
      (l.foldl (fun (st : IndexChanges × List Nat) id =>
        let labels := nodeLabelIdsInto g id st.2
        ((g.nodeAllAttrs id).foldl (fun ch av =>
          (labels.map g.labelName).foldl (fun ch l => ch.remove g.idx l av.1 av.2 id) ch) st.1, labels)) (c, buf)).1 =
      { c with changes := c.changes ++ opsChanges g.idx .node (nodesDeletedOps g l) } := by
    intro l
    induction l with
    | nil => intro c buf _; simp [nodesDeletedOps, opsChanges]
    | cons id rest ih =>
      intro c buf h
      simp only [List.foldl_cons, nodeLabelIdsInto] at ih ⊢
      rw [foldl_ops .node g.idx _ (fun av => [(⟨id, (g.nodeLabels id).map g.labelName, av.1, some av.2, none⟩ : StageOp)])
        (fun c av hc => by rw [removeLabels _ _ _ _ _ _ .node hc]; simp [opsChanges]) _ _ h]
      rw [ih _ _ (by simp [h])]
      simp [nodesDeletedOps, opsChanges, List.flatMap_append, List.flatMap_assoc, flatMap_map']
  rw [outer ids _ [] rfl]; rfl

/-! ### Relationships -/

/-- `match type_id { Some(t) => &types[t], None => edge_type(g, id) }` (`graph_writes.rs:304-307`). -/
def edgeTypeOf (g : G) (typeId : Option Nat) (id : Nat) : String :=
  match typeId with | some t => g.typeName t | none => edgeType g id

/-- `edge_attrs_set_with` (`graph_writes.rs:291-321`). -/
def edgeAttrsSetWith (g : G) (attrs : List (Nat × List (Nat × Value))) (typeId : Option Nat) : IndexChanges :=
  if !g.idx.hasColumns .relationship then IndexChanges.new .relationship else
  attrs.foldl (fun ch iw =>
    if iw.2.isEmpty then ch else
    let typeName := edgeTypeOf g typeId iw.1
    iw.2.foldl (fun ch an =>
      match g.attrName an.1 with
      | none => ch
      | some attr => stage g attr (g.relAttr iw.1 an.1) (some an.2) iw.1 ch typeName) ch)
    (IndexChanges.new .relationship)

/-- `edge_attrs_set` (`graph_writes.rs:273-278`). -/
def edgeAttrsSet (g : G) (attrs : List (Nat × List (Nat × Value))) : IndexChanges := edgeAttrsSetWith g attrs none
/-- `edge_attrs_set_of_type` (`graph_writes.rs:282-288`). -/
def edgeAttrsSetOfType (g : G) (t : Nat) (attrs : List (Nat × List (Nat × Value))) : IndexChanges :=
  edgeAttrsSetWith g attrs (some t)

def edgeAttrsSetOps (g : G) (attrs : List (Nat × List (Nat × Value))) (typeId : Option Nat) : List StageOp :=
  attrs.flatMap (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    ⟨iw.1, [edgeTypeOf g typeId iw.1], attr, g.relAttr iw.1 an.1, some an.2⟩)))

theorem edgeAttrsSetWith_changes (g : G) (attrs : List (Nat × List (Nat × Value))) (typeId : Option Nat)
    (hc : g.idx.hasColumns .relationship = true) :
    (edgeAttrsSetWith g attrs typeId).changes = opsChanges g.idx .relationship (edgeAttrsSetOps g attrs typeId) := by
  unfold edgeAttrsSetWith
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .relationship g.idx _ (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    (⟨iw.1, [edgeTypeOf g typeId iw.1], attr, g.relAttr iw.1 an.1, some an.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, edgeAttrsSetOps]
  · intro c iw hc'
    by_cases he : iw.2.isEmpty = true
    · simp [he, opsChanges, List.isEmpty_iff.1 he]
    · simp only [he, Bool.false_eq_true, if_false]
      rw [foldl_ops .relationship g.idx _ (fun an => ((g.attrName an.1).map (fun attr =>
        (⟨iw.1, [edgeTypeOf g typeId iw.1], attr, g.relAttr iw.1 an.1, some an.2⟩ : StageOp))).toList) ?_ _ _ hc']
      · simp only [filterMap_toList]
      · intro c an hca
        cases ha : g.attrName an.1 with
        | none => simp [opsChanges]
        | some attr => simp only; rw [stage_eq _ _ _ _ _ _ _ .relationship hca]; simp [opsChanges]

/-- `new_edge_attrs` (`graph_writes.rs:324-344`). -/
def newEdgeAttrs (g : G) (attrs : List (Nat × List (Nat × Value))) : IndexChanges :=
  if !g.idx.hasColumns .relationship then IndexChanges.new .relationship else
  attrs.foldl (fun ch iw =>
    if iw.2.isEmpty then ch else
    let typeName := edgeType g iw.1
    iw.2.foldl (fun ch an =>
      match g.attrName an.1 with
      | some attr => ch.add g.idx typeName attr an.2 iw.1
      | none => ch) ch)
    (IndexChanges.new .relationship)

def newEdgeAttrsOps (g : G) (attrs : List (Nat × List (Nat × Value))) : List StageOp :=
  attrs.flatMap (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    ⟨iw.1, [edgeType g iw.1], attr, none, some an.2⟩)))

theorem newEdgeAttrs_changes (g : G) (attrs : List (Nat × List (Nat × Value)))
    (hc : g.idx.hasColumns .relationship = true) :
    (newEdgeAttrs g attrs).changes = opsChanges g.idx .relationship (newEdgeAttrsOps g attrs) := by
  unfold newEdgeAttrs
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .relationship g.idx _ (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    (⟨iw.1, [edgeType g iw.1], attr, none, some an.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, newEdgeAttrsOps]
  · intro c iw hc'
    by_cases he : iw.2.isEmpty = true
    · simp [he, opsChanges, List.isEmpty_iff.1 he]
    · simp only [he, Bool.false_eq_true, if_false]
      rw [foldl_ops .relationship g.idx _ (fun an => ((g.attrName an.1).map (fun attr =>
        (⟨iw.1, [edgeType g iw.1], attr, none, some an.2⟩ : StageOp))).toList) ?_ _ _ hc']
      · simp only [filterMap_toList]
      · intro c an hca
        cases ha : g.attrName an.1 with
        | none => simp [opsChanges]
        | some attr => simp only; rw [add_is_stage, stage_eq _ _ _ _ _ _ _ .relationship hca]; simp [opsChanges]

/-- `bulk_edge_attrs` (`graph_writes.rs:350-368`). -/
def bulkEdgeAttrs (g : G) (rows : List (Nat × List (Nat × Value))) (t : Nat) : IndexChanges :=
  if !g.idx.hasColumns .relationship then IndexChanges.new .relationship else
  let typeName := g.typeName t
  rows.foldl (fun ch iw => iw.2.foldl (fun ch an =>
      match g.attrName an.1 with
      | some attr => ch.add g.idx typeName attr an.2 iw.1
      | none => ch) ch)
    (IndexChanges.new .relationship)

def bulkEdgeAttrsOps (g : G) (rows : List (Nat × List (Nat × Value))) (t : Nat) : List StageOp :=
  rows.flatMap (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    ⟨iw.1, [g.typeName t], attr, none, some an.2⟩)))

theorem bulkEdgeAttrs_changes (g : G) (rows : List (Nat × List (Nat × Value))) (t : Nat)
    (hc : g.idx.hasColumns .relationship = true) :
    (bulkEdgeAttrs g rows t).changes = opsChanges g.idx .relationship (bulkEdgeAttrsOps g rows t) := by
  unfold bulkEdgeAttrs
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .relationship g.idx _ (fun iw => iw.2.filterMap (fun an => (g.attrName an.1).map (fun attr =>
    (⟨iw.1, [g.typeName t], attr, none, some an.2⟩ : StageOp)))) ?_ _ _ rfl]
  · simp [IndexChanges.new, bulkEdgeAttrsOps]
  · intro c iw hc'
    rw [foldl_ops .relationship g.idx _ (fun an => ((g.attrName an.1).map (fun attr =>
      (⟨iw.1, [g.typeName t], attr, none, some an.2⟩ : StageOp))).toList) ?_ _ _ hc']
    · simp only [filterMap_toList]
    · intro c an hca
      cases ha : g.attrName an.1 with
      | none => simp [opsChanges]
      | some attr => simp only; rw [add_is_stage, stage_eq _ _ _ _ _ _ _ .relationship hca]; simp [opsChanges]

/-- `edges_deleted` (`graph_writes.rs:372-393`). -/
def edgesDeleted (g : G) (ids : List Nat) : IndexChanges :=
  if !g.idx.hasColumns .relationship then IndexChanges.new .relationship else
  ids.foldl (fun ch id =>
    let attrs := g.relAllAttrs id
    if attrs.isEmpty then ch else
    let typeName := edgeType g id
    attrs.foldl (fun ch av => ch.remove g.idx typeName av.1 av.2 id) ch)
    (IndexChanges.new .relationship)

def edgesDeletedOps (g : G) (ids : List Nat) : List StageOp :=
  ids.flatMap (fun id => (g.relAllAttrs id).map (fun av => ⟨id, [edgeType g id], av.1, some av.2, none⟩))

theorem edgesDeleted_changes (g : G) (ids : List Nat) (hc : g.idx.hasColumns .relationship = true) :
    (edgesDeleted g ids).changes = opsChanges g.idx .relationship (edgesDeletedOps g ids) := by
  unfold edgesDeleted
  simp only [hc, Bool.not_true, Bool.false_eq_true, if_false]
  rw [foldl_ops .relationship g.idx _ (fun id => (g.relAllAttrs id).map (fun av =>
    (⟨id, [edgeType g id], av.1, some av.2, none⟩ : StageOp))) ?_ _ _ rfl]
  · simp [IndexChanges.new, edgesDeletedOps]
  · intro c id hc'
    by_cases he : (g.relAllAttrs id).isEmpty = true
    · simp [he, opsChanges, List.isEmpty_iff.1 he]
    · simp only [he, Bool.false_eq_true, if_false]
      rw [foldl_ops .relationship g.idx _ (fun av => [(⟨id, [edgeType g id], av.1, some av.2, none⟩ : StageOp)])
        (fun c av hca => by rw [remove_is_stage, stage_eq _ _ _ _ _ _ _ .relationship hca]; simp [opsChanges]) _ _ hc']
      simp [flatMap_single]

/-! ## Semantics of the delete / label / edge paths -/

theorem nodeAllAttrs_unique (g : G) (hn : g.attrNames.Nodup) (id : Nat) (A : String) (v v' : Value)
    (h1 : (A, v) ∈ g.nodeAllAttrs id) (h2 : (A, v') ∈ g.nodeAllAttrs id) : v = v' := by
  have key : ∀ a w, (match g.attrNames[a]?, g.nodeAttr id a with
      | some n, some v => some (n, v) | _, _ => none) = some (A, w) →
      g.attrName a = some A ∧ g.nodeAttr id a = some w := by
    intro a w h; unfold G.attrName
    cases e1 : g.attrNames[a]? <;> cases e2 : g.nodeAttr id a <;> simp_all
  simp only [G.nodeAllAttrs, List.mem_filterMap, List.mem_range] at h1 h2
  obtain ⟨a, _, ha⟩ := h1
  obtain ⟨b, _, hb⟩ := h2
  obtain ⟨na, va⟩ := key a v ha
  obtain ⟨nb, vb⟩ := key b v' hb
  have : a = b := attrName_inj g hn a b A na nb
  subst this; rw [va] at vb; cases vb; rfl

theorem relAllAttrs_unique (g : G) (hn : g.attrNames.Nodup) (id : Nat) (A : String) (v v' : Value)
    (h1 : (A, v) ∈ g.relAllAttrs id) (h2 : (A, v') ∈ g.relAllAttrs id) : v = v' := by
  have key : ∀ a w, (match g.attrNames[a]?, g.relAttr id a with
      | some n, some v => some (n, v) | _, _ => none) = some (A, w) →
      g.attrName a = some A ∧ g.relAttr id a = some w := by
    intro a w h; unfold G.attrName
    cases e1 : g.attrNames[a]? <;> cases e2 : g.relAttr id a <;> simp_all
  simp only [G.relAllAttrs, List.mem_filterMap, List.mem_range] at h1 h2
  obtain ⟨a, _, ha⟩ := h1
  obtain ⟨b, _, hb⟩ := h2
  obtain ⟨na, va⟩ := key a v ha
  obtain ⟨nb, vb⟩ := key b v' hb
  have : a = b := attrName_inj g hn a b A na nb
  subst this; rw [va] at vb; cases vb; rfl

theorem hasColumns_of_hasColumn (x : FalkorDbIndex) (e : EntityType) (L A : String)
    (h : x.hasColumn e L A = true) : x.hasColumns e = true := by
  unfold FalkorDbIndex.hasColumn at h; unfold FalkorDbIndex.hasColumns
  cases hc : x.columns e with
  | nil => simp [hc, hmGet] at h
  | cons _ _ => simp

/-- **DELETE removes the deleted node's entries**: for a deleted node `id0` with attribute `A = v`
    under a label named `L`, the tuple `(kk, id0)` of column `(L, A)` ends removed if `v` encodes to
    `kk` — and `nodes_deleted` never adds anything. -/
theorem nodesDeleted_hit (g : G) (hn : g.attrNames.Nodup) (ids : List Nat) (hids : ∀ i ∈ ids, i < W)
    (L A : String) (hcol : g.idx.hasColumn .node L A = true) (kk id0 : Nat) (hid0 : id0 < W)
    (hmem : id0 ∈ ids) (v : Value) (hv : (A, v) ∈ g.nodeAllAttrs id0)
    (hL : L ∈ (g.nodeLabels id0).map g.labelName) :
    lastOp (colLog (L, A) (nodesDeleted g ids).changes) (enc kk id0) = outcome (some v) none kk := by
  rw [nodesDeleted_changes g ids (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ nodesDeletedOps g ids, op.id < W := by
    intro op hop; simp only [nodesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨i, hi, _, _, rfl⟩ := hop; exact hids i hi
  rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol (outcome (some v) none kk)]
  · rw [if_pos]
    exact ⟨⟨id0, (g.nodeLabels id0).map g.labelName, A, some v, none⟩,
      by simp only [nodesDeletedOps, List.mem_flatMap, List.mem_map]; exact ⟨id0, hmem, (A, v), hv, rfl⟩, rfl, rfl, hL⟩
  · intro op hop hid hattr _
    simp only [nodesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨i, _, ⟨A', v'⟩, hav, rfl⟩ := hop
    simp only at hid hattr ⊢; subst hid hattr
    rw [nodeAllAttrs_unique g hn _ _ _ _ hav hv]; rfl

/-- A change list built only from ops without a new value never ends a tuple added. -/
theorem lastOp_noadd (ix : FalkorDbIndex) (ent : EntityType) (ops : List StageOp) (k : IndexKey) (e : Nat)
    (h : ∀ op ∈ ops, op.new = none) : lastOp (colLog k (opsChanges ix ent ops)) e ≠ some true := by
  have hall : ∀ w ∈ colLog k (opsChanges ix ent ops), w.add = false := by
    intro w hw
    simp only [colLog, List.mem_map, List.mem_filter, opsChanges, List.mem_flatMap, opChanges] at hw
    obtain ⟨⟨kk, w'⟩, ⟨⟨op, hop, l, _, hin⟩, _⟩, rfl⟩ := hw
    rw [h op hop] at hin
    simp only [List.append_nil] at hin
    cases ho : op.old with
    | none => simp [ho] at hin
    | some o =>
      simp only [ho, pushL] at hin
      split at hin <;> simp at hin
      obtain ⟨_, rfl⟩ := hin; rfl
  unfold lastOp
  intro hc
  cases hl : (List.filter (fun w => decide (w.tup = e)) (colLog k (opsChanges ix ent ops))).getLast? with
  | none => rw [hl] at hc; cases hc
  | some w =>
    rw [hl] at hc; simp at hc
    have := List.mem_of_getLast? hl
    rw [List.mem_filter] at this
    rw [hall w this.1] at hc; cases hc

theorem nodesDeleted_noadd (g : G) (ids : List Nat) (k : IndexKey) (e : Nat) :
    lastOp (colLog k (nodesDeleted g ids).changes) e ≠ some true := by
  by_cases hc : g.idx.hasColumns .node = true
  · rw [nodesDeleted_changes g ids hc]
    apply lastOp_noadd
    intro op hop; simp only [nodesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨_, _, _, _, rfl⟩ := hop; rfl
  · simp only [Bool.not_eq_true] at hc
    simp [nodesDeleted, hc, IndexChanges.new, colLog, lastOp]

/-- **REMOVE `:L` removes the node's entries under `L`.** -/
theorem labelsRemoved_hit (g : G) (hn : g.attrNames.Nodup) (r c : List Nat) (hids : ∀ i ∈ r, i < W)
    (L A : String) (hcol : g.idx.hasColumn .node L A = true) (kk id0 lid : Nat) (hid0 : id0 < W)
    (hmem : (id0, lid) ∈ r.zip c) (hL : g.labelName lid = L) (v : Value) (hv : (A, v) ∈ g.nodeAllAttrs id0) :
    lastOp (colLog (L, A) (labelsRemoved g r c).changes) (enc kk id0) = outcome (some v) none kk := by
  rw [labelsRemoved_changes g r c (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ (r.zip c).flatMap (fun p => underLabelOps g p.1 p.2 false), op.id < W := by
    intro op hop; simp only [underLabelOps, List.mem_flatMap, List.mem_map, Bool.false_eq_true, if_false] at hop
    obtain ⟨p, hp, _, _, rfl⟩ := hop; exact hids _ (List.of_mem_zip hp).1
  rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol (outcome (some v) none kk)]
  · rw [if_pos]
    exact ⟨⟨id0, [g.labelName lid], A, some v, none⟩, by
      simp only [underLabelOps, List.mem_flatMap, List.mem_map, Bool.false_eq_true, if_false]
      exact ⟨(id0, lid), hmem, (A, v), hv, rfl⟩, rfl, rfl, by simp [hL]⟩
  · intro op hop hid hattr _
    simp only [underLabelOps, List.mem_flatMap, List.mem_map, Bool.false_eq_true, if_false] at hop
    obtain ⟨⟨i, l⟩, _, ⟨A', v'⟩, hav, rfl⟩ := hop
    simp only at hid hattr ⊢; subst hid hattr
    rw [nodeAllAttrs_unique g hn _ _ _ _ hav hv]; rfl

/-- **SET `:L` adds the node's current values under `L`.** -/
theorem labelsAdded_hit (g : G) (hn : g.attrNames.Nodup) (r c : List Nat) (hids : ∀ i ∈ r, i < W)
    (L A : String) (hcol : g.idx.hasColumn .node L A = true) (kk id0 lid : Nat) (hid0 : id0 < W)
    (hmem : (id0, lid) ∈ r.zip c) (hL : g.labelName lid = L) (v : Value) (hv : (A, v) ∈ g.nodeAllAttrs id0) :
    lastOp (colLog (L, A) (labelsAdded g r c).changes) (enc kk id0) = outcome none (some v) kk := by
  rw [labelsAdded_changes g r c (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ (r.zip c).flatMap (fun p => underLabelOps g p.1 p.2 true), op.id < W := by
    intro op hop; simp only [underLabelOps, List.mem_flatMap, List.mem_map, if_true] at hop
    obtain ⟨p, hp, _, _, rfl⟩ := hop; exact hids _ (List.of_mem_zip hp).1
  rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol (outcome none (some v) kk)]
  · rw [if_pos]
    exact ⟨⟨id0, [g.labelName lid], A, none, some v⟩, by
      simp only [underLabelOps, List.mem_flatMap, List.mem_map, if_true]
      exact ⟨(id0, lid), hmem, (A, v), hv, rfl⟩, rfl, rfl, by simp [hL]⟩
  · intro op hop hid hattr _
    simp only [underLabelOps, List.mem_flatMap, List.mem_map, if_true] at hop
    obtain ⟨⟨i, l⟩, _, ⟨A', v'⟩, hav, rfl⟩ := hop
    simp only at hid hattr ⊢; subst hid hattr
    rw [nodeAllAttrs_unique g hn _ _ _ _ hav hv]; rfl

/-- **DELETE of an edge removes its entries** (explicit or cascading). -/
theorem edgesDeleted_hit (g : G) (hn : g.attrNames.Nodup) (ids : List Nat) (hids : ∀ i ∈ ids, i < W)
    (T A : String) (hcol : g.idx.hasColumn .relationship T A = true) (kk id0 : Nat) (hid0 : id0 < W)
    (hmem : id0 ∈ ids) (v : Value) (hv : (A, v) ∈ g.relAllAttrs id0) (hT : edgeType g id0 = T) :
    lastOp (colLog (T, A) (edgesDeleted g ids).changes) (enc kk id0) = outcome (some v) none kk := by
  rw [edgesDeleted_changes g ids (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ edgesDeletedOps g ids, op.id < W := by
    intro op hop; simp only [edgesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨i, hi, _, _, rfl⟩ := hop; exact hids i hi
  rw [lastOp_ops g.idx .relationship _ T A kk id0 hops hid0 hcol (outcome (some v) none kk)]
  · rw [if_pos]
    exact ⟨⟨id0, [edgeType g id0], A, some v, none⟩,
      by simp only [edgesDeletedOps, List.mem_flatMap, List.mem_map]; exact ⟨id0, hmem, (A, v), hv, rfl⟩,
      rfl, rfl, by simp [hT]⟩
  · intro op hop hid hattr _
    simp only [edgesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨i, _, ⟨A', v'⟩, hav, rfl⟩ := hop
    simp only at hid hattr ⊢; subst hid hattr
    rw [relAllAttrs_unique g hn _ _ _ _ hav hv]; rfl

theorem edgesDeleted_noadd (g : G) (ids : List Nat) (k : IndexKey) (e : Nat) :
    lastOp (colLog k (edgesDeleted g ids).changes) e ≠ some true := by
  by_cases hc : g.idx.hasColumns .relationship = true
  · rw [edgesDeleted_changes g ids hc]
    apply lastOp_noadd
    intro op hop; simp only [edgesDeletedOps, List.mem_flatMap, List.mem_map] at hop
    obtain ⟨_, _, _, _, rfl⟩ := hop; rfl
  · simp only [Bool.not_eq_true] at hc
    simp [edgesDeleted, hc, IndexChanges.new, colLog, lastOp]

/-- **SET on an edge replaces its entry** (same uniqueness premises as `nodeAttrsSet_hit`). -/
theorem edgeAttrsSetWith_hit (g : G) (hn : g.attrNames.Nodup) (attrs : List (Nat × List (Nat × Value)))
    (typeId : Option Nat)
    (hkeys : (attrs.map (·.1)).Nodup) (hw : ∀ iw ∈ attrs, (iw.2.map (·.1)).Nodup)
    (hids : ∀ iw ∈ attrs, iw.1 < W) (T A : String) (hcol : g.idx.hasColumn .relationship T A = true)
    (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value)) (aid : Nat) (nv : Value)
    (hmem : (id0, written) ∈ attrs) (hamem : (aid, nv) ∈ written) (hA : g.attrName aid = some A)
    (hT : edgeTypeOf g typeId id0 = T) :
    lastOp (colLog (T, A) (edgeAttrsSetWith g attrs typeId).changes) (enc kk id0) =
      outcome (g.relAttr id0 aid) (some nv) kk := by
  rw [edgeAttrsSetWith_changes g attrs typeId (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ edgeAttrsSetOps g attrs typeId, op.id < W := by
    intro op hop
    simp only [edgeAttrsSetOps, List.mem_flatMap, List.mem_filterMap, Option.map_eq_some_iff] at hop
    obtain ⟨iw, hiw, an, _, attr, _, rfl⟩ := hop
    exact hids iw hiw
  rw [lastOp_ops g.idx .relationship _ T A kk id0 hops hid0 hcol (outcome (g.relAttr id0 aid) (some nv) kk)]
  · rw [if_pos]
    refine ⟨⟨id0, [edgeTypeOf g typeId id0], A, g.relAttr id0 aid, some nv⟩,
      ?_, rfl, rfl, by simp [hT]⟩
    simp only [edgeAttrsSetOps, List.mem_flatMap, List.mem_filterMap, Option.map_eq_some_iff]
    exact ⟨(id0, written), hmem, (aid, nv), hamem, A, hA, rfl⟩
  · intro op hop hid hattr _
    simp only [edgeAttrsSetOps, List.mem_flatMap, List.mem_filterMap, Option.map_eq_some_iff] at hop
    obtain ⟨⟨id', w'⟩, hiw, ⟨aid', nv'⟩, han, attr, hattr', rfl⟩ := hop
    simp only at hid hattr hattr' ⊢
    subst hid hattr
    have hw' : w' = written := nodup_keys_unique attrs hkeys _ _ _ hiw hmem
    subst hw'
    have ha : aid' = aid := attrName_inj g hn _ _ _ hattr' hA
    subst ha
    have hv : nv' = nv := nodup_keys_unique _ (hw _ hmem) _ _ _ han hamem
    subst hv
    rfl

/-- **Created nodes are indexed under the transaction's labels** (uniqueness premises as above). -/
theorem newNodeAttrs_hit (g : G) (hn : g.attrNames.Nodup) (attrs : List (Nat × List (Nat × Value)))
    (newLabels : List (Nat × List Nat))
    (hkeys : (attrs.map (·.1)).Nodup) (hw : ∀ iw ∈ attrs, (iw.2.map (·.1)).Nodup)
    (hids : ∀ iw ∈ attrs, iw.1 < W) (L A : String) (hcol : g.idx.hasColumn .node L A = true)
    (kk id0 : Nat) (hid0 : id0 < W) (written : List (Nat × Value)) (aid : Nat) (nv : Value) (lids : List Nat)
    (hmem : (id0, written) ∈ attrs) (hamem : (aid, nv) ∈ written) (hA : g.attrName aid = some A)
    (hlab : hmGet newLabels id0 = some lids) (hL : L ∈ lids.map g.labelName) :
    lastOp (colLog (L, A) (newNodeAttrs g attrs newLabels).changes) (enc kk id0) = outcome none (some nv) kk := by
  rw [newNodeAttrs_changes g attrs newLabels (hasColumns_of_hasColumn _ _ _ _ hcol)]
  have hops : ∀ op ∈ newNodeAttrsOps g attrs newLabels, op.id < W := by
    intro op hop
    simp only [newNodeAttrsOps, List.mem_flatMap] at hop
    obtain ⟨iw, hiw, hop⟩ := hop
    split at hop
    · simp at hop
    · simp only [List.mem_filterMap, Option.map_eq_some_iff] at hop
      obtain ⟨an, _, attr, _, rfl⟩ := hop; exact hids iw hiw
  rw [lastOp_ops g.idx .node _ L A kk id0 hops hid0 hcol (outcome none (some nv) kk)]
  · rw [if_pos]
    refine ⟨⟨id0, lids.map g.labelName, A, none, some nv⟩, ?_, rfl, rfl, hL⟩
    simp only [newNodeAttrsOps, List.mem_flatMap]
    refine ⟨(id0, written), hmem, ?_⟩
    simp only [hlab, List.mem_filterMap, Option.map_eq_some_iff]
    exact ⟨(aid, nv), hamem, A, hA, rfl⟩
  · intro op hop hid hattr _
    simp only [newNodeAttrsOps, List.mem_flatMap] at hop
    obtain ⟨⟨id', w'⟩, hiw, hop⟩ := hop
    split at hop
    · simp at hop
    · simp only [List.mem_filterMap, Option.map_eq_some_iff] at hop
      obtain ⟨⟨aid', nv'⟩, han, attr, hattr', rfl⟩ := hop
      simp only at hid hattr hattr' ⊢
      subst hid hattr
      have hw' : w' = written := nodup_keys_unique attrs hkeys _ _ _ hiw hmem
      subst hw'
      have ha : aid' = aid := attrName_inj g hn _ _ _ hattr' hA
      subst ha
      have hv : nv' = nv := nodup_keys_unique _ (hw _ hmem) _ _ _ han hamem
      subst hv
      rfl

end NI
