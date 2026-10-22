import FalkorNativeIndex.Index
set_option linter.unusedSimpArgs false

/-!
# Staged writes (`graph/src/index/falkordb/pending_writes.rs`) and `FalkorDbIndex::apply`
# (`falkordb_index.rs:364-400`) — #3307's "apply once per unit of work"

Proven (`apply_eq_sequential`, `applyColumn_eq_sequential`): collapsing a column's staged log with
`last_write_wins` and applying one `remove_batch` + one `insert_batch` gives **the same tree as
applying every staged change one by one** (`insert` / `remove`, in staging order), for any log
(adds and removes of one tuple interleaved arbitrarily). `extend_log` shows `extend` keeps each
column's changes in staging order, so a unit of work's whole sequence is what gets collapsed.

`sort_by_key` is a *stable* sort: modelled as `List.mergeSort` (stable in core:
`List.sublist_mergeSort`). The `(key, id)` tuple order is the order of `enc key id` (cow_btree
`enc_lt_iff`, ids `< 2^64`).
-/

namespace NI

/-- `IndexWrite` (`pending_writes.rs:24-28`). -/
structure IndexWrite where
  key : Nat
  id : Nat
  add : Bool
deriving DecidableEq, Repr

def IndexWrite.tup (w : IndexWrite) : Nat := enc w.key w.id

/-- `IndexChanges` (`pending_writes.rs:33-36`). -/
structure IndexChanges where
  entity : EntityType
  changes : List (IndexKey × IndexWrite)

namespace IndexChanges
/-- `new` (`:41-46`). -/
def new (e : EntityType) : IndexChanges := ⟨e, []⟩
/-- `is_empty` (`:50-52`). -/
def isEmpty (c : IndexChanges) : Bool := c.changes.isEmpty
/-- `push` (`:81-95`): staged only if the column exists and indexes the value. -/
def push (c : IndexChanges) (ix : FalkorDbIndex) (label attr : String) (v : Value) (id : Nat) (add : Bool) :
    IndexChanges :=
  match ix.encodeValue c.entity label attr v with
  | none => c
  | some key => { c with changes := c.changes ++ [((label, attr), ⟨key, id, add⟩)] }
/-- `remove` (`:56-65`). -/
def remove (c : IndexChanges) (ix : FalkorDbIndex) (label attr : String) (v : Value) (id : Nat) : IndexChanges :=
  c.push ix label attr v id false
/-- `add` (`:69-78`). -/
def add (c : IndexChanges) (ix : FalkorDbIndex) (label attr : String) (v : Value) (id : Nat) : IndexChanges :=
  c.push ix label attr v id true
end IndexChanges

/-- `ColumnWrites` = `HashMap<IndexKey, Vec<IndexWrite>>`. -/
abbrev ColumnWrites := List (IndexKey × List IndexWrite)

/-- `columns.entry(key).or_default().push(write)`. -/
def pushWrite (m : ColumnWrites) (k : IndexKey) (w : IndexWrite) : ColumnWrites :=
  match hmGet m k with
  | some _ => hmUpdate m k (fun l => l ++ [w])
  | none => m ++ [(k, [w])]

/-- `PendingIndexWrites` (`pending_writes.rs:107-110`). -/
structure PendingIndexWrites where
  nodes : ColumnWrites
  edges : ColumnWrites

namespace PendingIndexWrites
def empty : PendingIndexWrites := ⟨[], []⟩
/-- `is_empty` (`:115-117`). -/
def isEmpty (p : PendingIndexWrites) : Bool := p.nodes.isEmpty && p.edges.isEmpty
/-- `drain` (`:121-129`): the entity's columns, leaving that side empty. -/
def drain (p : PendingIndexWrites) : EntityType → ColumnWrites × PendingIndexWrites
  | .node => (p.nodes, { p with nodes := [] })
  | .relationship => (p.edges, { p with edges := [] })
/-- `clear` (`:132-135`). -/
def clear (_p : PendingIndexWrites) : PendingIndexWrites := ⟨[], []⟩
/-- `extend` (`:138-149`). -/
def extend (p : PendingIndexWrites) (c : IndexChanges) : PendingIndexWrites :=
  match c.entity with
  | .node => { p with nodes := c.changes.foldl (fun m kw => pushWrite m kw.1 kw.2) p.nodes }
  | .relationship => { p with edges := c.changes.foldl (fun m kw => pushWrite m kw.1 kw.2) p.edges }
end PendingIndexWrites

/-! ## `last_write_wins` (`pending_writes.rs:158-179`) -/

/-- `log.sort_by_key(|w| (w.key, w.id))` — stable. -/
def sortLog (log : List IndexWrite) : List IndexWrite := log.mergeSort (fun a b => decide (a.tup ≤ b.tup))

/-- The loop body: keep `w` unless `log[i + 1]` is the same tuple (`:166-172`). -/
def lwwGo : List IndexWrite → List IndexWrite
  | [] => []
  | [w] => [w]
  | w :: w' :: r => if w'.tup = w.tup then lwwGo (w' :: r) else w :: lwwGo (w' :: r)

/-- `last_write_wins(log, removes, adds)`: both outputs cleared first, then filled (`:163-178`).
    Returns `(sorted log, removes, adds)`. -/
def lastWriteWins (log : List IndexWrite) (_removes _adds : List Nat) : List IndexWrite × List Nat × List Nat :=
  let s := sortLog log
  let kept := lwwGo s
  (s, (kept.filter (fun w => !w.add)).map (·.tup), (kept.filter (·.add)).map (·.tup))

/-- The last staged change to tuple `e`, if any. -/
def lastOp (log : List IndexWrite) (e : Nat) : Option Bool :=
  ((log.filter (fun w => w.tup = e)).getLast?).map (·.add)

/-! ### Stability: sorting keeps each tuple's history in order -/

theorem sortLog_filter (log : List IndexWrite) (e : Nat) :
    (sortLog log).filter (fun w => w.tup = e) = log.filter (fun w => w.tup = e) := by
  have trans : ∀ a b c : IndexWrite, decide (a.tup ≤ b.tup) = true → decide (b.tup ≤ c.tup) = true →
      decide (a.tup ≤ c.tup) = true := by intro a b c h1 h2; simp at *; omega
  have total : ∀ a b : IndexWrite, (decide (a.tup ≤ b.tup) || decide (b.tup ≤ a.tup)) = true := by
    intro a b; simp; omega
  have hpw : (log.filter (fun w => w.tup = e)).Pairwise (fun a b => decide (a.tup ≤ b.tup) = true) := by
    apply List.Pairwise.imp_of_mem _ (List.pairwise_of_forall (R := fun _ _ => True) (fun _ _ => trivial))
    intro a b ha hb _
    simp [List.mem_filter] at ha hb; simp [ha.2, hb.2]
  have hsub := List.sublist_mergeSort trans total hpw (List.filter_sublist (l := log))
  have h1 := hsub.filter (fun w => decide (w.tup = e))
  rw [List.filter_filter] at h1
  simp only [Bool.and_self] at h1
  have hlen : ((sortLog log).filter (fun w => w.tup = e)).length = (log.filter (fun w => w.tup = e)).length :=
    ((List.mergeSort_perm log _).filter _).length_eq
  exact (h1.eq_of_length hlen.symm).symm

/-! ### The loop on a sorted log -/

def SortedTup (s : List IndexWrite) : Prop := s.Pairwise (fun a b => a.tup ≤ b.tup)

theorem lastOp_cons (a : IndexWrite) (l : List IndexWrite) (e : Nat) :
    lastOp (a :: l) e = match lastOp l e with
      | some b => some b
      | none => if a.tup = e then some a.add else none := by
  unfold lastOp
  by_cases h : a.tup = e
  · rw [List.filter_cons_of_pos (by simp [h]), List.getLast?_cons]
    cases hl : (List.filter (fun w => decide (w.tup = e)) l).getLast? <;> simp [h]
  · rw [List.filter_cons_of_neg (by simp [h])]
    cases (List.filter (fun w => decide (w.tup = e)) l).getLast? <;> simp [h]

theorem lastOp_none (l : List IndexWrite) (e : Nat) (h : ∀ x ∈ l, x.tup ≠ e) : lastOp l e = none := by
  unfold lastOp
  rw [List.filter_eq_nil_iff.2 (by intro x hx; simpa using h x hx)]
  rfl

theorem lastOp_head (a : IndexWrite) (l : List IndexWrite) : lastOp (a :: l) a.tup ≠ none := by
  rw [lastOp_cons]; cases lastOp l a.tup <;> simp

theorem lwwGo_spec : ∀ (s : List IndexWrite), SortedTup s → ∀ (e : Nat) (b : Bool),
    (∃ w ∈ lwwGo s, w.tup = e ∧ w.add = b) ↔ lastOp s e = some b
  | [], _, e, b => by simp [lwwGo, lastOp]
  | [w], _, e, b => by
    rw [lastOp_cons, lastOp_none [] e (by simp)]
    simp only [lwwGo, List.mem_singleton, exists_eq_left]
    by_cases h : w.tup = e <;> simp [h]
  | w :: w' :: r, hs, e, b => by
    have hs' : SortedTup (w' :: r) := hs.of_cons
    have ih := lwwGo_spec (w' :: r) hs' e b
    have hle : ∀ x ∈ w' :: r, w.tup ≤ x.tup := fun x hx => List.rel_of_pairwise_cons hs hx
    unfold lwwGo
    rw [lastOp_cons]
    by_cases heq : w'.tup = w.tup
    · simp only [heq, if_true, ih]
      by_cases he : w.tup = e
      · have : lastOp (w' :: r) e ≠ none := by rw [← he, ← heq]; exact lastOp_head w' r
        cases h : lastOp (w' :: r) e
        · exact absurd h this
        · simp
      · cases lastOp (w' :: r) e <;> simp [he]
    · simp only [heq, if_false, List.mem_cons, exists_eq_or_imp]
      have hlt : ∀ x ∈ w' :: r, w.tup < x.tup := by
        intro x hx
        have h1 := hle x hx
        rcases List.mem_cons.1 hx with hx | hx
        · subst hx; omega
        · have : w'.tup ≤ x.tup := List.rel_of_pairwise_cons hs' hx
          have : w.tup ≤ w'.tup := hle w' (by simp)
          omega
      by_cases he : w.tup = e
      · have hn : lastOp (w' :: r) e = none :=
          lastOp_none _ e (by intro x hx; have := hlt x hx; omega)
        rw [hn] at ih ⊢
        simp only [he, if_true, true_and, Option.some.injEq]
        constructor
        · rintro (h | ⟨x, hx, h1, h2⟩)
          · exact h
          · exact absurd (ih.1 ⟨x, hx, h1, h2⟩) (by simp)
        · intro h; exact Or.inl h
      · simp only [he, false_and, false_or, ih]
        cases lastOp (w' :: r) e <;> simp [he]

/-- **`last_write_wins` keeps exactly each tuple's last staged change.** -/
theorem lastWriteWins_spec (log : List IndexWrite) (rm ad : List Nat) (e : Nat) :
    (e ∈ (lastWriteWins log rm ad).2.2 ↔ lastOp log e = some true) ∧
    (e ∈ (lastWriteWins log rm ad).2.1 ↔ lastOp log e = some false) := by
  have hs : SortedTup (sortLog log) := by
    have := List.pairwise_mergeSort (le := fun a b : IndexWrite => decide (a.tup ≤ b.tup))
      (by intro a b c h1 h2; simp at *; omega) (by intro a b; simp; omega) log
    exact this.imp (by intro a b h; simpa using h)
  have hlast : lastOp (sortLog log) e = lastOp log e := by
    unfold lastOp; rw [sortLog_filter]
  rw [← hlast]
  constructor
  · rw [← lwwGo_spec _ hs e true]
    simp only [lastWriteWins, List.mem_map, List.mem_filter]
    constructor
    · rintro ⟨w, ⟨h1, h2⟩, h3⟩; exact ⟨w, h1, h3, h2⟩
    · rintro ⟨w, h1, h3, h2⟩; exact ⟨w, ⟨h1, h2⟩, h3⟩
  · rw [← lwwGo_spec _ hs e false]
    simp only [lastWriteWins, List.mem_map, List.mem_filter]
    constructor
    · rintro ⟨w, ⟨h1, h2⟩, h3⟩; exact ⟨w, h1, h3, by simpa using h2⟩
    · rintro ⟨w, h1, h3, h2⟩; exact ⟨w, ⟨h1, by simpa using h2⟩, h3⟩

/-! ## One column: collapsed batch = one by one -/

/-- Applying the staged changes one by one, `CowBTree::insert` / `remove` each. -/
def seqTree (t : List Nat) (log : List IndexWrite) : List Nat :=
  log.foldl (fun t w => if w.add then tInsert t w.tup else tRemove t w.tup) t

theorem seqTree_sorted : ∀ (log : List IndexWrite) (t : List Nat), Sorted t → Sorted (seqTree t log)
  | [], t, h => h
  | w :: ws, t, h => by
    simp only [seqTree, List.foldl_cons]
    apply seqTree_sorted ws
    split
    · exact sorted_sins _ _ h
    · exact sorted_erase _ _ h

theorem step_mem (t : List Nat) (h : Sorted t) (T e : Nat) (A : Bool) :
    e ∈ (if A = true then tInsert t T else tRemove t T) ↔ (if T = e then A = true else e ∈ t) := by
  by_cases hw : T = e
  · subst hw
    cases A
    · simp [tRemove, mem_erase_sorted _ _ h]
    · simp [tInsert, mem_sins]
  · rw [if_neg hw]
    cases A
    · simp only [Bool.false_eq_true, if_false, tRemove, mem_erase_sorted _ _ h]
      exact ⟨fun h1 => h1.1, fun h1 => ⟨h1, Ne.symm hw⟩⟩
    · simp only [if_true, tInsert, mem_sins]
      exact ⟨fun h1 => h1.elim id (fun h2 => absurd h2.symm hw), Or.inl⟩

theorem mem_seqTree (log : List IndexWrite) : ∀ (t : List Nat), Sorted t → ∀ e,
    e ∈ seqTree t log ↔ (match lastOp log e with | some b => b = true | none => e ∈ t) := by
  induction log with
  | nil => intro t _ e; simp [seqTree, lastOp]
  | cons w ws ih =>
    intro t h e
    simp only [seqTree, List.foldl_cons]
    have hs : Sorted (if w.add then tInsert t w.tup else tRemove t w.tup) := by
      split
      · exact sorted_sins _ _ h
      · exact sorted_erase _ _ h
    have ih' := ih _ hs e
    simp only [seqTree] at ih'
    rw [ih', lastOp_cons]
    cases lastOp ws e with
    | some b => rfl
    | none =>
      have key := step_mem t h w.tup e w.add
      by_cases hw : w.tup = e
      · rw [if_pos hw] at key ⊢; exact key
      · rw [if_neg hw] at key ⊢; exact key

/-- The column step of `apply_writes` (`falkordb_index.rs:396-398`): collapse, remove, add. -/
def applyColumn (c : IndexColumn) (log : List IndexWrite) (removes adds : List Nat) :
    IndexColumn × List Nat × List Nat :=
  let (_, removes, adds) := lastWriteWins log removes adds
  let (c, removes) := c.removeEncoded removes
  let (c, adds) := c.addEncoded adds
  (c, removes, adds)

/-- **#3307's claim, per column**: the collapsed batch leaves exactly the tree that applying the
    staged changes one by one would. -/
theorem applyColumn_eq_sequential (ix : NumericIndex) (hs : Sorted ix.tree) (log : List IndexWrite)
    (rm ad : List Nat) :
    (applyColumn (.numeric ix) log rm ad).1.ix.tree = seqTree ix.tree log := by
  apply sorted_ext
  · simp only [applyColumn, IndexColumn.removeEncoded, IndexColumn.addEncoded, IndexColumn.ix,
      NumericIndex.addEncoded, NumericIndex.removeEncoded]
    exact sorted_insertBatch _ _ (sorted_removeBatch _ _ hs)
  · exact seqTree_sorted _ _ hs
  · intro e
    rw [mem_seqTree _ _ hs]
    have lw := lastWriteWins_spec log rm ad e
    simp only [applyColumn, IndexColumn.removeEncoded, IndexColumn.addEncoded, IndexColumn.ix,
      NumericIndex.addEncoded, NumericIndex.removeEncoded, mem_insertBatch, mem_removeBatch,
      (sortDedup_spec _).2]
    rw [lw.1, lw.2]
    cases lastOp log e with
    | none => simp
    | some b => cases b <;> simp

/-! ## `FalkorDbIndex::apply_writes` / `apply` -/

/-- One iteration of `apply_writes`' loop (`falkordb_index.rs:392-399`). -/
def applyStep (e : EntityType) (st : FalkorDbIndex × List Nat × List Nat) (kl : IndexKey × List IndexWrite) :
    FalkorDbIndex × List Nat × List Nat :=
  match hmGet (st.1.columns e) kl.1 with
  | none => st                                                    -- column dropped: skipped
  | some entry =>
    let r := applyColumn entry.column kl.2 st.2.1 st.2.2
    (st.1.setColumns e (hmInsert (st.1.columns e) kl.1 ⟨r.1⟩), r.2.1, r.2.2)

/-- `apply_writes` (`falkordb_index.rs:384-400`): one pair of buffers for the whole unit of work. -/
def FalkorDbIndex.applyWrites (x : FalkorDbIndex) (e : EntityType) (writes : ColumnWrites) : FalkorDbIndex :=
  (writes.foldl (applyStep e) (x, [], [])).1

/-- `apply` (`falkordb_index.rs:364-376`): nodes then relationships; leaves `writes` empty. -/
def FalkorDbIndex.apply (x : FalkorDbIndex) (w : PendingIndexWrites) : FalkorDbIndex × PendingIndexWrites :=
  if w.isEmpty then (x, w)
  else
    let d1 := w.drain .node
    let x := x.applyWrites .node d1.1
    let d2 := d1.2.drain .relationship
    (x.applyWrites .relationship d2.1, d2.2)

theorem apply_empties (x : FalkorDbIndex) (w : PendingIndexWrites) : (x.apply w).2.isEmpty = true := by
  unfold FalkorDbIndex.apply
  split
  · assumption
  · rfl

def ColsSorted (x : FalkorDbIndex) (e : EntityType) : Prop :=
  ∀ l a ix, x.numeric e l a = some ix → Sorted ix.tree

theorem numeric_setColumns_insert (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (c : IndexColumn) (l a : String) :
    (x.setColumns e (hmInsert (x.columns e) k ⟨c⟩)).numeric e l a =
      if k = (l, a) then some c.ix else x.numeric e l a := by
  simp only [FalkorDbIndex.numeric, columns_setColumns, if_true, hmGet_insert]
  split <;> rfl

theorem applyFold_numeric (e : EntityType) (writes : ColumnWrites) :
    ∀ (x : FalkorDbIndex) (r a : List Nat), (writes.map (·.1)).Nodup → ColsSorted x e → ∀ (l at' : String),
    ((writes.foldl (applyStep e) (x, r, a)).1).numeric e l at' =
      match hmGet writes (l, at') with
      | none => x.numeric e l at'
      | some log => (x.numeric e l at').map (fun ix => ⟨seqTree ix.tree log⟩) := by
  induction writes with
  | nil => intro x r a _ _ l at'; simp [hmGet]
  | cons kl ws ih =>
    intro x r a hnd hs l at'
    obtain ⟨k, log⟩ := kl
    simp only [List.map_cons, List.nodup_cons] at hnd
    have notin : hmGet ws k = none := by
      unfold hmGet; rw [Option.map_eq_none_iff, List.find?_eq_none]
      intro p hp; simp only [beq_iff_eq]; intro h; apply hnd.1; rw [← h]; exact List.mem_map_of_mem hp
    have getcons : hmGet ((k, log) :: ws) (l, at') = if k = (l, at') then some log else hmGet ws (l, at') := by
      unfold hmGet; by_cases hk : k = (l, at') <;> simp [hk]
    simp only [List.foldl_cons, applyStep]
    rw [getcons]
    cases hc : hmGet (x.columns e) k with
    | none =>
      simp only
      rw [ih x r a hnd.2 hs l at']
      by_cases hk : k = (l, at')
      · subst hk
        have hn : x.numeric e l at' = none := by simp [FalkorDbIndex.numeric, hc]
        simp [notin, hn]
      · simp [hk]
    | some entry =>
      obtain ⟨⟨ix⟩⟩ := entry
      have hix : x.numeric e k.1 k.2 = some ix := by simp [FalkorDbIndex.numeric, hc, IndexColumn.ix]
      have hsix : Sorted ix.tree := hs _ _ ix hix
      have eqcol : (applyColumn (.numeric ix) log r a).1.ix = ⟨seqTree ix.tree log⟩ := by
        have := applyColumn_eq_sequential ix hsix log r a
        cases hcol : (applyColumn (.numeric ix) log r a).1 with
        | numeric ix' => rw [hcol] at this; simp [IndexColumn.ix] at this ⊢; cases ix'; simp_all
      have hs' : ColsSorted (x.setColumns e (hmInsert (x.columns e) k ⟨(applyColumn (.numeric ix) log r a).1⟩)) e := by
        intro l' a' ix' h
        rw [numeric_setColumns_insert] at h
        split at h
        · simp at h; rw [← h, eqcol]; exact seqTree_sorted _ _ hsix
        · exact hs _ _ _ h
      simp only
      rw [ih _ _ _ hnd.2 hs' l at', numeric_setColumns_insert]
      by_cases hk : k = (l, at')
      · subst hk
        simp [notin, eqcol, hix]
      · simp [hk]

/-- **`apply_writes`, per column**: a staged column's tree becomes `seqTree` of its log (the
    changes applied one by one, in staging order); unstaged and dropped columns are untouched. -/
theorem applyWrites_numeric (x : FalkorDbIndex) (e : EntityType) (writes : ColumnWrites)
    (hnd : (writes.map (·.1)).Nodup) (hs : ColsSorted x e) (l a : String) :
    (x.applyWrites e writes).numeric e l a =
      match hmGet writes (l, a) with
      | none => x.numeric e l a
      | some log => (x.numeric e l a).map (fun ix => ⟨seqTree ix.tree log⟩) :=
  applyFold_numeric e writes x [] [] hnd hs l a

/-! ## `extend` keeps staging order per column -/

theorem hmGet_pushWrite (m : ColumnWrites) (k k' : IndexKey) (w : IndexWrite) :
    hmGet (pushWrite m k w) k' = if k = k' then some ((hmGet m k').getD [] ++ [w]) else hmGet m k' := by
  unfold pushWrite
  cases h : hmGet m k with
  | some l =>
    rw [hmGet_update]
    by_cases hk : k = k'
    · subst hk; simp [h]
    · simp [hk]
  | none =>
    unfold hmGet at h ⊢
    rw [List.find?_append]
    by_cases hk : k = k'
    · subst hk
      rw [Option.map_eq_none_iff] at h
      simp [h]
    · simp only [hk, if_false]
      cases hf : List.find? (fun p => p.1 == k') m <;> simp [hk]

/-- After `extend`, a column's log is its old log followed by this write's changes to it, in order. -/
theorem extend_log (m : ColumnWrites) (cs : List (IndexKey × IndexWrite)) (k : IndexKey) :
    (hmGet (cs.foldl (fun m kw => pushWrite m kw.1 kw.2) m) k).getD [] =
      (hmGet m k).getD [] ++ (cs.filter (fun kw => kw.1 = k)).map (·.2) := by
  induction cs generalizing m with
  | nil => simp
  | cons kw cs ih =>
    simp only [List.foldl_cons, ih, hmGet_pushWrite, List.filter_cons]
    by_cases hk : kw.1 = k <;> simp [hk]

/-- `extend` never loses a column: one that had a log keeps it as a prefix. -/
theorem extend_isSome (m : ColumnWrites) (cs : List (IndexKey × IndexWrite)) (k : IndexKey) :
    (hmGet m k).isSome → (hmGet (cs.foldl (fun m kw => pushWrite m kw.1 kw.2) m) k).isSome := by
  induction cs generalizing m with
  | nil => simp
  | cons kw cs ih =>
    intro h; simp only [List.foldl_cons]; apply ih
    rw [hmGet_pushWrite]; split <;> simp_all

theorem clear_isEmpty (p : PendingIndexWrites) : p.clear.isEmpty = true := rfl
theorem new_isEmpty (e : EntityType) : (IndexChanges.new e).isEmpty = true := rfl
theorem drain_spec (p : PendingIndexWrites) :
    p.drain .node = (p.nodes, { p with nodes := [] }) ∧ p.drain .relationship = (p.edges, { p with edges := [] }) :=
  ⟨rfl, rfl⟩

/-- `push` stages `(enc(value), id, add)` iff the column exists and indexes the value. -/
theorem push_spec (c : IndexChanges) (ix : FalkorDbIndex) (l a : String) (v : Value) (id : Nat) (add : Bool) :
    (c.push ix l a v id add).changes =
      if ix.hasColumn c.entity l a then
        (match encodeNumeric v with
         | some k => c.changes ++ [((l, a), ⟨k, id, add⟩)]
         | none => c.changes)
      else c.changes := by
  unfold IndexChanges.push
  rw [encodeValue_spec]
  by_cases h : ix.hasColumn c.entity l a <;> simp [h]
  cases encodeNumeric v <;> rfl

theorem add_remove_spec (c : IndexChanges) (ix : FalkorDbIndex) (l a : String) (v : Value) (id : Nat) :
    c.add ix l a v id = c.push ix l a v id true ∧ c.remove ix l a v id = c.push ix l a v id false := ⟨rfl, rfl⟩

/-- The scenario of the doc comment (`pending_writes.rs:100-105`): `SET n.v = 5` then `DELETE n`
    stages `add (5, n)` then `remove (5, n)`; the collapsed batch leaves the node unindexed. -/
theorem set_then_delete :
    (applyColumn (.numeric ⟨[]⟩) [⟨5, 7, true⟩, ⟨5, 7, false⟩] [] []).1.ix.tree = [] := by
  rw [applyColumn_eq_sequential _ List.Pairwise.nil]
  simp [seqTree, tInsert, tRemove, sins, IndexWrite.tup]

end NI
