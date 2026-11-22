import FalkorNativeIndex.Index
set_option linter.unusedSimpArgs false

/-!
# Staged writes (`graph/src/index/falkordb/pending_writes.rs`) and `FalkorDbIndex::apply`
# (`falkordb_index.rs`) — #3307's "apply once per unit of work"

Proven (`writerRecord_eq_sequential`): collapsing a column's staged log with
`last_write_wins` and applying one `remove_batch` + one `insert_batch` gives **the same tree as
applying every staged change one by one** (`insert` / `remove`, in staging order), for any log
(adds and removes of one tuple interleaved arbitrarily). `extend_log` shows `extend` keeps each
column's changes in staging order, so a unit of work's whole sequence is what gets collapsed.

`sort_by_key` is a *stable* sort: modelled as `List.mergeSort` (stable in core:
`List.sublist_mergeSort`). The `(key, id)` tuple order is the order of `enc key id` (cow_btree
`enc_lt_iff`, ids `< 2^64`).
-/

namespace NI

/-- `2^128`: above every `enc key id` of `u64`s. Kept opaque so arithmetic on it stays linear. -/
@[irreducible] def M : Nat := W * W

theorem W_pos : 0 < W := Nat.succ_pos _

theorem M_pos : 0 < M := by unfold M; exact Nat.mul_pos W_pos W_pos

theorem enc_eq (k d : Nat) : enc k d = k * W + d := rfl

/-- `IndexWrite` (`pending_writes.rs`): one staged change of a `(key, id)` tuple in the tree of
    `kind`. -/
structure IndexWrite where
  key : Nat
  id : Nat
  kind : KeyKind
  add : Bool
deriving DecidableEq, Repr

/-- The tree entry a write names: `enc key id`. -/
@[irreducible] def IndexWrite.ent (w : IndexWrite) : Nat := enc w.key w.id

/-- Position of the write's tree in the sort (`KeyKind`'s derived `Ord`: `Scalar < Array`). -/
def KeyKind.rank : KeyKind → Nat
  | .scalar => 0
  | .array => 1

def IndexWrite.rank (w : IndexWrite) : Nat := w.kind.rank

/-- The sort key of `last_write_wins`' `tuple = |w| (w.kind, w.key, w.id)` (`pending_writes.rs`),
    as one number: the kind's rank above the tuple's `enc key id` (cow_btree's `enc_lt_iff`, ids
    `< 2^64`). For `key, id < 2^64` its order is the lexicographic order of `(kind, key, id)` and two
    writes share it iff they name one tuple of one tree (`tup_eq_iff`). -/
def IndexWrite.tup (w : IndexWrite) : Nat := w.rank * (M) + w.ent

theorem enc_lt_WW (k d : Nat) (hk : k < W) (hd : d < W) : enc k d < M := by
  have : (k + 1) * W ≤ W * W := Nat.mul_le_mul_right _ hk
  have h2 : (k + 1) * W = k * W + W := by rw [Nat.succ_mul]
  rw [enc_eq]
  unfold M
  omega

theorem ent_eq' (w : IndexWrite) : w.ent = enc w.key w.id := by unfold IndexWrite.ent; rfl

theorem ent_lt_M (w : IndexWrite) (hw : w.key < W ∧ w.id < W) : w.ent < M := by
  rw [ent_eq']; exact enc_lt_WW _ _ hw.1 hw.2

/-- A well-formed write has the sort key of tuple `e` of tree `kd` iff it names that tuple of that tree. -/
theorem tup_cond (w : IndexWrite) (hw : w.key < W ∧ w.id < W) (kd : KeyKind) (e : Nat) (he : e < M) :
    w.tup = kd.rank * M + e ↔ w.kind = kd ∧ w.ent = e := by
  have h1 : w.ent < M := ent_lt_M w hw
  unfold IndexWrite.tup IndexWrite.rank
  cases hk : w.kind <;> cases kd <;> simp [KeyKind.rank] <;> omega

/-- `IndexChanges` (`pending_writes.rs`). -/
structure IndexChanges where
  entity : EntityType
  changes : List (IndexKey × IndexWrite)

namespace IndexChanges
/-- `new`. -/
def new (e : EntityType) : IndexChanges := ⟨e, []⟩
/-- `is_empty`. -/
def isEmpty (c : IndexChanges) : Bool := c.changes.isEmpty
/-- `push`: `for_each_stored_key`, staging one write per key the column stores `v` under
    (none when there is no such column or it does not index `v`; a list stages one per numeric element). -/
def push (c : IndexChanges) (ix : FalkorDbIndex) (label attr : String) (v : Value) (id : Nat) (add : Bool) :
    IndexChanges :=
  { c with changes := c.changes ++
      ((ix.storedKeys c.entity label attr v).map (fun kk => ((label, attr), ⟨kk.2, id, kk.1, add⟩))) }
/-- `remove`. -/
def remove (c : IndexChanges) (ix : FalkorDbIndex) (label attr : String) (v : Value) (id : Nat) : IndexChanges :=
  c.push ix label attr v id false
/-- `add`. -/
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

/-- `PendingIndexWrites` (`pending_writes.rs`). -/
structure PendingIndexWrites where
  nodes : ColumnWrites
  edges : ColumnWrites

namespace PendingIndexWrites
def empty : PendingIndexWrites := ⟨[], []⟩
/-- `is_empty`. -/
def isEmpty (p : PendingIndexWrites) : Bool := p.nodes.isEmpty && p.edges.isEmpty
/-- `drain`: the entity's columns, leaving that side empty. -/
def drain (p : PendingIndexWrites) : EntityType → ColumnWrites × PendingIndexWrites
  | .node => (p.nodes, { p with nodes := [] })
  | .relationship => (p.edges, { p with edges := [] })
/-- `clear`. -/
def clear (_p : PendingIndexWrites) : PendingIndexWrites := ⟨[], []⟩
/-- `extend`. -/
def extend (p : PendingIndexWrites) (c : IndexChanges) : PendingIndexWrites :=
  match c.entity with
  | .node => { p with nodes := c.changes.foldl (fun m kw => pushWrite m kw.1 kw.2) p.nodes }
  | .relationship => { p with edges := c.changes.foldl (fun m kw => pushWrite m kw.1 kw.2) p.edges }
end PendingIndexWrites

/-! ## `last_write_wins` (`pending_writes.rs`) -/

/-- `log.sort_by_key(tuple)`, `tuple = |w| (w.kind, w.key, w.id)` — stable. -/
def sortLog (log : List IndexWrite) : List IndexWrite := log.mergeSort (fun a b => decide (a.tup ≤ b.tup))

/-- The loop body: keep `w` unless `log[i + 1]` is the same tuple. -/
def lwwGo : List IndexWrite → List IndexWrite
  | [] => []
  | [w] => [w]
  | w :: w' :: r => if w'.tup = w.tup then lwwGo (w' :: r) else w :: lwwGo (w' :: r)

/-- The tuples of one side of `last_write_wins`' output: each kept write of that side pushes its
    `(key, id)` onto `side[w.kind]` (`pending_writes.rs`). -/
def sideTuples (kept : List IndexWrite) (pick : IndexWrite → Bool) : EncodedTuples :=
  ⟨(kept.filter (fun w => pick w && decide (w.kind = .scalar))).map IndexWrite.ent,
   (kept.filter (fun w => pick w && decide (w.kind = .array))).map IndexWrite.ent⟩

theorem lwwGo_sublist : ∀ (s : List IndexWrite), (lwwGo s).Sublist s
  | [] => by simp [lwwGo]
  | [w] => by simp [lwwGo]
  | w :: w' :: r => by
    unfold lwwGo
    split
    · exact (lwwGo_sublist (w' :: r)).cons _
    · exact (lwwGo_sublist (w' :: r)).cons₂ _

/-- `last_write_wins(log, removes, adds)`: both outputs cleared first, then filled.
    Returns `(sorted log, removes, adds)`. -/
def lastWriteWins (log : List IndexWrite) (_removes _adds : EncodedTuples) :
    List IndexWrite × EncodedTuples × EncodedTuples :=
  let s := sortLog log
  let kept := lwwGo s
  (s, sideTuples kept (fun w => !w.add), sideTuples kept (fun w => w.add))

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

theorem lastOp_singleton (w : IndexWrite) : lastOp [w] w.tup = some w.add := by
  simp [lastOp]

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
    · simp only [heq, ite_true, ih]
      by_cases he : w.tup = e
      · have : lastOp (w' :: r) e ≠ none := by rw [← he, ← heq]; exact lastOp_head w' r
        cases h : lastOp (w' :: r) e
        · exact absurd h this
        · simp
      · cases lastOp (w' :: r) e <;> simp [he]
    · simp only [heq, ite_false, List.mem_cons, exists_eq_or_imp]
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
        simp only [he, ite_true, true_and, Option.some.injEq]
        constructor
        · rintro (h | ⟨x, hx, h1, h2⟩)
          · exact h
          · exact absurd (ih.1 ⟨x, hx, h1, h2⟩) (by simp)
        · intro h; exact Or.inl h
      · simp only [he, false_and, false_or, ih]
        cases lastOp (w' :: r) e <;> simp [he]

/-! ### One tree's view of a log

`last_write_wins` sorts the whole log by `(kind, key, id)` and collapses runs of one tuple, so each tree's
share is the collapse of that tree's own writes: it is stated per kind, on the ops (`entry`, `add?`) of
the writes of that kind. -/

/-- The `(entry, add?)` ops a log stages in the tree of `kd`, in staging order. -/
def kindOps (kd : KeyKind) (log : List IndexWrite) : List (Nat × Bool) :=
  (log.filter (fun w => decide (w.kind = kd))).map (fun w => (w.ent, w.add))

/-- The last op on entry `e` in a list of ops. -/
def lastOpOps (ops : List (Nat × Bool)) (e : Nat) : Option Bool :=
  ((ops.filter (fun o => decide (o.1 = e))).getLast?).map (·.2)

/-- The last staged change to the tuple `e` of the tree of `kd`. -/
def lastOpK (log : List IndexWrite) (kd : KeyKind) (e : Nat) : Option Bool :=
  lastOpOps (kindOps kd log) e

/-- Applying ops one by one, `CowBTree::insert` / `remove` each. -/
def seqOps (t : List Nat) (ops : List (Nat × Bool)) : List Nat :=
  ops.foldl (fun t o => if o.2 then tInsert t o.1 else tRemove t o.1) t

theorem kindOps_filter (log : List IndexWrite) (hwf : ∀ w ∈ log, w.key < W ∧ w.id < W) (kd : KeyKind)
    (e : Nat) (he : e < M) :
    ((log.filter (fun w => decide (w.kind = kd))).map (fun w => (w.ent, w.add))).filter
        (fun o => decide (o.1 = e)) =
      (log.filter (fun w => decide (w.tup = kd.rank * M + e))).map (fun w => (w.ent, w.add)) := by
  induction log with
  | nil => simp
  | cons w ws ih =>
    have hw := hwf w List.mem_cons_self
    have ih' := ih (fun x hx => hwf x (List.mem_cons_of_mem _ hx))
    have hc := tup_cond w hw kd e he
    generalize kd.rank * M + e = t at hc ih' ⊢
    by_cases h : w.kind = kd ∧ w.ent = e
    · have h1 : w.tup = t := hc.2 h
      simp [List.filter_cons, h.1, h.2, h1, ih']
    · have h1 : ¬ w.tup = t := fun x => h (hc.1 x)
      by_cases hk : w.kind = kd
      · have h2 : ¬ w.ent = e := fun x => h ⟨hk, x⟩
        simp [List.filter_cons, hk, h2, h1, ih']
      · simp [List.filter_cons, hk, h1, ih']

theorem lastOp_kind (log : List IndexWrite) (hwf : ∀ w ∈ log, w.key < W ∧ w.id < W) (kd : KeyKind)
    (e : Nat) (he : e < M) : lastOp log (kd.rank * M + e) = lastOpK log kd e := by
  unfold lastOp lastOpK lastOpOps kindOps
  rw [kindOps_filter log hwf kd e he]
  simp [List.getLast?_map, Option.map_map, Function.comp_def]

/-- **`last_write_wins` keeps exactly each tuple's last staged change**, in each tree: a tuple is in a
    side's share of a tree iff the last change to it, in that tree, is of that side. (Writes carry
    `u64` keys and ids.) -/
theorem lastWriteWins_spec (log : List IndexWrite) (hwf : ∀ w ∈ log, w.key < W ∧ w.id < W)
    (rm ad : EncodedTuples) (kd : KeyKind) (e : Nat) :
    (e ∈ (lastWriteWins log rm ad).2.2.get kd ↔ lastOpK log kd e = some true) ∧
    (e ∈ (lastWriteWins log rm ad).2.1.get kd ↔ lastOpK log kd e = some false) := by
  have hs : SortedTup (sortLog log) := by
    have := List.pairwise_mergeSort (le := fun a b : IndexWrite => decide (a.tup ≤ b.tup))
      (by intro a b c h1 h2; simp at *; omega) (by intro a b; simp; omega) log
    exact this.imp (by intro a b h; simpa using h)
  have hwf' : ∀ w ∈ lwwGo (sortLog log), w.key < W ∧ w.id < W := by
    intro w hw
    have h1 : w ∈ sortLog log := (lwwGo_sublist _).subset hw
    exact hwf w ((List.mem_mergeSort).1 h1)
  have key : ∀ (b : Bool), (∃ w ∈ lwwGo (sortLog log), w.add = b ∧ w.kind = kd ∧ w.ent = e) ↔
      lastOpK log kd e = some b := by
    intro b
    by_cases he : e < M
    · rw [← lastOp_kind log hwf kd e he]
      have hlast : lastOp (sortLog log) (kd.rank * M + e) = lastOp log (kd.rank * M + e) := by
        unfold lastOp; rw [sortLog_filter]
      rw [← hlast, ← lwwGo_spec _ hs _ b]
      constructor
      · rintro ⟨w, hw, h1, h2, h3⟩
        exact ⟨w, hw, (tup_cond w (hwf' w hw) kd e he).2 ⟨h2, h3⟩, h1⟩
      · rintro ⟨w, hw, h1, h2⟩
        obtain ⟨hk, he'⟩ := (tup_cond w (hwf' w hw) kd e he).1 h1
        exact ⟨w, hw, h2, hk, he'⟩
    · constructor
      · rintro ⟨w, hw, _, _, h3⟩
        exfalso; apply he
        rw [← h3]; exact ent_lt_M w (hwf' w hw)
      · intro h
        exfalso
        unfold lastOpK lastOpOps kindOps at h
        cases hl : ((log.filter (fun w => decide (w.kind = kd))).map (fun w => (w.ent, w.add))).filter
            (fun o => decide (o.1 = e)) with
        | nil => simp [hl] at h
        | cons o os =>
          have hmem : o ∈ ((log.filter (fun w => decide (w.kind = kd))).map (fun w => (w.ent, w.add))).filter
              (fun o => decide (o.1 = e)) := by rw [hl]; exact List.mem_cons_self
          simp only [List.mem_filter, List.mem_map, decide_eq_true_eq] at hmem
          obtain ⟨⟨w, hw, rfl⟩, ho⟩ := hmem
          exact he (by rw [← ho]; exact ent_lt_M w (hwf w hw.1))
  have mem_side : ∀ (pick : IndexWrite → Bool) (e : Nat),
      e ∈ (sideTuples (lwwGo (sortLog log)) pick).get kd ↔
        ∃ w ∈ lwwGo (sortLog log), pick w = true ∧ w.kind = kd ∧ w.ent = e := by
    intro pick e
    cases kd <;> simp only [sideTuples, EncodedTuples.get, List.mem_map, List.mem_filter, Bool.and_eq_true,
      decide_eq_true_eq] <;>
      exact ⟨fun ⟨w, ⟨h1, h2, h3⟩, h4⟩ => ⟨w, h1, h2, h3, h4⟩, fun ⟨w, h1, h2, h3, h4⟩ => ⟨w, ⟨h1, h2, h3⟩, h4⟩⟩
  constructor
  · show e ∈ (sideTuples (lwwGo (sortLog log)) (fun w => w.add)).get kd ↔ _
    rw [mem_side, ← key true]
  · show e ∈ (sideTuples (lwwGo (sortLog log)) (fun w => !w.add)).get kd ↔ _
    rw [mem_side, ← key false]
    constructor
    · rintro ⟨w, hw, h1, h2, h3⟩; exact ⟨w, hw, by simpa using h1, h2, h3⟩
    · rintro ⟨w, hw, h1, h2, h3⟩; exact ⟨w, hw, by simpa using h1, h2, h3⟩

/-! ## One column: collapsed batch = one by one -/

theorem seqOps_sorted : ∀ (ops : List (Nat × Bool)) (t : List Nat), Sorted t → Sorted (seqOps t ops)
  | [], t, h => h
  | o :: os, t, h => by
    simp only [seqOps, List.foldl_cons]
    apply seqOps_sorted os
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
  · rw [ite_eq_right hw]
    cases A
    · simp only [Bool.false_eq_true, ite_false, tRemove, mem_erase_sorted _ _ h]
      exact ⟨fun h1 => h1.1, fun h1 => ⟨h1, Ne.symm hw⟩⟩
    · simp only [ite_true, tInsert, mem_sins]
      exact ⟨fun h1 => h1.elim id (fun h2 => absurd h2.symm hw), Or.inl⟩

theorem lastOpOps_cons (o : Nat × Bool) (l : List (Nat × Bool)) (e : Nat) :
    lastOpOps (o :: l) e = match lastOpOps l e with
      | some b => some b
      | none => if o.1 = e then some o.2 else none := by
  unfold lastOpOps
  by_cases h : o.1 = e
  · rw [List.filter_cons_of_pos (by simp [h]), List.getLast?_cons]
    cases hl : (List.filter (fun w => decide (w.1 = e)) l).getLast? <;> simp [h]
  · rw [List.filter_cons_of_neg (by simp [h])]
    cases (List.filter (fun w => decide (w.1 = e)) l).getLast? <;> simp [h]

theorem mem_seqOps (ops : List (Nat × Bool)) : ∀ (t : List Nat), Sorted t → ∀ e,
    e ∈ seqOps t ops ↔ (match lastOpOps ops e with | some b => b = true | none => e ∈ t) := by
  induction ops with
  | nil => intro t _ e; simp [seqOps, lastOpOps]
  | cons o os ih =>
    intro t h e
    simp only [seqOps, List.foldl_cons]
    have hs : Sorted (if o.2 then tInsert t o.1 else tRemove t o.1) := by
      split
      · exact sorted_sins _ _ h
      · exact sorted_erase _ _ h
    have ih' := ih _ hs e
    simp only [seqOps] at ih'
    rw [ih', lastOpOps_cons]
    cases lastOpOps os e with
    | some b => rfl
    | none =>
      have key := step_mem t h o.1 e o.2
      by_cases hw : o.1 = e
      · rw [ite_eq_left hw] at key ⊢; exact key
      · rw [ite_eq_right hw] at key ⊢; exact key

/-- The tree `ix` becomes when the log's changes are applied one by one, each to the tree of its kind. -/
def seqIndex (ix : NumericIndex) (log : List IndexWrite) : NumericIndex :=
  ⟨seqOps ix.tree (kindOps .scalar log), seqOps ix.arrayTree (kindOps .array log)⟩

/-- The `log.retain(..)` of `writer_record` (`falkordb_index.rs`), `Building` only: keep every
    removal, and an addition only for an entity `place` puts below the cursor (`precedes next`);
    `place = None` (an edge this unit deleted) drops it (`is_some_and`). A `Ready` column keeps the
    whole log. `retain` is stable, so the order of the survivors is kept. -/
def stateLog (state : BuildState) (place : Nat → Option BuildCursor) (log : List IndexWrite) :
    List IndexWrite :=
  match state with
  | .building next => log.filter (fun w => !w.add || (place w.id).any (fun pl => pl.precedes next))
  | .ready => log

theorem stateLog_ready (place : Nat → Option BuildCursor) (log : List IndexWrite) :
    stateLog .ready place log = log := rfl

/-- `IndexColumn::writer_record` (`falkordb_index.rs`): drop what the state excludes, collapse
    to each tuple's last change (`last_write_wins`), **remove** what ended removed then **add** what
    ended added, one batched tree op each (per tree). Returns the column and the two (sorted) buffers. -/
def IndexColumn.writerRecord (c : IndexColumn) (state : BuildState) (log : List IndexWrite)
    (place : Nat → Option BuildCursor) (removes adds : EncodedTuples) :
    IndexColumn × EncodedTuples × EncodedTuples :=
  let log := stateLog state place log
  let (_, rm, ad) := lastWriteWins log removes adds
  let (c, removes) := c.removeEncoded rm
  let (c, adds) := c.addEncoded ad
  (c, removes, adds)

/-- `IndexEntry::writer_record` (`versioned.rs`): the column's `writer_record` at its state; the
    state itself is unchanged. -/
def IndexEntry.writerRecord (en : IndexEntry) (log : List IndexWrite) (place : Nat → Option BuildCursor)
    (sc : EncodedTuples × EncodedTuples) : IndexEntry × (EncodedTuples × EncodedTuples) :=
  let r := en.index.writerRecord en.state log place sc.1 sc.2
  (⟨r.1, en.state⟩, (r.2.1, r.2.2))

/-- Every write's key and id are `u64` (`IndexWrite`'s fields). -/
def LogWF (log : List IndexWrite) : Prop := ∀ w ∈ log, w.key < W ∧ w.id < W

theorem stateLog_sublist' (st : BuildState) (place : Nat → Option BuildCursor) (log : List IndexWrite) :
    (stateLog st place log).Sublist log := by
  cases st <;> simp [stateLog, List.filter_sublist]

theorem stateLog_wf (st : BuildState) (place : Nat → Option BuildCursor) (log : List IndexWrite)
    (h : LogWF log) : LogWF (stateLog st place log) :=
  fun w hw => h w ((stateLog_sublist' st place log).subset hw)

theorem removeTree_sorted (t rm : List Nat) (hs : Sorted t) : Sorted (removeTree t rm) := by
  unfold removeTree; split
  · exact hs
  · exact sorted_removeBatch _ _ hs

/-- One tree of a column recording a collapsed log: remove what ended removed, then add what ended added,
    is applying the ops one by one. -/
theorem tree_step (t : List Nat) (hs : Sorted t) (rm ad : List Nat) (ops : List (Nat × Bool))
    (hrm : ∀ e, e ∈ rm ↔ lastOpOps ops e = some false) (had : ∀ e, e ∈ ad ↔ lastOpOps ops e = some true) :
    addTree (removeTree t (sortDedup rm)) (sortDedup ad) = seqOps t ops := by
  apply sorted_ext
  · exact addTree_sorted _ _ (removeTree_sorted _ _ hs) (sortDedup_spec _).1
  · exact seqOps_sorted _ _ hs
  · intro e
    rw [addTree_spec, removeTree_spec, mem_seqOps ops t hs, (sortDedup_spec _).2, (sortDedup_spec _).2,
      hrm, had]
    cases lastOpOps ops e with
    | none => simp
    | some b => cases b <;> simp

/-- **A column records a unit's changes as the staged changes that its state admits, applied one by one**
    (#3307 collapsed to one batch each, #2351 for the `Building` filter), in each of its two trees
    (#2453). -/
theorem writerRecord_eq_sequential (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (st : BuildState) (log : List IndexWrite) (hwf : LogWF log) (place : Nat → Option BuildCursor)
    (rm ad : EncodedTuples) :
    (IndexColumn.writerRecord (.numeric ix) st log place rm ad).1.ix = seqIndex ix (stateLog st place log) := by
  have hwf' := stateLog_wf st place log hwf
  have lwS := lastWriteWins_spec (stateLog st place log) hwf' rm ad .scalar
  have lwA := lastWriteWins_spec (stateLog st place log) hwf' rm ad .array
  simp only [IndexColumn.writerRecord, IndexColumn.removeEncoded, IndexColumn.addEncoded, IndexColumn.ix,
    seqIndex, NumericIndex.removeEncoded, NumericIndex.addEncoded, NumericIndex.addTrees,
    NumericIndex.removeTrees, EncodedTuples.sortDedup]
  have e1 := tree_step ix.tree hs (lastWriteWins (stateLog st place log) rm ad).2.1.scalar
    (lastWriteWins (stateLog st place log) rm ad).2.2.scalar (kindOps .scalar (stateLog st place log))
    (fun e => (lwS e).2) (fun e => (lwS e).1)
  have e2 := tree_step ix.arrayTree ha (lastWriteWins (stateLog st place log) rm ad).2.1.array
    (lastWriteWins (stateLog st place log) rm ad).2.2.array (kindOps .array (stateLog st place log))
    (fun e => (lwA e).2) (fun e => (lwA e).1)
  rw [e1, e2]

/-- A `Ready` column takes the whole log (#3307's claim). -/
theorem writerRecord_ready (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (log : List IndexWrite) (hwf : LogWF log)
    (place : Nat → Option BuildCursor) (rm ad : EncodedTuples) :
    (IndexColumn.writerRecord (.numeric ix) .ready log place rm ad).1.ix = seqIndex ix log :=
  writerRecord_eq_sequential ix hs ha .ready log hwf place rm ad

/-- **`Building` filter**: every removal stays (`retain` keeps every non-addition) ... -/
theorem stateLog_removes (next : BuildCursor) (place : Nat → Option BuildCursor) (log : List IndexWrite)
    (w : IndexWrite) (hw : w ∈ log) (hr : w.add = false) : w ∈ stateLog (.building next) place log := by
  simp [stateLog, List.mem_filter, hw, hr]

/-- ... and an addition stays iff its entity is placed below the cursor. -/
theorem stateLog_adds (next : BuildCursor) (place : Nat → Option BuildCursor) (log : List IndexWrite)
    (w : IndexWrite) (hw : w ∈ log) (ha : w.add = true) :
    w ∈ stateLog (.building next) place log ↔ ∃ pl, place w.id = some pl ∧ pl.precedes next = true := by
  simp only [stateLog, List.mem_filter, hw, ha, true_and, Bool.not_true, Bool.false_or]
  cases place w.id <;> simp

/-- The filter only ever drops additions. -/
theorem stateLog_sublist (st : BuildState) (place : Nat → Option BuildCursor) (log : List IndexWrite) :
    (stateLog st place log).Sublist log := by
  cases st <;> simp [stateLog, List.filter_sublist]

theorem lastOpOps_single (x : Nat) (b : Bool) : lastOpOps [(x, b)] x = some b := by
  unfold lastOpOps
  rw [List.filter_cons_of_pos (by simp)]
  simp

theorem lastOpOps_two (x : Nat) (b b' : Bool) : lastOpOps [(x, b), (x, b')] x = some b' := by
  unfold lastOpOps
  rw [List.filter_cons_of_pos (by simp), List.filter_cons_of_pos (by simp)]
  simp

theorem kindOps_single (key id : Nat) (b : Bool) :
    kindOps .scalar [⟨key, id, .scalar, b⟩] = [(enc key id, b)] := by
  simp [kindOps, ent_eq']

/-- **A dropped addition does not cancel an earlier removal** (`versioned.rs`): within
    one unit, tuple `t` is removed and then added back for an entity at or above the cursor (as a
    reused edge id's new edge can be); the filter drops the addition *before* the collapse, so the
    removal still applies and the column ends without `t`. -/
theorem building_reuse_removed (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (next : BuildCursor) (key id : Nat) (hk : key < W) (hid : id < W)
    (place : Nat → Option BuildCursor) (hp : ∀ i, place i = none)
    (rm ad : EncodedTuples) (hin : enc key id ∈ ix.tree) :
    enc key id ∉ (IndexColumn.writerRecord (.numeric ix) (.building next)
      [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] place rm ad).1.ix.tree := by
  have hlog : stateLog (.building next) place [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] =
      [⟨key, id, .scalar, false⟩] := by
    simp [stateLog, hp, List.filter_cons]
  have hwf : LogWF [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] := by
    intro w hw; simp at hw; rcases hw with rfl | rfl <;> exact ⟨hk, hid⟩
  rw [writerRecord_eq_sequential ix hs ha _ _ hwf, hlog]
  simp only [seqIndex]
  rw [mem_seqOps _ _ hs, kindOps_single, lastOpOps_single]
  simp

/-- Without the filter order (collapse first) the same log would have *kept* `t`: the last change is
    the addition. This is why the filter runs first. -/
theorem collapse_first_would_keep (key id : Nat) :
    lastOpK [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] .scalar (enc key id) = some true := by
  have : kindOps .scalar [(⟨key, id, .scalar, false⟩ : IndexWrite), ⟨key, id, .scalar, true⟩] =
      [(enc key id, false), (enc key id, true)] := by simp [kindOps, ent_eq']
  unfold lastOpK; rw [this, lastOpOps_two]

/-- The same unit on a `Ready` column ends with `t` present. -/
theorem ready_reuse_kept (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (key id : Nat) (hk : key < W) (hid : id < W) (place : Nat → Option BuildCursor)
    (rm ad : EncodedTuples) :
    enc key id ∈ (IndexColumn.writerRecord (.numeric ix) .ready
      [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] place rm ad).1.ix.tree := by
  have hwf : LogWF [⟨key, id, .scalar, false⟩, ⟨key, id, .scalar, true⟩] := by
    intro w hw; simp at hw; rcases hw with rfl | rfl <;> exact ⟨hk, hid⟩
  rw [writerRecord_ready ix hs ha _ hwf]
  simp only [seqIndex]
  have : kindOps .scalar [(⟨key, id, .scalar, false⟩ : IndexWrite), ⟨key, id, .scalar, true⟩] =
      [(enc key id, false), (enc key id, true)] := by simp [kindOps, ent_eq']
  rw [mem_seqOps _ _ hs, this, lastOpOps_two]

/-! ## `FalkorDbIndex::apply_writes` / `apply` -/

/-- One iteration of `apply_writes`' loop (`falkordb_index.rs`): the column's entry records the
    log at its state; a column dropped since staging is skipped. `sc` is the unit's one pair of buffers. -/
def applyStep (place : EntityType → Nat → Option BuildCursor) (e : EntityType)
    (st : FalkorDbIndex × (EncodedTuples × EncodedTuples)) (kl : IndexKey × List IndexWrite) :
    FalkorDbIndex × (EncodedTuples × EncodedTuples) :=
  match hmGet (st.1.columns e) kl.1 with
  | none => st                                                    -- column dropped: skipped
  | some entry =>
    let r := entry.writerRecord kl.2 (place e) st.2
    (st.1.setColumns e (hmInsert (st.1.columns e) kl.1 r.1), r.2)

/-- `apply_writes` (`falkordb_index.rs`): one pair of buffers for the whole unit of work. -/
def FalkorDbIndex.applyWrites (x : FalkorDbIndex) (place : EntityType → Nat → Option BuildCursor)
    (e : EntityType) (writes : ColumnWrites) : FalkorDbIndex :=
  (writes.foldl (applyStep place e) (x, (EncodedTuples.empty, EncodedTuples.empty))).1

/-- `apply` (`falkordb_index.rs`): nodes then relationships; leaves `writes` empty. -/
def FalkorDbIndex.apply (x : FalkorDbIndex) (w : PendingIndexWrites)
    (place : EntityType → Nat → Option BuildCursor) : FalkorDbIndex × PendingIndexWrites :=
  if w.isEmpty then (x, w)
  else
    let d1 := w.drain .node
    let x := x.applyWrites place .node d1.1
    let d2 := d1.2.drain .relationship
    (x.applyWrites place .relationship d2.1, d2.2)

theorem apply_empties (x : FalkorDbIndex) (w : PendingIndexWrites) (place) : (x.apply w place).2.isEmpty = true := by
  unfold FalkorDbIndex.apply
  split
  · assumption
  · rfl

def ColsSorted (x : FalkorDbIndex) (e : EntityType) : Prop :=
  ∀ l a ix, x.numeric e l a = some ix → Sorted ix.tree ∧ Sorted ix.arrayTree

theorem entry_setColumns_insert (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (en : IndexEntry) (l a : String) :
    (x.setColumns e (hmInsert (x.columns e) k en)).entry e l a =
      if k = (l, a) then some en else x.entry e l a := by
  simp only [FalkorDbIndex.entry, columns_setColumns, ite_true, hmGet_insert]

theorem numeric_setColumns_insert (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (en : IndexEntry) (l a : String) :
    (x.setColumns e (hmInsert (x.columns e) k en)).numeric e l a =
      if k = (l, a) then some en.index.ix else x.numeric e l a := by
  simp only [FalkorDbIndex.numeric, entry_setColumns_insert]
  split <;> rfl

theorem seqIndex_sorted (ix : NumericIndex) (hs : Sorted ix.tree) (ha : Sorted ix.arrayTree)
    (log : List IndexWrite) : Sorted (seqIndex ix log).tree ∧ Sorted (seqIndex ix log).arrayTree :=
  ⟨seqOps_sorted _ _ hs, seqOps_sorted _ _ ha⟩

theorem applyFold_numeric (place : EntityType → Nat → Option BuildCursor) (e : EntityType) (writes : ColumnWrites) :
    ∀ (x : FalkorDbIndex) (sc : EncodedTuples × EncodedTuples), (writes.map (·.1)).Nodup → ColsSorted x e →
    (∀ kl ∈ writes, LogWF kl.2) →
    ∀ (l at' : String),
    ((writes.foldl (applyStep place e) (x, sc)).1).numeric e l at' =
      match hmGet writes (l, at') with
      | none => x.numeric e l at'
      | some log => (x.entry e l at').map
          (fun en => seqIndex en.index.ix (stateLog en.state (place e) log)) := by
  induction writes with
  | nil => intro x sc _ _ _ l at'; simp [hmGet, FalkorDbIndex.numeric]
  | cons kl ws ih =>
    intro x sc hnd hs hwf l at'
    obtain ⟨k, log⟩ := kl
    simp only [List.map_cons, List.nodup_cons] at hnd
    have notin : hmGet ws k = none := by
      unfold hmGet; rw [Option.map_eq_none_iff, List.find?_eq_none]
      intro p hp; simp only [beq_iff_eq]; intro h; apply hnd.1; rw [← h]; exact List.mem_map_of_mem hp
    have getcons : hmGet ((k, log) :: ws) (l, at') = if k = (l, at') then some log else hmGet ws (l, at') := by
      unfold hmGet; by_cases hk : k = (l, at') <;> simp [hk]
    have hwf' : ∀ kl ∈ ws, LogWF kl.2 := fun kl hkl => hwf kl (List.mem_cons_of_mem _ hkl)
    have hlog : LogWF log := hwf (k, log) List.mem_cons_self
    simp only [List.foldl_cons, applyStep]
    rw [getcons]
    cases hc : hmGet (x.columns e) k with
    | none =>
      simp only
      rw [ih x sc hnd.2 hs hwf' l at']
      by_cases hk : k = (l, at')
      · subst hk
        have hn : x.numeric e l at' = none := by simp [FalkorDbIndex.numeric, FalkorDbIndex.entry, hc]
        have hn' : x.entry e l at' = none := by simp [FalkorDbIndex.entry, hc]
        simp [notin, hn, hn']
      · simp [hk]
    | some entry =>
      obtain ⟨⟨ix⟩, st⟩ := entry
      have hentry : x.entry e k.1 k.2 = some ⟨.numeric ix, st⟩ := by simpa [FalkorDbIndex.entry] using hc
      have hix : x.numeric e k.1 k.2 = some ix := by simp [FalkorDbIndex.numeric, hentry, IndexColumn.ix]
      have hsix := hs _ _ ix hix
      have eqcol : (IndexEntry.writerRecord ⟨.numeric ix, st⟩ log (place e) sc).1.index.ix =
          seqIndex ix (stateLog st (place e) log) := by
        have := writerRecord_eq_sequential ix hsix.1 hsix.2 st log hlog (place e) sc.1 sc.2
        cases hcol : (IndexColumn.writerRecord (.numeric ix) st log (place e) sc.1 sc.2).1 with
        | numeric ix' =>
          rw [hcol] at this
          simp only [IndexEntry.writerRecord, hcol, IndexColumn.ix] at this ⊢
          exact this
      have hs' : ColsSorted (x.setColumns e (hmInsert (x.columns e) k
          (IndexEntry.writerRecord ⟨.numeric ix, st⟩ log (place e) sc).1)) e := by
        intro l' a' ix' h
        rw [numeric_setColumns_insert] at h
        split at h
        · simp at h; rw [← h, eqcol]; exact seqIndex_sorted _ hsix.1 hsix.2 _
        · exact hs _ _ _ h
      simp only
      rw [ih _ _ hnd.2 hs' hwf' l at', numeric_setColumns_insert]
      by_cases hk : k = (l, at')
      · subst hk
        first
          | simpa [notin, hix, hentry, IndexColumn.ix] using eqcol
          | (simp [notin, hix, hentry]; exact eqcol)
          | (simp [notin, hix, hentry, IndexColumn.ix]; simp only [IndexColumn.ix] at eqcol; exact eqcol)
      · simp [hk, entry_setColumns_insert]

/-- **`apply_writes`, per column**: a staged column's trees become `seqIndex` of its log filtered by its
    build state (the changes applied one by one, in staging order, each to the tree of its kind);
    unstaged and dropped columns are untouched. -/
theorem applyWrites_numeric (x : FalkorDbIndex) (place : EntityType → Nat → Option BuildCursor) (e : EntityType)
    (writes : ColumnWrites) (hnd : (writes.map (·.1)).Nodup) (hs : ColsSorted x e)
    (hwf : ∀ kl ∈ writes, LogWF kl.2) (l a : String) :
    (x.applyWrites place e writes).numeric e l a =
      match hmGet writes (l, a) with
      | none => x.numeric e l a
      | some log => (x.entry e l a).map
          (fun en => seqIndex en.index.ix (stateLog en.state (place e) log)) :=
  applyFold_numeric place e writes x _ hnd hs hwf l a

/-- **Writes never change a build state**: only `indexer_apply` moves the cursor or flips `Ready`. -/
theorem applyFold_state (place : EntityType → Nat → Option BuildCursor) (e : EntityType) (writes : ColumnWrites) :
    ∀ (x : FalkorDbIndex) (sc : EncodedTuples × EncodedTuples) (l a : String),
    (((writes.foldl (applyStep place e) (x, sc)).1).entry e l a).map (·.state) = (x.entry e l a).map (·.state) := by
  induction writes with
  | nil => intro x sc l a; rfl
  | cons kl ws ih =>
    intro x sc l a
    simp only [List.foldl_cons, applyStep]
    cases hc : hmGet (x.columns e) kl.1 with
    | none => exact ih x sc l a
    | some entry =>
      simp only
      rw [ih, entry_setColumns_insert]
      by_cases hk : kl.1 = (l, a)
      · have hx : x.entry e l a = some entry := by
          simp only [FalkorDbIndex.entry]; rw [← hk]; exact hc
        simp [hk, hx, IndexEntry.writerRecord]
      · simp [hk]

theorem applyWrites_state (x : FalkorDbIndex) (place : EntityType → Nat → Option BuildCursor) (e : EntityType)
    (writes : ColumnWrites) (l a : String) :
    ((x.applyWrites place e writes).entry e l a).map (·.state) = (x.entry e l a).map (·.state) :=
  applyFold_state place e writes x _ l a

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
    · simp only [hk, ite_false]
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

/-- `push` stages one write per key the column stores `v` under (`for_each_stored_key`) iff the column
    exists: `(key, id, kind, add)`, the key's tree in `kind`. -/
theorem push_spec (c : IndexChanges) (ix : FalkorDbIndex) (l a : String) (v : Value) (id : Nat) (add : Bool) :
    (c.push ix l a v id add).changes =
      if ix.hasColumn c.entity l a then
        c.changes ++ (forEachStoredKey v).map (fun kk => ((l, a), (⟨kk.2, id, kk.1, add⟩ : IndexWrite)))
      else c.changes := by
  unfold IndexChanges.push
  rw [storedKeys_spec]
  by_cases h : ix.hasColumn c.entity l a <;> simp [h]

theorem add_remove_spec (c : IndexChanges) (ix : FalkorDbIndex) (l a : String) (v : Value) (id : Nat) :
    c.add ix l a v id = c.push ix l a v id true ∧ c.remove ix l a v id = c.push ix l a v id false := ⟨rfl, rfl⟩

/-- The scenario of the doc comment (`pending_writes.rs`): `SET n.v = 5` then `DELETE n`
    stages `add (5, n)` then `remove (5, n)`; the collapsed batch leaves the node unindexed. -/
theorem set_then_delete (place : Nat → Option BuildCursor) (rm ad : EncodedTuples) :
    (IndexColumn.writerRecord (.numeric ⟨[], []⟩) .ready [⟨5, 7, .scalar, true⟩, ⟨5, 7, .scalar, false⟩]
      place rm ad).1.ix.tree = [] := by
  have h5 : 5 < W := by unfold W; decide
  have h7 : 7 < W := by unfold W; decide
  have hwf : LogWF [⟨5, 7, .scalar, true⟩, ⟨5, 7, .scalar, false⟩] := by
    intro w hw; simp at hw; rcases hw with rfl | rfl <;> exact ⟨h5, h7⟩
  have h := writerRecord_eq_sequential ⟨[], []⟩ List.Pairwise.nil List.Pairwise.nil .ready _ hwf place rm ad
  have h' := congrArg NumericIndex.tree h
  rw [h']
  simp [stateLog_ready, seqIndex, seqOps, kindOps, tInsert, tRemove, sins, ent_eq']

end NI
