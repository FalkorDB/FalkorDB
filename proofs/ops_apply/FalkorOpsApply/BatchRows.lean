import FalkorOpsApply.Correlated
import FalkorOpsApply.MergeSlot
/-!
# The correlated-batch plumbing of `runtime/batch.rs`

| here | there |
| --- | --- |
| `Col`, `Bat`, `Bat.cell` | `Column`, `Batch` (columns, `value_only` bitset, `selection`, `origin_rows`), `value_at` / `is_bound_at` |
| `Bat.originRow`, `Bat.hasOrigins` | `Batch::origin_row` (batch.rs:1305), `has_origins` |
| `Bat.gather` | `Batch::gather` (batch.rs:898) |
| `Bat.intoCompacted` | `Batch::into_compacted` (batch.rs:884) |
| `Bat.cloneSeq` | `Batch::clone_active_rows_seq_origin` (batch.rs:1170) |
| `Bat.mergeOverInput` | `Batch::merge_over_input` (batch.rs:944) |
| `setArg`, `OT` | `BatchOp::set_argument_batch` (batch.rs:1603) |
| `MergeSlot.Row.ofCells` (in `MergeSlot`) | `BatchRow::to_owned_row` (batch.rs:1439) |

Columns are a function from variable id to `Col` (everything past `ncols` is `Unbound`).
Out-of-range indices are modelled by a default (Rust panics there; callers keep them in range).
-/
namespace BatchRows

variable {V : Type} [Inhabited V]

/-- `Column`: `Unbound`, or one value per row (every typed shape is one list of values here). -/
inductive Col (V : Type) | unbound | vals (xs : List V)

/-- A batch: columns by variable id, the `value_only` bits, the physical length, the selection
vector and the `origin_rows` sidecar. -/
structure Bat (V : Type) where
  cols : Nat → Col V
  ncols : Nat
  vo : Nat → Bool
  len : Nat
  sel : Option (List Nat)
  origin : Option (List Nat)

def Col.gather : List Nat → Col V → Col V
  | _, .unbound => .unbound
  | idx, .vals xs => .vals (idx.map (fun i => xs.getD i default))

/-- `origin_row` (batch.rs:1305): `0` when no origin has been assigned. -/
def Bat.originRow (b : Bat V) (row : Nat) : Nat :=
  match b.origin with
  | none => 0
  | some o => o.getD row 0

def Bat.hasOrigins (b : Bat V) : Bool := b.origin.isSome

/-- **PROVEN** (`Batch::origin_row`, `has_origins`): reads the sidecar, `0` without one. -/
theorem originRow_spec (b : Bat V) (row : Nat) :
    (b.origin = none → b.originRow row = 0 ∧ b.hasOrigins = false) ∧
    (∀ o, b.origin = some o → b.originRow row = o.getD row 0 ∧ b.hasOrigins = true) := by
  refine ⟨fun h => ?_, fun o h => ?_⟩ <;> simp [Bat.originRow, Bat.hasOrigins, h]

/-- `Batch::gather` (batch.rs:898): the sidecar is kept only when it holds a non-zero origin. -/
def Bat.gather (b : Bat V) (idx : List Nat) : Bat V :=
  { cols := fun id => (b.cols id).gather idx
    ncols := b.ncols
    vo := b.vo
    len := idx.length
    sel := none
    origin := b.origin.bind (fun o =>
      let os := idx.map (fun i => o.getD i 0)
      if os.any (· != 0) then some os else none) }

/-- The slot of row `k` at variable `id`: `none` for an unbound column, else the value and
whether it is bound (a `value_only` column keeps its value unbound). -/
def Bat.cell (b : Bat V) (id k : Nat) : Option (V × Bool) :=
  match b.cols id with
  | .unbound => none
  | .vals xs => some (xs.getD k default, !b.vo id)

theorem gather_cell (b : Bat V) (idx : List Nat) (id j : Nat) (hj : j < idx.length) :
    (b.gather idx).cell id j = b.cell id (idx.getD j 0) := by
  unfold Bat.cell Bat.gather
  cases hc : b.cols id with
  | unbound => simp [Col.gather, hc]
  | vals xs => simp [Col.gather, hc, List.getElem?_map, hj, List.getD_eq_getElem?_getD]

theorem gather_len (b : Bat V) (idx : List Nat) : (b.gather idx).len = idx.length := rfl

/-- **PROVEN** (`Batch::gather`): row `j` of the gathered batch is row `idx[j]` of `self`, with its
origin tag; the result has no selection. -/
theorem gather_row (b : Bat V) (idx : List Nat) (j : Nat) (hj : j < idx.length) :
    (b.gather idx).sel = none ∧ (b.gather idx).len = idx.length ∧
    (∀ id, (b.gather idx).cell id j = b.cell id (idx.getD j 0)) ∧
    (b.gather idx).originRow j = b.originRow (idx.getD j 0) := by
  refine ⟨rfl, rfl, fun id => gather_cell b idx id j hj, ?_⟩
  have hmem : idx.getD j 0 ∈ idx := by
    simpa [List.getD_eq_getElem?_getD, hj] using List.getElem_mem hj
  unfold Bat.originRow Bat.gather
  cases ho : b.origin with
  | none => simp
  | some o =>
    simp only [Option.bind_some]
    by_cases hany : ((idx.map (fun i => o.getD i 0)).any (· != 0)) = true
    · simp only [hany, ↓reduceIte]
      simp [List.getD_eq_getElem?_getD, hj]
    · have hno : ∀ x ∈ idx.map (fun i => o.getD i 0), x = 0 := by
        intro x hx
        have := hany
        simp only [List.any_eq_true, bne_iff_ne, ne_eq, not_exists, not_and, Decidable.not_not] at this
        exact this x hx
      have h0 := hno (o.getD (idx.getD j 0) 0) (List.mem_map.mpr ⟨idx.getD j 0, hmem, rfl⟩)
      simp only [hany, ↓reduceIte]
      simpa using h0.symm

/-- `Batch::active_len`: the selection's length, or the physical length. -/
def Bat.activeIdx (b : Bat V) : List Nat :=
  match b.sel with
  | none => List.range b.len
  | some s => s

def Bat.activeLen (b : Bat V) : Nat := b.activeIdx.length

/-- `Batch::into_compacted` (batch.rs:884): a batch without a selection is returned unchanged,
else it is gathered at the selection. -/
def Bat.intoCompacted (b : Bat V) : Bat V :=
  match b.sel with
  | none => b
  | some s => b.gather s

/-- **PROVEN** (`Batch::into_compacted`): the result has no selection, as many rows as the input
had active rows, and row `k` is the input's `k`-th active row with its origin tag. -/
theorem intoCompacted_row (b : Bat V) (k : Nat) (hk : k < b.activeLen) :
    b.intoCompacted.sel = none ∧ b.intoCompacted.len = b.activeLen ∧
    (∀ id, b.intoCompacted.cell id k = b.cell id (b.activeIdx.getD k 0)) ∧
    b.intoCompacted.originRow k = b.originRow (b.activeIdx.getD k 0) := by
  unfold Bat.intoCompacted
  cases hs : b.sel with
  | some s =>
    have hk' : k < s.length := by simpa [Bat.activeLen, Bat.activeIdx, hs] using hk
    obtain ⟨h1, h2, h3, h4⟩ := gather_row b s k hk'
    simp only [Bat.activeLen, Bat.activeIdx, hs]
    exact ⟨h1, h2, h3, h4⟩
  | none =>
    have hk' : k < b.len := by simpa [Bat.activeLen, Bat.activeIdx, hs] using hk
    have hg : b.activeIdx.getD k 0 = k := by
      simp [Bat.activeIdx, hs, List.getD_eq_getElem?_getD, hk']
    have hg2 : (List.range b.len)[k]?.getD 0 = k := by simp [hk']
    simp [hs, Bat.activeLen, Bat.activeIdx, hg2]

theorem intoCompacted_sel_len (b : Bat V) :
    b.intoCompacted.sel = none ∧ b.intoCompacted.len = b.activeLen := by
  cases hs : b.sel <;> simp [Bat.intoCompacted, hs, Bat.activeLen, Bat.activeIdx, Bat.gather]

/-- `Batch::clone_active_rows_seq_origin` (batch.rs:1170): compact the active rows, then stamp
row `i` with origin `i`; the sidecar is omitted when `n ≤ 1`. -/
def Bat.cloneSeq (b : Bat V) : Bat V :=
  let c := b.intoCompacted
  { c with origin := if 1 < c.len then some (List.range c.len) else none }

/-- **PROVEN** (`Batch::clone_active_rows_seq_origin`): the snapshot is dense, holds the active
rows in active order, every row `k` reads back origin `k` (also when the sidecar is omitted), and
the sidecar exists exactly when there is more than one row. -/
theorem cloneSeq_spec (b : Bat V) :
    b.cloneSeq.sel = none ∧ b.cloneSeq.len = b.activeLen ∧
    b.cloneSeq.hasOrigins = decide (1 < b.activeLen) ∧
    ∀ k, k < b.activeLen →
      b.cloneSeq.originRow k = k ∧
      ∀ id, b.cloneSeq.cell id k = b.cell id (b.activeIdx.getD k 0) := by
  obtain ⟨hsel, hlen⟩ := intoCompacted_sel_len b
  refine ⟨by simpa [Bat.cloneSeq] using hsel, by simpa [Bat.cloneSeq] using hlen, ?_,
    fun k hk => ⟨?_, fun id => ?_⟩⟩
  · by_cases h : 1 < b.activeLen <;> simp [Bat.hasOrigins, Bat.cloneSeq, hlen, h]
  · by_cases h : 1 < b.activeLen
    · simp [Bat.originRow, Bat.cloneSeq, hlen, h, List.getD_eq_getElem?_getD, hk]
    · have : k = 0 := by omega
      simp [Bat.originRow, Bat.cloneSeq, hlen, h, this]
  · exact (intoCompacted_row b k hk).2.2.1 id

/-- `Correlated.tag` is the origin stamping: tagging a list from `m` pairs element `k` with `m + k`. -/
theorem tag_eq_range {α : Type} [Inhabited α] (xs : List α) (m : Nat) :
    Correlated.tag xs m = (List.range xs.length).map (fun k => (m + k, xs.getD k default)) := by
  induction xs generalizing m with
  | nil => simp [Correlated.tag]
  | cons x xs ih =>
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [Correlated.tag, ih (m + 1)]
    simp [Function.comp_def, Nat.add_assoc, Nat.add_comm 1]

/-- **PROVEN** (the link to `Correlated.tag`): the rows the sequential stamping hands to the
sub-plan are `tag rows 0` — row `k` of the snapshot carries origin `k` — so `Correlated`'s
`applyBatched`/`optionalBatched` start from exactly what `clone_active_rows_seq_origin` builds. -/
theorem cloneSeq_tag {α : Type} [Inhabited α] (b : Bat V) (rows : List α)
    (hlen : rows.length = b.activeLen) :
    Correlated.tag rows 0 =
      (List.range b.activeLen).map (fun k => (b.cloneSeq.originRow k, rows.getD k default)) := by
  rw [tag_eq_range, hlen]
  apply List.map_congr_left
  intro k hk
  have := (cloneSeq_spec b).2.2.2 k (List.mem_range.mp hk)
  simp [this.1]

/-! ## `merge_over_input` (batch.rs:944) -/

/-- `is_bound_at(id, _)`: the row argument is ignored; a column is bound unless `Unbound` or
flagged `value_only`. -/
def Bat.isBound (b : Bat V) (id : Nat) : Bool :=
  match b.cols id with
  | .unbound => false
  | .vals _ => !b.vo id

/-- `set_column` (batch.rs:1233): installs the column and clears the `value_only` marker. -/
def Bat.setColumn (b : Bat V) (id : Nat) (col : Col V) : Bat V :=
  { b with cols := fun j => if j = id then col else b.cols j,
           vo := fun j => if j = id then false else b.vo j,
           ncols := max b.ncols (id + 1) }

/-- `merge_over_input(self, input, origins)`: the sub-plan batch passes through compacted; every
input column bound at row 0 and not bound in the sub-plan batch is restored, gathered by origin. -/
def Bat.mergeOverInput (self input : Bat V) (origins : List Nat) : Bat V :=
  let missing := (List.range input.ncols).filter (fun id => input.isBound id && !self.isBound id)
  missing.foldl (fun out id => out.setColumn id ((input.cols id).gather origins)) self.intoCompacted

/-- `Row::merge` on one slot (row.rs:209): a slot bound in `other` replaces it, anything else
keeps `self`'s slot (value *and* bound bit). -/
def mergeCell (inp sub : Option (V × Bool)) : Option (V × Bool) :=
  match sub with
  | some (_, true) => sub
  | _ => inp

def boundOf : Option (V × Bool) → Bool
  | some (_, true) => true
  | _ => false

theorem setColumn_cell (b : Bat V) (id j k : Nat) (xs : List V) :
    (b.setColumn j (.vals xs)).cell id k =
      if id = j then some (xs.getD k default, true) else b.cell id k := by
  unfold Bat.setColumn Bat.cell
  by_cases h : id = j <;> simp [h]

theorem setColumn_cell_unb (b : Bat V) (id j k : Nat) :
    (b.setColumn j .unbound).cell id k = if id = j then none else b.cell id k := by
  unfold Bat.setColumn Bat.cell
  by_cases h : id = j <;> simp [h]

theorem foldl_restore (input : Bat V) (origins : List Nat) (k : Nat) (hk : k < origins.length) :
    ∀ (ms : List Nat) (c : Bat V), ms.Nodup →
      (∀ m ∈ ms, input.isBound m = true) →
      ∀ id, (ms.foldl (fun out m => out.setColumn m ((input.cols m).gather origins)) c).cell id k =
        if id ∈ ms then input.cell id (origins.getD k 0) else c.cell id k := by
  intro ms
  induction ms with
  | nil => intro c _ _ id; simp
  | cons m ms ih =>
    intro c hnd hb id
    have hnd' := List.nodup_cons.mp hnd
    simp only [List.foldl_cons]
    rw [ih _ hnd'.2 (fun x hx => hb x (by simp [hx])) id]
    have hbm := hb m (by simp)
    unfold Bat.isBound at hbm
    cases hcm : input.cols m with
    | unbound => simp [hcm] at hbm
    | vals xs =>
      simp only [hcm, Col.gather]
      rw [setColumn_cell]
      have hvo : input.vo m = false := by simpa [hcm] using hbm
      by_cases h1 : id = m
      · subst h1
        simp [hnd'.1, Bat.cell, hcm, hvo, List.getD_eq_getElem?_getD, List.getElem?_map, hk]
      · by_cases h2 : id ∈ ms <;> simp [h1, h2]

theorem boundOf_cell (b : Bat V) (id k : Nat) : boundOf (b.cell id k) = b.isBound id := by
  unfold Bat.cell Bat.isBound
  cases b.cols id with
  | unbound => simp [boundOf]
  | vals xs => cases hv : b.vo id <;> simp [boundOf, hv]

theorem mergeCell_bound (inp sub : Option (V × Bool)) (h : boundOf sub = true) :
    mergeCell inp sub = sub := by
  rcases sub with _ | ⟨v, _ | _⟩ <;> simp_all [mergeCell, boundOf]

theorem mergeCell_unbound (inp sub : Option (V × Bool)) (h : boundOf sub = false) :
    mergeCell inp sub = inp := by
  rcases sub with _ | ⟨v, _ | _⟩ <;> simp_all [mergeCell, boundOf]

/-- **PROVEN** (`Batch::merge_over_input`, the columnar `input_row.clone().merge(sub_row)`): for
every row `k` of the sub-plan batch and every variable, the output slot is bound exactly when
`Row::merge` leaves it bound, and then holds the same value: the sub-plan's own value when it
binds the slot, else the input row's (the row picked by the origin). Needs only that the input has
no column past `ncols` and one origin per active row.

The one place the columnar form is not `Row::merge` is a slot *neither side binds*: the output
keeps the sub-plan's unbound slot, `merge` keeps the input's. That is visible only to a reader of a
stored-but-unbound value (`get_by_id`, the #3368 family), and both slots are unbound. -/
theorem mergeOverInput_row (self input : Bat V) (origins : List Nat)
    (hin : ∀ id, input.ncols ≤ id → input.cols id = .unbound)
    (ho : origins.length = self.activeLen) (k : Nat) (hk : k < self.activeLen) (id : Nat) :
    boundOf ((self.mergeOverInput input origins).cell id k) =
      boundOf (mergeCell (input.cell id (origins.getD k 0)) (self.cell id (self.activeIdx.getD k 0))) ∧
    (boundOf (mergeCell (input.cell id (origins.getD k 0)) (self.cell id (self.activeIdx.getD k 0))) = true →
      (self.mergeOverInput input origins).cell id k =
        mergeCell (input.cell id (origins.getD k 0)) (self.cell id (self.activeIdx.getD k 0))) := by
  have hk' : k < origins.length := by omega
  have hnd : ((List.range input.ncols).filter
      (fun id => input.isBound id && !self.isBound id)).Nodup :=
    List.Nodup.sublist (List.filter_sublist) (List.nodup_range)
  have hfold := foldl_restore input origins k hk'
    ((List.range input.ncols).filter (fun id => input.isBound id && !self.isBound id))
    self.intoCompacted hnd (by
      intro m hm
      simp only [List.mem_filter, Bool.and_eq_true] at hm
      exact hm.2.1) id
  have hrow := (intoCompacted_row self k hk).2.2.1 id
  have hS := boundOf_cell self id (self.activeIdx.getD k 0)
  have hI := boundOf_cell input id (origins.getD k 0)
  have hin' : input.ncols ≤ id → input.isBound id = false := fun h => by
    simp [Bat.isBound, hin id h]
  unfold Bat.mergeOverInput
  simp only []
  rw [hfold, hrow]
  by_cases hsb : boundOf (self.cell id (self.activeIdx.getD k 0)) = true
  · have hsb2 : self.isBound id = true := by rw [← hS]; exact hsb
    have hnm : id ∉ (List.range input.ncols).filter
        (fun id => input.isBound id && !self.isBound id) := by
      simp [List.mem_filter, hsb2]
    simp only [hnm, ↓reduceIte, mergeCell_bound _ _ hsb]
    first | exact ⟨trivial, fun _ => trivial⟩ | exact ⟨rfl, fun _ => rfl⟩
  · have hsb' : boundOf (self.cell id (self.activeIdx.getD k 0)) = false := by simpa using hsb
    rw [mergeCell_unbound _ _ hsb']
    by_cases hib : input.isBound id = true
    · have hmem : id ∈ (List.range input.ncols).filter
          (fun id => input.isBound id && !self.isBound id) := by
        have hlt : id < input.ncols := Nat.lt_of_not_le (fun hc => by have := hin' hc; simp_all)
        have hsb2 : self.isBound id = false := by rw [← hS]; exact hsb'
        simp [List.mem_filter, hlt, hib, hsb2]
      simp only [hmem, ↓reduceIte]
      first | exact ⟨trivial, fun _ => trivial⟩ | exact ⟨rfl, fun _ => rfl⟩
    · have hnm : id ∉ (List.range input.ncols).filter
          (fun id => input.isBound id && !self.isBound id) := by
        simp [List.mem_filter, hib]
      simp only [hnm, ↓reduceIte]
      have hib' : input.isBound id = false := by simpa using hib
      refine ⟨by rw [hsb', hI, hib'], fun h => ?_⟩
      rw [hI, hib'] at h; exact absurd h (by simp)

/-- The slot neither side binds: `merge` keeps the input's unbound value, the columnar form keeps
the sub-plan's (a `value_only` column at the same id on both sides, different values). -/
example : let self : Bat Nat := ⟨fun _ => .vals [7], 1, fun _ => true, 1, none, none⟩
    let input : Bat Nat := ⟨fun _ => .vals [3], 1, fun _ => true, 1, none, none⟩
    (self.mergeOverInput input [0]).cell 0 0 = some (7, false) ∧
    mergeCell (input.cell 0 0) (self.cell 0 0) = some (3, false) := by decide

/-! ## `set_argument_batch` (batch.rs:1603)

The operator tree is modelled by what the dispatch does to it: an `Argument` leaf stores the batch,
`Once` ignores it, every other operator resets its queued state (`emitter.reset()`,
`pending_batches.clear()`, `hash_table = None`, …: the `reset` mark), a `ValueHashJoin` also sets
`per_origin := len batch > 1`, a `CartesianProduct` / `ValueHashJoin` hands its right children the
*compacted* batch (`batch.clone().into_compacted()`), and the main child gets the batch itself.
`Sort`/`Aggregate`/`Commit` skip a missing child (`if let Some`). -/

/-- Operator state the dispatch touches: how often it was reset, and the join's `per_origin`. -/
structure St where
  resets : Nat
  perOrigin : Bool
  isVhj : Bool

inductive OT (B : Type) where
  | argument (slot : Option B)
  | once
  | node (st : St) (rights : List (OT B)) (child : List (OT B))

variable {B : Type}

/-- `f` applied `n` times (the compaction a right child's batch went through). -/
def iter (f : B → B) : Nat → B → B
  | 0, x => x
  | n + 1, x => iter f n (f x)

mutual
/-- `BatchOp::set_argument_batch`: `compact` is `into_compacted`, `len` the batch's length. -/
def setArg (compact : B → B) (len : B → Nat) (b : B) : OT B → OT B
  | .argument _ => .argument (some b)
  | .once => .once
  | .node st rs cs =>
    .node { st with resets := st.resets + 1,
                    perOrigin := if st.isVhj then decide (1 < len b) else st.perOrigin }
      (setArgL compact len (compact b) rs) (setArgL compact len b cs)
def setArgL (compact : B → B) (len : B → Nat) (b : B) : List (OT B) → List (OT B)
  | [] => []
  | t :: ts => setArg compact len b t :: setArgL compact len b ts
end

mutual
/-- The `Argument` slots of a tree, left to right (right children first, as in the node). -/
def slots : OT B → List (Option B)
  | .argument s => [s]
  | .once => []
  | .node _ rs cs => slotsL rs ++ slotsL cs
def slotsL : List (OT B) → List (Option B)
  | [] => []
  | t :: ts => slots t ++ slotsL ts
end

mutual
/-- The tree with every slot cleared and every counter zeroed: its shape. -/
def shape : OT B → OT B
  | .argument _ => .argument none
  | .once => .once
  | .node st rs cs => .node { st with resets := 0, perOrigin := false } (shapeL rs) (shapeL cs)
def shapeL : List (OT B) → List (OT B)
  | [] => []
  | t :: ts => shape t :: shapeL ts
end

mutual
/-- **PROVEN** (`BatchOp::set_argument_batch`): every `Argument` leaf of the tree holds a batch
afterwards — the batch itself, or (below a `CartesianProduct` / `ValueHashJoin` right child) the
batch compacted once per such join on the way down. No leaf is left empty or stale, and `Once`
has none. -/
theorem slots_setArg (compact : B → B) (len : B → Nat) : ∀ (t : OT B) (b : B),
    ∀ s ∈ slots (setArg compact len b t), ∃ n, s = some (iter compact n b)
  | .argument _, b => by simp [setArg, slots]; exact ⟨0, rfl⟩
  | .once, b => by simp [setArg, slots]
  | .node st rs cs, b => by
    intro s hs
    simp only [setArg, slots, List.mem_append] at hs
    rcases hs with h | h
    · obtain ⟨n, hn⟩ := slotsL_setArg compact len rs (compact b) s h
      exact ⟨n + 1, by rw [hn]; rfl⟩
    · exact slotsL_setArg compact len cs b s h
theorem slotsL_setArg (compact : B → B) (len : B → Nat) : ∀ (ts : List (OT B)) (b : B),
    ∀ s ∈ slotsL (setArgL compact len b ts), ∃ n, s = some (iter compact n b)
  | [], b => by simp [setArgL, slotsL]
  | t :: ts, b => by
    intro s hs
    simp only [setArgL, slotsL, List.mem_append] at hs
    rcases hs with h | h
    · exact slots_setArg compact len t b s h
    · exact slotsL_setArg compact len ts b s h
end

/-- **PROVEN**: with `into_compacted` idempotent (a batch without a selection is returned
unchanged), every leaf holds the batch or its compaction. -/
theorem slots_setArg_idem (compact : B → B) (len : B → Nat) (hid : ∀ x, compact (compact x) = compact x)
    (t : OT B) (b : B) : ∀ s ∈ slots (setArg compact len b t), s = some b ∨ s = some (compact b) := by
  intro s hs
  obtain ⟨n, hn⟩ := slots_setArg compact len t b s hs
  have hit : ∀ m x, iter compact m (compact x) = compact x := by
    intro m
    induction m with
    | zero => intro x; rfl
    | succ m ih => intro x; show iter compact m (compact (compact x)) = _; rw [hid, ih]
  rcases n with _ | m
  · left; exact hn
  · right; rw [hn]; exact congrArg some (hit m b)

/-- `ValueHashJoin`'s `per_origin` after the dispatch is `len batch > 1` (batch.rs:1700). -/
theorem setArg_vhj_perOrigin (compact : B → B) (len : B → Nat) (b : B) (st : St) (hv : st.isVhj = true)
    (rs cs : List (OT B)) :
    ∃ st' rs' cs', setArg compact len b (.node st rs cs) = .node st' rs' cs' ∧
      st'.perOrigin = decide (1 < len b) := by
  refine ⟨_, _, _, rfl, ?_⟩
  simp [setArg, hv]

/-- Any other operator keeps `per_origin` and is reset once. -/
theorem setArg_resets (compact : B → B) (len : B → Nat) (b : B) (st : St) (hv : st.isVhj = false)
    (rs cs : List (OT B)) :
    ∃ st' rs' cs', setArg compact len b (.node st rs cs) = .node st' rs' cs' ∧
      st'.perOrigin = st.perOrigin ∧ st'.resets = st.resets + 1 := by
  refine ⟨_, _, _, rfl, ?_⟩
  simp [setArg, hv]

mutual
/-- The dispatch leaves the tree's shape alone (only slots and counters change). -/
theorem shape_setArg (compact : B → B) (len : B → Nat) : ∀ (t : OT B) (b : B),
    shape (setArg compact len b t) = shape t
  | .argument _, b => rfl
  | .once, b => rfl
  | .node st rs cs, b => by
    simp only [setArg, shape, shape_setArgL compact len rs (compact b),
      shape_setArgL compact len cs b]
theorem shape_setArgL (compact : B → B) (len : B → Nat) : ∀ (ts : List (OT B)) (b : B),
    shapeL (setArgL compact len b ts) = shapeL ts
  | [], b => rfl
  | t :: ts, b => by
    simp only [setArgL, shapeL, shape_setArg compact len t b, shape_setArgL compact len ts b]
end

end BatchRows
