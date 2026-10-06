import Columnar.Concat
/-
The remaining `runtime/batch.rs` functions: column helpers, the builder's per-column push
paths and merge plan, batch constructors/accessors, `merge_over_input`,
`clone_active_rows_seq_origin`, `BatchRow::new`, `ActiveIndices::size_hint`, and the
operator-tree plumbing (`set_argument_batch`, `inspect_context`, `BatchOp::next`).

| here | there |
| --- | --- |
| `Column.isEmptyC` | `Column::is_empty` (batch.rs:370) |
| `Column.compareAt` | `Column::compare_at` (:400) |
| `CB`, `CB.pushBound`, `CB.pushLeftOf` | `ColumnBuilder::push_bound` (:430), `push_left_of` (:443) |
| `forRights`, `planWidth` | `MergePlan::for_rights` (:499), `width` (:520) |
| `BB`, `BB.new/len/isEmpty/growCols/finishRow` | `BatchBuilder::new` (:548), `len` (:554), `is_empty` (:560), `grow_cols` (:722) |
| `BB.pushMerged`, `BB.pushPlanned` | `push_merged` (:660), `push_planned` (:694) |
| `Batch.new0`, `Batch.fromColumns` | `Batch::new` (:831), `from_columns` (:866) |
| `Batch.mergeOverInput` | `merge_over_input` (:943) |
| `Batch.seqOrigin` | `clone_active_rows_seq_origin` (:1163) |
| accessors | `len` (:1172), `active_len` (:1178), `is_empty` (:1184), `selection` (:1190), `column` (:1213), `compare_rows_at` (:1282), `origin_row` (:1298), `set_origin_rows` (:1319), `num_columns` (:1328), `is_bound_at` (:1336), `extract_node_ids` (:1372), `extract_rel_ids` (:1385) |
| `BatchRow.new`, `sizeHint` | `BatchRow::new` (:1407), `ActiveIndices::size_hint` (:1481) |
| `OpTree`, `setArg`, `inspectCtx`, `opNext` | `BatchOp::set_argument_batch` (:1590), `inspect_context` (:1722), `Iterator for BatchOp::next` (:1771) |
-/
namespace Columnar

variable {F : Type}

namespace Column

/-- batch.rs:370 -/
def isEmptyC (c : Column F) : Bool := c.len == 0

theorem isEmptyC_iff (c : Column F) : c.isEmptyC = true ↔ c.len = 0 := by simp [isEmptyC]
theorem isEmptyC_unbound : (Column.unbound : Column F).isEmptyC = true := rfl

/-- batch.rs:400-414. `cv` = `Value::compare_value(..).0`; `pcmp` = `f64::partial_cmp`. -/
def compareAt [FloatModel F] (cv : V F → V F → Ordering) (c : Column F) (a b : Nat) : Ordering :=
  match c with
  | .ints v => compare (v.getD a 0) (v.getD b 0)
  | .floats v => match v[a]?, v[b]? with
    | some x, some y => (FloatModel.pcmp x y).getD .lt
    | _, _ => .lt
  | .values v => match v[a]?, v[b]? with
    | some x, some y => cv x y
    | _, _ => .eq
  | c => cv ((c.get? a).getD .null) ((c.get? b).getD .null)

/-- The `Values` lane is `compare_value`; an int lane is `Int` order; on the float lane a
NaN pair compares `Less` BOTH ways (`unwrap_or(Less)`), so the comparator is not
antisymmetric — the ORDER BY panic of known issue #2891. -/
theorem compareAt_spec [FloatModel F] (cv : V F → V F → Ordering) (vs : List (V F)) (is : List Int)
    (fs : List F) (a b : Nat) (ha : a < fs.length) (hb : b < fs.length)
    (hnan : FloatModel.pcmp fs[a] fs[b] = none) :
    ((Column.values vs).compareAt cv a b = match vs[a]?, vs[b]? with
      | some x, some y => cv x y | _, _ => .eq) ∧
    (Column.ints is).compareAt cv a b = compare (is.getD a 0) (is.getD b 0) ∧
    (Column.floats fs).compareAt cv a b = .lt ∧ (Column.floats fs).compareAt cv b a = .lt := by
  have hnan' : FloatModel.pcmp fs[b] fs[a] = none := by rw [FloatModel.pcmp_swap, hnan]; rfl
  refine ⟨rfl, rfl, ?_, ?_⟩
  · simp [compareAt, List.getElem?_eq_getElem ha, List.getElem?_eq_getElem hb, hnan]
  · simp [compareAt, List.getElem?_eq_getElem ha, List.getElem?_eq_getElem hb, hnan']

end Column

/-! ## `ColumnBuilder` -/

structure CB (F : Type) where
  values : List (V F)
  present : Bool
  anyBound : Bool

/-- batch.rs:430-437 -/
def CB.pushBound (c : CB F) (v : V F) : CB F := ⟨c.values ++ [v], true, true⟩

/-- batch.rs:443-462: copy the left base row's slot; a value-only slot marks presence only
when non-null and never binds; an absent/`Unbound` slot pushes `Null`. -/
def CB.pushLeftOf (c : CB F) (base : Batch F) (id row : Nat) : CB F :=
  match base.cols[id]? with
  | some col =>
    if col.isUnbound then { c with values := c.values ++ [.null] }
    else
      let v := (col.get? row).getD .null
      if base.vo id then { c with values := c.values ++ [v], present := c.present || !v.isNull }
      else c.pushBound v
  | none => { c with values := c.values ++ [.null] }

theorem pushBound_spec (c : CB F) (v : V F) :
    (c.pushBound v).values = c.values ++ [v] ∧ (c.pushBound v).present ∧ (c.pushBound v).anyBound :=
  ⟨rfl, rfl, rfl⟩

/-- `push_left_of` appends exactly the base row's `value_at` (or `Null` when unbound/absent),
and binds the slot iff the base binds it. -/
theorem pushLeftOf_spec (c : CB F) (base : Batch F) (id row : Nat) :
    (c.pushLeftOf base id row).values = c.values ++ [(base.valueAt id row).getD .null] ∧
    ((c.pushLeftOf base id row).anyBound = (c.anyBound || base.isBound id)) := by
  unfold CB.pushLeftOf Batch.valueAt Batch.isBound Batch.column
  cases h : base.cols[id]? with
  | none => simp [Column.isUnbound]
  | some col =>
    simp only [Option.getD_some]
    cases hu : col.isUnbound
    · cases hv : base.vo id <;> simp [CB.pushBound, hu, hv]
    · simp [hu]

/-! ## `MergePlan` (batch.rs:499, :520) -/

inductive Src where
  | right (i : Nat)
  | left
  deriving DecidableEq

/-- Does right `i` bind slot `id` (`is_bound_at(id, 0)`: column-level). -/
def bindsAt (rights : List (Batch F)) (id i : Nat) : Bool := (rights[i]?.map (·.isBound id)).getD false

/-- For each slot up to the widest right, the LAST right that binds it (else the left). -/
def lastBinder (rights : List (Batch F)) (id : Nat) : Src :=
  ((List.range rights.length).reverse.find? (bindsAt rights id)).elim .left .right

def forRights (rights : List (Batch F)) : List Src :=
  let width := (rights.map fun r => r.cols.length).foldl max 0
  (List.range width).map (lastBinder rights)

def planWidth (p : List Src) : Nat := p.length

theorem forRights_spec (rights : List (Batch F)) (id : Nat) (h : id < planWidth (forRights rights)) :
    (forRights rights)[id]'h = lastBinder rights id ∧
    planWidth (forRights rights) = (rights.map fun r => r.cols.length).foldl max 0 := by
  simp [forRights, planWidth]

/-- `Right i` is a binding right. -/
theorem lastBinder_right (rights : List (Batch F)) (id i : Nat) (h : lastBinder rights id = .right i) :
    i < rights.length ∧ bindsAt rights id i = true := by
  unfold lastBinder at h
  cases hf : (List.range rights.length).reverse.find? (bindsAt rights id) with
  | none => rw [hf] at h; cases h
  | some j =>
    rw [hf] at h
    simp only [Option.elim_some, Src.right.injEq] at h
    subst h
    have hm := List.mem_of_find?_eq_some hf
    exact ⟨by simpa using hm, List.find?_some hf⟩

/-- `Left`: no right binds the slot. -/
theorem lastBinder_left (rights : List (Batch F)) (id : Nat) (h : lastBinder rights id = .left) :
    ∀ i < rights.length, bindsAt rights id i = false := by
  unfold lastBinder at h
  cases hf : (List.range rights.length).reverse.find? (bindsAt rights id) with
  | some j => rw [hf] at h; cases h
  | none =>
    intro i hi
    have := List.find?_eq_none.mp hf i (by simp [hi])
    simpa using this

/-! ## `BatchBuilder` -/

structure BB (F : Type) where
  cols : List (CB F)
  origins : List Nat
  anyOrigin : Bool
  rows : Nat

/-- batch.rs:548 (`Default`) -/
def BB.new : BB F := ⟨[], [], false, 0⟩
def BB.len (b : BB F) : Nat := b.rows
def BB.isEmpty (b : BB F) : Bool := b.rows == 0

theorem bbNew_spec : (BB.new : BB F).len = 0 ∧ (BB.new : BB F).isEmpty = true ∧ (BB.new : BB F).cols = [] := ⟨rfl, rfl, rfl⟩
theorem bbIsEmpty_iff (b : BB F) : b.isEmpty = true ↔ b.len = 0 := by simp [BB.isEmpty, BB.len]

/-- batch.rs:722-734: add `Null`-back-filled columns up to `n`. -/
def BB.growCols (b : BB F) (n : Nat) : BB F :=
  { b with cols := b.cols ++ List.replicate (n - b.cols.length) ⟨List.replicate b.rows .null, false, false⟩ }

theorem growCols_spec (b : BB F) (n : Nat) :
    (b.growCols n).cols.length = max n b.cols.length ∧
    (∀ i (h : i < b.cols.length) (h' : i < (b.growCols n).cols.length), (b.growCols n).cols[i] = b.cols[i]) ∧
    (∀ i (h : b.cols.length ≤ i) (h' : i < (b.growCols n).cols.length),
      (b.growCols n).cols[i].values = List.replicate b.rows .null ∧ (b.growCols n).cols[i].anyBound = false) := by
  refine ⟨by simp [BB.growCols]; omega, fun i h h' => by simp [BB.growCols, List.getElem_append_left h], ?_⟩
  intro i h h'
  simp [BB.growCols, List.getElem_append_right h]

def BB.finishRow (b : BB F) (origin : Nat) : BB F :=
  { b with anyOrigin := b.anyOrigin || origin != 0, origins := b.origins ++ [origin], rows := b.rows + 1 }

/-- batch.rs:660-678: per slot, the right row's value when the right binds it, else the left. -/
def BB.pushMerged (b : BB F) (left : Batch F) (lr : Nat) (right : Batch F) (rr : Nat) (origin : Nat) : BB F :=
  let g := b.growCols (max left.cols.length right.cols.length)
  let cols := (List.range g.cols.length).zip g.cols |>.map fun (id, c) =>
    if right.isBound id then c.pushBound ((right.valueAt id rr).getD .null)
    else c.pushLeftOf left id lr
  ({ g with cols }).finishRow origin

/-- batch.rs:694-718: the plan names the right per slot. -/
def BB.pushPlanned (b : BB F) (left : Batch F) (lr : Nat) (rights : List (Batch F)) (cursor : List Nat)
    (plan : List Src) (origin : Nat) : BB F :=
  let g := b.growCols (max (planWidth plan) left.cols.length)
  let cols := (List.range g.cols.length).zip g.cols |>.map fun (id, c) =>
    match plan[id]? with
    | some (.right i) =>
      let r := rights.getD i ⟨0, none, [], none, fun _ => false⟩
      c.pushBound (((r.column id).get? (cursor.getD i 0)).getD .null)
    | _ => c.pushLeftOf left id lr
  ({ g with cols }).finishRow origin

/-- One appended row: every column gains exactly one value; row count and origins advance. -/
theorem pushMerged_spec (b : BB F) (left right : Batch F) (lr rr origin : Nat) :
    let b' := b.pushMerged left lr right rr origin
    b'.rows = b.rows + 1 ∧ b'.origins = b.origins ++ [origin] ∧
    b'.cols.length = max (max left.cols.length right.cols.length) b.cols.length ∧
    ∀ id (h : id < b'.cols.length) (h2 : id < (b.growCols (max left.cols.length right.cols.length)).cols.length),
      b'.cols[id].values = (b.growCols (max left.cols.length right.cols.length)).cols[id].values ++
        [if right.isBound id then (right.valueAt id rr).getD .null else (left.valueAt id lr).getD .null] := by
  intro b'
  refine ⟨rfl, rfl, by simp [b', BB.pushMerged, BB.finishRow, (growCols_spec b _).1], fun id h h2 => ?_⟩
  simp only [b', BB.pushMerged, BB.finishRow, List.getElem_map, List.getElem_zip, List.getElem_range]
  split
  · rfl
  · exact (pushLeftOf_spec _ _ _ _).1

theorem pushPlanned_spec (b : BB F) (left : Batch F) (lr : Nat) (rights : List (Batch F)) (cur : List Nat)
    (plan : List Src) (origin : Nat) :
    let b' := b.pushPlanned left lr rights cur plan origin
    b'.rows = b.rows + 1 ∧ b'.origins = b.origins ++ [origin] ∧
    ∀ id (h : id < b'.cols.length) (h2 : id < (b.growCols (max (planWidth plan) left.cols.length)).cols.length),
      plan[id]? = some .left ∨ plan[id]? = none →
      b'.cols[id].values = (b.growCols (max (planWidth plan) left.cols.length)).cols[id].values ++
        [(left.valueAt id lr).getD .null] := by
  intro b'
  refine ⟨rfl, rfl, fun id h h2 hp => ?_⟩
  simp only [b', BB.pushPlanned, BB.finishRow, List.getElem_map, List.getElem_zip, List.getElem_range]
  rcases hp with hp | hp <;> simp only [hp] <;> exact (pushLeftOf_spec _ _ _ _).1

/-! ## Batch constructors and accessors -/

namespace Batch

/-- batch.rs:831 -/
def new0 (n : Nat) : Batch F := ⟨0, none, List.replicate n .unbound, none, fun _ => false⟩

theorem new0_spec (n i : Nat) : (new0 n : Batch F).column i = .unbound ∧ (new0 n : Batch F).activeLen = 0 ∧
    (new0 n : Batch F).isBound i = false ∧ (new0 n : Batch F).WF := by
  refine ⟨?_, by simp [new0, activeLen, active], ?_, ?_⟩
  · simp only [column, new0, List.getElem?_replicate]; split <;> rfl
  · simp only [isBound, column, new0, List.getElem?_replicate]; split <;> rfl
  · constructor
    · intro c hc; simp [new0] at hc; obtain ⟨_, rfl⟩ := hc; intro h; cases h
    · intro s hs; cases hs
    · intro s hs; cases hs
    · intro o ho; cases ho

/-- batch.rs:866: `set_column(i, cols[i])` in order. -/
def fromColumns [FloatModel F] (cs : List (Column F)) : Batch F :=
  (cs.zipIdx).foldl (fun b p => b.setColumn p.2 p.1) (new0 0)

theorem foldl_setColumn_column [FloatModel F] (ps : List (Column F × Nat)) (b : Batch F) (j : Nat) :
    (ps.foldl (fun b p => b.setColumn p.2 p.1) b).column j =
      ((ps.reverse.find? (·.2 = j)).map (normCol ·.1)).getD (b.column j) := by
  induction ps generalizing b with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.foldl_cons, ih, column_setColumn, List.reverse_cons, List.find?_append]
    cases hf : ps.reverse.find? (·.2 = j) with
    | some q => simp
    | none =>
      by_cases h : p.2 = j
      · simp [h]
      · have : ¬ j = p.2 := fun e => h e.symm
        simp [h, this]

/-- Column `i` of `from_columns(cs)` is `cs[i]` (re-classified like any `set_column`). -/
theorem fromColumns_column [FloatModel F] (cs : List (Column F)) (i : Nat) (hi : i < cs.length) :
    (fromColumns cs).column i = normCol cs[i] := by
  unfold fromColumns
  rw [foldl_setColumn_column]
  have hmem : (cs[i], i) ∈ cs.zipIdx.reverse := by
    rw [List.mem_reverse, List.mk_mem_zipIdx_iff_getElem?]; simp [hi]
  cases hf : cs.zipIdx.reverse.find? (·.2 = i) with
  | none =>
    have := List.find?_eq_none.mp hf _ hmem; simp at this
  | some p =>
    have hp := List.find?_some hf
    have hm := List.mem_of_find?_eq_some hf
    simp only [decide_eq_true_eq] at hp
    obtain ⟨x, j⟩ := p
    simp only at hp; subst hp
    rw [List.mem_reverse, List.mk_mem_zipIdx_iff_getElem?] at hm
    rw [List.getElem?_eq_getElem hi] at hm
    cases hm; rfl

/-- batch.rs:1172-1190, 1213, 1298-1336, 1372, 1385 -/
def lenB (b : Batch F) : Nat := b.len
def isEmptyB (b : Batch F) : Bool := b.activeLen == 0
def selection (b : Batch F) : Option (List Nat) := b.sel
def numColumns (b : Batch F) : Nat := b.cols.length
def setOriginRows (b : Batch F) (o : List Nat) : Batch F := { b with origins := some o }
def extractNodeIds (b : Batch F) (i : Nat) : Option (List Nat) :=
  match b.column i with | .nodeIds v => some v | _ => none
def extractRelIds (b : Batch F) (i : Nat) : Option (List Nat) :=
  match b.column i with | .relIds v => some v | _ => none

theorem accessors_spec (b : Batch F) (i r : Nat) (o : List Nat) :
    b.lenB = b.len ∧ b.activeLen = (b.sel.map List.length).getD b.len ∧
    (b.isEmptyB = true ↔ b.activeLen = 0) ∧ b.selection = b.sel ∧
    b.column i = b.cols[i]?.getD .unbound ∧ b.numColumns = b.cols.length ∧
    (b.setOriginRows o).originRow r = o[r]?.getD 0 ∧
    (b.isBound i = true ↔ (b.column i).isUnbound = false ∧ b.vo i = false) ∧
    (∀ v, b.extractNodeIds i = some v ↔ b.column i = .nodeIds v) ∧
    (∀ v, b.extractRelIds i = some v ↔ b.column i = .relIds v) := by
  refine ⟨rfl, ?_, by simp [isEmptyB], rfl, rfl, rfl, rfl, by simp [isBound], fun v => ?_, fun v => ?_⟩
  · cases h : b.sel <;> simp [activeLen, active, h]
  · unfold extractNodeIds; cases b.column i <;> simp
  · unfold extractRelIds; cases b.column i <;> simp

/-- `origin_row`: `0` without a sidecar. -/
theorem originRow_spec (b : Batch F) (r : Nat) :
    b.originRow r = (match b.origins with | none => 0 | some o => o[r]?.getD 0) := rfl

/-- `Batch::rows_only` (batch.rs:853, #2845): `len` rows, no column, no selection,
no origins — what a projection naming nothing produces. -/
def rowsOnly (n : Nat) : Batch F := ⟨n, none, [], none, fun _ => false⟩

theorem rowsOnly_spec (n i r : Nat) :
    (rowsOnly n : Batch F).activeLen = n ∧ (rowsOnly n : Batch F).numColumns = 0 ∧
    (rowsOnly n : Batch F).column i = .unbound ∧ (rowsOnly n : Batch F).originRow r = 0 ∧
    (rowsOnly n : Batch F).WF := by
  refine ⟨by simp [rowsOnly, activeLen, active], rfl, rfl, rfl, ?_⟩
  constructor
  · intro c hc; cases hc
  · intro s hs; cases hs
  · intro s hs; cases hs
  · intro o ho; cases ho

/-- `Batch::has_origins` (batch.rs:1312, #2845): whether the origin sidecar is set. -/
def hasOrigins (b : Batch F) : Bool := b.origins.isSome

/-- `has_origins` tells "never stamped" from "all origins 0", which `origin_row`
cannot: both read 0 everywhere, only the sidecar differs. -/
theorem hasOrigins_spec (b : Batch F) (o : List Nat) (r : Nat) :
    (b.hasOrigins = true ↔ ∃ o, b.origins = some o) ∧
    (b.setOriginRows o).hasOrigins = true ∧
    (b.hasOrigins = false → b.originRow r = 0) := by
  refine ⟨?_, rfl, ?_⟩
  · cases h : b.origins <;> simp [hasOrigins, h]
  · intro h; cases ho : b.origins with
    | none => simp [originRow, ho]
    | some _ => simp [hasOrigins, ho] at h

theorem hasOrigins_zero_not_unset :
    let b : Batch F := { (rowsOnly 2 : Batch F) with origins := some [0, 0] }
    b.hasOrigins = true ∧ (rowsOnly 2 : Batch F).hasOrigins = false ∧
    (∀ r, b.originRow r = (rowsOnly 2 : Batch F).originRow r) := by
  refine ⟨rfl, rfl, fun r => ?_⟩
  simp only [originRow, rowsOnly]
  match r with
  | 0 => rfl
  | 1 => rfl
  | _ + 2 => rfl

/-- batch.rs:1282-1293: unbound compares `Equal`; else `compare_value` of the two cells. -/
def compareRowsAt (cv : V F → V F → Ordering) (b : Batch F) (i x y : Nat) : Ordering :=
  if (b.column i).isUnbound then .eq
  else cv (((b.column i).get? x).getD .null) (((b.column i).get? y).getD .null)

theorem compareRowsAt_spec (cv : V F → V F → Ordering) (b : Batch F) (i x y : Nat) :
    ((b.column i).isUnbound = true → compareRowsAt cv b i x y = .eq) ∧
    ((b.column i).isUnbound = false → compareRowsAt cv b i x y =
      cv ((b.valueAt i x).getD .null) ((b.valueAt i y).getD .null)) := by
  refine ⟨fun h => by simp [compareRowsAt, h], fun h => by simp [compareRowsAt, valueAt, h]⟩

/-- batch.rs:1163-1168: compact, then origins `0..n` when more than one row. -/
def seqOrigin (b : Batch F) : Option (Batch F) :=
  b.compact.map fun c => { c with origins := if c.len > 1 then some (List.range c.len) else none }

/-- Row `k` of the clone is active row `k`, tagged with origin `k` (row 0 reads origin 0 either way). -/
theorem seqOrigin_spec (b : Batch F) (h : b.WF) :
    ∃ c, b.seqOrigin = some c ∧ c.len = b.activeLen ∧
      ∀ k (hk : k < b.activeLen) i, c.obs i k = b.obs i (b.active[k]'hk) ∧ c.originRow k = k := by
  obtain ⟨b', hc, _, hl, _, hobs⟩ := compact_spec h
  refine ⟨{ b' with origins := if b'.len > 1 then some (List.range b'.len) else none },
    by simp [seqOrigin, hc], by simp [hl], fun k hk i => ⟨(hobs k hk i).1, ?_⟩⟩
  simp only [originRow]
  by_cases hlen : b'.len > 1
  · simp only [hlen, ite_true]
    rw [List.getElem?_range (by omega)]; rfl
  · simp only [hlen, ite_false]; omega

/-- batch.rs:943-961: restore the input columns the sub-plan does not bind, gathered by the
output rows' origins; keep everything the sub-plan binds. -/
def mergeOverInput [FloatModel F] (out input : Batch F) (origins : List Nat) : Option (Batch F) := do
  let missing := (List.range input.cols.length).filter fun id => input.isBound id && !out.isBound id
  let o ← out.compact
  missing.foldlM (fun acc id => ((input.column id).gather origins).map (acc.setColumn id)) o

theorem mergeOverInput_bound [FloatModel F] (out input : Batch F) (origins : List Nat) (m : Batch F)
    (h : mergeOverInput out input origins = some m) (o : Batch F) (ho : out.compact = some o) (id : Nat)
    (hb : out.isBound id = true ∨ input.isBound id = false) : m.column id = o.column id := by
  unfold mergeOverInput at h
  simp only [ho, Option.bind_eq_bind, Option.bind_some] at h
  suffices H : ∀ (ms : List Nat) (acc : Batch F), id ∉ ms →
      ms.foldlM (fun acc id => ((input.column id).gather origins).map (acc.setColumn id)) acc = some m →
      m.column id = acc.column id by
    apply H _ o _ h
    simp only [List.mem_filter, List.mem_range, Bool.and_eq_true, Bool.not_eq_true', not_and]
    intro _ hi
    rcases hb with hb | hb <;> simp_all
  intro ms
  induction ms with
  | nil => intro acc _ hm; simp at hm; rw [hm]
  | cons x xs ih =>
    intro acc hn hm
    simp only [List.foldlM_cons] at hm
    cases hg : (input.column x).gather origins with
    | none => simp [hg] at hm
    | some c =>
      simp only [hg, Option.map_some, Option.bind_some] at hm
      rw [ih _ (fun h' => hn (List.mem_cons_of_mem _ h')) hm, column_setColumn]
      have : id ≠ x := fun e => hn (e ▸ List.mem_cons_self ..)
      simp [this]

end Batch

/-! ## `BatchRow::new` (:1407), `ActiveIndices::size_hint` (:1481) -/

structure BatchRow (F : Type) where
  batch : Batch F
  row : Nat

def BatchRow.new (b : Batch F) (r : Nat) : BatchRow F := ⟨b, r⟩

theorem batchRowNew_spec (b : Batch F) (r : Nat) : (BatchRow.new b r).batch = b ∧ (BatchRow.new b r).row = r :=
  ⟨rfl, rfl⟩

/-- `remaining = (selection.len or len) - pos`, as both bounds. -/
def sizeHint (b : Batch F) (pos : Nat) : Nat × Option Nat :=
  let rem := (b.sel.map List.length).getD b.len - pos
  (rem, some rem)

theorem sizeHint_spec (b : Batch F) (pos : Nat) :
    sizeHint b pos = (b.activeLen - pos, some (b.activeLen - pos)) := by
  cases h : b.sel <;> simp [sizeHint, Batch.activeLen, Batch.active, h]

/-! ## Operator-tree plumbing (:1590, :1722, :1771) -/

/-- The operator tree, by how `set_argument_batch` treats each node: an `Argument` leaf, a
`Once` leaf, a pass-through node (one child), a reset-then-pass node (emitter-backed ops,
`ProcedureCall`, `IncludePending`), a fan-out node (`CartesianProduct`, `ValueHashJoin`: the
compacted batch to every right child too), and a `Union` (stores the compacted batch). -/
inductive OpTree (B : Type) where
  | argument (slot : Option B)
  | once
  | pass (child : OpTree B)
  | reset (stale : Bool) (child : OpTree B)
  | fanout (rights : List (OpTree B)) (child : OpTree B)
  | union (stored : Option B) (current : Option (OpTree B))

mutual
def setArg {B : Type} (compactB : B → B) (b : B) : OpTree B → OpTree B
  | .argument _ => .argument (some b)
  | .once => .once
  | .pass c => .pass (setArg compactB b c)
  | .reset _ c => .reset false (setArg compactB b c)
  | .fanout rs c => .fanout (setArgL compactB (compactB b) rs) (setArg compactB b c)
  | .union _ cur => .union (some (compactB b)) (cur.attach.map fun ⟨c, _⟩ => setArg compactB (compactB b) c)
def setArgL {B : Type} (compactB : B → B) (b : B) : List (OpTree B) → List (OpTree B)
  | [] => []
  | c :: cs => setArg compactB b c :: setArgL compactB b cs
end

-- The argument slots reachable through pass-through nodes, left to right.
mutual
def args {B : Type} : OpTree B → List (Option B)
  | .argument s => [s]
  | .once => []
  | .pass c => args c
  | .reset _ c => args c
  | .fanout rs c => argsL rs ++ args c
  | .union _ cur => match cur with | some c => args c | none => []
def argsL {B : Type} : List (OpTree B) → List (Option B)
  | [] => []
  | c :: cs => args c ++ argsL cs
end

mutual
theorem setArg_args {B : Type} (cb : B → B) (b : B) (hidem : ∀ x, cb (cb x) = cb x) :
    ∀ t : OpTree B, ∀ s ∈ args (setArg cb b t), s = some b ∨ s = some (cb b)
  | .argument _ => by simp [setArg, args]
  | .once => by simp [setArg, args]
  | .pass c => by simpa [setArg, args] using setArg_args cb b hidem c
  | .reset _ c => by simpa [setArg, args] using setArg_args cb b hidem c
  | .fanout rs c => by
      intro s hs
      simp only [setArg, args, List.mem_append] at hs
      rcases hs with hs | hs
      · rcases setArgL_args cb (cb b) hidem rs s hs with h | h
        · exact Or.inr h
        · exact Or.inr (by rw [h, hidem])
      · exact setArg_args cb b hidem c s hs
  | .union _ cur => by
      intro s hs
      cases cur with
      | none => simp [setArg, args] at hs
      | some c =>
        simp only [setArg, args, Option.attach_some, Option.map_some] at hs
        rcases setArg_args cb (cb b) hidem c s hs with h | h
        · exact Or.inr h
        · exact Or.inr (by rw [h, hidem])
theorem setArgL_args {B : Type} (cb : B → B) (b : B) (hidem : ∀ x, cb (cb x) = cb x) :
    ∀ ts : List (OpTree B), ∀ s ∈ argsL (setArgL cb b ts), s = some b ∨ s = some (cb b)
  | [] => by simp [setArgL, argsL]
  | t :: ts => by
      intro s hs
      simp only [setArgL, argsL, List.mem_append] at hs
      rcases hs with hs | hs
      · exact setArg_args cb b hidem t s hs
      · exact setArgL_args cb b hidem ts s hs
end

/-- **`set_argument_batch`**: every reachable `Argument` leaf now holds the new batch (the
compacted copy under a fan-out / union), and every emitter-backed node is reset (no stale
rows survive into the new iteration). -/
theorem setArg_resets {B : Type} (cb : B → B) (b : B) (st : Bool) (c : OpTree B) :
    setArg cb b (.reset st c) = .reset false (setArg cb b c) := by rw [setArg]

/-- `inspect_context` (:1722): `None` exactly for the synthetic leaves. -/
def inspectCtx {B : Type} : OpTree B → Bool
  | .argument _ | .once => false
  | _ => true

theorem inspectCtx_spec {B : Type} (t : OpTree B) :
    inspectCtx t = false ↔ (∃ s, t = .argument s) ∨ t = .once := by
  cases t <;> simp [inspectCtx]

/-- `BatchOp::next` (:1771): leaves hand out their batch once (`take`); an op with an
inspect context first checks the timeout and memory budget, failing with their error. -/
def opNext {B : Type} (timeout mem : Except String Unit) (hasCtx : Bool) (dispatch : Option (Except String B)) :
    Option (Except String B) :=
  if hasCtx then
    match timeout with
    | .error e => some (.error e)
    | .ok () => match mem with
      | .error e => some (.error e)
      | .ok () => dispatch
  else dispatch

def leafNext {B : Type} (slot : Option B) : Option (Except String B) × Option B := (slot.map .ok, none)

theorem opNext_spec {B : Type} (d : Option (Except String B)) (e : String) (s : Option B) :
    opNext (.ok ()) (.ok ()) true d = d ∧ opNext (.error e) (.ok ()) true d = some (.error e) ∧
    opNext (.ok ()) (.error e) true d = some (.error e) ∧ (∀ t m, opNext t m false d = d) ∧
    leafNext s = (s.map .ok, none) := by
  refine ⟨rfl, rfl, rfl, fun _ _ => rfl, rfl⟩

end Columnar
