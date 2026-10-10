/-
Filter, IncludePending and ProcedureCall (graph/src/runtime/ops/filter.rs,
include_pending.rs, procedure_call.rs).

| here | there |
| --- | --- |
| `FilterSt.new`, `Verdict`, `filterEval`, `filterNext` | `FilterOp::new` (filter.rs:39), `eval` (:57), `next` (:121) |
| `IpSt.new`, `captureArg`, `initPending`, `pendingRows`, `childRows` | `IncludePendingOp::new` (include_pending.rs:49), `capture_argument` (:72), `init_pending_state` (:83), `emit_pending_batch` (:110), `next` (:161) |
| `pcNew`, `pcReset`, `remap`, `pcRows`, `nextInput` | `ProcedureCallOp::new` (procedure_call.rs:59), `reset_state` (:115), `emit_pending_rows` (:125), `remap_procedure_batch` (:153), `next_input_row` (:192), `next` (:228) |
-/
namespace FalkorAggOps.StreamOps

/-! ## Filter -/

structure FilterSt (C T : Type) where
  child : C
  tree : T

def FilterSt.new {C T : Type} (c : C) (t : T) : FilterSt C T := ⟨c, t⟩
theorem filterNew_spec {C T : Type} (c : C) (t : T) : (FilterSt.new c t).child = c ∧ (FilterSt.new c t).tree = t :=
  ⟨rfl, rfl⟩

/-- The evaluated predicate column: a boolean mask with a null bitmap, or per-row values. -/
inductive PV where
  | t | f | null | other (name : String)

inductive Verdict where
  | bools (mask nulls : List Bool)
  | values (vs : List PV)

/-- filter.rs:57-117 over the active rows: keep `true` rows, drop `false`/`null`, error on the
first non-boolean value; `none` = nothing passes (the caller pulls the next batch). -/
def filterEval (rows : List Nat) (v : Verdict) : Except String (Option (List Nat)) :=
  if rows.isEmpty then .ok none else
  let passing : Except String (List Nat) := match v with
    | .bools mask nulls =>
      .ok ((rows.zipIdx).filterMap fun (r, i) => if mask.getD i false && !nulls.getD i false then some r else none)
    | .values vs =>
      (rows.zipIdx).foldlM (fun acc (r, i) => match vs.getD i .null with
        | .t => .ok (acc ++ [r])
        | .f | .null => .ok acc
        | .other n => .error s!"Type mismatch: expected Boolean but was {n}") []
  passing.map fun p => if p.isEmpty then none else some p

theorem filterEval_bools (rows : List Nat) (mask nulls : List Bool) (h : rows ≠ []) :
    filterEval rows (.bools mask nulls) =
      .ok (let p := (rows.zipIdx).filterMap fun (r, i) => if mask.getD i false && !nulls.getD i false then some r else none
           if p.isEmpty then none else some p) := by
  cases rows with
  | nil => exact absurd rfl h
  | cons r rs => simp [filterEval, Except.map]

/-- A row passes iff its verdict is `true` and not null (the boolean lane), and the passing
rows keep their order. -/
theorem filter_mem (rows : List Nat) (mask nulls : List Bool) (r : Nat) :
    r ∈ (rows.zipIdx).filterMap (fun (p : Nat × Nat) => if mask.getD p.2 false && !nulls.getD p.2 false then some p.1 else none) ↔
      ∃ i, rows[i]? = some r ∧ mask.getD i false = true ∧ nulls.getD i false = false := by
  simp only [List.mem_filterMap, Option.ite_none_right_eq_some, Option.some.injEq, Prod.exists,
    List.mk_mem_zipIdx_iff_getElem?, Bool.and_eq_true, Bool.not_eq_true']
  constructor
  · rintro ⟨a, i, h1, ⟨h2, h3⟩, rfl⟩; exact ⟨i, h1, h2, h3⟩
  · rintro ⟨i, h1, h2, h3⟩; exact ⟨r, i, h1, ⟨h2, h3⟩, rfl⟩

theorem filterEval_empty (v : Verdict) : filterEval [] v = .ok none := rfl

/-- A non-boolean value on the per-row lane is a type error (C: same message). -/
theorem filterEval_type_error (r : Nat) (n : String) :
    filterEval [r] (.values [.other n]) = .error s!"Type mismatch: expected Boolean but was {n}" := by
  simp [filterEval, List.zipIdx, Except.map, List.foldlM, bind, Except.bind]

/-- filter.rs:121-134: skip batches where nothing passes; errors propagate. -/
def filterNext {B : Type} (eval : B → Except String (Option B)) : List (Except String B) → Option (Except String B) × List (Except String B)
  | [] => (none, [])
  | .error e :: rest => (some (.error e), rest)
  | .ok b :: rest => match eval b with
    | .ok (some r) => (some (.ok r), rest)
    | .ok none => filterNext eval rest
    | .error e => (some (.error e), rest)

theorem filterNext_skips {B : Type} (eval : B → Except String (Option B)) (b : B) (rest : List (Except String B))
    (h : eval b = .ok none) : filterNext eval (.ok b :: rest) = filterNext eval rest := by
  simp [filterNext, h]

/-! ## IncludePending -/

structure IpSt (N R : Type) where
  alias : Nat
  pendingExtra : Option (List N)
  deleted : Option (List N)
  removed : Option (List N)
  childExhausted : Bool
  argRows : List R
  cur : Option N
  argIdx : Nat

def IpSt.new {N R : Type} (alias : Nat) : IpSt N R := ⟨alias, none, none, none, false, [], none, 0⟩

theorem ipNew_spec {N R : Type} (a : Nat) :
    (IpSt.new a : IpSt N R).deleted = none ∧ (IpSt.new a : IpSt N R).argRows = [] ∧
    (IpSt.new a : IpSt N R).childExhausted = false ∧ (IpSt.new a : IpSt N R).cur = none := ⟨rfl, rfl, rfl, rfl⟩

/-- include_pending.rs:72-81: snapshot the argument batch's active rows. -/
def captureArg {N R : Type} (s : IpSt N R) (rows : List R) : IpSt N R := { s with argRows := rows }

theorem captureArg_spec {N R : Type} (s : IpSt N R) (rows : List R) : (captureArg s rows).argRows = rows := rfl

/-- include_pending.rs:83-108: deleted / label-removed snapshots; pending nodes carrying ALL
the pattern's labels — none when some label does not exist yet. -/
def initPending {N R : Type} [DecidableEq N] (s : IpSt N R) (labels : List String) (labelId : String → Option Nat)
    (deleted : List N) (removedFor : List Nat → List N) (pendingWith : List Nat → List N) : IpSt N R :=
  let ids := labels.filterMap labelId
  { s with deleted := some deleted, removed := some (removedFor ids),
           pendingExtra := some (if ids.length = labels.length then pendingWith ids else []) }

theorem initPending_spec {N R : Type} [DecidableEq N] (s : IpSt N R) (ls : List String) (lid : String → Option Nat)
    (d : List N) (rf : List Nat → List N) (pw : List Nat → List N) (l : String) (hl : l ∈ ls) (hu : lid l = none) :
    (initPending s ls lid d rf pw).pendingExtra = some [] := by
  simp only [initPending]
  have : (ls.filterMap lid).length < ls.length := by
    induction ls with
    | nil => simp at hl
    | cons x xs ih =>
      rcases List.mem_cons.mp hl with rfl | hx
      · simp only [List.filterMap_cons, hu, List.length_cons]
        have := List.length_filterMap_le lid xs; omega
      · cases hx' : lid x
        · simp only [List.filterMap_cons, hx', List.length_cons]
          have := List.length_filterMap_le lid xs; omega
        · simp only [List.filterMap_cons, hx', List.length_cons]; have := ih hx; omega
  simp [Nat.ne_of_lt this]

/-- include_pending.rs:110-159: each pending node crossed with every argument row (one empty row
when there is no argument), the node bound at the alias; `BATCH_SIZE` chunks, resuming. -/
def pendingRows {N R : Type} (bind : R → N → R) (empty : R) (args : List R) (nodes : List N) : List R :=
  nodes.flatMap fun n => (if args.isEmpty then [empty] else args).map fun r => bind r n

/-- include_pending.rs:161-209: child rows whose node is deleted or lost the label this query are
dropped; then the pending nodes follow. -/
def childRows {N R : Type} [DecidableEq N] (nodeOf : R → Option N) (deleted removed : List N) (rows : List R) : List R :=
  rows.filter fun r => match nodeOf r with
    | some n => !(deleted.contains n || removed.contains n)
    | none => false

theorem includePending_spec {N R : Type} [DecidableEq N] (nodeOf : R → Option N) (bind : R → N → R) (empty : R)
    (deleted removed : List N) (rows args : List R) (nodes : List N) (r : R) :
    (r ∈ childRows nodeOf deleted removed rows ↔
      r ∈ rows ∧ ∃ n, nodeOf r = some n ∧ n ∉ deleted ∧ n ∉ removed) ∧
    (pendingRows bind empty args nodes).length = nodes.length * (max args.length 1) := by
  constructor
  · simp only [childRows, List.mem_filter]
    constructor
    · rintro ⟨h1, h2⟩
      cases hn : nodeOf r with
      | none => rw [hn] at h2; cases h2
      | some n => rw [hn] at h2; simp at h2; exact ⟨h1, n, rfl, h2.1, h2.2⟩
    · rintro ⟨h1, n, hn, hd, hr⟩; refine ⟨h1, ?_⟩; rw [hn]; simp [hd, hr]
  · simp only [pendingRows, List.length_flatMap, List.length_map]
    have hs : ∀ (l : List N) (c : Nat), (l.map fun _ => c).sum = l.length * c := by
      intro l c; induction l with
      | nil => simp
      | cons x xs ih => simp [ih, Nat.succ_mul]; omega
    rw [hs]
    cases args <;> simp

/-! ## ProcedureCall -/

/-- procedure_call.rs:59-113: refuse a write procedure in a read-only query; each yielded name
reads the procedure column of the same name (else its yield position); the yield mask has a bit
per source position below 32. -/
structure PcCfg where
  outVars : List Nat
  srcPos : List Nat
  mask : Nat

def srcPosOf (schema : List String) (yields : List (Nat × Option String)) : List Nat :=
  (yields.zipIdx).map fun ((_, name), pos) => (name.bind fun n => schema.idxOf? n).getD pos

def maskOf (pos : List Nat) : Nat := pos.foldl (fun m p => if p < 32 then m ||| (1 <<< p) else m) 0

def pcNew (rtWrite fnWrite : Bool) (schema : List String) (yields : List (Nat × Option String)) : Except String PcCfg :=
  if !rtWrite && fnWrite then .error "graph.RO_QUERY is to be executed only on read-only queries"
  else .ok ⟨yields.map (·.1), srcPosOf schema yields, maskOf (srcPosOf schema yields)⟩

theorem pcNew_spec (w fw : Bool) (schema : List String) (ys : List (Nat × Option String)) :
    (w = false → fw = true → pcNew w fw schema ys = .error "graph.RO_QUERY is to be executed only on read-only queries") ∧
    (∀ c, pcNew w fw schema ys = .ok c → c.outVars = ys.map (·.1) ∧ c.srcPos.length = ys.length) := by
  refine ⟨fun h1 h2 => by simp [pcNew, h1, h2], fun c hc => ?_⟩
  unfold pcNew at hc
  split at hc
  · cases hc
  · cases hc; simp [srcPosOf]

theorem srcPos_named (schema : List String) (ys : List (Nat × Option String)) (i : Nat) (hi : i < ys.length)
    (n : String) (k : Nat) (hn : (ys[i]).2 = some n) (hk : schema.idxOf? n = some k) :
    (srcPosOf schema ys)[i]'(by simp [srcPosOf]; exact hi) = k := by
  simp [srcPosOf, hn, hk]

/-- procedure_call.rs:115-123 -/
structure PcSt where
  pendingRow : Option Nat
  pendingIdx : Nat
  childPos : Nat
  childExhausted : Bool
  hasChildBatch : Bool

def pcReset (_ : PcSt) : PcSt := ⟨none, 0, 0, false, false⟩
theorem pcReset_spec (s : PcSt) : pcReset s = ⟨none, 0, 0, false, false⟩ := rfl

/-- procedure_call.rs:153-190: each output var gets the procedure column at its source position
(a missing/unbound column becomes all-`Null`); the selection is kept. -/
def remap {V : Type} (cols : List (Option (List V))) (len : Nat) (nul : V) (outVars srcPos : List Nat) :
    List (Nat × List V) :=
  (outVars.zip srcPos).map fun (v, p) => (v, ((cols[p]?).bind id).getD (List.replicate len nul))

theorem remap_spec {V : Type} (cols : List (Option (List V))) (len : Nat) (nul : V) (ov sp : List Nat)
    (k : Nat) (hk : k < ov.length) (hk' : k < sp.length) :
    (remap cols len nul ov sp)[k]'(by simp [remap]; omega) =
      (ov[k], ((cols[sp[k]]?).bind id).getD (List.replicate len nul)) := by
  simp [remap]

/-- procedure_call.rs:125-151 + :228-300: every input row (in order) is followed by one output
row per procedure result row, merged over the input row. -/
def pcRows {R P O : Type} (call : R → Except String (List P)) (merge : R → P → O) : List R → Except String (List O)
  | [] => .ok []
  | r :: rs => do
    let ps ← call r
    let rest ← pcRows call merge rs
    pure (ps.map (merge r) ++ rest)

theorem pcRows_ok {R P O : Type} (call : R → Except String (List P)) (merge : R → P → O) (f : R → List P)
    (hf : ∀ r, call r = .ok (f r)) (rs : List R) :
    pcRows call merge rs = .ok (rs.flatMap fun r => (f r).map (merge r)) := by
  induction rs with
  | nil => rfl
  | cons r rs ih => simp [pcRows, hf, ih, bind, Except.bind, pure, Except.pure]

/-- procedure_call.rs:192-226: walk the active rows of the current child batch, then pull the
next child batch; a child error is returned; exhaustion is sticky. -/
def nextInput {B : Type} (active : B → List Nat) : List (Except String B) → List (Except String Nat)
  | [] => []
  | .error e :: _ => [.error e]
  | .ok b :: bs => (active b).map .ok ++ nextInput active bs

theorem nextInput_ok {B : Type} (active : B → List Nat) (bs : List B) :
    nextInput active (bs.map .ok) = (bs.flatMap active).map .ok := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [nextInput, ih]

end FalkorAggOps.StreamOps
