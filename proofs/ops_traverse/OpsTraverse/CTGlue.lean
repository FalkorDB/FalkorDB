import OpsTraverse.Drive
/-
CondTraverse / ExpandInto / scan operator glue and constructors.

| here | there |
| --- | --- |
| `ncolsT`               | `TraversalMatrix::ncols` (cond_traverse.rs:68) |
| `emptyIter`            | `empty_edge_iter` (cond_traverse.rs:210) |
| `ctNew`                | `CondTraverseOp::new` (cond_traverse.rs:238) |
| `buildState`           | `CondTraverseOp::build_state` (cond_traverse.rs:353) |
| `trimToCap`            | `CondTraverseOp::trim_to_cap` (cond_traverse.rs:1030) |
| `outAlias`             | `CondTraverseOp::out_alias_id` (cond_traverse.rs:1053) |
| `nullPad`              | `CondTraverseOp::null_pad` (cond_traverse.rs:1065) |
| `ctPack`               | one child batch of `CondTraverseOp::next` (cond_traverse.rs:1086) |
| `eiNew`, `eiTrim`      | `ExpandIntoOp::new` (expand_into.rs:72), `next` (:269) |
| `EmitCfg` constructors | `NodeByLabelScanOp::new` (node_by_label_scan.rs:56), `NodeByIdSeekOp::new` (node_by_id_seek.rs:32), `NodeByLabelAndIdScanOp::new` (node_by_label_and_id_scan.rs:34), `PathBuilderOp::new` (path_builder.rs:44), `AllShortestPathsOp::new` (all_shortest_paths.rs:59) |

GraphBLAS matrices are their entry lists (`(row, col)` pairs) with a declared shape.
-/
namespace OpsTraverse.CTGlue

/-! ## Matrix glue -/

structure Mat where
  nrows : Nat
  ncols : Nat
  entries : List (Nat × Nat)

inductive TM where
  | bool (m : Mat)
  | u64 (fwd : Mat)   -- a tensor; only its forward layer is read

/-- cond_traverse.rs:68-73 -/
def ncolsT : TM → Nat
  | .bool m => m.ncols
  | .u64 f => f.ncols

theorem ncolsT_spec (m : Mat) : ncolsT (.bool m) = m.ncols ∧ ncolsT (.u64 m) = m.ncols := ⟨rfl, rfl⟩

/-- cond_traverse.rs:210-212: an iterator over a fresh 0×0 matrix. -/
def emptyMat : Mat := ⟨0, 0, []⟩
def iterOf (m : Mat) (lo hi : Nat) : List (Nat × Nat) := m.entries.filter fun p => lo ≤ p.1 ∧ p.1 ≤ hi
def emptyIter : List (Nat × Nat) := iterOf emptyMat 0 (2 ^ 64 - 1)

theorem emptyIter_nil : emptyIter = [] := rfl

/-! `attrs_is_static_empty` (formerly cond_traverse.rs:240) was removed by #2390
(d2c42e032): the planner now lowers every fixed-length traverse's inline attrs to a
`Filter` above the operator and embeds a stripped pattern (`planner/mod.rs:644`
`strip_rel_attrs`; proofs/planner_build `E.planMatch_stripped`, `E.firstRel_edge_pred`), so the batched path no
longer needs to inspect them. Historical model: `pre2390_eligible`. -/

/-! ## `CondTraverseOp::new` (cond_traverse.rs:238) -/

structure RelPat where
  fromAlias : Nat
  fromName : Option String
  toAlias : Nat
  relAlias : Nat
  bidir : Bool

structure ChildCT where
  emitRel : Bool
  bidir : Bool
  fromAlias : Nat

structure CtCfg where
  dedup : Bool
  dedupSrc : Option Nat
  eligible : Bool
  toCol : Option Nat
  produced : Nat

def ctNew (rp : RelPat) (emitRel : Bool) (siblings : List Nat) (chainBidir : List Bool)
    (child : Option ChildCT) : CtCfg :=
  let anon := rp.fromName.any (·.startsWith "_anon")
  let (dedup, src) :=
    if !emitRel && rp.bidir && anon then
      match child with
      | some c => if !c.emitRel && c.bidir then (true, some c.fromAlias) else (false, none)
      | none => (false, none)
    else (false, none)
  -- cond_traverse.rs:304-308 (#2390: the inline-attr emptiness test is gone)
  let eligible := !emitRel && !rp.bidir && !dedup && siblings.isEmpty &&
    chainBidir.all (!·)
  { dedup, dedupSrc := src, eligible, toCol := if rp.toAlias ≠ rp.fromAlias then some rp.toAlias else none,
    produced := 0 }

/-- Cross-row bidirectional dedup is armed exactly for an anonymous-intermediate,
non-emitting bidirectional hop over a non-emitting bidirectional child CT, keyed by the
child's source alias; the F·A batched path is taken only for directed, non-emitting,
sibling-free hops whose chain hops are all directed; a self-loop
binds one endpoint column. -/
theorem ctNew_spec (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool) (ch : Option ChildCT) :
    let c := ctNew rp er sib cb ch
    (c.dedup = true ↔ er = false ∧ rp.bidir = true ∧ rp.fromName.any (·.startsWith "_anon") = true ∧
      ∃ x, ch = some x ∧ x.emitRel = false ∧ x.bidir = true) ∧
    (c.dedup = true → c.dedupSrc = ch.map (·.fromAlias)) ∧
    (c.eligible = true → er = false ∧ rp.bidir = false ∧ sib = [] ∧ ∀ b ∈ cb, b = false) ∧
    (c.toCol = none ↔ rp.toAlias = rp.fromAlias) ∧ c.produced = 0 := by
  simp only [ctNew]
  refine ⟨?_, ?_, ?_, ?_, by first | rfl | trivial⟩
  · by_cases h : (!er && rp.bidir && rp.fromName.any (·.startsWith "_anon")) = true
    · simp only [h, ite_true]
      simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      cases ch with
      | none => simp
      | some x => by_cases hx : (!x.emitRel && x.bidir) = true
                  · simp only [hx, ite_true]; simp at hx; simp [h1, h2, h3, hx]
                  · simp only [hx, Bool.false_eq_true, ite_false]; simp at hx
                    constructor
                    · intro h; cases h
                    · rintro ⟨_, _, _, y, hy, h4, h5⟩
                      cases hy; simp [hx h4] at h5
    · simp only [h, Bool.false_eq_true, ite_false, false_iff, not_and]
      intro h1 h2 h3; simp [h1, h2, h3] at h
  · intro hd
    split at hd
    · split at hd
      · split at hd
        · simp_all
        · simp at hd
      · simp at hd
    · simp at hd
  · intro he
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_iff, List.all_eq_true] at he
    obtain ⟨⟨⟨⟨h1, h2⟩, _⟩, h4⟩, h6⟩ := he
    exact ⟨h1, h2, h4, fun b hb => by simpa using h6 b hb⟩
  · by_cases h : rp.toAlias = rp.fromAlias <;> simp [h]

/-- The F·A batched path is taken exactly for non-emitting, directed, dedup-free,
sibling-free hops whose fused chain is all directed (cond_traverse.rs:304-308). -/
theorem ctNew_eligible_iff (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool)
    (ch : Option ChildCT) :
    (ctNew rp er sib cb ch).eligible = true ↔
      er = false ∧ rp.bidir = false ∧ (ctNew rp er sib cb ch).dedup = false ∧ sib = [] ∧
        ∀ b ∈ cb, b = false := by
  simp only [ctNew, Bool.and_eq_true, List.isEmpty_iff, List.all_eq_true,
    Bool.not_eq_eq_eq_not, Bool.not_true]
  constructor
  · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h6⟩; exact ⟨h1, h2, h3, h4, h6⟩
  · rintro ⟨h1, h2, h3, h4, h6⟩; exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h6⟩

/-- Historical (`pre2390`, a9377c636 cond_traverse.rs:302-310): a single hop also needed
the edge's and both endpoints' inline attrs to be `{}`. Fixed shape by #2390 (d2c42e032). -/
def pre2390_eligible (er bidir dedup : Bool) (sib : List Nat) (cb : List Bool)
    (attrsEmpty fromEmpty toEmpty : Bool) : Bool :=
  !er && !bidir && !dedup && sib.isEmpty && (!cb.isEmpty || (attrsEmpty && fromEmpty && toEmpty)) &&
    cb.all (!·)

/-- #2390 only widens the batched path: whatever was eligible before still is, and a hop with
non-empty endpoint attrs is now eligible too (the planner's Filter enforces them). -/
theorem pre2390_eligible_le (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool)
    (ch : Option ChildCT) (a f t : Bool)
    (h : pre2390_eligible er rp.bidir (ctNew rp er sib cb ch).dedup sib cb a f t = true) :
    (ctNew rp er sib cb ch).eligible = true := by
  simp only [pre2390_eligible, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, _⟩, h6⟩ := h
  simp only [ctNew, Bool.and_eq_true] at h3 ⊢
  exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h6⟩

theorem pre2390_widened :
    pre2390_eligible false false false [] [] true false true = false ∧
    (ctNew ⟨0, none, 1, 2, false⟩ false [] [] none).eligible = true := by decide

/-! ## `build_state` (cond_traverse.rs:353) -/

structure CtState where
  fwdSrc : List Nat
  fwdDst : List Nat
  revSrc : List Nat
  revDst : List Nat
  edgeTypes : List Nat
  noMatch : Bool
  hasRev : Bool

/-- `resolve` = `g.resolve_label_ids` (`None` when a label is unknown); `typeId` =
`get_type_id`; `fwdIter`/`revIter` = whether `build_unrestricted_iter` found the matrix. -/
def buildState (resolve : List String → Option (List Nat)) (typeId : String → Option Nat)
    (ntensors : Nat) (fromL toL : List String) (types : List String) (bidir transposed : Bool)
    (fwdOk revOk : Bool) : CtState :=
  let (fs, fd) := if transposed then (toL, fromL) else (fromL, toL)
  let fsi := resolve fs
  let fdi := resolve fd
  let (rsi, rdi) := if bidir then
      (if transposed then (resolve fromL, resolve toL) else (resolve toL, resolve fromL))
    else (some [], some [])
  let noMatch := fsi.isNone || fdi.isNone || rsi.isNone || rdi.isNone || !fwdOk || (bidir && !revOk)
  { fwdSrc := fsi.getD [], fwdDst := fdi.getD [], revSrc := rsi.getD [], revDst := rdi.getD [],
    edgeTypes := if types.isEmpty then List.range ntensors else types.filterMap typeId,
    noMatch, hasRev := bidir }

/-- An unknown label or relationship type sets `no_match` (no rows, but the child is still
drained); the source/destination label sets swap under transposition; no type means every
relationship tensor. -/
theorem buildState_spec (resolve : List String → Option (List Nat)) (typeId : String → Option Nat)
    (n : Nat) (fromL toL types : List String) (bidir tr fwdOk revOk : Bool) :
    let s := buildState resolve typeId n fromL toL types bidir tr fwdOk revOk
    ((resolve fromL).isNone → s.noMatch = true) ∧ ((resolve toL).isNone → s.noMatch = true) ∧
    (fwdOk = false → s.noMatch = true) ∧
    (tr = false → s.fwdSrc = (resolve fromL).getD [] ∧ s.fwdDst = (resolve toL).getD []) ∧
    (tr = true → s.fwdSrc = (resolve toL).getD [] ∧ s.fwdDst = (resolve fromL).getD []) ∧
    (types = [] → s.edgeTypes = List.range n) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => by simp [buildState, h], fun h => by simp [buildState, h],
    fun h => by simp [buildState, h], fun h => by simp [buildState, h]⟩
  · cases tr <;> simp [buildState, h]
  · cases tr <;> simp [buildState, h]

/-! ## `trim_to_cap`, `out_alias_id`, `null_pad` -/

/-- cond_traverse.rs:1030-1050: keep the first `cap - produced` active rows. -/
def trimToCap {R : Type} (cap : Option Nat) (produced : Nat) (rows : List R) : List R × Nat :=
  match cap with
  | none => (rows, produced)
  | some c =>
    let remaining := c - produced
    if rows.length > remaining then
      let t := rows.take remaining
      (t, produced + t.length)
    else (rows, produced + rows.length)

theorem trimToCap_spec {R : Type} (cap : Nat) (p : Nat) (rows : List R) :
    (trimToCap (some cap) p rows).1 = rows.take (cap - p) ∧
    (trimToCap (some cap) p rows).2 = p + (rows.take (cap - p)).length ∧
    trimToCap none p rows = (rows, p) := by
  unfold trimToCap
  by_cases h : rows.length > cap - p
  · simp [h]
  · simp only [h, ite_false]
    have : rows.take (cap - p) = rows := List.take_of_length_le (by omega)
    simp [this]

/-- cond_traverse.rs:1053-1062 -/
def outAlias (chainLastTo : Option Nat) (transposed : Bool) (fromA toA : Nat) : Nat :=
  match chainLastTo with
  | some t => t
  | none => if transposed then fromA else toA

theorem outAlias_spec (c : Option Nat) (tr : Bool) (f t : Nat) :
    (∀ x, c = some x → outAlias c tr f t = x) ∧ outAlias none true f t = f ∧ outAlias none false f t = t := by
  refine ⟨fun x hx => by simp [outAlias, hx], rfl, rfl⟩

/-- cond_traverse.rs:1065-1082: gather the unmatched rows, bind the edge and the out alias to
`Null` (`none`). -/
def nullPad {V : Type} (rows : List (Nat → Option V)) (unmatched : List Nat) (relA outA : Nat)
    (nul : V) : List (Nat → Option V) :=
  unmatched.filterMap fun i => rows[i]?.map fun r => fun j =>
    if j = outA then some nul else if j = relA then some nul else r j

theorem nullPad_spec {V : Type} (rows : List (Nat → Option V)) (un : List Nat) (relA outA : Nat) (nul : V)
    (hin : ∀ i ∈ un, i < rows.length) :
    (nullPad rows un relA outA nul).length = un.length ∧
    ∀ k (hk : k < un.length) (hk' : k < (nullPad rows un relA outA nul).length),
      let r := (nullPad rows un relA outA nul)[k]
      r relA = some nul ∧ r outA = some nul ∧
      ∀ j, j ≠ relA → j ≠ outA → r j = (rows[un[k]]'(hin _ (List.getElem_mem hk))) j := by
  induction un with
  | nil => simp [nullPad]
  | cons i is ih =>
    have hi : i < rows.length := hin i (by simp)
    obtain ⟨hl, hg⟩ := ih (fun j hj => hin j (List.mem_cons_of_mem _ hj))
    have e : nullPad rows (i :: is) relA outA nul = (fun j =>
        if j = outA then some nul else if j = relA then some nul else rows[i] j) :: nullPad rows is relA outA nul := by
      simp [nullPad, List.getElem?_eq_getElem hi]
    rw [e]
    refine ⟨by simp [hl], fun k hk hk' => ?_⟩
    cases k with
    | zero =>
      simp only [List.getElem_cons_zero]
      refine ⟨by by_cases h : relA = outA <;> simp [h], by simp, fun j h1 h2 => by simp [h1, h2]⟩
    | succ k =>
      simp only [List.getElem_cons_succ]
      exact hg k (by simpa using hk) (by simpa using hk')

/-! ## `CondTraverseOp::next` (cond_traverse.rs:1086) -/

/-- Output batches for one child batch: the batched F·A path's batches when it handles the
batch, else the emitter's packed batches followed (for OPTIONAL) by one null-padded batch of the
rows that expanded to nothing. -/
def ctPack {C R : Type} (batched : C → Option (List (List R))) (emitted : C → List (List R))
    (padded : C → List R) (optional : Bool) (c : C) : List (List R) :=
  match batched c with
  | some bs => bs
  | none => emitted c ++ (if optional && !(padded c).isEmpty then [padded c] else [])

/-- **CondTraverse stream**: the rows emitted are the first `cap` rows of, per child batch in
order, its expansion (and, for OPTIONAL on the per-row path, its null-padded unmatched rows). -/
theorem ctNext_stream {C R : Type} (batched : C → Option (List (List R))) (emitted : C → List (List R))
    (padded : C → List R) (opt : Bool) (cap : Nat) (cs : List C) :
    (Drive.capDrive (ctPack batched emitted padded opt) cap 0 [] cs).flatten =
      (cs.flatMap (ctPack batched emitted padded opt)).flatten.take cap := by
  rw [Drive.capDrive_flatten]; simp

/-! ## `ExpandIntoOp::new` / `next` (expand_into.rs:72, :269) -/

structure EiCfg where
  synthetic : Bool
  relCol : Option Nat
  produced : Nat

/-- expand_into.rs:84-97: a synthetic multi-label check `(a:A:B)` (same alias, labels only on
`to`) binds no relationship column. -/
def eiNew (fromA toA relA : Nat) (fromLabels toLabels : List String) : EiCfg :=
  let synthetic := fromA == toA && fromLabels.isEmpty && !toLabels.isEmpty
  { synthetic, relCol := if synthetic then none else some relA, produced := 0 }

theorem eiNew_spec (f t r : Nat) (fl tl : List String) :
    ((eiNew f t r fl tl).synthetic = true ↔ f = t ∧ fl = [] ∧ tl ≠ []) ∧
    ((eiNew f t r fl tl).relCol = none ↔ (eiNew f t r fl tl).synthetic = true) ∧
    (eiNew f t r fl tl).produced = 0 := by
  refine ⟨by simp [eiNew, and_assoc], ?_, rfl⟩
  simp only [eiNew]; split <;> simp_all

/-- expand_into.rs:277-285: `set_selection(0..remaining)` keeps the first `remaining` rows. -/
theorem eiNext_stream {C R : Type} (pack : C → List (List R)) (cap : Nat) (cs : List C) :
    (Drive.capDrive pack cap 0 [] cs).flatten = (cs.flatMap pack).flatten.take cap := by
  rw [Drive.capDrive_flatten]; simp

/-! ## Scan / builder constructors -/

/-- The emitter configuration a constructor installs: bound column(s) and pack cap. -/
structure EmitCfg where
  col : Option Nat
  cap : Option Nat

def labelScanNew (alias : Nat) (cap : Option Nat) : EmitCfg := ⟨some alias, cap⟩
def idSeekNew (alias : Nat) : EmitCfg := ⟨some alias, none⟩
def labelIdScanNew (alias : Nat) (cap : Option Nat) : EmitCfg := ⟨some alias, cap⟩
def aspNew (pathAlias : Nat) : EmitCfg := ⟨some pathAlias, none⟩

structure PbSt (C P : Type) where
  child : C
  paths : P

def pathBuilderNew {C P : Type} (child : C) (paths : P) : PbSt C P := ⟨child, paths⟩

theorem ctors_spec (a : Nat) (cap : Option Nat) :
    labelScanNew a cap = ⟨some a, cap⟩ ∧ idSeekNew a = ⟨some a, none⟩ ∧
    labelIdScanNew a cap = ⟨some a, cap⟩ ∧ aspNew a = ⟨some a, none⟩ := ⟨rfl, rfl, rfl, rfl⟩

theorem pathBuilderNew_spec {C P : Type} (c : C) (p : P) :
    (pathBuilderNew c p).child = c ∧ (pathBuilderNew c p).paths = p := ⟨rfl, rfl⟩

/-- `AllShortestPathsOp::next` and the scans' `next`: the shared uncapped loop. -/
theorem uncapped_stream {C R : Type} (pack : C → List (List R)) (cs : List C) :
    (Drive.drive pack [] cs).flatten = (cs.flatMap pack).flatten := by
  rw [Drive.drive_flatten]; simp

end OpsTraverse.CTGlue
