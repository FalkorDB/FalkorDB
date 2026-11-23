import OpsTraverse.Drive
/-
CondTraverse / ExpandInto / scan operator glue and constructors.

| here | there |
| --- | --- |
| `ncolsT`               | `TraversalMatrix::ncols` (cond_traverse.rs) |
| `emptyIter`            | `empty_edge_iter` (cond_traverse.rs) |
| `ctNew`                | `CondTraverseOp::new` (cond_traverse.rs) |
| `buildState`           | `CondTraverseOp::build_state` (cond_traverse.rs; multi-type branch :506-535, #3492) |
| `trimToCap`            | `CondTraverseOp::trim_to_cap` (cond_traverse.rs) |
| `outAlias`             | `CondTraverseOp::out_alias_id` (cond_traverse.rs) |
| `nullPad`              | `CondTraverseOp::null_pad` (cond_traverse.rs) |
| `ctPack`               | one child batch of `CondTraverseOp::next` (cond_traverse.rs) |
| `eiNew`, `eiTrim`      | `ExpandIntoOp::new` (expand_into.rs), `next` |
| `EmitCfg` constructors | `NodeByLabelScanOp::new` (node_by_label_scan.rs), `NodeByIdSeekOp::new` (node_by_id_seek.rs), `NodeByLabelAndIdScanOp::new` (node_by_label_and_id_scan.rs), `PathBuilderOp::new` (path_builder.rs), `AllShortestPathsOp::new` (all_shortest_paths.rs) |

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
  | union (fwds : List Mat)   -- several types kept apart (#3492); forward layers

/-- cond_traverse.rs: the union's column count is the first tensor's (every
relationship tensor is sized to the node capacity), 0 for an (unreachable) empty list. -/
def ncolsT : TM → Nat
  | .bool m => m.ncols
  | .u64 f => f.ncols
  | .union fs => (fs.head?.map Mat.ncols).getD 0

theorem ncolsT_spec (m : Mat) : ncolsT (.bool m) = m.ncols ∧ ncolsT (.u64 m) = m.ncols := ⟨rfl, rfl⟩

theorem ncolsT_union (m : Mat) (rest : List Mat) : ncolsT (.union (m :: rest)) = m.ncols := rfl
theorem ncolsT_union_nil : ncolsT (.union []) = 0 := rfl

/-- When the tensors share a column count (node capacity), the union's `ncols` is every member's. -/
theorem ncolsT_union_all (fs : List Mat) (c : Nat) (hne : fs ≠ []) (h : ∀ f ∈ fs, f.ncols = c) :
    ncolsT (.union fs) = c := by
  cases fs with
  | nil => exact absurd rfl hne
  | cons a t => simp [ncolsT, h a (by simp)]

/-- cond_traverse.rs: an iterator over a fresh 0×0 matrix. -/
def emptyMat : Mat := ⟨0, 0, []⟩
def iterOf (m : Mat) (lo hi : Nat) : List (Nat × Nat) := m.entries.filter fun p => lo ≤ p.1 ∧ p.1 ≤ hi
def emptyIter : List (Nat × Nat) := iterOf emptyMat 0 (2 ^ 64 - 1)

theorem emptyIter_nil : emptyIter = [] := rfl

/-! There is no `attrs_is_static_empty` check: the planner lowers every fixed-length traverse's inline attrs to a
`Filter` above the operator and embeds a stripped pattern (`planner/mod.rs`
`strip_rel_attrs`; proofs/planner_build `E.planMatch_stripped`, `E.firstRel_edge_pred`), so the batched path no
longer needs to inspect them. -/

/-! ## `CondTraverseOp::new` (cond_traverse.rs) -/

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
  /-- `child_ct.relationship_pattern.alias.id` -/
  relAlias : Nat

structure CtCfg where
  dedup : Bool
  dedupSrc : Option Nat
  eligible : Bool
  toCol : Option Nat
  produced : Nat
  /-- the hint the emitter's `pack_ceiling` is sized from (`with_binding(.., pack_hint)`;
  proofs/columnar `Emitter.withBinding`); the hard stop is `record_cap` (`Drive.capDrive`). -/
  emitHint : Option Nat
  /-- `child_ct.emitter.lift_pack_ceiling()` (proofs/columnar `Emitter.liftPackCeiling`). -/
  liftChild : Bool
  /-- `BidirDedup::new([rp.from.alias.id, child_ct.relationship_pattern.alias.id])`: the
  child hop's own bindings, left out of the record identity (`BidirDedup.rec`). -/
  dedupSkip : Option (Nat × Nat)

def ctNew (rp : RelPat) (emitRel : Bool) (siblings : List Nat) (chainBidir : List Bool)
    (child : Option ChildCT) (packHint : Option Nat) : CtCfg :=
  -- cond_traverse.rs `rp.from.alias.is_unnamed()` (no name)
  let anon := rp.fromName.isNone
  let (dedup, src) :=
    if !emitRel && rp.bidir && anon then
      match child with
      | some c => if !c.emitRel && c.bidir then (true, some c.fromAlias) else (false, none)
      | none => (false, none)
    else (false, none)
  -- cond_traverse.rs (#2390: the inline-attr emptiness test is gone)
  let eligible := !emitRel && !rp.bidir && !dedup && siblings.isEmpty &&
    chainBidir.all (!·)
  { dedup, dedupSrc := src, eligible, toCol := if rp.toAlias ≠ rp.fromAlias then some rp.toAlias else none,
    produced := 0, emitHint := packHint, liftChild := dedup,
    dedupSkip := if dedup then child.map (fun c => (rp.fromAlias, c.relAlias)) else none }

/-- The dedup state skips exactly this hop's `from` slot (the child's intermediate node) and
the child's edge slot, and exists only when the dedup is armed. -/
theorem ctNew_dedupSkip (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool)
    (ch : Option ChildCT) (ph : Option Nat) :
    (ctNew rp er sib cb ch ph).dedupSkip =
      if (ctNew rp er sib cb ch ph).dedup then ch.map (fun c => (rp.fromAlias, c.relAlias))
      else none := rfl

/-- #3071: the emitter is sized from the soft `pack_hint`, and the child CT is lifted to
full batches exactly when the dedup is armed (the dedup set now outlives a child batch,
`BidirDedup.dedupStream_eq`, so batch size never changes rows). -/
theorem ctNew_pack (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool) (ch : Option ChildCT)
    (ph : Option Nat) :
    (ctNew rp er sib cb ch ph).emitHint = ph ∧
    (ctNew rp er sib cb ch ph).liftChild = (ctNew rp er sib cb ch ph).dedup := ⟨rfl, rfl⟩

/-- Cross-row bidirectional dedup is armed exactly for an anonymous-intermediate,
non-emitting bidirectional hop over a non-emitting bidirectional child CT, keyed by the
child's source alias; the F·A batched path is taken only for directed, non-emitting,
sibling-free hops whose chain hops are all directed; a self-loop
binds one endpoint column. -/
theorem ctNew_spec (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool) (ch : Option ChildCT)
    (ph : Option Nat) :
    let c := ctNew rp er sib cb ch ph
    (c.dedup = true ↔ er = false ∧ rp.bidir = true ∧ rp.fromName = none ∧
      ∃ x, ch = some x ∧ x.emitRel = false ∧ x.bidir = true) ∧
    (c.dedup = true → c.dedupSrc = ch.map (·.fromAlias)) ∧
    (c.eligible = true → er = false ∧ rp.bidir = false ∧ sib = [] ∧ ∀ b ∈ cb, b = false) ∧
    (c.toCol = none ↔ rp.toAlias = rp.fromAlias) ∧ c.produced = 0 := by
  simp only [ctNew]
  refine ⟨?_, ?_, ?_, ?_, by first | rfl | trivial⟩
  · by_cases h : (!er && rp.bidir && rp.fromName.isNone) = true
    · simp only [h, ite_true]
      simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [Option.isNone_iff_eq_none] at h3
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
sibling-free hops whose fused chain is all directed (cond_traverse.rs). -/
theorem ctNew_eligible_iff (rp : RelPat) (er : Bool) (sib : List Nat) (cb : List Bool)
    (ch : Option ChildCT) (ph : Option Nat) :
    (ctNew rp er sib cb ch ph).eligible = true ↔
      er = false ∧ rp.bidir = false ∧ (ctNew rp er sib cb ch ph).dedup = false ∧ sib = [] ∧
        ∀ b ∈ cb, b = false := by
  simp only [ctNew, Bool.and_eq_true, List.isEmpty_iff, List.all_eq_true,
    Bool.not_eq_eq_eq_not, Bool.not_true]
  constructor
  · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h6⟩; exact ⟨h1, h2, h3, h4, h6⟩
  · rintro ⟨h1, h2, h3, h4, h6⟩; exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h6⟩

/-! ## `build_state` (cond_traverse.rs) -/

structure CtState where
  fwdSrc : List Nat
  fwdDst : List Nat
  revSrc : List Nat
  revDst : List Nat
  edgeTypes : List Nat
  noMatch : Bool
  /-- the single-type `rev_iter` exists (never for a multi-type pattern: `!multi_type`, :552) -/
  hasRev : Bool
  /-- `rp.types.len() > 1`: the per-type `UnionIter` carries the pattern (`CtState::union`) -/
  multi : Bool
  /-- `union.rev` exists: a bidirectional multi-type pattern -/
  unionRev : Bool

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
    noMatch, hasRev := bidir && !(decide (types.length > 1)),
    multi := decide (types.length > 1), unionRev := bidir && decide (types.length > 1) }

/-- An unknown label sets `no_match` (no rows, but the child is still drained), and so does a
builder that found no matrix: `fwdOk`/`revOk` say whether the iterator builders found theirs, which
for a single or untyped pattern fails on an unknown type and for a multi-type pattern only when NO
type exists (see `UnionGlue.fwdOkOf`); the source/destination label sets swap under transposition; no type means every
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

/-- **Multi-type state (#3492).** Several types stream through the union iterators: the plain
reverse iterator is never built (`hasRev = false`), the union carries the reverse half exactly for
a bidirectional pattern; a single-type or untyped pattern keeps the previous state. -/
theorem buildState_multi (resolve : List String → Option (List Nat)) (typeId : String → Option Nat)
    (n : Nat) (fromL toL types : List String) (bidir tr fwdOk revOk : Bool) :
    let s := buildState resolve typeId n fromL toL types bidir tr fwdOk revOk
    (types.length > 1 → s.multi = true ∧ s.hasRev = false ∧ s.unionRev = bidir) ∧
    (types.length ≤ 1 → s.multi = false ∧ s.hasRev = bidir ∧ s.unionRev = false) := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · have : decide (types.length > 1) = true := decide_eq_true h
    simp [buildState, this]
  · have : decide (types.length > 1) = false := decide_eq_false (by omega)
    simp [buildState, this]

/-! ## `trim_to_cap`, `out_alias_id`, `null_pad` -/

/-- cond_traverse.rs: keep the first `cap - produced` active rows. -/
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

/-- cond_traverse.rs -/
def outAlias (chainLastTo : Option Nat) (transposed : Bool) (fromA toA : Nat) : Nat :=
  match chainLastTo with
  | some t => t
  | none => if transposed then fromA else toA

theorem outAlias_spec (c : Option Nat) (tr : Bool) (f t : Nat) :
    (∀ x, c = some x → outAlias c tr f t = x) ∧ outAlias none true f t = f ∧ outAlias none false f t = t := by
  refine ⟨fun x hx => by simp [outAlias, hx], rfl, rfl⟩

/-- cond_traverse.rs: gather the unmatched rows, bind the edge and the out alias to
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

/-! ## `CondTraverseOp::next` (cond_traverse.rs) -/

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

/-! ## `ExpandIntoOp::new` / `next` (expand_into.rs) -/

structure EiCfg where
  synthetic : Bool
  relCol : Option Nat
  produced : Nat
  /-- the soft `pack_hint` the emitter is sized from; the hard stop is `record_cap`. -/
  emitHint : Option Nat

/-- expand_into.rs: a synthetic multi-label check `(a:A:B)` (same alias, labels only on
`to`) binds no relationship column. -/
def eiNew (fromA toA relA : Nat) (fromLabels toLabels : List String) (packHint : Option Nat) :
    EiCfg :=
  let synthetic := fromA == toA && fromLabels.isEmpty && !toLabels.isEmpty
  { synthetic, relCol := if synthetic then none else some relA, produced := 0,
    emitHint := packHint }

theorem eiNew_spec (f t r : Nat) (fl tl : List String) (ph : Option Nat) :
    ((eiNew f t r fl tl ph).synthetic = true ↔ f = t ∧ fl = [] ∧ tl ≠ []) ∧
    ((eiNew f t r fl tl ph).relCol = none ↔ (eiNew f t r fl tl ph).synthetic = true) ∧
    (eiNew f t r fl tl ph).produced = 0 ∧ (eiNew f t r fl tl ph).emitHint = ph := by
  refine ⟨by simp [eiNew, and_assoc], ?_, rfl, rfl⟩
  simp only [eiNew]; split <;> simp_all

/-- expand_into.rs: `set_selection(0..remaining)` keeps the first `remaining` rows. -/
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
