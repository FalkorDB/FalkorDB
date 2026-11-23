import OpsTraverse.Drive
/-
CondTraverse / ExpandInto / scan operator glue and constructors.

| here | there |
| --- | --- |
| `ncolsT`               | `TraversalMatrix::ncols` (cond_traverse.rs; `BoolUnion` arm since #3639) |
| `emptyIter`            | `empty_edge_iter` (cond_traverse.rs) |
| `ctNew`                | `CondTraverseOp::new` (cond_traverse.rs; #3639: `node_unique`, chain-pair siblings, no bidir dedup) |
| `buildState`           | `CondTraverseOp::build_state` (cond_traverse.rs; multi-type branch, #3492; `multi_edge`, #3639) |
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
  | boolUnion (ms : List Mat)   -- `BoolUnion` (#3639): per-type boolean adjacencies, kept apart

/-- cond_traverse.rs: the union's column count is the first tensor's (every
relationship tensor is sized to the node capacity), 0 for an (unreachable) empty list. -/
def ncolsT : TM → Nat
  | .bool m => m.ncols
  | .u64 f => f.ncols
  | .union fs => (fs.head?.map Mat.ncols).getD 0
  | .boolUnion ms => (ms.head?.map Mat.ncols).getD 0

theorem ncolsT_spec (m : Mat) : ncolsT (.bool m) = m.ncols ∧ ncolsT (.u64 m) = m.ncols := ⟨rfl, rfl⟩

theorem ncolsT_union (m : Mat) (rest : List Mat) : ncolsT (.union (m :: rest)) = m.ncols := rfl
theorem ncolsT_union_nil : ncolsT (.union []) = 0 := rfl

/-- `BoolUnion` (#3639): the first member's column count, 0 when empty. -/
theorem ncolsT_boolUnion (m : Mat) (rest : List Mat) :
    ncolsT (.boolUnion (m :: rest)) = m.ncols ∧ ncolsT (.boolUnion []) = 0 := ⟨rfl, rfl⟩

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

/-- A sibling CondTraverse compared by its endpoints (`SiblingEnds`). -/
structure SibEnds where
  fromAlias : Nat
  toAlias : Nat
  transposed : Bool

structure CtCfg where
  nodeUnique : Option (List SibEnds)
  eligible : Bool
  toCol : Option Nat
  produced : Nat

/-- `CondTraverseOp::new` (#3639: the bidirectional dedup is gone). `chain` lists the fused hops'
`(alias, bidirectional)`; `ncs` is what `node_comparable_siblings` returns for this pattern. -/
def ctNew (rp : RelPat) (emitRel bind optional : Bool) (siblings : List Nat)
    (chain : List (Nat × Bool)) (ncs : Option (List SibEnds)) : CtCfg :=
  let nodeUnique :=
    if !emitRel && !bind && !optional && chain.isEmpty && !rp.bidir then ncs else none
  -- cond_traverse.rs (#2390: the inline-attr emptiness test is gone)
  let eligible := !emitRel && !rp.bidir &&
    (siblings.isEmpty || nodeUnique.isSome ||
      (chain.length == 1 && siblings.all fun i => i == rp.relAlias || chain.any (·.1 == i))) &&
    chain.all (fun h => !h.2)
  { nodeUnique, eligible, toCol := if rp.toAlias ≠ rp.fromAlias then some rp.toAlias else none,
    produced := 0 }

/-- Siblings are compared by node only for a plain, non-emitting, unbound, non-optional,
unchained, directed hop, and then exactly as `node_comparable_siblings` says; the F·A batched
path is taken only for directed, non-emitting hops whose chain hops are all directed and whose
siblings are node-compared or, on a two-hop chain, only the chain's own edges; a self-loop binds
one endpoint column. -/
theorem ctNew_spec (rp : RelPat) (er bd op : Bool) (sib : List Nat) (ch : List (Nat × Bool))
    (ncs : Option (List SibEnds)) :
    let c := ctNew rp er bd op sib ch ncs
    (c.nodeUnique.isSome → er = false ∧ bd = false ∧ op = false ∧ ch = [] ∧ rp.bidir = false ∧
      c.nodeUnique = ncs) ∧
    (c.eligible = true → er = false ∧ rp.bidir = false ∧ (∀ h ∈ ch, h.2 = false) ∧
      (sib = [] ∨ c.nodeUnique.isSome ∨
        (∃ a b, ch = [(a, b)] ∧ ∀ i ∈ sib, i = rp.relAlias ∨ i = a))) ∧
    (c.toCol = none ↔ rp.toAlias = rp.fromAlias) ∧ c.produced = 0 := by
  refine ⟨?_, ?_, ?_, rfl⟩
  · simp only [ctNew]
    split
    · rename_i h
      simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_iff] at h
      obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
      intro _; exact ⟨h1, h2, h3, h4, h5, rfl⟩
    · simp
  · intro he
    simp only [ctNew] at he ⊢
    simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', List.isEmpty_iff,
      List.all_eq_true, beq_iff_eq] at he ⊢
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := he
    refine ⟨h1, h2, fun h hh => by simpa using h4 h hh, ?_⟩
    rcases h3 with (h3 | h3) | ⟨hl, hs⟩
    · exact Or.inl h3
    · exact Or.inr (Or.inl h3)
    · refine Or.inr (Or.inr ?_)
      match ch, hl with
      | [(a, b)], _ =>
        refine ⟨a, b, rfl, fun i hi => ?_⟩
        have := hs i hi
        simp only [beq_iff_eq, List.any_cons, List.any_nil, Bool.or_false] at this
        rcases this with h | h
        · exact Or.inl h
        · exact Or.inr h.symm
  · by_cases h : rp.toAlias = rp.fromAlias <;> simp [ctNew, h]

/-- The F·A batched path is taken exactly for non-emitting, directed hops whose fused chain is
all directed and whose siblings it can account for (cond_traverse.rs `batched_eligible`). -/
theorem ctNew_eligible_iff (rp : RelPat) (er bd op : Bool) (sib : List Nat) (ch : List (Nat × Bool))
    (ncs : Option (List SibEnds)) :
    (ctNew rp er bd op sib ch ncs).eligible = true ↔
      er = false ∧ rp.bidir = false ∧
        (sib = [] ∨ (ctNew rp er bd op sib ch ncs).nodeUnique.isSome ∨
          (ch.length = 1 ∧ ∀ i ∈ sib, i = rp.relAlias ∨ ∃ h ∈ ch, h.1 = i)) ∧
        ∀ h ∈ ch, h.2 = false := by
  simp only [ctNew, Bool.and_eq_true, Bool.or_eq_true, List.isEmpty_iff,
    List.all_eq_true, List.any_eq_true, beq_iff_eq, Bool.not_eq_eq_eq_not, Bool.not_true]
  constructor
  · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩; exact ⟨h1, h2, or_assoc.1 h3, h4⟩
  · rintro ⟨h1, h2, h3, h4⟩; exact ⟨⟨⟨h1, h2⟩, or_assoc.2 h3⟩, h4⟩

/-- Without siblings or a chain the rule is the one before #3639. -/
theorem ctNew_eligible_plain (rp : RelPat) (er bd op : Bool) (ncs : Option (List SibEnds)) :
    (ctNew rp er bd op [] [] ncs).eligible = (!er && !rp.bidir) := by
  simp [ctNew]

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
  /-- `multi_edge` (#3639): some matched tensor holds a parallel edge (`Tensor::has_multi_edge`) -/
  multiEdge : Bool

/-- `resolve` = `g.resolve_label_ids` (`None` when a label is unknown); `typeId` =
`get_type_id`; `fwdIter`/`revIter` = whether `build_unrestricted_iter` found the matrix. -/
def buildState (resolve : List String → Option (List Nat)) (typeId : String → Option Nat)
    (ntensors : Nat) (fromL toL : List String) (types : List String) (bidir transposed : Bool)
    (fwdOk revOk : Bool) (hasMulti : Nat → Bool := fun _ => false) : CtState :=
  let (fs, fd) := if transposed then (toL, fromL) else (fromL, toL)
  let fsi := resolve fs
  let fdi := resolve fd
  let (rsi, rdi) := if bidir then
      (if transposed then (resolve fromL, resolve toL) else (resolve toL, resolve fromL))
    else (some [], some [])
  let noMatch := fsi.isNone || fdi.isNone || rsi.isNone || rdi.isNone || !fwdOk || (bidir && !revOk)
  let edgeTypes := if types.isEmpty then List.range ntensors else types.filterMap typeId
  { fwdSrc := fsi.getD [], fwdDst := fdi.getD [], revSrc := rsi.getD [], revDst := rdi.getD [],
    edgeTypes, noMatch, hasRev := bidir && !(decide (types.length > 1)),
    multi := decide (types.length > 1), unionRev := bidir && decide (types.length > 1),
    multiEdge := edgeTypes.any hasMulti }

/-- **`multi_edge` (#3639)**: set exactly when a matched tensor (`edge_type_indices`) holds a
parallel edge; it disables the collapse (`CondTraverse.expandRowN`). -/
theorem buildState_multiEdge (resolve : List String → Option (List Nat))
    (typeId : String → Option Nat) (n : Nat) (fromL toL types : List String)
    (bidir tr fwdOk revOk : Bool) (hm : Nat → Bool) :
    let s := buildState resolve typeId n fromL toL types bidir tr fwdOk revOk hm
    (s.multiEdge = true ↔ ∃ t ∈ s.edgeTypes, hm t = true) := by
  simp [buildState]

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

/-- expand_into.rs: a synthetic multi-label check `(a:A:B)` (same alias, labels only on
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
