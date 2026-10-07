import PlannerBuild.Expr
/-
# `plan_match`: per-component operator choice (mod.rs:1832-2403)

| here | there |
| --- | --- |
| `QN`, `QR`, `Comp` | `QueryNode`, `QueryRelationship`, a connected component (ast.rs:470-640) |
| `MSt` | `Planner.visited` / `verified_labels` |
| `markLabels`, `unverified` | `mark_labels_verified` mod.rs:1787-1805, `unverified_labels` mod.rs:1809-1821 |
| `relKey`, `sortRels` | the `sort_by_key` of mod.rs:1877-1909 (stable) |
| `emitRel`, `opEmit` | the `emit_rel` closure mod.rs:2002-2011; `emit_rel(r) \|\| edge_pred.is_some()` |
| `scanOf`, `withF` | `AllNodeScan`/`NodeByLabelScan` choice; optional inline-attribute `Filter` |
| `LW`, `lowerInline` | `lowered_attrs` mod.rs:1852, `lower_inline_attrs` mod.rs:598-607 |
| `stripNode`, `stripRel`, `stripEndpoints` | `strip_node_attrs` mod.rs:623-634, `strip_rel_attrs` :644-667, `strip_endpoint_attrs` :673-693 |
| `opRel`, `edgePred`, `topFilters` | mod.rs:2032-2043 (and 2201-2212); the edge Filter :2162-2169 / 2295-2301; both endpoints :2182-2187 / 2310-2315 |
| `firstOp`/`firstRel`, `chainOp`/`chainRel` | first relationship mod.rs:2044-2196, the others mod.rs:2199-2321 |
| `MP.stripped` | the invariant `optimizer/references.rs` relies on (scan / fixed-length operators carry no inline map) |
| `pre2390_firstRel`, `pre2390_chainRel` | the same at a9377c636 (before #2390), kept for the fixed bug |
| `nodeOnly` | node-only component mod.rs:1913-1990 |
| `boundLabelFilters` | mod.rs:1858-1872 |
| `planComp`, `planMatch` | the component loop and the join / WHERE / bound filters mod.rs:1853-2403 |

Theorems added for #2390 (d2c42e032): `lowerInline_once` (a map is lowered at most once —
the `And(p, p)` duplicate is gone), `lowerInline_two_maps` (two maps on one alias are both
lowered), `stripNode_spec`/`stripRel_spec`/`stripEndpoints_spec`, `firstRel_lowers_from`/`_to`
and `chainRel_lowers_from`/`_to` (both endpoints lowered for every operator kind, bound or
not), `firstRel_edge_pred`/`chainRel_edge_pred` (a fixed hop's edge map is a Filter directly
above an operator that emits every parallel edge), `walk_keeps_edge_attrs`,
`firstRel_stripped`/`chainRel_stripped`/`planComp_stripped`/`planMatch_stripped`, and the
fixed bug `pre2390_firstRel_varlen_drops` / `pre2390_chainRel_from_on_pattern`.
| `buildPatternSubPlan` | `build_pattern_sub_plan` mod.rs:1038-1049 |

Expressions and the WHERE (`plan_filter`, FilterPlan.lean) are parameters;
operators are recorded with the relationship they traverse.
-/
namespace PlannerBuild.E

structure QN where
  alias : V
  labels : List String := []
  attrs : List (String × Ex) := []
  /-- Identity of the attrs map (`Arc::as_ptr(attrs)`), the third component of
  `lower_inline_attrs`'s key. -/
  ptr : Nat := 0

inductive Kind | fixed | varLen | shortest
  deriving DecidableEq

structure QR where
  alias : V
  named : Bool
  frm : QN
  to : QN
  kind : Kind := .fixed
  attrs : List (String × Ex) := []
  ptr : Nat := 0

structure Comp where
  nodes : List QN
  rels : List QR
  paths : List (V × List V) := []

structure MSt where
  visited : List V
  verified : List (V × List String)

def MSt.bound (st : MSt) (v : V) : Bool := st.visited.contains v

/-! ## Label bookkeeping -/

def lookupV (m : List (V × List String)) (v : V) : List String :=
  match m.find? (fun p => p.1 == v) with
  | some p => p.2
  | none => []

/-- `mark_labels_verified`: no entry for an empty label list; else union in. -/
def markLabels (st : MSt) (v : V) (ls : List String) : MSt :=
  if ls.isEmpty then st
  else
    let old := lookupV st.verified v
    { st with verified := (v, old ++ ls.filter (fun l => !old.contains l)) :: st.verified }

def unverified (st : MSt) (n : QN) : List String :=
  n.labels.filter fun l => !(lookupV st.verified n.alias).contains l

theorem lookupV_cons_self (m : List (V × List String)) (v : V) (ls : List String) :
    lookupV ((v, ls) :: m) v = ls := by simp [lookupV]

theorem lookupV_cons_other (m : List (V × List String)) (v w : V) (ls : List String) (h : w ≠ v) :
    lookupV ((v, ls) :: m) w = lookupV m w := by
  simp only [lookupV, List.find?_cons]
  have : (v == w) = false := by rw [beq_eq_false_iff_ne]; exact fun e => h e.symm
  simp [this]

/-- After marking, the marked labels are verified… -/
theorem unverified_after_mark (st : MSt) (n : QN) (h : n.labels ≠ []) :
    unverified (markLabels st n.alias n.labels) n = [] := by
  unfold markLabels unverified
  have : n.labels.isEmpty = false := by cases hl : n.labels <;> simp_all
  simp only [this, Bool.false_eq_true, ↓reduceIte, lookupV_cons_self]
  rw [List.filter_eq_nil_iff]
  intro l hl
  by_cases hc : (lookupV st.verified n.alias).contains l <;> simp [hc, hl]

/-- …and what was verified stays verified. -/
theorem markLabels_mono (st : MSt) (v w : V) (ls : List String) (l : String)
    (h : (lookupV st.verified w).contains l) : (lookupV (markLabels st v ls).verified w).contains l := by
  unfold markLabels
  split
  · exact h
  · by_cases hw : w = v
    · subst hw; simp only [lookupV_cons_self]; simp_all
    · simp only [lookupV_cons_other _ _ _ _ hw]; exact h

theorem unverified_spec (st : MSt) (n : QN) (l : String) :
    l ∈ unverified st n ↔ l ∈ n.labels ∧ ¬ (lookupV st.verified n.alias).contains l := by
  simp [unverified]

/-! ## Operators -/

inductive MP
  | arg
  | allScan (n : QN)
  | labelScan (n : QN)
  | filter (e : Ex) (c : MP)
  | pathB (ps : List (V × List V)) (c : MP)
  | cvlt (r : QR) (expandInto : Bool) (pathVar : Option V) (cs : List MP)
  | shortest (r : QR) (cs : List MP)
  | expandInto (r : QR) (emit : Bool) (cs : List MP)
  | condTr (r : QR) (emit : Bool) (cs : List MP)
  | cp (cs : List MP)

def scanOf (n : QN) : MP := if n.labels.isEmpty then .allScan n else .labelScan n

def withF : Option Ex → MP → MP
  | some e, c => .filter e c
  | none, c => c

/-- `emit_relationship`: a named edge, or one a named path lists (by id only, mod.rs:2003-2011). -/
def emitRel (c : Comp) (r : QR) : Bool :=
  r.named || c.paths.any fun p => p.2.any fun v => v.id == r.alias.id

/-- Strip the inline-attribute filters: the operator actually chosen. -/
def MP.core : MP → MP
  | .filter _ c => c.core
  | p => p

/-- The stack of `Filter`s above the operator, top first. -/
def MP.peel : MP → List Ex
  | .filter e c => e :: c.peel
  | _ => []

theorem core_withF (f : Option Ex) (c : MP) : (withF f c).core = c.core := by
  cases f <;> rfl

theorem peel_withF (f : Option Ex) (c : MP) : (withF f c).peel = f.toList ++ c.peel := by
  cases f <;> rfl

def MP.isCtEi : MP → Bool
  | .expandInto .. | .condTr .. => true
  | _ => false

/-! ## Lowering inline attributes once (#2390, d2c42e032) -/

/-- `lowered_attrs` (mod.rs:1850): (alias id, scope, attrs-map identity) already lowered. -/
abbrev LW := List (Nat × Nat × Nat)

def lwKey (alias : V) (ptr : Nat) : Nat × Nat × Nat := (alias.id, alias.scope, ptr)

/-- `lower_inline_attrs` (mod.rs:598-608): `HashSet::insert` returns false on a repeat → `None`;
otherwise the key is recorded and the map is lowered with `inline_attrs_to_filter`. -/
def lowerInline (lw : LW) (alias : V) (attrs : List (String × Ex)) (ptr : Nat) : Option Ex × LW :=
  if lw.contains (lwKey alias ptr) then (none, lw)
  else (inlineAttrsToFilter alias attrs, lw ++ [lwKey alias ptr])

def lowerN (lw : LW) (n : QN) : Option Ex × LW := lowerInline lw n.alias n.attrs n.ptr

theorem lowerInline_fresh (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat)
    (h : lw.contains (lwKey a p) = false) :
    lowerInline lw a as p = (inlineAttrsToFilter a as, lw ++ [lwKey a p]) := by
  simp only [lowerInline, h, Bool.false_eq_true, ↓reduceIte]

theorem lowerInline_mem (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat) :
    (lowerInline lw a as p).2.contains (lwKey a p) = true := by
  cases h : lw.contains (lwKey a p)
  · simp only [lowerInline, h, Bool.false_eq_true, ↓reduceIte, List.contains_append,
      List.contains_cons, beq_self_eq_true, Bool.true_or, Bool.or_true]
  · simp only [lowerInline, h, ↓reduceIte]

theorem lowerInline_of_mem (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat)
    (h : lw.contains (lwKey a p) = true) : lowerInline lw a as p = (none, lw) := by
  simp only [lowerInline, h, ↓reduceIte]

/-- A map is lowered **at most once**: a second request for the same (alias, map) yields nothing. -/
theorem lowerInline_once (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat) :
    (lowerInline (lowerInline lw a as p).2 a as p).1 = none := by
  rw [lowerInline_of_mem _ _ _ _ (lowerInline_mem lw a as p)]

theorem lowerInline_mono (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat) (k : Nat × Nat × Nat)
    (h : lw.contains k = true) : (lowerInline lw a as p).2.contains k = true := by
  cases h' : lw.contains (lwKey a p)
  · simp only [lowerInline, h', Bool.false_eq_true, ↓reduceIte, List.contains_append, h,
      Bool.true_or]
  · simp only [lowerInline, h', ↓reduceIte, h]

theorem lowerInline_other (lw : LW) (a : V) (as : List (String × Ex)) (p : Nat) (k : Nat × Nat × Nat)
    (hk : k ≠ lwKey a p) : (lowerInline lw a as p).2.contains k = lw.contains k := by
  cases h' : lw.contains (lwKey a p)
  · have : (k == lwKey a p) = false := by simpa using hk
    simp only [lowerInline, h', Bool.false_eq_true, ↓reduceIte, List.contains_append,
      List.contains_cons, List.contains_nil, Bool.or_false, this]
  · simp only [lowerInline, h', ↓reduceIte]

/-- Identity, not equality: two distinct maps on one alias are both lowered
(`(a {x:1})-->(c), (a {x:2})-->(d)`). -/
theorem lowerInline_two_maps (a : V) (as bs : List (String × Ex)) (p q : Nat) (hpq : p ≠ q) :
    (lowerInline (lowerInline [] a as p).2 a bs q).1 = inlineAttrsToFilter a bs := by
  have : (lowerInline [] a as p).2.contains (lwKey a q) = false := by
    rw [lowerInline_other _ _ _ _ _ (by simp [lwKey]; exact fun h => hpq h.symm)]; rfl
  rw [lowerInline_fresh _ _ _ _ this]

/-- `strip_node_attrs` (mod.rs:623-633): the same node when there is nothing to strip. -/
def stripNode (n : QN) : QN := if n.attrs.isEmpty then n else { n with attrs := [] }

/-- `strip_rel_attrs` (mod.rs:644-666). `Arc::ptr_eq(&from, &rel.from)` holds exactly when
`strip_node_attrs` returned its argument, i.e. when `from` has no attrs (`stripNode_self`). -/
def stripRel (r : QR) : QR :=
  if r.attrs.isEmpty && r.frm.attrs.isEmpty && r.to.attrs.isEmpty then r
  else { r with attrs := [], frm := stripNode r.frm, to := stripNode r.to }

/-- `strip_endpoint_attrs` (mod.rs:673-694): the walks keep their edge's own attrs. -/
def stripEndpoints (r : QR) : QR :=
  if r.frm.attrs.isEmpty && r.to.attrs.isEmpty then r
  else { r with frm := stripNode r.frm, to := stripNode r.to }

theorem stripNode_self (n : QN) (h : n.attrs = []) : stripNode n = n := by simp [stripNode, h]

theorem stripNode_spec (n : QN) :
    (stripNode n).attrs = [] ∧ (stripNode n).alias = n.alias ∧ (stripNode n).labels = n.labels := by
  unfold stripNode; split <;> simp_all

theorem stripRel_spec (r : QR) :
    (stripRel r).attrs = [] ∧ (stripRel r).frm.attrs = [] ∧ (stripRel r).to.attrs = [] ∧
    (stripRel r).alias = r.alias ∧ (stripRel r).named = r.named ∧ (stripRel r).kind = r.kind ∧
    (stripRel r).frm.alias = r.frm.alias ∧ (stripRel r).to.alias = r.to.alias ∧
    (stripRel r).frm.labels = r.frm.labels ∧ (stripRel r).to.labels = r.to.labels := by
  unfold stripRel
  split
  · rename_i h; simp only [Bool.and_eq_true, List.isEmpty_iff] at h; simp [h]
  · simp [(stripNode_spec r.frm), (stripNode_spec r.to)]

theorem stripEndpoints_spec (r : QR) :
    (stripEndpoints r).attrs = r.attrs ∧ (stripEndpoints r).frm.attrs = [] ∧
    (stripEndpoints r).to.attrs = [] ∧ (stripEndpoints r).alias = r.alias ∧
    (stripEndpoints r).named = r.named ∧ (stripEndpoints r).kind = r.kind ∧
    (stripEndpoints r).frm.alias = r.frm.alias ∧ (stripEndpoints r).to.alias = r.to.alias ∧
    (stripEndpoints r).frm.labels = r.frm.labels ∧ (stripEndpoints r).to.labels = r.to.labels := by
  unfold stripEndpoints
  split
  · rename_i h; simp only [Bool.and_eq_true, List.isEmpty_iff] at h; simp [h]
  · simp [(stripNode_spec r.frm), (stripNode_spec r.to)]

/-! ## Stripped plans: no operator carries a predicate-position inline map -/

/-- Scans and fixed-length operators carry no inline attrs at all; walks carry only
their edge's own (they prune per edge with them). This is what
`optimizer/references.rs` relies on when it treats these operators as reading
nothing (proofs/optimizer_rewrites `References.refsNew_complete`). -/
def MP.stripped : MP → Bool
  | .arg => true
  | .allScan n | .labelScan n => n.attrs.isEmpty
  | .filter _ c | .pathB _ c => c.stripped
  | .cvlt r _ _ cs | .shortest r cs => r.frm.attrs.isEmpty && r.to.attrs.isEmpty && strippedL cs
  | .expandInto r _ cs | .condTr r _ cs =>
      r.attrs.isEmpty && r.frm.attrs.isEmpty && r.to.attrs.isEmpty && strippedL cs
  | .cp cs => strippedL cs
where
  strippedL : List MP → Bool
    | [] => true
    | c :: cs => c.stripped && strippedL cs

theorem stripped_withF (f : Option Ex) (c : MP) : (withF f c).stripped = c.stripped := by
  cases f <;> rfl

theorem scanOf_stripNode (n : QN) : (scanOf (stripNode n)).stripped = true := by
  unfold scanOf; split <;> simp [MP.stripped, (stripNode_spec n).1]

/-! ## Relationship order -/

def tier (r : QR) : Nat :=
  match r.kind with
  | .fixed => 0
  | .varLen => 1
  | .shortest => 2

/-- `(base, has_bound, unfiltered)`; `has_bound` is 0 when an endpoint is
bound; `unfiltered` only distinguishes hops leaving a bound node. -/
def relKey (st : MSt) (filterVars : List Nat) (r : QR) : Nat × Nat × Nat :=
  let hb := if st.bound r.frm.alias || st.bound r.to.alias then 0 else 1
  let unf := if hb = 0 then
      (if [r.frm, r.to].any (fun n => !n.attrs.isEmpty || filterVars.contains n.alias.id) then 0 else 1)
    else 0
  (tier r, hb, unf)

def keyLe (a b : Nat × Nat × Nat) : Bool :=
  a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && a.2.2 ≤ b.2.2)))

def sortRels (st : MSt) (fv : List Nat) (rs : List QR) : List QR :=
  rs.mergeSort fun a b => keyLe (relKey st fv a) (relKey st fv b)

theorem keyLe_trans (a b c : Nat × Nat × Nat) (h1 : keyLe a b = true) (h2 : keyLe b c = true) :
    keyLe a c = true := by
  obtain ⟨a1, a2, a3⟩ := a; obtain ⟨b1, b2, b3⟩ := b; obtain ⟨c1, c2, c3⟩ := c
  simp only [keyLe, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at *
  omega

theorem keyLe_total (a b : Nat × Nat × Nat) : (keyLe a b || keyLe b a) = true := by
  obtain ⟨a1, a2, a3⟩ := a; obtain ⟨b1, b2, b3⟩ := b
  simp only [keyLe, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  omega

/-- The order is a permutation, sorted by key: fixed-length hops first, then
variable-length, then shortest paths; within a tier, hops touching a bound
node first (mod.rs:1877-1909). -/
theorem sortRels_spec (st : MSt) (fv : List Nat) (rs : List QR) :
    (sortRels st fv rs).Perm rs ∧
    (sortRels st fv rs).Pairwise (fun a b => keyLe (relKey st fv a) (relKey st fv b) = true) := by
  refine ⟨List.mergeSort_perm _ _, ?_⟩
  exact List.pairwise_mergeSort
    (fun a b c h1 h2 => keyLe_trans (relKey st fv a) (relKey st fv b) (relKey st fv c) h1 h2)
    (fun a b => keyLe_total (relKey st fv a) (relKey st fv b)) rs

theorem sortRels_tiers (st : MSt) (fv : List Nat) (rs : List QR) :
    (sortRels st fv rs).Pairwise (fun a b => tier a ≤ tier b) := by
  apply (sortRels_spec st fv rs).2.imp
  intro a b h
  simp only [keyLe, relKey, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  omega

/-! ## The first relationship of a component -/

def visit (st : MSt) (r : QR) : MSt :=
  let st := { st with visited := st.visited ++ [r.frm.alias, r.to.alias, r.alias] }
  markLabels (markLabels st r.frm.alias r.frm.labels) r.to.alias r.to.labels

def isWalk (r : QR) : Bool := r.kind != .fixed

/-- The pattern the operator embeds (mod.rs:2034-2038 / 2203-2207). -/
def opRel (r : QR) : QR := if isWalk r then stripEndpoints r else stripRel r

/-- The edge predicate (mod.rs:2039-2043 / 2208-2212): walks never get one. -/
def edgePred (lw : LW) (r : QR) : Option Ex × LW :=
  if isWalk r then (none, lw) else lowerInline lw r.alias r.attrs r.ptr

/-- `emit_rel(relationship) || edge_pred.is_some()`. -/
def opEmit (c : Comp) (lw : LW) (r : QR) : Bool := emitRel c r || (edgePred lw r).1.isSome

/-- The operator of the first hop (mod.rs:2044-2155), before any Filter. -/
def firstOp (st : MSt) (c : Comp) (lw : LW) (r : QR) : MP :=
  match r.kind with
  | .shortest => .shortest (opRel r) []
  | .varLen =>
    if st.bound r.frm.alias then
      .cvlt (opRel r) (r.frm.alias.id != r.to.alias.id && st.bound r.to.alias) none []
    else .cvlt (opRel r) false none [scanOf (opRel r).frm]
  | .fixed =>
    if r.frm.alias.id = r.to.alias.id then
      .expandInto (opRel r) (opEmit c lw r)
        (if st.bound r.frm.alias then [] else [scanOf (opRel r).frm])
    else if st.bound r.frm.alias && st.bound r.to.alias then .expandInto (opRel r) (opEmit c lw r) []
    else .condTr (opRel r) (opEmit c lw r) []

/-- Edge Filter on a CondTraverse/ExpandInto (mod.rs:2162-2169), then both endpoints'
inline attrs, `to` then `from`, **whether or not they are bound** (mod.rs:2183-2189). -/
def topFilters (lw : LW) (r : QR) (op : MP) : MP × LW :=
  let (ep, lw) := edgePred lw r
  let res := if op.isCtEi then withF ep op else op
  let (ft, lw) := lowerN lw r.to
  let res := withF ft res
  let (ff, lw) := lowerN lw r.frm
  (withF ff res, lw)

def firstRel (st : MSt) (c : Comp) (lw : LW) (r : QR) : MP × LW :=
  topFilters lw r (firstOp st c lw r)

/-- The operator chosen for the first hop (attribute filters stripped). -/
theorem firstOp_isCtEi (st : MSt) (c : Comp) (lw : LW) (r : QR) :
    (firstOp st c lw r).isCtEi = (r.kind == .fixed) := by
  unfold firstOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.isCtEi]

theorem firstOp_core (st : MSt) (c : Comp) (lw : LW) (r : QR) :
    (firstOp st c lw r).core = firstOp st c lw r := by
  unfold firstOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.core]

theorem topFilters_core (lw : LW) (r : QR) (op : MP) : (topFilters lw r op).1.core = op.core := by
  unfold topFilters
  cases op.isCtEi <;> simp [core_withF]

/-- The operator chosen for the first hop (attribute filters stripped): `firstOp`'s table. -/
theorem firstRel_core (st : MSt) (c : Comp) (lw : LW) (r : QR) :
    (firstRel st c lw r).1.core = firstOp st c lw r := by
  rw [firstRel, topFilters_core, firstOp_core]

theorem edgePred_lw (lw : LW) (r : QR) (k : Nat × Nat × Nat) (h : lw.contains k = true) :
    (edgePred lw r).2.contains k = true := by
  unfold edgePred; split
  · exact h
  · exact lowerInline_mono _ _ _ _ _ h

/-- The filters stacked above the operator, top first: `from`'s, `to`'s, then the edge's
(fixed-length only). -/
theorem topFilters_peel (lw : LW) (r : QR) (op : MP) (hop : op.peel = []) :
    (topFilters lw r op).1.peel =
      (lowerN (lowerN (edgePred lw r).2 r.to).2 r.frm).1.toList ++
      (lowerN (edgePred lw r).2 r.to).1.toList ++
      (if op.isCtEi then (edgePred lw r).1.toList else []) := by
  unfold topFilters
  cases op.isCtEi <;> simp [peel_withF, hop]

theorem firstOp_peel (st : MSt) (c : Comp) (lw : LW) (r : QR) : (firstOp st c lw r).peel = [] := by
  unfold firstOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.peel]

/-- **#2390's fix.** Both endpoints' inline attrs become Filters above the first hop for every
operator kind and whether or not the endpoint was bound by an earlier clause — as long as
that (alias, map) was not already lowered in this clause. Before #2390 a bound `from` of a
var-length / shortest-path hop got none (`pre2390_firstRel_varlen_drops`). -/
theorem firstRel_lowers_from (st : MSt) (c : Comp) (lw : LW) (r : QR) (f : Ex)
    (hk : (lowerN (edgePred lw r).2 r.to).2.contains (lwKey r.frm.alias r.frm.ptr) = false)
    (hf : inlineAttrsToFilter r.frm.alias r.frm.attrs = some f) :
    f ∈ (firstRel st c lw r).1.peel := by
  unfold firstRel
  rw [topFilters_peel _ _ _ (firstOp_peel st c lw r)]
  simp only [lowerN] at hk ⊢
  rw [lowerInline_fresh _ _ _ _ hk, hf]; simp

theorem firstRel_lowers_to (st : MSt) (c : Comp) (lw : LW) (r : QR) (f : Ex)
    (hk : (edgePred lw r).2.contains (lwKey r.to.alias r.to.ptr) = false)
    (hf : inlineAttrsToFilter r.to.alias r.to.attrs = some f) :
    f ∈ (firstRel st c lw r).1.peel := by
  unfold firstRel
  rw [topFilters_peel _ _ _ (firstOp_peel st c lw r)]
  simp only [lowerN] at hk ⊢
  rw [lowerInline_fresh _ _ _ _ hk, hf]; simp

/-- A fixed-length hop with an inline edge map: the predicate is a Filter right above the
operator, and the operator is told to emit every parallel edge (`emit_relationship`), so the
Filter can test each one (mod.rs:2155-2169). -/
theorem firstRel_edge_pred (st : MSt) (c : Comp) (lw : LW) (r : QR) (f : Ex)
    (hfix : r.kind = .fixed) (hk : lw.contains (lwKey r.alias r.ptr) = false)
    (hf : inlineAttrsToFilter r.alias r.attrs = some f) :
    f ∈ (firstRel st c lw r).1.peel ∧ opEmit c lw r = true := by
  have he : edgePred lw r = (some f, lw ++ [lwKey r.alias r.ptr]) := by
    simp [edgePred, isWalk, hfix, lowerInline_fresh _ _ _ _ hk, hf]
  refine ⟨?_, by simp [opEmit, he]⟩
  unfold firstRel
  rw [topFilters_peel _ _ _ (firstOp_peel st c lw r), firstOp_isCtEi, he]
  simp [hfix]

/-- A walk keeps its edge's own attrs on the pattern (it prunes per edge) and gets no edge Filter. -/
theorem walk_keeps_edge_attrs (lw : LW) (r : QR) (h : isWalk r = true) :
    (opRel r).attrs = r.attrs ∧ (opRel r).frm.attrs = [] ∧ (opRel r).to.attrs = [] ∧
    (edgePred lw r).1 = none := by
  simp [opRel, edgePred, h, stripEndpoints_spec]

theorem opRel_stripped (r : QR) :
    (opRel r).frm.attrs = [] ∧ (opRel r).to.attrs = [] ∧ (isWalk r = false → (opRel r).attrs = []) := by
  unfold opRel; split
  · rename_i h; exact ⟨(stripEndpoints_spec r).2.1, (stripEndpoints_spec r).2.2.1, by simp [h]⟩
  · exact ⟨(stripRel_spec r).2.1, (stripRel_spec r).2.2.1, fun _ => (stripRel_spec r).1⟩

theorem topFilters_stripped (lw : LW) (r : QR) (op : MP) :
    (topFilters lw r op).1.stripped = op.stripped := by
  unfold topFilters
  cases op.isCtEi <;> simp [stripped_withF]

/-- Every operator of the first hop carries a stripped pattern. -/
theorem firstRel_stripped (st : MSt) (c : Comp) (lw : LW) (r : QR) :
    (firstRel st c lw r).1.stripped = true := by
  unfold firstRel; rw [topFilters_stripped]
  obtain ⟨h1, h2, h3⟩ := opRel_stripped r
  have hs : (scanOf (opRel r).frm).stripped = true := by
    unfold scanOf; split <;> simp [MP.stripped, h1]
  have h3' : r.kind = .fixed → (opRel r).attrs = [] := fun hk => h3 (by simp [isWalk, hk])
  unfold firstOp
  cases hk : r.kind <;> by_cases hb1 : st.bound r.frm.alias <;> by_cases hb2 : st.bound r.to.alias <;>
    by_cases h4 : r.frm.alias.id = r.to.alias.id <;>
    simp [hb1, hb2, h4, MP.stripped, MP.stripped.strippedL, h1, h2, hs, h3', hk]

/-! ### Historical: before #2390 (a9377c636 mod.rs:1890-2048) -/

def attrF (n : QN) : Option Ex := inlineAttrsToFilter n.alias n.attrs
def relF (r : QR) : Option Ex := inlineAttrsToFilter r.alias r.attrs

/-- `plan_match`'s first hop before #2390: endpoint filters only for *unbound* endpoints,
the unstripped pattern embedded in the operator. -/
def pre2390_firstRel (st : MSt) (c : Comp) (r : QR) : MP :=
  let res := match r.kind with
    | .shortest => .shortest r []
    | .varLen =>
      if st.bound r.frm.alias then
        .cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.to.alias) none []
      else .cvlt r false none [withF (attrF r.frm) (scanOf r.frm)]
    | .fixed =>
      if r.frm.alias.id = r.to.alias.id then
        if st.bound r.frm.alias then withF (attrF r.frm) (.expandInto r (emitRel c r) [])
        else .expandInto r (emitRel c r) [withF (attrF r.frm) (scanOf r.frm)]
      else if st.bound r.frm.alias && st.bound r.to.alias then
        withF (attrF r.to) (withF (attrF r.frm) (.expandInto r (emitRel c r) []))
      else withF (relF r) (.condTr r (emitRel c r) [])
  let res := if st.bound r.to.alias then res else withF (attrF r.to) res
  if st.bound r.frm.alias then res else withF (attrF r.frm) res

/-- `MATCH (a:N) WITH a MATCH (a {x: 1})-[:R*1..2]->(b)`: the bound `a`'s map `{x: 1}`. -/
def aV : V := ⟨0, 0⟩
def xOne : Ex := .node (.const (.int 1)) []
def qVL : QR := { alias := ⟨2, 0⟩, named := false, frm := { alias := aV, attrs := [("x", xOne)], ptr := 7 },
                  to := { alias := ⟨1, 0⟩ }, kind := .varLen }
def stA : MSt := { visited := [aV], verified := [] }

/-- **Bug fixed by #2390 (d2c42e032)**: the old plan has no Filter for the bound `a` and the
CondVarLenTraverse reads only the edge's attrs, so `x = 1` was never enforced (live: rows
2, 3, 3; C 2, 3). The new plan lowers it. -/
theorem pre2390_firstRel_varlen_drops (c : Comp) :
    (pre2390_firstRel stA c qVL).peel = [] ∧
    (firstRel stA c [] qVL).1.peel = [attrEq aV ("x", xOne)] := by
  constructor <;> rfl

end PlannerBuild.E
