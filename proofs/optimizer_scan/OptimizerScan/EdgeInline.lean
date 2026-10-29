import OptimizerScan.IndexPipeline
/-
# Edge index scans, inline-attribute rewrites, `distance()` scans and the
# `NodeByIndexScanOp` runtime (origin/main 8743953a8)

| here | there |
| --- | --- |
| `QNode`, `QRel`, `IRs` | `QueryNode`, `QueryRelationship`, the `IR` variants the pass matches |
| `nAlias`/`nLabels`/`nIndexed`/`nMatch`/`nBuild` | `impl IndexSubject for Arc<QueryNode>` `utilize_index.rs:151-216` |
| `eAlias`/`eLabels`/`eIndexed`/`eFunc`/`eMatch`/`eBuild` | `impl IndexSubject for Arc<QueryRelationship>` `utilize_index.rs:218-310` |
| `endpointOK`, `edgeScanSel`, `edgeRefSel` | `EdgeByIndexScanOp::next` `runtime/ops/edge_by_index_scan.rs:330-380` (bound-endpoint filter), `CondTraverse` + `Filter` |
| `distScan`, `pointSel` | `try_distance_index_scan` 421, `IndexQuery::Point` arm of `build_query_node` (`index/mod.rs:1545`) |
| `extraLabels`, `scanRow`, `evalIQE` | `NodeByIndexScanOp::{new,next,evaluate_index_query}` `runtime/ops/node_by_index_scan.rs:56,293,98` |
-/
namespace OptimizerScan.Index

/-! ## The two `IndexSubject` instances -/

structure QNode where
  alias : Nat
  labels : List Nat
  attrs : List (Nat × T)

structure QRel where
  alias : Nat
  types : List Nat
  attrs : List (Nat × T)
  src : QNode
  dst : QNode

inductive IRs
  | nodeByLabelScan (n : QNode)
  | condTraverse (r : QRel) (transposed : Bool) (siblings : List Nat)
  | nodeByIndexScan (n : QNode) (L : Nat) (q : IQ)
  | edgeByIndexScan (r : QRel) (q : IQ) (transposed : Bool)
  | other

def nAlias (n : QNode) : Nat := n.alias
def nLabels (n : QNode) : List Nat := n.labels
/-- `graph.is_indexed(label, attr, Range)`. -/
def nIndexed (idx : Nat → List Nat) (L k : Nat) : Bool := k ∈ idx L
def nMatch : IRs → Option QNode
  | .nodeByLabelScan n => if n.labels.isEmpty then none else some n
  | _ => none
def nBuild (n : QNode) (L : Nat) (q : IQ) : IRs := .nodeByIndexScan n L q

def eAlias (r : QRel) : Nat := r.alias
def eLabels (r : QRel) : List Nat := r.types
/-- `graph.is_edge_indexed(type, attr, Range)`. -/
def eIndexed (eidx : Nat → List Nat) (T' k : Nat) : Bool := k ∈ eidx T'
/-- `try_func_scan` for edges: no `distance()` path. -/
def eFunc (_r : QRel) : Option (QRel × Nat × IQ) := none
def eMatch : IRs → Option (QRel × Bool)
  | .condTraverse r tr sib =>
    if r.types.length ≠ 1 then none
    else if sib.any (· != r.alias) then none
    else some (r, tr)
  | _ => none
def eBuild (r : QRel) (q : IQ) (tr : Bool) : IRs := .edgeByIndexScan r q tr

theorem subject_accessors (n : QNode) (r : QRel) (idx : Nat → List Nat) (L k : Nat) :
    nAlias n = n.alias ∧ nLabels n = n.labels ∧ (nIndexed idx L k = true ↔ k ∈ idx L) ∧
    eAlias r = r.alias ∧ eLabels r = r.types ∧ (eIndexed idx L k = true ↔ k ∈ idx L) ∧
    eFunc r = none ∧ nBuild n L = IRs.nodeByIndexScan n L := by
  refine ⟨rfl, rfl, by simp [nIndexed], rfl, rfl, by simp [eIndexed], rfl, rfl⟩


/-- `match_scan_source` (node): a `NodeByLabelScan` with at least one label. -/
theorem nMatch_spec (ir : IRs) (n : QNode) : nMatch ir = some n ↔ ir = .nodeByLabelScan n ∧ n.labels ≠ [] := by
  cases ir <;> simp [nMatch]
  constructor
  · rintro ⟨h1, rfl⟩; exact ⟨rfl, h1⟩
  · rintro ⟨rfl, h1⟩; exact ⟨h1, rfl⟩

/-- `match_scan_source` (edge): exactly one relationship type and no uniqueness
constraint on any other edge alias (C `reduce_cond_op` gate). -/
theorem eMatch_spec (ir : IRs) (r : QRel) (tr : Bool) :
    eMatch ir = some (r, tr) ↔ ∃ sib, ir = .condTraverse r tr sib ∧ r.types.length = 1 ∧ ∀ s ∈ sib, s = r.alias := by
  cases ir with
  | condTraverse r' tr' sib =>
    simp only [eMatch]
    split
    · simp only [reduceCtorEq, false_iff, not_exists, not_and]
      intro s h; cases h; omega
    · split
      · next h1 h2 =>
        simp only [reduceCtorEq, false_iff, not_exists, not_and]
        intro s h; cases h; intro _ hall
        obtain ⟨x, hx, hne⟩ := List.any_eq_true.mp h2
        simp [hall x hx] at hne
      · next h1 h2 =>
        simp only [Option.some.injEq, Prod.mk.injEq, IRs.condTraverse.injEq]
        constructor
        · rintro ⟨rfl, rfl⟩
          refine ⟨sib, ⟨rfl, rfl, rfl⟩, by omega, fun s hs => ?_⟩
          exact Classical.byContradiction fun hne => h2 (List.any_eq_true.mpr ⟨s, hs, by simpa using hne⟩)
        · rintro ⟨_, ⟨h1', h2', -⟩, -, -⟩; exact ⟨h1', h2'⟩
  | _ => simp [eMatch]

/-! ## Edge-index soundness -/

/-- An edge as the runtime sees it: endpoints plus the property entity (its
single type is `ent.labels`). -/
structure EdgeE where
  src : Nat
  dst : Nat
  ent : Node

/-- The bound-endpoint filter of `EdgeByIndexScanOp` (`edge_by_index_scan.rs:338-372`),
which is also what `CondTraverse` enforces for bound endpoints. -/
def endpointOK (bf bt : Option Nat) (tr same : Bool) (e : EdgeE) : Bool :=
  let ft := if tr then (e.dst, e.src) else (e.src, e.dst)
  bf.all (· == ft.1) && bt.all (· == ft.2) && (!same || ft.1 == ft.2)

/-- `EdgeByIndexScan` (+ kept Filter) after the edge pass on type `T'`: the same
`utilize` as the node pass (`try_index_rewrite::<Arc<QueryRelationship>>` runs the
generic `try_filter_pushdown`), selecting edge entities, then the endpoint check. -/
def edgeScanSel (eidx : Nat → List Nat) (opq : Nat → Node → Bool) (T' : Nat) (f : F)
    (bf bt : Option Nat) (tr same : Bool) (e : EdgeE) : Bool :=
  (utilize eidx [T'] f).sel eidx opq e.ent && endpointOK bf bt tr same e

/-- `Filter(f) → CondTraverse([:T'])`: edges of type `T'` meeting the bound
endpoints and satisfying `f`. -/
def edgeRefSel (eidx : Nat → List Nat) (opq : Nat → Node → Bool) (T' : Nat) (f : F)
    (bf bt : Option Nat) (tr same : Bool) (e : EdgeE) : Bool :=
  (reference [T'] f).sel eidx opq e.ent && endpointOK bf bt tr same e

/-- **PROVEN** (edge headline): for every child row (bound endpoints `bf`/`bt`,
`transposed`, self-loop pattern `same`) the edge-index plan emits exactly the
edges `Filter → CondTraverse` emits, under the same hypotheses as the node
headline `utilize_sound`. -/
theorem edge_utilize_sound (eidx : Nat → List Nat) (opq : Nat → Node → Bool) (T' : Nat) (f : F)
    (hf : goodF f = true) (bf bt : Option Nat) (tr same : Bool) (e : EdgeE) (he : Faithful e.ent) :
    edgeScanSel eidx opq T' f bf bt tr same e = edgeRefSel eidx opq T' f bf bt tr same e := by
  simp only [edgeScanSel, edgeRefSel, utilize_sound eidx T' opq f hf e.ent he]

/-! ## HISTORICAL: the inline-attribute path (`(n:L {k: v})`), removed by #2390 (d2c42e032)

Until #2390 `try_index_rewrite` had a second path for patterns carrying inline attributes. #2390
removed it (with `IndexSubject::inline_attrs`): the planner now lowers inline attributes to an
`IR::Filter` and strips them from the pattern, so they reach the index through the filter path.
That removal loses nothing on literal values: the old inline rewrite (`get_inline_attr_index` 721, `needs_inline_post_filter` 897, `apply_inline_rewrite` 1038 at a9377c636) was already exactly `utilize` of the equivalent filter `n.k = v`. Lines below cite
a9377c636. -/

/-! ## `distance()` scans -/

/-- The comparison operator and which child holds `distance(...)`. -/
def distScan (op : Op) (attrLeft : Bool) (lhsIsProp : Bool) (rhsIsProp : Bool) : Bool :=
  (match op with
   | .lt | .le => attrLeft
   | .gt | .ge => !attrLeft
   | .eq => false) &&
  -- exactly one argument of `distance` contains a property (`(Some, None) | (None, Some)`)
  (lhsIsProp != rhsIsProp)

/-- Semantics of the pushed filter `distance(n.k, p) op r` (or flipped): only
`<`/`<=` with distance on the small side reach here. -/
def distHolds (op : Op) (attrLeft : Bool) (d r : Int) : Bool :=
  match op, attrLeft with
  | .lt, true => decide (d < r) | .le, true => decide (d ≤ r)
  | .gt, false => decide (d < r) | .ge, false => decide (d ≤ r)
  | _, _ => false

/-- `IndexQuery::Point` against RediSearch's GEO radius filter (inclusive
radius). `geoHit` is RS's answer; completeness is the documented contract,
which fails beyond |lat| > 85.05 (index_layer bug 12). -/
def pointSel (geoHit : Bool) (holds : Bool) : Bool := geoHit && holds

/-- **PROVEN**: every accepted `distance` shape implies `d ≤ r`, so with a
complete GEO filter (`d ≤ r → geoHit`) and the Filter kept above (the attribute
side is a `FuncInvocation`, so `needs_post_filter` is true), the scan selects
exactly the rows the Filter selects. -/
theorem distance_sound (op : Op) (al lp rp : Bool) (h : distScan op al lp rp = true) (d r : Int)
    (geoHit : Bool) (hcomplete : d ≤ r → geoHit = true) :
    pointSel geoHit (distHolds op al d r) = distHolds op al d r := by
  cases hh : distHolds op al d r
  · simp [pointSel]
  · have : d ≤ r := by
      cases op <;> cases al <;> simp [distHolds] at hh <;> omega
    simp [pointSel, hcomplete this]

theorem distScan_shapes (op : Op) (al lp rp : Bool) (h : distScan op al lp rp = true) :
    (op = .lt ∨ op = .le) ∧ al = true ∨ (op = .gt ∨ op = .ge) ∧ al = false := by
  cases op <;> cases al <;> simp_all [distScan]

/-- The Filter is always kept: the attribute side is a function call. -/
theorem distance_keeps_filter (t : T) : nonIdxT (.abs t) = true := rfl

/-! ## `NodeByIndexScanOp` -/

/-- `NodeByIndexScanOp::new` (`node_by_index_scan.rs:66-70`). -/
def extraLabels (ls : List Nat) : Option (List Nat) := if ls.length > 1 then some ls.tail else none

/-- `evaluate_index_query` with its error: `InList` over a non-list value is
`Err("IN operator requires a list")` (`node_by_index_scan.rs:241`). -/
def evalIQE (q : IQ) : Except String Q :=
  match q with
  | .inList _ (.term t) => match evalC t with
    | .arr _ => .ok (evalIQ q)
    | _ => .error "IN operator requires a list"
  | _ => .ok (evalIQ q)

/-- One input row of `NodeByIndexScanOp::next` (`node_by_index_scan.rs:308-335`):
index when `can_utilize_index`, else `get_nodes(labels)`, then the extra labels. -/
def scanRow (idx : Nat → List Nat) (ls : List Nat) (L : Nat) (q : IQ) (n : Node) : Bool :=
  let qv := evalIQ q
  (if canUtilize qv then idxSel idx L qv n else hasAll ls n) &&
  (match extraLabels ls with | some ex => ex.all (· ∈ n.labels) | none => true)

theorem extraLabels_spec (L : Nat) (rest : List Nat) :
    extraLabels (L :: rest) = if rest = [] then none else some rest := by
  cases rest <;> simp [extraLabels]

/-- **PROVEN**: the op's per-row selection is `Plan.sel (.idxScan ..)` once the
indexed label is first (which `reorder_subject_labels` guarantees) and the labels
are distinct. -/
theorem scanRow_eq_sel (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (rest : List Nat)
    (hL : L ∉ rest) (q : IQ) (n : Node) :
    scanRow idx (L :: rest) L q n = (Plan.idxScan (L :: rest) L q).sel idx opq n := by
  have he : (L :: rest).erase L = rest := by simp
  simp only [scanRow, Plan.sel, he, extraLabels_spec]
  cases rest with
  | nil => simp [hasAll]
  | cons _ _ => simp [hasAll]

theorem evalIQE_ok (q : IQ) (h : ∀ t k, q ≠ .inList k (.term t)) : evalIQE q = .ok (evalIQ q) := by
  cases q with
  | inList k r =>
    cases r with
    | list _ => rfl
    | term t => exact absurd rfl (h t k)
  | _ => rfl

theorem evalIQE_inList_term (k : Nat) (t : T) :
    evalIQE (.inList k (.term t)) = (match evalC t with
      | .arr _ => .ok (evalIQ (.inList k (.term t)))
      | _ => .error "IN operator requires a list") := rfl

theorem evalIQE_inList_err (k : Nat) (t : T) (h : ∀ xs, evalC t ≠ .arr xs) :
    evalIQE (.inList k (.term t)) = .error "IN operator requires a list" := by
  rw [evalIQE_inList_term]
  split
  · next xs hx => exact absurd hx (h xs)
  · rfl

/-- `can_utilize_index`: an empty `Or`/`And` (e.g. `IN [NULL]` after the runtime
drops non-scalars) is refused, so the kept Filter re-establishes correctness. -/
theorem canUtilize_empty : canUtilize (.or []) = false ∧ canUtilize (.and []) = false := ⟨rfl, rfl⟩

theorem isIndexable_spec (v : V) :
    isIndexable v = match v with | .i x => !lossy x | .date _ | .arr _ => false | _ => true := by
  cases v <;> rfl

/-- #2383: the runtime keeps an `IN` list whole, filtered to its scalar members. -/
theorem evalIQ_inList (k : Nat) (r : R) :
    evalIQ (.inList k r) = .inList k ((listVals r).filter isPrim) := rfl

end OptimizerScan.Index
