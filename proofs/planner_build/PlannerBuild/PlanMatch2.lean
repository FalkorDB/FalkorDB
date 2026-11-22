import PlannerBuild.PlanMatch
import PlannerBuild.Lift
import PlannerBuild.HopOrder
/-
# `plan_match`: components, bound-label filters, path elision, joining

Continues PlanMatch.lean. Theorems: `chainRel_core` (operator table for the
later hops — including the self-loop case `ExpandInto(scan, res)` behind bug 4),
`planComp_binds` (every endpoint and edge alias of a component is bound
afterwards), `boundLabels_spec` (an already-bound endpoint with labels not yet
enforced gets exactly a `hasLabels(alias, missing)` filter, after which none
are missing), `nodeOnly_*`, `planMatch_join`, `elide_spec`,
`buildPatternSubPlan_restores`. #2390: `chainRel_lowers_from`/`_to`, `chainRel_edge_pred`,
`nodeOnly_bound_filter`, the `stripped` invariant (`bindPath_stripped`, `elide_stripped`,
`planComp_stripped`, `planMatch_stripped`). #2981: `planMatch` = `planFrom` (the component loop)
on `liftPattern`'s rewritten components, the lifted filters first (`planMatch_nolift`,
`planMatch_join`, `planMatch_join_nolift`); `q2923_unlifted` / `q2923_lifted`,
`q2923_bound_not_lifted`, `lifted_rel_emitted`.
-/
namespace PlannerBuild.E

/-- The operator of a later hop (mod.rs:2496-2585), stacked on `res`. An unbound self-loop
scans its node on top of `res` (#3102). -/
def chainOp (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) : MP :=
  match r.kind with
  | .shortest => .shortest (opRel r) [res]
  | .varLen => .cvlt (opRel r) (r.frm.alias.id != r.to.alias.id && st.bound r.frm.alias && st.bound r.to.alias)
      none [res]
  | .fixed =>
    if r.frm.alias.id = r.to.alias.id then .expandInto (opRel r) (opEmit c lw r)
      (if st.bound r.frm.alias then [res] else [scanOn (opRel r).frm res])
    else if st.bound r.frm.alias && st.bound r.to.alias then .expandInto (opRel r) (opEmit c lw r) [res]
    else .condTr (opRel r) (opEmit c lw r) [res]

/-- A later hop: the operator, its edge Filter, then both endpoints — `from` included, which
before #2390 had no lowering here at all (mod.rs:2586-2616). -/
def chainRel (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) : MP × LW :=
  topFilters lw r (chainOp st c lw res r)

theorem chainOp_core (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) :
    (chainOp st c lw res r).core = chainOp st c lw res r := by
  unfold chainOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.core]

theorem chainOp_peel (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) :
    (chainOp st c lw res r).peel = [] := by
  unfold chainOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.peel]

theorem chainOp_isCtEi (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) :
    (chainOp st c lw res r).isCtEi = (r.kind == .fixed) := by
  unfold chainOp
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, MP.isCtEi]

/-- Operator table for the later hops (attribute filters stripped). -/
theorem chainRel_core (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) :
    (chainRel st c lw res r).1.core = chainOp st c lw res r := by
  rw [chainRel, topFilters_core, chainOp_core]

/-- **#2999 fixed by #3102 (61284ea4f)**: an unbound self-loop hop after other hops is
`ExpandInto(Scan(res))` — one child, the chain under the scan — so every earlier hop runs
(Match.lean `selfloop_scan_keeps_chain`). (`orderRels` also moves such a hop behind the hop that
binds its node whenever one comes later, so this shape is now only the no-binder case.) -/
theorem chainRel_selfloop_unbound (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR)
    (hk : r.kind = .fixed) (hs : r.frm.alias.id = r.to.alias.id) (hb : st.bound r.frm.alias = false) :
    (chainRel st c lw res r).1.core = .expandInto (opRel r) (opEmit c lw r) [scanOn (opRel r).frm res] := by
  rw [chainRel_core]; simp [chainOp, hk, hs, hb]

/-- **#2390**: a later hop lowers its `from` endpoint too (it used to be left on the
CondTraverse's pattern, invisible to the optimizer), and its `to` endpoint even when bound. -/
theorem chainRel_lowers_from (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) (f : Ex)
    (hk : (lowerN (edgePred lw r).2 r.to).2.contains (lwKey r.frm.alias r.frm.ptr) = false)
    (hf : inlineAttrsToFilter r.frm.alias r.frm.attrs = some f) :
    f ∈ (chainRel st c lw res r).1.peel := by
  unfold chainRel
  rw [topFilters_peel _ _ _ (chainOp_peel st c lw res r)]
  simp only [lowerN] at hk ⊢
  rw [lowerInline_fresh _ _ _ _ hk, hf]; simp

theorem chainRel_lowers_to (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) (f : Ex)
    (hk : (edgePred lw r).2.contains (lwKey r.to.alias r.to.ptr) = false)
    (hf : inlineAttrsToFilter r.to.alias r.to.attrs = some f) :
    f ∈ (chainRel st c lw res r).1.peel := by
  unfold chainRel
  rw [topFilters_peel _ _ _ (chainOp_peel st c lw res r)]
  simp only [lowerN] at hk ⊢
  rw [lowerInline_fresh _ _ _ _ hk, hf]; simp

theorem chainRel_edge_pred (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) (f : Ex)
    (hfix : r.kind = .fixed) (hk : lw.contains (lwKey r.alias r.ptr) = false)
    (hf : inlineAttrsToFilter r.alias r.attrs = some f) :
    f ∈ (chainRel st c lw res r).1.peel ∧ opEmit c lw r = true := by
  have he : edgePred lw r = (some f, lw ++ [lwKey r.alias r.ptr]) := by
    simp [edgePred, isWalk, hfix, lowerInline_fresh _ _ _ _ hk, hf]
  refine ⟨?_, by simp [opEmit, he]⟩
  unfold chainRel
  rw [topFilters_peel _ _ _ (chainOp_peel st c lw res r), chainOp_isCtEi, he]
  simp [hfix]

theorem chainRel_stripped (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR)
    (hres : res.stripped = true) : (chainRel st c lw res r).1.stripped = true := by
  unfold chainRel; rw [topFilters_stripped]
  obtain ⟨h1, h2, h3⟩ := opRel_stripped r
  have hs : (scanOn (opRel r).frm res).stripped = true := scanOn_stripped _ _ (by simp [h1]) hres
  have h3' : r.kind = .fixed → (opRel r).attrs = [] := fun hk => h3 (by simp [isWalk, hk])
  unfold chainOp
  cases hk : r.kind <;> by_cases hb1 : st.bound r.frm.alias <;> by_cases hb2 : st.bound r.to.alias <;>
    by_cases h4 : r.frm.alias.id = r.to.alias.id <;>
    simp [hb1, hb2, h4, MP.stripped, MP.stripped.strippedL, h1, h2, hs, h3', hk, hres]

/-- `MATCH (a)-[:R]->(b)-[:R]->(c), (b {x:1})-[:S]->(d)`-shape: a fixed later hop leaving a bound
`b` with attrs. Before #2390 the CondTraverse kept `{x: 1}` on its pattern with no Filter
(enforced only by its per-row re-check, which #2390 also removed); now it is a Filter. -/
def bV : V := ⟨1, 0⟩
def qB : QR := { alias := ⟨5, 0⟩, named := false, frm := { alias := bV, attrs := [("x", xOne)], ptr := 9 },
                 to := { alias := ⟨4, 0⟩, ptr := 0 }, ptr := 0 }
def stB : MSt := { visited := [bV], verified := [] }

/-! ## Labels on already-bound endpoints (mod.rs:2048-2059) -/

def labelStep (acc : MSt × List Ex) (n : QN) : MSt × List Ex :=
  if acc.1.bound n.alias then
    let missing := unverified acc.1 n
    if missing.isEmpty then acc
    else (markLabels acc.1 n.alias missing, acc.2 ++ [hasLabelsFilter n.alias missing])
  else acc

def boundLabelFilters (st : MSt) (rels : List QR) : MSt × List Ex :=
  (rels.flatMap fun r => [r.frm, r.to]).foldl labelStep (st, [])

theorem labelStep_spec (acc : MSt × List Ex) (n : QN) (hb : acc.1.bound n.alias = true)
    (hm : unverified acc.1 n ≠ []) :
    labelStep acc n = (markLabels acc.1 n.alias (unverified acc.1 n),
      acc.2 ++ [hasLabelsFilter n.alias (unverified acc.1 n)]) := by
  have : (unverified acc.1 n).isEmpty = false := by
    cases h : unverified acc.1 n <;> simp_all
  simp [labelStep, hb, this]

theorem unverified_mark_self (st : MSt) (n : QN) : unverified (markLabels st n.alias (unverified st n)) n = [] := by
  unfold markLabels
  split
  · rename_i h; simpa using h
  · simp only [unverified, lookupV_cons_self]
    rw [List.filter_eq_nil_iff]
    intro l hl
    by_cases hc : (lookupV st.verified n.alias).contains l
    · simp only [List.contains_iff_mem] at hc; simp [hc]
    · simp [hl]

/-- After the filter, no label of that endpoint is missing. -/
theorem labelStep_verifies (acc : MSt × List Ex) (n : QN) (hb : acc.1.bound n.alias = true) :
    unverified (labelStep acc n).1 n = [] := by
  unfold labelStep
  simp only [hb, ↓reduceIte]
  split
  · rename_i h; simpa using h
  · exact unverified_mark_self _ _

theorem labelStep_visited (acc : MSt × List Ex) (n : QN) : (labelStep acc n).1.visited = acc.1.visited := by
  unfold labelStep markLabels
  by_cases hb : acc.1.bound n.alias
  · simp only [hb, ↓reduceIte]
    by_cases he : (unverified acc.1 n).isEmpty <;> simp [he]
  · simp [hb]

theorem boundLabelFilters_visited (st : MSt) (rels : List QR) :
    (boundLabelFilters st rels).1.visited = st.visited := by
  unfold boundLabelFilters
  generalize (rels.flatMap fun r => [r.frm, r.to]) = ns
  suffices ∀ acc : MSt × List Ex, (ns.foldl labelStep acc).1.visited = acc.1.visited from this _
  induction ns with
  | nil => intro; rfl
  | cons n ns ih => intro acc; simp only [List.foldl_cons]; rw [ih, labelStep_visited]

/-! ## Node-only components (mod.rs:2228-2315) -/

def u32max : Nat := 4294967295

def nodeOnly (st : MSt) (lw : LW) (n : QN) (paths : List (V × List V)) : MP × MSt × List Ex × LW :=
  if st.bound n.alias then
    let (af, lw) := lowerN lw n
    let bf := af.toList
    if n.labels.isEmpty then (.arg, st, bf, lw)
    else
      let rel : QR := { alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
                        frm := { alias := n.alias, ptr := 0 }, to := { alias := n.alias, labels := n.labels, ptr := 0 }, ptr := 0 }
      (.expandInto rel false [.arg], markLabels st n.alias n.labels, bf, lw)
  else
    let (af, lw) := lowerN lw n
    let res := withF af (scanOf (stripNode n))
    let st' := markLabels { st with visited := st.visited ++ [n.alias] } n.alias n.labels
    (if paths.isEmpty then res else .pathB paths res, st', [], lw)

theorem nodeOnly_bound_plain (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (hl : n.labels = []) : (nodeOnly st lw n ps).1 = .arg := by
  simp [nodeOnly, hb, hl]

/-- Extra labels on a bound node: a synthetic self-loop `ExpandInto` whose
edge id is `u32::MAX - id` (mod.rs:2246-2263) checks them. -/
theorem nodeOnly_bound_labels (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (hl : n.labels ≠ []) :
    ∃ r, (nodeOnly st lw n ps).1 = .expandInto r false [.arg] ∧
      r.alias = ⟨u32max - n.alias.id, n.alias.scope⟩ ∧ r.frm.alias = n.alias ∧ r.to.alias = n.alias ∧
      r.to.labels = n.labels ∧ r.frm.labels = [] := by
  have : n.labels.isEmpty = false := by cases h : n.labels <;> simp_all
  exact ⟨{ alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
           frm := { alias := n.alias, ptr := 0 }, to := { alias := n.alias, labels := n.labels, ptr := 0 }, ptr := 0 },
    by simp [nodeOnly, hb, this], rfl, rfl, rfl, rfl, rfl⟩

/-- A bound node's map becomes a bound filter, at most once per (alias, map). -/
theorem nodeOnly_bound_filter (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) : (nodeOnly st lw n ps).2.2.1 = (lowerN lw n).1.toList := by
  unfold nodeOnly; simp only [hb, ↓reduceIte]; split <;> rfl

theorem nodeOnly_unbound (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = false) :
    (nodeOnly st lw n ps).1.core =
        (if ps.isEmpty then scanOf (stripNode n)
         else .pathB ps (withF (lowerN lw n).1 (scanOf (stripNode n)))) ∧
      (nodeOnly st lw n ps).2.1.bound n.alias = true := by
  constructor
  · unfold nodeOnly; simp only [hb, Bool.false_eq_true, ↓reduceIte]
    split <;> simp [core_withF, MP.core, scanOf] <;> split <;> rfl
  · simp only [nodeOnly, hb, Bool.false_eq_true, ↓reduceIte]
    unfold markLabels; split <;> simp [MSt.bound]

theorem nodeOnly_stripped (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V)) :
    (nodeOnly st lw n ps).1.stripped = true := by
  unfold nodeOnly
  by_cases hb : st.bound n.alias
  · simp only [hb, ↓reduceIte]
    split <;> simp [MP.stripped, MP.stripped.strippedL]
  · simp only [hb, Bool.false_eq_true, ↓reduceIte]
    split <;> simp [MP.stripped, stripped_withF, scanOf_stripNode]

/-! ## A component with relationships -/

/-- Path elision (mod.rs:2617-2657): a named path over the single var-length
hop `(from)-[rel]->(to)` is bound by the CondVarLenTraverse itself. -/
def elidable (rels : List QR) (p : V × List V) : Bool :=
  match rels with
  | [r] => r.kind == .varLen && (p.2.map V.id == [r.frm.alias.id, r.alias.id, r.to.alias.id])
  | _ => false

/-- Bind the path variable on the first CondVarLenTraverse of that relationship
still without one (pre-order). -/
def bindPath (rid : Nat) (pv : V) : MP → Option MP
  | .cvlt r ei none cs => if r.alias.id = rid then some (.cvlt r ei (some pv) cs) else
      (bindPathL rid pv cs).map (.cvlt r ei none)
  | .filter e c => (bindPath rid pv c).map (.filter e)
  | .pathB ps c => (bindPath rid pv c).map (.pathB ps)
  | .cvlt r ei (some v) cs => (bindPathL rid pv cs).map (.cvlt r ei (some v))
  | .shortest r cs => (bindPathL rid pv cs).map (.shortest r)
  | .expandInto r e cs => (bindPathL rid pv cs).map (.expandInto r e)
  | .condTr r e cs => (bindPathL rid pv cs).map (.condTr r e)
  | .cp cs => (bindPathL rid pv cs).map .cp
  | .allScan n cs => (bindPathL rid pv cs).map (.allScan n)
  | .labelScan n cs => (bindPathL rid pv cs).map (.labelScan n)
  | _ => none
where
  bindPathL (rid : Nat) (pv : V) : List MP → Option (List MP)
    | [] => none
    | c :: cs => match bindPath rid pv c with
      | some c' => some (c' :: cs)
      | none => (bindPathL rid pv cs).map (c :: ·)

def elideStep (rels : List QR) (acc : MP × List (V × List V)) (p : V × List V) : MP × List (V × List V) :=
  if elidable rels p then
    match p.2[1]? with
    | some rv => match bindPath rv.id p.1 acc.1 with
      | some res' => (res', acc.2)
      | none => (acc.1, acc.2 ++ [p])
    | none => (acc.1, acc.2 ++ [p])
  else (acc.1, acc.2 ++ [p])

def elide (rels : List QR) (paths : List (V × List V)) (res : MP) : MP :=
  let x := paths.foldl (elideStep rels) (res, [])
  if x.2.isEmpty then x.1 else .pathB x.2 x.1

theorem elide_fold (rels : List QR) (paths : List (V × List V)) (acc : MP × List (V × List V))
    (h : ∀ p ∈ paths, elidable rels p = false) :
    paths.foldl (elideStep rels) acc = (acc.1, acc.2 ++ paths) := by
  induction paths generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, elideStep, h p (by simp), Bool.false_eq_true, ↓reduceIte]
    rw [ih _ (fun q hq => h q (by simp [hq]))]; simp

/-- No elidable path: one PathBuilder with all of them (none: nothing). -/
theorem elide_none (rels : List QR) (paths : List (V × List V)) (res : MP)
    (h : ∀ p ∈ paths, elidable rels p = false) :
    elide rels paths res = if paths.isEmpty then res else .pathB paths res := by
  simp [elide, elide_fold rels paths _ h]

/-- The single var-length hop's own path is bound on the traverse, no PathBuilder. -/
theorem elide_varlen (r : QR) (p : V × List V) (res res' : MP) (rv : V)
    (he : elidable [r] p = true) (h1 : p.2[1]? = some rv) (hb : bindPath rv.id p.1 res = some res') :
    elide [r] [p] res = res' := by
  simp [elide, elideStep, he, h1, hb]

def planRels (c : Comp) : List QR → MP → MSt → LW → MP × MSt × LW
  | [], res, st, lw => (res, st, lw)
  | r :: rs, res, st, lw =>
    let x := chainRel st c lw res r
    planRels c rs x.1 (visit st r) x.2

/-- The `HopShape` closure (mod.rs:2218-2227): nodes are `(id, scope_id)`; `self_loop`
compares ids only; `fixed` = neither var-length nor shortest. -/
def hopShape (r : QR) : HopOrder.HopShape V :=
  { frm := r.frm.alias, to := r.to.alias, selfLoop := r.frm.alias.id == r.to.alias.id,
    fixed := r.kind == .fixed }

/-! ## The promoted walk (#2558, PR #3593: mod.rs:2063-2217) -/

/-- `is_constrained` (mod.rs:2067-2069): inline attributes, or a WHERE anchor. -/
def isConstrained (wi : WI) (n : QN) : Bool := !n.attrs.isEmpty || wi.anchors.contains n.alias

/-- `anchored` (mod.rs:2070-2072). -/
def anchored (wi : WI) (r : QR) : Bool := isConstrained wi r.frm || isConstrained wi r.to

/-- `references_variable` (mod.rs:2099-2104): the edge's or an endpoint's attribute map reads a
variable (the map's variables are those of its values). -/
def refsVar (r : QR) : Bool :=
  [r.attrs, r.frm.attrs, r.to.attrs].any fun m => m.any fun p => !(scopedVars p.2).isEmpty

/-- `where_visible` (mod.rs:2112-2113): the only component of the clause, the WHERE holds no
pattern (`wi.single`, set by `planFrom`), and the component is a tree. -/
def whereVisible (wi : WI) (c : Comp) : Bool := wi.single && c.rels.length + 1 == c.nodes.length

/-- `scan_score` (mod.rs:2114-2120). -/
def scanScore (wi : WI) (c : Comp) (n : QN) : Nat :=
  endpointScanScore (!n.attrs.isEmpty || (whereVisible wi c && wi.fv.contains n.alias)) (!n.labels.isEmpty)

/-- `is_anchored_end` (mod.rs:2121-2124). -/
def anchoredEnd (wi : WI) (c : Comp) (n : QN) : Bool :=
  !n.attrs.isEmpty || (whereVisible wi c && wi.anchors.contains n.alias)

/-- `leads_from_anchor` (mod.rs:2125-2132): the end `select_var_len_scan_node` will scan is anchored. -/
def leadsFromAnchor (wi : WI) (c : Comp) (r : QR) : Bool :=
  anchoredEnd wi c (if walkScansTo (scanScore wi c r.frm) (scanScore wi c r.to) then r.to else r.frm)

/-- `leading_walk` (mod.rs:2093-2152): nothing bound before the clause, no fixed hop anchored, every
hop free to move (no allShortestPaths, no attribute map reading a variable); then the first
var-length hop with distinct end ids that leads from an anchor. -/
def leadingWalk (wi : WI) (c : Comp) : Option V :=
  let leads := !wi.boundBefore && !(c.rels.any fun r => r.kind == .fixed && anchored wi r)
  let canMove := c.rels.all fun r => r.kind != .shortest && !refsVar r
  if leads && canMove then
    (c.rels.find? fun r => r.kind == .varLen && r.frm.alias.id != r.to.alias.id && leadsFromAnchor wi c r).map
      (·.alias)
  else none

/-- **PROVEN**: a promoted walk is a var-length hop of the component with distinct end ids, leading
from an anchored end, in a clause nothing was bound before, where no fixed hop is anchored and every
hop can move. -/
theorem leadingWalk_spec (wi : WI) (c : Comp) (x : V) (h : leadingWalk wi c = some x) :
    ∃ r ∈ c.rels, r.alias = x ∧ r.kind = .varLen ∧ r.frm.alias.id ≠ r.to.alias.id ∧
      leadsFromAnchor wi c r = true ∧ wi.boundBefore = false ∧
      (∀ q ∈ c.rels, q.kind = .fixed → anchored wi q = false) ∧
      (∀ q ∈ c.rels, q.kind ≠ .shortest ∧ refsVar q = false) := by
  unfold leadingWalk at h
  simp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.any_eq_false, List.all_eq_true,
      Bool.and_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true'] at hc
    obtain ⟨⟨hb, hf⟩, hm⟩ := hc
    obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 h
    have hmem := List.mem_of_find?_eq_some hr
    have hp := List.find?_some hr
    simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hp
    refine ⟨r, hmem, rfl, hp.1.1, hp.1.2, hp.2, hb, fun q hq hk => ?_, fun q hq => ?_⟩
    · have := hf q hq; simp only [hk, beq_self_eq_true, Bool.true_and] at this; simpa using this
    · have := hm q hq; simpa using this
  · simp at h

/-! ## Outward order (mod.rs:2191-2217) -/

/-- The two endpoints of a hop (what the Rust inserts into `bound`). -/
def ends (r : QR) : List V := [r.frm.alias, r.to.alias]

/-- The hop touches the bound set. -/
def touchesB (b : List V) (r : QR) : Bool := b.contains r.frm.alias || b.contains r.to.alias

/-- `position(p)` then `remove`: the first element satisfying `p`, and the rest in order. -/
def pickFirst (p : QR → Bool) : List QR → Option (QR × List QR)
  | [] => none
  | r :: rs => if p r then some (r, rs) else (pickFirst p rs).map fun x => (x.1, r :: x.2)

theorem pickFirst_some (p : QR → Bool) : ∀ (l : List QR) (a : QR) (l' : List QR),
    pickFirst p l = some (a, l') → p a = true ∧ (a :: l').Perm l
  | [], _, _, h => by simp [pickFirst] at h
  | r :: rs, a, l', h => by
    unfold pickFirst at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
      exact ⟨by assumption, List.Perm.refl _⟩
    · obtain ⟨⟨b, bs⟩, hb, he⟩ := Option.map_eq_some_iff.1 h
      simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      obtain ⟨h1, h2⟩ := pickFirst_some p rs b bs hb
      exact ⟨h1, (List.Perm.swap r b bs).trans (List.Perm.cons r h2)⟩

theorem pickFirst_none (p : QR → Bool) : ∀ (l : List QR), pickFirst p l = none → ∀ q ∈ l, p q = false
  | [], _, q, hq => by simp at hq
  | r :: rs, h, q, hq => by
    unfold pickFirst at h
    split at h
    · simp at h
    · rename_i hr
      have := pickFirst_none p rs (by simpa using h)
      rcases List.mem_cons.1 hq with rfl | hq
      · simpa using hr
      · exact this q hq

theorem pickFirst_isSome (p : QR → Bool) : ∀ (l : List QR), (∃ q ∈ l, p q = true) →
    ∃ a l', pickFirst p l = some (a, l')
  | l, ⟨q, hq, hp⟩ => by
    cases h : pickFirst p l with
    | none => have := pickFirst_none p l h q hq; simp_all
    | some x => exact ⟨x.1, x.2, rfl⟩

/-- The `position(..).or_else(|| (!rest.is_empty()).then_some(0))` step: the first hop touching the
bound set, else the first hop left. -/
def pickNext (b : List V) (rest : List QR) : Option (QR × List QR) :=
  match pickFirst (touchesB b) rest with
  | some x => some x
  | none => match rest with
    | [] => none
    | r :: rs => some (r, rs)

theorem pickNext_some (b : List V) (rest : List QR) (a : QR) (l' : List QR)
    (h : pickNext b rest = some (a, l')) :
    (a :: l').Perm rest ∧ (touchesB b a = true ∨ ∀ q ∈ a :: l', touchesB b q = false) := by
  unfold pickNext at h
  split at h
  · rename_i x hx
    simp only [Option.some.injEq] at h; subst h
    exact ⟨(pickFirst_some _ _ _ _ hx).2, Or.inl (pickFirst_some _ _ _ _ hx).1⟩
  · rename_i hx
    split at h
    · simp at h
    · rename_i r rs
      simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h
      exact ⟨List.Perm.refl _, Or.inr (pickFirst_none _ _ hx)⟩

theorem pickNext_none (b : List V) (rest : List QR) (h : pickNext b rest = none) : rest = [] := by
  unfold pickNext at h
  split at h
  · simp at h
  · split at h
    · rfl
    · simp at h

/-- The `while let` loop: emit `r`, add its ends to `bound`, pick the next hop. Fuel = the hops
left (each turn removes one). -/
def outwardGo : Nat → List V → QR → List QR → List QR
  | 0, _, r, rest => r :: rest
  | n + 1, b, r, rest =>
    r :: match pickNext (b ++ ends r) rest with
      | none => []
      | some x => outwardGo n (b ++ ends r) x.1 x.2

/-- `if let Some(lead) = leading_walk { … }` (mod.rs:2191-2217): the promoted walk first, then
outward. (`expect`: the walk is one of the component's hops — `leadingWalk_spec`.) -/
def leadOrder (lead : Option V) (rs : List QR) : List QR :=
  match lead with
  | none => rs
  | some x => match pickFirst (fun r => r.alias == x) rs with
    | none => rs
    | some y => outwardGo y.2.length [] y.1 y.2

theorem outwardGo_perm : ∀ (n : Nat) (b : List V) (r : QR) (rest : List QR),
    (outwardGo n b r rest).Perm (r :: rest)
  | 0, _, _, _ => List.Perm.refl _
  | n + 1, b, r, rest => by
    unfold outwardGo
    apply List.Perm.cons
    split
    · rename_i h; rw [pickNext_none _ _ h]
    · rename_i x h
      exact (outwardGo_perm n _ x.1 x.2).trans (pickNext_some _ _ _ _ h).1

theorem leadOrder_perm (lead : Option V) (rs : List QR) : (leadOrder lead rs).Perm rs := by
  unfold leadOrder
  split
  · exact List.Perm.refl _
  · split
    · exact List.Perm.refl _
    · rename_i y h
      exact (outwardGo_perm _ _ _ _).trans (pickFirst_some _ _ _ _ h).2

/-- Walking outward from `b`: every hop touches what the hops before it bound, unless no hop left
does (then the next one starts afresh: a component left disconnected by the walk). -/
inductive Outward : List V → List QR → Prop
  | nil (b : List V) : Outward b []
  | next (b : List V) (r : QR) (rs : List QR) :
      (touchesB b r = true ∨ ∀ q ∈ r :: rs, touchesB b q = false) →
      Outward (b ++ ends r) rs → Outward b (r :: rs)

theorem outwardGo_outward : ∀ (n : Nat) (b : List V) (r : QR) (rest : List QR), rest.length ≤ n →
    (touchesB b r = true ∨ ∀ q ∈ r :: rest, touchesB b q = false) →
    Outward b (outwardGo n b r rest)
  | 0, b, r, rest, hl, hc => by
    have : rest = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    exact .next b r [] hc (.nil _)
  | n + 1, b, r, rest, hl, hc => by
    unfold outwardGo
    refine .next b r _ ?_ ?_
    · rcases hc with hc | hc
      · exact Or.inl hc
      · refine Or.inr fun q hq => ?_
        rcases List.mem_cons.1 hq with rfl | hq
        · exact hc q (by simp)
        · split at hq
          · simp at hq
          · rename_i x h
            have hp := (pickNext_some _ _ _ _ h).1
            have hq' := (outwardGo_perm n _ x.1 x.2).mem_iff.1 hq
            exact hc q (List.mem_cons_of_mem _ (hp.mem_iff.1 hq'))
    · split
      · exact .nil _
      · rename_i x h
        obtain ⟨hp, ht⟩ := pickNext_some _ _ _ _ h
        have := hp.length_eq
        simp only [List.length_cons] at this
        exact outwardGo_outward n _ x.1 x.2 (by omega) ht

theorem outwardGo_head (n : Nat) (b : List V) (r : QR) (rest : List QR) :
    ∃ t, outwardGo n b r rest = r :: t := by
  cases n <;> exact ⟨_, rfl⟩

/-- **PROVEN** (#3593): with a promoted walk that is one of the hops, the order starts with it and
walks outward from its ends; a permutation either way (`leadOrder_perm`). -/
theorem leadOrder_spec (x : V) (rs : List QR) (h : ∃ r ∈ rs, r.alias = x) :
    ∃ r t, leadOrder (some x) rs = r :: t ∧ r.alias = x ∧ Outward [] (r :: t) := by
  obtain ⟨r0, hr0, hx⟩ := h
  obtain ⟨a, l', ha⟩ := pickFirst_isSome (fun r => r.alias == x) rs ⟨r0, hr0, by simp [hx]⟩
  have hp := (pickFirst_some _ _ _ _ ha).1
  obtain ⟨t, ht⟩ := outwardGo_head l'.length [] a l'
  refine ⟨a, t, ?_, by simpa using hp, ?_⟩
  · simp only [leadOrder, ha, ht]
  · rw [← ht]
    exact outwardGo_outward _ _ _ _ (Nat.le_refl _) (Or.inr fun q _ => by simp [touchesB])

/-- Without a promoted walk, the order is the sort's. -/
theorem leadOrder_none (rs : List QR) : leadOrder none rs = rs := rfl

/-- `sort_by_key`, the promoted walk's outward reorder, then
`self_loops_after_binders(…, |node| self.visited.contains(node))` (mod.rs:2152-2227, #3102, #3593). -/
def orderRels (st : MSt) (fv : List V) (lead : Option V) (rs : List QR) : List QR :=
  HopOrder.selfLoopsAfterBinders (leadOrder lead (sortRels st fv lead rs)) hopShape st.bound

theorem orderRels_perm (st : MSt) (fv : List V) (lead : Option V) (rs : List QR) :
    (orderRels st fv lead rs).Perm rs :=
  (HopOrder.selfLoops_perm _ _ _).trans ((leadOrder_perm _ _).trans (sortRels_spec st fv lead rs).1)

/-- The planner discharges `HopOrder.WF` when each alias id names one variable (`(id, scope)`
pairs with equal ids are equal) — so on such patterns `orderRels` is the pre-#3102 loop over the
(#3593: lead-first) sorted hops. -/
theorem orderRels_eq_reference (st : MSt) (fv : List V) (lead : Option V) (rs : List QR)
    (hid : ∀ r ∈ rs, r.frm.alias.id = r.to.alias.id → r.frm.alias = r.to.alias) :
    orderRels st fv lead rs =
      HopOrder.pre3102_reference hopShape st.bound (leadOrder lead (sortRels st fv lead rs)) := by
  apply HopOrder.eq_reference
  intro x hx hl
  have := hid x (((leadOrder_perm _ _).trans (sortRels_spec st fv lead rs).1).mem_iff.1 hx)
  simp [hopShape] at hl ⊢
  exact this hl

/-- **PROVEN** (#3593): when `plan_match` promotes a walk, its component's hops start with that walk
and the rest follow outward from it (before `self_loops_after_binders`, which moves only fixed
self-loops). -/
theorem planOrder_lead (st : MSt) (wi : WI) (c : Comp) (x : V) (h : leadingWalk wi c = some x) :
    ∃ r t, leadOrder (leadingWalk wi c) (sortRels st wi.fv (leadingWalk wi c) c.rels) = r :: t ∧
      r.alias = x ∧ Outward [] (r :: t) := by
  obtain ⟨r, hr, hx, -⟩ := leadingWalk_spec wi c x h
  rw [h]
  have hm : ∃ q ∈ sortRels st wi.fv (some x) c.rels, q.alias = x :=
    ⟨r, (sortRels_spec st wi.fv (some x) c.rels).1.mem_iff.2 hr, hx⟩
  exact leadOrder_spec x _ hm

def planComp (st : MSt) (wi : WI) (lw : LW) (c : Comp) : MP × MSt × List Ex × LW :=
  let lb := boundLabelFilters st c.rels
  match orderRels lb.1 wi.fv (leadingWalk wi c) c.rels with
  | [] => match c.nodes with
    | n :: _ => let x := nodeOnly lb.1 lw n c.paths; (x.1, x.2.1, lb.2 ++ x.2.2.1, x.2.2.2)
    | [] => (.arg, lb.1, lb.2, lw)
  | r :: rs =>
    let f := firstRel lb.1 c lw r
    let x := planRels c rs f.1 (visit lb.1 r) f.2
    (elide c.rels c.paths x.1, x.2.1, lb.2, x.2.2)

theorem visit_binds (st : MSt) (r : QR) (v : V) (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) :
    (visit st r).bound v = true := by
  unfold visit markLabels
  split <;> split <;> simp [MSt.bound] <;> rcases hv with rfl | rfl | rfl <;> simp

theorem visit_mono (st : MSt) (r : QR) (v : V) (h : st.bound v = true) : (visit st r).bound v = true := by
  have h' : v ∈ st.visited := by simpa [MSt.bound] using h
  unfold visit markLabels
  split <;> split <;> simp [MSt.bound, h']

theorem planRels_mono (c : Comp) (rs : List QR) (res : MP) (st : MSt) (lw : LW) (v : V)
    (h : st.bound v = true) : (planRels c rs res st lw).2.1.bound v = true := by
  induction rs generalizing res st lw with
  | nil => exact h
  | cons r rs ih => simp only [planRels]; exact ih _ _ _ (visit_mono st r v h)

theorem planRels_binds (c : Comp) (rs : List QR) (res : MP) (st : MSt) (lw : LW) (r : QR) (hr : r ∈ rs)
    (v : V) (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) :
    (planRels c rs res st lw).2.1.bound v = true := by
  induction rs generalizing res st lw with
  | nil => simp at hr
  | cons r' rs ih =>
    simp only [planRels]
    rcases List.mem_cons.1 hr with rfl | hr
    · exact planRels_mono c rs _ _ _ v (visit_binds st _ v hv)
    · exact ih _ _ _ hr

/-- After a component with relationships, every endpoint and every edge alias
is bound for the clauses that follow (mod.rs:2490-2497, 2610-2616). -/
theorem planComp_binds (st : MSt) (wi : WI) (lw : LW) (c : Comp) (r : QR) (hr : r ∈ c.rels) (v : V)
    (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) : (planComp st wi lw c).2.1.bound v = true := by
  have hp := orderRels_perm (boundLabelFilters st c.rels).1 wi.fv (leadingWalk wi c) c.rels
  cases hs : orderRels (boundLabelFilters st c.rels).1 wi.fv (leadingWalk wi c) c.rels with
  | nil => rw [hs] at hp; have := hp.mem_iff.2 hr; simp at this
  | cons r0 rs =>
    have e : (planComp st wi lw c).2.1 =
        (planRels c rs (firstRel (boundLabelFilters st c.rels).1 c lw r0).1
          (visit (boundLabelFilters st c.rels).1 r0) (firstRel (boundLabelFilters st c.rels).1 c lw r0).2).2.1 := by
      unfold planComp; simp only [hs]
    rw [e]
    rw [hs] at hp
    rcases List.mem_cons.1 (hp.mem_iff.2 hr) with rfl | hr'
    · exact planRels_mono _ _ _ _ _ v (visit_binds _ _ v hv)
    · exact planRels_binds _ _ _ _ _ r hr' v hv

/-! ### Every operator `plan_match` emits carries a stripped pattern -/

mutual
theorem bindPath_stripped (rid : Nat) (pv : V) : (m m' : MP) → bindPath rid pv m = some m' →
    m'.stripped = m.stripped
  | .cvlt r ei none cs, m', h => by
    simp only [bindPath] at h
    split at h
    · cases h; simp [MP.stripped]
    · obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
      simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .cvlt r ei (some v) cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .filter e c, m', h => by
    simp only [bindPath] at h
    obtain ⟨c', hc, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPath_stripped rid pv c c' hc]
  | .pathB ps c, m', h => by
    simp only [bindPath] at h
    obtain ⟨c', hc, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPath_stripped rid pv c c' hc]
  | .shortest r cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .expandInto r e cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .condTr r e cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .cp cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .arg, m', h => by simp [bindPath] at h
  | .allScan n cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]
  | .labelScan n cs, m', h => by
    simp only [bindPath] at h
    obtain ⟨cs', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
    simp [MP.stripped, bindPathL_stripped rid pv cs cs' hcs]

theorem bindPathL_stripped (rid : Nat) (pv : V) : (cs cs' : List MP) →
    bindPath.bindPathL rid pv cs = some cs' → MP.stripped.strippedL cs' = MP.stripped.strippedL cs
  | [], _, h => by simp [bindPath.bindPathL] at h
  | c :: cs, cs', h => by
    simp only [bindPath.bindPathL] at h
    split at h
    · rename_i c' hc; cases h
      simp [MP.stripped.strippedL, bindPath_stripped rid pv c c' hc]
    · obtain ⟨cs'', hcs, rfl⟩ := Option.map_eq_some_iff.1 h
      simp [MP.stripped.strippedL, bindPathL_stripped rid pv cs cs'' hcs]
end

theorem elide_stripped (rels : List QR) (paths : List (V × List V)) (res : MP) (h : res.stripped = true) :
    (elide rels paths res).stripped = true := by
  have hf : ∀ (ps : List (V × List V)) (acc : MP × List (V × List V)), acc.1.stripped = true →
      (ps.foldl (elideStep rels) acc).1.stripped = true := by
    intro ps
    induction ps with
    | nil => intro acc h; exact h
    | cons p ps ih =>
      intro acc ha
      simp only [List.foldl_cons]
      apply ih
      unfold elideStep
      split
      · split
        · rename_i rv _
          split
          · rename_i res' hb; rw [bindPath_stripped _ _ _ _ hb]; exact ha
          · exact ha
        · exact ha
      · exact ha
  have := hf paths (res, []) h
  rw [elide]
  split <;> simp [MP.stripped, this]

theorem planRels_stripped (c : Comp) (rs : List QR) (res : MP) (st : MSt) (lw : LW)
    (h : res.stripped = true) : (planRels c rs res st lw).1.stripped = true := by
  induction rs generalizing res st lw with
  | nil => exact h
  | cons r rs ih => simp only [planRels]; exact ih _ _ _ (chainRel_stripped _ _ _ _ _ h)

/-- **Planner invariant (#2390)**: no scan or fixed-length operator of a component's plan carries
an inline map, and a walk carries only its edge's. The predicates live in Filters. -/
theorem planComp_stripped (st : MSt) (wi : WI) (lw : LW) (c : Comp) :
    (planComp st wi lw c).1.stripped = true := by
  unfold planComp
  dsimp only
  cases orderRels (boundLabelFilters st c.rels).1 wi.fv (leadingWalk wi c) c.rels with
  | nil =>
    cases c.nodes with
    | nil => rfl
    | cons n _ => exact nodeOnly_stripped _ _ _ _
  | cons r rs => exact elide_stripped _ _ _ (planRels_stripped _ _ _ _ _ (firstRel_stripped _ _ _ _))

/-! ## The whole pattern -/

/-- The clause-level facts `plan_match` adds (mod.rs:1999-2687): `bound_before_clause` (anything
bound before the component loop) and `single_component` (one component, no pattern in the WHERE). -/
def WI.forClause (wi : WI) (st : MSt) (comps : List Comp) : WI :=
  { wi with boundBefore := !st.visited.isEmpty, single := comps.length == 1 && !wi.patternWhere }

/-- The component loop from the lifted filters `pre` (`bound_filters`' initial content). -/
def planFrom (st : MSt) (wi : WI) (comps : List Comp) (pf : Option (MP → MP)) (pre : List Ex) : MP × MSt :=
  let wc := wi.forClause st comps
  let step := fun (acc : List MP × MSt × List Ex × LW) (c : Comp) =>
    let x := planComp acc.2.1 wc acc.2.2.2 c
    (acc.1 ++ [x.1], x.2.1, acc.2.2.1 ++ x.2.2.1, x.2.2.2)
  let x := comps.foldl step ([], st, pre, [])
  let res := match x.1 with
    | [p] => p
    | ps => .cp ps
  let res := match pf with
    | some f => f res
    | none => res
  let res := match x.2.2.1 with
    | [] => res
    | fs => .filter (andOf fs) res   -- `ExprIR::and_of(bound_filters)` (#3335)
  (res, x.2.1)

/-- `plan_match`: lift the cross-entity inline attrs (#2981, mod.rs:1999-2687), then plan the
rewritten components with the lifted filters first in `bound_filters`. -/
def planMatch (st : MSt) (wi : WI) (comps : List Comp) (pf : Option (MP → MP)) : MP × MSt :=
  planFrom st wi (liftPattern st comps).1 pf (liftPattern st comps).2

/-- With nothing to lift, `plan_match` is the component loop on the pattern as written. -/
theorem planMatch_nolift (st : MSt) (wi : WI) (comps : List Comp) (pf : Option (MP → MP))
    (h : liftPattern st comps = (comps, [])) : planMatch st wi comps pf = planFrom st wi comps pf [] := by
  simp [planMatch, h]

/-! ### #2923 / #2981: an inline map that reads another component -/

section lift2923
set_option linter.unusedSimpArgs false

def v2923b : V := ⟨1, 0⟩
def v2923c : V := ⟨2, 0⟩
/-- `b.v` -/
def bDotV : Ex := .node (.prop "v") [Ex.v v2923b]
/-- `MATCH (b), (c {v: b.v})`: two components. -/
def q2923c : Comp := { nodes := [{ alias := v2923c, attrs := [("v", bDotV)], ptr := 1 }], rels := [] }
def q2923 : List Comp := [{ nodes := [{ alias := v2923b, ptr := 0 }], rels := [] }, q2923c]

/-- Before #2981 (the components planned as written): `c.v = b.v` is a Filter inside the
second CartesianProduct branch, where `b` is not bound, so no row passes. -/
theorem q2923_unlifted :
    (planFrom ⟨[], []⟩ {} q2923 none []).1 =
      .cp [.allScan { alias := v2923b, ptr := 0 } [],
           .filter (andOf [attrEq v2923c ("v", bDotV)]) (.allScan { alias := v2923c, ptr := 1 } [])] := by
  simp [planFrom, planComp, q2923, q2923c, boundLabelFilters, orderRels, sortRels, leadingWalk, leadOrder, WI.forClause, HopOrder.selfLoopsAfterBinders, nodeOnly, lowerN, lowerInline, lwKey, MSt.bound, withF, scanOf, stripNode, inlineAttrsToFilter, markLabels, v2923b, v2923c, rewriteComp, keepNode, keepAttrs, readsPending, bDotV, nodes, nodesL, Ex.v]

/-- Since #2981 the entry is lifted above the product, where both are bound. -/
theorem q2923_lifted :
    (planMatch ⟨[], []⟩ {} q2923 none).1 =
      .filter (attrEq v2923c ("v", bDotV))
        (.cp [.allScan { alias := v2923b, ptr := 0 } [], .allScan { alias := v2923c, ptr := 1 } []]) := by
  simp only [planMatch, liftPattern, pendingOf, patVars, liftOrder]
  simp [planFrom, planComp, q2923, q2923c, boundLabelFilters, orderRels, sortRels, leadingWalk, leadOrder, WI.forClause, HopOrder.selfLoopsAfterBinders, nodeOnly, lowerN, lowerInline, lwKey, MSt.bound, withF, scanOf, stripNode, inlineAttrsToFilter, markLabels, v2923b, v2923c, rewriteComp, keepNode, keepAttrs, readsPending, bDotV, nodes, nodesL, Ex.v,
    liftItems, liftItem]
  simp only [andOf]
  rw [flat_single _ _ _ (by decide), flat_single _ _ _ (by decide)]

/-- With `b` bound by an earlier clause the entry is not lifted: it reads no pending variable. -/
theorem q2923_bound_not_lifted :
    liftPattern ⟨[v2923b], []⟩ [q2923c] = ([q2923c], []) := by
  rfl

/-- `MATCH (c), (a)-[r:R {w: c.v}]->(b)` with `r` anonymous: the edge's entry is lifted
and the edge is emitted (`lifted_rels`), so the Filter sees every parallel edge. -/
def r2981 : QR :=
  { alias := ⟨5, 0⟩, named := false, frm := { alias := ⟨3, 0⟩, ptr := 0 }, to := { alias := ⟨4, 0⟩, ptr := 0 },
    attrs := [("w", .node (.prop "v") [Ex.v v2923c])], ptr := 2 }
def q2981c : Comp := { nodes := [r2981.frm, r2981.to], rels := [r2981] }
def q2981rel : List Comp := [{ nodes := [{ alias := v2923c, ptr := 0 }], rels := [] }, q2981c]

theorem lifted_rel_emitted :
    let L := liftPattern ⟨[], []⟩ q2981rel
    L.2 = [andOf [attrEq r2981.alias ("w", .node (.prop "v") [Ex.v v2923c])]] ∧
    ((L.1.getD 1 q2981c).rels.map fun r => (r.attrs.length, emitRel (L.1.getD 1 q2981c) r)) = [(0, true)] ∧
    emitRel q2981c r2981 = false := by
  refine ⟨rfl, rfl, rfl⟩

end lift2923

/-- One component: its plan; several: a CartesianProduct of them in order
(mod.rs:2662-2666); the lifted (#2981) then bound-label/attr filters on top (mod.rs:2678-2684). -/
theorem planMatch_join (st : MSt) (wi : WI) (c : Comp) :
    (planMatch st wi [c] none).1 =
      let c' := rewriteComp (pendingOf st [c]) c
      match (liftPattern st [c]).2 ++ (planComp st (wi.forClause st [c']) [] c').2.2.1 with
      | [] => (planComp st (wi.forClause st [c']) [] c').1
      | fs => .filter (andOf fs) (planComp st (wi.forClause st [c']) [] c').1 := by
  simp [planMatch, planFrom, liftPattern]

/-- The same, for a component with nothing to lift (the pre-#2981 statement). -/
theorem planMatch_join_nolift (st : MSt) (wi : WI) (c : Comp)
    (hl : liftPattern st [c] = ([c], [])) :
    (planMatch st wi [c] none).1 =
      match (planComp st (wi.forClause st [c]) [] c).2.2.1 with
      | [] => (planComp st (wi.forClause st [c]) [] c).1
      | fs => .filter (andOf fs) (planComp st (wi.forClause st [c]) [] c).1 := by
  rw [planMatch_nolift _ _ _ _ hl]; simp [planFrom]

theorem planMatch_cp (st : MSt) (wi : WI) (c1 c2 : Comp)
    (hl : liftPattern st [c1, c2] = ([c1, c2], []))
    (h1 : (planComp st (wi.forClause st [c1, c2]) [] c1).2.2.1 = [])
    (h2 : (planComp (planComp st (wi.forClause st [c1, c2]) [] c1).2.1 (wi.forClause st [c1, c2]) (planComp st (wi.forClause st [c1, c2]) [] c1).2.2.2 c2).2.2.1 = []) :
    (planMatch st wi [c1, c2] none).1 =
      .cp [(planComp st (wi.forClause st [c1, c2]) [] c1).1, (planComp (planComp st (wi.forClause st [c1, c2]) [] c1).2.1 (wi.forClause st [c1, c2]) (planComp st (wi.forClause st [c1, c2]) [] c1).2.2.2 c2).1] := by
  rw [planMatch_nolift _ _ _ _ hl]; simp [planFrom, h1, h2]

/-- The whole MATCH plan (without the WHERE, a parameter) is stripped. -/
theorem planMatch_stripped (st : MSt) (wi : WI) (comps : List Comp) :
    (planMatch st wi comps none).1.stripped = true := by
  have hf : ∀ (w : WI) (cs : List Comp) (acc : List MP × MSt × List Ex × LW),
      MP.stripped.strippedL acc.1 = true →
      MP.stripped.strippedL (cs.foldl (fun (acc : List MP × MSt × List Ex × LW) (c : Comp) =>
        let x := planComp acc.2.1 w acc.2.2.2 c
        (acc.1 ++ [x.1], x.2.1, acc.2.2.1 ++ x.2.2.1, x.2.2.2)) acc).1 = true := by
    intro w cs
    induction cs with
    | nil => intro acc h; exact h
    | cons c cs ih =>
      intro acc h
      simp only [List.foldl_cons]
      apply ih
      have hl : ∀ (l : List MP) (m : MP), MP.stripped.strippedL l = true → m.stripped = true →
          MP.stripped.strippedL (l ++ [m]) = true := by
        intro l m; induction l with
        | nil => intro _ hm; simp [MP.stripped.strippedL, hm]
        | cons a l ihl =>
          intro hl hm
          simp only [List.cons_append, MP.stripped.strippedL, Bool.and_eq_true] at hl ⊢
          exact ⟨hl.1, ihl hl.2 hm⟩
      exact hl _ _ h (planComp_stripped _ _ _ _)
  simp only [planMatch]
  generalize (liftPattern st comps).2 = pre
  generalize (liftPattern st comps).1 = comps
  have h0 := hf (wi.forClause st comps) comps ([], st, pre, []) rfl
  simp only [planFrom]
  generalize (comps.foldl _ ([], st, pre, [])) = x at h0 ⊢
  have hr : (match x.1 with | [p] => p | ps => MP.cp ps).stripped = true := by
    split
    · rename_i p heq; rw [heq] at h0; simpa [MP.stripped.strippedL] using h0
    · simpa [MP.stripped] using h0
  split <;> simp [MP.stripped, hr]

/-! ## `build_pattern_sub_plan` (mod.rs:1226-1237) -/

def buildPatternSubPlan (st : MSt) (wi : WI) (comps : List Comp) (addArgs : MP → MP) : MP × MSt :=
  ((addArgs (planMatch st wi comps none).1), st)

/-- The sub-plan of a pattern predicate leaves `visited` and `verified_labels`
as they were: its variables are not bound for the clauses that follow. -/
theorem buildPatternSubPlan_restores (st : MSt) (wi : WI) (comps : List Comp) (aa : MP → MP) :
    (buildPatternSubPlan st wi comps aa).2 = st ∧
    (buildPatternSubPlan st wi comps aa).1 = aa (planMatch st wi comps none).1 := ⟨rfl, rfl⟩

/-! ### #3340: the `Argument` payload (`argument_vars`, mod.rs:1221-1223)

The runtime ignores the payload (`IR::Argument(_)`), so `addArgs` and the row
semantics are unchanged; only `select_scan_node` reads it. Here: the payload
is the outer row's bound variables, read with the pattern's own variables not
yet marked visited. -/

/-- `argument_vars`: everything visited so far. -/
def argumentVars (st : MSt) : List V := st.visited

theorem argumentVars_spec (st : MSt) (v : V) : v ∈ argumentVars st ↔ st.bound v = true := by
  simp [argumentVars, MSt.bound]

/-- `build_pattern_sub_plan` with the payload: `add_argument_to_leaves(plan, Some(argument_vars()))`
after `visited` is restored. -/
def buildPatternSubPlanV (st : MSt) (wi : WI) (comps : List Comp) (aa : List V → MP → MP) : MP × MSt :=
  (aa (argumentVars st) (planMatch st wi comps none).1, st)

/-- The sub-plan's taps carry exactly the outer bound variables (not the pattern's
own, which `plan_match` marks visited), and `visited`/`verified_labels` are restored. -/
theorem buildPatternSubPlanV_payload (st : MSt) (wi : WI) (comps : List Comp) (aa : List V → MP → MP) :
    (buildPatternSubPlanV st wi comps aa).2 = st ∧
    (buildPatternSubPlanV st wi comps aa).1 = aa st.visited (planMatch st wi comps none).1 ∧
    (buildPatternSubPlanV st wi comps (fun _ => aa [])).1 =
      (buildPatternSubPlan st wi comps (aa [])).1 := ⟨rfl, rfl, rfl⟩


/-! ### #2558: `MATCH (a)-[:R]->(b)-[*]->(c {id: …})` — the walk on the anchored `c` leads -/

def v2558a : V := ⟨0, 0⟩
def v2558b : V := ⟨1, 0⟩
def v2558c : V := ⟨2, 0⟩
def n2558a : QN := { alias := v2558a, ptr := 0 }
def n2558b : QN := { alias := v2558b, ptr := 0 }
def n2558c : QN := { alias := v2558c, attrs := [("id", tt)], ptr := 1 }
def r2558 : QR := { alias := ⟨3, 0⟩, named := false, frm := n2558a, to := n2558b, ptr := 0 }
def w2558 : QR := { alias := ⟨4, 0⟩, named := false, frm := n2558b, to := n2558c, kind := .varLen, ptr := 0 }
def q2558 : Comp := { nodes := [n2558a, n2558b, n2558c], rels := [r2558, w2558] }

/-- **PROVEN**: with nothing bound and no WHERE, the var-length hop into the inline-constrained
`c` is promoted, and the plan orders it before the fixed hop (which then touches `b`). -/
theorem ex2558_walk_leads :
    leadingWalk (({} : WI).forClause ⟨[], []⟩ [q2558]) q2558 = some w2558.alias ∧
    orderRels ⟨[], []⟩ [] (some w2558.alias) q2558.rels = [w2558, r2558] := by
  constructor
  · decide
  · have hs : sortRels ⟨[], []⟩ [] (some w2558.alias) q2558.rels = [w2558, r2558] := by
      simp [sortRels, q2558, relKey, keyLe, w2558, r2558]
    simp only [orderRels, hs]
    rfl

end PlannerBuild.E
