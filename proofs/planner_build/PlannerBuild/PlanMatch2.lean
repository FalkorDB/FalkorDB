import PlannerBuild.PlanMatch
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
`planComp_stripped`, `planMatch_stripped`), `pre2390_chainRel_from_on_pattern`.
-/
namespace PlannerBuild.E

/-- The operator of a later hop (mod.rs:2213-2293), stacked on `res`. -/
def chainOp (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR) : MP :=
  match r.kind with
  | .shortest => .shortest (opRel r) [res]
  | .varLen => .cvlt (opRel r) (r.frm.alias.id != r.to.alias.id && st.bound r.frm.alias && st.bound r.to.alias)
      none [res]
  | .fixed =>
    if r.frm.alias.id = r.to.alias.id then .expandInto (opRel r) (opEmit c lw r)
      (if st.bound r.frm.alias then [res] else [scanOf (opRel r).frm, res])
    else if st.bound r.frm.alias && st.bound r.to.alias then .expandInto (opRel r) (opEmit c lw r) [res]
    else .condTr (opRel r) (opEmit c lw r) [res]

/-- A later hop: the operator, its edge Filter, then both endpoints — `from` included, which
before #2390 had no lowering here at all (mod.rs:2295-2315). -/
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

/-- Bug 4's shape: an unbound self-loop hop after other hops is
`ExpandInto(scan, res)`, and the runtime only builds child 0 (Match.lean
`selfloop_ignores_chain`). PR #3102 plans it after the hop that binds the node. -/
theorem chainRel_selfloop_unbound (st : MSt) (c : Comp) (lw : LW) (res : MP) (r : QR)
    (hk : r.kind = .fixed) (hs : r.frm.alias.id = r.to.alias.id) (hb : st.bound r.frm.alias = false) :
    (chainRel st c lw res r).1.core = .expandInto (opRel r) (opEmit c lw r) [scanOf (opRel r).frm, res] := by
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
  have hs : (scanOf (opRel r).frm).stripped = true := by
    unfold scanOf; split <;> simp [MP.stripped, h1]
  have h3' : r.kind = .fixed → (opRel r).attrs = [] := fun hk => h3 (by simp [isWalk, hk])
  unfold chainOp
  cases hk : r.kind <;> by_cases hb1 : st.bound r.frm.alias <;> by_cases hb2 : st.bound r.to.alias <;>
    by_cases h4 : r.frm.alias.id = r.to.alias.id <;>
    simp [hb1, hb2, h4, MP.stripped, MP.stripped.strippedL, h1, h2, hs, h3', hk, hres]

/-- Historical (a9377c636 mod.rs:2055-2236): a later hop lowered only its unbound `to`. -/
def pre2390_chainRel (st : MSt) (c : Comp) (res : MP) (r : QR) : MP :=
  let res := match r.kind with
    | .shortest => .shortest r [res]
    | .varLen =>
      let cv := MP.cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.frm.alias && st.bound r.to.alias)
        none [res]
      if st.bound r.frm.alias then cv else withF (attrF r.frm) cv
    | .fixed =>
      if r.frm.alias.id = r.to.alias.id then
        if st.bound r.frm.alias then withF (attrF r.frm) (.expandInto r (emitRel c r) [res])
        else .expandInto r (emitRel c r) [withF (attrF r.frm) (scanOf r.frm), res]
      else if st.bound r.frm.alias && st.bound r.to.alias then
        withF (attrF r.to) (withF (attrF r.frm) (.expandInto r (emitRel c r) [res]))
      else withF (relF r) (.condTr r (emitRel c r) [res])
  if st.bound r.to.alias then res else withF (attrF r.to) res

/-- `MATCH (a)-[:R]->(b)-[:R]->(c), (b {x:1})-[:S]->(d)`-shape: a fixed later hop leaving a bound
`b` with attrs. Before #2390 the CondTraverse kept `{x: 1}` on its pattern with no Filter
(enforced only by its per-row re-check, which #2390 also removed); now it is a Filter. -/
def bV : V := ⟨1, 0⟩
def qB : QR := { alias := ⟨5, 0⟩, named := false, frm := { alias := bV, attrs := [("x", xOne)], ptr := 9 },
                 to := { alias := ⟨4, 0⟩ } }
def stB : MSt := { visited := [bV], verified := [] }

theorem pre2390_chainRel_from_on_pattern (c : Comp) :
    (pre2390_chainRel stB c .arg qB).peel = [] ∧
    (chainRel stB c [] .arg qB).1.peel = [attrEq bV ("x", xOne)] := by
  constructor <;> rfl

/-! ## Labels on already-bound endpoints (mod.rs:1858-1872) -/

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
    · simp [hc, hl]

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

/-! ## Node-only components (mod.rs:1913-1990) -/

def u32max : Nat := 4294967295

def nodeOnly (st : MSt) (lw : LW) (n : QN) (paths : List (V × List V)) : MP × MSt × List Ex × LW :=
  if st.bound n.alias then
    let (af, lw) := lowerN lw n
    let bf := af.toList
    if n.labels.isEmpty then (.arg, st, bf, lw)
    else
      let rel : QR := { alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
                        frm := { alias := n.alias }, to := { alias := n.alias, labels := n.labels } }
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
edge id is `u32::MAX - id` (mod.rs:1936-1941) checks them. -/
theorem nodeOnly_bound_labels (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (hl : n.labels ≠ []) :
    ∃ r, (nodeOnly st lw n ps).1 = .expandInto r false [.arg] ∧
      r.alias = ⟨u32max - n.alias.id, n.alias.scope⟩ ∧ r.frm.alias = n.alias ∧ r.to.alias = n.alias ∧
      r.to.labels = n.labels ∧ r.frm.labels = [] := by
  have : n.labels.isEmpty = false := by cases h : n.labels <;> simp_all
  exact ⟨{ alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
           frm := { alias := n.alias }, to := { alias := n.alias, labels := n.labels } },
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

/-- Path elision (mod.rs:2350-2387): a named path over the single var-length
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

def planComp (st : MSt) (fv : List Nat) (lw : LW) (c : Comp) : MP × MSt × List Ex × LW :=
  let lb := boundLabelFilters st c.rels
  match sortRels lb.1 fv c.rels with
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
is bound for the clauses that follow (mod.rs:2189-2196, 2317-2324). -/
theorem planComp_binds (st : MSt) (fv : List Nat) (lw : LW) (c : Comp) (r : QR) (hr : r ∈ c.rels) (v : V)
    (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) : (planComp st fv lw c).2.1.bound v = true := by
  have hp := (sortRels_spec (boundLabelFilters st c.rels).1 fv c.rels).1
  cases hs : sortRels (boundLabelFilters st c.rels).1 fv c.rels with
  | nil => rw [hs] at hp; have := hp.mem_iff.2 hr; simp at this
  | cons r0 rs =>
    have e : (planComp st fv lw c).2.1 =
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
  | .allScan _, m', h => by simp [bindPath] at h
  | .labelScan _, m', h => by simp [bindPath] at h

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
theorem planComp_stripped (st : MSt) (fv : List Nat) (lw : LW) (c : Comp) :
    (planComp st fv lw c).1.stripped = true := by
  unfold planComp
  dsimp only
  cases sortRels (boundLabelFilters st c.rels).1 fv c.rels with
  | nil =>
    cases c.nodes with
    | nil => rfl
    | cons n _ => exact nodeOnly_stripped _ _ _ _
  | cons r rs => exact elide_stripped _ _ _ (planRels_stripped _ _ _ _ _ (firstRel_stripped _ _ _ _))

/-! ## The whole pattern -/

def planMatch (st : MSt) (fv : List Nat) (comps : List Comp) (pf : Option (MP → MP)) : MP × MSt :=
  let step := fun (acc : List MP × MSt × List Ex × LW) (c : Comp) =>
    let x := planComp acc.2.1 fv acc.2.2.2 c
    (acc.1 ++ [x.1], x.2.1, acc.2.2.1 ++ x.2.2.1, x.2.2.2)
  let x := comps.foldl step ([], st, [], [])
  let res := match x.1 with
    | [p] => p
    | ps => .cp ps
  let res := match pf with
    | some f => f res
    | none => res
  let res := match x.2.2.1 with
    | [] => res
    | [f] => .filter f res
    | fs => .filter (.node .and fs) res
  (res, x.2.1)

/-- One component: its plan; several: a CartesianProduct of them in order
(mod.rs:2391-2395); bound-label/attr filters on top (mod.rs:2404-2412). -/
theorem planMatch_join (st : MSt) (fv : List Nat) (c : Comp) :
    (planMatch st fv [c] none).1 =
      match (planComp st fv [] c).2.2.1 with
      | [] => (planComp st fv [] c).1
      | [f] => .filter f (planComp st fv [] c).1
      | fs => .filter (.node .and fs) (planComp st fv [] c).1 := by
  simp [planMatch]

theorem planMatch_cp (st : MSt) (fv : List Nat) (c1 c2 : Comp)
    (h1 : (planComp st fv [] c1).2.2.1 = [])
    (h2 : (planComp (planComp st fv [] c1).2.1 fv (planComp st fv [] c1).2.2.2 c2).2.2.1 = []) :
    (planMatch st fv [c1, c2] none).1 =
      .cp [(planComp st fv [] c1).1, (planComp (planComp st fv [] c1).2.1 fv (planComp st fv [] c1).2.2.2 c2).1] := by
  simp [planMatch, h1, h2]

/-- The whole MATCH plan (without the WHERE, a parameter) is stripped. -/
theorem planMatch_stripped (st : MSt) (fv : List Nat) (comps : List Comp) :
    (planMatch st fv comps none).1.stripped = true := by
  have hf : ∀ (cs : List Comp) (acc : List MP × MSt × List Ex × LW),
      MP.stripped.strippedL acc.1 = true →
      MP.stripped.strippedL (cs.foldl (fun (acc : List MP × MSt × List Ex × LW) (c : Comp) =>
        let x := planComp acc.2.1 fv acc.2.2.2 c
        (acc.1 ++ [x.1], x.2.1, acc.2.2.1 ++ x.2.2.1, x.2.2.2)) acc).1 = true := by
    intro cs
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
  have h0 := hf comps ([], st, [], []) rfl
  simp only [planMatch]
  generalize (comps.foldl _ ([], st, [], [])) = x at h0 ⊢
  have hr : (match x.1 with | [p] => p | ps => MP.cp ps).stripped = true := by
    split
    · rename_i p heq; rw [heq] at h0; simpa [MP.stripped.strippedL] using h0
    · simpa [MP.stripped] using h0
  split <;> simp [MP.stripped, hr]

/-! ## `build_pattern_sub_plan` (mod.rs:1038-1049) -/

def buildPatternSubPlan (st : MSt) (fv : List Nat) (comps : List Comp) (addArgs : MP → MP) : MP × MSt :=
  ((addArgs (planMatch st fv comps none).1), st)

/-- The sub-plan of a pattern predicate leaves `visited` and `verified_labels`
as they were: its variables are not bound for the clauses that follow. -/
theorem buildPatternSubPlan_restores (st : MSt) (fv : List Nat) (comps : List Comp) (aa : MP → MP) :
    (buildPatternSubPlan st fv comps aa).2 = st ∧
    (buildPatternSubPlan st fv comps aa).1 = aa (planMatch st fv comps none).1 := ⟨rfl, rfl⟩

end PlannerBuild.E
