import PlannerBuild.PlanMatch
/-
# `plan_match`: components, bound-label filters, path elision, joining

Continues PlanMatch.lean. Theorems: `chainRel_core` (operator table for the
later hops — including the self-loop case `ExpandInto(scan, res)` behind bug 4),
`planComp_binds` (every endpoint and edge alias of a component is bound
afterwards), `boundLabels_spec` (an already-bound endpoint with labels not yet
enforced gets exactly a `hasLabels(alias, missing)` filter, after which none
are missing), `nodeOnly_*`, `planMatch_join`, `elide_spec`,
`buildPatternSubPlan_restores`.
-/
namespace PlannerBuild.E

theorem chainRel_core (st : MSt) (c : Comp) (res : MP) (r : QR) :
    (chainRel st c res r).core =
      match r.kind with
      | .shortest => .shortest r [res]
      | .varLen => .cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.frm.alias && st.bound r.to.alias)
          none [res]
      | .fixed =>
        if r.frm.alias.id = r.to.alias.id then .expandInto r (emitRel c r)
          (if st.bound r.frm.alias then [res] else [withF (attrF r.frm) (scanOf r.frm), res])
        else if st.bound r.frm.alias && st.bound r.to.alias then .expandInto r (emitRel c r) [res]
        else .condTr r (emitRel c r) [res] := by
  unfold chainRel
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, core_withF, MP.core]

/-- Bug 4's shape: an unbound self-loop hop after other hops is
`ExpandInto(scan, res)`, and the runtime only builds child 0 (Match.lean
`selfloop_ignores_chain`). PR #3102 plans it after the hop that binds the node. -/
theorem chainRel_selfloop_unbound (st : MSt) (c : Comp) (res : MP) (r : QR) (hk : r.kind = .fixed)
    (hs : r.frm.alias.id = r.to.alias.id) (hb : st.bound r.frm.alias = false) :
    (chainRel st c res r).core = .expandInto r (emitRel c r) [withF (attrF r.frm) (scanOf r.frm), res] := by
  rw [chainRel_core]; simp [hk, hs, hb]

/-! ## Labels on already-bound endpoints (mod.rs:1733-1747) -/

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

/-! ## Node-only components (mod.rs:1786-1863) -/

def u32max : Nat := 4294967295

def nodeOnly (st : MSt) (n : QN) (paths : List (V × List V)) : MP × MSt × List Ex :=
  if st.bound n.alias then
    let bf := (attrF n).toList
    if n.labels.isEmpty then (.arg, st, bf)
    else
      let rel : QR := { alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
                        frm := { alias := n.alias }, to := { alias := n.alias, labels := n.labels } }
      (.expandInto rel false [.arg], markLabels st n.alias n.labels, bf)
  else
    let res := withF (attrF n) (scanOf n)
    let st' := markLabels { st with visited := st.visited ++ [n.alias] } n.alias n.labels
    (if paths.isEmpty then res else .pathB paths res, st', [])

theorem nodeOnly_bound_plain (st : MSt) (n : QN) (ps : List (V × List V)) (hb : st.bound n.alias = true)
    (hl : n.labels = []) : (nodeOnly st n ps).1 = .arg := by
  simp [nodeOnly, hb, hl]

/-- Extra labels on a bound node: a synthetic self-loop `ExpandInto` whose
edge id is `u32::MAX - id` (mod.rs:1810-1815) checks them. -/
theorem nodeOnly_bound_labels (st : MSt) (n : QN) (ps : List (V × List V)) (hb : st.bound n.alias = true)
    (hl : n.labels ≠ []) : ∃ r, (nodeOnly st n ps).1 = .expandInto r false [.arg] ∧
      r.alias = ⟨u32max - n.alias.id, n.alias.scope⟩ ∧ r.frm.alias = n.alias ∧ r.to.alias = n.alias ∧
      r.to.labels = n.labels ∧ r.frm.labels = [] := by
  have : n.labels.isEmpty = false := by cases h : n.labels <;> simp_all
  exact ⟨{ alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
           frm := { alias := n.alias }, to := { alias := n.alias, labels := n.labels } },
    by simp [nodeOnly, hb, this], rfl, rfl, rfl, rfl, rfl⟩

theorem nodeOnly_unbound (st : MSt) (n : QN) (ps : List (V × List V)) (hb : st.bound n.alias = false) :
    (nodeOnly st n ps).1.core = (if ps.isEmpty then scanOf n else .pathB ps (withF (attrF n) (scanOf n))) ∧
      (nodeOnly st n ps).2.1.bound n.alias = true := by
  constructor
  · unfold nodeOnly; simp only [hb, Bool.false_eq_true, ↓reduceIte]
    split <;> simp [core_withF, MP.core, scanOf] <;> split <;> rfl
  · simp only [nodeOnly, hb, Bool.false_eq_true, ↓reduceIte]
    unfold markLabels; split <;> simp [MSt.bound]

/-! ## A component with relationships -/

/-- Path elision (mod.rs:2245-2282): a named path over the single var-length
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

def planRels (c : Comp) : List QR → MP → MSt → MP × MSt
  | [], res, st => (res, st)
  | r :: rs, res, st => planRels c rs (chainRel st c res r) (visit st r)

def planComp (st : MSt) (fv : List Nat) (c : Comp) : MP × MSt × List Ex :=
  let lb := boundLabelFilters st c.rels
  match sortRels lb.1 fv c.rels with
  | [] => match c.nodes with
    | n :: _ => let x := nodeOnly lb.1 n c.paths; (x.1, x.2.1, lb.2 ++ x.2.2)
    | [] => (.arg, lb.1, lb.2)
  | r :: rs =>
    let x := planRels c rs (firstRel lb.1 c r) (visit lb.1 r)
    (elide c.rels c.paths x.1, x.2, lb.2)

theorem visit_binds (st : MSt) (r : QR) (v : V) (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) :
    (visit st r).bound v = true := by
  unfold visit markLabels
  split <;> split <;> simp [MSt.bound] <;> rcases hv with rfl | rfl | rfl <;> simp

theorem visit_mono (st : MSt) (r : QR) (v : V) (h : st.bound v = true) : (visit st r).bound v = true := by
  have h' : v ∈ st.visited := by simpa [MSt.bound] using h
  unfold visit markLabels
  split <;> split <;> simp [MSt.bound, h']

theorem planRels_mono (c : Comp) (rs : List QR) (res : MP) (st : MSt) (v : V)
    (h : st.bound v = true) : (planRels c rs res st).2.bound v = true := by
  induction rs generalizing res st with
  | nil => exact h
  | cons r rs ih => simp only [planRels]; exact ih _ _ (visit_mono st r v h)

theorem planRels_binds (c : Comp) (rs : List QR) (res : MP) (st : MSt) (r : QR) (hr : r ∈ rs)
    (v : V) (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) :
    (planRels c rs res st).2.bound v = true := by
  induction rs generalizing res st with
  | nil => simp at hr
  | cons r' rs ih =>
    simp only [planRels]
    rcases List.mem_cons.1 hr with rfl | hr
    · exact planRels_mono c rs _ _ v (visit_binds st _ v hv)
    · exact ih _ _ hr

/-- After a component with relationships, every endpoint and every edge alias
is bound for the clauses that follow (mod.rs:2050-2054, 2222-2226). -/
theorem planComp_binds (st : MSt) (fv : List Nat) (c : Comp) (r : QR) (hr : r ∈ c.rels) (v : V)
    (hv : v = r.frm.alias ∨ v = r.to.alias ∨ v = r.alias) : (planComp st fv c).2.1.bound v = true := by
  have hp := (sortRels_spec (boundLabelFilters st c.rels).1 fv c.rels).1
  cases hs : sortRels (boundLabelFilters st c.rels).1 fv c.rels with
  | nil => rw [hs] at hp; have := hp.mem_iff.2 hr; simp at this
  | cons r0 rs =>
    have e : (planComp st fv c).2.1 =
        (planRels c rs (firstRel (boundLabelFilters st c.rels).1 c r0) (visit (boundLabelFilters st c.rels).1 r0)).2 := by
      unfold planComp; simp only [hs]
    rw [e]
    rw [hs] at hp
    rcases List.mem_cons.1 (hp.mem_iff.2 hr) with rfl | hr'
    · exact planRels_mono _ _ _ _ v (visit_binds _ _ v hv)
    · exact planRels_binds _ _ _ _ r hr' v hv

/-! ## The whole pattern -/

def planMatch (st : MSt) (fv : List Nat) (comps : List Comp) (pf : Option (MP → MP)) : MP × MSt :=
  let step := fun (acc : List MP × MSt × List Ex) (c : Comp) =>
    let x := planComp acc.2.1 fv c
    (acc.1 ++ [x.1], x.2.1, acc.2.2 ++ x.2.2)
  let x := comps.foldl step ([], st, [])
  let res := match x.1 with
    | [p] => p
    | ps => .cp ps
  let res := match pf with
    | some f => f res
    | none => res
  let res := match x.2.2 with
    | [] => res
    | [f] => .filter f res
    | fs => .filter (.node .and fs) res
  (res, x.2.1)

/-- One component: its plan; several: a CartesianProduct of them in order
(mod.rs:2286-2290); bound-label/attr filters on top (mod.rs:2299-2307). -/
theorem planMatch_join (st : MSt) (fv : List Nat) (c : Comp) :
    (planMatch st fv [c] none).1 =
      match (planComp st fv c).2.2 with
      | [] => (planComp st fv c).1
      | [f] => .filter f (planComp st fv c).1
      | fs => .filter (.node .and fs) (planComp st fv c).1 := by
  simp [planMatch]

theorem planMatch_cp (st : MSt) (fv : List Nat) (c1 c2 : Comp)
    (h1 : (planComp st fv c1).2.2 = []) (h2 : (planComp (planComp st fv c1).2.1 fv c2).2.2 = []) :
    (planMatch st fv [c1, c2] none).1 = .cp [(planComp st fv c1).1, (planComp (planComp st fv c1).2.1 fv c2).1] := by
  simp [planMatch, h1, h2]

/-! ## `build_pattern_sub_plan` (mod.rs:916-927) -/

def buildPatternSubPlan (st : MSt) (fv : List Nat) (comps : List Comp) (addArgs : MP → MP) : MP × MSt :=
  ((addArgs (planMatch st fv comps none).1), st)

/-- The sub-plan of a pattern predicate leaves `visited` and `verified_labels`
as they were: its variables are not bound for the clauses that follow. -/
theorem buildPatternSubPlan_restores (st : MSt) (fv : List Nat) (comps : List Comp) (aa : MP → MP) :
    (buildPatternSubPlan st fv comps aa).2 = st ∧
    (buildPatternSubPlan st fv comps aa).1 = aa (planMatch st fv comps none).1 := ⟨rfl, rfl⟩

end PlannerBuild.E
