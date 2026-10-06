/-
# Pass sequencing, tree rebuilds, and edge-filter absorption

| here | there |
| --- | --- |
| `passes`, `optimize` | `optimize` optimizer/mod.rs:124-167 (pass order; NestedPlans recursion) |
| `rebuildHJ` | `rebuild_with_hash_joins` replace_cartesian_with_hash_join.rs:80-99 |
| `splitCP`, `rebuildCP` | `rebuild_with_cp_split` / `rebuild_node` push_filters_down.rs:389-466 |
| `findVlt` | `find_descendant_vlt` absorb_edge_filters_into_vlt.rs:58-83 |
| `absorb`, `semChain` | `absorb_edge_filters_into_vlt` absorb_edge_filters_into_vlt.rs:90-181 |

* `optimize_sound`: if every pass preserves a relation `R` (reflexive,
  transitive), so does `optimize`; `optimize_nested` — each nested plan is
  optimized independently; `passes_order` — the exact order (reduce_var_len_path twice).
* `rebuildHJ_id` (no rewrite: a plain copy) and `rebuildHJ_sound` (a
  congruence `R` + sound local rewrites ⇒ the rebuilt tree is `R`-equal).
* `rebuildCP_*`: identity off the target; at the target the CartesianProduct
  children are permuted (`splitCP_perm`) with the conjunct over the extracted ones.
* `findVlt_spec`, `absorb_correct`: under the per-edge reading of an edge-only
  predicate on a variable-length relationship (C FalkorDB's semantics, which
  Rust's runtime also applies — `WHERE r.w = 1` keeps paths whose every edge
  has `w = 1` on both), absorbing it into the traverse's per-hop filter (through
  PathBuilder / Filter nodes) keeps exactly the same rows.
-/
namespace Falkor.Opt.Seq

/-! ## `optimize` -/

inductive Pass
  | reduceCount | reduceExpandInto | reduceVarLenPath | eliminateTrueFilters | selectScanNode
  | pushFiltersDown | fuseAnonymousTraverse | replaceCartesianWithHashJoin | absorbEdgeFiltersIntoVlt
  | utilizeIndex | utilizeNodeById | reorderLabels | fuseOptionalTraverse | reduceBoundEdge
  deriving DecidableEq, Repr

/-- mod.rs:135-164, in order. -/
def passes : List Pass :=
  [.reduceCount, .reduceExpandInto, .reduceVarLenPath, .eliminateTrueFilters, .selectScanNode,
   .pushFiltersDown, .fuseAnonymousTraverse, .replaceCartesianWithHashJoin, .absorbEdgeFiltersIntoVlt,
   .reduceVarLenPath, .utilizeIndex, .utilizeNodeById, .reorderLabels, .fuseOptionalTraverse,
   .reduceBoundEdge]

theorem passes_order :
    passes.length = 15 ∧ passes.count .reduceVarLenPath = 2 ∧
    passes.head? = some .reduceCount ∧ passes.getLast? = some .reduceBoundEdge ∧
    passes.idxOf .absorbEdgeFiltersIntoVlt < 9 ∧ passes[9]? = some .reduceVarLenPath ∧
    passes.idxOf .fuseOptionalTraverse = 13 := by decide

/-- A plan tree whose root may be `NestedPlans`. -/
inductive OPlan (P : Type)
  | plain (p : P)
  | nested (main : OPlan P) (ns : List (OPlan P))

variable {P : Type} (run : Pass → P → P)

def applyAll (p : P) : P := passes.foldl (fun p ps => run ps p) p

def optimize : OPlan P → OPlan P
  | .plain p => .plain (applyAll run p)
  | .nested m ns => .nested (optimize m) (ns.attach.map fun ⟨n, _⟩ => optimize n)
termination_by p => sizeOf p
decreasing_by
  all_goals simp_wf
  all_goals first | omega | (have := List.sizeOf_lt_of_mem ‹_ ∈ ns›; omega)

theorem foldl_sound (R : P → P → Prop) (hrefl : ∀ p, R p p) (htrans : ∀ a b c, R a b → R b c → R a c)
    (hp : ∀ ps p, R p (run ps p)) (l : List Pass) (p : P) : R p (l.foldl (fun p ps => run ps p) p) := by
  induction l generalizing p with
  | nil => exact hrefl p
  | cons ps l ih => exact htrans _ _ _ (hp ps p) (ih _)

/-- **Sequencing.** Every pass preserving `R` ⇒ the whole optimizer preserves it. -/
theorem applyAll_sound (R : P → P → Prop) (hrefl : ∀ p, R p p) (htrans : ∀ a b c, R a b → R b c → R a c)
    (hp : ∀ ps p, R p (run ps p)) (p : P) : R p (applyAll run p) :=
  foldl_sound run R hrefl htrans hp passes p

theorem optimize_nested (m : OPlan P) (ns : List (OPlan P)) :
    optimize run (.nested m ns) = .nested (optimize run m) (ns.map (optimize run)) := by
  rw [optimize]; simp

theorem optimize_plain (p : P) : optimize run (.plain p) = .plain (applyAll run p) := by
  rw [optimize]

/-! ## `rebuild_with_hash_joins` -/

inductive T (D : Type)
  | node (d : D) (cs : List (T D))

variable {D : Type}

/-- Bottom-up copy; at a `Filter(CartesianProduct)` the local rewrite `rw` is
tried and, if it fires, the result is processed again (fuel bounds the
re-processing; the Rust relies on each rewrite consuming one conjunct). -/
def rebuildHJ (rw : T D → Option (T D)) : Nat → T D → T D
  | 0, t => t
  | fuel + 1, .node d cs =>
    match rw (.node d cs) with
    | some t' => rebuildHJ rw fuel t'
    | none => .node d (cs.attach.map fun ⟨c, _⟩ => rebuildHJ rw (fuel + 1) c)
termination_by fuel t => (fuel, sizeOf t)
decreasing_by
  · simp_wf; omega
  · simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; right; omega

/-- With no applicable rewrite the pass is a plain copy. -/
theorem rebuildHJ_id (rw : T D → Option (T D)) (h : ∀ t, rw t = none) : ∀ fuel t, rebuildHJ rw fuel t = t
  | 0, t => by rw [rebuildHJ]
  | fuel + 1, .node d cs => by
    have ih : ∀ c ∈ cs, rebuildHJ rw (fuel + 1) c = c := fun c hc => rebuildHJ_id rw h (fuel + 1) c
    rw [rebuildHJ, h]
    simp only
    congr 1
    rw [List.attach_map_val (f := fun x => rebuildHJ rw (fuel + 1) x), List.map_congr_left ih, List.map_id']
termination_by fuel t => (fuel, sizeOf t)
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; right; omega

/-- With a congruence `R` and sound local rewrites, the rebuilt plan is `R`-equal. -/
theorem rebuildHJ_sound (rw : T D → Option (T D)) (R : T D → T D → Prop)
    (hrefl : ∀ t, R t t) (htrans : ∀ a b c, R a b → R b c → R a c)
    (hcong : ∀ d (cs cs' : List (T D)), cs.length = cs'.length →
      (∀ i (h1 : i < cs.length) (h2 : i < cs'.length), R cs[i] cs'[i]) → R (.node d cs) (.node d cs'))
    (hrw : ∀ t t', rw t = some t' → R t t') : ∀ fuel t, R t (rebuildHJ rw fuel t)
  | 0, t => by rw [rebuildHJ]; exact hrefl t
  | fuel + 1, .node d cs => by
    have ih : ∀ c ∈ cs, R c (rebuildHJ rw (fuel + 1) c) := fun c hc =>
      rebuildHJ_sound rw R hrefl htrans hcong hrw (fuel + 1) c
    rw [rebuildHJ]
    cases h : rw (.node d cs) with
    | some t' => exact htrans _ _ _ (hrw _ _ h) (rebuildHJ_sound rw R hrefl htrans hcong hrw fuel t')
    | none =>
      simp only
      apply hcong _ _ _ (by simp)
      intro i h1 h2
      simp only [List.getElem_map, List.getElem_attach]
      exact ih _ (List.getElem_mem h1)
termination_by fuel t => (fuel, sizeOf t)
decreasing_by
  · simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; right; omega
  · simp_wf; omega

/-! ## `rebuild_with_cp_split` -/

/-- The target Filter over a CartesianProduct (children `cs`) becomes
`CP(others ++ [Filter(conj, CP(extracted))])`, under the remaining filter if any. -/
def splitCP (filterD cpD : List (T D) → D) (mkFilter : List Nat → D) (cs : List (T D))
    (solving : List Nat) (conj : Nat) (remaining : List Nat) : T D :=
  let idx := (List.range cs.length).zip cs
  let extracted := (idx.filter fun p => solving.contains p.1).map Prod.snd
  let others := (idx.filter fun p => !solving.contains p.1).map Prod.snd
  let inner := T.node (mkFilter [conj]) [.node (cpD extracted) extracted]
  let cp := T.node (cpD (others ++ [inner])) (others ++ [inner])
  if remaining.isEmpty then cp else .node (mkFilter remaining) [cp]

theorem filter_split_perm {α : Type} (p : α → Bool) (l : List α) :
    (l.filter (fun x => !p x) ++ l.filter p).Perm l := by
  have := List.filter_append_perm (fun x => !p x) l
  simpa using this

/-- No child is lost or duplicated: the extracted and the other children are a
permutation of the original ones. -/
theorem splitCP_perm (cs : List (T D)) (solving : List Nat) :
    let idx := (List.range cs.length).zip cs
    ((idx.filter fun p => !solving.contains p.1).map Prod.snd ++
      (idx.filter fun p => solving.contains p.1).map Prod.snd).Perm cs := by
  intro idx
  rw [← List.map_append]
  have h1 := (filter_split_perm (fun p : Nat × T D => solving.contains p.1) idx).map Prod.snd
  have h2 : idx.map Prod.snd = cs := by
    simp [idx, List.map_snd_zip]
  rw [h2] at h1; exact h1

/-- Off the target, `rebuild_node` is a plain copy. -/
def rebuildCP (isTarget : T D → Bool) (onTarget : T D → T D) : T D → T D
  | .node d cs => if isTarget (.node d cs) then onTarget (.node d cs)
    else .node d (cs.attach.map fun ⟨c, _⟩ => rebuildCP isTarget onTarget c)
termination_by t => sizeOf t
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

theorem rebuildCP_off (isTarget : T D → Bool) (onTarget : T D → T D) (h : ∀ t, isTarget t = false) :
    ∀ t, rebuildCP isTarget onTarget t = t
  | .node d cs => by
    have ih : ∀ c ∈ cs, rebuildCP isTarget onTarget c = c := fun c hc => rebuildCP_off isTarget onTarget h c
    rw [rebuildCP, h]
    simp only [Bool.false_eq_true, ↓reduceIte]
    congr 1
    rw [List.attach_map_val (f := fun x => rebuildCP isTarget onTarget x), List.map_congr_left ih, List.map_id']
termination_by t => sizeOf t
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

theorem rebuildCP_on (isTarget : T D → Bool) (onTarget : T D → T D) (t : T D) (h : isTarget t = true) :
    rebuildCP isTarget onTarget t = onTarget t := by
  cases t; rw [rebuildCP, h]; rfl

/-! ## Edge-filter absorption -/

/-- Operators between the Filter and the traverse that `find_descendant_vlt` walks through. -/
inductive Tr | pathB | filter (cs : List Nat)
  deriving DecidableEq

/-- `Filter → (PathBuilder | Filter)* → CondVarLenTraverse`, every node single-child. -/
inductive Shape
  | vlt (edge : Nat) (ef : List Nat)
  | tr (t : Tr) (child : Shape)
  | other

def findVlt : Shape → Option (List Tr × Nat × List Nat)
  | .vlt e ef => some ([], e, ef)
  | .tr t c => (findVlt c).map fun (ts, e, ef) => (t :: ts, e, ef)
  | .other => none

theorem findVlt_spec (s : Shape) (ts : List Tr) (e : Nat) (ef : List Nat) (h : findVlt s = some (ts, e, ef)) :
    s = ts.foldr .tr (.vlt e ef) := by
  induction s generalizing ts with
  | vlt e' ef' => simp [findVlt] at h; obtain ⟨rfl, rfl, rfl⟩ := h; rfl
  | tr t c ih =>
    simp only [findVlt, Option.map_eq_some_iff] at h
    obtain ⟨⟨ts', e', ef'⟩, h1, h2⟩ := h
    simp at h2; obtain ⟨rfl, rfl, rfl⟩ := h2
    rw [ih _ h1]; rfl
  | other => simp [findVlt] at h

/-- Split conjuncts: those mentioning only the edge (and something) go in. -/
def isEdgeOnly (vars : Nat → List Nat) (e c : Nat) : Bool := !(vars c).isEmpty && (vars c).all (· == e)

def absorbSplit (vars : Nat → List Nat) (e : Nat) (cs : List Nat) : List Nat × List Nat :=
  (cs.filter (isEdgeOnly vars e), cs.filter fun c => !isEdgeOnly vars e c)

variable {R : Type} (edges : R → List Nat) (pe : Nat → Nat → Bool) (holds : Nat → R → Bool)
  (addPath : R → R) (vars : Nat → List Nat)

def keepAll (cs : List Nat) (r : R) : Bool := cs.all fun c => holds c r

/-- Rows of the traverse (base candidates filtered per hop), then the chain bottom-up. -/
def semVlt (base : List R) (ef : List Nat) : List R :=
  base.filter fun r => ef.all fun c => (edges r).all (pe c)

def semTr : Tr → List R → List R
  | .pathB, rows => rows.map addPath
  | .filter cs, rows => rows.filter (keepAll holds cs)

def semChain (ts : List Tr) (rows : List R) : List R := ts.foldr (semTr holds addPath) rows

/-- Hypotheses: an edge-only conjunct holds on a row iff it holds on every edge
of its path (per-edge reading); PathBuilder keeps edges and verdicts. -/
structure Per : Prop where
  perEdge : ∀ e c r, isEdgeOnly vars e c = true → holds c r = (edges r).all (pe c)
  pathEdges : ∀ r, edges (addPath r) = edges r
  pathHolds : ∀ c r, holds c (addPath r) = holds c r

theorem semChain_filter (h : Per edges pe holds addPath vars) (p : R → Bool)
    (hp : ∀ r, p (addPath r) = p r) (ts : List Tr) (rows : List R) :
    semChain holds addPath ts (rows.filter p) = (semChain holds addPath ts rows).filter p := by
  induction ts with
  | nil => rfl
  | cons t ts ih =>
    simp only [semChain, List.foldr_cons] at *
    rw [ih]
    cases t with
    | pathB =>
      simp only [semTr]
      rw [List.filter_map]
      congr 1
      apply List.filter_congr; intro r _; simp [hp]
    | filter cs =>
      simp only [semTr, List.filter_filter]
      congr 1; funext r; exact Bool.and_comm _ _

theorem keepAll_split (h : Per edges pe holds addPath vars) (e : Nat) (cs : List Nat) (r : R) :
    keepAll holds cs r =
      (((absorbSplit vars e cs).1.all fun c => (edges r).all (pe c)) && keepAll holds (absorbSplit vars e cs).2 r) := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [absorbSplit, keepAll] at ih ⊢
    by_cases he : isEdgeOnly vars e c = true
    · simp only [List.filter_cons, he, ↓reduceIte, Bool.not_true, Bool.false_eq_true, List.all_cons, ih,
        h.perEdge e c r he, Bool.and_assoc]
    · simp only [Bool.not_eq_true] at he
      simp only [List.filter_cons, he, Bool.false_eq_true, ↓reduceIte, Bool.not_false, List.all_cons, ih]
      cases holds c r <;> simp

/-- **Absorption is exact.** `Filter(cs) → chain → VLT(e, ef)` keeps the same
rows as `Filter(rest) → chain → VLT(e, ef ++ edgeOnly)`. -/
theorem absorb_correct (h : Per edges pe holds addPath vars) (ts : List Tr) (e : Nat) (ef cs : List Nat)
    (base : List R) :
    let sp := absorbSplit vars e cs
    (semChain holds addPath ts (semVlt edges pe base ef)).filter (keepAll holds cs) =
      (semChain holds addPath ts (semVlt edges pe base (ef ++ sp.1))).filter (keepAll holds sp.2) := by
  intro sp
  have hv : semVlt edges pe base (ef ++ sp.1) =
      (semVlt edges pe base ef).filter (fun r => sp.1.all fun c => (edges r).all (pe c)) := by
    simp only [semVlt, List.filter_filter, List.all_append]
    congr 1; funext r; exact Bool.and_comm _ _
  rw [hv, semChain_filter edges pe holds addPath vars h]
  · rw [List.filter_filter]
    congr 1; funext r
    rw [keepAll_split edges pe holds addPath vars h e cs r, Bool.and_comm]
  · intro r; simp [h.pathEdges]

/-- Nothing edge-only: nothing changes. -/
theorem absorbSplit_none (e : Nat) (cs : List Nat) (h : ∀ c ∈ cs, isEdgeOnly vars e c = false) :
    absorbSplit vars e cs = ([], cs) := by
  simp only [absorbSplit, List.filter_eq_nil_iff, Prod.mk.injEq]
  refine ⟨fun c hc => by simp [h c hc], ?_⟩
  rw [List.filter_eq_self]; intro c hc; simp [h c hc]

end Falkor.Opt.Seq
