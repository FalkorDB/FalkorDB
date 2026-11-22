import PlannerBuild.Match
/-
# Small plan-tree helpers of planner/mod.rs
-/
namespace PlannerBuild

/-! ## `subtree_contains` -/

/- BFS `any` over the subtree = some operator of the subtree satisfies `pr`. -/
mutual
def subtreeContains (pr : Op → Bool) : Plan → Bool
  | .node op cs => pr op || subtreeContainsL pr cs
def subtreeContainsL (pr : Op → Bool) : List Plan → Bool
  | [] => false
  | c :: cs => subtreeContains pr c || subtreeContainsL pr cs
end

mutual
def Plan.ops : Plan → List Op
  | .node op cs => op :: Plan.opsL cs
def Plan.opsL : List Plan → List Op
  | [] => []
  | c :: cs => Plan.ops c ++ Plan.opsL cs
end

mutual
theorem subtreeContains_iff (pr : Op → Bool) : ∀ p, subtreeContains pr p = (Plan.ops p).any pr
  | .node op cs => by simp [subtreeContains, Plan.ops, subtreeContainsL_iff pr cs]
theorem subtreeContainsL_iff (pr : Op → Bool) : ∀ l, subtreeContainsL pr l = (Plan.opsL l).any pr
  | [] => rfl
  | c :: cs => by simp [subtreeContainsL, Plan.opsL, subtreeContains_iff pr c, subtreeContainsL_iff pr cs]
end

/-! ## `add_argument_to_leaves` -/

/-- Every leaf reached by the walk is an `Argument` (MERGE: only its input is walked). -/
def argLeaves : Plan → Bool
  | .node (.merge _) (i :: _ :: _) => argLeaves i
  | .node (.merge _) _ => true
  | .node .argument [] => true
  | .node _ [] => false
  | .node _ (c :: cs) => argLeaves c && (cs.attach.all fun ⟨x, _⟩ => argLeaves x)

theorem addArgs_leaves : ∀ p, argLeaves (addArgs p) = true
  | .node (.merge q) (i :: m :: rest) => by
    simp only [addArgs, argLeaves]; exact addArgs_leaves i
  | .node (.merge q) [] => by simp [addArgs, argLeaves]
  | .node (.merge q) [c] => by simp [addArgs, argLeaves]
  | .node .argument [] => by simp [addArgs, argLeaves]
  | .node op [] => by
    cases op <;> simp [addArgs, argLeaves]
  | .node op (c :: cs) => by
    have hc := addArgs_leaves c
    have hcs : ∀ x ∈ cs, argLeaves (addArgs x) = true := fun x _ => addArgs_leaves x
    cases op <;> (try simp_all [addArgs, argLeaves])
    cases cs <;> simp_all [addArgs, argLeaves]
termination_by p => sizeOf p
decreasing_by
  all_goals first
    | (simp_wf; omega)
    | (simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega)

variable {V G : Type} (S : Sem V G)

/-- The tap is transparent for a per-row leaf: `Op(Argument)` = `Op` on the argument row. -/
theorem addArgs_leaf_ev (op : Op) (h : op.isBag = false) (hc : op.isCorr = false)
    (hn : op ≠ .union) (hcp : op ≠ .cartesian) (hm : op ≠ .orMux) (r : Rec V) (g : G) :
    ev S (.node op [.node .argument []]) r g = ev S (.node op []) r g := by
  cases op <;> simp_all [ev, Op.isBag, Op.isCorr, forRows_single]

/-! ## `stitch_below_apply_chain` -/

def stitchBelow (chain input : Plan) : Plan := insertAt chain (satChain chain) input

theorem satChain_get (p : Plan) : ∃ q, p.get (satChain p) = some q ∧ saturated q = false := by
  match p with
  | .node .apply (c :: _ :: _) =>
    obtain ⟨q, h1, h2⟩ := satChain_get c
    exact ⟨q, by simpa [satChain, Plan.get] using h1, h2⟩
  | .node .apply [] => exact ⟨_, rfl, rfl⟩
  | .node .apply [_] => exact ⟨_, rfl, rfl⟩
  | .node (.filter _) _ | .node (.project _) _ | .node .argument _ | .node (.aggregate _) _
  | .node .distinct _ | .node (.sort _) _ | .node (.skip _) _ | .node (.limit _) _
  | .node (.unwind _ _) _ | .node (.scan _) _ | .node (.hop _ _) _ | .node .cartesian _
  | .node .semiApply _ | .node .antiSemiApply _ | .node .orMux _ | .node (.optional _) _
  | .node (.pathBuilder _) _ | .node (.create _) _ | .node (.merge _) _ | .node (.set _) _
  | .node (.remove _) _ | .node (.delete _) _ | .node .commit _ | .node (.forEach _ _) _
  | .node .union _ | .node (.procCall _) _ => exact ⟨_, rfl, rfl⟩

/-- The input lands as child 0 of the first unsaturated node down the chain —
for a chain from `extract_clause_expr_comprehensions`, its single-child Apply. -/
theorem stitchBelow_get (chain input : Plan) :
    ∃ q, chain.get (satChain chain) = some q ∧ saturated q = false ∧
      (stitchBelow chain input).get (satChain chain) = some (q.push0 input) := by
  obtain ⟨q, h1, h2⟩ := satChain_get chain
  exact ⟨q, h1, h2, get_modify_self _ chain _ q h1⟩

/-- …so `Apply(sub)` becomes `Apply(input, sub)`: the sub-plan runs once per input row. -/
theorem stitchBelow_ev (sub input : Plan) (r : Rec V) (g : G) :
    ev S (stitchBelow (.node .apply [sub]) input) r g =
      forRows (fun row g => ev S sub row g) (ev S input r g).1 (ev S input r g).2 := by
  simp [stitchBelow, satChain, insertAt, Plan.modify, Plan.push0, ev]

/-! ## `ensure_apply_has_input` -/

def lone (op : Op) (cs : List Plan) : Bool := op == .apply && cs.length == 1

def ensureInput : Plan → Plan
  | .node op cs =>
    if lone op cs then .node .apply (.node .argument [] :: cs.attach.map fun ⟨c, _⟩ => ensureInput c)
    else .node op (cs.attach.map fun ⟨c, _⟩ => ensureInput c)
termination_by p => sizeOf p
decreasing_by all_goals (simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega)

/-- The inserted `Argument` changes nothing: `Apply(Argument, s)` = `Apply(s)`. -/
theorem ensureInput_ev_local (s : Plan) (r : Rec V) (g : G) :
    ev S (.node .apply [.node .argument [], s]) r g = ev S (.node .apply [s]) r g := by
  simp [ev, forRows_single]

/-- No single-child Apply is left. -/
def noLoneApply : Plan → Bool
  | .node op cs => !lone op cs && cs.attach.all fun ⟨c, _⟩ => noLoneApply c
termination_by p => sizeOf p
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

theorem noLoneApply_eq (op : Op) (cs : List Plan) :
    noLoneApply (.node op cs) = (!lone op cs && cs.all noLoneApply) := by
  rw [noLoneApply]; congr 1
  rw [Bool.eq_iff_iff]; simp

theorem ensureInput_eq (op : Op) (cs : List Plan) :
    ensureInput (.node op cs) =
      if lone op cs then .node .apply (.node .argument [] :: cs.map ensureInput)
      else .node op (cs.map ensureInput) := by
  rw [ensureInput]; simp

theorem ensureInput_saturates : ∀ p, noLoneApply (ensureInput p) = true
  | .node op cs => by
    have ih : ∀ c ∈ cs, noLoneApply (ensureInput c) = true := fun c _ => ensureInput_saturates c
    have hall : (cs.map ensureInput).all noLoneApply = true := by
      simp only [List.all_map, List.all_eq_true, Function.comp]; exact ih
    rw [ensureInput_eq]
    split
    · rw [noLoneApply_eq]
      simp only [List.all_cons, hall, Bool.and_true, noLoneApply_eq, lone, List.length_cons,
        List.length_map, List.all_nil]
      rename_i hl
      simp [lone] at hl
      simp [hl.2]
    · rename_i hl
      rw [noLoneApply_eq]
      simp only [lone, List.length_map] at hl ⊢
      simp [hl, hall]
termination_by p => sizeOf p
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

/-! ## `is_redundant_optional_match` -/

/-- What the check reads of an `OPTIONAL MATCH` clause. -/
structure RedOpt where
  optional : Bool
  rels : Nat
  paths : Nat
  nodes : List Nat
  visited : List Nat

def isRedundantOptional (c : RedOpt) : Bool :=
  c.optional && c.rels == 0 && c.paths == 0 && !c.nodes.isEmpty && c.nodes.all (c.visited.contains ·)

/-- A pattern of already-bound nodes only either keeps the row or drops it
(it binds nothing new); then OPTIONAL MATCH (padding no variable) is the
identity on the driving table — dropping the clause is exact. -/
theorem redundant_optional_identity (mt : Rec V → G → List (Rec V))
    (hm : ∀ r g, mt r g = [r] ∨ mt r g = []) (hpad : ∀ r, S.pad [] r = r) (T : Res V G) :
    specOptional S [] mt T = T := by
  obtain ⟨rows, g⟩ := T
  unfold specOptional
  rw [forRows_ro (k := fun r g => [r])]
  · simp
  · intro r g
    rcases hm r g with h | h <;> simp [h, hpad]

/-! ## `set_include_pending_on_scans` (mod.rs)

On a tree whose nodes are scans (`NodeByLabelScan`/`AllNodeScan`), other
operators, or `IncludePending`: every scan node's payload becomes
`IncludePending` and the scan itself is pushed as its last child (a leaf). -/

inductive K | scan (n : Nat) | ip (n : Nat) | other (n : Nat)
  deriving DecidableEq

inductive PT | node (k : K) (cs : List PT)

def K.isScan : K → Bool | .scan _ => true | _ => false
def K.isIP : K → Bool | .ip _ => true | _ => false

def setIP : PT → PT
  | .node (.scan n) cs => .node (.ip n) (cs.attach.map (fun ⟨c, _⟩ => setIP c) ++ [.node (.scan n) []])
  | .node k cs => .node k (cs.attach.map fun ⟨c, _⟩ => setIP c)
termination_by p => sizeOf p
decreasing_by all_goals (simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega)

theorem setIP_scan (n : Nat) (cs : List PT) :
    setIP (.node (.scan n) cs) = .node (.ip n) (cs.map setIP ++ [.node (.scan n) []]) := by
  rw [setIP]; simp

theorem setIP_other (k : K) (hk : k.isScan = false) (cs : List PT) :
    setIP (.node k cs) = .node k (cs.map setIP) := by
  cases k with
  | scan n => simp [K.isScan] at hk
  | ip n => rw [setIP]; simp; intro _ h; cases h
  | other n => rw [setIP]; simp; intro _ h; cases h

/-- Every scan in the result is a leaf (the last child of its `IncludePending`). -/
def scansLeaf : PT → Bool
  | .node k cs => (!k.isScan || cs.isEmpty) && cs.attach.all fun ⟨c, _⟩ => scansLeaf c
termination_by p => sizeOf p
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

theorem scansLeaf_eq (k : K) (cs : List PT) :
    scansLeaf (.node k cs) = ((!k.isScan || cs.isEmpty) && cs.all scansLeaf) := by
  rw [scansLeaf]; congr 1; rw [Bool.eq_iff_iff]; simp

theorem setIP_scansLeaf : ∀ p, scansLeaf (setIP p) = true
  | .node k cs => by
    have ih : ∀ c ∈ cs, scansLeaf (setIP c) = true := fun c _ => setIP_scansLeaf c
    have hall : (cs.map setIP).all scansLeaf = true := by
      simp only [List.all_map, List.all_eq_true, Function.comp]; exact ih
    cases hk : k.isScan
    · rw [setIP_other k hk, scansLeaf_eq]; simp [hk, hall]
    · cases k with
      | scan n =>
        rw [setIP_scan, scansLeaf_eq]
        simp only [K.isScan, List.all_append, hall, List.all_cons, List.all_nil, scansLeaf_eq]
        simp
      | ip n => simp [K.isScan] at hk
      | other n => simp [K.isScan] at hk
termination_by p => sizeOf p
decreasing_by simp_wf; have := List.sizeOf_lt_of_mem ‹_ ∈ cs›; omega

/-! ## `may_share_edge` (#3639)

Two relationship patterns of one component can bind the same physical edge: both are
fixed-length (no `min_hops`, not a shortest path) and their type sets overlap, an empty set
matching every type. -/

structure SRel where
  fixed : Bool
  types : List String

def mayShareEdge (a b : SRel) : Bool :=
  a.fixed && b.fixed && (a.types.isEmpty || b.types.isEmpty || a.types.any (fun t => b.types.contains t))

/-- A type both can carry exists (the empty set stands for every type). -/
theorem mayShareEdge_spec (a b : SRel) :
    mayShareEdge a b = true ↔ a.fixed = true ∧ b.fixed = true ∧
      ∃ t, (a.types = [] ∨ t ∈ a.types) ∧ (b.types = [] ∨ t ∈ b.types) := by
  simp only [mayShareEdge, Bool.and_eq_true, Bool.or_eq_true, List.isEmpty_iff, List.any_eq_true,
    List.contains_iff_mem]
  constructor
  · rintro ⟨⟨ha, hb⟩, (h | h) | ⟨t, h1, h2⟩⟩
    · exact ⟨ha, hb, (b.types.head? |>.getD ""), Or.inl h, by
        cases hbt : b.types with
        | nil => exact Or.inl rfl
        | cons x xs => exact Or.inr (by simp)⟩
    · exact ⟨ha, hb, (a.types.head? |>.getD ""), by
        cases hat : a.types with
        | nil => exact Or.inl rfl
        | cons x xs => exact Or.inr (by simp), Or.inl h⟩
    · exact ⟨ha, hb, t, Or.inr h1, Or.inr h2⟩
  · rintro ⟨ha, hb, t, h1 | h1, h2 | h2⟩
    · exact ⟨⟨ha, hb⟩, Or.inl (Or.inl h1)⟩
    · exact ⟨⟨ha, hb⟩, Or.inl (Or.inl h1)⟩
    · exact ⟨⟨ha, hb⟩, Or.inl (Or.inr h2)⟩
    · exact ⟨⟨ha, hb⟩, Or.inr ⟨t, h1, h2⟩⟩

theorem mayShareEdge_comm (a b : SRel) : mayShareEdge a b = mayShareEdge b a := by
  rw [Bool.eq_iff_iff, mayShareEdge_spec, mayShareEdge_spec]
  constructor <;> rintro ⟨ha, hb, t, h1, h2⟩ <;> exact ⟨hb, ha, t, h2, h1⟩

/-- A variable-length pattern never shares (its edges are already distinct per path). -/
theorem mayShareEdge_varlen (a b : SRel) (h : a.fixed = false) : mayShareEdge a b = false := by
  simp [mayShareEdge, h]

/-- Disjoint non-empty type sets never share. -/
theorem mayShareEdge_disjoint (a b : SRel) (ha : a.types ≠ []) (hb : b.types ≠ [])
    (hd : ∀ t ∈ a.types, t ∉ b.types) : mayShareEdge a b = false := by
  cases h : mayShareEdge a b
  · rfl
  · obtain ⟨-, -, t, h1 | h1, h2 | h2⟩ := (mayShareEdge_spec a b).1 h
    all_goals first | exact absurd h1 ha | exact absurd h2 hb | exact absurd h2 (hd t h1)

/-! ## `plan_match`'s sibling edges (#3639)

`component_siblings`: a relationship of the component takes part in uniqueness when it is named
or another relationship of the component (a different alias) may bind the same edge;
`siblings_of` gives that list to a member and nothing to a non-member. -/

structure CRel where
  id : Nat
  named : Bool
  rel : SRel

def componentSiblings (rs : List CRel) : List Nat :=
  (rs.filter fun r => r.named || rs.any fun o => o.id != r.id && mayShareEdge r.rel o.rel).map (·.id)

def siblingsOf (rs : List CRel) (r : CRel) : List Nat :=
  if (componentSiblings rs).contains r.id then componentSiblings rs else []

theorem mem_componentSiblings {rs : List CRel} {r : CRel} (hr : r ∈ rs)
    (h : r.named = true ∨ ∃ o ∈ rs, o.id ≠ r.id ∧ mayShareEdge r.rel o.rel = true) :
    r.id ∈ componentSiblings rs := by
  unfold componentSiblings
  refine List.mem_map.2 ⟨r, List.mem_filter.2 ⟨hr, ?_⟩, rfl⟩
  rcases h with h | ⟨o, ho, hne, hs⟩
  · simp [h]
  · simp only [Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq]
    exact Or.inr ⟨o, ho, hne, hs⟩

/-- A named edge always checks the whole sibling set. -/
theorem siblingsOf_named {rs : List CRel} {r : CRel} (hr : r ∈ rs) (hn : r.named = true) :
    siblingsOf rs r = componentSiblings rs := by
  simp [siblingsOf, mem_componentSiblings hr (Or.inl hn)]

/-- Two relationships that may bind one edge check each other, named or anonymous: the result
no longer depends on naming. -/
theorem siblingsOf_share {rs : List CRel} {r o : CRel} (hr : r ∈ rs) (ho : o ∈ rs)
    (hne : o.id ≠ r.id) (hs : mayShareEdge r.rel o.rel = true) :
    o.id ∈ siblingsOf rs r ∧ r.id ∈ siblingsOf rs o := by
  have h1 := mem_componentSiblings hr (Or.inr ⟨o, ho, hne, hs⟩)
  have h2 := mem_componentSiblings ho
    (Or.inr ⟨r, hr, fun h => hne h.symm, by rw [mayShareEdge_comm]; exact hs⟩)
  simp [siblingsOf, h1, h2]

/-- An anonymous relationship that can share an edge with no other checks nothing, and is
checked by nobody. -/
theorem siblingsOf_lone {rs : List CRel} {r : CRel} (hn : r.named = false)
    (hu : ∀ o ∈ rs, o.id ≠ r.id → mayShareEdge r.rel o.rel = false)
    (hid : ∀ o ∈ rs, o.id = r.id → o = r) :
    siblingsOf rs r = [] ∧ ∀ o ∈ rs, r.id ∉ siblingsOf rs o := by
  have hnot : r.id ∉ componentSiblings rs := by
    intro h
    unfold componentSiblings at h
    obtain ⟨x, hx, hxid⟩ := List.mem_map.1 h
    obtain ⟨hxr, hxp⟩ := List.mem_filter.1 hx
    have := hid x hxr hxid
    subst this
    simp only [hn, Bool.false_or, List.any_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq] at hxp
    obtain ⟨o, ho, hne, hs⟩ := hxp
    rw [hu o ho hne] at hs
    exact Bool.false_ne_true hs
  refine ⟨by simp [siblingsOf, hnot], fun o _ h => ?_⟩
  unfold siblingsOf at h
  split at h
  · exact hnot h
  · simp at h

/-- Every operator of the component checks the same list, or none. -/
theorem siblingsOf_uniform (rs : List CRel) (r : CRel) :
    siblingsOf rs r = [] ∨ siblingsOf rs r = componentSiblings rs := by
  unfold siblingsOf; split <;> simp

end PlannerBuild
