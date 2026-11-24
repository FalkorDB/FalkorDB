import PlannerBuild.PlanMatch
/-
# Lifting cross-entity inline attributes (mod.rs)

`plan_match` first rewrites the pattern: an inline-attribute entry whose value
reads a variable the pattern binds, other than the entity itself (and, for a
relationship, its endpoints), is taken out of the map and becomes a filter in
`bound_filters`, so it sits above the whole pattern like a `WHERE`. Such an
edge is put in `lifted_rels`: it must be emitted for the filter to read it.

Abstractions (documented, not load-bearing for a theorem here):
* the lifted filters are listed in component order; Rust walks
  `pattern.relationships()` then `pattern.nodes()` of the whole pattern, which is the
  same order when the components are listed in pattern order;
* a kept map keeps its map identity `ptr` (Rust allocates a fresh `Arc` for it);
  `lower_inline_attrs` keys on it, and the old map is gone from the rewritten pattern;
* an endpoint's entry in `nodes()` lifts into `dropped` (mod.rs): its map was
  already handled through the relationship, so the model skips it.
-/
namespace PlannerBuild.E

/-- `pattern.variables()` of the whole pattern. -/
def patVars (comps : List Comp) : List V :=
  comps.flatMap fun c => c.nodes.map (·.alias) ++ c.rels.map (·.alias) ++ c.paths.map (·.1)

/-- An entry moves when its value reads a pending variable not in `loc`. -/
def readsPending (pending loc : List V) (kv : String × Ex) : Bool :=
  (nodes kv.2).any fun d => match d with
    | .var v => pending.contains v && !loc.contains v
    | _ => false

/-- The kept part of a map, `none` when nothing moves. -/
def keepAttrs (pending loc : List V) (attrs : List (String × Ex)) : Option (List (String × Ex)) :=
  if attrs.any (readsPending pending loc) then some (attrs.filter fun kv => !readsPending pending loc kv)
  else none

/-- The lifted filter of one map (once per `(alias, map)`, the `seen` set). -/
def liftItem (pending : List V) (seen : LW) (it : V × List (String × Ex) × Nat × List V) : List Ex × LW :=
  let (alias, attrs, ptr, loc) := it
  if attrs.any (readsPending pending loc) then
    if seen.contains (lwKey alias ptr) then ([], seen)
    else ((inlineAttrsToFilter alias (attrs.filter (readsPending pending loc))).toList,
          seen ++ [lwKey alias ptr])
  else ([], seen)

def liftItems (pending : List V) : LW → List (V × List (String × Ex) × Nat × List V) → List Ex
  | _, [] => []
  | seen, it :: its => (liftItem pending seen it).1 ++ liftItems pending (liftItem pending seen it).2 its

def keepNode (pending : List V) (n : QN) : QN :=
  match keepAttrs pending [n.alias] n.attrs with
  | some kept => { n with attrs := kept }
  | none => n

/-- A fixed-length edge's map is split; variable-length and shortest keep theirs. -/
def relAttrs? (pending : List V) (r : QR) : Option (List (String × Ex)) :=
  if r.kind = .fixed then keepAttrs pending [r.alias, r.frm.alias, r.to.alias] r.attrs else none

def keepRel (pending : List V) (r : QR) : QR :=
  { r with frm := keepNode pending r.frm, to := keepNode pending r.to,
           attrs := (relAttrs? pending r).getD r.attrs,
           lifted := r.lifted || (relAttrs? pending r).isSome }

/-- The maps visited, in order: each edge's endpoints and itself, then the nodes no edge touches. -/
def liftOrder (comps : List Comp) : List (V × List (String × Ex) × Nat × List V) :=
  let rels := comps.flatMap (·.rels)
  let ends := rels.flatMap fun r => [r.frm.alias, r.to.alias]
  rels.flatMap (fun r =>
      [(r.frm.alias, r.frm.attrs, r.frm.ptr, [r.frm.alias]), (r.to.alias, r.to.attrs, r.to.ptr, [r.to.alias])] ++
      (if r.kind = .fixed then [(r.alias, r.attrs, r.ptr, [r.alias, r.frm.alias, r.to.alias])] else [])) ++
    ((comps.flatMap (·.nodes)).filter fun n => !ends.contains n.alias).map
      fun n => (n.alias, n.attrs, n.ptr, [n.alias])

def pendingOf (st : MSt) (comps : List Comp) : List V :=
  (patVars comps).filter fun v => !st.bound v

/-- One component with its maps split. -/
def rewriteComp (pending : List V) (c : Comp) : Comp :=
  { c with nodes := c.nodes.map (keepNode pending), rels := c.rels.map (keepRel pending) }

/-- `lift_cross_entity_attrs`: the rewritten components and the lifted filters. -/
def liftPattern (st : MSt) (comps : List Comp) : List Comp × List Ex :=
  let pending := pendingOf st comps
  (comps.map (rewriteComp pending), liftItems pending [] (liftOrder comps))

/-- `pattern_attrs_read_visited`: some inline map of the pattern reads a bound variable. -/
def attrsReadVisited (st : MSt) (comps : List Comp) : Bool :=
  let reads := fun (attrs : List (String × Ex)) =>
    attrs.any fun kv => (nodes kv.2).any fun d => match d with
      | .var v => st.bound v
      | _ => false
  comps.any fun c => c.nodes.any (fun n => reads n.attrs) || c.rels.any (fun r => reads r.attrs)

/-! ## Nothing to lift -/

/-- A map none of whose entries reads a pending variable. -/
def NoCross (pending loc : List V) (attrs : List (String × Ex)) : Prop :=
  attrs.any (readsPending pending loc) = false

theorem keepAttrs_none (pending loc : List V) (attrs : List (String × Ex))
    (h : NoCross pending loc attrs) : keepAttrs pending loc attrs = none := by
  simp [keepAttrs, NoCross] at h ⊢; exact h

theorem liftItem_none (pending : List V) (seen : LW) (it : V × List (String × Ex) × Nat × List V)
    (h : NoCross pending it.2.2.2 it.2.1) : liftItem pending seen it = ([], seen) := by
  obtain ⟨a, attrs, p, l⟩ := it
  simp only [NoCross] at h
  simp [liftItem, h]

theorem liftItems_none (pending : List V) (its : List (V × List (String × Ex) × Nat × List V))
    (h : ∀ it ∈ its, NoCross pending it.2.2.2 it.2.1) (seen : LW) : liftItems pending seen its = [] := by
  induction its generalizing seen with
  | nil => rfl
  | cons it its ih =>
    simp only [liftItems, liftItem_none pending seen it (h it (by simp))]
    exact ih (fun x hx => h x (by simp [hx])) seen

theorem keepNode_id (pending : List V) (n : QN) (h : NoCross pending [n.alias] n.attrs) :
    keepNode pending n = n := by
  simp [keepNode, keepAttrs_none _ _ _ h]

/-- A map-free pattern (the common case, and every pattern of the bug theorems) is not
rewritten and lifts nothing. -/
theorem liftPattern_noattrs (st : MSt) (comps : List Comp)
    (hn : ∀ c ∈ comps, ∀ n ∈ c.nodes, n.attrs = [])
    (hr : ∀ c ∈ comps, ∀ r ∈ c.rels, r.attrs = [] ∧ r.frm.attrs = [] ∧ r.to.attrs = [] ∧ r.lifted = false) :
    liftPattern st comps = (comps, []) := by
  simp only [liftPattern, Prod.mk.injEq]
  constructor
  · conv => rhs; rw [← List.map_id comps]
    apply List.map_congr_left
    intro c hc
    have h1 : c.nodes.map (keepNode (pendingOf st comps)) = c.nodes := by
      conv => rhs; rw [← List.map_id c.nodes]
      apply List.map_congr_left
      intro n hn'
      simp [keepNode, keepAttrs, hn c hc n hn']
    have h2 : c.rels.map (keepRel (pendingOf st comps)) = c.rels := by
      conv => rhs; rw [← List.map_id c.rels]
      apply List.map_congr_left
      intro r hr'
      obtain ⟨a, b, d, e⟩ := hr c hc r hr'
      cases r
      simp_all [keepRel, keepNode, keepAttrs, relAttrs?]
    simp [rewriteComp, h1, h2]
  · apply liftItems_none
    intro it hit
    simp only [liftOrder, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_filter] at hit
    rcases hit with ⟨r, ⟨c, hc, hrc⟩, hit⟩ | ⟨n, ⟨⟨c, hc, hnc⟩, _⟩, rfl⟩
    · obtain ⟨a, b, d, _⟩ := hr c hc r hrc
      simp only [List.mem_cons, List.mem_nil_iff, or_false] at hit
      rcases hit with (rfl | rfl) | hit
      · simp [NoCross, b]
      · simp [NoCross, d]
      · split at hit
        · simp only [List.mem_singleton] at hit; subst hit; simp [NoCross, a]
        · simp at hit
    · simp [NoCross, hn c hc n hnc]

end PlannerBuild.E
