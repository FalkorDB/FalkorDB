import FalkorBinder.Resolve
/-
# CALL {} entry projection as a record boundary (#2845, eb470521b)

Folded in from the retired proofs/pr2845_review. Model of `bind_call_body`
(graph/src/planner/binder.rs:824-971) and `build_import_projections`
(binder.rs:979-1003); scope tables, `freshVar`, `mint_fresh_iff`,
`projectName` are this project's (Table.lean, Resolve.lean). Counterexamples
on main before #2845 (da6f808c3) were removed (in 17ccc5412)
(fixed by #2845, eb470521b; closes #2601, #2602).

* Every CALL body gets an entry projection (`bodyNew_entry`, `bodyNew_empty`,
  `branchNew_entry`), unchanged when something is imported (`bodyNew_eq_old`).
* Fresh ids: entry ids are `0..k-1`, the next mint is `k` (`importProj_spec`,
  `entry_ids_below`); body scope disjoint from outer scopes (`body_scope_disjoint`).
* No slot aliasing for plain bodies at runtime (`body_own_unbound`,
  `body_import_value`, `new_scan_scans`).
* UNION bodies alias binder keys `(scope_id, id)` (`union_branch_label_alias`,
  open, #3027 fixes it: `relabel_3027`, `union_branch_label_3027`).
* W6-opt-1 (push_filters_down Case 2, see proofs/optimizer_rewrites
  `filter_pushed_through_boundary`): the pushed read is unbound
  (`pushed_read_unbound`).
-/
namespace FalkorBinder.Pr2845

/-- `build_import_projections` (binder.rs:979): one `project_name` per
imported variable, in the HashMap's iteration order (`imported`, any order),
paired with the outer variable it reads. The final `sort_by` name only
reorders the pairs. -/
def importProj (e : Env) (scope : Nat) : List (Name × Var) → Env × List (Var × Var)
  | [] => (e, [])
  | (n, outer) :: rest =>
    let (e1, inner) := projectName e scope n outer.ty
    let (e2, ps) := importProj e1 scope rest
    (e2, (inner, outer) :: ps)

/-- Bound clauses: the entry `With` (its `(inner, outer)` pairs) or any other clause. -/
inductive Clause
  | with_ (exprs : List (Var × Var))
  | other (tag : Nat)
  deriving DecidableEq, Repr

/-- The merged Query arm (binder.rs:845-871): push a fresh scope (index
`outerLen`), always emit the entry projection, then the remaining clauses. -/
def bodyNew (outerLen : Nat) (imported : List (Name × Var)) (rest : List Clause) : List Clause :=
  .with_ (importProj [] outerLen imported).2 :: rest

/-- origin/main's Query arm: the `With` only when something is imported. -/
def bodyOld (outerLen : Nat) (imported : List (Name × Var)) (rest : List Clause) : List Clause :=
  if imported.isEmpty then rest else .with_ (importProj [] outerLen imported).2 :: rest

/-- The UNION arm (binder.rs:899-925): each branch binder starts with
`env_stack: vec![HashMap::new()]`, so its scope is 0 (origin/main and this
PR); #3027 changes it to `env_stack.len()` empty tables + 1, i.e. scope
`outerLen`. -/
def branchScope (outerLen : Nat) (with3027 : Bool) : Nat := if with3027 then outerLen else 0

def branchNew (outerLen : Nat) (with3027 : Bool) (imported : List (Name × Var))
    (rest : List Clause) : List Clause :=
  .with_ (importProj [] (branchScope outerLen with3027) imported).2 :: rest

/-! ## Claim 1: every body starts with its entry projection -/

theorem bodyNew_entry (o : Nat) (imp : List (Name × Var)) (rest : List Clause) :
    ∃ ps, bodyNew o imp rest = .with_ ps :: rest ∧ ps.length = imp.length := by
  refine ⟨_, rfl, ?_⟩
  suffices ∀ e, (importProj e o imp).2.length = imp.length from this []
  induction imp with
  | nil => intro e; rfl
  | cons p r ih => intro e; obtain ⟨n, v⟩ := p; simp [importProj, ih]

theorem branchNew_entry (o : Nat) (b : Bool) (imp : List (Name × Var)) (rest : List Clause) :
    ∃ ps, branchNew o b imp rest = .with_ ps :: rest := ⟨_, rfl⟩

/-- An import-less body: the entry projection names nothing. -/
theorem bodyNew_empty (o : Nat) (rest : List Clause) : bodyNew o [] rest = .with_ [] :: rest := rfl

/-- With at least one import the PR changes nothing. -/
theorem bodyNew_eq_old (o : Nat) (imp : List (Name × Var)) (rest : List Clause)
    (h : imp ≠ []) : bodyNew o imp rest = bodyOld o imp rest := by
  cases imp with
  | nil => exact absurd rfl h
  | cons _ _ => rfl

/-! ## Claim 3: fresh-id minting -/

/-- Names distinct: what a `HashMap` key set guarantees. -/
def Distinct (imp : List (Name × Var)) : Prop := (imp.map (·.1)).Nodup

theorem has_false_of_not_mem (e : Env) (k : Name) (h : k ∉ e.map (·.1)) : e.has k = false := by
  simp only [Env.has]
  rw [List.any_eq_false]
  intro p hp hk
  apply h
  simp only [beq_iff_eq] at hk
  exact List.mem_map.mpr ⟨p, hp, hk⟩

/-- The import projections mint ids `len e, len e + 1, …` in order, all in
the body scope, and leave a table of size `len e + k` whose keys are the
old keys followed by the imported names. -/
theorem importProj_spec (s : Nat) :
    ∀ (imp : List (Name × Var)) (e : Env), Distinct imp →
      (∀ p ∈ imp, p.1 ∉ e.map (·.1)) →
      (importProj e s imp).1.length = e.length + imp.length ∧
      (importProj e s imp).1.map (·.1) = e.map (·.1) ++ imp.map (·.1) ∧
      (importProj e s imp).2.map (·.1.id) = (List.range imp.length).map (e.length + ·) ∧
      ((importProj e s imp).1.ids = e.ids ++ (List.range imp.length).map (e.length + ·)) ∧
      ∀ q ∈ (importProj e s imp).2, q.1.scope = s
  | [], e, _, _ => by simp [importProj]
  | (n, v) :: rest, e, hd, hfresh => by
    have hn : n ∉ e.map (·.1) := hfresh (n, v) (by simp)
    have hk := has_false_of_not_mem e n hn
    have hdist : Distinct rest := (List.nodup_cons.mp hd).2
    have hnr : n ∉ rest.map (·.1) := (List.nodup_cons.mp hd).1
    have hfresh' : ∀ p ∈ rest, p.1 ∉ (e.insert n (freshVar e n v.ty s)).map (·.1) := by
      intro p hp
      rw [Env.insert_new_eq hk]
      simp only [List.map_append, List.map_cons, List.map_nil, List.mem_append, List.mem_singleton]
      rintro (h | h)
      · exact hfresh p (by simp [hp]) h
      · exact hnr (h ▸ List.mem_map.mpr ⟨p, hp, rfl⟩)
    obtain ⟨ih1, ih2, ih3, ih4, ih5⟩ := importProj_spec s rest _ hdist hfresh'
    have hins := Env.insert_new_eq (v := freshVar e n v.ty s) hk
    simp only [hins] at ih1 ih2 ih3 ih4 ih5
    simp only [importProj, projectName, hins]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [ih1]; simp; omega
    · rw [ih2]; simp
    · simp only [freshVar] at ih3
      simp only [List.map_cons, freshVar, List.length_cons]
      rw [ih3, List.range_succ_eq_map]
      simp only [List.length_append, List.length_cons, List.length_nil, List.map_cons,
        List.map_map, Function.comp_def, Nat.add_zero]
      congr 1
      apply List.map_congr_left; intro x _; omega
    · simp only [freshVar] at ih4
      simp only [List.length_cons, freshVar]
      rw [ih4, List.range_succ_eq_map]
      simp only [Env.ids, List.map_append, List.map_cons, List.map_nil, List.length_append,
        List.length_cons, List.length_nil, List.map_map, Function.comp_def, Nat.add_zero,
        List.append_assoc, List.cons_append, List.nil_append]
      congr 2
      apply List.map_congr_left; intro x _; omega
    · intro q hq
      simp only [List.mem_cons] at hq
      rcases hq with rfl | hq
      · rfl
      · exact ih5 q hq

/-- Every inner id of the entry projection is below the table size the body
continues from: `mint_fresh_iff`'s condition holds, so every id the body
mints afterwards (`fresh_var` at `len()`) avoids them. -/
theorem entry_ids_below (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp) :
    let r := importProj [] s imp
    r.1.Covers ∧ (∀ j, r.1.length + j ∉ r.2.map (·.1.id)) ∧ r.1.length = imp.length := by
  obtain ⟨h1, -, h3, h4, -⟩ := importProj_spec s imp [] hd (by simp)
  simp only [List.length_nil, Nat.zero_add, List.map_id'] at h1 h3 h4
  refine ⟨?_, ?_, h1⟩
  · intro i hi
    simp only [Env.ids, List.map_nil, List.nil_append] at h4
    rw [Env.ids, h4] at hi
    rw [h1]; simpa using hi
  · rw [mint_fresh_iff, h3, h1]; intro i hi; simpa using hi

/-- The body scope (`outerLen`) is never a live outer scope. -/
theorem body_scope_disjoint (outerLen s : Nat) (h : s < outerLen) : outerLen ≠ s := by omega

/-! ## Claim 2 for UNION bodies: (scope, id) keys still alias -/

/-- `node_labels: HashMap<(u32,u32), OrderSet<..>>` and `update_graph_labels`
(binder.rs:160-190) for a non-optional MATCH node: the accumulated labels
for the key `(scope_id, id)` replace the node's own. -/
abbrev Labels := List ((Nat × Nat) × List Name)

def relabel (L : Labels) (v : Var) (own : List Name) : List Name :=
  match L.lookup (v.scope, v.id) with
  | some acc => acc
  | none => own

/-- `MATCH (n:A) CALL { MATCH (q:B) RETURN q UNION MATCH (q:B) RETURN q }`:
step 2b of `bind_call_subquery` (binder.rs:731) runs the OUTER binder's
`update_all_node_labels` over the bound body, recursing into the UNION
branches (binder.rs:140-143). The outer table holds `(0,0) ↦ [A]` for `n`;
the branch's `q` is minted at scope 0 (the branch binder's only scope) after
its entry projection (no imports), id 0: same key. The branch scans `(q:A)`. -/
def outerLabels : Labels := [((0, 0), ["A"])]

def branchQ (with3027 : Bool) : Var :=
  let s := branchScope 1 with3027
  freshVar (importProj [] s []).1 "q" .node s

theorem union_branch_label_alias : relabel outerLabels (branchQ false) ["B"] = ["A"] := by decide

/-- #3027's scope numbering removes the collision: with every key of the
outer table in a live outer scope (`< outerLen`; step 2c of
`bind_call_subquery` removes inner-scope keys), a branch variable's key
`(outerLen, _)` is never found. -/
theorem relabel_3027 (L : Labels) (outerLen : Nat) (hL : ∀ p ∈ L, p.1.1 < outerLen)
    (v : Var) (hv : v.scope = branchScope outerLen true) (own : List Name) :
    relabel L v own = own := by
  unfold relabel
  have : L.lookup (v.scope, v.id) = none := by
    rw [List.lookup_eq_none_iff]
    intro p hp
    have := hL p hp
    simp only [bne_iff_ne, ne_eq]
    intro heq
    rw [← heq] at this
    simp [hv, branchScope] at this
  rw [this]

theorem union_branch_label_3027 : relabel outerLabels (branchQ true) ["B"] = ["B"] := by decide

/-! ## Runtime side: the entry projection is a record boundary


A runtime row is a dense array indexed by `Variable.id` alone
(runtime/row.rs; `scope_id` never reaches the row). Modelled as a function
`Nat → Option V` (`none` = unbound slot).

* `Argument` (runtime/ops/argument.rs) hands the sub-plan the outer row as is.
* `ProjectOp::eval` (runtime/ops/project.rs:54-147) builds a fresh row from
  its projections only (`Row::with_capacity`, `result.insert(target, …)`,
  then `BatchBuilder::push_row`) and drops its input.
-/
abbrev RowF (V : Type) := Nat → Option V

/-- The fresh row an entry projection `(inner, outer)` pairs builds. -/
def projRow {V : Type} (ps : List (Var × Var)) (r : RowF V) : RowF V := fun i =>
  match ps.find? (fun p => p.1.id == i) with
  | some p => r p.2.id
  | none => none

/-- What the first operator of the body sees: origin/main had no projection
when nothing was imported (the `Argument` row itself), the merged code always
has one. -/
def bodyInputOld {V : Type} (ps : List (Var × Var)) (imported : Bool) (r : RowF V) : RowF V :=
  if imported then projRow ps r else r

def bodyInputNew {V : Type} (ps : List (Var × Var)) (r : RowF V) : RowF V := projRow ps r

theorem projRow_unbound {V : Type} (ps : List (Var × Var)) (r : RowF V) (i : Nat)
    (h : i ∉ ps.map (·.1.id)) : projRow ps r i = none := by
  unfold projRow
  have : ps.find? (fun p => p.1.id == i) = none := by
    rw [List.find?_eq_none]
    intro p hp heq
    simp only [beq_iff_eq] at heq
    exact h (heq ▸ List.mem_map.mpr ⟨p, hp, rfl⟩)
  rw [this]

/-- An imported slot carries exactly the outer value (targets are distinct:
`importProj_spec` mints them `0..k-1`). -/
theorem projRow_import {V : Type} (ps : List (Var × Var)) (r : RowF V) (p : Var × Var)
    (hp : p ∈ ps) (hd : (ps.map (·.1.id)).Nodup) : projRow ps r p.1.id = r p.2.id := by
  unfold projRow
  induction ps with
  | nil => simp at hp
  | cons q qs ih =>
    simp only [List.find?_cons]
    by_cases hq : q.1.id = p.1.id
    · have : q = p := by
        rcases List.mem_cons.mp hp with rfl | hmem
        · rfl
        · exfalso
          have hn := (List.nodup_cons.mp hd).1
          apply hn
          show q.1.id ∈ _
          rw [hq]
          exact List.mem_map.mpr ⟨p, hmem, rfl⟩
      subst this; simp
    · have hmem : p ∈ qs := by
        rcases List.mem_cons.mp hp with rfl | hmem
        · exact absurd rfl hq
        · exact hmem
      have hb : (q.1.id == p.1.id) = false := by simpa using hq
      rw [hb]
      exact ih hmem (List.nodup_cons.mp hd).2

/-- **Claim 2 (no slot aliasing), plain bodies.** For any outer row, every
variable the body mints itself (id `≥ k`, `entry_ids_below`) is unbound
when the body starts — so a body scan endpoint never looks already bound
(#2601) and a body predicate never reads an outer value (#2602). -/
theorem body_own_unbound {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (j : Nat) (hj : imp.length ≤ j) :
    bodyInputNew (importProj [] s imp).2 r j = none := by
  apply projRow_unbound
  obtain ⟨-, -, h3, -, -⟩ := importProj_spec s imp [] hd (by simp)
  rw [h3]; simp; omega

/-- … and every imported variable reads its outer value. -/
theorem body_import_value {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (p : Var × Var) (hp : p ∈ (importProj [] s imp).2) :
    bodyInputNew (importProj [] s imp).2 r p.1.id = r p.2.id := by
  apply projRow_import _ _ _ hp
  obtain ⟨-, -, h3, -, -⟩ := importProj_spec s imp [] hd (by simp)
  rw [h3]; simp only [List.length_nil, Nat.zero_add, List.map_id']
  exact List.nodup_range

/-- Runtime choice of a scan endpoint (#2601): an already-bound slot turns a
full scan into an expand from that slot. -/
def scanMode {V : Type} (r : RowF V) (endpoint : Nat) : Bool := (r endpoint).isSome

theorem new_scan_scans {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (j : Nat) (hj : imp.length ≤ j) :
    scanMode (bodyInputNew (importProj [] s imp).2 r) j = false := by
  simp [scanMode, body_own_unbound s imp hd r j hj]

/-- At that `Argument` the row is the Optional's left input, i.e. the entry
projection's output: slot 0 is unbound for every outer row, so the pushed
`n.id` read fails ("Variable n not found"; C returns `[[1,3],[2,3],[3,3]]`). -/
theorem pushed_read_unbound {V : Type} (r : RowF V) :
    bodyInputNew (importProj [] 1 []).2 r 0 = none := rfl

end FalkorBinder.Pr2845
