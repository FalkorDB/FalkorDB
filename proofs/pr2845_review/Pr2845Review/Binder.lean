/-
# PR #2845, binder side: the entry projection and fresh-id minting

Model of the merged `bind_call_body` (graph/src/planner/binder.rs:824-971 at
merge commit 39560cbdf; identical on origin/main a9377c636, where #2845 landed as eb470521b) and `build_import_projections` (binder.rs:979-1003).
Scope tables, `fresh_var` and `mint_fresh_iff` are copied from
proofs/binder (Table.lean) unchanged.
-/
namespace Pr2845

abbrev Name := String

inductive Ty
  | any | node | rel | path
  deriving DecidableEq, Repr

/-- `parser::ast::Variable`; the record slot is `id` alone. -/
structure Var where
  name  : Name
  id    : Nat
  scope : Nat
  ty    : Ty
  deriving DecidableEq, Repr

abbrev Env := List (Name × Var)

namespace Env
def has (e : Env) (k : Name) : Bool := e.any (·.1 == k)
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == k then (k, v) else p) else e ++ [(k, v)]
def ids (e : Env) : List Nat := e.map (·.2.id)
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length
end Env

/-- `Binder::fresh_var` (binder.rs:2243): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name, id := e.length, scope, ty }

/-- copied from proofs/binder `mint_fresh_iff`. -/
theorem mint_fresh_iff (len : Nat) (H : List Nat) :
    (∀ j, len + j ∉ H) ↔ (∀ i ∈ H, i < len) := by
  constructor
  · intro h i hi
    apply Classical.byContradiction
    intro hlt
    have := h (i - len)
    rw [show len + (i - len) = i by omega] at this
    exact this hi
  · intro h j hj
    have := h _ hj
    omega

theorem Env.insert_new_eq {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    e.insert k v = e ++ [(k, v)] := by
  simp [Env.insert, h]

/-- `project_name` (binder.rs:2165) on the body's own (top) scope table. -/
def projectName (e : Env) (scope : Nat) (n : Name) (t : Ty) : Env × Var :=
  let v := freshVar e n t scope
  (e.insert n v, v)

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

/-- origin/main emitted no boundary there (the #2601/#2602 root cause). -/
theorem pre2845_bodyOld_empty (o : Nat) (rest : List Clause) : bodyOld o [] rest = rest := rfl

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

end Pr2845
