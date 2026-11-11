import FalkorBinder.Table
/-
# Name resolution: `resolve_name`, `define_name_in_scope`, `project_name`

The binder state that name resolution reads (binder.rs:45-71), restricted to
the two innermost scopes: `resolve_name` only ever looks at the current table
and, when `use_parent_scope` is set, at `env_stack[len - 2]`
(binder.rs:2216-2219).
-/
namespace FalkorBinder

structure St where
  /-- `env_stack.last()` -/
  cur : Env
  /-- `env_stack[len - 2]` (empty at depth 0) -/
  parent : Env
  /-- `env_stack.len() - 1`: the scope id of `cur` -/
  depth : Nat
  /-- `use_parent_scope` (set by `push_scope`, cleared by `commit_scope`) -/
  useParent : Bool
  /-- `parent_to_child_scope`: parent name ↦ its projected child variable -/
  p2c : Env
  /-- `copy_from_parent`: name ↦ (parent variable, its copy in `cur`) -/
  cfp : List (Name × (Var × Var))

/-- Names the binder reserves for itself (binder.rs:2181, 2192). -/
def isSpecial (n : Name) : Bool :=
  n == "__agg_order_by_placeholder__" || n.startsWith "_anon"

/-- Innermost-first lookup in the comprehension locals (binder.rs:2201-2205). -/
def lookupLocals : List Env → Name → Option Var
  | [], _ => none
  | l :: ls, n => match lookupLocals ls n with
    | some v => some v
    | none => l.find? n

/-- `Binder::resolve_name` (binder.rs:2175-2241), for a non-special name.
`locals` is in push order, so the *last* one is the innermost — matching
`locals.iter().rev()`. Returns the new state and the variable. -/
def resolve (st : St) (locals : List Env) (n : Name) : Except String (St × Var) :=
  match lookupLocals locals n with
  | some v => .ok (st, v)
  | none =>
  match st.cur.find? n with
  | some v => .ok (st, v)
  | none =>
  if st.useParent then
    match st.p2c.find? n with
    | some v => .ok (st, v)
    | none =>
    match st.cfp.lookup n with
    | some (_, v) => .ok (st, v)
    | none =>
    match st.parent.find? n with
    | some pv =>
      -- binder.rs:2223-2236: copy into the current scope, id = len()
      let copy : Var := { pv with id := st.cur.length, scope := st.depth }
      .ok ({ st with cur := st.cur.insert n copy, cfp := (n, (pv, copy)) :: st.cfp }, copy)
    | none => .error s!"'{n}' not defined"
  else .error s!"'{n}' not defined"

/-- The spec's visibility (openCypher, and C for ORDER BY/WHERE after a
projection): a name is in scope if a comprehension binds it, the current
clause's scope binds it, or — only while a projection's ORDER BY / WHERE /
SKIP / LIMIT is being bound — the scope before the projection binds it. -/
def visible (st : St) (locals : List Env) (n : Name) : Prop :=
  (lookupLocals locals n).isSome ∨ (st.cur.find? n).isSome ∨
  (st.useParent = true ∧
    ((st.p2c.find? n).isSome ∨ (st.cfp.lookup n).isSome ∨ (st.parent.find? n).isSome))

/-- `resolve_name` succeeds exactly on the visible names. -/
theorem resolve_ok_iff (st : St) (locals : List Env) (n : Name) :
    (∃ r, resolve st locals n = .ok r) ↔ visible st locals n := by
  unfold resolve visible
  cases h1 : lookupLocals locals n <;> cases h2 : st.cur.find? n <;>
    cases h3 : st.useParent <;> cases h4 : st.p2c.find? n <;>
    cases h5 : st.cfp.lookup n <;> cases h6 : st.parent.find? n <;>
    simp only [Option.isSome_some, Option.isSome_none] <;> simp

/-- …and what it returns is the innermost binding: a comprehension local
shadows the clause scope, which shadows the pre-projection scope. -/
theorem resolve_innermost (st : St) (locals : List Env) (n : Name) (st' : St) (v : Var)
    (h : resolve st locals n = .ok (st', v)) :
    lookupLocals locals n = some v ∨
    (lookupLocals locals n = none ∧ st.cur.find? n = some v) ∨
    (lookupLocals locals n = none ∧ st.cur.find? n = none ∧ st.useParent = true ∧
      (st.p2c.find? n = some v ∨
       (st.p2c.find? n = none ∧ ∃ pv, st.cfp.lookup n = some (pv, v)) ∨
       (st.p2c.find? n = none ∧ st.cfp.lookup n = none ∧
         ∃ pv, st.parent.find? n = some pv ∧
           v = { pv with id := st.cur.length, scope := st.depth }))) := by
  unfold resolve at h
  cases h1 : lookupLocals locals n <;> cases h2 : st.cur.find? n <;>
    cases h3 : st.useParent <;> cases h4 : st.p2c.find? n <;>
    cases h5 : st.cfp.lookup n <;> cases h6 : st.parent.find? n <;>
    simp only [h1, h2, h3, h4, h5, h6] at h <;>
    (try simp at h) <;> (try obtain ⟨rfl, rfl⟩ := h) <;> simp_all <;> exact ⟨_, rfl⟩

/-- The parent copy is minted under a *new* key (the name was not in `cur`),
so it keeps the current table covering (`mint_new_covers`). -/
theorem resolve_covers (st : St) (locals : List Env) (n : Name) (st' : St) (v : Var)
    (hc : st.cur.Covers) (hk : st.cur.find? n = none → st.cur.has n = false)
    (h : resolve st locals n = .ok (st', v)) : st'.cur.Covers := by
  unfold resolve at h
  cases h1 : lookupLocals locals n <;> simp [h1] at h
  · cases h2 : st.cur.find? n <;> simp [h2] at h
    · cases h3 : st.useParent <;> simp [h3] at h
      cases h4 : st.p2c.find? n <;> simp [h4] at h
      · cases h5 : st.cfp.lookup n <;> simp [h5] at h
        · cases h6 : st.parent.find? n <;> simp [h6] at h
          rw [← h.1]
          exact mint_new_covers hc (hk h2) _ _ _
        · rw [← h.1]; exact hc
      · rw [← h.1]; exact hc
    · rw [← h.1]; exact hc
  · rw [← h.1]; exact hc

/-! ## `define_name_in_scope` and `ensure_type` -/

/-- `Binder::ensure_type` (binder.rs:2259-2275). -/
def ensureType (existing ty : Ty) : Bool :=
  !((ty == .rel && (existing == .node || existing == .path)) ||
    (ty == .node && (existing == .rel || existing == .path)) ||
    (ty == .path && (existing == .node || existing == .rel)))

/-- The binder rejects exactly the three entity kinds used as one another. -/
theorem ensureType_table (e t : Ty) :
    ensureType e t = false ↔ (e ≠ t ∧ e ≠ .any ∧ t ≠ .any) := by
  cases e <;> cases t <;> decide

/-- openCypher's rule: a name already bound to a value of kind `e` may be
reused in a pattern as kind `t` only when `e = t`; C additionally accepts an
`Any`-typed (non-entity) binding and fails or filters at runtime. The binder
agrees with C (`WITH 1 AS a MATCH (a) RETURN a` returns `[1]` on both), and
both accept strictly more than openCypher — recorded as a spec gap, not a
Rust/C divergence. -/
theorem ensureType_accepts_any_binding : ensureType .any .node = true := by decide

/-- `Binder::define_name_in_scope` (binder.rs:2143-2163). -/
def defineName (e : Env) (depth : Nat) (n : Name) (ty : Ty) (allowReuse : Bool) :
    Except String (Env × Var) :=
  if !n.startsWith "_anon" then
    match e.find? n with
    | some ex =>
      if !allowReuse then .error s!"Variable `{n}` already declared"
      else if !ensureType ex.ty ty then
        .error s!"The alias '{n}' was specified for both a node and a relationship."
      else .ok (e, ex)
    | none => let v := freshVar e n ty depth; .ok (e.insert n v, v)
  else let v := freshVar e n ty depth; .ok (e.insert n v, v)

/-- `define_name_in_scope` looks only at the *current* table: a name that
`resolve_name` would find through the parent copy is **redefined** as a
fresh, unconstrained variable. This is root cause 6 — the pattern predicate
in `MATCH (n) WITH n.v AS k WHERE (n)-->() RETURN k` gets its own `n` (Rust
returns every row; C one). Pattern comprehensions pre-copy parent names
(binder.rs:1882-1892); pattern predicates (binder.rs:2073-2102) do not. -/
theorem defineName_ignores_parent :
    let pv : Var := ⟨"n", 0, 0, .node⟩
    let st : St := { cur := [("k", ⟨"k", 0, 1, .any⟩)], parent := [("n", pv)], depth := 1,
                     useParent := true, p2c := [], cfp := [] }
    (∃ st' v, resolve st [] "n" = .ok (st', v) ∧ v.name = "n") ∧
    (∃ e v, defineName st.cur st.depth "n" .node true = .ok (e, v) ∧ v ≠ pv ∧
      v.scope = 1 ∧ v.id = 1) := by
  refine ⟨⟨_, _, rfl, rfl⟩, ⟨_, _, rfl, by decide, rfl, rfl⟩⟩

/-- `Binder::project_name` (binder.rs:2165-2173): always a fresh id, and the
key is overwritten if present. -/
def projectName (e : Env) (depth : Nat) (n : Name) (ty : Ty) : Env × Var :=
  let v := freshVar e n ty depth
  (e.insert n v, v)

/-- `project_name` over an already-present key leaves `len()` unchanged but
hands out id `len()`: the new id is **not covered**. Every caller is guarded
against this — `bind_projection` pushes an empty scope and rejects duplicate
aliases (binder.rs:1123, 1150), `build_import_projections` runs on an empty
table, and `bind_call_subquery` first rejects names already in the outer
scope (binder.rs:766). -/
theorem projectName_present_uncovered :
    let e : Env := [("a", ⟨"a", 0, 0, .any⟩)]
    ¬ (projectName e 0 "a" .any).1.Covers := by
  intro e h
  have := h 1 (by decide)
  simp [projectName, e, Env.insert, Env.has, freshVar] at this

theorem projectName_new_covers {e : Env} (h : e.Covers) {n : Name} (hk : e.has n = false)
    (d : Nat) (t : Ty) : (projectName e d n t).1.Covers :=
  mint_new_covers h hk _ _ _

end FalkorBinder
