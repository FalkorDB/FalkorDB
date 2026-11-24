import FalkorBinder.Table
/-
# Name resolution: `resolve_name`, `define_name_in_scope`, `project_name`

The binder state that name resolution reads (binder.rs), restricted to
the two innermost scopes: `resolve_name` only ever looks at the current table
and, when `use_parent_scope` is set, at `env_stack[len - 2]`
(binder.rs).
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
  /-- `match_rel_vars` (#2981): the `(id, scope_id)` of every named relationship a MATCH bound -/
  matchRels : List (Nat × Nat) := []

/-- The name `resolve_name` treats specially (binder.rs). Since #3303 no `_anon` name
is special: an unnamed entity has no name to resolve. -/
def isSpecial (n : Name) : Bool :=
  n == "__agg_order_by_placeholder__"

/-- A written `_anon…` name is an ordinary name (#3303; before, `resolve_name` minted a fresh
variable for it whatever the scope held). -/
theorem isSpecial_anon_name : isSpecial "_anon_0" = false := by decide

/-- Innermost-first lookup in the comprehension locals (binder.rs). -/
def lookupLocals : List Env → Name → Option Var
  | [], _ => none
  | l :: ls, n => match lookupLocals ls n with
    | some v => some v
    | none => l.find? n

/-- `Binder::resolve_name` (binder.rs), for a non-special name.
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
      -- binder.rs: copy into the current scope, id = len()
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
          exact mint_new_covers' hc (hk h2) _ rfl
        · rw [← h.1]; exact hc
      · rw [← h.1]; exact hc
    · rw [← h.1]; exact hc
  · rw [← h.1]; exact hc

/-- The reserved key `resolve_name` files an `__agg_order_by_placeholder__` under (binder.rs). -/
def placeholderKey (id : Nat) : Name := s!"__agg_placeholder_{id}"

/-- `Binder::resolve_name` in full (binder.rs): the aggregate-ORDER-BY placeholder always
gets a fresh variable at `len()` under its own key; every other name is `resolve`. -/
def resolveName (st : St) (locals : List Env) (n : Name) : Except String (St × Var) :=
  if n == "__agg_order_by_placeholder__" then
    let v := freshVar st.cur n .any st.depth
    .ok ({ st with cur := st.cur.insert (placeholderKey v.id) v }, v)
  else resolve st locals n

/-- Any name but the placeholder resolves as `resolve` does. -/
theorem resolveName_other (st : St) (locals : List Env) (n : Name)
    (h : n ≠ "__agg_order_by_placeholder__") : resolveName st locals n = resolve st locals n := by
  simp [resolveName, h]

/-- The placeholder is minted fresh at `len()` in the current scope, whatever the scope holds, and
never reads a local, the parent or the table. -/
theorem resolveName_placeholder (st : St) (locals : List Env) :
    ∃ v, resolveName st locals "__agg_order_by_placeholder__" =
      .ok ({ st with cur := st.cur.insert (placeholderKey st.cur.length) v }, v) ∧
      v.id = st.cur.length ∧ v.scope = st.depth ∧ v.name = some "__agg_order_by_placeholder__" := by
  refine ⟨freshVar st.cur "__agg_order_by_placeholder__" .any st.depth, ?_, rfl, rfl, rfl⟩
  simp [resolveName, freshVar]

/-- Under a fresh reserved key the placeholder keeps the table covering (the key is
`__agg_placeholder_<len>`; `KeysFresh` for it is the hypothesis, as for the loop variables). -/
theorem resolveName_placeholder_covers (st : St) (locals : List Env) (hc : st.cur.Covers)
    (hk : st.cur.has (placeholderKey st.cur.length) = false) (st' : St) (v : Var)
    (h : resolveName st locals "__agg_order_by_placeholder__" = .ok (st', v)) : st'.cur.Covers := by
  simp only [resolveName, beq_self_eq_true, if_true, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact mint_new_covers hc hk _ _ _

/-! ## `define_name_in_scope` and `ensure_type` -/

/-- `Binder::ensure_type` (binder.rs). -/
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

/-- `Binder::define_name_in_scope` (binder.rs). Since #3303 every name is looked up
(no `_anon` name is minted afresh unconditionally). -/
def defineName (e : Env) (depth : Nat) (n : Name) (ty : Ty) (allowReuse : Bool) :
    Except String (Env × Var) :=
  match e.find? n with
  | some ex =>
    if !allowReuse then .error s!"Variable `{n}` already declared"
    else if !ensureType ex.ty ty then
      .error s!"The alias '{n}' was specified for both a node and a relationship."
    else .ok (e, ex)
  | none => let v := freshVar e n ty depth; .ok (e.insert n v, v)

/-- A name already in the current table is reused (when reuse is allowed and the kinds are
compatible): same variable, same table. -/
theorem defineName_reuse (e : Env) (d : Nat) (n : Name) (ty : Ty) (ex : Var)
    (hf : e.find? n = some ex) (hty : ensureType ex.ty ty = true) :
    defineName e d n ty true = .ok (e, ex) := by
  simp [defineName, hf, hty]

/-- Without reuse an existing name is an error. -/
theorem defineName_noreuse (e : Env) (d : Nat) (n : Name) (ty : Ty) (ex : Var)
    (hf : e.find? n = some ex) :
    defineName e d n ty false = .error s!"Variable `{n}` already declared" := by
  simp [defineName, hf]

/-- A name not in the table is minted at `len()` and inserted, and `Scope::get` returns it. -/
theorem defineName_fresh (e : Env) (d : Nat) (n : Name) (ty : Ty) (b : Bool)
    (hf : e.find? n = none) :
    ∃ v, defineName e d n ty b = .ok (e.insert n v, v) ∧ v = freshVar e n ty d ∧
      (e.insert n v).find? n = some v := by
  exact ⟨_, by simp [defineName, hf], rfl, Env.find?_insert_self _ _ _⟩

/-- A fresh definition keeps the table covering. -/
theorem defineName_covers (e : Env) (d : Nat) (n : Name) (ty : Ty) (b : Bool) (hc : e.Covers)
    (e' : Env) (v : Var) (h : defineName e d n ty b = .ok (e', v)) : e'.Covers := by
  unfold defineName at h
  cases hf : e.find? n with
  | some ex =>
    simp only [hf] at h
    split at h
    · simp at h
    · split at h
      · simp at h
      · simp only [Except.ok.injEq, Prod.mk.injEq] at h; rw [← h.1]; exact hc
  | none =>
    simp only [hf, Except.ok.injEq, Prod.mk.injEq] at h
    rw [← h.1]
    have hn : e.has n = false := by
      cases hh : e.has n
      · rfl
      · have := (Env.has_iff_find e n).1 hh; simp [hf] at this
    exact mint_new_covers hc hn _ _ _

/-! ## Aliases of pattern entities -/

/-- `Binder::define_alias_in_scope` (binder.rs): a name is `define_name_in_scope`d; an
unnamed entity reuses the `lookup_anon` hit of the current scope, else gets a nameless variable
minted at `len()` and pushed to the scope's unnamed list. -/
def defineAlias (st : St) (a : Alias) (ty : Ty) (allowReuse : Bool) : Except String (St × Var) :=
  match a with
  | .named n => match defineName st.cur st.depth n ty allowReuse with
    | .ok (e, v) => .ok ({ st with cur := e }, v)
    | .error err => .error err
  | .anon k => match st.cur.lookupAnon k with
    | some v => .ok (st, v)
    | none =>
      let v := freshAnon st.cur k ty st.depth
      .ok ({ st with cur := st.cur.pushUnnamed v }, v)

/-- `Binder::lookup_alias` (binder.rs). -/
def lookupAlias (st : St) : Alias → Option Var
  | .named n => st.cur.find? n
  | .anon k => st.cur.lookupAnon k

/-- `Binder::resolve_alias` (binder.rs). -/
def resolveAlias (st : St) (locals : List Env) : Alias → Except String (St × Var)
  | .named n => resolve st locals n
  | .anon k => match st.cur.lookupAnon k with
    | some v => .ok (st, v)
    | none => .error "unnamed entity is not bound"

/-- `denotes` (binder.rs): whether a bound variable is the one the parser wrote. -/
def denotes (v : Var) : Alias → Bool
  | .named n => v.name == some n
  | .anon k => v.anon == some k

/-- An unnamed entity's variable is new, nameless, at `len()`, reached by no name, and is the one
every later `lookup_alias` / `resolve_alias` of that entity returns. -/
theorem defineAlias_anon_fresh (st : St) (k : Nat) (t : Ty) (b : Bool) (h : st.cur.lookupAnon k = none) :
    ∃ st' v, defineAlias st (.anon k) t b = .ok (st', v) ∧ v.name = none ∧ v.anon = some k ∧
      v.id = st.cur.length ∧ st'.cur = st.cur.pushUnnamed v ∧
      (∀ n, st'.cur.find? n = st.cur.find? n) ∧ lookupAlias st' (.anon k) = some v ∧
      resolveAlias st' [] (.anon k) = .ok (st', v) ∧ denotes v (.anon k) = true := by
  have hl := Env.lookupAnon_pushUnnamed_self st.cur (freshAnon st.cur k t st.depth) k
    (by simp [freshAnon]) h
  simp only [defineAlias, h]
  refine ⟨_, _, rfl, rfl, rfl, rfl, rfl, fun n => Env.pushUnnamed_find _ _ n, ?_, ?_,
    by simp [denotes, freshAnon]⟩
  · simpa [lookupAlias] using hl
  · simp [resolveAlias, hl]

/-- A second binding of the same unnamed entity returns the same variable and changes nothing. -/
theorem defineAlias_anon_reuse (st : St) (k : Nat) (v : Var) (t : Ty) (b : Bool)
    (h : st.cur.lookupAnon k = some v) : defineAlias st (.anon k) t b = .ok (st, v) := by
  simp [defineAlias, h]

/-- An unnamed entity's minted variable keeps the table covering. -/
theorem defineAlias_anon_covers (st : St) (k : Nat) (t : Ty) (b : Bool) (hc : st.cur.Covers)
    (st' : St) (v : Var) (h : defineAlias st (.anon k) t b = .ok (st', v)) : st'.cur.Covers := by
  unfold defineAlias at h
  cases hl : st.cur.lookupAnon k <;> simp [hl] at h
  · obtain ⟨rfl, rfl⟩ := h; exact pushUnnamed_covers hc _ _ _
  · rw [← h.1]; exact hc

/-- A written name, `_anon_0` included, is an ordinary name: `defineAlias (.named n)` is
`define_name_in_scope`, so a written `_anon_0` already in scope is reused, not shadowed. -/
theorem defineAlias_named_anon_spelling (st : St) (v : Var) (hv : st.cur.find? "_anon_0" = some v)
    (hty : ensureType v.ty .node = true) :
    defineAlias st (.named "_anon_0") .node true = .ok (st, v) := by
  simp [defineAlias, defineName, hv, hty]

/-- After any successful `define_alias_in_scope`, `lookup_alias` returns the bound variable
(a named alias through its name, an unnamed one through its number). -/
theorem lookupAlias_defineAlias (st : St) (a : Alias) (t : Ty) (b : Bool) (st' : St) (v : Var)
    (h : defineAlias st a t b = .ok (st', v)) : lookupAlias st' a = some v := by
  cases a with
  | named n =>
    unfold defineAlias at h
    cases hd : defineName st.cur st.depth n t b with
    | error e => simp [hd] at h
    | ok r =>
      obtain ⟨e', v'⟩ := r
      simp only [hd, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      unfold defineName at hd
      cases hf : st.cur.find? n with
      | some ex =>
        simp only [hf] at hd
        split at hd
        · simp at hd
        · split at hd
          · simp at hd
          · simp only [Except.ok.injEq, Prod.mk.injEq] at hd
            obtain ⟨rfl, rfl⟩ := hd
            simpa [lookupAlias] using hf
      | none =>
        simp only [hf, Except.ok.injEq, Prod.mk.injEq] at hd
        obtain ⟨rfl, rfl⟩ := hd
        simpa [lookupAlias] using Env.find?_insert_self _ _ _
  | anon k =>
    unfold defineAlias at h
    cases hl : st.cur.lookupAnon k with
    | some w => simp only [hl, Except.ok.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; simpa [lookupAlias] using hl
    | none =>
      simp only [hl, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simpa [lookupAlias] using Env.lookupAnon_pushUnnamed_self st.cur (freshAnon st.cur k t st.depth) k
        (by simp [freshAnon]) hl

/-- `resolve_alias` of an alias `lookup_alias` finds returns it, leaving the state alone. -/
theorem resolveAlias_of_lookup (st : St) (a : Alias) (v : Var)
    (h : lookupAlias st a = some v) : resolveAlias st [] a = .ok (st, v) := by
  cases a with
  | named n =>
    have h' : st.cur.find? n = some v := h
    simp [resolveAlias, resolve, h', lookupLocals]
  | anon k =>
    have h' : st.cur.lookupAnon k = some v := h
    simp [resolveAlias, h']

/-- A name never denotes an unnamed variable, nor an unnamed alias a named one. -/
theorem denotes_exclusive (v : Var) (n : Name) (k : Nat) :
    ¬ (denotes v (.named n) = true ∧ v.anon = some k ∧ v.name = none) := by
  intro ⟨h1, _, h3⟩
  simp [denotes, h3] at h1

/-- The variables the binder mints denote their own alias: a named one from `fresh_var`, an unnamed
one from `define_alias_in_scope`. -/
theorem denotes_freshVar (e : Env) (n : Name) (t : Ty) (s : Nat) :
    denotes (freshVar e n t s) (.named n) = true := by simp [denotes, freshVar]

theorem denotes_freshAnon (e : Env) (k : Nat) (t : Ty) (s : Nat) :
    denotes (freshAnon e k t s) (.anon k) = true := by simp [denotes, freshAnon]

/-- An unnamed variable denotes exactly its own number, and no spelling of a name. -/
theorem denotes_anon_iff (v : Var) (k j : Nat) (h : v.anon = some k) :
    denotes v (.anon j) = true ↔ j = k := by
  simp only [denotes, h, beq_iff_eq, Option.some.injEq]
  exact ⟨fun x => x.symm, fun x => x.symm⟩

/-- `define_name_in_scope` looks only at the *current* table: a name that
`resolve_name` would find through the parent copy is **redefined** as a
fresh, unconstrained variable. This is root cause 6 — the pattern predicate
in `MATCH (n) WITH n.v AS k WHERE (n)-->() RETURN k` gets its own `n` (Rust
returns every row; C one). Pattern comprehensions pre-copy parent names
(binder.rs); pattern predicates (binder.rs) do not. -/
theorem defineName_ignores_parent :
    let pv : Var := ⟨"n", 0, 0, .node, none⟩
    let st : St := { cur := [("k", ⟨"k", 0, 1, .any, none⟩)], parent := [("n", pv)], depth := 1,
                     useParent := true, p2c := [], cfp := [] }
    (∃ st' v, resolve st [] "n" = .ok (st', v) ∧ v.name = "n") ∧
    (∃ e v, defineName st.cur st.depth "n" .node true = .ok (e, v) ∧ v ≠ pv ∧
      v.scope = 1 ∧ v.id = 1) := by
  refine ⟨⟨_, _, rfl, rfl⟩, ⟨_, _, rfl, by decide, rfl, rfl⟩⟩

/-- `Binder::project_name` (binder.rs): always a fresh id, and the
key is overwritten if present. -/
def projectName (e : Env) (depth : Nat) (n : Name) (ty : Ty) : Env × Var :=
  let v := freshVar e n ty depth
  (e.insert n v, v)

/-- `project_name` over an already-present key leaves `len()` unchanged but
hands out id `len()`: the new id is **not covered**. Every caller is guarded
against this — `bind_projection` pushes an empty scope and rejects duplicate
aliases (binder.rs), `build_import_projections` runs on an empty
table, and `bind_call_subquery` first rejects names already in the outer
scope (binder.rs). -/
theorem projectName_present_uncovered :
    let e : Env := [("a", ⟨"a", 0, 0, .any, none⟩)]
    ¬ (projectName e 0 "a" .any).1.Covers := by
  intro e h
  have := h 1 (by decide)
  simp [projectName, e, Env.insert, Env.has, freshVar] at this

theorem projectName_new_covers {e : Env} (h : e.Covers) {n : Name} (hk : e.has n = false)
    (d : Nat) (t : Ty) : (projectName e d n t).1.Covers :=
  mint_new_covers h hk _ _ _

end FalkorBinder
