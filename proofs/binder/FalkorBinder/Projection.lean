import FalkorBinder.Resolve
import FalkorBinder.AggEq
/-
# WITH / RETURN: `bind_projection` (binder.rs)

An expression is abstracted to the names it mentions (in DFS order), whether
it is an aggregation, and whether it is a bare variable (the case that
records a `parent_to_child_scope` entry, binder.rs).

ORDER BY keys are first rewritten by `replace_projected_subtrees` (`AggEq.lean`):
`bindProjectionR` takes the names of the rewritten raw tree (`reVars`), so a
key equal to a projection reads its alias.
-/
namespace FalkorBinder

structure PExpr where
  vars  : List Name
  agg   : Bool := false
  /-- the expression is exactly `ExprIR::Variable` (binder.rs) -/
  bare  : Bool := false
  deriving Repr, DecidableEq

instance {α} [DecidableEq α] : DecidableEq (Except String α) := fun a b =>
  match a, b with
  | .ok x, .ok y =>
    if h : x = y then isTrue (h ▸ rfl) else isFalse (by intro h'; cases h'; exact h rfl)
  | .error x, .error y =>
    if h : x = y then isTrue (h ▸ rfl) else isFalse (by intro h'; cases h'; exact h rfl)
  | .ok _, .error _ => isFalse (by intro h; cases h)
  | .error _, .ok _ => isFalse (by intro h; cases h)

def PExpr.isVar (e : PExpr) : Option Name :=
  match e.vars, e.bare with
  | [v], true => some v
  | _, _ => none

/-- Bind every name of an expression, left to right, threading the state
(`bind_expr` → `resolve_name` per `ExprIR::Variable`, binder.rs). -/
def resolveAll (st : St) : List Name → Except String (St × List Var)
  | [] => .ok (st, [])
  | n :: ns => do
    let (st, v) ← resolve st [] n
    let (st, vs) ← resolveAll st ns
    pure (st, v :: vs)

/-- `push_scope` (binder.rs) as seen by name resolution. -/
def pushScope (st : St) : St :=
  { cur := [], parent := st.cur, depth := st.depth + 1, useParent := true,
    p2c := [], cfp := [], matchRels := st.matchRels }

structure Proj where
  items   : List (Name × PExpr)
  orderBy : List PExpr := []
  /-- WHERE, SKIP and LIMIT: bound after the snapshot at binder.rs -/
  tail    : List PExpr := []

/-- Project the items into the freshly pushed scope (binder.rs). -/
def projectItems (st : St) (seen : List Name) :
    List (Name × PExpr × List Var) → Except String St
  | [] => .ok st
  | (a, e, vs) :: rest =>
    if seen.contains a then
      .error "Error: Multiple result columns with the same name are not supported."
    else
      let (cur, bv) := projectName st.cur st.depth a .any
      let p2c := match e.isVar, vs with
        -- `var_name.name.as_ref().unwrap()` (binder.rs): a projected variable is named
        | some _, [v] => match v.name with
          | some n => st.p2c.insert n bv
          | none => st.p2c
        | _, _ => st.p2c
      projectItems { st with cur, p2c } (a :: seen) rest

/-- `Binder::bind_projection` for the non-`*` form, returning the scope the
next clause sees. -/
def bindProjection (st : St) (p : Proj) : Except String St := do
  -- 1. items are bound in the scope *before* the projection (binder.rs)
  let (st, bound) ← p.items.foldlM (init := (st, ([] : List (Name × PExpr × List Var))))
    fun (st, acc) (a, e) => do
      let (st, vs) ← resolveAll st e.vars
      pure (st, acc ++ [(a, e, vs)])
  -- 2. push_scope + project (binder.rs)
  let st ← projectItems (pushScope st) [] bound
  -- 3. grouping names (binder.rs)
  let groupBy := (p.items.filter (fun ie => !ie.2.agg)).flatMap (fun ie => ie.2.vars)
  -- 4. ORDER BY, with the parent scope visible (binder.rs)
  let (st, _) ← p.orderBy.foldlM (init := (st, ([] : List Var)))
    fun (st, acc) e => do let (st, vs) ← resolveAll st e.vars; pure (st, acc ++ vs)
  -- 5. aggregation: copies must be grouping names (binder.rs)
  if p.items.any (·.2.agg) && st.cfp.any (fun c => !groupBy.contains c.1) then
    throw "ORDER BY cannot reference variables not projected"
  -- 6. snapshot, then WHERE / SKIP / LIMIT (binder.rs)
  let before := st.cur.keys
  let (st, _) ← p.tail.foldlM (init := (st, ([] : List Var)))
    fun (st, acc) e => do let (st, vs) ← resolveAll st e.vars; pure (st, acc ++ vs)
  -- 7. remove the copies made only by WHERE / SKIP / LIMIT (binder.rs)
  let filterOnly := (st.cfp.map (·.1)).filter (fun k => !before.contains k)
  let cur := st.cur.retain (fun k => !filterOnly.contains k)
  -- 8. commit_scope (binder.rs)
  pure { st with cur, useParent := false, p2c := [], cfp := [] }

/-- The spec: the scope after `WITH`/`RETURN` is exactly its aliases. -/
def specProjectionScope (p : Proj) : List Name := p.items.map (·.1)

def keys (st : St) : List Name := st.cur.keys

def st0 : St :=
  { cur := [("n", ⟨"n", 0, 0, .node, none⟩)], parent := [], depth := 0,
    useParent := false, p2c := [], cfp := [] }

/-- `MATCH (n) WITH n.v AS k RETURN …`: the scope is `[k]`, as specified. -/

theorem with_plain :
    (bindProjection st0 { items := [("k", { vars := ["n"] })] }).map keys =
      .ok (specProjectionScope { items := [("k", { vars := ["n"] })] }) := by
  decide

/-- **Root cause 4** — `MATCH (n) WITH n.v AS k ORDER BY n.w`: the ORDER BY
copy of `n` is made before the snapshot, so it survives step 7 and the next
clause sees `n` (binder accepts `RETURN n`; `RETURN *` returns `k, n`).
openCypher and C: the scope is `[k]`. (Since #3011 `ORDER BY n.v` itself is
rewritten to `k` and no longer leaks: `orderBy_projected_no_leak`.) -/
theorem orderBy_leaks :
    (bindProjection st0 { items := [("k", { vars := ["n"] })], orderBy := [{ vars := ["n"] }] }).map keys
      = .ok ["k", "n"] ∧
    specProjectionScope { items := [("k", { vars := ["n"] })], orderBy := [{ vars := ["n"] }] } = ["k"] := by
  decide

/- The names of a raw expression, in DFS order (what `bind_expr` resolves). -/
mutual
def reVars : Agg.RE → List Name
  | .node d cs => (match d with | .var v => [v] | _ => []) ++ reVarsL cs
def reVarsL : List Agg.RE → List Name
  | [] => []
  | e :: es => reVars e ++ reVarsL es
end

/-- `bind_projection` with its ORDER BY step: each key is rewritten by
`replace_projected_subtrees` against the raw items, then bound. -/
def bindProjectionR (c : Agg.Ctx) (st : St) (items : List (Name × PExpr × Agg.RE))
    (orderBy : List Agg.RE) (tail : List PExpr := []) : Except String St :=
  bindProjection st
    { items := items.map (fun it => (it.1, it.2.1)),
      orderBy := orderBy.map (fun e =>
        { vars := reVars (Agg.replaceProjected c (items.map (fun it => (it.1, it.2.2, it.2.1.agg))) e) }),
      tail }

/-- #3011: `MATCH (n) WITH n.v AS k ORDER BY n.v` sorts by `k`, so no copy of
`n` is made and the scope is `[k]`, as specified. -/
theorem orderBy_projected_no_leak :
    (bindProjectionR Agg.c0 st0 [("k", { vars := ["n"] }, Agg.nv)] [Agg.nv]).map keys = .ok ["k"] := by
  decide

/-- …but a key that is not a projection, `ORDER BY n.w`, still leaks `n` (W2-binder-5). -/
theorem orderBy_unprojected_leaks :
    (bindProjectionR Agg.c0 st0 [("k", { vars := ["n"] }, Agg.nv)]
      [.node (.prop "w") [.node (.var "n") []]]).map keys = .ok ["k", "n"] := by
  decide

/-- The WHERE copy, by contrast, is removed from the table … -/
theorem where_copy_removed :
    (bindProjection st0 { items := [("k", { vars := ["n"] })], tail := [{ vars := ["n"] }] }).map keys
      = .ok ["k"] := by
  decide

/-- … but the Filter still reads it, at id 1 of scope 1, while the table
the next clause mints from has length 1 (**root cause 2**): the next
variable of the scope (`x` in `MATCH (x)-->(y)`) gets id 1 too. -/
theorem where_copy_id_reused :
    (do let st ← projectItems (pushScope st0) [] [("k", { vars := ["n"] }, [⟨"n", 0, 0, .node, none⟩])]
        let (_, v) ← resolve st [] "n"
        pure v.id : Except String Nat) = .ok 1 ∧
    ((bindProjection st0 { items := [("k", { vars := ["n"] })], tail := [{ vars := ["n"] }] }).map
      (fun st => (freshVar st.cur "x" .node st.depth).id)) = .ok 1 := by
  decide

/-- A bare-variable item records `parent_to_child_scope`, so an ORDER BY on
the *old* name resolves to the projected variable and makes no copy:
`WITH n AS m ORDER BY n.v` leaves the scope `[m]`. -/
theorem orderBy_old_name_of_projected :
    (bindProjection st0
      { items := [("m", { vars := ["n"], bare := true })], orderBy := [{ vars := ["n"] }] }).map keys
      = .ok ["m"] := by
  decide

/-- The aggregation check compares *variable names* inside the grouping
expressions, not the expressions: `RETURN n.v % 2 AS k, count(*) AS c
ORDER BY n.v` is accepted (C and openCypher reject — `n.v` is not a grouping
key, so the order is not determined by the group). -/
theorem agg_orderBy_by_name :
    (bindProjection st0
      { items := [("k", { vars := ["n"] }), ("c", (PExpr.mk [] true false))], orderBy := [{ vars := ["n"] }] }).map keys = .ok ["k", "c", "n"] := by
  decide

/-- and it does reject an ORDER BY name that is in no grouping key. -/
theorem agg_orderBy_rejects :
    (bindProjection st0
      { items := [("c", (PExpr.mk ["n"] true false))], orderBy := [{ vars := ["n"] }] }).map keys =
      .error "ORDER BY cannot reference variables not projected" := by
  decide

theorem Env.has_iff (e : Env) (k : Name) : e.has k = true ↔ k ∈ e.keys := by
  simp only [Env.has, Env.keys, List.any_eq_true, List.mem_filterMap, Key.beq_eq, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨k', v⟩, hm, rfl⟩; exact ⟨_, hm, rfl⟩
  · rintro ⟨⟨k', v⟩, hm, h⟩
    cases k' with
    | nm n => simp at h; subst h; exact ⟨_, hm, rfl⟩
    | un => simp at h

/-- Whatever ORDER BY references, the projection scope starts with every
alias, in order, each under a fresh id (binder.rs). -/
theorem projectItems_keys (st : St) (seen : List Name) (items : List (Name × PExpr × List Var))
    (st' : St) (hks : keys st = seen.reverse) (h : projectItems st seen items = .ok st') :
    keys st' = keys st ++ items.map (·.1) := by
  induction items generalizing st seen with
  | nil => simp [projectItems] at h; subst h; simp
  | cons it rest ih =>
    obtain ⟨a, e, vs⟩ := it
    simp only [projectItems] at h
    split at h
    · cases h
    · rename_i hseen
      have hn : st.cur.has a = false := by
        cases hh : st.cur.has a
        · rfl
        · exfalso; apply hseen
          have := (Env.has_iff _ _).1 hh
          simp only [keys] at hks
          rw [hks] at this
          simpa using this
      have hk' : ∀ p2c, keys { st with cur := (projectName st.cur st.depth a Ty.any).1, p2c := p2c }
          = keys st ++ [a] := by
        intro p2c
        simp only [keys, projectName, Env.insert_new_eq hn, Env.keys, List.filterMap_append]; rfl
      have := ih _ _ (by rw [hk', hks]; simp) h
      rw [this, hk']; simp

end FalkorBinder
