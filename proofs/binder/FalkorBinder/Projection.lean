import FalkorBinder.Resolve
/-
# WITH / RETURN: `bind_projection` (binder.rs:1059-1301)

An expression is abstracted to the names it mentions (in DFS order), whether
it is an aggregation, and whether it is a bare variable (the case that
records a `parent_to_child_scope` entry, binder.rs:1156-1158).
-/
namespace FalkorBinder

structure PExpr where
  vars  : List Name
  agg   : Bool := false
  /-- the expression is exactly `ExprIR::Variable` (binder.rs:1156) -/
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
(`bind_expr` → `resolve_name` per `ExprIR::Variable`, binder.rs:1753-1756). -/
def resolveAll (st : St) : List Name → Except String (St × List Var)
  | [] => .ok (st, [])
  | n :: ns => do
    let (st, v) ← resolve st [] n
    let (st, vs) ← resolveAll st ns
    pure (st, v :: vs)

/-- `push_scope` (binder.rs:251) as seen by name resolution. -/
def pushScope (st : St) : St :=
  { cur := [], parent := st.cur, depth := st.depth + 1, useParent := true,
    p2c := [], cfp := [] }

structure Proj where
  items   : List (Name × PExpr)
  orderBy : List PExpr := []
  /-- WHERE, SKIP and LIMIT: bound after the snapshot at binder.rs:1245 -/
  tail    : List PExpr := []

/-- Project the items into the freshly pushed scope (binder.rs:1147-1171). -/
def projectItems (st : St) (seen : List Name) :
    List (Name × PExpr × List Var) → Except String St
  | [] => .ok st
  | (a, e, vs) :: rest =>
    if seen.contains a then
      .error "Error: Multiple result columns with the same name are not supported."
    else
      let (cur, bv) := projectName st.cur st.depth a .any
      let p2c := match e.isVar, vs with
        | some _, [v] => st.p2c.insert v.name bv
        | _, _ => st.p2c
      projectItems { st with cur, p2c } (a :: seen) rest

/-- `Binder::bind_projection` for the non-`*` form, returning the scope the
next clause sees. -/
def bindProjection (st : St) (p : Proj) : Except String St := do
  -- 1. items are bound in the scope *before* the projection (binder.rs:1071-1078)
  let (st, bound) ← p.items.foldlM (init := (st, ([] : List (Name × PExpr × List Var))))
    fun (st, acc) (a, e) => do
      let (st, vs) ← resolveAll st e.vars
      pure (st, acc ++ [(a, e, vs)])
  -- 2. push_scope + project (binder.rs:1147-1171)
  let st ← projectItems (pushScope st) [] bound
  -- 3. grouping names (binder.rs:1177-1192)
  let groupBy := (p.items.filter (fun ie => !ie.2.agg)).flatMap (fun ie => ie.2.vars)
  -- 4. ORDER BY, with the parent scope visible (binder.rs:1212-1215)
  let (st, _) ← p.orderBy.foldlM (init := (st, ([] : List Var)))
    fun (st, acc) e => do let (st, vs) ← resolveAll st e.vars; pure (st, acc ++ vs)
  -- 5. aggregation: copies must be grouping names (binder.rs:1230-1241)
  if p.items.any (·.2.agg) && st.cfp.any (fun c => !groupBy.contains c.1) then
    throw "ORDER BY cannot reference variables not projected"
  -- 6. snapshot, then WHERE / SKIP / LIMIT (binder.rs:1245-1255)
  let before := st.cur.map (·.1)
  let (st, _) ← p.tail.foldlM (init := (st, ([] : List Var)))
    fun (st, acc) e => do let (st, vs) ← resolveAll st e.vars; pure (st, acc ++ vs)
  -- 7. remove the copies made only by WHERE / SKIP / LIMIT (binder.rs:1260-1268)
  let filterOnly := (st.cfp.map (·.1)).filter (fun k => !before.contains k)
  let cur := st.cur.retain (fun k => !filterOnly.contains k)
  -- 8. commit_scope (binder.rs:1275-1276)
  pure { st with cur, useParent := false, p2c := [], cfp := [] }

/-- The spec: the scope after `WITH`/`RETURN` is exactly its aliases. -/
def specProjectionScope (p : Proj) : List Name := p.items.map (·.1)

def keys (st : St) : List Name := st.cur.map (·.1)

def st0 : St :=
  { cur := [("n", ⟨"n", 0, 0, .node⟩)], parent := [], depth := 0,
    useParent := false, p2c := [], cfp := [] }

/-- `MATCH (n) WITH n.v AS k RETURN …`: the scope is `[k]`, as specified. -/

theorem with_plain :
    (bindProjection st0 { items := [("k", { vars := ["n"] })] }).map keys =
      .ok (specProjectionScope { items := [("k", { vars := ["n"] })] }) := by
  decide

/-- **Root cause 4** — `MATCH (n) WITH n.v AS k ORDER BY n.v`: the ORDER BY
copy of `n` is made before the snapshot, so it survives step 7 and the next
clause sees `n` (binder accepts `RETURN n`; `RETURN *` returns `k, n`).
openCypher and C: the scope is `[k]`. -/
theorem orderBy_leaks :
    (bindProjection st0 { items := [("k", { vars := ["n"] })], orderBy := [{ vars := ["n"] }] }).map keys
      = .ok ["k", "n"] ∧
    specProjectionScope { items := [("k", { vars := ["n"] })], orderBy := [{ vars := ["n"] }] } = ["k"] := by
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
    (do let st ← projectItems (pushScope st0) [] [("k", { vars := ["n"] }, [⟨"n", 0, 0, .node⟩])]
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

theorem Env.has_iff (e : Env) (k : Name) : e.has k = true ↔ k ∈ e.map (·.1) := by
  simp [Env.has, List.any_eq_true]

/-- Whatever ORDER BY references, the projection scope starts with every
alias, in order, each under a fresh id (binder.rs:1147-1171). -/
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
        simp only [keys, projectName, Env.insert_new_eq hn, List.map_append]; rfl
      have := ih _ _ (by rw [hk', hks]; simp) h
      rw [this, hk']; simp

end FalkorBinder
