import FalkorBinder.Projection
import FalkorBinder.Scopes
/-
# The scope stack and the label post-pass

| here | there |
| --- | --- |
| `BSt` | `Binder` fields `env_stack`, `use_parent_scope`, `parent_to_child_scope`, `copy_from_parent`, `retired_scope_vars` (binder.rs:44-71) |
| `BSt.default` | `Default for Binder` binder.rs:73-85 |
| `BSt.current`, `BSt.setCurrent` | `current_env` / `current_env_mut` binder.rs:239-249 |
| `BSt.push`, `BSt.commit` | `push_scope` / `commit_scope` binder.rs:251-259 |
| `sortedScopeVars`, `BSt.scopeVars` | `sorted_scope_vars` binder.rs:2686-2690, `scope_vars` binder.rs:114-116 |
| `toSt` | the two innermost scopes as name resolution sees them (Resolve.lean `St`) |
| `mergedLabels`, `updNode`, `updGraph` | `update_graph_labels` binder.rs:160-237 |
-/
namespace FalkorBinder

structure BSt where
  stack : List Env
  useParent : Bool
  p2c : Env
  cfp : List (Name × (Var × Var))
  retired : List (List Var)

def BSt.default : BSt := ⟨[[]], false, [], [], []⟩

/-- `env_stack.last()` (the stack is never empty). -/
def BSt.current (b : BSt) : Env := b.stack.getLastD []

def BSt.setCurrent (b : BSt) (e : Env) : BSt :=
  { b with stack := b.stack.dropLast ++ [e] }

def BSt.push (b : BSt) : BSt := { b with stack := b.stack ++ [[]], useParent := true }

def BSt.commit (b : BSt) : BSt := { b with useParent := false, p2c := [] }

def sortedScopeVars (e : Env) : List Var := (e.map (·.2)).mergeSort fun a b => decide (a.id ≤ b.id)

def BSt.scopeVars (b : BSt) : List (List Var) := b.stack.map sortedScopeVars

/-- Resolution's view: current table, its parent, depth = scope id. -/
def BSt.toSt (b : BSt) : St :=
  { cur := b.current, parent := b.stack.dropLast.getLastD [], depth := b.stack.length - 1,
    useParent := b.useParent, p2c := b.p2c, cfp := b.cfp }

theorem default_spec : BSt.default.stack.length = 1 ∧ BSt.default.current = [] ∧
    BSt.default.scopeVars = [[]] ∧ BSt.default.useParent = false :=
  ⟨rfl, rfl, by simp [BSt.default, BSt.scopeVars, sortedScopeVars], rfl⟩

theorem current_setCurrent (b : BSt) (e : Env) : (b.setCurrent e).current = e := by
  simp [BSt.setCurrent, BSt.current]

/-- `current_env_mut` touches only the innermost table. -/
theorem setCurrent_outer (b : BSt) (e : Env) (h : b.stack ≠ []) :
    (b.setCurrent e).stack.dropLast = b.stack.dropLast ∧ (b.setCurrent e).stack.length = b.stack.length := by
  simp only [BSt.setCurrent, List.dropLast_concat, List.length_append, List.length_dropLast,
    List.length_singleton, true_and]
  have : b.stack.length ≠ 0 := by simpa using h
  omega

theorem push_current (b : BSt) : b.push.current = [] := by simp [BSt.push, BSt.current]

/-- `push_scope` opens an empty table one level deeper, with the old current
table as its parent, and turns parent lookup on. It does **not** clear
`parent_to_child_scope` / `copy_from_parent` (the projection model's
`pushScope` starts them empty; `bind_projection` clears/fills them itself). -/
theorem push_toSt (b : BSt) (h : b.stack ≠ []) :
    b.push.toSt = { pushScope b.toSt with p2c := b.p2c, cfp := b.cfp } := by
  simp only [BSt.toSt, BSt.push, pushScope, BSt.current, List.dropLast_concat, List.length_append,
    List.length_singleton, Nat.add_sub_cancel]
  have : b.stack.length ≠ 0 := by simpa using h
  congr 1
  · simp
  · omega

theorem commit_spec (b : BSt) : b.commit.useParent = false ∧ b.commit.p2c = [] ∧ b.commit.stack = b.stack :=
  ⟨rfl, rfl, rfl⟩

theorem keLe_trans (a b c : Var) (h1 : decide (a.id ≤ b.id) = true) (h2 : decide (b.id ≤ c.id) = true) :
    decide (a.id ≤ c.id) = true := by simp at *; omega

theorem keLe_total (a b : Var) : (decide (a.id ≤ b.id) || decide (b.id ≤ a.id)) = true := by
  simp; omega

/-- `sorted_scope_vars`: the table's variables, sorted by id, one per entry —
so the planner's `scope_vars[s].len()` is the table's `len()`. -/
theorem sortedScopeVars_spec (e : Env) :
    (sortedScopeVars e).Perm (e.map (·.2)) ∧ (sortedScopeVars e).length = e.length ∧
    (sortedScopeVars e).Pairwise (fun a b => a.id ≤ b.id) := by
  refine ⟨List.mergeSort_perm _ _, by simp [sortedScopeVars], ?_⟩
  have := List.pairwise_mergeSort (le := fun a b : Var => decide (a.id ≤ b.id))
    (fun a b c h1 h2 => keLe_trans a b c h1 h2) (fun a b => keLe_total a b) (e.map (·.2))
  exact this.imp (fun h => by simpa using h)

/-- `scope_vars`: one sorted table per live scope, lengths preserved. -/
theorem scopeVars_lengths (b : BSt) (s : Nat) :
    lenAt b.scopeVars s = (b.stack.getD s []).length := by
  simp only [lenAt, BSt.scopeVars]
  by_cases hs : s < b.stack.length
  · simp [List.getD, hs, (sortedScopeVars_spec _).2.1]
  · have : b.stack.length ≤ s := by omega
    simp [List.getD, List.getElem?_eq_none this]

/-! ## `update_graph_labels` -/

abbrev Key := Nat × Nat

def mergedLabels (union : Bool) (own acc : List Name) : List Name :=
  if union then own ++ acc.filter (fun l => !own.contains l) else acc

/-- One node (alias key, labels): replaced by the accumulated set, or merged
into its own labels for an OPTIONAL MATCH; untouched without an entry. -/
def updNode (nl : List (Key × List Name)) (union : Bool) (n : Key × List Name) : Key × List Name :=
  match nl.lookup n.1 with
  | some acc => (n.1, mergedLabels union n.2 acc)
  | none => n

/-- Nodes, then each relationship's two endpoints (binder.rs:183-236). -/
def updGraph (nl : List (Key × List Name)) (union : Bool) (nodes : List (Key × List Name))
    (rels : List ((Key × List Name) × (Key × List Name))) :
    List (Key × List Name) × List ((Key × List Name) × (Key × List Name)) :=
  (nodes.map (updNode nl union), rels.map fun r => (updNode nl union r.1, updNode nl union r.2))

theorem updNode_replace (nl : List (Key × List Name)) (n : Key × List Name) (acc : List Name)
    (h : nl.lookup n.1 = some acc) : updNode nl false n = (n.1, acc) := by
  simp [updNode, h, mergedLabels]

/-- OPTIONAL MATCH: the clause's own labels are kept, in order, and every
accumulated label is added once. -/
theorem updNode_union (nl : List (Key × List Name)) (n : Key × List Name) (acc : List Name)
    (h : nl.lookup n.1 = some acc) :
    (updNode nl true n).2.take n.2.length = n.2 ∧ ∀ l ∈ acc, l ∈ (updNode nl true n).2 := by
  simp only [updNode, h, mergedLabels, ↓reduceIte, List.take_left']
  refine ⟨by simp, fun l hl => ?_⟩
  by_cases ho : l ∈ n.2
  · exact List.mem_append_left _ ho
  · exact List.mem_append_right _ (List.mem_filter.2 ⟨hl, by simpa using ho⟩)

theorem updNode_none (nl : List (Key × List Name)) (u : Bool) (n : Key × List Name)
    (h : nl.lookup n.1 = none) : updNode nl u n = n := by simp [updNode, h]

/-- Running the post-pass twice changes nothing more (idempotent). -/
theorem updNode_idem (nl : List (Key × List Name)) (u : Bool) (n : Key × List Name) :
    updNode nl u (updNode nl u n) = updNode nl u n := by
  cases h : nl.lookup n.1 with
  | none => simp [updNode, h]
  | some acc =>
    simp only [updNode, h]
    cases u
    · simp [mergedLabels]
    · simp only [mergedLabels, ↓reduceIte]
      have hall : ∀ l ∈ acc, l ∈ n.2 ++ acc.filter (fun l => !n.2.contains l) := by
        intro l hl
        by_cases ho : l ∈ n.2
        · exact List.mem_append_left _ ho
        · exact List.mem_append_right _ (List.mem_filter.2 ⟨hl, by simpa using ho⟩)
      have : acc.filter (fun l => !(n.2 ++ acc.filter (fun l => !n.2.contains l)).contains l) = [] := by
        rw [List.filter_eq_nil_iff]
        intro l hl
        rw [List.contains_iff_mem.2 (hall l hl)]; simp
      rw [this, List.append_nil]

end FalkorBinder
