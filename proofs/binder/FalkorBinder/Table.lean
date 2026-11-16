/-
# Scope tables and id minting

`Binder.env_stack : Vec<HashMap<Arc<String>, Variable>>` (binder.rs:47). A
variable's id is minted as the *current size* of its scope's table
(`fresh_var`, binder.rs:2243-2257; the parent copy in `resolve_name`,
binder.rs:2224), and the planner mints its own ids the same way, as
`scope_vars[scope].len()`, pushing each one so the next is one larger
(`Planner::fresh_var`, planner/mod.rs:843-857).

A scope is `Scope { named: HashMap<Arc<String>, Variable>, unnamed: Vec<Variable> }`
(binder.rs:47-121, #3528). It is modelled as one association list whose keys are
`some name` for a named variable and `none` for an unnamed pattern entity, which no
name reaches. Named keys are distinct; `insert` replaces the value of a present
key and appends otherwise, so `length` is `Scope::len()` (named + unnamed).
-/
namespace FalkorBinder

abbrev Name := String

/-- `runtime::functions::Type`, restricted to what the binder inspects. -/
inductive Ty
  | any | node | rel | path
  deriving DecidableEq, Repr

/-- `parser::ast::Variable` (ast.rs:161). Equality in Rust is by `id` alone
(ast.rs:186-192) — the record slot. An unnamed variable (`name: None`) has name `""`
here; its `anon` number is display-only and not modelled. -/
structure Var where
  name  : Name
  id    : Nat
  scope : Nat
  ty    : Ty
  deriving DecidableEq, Repr

/-- `parser::ast::Alias` (ast.rs:84-94, #3528): a pattern entity is named by the user or
is the parser's `k`-th unnamed entity. No name spells an unnamed one. -/
inductive Alias
  | named (n : Name)
  | anon (k : Nat)
  deriving DecidableEq, Repr

/-- `Alias::name`'s `is_none()`, i.e. `Variable::is_unnamed` on the bound entity. -/
def Alias.isAnon : Alias → Bool
  | .named _ => false
  | .anon _ => true

/-- One scope table (`Scope`): `some n` keys the named variables, `none` the unnamed ones. -/
abbrev Env := List (Option Name × Var)

namespace Env

/-- `Scope::get`: named variables only. -/
def find? (e : Env) (k : Name) : Option Var := e.lookup (some k)

/-- `Scope::contains_key`. -/
def has (e : Env) (k : Name) : Bool := e.any (·.1 == some k)

/-- `Scope::insert`. -/
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == some k then (some k, v) else p) else e ++ [(some k, v)]

/-- `Scope::remove`. -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != some k)

/-- `Scope::retain`: keeps the named variables `keep` accepts; unnamed ones always stay. -/
def retain (e : Env) (keep : Name → Bool) : Env :=
  e.filter (fun p => match p.1 with | some n => keep n | none => true)

/-- `Scope::push_unnamed`. -/
def pushUnnamed (e : Env) (v : Var) : Env := e ++ [(none, v)]

/-- `Scope::keys`: the names in scope (unnamed variables have none). -/
def names (e : Env) : List Name := e.filterMap (·.1)

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

end Env

/-- `Binder::fresh_var` (binder.rs:2243): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name, id := e.length, scope, ty }

theorem Env.insert_new_eq {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    e.insert k v = e ++ [(some k, v)] := by
  simp [Env.insert, h]

theorem Env.lookup_map_replace (e : Env) (k m : Name) (v : Var) :
    (e.map (fun p => if (p.1 == some k) = true then (some k, v) else p)).lookup (some m) =
      if m = k ∧ e.any (·.1 == some k) = true then some v else e.lookup (some m) := by
  induction e with
  | nil => simp [List.lookup]
  | cons p ps ih =>
    obtain ⟨pk, pv⟩ := p
    by_cases hp : pk = some k
    · subst hp
      by_cases hm : m = k
      · subst hm; simp [List.lookup]
      · have hmk : (m == k) = false := by simp [hm]
        simp [hm] at ih
        simp [List.lookup, hmk, ih, hm]
    · have hp' : (pk == some k) = false := by simpa using hp
      by_cases hq : pk = some m
      · subst hq
        have hmk : m ≠ k := fun h => hp (by rw [h])
        simp [List.lookup, hmk]
      · have hq' : (some m == pk) = false := by
          simp only [beq_eq_false_iff_ne]; exact fun h => hq h.symm
        simp only [List.map_cons, hp', Bool.false_eq_true, ite_false, List.lookup, hq', ih,
          List.any_cons, Bool.false_or]

theorem Env.find?_insert (e : Env) (k m : Name) (v : Var) :
    (e.insert k v).find? m = if m = k then some v else e.find? m := by
  unfold Env.insert Env.find? Env.has
  by_cases hk : e.any (·.1 == some k) = true
  · simp only [hk, ite_true, Env.lookup_map_replace, and_true]
  · simp only [Bool.not_eq_true] at hk
    simp only [hk, Bool.false_eq_true, ite_false]
    by_cases hm : m = k
    · subst hm
      have : e.lookup (some m) = none := by
        rw [List.lookup_eq_none_iff]
        intro q hq; rw [List.any_eq_false] at hk
        have := hk q hq; simp at this; simp; exact fun h => this (Eq.symm h)
      simp [List.lookup_append, this, List.lookup]
    · have hmk : (m == k) = false := by simp [hm]
      simp [List.lookup_append, List.lookup, hm]
      cases e.lookup (some m) <;> simp [hmk]

/-- `Scope::push_unnamed`: no name reaches the new entry, and it takes a slot (`len()` grows). -/
theorem Env.find?_pushUnnamed (e : Env) (v : Var) (n : Name) : (e.pushUnnamed v).find? n = e.find? n := by
  simp [Env.pushUnnamed, Env.find?, List.lookup_append, List.lookup]

theorem Env.pushUnnamed_length (e : Env) (v : Var) : (e.pushUnnamed v).length = e.length + 1 := by
  simp [Env.pushUnnamed]

theorem Env.names_pushUnnamed (e : Env) (v : Var) : (e.pushUnnamed v).names = e.names := by
  simp [Env.pushUnnamed, Env.names]

/-- `Scope::retain` keeps every unnamed variable, whatever `keep` says. -/
theorem Env.retain_keeps_unnamed (e : Env) (keep : Name → Bool) (v : Var) (h : (none, v) ∈ e) :
    (none, v) ∈ e.retain keep := by
  simp [Env.retain, List.mem_filter, h]

/-- `Scope::retain` on the named variables is `HashMap::retain`. -/
theorem Env.retain_find? (e : Env) (keep : Name → Bool) (n : Name) :
    (e.retain keep).find? n = if keep n then e.find? n else none := by
  unfold Env.retain Env.find?
  induction e with
  | nil => simp [List.lookup]
  | cons p ps ih =>
    obtain ⟨k, v⟩ := p
    cases k with
    | none => simp [List.filter, List.lookup, ih]
    | some k =>
      by_cases hk : k = n
      · subst hk; by_cases hp : keep k <;> simp [List.filter, List.lookup, hp, ih]
      · have : (n == k) = false := by simp; exact fun h => hk h.symm
        by_cases hp : keep k <;> simp [List.filter, List.lookup, hp, ih, this]

theorem Env.insert_old_length {e : Env} {k : Name} {v : Var} (h : e.has k = true) :
    (e.insert k v).length = e.length := by
  simp [Env.insert, h]

theorem Env.insert_new_length {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    (e.insert k v).length = e.length + 1 := by
  simp [Env.insert_new_eq h]

/-- The minted id is not in use, provided the table covers its ids. -/
theorem fresh_not_used {e : Env} (h : e.Covers) (n : Name) (t : Ty) (s : Nat) :
    (freshVar e n t s).id ∉ e.ids := by
  intro hm
  have := h _ hm
  simp [freshVar] at this

/-- Minting under a *new* key (every `fresh_var` + `insert` pair in the
binder: `define_name_in_scope`, `project_name`, the `_quant_`, `_lc_`,
`_reduce_`, `__agg_placeholder_` and `_hidden_` reservations, and
the parent copy in `resolve_name`) keeps the table covering. -/
theorem mint_new_covers {e : Env} (h : e.Covers) {k : Name} (hk : e.has k = false)
    (n : Name) (t : Ty) (s : Nat) :
    (e.insert k (freshVar e n t s)).Covers := by
  intro i hi
  rw [Env.insert_new_eq hk] at hi ⊢
  simp [Env.ids] at hi
  simp
  rcases hi with ⟨a, b, hab⟩ | hi
  · have := h i (by simp [Env.ids]; exact ⟨a, b, hab⟩)
    omega
  · simp [freshVar] at hi; omega

/-- The same key re-inserted with an id already below `len()` (the
`YIELD field AS alias` re-insert, binder.rs:633) keeps the table covering. -/
theorem reinsert_covers {e : Env} (h : e.Covers) {k : Name} (hk : e.has k = true)
    (v : Var) (hv : v.id < e.length) : (e.insert k v).Covers := by
  intro i hi
  rw [Env.insert_old_length hk]
  simp only [Env.insert, hk, ite_true, Env.ids, List.map_map, List.mem_map,
    Function.comp] at hi
  obtain ⟨p, hp, rfl⟩ := hi
  by_cases hpk : p.1 == some k
  · simp [hpk]; exact hv
  · simp [hpk]; exact h _ (List.mem_map_of_mem hp)

/-! ## The planner's side: `mint_fresh_iff`

The planner receives `scope_vars[s]` (the table sorted by id,
`sorted_scope_vars`, binder.rs:2686) and mints `len`, `len + 1`, … in that
scope. None of those collides with an id `H` the bound IR uses in `s` iff
every such id is below `len`. So *the* soundness condition of length-based
minting is `∀ i ∈ H, i < len` — and `Env.Covers` alone is not enough: an
id the IR uses may have been *removed* from the table. -/

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

/-- Removing an entry can only shrink the table. -/
theorem erase_length_le (e : Env) (k : Name) : (e.erase k).length ≤ e.length := by
  simp [Env.erase]; exact List.length_filter_le _ _

/-- A covering table stays covering after `erase`/`retain` **only** for the
ids it still holds; an id the IR still uses may now be `≥ len()`.
Concretely (binder.rs:1260-1268): after `MATCH (n) WITH n.v AS k WHERE n.v = 0`
the WITH scope is `{k ↦ 0, n ↦ 1}` (`n` copied for the WHERE, id 1, used by
the Filter), then `n` is removed: the table is `{k ↦ 0}` and the next
variable of the scope, `x` in `MATCH (x)-->(y)`, is minted with id 1. -/
def withScope : Env :=
  [("k", ⟨"k", 0, 1, .any⟩), ("n", ⟨"n", 1, 1, .node⟩)]

def withScopeAfter : Env := withScope.erase "n"

theorem withScope_covers : withScope.Covers := by
  intro i hi; simp [withScope, Env.ids] at hi; rcases hi with rfl | rfl <;> decide

theorem erase_breaks_covers :
    withScope.Covers ∧ 1 ∈ withScope.ids ∧
    (freshVar withScopeAfter "x" .node 1).id = 1 := by
  refine ⟨withScope_covers, by decide, by decide⟩

end FalkorBinder
