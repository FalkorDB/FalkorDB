/-
# Scope tables and id minting

`Binder.env_stack : Vec<Scope>` (binder.rs:149), a `Scope` being the named variables by name and,
since #3303, the unnamed pattern entities in a list of their own (`Scope`, binder.rs:46-120). A
variable's id is minted as the *current size* of its scope's table
(`fresh_var`, binder.rs:2243-2257; the parent copy in `resolve_name`,
binder.rs:2224), and the planner mints its own ids the same way, as
`scope_vars[scope].len()`, pushing each one so the next is one larger
(`Planner::fresh_var`, planner/mod.rs:843-857).

A `Scope` is modelled as one association list whose keys are a name (`Key.nm`, the `named`
map, distinct keys) or `Key.un` (an entry of `unnamed`, which no name reaches). `insert`
replaces the value of a present name and appends otherwise, so `length` is `Scope::len()`
(`named.len() + unnamed.len()`, binder.rs:110-112).
-/
namespace FalkorBinder

abbrev Name := String

/-- `runtime::functions::Type`, restricted to what the binder inspects. -/
inductive Ty
  | any | node | rel | path
  deriving DecidableEq, Repr

/-- `parser::ast::Variable` (ast.rs:165-173). Equality in Rust is by `VarKey`
(`id`, `scope`) since #3933 (ast.rs:188-197); see `Var.key`. `name` is `none` for an unnamed pattern entity, whose
parser number is `anon` (#3303). -/
structure Var where
  name  : Option Name
  id    : Nat
  scope : Nat
  ty    : Ty
  anon  : Option Nat := none
  deriving DecidableEq, Repr

/-- `Variable::key` (ast.rs:218-223): the `VarKey` `(id, scope)`. -/
def Var.key (v : Var) : Nat × Nat := (v.id, v.scope)

/-- What names a pattern entity before binding: `parser::ast::Alias` (ast.rs:86-92, #3303). -/
inductive Alias
  | named (n : Name)
  | anon (k : Nat)
  deriving DecidableEq, Repr

/-- A scope entry's key: a name of `Scope.named`, or an entry of `Scope.unnamed`. -/
inductive Key
  | nm (n : Name)
  | un
  deriving DecidableEq, Repr

instance : Coe Name Key := ⟨Key.nm⟩

@[simp] theorem Key.beq_eq (a b : Key) : (a == b) = decide (a = b) := rfl

/-- One scope table (`Scope`). -/
abbrev Env := List (Key × Var)

namespace Env

/-- `Scope::get` (binder.rs:53-58): the named map only. -/
def find? (e : Env) (k : Name) : Option Var := e.lookup (.nm k)

/-- `Scope::contains_key` (binder.rs:60-65). -/
def has (e : Env) (k : Name) : Bool := e.any (·.1 == .nm k)

/-- `Scope::insert` (binder.rs:67-73), `HashMap::insert` on `named`. -/
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == .nm k then (.nm k, v) else p) else e ++ [(.nm k, v)]

/-- `Scope::remove` (binder.rs:75-80). -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != .nm k)

/-- `Scope::retain` (binder.rs:115-121): filters the named entries; unnamed ones always stay. -/
def retain (e : Env) (keep : Name → Bool) : Env := e.filter (fun p => match p.1 with
  | .nm n => keep n
  | .un => true)

/-- `Scope::push_unnamed` (binder.rs:83-89). -/
def pushUnnamed (e : Env) (v : Var) : Env := e ++ [(.un, v)]

/-- `Scope::keys` (binder.rs:96-100): the names; unnamed entries have none. -/
def keys (e : Env) : List Name := e.filterMap (fun p => match p.1 with
  | .nm n => some n
  | .un => none)

/-- `Scope::iter` (binder.rs:91-95): the named entries. -/
def named (e : Env) : List (Name × Var) := e.filterMap (fun p => match p.1 with
  | .nm n => some (n, p.2)
  | .un => none)

/-- `Scope::values` (binder.rs:101-105) / `into_values` (binder.rs:106-109): every variable. -/
def values (e : Env) : List Var := e.map (·.2)

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- The `VarKey`s `(id, scope)` of the table (`values().map(Variable::key)`, #3933). -/
def varKeys (e : Env) : List (Nat × Nat) := e.map (·.2.key)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

end Env

/-- `Binder::fresh_var` (binder.rs:2384-2399): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name := some name, id := e.length, scope, ty }

/-- The unnamed variable `define_alias_in_scope` mints for `Alias::Anon k` (binder.rs:2276-2280):
`fresh_var(None, …)` with `anon = Some(k)`. -/
def freshAnon (e : Env) (k : Nat) (ty : Ty) (scope : Nat) : Var :=
  { name := none, id := e.length, scope, ty, anon := some k }

/-- `Scope::len` counts the unnamed entries (binder.rs:110-113). -/
theorem Env.pushUnnamed_length (e : Env) (v : Var) : (e.pushUnnamed v).length = e.length + 1 := by
  simp [Env.pushUnnamed]

/-- An unnamed entry is reached by no name. -/
theorem Env.pushUnnamed_find (e : Env) (v : Var) (n : Name) :
    (e.pushUnnamed v).find? n = e.find? n := by
  have : (Key.nm n == Key.un) = false := by simp
  simp [Env.pushUnnamed, Env.find?, List.lookup_append, List.lookup, this]

/-- `Scope::retain` keeps every unnamed entry. -/
theorem Env.retain_keeps_unnamed (e : Env) (keep : Name → Bool) (v : Var) (h : (.un, v) ∈ e) :
    (.un, v) ∈ e.retain keep := by
  simp [Env.retain, List.mem_filter, h]

/-- `Scope::values` covers every entry, named or not, so `len()` is its length. -/
theorem Env.values_length (e : Env) : e.values.length = e.length := by
  simp [Env.values]

/-- An unnamed entry adds no key, no named entry, but one value. -/
theorem Env.pushUnnamed_views (e : Env) (v : Var) :
    (e.pushUnnamed v).keys = e.keys ∧ (e.pushUnnamed v).named = e.named ∧
      (e.pushUnnamed v).values = e.values ++ [v] := by
  simp [Env.pushUnnamed, Env.keys, Env.named, Env.values, List.filterMap_append]

/-- `Scope::keys` are the names of `Scope::iter`. -/
theorem Env.keys_named (e : Env) : e.keys = e.named.map (·.1) := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    cases k <;> simp [Env.keys, Env.named] at ih ⊢ <;> exact ih

/-- `Scope::get` / `contains_key` / `remove` see the named entries only. -/
theorem Env.has_iff_find (e : Env) (n : Name) : e.has n = true ↔ (e.find? n).isSome := by
  induction e with
  | nil => simp [Env.has, Env.find?]
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : k = .nm n
    · subst hk; simp [Env.has, Env.find?, List.lookup]
    · have : (Key.nm n == k) = false := by simp; exact fun h => hk h.symm
      simp only [Env.has, Env.find?, List.any_cons, List.lookup, this] at ih ⊢
      simp [hk]

theorem Env.erase_find (e : Env) (n : Name) : (e.erase n).find? n = none := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : k = .nm n
    · subst hk; simpa [Env.erase] using ih
    · have : (Key.nm n == k) = false := by simp; exact fun h => hk h.symm
      simp only [Env.erase, Env.find?, List.filter_cons] at ih ⊢
      simp [hk, List.lookup, this, ih]

/-- Pushing an unnamed variable minted at `len()` keeps the table covering. -/
theorem pushUnnamed_covers {e : Env} (h : e.Covers) (k : Nat) (t : Ty) (s : Nat) :
    (e.pushUnnamed (freshAnon e k t s)).Covers := by
  intro i hi
  simp only [Env.pushUnnamed, Env.ids, List.map_append, List.mem_append, List.map_cons,
    List.map_nil, List.mem_singleton] at hi
  simp only [Env.pushUnnamed, List.length_append, List.length_singleton]
  rcases hi with hi | hi
  · have := h i hi; omega
  · simp [freshAnon] at hi; omega

theorem Env.insert_new_eq {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    e.insert k v = e ++ [(.nm k, v)] := by
  simp [Env.insert, h]

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

/-- The same for any variable whose id is `len()` (the parent copy in `resolve_name`, which
keeps the parent variable's name, type and `anon`). -/
theorem mint_new_covers' {e : Env} (h : e.Covers) {k : Name} (hk : e.has k = false)
    (v : Var) (hv : v.id = e.length) : (e.insert k v).Covers := by
  intro i hi
  rw [Env.insert_new_eq hk] at hi ⊢
  simp [Env.ids] at hi
  simp
  rcases hi with ⟨a, b, hab⟩ | hi
  · have := h i (by simp [Env.ids]; exact ⟨a, b, hab⟩)
    omega
  · omega

/-- The same key re-inserted with an id already below `len()` (the
`YIELD field AS alias` re-insert, binder.rs:633) keeps the table covering. -/
theorem reinsert_covers {e : Env} (h : e.Covers) {k : Name} (hk : e.has k = true)
    (v : Var) (hv : v.id < e.length) : (e.insert k v).Covers := by
  intro i hi
  rw [Env.insert_old_length hk]
  simp only [Env.insert, hk, ite_true, Env.ids, List.map_map, List.mem_map,
    Function.comp] at hi
  obtain ⟨p, hp, rfl⟩ := hi
  by_cases hpk : p.1 == k
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
  [("k", ⟨"k", 0, 1, .any, none⟩), ("n", ⟨"n", 1, 1, .node, none⟩)]

def withScopeAfter : Env := withScope.erase "n"

theorem withScope_covers : withScope.Covers := by
  intro i hi; simp [withScope, Env.ids] at hi; rcases hi with rfl | rfl <;> decide

theorem erase_breaks_covers :
    withScope.Covers ∧ 1 ∈ withScope.ids ∧
    (freshVar withScopeAfter "x" .node 1).id = 1 := by
  refine ⟨withScope_covers, by decide, by decide⟩

end FalkorBinder
