/-
# Scope tables and id minting

`Binder.env_stack : Vec<Scope>` (binder.rs), a `Scope` being the named variables by name and,
the unnamed pattern entities in a list of their own (`Scope`, binder.rs). A
variable's id is minted as the *current size* of its scope's table
(`fresh_var`, binder.rs; the parent copy in `resolve_name`,
binder.rs), and the planner mints its own ids the same way, as
`scope_vars[scope].len()`, pushing each one so the next is one larger
(`Planner::fresh_var`, planner/mod.rs).

A `Scope` is modelled as one association list whose keys are a name (`Key.nm`, the `named`
map, distinct keys) or `Key.un` (an entry of `unnamed`, which no name reaches). `insert`
replaces the value of a present name and appends otherwise, so `length` is `Scope::len()`
(`named.len() + unnamed.len()`, binder.rs).
-/
namespace FalkorBinder

abbrev Name := String

/-- `runtime::functions::Type`, restricted to what the binder inspects. -/
inductive Ty
  | any | node | rel | path
  deriving DecidableEq, Repr

/-- `parser::ast::Variable` (ast.rs). Equality in Rust is by `id` alone
(ast.rs) — the record slot. `name` is `none` for an unnamed pattern entity, whose
parser number is `anon` (#3303). -/
structure Var where
  name  : Option Name
  id    : Nat
  scope : Nat
  ty    : Ty
  anon  : Option Nat := none
  deriving DecidableEq, Repr

/-- What names a pattern entity before binding: `parser::ast::Alias` (ast.rs, #3303). -/
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

/-- `Scope::get` (binder.rs): the named map only. -/
def find? (e : Env) (k : Name) : Option Var := e.lookup (.nm k)

/-- `Scope::contains_key` (binder.rs). -/
def has (e : Env) (k : Name) : Bool := e.any (·.1 == .nm k)

/-- `Scope::insert` (binder.rs), `HashMap::insert` on `named`. -/
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == .nm k then (.nm k, v) else p) else e ++ [(.nm k, v)]

/-- `Scope::remove` (binder.rs). -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != .nm k)

/-- `Scope::retain` (binder.rs): filters the named entries; unnamed ones always stay. -/
def retain (e : Env) (keep : Name → Bool) : Env := e.filter (fun p => match p.1 with
  | .nm n => keep n
  | .un => true)

/-- `Scope::push_unnamed` (binder.rs). -/
def pushUnnamed (e : Env) (v : Var) : Env := e ++ [(.un, v)]

/-- `Scope::keys` (binder.rs): the names; unnamed entries have none. -/
def keys (e : Env) : List Name := e.filterMap (fun p => match p.1 with
  | .nm n => some n
  | .un => none)

/-- `Scope::iter` (binder.rs): the named entries. -/
def named (e : Env) : List (Name × Var) := e.filterMap (fun p => match p.1 with
  | .nm n => some (n, p.2)
  | .un => none)

/-- `Scope::unnamed`: the nameless variables, in insertion order. -/
def unnamed (e : Env) : List Var := e.filterMap (fun p => match p.1 with
  | .un => some p.2
  | .nm _ => none)

/-- `Scope::lookup_anon` (binder.rs): the unnamed variable the parser numbered `k`
(`find` over `unnamed`, so the first). -/
def lookupAnon (e : Env) (k : Nat) : Option Var := e.unnamed.find? (fun v => v.anon == some k)

/-- `Scope::values` (binder.rs) / `into_values` (binder.rs): every variable. -/
def values (e : Env) : List Var := e.map (·.2)

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

end Env

/-- `Binder::fresh_var` (binder.rs): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name := some name, id := e.length, scope, ty }

/-- The unnamed variable `define_alias_in_scope` mints for `Alias::Anon k` (binder.rs):
`fresh_var(None, …)` with `anon = Some(k)`. -/
def freshAnon (e : Env) (k : Nat) (ty : Ty) (scope : Nat) : Var :=
  { name := none, id := e.length, scope, ty, anon := some k }

/-- `Scope::len` counts the unnamed entries (binder.rs). -/
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

/-- `lookup_anon` finds only a variable whose parser number is `k`. -/
theorem Env.lookupAnon_spec (e : Env) (k : Nat) (v : Var) (h : e.lookupAnon k = some v) :
    v.anon = some k ∧ v ∈ e.unnamed := by
  unfold Env.lookupAnon at h
  have := List.find?_some h
  exact ⟨by simpa using this, List.mem_of_find?_eq_some h⟩

/-- A table without the number has no hit, and a new unnamed variable becomes the hit. -/
theorem Env.lookupAnon_pushUnnamed_self (e : Env) (v : Var) (k : Nat) (hv : v.anon = some k)
    (h : e.lookupAnon k = none) : (e.pushUnnamed v).lookupAnon k = some v := by
  have hu : (e.pushUnnamed v).unnamed = e.unnamed ++ [v] := by
    simp [Env.pushUnnamed, Env.unnamed, List.filterMap_append]
  have hn : e.unnamed.find? (fun v => v.anon == some k) = none := h
  simp [Env.lookupAnon, hu, List.find?_append, hn, hv]

/-- Pushing an unnamed variable for another number changes no earlier hit. -/
theorem Env.lookupAnon_pushUnnamed_other (e : Env) (v : Var) (k j : Nat) (hv : v.anon = some j)
    (hne : j ≠ k) : (e.pushUnnamed v).lookupAnon k = e.lookupAnon k := by
  have hu : (e.pushUnnamed v).unnamed = e.unnamed ++ [v] := by
    simp [Env.pushUnnamed, Env.unnamed, List.filterMap_append]
  have : (v.anon == some k) = false := by simp [hv, hne]
  simp [Env.lookupAnon, hu, List.find?_append, this]

/-- No name reaches an unnamed variable and no name-keyed operation changes the unnamed list:
`insert`, `erase` and `retain` leave `lookup_anon` alone for the entries `retain` keeps. -/
theorem Env.lookupAnon_insert (e : Env) (n : Name) (v : Var) (k : Nat) :
    (e.insert n v).lookupAnon k = e.lookupAnon k := by
  have : (e.insert n v).unnamed = e.unnamed := by
    unfold Env.insert
    split
    · simp only [Env.unnamed, List.filterMap_map]
      congr 1
      funext p
      by_cases h : p.1 == .nm n <;> simp_all [Function.comp]
      <;> (rcases p with ⟨_ | _, _⟩ <;> simp_all)
    · simp [Env.unnamed, List.filterMap_append]
  simp [Env.lookupAnon, this]

theorem Env.lookupAnon_retain (e : Env) (keep : Name → Bool) (k : Nat) :
    (e.retain keep).lookupAnon k = e.lookupAnon k := by
  have : (e.retain keep).unnamed = e.unnamed := by
    simp only [Env.retain, Env.unnamed]
    induction e with
    | nil => rfl
    | cons p rest ih =>
      rcases p with ⟨_ | _, v⟩
      · by_cases h : keep (‹Name›) = true <;> simp_all
      · simp_all
  simp [Env.lookupAnon, this]

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

/-- `Scope::insert` then `Scope::get` of the same name returns the inserted variable. -/
theorem Env.find?_insert_self (e : Env) (n : Name) (v : Var) : (e.insert n v).find? n = some v := by
  have key : ∀ e : Env, Env.has e n = true →
      (e.map (fun p => if p.1 == Key.nm n then (Key.nm n, v) else p)).lookup (Key.nm n) = some v := by
    intro e
    induction e with
    | nil => simp [Env.has]
    | cons p rest ih =>
      obtain ⟨k, w⟩ := p
      intro h
      by_cases hk : k = .nm n
      · subst hk; simp [List.lookup]
      · have hne : (Key.nm n == k) = false := by simp; exact fun h => hk h.symm
        have hr : Env.has rest n = true := by
          simp [Env.has, hk] at h ⊢; exact h
        simpa [List.lookup, hk, hne] using ih hr
  unfold Env.insert
  split
  · rename_i h
    simpa [Env.find?] using key e h
  · rename_i h
    have hn : e.lookup (Key.nm n) = none := by
      rw [List.lookup_eq_none_iff]
      intro p hp
      simp only [Env.has, Bool.not_eq_true, List.any_eq_false] at h
      have := h p hp
      simp only [Key.beq_eq, decide_eq_false_iff_not] at this
      simpa [bne_iff_ne, ne_eq] using fun h' => this h'.symm
    simp [Env.find?, List.lookup_append, hn, List.lookup]

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
`YIELD field AS alias` re-insert, binder.rs) keeps the table covering. -/
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
`sorted_scope_vars`, binder.rs) and mints `len`, `len + 1`, … in that
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

/-- `Planner::fresh_var` (planner/mod.rs): the planner's copy of one scope's
`scope_vars[scope_id]` (a list of variables); the new variable is nameless (`alias: None`), has id
`len()`, and is pushed so the next mint is one larger. -/
def plannerFresh (sv : List Var) (scope : Nat) (ty : Ty) : Var × List Var :=
  let v : Var := { name := none, id := sv.length, scope, ty }
  (v, sv ++ [v])

/-- The minted id is `len()`, the list grows by one, and the next mint is `len + 1`. -/
theorem plannerFresh_spec (sv : List Var) (scope : Nat) (ty : Ty) :
    (plannerFresh sv scope ty).1.id = sv.length ∧
    (plannerFresh sv scope ty).2.length = sv.length + 1 ∧
    (plannerFresh (plannerFresh sv scope ty).2 scope ty).1.id = sv.length + 1 := by
  simp [plannerFresh]

/-- A list whose ids are all below its length mints an id no entry holds, and stays so. -/
theorem plannerFresh_covers (sv : List Var) (scope : Nat) (ty : Ty)
    (h : ∀ v ∈ sv, v.id < sv.length) :
    (plannerFresh sv scope ty).1.id ∉ sv.map (·.id) ∧
    ∀ v ∈ (plannerFresh sv scope ty).2, v.id < (plannerFresh sv scope ty).2.length := by
  refine ⟨?_, ?_⟩
  · intro hm
    obtain ⟨v, hv, he⟩ := List.mem_map.1 hm
    have := h v hv
    simp [plannerFresh] at he
    omega
  · intro v hv
    simp only [plannerFresh, List.mem_append, List.mem_singleton, List.length_append,
      List.length_singleton] at hv ⊢
    rcases hv with hv | rfl
    · have := h v hv; omega
    · simp

/-- Removing an entry can only shrink the table. -/
theorem erase_length_le (e : Env) (k : Name) : (e.erase k).length ≤ e.length := by
  simp [Env.erase]; exact List.length_filter_le _ _

/-- A covering table stays covering after `erase`/`retain` **only** for the
ids it still holds; an id the IR still uses may now be `≥ len()`.
Concretely (binder.rs): after `MATCH (n) WITH n.v AS k WHERE n.v = 0`
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
