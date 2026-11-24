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
(`named.len() + unnamed.len() - aliased`, binder.rs): the second names
`import_parent_names` enters with `insert_alias` take no id, and an `Env` leaves them out; they are
kept beside it while a pattern is bound (`RScope` and `Env.withAliases` below prove that the Rust
scope with its `aliased` counter has length `Env.length` and the same lookups).
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

/-- `HashMap::remove` on `named` returning the entry: the first entry keyed `n` and the table
without it. -/
def take1 : Env → Name → Option (Var × Env)
  | [], _ => none
  | (k, v) :: r, n => if k = .nm n then some (v, r) else
    match take1 r n with
    | some (w, r') => some (w, (k, v) :: r')
    | none => none

/-- `Binder::hide_names` (binder.rs): each name present is removed (`Scope::remove`) and its
variable pushed to the unnamed list (`Scope::push_unnamed`), so no name reaches it but it still
counts toward `len()`. -/
def hide (e : Env) : List Name → Env
  | [] => e
  | n :: ns => match take1 e n with
    | some (v, r) => hide (r ++ [(.un, v)]) ns
    | none => hide e ns

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
`insert` leaves `lookup_anon` alone, and `hide` (`hide_names`) keeps every earlier hit. -/
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

theorem Env.take1_spec (e : Env) (n : Name) (v : Var) (r : Env) (h : Env.take1 e n = some (v, r)) :
    r.length + 1 = e.length ∧ (∀ p ∈ r, p ∈ e) ∧ (.nm n, v) ∈ e ∧ r.unnamed = e.unnamed := by
  induction e generalizing r with
  | nil => simp [Env.take1] at h
  | cons p rest ih =>
    obtain ⟨k, w⟩ := p
    simp only [Env.take1] at h
    split at h
    · rename_i hk
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      subst hk
      refine ⟨by simp, fun p hp => by simp [hp], by simp, by simp [Env.unnamed]⟩
    · rename_i hk
      cases ht : Env.take1 rest n with
      | none => simp [ht] at h
      | some q =>
        obtain ⟨w', r'⟩ := q
        simp only [ht, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨h1, h2, h3, h4⟩ := ih r' ht
        refine ⟨by simp; omega, ?_, by simp [h3], ?_⟩
        · intro p hp; simp at hp; rcases hp with rfl | hp
          · simp
          · simp [h2 p hp]
        · cases k with
          | nm m => simpa [Env.unnamed] using h4
          | un => simp [Env.unnamed] at h4 ⊢; exact h4

/-- `hide_names` keeps the table's size: every hidden variable still counts toward `len()`. -/
theorem Env.hide_length (e : Env) (ns : List Name) : (e.hide ns).length = e.length := by
  induction ns generalizing e with
  | nil => rfl
  | cons n ns ih =>
    simp only [Env.hide]
    cases ht : Env.take1 e n with
    | none => exact ih e
    | some q =>
      obtain ⟨v, r⟩ := q
      have := (Env.take1_spec e n v r ht).1
      rw [ih]; simp; omega

/-- … and every id it holds was held before. -/
theorem Env.hide_ids (e : Env) (ns : List Name) : ∀ i ∈ (e.hide ns).ids, i ∈ e.ids := by
  induction ns generalizing e with
  | nil => intro i hi; exact hi
  | cons n ns ih =>
    simp only [Env.hide]
    cases ht : Env.take1 e n with
    | none => exact ih e
    | some q =>
      obtain ⟨v, r⟩ := q
      obtain ⟨-, h2, h3, -⟩ := Env.take1_spec e n v r ht
      intro i hi
      have := ih _ i hi
      simp only [Env.ids, List.map_append, List.mem_append, List.mem_map, List.map_cons,
        List.map_nil, List.mem_singleton] at this ⊢
      rcases this with ⟨p, hp, rfl⟩ | rfl
      · exact ⟨p, h2 p hp, rfl⟩
      · exact ⟨_, h3, rfl⟩

/-- So `hide_names` keeps a covering table covering (where removing the names did not:
`Scope::remove` before #3062 freed ids the plan still used). -/
theorem hide_covers {e : Env} (h : e.Covers) (ns : List Name) : (e.hide ns).Covers := by
  intro i hi
  rw [Env.hide_length]
  exact h i (Env.hide_ids e ns i hi)

/-- The unnamed list only grows at its end under `hide_names`. -/
theorem Env.hide_unnamed (e : Env) (ns : List Name) : ∃ l, (e.hide ns).unnamed = e.unnamed ++ l := by
  induction ns generalizing e with
  | nil => exact ⟨[], by simp [Env.hide]⟩
  | cons n ns ih =>
    simp only [Env.hide]
    cases ht : Env.take1 e n with
    | none => exact ih e
    | some q =>
      obtain ⟨v, r⟩ := q
      obtain ⟨-, -, -, h4⟩ := Env.take1_spec e n v r ht
      obtain ⟨l, hl⟩ := ih (r ++ [(.un, v)])
      refine ⟨v :: l, ?_⟩
      rw [hl]; simp [Env.unnamed, List.filterMap_append] at h4 ⊢; rw [h4]

theorem Env.lookupAnon_hide (e : Env) (ns : List Name) (k : Nat) (v : Var)
    (h : e.lookupAnon k = some v) : (e.hide ns).lookupAnon k = some v := by
  obtain ⟨l, hl⟩ := Env.hide_unnamed e ns
  have h' : e.unnamed.find? (fun v => v.anon == some k) = some v := h
  simp [Env.lookupAnon, hl, List.find?_append, h']

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

/-- **W2-binder-3, fixed by #3062.** After `MATCH (n) WITH n.v AS k WHERE n.v = 0` the WITH scope
is `{k ↦ 0, n ↦ 1}` (`n` copied for the WHERE, id 1, read by the Filter). `bind_projection` now
hides `n` (`hide_names`) rather than removing it: the table keeps size 2, so the next variable of
the scope, `x` in `MATCH (x)-->(y)`, gets id 2, not the Filter's 1. -/
def withScope : Env :=
  [("k", ⟨"k", 0, 1, .any, none⟩), ("n", ⟨"n", 1, 1, .node, none⟩)]

def withScopeAfter : Env := withScope.hide ["n"]

theorem withScope_covers : withScope.Covers := by
  intro i hi; simp [withScope, Env.ids] at hi; rcases hi with rfl | rfl <;> decide

theorem hide_keeps_where_copy :
    withScope.Covers ∧ 1 ∈ withScope.ids ∧ withScopeAfter.keys = ["k"] ∧
    withScopeAfter.Covers ∧ (freshVar withScopeAfter "x" .node 1).id = 2 := by
  exact ⟨withScope_covers, by decide, by decide, hide_covers withScope_covers _, by decide⟩

/-! ## The `aliased` counter: `Scope::len`, `insert_alias`, `remove_alias` (#3062)

`import_parent_names` enters a renamed parent name (`n` of `WITH n AS m`) as a second name for the
projected variable with `Scope::insert_alias`, and `drop_aliases` removes it with
`Scope::remove_alias`. `RScope` is the Rust `Scope` with its counter; `Env.withAliases e al` is the
scope an `Env` stands for while the second names `al` are entered over it. -/

/-- `Scope` (binder.rs) as Rust holds it. -/
structure RScope where
  named : List (Name × Var)
  unnamed : List Var
  aliased : Nat

/-- `Scope::len` (binder.rs). -/
def RScope.len (s : RScope) : Nat := s.named.length + s.unnamed.length - s.aliased

/-- `Scope::get` (binder.rs). -/
def RScope.get (s : RScope) (n : Name) : Option Var := s.named.lookup n

/-- `Scope::insert_alias` (binder.rs): `HashMap::insert`, counting a new key. -/
def RScope.insertAlias (s : RScope) (n : Name) (v : Var) : RScope :=
  if (s.named.lookup n).isSome then
    { s with named := s.named.map fun p => if p.1 == n then (n, v) else p }
  else { s with named := s.named ++ [(n, v)], aliased := s.aliased + 1 }

/-- `Scope::remove_alias` (binder.rs): `HashMap::remove`, uncounting a present key. -/
def RScope.removeAlias (s : RScope) (n : Name) : RScope :=
  if (s.named.lookup n).isSome then
    { s with named := s.named.filter (·.1 != n), aliased := s.aliased - 1 }
  else s

/-- `Binder::drop_aliases` (binder.rs). -/
def RScope.dropAliases (s : RScope) (ns : List Name) : RScope := ns.foldl RScope.removeAlias s

def Env.withAliases (e : Env) (al : List (Name × Var)) : RScope :=
  ⟨e.named ++ al, e.unnamed, al.length⟩

theorem Env.named_unnamed_length (e : Env) : e.named.length + e.unnamed.length = e.length := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    cases k <;> simp [Env.named, Env.unnamed] at ih ⊢ <;> omega

theorem Env.named_lookup (e : Env) (n : Name) : e.named.lookup n = e.find? n := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    cases k with
    | nm m =>
      by_cases h : m = n
      · subst h; simp [Env.named, Env.find?, List.lookup]
      · have h1 : (n == m) = false := by simp; exact fun h' => h h'.symm
        have h2 : (Key.nm n == Key.nm m) = false := by simp; exact fun h' => h h'.symm
        simp only [Env.named, Env.find?, List.filterMap_cons, List.lookup, h1, h2] at ih ⊢
        exact ih
    | un =>
      have h2 : (Key.nm n == Key.un) = false := by simp
      simp only [Env.named, Env.find?, List.filterMap_cons, List.lookup, h2] at ih ⊢
      exact ih

/-- **`len()` ignores the second names**: the Rust scope with aliases `al` entered over `e` has
`len()` = `e.length`, so `fresh_var` mints exactly what it would without them. -/
theorem Env.withAliases_len (e : Env) (al : List (Name × Var)) :
    (e.withAliases al).len = e.length := by
  simp only [Env.withAliases, RScope.len, List.length_append]
  have := Env.named_unnamed_length e
  omega

/-- … and `Scope::get` sees the table's names first, then the second names. -/
theorem Env.withAliases_get (e : Env) (al : List (Name × Var)) (n : Name) :
    (e.withAliases al).get n = (e.find? n).or (al.lookup n) := by
  simp only [Env.withAliases, RScope.get, List.lookup_append, Env.named_lookup]

/-- `insert_alias` of a name in neither the table nor the second names counts it and enters it. -/
theorem Env.insertAlias_withAliases (e : Env) (al : List (Name × Var)) (n : Name) (v : Var)
    (he : e.find? n = none) (ha : al.lookup n = none) :
    (e.withAliases al).insertAlias n v = e.withAliases (al ++ [(n, v)]) := by
  have : ((e.named ++ al).lookup n).isSome = false := by
    simp [List.lookup_append, Env.named_lookup, he, ha]
  simp [RScope.insertAlias, Env.withAliases, this]

theorem Env.named_filter_of_find (e : Env) (n : Name) (he : e.find? n = none) :
    e.named.filter (·.1 != n) = e.named := by
  rw [List.filter_eq_self]
  intro p hp
  have hl : e.named.lookup n = none := by rw [Env.named_lookup]; exact he
  rw [List.lookup_eq_none_iff] at hl
  have := hl p hp
  simp only [bne_iff_ne, ne_eq] at this ⊢
  intro h; exact this (by simp [h])

/-- **`drop_aliases` undoes the import**: removing every second name, each absent from the
table and entered once, leaves the scope the table alone stands for, counter back to 0. -/
theorem Env.dropAliases_withAliases (e : Env) (al : List (Name × Var))
    (he : ∀ p ∈ al, e.find? p.1 = none) (hd : (al.map (·.1)).Nodup) :
    (e.withAliases al).dropAliases (al.map (·.1)) = e.withAliases [] := by
  induction al with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨n, v⟩ := p
    simp only [List.map_cons, List.nodup_cons] at hd
    have hn : e.find? n = none := he (n, v) (by simp)
    have hrest : rest.filter (·.1 != n) = rest := by
      rw [List.filter_eq_self]
      intro q hq
      simp only [bne_iff_ne, ne_eq]
      intro h; exact hd.1 (List.mem_map.2 ⟨q, hq, h⟩)
    have step : (e.withAliases ((n, v) :: rest)).removeAlias n = e.withAliases rest := by
      have : ((e.named ++ (n, v) :: rest).lookup n).isSome = true := by
        simp [List.lookup_append, Env.named_lookup, hn, List.lookup]
      simp only [RScope.removeAlias, Env.withAliases, this, ite_true, List.filter_append,
        Env.named_filter_of_find e n hn, List.filter_cons, bne_self_eq_false, hrest]
      simp
    simp only [RScope.dropAliases, List.map_cons, List.foldl_cons] at ih ⊢
    rw [step]
    exact ih (fun q hq => he q (by simp [hq])) hd.2

end FalkorBinder
