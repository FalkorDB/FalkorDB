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

Since #3016 the binder never drops an entry whose id may be in use: `Scope::reserve` lists a
variable nameless (once per id), `Scope::lift` names a reserved variable for a nested pattern, and
`Binder::hide_names` moves a named variable to the nameless list. On a table listing each id and
name once (`Env.WF`) hiding keeps the ids and the length (`Env.hideNames_wf`), so `Covers` holds
throughout (`Env.hideNames_covers`, `hide_keeps_slot`).
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

/-- `HashMap::remove` on `named` (inside `Binder::hide_names` / `restore_shadowed`, binder.rs). -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != .nm k)

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

/-- `Scope::unnamed_ids` (#3016): the ids of the unnamed list. -/
def unnamedIds (e : Env) : List Nat := e.unnamed.map (·.id)

/-- `Scope::reserve` (binder.rs, #3016): keep `v`'s id taken without a name; a variable whose id
is already reserved is not listed again. -/
def reserve (e : Env) (v : Var) : Env := if e.unnamedIds.contains v.id then e else e ++ [(.un, v)]

/-- `Scope::unreserve` (binder.rs, #3016): drop the reserved entry with this id. -/
def unreserve (e : Env) (i : Nat) : Env := e.filter (fun p => match p.1 with
  | .un => p.2.id != i
  | .nm _ => true)

/-- `Scope::lift` (binder.rs, #3016): un-reserve `v` and put it under `n`. -/
def lift (e : Env) (n : Name) (v : Var) : Env := (e.unreserve v.id).insert n v

/-- One name of `Binder::hide_names` (binder.rs, #3016): take it out of `named` and reserve its
variable. -/
def hideName (e : Env) (n : Name) : Env := match e.find? n with
  | some v => (e.erase n).reserve v
  | none => e

/-- `Binder::hide_names` (binder.rs, #3016). -/
def hideNames (e : Env) (ns : List Name) : Env := ns.foldl hideName e

/-- `Scope::values` (binder.rs) / `into_values` (binder.rs): every variable. -/
def values (e : Env) : List Var := e.map (·.2)

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

/-- The table lists each id once and each name once (#3016: what the Rust test
`scope_tables_list_each_variable_once` checks, with `Covers` it makes the ids exactly `0..len`). -/
def WF (e : Env) : Prop := e.ids.Nodup ∧ e.keys.Nodup

end Env

/-- `Binder::fresh_var` (binder.rs): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name := some name, id := e.length, scope, ty }

/-- The unnamed variable `define_alias_in_scope` mints for `Alias::Anon k` (binder.rs):
`fresh_var(None, …)` with `anon = Some(k)`. -/
def freshAnon (e : Env) (k : Nat) (ty : Ty) (scope : Nat) : Var :=
  { name := none, id := e.length, scope, ty, anon := some k }

theorem Env.unnamedIds_sub (e : Env) (i : Nat) (h : i ∈ e.unnamedIds) : i ∈ e.ids := by
  induction e with
  | nil => simp [Env.unnamedIds, Env.unnamed] at h
  | cons p rest ih =>
    obtain ⟨k, w⟩ := p
    cases k with
    | nm m =>
      have : i ∈ Env.unnamedIds rest := by simpa [Env.unnamedIds, Env.unnamed] using h
      simp [Env.ids] at ih ⊢; exact Or.inr (ih this)
    | un =>
      simp only [Env.unnamedIds, Env.unnamed, List.filterMap_cons, List.map_cons,
        List.mem_cons] at h
      simp only [Env.ids, List.map_cons, List.mem_cons]
      rcases h with h | h
      · exact Or.inl h
      · exact Or.inr (ih h)

/-- A variable whose id is not reserved yet is appended to the unnamed list. -/
theorem Env.reserve_eq_append {e : Env} {v : Var} (h : v.id ∉ e.unnamedIds) :
    e.reserve v = e ++ [(.un, v)] := by
  simp [Env.reserve, h]

/-- An id already reserved is not listed twice (`unnamed_ids`). -/
theorem Env.reserve_dup {e : Env} {v : Var} (h : v.id ∈ e.unnamedIds) : e.reserve v = e := by
  simp [Env.reserve, h]

/-- `Scope::len` counts the unnamed entries: reserving a new id adds one. -/
theorem Env.reserve_length {e : Env} {v : Var} (h : v.id ∉ e.unnamedIds) :
    (e.reserve v).length = e.length + 1 := by
  simp [Env.reserve_eq_append h]

/-- A reserved entry is reached by no name. -/
theorem Env.reserve_find (e : Env) (v : Var) (n : Name) : (e.reserve v).find? n = e.find? n := by
  unfold Env.reserve
  split
  · rfl
  · have : (Key.nm n == Key.un) = false := by simp
    simp [Env.find?, List.lookup_append, List.lookup, this]

theorem Env.unnamed_append_un (e : Env) (v : Var) : Env.unnamed (e ++ [(Key.un, v)]) = e.unnamed ++ [v] := by
  simp [Env.unnamed, List.filterMap_append]

/-- `Scope::reserve` keeps every unnamed entry. -/
theorem Env.reserve_keeps_unnamed (e : Env) (w v : Var) (h : (.un, v) ∈ e) : (.un, v) ∈ e.reserve w := by
  unfold Env.reserve; split
  · exact h
  · exact List.mem_append_left _ h

/-- `lookup_anon` finds only a variable whose parser number is `k`. -/
theorem Env.lookupAnon_spec (e : Env) (k : Nat) (v : Var) (h : e.lookupAnon k = some v) :
    v.anon = some k ∧ v ∈ e.unnamed := by
  unfold Env.lookupAnon at h
  have := List.find?_some h
  exact ⟨by simpa using this, List.mem_of_find?_eq_some h⟩

/-- A table without the number has no hit, and a newly reserved unnamed variable becomes the hit. -/
theorem Env.lookupAnon_reserve_self (e : Env) (v : Var) (k : Nat) (hv : v.anon = some k)
    (hf : v.id ∉ e.unnamedIds) (h : e.lookupAnon k = none) : (e.reserve v).lookupAnon k = some v := by
  have hn : e.unnamed.find? (fun v => v.anon == some k) = none := h
  simp [Env.reserve_eq_append hf, Env.lookupAnon, Env.unnamed_append_un, List.find?_append, hn, hv]

/-- Reserving a variable of another number (or a named one) changes no hit. -/
theorem Env.lookupAnon_reserve_other (e : Env) (v : Var) (k : Nat) (hv : v.anon ≠ some k) :
    (e.reserve v).lookupAnon k = e.lookupAnon k := by
  unfold Env.reserve
  split
  · rfl
  · have : (v.anon == some k) = false := by simpa using hv
    simp [Env.lookupAnon, Env.unnamed_append_un, List.find?_append, this]

/-- No name-keyed operation changes the unnamed list: `insert` leaves `lookup_anon` alone. -/
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

/-- `HashMap::remove` on `named` leaves the unnamed list alone. -/
theorem Env.erase_unnamed (e : Env) (n : Name) : (e.erase n).unnamed = e.unnamed := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, w⟩ := p
    cases k with
    | nm m =>
      by_cases hm : m = n
      · subst hm; simpa [Env.erase, Env.unnamed] using ih
      · have : (Key.nm m != Key.nm n) = true := by simp [hm]
        simp only [Env.erase, List.filter_cons, this, ite_true, Env.unnamed,
          List.filterMap_cons] at ih ⊢
        exact ih
    | un => simp only [Env.erase, Env.unnamed, List.filter_cons] at ih ⊢; simp [ih]

/-- Hiding a name never touches an unnamed entry. -/
theorem Env.hideName_keeps_unnamed (e : Env) (n : Name) (v : Var) (h : (.un, v) ∈ e) :
    (.un, v) ∈ e.hideName n := by
  unfold Env.hideName
  split
  · apply Env.reserve_keeps_unnamed
    simp [Env.erase, List.mem_filter, h]
  · exact h

/-- Hiding a name leaves `lookup_anon` alone unless the hidden variable is itself an unnamed
entity of that number (a named variable has none). -/
theorem Env.lookupAnon_hideName (e : Env) (n : Name) (k : Nat)
    (hv : ∀ v, e.find? n = some v → v.anon ≠ some k) : (e.hideName n).lookupAnon k = e.lookupAnon k := by
  unfold Env.hideName
  split
  · rename_i v hf
    rw [Env.lookupAnon_reserve_other _ _ _ (hv v hf)]
    simp [Env.lookupAnon, Env.erase_unnamed]
  · rfl

/-- `Scope::values` covers every entry, named or not, so `len()` is its length. -/
theorem Env.values_length (e : Env) : e.values.length = e.length := by
  simp [Env.values]

/-- A newly reserved entry adds no key, no named entry, but one value. -/
theorem Env.reserve_views (e : Env) (v : Var) (h : v.id ∉ e.unnamedIds) :
    (e.reserve v).keys = e.keys ∧ (e.reserve v).named = e.named ∧
      (e.reserve v).values = e.values ++ [v] := by
  simp [Env.reserve_eq_append h, Env.keys, Env.named, Env.values, List.filterMap_append]

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

/-- A variable minted at `len()` is not reserved yet, and reserving it keeps the table covering
(the list comprehension, quantifier, reduce, placeholder, unnamed-entity and CALL-without-YIELD
reservations, #3016). -/
theorem reserve_covers {e : Env} (h : e.Covers) (v : Var) (hv : v.id = e.length) :
    v.id ∉ e.unnamedIds ∧ (e.reserve v).Covers := by
  have hn : v.id ∉ e.unnamedIds := fun hm => by
    have := h _ (Env.unnamedIds_sub _ _ hm); omega
  refine ⟨hn, ?_⟩
  rw [Env.reserve_eq_append hn]
  intro i hi
  simp only [Env.ids, List.map_append, List.mem_append, List.map_cons, List.map_nil,
    List.mem_singleton] at hi
  simp only [List.length_append, List.length_singleton]
  rcases hi with hi | hi
  · have := h i hi; omega
  · omega

/-- Reserving an unnamed entity minted at `len()` keeps the table covering. -/
theorem reserveAnon_covers {e : Env} (h : e.Covers) (k : Nat) (t : Ty) (s : Nat) :
    (e.reserve (freshAnon e k t s)).Covers := (reserve_covers h _ rfl).2

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
binder: `define_name_in_scope`, `project_name` and the parent copy in `resolve_name`; since
#3016 the loop-variable, placeholder and hidden-output reservations are nameless, `reserve_covers`)
keeps the table covering. -/
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

/-! ## Hiding a name keeps its slot (#3016, fixes #3004)

Before #3016 `bind_projection` and the pattern cleanup removed a name while its id was still used
by the bound IR, so the next variable of the scope was minted with that id. `hide_names` moves the
variable to the unnamed list instead: on a table that lists each id and name once (`WF`), hiding
keeps the same ids and the same length, so `Covers` is preserved and the next mint is fresh. -/

theorem Env.erase_find_other (e : Env) (n m : Name) (h : m ≠ n) : (e.erase n).find? m = e.find? m := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : k = .nm n
    · subst hk
      have : (Key.nm m == Key.nm n) = false := by simp [h]
      simpa [Env.erase, Env.find?, List.lookup, this] using ih
    · have hk' : (k != Key.nm n) = true := by simp [hk]
      simp only [Env.erase, Env.find?, List.filter_cons, hk', ite_true, List.lookup] at ih ⊢
      split <;> simp_all

theorem Env.keys_erase (e : Env) (n : Name) : (e.erase n).keys = e.keys.filter (· != n) := by
  induction e with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    cases k with
    | nm m =>
      by_cases hm : m = n
      · subst hm; simpa [Env.erase, Env.keys] using ih
      · have : (Key.nm m != Key.nm n) = true := by simp [hm]
        simp only [Env.erase, List.filter_cons, this, ite_true, Env.keys, List.filterMap_cons] at ih ⊢
        simp [ih, hm]
    | un => simp only [Env.erase, Env.keys, List.filter_cons] at ih ⊢; simp [ih]

theorem Env.erase_perm : ∀ (e : Env) (n : Name) (v : Var), e.keys.Nodup → e.find? n = some v →
    (e.erase n).length + 1 = e.length ∧ ((e.erase n).ids ++ [v.id]).Perm e.ids
  | [], n, v, _, h => by simp [Env.find?] at h
  | (k, w) :: rest, n, v, hk, h => by
    by_cases hkn : k = .nm n
    · subst hkn
      have hv : w = v := by simpa [Env.find?, List.lookup] using h
      subst hv
      have hnot : n ∉ Env.keys rest := by
        have : (Env.keys ((Key.nm n, w) :: rest)) = n :: Env.keys rest := by simp [Env.keys]
        rw [this] at hk; exact (List.nodup_cons.1 hk).1
      have hr : Env.erase rest n = rest := by
        apply List.filter_eq_self.2
        intro p hp
        obtain ⟨k', w'⟩ := p
        simp only [bne_iff_ne, ne_eq, decide_eq_true_eq]
        intro he; subst he
        exact hnot (by simp only [Env.keys, List.mem_filterMap]; exact ⟨_, hp, rfl⟩)
      have he : Env.erase ((Key.nm n, w) :: rest) n = rest := by
        simp only [Env.erase, List.filter_cons, bne_self_eq_false]; exact hr
      rw [he]
      refine ⟨by simp, ?_⟩
      simp only [Env.ids, List.map_cons]
      exact List.perm_append_singleton _ _
    · have hk' : (k != Key.nm n) = true := by simp [hkn]
      have he : Env.erase ((k, w) :: rest) n = (k, w) :: Env.erase rest n := by
        simp only [Env.erase, List.filter_cons, hk', ite_true]
      have hf : Env.find? rest n = some v := by
        have : (Key.nm n == k) = false := by simp; exact fun h => hkn h.symm
        simpa [Env.find?, List.lookup, this] using h
      have hkr : (Env.keys rest).Nodup := by
        cases k with
        | nm m => simp [Env.keys] at hk; exact hk.2
        | un => simpa [Env.keys] using hk
      obtain ⟨hl, hp⟩ := Env.erase_perm rest n v hkr hf
      rw [he]
      refine ⟨by simp; omega, ?_⟩
      simp only [Env.ids, List.map_cons, List.cons_append] at hp ⊢
      exact hp.cons _

/-- **Hiding a name keeps its id**: on a `WF` table, `hide_names` of one name gives a `WF` table of
the same length holding the same ids. -/
theorem Env.hideName_wf {e : Env} (hw : e.WF) (n : Name) :
    (e.hideName n).WF ∧ (e.hideName n).length = e.length ∧ (e.hideName n).ids.Perm e.ids := by
  unfold Env.hideName
  split
  · rename_i v hf
    obtain ⟨hl, hp⟩ := Env.erase_perm e n v hw.2 hf
    have hnd : ((e.erase n).ids ++ [v.id]).Nodup := hp.nodup_iff.2 hw.1
    have hni : v.id ∉ (e.erase n).unnamedIds := fun hm => by
      have := Env.unnamedIds_sub _ _ hm
      have := (List.nodup_append.1 hnd).2.2 _ this _ (List.mem_singleton_self _)
      exact this rfl
    rw [Env.reserve_eq_append hni]
    have hids : Env.ids (e.erase n ++ [(Key.un, v)]) = (e.erase n).ids ++ [v.id] := by
      simp [Env.ids]
    have hkeys : Env.keys (e.erase n ++ [(Key.un, v)]) = (e.erase n).keys := by
      simp [Env.keys, List.filterMap_append]
    refine ⟨⟨by rw [hids]; exact hnd, ?_⟩, by simp; omega, by rw [hids]; exact hp⟩
    rw [hkeys, Env.keys_erase]; exact hw.2.filter _
  · exact ⟨hw, rfl, List.Perm.refl _⟩

theorem Env.hideNames_wf : ∀ (ns : List Name) {e : Env}, e.WF →
    (e.hideNames ns).WF ∧ (e.hideNames ns).length = e.length ∧ (e.hideNames ns).ids.Perm e.ids
  | [], e, hw => ⟨hw, rfl, List.Perm.refl _⟩
  | n :: ns, e, hw => by
    obtain ⟨hw1, hl1, hp1⟩ := Env.hideName_wf hw n
    obtain ⟨hw2, hl2, hp2⟩ := Env.hideNames_wf ns hw1
    exact ⟨hw2, by simp only [Env.hideNames, List.foldl_cons] at hl2 ⊢; omega,
      by simp only [Env.hideNames, List.foldl_cons] at hp2 ⊢; exact hp2.trans hp1⟩

/-- So hiding keeps the table covering: every id the bound IR uses stays below `len()`. -/
theorem Env.hideNames_covers {e : Env} (hw : e.WF) (hc : e.Covers) (ns : List Name) :
    (e.hideNames ns).Covers ∧ ∀ i ∈ e.ids, i ∈ (e.hideNames ns).ids := by
  obtain ⟨-, hl, hp⟩ := Env.hideNames_wf ns hw
  refine ⟨fun i hi => ?_, fun i hi => hp.mem_iff.2 hi⟩
  rw [hl]; exact hc i (hp.mem_iff.1 hi)

theorem Env.hideName_find_self (e : Env) (n : Name) : (e.hideName n).find? n = none := by
  unfold Env.hideName
  split
  · rw [Env.reserve_find, Env.erase_find]
  · assumption

theorem Env.hideName_find_other (e : Env) (n m : Name) (h : m ≠ n) :
    (e.hideName n).find? m = e.find? m := by
  unfold Env.hideName
  split
  · rw [Env.reserve_find, Env.erase_find_other _ _ _ h]
  · rfl

/-- …and no name reaches a hidden variable any more, while every other name is unchanged. -/
theorem Env.hideNames_find : ∀ (ns : List Name) (e : Env) (m : Name),
    (e.hideNames ns).find? m = if m ∈ ns then none else e.find? m
  | [], e, m => by simp [Env.hideNames]
  | n :: ns, e, m => by
    have ih := Env.hideNames_find ns (e.hideName n) m
    simp only [Env.hideNames, List.foldl_cons] at ih ⊢
    rw [ih]
    by_cases hm : m = n
    · subst hm; simp [Env.hideName_find_self]
    · rw [Env.hideName_find_other _ _ _ hm]; simp [hm]

/-- Concretely (binder.rs): after `MATCH (n) WITH n.v AS k WHERE n.v = 0`
the WITH scope is `{k ↦ 0, n ↦ 1}` (`n` copied for the WHERE, id 1, used by
the Filter). Before #3016 `n` was removed, so `x` in a following `MATCH (x)-->(y)` was minted
with id 1, the Filter's slot (#3004). Now `n` is hidden: no name reaches it, its id stays in
the table, and `x` gets id 2. -/
def withScope : Env :=
  [("k", ⟨"k", 0, 1, .any, none⟩), ("n", ⟨"n", 1, 1, .node, none⟩)]

def withScopeAfter : Env := withScope.hideNames ["n"]

theorem withScope_covers : withScope.Covers := by
  intro i hi; simp [withScope, Env.ids] at hi; rcases hi with rfl | rfl <;> decide

theorem hide_keeps_slot :
    withScope.Covers ∧ withScopeAfter.find? "n" = none ∧ 1 ∈ withScopeAfter.ids ∧
    withScopeAfter.Covers ∧ (freshVar withScopeAfter "x" .node 1).id = 2 := by
  refine ⟨withScope_covers, by decide, by decide, ?_, by decide⟩
  unfold Env.Covers; decide

end FalkorBinder
