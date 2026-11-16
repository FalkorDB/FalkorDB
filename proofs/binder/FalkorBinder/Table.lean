/-
# Scope tables and id minting

`Binder.env_stack : Vec<Scope>` (binder.rs:48). Since #3016 a `Scope`
(binder.rs:84-93) is a name map plus a nameless `reserved` list: the variables
whose ids stay taken but that no name reaches. A variable's id is minted as the
*current size* of its scope (`Scope::len` = names + reserved, binder.rs:96-98;
`fresh_var`, binder.rs:2387-2401; the parent copy in `resolve_name`,
binder.rs:2368), and the planner mints its own ids the same way, as
`scope_vars[scope].len()`, pushing each one so the next is one larger
(`Planner::fresh_var`, planner/mod.rs:843-857).

A scope is modelled as one association list keyed by `Key`: `.name n` for an
entry of the `HashMap` (distinct keys; `insert` replaces the value of a present
key and appends otherwise) and `.rsv` for an entry of `reserved` (one per id,
`reserved_ids`). No name a query writes is a `.rsv` key, so `length` is
`Scope::len()` and `find?` sees only the names.
-/
namespace FalkorBinder

abbrev Name := String

/-- `runtime::functions::Type`, restricted to what the binder inspects. -/
inductive Ty
  | any | node | rel | path
  deriving DecidableEq, Repr

/-- `parser::ast::Variable` (ast.rs:92). Equality in Rust is by `id` alone
(ast.rs:113-119) — the record slot. -/
structure Var where
  name  : Name
  id    : Nat
  scope : Nat
  ty    : Ty
  deriving DecidableEq, Repr

/-- A scope entry's key: a name of `Scope::names`, or a slot of `Scope::reserved`. -/
inductive Key
  | name (n : Name)
  | rsv
  deriving DecidableEq, Repr

instance : Coe Name Key := ⟨Key.name⟩

@[simp] theorem Key.name_beq_name (a b : Name) : (Key.name a == Key.name b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

@[simp] theorem Key.name_beq_rsv (a : Name) : (Key.name a == Key.rsv) = false := rfl
@[simp] theorem Key.rsv_beq_name (a : Name) : (Key.rsv == Key.name a) = false := rfl

/-- One scope table (`Scope`, binder.rs:84-93). -/
abbrev Env := List (Key × Var)

namespace Env

/-- `HashMap::get` on `names` (through `Deref`, binder.rs:146-150). -/
def find? (e : Env) (k : Name) : Option Var := e.lookup (.name k)

def has (e : Env) (k : Name) : Bool := e.any (·.1 == .name k)

/-- `HashMap::insert` on `names`. -/
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == .name k then (.name k, v) else p) else e ++ [(.name k, v)]

/-- `HashMap::remove` on `names`. -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != .name k)

/-- `HashMap::retain` on `names` (reserved slots are untouched). -/
def retain (e : Env) (keep : Name → Bool) : Env :=
  e.filter (fun p => match p.1 with | .name n => keep n | .rsv => true)

/-- `names.keys()`. -/
def names (e : Env) : List Name :=
  e.filterMap (fun p => match p.1 with | .name n => some n | .rsv => none)

/-- `reserved_ids.contains(id)`. -/
def rsvHas (e : Env) (id : Nat) : Bool := e.any (fun p => p.1 == .rsv && p.2.id == id)

/-- `Scope::reserve` (binder.rs:104-112): listed once per id. -/
def reserve (e : Env) (v : Var) : Env := if e.rsvHas v.id then e else e ++ [(.rsv, v)]

/-- `Scope::unreserve` (binder.rs:114-121). -/
def unreserve (e : Env) (id : Nat) : Env := e.filter (fun p => !(p.1 == .rsv && p.2.id == id))

/-- `Scope::lift` (binder.rs:127-134). -/
def lift (e : Env) (k : Name) (v : Var) : Env := (e.unreserve v.id).insert k v

/-- One step of `Binder::hide_names` (binder.rs:355-366): the name goes, its
variable stays reserved. -/
def hide (e : Env) (k : Name) : Env :=
  match e.find? k with
  | some v => (e.erase k).reserve v
  | none => e

def hideAll (e : Env) (ks : List Name) : Env := ks.foldl hide e

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

end Env

/-- `Binder::fresh_var` (binder.rs:2243): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name, id := e.length, scope, ty }

theorem Env.insert_new_eq {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    e.insert k v = e ++ [(.name k, v)] := by
  simp [Env.insert, h]

theorem Env.insert_old_length {e : Env} {k : Name} {v : Var} (h : e.has k = true) :
    (e.insert k v).length = e.length := by
  simp [Env.insert, h]

theorem Env.insert_new_length {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    (e.insert k v).length = e.length + 1 := by
  simp [Env.insert_new_eq h]

theorem Env.names_append (e f : Env) : (e ++ f).names = e.names ++ f.names := by
  simp [Env.names, List.filterMap_append]

theorem Env.names_name (k : Name) (v : Var) : Env.names [(.name k, v)] = [k] := rfl

theorem Env.has_iff (e : Env) (k : Name) : e.has k = true ↔ k ∈ e.names := by
  induction e with
  | nil => simp [Env.has, Env.names]
  | cons p ps ih =>
    obtain ⟨key, v⟩ := p
    cases key with
    | name n =>
      have : Env.names ((Key.name n, v) :: ps) = n :: Env.names ps := rfl
      rw [this]; simp only [Env.has, List.any_cons] at ih ⊢
      rw [List.mem_cons, ← ih, Bool.or_eq_true]
      have : (Key.name n == Key.name k) = true ↔ k = n := by
        simp only [beq_iff_eq, Key.name.injEq]; exact eq_comm
      rw [this]
    | rsv =>
      have : Env.names ((Key.rsv, v) :: ps) = Env.names ps := rfl
      rw [this, ← ih]; simp [Env.has]

/-- The minted id is not in use, provided the table covers its ids. -/
theorem fresh_not_used {e : Env} (h : e.Covers) (n : Name) (t : Ty) (s : Nat) :
    (freshVar e n t s).id ∉ e.ids := by
  intro hm
  have := h _ hm
  simp [freshVar] at this

/-- Minting under a *new* name (every `fresh_var` + `insert` pair in the
binder: `define_name_in_scope`, `project_name`, the `_anon` slots and the
parent copy in `resolve_name`) keeps the table covering. The nameless
reservations (`reserve_fresh_covers`) replaced the `_quant_`, `_lc_`,
`_reduce_`, `__agg_placeholder_` and `_hidden_` keys in #3016. -/
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

/-- A variable minted at `len()` is not yet reserved in a covering table. -/
theorem rsvHas_fresh {e : Env} (h : e.Covers) : e.rsvHas e.length = false := by
  unfold Env.rsvHas
  rw [List.any_eq_false]
  intro p hp hc
  simp only [Bool.and_eq_true, beq_iff_eq] at hc
  have := h p.2.id (List.mem_map_of_mem hp)
  omega

/-- Reserving the variable minted at `len()` (`fresh_var` + `Scope::reserve`)
appends one nameless slot and keeps the table covering. -/
theorem reserve_fresh_eq {e : Env} (h : e.Covers) (n : Name) (t : Ty) (s : Nat) :
    e.reserve (freshVar e n t s) = e ++ [(.rsv, freshVar e n t s)] := by
  simp [Env.reserve, freshVar, rsvHas_fresh h]

theorem reserve_fresh_covers {e : Env} (h : e.Covers) (n : Name) (t : Ty) (s : Nat) :
    (e.reserve (freshVar e n t s)).Covers := by
  rw [reserve_fresh_eq h]
  intro i hi
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
  [("k", ⟨"k", 0, 1, .any⟩), ("n", ⟨"n", 1, 1, .node⟩)]

def withScopeAfter : Env := withScope.erase "n"

theorem withScope_covers : withScope.Covers := by
  intro i hi; simp [withScope, Env.ids] at hi; rcases hi with rfl | rfl <;> decide

theorem erase_breaks_covers :
    withScope.Covers ∧ 1 ∈ withScope.ids ∧
    (freshVar withScopeAfter "x" .node 1).id = 1 := by
  refine ⟨withScope_covers, by decide, by decide⟩

end FalkorBinder
