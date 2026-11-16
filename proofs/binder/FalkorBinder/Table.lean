/-
# Scope tables and id minting

`Binder.env_stack : Vec<HashMap<Arc<String>, Variable>>` (binder.rs:47). A
variable's id is minted as the *current size* of its scope's table
(`fresh_var`, binder.rs:2243-2257; the parent copy in `resolve_name`,
binder.rs:2224), and the planner mints its own ids the same way, as
`scope_vars[scope].len()`, pushing each one so the next is one larger
(`Planner::fresh_var`, planner/mod.rs:843-857).

A `HashMap` is modelled as an association list with distinct keys; `insert`
replaces the value of a present key and appends otherwise, so `length` is the
map's `len()`.
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

/-- One scope table. -/
abbrev Env := List (Name × Var)

namespace Env

def find? (e : Env) (k : Name) : Option Var := e.lookup k

def has (e : Env) (k : Name) : Bool := e.any (·.1 == k)

/-- `HashMap::insert`. -/
def insert (e : Env) (k : Name) (v : Var) : Env :=
  if e.has k then e.map (fun p => if p.1 == k then (k, v) else p) else e ++ [(k, v)]

/-- `HashMap::remove`. -/
def erase (e : Env) (k : Name) : Env := e.filter (fun p => p.1 != k)

/-- `HashMap::retain`. -/
def retain (e : Env) (keep : Name → Bool) : Env := e.filter (fun p => keep p.1)

def ids (e : Env) : List Nat := e.map (·.2.id)

/-- Every id in the table is below its size: minting at `len()` is fresh. -/
def Covers (e : Env) : Prop := ∀ i ∈ e.ids, i < e.length

end Env

/-- `Binder::fresh_var` (binder.rs:2243): the id is the table's size. -/
def freshVar (e : Env) (name : Name) (ty : Ty) (scope : Nat) : Var :=
  { name, id := e.length, scope, ty }

theorem Env.insert_new_eq {e : Env} {k : Name} {v : Var} (h : e.has k = false) :
    e.insert k v = e ++ [(k, v)] := by
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
`_reduce_`, `__agg_placeholder_`, `_hidden_` and `_anon` reservations, and
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

/-! ## Hiding instead of removing: `hide_names` (binder.rs:2215-2233, #3062)

Since #3062 the binder no longer *removes* a name it takes out of sight (the WHERE/SKIP/LIMIT copies
of `bind_projection`, a pattern's own names after a pattern predicate or comprehension): it
removes the entry and re-inserts the same variable under a key no query can spell,
`` _slot_hidden`{name}`{id} `` (then `` …`{n} `` for the first `n ≥ 1` whose key is free). The
table keeps its size, so `len()`-minting stays fresh (`hide_covers`, `hideAll_covers`). -/

/-- The `n`-th key `hide_names` tries (binder.rs:2225, 2229). -/
def hiddenKey (name : Name) (id n : Nat) : Name :=
  if n = 0 then "_slot_hidden`" ++ name ++ "`" ++ toString id
  else "_slot_hidden`" ++ name ++ "`" ++ toString id ++ ("`" ++ toString n)

/-- The `while self.current_env().contains_key(&key)` search (binder.rs:2227-2230), from try `n`,
with `fuel` tries left. With `fuel = len()` and `n = 0` the bound is never reached: the
`len() + 1` candidates are pairwise distinct (`hiddenKey_inj`), so one is free
(`hiddenKeyFrom_fresh`) and the search stops there, as the Rust loop does. -/
def hiddenKeyFrom (e : Env) (name : Name) (id : Nat) : Nat → Nat → Name
  | 0, n => hiddenKey name id n
  | f + 1, n => if e.has (hiddenKey name id n) then hiddenKeyFrom e name id f (n + 1)
                else hiddenKey name id n

/-- `hide_names` on one name (binder.rs:2219-2232): `remove`, then `insert` under the free key. -/
def Env.hide (e : Env) (k : Name) : Env :=
  match e.find? k with
  | none => e
  | some v =>
    let e' := e.erase k
    e'.insert (hiddenKeyFrom e' k v.id e'.length 0) v

/-- `hide_names` (binder.rs:2215). -/
def Env.hideAll (e : Env) (ks : List Name) : Env := ks.foldl Env.hide e

/-- `hide_new_names` (binder.rs:2202-2213): hide every key not in `outer` and not `_anon…`. The
Rust collects the keys in `HashMap` order; the model takes the table's order (the theorems below
hold for every order). -/
def Env.hideNew (outer : List Name) (e : Env) : Env :=
  e.hideAll ((e.map (·.1)).filter (fun k => !outer.contains k && !k.startsWith "_anon"))

/-- A `HashMap`'s keys are distinct. -/
def Env.Distinct (e : Env) : Prop := (e.map (·.1)).Nodup

theorem Env.has_iff (e : Env) (k : Name) : e.has k = true ↔ k ∈ e.map (·.1) := by
  simp [Env.has, List.any_eq_true]

/-- Decimal digits read back (the inverse of `Nat.toDigits 10`). -/
def digVal (l : List Char) : Nat := l.foldl (fun acc c => acc * 10 + (c.toNat - 48)) 0

theorem digVal_append (l : List Char) (c : Char) :
    digVal (l ++ [c]) = digVal l * 10 + (c.toNat - 48) := by
  simp [digVal, List.foldl_append]

theorem digVal_toDigits (n : Nat) : digVal (Nat.toDigits 10 n) = n := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    rw [Nat.toDigits_eq_ite (by decide)]
    split
    · rename_i h
      simp [digVal, Nat.toNat_digitChar_sub_48_of_lt_ten h]
    · rename_i h
      rw [digVal_append, ih (n / 10) (by omega),
        Nat.toNat_digitChar_sub_48_of_lt_ten (Nat.mod_lt _ (by decide))]
      omega

theorem nat_toString_inj {a b : Nat} (h : toString a = toString b) : a = b := by
  have h' : (toString a).toList = (toString b).toList := by rw [h]
  simp only [Nat.toString_eq_repr, Nat.toList_repr] at h'
  rw [← digVal_toDigits a, ← digVal_toDigits b, h']

/-- The candidate keys are pairwise distinct. -/
theorem hiddenKey_inj (name : Name) (id : Nat) {a b : Nat}
    (h : hiddenKey name id a = hiddenKey name id b) : a = b := by
  unfold hiddenKey at h
  generalize "_slot_hidden`" ++ name ++ "`" ++ toString id = base at h
  have h1 : ("`" : String).length = 1 := by decide
  by_cases ha : a = 0 <;> by_cases hb : b = 0 <;> simp only [ha, hb, ite_true, ite_false] at h
  · omega
  · have := congrArg String.length h; simp [String.length_append, h1] at this <;> omega
  · have := congrArg String.length h; simp [String.length_append, h1] at this <;> omega
  · exact nat_toString_inj ((String.append_right_inj _).1 ((String.append_right_inj _).1 h))

/-- If the search returned a taken key, every candidate it went through was taken. -/
theorem hiddenKeyFrom_taken (e : Env) (name : Name) (id : Nat) : ∀ (f n : Nat),
    e.has (hiddenKeyFrom e name id f n) = true → ∀ j ≤ f, e.has (hiddenKey name id (n + j)) = true
  | 0, n, h, j, hj => by
    have : j = 0 := by omega
    subst this; simpa [hiddenKeyFrom] using h
  | f + 1, n, h, j, hj => by
    simp only [hiddenKeyFrom] at h
    by_cases hn : e.has (hiddenKey name id n) = true
    · simp only [hn, ite_true] at h
      cases j with
      | zero => simpa using hn
      | succ j =>
        have := hiddenKeyFrom_taken e name id f (n + 1) h j (by omega)
        rwa [show n + 1 + j = n + (j + 1) by omega] at this
    · simp only [hn, Bool.false_eq_true, ite_false] at h <;> exact absurd h hn

/-- Pigeonhole: `m` distinct values all in `K` need `m ≤ K.length`. -/
theorem inj_range_le (f : Nat → Name) (hf : ∀ a b, f a = f b → a = b) : ∀ (m : Nat) (K : List Name),
    (∀ j < m, f j ∈ K) → m ≤ K.length
  | 0, _, _ => Nat.zero_le _
  | m + 1, K, h => by
    have hm : f m ∈ K := h m (by omega)
    have := inj_range_le f hf m (K.erase (f m)) (fun j hj =>
      (List.mem_erase_of_ne (fun e => by have := hf _ _ e; omega)).2 (h j (by omega)))
    rw [List.length_erase_of_mem hm] at this
    have : 0 < K.length := List.length_pos_of_mem hm
    omega

/-- **The re-key is fresh**: the key `hide_names` settles on is not in the table. -/
theorem hiddenKeyFrom_fresh (e : Env) (name : Name) (id : Nat) :
    e.has (hiddenKeyFrom e name id e.length 0) = false := by
  cases h : e.has (hiddenKeyFrom e name id e.length 0)
  · rfl
  · exfalso
    have hall := hiddenKeyFrom_taken e name id e.length 0 h
    have := inj_range_le (hiddenKey name id) (fun a b => hiddenKey_inj name id) (e.length + 1)
      (e.map (·.1)) (fun j hj => by
        have := hall j (by omega); rw [Env.has_iff] at this; simpa using this)
    simp at this; omega

theorem Env.find_mem {e : Env} {k : Name} {v : Var} (h : e.find? k = some v) : (k, v) ∈ e := by
  unfold Env.find? at h
  induction e with
  | nil => simp [List.lookup] at h
  | cons p ps ih =>
    simp only [List.lookup] at h
    split at h
    · rename_i hk
      simp at h; subst h
      have : k = p.1 := by simpa using hk
      subst this; simp
    · exact List.mem_cons_of_mem _ (ih h)

theorem Env.erase_length_of_find {e : Env} {k : Name} {v : Var} (hd : e.Distinct)
    (h : e.find? k = some v) : (e.erase k).length + 1 = e.length := by
  have hm := Env.find_mem h
  unfold Env.Distinct at hd
  induction e with
  | nil => simp at hm
  | cons p ps ih =>
    rw [List.map_cons, List.nodup_cons] at hd
    simp only [Env.erase, List.filter_cons] at ih ⊢
    by_cases hpk : p.1 = k
    · have hps : ps.filter (fun q => q.1 != k) = ps := by
        rw [List.filter_eq_self]; intro q hq
        have : q.1 ≠ k := fun e => hd.1 (by rw [hpk, ← e]; exact List.mem_map_of_mem hq)
        simpa using this
      simp [hpk, hps]
    · have hm' : (k, v) ∈ ps := by
        rcases List.mem_cons.1 hm with e | e
        · exact absurd (by rw [← e]) hpk
        · exact e
      have hf : ps.lookup k = some v := by
        simp only [Env.find?, List.lookup] at h
        have : (k == p.1) = false := by simp; exact fun e => hpk e.symm
        rw [this] at h; exact h
      have := ih hd.2 hf hm'
      simp [hpk]; omega

theorem Env.erase_sub {e : Env} {k : Name} {p : Name × Var} (h : p ∈ e.erase k) : p ∈ e :=
  (List.mem_filter.1 h).1

theorem Env.erase_distinct {e : Env} (k : Name) (hd : e.Distinct) : (e.erase k).Distinct := by
  unfold Env.Distinct Env.erase at *
  induction e with
  | nil => simp
  | cons p ps ih =>
    rw [List.map_cons, List.nodup_cons] at hd
    rw [List.filter_cons]
    split
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨fun hm => hd.1 ?_, ih hd.2⟩
      obtain ⟨q, hq, hqe⟩ := List.mem_map.1 hm
      rw [← hqe]; exact List.mem_map_of_mem (List.mem_filter.1 hq).1
    · exact ih hd.2

/-- Hiding keeps the keys distinct, the size, and every id held. -/
theorem Env.hide_spec (e : Env) (k : Name) (hd : e.Distinct) :
    (e.hide k).Distinct ∧ (e.hide k).length = e.length ∧ (∀ i ∈ (e.hide k).ids, i ∈ e.ids) := by
  unfold Env.hide
  cases hf : e.find? k with
  | none => exact ⟨hd, rfl, fun _ h => h⟩
  | some v =>
    simp only
    have hfr := hiddenKeyFrom_fresh (e.erase k) k v.id
    rw [Env.insert_new_eq hfr]
    refine ⟨?_, ?_, ?_⟩
    · unfold Env.Distinct
      rw [List.map_append, List.nodup_append]
      refine ⟨Env.erase_distinct k hd, by simp, ?_⟩
      intro a ha b hb hab
      simp at hb; subst hb; subst hab
      exact absurd ((Env.has_iff _ _).2 ha) (by rw [hfr]; simp)
    · simp; exact Env.erase_length_of_find hd hf
    · intro i hi
      simp only [Env.ids, List.map_append, List.mem_append, List.mem_map] at hi ⊢
      rcases hi with ⟨p, hp, rfl⟩ | ⟨p, hp, rfl⟩
      · exact ⟨p, Env.erase_sub hp, rfl⟩
      · simp at hp; subst hp; exact ⟨_, Env.find_mem hf, rfl⟩

theorem Env.hideAll_spec (ks : List Name) : ∀ e : Env, e.Distinct →
    (e.hideAll ks).Distinct ∧ (e.hideAll ks).length = e.length ∧
      (∀ i ∈ (e.hideAll ks).ids, i ∈ e.ids) := by
  induction ks with
  | nil => intro e hd; exact ⟨hd, rfl, fun _ h => h⟩
  | cons k ks ih =>
    intro e hd
    obtain ⟨d1, l1, i1⟩ := Env.hide_spec e k hd
    obtain ⟨d2, l2, i2⟩ := ih (e.hide k) d1
    exact ⟨d2, by simp only [Env.hideAll, List.foldl_cons] at *; rw [l2, l1],
      fun i hi => i1 i (i2 i hi)⟩

/-- **#3062 fixes root cause 2**: hiding never shrinks the table and never adds an id, so a
covering table stays covering — no id the bound IR still uses is handed out again. -/
theorem Env.hideAll_covers (e : Env) (ks : List Name) (hd : e.Distinct) (hc : e.Covers) :
    (e.hideAll ks).Covers ∧ (e.hideAll ks).length = e.length := by
  obtain ⟨-, hl, hi⟩ := Env.hideAll_spec ks e hd
  exact ⟨fun i h => by rw [hl]; exact hc i (hi i h), hl⟩

theorem Env.hide_covers (e : Env) (k : Name) (hd : e.Distinct) (hc : e.Covers) :
    (e.hide k).Covers ∧ (e.hide k).length = e.length :=
  Env.hideAll_covers e [k] hd hc

theorem Env.hideNew_covers (outer : List Name) (e : Env) (hd : e.Distinct) (hc : e.Covers) :
    (e.hideNew outer).Covers ∧ (e.hideNew outer).length = e.length :=
  Env.hideAll_covers e _ hd hc

end FalkorBinder
