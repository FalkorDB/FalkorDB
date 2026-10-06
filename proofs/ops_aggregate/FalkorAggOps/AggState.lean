import FalkorAggOps.AggPlan
/-
Aggregate operator state: accumulator zeroing / unbinding, `GroupKey` Eq/Hash,
`hash_u64`, `args_for`, `new` (graph/src/runtime/ops/aggregate.rs).
Table in `AggPlan.lean`. An environment is `Nat → Option Z` (slot ↦ bound value);
`Row::insert` = `upd`, `Row::unbind` = `clr`.
-/
namespace FalkorAggOps.AggPlan

variable {Z : Type}

abbrev Env (Z : Type) := Nat → Option Z
def upd (env : Env Z) (k : Nat) (z : Z) : Env Z := fun j => if j = k then some z else env j
def clr (env : Env Z) (k : Nat) : Env Z := fun j => if j = k then none else env j

-- The outermost aggregate calls of a tree, pre-order (no descent into an aggregate).
mutual
def outer : Ex Z → List (Fn Z × List (Ex Z))
  | .func f cs => if f.init.isSome then [(f, cs)] else outerL cs
  | .var _ | .const _ => []
  | .prop _ cs | .distinct cs | .other cs => outerL cs
def outerL : List (Ex Z) → List (Fn Z × List (Ex Z))
  | [] => []
  | e :: es => outer e ++ outerL es
end

/-! ## `set_agg_expr_zero` (aggregate.rs:1082) — `none` = the `unreachable!()` / underflow panic -/

mutual
def setZero : Ex Z → Env Z → Option (Env Z)
  | .func f cs, env =>
    match f.init with
    | some z => match cs.getLast? with
      | some (.var k) => some (upd env k z)
      | _ => none
    | none => setZeroL cs env
  | .var _, env | .const _, env => some env
  | .prop _ cs, env | .distinct cs, env | .other cs, env => setZeroL cs env
def setZeroL : List (Ex Z) → Env Z → Option (Env Z)
  | [], env => some env
  | e :: es, env => (setZero e env).bind (setZeroL es)
end

def stepZ (env : Env Z) (p : Fn Z × List (Ex Z)) : Option (Env Z) :=
  match p.1.init with
  | some z => match p.2.getLast? with
    | some (.var k) => some (upd env k z)
    | _ => none
  | none => none

mutual
theorem setZero_eq : ∀ (e : Ex Z) (env : Env Z), setZero e env = (outer e).foldlM stepZ env
  | .func f cs, env => by
      unfold setZero outer
      cases h : f.init with
      | some z => simp [stepZ, h]
      | none => simp [setZeroL_eq cs env]
  | .var _, _ => rfl
  | .const _, _ => rfl
  | .prop _ cs, env => by simp [setZero, outer, setZeroL_eq cs env]
  | .distinct cs, env => by simp [setZero, outer, setZeroL_eq cs env]
  | .other cs, env => by simp [setZero, outer, setZeroL_eq cs env]
theorem setZeroL_eq : ∀ (es : List (Ex Z)) (env : Env Z), setZeroL es env = (outerL es).foldlM stepZ env
  | [], _ => rfl
  | e :: es, env => by
      simp only [setZeroL, outerL, List.foldlM_append, setZero_eq e env]
      cases (outer e).foldlM stepZ env with
      | none => rfl
      | some env' => simp [setZeroL_eq es env']
end

/-- Every outermost aggregate call has its accumulator variable as last child. -/
def AccWF (l : List (Fn Z × List (Ex Z))) : Prop :=
  ∀ p ∈ l, ∃ k, p.2.getLast? = some (.var k)

theorem foldlM_stepZ_ok (l : List (Fn Z × List (Ex Z))) (env : Env Z) (hw : AccWF l)
    (hi : ∀ p ∈ l, p.1.init.isSome) : ∃ env', l.foldlM stepZ env = some env' := by
  induction l generalizing env with
  | nil => exact ⟨env, rfl⟩
  | cons p ps ih =>
    obtain ⟨k, hk⟩ := hw p (List.mem_cons_self ..)
    have hs := hi p (List.mem_cons_self ..)
    cases hz : p.1.init with
    | none => simp [hz] at hs
    | some z =>
      simp only [List.foldlM_cons, stepZ, hz, hk]
      exact ih _ (fun q hq => hw q (List.mem_cons_of_mem _ hq)) (fun q hq => hi q (List.mem_cons_of_mem _ hq))

mutual
theorem outer_agg : ∀ (e : Ex Z), ∀ p ∈ outer e, p.1.init.isSome
  | .func f cs => by
      unfold outer; split
      · rename_i h; intro p hp; simp at hp; subst hp; exact h
      · exact outerL_agg cs
  | .var _ => by simp [outer]
  | .const _ => by simp [outer]
  | .prop _ cs => by unfold outer; exact outerL_agg cs
  | .distinct cs => by unfold outer; exact outerL_agg cs
  | .other cs => by unfold outer; exact outerL_agg cs
theorem outerL_agg : ∀ (es : List (Ex Z)), ∀ p ∈ outerL es, p.1.init.isSome
  | [] => by simp [outerL]
  | e :: es => by
      intro p hp; simp only [outerL, List.mem_append] at hp
      rcases hp with hp | hp
      · exact outer_agg e p hp
      · exact outerL_agg es p hp
end

/-- **`set_agg_expr_zero`** writes `initial` into the accumulator slot of exactly the
outermost aggregate calls, in pre-order, and never panics on a well-formed plan. -/
theorem setZero_spec (e : Ex Z) (env : Env Z) (hw : AccWF (outer e)) :
    setZero e env = (outer e).foldlM stepZ env ∧ ∃ env', setZero e env = some env' := by
  rw [setZero_eq]
  exact ⟨rfl, foldlM_stepZ_ok _ env hw (outer_agg e)⟩

/-! ## `unbind_agg_accumulators` (aggregate.rs:1230) -/

mutual
def unbindAcc : Ex Z → Env Z → Env Z
  | .func f cs, env =>
    if f.init.isSome then
      if 2 ≤ cs.length then
        match cs.getLast? with
        | some (.var k) => clr env k
        | _ => env
      else env
    else unbindAccL cs env
  | .var _, env | .const _, env => env
  | .prop _ cs, env | .distinct cs, env | .other cs, env => unbindAccL cs env
def unbindAccL : List (Ex Z) → Env Z → Env Z
  | [], env => env
  | e :: es, env => unbindAccL es (unbindAcc e env)
end

def stepU (env : Env Z) (p : Fn Z × List (Ex Z)) : Env Z :=
  if 2 ≤ p.2.length then
    match p.2.getLast? with
    | some (.var k) => clr env k
    | _ => env
  else env

def accIs : Option (Ex Z) → Nat → Bool
  | some (.var k), j => k == j
  | _, _ => false

theorem accIs_iff (o : Option (Ex Z)) (j : Nat) : accIs o j = true ↔ o = some (.var j) := by
  cases o with
  | none => simp [accIs]
  | some x => cases x <;> simp [accIs]

theorem stepU_apply (e : Env Z) (p : Fn Z × List (Ex Z)) (j : Nat) :
    stepU e p j = if 2 ≤ p.2.length ∧ accIs p.2.getLast? j = true then none else e j := by
  unfold stepU
  by_cases h2 : 2 ≤ p.2.length
  · simp only [h2, ite_true, true_and]
    cases hl : p.2.getLast? with
    | none => simp [accIs]
    | some x =>
      cases x <;> simp [accIs]
      rename_i k
      simp only [clr]
      by_cases hj : j = k <;> simp [hj]
      intro h; exact absurd h.symm hj
  · simp [h2]

mutual
theorem unbindAcc_eq : ∀ (e : Ex Z) (env : Env Z), unbindAcc e env = (outer e).foldl stepU env
  | .func f cs, env => by
      unfold unbindAcc outer
      cases h : f.init.isSome
      · simp [unbindAccL_eq cs env]
      · simp [stepU]
  | .var _, _ => rfl
  | .const _, _ => rfl
  | .prop _ cs, env => by simp [unbindAcc, outer, unbindAccL_eq cs env]
  | .distinct cs, env => by simp [unbindAcc, outer, unbindAccL_eq cs env]
  | .other cs, env => by simp [unbindAcc, outer, unbindAccL_eq cs env]
theorem unbindAccL_eq : ∀ (es : List (Ex Z)) (env : Env Z), unbindAccL es env = (outerL es).foldl stepU env
  | [], _ => rfl
  | e :: es, env => by
      simp only [unbindAccL, outerL, List.foldl_append, unbindAcc_eq e env, unbindAccL_eq es]
end

/-- Every slot `set_agg_expr_zero` wrote is unbound again by `unbind_agg_accumulators`,
provided every outermost aggregate has an argument besides its accumulator (true of every
parsed call: `count(*)` is `count(1)`, cypher.rs:2023). Other slots are untouched. -/
theorem unbind_after_zero (l : List (Fn Z × List (Ex Z))) (env : Env Z) (k : Nat)
    (h2 : ∀ p ∈ l, 2 ≤ p.2.length) (hk : ∃ p ∈ l, p.2.getLast? = some (.var k)) :
    l.foldl stepU env k = none := by
  induction l generalizing env with
  | nil => simp at hk
  | cons p ps ih =>
    simp only [List.foldl_cons]
    rcases hk with ⟨q, hq, hqk⟩
    rcases List.mem_cons.mp hq with rfl | hq'
    · have hc : stepU env q k = none := by
        rw [stepU_apply]; simp [h2 q (List.mem_cons_self ..), hqk, accIs]
      have keep : ∀ (l : List (Fn Z × List (Ex Z))) (e : Env Z), e k = none → l.foldl stepU e k = none := by
        intro l; induction l with
        | nil => intro e he; exact he
        | cons r rs ihr =>
          intro e he; apply ihr
          rw [stepU_apply]; split <;> simp_all
      exact keep ps _ hc
    · exact ih _ (fun r hr => h2 r (List.mem_cons_of_mem _ hr)) ⟨q, hq', hqk⟩

theorem unbind_other (l : List (Fn Z × List (Ex Z))) (env : Env Z) (k : Nat)
    (hk : ∀ p ∈ l, p.2.getLast? ≠ some (.var k)) : l.foldl stepU env k = env k := by
  induction l generalizing env with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.foldl_cons]
    rw [ih _ (fun r hr => hk r (List.mem_cons_of_mem _ hr))]
    have hp := hk p (List.mem_cons_self ..)
    have : accIs p.2.getLast? k = false := by
      cases h : accIs p.2.getLast? k
      · rfl
      · exact absurd ((accIs_iff _ _).mp h) hp
    rw [stepU_apply]; simp [this]

/-- Latent asymmetry: a one-child aggregate call (accumulator only) is zeroed but never
unbound — its accumulator would leak downstream. No parsed call has that shape. -/
theorem one_child_not_unbound (f : Fn Z) (z : Z) (k : Nat) (env : Env Z) (hf : f.init = some z) :
    unbindAcc (.func f [.var k]) env = env ∧ setZero (.func f [.var k]) env = some (upd env k z) := by
  simp [unbindAcc, setZero, hf]

/-! ## `GroupKey` (aggregate.rs:90 `eq`, :101 `hash`) -/

/-- `self.0 == other.0`: slice equality = same length and pairwise `Value::eq`. -/
def gkEq (veq : Z → Z → Bool) : List Z → List Z → Bool
  | [], [] => true
  | a :: as, b :: bs => veq a b && gkEq veq as bs
  | _, _ => false

/-- `for v in &self.0 { v.hash(state) }`. -/
def gkHash {S : Type} (hv : S → Z → S) (st : S) (vs : List Z) : S := vs.foldl hv st

theorem gkEq_iff (veq : Z → Z → Bool) (a b : List Z) :
    gkEq veq a b = true ↔ a.length = b.length ∧ ∀ i (h1 : i < a.length) (h2 : i < b.length),
      veq a[i] b[i] = true := by
  induction a generalizing b with
  | nil => cases b <;> simp [gkEq]
  | cons x xs ih =>
    cases b with
    | nil => simp [gkEq]
    | cons y ys =>
      simp only [gkEq, Bool.and_eq_true, ih, List.length_cons, Nat.add_right_cancel_iff]
      constructor
      · rintro ⟨hxy, hl, hall⟩
        refine ⟨hl, fun i h1 h2 => ?_⟩
        cases i with
        | zero => exact hxy
        | succ i => exact hall i (by simpa using h1) (by simpa using h2)
      · rintro ⟨hl, hall⟩
        exact ⟨hall 0 (by simp) (by simp), hl, fun i h1 h2 => hall (i + 1) (by simpa using h1) (by simpa using h2)⟩

/-- The `HashMap` contract: keys equal under `GroupKey::eq` hash equally, given that
`Value`'s `Hash` agrees with its `Eq` (hypothesis `hc`). -/
theorem gkHash_consistent {S : Type} (veq : Z → Z → Bool) (hv : S → Z → S)
    (hc : ∀ s x y, veq x y = true → hv s x = hv s y) (st : S) (a b : List Z)
    (h : gkEq veq a b = true) : gkHash hv st a = gkHash hv st b := by
  induction a generalizing b st with
  | nil => cases b <;> simp_all [gkEq]
  | cons x xs ih =>
    cases b with
    | nil => simp [gkEq] at h
    | cons y ys =>
      simp only [gkEq, Bool.and_eq_true] at h
      simp only [gkHash, List.foldl_cons]
      rw [hc st x y h.1]
      exact ih _ ys h.2

/-! ## `hash_u64` (aggregate.rs:1254 trait decl, :1258 `impl HashU64 for Row`) -/

/-- `FxHasher::default()`, `self.hash(&mut hasher)`, `hasher.finish()`. -/
def hashU64 {R S : Type} (fxDefault : S) (rowHash : S → R → S) (finish : S → Nat) (r : R) : Nat :=
  finish (rowHash fxDefault r)

/-- `hash_u64` depends only on the row's `Hash` input: rows feeding the same
`(key, value)` sequence get the same group id. (It is not injective — the
collision bug is `hashed_dedup_merges_groups`.) -/
theorem hashU64_spec {R S : Type} (d : S) (rh : S → R → S) (fin : S → Nat) (r₁ r₂ : R)
    (heq : rh d r₁ = rh d r₂) : hashU64 d rh fin r₁ = hashU64 d rh fin r₂ ∧
    hashU64 d rh fin r₁ = fin (rh d r₁) := by
  simp [hashU64, heq]

/-! ## `VectorizableAgg::args_for` (aggregate.rs:208) -/

def argsFor (vt vd : List Z → Except String Unit) (value : Z) (extra : List Z) :
    Except String (List Z) := do
  let inputs := value :: extra
  vt inputs
  vd inputs
  pure inputs

/-- `[value] ++ extra_args`, type-checked first, then domain-checked (the guard that keeps
an out-of-range percentile from reaching the kernel). -/
theorem argsFor_spec (vt vd : List Z → Except String Unit) (v : Z) (extra : List Z) :
    argsFor vt vd v extra =
      (match vt (v :: extra) with
       | .error e => .error e
       | .ok () => match vd (v :: extra) with
         | .error e => .error e
         | .ok () => .ok (v :: extra)) := by
  unfold argsFor
  cases h1 : vt (v :: extra) <;> cases h2 : vd (v :: extra) <;>
    simp [h1, h2, bind, Except.bind, pure, Except.pure]

/-! ## `AggregateOp::new` (aggregate.rs:280) -/

structure AggSt (C E G A : Type) (Z : Type) where
  child : Option C
  defaultAcc : Option (Env Z)
  errors : List E
  groups : List G
  vectorized : Option A

def AggSt.new {C E G A : Type} (child : C) (aggs : List (Ex Z)) : Option (AggSt C E G A Z) :=
  match aggs.foldlM (fun env t => setZero t env) (fun _ => none) with
  | some acc => some { child := some child, defaultAcc := some acc, errors := [], groups := [],
                       vectorized := none }
  | none => none

theorem foldlM_setZero (aggs : List (Ex Z)) (env : Env Z) :
    aggs.foldlM (fun env t => setZero t env) env = (outerL aggs).foldlM stepZ env := by
  rw [← setZeroL_eq]
  induction aggs generalizing env with
  | nil => rfl
  | cons t ts ih =>
    simp only [List.foldlM_cons, setZeroL]
    cases setZero t env with
    | none => rfl
    | some e => exact ih e

/-- The built operator has not consumed its child, has no errors or groups, has not run the
analysis, and its default accumulator row is exactly the zeroing of every outermost aggregate
of every aggregation expression. -/
theorem aggNew_spec {C E G A : Type} (c : C) (aggs : List (Ex Z)) (s : AggSt C E G A Z)
    (h : AggSt.new c aggs = some s) :
    s.child = some c ∧ s.errors = [] ∧ s.groups = [] ∧ s.vectorized = none ∧
    s.defaultAcc = (outerL aggs).foldlM stepZ (fun _ => none) := by
  unfold AggSt.new at h
  have hfold := foldlM_setZero aggs (fun _ => none)
  split at h
  · rename_i acc hacc; cases h; rw [← hfold, hacc]; exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  · cases h

end FalkorAggOps.AggPlan
