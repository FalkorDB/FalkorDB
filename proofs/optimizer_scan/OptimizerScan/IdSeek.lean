/-
# Id seeks: `utilize_node_by_id` + `Runtime::evaluate_id_filter`

| here | there |
| --- | --- |
| `Op`, `Op.flip`            | `ExprIR::{Eq,Gt,Ge,Lt,Le}` and the flip table, `planner/optimizer/utilize_node_by_id.rs` |
| `getIdFilter`              | `get_id_filter`, `utilize_node_by_id.rs` |
| `collectFilters`           | the AND walk of `utilize_node_by_id`, `utilize_node_by_id.rs` |
| `V.float c f`              | a `Value::Float` bound, by its `(first_id_ge(f), last_id_le(f))`, `runtime/runtime.rs` |
| `firstIdWhere`             | `first_id_where`, the binary search over `i64::MIN..=i64::MAX + 1`, `runtime/runtime.rs` |
| `bounds`                   | the `(ceil, floor)` match of `evaluate_id_filter`: `Int`, non-NaN `Float`, else `return Ok(None)` |
| `step`                     | one arm of the `match op` on the `i128` range `[min, max]`, `runtime/runtime.rs` |
| `evalIdFilter`             | the filter loop of `Runtime::evaluate_id_filter` (`if min > max { return Ok(None) }` after each arm), then `insert_range(min as u64..=max as u64)`, `runtime/runtime.rs` |
| `evalIdFilterMain`         | `Runtime::evaluate_id_filter`: the loop, then `node_count() == 0 → Ok(None)`, `runtime.rs` |
| `maxNodeId`                | `Graph::max_node_id`, `graph/graph.rs` (`node_ids.max_id()`, id_space.rs) |
| `idMax`                    | the loop's initial `max = node_id_bound().saturating_sub(1)`, `runtime/runtime.rs` (`Graph::node_id_bound` → `IdSpace::bound`, id_space.rs) |
| `seek`                     | `NodeByIdSeekOp::next`: range minus `deleted_nodes`, `runtime/ops/node_by_id_seek.rs` |
| `G.wf`                     | `Graph::node_count` is the number of live ids below the id bound (`id_space.rs`) |
| `reference`                | `Filter(id(n) op v)` over `AllNodeScan` — Cypher semantics, what the rewrite must preserve |
-/
namespace OptimizerScan.IdSeek

inductive Op | eq | gt | ge | lt | le
  deriving DecidableEq, Repr

/-- The flip applied when `id(n)` is on the right: `v op id(n)` becomes `id(n) (flip op) v`
(`utilize_node_by_id.rs`). -/
def Op.flip : Op → Op
  | .eq => .eq | .gt => .lt | .ge => .le | .lt => .gt | .le => .ge

/-- The comparison `x op y` on integers (Cypher, both sides integers). -/
def Op.holds : Op → Int → Int → Bool
  | .eq, x, y => decide (x = y)
  | .gt, x, y => decide (x > y)
  | .ge, x, y => decide (x ≥ y)
  | .lt, x, y => decide (x < y)
  | .le, x, y => decide (x ≤ y)

/-- **PROVEN**: the flip table is right — `y op x ↔ x (flip op) y`. -/
theorem flip_correct (op : Op) (x y : Int) : op.holds y x = op.flip.holds x y := by
  cases op <;> simp [Op.holds, Op.flip, eq_comm]

theorem flip_flip (op : Op) : op.flip.flip = op := by cases op <;> rfl

/-- The operand of an id comparison, after evaluation (since PR #3017 every value is accepted).
A float is represented by where it falls among the `f64` images of the ids: `float c f` is the
float `v` with `c = first_id_ge(v)`, the least `i` with `(i as f64) >= v`, and
`f = last_id_le(v)`, the greatest `i` with `(i as f64) <= v` (`runtime.rs`; their search is
`firstIdWhere`). `nan` is `Value::Float(NaN)`; `null` and `str` stand for every other value. -/
inductive V | int (i : Int) | float (c f : Int) | nan | null | str (s : String)
  deriving DecidableEq, Repr

/-- Cypher's `id(n) op v` for an integer id `x`, as a WHERE predicate (null/false both drop).
Int/Int compare exactly; Int/Float compare `(x as f64)` with the float (`compare_value`,
`value.rs`). As `i ↦ i as f64` is monotone, `(x as f64) >= v ↔ c ≤ x` and
`(x as f64) <= v ↔ x ≤ f`, so `>` is `x > f` and `<` is `x < c`. NaN, null and every other
type never compare true. -/
def cypherHolds (op : Op) (x : Int) : V → Bool
  | .int i => op.holds x i
  | .float c f => match op with
    | .eq => decide (c ≤ x ∧ x ≤ f)
    | .gt => decide (x > f)
    | .ge => decide (x ≥ c)
    | .lt => decide (x < c)
    | .le => decide (x ≤ f)
  | .nan => false
  | .null => false
  | .str _ => false

/-! ### The optimizer side: which filters become an id seek -/

/-- A variable is its `(id, scope_id)` pair (`references.rs` module doc). -/
abbrev Var := Nat × Nat

/-- Minimal expression shapes that matter to `get_id_filter`. -/
inductive E
  | idOf (var : Var)          -- `id(var)` — FuncInvocation "id" over Variable
  | lit (v : V)                -- anything without a Variable
  | refs (var : Var)           -- any expression mentioning `var`
  deriving DecidableEq, Repr

/-- `subtree_references_variable(expr, id, scope)` (`references.rs`, proven in
    `proofs/optimizer_rewrites`: `References.subtreeRefs_iff`): some `Variable` node is `(id, scope)`. -/
def E.references (x : Var) : E → Bool
  | .idOf v => v == x
  | .lit _ => false
  | .refs v => v == x

structure Cmp where
  op : Op
  lhs : E
  rhs : E
  deriving DecidableEq, Repr

/-- `get_id_filter` (`utilize_node_by_id.rs`). Since #2390 the `id()` argument must be the alias
    by `(id, scope_id)` (lines 62-64, 76-78) and the other side must not reference it by
    `(id, scope_id)`; before, both tests compared `Variable`s with `==`, which compares ids only. -/
def getIdFilter (alias : Var) (c : Cmp) : Option (E × Op) :=
  match c.lhs with
  | .idOf v =>
      if v == alias && !c.rhs.references alias then some (c.rhs, c.op)
      else match c.rhs with
        | .idOf w => if w == alias && !c.lhs.references alias then some (c.lhs, c.op.flip) else none
        | _ => none
  | _ => match c.rhs with
    | .idOf w => if w == alias && !c.lhs.references alias then some (c.lhs, c.op.flip) else none
    | _ => none

/-- **PROVEN** (`get_id_filter`): a hit is `id(alias) op e` read as is, or `e op id(alias)` with the
    operator flipped (`flip_correct`), and the value side never mentions the alias. -/
theorem getIdFilter_spec (alias : Var) (c : Cmp) (e : E) (op : Op) (h : getIdFilter alias c = some (e, op)) :
    (c.lhs = .idOf alias ∧ e = c.rhs ∧ op = c.op ∨ c.rhs = .idOf alias ∧ e = c.lhs ∧ op = c.op.flip) ∧
    e.references alias = false := by
  obtain ⟨cop, l, r⟩ := c
  cases l <;> cases r <;> simp only [getIdFilter] at h <;>
    (try (split at h <;> (try split at h))) <;>
    (try simp only [Option.some.injEq, Prod.mk.injEq] at h) <;> (try obtain ⟨rfl, rfl⟩ := h) <;>
    simp_all [E.references]

/-- **PROVEN** (#2390 scope check): `id(v)` over a variable with the alias's id but another scope is
    not an id filter for the alias. -/
theorem getIdFilter_other_scope (i s s' : Nat) (hs : s ≠ s') (op : Op) (v : V) :
    getIdFilter (i, s) ⟨op, .idOf (i, s'), .lit v⟩ = none ∧
    getIdFilter (i, s) ⟨op, .lit v, .idOf (i, s')⟩ = none := by
  simp [getIdFilter, E.references, Ne.symm hs]

/-- The AND walk: every conjunct must be an id comparison, else nothing
(`utilize_node_by_id.rs`). -/
def collectFilters (alias : Var) : List Cmp → List (E × Op)
  | cs => match cs.mapM (getIdFilter alias) with
    | some fs => fs
    | none => []

/-! ### `first_id_where`: the binary search behind a float bound -/

/-- `first_id_where` (`runtime.rs`) over `[lo, hi]`; `p i` is `reached(i as f64)`. The `i128`
midpoint `lo + (hi - lo) / 2` divides a non-negative number, so Rust's truncating `/` is `Int`'s. -/
def firstIdWhere (p : Int → Bool) (lo hi : Int) : Int :=
  if lo < hi then
    let mid := lo + (hi - lo) / 2
    if p mid then firstIdWhere p lo mid else firstIdWhere p (mid + 1) hi
  else lo
termination_by (hi - lo).toNat
decreasing_by all_goals omega

/-- The bounds of the Rust search: `i64::MIN ..= i64::MAX + 1`. -/
def firstIdWhereRust (p : Int → Bool) : Int := firstIdWhere p (-2 ^ 63) (2 ^ 63)

/-- **PROVEN** (`first_id_where`): for a monotone `p` the search returns the least point of
`[lo, hi)` where `p` holds, or `hi` when there is none. -/
theorem firstIdWhere_spec (p : Int → Bool) (hmono : ∀ x y, x ≤ y → p x = true → p y = true) :
    ∀ (n : Nat) (lo hi : Int), (hi - lo).toNat = n → lo ≤ hi →
      lo ≤ firstIdWhere p lo hi ∧ firstIdWhere p lo hi ≤ hi ∧
      (∀ x, lo ≤ x → x < firstIdWhere p lo hi → p x = false) ∧
      (firstIdWhere p lo hi < hi → p (firstIdWhere p lo hi) = true) := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro lo hi hn hle
    rw [firstIdWhere]
    by_cases hlt : lo < hi
    · simp only [hlt, ite_true]
      have hm1 : lo ≤ lo + (hi - lo) / 2 := by omega
      have hm2 : lo + (hi - lo) / 2 < hi := by omega
      by_cases hp : p (lo + (hi - lo) / 2) = true
      · simp only [hp, ite_true]
        obtain ⟨h1, h2, h3, h4⟩ := ih _ (by omega) lo (lo + (hi - lo) / 2) rfl hm1
        refine ⟨h1, by omega, h3, fun _ => ?_⟩
        by_cases he : firstIdWhere p lo (lo + (hi - lo) / 2) < lo + (hi - lo) / 2
        · exact h4 he
        · have : firstIdWhere p lo (lo + (hi - lo) / 2) = lo + (hi - lo) / 2 := by omega
          rw [this]; exact hp
      · simp only [hp, ite_false, Bool.false_eq_true]
        obtain ⟨h1, h2, h3, h4⟩ := ih _ (by omega) (lo + (hi - lo) / 2 + 1) hi rfl (by omega)
        refine ⟨by omega, h2, fun x hx1 hx2 => ?_, h4⟩
        by_cases hxm : x ≤ lo + (hi - lo) / 2
        · cases hpx : p x with
          | false => rfl
          | true => exact absurd (hmono _ _ hxm hpx) hp
        · exact h3 x (by omega) hx2
    · simp only [hlt, ite_false]
      refine ⟨Int.le_refl _, hle, fun x h1 h2 => by exfalso; omega, fun h => by exfalso; omega⟩

/-- **PROVEN** (`first_id_where`, as called): for a monotone `reached ∘ (· as f64)` the result
`r` splits `[i64::MIN, i64::MAX]` exactly: `p x ↔ r ≤ x`, and `r = i64::MAX + 1` when `p` never
holds there. -/
theorem firstIdWhereRust_iff (p : Int → Bool) (hmono : ∀ x y, x ≤ y → p x = true → p y = true)
    (x : Int) (h1 : -2 ^ 63 ≤ x) (h2 : x < 2 ^ 63) :
    p x = true ↔ firstIdWhereRust p ≤ x := by
  obtain ⟨_, _, h3, h4⟩ := firstIdWhere_spec p hmono _ (-2 ^ 63) (2 ^ 63) rfl (by omega)
  unfold firstIdWhereRust
  constructor
  · intro hp
    false_or_by_contra
    rename_i hne
    have := h3 x h1 (by omega)
    rw [hp] at this; exact Bool.noConfusion this
  · intro hr
    exact hmono _ _ hr (h4 (by omega))

/-! ### The runtime side (PR #3017) -/

/-- The `(ceil, floor)` match of `evaluate_id_filter` (`runtime.rs`): an `Int` is its own
ceiling and floor, a non-NaN `Float` is `(first_id_ge, last_id_le)`; anything else is
`return Ok(None)`. -/
def bounds : V → Option (Int × Int)
  | .int i => some (i, i)
  | .float c f => some (c, f)
  | _ => none

/-- `x op v` read through `v`'s `(ceil, floor)`. -/
def holdsCF : Op → Int → Int × Int → Bool
  | .eq, x, (c, f) => decide (c ≤ x ∧ x ≤ f)
  | .gt, x, (_, f) => decide (x > f)
  | .ge, x, (c, _) => decide (x ≥ c)
  | .lt, x, (c, _) => decide (x < c)
  | .le, x, (_, f) => decide (x ≤ f)

/-- The `(ceil, floor)` pair is exact: a value with no pair never compares true, and every other
comparison is `holdsCF` on its pair. -/
theorem cypherHolds_bounds (op : Op) (x : Int) (v : V) :
    cypherHolds op x v = match bounds v with
      | none => false
      | some cf => holdsCF op x cf := by
  cases v <;> cases op <;> simp [cypherHolds, bounds, holdsCF, Op.holds]
  rw [Bool.eq_iff_iff]; simp; omega

/-- One arm of `evaluate_id_filter`'s `match op` on the running `i128` range `[min, max]`
(`runtime.rs`). The `saturating_add(1)` / `saturating_sub(1)` never saturate: a bound is an
`i64` or lies in `[i64::MIN - 1, i64::MAX + 1]`, far inside `i128`. -/
def step : Int × Int → Op × (Int × Int) → Int × Int
  | (mn, mx), (.eq, (c, f)) => (max mn c, min mx f)
  | (mn, mx), (.gt, (_, f)) => (max mn (f + 1), mx)
  | (mn, mx), (.ge, (c, _)) => (max mn c, mx)
  | (mn, mx), (.lt, (c, _)) => (mn, min mx (c - 1))
  | (mn, mx), (.le, (_, f)) => (mn, min mx f)

inductive Res | empty | range (lo hi : Nat)
  deriving DecidableEq, Repr

/-- `Runtime::evaluate_id_filter`'s loop (`runtime.rs`), filters evaluated in order: a value
with no `(ceil, floor)` returns `Ok(None)`, and so does a step that leaves `min > max`; at the end
`min as u64 ..= max as u64` is the seek range. -/
def evalIdFilter (maxId : Nat) (fs : List (V × Op)) : Res :=
  go (0, maxId) fs
where
  go : Int × Int → List (V × Op) → Res
    | (a, b), [] => .range a.toNat b.toNat
    | r, (v, op) :: rest =>
      match bounds v with
      | none => .empty
      | some cf =>
        let r' := step r (op, cf)
        if r'.1 > r'.2 then .empty else go r' rest

/-- Graph model for id seeks. Ids are dense: every id `< bound` is live unless deleted
(`Graph::node_count + deleted_nodes.len()` is the id bound; `id_space.rs`). -/
structure G where
  nodeCount : Nat
  deleted : List Nat
  deriving Repr

def G.bound (g : G) : Nat := g.nodeCount + g.deleted.length
def G.live (g : G) (x : Nat) : Bool := decide (x < g.bound) && !g.deleted.contains x

/-- `Graph::max_node_id` (`graph.rs` → `IdSpace::max_id`, id_space.rs). -/
def maxNodeId (g : G) : Nat := if g.nodeCount = 0 then 0 else g.bound - 1

/-- `Runtime::evaluate_id_filter` (#3394, `runtime.rs`): the same loop, then
`if self.graph().node_count() == 0 { return Ok(None) }` before `insert_range`.
The loop's early `Ok(None)` comes first, so only a `range` result is affected. -/
def evalIdFilterMain (nodeCount maxId : Nat) (fs : List (V × Op)) : Res :=
  match evalIdFilter maxId fs with
  | .range a b => if nodeCount = 0 then .empty else .range a b
  | r => r

/-- PR #2979: the loop's initial upper bound, `node_id_bound().saturating_sub(1)`
(`Nat` subtraction saturates at 0). -/
def idMax (g : G) : Nat := g.bound - 1

/-- On a graph with nodes, `idMax` is `max_node_id`; they differ only when `node_count = 0`. -/
theorem idMax_eq_maxNodeId (g : G) (hne : g.nodeCount ≠ 0) : idMax g = maxNodeId g := by
  simp [idMax, maxNodeId, hne]

/-- `NodeByIdSeekOp`: every id of the range that is not deleted (no liveness check
beyond `deleted_nodes`, `node_by_id_seek.rs`). Membership test; `none` would be a query error,
which the loop no longer raises for any value (PR #3017).
The filter loop starts at `[0, idMax g]` (PR #2979). -/
def seek (g : G) (fs : List (V × Op)) (x : Nat) : Option Bool :=
  match evalIdFilterMain g.nodeCount (idMax g) fs with
  | .empty => some false
  | .range a b => some (decide (a ≤ x ∧ x ≤ b) && !g.deleted.contains x)

/-- On a graph with nodes the #3394 check is a no-op. -/
theorem seek_nonempty (g : G) (fs : List (V × Op)) (hne : g.nodeCount ≠ 0) (x : Nat) :
    seek g fs x = (match evalIdFilter (maxNodeId g) fs with
      | .empty => some false
      | .range a b => some (decide (a ≤ x ∧ x ≤ b) && !g.deleted.contains x)) := by
  unfold seek evalIdFilterMain
  rw [idMax_eq_maxNodeId g hne]
  cases evalIdFilter (maxNodeId g) fs <;> simp [hne]

/-- Every conjunct holds. -/
def sat (fs : List (V × Op)) (x : Int) : Bool := fs.all (fun p => cypherHolds p.2 x p.1)

/-- The reference: `Filter(AND of id(n) op v)` over `AllNodeScan`. -/
def reference (g : G) (fs : List (V × Op)) (x : Nat) : Bool :=
  g.live x && sat fs x

/-- Well-formedness: `node_count` counts the live ids below the id bound. -/
def G.wf (g : G) : Prop := g.nodeCount = ((List.range g.bound).filter g.live).length

/-- A well-formed graph with `node_count = 0` has no live id. -/
theorem live_false_of_zero (g : G) (hwf : g.wf) (h0 : g.nodeCount = 0) (x : Nat) :
    g.live x = false := by
  cases hl : g.live x with
  | false => rfl
  | true =>
    have hx : x < g.bound := by
      simp only [G.live, Bool.and_eq_true, decide_eq_true_eq] at hl; exact hl.1
    have hm : x ∈ (List.range g.bound).filter g.live :=
      List.mem_filter.2 ⟨List.mem_range.2 hx, hl⟩
    unfold G.wf at hwf
    rw [h0] at hwf
    have := List.eq_nil_of_length_eq_zero hwf.symm
    rw [this] at hm; simp at hm

/-! ### Soundness of the range folding -/

/-- **PROVEN** (`step` is exact): the new range holds exactly the ids of `[mn,mx]` that
satisfy the comparison. -/
theorem step_some (mn mx a b : Int) (op : Op) (cf : Int × Int) (h : step (mn, mx) (op, cf) = (a, b)) :
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ holdsCF op x cf = true) := by
  intro x
  obtain ⟨c, f⟩ := cf
  cases op <;> simp only [step, Prod.mk.injEq] at h <;> obtain ⟨rfl, rfl⟩ := h <;>
    simp [holdsCF] <;> omega

/-- **PROVEN** (the `if min > max { return Ok(None) }` is exact): an inverted range means no id
of `[mn,mx]` satisfies the comparison. -/
theorem step_none (mn mx a b : Int) (op : Op) (cf : Int × Int) (h : step (mn, mx) (op, cf) = (a, b))
    (hgt : a > b) : ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ holdsCF op x cf = true) := by
  intro x hx
  have := (step_some mn mx a b op cf h x).mpr hx
  omega

/-- **PROVEN** (the range only narrows): each arm keeps `mn ≤ min` and `max ≤ mx`; with the
`min > max` check this is what keeps `0 ≤ min ≤ max < node_id_bound` for the `as u64` casts. -/
theorem step_nonempty (mn mx a b : Int) (op : Op) (cf : Int × Int)
    (h : step (mn, mx) (op, cf) = (a, b)) : mn ≤ a ∧ b ≤ mx := by
  obtain ⟨c, f⟩ := cf
  cases op <;> simp only [step, Prod.mk.injEq] at h <;> obtain ⟨rfl, rfl⟩ := h <;> omega

/-- The loop over `Int` ranges; `none` is `Ok(None)`. -/
def fold : Int × Int → List (V × Op) → Option (Int × Int)
  | r, [] => some r
  | r, (v, op) :: rest => match bounds v with
    | none => none
    | some cf =>
      let r' := step r (op, cf)
      if r'.1 > r'.2 then none else fold r' rest

/-- **PROVEN**: an empty fold means no id in the initial range satisfies every comparison. -/
theorem fold_none : ∀ (fs : List (V × Op)) (mn mx : Int), fold (mn, mx) fs = none →
    ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ sat fs x = true)
  | [], _, _, h => by simp [fold] at h
  | (v, op) :: rest, mn, mx, h => by
    intro x ⟨h1, h2, h3⟩
    simp only [sat, List.all_cons, Bool.and_eq_true] at h3
    rw [cypherHolds_bounds] at h3
    simp only [fold] at h
    split at h
    · next heq => simp [heq] at h3
    · next cf heq =>
      try simp only [heq] at h3
      rcases hs : step (mn, mx) (op, cf) with ⟨a, b⟩
      rw [hs] at h
      simp only at h
      split at h
      · next hgt => exact step_none mn mx a b op cf hs hgt x ⟨h1, h2, h3.1⟩
      · have := (step_some mn mx a b op cf hs x).mpr ⟨h1, h2, h3.1⟩
        exact fold_none rest a b h x ⟨this.1, this.2, h3.2⟩

/-- **PROVEN**: the fold is exact — `x` is in the final range iff it is in the initial one
and satisfies every comparison. -/
theorem fold_some : ∀ (fs : List (V × Op)) (mn mx a b : Int), fold (mn, mx) fs = some (a, b) →
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ sat fs x = true)
  | [], mn, mx, a, b, h => by
    simp [fold] at h; obtain ⟨rfl, rfl⟩ := h; intro x; simp [sat]
  | (v, op) :: rest, mn, mx, a, b, h => by
    intro x
    simp only [sat, List.all_cons, Bool.and_eq_true]
    rw [cypherHolds_bounds]
    simp only [fold] at h
    split at h
    · simp at h
    · next cf heq =>
      try simp only [heq]
      rcases hs' : step (mn, mx) (op, cf) with ⟨a', b'⟩
      rw [hs'] at h
      simp only at h
      split at h
      · simp at h
      · have hs := step_some mn mx a' b' op cf hs' x
        have ih := fold_some rest a' b' a b h x
        simp only [sat] at ih
        rw [ih]
        constructor
        · rintro ⟨h1, h2, h4⟩
          have := hs.mp ⟨h1, h2⟩
          exact ⟨this.1, this.2.1, this.2.2, h4⟩
        · rintro ⟨h1, h2, h3, h4⟩
          have := hs.mpr ⟨h1, h2, h3⟩
          exact ⟨this.1, this.2, h4⟩

/-- **PROVEN** (the `as u64` casts are exact): a non-empty fold from a non-empty range ends in a
non-empty sub-range, so `insert_range(min as u64..=max as u64)` never sees `min > max` and,
from `[0, idMax]`, both ends are in `[0, node_id_bound)`. -/
theorem fold_bounds : ∀ (fs : List (V × Op)) (mn mx a b : Int), mn ≤ mx →
    fold (mn, mx) fs = some (a, b) → mn ≤ a ∧ a ≤ b ∧ b ≤ mx
  | [], mn, mx, a, b, hle, h => by
    simp [fold] at h; obtain ⟨rfl, rfl⟩ := h; omega
  | (v, op) :: rest, mn, mx, a, b, _, h => by
    simp only [fold] at h
    split at h
    · simp at h
    · next cf _ =>
      rcases hs : step (mn, mx) (op, cf) with ⟨a', b'⟩
      rw [hs] at h
      simp only at h
      split at h
      · simp at h
      · next hng =>
        have := step_nonempty mn mx a' b' op cf hs
        have := fold_bounds rest a' b' a b (by omega) h
        omega

/-- The runtime loop is the fold followed by the `as u64` casts. -/
theorem go_eq_fold : ∀ (fs : List (V × Op)) (r : Int × Int),
    evalIdFilter.go r fs =
      match fold r fs with | none => .empty | some (a, b) => .range a.toNat b.toNat
  | [], (a, b) => by simp [evalIdFilter.go, fold]
  | (v, op) :: rest, r => by
    simp only [evalIdFilter.go, fold]
    split
    · rfl
    · split
      · rfl
      · exact go_eq_fold rest _

/-- For a non-empty graph `NodeByIdSeek` returns exactly the nodes `Filter(id(n) op v …)` over
`AllNodeScan` would, for every operand (the non-empty half of `seek_sound`). -/
theorem seek_sound_nonempty (g : G) (fs : List (V × Op)) (hne : g.nodeCount ≠ 0) (x : Nat) :
    seek g fs x = some (reference g fs x) := by
  have hb : 0 < g.bound := by unfold G.bound; omega
  have hmax : (idMax g : Int) = (g.bound : Int) - 1 := by unfold idMax; omega
  unfold seek evalIdFilterMain evalIdFilter reference G.live
  rw [go_eq_fold]
  cases hf : fold (0, (idMax g : Int)) fs with
  | none =>
    have hn := fold_none fs 0 _ hf x
    simp only [Option.some.injEq]
    cases hs : sat fs x
    · simp
    · have : ¬ (x < g.bound) := fun hx => hn ⟨by omega, by omega, hs⟩
      simp [this]
  | some r =>
    obtain ⟨a, b⟩ := r
    have hab := fold_bounds fs 0 _ a b (by omega) hf
    have hiff := fold_some fs 0 _ a b hf x
    simp only [hne, ite_false, Option.some.injEq]
    have hr : (a.toNat ≤ x ∧ x ≤ b.toNat) ↔ (x < g.bound ∧ sat fs x = true) := by
      constructor
      · intro h
        have := hiff.mp ⟨by omega, by omega⟩
        exact ⟨by omega, this.2.2⟩
      · intro h
        have := hiff.mpr ⟨by omega, by omega, h.2⟩
        omega
    cases hs : sat fs x <;> cases hc : g.deleted.contains x <;> simp_all

/-- **PROVEN** (headline, id-seek soundness, PR #3017): on any well-formed graph — including one
that never held a node — and for every operand (negative or huge integers, floats, NaN, null,
strings), `NodeByIdSeek` returns exactly the nodes `Filter(id(n) op v …)` over `AllNodeScan` would.
PR #3017 removed the `0 ≤ v < 2^63` integer hypothesis (fixes W2-rewrites-2 / W2-scan-*);
#3394 removed the `nodeCount ≠ 0` hypothesis (closes #3353 / W2-scan-2). -/
theorem seek_sound (g : G) (hwf : g.wf) (fs : List (V × Op)) (x : Nat) :
    seek g fs x = some (reference g fs x) := by
  by_cases hne : g.nodeCount = 0
  · have hl := live_false_of_zero g hwf hne x
    unfold seek evalIdFilterMain evalIdFilter reference
    rw [go_eq_fold, hl]
    cases fold (0, (idMax g : Int)) fs with
    | none => rfl
    | some r => obtain ⟨a, b⟩ := r; simp [hne]
  · exact seek_sound_nonempty g fs hne x

/-! ### The bounds PR #3017 fixed, on five live nodes 0..4 -/

/-- Five live nodes 0..4. -/
def g5 : G := { nodeCount := 5, deleted := [] }

/-- `id(n) > -1` keeps every node (it returned none while the bound was cast `as u64`). -/
theorem id_gt_neg_all : evalIdFilter (maxNodeId g5) [(.int (-1), .gt)] = .range 0 4 := by decide

/-- `id(n) <= -1` and `id(n) < -1` keep none (they returned every node). -/
theorem id_le_neg_none : evalIdFilter (maxNodeId g5) [(.int (-1), .le)] = .empty ∧
    evalIdFilter (maxNodeId g5) [(.int (-1), .lt)] = .empty := by decide

/-- `id(n) = 1.0` is node 1, `id(n) > 1.5` (`first_id_ge = 2`, `last_id_le = 1`) starts at 2,
`id(n) = 1.5` is empty, and `id(n) = null` / `= NaN` / `= '1'` keep none (they all raised
"Node ID must be an integer"). -/
theorem id_nonint_bounds :
    evalIdFilter (maxNodeId g5) [(.float 1 1, .eq)] = .range 1 1 ∧
    evalIdFilter (maxNodeId g5) [(.float 2 1, .gt)] = .range 2 4 ∧
    evalIdFilter (maxNodeId g5) [(.float 2 1, .eq)] = .empty ∧
    evalIdFilter (maxNodeId g5) [(.null, .eq)] = .empty ∧
    evalIdFilter (maxNodeId g5) [(.nan, .eq)] = .empty ∧
    evalIdFilter (maxNodeId g5) [(.str "1", .eq)] = .empty := by decide

/-- A graph that has never held a node (W2-scan-2, #3353). #3394 made `NodeByIdSeek` empty on it (it once yielded the phantom node 0). -/
def gEmpty : G := { nodeCount := 0, deleted := [] }
theorem gEmpty_wf : gEmpty.wf := by simp [G.wf, gEmpty, G.bound]
example : reference gEmpty [(.int 0, .eq)] 0 = false := by decide

/-- **PROVEN** (#3394): the seek on a never-populated graph is empty, matching the
reference (a special case of `seek_sound`). -/
theorem empty_graph_seek_none (fs : List (V × Op)) (x : Nat) : seek gEmpty fs x ≠ some true := by
  unfold seek evalIdFilterMain
  cases evalIdFilter (idMax gEmpty) fs <;> simp [gEmpty]
example : seek gEmpty [(.int 0, .eq)] 0 = some false := by decide

example : seek { nodeCount := 0, deleted := [0, 1] } [(.int 0, .eq)] 0 = some false := by decide

/-- On an emptied graph (ids 0, 1 handed out, both deleted) `id(n) = 1 AND id(n) = 'x'` seeks
nothing; it raised "Node ID must be an integer" before PR #3017. -/
example : seek { nodeCount := 0, deleted := [0, 1] } [(.int 1, .eq), (.str "x", .eq)] 0 = some false := by
  decide

end OptimizerScan.IdSeek

