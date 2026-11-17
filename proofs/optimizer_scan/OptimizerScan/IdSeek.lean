/-
# Id seeks: `utilize_node_by_id` + `Runtime::evaluate_id_filter`

Re-targeted to PR #3017 (6e3b66969): `evaluate_id_filter` now narrows `i128` bounds, turns a
float into its ceil/floor id (`first_id_ge`/`last_id_le`, a binary search over the `f64` cast
beyond 2^53) and returns `Ok(None)` for a null, NaN or non-numeric bound instead of the
"Node ID must be an integer" error; a negative bound is no longer reinterpreted as `u64`.

| here | there |
| --- | --- |
| `Op`, `Op.flip`            | `ExprIR::{Eq,Gt,Ge,Lt,Le}` and the flip table, `planner/optimizer/utilize_node_by_id.rs:80-87` |
| `getIdFilter`              | `get_id_filter`, `utilize_node_by_id.rs:52-92` |
| `collectFilters`           | the AND walk of `utilize_node_by_id`, `utilize_node_by_id.rs:106-119` |
| `Fl`, `Fl.geOf`, `Fl.gtOf` | a non-NaN `f64` (finite `n/(d+1)` or `±inf`) and `x >= f`, `x > f` for an integer-valued `x` |
| `rne`, `castF64`           | Rust's `i128 as f64` (round to nearest, ties to even); laws `cast_exact`, `cast_large`, `cast_neg_large` |
| `search`, `firstIdWhere`   | `first_id_where`, `runtime/runtime.rs:437-449` |
| `firstIdGe`, `lastIdLe`    | `first_id_ge` (`runtime.rs:451-456`), `last_id_le` (`runtime.rs:459-464`) |
| `bounds`                   | the `(ceil, floor)` match, `runtime.rs:1491-1495` (`_ => return Ok(None)`) |
| `step`                     | one arm of the `match op` plus the `min > max` check, `runtime.rs:1496-1512` |
| `evalIdFilter`             | the filter loop of `Runtime::evaluate_id_filter`, `runtime.rs:1473-1512` |
| `evalIdFilterMain`         | `Runtime::evaluate_id_filter`: the loop, then `node_count() == 0 → Ok(None)`, `runtime.rs:1516-1522` |
| `maxNodeId`                | `Graph::max_node_id`, `graph/graph.rs:1783-1786` (since #2846 `node_ids.max_id()`, id_space.rs:353 — same arithmetic) |
| `seek`                     | `NodeByIdSeekOp::next`: range minus `deleted_nodes`, `runtime/ops/node_by_id_seek.rs:63-73` |
| `pre3394_seek`             | the same without the #3394 `node_count` check (b6567cb97) |
| `G.wf`                     | `Graph::node_count` is the number of live ids below the id bound (`id_space.rs`) |
| `labelIdScan`              | `NodeByLabelAndIdScanOp::next`: `get_nodes(labels, min)` ∩ range, `runtime/ops/node_by_label_and_id_scan.rs:69-82` |
| `reference`                | `Filter(id(n) op v)` over `AllNodeScan` — Cypher semantics, what the rewrite must preserve |
-/
namespace OptimizerScan.IdSeek

inductive Op | eq | gt | ge | lt | le
  deriving DecidableEq, Repr

/-- The flip applied when `id(n)` is on the right: `v op id(n)` becomes `id(n) (flip op) v`
(`utilize_node_by_id.rs:80-87`). -/
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

/-- A non-NaN `f64`: a finite value `n / (d + 1)` (every finite `f64` is one) or an infinity. -/
inductive Fl | fin (n : Int) (d : Nat) | posInf | negInf
  deriving DecidableEq, Repr

/-- `(a : f64) >= f` for an integer-valued `a`. -/
def Fl.geOf (a : Int) : Fl → Bool
  | .fin n d => decide (n ≤ a * ((d : Int) + 1))
  | .posInf => false
  | .negInf => true

/-- `(a : f64) > f` for an integer-valued `a`. -/
def Fl.gtOf (a : Int) : Fl → Bool
  | .fin n d => decide (n < a * ((d : Int) + 1))
  | .posInf => false
  | .negInf => true

/-- The operand of an id comparison, after evaluation: an `i64`, a non-NaN float, NaN, null or
any other (non-numeric) value. -/
inductive V | int (i : Int) | float (f : Fl) | nan | null | str (s : String)
  deriving DecidableEq, Repr

/-- Cypher's `id(n) op v` for an integer id `x`, as a WHERE predicate (null/false both drop).
Int/Float compare numerically (the id is cast to `f64`, exact below 2^53); NaN, null and
non-numeric values never compare true. -/
def cypherHolds (op : Op) (x : Nat) : V → Bool
  | .int i => op.holds x i
  | .float f => match op with
    | .eq => Fl.geOf x f && !Fl.gtOf x f
    | .gt => Fl.gtOf x f
    | .ge => Fl.geOf x f
    | .lt => !Fl.geOf x f
    | .le => !Fl.gtOf x f
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

/-- `subtree_references_variable(expr, id, scope)` (`references.rs:327`, proven in
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

/-- `get_id_filter` (`utilize_node_by_id.rs:52-92`). Since #2390 the `id()` argument must be the alias
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
(`utilize_node_by_id.rs:106-119`). -/
def collectFilters (alias : Var) : List Cmp → List (E × Op)
  | cs => match cs.mapM (getIdFilter alias) with
    | some fs => fs
    | none => []

/-! ### The runtime side -/

/-! #### `i128 as f64` -/

/-- Round a natural number to the nearest `f64` (53-bit significand, ties to even). Below 2^53
every integer is exact; above, keep the top 53 bits of `m` and round the dropped `k` bits. -/
def rne (m : Nat) : Nat :=
  if m < 2 ^ 53 then m else
    (if 2 ^ (m.log2 - 52) / 2 < m % 2 ^ (m.log2 - 52) ∨
        (m % 2 ^ (m.log2 - 52) = 2 ^ (m.log2 - 52) / 2 ∧ (m / 2 ^ (m.log2 - 52)) % 2 = 1)
      then m / 2 ^ (m.log2 - 52) + 1 else m / 2 ^ (m.log2 - 52)) * 2 ^ (m.log2 - 52)

/-- `i as f64` for an `i128` (the result of casting an integer is an integer; rounding is
symmetric in the sign). -/
def castF64 (i : Int) : Int := if 0 ≤ i then (rne i.toNat : Int) else -((rne (-i).toNat : Nat) : Int)

theorem rne_small (m : Nat) (h : m < 2 ^ 53) : rne m = m := by simp [rne, h]

theorem rne_large (m : Nat) (h : 2 ^ 53 ≤ m) : 2 ^ 53 ≤ rne m := by
  have hm : m ≠ 0 := by omega
  have hl : 53 ≤ m.log2 := (Nat.le_log2 hm).2 h
  have hp : 2 ^ m.log2 ≤ m := Nat.log2_self_le hm
  have hsplit : 2 ^ m.log2 = 2 ^ 52 * 2 ^ (m.log2 - 52) := by
    rw [← Nat.pow_add]; congr 1; omega
  have hk : 0 < 2 ^ (m.log2 - 52) := Nat.pow_pos (by decide)
  have hk2 : 2 ≤ 2 ^ (m.log2 - 52) := by
    have : 2 ^ 1 ≤ 2 ^ (m.log2 - 52) := Nat.pow_le_pow_right (by decide) (by omega)
    simpa using this
  have hq : 2 ^ 52 ≤ m / 2 ^ (m.log2 - 52) :=
    (Nat.le_div_iff_mul_le hk).2 (by rw [← hsplit]; exact hp)
  have hlow : 2 ^ 53 ≤ m / 2 ^ (m.log2 - 52) * 2 ^ (m.log2 - 52) := by
    have h1 : 2 ^ 53 = 2 ^ 52 * 2 := by decide
    rw [h1]
    exact Nat.le_trans (Nat.mul_le_mul_left _ hk2) (Nat.mul_le_mul_right _ hq)
  unfold rne
  rw [if_neg (by omega)]
  split
  · exact Nat.le_trans hlow (Nat.mul_le_mul_right _ (Nat.le_succ _))
  · exact hlow

/-- **PROVEN**: the cast is exact strictly inside `±2^53`. -/
theorem cast_exact (i : Int) (h1 : -2 ^ 53 < i) (h2 : i < 2 ^ 53) : castF64 i = i := by
  unfold castF64
  split
  · rw [rne_small _ (by omega)]; omega
  · rw [rne_small _ (by omega)]; omega

/-- **PROVEN**: an integer at or above 2^53 casts to at least 2^53. -/
theorem cast_large (i : Int) (h : 2 ^ 53 ≤ i) : 2 ^ 53 ≤ castF64 i := by
  unfold castF64
  rw [if_pos (by omega)]
  have := rne_large i.toNat (by omega)
  omega

/-- **PROVEN**: an integer at or below -2^53 casts to at most -2^53. -/
theorem cast_neg_large (i : Int) (h : i ≤ -2 ^ 53) : castF64 i ≤ -2 ^ 53 := by
  unfold castF64
  rw [if_neg (by omega)]
  have := rne_large (-i).toNat (by omega)
  omega

theorem cast_lt (i : Int) (h : i < 2 ^ 53) : castF64 i < 2 ^ 53 := by
  by_cases h1 : -2 ^ 53 < i
  · rw [cast_exact i h1 h]; exact h
  · have := cast_neg_large i (by omega); omega

theorem cast_nonneg (i : Int) (h : 0 ≤ i) : 0 ≤ castF64 i := by
  by_cases h2 : i < 2 ^ 53
  · rw [cast_exact i (by omega) h2]; exact h
  · have := cast_large i (by omega); omega

/-! #### `first_id_where`, `first_id_ge`, `last_id_le` -/

/-- The `while lo < hi` binary search of `first_id_where` (`runtime.rs:437-449`); `(hi - lo) / 2`
is non-negative, so `i128`'s truncating division agrees with `Int` division. -/
def search (p : Int → Bool) (lo hi : Int) : Int :=
  if lo < hi then
    if p (lo + (hi - lo) / 2) then search p lo (lo + (hi - lo) / 2)
    else search p (lo + (hi - lo) / 2 + 1) hi
  else lo
termination_by (hi - lo).toNat
decreasing_by all_goals omega

/-- `first_id_where(reached)` over `i64::MIN..=i64::MAX + 1`. -/
def firstIdWhere (p : Int → Bool) : Int := search p (-2 ^ 63) (2 ^ 63)

/-- **PROVEN** (the binary search, for any predicate): the result `r` lies in `[lo, hi]`, is
`hi` or satisfies `p`, and is `lo` or follows an id that does not. -/
theorem search_spec (p : Int → Bool) (lo hi : Int) (h : lo ≤ hi) :
    lo ≤ search p lo hi ∧ search p lo hi ≤ hi ∧
    (search p lo hi = hi ∨ p (search p lo hi) = true) ∧
    (search p lo hi = lo ∨ p (search p lo hi - 1) = false) := by
  rw [search]
  split
  · next hlt =>
    split
    · next hp =>
      have ih := search_spec p lo (lo + (hi - lo) / 2) (by omega)
      obtain ⟨a, b, c, d⟩ := ih
      refine ⟨a, by omega, ?_, d⟩
      rcases c with c | c
      · right; rw [c]; exact hp
      · right; exact c
    · next hp =>
      have ih := search_spec p (lo + (hi - lo) / 2 + 1) hi (by omega)
      obtain ⟨a, b, c, d⟩ := ih
      refine ⟨by omega, b, c, ?_⟩
      rcases d with d | d
      · right; rw [d]; simpa using hp
      · right; exact d
  · exact ⟨Int.le_refl _, h, Or.inl (by omega), Or.inl rfl⟩
termination_by (hi - lo).toNat
decreasing_by all_goals omega

/-- `f.abs() < 9_007_199_254_740_992.0` (2^53). -/
def Fl.small : Fl → Bool
  | .fin n d => decide (n.natAbs < 2 ^ 53 * (d + 1))
  | _ => false

/-- `f.ceil() as i128` for a finite float (only used when `Fl.small`). -/
def Fl.ceil : Fl → Int
  | .fin n d => -((-n) / ((d : Int) + 1))
  | _ => 0

/-- `f.floor() as i128` for a finite float (only used when `Fl.small`). -/
def Fl.floor : Fl → Int
  | .fin n d => n / ((d : Int) + 1)
  | _ => 0

/-- `first_id_ge` (`runtime.rs:451-456`). -/
def firstIdGe (f : Fl) : Int :=
  if f.small then f.ceil else firstIdWhere (fun i => Fl.geOf (castF64 i) f)

/-- `last_id_le` (`runtime.rs:459-464`). -/
def lastIdLe (f : Fl) : Int :=
  if f.small then f.floor else firstIdWhere (fun i => Fl.gtOf (castF64 i) f) - 1

/-- A non-small float lies at or beyond `±2^53`: on every id strictly inside, it compares
the same way. -/
theorem not_small_cases (f : Fl) (h : f.small = false) :
    (∀ y : Int, y < 2 ^ 53 → Fl.geOf y f = false ∧ Fl.gtOf y f = false) ∨
    (∀ y : Int, -2 ^ 53 < y → Fl.geOf y f = true ∧ Fl.gtOf y f = true) := by
  cases f with
  | posInf => left; intro y _; simp [Fl.geOf, Fl.gtOf]
  | negInf => right; intro y _; simp [Fl.geOf, Fl.gtOf]
  | fin n d =>
    simp only [Fl.small, decide_eq_false_iff_not, Nat.not_lt] at h
    have hD : (0 : Int) < (d : Int) + 1 := by omega
    by_cases hn : 0 ≤ n
    · left
      intro y hy
      have h1 : y * ((d : Int) + 1) < 2 ^ 53 * ((d : Int) + 1) := Int.mul_lt_mul_of_pos_right hy hD
      have h2 : (2 : Int) ^ 53 * ((d : Int) + 1) ≤ n := by omega
      simp only [Fl.geOf, Fl.gtOf, decide_eq_false_iff_not]; omega
    · right
      intro y hy
      have h1 : -2 ^ 53 * ((d : Int) + 1) < y * ((d : Int) + 1) := Int.mul_lt_mul_of_pos_right hy hD
      have h2 : n ≤ -2 ^ 53 * ((d : Int) + 1) := by omega
      simp only [Fl.geOf, Fl.gtOf, decide_eq_true_eq]; omega

/-- **PROVEN** (`first_id_ge`): on every id `0 ≤ x < 2^53`, `first_id_ge(f) ≤ x` iff `x >= f`. -/
theorem firstIdGe_spec (f : Fl) (x : Int) (h0 : 0 ≤ x) (h1 : x < 2 ^ 53) :
    (firstIdGe f ≤ x) ↔ Fl.geOf x f = true := by
  unfold firstIdGe
  split
  · next hs =>
    cases f with
    | fin n d =>
      have hD : (0 : Int) < (d : Int) + 1 := by omega
      simp only [Fl.ceil, Fl.geOf, decide_eq_true_eq]
      have := @Int.le_ediv_iff_mul_le (-x) (-n) ((d : Int) + 1) hD
      rw [Int.neg_mul] at this
      omega
    | posInf => simp [Fl.small] at hs
    | negInf => simp [Fl.small] at hs
  · next hs =>
    have hs' : f.small = false := by simpa using hs
    obtain ⟨_, _, hr, hl⟩ := search_spec (fun i => Fl.geOf (castF64 i) f) (-2 ^ 63) (2 ^ 63) (by omega)
    unfold firstIdWhere
    rcases not_small_cases f hs' with hA | hB
    · rw [(hA x h1).1]
      simp only [Bool.false_eq_true, iff_false, Int.not_le]
      generalize search _ _ _ = r at hr hl
      rcases hr with hr | hr
      · omega
      · by_cases hrs : r < 2 ^ 53
        · have := (hA (castF64 r) (cast_lt r hrs)).1; simp_all
        · omega
    · rw [(hB x (by omega)).1]
      simp only [iff_true]
      generalize search _ _ _ = r at hr hl
      rcases hl with hl | hl
      · omega
      · by_cases hrs : 0 < r
        · have := (hB (castF64 (r - 1)) (by have := cast_nonneg (r - 1) (by omega); omega)).1
          simp_all
        · omega

/-- **PROVEN** (`last_id_le`): on every id `0 ≤ x < 2^53`, `x ≤ last_id_le(f)` iff `x <= f`. -/
theorem lastIdLe_spec (f : Fl) (x : Int) (h0 : 0 ≤ x) (h1 : x < 2 ^ 53) :
    (x ≤ lastIdLe f) ↔ Fl.gtOf x f = false := by
  unfold lastIdLe
  split
  · next hs =>
    cases f with
    | fin n d =>
      have hD : (0 : Int) < (d : Int) + 1 := by omega
      simp only [Fl.floor, Fl.gtOf, decide_eq_false_iff_not, Int.not_lt]
      exact Int.le_ediv_iff_mul_le hD
    | posInf => simp [Fl.small] at hs
    | negInf => simp [Fl.small] at hs
  · next hs =>
    have hs' : f.small = false := by simpa using hs
    obtain ⟨_, _, hr, hl⟩ := search_spec (fun i => Fl.gtOf (castF64 i) f) (-2 ^ 63) (2 ^ 63) (by omega)
    unfold firstIdWhere
    rcases not_small_cases f hs' with hA | hB
    · rw [(hA x h1).2]
      simp only [iff_true]
      generalize search _ _ _ = r at hr hl
      rcases hr with hr | hr
      · omega
      · by_cases hrs : r < 2 ^ 53
        · have := (hA (castF64 r) (cast_lt r hrs)).2; simp_all
        · omega
    · rw [(hB x (by omega)).2]
      simp only [Bool.true_eq_false, iff_false, Int.not_le]
      generalize search _ _ _ = r at hr hl
      rcases hl with hl | hl
      · omega
      · by_cases hrs : 0 < r
        · have := (hB (castF64 (r - 1)) (by have := cast_nonneg (r - 1) (by omega); omega)).2
          simp_all
        · omega

/-! #### The loop -/

/-- The `(ceil, floor)` of a bound (`runtime.rs:1491-1495`); `none` is `_ => return Ok(None)`
(null, NaN, any non-numeric value). -/
def bounds : V → Option (Int × Int)
  | .int i => some (i, i)
  | .float f => some (firstIdGe f, lastIdLe f)
  | _ => none

/-- `x op v` read through the bound's `(ceil, floor)`: `= ↔ ceil ≤ x ≤ floor`, `> ↔ floor < x`,
`>= ↔ ceil ≤ x`, `< ↔ x < ceil`, `<= ↔ x ≤ floor`. -/
def Op.holdsB : Op → Int → Int → Int → Bool
  | .eq, x, c, fl => decide (c ≤ x ∧ x ≤ fl)
  | .gt, x, _, fl => decide (fl < x)
  | .ge, x, c, _ => decide (c ≤ x)
  | .lt, x, c, _ => decide (x < c)
  | .le, x, _, fl => decide (x ≤ fl)

/-- One arm of `evaluate_id_filter`'s `match op` on the running `i128` `[min, max]`, then
`if min > max { return Ok(None) }` (`runtime.rs:1496-1512`). The `saturating_add/sub(1)` never
saturate: every bound lies in `[i64::MIN - 1, i64::MAX + 1]`. -/
def step : Int × Int → Op × Int × Int → Option (Int × Int)
  | (mn, mx), (op, c, fl) =>
    let r : Int × Int := match op with
      | .eq => (max mn c, min mx fl)
      | .gt => (max mn (fl + 1), mx)
      | .ge => (max mn c, mx)
      | .lt => (mn, min mx (c - 1))
      | .le => (mn, min mx fl)
    if r.1 > r.2 then none else some r

inductive Res | err (msg : String) | empty | range (lo hi : Nat)
  deriving DecidableEq, Repr

/-- `Runtime::evaluate_id_filter`'s loop (`runtime.rs:1473-1512`), filters evaluated in order;
a bound with no `(ceil, floor)` or an empty step returns `Ok(None)`. `min as u64..=max as u64`
is `toNat` (`0 ≤ min ≤ max`, `step_nonempty`). `err` stands for an evaluation error of the
bound expression (the `?`), outside this model. -/
def evalIdFilter (maxId : Nat) (fs : List (V × Op)) : Res :=
  go (0, maxId) fs
where
  go : Int × Int → List (V × Op) → Res
    | (a, b), [] => .range a.toNat b.toNat
    | r, (v, op) :: rest =>
      match bounds v with
      | none => .empty
      | some (c, fl) => match step r (op, c, fl) with
        | none => .empty
        | some r' => go r' rest

/-- Graph model for id seeks. Ids are dense: every id `< bound` is live unless deleted
(`Graph::node_count + deleted_nodes.len()` is the id bound; `id_space.rs`). -/
structure G where
  nodeCount : Nat
  deleted : List Nat
  deriving Repr

def G.bound (g : G) : Nat := g.nodeCount + g.deleted.length
def G.live (g : G) (x : Nat) : Bool := decide (x < g.bound) && !g.deleted.contains x

/-- `Graph::max_node_id` (`graph.rs:1783-1786` → `IdSpace::max_id`, id_space.rs:353). -/
def maxNodeId (g : G) : Nat := if g.nodeCount = 0 then 0 else g.bound - 1

/-- `Runtime::evaluate_id_filter` (#3394, `runtime.rs:1473-1523`): the loop, then
`if self.graph().node_count() == 0 { return Ok(None) }` before `insert_range(min..=max)`.
The loop's error and early `Ok(None)` come first, so only a `range` result is affected. -/
def evalIdFilterMain (nodeCount maxId : Nat) (fs : List (V × Op)) : Res :=
  match evalIdFilter maxId fs with
  | .range a b => if nodeCount = 0 then .empty else .range a b
  | r => r

/-- `NodeByIdSeekOp`: every id of the range that is not deleted (no liveness check
beyond `deleted_nodes`, `node_by_id_seek.rs:63-73`). Membership test; `none` = query error. -/
def seek (g : G) (fs : List (V × Op)) (x : Nat) : Option Bool :=
  match evalIdFilterMain g.nodeCount (maxNodeId g) fs with
  | .err _ => none
  | .empty => some false
  | .range a b => some (decide (a ≤ x ∧ x ≤ b) && !g.deleted.contains x)

/-- History: `NodeByIdSeekOp` without the #3394 check, over the loop alone. -/
def pre3394_seek (g : G) (fs : List (V × Op)) (x : Nat) : Option Bool :=
  match evalIdFilter (maxNodeId g) fs with
  | .err _ => none
  | .empty => some false
  | .range a b => some (decide (a ≤ x ∧ x ≤ b) && !g.deleted.contains x)

/-- On a graph with nodes the #3394 check is a no-op. -/
theorem seek_eq_pre3394 (g : G) (fs : List (V × Op)) (hne : g.nodeCount ≠ 0) (x : Nat) :
    seek g fs x = pre3394_seek g fs x := by
  unfold seek pre3394_seek evalIdFilterMain
  cases evalIdFilter (maxNodeId g) fs <;> simp [hne]

/-- The reference: `Filter(AND of id(n) op v)` over `AllNodeScan`. -/
def reference (g : G) (fs : List (V × Op)) (x : Nat) : Bool :=
  g.live x && fs.all (fun p => cypherHolds p.2 x p.1)

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

/-- Every conjunct, each given as its operator and `(ceil, floor)`. -/
def natSat (fs : List (Op × Int × Int)) (x : Int) : Bool :=
  fs.all (fun p => p.1.holdsB x p.2.1 p.2.2)

/-- **PROVEN** (`step` is exact, empty case): `none` means no id of `[mn,mx]` satisfies it. -/
theorem step_none (mn mx c fl : Int) (op : Op) (h : step (mn, mx) (op, c, fl) = none) :
    ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ op.holdsB x c fl = true) := by
  intro x hx
  cases op <;> simp only [step] at h <;> split at h <;> simp_all [Op.holdsB] <;> omega

/-- **PROVEN** (`step` is exact): the new range holds exactly the ids of `[mn,mx]` that
satisfy the comparison. -/
theorem step_some (mn mx c fl a b : Int) (op : Op) (h : step (mn, mx) (op, c, fl) = some (a, b)) :
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ op.holdsB x c fl = true) := by
  intro x
  cases op <;> simp only [step] at h <;> split at h <;> simp_all [Op.holdsB] <;>
    (obtain ⟨rfl, rfl⟩ := h) <;> omega

/-- **PROVEN** (invariant `min ≤ max`): every step yields a non-empty range that starts no lower
than the last, so `0 ≤ min ≤ max` and `insert_range(min as u64..=max as u64)` is never inverted
or wrapped. -/
theorem step_nonempty (mn mx c fl a b : Int) (op : Op) (h0 : mn ≤ mx)
    (h : step (mn, mx) (op, c, fl) = some (a, b)) : a ≤ b ∧ mn ≤ a ∧ b ≤ mx := by
  cases op <;> simp only [step] at h <;> split at h <;> simp_all <;>
    (obtain ⟨rfl, rfl⟩ := h) <;> omega

/-- The fold over `(op, ceil, floor)` operands. -/
def fold : Int × Int → List (Op × Int × Int) → Option (Int × Int)
  | r, [] => some r
  | r, (op, c, fl) :: rest => match step r (op, c, fl) with
    | none => none
    | some r' => fold r' rest

/-- **PROVEN**: an empty fold means no id in the initial range satisfies every comparison. -/
theorem fold_none : ∀ (fs : List (Op × Int × Int)) (mn mx : Int), fold (mn, mx) fs = none →
    ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ natSat fs x = true)
  | [], _, _, h => by simp [fold] at h
  | (op, c, fl) :: rest, mn, mx, h => by
    intro x ⟨h1, h2, h3⟩
    simp only [natSat, List.all_cons, Bool.and_eq_true] at h3
    simp only [fold] at h
    split at h
    · next heq => exact step_none mn mx c fl op heq x ⟨h1, h2, h3.1⟩
    · next r' heq =>
      obtain ⟨a, b⟩ := r'
      have := (step_some mn mx c fl a b op heq x).mpr ⟨h1, h2, h3.1⟩
      exact fold_none rest a b h x ⟨this.1, this.2, h3.2⟩

/-- **PROVEN**: the fold is exact — `x` is in the final range iff it is in the initial one
and satisfies every comparison. -/
theorem fold_some : ∀ (fs : List (Op × Int × Int)) (mn mx a b : Int), fold (mn, mx) fs = some (a, b) →
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ natSat fs x = true)
  | [], mn, mx, a, b, h => by
    simp [fold] at h; obtain ⟨rfl, rfl⟩ := h; intro x; simp [natSat]
  | (op, c, fl) :: rest, mn, mx, a, b, h => by
    intro x
    simp only [natSat, List.all_cons, Bool.and_eq_true]
    simp only [fold] at h
    split at h
    · simp at h
    · next r' heq =>
      obtain ⟨a', b'⟩ := r'
      have hs := step_some mn mx c fl a' b' op heq x
      have ih := fold_some rest a' b' a b h x
      simp only [natSat] at ih
      rw [ih]
      constructor
      · rintro ⟨h1, h2, h4⟩
        have := hs.mp ⟨h1, h2⟩
        exact ⟨this.1, this.2.1, this.2.2, h4⟩
      · rintro ⟨h1, h2, h3, h4⟩
        have := hs.mpr ⟨h1, h2, h3⟩
        exact ⟨this.1, this.2, h4⟩

/-- **PROVEN**: a fold from a non-empty range ends in a non-empty sub-range. -/
theorem fold_nonempty : ∀ (fs : List (Op × Int × Int)) (mn mx a b : Int), mn ≤ mx →
    fold (mn, mx) fs = some (a, b) → a ≤ b ∧ mn ≤ a ∧ b ≤ mx
  | [], mn, mx, a, b, h0, h => by
    simp [fold] at h; obtain ⟨rfl, rfl⟩ := h; omega
  | (op, c, fl) :: rest, mn, mx, a, b, h0, h => by
    simp only [fold] at h
    split at h
    · simp at h
    · next r' heq =>
      obtain ⟨a', b'⟩ := r'
      have := step_nonempty mn mx c fl a' b' op h0 heq
      have := fold_nonempty rest a' b' a b this.1 h
      omega

/-- The `(op, ceil, floor)` list the loop sees, or `none` when some bound has no `(ceil, floor)`. -/
def boundsAll (fs : List (V × Op)) : Option (List (Op × Int × Int)) :=
  fs.mapM (fun p => (bounds p.1).map (fun b => (p.2, b)))

/-- **PROVEN**: the runtime loop is the fold over the bounds; a bound with no `(ceil, floor)`
(null, NaN, non-numeric) empties the seek. -/
theorem go_eq_fold : ∀ (fs : List (V × Op)) (r : Int × Int),
    evalIdFilter.go r fs =
      match boundsAll fs with
      | none => .empty
      | some bs => match fold r bs with | none => .empty | some (a, b) => .range a.toNat b.toNat
  | [], (a, b) => by simp [evalIdFilter.go, boundsAll, fold]
  | (v, op) :: rest, r => by
    simp only [evalIdFilter.go, boundsAll, List.mapM_cons]
    cases hb : bounds v with
    | none => simp
    | some cf =>
      obtain ⟨c, fl⟩ := cf
      simp only [Option.map_some]
      cases hs : step r (op, c, fl) with
      | none =>
        simp only []
        cases hr : rest.mapM (fun p => (bounds p.1).map (fun b => (p.2, b))) <;>
          simp [fold, hs]
      | some r' =>
        have ih := go_eq_fold rest r'
        simp only [boundsAll] at ih
        simp only [ih]
        cases hr : rest.mapM (fun p => (bounds p.1).map (fun b => (p.2, b))) <;>
          simp [fold, hs]

theorem holdsB_int (op : Op) (x i : Int) : op.holdsB x i i = op.holds x i := by
  cases op <;> simp only [Op.holdsB, Op.holds, decide_eq_decide] <;> omega

/-- **PROVEN** (each bound is exact): on every id `x < 2^53`, the `(ceil, floor)` the runtime
derives from `v` decides `id op v` exactly as Cypher does, and a bound with none never holds. -/
theorem bounds_exact (v : V) (op : Op) (x : Nat) (hx : x < 2 ^ 53) :
    match bounds v with
    | none => cypherHolds op x v = false
    | some (c, fl) => op.holdsB x c fl = cypherHolds op x v := by
  cases v with
  | int i => simp only [bounds, cypherHolds, holdsB_int]
  | float f =>
    have hg := firstIdGe_spec f x (by omega) (by omega)
    have hl := lastIdLe_spec f x (by omega) (by omega)
    simp only [bounds]
    cases op <;> simp only [Op.holdsB, cypherHolds] <;>
      cases h1 : Fl.geOf (↑x) f <;> cases h2 : Fl.gtOf (↑x) f <;> simp_all <;> omega
  | nan => simp [bounds, cypherHolds]
  | null => simp [bounds, cypherHolds]
  | str s => simp [bounds, cypherHolds]

theorem boundsAll_cons (v : V) (op : Op) (rest : List (V × Op)) :
    boundsAll ((v, op) :: rest) =
      match bounds v, boundsAll rest with
      | some cf, some bs => some ((op, cf) :: bs)
      | _, _ => none := by
  simp only [boundsAll, List.mapM_cons]
  cases bounds v <;> cases rest.mapM (fun p => (bounds p.1).map (fun b => (p.2, b))) <;> rfl

theorem boundsAll_none (fs : List (V × Op)) (h : boundsAll fs = none) (x : Nat) (hx : x < 2 ^ 53) :
    fs.all (fun p => cypherHolds p.2 x p.1) = false := by
  induction fs with
  | nil => simp [boundsAll] at h
  | cons p rest ih =>
    obtain ⟨v, op⟩ := p
    rw [boundsAll_cons] at h
    have he := bounds_exact v op x hx
    simp only [List.all_cons, Bool.and_eq_false_iff]
    cases hb : bounds v with
    | none => rw [hb] at he; left; exact he
    | some cf =>
      right; apply ih
      rw [hb] at h
      cases hr : boundsAll rest with
      | none => rfl
      | some bs => rw [hr] at h; simp at h

theorem boundsAll_some (fs : List (V × Op)) (bs : List (Op × Int × Int)) (h : boundsAll fs = some bs)
    (x : Nat) (hx : x < 2 ^ 53) :
    fs.all (fun p => cypherHolds p.2 x p.1) = natSat bs x := by
  induction fs generalizing bs with
  | nil => simp [boundsAll] at h; subst h; rfl
  | cons p rest ih =>
    obtain ⟨v, op⟩ := p
    rw [boundsAll_cons] at h
    have he := bounds_exact v op x hx
    cases hb : bounds v with
    | none => rw [hb] at h; simp at h
    | some cf =>
      obtain ⟨c, fl⟩ := cf
      rw [hb] at he h
      cases hr : boundsAll rest with
      | none => rw [hr] at h; simp at h
      | some bs' =>
        rw [hr] at h
        simp only [Option.some.injEq] at h
        subst h
        simp only [List.all_cons, natSat] at *
        rw [← he, ih bs' hr]

theorem evalIdFilter_ne_err (maxId : Nat) (fs : List (V × Op)) (m : String) :
    evalIdFilter maxId fs ≠ .err m := by
  unfold evalIdFilter
  rw [go_eq_fold]
  cases boundsAll fs with
  | none => simp
  | some bs =>
    dsimp only
    cases fold (0, (maxId : Int)) bs with
    | none => simp
    | some r => obtain ⟨a, b⟩ := r; simp

/-- On a graph with nodes, the seek's range test for `x` is `x` live and in the fold's range. -/
theorem range_case (g : G) (h0 : g.nodeCount ≠ 0) (bs : List (Op × Int × Int)) (a b : Int)
    (hf : fold (0, (maxNodeId g : Int)) bs = some (a, b)) (x : Nat) (hsat : natSat bs x = true) :
    (decide (a.toNat ≤ x ∧ x ≤ b.toNat) && !g.deleted.contains x) = g.live x := by
  have hm : (maxNodeId g : Int) = g.bound - 1 := by
    unfold maxNodeId G.bound at *; simp [h0]; omega
  have hb : 0 < g.bound := by unfold G.bound; omega
  have hne := fold_nonempty bs 0 _ a b (by omega) hf
  have hi := fold_some bs 0 _ a b hf x
  rw [hsat] at hi
  simp only [and_true] at hi
  simp only [G.live]
  congr 1
  rw [decide_eq_decide]
  omega

theorem out_of_range (g : G) (bs : List (Op × Int × Int)) (a b : Int)
    (hf : fold (0, (maxNodeId g : Int)) bs = some (a, b)) (x : Nat) (hsat : natSat bs x = false) :
    (decide (a.toNat ≤ x ∧ x ≤ b.toNat) && !g.deleted.contains x) = false := by
  have hne := fold_nonempty bs 0 _ a b (by omega) hf
  have hi := fold_some bs 0 _ a b hf x
  rw [hsat] at hi
  simp only [Bool.false_eq_true, and_false, iff_false] at hi
  simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
  left; omega

/-- **PROVEN** (headline, PR #3017): on any well-formed graph whose ids stay below 2^53 —
including one that never held a node — and for **any** operands (negative or huge integers,
integral or fractional floats, `±inf`, NaN, null, strings), `NodeByIdSeek` returns exactly the
nodes `Filter(id(n) op v …)` over `AllNodeScan` would. Before the PR the seek dropped every row
for `id(n) > -1`, kept every row for `id(n) < -1`, and errored on floats and null. -/
theorem seek_sound_all (g : G) (hwf : g.wf) (hb : g.bound ≤ 2 ^ 53) (fs : List (V × Op)) (x : Nat) :
    seek g fs x = some (reference g fs x) := by
  by_cases h0 : g.nodeCount = 0
  · have hl := live_false_of_zero g hwf h0 x
    have hne := evalIdFilter_ne_err (maxNodeId g) fs
    unfold seek evalIdFilterMain reference
    rw [hl, Bool.false_and]
    cases h : evalIdFilter (maxNodeId g) fs with
    | err m => exact absurd h (hne m)
    | empty => rfl
    | range a b => simp [h0]
  · have hm : (maxNodeId g : Int) = g.bound - 1 := by
      unfold maxNodeId G.bound at *; simp [h0]; omega
    unfold seek evalIdFilterMain evalIdFilter reference
    rw [go_eq_fold]
    cases hba : boundsAll fs with
    | none =>
      dsimp only
      by_cases hxb : x < g.bound
      · rw [boundsAll_none fs hba x (by omega)]; simp
      · simp [G.live, hxb]
    | some bs =>
      dsimp only
      cases hf : fold (0, (maxNodeId g : Int)) bs with
      | none =>
        dsimp only
        by_cases hxb : x < g.bound
        · rw [boundsAll_some fs bs hba x (by omega)]
          have := fold_none bs 0 _ hf x
          have hn : natSat bs x = false := by
            cases hs : natSat bs x
            · rfl
            · exact absurd ⟨by omega, by omega, hs⟩ this
          simp [hn]
        · simp [G.live, hxb]
      | some r =>
        obtain ⟨a, b⟩ := r
        dsimp only
        rw [ite_eq_right_iff.mpr (fun h => absurd h h0)]
        dsimp only
        congr 1
        by_cases hxb : x < g.bound
        · rw [boundsAll_some fs bs hba x (by omega)]
          cases hs : natSat bs x
          · rw [out_of_range g bs a b hf x hs]; simp
          · rw [range_case g h0 bs a b hf x hs]; simp
        · have hne := fold_nonempty bs 0 _ a b (by omega) hf
          have : (decide (a.toNat ≤ x ∧ x ≤ b.toNat) && !g.deleted.contains x) = false := by
            simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; left; omega
          rw [this]; simp [G.live, hxb]

theorem boundsAll_int (fs : List (Int × Op)) :
    boundsAll (fs.map (fun p => (V.int p.1, p.2))) = some (fs.map (fun p => (p.2, p.1, p.1))) := by
  induction fs with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons]
    rw [boundsAll_cons, ih]
    rfl

theorem all_cypher (fs : List (Int × Op)) (x : Nat) :
    (fs.map (fun p => (V.int p.1, p.2))).all (fun p => cypherHolds p.2 x p.1)
      = natSat (fs.map (fun p => (p.2, p.1, p.1))) x := by
  induction fs with
  | nil => rfl
  | cons p rest ih =>
    simp only [natSat] at ih
    rw [List.map_cons, List.all_cons, ih]
    simp only [natSat, List.map_cons, List.all_cons, cypherHolds, holdsB_int]

/-- **PROVEN** (history, before #3394): for a non-empty graph and integer operands of any sign
and size the seek without the #3394 check returns exactly the nodes `Filter(id(n) op v …)` over
`AllNodeScan` would; dropping `nodeCount ≠ 0` gave the phantom node 0 (W2-scan-2). -/
theorem pre3394_seek_sound_int (g : G) (fs : List (Int × Op)) (hne : g.nodeCount ≠ 0) (x : Nat) :
    pre3394_seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  have hm : (maxNodeId g : Int) = g.bound - 1 := by
    unfold maxNodeId G.bound at *; simp [hne]; omega
  unfold pre3394_seek evalIdFilter reference
  rw [go_eq_fold, boundsAll_int, all_cypher]
  dsimp only
  cases hf : fold (0, (maxNodeId g : Int)) (fs.map (fun p => (p.2, p.1, p.1))) with
  | none =>
    dsimp only
    have := fold_none _ 0 _ hf x
    cases hs : natSat (fs.map (fun p => (p.2, p.1, p.1))) x
    · simp
    · have hxb : ¬ x < g.bound := fun hxb => this ⟨by omega, by omega, hs⟩
      simp [G.live, hxb]
  | some r =>
    obtain ⟨a, b⟩ := r
    dsimp only
    congr 1
    cases hs : natSat (fs.map (fun p => (p.2, p.1, p.1))) x
    · rw [out_of_range g _ a b hf x hs]; simp
    · rw [range_case g hne _ a b hf x hs]; simp

/-- **PROVEN** (history, before #3394, statement unchanged): non-negative `i64` operands. -/
theorem pre3394_seek_sound (g : G) (fs : List (Nat × Op)) (hne : g.nodeCount ≠ 0)
    (_hsmall : ∀ p ∈ fs, p.1 < 2 ^ 63) (x : Nat) :
    pre3394_seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  have := pre3394_seek_sound_int g (fs.map (fun p => ((p.1 : Int), p.2))) hne x
  simpa [List.map_map, Function.comp_def] using this

/-- **PROVEN** (PR #3017): on any well-formed graph (any size), for integer operands of any sign,
`NodeByIdSeek` returns exactly the nodes `Filter(id(n) op v …)` over `AllNodeScan` would. Before
the PR a negative operand was reinterpreted as `u64` (`id(n) > -1` → no rows). -/
theorem seek_sound_int (g : G) (hwf : g.wf) (fs : List (Int × Op)) (x : Nat) :
    seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  by_cases hne : g.nodeCount = 0
  · have hl := live_false_of_zero g hwf hne x
    unfold seek evalIdFilterMain evalIdFilter reference
    rw [go_eq_fold, boundsAll_int, hl]
    simp only []
    cases fold (0, (maxNodeId g : Int)) (fs.map (fun p => (p.2, p.1, p.1))) with
    | none => rfl
    | some r => obtain ⟨a, b⟩ := r; simp [hne]
  · rw [seek_eq_pre3394 g _ hne]
    exact pre3394_seek_sound_int g fs hne x

/-- **PROVEN** (headline, statement unchanged since #3394): on any well-formed graph, with
non-negative integer operands, `NodeByIdSeek` returns exactly the nodes `Filter(id(n) op v …)`
over `AllNodeScan` would. Since PR #3017 the operand hypotheses are no longer needed:
`seek_sound_int` (any `i64`), `seek_sound_all` (any operand, ids below 2^53). -/
theorem seek_sound (g : G) (hwf : g.wf) (fs : List (Nat × Op))
    (_hsmall : ∀ p ∈ fs, p.1 < 2 ^ 63) (x : Nat) :
    seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  have := seek_sound_int g hwf (fs.map (fun p => ((p.1 : Int), p.2))) x
  simpa [List.map_map, Function.comp_def] using this

/-! ### Before PR #3017: the old counterexamples now agree with Cypher -/

/-- Five live nodes 0..4. -/
def g5 : G := { nodeCount := 5, deleted := [] }

/-- `id(n) > -1`: every node satisfies it; before the PR the seek returned none
(`-1 as u64 = 2^64-1`), now all five. -/
example : evalIdFilter (maxNodeId g5) [(.int (-1), .gt)] = .range 0 4 := by decide
example : reference g5 [(.int (-1), .gt)] 0 = true := by decide

/-- `id(n) < -1`: no node satisfies it; before the PR the seek returned all five, now none. -/
example : evalIdFilter (maxNodeId g5) [(.int (-1), .lt)] = .empty := by decide
example : reference g5 [(.int (-1), .lt)] 0 = false := by decide

/-- `id(n) = 1.0`: Cypher's `1 = 1.0` is true; before the PR the runtime errored, now node 1. -/
example : evalIdFilter (maxNodeId g5) [(.float (.fin 1 0), .eq)] = .range 1 1 := by decide
example : reference g5 [(.float (.fin 1 0), .eq)] 1 = true := by decide

/-- `id(n) > 1.5` (`3/2`) starts at 2, `id(n) = 1.5` is empty. -/
example : evalIdFilter (maxNodeId g5) [(.float (.fin 3 1), .gt)] = .range 2 4 := by decide
example : evalIdFilter (maxNodeId g5) [(.float (.fin 3 1), .eq)] = .empty := by decide

/-- `id(n) = null`: Cypher drops every row; before the PR the runtime errored, now empty. -/
example : evalIdFilter (maxNodeId g5) [(.null, .eq)] = .empty := by decide
example : reference g5 [(.null, .eq)] 1 = false := by decide

/-- A graph that has never held a node (W2-scan-2, #3353). Before #3394 `max_node_id` was 0,
the range `{0}`, nothing deleted, so `NodeByIdSeek` yielded node 0, which does not exist. -/
def gEmpty : G := { nodeCount := 0, deleted := [] }
theorem gEmpty_wf : gEmpty.wf := by simp [G.wf, gEmpty, G.bound]
example : reference gEmpty [(.int 0, .eq)] 0 = false := by decide

/-- **PROVEN** (#3394): the seek on a never-populated graph is empty, matching the
reference (a special case of `seek_sound_all`). -/
theorem empty_graph_seek_none (fs : List (V × Op)) (x : Nat) : seek gEmpty fs x ≠ some true := by
  unfold seek evalIdFilterMain
  cases evalIdFilter (maxNodeId gEmpty) fs <;> simp [gEmpty]
example : seek gEmpty [(.int 0, .eq)] 0 = some false := by decide

/-- …while an emptied graph (every node deleted) was fine even before, because 0 is in
`deleted`. -/
example : pre3394_seek { nodeCount := 0, deleted := [0, 1] } [(.int 0, .eq)] 0 = some false := by
  decide
example : seek { nodeCount := 0, deleted := [0, 1] } [(.int 0, .eq)] 0 = some false := by decide

/-- A later non-numeric conjunct no longer errors: `id(n) = 100 AND id(n) = 'x'` and
`id(n) = 1 AND id(n) = 'x'` are both empty, as in Cypher. -/
example : evalIdFilter 4 [(.int 100, .eq), (.str "x", .eq)] = .empty := by decide
example : evalIdFilter 4 [(.int 1, .eq), (.str "x", .eq)] = .empty := by decide

end OptimizerScan.IdSeek
