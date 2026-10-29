/-
# Id seeks: `utilize_node_by_id` + `Runtime::evaluate_id_filter`

| here | there |
| --- | --- |
| `Op`, `Op.flip`            | `ExprIR::{Eq,Gt,Ge,Lt,Le}` and the flip table, `planner/optimizer/utilize_node_by_id.rs:80-87` |
| `getIdFilter`              | `get_id_filter`, `utilize_node_by_id.rs:52-92` |
| `collectFilters`           | the AND walk of `utilize_node_by_id`, `utilize_node_by_id.rs:106-119` |
| `asU64`                    | `Value::Int(id) => id as u64`, `runtime/runtime.rs:1428` |
| `step`                     | one arm of the `match op`, `runtime/runtime.rs:1433-1468` |
| `evalIdFilter`             | the filter loop of `Runtime::evaluate_id_filter`, `runtime/runtime.rs:1415-1469` |
| `evalIdFilterMain`         | `Runtime::evaluate_id_filter` on main b84473a7f (#3394): the loop, then `node_count() == 0 → Ok(None)`, `runtime.rs:1470-1479` |
| `maxNodeId`                | `Graph::max_node_id`, `graph/graph.rs:1624-1604` (since #2846 `node_ids.max_id()`, id_space.rs:320 — same arithmetic) |
| `seek`                     | `NodeByIdSeekOp::next`: range minus `deleted_nodes`, `runtime/ops/node_by_id_seek.rs:63-73` |
| `pre3394_seek`             | the same before #3394 (b6567cb97): no `node_count` check |
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

/-- The operand of an id comparison, after evaluation. Only `int` is accepted by the runtime. -/
inductive V | int (i : Int) | float (f : Int) | null | str (s : String)
  deriving DecidableEq, Repr

/-- Cypher's `id(n) op v` for an integer id `x`, as a WHERE predicate (null/false both drop).
Int/Float compare numerically; `float f` stands for the float whose value is the integer `f`. -/
def cypherHolds (op : Op) (x : Nat) : V → Bool
  | .int i => op.holds x i
  | .float f => op.holds x f
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

/-- `id as u64`: two's-complement reinterpretation (`runtime.rs:1428`). -/
def asU64 (i : Int) : Nat := (i % (2 ^ 64 : Int)).toNat

/-- One arm of `evaluate_id_filter`'s `match op` on the running `[min, max]`
(`runtime.rs:1433-1468`). `none` is the early `return Ok(None)`. -/
def step : Nat × Nat → Op × Nat → Option (Nat × Nat)
  | (mn, mx), (.eq, id) => if id < mn ∨ id > mx then none else some (id, id)
  | (mn, mx), (.gt, id) => if id ≥ mx then none else some (max mn (id + 1), mx)
  | (mn, mx), (.ge, id) => if id > mx then none else some (max mn id, mx)
  | (mn, mx), (.lt, id) => if id ≤ mn then none else some (mn, min mx (id - 1))
  | (mn, mx), (.le, id) => if id < mn then none else some (mn, min mx id)

inductive Res | err (msg : String) | empty | range (lo hi : Nat)
  deriving DecidableEq, Repr

/-- `Runtime::evaluate_id_filter` (`runtime.rs:1415-1473`), filters evaluated in order;
the first non-integer is an error, the first empty step returns `Ok(None)`. -/
def evalIdFilter (maxId : Nat) (fs : List (V × Op)) : Res :=
  go (0, maxId) fs
where
  go : Nat × Nat → List (V × Op) → Res
    | (a, b), [] => .range a b
    | r, (v, op) :: rest =>
      match v with
      | .int i => match step r (op, asU64 i) with
        | none => .empty
        | some r' => go r' rest
      | _ => .err "Node ID must be an integer"

/-- Graph model for id seeks. Ids are dense: every id `< bound` is live unless deleted
(`Graph::node_count + deleted_nodes.len()` is the id bound; `id_space.rs`). -/
structure G where
  nodeCount : Nat
  deleted : List Nat
  deriving Repr

def G.bound (g : G) : Nat := g.nodeCount + g.deleted.length
def G.live (g : G) (x : Nat) : Bool := decide (x < g.bound) && !g.deleted.contains x

/-- `Graph::max_node_id` (`graph.rs:1624-1604` → `IdSpace::max_id`, id_space.rs:320). -/
def maxNodeId (g : G) : Nat := if g.nodeCount = 0 then 0 else g.bound - 1

/-- `Runtime::evaluate_id_filter` on main (#3394, `runtime.rs:1415-1479`): the same loop, then
`if self.g.borrow().node_count() == 0 { return Ok(None) }` before `insert_range(min..=max)`.
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

/-- History: `NodeByIdSeekOp` before #3394 (b6567cb97), over the loop alone. -/
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

def natSat (fs : List (Nat × Op)) (x : Nat) : Bool := fs.all (fun p => p.2.holds x p.1)

/-- **PROVEN** (`step` is exact, empty case): `none` means no id of `[mn,mx]` satisfies it. -/
theorem step_none (mn mx id : Nat) (op : Op) (h : step (mn, mx) (op, id) = none) :
    ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ op.holds x id = true) := by
  intro x hx
  cases op <;> simp only [step] at h <;> split at h <;> simp_all [Op.holds] <;> omega

/-- **PROVEN** (`step` is exact): the new range holds exactly the ids of `[mn,mx]` that
satisfy the comparison. -/
theorem step_some (mn mx id a b : Nat) (op : Op) (h : step (mn, mx) (op, id) = some (a, b)) :
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ op.holds x id = true) := by
  intro x
  cases op <;> simp only [step] at h <;> split at h <;> simp_all [Op.holds] <;>
    (obtain ⟨rfl, rfl⟩ := h) <;> omega

/-- **PROVEN** (invariant `min ≤ max`): from a non-empty range every step yields a
non-empty range, so `insert_range(min..=max)` never sees `min > max`. -/
theorem step_nonempty (mn mx id a b : Nat) (op : Op) (h0 : mn ≤ mx)
    (h : step (mn, mx) (op, id) = some (a, b)) : a ≤ b := by
  cases op <;> simp only [step] at h <;> split at h <;> simp_all <;>
    (obtain ⟨rfl, rfl⟩ := h) <;> omega

/-- The fold over natural-number operands. -/
def fold : Nat × Nat → List (Nat × Op) → Option (Nat × Nat)
  | r, [] => some r
  | r, (id, op) :: rest => match step r (op, id) with
    | none => none
    | some r' => fold r' rest

/-- **PROVEN**: an empty fold means no id in the initial range satisfies every comparison. -/
theorem fold_none : ∀ (fs : List (Nat × Op)) (mn mx : Nat), fold (mn, mx) fs = none →
    ∀ x, ¬ (mn ≤ x ∧ x ≤ mx ∧ natSat fs x = true)
  | [], _, _, h => by simp [fold] at h
  | (id, op) :: rest, mn, mx, h => by
    intro x ⟨h1, h2, h3⟩
    simp only [natSat, List.all_cons, Bool.and_eq_true] at h3
    simp only [fold] at h
    split at h
    · next heq => exact step_none mn mx id op heq x ⟨h1, h2, h3.1⟩
    · next r' heq =>
      obtain ⟨a, b⟩ := r'
      have := (step_some mn mx id a b op heq x).mpr ⟨h1, h2, h3.1⟩
      exact fold_none rest a b h x ⟨this.1, this.2, h3.2⟩

/-- **PROVEN**: the fold is exact — `x` is in the final range iff it is in the initial one
and satisfies every comparison. -/
theorem fold_some : ∀ (fs : List (Nat × Op)) (mn mx a b : Nat), fold (mn, mx) fs = some (a, b) →
    ∀ x, (a ≤ x ∧ x ≤ b) ↔ (mn ≤ x ∧ x ≤ mx ∧ natSat fs x = true)
  | [], mn, mx, a, b, h => by
    simp [fold] at h; obtain ⟨rfl, rfl⟩ := h; intro x; simp [natSat]
  | (id, op) :: rest, mn, mx, a, b, h => by
    intro x
    simp only [natSat, List.all_cons, Bool.and_eq_true]
    simp only [fold] at h
    split at h
    · simp at h
    · next r' heq =>
      obtain ⟨a', b'⟩ := r'
      have hs := step_some mn mx id a' b' op heq x
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

theorem asU64_of_small (i : Nat) (h : i < 2 ^ 63) : asU64 (i : Int) = i := by
  unfold asU64
  rw [Int.emod_eq_of_lt (by omega) (by omega)]
  simp

/-- The runtime loop agrees with `fold` when every operand is a non-negative `i64`. -/
theorem go_eq_fold : ∀ (fs : List (Nat × Op)) (r : Nat × Nat),
    (∀ p ∈ fs, p.1 < 2 ^ 63) →
    evalIdFilter.go r (fs.map (fun p => (V.int p.1, p.2))) =
      match fold r fs with | none => .empty | some (a, b) => .range a b
  | [], (a, b), _ => by simp [evalIdFilter.go, fold]
  | (id, op) :: rest, r, h => by
    have hid := asU64_of_small id (h (id, op) (by simp))
    simp only [List.map, evalIdFilter.go, fold, hid]
    split
    · rfl
    · exact go_eq_fold rest _ (fun p hp => h p (by simp [hp]))

theorem all_cypher (fs : List (Nat × Op)) (x : Nat) :
    (fs.map (fun p => (V.int p.1, p.2))).all (fun p => cypherHolds p.2 x p.1) = natSat fs x := by
  induction fs with
  | nil => rfl
  | cons p rest ih =>
    simp only [natSat] at ih
    simp only [List.map, List.all_cons, natSat, cypherHolds]
    rw [← ih]; rfl

/-- **PROVEN** (history, before #3394): for a non-empty graph and non-negative integer operands
the pre-#3394 `NodeByIdSeek` returned exactly the nodes `Filter(id(n) op v …)` over
`AllNodeScan` would; dropping `nodeCount ≠ 0` gave the phantom node 0 (W2-scan-2). -/
theorem pre3394_seek_sound (g : G) (fs : List (Nat × Op)) (hne : g.nodeCount ≠ 0)
    (hsmall : ∀ p ∈ fs, p.1 < 2 ^ 63) (x : Nat) :
    pre3394_seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  have hmax : maxNodeId g = g.bound - 1 := by simp [maxNodeId, hne]
  have hb : 0 < g.bound := by unfold G.bound; omega
  unfold pre3394_seek evalIdFilter reference G.live
  rw [go_eq_fold fs _ hsmall, all_cypher]
  cases hf : fold (0, maxNodeId g) fs with
  | none =>
    have := fold_none fs 0 _ hf x
    simp only [Option.some.injEq]
    cases hs : natSat fs x <;> cases hc : g.deleted.contains x <;> simp_all <;> omega
  | some r =>
    obtain ⟨a, b⟩ := r
    have := fold_some fs 0 _ a b hf x
    simp only [Option.some.injEq]
    cases hs : natSat fs x <;> cases hc : g.deleted.contains x <;> simp_all <;> omega

/-- **PROVEN** (headline, id-seek soundness, main b84473a7f): on any well-formed graph —
including one that never held a node — with non-negative integer operands, `NodeByIdSeek`
returns exactly the nodes `Filter(id(n) op v …)` over `AllNodeScan` would. The remaining
hypothesis is a confirmed bug: drop `v ≥ 0` and you get the negative-id counterexamples
below. #3394 removed the `nodeCount ≠ 0` hypothesis (closes #3353 / W2-scan-2). -/
theorem seek_sound (g : G) (hwf : g.wf) (fs : List (Nat × Op))
    (hsmall : ∀ p ∈ fs, p.1 < 2 ^ 63) (x : Nat) :
    seek g (fs.map (fun p => (V.int p.1, p.2))) x
      = some (reference g (fs.map (fun p => (V.int p.1, p.2))) x) := by
  by_cases hne : g.nodeCount = 0
  · have hl := live_false_of_zero g hwf hne x
    unfold seek evalIdFilterMain evalIdFilter reference
    rw [go_eq_fold fs _ hsmall, hl]
    cases fold (0, maxNodeId g) fs with
    | none => rfl
    | some r => obtain ⟨a, b⟩ := r; simp [hne]
  · rw [seek_eq_pre3394 g _ hne]
    exact pre3394_seek_sound g fs hne hsmall x

/-! ### Counterexamples — each reproduced against the real server (see header of
`OptimizerScan.lean`) -/

/-- Five live nodes 0..4. -/
def g5 : G := { nodeCount := 5, deleted := [] }

/-- `id(n) > -1`: every node satisfies it, the seek returns none (`-1 as u64 = 2^64-1`). -/
example : evalIdFilter (maxNodeId g5) [(.int (-1), .gt)] = .empty := by decide
example : reference g5 [(.int (-1), .gt)] 0 = true := by decide

/-- `id(n) < -1`: no node satisfies it, the seek returns all five. -/
example : evalIdFilter (maxNodeId g5) [(.int (-1), .lt)] = .range 0 4 := by decide
example : reference g5 [(.int (-1), .lt)] 0 = false := by decide

/-- `id(n) = 1.0`: Cypher's `1 = 1.0` is true, the runtime errors. -/
example : evalIdFilter (maxNodeId g5) [(.float 1, .eq)] = .err "Node ID must be an integer" := by
  decide
example : reference g5 [(.float 1, .eq)] 1 = true := by decide

/-- `id(n) = null`: Cypher drops every row; the runtime errors. -/
example : reference g5 [(.null, .eq)] 1 = false := by decide

/-- A graph that has never held a node (W2-scan-2, #3353). Before #3394 `max_node_id` was 0,
the range `{0}`, nothing deleted, so `NodeByIdSeek` yielded node 0, which does not exist. -/
def gEmpty : G := { nodeCount := 0, deleted := [] }
theorem gEmpty_wf : gEmpty.wf := by simp [G.wf, gEmpty, G.bound]
example : reference gEmpty [(.int 0, .eq)] 0 = false := by decide

/-- **PROVEN** (#3394): on main the seek on a never-populated graph is empty, matching the
reference (a special case of `seek_sound`). -/
theorem empty_graph_seek_none (fs : List (V × Op)) (x : Nat) : seek gEmpty fs x ≠ some true := by
  unfold seek evalIdFilterMain
  cases evalIdFilter (maxNodeId gEmpty) fs <;> simp [gEmpty]
example : seek gEmpty [(.int 0, .eq)] 0 = some false := by decide

/-- …while an emptied graph (every node deleted) was fine even before, because 0 is in
`deleted`. -/
example : pre3394_seek { nodeCount := 0, deleted := [0, 1] } [(.int 0, .eq)] 0 = some false := by
  decide
example : seek { nodeCount := 0, deleted := [0, 1] } [(.int 0, .eq)] 0 = some false := by decide

/-- An error in a later conjunct is masked when an earlier one already empties the range
(`id(n) = 100 AND id(n) = 'x'` returns no rows instead of the type error). Modelled only. -/
example : evalIdFilter 4 [(.int 100, .eq), (.str "x", .eq)] = .empty := by decide

end OptimizerScan.IdSeek
