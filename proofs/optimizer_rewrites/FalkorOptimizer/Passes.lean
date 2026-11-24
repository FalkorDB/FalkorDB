/-
# The remaining passes
-/
import FalkorOptimizer.Basic

namespace Falkor.Opt.Passes

open Falkor.Opt

/-! ## fuse_anonymous_traverse -/

def out (E : List (Nat × Nat)) (a : Nat) : List Nat := (E.filter (fun e => e.1 == a)).map Prod.snd

/-- emit_relationship = false: one row per distinct destination. -/
def travCollapsed (E : List (Nat × Nat)) (a : Nat) : List Nat := (out E a).eraseDups

/-- Two chained collapsed traverses, the anonymous middle dropped. -/
def unfused (E : List (Nat × Nat)) (a : Nat) : List Nat :=
  (travCollapsed E a).flatMap (travCollapsed E)

/-- The fused chain: boolean product, one row per reachable `c`. -/
def fused (E : List (Nat × Nat)) (a : Nat) : List Nat :=
  ((travCollapsed E a).flatMap (out E)).eraseDups

/-- Fusion preserves the *set* of rows. -/
theorem fuse_same_support (E : List (Nat × Nat)) (a c : Nat) :
    c ∈ fused E a ↔ c ∈ unfused E a := by
  simp [fused, unfused, travCollapsed, List.mem_eraseDups, List.mem_flatMap]

/-- … but not the multiset: diamond `a→b1→c`, `a→b2→c` gives 2 unfused rows,
1 fused. C FalkorDB also returns 1 (it builds the same algebraic expression);
openCypher says 2. Rust mirrors C here. -/
theorem fuse_changes_multiplicity :
    unfused [(0,1),(0,2),(1,3),(2,3)] 0 = [3, 3] ∧ fused [(0,1),(0,2),(1,3),(2,3)] 0 = [3] := by
  decide

/-! ## fuse_optional_traverse -/

def nulls (vs : List Var) : Row := vs.map (fun v => (v, Val.null))

/-- `Optional(vars)`: input row, sub-plan run with the row as `Argument`;
no match → the row null-padded on `vars`. -/
def optionalOp (vars : List Var) (sub : Row → List Row) (xs : List Row) : List Row :=
  xs.flatMap (fun x => match sub x with | [] => [x ++ nulls vars] | s => s)

/-- `CondTraverse{optional: true}`: null-pads the edge and destination. -/
def optCT (edge dst : Var) (trav : Row → List Row) (xs : List Row) : List Row :=
  xs.flatMap (fun x => match trav x with | [] => [x ++ nulls [edge, dst]] | s => s)

theorem get_nulls (vs : List Var) (v : Var) : (nulls vs).get v = .null := by
  unfold Row.get nulls
  cases h : (List.map (fun v => (v, Val.null)) vs).find? (fun p => p.1 == v) with
  | none => rfl
  | some p =>
    have := List.mem_of_find?_eq_some h
    simp only [List.mem_map] at this
    obtain ⟨_, _, rfl⟩ := this; rfl

theorem get_pad (x : Row) (vs ws : List Var) (v : Var) :
    (x ++ nulls vs).get v = (x ++ nulls ws).get v := by
  by_cases h : x.binds v
  · rw [get_append_left _ _ _ h, get_append_left _ _ _ h]
  · rw [get_append_right _ _ _ h, get_append_right _ _ _ h, get_nulls, get_nulls]

/-- Fusion is exact on every column read, for the fusable shape (sub-plan is
the traverse over `Argument`, `fusable_traverse` l.40-78): same rows, same
order. Unbound reads as NULL, so which columns get padded is irrelevant. -/
theorem fuse_optional_correct (vars : List Var) (edge dst : Var) (trav : Row → List Row)
    (xs : List Row) (v : Var) :
    (optionalOp vars trav xs).map (·.get v) = (optCT edge dst trav xs).map (·.get v) := by
  unfold optionalOp optCT
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons, List.map_append, ih]
    congr 1
    split
    · simp [get_pad x vars [edge, dst] v]
    · rfl

/-! ## utilize_node_by_id / evaluate_id_filter -/

inductive IdOp where | eq | gt | ge | lt | le
deriving DecidableEq, Repr

/-- `get_id_filter` flips the comparison when `id()` is on the right. -/
def IdOp.flip : IdOp → IdOp
  | .eq => .eq | .gt => .lt | .ge => .le | .lt => .gt | .le => .ge

def IdOp.holds : IdOp → Int → Int → Bool
  | .eq, n, c => n == c
  | .gt, n, c => decide (n > c)
  | .ge, n, c => decide (n ≥ c)
  | .lt, n, c => decide (n < c)
  | .le, n, c => decide (n ≤ c)

/-- `c op n`, the query's written order when `id()` is on the right. -/
def IdOp.holdsRev : IdOp → Int → Int → Bool
  | .eq, c, n => c == n
  | .gt, c, n => decide (c > n)
  | .ge, c, n => decide (c ≥ n)
  | .lt, c, n => decide (c < n)
  | .le, c, n => decide (c ≤ n)

/-- The flip in `get_id_filter` is right: `c op id(n)` ≡ `id(n) (flip op) c`. -/
theorem flip_holds (op : IdOp) (n c : Int) : op.flip.holds n c = op.holdsRev c n := by
  cases op <;> simp only [IdOp.flip, IdOp.holds, IdOp.holdsRev] <;>
    apply Bool.eq_iff_iff.2 <;> simp <;> omega

/-- One iteration of the `for (expr, op) in filter` loop of `evaluate_id_filter` (runtime.rs,
PR #3017) for an integer bound `c` (its own ceiling and floor), on the `i128` range. -/
def step (lo hi c : Int) : IdOp → Int × Int
  | .eq => (max lo c, min hi c)
  | .gt => (max lo (c + 1), hi)
  | .ge => (max lo c, hi)
  | .lt => (lo, min hi (c - 1))
  | .le => (lo, min hi c)

/-- The whole function. `Ok(None)` = empty: a non-integer bound (floats are modelled in
`proofs/optimizer_scan` `IdSeek`), or a step that leaves `min > max`. At the end the range is
cast `min as u64 ..= max as u64`. No value is an error since PR #3017. -/
def idFilter (maxId : Nat) (fs : List (Val × IdOp)) : Except Err (Option (Nat × Nat)) :=
  go 0 maxId fs
where
  go (lo hi : Int) : List (Val × IdOp) → Except Err (Option (Nat × Nat))
    | [] => .ok (some (lo.toNat, hi.toNat))
    | (v, op) :: rest =>
      match v with
      | .int i =>
        let r := step lo hi i op
        if r.1 > r.2 then .ok none else go r.1 r.2 rest
      | _ => .ok none

def inRange (r : Option (Nat × Nat)) (n : Nat) : Bool :=
  match r with
  | none => false
  | some (lo, hi) => decide (lo ≤ n ∧ n ≤ hi)

/-- The `saturating_add(1)` / `saturating_sub(1)` never saturate on an `i64` bound, and the
range only narrows, so from `[0, max]` both ends stay in `[0, max]` for the `as u64` casts. -/
theorem step_no_wrap (lo hi c : Int) (hc : -2 ^ 63 ≤ c ∧ c < 2 ^ 63) (op : IdOp) (lo' hi' : Int)
    (h : step lo hi c op = (lo', hi')) :
    (op = .gt → c + 1 < 2 ^ 127) ∧ (op = .lt → -2 ^ 127 ≤ c - 1) ∧ lo ≤ lo' ∧ hi' ≤ hi := by
  cases op <;> simp only [step, Prod.mk.injEq] at h <;> obtain ⟨rfl, rfl⟩ := h <;>
    simp <;> omega

theorem step_correct (lo hi c n : Int) (op : IdOp) :
    decide ((step lo hi c op).1 ≤ n ∧ n ≤ (step lo hi c op).2) =
      (decide (lo ≤ n ∧ n ≤ hi) && op.holds n c) := by
  cases op <;> simp only [step, IdOp.holds] <;>
    apply Bool.eq_iff_iff.2 <;> simp only [decide_eq_true_eq, Bool.and_eq_true, Int.max_le, Int.le_min, beq_iff_eq] <;> omega

/-- **evaluate_id_filter is correct for every integer bound** (PR #3017; it was only for bounds
in `[0, 2^63)` while the bound was cast `as u64`): a node id `n` is in the seek range iff
`n ≤ max` and every comparison holds. -/
theorem idFilter_correct (maxId : Nat) (fs : List (Int × IdOp)) (n : Nat) :
    (match idFilter maxId (fs.map fun p => (Val.int p.1, p.2)) with
     | .ok r => inRange r n
     | .error _ => false) =
    (decide (n ≤ maxId) && fs.all (fun p => p.2.holds n p.1)) := by
  unfold idFilter
  suffices H : ∀ lo hi : Int, 0 ≤ lo → lo ≤ hi →
     (match idFilter.go lo hi (fs.map fun p => (Val.int p.1, p.2)) with
     | .ok r => inRange r n | .error _ => false) =
     (decide (lo ≤ n ∧ n ≤ hi) && fs.all (fun p => p.2.holds n p.1)) by
    rw [H 0 maxId (by omega) (by omega)]
    congr 1
    apply Bool.eq_iff_iff.2; simp
  intro lo hi hlo hle
  induction fs generalizing lo hi with
  | nil =>
    simp only [List.map_nil, idFilter.go, inRange, List.all_nil, Bool.and_true]
    apply Bool.eq_iff_iff.2; simp; omega
  | cons p fs ih =>
    obtain ⟨c, op⟩ := p
    have hs := step_correct lo hi c n op
    simp only [List.map_cons, idFilter.go, List.all_cons]
    rcases hst : step lo hi c op with ⟨lo', hi'⟩
    rw [hst] at hs
    simp only at hs ⊢
    have hn : lo ≤ lo' ∧ hi' ≤ hi := by
      cases op <;> simp only [step, Prod.mk.injEq] at hst <;> obtain ⟨rfl, rfl⟩ := hst <;> omega
    by_cases hgt : lo' > hi'
    · simp only [hgt, ite_true]
      simp only [inRange]
      rw [← Bool.and_assoc, ← hs]
      have : decide (lo' ≤ (n : Int) ∧ (n : Int) ≤ hi') = false := by simp; omega
      rw [this, Bool.false_and]
    · simp only [hgt, ite_false]
      rw [ih lo' hi' (by omega) (by omega), hs, Bool.and_assoc]

/-- A non-integer bound (null, a string, a boolean) leaves no range: the Filter it replaces never
compares it true (it raised "Node ID must be an integer" before PR #3017). -/
theorem idFilter_nonint (maxId : Nat) (v : Val) (hv : ∀ i, v ≠ .int i) (op : IdOp)
    (rest : List (Val × IdOp)) : idFilter maxId ((v, op) :: rest) = .ok none := by
  cases v with
  | int i => exact absurd rfl (hv i)
  | _ => rfl

/-! ### The bounds PR #3017 fixed -/

/-- `MATCH (n) WHERE id(n) > -1`: every node (it returned 0 rows while `-1` was cast `as u64`). -/
theorem id_gt_neg_all : idFilter 5 [(.int (-1), .gt)] = .ok (some (0, 5)) := rfl

/-- `id(n) <= -1`: no node (it returned every node, as C does, #2266). -/
theorem id_le_neg_none : idFilter 5 [(.int (-1), .le)] = .ok none := rfl

/-- `id(n) = null`: no row, as the Filter it replaces (it raised "Node ID must be an integer").
Same for `'1'`, `true`. -/
theorem id_eq_null_empty :
    idFilter 5 [(.null, .eq)] = .ok none ∧
    passB (.eq (.var ⟨0,0⟩) (.lit .null)) [(⟨0,0⟩, .int 1)] = false := ⟨rfl, rfl⟩

/-! ## reduce_count -/

structure NodeRec where
  labels : List String

def labelCount (g : List NodeRec) (l : String) : Nat := (g.filter (fun n => n.labels.contains l)).length

/-- The count is baked into a constant `Project` at plan time. -/
def reduceCount (gPlan : List NodeRec) (l : String) : Nat := labelCount gPlan l

theorem reduce_count_correct (g : List NodeRec) (l : String) :
    reduceCount g l = (g.filter (fun n => n.labels.contains l)).length := rfl

/-- `CREATE (:Z) RETURN 1 AS c UNION ALL MATCH (n:Z) RETURN count(n) AS c`
returns 0; the unreduced `count(*)` returns 1 (C behaves the same). -/
theorem reduce_count_stale :
    reduceCount [] "Z" ≠ labelCount [⟨["Z"]⟩] "Z" := by decide

/-! ## reorder_labels -/

def labelScanOk (has : String → Bool) (labels : List String) : Bool := labels.all has

def reorder (labelId : String → Nat) (labels : List String) : List String :=
  labels.mergeSort (fun a b => decide (labelId a ≤ labelId b))

theorem reorder_perm (labelId : String → Nat) (labels : List String) :
    (reorder labelId labels).Perm labels := List.mergeSort_perm _ _

/-- The label test is order-independent, so the reorder never changes results. -/
theorem reorder_preserves (has : String → Bool) (labelId : String → Nat) (labels : List String) :
    labelScanOk has (reorder labelId labels) = labelScanOk has labels := by
  unfold labelScanOk
  have hp := reorder_perm labelId labels
  apply Bool.eq_iff_iff.2
  simp only [List.all_eq_true]
  exact ⟨fun h x hx => h x (hp.mem_iff.2 hx), fun h x hx => h x (hp.mem_iff.1 hx)⟩

/-! ## replace_cartesian_with_hash_join -/

/-- Build/probe hash join (value_hash_join.rs), left-major. -/
def vhj {α β K} [BEq K] (kl : α → K) (kr : β → K) (xs : List α) (ys : List β) : List (α × β) :=
  xs.flatMap (fun x => (ys.filter (fun y => kl x == kr y)).map (fun y => (x, y)))

/-- `Filter(lhs = rhs)` over `CartesianProduct`. -/
def cpEq {α β} (eqv : α → β → Bool) (xs : List α) (ys : List β) : List (α × β) :=
  (xs.flatMap (fun x => ys.map (fun y => (x, y)))).filter (fun p => eqv p.1 p.2)

/-- **Hash join is exact** whenever key equality coincides with Cypher `=`. -/
theorem vhj_correct {α β K} [BEq K] (kl : α → K) (kr : β → K) (eqv : α → β → Bool)
    (h : ∀ x y, (kl x == kr y) = eqv x y) (xs : List α) (ys : List β) :
    vhj kl kr xs ys = cpEq eqv xs ys := by
  unfold vhj cpEq
  rw [List.filter_flatMap]
  congr 1; funext x
  rw [List.filter_map]
  congr 1
  apply List.filter_congr
  intro y _
  simp [h]

/-- Numbers: an integer or an integral float (`flt f` = the float whose value is `f`). -/
inductive Num where | int (i : Int) | flt (f : Int)
deriving DecidableEq, Repr

/-- `i as f64` for `0 ≤ i ≤ 2^54` (round half to even, spacing 2 above 2^53). -/
def toF (i : Int) : Int :=
  if 2 ^ 53 < i ∧ i ≤ 2 ^ 54 then
    (if i % 4 = 1 then i - 1 else if i % 4 = 3 then i + 1 else i)
  else i

/-- Cypher `=` as the Filter evaluates it (`compare_value`, Int vs Float via `as f64`). -/
def numEq : Num → Num → Bool
  | .int a, .int b => a == b
  | .flt a, .flt b => a == b
  | .int a, .flt b => toF a == b
  | .flt a, .int b => a == toF b

/-- `key_as_i64`: exact value. -/
def key : Num → Int
  | .int a => a
  | .flt f => f

/-- `(2^53+1) = 2^53.0` is TRUE in the Filter but the join keys differ, so the
rewrite drops the row. Confirmed: `MATCH (a:L),(b:L) WHERE a.x = b.x` misses
the pair (9007199254740993, 9007199254740992.0) that `(a.x = b.x) = true` finds. -/
theorem hash_join_drops_row :
    numEq (.int (2^53 + 1)) (.flt (2^53)) = true ∧
    vhj key key [Num.int (2^53 + 1)] [Num.flt (2^53)] = [] ∧
    cpEq numEq [Num.int (2^53 + 1)] [Num.flt (2^53)] = [(.int (2^53 + 1), .flt (2^53))] := by
  decide

-- The rounding is real (`i as f64`, evaluated natively):
#eval (Float.ofNat 9007199254740993 == 9007199254740992.0)   -- true

/-! ## push_filters_down: which children a conjunct may enter -/

inductive Kind where
  | project | aggregate | merge | argument | includePending | semiApply | antiSemiApply
  | orApplyMultiplexer | optional
  | create | delete | set | remove | commit | forEach
  | limit | skip | sort | distinct | cartesianProduct | apply | unwind | condTraverse
  | other
deriving DecidableEq, Repr

/-- push_filters_down.rs: children the filter may *not* look through
(#3582 adds the write operators `Create`, `Delete`, `Set`, `Remove`, `Commit`,
`ForEach`). -/
def excludedChild : Kind → Bool
  | .project | .aggregate | .merge | .argument | .includePending | .semiApply | .antiSemiApply
  | .orApplyMultiplexer | .optional
  | .create | .delete | .set | .remove | .commit | .forEach => true
  | _ => false

/-- The operators that are transparent for a row filter (the theorems
`push_cp_left`, `push_cp_right`, `push_extend`, `push_sort_perm` in `Basic`). -/
def transparent : Kind → Bool
  | .sort | .distinct | .cartesianProduct | .apply | .unwind | .condTraverse => true
  | _ => false

/-- The exclusion list lets the filter through `Limit` and `Skip`, which are not
transparent (`limit_not_transparent`, `skip_not_transparent`). -/
theorem limit_skip_not_excluded :
    excludedChild .limit = false ∧ transparent .limit = false ∧
    excludedChild .skip = false ∧ transparent .skip = false := by decide

/-- Routing test `conj_vars ⊆ child_vars` (push_filters_down.rs) over
bare ids (`collect_expr_variables`, `collect_subtree_variables`, mod.rs). -/
def routesIdOnly (conj : List Var) (child : List Var) : Bool :=
  conj.all (fun v => (child.map Var.id).contains v.id)

def routesScoped (conj : List Var) (child : List Var) : Bool :=
  conj.all (fun v => child.contains v)

theorem routesScoped_sound (conj child : List Var) (h : routesScoped conj child = true) :
    ∀ v ∈ conj, v ∈ child := by
  simpa [routesScoped] using h

/-- `MATCH (o1:P) CALL { MATCH (q:P) WHERE q.v = 1 RETURN q }`: `q = (0, 1)` is
routed onto the `Argument` that only carries the outer `o1 = (0, 0)` (inherited
via Case 2, push_filters_down.rs). Known root cause (#2557). -/
theorem id_only_routing_unsound :
    routesIdOnly [⟨0, 1⟩] [⟨0, 0⟩] = true ∧ routesScoped [⟨0, 1⟩] [⟨0, 0⟩] = false := by decide

end Falkor.Opt.Passes
