/-
# The remaining passes

| here | there |
| --- | --- |
| `out`, `travCollapsed` | `CondTraverse` with `emit_relationship = false` (`runtime/ops/cond_traverse.rs`): one row per distinct `(src, dst)` |
| `unfused`, `fused`     | two chained anonymous CTs vs the fused `F·A1·A2` chain (`optimizer/fuse_anonymous_traverse.rs:194-287`) |
| `optionalOp`, `optCT`  | `Optional` (`runtime/ops/optional.rs`) vs `CondTraverse{optional: true}` (`fuse_optional_traverse.rs:105-156`) |
| `IdOp`, `step`, `idFilter` | `Runtime::evaluate_id_filter` (`graph/src/runtime/runtime.rs:1309-1363`), the operator `utilize_node_by_id` (`optimizer/utilize_node_by_id.rs:107-155`) introduces |
| `wrap`                 | Rust `id as u64` on an `i64` (two's-complement reinterpretation) |
| `flip`                 | the operator flip in `get_id_filter` (`utilize_node_by_id.rs:77-84`) |
| `reduceCount`          | `reduce_count` (`optimizer/reduce_count.rs:54-189`): count read from the graph at *plan* time |
| `labelScanOk`, `reorder` | `NodeByLabelScan` label test and `reorder_labels` (`optimizer/reorder_labels.rs:16-47`) stable sort by label id |
| `vhj`, `cpEq`          | `ValueHashJoin` (`runtime/ops/value_hash_join.rs`, `key_as_i64` l.153) vs `Filter(=)` over `CartesianProduct` (`replace_cartesian_with_hash_join.rs:101-217`) |
| `toF`                  | `i as f64` in `Value::compare_value` for Int vs Float, exact in `[0, 2^54]` |
| `excludedChild`        | the operator list at `push_filters_down.rs:165-178` |
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

/-- Rust `id as u64`. -/
def wrap (i : Int) : Nat := (i % 2 ^ 64).toNat

/-- One iteration of the `for (expr, op) in filter` loop (runtime.rs:1326-1360). -/
def step (lo hi c : Nat) : IdOp → Option (Nat × Nat)
  | .eq => if c < lo ∨ c > hi then none else some (c, c)
  | .gt => if c ≥ hi then none else some (max lo (c + 1), hi)
  | .ge => if c > hi then none else some (max lo c, hi)
  | .lt => if c ≤ lo then none else some (lo, min hi (c - 1))
  | .le => if c < lo then none else some (lo, min hi c)

/-- The whole function. `Ok(None)` = empty; a non-integer value is an error. -/
def idFilter (maxId : Nat) (fs : List (Val × IdOp)) : Except Err (Option (Nat × Nat)) :=
  go 0 maxId fs
where
  go (lo hi : Nat) : List (Val × IdOp) → Except Err (Option (Nat × Nat))
    | [] => .ok (some (lo, hi))
    | (v, op) :: rest =>
      match v with
      | .int i =>
        match step lo hi (wrap i) op with
        | none => .ok none
        | some (lo', hi') => go lo' hi' rest
      | _ => .error .typeMismatch     -- "Node ID must be an integer"

def inRange (r : Option (Nat × Nat)) (n : Nat) : Bool :=
  match r with
  | none => false
  | some (lo, hi) => decide (lo ≤ n ∧ n ≤ hi)

/-- `+ 1` / `- 1` never overflow/underflow: the guards run first. -/
theorem step_no_wrap (lo hi c : Nat) (hhi : hi < 2 ^ 64) (op : IdOp) (lo' hi' : Nat)
    (h : step lo hi c op = some (lo', hi')) :
    (op = .gt → c + 1 < 2 ^ 64) ∧ (op = .lt → 1 ≤ c) := by
  cases op <;> simp [step] at h ⊢ <;> omega

theorem step_correct (lo hi c n : Nat) (op : IdOp) :
    inRange (step lo hi c op) n = (decide (lo ≤ n ∧ n ≤ hi) && op.holds n c) := by
  cases op <;> simp only [step] <;> split <;> simp only [inRange, IdOp.holds] <;>
    apply Bool.eq_iff_iff.2 <;> simp <;> omega

theorem wrap_of_nonneg (i : Int) (h0 : 0 ≤ i) (h1 : i < 2 ^ 64) : (wrap i : Int) = i := by
  unfold wrap
  rw [Int.emod_eq_of_lt h0 h1]
  omega

/-- **evaluate_id_filter is correct for integer arguments in `[0, 2^63)`**:
a node id `n` is in the seek range iff `n ≤ max_node_id` and every comparison
holds. -/
theorem idFilter_correct (maxId : Nat) (fs : List (Int × IdOp))
    (hfs : ∀ p ∈ fs, 0 ≤ p.1 ∧ p.1 < 2 ^ 63) (n : Nat) :
    (match idFilter maxId (fs.map fun p => (Val.int p.1, p.2)) with
     | .ok r => inRange r n
     | .error _ => false) =
    (decide (n ≤ maxId) && fs.all (fun p => p.2.holds n p.1)) := by
  unfold idFilter
  suffices H : ∀ lo hi, (match idFilter.go lo hi (fs.map fun p => (Val.int p.1, p.2)) with
     | .ok r => inRange r n | .error _ => false) =
     (decide (lo ≤ n ∧ n ≤ hi) && fs.all (fun p => p.2.holds n p.1)) by
    rw [H 0 maxId]; simp
  intro lo hi
  induction fs generalizing lo hi with
  | nil => simp [idFilter.go, inRange]
  | cons p fs ih =>
    obtain ⟨c, op⟩ := p
    have hc := hfs (c, op) (by simp)
    have hop : op.holds n (wrap c) = op.holds n c := by
      have hw : (wrap c : Int) = c := wrap_of_nonneg c hc.1 (by omega)
      rw [hw]
    have hs := step_correct lo hi (wrap c) n op
    rw [hop] at hs
    simp only [List.map_cons, idFilter.go, List.all_cons]
    cases hst : step lo hi (wrap c) op with
    | none =>
      rw [hst] at hs
      simp only [hst, inRange] at hs ⊢
      rw [← Bool.and_assoc, ← hs, Bool.false_and]
    | some st =>
      obtain ⟨lo', hi'⟩ := st
      rw [hst] at hs
      simp only [hst]
      rw [ih (fun p hp => hfs p (by simp [hp])) lo' hi']
      simp only [inRange] at hs
      rw [hs, Bool.and_assoc]

/-! ### Counterexamples (confirmed on the server, see the root file) -/

/-- `MATCH (n) WHERE id(n) > -1`: Rust returns 0 rows, C and openCypher all. -/
theorem id_gt_neg_empty (maxId : Nat) (h : maxId < 2 ^ 64) :
    idFilter maxId [(.int (-1), .gt)] = .ok none := by
  have hw : wrap (-1) = 2 ^ 64 - 1 := rfl
  simp only [idFilter, idFilter.go, hw, step]
  rw [ite_eq_left_iff.mpr (by omega)]

/-- `id(n) <= -1`: every node (issue #2266, C too). -/
theorem id_le_neg_all :
    idFilter 5 [(.int (-1), .le)] = .ok (some (0, 5)) := rfl

/-- `id(n) = null`: the Filter it replaces drops the row, the seek errors
("Node ID must be an integer"). Same for `1.0`, `'1'`, `true`. -/
theorem id_eq_null_errors :
    idFilter 5 [(.null, .eq)] = .error .typeMismatch ∧
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
  | limit | skip | sort | distinct | cartesianProduct | apply | unwind | condTraverse
  | other
deriving DecidableEq, Repr

/-- push_filters_down.rs:165-178: children the filter may *not* look through. -/
def excludedChild : Kind → Bool
  | .project | .aggregate | .merge | .argument | .includePending | .semiApply | .antiSemiApply
  | .orApplyMultiplexer | .optional => true
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

/-- Routing test `conj_vars ⊆ child_vars` (push_filters_down.rs:268-280) over
bare ids (`collect_expr_variables`, `collect_subtree_variables`, mod.rs:91-109). -/
def routesIdOnly (conj : List Var) (child : List Var) : Bool :=
  conj.all (fun v => (child.map Var.id).contains v.id)

def routesScoped (conj : List Var) (child : List Var) : Bool :=
  conj.all (fun v => child.contains v)

theorem routesScoped_sound (conj child : List Var) (h : routesScoped conj child = true) :
    ∀ v ∈ conj, v ∈ child := by
  simpa [routesScoped] using h

/-- `MATCH (o1:P) CALL { MATCH (q:P) WHERE q.v = 1 RETURN q }`: `q = (0, 1)` is
routed onto the `Argument` that only carries the outer `o1 = (0, 0)` (inherited
via Case 2, push_filters_down.rs:214-261). Known root cause (#2557). -/
theorem id_only_routing_unsound :
    routesIdOnly [⟨0, 1⟩] [⟨0, 0⟩] = true ∧ routesScoped [⟨0, 1⟩] [⟨0, 0⟩] = false := by decide

end Falkor.Opt.Passes
