/-
Aggregate operator: expression analysis, accumulator zeroing/unbinding, `GroupKey`,
`args_for`, `new` (graph/src/runtime/ops/aggregate.rs).

| here | there |
| --- | --- |
| `Ex`, `Fn`, `Cst`            | `ExprIR<Variable>` tree (`DynNode`), `GraphFn`, `ExprIR::Constant` |
| `hasAgg` / `hasAggL`         | `subtree_has_aggregate` (aggregate.rs:167) |
| `analyzeAgg`                 | `AggregateOp::analyze_agg_tree` (aggregate.rs:382) |
| `analyzeKey`, `analyze`      | `AggregateOp::analyze` (aggregate.rs:313) |
| `setZero` / `setZeroL`       | `AggregateOp::set_agg_expr_zero` (aggregate.rs:1082) |
| `unbindAcc` / `unbindAccL`   | `unbind_agg_accumulators` (aggregate.rs:1230) |
| `gkEq`, `gkHash`             | `PartialEq` / `Hash for GroupKey` (aggregate.rs:90, :101) |
| `hashU64`                    | `HashU64::hash_u64` (aggregate.rs:1254 decl, :1258 impl for `Row`) |
| `argsFor`                    | `VectorizableAgg::args_for` (aggregate.rs:208) |
| `AggSt.new`                  | `AggregateOp::new` (aggregate.rs:280) |

`is_aggregate()` = `matches!(fn_type, FnType::Aggregation{..})` (functions/mod.rs:783), and the
`initial` value lives in that variant, so a function is aggregate iff `init = some z`.
-/
namespace FalkorAggOps.AggPlan

variable {Z : Type}

/-- Literal kinds that matter: `count(<non-null literal>)` is `count(*)` (aggregate.rs:449). -/
inductive Cst where
  | btrue | bfalse | int | float | str | null | other
  deriving DecidableEq

structure Fn (Z : Type) where
  isCount : Bool          -- `func.name.eq_ignore_ascii_case("count")`
  init : Option Z         -- `Some initial` iff `FnType::Aggregation`

inductive Ex (Z : Type) where
  | var (v : Nat)
  | const (c : Cst)
  | prop (attr : Nat) (cs : List (Ex Z))
  | func (f : Fn Z) (cs : List (Ex Z))
  | distinct (cs : List (Ex Z))
  | other (cs : List (Ex Z))

def Ex.children : Ex Z → List (Ex Z)
  | .var _ | .const _ => []
  | .prop _ cs | .func _ cs | .distinct cs | .other cs => cs

def isAggNode : Ex Z → Bool
  | .func f _ => f.init.isSome
  | _ => false

/-! ## `subtree_has_aggregate` (aggregate.rs:167) -/

mutual
def hasAgg : Ex Z → Bool
  | .func f cs => if f.init.isSome then true else hasAggL cs
  | .var _ | .const _ => false
  | .prop _ cs | .distinct cs | .other cs => hasAggL cs
def hasAggL : List (Ex Z) → Bool
  | [] => false
  | e :: es => hasAgg e || hasAggL es
end

-- Pre-order node list (reference spec).
mutual
def nodes : Ex Z → List (Ex Z)
  | e@(.var _) | e@(.const _) => [e]
  | e@(.prop _ cs) | e@(.func _ cs) | e@(.distinct cs) | e@(.other cs) => e :: nodesL cs
def nodesL : List (Ex Z) → List (Ex Z)
  | [] => []
  | e :: es => nodes e ++ nodesL es
end

mutual
theorem hasAgg_spec : ∀ e : Ex Z, hasAgg e = (nodes e).any isAggNode
  | .var _ => rfl
  | .const _ => rfl
  | .prop _ cs => by simp [hasAgg, nodes, isAggNode, hasAggL_spec cs]
  | .func f cs => by
      cases h : f.init.isSome <;> simp [hasAgg, nodes, isAggNode, h, hasAggL_spec cs]
  | .distinct cs => by simp [hasAgg, nodes, isAggNode, hasAggL_spec cs]
  | .other cs => by simp [hasAgg, nodes, isAggNode, hasAggL_spec cs]
theorem hasAggL_spec : ∀ es : List (Ex Z), hasAggL es = (nodesL es).any isAggNode
  | [] => rfl
  | e :: es => by simp [hasAggL, nodesL, hasAgg_spec e, hasAggL_spec es]
end

/-- **`subtree_has_aggregate`**: true iff some node of the subtree (itself included) is an
aggregate call. -/
theorem hasAgg_iff (e : Ex Z) : hasAgg e = true ↔ ∃ n ∈ nodes e, isAggNode n = true := by
  rw [hasAgg_spec]; simp

/-! ## `analyze_agg_tree` (aggregate.rs:382) -/

inductive Input (Z : Type) where
  | var (v : Nat)
  | computed (e : Ex Z)

structure VAgg (Z : Type) where
  fn : Fn Z
  input : Option (Input Z)
  accVar : Nat
  distinct : Bool        -- `distinct_idx.is_some()`
  extra : List Cst       -- `extra_args`

def isVar : Ex Z → Option Nat | .var v => some v | _ => none
def isConst : Ex Z → Option Cst | .const c => some c | _ => none
def countStarLit : Cst → Bool
  | .btrue | .int | .float | .str => true
  | _ => false

/-- Single argument (aggregate.rs:442-467). `none` = whole analysis bails. -/
def singleArg (f : Fn Z) (arg : Ex Z) : Option (Option (Input Z)) :=
  match arg with
  | .var v => some (some (.var v))
  | .const c => if countStarLit c && f.isCount then some none
                else if hasAgg arg then none else some (some (.computed arg))
  | _ => if hasAgg arg then none else some (some (.computed arg))

/-- Multi-argument tail check (aggregate.rs:478-483): children `1..n-1` all constants. -/
def constsOf : List (Ex Z) → Option (List Cst)
  | [] => some []
  | e :: es => match isConst e, constsOf es with
    | some c, some cs => some (c :: cs)
    | _, _ => none

def multiArg (arg : Ex Z) : Option (Input Z) :=
  match arg with
  | .var v => some (.var v)
  | _ => if hasAgg arg then none else some (.computed arg)

def distinctInput (ds : List (Ex Z)) : Option (Input Z) :=
  match ds with
  | [inner] => match inner with
    | .var v => some (.var v)
    | _ => if hasAgg inner then none else some (.computed inner)
  | _ => none

/-- The per-shape body of aggregate.rs:404-505, once `acc` (the last child) is known. -/
def aggBody (f : Fn Z) (acc : Nat) (cs : List (Ex Z)) : Option (VAgg Z) :=
  match cs with
  | [.distinct ds, _] => (distinctInput ds).map fun i => ⟨f, some i, acc, true, []⟩
  | [_] => some ⟨f, none, acc, false, []⟩
  | [arg, _] => (singleArg f arg).map fun i => ⟨f, i, acc, false, []⟩
  | arg :: rest =>
    match constsOf rest.dropLast, multiArg arg with
    | some xs, some i => some ⟨f, some i, acc, false, xs⟩
    | _, _ => none
  | [] => none

def analyzeAgg (e : Ex Z) : Option (VAgg Z) :=
  match e with
  | .func f cs =>
    if f.init.isNone then none else
    match cs.getLast? with
    | none => none
    | some last =>
      match isVar last with
      | none => none
      | some acc => aggBody f acc cs
  | _ => none

theorem aggBody_fields (f : Fn Z) (acc : Nat) (cs : List (Ex Z)) (a : VAgg Z)
    (h : aggBody f acc cs = some a) : a.fn = f ∧ a.accVar = acc := by
  unfold aggBody at h
  split at h
  · simp only [Option.map_eq_some_iff] at h; obtain ⟨_, _, rfl⟩ := h; exact ⟨rfl, rfl⟩
  · cases h; exact ⟨rfl, rfl⟩
  · simp only [Option.map_eq_some_iff] at h; obtain ⟨_, _, rfl⟩ := h; exact ⟨rfl, rfl⟩
  · split at h
    · cases h; exact ⟨rfl, rfl⟩
    · cases h
  · cases h

/-- The fast path is only taken for an aggregate call whose last child is the accumulator. -/
theorem analyzeAgg_shape (e : Ex Z) (a : VAgg Z) (h : analyzeAgg e = some a) :
    ∃ cs, e = .func a.fn cs ∧ a.fn.init.isSome ∧ cs.getLast? = some (.var a.accVar) := by
  unfold analyzeAgg at h
  split at h
  · rename_i f cs
    split at h
    · cases h
    · rename_i hi
      split at h
      · cases h
      · rename_i last hl
        split at h
        · cases h
        · rename_i acc hv
          have hlast : last = .var acc := by cases last <;> simp_all [isVar]
          subst hlast
          obtain ⟨h1, h2⟩ := aggBody_fields f acc cs a h
          subst h1; subst h2
          exact ⟨cs, rfl, by cases hh : a.fn.init <;> simp_all, hl⟩
  · cases h

theorem constsOf_spec (es : List (Ex Z)) (xs : List Cst) (h : constsOf es = some xs) :
    es = xs.map Ex.const := by
  induction es generalizing xs with
  | nil => simp [constsOf] at h; subst h; rfl
  | cons e es ih =>
    unfold constsOf at h
    split at h
    · rename_i c cs hc hcs
      cases h; cases e <;> simp [isConst] at hc; subst hc; simp [ih cs hcs]
    · cases h

/-- A computed (column-evaluated) input never contains an aggregate call — the guard
the comments call "defence": the per-row path keeps every nested aggregate. -/
theorem singleArg_computed (f : Fn Z) (arg x : Ex Z) (h : singleArg f arg = some (some (.computed x))) :
    x = arg ∧ hasAgg x = false := by
  unfold singleArg at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i hh; cases h; exact ⟨rfl, by simpa using hh⟩
  · split at h
    · cases h
    · rename_i hh; cases h; exact ⟨rfl, by simpa using hh⟩

theorem multiArg_computed (arg x : Ex Z) (h : multiArg arg = some (.computed x)) :
    x = arg ∧ hasAgg x = false := by
  unfold multiArg at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hh; cases h; exact ⟨rfl, by simpa using hh⟩

theorem distinctInput_computed (ds : List (Ex Z)) (x : Ex Z) (h : distinctInput ds = some (.computed x)) :
    ds = [x] ∧ hasAgg x = false := by
  unfold distinctInput at h
  split at h
  · rename_i inner
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i hh; cases h; exact ⟨rfl, by simpa using hh⟩
  · cases h

/-- `count(*)` (parsed as `count(1)`, cypher.rs:2023) needs no input column. -/
theorem countStar_no_input (f : Fn Z) (z : Z) (acc : Nat) (hf : f.init = some z) (hc : f.isCount = true) :
    analyzeAgg (.func f [.const .int, .var acc]) = some ⟨f, none, acc, false, []⟩ := by
  simp [analyzeAgg, aggBody, hf, isVar, singleArg, countStarLit, hc]

/-- `percentileDisc(x, 0.5)`: the constant tail becomes `extra_args`, in order. -/
theorem multi_extra (f : Fn Z) (z : Z) (v acc : Nat) (cs : List Cst) (hf : f.init = some z)
    (hn : cs ≠ []) :
    analyzeAgg (.func f (.var v :: (cs.map Ex.const ++ [.var acc]))) =
      some ⟨f, some (.var v), acc, false, cs⟩ := by
  have hcs : ∀ cs : List Cst, constsOf (cs.map (Ex.const (Z := Z))) = some cs := by
    intro cs; induction cs with
    | nil => rfl
    | cons c cs ih => simp [constsOf, isConst, ih]
  obtain ⟨c, cs', rfl⟩ : ∃ c cs', cs = c :: cs' := by
    cases cs with
    | nil => exact absurd rfl hn
    | cons c cs' => exact ⟨c, cs', rfl⟩
  have hl : ((Ex.var v :: (c :: cs').map Ex.const) ++ [Ex.var (Z := Z) acc]).getLast? = some (.var acc) :=
    List.getLast?_concat
  have hd : ((c :: cs').map (Ex.const (Z := Z)) ++ [Ex.var acc]).dropLast = (c :: cs').map Ex.const :=
    List.dropLast_concat
  have e : Ex.var v :: ((c :: cs').map Ex.const ++ [Ex.var (Z := Z) acc]) =
      (Ex.var v :: (c :: cs').map Ex.const) ++ [Ex.var acc] := rfl
  unfold analyzeAgg
  rw [e]
  simp only [hf, Option.isNone_some, Bool.false_eq_true, ite_false, hl, isVar]
  rw [← e]
  generalize hR : (c :: cs').map (Ex.const (Z := Z)) ++ [Ex.var acc] = R at hd
  have hR2 : ∃ x y t, R = x :: y :: t := by
    subst hR; cases cs' <;> simp
  obtain ⟨x, y, t, rfl⟩ := hR2
  unfold aggBody
  simp only [hd, hcs, multiArg]

/-! ## `analyze` (aggregate.rs:313) -/

inductive KeyKind (Z : Type) where
  | var (v : Nat)
  | prop (v : Nat) (attr : Nat)
  | computed (e : Ex Z)

/-- One key (aggregate.rs:316-370). `none` = the whole operator bails. -/
def analyzeKey (e : Ex Z) : Option (KeyKind Z) :=
  match e with
  | .var v => some (.var v)
  | .prop attr cs =>
    match cs with
    | [c] => match c with
      | .var v => some (.prop v attr)
      | _ => some (.computed e)
    | _ => none
  | _ => if hasAgg e then none else some (.computed e)

def analyze (keys aggs : List (Ex Z)) : Option (List (KeyKind Z) × List (VAgg Z)) :=
  match keys.mapM analyzeKey, aggs.mapM analyzeAgg with
  | some ks, some as => some (ks, as)
  | _, _ => none

/-- `analyze` succeeds iff every key and every aggregate expression is vectorizable; the
kinds are then in key/aggregate order (one per expression). -/
theorem analyze_spec (keys aggs : List (Ex Z)) (ks : List (KeyKind Z)) (as : List (VAgg Z)) :
    analyze keys aggs = some (ks, as) ↔
      keys.mapM analyzeKey = some ks ∧ aggs.mapM analyzeAgg = some as := by
  unfold analyze; split <;> simp_all

theorem mapM_length {α β : Type} (f : α → Option β) (xs : List α) (ys : List β)
    (h : xs.mapM f = some ys) : ys.length = xs.length := by
  induction xs generalizing ys with
  | nil => simp at h; subst h; rfl
  | cons x xs ih =>
    simp only [List.mapM_cons] at h
    cases hx : f x with
    | none => simp [hx] at h
    | some b =>
      cases hxs : xs.mapM f with
      | none => simp [hx, hxs] at h
      | some bs => simp [hx, hxs] at h; subst h; simp [ih _ hxs]

theorem analyze_lengths (keys aggs : List (Ex Z)) (ks : List (KeyKind Z)) (as : List (VAgg Z))
    (h : analyze keys aggs = some (ks, as)) : ks.length = keys.length ∧ as.length = aggs.length := by
  rw [analyze_spec] at h
  exact ⟨mapM_length _ _ _ h.1, mapM_length _ _ _ h.2⟩

/-- A key that is a bare variable or `var.prop` maps to the bulk kinds; a computed key is
aggregate-free. -/
theorem analyzeKey_spec (e : Ex Z) (k : KeyKind Z) (h : analyzeKey e = some k) :
    (∀ v, k = .var v → e = .var v) ∧ (∀ v a, k = .prop v a → e = .prop a [.var v]) ∧
    (∀ x, k = .computed x → x = e) := by
  unfold analyzeKey at h
  split at h
  · cases h; simp
  · split at h
    · split at h
      · cases h; simp
      · cases h; simp
    · cases h
  · split at h
    · cases h
    · cases h; simp

end FalkorAggOps.AggPlan
