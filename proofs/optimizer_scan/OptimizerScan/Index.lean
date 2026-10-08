/-
# Range-index utilization: `utilize_index` + the index engine it pushes into

A plan is modelled as a membership test on nodes (`Plan.sel`): each scan yields every node
at most once, so a plan's result multiset is `g.nodes.filter sel`, and two plans agree iff
their `sel` agree on every node of the graph.

| here | there |
| --- | --- |
| `P`, `V`                   | `runtime::value::Value` restricted to Bool / Int / String / temporal / list-of-scalars (strings as `Nat` codes; Float, Point, Map not modelled) |
| `eq3`, `ord3`, `Op.holds`  | Cypher three-valued `=`/`<`/… as a WHERE predicate (`runtime/value.rs` compare_value, `eval.rs`) — the reference semantics |
| `Fld`, `enc`, `arrEnc`     | `Document::set`, `index/mod.rs:716-846` (Bool→0/1, Int NUMERIC; temporal not indexed since #3076; String→TAG; list→`:numeric:arr`/`:string:arr`) |
| `Q`                        | `IndexQuery<Value>` after evaluation, `index/indexer.rs` |
| `valueToNumeric`           | `Index::value_to_numeric`, `index/mod.rs:1356-1363` |
| `lossy`                    | `Index::int_loses_f64_precision`, `index/mod.rs:1370-1372` |
| `build`/`buildAll`/`buildSome` | `Index::build_query_node`, `index/mod.rs:1476-1687` (`null_mut` ↦ `none`) |
| `idxSel`                   | `Index::query` + `Graph::get_indexed_nodes` (`index/mod.rs:1690`, `graph/graph.rs:3775`) |
| `isIndexable`, `canUtilize`| `NodeByIndexScanOp::can_utilize_index`, `runtime/ops/node_by_index_scan.rs:250-287` |
| `evalIQ`                   | `NodeByIndexScanOp::evaluate_index_query`, `node_by_index_scan.rs:98-244` (InList → `Or` of the scalar items) |
| `T`, `R`, `A`, `F`         | `ExprIR` filter trees: terms, IN right-hand sides, atoms, AND/OR roots |
| `hasT`/`hasR`              | `subtree_has_property_of`, `utilize_index.rs:547-566` |
| `firstP`/`firstR`          | `extract_attribute_from_subtree` (BFS first `Property`), `utilize_index.rs:357-368` |
| `nonIdxT`/`nonIdxA`        | `is_non_indexable_subexpr`, `utilize_index.rs:855-873` |
| `needsPost`                | `needs_post_filter`, `utilize_index.rs:802-848` |
| `buildOp`                  | `build_op_query`, `utilize_index.rs:315-355` |
| `trySingle`                | `try_single_filter_scan` + `try_in_filter_scan` + `extract_attribute_and_expression_from_filter`, `utilize_index.rs:376-418, 602-704` |
| `mergeRange`               | `merge_range_queries`, `utilize_index.rs:481-544` |
| `tryPushdown`              | `try_filter_pushdown`, `utilize_index.rs:727-783` |
| `utilize`                  | `apply_filter_pushdown` + `try_index_rewrite`, `utilize_index.rs:973-1046` |
| `Plan.idxScan` sel         | `NodeByIndexScanOp::next` (index or label-scan fallback, then extra labels), `node_by_index_scan.rs:301-328` |
-/
namespace OptimizerScan.Index

/-! ## Values and the reference (Cypher) semantics -/

inductive P | b (x : Bool) | i (x : Int) | s (x : Nat)
  deriving DecidableEq, Repr

inductive V | null | b (x : Bool) | i (x : Int) | s (x : Nat) | date (ts : Int) | arr (xs : List P)
  deriving DecidableEq, Repr

def P.toV : P → V
  | .b x => .b x
  | .i x => .i x
  | .s x => .s x

inductive Op | eq | lt | le | gt | ge
  deriving DecidableEq, Repr

def Op.flip : Op → Op
  | .eq => .eq | .lt => .gt | .le => .ge | .gt => .lt | .ge => .le

def ordI (x y : Int) : Ordering := if x < y then .lt else if x = y then .eq else .gt

def bI (x : Bool) : Int := if x then 1 else 0

def ord3 : V → V → Option Ordering
  | .i x, .i y => some (ordI x y)
  | .s x, .s y => some (ordI x y)
  | .b x, .b y => some (ordI (bI x) (bI y))
  | .date x, .date y => some (ordI x y)
  | _, _ => none

def eq3 : V → V → Option Bool
  | .null, _ => none
  | _, .null => none
  | a, b => some (a == b)

/-- `a op b` as a WHERE predicate: only `true` keeps the row. -/
def Op.holds : Op → V → V → Bool
  | .eq, a, b => eq3 a b == some true
  | .lt, a, b => ord3 a b == some .lt
  | .le, a, b => ord3 a b == some .lt || ord3 a b == some .eq
  | .gt, a, b => ord3 a b == some .gt
  | .ge, a, b => ord3 a b == some .gt || ord3 a b == some .eq

theorem ordI_swap (x y : Int) : ordI y x = (ordI x y).swap := by
  unfold ordI
  by_cases h1 : x < y <;> by_cases h2 : y < x <;> by_cases h3 : x = y <;>
    simp [h1, h2, h3, Ordering.swap] <;> omega

theorem ord3_swap (a b : V) : ord3 b a = (ord3 a b).map Ordering.swap := by
  cases a <;> cases b <;> first | rfl | exact congrArg some (ordI_swap _ _)

theorem beq_symm (a b : V) : (a == b) = (b == a) := by
  rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact ⟨fun e => e.symm, fun e => e.symm⟩

theorem eq3_swap (a b : V) : eq3 b a = eq3 a b := by
  cases a <;> cases b <;> first | rfl | exact congrArg some (beq_symm _ _)

/-- **PROVEN**: the operator flip used when the property is on the right
(`utilize_index.rs:687-698`) is sound: `b op a ↔ a (flip op) b`. -/
theorem flip_correct (op : Op) (a b : V) : op.holds b a = op.flip.holds a b := by
  cases op <;> simp only [Op.holds, Op.flip, ord3_swap a b, eq3_swap a b] <;>
    cases h : ord3 a b <;> (try rename_i o; cases o) <;> simp [Ordering.swap]

/-- A node: labels and properties. -/
structure Node where
  labels : List Nat
  prop : Nat → V

/-! ## The index engine -/

inductive Fld | num (x : Int) | tag (x : Nat)
  deriving DecidableEq, Repr

def encP : P → Fld
  | .b x => .num (bI x)
  | .i x => .num x
  | .s x => .tag x

/-- `Document::set` (`index/mod.rs:758-845`). Since #3076 (e20300436) temporal values are not
    indexed (`index/mod.rs:803`; before, they went in as NUMERIC timestamps — `IndexCex` C2). -/
def enc : V → Option Fld
  | .b x => some (.num (bI x))
  | .i x => some (.num x)
  | .s x => some (.tag x)
  | .date _ => none
  | .null => none
  | .arr _ => none

def arrEnc : V → List Fld
  | .arr xs => xs.map encP
  | _ => []

/-- Evaluated index queries (`IndexQuery<Value>`). -/
inductive Q
  | eq (k : Nat) (v : V)
  | range (k : Nat) (lo hi : Option V) (il ih : Bool)
  | contains (k : Nat) (v : V)
  | and (qs : List Q)
  | or (qs : List Q)
  deriving Repr

def valueToNumeric : V → Option Int
  | .i x => some x
  | .b x => some (bI x)
  | _ => none

/-- `int_loses_f64_precision`: `unsigned_abs & 0x7FF0_0000_0000_0000 != 0`. -/
def lossy (x : Int) : Bool := (x.natAbs &&& 0x7FF0000000000000) != 0

def geB (x c : Int) (incl : Bool) : Bool := if incl then decide (c ≤ x) else decide (c < x)
def leB (x c : Int) (incl : Bool) : Bool := if incl then decide (x ≤ c) else decide (x < c)

/-- A range bound as an optional string (`some none` = open bound, `none` = not a string). -/
def strB : Option V → Option (Option Nat)
  | none => some none
  | some (.s x) => some (some x)
  | some _ => none

def numB : Option V → Option (Option Int)
  | none => some none
  | some v => (valueToNumeric v).map some

def isStrB : Option V → Bool
  | some (.s _) => true
  | _ => false

def within (x : Int) (lo hi : Option Int) (il ih : Bool) : Bool :=
  (match lo with | none => true | some c => geB x c il) &&
  (match hi with | none => true | some c => leB x c ih)

mutual
/-- `Index::build_query_node` (`index/mod.rs:1476-1687`); `none` is a null query node. -/
def build (F : List Nat) : Q → Option (Node → Bool)
  | .eq k v =>
    if k ∈ F then
      match valueToNumeric v, v with
      | some d, _ => some (fun n => enc (n.prop k) == some (.num d))
      | none, .s x => some (fun n => enc (n.prop k) == some (.tag x))
      | none, _ => none
    else none
  | .range k lo hi il ih =>
    if k ∈ F then
      if isStrB lo || isStrB hi then
        match strB lo, strB hi with
        | some lo', some hi' =>
          -- `build_string_range_node`: equal bounds with an exclusive side select nothing
          -- (#3072, `index/mod.rs:1433-1438`); both inclusive become an exact token match
          -- (`index/mod.rs:1443-1455`).
          some (fun n =>
            if lo'.isSome ∧ lo' = hi' then (il && ih) && enc (n.prop k) == some (.tag (lo'.getD 0))
            else match enc (n.prop k) with
              | some (.tag x) => within x (lo'.map Int.ofNat) (hi'.map Int.ofNat) il ih
              | _ => false)
        | _, _ => none
      else
        match numB lo, numB hi with
        | some lo', some hi' =>
          some (fun n => match enc (n.prop k) with
            | some (.num x) => within x lo' hi' il ih
            | _ => false)
        | _, _ => none
    else none
  | .contains k v =>
    if k ∈ F then
      match v with
      | .b x => some (fun n => (arrEnc (n.prop k)).contains (.num (bI x)))
      | .i x => some (fun n => (arrEnc (n.prop k)).contains (.num x))
      | .s x => some (fun n => (arrEnc (n.prop k)).contains (.tag x))
      | _ => none
    else none
  | .and qs => (buildAll F qs).map (fun fs n => fs.all (· n))
  | .or qs => some (fun n => (buildSome F qs).any (· n))

/-- AND: one null child nulls the whole intersection (`index/mod.rs:1568-1581`). -/
def buildAll (F : List Nat) : List Q → Option (List (Node → Bool))
  | [] => some []
  | q :: qs => match build F q, buildAll F qs with
    | some f, some fs => some (f :: fs)
    | _, _ => none

/-- OR: null children are silently skipped (`index/mod.rs:1582-1594`). -/
def buildSome (F : List Nat) : List Q → List (Node → Bool)
  | [] => []
  | q :: qs => match build F q with
    | some f => f :: buildSome F qs
    | none => buildSome F qs
end

/-- The index on label `L` (its field list `idx L`), queried with `q`. -/
def bsel (F : List Nat) (q : Q) (n : Node) : Bool :=
  match build F q with
  | some f => f n
  | none => false

def idxSel (idx : Nat → List Nat) (L : Nat) (q : Q) (n : Node) : Bool :=
  L ∈ n.labels && bsel (idx L) q n

def isIndexable : V → Bool
  | .i x => !lossy x
  | .b _ | .s _ => true
  | .null => true
  | .date _ => false
  | .arr _ => false

mutual
/-- `can_utilize_index` (`node_by_index_scan.rs:250-287`). -/
def canUtilize : Q → Bool
  | .eq _ v => isIndexable v
  | .range _ lo hi _ _ => (lo.map isIndexable).getD true && (hi.map isIndexable).getD true
  | .contains _ v => match v with | .b _ | .i _ | .s _ => true | _ => false
  | .and qs => !qs.isEmpty && canAll qs
  | .or qs => !qs.isEmpty && canAll qs
def canAll : List Q → Bool
  | [] => true
  | q :: qs => canUtilize q && canAll qs
end

/-! ## Filter expressions -/

inductive T
  | prop (k : Nat)              -- `n.k` on the scanned node
  | lit (v : V)                 -- a literal (after the planner's constant folding)
  | param (v : V)               -- `$p`, or a variable bound by another operator
  | abs (t : T)                 -- a FuncInvocation (`abs(...)`, …)
  | add (a b : T)               -- `ExprIR::Add` (not a FuncInvocation!)
  deriving Repr

inductive R | list (ts : List T) | term (t : T)
  deriving Repr

inductive A
  | cmp (op : Op) (l r : T)
  | inn (l : T) (r : R)
  | opq (id : Nat)              -- any conjunct the pass cannot index (e.g. `n.k > 0` on an unindexed key)
  deriving Repr

inductive F | atom (a : A) | and (as : List A) | or (as : List A)
  deriving Repr

def evalT (n : Node) : T → V
  | .prop k => n.prop k
  | .lit v => v
  | .param v => v
  | .abs t => match evalT n t with
    | .i x => .i x.natAbs
    | _ => .null
  | .add a b => match evalT n a, evalT n b with
    | .i x, .i y => .i (x + y)
    | _, _ => .null

def inList (x : V) (ys : List V) : Bool := ys.any (fun y => eq3 x y == some true)

def evalA (opq : Nat → Node → Bool) (n : Node) : A → Bool
  | .cmp op l r => op.holds (evalT n l) (evalT n r)
  | .inn l (.list ts) => inList (evalT n l) (ts.map (evalT n))
  | .inn l (.term t) => match evalT n t with
    | .arr xs => inList (evalT n l) (xs.map P.toV)
    | _ => false
  | .opq i => opq i n

def evalF (opq : Nat → Node → Bool) (n : Node) : F → Bool
  | .atom a => evalA opq n a
  | .and as => as.all (evalA opq n)
  | .or as => as.any (evalA opq n)

/-! ## The pass -/

def hasT : T → Bool
  | .prop _ => true
  | .lit _ | .param _ => false
  | .abs t => hasT t
  | .add a b => hasT a || hasT b

def hasR : R → Bool
  | .list ts => ts.any hasT
  | .term t => hasT t

/-- First `Property` of the subtree (the Rust walk is BFS; this is leftmost-DFS, which
agrees on every subtree with at most one property per depth — all shapes used below). -/
def firstP : T → Option Nat
  | .prop k => some k
  | .lit _ | .param _ => none
  | .abs t => firstP t
  | .add a b => (firstP a).orElse (fun _ => firstP b)

def firstR : R → Option Nat
  | .list ts => ts.findSome? firstP
  | .term t => firstP t

def litFlag : V → Bool
  | .i x => lossy x
  | .arr _ => true            -- a list literal is an `ExprIR::List` node
  | _ => false                -- note: a folded `date(...)` constant is NOT flagged

def nonIdxT : T → Bool
  | .prop _ => false
  | .lit v => litFlag v
  | .param _ => true
  | .abs _ => true
  | .add a b => nonIdxT a || nonIdxT b

def nonIdxR : R → Bool
  | .list _ => true
  | .term t => nonIdxT t

def nonIdxA : A → Bool
  | .cmp _ l r => nonIdxT l || nonIdxT r
  | .inn l r => nonIdxT l || nonIdxR r
  | .opq _ => false

def scalarLit : T → Bool
  | .lit (.i x) => !lossy x
  | .lit (.b _) | .lit (.s _) => true
  | _ => false

/-- `needs_post_filter` (`utilize_index.rs:802-848`). -/
def needsPost : F → Bool
  | .atom (.inn (.prop _) (.list ts)) =>
    if !ts.isEmpty && ts.all scalarLit then false else true
  | .atom a => nonIdxA a
  | .and as => as.any nonIdxA
  | .or as => as.any nonIdxA

/-- Unevaluated index queries (`IndexQuery<QueryExpr<Variable>>`). -/
inductive IQ
  | eq (k : Nat) (t : T)
  | range (k : Nat) (lo hi : Option T) (il ih : Bool)
  | inList (k : Nat) (r : R)
  | contains (k : Nat) (t : T)
  | and (qs : List IQ)
  | or (qs : List IQ)
  deriving Repr

/-- `build_op_query` (`utilize_index.rs:315-355`). -/
def buildOp (op : Op) (k : Nat) (c : T) : IQ :=
  match op with
  | .eq => .eq k c
  | .gt => .range k (some c) none false false
  | .ge => .range k (some c) none true false
  | .lt => .range k none (some c) false false
  | .le => .range k none (some c) false true

def findLabel (idx : Nat → List Nat) (ls : List Nat) (k : Nat) : Option Nat :=
  ls.find? (fun L => k ∈ idx L)

def nestedList : R → Bool
  | .list ts => ts.any (fun t => match t with | .lit (.arr _) => true | _ => false)
  | .term _ => false

def anyPropR : R → Bool := hasR

/-- `try_single_filter_scan` / `try_in_filter_scan` (`utilize_index.rs:602-704`). -/
def trySingle (idx : Nat → List Nat) (ls : List Nat) : A → Option (Nat × IQ)
  | .inn l r =>
    match hasT l, hasR r with
    | true, false => do
      let k ← firstP l
      let L ← findLabel idx ls k
      if nestedList r || anyPropR r then none else some (L, .inList k r)
    | false, true => do
      let k ← firstR r
      let L ← findLabel idx ls k
      some (L, .contains k l)
    | _, _ => none
  | .cmp op l r =>
    match hasT l, hasT r with
    | true, false => do
      let k ← firstP l
      let L ← findLabel idx ls k
      match l with
      | .prop k' => some (L, buildOp op k' r)
      | _ => none
    | false, true => do
      let k ← firstP r
      let L ← findLabel idx ls k
      match r with
      | .prop k' => some (L, buildOp op.flip k' l)
      | _ => none
    | _, _ => none
  | .opq _ => none

/-- `merge_range_queries` (`utilize_index.rs:481-544`). -/
def mergeRange : IQ → IQ → IQ
  | .range k lo hi il ih, .range k' lo' hi' il' ih' =>
    if k = k' then
      if (lo.isSome && lo'.isSome) || (hi.isSome && hi'.isSome) then
        .and [.range k lo hi il ih, .range k' lo' hi' il' ih']
      else
        let (l, i) := if lo.isSome then (lo, il) else (lo', il')
        let (h, j) := if hi.isSome then (hi, ih) else (hi', ih')
        .range k l h i j
    else .and [.range k lo hi il ih, .range k' lo' hi' il' ih']
  | a, b => .and [a, b]

/-- Fold one pushed conjunct into the running query: the first one fixes the label, later
ones are merged into it on that label (`utilize_index.rs:739-744`). -/
def mergeInto (m : Option (Nat × IQ)) (L : Nat) (q : IQ) : Option (Nat × IQ) :=
  match m with
  | none => some (L, q)
  | some (L0, q0) => some (L0, mergeRange q0 q)

/-- The AND branch of `try_filter_pushdown` (`utilize_index.rs:733-750`). -/
def pushAnd (idx : Nat → List Nat) (ls : List Nat) :
    List A → Option (Nat × IQ) → List A → Option (Nat × IQ) × List A
  | [], m, rem => (m, rem)
  | a :: as, m, rem =>
    match trySingle idx ls a with
    | some (L, q) => pushAnd idx ls as (mergeInto m L q) rem
    | none => pushAnd idx ls as m (rem ++ [a])

/-- The OR branch: every disjunct must convert (`utilize_index.rs:751-768`). -/
def pushOr (idx : Nat → List Nat) (ls : List Nat) : List A → Option (Option Nat × List IQ)
  | [] => some (none, [])
  | a :: as => match trySingle idx ls a, pushOr idx ls as with
    | some (L, q), some (_, qs) => some (some L, q :: qs)   -- label of the first disjunct
    | _, _ => none

def isArrayContains : A → Bool
  | .inn l r => !hasT l && hasR r
  | _ => false

/-- `try_filter_pushdown` (`utilize_index.rs:727-783`). -/
def tryPushdown (idx : Nat → List Nat) (ls : List Nat) : F → Option (Nat × IQ × List A)
  | .and as =>
    match pushAnd idx ls as none [] with
    | (some (L, q), rem) => some (L, q, rem)
    | (none, _) => none
  | .or as =>
    match pushOr idx ls as with
    | some (some L, qs) => some (L, .or qs, [])
    | _ => none
  | .atom a =>
    match trySingle idx ls a with
    | some (L, q) => some (L, q, if isArrayContains a then [a] else [])
    | none => none

/-! ## Plans and their semantics -/

def evalC : T → V := evalT { labels := [], prop := fun _ => .null }

def listVals : R → List V
  | .list ts => ts.map evalC
  | .term t => match evalC t with | .arr xs => xs.map P.toV | _ => []

def isPrim : V → Bool | .b _ | .i _ | .s _ => true | _ => false

mutual
/-- `evaluate_index_query` (`node_by_index_scan.rs:98-244`). -/
def evalIQ : IQ → Q
  | .eq k t => .eq k (evalC t)
  | .range k lo hi il ih => .range k (lo.map evalC) (hi.map evalC) il ih
  | .inList k r => .or (((listVals r).filter isPrim).map (.eq k))
  | .contains k t => .contains k (evalC t)
  | .and qs => .and (evalIQs qs)
  | .or qs => .or (evalIQs qs)
def evalIQs : List IQ → List Q
  | [] => []
  | q :: qs => evalIQ q :: evalIQs qs
end

inductive Plan
  | labelScan (ls : List Nat)
  | idxScan (ls : List Nat) (L : Nat) (q : IQ)
  | filter (f : F) (p : Plan)

def hasAll (ls : List Nat) (n : Node) : Bool := ls.all (· ∈ n.labels)

def Plan.sel (idx : Nat → List Nat) (opq : Nat → Node → Bool) : Plan → Node → Bool
  | .labelScan ls, n => hasAll ls n
  | .idxScan ls L q, n =>
    let qv := evalIQ q
    (if canUtilize qv then idxSel idx L qv n else hasAll ls n) && hasAll (ls.erase L) n
  | .filter f p, n => Plan.sel idx opq p n && evalF opq n f

/-- `apply_filter_pushdown` driven by `try_index_rewrite` (`utilize_index.rs:973-1046`). -/
def utilize (idx : Nat → List Nat) (ls : List Nat) (f : F) : Plan :=
  match tryPushdown idx ls f with
  | none => .filter f (.labelScan ls)
  | some (L, q, rem) =>
    let keep := needsPost f
    let scan := Plan.idxScan ls L q
    if rem.isEmpty then (if keep then .filter f scan else scan)
    else if keep then .filter f scan else .filter (.and rem) scan

/-- What the rewrite must preserve: `Filter(f)` over `NodeByLabelScan(ls)`. -/
def reference (ls : List Nat) (f : F) : Plan := .filter f (.labelScan ls)

end OptimizerScan.Index
