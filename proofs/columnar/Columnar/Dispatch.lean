/-!
# `VectorEval` dispatch and its leaves (`runtime/vector_expr.rs`, `vectorized.rs:70`)
(origin/main 3fec7d7c9)

The operator lanes (`eval_and_or`, `eval_case`, `eval_function`, `arithmetic`,
`compare_columns`, `negate_bools`, `eval_variable`) are proved equal to the per-row
evaluator in `VectorExpr`/`Arith`/`Lanes` (and proofs/expr_semantics). This file closes
the tree walk around them:

| Lean | Rust |
| --- | --- |
| `Kind`, `Node` | the `ExprIR` variants `eval` inspects, with children |
| `fromExprIR` | `CmpOp::from_expr_ir` (vectorized.rs:70) |
| `Arms`, `eval` | `VectorEval::eval` (vector_expr.rs:211-271) — arms in source order, guards included |
| `evalValues` | `eval_values` (:286) |
| `propertyValues` | `property_values` (:357) |
| `hasLabels` | `eval_has_labels` (:642) |
| `perRow` | `eval_per_row` (:706) |
| `VectorEval.new` | `new` (:202) |

**`eval_sound`** (main theorem): if every arm is sound *given sound children* — which is
what the lane theorems state — then `eval` is sound for every tree: whenever it returns a
column, entry `i` equals the per-row evaluator on row `rows[i]`. The proof is the
structural induction the Rust recursion performs; the guards (`num_children() == 1/2`, the
`from_expr_ir` check, the function-invocation filter, the `hasLabels` attempt) route each
node to exactly one arm or to `eval_per_row`. Also: `evalValues_sound`,
`propertyValues_spec`/`_none_iff`, `hasLabels_sound` (incl. the unregistered-label
shortcut and the "type-check the whole list first" order, test11), `perRow_spec` (first
row error = the batch error), `fromExprIR_*` (exactly the six comparisons; `flip`
commutes with swapping operands).
-/

namespace Columnar.Dispatch

/-! ## `CmpOp::from_expr_ir` -/

inductive CmpOp where | eq | neq | lt | le | gt | ge
  deriving DecidableEq

def CmpOp.flip : CmpOp → CmpOp
  | .eq => .eq | .neq => .neq | .lt => .gt | .le => .ge | .gt => .lt | .ge => .le

variable {Val : Type}

/-- The `ExprIR` shapes `VectorEval::eval` distinguishes. -/
inductive Kind (Val : Type) where
  | const (v : Val) | param (name : String) | var (id : Nat) | prop (attr : String)
  | paren | and | or | not
  | add | sub | mul | div | modulo
  | case (hasSubject : Bool)
  | eq | neq | lt | le | gt | ge
  | func (name : String) (aggregation structFn write : Bool)
  | list | distinct
  | other (tag : Nat)

inductive Node (Val : Type) where
  | mk (k : Kind Val) (cs : List (Node Val))

def Node.kind : Node Val → Kind Val | .mk k _ => k
def Node.cs : Node Val → List (Node Val) | .mk _ cs => cs

/-- `CmpOp::from_expr_ir`. -/
def fromExprIR : Kind Val → Option CmpOp
  | .eq => some .eq | .neq => some .neq | .lt => some .lt | .le => some .le
  | .gt => some .gt | .ge => some .ge | _ => none

def isCmp : Kind Val → Bool
  | .eq | .neq | .lt | .le | .gt | .ge => true
  | _ => false

theorem fromExprIR_isSome (k : Kind Val) : (fromExprIR k).isSome = isCmp k := by
  cases k <;> rfl

/-- Ordering semantics of a comparison IR node and of a `CmpOp`. -/
def kindSem : Kind Val → Ordering → Bool
  | .eq, o => o == .eq | .neq, o => o != .eq | .lt, o => o == .lt
  | .le, o => o != .gt | .gt, o => o == .gt | .ge, o => o != .lt | _, _ => false
def opSem : CmpOp → Ordering → Bool
  | .eq, o => o == .eq | .neq, o => o != .eq | .lt, o => o == .lt
  | .le, o => o != .gt | .gt, o => o == .gt | .ge, o => o != .lt

/-- `from_expr_ir` preserves meaning … -/
theorem fromExprIR_sem (k : Kind Val) (op : CmpOp) (h : fromExprIR k = some op) (o : Ordering) :
    opSem op o = kindSem k o := by
  cases k <;> simp [fromExprIR] at h <;> subst h <;> rfl

/-- … and `flip` is the operand swap (`a op b = b (flip op) a`). -/
theorem flip_sem (op : CmpOp) (o : Ordering) : opSem op.flip o.swap = opSem op o := by
  cases op <;> cases o <;> rfl

/-! ## The tree walk -/

/-- Everything `eval` delegates to. `C` = `ExprColumn`; `Row` = a row index. Arms that
recurse receive the recursive evaluator restricted to the node's children. -/
structure Arms (Val C : Type) where
  scalar : Val → C
  params : String → Option Val
  evalVariable : Nat → List Nat → C
  evalProperty : String → Node Val → List Nat → Option C
  perRow : Node Val → List Nat → Except String C
  andOr : (Node Val → List Nat → Except String C) → Node Val → List Nat → Bool → Except String C
  negate : C → Nat → Except String C
  arith : Kind Val → C → C → Nat → Except String C
  case : (Node Val → List Nat → Except String C) → Bool → Node Val → List Nat → Except String C
  compare : C → C → CmpOp → Nat → C
  hasLabels : Node Val → List Nat → Option C
  evalFunction : (Node Val → List Nat → Except String C) → Node Val → List Nat → Except String C

variable {C : Type} (A : Arms Val C)

def isArith : Kind Val → Bool
  | .add | .sub | .mul | .div | .modulo => true
  | _ => false

def isDistinct : Node Val → Bool
  | .mk .distinct _ => true
  | _ => false

/-- `VectorEval::eval` (:211). Each `if` is one guarded arm, in source order; a failed
guard falls through to the next arm, the last being `_ => eval_per_row`. -/
noncomputable def eval : Node Val → List Nat → Except String C
  | .mk k cs, rows =>
    let n := Node.mk k cs
    let len := rows.length
    -- the recursive evaluator, available on strictly smaller trees (children, in particular)
    let sub : Node Val → List Nat → Except String C := fun c r =>
      if h : sizeOf c < sizeOf (Node.mk k cs) then eval c r else .error "not a subtree"
    match k with
    | .const v => .ok (A.scalar v)
    | .param name =>
      (match A.params name with
       | some v => .ok (A.scalar v)
       | none => .error s!"Parameter {name} not found")
    | .var id => .ok (A.evalVariable id rows)
    | .prop attr =>
      (match cs with
       | [c] => (match A.evalProperty attr c rows with
         | some col => .ok col
         | none => A.perRow n rows)
       | _ => A.perRow n rows)
    | .paren =>
      (match cs with
       | [c] => sub c rows
       | _ => A.perRow n rows)
    | .and => A.andOr sub n rows false
    | .or => A.andOr sub n rows true
    | .not =>
      (match cs with
       | [c] => do let x ← sub c rows; A.negate x len
       | _ => A.perRow n rows)
    | .case hs => A.case sub hs n rows
    | .func name agg sf w =>
      if !agg && !sf && !w && !cs.any isDistinct then
        if name = "hasLabels" then
          (match A.hasLabels n rows with
           | some col => .ok col
           | none => A.evalFunction sub n rows)
        else A.evalFunction sub n rows
      else A.perRow n rows
    | k' =>
      (match cs with
       | [l, r] =>
         if isArith k' then do
           let x ← sub l rows; let y ← sub r rows; A.arith k' x y len
         else (match fromExprIR k' with
           | some op => do let x ← sub l rows; let y ← sub r rows; pure (A.compare x y op len)
           | none => A.perRow n rows)
       | _ => A.perRow n rows)
termination_by n => sizeOf n
decreasing_by all_goals exact h

/-- The per-row semantics and the soundness contract. -/
structure RowSem (Val C : Type) where
  get : C → Nat → Val
  rowEval : Node Val → Nat → Except String Val

variable (S : RowSem Val C)

/-- Column `col` answers `n` on `rows`: entry `i` = per-row value of row `rows[i]`. -/
def Sound (n : Node Val) (rows : List Nat) (col : C) : Prop :=
  ∀ i (h : i < rows.length), S.rowEval n rows[i] = .ok (S.get col i)

/-- A child evaluator is sound on the children of `n`. -/
def SubSound (n : Node Val) (ev : Node Val → List Nat → Except String C) : Prop :=
  ∀ c ∈ n.cs, ∀ r col, ev c r = .ok col → Sound S c r col

/-- What the leaves and lanes guarantee (each proved in its own module, or by the runtime
contract for parameters / property reads). -/
structure Lawful : Prop where
  scalar : ∀ v n rows, (∀ r ∈ rows, S.rowEval n r = .ok v) → Sound S n rows (A.scalar v)
  const : ∀ v cs r, S.rowEval (.mk (.const v) cs) r = .ok v
  param : ∀ name v cs r, A.params name = some v → S.rowEval (.mk (.param name) cs) r = .ok v
  var : ∀ id cs rows, Sound S (.mk (.var id) cs) rows (A.evalVariable id rows)
  prop : ∀ attr c rows col, A.evalProperty attr c rows = some col → Sound S (.mk (.prop attr) [c]) rows col
  perRow : ∀ n rows col, A.perRow n rows = .ok col → Sound S n rows col
  paren : ∀ c r, S.rowEval (.mk .paren [c]) r = S.rowEval c r
  andOr : ∀ ev n rows b col, SubSound S n ev → A.andOr ev n rows b = .ok col → Sound S n rows col
  negate : ∀ c rows x col, Sound S c rows x → A.negate x rows.length = .ok col → Sound S (.mk .not [c]) rows col
  arith : ∀ k l r rows x y col, isArith k → Sound S l rows x → Sound S r rows y →
    A.arith k x y rows.length = .ok col → Sound S (.mk k [l, r]) rows col
  case : ∀ ev hs n rows col, SubSound S n ev → A.case ev hs n rows = .ok col → Sound S n rows col
  compare : ∀ k op l r rows x y, fromExprIR k = some op → Sound S l rows x → Sound S r rows y →
    Sound S (.mk k [l, r]) rows (A.compare x y op rows.length)
  hasLabels : ∀ n rows col, A.hasLabels n rows = some col → Sound S n rows col
  func : ∀ ev n rows col, SubSound S n ev → A.evalFunction ev n rows = .ok col → Sound S n rows col

theorem bind_ok {α β} {x : Except String α} {f : α → Except String β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

/-- **`eval` is sound for every expression tree.** -/
theorem eval_sound (L : Lawful A S) : ∀ (n : Node Val) (rows : List Nat) (col : C),
    eval A n rows = .ok col → Sound S n rows col
  | .mk k cs, rows, col, h => by
    have sub_sound : SubSound S (.mk k cs)
        (fun c r => if _ : sizeOf c < sizeOf (Node.mk k cs) then eval A c r else .error "not a subtree") := by
      intro c hc r col' hc'
      simp only [Node.cs] at hc
      have hlt : sizeOf c < sizeOf (Node.mk k cs) := by
        have := List.sizeOf_lt_of_mem hc; simp; omega
      simp only [hlt, dite_true] at hc'
      exact eval_sound L c r col' hc'
    unfold eval at h
    cases k with
    | const v =>
      simp only [Except.ok.injEq] at h; subst h
      exact L.scalar v _ rows (fun r _ => L.const v cs r)
    | param name =>
      simp only at h
      split at h
      · rename_i v hv; simp only [Except.ok.injEq] at h; subst h
        exact L.scalar v _ rows (fun r _ => L.param name v cs r hv)
      · simp at h
    | var id => simp only [Except.ok.injEq] at h; subst h; exact L.var id cs rows
    | prop attr =>
      simp only at h
      split at h
      · split at h
        · rename_i hp; simp only [Except.ok.injEq] at h; rw [← h]; exact L.prop attr _ rows _ hp
        · exact L.perRow _ rows col h
      · exact L.perRow _ rows col h
    | paren =>
      simp only at h
      split at h
      · rename_i c
        have hs := sub_sound c (by simp [Node.cs]) rows col h
        intro i hi; rw [L.paren]; exact hs i hi
      · exact L.perRow _ rows col h
    | and => exact L.andOr _ _ rows false col sub_sound h
    | or => exact L.andOr _ _ rows true col sub_sound h
    | not =>
      simp only at h
      split at h
      · rename_i c
        obtain ⟨x, hx, hn⟩ := bind_ok h
        exact L.negate c rows x col (sub_sound c (by simp [Node.cs]) rows x hx) hn
      · exact L.perRow _ rows col h
    | case hs => exact L.case _ hs _ rows col sub_sound h
    | func name agg sf w =>
      simp only at h
      split at h
      · split at h
        · split at h
          · rename_i c hc; simp only [Except.ok.injEq] at h; subst h; exact L.hasLabels _ rows _ hc
          · exact L.func _ _ rows col sub_sound h
        · exact L.func _ _ rows col sub_sound h
      · exact L.perRow _ rows col h
    | list | distinct | other _ | add | sub | mul | div | modulo | eq | neq | lt | le | gt | ge =>
      simp only at h
      split at h
      · rename_i l r
        have hl := fun x hx => sub_sound l (by simp [Node.cs]) rows x hx
        have hr := fun y hy => sub_sound r (by simp [Node.cs]) rows y hy
        split at h
        · rename_i ha
          obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
          exact L.arith _ l r rows x y col ha (hl x hx) (hr y hy) h
        · split at h
          · rename_i op hop
            obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
            simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
            exact L.compare _ op l r rows x y hop (hl x hx) (hr y hy)
          · exact L.perRow _ rows col h
      · exact L.perRow _ rows col h
termination_by n => sizeOf n
decreasing_by all_goals exact hlt

/-- A comparison node with two children always takes the `compare_columns` arm. -/
theorem eval_cmp (k : Kind Val) (l r : Node Val) (rows : List Nat) (op : CmpOp)
    (h : fromExprIR k = some op) (x y : C)
    (hx : eval A l rows = .ok x) (hy : eval A r rows = .ok y) :
    eval A (.mk k [l, r]) rows = .ok (A.compare x y op rows.length) := by
  have hl : ∀ a, sizeOf l < a + (1 + sizeOf l + (1 + sizeOf r + 1)) := by intro a; omega
  have hr : ∀ a, sizeOf r < a + (1 + sizeOf l + (1 + sizeOf r + 1)) := by intro a; omega
  cases k <;> simp [fromExprIR] at h <;> subst h <;> unfold eval <;>
    simp [isArith, fromExprIR, hl, hr, hx, hy, bind, Except.bind, pure, Except.pure]

/-- A missing parameter is the one error `eval` raises itself. -/
theorem eval_param_missing (name : String) (cs : List (Node Val)) (rows : List Nat)
    (h : A.params name = none) : eval A (.mk (.param name) cs) rows = .error s!"Parameter {name} not found" := by
  simp [eval, h]

/-! ## `property_values`, `eval_values`, `eval_has_labels`, `eval_per_row` -/

/-- A batch column, as `property_values`/`eval_has_labels` see it. -/
inductive Col where
  | nodeIds (ids : List Nat) | relIds (ids : List Nat) | other

structure Store (Val : Type) where
  column : Nat → Col                         -- `batch.column(var.id)`
  matNode : List Nat → String → List Val     -- `materialize_node_property_values`
  matRel : List Nat → String → List Val
  nodeProp : Nat → String → Val              -- per-row `n.attr`
  relProp : Nat → String → Val
  labelId : String → Option Nat
  hasLabelId : Nat → Nat → Bool
  hasLabelName : Nat → String → Bool         -- per-row `node_has_label`

variable (G : Store Val)

/-- The bulk reads are the per-row reads, row by row (runtime contract). -/
structure Store.Lawful : Prop where
  matNode : ∀ ids a, G.matNode ids a = ids.map (G.nodeProp · a)
  matRel : ∀ ids a, G.matRel ids a = ids.map (G.relProp · a)
  label : ∀ id name, G.hasLabelName id name = match G.labelId name with
    | some l => G.hasLabelId id l
    | none => false                           -- an unregistered label is on no node

def gather (ids : List Nat) (rows : List Nat) : List Nat := rows.map (fun r => ids.getD r 0)

/-- `property_values` (:357). -/
def propertyValues (attr : String) (child : Node Val) (rows : List Nat) : Option (List Val) :=
  match child with
  | .mk (.var id) _ => match G.column id with
    | .nodeIds ids => some (G.matNode (gather ids rows) attr)
    | .relIds ids => some (G.matRel (gather ids rows) attr)
    | .other => none
  | _ => none

theorem propertyValues_none_iff (attr : String) (child : Node Val) (rows : List Nat) :
    propertyValues G attr child rows = none ↔
      ∀ id cs, child = .mk (.var id) cs → G.column id = .other := by
  rcases child with ⟨k, cs⟩
  cases k <;> simp [propertyValues]
  split <;> simp_all

/-- Row `i` of the bulk fetch is the property of the entity at `rows[i]`. -/
theorem propertyValues_spec (L : G.Lawful) (attr : String) (id : Nat) (cs : List (Node Val))
    (rows : List Nat) (vs : List Val) (h : propertyValues G attr (.mk (.var id) cs) rows = some vs) :
    vs.length = rows.length ∧ ∀ i (hi : i < rows.length),
      (∃ ids, G.column id = .nodeIds ids ∧ vs[i]? = some (G.nodeProp (ids.getD rows[i] 0) attr)) ∨
      (∃ ids, G.column id = .relIds ids ∧ vs[i]? = some (G.relProp (ids.getD rows[i] 0) attr)) := by
  simp only [propertyValues] at h
  cases hc : G.column id with
  | nodeIds ids =>
    simp only [hc, Option.some.injEq] at h; subst h
    refine ⟨by simp [L.matNode, gather], fun i hi => .inl ⟨ids, rfl, ?_⟩⟩
    simp [L.matNode, gather, hi]
  | relIds ids =>
    simp only [hc, Option.some.injEq] at h; subst h
    refine ⟨by simp [L.matRel, gather], fun i hi => .inr ⟨ids, rfl, ?_⟩⟩
    simp [L.matRel, gather, hi]
  | other => simp [hc] at h

/-- `eval_values` (:286): property fast path, variable passthrough, else
`eval(..).into_values(len)`. -/
noncomputable def evalValues (intoValues : C → Nat → List Val) (valueAt : Nat → Nat → Option Val) (null : Val)
    (n : Node Val) (rows : List Nat) : Except String (List Val) :=
  let fallback := (eval A n rows).map fun c => intoValues c rows.length
  match n with
  | .mk (.prop attr) [c] => match propertyValues G attr c rows with
    | some vs => .ok vs
    | none => fallback
  | .mk (.var id) _ => .ok (rows.map fun r => (valueAt id r).getD null)
  | _ => fallback

/-- Each `eval_values` arm returns the per-row values (given the three leaf contracts). -/
theorem evalValues_sound (L : Lawful A S) (intoValues : C → Nat → List Val)
    (valueAt : Nat → Nat → Option Val) (null : Val)
    (hInto : ∀ c rows n, Sound S n rows c → ∀ i (hi : i < rows.length),
        S.rowEval n rows[i] = .ok ((intoValues c rows.length).getD i null))
    (hVar : ∀ id cs r, S.rowEval (.mk (.var id) cs) r = .ok ((valueAt id r).getD null))
    (hProp : ∀ attr c rows vs, propertyValues G attr c rows = some vs →
        ∀ i (hi : i < rows.length), S.rowEval (.mk (.prop attr) [c]) rows[i] = .ok (vs.getD i null))
    (n : Node Val) (rows : List Nat) (vs : List Val) (h : evalValues A G intoValues valueAt null n rows = .ok vs) :
    ∀ i (hi : i < rows.length), S.rowEval n rows[i] = .ok (vs.getD i null) := by
  have fb : ∀ (vs : List Val), ((eval A n rows).map fun c => intoValues c rows.length) = .ok vs →
      ∀ i (hi : i < rows.length), S.rowEval n rows[i] = .ok (vs.getD i null) := by
    intro vs h
    cases he : eval A n rows with
    | error e => simp [he, Except.map] at h
    | ok c =>
      simp only [he, Except.map, Except.ok.injEq] at h; subst h
      exact hInto c rows n (eval_sound A S L n rows c he)
  unfold evalValues at h
  split at h
  · split at h
    · rename_i hp; simp only [Except.ok.injEq] at h; subst h; exact hProp _ _ rows _ hp
    · exact fb vs h
  · simp only [Except.ok.injEq] at h; subst h
    intro i hi; rw [hVar]; simp [hi]
  · exact fb vs h

/-- `eval_has_labels` (:642). `isStrConst` recognises `ExprIR::Constant(Value::String)`;
the list's children are all checked (`mapM`) before any lookup. -/
structure LabelShape where
  isStrConst : Node Val → Option String

def hasLabels (LS : @LabelShape Val) (n : Node Val) (rows : List Nat) : Option (List Bool) :=
  match n with
  | .mk _ [.mk (.var id) _, .mk .list (e :: es)] =>
    match G.column id with
    | .nodeIds ids =>
      match (e :: es).mapM LS.isStrConst with
      | none => none
      | some names =>
        match names.mapM G.labelId with
        | none => some (rows.map fun _ => false)
        | some ls => some (rows.map fun r => ls.all fun l => G.hasLabelId (ids.getD r 0) l)
    | _ => none
  | _ => none

theorem mapM_none {α β} (f : α → Option β) : ∀ (l : List α), (∃ x ∈ l, f x = none) → l.mapM f = none
  | [], ⟨_, hx, _⟩ => by simp at hx
  | a :: l, ⟨x, hx, hn⟩ => by
    simp only [List.mapM_cons, Option.bind_eq_bind]
    cases hfa : f a with
    | none => rfl
    | some b =>
      have hxl : x ∈ l := by
        rcases List.mem_cons.1 hx with rfl | h
        · simp [hfa] at hn
        · exact h
      simp [mapM_none f l ⟨x, hxl, hn⟩]

theorem mapM_none_exists {α β} (f : α → Option β) : ∀ (l : List α), l.mapM f = none → ∃ x ∈ l, f x = none
  | [], h => by simp [pure] at h
  | a :: l, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind] at h
    cases hfa : f a with
    | none => exact ⟨a, by simp, hfa⟩
    | some b =>
      simp only [hfa, Option.bind_some] at h
      cases hl : l.mapM f with
      | none => obtain ⟨x, hx, hn⟩ := mapM_none_exists f l hl; exact ⟨x, by simp [hx], hn⟩
      | some r => simp [hl, pure] at h

theorem all_labels (L : G.Lawful) (x : Nat) : ∀ (names : List String) (ls : List Nat),
    names.mapM G.labelId = some ls →
    (ls.all fun l => G.hasLabelId x l) = (names.all fun nm => G.hasLabelName x nm)
  | [], ls, h => by simp [pure] at h; subst h; rfl
  | a :: as, ls, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨l, hl, ls', hls', rfl⟩ := h
    simp [L.label, hl, all_labels L x as ls' hls']

/-- `hasLabels` answers `∀ name ∈ list, node_has_label(id, name)` on every row. -/
theorem hasLabels_sound (L : G.Lawful) (LS : @LabelShape Val) (n : Node Val) (rows : List Nat)
    (bs : List Bool) (h : hasLabels G LS n rows = some bs) :
    ∃ (id : Nat) (ids : List Nat) (names : List String), G.column id = .nodeIds ids ∧ bs.length = rows.length ∧
      ∀ i (hi : i < rows.length), bs[i]? = some (names.all fun nm => G.hasLabelName (ids.getD rows[i] 0) nm) := by
  unfold hasLabels at h
  split at h
  · rename_i id _ e es
    split at h
    · rename_i ids hc
      split at h
      · simp at h
      · rename_i names hn
        refine ⟨id, ids, names, hc, ?_⟩
        split at h
        · rename_i hl
          simp only [Option.some.injEq] at h; subst h
          -- some name is unregistered: false everywhere, and so is the per-row answer
          obtain ⟨nm, hnm, hnone⟩ := mapM_none_exists G.labelId names hl
          refine ⟨by simp, fun i hi => ?_⟩
          simp only [List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some, Option.some.injEq]
          symm; rw [List.all_eq_false]; exact ⟨nm, hnm, by simp [L.label, hnone]⟩
        · rename_i ls hls
          simp only [Option.some.injEq] at h; subst h
          refine ⟨by simp, fun i hi => ?_⟩
          simp only [List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some, Option.some.injEq]
          exact all_labels G L _ names ls hls
    · simp at h
  · simp at h

/-- The whole list is type-checked before any label is resolved: a non-constant element
makes the shape unclaimed (`none`, per-row path raises the type error) even when an
earlier name is unregistered (`hasLabels(n, ['NeverUsed', 1])`, test11). -/
theorem hasLabels_typecheck_first (LS : @LabelShape Val) (k : Kind Val) (id : Nat) (cs : List (Node Val))
    (e : Node Val) (es : List (Node Val)) (rows : List Nat) (bad : Node Val) (hb : bad ∈ e :: es)
    (hbad : LS.isStrConst bad = none) :
    hasLabels G LS (.mk k [.mk (.var id) cs, .mk .list (e :: es)]) rows = none := by
  have : (e :: es).mapM LS.isStrConst = none := mapM_none _ _ ⟨bad, hb, hbad⟩
  unfold hasLabels
  cases hc : G.column id <;> simp only [hc] <;> rw [this]

/-- `eval_per_row` (:706): `ExprColumn::Values` of the per-row results; the first row
error is the batch error. -/
def perRow (ev : Nat → Except String Val) (rows : List Nat) : Except String (List Val) :=
  rows.mapM ev

theorem perRow_spec (ev : Nat → Except String Val) : ∀ (rows : List Nat) (vs : List Val),
    perRow ev rows = .ok vs → vs.length = rows.length ∧
      ∀ i (hi : i < rows.length), ∃ v, vs[i]? = some v ∧ ev rows[i] = .ok v
  | [], vs, h => by simp [perRow, List.mapM_nil, pure, Except.pure] at h; subst h; simp
  | r :: rs, vs, h => by
    simp only [perRow, List.mapM_cons] at h
    obtain ⟨v, hv, h⟩ := bind_ok h
    obtain ⟨ws, hws, h⟩ := bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    obtain ⟨hl, hi⟩ := perRow_spec ev rs ws hws
    refine ⟨by simp [hl], fun i hi' => ?_⟩
    cases i with
    | zero => exact ⟨v, rfl, hv⟩
    | succ i => simpa using hi i (by simpa using hi')

theorem perRow_error (ev : Nat → Except String Val) : ∀ (rows : List Nat) (e : String),
    perRow ev rows = .error e → ∃ r ∈ rows, ev r = .error e
  | [], e, h => by simp [perRow, pure, Except.pure] at h
  | r :: rs, e, h => by
    simp only [perRow, List.mapM_cons] at h
    cases hv : ev r with
    | error e' => simp [hv, bind, Except.bind] at h; subst h; exact ⟨r, by simp, hv⟩
    | ok v =>
      simp only [hv, bind, Except.bind] at h
      cases hw : List.mapM ev rs with
      | error e' =>
        simp [hw] at h; subst h
        obtain ⟨r', hr', he⟩ := perRow_error ev rs e' hw
        exact ⟨r', by simp [hr'], he⟩
      | ok ws => simp [hw, pure, Except.pure] at h

/-- `VectorEval::new` (:202): holds the runtime and nothing else. -/
structure VectorEval (R : Type) where
  runtime : R

def VectorEval.new {R : Type} (r : R) : VectorEval R := ⟨r⟩

theorem new_runtime {R : Type} (r : R) : (VectorEval.new r).runtime = r := rfl

end Columnar.Dispatch
