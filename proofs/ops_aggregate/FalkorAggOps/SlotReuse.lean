import FalkorAggOps.AggNext
/-
# Slot reuse across a WITH/RETURN aggregation (gap closed 2026-10-20)

`AggNext.finishGroup` models the per-group closure of `AggregateOp::next` over abstract slot
ids. What it did not model is **which** ids the binder hands it: a `Row` is keyed by the bare
`Variable.id` (row.rs:104-123, `insert_by_id`), and the binder numbers every scope from 0, so
the ids of the scope *before* a projection (pre-projection key aliases, aggregate
accumulators) and the ids of the scope *after* it (projected names, and every variable the
next clause mints) overlap. Nor did it model that `Row::unbind` clears only the bound bit and
keeps the value (row.rs:160-166), which a later operator still reads.

| here | there |
| --- | --- |
| `PItem`, `layoutGo`, `projLayout` | id minting of one projection: `bind_projection` binds every item with `bind_expr` **before** `push_scope` (binder.rs:1072-1078), so the parser's accumulator child `__agg_order_by_placeholder__` (cypher.rs:2046-2048, :2072-2074) gets `fresh_var` id = `len` of the pre-projection scope, kept unique under `__agg_placeholder_{id}` (binder.rs:2193-2200); the aliases then get `project_name` ids `0, 1, …` in the new scope (binder.rs:1149-1167, :2177-2185); the planner splits them into keys / aggregations in order (planner/mod.rs:2496-2512); a key whose tree is a bare variable records it as `original_var` (aggregate.rs:1160, :1199) |
| `nextId` | the next clause's first new variable: the projection scope holds exactly its aliases (binder `projectItems_keys`), and `fresh_var` mints `len` (binder.rs:2255-2266) |
| `finishVals` | the emitted row's stored values: `unbind` keeps the value (row.rs:160-166), so values = `finishCore` row + outputs |
| `cell` | what a downstream operator or the reply reads at a slot of the one-row batch: `push_row` marks a column present iff the slot is bound or holds a non-null value (batch.rs:631-640), `finish` drops a non-present column (batch.rs:763-778), `value_at` returns the stored value (reply.rs:655 `unwrap`s it) |
| `AV`, `finEv` | the values involved, and `ExprEval::eval` on an aggregate call: read the accumulator child's slot (`value_at(key.id)`, "Variable not found" if absent) and apply the finalizer (eval.rs:864-880); `finalize_avg` is `unreachable!` on a non-list accumulator (aggregation.rs:593-596) |

The general properties a fixed `next` should satisfy are `OutputsOwnAcc`, `KeysBound` and
`EmitsOnlyScope` (together `AggRowSpec`); each `bug…` theorem refutes one on the ids the
binder assigns in the issue's query.
-/
namespace FalkorAggOps.AggPlan

variable {Z : Type}

/-! ## The binder's id layout for one projection -/

/-- One projected item: a grouping key (with `some o` when its tree is a bare variable of id `o`
in the pre-projection scope), or an aggregate call `f(args…)` (the accumulator child is added by
the layout). Item expressions are assumed to mint no other ids (true for variables, literals and
property reads, which is every item in the repros). -/
inductive PItem (Z : Type) where
  | key (orig : Option Nat)
  | agg (f : Fn Z) (args : List (Ex Z))

/-- Item `i` gets id `i` in the new scope; the `j`-th aggregate's accumulator gets
`preLen + j` in the pre-projection scope. -/
def layoutGo : Nat → Nat → List (PItem Z) → List KeySpec × List (Nat × Ex Z)
  | _, _, [] => ([], [])
  | i, a, .key o :: is => let r := layoutGo (i + 1) a is; (⟨i, o⟩ :: r.1, r.2)
  | i, a, .agg f args :: is =>
    let r := layoutGo (i + 1) (a + 1) is; (r.1, (i, .func f (args ++ [.var a])) :: r.2)

/-- (`keys`, `aggregations`) of the `Aggregate` operator, with binder ids, for a projection whose
pre-projection scope holds `preLen` variables. -/
def projLayout (preLen : Nat) (items : List (PItem Z)) : List KeySpec × List (Nat × Ex Z) :=
  layoutGo 0 preLen items

/-- The id the clause after the projection gives its first new variable. -/
def nextId (items : List (PItem Z)) : Nat := items.length

/-! ## The emitted row: bound slots, stored values, what downstream reads -/

/-- Stored values of the emitted row (`unbind` keeps them). -/
def finishVals {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Except E (Env Z) :=
  match finishCore ev keys aggs key acc with
  | .error e => .error e
  | .ok (acc3, outs) => .ok (updAll acc3 outs)

/-- The value a reader sees at slot `j`: bound ⇒ its value; unbound with a non-null stored value
⇒ that value (a "value-present-but-unbound" column); otherwise the column is absent. -/
def cell (isNull : Z → Bool) (bnd vals : Env Z) (j : Nat) : Option Z :=
  match bnd j with
  | some v => some v
  | none => match vals j with
    | some v => if isNull v then none else some v
    | none => none

/-- Slot `j` of a finished group: `none` if the group errored (or panicked). -/
def slotAt {E : Type} (r : Except E (Env Z)) (j : Nat) : Option (Option Z) :=
  match r with
  | .ok e => some (e j)
  | .error _ => none

/-- What a reader sees at slot `j` of the emitted row (`none` if the group errored). -/
def emitCell {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) (j : Nat) : Option (Option Z) :=
  match finishGroup ev keys aggs key acc, finishVals ev keys aggs key acc with
  | .ok b, .ok v => some (cell isNull b v j)
  | _, _ => none

/-! ## The properties a correct `next` satisfies -/

/-- **Each aggregation reads its own accumulator.** Every output is bound to its expression
evaluated over the group's key row and accumulators alone (`combinedEnv`), i.e. no earlier
output written during the loop changes a later aggregation's input. (No projection item can
reference another item's alias, so nothing legitimate reads an output.) -/
def OutputsOwnAcc {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  ∀ p ∈ aggs, ∀ v, ev p.2 (combinedEnv keys key acc) = .ok v →
    slotAt (finishGroup ev keys aggs key acc) p.1 = some (some v)

/-- **Every grouping column is bound.** Each key name is bound on the emitted row to its key
value — including a `null` key, which is otherwise dropped from the batch. -/
def KeysBound {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  ∀ k ∈ keys, ∀ v, key k.name = some v →
    slotAt (finishGroup ev keys aggs key acc) k.name = some (some v)

/-- **The emitted row carries only the new scope.** A slot that is neither in the key row
(projected keys and `copy_from_parent` copies) nor an aggregate output reads as absent: no
pre-projection alias and no accumulator value survives into ids the next clause will reuse. -/
def EmitsOnlyScope {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z)
    (keys : List KeySpec) (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  ∀ j v, key j = none → j ∉ aggs.map Prod.fst → emitCell isNull ev keys aggs key acc j ≠ some (some v)

/-- The three together: what a fix of `AggregateOp::next` should prove for binder ids. -/
def AggRowSpec {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z)
    (keys : List KeySpec) (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  OutputsOwnAcc ev keys aggs key acc ∧ KeysBound ev keys aggs key acc ∧
    EmitsOnlyScope isNull ev keys aggs key acc

/-! ## Concrete values and finalizers -/

/-- The values in the repros. `str n` stands for a string literal; `avgAcc s c` for avg's
`[sum, count, overflow]` list accumulator. -/
inductive AV where
  | int (n : Int)
  | str (n : Nat)
  | null
  | node (id : Nat)
  | avgAcc (s : Int) (c : Nat)
  deriving DecidableEq

def AV.isNull : AV → Bool
  | .null => true
  | _ => false

inductive AErr where
  /-- `finalize_avg`'s `unreachable!("finalize_avg expects a list")`: a server panic -/
  | avgPanic
  | notFound
  | other
  deriving DecidableEq

def cntF : Fn AV := ⟨true, some (.int 0)⟩
def sumF : Fn AV := ⟨false, some (.int 0)⟩
def avgF : Fn AV := ⟨false, some (.avgAcc 0 0)⟩

/-- The finalizer, chosen by the function's initial accumulator (count/sum: identity; avg:
`finalize_avg`, which panics on a non-list). -/
def finalize : Option AV → AV → Except AErr AV
  | some (.avgAcc _ _), .avgAcc s c => .ok (if c = 0 then .null else .int (s / c))
  | some (.avgAcc _ _), _ => .error .avgPanic
  | _, v => .ok v

/-- `ExprEval::eval` on an aggregate call after consumption (eval.rs:864-880). -/
def finEv : Ex AV → Env AV → Except AErr AV
  | .func f cs, env => match cs.getLast? with
    | some (.var k) => match env k with
      | some v => finalize f.init v
      | none => .error .notFound
    | _ => .error .other
  | _, _ => .error .other

def env0 : Env AV := fun _ => none

/-! ## #3375: an output overwrites the next aggregation's accumulator -/

/-- `UNWIND [1,2,3] AS x RETURN 'a' AS k1, 'b' AS k2, count(x), sum(x)`: `x` = 0, so the
accumulators are 1 and 2, and the outputs are 2 and 3. -/
def items3375 (f2 : Fn AV) : List (PItem AV) :=
  [.key none, .key none, .agg cntF [.var 0], .agg f2 [.var 0]]

def key3375 : Env AV := upd (upd env0 0 (.str 0)) 1 (.str 1)
/-- After three rows: count 3, sum 6 (avg: sum 6 over 3). -/
def acc3375 (second : AV) : Env AV := upd (upd env0 1 (.int 3)) 2 second

theorem layout3375 (f2 : Fn AV) : projLayout 1 (items3375 f2) =
    ([⟨0, none⟩, ⟨1, none⟩],
     [(2, .func cntF [.var 0, .var 1]), (3, .func f2 [.var 0, .var 2])]) := rfl

/-- **#3375**: `count(x)`'s output (id 2) is written into `combined` before `sum(x)` reads its
accumulator, which also has id 2, so `sum(x)` returns 3 (the count). Live: Rust `3, 3`, C `3, 6`. -/
theorem bug3375_output_overwrites_next_acc :
    let L := projLayout 1 (items3375 sumF)
    slotAt (finishGroup finEv L.1 L.2 key3375 (acc3375 (.int 6))) 3 = some (some (.int 3)) ∧
    finEv (.func sumF [.var 0, .var 2]) (combinedEnv L.1 key3375 (acc3375 (.int 6))) = .ok (.int 6) ∧
    ¬ OutputsOwnAcc finEv L.1 L.2 key3375 (acc3375 (.int 6)) := by
  refine ⟨by decide, rfl, fun h => ?_⟩
  have := h (3, .func sumF [.var 0, .var 2]) (List.Mem.tail _ (List.Mem.head _)) (.int 6) rfl
  revert this; decide

/-- **#3375, `avg`**: the same overwrite hands `finalize_avg` an integer, and the server panics
(live: `internal error: entered unreachable code: finalize_avg expects a list`; C returns 2). -/
theorem bug3375_avg_panics :
    let L := projLayout 1 (items3375 avgF)
    finishGroup finEv L.1 L.2 key3375 (acc3375 (.avgAcc 6 3)) = .error .avgPanic ∧
    finEv (.func avgF [.var 0, .var 2]) (combinedEnv L.1 key3375 (acc3375 (.avgAcc 6 3))) = .ok (.int 2) := by
  exact ⟨rfl, rfl⟩

/-- Control (live: Rust and C both `3, 6`): with one key the second accumulator (id 2) is not an
output id, so `sum` reads its own value. -/
theorem agree3375_one_key :
    let L := projLayout 1 [PItem.key none, .agg cntF [.var 0], .agg sumF [.var 0]]
    slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.str 0)) (acc3375 (.int 6))) 2 =
      some (some (.int 6)) := by decide

/-! ## #3150 (fixed by #3522): a null key shares its slot with an accumulator -/

/-- `RETURN null AS x, count(1) AS n`: the pre-projection scope is empty, so `count`'s
accumulator is id 0 — the id of `x`. -/
def items3150 : List (PItem AV) := [.key none, .agg cntF [.const .int]]

theorem layout3150 : projLayout 0 items3150 =
    ([⟨0, none⟩], [(1, .func cntF [.const .int, .var 0])]) := rfl

/-- **History, before #3522 (ca36bfa15): the keys-bound property on the old order** (accumulators
unbound after the keys were written, `preFinishGroup`). -/
def PreKeysBound {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  ∀ k ∈ keys, ∀ v, key k.name = some v →
    slotAt (preFinishGroup ev keys aggs key acc) k.name = some (some v)

/-- **#3150 fixed by #3522 (ca36bfa15)**: `AggregateOp::next` unbinds the accumulators before it
writes the keys (aggregate.rs:1184-1192), so on the same ids the null key `x` is bound to `null`
and the column is present (live: `null, 1`, as C), and `KeysBound` holds. -/
theorem fixed3522_null_key_bound :
    let L := projLayout 0 items3150
    slotAt (finishGroup finEv L.1 L.2 (upd env0 0 .null) (upd env0 0 (.int 1))) 0 = some (some .null) ∧
    emitCell AV.isNull finEv L.1 L.2 (upd env0 0 .null) (upd env0 0 (.int 1)) 0 = some (some .null) ∧
    emitCell AV.isNull finEv L.1 L.2 (upd env0 0 (.str 0)) (upd env0 0 (.int 1)) 0 =
      some (some (.str 0)) ∧
    KeysBound finEv L.1 L.2 (upd env0 0 .null) (upd env0 0 (.int 1)) := by
  refine ⟨by decide, by decide, by decide, fun k hk v hv => ?_⟩
  have hk' : k = ⟨0, none⟩ := by simpa [projLayout, layoutGo, items3150] using hk
  subst hk'
  revert hv; revert v; decide

/-- **General `KeysBound` on main**: for any keys, accumulators and evaluator, as long as no key
name is an aggregate output name and the group evaluates, every key is bound to its value
(`AggNext.finishGroup_keys_bound`). -/
theorem keysBound_main {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z)
    (hdisj : ∀ k ∈ keys, k.name ∉ aggs.map Prod.fst)
    (hok : ∃ out, finishGroup ev keys aggs key acc = .ok out) :
    KeysBound ev keys aggs key acc := by
  obtain ⟨out, ho⟩ := hok
  intro k hk v hv
  simp [slotAt, ho, finishGroup_keys_bound ev keys aggs key acc out hdisj ho k hk v hv]

/-- The PR's second test, `WITH 1 AS a, 2 AS b RETURN b AS x, a AS y, count(*)`: pre-projection
`a, b` = 0, 1 so the accumulator is 2; the keys `x` (id 0, original 1) and `y` (id 1, original 0)
swap the pre-projection slots, and each alias write lands on the other's slot id. Both keys stay
bound to their own value (`x = 2`, `y = 1`) and the count is output 3. -/
theorem keysBound_swap_3522 :
    let L := projLayout 2 [PItem.key (some 1), .key (some 0), .agg cntF [.const .int]]
    let key := upd (upd env0 0 (.int 2)) 1 (.int 1)
    let acc := upd env0 2 (.int 7)
    L.1 = [⟨0, some 1⟩, ⟨1, some 0⟩] ∧
    slotAt (finishGroup finEv L.1 L.2 key acc) 0 = some (some (.int 2)) ∧
    slotAt (finishGroup finEv L.1 L.2 key acc) 1 = some (some (.int 1)) ∧
    slotAt (finishGroup finEv L.1 L.2 key acc) 2 = some (some (.int 7)) ∧
    KeysBound finEv L.1 L.2 key acc := by
  refine ⟨rfl, by decide, by decide, by decide, fun k hk v hv => ?_⟩
  simp only [projLayout, layoutGo, List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl <;> revert hv <;> revert v <;> decide

/-! ## #3333: a pre-projection key alias binds the next clause's first variable -/

/-- `MATCH (a)-[:R]->(x)-[:R]->(m) WITH m, count(*) AS c1`: nodes first, so `a, x, m` = 0, 1, 2
and the two anonymous relationships 3, 4 (binder `rustBind`); the accumulator is 5. The key `m`
is id 0 with `original_var` 2. -/
def items3333 : List (PItem AV) := [.key (some 2), .agg cntF [.const .int]]

theorem layout3333 : projLayout 5 items3333 =
    ([⟨0, some 2⟩], [(1, .func cntF [.const .int, .var 5])]) := rfl

/-- **#3333**: the alias write `acc.insert(original_var, …)` binds slot 2 to `m`'s node, and the
following `MATCH (m)-[r:R]->(b …)` gives `b` id `nextId = 2`, so `b` arrives already bound to
`m`: the traversal looks for an edge from `m` to `m` and finds none. Live: Rust `[]`, C
`[3, 1, 4]`. -/
theorem bug3333_alias_binds_next_var :
    let L := projLayout 5 items3333
    nextId items3333 = 2 ∧
    slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 5 (.int 1))) 2 =
      some (some (.node 2)) ∧
    ¬ EmitsOnlyScope AV.isNull finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 5 (.int 1)) := by
  refine ⟨rfl, by decide, fun h => ?_⟩
  exact h 2 (.node 2) (by decide) (by decide) (by decide)

/-- Controls, both returning `[3, 1, 4]` live on Rust: with one hop before the WITH (`x, m` = 0, 1)
the alias slot 1 is owned by the output `c1` and is skipped; with an extra item `0 AS z` the alias
slot 2 is overwritten by `z`'s key value and `b` gets id 3, which is unbound. -/
theorem agree3333_controls :
    (let L := projLayout 3 [PItem.key (some 1), .agg cntF [.const .int]]
     slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 3 (.int 1))) 2 = some none) ∧
    (let L := projLayout 5 [PItem.key (some 2), .agg cntF [.const .int], .key none]
     slotAt (finishGroup finEv L.1 L.2 (upd (upd env0 0 (.node 2)) 2 (.int 0)) (upd env0 5 (.int 1))) 3 =
       some none) := by
  decide

/-! ## #3369 and #3368: a `count(*)` accumulator is left in the slot of a later MERGE variable -/

/-- `… MERGE (a:A …) WITH count(*) AS c0` (or `UNWIND … AS r0 UNWIND … AS a WITH count(*) AS c0`):
`r0, a` = 0, 1, so the accumulator is 2. In the next segment `UNWIND … AS r1 MERGE (b:B …)` gives
`r1, b` = 1, 2. -/
def itemsCount : List (PItem AV) := [.agg cntF [.const .int]]

theorem layoutCount2 : projLayout 2 itemsCount = ([], [(0, .func cntF [.const .int, .var 2])]) := rfl

/-- The shared cause: a keyless `count(*)` over a two-variable scope leaves the integer count at
slot 2, unbound but present, which `EmitsOnlyScope` forbids. -/
theorem countAcc_left_at_2 :
    let L := projLayout 2 itemsCount
    slotAt (finishGroup finEv L.1 L.2 env0 (upd env0 2 (.int 1))) 2 = some none ∧
    emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some (some (.int 1)) ∧
    ¬ EmitsOnlyScope AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  exact h 2 (.int 1) (by decide) (by decide) (by decide)

/-- **#3369**: both counts leave their accumulator where the next segment's MERGE node lands:
segment 1's at 2 (`b`), and segment 2's `WITH c0, count(*) AS c1` (pre-scope `c0, r1, b`, so
accumulator 3) at 3, the id of `c` in `UNWIND … AS r2 MERGE (c:C …)` (`c0, c1, r2, c`).
Live (fresh graph): both leftovers are needed — `collect(1)` in either segment, or a third
variable before the first count, makes the query correct — and then the second MERGE returns
`c` = the integer 1, so `c.id` fails with "expected Map, Node, … but was Integer" (C: `'c1'`). -/
theorem bug3369_count_acc_in_merge_slot :
    (let L := projLayout 2 itemsCount
     emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some (some (.int 1))) ∧
    (let L := projLayout 3 [PItem.key (some 0), .agg cntF [.const .int]]
     L = ([⟨0, some 0⟩], [(1, .func cntF [.const .int, .var 3])]) ∧
     emitCell AV.isNull finEv L.1 L.2 (upd env0 0 (.int 1)) (upd env0 3 (.int 1)) 3 = some (some (.int 1)) ∧
     ¬ EmitsOnlyScope AV.isNull finEv L.1 L.2 (upd env0 0 (.int 1)) (upd env0 3 (.int 1))) := by
  refine ⟨by decide, rfl, by decide, fun h => ?_⟩
  exact h 3 (.int 1) (by decide) (by decide) (by decide)

/-- **#3368**: same first segment (count left at 2, MERGE `b` = 2); then `WITH b, r1 WITH count(*)
AS c1` counts over a two-variable scope again, leaving the count at 2, and `UNWIND … AS r2 MERGE
(c:C …)` gives `c` id 2. With `WITH b` alone the accumulator is 1, which `r2` overwrites, and the
query is correct (as reported). Live: Rust returns `[B] b1` for `c` and creates 2 nodes; C
`[C] c1`, 3 nodes. -/
theorem bug3368_count_acc_in_merge_slot :
    (let L := projLayout 2 itemsCount
     emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some (some (.int 1)) ∧
     ¬ EmitsOnlyScope AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1))) ∧
    (let L := projLayout 1 itemsCount
     L = ([], [(0, .func cntF [.const .int, .var 1])])) := by
  exact ⟨⟨countAcc_left_at_2.2.1, countAcc_left_at_2.2.2⟩, rfl⟩

end FalkorAggOps.AggPlan
