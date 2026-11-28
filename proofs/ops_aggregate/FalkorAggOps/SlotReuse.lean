import FalkorAggOps.AggNext
/-
# Slot reuse across a WITH/RETURN aggregation (gap closed 2026-10-20; bugs fixed by #3579)

`AggNext.finishGroup` models the per-group closure of `AggregateOp::next` over abstract slot
ids. This file mints the ids the binder hands it: a `Row` is keyed by the bare `Variable.id`
(row.rs, `insert_by_id`), and the binder numbers every scope from 0, so the ids of the scope
*before* a projection (pre-projection key aliases, aggregate accumulators) and the ids of the
scope *after* it (projected names, and every variable the next clause mints) overlap.

The properties `OutputsOwnAcc`, `KeysBound` and `EmitsOnlyScope` (together `AggRowSpec`) were
refuted on main by #3375, #3333, #3369 and #3368. Since #3579 `next` evaluates every item over
an old-scope-only `eval_env` and emits only the key row plus the outputs, so `aggRowSpec_main`
proves all three for any ids (given distinct output names that are not key names, and a group
that evaluates), and each issue's query is re-checked on the ids the binder assigns.
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

/-- Stored values of the emitted row. Since #3579 nothing is unbound, so the stored values are
the bound slots (`Row::insert` sets the bit). -/
def finishVals {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Except E (Env Z) :=
  finishGroup ev keys aggs key acc

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
evaluated over the group's accumulators and pre-projection key aliases alone (`evalEnv`), i.e.
no output written during the loop changes a later aggregation's input. (No projection item can
reference another item's alias, so nothing legitimate reads an output.) -/
def OutputsOwnAcc {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Prop :=
  ∀ p ∈ aggs, ∀ v, ev p.2 (evalEnv keys key acc) = .ok v →
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

/-- The three together: what `AggregateOp::next` proves for binder ids since #3579. -/
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

/-- `ExprEval::eval` on an aggregate call after consumption (eval.rs). -/
def finEv : Ex AV → Env AV → Except AErr AV
  | .func f cs, env => match cs.getLast? with
    | some (.var k) => match env k with
      | some v => finalize f.init v
      | none => .error .notFound
    | _ => .error .other
  | _, _ => .error .other

def env0 : Env AV := fun _ => none


/-! ## The properties hold on any ids since #3579 -/

/-- **`OutputsOwnAcc` on main (#3375 fixed by #3579)**: with distinct output names, every item
that evaluates is bound to its value over `eval_env`, once the group evaluates. -/
theorem outputsOwnAcc_main {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z)
    (hnd : (aggs.map Prod.fst).Nodup)
    (hok : ∃ out, finishGroup ev keys aggs key acc = .ok out) :
    OutputsOwnAcc ev keys aggs key acc := by
  obtain ⟨out, ho⟩ := hok
  intro p hp v hv
  simp [slotAt, ho, finishGroup_outputs_eval ev keys aggs key acc out hnd ho p hp v hv]

/-- **`EmitsOnlyScope` on main (#3333, #3368, #3369 fixed by #3579)**: for any ids, a slot that is
neither a key nor an output reads as absent on the emitted row. -/
theorem emitsOnlyScope_main {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z)
    (keys : List KeySpec) (aggs : List (Nat × Ex Z)) (key acc : Env Z) :
    EmitsOnlyScope isNull ev keys aggs key acc := by
  intro j v hk hj
  unfold emitCell finishVals
  cases ho : finishGroup ev keys aggs key acc with
  | error e => simp
  | ok out =>
    have hn := finishGroup_only_scope ev keys aggs key acc out ho j hk hj
    simp [cell, hn]

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

/-- **`AggRowSpec` on main (#3579)**: the three properties together. -/
theorem aggRowSpec_main {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z)
    (keys : List KeySpec) (aggs : List (Nat × Ex Z)) (key acc : Env Z)
    (hnd : (aggs.map Prod.fst).Nodup)
    (hdisj : ∀ k ∈ keys, k.name ∉ aggs.map Prod.fst)
    (hok : ∃ out, finishGroup ev keys aggs key acc = .ok out) :
    AggRowSpec isNull ev keys aggs key acc :=
  ⟨outputsOwnAcc_main ev keys aggs key acc hnd hok, keysBound_main ev keys aggs key acc hdisj hok,
    emitsOnlyScope_main isNull ev keys aggs key acc⟩

/-- The layout gives the items distinct new-scope ids, so its output names are distinct and no
key name is an output name: the premises of `aggRowSpec_main` hold for every `projLayout`. -/
theorem layoutGo_names (i a : Nat) (items : List (PItem Z)) :
    let r := layoutGo i a items
    (∀ k ∈ r.1, i ≤ k.name) ∧ (∀ p ∈ r.2, i ≤ p.1) ∧ (r.2.map Prod.fst).Nodup ∧
      ∀ k ∈ r.1, k.name ∉ r.2.map Prod.fst := by
  induction items generalizing i a with
  | nil => simp [layoutGo]
  | cons it is ih =>
    cases it with
    | key o =>
      obtain ⟨h1, h2, h3, h4⟩ := ih (i + 1) a
      simp only [layoutGo, List.mem_cons, forall_eq_or_imp]
      refine ⟨⟨Nat.le_refl _, fun k hk => by have := h1 k hk; omega⟩,
        fun p hp => by have := h2 p hp; omega, h3, ⟨fun hm => ?_, h4⟩⟩
      obtain ⟨p, hp, he⟩ := List.mem_map.mp hm
      have := h2 p hp; have he' : p.1 = i := he; omega
    | agg f args =>
      obtain ⟨h1, h2, h3, h4⟩ := ih (i + 1) (a + 1)
      simp only [layoutGo, List.mem_cons, forall_eq_or_imp, List.map_cons, List.nodup_cons]
      refine ⟨fun k hk => by have := h1 k hk; omega, ⟨Nat.le_refl _, fun p hp => by
        have := h2 p hp; omega⟩, ⟨fun hm => ?_, h3⟩, fun k hk hm => ?_⟩
      · obtain ⟨p, hp, he⟩ := List.mem_map.mp hm
        have := h2 p hp; omega
      · rcases hm with he | hm
        · have := h1 k hk; omega
        · exact h4 k hk hm

/-- **`AggRowSpec` for every projection the binder lays out**, whenever the group evaluates. -/
theorem aggRowSpec_layout {E : Type} (isNull : Z → Bool) (ev : Ex Z → Env Z → Except E Z)
    (preLen : Nat) (items : List (PItem Z)) (key acc : Env Z)
    (hok : ∃ out, finishGroup ev (projLayout preLen items).1 (projLayout preLen items).2 key acc = .ok out) :
    AggRowSpec isNull ev (projLayout preLen items).1 (projLayout preLen items).2 key acc := by
  obtain ⟨-, -, hnd, hdisj⟩ := layoutGo_names (Z := Z) 0 preLen items
  exact aggRowSpec_main isNull ev _ _ key acc hnd hdisj hok

/-! ## #3375: an output no longer overwrites the next aggregation's accumulator -/

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

/-- **#3375 fixed by #3579**: `count(x)`'s output (id 2) is no longer written where `sum(x)` reads
its accumulator (also id 2), so `sum(x)` returns 6 (live before: `3, 3`; C `3, 6`), the count is
3, and `OutputsOwnAcc` holds. -/
theorem fixed3579_3375_sum_own_acc :
    let L := projLayout 1 (items3375 sumF)
    slotAt (finishGroup finEv L.1 L.2 key3375 (acc3375 (.int 6))) 3 = some (some (.int 6)) ∧
    slotAt (finishGroup finEv L.1 L.2 key3375 (acc3375 (.int 6))) 2 = some (some (.int 3)) ∧
    OutputsOwnAcc finEv L.1 L.2 key3375 (acc3375 (.int 6)) :=
  ⟨by decide, by decide, (aggRowSpec_layout AV.isNull finEv 1 (items3375 sumF) key3375
    (acc3375 (.int 6)) ⟨_, rfl⟩).1⟩

/-- **#3375, `avg`, fixed by #3579**: `finalize_avg` gets its own list accumulator and returns 2
(before: the server panicked, `finalize_avg expects a list`; C 2). -/
theorem fixed3579_3375_avg :
    let L := projLayout 1 (items3375 avgF)
    slotAt (finishGroup finEv L.1 L.2 key3375 (acc3375 (.avgAcc 6 3))) 3 = some (some (.int 2)) := by
  decide

/-- Control (Rust and C both `3, 6`): with one key the second accumulator (id 2) is not an
output id, so `sum` reads its own value. -/
theorem agree3375_one_key :
    let L := projLayout 1 [PItem.key none, .agg cntF [.var 0], .agg sumF [.var 0]]
    slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.str 0)) (acc3375 (.int 6))) 2 =
      some (some (.int 6)) := by decide

/-! ## A null key shares its slot with an accumulator -/

/-- `RETURN null AS x, count(1) AS n`: the pre-projection scope is empty, so `count`'s
accumulator is id 0 — the id of `x`. -/
def items3150 : List (PItem AV) := [.key none, .agg cntF [.const .int]]

theorem layout3150 : projLayout 0 items3150 =
    ([⟨0, none⟩], [(1, .func cntF [.const .int, .var 0])]) := rfl

/-- **#3150 (fixed by #3522, still so after #3579)**: the emitted row starts as the key row and
no accumulator is written to it, so on the same ids the null key `x` is bound to `null` and the
column is present (live: `null, 1`, as C), and `KeysBound` holds. -/
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

/-- `WITH 1 AS a, 2 AS b RETURN b AS x, a AS y, count(*)` (#3522's second test): pre-projection
`a, b` = 0, 1 so the accumulator is 2; the keys `x` (id 0, original 1) and `y` (id 1, original 0)
swap the pre-projection slots. Both keys stay bound to their own value (`x = 2`, `y = 1`) and the
count is output at id 2. -/
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

/-! ## #3333: a pre-projection key alias no longer binds the next clause's first variable -/

/-- `MATCH (a)-[:R]->(x)-[:R]->(m) WITH m, count(*) AS c1`: nodes first, so `a, x, m` = 0, 1, 2
and the two anonymous relationships 3, 4 (binder `rustBind`); the accumulator is 5. The key `m`
is id 0 with `original_var` 2. -/
def items3333 : List (PItem AV) := [.key (some 2), .agg cntF [.const .int]]

theorem layout3333 : projLayout 5 items3333 =
    ([⟨0, some 2⟩], [(1, .func cntF [.const .int, .var 5])]) := rfl

/-- **#3333 fixed by #3579**: the following `MATCH (m)-[r:R]->(b …)` gives `b` id `nextId = 2`,
and slot 2 is no longer written on the emitted row (before: bound to `m`'s node, so the query
returned `[]`; C `[3, 1, 4]`): `EmitsOnlyScope` holds. -/
theorem fixed3579_3333_next_var_free :
    let L := projLayout 5 items3333
    nextId items3333 = 2 ∧
    slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 5 (.int 1))) 2 = some none ∧
    EmitsOnlyScope AV.isNull finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 5 (.int 1)) :=
  ⟨rfl, by decide, emitsOnlyScope_main _ _ _ _ _ _⟩

/-- Controls (`[3, 1, 4]` live on Rust before and after): with one hop before the WITH (`x, m` =
0, 1) slot 2 is free; with an extra item `0 AS z`, `b` gets id 3, which is free. -/
theorem agree3333_controls :
    (let L := projLayout 3 [PItem.key (some 1), .agg cntF [.const .int]]
     slotAt (finishGroup finEv L.1 L.2 (upd env0 0 (.node 2)) (upd env0 3 (.int 1))) 2 = some none) ∧
    (let L := projLayout 5 [PItem.key (some 2), .agg cntF [.const .int], .key none]
     slotAt (finishGroup finEv L.1 L.2 (upd (upd env0 0 (.node 2)) 2 (.int 0)) (upd env0 5 (.int 1))) 3 =
       some none) := by
  decide

/-! ## #3369 and #3368: no `count(*)` accumulator is left in the slot of a later MERGE variable -/

/-- `… MERGE (a:A …) WITH count(*) AS c0` (or `UNWIND … AS r0 UNWIND … AS a WITH count(*) AS c0`):
`r0, a` = 0, 1, so the accumulator is 2. In the next segment `UNWIND … AS r1 MERGE (b:B …)` gives
`r1, b` = 1, 2. -/
def itemsCount : List (PItem AV) := [.agg cntF [.const .int]]

theorem layoutCount2 : projLayout 2 itemsCount = ([], [(0, .func cntF [.const .int, .var 2])]) := rfl

/-- The shared cause, fixed by #3579: a keyless `count(*)` over a two-variable scope no longer
leaves the count at slot 2 (before: unbound but present, `1`); the reader sees slot 2 absent. -/
theorem fixed3579_countAcc_not_left :
    let L := projLayout 2 itemsCount
    slotAt (finishGroup finEv L.1 L.2 env0 (upd env0 2 (.int 1))) 2 = some none ∧
    emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some none ∧
    EmitsOnlyScope AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) :=
  ⟨by decide, by decide, emitsOnlyScope_main _ _ _ _ _ _⟩

/-- **#3369 fixed by #3579**: neither count leaves its accumulator where the next segment's MERGE
node lands: segment 1's slot 2 (`b`) and segment 2's (`WITH c0, count(*) AS c1` over `c0, r1, b`,
accumulator 3) slot 3, the id of `c` in `UNWIND … AS r2 MERGE (c:C …)`, are absent (before: `c`
was the integer 1 and `c.id` failed; C `'c1'`). -/
theorem fixed3579_3369_merge_slot_free :
    (let L := projLayout 2 itemsCount
     emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some none) ∧
    (let L := projLayout 3 [PItem.key (some 0), .agg cntF [.const .int]]
     L = ([⟨0, some 0⟩], [(1, .func cntF [.const .int, .var 3])]) ∧
     emitCell AV.isNull finEv L.1 L.2 (upd env0 0 (.int 1)) (upd env0 3 (.int 1)) 3 = some none ∧
     EmitsOnlyScope AV.isNull finEv L.1 L.2 (upd env0 0 (.int 1)) (upd env0 3 (.int 1))) :=
  ⟨by decide, rfl, by decide, emitsOnlyScope_main _ _ _ _ _ _⟩

/-- **#3368 fixed by #3579**: same first segment; then `WITH b, r1 WITH count(*) AS c1` counts
over a two-variable scope again, and `UNWIND … AS r2 MERGE (c:C …)` gives `c` id 2, which is now
free (before: Rust returned `[B] b1` for `c`; C `[C] c1`). -/
theorem fixed3579_3368_merge_slot_free :
    (let L := projLayout 2 itemsCount
     emitCell AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1)) 2 = some none ∧
     EmitsOnlyScope AV.isNull finEv L.1 L.2 env0 (upd env0 2 (.int 1))) ∧
    (let L := projLayout 1 itemsCount
     L = ([], [(0, .func cntF [.const .int, .var 1])])) :=
  ⟨⟨fixed3579_countAcc_not_left.2.1, fixed3579_countAcc_not_left.2.2⟩, rfl⟩

end FalkorAggOps.AggPlan
