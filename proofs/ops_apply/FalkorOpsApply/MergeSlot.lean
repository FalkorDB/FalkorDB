import FalkorOpsApply.Ctors
/-
# MERGE reads a value-only slot as its node (gap closed 2026-10-20; #3368, #3369)

`Merge.lean` treats the merged variable as a clause-level constant (`Clause.var`) and the
pattern hash as the property key alone, so the row the MERGE receives never influenced the
outcome. Two details of `MergeOp` at fb3d516d0 were outside the model:

* the row is a dense value vector plus a bound bitset (row.rs:65-72); a slot whose column is
  "value-present-but-unbound" (batch.rs:631-640 `push_row`, :763-778 `finish`) is rebuilt by
  `to_owned_row` (batch.rs:1431-1442) as stored-but-unbound (`insert_by_id` then
  `unbind_by_id`, row.rs:113-123, :222-227);
* `compute_merge_pattern_hash` asks `vars.get_by_id(node.alias.id)` (merge.rs:189), which reads
  the stored value whatever the bound bit says (row.rs:128-133). A present value — any value,
  not only a node — replaces the label and property tokens (`Ctors.nodeToks_bound`). On a
  no-match row `do_create_fallback` (merge.rs:113-178) looks that hash up in the query-wide
  `merge_pattern_cache` and, on a hit, re-inserts the cached `(var id, value)` pairs
  (merge.rs:122-127) — the ids of whichever MERGE created the entry.

The match sub-plan itself is not affected: the binder knows the variable is new, so it plans
a label scan that writes the slot and never reads it (live: `OPTIONAL MATCH (c:C {id: …})`
returns null in the same position, #3368). `MergeOp::next`'s all-bound shortcut
(merge.rs:369-388) uses `is_bound_by_id` and is only reached when matches exist.

Where does a value-only slot at a MERGE variable's id come from?  ops_aggregate
`FalkorAggOps.AggPlan.bug3368_count_acc_in_merge_slot` / `bug3369_count_acc_in_merge_slot`
(property `EmitsOnlyScope`): a keyless `count(*)` over a 2-variable scope leaves its
accumulator, unbound but present, at slot 2; `WITH c0, count(*) AS c1` over `c0, r1, b` leaves
it at slot 3. That row shape is restated here (`row2`, `row3_68`, `row3_69`), not imported.

| here | there |
| --- | --- |
| `MV`, `Row`, `Row.get`, `Row.isBound`, `Row.insert`, `Row.unbind` | `Value`, `Row` (row.rs:65), `get_by_id` (:128), `is_bound_by_id` (:199), `insert_by_id` (:113), `unbind_by_id` (:222) |
| `Cell`, `Row.ofCells` | a columnar batch row (`Column::Unbound` / bound / `value_only`) and `BatchRow::to_owned_row` (batch.rs:1431) |
| `MClause`, `nodeOf`, `attrsOf` | a one-node pattern `MERGE (v:L {id: r.id})` |
| `rustKey` | `compute_merge_pattern_hash` (merge.rs:180-246) = `Ctors.mergeHashInput` with `get := Row.get` (FxHash taken as injective on token streams; collisions are FINDINGS 6) |
| `fixKey` | the same with `get` returning a value only for a slot bound to a node |
| `matchK` | the match sub-plan (`run_batch(merge_child_idx)`, merge.rs:313-343): label scan + property filter |
| `mergeRowK` | one input row of `MergeOp::next` (merge.rs:347-434) with `do_create_fallback` (merge.rs:113-178) |

Properties a fix should prove: `KeyFromBoundNodeOnly` (MERGE takes a pattern variable as given
only when the row binds it to a node) and `MergeBindsOwnNode` (after MERGE the variable is bound
to a node with the pattern's label and properties). `fixKey_from_bound_node` and
`fixed_binds_own_node` prove the target is reachable.
-/
namespace MergeSlot

open Ctors

/-! ## Values, rows, cells -/

/-- The values in the repros. `map k` is the row map `{id: 'k'}`, `str k` the string `'k'`. -/
inductive MV where
  | null
  | int (n : Int)
  | list (l : List Int)
  | str (k : Nat)
  | map (k : Nat)
  | node (id : Nat)
  deriving DecidableEq, Repr

/-- `Row`: dense values (`Null`-padded) + bound bits (row.rs:65-72). -/
structure Row where
  vals : List MV
  bnd : List Bool
  deriving DecidableEq, Repr

def Row.empty : Row := ⟨[], []⟩

/-- `get_by_id` (row.rs:128-133): the stored value, bound or not; `none` only out of range. -/
def Row.get (r : Row) (i : Nat) : Option MV := r.vals[i]?

/-- `is_bound_by_id`. -/
def Row.isBound (r : Row) (i : Nat) : Bool := (r.bnd[i]?).getD false

def pad {β : Type} (l : List β) (n : Nat) (d : β) : List β := l ++ List.replicate (n - l.length) d

/-- `insert_by_id` (row.rs:113-123): grow with `Null`, store, set the bound bit. -/
def Row.insert (r : Row) (i : Nat) (v : MV) : Row :=
  ⟨(pad r.vals (i + 1) .null).set i v, (pad r.bnd (i + 1) false).set i true⟩

/-- `unbind_by_id` (row.rs:222-227): clear the bit, keep the value. -/
def Row.unbind (r : Row) (i : Nat) : Row := ⟨r.vals, r.bnd.set i false⟩

/-- One column of a batch row: `Column::Unbound`, a bound value, or a `value_only` value. -/
inductive Cell where
  | unbound
  | bound (v : MV)
  | valueOnly (v : MV)

/-- `BatchRow::to_owned_row` (batch.rs:1431-1442). -/
def Row.ofCellsGo : Nat → List Cell → Row → Row
  | _, [], r => r
  | i, .unbound :: cs, r => Row.ofCellsGo (i + 1) cs r
  | i, .bound v :: cs, r => Row.ofCellsGo (i + 1) cs (r.insert i v)
  | i, .valueOnly v :: cs, r => Row.ofCellsGo (i + 1) cs ((r.insert i v).unbind i)

def Row.ofCells (cs : List Cell) : Row := Row.ofCellsGo 0 cs Row.empty

/-- What `push_row` makes of slot `i` for the next operator / the reply (batch.rs:631-640):
bound ⇒ the value; unbound but non-null ⇒ the value (`value_only`); otherwise absent. -/
def Row.read (r : Row) (i : Nat) : Option MV :=
  match r.get i with
  | some v => if r.isBound i || v != .null then some v else none
  | none => none

theorem pad_length {β : Type} (l : List β) (n : Nat) (d : β) : n ≤ (pad l n d).length := by
  simp [pad]; omega

theorem Row.get_insert_self (r : Row) (i : Nat) (v : MV) : (r.insert i v).get i = some v := by
  have := pad_length r.vals (i + 1) .null
  simp only [Row.insert, Row.get]
  rw [List.getElem?_set_self (by omega)]

theorem Row.isBound_insert_self (r : Row) (i : Nat) (v : MV) : (r.insert i v).isBound i = true := by
  have := pad_length r.bnd (i + 1) false
  simp only [Row.insert, Row.isBound]
  rw [List.getElem?_set_self (by omega)]; rfl

/-! ## Graph, clause, hash key -/

structure GNode where
  id : Nat
  label : Nat
  attr : MV
  deriving DecidableEq, Repr

abbrev Graph := List GNode

/-- `MERGE (var:label {id: <slot src>.id})`. -/
structure MClause where
  var : Nat
  label : Nat
  src : Nat
  /-- plan index of this MERGE operator (`MergeOp::idx`): the cache key's first half since #3050 -/
  op : Nat

/-- The cache as it was before #3050: one key space for every MERGE of the query. -/
def MClause.erase (cl : MClause) : MClause := { cl with op := 0 }

/-- `{id: r.id}` evaluated on the row (`r` = the map at slot `src`). -/
def MClause.attr (cl : MClause) (r : Row) : MV :=
  match r.get cl.src with
  | some (.map k) => .str k
  | _ => .null

/-- The property map as `nodeToks` sees it (`none` = Null). -/
def attrsOf (cl : MClause) (r : Row) : Unit → Except String (List (String × Option MV)) :=
  fun _ => match cl.attr r with
    | .null => .ok [("id", none)]
    | v => .ok [("id", some v)]

def nodeOf (cl : MClause) : PNode Nat Unit := ⟨cl.var, [cl.label], ()⟩

abbrev Key := Except String (List (Tok MV Nat Unit))

/-- `compute_merge_pattern_hash` on main: `get_by_id`, bound or not (merge.rs:189). -/
def rustKey (cl : MClause) (r : Row) : Key :=
  mergeHashInput (T := Unit) r.get (attrsOf cl r) [nodeOf cl] []

/-- The fixed lookup: a slot counts only when the row binds it to a node. -/
def boundNodeGet (r : Row) (i : Nat) : Option MV :=
  if r.isBound i then
    match r.get i with
    | some (.node id) => some (.node id)
    | _ => none
  else none

def fixKey (cl : MClause) (r : Row) : Key :=
  mergeHashInput (T := Unit) (boundNodeGet r) (attrsOf cl r) [nodeOf cl] []

/-- The pattern-variable value a key was built from, if any (`Tok.val` = "taken as given"). -/
def keyVal : Key → Option MV
  | .ok [.val v] => some v
  | _ => none

/-! ## One MERGE row -/

/-- The key with property-map *names* dropped (every map here has the single key `id`), so
that key equality reduces in the kernel (String equality does not). -/
inductive KTok where
  | val (v : MV)
  | label (l : Nat)
  | attrs (vs : List (Option MV))
  | types
  deriving DecidableEq

def KTok.ofTok : Tok MV Nat Unit → KTok
  | .val v => .val v
  | .label l => .label l
  | .attrs m => .attrs (m.map Prod.snd)
  | .types _ => .types

def norm : Key → Option (List KTok)
  | .ok ts => some (ts.map KTok.ofTok)
  | .error _ => none

/-- `merge_pattern_cache`: hash ↦ cached `(var id, value)` pairs (hash = the token stream). -/
abbrev Cache := List ((Nat × Key) × List (Nat × MV))

def lookup (c : Cache) (k : Nat × Key) : Option (List (Nat × MV)) :=
  (c.find? (fun e => decide (e.1.1 = k.1 ∧ norm e.1.2 = norm k.2))).map Prod.snd

structure St where
  g : Graph
  next : Nat
  cache : Cache

/-- The match sub-plan: label scan + `{id: …}` filter; the new variable's slot is not read. -/
def matchK (g : Graph) (cl : MClause) (r : Row) : List GNode :=
  g.filter (fun n => n.label == cl.label && n.attr == cl.attr r)

/-- One input row of `MergeOp::next` without SET (merge.rs:347-434): matches ⇒ one output per
match with the variable bound; none ⇒ `do_create_fallback` (merge.rs:113-178). -/
def mergeRowK (key : MClause → Row → Key) (cl : MClause) (st : St) (r : Row) :
    Except String (St × List Row) :=
  match matchK st.g cl r with
  | m :: ms => .ok (st, (m :: ms).map (fun m => r.insert cl.var (.node m.id)))
  | [] =>
    match key cl r with
    | .error e => .error e
    | .ok _ =>
      match lookup st.cache (cl.op, key cl r) with
      | some pairs => .ok (st, [pairs.foldl (fun r p => r.insert p.1 p.2) r])     -- 122-135
      | none =>                                                                     -- 136-177
        let n : GNode := ⟨st.next, cl.label, cl.attr r⟩
        .ok (⟨st.g ++ [n], st.next + 1, ((cl.op, key cl r), [(cl.var, .node st.next)]) :: st.cache⟩,
             [r.insert cl.var (.node st.next)])

/-- A clause over a stream of rows (each row one batch: the repros have one row per MERGE). -/
def mergeRows (key : MClause → Row → Key) (cl : MClause) :
    St → List Row → Except String (St × List Row)
  | st, [] => .ok (st, [])
  | st, r :: rs => match mergeRowK key cl st r with
    | .error e => .error e
    | .ok (st', o) => match mergeRows key cl st' rs with
      | .error e => .error e
      | .ok (st'', os) => .ok (st'', o ++ os)

/-! ## The properties -/

/-- **MERGE takes a pattern variable as given only if the row binds it to a node.** Whenever the
pattern key is built from the variable's value (rather than its labels and properties), the
row has the variable's bound bit set and the value is a node. -/
def KeyFromBoundNodeOnly (key : MClause → Row → Key) (cl : MClause) (r : Row) : Prop :=
  ∀ v, keyVal (key cl r) = some v → r.isBound cl.var = true ∧ ∃ id, v = .node id

/-- **After MERGE the variable is bound to its own node.** For a row that does not bind the
pattern variable, every output row binds it to a node of the resulting graph that carries the
pattern's label and properties. -/
def MergeBindsOwnNode (key : MClause → Row → Key) (cl : MClause) (st : St) (r : Row) : Prop :=
  ∀ st' outs, mergeRowK key cl st r = .ok (st', outs) →
    ∀ o ∈ outs, o.isBound cl.var = true ∧
      ∃ n ∈ st'.g, o.get cl.var = some (.node n.id) ∧ n.label = cl.label ∧ n.attr = cl.attr r

/-! ## The fix reaches both properties -/

theorem fixKey_unbound (cl : MClause) (r : Row) (h : r.isBound cl.var = false) :
    fixKey cl r = mergeHashInput (T := Unit) (fun _ => none) (attrsOf cl r) [nodeOf cl] [] := by
  simp [fixKey, mergeHashInput, concatM, nodeToks, nodeOf, boundNodeGet, h]

theorem fixKey_from_bound_node (cl : MClause) (r : Row) : KeyFromBoundNodeOnly fixKey cl r := by
  intro v hv
  unfold fixKey at hv
  cases hb : r.isBound cl.var with
  | false =>
    simp only [mergeHashInput, concatM, nodeToks, nodeOf, boundNodeGet, hb] at hv
    simp only [Bool.false_eq_true, ite_false, attrsOf] at hv
    split at hv <;> simp [bind, Except.bind, pure, Except.pure, firstNull, keyVal] at hv
  | true =>
    cases hg : r.get cl.var with
    | none =>
      simp only [mergeHashInput, concatM, nodeToks, nodeOf, boundNodeGet, hb, hg] at hv
      simp only [ite_true, attrsOf] at hv
      split at hv <;> simp [bind, Except.bind, pure, Except.pure, firstNull, keyVal] at hv
    | some w =>
      cases w with
      | node id =>
        simp [mergeHashInput, concatM, nodeToks, nodeOf, boundNodeGet, hb, hg, bind, Except.bind,
          pure, Except.pure, keyVal] at hv
        exact ⟨rfl, id, hv.symm⟩
      | _ =>
        simp only [mergeHashInput, concatM, nodeToks, nodeOf, boundNodeGet, hb, hg] at hv
        simp only [ite_true, attrsOf] at hv
        split at hv <;> simp [bind, Except.bind, pure, Except.pure, firstNull, keyVal] at hv

/-- The cache entries a fixed MERGE can hit for this clause's label point at one of this
clause's nodes (true of a per-clause cache, C's `unique_entities`; the cross-clause sharing is
the separate `Merge.cross_clause_unbound`). -/
def CacheOK (cl : MClause) (st : St) : Prop :=
  ∀ r k pairs, k = fixKey cl r → r.isBound cl.var = false → lookup st.cache (cl.op, k) = some pairs →
    ∃ n ∈ st.g, pairs = [(cl.var, .node n.id)] ∧ n.label = cl.label ∧ n.attr = cl.attr r

theorem mem_matchK {g : Graph} {cl : MClause} {r : Row} {n : GNode} (h : n ∈ matchK g cl r) :
    n ∈ g ∧ n.label = cl.label ∧ n.attr = cl.attr r := by
  simp [matchK] at h; exact ⟨h.1, h.2.1, h.2.2⟩

/-- **The fixed MERGE satisfies `MergeBindsOwnNode`** on every row that does not bind the
variable, given a consistent cache. -/
theorem fixed_binds_own_node (cl : MClause) (st : St) (r : Row) (hu : r.isBound cl.var = false)
    (hc : CacheOK cl st) : MergeBindsOwnNode fixKey cl st r := by
  intro st' outs h o ho
  unfold mergeRowK at h
  split at h
  · rename_i m ms hm
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp only [List.mem_map] at ho
    obtain ⟨n, hn, rfl⟩ := ho
    have := mem_matchK (hm ▸ hn)
    exact ⟨Row.isBound_insert_self _ _ _, n, this.1, Row.get_insert_self _ _ _, this.2⟩
  · split at h
    · cases h
    · split at h
      · rename_i pairs hl
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨n, hn, rfl, hl1, hl2⟩ := hc r _ pairs rfl hu hl
        simp only [List.mem_singleton] at ho; subst ho
        exact ⟨Row.isBound_insert_self _ _ _, n, hn, Row.get_insert_self _ _ _, hl1, hl2⟩
      · simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp only [List.mem_singleton] at ho; subst ho
        exact ⟨Row.isBound_insert_self _ _ _, ⟨st.next, cl.label, cl.attr r⟩, by simp,
          Row.get_insert_self _ _ _, rfl, rfl⟩

/-! ## The repros (fresh graph): rows restated from ops_aggregate `SlotReuse` -/

/-- Labels `A, B, C` = 0, 1, 2; strings `'a1', 'b1', 'c1'` = 10, 11, 12. -/
def clA : MClause := ⟨1, 0, 0, 0⟩      -- UNWIND … AS r0 MERGE (a:A {id: r0.id}):      r0, a = 0, 1
def clB : MClause := ⟨2, 1, 1, 1⟩      -- WITH count(*) AS c0 UNWIND … AS r1 MERGE (b:B …): c0, r1, b = 0, 1, 2
def clC68 : MClause := ⟨2, 2, 1, 2⟩    -- #3368: WITH count(*) AS c1 UNWIND … AS r2 MERGE (c:C …): c1, r2, c = 0, 1, 2
def clC69 : MClause := ⟨3, 2, 2, 2⟩    -- #3369: WITH c0, count(*) AS c1 UNWIND … AS r2 MERGE (c:C …): c0, c1, r2, c = 0..3

def st0 : St := ⟨[], 0, []⟩
def row1 : Row := Row.ofCells [.bound (.map 10)]

/-- Segment 2's MERGE input: `c0` = 1, `r1`, and the first `count(*)`'s accumulator left at
slot 2 (= `b`), value-only (ops_aggregate `countAcc_left_at_2`). `acc` is its value. -/
def row2 (acc : MV) : Row := Row.ofCells [.bound (.int 1), .bound (.map 11), .valueOnly acc]
/-- #3368's last MERGE input: `c1` = 1, `r2`, the second count's accumulator at slot 2 (= `c`). -/
def row3_68 (acc : MV) : Row := Row.ofCells [.bound (.int 1), .bound (.map 12), .valueOnly acc]
/-- #3369's last MERGE input: `c0`, `c1`, `r2`, and `WITH c0, count(*)`'s accumulator at slot 3
(= `c`), value-only (ops_aggregate `bug3369_count_acc_in_merge_slot`). -/
def row3_69 (acc : MV) : Row :=
  Row.ofCells [.bound (.int 1), .bound (.int 1), .bound (.map 12), .valueOnly acc]

/-- Run segments 1 and 2, then the last MERGE on `last`; returns (final state, last outputs). -/
def run (shared : Bool) (key : MClause → Row → Key) (acc1 : MV) (cl : MClause) (last : Row) :
    Except String (St × List Row) :=
  let adj : MClause → MClause := if shared then MClause.erase else id
  match mergeRows key (adj clA) st0 [row1] with
  | .error e => .error e
  | .ok (s1, _) => match mergeRows key (adj clB) s1 [row2 acc1] with
    | .error e => .error e
    | .ok (s2, _) => mergeRows key (adj cl) s2 [last]

def outVar (res : Except String (St × List Row)) (i : Nat) : Option (Option MV) :=
  match res with
  | .ok (_, [o]) => some (o.read i)
  | _ => none

def nodes (res : Except String (St × List Row)) : Option Graph :=
  match res with
  | .ok (s, _) => some s.g
  | _ => none

/-- The state the last MERGE starts from (after segments 1 and 2). -/
def before (shared : Bool) (key : MClause → Row → Key) (acc1 : MV) : St :=
  let adj : MClause → MClause := if shared then MClause.erase else id
  match mergeRows key (adj clA) st0 [row1] with
  | .ok (s1, _) => match mergeRows key (adj clB) s1 [row2 acc1] with
    | .ok (s2, _) => s2
    | .error _ => st0
  | .error _ => st0

/-! ### #3368 -/

/-- **#3368, merge side.** Both counts are 1 and both sit, value-only, at the MERGE variable's
slot 2. `b`'s MERGE hashes `Int 1` instead of `:B {id:'b1'}`, misses, creates `B b1` (node 1)
and caches `[(2, node 1)]` under that key; `c`'s MERGE hashes the same `Int 1`, hits, and binds
`c` to node 1: `labels(c), c.id` = `[B], 'b1'`, 2 nodes. Live (main): exactly that; C (and the
fixed key here): `[C], 'c1'`, 3 nodes. -/
theorem pre3050_bug3368_merge_takes_leftover_as_node :
    keyVal (rustKey clC68 (row3_68 (.int 1))) = some (.int 1) ∧
    (row3_68 (.int 1)).isBound clC68.var = false ∧
    ¬ KeyFromBoundNodeOnly rustKey clC68 (row3_68 (.int 1)) ∧
    outVar (run true rustKey (.int 1) clC68 (row3_68 (.int 1))) 2 = some (some (.node 1)) ∧
    nodes (run true rustKey (.int 1) clC68 (row3_68 (.int 1))) =
      some [⟨0, 0, .str 10⟩, ⟨1, 1, .str 11⟩] ∧
    ¬ MergeBindsOwnNode rustKey clC68.erase (before true rustKey (.int 1)) (row3_68 (.int 1)) ∧
    -- the fixed key: c = a new `C c1` node, 3 nodes (C's answer)
    outVar (run true fixKey (.int 1) clC68 (row3_68 (.int 1))) 2 = some (some (.node 2)) ∧
    nodes (run true fixKey (.int 1) clC68 (row3_68 (.int 1))) =
      some [⟨0, 0, .str 10⟩, ⟨1, 1, .str 11⟩, ⟨2, 2, .str 12⟩] := by
  refine ⟨by decide, by decide, fun h => ?_, by decide, by decide, fun h => ?_, by decide, by decide⟩
  · have := (h (.int 1) (by decide)).2
    obtain ⟨_, h⟩ := this; cases h
  · have := h _ _ rfl _ (List.mem_singleton_self _)
    revert this; decide

/-- The leftover only matters through key equality. Live on main, each matching the model:
counts 2 then 1 (two `a` rows) → correct `[C] c1`; counts 2 and 2 (two `a`, two `b` rows) →
`[B] b1` (and live `b2` is never created either: its row also hashes `Int 2` and hits `b1`'s
entry — the same mechanism on one more row, not modelled here); `collect(1)` in both segments (leftover `[1]` twice) → `[B] b1`; `count(*)` then
`collect(1)` (`1` vs `[1]`) → correct. So it fires whenever the two leftovers are equal, list
or not; "does not fire for a list" holds only when a list meets an integer. -/
theorem pre3050_bug3368_fires_iff_leftovers_equal :
    outVar (run true rustKey (.int 2) clC68 (row3_68 (.int 1))) 2 = some (some (.node 2)) ∧
    outVar (run true rustKey (.int 2) clC68 (row3_68 (.int 2))) 2 = some (some (.node 1)) ∧
    outVar (run true rustKey (.list [1]) clC68 (row3_68 (.list [1]))) 2 = some (some (.node 1)) ∧
    outVar (run true rustKey (.int 1) clC68 (row3_68 (.list [1]))) 2 = some (some (.node 2)) := by
  decide

/-! ### #3369 -/

/-- **#3369, merge side.** `b`'s MERGE caches `[(2, node 1)]` under `Int 1` as above; `c` (id 3)
holds the second count `Int 1`, value-only, so its MERGE hits that entry and re-inserts
`(2, node 1)` — slot 2 is `r2` in this scope — leaving `c` an unbound `Int 1`, which the next
operator reads as the column value: `RETURN c` gives `1`, `c.id` (inside `collect(c.id)`) fails
with "Type mismatch: expected Map, Node, … but was Integer". Live (main): `RETURN c, c0, c1` →
`1, 1, 1` with 2 nodes, the issue query → that error; C: `'c1'`, 3 nodes. -/
theorem pre3050_bug3369_merge_leaves_int_in_node_var :
    keyVal (rustKey clC69 (row3_69 (.int 1))) = some (.int 1) ∧
    ¬ KeyFromBoundNodeOnly rustKey clC69 (row3_69 (.int 1)) ∧
    outVar (run true rustKey (.int 1) clC69 (row3_69 (.int 1))) 3 = some (some (.int 1)) ∧
    outVar (run true rustKey (.int 1) clC69 (row3_69 (.int 1))) 2 = some (some (.node 1)) ∧
    nodes (run true rustKey (.int 1) clC69 (row3_69 (.int 1))) =
      some [⟨0, 0, .str 10⟩, ⟨1, 1, .str 11⟩] ∧
    ¬ MergeBindsOwnNode rustKey clC69.erase (before true rustKey (.int 1)) (row3_69 (.int 1)) ∧
    outVar (run true fixKey (.int 1) clC69 (row3_69 (.int 1))) 3 = some (some (.node 2)) := by
  refine ⟨by decide, fun h => ?_, by decide, by decide, by decide, fun h => ?_, by decide⟩
  · obtain ⟨_, h⟩ := (h (.int 1) (by decide)).2; cases h
  · have := h _ _ rfl _ (List.mem_singleton_self _)
    revert this; decide

/-- #3369 controls (live on main, matching): `collect(1)` as the first fold (`[1]` vs `1`) →
`'c1'`; `collect(1)` in both segments → `RETURN c` gives `[1]`. -/
theorem pre3050_bug3369_controls :
    outVar (run true rustKey (.list [1]) clC69 (row3_69 (.int 1))) 3 = some (some (.node 2)) ∧
    outVar (run true rustKey (.list [1]) clC69 (row3_69 (.list [1]))) 3 = some (some (.list [1])) := by
  decide

/-! ## On main (#3050 keys the cache by `(operator idx, hash)`) -/

/-- **#3368 and #3369 as filed no longer reproduce on main.**  `b`'s and `c`'s MERGEs are
different operators, so `c`'s lookup of `(op c, hash Int 1)` misses `b`'s entry
`(op b, hash Int 1)`: `c` gets its own new node (id 2, three nodes), even with the
unchanged `rustKey`. Model only: the live repros were not re-run on main. -/
theorem bug3368_filed_repro_fixed_by_3050 :
    outVar (run false rustKey (.int 1) clC68 (row3_68 (.int 1))) 2 = some (some (.node 2)) ∧
    nodes (run false rustKey (.int 1) clC68 (row3_68 (.int 1))) =
      some [⟨0, 0, .str 10⟩, ⟨1, 1, .str 11⟩, ⟨2, 2, .str 12⟩] ∧
    outVar (run false rustKey (.int 1) clC69 (row3_69 (.int 1))) 3 = some (some (.node 2)) := by
  decide

/-- Segment 2's MERGE input for a second `UNWIND` element (`{id:'b2'}` = 13): the same
leftover accumulator. -/
def row2b (acc : MV) : Row := Row.ofCells [.bound (.int 1), .bound (.map 13), .valueOnly acc]

/-- State after segment 1 (`a`'s MERGE). -/
def seg1 (key : MClause → Row → Key) : St :=
  match mergeRows key clA st0 [row1] with
  | .ok (s, _) => s
  | .error _ => st0

/-- (node count, the MERGE variable's value on each output row). -/
def summ (res : Except String (St × List Row)) : Option (Nat × List (Option MV)) :=
  match res with
  | .ok (s, os) => some (s.g.length, os.map (fun o => o.read 2))
  | .error _ => none

/-- **Residual, inside one MERGE operator (model only, not re-run live).**  The key is still
built from the value-only leftover (`KeyFromBoundNodeOnly` still fails for `rustKey`), so two
rows of the SAME MERGE that carry the same leftover share one `(op, hash)` entry: the second
row (`b2`) hits the first row's entry, binds `b1` and creates nothing. `fixKey` creates both
(three nodes). The filed #3368 text already notes this ("its row also hashes `Int 2` and hits
`b1`'s entry"). -/
theorem same_op_leftover_shares_entry :
    summ (mergeRows rustKey clB (seg1 rustKey) [row2 (.int 1), row2b (.int 1)]) =
      some (2, [some (.node 1), some (.node 1)]) ∧
    summ (mergeRows fixKey clB (seg1 fixKey) [row2 (.int 1), row2b (.int 1)]) =
      some (3, [some (.node 1), some (.node 2)]) := by
  decide

end MergeSlot
