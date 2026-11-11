import OpsTraverse.CondTraverse
/-
# How an Aggregate output row feeds a later CondTraverse (#3372)

Every other file here proves CondTraverse on an abstract `RowIn` (the source and
destination bindings it reads). This file models where those bindings come
from when the source is an Aggregate across a `WITH`: the binder's slot ids, the
row the Aggregate emits, and the slot reads in CondTraverse.

| here | there |
| --- | --- |
| `Val`, `Slots`, `get`, `ins` | a batch row: `Batch::value_at` (`runtime/batch.rs:1263-1272`) reads a column by `Variable.id` alone, never by `scope_id` |
| `declare`, `declareAll` | `Binder::fresh_var` (`planner/binder.rs:2255-2269`: the id is the current scope's env length) via `define_name_in_scope` (`:2155-2175`) and `project_name` (`:2177-2185`); a projection opens a fresh scope (`push_scope`, `:1156`) so its ids restart at 0; `bind_graph` declares nodes before relationships (`:1351`, `:1367`) |
| `aggRow` | `AggregateOp::next` (`runtime/ops/aggregate.rs:1150-1213`): agg outputs (`:1177`), then the *pre-projection* slot of every bare-variable key (`:1187-1196`), then the projected keys (`:1197`) |
| `inputRow` | the endpoint reads of `CondTraverseOp::expand_row` (`runtime/ops/cond_traverse.rs:915-928`), and the same destination read on the batched path (`:799-803`, skip on mismatch) |
| `hopOut` | `expandRow` in collapse mode (anonymous edge) on that `RowIn`, then `set_column(to_alias.id, …)` (`cond_traverse.rs:875`) |

## The bug

The binder numbers each scope from 0, and a batch slot is keyed by id only.
`aggregate.rs:1187-1196` re-binds every bare-variable grouping key at its
*pre-projection* id ("so downstream operators can find them"), an id of the
previous scope. A variable that the next MATCH declares fresh can get that same
id; CondTraverse then reads the stale node as an already-bound destination and
keeps only edges that end there (`cond_traverse.rs:799-803`, `:922-928`).

`MATCH (a)-[:E]->(b)-[:E]->(c) WITH c, count(*) AS k MATCH (c)-[:E]->(d)`:
scope 0 is `[a, b, c, _anon, _anon]`, so the old `c` is slot 2; scope 1 is
`[c, k]`, so `d` is slot 2. Every row carries `d = c`, and with no self-loop
nothing matches.

Live (Rust main 9d3869aa8, port 6660, the issue's graph): the issue query
returns 0 (expected 4; C 4.20.7 returns 4 per the issue); the narrowed query 0
(expected 4); `WITH c, count(*) AS k, 1 AS z …` returns 4 (`d` moves to slot 3);
a three-hop prefix (old `c` = slot 3) returns 2 without `z` and 0 with it. Each
of those numbers is a theorem below.

Gap closed 2026-10-20.
-/

namespace OpsTraverse.SlotFeed

open OpsTraverse

/-! ## Rows, slots and the binder -/

/-- The value kinds that matter here: a node id or a scalar (`count(*)`). -/
inductive Val where
  | node (n : Nat)
  | int  (n : Nat)
deriving DecidableEq, Repr

/-- A batch row: bindings by slot id (`Batch::value_at`, `batch.rs:1263`). The
first binding of a slot wins, which `ins` keeps unique. -/
abbrev Slots := List (Nat × Val)

def get (r : Slots) (i : Nat) : Option Val := (r.find? (·.1 == i)).map (·.2)

/-- `Row::insert` / `Batch::set_column`: replace the slot's binding. -/
def ins (r : Slots) (i : Nat) (v : Val) : Slots := (i, v) :: r.filter (·.1 != i)

/-- A variable name of the repro queries; `anon i` is the binder's `_anon_i`. -/
inductive Nm where
  | a | b | c | d | e | x | k | z
  | anon (i : Nat)
deriving DecidableEq, Repr

def Nm.isAnon : Nm → Bool
  | .anon _ => true
  | _ => false

/-- Position of a name in a scope's env. -/
def pos (env : List Nm) (n : Nm) : Option Nat :=
  match env with
  | [] => none
  | y :: ys => if y = n then some 0 else (pos ys n).map (· + 1)

/-- `define_name_in_scope` (`binder.rs:2155`): an existing name keeps its id,
an anonymous one (`_anon…`) or a new one gets `fresh_var`'s id, the scope's env
length (`binder.rs:2261`). -/
def declare (env : List Nm) (n : Nm) : List Nm :=
  if n.isAnon then env ++ [n]
  else match pos env n with
    | some _ => env
    | none => env ++ [n]

def declareAll (env : List Nm) (ns : List Nm) : List Nm := ns.foldl declare env

/-- The slot a name resolves to (0 if absent; only used on declared names). -/
def slot (env : List Nm) (n : Nm) : Nat := (pos env n).getD 0

/-! ## The Aggregate output row -/

/-- A grouping key: its projected slot, and `some` pre-projection slot when the
key expression is a bare variable (`ExprIR::Variable(original_var)`). -/
structure Key where
  newId : Nat
  oldId : Option Nat

/-- `AggregateOp::next`, one group (`aggregate.rs:1150-1213`): `acc` starts
with the aggregate outputs (`:1177`); every bare-variable key is inserted at
its pre-projection id unless an output owns that id (`:1187-1196`); then the
projected keys are merged in (`:1197`). The accumulator unbind and the output
re-bind (`:1201-1211`) touch only accumulator/output slots and are omitted. -/
def aggRow (keys : List (Key × Val)) (aggs : List (Nat × Val)) : Slots :=
  let acc1 := keys.foldl (fun acc kv =>
    match kv.1.oldId with
    | some o => if aggs.any (·.1 == o) then acc else ins acc o kv.2
    | none => acc) aggs
  keys.foldl (fun acc kv => ins acc kv.1.newId kv.2) acc1

/-! ## What CondTraverse reads -/

def nodeOf : Option Val → Option Nat
  | some (.node n) => some n
  | _ => none

/-- The endpoint reads of `expand_row` (`cond_traverse.rs:915-928`): a slot
bound to a node is a bound endpoint; a slot bound to anything else drops the
row (`return Ok(())`); an unbound slot is free. -/
def inputRow (r : Slots) (fromId toId : Nat) : Option RowIn :=
  match get r fromId, get r toId with
  | some (.int _), _ => none
  | _, some (.int _) => none
  | f, t => some ⟨nodeOf f, nodeOf t, []⟩

/-- One anonymous directed hop from `fromId` to `toId`: the collapsed
expansion (`expandRow … false false`), each result written at `toId`
(`cond_traverse.rs:875`). -/
def hopOut (g : Graph) (fromId toId : Nat) (r : Slots) : List Slots :=
  match inputRow r fromId toId with
  | none => []
  | some ri => (expandRow g false false ri).map fun t => ins r toId (.node t.2.1)

/-! ## The property a fix must establish -/

/-- **Property (should hold).** Crossing a `WITH`, the row an operator emits
binds only the slots the new scope declared. Then a variable the next clause
declares fresh starts unbound, whatever id it gets. -/
def ScopedRow (declared : List Nat) (r : Slots) : Prop := ∀ p ∈ r, p.1 ∈ declared

theorem get_none_of_not_mem {r : Slots} {i : Nat} (h : ∀ p ∈ r, p.1 ≠ i) : get r i = none := by
  unfold get
  have : r.find? (·.1 == i) = none := by
    rw [List.find?_eq_none]; intro p hp; simpa using h p hp
  simp [this]

/-- **What the property buys.** On a scoped row a fresh destination slot is
free, so the traverse reads exactly `⟨source, none, []⟩` and the existing
single-hop theorems (`collapse_directed_iff`, `batched_eq_collapse`) give the
source's true out-neighbours. A fix that makes `aggRow` scoped proves #3372
closed through this lemma. -/
theorem scoped_inputRow {declared : List Nat} {r : Slots} (hs : ScopedRow declared r)
    {fromId toId n : Nat} (hf : get r fromId = some (.node n)) (ht : toId ∉ declared) :
    inputRow r fromId toId = some ⟨some n, none, []⟩ := by
  have hn : get r toId = none :=
    get_none_of_not_mem fun p hp he => ht (he ▸ hs p hp)
  simp [inputRow, hf, hn, nodeOf]

/-! ## The query, parameterised by the binder layout -/

/-- `MATCH <prefix ending in c> WITH c, count(*) AS k, <extra constants>
MATCH (c)-[:E]->(d)[-[:E]->(e)]`: the scope-0 env (nodes, then anonymous
relationships), the projected names and the second MATCH's names. -/
structure Q where
  scope0 : List Nm
  proj   : List Nm
  match2 : List Nm

def Q.env1 (q : Q) : List Nm := declareAll (declareAll [] q.proj) q.match2
def Q.cOld (q : Q) : Nat := slot (declareAll [] q.scope0) .c
def Q.cNew (q : Q) : Nat := slot q.env1 .c
def Q.k (q : Q) : Nat := slot q.env1 .k
def Q.d (q : Q) : Nat := slot q.env1 .d
def Q.e (q : Q) : Nat := slot q.env1 .e
/-- The slots the projection declared (what a scoped row may bind). -/
def Q.declared (q : Q) : List Nat := q.proj.map (slot q.env1)

/-- The row Aggregate emits for group `(c, n)` on main. Extra projected
constants are keys with no pre-projection slot. -/
def Q.groupRow (q : Q) (c n : Nat) : Slots :=
  aggRow ((⟨q.cNew, some q.cOld⟩, .node c) ::
      ((q.proj.filter fun y => y != .c && y != .k).map fun y => (⟨slot q.env1 y, none⟩, .int 1)))
    [(q.k, .int n)]

/-- The same row restricted to the declared slots: what a fix must emit. -/
def Q.scopedRow (q : Q) (c n : Nat) : Slots :=
  (q.groupRow c n).filter fun p => q.declared.contains p.1

/-- Groups `(c, number of prefix matches ending at c)`. -/
def groups (cs : List Nat) : List (Nat × Nat) := (dedup cs).map fun c => (c, cs.count c)

/-- Anonymous-hop walk ends of length `len` from every node (the scope-0 prefix,
collapse semantics: one row per walk of node pairs). -/
def walkEnds (g : Graph) (nodes : List Nat) : Nat → List Nat
  | 0 => nodes
  | len + 1 =>
    (walkEnds g nodes len).flatMap fun s =>
      (expandRow g false false ⟨some s, none, []⟩).map (·.2.1)

/-- `RETURN count(*)` after one outgoing hop from `c` (narrowed repro). -/
def oneHopCount (q : Q) (g : Graph) (row : Nat → Nat → Slots) (cs : List Nat) : Nat :=
  ((groups cs).flatMap fun gn => hopOut g q.cNew q.d (row gn.1 gn.2)).length

/-- The issue query: `… MATCH (c)-[:E]->(d)-[:E]->(e) WITH c, in2, count(*) AS out2
RETURN sum(in2 * out2)`. -/
def splitProduct (q : Q) (g : Graph) (row : Nat → Nat → Slots) (cs : List Nat) : Nat :=
  ((groups cs).map fun gn =>
    gn.2 * ((hopOut g q.cNew q.d (row gn.1 gn.2)).flatMap (hopOut g q.d q.e)).length).sum

/-! ## The issue's graph and the live numbers -/

/-- `v1..v7` are node ids `0..6`; edges in creation order:
`v1→v2, v2→v3, v1→v6, v6→v3, v3→v4, v4→v5, v3→v7, v7→v5`. -/
def gDag : Graph :=
  [⟨0, 0, 1⟩, ⟨1, 1, 2⟩, ⟨2, 0, 5⟩, ⟨3, 5, 2⟩, ⟨4, 2, 3⟩, ⟨5, 3, 4⟩, ⟨6, 2, 6⟩, ⟨7, 6, 4⟩]
def nodes7 : List Nat := [0, 1, 2, 3, 4, 5, 6]

/-- Two-hop prefix `(a)-[:E]->(b)-[:E]->(c)`, `WITH c, count(*) AS k`. -/
def q2 (extra : List Nm) (match2 : List Nm) : Q :=
  { scope0 := [.a, .b, .c, .anon 0, .anon 1],
    proj := [.c, .k] ++ extra, match2 := match2 }
/-- Three-hop prefix `(a)-[:E]->(b)-[:E]->(x)-[:E]->(c)`. -/
def q3 (extra : List Nm) : Q :=
  { scope0 := [.a, .b, .x, .c, .anon 0, .anon 1, .anon 2],
    proj := [.c, .k] ++ extra, match2 := [.c, .d, .anon 3] }

def cs2 : List Nat := walkEnds gDag nodes7 2
def cs3 : List Nat := walkEnds gDag nodes7 3

/-- The binder layout: the old `c` and the fresh `d` share slot 2. -/
theorem bug3372_layout :
    (q2 [] [.c, .d, .e, .anon 2, .anon 3]).cOld = 2 ∧
    (q2 [] [.c, .d, .e, .anon 2, .anon 3]).d = 2 ∧
    (q2 [] [.c, .d, .e, .anon 2, .anon 3]).e = 3 ∧
    (q2 [] [.c, .d, .e, .anon 2, .anon 3]).cNew = 0 := by decide

/-- **Bug #3372, the cause.** The Aggregate row for group `c = v3` binds slot 2
(the pre-projection `c`), which scope 1 never declared: `ScopedRow` fails. -/
theorem bug3372_aggregate_leaks_preprojection_slot :
    ¬ ScopedRow (q2 [] [.c, .d, .anon 2]).declared
      ((q2 [] [.c, .d, .anon 2]).groupRow 2 2) := by
  intro h
  exact absurd (h (2, .node 2) (by decide)) (by decide)

/-- **Bug #3372, the issue query.** `sum(in2 * out2)` is 0 on main; on scoped
rows it is 4 (the continuous 4-hop count, Neo4j, Memgraph, C 4.20.7). Live
Rust main: 0. -/
theorem bug3372_split_product_zero :
    let q := q2 [] [.c, .d, .e, .anon 2, .anon 3]
    splitProduct q gDag q.groupRow cs2 = 0 ∧
    splitProduct q gDag q.scopedRow cs2 = 4 := by decide

/-- **Bug #3372, narrowed.** `WITH c, count(*) AS k MATCH (c)-[:E]->(d)
RETURN count(*)`: main 0, scoped 4 (live: 0). -/
theorem bug3372_slot_reuse_drops_rows :
    let q := q2 [] [.c, .d, .anon 2]
    oneHopCount q gDag q.groupRow cs2 = 0 ∧
    oneHopCount q gDag q.scopedRow cs2 = 4 := by decide

/-- The live probes that pin the cause to the slot collision, not to the
traverse: one extra projected constant moves `d` to slot 3 and the two-hop
query recovers (live 4); a three-hop prefix puts the old `c` at slot 3, so it
is right without the constant (live 2) and wrong with it (live 0). -/
theorem bug3372_shift_probes :
    let q2z := q2 [.z] [.c, .d, .anon 2]
    oneHopCount q2z gDag q2z.groupRow cs2 = 4 ∧
    oneHopCount (q3 []) gDag (q3 []).groupRow cs3 = 2 ∧
    oneHopCount (q3 [.z]) gDag (q3 [.z]).groupRow cs3 = 0 ∧
    oneHopCount (q3 [.z]) gDag (q3 [.z]).scopedRow cs3 = 2 := by decide

end OpsTraverse.SlotFeed
