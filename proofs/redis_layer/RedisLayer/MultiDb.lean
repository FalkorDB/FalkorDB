import RedisLayer.GraphCore
/-!
# Several Redis databases over one `GRAPH_REGISTRY` (#2782) — fixed by #3424

The keyspace is keyed by `(db, key)`. Until #3424 the registry was keyed by the name
only, so a graph name living in two Redis databases displaced the other database's
entry; since #3424 it is keyed by graph identity (`GraphCore.Reg`), each entry
recording the graph's database and key.

| here | there |
| --- | --- |
| `Srv.keys`  | the Redis keyspace, one per database (`RedisModule_OpenKey` is per selected db) |
| `Srv.reg`   | `GRAPH_REGISTRY: Mutex<HashMap<usize, RegisteredGraph>>` `src/graph_core.rs:165-167` (`GraphCore.Reg`) |
| `query`     | `graph_query` `src/commands/query.rs:42-157`: open the key in the current db; existing graph → `query_mut`; absent → new graph, `set_value`, `register_graph(ctx, name, graph)` `:155` (`GraphCore.register` at the selected db) |
| `writeCheck`| `QuerySession` writer escalation `src/query_session.rs:352`: `graph_is_registered` (`GraphCore.isRegistered`) else "graph was deleted or replaced while the query was running, aborting" |
| `Pre3424.*` | the name-keyed registry before #3424 (`343e077f2` `graph_core.rs:136-165`), kept for the historical counterexample |

Every query in the repro is a write (`CREATE`), so it reaches the writer check.
-/
namespace RedisLayer.MultiDb
open RedisLayer.GraphCore

structure Srv where
  keys : List ((Int × List Char) × Nat)
  reg  : Reg
  next : Nat

def Srv.empty : Srv := ⟨[], [], 0⟩

def Srv.lookup (s : Srv) (db : Int) (k : List Char) : Option Nat :=
  (s.keys.find? (·.1 = (db, k))).map (·.2)

inductive Res | ok | aborted deriving DecidableEq, Repr

/-- `GRAPH.QUERY k "CREATE …"` in database `db`. -/
def query (s : Srv) (db : Int) (k : List Char) : Srv × Res :=
  match s.lookup db k with
  | some g => (s, if isRegistered s.reg g then .ok else .aborted)
  | none =>
    let g := s.next
    ({ keys := ((db, k), g) :: s.keys, reg := register s.reg db k g, next := g + 1 }, .ok)

def run (s : Srv) : List (Int × List Char) → List Res
  | [] => []
  | (db, k) :: rest => let (s', r) := query s db k; r :: run s' rest

/-- **Every stored graph is registered** — the invariant the writer check
(`graph_is_registered`) relies on: a graph that some key in some database holds is in
the registry, so a write to it is never refused. -/
def KeyspaceRegistered (s : Srv) : Prop := ∀ e ∈ s.keys, isRegistered s.reg e.2 = true

/-- **Writes to a live graph succeed** (C parity): running the queries from the empty
server, none is aborted. -/
def WritesSucceed (qs : List (Int × List Char)) : Prop := ∀ r ∈ run Srv.empty qs, r = .ok

theorem empty_registered : KeyspaceRegistered Srv.empty := by
  intro e he; simp [Srv.empty] at he

/-- Under the invariant, a query on an existing graph is never aborted. -/
theorem existing_ok (s : Srv) (db : Int) (k : List Char) (g : Nat) (hi : KeyspaceRegistered s)
    (hg : s.lookup db k = some g) : (query s db k).2 = .ok := by
  have hm : ∃ e ∈ s.keys, e.2 = g := by
    unfold Srv.lookup at hg
    cases hf : s.keys.find? (·.1 = (db, k)) with
    | none => rw [hf] at hg; cases hg
    | some e => rw [hf] at hg; simp at hg; exact ⟨e, List.mem_of_find?_eq_some hf, hg⟩
  obtain ⟨e, he, rfl⟩ := hm
  simp [query, hg, hi e he]

/-- **#2782 fixed (#3424)**: every query keeps the invariant — creating a graph registers
it and displaces nothing (`register_keeps_registered`). -/
theorem query_registered (s : Srv) (db : Int) (k : List Char) (hi : KeyspaceRegistered s) :
    KeyspaceRegistered (query s db k).1 := by
  unfold query
  cases hl : s.lookup db k with
  | some g => exact hi
  | none =>
    intro e he
    simp only [List.mem_cons] at he
    rcases he with rfl | he
    · exact register_registered s.reg db k s.next
    · exact register_keeps_registered s.reg db k s.next e.2 (hi e he)

theorem run_ok (s : Srv) (qs : List (Int × List Char)) (hi : KeyspaceRegistered s) :
    ∀ r ∈ run s qs, r = .ok := by
  induction qs generalizing s with
  | nil => simp [run]
  | cons q qs ih =>
    obtain ⟨db, k⟩ := q
    intro r hr
    simp only [run, List.mem_cons] at hr
    rcases hr with rfl | hr
    · cases hl : s.lookup db k with
      | some g => exact existing_ok s db k g hi hl
      | none => simp [query, hl]
    · exact ih _ (query_registered s db k hi) r hr

/-- **#2782 fixed (#3424)**: from the empty server, no write to a live graph is ever
refused, whatever databases and names the queries use. -/
theorem writes_succeed (qs : List (Int × List Char)) : WritesSucceed qs :=
  run_ok Srv.empty qs empty_registered

/-- One name, two databases (the live repro: db0, db1, db0). -/
def repro : List (Int × List Char) := [(0, "shared".toList), (1, "shared".toList), (0, "shared".toList)]

/-- The #2782 repro now succeeds three times, as in C. -/
theorem fixed2782_repro : run Srv.empty repro = [.ok, .ok, .ok] := by decide

/-! ## Historical: before #3424 (`343e077f2`), a name-keyed registry -/

namespace Pre3424

abbrev Reg := List (List Char × Nat)
def Reg.insert (r : Reg) (k : List Char) (g : Nat) : Reg := (k, g) :: r.filter (·.1 ≠ k)
def Reg.get (r : Reg) (k : List Char) : Option Nat := (r.find? (·.1 = k)).map (·.2)
/-- `register_graph(name, arc)`: `insert(name, arc)`, displacing the old entry. -/
def register (r : Reg) (k : List Char) (g : Nat) : Reg × Option Nat := (r.insert k g, r.get k)
/-- `graph_is_registered`: by allocation, under any key. -/
def isRegistered (r : Reg) (g : Nat) : Bool := r.any (·.2 = g)

structure Srv where
  keys : List ((Int × List Char) × Nat)
  reg  : Reg
  next : Nat

def Srv.empty : Srv := ⟨[], [], 0⟩

def Srv.lookup (s : Srv) (db : Int) (k : List Char) : Option Nat :=
  (s.keys.find? (·.1 = (db, k))).map (·.2)

def query (s : Srv) (db : Int) (k : List Char) : Srv × Res :=
  match s.lookup db k with
  | some g => (s, if isRegistered s.reg g then .ok else .aborted)
  | none =>
    let g := s.next
    ({ keys := ((db, k), g) :: s.keys, reg := (register s.reg k g).1, next := g + 1 }, .ok)

def run (s : Srv) : List (Int × List Char) → List Res
  | [] => []
  | (db, k) :: rest => let (s', r) := query s db k; r :: run s' rest

def KeyspaceRegistered (s : Srv) : Prop := ∀ e ∈ s.keys, isRegistered s.reg e.2 = true
def WritesSucceed (qs : List (Int × List Char)) : Prop := ∀ r ∈ run Srv.empty qs, r = .ok

/-- **BUG #2782 before #3424** (fixed by #3424): creating `shared` in db1 re-registered the
name and displaced db0's graph, so the db0 graph, still stored under its key, was no
longer registered and the next write to it aborted. Live then: Rust `Labels added: 1`,
`Labels added: 1`, then "graph was deleted or replaced while the query was running,
aborting"; C succeeds three times. -/
theorem pre3424_bug2782_same_name_other_db_aborts :
    run Srv.empty repro = [.ok, .ok, .aborted] ∧ ¬ WritesSucceed repro := by
  have h : run Srv.empty repro = [.ok, .ok, .aborted] := by decide
  refine ⟨h, fun hw => ?_⟩
  have := hw .aborted (by rw [h]; simp); cases this

/-- Before #3424 the invariant broke at the second create. -/
theorem pre3424_bug2782_invariant_broken :
    KeyspaceRegistered (query Srv.empty 0 "shared".toList).1 ∧
    ¬ KeyspaceRegistered (query (query Srv.empty 0 "shared".toList).1 1 "shared".toList).1 := by
  refine ⟨?_, ?_⟩
  · intro e he
    simp [query, Srv.lookup, Srv.empty] at he
    subst he; decide
  · intro h
    have := h ((0, "shared".toList), 0) (by decide)
    revert this; decide

end Pre3424

end RedisLayer.MultiDb
