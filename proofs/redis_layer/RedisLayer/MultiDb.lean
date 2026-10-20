import RedisLayer.GraphCore
/-!
# Several Redis databases over one name-keyed `GRAPH_REGISTRY` (#2782) — gap closed 2026-10-20

`GraphCore.Reg` modelled the registry over a single keyspace, so a graph name living in
two Redis databases (`SELECT 0` / `SELECT 1`) could not be expressed. Here the keyspace
is keyed by `(db, key)` while the registry stays keyed by the name only, as in Rust.

| here | there (fb3d516d0) |
| --- | --- |
| `Srv.keys`  | the Redis keyspace, one per database (`RedisModule_OpenKey` is per selected db) |
| `Srv.reg`   | `GRAPH_REGISTRY: Mutex<HashMap<String, Arc<…>>>` `src/graph_core.rs:136-138` — no db index |
| `query`     | `graph_query` `src/commands/query.rs:72-147`: open the key in the current db; existing graph → `query_mut`; absent → new graph, `set_value` at `:145`, `register_graph(name, graph)` `:146` (`GraphCore.register`, which displaces the old entry, `graph_core.rs:150`) |
| `writeCheck`| `QuerySession` writer escalation `src/query_session.rs:360`: `graph_is_registered` (`GraphCore.isRegistered`) else "graph was deleted or replaced while the query was running, aborting" |

Every query in the repro is a write (`CREATE`), so it reaches the writer check.
-/
namespace RedisLayer.MultiDb
open RedisLayer.GraphCore

structure Srv where
  keys : List ((Nat × List Char) × Nat)
  reg  : Reg
  next : Nat

def Srv.empty : Srv := ⟨[], [], 0⟩

def Srv.lookup (s : Srv) (db : Nat) (k : List Char) : Option Nat :=
  (s.keys.find? (·.1 = (db, k))).map (·.2)

inductive Res | ok | aborted deriving DecidableEq, Repr

/-- `GRAPH.QUERY k "CREATE …"` in database `db`. -/
def query (s : Srv) (db : Nat) (k : List Char) : Srv × Res :=
  match s.lookup db k with
  | some g => (s, if isRegistered s.reg g then .ok else .aborted)
  | none =>
    let g := s.next
    ({ keys := ((db, k), g) :: s.keys, reg := (register s.reg k g).1, next := g + 1 }, .ok)

def run (s : Srv) : List (Nat × List Char) → List Res
  | [] => []
  | (db, k) :: rest => let (s', r) := query s db k; r :: run s' rest

/-- **Every stored graph is registered** — the invariant the writer check
(`graph_is_registered`) relies on: a graph that some key in some database holds is in
the registry, so a write to it is never refused. -/
def KeyspaceRegistered (s : Srv) : Prop := ∀ e ∈ s.keys, isRegistered s.reg e.2 = true

/-- **Writes to a live graph succeed** (C parity): running the queries from the empty
server, none is aborted. -/
def WritesSucceed (qs : List (Nat × List Char)) : Prop := ∀ r ∈ run Srv.empty qs, r = .ok

theorem empty_registered : KeyspaceRegistered Srv.empty := by
  intro e he; simp [Srv.empty] at he

/-- Under the invariant, a query on an existing graph is never aborted. -/
theorem existing_ok (s : Srv) (db : Nat) (k : List Char) (g : Nat) (hi : KeyspaceRegistered s)
    (hg : s.lookup db k = some g) : (query s db k).2 = .ok := by
  have hm : ∃ e ∈ s.keys, e.2 = g := by
    unfold Srv.lookup at hg
    cases hf : s.keys.find? (·.1 = (db, k)) with
    | none => rw [hf] at hg; cases hg
    | some e => rw [hf] at hg; simp at hg; exact ⟨e, List.mem_of_find?_eq_some hf, hg⟩
  obtain ⟨e, he, rfl⟩ := hm
  simp [query, hg, hi e he]

/-- One name, two databases (the live repro: db0, db1, db0). -/
def repro : List (Nat × List Char) := [(0, "shared".toList), (1, "shared".toList), (0, "shared".toList)]

/-- **BUG #2782**: creating `shared` in db1 re-registers the name and displaces db0's graph,
so the db0 graph, still stored under its key, is no longer registered and the next write
to it aborts. Live: Rust `Labels added: 1`, `Labels added: 1`, then "graph was deleted
or replaced while the query was running, aborting"; C succeeds three times. -/
theorem bug2782_same_name_other_db_aborts :
    run Srv.empty repro = [.ok, .ok, .aborted] ∧ ¬ WritesSucceed repro := by
  have h : run Srv.empty repro = [.ok, .ok, .aborted] := by decide
  refine ⟨h, fun hw => ?_⟩
  have := hw .aborted (by rw [h]; simp); cases this

/-- The invariant breaks at the second create. -/
theorem bug2782_invariant_broken :
    KeyspaceRegistered (query Srv.empty 0 "shared".toList).1 ∧
    ¬ KeyspaceRegistered (query (query Srv.empty 0 "shared".toList).1 1 "shared".toList).1 := by
  refine ⟨?_, ?_⟩
  · intro e he
    simp [query, Srv.lookup, Srv.empty] at he
    subst he; decide
  · intro h
    have := h ((0, "shared".toList), 0) (by decide)
    revert this; decide

end RedisLayer.MultiDb
