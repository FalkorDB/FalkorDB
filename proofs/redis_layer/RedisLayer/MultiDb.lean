import RedisLayer.GraphCore
/-!
# Several Redis databases over one `GRAPH_REGISTRY` (#2782, fixed by #3424)

The keyspace is keyed by `(db, key)`; the registry by graph identity (`GraphCore.Reg`), each
entry carrying the database and name of its key. `GRAPH.QUERY k "CREATE …"` in database `db`
(`graph_query`'s create/open path): an existing graph runs its write past the writer check
(`graph_is_registered`); an empty key gets a new graph, stored and then registered with
`register_graph(ctx, …)` at the context's selected database.

Every query here is a write (`CREATE`), so it reaches the writer check.
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
    ({ keys := ((db, k), g) :: s.keys, reg := register s.reg db k g, next := g + 1 }, .ok)

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

/-- **The invariant is kept by every query**: registering a new graph displaces no other
graph's entry, in whichever database (`GraphCore.register_keeps_others`). -/
theorem query_keeps (s : Srv) (db : Nat) (k : List Char) (hi : KeyspaceRegistered s) :
    KeyspaceRegistered (query s db k).1 := by
  unfold query
  split
  · exact hi
  · intro e he
    simp only [List.mem_cons] at he
    rcases he with rfl | he
    · exact register_registered _ _ _ _
    · exact isRegistered_register_mono _ _ _ _ _ (hi e he)

theorem run_ok (s : Srv) (qs : List (Nat × List Char)) (hi : KeyspaceRegistered s) :
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
    · exact ih _ (query_keeps s db k hi) r hr

/-- **#2782 fixed**: every sequence of writes, over any databases and names, succeeds. -/
theorem writes_succeed (qs : List (Nat × List Char)) : WritesSucceed qs :=
  run_ok Srv.empty qs empty_registered

/-- One name, two databases (the #2782 repro: db0, db1, db0): all three succeed, as in C. -/
def repro : List (Nat × List Char) := [(0, "shared".toList), (1, "shared".toList), (0, "shared".toList)]

theorem repro_ok : run Srv.empty repro = [.ok, .ok, .ok] := by decide

end RedisLayer.MultiDb
