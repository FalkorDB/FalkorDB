/-!
# `GRAPH.RO_QUERY` never admits a write

The write detection is split over three places, all modelled here:

1. the parser's `write` flag (`graph/src/parser/cypher.rs:780-910`,
   `parse_single_query`): set by any of CREATE/MERGE/DELETE/DETACH/SET/REMOVE/FOREACH,
   handed to the next WITH/RETURN and reset after it; the final `Query { write }` carries
   the flag of a trailing updating segment;
2. the planner (`graph/src/planner/mod.rs:2416-2420`, `:2747-2751`): a WITH/RETURN
   whose flag is set gets a `Commit` child; a query whose final flag is set is wrapped in
   `Commit`; `CREATE/DROP INDEX` become `CreateIndex`/`DropIndex`
   (`planner/mod.rs:3170-3194`), and so does `CALL db.idx.fulltext.drop`
   (`planner/mod.rs:2844`);
3. `execute_query` (`src/graph_core.rs:597-608`): a plan with any `Commit`,
   `CreateIndex` or `DropIndex` node is a write, and RO_QUERY refuses it; and the
   runtime guards for what the plan scan cannot see: write *procedures*
   (`runtime/eval.rs:936`, `:956` — `!rt.write && func.write`) and index DDL
   (`runtime/index_ddl.rs:38`, `:90`).

`Runtime::new(.., is_write = false, ..)` for every plan that reaches execution on the
read path (`graph_core.rs:619-633`), so `rt.write = false` there.
-/

namespace RedisLayer.Ro

/-- A clause as the parser sees it. `call w` is a procedure call; `w` is its
`write procedure:` registration (`functions/mod.rs:355`). `sub q` is `CALL { q }`. -/
inductive Clause where
  | read
  | call (writeProc : Bool)
  | ftDrop                -- `CALL db.idx.fulltext.drop(...)`
  | update                -- CREATE / MERGE / DELETE / SET / REMOVE / FOREACH
  | createIndex | dropIndex
  | with_
  | ret
  | sub (q : List Clause)

/-- Plan operators, reduced to what the write check looks at. -/
inductive Op where
  | commit | createIndex | dropIndex
  | call (writeProc : Bool)
  | update
  | other
  deriving DecidableEq

/-- A plan tree, flattened to the list `plan.iter()` visits (order is irrelevant to
`any`). -/
abbrev Plan := List Op

mutual
/-- Parser + planner for one query. `w` is the running `write` flag. -/
def plan : List Clause → Bool → Plan
  | [], w => if w then [.commit] else []                 -- `planner/mod.rs:2749`
  | .read :: cs, w => .other :: plan cs w
  | .call p :: cs, w => .call p :: plan cs w
  | .ftDrop :: cs, w => .dropIndex :: plan cs w
  | .update :: cs, _ => .update :: plan cs true          -- `cypher.rs:824`
  | .createIndex :: cs, w => .createIndex :: plan cs w
  | .dropIndex :: cs, w => .dropIndex :: plan cs w
  | .with_ :: cs, w => (if w then [.other, .commit] else [.other]) ++ plan cs false  -- `:829`, `:879`
  | .ret :: cs, w => (if w then [.other, .commit] else [.other]) ++ plan cs false    -- `:882-883`
  | .sub q :: cs, w => .other :: planSub q ++ plan cs w
def planSub : List Clause → Plan
  | q => plan q false
end

/-- `plan.iter().any(|n| matches!(n, Commit | CreateIndex | DropIndex))`. -/
def isW : Op → Bool
  | .commit | .createIndex | .dropIndex => true
  | _ => false

def isWritePlan (p : Plan) : Bool := p.any isW

-- A clause list that mutates the graph or its schema.
mutual
def mutates : List Clause → Bool
  | [] => false
  | c :: cs => mutatesC c || mutates cs
def mutatesC : Clause → Bool
  | .update | .createIndex | .dropIndex | .ftDrop | .call true => true
  | .sub q => mutates q
  | _ => false
end

/-- The outcome of RO_QUERY: rejected before execution, rejected by a runtime guard, or
executed with the list of mutations it performed. -/
inductive Outcome where
  | rejected
  | ran (mutations : Nat)
  deriving DecidableEq

/-- Execution on the read path (`rt.write = false`): an `update` op only buffers into
`Pending`, which is applied only by `Commit`; a write procedure fails its guard. -/
def runRead (p : Plan) : Outcome :=
  if p.any (· == .call true) then .rejected
  else .ran (p.countP (· == .commit))

def roQuery (q : List Clause) : Outcome :=
  if isWritePlan (plan q false) then .rejected else runRead (plan q false)

-- Mutation the *plan* can see (everything but write procedures).
mutual
def mutatesP : List Clause → Bool
  | [] => false
  | c :: cs => mutatesPC c || mutatesP cs
def mutatesPC : Clause → Bool
  | .update | .createIndex | .dropIndex | .ftDrop => true
  | .sub q => mutatesP q
  | _ => false
end

-- Write procedures anywhere in the clause list (including inside `CALL {}`).
mutual
def hasWriteProc : List Clause → Bool
  | [] => false
  | c :: cs => hasWriteProcC c || hasWriteProc cs
def hasWriteProcC : Clause → Bool
  | .call true => true
  | .sub q => hasWriteProc q
  | _ => false
end

theorem isWritePlan_append (p q : Plan) :
    isWritePlan (p ++ q) = (isWritePlan p || isWritePlan q) := by
  simp [isWritePlan, List.any_append]

theorem isWritePlan_cons (o : Op) (p : Plan) :
    isWritePlan (o :: p) = (isW o || isWritePlan p) := by
  simp [isWritePlan]

theorem isWritePlan_true : ∀ q : List Clause, isWritePlan (plan q true) = true
  | [] => by simp [plan, isWritePlan, isW]
  | c :: cs => by
    cases c <;> simp only [plan, isWritePlan_append, isWritePlan_cons] <;>
      first | (simp [isW, isWritePlan_true cs]; done) | (simp [isW, isWritePlan])

mutual
/-- **Planner soundness**: a clause list with a plan-visible mutation always plans to a
plan the `execute_query` scan classifies as a write, whatever flag it starts with. -/
theorem plan_write : ∀ (q : List Clause) (w : Bool), mutatesP q = true →
    isWritePlan (plan q w) = true
  | [], _, h => by simp [mutatesP] at h
  | c :: cs, w, h => by
    simp only [mutatesP, Bool.or_eq_true] at h
    cases c with
    | update => simp only [plan, isWritePlan_cons, isWritePlan_true]; simp [isW]
    | createIndex => simp [plan, isWritePlan, isW]
    | dropIndex => simp [plan, isWritePlan, isW]
    | ftDrop => simp [plan, isWritePlan, isW]
    | sub q =>
      simp only [plan, isWritePlan_cons, isWritePlan_append]
      rcases h with h | h
      · simp [planSub, plan_write q false (by simpa [mutatesPC] using h)]
      · simp [plan_write cs w h]
    | read =>
      simp only [plan, isWritePlan_cons]
      simp [plan_write cs w (by simpa [mutatesPC] using h)]
    | call p =>
      simp only [plan, isWritePlan_cons]
      simp [plan_write cs w (by simpa [mutatesPC] using h)]
    | with_ =>
      simp only [plan, isWritePlan_append]
      simp [plan_write cs false (by simpa [mutatesPC] using h)]
    | ret =>
      simp only [plan, isWritePlan_append]
      simp [plan_write cs false (by simpa [mutatesPC] using h)]
end

mutual
/-- Every write procedure in the query survives into the plan as a `call true` op. -/
theorem plan_proc : ∀ (q : List Clause) (w : Bool), hasWriteProc q = true →
    (plan q w).any (· == .call true) = true
  | [], _, h => by simp [hasWriteProc] at h
  | c :: cs, w, h => by
    simp only [hasWriteProc, Bool.or_eq_true] at h
    cases c with
    | call p =>
      cases p
      · simp only [plan, List.any_cons]; simp [plan_proc cs w (by simpa [hasWriteProcC] using h)]
      · simp [plan]
    | sub q =>
      simp only [plan, List.any_cons, List.any_append]
      rcases h with h | h
      · simp [planSub, plan_proc q false (by simpa [hasWriteProcC] using h)]
      · simp [plan_proc cs w h]
    | with_ =>
      simp only [plan, List.any_append]; simp [plan_proc cs false (by simpa [hasWriteProcC] using h)]
    | ret =>
      simp only [plan, List.any_append]; simp [plan_proc cs false (by simpa [hasWriteProcC] using h)]
    | read => simp only [plan, List.any_cons]; simp [plan_proc cs w (by simpa [hasWriteProcC] using h)]
    | update => simp only [plan, List.any_cons]; simp [plan_proc cs true (by simpa [hasWriteProcC] using h)]
    | createIndex => simp only [plan, List.any_cons]; simp [plan_proc cs w (by simpa [hasWriteProcC] using h)]
    | dropIndex => simp only [plan, List.any_cons]; simp [plan_proc cs w (by simpa [hasWriteProcC] using h)]
    | ftDrop => simp only [plan, List.any_cons]; simp [plan_proc cs w (by simpa [hasWriteProcC] using h)]
end

theorem countP_commit_zero (p : Plan) (h : isWritePlan p = false) :
    p.countP (· == .commit) = 0 := by
  induction p with
  | nil => rfl
  | cons o os ih =>
    rw [isWritePlan_cons] at h
    simp only [Bool.or_eq_false_iff] at h
    cases o <;> simp_all [isW]

/-- **RO_QUERY never admits a write.** Either it is rejected, or it runs, performs no
mutation, and the query contained neither a plan-visible mutation nor a write
procedure. -/
theorem ro_never_writes (q : List Clause) (m : Nat) (h : roQuery q = .ran m) :
    m = 0 ∧ mutatesP q = false ∧ hasWriteProc q = false := by
  unfold roQuery at h
  split at h
  · simp at h
  · rename_i hw
    simp only [Bool.not_eq_true] at hw
    unfold runRead at h
    split at h
    · simp at h
    · rename_i hp
      simp only [Outcome.ran.injEq] at h
      refine ⟨h ▸ countP_commit_zero _ hw, ?_, ?_⟩
      · cases hm : mutatesP q
        · rfl
        · have := plan_write q false hm; rw [hw] at this; simp at this
      · cases hm : hasWriteProc q
        · rfl
        · exact absurd (plan_proc q false hm) hp

/-- The runtime guard is load-bearing: a query whose only mutation is a write procedure
has a *read* plan, and only `eval.rs:936/956` stops it. -/
theorem write_proc_invisible_to_scan : isWritePlan (plan [.call true] false) = false := by
  simp [plan, isWritePlan, isW]

/-- `CALL { CREATE () } RETURN 1` is refused before execution (C agrees, checked live). -/
theorem subquery_write_rejected : roQuery [.sub [.update], .ret] = .rejected := by
  simp [roQuery, plan, planSub, isWritePlan, isW]

end RedisLayer.Ro
