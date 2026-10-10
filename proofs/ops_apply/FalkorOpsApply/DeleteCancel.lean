import FalkorOpsApply.Delete
/-
DELETE after #2846: the unwinds can fail.

`delete_pending_node(id, g)?` (delete.rs:408-412) and
`remove_pending_relationships_for_node(id, g)?` (:252-258, :447-453) now hand the
unwound ids back to the graph's id space themselves and propagate its refusal
(`Graph::cancel_*_id`, graph.rs:1401/:1414) with `?`. The two hooks are:

* `okP id` — the graph accepts the unwind of pending-created node `id`
  (its id and its pending edges' ids);
* `okC id` — the graph accepts handing back the pending edges incident on the
  committed node `id` (vacuously `true` when there are none — the fast path
  at :262-267 never calls it).

The `E` functions below are the faithful, fallible versions. Each is proved
equal to `.ok` of the infallible model in Delete.lean when the hooks accept
(`*_E_ok`), so every theorem there (`delNodesBulk_spec`, `deleteBatch_vars`,
…) holds verbatim for a DELETE whose cancels succeed; and a refusal surfaces
as the operator's error (`delNodeE_refusedP`, `delNodeE_refusedC`).
-/
namespace Delete

variable (okP okC : Nat → Bool)

def cancelErr : String := "internal error in the id space"

/-- `delete_entity`, node arm (delete.rs:399-465). -/
def delNodeE (st : St) (id : Nat) : Except String St :=
  if st.pd id then .ok st
  else if st.pc id then
    if okP id then .ok { st with pc := clrS st.pc id, gd := setS st.gd id } else .error cancelErr
  else if !st.gd id then
    if okC id then .ok { st with pd := setS st.pd id } else .error cancelErr
  else .ok st

theorem delNodeE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) (st : St) (id : Nat) :
    delNodeE okP okC st id = .ok (delNode st id) := by
  unfold delNodeE delNode; simp [hP, hC]; split <;> (try split) <;> (try split) <;> rfl

theorem delNodeE_refusedP (st : St) (id : Nat) (h0 : st.pd id = false) (h1 : st.pc id = true)
    (h : okP id = false) : delNodeE okP okC st id = .error cancelErr := by
  simp [delNodeE, h0, h1, h]

theorem delNodeE_refusedC (st : St) (id : Nat) (h0 : st.pd id = false) (h1 : st.pc id = false)
    (h2 : st.gd id = false) (h : okC id = false) : delNodeE okP okC st id = .error cancelErr := by
  simp [delNodeE, h0, h1, h2, h]

/-- A successful node delete leaves the node gone. -/
theorem delNodeE_gone (st st' : St) (id : Nat) (h : delNodeE okP okC st id = .ok st') :
    goneN st' id = true := by
  unfold delNodeE at h
  by_cases h1 : st.pd id
  · simp [h1] at h; subst h; simp [goneN, h1]
  · by_cases h2 : st.pc id
    · by_cases hp : okP id
      · simp [h1, h2, hp] at h; subst h; simp [goneN, setS, clrS]
      · simp [h1, h2, hp] at h
    · by_cases h3 : st.gd id
      · simp [h1, h2, h3] at h; subst h; simp [goneN, h3, h2]
      · by_cases hc : okC id
        · simp [h1, h2, h3, hc] at h; subst h; simp [goneN, setS]
        · simp [h1, h2, h3, hc] at h

mutual
def delEntityE : DV → St → Except String St
  | .node id, st => delNodeE okP okC st id
  | .rel id, st => .ok (delRel st id)
  | .path vs, st => delEntityLE vs st
  | .null, st => .ok st
  | .other, _ => .error mismatch
def delEntityLE : List DV → St → Except String St
  | [], st => .ok st
  | v :: vs, st => do let st' ← delEntityE v st; delEntityLE vs st'
end

mutual
theorem delEntityE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) :
    ∀ (v : DV) (st : St), delEntityE okP okC v st = delEntity v st
  | .node id, st => by simp [delEntityE, delEntity, delNodeE_ok okP okC hP hC]
  | .rel _, _ => rfl
  | .path vs, st => by simp only [delEntityE, delEntity]; exact delEntityLE_ok hP hC vs st
  | .null, _ => rfl
  | .other, _ => rfl
theorem delEntityLE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) :
    ∀ (vs : List DV) (st : St), delEntityLE okP okC vs st = delEntityL vs st
  | [], _ => rfl
  | v :: vs, st => by
    simp only [delEntityLE, delEntityL, delEntityE_ok hP hC v st, bind]
    cases delEntity v st with
    | error e => rfl
    | ok st' => exact delEntityLE_ok hP hC vs st'
end

/-- `delete_nodes_bulk` classification loop (delete.rs:197-207); the
pending-created branch is `delete_entity` and propagates its error. -/
def bulkScanE : List Nat → St × List Nat → Except String (St × List Nat)
  | [], acc => .ok acc
  | id :: ids, (st, com) =>
    if st.pd id then bulkScanE ids (st, com)
    else if st.pc id then (delNodeE okP okC st id).bind fun st' => bulkScanE ids (st', com)
    else if !st.gd id then bulkScanE ids (st, com ++ [id])
    else bulkScanE ids (st, com)

theorem bulkScanE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) :
    ∀ (ids : List Nat) (acc : St × List Nat), bulkScanE okP okC ids acc = .ok (bulkScan ids acc)
  | [], _ => rfl
  | id :: ids, (st, com) => by
    simp only [bulkScanE, bulkScan, delNodeE_ok okP okC hP hC, Except.bind]
    split
    · exact bulkScanE_ok hP hC ids _
    · split
      · exact bulkScanE_ok hP hC ids _
      · split <;> exact bulkScanE_ok hP hC ids _

/-- delete.rs:243-268: per committed id, the pending-edge cascade (`okC`, `?`),
then `deleted_node(id)`. -/
def markCommitted : List Nat → St → Except String St
  | [], st => .ok st
  | id :: ids, st => if okC id then markCommitted ids { st with pd := setS st.pd id } else .error cancelErr

theorem markCommitted_ok (hC : ∀ i, okC i = true) :
    ∀ (com : List Nat) (st : St), markCommitted okC com st = .ok { st with pd := fun j => com.contains j || st.pd j }
  | [], st => by simp [markCommitted]
  | id :: ids, st => by
    simp only [markCommitted, hC, ite_true, markCommitted_ok hC ids]
    congr 2; funext j; simp only [setS, List.contains_cons]
    by_cases hj : j = id
    · subst hj; simp
    · have : (j == id) = false := by simp [hj]
      simp [hj, this]

/-- `delete_nodes_bulk` (delete.rs:191). -/
def delNodesBulkE (st : St) (ids : List Nat) : Except String St :=
  (bulkScanE okP okC ids (st, [])).bind fun (st1, com) =>
    if com.isEmpty then .ok st1 else markCommitted okC com st1

theorem delNodesBulkE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) (st : St) (ids : List Nat) :
    delNodesBulkE okP okC st ids = .ok (delNodesBulk st ids) := by
  unfold delNodesBulkE delNodesBulk
  rw [bulkScanE_ok okP okC hP hC]
  simp only [Except.bind]
  generalize bulkScan ids (st, []) = r
  obtain ⟨st1, com⟩ := r
  simp only
  split
  · rename_i he
    have : com = [] := List.isEmpty_iff.1 he
    subst this; simp
  · exact markCommitted_ok okC hC com st1

/-- The variable-tree scan (delete.rs:144-157) over the fallible `delete_entity`. -/
def scanE (st : St) : List DV → Except String (St × List Nat × List Nat)
  | [] => .ok (st, [], [])
  | v :: vs => match v with
    | .node id => (scanE st vs).map fun r => (r.1, id :: r.2.1, r.2.2)
    | .rel id => (scanE st vs).map fun r => (r.1, r.2.1, id :: r.2.2)
    | w => do let st' ← delEntityE okP okC w st; scanE st' vs

theorem scanE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) :
    ∀ (st : St) (vs : List DV), scanE okP okC st vs = scan st vs
  | st, [] => rfl
  | st, v :: vs => by
    cases v with
    | node id => simp only [scanE, scan, scanE_ok hP hC st vs]
    | rel id => simp only [scanE, scan, scanE_ok hP hC st vs]
    | path ps =>
      simp only [scanE, scan, delEntityE_ok okP okC hP hC, bind]
      cases delEntity (.path ps) st with
      | error e => rfl
      | ok st' => exact scanE_ok hP hC st' vs
    | null => simp only [scanE, scan, delEntityE_ok okP okC hP hC, bind]; exact scanE_ok hP hC st vs
    | other => rfl

/-- `delete_batch` (delete.rs:127) over the fallible pieces. -/
def deleteBatchE (st : St) (vals : List DV) (exprs : List (Except String DV)) : Except String St := do
  let (st1, ns, rs) ← scanE okP okC st vals
  let st2 ← if ns.isEmpty then .ok st1 else delNodesBulkE okP okC st1 ns
  let st3 := if rs.isEmpty then st2 else delRelsBulk st2 rs
  exprs.foldlM (fun s e => do let v ← e; delEntityE okP okC v s) st3

/-- **`deleteBatchE_ok`**: when every cancel is accepted the fallible DELETE is
exactly the model `deleteBatch`, so `deleteBatch_vars` and the rest apply. -/
theorem deleteBatchE_ok (hP : ∀ i, okP i = true) (hC : ∀ i, okC i = true) (st : St) (vals : List DV)
    (exprs : List (Except String DV)) : deleteBatchE okP okC st vals exprs = deleteBatch st vals exprs := by
  unfold deleteBatchE deleteBatch
  rw [scanE_ok okP okC hP hC]
  simp only [bind, Except.bind]
  cases scan st vals with
  | error e => rfl
  | ok r =>
    obtain ⟨st1, ns, rs⟩ := r
    simp only
    have hf : (fun s (e : Except String DV) => (do let v ← e; delEntityE okP okC v s : Except String St)) =
        (fun s e => do let v ← e; delEntity v s) := by
      funext s e; cases e with
      | error _ => rfl
      | ok v => simp [bind, Except.bind, delEntityE_ok okP okC hP hC]
    by_cases hn : ns.isEmpty
    · simp only [hn, ite_true]; simp only [bind, Except.bind] at hf ⊢; rw [hf]
    · simp only [hn, Bool.false_eq_true, ite_false, delNodesBulkE_ok okP okC hP hC]
      simp only [bind, Except.bind] at hf ⊢; rw [hf]

end Delete
