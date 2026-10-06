/-
# Staged properties: `set_node_attribute`, reads through `Pending`, commit

`graph/src/runtime/pending.rs` and `graph/src/runtime/runtime.rs`.

* `upsert` — `Pending::set_node_attribute` (pending.rs:440):
  `binary_search_by_key`, overwrite on `Ok`, `insert` on `Err`.
* `readQ` — `Runtime::get_node_attribute_no_delete_check` (runtime.rs:1479):
  the pending value first (`Some(Null)` for a staged removal), then the store.
  The evaluator reads `Some(Null)` and `None` both as `null`, so `normQ` folds
  them.
* `commitExisting` — `Graph::set_nodes_attributes` → `insert_attrs` →
  `merge_span` (pending.rs:1159, graph.rs:1674).
* `commitNew` — `import_node_attrs` → `import_attrs` (pending.rs:1151,
  attribute_store.rs:1351): drop nulls, `set_span`.

Theorems: the pending list stays sorted/unique under any sequence of writes
and reads back the last write; commit then stores exactly what the query read
(read-your-writes survives commit), for existing and for new entities.

The last section models `SET n = m` (`ops/set.rs:185-224`), whose *Node* and
*Relationship* sources — and every relationship-target `SET r = …` form — skip
`clear_node_attributes`, so attributes staged earlier in the same query
survive a replace (known as #2776; `replace_keeps_staged` is the Lean
counterexample).
-/
import PendingCommit.Span

namespace PendingCommit

/-- `set_node_attribute`'s binary-search upsert. -/
def upsert : List Entry → Nat → Val → List Entry
  | [], k, v => [(k, v)]
  | e :: es, k, v =>
      if k < e.1 then (k, v) :: e :: es
      else if k = e.1 then (k, v) :: es
      else e :: upsert es k v

theorem upsert_keys_gt (l : List Entry) (k : Nat) (v : Val) (b : Nat)
    (hl : ∀ a ∈ l, b < a.1) (hk : b < k) : ∀ a ∈ upsert l k v, b < a.1 := by
  induction l with
  | nil => simp [upsert]; exact hk
  | cons e es ih =>
    have he := hl e (by simp)
    have hes : ∀ a ∈ es, b < a.1 := fun a ha => hl a (by simp [ha])
    simp only [upsert]
    split
    · intro a ha; simp at ha; rcases ha with rfl | rfl | ha
      · exact hk
      · exact he
      · exact hes a ha
    · split
      · intro a ha; simp at ha; rcases ha with rfl | ha
        · exact hk
        · exact hes a ha
      · intro a ha; simp at ha; rcases ha with rfl | ha
        · exact he
        · exact ih hes a ha

theorem upsert_sorted (l : List Entry) (k : Nat) (v : Val) (h : Sorted l) : Sorted (upsert l k v) := by
  induction l with
  | nil => simp [upsert, Sorted]
  | cons e es ih =>
    simp only [upsert]
    split
    · rename_i hlt
      unfold Sorted
      refine List.pairwise_cons.2 ⟨?_, h⟩
      intro a ha; simp at ha; rcases ha with rfl | ha
      · exact hlt
      · have := sorted_head_lt h a ha; omega
    · split
      · rename_i _ heq
        unfold Sorted
        refine List.pairwise_cons.2 ⟨?_, sorted_tail h⟩
        intro a ha; have := sorted_head_lt h a ha; omega
      · rename_i h1 h2
        unfold Sorted
        refine List.pairwise_cons.2 ⟨?_, ih (sorted_tail h)⟩
        exact upsert_keys_gt es k v e.1 (sorted_head_lt h) (by omega)

theorem get_upsert (l : List Entry) (k : Nat) (v : Val) (j : Nat) :
    get (upsert l k v) j = if j = k then some v else get l j := by
  induction l with
  | nil => simp [upsert, get]
  | cons e es ih =>
    simp only [upsert]
    by_cases h1 : k < e.1
    · simp only [h1, ite_true]; by_cases hj : j = k <;> simp [get, hj]
    · simp only [h1, ite_false]
      by_cases h2 : k = e.1
      · simp only [h2, ite_true]; by_cases hj : j = e.1 <;> simp [get, hj]
      · simp only [h2, ite_false]; rw [get, ih]
        by_cases hj : j = k
        · subst hj; simp [h2]
        · by_cases hje : j = e.1
          · subst hje; simp [get, hj]
          · simp [get, hj, hje]

/-- A query's writes to one entity, in order: `SET n.k = v` (`v = null` for
`REMOVE n.k` / `SET n.k = null`). -/
def stageAll (ws : List Entry) : List Entry := ws.foldl (fun p w => upsert p w.1 w.2) []

/-- Reference: the last write to `k`, if any. -/
def lastWrite (ws : List Entry) (k : Nat) : Option Val :=
  ws.foldl (fun r w => if w.1 = k then some w.2 else r) none

theorem stageAll_aux (ws : List Entry) (p : List Entry) (hp : Sorted p) :
    Sorted (ws.foldl (fun p w => upsert p w.1 w.2) p) ∧
    ∀ k, get (ws.foldl (fun p w => upsert p w.1 w.2) p) k =
      ws.foldl (fun r w => if w.1 = k then some w.2 else r) (get p k) := by
  induction ws generalizing p with
  | nil => simp [hp]
  | cons w ws ih =>
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih (upsert p w.1 w.2) (upsert_sorted p w.1 w.2 hp)
    refine ⟨h1, fun k => ?_⟩
    rw [h2, get_upsert]
    by_cases hk : k = w.1
    · simp [hk]
    · simp [hk, Ne.symm hk]

/-- **`stage_sorted_lastWrite`**: whatever sequence of property writes a query
makes, the staged list stays strictly sorted (the precondition
`insert_attrs_rows`/`merge_span` assert) and holds exactly the last write per
attribute. -/
theorem stage_sorted_lastWrite (ws : List Entry) :
    Sorted (stageAll ws) ∧ ∀ k, get (stageAll ws) k = lastWrite ws k := by
  have := stageAll_aux ws [] (by simp [Sorted])
  exact ⟨this.1, fun k => by rw [stageAll, this.2]; rfl⟩

/-- What a later clause in the same query reads (`n.k`), with `Some(Null)`
and `None` both meaning `null` (`normQ`). -/
def readQ (stored pend : List Entry) (k : Nat) : Option Val :=
  match get pend k with
  | some v => some v
  | none => get stored k

def normQ : Option Val → Option Val
  | some .null => none
  | o => o

/-- **`read_your_writes_existing`**: for an entity that already existed, what
the query read before `Commit` is exactly what the store holds after it. -/
theorem read_your_writes_existing (stored ws : List Entry) (hs : Sorted stored) (hn : NoNull stored)
    (k : Nat) :
    get (mergeSpan stored (stageAll ws)).1 k = normQ (readQ stored (stageAll ws) k) := by
  have ⟨hsort, _⟩ := stage_sorted_lastWrite ws
  rw [(mergeSpan_correct stored _ hs hn hsort).1 k]
  unfold refGet readQ normQ
  cases get (stageAll ws) k with
  | none => cases h : get stored k with
    | none => rfl
    | some v =>
      obtain ⟨e, he, _, rfl⟩ := get_some_mem h
      have := hn e he
      cases hv : e.2 with
      | null => simp [hv, Val.isNull] at this
      | inl i => rfl
      | heap s => rfl
  | some v => cases v <;> rfl

/-- `import_attrs`: non-null pairs only, then `set_span`. -/
def commitNew (pend : List Entry) : List Entry := pend.filter (fun e => !e.2.isNull)

theorem get_filter_nonnull (l : List Entry) (hl : Sorted l) (k : Nat) :
    get (l.filter (fun e => !e.2.isNull)) k = normQ (get l k) := by
  induction l with
  | nil => rfl
  | cons e es ih =>
    have ih := ih (sorted_tail hl)
    by_cases hk : k = e.1
    · subst hk
      have hnone : get es e.1 = none := by
        apply get_none_of_all_gt; exact sorted_head_lt hl
      cases hv : e.2 with
      | null =>
        have hp : (!e.2.isNull) = false := by rw [hv]; rfl
        rw [List.filter_cons, hp]
        simp only [Bool.false_eq_true, ite_false]
        rw [ih, hnone]; simp [get, hv, normQ]
      | inl i => simp [List.filter_cons, hv, Val.isNull, get, normQ]
      | heap s => simp [List.filter_cons, hv, Val.isNull, get, normQ]
    · cases hv : e.2.isNull <;> simp [List.filter_cons, hv, get, hk, ih]

/-- **`read_your_writes_new`**: for an entity created in this query (no stored
span — a reclaimed id's attributes were cleared by `delete_nodes`'
`remove_all`), `import_attrs` stores exactly what the query read. -/
theorem read_your_writes_new (ws : List Entry) (k : Nat) :
    get (commitNew (stageAll ws)) k = normQ (readQ [] (stageAll ws) k) := by
  unfold commitNew readQ
  rw [get_filter_nonnull _ (stage_sorted_lastWrite ws).1]
  cases get (stageAll ws) k <;> rfl

/-- `import_attrs`' count is `properties_set` for created entities: the
non-null staged pairs (attribute_store.rs:1370). -/
theorem commitNew_count (pend : List Entry) : (commitNew pend).length = refSet pend := rfl

/-! ## `SET n = m`: replace without clearing staged attributes (known #2776) -/

/-- `ops/set.rs:185-199` (`Value::Map` source): clear the staged attributes,
stage `null` for every *committed* key, then stage the map's entries. -/
def replaceFromMap (stored pend src : List Entry) : List Entry :=
  let p0 : List Entry := []
  let p1 := stored.foldl (fun p e => upsert p e.1 .null) p0
  src.foldl (fun p e => upsert p e.1 e.2) p1

/-- `ops/set.rs:200-218` (`Value::Node` source): the same, minus the clear. -/
def replaceFromNode (stored pend src : List Entry) : List Entry :=
  let p1 := stored.foldl (fun p e => upsert p e.1 .null) pend
  src.foldl (fun p e => upsert p e.1 e.2) p1

/-- `CREATE (n {a:1}), (m {b:2}) SET n = m`: `a` is staged (not yet stored), so
the node-source replace keeps it. C returns `{b: 2}`. -/
theorem replace_keeps_staged :
    normQ (readQ [] (replaceFromNode [] [(0, .inl 1)] [(1, .inl 2)]) 0) = some (.inl 1) ∧
    normQ (readQ [] (replaceFromMap [] [(0, .inl 1)] [(1, .inl 2)]) 0) = none := by
  decide

end PendingCommit
