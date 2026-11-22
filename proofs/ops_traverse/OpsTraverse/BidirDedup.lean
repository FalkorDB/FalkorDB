import OpsTraverse.Basic
/-
# The bidirectional (source, destination) dedup of chained anonymous undirected hops

| here | there |
| --- | --- |
| `dedupRows`    | the `bidir_dedup` loop of `CondTraverseOp::expand_row` (`cond_traverse.rs`): key `(src_key, out.to)`, `swap_remove` on a repeat |
| `dedupStream`  | the `bidir_dedup` set kept across child batches (#3071) |
| `clearDedup`   | `CondTraverseOp::clear_bidir_dedup` (called from `set_argument_batch`, a new argument batch) |
| key source     | `dedup_source_alias` = the *child* CT's `from` alias (`cond_traverse.rs`) |

A result row is `(outer, key source, destination)`; `outer` stands for every
other column of the row (an `UNWIND` variable, an earlier MATCH's node, or the
real scan source when the key source is an intermediate node). The intended
semantics (C FalkorDB's algebraic expression) is: one row per distinct *row*
`(outer, source, destination)`. The model shows the two ways Rust's key
still diverges from it (a third, the per-batch reset, is fixed by #3071: `dedupStream_eq`). (`swap_remove` reorders the survivors; order is not modelled,
only membership and multiplicity.)
-/

namespace OpsTraverse.BidirDedup

abbrev Row := Nat × Nat × Nat   -- (outer, keySource, dest)

def key (r : Row) : Nat × Nat := (r.2.1, r.2.2)

def dedupRows : List (Nat × Nat) → List Row → List Row
  | _, [] => []
  | seen, r :: rs =>
    if key r ∈ seen then dedupRows seen rs else r :: dedupRows (key r :: seen) rs

/-- The `seen` set after deduplicating `rs` starting from `seen`. -/
def seenAfter : List (Nat × Nat) → List Row → List (Nat × Nat)
  | seen, [] => seen
  | seen, r :: rs => if key r ∈ seen then seenAfter seen rs else seenAfter (key r :: seen) rs

/-- `CondTraverseOp::next` over child batches: one `seen` set threaded through all of them. -/
def dedupStream : List (Nat × Nat) → List (List Row) → List Row
  | _, [] => []
  | seen, b :: bs => dedupRows seen b ++ dedupStream (seenAfter seen b) bs

/-- `clear_bidir_dedup`: forget every pair (a new argument batch). -/
def clearDedup (_seen : List (Nat × Nat)) : List (Nat × Nat) := []

theorem dedupRows_append : ∀ (seen : List (Nat × Nat)) (a b : List Row),
    dedupRows seen (a ++ b) = dedupRows seen a ++ dedupRows (seenAfter seen a) b
  | _, [], _ => rfl
  | seen, r :: a, b => by
    simp only [List.cons_append, dedupRows, seenAfter]
    split <;> simp [dedupRows_append]

theorem dedupRows_sub : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ dedupRows seen rs → r ∈ rs
  | _, [], _, h => by simp [dedupRows] at h
  | seen, x :: rs, r, h => by
    unfold dedupRows at h
    split at h
    · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)

theorem dedupRows_key_fresh : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ dedupRows seen rs → key r ∉ seen
  | _, [], _, h => by simp [dedupRows] at h
  | seen, x :: rs, r, h => by
    unfold dedupRows at h
    split at h
    · exact dedupRows_key_fresh _ _ _ h
    · rename_i hx
      rcases List.mem_cons.1 h with rfl | h
      · exact hx
      · have := dedupRows_key_fresh _ _ _ h
        exact fun hm => this (List.mem_cons_of_mem _ hm)

/-- Within one batch the surviving keys are pairwise distinct. -/
theorem dedupRows_keys_nodup : ∀ (seen : List (Nat × Nat)) (rs : List Row),
    ((dedupRows seen rs).map key).Nodup
  | _, [] => by simp [dedupRows]
  | seen, x :: rs => by
    unfold dedupRows
    split
    · exact dedupRows_keys_nodup _ _
    · simp only [List.map_cons, List.nodup_cons, List.mem_map]
      refine ⟨?_, dedupRows_keys_nodup _ _⟩
      rintro ⟨r, hr, hk⟩
      exact dedupRows_key_fresh _ _ _ hr (hk ▸ List.mem_cons_self ..)

/-- A key-complete dedup never loses a row: every input key is still present. -/
theorem dedupRows_keys_complete : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ rs → key r ∉ seen → ∃ r' ∈ dedupRows seen rs, key r' = key r
  | _, [], _, h, _ => by simp at h
  | seen, x :: rs, r, h, hs => by
    unfold dedupRows
    split
    · rename_i hx
      rcases List.mem_cons.1 h with rfl | h
      · exact absurd hx hs
      · exact dedupRows_keys_complete _ _ _ h hs
    · by_cases hk : key r = key x
      · exact ⟨x, List.mem_cons_self .., hk.symm⟩
      · rcases List.mem_cons.1 h with rfl | h
        · exact absurd rfl hk
        · have hs' : key r ∉ key x :: seen := by
            simp only [List.mem_cons, not_or]; exact ⟨hk, hs⟩
          obtain ⟨r', hr', he⟩ := dedupRows_keys_complete _ _ _ h hs'
          exact ⟨r', List.mem_cons_of_mem _ hr', he⟩

/-! ## Counterexamples (each reproduced against the engine)

The `lean_ops_traverse::bug_*` Rust tests cited below are not in this PR: as for every area
(see the FINDINGS.md header) they were run locally against the real engine and are not
committed, because many `bug_*` tests fail on purpose. The Rust and C counts quoted are from
those runs; the tests land with their fixes. -/

/-- (a) Rows that differ only in an outer column collapse:
`UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)` loses every `x = 2` row.
`lean_ops_traverse::bug_bidir_dedup_drops_rows_differing_in_outer_column`
(Rust 5 rows, C 10). -/
theorem drops_outer_rows :
    dedupRows [] [(1, 1, 3), (2, 1, 3)] = [(1, 1, 3)] := by decide

/-- (b) With three hops the key source is the intermediate node, so start
nodes 1 and 4 reaching destination 3 through the same intermediate 2 collide.
`lean_ops_traverse::bug_bidir_dedup_three_hops_keys_on_intermediate` (Rust 7 rows, C 12). -/
theorem three_hops_collide :
    -- rows are (start, intermediate, dest)
    dedupRows [] [(1, 2, 3), (4, 2, 3)] = [(1, 2, 3)] := by decide

/-! ## #3071: the dedup no longer depends on batching

Part (c) of W2-traverse-2 (the set was cleared on every child batch, so the result
depended on where the child's output was cut into batches) is fixed: the set lives until
`clear_bidir_dedup` (a new argument batch). -/

/-- **PROVEN**: deduplicating batch by batch with one threaded set is deduplicating the
concatenated child output. -/
theorem dedupStream_eq : ∀ (seen : List (Nat × Nat)) (bs : List (List Row)),
    dedupStream seen bs = dedupRows seen bs.flatten
  | _, [] => by simp [dedupStream, dedupRows]
  | seen, b :: bs => by
    simp only [dedupStream, List.flatten_cons, dedupRows_append, dedupStream_eq]

/-- **PROVEN**: so any two batchings of the same child rows give the same result. -/
theorem dedupStream_batch_independent (bs cs : List (List Row)) (h : bs.flatten = cs.flatten) :
    dedupStream (clearDedup []) bs = dedupStream (clearDedup []) cs := by
  rw [dedupStream_eq, dedupStream_eq, h]

/-- The former counterexample input (c): both batchings now give one row. -/
theorem batch_split_same_result :
    dedupStream [] [[(0, 1, 1), (0, 1, 1)]] = dedupStream [] [[(0, 1, 1)], [(0, 1, 1)]] ∧
    dedupStream [] [[(0, 1, 1)], [(0, 1, 1)]] = [(0, 1, 1)] := by
  decide

/-- **PROVEN**: across child batches the surviving keys stay pairwise distinct. -/
theorem dedupStream_keys_nodup (seen : List (Nat × Nat)) (bs : List (List Row)) :
    ((dedupStream seen bs).map key).Nodup := by
  rw [dedupStream_eq]; exact dedupRows_keys_nodup _ _

end OpsTraverse.BidirDedup
