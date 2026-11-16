import OpsTraverse.Basic
/-
# The bidirectional (source, destination) dedup of chained anonymous undirected hops

| here | there |
| --- | --- |
| `dedupRows`    | the `bidir_dedup` loop of `CondTraverseOp::expand_row` (`cond_traverse.rs:1117-1137`): key `(src_key, out.to)`, `swap_remove` on a repeat |
| `dedupStream`  | since #3071: one `seen` set over every child batch of an argument scope (`CondTraverseOp::next`, `cond_traverse.rs:1408`; no clear on a child batch, `:1496-1500`); cleared by `clear_bidir_dedup` (`:374-378`) from `set_argument_batch` (`batch.rs:1643`) |
| `dedupArgs`    | one `dedupStream` per argument batch (Apply/Optional re-seeding) |
| `dedupBatches` | before #3071: `seen.clear()` on every new child batch (343e077f2 `cond_traverse.rs:1476-1480`) |
| key source     | `dedup_source_alias` = the *child* CT's `from` alias (`cond_traverse.rs:418-437`) |

A result row is `(outer, key source, destination)`; `outer` stands for every
other column of the row (an `UNWIND` variable, an earlier MATCH's node, or the
real scan source when the key source is an intermediate node). The intended
semantics (C FalkorDB's algebraic expression) is: one row per distinct *row*
`(outer, source, destination)`. The model shows the three ways Rust's key
diverges from it. (`swap_remove` reorders the survivors; order is not modelled,
only membership and multiplicity.)
-/

namespace OpsTraverse.BidirDedup

abbrev Row := Nat × Nat × Nat   -- (outer, keySource, dest)

def key (r : Row) : Nat × Nat := (r.2.1, r.2.2)

def dedupRows : List (Nat × Nat) → List Row → List Row
  | _, [] => []
  | seen, r :: rs =>
    if key r ∈ seen then dedupRows seen rs else r :: dedupRows (key r :: seen) rs

/-- Before #3071: `seen` cleared per child batch. -/
def dedupBatches (bs : List (List Row)) : List Row := bs.flatMap (dedupRows [])

/-- Since #3071: one `seen` set across the child batches of an argument scope. -/
def dedupStream (bs : List (List Row)) : List Row := dedupRows [] bs.flatten

/-- Since #3071: `set_argument_batch` clears `seen` (`clear_bidir_dedup`) for each
argument batch; each argument batch's child batches share one set. -/
def dedupArgs (args : List (List (List Row))) : List Row := args.flatMap dedupStream

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

/-- **PROVEN** (#3071): the dedup no longer depends on how the child's output is cut into
batches — only the concatenated row stream matters. -/
theorem dedupStream_batch_invariant (bs bs' : List (List Row)) (h : bs.flatten = bs'.flatten) :
    dedupStream bs = dedupStream bs' := by simp [dedupStream, h]

/-- **PROVEN** (#3071): within an argument scope the surviving keys are pairwise distinct
across all child batches (one row per `(source, destination)` pair). -/
theorem dedupStream_keys_nodup (bs : List (List Row)) : ((dedupStream bs).map key).Nodup :=
  dedupRows_keys_nodup _ _

/-- **PROVEN** (#3071): every input key still has a row in the output. -/
theorem dedupStream_keys_complete (bs : List (List Row)) (r : Row) (h : r ∈ bs.flatten) :
    ∃ r' ∈ dedupStream bs, key r' = key r :=
  dedupRows_keys_complete [] _ r h (by simp)

/-- **PROVEN** (#3071, `clear_bidir_dedup` from `set_argument_batch`): each argument batch
starts from an empty `seen` set, so rows of one argument scope never suppress rows of the
next. -/
theorem dedupArgs_cons (a : List (List Row)) (rest : List (List (List Row))) :
    dedupArgs (a :: rest) = dedupStream a ++ dedupArgs rest := by simp [dedupArgs]

theorem dedupArgs_scope_fresh :
    dedupArgs [[[(0, 1, 1)]], [[(0, 1, 1)]]] = [(0, 1, 1), (0, 1, 1)] := by decide

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

/-- (c) Before #3071 (fixed by #3071, see `batch_split_fixed`): clearing `seen` per input
batch made the answer depend on where the child's output was cut into `BATCH_SIZE` batches.
`lean_ops_traverse::bug_bidir_dedup_depends_on_batch_size` (Rust count 3/4, C 1). -/
theorem batch_split_changes_result :
    dedupBatches [[(0, 1, 1), (0, 1, 1)]] ≠ dedupBatches [[(0, 1, 1)], [(0, 1, 1)]] := by
  decide

/-- (c) fixed by #3071: the same two splits give the same rows. -/
theorem batch_split_fixed :
    dedupStream [[(0, 1, 1), (0, 1, 1)]] = dedupStream [[(0, 1, 1)], [(0, 1, 1)]] := by decide

end OpsTraverse.BidirDedup
