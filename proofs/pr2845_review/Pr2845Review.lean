/-
# Review of PR #2845 ("give every CALL {} body its entry projection as a record boundary")

Model of the merged branch, commit 39560cbdf (PR head ab1f46382 merged with
origin/main 89d68334a). **#2845 has since landed on main as eb470521b**: the
modelled functions are byte-identical on origin/main a9377c636 (only a doc
comment moved in batch.rs, putting `has_origins` at :1312), so this project now
describes origin/main and its COVERAGE rows are live. Below, "origin/main"
means main *before* #2845 (da6f808c3); its counterexamples are kept as
historical notes named `pre2845_…` (fixed by #2845, eb470521b; closes
#2601, #2602). Built with `lake build`; no `sorry`,
`admit`, `axiom` or `native_decide`. 36 theorems.

| here | there (39560cbdf) |
| --- | --- |
| `Env`, `freshVar`, `mint_fresh_iff` | copied from proofs/binder (binder.rs:2243) |
| `projectName`, `importProj` | `project_name` :2165, `build_import_projections` :979 |
| `bodyNew` / `bodyOld` | Query arm of `bind_call_body` :845-871 / origin/main |
| `branchNew`, `branchScope` | UNION arm :899-925 (`env_stack: vec![HashMap::new()]`; #3027: `len()+1`) |
| `relabel` | `update_graph_labels` :160-190 via `bind_call_subquery` step 2b :731 |
| `projRow`, `bodyInputNew/Old` | `ProjectOp::eval` row build (project.rs:108-129); `Argument` |
| `inherited`, `pushable` | push_filters_down.rs Case 2 :207-249, augmentation :252-262 |
| `Batch`, `rowsOnly`, `hasOrigins`, `gather` | batch.rs :853, :1312, :897 |
| `projectEmptyFast/General`, `projectMerged/Main` | project.rs:98-105, :54-147 |
| `shouldExpandNew/Old`, `finishBatch` | batched_result_emitter.rs `start_batch` :602, `finish_batch` :648 |
| `concatColless` | `Batch::concat` :979 on column-less batches |

## Claims checked
1. Every CALL body gets an entry projection: PROVEN (`bodyNew_entry`,
   `bodyNew_empty`, `branchNew_entry`); unchanged when something is imported
   (`bodyNew_eq_old`); origin/main omitted it (`pre2845_bodyOld_empty`).
2. No slot aliasing between body and outer scope:
   * plain bodies, runtime rows: PROVEN — every body-minted slot is unbound at
     the body's entry for every outer row, imports carry the outer value
     (`body_own_unbound`, `body_import_value`, `new_scan_scans`; origin/main
     counterexample `pre2845_old_aliases`).
   * UNION bodies: REFUTED — the aliasing there is on binder keys
     `(scope_id, id)`, which the runtime boundary does not touch: a branch's
     `q` has key `(0,0)` like the outer `n` and inherits its labels
     (`union_branch_label_alias`); #3027's numbering removes it
     (`relabel_3027`, `union_branch_label_3027`). Live: #3025's 4 queries still
     wrong on the merged build; correct on merged+#3027.
   * optimizer: REFUTED — `push_filters_down` Case 2 inherits the OUTER left
     branch's ids into an `Optional`'s right branch inside the body
     (`filter_pushed_through_boundary`), where the row is the entry projection's
     (`pushed_read_unbound`). Live: `MATCH (x:A:B) CALL { OPTIONAL MATCH
     (n:A:B {id:3}) RETURN n.id AS nid } RETURN x.id, nid` → Rust "Variable n not
     found" (main and merged), C `[[1,3],[2,3],[3,3]]`. Fix modelled
     (`fixed_keeps_filter`, `fixed_conservative`, `fixed_optional_inherits_left`)
     and validated live.
3. Fresh-id minting: PROVEN — entry ids are `0..k-1` in the body scope, the
   table covers them, the next mint is `k` (`importProj_spec`,
   `entry_ids_below`), body scope disjoint from live outer scopes
   (`body_scope_disjoint`).
4. Batch/emitter per-row semantics: PROVEN — the emitter keeps each result's
   origin for every parent shape (`emit_origin_sound`; origin/main lost it,
   `pre2845_emit_origin_lost_old`), row count = results (`emit_len`), residual 0-row
   case unchanged and unreachable (`emit_len_residual`); concat of column-less
   batches = per-row builder (`concatColless_perRow`).
5. Project semantics: PROVEN — the fast path equals the general path exactly
   (`projectEmpty_fast_eq_general`), so the merged operator equals
   origin/main's on every input (`projectMerged_eq_main`) and `eval_sound`
   carries over.

## Gaps
Values and expression evaluation are abstract (proofs/columnar covers them);
Apply's merge of body output into the outer row is not modelled (relies on
the pre-existing remap Project); the optimizer model is the ancestor walk
only, not the full tree rewrite.
-/
import Pr2845Review.Binder
import Pr2845Review.Row
import Pr2845Review.Optimizer
import Pr2845Review.Batch
