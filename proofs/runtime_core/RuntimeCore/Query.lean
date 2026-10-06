/-
# `Runtime::query`: the result-set cap and the write drain

| here | there |
| --- | --- |
| `capLoop`  | the `if self.result_set_size >= 0` loop, `runtime/runtime.rs:520-546` |
| `query`    | `Runtime::query`, `runtime/runtime.rs:514-561` (rows + number of child batches pulled) |

A batch is modelled by its active rows (`active_indices`); `set_selection(take keep)`
keeps the first `keep` of them. The second component counts how many batches were pulled
from the root operator — pulling is what runs `CommitOp`/pending side effects.
-/
namespace RuntimeCore.Query

variable {α : Type}

/-- The capped loop. `total` is the running `total`; returns (result batches, #pulled). -/
def capLoop (limit : Nat) (write : Bool) : Nat → List (List α) → List (List α) × Nat
  | _, [] => ([], 0)
  | total, b :: bs =>
    let total' := total + b.length
    if total' ≥ limit then
      -- `keep = batch.active_len() - (total - limit)`, then drain the rest if `write`
      ([b.take (b.length - (total' - limit))], 1 + if write then bs.length else 0)
    else
      let r := capLoop limit write total' bs
      (b :: r.1, r.2 + 1)

/-- `Runtime::query` (`runtime.rs:514-561`). `rss` is `result_set_size`. -/
def query (rss : Int) (write : Bool) (bs : List (List α)) : List (List α) × Nat :=
  if rss ≥ 0 then capLoop rss.toNat write 0 bs else (bs, bs.length)

/-- **PROVEN** (no `usize` underflow): when the cap is hit, `total - limit ≤ active_len`,
so `batch.active_len() - (total - limit)` is exact and equals `limit - total_before`. -/
theorem keep_exact (limit total len : Nat) (h0 : total ≤ limit) (h1 : total + len ≥ limit) :
    total + len - limit ≤ len ∧ len - (total + len - limit) = limit - total := by omega

theorem capLoop_flatten (limit : Nat) (write : Bool) :
    ∀ (total : Nat) (bs : List (List α)), total ≤ limit →
      (capLoop limit write total bs).1.flatten = bs.flatten.take (limit - total)
  | _, [], _ => by simp [capLoop]
  | total, b :: bs, h => by
    simp only [capLoop]
    split
    · next hge =>
      simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
      rw [(keep_exact limit total b.length h hge).2, List.take_append_of_le_length (by omega)]
    · next hlt =>
      simp only [List.flatten_cons]
      have e : limit - total = b.length + (limit - (total + b.length)) := by omega
      rw [capLoop_flatten limit write (total + b.length) bs (by omega), e]
      rw [List.take_length_add_append]

/-- **PROVEN** (headline): with `result_set_size = n ≥ 0` the rows returned are exactly
the first `n` rows the plan produces; with a negative size, all of them. -/
theorem query_rows (rss : Int) (write : Bool) (bs : List (List α)) :
    (query rss write bs).1.flatten =
      if rss ≥ 0 then bs.flatten.take rss.toNat else bs.flatten := by
  simp only [query]
  split
  · rw [capLoop_flatten _ _ 0 bs (Nat.zero_le _)]; rfl
  · rfl

theorem capLoop_pulls_all_on_write (limit : Nat) :
    ∀ (total : Nat) (bs : List (List α)), (capLoop limit true total bs).2 = bs.length
  | _, [] => rfl
  | total, b :: bs => by
    simp only [capLoop]
    split
    · simp; omega
    · simp [capLoop_pulls_all_on_write limit _ bs]

/-- **PROVEN**: a write query always pulls every batch (so every `Commit` runs), however
small the result-set cap. -/
theorem query_write_pulls_all (rss : Int) (bs : List (List α)) :
    (query rss true bs).2 = bs.length := by
  simp only [query]; split
  · exact capLoop_pulls_all_on_write _ 0 bs
  · rfl

/-- A read query stops pulling at the batch that fills the cap. -/
example : (query 1 false [[1], [2], [3]]).2 = 1 ∧ (query 1 false [[1], [2], [3]]).1 = [[1]] := by
  decide

/-- `RESULTSET_SIZE 0` returns no rows but still runs the plan once (issue #2683 is only
a documentation request). -/
example : (query 0 false [[1, 2], [3]]).1 = [[]] := by decide

end RuntimeCore.Query
