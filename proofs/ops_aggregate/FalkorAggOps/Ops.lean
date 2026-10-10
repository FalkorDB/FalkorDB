/-
Operators: Limit, Skip, Sort (full sort and the bounded top-k heap), Project,
Unwind, Distinct and the Aggregate operator's grouping and DISTINCT state.

A batch is modelled by its list of ACTIVE rows (`Batch::active_indices`,
batch.rs:1204 — the selection vector when present, else `0..len`). A stream
of batches is a `List (List α)`; its row sequence is `flatten`.
-/

namespace FalkorAggOps

/-! ## Limit (ops/limit.rs:52) -/

/-- `LimitOp::next`, run to exhaustion. `remaining = 0` stops pulling
(limit.rs:54); an empty batch is skipped (:65); a batch that fits passes
through (:70); otherwise its first `remaining` active rows are kept (:78). -/
def limitOp {α : Type} : Nat → List (List α) → List (List α)
  | 0, _ => []
  | _, [] => []
  | r + 1, b :: bs =>
    if b.length = 0 then limitOp (r + 1) bs
    else if r + 1 ≥ b.length then b :: limitOp (r + 1 - b.length) bs
    else [b.take (r + 1)]

theorem limitOp_spec {α : Type} (k : Nat) (bs : List (List α)) :
    (limitOp k bs).flatten = bs.flatten.take k := by
  induction bs generalizing k with
  | nil => cases k <;> simp [limitOp]
  | cons b bs ih =>
    cases k with
    | zero => simp [limitOp]
    | succ r =>
      simp only [limitOp]
      by_cases h0 : b.length = 0
      · have : b = [] := List.eq_nil_of_length_eq_zero h0
        subst this; simp [ih]
      · simp only [h0, if_false]
        by_cases h1 : r + 1 ≥ b.length
        · simp only [h1, if_true, List.flatten_cons, ih, List.take_append]
          have : b.take (r + 1) = b := List.take_of_length_le h1
          rw [this]
        · simp only [h1, if_false, List.flatten_cons, List.flatten_nil, List.append_nil,
            List.take_append]
          have : r + 1 - b.length = 0 := by omega
          simp [this]

/-- `LIMIT 0` yields nothing and pulls nothing. -/
theorem limit_zero {α : Type} (bs : List (List α)) : limitOp 0 bs = [] := by
  cases bs <;> rfl

/-! ## Skip (ops/skip.rs:50) -/

def skipOp {α : Type} : Nat → List (List α) → List (List α)
  | _, [] => []
  | 0, b :: bs => b :: skipOp 0 bs
  | r + 1, b :: bs =>
    if r + 1 ≥ b.length then skipOp (r + 1 - b.length) bs
    else b.drop (r + 1) :: skipOp 0 bs

theorem skipOp_zero {α : Type} (bs : List (List α)) : skipOp 0 bs = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [skipOp, ih]

theorem skipOp_spec {α : Type} (s : Nat) (bs : List (List α)) :
    (skipOp s bs).flatten = bs.flatten.drop s := by
  induction bs generalizing s with
  | nil => cases s <;> simp [skipOp]
  | cons b bs ih =>
    cases s with
    | zero => simp [skipOp_zero]
    | succ r =>
      simp only [skipOp]
      by_cases h1 : r + 1 ≥ b.length
      · simp only [h1, if_true, ih, List.flatten_cons, List.drop_append]
        have : b.drop (r + 1) = [] := List.drop_eq_nil_of_le h1
        simp [this]
      · simp only [h1, if_false, List.flatten_cons, skipOp_zero, List.drop_append]
        have : r + 1 - b.length = 0 := by omega
        simp [this]

/-- `SKIP` beyond the input yields nothing. -/
theorem skip_beyond {α : Type} (s : Nat) (bs : List (List α)) (h : bs.flatten.length ≤ s) :
    (skipOp s bs).flatten = [] := by
  rw [skipOp_spec]; exact List.drop_eq_nil_of_le h

/-- `SKIP s LIMIT k` (planned as Limit over Skip) is the `[s, s+k)` window. -/
theorem skip_limit {α : Type} (s k : Nat) (bs : List (List α)) :
    (limitOp k (skipOp s bs)).flatten = (bs.flatten.drop s).take k := by
  rw [limitOp_spec, skipOp_spec]

/-! ## Sort: insertion-sort spec and the bounded top-k heap (ops/sort.rs) -/

/-- Rows are identified with their position in the sort's total order
(keys, then row content, then arrival `seq`: `HeapEntry::cmp`, sort.rs:172).
That order is a strict total order exactly when `compare_value` is one —
false for NaN / Int-Float near 2^53 (known #2891). We model the order by `Nat`. -/
def ins (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x < y then x :: y :: ys else y :: ins x ys

def isort (xs : List Nat) : List Nat := xs.foldl (fun acc x => ins x acc) []

def Sorted : List Nat → Prop
  | [] => True
  | [_] => True
  | a :: b :: t => a ≤ b ∧ Sorted (b :: t)

theorem length_ins (x : Nat) (l : List Nat) : (ins x l).length = l.length + 1 := by
  induction l with
  | nil => rfl
  | cons y ys ih => unfold ins; split <;> simp [ih]

/-- Inserting beyond position `c` never affects the first `c` rows. -/
theorem take_ins_take (c x : Nat) (l : List Nat) :
    (ins x (l.take c)).take c = (ins x l).take c := by
  induction l generalizing c with
  | nil => simp
  | cons y ys ih =>
    cases c with
    | zero => simp
    | succ c =>
      simp only [List.take_succ_cons, ins]
      by_cases h : x < y
      · simp only [h, if_true, List.take_succ_cons]
        congr 1
        cases c with
        | zero => simp
        | succ c => simp [List.take_succ_cons, List.take_take]
      · simp only [h, if_false, List.take_succ_cons, ih]

/-- The semantic heap step: insert, then evict the worst row if over capacity. -/
def semStep (c : Nat) (h : List Nat) (x : Nat) : List Nat := (ins x h).take c

theorem topk_sem (c : Nat) (xs : List Nat) :
    xs.foldl (semStep c) [] = (isort xs).take c := by
  have key : ∀ (l : List Nat) (acc : List Nat),
      l.foldl (semStep c) (acc.take c) = (l.foldl (fun a x => ins x a) acc).take c := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons x l ih =>
      intro acc
      simp only [List.foldl]
      rw [show semStep c (acc.take c) x = (ins x acc).take c from take_ins_take c x acc]
      exact ih (ins x acc)
  have := key xs []
  simpa [isort] using this

theorem sorted_ins (x : Nat) (l : List Nat) (h : Sorted l) : Sorted (ins x l) := by
  induction l with
  | nil => trivial
  | cons y ys ih =>
    unfold ins
    by_cases hxy : x < y
    · simp only [hxy, if_true]; exact ⟨Nat.le_of_lt hxy, h⟩
    · simp only [hxy, if_false]
      cases ys with
      | nil => exact ⟨by omega, trivial⟩
      | cons z zs =>
        have hs : Sorted (z :: zs) := h.2
        have ih' := ih hs
        unfold ins at ih' ⊢
        by_cases hxz : x < z
        · simp only [hxz, if_true] at ih' ⊢; exact ⟨by omega, ih'⟩
        · simp only [hxz, if_false] at ih' ⊢; exact ⟨h.1, ih'⟩

theorem sorted_isort (xs : List Nat) : Sorted (isort xs) := by
  have : ∀ acc, Sorted acc → Sorted (xs.foldl (fun a x => ins x a) acc) := by
    induction xs with
    | nil => intro acc h; exact h
    | cons x xs ih => intro acc h; exact ih _ (sorted_ins x acc h)
  exact this [] trivial

theorem sorted_take (c : Nat) (l : List Nat) (h : Sorted l) : Sorted (l.take c) := by
  induction l generalizing c with
  | nil => simp [Sorted]
  | cons a t ih =>
    cases c with
    | zero => trivial
    | succ c =>
      cases t with
      | nil => cases c <;> trivial
      | cons b t' =>
        cases c with
        | zero => trivial
        | succ c =>
          simp only [List.take_succ_cons]
          have := ih (c + 1) h.2
          simp only [List.take_succ_cons] at this
          exact ⟨h.1, this⟩

theorem ins_lt_last (x : Nat) (h : List Nat) (hne : h ≠ []) (hx : x < h.getLast hne) :
    ins x h = ins x h.dropLast ++ [h.getLast hne] := by
  induction h with
  | nil => exact absurd rfl hne
  | cons y t ih =>
    cases t with
    | nil => simp [ins] at *; omega
    | cons z zs =>
      have hl : (y :: z :: zs).getLast hne = (z :: zs).getLast (by simp) := by simp
      rw [hl] at hx ⊢
      simp only [List.dropLast_cons_cons]
      unfold ins
      by_cases hxy : x < y
      · simp only [hxy, if_true, List.cons_append]
        congr 2
        exact (List.dropLast_concat_getLast (by simp)).symm
      · simp only [hxy, if_false, List.cons_append]
        congr 1
        exact ih (by simp) hx

/-- `build_top_k`'s loop body (sort.rs:425-469): push while below capacity;
when full, replace the worst (`peek`, the last of the sorted view) only if the
new row sorts strictly before it. `cap = 0` drains the child (sort.rs:393). -/
def rustStep (c : Nat) (h : List Nat) (x : Nat) : List Nat :=
  if c = 0 then [] else
  if h.length < c then ins x h
  else match h.getLast? with
    | none => h
    | some w => if x < w then ins x h.dropLast else h

theorem sorted_le_last : ∀ (h : List Nat) (hne : h ≠ []), Sorted h → ∀ y ∈ h, y ≤ h.getLast hne
  | [], hne, _, _, _ => absurd rfl hne
  | [a], _, _, y, hy => by simp at hy; simp [hy]
  | a :: b :: t, _, hs, y, hy => by
    have ih := sorted_le_last (b :: t) (by simp) hs.2
    rw [List.getLast_cons_cons]
    cases List.mem_cons.mp hy with
    | inl e => subst e; exact Nat.le_trans hs.1 (ih b (by simp))
    | inr e => exact ih y e

theorem ins_all_le (x : Nat) : ∀ (h : List Nat), (∀ y ∈ h, y ≤ x) → ins x h = h ++ [x]
  | [], _ => rfl
  | y :: t, hall => by
    have hy : ¬ x < y := by have := hall y (by simp); omega
    simp only [ins, hy, if_false, List.cons_append]
    rw [ins_all_le x t (fun z hz => hall z (by simp [hz]))]

theorem take_ins_ge_last (x : Nat) (h : List Nat) (hs : Sorted h) (hne : h ≠ [])
    (hx : h.getLast hne ≤ x) : (ins x h).take h.length = h := by
  rw [ins_all_le x h (fun y hy => Nat.le_trans (sorted_le_last h hne hs y hy) hx)]
  simp

/-- The Rust heap step is the semantic step on every reachable heap
(sorted, at most `c` rows). -/
theorem rustStep_eq_sem (c : Nat) (h : List Nat) (x : Nat) (hs : Sorted h)
    (hl : h.length ≤ c) : rustStep c h x = semStep c h x := by
  unfold rustStep semStep
  by_cases hc : c = 0
  · simp [hc]
  simp only [hc, if_false]
  by_cases hlt : h.length < c
  · simp only [hlt, if_true]
    rw [List.take_of_length_le]; rw [length_ins]; omega
  · simp only [hlt, if_false]
    have heq : h.length = c := by omega
    have hne : h ≠ [] := by intro e; subst e; simp at heq; exact hc heq.symm
    rw [List.getLast?_eq_some_getLast hne]
    simp only
    by_cases hx : x < h.getLast hne
    · simp only [hx, if_true]
      rw [ins_lt_last x h hne hx, List.take_append_of_le_length]
      · rw [List.take_of_length_le]; rw [length_ins, List.length_dropLast]; omega
      · rw [length_ins, List.length_dropLast]; omega
    · simp only [hx, if_false]
      rw [← heq]; exact (take_ins_ge_last x h hs hne (by omega)).symm

/-- Headline: the streaming top-k heap returns exactly the first `cap` rows of
the full sort (sort.rs:13-17 claim, and the full-sort path's `truncate`). -/
theorem topk_eq_take_sort (c : Nat) (xs : List Nat) :
    xs.foldl (rustStep c) [] = (isort xs).take c := by
  rw [← topk_sem]
  have : ∀ (l : List Nat) (h : List Nat), Sorted h → h.length ≤ c →
      l.foldl (rustStep c) h = l.foldl (semStep c) h := by
    intro l
    induction l with
    | nil => intro h _ _; rfl
    | cons x l ih =>
      intro h hs hl
      simp only [List.foldl]
      rw [rustStep_eq_sem c h x hs hl]
      apply ih
      · exact sorted_take c _ (sorted_ins x h hs)
      · simp [semStep]; omega
  exact this xs [] trivial (by simp)

/-- `ORDER BY … SKIP s LIMIT k`: the Sort op keeps `cap = k + s` rows
(sort.rs:493, `saturating_add` — no wrap), and Skip/Limit above it cut the
window; that is the `[s, s+k)` window of the full sort. -/
theorem sort_skip_limit (s k : Nat) (xs : List Nat) :
    ((xs.foldl (rustStep (k + s)) []).drop s).take k = ((isort xs).drop s).take k := by
  rw [topk_eq_take_sort, List.drop_take]
  simp [List.take_take]

/-! ## Project and Unwind: batching is invisible -/

/-- `ProjectOp::eval` (project.rs:54): one output row per active input row. -/
theorem project_flatten {α β : Type} (f : α → β) (bs : List (List α)) :
    (bs.map (List.map f)).flatten = bs.flatten.map f := by
  simp [List.map_flatten]

/-- `UNWIND` (unwind.rs:90, `eval_iter_expr`): a list spreads, `null` yields
nothing, a scalar yields itself. -/
inductive UV where
  | null
  | scalar (n : Nat)
  | list (l : List UV)

def unwindOne : UV → List UV
  | .null => []
  | .list l => l
  | v => [v]

/-- The emitter packs the flattened output into chunks of `BATCH_SIZE`
(batched_result_emitter.rs); whatever the chunking, the row sequence is
the flat-map of the input. -/
def chunks {α : Type} (n : Nat) (xs : List α) : List (List α) :=
  if h : n = 0 ∨ xs = [] then (if xs = [] then [] else [xs]) else
    xs.take n :: chunks n (xs.drop n)
termination_by xs.length
decreasing_by
  simp only [not_or] at h
  have : xs.length ≠ 0 := fun e => h.2 (List.eq_nil_of_length_eq_zero e)
  simp [List.length_drop]; omega

theorem chunks_flatten {α : Type} (n : Nat) (xs : List α) : (chunks n xs).flatten = xs := by
  induction xs using (measure List.length).wf.induction with
  | h xs ih =>
    unfold chunks
    split
    · split <;> simp_all
    · rename_i h
      simp only [not_or] at h
      have hl : (xs.drop n).length < xs.length := by
        have : xs.length ≠ 0 := fun e => h.2 (List.eq_nil_of_length_eq_zero e)
        simp [List.length_drop]; omega
      simp [ih _ hl]

theorem unwind_batches (n : Nat) (bs : List (List UV)) :
    (chunks n (bs.flatten.flatMap unwindOne)).flatten = bs.flatten.flatMap unwindOne :=
  chunks_flatten n _

theorem unwind_null : unwindOne .null = [] := rfl

/-! ## Grouping (ops/aggregate.rs `GroupMap`, :59) — bag semantics -/

/-- `groups.entry(key).or_insert_with(..)` then fold: an association list,
new keys appended (HashMap iteration order is not modelled). -/
def groupInsert {K A : Type} [DecidableEq K] (k : K) (a : A) :
    List (K × List A) → List (K × List A)
  | [] => [(k, [a])]
  | (k', as) :: g => if k' = k then (k', as ++ [a]) :: g else (k', as) :: groupInsert k a g

def groupAll {K A : Type} [DecidableEq K] (rows : List (K × A)) : List (K × List A) :=
  rows.foldl (fun g r => groupInsert r.1 r.2 g) []

def lookupG {K A : Type} [DecidableEq K] (k : K) : List (K × List A) → Option (List A)
  | [] => none
  | (k', as) :: g => if k' = k then some as else lookupG k g

theorem lookup_groupInsert {K A : Type} [DecidableEq K] (k j : K) (a : A) (g : List (K × List A)) :
    lookupG j (groupInsert k a g) =
      if k = j then some ((lookupG j g).getD [] ++ [a]) else lookupG j g := by
  induction g with
  | nil => simp [groupInsert, lookupG]
  | cons e g ih =>
    obtain ⟨k', as⟩ := e
    simp only [groupInsert]
    by_cases h1 : k' = k
    · subst h1; by_cases h2 : k' = j <;> simp [lookupG, h2]
    · simp only [h1, if_false, lookupG]
      by_cases h2 : k' = j
      · subst h2; simp [lookupG, h1, Ne.symm h1]
      · simp [h2, ih]

/-- Every group holds exactly the rows with its key, in arrival order: the
aggregate of group `k` is the aggregate of `filter (key = k)` — bag
semantics, nulls included as ordinary keys (C, openCypher). -/
theorem groupAll_spec {K A : Type} [DecidableEq K] (rows : List (K × A)) (k : K) :
    (lookupG k (groupAll rows)).getD [] = (rows.filter (fun r => r.1 = k)).map Prod.snd := by
  have : ∀ (g : List (K × List A)),
      (lookupG k (rows.foldl (fun g r => groupInsert r.1 r.2 g) g)).getD [] =
        (lookupG k g).getD [] ++ (rows.filter (fun r => r.1 = k)).map Prod.snd := by
    induction rows with
    | nil => intro g; simp
    | cons r rows ih =>
      intro g
      simp only [List.foldl, ih, lookup_groupInsert, List.filter]
      by_cases h : r.1 = k <;> simp [h]
  simpa [groupAll, lookupG] using this []

/-- Keyed aggregation over no rows has no groups; keyless aggregation
pre-inserts the single empty-key group (aggregate.rs:524, :879), so it always
emits one row of initial accumulators (`count`=0, `sum`=0.0, `avg`=null,
`collect`=[] …; checked in `agrees_empty_input_aggregates`). -/
theorem keyed_empty {K A : Type} [DecidableEq K] : groupAll ([] : List (K × A)) = [] := rfl

def keylessGroups {A : Type} (rows : List A) : List (Unit × List A) :=
  [((), rows)]

theorem keyless_one_row {A : Type} (rows : List A) : (keylessGroups rows).length = 1 := rfl

/-! ## DISTINCT state keyed by a hash (distinct.rs:79, aggregate.rs:715/:959) -/

/-- Keep the first row of each `key` class (`ValuesDeduper::is_seen`). -/
def dedupBy {α β : Type} [DecidableEq β] (key : α → β) : List β → List α → List α
  | _, [] => []
  | seen, x :: xs => if key x ∈ seen then dedupBy key seen xs
                     else x :: dedupBy key (key x :: seen) xs

theorem dedup_injective {α β : Type} [DecidableEq α] [DecidableEq β] (h : α → β)
    (inj : ∀ a b, h a = h b → a = b) (seen : List α) (xs : List α) :
    dedupBy h (seen.map h) xs = dedupBy id seen xs := by
  induction xs generalizing seen with
  | nil => rfl
  | cons x xs ih =>
    simp only [dedupBy, id]
    have hm : (h x ∈ seen.map h) ↔ x ∈ seen := by
      constructor
      · intro hx; obtain ⟨y, hy, e⟩ := List.mem_map.mp hx; rw [← inj _ _ e]; exact hy
      · intro hx; exact List.mem_map.mpr ⟨x, hx, rfl⟩
    by_cases hx : x ∈ seen
    · simp [hx, hm.mpr hx, ih]
    · have : ¬ h x ∈ seen.map h := fun e => hx (hm.mp e)
      simp only [this, hx, if_false]
      have := ih (x :: seen)
      simp only [List.map_cons] at this
      rw [this]

/-- …and with a collision it drops a row (`#2780`; the grouped-aggregate
variant below is new). -/
theorem dedup_collision_drops :
    dedupBy (fun n : Nat => n % 2) [] [0, 2] = [0] ∧ dedupBy id [] [0, 2] = [0, 2] := by
  decide

/-- Per-group DISTINCT: the dedup table is keyed by `(distinct node,
hash(group key row))`, not by the group key (aggregate.rs:715, :959). Two
groups whose key rows collide share one `seen` set. -/
def dcgGo {K : Type} [DecidableEq K] (hk : K → Nat) (k : K) :
    List (Nat × Nat) → List (K × Nat) → Nat
  | _, [] => 0
  | seen, (k', v) :: rs =>
    if (hk k', v) ∈ seen then dcgGo hk k seen rs
    else (if k' = k then 1 else 0) + dcgGo hk k ((hk k', v) :: seen) rs

def distinctCountGrouped {K : Type} [DecidableEq K] (hk : K → Nat) (rows : List (K × Nat))
    (k : K) : Nat :=
  dcgGo hk k [] rows

-- Two distinct keys 0 and 2 that hash alike (hk = · % 2), one value each.
/-- The second group's `count(DISTINCT v)` is 0 instead of 1 — confirmed by
`lean_ops_aggregate::bug_grouped_count_distinct_shares_state_on_group_hash_collision`
with the real FxHash collision `[0,0]` / `[1,-1452335207727870361]`. -/
theorem hashed_dedup_merges_groups :
    distinctCountGrouped (fun n : Nat => n % 2) [(0, 7), (2, 7)] 2 = 0 ∧
    distinctCountGrouped (fun n : Nat => n) [(0, 7), (2, 7)] 2 = 1 := by
  decide

/-- `count(DISTINCT v)` fed batch by batch, with an optional wipe of the
(runtime-global) dedup map between batches — what `ApplyOp`'s per-row mode
does after every subquery (apply.rs:283). -/
def distinctCountBatches (clear : Bool) : List Nat → List (List Nat) → Nat
  | _, [] => 0
  | seen, b :: bs =>
    let d := dedupBy id seen b
    d.length + distinctCountBatches clear (if clear then [] else (d.reverse ++ seen)) bs

theorem dedup_seen_mono (seen xs : List Nat) :
    ∀ y, y ∈ seen → y ∉ dedupBy id seen xs := by
  induction xs generalizing seen with
  | nil => intro y _; simp [dedupBy]
  | cons x xs ih =>
    intro y hy
    simp only [dedupBy, id]
    split
    · exact ih seen y hy
    · rename_i hx
      intro hm
      cases List.mem_cons.mp hm with
      | inl e => subst e; exact hx hy
      | inr e => exact ih (x :: seen) y (List.mem_cons_of_mem _ hy) e

/-- Correct for a single batch — the bug needs a batch boundary. -/
theorem distinct_one_batch (b : List Nat) :
    distinctCountBatches true [] [b] = (dedupBy id [] b).length := by
  simp [distinctCountBatches]

/-- Counterexample (#2779, confirmed by
`lean_ops_aggregate::bug_count_distinct_across_call_subquery_overcounts`:
3000 rows = 3 batches, `count(DISTINCT i % 2)` = 6, C = 2). -/
theorem distinct_fold_clear_overcounts :
    distinctCountBatches true [] [[1, 0], [1, 0], [1, 0]] = 6 ∧
    distinctCountBatches false [] [[1, 0], [1, 0], [1, 0]] = 2 ∧
    (dedupBy id [] [1, 0, 1, 0, 1, 0]).length = 2 := by
  decide

/-! ## eval.rs:877 — an aggregate's finalizer `return`s out of the stack machine -/

/-- A two-node fragment of `ExprEval::eval_compound`'s stack machine: list
literals and finalized aggregates. -/
inductive E where
  | agg (acc : Nat)
  | lit (n : Nat)
  | list (es : List E)

inductive R where
  | num (n : Nat)
  | lst (rs : List R)

/-- Reference: recursive evaluation (what `eval_node` does for Map, Add, …). -/
def evalRef : E → R
  | .agg a => .num a
  | .lit n => .num n
  | .list es => .lst (es.attach.map fun ⟨e, _⟩ => evalRef e)

/-- The Rust stack machine: on `FuncInvocation` of an aggregate with
`agg_group_key = None` it executes `return match finalize { … }` — leaving the
whole function with that value, abandoning every pending `List` frame. -/
def evalStack (e : E) : R :=
  match e with
  | .list es =>
    match es.find? (fun e => match e with | .agg _ => true | _ => false) with
    -- The children are pushed in order and popped last-first, so the LAST
    -- aggregate child is the first one evaluated and its `return` wins.
    | some _ =>
      match (es.reverse.find? (fun e => match e with | .agg _ => true | _ => false)) with
      | some (.agg a) => .num a
      | _ => evalRef e
    | none => evalRef e
  | e => evalRef e

theorem stackEval_return_drops_frame :
    evalStack (.list [.agg 1, .agg 2]) = .num 2 ∧
    evalRef (.list [.agg 1, .agg 2]) = .lst [.num 1, .num 2] := by
  constructor
  · rfl
  · simp [evalRef]

/-! ## finalize_percentile_{disc,cont}: index bounds and the NaN comparator -/

/-- `percentileDisc` index `ceil(n·p) − 1` (aggregation.rs:621) with the
percentile a rational `a/b ∈ (0,1]` is in bounds. (For f64: `n·p ≤ n` holds
under round-to-nearest because `p ≤ 1` and `n·1 = n` exactly.) -/
theorem disc_index_in_bounds (n a b : Nat) (hn : 0 < n) (ha : 0 < a) (hab : a ≤ b) :
    (n * a + b - 1) / b - 1 < n ∧ 1 ≤ (n * a + b - 1) / b := by
  have hb : 0 < b := Nat.lt_of_lt_of_le ha hab
  have h1 : n * a ≤ n * b := Nat.mul_le_mul_left n hab
  have h2 : 1 ≤ n * a := Nat.mul_pos hn ha
  constructor
  · have : (n * a + b - 1) / b < n + 1 := by
      rw [Nat.div_lt_iff_lt_mul hb, Nat.add_mul, Nat.one_mul]; omega
    omega
  · rw [Nat.le_div_iff_mul_le hb]; omega

/-- `percentileCont` reads `values[index + 1]` only when the fraction is
nonzero (aggregation.rs:661-665); then `index + 1 < n`. -/
theorem cont_index_in_bounds (n a b : Nat) (hab : a ≤ b) (hb : 0 < b)
    (hfrac : (n - 1) * a % b ≠ 0) : (n - 1) * a / b + 1 < n := by
  by_cases heq : a = b
  · subst heq; simp at hfrac
  have hlt : a < b := Nat.lt_of_le_of_ne hab heq
  have hn : 0 < n - 1 := by
    rcases Nat.eq_zero_or_pos (n - 1) with h | h
    · rw [h] at hfrac; simp at hfrac
    · exact h
  have : (n - 1) * a < (n - 1) * b := Nat.mul_lt_mul_of_pos_left hlt hn
  have : (n - 1) * a / b < n - 1 := by rw [Nat.div_lt_iff_lt_mul hb]; exact this
  omega

/-- The percentile sort comparator `partial_cmp(..).unwrap_or(Equal)`
(aggregation.rs:615, :645) with NaN (modelled as `none`) is not transitive:
1 ~ NaN ~ 3 but 1 < 3. std's sort panics on such comparators — confirmed:
`lean_ops_aggregate::bug_percentile_with_nan_panics` (and it kills a live
server). -/
def pcmpEq : Option Nat → Option Nat → Ordering
  | some a, some b => compare a b
  | _, _ => .eq

theorem partialCmpEq_not_transitive :
    pcmpEq (some 1) none = .eq ∧ pcmpEq none (some 3) = .eq ∧ pcmpEq (some 1) (some 3) = .lt := by
  decide

end FalkorAggOps
