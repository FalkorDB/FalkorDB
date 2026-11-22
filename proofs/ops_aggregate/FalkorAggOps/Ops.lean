/-
Operators: Limit, Skip, Sort (full sort and the bounded top-k heap), Project,
Unwind, Distinct and the Aggregate operator's grouping and DISTINCT state.

A batch is modelled by its list of ACTIVE rows (`Batch::active_indices`,
batch.rs — the selection vector when present, else `0..len`). A stream
of batches is a `List (List α)`; its row sequence is `flatten`.
-/

namespace FalkorAggOps

/-! ## Limit (ops/limit.rs) -/

/-- `LimitOp::next`, run to exhaustion. `remaining = 0` stops pulling
(limit.rs); an empty batch is skipped (:65); a batch that fits passes
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
      · simp only [h0, ite_false]
        by_cases h1 : r + 1 ≥ b.length
        · simp only [h1, ite_true, List.flatten_cons, ih, List.take_append]
          have : b.take (r + 1) = b := List.take_of_length_le h1
          rw [this]
        · simp only [h1, ite_false, List.flatten_cons, List.flatten_nil, List.append_nil,
            List.take_append]
          have : r + 1 - b.length = 0 := by omega
          simp [this]

/-- `LIMIT 0` yields nothing and pulls nothing. -/
theorem limit_zero {α : Type} (bs : List (List α)) : limitOp 0 bs = [] := by
  cases bs <;> rfl

/-! ## Skip (ops/skip.rs) -/

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
      · simp only [h1, ite_true, ih, List.flatten_cons, List.drop_append]
        have : b.drop (r + 1) = [] := List.drop_eq_nil_of_le h1
        simp [this]
      · simp only [h1, ite_false, List.flatten_cons, skipOp_zero, List.drop_append]
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
(keys, then row content, then arrival `seq`: `HeapEntry::cmp`, sort.rs).
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
      · simp only [h, ite_true, List.take_succ_cons]
        congr 1
        cases c with
        | zero => simp
        | succ c => simp [List.take_succ_cons, List.take_take]
      · simp only [h, ite_false, List.take_succ_cons, ih]

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
    · simp only [hxy, ite_true]; exact ⟨Nat.le_of_lt hxy, h⟩
    · simp only [hxy, ite_false]
      cases ys with
      | nil => exact ⟨by omega, trivial⟩
      | cons z zs =>
        have hs : Sorted (z :: zs) := h.2
        have ih' := ih hs
        unfold ins at ih' ⊢
        by_cases hxz : x < z
        · simp only [hxz, ite_true] at ih' ⊢; exact ⟨by omega, ih'⟩
        · simp only [hxz, ite_false] at ih' ⊢; exact ⟨h.1, ih'⟩

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
      · simp only [hxy, ite_true, List.cons_append]
        congr 2
        exact (List.dropLast_concat_getLast (by simp)).symm
      · simp only [hxy, ite_false, List.cons_append]
        congr 1
        exact ih (by simp) hx

/-- `build_top_k`'s loop body (sort.rs): push while below capacity;
when full, replace the worst (`peek`, the last of the sorted view) only if the
new row sorts strictly before it. `cap = 0` drains the child (sort.rs). -/
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
    simp only [ins, hy, ite_false, List.cons_append]
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
  simp only [hc, ite_false]
  by_cases hlt : h.length < c
  · simp only [hlt, ite_true]
    rw [List.take_of_length_le]; rw [length_ins]; omega
  · simp only [hlt, ite_false]
    have heq : h.length = c := by omega
    have hne : h ≠ [] := by intro e; subst e; simp at heq; exact hc heq.symm
    rw [List.getLast?_eq_some_getLast hne]
    simp only
    by_cases hx : x < h.getLast hne
    · simp only [hx, ite_true]
      rw [ins_lt_last x h hne hx, List.take_append_of_le_length]
      · rw [List.take_of_length_le]; rw [length_ins, List.length_dropLast]; omega
      · rw [length_ins, List.length_dropLast]; omega
    · simp only [hx, ite_false]
      rw [← heq]; exact (take_ins_ge_last x h hs hne (by omega)).symm

/-- Headline: the streaming top-k heap returns exactly the first `cap` rows of
the full sort (sort.rs claim, and the full-sort path's `truncate`). -/
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
(sort.rs, `saturating_add` — no wrap), and Skip/Limit above it cut the
window; that is the `[s, s+k)` window of the full sort. -/
theorem sort_skip_limit (s k : Nat) (xs : List Nat) :
    ((xs.foldl (rustStep (k + s)) []).drop s).take k = ((isort xs).drop s).take k := by
  rw [topk_eq_take_sort, List.drop_take]
  simp [List.take_take]

/-! ## Project and Unwind: batching is invisible -/

/-- `ProjectOp::eval` (project.rs): one output row per active input row. -/
theorem project_flatten {α β : Type} (f : α → β) (bs : List (List α)) :
    (bs.map (List.map f)).flatten = bs.flatten.map f := by
  simp [List.map_flatten]

/-- `UNWIND` (unwind.rs, `eval_iter_expr`): a list spreads, `null` yields
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

/-! ## Grouping (ops/aggregate.rs `GroupMap`, :65) — bag semantics -/

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
    · simp only [h1, ite_false, lookupG]
      by_cases h2 : k' = j
      · subst h2; simp [Ne.symm h1]
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
pre-inserts the single empty-key group (aggregate.rs), so it always
emits one row of initial accumulators (`count`=0, `sum`=0.0, `avg`=null,
`collect`=[] …; checked in `agrees_empty_input_aggregates`). -/
theorem keyed_empty {K A : Type} [DecidableEq K] : groupAll ([] : List (K × A)) = [] := rfl

def keylessGroups {A : Type} (rows : List A) : List (Unit × List A) :=
  [((), rows)]

theorem keyless_one_row {A : Type} (rows : List A) : (keylessGroups rows).length = 1 := rfl

/-! ## DISTINCT deduplication

The per-group state is keyed by a group id; see `GroupId.lean`
(`distinct_by_group_id_exact`, `fixed2977_collision_example`). -/

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
      simp only [this, hx, ite_false]
      have := ih (x :: seen)
      simp only [List.map_cons] at this
      rw [this]

/-- …and with a collision it drops a row (`#2780`). -/
theorem dedup_collision_drops :
    dedupBy (fun n : Nat => n % 2) [] [0, 2] = [0] ∧ dedupBy id [] [0, 2] = [0, 2] := by
  decide

/-! ## DISTINCT across batches: the per-row CALL {} reset is scoped (#2968, fixes #2779)

`Runtime::value_dedupers` is keyed `(distinct node, group id)` and shared by the whole query.
After each per-row run of a CALL {} body, `ApplyOp::next_per_row` (apply.rs) now removes only
the body's own entries: `(node, gid)` for each `Distinct` node of the sub-plan
(`sub_plan_distincts`) and `gid < aggregate_group_counts[node]` (the largest `groups.len()`
the body's aggregate reached, recorded by `consume_input_*`, aggregate.rs).  Entries are
modelled as `(node, group id, value)`. -/

abbrev Entry := Nat × Nat × Nat

/-- The per-row reset: drop `(node, gid, _)` with `node ∈ subs ∧ gid < cnt node`. -/
def scopedClear (subs : List Nat) (cnt : Nat → Nat) (m : List Entry) : List Entry :=
  m.filter fun e => !(subs.contains e.1 && decide (e.2.1 < cnt e.1))

/-- The values seen by group 0 of distinct node `o`. -/
def seenOf (o : Nat) (m : List Entry) : List Nat :=
  (m.filter fun e => e.1 == o && e.2.1 == 0).map (·.2.2)

/-- An outer keyless `count(DISTINCT v)` (distinct node `o`, group 0) fed batch by batch,
with the scoped reset of the CALL {} below it run between batches. -/
def distinctCountScoped (o : Nat) (subs : List Nat) (cnt : Nat → Nat) :
    List Entry → List (List Nat) → Nat
  | _, [] => 0
  | m, b :: bs =>
    let d := dedupBy id (seenOf o m) b
    d.length + distinctCountScoped o subs cnt (scopedClear subs cnt (d.reverse.map (o, 0, ·) ++ m)) bs

theorem dedupBy_append (seen a b : List Nat) :
    dedupBy id seen (a ++ b) = dedupBy id seen a ++ dedupBy id ((dedupBy id seen a).reverse ++ seen) b := by
  induction a generalizing seen with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.cons_append, dedupBy, id]
    split
    · exact ih seen
    · rw [ih (x :: seen)]; simp

theorem seenOf_append (o : Nat) (a m : List Entry) : seenOf o (a ++ m) = seenOf o a ++ seenOf o m := by
  simp [seenOf, List.filter_append]

theorem seenOf_tag (o : Nat) (d : List Nat) : seenOf o (d.map (o, 0, ·)) = d := by
  induction d with
  | nil => rfl
  | cons x xs ih => simp only [seenOf, List.map_cons, List.filter_cons] at ih ⊢; simp [ih]

/-- The reset never touches a distinct node outside the sub-plan. -/
theorem seenOf_scopedClear (o : Nat) (subs : List Nat) (cnt : Nat → Nat) (m : List Entry)
    (ho : o ∉ subs) : seenOf o (scopedClear subs cnt m) = seenOf o m := by
  unfold seenOf scopedClear
  rw [List.filter_filter]
  congr 1
  apply List.filter_congr
  intro e _
  by_cases h : e.1 = o
  · subst h; simp [ho]
  · simp [h]

/-- After the reset no entry of the sub-plan's distinct nodes with a group id below the
recorded count survives: the next row's body starts from an empty dedup state. -/
theorem scopedClear_clears (subs : List Nat) (cnt : Nat → Nat) (m : List Entry) (e : Entry)
    (he : e ∈ scopedClear subs cnt m) (hs : e.1 ∈ subs) : cnt e.1 ≤ e.2.1 := by
  simp only [scopedClear, List.mem_filter, Bool.not_eq_true', Bool.and_eq_false_iff,
    decide_eq_false_iff_not, Nat.not_lt] at he
  rcases he.2 with h | h
  · have := List.contains_iff_mem.mpr hs; rw [this] at h; cases h
  · exact h

/-- Every other entry is kept. -/
theorem scopedClear_keeps (subs : List Nat) (cnt : Nat → Nat) (m : List Entry) (e : Entry)
    (he : e ∈ m) (hs : e.1 ∉ subs ∨ cnt e.1 ≤ e.2.1) : e ∈ scopedClear subs cnt m := by
  simp only [scopedClear, List.mem_filter]
  refine ⟨he, ?_⟩
  rcases hs with h | h
  · simp [h]
  · simp [Nat.not_lt.mpr h]

theorem distinctCountScoped_go (o : Nat) (subs : List Nat) (cnt : Nat → Nat) (ho : o ∉ subs)
    (m : List Entry) (bs : List (List Nat)) :
    distinctCountScoped o subs cnt m bs = (dedupBy id (seenOf o m) bs.flatten).length := by
  induction bs generalizing m with
  | nil => rfl
  | cons b bs ih =>
    simp only [distinctCountScoped, List.flatten_cons, dedupBy_append, List.length_append, ih,
      seenOf_scopedClear o subs cnt _ ho, seenOf_append]
    rw [seenOf_tag]

/-- **#2779 / W2-agg-1 fixed**: an outer `count(DISTINCT)` whose distinct node is not inside
the CALL {} counts exactly as over the whole input in one batch, however the rows are
batched. -/
theorem distinct_scoped_clear_exact (o : Nat) (subs : List Nat) (cnt : Nat → Nat) (ho : o ∉ subs)
    (bs : List (List Nat)) :
    distinctCountScoped o subs cnt [] bs = (dedupBy id [] bs.flatten).length :=
  distinctCountScoped_go o subs cnt ho [] bs

/-- #2779's query: 3 batches of `[1, 0]` give 2 (C: 2; before #2968: 6). -/
theorem fixed2779_example :
    distinctCountScoped 0 [1] (fun _ => 1) [] [[1, 0], [1, 0], [1, 0]] = 2 := by
  decide

/-! ## eval.rs — an aggregate's finalized value is pushed

The `FuncInvocation` arm of `ExprEval::eval_compound` pushes the finalized value and
`continue`s (eval.rs at 289c201fe), so any enclosing frame, e.g. the
list literal `[min(x), max(x)]`, still runs. -/

/-- A two-node fragment of `ExprEval::eval_compound`'s stack machine: list
literals and finalized aggregates (`agg acc` stands for the accumulator
already passed through its finalizer). -/
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

theorem evalRef_list (es : List E) : evalRef (.list es) = .lst (es.map evalRef) := by
  simp [evalRef]

/-- One step of the stack machine. `push` is a leaf's value: a literal, or an
aggregate's finalized accumulator, which main pushes (eval.rs at
289c201fe). `mk n` is a list literal's frame: it pops its `n` operands and
pushes the list. -/
inductive I where
  | push (r : R)
  | mk (n : Nat)

/-- Postfix code: operands in source order, then the list frame. -/
def compile : E → List I
  | .agg a => [.push (.num a)]
  | .lit n => [.push (.num n)]
  | .list es => (es.attach.flatMap fun ⟨e, _⟩ => compile e) ++ [.mk es.length]

def step : I → List R → Option (List R)
  | .push r, st => some (r :: st)
  | .mk n, st => if n ≤ st.length then some (.lst (st.take n).reverse :: st.drop n) else none

def run : List I → List R → Option (List R)
  | [], st => some st
  | i :: is, st => (step i st).bind (run is)

/-- Main's stack machine (eval.rs at 289c201fe): the aggregate arm
pushes and `continue`s; the result is the single value left on the stack. -/
def evalStack (e : E) : R :=
  match run (compile e) [] with
  | some [r] => r
  | _ => .lst []

theorem run_append (a b : List I) (st : List R) :
    run (a ++ b) st = (run a st).bind (run b) := by
  induction a generalizing st with
  | nil => simp [run]
  | cons i is ih =>
    simp only [List.cons_append, run]
    cases step i st <;> simp [ih]

theorem run_compile (e : E) (k : List I) (st : List R) :
    run (compile e ++ k) st = run k (evalRef e :: st) := by
  match e with
  | .agg a => simp [compile, run, step, evalRef]
  | .lit n => simp [compile, run, step, evalRef]
  | .list es =>
    have hl : ∀ (xs : List E), (∀ x ∈ xs, x ∈ es) → ∀ (k : List I) (st : List R),
        run ((xs.flatMap compile) ++ k) st = run k ((xs.map evalRef).reverse ++ st) := by
      intro xs hxs
      induction xs with
      | nil => intro k st; simp
      | cons x xs ih =>
        intro k st
        have hx : x ∈ es := hxs x (by simp)
        have _ : sizeOf x < sizeOf (E.list es) := by
          have := List.sizeOf_lt_of_mem hx; simp; omega
        simp only [List.flatMap_cons, List.append_assoc, List.map_cons, List.reverse_cons]
        rw [run_compile x, ih (fun y hy => hxs y (by simp [hy]))]
        simp
    have hfm : (es.attach.flatMap fun ⟨e, _⟩ => compile e) = es.flatMap compile := by
      simp [List.flatMap]
    simp only [compile, hfm, List.append_assoc]
    rw [hl es (fun _ h => h)]
    simp [run, step, evalRef_list]
termination_by sizeOf e

/-- **#2951, general.** Main's stack machine agrees with the recursive
reference on every expression: no aggregate abandons an enclosing frame. -/
theorem evalStack_eq_evalRef (e : E) : evalStack e = evalRef e := by
  have := run_compile e [] []
  simp only [List.append_nil] at this
  simp [evalStack, this, run]

/-- **#2951.** A list literal of finalized aggregates yields one element per
item, each the finalized value, in order. -/
theorem evalStack_list_of_aggs (accs : List Nat) :
    evalStack (.list (accs.map .agg)) = .lst (accs.map .num) := by
  rw [evalStack_eq_evalRef, evalRef_list]
  simp [List.map_map, Function.comp_def, evalRef]

theorem evalStack_list_of_aggs_length (accs : List Nat) :
    ∃ rs, evalStack (.list (accs.map .agg)) = .lst rs ∧ rs.length = accs.length :=
  ⟨accs.map .num, evalStack_list_of_aggs accs, by simp⟩

/-- `size(·)`: the length of a list, a type error (`none`) on a number. -/
def rsize : R → Option Nat
  | .lst rs => some rs.length
  | .num _ => none

/-- `UNWIND [1, 2] AS x RETURN [min(x), max(x)]` is `[1, 2]`
(tests/flow/test_aggregation.py `test_list_literal_of_aggregates`). -/
theorem main_min_max_list : evalStack (.list [.agg 1, .agg 2]) = .lst [.num 1, .num 2] :=
  evalStack_list_of_aggs [1, 2]

/-- `[count(x), 5]` over `[1, 2]` is `[2, 5]`. -/
theorem main_count_lit_list : evalStack (.list [.agg 2, .lit 5]) = .lst [.num 2, .num 5] := by
  rw [evalStack_eq_evalRef]; simp [evalRef]

/-- `size([count(x)])` is 1, not a type error. -/
theorem main_size_singleton_agg : rsize (evalStack (.list [.agg 2])) = some 1 := by
  rw [evalStack_eq_evalRef]; simp [evalRef, rsize]

/-! ## finalize_percentile_{disc,cont}: index bounds and the NaN comparator -/

/-- `percentileDisc` index `ceil(n·p) − 1` (aggregation.rs) with the
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
nonzero (aggregation.rs); then `index + 1 < n`. -/
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

/-- Main (aggregation.rs): `sort_by(|a, b| sort_cmp_f64(a.get_numeric(),
b.get_numeric()))` — `partial_cmp`, else NaN (`none`) above every number and equal to itself.
Numbers are `Nat`s here (any total order on the non-NaN values gives the same laws; the
f64 version is proofs/value_order `SC.scF64_key`/`SC.scF64_lw`). -/
def pctCmp : Option Nat → Option Nat → Ordering
  | some a, some b => compare a b
  | some _, none => .lt
  | none, some _ => .gt
  | none, none => .eq

/-- **Fixed by #2912**: the percentile comparator is a total preorder — reflexive,
antisymmetric, transitive — so the sort cannot panic on NaN. -/
theorem pctCmp_total (x y z : Option Nat) :
    pctCmp x x = .eq ∧ pctCmp y x = (pctCmp x y).swap ∧
    (pctCmp x y ≠ .gt → pctCmp y z ≠ .gt → pctCmp x z ≠ .gt) := by
  cases x <;> cases y <;> cases z <;> simp [pctCmp, Nat.compare_eq_gt, Nat.compare_swap] <;> omega

end FalkorAggOps
