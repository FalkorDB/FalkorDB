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
pre-inserts the single empty-key group (aggregate.rs:751-759, :1111-1119), so it always
emits one row of initial accumulators (`count`=0, `sum`=0.0, `avg`=null,
`collect`=[] …; checked in `agrees_empty_input_aggregates`). -/
theorem keyed_empty {K A : Type} [DecidableEq K] : groupAll ([] : List (K × A)) = [] := rfl

def keylessGroups {A : Type} (rows : List A) : List (Unit × List A) :=
  [((), rows)]

theorem keyless_one_row {A : Type} (rows : List A) : (keylessGroups rows).length = 1 := rfl

/-! ### Per-row grouping with implicit keys (#3579; aggregate.rs:1104-1252)

`consume_input_per_row` pre-inserts the empty-key group only when there are neither explicit
nor implicit keys (:1113); `consume_batch_per_row` groups by the explicit key values followed by
the implicit key values (:1166-1188; a slotted key's error fails the row, an unslotted one under a
`CASE` groups as `null`, `implicitVal`), and seeds a new group's accumulator row with each slotted
key's grouped value at its slot, then the first row's values of the unslotted keys' variables
(`lazy_vars`, :1207-1228, `seedNew`). -/

/-- `consume_input_per_row` + `consume_batch_per_row` grouping: `keyless` is
`keys.is_empty() && implicit.is_empty()`. -/
def perRowGroups {V A : Type} [DecidableEq V] (keyless : Bool) (rows : List (List V × A)) :
    List (List V × List A) :=
  rows.foldl (fun g r => groupInsert r.1 r.2 g) (if keyless then [([], [])] else [])

/-- The per-row group key: explicit key values, then implicit key values (:1166-1188). -/
def rowKey {V : Type} (explicit implicit : List V) : List V := explicit ++ implicit

/-- With keys, every group still holds exactly the rows with its key (bag semantics). -/
theorem perRowGroups_spec {V A : Type} [DecidableEq V] (rows : List (List V × A)) (k : List V) :
    perRowGroups false rows = groupAll rows ∧
    (lookupG k (perRowGroups false rows)).getD [] = (rows.filter (fun r => r.1 = k)).map Prod.snd :=
  ⟨rfl, groupAll_spec rows k⟩

/-- Without explicit or implicit keys every row has the empty key: exactly one group, holding all
rows (one output row even for an empty input). -/
theorem perRowGroups_keyless {V A : Type} [DecidableEq V] (rows : List (List V × A))
    (h : ∀ r ∈ rows, r.1 = []) : perRowGroups true rows = [([], rows.map Prod.snd)] := by
  unfold perRowGroups
  simp only [if_true]
  suffices ∀ (acc : List A), rows.foldl (fun g r => groupInsert r.1 r.2 g) [([], acc)] =
      [([], acc ++ rows.map Prod.snd)] by simpa using this []
  induction rows with
  | nil => intro acc; simp
  | cons r rows ih =>
    intro acc
    have hr : r.1 = [] := h r (by simp)
    simp only [List.foldl, groupInsert, hr, if_true]
    rw [ih (fun r' hr' => h r' (by simp [hr']))]
    simp

/-- **Implicit keys are keys (#3571)**: with an implicit key and no input row there is no group,
so `MATCH (n) RETURN n.v + count(*)` on an empty graph emits no row (it emitted one before). -/
theorem implicit_empty_input_no_group {V A : Type} [DecidableEq V] :
    perRowGroups false ([] : List (List V × A)) = [] := rfl

/-- Rows of one group agree on every implicit key value: the explicit parts have one length
(one value per key), so equal group keys have equal implicit parts. -/
theorem rowKey_implicit_eq {V : Type} (e₁ e₂ i₁ i₂ : List V) (hl : e₁.length = e₂.length)
    (h : rowKey e₁ i₁ = rowKey e₂ i₂) : e₁ = e₂ ∧ i₁ = i₂ :=
  List.append_inj h hl

/-- The lazy-variable step of a new group's accumulator row: each variable of `lazy_vars` bound in
the first row is written (:1222-1226). -/
def seedAcc {V : Type} (vars : List Nat) (row acc : Nat → Option V) : Nat → Option V :=
  vars.foldl (fun a v => match row v with
    | some x => fun j => if j = v then some x else a j
    | none => a) acc

/-- Slots that are not implicit variables keep the default accumulator. -/
theorem seedAcc_other {V : Type} (vars : List Nat) (row acc : Nat → Option V) (j : Nat)
    (hj : j ∉ vars) : seedAcc vars row acc j = acc j := by
  induction vars generalizing acc with
  | nil => rfl
  | cons v vs ih =>
    simp only [seedAcc, List.foldl] at ih ⊢
    have hv : j ≠ v := fun h => hj (by simp [h])
    rw [ih _ (fun h => hj (by simp [h]))]
    cases row v <;> simp [hv]

/-- An implicit variable bound in the first row holds that row's value. -/
theorem seedAcc_var {V : Type} (vars : List Nat) (row acc : Nat → Option V) (j : Nat) (x : V)
    (hj : j ∈ vars) (hx : row j = some x) : seedAcc vars row acc j = some x := by
  induction vars generalizing acc with
  | nil => simp at hj
  | cons v vs ih =>
    simp only [seedAcc, List.foldl] at ih ⊢
    by_cases hin : j ∈ vs
    · exact ih _ hin
    · have hv : j = v := by simpa [hin] using hj
      subst hv
      have := seedAcc_other vs row (fun i => if i = j then some x else acc i) j hin
      simp only [seedAcc] at this
      rw [hx]; simp [this]

/-- One implicit key's group value (aggregate.rs:1177-1187): a slotted key (`slotted = true`)
propagates its evaluation error (`value?`, the row fails); an unslotted one, under a `CASE`, reads an
error as `null` (`value.unwrap_or(Value::Null)`). -/
def implicitVal {V E : Type} (null : V) (slotted : Bool) (r : Except E V) : Except E V :=
  if slotted then r else match r with
    | .ok v => .ok v
    | .error _ => .ok null

theorem implicitVal_ok {V E : Type} (null : V) (slotted : Bool) (v : V) :
    implicitVal null slotted (.ok v : Except E V) = .ok v := by
  cases slotted <;> rfl

theorem implicitVal_slotted_err {V E : Type} (null : V) (e : E) :
    implicitVal null true (.error e : Except E V) = .error e := rfl

/-- **An implicit key under a `CASE` never fails grouping (#3579)**: its error groups as `null`. -/
theorem implicitVal_guarded_err {V E : Type} (null : V) (e : E) :
    implicitVal null false (.error e : Except E V) = .ok null := rfl

/-- The slot step (aggregate.rs:1213-1217): each implicit key with a slot writes the value the row
grouped by (the matching implicit part of the group key) at its slot. -/
def seedSlots {V : Type} : List (Option Nat) → List V → (Nat → Option V) → Nat → Option V
  | some j :: ss, x :: xs, acc => seedSlots ss xs (fun i => if i = j then some x else acc i)
  | none :: ss, _ :: xs, acc => seedSlots ss xs acc
  | _, _, acc => acc

/-- A new group's accumulator row (aggregate.rs:1209-1227): `default_acc`, the slotted keys'
grouped values at their slots, then the lazy variables from the first row. -/
def seedNew {V : Type} (slots : List (Option Nat)) (ivals : List V) (lazy : List Nat)
    (row acc : Nat → Option V) : Nat → Option V :=
  seedAcc lazy row (seedSlots slots ivals acc)

theorem seedSlots_other {V : Type} (slots : List (Option Nat)) (ivals : List V)
    (acc : Nat → Option V) (j : Nat) (hj : some j ∉ slots) : seedSlots slots ivals acc j = acc j := by
  induction slots generalizing ivals acc with
  | nil => cases ivals <;> rfl
  | cons s ss ih =>
    cases ivals with
    | nil => cases s <;> rfl
    | cons x xs =>
      have hss : some j ∉ ss := fun h => hj (List.mem_cons_of_mem _ h)
      cases s with
      | none => exact ih xs acc hss
      | some k =>
        have hk : j ≠ k := fun h => hj (by simp [h])
        simp only [seedSlots]; rw [ih xs _ hss]; simp [hk]

/-- **A slotted key reports the value it grouped by (#3579)**: with distinct slots, a slot holds
the grouped value of its key (the item then reads that slot, so a volatile subtree such as `rand()`
is not drawn again at finalize). -/
theorem seedSlots_slot {V : Type} (slots : List (Option Nat)) (ivals : List V)
    (acc : Nat → Option V) (n j : Nat) (x : V) (hnd : (slots.filterMap id).Nodup)
    (hs : slots[n]? = some (some j)) (hx : ivals[n]? = some x) :
    seedSlots slots ivals acc j = some x := by
  induction slots generalizing ivals acc n with
  | nil => simp at hs
  | cons s ss ih =>
    cases ivals with
    | nil => simp at hx
    | cons y ys =>
      cases n with
      | zero =>
        simp at hs hx; subst hs; subst hx
        have hnot : some j ∉ ss := by
          intro hm
          have : j ∈ ss.filterMap id := List.mem_filterMap.mpr ⟨some j, hm, rfl⟩
          simp at hnd; exact hnd.1 (by simpa using this)
        simp only [seedSlots]; rw [seedSlots_other _ _ _ _ hnot]; simp
      | succ m =>
        simp at hs hx
        cases s with
        | none => exact ih ys acc m (by simpa using hnd) hs hx
        | some k =>
          simp only [seedSlots]
          exact ih ys _ m (by simp at hnd; exact hnd.2) hs hx

/-- With slots above every lazy variable's id (aggregate.rs:350-357: slot ids exceed every id the
key and aggregation trees read), the lazy step does not touch a slot. -/
theorem seedNew_slot {V : Type} (slots : List (Option Nat)) (ivals : List V) (lazy : List Nat)
    (row acc : Nat → Option V) (n j : Nat) (x : V) (hnd : (slots.filterMap id).Nodup)
    (hs : slots[n]? = some (some j)) (hx : ivals[n]? = some x) (hj : j ∉ lazy) :
    seedNew slots ivals lazy row acc j = some x := by
  unfold seedNew; rw [seedAcc_other _ _ _ _ hj]; exact seedSlots_slot _ _ _ _ _ _ hnd hs hx

theorem seedNew_var {V : Type} (slots : List (Option Nat)) (ivals : List V) (lazy : List Nat)
    (row acc : Nat → Option V) (j : Nat) (x : V) (hj : j ∈ lazy) (hx : row j = some x) :
    seedNew slots ivals lazy row acc j = some x :=
  seedAcc_var _ _ _ _ _ hj hx

/-- Every other slot (the accumulators) keeps `default_acc`. -/
theorem seedNew_other {V : Type} (slots : List (Option Nat)) (ivals : List V) (lazy : List Nat)
    (row acc : Nat → Option V) (j : Nat) (hl : j ∉ lazy) (hs : some j ∉ slots) :
    seedNew slots ivals lazy row acc j = acc j := by
  unfold seedNew; rw [seedAcc_other _ _ _ _ hl, seedSlots_other _ _ _ _ hs]

/-! ## DISTINCT state keyed by a hash (history: distinct.rs:79, aggregate.rs:724/:968 at ca36bfa15)

#2977 (c652e505e) keys the per-group state by a group id instead; see `GroupId.lean`
(`distinct_by_group_id_exact`, `fixed2977_collision_example`). The definitions below are the
pre-#2977 model (its counterexample was removed in bbee75955; the last revision that has them is 17ccc5412). -/

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

/-- …and with a collision it drops a row (`#2780`; the grouped-aggregate
variant below is new). -/
theorem dedup_collision_drops :
    dedupBy (fun n : Nat => n % 2) [] [0, 2] = [0] ∧ dedupBy id [] [0, 2] = [0, 2] := by
  decide

/-- Per-group DISTINCT, pre-#2977: the dedup table was keyed by `(distinct node,
hash(group key row))`, not by the group key (aggregate.rs:724, :968 at ca36bfa15). Two
groups whose key rows collide shared one `seen` set. -/
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

/-! ## eval.rs:886-893 — an aggregate's finalized value is pushed (#2951)

Re-targeted to 289c201fe. Before #2951 (0912f4500, closes #2945, FINDINGS
W2-agg-5) the `FuncInvocation` arm of `ExprEval::eval_compound` ran
`return match finalize { … }` (eval.rs:877 at fb3d516d0), leaving the whole
stack machine with that one value; that model was removed (in 17ccc5412). On main the arm pushes the finalized value and
`continue`s (eval.rs:883-893 at 289c201fe), so any enclosing frame, e.g. the
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
aggregate's finalized accumulator, which main pushes (eval.rs:889-892 at
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

/-- Main's stack machine (eval.rs:883-893 at 289c201fe): the aggregate arm
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

/-! ### History: before #2951 (eval.rs:877 at fb3d516d0) -/

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

/-- History (e8f8a3017, fixed by #2912 b033aecc6): the percentile sort comparator
`partial_cmp(..).unwrap_or(Equal)` (then aggregation.rs:615, :645) with NaN (modelled as
`none`) was not transitive: 1 ~ NaN ~ 3 but 1 < 3. std's sort panicked on it (and killed a
live server; `lean_ops_aggregate::bug_percentile_with_nan_panics`). -/
def pcmpEq : Option Nat → Option Nat → Ordering
  | some a, some b => compare a b
  | _, _ => .eq

/-- Main (aggregation.rs:625, :651): `sort_by(|a, b| sort_cmp_f64(a.get_numeric(),
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
