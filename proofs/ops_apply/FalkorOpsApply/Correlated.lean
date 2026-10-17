/-
# Correlated sub-plans: Apply, Optional, SemiApply, AntiSemiApply, OR-apply

All five operators share one idiom (Rust `graph/src/runtime/ops/`):

* the outer batch's active rows are compacted and tagged with sequential
  origins `0..n` (`Batch::clone_active_rows_seq_origin`, `batch.rs:1163`),
* ONE instance of the right sub-plan runs over that whole multi-row argument
  batch (`run_batch` + `set_argument_batch`), and
* every sub-plan output row is correlated back to its input row by its
  `origin_row` tag (`merge_over_input`, `batch.rs:944`; `matched[o]`,
  `HashSet<u32>` of origins).

The reference semantics (openCypher, and what the C engine does with its
per-record `Argument` tap) is *per row*: run the sub-plan on each input row alone.

The batched idiom is only equal to the per-row reference when the sub-plan is
**separable**: running it on a tagged batch is the same as running it on each row
alone and re-tagging the results.  `Separable` below is exactly that condition,
and the headline theorems are

* `applyBatched_eq_ref`       — batched Apply  = per-row Apply (same list)
* `optionalBatched_perm_ref`  — batched Optional is a permutation of per-row Optional
* `semiBatched_eq_ref`        — SemiApply / AntiSemiApply = per-row EXISTS filter
* `orApply_eq_ref`            — OR-apply multiplexer = per-row disjunction
* `applyPerRow_eq_ref`, `optionalPerRow_eq_ref` — the per-row fallback modes are
  the reference for *every* sub-plan (no separability needed).

The planner guards batching with a *syntactic* blacklist (`apply.rs:119`,
`optional.rs:104`).  `vhj_not_separable` + `canBatchApply_admits_vhj` show the
blacklist misses `IR::ValueHashJoin`, whose runtime operator
(`value_hash_join.rs:376-446`) joins left and right rows without comparing
origins; `vhj_batched_16_vs_8` is the concrete counterexample, reproduced live
(Rust 16 rows, C 8) with
`UNWIND [1,2] AS x MATCH (a:A),(b:B) WHERE a.v = b.v RETURN count(*)`.
-/

namespace Correlated

/-- A sub-plan: consumes an argument batch of `(origin, row)` pairs and produces
`(origin, output)` pairs.  Abstracts `run_batch(idx)` + `set_argument_batch` +
draining the iterator (`BatchOp`, `batch.rs:1499`). -/
abbrev SubPlan (α β : Type) := List (Nat × α) → List (Nat × β)

/-- Sequential origin tagging: `clone_active_rows_seq_origin` (`batch.rs:1163`),
row `i` of the compacted active rows gets origin `i`. (The Rust code omits the
sidecar when `n ≤ 1`, which `origin_row` (`batch.rs:1298`) reads back as `0` —
identical tags.) -/
def tag {α : Type} : List α → Nat → List (Nat × α)
  | [], _ => []
  | x :: xs, n => (n, x) :: tag xs (n + 1)

/-- Running the sub-plan on one row alone (per-row mode builds a single-row
argument batch, `apply.rs:303-305`, `optional.rs:252-254`). -/
def single {α β : Type} (f : SubPlan α β) (x : α) : List β :=
  (f [(0, x)]).map Prod.snd

/-- **Separable** sub-plan: its result on a tagged batch is the concatenation, in
argument order, of its results on each row alone, each re-tagged with that row's
origin.  Scans, traversals, filters, projections, `Unwind`, `PathBuilder`,
`SemiApply` over separable plans, … all satisfy it; `Aggregate`, `Sort`, `Limit`,
`Skip`, `Distinct`, `CartesianProduct` and `ValueHashJoin` do not. -/
def Separable {α β : Type} (f : SubPlan α β) : Prop :=
  ∀ args : List (Nat × α),
    f args = args.flatMap (fun p => (single f p.2).map (fun y => (p.1, y)))

theorem tag_length {α : Type} (xs : List α) (n : Nat) : (tag xs n).length = xs.length := by
  induction xs generalizing n with
  | nil => rfl
  | cons x xs ih => simp [tag, ih]

theorem mem_tag {α : Type} {xs : List α} {n i : Nat} {x : α} :
    (i, x) ∈ tag xs n → n ≤ i ∧ i < n + xs.length := by
  induction xs generalizing n with
  | nil => simp [tag]
  | cons y ys ih =>
    intro h
    simp [tag] at h
    rcases h with ⟨rfl, rfl⟩ | h
    · simp
    · have := ih h; simp at this ⊢; omega

theorem tag_map_snd {α : Type} (xs : List α) (n : Nat) : (tag xs n).map Prod.snd = xs := by
  induction xs generalizing n with
  | nil => rfl
  | cons x xs ih => simp [tag, ih]

/-! ## Apply (batched and per-row) — `graph/src/runtime/ops/apply.rs` -/

/-- Batched Apply, flattened over output batches (`next_batched`, `apply.rs:152-232`):
the sub-plan runs once over `tag xs 0`; each output `(o, y)` is merged against
input row `o` (`merge_over_input`, `batch.rs:944`, the columnar equivalent of
`input_row.clone().merge(sub_row)`).  `gather` indexes by origin; an out-of-range
origin would panic in Rust, modelled by dropping (it cannot happen for separable
plans, `mem_tag`). -/
def applyBatched {α β γ : Type} (merge : α → β → γ) (f : SubPlan α β) (xs : List α) : List γ :=
  (f (tag xs 0)).filterMap (fun p => xs[p.1]?.map (fun x => merge x p.2))

/-- Per-row Apply (`next_per_row`, `apply.rs:238-337`): one fresh sub-plan per
input row, results merged with `p.env.clone().merge(&row)` (`apply.rs:251-253`). -/
def applyPerRow {α β γ : Type} (merge : α → β → γ) (f : SubPlan α β) (xs : List α) : List γ :=
  xs.flatMap (fun x => (single f x).map (merge x))

/-- Reference (openCypher / C per-record Argument): for each input row, every
sub-plan result merged with that row, in input order. -/
def applyRef {α β γ : Type} (merge : α → β → γ) (f : SubPlan α β) (xs : List α) : List γ :=
  xs.flatMap (fun x => (single f x).map (merge x))

theorem applyPerRow_eq_ref {α β γ : Type} (merge : α → β → γ) (f : SubPlan α β) (xs : List α) :
    applyPerRow merge f xs = applyRef merge f xs := rfl

/-- Key lemma: looking the origin back up in the (suffix-extended) input recovers
exactly the row the result came from. -/
theorem lookup_tagged {α β γ : Type} (merge : α → β → γ) (g : α → List β) :
    ∀ (pre ys : List α),
      ((tag ys pre.length).flatMap (fun p => (g p.2).map (fun y => (p.1, y)))).filterMap
          (fun p => (pre ++ ys)[p.1]?.map (fun x => merge x p.2))
        = ys.flatMap (fun x => (g x).map (merge x))
  | _, [] => by simp [tag]
  | pre, y :: ys => by
    have ih := lookup_tagged merge g (pre ++ [y]) ys
    simp only [List.length_append, List.length_singleton, List.append_assoc,
      List.singleton_append] at ih
    simp only [tag, List.flatMap_cons, List.filterMap_append, ih]
    congr 1
    simp [List.filterMap_map, Function.comp_def]

theorem applyBatched_eq_ref {α β γ : Type} (merge : α → β → γ) (f : SubPlan α β)
    (hf : Separable f) (xs : List α) :
    applyBatched merge f xs = applyRef merge f xs := by
  unfold applyBatched applyRef
  rw [hf]
  have := lookup_tagged merge (single f) [] xs
  simpa using this

/-! ## Optional — `graph/src/runtime/ops/optional.rs` and the Optional arm of Apply -/

/-- Origins that produced at least one sub-plan row (`matched[o] = true`,
`optional.rs:164-166`, `apply.rs:201-203`). -/
def matchedOrigins {α β : Type} (f : SubPlan α β) (xs : List α) : List Nat :=
  (f (tag xs 0)).map Prod.fst

/-- Batched Optional, flattened (`optional.rs:125-189`): every sub-plan batch is
emitted merged over the input as it arrives; only once the sub-plan is exhausted
is ONE fallback batch emitted for all unmatched origins, gathered from
`input_ref` in origin order with the optional variables set to NULL
(`optional.rs:175-184`). -/
def optionalBatched {α β γ : Type} (merge : α → β → γ) (nullFill : α → γ)
    (f : SubPlan α β) (xs : List α) : List γ :=
  applyBatched merge f xs ++
    ((tag xs 0).filter (fun p => !(matchedOrigins f xs).contains p.1)).map (fun p => nullFill p.2)

/-- Per-row Optional (`optional.rs:195-286`): per row, the sub-plan results, or
the NULL-filled row if there were none (`had_result`). -/
def optionalPerRow {α β γ : Type} (merge : α → β → γ) (nullFill : α → γ)
    (f : SubPlan α β) (xs : List α) : List γ :=
  xs.flatMap (fun x => match single f x with
    | [] => [nullFill x]
    | ys => ys.map (merge x))

/-- Reference OPTIONAL MATCH: per row, the matches, or one NULL-padded row. -/
def optionalRef {α β γ : Type} (merge : α → β → γ) (nullFill : α → γ)
    (f : SubPlan α β) (xs : List α) : List γ :=
  xs.flatMap (fun x => if single f x = [] then [nullFill x] else (single f x).map (merge x))

theorem optionalPerRow_eq_ref {α β γ : Type} (merge : α → β → γ) (nullFill : α → γ)
    (f : SubPlan α β) (xs : List α) :
    optionalPerRow merge nullFill f xs = optionalRef merge nullFill f xs := by
  unfold optionalPerRow optionalRef
  congr 1; funext x
  cases h : single f x <;> simp

theorem mem_fst_flatMap {α β : Type} (g : α → List β) (ps : List (Nat × α)) (i : Nat) :
    i ∈ (ps.flatMap (fun p => (g p.2).map (fun y => (p.1, y)))).map Prod.fst ↔
      ∃ x, (i, x) ∈ ps ∧ g x ≠ [] := by
  simp only [List.mem_map, List.mem_flatMap]
  constructor
  · rintro ⟨_, ⟨p, hp, y, hy, rfl⟩, rfl⟩
    exact ⟨p.2, by simpa using hp, List.ne_nil_of_mem hy⟩
  · rintro ⟨x, hx, hg⟩
    obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hg
    exact ⟨(i, y), ⟨(i, x), hx, y, hy, rfl⟩, rfl⟩

/-- Tags are unique: one origin, one row. -/
theorem tag_unique {α : Type} {xs : List α} {n i : Nat} {x x' : α} :
    (i, x) ∈ tag xs n → (i, x') ∈ tag xs n → x = x' := by
  induction xs generalizing n with
  | nil => simp [tag]
  | cons y ys ih =>
    intro h h'
    simp only [tag, List.mem_cons, Prod.mk.injEq] at h h'
    rcases h with ⟨rfl, rfl⟩ | h <;> rcases h' with ⟨h1, rfl⟩ | h'
    · rfl
    · have := (mem_tag h').1; omega
    · have := (mem_tag h).1; omega
    · exact ih h h'

theorem contains_iff_generic {α β : Type} (g : α → List β) (ps : List (Nat × α)) (i : Nat) (x : α)
    (hx : (i, x) ∈ ps) (huniq : ∀ x', (i, x') ∈ ps → x' = x) :
    ((ps.flatMap (fun p => (g p.2).map (fun y => (p.1, y)))).map Prod.fst).contains i
      = !(g x).isEmpty := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.contains_iff_mem, mem_fst_flatMap]
  constructor
  · rintro ⟨x', hx', hg⟩
    rw [huniq x' hx'] at hg
    cases h : g x <;> simp_all
  · intro h
    refine ⟨x, hx, ?_⟩
    cases h' : g x <;> simp_all

/-- For a separable sub-plan, origin `i` is matched iff row `i` alone has results. -/
theorem matched_iff {α β : Type} (f : SubPlan α β) (hf : Separable f)
    (xs : List α) (i : Nat) (x : α) (h : (i, x) ∈ tag xs 0) :
    (matchedOrigins f xs).contains i = !(single f x).isEmpty := by
  unfold matchedOrigins
  rw [hf]
  exact contains_iff_generic (single f) _ i x h (fun x' h' => tag_unique h' h)

/-- The fallback part of `optionalBatched` is the per-row fallback, in input order. -/
theorem fallback_eq {α β γ : Type} (nullFill : α → γ) (f : SubPlan α β) (hf : Separable f)
    (xs : List α) :
    ((tag xs 0).filter (fun p => !(matchedOrigins f xs).contains p.1)).map (fun p => nullFill p.2)
      = xs.flatMap (fun x => if single f x = [] then [nullFill x] else []) := by
  have key : ∀ (ys : List α) (n : Nat), (∀ i x, (i, x) ∈ tag ys n → (i, x) ∈ tag xs 0) →
      ((tag ys n).filter (fun p => !(matchedOrigins f xs).contains p.1)).map
          (fun p => nullFill p.2)
        = ys.flatMap (fun x => if single f x = [] then [nullFill x] else []) := by
    intro ys
    induction ys with
    | nil => intro n _; simp [tag]
    | cons y ys ih =>
      intro n hsub
      simp only [tag, List.filter_cons, List.flatMap_cons]
      have hy := matched_iff f hf xs n y (hsub n y (by simp [tag]))
      have ih' := ih (n + 1) (fun i x h => hsub i x (by simp [tag, h]))
      rw [hy, ← ih']
      cases hs : single f y <;> simp
  exact key xs 0 (fun _ _ h => h)

/-- **Batched Optional is a permutation of the reference**: same rows with the same
multiplicities, but all NULL-fallback rows of an input batch come *after* all
matched rows (C emits them interleaved, in input order).  See
`optionalBatched_order_differs` and the live repro in the report. -/
theorem optionalBatched_perm_ref {α β γ : Type} (merge : α → β → γ) (nullFill : α → γ)
    (f : SubPlan α β) (hf : Separable f) (xs : List α) :
    (optionalBatched merge nullFill f xs).Perm (optionalRef merge nullFill f xs) := by
  unfold optionalBatched
  rw [applyBatched_eq_ref merge f hf, fallback_eq nullFill f hf]
  unfold applyRef optionalRef
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.flatMap_cons]
    by_cases h : single f x = []
    · simp only [h, List.map_nil, List.nil_append, ite_true]
      -- [] ++ A ++ ([n] ++ B) ~ [n] ++ C  where A ++ B ~ C
      exact (List.perm_middle).trans (List.Perm.cons _ ih)
    · simp only [h, ite_false, List.nil_append, List.append_assoc]
      exact List.Perm.append_left _ ih

/-! ## SemiApply / AntiSemiApply — `graph/src/runtime/ops/semi_apply.rs` -/

/-- `SemiApplyOp::next` (`semi_apply.rs:66-113`), one input batch: the sub-plan
runs once over all active rows; a row passes iff `has_result ^ is_anti`
(`semi_apply.rs:101-106`).  (An empty passing set makes the Rust loop pull the
next batch, which is the same as emitting nothing for this one.) -/
def semiBatched {α β : Type} (isAnti : Bool) (f : SubPlan α β) (xs : List α) : List α :=
  ((tag xs 0).filter (fun p => (matchedOrigins f xs).contains p.1 ^^ isAnti)).map Prod.snd

/-- Reference: keep a row iff `EXISTS` (resp. `NOT EXISTS`) the pattern for it. -/
def semiRef {α β : Type} (isAnti : Bool) (f : SubPlan α β) (xs : List α) : List α :=
  xs.filter (fun x => !(single f x).isEmpty ^^ isAnti)

theorem filter_tag_eq {α : Type} (P : Nat → Bool) (Q : α → Bool) :
    ∀ (ys : List α) (n : Nat), (∀ i x, (i, x) ∈ tag ys n → P i = Q x) →
      ((tag ys n).filter (fun p => P p.1)).map Prod.snd = ys.filter Q
  | [], _, _ => by simp [tag]
  | y :: ys, n, h => by
    simp only [tag, List.filter_cons]
    have hy : P n = Q y := h n y (by simp [tag])
    have ih := filter_tag_eq P Q ys (n + 1) (fun i x hm => h i x (by simp [tag, hm]))
    rw [hy]; cases Q y <;> simp [ih]

theorem semiBatched_eq_ref {α β : Type} (isAnti : Bool) (f : SubPlan α β) (hf : Separable f)
    (xs : List α) : semiBatched isAnti f xs = semiRef isAnti f xs := by
  unfold semiBatched semiRef
  exact filter_tag_eq (fun i => (matchedOrigins f xs).contains i ^^ isAnti) _ xs 0
    (fun i x h => by simp only [matched_iff f hf xs i x h])

theorem any_congr_mem {δ : Type} (l : List δ) {p q : δ → Bool} (h : ∀ b ∈ l, p b = q b) :
    l.any p = l.any q := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    simp only [List.any_cons]
    rw [h b (by simp), ih (fun c hc => h c (by simp [hc]))]

/-! ## OR-apply multiplexer — `graph/src/runtime/ops/or_apply_multiplexer.rs` -/

/-- `OrApplyMultiplexerOp::next` (`or_apply_multiplexer.rs:73-129`), one input
batch: every branch runs once over a fresh copy of the tagged rows; origin `i`
passes iff some branch has `has_result ^ anti_flags[b]`. -/
def orApply {α β : Type} (branches : List (SubPlan α β × Bool)) (xs : List α) : List α :=
  ((tag xs 0).filter (fun p =>
      branches.any (fun b => (matchedOrigins b.1 xs).contains p.1 ^^ b.2))).map Prod.snd

def orRef {α β : Type} (branches : List (SubPlan α β × Bool)) (xs : List α) : List α :=
  xs.filter (fun x => branches.any (fun b => !(single b.1 x).isEmpty ^^ b.2))

theorem orApply_eq_ref {α β : Type} (branches : List (SubPlan α β × Bool))
    (hs : ∀ b ∈ branches, Separable b.1) (xs : List α) :
    orApply branches xs = orRef branches xs := by
  unfold orApply orRef
  exact filter_tag_eq (fun i => branches.any (fun b => (matchedOrigins b.1 xs).contains i ^^ b.2))
    _ xs 0 (fun i x h => any_congr_mem branches (fun b hb => by
      simp only [matched_iff b.1 (hs b hb) xs i x h]))

/-- With a single branch the multiplexer is SemiApply. -/
theorem orApply_single {α β : Type} (f : SubPlan α β) (anti : Bool) (hf : Separable f)
    (xs : List α) : orApply [(f, anti)] xs = semiBatched anti f xs := by
  rw [orApply_eq_ref _ (by simpa using hf), semiBatched_eq_ref anti f hf]
  unfold orRef semiRef; simp

/-! ## Separable building blocks -/

/-- A per-row operator (scan, traverse, filter, project, unwind…): output depends
only on the row and keeps its origin.  Every such operator is separable. -/
def rowLocal {α β : Type} (g : α → List β) : SubPlan α β :=
  fun args => args.flatMap (fun p => (g p.2).map (fun y => (p.1, y)))

theorem rowLocal_separable {α β : Type} (g : α → List β) : Separable (rowLocal g) := by
  intro args
  unfold rowLocal single
  simp [List.map_map, Function.comp_def]

/-- Separable plans compose (a pipeline of row-local operators stays row-local). -/
def compose {α β γ : Type} (f : SubPlan α β) (g : SubPlan β γ) : SubPlan α γ := fun a => g (f a)

theorem compose_separable {α β γ : Type} (f : SubPlan α β) (g : SubPlan β γ)
    (hf : Separable f) (hg : Separable g) : Separable (compose f g) := by
  intro args
  unfold compose
  have hsf : ∀ x, single (fun a => g (f a)) x = (single f x).flatMap (single g) := by
    intro x
    show (g (f [(0, x)])).map Prod.snd = _
    rw [hg (f [(0, x)])]
    simp [List.map_flatMap, List.flatMap_map, Function.comp_def, List.map_map, single]
  rw [hg, hf]
  simp only [List.flatMap_assoc, List.flatMap_map, hsf, List.map_flatMap]

/-! ## Non-separable operators and the batching guard -/

/-- `ValueHashJoin` (`value_hash_join.rs:366-446`), both sides fed the same
argument batch (`batch.rs:1678-1689`): the build side is materialised over ALL
argument rows; each probe row is merged with every build row with an equal key —
the origin of the merged row is the probe row's (`Row::merge` does not touch
`origin_row`), and origins are never compared. -/
def vhj {α a b k : Type} [BEq k] (lhs : α → List a) (rhs : α → List b)
    (lk : a → k) (rk : b → k) : SubPlan α (a × b) :=
  fun args =>
    let right := args.flatMap (fun p => (rhs p.2).map (fun y => (p.1, y)))
    (args.flatMap (fun p => (lhs p.2).map (fun y => (p.1, y)))).flatMap
      (fun l => (right.filter (fun r => lk l.2 == rk r.2)).map (fun r => (l.1, (l.2, r.2))))

/-- The live repro's data: two `:A` nodes and two `:B` nodes, all `v = 1`. -/
def scanA (_ : Nat) : List Nat := [1, 2]
def scanB (_ : Nat) : List Nat := [1, 2]
def joinQ : SubPlan Nat (Nat × Nat) := vhj scanA scanB (fun _ => (1 : Nat)) (fun _ => (1 : Nat))

/-- Batched Apply over `UNWIND [1,2] AS x`: 16 rows; per-row reference: 8. -/
theorem vhj_batched_16_vs_8 :
    (applyBatched (fun x y => (x, y)) joinQ [1, 2]).length = 16 ∧
    (applyRef (fun x y => (x, y)) joinQ [1, 2]).length = 8 := by decide

theorem vhj_not_separable : ¬ Separable joinQ := by
  intro h
  have := congrArg List.length (h [(0, 1), (1, 2)])
  revert this; decide

/-- Aggregate (`count(*)` over the argument batch) is not separable either, which
is why Apply/Optional fall back to per-row mode for it. -/
def countAll {α : Type} : SubPlan α Nat := fun args => [(0, args.length)]

theorem count_not_separable : ¬ Separable (countAll (α := Nat)) := by
  intro h
  have := h [(0, 1), (1, 2)]
  revert this; decide

/-- The IR kinds the guards look at (`planner/mod.rs:150-300`). -/
inductive IRKind
  | scan | traverse | filter | project | unwind | pathBuilder | semiApply | orApply
  | aggregate | cartesian | valueHashJoin | optional | apply | merge | union
  | sort | limit | skip | distinct
  deriving DecidableEq, Repr

/-- `ApplyOp::new`'s `can_batch` (`apply.rs:119-133`). -/
def canBatchApply (sub : List IRKind) : Bool :=
  !sub.any (fun k => k ∈ [.aggregate, .cartesian, .optional, .apply, .merge, .union,
                          .sort, .limit, .skip, .distinct])

/-- `OptionalOp::new`'s `can_batch` (`optional.rs:104-106`). -/
def canBatchOptional (sub : List IRKind) : Bool :=
  !sub.any (fun k => k ∈ [.aggregate, .cartesian])

/-- The guard admits a sub-plan containing `ValueHashJoin`, which is not
separable: batched mode is then wrong (`vhj_batched_16_vs_8`). -/
theorem canBatchApply_admits_vhj :
    canBatchApply [.scan, .valueHashJoin, .scan] = true ∧
    canBatchOptional [.scan, .valueHashJoin, .scan] = true := by decide

/-- Optional ordering: rows `[3,1,2]`, pattern matches only 1 and 2 — batched
emits `1,2,3` (fallback last), the reference/C emit `3,1,2`.  Live:
`UNWIND [3,1,2] AS x OPTIONAL MATCH (a:A {w:x}) RETURN collect(x)`. -/
def matchesW (x : Nat) : List Nat := if x ≤ 2 then [x] else []

theorem optionalBatched_order_differs :
    optionalBatched (fun x _ => x) id (rowLocal matchesW) [3, 1, 2] = [1, 2, 3] ∧
    optionalRef (fun x _ => x) id (rowLocal matchesW) [3, 1, 2] = [3, 1, 2] := by decide

end Correlated
