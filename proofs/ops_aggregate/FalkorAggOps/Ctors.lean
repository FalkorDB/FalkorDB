import FalkorAggOps.Ops
/-
Operator constructors (`new`) of Limit, Skip, Distinct, Project, Unwind, Sort
and Commit. Each builds a record from its arguments; the fields the operator's
`next` reads (`remaining`, `remaining_skip`, the dedup set, the pack ceiling,
the sort cursor, the RO gate) are what these theorems pin down, and each is
tied to the `next` model in Ops.lean where there is one.

Borrowed fields (`runtime`, `child`, trees, `idx`) are opaque type parameters:
the constructors only move them into the struct.
-/

namespace FalkorAggOps

/-! ## LimitOp::new (ops/limit.rs:32) -/

structure LimitSt (R C I : Type) where
  runtime : R
  child : C
  remaining : Nat
  idx : I

/-- limit.rs:32-44: `remaining: limit`. -/
def LimitSt.new {R C I : Type} (runtime : R) (child : C) (limit : Nat) (idx : I) : LimitSt R C I :=
  { runtime, child, remaining := limit, idx }

/-- The constructed state has `remaining = limit`, so running `next` to exhaustion
from it yields exactly `take limit` of the input rows (`limitOp_spec`). -/
theorem limit_new_spec {R C I α : Type} (r : R) (c : C) (k : Nat) (i : I) (bs : List (List α)) :
    let s := LimitSt.new r c k i
    s.remaining = k ∧ s.runtime = r ∧ s.child = c ∧ s.idx = i ∧
    (limitOp s.remaining bs).flatten = bs.flatten.take k :=
  ⟨rfl, rfl, rfl, rfl, limitOp_spec k bs⟩

/-! ## SkipOp::new (ops/skip.rs:32) -/

structure SkipSt (R C I : Type) where
  runtime : R
  child : C
  remainingSkip : Nat
  idx : I

/-- skip.rs:32-44: `remaining_skip: skip`. -/
def SkipSt.new {R C I : Type} (runtime : R) (child : C) (skip : Nat) (idx : I) : SkipSt R C I :=
  { runtime, child, remainingSkip := skip, idx }

theorem skip_new_spec {R C I α : Type} (r : R) (c : C) (k : Nat) (i : I) (bs : List (List α)) :
    let s := SkipSt.new r c k i
    s.remainingSkip = k ∧ s.runtime = r ∧ s.child = c ∧ s.idx = i ∧
    (skipOp s.remainingSkip bs).flatten = bs.flatten.drop k :=
  ⟨rfl, rfl, rfl, rfl, skipOp_spec k bs⟩

/-! ## DistinctOp::new (ops/distinct.rs:42) -/

/-- `ValuesDeduper::with_capacity(1024)`: capacity is a sizing hint only; the
seen-set starts empty. -/
structure DistinctSt (R C I H : Type) where
  runtime : R
  child : C
  seen : List H
  capacity : Nat
  idx : I

def DistinctSt.new {R C I H : Type} (runtime : R) (child : C) (idx : I) : DistinctSt R C I H :=
  { runtime, child, seen := [], capacity := 1024, idx }

/-- The dedup set starts empty, so `next` from a fresh op is `dedupBy h []` —
with an injective hash, exact first-occurrence dedup (`dedup_injective`). -/
theorem distinct_new_spec {R C I α H : Type} [DecidableEq α] [DecidableEq H] (r : R) (c : C) (i : I)
    (h : α → H) (inj : ∀ a b, h a = h b → a = b) (xs : List α) :
    let s : DistinctSt R C I H := DistinctSt.new r c i
    s.seen = [] ∧ s.capacity = 1024 ∧ s.runtime = r ∧ s.child = c ∧ s.idx = i ∧
    dedupBy h s.seen xs = dedupBy id [] xs := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_⟩
  show dedupBy h [] xs = dedupBy id [] xs
  simpa using dedup_injective h inj [] xs

/-! ## ProjectOp::new (ops/project.rs:33) -/

structure ProjectSt (R C T P I : Type) where
  runtime : R
  child : C
  trees : T
  copyFromParent : P
  idx : I

def ProjectSt.new {R C T P I : Type} (runtime : R) (child : C) (trees : T) (cfp : P) (idx : I) :
    ProjectSt R C T P I :=
  { runtime, child, trees, copyFromParent := cfp, idx }

theorem project_new_spec {R C T P I : Type} (r : R) (c : C) (t : T) (p : P) (i : I) :
    ProjectSt.new r c t p i = ⟨r, c, t, p, i⟩ := rfl

/-! ## UnwindOp::new (ops/unwind.rs:64) and `BatchedResultEmitter::with_binding`
(ops/batched_result_emitter.rs:553) -/

/-- `BATCH_SIZE` (batch.rs:81). -/
def BATCH_SIZE : Nat := 1024

/-- batched_result_emitter.rs:557-560. -/
def packCeiling (recordCap : Option Nat) : Nat :=
  match recordCap with
  | some cap => if cap < BATCH_SIZE then max cap 1 else BATCH_SIZE
  | none => BATCH_SIZE

theorem packCeiling_bounds (cap : Option Nat) :
    1 ≤ packCeiling cap ∧ packCeiling cap ≤ BATCH_SIZE := by
  unfold packCeiling BATCH_SIZE
  cases cap with
  | none => simp
  | some k =>
    simp only
    by_cases h : k < 1024
    · simp only [h, if_true]; omega
    · simp only [h, if_false]; omega

structure EmitterSt (B : Type) where
  binding : B
  batch : Option Unit       -- no parent batch installed
  pending : Option Unit     -- no pending iterator
  cursor : Nat
  packCeiling : Nat

structure UnwindSt (R C L I B : Type) where
  runtime : R
  child : C
  list : L
  emitter : EmitterSt B
  idx : I

def UnwindSt.new {R C L I B : Type} (runtime : R) (child : C) (list : L) (nameId : B)
    (recordCap : Option Nat) (idx : I) : UnwindSt R C L I B :=
  { runtime, child, list,
    emitter := { binding := nameId, batch := none, pending := none, cursor := 0,
                 packCeiling := packCeiling recordCap },
    idx }

/-- The emitter starts empty (no batch, no pending iterator, cursor 0), bound
to the UNWIND variable, with a pack ceiling in `[1, BATCH_SIZE]` that equals the
downstream row budget when it is below a batch (clamped to 1 for `LIMIT 0`). -/
theorem unwind_new_spec {R C L I B : Type} (r : R) (c : C) (l : L) (b : B) (cap : Option Nat) (i : I) :
    let s := UnwindSt.new r c l b cap i
    s.emitter.binding = b ∧ s.emitter.batch = none ∧ s.emitter.pending = none ∧
    s.emitter.cursor = 0 ∧ 1 ≤ s.emitter.packCeiling ∧ s.emitter.packCeiling ≤ BATCH_SIZE ∧
    (∀ k, 0 < k → k < BATCH_SIZE → cap = some k → s.emitter.packCeiling = k) ∧
    (cap = none → s.emitter.packCeiling = BATCH_SIZE) ∧
    s.runtime = r ∧ s.child = c ∧ s.list = l ∧ s.idx = i := by
  refine ⟨rfl, rfl, rfl, rfl, (packCeiling_bounds cap).1, (packCeiling_bounds cap).2, ?_, ?_,
    rfl, rfl, rfl, rfl⟩
  · intro k hk hlt e; subst e; simp [UnwindSt.new, packCeiling, hlt]; omega
  · intro e; subst e; rfl

/-! ## SortOp::new (ops/sort.rs:249) -/

structure SortSt (R C T I Bt : Type) where
  runtime : R
  child : Option C
  trees : T
  sorted : Option Bt
  order : List Nat
  pos : Nat
  idx : I
  limit : Option Nat
  skip : Nat

def SortSt.new {R C T I Bt : Type} (runtime : R) (child : C) (trees : T) (idx : I)
    (limit : Option Nat) (skip : Nat) : SortSt R C T I Bt :=
  { runtime, child := some child, trees, sorted := none, order := [], pos := 0, idx, limit, skip }

/-- Not yet consumed (`child = Some`, so the first `next` builds the sort),
nothing buffered, cursor at 0; `limit`/`skip` are kept for the `k + skip` cap of
`next` (`sort_skip_limit`). -/
theorem sort_new_spec {R C T I Bt : Type} (r : R) (c : C) (t : T) (i : I) (lim : Option Nat) (s : Nat) :
    let st : SortSt R C T I Bt := SortSt.new r c t i lim s
    st.child = some c ∧ st.sorted = none ∧ st.order = [] ∧ st.pos = 0 ∧
    st.limit = lim ∧ st.skip = s ∧ st.runtime = r ∧ st.trees = t ∧ st.idx = i :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-! ## CommitOp::new (ops/commit.rs:45) -/

structure CommitSt (R C I Bt : Type) where
  runtime : R
  child : Option C
  results : List Bt
  idx : I
  isRoot : Bool

/-- commit.rs:45-66: refuse a read-only runtime, otherwise record whether the
op is the plan root (`plan.node(idx).parent().is_none()`). -/
def CommitSt.new {R C I Bt : Type} (runtime : R) (write : Bool) (hasParent : Bool) (child : C)
    (idx : I) : Except String (CommitSt R C I Bt) :=
  if !write then .error "graph.RO_QUERY is to be executed only on read-only queries"
  else .ok { runtime, child := some child, results := [], idx, isRoot := !hasParent }

theorem commit_new_ro {R C I Bt : Type} (r : R) (p : Bool) (c : C) (i : I) :
    (CommitSt.new r false p c i : Except String (CommitSt R C I Bt)) =
      .error "graph.RO_QUERY is to be executed only on read-only queries" := rfl

theorem commit_new_write {R C I Bt : Type} (r : R) (p : Bool) (c : C) (i : I) :
    (CommitSt.new r true p c i : Except String (CommitSt R C I Bt)) =
      .ok { runtime := r, child := some c, results := [], idx := i, isRoot := !p } := rfl

end FalkorAggOps
