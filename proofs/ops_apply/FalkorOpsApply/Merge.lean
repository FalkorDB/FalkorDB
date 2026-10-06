/-
# MERGE — `graph/src/runtime/ops/merge.rs`

Reference semantics (openCypher): for each input row, in order, if the pattern
matchesK in the *current* graph (including everything earlier rows of the same
clause created), bind every match (and apply `ON MATCH SET`), otherwise create
the pattern once (and apply `ON CREATE SET`).

Rust (`MergeOp::next`, `merge.rs:279-444`): per input batch of up to
`BATCH_SIZE` rows, the match sub-plan runs ONCE over all rows (`merge.rs:313-343`)
against the graph as it is at the start of the batch (the scans see pending
creations of *earlier batches* through `IncludePending`,
`planner/mod.rs:3008`); then rows are processed in order (`merge.rs:347-431`):

* no match → `do_create_fallback` (`merge.rs:113-178`): look the pattern hash up
  in `runtime.merge_pattern_cache` (`runtime.rs:163`); on a hit bind the cached
  `(var id, value)` pairs and apply ON MATCH; on a miss create, cache, apply
  ON CREATE;
* matchesK → first match emitted inline, the rest queued in `pending` and emitted
  after the batch (`drain_pending`, `merge.rs:252-273`) — unless every pattern
  variable is bound (`all_vars_bound`, `merge.rs:368-381`), in which case only
  the first match is used.

The model is one node pattern `MERGE (n:L {k: key(row)})`: a node is `(id, k)`,
the pattern hash is the key (the FxHash collision bug is already known, issue
2913/FINDINGS #6, and is assumed away), `ON CREATE`/`ON MATCH SET n.k = e(row)`
are optional key updates.

Theorems:
* `rust_eq_ref_noSet` — **without SET clauses** Rust's MERGE is the reference
  for every batch size: same final graph, same next id, and the emitted rows are
  a permutation of the reference's (the within-batch cache exactly simulates the
  visibility of earlier rows' creations).
* `batch_dependent` — with `ON CREATE SET` touching the merged property the
  result depends on `BATCH_SIZE`: rows that straddle a batch boundary see the
  earlier row's creation, rows inside one batch do not. CONFIRMED live.
* `cross_clause_unbound` — the cache is keyed by the hash only and shared by all
  MERGE clauses of the query: a second clause whose pattern hashes equal binds
  the FIRST clause's variable id and leaves its own unbound
  ("Variable b not found"). CONFIRMED live.
* `allBound_first_only` — the all-bound shortcut returns one row where the
  reference returns one per match (visible through a path variable). CONFIRMED.
-/

namespace Merge

variable {α : Type}

structure Node where
  id : Nat
  k : Nat
  deriving DecidableEq, Repr

abbrev Graph := List Node

structure Clause (α : Type) where
  /-- the variable id of `n` (binder-assigned; distinct clauses ⇒ distinct ids,
  except UNION branches, which restart numbering) -/
  var : Nat
  key : α → Nat
  onCreate : Option (α → Nat)
  onMatch : Option (α → Nat)

def matchesK (g : Graph) (k : Nat) : List Node := g.filter (fun n => n.k == k)

def setKey (g : Graph) (id v : Nat) : Graph :=
  g.map (fun n => if n.id == id then { n with k := v } else n)

def applySet (s : Option (α → Nat)) (r : α) (g : Graph) (id : Nat) : Graph :=
  match s with
  | none => g
  | some f => setKey g id (f r)

/-! ## Reference (openCypher, per row) -/

structure RState where
  g : Graph
  next : Nat
  out : List (α × Option Nat)

def refRow (cl : Clause α) (st : RState (α := α)) (r : α) : RState (α := α) :=
  match matchesK st.g (cl.key r) with
  | [] =>
    let g := st.g ++ [⟨st.next, cl.key r⟩]
    ⟨applySet cl.onCreate r g st.next, st.next + 1, st.out ++ [(r, some st.next)]⟩
  | ms =>
    ⟨ms.foldl (fun g m => applySet cl.onMatch r g m.id) st.g, st.next,
      st.out ++ ms.map (fun m => (r, some m.id))⟩

def refMerge (cl : Clause α) (st : RState (α := α)) (rows : List α) : RState (α := α) :=
  rows.foldl (refRow cl) st

/-! ## Rust -/

/-- `merge_pattern_cache`: hash ↦ cached `(var id, entity id)` bindings. -/
abbrev Cache := List (Nat × (Nat × Nat))

def lookup (c : Cache) (h : Nat) : Option (Nat × Nat) := (c.find? (fun e => e.1 == h)).map Prod.snd

structure SState where
  g : Graph
  next : Nat
  cache : Cache
  out : List (α × Option Nat)

/-- Binding of the clause's own variable after inserting the cached pairs by
*their* variable ids (`vars.insert_by_id(*id, …)`, `merge.rs:125-127`). -/
def bindCached (cl : Clause α) (cv : Nat × Nat) : Option Nat :=
  if cv.1 == cl.var then some cv.2 else none

/-- `do_create_fallback` (`merge.rs:113-178`). -/
def createFallback (cl : Clause α) (st : SState (α := α)) (r : α) :
    SState (α := α) × (α × Option Nat) :=
  match lookup st.cache (cl.key r) with
  | some cv =>                                                      -- 122-135
    ({ st with g := applySet cl.onMatch r st.g cv.2 }, (r, bindCached cl cv))
  | none =>                                                         -- 136-177
    let g := st.g ++ [⟨st.next, cl.key r⟩]
    ({ g := applySet cl.onCreate r g st.next, next := st.next + 1,
       cache := (cl.key r, (cl.var, st.next)) :: st.cache, out := st.out },
     (r, some st.next))

/-- Phase A of one batch (`merge.rs:347-431`): row by row against the batch-start
match groups. Returns the state (with `out` extended) and the queue of remaining
matchesK. -/
def phaseA (cl : Clause α) (allBound : Bool) :
    SState (α := α) → List (α × List Node) → SState (α := α) × List (α × List Node)
  | st, [] => (st, [])
  | st, (r, []) :: rest =>
    let (st', o) := createFallback cl st r
    phaseA cl allBound { st' with out := st'.out ++ [o] } rest
  | st, (r, m :: ms) :: rest =>
    let st' := { st with g := applySet cl.onMatch r st.g m.id, out := st.out ++ [(r, some m.id)] }
    let (st'', q) := phaseA cl allBound st' rest
    (st'', if allBound || ms.isEmpty then q else (r, ms) :: q)

/-- Phase B (`drain_pending`, `merge.rs:252-273`): the queued matchesK, row by row. -/
def phaseB (cl : Clause α) : SState (α := α) → List (α × List Node) → SState (α := α)
  | st, [] => st
  | st, (r, ms) :: rest =>
    phaseB cl { st with g := ms.foldl (fun g m => applySet cl.onMatch r g m.id) st.g,
                        out := st.out ++ ms.map (fun m => (r, some m.id)) } rest

/-- One input batch: match groups computed ONCE against the batch-start graph
(`merge.rs:313-343`), then phase A, then phase B. -/
def rustBatch (cl : Clause α) (allBound : Bool) (st : SState (α := α)) (rows : List α) :
    SState (α := α) :=
  let groups := rows.map (fun r => (r, matchesK st.g (cl.key r)))
  let (st', q) := phaseA cl allBound st groups
  phaseB cl st' q

/-- The child's batches, of size `B`. -/
def batches (B : Nat) (hB : 0 < B) (l : List α) : List (List α) :=
  if h : l = [] then [] else l.take B :: batches B hB (l.drop B)
termination_by l.length
decreasing_by
  simp only [List.length_drop]
  have : l.length ≠ 0 := by simpa using h
  omega

def rustMerge (cl : Clause α) (allBound : Bool) (st : SState (α := α)) (bs : List (List α)) :
    SState (α := α) :=
  bs.foldl (rustBatch cl allBound) st

/-! ## Without SET clauses, Rust = reference for any batching -/

/-- No ON CREATE / ON MATCH SET on the merged property. -/
def NoSet (cl : Clause α) : Prop := cl.onCreate = none ∧ cl.onMatch = none

theorem applySet_none (r : α) (g : Graph) (id : Nat) : applySet none r g id = g := rfl

theorem foldl_applySet_none (r : α) (ms : List Node) (g : Graph) :
    ms.foldl (fun g m => applySet (none : Option (α → Nat)) r g m.id) g = g := by
  induction ms generalizing g with
  | nil => rfl
  | cons m ms ih => simp only [List.foldl_cons, applySet_none]; exact ih g

theorem foldl_const_self {β γ : Type} (g : β) (ms : List γ) : ms.foldl (fun g _ => g) g = g := by
  induction ms with
  | nil => rfl
  | cons _ _ ih => simpa using ih

theorem matches_append (g h : Graph) (k : Nat) : matchesK (g ++ h) k = matchesK g k ++ matchesK h k := by
  simp [matchesK]

theorem matches_single (id k k' : Nat) : matchesK [⟨id, k'⟩] k = if k' = k then [⟨id, k'⟩] else [] := by
  by_cases h : k' = k <;> simp [matchesK, h]

/-- Cache entries point at nodes of the graph with that key, and carry this
clause's variable. -/
def CacheOK (cl : Clause α) (g : Graph) (c : Cache) : Prop :=
  ∀ h v id, lookup c h = some (v, id) → v = cl.var ∧ ⟨id, h⟩ ∈ g

/-- Within-batch invariant.  `g0` is the batch-start graph; `created` the nodes
created so far in this batch (keys absent from `g0`, at most one per key), and
the cache knows exactly those for keys absent from `g0`. -/
structure J (cl : Clause α) (g0 : Graph) (g : Graph) (c : Cache) : Prop where
  ext : ∃ created, g = g0 ++ created ∧ ∀ n ∈ created, matchesK g0 n.k = []
  cache_ok : CacheOK cl g c
  fresh : ∀ k, matchesK g0 k = [] →
    (lookup c k = none → matchesK g k = []) ∧
    (∀ v id, lookup c k = some (v, id) → matchesK g k = [⟨id, k⟩])

theorem lookup_cons_self (c : Cache) (h v id : Nat) : lookup ((h, (v, id)) :: c) h = some (v, id) := by
  simp [lookup]

theorem lookup_cons_other (c : Cache) (h h' e : Nat) (e' : Nat) (hne : h' ≠ h) :
    lookup ((h, (e, e')) :: c) h' = lookup c h' := by
  have : (h == h') = false := by simp; exact fun e => hne e.symm
  simp [lookup, List.find?_cons, this]

theorem mem_matches {g : Graph} {k : Nat} {n : Node} : n ∈ matchesK g k ↔ n ∈ g ∧ n.k = k := by
  simp [matchesK]

/-- Per-row relation between the Rust phase-A step and the reference step. -/
theorem phaseA_ref (cl : Clause α) (hs : NoSet cl) (g0 : Graph) :
    ∀ (rows : List α) (st : SState (α := α)) (rst : RState (α := α)),
      J cl g0 st.g st.cache → rst.g = st.g → rst.next = st.next →
      (∀ n ∈ st.g, n.id < st.next) →
      let res := phaseA cl false st (rows.map (fun r => (r, matchesK g0 (cl.key r))))
      let ref := rows.foldl (refRow cl) rst
      ref.g = res.1.g ∧ ref.next = res.1.next ∧ J cl g0 res.1.g res.1.cache ∧
      (∀ n ∈ res.1.g, n.id < res.1.next) ∧
      ∃ Ls : List (List (α × Option Nat)),
        res.1.out = st.out ++ Ls.flatMap (List.take 1) ∧
        ref.out = rst.out ++ Ls.flatten ∧
        (res.2.flatMap (fun q => q.2.map (fun m => (q.1, some m.id)))) = Ls.flatMap (List.drop 1) ∧
        -- phase B leaves the graph alone (NoSet)
        True := by
  intro rows
  induction rows with
  | nil =>
    intro st rst hJ hg hn hlt
    simp only [List.map_nil, phaseA, refMerge, List.foldl_nil]
    exact ⟨hg, hn, hJ, hlt, [], by simp, by simp, by simp, trivial⟩
  | cons r rows ih =>
    intro st rst hJ hg hn hlt
    obtain ⟨hoc, hom⟩ := hs
    simp only [List.map_cons, refMerge, List.foldl_cons]
    -- the reference sees the current graph; relate to g0
    obtain ⟨created, hgc, hcr⟩ := hJ.ext
    cases hm0 : matchesK g0 (cl.key r) with
    | nil =>
      -- no batch-start match: cache decides
      simp only [phaseA]
      cases hlk : lookup st.cache (cl.key r) with
      | none =>
        have hcur : matchesK st.g (cl.key r) = [] := (hJ.fresh _ hm0).1 hlk
        simp only [createFallback, hlk]
        have href : refRow cl rst r =
            ⟨rst.g ++ [⟨rst.next, cl.key r⟩], rst.next + 1, rst.out ++ [(r, some rst.next)]⟩ := by
          simp only [refRow, hg, hcur, hoc, applySet]
        rw [href]
        have := ih { g := st.g ++ [⟨st.next, cl.key r⟩], next := st.next + 1,
                     cache := (cl.key r, (cl.var, st.next)) :: st.cache,
                     out := st.out ++ [(r, some st.next)] }
                   ⟨rst.g ++ [⟨rst.next, cl.key r⟩], rst.next + 1, rst.out ++ [(r, some rst.next)]⟩
          ?_ (by simp [hg, hn]) (by simp [hn]) ?_
        · simp only [hoc, applySet] at this ⊢
          obtain ⟨a1, a2, a3, a4, Ls, b1, b2, b3, _⟩ := this
          refine ⟨a1, a2, a3, a4, [(r, some st.next)] :: Ls, ?_, ?_, ?_, trivial⟩
          · rw [b1]; simp
          · rw [b2, hn]; simp
          · rw [b3]; simp
        · -- invariant after creation
          refine ⟨⟨created ++ [⟨st.next, cl.key r⟩], by simp [hgc], ?_⟩, ?_, ?_⟩
          · intro n hn'
            simp only [List.mem_append, List.mem_singleton] at hn'
            rcases hn' with hn' | rfl
            · exact hcr n hn'
            · exact hm0
          · intro h v id hl
            by_cases hh : h = cl.key r
            · subst hh; rw [lookup_cons_self] at hl; cases hl; simp
            · rw [lookup_cons_other _ _ _ _ _ hh] at hl
              obtain ⟨e1, e2⟩ := hJ.cache_ok h v id hl
              exact ⟨e1, by simp [e2]⟩
          · intro k hk
            by_cases hh : k = cl.key r
            · subst hh
              constructor
              · intro hl; rw [lookup_cons_self] at hl; exact absurd hl (by simp)
              · intro v id hl
                rw [lookup_cons_self] at hl
                simp only [Option.some.injEq, Prod.mk.injEq] at hl
                obtain ⟨rfl, rfl⟩ := hl
                rw [matches_append, hcur, matches_single]; simp
            · rw [lookup_cons_other _ _ _ _ _ hh]
              have hne : ¬ (cl.key r = k) := fun e => hh e.symm
              refine ⟨fun hl => ?_, fun v id hl => ?_⟩
              · rw [matches_append, (hJ.fresh k hk).1 hl, matches_single]; simp [hne]
              · rw [matches_append, (hJ.fresh k hk).2 v id hl, matches_single]; simp [hne]
        · intro n hn'
          simp only [List.mem_append, List.mem_singleton] at hn'
          rcases hn' with hn' | rfl
          · have := hlt n hn'; simp only; omega
          · simp
      | some cv =>
        obtain ⟨v, id⟩ := cv
        have hcur : matchesK st.g (cl.key r) = [⟨id, cl.key r⟩] := (hJ.fresh _ hm0).2 v id hlk
        have hv : v = cl.var := (hJ.cache_ok _ v id hlk).1
        simp only [createFallback, hlk, hom, applySet]
        have href : refRow cl rst r = ⟨rst.g, rst.next, rst.out ++ [(r, some id)]⟩ := by
          simp only [refRow, hg, hcur, hom]
          simp [applySet]
        rw [href]
        have := ih { st with out := st.out ++ [(r, bindCached cl (v, id))] }
                   ⟨rst.g, rst.next, rst.out ++ [(r, some id)]⟩ hJ hg hn hlt
        simp only [bindCached, hv, beq_self_eq_true, if_true] at this ⊢
        obtain ⟨a1, a2, a3, a4, Ls, b1, b2, b3, _⟩ := this
        refine ⟨a1, a2, a3, a4, [(r, some id)] :: Ls, ?_, ?_, ?_, trivial⟩
        · rw [b1]; simp
        · rw [b2]; simp
        · rw [b3]; simp
    | cons m ms =>
      -- batch-start matchesK exist; the current graph has exactly those
      have hcur : matchesK st.g (cl.key r) = m :: ms := by
        rw [hgc, matches_append, hm0]
        have : matchesK created (cl.key r) = [] := by
          simp only [matchesK, List.filter_eq_nil_iff, beq_iff_eq]
          intro n hn' hk
          have := hcr n hn'; rw [hk, hm0] at this; cases this
        simp [this]
      simp only [phaseA, hom, applySet]
      have href : refRow cl rst r =
          ⟨rst.g, rst.next, rst.out ++ (m :: ms).map (fun m => (r, some m.id))⟩ := by
        simp only [refRow, hg, hcur, hom]
        simp [foldl_applySet_none, applySet, foldl_const_self]
      rw [href]
      have := ih { st with out := st.out ++ [(r, some m.id)] }
                 ⟨rst.g, rst.next, rst.out ++ (m :: ms).map (fun m => (r, some m.id))⟩ hJ hg hn hlt
      obtain ⟨a1, a2, a3, a4, Ls, b1, b2, b3, _⟩ := this
      refine ⟨a1, a2, a3, a4, ((m :: ms).map (fun m => (r, some m.id))) :: Ls, ?_, ?_, ?_, trivial⟩
      · simp only at b1; rw [b1]; simp
      · simp only at b2; rw [b2]; simp
      · simp only [Bool.false_or] at b3 ⊢
        cases ms with
        | nil => simp [b3]
        | cons m' ms' => simp [b3]


theorem heads_tails_perm {β : Type} (Ls : List (List β)) :
    (Ls.flatMap (List.take 1) ++ Ls.flatMap (List.drop 1)).Perm Ls.flatten := by
  induction Ls with
  | nil => simp
  | cons L Ls ih =>
    simp only [List.flatMap_cons, List.flatten_cons]
    have h1 : (List.take 1 L ++ Ls.flatMap (List.take 1) ++ (List.drop 1 L ++ Ls.flatMap (List.drop 1))).Perm
        (List.take 1 L ++ (List.drop 1 L ++ (Ls.flatMap (List.take 1) ++ Ls.flatMap (List.drop 1)))) := by
      rw [List.append_assoc]
      apply List.Perm.append_left
      rw [← List.append_assoc, ← List.append_assoc]
      exact List.perm_append_comm.append_right _
    refine h1.trans ?_
    rw [← List.append_assoc, List.take_append_drop]
    exact ih.append_left L

theorem phaseB_noSet (cl : Clause α) (hs : NoSet cl) :
    ∀ (q : List (α × List Node)) (st : SState (α := α)),
      phaseB cl st q = { st with out := st.out ++ q.flatMap (fun q => q.2.map (fun m => (q.1, some m.id))) }
  | [], st => by simp [phaseB]
  | (r, ms) :: rest, st => by
    simp only [phaseB, hs.2, foldl_applySet_none]
    rw [phaseB_noSet cl hs rest]
    simp [List.append_assoc]

/-- Global invariant between batches. -/
def G (cl : Clause α) (st : SState (α := α)) : Prop :=
  CacheOK cl st.g st.cache ∧ ∀ n ∈ st.g, n.id < st.next

theorem J_init (cl : Clause α) (st : SState (α := α)) (h : G cl st) : J cl st.g st.g st.cache := by
  refine ⟨⟨[], by simp, by simp⟩, h.1, ?_⟩
  intro k hk
  refine ⟨fun _ => hk, fun v id hl => ?_⟩
  have := (h.1 k v id hl).2
  have : (⟨id, k⟩ : Node) ∈ matchesK st.g k := mem_matches.mpr ⟨this, rfl⟩
  rw [hk] at this; cases this

/-- One batch, without SET: Rust's batch = the reference over the same rows. -/
theorem rustBatch_ref (cl : Clause α) (hs : NoSet cl) (st : SState (α := α)) (hG : G cl st)
    (rows : List α) :
    let R := rustBatch cl false st rows
    let F := rows.foldl (refRow cl) (⟨st.g, st.next, []⟩ : RState (α := α))
    R.g = F.g ∧ R.next = F.next ∧ G cl R ∧
    ∃ o, R.out = st.out ++ o ∧ o.Perm F.out := by
  have hA := phaseA_ref cl hs st.g rows st ⟨st.g, st.next, []⟩ (J_init cl st hG) rfl rfl hG.2
  simp only at hA
  obtain ⟨a1, a2, a3, a4, Ls, b1, b2, b3, _⟩ := hA
  simp only [rustBatch]
  rw [phaseB_noSet cl hs]
  refine ⟨a1.symm, a2.symm, ⟨a3.cache_ok, a4⟩, Ls.flatMap (List.take 1) ++ Ls.flatMap (List.drop 1), ?_, ?_⟩
  · simp only; rw [b1, b3, List.append_assoc]
  · rw [b2]; simpa using heads_tails_perm Ls

theorem refFold_out (cl : Clause α) (rows : List α) :
    ∀ (rst : RState (α := α)),
      (rows.foldl (refRow cl) rst).out = rst.out ++ (rows.foldl (refRow cl) ⟨rst.g, rst.next, []⟩).out ∧
      (rows.foldl (refRow cl) rst).g = (rows.foldl (refRow cl) ⟨rst.g, rst.next, []⟩).g ∧
      (rows.foldl (refRow cl) rst).next = (rows.foldl (refRow cl) ⟨rst.g, rst.next, []⟩).next := by
  induction rows with
  | nil => intro rst; simp
  | cons r rows ih =>
    intro rst
    simp only [List.foldl_cons]
    have e1 := ih (refRow cl rst r)
    have e2 := ih (refRow cl ⟨rst.g, rst.next, []⟩ r)
    have hr : ∀ (st : RState (α := α)), (refRow cl st r).g = (refRow cl ⟨st.g, st.next, []⟩ r).g ∧
        (refRow cl st r).next = (refRow cl ⟨st.g, st.next, []⟩ r).next ∧
        (refRow cl st r).out = st.out ++ (refRow cl ⟨st.g, st.next, []⟩ r).out := by
      intro st; simp only [refRow]; split <;> simp
    obtain ⟨h1, h2, h3⟩ := hr rst
    rw [e1.1, e2.1, h1, h2, h3, e1.2.1, e2.2.1, e1.2.2, e2.2.2, h1, h2]
    simp [List.append_assoc]

/-- **MERGE without SET clauses matches openCypher for every batching.**
For ANY split of the input rows into batches `bs` (in particular
`BATCH_SIZE`-sized ones), starting from any state whose cache is consistent
(e.g. a fresh query: empty cache), Rust's MERGE ends with the same graph and id
counter as the per-row reference, and emits a permutation of its rows (Rust
emits each batch's first matches, then the remaining matches). -/
theorem rust_eq_ref_noSet (cl : Clause α) (hs : NoSet cl) :
    ∀ (bs : List (List α)) (st : SState (α := α)), G cl st →
      let R := rustMerge cl false st bs
      let F := bs.flatten.foldl (refRow cl) (⟨st.g, st.next, []⟩ : RState (α := α))
      R.g = F.g ∧ R.next = F.next ∧ ∃ o, R.out = st.out ++ o ∧ o.Perm F.out := by
  intro bs
  induction bs with
  | nil => intro st _; exact ⟨rfl, rfl, [], by simp [rustMerge], by simp⟩
  | cons b bs ih =>
    intro st hG
    simp only [rustMerge, List.foldl_cons, List.flatten_cons, List.foldl_append]
    obtain ⟨a1, a2, a3, o1, o1e, o1p⟩ := rustBatch_ref cl hs st hG b
    obtain ⟨c1, c2, o2, o2e, o2p⟩ := ih (rustBatch cl false st b) a3
    simp only [rustMerge] at c1 c2 o2e
    have hf := refFold_out cl bs.flatten (b.foldl (refRow cl) ⟨st.g, st.next, []⟩)
    refine ⟨?_, ?_, o1 ++ o2, ?_, ?_⟩
    · rw [c1, hf.2.1, a1, a2]
    · rw [c2, hf.2.2, a1, a2]
    · rw [o2e, o1e, List.append_assoc]
    · rw [hf.1, ← a1, ← a2]; exact o1p.append o2p

theorem batches_flatten (B : Nat) (hB : 0 < B) (l : List α) : (batches B hB l).flatten = l := by
  induction l using batches.induct B hB with
  | case1 => rw [batches]; simp
  | case2 l h ih => rw [batches, dif_neg h]; simp [ih]

/-- The `BATCH_SIZE` instance, fresh query (empty cache). -/
theorem merge_noSet_any_batch_size (cl : Clause α) (hs : NoSet cl) (B : Nat) (hB : 0 < B)
    (g : Graph) (next : Nat) (hid : ∀ n ∈ g, n.id < next) (rows : List α) :
    let R := rustMerge cl false ⟨g, next, [], []⟩ (batches B hB rows)
    let F := refMerge cl ⟨g, next, []⟩ rows
    R.g = F.g ∧ R.next = F.next ∧ R.out.Perm F.out := by
  have hG : G cl (⟨g, next, [], []⟩ : SState (α := α)) :=
    ⟨fun h v id hl => by simp [lookup] at hl, hid⟩
  have := rust_eq_ref_noSet cl hs (batches B hB rows) ⟨g, next, [], []⟩ hG
  simp only [batches_flatten] at this
  obtain ⟨a, b, o, oe, op⟩ := this
  refine ⟨by simpa [refMerge] using a, by simpa [refMerge] using b, ?_⟩
  rw [oe]; simpa [refMerge] using op


/-! ## Counterexamples (each reproduced against the live Rust server and C) -/

/-- `UNWIND … AS x MERGE (n:L {v: CASE x WHEN 1 THEN 1 ELSE 2 END}) ON CREATE SET n.v = 2`. -/
def clCase : Clause Nat := ⟨0, fun x => if x = 1 then 1 else 2, some (fun _ => 2), none⟩

def st0 : SState (α := Nat) := ⟨[], 0, [], []⟩

theorem batches_1024 : batches 1024 (by decide) [1, 2] = [[1, 2]] := by
  simp [batches]

theorem batches_1 : batches 1 (by decide) [1, 2] = [[1], [2]] := by
  simp [batches]

/-- **Batch-size dependence.**  Rows 1 and 2 in one batch: row 2 does not see the
node row 1 created (and re-keyed to 2), so a second node is created.  In
separate batches (in the live repro: rows 1 and 1025 of `UNWIND range(1,1025)`)
row 2 matches it.  The reference creates 1 node; C creates 2 (C evaluates all
matches before any creation). -/
theorem batch_dependent :
    (rustMerge clCase false st0 [[1, 2]]).g.length = 2 ∧
    (rustMerge clCase false st0 [[1], [2]]).g.length = 1 ∧
    (refMerge clCase ⟨[], 0, []⟩ [1, 2]).g.length = 1 := by decide

/-- **Cross-clause cache.**  `MERGE (a:M {v:1}) ON CREATE SET a.v = 2 MERGE (b:M {v:1})`:
the second clause misses in the graph, hits the first clause's cache entry
(same hash: same label, same property map), inserts it under `a`'s variable id,
and `b` stays unbound. -/
def clA : Clause Unit := ⟨0, fun _ => 1, some (fun _ => 2), none⟩
def clB : Clause Unit := ⟨1, fun _ => 1, none, none⟩

def afterA : SState (α := Unit) := rustMerge clA false ⟨[], 0, [], []⟩ [[()]]
def afterB : SState (α := Unit) := rustMerge clB false { afterA with out := [] } [[()]]

theorem cross_clause_unbound :
    afterB.out = [((), none)] ∧ afterB.g.length = 1 ∧
    (refMerge clB (refMerge clA ⟨[], 0, []⟩ [()]) [()]).out = [((), some 0), ((), some 1)] ∧
    (refMerge clB (refMerge clA ⟨[], 0, []⟩ [()]) [()]).g.length = 2 := by decide

/-- With a *per-clause* cache (`cache := []` at each clause, as C's per-op
`unique_entities` rax) the second clause is correct. -/
theorem per_clause_cache_fixes :
    (rustMerge clB false { afterA with out := [], cache := [] } [[()]]).out = [((), some 1)] := by
  decide

/-- **All-bound shortcut** (`merge.rs:368-400`): two matches (parallel edges of
`MATCH (a),(b) MERGE p=(a)-[:R]->(b)`, modelled as two nodes with the key) but
only the first is emitted. The reference (and C) emit both. -/
def clK : Clause Unit := ⟨0, fun _ => 1, none, none⟩
def gPar : Graph := [⟨0, 1⟩, ⟨1, 1⟩]

theorem allBound_first_only :
    (rustMerge clK true ⟨gPar, 2, [], []⟩ [[()]]).out = [((), some 0)] ∧
    (rustMerge clK false ⟨gPar, 2, [], []⟩ [[()]]).out = [((), some 0), ((), some 1)] ∧
    (refMerge clK ⟨gPar, 2, []⟩ [()]).out = [((), some 0), ((), some 1)] := by decide

end Merge
