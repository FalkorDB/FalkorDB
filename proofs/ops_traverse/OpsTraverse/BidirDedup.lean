import OpsTraverse.Basic
/-
# The bidirectional dedup of chained anonymous undirected hops

| here | there |
| --- | --- |
| `St`, `bdNew`  | `struct BidirDedup` (`cond_traverse.rs`), `BidirDedup::new` |
| `clear`        | `BidirDedup::clear`; `CondTraverseOp::clear_bidir_dedup` (called from `set_argument_batch`, a new argument batch) |
| `enter`        | `BidirDedup::enter`: switch to the row's record, forgetting the previous record's destinations |
| `dedupRows`    | the `bidir_dedup` loop of `CondTraverseOp::expand_row` (`cond_traverse.rs`): `enter`, then `swap_remove` a destination already in `seen` |
| `dedupStream`  | `CondTraverseOp::next`: one state threaded through every child batch |
| `rec`          | the record identity `enter` hashes: the correlation tag and every bound column but the child hop's own bindings (its intermediate node and its edge, the `skip` slots) |

A result row is `(outer, key source, destination)`; `outer` stands for every
other column of the row (an `UNWIND` variable, an earlier MATCH's node, the real scan source
when the key source is an intermediate node, the correlation tag). The record of a row is
`(outer, key source)`: what the hop's input row binds besides the child hop's own slots. The
identity is modelled exactly (a 64-bit hash collision of two records is not modelled).
The intended semantics (C FalkorDB's `F * A * A`, one row of `F` per record) is one row per
input record and distinct destination (`cSem`). Rows whose key source is not a node are kept
without a check (`has_source`); not modelled. (`swap_remove` reorders the survivors; order is
not modelled, only membership and multiplicity.)
-/

namespace OpsTraverse.BidirDedup

abbrev Row := Nat × Nat × Nat   -- (outer, keySource, dest)

/-- The record a row belongs to. -/
def rec (r : Row) : Nat × Nat := (r.1, r.2.1)

/-- `BidirDedup { record, seen }` (`skip` only selects the columns `rec` reads). -/
structure St where
  record : Option (Nat × Nat)
  seen : List Nat
  deriving DecidableEq

/-- `BidirDedup::new`: no record, nothing seen. -/
def bdNew : St := ⟨none, []⟩

/-- `BidirDedup::clear`. -/
def clear (_s : St) : St := ⟨none, []⟩

/-- `BidirDedup::enter`: a new record starts with an empty `seen`. -/
def enter (k : Nat × Nat) (s : St) : St :=
  if s.record = some k then s else ⟨some k, []⟩

def dedupRows : St → List Row → List Row
  | _, [] => []
  | s, r :: rs =>
    if r.2.2 ∈ (enter (rec r) s).seen then dedupRows (enter (rec r) s) rs
    else r :: dedupRows ⟨(enter (rec r) s).record, r.2.2 :: (enter (rec r) s).seen⟩ rs

/-- The state after deduplicating `rs` from `s`. -/
def stAfter : St → List Row → St
  | s, [] => s
  | s, r :: rs =>
    if r.2.2 ∈ (enter (rec r) s).seen then stAfter (enter (rec r) s) rs
    else stAfter ⟨(enter (rec r) s).record, r.2.2 :: (enter (rec r) s).seen⟩ rs

/-- `CondTraverseOp::next` over child batches: one state threaded through all of them. -/
def dedupStream : St → List (List Row) → List Row
  | _, [] => []
  | s, b :: bs => dedupRows s b ++ dedupStream (stAfter s b) bs

theorem bdNew_spec : bdNew.record = none ∧ bdNew.seen = [] := ⟨rfl, rfl⟩

theorem clear_spec (s : St) : clear s = bdNew := rfl

theorem enter_same (k : Nat × Nat) (s : St) (h : s.record = some k) : enter k s = s := by
  simp [enter, h]

theorem enter_new (k : Nat × Nat) (s : St) (h : s.record ≠ some k) : enter k s = ⟨some k, []⟩ := by
  simp [enter, h]

/-- `enter` always leaves the state on the row's record. -/
theorem enter_record (k : Nat × Nat) (s : St) : (enter k s).record = some k := by
  unfold enter; split <;> simp_all

theorem dedupRows_append : ∀ (s : St) (a b : List Row),
    dedupRows s (a ++ b) = dedupRows s a ++ dedupRows (stAfter s a) b
  | _, [], _ => rfl
  | s, r :: a, b => by
    simp only [List.cons_append, dedupRows, stAfter]
    split <;> simp [dedupRows_append]

theorem dedupRows_sub : ∀ (s : St) (rs : List Row) (r : Row),
    r ∈ dedupRows s rs → r ∈ rs
  | _, [], _, h => by simp [dedupRows] at h
  | s, x :: rs, r, h => by
    unfold dedupRows at h
    split at h
    · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)

/-! ## The stream: batching is invisible -/

/-- **PROVEN**: deduplicating batch by batch with one threaded state is deduplicating the
concatenated child output. -/
theorem dedupStream_eq : ∀ (s : St) (bs : List (List Row)),
    dedupStream s bs = dedupRows s bs.flatten
  | _, [] => by simp [dedupStream, dedupRows]
  | s, b :: bs => by
    simp only [dedupStream, List.flatten_cons, dedupRows_append, dedupStream_eq]

/-- **PROVEN**: so any two batchings of the same child rows give the same result (a cleared
state, `clear_bidir_dedup`, starts the stream). -/
theorem dedupStream_batch_independent (s : St) (bs cs : List (List Row))
    (h : bs.flatten = cs.flatten) :
    dedupStream (clear s) bs = dedupStream (clear s) cs := by
  rw [dedupStream_eq, dedupStream_eq, h]

/-! ## Correctness: one row per record and destination -/

/-- Keep the first occurrence of each destination not in `seen`. -/
def firstKeep : List Nat → List Nat → List Nat
  | _, [] => []
  | seen, d :: ds => if d ∈ seen then firstKeep seen ds else d :: firstKeep (d :: seen) ds

/-- `seen` after `firstKeep`. -/
def fkSeen : List Nat → List Nat → List Nat
  | seen, [] => seen
  | seen, d :: ds => if d ∈ seen then fkSeen seen ds else fkSeen (d :: seen) ds

def mk (k : Nat × Nat) (d : Nat) : Row := (k.1, k.2, d)

/-- Adjacent groups have different records. -/
def adjDistinct : List ((Nat × Nat) × List Nat) → Prop
  | g :: h :: rest => g.1 ≠ h.1 ∧ adjDistinct (h :: rest)
  | _ => True

/-- The rows the hop's input carries: each record with the destinations the two hops reach
from it (one per intermediate, repeats included), contiguous and in order. -/
def rustIn (gs : List ((Nat × Nat) × List Nat)) : List Row :=
  gs.flatMap fun g => g.2.map (mk g.1)

/-- C's answer: per record, each destination once. -/
def cSem (gs : List ((Nat × Nat) × List Nat)) : List Row :=
  gs.flatMap fun g => (firstKeep [] g.2).map (mk g.1)

theorem group_run (k : Nat × Nat) : ∀ (seen ds : List Nat) (rest : List Row),
    dedupRows ⟨some k, seen⟩ (ds.map (mk k) ++ rest) =
      (firstKeep seen ds).map (mk k) ++ dedupRows ⟨some k, fkSeen seen ds⟩ rest
  | _, [], _ => rfl
  | seen, d :: ds, rest => by
    have hr : rec (mk k d) = k := rfl
    have he : enter (rec (mk k d)) ⟨some k, seen⟩ = ⟨some k, seen⟩ := by
      rw [hr]; exact enter_same _ _ rfl
    have hm : (mk k d).2.2 = d := rfl
    simp only [List.map_cons, List.cons_append, dedupRows, he, firstKeep, fkSeen, hm]
    by_cases hd : d ∈ seen
    · simp only [hd, ite_true]; exact group_run k seen ds rest
    · simp only [hd, ite_false, List.map_cons, List.cons_append]
      exact congrArg _ (group_run k (d :: seen) ds rest)

theorem rustIn_filter (gs : List ((Nat × Nat) × List Nat)) :
    rustIn gs = rustIn (gs.filter fun g => !g.2.isEmpty) := by
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    rcases g with ⟨k, _ | ⟨d, ds⟩⟩
    · simpa [rustIn] using ih
    · simp only [rustIn] at ih ⊢; simp [ih]

theorem cSem_filter (gs : List ((Nat × Nat) × List Nat)) :
    cSem gs = cSem (gs.filter fun g => !g.2.isEmpty) := by
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    rcases g with ⟨k, _ | ⟨d, ds⟩⟩
    · simpa [cSem, firstKeep] using ih
    · simp only [cSem] at ih ⊢; simp [ih]

theorem dedup_groups : ∀ (s : St) (gs : List ((Nat × Nat) × List Nat)),
    (∀ g ∈ gs, g.2 ≠ []) → adjDistinct gs →
    (∀ g ∈ gs.head?, s.record ≠ some g.1) →
    dedupRows s (rustIn gs) = cSem gs
  | _, [], _, _, _ => rfl
  | s, (k, ds) :: gs, hne, hch, hhd => by
    cases ds with
    | nil => exact absurd rfl (hne _ (List.mem_cons_self ..))
    | cons d ds =>
      have hs : s.record ≠ some k := hhd _ rfl
      have he : enter (rec (mk k d)) s = ⟨some k, []⟩ := enter_new _ _ hs
      have hrun := group_run k [d] ds (rustIn gs)
      have ih := dedup_groups ⟨some k, fkSeen [d] ds⟩ gs
        (fun g hg => hne g (List.mem_cons_of_mem _ hg))
        (by cases gs with
            | nil => trivial
            | cons g' gs' => exact hch.2)
        (by
          intro g hg
          cases gs with
          | nil => cases hg
          | cons g' gs' =>
            cases hg
            have := hch.1
            exact fun h => this (Option.some.inj h))
      simp only [rustIn, cSem, List.flatMap_cons, List.map_cons, List.cons_append] at hrun ih ⊢
      have hm : (mk k d).2.2 = d := rfl
      simp only [dedupRows, he, hm, List.not_mem_nil, ite_false, firstKeep]
      simp only [List.map_cons, List.cons_append]
      rw [hrun, ih]

/-- **PROVEN** (#3071, headline): when adjacent records are distinct (records that reach no
destination aside), the dedup returns exactly C's rows: per input record, each destination
it reaches once, whatever the record shares with others (`UNWIND`, an outer node) and however
the child output is cut into batches. -/
theorem dedup_eq_cSem (gs : List ((Nat × Nat) × List Nat)) (bs : List (List Row))
    (hbs : bs.flatten = rustIn gs)
    (hch : adjDistinct (gs.filter fun g => !g.2.isEmpty)) :
    dedupStream bdNew bs = cSem gs := by
  rw [dedupStream_eq, hbs, rustIn_filter, cSem_filter]
  refine dedup_groups _ _ ?_ hch ?_
  · intro g hg
    have := (List.mem_filter.1 hg).2
    cases h : g.2 <;> simp_all
  · intro g _; simp [bdNew]

/-! ## The former counterexamples of W2-traverse-2, now one row per record -/

/-- (a) Rows that differ only in an outer column are different records and both kept:
`UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)` keeps the `x = 2` rows. -/
theorem outer_rows_kept :
    dedupRows bdNew [(1, 1, 3), (2, 1, 3)] = [(1, 1, 3), (2, 1, 3)] := by decide

/-- (b) Three hops: start nodes 1 and 4 reaching destination 3 through the same
intermediate 2 are different records (the start node is bound in both). -/
theorem three_hops_kept :
    -- rows are (start, intermediate, dest)
    dedupRows bdNew [(1, 2, 3), (4, 2, 3)] = [(1, 2, 3), (4, 2, 3)] := by decide

/-- One record reaching a destination through two intermediates still gets it once, also when
the two rows arrive in different child batches. -/
theorem same_record_once :
    dedupStream bdNew [[(0, 1, 1)], [(0, 1, 1)]] = [(0, 1, 1)] := by decide

/-! ## Still open (W2-traverse-2, narrowed): adjacent records with equal bindings merge

`enter` cannot tell two consecutive records apart when every column it reads is equal
(`UNWIND [1, 1] AS i MATCH (a:A)-[]-()-[]-(b)`), so the second record's destinations are
dropped; C gives each record its rows. The code before #3071 dropped them too (same key). -/
theorem equal_adjacent_records_merge :
    dedupStream bdNew [rustIn [((1, 1), [3]), ((1, 1), [3])]] ≠
      cSem [((1, 1), [3]), ((1, 1), [3])] := by decide

end OpsTraverse.BidirDedup
