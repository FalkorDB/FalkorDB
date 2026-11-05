/-
# Batched deletion: the per-pair plan agrees with the sequential fold

`Remove.lean` models `remove_all`'s slow path as `removeOne` per edge, folded over
the batch. `Tensor::remove_all` no longer works that way. To make demotion linear
in the batch rather than quadratic (issue #2429, fixed by #2431), it splits into

* a **read phase** that reads `dp`/`dm`/`me` and writes none of them: it reads
  each touched pair's `me` row *once*, replays that pair's transitions into a
  `PairPlan`, and buffers the `me` removals; and
* a **write phase** that writes without reading those layers back.

The equivalence of that split with the sequential fold is what this file proves.
It was previously carried by regression tests alone, and it was the one place
where the code had moved and the model had not followed.

```rust
enum PairPlan {
    Multi(Vec<u64>),                        // the me row, read once, kept sorted
    Single { id: u64, demoted: bool },
    Emptied,
    Absent,
}
```

The Rust keeps the row as a sorted `Vec` and reaches into it with
`binary_search`; that is a lookup strategy, not content, and the row *is* a set of
ids (the Rust asserts the sortedness it relies on). So [`PairPlan.multi`] carries
a `Finset`, and the two agree on what the plan holds.

## Why this is not merely bookkeeping

Two steps make the interleaving delicate, and both are why a per-pair formulation
could plausibly be wrong rather than obviously right.

* A `multi` plan can never step to an empty row. That is `removeOne_survivor`
  (`Remove.lean`): a `MULTI` pair holds ≥ 2 ids, so erasing one leaves a
  survivor. The Rust carries an `unreachable!()` there.
* `demoted := true` is what makes a *later* removal of the survivor **in the same
  batch** behave as if the demotion had already been written: the plan is
  `single`, so that removal takes the `emptied` arm. The sequential fold gets this
  by actually writing the demotion first. Agreement here is the crux.

## The two paths are equal as *states*, not as terms

They are not equal as terms, and the obstruction is instructive rather than
technical. Where a pair demotes and is then emptied in the same batch, the
sequential fold writes the survivor into `dp` and removes it again; the batched
path never writes it. `Layer` carries a total `val` function alongside its
pattern, so those two layers differ in `val` **at a coordinate outside the
pattern** — a cell neither GraphBLAS nor `tensor.rs` can read, since every read
goes through `Layer.get`, which consults the pattern first.

So the statement to prove is observational, and [`TEquiv`] is that notion: equal
patterns, equal values where the pattern says a value is stored, and equal
`me`/`mt`/`dm`/capacity. Anyone attempting term equality here will get stuck on
exactly this, which is worth knowing before starting.

## What is proved

* [`tequiv_applyPlan_removeFold`] — for one pair, replaying its plan and applying
  it once is observationally equal to folding `removeOne` over the same ids.
* [`reported_iff`] — and the two report the same pair as emptied.
* [`applyPlan_comm`] — plans for distinct pairs commute, so the write phase's
  hash-map order is irrelevant. Each component's update is named (`dpOp`, `dmOp`,
  `mtOp`) as a function of the *original* tensor, which is what makes this nine
  one-line cases: the decision each shape takes reads `m`, which neither shape
  writes, so both orders take the same decisions. The `me` half is `me_sdiff_comm`,
  and that is where `key_inj` enters — distinct pairs own distinct rows.
* [`inv_applyPlan`] — hence the batched path preserves `Inv`, inherited through
  [`TEquiv`] from what `Remove.lean` already proves about the fold.

The write phase's own ordering (`me` first, then the forward and backward layers)
is not modelled as an order: `applyPlan` is one state update, and
[`applyPlan_comm`] is what says the order cannot matter.
-/
import Tensor.Remove

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair}

/-! ## Observational equality

`Layer.val` is total but only meaningful on `Layer.dom`, and every read in
`tensor.rs` goes through `Layer.get`, which checks the pattern first. Two tensors
that agree on every `get` and on the Boolean structures are therefore
indistinguishable to any operation in this development. -/

/-- Observational equality: agreement on everything a reader can observe. -/
structure TEquiv (a b : Tensor) : Prop where
  m_get : ∀ q, a.m.get q = b.m.get q
  dp_get : ∀ q, a.dp.get q = b.dp.get q
  dm : a.dm = b.dm
  mt : a.mt = b.mt
  me : a.me = b.me
  nrows : a.nrows = b.nrows
  ncols : a.ncols = b.ncols

namespace TEquiv

theorem refl (a : Tensor) : TEquiv a a :=
  ⟨fun _ => rfl, fun _ => rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem symm {a b : Tensor} (h : TEquiv a b) : TEquiv b a :=
  ⟨fun q => (h.m_get q).symm, fun q => (h.dp_get q).symm, h.dm.symm, h.mt.symm, h.me.symm,
   h.nrows.symm, h.ncols.symm⟩

theorem trans {a b c : Tensor} (h1 : TEquiv a b) (h2 : TEquiv b c) : TEquiv a c :=
  ⟨fun q => (h1.m_get q).trans (h2.m_get q), fun q => (h1.dp_get q).trans (h2.dp_get q),
   h1.dm.trans h2.dm, h1.mt.trans h2.mt, h1.me.trans h2.me, h1.nrows.trans h2.nrows,
   h1.ncols.trans h2.ncols⟩

/-- `get` determines the pattern, so agreement on `get` gives agreement on `dom`. -/
theorem m_dom {a b : Tensor} (h : TEquiv a b) : a.m.dom = b.m.dom := by
  ext q
  have := h.m_get q
  simp only [Layer.get] at this
  by_cases hq : q ∈ a.m.dom <;> by_cases hq' : q ∈ b.m.dom <;> simp_all

theorem dp_dom {a b : Tensor} (h : TEquiv a b) : a.dp.dom = b.dp.dom := by
  ext q
  have := h.dp_get q
  simp only [Layer.get] at this
  by_cases hq : q ∈ a.dp.dom <;> by_cases hq' : q ∈ b.dp.dom <;> simp_all

theorem effGet {a b : Tensor} (h : TEquiv a b) (q : Pair) : a.effGet q = b.effGet q := by
  simp only [Tensor.effGet, h.dp_get q, h.dm, h.m_get q, h.dp_dom]

theorem effDom {a b : Tensor} (h : TEquiv a b) : a.effDom = b.effDom := by
  ext q; rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, h.effGet q]

theorem meRow {a b : Tensor} (h : TEquiv a b) (k : Addr) : a.meRow k = b.meRow k := by
  simp only [Tensor.meRow, h.me]

theorem edgesAt {a b : Tensor} (h : TEquiv a b) (q : Pair) : a.edgesAt q = b.edgesAt q :=
  edgesAt_congr_at (h.effGet q) (h.meRow _)

theorem multiPairs {a b : Tensor} (h : TEquiv a b) : a.multiPairs = b.multiPairs :=
  multiPairs_congr h.effDom h.effGet

/-- `Inv` is stated entirely in terms of `get`, the Boolean structures and the
capacity, so it transfers. This is what lets the batched path inherit every
invariant `Remove.lean` proves of the fold instead of re-proving them. -/
theorem inv {a b : Tensor} (h : TEquiv a b) (hb : Inv b) : Inv a := by
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, in_range := ?_,
           valid_ids := ?_, mt_eq := ?_ }
  · rw [h.dm, h.m_dom]; exact hb.dm_sub_m
  · rw [h.dp_dom, h.dm]; exact hb.dp_disj_dm
  · intro q hq
    have hq' : q ∈ b.dp.dom := by rwa [h.dp_dom] at hq
    have hbv : b.dp.get q = some (b.dp.val q) := Layer.get_of_mem hq'
    have hav : a.dp.get q = some (a.dp.val q) := Layer.get_of_mem hq
    have hval : a.dp.val q = b.dp.val q := by
      have := (h.dp_get q).symm.trans hav
      rw [hbv] at this; exact (Option.some_inj.mp this).symm
    rw [h.m_get q, hval]; exact hb.cancel_clean q hq'
  · intro q hq; rw [h.meRow]; exact hb.multi_iff q (by rwa [← h.effGet q])
  · intro q hq; rw [h.meRow]; exact hb.row_empty q (by rwa [← h.effGet q])
  · intro x hx
    obtain ⟨q, hqd, hqk⟩ := hb.me_keyed x (by rwa [h.me] at hx)
    exact ⟨q, by rwa [h.effDom], hqk⟩
  · rw [h.m_dom, h.dp_dom, h.nrows, h.ncols]; exact hb.in_range
  · intro q i hi; rw [h.edgesAt q] at hi; exact hb.valid_ids q i hi
  · intro q; rw [h.mt, h.effDom]; exact hb.mt_eq q

end TEquiv

/-! ## The plan -/

/-- One touched pair's read-phase state. `multi` carries the row read once; the
`Bool` on `single` is the Rust's `demoted`, which records that this pair's
surviving id has to be written back inline. -/
inductive PairPlan where
  | multi (ids : Finset Nat)
  | single (id : Nat) (demoted : Bool)
  | emptied
  | absent
  deriving DecidableEq

/-- The plan plus the two things the read phase accumulates beside it: the ids to
drop from this pair's `me` row, and whether the pair was reported emptied. -/
structure PairState where
  plan : PairPlan
  del : Finset Nat
  reported : Bool

/-- `plans.entry(..).or_insert_with(|| match self.eff_get(src, dst) { .. })`. -/
def initPlan (t : Tensor) (p : Pair) : PairPlan :=
  match t.effGet p with
  | some v => if v = MULTI then .multi (t.meRow (key p)) else .single v false
  | none => .absent

def initState (t : Tensor) (p : Pair) : PairState :=
  { plan := initPlan t p, del := ∅, reported := false }

/-- One named edge, advancing this pair's plan. Mirrors the read phase's
`match plan` arm for arm. -/
def stepPlan (s : PairState) (e : Nat) : PairState :=
  match s.plan with
  | .multi ids =>
      if e ∈ ids then
        if (ids.erase e).card = 1 then
          -- Down to one edge: the pair demotes and its whole row leaves `me`.
          -- Destructured as an `Option` to mirror `removeOne`, whose `none` arm
          -- `removeOne_survivor` shows unreachable; the same argument covers this
          -- one, and a total definition has to keep it.
          match (ids.erase e).min with
          | some last =>
              { plan := .single last true, del := insert last (insert e s.del),
                reported := s.reported }
          | none => { plan := .multi (ids.erase e), del := insert e s.del,
                      reported := s.reported }
        else
          { plan := .multi (ids.erase e), del := insert e s.del, reported := s.reported }
      else s
  | .single inline _ =>
      if inline = e then { plan := .emptied, del := s.del, reported := true } else s
  | .emptied => s
  | .absent => s

/-- The read phase over one pair's named ids, in batch order. -/
def planFold (t : Tensor) (p : Pair) (ids : List Nat) : PairState :=
  ids.foldl stepPlan (initState t p)

/-- The write phase for one pair: drop the buffered `me` entries, then settle the
forward and backward layers according to the plan. `m` is never written by
deletion, so reading `t.m` here rather than the fold's intermediate is the same
read — which is what makes the phase split possible at all. -/
def applyPlan (t : Tensor) (p : Pair) (s : PairState) : Tensor :=
  let t1 : Tensor := { t with me := t.me \ s.del.image (fun i => (key p, i)) }
  match s.plan with
  | .emptied => deletePair t1 p
  | .single id true =>
      if t.m.get p = some id then { t1 with dp := t1.dp.remove p }
      else { t1 with dp := t1.dp.set p id }
  | _ => t1

/-- The sequential fold this is supposed to agree with, restricted to one pair. -/
def removeFold (t : Tensor) (p : Pair) (ids : List Nat) : Tensor :=
  ids.foldl (fun t' i => (removeOne t' i p).1) t

/-! ## `me` bookkeeping

The deletions are applied as one set difference where the fold erases one entry at
a time; these lemmas move an erase across the difference. -/

@[simp] theorem me_sdiff_empty (s : Finset (Addr × Nat)) (k : Addr) :
    s \ (∅ : Finset Nat).image (fun i => (k, i)) = s := by simp

theorem me_sdiff_insert (s : Finset (Addr × Nat)) (d : Finset Nat) (k : Addr) (i : Nat) :
    s \ (insert i d).image (fun j => (k, j)) = (s \ d.image (fun j => (k, j))).erase (k, i) := by
  ext x
  simp only [Finset.mem_sdiff, Finset.mem_erase, Finset.mem_image, Finset.mem_insert]
  constructor
  · rintro ⟨hx, hni⟩
    refine ⟨?_, hx, ?_⟩
    · rintro rfl; exact hni ⟨i, Or.inl rfl, rfl⟩
    · rintro ⟨j, hj, hjx⟩; exact hni ⟨j, Or.inr hj, hjx⟩
  · rintro ⟨hne, hx, hnd⟩
    refine ⟨hx, ?_⟩
    rintro ⟨j, hj | hj, hjx⟩
    · exact hne (by rw [← hjx, hj])
    · exact hnd ⟨j, hj, hjx⟩

/-- The row of `p` after the buffered deletions is what the plan still holds; this
is the invariant that makes re-reading the row unnecessary. -/
theorem meRow_sdiff (t : Tensor) (p : Pair) (d : Finset Nat) :
    meRowOf (t.me \ d.image (fun i => (key p, i))) (key p) = t.meRow (key p) \ d := by
  ext i
  simp only [mem_meRowOf, Finset.mem_sdiff, Finset.mem_image, mem_meRow]
  constructor
  · rintro ⟨hx, hni⟩
    exact ⟨hx, fun hi => hni ⟨i, hi, rfl⟩⟩
  · rintro ⟨hx, hni⟩
    refine ⟨hx, ?_⟩
    rintro ⟨j, hj, hjx⟩
    exact hni (by rw [← ((Prod.mk.injEq _ _ _ _).mp hjx).2]; exact hj)

/-! ## Shapes of `applyPlan`

For every plan but a demote, `applyPlan` touches `me` alone — so `effGet` at the
pair is still the *original* tensor's, which is what lets the read phase decide
without consulting anything it has written. -/

@[simp] theorem applyPlan_multi {s : PairState} {r : Finset Nat} (h : s.plan = .multi r) :
    applyPlan t p s = { t with me := t.me \ s.del.image (fun i => (key p, i)) } := by
  simp [applyPlan, h]

@[simp] theorem applyPlan_absent {s : PairState} (h : s.plan = .absent) :
    applyPlan t p s = { t with me := t.me \ s.del.image (fun i => (key p, i)) } := by
  simp [applyPlan, h]

@[simp] theorem applyPlan_single_clean {s : PairState} {i : Nat} (h : s.plan = .single i false) :
    applyPlan t p s = { t with me := t.me \ s.del.image (fun i => (key p, i)) } := by
  simp [applyPlan, h]

@[simp] theorem applyPlan_emptied {s : PairState} (h : s.plan = .emptied) :
    applyPlan t p s
      = deletePair { t with me := t.me \ s.del.image (fun i => (key p, i)) } p := by
  simp [applyPlan, h]

theorem applyPlan_demote {s : PairState} {i : Nat} (h : s.plan = .single i true) :
    applyPlan t p s =
      (let t1 : Tensor := { t with me := t.me \ s.del.image (fun j => (key p, j)) }
       if t.m.get p = some i then { t1 with dp := t1.dp.remove p }
       else { t1 with dp := t1.dp.set p i }) := by
  simp [applyPlan, h]

/-- Only `me` changed, so the forward layers — and hence `effGet` — are the
original's. -/
theorem effGet_me_only {d : Finset Nat} (q : Pair) :
    ({ t with me := t.me \ d.image (fun i => (key p, i)) } : Tensor).effGet q = t.effGet q := rfl

/-! ## The read phase's invariant

What the plan claims about the tensor it was read from. Everything here is about
the *original* `t`: the read phase never observes its own writes, which is the
property that makes it a read phase. -/

def Consistent (t : Tensor) (p : Pair) (s : PairState) : Prop :=
  match s.plan with
  | .multi r =>
      t.effGet p = some MULTI ∧ p ∉ t.dm ∧ t.meRow (key p) \ s.del = r ∧ 2 ≤ r.card
        ∧ s.reported = false
  | .single i true =>
      t.effGet p = some MULTI ∧ p ∉ t.dm ∧ t.meRow (key p) \ s.del = ∅ ∧ i ≠ MULTI
        ∧ s.reported = false
  | .single i false =>
      t.effGet p = some i ∧ i ≠ MULTI ∧ s.del = ∅ ∧ s.reported = false
  -- `emptied` is absorbing, and it is the only plan that reports; that biconditional
  -- is what makes the report agreement provable rather than merely plausible.
  -- `t.effGet p ≠ none` records that only a pair that was *there* can be emptied,
  -- which is what stops the batch reporting a pair it never held.
  | .emptied => s.reported = true ∧ t.effGet p ≠ none
  | .absent => t.effGet p = none ∧ s.del = ∅ ∧ s.reported = false

/-- The plan the read phase starts from is consistent, which is where `Inv` is
used: `multi_iff` supplies the "at least two ids" that makes the demote step's
`unreachable!()` sound, and `not_mem_dm_of_multi` the fact that a `MULTI` pair
carries no tombstone. -/
theorem consistent_initState (h : Inv t) : Consistent t p (initState t p) := by
  have hplan : (initState t p).plan = initPlan t p := rfl
  have hdel : (initState t p).del = ∅ := rfl
  rcases hv : t.effGet p with _ | v
  · have hip : initPlan t p = .absent := by simp [initPlan, hv]
    simp only [Consistent, hplan, hip]
    exact ⟨hv, hdel, rfl⟩
  · by_cases hM : v = MULTI
    · subst hM
      have hip : initPlan t p = .multi (t.meRow (key p)) := by simp [initPlan, hv]
      simp only [Consistent, hplan, hip]
      exact ⟨hv, not_mem_dm_of_multi h hv, by rw [hdel, Finset.sdiff_empty],
        h.multi_iff p hv, rfl⟩
    · have hip : initPlan t p = .single v false := by simp [initPlan, hv, hM]
      simp only [Consistent, hplan, hip]
      exact ⟨hv, hM, hdel, rfl⟩

/-! ## The step lemma

One named edge: replaying it into the plan and applying the plan once is
observationally what the fold's `removeOne` does to the applied plan. Every case
of `stepPlan` is matched against the corresponding case of `removeOne`, and the
`multi` case splits again on whether this removal demotes the pair. -/

/-- `applyPlan` on a plan that is not a demote and not `emptied` touches `me` only,
so the pair still reads as it did in the original tensor. -/
private theorem me_only_effGet {d : Finset Nat} (q : Pair) :
    ({ t with me := t.me \ d.image (fun i => (key p, i)) } : Tensor).effGet q = t.effGet q := rfl

private theorem me_only_meRow {d : Finset Nat} :
    ({ t with me := t.me \ d.image (fun i => (key p, i)) } : Tensor).meRow (key p)
      = t.meRow (key p) \ d := meRow_sdiff t p d

theorem applyPlan_stepPlan (h : Inv t) {s : PairState} (hc : Consistent t p s) (e : Nat) :
    TEquiv (applyPlan t p (stepPlan s e)) (removeOne (applyPlan t p s) e p).1
      ∧ Consistent t p (stepPlan s e)
      ∧ ((removeOne (applyPlan t p s) e p).2 = some p
          ↔ ((stepPlan s e).reported = true ∧ s.reported = false)) := by
  cases hplan : s.plan with
  | multi r =>
    simp only [Consistent, hplan] at hc
    obtain ⟨hv, hdm, hrow, hcard, hrep⟩ := hc
    have ha : applyPlan t p s = { t with me := t.me \ s.del.image (fun i => (key p, i)) } :=
      applyPlan_multi hplan
    have ham : (applyPlan t p s).m = t.m := by rw [ha]
    have haget : (applyPlan t p s).effGet p = some MULTI := by rw [ha]; exact hv
    have harow : (applyPlan t p s).meRow (key p) = r := by
      rw [ha]; exact (meRow_sdiff t p s.del).trans hrow
    have hafter : rowAfterErase (applyPlan t p s) p e = r.erase e := by
      rw [rowAfterErase_eq, harow]
    by_cases hmem : e ∈ r
    · by_cases hone : (r.erase e).card = 1
      · -- demote: the survivor leaves `me` and returns inline
        obtain ⟨last, hlast⟩ := Finset.card_eq_one.mp hone
        have hmin : (rowAfterErase (applyPlan t p s) p e).min = some last := by
          rw [hafter, hlast]; rfl
        have hncard : ¬ 2 ≤ (rowAfterErase (applyPlan t p s) p e).card := by
          rw [hafter, hone]; omega
        have hstep : stepPlan s e =
            { plan := .single last true, del := insert last (insert e s.del),
              reported := s.reported } := by
          simp only [stepPlan, hplan, if_pos hmem, if_pos hone,
            show (r.erase e).min = some last from by rw [hlast]; rfl]
        have hlastM : last ≠ MULTI := by
          have hlr : last ∈ r := Finset.mem_of_mem_erase (by rw [hlast]; simp)
          have h2 : last ∈ t.meRow (key p) \ s.del := by rw [hrow]; exact hlr
          exact (h.valid_ids p last
            (by rw [edgesAt_of_multi hv]; exact (Finset.mem_sdiff.mp h2).1)).ne_multi
        have hmeq : ((applyPlan t p s).me.erase (key p, e)).erase (key p, last)
            = t.me \ (insert last (insert e s.del)).image (fun j => (key p, j)) := by
          rw [me_sdiff_insert, me_sdiff_insert, ha]
        refine ⟨?_, ?_, ?_⟩
        · rw [applyPlan_demote (t := t) (p := p) (s := stepPlan s e) (by rw [hstep])]
          by_cases hm : t.m.get p = some last
          · rw [removeOne_demote_cancel haget hncard hmin (by rw [ham]; exact hm)]
            simp only [hstep, if_pos hm]
            exact ⟨fun q => by rw [ha], fun q => by rw [ha], by rw [ha], by rw [ha],
              hmeq.symm, by rw [ha], by rw [ha]⟩
          · rw [removeOne_demote_shadow haget hncard hmin (by rw [ham]; exact hm)]
            simp only [hstep, if_neg hm]
            exact ⟨fun q => by rw [ha], fun q => by rw [ha], by rw [ha], by rw [ha],
              hmeq.symm, by rw [ha], by rw [ha]⟩
        · simp only [Consistent, hstep]
          refine ⟨hv, hdm, ?_, hlastM, hrep⟩
          rw [Finset.sdiff_insert, Finset.sdiff_insert, hrow, hlast]
          simp
        · by_cases hm : t.m.get p = some last
          · rw [removeOne_demote_cancel haget hncard hmin (by rw [ham]; exact hm)]
            simp [hstep]
          · rw [removeOne_demote_shadow haget hncard hmin (by rw [ham]; exact hm)]
            simp [hstep]
      · -- still multi
        have hge2 : 2 ≤ (r.erase e).card := by
          have := Finset.card_erase_of_mem hmem; omega
        have hstep : stepPlan s e =
            { plan := .multi (r.erase e), del := insert e s.del, reported := s.reported } := by
          simp only [stepPlan, hplan, if_pos hmem, if_neg hone]
        refine ⟨?_, ?_, ?_⟩
        · have hap := applyPlan_multi (t := t) (p := p) (s := stepPlan s e) (by rw [hstep])
          rw [hap, removeOne_still_multi haget (by rw [hafter]; exact hge2)]
          simp only [hstep]
          exact ⟨fun q => by rw [ha], fun q => by rw [ha], by rw [ha], by rw [ha],
            by rw [me_sdiff_insert, ha], by rw [ha], by rw [ha]⟩
        · simp only [Consistent, hstep]
          exact ⟨hv, hdm, by rw [Finset.sdiff_insert, hrow], hge2, hrep⟩
        · rw [removeOne_still_multi haget (by rw [hafter]; exact hge2)]
          simp [hstep]
    · -- an e this pair does not hold: both sides are no-ops
      have hstep : stepPlan s e = s := by simp only [stepPlan, hplan, if_neg hmem]
      have hnotin : (key p, e) ∉ (applyPlan t p s).me := by
        rw [← mem_meRow, harow]; exact hmem
      have hnoop : (applyPlan t p s).me.erase (key p, e) = (applyPlan t p s).me :=
        Finset.erase_eq_of_notMem hnotin
      have hst : 2 ≤ (rowAfterErase (applyPlan t p s) p e).card := by
        rw [hafter, Finset.erase_eq_of_notMem hmem]; exact hcard
      refine ⟨?_, ?_, ?_⟩
      · rw [hstep, removeOne_still_multi haget hst]
        exact ⟨fun q => rfl, fun q => rfl, rfl, rfl, hnoop.symm, rfl, rfl⟩
      · rw [hstep]; simp only [Consistent, hplan]; exact ⟨hv, hdm, hrow, hcard, hrep⟩
      · rw [removeOne_still_multi haget hst]; simp [hstep]
  | single i b =>
    rcases b with _ | _
    · -- a clean single-edge pair: `applyPlan` is the identity
      simp only [Consistent, hplan] at hc
      obtain ⟨hv, hiM, hdel, hrep⟩ := hc
      have ha : applyPlan t p s = t := by
        rw [applyPlan_single_clean hplan, hdel]; simp
      have haget : (applyPlan t p s).effGet p = some i := by rw [ha]; exact hv
      by_cases hid : i = e
      · have hage : (applyPlan t p s).effGet p = some e := by rw [haget, hid]
        have heM : e ≠ MULTI := by rw [← hid]; exact hiM
        have hstep : stepPlan s e = { plan := .emptied, del := s.del, reported := true } := by
          simp only [stepPlan, hplan, if_pos hid]
        refine ⟨?_, ?_, ?_⟩
        · have hap := applyPlan_emptied (t := t) (p := p) (s := stepPlan s e) (by rw [hstep])
          rw [hap, removeOne_single hage heM]
          simp only [hstep, hdel]
          rw [show t.me \ (∅ : Finset Nat).image (fun j => (key p, j)) = t.me by simp, ha]
          exact TEquiv.refl _
        · simp only [Consistent, hstep]
          exact ⟨trivial, by rw [hv]; simp⟩
        · rw [removeOne_single hage heM]; simp [hstep, hrep]
      · have hstep : stepPlan s e = s := by simp only [stepPlan, hplan, if_neg hid]
        have hnot : e ∉ (applyPlan t p s).edgesAt p := by
          rw [ha, edgesAt_of_single hv hiM]; simpa using Ne.symm hid
        refine ⟨?_, ?_, ?_⟩
        · rw [hstep, removeOne_noop_of_not_mem hnot (by rw [haget]; simpa using hiM)]
          exact TEquiv.refl _
        · rw [hstep]; simp only [Consistent, hplan]; exact ⟨hv, hiM, hdel, hrep⟩
        · rw [removeOne_noop_of_not_mem hnot (by rw [haget]; simpa using hiM)]; simp [hstep]
    · -- a demoted pair: the survivor is live inline, in `dp` or via `m`
      simp only [Consistent, hplan] at hc
      obtain ⟨hv, hdm, hrow, hiM, hrep⟩ := hc
      have ha := applyPlan_demote (t := t) (p := p) (s := s) hplan
      have haget : (applyPlan t p s).effGet p = some i := by
        rw [ha]
        by_cases hm : t.m.get p = some i
        · simp only [hm, if_pos]
          exact (effGet_of_m (by simp) (by simpa using hdm)).trans hm
        · simp only [hm, if_neg]
          exact effGet_of_dp (by simp)
      by_cases hid : i = e
      · have hage : (applyPlan t p s).effGet p = some e := by rw [haget, hid]
        have heM : e ≠ MULTI := by rw [← hid]; exact hiM
        have hstep : stepPlan s e = { plan := .emptied, del := s.del, reported := true } := by
          simp only [stepPlan, hplan, if_pos hid]
        refine ⟨?_, ?_, ?_⟩
        · have hap := applyPlan_emptied (t := t) (p := p) (s := stepPlan s e) (by rw [hstep])
          rw [hap, removeOne_single hage heM, ha]
          simp only [hstep]
          -- Both sides delete the pair. Every component but `dp` is literally the
          -- same term; `dp` agrees on every `get` and differs only in the value
          -- left behind at `p`, which is outside the pattern — the whole reason
          -- this file proves `TEquiv` and not equality.
          by_cases hm : t.m.get p = some i
          · rw [if_pos hm]
            refine ⟨fun q => rfl, ?_, rfl, rfl, rfl, rfl, rfl⟩
            intro q
            by_cases hq : q = p
            · rw [hq]; simp [deletePair, Layer.get, Layer.remove]
            · simp [deletePair, Layer.get, Layer.remove, Finset.mem_erase, hq]
          · rw [if_neg hm]
            refine ⟨fun q => rfl, ?_, rfl, rfl, rfl, rfl, rfl⟩
            intro q
            by_cases hq : q = p
            · rw [hq]; simp [deletePair, Layer.get, Layer.remove, Layer.set]
            · simp [deletePair, Layer.get, Layer.remove, Layer.set, Finset.mem_erase,
                Finset.mem_insert, hq]
        · simp only [Consistent, hstep]
          exact ⟨trivial, by rw [hv]; simp⟩
        · rw [removeOne_single hage heM]; simp [hstep, hrep]
      · have hstep : stepPlan s e = s := by simp only [stepPlan, hplan, if_neg hid]
        have hnot : e ∉ (applyPlan t p s).edgesAt p := by
          rw [edgesAt_of_single haget hiM]; simpa using Ne.symm hid
        refine ⟨?_, ?_, ?_⟩
        · rw [hstep, removeOne_noop_of_not_mem hnot (by rw [haget]; simpa using hiM)]
          exact TEquiv.refl _
        · rw [hstep]; simp only [Consistent, hplan]; exact ⟨hv, hdm, hrow, hiM, hrep⟩
        · rw [removeOne_noop_of_not_mem hnot (by rw [haget]; simpa using hiM)]; simp [hstep]
  | emptied =>
    simp only [Consistent, hplan] at hc
    have hstep : stepPlan s e = s := by simp only [stepPlan, hplan]
    have haget : (applyPlan t p s).effGet p = none := by
      rw [applyPlan_emptied hplan]; exact deletePair_effGet_self
    have hnot : e ∉ (applyPlan t p s).edgesAt p := by simp [edgesAt, haget]
    refine ⟨?_, ?_, ?_⟩
    · rw [hstep, removeOne_noop_of_not_mem hnot (by rw [haget]; simp)]; exact TEquiv.refl _
    · rw [hstep]; simp only [Consistent, hplan]; exact hc
    · rw [removeOne_noop_of_not_mem hnot (by rw [haget]; simp)]; simp [hstep]
  | absent =>
    simp only [Consistent, hplan] at hc
    obtain ⟨hv, hdel, hrep⟩ := hc
    have hstep : stepPlan s e = s := by simp only [stepPlan, hplan]
    have ha : applyPlan t p s = t := by rw [applyPlan_absent hplan, hdel]; simp
    have haget : (applyPlan t p s).effGet p = none := by rw [ha]; exact hv
    have hnot : e ∉ (applyPlan t p s).edgesAt p := by simp [edgesAt, haget]
    refine ⟨?_, ?_, ?_⟩
    · rw [hstep, removeOne_noop_of_not_mem hnot (by rw [haget]; simp)]; exact TEquiv.refl _
    · rw [hstep]; simp only [Consistent, hplan]; exact ⟨hv, hdel, hrep⟩
    · rw [removeOne_noop_of_not_mem hnot (by rw [haget]; simp)]; simp [hstep]

/-! ## `removeOne` respects observational equality

Needed to push the induction through: the fold continues from the *sequential*
tensor while the plan continues from the *applied* one, so the two tails must stay
related. Everything `removeOne` reads — `effGet` at the pair, the `me` row, the
committed value — is `TEquiv`-invariant, and everything it writes preserves the
relation. -/

theorem TEquiv.me_erase {a b : Tensor} (h : TEquiv a b) (x : Addr × Nat) :
    TEquiv { a with me := a.me.erase x } { b with me := b.me.erase x } :=
  ⟨h.m_get, h.dp_get, h.dm, h.mt, by rw [h.me], h.nrows, h.ncols⟩

theorem TEquiv.deletePair_congr {a b : Tensor} (h : TEquiv a b) (q : Pair) :
    TEquiv (deletePair a q) (deletePair b q) := by
  refine ⟨h.m_get, ?_, ?_, by simp [deletePair, h.mt], by simp [deletePair, h.me],
    h.nrows, h.ncols⟩
  · intro r
    by_cases hr : r = q
    · rw [hr]; simp [deletePair]
    · simp only [deletePair]; rw [Layer.get_remove_ne hr, Layer.get_remove_ne hr]; exact h.dp_get r
  · simp only [deletePair, h.dm, h.m_dom]

theorem TEquiv.removeOne_congr {a b : Tensor} (h : TEquiv a b) (e : Nat) (q : Pair) :
    TEquiv (removeOne a e q).1 (removeOne b e q).1 ∧ (removeOne a e q).2 = (removeOne b e q).2 := by
  have hrow : rowAfterErase a q e = rowAfterErase b q e := by
    simp only [rowAfterErase, h.me]
  cases hg : a.effGet q with
  | none =>
    have hg' : b.effGet q = none := by rw [← h.effGet q]; exact hg
    simp only [removeOne, hg, hg']
    exact ⟨h, trivial⟩
  | some v =>
    have hg' : b.effGet q = some v := by rw [← h.effGet q]; exact hg
    by_cases hM : v = MULTI
    · subst hM
      by_cases hc : 2 ≤ (rowAfterErase a q e).card
      · rw [removeOne_still_multi hg hc, removeOne_still_multi hg' (by rw [← hrow]; exact hc)]
        exact ⟨h.me_erase _, rfl⟩
      · match hmin : (rowAfterErase a q e).min with
        | none =>
          have hmin' : (rowAfterErase b q e).min = none := by rw [← hrow]; exact hmin
          simp only [removeOne, hg, hg', if_pos rfl, if_neg hc,
            if_neg (show ¬ 2 ≤ (rowAfterErase b q e).card by rw [← hrow]; exact hc), hmin, hmin']
          exact ⟨(h.me_erase _).deletePair_congr q, rfl⟩
        | some last =>
          have hmin' : (rowAfterErase b q e).min = some last := by rw [← hrow]; exact hmin
          by_cases hm : a.m.get q = some last
          · rw [removeOne_demote_cancel hg hc hmin hm,
              removeOne_demote_cancel hg' (by rw [← hrow]; exact hc) hmin'
                (by rw [← h.m_get q]; exact hm)]
            refine ⟨⟨h.m_get, ?_, h.dm, h.mt, by rw [h.me], h.nrows, h.ncols⟩, rfl⟩
            intro r
            by_cases hr : r = q
            · rw [hr]; simp
            · rw [Layer.get_remove_ne hr, Layer.get_remove_ne hr]; exact h.dp_get r
          · rw [removeOne_demote_shadow hg hc hmin hm,
              removeOne_demote_shadow hg' (by rw [← hrow]; exact hc) hmin'
                (by rw [← h.m_get q]; exact hm)]
            refine ⟨⟨h.m_get, ?_, h.dm, h.mt, by rw [h.me], h.nrows, h.ncols⟩, rfl⟩
            intro r
            by_cases hr : r = q
            · rw [hr]; simp
            · rw [Layer.get_set_ne hr, Layer.get_set_ne hr]; exact h.dp_get r
    · by_cases he : v = e
      · rw [he] at hg hg'
        rw [removeOne_single hg (by rw [← he]; exact hM), removeOne_single hg' (by rw [← he]; exact hM)]
        exact ⟨h.deletePair_congr q, rfl⟩
      · simp only [removeOne, hg, hg', if_neg hM, if_neg he]
        exact ⟨h, trivial⟩

/-- The fold preserves observational equality, so the plan's tail and the
sequential tail stay related for the rest of the batch. -/
theorem TEquiv.foldl_removeOne {a b : Tensor} (h : TEquiv a b) (p : Pair) :
    ∀ es : List Nat, TEquiv (es.foldl (fun t' i => (removeOne t' i p).1) a)
      (es.foldl (fun t' i => (removeOne t' i p).1) b) := by
  intro es
  induction es generalizing a b with
  | nil => exact h
  | cons e rest ih => exact ih ((h.removeOne_congr e p).1)

/-! ## One pair, the whole batch -/

/-- Before any edge is named, the write phase is the identity: the plan holds no
deletions and no demote. -/
@[simp] theorem applyPlan_initState (t : Tensor) (p : Pair) :
    applyPlan t p (initState t p) = t := by
  have hdel : (initState t p).del = ∅ := rfl
  rcases hv : t.effGet p with _ | v
  · have hip : (initState t p).plan = .absent := by simp [initState, initPlan, hv]
    rw [applyPlan_absent hip, hdel]; simp
  · by_cases hM : v = MULTI
    · have hip : (initState t p).plan = .multi (t.meRow (key p)) := by
        simp [initState, initPlan, hv, hM]
      rw [applyPlan_multi hip, hdel]; simp
    · have hip : (initState t p).plan = .single v false := by simp [initState, initPlan, hv, hM]
      rw [applyPlan_single_clean hip, hdel]; simp

private theorem planFold_aux (h : Inv t) (p : Pair) :
    ∀ (es : List Nat) (s : PairState), Consistent t p s →
      TEquiv (applyPlan t p (es.foldl stepPlan s))
        (es.foldl (fun t' i => (removeOne t' i p).1) (applyPlan t p s))
      ∧ Consistent t p (es.foldl stepPlan s) := by
  intro es
  induction es with
  | nil => intro s hc; exact ⟨TEquiv.refl _, hc⟩
  | cons e rest ih =>
    intro s hc
    obtain ⟨hstep, hcons, _⟩ := applyPlan_stepPlan h hc e
    obtain ⟨hrest, hcrest⟩ := ih (stepPlan s e) hcons
    refine ⟨?_, hcrest⟩
    simp only [List.foldl_cons]
    -- the tails start from observationally equal tensors and stay related
    exact hrest.trans (hstep.foldl_removeOne p rest)

/-- **The read phase's plan, applied once, is the sequential fold.** -/
theorem tequiv_applyPlan_removeFold (h : Inv t) (p : Pair) (es : List Nat) :
    TEquiv (applyPlan t p (planFold t p es)) (removeFold t p es) := by
  have := (planFold_aux h p es (initState t p) (consistent_initState h)).1
  rwa [applyPlan_initState] at this

/-! ## What the batch reports

`Consistent` pins `reported` to the absorbing `emptied` plan, so the read phase's
report is not a separate claim to check: it is determined by the plan, and the
plan is determined by the fold. -/

theorem reported_iff_emptied {s : PairState} (hc : Consistent t p s) :
    s.reported = true ↔ s.plan = PairPlan.emptied := by
  cases hplan : s.plan <;> simp only [Consistent, hplan] at hc
  · exact ⟨fun hr => absurd (hc.2.2.2.2.symm.trans hr) (by simp), fun hc' => absurd hc' (by simp)⟩
  · rename_i i b
    rcases b with _ | _
    · exact ⟨fun hr => absurd (hc.2.2.2.symm.trans hr) (by simp), fun hc' => absurd hc' (by simp)⟩
    · exact ⟨fun hr => absurd (hc.2.2.2.2.symm.trans hr) (by simp), fun hc' => absurd hc' (by simp)⟩
  · exact ⟨fun _ => rfl, fun _ => hc.1⟩
  · exact ⟨fun hr => absurd (hc.2.2.symm.trans hr) (by simp), fun hc' => absurd hc' (by simp)⟩

theorem applyPlan_effGet_none_iff {s : PairState} (hc : Consistent t p s) :
    (applyPlan t p s).effGet p = none ↔ (s.plan = PairPlan.emptied ∨ s.plan = PairPlan.absent) := by
  cases hplan : s.plan <;> simp only [Consistent, hplan] at hc
  · rename_i r
    rw [applyPlan_multi hplan]
    simp only [me_only_effGet, hc.1]
    exact ⟨fun hcon => absurd hcon (by simp), fun hcon => by simp at hcon⟩
  · rename_i i b
    rcases b with _ | _
    · rw [applyPlan_single_clean hplan, hc.2.2.1]
      simp only [me_only_effGet, hc.1]
      exact ⟨fun hcon => absurd hcon (by simp), fun hcon => by simp at hcon⟩
    · rw [applyPlan_demote hplan]
      have : (applyPlan t p s).effGet p = some i := by
        rw [applyPlan_demote hplan]
        by_cases hm : t.m.get p = some i
        · simp only [hm, if_pos]
          exact (effGet_of_m (by simp) (by simpa using hc.2.1)).trans hm
        · simp only [hm, if_neg]; exact effGet_of_dp (by simp)
      rw [applyPlan_demote hplan] at this
      rw [this]
      exact ⟨fun hcon => absurd hcon (by simp), fun hcon => by simp at hcon⟩
  · rw [applyPlan_emptied hplan]
    exact ⟨fun _ => Or.inl rfl, fun _ => deletePair_effGet_self⟩
  · rw [applyPlan_absent hplan, hc.2.1]
    simp only [me_sdiff_empty]
    exact ⟨fun _ => by simp, fun _ => by simpa using hc.1⟩

/-- **The batch reports a pair exactly when it emptied one that was there.** -/
theorem reported_iff (h : Inv t) (p : Pair) (es : List Nat) :
    (planFold t p es).reported = true
      ↔ (t.effGet p ≠ none ∧ (removeFold t p es).effGet p = none) := by
  have hc : Consistent t p (planFold t p es) :=
    (planFold_aux h p es (initState t p) (consistent_initState h)).2
  have hteq := tequiv_applyPlan_removeFold h p es
  rw [reported_iff_emptied hc]
  constructor
  · intro hem
    refine ⟨?_, ?_⟩
    · simp only [Consistent, hem] at hc; exact hc.2
    · rw [← hteq.effGet p, (applyPlan_effGet_none_iff hc).mpr (Or.inl hem)]
  · rintro ⟨hne, hnone⟩
    rcases (applyPlan_effGet_none_iff hc).mp (by rw [hteq.effGet p]; exact hnone) with hem | hab
    · exact hem
    · simp only [Consistent, hab] at hc; exact absurd hc.1 hne

/-! ## Invariants -/

theorem inv_removeFold (h : Inv t) (es : List Nat) : Inv (removeFold t p es) := by
  induction es generalizing t with
  | nil => exact h
  | cons e rest ih => exact ih (removeOne_spec h).1

/-- **The batched path preserves every invariant**, inherited through [`TEquiv`]
rather than proved a second time. -/
theorem inv_applyPlan (h : Inv t) (es : List Nat) :
    Inv (applyPlan t p (planFold t p es)) :=
  (tequiv_applyPlan_removeFold h p es).inv (inv_removeFold h es)

/-- And it denotes the same multigraph, which is the statement a query sees. -/
theorem edgesAt_applyPlan (h : Inv t) (p : Pair) (es : List Nat) (q : Pair) :
    (applyPlan t p (planFold t p es)).edgesAt q = (removeFold t p es).edgesAt q :=
  (tequiv_applyPlan_removeFold h p es).edgesAt q

/-! ## Plans for distinct pairs commute

The write phase applies the plans in hash-map order. That this cannot matter is
the second half of the obligation, and it rests on each plan touching only its
own pair: one `me` row (distinct pairs have distinct keys, by `key_inj`), one
forward cell, one backward cell.

Every `applyPlan` has one of three shapes, and naming them turns the argument
into nine cases rather than twenty-five. -/

inductive Shape where
  | meOnly
  | demote (i : Nat)
  | emptied

def shapeOf (s : PairState) : Shape :=
  match s.plan with
  | .emptied => .emptied
  | .single i true => .demote i
  | _ => .meOnly

def applyShape (t : Tensor) (p : Pair) (D : Finset Nat) : Shape → Tensor
  | .meOnly => { t with me := t.me \ D.image (fun i => (key p, i)) }
  | .demote i =>
      let t1 : Tensor := { t with me := t.me \ D.image (fun j => (key p, j)) }
      if t.m.get p = some i then { t1 with dp := t1.dp.remove p }
      else { t1 with dp := t1.dp.set p i }
  | .emptied => deletePair { t with me := t.me \ D.image (fun i => (key p, i)) } p

theorem applyPlan_eq_applyShape (t : Tensor) (p : Pair) (s : PairState) :
    applyPlan t p s = applyShape t p s.del (shapeOf s) := by
  cases hp : s.plan with
  | multi r => rw [applyPlan_multi hp]; simp [applyShape, shapeOf, hp]
  | single i b =>
    rcases b with _ | _
    · rw [applyPlan_single_clean hp]; simp [applyShape, shapeOf, hp]
    · rw [applyPlan_demote hp]; simp [applyShape, shapeOf, hp]
  | emptied => rw [applyPlan_emptied hp]; simp [applyShape, shapeOf, hp]
  | absent => rw [applyPlan_absent hp]; simp [applyShape, shapeOf, hp]

/-- `m` is never written by deletion — the property the whole phase split relies
on, since the write phase reads `m` to decide whether a demote cancels. -/
@[simp] theorem applyShape_m (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).m = t.m := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

@[simp] theorem applyShape_nrows (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).nrows = t.nrows := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

@[simp] theorem applyShape_ncols (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).ncols = t.ncols := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

@[simp] theorem applyShape_me (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).me = t.me \ D.image (fun i => (key p, i)) := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

theorem applyShape_dp_get_ne {q : Pair} (hq : q ≠ p) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).dp.get q = t.dp.get q := by
  cases k with
  | meOnly => rfl
  | demote i =>
    by_cases hm : t.m.get p = some i <;>
      simp [applyShape, hm, Layer.get_remove_ne hq, Layer.get_set_ne hq]
  | emptied => simp [applyShape, deletePair, Layer.get_remove_ne hq]

theorem applyShape_dm_ne {q : Pair} (hq : q ≠ p) (D : Finset Nat) (k : Shape) :
    (q ∈ (applyShape t p D k).dm ↔ q ∈ t.dm) := by
  cases k with
  | meOnly => exact Iff.rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied =>
    by_cases hd : p ∈ t.m.dom <;>
      simp [applyShape, deletePair, hd, Finset.mem_insert, hq]

theorem applyShape_mt_ne {q : Pair} (hq : q ≠ (p.2, p.1)) (D : Finset Nat) (k : Shape) :
    (q ∈ (applyShape t p D k).mt ↔ q ∈ t.mt) := by
  cases k with
  | meOnly => exact Iff.rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => simp [applyShape, deletePair, Finset.mem_erase, hq]

/-- `me` deletions for distinct pairs land in distinct rows, so they commute.
This is where `key_inj` earns its place in the argument. -/
theorem me_sdiff_comm {me : Finset (Addr × Nat)} {k k' : Addr} (hk : k ≠ k') (D D' : Finset Nat) :
    (me \ D.image (fun i => (k, i))) \ D'.image (fun i => (k', i))
      = (me \ D'.image (fun i => (k', i))) \ D.image (fun i => (k, i)) := by
  ext x; simp only [Finset.mem_sdiff]; tauto

/-! ### Each component, named

Describing the three components a shape can touch turns commutation into three
small computations about operations at distinct points. -/

theorem applyShape_dp (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).dp =
      match k with
      | .meOnly => t.dp
      | .demote i => if t.m.get p = some i then t.dp.remove p else t.dp.set p i
      | .emptied => t.dp.remove p := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

theorem applyShape_dm (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).dm =
      match k with
      | .emptied => if p ∈ t.m.dom then insert p t.dm else t.dm
      | _ => t.dm := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

theorem applyShape_mt (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).mt =
      match k with
      | .emptied => t.mt.erase (p.2, p.1)
      | _ => t.mt := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, hm]
  | emptied => rfl

/-! ### Layer operations at distinct points commute -/

theorem Layer.ext' {α : Type} {L M : Layer α} (hd : L.dom = M.dom) (hv : L.val = M.val) :
    L = M := by cases L; cases M; simp_all

theorem Layer.remove_remove_comm {α : Type} (L : Layer α) (a b : Pair) :
    (L.remove a).remove b = (L.remove b).remove a := by
  simp [Layer.remove, Finset.erase_right_comm]

theorem Layer.set_set_comm {α : Type} (L : Layer α) {a b : Pair} (h : a ≠ b) (u v : α) :
    (L.set a u).set b v = (L.set b v).set a u := by
  refine Layer.ext' (Finset.insert_comm _ _ _) ?_
  funext q
  by_cases hqa : q = a
  · have hqb : q ≠ b := fun hc => h (hqa ▸ hc)
    simp [Layer.set, hqa, hqb, h]
  · by_cases hqb : q = b
    · simp [Layer.set, hqa, hqb, h, Ne.symm h]
    · simp [Layer.set, hqa, hqb]

theorem Layer.remove_set_comm {α : Type} (L : Layer α) {a b : Pair} (h : a ≠ b) (v : α) :
    (L.remove a).set b v = (L.set b v).remove a := by
  refine Layer.ext' ?_ rfl
  show insert b (L.dom.erase a) = (insert b L.dom).erase a
  ext q
  simp only [Finset.mem_insert, Finset.mem_erase]
  constructor
  · rintro (rfl | ⟨hqa, hq⟩)
    · exact ⟨Ne.symm h, Or.inl rfl⟩
    · exact ⟨hqa, Or.inr hq⟩
  · rintro ⟨hqa, rfl | hq⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨hqa, hq⟩

/-! ### The three component operations, and their commutation

Naming each component's update as a function of the *original* tensor is what
makes the nine-case argument nine one-liners: the decision each shape takes reads
`m`, which neither shape writes, so both orders take the same decisions. -/

def dpOp (m L : Layer Nat) (p : Pair) : Shape → Layer Nat
  | .meOnly => L
  | .demote i => if m.get p = some i then L.remove p else L.set p i
  | .emptied => L.remove p

def dmOp (mdom S : Finset Pair) (p : Pair) : Shape → Finset Pair
  | .emptied => if p ∈ mdom then insert p S else S
  | _ => S

def mtOp (S : Finset Pair) (p : Pair) : Shape → Finset Pair
  | .emptied => S.erase (p.2, p.1)
  | _ => S

theorem applyShape_dp' (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).dp = dpOp t.m t.dp p k := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, dpOp, hm]
  | emptied => rfl

theorem applyShape_dm' (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).dm = dmOp t.m.dom t.dm p k := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, dmOp, hm]
  | emptied => rfl

theorem applyShape_mt' (t : Tensor) (p : Pair) (D : Finset Nat) (k : Shape) :
    (applyShape t p D k).mt = mtOp t.mt p k := by
  cases k with
  | meOnly => rfl
  | demote i => by_cases hm : t.m.get p = some i <;> simp [applyShape, mtOp, hm]
  | emptied => rfl

theorem dpOp_comm (m L : Layer Nat) {p p' : Pair} (h : p ≠ p') (k k' : Shape) :
    dpOp m (dpOp m L p k) p' k' = dpOp m (dpOp m L p' k') p k := by
  cases k with
  | meOnly => cases k' <;> rfl
  | demote i =>
    cases k' with
    | meOnly => rfl
    | demote i' =>
      simp only [dpOp]
      split_ifs <;>
        first
          | exact Layer.remove_remove_comm _ _ _
          | exact Layer.remove_set_comm _ h _
          | exact (Layer.remove_set_comm _ (Ne.symm h) _).symm
          | exact Layer.set_set_comm _ h _ _
    | emptied =>
      simp only [dpOp]
      split_ifs <;>
        first
          | exact Layer.remove_remove_comm _ _ _
          | exact Layer.remove_set_comm _ h _
          | exact (Layer.remove_set_comm _ (Ne.symm h) _).symm
  | emptied =>
    cases k' with
    | meOnly => rfl
    | demote i' =>
      simp only [dpOp]
      split_ifs <;>
        first
          | exact Layer.remove_remove_comm _ _ _
          | exact Layer.remove_set_comm _ h _
          | exact (Layer.remove_set_comm _ (Ne.symm h) _).symm
    | emptied => exact Layer.remove_remove_comm _ _ _

theorem dmOp_comm (mdom S : Finset Pair) (p p' : Pair) (k k' : Shape) :
    dmOp mdom (dmOp mdom S p k) p' k' = dmOp mdom (dmOp mdom S p' k') p k := by
  cases k with
  | meOnly => cases k' <;> rfl
  | demote i => cases k' <;> rfl
  | emptied =>
    cases k' with
    | meOnly => rfl
    | demote i' => rfl
    | emptied =>
      simp only [dmOp]
      split_ifs <;> first | rfl | exact Finset.insert_comm _ _ _

theorem mtOp_comm (S : Finset Pair) (p p' : Pair) (k k' : Shape) :
    mtOp (mtOp S p k) p' k' = mtOp (mtOp S p' k') p k := by
  cases k with
  | meOnly => cases k' <;> rfl
  | demote i => cases k' <;> rfl
  | emptied =>
    cases k' with
    | meOnly => rfl
    | demote i' => rfl
    | emptied => exact Finset.erase_right_comm

/-- **Plans for distinct pairs commute.** The write phase may apply them in any
order, which is what licenses the Rust iterating a hash map. -/
theorem applyShape_comm {p p' : Pair} (hne : p ≠ p')
    (D D' : Finset Nat) (k k' : Shape) :
    TEquiv (applyShape (applyShape t p D k) p' D' k')
      (applyShape (applyShape t p' D' k') p D k) := by
  have hkey : key p ≠ key p' := fun hc => hne (key_inj hc)
  refine ⟨fun q => by simp only [applyShape_m], ?_, ?_, ?_, ?_,
    by simp only [applyShape_nrows], by simp only [applyShape_ncols]⟩
  · intro q
    simp only [applyShape_dp', applyShape_m]
    rw [dpOp_comm _ _ hne]
  · simp only [applyShape_dm', applyShape_m]
    rw [dmOp_comm]
  · simp only [applyShape_mt']
    rw [mtOp_comm]
  · simp only [applyShape_me]
    exact me_sdiff_comm hkey _ _

/-- The same statement about plans rather than shapes. -/
theorem applyPlan_comm {p p' : Pair} (hne : p ≠ p')
    (s s' : PairState) :
    TEquiv (applyPlan (applyPlan t p s) p' s') (applyPlan (applyPlan t p' s') p s) := by
  rw [applyPlan_eq_applyShape, applyPlan_eq_applyShape, applyPlan_eq_applyShape,
    applyPlan_eq_applyShape]
  exact applyShape_comm hne _ _ _ _

end Tensor
end FalkorDB
