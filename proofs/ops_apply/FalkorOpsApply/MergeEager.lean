import FalkorOpsApply.Merge
/-
# MERGE matches every input row before it creates anything (#3067, fixes #3039)

`Merge.lean` models `MergeOp::next` as it was at ca36bfa15: the match sub-plan ran once per
INPUT BATCH, so a row saw the creations (and `ON CREATE SET` rewrites) of earlier batches
but not of earlier rows of its own batch. That model is kept there as history
(`rustBatch`/`rustMerge`/`phaseA`).

Since #3067 (b47d6d6a3) `MergeOp::match_all` (merge.rs:260-297) first consumes the whole
child and runs the match sub-plan for every row, against the graph as it is before any
creation; `next` (merge.rs:329-441) then walks the matched rows in input order. For each row:
no match → `do_create_fallback` (cache hit = the node an earlier row created); matches →
the first inline, the rest queued, and `drain_pending` runs right after the row (merge.rs:434-437,
inside the `while`), so the row's matches are emitted in order before the next row.

| here | there |
| --- | --- |
| `matchAll` | `MergeOp::match_all` (merge.rs:260), per batch, `matched.extend` |
| `rowMain` | one iteration of the `while` in `MergeOp::next` (merge.rs:350-437) |
| `mainMerge` | `MergeOp::next` run to exhaustion |

Theorems: `mainMerge_batch_independent` (the result does not depend on how the input is cut
into batches), `mainMerge_noSet_eq_ref` (without SET clauses: same graph, same ids and the
SAME ROW ORDER as the per-row openCypher reference), `fixed3067_batch_independent` (the
filed repro now gives the same 2 nodes whether rows 1 and 2 share a batch or not, as C).
-/
namespace Merge

variable {α : Type}

/-- `match_all`: each input batch's rows with their matches in the graph `g`, the batches
concatenated in order. -/
def matchAll (cl : Clause α) (g : Graph) (bs : List (List α)) : List (α × List Node) :=
  bs.flatMap (fun b => b.map (fun r => (r, matchesK g (cl.key r))))

/-- One row of the `while` in `MergeOp::next` (merge.rs:350-437): create fallback when it
has no match; otherwise `ON MATCH` on the first match, the rest queued and drained at once
(`drain_pending` = `phaseB`), unless the all-bound shortcut keeps only the first. -/
def rowMain (cl : Clause α) (allBound : Bool) (st : SState (α := α)) :
    α × List Node → SState (α := α)
  | (r, []) =>
    let (st', o) := createFallback cl st r
    { st' with out := st'.out ++ [o] }
  | (r, m :: ms) =>
    let st1 := { st with g := applySet cl.onMatch r st.g m.id, out := st.out ++ [(r, some m.id)] }
    if allBound || ms.isEmpty then st1 else phaseB cl st1 [(r, ms)]

/-- `MergeOp::next` after #3067: match everything against the start graph, then process the
rows in order. -/
def mainMerge (cl : Clause α) (allBound : Bool) (st : SState (α := α)) (bs : List (List α)) :
    SState (α := α) :=
  (matchAll cl st.g bs).foldl (rowMain cl allBound) st

theorem matchAll_eq (cl : Clause α) (g : Graph) (bs : List (List α)) :
    matchAll cl g bs = bs.flatten.map (fun r => (r, matchesK g (cl.key r))) := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp only [matchAll, List.flatMap_cons, List.flatten_cons, List.map_append] at ih ⊢; rw [ih]

/-- **The result does not depend on the batch boundaries** (the pre-#3067 model did). -/
theorem mainMerge_batch_independent (cl : Clause α) (allBound : Bool) (st : SState (α := α))
    (bs : List (List α)) : mainMerge cl allBound st bs = mainMerge cl allBound st [bs.flatten] := by
  simp only [mainMerge, matchAll_eq, List.flatten_cons, List.flatten_nil, List.append_nil]

/-- Per row, the new `MergeOp::next` step agrees with the openCypher step (no SET). -/
theorem rowMain_ref (cl : Clause α) (hs : NoSet cl) (g0 : Graph) (r : α)
    (st : SState (α := α)) (rst : RState (α := α))
    (hJ : J cl g0 st.g st.cache) (hg : rst.g = st.g) (hn : rst.next = st.next)
    (hlt : ∀ n ∈ st.g, n.id < st.next) :
    ∃ d, (refRow cl rst r).g = (rowMain cl false st (r, matchesK g0 (cl.key r))).g ∧
      (refRow cl rst r).next = (rowMain cl false st (r, matchesK g0 (cl.key r))).next ∧
      J cl g0 (rowMain cl false st (r, matchesK g0 (cl.key r))).g
        (rowMain cl false st (r, matchesK g0 (cl.key r))).cache ∧
      (∀ n ∈ (rowMain cl false st (r, matchesK g0 (cl.key r))).g,
        n.id < (rowMain cl false st (r, matchesK g0 (cl.key r))).next) ∧
      (rowMain cl false st (r, matchesK g0 (cl.key r))).out = st.out ++ d ∧
      (refRow cl rst r).out = rst.out ++ d := by
  obtain ⟨hoc, hom⟩ := hs
  obtain ⟨created, hgc, hcr⟩ := hJ.ext
  cases hm0 : matchesK g0 (cl.key r) with
  | nil =>
    cases hlk : lookup st.cache (cl.op, cl.key r) with
    | none =>
      have hcur : matchesK st.g (cl.key r) = [] := (hJ.fresh _ hm0).1 hlk
      have href : refRow cl rst r =
          ⟨rst.g ++ [⟨rst.next, cl.key r⟩], rst.next + 1, rst.out ++ [(r, some rst.next)]⟩ := by
        simp only [refRow, hg, hcur, hoc, applySet]
      have hrow : rowMain cl false st (r, []) =
          ⟨st.g ++ [⟨st.next, cl.key r⟩], st.next + 1,
            ((cl.op, cl.key r), (cl.var, st.next)) :: st.cache,
            st.out ++ [(r, some st.next)]⟩ := by
        simp only [rowMain, createFallback, hlk, hoc, applySet]
      rw [href, hrow]
      refine ⟨[(r, some st.next)], by simp [hg, hn], by simp [hn], ?_, ?_, rfl, by simp [hn]⟩
      · refine ⟨⟨created ++ [⟨st.next, cl.key r⟩], by simp [hgc], ?_⟩, ?_, ?_⟩
        · intro n hn'
          simp only [List.mem_append, List.mem_singleton] at hn'
          rcases hn' with hn' | rfl
          · exact hcr n hn'
          · exact hm0
        · intro h v id hl
          by_cases hh : h = cl.key r
          · subst hh; rw [lookup_cons_self] at hl; cases hl; simp
          · rw [lookup_cons_other _ _ _ _ _ (by simpa using hh)] at hl
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
          · rw [lookup_cons_other _ _ _ _ _ (by simpa using hh)]
            have hne : ¬ (cl.key r = k) := fun e => hh e.symm
            refine ⟨fun hl => ?_, fun v id hl => ?_⟩
            · rw [matches_append, (hJ.fresh k hk).1 hl, matches_single]; simp [hne]
            · rw [matches_append, (hJ.fresh k hk).2 v id hl, matches_single]; simp [hne]
      · intro n hn'
        simp only [List.mem_append, List.mem_singleton] at hn'
        rcases hn' with hn' | rfl
        · show n.id < st.next + 1
          have := hlt n hn'; omega
        · simp
    | some cv =>
      obtain ⟨v, id⟩ := cv
      have hcur : matchesK st.g (cl.key r) = [⟨id, cl.key r⟩] := (hJ.fresh _ hm0).2 v id hlk
      have hv : v = cl.var := (hJ.cache_ok _ v id hlk).1
      have href : refRow cl rst r = ⟨rst.g, rst.next, rst.out ++ [(r, some id)]⟩ := by
        simp only [refRow, hg, hcur, hom]
        simp [applySet]
      have hrow : rowMain cl false st (r, []) =
          { st with out := st.out ++ [(r, some id)] } := by
        simp only [rowMain, createFallback, hlk, hom, applySet, bindCached, hv,
          beq_self_eq_true, ite_true]
      rw [href, hrow]
      exact ⟨[(r, some id)], by simp [hg], by simp [hn], hJ, hlt, rfl, rfl⟩
  | cons m ms =>
    have hcur : matchesK st.g (cl.key r) = m :: ms := by
      rw [hgc, matches_append, hm0]
      have : matchesK created (cl.key r) = [] := by
        simp only [matchesK, List.filter_eq_nil_iff, beq_iff_eq]
        intro n hn' hk
        have := hcr n hn'; rw [hk, hm0] at this; cases this
      simp [this]
    have href : refRow cl rst r =
        ⟨rst.g, rst.next, rst.out ++ (m :: ms).map (fun m => (r, some m.id))⟩ := by
      simp only [refRow, hg, hcur, hom]
      simp [foldl_applySet_none, applySet, foldl_const_self]
    have hrow : rowMain cl false st (r, m :: ms) =
        { st with out := st.out ++ (m :: ms).map (fun m => (r, some m.id)) } := by
      simp only [rowMain, hom, applySet, Bool.false_or]
      split
      · rename_i h; simp at h; subst h; simp
      · rw [phaseB_noSet _ ⟨hoc, hom⟩]; simp [List.append_assoc]
    rw [href, hrow]
    exact ⟨_, by simp [hg], by simp [hn], hJ, hlt, rfl, rfl⟩

theorem mainRows_ref (cl : Clause α) (hs : NoSet cl) (g0 : Graph) :
    ∀ (rows : List α) (st : SState (α := α)) (rst : RState (α := α)),
      J cl g0 st.g st.cache → rst.g = st.g → rst.next = st.next →
      (∀ n ∈ st.g, n.id < st.next) →
      let res := (rows.map (fun r => (r, matchesK g0 (cl.key r)))).foldl (rowMain cl false) st
      let ref := rows.foldl (refRow cl) rst
      ref.g = res.g ∧ ref.next = res.next ∧ J cl g0 res.g res.cache ∧
      (∀ n ∈ res.g, n.id < res.next) ∧
      ∃ o, res.out = st.out ++ o ∧ ref.out = rst.out ++ o := by
  intro rows
  induction rows with
  | nil =>
    intro st rst hJ hg hn hlt
    exact ⟨hg, hn, hJ, hlt, [], by simp, by simp⟩
  | cons r rows ih =>
    intro st rst hJ hg hn hlt
    obtain ⟨d, e1, e2, e3, e4, e5, e6⟩ := rowMain_ref cl hs g0 r st rst hJ hg hn hlt
    simp only [List.map_cons, List.foldl_cons]
    obtain ⟨a1, a2, a3, a4, o, b1, b2⟩ :=
      ih (rowMain cl false st (r, matchesK g0 (cl.key r))) (refRow cl rst r) e3 e1 e2 e4
    exact ⟨a1, a2, a3, a4, d ++ o, by rw [b1, e5, List.append_assoc],
      by rw [b2, e6, List.append_assoc]⟩

/-- **MERGE without SET clauses matches openCypher after #3067, for every batching and with
the rows in the reference's order.** Same graph, same next id; unlike the pre-#3067 model
(`rust_eq_ref_noSet`, output a permutation) the emitted rows are EXACTLY the reference's,
because each row's remaining matches are drained before the next row. -/
theorem mainMerge_noSet_eq_ref (cl : Clause α) (hs : NoSet cl) (st : SState (α := α))
    (hG : G cl st) (bs : List (List α)) :
    let R := mainMerge cl false st bs
    let F := bs.flatten.foldl (refRow cl) (⟨st.g, st.next, []⟩ : RState (α := α))
    R.g = F.g ∧ R.next = F.next ∧ R.out = st.out ++ F.out := by
  have hA := mainRows_ref cl hs st.g bs.flatten st ⟨st.g, st.next, []⟩ (J_init cl st hG) rfl rfl hG.2
  simp only at hA
  obtain ⟨a1, a2, _, _, o, b1, b2⟩ := hA
  have hm : mainMerge cl false st bs =
      (bs.flatten.map (fun r => (r, matchesK st.g (cl.key r)))).foldl (rowMain cl false) st := by
    simp only [mainMerge, matchAll_eq]
  simp only [hm]
  refine ⟨a1.symm, a2.symm, ?_⟩
  rw [b1]; simp at b2; rw [b2]

/-! ### The filed repro (#3039) -/

/-- **#3067 fixes it**: the boundary no longer matters; both give 2 nodes, as C does (and as
the issue's `UNWIND range(1,N)` repro does for every N). -/
theorem fixed3067_batch_independent :
    (mainMerge clCase false st0 [[1, 2]]).g.length = 2 ∧
    (mainMerge clCase false st0 [[1], [2]]).g.length = 2 := by decide

end Merge
