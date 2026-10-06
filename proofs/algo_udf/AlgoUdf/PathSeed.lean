import AlgoUdf.PathDispatch
/-! # The BFS-seeded bound never empties the answer (run_path_algo :2628-2646)

For `pathCount ∈ {0, 1}` with a finite-weight BFS path within `maxCost`, the
enumeration starts with `bound = weight(BFS path)`. `bfs_seed_keeps_answer`
shows the BFS path is itself a qualifying path (`ext_of_simple`), so by
`dfs_k1` / `dfs_k0` the enumeration's result is non-empty. (The DFS `step` is a
function, so the final state of the run is unique.)
-/
namespace AlgoUdf.PathDispatch
open AlgoUdf AlgoUdf.Graph AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)

def wfold (g : G) (es : List Rel) (a : Nat) : Option Nat :=
  es.foldl (fun acc r => acc.bind fun x => (g.wt r.2.2).map (x + ·)) (some a)

theorem foldl_none (g : G) (es : List Rel) :
    es.foldl (fun acc r => acc.bind fun x => (g.wt r.2.2).map (x + ·)) none = none := by
  induction es with
  | nil => rfl
  | cons r t ih => simp [List.foldl_cons, ih]

theorem wfold_cons (g : G) (r : Rel) (rest : List Rel) (a : Nat) :
    wfold g (r :: rest) a = (g.wt r.2.2).bind (fun c => wfold g rest (a + c)) := by
  unfold wfold
  simp only [List.foldl_cons, Option.bind_some]
  cases g.wt r.2.2 with
  | none => simp [foldl_none]
  | some c => rfl

theorem bfsWeight_eq (g : G) (es : List Rel) : bfsWeight g es = wfold g es 0 := rfl

theorem costSum_cons (g : G) (r : Rel) (rest : List Rel) :
    costSum g (r :: rest) = g.cost r.2.2 + costSum g rest := by
  simp [costSum]

/-- A simple hop-walk with finite weights, within maxCost/maxLen, ending at
the target and not passing through it earlier, is a qualifying extension. -/
theorem ext_of_simple (g : G) (cfg : Cfg) :
    ∀ {u es ns v}, NWalk g u es ns v → es ≠ [] →
      ∀ pw pc d vis W, (∀ n ∈ ns, n ∉ vis) → ns.Nodup → wfold g es pw = some W →
        cfg.maxCost.all (pc + costSum g es ≤ ·) = true → d + es.length ≤ cfg.maxLen →
        cfg.tgt.all (· == v) = true → (∀ n ∈ ns.dropLast, cfg.tgt ≠ some n) →
        Ext g cfg (succsRev g u) pw pc d vis es W (pc + costSum g es) := by
  intro u es ns v h
  induction h with
  | nil => intro h; exact absurd rfl h
  | @cons u v1 w r rest ns' ha hrest ih =>
    intro _ pw pc d vis W hvis hnd hw hc hl ht hmid
    rw [wfold_cons] at hw
    cases hwr : g.wt r.2.2 with
    | none => rw [hwr] at hw; cases hw
    | some c =>
      rw [hwr] at hw; simp only [Option.bind_some] at hw
      have hmem : (r, v1) ∈ succsRev g u := by simpa [succsRev] using (mem_succs g u r v1).mpr ha
      have hv1 : v1 ∉ vis := hvis v1 (by simp)
      have hcost1 : cfg.maxCost.all (pc + g.cost r.2.2 ≤ ·) = true := by
        cases hm : cfg.maxCost with
        | none => rfl
        | some m => rw [hm] at hc; rw [costSum_cons] at hc; simp at hc ⊢; omega
      cases rest with
      | nil =>
        cases hrest
        simp only [wfold, List.foldl_nil, Option.some.injEq] at hw; subst hw
        simp only [costSum, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
        exact .last hmem hv1 hwr hcost1 (by simpa using hl) ht
      | cons r2 rest2 =>
        have hns : ns' ≠ [] := by
          intro h; subst h; have := hrest.length; simp at this
        have hnt : (cfg.tgt.all (· == v1) && cfg.tgt.isSome) = false := by
          have : cfg.tgt ≠ some v1 := hmid v1 (by
            cases ns' with
            | nil => exact absurd rfl hns
            | cons a t => simp)
          cases htg : cfg.tgt with
          | none => rfl
          | some t => rw [htg] at this; simp at this ⊢; exact fun h => this h
        have hsub := ih (by simp) (pw + c) (pc + g.cost r.2.2) (d + 1) (v1 :: vis) W
          (by
            intro n hn hn'
            simp only [List.mem_cons] at hn'
            rcases hn' with rfl | hn'
            · exact (List.nodup_cons.mp hnd).1 hn
            · exact hvis n (by simp [hn]) hn')
          (List.nodup_cons.mp hnd).2 hw
          (by rw [costSum_cons] at hc; rw [Nat.add_assoc]; exact hc)
          (by simp at hl ⊢; omega) ht
          (by
            intro n hn
            apply hmid n
            cases ns' with
            | nil => exact absurd rfl hns
            | cons a t => simp only [List.dropLast_cons₂]; exact List.mem_cons_of_mem _ hn)
        rw [costSum_cons, ← Nat.add_assoc]
        exact .more hmem hv1 hwr hcost1 hnt (by simp at hl; omega) hsub

theorem nodup_dropLast_last {ns : List Nat} {t : Nat} (hnd : ns.Nodup) (hl : ns.getLast? = some t) :
    t ∉ ns.dropLast := by
  intro hm
  have hne : ns ≠ [] := by intro h; subst h; simp at hl
  have hsplit : ns = ns.dropLast ++ [t] := by
    have h1 := List.getLast?_eq_getLast hne
    rw [hl] at h1; cases h1
    exact (List.dropLast_concat_getLast hne).symm
  rw [hsplit] at hnd
  exact (List.nodup_append.mp hnd).2.2 t hm t (by simp) rfl

theorem nwalk_getLast {g : G} {u es ns v} (h : NWalk g u es ns v) (hne : es ≠ []) :
    ns.getLast? = some v := by
  induction h with
  | nil => exact absurd rfl hne
  | @cons _ v1 _ _ rest ns' _ hrest ih =>
    cases rest with
    | nil => cases hrest; rfl
    | cons r2 rest2 =>
      have := ih (by simp)
      cases ns' with
      | nil => simp at this
      | cons a t => rw [List.getLast?_cons_cons]; exact this

/-- **The BFS-seeded bound keeps the answer** (pathCount = 1). -/
theorem bfs_seed_keeps_answer_k1 (g : G) (c : PCfg) (t : Nat) (ht : c.tgt = some t)
    (hne : c.src ≠ t) (hk : c.k = 1) (es : List Rel)
    (hb : BfsBound.bfsFindBound g t c.src c.maxLen = some (some es)) (W : Nat)
    (hW : bfsWeight g es = some W) (hc : c.maxCost.all (bfsCost g es ≤ ·) = true) :
    ∃ F, Star g (dcfg c) (init g (dcfg c) (some W)) F ∧ step g (dcfg c) F = none ∧ F.results ≠ [] := by
  obtain ⟨F, h1, h2, h3, _⟩ := dfs_k1 g (dcfg c) hk (some W)
  refine ⟨F, h1, h2, fun hr => ?_⟩
  obtain ⟨ns, hnw, hlen1, hlen2, hnd⟩ := (BfsBound.bfs_sound g c.src t c.maxLen hne).2 es hb
  have hne' : es ≠ [] := by intro h; subst h; simp at hlen1
  have hlast := nwalk_getLast hnw hne'
  have hq : Qual g c es W (0 + costSum g es) :=
    ext_of_simple g (dcfg c) hnw hne' 0 0 0 [c.src] W
      (by intro n hn hn'; simp at hn'; subst hn'; exact (List.nodup_cons.mp hnd).1 hn)
      (List.nodup_cons.mp hnd).2 (by rw [← bfsWeight_eq]; exact hW)
      (by cases hm : c.maxCost <;> simp_all [dcfg, bfsCost] <;> exact of_decide_eq_true hc) (by simpa [dcfg] using hlen2) (by simp [dcfg, ht])
      (by
        intro n hn he
        simp [dcfg, ht] at he; subst he
        exact nodup_dropLast_last (List.nodup_cons.mp hnd).2 hlast hn)
  exact h3 hr es W _ hq (by simp [BLe])

/-- Same for pathCount = 0. -/
theorem bfs_seed_keeps_answer_k0 (g : G) (c : PCfg) (t : Nat) (ht : c.tgt = some t)
    (hne : c.src ≠ t) (hk : c.k = 0) (es : List Rel)
    (hb : BfsBound.bfsFindBound g t c.src c.maxLen = some (some es)) (W : Nat)
    (hW : bfsWeight g es = some W) (hc : c.maxCost.all (bfsCost g es ≤ ·) = true) :
    ∃ F, Star g (dcfg c) (init g (dcfg c) (some W)) F ∧ step g (dcfg c) F = none ∧ F.results ≠ [] := by
  obtain ⟨F, h1, h2, h3, _⟩ := dfs_k0 g (dcfg c) hk (some W)
  refine ⟨F, h1, h2, fun hr => ?_⟩
  obtain ⟨ns, hnw, hlen1, hlen2, hnd⟩ := (BfsBound.bfs_sound g c.src t c.maxLen hne).2 es hb
  have hne' : es ≠ [] := by intro h; subst h; simp at hlen1
  have hlast := nwalk_getLast hnw hne'
  have hq : Qual g c es W (0 + costSum g es) :=
    ext_of_simple g (dcfg c) hnw hne' 0 0 0 [c.src] W
      (by intro n hn hn'; simp at hn'; subst hn'; exact (List.nodup_cons.mp hnd).1 hn)
      (List.nodup_cons.mp hnd).2 (by rw [← bfsWeight_eq]; exact hW)
      (by cases hm : c.maxCost <;> simp_all [dcfg, bfsCost] <;> exact of_decide_eq_true hc) (by simpa [dcfg] using hlen2) (by simp [dcfg, ht])
      (by
        intro n hn he
        simp [dcfg, ht] at he; subst he
        exact nodup_dropLast_last (List.nodup_cons.mp hnd).2 hlast hn)
  exact h3 hr es W _ hq (by simp [BLe])

end AlgoUdf.PathDispatch
