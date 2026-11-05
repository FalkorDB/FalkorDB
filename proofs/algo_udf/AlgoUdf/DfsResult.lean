import AlgoUdf.DfsExplore
/-! # What `enumerate_paths` returns for `pathCount` 1 and 0

* `step_bound_mono` — for `pathCount ∈ {0, 1}` the pruning bound never rises.
* `dfs_complete` — the loop terminates and every qualifying path of weight ≤
  the final bound was handed to `record_found_path`.
* `dfs_k1` — `pathCount = 1`: an empty result means no qualifying path of
  weight ≤ the initial bound; otherwise the single result is a qualifying path
  that no qualifying path beats under `cmp_found_path` (weight, cost, hops).
* `dfs_k0` — `pathCount = 0`: the results are exactly the qualifying paths of
  minimum weight (each kept path qualifies, all have that weight, every
  qualifying path of that weight is kept).
-/
namespace AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

theorem fpLt_iff (a b : FP) : fpLt a b = true ↔
    a.w < b.w ∨ (a.w = b.w ∧ (a.c < b.c ∨ (a.c = b.c ∧ a.es.length < b.es.length))) := by
  simp [fpLt]

theorem fpLt_irrefl (a : FP) : fpLt a a = false := by
  cases h : fpLt a a with
  | false => rfl
  | true => rw [fpLt_iff] at h; omega

theorem fpLt_trans {a b c : FP} (h1 : fpLt a b = true) (h2 : fpLt b c = true) : fpLt a c = true := by
  rw [fpLt_iff] at *; omega

theorem recordB_bound_le (res : List FP) (f : FP) (k : Nat) (b : Option Nat) (hk : k ≤ 1)
    (hf : b.any (· < f.w) = false) : BoundLe (recordB res f k b).2 b := by
  have fle : BoundLe (some f.w) b := by
    intro y hy; subst hy; simp at hf; exact ⟨f.w, rfl, hf⟩
  unfold recordB
  split
  · split
    · exact fle
    · split
      · exact fle
      · exact .refl _
  · rw [if_pos (by omega)]
    split
    · exact fle
    · split
      · exact .refl _
      · split
        · exact fle
        · exact .refl _

theorem step_bound_mono (g : G) (cfg : Cfg) (hk : cfg.k ≤ 1) : BoundMono g cfg := by
  intro st st' _ hs
  unfold step at hs
  try dsimp only at hs
  split at hs
  · cases hs
  · split at hs
    · split at hs
      · cases hs; exact .refl _
      · split at hs
        · cases hs; exact .refl _
        · split at hs
          · cases hs; exact .refl _
          · rename_i hb
            split at hs
            · cases hs; exact .refl _
            · cases hs
              dsimp only
              split
              · exact recordB_bound_le _ _ _ _ hk (by simpa using hb)
              · exact .refl _
    · split at hs
      · cases hs
      · cases hs; exact .refl _

/-- **Completeness and termination of `enumerate_paths`.** -/
theorem dfs_complete (g : G) (cfg : Cfg) (hm : BoundMono g cfg) (b : Option Nat) :
    ∃ F, Star g cfg (init g cfg b) F ∧ step g cfg F = none ∧
      ∀ es W C, Ext g cfg (succsRev g cfg.src) 0 0 0 [cfg.src] es W C → BLe W F.bound →
        (⟨es, W, C⟩ : FP) ∈ F.log := by
  by_cases h0 : 0 < cfg.maxLen
  · obtain ⟨F, h1, h2, _, h4⟩ := explore hm _ (succsRev g cfg.src) (init g cfg b) _ [] (inv_init g cfg b)
      rfl (by simp [init, h0]) rfl
    refine ⟨F, h1, step_root_done h2 (Or.inr rfl), ?_⟩
    intro es W C hext hW
    have := h4 es W C (by simpa [init] using hext) hW
    simpa [pathOf, init] using this
  · refine ⟨init g cfg b, .refl _, step_root_done rfl (Or.inl (by simp [init, h0])), ?_⟩
    intro es W C hext _
    cases hext <;> omega

/-- Every step either leaves `(results, bound, log)` alone or records one path
`f` that passed the `nw > bound` test. -/
theorem step_record {g : G} {cfg : Cfg} {st st' : DS} (hs : step g cfg st = some st') :
    (st'.results = st.results ∧ st'.bound = st.bound ∧ st'.log = st.log) ∨
    ∃ f : FP, st.bound.any (· < f.w) = false ∧ st'.results = (recordB st.results f cfg.k st.bound).1 ∧
      st'.bound = (recordB st.results f cfg.k st.bound).2 ∧ st'.log = st.log ++ [f] := by
  cases hfs : st.frames with
  | nil => simp [step, hfs] at hs
  | cons fr below =>
    have back : (fr.lv = none ∨ fr.lv = some []) →
        (st'.results = st.results ∧ st'.bound = st.bound ∧ st'.log = st.log) := by
      intro hlv
      by_cases hb : below = []
      · subst hb; rw [step_root_done hfs hlv] at hs; cases hs
      · rw [step_back hfs hlv hb] at hs; cases hs; exact ⟨rfl, rfl, rfl⟩
    cases hlv : fr.lv with
    | none => exact Or.inl (back (Or.inl hlv))
    | some L =>
      cases L with
      | nil => exact Or.inl (back (Or.inr hlv))
      | cons p rest =>
        obtain ⟨r, x⟩ := p
        have skip : (st.onPath x = true ∨ g.wt r.2.2 = none ∨
            (∃ c, g.wt r.2.2 = some c ∧ (st.bound.any (· < fr.pw + c) = true ∨
              cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = true))) →
            (st'.results = st.results ∧ st'.bound = st.bound ∧ st'.log = st.log) := by
          intro h; rw [step_skip hfs hlv h] at hs; cases hs; exact ⟨rfl, rfl, rfl⟩
        by_cases hon : st.onPath x = true
        · exact Or.inl (skip (Or.inl hon))
        cases hw : g.wt r.2.2 with
        | none => exact Or.inl (skip (Or.inr (Or.inl hw)))
        | some c =>
          by_cases hb : st.bound.any (· < fr.pw + c) = true
          · exact Or.inl (skip (Or.inr (Or.inr ⟨c, hw, Or.inl hb⟩)))
          by_cases hmc : cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = true
          · exact Or.inl (skip (Or.inr (Or.inr ⟨c, hw, Or.inr hmc⟩)))
          rw [step_push hfs hlv (by simpa using hon) hw (by simpa using hb) (by simpa using hmc)] at hs
          cases hs
          by_cases hat : cfg.tgt.all (· == x) = true
          · right
            refine ⟨⟨pathOf ({ fr with lv := some rest } :: below) ++ [r], fr.pw + c, fr.pc + g.cost r.2.2⟩,
              by simpa using hb, ?_, ?_, ?_⟩ <;> simp [hat]
          · left; simp [hat]

/-! ## pathCount = 1 -/

def K1Inv (b0 : Option Nat) (st : DS) : Prop :=
  (st.results = [] ∧ st.log = [] ∧ st.bound = b0) ∨
  (∃ best, st.results = [best] ∧ best ∈ st.log ∧ (∀ f ∈ st.log, fpLt f best = false) ∧
    st.bound = some best.w)

theorem k1_step {g : G} {cfg : Cfg} (hk : cfg.k = 1) {b0 : Option Nat} {st st' : DS}
    (I : K1Inv b0 st) (hs : step g cfg st = some st') : K1Inv b0 st' := by
  rcases step_record hs with ⟨h1, h2, h3⟩ | ⟨f, hbf, h1, h2, h3⟩
  · unfold K1Inv; rw [h1, h2, h3]; exact I
  · unfold K1Inv; rw [h1, h2, h3]
    simp only [recordB, hk, if_true]
    rcases I with ⟨r1, r2, _⟩ | ⟨best, r1, r2, r3, r4⟩
    · rw [r1, r2]; right; exact ⟨f, rfl, by simp, by simp [fpLt_irrefl], rfl⟩
    · rw [r1]
      dsimp only
      by_cases hlt : fpLt f best = true
      · rw [if_pos hlt]
        right; refine ⟨f, rfl, by simp, ?_, rfl⟩
        intro f' hf
        simp only [List.mem_append, List.mem_singleton] at hf
        rcases hf with hf | rfl
        · cases hc : fpLt f' f with
          | false => rfl
          | true => have := r3 f' hf; rw [fpLt_trans hc hlt] at this; cases this
        · exact fpLt_irrefl _
      · rw [if_neg hlt]
        right; refine ⟨best, rfl, List.mem_append_left _ r2, ?_, r4⟩
        intro f' hf
        simp only [List.mem_append, List.mem_singleton] at hf
        rcases hf with hf | rfl
        · exact r3 f' hf
        · simpa using hlt

theorem k1_star {g : G} {cfg : Cfg} (hk : cfg.k = 1) {b0 : Option Nat} {st : DS}
    (h : Star g cfg (init g cfg b0) st) : K1Inv b0 st := by
  generalize hi : init g cfg b0 = s0 at h
  induction h with
  | refl => subst hi; left; exact ⟨rfl, rfl, rfl⟩
  | tail _ hs ih => exact k1_step hk ih hs

/-- **`pathCount = 1`**: the enumeration's answer is a `cmp_found_path`-minimum
over *all* qualifying paths, and it is empty only if no qualifying path weighs at
most the seed bound. -/
theorem dfs_k1 (g : G) (cfg : Cfg) (hk : cfg.k = 1) (b0 : Option Nat) :
    ∃ F, Star g cfg (init g cfg b0) F ∧ step g cfg F = none ∧
      (F.results = [] → ∀ es W C, Ext g cfg (succsRev g cfg.src) 0 0 0 [cfg.src] es W C → ¬ BLe W b0) ∧
      (∀ best, F.results = [best] → Good g cfg best ∧
        ∀ es W C, Ext g cfg (succsRev g cfg.src) 0 0 0 [cfg.src] es W C → fpLt ⟨es, W, C⟩ best = false) := by
  obtain ⟨F, h1, h2, h3⟩ := dfs_complete g cfg (step_bound_mono g cfg (by omega)) b0
  refine ⟨F, h1, h2, ?_, ?_⟩
  · intro hr es W C hext hW
    rcases k1_star hk h1 with ⟨_, hl, hb⟩ | ⟨best, hr', _⟩
    · have := h3 es W C hext (by rw [hb]; exact hW); rw [hl] at this; cases this
    · rw [hr] at hr'; cases hr'
  · intro best hr
    refine ⟨dfs_sound g cfg b0 F h1 best (by rw [hr]; simp), ?_⟩
    intro es W C hext
    rcases k1_star hk h1 with ⟨hr', _⟩ | ⟨best', hr', _, h4, hb⟩
    · rw [hr] at hr'; cases hr'
    · rw [hr] at hr'; cases hr'
      by_cases hle : W ≤ best.w
      · exact h4 _ (h3 es W C hext (by rw [hb]; simp [BLe, hle]))
      · cases hc : fpLt ⟨es, W, C⟩ best with
        | false => rfl
        | true => rw [fpLt_iff] at hc; simp at hc; omega

/-! ## pathCount = 0 -/

def K0Inv (b0 : Option Nat) (st : DS) : Prop :=
  (st.results = [] ∧ st.log = [] ∧ st.bound = b0) ∨
  (∃ m, st.results ≠ [] ∧ st.bound = some m ∧ (∀ f ∈ st.results, f ∈ st.log ∧ f.w = m) ∧
    (∀ f ∈ st.log, m ≤ f.w) ∧ (∀ f ∈ st.log, f.w = m → f ∈ st.results))

theorem k0_step {g : G} {cfg : Cfg} (hk : cfg.k = 0) {b0 : Option Nat} {st st' : DS}
    (I : K0Inv b0 st) (hs : step g cfg st = some st') : K0Inv b0 st' := by
  rcases step_record hs with ⟨h1, h2, h3⟩ | ⟨f, hbf, h1, h2, h3⟩
  · unfold K0Inv; rw [h1, h2, h3]; exact I
  · unfold K0Inv; rw [h1, h2, h3]
    simp only [recordB, hk, show (0 : Nat) ≠ 1 by decide, if_false, if_true]
    rcases I with ⟨r1, r2, _⟩ | ⟨m, hne, hb, hres, hlog, hall⟩
    · rw [r1, r2]; right
      refine ⟨f.w, by simp, rfl, ?_, ?_, ?_⟩ <;> simp
    · obtain ⟨best, rest, hbr⟩ : ∃ best rest, st.results = best :: rest := by
        cases h : st.results with
        | nil => exact absurd h hne
        | cons a t => exact ⟨a, t, rfl⟩
      have hbm : best.w = m := (hres best (by rw [hbr]; simp)).2
      rw [hb] at hbf
      simp at hbf
      rw [hbr]; dsimp only
      rw [if_neg (by omega)]
      by_cases hlt : f.w < best.w
      · rw [if_pos hlt]
        right
        refine ⟨f.w, by simp, rfl, ?_, ?_, ?_⟩
        · intro f' hf; simp at hf; subst hf; simp
        · intro f' hf
          simp only [List.mem_append, List.mem_singleton] at hf
          rcases hf with hf | rfl
          · have := hlog f' hf; omega
          · exact Nat.le_refl _
        · intro f' hf hfw
          simp only [List.mem_append, List.mem_singleton] at hf
          rcases hf with hf | rfl
          · have := hlog f' hf; omega
          · simp
      · rw [if_neg hlt]
        have heq : f.w = m := by omega
        right
        refine ⟨m, by simp, hb, ?_, ?_, ?_⟩
        · intro f' hf
          rw [← hbr] at hf
          simp only [List.mem_append, List.mem_singleton] at hf
          rcases hf with hf | rfl
          · exact ⟨List.mem_append_left _ (hres f' hf).1, (hres f' hf).2⟩
          · exact ⟨by simp, heq⟩
        · intro f' hf
          simp only [List.mem_append, List.mem_singleton] at hf
          rcases hf with hf | rfl
          · exact hlog f' hf
          · omega
        · intro f' hf hfw
          rw [← hbr]
          simp only [List.mem_append, List.mem_singleton] at hf ⊢
          rcases hf with hf | rfl
          · exact Or.inl (hall f' hf hfw)
          · exact Or.inr rfl

theorem k0_star {g : G} {cfg : Cfg} (hk : cfg.k = 0) {b0 : Option Nat} {st : DS}
    (h : Star g cfg (init g cfg b0) st) : K0Inv b0 st := by
  generalize hi : init g cfg b0 = s0 at h
  induction h with
  | refl => subst hi; left; exact ⟨rfl, rfl, rfl⟩
  | tail _ hs ih => exact k0_step hk ih hs

/-- **`pathCount = 0`**: the results are exactly the minimum-weight qualifying
paths (when one weighs at most the seed bound). -/
theorem dfs_k0 (g : G) (cfg : Cfg) (hk : cfg.k = 0) (b0 : Option Nat) :
    ∃ F, Star g cfg (init g cfg b0) F ∧ step g cfg F = none ∧
      (F.results = [] → ∀ es W C, Ext g cfg (succsRev g cfg.src) 0 0 0 [cfg.src] es W C → ¬ BLe W b0) ∧
      (F.results ≠ [] → ∃ m, (∀ f ∈ F.results, Good g cfg f ∧ f.w = m) ∧
        (∀ es W C, Ext g cfg (succsRev g cfg.src) 0 0 0 [cfg.src] es W C →
          m ≤ W ∧ (W = m → (⟨es, W, C⟩ : FP) ∈ F.results))) := by
  obtain ⟨F, h1, h2, h3⟩ := dfs_complete g cfg (step_bound_mono g cfg (by omega)) b0
  refine ⟨F, h1, h2, ?_, ?_⟩
  · intro hr es W C hext hW
    rcases k0_star hk h1 with ⟨_, hl, hb⟩ | ⟨m, hne, _⟩
    · have := h3 es W C hext (by rw [hb]; exact hW); rw [hl] at this; cases this
    · exact hne hr
  · intro hne
    rcases k0_star hk h1 with ⟨hr, _⟩ | ⟨m, _, hb, hres, hlog, hall⟩
    · exact absurd hr hne
    · refine ⟨m, fun f hf => ⟨dfs_sound g cfg b0 F h1 f hf, (hres f hf).2⟩, ?_⟩
      intro es W C hext
      by_cases hle : W ≤ m
      · have hin := h3 es W C hext (by rw [hb]; simp [BLe, hle])
        have := hlog _ hin
        exact ⟨this, fun hw => hall _ hin hw⟩
      · exact ⟨by omega, fun hw => by omega⟩

end AlgoUdf.Dfs
