import AlgoUdf.DfsSound
/-! # `enumerate_paths` completeness and termination

`Ext S pw pc d vis es W C` is a *qualifying extension*: a simple continuation
`es` of the live path (current prefix weight `pw`, cost `pc`, depth `d`, nodes
`vis`) whose first step is drawn from `S` and every later step from
`node_successors`, with finite weights, every prefix cost within `maxCost`, at
most `maxLen` hops in total, ending at the target (any node for SSpaths) and —
for SPpaths — not passing through the target before its end (:2547-2558).

`explore` (the subtree lemma): from any reachable frame whose pending list is
`L`, the loop comes back to the same frame with `L` exhausted, having handed
every qualifying extension through `L` of weight ≤ the final bound to
`record_found_path`. `dfs_complete` instantiates it at the root: the loop
terminates and every qualifying path of weight ≤ the final bound is recorded.
Bound monotonicity is proven for `pathCount ∈ {0, 1}` (`step_bound_mono`); the
k ≥ 2 branch is covered by the hypothesis `BoundMono`.
-/
namespace AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

inductive Ext (g : G) (cfg : Cfg) :
    List (Rel × Nat) → Nat → Nat → Nat → List Nat → List Rel → Nat → Nat → Prop
  | last {S : List (Rel × Nat)} {pw pc d : Nat} {vis : List Nat} {r : Rel} {x c : Nat} :
      (r, x) ∈ S → x ∉ vis → g.wt r.2.2 = some c →
      cfg.maxCost.all (pc + g.cost r.2.2 ≤ ·) = true → d + 1 ≤ cfg.maxLen →
      cfg.tgt.all (· == x) = true →
      Ext g cfg S pw pc d vis [r] (pw + c) (pc + g.cost r.2.2)
  | more {S : List (Rel × Nat)} {pw pc d : Nat} {vis : List Nat} {r : Rel} {x c : Nat}
      {rest : List Rel} {W C : Nat} :
      (r, x) ∈ S → x ∉ vis → g.wt r.2.2 = some c →
      cfg.maxCost.all (pc + g.cost r.2.2 ≤ ·) = true →
      (cfg.tgt.all (· == x) && cfg.tgt.isSome) = false → d + 1 < cfg.maxLen →
      Ext g cfg (succsRev g x) (pw + c) (pc + g.cost r.2.2) (d + 1) (x :: vis) rest W C →
      Ext g cfg S pw pc d vis (r :: rest) W C

/-- `W ≤ bound`, `none` being `+∞`. -/
def BLe (W : Nat) (b : Option Nat) : Prop := b.all (W ≤ ·) = true

/-- `b' ≤ b` on `Option Nat` with `none = +∞`. -/
def BoundLe (b' b : Option Nat) : Prop := ∀ y, b = some y → ∃ y', b' = some y' ∧ y' ≤ y

theorem BoundLe.refl (b : Option Nat) : BoundLe b b := fun y h => ⟨y, h, Nat.le_refl _⟩
theorem BoundLe.trans {a b c : Option Nat} (h1 : BoundLe a b) (h2 : BoundLe b c) : BoundLe a c := by
  intro y hy
  obtain ⟨y', h1', h2'⟩ := h2 y hy
  obtain ⟨y'', h1'', h2''⟩ := h1 y' h1'
  exact ⟨y'', h1'', Nat.le_trans h2'' h2'⟩
theorem BLe.mono {W : Nat} {b b' : Option Nat} (h : BLe W b') (hb : BoundLe b' b) : BLe W b := by
  unfold BLe at *
  cases hb' : b with
  | none => rfl
  | some y =>
    obtain ⟨y', h1, h2⟩ := hb y hb'
    rw [h1] at h; simp at h ⊢; omega

/-- The bound never increases along a step. -/
def BoundMono (g : G) (cfg : Cfg) : Prop :=
  ∀ st st', Inv g cfg st → step g cfg st = some st' → BoundLe st'.bound st.bound

theorem Ext.first_le {g : G} {cfg : Cfg} {S pw pc d vis es W C} (h : Ext g cfg S pw pc d vis es W C) :
    ∃ r x c rest, es = r :: rest ∧ (r, x) ∈ S ∧ x ∉ vis ∧ g.wt r.2.2 = some c ∧
      cfg.maxCost.all (pc + g.cost r.2.2 ≤ ·) = true ∧ pw + c ≤ W := by
  induction h with
  | @last _ pw _ _ _ r x c h1 h2 h3 h4 _ _ => exact ⟨r, x, c, [], rfl, h1, h2, h3, h4, Nat.le_refl _⟩
  | @more _ pw _ _ _ r x c rest _ _ h1 h2 h3 h4 _ _ _ ih =>
    obtain ⟨_, _, c', _, _, _, _, _, _, hle⟩ := ih
    exact ⟨r, x, c, rest, rfl, h1, h2, h3, h4, by omega⟩

/-! ## Step equations -/

theorem pathOf_lv (fr : Frame) (lv : Option (List (Rel × Nat))) (below : List Frame) :
    pathOf ({ fr with lv := lv } :: below) = pathOf (fr :: below) := by
  simp [pathOf, List.filterMap_cons]

theorem step_back {g : G} {cfg : Cfg} {st : DS} {fr : Frame} {below : List Frame}
    (hfs : st.frames = fr :: below) (hlv : fr.lv = none ∨ fr.lv = some []) (hb : below ≠ []) :
    step g cfg st = some { st with frames := below, onPath := upd st.onPath fr.node false } := by
  unfold step; rw [hfs]
  rcases hlv with h | h <;> simp [h, hb]

theorem step_root_done {g : G} {cfg : Cfg} {st : DS} {fr : Frame}
    (hfs : st.frames = [fr]) (hlv : fr.lv = none ∨ fr.lv = some []) : step g cfg st = none := by
  unfold step; rw [hfs]
  rcases hlv with h | h <;> simp [h]

theorem Star.head {g : G} {cfg : Cfg} {a b c : DS} (h : step g cfg a = some b) (h' : Star g cfg b c) :
    Star g cfg a c := by
  induction h' with
  | refl => exact .tail (.refl _) h
  | tail _ hs ih => exact .tail ih hs

theorem Star.trans {g : G} {cfg : Cfg} {a b c : DS} (h : Star g cfg a b) (h' : Star g cfg b c) :
    Star g cfg a c := by
  induction h' with
  | refl => exact h
  | tail _ hs ih => exact .tail ih hs

theorem Star.inv {g : G} {cfg : Cfg} {a b : DS} (I : Inv g cfg a) (h : Star g cfg a b) : Inv g cfg b := by
  induction h with
  | refl => exact I
  | tail _ hs ih => exact inv_step ih hs

theorem Star.bound {g : G} {cfg : Cfg} (hm : BoundMono g cfg) {a b : DS} (I : Inv g cfg a)
    (h : Star g cfg a b) : BoundLe b.bound a.bound := by
  induction h with
  | refl => exact .refl _
  | tail h1 hs ih => exact (hm _ _ (Star.inv I h1) hs).trans ih

theorem Star.log {g : G} {cfg : Cfg} {a b : DS} (h : Star g cfg a b) : ∃ ext, b.log = a.log ++ ext := by
  induction h with
  | refl => exact ⟨[], by simp⟩
  | @tail b c _ hs ih =>
    obtain ⟨e, he⟩ := ih
    unfold step at hs
    try dsimp only at hs
    split at hs
    · cases hs
    · split at hs
      · split at hs
        · cases hs; exact ⟨e, he⟩
        · split at hs
          · cases hs; exact ⟨e, he⟩
          · split at hs
            · cases hs; exact ⟨e, he⟩
            · split at hs
              · cases hs; exact ⟨e, he⟩
              · cases hs
                dsimp only
                split
                · exact ⟨e ++ [_], by rw [he, List.append_assoc]⟩
                · exact ⟨e, he⟩
      · split at hs
        · cases hs
        · cases hs; exact ⟨e, he⟩


/-- A successor is skipped (on path, non-finite weight, over the bound, over maxCost). -/
theorem step_skip {g : G} {cfg : Cfg} {st : DS} {fr : Frame} {below : List Frame} {r : Rel} {x : Nat}
    {rest : List (Rel × Nat)} (hfs : st.frames = fr :: below) (hlv : fr.lv = some ((r, x) :: rest))
    (hskip : st.onPath x = true ∨ g.wt r.2.2 = none ∨
      (∃ c, g.wt r.2.2 = some c ∧ (st.bound.any (· < fr.pw + c) = true ∨
        cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = true))) :
    step g cfg st = some { st with frames := { fr with lv := some rest } :: below } := by
  unfold step; rw [hfs]; dsimp only; rw [hlv]; dsimp only
  by_cases hon : st.onPath x = true
  · rw [if_pos hon]
  · rw [if_neg hon]
    rcases hskip with h | h | ⟨c, hc, h⟩
    · exact absurd h hon
    · rw [h]
    · rw [hc]; dsimp only
      rcases h with h | h
      · rw [if_pos h]
      · by_cases hb : st.bound.any (· < fr.pw + c) = true
        · rw [if_pos hb]
        · rw [if_neg hb, if_pos h]

theorem step_push {g : G} {cfg : Cfg} {st : DS} {fr : Frame} {below : List Frame} {r : Rel} {x c : Nat}
    {rest : List (Rel × Nat)} (hfs : st.frames = fr :: below) (hlv : fr.lv = some ((r, x) :: rest))
    (hon : st.onPath x = false) (hw : g.wt r.2.2 = some c)
    (hb : st.bound.any (· < fr.pw + c) = false)
    (hmc : cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = false) :
    step g cfg st = some
      ⟨⟨if (!(cfg.tgt.all (· == x) && cfg.tgt.isSome) && decide (below.length + 1 < cfg.maxLen))
          then some (succsRev g x) else none, x, fr.pw + c, fr.pc + g.cost r.2.2, some r⟩ ::
          { fr with lv := some rest } :: below,
        upd st.onPath x true,
        (if cfg.tgt.all (· == x) then recordB st.results
            ⟨pathOf ({ fr with lv := some rest } :: below) ++ [r], fr.pw + c, fr.pc + g.cost r.2.2⟩ cfg.k st.bound
          else (st.results, st.bound)).1,
        (if cfg.tgt.all (· == x) then recordB st.results
            ⟨pathOf ({ fr with lv := some rest } :: below) ++ [r], fr.pw + c, fr.pc + g.cost r.2.2⟩ cfg.k st.bound
          else (st.results, st.bound)).2,
        if cfg.tgt.all (· == x) then
          st.log ++ [⟨pathOf ({ fr with lv := some rest } :: below) ++ [r], fr.pw + c, fr.pc + g.cost r.2.2⟩]
        else st.log⟩ := by
  unfold step; rw [hfs]; dsimp only; rw [hlv]; dsimp only
  rw [if_neg (by rw [hon]; simp), hw]; dsimp only
  rw [if_neg (by rw [hb]; simp), if_neg (by rw [hmc]; simp)]

/-- Split a qualifying extension on its first pending successor. -/
theorem Ext.cons_split {g : G} {cfg : Cfg} {p : Rel × Nat} {rest : List (Rel × Nat)} {pw pc d vis es W C}
    (h : Ext g cfg (p :: rest) pw pc d vis es W C) :
    Ext g cfg rest pw pc d vis es W C ∨ Ext g cfg [p] pw pc d vis es W C := by
  cases h with
  | last h1 h2 h3 h4 h5 h6 =>
    simp only [List.mem_cons] at h1
    rcases h1 with rfl | h1
    · exact Or.inr (.last (by simp) h2 h3 h4 h5 h6)
    · exact Or.inl (.last h1 h2 h3 h4 h5 h6)
  | more h1 h2 h3 h4 h5 h6 h7 =>
    simp only [List.mem_cons] at h1
    rcases h1 with rfl | h1
    · exact Or.inr (.more (by simp) h2 h3 h4 h5 h6 h7)
    · exact Or.inl (.more h1 h2 h3 h4 h5 h6 h7)

/-- Conclusion of exploring a frame whose pending list is `L`. -/
def Explored (g : G) (cfg : Cfg) (st : DS) (fr : Frame) (below : List Frame)
    (L : List (Rel × Nat)) : Prop :=
  ∃ st', Star g cfg st st' ∧ st'.frames = { fr with lv := some [] } :: below ∧
    (∀ y, st'.onPath y = st.onPath y) ∧
    ∀ es W C, Ext g cfg L fr.pw fr.pc below.length ((fr :: below).map (·.node)) es W C →
      BLe W st'.bound → (⟨pathOf (fr :: below) ++ es, W, C⟩ : FP) ∈ st'.log

theorem log_sub {g : G} {cfg : Cfg} {a b : DS} (h : Star g cfg a b) {f : FP} (hf : f ∈ a.log) :
    f ∈ b.log := by
  obtain ⟨e, he⟩ := Star.log h; rw [he]; exact List.mem_append_left _ hf

end AlgoUdf.Dfs
