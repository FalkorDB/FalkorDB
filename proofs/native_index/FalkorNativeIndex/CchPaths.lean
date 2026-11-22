import FalkorNativeIndex.CchMetric
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# Up-down paths: the shortest path is the cheapest pair of ascending arc paths (`cch/query.rs`, module doc)

Rank space. `Ctx` bundles what Phases 1 and 2 establish about the chordal rows `up`, the original edge weights `wt`
and the customized arc weights `A` (`customize_correct`, `row_iff`, `up_closure`); `N` bounds every rank.
-/
namespace NI

structure Ctx (up : Nat → List Nat) (wt A : Nat → Nat → Wt) (N : Nat) : Prop where
  gt : ∀ r w, w ∈ up r → r < w
  lt : ∀ r w, w ∈ up r → w < N
  closure : ∀ x r w, r ∈ up x → w ∈ up x → r < w → w ∈ up r
  isMin : ∀ u v, ArcP up u v → IsMin wt (min u v) u v (A u v)
  edgeArc : ∀ u v c, wt u v = some c → ArcP up u v
  sorted : ∀ r, (up r).Pairwise (· < ·)

/-- An ascending path `a = a₀ < a₁ < ... < a_k = z` over the upward rows, `k` arcs, each with a finite weight
    `wf a_i a_{i+1}` (`A` for the forward search, `A` flipped for the backward one), total `e`. -/
inductive UpL (up : Nat → List Nat) (wf : Nat → Nat → Wt) : Nat → Nat → Nat → Nat → Prop where
  | nil {a : Nat} : UpL up wf 0 a a 0
  | cons {a m z k w e : Nat} : m ∈ up a → wf a m = some w → UpL up wf k m z e → UpL up wf (k + 1) a z (w + e)

def UpPath (up : Nat → List Nat) (wf : Nat → Nat → Wt) (a z e : Nat) : Prop := ∃ k, UpL up wf k a z e

theorem UpL.append {up wf} {k1 a m e1} (h : UpL up wf k1 a m e1) :
    ∀ {k2 z e2}, UpL up wf k2 m z e2 → UpL up wf (k1 + k2) a z (e1 + e2) := by
  induction h with
  | nil => intro k2 z e2 h2; simpa using h2
  | @cons a x m' k w e hm hw _ ih =>
    intro k2 z e2 h2
    have := UpL.cons hm hw (ih h2)
    have e1 : k + 1 + k2 = k + k2 + 1 := by omega
    rw [e1, show w + e + e2 = w + (e + e2) by omega]
    exact this

theorem UpPath.append {up wf a m z e1 e2} (h1 : UpPath up wf a m e1) (h2 : UpPath up wf m z e2) :
    UpPath up wf a z (e1 + e2) := by
  obtain ⟨k1, h1⟩ := h1; obtain ⟨k2, h2⟩ := h2; exact ⟨_, h1.append h2⟩

theorem UpPath.refl (up wf a) : UpPath up wf a a 0 := ⟨0, .nil⟩

theorem UpPath.step {up wf a m w} (hm : m ∈ up a) (hw : wf a m = some w) : UpPath up wf a m w :=
  ⟨1, by simpa using UpL.cons hm hw UpL.nil⟩

/-- The last step of an ascending path. -/
theorem UpPath.snoc {up wf s r e} (h : UpPath up wf s r e) :
    (r = s ∧ e = 0) ∨ ∃ a e0 w, UpPath up wf s a e0 ∧ r ∈ up a ∧ wf a r = some w ∧ e = e0 + w := by
  obtain ⟨k, h⟩ := h
  induction h with
  | nil => exact Or.inl ⟨rfl, rfl⟩
  | @cons a m z k w e hm hw hrest ih =>
    rcases ih with ⟨rfl, rfl⟩ | ⟨b, e0, w2, hp, hz, hw2, he⟩
    · exact Or.inr ⟨a, 0, w, UpPath.refl _ _ _, hm, hw, by omega⟩
    · refine Or.inr ⟨b, w + e0, w2, ?_, hz, hw2, by omega⟩
      exact (UpPath.step hm hw).append hp

/-- Real walks (possibly empty). -/
def WR (wt : Nat → Nat → Wt) (N s t c : Nat) : Prop := (s = t ∧ c = 0) ∨ WP wt N s t c

theorem WR.trans {wt N s m t c1 c2} (h1 : WR wt N s m c1) (hm : m < N) (h2 : WR wt N m t c2) :
    WR wt N s t (c1 + c2) := by
  rcases h1 with ⟨rfl, rfl⟩ | h1
  · simpa using h2
  · rcases h2 with ⟨rfl, rfl⟩ | h2
    · exact Or.inr (by simpa using h1)
    · exact Or.inr (h1.trans hm h2)

theorem arc_wr {up wt A N} (h : Ctx up wt A N) (a m w : Nat) (harc : ArcP up a m) (hw : A a m = some w) :
    WP wt N a m w := by
  have := (h.isMin a m harc).2 w hw
  exact this.mono (by
    have : min a m ≤ N := by
      rcases harc.2 with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · have := h.lt _ _ h2; omega
      · have := h.lt _ _ h2; omega
    exact this)

/-- A forward path is a real walk. -/
theorem upPath_fwd {up wt A N} (h : Ctx up wt A N) {a z e} (p : UpPath up (fun x y => A x y) a z e) :
    WR wt N a z e := by
  obtain ⟨k, p⟩ := p
  induction p with
  | nil => exact Or.inl ⟨rfl, rfl⟩
  | @cons a m z k w e hm hw _ ih =>
    have harc : ArcP up a m := ⟨Nat.ne_of_lt (h.gt _ _ hm), Or.inl ⟨h.gt _ _ hm, hm⟩⟩
    exact WR.trans (Or.inr (arc_wr h a m w harc hw)) (h.lt _ _ hm) ih

/-- A backward path is a real walk, read from its top to its start. -/
theorem upPath_bwd {up wt A N} (h : Ctx up wt A N) {a z e} (p : UpPath up (fun x y => A y x) a z e) :
    WR wt N z a e := by
  obtain ⟨k, p⟩ := p
  induction p with
  | nil => exact Or.inl ⟨rfl, rfl⟩
  | @cons a m z k w e hm hw _ ih =>
    have harc : ArcP up m a := ⟨(Nat.ne_of_lt (h.gt _ _ hm)).symm, Or.inr ⟨h.gt _ _ hm, hm⟩⟩
    have hmN := h.lt _ _ hm
    have := WR.trans ih hmN (Or.inr (arc_wr h m a w harc hw))
    rwa [Nat.add_comm] at this

/-- The cheapest detour through a lower rank: with `b` below both `x` and `y`, the arc `x -> y` is no dearer than
    `x -> b -> y` (customization's `IsMin`). -/
theorem arc_detour {up wt A N} (h : Ctx up wt A N) (b x y w1 w2 : Nat) (hbx : b < x) (hby : b < y) (hxy : x ≠ y)
    (harc : ArcP up x y) (h1 : A x b = some w1) (hb1 : ArcP up x b) (h2 : A b y = some w2) (hb2 : ArcP up b y) :
    ∃ e, A x y = some e ∧ e ≤ w1 + w2 := by
  have p1 := (h.isMin x b hb1).2 w1 h1
  have p2 := (h.isMin b y hb2).2 w2 h2
  have e1 : min x b = b := by omega
  have e2 : min b y = b := by omega
  rw [e1] at p1; rw [e2] at p2
  have hm : b < min x y := by omega
  have := (p1.mono (by omega : b ≤ min x y)).trans hm (p2.mono (by omega : b ≤ min x y))
  obtain ⟨e, he, hle⟩ := (h.isMin x y harc).1 _ this
  exact ⟨e, he, hle⟩

/-- **The valley lemma**: an ascending path down to `b` (backward weights) and one up from `b` (forward
    weights) can be replaced by an up-down pair no dearer, with its peak above `b`. -/
theorem valley {up wt A N} (h : Ctx up wt A N) :
    ∀ (n : Nat) (k1 k2 b m1 m2 e1 e2 : Nat), k1 + k2 ≤ n →
      UpL up (fun x y => A y x) k1 b m1 e1 → UpL up (fun x y => A x y) k2 b m2 e2 →
      ∃ m e3 e4, UpPath up (fun x y => A x y) m1 m e3 ∧ UpPath up (fun x y => A y x) m2 m e4 ∧ e3 + e4 ≤ e1 + e2 := by
  intro n
  induction n with
  | zero =>
    intro k1 k2 b m1 m2 e1 e2 hk p1 p2
    cases p1 with
    | nil => exact ⟨m2, e2, 0, ⟨k2, p2⟩, UpPath.refl _ _ _, by omega⟩
    | cons _ _ _ => omega
  | succ n ih =>
    intro k1 k2 b m1 m2 e1 e2 hk p1 p2
    cases p1 with
    | nil => exact ⟨m2, e2, 0, ⟨k2, p2⟩, UpPath.refl _ _ _, by omega⟩
    | @cons _ x1 _ k1' w1 e1' hx hw1 r1 =>
      cases p2 with
      | nil => exact ⟨m1, 0, w1 + e1', UpPath.refl _ _ _, ⟨_, UpL.cons hx hw1 r1⟩, by omega⟩
      | @cons _ y1 _ k2' w2 e2' hy hw2 r2 =>
        have hbx := h.gt _ _ hx
        have hby := h.gt _ _ hy
        rcases Nat.lt_trichotomy x1 y1 with hlt | heq | hgt
        · have hyx : y1 ∈ up x1 := h.closure b x1 y1 hx hy hlt
          obtain ⟨e', he', hle⟩ := arc_detour h b x1 y1 w1 w2 hbx hby (by omega)
            ⟨by omega, Or.inl ⟨hlt, hyx⟩⟩ hw1 ⟨by omega, Or.inr ⟨hbx, hx⟩⟩ hw2 ⟨by omega, Or.inl ⟨hby, hy⟩⟩
          obtain ⟨m, e3, e4, a1, a2, a3⟩ := ih k1' (k2' + 1) x1 m1 m2 e1' (e' + e2') (by omega) r1
            (UpL.cons hyx he' r2)
          exact ⟨m, e3, e4, a1, a2, by omega⟩
        · subst heq
          obtain ⟨m, e3, e4, a1, a2, a3⟩ := ih k1' k2' x1 m1 m2 e1' e2' (by omega) r1 r2
          exact ⟨m, e3, e4, a1, a2, by omega⟩
        · have hxy : x1 ∈ up y1 := h.closure b y1 x1 hy hx hgt
          obtain ⟨e', he', hle⟩ := arc_detour h b x1 y1 w1 w2 hbx hby (by omega)
            ⟨by omega, Or.inr ⟨hgt, hxy⟩⟩ hw1 ⟨by omega, Or.inr ⟨hbx, hx⟩⟩ hw2 ⟨by omega, Or.inl ⟨hby, hy⟩⟩
          obtain ⟨m, e3, e4, a1, a2, a3⟩ := ih (k1' + 1) k2' y1 m1 m2 (e' + e1') e2' (by omega)
            (UpL.cons hxy he' r1) r2
          exact ⟨m, e3, e4, a1, a2, by omega⟩

/-- **Every walk is dominated by an up-down pair**: for a walk `s -> t` of weight `c`, there is a rank `m` with an
    ascending forward path `s ~> m` and an ascending backward path `t ~> m` of total weight at most `c`. -/
theorem updown_complete {up wt A N} (h : Ctx up wt A N) :
    ∀ (b s t c : Nat), WP wt b s t c →
      ∃ m e1 e2, UpPath up (fun x y => A x y) s m e1 ∧ UpPath up (fun x y => A y x) t m e2 ∧ e1 + e2 ≤ c := by
  intro b
  induction b with
  | zero =>
    intro s t c p
    cases p with
    | step _ hb _ => omega
    | edge he =>
      have harc := h.edgeArc s t c he
      have hmin : min s t ≥ 0 := Nat.zero_le _
      obtain ⟨e, hAe, hle⟩ := (h.isMin s t harc).1 c (by simpa using (WP.edge he : WP wt 0 s t c).mono (Nat.zero_le (min s t)))
      rcases harc.2 with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact ⟨t, e, 0, UpPath.step h2 hAe, UpPath.refl _ _ _, by omega⟩
      · exact ⟨s, 0, e, UpPath.refl _ _ _, UpPath.step h2 hAe, by omega⟩
  | succ b ih =>
    intro s t c p
    rcases p.split with p0 | ⟨c1, c2, p1, p2, hle⟩
    · exact ih s t c p0
    · obtain ⟨m1, e1, e1', f1, g1, l1⟩ := ih s b c1 p1
      obtain ⟨m2, e2, e2', f2, g2, l2⟩ := ih b t c2 p2
      obtain ⟨k1, g1'⟩ := g1
      obtain ⟨k2, f2'⟩ := f2
      obtain ⟨m, e3, e4, a1, a2, a3⟩ := valley h (k1 + k2) k1 k2 b m1 m2 e1' e2 (Nat.le_refl _) g1' f2'
      exact ⟨m, e1 + e3, e2' + e4, f1.append a1, g2.append a2, by omega⟩

/-- **Up-down paths are real walks**: ascending paths from `s` and `t` to a common `m` give a walk `s -> t` of
    their total weight. -/
theorem updown_sound {up wt A N} (h : Ctx up wt A N) {s t m e1 e2 : Nat}
    (hm : m < N) (p1 : UpPath up (fun x y => A x y) s m e1) (p2 : UpPath up (fun x y => A y x) t m e2) :
    WR wt N s t (e1 + e2) :=
  WR.trans (upPath_fwd h p1) hm (upPath_bwd h p2)

end NI
