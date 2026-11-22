import FalkorNativeIndex.CchTopo
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# CCH customization (`cch/customize.rs`): every arc holds the cheapest path through lower ranks

Weights are `Nat` with `none` = `+inf` (a closed or absent edge). The directed arc weight `A u v` is indexed by the
rank pair (an arc slot is `up_range(lower).start + position of higher in lower's row`, `arc_slot`); `A u v` with
`u < v` is `upward_weight[slot]`, with `u > v` it is `downward_weight[slot]` of the arc `{v, u}`.
-/
namespace NI

abbrev Wt := Option Nat

def wadd : Wt → Wt → Wt
  | some a, some b => some (a + b)
  | _, _ => none

/-- `a < b` on `[0, +inf]`. -/
def wlt : Wt → Wt → Bool
  | some a, some b => decide (a < b)
  | some _, none => true
  | none, _ => false

/-- The strict-`<` improvement: `if offer < current { offer } else { current }`. -/
def wmin (cur offer : Wt) : Wt := if wlt offer cur then offer else cur

/-- A walk `u → ... → v` in the original directed graph (edge weight `wt`, `none` = no edge or closed) whose
    intermediate vertices are all below `b`, of total weight `c`. -/
inductive WP (wt : Nat → Nat → Wt) (b : Nat) : Nat → Nat → Nat → Prop where
  | edge {u v c : Nat} : wt u v = some c → WP wt b u v c
  | step {u v w c1 c2 : Nat} : wt u v = some c1 → v < b → WP wt b v w c2 → WP wt b u w (c1 + c2)

theorem WP.mono {wt : Nat → Nat → Wt} {b b' : Nat} (h : b ≤ b') {u w c : Nat} (p : WP wt b u w c) : WP wt b' u w c := by
  induction p with
  | edge he => exact .edge he
  | step he hb _ ih => exact .step he (by omega) ih

theorem WP.trans {wt : Nat → Nat → Wt} {b : Nat} {u m w c1 c2 : Nat} (p : WP wt b u m c1) (hm : m < b)
    (q : WP wt b m w c2) : WP wt b u w (c1 + c2) := by
  induction p with
  | edge he => exact .step he hm q
  | @step u v w' a1 a2 he hb _ ih =>
    have := WP.step he hb (ih hm q)
    rwa [← Nat.add_assoc] at this

/-- The weighted split: through `k` or not. -/
theorem WP.split {wt : Nat → Nat → Wt} {k u w c : Nat} (p : WP wt (k + 1) u w c) :
    WP wt k u w c ∨ ∃ c1 c2, WP wt k u k c1 ∧ WP wt k k w c2 ∧ c1 + c2 ≤ c := by
  induction p with
  | edge he => exact Or.inl (.edge he)
  | @step u v w c1 c' he hb _ ih =>
    by_cases hvk : v < k
    · rcases ih with h | ⟨d1, d2, h1, h2, hle⟩
      · exact Or.inl (.step he hvk h)
      · exact Or.inr ⟨c1 + d1, d2, .step he hvk h1, h2, by omega⟩
    · have hv' : v = k := by omega
      subst hv'
      refine Or.inr ⟨c1, ?_⟩
      rcases ih with h | ⟨d1, d2, h1, h2, hle⟩
      · exact ⟨c', .edge he, h, by omega⟩
      · exact ⟨d2, .edge he, h2, by omega⟩

/-- `d` is the cheapest weight of a walk `u → v` through vertices below `b` (`none`: there is none). -/
def IsMin (wt : Nat → Nat → Wt) (b u v : Nat) (d : Wt) : Prop :=
  (∀ c, WP wt b u v c → ∃ e, d = some e ∧ e ≤ c) ∧ (∀ e, d = some e → WP wt b u v e)

/-- **The sweep's one step is Floyd-Warshall**: given the cheapest walks through `< k`, those through `< k + 1`
    are the better of the old one and the detour through `k`. -/
theorem isMin_fw (wt : Nat → Nat → Wt) (k u v : Nat) (d0 d1 d2 : Wt)
    (h0 : IsMin wt k u v d0) (h1 : IsMin wt k u k d1) (h2 : IsMin wt k k v d2) :
    IsMin wt (k + 1) u v (wmin d0 (wadd d1 d2)) := by
  constructor
  · intro c hc
    rcases hc.split with h | ⟨c1, c2, p1, p2, hle⟩
    · obtain ⟨e, he, hec⟩ := h0.1 c h
      subst he
      unfold wmin
      cases hoff : wadd d1 d2 with
      | none => simp [wlt]; exact hec
      | some x =>
        simp only [wlt]
        by_cases hx : x < e
        · simp [hx]; omega
        · simp [hx]; exact hec
    · obtain ⟨e1, he1, hc1⟩ := h1.1 c1 p1
      obtain ⟨e2, he2, hc2⟩ := h2.1 c2 p2
      subst he1 he2
      unfold wmin
      cases hd : d0 with
      | none => simp [wadd, wlt]; omega
      | some x =>
        simp only [wadd, wlt]
        by_cases hx : e1 + e2 < x
        · simp [hx]; omega
        · simp [hx]; omega
  · intro e he
    unfold wmin at he
    by_cases hl : wlt (wadd d1 d2) d0 = true
    · rw [ifP hl] at he
      cases hd1 : d1 with
      | none => simp [hd1, wadd] at he
      | some a =>
        cases hd2 : d2 with
        | none => simp [hd1, hd2, wadd] at he
        | some b =>
          simp only [hd1, hd2, wadd, Option.some.injEq] at he
          subst he
          exact (h1.2 a hd1).mono (by omega) |>.trans (by omega) (h2.2 b hd2 |>.mono (by omega))
    · rw [ifN hl] at he
      exact (h0.2 e he).mono (by omega)

/-! ## The sweep (`relax`, `customize.rs`)

For each rank `x` in increasing order, every ordered pair `y ≠ z` of its upper neighbours is offered the detour
`y → x → z`, taking it on a strict improvement and recording `x` as the middle. (The Rust visits each unordered
pair once and improves its upward and downward arc; those are the two ordered pairs. The offered weights read
arcs that have `x` as an endpoint, which no write of the same `x` touches — a write needs both ends in `up(x)` —
so the sequential loop and the simultaneous update below agree.) `up` is a rank's upward row. -/

/-- An arc of the chordal supergraph, between two distinct ranks, in either direction. -/
def ArcP (up : Nat → List Nat) (u v : Nat) : Prop :=
  u ≠ v ∧ ((u < v ∧ v ∈ up u) ∨ (v < u ∧ u ∈ up v))

/-- The weights after the detours through `x`. -/
def relaxA (up : Nat → List Nat) (x : Nat) (A : Nat → Nat → Wt) : Nat → Nat → Wt := fun y z =>
  if y ∈ up x ∧ z ∈ up x ∧ y ≠ z then wmin (A y z) (wadd (A y x) (A x z)) else A y z

/-- The middles after the detours through `x`. -/
def relaxM (up : Nat → List Nat) (x : Nat) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat) :
    Nat → Nat → Option Nat := fun y z =>
  if y ∈ up x ∧ z ∈ up x ∧ y ≠ z ∧ wlt (wadd (A y x) (A x z)) (A y z) = true then some x else M y z

def relaxStep (up : Nat → List Nat) (x : Nat) (st : (Nat → Nat → Wt) × (Nat → Nat → Option Nat)) :
    (Nat → Nat → Wt) × (Nat → Nat → Option Nat) := (relaxA up x st.1, relaxM up x st.1 st.2)

/-- The seed (`seed`, `customize.rs`): every original edge's weight, the diagonal dropped. -/
def seedA (w : Nat → Nat → Wt) : Nat → Nat → Wt := fun u v => if u = v then none else w u v

/-- `customize` on ranks `0 .. n-1`: seed, then the sweep. -/
def customizeRun (up : Nat → List Nat) (w : Nat → Nat → Wt) (n : Nat) :
    (Nat → Nat → Wt) × (Nat → Nat → Option Nat) :=
  (List.range n).foldl (fun st x => relaxStep up x st) (seedA w, fun _ _ => none)

/-- The invariant after the ranks `< k` are processed. -/
def MetricInv (up : Nat → List Nat) (wt : Nat → Nat → Wt) (k : Nat) (A : Nat → Nat → Wt)
    (M : Nat → Nat → Option Nat) : Prop :=
  (∀ u v, ArcP up u v → IsMin wt (min k (min u v)) u v (A u v)) ∧
  (∀ u v, ArcP up u v → (M u v = none → A u v = wt u v) ∧
    ∀ m, M u v = some m → m < min u v ∧ m < k ∧ ArcP up u m ∧ ArcP up m v ∧
      ∃ e1 e2, A u m = some e1 ∧ A m v = some e2 ∧ A u v = some (e1 + e2))

theorem wp_pth (adj : Nat → List Nat) (wt : Nat → Nat → Wt) (hcov : ∀ u v c, wt u v = some c → v ∈ adj u)
    {b u v c : Nat} (p : WP wt b u v c) : Pth adj b u v := by
  induction p with
  | edge he => exact .edge (hcov _ _ _ he)
  | step he hb _ ih => exact .step (hcov _ _ _ he) hb ih

theorem isMin_none_of_not_up (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (wt : Nat → Nat → Wt)
    (hcov : ∀ u v c, wt u v = some c → v ∈ adj u) (k u : Nat) (hku : k < u) (hu : u ∉ rowOf adj k) :
    IsMin wt k u k none := by
  refine ⟨fun c hc => ?_, fun e he => by cases he⟩
  exact absurd (row_complete adj hs k u hku (Pth.symm hs (wp_pth adj wt hcov hc))) hu

theorem isMin_none_of_not_up' (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (wt : Nat → Nat → Wt)
    (hcov : ∀ u v c, wt u v = some c → v ∈ adj u) (k v : Nat) (hkv : k < v) (hv : v ∉ rowOf adj k) :
    IsMin wt k k v none := by
  refine ⟨fun c hc => ?_, fun e he => by cases he⟩
  exact absurd (row_complete adj hs k v hkv (wp_pth adj wt hcov hc)) hv

theorem row_gt (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (r w : Nat) (h : w ∈ rowOf adj r) :
    r < w := (row_sound adj hs r w h).1

theorem relaxA_unchanged (up : Nat → List Nat) (hgt : ∀ r w, w ∈ up r → r < w) (k : Nat) (A : Nat → Nat → Wt)
    (y z : Nat) (h : ¬ (k < y ∧ k < z)) : relaxA up k A y z = A y z := by
  unfold relaxA
  rw [ifN]
  rintro ⟨h1, h2, _⟩
  exact h ⟨hgt _ _ h1, hgt _ _ h2⟩

theorem relaxM_unchanged (up : Nat → List Nat) (hgt : ∀ r w, w ∈ up r → r < w) (k : Nat) (A : Nat → Nat → Wt)
    (M : Nat → Nat → Option Nat) (y z : Nat) (h : ¬ (k < y ∧ k < z)) : relaxM up k A M y z = M y z := by
  unfold relaxM
  rw [ifN]
  rintro ⟨h1, h2, _⟩
  exact h ⟨hgt _ _ h1, hgt _ _ h2⟩

/-- **One sweep step keeps the invariant** (`up` the chordal rows of `adj`, `wt` the diagonal-free edge weights
    inside the skeleton). -/
theorem metricInv_step (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (wt : Nat → Nat → Wt)
    (hcov : ∀ u v c, wt u v = some c → v ∈ adj u) (k : Nat) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat)
    (h : MetricInv (rowOf adj) wt k A M) :
    MetricInv (rowOf adj) wt (k + 1) (relaxA (rowOf adj) k A) (relaxM (rowOf adj) k A M) := by
  have hgt := row_gt adj hs
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun u v harc => ?_, fun u v harc => ?_⟩
  · by_cases hm : k < min u v
    · -- both ends above `k`: the detour through `k` is offered
      have hu : k < u := by omega
      have hv : k < v := by omega
      have hb : min k (min u v) = k := by omega
      have hb' : min (k + 1) (min u v) = k + 1 := by omega
      rw [hb']
      have hd0 : IsMin wt k u v (A u v) := by have := h1 u v harc; rwa [hb] at this
      have hg1 : IsMin wt k u k (if u ∈ rowOf adj k then A u k else none) := by
        by_cases hmem : u ∈ rowOf adj k
        · rw [ifP hmem]
          have := h1 u k ⟨by omega, Or.inr ⟨hu, hmem⟩⟩
          have e : min k (min u k) = k := by omega
          rwa [e] at this
        · rw [ifN hmem]; exact isMin_none_of_not_up adj hs wt hcov k u hu hmem
      have hg2 : IsMin wt k k v (if v ∈ rowOf adj k then A k v else none) := by
        by_cases hmem : v ∈ rowOf adj k
        · rw [ifP hmem]
          have := h1 k v ⟨by omega, Or.inl ⟨hv, hmem⟩⟩
          have e : min k (min k v) = k := by omega
          rwa [e] at this
        · rw [ifN hmem]; exact isMin_none_of_not_up' adj hs wt hcov k v hv hmem
      have hfw := isMin_fw wt k u v _ _ _ hd0 hg1 hg2
      have hne : u ≠ v := harc.1
      have heq : relaxA (rowOf adj) k A u v =
          wmin (A u v) (wadd (if u ∈ rowOf adj k then A u k else none) (if v ∈ rowOf adj k then A k v else none)) := by
        unfold relaxA
        by_cases hc : u ∈ rowOf adj k ∧ v ∈ rowOf adj k ∧ u ≠ v
        · rw [ifP hc, ifP hc.1, ifP hc.2.1]
        · rw [ifN hc]
          have : wadd (if u ∈ rowOf adj k then A u k else none) (if v ∈ rowOf adj k then A k v else none) = none := by
            by_cases a1 : u ∈ rowOf adj k
            · have a2 : v ∉ rowOf adj k := fun a2 => hc ⟨a1, a2, hne⟩
              simp [a2, wadd]
            · simp [a1, wadd]
          rw [this]; simp [wmin, wlt]
      rw [heq]; exact hfw
    · have hu : ¬ (k < u ∧ k < v) := by omega
      rw [relaxA_unchanged _ hgt k A u v hu]
      have := h1 u v harc
      have e : min (k + 1) (min u v) = min k (min u v) := by omega
      rw [e]; exact this
  · by_cases hc : (u ∈ rowOf adj k ∧ v ∈ rowOf adj k ∧ u ≠ v ∧ wlt (wadd (A u k) (A k v)) (A u v) = true)
    · obtain ⟨hu, hv, hne, hlt⟩ := hc
      have hM : relaxM (rowOf adj) k A M u v = some k := by
        unfold relaxM; rw [ifP ⟨hu, hv, hne, hlt⟩]
      have hA : relaxA (rowOf adj) k A u v = wadd (A u k) (A k v) := by
        unfold relaxA; rw [ifP ⟨hu, hv, hne⟩]; unfold wmin; rw [ifP hlt]
      have hku : k < u := hgt _ _ hu
      have hkv : k < v := hgt _ _ hv
      obtain ⟨e1, e2, he1, he2⟩ : ∃ e1 e2, A u k = some e1 ∧ A k v = some e2 := by
        cases h1' : A u k with
        | none => simp [h1', wadd, wlt] at hlt
        | some e1 =>
          cases h2' : A k v with
          | none => simp [h1', h2', wadd, wlt] at hlt
          | some e2 => exact ⟨e1, e2, rfl, rfl⟩
      refine ⟨fun h => absurd (hM.symm.trans h) (by simp), fun m hm => ?_⟩
      obtain rfl : k = m := Option.some.inj (hM.symm.trans hm)
      refine ⟨by omega, by omega, ⟨by omega, Or.inr ⟨hku, hu⟩⟩, ⟨by omega, Or.inl ⟨hkv, hv⟩⟩, e1, e2, ?_, ?_, ?_⟩
      · rw [relaxA_unchanged _ hgt k A u k (by omega)]; exact he1
      · rw [relaxA_unchanged _ hgt k A k v (by omega)]; exact he2
      · rw [hA, he1, he2]; rfl
    · have hM : relaxM (rowOf adj) k A M u v = M u v := by unfold relaxM; rw [ifN hc]
      have hA : relaxA (rowOf adj) k A u v = A u v := by
        unfold relaxA
        by_cases hc' : u ∈ rowOf adj k ∧ v ∈ rowOf adj k ∧ u ≠ v
        · rw [ifP hc']; unfold wmin
          have : ¬ wlt (wadd (A u k) (A k v)) (A u v) = true := fun h => hc ⟨hc'.1, hc'.2.1, hc'.2.2, h⟩
          rw [ifN this]
        · rw [ifN hc']
      obtain ⟨hn, hsome⟩ := h2 u v harc
      rw [hM, hA]
      refine ⟨hn, fun m hm => ?_⟩
      obtain ⟨a1, a2, a3, a4, e1, e2, he1, he2, he⟩ := hsome m hm
      refine ⟨a1, by omega, a3, a4, e1, e2, ?_, ?_, he⟩
      · rw [relaxA_unchanged _ hgt k A u m (by omega)]; exact he1
      · rw [relaxA_unchanged _ hgt k A m v (by omega)]; exact he2

theorem metricInv_init (up : Nat → List Nat) (w : Nat → Nat → Wt) :
    MetricInv up (seedA w) 0 (seedA w) (fun _ _ => none) := by
  refine ⟨fun u v harc => ?_, fun u v harc => ⟨fun _ => rfl, fun m hm => by cases hm⟩⟩
  have hne : u ≠ v := harc.1
  have hb : min 0 (min u v) = 0 := by omega
  rw [hb]
  constructor
  · intro c hc
    cases hc with
    | edge he => exact ⟨c, he, Nat.le_refl _⟩
    | step _ hb' _ => omega
  · intro e he
    exact .edge he

/-- **Customization is correct.** After the sweep over ranks `0 .. n-1`, for every arc `{u, v}` of the chordal
    supergraph, the weight `A u v` (`upward_weight` for `u < v`, `downward_weight` for `u > v`) is exactly the
    cheapest walk `u → v` in the original graph whose intermediate vertices all rank below `min u v`
    (`IsMin`: some walk of that weight exists, none is cheaper, and `none` = `+inf` iff there is no walk); and its
    middle, if any, is a strictly lower rank `m` with `A u v = A u m + A m v`, `None` meaning `A u v` is the
    original edge's own weight. Weights are `Nat`; `adj` is the symmetric skeleton, `w` the edge weights, which
    must lie inside the skeleton (the `CchInput` contract). -/
theorem customize_correct (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (w : Nat → Nat → Wt)
    (hcov : ∀ u v c, seedA w u v = some c → v ∈ adj u) (n : Nat) :
    let r := customizeRun (rowOf adj) w n
    ∀ u v, u < n → v < n → ArcP (rowOf adj) u v →
      IsMin (seedA w) (min u v) u v (r.1 u v) ∧ (r.2 u v = none → r.1 u v = seedA w u v) ∧
      ∀ m, r.2 u v = some m → m < min u v ∧ ArcP (rowOf adj) u m ∧ ArcP (rowOf adj) m v ∧
        ∃ e1 e2, r.1 u m = some e1 ∧ r.1 m v = some e2 ∧ r.1 u v = some (e1 + e2) := by
  intro r
  have hfin : ∀ n, MetricInv (rowOf adj) (seedA w) n (customizeRun (rowOf adj) w n).1 (customizeRun (rowOf adj) w n).2 := by
    intro n
    induction n with
    | zero => exact metricInv_init (rowOf adj) w
    | succ n ih =>
      unfold customizeRun at *
      rw [List.range_succ, List.foldl_append]
      exact metricInv_step adj hs (seedA w) hcov n _ _ ih
  have hfin := hfin n
  obtain ⟨h1, h2⟩ := hfin
  intro u v hu hv harc
  have e : min n (min u v) = min u v := by omega
  refine ⟨?_, (h2 u v harc).1, fun m hm => ?_⟩
  · have := h1 u v harc; rwa [e] at this
  · obtain ⟨a1, _, a3, a4, a5⟩ := (h2 u v harc).2 m hm
    exact ⟨a1, a3, a4, a5⟩

/-! ## Path unpacking (`query.rs`) -/

/-- The walk `a → l₀ → l₁ → ...` uses original edges only, with total weight `c`. -/
inductive EdgesFrom (wt : Nat → Nat → Wt) : Nat → List Nat → Nat → Prop where
  | nil {a : Nat} : EdgesFrom wt a [] 0
  | cons {a x : Nat} {l : List Nat} {w c : Nat} : wt a x = some w → EdgesFrom wt x l c →
      EdgesFrom wt a (x :: l) (w + c)

/-- Where the walk `a → l` ends. -/
def lastOr (a : Nat) : List Nat → Nat
  | [] => a
  | x :: l => lastOr x l

theorem EdgesFrom.append {wt : Nat → Nat → Wt} {a : Nat} {l1 : List Nat} {c1 : Nat} (h : EdgesFrom wt a l1 c1) :
    ∀ {l2 c2}, EdgesFrom wt (lastOr a l1) l2 c2 → EdgesFrom wt a (l1 ++ l2) (c1 + c2) := by
  induction h with
  | nil => intro l2 c2 h2; simpa [lastOr] using h2
  | @cons a x l w c hw _ ih =>
    intro l2 c2 h2
    have := EdgesFrom.cons hw (ih (by simpa [lastOr] using h2))
    simpa [Nat.add_assoc] using this

/-- `unpack`'s expansion of the arc `a → b` (`M a b` its middle), as a recursion on fuel: an arc with no middle is
    its far end, one with middle `m` is `a → m` then `m → b`; a missing arc fails. -/
def unpackRec (up : Nat → List Nat) (M : Nat → Nat → Option Nat) : Nat → Nat → Nat → Option (List Nat)
  | 0, _, _ => none
  | f + 1, a, b =>
    if a ≠ b ∧ ((a < b ∧ b ∈ up a) ∨ (b < a ∧ a ∈ up b)) then
      match M a b with
      | none => some [b]
      | some m => (unpackRec up M f a m).bind (fun l1 => (unpackRec up M f m b).map (fun l2 => l1 ++ l2))
    else none

theorem lastOr_append (a : Nat) (l1 l2 : List Nat) : lastOr a (l1 ++ l2) = lastOr (lastOr a l1) l2 := by
  induction l1 generalizing a with
  | nil => rfl
  | cons x xs ih => exact ih x

/-- **Unpacking a customized arc gives a valid walk of the original graph with the arc's weight.** For an arc
    `a b` of finite weight `c` (and any fuel above `a + b`, which the Rust's explicit stack replaces), the
    expansion succeeds and is a walk `a → ... → b` over original edges of total weight exactly `c`. -/
theorem unpack_correct (up : Nat → List Nat) (wt : Nat → Nat → Wt) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat)
    (hM : ∀ u v, ArcP up u v → (M u v = none → A u v = wt u v) ∧
      ∀ m, M u v = some m → m < min u v ∧ ArcP up u m ∧ ArcP up m v ∧
        ∃ e1 e2, A u m = some e1 ∧ A m v = some e2 ∧ A u v = some (e1 + e2)) :
    ∀ f a b c, a + b < f → ArcP up a b → A a b = some c →
      ∃ l, unpackRec up M f a b = some l ∧ lastOr a l = b ∧ EdgesFrom wt a l c := by
  intro f
  induction f with
  | zero => intro a b c h; omega
  | succ f ih =>
    intro a b c hf harc hA
    have hcond : a ≠ b ∧ ((a < b ∧ b ∈ up a) ∨ (b < a ∧ a ∈ up b)) := harc
    obtain ⟨hn, hsome⟩ := hM a b harc
    cases hmid : M a b with
    | none =>
      have hw : wt a b = some c := by rw [← hn hmid]; exact hA
      refine ⟨[b], ?_, rfl, ?_⟩
      · simp [unpackRec, hcond, hmid]
      · have := EdgesFrom.cons hw (EdgesFrom.nil (wt := wt) (a := b))
        simpa using this
    | some m =>
      obtain ⟨hm, hr1, hr2, e1, e2, h1, h2, h3⟩ := hsome m hmid
      have hc : c = e1 + e2 := by rw [h3] at hA; exact (Option.some.inj hA).symm
      have hm1 : m < a := by omega
      have hm2 : m < b := by omega
      obtain ⟨l1, hl1, hv1, hw1⟩ := ih a m e1 (by omega) hr1 h1
      obtain ⟨l2, hl2, hv2, hw2⟩ := ih m b e2 (by omega) hr2 h2
      refine ⟨l1 ++ l2, ?_, ?_, ?_⟩
      · simp [unpackRec, hcond, hmid, hl1, hl2]
      · rw [lastOr_append, hv1]; exact hv2
      · rw [hc]; exact hw1.append (by rw [hv1]; exact hw2)

end NI
