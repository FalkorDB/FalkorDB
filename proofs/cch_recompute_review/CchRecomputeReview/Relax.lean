import CchRecomputeReview.Basic

/-!
# Theorem 1: `relax`, as written, is the full customization

`relax` (`customize.rs:264-312` @main d6fcd5aac; the same body at `customize.rs:491-539` @08c) is modelled loop for
loop: `relaxX` is the `x` loop over `topology.ranks()`, `relaxY` the `(i, y)` loop over `up(x)`, `relaxZ` the
`(j, z)` loop over `up(x)` after `i`, with the cursor walk into `up(y)`. A `none` is the
`panic!("chordal closure broken")` at `:291-293`.

* `relax_isFull`: if `relax` returns, every arc holds what 03 §6b says it reads (`IsFull`).
* `relax_some`: under chordality (**P-chordal**) it returns.
* `relax_none_of_not_chordal`: it panics on every non-chordal topology, so the panic is exactly P-chordal.
-/

namespace CchRecompute

variable {W : Type}

/-! ## The cursor walk -/

/-- `Iterator::position(|&above| above == z)`. -/
def position : List Nat → Nat → Option Nat
  | [], _ => none
  | a :: t, z => if a = z then some 0 else (position t z).map (· + 1)

/-- `customize.rs:288-294`: `cursor += above_y[cursor..].iter().position(..)`; `none` is the panic. -/
def cursorStep (row : List Nat) (cursor z : Nat) : Option Nat :=
  (position (row.drop cursor) z).map (cursor + ·)

theorem position_mem : ∀ {l : List Nat} {z p : Nat}, position l z = some p → z ∈ l
  | [], _, _, h => by simp [position] at h
  | a :: t, z, p, h => by
    simp only [position] at h
    split at h
    · rename_i hz; subst hz; exact List.mem_cons_self
    · cases hp : position t z with
      | none => rw [hp] at h; simp at h
      | some q => exact List.mem_cons_of_mem _ (position_mem hp)

theorem position_isSome : ∀ {l : List Nat} {z : Nat}, z ∈ l → ∃ p, position l z = some p
  | [], _, h => by simp at h
  | a :: t, z, h => by
    simp only [position]
    split
    · exact ⟨0, rfl⟩
    · rename_i hne
      have : z ∈ t := by
        rcases List.mem_cons.mp h with h | h
        · exact absurd h.symm hne
        · exact h
      obtain ⟨q, hq⟩ := position_isSome this
      exact ⟨q + 1, by simp [hq]⟩

/-- In an ascending row, everything above the found `z` is at or after its position. -/
theorem mem_drop_position : ∀ {l : List Nat} {z p z' : Nat}, l.Pairwise (· < ·) →
    position l z = some p → z' ∈ l → z < z' → z' ∈ l.drop p
  | [], _, _, _, _, h, _, _ => by simp [position] at h
  | a :: t, z, p, z', hs, h, hz', hlt => by
    simp only [position] at h
    split at h
    · cases h; simpa using hz'
    · rename_i hne
      cases hq : position t z with
      | none => rw [hq] at h; simp at h
      | some q =>
        rw [hq] at h; simp at h; subst h
        have hzt := position_mem hq
        rcases List.mem_cons.mp hz' with h1 | h1
        · subst h1
          have := (List.pairwise_cons.mp hs).1 z hzt
          omega
        · simpa using mem_drop_position (List.pairwise_cons.mp hs).2 hq h1 hlt

theorem Pairwise.drop {l : List Nat} (h : l.Pairwise (· < ·)) (k : Nat) : (l.drop k).Pairwise (· < ·) :=
  h.sublist (List.drop_sublist _ _)

/-! ## The loops -/

/-- `customize.rs:284-309`: the `z` loop of one `(x, y)`, `x_to_y`/`y_to_x` read once before it (`:275-276`).
The write goes to slot `y_slots_start + cursor`, the arc `(y, up(y)[cursor]) = (y, z)`. -/
def relaxZ (o : Ops W) (T : Topo) (x y : Nat) (xToY yToX : W) :
    Nat → List Nat → Metric W → Option (Metric W)
  | _, [], M => some M
  | cursor, z :: zs, M =>
    let xToZ := (M x z).up
    let zToX := (M x z).dn
    match cursorStep (T.up y) cursor z with
    | none => none
    | some c =>
      let e := M y z
      let yzx := o.add yToX xToZ
      let zyx := o.add zToX xToY
      let e' : Entry W :=
        { up := if o.lt yzx e.up then yzx else e.up
          dn := if o.lt zyx e.dn then zyx else e.dn
          mup := if o.lt yzx e.up then some x else e.mup
          mdn := if o.lt zyx e.dn then some x else e.mdn }
      relaxZ o T x y xToY yToX c zs (M.set y z e')

/-- `customize.rs:272-310`: for each `y` of `up(x)`, the `z`s after it. -/
def relaxY (o : Ops W) (T : Topo) (x : Nat) : List Nat → Metric W → Option (Metric W)
  | [], M => some M
  | y :: rest, M =>
    match relaxZ o T x y (M x y).up (M x y).dn 0 rest M with
    | none => none
    | some M' => relaxY o T x rest M'

/-- `customize.rs:268`: `for x in topology.ranks()`. -/
def relaxX (o : Ops W) (T : Topo) : List Nat → Metric W → Option (Metric W)
  | [], M => some M
  | x :: xs, M =>
    match relaxY o T x (T.up x) M with
    | none => none
    | some M' => relaxX o T xs M'

/-- `relax(topology, metric)`, `none` = the panic. -/
def relax (o : Ops W) (T : Topo) (M : Metric W) : Option (Metric W) := relaxX o T (List.range T.n) M

/-! ## What one `(x, y)` loop writes -/

/-- One step of the `z` loop writes `upd` at `(y, z)`: `y_to_x + x_to_z` is `dn(x,y) + up(x,z)`, and
`z_to_x + x_to_y` is `dn(x,z) + up(x,y)`. -/
theorem relaxZ_cons (o : Ops W) (T : Topo) {x y cursor c z : Nat} {zs : List Nat} {xToY yToX : W}
    {M : Metric W} (hu : (M x y).up = xToY) (hd : (M x y).dn = yToX)
    (hc : cursorStep (T.up y) cursor z = some c) :
    relaxZ o T x y xToY yToX cursor (z :: zs) M =
      relaxZ o T x y xToY yToX c zs (M.set y z (upd o M x y z (M y z))) := by
  simp only [relaxZ, hc, upd, hu, hd]

theorem relaxZ_cons_none (o : Ops W) (T : Topo) {x y cursor z : Nat} {zs : List Nat} {xToY yToX : W}
    {M : Metric W} (hc : cursorStep (T.up y) cursor z = none) :
    relaxZ o T x y xToY yToX cursor (z :: zs) M = none := by
  simp only [relaxZ, hc]

theorem relaxZ_spec (o : Ops W) (T : Topo) {x y : Nat} {xToY yToX : W} (hxy : x < y) :
    ∀ (Z : List Nat) (c : Nat) (M M' : Metric W), Z.Pairwise (· < ·) → (∀ z ∈ Z, y < z) →
      (M x y).up = xToY → (M x y).dn = yToX →
      relaxZ o T x y xToY yToX c Z M = some M' →
      (∀ z ∈ Z, M' y z = upd o M x y z (M y z)) ∧ (∀ a b, ¬(a = y ∧ b ∈ Z) → M' a b = M a b)
  | [], c, M, M', _, _, _, _, h => by
    simp [relaxZ] at h; subst h; simp
  | z :: zs, c, M, M', hs, hgt, hu, hd, h => by
    cases hc : cursorStep (T.up y) c z with
    | none => rw [relaxZ_cons_none o T hc] at h; simp at h
    | some c' =>
      rw [relaxZ_cons o T hu hd hc] at h
      have hs' := (List.pairwise_cons.mp hs)
      have hxy' : x ≠ y := Nat.ne_of_lt hxy
      have hM1xy : (M.set y z (upd o M x y z (M y z))) x y = M x y := Metric.set_ne _ _ (by omega)
      obtain ⟨ih1, ih2⟩ := relaxZ_spec o T hxy zs c' _ M' hs'.2
        (fun w hw => hgt w (List.mem_cons_of_mem _ hw)) (by rw [hM1xy, hu]) (by rw [hM1xy, hd]) h
      refine ⟨?_, ?_⟩
      · intro w hw
        rcases List.mem_cons.mp hw with rfl | hw
        · have hz : w ∉ zs := fun hm => Nat.lt_irrefl w (hs'.1 w hm)
          rw [ih2 y w (fun ⟨_, hm⟩ => hz hm), Metric.set_same]
        · have hne : z ≠ w := Nat.ne_of_lt (hs'.1 w hw)
          rw [ih1 w hw]
          have h1 : (M.set y z (upd o M x y z (M y z))) y w = M y w :=
            Metric.set_ne _ _ (fun ⟨_, h⟩ => hne h.symm)
          have h2 : (M.set y z (upd o M x y z (M y z))) x w = M x w :=
            Metric.set_ne _ _ (fun ⟨h, _⟩ => hxy' h)
          rw [h1]
          exact upd_ext o _ hM1xy h2
      · intro a b hab
        have hab1 : ¬(a = y ∧ b ∈ zs) := fun ⟨h1, h2⟩ => hab ⟨h1, List.mem_cons_of_mem _ h2⟩
        rw [ih2 a b hab1]
        exact Metric.set_ne _ _ (fun ⟨h1, h2⟩ => hab ⟨h1, h2 ▸ List.mem_cons_self⟩)

theorem relaxY_spec (o : Ops W) (T : Topo) {x : Nat} :
    ∀ (L : List Nat) (M M' : Metric W), L.Pairwise (· < ·) → (∀ y ∈ L, x < y) →
      relaxY o T x L M = some M' →
      (∀ y z, y ∈ L → z ∈ L → y < z → M' y z = upd o M x y z (M y z)) ∧
      (∀ a b, ¬(a ∈ L ∧ b ∈ L ∧ a < b) → M' a b = M a b)
  | [], M, M', _, _, h => by simp [relaxY] at h; subst h; simp
  | y :: rest, M, M', hs, hgt, h => by
    simp only [relaxY] at h
    split at h
    · simp at h
    · rename_i M1 hZ
      have hs' := List.pairwise_cons.mp hs
      have hxy := hgt y List.mem_cons_self
      obtain ⟨z1, z2⟩ := relaxZ_spec o T hxy rest 0 M M1 hs'.2 hs'.1 rfl rfl hZ
      obtain ⟨y1, y2⟩ := relaxY_spec o T rest M1 M' hs'.2
        (fun w hw => hgt w (List.mem_cons_of_mem _ hw)) h
      have hyrest : y ∉ rest := fun hm => Nat.lt_irrefl y (hs'.1 y hm)
      refine ⟨?_, ?_⟩
      · intro a b ha hb hab
        rcases List.mem_cons.mp ha with rfl | ha'
        · rcases List.mem_cons.mp hb with rfl | hb'
          · exact absurd hab (Nat.lt_irrefl _)
          · rw [y2 a b (fun ⟨h, _, _⟩ => hyrest h), z1 b hb']
        · rcases List.mem_cons.mp hb with hby | hb'
          · subst hby; exact absurd hab (Nat.lt_asymm (hs'.1 a ha'))
          · rw [y1 a b ha' hb' hab]
            have hay : a ≠ y := fun h => hyrest (h ▸ ha')
            have hxy' : x ≠ y := Nat.ne_of_lt hxy
            rw [z2 a b (fun ⟨h, _⟩ => hay h)]
            exact upd_ext o _ (z2 x a (fun ⟨h, _⟩ => hxy' h)) (z2 x b (fun ⟨h, _⟩ => hxy' h))
      · intro a b hab
        have h1 : ¬(a ∈ rest ∧ b ∈ rest ∧ a < b) := fun ⟨p, q, r⟩ =>
          hab ⟨List.mem_cons_of_mem _ p, List.mem_cons_of_mem _ q, r⟩
        rw [y2 a b h1]
        apply z2
        rintro ⟨rfl, hb⟩
        exact hab ⟨List.mem_cons_self, List.mem_cons_of_mem _ hb, hs'.1 b hb⟩

/-! ## The invariant of the `x` loop -/

/-- After the apexes below `k`: every arc holds the fold over its apexes below `k`. -/
def RInv (o : Ops W) (T : Topo) (S : Seeds W) (k : Nat) (M : Metric W) : Prop :=
  ∀ y z, z ∈ T.up y →
    M y z = ((List.range (min y k)).filter (T.apex y z)).foldl (fun e x => upd o M x y z e) (seedE S y z)

theorem rinv_zero (o : Ops W) (T : Topo) (S : Seeds W) : RInv o T S 0 (seedM S) := by
  intro y z _; simp [seedM]

theorem rinv_step (o : Ops W) {T : Topo} (hV : T.Valid) {S : Seeds W} {k : Nat} {M M' : Metric W}
    (hI : RInv o T S k M) (h : relaxY o T k (T.up k) M = some M') : RInv o T S (k + 1) M' := by
  obtain ⟨w1, w2⟩ := relaxY_spec o T (T.up k) M M' (hV.sorted k) (hV.above k) h
  -- rows at or below k are not written
  have hlow : ∀ a b, a ≤ k → M' a b = M a b := fun a b ha =>
    w2 a b (fun ⟨h1, _, _⟩ => by have := hV.above k a h1; omega)
  have hfold : ∀ (l : List Nat) (y z : Nat) (e : Entry W), (∀ x ∈ l, x ≤ k) →
      l.foldl (fun e x => upd o M' x y z e) e = l.foldl (fun e x => upd o M x y z e) e :=
    fun l y z e hl => foldl_ext _ _ l e (fun x hx a => upd_ext o a (hlow x y (hl x hx)) (hlow x z (hl x hx)))
  intro y z hyz
  have hyz' := hV.above y z hyz
  by_cases hky : k < y
  · have e1 : min y (k + 1) = k + 1 := by omega
    have e0 : min y k = k := by omega
    have hI' := hI y z hyz
    rw [e0] at hI'
    rw [e1, List.range_succ, List.filter_append, List.foldl_append]
    have hr : ∀ x ∈ (List.range k).filter (T.apex y z), x ≤ k := fun x hx => by
      have := (List.mem_range.mp (List.mem_filter.mp hx).1); omega
    rw [hfold _ y z _ hr, ← hI']
    by_cases hp : T.apex y z k = true
    · have hp' : y ∈ T.up k ∧ z ∈ T.up k := by simpa [Topo.apex] using hp
      rw [List.filter_cons_of_pos hp, List.filter_nil, List.foldl_cons, List.foldl_nil]
      rw [w1 y z hp'.1 hp'.2 hyz']
      exact upd_ext o _ (hlow k y (Nat.le_refl _)).symm (hlow k z (Nat.le_refl _)).symm
    · rw [List.filter_cons_of_neg hp, List.filter_nil, List.foldl_nil]
      apply w2
      rintro ⟨h1, h2, _⟩
      exact hp (by simp [Topo.apex, h1, h2])
  · have e1 : min y (k + 1) = y := by omega
    have e0 : min y k = y := by omega
    have hI' := hI y z hyz
    rw [e0] at hI'
    rw [e1]
    have hr : ∀ x ∈ (List.range y).filter (T.apex y z), x ≤ k := fun x hx => by
      have := (List.mem_range.mp (List.mem_filter.mp hx).1); omega
    rw [hfold _ y z _ hr, ← hI']
    apply w2
    rintro ⟨h1, _, _⟩
    have := hV.above k y h1; omega

theorem relaxX_spec (o : Ops W) {T : Topo} (hV : T.Valid) (S : Seeds W) :
    ∀ (m k : Nat) (M R : Metric W), RInv o T S k M → relaxX o T (List.range' k m) M = some R →
      RInv o T S (k + m) R
  | 0, k, M, R, hI, h => by simp [relaxX] at h; subst h; simpa using hI
  | m + 1, k, M, R, hI, h => by
    simp only [List.range'_succ, relaxX] at h
    split at h
    · simp at h
    · rename_i M1 hY
      have := relaxX_spec o hV S m (k + 1) M1 R (rinv_step o hV hI hY) (by simpa using h)
      simpa [Nat.add_assoc, Nat.add_comm 1 m] using this

/-- **Theorem 1.** If `relax` returns, every arc `a = (lo, hi)` holds
`min(seed(a), min over x ∈ down(lo) ∩ down(hi) of dn(x,lo) + up(x,hi))` and the mirror direction, each kept
under a strict `<` in increasing `x`, the middle being the apex of the last strict improvement (`IsFull`). -/
theorem relax_isFull (o : Ops W) {T : Topo} (hV : T.Valid) (S : Seeds W) {R : Metric W}
    (h : relax o T (seedM S) = some R) : IsFull o T S R := by
  have hI : RInv o T S T.n R := by
    have := relaxX_spec o hV S T.n 0 (seedM S) R (rinv_zero o T S)
      (by simpa [relax, List.range_eq_range'] using h)
    simpa using this
  intro lo hi harc
  have hlo : lo < T.n := Nat.lt_trans (hV.above lo hi harc) (hV.bound lo hi harc)
  have := hI lo hi harc
  rwa [Nat.min_eq_left (Nat.le_of_lt hlo)] at this

/-! ## When `relax` returns -/

theorem relaxZ_some (o : Ops W) (T : Topo) (hV : T.Valid) {x y : Nat} {xToY yToX : W} :
    ∀ (Z : List Nat) (c : Nat) (M : Metric W), Z.Pairwise (· < ·) → (∀ z ∈ Z, z ∈ (T.up y).drop c) →
      ∃ M', relaxZ o T x y xToY yToX c Z M = some M'
  | [], _, M, _, _ => ⟨M, rfl⟩
  | z :: zs, c, M, hs, hin => by
    obtain ⟨p, hp⟩ := position_isSome (hin z List.mem_cons_self)
    have hcs : cursorStep (T.up y) c z = some (c + p) := by simp [cursorStep, hp]
    have hs' := List.pairwise_cons.mp hs
    have hin' : ∀ w ∈ zs, w ∈ (T.up y).drop (c + p) := fun w hw => by
      rw [← List.drop_drop]
      exact mem_drop_position (Pairwise.drop (hV.sorted y) c) hp
        (hin w (List.mem_cons_of_mem _ hw)) (hs'.1 w hw)
    simp only [relaxZ, hcs]
    exact relaxZ_some o T hV zs _ _ hs'.2 hin'

theorem relaxY_some (o : Ops W) (T : Topo) (hV : T.Valid) (hC : T.Chordal) {x : Nat} :
    ∀ (L : List Nat) (M : Metric W), L.Pairwise (· < ·) → (∀ y ∈ L, y ∈ T.up x) →
      ∃ M', relaxY o T x L M = some M'
  | [], M, _, _ => ⟨M, rfl⟩
  | y :: rest, M, hs, hin => by
    have hs' := List.pairwise_cons.mp hs
    obtain ⟨M1, h1⟩ := relaxZ_some o T hV (x := x) (xToY := (M x y).up) (yToX := (M x y).dn) rest 0 M hs'.2
      (fun z hz => by
        simpa using hC x y z (hin y List.mem_cons_self) (hin z (List.mem_cons_of_mem _ hz)) (hs'.1 z hz))
    obtain ⟨M2, h2⟩ := relaxY_some o T hV hC rest M1 hs'.2 (fun w hw => hin w (List.mem_cons_of_mem _ hw))
    exact ⟨M2, by simp only [relaxY, h1, h2]⟩

theorem relaxX_some (o : Ops W) (T : Topo) (hV : T.Valid) (hC : T.Chordal) :
    ∀ (L : List Nat) (M : Metric W), ∃ R, relaxX o T L M = some R
  | [], M => ⟨M, rfl⟩
  | x :: xs, M => by
    obtain ⟨M1, h1⟩ := relaxY_some o T hV hC (T.up x) M (hV.sorted x) (fun _ h => h)
    obtain ⟨R, h2⟩ := relaxX_some o T hV hC xs M1
    exact ⟨R, by simp only [relaxX, h1, h2]⟩

/-- **Theorem 1, existence.** Under P-chordal `relax` never panics, and what it returns is the full
customization. -/
theorem relax_some (o : Ops W) {T : Topo} (hV : T.Valid) (hC : T.Chordal) (S : Seeds W) :
    ∃ R, relax o T (seedM S) = some R ∧ IsFull o T S R := by
  obtain ⟨R, h⟩ := relaxX_some o T hV hC (List.range T.n) (seedM S)
  exact ⟨R, h, relax_isFull o hV S h⟩

/-! ## When `relax` panics -/

theorem position_none {l : List Nat} {z : Nat} (h : z ∉ l) : position l z = none := by
  cases hp : position l z with
  | none => rfl
  | some p => exact absurd (position_mem hp) h

theorem relaxZ_none (o : Ops W) (T : Topo) {x y : Nat} {xToY yToX : W} :
    ∀ (Z : List Nat) (c : Nat) (M : Metric W), (∃ z ∈ Z, z ∉ T.up y) →
      relaxZ o T x y xToY yToX c Z M = none
  | [], _, _, ⟨_, hz, _⟩ => by simp at hz
  | z :: zs, c, M, ⟨z', hz', hn⟩ => by
    cases hc : cursorStep (T.up y) c z with
    | none => simp only [relaxZ, hc]
    | some c' =>
      simp only [relaxZ, hc]
      rcases List.mem_cons.mp hz' with rfl | hz'
      · exfalso
        simp only [cursorStep] at hc
        rw [position_none (fun hm => hn (List.mem_of_mem_drop hm))] at hc
        simp at hc
      · exact relaxZ_none o T zs c' _ ⟨z', hz', hn⟩

theorem relaxY_none (o : Ops W) (T : Topo) {x : Nat} :
    ∀ (L : List Nat) (M : Metric W), L.Pairwise (· < ·) →
      (∃ y ∈ L, ∃ z ∈ L, y < z ∧ z ∉ T.up y) → relaxY o T x L M = none
  | [], _, _, ⟨_, hy, _⟩ => by simp at hy
  | y0 :: rest, M, hs, ⟨y, hy, z, hz, hlt, hn⟩ => by
    have hs' := List.pairwise_cons.mp hs
    simp only [relaxY]
    by_cases hy0 : y = y0
    · subst hy0
      have hzr : z ∈ rest := by
        rcases List.mem_cons.mp hz with rfl | hz
        · exact absurd hlt (Nat.lt_irrefl _)
        · exact hz
      rw [relaxZ_none o T rest 0 M ⟨z, hzr, hn⟩]
    · have hyr : y ∈ rest := by
        rcases List.mem_cons.mp hy with h | h
        · exact absurd h hy0
        · exact h
      have hzr : z ∈ rest := by
        rcases List.mem_cons.mp hz with rfl | hz
        · have := hs'.1 y hyr; omega
        · exact hz
      split
      · rfl
      · exact relaxY_none o T rest _ hs'.2 ⟨y, hyr, z, hzr, hlt, hn⟩

theorem relaxX_none (o : Ops W) {T : Topo} (hV : T.Valid) :
    ∀ (L : List Nat) (M : Metric W), (∃ x ∈ L, ∃ y ∈ T.up x, ∃ z ∈ T.up x, y < z ∧ z ∉ T.up y) →
      relaxX o T L M = none
  | [], _, ⟨_, hx, _⟩ => by simp at hx
  | x0 :: xs, M, ⟨x, hx, hbad⟩ => by
    simp only [relaxX]
    by_cases hx0 : x = x0
    · subst hx0
      rw [relaxY_none o T (T.up x) M (hV.sorted x) hbad]
    · have hxs : x ∈ xs := by
        rcases List.mem_cons.mp hx with h | h
        · exact absurd h hx0
        · exact h
      split
      · rfl
      · exact relaxX_none o hV xs _ ⟨x, hxs, hbad⟩

/-- **The panic is exactly P-chordal.** On a topology with a `z` missing from `up(y)`, `relax` panics
(`customize.rs:291-293`), whatever the metric. -/
theorem relax_none_of_not_chordal (o : Ops W) {T : Topo} (hV : T.Valid) (hC : ¬T.Chordal) (M : Metric W) :
    relax o T M = none := by
  have : ∃ x y z, y ∈ T.up x ∧ z ∈ T.up x ∧ y < z ∧ z ∉ T.up y := by
    apply Classical.byContradiction
    intro hn
    apply hC
    intro x y z hy hz hlt
    apply Classical.byContradiction
    intro hzy
    exact hn ⟨x, y, z, hy, hz, hlt, hzy⟩
  obtain ⟨x, y, z, hy, hz, hlt, hn⟩ := this
  have hx : x < T.n := Nat.lt_trans (hV.above x y hy) (hV.bound x y hy)
  exact relaxX_none o hV _ M ⟨x, List.mem_range.mpr hx, y, hy, z, hz, hlt, hn⟩

end CchRecompute
