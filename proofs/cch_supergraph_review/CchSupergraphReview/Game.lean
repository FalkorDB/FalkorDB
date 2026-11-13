import FalkorNativeIndex.Tree
set_option linter.unusedVariables false

/-!
# The elimination game (03 §4 step 3; `order.rs:362-380` @main)

The game is played in rank space on an undirected graph. `G` is the rank-space skeleton as a relation (its
symmetric closure is taken, so the game is the textbook one whatever `G` is). Ranks are eliminated in increasing
order. Eliminating `k` joins every two of its not-yet-eliminated neighbours.

* `Adj G k a b`: `a` and `b` are adjacent once ranks `0 .. k-1` are eliminated (`Adj G 0` is the skeleton).
* `up G r w`: `w > r` is adjacent to `r` when `r` is eliminated. This is the game's upward row: no later step adds
  or removes an edge at `r` (`Adj_stable`), so it is also `r`'s row in the filled graph.
* `gameRow n G r`: that row as an ascending list, which is what `Topology::up` must return.
* `par n G r`: its first entry, the elimination-tree parent (Liu's etree: the lowest higher neighbour in the filled
  graph; `order.rs:375-380`).

Proved here, for every `G` whose entries lie below `n` (`Supp`):

* `clique`: **the filled graph is chordal in the CCH sense**: `y < z` both in `up(x)` gives `z ∈ up(y)`.
* `skeleton_in_up`: every skeleton edge is an upward arc of its lower end.
* `absorb_iff`: **Gilbert–Ng–Peyton child absorption is exact**: `up(r)` is `r`'s higher skeleton neighbours plus,
  for every etree child `c` of `r`, `up(c) ∖ {r}`. This is the recurrence `chordal_supergraph` runs
  (`order.rs:364-373`).
* `row_subtree`: `w ∈ up(r)` iff `r < w` and `w` is a skeleton neighbour of some etree descendant (or self) of `r`.
  It characterizes the column counts (03 §3 step 8, Gilbert–Ng–Peyton: the count of column `r` is the number of
  distinct higher neighbours of `r`'s subtree).
-/
namespace CchSG

open CowBTree (Sorted)

/-- The game's adjacency after ranks `0 .. k-1` are eliminated. -/
def Adj (G : Nat → Nat → Bool) : Nat → Nat → Nat → Bool
  | 0, a, b => G a b || G b a
  | k + 1, a, b => Adj G k a b || (decide (k < a) && decide (k < b) && (a != b) && Adj G k k a && Adj G k k b)

/-- `w` is a higher neighbour of `r` when `r` is eliminated: the game's upward row. -/
def up (G : Nat → Nat → Bool) (r w : Nat) : Bool := decide (r < w) && Adj G r r w

/-- Every entry of the skeleton is below `n`. -/
def Supp (n : Nat) (G : Nat → Nat → Bool) : Prop := ∀ a b, G a b = true → a < n ∧ b < n

/-- `r`'s upward row, ascending: what `Topology::up(r)` must be. -/
def gameRow (n : Nat) (G : Nat → Nat → Bool) (r : Nat) : List Nat := (List.range n).filter (up G r)

/-- The elimination-tree parent: the first entry of the row (`order.rs:426-427`). -/
def par (n : Nat) (G : Nat → Nat → Bool) (r : Nat) : Option Nat := (gameRow n G r).head?

variable {G : Nat → Nat → Bool}

@[simp] theorem Adj_zero (a b : Nat) : Adj G 0 a b = (G a b || G b a) := rfl

theorem Adj_succ (k a b : Nat) :
    Adj G (k + 1) a b = (Adj G k a b || (decide (k < a) && decide (k < b) && (a != b) && Adj G k k a && Adj G k k b)) := rfl

theorem Adj_symm : ∀ (k a b : Nat), Adj G k a b = Adj G k b a
  | 0, a, b => by simp [Bool.or_comm]
  | k + 1, a, b => by
    rw [Adj_succ, Adj_succ, Adj_symm k a b]
    cases Adj G k b a <;> simp only [Bool.false_or, Bool.true_or]
    cases decide (k < a) <;> cases decide (k < b) <;> cases Adj G k k a <;> cases Adj G k k b <;>
      simp [bne_comm]

theorem Adj_mono_succ {k a b : Nat} (h : Adj G k a b = true) : Adj G (k + 1) a b = true := by
  rw [Adj_succ, h]; rfl

theorem Adj_mono {k a b : Nat} (h : Adj G k a b = true) : ∀ j, Adj G (k + j) a b = true
  | 0 => h
  | j + 1 => Adj_mono_succ (Adj_mono h j)

theorem Adj_mono_le {k k' a b : Nat} (h : Adj G k a b = true) (hk : k ≤ k') : Adj G k' a b = true := by
  have := Adj_mono h (k' - k); rwa [Nat.add_sub_cancel' hk] at this

/-- An edge at `a` is never added once `a` is eliminated. -/
theorem Adj_stable_succ {k a b : Nat} (hk : a ≤ k) : Adj G (k + 1) a b = Adj G k a b := by
  rw [Adj_succ]
  have : decide (k < a) = false := by simp; omega
  simp [this]

theorem Adj_stable {a b : Nat} : ∀ j, Adj G (a + j) a b = Adj G a a b
  | 0 => rfl
  | j + 1 => by rw [← Nat.add_assoc, Adj_stable_succ (by omega), Adj_stable j]

/-- The game's edges: a skeleton edge, or a fill edge added when a common lower neighbour `c` was eliminated. -/
theorem Adj_char : ∀ (k a b : Nat), Adj G k a b = true ↔ (G a b || G b a) = true ∨
    ∃ c, c < k ∧ c < a ∧ c < b ∧ a ≠ b ∧ Adj G c c a = true ∧ Adj G c c b = true
  | 0, a, b => by simp
  | k + 1, a, b => by
    rw [Adj_succ, Bool.or_eq_true, Adj_char k a b]
    constructor
    · rintro ((h | ⟨c, h1, h2, h3, h4, h5, h6⟩) | h)
      · exact Or.inl h
      · exact Or.inr ⟨c, by omega, h2, h3, h4, h5, h6⟩
      · simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne] at h
        obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
        exact Or.inr ⟨k, by omega, h1, h2, h3, h4, h5⟩
    · rintro (h | ⟨c, h1, h2, h3, h4, h5, h6⟩)
      · exact Or.inl (Or.inl h)
      · by_cases hc : c < k
        · exact Or.inl (Or.inr ⟨c, hc, h2, h3, h4, h5, h6⟩)
        · have : c = k := by omega
          subst this
          right; simp [h2, h3, h4, h5, h6]

theorem up_lt {r w : Nat} (h : up G r w = true) : r < w := by
  simp [up] at h; exact h.1

theorem up_E {r w : Nat} (h : up G r w = true) : Adj G r r w = true := by
  simp [up] at h; exact h.2

theorem up_of {r w : Nat} (h1 : r < w) (h2 : Adj G r r w = true) : up G r w = true := by
  simp [up, h1, h2]

/-- **The upward row**: a higher skeleton neighbour, or a higher member of `up(c)` for a lower `c` with
    `r ∈ up(c)`. -/
theorem up_char (r w : Nat) : up G r w = true ↔ r < w ∧ ((G r w || G w r) = true ∨
    ∃ c, c < r ∧ up G c r = true ∧ up G c w = true) := by
  constructor
  · intro h
    refine ⟨up_lt h, ?_⟩
    rcases (Adj_char r r w).1 (up_E h) with h0 | ⟨c, h1, _, h3, _, h5, h6⟩
    · exact Or.inl h0
    · exact Or.inr ⟨c, h1, up_of h1 h5, up_of h3 h6⟩
  · rintro ⟨hlt, h0 | ⟨c, h1, h2, h3⟩⟩
    · exact up_of hlt ((Adj_char r r w).2 (Or.inl h0))
    · exact up_of hlt ((Adj_char r r w).2 (Or.inr ⟨c, h1, h1, up_lt h3, by omega, up_E h2, up_E h3⟩))

/-- **Chordal in the CCH sense** (03 §4 Contract, I4/I5): if `y < z` are both in `up(x)`, then `z ∈ up(y)`. -/
theorem clique {x y z : Nat} (hy : up G x y = true) (hz : up G x z = true) (hyz : y < z) : up G y z = true := by
  have hxy := up_lt hy
  have hxz := up_lt hz
  have h1 : Adj G (x + 1) y z = true := by
    rw [Adj_succ]; simp [hxy, hxz, Nat.ne_of_lt hyz, up_E hy, up_E hz]
  exact up_of hyz (Adj_mono_le h1 (by omega))

/-- **Every skeleton edge is an arc**: the higher end is in the lower end's upward row. -/
theorem skeleton_in_up {a b : Nat} (hab : a < b) (h : (G a b || G b a) = true) : up G a b = true :=
  up_of hab ((Adj_char a a b).2 (Or.inl h))

/-- The game's row is final once `r` is eliminated: in the filled graph `Adj G n`, `r`'s higher neighbours are
    exactly `up(r)`. -/
theorem up_final {r w : Nat} (j : Nat) (hrw : r < w) : Adj G (r + j) r w = true ↔ up G r w = true := by
  rw [Adj_stable]; simp [up, hrw]

/-! ## Support, the ascending row and the parent -/

theorem Adj_supp {n : Nat} (hs : Supp n G) : ∀ (k a b : Nat), a ≠ b → Adj G k a b = true → a < n ∧ b < n
  | k, a, b, hab, h => by
    rcases (Adj_char k a b).1 h with h0 | ⟨c, h1, h2, h3, h4, h5, h6⟩
    · rcases Bool.or_eq_true _ _ ▸ h0 with h0 | h0
      · exact hs a b h0
      · exact (hs b a h0).symm
    · exact ⟨(Adj_supp hs c c a (by omega) h5).2, (Adj_supp hs c c b (by omega) h6).2⟩

theorem up_supp {n : Nat} (hs : Supp n G) {r w : Nat} (h : up G r w = true) : r < n ∧ w < n :=
  Adj_supp hs r r w (Nat.ne_of_lt (up_lt h)) (up_E h)

theorem mem_gameRow {n : Nat} (hs : Supp n G) {r w : Nat} : w ∈ gameRow n G r ↔ up G r w = true := by
  simp only [gameRow, List.mem_filter, List.mem_range]
  exact ⟨fun h => h.2, fun h => ⟨(up_supp hs h).2, h⟩⟩

theorem gameRow_sorted (n : Nat) (G : Nat → Nat → Bool) (r : Nat) : Sorted (gameRow n G r) :=
  List.Pairwise.filter _ List.pairwise_lt_range

theorem head_min {l : List Nat} (hl : Sorted l) {p : Nat} (h : l.head? = some p) : p ∈ l ∧ ∀ w ∈ l, p ≤ w := by
  cases l with
  | nil => simp at h
  | cons a t =>
    simp only [List.head?_cons, Option.some.injEq] at h
    subst h
    refine ⟨List.mem_cons_self, fun w hw => ?_⟩
    rcases List.mem_cons.1 hw with rfl | hw
    · exact Nat.le_refl _
    · exact Nat.le_of_lt ((List.pairwise_cons.1 hl).1 w hw)

/-- The parent is the lowest member of the row. -/
theorem par_spec {n : Nat} (hs : Supp n G) {c p : Nat} (h : par n G c = some p) :
    up G c p = true ∧ ∀ w, up G c w = true → p ≤ w := by
  obtain ⟨hm, hmin⟩ := head_min (gameRow_sorted n G c) h
  exact ⟨(mem_gameRow hs).1 hm, fun w hw => hmin w ((mem_gameRow hs).2 hw)⟩

theorem par_exists {n : Nat} (hs : Supp n G) {c w : Nat} (h : up G c w = true) : ∃ p, par n G c = some p := by
  have hm : w ∈ gameRow n G c := (mem_gameRow hs).2 h
  cases hr : gameRow n G c with
  | nil => rw [hr] at hm; simp at hm
  | cons a t => exact ⟨a, by simp [par, hr]⟩

theorem par_lt {n : Nat} (hs : Supp n G) {c p : Nat} (h : par n G c = some p) : c < p :=
  up_lt (par_spec hs h).1

/-! ## Child absorption (Gilbert–Ng–Peyton), the recurrence `chordal_supergraph` runs -/

/-- From any lower `c` whose row holds `r` and `w > r`, climbing the elimination tree reaches a child of `r`
    whose row holds `w`. -/
theorem climb {n : Nat} (hs : Supp n G) {r w : Nat} (hrw : r < w) :
    ∀ d c, r - c = d → c < r → up G c r = true → up G c w = true →
      ∃ c', c' < r ∧ par n G c' = some r ∧ up G c' w = true := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro c hd hc hcr hcw
    obtain ⟨p, hp⟩ := par_exists hs hcr
    obtain ⟨hcp, hmin⟩ := par_spec hs hp
    have hpr : p ≤ r := hmin r hcr
    by_cases hpe : p = r
    · subst hpe; exact ⟨c, hc, hp, hcw⟩
    · have hpr' : p < r := by omega
      have hcp' : c < p := up_lt hcp
      exact ih (r - p) (by omega) p rfl hpr' (clique hcp hcr hpr') (clique hcp hcw (by omega))

/-- **Child absorption is exact**: `up(r) = {w > r : skeleton} ∪ ⋃ {up(c) ∖ {r} : par(c) = r}` (`order.rs:367`). -/
theorem absorb_iff {n : Nat} (hs : Supp n G) (r w : Nat) : up G r w = true ↔
    (r < w ∧ (G r w || G w r) = true) ∨ ∃ c, c < r ∧ par n G c = some r ∧ up G c w = true ∧ w ≠ r := by
  constructor
  · intro h
    have hrw := up_lt h
    rcases ((up_char r w).1 h).2 with h0 | ⟨c, hc, hcr, hcw⟩
    · exact Or.inl ⟨hrw, h0⟩
    · obtain ⟨c', h1, h2, h3⟩ := climb hs hrw (r - c) c rfl hc hcr hcw
      exact Or.inr ⟨c', h1, h2, h3, by omega⟩
  · rintro (⟨hrw, h0⟩ | ⟨c, hc, hp, hcw, hne⟩)
    · exact skeleton_in_up hrw h0
    · obtain ⟨hcr, hmin⟩ := par_spec hs hp
      have := hmin w hcw
      exact clique hcr hcw (by omega)

/-! ## The row subtree (column counts) -/

/-- `Desc par k r`: `k` is `r` or an elimination-tree descendant of `r`. -/
inductive Desc (pa : Nat → Option Nat) : Nat → Nat → Prop
  | refl (r : Nat) : Desc pa r r
  | step (k p r : Nat) : pa k = some p → Desc pa p r → Desc pa k r

theorem desc_le {n : Nat} (hs : Supp n G) {k r : Nat} (h : Desc (par n G) k r) : k ≤ r := by
  induction h with
  | refl => exact Nat.le_refl _
  | step k p r hp _ ih => exact Nat.le_trans (Nat.le_of_lt (par_lt hs hp)) ih

theorem desc_trans {pa : Nat → Option Nat} {k c r : Nat} (h : Desc pa k c) (hc : pa c = some r) : Desc pa k r := by
  induction h with
  | refl => exact Desc.step _ _ _ hc (Desc.refl r)
  | step k p c' hp _ ih => exact Desc.step _ _ _ hp (ih hc)

/-- **Row subtree**: `w ∈ up(r)` iff `r < w` and `w` is a skeleton neighbour of `r` or of one of its etree
    descendants. So `|up(r)|`, the count of column `r`, is the number of distinct higher skeleton neighbours of
    `r`'s subtree, which is what Gilbert–Ng–Peyton's column counts compute. -/
theorem row_subtree {n : Nat} (hs : Supp n G) (r w : Nat) : up G r w = true ↔
    r < w ∧ ∃ k, Desc (par n G) k r ∧ (G k w || G w k) = true := by
  constructor
  · intro h
    have hrw := up_lt h
    refine ⟨hrw, ?_⟩
    induction r using Nat.strongRecOn with
    | _ r ih =>
      rcases (absorb_iff hs r w).1 h with ⟨_, h0⟩ | ⟨c, hc, hp, hcw, _⟩
      · exact ⟨r, Desc.refl r, h0⟩
      · obtain ⟨k, hk, hkw⟩ := ih c hc hcw (up_lt hcw)
        exact ⟨k, desc_trans hk hp, hkw⟩
  · rintro ⟨hrw, k, hk, hkw⟩
    -- going up from `k`, `w` stays in every row on the path to `r`
    have key : ∀ k r, Desc (par n G) k r → r < w → up G k w = true → up G r w = true := by
      intro k r hd
      induction hd with
      | refl => exact fun _ h => h
      | step k p r hp hpr ih =>
        intro hrw hkw
        obtain ⟨hkp, hmin⟩ := par_spec hs hp
        have := desc_le hs hpr
        exact ih hrw (clique hkp hkw (by omega))
    exact key k r hk hrw (skeleton_in_up (by have := desc_le hs hk; omega) hkw)

end CchSG
