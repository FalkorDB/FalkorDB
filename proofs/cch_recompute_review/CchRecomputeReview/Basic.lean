/-!
# Basic definitions: weights, arcs, the metric, and what an arc reads

Ranks are `Nat`. An arc is a pair `(lo, hi)` with `hi ∈ up lo` (so `lo < hi`). The Rust indexes the four
metric arrays by the arc's slot (`order.rs:262-302` @main: `up_range(lo).start + i`) or, in the redesign, by its
id (03 §2 "Arc identity", I3′). Both are injective on arcs, so the model indexes the metric by the arc itself:
`Metric W := Nat → Nat → Entry W`, and only its values on arcs are ever compared.
-/

namespace CchRecompute

/-! ## Weights -/

/-- The four weight operations the customizations use. For the Rust: `add` is `f64 +`, `lt` is `f64 <`,
`beq` is `f64 ==` (`partial.rs:221-236` @08c test `!=`, its negation), and `top` is `f64::INFINITY`.
No law is assumed here; each theorem names the laws it needs. -/
structure Ops (W : Type) where
  add : W → W → W
  lt : W → W → Bool
  beq : W → W → Bool
  top : W

/-- **Agreement.** `R` is the relation under which the scoped result is proved to agree with the full one.
`R = Eq` (exact, bitwise) needs only `beq_sound`: `beq a b = true → a = b`. For IEEE `f64` restricted to the
values `validate` admits (`customize.rs:136-161` @main: not NaN, `≥ -0.0`, `+∞` allowed), `R := (· == ·)` satisfies
every field: `==` is an equivalence without NaN, `+` and `<` respect it (the only `==`-equal, bit-different pair
is `±0.0`, and `x + ±0.0`, `±0.0 < x` do not depend on the sign), and no `+∞ + -∞` arises. That is the premise
**P-f64**, an IEEE fact this core-only project cannot check (no float model); `Weights.lean` checks it on a
signed-zero model `SZ`. -/
structure Agree {W : Type} (o : Ops W) (R : W → W → Prop) : Prop where
  refl : ∀ a, R a a
  symm : ∀ {a b}, R a b → R b a
  trans : ∀ {a b c}, R a b → R b c → R a c
  beq_sound : ∀ {a b}, o.beq a b = true → R a b
  add_congr : ∀ {a a' b b'}, R a a' → R b b' → R (o.add a b) (o.add a' b')
  lt_congr : ∀ {a a' b b'}, R a a' → R b b' → o.lt a b = o.lt a' b'

/-- Exact agreement: `R = Eq` needs only that the change test `beq` is sound for equality. -/
theorem agree_eq {W : Type} (o : Ops W) (h : ∀ a b, o.beq a b = true → a = b) : Agree o Eq where
  refl _ := rfl
  symm h := h.symm
  trans h1 h2 := h1.trans h2
  beq_sound hb := h _ _ hb
  add_congr h1 h2 := by subst h1; subst h2; rfl
  lt_congr h1 h2 := by subst h1; subst h2; rfl

/-- `+∞` laws used by Lemma A: an apex through a `+∞` arc never wins. For `f64` on the admitted values:
`∞ + x = x + ∞ = ∞` (no `-∞`, no NaN) and `∞ < x` is false. -/
structure TopLaws {W : Type} (o : Ops W) : Prop where
  top_add : ∀ a, o.add o.top a = o.top
  add_top : ∀ a, o.add a o.top = o.top
  top_not_lt : ∀ a, o.lt o.top a = false

/-! ## The metric -/

/-- One arc's four values (`customize.rs:94-99` @main, `:243-248` @08c): `up` is `lo → hi`, `dn` is
`hi → lo`, and `mup`/`mdn` the apex each routes through (`None`: the arc's own edge). -/
structure Entry (W : Type) where
  up : W
  dn : W
  mup : Option Nat
  mdn : Option Nat
deriving DecidableEq

abbrev Metric (W : Type) := Nat → Nat → Entry W

/-- Write one arc. -/
def Metric.set {W : Type} (M : Metric W) (y z : Nat) (e : Entry W) : Metric W :=
  fun a b => if a = y ∧ b = z then e else M a b

theorem Metric.set_same {W : Type} (M : Metric W) (y z : Nat) (e : Entry W) : M.set y z e y z = e := by
  simp [Metric.set]

theorem Metric.set_ne {W : Type} (M : Metric W) {y z a b : Nat} (e : Entry W) (h : ¬(a = y ∧ b = z)) :
    M.set y z e a b = M a b := by
  simp only [Metric.set]; split
  · exact absurd ‹_› h
  · rfl

/-- Seeds per arc: `(lo → hi, hi → lo)`, `top` for "no edge" (`seed_of`, `partial.rs:47-49` @08c; the
`f64::INFINITY` fill of `seed`, `customize.rs:208-214` @main). The node-space lookup through
`node_of`/`rank_of` is a bijection (I1) and is not modelled. -/
abbrev Seeds (W : Type) := Nat → Nat → W × W

def seedE {W : Type} (S : Seeds W) (lo hi : Nat) : Entry W := ⟨(S lo hi).1, (S lo hi).2, none, none⟩

/-- The metric `seed` builds before relaxing: every arc at its seed, no middle. -/
def seedM {W : Type} (S : Seeds W) : Metric W := fun lo hi => seedE S lo hi

/-! ## Topology -/

/-- The chordal supergraph as upward rows over ranks (`order.rs:129-139` @main, `Topology`), with the
downward rows (`downward`, built by `transpose`). -/
structure Topo where
  n : Nat
  up : Nat → List Nat
  down : Nat → List Nat

/-- The representation invariants of `RankRows` (`order.rs:141-148`: "Each row's ranks, ascending"; `up(r)`
lists ranks above `r`, all below the rank count). They hold by construction of `chordal_supergraph`; that is
the premise **P-rows**, owned by the supergraph project. -/
structure Topo.Valid (T : Topo) : Prop where
  sorted : ∀ x, (T.up x).Pairwise (· < ·)
  above : ∀ x y, y ∈ T.up x → x < y
  bound : ∀ x y, y ∈ T.up x → y < T.n

/-- **P-chordal** (`customize.rs:261-263` @main: "a `z` missing from `up(y)`" is a bug). -/
def Topo.Chordal (T : Topo) : Prop :=
  ∀ x y z, y ∈ T.up x → z ∈ T.up x → y < z → z ∈ T.up y

/-- **P-transpose**: `downward` is the transpose of `upward`, rows ascending (`order.rs:119-121`, `transpose`). -/
def Topo.DownOK (T : Topo) : Prop :=
  ∀ r, T.down r = (List.range T.n).filter (fun x => decide (r ∈ T.up x))

/-- Is `x` an apex of the arc `(lo, hi)`: both `(x, lo)` and `(x, hi)` are arcs. -/
def Topo.apex (T : Topo) (lo hi x : Nat) : Bool := decide (lo ∈ T.up x ∧ hi ∈ T.up x)

/-- 03 §6b "What an arc reads": `x ∈ down(lo) ∩ down(hi)`, ascending. Every such `x` is below `lo`. -/
def Topo.commonDown (T : Topo) (lo hi : Nat) : List Nat := (List.range lo).filter (T.apex lo hi)

/-! ## What an arc reads: one apex step, and the pull specification -/

/-- One apex `x` of arc `(y, z)`, `y < z`, applied to the running entry `e`: `y → x → z` costs
`dn(x,y) + up(x,z)` and `z → x → y` costs `dn(x,z) + up(x,y)`; each replaces the running value only on a strict
`<`, and then records `x`. This is, term for term and operand for operand, `relax`'s body
(`customize.rs:296-308` @main: `y_to_x + x_to_z`, `z_to_x + x_to_y`) and `recustomize_seeded`'s merge body
(`partial.rs:205-216` @08c: `downward_weight[x_lo] + upward_weight[x_hi]`, `downward_weight[x_hi] +
upward_weight[x_lo]`). The two directions write different arrays, so they are independent. -/
def upd {W : Type} (o : Ops W) (M : Metric W) (x y z : Nat) (e : Entry W) : Entry W :=
  let cu := o.add (M x y).dn (M x z).up
  let cd := o.add (M x z).dn (M x y).up
  { up := if o.lt cu e.up then cu else e.up
    dn := if o.lt cd e.dn then cd else e.dn
    mup := if o.lt cu e.up then some x else e.mup
    mdn := if o.lt cd e.dn then some x else e.mdn }

/-- 03 §6b "What an arc reads", as a computation: from the seed, every apex in increasing `x`. -/
def recomp {W : Type} (o : Ops W) (T : Topo) (S : Seeds W) (M : Metric W) (lo hi : Nat) : Entry W :=
  (T.commonDown lo hi).foldl (fun e x => upd o M x lo hi e) (seedE S lo hi)

/-- **The full customization, as a specification:** every arc holds what it reads from the others. -/
def IsFull {W : Type} (o : Ops W) (T : Topo) (S : Seeds W) (F : Metric W) : Prop :=
  ∀ lo hi, hi ∈ T.up lo → F lo hi = recomp o T S F lo hi

/-! ## Agreement on entries -/

/-- Weights agree. -/
def WR {W : Type} (R : W → W → Prop) (e e' : Entry W) : Prop := R e.up e'.up ∧ R e.dn e'.dn

/-- Weights agree and the middles are equal. -/
def ER {W : Type} (R : W → W → Prop) (e e' : Entry W) : Prop := WR R e e' ∧ e.mup = e'.mup ∧ e.mdn = e'.mdn

theorem ER.toWR {W : Type} {R : W → W → Prop} {e e' : Entry W} (h : ER R e e') : WR R e e' := h.1

theorem ER_eq {W : Type} {e e' : Entry W} : ER Eq e e' ↔ e = e' := by
  constructor
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    cases e; cases e'; simp_all
  · rintro rfl; exact ⟨⟨rfl, rfl⟩, rfl, rfl⟩

theorem ER.refl {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) (e : Entry W) : ER R e e :=
  ⟨⟨hA.refl _, hA.refl _⟩, rfl, rfl⟩

theorem ER.symm {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e e' : Entry W}
    (h : ER R e e') : ER R e' e :=
  ⟨⟨hA.symm h.1.1, hA.symm h.1.2⟩, h.2.1.symm, h.2.2.symm⟩

theorem ER.trans {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e1 e2 e3 : Entry W}
    (h1 : ER R e1 e2) (h2 : ER R e2 e3) : ER R e1 e3 :=
  ⟨⟨hA.trans h1.1.1 h2.1.1, hA.trans h1.1.2 h2.1.2⟩, h1.2.1.trans h2.2.1, h1.2.2.trans h2.2.2⟩

theorem WR.trans {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e1 e2 e3 : Entry W}
    (h1 : WR R e1 e2) (h2 : WR R e2 e3) : WR R e1 e3 :=
  ⟨hA.trans h1.1 h2.1, hA.trans h1.2 h2.2⟩

theorem WR.symm {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e e' : Entry W}
    (h : WR R e e') : WR R e' e :=
  ⟨hA.symm h.1, hA.symm h.2⟩

/-- One apex step respects agreement: only the weights of `(x, y)` and `(x, z)` are read, and a strict `<`
between agreeing values decides the same way, so the middles come out equal. -/
theorem upd_congr {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {M M' : Metric W}
    {x y z : Nat} {e e' : Entry W} (hy : WR R (M x y) (M' x y)) (hz : WR R (M x z) (M' x z))
    (he : ER R e e') : ER R (upd o M x y z e) (upd o M' x y z e') := by
  have hcu := hA.add_congr hy.2 hz.1
  have hcd := hA.add_congr hz.2 hy.1
  have lu := hA.lt_congr hcu he.1.1
  have ld := hA.lt_congr hcd he.1.2
  simp only [upd]
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · simp only; rw [lu]; split
    · exact hcu
    · exact he.1.1
  · simp only; rw [ld]; split
    · exact hcd
    · exact he.1.2
  · simp only; rw [lu]; split
    · rfl
    · exact he.2.1
  · simp only; rw [ld]; split
    · rfl
    · exact he.2.2

/-- Folding agreeing steps over the same list keeps agreement. -/
theorem foldl_congr {W : Type} {R : W → W → Prop} (f g : Entry W → Nat → Entry W) :
    ∀ (l : List Nat) (e e' : Entry W),
      (∀ x ∈ l, ∀ a a', ER R a a' → ER R (f a x) (g a' x)) → ER R e e' →
      ER R (l.foldl f e) (l.foldl g e')
  | [], _, _, _, he => he
  | x :: t, e, e', h, he => by
    simp only [List.foldl_cons]
    exact foldl_congr f g t _ _ (fun y hy a a' ha => h y (List.mem_cons_of_mem _ hy) a a' ha)
      (h x List.mem_cons_self e e' he)

/-- **What an arc reads, as a congruence:** `recomp` depends only on the weights of the arcs `(x, lo)` and
`(x, hi)` for apexes `x`. -/
theorem recomp_congr {W : Type} {o : Ops W} {R : W → W → Prop} (hA : Agree o R) (T : Topo) (S : Seeds W)
    {M M' : Metric W} {lo hi : Nat}
    (h : ∀ x ∈ T.commonDown lo hi, WR R (M x lo) (M' x lo) ∧ WR R (M x hi) (M' x hi)) :
    ER R (recomp o T S M lo hi) (recomp o T S M' lo hi) :=
  foldl_congr _ _ _ _ _ (fun x hx _ _ ha => upd_congr hA (h x hx).1 (h x hx).2 ha) (ER.refl hA _)

/-- `upd` reads only the arcs `(x, y)` and `(x, z)`. -/
theorem upd_ext {W : Type} (o : Ops W) {M M' : Metric W} {x y z : Nat} (e : Entry W)
    (hy : M x y = M' x y) (hz : M x z = M' x z) : upd o M x y z e = upd o M' x y z e := by
  simp only [upd, hy, hz]

theorem foldl_ext {W : Type} (f g : Entry W → Nat → Entry W) :
    ∀ (l : List Nat) (e : Entry W), (∀ x ∈ l, ∀ a, f a x = g a x) → l.foldl f e = l.foldl g e
  | [], _, _ => rfl
  | x :: t, e, h => by
    simp only [List.foldl_cons]
    rw [h x List.mem_cons_self e]
    exact foldl_ext f g t _ (fun y hy a => h y (List.mem_cons_of_mem _ hy) a)

/-- Exact version of `recomp_congr`. -/
theorem recomp_ext {W : Type} {o : Ops W} (T : Topo) (S : Seeds W) {M M' : Metric W} {lo hi : Nat}
    (h : ∀ x ∈ T.commonDown lo hi, M x lo = M' x lo ∧ M x hi = M' x hi) :
    recomp o T S M lo hi = recomp o T S M' lo hi :=
  foldl_ext _ _ _ _ (fun x hx a => upd_ext o a (h x hx).1 (h x hx).2)

/-! ## Facts about apexes -/

theorem mem_commonDown {T : Topo} {lo hi x : Nat} :
    x ∈ T.commonDown lo hi ↔ x < lo ∧ lo ∈ T.up x ∧ hi ∈ T.up x := by
  simp [Topo.commonDown, Topo.apex, List.mem_filter, List.mem_range]

theorem commonDown_sorted (T : Topo) (lo hi : Nat) : (T.commonDown lo hi).Pairwise (· < ·) := by
  unfold Topo.commonDown
  apply List.Pairwise.filter
  rw [List.range_eq_range']
  exact List.pairwise_lt_range'

end CchRecompute
