-- Copied verbatim from proofs/value_order `FalkorValueOrder/Laws.lean` (2026-10-12, re-target to 4a4f2a72f); only the import paths differ.
import FalkorAggOps.VO.Model
/-!
# Generic comparator laws (used by `SortCmp.lean`)

A comparator `c : α → α → Ordering` is a total preorder when it is
antisymmetric (`c b a = (c a b).swap`) and transitive on `≠ gt`. This file
proves the facts the `sort_cmp` proofs need, for any `α`:

* `Lw` — global laws; `lt_le_lt'`, `le_lt_lt'`, `eq_eq_eq'` (from swap + trans).
* `thn` — `Ordering::then_with` / `then`; `thn_swap`, `thn_trans`: a key
  comparator followed by a second comparator is lawful when the second one is
  lawful on the pairs/triples the key leaves equal.
* `lexA` — slice order (`[T]::cmp`, and the `zip`/`find`/`unwrap_or_else(len)`
  shape of `sort_cmp`'s list arm): `lexA_swap`, `lexA_trans`, `lexA_refl`,
  `lexA_eq` with the laws required only of elements from the lists (`P`).
* `cmp3` (Model.lean) is a total order with `eq ↔ =`: `cmp3_lw`, `cmp3_eq_iff`.
-/
namespace ValueOrder.Laws
open ValueOrder

structure Lw {α : Type} (c : α → α → Ordering) : Prop where
  swap : ∀ a b, c b a = (c a b).swap
  trans : ∀ a b d, c a b ≠ .gt → c b d ≠ .gt → c a d ≠ .gt

section derived
variable {α : Type} (P : α → Prop) (c : α → α → Ordering)
  (hswap : ∀ a b, P a → P b → c b a = (c a b).swap)
  (htrans : ∀ a b d, P a → P b → P d → c a b ≠ .gt → c b d ≠ .gt → c a d ≠ .gt)
include hswap htrans

theorem lt_le_lt' (a b d : α) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b = .lt) (h2 : c b d ≠ .gt) : c a d = .lt := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd (by simp [h1]) h2
  cases had : c a d
  · rfl
  · have hda : c d a = .eq := by rw [hswap a d ha hd, had]; rfl
    have := htrans b d a hb hd ha h2 (by simp [hda])
    rw [hswap a b ha hb, h1] at this; exact absurd rfl this
  · exact absurd had hle

theorem le_lt_lt' (a b d : α) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b ≠ .gt) (h2 : c b d = .lt) : c a d = .lt := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd h1 (by simp [h2])
  cases had : c a d
  · rfl
  · have hda : c d a = .eq := by rw [hswap a d ha hd, had]; rfl
    have := htrans d a b hd ha hb (by simp [hda]) h1
    rw [hswap b d hb hd, h2] at this; exact absurd rfl this
  · exact absurd had hle

theorem eq_eq_eq' (a b d : α) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b = .eq) (h2 : c b d = .eq) : c a d = .eq := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd (by simp [h1]) (by simp [h2])
  have hba : c b a = .eq := by rw [hswap a b ha hb, h1]; rfl
  have hdb : c d b = .eq := by rw [hswap b d hb hd, h2]; rfl
  have hge : c d a ≠ .gt := htrans d b a hd hb ha (by simp [hdb]) (by simp [hba])
  rw [hswap a d ha hd] at hge
  cases had : c a d
  · rw [had] at hge; simp [Ordering.swap] at hge
  · rfl
  · exact absurd had hle
end derived

/-- `Ordering::then` / `then_with`. -/
def thn (a b : Ordering) : Ordering :=
  match a with
  | .eq => b
  | o => o

@[simp] theorem thn_eq (b : Ordering) : thn .eq b = b := rfl
@[simp] theorem thn_lt (b : Ordering) : thn .lt b = .lt := rfl
@[simp] theorem thn_gt (b : Ordering) : thn .gt b = .gt := rfl

theorem thn_eq_eq (a b : Ordering) : thn a b = .eq ↔ a = .eq ∧ b = .eq := by
  cases a <;> simp [thn]

theorem swap_swap' (o : Ordering) : o.swap.swap = o := by cases o <;> rfl

/-- `then` after a lawful key comparator is antisymmetric when the second
comparator is, on pairs the key leaves equal. -/
theorem thn_swap {α κ : Type} (ck : κ → κ → Ordering) (hk : Lw ck) (k : α → κ)
    (c2 : α → α → Ordering) (a b : α)
    (h2 : ck (k a) (k b) = .eq → c2 b a = (c2 a b).swap) :
    thn (ck (k b) (k a)) (c2 b a) = (thn (ck (k a) (k b)) (c2 a b)).swap := by
  rw [hk.swap (k a) (k b)]
  cases h : ck (k a) (k b) <;> simp [thn, Ordering.swap, h2, h]

theorem thn_trans {α κ : Type} (ck : κ → κ → Ordering) (hk : Lw ck) (k : α → κ)
    (c2 : α → α → Ordering) (a b d : α)
    (h2 : ck (k a) (k b) = .eq → ck (k b) (k d) = .eq →
      c2 a b ≠ .gt → c2 b d ≠ .gt → c2 a d ≠ .gt) :
    thn (ck (k a) (k b)) (c2 a b) ≠ .gt → thn (ck (k b) (k d)) (c2 b d) ≠ .gt →
    thn (ck (k a) (k d)) (c2 a d) ≠ .gt := by
  intro h1 h3
  have sw : ∀ x y : κ, True → True → ck y x = (ck x y).swap := fun x y _ _ => hk.swap x y
  have tr : ∀ x y z : κ, True → True → True → ck x y ≠ .gt → ck y z ≠ .gt → ck x z ≠ .gt :=
    fun x y z _ _ _ => hk.trans x y z
  cases e1 : ck (k a) (k b) <;> cases e2 : ck (k b) (k d) <;> simp only [e1, e2, thn_lt, thn_gt, thn_eq] at h1 h3
  all_goals first
    | exact absurd rfl h1
    | exact absurd rfl h3
    | (rw [lt_le_lt' (fun _ => True) ck sw tr _ _ _ trivial trivial trivial e1 (by simp [e2])]; simp)
    | (rw [le_lt_lt' (fun _ => True) ck sw tr _ _ _ trivial trivial trivial (by simp [e1]) e2]; simp)
    | (rw [eq_eq_eq' (fun _ => True) ck sw tr _ _ _ trivial trivial trivial e1 e2]; exact h2 e1 e2 h1 h3)

/-! ## Slice order -/

/-- Lexicographic, then shorter first: `[T]::cmp`, and `sort_cmp`'s list arm. -/
def lexA {α : Type} (c : α → α → Ordering) : List α → List α → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | x :: xs, y :: ys => thn (c x y) (lexA c xs ys)

/-- `zip(..).map(cmp).find(ne).unwrap_or(Equal)`: the first unequal pair, ignoring
the longer tail. -/
def zipNe {α : Type} (c : α → α → Ordering) : List α → List α → Ordering
  | x :: xs, y :: ys => thn (c x y) (zipNe c xs ys)
  | _, _ => .eq

theorem zipNe_eq_lexA {α : Type} (c : α → α → Ordering) :
    ∀ xs ys : List α, xs.length = ys.length → zipNe c xs ys = lexA c xs ys
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    simp only [zipNe, lexA]
    rw [zipNe_eq_lexA c xs ys (by simpa using h)]

section lexlaws
variable {α : Type} (P : α → Prop) (c : α → α → Ordering)

theorem lexA_refl (hr : ∀ a, P a → c a a = .eq) :
    ∀ xs : List α, (∀ x ∈ xs, P x) → lexA c xs xs = .eq
  | [], _ => rfl
  | x :: xs, h => by
    simp only [lexA, hr x (h x (by simp)), thn_eq]
    exact lexA_refl hr xs (fun y hy => h y (by simp [hy]))

theorem lexA_swap (hswap : ∀ a b, P a → P b → c b a = (c a b).swap) :
    ∀ xs ys : List α, (∀ x ∈ xs, P x) → (∀ y ∈ ys, P y) →
    lexA c ys xs = (lexA c xs ys).swap
  | [], [] , _, _ => rfl
  | [], _ :: _, _, _ => rfl
  | _ :: _, [], _, _ => rfl
  | x :: xs, y :: ys, hx, hy => by
      have e := hswap x y (hx x (by simp)) (hy y (by simp))
      have ih := lexA_swap hswap xs ys (fun z hz => hx z (by simp [hz])) (fun z hz => hy z (by simp [hz]))
      simp only [lexA, e]
      cases c x y <;> simp [thn, Ordering.swap, ih]

theorem lexA_trans
    (hswap : ∀ a b, P a → P b → c b a = (c a b).swap)
    (htrans : ∀ a b d, P a → P b → P d → c a b ≠ .gt → c b d ≠ .gt → c a d ≠ .gt) :
    ∀ xs ys zs : List α,
    (∀ x ∈ xs, P x) → (∀ y ∈ ys, P y) → (∀ z ∈ zs, P z) →
    lexA c xs ys ≠ .gt → lexA c ys zs ≠ .gt → lexA c xs zs ≠ .gt
  | [], [], _, _, _, _, _, h2 => h2
  | [], _ :: _, [], _, _, _, _, h2 => by simp [lexA] at h2
  | [], _ :: _, _ :: _, _, _, _, _, _ => by simp [lexA]
  | _ :: _, [], _, _, _, _, h1, _ => by simp [lexA] at h1
  | _ :: _, _ :: _, [], _, _, _, _, h2 => by simp [lexA] at h2
  | x :: xs, y :: ys, z :: zs, hx, hy, hz, h1, h2 => by
      have px := hx x (by simp); have py := hy y (by simp); have pz := hz z (by simp)
      have ih := lexA_trans hswap htrans xs ys zs (fun w hw => hx w (by simp [hw]))
        (fun w hw => hy w (by simp [hw])) (fun w hw => hz w (by simp [hw]))
      simp only [lexA] at h1 h2 ⊢
      cases e1 : c x y <;> cases e2 : c y z <;> simp only [e1, e2, thn_lt, thn_gt, thn_eq] at h1 h2
      all_goals first
        | exact absurd rfl h1
        | exact absurd rfl h2
        | (rw [lt_le_lt' P c hswap htrans x y z px py pz e1 (by simp [e2])]; simp)
        | (rw [le_lt_lt' P c hswap htrans x y z px py pz (by simp [e1]) e2]; simp)
        | (rw [eq_eq_eq' P c hswap htrans x y z px py pz e1 e2]; simpa using ih h1 h2)

/-- `lexA` is `eq` exactly when the lists have one length and are pairwise `eq`. -/
theorem lexA_eq : ∀ xs ys : List α, lexA c xs ys = .eq →
    xs.length = ys.length ∧ ∀ i (h1 : i < xs.length) (h2 : i < ys.length), c xs[i] ys[i] = .eq
  | [], [], _ => ⟨rfl, fun _ h => by simp at h⟩
  | [], _ :: _, h => by simp [lexA] at h
  | _ :: _, [], h => by simp [lexA] at h
  | x :: xs, y :: ys, h => by
    simp only [lexA, thn_eq_eq] at h
    obtain ⟨hl, hi⟩ := lexA_eq xs ys h.2
    refine ⟨by simp [hl], fun i h1 h2 => ?_⟩
    cases i with
    | zero => exact h.1
    | succ i => exact hi i (by simpa using h1) (by simpa using h2)
end lexlaws

theorem lexA_lw {α : Type} (c : α → α → Ordering) (hc : Lw c) : Lw (lexA c) :=
  ⟨fun xs ys => lexA_swap (fun _ => True) c (fun a b _ _ => hc.swap a b) xs ys
      (fun _ _ => trivial) (fun _ _ => trivial),
   fun xs ys zs => lexA_trans (fun _ => True) c (fun a b _ _ => hc.swap a b)
      (fun a b d _ _ _ => hc.trans a b d) xs ys zs
      (fun _ _ => trivial) (fun _ _ => trivial) (fun _ _ => trivial)⟩

/-! ## Total orders on keys -/

theorem cmpI_lw : Lw cmpI := ⟨fun a b => cmpI_swap a b, fun a b d => cmpI_le_trans a b d⟩

theorem cmpN_lw : Lw cmpN :=
  ⟨fun a b => cmpI_swap (a : Int) b, fun a b d => cmpI_le_trans (a : Int) b d⟩

theorem cmpI_eq_iff (a b : Int) : cmpI a b = .eq ↔ a = b := by
  unfold cmpI
  by_cases h1 : a < b
  · simp [h1]; omega
  · by_cases h2 : a = b <;> simp [h1, h2]

theorem cmpN_eq_iff (a b : Nat) : cmpN a b = .eq ↔ a = b := by
  unfold cmpN; rw [cmpI_eq_iff]; omega

theorem cmp3_thn (a b : Nat × Int × Nat) :
    cmp3 a b = thn (cmpI (a.1 : Int) b.1) (thn (cmpI a.2.1 b.2.1) (cmpI a.2.2 b.2.2)) := by
  unfold cmp3 thn
  cases cmpI (a.1 : Int) b.1 <;> cases cmpI a.2.1 b.2.1 <;> rfl

theorem cmp3_lw : Lw cmp3 := by
  have inner : Lw (fun (a b : Nat × Int × Nat) => thn (cmpI a.2.1 b.2.1) (cmpI (a.2.2 : Int) b.2.2)) :=
    ⟨fun a b => thn_swap cmpI cmpI_lw (fun x : Nat × Int × Nat => x.2.1)
        (fun x y => cmpI (x.2.2 : Int) y.2.2) a b (fun _ => cmpI_swap _ _),
     fun a b d => thn_trans cmpI cmpI_lw (fun x : Nat × Int × Nat => x.2.1)
        (fun x y => cmpI (x.2.2 : Int) y.2.2) a b d (fun _ _ => cmpI_le_trans _ _ _)⟩
  constructor
  · intro a b
    rw [cmp3_thn, cmp3_thn]
    exact thn_swap cmpI cmpI_lw (fun x : Nat × Int × Nat => (x.1 : Int))
      (fun x y => thn (cmpI x.2.1 y.2.1) (cmpI (x.2.2 : Int) y.2.2)) a b (fun _ => inner.swap a b)
  · intro a b d
    rw [cmp3_thn, cmp3_thn, cmp3_thn]
    exact thn_trans cmpI cmpI_lw (fun x : Nat × Int × Nat => (x.1 : Int))
      (fun x y => thn (cmpI x.2.1 y.2.1) (cmpI (x.2.2 : Int) y.2.2)) a b d
      (fun _ _ => inner.trans a b d)

theorem cmp3_eq_iff (a b : Nat × Int × Nat) : cmp3 a b = .eq ↔ a = b := by
  obtain ⟨a1, a2, a3⟩ := a
  obtain ⟨b1, b2, b3⟩ := b
  rw [cmp3_thn, thn_eq_eq, thn_eq_eq, cmpI_eq_iff, cmpI_eq_iff, cmpI_eq_iff]
  simp only [Prod.mk.injEq]
  omega

theorem cmp3_refl (a : Nat × Int × Nat) : cmp3 a a = .eq := (cmp3_eq_iff a a).2 rfl

#print axioms cmp3_lw
#print axioms lexA_trans

end ValueOrder.Laws
