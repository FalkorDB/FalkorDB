import FalkorValueOrder.Laws
/-!
# `Value::sort_cmp` (#2912, origin/main b033aecc6) and `Hash for Value`

See the report in `FalkorValueOrder.lean` ("Wave 9").

| here | there (origin/main b033aecc6, graph/src/runtime/value.rs) |
| --- | --- |
| `scF64`      | `sort_cmp_f64` (:1888) |
| `scF64` (as `c32`) | `sort_cmp_f32` (:1900) — the same body on main |
| `scIntFloat` | `sort_cmp_int_float` (:1914) |
| `sc`, `scList`, `scVals` | `Value::sort_cmp` (:1526), `sort_cmp_map` (:1563) |
| `hwords`     | `Hash for Value` (:822), in Model.lean |
-/
namespace ValueOrder.SC
open ValueOrder ValueOrder.Laws

/-! ## Floats -/

def isNaN (f : F) : Bool := f.nkey.isNone

/-- `f.is_sign_negative()`. -/
def isSignNeg : F → Bool
  | .nan neg _ => neg
  | .negZero => true
  | .num n => decide (n < 0)
  | .frac lo _ => decide (lo < 0)
  | .negInf => true
  | .posInf => false

/-- The key `sort_cmp_f64` orders by: the numeric position, NaN above all. -/
def fkey (f : F) : Nat × Int × Nat := f.nkey.getD (3, 0, 0)

/-- `sort_cmp_f64` (value.rs:1888): `partial_cmp`, else `a.is_nan().cmp(&b.is_nan())`. -/
def scF64 (a b : F) : Ordering :=
  match fpcmp a b with
  | some o => o
  | none => cmpB (isNaN a) (isNaN b)

/-- `TWO_POW_63`. -/
def P63 : Int := 9223372036854775808

/-- `f >= g` on floats. -/
def fge (f g : F) : Bool :=
  match fpcmp f g with
  | some .gt | some .eq => true
  | _ => false

/-- `f < g` on floats. -/
def fltF (f g : F) : Bool := fpcmp f g == some .lt

/-- `(f.trunc() as i64, sign of f - f.trunc())` for a non-NaN `f` in `[-2^63, 2^63)`. -/
def truncParts : F → Int × Ordering
  | .num n => (n, .eq)
  | .frac lo _ => if 0 ≤ lo then (lo, .gt) else (lo + 1, .lt)
  | _ => (0, .eq)          -- ±0.0 (NaN and ±inf never get here)

/-- `frac > 0 ⇒ Less`, `frac < 0 ⇒ Greater`, else `Equal`. -/
def fracCmp : Ordering → Ordering
  | .gt => .lt
  | .lt => .gt
  | .eq => .eq

/-- `sort_cmp_int_float` (value.rs:1914). -/
def scIntFloat (i : Int) (f : F) : Ordering :=
  if isNaN f || fge f (.num P63) then .lt
  else if fltF f (.num (-P63)) then .gt
  else thn (cmpI i (truncParts f).1) (fracCmp (truncParts f).2)

theorem cmpI_of_lt {a b : Int} (h : a < b) : cmpI a b = .lt := by simp [cmpI, h]
theorem cmpI_of_gt {a b : Int} (h : b < a) : cmpI a b = .gt := by
  unfold cmpI; rw [ite_eq_right_iff.mpr (fun h => absurd h (by omega)), ite_eq_right_iff.mpr (fun h => absurd h (by omega))]
theorem cmpI_of_eq {a b : Int} (h : a = b) : cmpI a b = .eq := by simp [cmpI, h]

theorem cmpI_lt_iff (a b : Int) : cmpI a b = .lt ↔ a < b := by
  unfold cmpI; by_cases h1 : a < b <;> simp [h1]; by_cases h2 : a = b <;> simp [h2]
theorem cmpI_gt_iff (a b : Int) : cmpI a b = .gt ↔ b < a := by
  unfold cmpI; by_cases h1 : a < b <;> simp [h1]
  · omega
  · by_cases h2 : a = b <;> simp [h2] <;> omega

theorem thn_lt_iff (a b : Ordering) : thn a b = .lt ↔ a = .lt ∨ (a = .eq ∧ b = .lt) := by
  cases a <;> simp [thn]

theorem cmp3_one (a b : Int) (r s : Nat) : cmp3 (1, a, r) (1, b, s) = thn (cmpI a b) (cmpN r s) := by
  unfold cmp3 thn; simp only [cmpI_of_eq (show ((1 : Nat) : Int) = (1 : Nat) from rfl)]
  cases cmpI a b <;> rfl

theorem nkey_fst {f : F} {y : Nat × Int × Nat} (h : f.nkey = some y) : y.1 ≤ 2 := by
  cases f <;> simp [F.nkey] at h <;> subst h <;> simp

theorem cmp3_top_gt (y : Nat × Int × Nat) (h : y.1 ≤ 2) : cmp3 (3, 0, 0) y = .gt := by
  rw [Laws.cmp3_thn, cmpI_of_gt (by simp; omega)]; rfl

theorem cmp3_top_lt (y : Nat × Int × Nat) (h : y.1 ≤ 2) : cmp3 y (3, 0, 0) = .lt := by
  rw [Laws.cmp3_thn, cmpI_of_lt (by simp; omega)]; rfl

theorem scF64_key (a b : F) : scF64 a b = cmp3 (fkey a) (fkey b) := by
  unfold scF64 fpcmp fkey isNaN
  cases ha : a.nkey <;> cases hb : b.nkey
  · simp [cmpB, cmpN, cmp3, cmpI]
  · simp only [Option.isNone_none, Option.isNone_some, Option.getD_none, Option.getD_some]
    rw [cmp3_top_gt _ (nkey_fst hb)]; simp [cmpB, cmpN, cmpI]
  · simp only [Option.isNone_none, Option.isNone_some, Option.getD_none, Option.getD_some]
    rw [cmp3_top_lt _ (nkey_fst ha)]; simp [cmpB, cmpN, cmpI]
  · rfl

/-- `fge`/`fltF` against an integral float, through the key. -/
theorem fpcmp_num_right (f : F) (b : Int) (h : isNaN f = false) :
    fpcmp f (.num b) = some (cmp3 (fkey f) (1, b, 0)) := by
  cases f <;> simp_all [isNaN, fpcmp, F.nkey, fkey]

theorem fge_iff (f : F) (b : Int) (h : isNaN f = false) :
    fge f (.num b) = true ↔ cmp3 (fkey f) (1, b, 0) ≠ .lt := by
  unfold fge; rw [fpcmp_num_right f b h]
  cases cmp3 (fkey f) (1, b, 0) <;> simp

theorem fltF_iff (f : F) (b : Int) (h : isNaN f = false) :
    fltF f (.num b) = true ↔ cmp3 (fkey f) (1, b, 0) = .lt := by
  unfold fltF; rw [fpcmp_num_right f b h]
  cases cmp3 (fkey f) (1, b, 0) <;> simp

/-- **No rounding**: the Int/Float arm is the numeric-key order, for every `i64`. -/
theorem scIntFloat_key (i : Int) (lo : I64MIN ≤ i) (hi : i ≤ I64MAX) (f : F) :
    scIntFloat i f = cmp3 (1, i, 0) (fkey f) := by
  unfold I64MIN at lo; unfold I64MAX at hi
  cases f with
  | nan neg p => simp [scIntFloat, isNaN, F.nkey, fkey, cmp3, cmpI]
  | negInf => simp [scIntFloat, isNaN, F.nkey, fkey, cmp3, cmpI, fge, fltF, fpcmp, P63]
  | posInf => simp [scIntFloat, isNaN, F.nkey, fkey, cmp3, cmpI, fge, fpcmp, P63]
  | negZero =>
    have h1 : fge .negZero (.num P63) = false := by
      simp [fge, fpcmp, F.nkey, cmp3_one, cmpI_of_lt (show (0 : Int) < P63 by decide)]
    have h2 : fltF .negZero (.num (-P63)) = false := by
      simp [fltF, fpcmp, F.nkey, cmp3_one, cmpI_of_gt (show -P63 < (0 : Int) by decide)]
    simp only [scIntFloat, h1, h2, isNaN, F.nkey, fkey, truncParts, fracCmp, Option.isNone_some,
      Bool.false_or, Option.getD_some, cmp3_one, Bool.false_eq_true, ite_false]
    cases cmpI i 0 <;> rfl
  | num n =>
    simp only [fkey, F.nkey, Option.getD_some, cmp3_one]
    by_cases hp : P63 ≤ n
    · have : fge (.num n) (.num P63) = true := by
        rw [fge_iff _ _ rfl]; simp only [fkey, F.nkey, Option.getD_some, cmp3_one, ne_eq,
          thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN]
        simp only [P63] at *; omega
      simp only [scIntFloat, this, Bool.or_true, ite_true]
      rw [cmpI_of_lt (a := i) (by simp only [P63] at *; omega)]; simp [thn]
    · have h1 : fge (.num n) (.num P63) = false := by
        cases h : fge (.num n) (.num P63)
        · rfl
        · rw [fge_iff _ _ rfl] at h; simp only [fkey, F.nkey, Option.getD_some, cmp3_one] at h
          simp only [ne_eq, thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN] at h
          simp only [P63] at *; omega
      by_cases hm : n < -P63
      · have : fltF (.num n) (.num (-P63)) = true := by
          rw [fltF_iff _ _ rfl]; simp only [fkey, F.nkey, Option.getD_some, cmp3_one]
          rw [cmpI_of_lt hm]; rfl
        simp only [scIntFloat, h1, this, isNaN, F.nkey, Option.isNone_some, Bool.false_or,
          Bool.false_eq_true, ite_false, ite_true]
        rw [cmpI_of_gt (a := i) (by simp only [P63] at *; omega)]; simp [thn]
      · have h2 : fltF (.num n) (.num (-P63)) = false := by
          cases h : fltF (.num n) (.num (-P63))
          · rfl
          · rw [fltF_iff _ _ rfl] at h; simp only [fkey, F.nkey, Option.getD_some, cmp3_one] at h
            simp only [thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN] at h
            omega
        simp only [scIntFloat, h1, h2, isNaN, F.nkey, Option.isNone_some, Bool.false_or,
          Bool.false_eq_true, ite_false, truncParts, fracCmp]
        cases cmpI i n <;> rfl
  | frac l r =>
    simp only [fkey, F.nkey, Option.getD_some, cmp3_one]
    have hr : cmpN 0 (r + 1) = .lt := cmpI_of_lt (by omega)
    by_cases hp : P63 ≤ l
    · have : fge (.frac l r) (.num P63) = true := by
        rw [fge_iff _ _ rfl]; simp only [fkey, F.nkey, Option.getD_some, cmp3_one, ne_eq,
          thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN]
        simp only [P63] at *; omega
      simp only [scIntFloat, this, Bool.or_true, ite_true]
      rw [cmpI_of_lt (a := i) (by simp only [P63] at *; omega)]; simp [thn]
    · have h1 : fge (.frac l r) (.num P63) = false := by
        cases h : fge (.frac l r) (.num P63)
        · rfl
        · rw [fge_iff _ _ rfl] at h; simp only [fkey, F.nkey, Option.getD_some, cmp3_one] at h
          simp only [ne_eq, thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN] at h
          simp only [P63] at *; omega
      by_cases hm : l < -P63
      · have : fltF (.frac l r) (.num (-P63)) = true := by
          rw [fltF_iff _ _ rfl]; simp only [fkey, F.nkey, Option.getD_some, cmp3_one]
          rw [cmpI_of_lt hm]; rfl
        simp only [scIntFloat, h1, this, isNaN, F.nkey, Option.isNone_some, Bool.false_or,
          Bool.false_eq_true, ite_false, ite_true]
        rw [cmpI_of_gt (a := i) (by simp only [P63] at *; omega)]; simp [thn]
      · have h2 : fltF (.frac l r) (.num (-P63)) = false := by
          cases h : fltF (.frac l r) (.num (-P63))
          · rfl
          · rw [fltF_iff _ _ rfl] at h; simp only [fkey, F.nkey, Option.getD_some, cmp3_one] at h
            simp only [thn_lt_iff, cmpI_lt_iff, cmpI_eq_iff, cmpN] at h
            omega
        simp only [scIntFloat, h1, h2, isNaN, F.nkey, Option.isNone_some, Bool.false_or,
          Bool.false_eq_true, ite_false, truncParts]
        rw [hr]
        by_cases h0 : 0 ≤ l
        · simp only [h0, ite_true, fracCmp]; try (cases cmpI i l <;> rfl)
        · simp only [h0, ite_false, fracCmp]
          by_cases e1 : i < l
          · rw [cmpI_of_lt (a := i) (b := l) e1, cmpI_of_lt (a := i) (b := l + 1) (by omega)]; rfl
          · by_cases e2 : i = l
            · rw [cmpI_of_eq (a := i) (b := l) e2, cmpI_of_lt (a := i) (b := l + 1) (by omega)]; rfl
            · by_cases e3 : i = l + 1
              · rw [cmpI_of_gt (a := i) (b := l) (by omega), cmpI_of_eq (a := i) (b := l + 1) e3]; rfl
              · rw [cmpI_of_gt (a := i) (b := l) (by omega),
                  cmpI_of_gt (a := i) (b := l + 1) (by omega)]; rfl

/-! ## `Value::sort_cmp` -/

mutual
/-- `Value::sort_cmp` (value.rs:1526), arm by arm; `c32` is `sort_cmp_f32`. -/
def sc (c32 : F → F → Ordering) : V → V → Ordering
  | .bool a, .bool b => cmpB a b
  | .int a, .int b => cmpI a b
  | .flt a, .flt b => scF64 a b
  | .int i, .flt f => scIntFloat i f
  | .flt f, .int i => (scIntFloat i f).swap
  | .str a, .str b => cmpN a b
  | .list a, .list b => scList c32 a b
  | .map ka va, .map kb vb =>
      thn (cmpN ka.length kb.length) (thn (Laws.lexA cmpN ka kb) (scVals c32 va vb))
  | .vecf32 a, .vecf32 b => thn (cmpN a.length b.length) (Laws.zipNe c32 a b)
  | .node a, .node b => cmpN a b
  | .point alat alon, .point blat blon => thn (c32 alon blon) (c32 alat blat)
  | a, b => cmpN (order a) (order b)

/-- The list arm: `zip`, `sort_cmp`, first `ne`, else `len().cmp`. -/
def scList (c32 : F → F → Ordering) : List V → List V → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | x :: xs, y :: ys => thn (sc c32 x y) (scList c32 xs ys)

/-- `sort_cmp_map`'s value loop (keys already equal): first `ne`, else `Equal`. -/
def scVals (c32 : F → F → Ordering) : List V → List V → Ordering
  | x :: xs, y :: ys => thn (sc c32 x y) (scVals c32 xs ys)
  | _, _ => .eq
end

theorem scList_eq (c32 : F → F → Ordering) :
    ∀ xs ys, scList c32 xs ys = Laws.lexA (sc c32) xs ys
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | x :: xs, y :: ys => by simp only [scList, Laws.lexA, scList_eq c32 xs ys]

theorem scVals_eq (c32 : F → F → Ordering) :
    ∀ xs ys, scVals c32 xs ys = Laws.zipNe (sc c32) xs ys
  | x :: xs, y :: ys => by simp only [scVals, Laws.zipNe, scVals_eq c32 xs ys]
  | [], _ => by cases ‹List V› <;> rfl
  | _ :: _, [] => rfl

/-- main's `sort_cmp`: `sort_cmp_f32` has the same body as `sort_cmp_f64`. -/
def scMain : V → V → Ordering := sc scF64

/-! ## Well-formedness and size -/

mutual
/-- Ints are `i64`; a Float `num n` is a representable `f64` (`n as f64` exact);
a map has one value per key. -/
def wf : V → Bool
  | .int i => decide (I64MIN ≤ i ∧ i ≤ I64MAX)
  | .flt (.num n) => decide (roundF n = n)
  | .list xs => wfL xs
  | .map ks vs => decide (ks.length = vs.length) && wfL vs
  | _ => true

def wfL : List V → Bool
  | [] => true
  | x :: xs => wf x && wfL xs
end

theorem wfL_mem : ∀ {xs : List V}, wfL xs = true → ∀ x ∈ xs, wf x = true
  | [], _, _, h => by simp at h
  | y :: ys, h, x, hx => by
    simp only [wfL, Bool.and_eq_true] at h
    rcases List.mem_cons.mp hx with rfl | hx
    · exact h.1
    · exact wfL_mem h.2 x hx

mutual
def vsize : V → Nat
  | .list xs => 1 + vsizeL xs
  | .map _ vs => 1 + vsizeL vs
  | _ => 1

def vsizeL : List V → Nat
  | [] => 0
  | x :: xs => vsize x + vsizeL xs
end

theorem vsize_pos (v : V) : 1 ≤ vsize v := by
  cases v <;> simp [vsize] <;> omega

theorem vsizeL_mem : ∀ {xs : List V} {x : V}, x ∈ xs → vsize x ≤ vsizeL xs
  | [], _, h => by simp at h
  | y :: ys, x, h => by
    simp only [vsizeL]
    rcases List.mem_cons.mp h with rfl | h
    · omega
    · have := vsizeL_mem h; omega

/-! ## Ranks and scalar keys -/

/-- The type order, with Int and Float sharing Int's rank (`no_tag_between_int_and_float`). -/
def rank : V → Nat
  | .flt _ => orderTag .int
  | v => orderTag v.tag

def isComp : V → Bool
  | .list _ | .map .. | .vecf32 _ | .point .. => true
  | _ => false

/-- The key a scalar sorts by inside its rank. -/
def sk : V → Nat × Int × Nat
  | .bool b => (0, if b then 1 else 0, 0)
  | .int i => (1, i, 0)
  | .flt f => fkey f
  | .str s => (0, s, 0)
  | .node n => (0, n, 0)
  | _ => (0, 0, 0)

theorem sc_rank_ne (c32 : F → F → Ordering) (a b : V) (h : rank a ≠ rank b) :
    sc c32 a b = cmpN (rank a) (rank b) := by
  cases a <;> cases b <;> simp_all [sc, rank, order, V.tag, orderTag, cmpN, cmpI]

theorem rank_isComp {a b : V} (h : rank a = rank b) : isComp a = isComp b := by
  cases a <;> cases b <;> simp_all [rank, isComp, V.tag, orderTag]

theorem cmp3_mid (x y : Int) (n : Nat) : cmp3 (n, x, 0) (n, y, 0) = cmpI x y := by
  unfold cmp3; rw [cmpI_of_eq rfl]
  cases h : cmpI x y <;> simp [cmpI]

theorem sc_scalar (c32 : F → F → Ordering) (a b : V) (ha : isComp a = false)
    (hb : isComp b = false) (wa : wf a = true) (wb : wf b = true) :
    sc c32 a b = thn (cmpN (rank a) (rank b)) (cmp3 (sk a) (sk b)) := by
  by_cases hr : rank a = rank b
  · rw [(Laws.cmpN_eq_iff _ _).2 hr, Laws.thn_eq]
    cases a <;> cases b <;> simp [isComp, rank, V.tag, orderTag, sc, sk] at ha hb hr ⊢
    all_goals first
      | exact (Laws.cmp3_refl _).symm
      | exact (cmp3_mid _ _ _).symm
      | exact scF64_key _ _
      | (rename_i x y; cases x <;> cases y <;> decide)
      | (simp only [wf, decide_eq_true_eq] at wa; exact scIntFloat_key _ wa.1 wa.2 _)
      | (simp only [wf, decide_eq_true_eq] at wb
         rw [scIntFloat_key _ wb.1 wb.2 _, Laws.cmp3_lw.swap (1, _, 0)])
  · rw [sc_rank_ne c32 a b hr]
    have : cmpN (rank a) (rank b) ≠ .eq := fun h => hr ((Laws.cmpN_eq_iff _ _).1 h)
    cases h : cmpN (rank a) (rank b) <;> simp_all

theorem rank_list {xs : List V} {d : V} (h : rank (.list xs) = rank d) : ∃ zs, d = .list zs := by
  cases d <;> simp_all [rank, V.tag, orderTag]
theorem rank_map {ks : List Nat} {vs : List V} {d : V} (h : rank (.map ks vs) = rank d) :
    ∃ ks' vs', d = .map ks' vs' := by
  cases d <;> simp_all [rank, V.tag, orderTag]
theorem rank_vec {xs : List F} {d : V} (h : rank (.vecf32 xs) = rank d) : ∃ zs, d = .vecf32 zs := by
  cases d <;> simp_all [rank, V.tag, orderTag]
theorem rank_point {x y : F} {d : V} (h : rank (.point x y) = rank d) : ∃ x' y', d = .point x' y' := by
  cases d <;> simp_all [rank, V.tag, orderTag]

/-! ## The laws, by induction on size -/

def Pn (n : Nat) (v : V) : Prop := vsize v ≤ n ∧ wf v = true

structure IHn (c32 : F → F → Ordering) (n : Nat) : Prop where
  refl : ∀ a, Pn n a → sc c32 a a = .eq
  swap : ∀ a b, Pn n a → Pn n b → sc c32 b a = (sc c32 a b).swap
  trans : ∀ a b d, Pn n a → Pn n b → Pn n d →
    sc c32 a b ≠ .gt → sc c32 b d ≠ .gt → sc c32 a d ≠ .gt

theorem Pn_list {n : Nat} {xs : List V} (h : Pn (n + 1) (.list xs)) : ∀ x ∈ xs, Pn n x := by
  intro x hx
  obtain ⟨h1, h2⟩ := h
  simp only [vsize] at h1; simp only [wf] at h2
  exact ⟨by have := vsizeL_mem hx; omega, wfL_mem h2 x hx⟩

theorem Pn_map {n : Nat} {ks : List Nat} {vs : List V} (h : Pn (n + 1) (.map ks vs)) :
    ks.length = vs.length ∧ ∀ x ∈ vs, Pn n x := by
  obtain ⟨h1, h2⟩ := h
  simp only [vsize] at h1; simp only [wf, Bool.and_eq_true, decide_eq_true_eq] at h2
  exact ⟨h2.1, fun x hx => ⟨by have := vsizeL_mem hx; omega, wfL_mem h2.2 x hx⟩⟩

theorem lenEq_of_lexA_eq {xs ys : List Nat} (h : Laws.lexA cmpN xs ys = .eq) : xs.length = ys.length :=
  (Laws.lexA_eq cmpN xs ys h).1

section kinds
variable (c32 : F → F → Ordering) (hc : Laws.Lw c32) (hr32 : ∀ x, c32 x x = .eq)
include hc

theorem vec_swap (a b : List F) :
    sc c32 (.vecf32 b) (.vecf32 a) = (sc c32 (.vecf32 a) (.vecf32 b)).swap := by
  simp only [sc]
  exact Laws.thn_swap cmpN Laws.cmpN_lw (fun l : List F => l.length) (Laws.zipNe c32) a b
    (fun h => by
      have hl := (Laws.cmpN_eq_iff _ _).1 h
      rw [Laws.zipNe_eq_lexA _ _ _ hl, Laws.zipNe_eq_lexA _ _ _ hl.symm]
      exact (Laws.lexA_lw c32 hc).swap a b)

theorem vec_trans (a b d : List F) :
    sc c32 (.vecf32 a) (.vecf32 b) ≠ .gt → sc c32 (.vecf32 b) (.vecf32 d) ≠ .gt →
    sc c32 (.vecf32 a) (.vecf32 d) ≠ .gt := by
  simp only [sc]
  exact Laws.thn_trans cmpN Laws.cmpN_lw (fun l : List F => l.length) (Laws.zipNe c32) a b d
    (fun h1 h2 => by
      have l1 := (Laws.cmpN_eq_iff _ _).1 h1; have l2 := (Laws.cmpN_eq_iff _ _).1 h2
      rw [Laws.zipNe_eq_lexA _ _ _ l1, Laws.zipNe_eq_lexA _ _ _ l2,
        Laws.zipNe_eq_lexA _ _ _ (l1.trans l2)]
      exact (Laws.lexA_lw c32 hc).trans a b d)

theorem point_swap (x y x' y' : F) :
    sc c32 (.point x' y') (.point x y) = (sc c32 (.point x y) (.point x' y')).swap := by
  simp only [sc]
  exact Laws.thn_swap c32 hc (fun p : F × F => p.2) (fun p q => c32 p.1 q.1) (x, y) (x', y')
    (fun _ => hc.swap x x')

theorem point_trans (x y x' y' x'' y'' : F) :
    sc c32 (.point x y) (.point x' y') ≠ .gt → sc c32 (.point x' y') (.point x'' y'') ≠ .gt →
    sc c32 (.point x y) (.point x'' y'') ≠ .gt := by
  simp only [sc]
  exact Laws.thn_trans c32 hc (fun p : F × F => p.2) (fun p q => c32 p.1 q.1) (x, y) (x', y')
    (x'', y'') (fun _ _ => hc.trans x x' x'')
end kinds

theorem list_swap (c32 : F → F → Ordering) (n : Nat) (ih : IHn c32 n) (xs ys : List V)
    (hx : ∀ x ∈ xs, Pn n x) (hy : ∀ y ∈ ys, Pn n y) :
    sc c32 (.list ys) (.list xs) = (sc c32 (.list xs) (.list ys)).swap := by
  simp only [sc, scList_eq]
  exact Laws.lexA_swap (Pn n) (sc c32) ih.swap xs ys hx hy

theorem list_trans (c32 : F → F → Ordering) (n : Nat) (ih : IHn c32 n) (xs ys zs : List V)
    (hx : ∀ x ∈ xs, Pn n x) (hy : ∀ y ∈ ys, Pn n y) (hz : ∀ z ∈ zs, Pn n z) :
    sc c32 (.list xs) (.list ys) ≠ .gt → sc c32 (.list ys) (.list zs) ≠ .gt →
    sc c32 (.list xs) (.list zs) ≠ .gt := by
  simp only [sc, scList_eq]
  exact Laws.lexA_trans (Pn n) (sc c32) ih.swap ih.trans xs ys zs hx hy hz

/-- The value half of a map compare, as `lexA` once the lengths agree. -/
def mapC (c32 : F → F → Ordering) (m1 m2 : List Nat × List V) : Ordering :=
  thn (Laws.lexA cmpN m1.1 m2.1) (scVals c32 m1.2 m2.2)

theorem sc_map (c32 : F → F → Ordering) (ka kb : List Nat) (va vb : List V) :
    sc c32 (.map ka va) (.map kb vb) =
      thn (cmpN (ka, va).1.length (kb, vb).1.length) (mapC c32 (ka, va) (kb, vb)) := rfl

theorem map_swap (c32 : F → F → Ordering) (n : Nat) (ih : IHn c32 n) (ka kb : List Nat)
    (va vb : List V) (ha : ka.length = va.length ∧ ∀ x ∈ va, Pn n x)
    (hb : kb.length = vb.length ∧ ∀ x ∈ vb, Pn n x) :
    sc c32 (.map kb vb) (.map ka va) = (sc c32 (.map ka va) (.map kb vb)).swap := by
  rw [sc_map, sc_map]
  exact Laws.thn_swap cmpN Laws.cmpN_lw (fun m : List Nat × List V => m.1.length) (mapC c32)
    (ka, va) (kb, vb) (fun h => by
      have hl := (Laws.cmpN_eq_iff _ _).1 h
      have hv : va.length = vb.length := by simp at hl; omega
      simp only [mapC]
      rw [scVals_eq, scVals_eq, Laws.zipNe_eq_lexA _ _ _ hv, Laws.zipNe_eq_lexA _ _ _ hv.symm]
      exact Laws.thn_swap (Laws.lexA cmpN) (Laws.lexA_lw cmpN Laws.cmpN_lw)
        (fun m : List Nat × List V => m.1) (fun m1 m2 => Laws.lexA (sc c32) m1.2 m2.2)
        (ka, va) (kb, vb) (fun _ => Laws.lexA_swap (Pn n) (sc c32) ih.swap va vb ha.2 hb.2))

theorem map_trans (c32 : F → F → Ordering) (n : Nat) (ih : IHn c32 n) (ka kb kd : List Nat)
    (va vb vd : List V) (ha : ka.length = va.length ∧ ∀ x ∈ va, Pn n x)
    (hb : kb.length = vb.length ∧ ∀ x ∈ vb, Pn n x) (hd : kd.length = vd.length ∧ ∀ x ∈ vd, Pn n x) :
    sc c32 (.map ka va) (.map kb vb) ≠ .gt → sc c32 (.map kb vb) (.map kd vd) ≠ .gt →
    sc c32 (.map ka va) (.map kd vd) ≠ .gt := by
  rw [sc_map, sc_map, sc_map]
  exact Laws.thn_trans cmpN Laws.cmpN_lw (fun m : List Nat × List V => m.1.length) (mapC c32)
    (ka, va) (kb, vb) (kd, vd) (fun h1 h2 => by
      have l1 := (Laws.cmpN_eq_iff _ _).1 h1; have l2 := (Laws.cmpN_eq_iff _ _).1 h2
      simp at l1 l2
      have e1 : va.length = vb.length := by omega
      have e2 : vb.length = vd.length := by omega
      simp only [mapC]
      rw [scVals_eq, scVals_eq, scVals_eq, Laws.zipNe_eq_lexA _ _ _ e1,
        Laws.zipNe_eq_lexA _ _ _ e2, Laws.zipNe_eq_lexA _ _ _ (e1.trans e2)]
      exact Laws.thn_trans (Laws.lexA cmpN) (Laws.lexA_lw cmpN Laws.cmpN_lw)
        (fun m : List Nat × List V => m.1) (fun m1 m2 => Laws.lexA (sc c32) m1.2 m2.2)
        (ka, va) (kb, vb) (kd, vd)
        (fun _ _ => Laws.lexA_trans (Pn n) (sc c32) ih.swap ih.trans va vb vd ha.2 hb.2 hd.2))

theorem rank_le (c32 : F → F → Ordering) (a b : V) (h : sc c32 a b ≠ .gt) : rank a ≤ rank b := by
  by_cases hr : rank a = rank b
  · omega
  · rw [sc_rank_ne c32 a b hr] at h
    have := (Laws.cmpN_eq_iff (rank a) (rank b))
    unfold cmpN cmpI at h this; split at h <;> simp_all <;> omega

theorem rank_trans_ne (c32 : F → F → Ordering) (a b d : V) (h1 : sc c32 a b ≠ .gt)
    (h2 : sc c32 b d ≠ .gt) (hne : rank a = rank d → False) : sc c32 a d ≠ .gt := by
  have e1 := rank_le c32 a b h1
  have e2 := rank_le c32 b d h2
  rw [sc_rank_ne c32 a d hne]
  have hne' : rank a ≠ rank d := hne
  have : rank a < rank d := by omega
  unfold cmpN cmpI; simp [show (rank a : Int) < rank d by omega]

theorem scalar_refl (c32 : F → F → Ordering) (a : V) (ha : isComp a = false) (wa : wf a = true) :
    sc c32 a a = .eq := by
  rw [sc_scalar c32 a a ha ha wa wa, (Laws.cmpN_eq_iff _ _).2 rfl, Laws.thn_eq, Laws.cmp3_refl]

/-- **`sort_cmp` is a total preorder** on well-formed values of size `≤ n`, for any
lawful `sort_cmp_f32`. -/
theorem sc_laws (c32 : F → F → Ordering) (hc : Laws.Lw c32) (hr32 : ∀ x, c32 x x = .eq) :
    ∀ n, IHn c32 n
  | 0 => ⟨fun a ha => absurd ha.1 (by have := vsize_pos a; omega),
          fun a _ ha => absurd ha.1 (by have := vsize_pos a; omega),
          fun a _ _ ha => absurd ha.1 (by have := vsize_pos a; omega)⟩
  | n + 1 => by
    have ih := sc_laws c32 hc hr32 n
    refine ⟨?_, ?_, ?_⟩
    · -- reflexive
      intro a ha
      cases hca : isComp a
      · exact scalar_refl c32 a hca ha.2
      · cases a with
        | list xs =>
          simp only [sc, scList_eq]
          exact Laws.lexA_refl (Pn n) (sc c32) ih.refl xs (Pn_list ha)
        | map ks vs =>
          obtain ⟨hl, hv⟩ := Pn_map ha
          simp only [sc, scVals_eq, Laws.zipNe_eq_lexA _ vs vs rfl,
            Laws.lexA_refl (fun _ => True) cmpN (fun x _ => (Laws.cmpN_eq_iff x x).2 rfl) ks
              (fun _ _ => trivial),
            Laws.lexA_refl (Pn n) (sc c32) ih.refl vs hv, (Laws.cmpN_eq_iff _ _).2 rfl, Laws.thn_eq]
        | vecf32 xs =>
          simp only [sc, Laws.zipNe_eq_lexA _ xs xs rfl,
            Laws.lexA_refl (fun _ => True) c32 (fun x _ => hr32 x) xs (fun _ _ => trivial),
            (Laws.cmpN_eq_iff _ _).2 rfl, Laws.thn_eq]
        | point x y => simp only [sc, hr32, Laws.thn_eq]
        | _ => simp [isComp] at hca
    · -- antisymmetric
      intro a b ha hb
      by_cases hr : rank a = rank b
      · cases hca : isComp a
        · have hcb : isComp b = false := by rw [← rank_isComp hr]; exact hca
          rw [sc_scalar c32 a b hca hcb ha.2 hb.2, sc_scalar c32 b a hcb hca hb.2 ha.2]
          exact Laws.thn_swap cmpN Laws.cmpN_lw rank (fun x y => cmp3 (sk x) (sk y)) a b
            (fun _ => Laws.cmp3_lw.swap (sk a) (sk b))
        · cases a with
          | list xs =>
            obtain ⟨ys, rfl⟩ := rank_list hr
            exact list_swap c32 n ih xs ys (Pn_list ha) (Pn_list hb)
          | map ka va =>
            obtain ⟨kb, vb, rfl⟩ := rank_map hr
            exact map_swap c32 n ih ka kb va vb (Pn_map ha) (Pn_map hb)
          | vecf32 xs =>
            obtain ⟨ys, rfl⟩ := rank_vec hr
            exact vec_swap c32 hc xs ys
          | point x y =>
            obtain ⟨x', y', rfl⟩ := rank_point hr
            exact point_swap c32 hc x y x' y'
          | _ => simp [isComp] at hca
      · rw [sc_rank_ne c32 a b hr, sc_rank_ne c32 b a (Ne.symm hr)]
        exact Laws.cmpN_lw.swap _ _
    · -- transitive
      intro a b d ha hb hd h1 h2
      by_cases hab : rank a = rank b
      · by_cases hbd : rank b = rank d
        · cases hca : isComp a
          · have hcb : isComp b = false := by rw [← rank_isComp hab]; exact hca
            have hcd : isComp d = false := by rw [← rank_isComp hbd]; exact hcb
            rw [sc_scalar c32 a b hca hcb ha.2 hb.2] at h1
            rw [sc_scalar c32 b d hcb hcd hb.2 hd.2] at h2
            rw [sc_scalar c32 a d hca hcd ha.2 hd.2]
            exact Laws.thn_trans cmpN Laws.cmpN_lw rank (fun x y => cmp3 (sk x) (sk y)) a b d
              (fun _ _ => Laws.cmp3_lw.trans (sk a) (sk b) (sk d)) h1 h2
          · cases a with
            | list xs =>
              obtain ⟨ys, rfl⟩ := rank_list hab
              obtain ⟨zs, rfl⟩ := rank_list hbd
              exact list_trans c32 n ih xs ys zs (Pn_list ha) (Pn_list hb) (Pn_list hd) h1 h2
            | map ka va =>
              obtain ⟨kb, vb, rfl⟩ := rank_map hab
              obtain ⟨kd, vd, rfl⟩ := rank_map hbd
              exact map_trans c32 n ih ka kb kd va vb vd (Pn_map ha) (Pn_map hb) (Pn_map hd) h1 h2
            | vecf32 xs =>
              obtain ⟨ys, rfl⟩ := rank_vec hab
              obtain ⟨zs, rfl⟩ := rank_vec hbd
              exact vec_trans c32 hc xs ys zs h1 h2
            | point x y =>
              obtain ⟨x', y', rfl⟩ := rank_point hab
              obtain ⟨x'', y'', rfl⟩ := rank_point hbd
              exact point_trans c32 hc x y x' y' x'' y'' h1 h2
            | _ => simp [isComp] at hca
        · exact rank_trans_ne c32 a b d h1 h2 (fun h => hbd (by have := rank_le c32 a b h1; have := rank_le c32 b d h2; omega))
      · exact rank_trans_ne c32 a b d h1 h2 (fun h => hab (by have := rank_le c32 a b h1; have := rank_le c32 b d h2; omega))

theorem lw_of_key {α κ : Type} (k : α → κ) (ck : κ → κ → Ordering) (h : Laws.Lw ck) :
    Laws.Lw (fun a b => ck (k a) (k b)) :=
  ⟨fun a b => h.swap (k a) (k b), fun a b d => h.trans (k a) (k b) (k d)⟩

theorem scF64_lw : Laws.Lw scF64 := by
  have := lw_of_key fkey cmp3 Laws.cmp3_lw
  exact ⟨fun a b => by rw [scF64_key, scF64_key]; exact this.swap a b,
         fun a b d => by rw [scF64_key, scF64_key, scF64_key]; exact this.trans a b d⟩

theorem scF64_refl (x : F) : scF64 x x = .eq := by rw [scF64_key]; exact Laws.cmp3_refl _

/-- **main's `sort_cmp` is a total preorder** on well-formed values (nested lists and
maps included): reflexive, antisymmetric, transitive. -/
theorem scMain_total (a b d : V) (wa : wf a = true) (wb : wf b = true) (wd : wf d = true) :
    scMain a a = .eq ∧ scMain b a = (scMain a b).swap ∧
    (scMain a b ≠ .gt → scMain b d ≠ .gt → scMain a d ≠ .gt) := by
  have L := sc_laws scF64 scF64_lw scF64_refl (vsize a + vsize b + vsize d)
  have pa : Pn (vsize a + vsize b + vsize d) a := ⟨by omega, wa⟩
  have pb : Pn (vsize a + vsize b + vsize d) b := ⟨by omega, wb⟩
  have pd : Pn (vsize a + vsize b + vsize d) d := ⟨by omega, wd⟩
  exact ⟨L.refl a pa, L.swap a b pa pb, L.trans a b d pa pb pd⟩

/-! ## Main: `sort_cmp`-equal does not imply `hash`-equal -/

/-- A NaN with another payload/sign (`-(0.0/0.0)` vs `f64::NAN`, say). -/
def nan2 : F := .nan true (2 ^ 51 - 1)

theorem main_float_nan_hash_differs :
    scMain (.flt F.qnan) (.flt nan2) = .eq ∧ hwords (.flt F.qnan) ≠ hwords (.flt nan2) := by
  decide

theorem main_vec_nan_hash_differs :
    scMain (.vecf32 [F.qnan]) (.vecf32 [nan2]) = .eq ∧
    hwords (.vecf32 [F.qnan]) ≠ hwords (.vecf32 [nan2]) := by decide

theorem main_vec_zero_hash_differs :
    scMain (.vecf32 [.negZero]) (.vecf32 [.num 0]) = .eq ∧
    hwords (.vecf32 [.negZero]) ≠ hwords (.vecf32 [.num 0]) := by decide

theorem main_point_zero_hash_differs :
    scMain (.point .negZero (.num 0)) (.point (.num 0) (.num 0)) = .eq ∧
    hwords (.point .negZero (.num 0)) ≠ hwords (.point (.num 0) (.num 0)) := by decide

theorem main_point_nan_hash_differs :
    scMain (.point F.qnan (.num 1)) (.point nan2 (.num 1)) = .eq ∧
    hwords (.point F.qnan (.num 1)) ≠ hwords (.point nan2 (.num 1)) := by decide

/-- **Refuted on main**: `sort_cmp`-equal keys need not hash alike. -/
theorem main_hash_not_consistent :
    ¬ ∀ a b : V, wf a = true → wf b = true → scMain a b = .eq → hwords a = hwords b := fun h =>
  main_vec_zero_hash_differs.2 (h _ _ rfl rfl main_vec_zero_hash_differs.1)

/-- main's `GroupKey` equality is `Value::eq` (`compare_value(..).0 == Equal`), and it
too merges signed zeros in vectors and points while the hash keeps them apart. -/
theorem vec_zero_eq_but_hash_differs :
    eqV (.vecf32 [.negZero]) (.vecf32 [.num 0]) = true ∧
    hwords (.vecf32 [.negZero]) ≠ hwords (.vecf32 [.num 0]) ∧
    eqV (.point .negZero (.num 0)) (.point (.num 0) (.num 0)) = true ∧
    hwords (.point .negZero (.num 0)) ≠ hwords (.point (.num 0) (.num 0)) := by decide

#print axioms scMain_total
#print axioms main_hash_not_consistent
#print axioms vec_zero_eq_but_hash_differs

end ValueOrder.SC
