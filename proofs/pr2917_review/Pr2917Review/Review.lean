import Pr2917Review.SortCmp
/-!
# PR #2917 at 14e341cf4: grouping keys compare by `sort_cmp` and hash consistently

| here | there (branch fix/2913-compare-mismatches @ 14e341cf4) |
| --- | --- |
| `scF32`        | `sort_cmp_f32` (value.rs:1905): `-0.0` sorts before `0.0` |
| `f32HashBits`  | `f32_hash_bits` (value.rs:1919): one value for every NaN |
| `floatWordPR`  | `Hash for Value`, `Float` arm (value.rs:839-850): `else if x.is_nan()` |
| `hw`, `hwordsPR` | `Hash for Value` (value.rs:822); VecF32/Point via `f32_hash_bits` (:876-886) |
| `scPR`         | `Value::sort_cmp` (value.rs:1529) with the new `sort_cmp_f32` |
| `gkEqR`, `gkHashR` | `GroupKey::eq` (ops/aggregate.rs:94), `GroupKey::hash` (:110) |
| `gkEq`, `gkHash`, `gkHash_consistent` | copied from proofs/ops_aggregate `AggState.lean` |
| `hwordsPre`, `scPre` | the PR at 203f2022c (Float NaN canonical; vectors/points still `to_bits`, main's `sort_cmp_f32`) |
-/
namespace Pr2917
open ValueOrder ValueOrder.Laws ValueOrder.SC

/-! ## `sort_cmp_f32` -/

/-- `sort_cmp_f32` (value.rs:1905). -/
def scF32 (a b : F) : Ordering :=
  match fpcmp a b with
  | some .eq => cmpB (isSignNeg b) (isSignNeg a)
  | some o => o
  | none => cmpB (isNaN a) (isNaN b)

/-- Its key: like `fkey`, but `-0.0` strictly below `+0.0`. -/
def fkey32 : F → Nat × Int × Nat
  | .nan _ _ => (3, 0, 0)
  | .negInf => (0, 0, 0)
  | .negZero => (1, 0, 0)
  | .num n => (1, n, 1)
  | .frac lo r => (1, lo, r + 2)
  | .posInf => (2, 0, 0)

theorem cmpI_trich (x y : Int) :
    (cmpI x y = .lt ∧ x < y) ∨ (cmpI x y = .eq ∧ x = y) ∨ (cmpI x y = .gt ∧ y < x) := by
  unfold cmpI
  by_cases h1 : x < y
  · simp [h1]
  · by_cases h2 : x = y
    · simp [h2]
    · right; right; simp [h1, h2]; omega

/-- One case of `scF32_key` with both sides in the same numeric class. -/
macro "f32case" x:term:max y:term:max : tactic => `(tactic| (
  simp only [scF32, fpcmp, F.nkey, fkey32, cmp3_one]
  rcases cmpI_trich $x $y with ⟨h, h'⟩ | ⟨h, h'⟩ | ⟨h, h'⟩ <;> simp only [h, thn] <;>
    (try subst h') <;> simp [cmpN, cmpI, cmpB, isSignNeg] <;>
    (repeat' split) <;> (first | rfl | omega | (simp_all; done) | (exfalso; omega))))

theorem cmpN_lt {a b : Nat} (h : a < b) : cmpN a b = .lt := cmpI_of_lt (by omega)
theorem cmpN_gt {a b : Nat} (h : b < a) : cmpN a b = .gt := cmpI_of_gt (by omega)

theorem scF32_key (a b : F) : scF32 a b = cmp3 (fkey32 a) (fkey32 b) := by
  cases a <;> cases b
  all_goals first
    | (simp [scF32, fpcmp, F.nkey, fkey32, cmp3, cmpI, isNaN, cmpB, cmpN]; done)
    | skip
  case negZero.num m => f32case 0 m
  case negZero.frac l r => f32case 0 l
  case num.negZero n => f32case n 0
  case num.num n m => f32case n m
  case num.frac n l r => f32case n l
  case frac.negZero l r =>
    simp only [scF32, fpcmp, F.nkey, fkey32, cmp3_one]
    rcases cmpI_trich l 0 with ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, _⟩ <;> simp only [h, thn]
    rw [cmpN_gt (by omega), cmpN_gt (by omega)]
  case frac.num l r m =>
    simp only [scF32, fpcmp, F.nkey, fkey32, cmp3_one]
    rcases cmpI_trich l m with ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, _⟩ <;> simp only [h, thn]
    rw [cmpN_gt (by omega), cmpN_gt (by omega)]
  case frac.frac l r l' r' =>
    simp only [scF32, fpcmp, F.nkey, fkey32, cmp3_one]
    rcases cmpI_trich l l' with ⟨h, _⟩ | ⟨h, h'⟩ | ⟨h, _⟩ <;> simp only [h, thn]
    subst h'
    rcases cmpI_trich r r' with ⟨g, g'⟩ | ⟨g, g'⟩ | ⟨g, g'⟩
    · rw [cmpN_lt (by omega), cmpN_lt (by omega)]
    · have : r = r' := by omega
      subst this
      simp [cmpN, cmpI, cmpB, isSignNeg]
    · rw [cmpN_gt (by omega), cmpN_gt (by omega)]

theorem scF32_lw : Lw scF32 := by
  have := lw_of_key fkey32 cmp3 cmp3_lw
  exact ⟨fun a b => by rw [scF32_key, scF32_key]; exact this.swap a b,
         fun a b d => by rw [scF32_key, scF32_key, scF32_key]; exact this.trans a b d⟩

theorem scF32_refl (x : F) : scF32 x x = .eq := by rw [scF32_key]; exact cmp3_refl _

/-- `Value::sort_cmp` at 14e341cf4. -/
def scPR : V → V → Ordering := sc scF32

/-- **`sort_cmp` is a total preorder on the extended model** (every well-formed value,
nested lists/maps/vectors/points included). -/
theorem scPR_total (a b d : V) (wa : wf a = true) (wb : wf b = true) (wd : wf d = true) :
    scPR a a = .eq ∧ scPR b a = (scPR a b).swap ∧
    (scPR a b ≠ .gt → scPR b d ≠ .gt → scPR a d ≠ .gt) := by
  have L := sc_laws scF32 scF32_lw scF32_refl (vsize a + vsize b + vsize d)
  have pa : Pn (vsize a + vsize b + vsize d) a := ⟨by omega, wa⟩
  have pb : Pn (vsize a + vsize b + vsize d) b := ⟨by omega, wb⟩
  have pd : Pn (vsize a + vsize b + vsize d) d := ⟨by omega, wd⟩
  exact ⟨L.refl a pa, L.swap a b pa pb, L.trans a b d pa pb pd⟩

/-! ## `Hash for Value` -/

/-- `f32::NAN.to_bits()`. -/
def F32_NAN_BITS : Nat := 0x7fc00000

/-- `f32_hash_bits` (value.rs:1919). -/
def f32HashBits (f : F) : Nat := if isNaN f then F32_NAN_BITS else bits32 f

/-- The `Float` arm at 14e341cf4: the integer if `x - (x as i64) as f64 == 0`, else
one word for every NaN, else `to_bits`. -/
def floatWordPR (f : F) : UInt64 :=
  if diffZero f then w (asI64 f) else if isNaN f then NAN_BITS else bits64 f

mutual
/-- `Hash for Value` as its word stream, with the Float word `fw` and the f32 component
word `f32w` as parameters (main: `floatWord`/`bits32` = Model's `hwords`). -/
def hw (fw : F → UInt64) (f32w : F → Nat) : V → List UInt64
  | .null => [0]
  | .bool b => [1, if b then 1 else 0]
  | .int i => [2, w i]
  | .flt f => [2, fw f]
  | .str s => [3, UInt64.ofNat s, 0xff]
  | .list xs => [4, UInt64.ofNat xs.length] ++ hwL fw f32w xs
  | .map ks vs => [5, UInt64.ofNat ks.length] ++ hwM fw f32w ks vs
  | .node n => [6, UInt64.ofNat n]
  | .vecf32 xs => [9] ++ xs.map (fun f => UInt64.ofNat (f32w f))
  | .point lat lon => [10, UInt64.ofNat (f32w lat), UInt64.ofNat (f32w lon)]

def hwL (fw : F → UInt64) (f32w : F → Nat) : List V → List UInt64
  | [] => []
  | x :: xs => hw fw f32w x ++ hwL fw f32w xs

def hwM (fw : F → UInt64) (f32w : F → Nat) : List Nat → List V → List UInt64
  | k :: ks, v :: vs => [UInt64.ofNat k, 0xff] ++ hw fw f32w v ++ hwM fw f32w ks vs
  | _, _ => []
end

/-- The hash at 14e341cf4. -/
def hwordsPR : V → List UInt64 := hw floatWordPR f32HashBits

/-! ### Floats: equal keys, equal words -/

theorem f32HashBits_of_eq (x y : F) (h : scF32 x y = .eq) : f32HashBits x = f32HashBits y := by
  rw [scF32_key, cmp3_eq_iff] at h
  cases x <;> cases y <;> simp [fkey32] at h <;> (try subst h) <;> (try (obtain ⟨rfl, rfl⟩ := h)) <;>
    simp [f32HashBits, isNaN, F.nkey]

theorem floatWordPR_of_key (x y : F) (h : fkey x = fkey y) : floatWordPR x = floatWordPR y := by
  cases x <;> cases y <;> simp [fkey, F.nkey] at h <;> (try subst h) <;>
    (try (obtain ⟨rfl, rfl⟩ := h)) <;>
    simp [floatWordPR, diffZero, asI64, isNaN, F.nkey, satI64, I64MIN, I64MAX, roundF, roundNat,
      bitlen, bitlenAux] <;> (try omega)

/-- An `Int` and a `Float` with one key hash alike (well-formed: `i64`, representable). -/
theorem int_float_hash (i : Int) (f : F) (wi : I64MIN ≤ i ∧ i ≤ I64MAX)
    (wf' : ∀ n, f = .num n → roundF n = n) (h : (1, i, 0) = fkey f) :
    [(2 : UInt64), w i] = [2, floatWordPR f] := by
  cases f <;> simp [fkey, F.nkey] at h
  · -- negZero
    subst h; simp [floatWordPR, diffZero, asI64]
  · -- num
    subst h
    have hr := wf' _ rfl
    have hs : satI64 i = i := by unfold satI64; split <;> (try split) <;> omega
    simp [floatWordPR, diffZero, asI64, hs, hr]

/-! ### The consistency theorem -/

theorem hwL_of_lexA (fw : F → UInt64) (f32w : F → Nat) :
    ∀ xs ys : List V, (∀ x ∈ xs, ∀ y ∈ ys, scPR x y = .eq → hw fw f32w x = hw fw f32w y) →
    lexA scPR xs ys = .eq → hwL fw f32w xs = hwL fw f32w ys
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp [lexA] at h
  | _ :: _, [], _, h => by simp [lexA] at h
  | x :: xs, y :: ys, H, h => by
    simp only [lexA, thn_eq_eq] at h
    simp only [hwL, H x (by simp) y (by simp) h.1,
      hwL_of_lexA fw f32w xs ys (fun a ha b hb => H a (by simp [ha]) b (by simp [hb])) h.2]

theorem hwM_of_lexA (fw : F → UInt64) (f32w : F → Nat) :
    ∀ (ks : List Nat) (xs ys : List V), (∀ x ∈ xs, ∀ y ∈ ys, scPR x y = .eq → hw fw f32w x = hw fw f32w y) →
    lexA scPR xs ys = .eq → hwM fw f32w ks xs = hwM fw f32w ks ys
  | [], _, _, _, _ => by simp [hwM]
  | _ :: _, [], [], _, _ => rfl
  | _ :: _, [], _ :: _, _, h => by simp [lexA] at h
  | _ :: _, _ :: _, [], _, h => by simp [lexA] at h
  | k :: ks, x :: xs, y :: ys, H, h => by
    simp only [lexA, thn_eq_eq] at h
    simp only [hwM, H x (by simp) y (by simp) h.1,
      hwM_of_lexA fw f32w ks xs ys (fun a ha b hb => H a (by simp [ha]) b (by simp [hb])) h.2]

theorem lexA_cmpN_eq : ∀ xs ys : List Nat, lexA cmpN xs ys = .eq → xs = ys
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [lexA] at h
  | _ :: _, [], h => by simp [lexA] at h
  | x :: xs, y :: ys, h => by
    simp only [lexA, thn_eq_eq] at h
    rw [(cmpN_eq_iff x y).1 h.1, lexA_cmpN_eq xs ys h.2]

theorem vec_map_of_zipNe : ∀ xs ys : List F, xs.length = ys.length → zipNe scF32 xs ys = .eq →
    xs.map (fun f => UInt64.ofNat (f32HashBits f)) = ys.map (fun f => UInt64.ofNat (f32HashBits f))
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | x :: xs, y :: ys, hl, h => by
    simp only [zipNe, thn_eq_eq] at h
    simp only [List.map_cons, f32HashBits_of_eq x y h.1,
      vec_map_of_zipNe xs ys (by simpa using hl) h.2]

theorem scalar_hash (a b : V) (ha : isComp a = false) (hr : rank a = rank b)
    (hk : sk a = sk b) (wa : wf a = true) (wb : wf b = true) : hwordsPR a = hwordsPR b := by
  cases a <;> cases b <;> simp [isComp, rank, V.tag, orderTag] at ha hr <;>
    simp [sk] at hk <;> (try subst hk) <;> simp [hwordsPR, hw]
  all_goals first
    | exact floatWordPR_of_key _ _ hk
    | (simp only [wf, decide_eq_true_eq] at wa
       have := int_float_hash _ _ wa (fun n hn => by subst hn; simpa [wf] using wb) hk
       simpa using this)
    | (simp only [wf, decide_eq_true_eq] at wb
       have := int_float_hash _ _ wb (fun n hn => by subst hn; simpa [wf] using wa) hk.symm
       simpa using this.symm)
    | (rename_i x y; cases x <;> cases y <;> simp_all <;> done)
    | (rename_i x y; have e : x = y := by omega
       subst e; rfl)

/-- **`sort_cmp`-equal ⇒ hash-equal at 14e341cf4**, for every well-formed value. -/
theorem hash_consistent : ∀ n (a b : V), vsize a ≤ n → vsize b ≤ n → wf a = true → wf b = true →
    scPR a b = .eq → hwordsPR a = hwordsPR b
  | 0, a, _, ha, _, _, _, _ => absurd ha (by have := vsize_pos a; omega)
  | n + 1, a, b, sa, sb, wa, wb, h => by
    have ih := hash_consistent n
    by_cases hr : rank a = rank b
    · cases hca : isComp a
      · have hcb : isComp b = false := by rw [← rank_isComp hr]; exact hca
        unfold scPR at h
        rw [sc_scalar scF32 a b hca hcb wa wb, (cmpN_eq_iff _ _).2 hr, thn_eq, cmp3_eq_iff] at h
        exact scalar_hash a b hca hr h wa wb
      · have pa : Pn (n + 1) a := ⟨sa, wa⟩
        have pb : Pn (n + 1) b := ⟨sb, wb⟩
        cases a with
        | list xs =>
          obtain ⟨ys, rfl⟩ := rank_list hr
          unfold scPR at h; simp only [sc, scList_eq] at h
          have hx := Pn_list pa; have hy := Pn_list pb
          simp only [hwordsPR, hw]
          rw [hwL_of_lexA _ _ xs ys (fun x hxm y hym e =>
            ih x y (hx x hxm).1 (hy y hym).1 (hx x hxm).2 (hy y hym).2 e) h]
          rw [(lexA_eq _ xs ys h).1]
        | map ka va =>
          obtain ⟨kb, vb, rfl⟩ := rank_map hr
          unfold scPR at h; simp only [sc, thn_eq_eq] at h
          obtain ⟨h1, h2, h3⟩ := h
          have hk := lexA_cmpN_eq ka kb h2
          subst hk
          obtain ⟨la, hx⟩ := Pn_map pa; obtain ⟨lb, hy⟩ := Pn_map pb
          rw [scVals_eq, zipNe_eq_lexA _ _ _ (by omega)] at h3
          simp only [hwordsPR, hw]
          rw [hwM_of_lexA _ _ ka va vb (fun x hxm y hym e =>
            ih x y (hx x hxm).1 (hy y hym).1 (hx x hxm).2 (hy y hym).2 e) h3]
        | vecf32 xs =>
          obtain ⟨ys, rfl⟩ := rank_vec hr
          unfold scPR at h; simp only [sc, thn_eq_eq] at h
          have hl := (cmpN_eq_iff _ _).1 h.1
          simp only [hwordsPR, hw, vec_map_of_zipNe xs ys hl h.2]
        | point x y =>
          obtain ⟨x', y', rfl⟩ := rank_point hr
          unfold scPR at h; simp only [sc, thn_eq_eq] at h
          simp only [hwordsPR, hw, f32HashBits_of_eq _ _ h.1, f32HashBits_of_eq _ _ h.2]
        | _ => simp [isComp] at hca
    · unfold scPR at h
      rw [sc_rank_ne scF32 a b hr] at h
      exact absurd ((cmpN_eq_iff _ _).1 h) hr

theorem hashPR_consistent (a b : V) (wa : wf a = true) (wb : wf b = true) (h : scPR a b = .eq) :
    hwordsPR a = hwordsPR b :=
  hash_consistent (vsize a + vsize b) a b (by omega) (by omega) wa wb h

/-! ## `GroupKey` (ops/aggregate.rs) -/

/-- `GroupKey::eq` at 14e341cf4 (aggregate.rs:94): equal lengths, every pair `sort_cmp`-equal. -/
def gkEqR (a b : List V) : Bool :=
  a.length == b.length && (a.zip b).all (fun p => scPR p.1 p.2 == .eq)

/-- `GroupKey::hash` (aggregate.rs:110): `for v in &self.0 { v.hash(state) }`. -/
def gkHashR (st : List UInt64) (vs : List V) : List UInt64 := vs.foldl (fun s v => s ++ hwordsPR v) st

/-! Copied from proofs/ops_aggregate `FalkorAggOps/AggState.lean` (`gkEq`, `gkHash`,
`gkHash_consistent`), where `hc` was an undischarged hypothesis. -/

def gkEq {Z : Type} (veq : Z → Z → Bool) : List Z → List Z → Bool
  | [], [] => true
  | a :: as, b :: bs => veq a b && gkEq veq as bs
  | _, _ => false

def gkHash {Z S : Type} (hv : S → Z → S) (st : S) (vs : List Z) : S := vs.foldl hv st

theorem gkHash_consistent {Z S : Type} (veq : Z → Z → Bool) (hv : S → Z → S)
    (hc : ∀ s x y, veq x y = true → hv s x = hv s y) (st : S) (a b : List Z)
    (h : gkEq veq a b = true) : gkHash hv st a = gkHash hv st b := by
  induction a generalizing b st with
  | nil => cases b <;> simp_all [gkEq]
  | cons x xs ih =>
    cases b with
    | nil => simp [gkEq] at h
    | cons y ys =>
      simp only [gkEq, Bool.and_eq_true] at h
      simp only [gkHash, List.foldl_cons]
      rw [hc st x y h.1]
      exact ih _ ys h.2

/-- Well-formed values (what the engine builds). -/
abbrev WV := {v : V // wf v = true}

def veqPR (x y : WV) : Bool := scPR x.1 y.1 == .eq
def hvPR (s : List UInt64) (x : WV) : List UInt64 := s ++ hwordsPR x.1

/-- **ops_aggregate's `hc`, discharged for the PR**: `Value`'s hash agrees with the
`sort_cmp` equality `GroupKey::eq` now uses. -/
theorem hc_pr : ∀ s x y, veqPR x y = true → hvPR s x = hvPR s y := by
  intro s x y h
  simp only [veqPR, beq_iff_eq] at h
  simp only [hvPR, hashPR_consistent x.1 y.1 x.2 y.2 h]

/-- The `HashMap<GroupKey, _>` contract at 14e341cf4: equal keys hash alike. -/
theorem gk_hash_consistent (st : List UInt64) (a b : List WV) (h : gkEq veqPR a b = true) :
    gkHash hvPR st a = gkHash hvPR st b :=
  gkHash_consistent veqPR hvPR hc_pr st a b h

/-- `gkEq veqPR` is `GroupKey::eq` (length check + pairwise `sort_cmp == Equal`). -/
theorem gkEq_spec : ∀ a b : List WV, gkEq veqPR a b = gkEqR (a.map (·.1)) (b.map (·.1))
  | [], [] => rfl
  | [], _ :: _ => by simp [gkEq, gkEqR]
  | _ :: _, [] => by simp [gkEq, gkEqR]
  | x :: xs, y :: ys => by
    have ih := gkEq_spec xs ys
    simp only [gkEqR, List.map_cons, List.length_cons, List.zip_cons_cons, List.all_cons] at ih ⊢
    simp only [gkEq, veqPR, ih, gkEqR]
    cases (scPR x.1 y.1 == .eq) <;> simp [Nat.add_right_cancel_iff]

/-! ## The PR's claims -/

/-- Float `-0.0` and `0.0` are one key. -/
theorem float_signed_zero_one_key :
    scPR (.flt .negZero) (.flt (.num 0)) = .eq ∧
    hwordsPR (.flt .negZero) = hwordsPR (.flt (.num 0)) := by decide

/-- Every Float NaN, whatever its sign and payload, is one key. -/
theorem float_nans_one_key (s s' : Bool) (p p' : Nat) :
    scPR (.flt (.nan s p)) (.flt (.nan s' p')) = .eq ∧
    hwordsPR (.flt (.nan s p)) = hwordsPR (.flt (.nan s' p')) := by
  simp [scPR, sc, scF64, fpcmp, F.nkey, isNaN, cmpB, cmpN, cmpI, hwordsPR, hw, floatWordPR, diffZero]

/-- Vector NaN components are one key, whatever their sign and payload. -/
theorem vec_nans_one_key (s s' : Bool) (p p' : Nat) :
    scPR (.vecf32 [.nan s p]) (.vecf32 [.nan s' p']) = .eq ∧
    hwordsPR (.vecf32 [.nan s p]) = hwordsPR (.vecf32 [.nan s' p']) := by
  simp [scPR, sc, scF32, zipNe, fpcmp, F.nkey, isNaN, cmpB, cmpN, cmpI, thn, hwordsPR, hw,
    f32HashBits]

/-- Point NaN components are one key. (The Rust `point()` constructor refuses a NaN
coordinate — "latitude must be a finite number" — so this is defensive.) -/
theorem point_nans_one_key (s s' : Bool) (p p' : Nat) (y : F) :
    scPR (.point (.nan s p) y) (.point (.nan s' p') y) = .eq ∧
    hwordsPR (.point (.nan s p) y) = hwordsPR (.point (.nan s' p') y) := by
  refine ⟨?_, ?_⟩
  · simp only [scPR, sc, scF32_refl, thn_eq]
    simp [scF32, fpcmp, F.nkey, isNaN, cmpB, cmpN, cmpI]
  · simp [hwordsPR, hw, f32HashBits, isNaN, F.nkey]

/-- Vector and point signed zeros stay apart (as in C: C groups them by their bits). -/
theorem signed_zeros_apart :
    scPR (.vecf32 [.negZero]) (.vecf32 [.num 0]) = .lt ∧
    scPR (.point .negZero (.num 0)) (.point (.num 0) (.num 0)) = .lt ∧
    scPR (.list [.vecf32 [.negZero]]) (.list [.vecf32 [.num 0]]) = .lt := by decide

/-! ## Before 14e341cf4 (the PR at 203f2022c): the counterexamples -/

/-- 203f2022c: main's `sort_cmp_f32`, Float NaN already canonical, components hashed by `to_bits`. -/
def scPre : V → V → Ordering := sc scF64
def hwordsPre : V → List UInt64 := hw floatWordPR bits32

def nan2 : F := .nan true (2 ^ 51 - 1)

/-- **pre-14e341cf4**: two vector NaNs were `sort_cmp`-equal but hashed apart (Rust 2 groups, C 1). -/
theorem pre14e341_vec_nan_hash_differs :
    scPre (.vecf32 [F.qnan]) (.vecf32 [nan2]) = .eq ∧
    hwordsPre (.vecf32 [F.qnan]) ≠ hwordsPre (.vecf32 [nan2]) := by decide

/-- **pre-14e341cf4**: `-0.0`/`0.0` vector components were `sort_cmp`-equal but hashed
apart: the `HashMap` contract was broken (C keeps them apart). -/
theorem pre14e341_vec_zero_eq_but_hash_differs :
    scPre (.vecf32 [.negZero]) (.vecf32 [.num 0]) = .eq ∧
    hwordsPre (.vecf32 [.negZero]) ≠ hwordsPre (.vecf32 [.num 0]) := by decide

theorem pre14e341_point_zero_eq_but_hash_differs :
    scPre (.point .negZero (.num 0)) (.point (.num 0) (.num 0)) = .eq ∧
    hwordsPre (.point .negZero (.num 0)) ≠ hwordsPre (.point (.num 0) (.num 0)) := by decide

#print axioms scPR_total
#print axioms hashPR_consistent
#print axioms gk_hash_consistent
#print axioms float_nans_one_key
#print axioms vec_nans_one_key
#print axioms pre14e341_vec_nan_hash_differs

end Pr2917
