import FalkorValueOrder.SortCmp
/-!
# Main (4a4f2a72f): `sort_cmp` with #2917's `sort_cmp_f32`, and `Hash for Value`

#2917 (7fdf06a0e) is merged; the files it touched in this area are byte-identical to the
PR head 14e341cf4 that proofs/pr2917_review modelled. `scF32`, `fkey32` and the
hash-consistency proof are copied from proofs/pr2917_review `Review.lean`, re-stated for
this project's `hwords` (Model.lean, now main's hash).

| here | there (origin/main 4a4f2a72f, graph/src/runtime/value.rs) |
| --- | --- |
| `scF32`      | `sort_cmp_f32` (:1905): `-0.0` sorts before `0.0`, NaN on top |
| `scMain`     | `Value::sort_cmp` (:1529) / `sort_cmp_map` (:1566) = `sc scF32` |
| `hwords`, `floatWord`, `f32HashBits` | `Hash for Value` (:823, Float arm :839-850, VecF32/Point :876-886), `f32_hash_bits` (:1919) — Model.lean |
| `cmpKey`     | `OrderedKey::cmp_key` (ops/sort.rs:76) |
| `compareAt`  | `Column::compare_at` (batch.rs:400) |
| `compareRowsAt` | `Batch::compare_rows_at` (batch.rs:1282) |
| `listSort`   | `list.sort` (functions/list.rs:210): `sort_by(Value::sort_cmp)` |
-/
namespace ValueOrder.Main
open ValueOrder ValueOrder.Laws ValueOrder.SC

/-! ## `sort_cmp_f32` (copied from pr2917_review) -/

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

/-! ## Main's `sort_cmp` is a total preorder -/

/-- `Value::sort_cmp` on main (value.rs:1529). -/
def scMain : V → V → Ordering := sc scF32

/-- **Main's `sort_cmp` is a total preorder** on every well-formed value (Ints are `i64`,
Float `num`s representable, one value per map key), nested lists/maps/vectors/points
included: reflexive, antisymmetric, transitive. `slice::sort_by` cannot panic on it. -/
theorem scMain_total (a b d : V) (wa : wf a = true) (wb : wf b = true) (wd : wf d = true) :
    scMain a a = .eq ∧ scMain b a = (scMain a b).swap ∧
    (scMain a b ≠ .gt → scMain b d ≠ .gt → scMain a d ≠ .gt) := by
  have L := sc_laws scF32 scF32_lw scF32_refl (vsize a + vsize b + vsize d)
  have pa : Pn (vsize a + vsize b + vsize d) a := ⟨by omega, wa⟩
  have pb : Pn (vsize a + vsize b + vsize d) b := ⟨by omega, wb⟩
  have pd : Pn (vsize a + vsize b + vsize d) d := ⟨by omega, wd⟩
  exact ⟨L.refl a pa, L.swap a b pa pb, L.trans a b d pa pb pd⟩

/-- The NaN/2^53 inputs that broke `compare_value` are ordered by `sort_cmp`: NaN is above
every number and equal to itself, `2^53+1 > 2^53.0`, `{a:1,b:1} < {a:1,b:2}` with
`{a:'s',b:3}` above both (maps compare values by `sort_cmp`, types by rank). -/
theorem scMain_fixes_2891 :
    scMain (.flt F.qnan) (.flt (.num 1)) = .gt ∧ scMain (.flt (.num 1)) (.flt F.qnan) = .lt ∧
    scMain (.flt F.qnan) (.flt F.qnan) = .eq ∧
    scMain (.int (P53 + 1)) (.flt (.num P53)) = .gt ∧
    scMain m1 m3 = .lt ∧ scMain m1 m2 ≠ .eq ∧ scMain m2 m3 ≠ .eq := by decide

/-! ## `Hash for Value` agrees with `sort_cmp` (copied from pr2917_review, on `hwords`) -/

theorem f32HashBits_of_eq (x y : F) (h : scF32 x y = .eq) : f32HashBits x = f32HashBits y := by
  rw [scF32_key, cmp3_eq_iff] at h
  cases x <;> cases y <;> simp [fkey32] at h <;> (try subst h) <;> (try (obtain ⟨rfl, rfl⟩ := h)) <;>
    simp [f32HashBits, F.nkey]

theorem floatWord_of_key (x y : F) (h : fkey x = fkey y) : floatWord x = floatWord y := by
  cases x <;> cases y <;> simp [fkey, F.nkey] at h <;> (try subst h) <;>
    (try (obtain ⟨rfl, rfl⟩ := h)) <;>
    simp [floatWord, diffZero, asI64, F.nkey, satI64, I64MIN, I64MAX, roundF, roundNat,
      bitlen, bitlenAux] <;> (try omega)

/-- An `Int` and a `Float` with one key hash alike (well-formed: `i64`, representable). -/
theorem int_float_hash (i : Int) (f : F) (wi : I64MIN ≤ i ∧ i ≤ I64MAX)
    (wf' : ∀ n, f = .num n → roundF n = n) (h : (1, i, 0) = fkey f) :
    [(2 : UInt64), w i] = [2, floatWord f] := by
  cases f <;> simp [fkey, F.nkey] at h
  · subst h; simp [floatWord, diffZero, asI64]
  · subst h
    have hr := wf' _ rfl
    have hs : satI64 i = i := by unfold satI64; split <;> (try split) <;> omega
    simp [floatWord, diffZero, asI64, hs, hr]

theorem hwordsList_of_lexA :
    ∀ xs ys : List V, (∀ x ∈ xs, ∀ y ∈ ys, scMain x y = .eq → hwords x = hwords y) →
    lexA scMain xs ys = .eq → hwordsList xs = hwordsList ys
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp [lexA] at h
  | _ :: _, [], _, h => by simp [lexA] at h
  | x :: xs, y :: ys, H, h => by
    simp only [lexA, thn_eq_eq] at h
    simp only [hwordsList, H x (by simp) y (by simp) h.1,
      hwordsList_of_lexA xs ys (fun a ha b hb => H a (by simp [ha]) b (by simp [hb])) h.2]

theorem hwordsMap_of_lexA :
    ∀ (ks : List Nat) (xs ys : List V), (∀ x ∈ xs, ∀ y ∈ ys, scMain x y = .eq → hwords x = hwords y) →
    lexA scMain xs ys = .eq → hwordsMap ks xs = hwordsMap ks ys
  | [], _, _, _, _ => by simp [hwordsMap]
  | _ :: _, [], [], _, _ => rfl
  | _ :: _, [], _ :: _, _, h => by simp [lexA] at h
  | _ :: _, _ :: _, [], _, h => by simp [lexA] at h
  | k :: ks, x :: xs, y :: ys, H, h => by
    simp only [lexA, thn_eq_eq] at h
    simp only [hwordsMap, H x (by simp) y (by simp) h.1,
      hwordsMap_of_lexA ks xs ys (fun a ha b hb => H a (by simp [ha]) b (by simp [hb])) h.2]

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
    (hk : sk a = sk b) (wa : wf a = true) (wb : wf b = true) : hwords a = hwords b := by
  cases a <;> cases b <;> simp [isComp, rank, V.tag, orderTag] at ha hr <;>
    simp [sk] at hk <;> (try subst hk) <;> simp [hwords]
  all_goals first
    | exact floatWord_of_key _ _ hk
    | (simp only [wf, decide_eq_true_eq] at wa
       have := int_float_hash _ _ wa (fun n hn => by subst hn; simpa [wf] using wb) hk
       simpa using this)
    | (simp only [wf, decide_eq_true_eq] at wb
       have := int_float_hash _ _ wb (fun n hn => by subst hn; simpa [wf] using wa) hk.symm
       simpa using this.symm)
    | (rename_i x y; cases x <;> cases y <;> simp_all <;> done)
    | (rename_i x y; have e : x = y := by omega
       subst e; rfl)

theorem hash_consistent : ∀ n (a b : V), vsize a ≤ n → vsize b ≤ n → wf a = true → wf b = true →
    scMain a b = .eq → hwords a = hwords b
  | 0, a, _, ha, _, _, _, _ => absurd ha (by have := vsize_pos a; omega)
  | n + 1, a, b, sa, sb, wa, wb, h => by
    have ih := hash_consistent n
    by_cases hr : rank a = rank b
    · cases hca : isComp a
      · have hcb : isComp b = false := by rw [← rank_isComp hr]; exact hca
        unfold scMain at h
        rw [sc_scalar scF32 a b hca hcb wa wb, (cmpN_eq_iff _ _).2 hr, thn_eq, cmp3_eq_iff] at h
        exact scalar_hash a b hca hr h wa wb
      · have pa : Pn (n + 1) a := ⟨sa, wa⟩
        have pb : Pn (n + 1) b := ⟨sb, wb⟩
        cases a with
        | list xs =>
          obtain ⟨ys, rfl⟩ := rank_list hr
          unfold scMain at h; simp only [sc, scList_eq] at h
          have hx := Pn_list pa; have hy := Pn_list pb
          simp only [hwords]
          rw [hwordsList_of_lexA xs ys (fun x hxm y hym e =>
            ih x y (hx x hxm).1 (hy y hym).1 (hx x hxm).2 (hy y hym).2 e) h]
          rw [(lexA_eq _ xs ys h).1]
        | map ka va =>
          obtain ⟨kb, vb, rfl⟩ := rank_map hr
          unfold scMain at h; simp only [sc, thn_eq_eq] at h
          obtain ⟨h1, h2, h3⟩ := h
          have hk := lexA_cmpN_eq ka kb h2
          subst hk
          obtain ⟨la, hx⟩ := Pn_map pa; obtain ⟨lb, hy⟩ := Pn_map pb
          rw [scVals_eq, zipNe_eq_lexA _ _ _ (by omega)] at h3
          simp only [hwords]
          rw [hwordsMap_of_lexA ka va vb (fun x hxm y hym e =>
            ih x y (hx x hxm).1 (hy y hym).1 (hx x hxm).2 (hy y hym).2 e) h3]
        | vecf32 xs =>
          obtain ⟨ys, rfl⟩ := rank_vec hr
          unfold scMain at h; simp only [sc, thn_eq_eq] at h
          have hl := (cmpN_eq_iff _ _).1 h.1
          simp only [hwords, vec_map_of_zipNe xs ys hl h.2]
        | point x y =>
          obtain ⟨x', y', rfl⟩ := rank_point hr
          unfold scMain at h; simp only [sc, thn_eq_eq] at h
          simp only [hwords, f32HashBits_of_eq _ _ h.1, f32HashBits_of_eq _ _ h.2]
        | _ => simp [isComp] at hca
    · unfold scMain at h
      rw [sc_rank_ne scF32 a b hr] at h
      exact absurd ((cmpN_eq_iff _ _).1 h) hr

/-- **On main, `sort_cmp`-equal ⇒ `hash`-equal** for every well-formed value: the
`HashMap<GroupKey, _>` contract (`GroupKey::eq` is `sort_cmp`, aggregate.rs:94). -/
theorem main_hash_consistent (a b : V) (wa : wf a = true) (wb : wf b = true)
    (h : scMain a b = .eq) : hwords a = hwords b :=
  hash_consistent (vsize a + vsize b) a b (by omega) (by omega) wa wb h

/-- W9-hash-1 fixed: every Float NaN is one key and one hash word (hash-only DISTINCT
keeps one), and so are vector NaN components. -/
theorem main_nans_one_key (s s' : Bool) (p p' : Nat) :
    scMain (.flt (.nan s p)) (.flt (.nan s' p')) = .eq ∧
    hwords (.flt (.nan s p)) = hwords (.flt (.nan s' p')) ∧
    rowHash [.flt (.nan s p)] = rowHash [.flt (.nan s' p')] ∧
    hwords (.vecf32 [.nan s p]) = hwords (.vecf32 [.nan s' p']) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [scMain, sc, scF64, fpcmp, F.nkey, isNaN, cmpB, cmpN, cmpI]
  · simp [hwords, floatWord, diffZero, F.nkey]
  · simp [rowHash, hwords, floatWord, diffZero, F.nkey]
  · simp [hwords, f32HashBits, F.nkey]

/-- Float `-0.0`/`0.0` are one key; vector/point signed zeros are not (as in C). -/
theorem signed_zeros_apart :
    scMain (.flt .negZero) (.flt (.num 0)) = .eq ∧
    hwords (.flt .negZero) = hwords (.flt (.num 0)) ∧
    scMain (.vecf32 [.negZero]) (.vecf32 [.num 0]) = .lt ∧
    scMain (.point .negZero (.num 0)) (.point (.num 0) (.num 0)) = .lt := by decide

/-! ## The sorts that use it -/

/-- `OrderedKey::cmp_key` (ops/sort.rs:76): `sort_cmp`, reversed for DESC. -/
def cmpKey (desc : Bool) (a b : V) : Ordering :=
  if desc then (scMain a b).swap else scMain a b

/-- One ORDER BY column (either direction) is a total preorder on well-formed values. -/
theorem cmpKey_total (desc : Bool) (a b d : V) (wa : wf a = true) (wb : wf b = true)
    (wd : wf d = true) :
    cmpKey desc a a = .eq ∧ cmpKey desc b a = (cmpKey desc a b).swap ∧
    (cmpKey desc a b ≠ .gt → cmpKey desc b d ≠ .gt → cmpKey desc a d ≠ .gt) := by
  have sw : ∀ x y, wf x = true → wf y = true → (scMain x y).swap = scMain y x :=
    fun x y wx wy => ((scMain_total x y y wx wy wy).2.1).symm
  have r1 := (scMain_total a b d wa wb wd).1
  cases desc
  · exact scMain_total a b d wa wb wd
  · simp only [cmpKey, if_true]
    refine ⟨by rw [r1]; rfl, by rw [sw b a wb wa]; cases scMain a b <;> rfl, ?_⟩
    rw [sw a b wa wb, sw b d wb wd, sw a d wa wd]
    intro h1 h2
    exact (scMain_total d b a wd wb wa).2.2 h2 h1

/-- A typed sort-key column (batch.rs `Column`): the lanes `compare_at` dispatches on. -/
inductive Col where
  | ints (v : List Int)
  | floats (v : List F)
  | values (v : List V)
  | nodeIds (v : List Nat)
  | unbound

/-- `Column::get` as a `Value` (Unbound reads Null). -/
def Col.get : Col → Nat → V
  | .ints v, i => .int (v.getD i 0)
  | .floats v, i => .flt (v.getD i (.num 0))
  | .values v, i => v.getD i .null
  | .nodeIds v, i => .node (v.getD i 0)
  | .unbound, _ => .null

/-- `Column::compare_at` (batch.rs:400): raw lanes, else `sort_cmp` on `get`. -/
def compareAt : Col → Nat → Nat → Ordering
  | .ints v, a, b => cmpI (v.getD a 0) (v.getD b 0)
  | .floats v, a, b => scF64 (v.getD a (.num 0)) (v.getD b (.num 0))
  | .values v, a, b => scMain (v.getD a .null) (v.getD b .null)
  | c, a, b => scMain (c.get a) (c.get b)

/-- Every lane of `compare_at` is `sort_cmp` on the materialised values (so the typed
fast paths sort exactly like the `Values` path, and like `list.sort`). -/
theorem compareAt_eq_scMain (c : Col) (a b : Nat) : compareAt c a b = scMain (c.get a) (c.get b) := by
  cases c <;> rfl

/-- `Batch::compare_rows_at` (batch.rs:1282): Unbound columns tie, else `sort_cmp`. -/
def compareRowsAt (c : Col) (a b : Nat) : Ordering :=
  match c with
  | .unbound => .eq
  | c => scMain (c.get a) (c.get b)

theorem compareRowsAt_eq_scMain (c : Col) (a b : Nat) :
    compareRowsAt c a b = scMain (c.get a) (c.get b) := by
  cases c <;> simp [compareRowsAt, Col.get, scMain, sc, order, V.tag, orderTag, cmpN, cmpI]

/-- `list.sort`'s comparator (functions/list.rs:210) is `sort_cmp`; with `scMain_total`,
`sort_by` sees a total order (W2-strlist-1 fixed by #2912). Its insertion-sort model: -/
def insertBy (c : V → V → Ordering) (x : V) : List V → List V
  | [] => [x]
  | y :: ys => if c x y = .gt then y :: insertBy c x ys else x :: y :: ys

def listSort (xs : List V) : List V := xs.foldr (insertBy scMain) []

theorem insertBy_perm (c : V → V → Ordering) (x : V) :
    ∀ ys, (insertBy c x ys).Perm (x :: ys)
  | [] => List.Perm.refl _
  | y :: ys => by
    unfold insertBy
    split
    · exact ((insertBy_perm c x ys).cons y).trans (List.Perm.swap x y ys)
    · exact List.Perm.refl _

/-- `list.sort` returns a permutation of its input. -/
theorem listSort_perm (xs : List V) : (listSort xs).Perm xs := by
  induction xs with
  | nil => exact List.Perm.refl _
  | cons x xs ih =>
    simp only [listSort, List.foldr_cons]
    exact (insertBy_perm scMain x _).trans (ih.cons x)

#print axioms scMain_total
#print axioms main_hash_consistent
#print axioms main_nans_one_key
#print axioms signed_zeros_apart
#print axioms cmpKey_total
#print axioms compareAt_eq_scMain
#print axioms compareRowsAt_eq_scMain
#print axioms listSort_perm

end ValueOrder.Main
