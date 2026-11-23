/-
CartesianProduct and ValueHashJoin (graph/src/runtime/ops/cartesian_product.rs,
value_hash_join.rs).
-/
namespace CartVhj

/-! ## The odometer (cartesian_product.rs) -/

/-- Rightmost position first: increment; if it overflows, reset to 0 and carry left.
`true` = a new combination was reached; `false` = wrapped (all zeros again). -/
def adv : List Nat → List Nat → List Nat × Bool
  | c :: cs, l :: ls =>
    match adv cs ls with
    | (cs', true) => (c :: cs', true)
    | (cs', false) => if c + 1 < l then ((c + 1) :: cs', true) else (0 :: cs', false)
  | _, _ => ([], false)

/-- All combinations, leftmost slowest (`rows` of the product, in emission order). -/
def allTuples : List Nat → List (List Nat)
  | [] => [[]]
  | l :: ls => (List.range l).flatMap fun i => (allTuples ls).map (i :: ·)

/-- The cursors visited from `c`: `c`, then each successful `advance`. -/
def orbit (ls : List Nat) : Nat → List Nat → List (List Nat)
  | 0, _ => []
  | n + 1, c => c :: match adv c ls with
    | (c', true) => orbit ls n c'
    | (_, false) => []

def zeros (ls : List Nat) : List Nat := ls.map fun _ => 0

theorem adv_wrap (ls : List Nat) : ∀ c, (adv c ls).2 = false → (adv c ls).1 = zeros ls ∨ c.length ≠ ls.length := by
  induction ls with
  | nil => intro c _; cases c <;> simp [adv, zeros]
  | cons l ls ih =>
    intro c h
    cases c with
    | nil => right; simp
    | cons c cs =>
      simp only [adv] at h ⊢
      cases ha : adv cs ls with
      | mk cs' b =>
        cases b
        · rw [ha] at h; simp only at h
          by_cases hl : c + 1 < l
          · simp [hl] at h
          · simp only [hl, ite_false]
            rcases ih cs (by rw [ha]) with h1 | h1
            · rw [ha] at h1; left; simp only at h1; simp [zeros, h1]
            · right; simp; exact h1
        · rw [ha] at h; simp at h

theorem sum_map_const : ∀ (n c : Nat), ((List.range n).map fun _ => c).sum = n * c := by
  intro n c; induction n with
  | zero => simp
  | succ n ihn => rw [List.range_succ, List.map_append, List.sum_append, ihn]; simp [Nat.succ_mul]

/-- **The product's row count** is the product of the branch lengths (any empty branch ⇒
none: `is_empty`, :94). -/
theorem allTuples_length (ls : List Nat) : (allTuples ls).length = ls.foldr (· * ·) 1 := by
  induction ls with
  | nil => rfl
  | cons l ls ih =>
    have hs := sum_map_const
    simp only [allTuples, List.length_flatMap, List.length_map, ih, List.foldr_cons]
    rw [hs]

theorem allTuples_mem (ls c : List Nat) : c ∈ allTuples ls ↔ c.length = ls.length ∧ ∀ i (h1 : i < c.length) (h2 : i < ls.length), c[i] < ls[i] := by
  induction ls generalizing c with
  | nil => cases c <;> simp [allTuples]
  | cons l ls ih =>
    cases c with
    | nil => simp [allTuples]
    | cons x xs =>
      simp only [allTuples, List.mem_flatMap, List.mem_range, List.mem_map, List.cons.injEq, List.length_cons,
        Nat.add_right_cancel_iff]
      constructor
      · rintro ⟨i, hi, t, ht, rfl, rfl⟩
        obtain ⟨hl, hg⟩ := (ih t).mp ht
        refine ⟨hl, fun k h1 h2 => ?_⟩
        cases k with
        | zero => simpa using hi
        | succ k => simpa using hg k (by simpa using h1) (by simpa using h2)
      · rintro ⟨hl, hg⟩
        refine ⟨x, by simpa using hg 0 (by simp) (by simp), xs, (ih xs).mpr ⟨hl, fun k h1 h2 => ?_⟩, rfl, rfl⟩
        have := hg (k + 1) (by simpa using h1) (by simpa using h2)
        simpa using this

/-- `advance` from a valid cursor stays valid when it succeeds. -/
theorem adv_valid (ls : List Nat) : ∀ c, c ∈ allTuples ls → (adv c ls).2 = true → (adv c ls).1 ∈ allTuples ls := by
  induction ls with
  | nil => intro c _ h; cases c <;> simp [adv] at h
  | cons l ls ih =>
    intro c hc h
    cases c with
    | nil => simp [allTuples] at hc
    | cons x xs =>
      rw [allTuples_mem] at hc
      obtain ⟨hl, hg⟩ := hc
      have hx : x < l := by have := hg 0 (by simp) (by simp); simpa using this
      have hxs : xs ∈ allTuples ls := (allTuples_mem ls xs).mpr ⟨by simpa using hl, fun k h1 h2 => by
        have := hg (k + 1) (by simpa using h1) (by simpa using h2); simpa using this⟩
      simp only [adv] at h ⊢
      cases ha : adv xs ls with
      | mk t' b =>
        rw [ha] at h
        cases b
        · simp only at h ⊢
          by_cases hl1 : x + 1 < l
          · simp only [hl1, ite_true]
            have hw := adv_wrap ls xs (by rw [ha])
            rw [ha] at hw
            have hz : t' = zeros ls := by
              rcases hw with hw | hw
              · exact hw
              · exact absurd ((allTuples_mem ls xs).mp hxs).1 hw
            subst hz
            rw [allTuples_mem]
            refine ⟨by simp [zeros], fun k h1 h2 => ?_⟩
            cases k with
            | zero => simpa using hl1
            | succ k =>
              have := ((allTuples_mem ls xs).mp hxs).2 k (by rw [((allTuples_mem ls xs).mp hxs).1]; simpa using h2) (by simpa using h2)
              simp [zeros]; omega
          · simp [hl1] at h
        · simp only
          have := ih xs hxs (by rw [ha])
          rw [ha] at this
          simp only at this
          rw [allTuples_mem] at this ⊢
          refine ⟨by simp [this.1], fun k h1 h2 => ?_⟩
          cases k with
          | zero => simpa using hx
          | succ k => have := this.2 k (by simpa using h1) (by simpa using h2); simpa using this

/-- Every cursor the odometer visits from the zero cursor is a valid combination. -/
theorem orbit_valid (ls : List Nat) (hpos : ∀ l ∈ ls, 0 < l) : ∀ n c, c ∈ allTuples ls → ∀ d ∈ orbit ls n c, d ∈ allTuples ls := by
  intro n
  induction n with
  | zero => intro c _ d hd; simp [orbit] at hd
  | succ n ih =>
    intro c hc d hd
    simp only [orbit, List.mem_cons] at hd
    rcases hd with rfl | hd
    · exact hc
    · cases ha : adv c ls with
      | mk c' b =>
        rw [ha] at hd
        cases b
        · simp at hd
        · exact ih c' (by have := adv_valid ls c hc (by rw [ha]); rw [ha] at this; exact this) d hd

theorem zeros_mem (ls : List Nat) (hpos : ∀ l ∈ ls, 0 < l) : zeros ls ∈ allTuples ls := by
  rw [allTuples_mem]
  refine ⟨by simp [zeros], fun i h1 h2 => ?_⟩
  simp [zeros]; exact hpos _ (List.getElem_mem h2)

/-- `RightSide::new` (:80): plan from the branches, cursor all zeros, lengths recorded. -/
structure RS where
  lens : List Nat
  cursor : List Nat

def rsNew (lens : List Nat) : RS := ⟨lens, zeros lens⟩
def rsEmpty (r : RS) : Bool := r.lens.any (· == 0)
def rewind (r : RS) : RS := { r with cursor := zeros r.lens }

theorem rs_spec (lens : List Nat) (r : RS) :
    (rsNew lens).cursor = zeros lens ∧ (rewind r).cursor = zeros r.lens ∧
    (rsEmpty r = true ↔ ∃ l ∈ r.lens, l = 0) ∧
    (rsEmpty r = false → zeros r.lens ∈ allTuples r.lens) := by
  refine ⟨rfl, rfl, by simp [rsEmpty], fun h => zeros_mem _ ?_⟩
  intro l hl
  simp only [rsEmpty, List.any_eq_false, beq_iff_eq] at h
  exact Nat.pos_of_ne_zero (h l hl)

/-- `CartesianProductOp::new` (:123): nothing materialised, no left batch. -/
structure CpSt (C R B : Type) where
  child : C
  rights : List R
  right : Option RS
  leftBatch : Option B
  leftPos : Nat

def CpSt.new {C R B : Type} (child : C) (rights : List R) : CpSt C R B := ⟨child, rights, none, none, 0⟩

theorem cpNew_spec {C R B : Type} (c : C) (rs : List R) :
    (CpSt.new c rs : CpSt C R B).right = none ∧ (CpSt.new c rs : CpSt C R B).leftBatch = none ∧
    (CpSt.new c rs : CpSt C R B).leftPos = 0 := ⟨rfl, rfl, rfl⟩

/-- `materialize_right` (:144): drain each right child fully, concatenating its batches; the
first error aborts. A branch = its stream of row lists. -/
def materialize {R : Type} : List (List (Except String (List R))) → Except String (List (List R))
  | [] => .ok []
  | br :: brs => do
    let rows ← br.foldlM (fun acc b => b.map (acc ++ ·)) []
    let rest ← materialize brs
    pure (rows :: rest)

theorem materialize_ok {R : Type} (brs : List (List (List R))) :
    materialize (brs.map fun br => br.map .ok) = .ok (brs.map List.flatten) := by
  induction brs with
  | nil => rfl
  | cons br brs ih =>
    have : ∀ (acc : List R) (bs : List (List R)),
        (bs.map (Except.ok (ε := String))).foldlM (fun acc b => b.map (acc ++ ·)) acc = .ok (acc ++ bs.flatten) := by
      intro acc bs
      induction bs generalizing acc with
      | nil => simp [pure, Except.pure]
      | cons b bs ihb =>
        rw [List.map_cons, List.foldlM_cons]
        exact (ihb (acc ++ b)).trans (by simp)
    simp only [List.map_cons, materialize, bind, Except.bind]
    rw [this, ih]; simp [pure, Except.pure]

/-- `next` (:162): every left row (in order) paired with every right combination in odometer
order, the merge taking each slot from the last right that binds it. -/
def cpRows {L R O : Type} (merge : L → List R → O) (lefts : List L) (branches : List (List R))
    (combos : List (List Nat)) (pick : List (List R) → List Nat → List R) : List O :=
  lefts.flatMap fun l => combos.map fun c => merge l (pick branches c)

theorem cpRows_length {L R O : Type} (merge : L → List R → O) (lefts : List L) (brs : List (List R)) :
    (cpRows merge lefts brs (allTuples (brs.map List.length)) (fun _ _ => [])).length =
      lefts.length * (brs.map List.length).foldr (· * ·) 1 := by
  simp only [cpRows, List.length_flatMap, List.length_map, allTuples_length]
  have := sum_map_const lefts.length ((brs.map List.length).foldr (· * ·) 1)
  rw [← this]
  congr 1
  apply List.ext_getElem <;> simp

/-! ## ValueHashJoin -/

/-- Join keys: what `compare_value` and the integer fast path need. -/
inductive K where
  | null
  | int (n : Int)
  | float (f : Int) (exact : Bool)   -- `exact` = the float is an integer value `f`
  | other (tag : Nat)                -- strings, lists, maps, paths, … (compared by `cv`)
deriving DecidableEq

/-- value_hash_join.rs `INT_KEY_LIMIT`: `1 << f64::MANTISSA_DIGITS` = 2^53. -/
def lim : Nat := 2 ^ 53

/-- value_hash_join.rs `key_as_i64`: an int, or a whole float, of magnitude below 2^53; anything
else (larger magnitudes included) is `None`. -/
def keyAsI64 : K → Option Int
  | .int n => if n.natAbs < lim then some n else none
  | .float f true => if f.natAbs < lim then some f else none
  | _ => none

theorem keyAsI64_spec (n f : Int) :
    (n.natAbs < lim → keyAsI64 (.int n) = some n) ∧ (lim ≤ n.natAbs → keyAsI64 (.int n) = none) ∧
    (f.natAbs < lim → keyAsI64 (.float f true) = some f) ∧
    (lim ≤ f.natAbs → keyAsI64 (.float f true) = none) ∧
    keyAsI64 (.float f false) = none ∧ keyAsI64 .null = none := by
  refine ⟨fun h => by simp [keyAsI64, h], fun h => ?_, fun h => by simp [keyAsI64, h], fun h => ?_, rfl, rfl⟩
  · simp [keyAsI64]; omega
  · simp [keyAsI64]; omega

/-! ### `i as f64` and Cypher `=` on numeric keys -/

/-- The binary exponent `e` with `m / 2^e < 2^53`, the smallest such (fuel 11 covers every `i64`
magnitude, at most 2^63). -/
def expOf : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, m => if m < lim then 0 else expOf fuel (m / 2) + 1

/-- Round `m` to a multiple of `2^e`: to nearest, ties to even. -/
def rneAt (m e : Nat) : Nat :=
  (if 2 ^ e / 2 < m % 2 ^ e ∨ (m % 2 ^ e = 2 ^ e / 2 ∧ (m / 2 ^ e) % 2 = 1)
    then m / 2 ^ e + 1 else m / 2 ^ e) * 2 ^ e

/-- An `i64` magnitude as an `f64`: exact up to 2^53, rounded to 53 significant bits above. -/
def rneNat (m : Nat) : Nat := if m ≤ lim then m else rneAt m (expOf 11 m)

/-- `i as f64`, as the integer it is. -/
def toF64 (n : Int) : Int := if n < 0 then -((rneNat n.natAbs : Nat) : Int) else ((rneNat n.natAbs : Nat) : Int)

theorem expOf_le : ∀ fuel m, expOf fuel m ≤ fuel
  | 0, _ => Nat.le_refl _
  | fuel + 1, m => by
      unfold expOf; split
      · omega
      · have := expOf_le fuel (m / 2); omega

theorem toF64_natAbs (n : Int) : (toF64 n).natAbs = rneNat n.natAbs := by
  unfold toF64; split <;> simp

theorem toF64_small (n : Int) (h : n.natAbs ≤ lim) : toF64 n = n := by
  have e : rneNat n.natAbs = n.natAbs := by simp [rneNat, h]
  unfold toF64; rw [e]; split <;> omega

/-- Above 2^53 the rounded value stays at least 2^53. -/
theorem toF64_big (n : Int) (h : lim < n.natAbs) : lim ≤ (toF64 n).natAbs := by
  rw [toF64_natAbs]
  have hn : ¬ n.natAbs ≤ lim := by omega
  simp only [rneNat, hn, ite_false, rneAt]
  have he : expOf 11 n.natAbs ≤ 53 := by have := expOf_le 11 n.natAbs; omega
  generalize expOf 11 n.natAbs = e at he ⊢
  generalize n.natAbs = m at h ⊢
  have hp : 0 < 2 ^ e := Nat.pow_pos (by decide)
  have h1 : 2 ^ (53 - e) ≤ m / 2 ^ e := by
    rw [← Nat.pow_div he (by decide)]
    exact Nat.div_le_div_right (by unfold lim at h; omega)
  have h2 : lim = 2 ^ (53 - e) * 2 ^ e := by
    unfold lim; rw [← Nat.pow_add]; congr 1; omega
  have h3 : m / 2 ^ e * 2 ^ e ≤ (if 2 ^ e / 2 < m % 2 ^ e ∨ (m % 2 ^ e = 2 ^ e / 2 ∧ (m / 2 ^ e) % 2 = 1)
      then m / 2 ^ e + 1 else m / 2 ^ e) * 2 ^ e := by
    split
    · exact Nat.mul_le_mul_right _ (Nat.le_succ _)
    · exact Nat.le_refl _
  exact Nat.le_trans (h2 ▸ Nat.mul_le_mul_right _ h1) h3

theorem toF64_ge (n : Int) (h : lim ≤ n.natAbs) : lim ≤ (toF64 n).natAbs := by
  by_cases e : n.natAbs ≤ lim
  · rw [toF64_small n e]; exact h
  · exact toF64_big n (by omega)

/-- Below 2^53, `as f64` hits a given small integer only from that integer itself. -/
theorem toF64_eq_small (n m : Int) (hn : n.natAbs < lim) : toF64 m = n ↔ m = n := by
  by_cases hm : m.natAbs ≤ lim
  · rw [toF64_small m hm]
  · have hb := toF64_big m (by omega)
    constructor
    · intro e; rw [e] at hb; omega
    · intro e; subst e; omega

/-- Cypher `=` on numeric keys, as the `Filter` evaluates it (`compare_value`: Int vs Float
through `as f64`). Null, non-whole floats and other keys are not modelled here (`false`). -/
def numEq : K → K → Bool
  | .int a, .int b => a == b
  | .int a, .float f true => toF64 a == f
  | .float f true, .int a => f == toF64 a
  | .float a true, .float b true => a == b
  | _, _ => false

/-- A conclusive `compare_value` that is `Equal` exactly on `numEq`. -/
def numCv (a b : K) : Ordering × Bool := (if numEq a b then .eq else .lt, false)

/-- value_hash_join.rs `keys_match`: equal AND not disjoint-or-null (`cv` = `compare_value`). -/
def keysMatch (cv : K → K → Ordering × Bool) (a b : K) : Bool := (cv a b).1 == .eq && !(cv a b).2

theorem keysMatch_spec (cv : K → K → Ordering × Bool) (a b : K) :
    keysMatch cv a b = true ↔ (cv a b).1 = .eq ∧ (cv a b).2 = false := by
  simp [keysMatch]

theorem keysMatch_numCv (a b : K) : keysMatch numCv a b = numEq a b := by
  unfold keysMatch numCv; cases numEq a b <;> rfl

/-- Every key the integer table holds is below 2^53. -/
theorem keyAsI64_bound (k : K) (n : Int) (h : keyAsI64 k = some n) : n.natAbs < lim := by
  unfold keyAsI64 at h
  split at h
  · split at h <;> simp at h; subst h; assumption
  · split at h <;> simp at h; subst h; assumption
  · simp at h

/-- **Integer table is exact** (`key_as_i64`): on two keys it maps, Cypher `=` is equality of the
`i64`s. -/
theorem keyAsI64_exact (a b : K) (m m' : Int) (ha : keyAsI64 a = some m) (hb : keyAsI64 b = some m') :
    numEq a b = true ↔ m = m' := by
  have hm := keyAsI64_bound a m ha
  have hm' := keyAsI64_bound b m' hb
  cases a with
  | int x =>
    simp [keyAsI64] at ha; obtain ⟨_, rfl⟩ := ha
    cases b with
    | int y => simp [keyAsI64] at hb; obtain ⟨_, rfl⟩ := hb; simp [numEq]
    | float g ex =>
      cases ex <;> simp [keyAsI64] at hb
      obtain ⟨_, rfl⟩ := hb
      simp [numEq, toF64_small x (by omega)]
    | _ => simp [keyAsI64] at hb
  | float f ex =>
    cases ex <;> simp [keyAsI64] at ha
    obtain ⟨_, rfl⟩ := ha
    cases b with
    | int y =>
      simp [keyAsI64] at hb; obtain ⟨_, rfl⟩ := hb
      simp [numEq, toF64_small y (by omega)]
    | float g ex =>
      cases ex <;> simp [keyAsI64] at hb
      obtain ⟨_, rfl⟩ := hb; simp [numEq]
    | _ => simp [keyAsI64] at hb
  | _ => simp [keyAsI64] at ha

/-- **A key the integer table cannot hold equals none of its keys**: a probe key `key_as_i64`
rejects (2^53 or more, non-whole, …) is `=` to no key below 2^53, so the integer table rightly
finds nothing for it. -/
theorem keyAsI64_none_sep (a p : K) (m : Int) (ha : keyAsI64 a = some m) (hp : keyAsI64 p = none) :
    numEq a p = false := by
  have hm := keyAsI64_bound a m ha
  cases a with
  | int x =>
    simp [keyAsI64] at ha; obtain ⟨_, rfl⟩ := ha
    cases p with
    | int y =>
      simp [keyAsI64] at hp; simp only [numEq, beq_eq_false_iff_ne, ne_eq]; omega
    | float g ex =>
      cases ex
      · rfl
      · simp [keyAsI64] at hp
        simp only [numEq, toF64_small x (by omega), beq_eq_false_iff_ne, ne_eq]; omega
    | _ => rfl
  | float f ex =>
    cases ex <;> simp [keyAsI64] at ha
    obtain ⟨_, rfl⟩ := ha
    cases p with
    | int y =>
      simp [keyAsI64] at hp
      have := toF64_ge y (by omega)
      simp only [numEq, beq_eq_false_iff_ne, ne_eq]; intro e; rw [← e] at this; omega
    | float g ex =>
      cases ex
      · rfl
      · simp [keyAsI64] at hp; simp only [numEq, beq_eq_false_iff_ne, ne_eq]; omega
    | _ => rfl
  | _ => simp [keyAsI64] at ha

/-- `=` is not transitive across `Int` and `Float` above 2^53: `2^53 = 2^53.0 = 2^53 + 1` while
`2^53 ≠ 2^53 + 1`. So `keys_match` alone may not group build keys (what the code before #3034 did). -/
theorem numEq_not_transitive :
    numEq (.int (2 ^ 53)) (.float (2 ^ 53) true) = true ∧
    numEq (.float (2 ^ 53) true) (.int (2 ^ 53 + 1)) = true ∧
    numEq (.int (2 ^ 53)) (.int (2 ^ 53 + 1)) = false := by decide

/-! ### `JoinKey::hash`, `f32_join_bits`, `hash_value` -/

/-- value_hash_join.rs `JoinKey::hash` on scalar keys, as what it feeds the hasher: an `Int` above
2^53 hashes as `x as f64 as i64` (the whole `Float` it equals); a whole `Float` hashes as its `i64`
(`Value::hash`, the same tag 2 as `Int`); so both meet in `.int`. Other keys hash as `Value::hash`
does. (The List/Path/Map arms apply the same rule element-wise, VecF32/Point through
`f32_join_bits`; containers are `K.other` here.) -/
def joinNorm : K → K
  | .int x => if lim < x.natAbs then .int (toF64 x) else .int x
  | .float f true => .int f
  | k => k

/-- value_hash_join.rs `hash_value`: the seeded hasher over `JoinKey` — any function of what it feeds. -/
def joinHash (enc : K → Nat) (k : K) : Nat := enc (joinNorm k)

/-- **Keys `=` equates hash alike** (the #3034 fix: before it `Int(2^53+1)` and `Float(2^53)` hashed
apart, so a probe could not find its match). -/
theorem joinHash_numEq (enc : K → Nat) (a b : K) (e : numEq a b = true) : joinHash enc a = joinHash enc b := by
  unfold joinHash; congr 1
  cases a with
  | int x =>
    cases b with
    | int y => simp [numEq] at e; subst e; rfl
    | float g ex =>
      cases ex
      · simp [numEq] at e
      · simp [numEq] at e; subst e
        simp only [joinNorm]; split
        · rfl
        · rw [toF64_small x (by omega)]
    | _ => simp [numEq] at e
  | float f ex =>
    cases ex
    · cases b <;> simp [numEq] at e
    cases b with
    | int y =>
      simp [numEq] at e; subst e
      simp only [joinNorm]; split
      · rfl
      · rw [toF64_small y (by omega)]
    | float g ex =>
      cases ex
      · simp [numEq] at e
      · simp [numEq] at e; subst e; rfl
    | _ => simp [numEq] at e
  | _ => cases b <;> simp [numEq] at e

/-- The model's hash is a function of the key. -/
def hashValue (h : K → Nat) (k : K) : Nat := h k

theorem hashValue_det (h : K → Nat) (a b : K) (e : a = b) : hashValue h a = hashValue h b := by rw [e]

/-- An `f32` component of a point or a vector: NaN (any payload), ±0, or another finite/infinite
value by its bits (never the bits of a zero). -/
inductive F32 where
  | nan (payload : Nat)
  | zero (neg : Bool)
  | bits (b : Nat)

/-- `f32::NAN.to_bits()`. -/
def nanBits : Nat := 0x7fc00000

/-- value_hash_join.rs `f32_join_bits`: one value for every NaN, one for both zeros. -/
def f32JoinBits : F32 → Nat
  | .nan _ => nanBits
  | .zero _ => 0
  | .bits b => b

theorem f32JoinBits_spec (p q : Nat) (b : Nat) :
    f32JoinBits (.zero true) = f32JoinBits (.zero false) ∧ f32JoinBits (.nan p) = f32JoinBits (.nan q) ∧
    f32JoinBits (.bits b) = b := ⟨rfl, rfl, rfl⟩

/-! ### Build grouping and probe -/

/-- value_hash_join.rs `same_build_key`: two build keys share an entry only when every probe key
matching one matches the other. An `Int` and a `Float` share one only below 2^53. (`sbo` = the
element-wise List/Path/Map arms, on keys modelled as `K.other`.) -/
def sameBuildKey (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) : K → K → Bool
  | .int x, .int y => x == y
  | .int n, .float f e => decide (n.natAbs < lim) && keysMatch cv (.int n) (.float f e)
  | .float f e, .int n => decide (n.natAbs < lim) && keysMatch cv (.float f e) (.int n)
  | .other a, .other b => sbo a b
  | a, b => keysMatch cv a b

theorem beq_toF64_small (x y : Int) (hx : x.natAbs < lim) : (x == toF64 y) = (x == y) := by
  have := toF64_eq_small x y hx
  rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]
  exact ⟨fun h => (this.1 h.symm).symm, fun h => (this.2 h.symm).symm⟩

/-- **Grouping is sound**: build keys sharing an entry are `=` to exactly the same probe keys. -/
theorem sameBuildKey_sound (sbo : Nat → Nat → Bool) (a b p : K) (h : sameBuildKey numCv sbo a b = true) :
    numEq a p = numEq b p := by
  cases a with
  | int x =>
    cases b with
    | int y => simp [sameBuildKey] at h; subst h; rfl
    | float g ex =>
      cases ex
      · simp [sameBuildKey, keysMatch_numCv, numEq] at h
      · simp [sameBuildKey, keysMatch_numCv, numEq] at h
        obtain ⟨hx, e⟩ := h
        rw [toF64_small x (by omega)] at e; subst e
        cases p with
        | int y => simp only [numEq]; rw [beq_toF64_small x y hx]
        | float g ex => cases ex <;> simp [numEq, toF64_small x (by omega)]
        | _ => rfl
    | null => simp [sameBuildKey, keysMatch_numCv, numEq] at h
    | other t => simp [sameBuildKey, keysMatch_numCv, numEq] at h
  | float f ex =>
    cases ex
    · cases b <;> simp [sameBuildKey, keysMatch_numCv, numEq] at h
    cases b with
    | int y =>
      simp [sameBuildKey, keysMatch_numCv, numEq] at h
      obtain ⟨hy, e⟩ := h
      rw [toF64_small y (by omega)] at e; subst e
      cases p with
      | int z => simp only [numEq]; rw [beq_toF64_small f z hy]
      | float g ex => cases ex <;> simp [numEq, toF64_small f (by omega)]
      | _ => rfl
    | float g ex =>
      cases ex
      · simp [sameBuildKey, keysMatch_numCv, numEq] at h
      · simp [sameBuildKey, keysMatch_numCv, numEq] at h; subst h; rfl
    | _ => simp [sameBuildKey, keysMatch_numCv, numEq] at h
  | null => cases b <;> simp [sameBuildKey, keysMatch_numCv, numEq] at h
  | other t =>
    cases b with
    | other u => cases p <;> rfl
    | _ => simp [sameBuildKey, keysMatch_numCv, numEq] at h

/-- `2^53` and `2^53.0` are `keys_match`-equal but never share a build entry. -/
theorem sameBuildKey_splits_2pow53 (sbo : Nat → Nat → Bool) :
    keysMatch numCv (.int (2 ^ 53)) (.float (2 ^ 53) true) = true ∧
    sameBuildKey numCv sbo (.int (2 ^ 53)) (.float (2 ^ 53) true) = false ∧
    sameBuildKey numCv sbo (.int 1) (.float 1 true) = true := by
  simp only [sameBuildKey, keysMatch_numCv]; decide

/-- The value table: hash ↦ bucket of `(key, refs)` entries. -/
abbrev VT := Nat → List (K × List Nat)

/-- Append `slot` to the first entry whose key groups with `key` (`g`), else open a new entry. -/
def addRef (g : K → K → Bool) (key : K) (slot : Nat) : List (K × List Nat) → List (K × List Nat)
  | [] => [(key, [slot])]
  | (k, rs) :: es => if g k key then (k, rs ++ [slot]) :: es else (k, rs) :: addRef g key slot es

/-- value_hash_join.rs `insert_value`: never-equal keys (null, NaN) are skipped; otherwise the slot
joins the bucket entry `same_build_key` groups it with. -/
def insertValue (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (neverEq : K → Bool)
    (t : VT) (key : K) (slot : Nat) : VT :=
  if neverEq key then t else fun hk => if hk = h key then addRef (sameBuildKey cv sbo) key slot (t hk) else t hk

/-- Insertion into a sorted list (build order). -/
def insSorted (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x ≤ y then x :: y :: ys else y :: insSorted x ys

/-- `sorted_by_key(|r| (r.batch, r.row))`: slots are numbered in build order. -/
def isort : List Nat → List Nat
  | [] => []
  | x :: xs => insSorted x (isort xs)

theorem insSorted_perm (x : Nat) : ∀ l, (insSorted x l).Perm (x :: l)
  | [] => List.Perm.refl _
  | y :: ys => by
      unfold insSorted; split
      · exact List.Perm.refl _
      · exact ((insSorted_perm x ys).cons y).trans (List.Perm.swap x y ys)

theorem isort_perm : ∀ l, (isort l).Perm l
  | [] => List.Perm.refl _
  | x :: xs => (insSorted_perm x _).trans ((isort_perm xs).cons x)

/-- Probe (`fill_matches`, value arm): every entry of the key's bucket that `keys_match`es; one
entry's refs as they are, several entries' refs merged back into build order. -/
def probe (cv : K → K → Ordering × Bool) (h : K → Nat) (t : VT) (key : K) : List Nat :=
  match (t (h key)).filter fun e => keysMatch cv e.1 key with
  | [] => []
  | [e] => e.2
  | es => isort (es.flatMap (·.2))

/-- **The probe returns the rows of every matching entry** (one entry or several), each once. -/
theorem probe_perm (cv : K → K → Ordering × Bool) (h : K → Nat) (t : VT) (key : K) :
    (probe cv h t key).Perm (((t (h key)).filter fun e => keysMatch cv e.1 key).flatMap (·.2)) := by
  unfold probe
  split
  · rename_i e; rw [e]; exact List.Perm.refl _
  · rename_i x e; rw [e]; simp
  · exact isort_perm _

/-- Inserting a never-equal key changes nothing; otherwise the slot is appended to the entry
`same_build_key` groups it with (or a new entry), in the key's hash bucket; other buckets untouched. -/
theorem insertValue_spec (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool)
    (t : VT) (key : K) (s : Nat) :
    (ne key = true → insertValue cv sbo h ne t key s = t) ∧
    (ne key = false → insertValue cv sbo h ne t key s (h key) = addRef (sameBuildKey cv sbo) key s (t (h key)) ∧
      ∀ hk, hk ≠ h key → insertValue cv sbo h ne t key s hk = t hk) := by
  refine ⟨fun hn => by simp [insertValue, hn], fun hn => ⟨by simp [insertValue, hn], fun hk hne => by simp [insertValue, hn, hne]⟩⟩

/-- A fresh key's probe after insertion finds exactly the new slot (given `keys_match` is
reflexive on joinable keys). -/
theorem probe_insert_fresh (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool)
    (t : VT) (key : K) (s : Nat)
    (hn : ne key = false) (hfresh : t (h key) = []) (hrefl : keysMatch cv key key = true) :
    probe cv h (insertValue cv sbo h ne t key s) key = [s] := by
  simp [probe, insertValue, hn, hfresh, addRef, hrefl]

/-- The integer fast path: `int ↦ refs`. -/
abbrev IT := Int → List Nat

def intInsert (t : IT) (n : Int) (s : Nat) : IT := fun m => if m = n then t m ++ [s] else t m

/-- `promote_int_table`: every integer key `n` (the drained domain `dom`) is re-keyed as
`Value::Int(n)` into its hash bucket with its refs intact. -/
def promote (h : K → Nat) (dom : List Int) (it : IT) : VT :=
  dom.foldl (fun t n => fun hk => if hk = h (.int n) then t hk ++ [(.int n, it n)] else t hk) (fun _ => [])

theorem promote_bucket (h : K → Nat) (dom : List Int) (it : IT) (hk : Nat) :
    promote h dom it hk = (dom.filter fun n => h (.int n) = hk).map fun n => (.int n, it n) := by
  unfold promote
  suffices H : ∀ (t : VT), dom.foldl (fun t n => fun hk => if hk = h (.int n) then t hk ++ [(.int n, it n)] else t hk) t hk
      = t hk ++ (dom.filter fun n => h (.int n) = hk).map fun n => (.int n, it n) by simpa using H (fun _ => [])
  induction dom with
  | nil => intro t; simp
  | cons n ns ih =>
    intro t
    simp only [List.foldl_cons]
    rw [ih]
    by_cases e : hk = h (.int n)
    · simp [e]
    · have : ¬ h (.int n) = hk := fun x => e x.symm
      simp [e, this]

inductive JT where
  | int (t : IT) (dom : List Int)
  | value (t : VT) (dom : List Nat)

/-- value_hash_join.rs `JoinHashTable::is_empty` (`dom` = the occupied keys / buckets). -/
def jtEmpty : JT → Bool
  | .int _ dom => dom.isEmpty
  | .value _ dom => dom.isEmpty

theorem jtEmpty_spec (t : IT) (v : VT) (d : List Int) (e : List Nat) :
    jtEmpty (.int t d) = d.isEmpty ∧ jtEmpty (.value v e) = e.isEmpty := ⟨rfl, rfl⟩

/-- One origin's table in `build_hash_table`, over the right rows' evaluated keys (`none` = null,
skipped): start on the integer table; the first key `key_as_i64` rejects (the `Column::Ints` fast
path takes exactly the ints below 2^53, the same ones) promotes it and switches to the value table
for the rest. -/
def build (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool) :
    JT → List (Option K × Nat) → JT
  | t, [] => t
  | t, (none, _) :: rest => build cv sbo h ne t rest
  | .int it dom, (some k, s) :: rest =>
    match keyAsI64 k with
    | some n => build cv sbo h ne (.int (intInsert it n s) (if dom.contains n then dom else dom ++ [n])) rest
    | none =>
      let v := insertValue cv sbo h ne (promote h dom it) k s
      build cv sbo h ne (.value v (dom.map (fun n => h (.int n)) ++ [h k])) rest
  | .value v dom, (some k, s) :: rest =>
    build cv sbo h ne (.value (insertValue cv sbo h ne v k s) (dom ++ [h k])) rest

/-- On an all-integer-valued build side the table is the integer table, and a probe for `n`
finds exactly the right rows whose key is `n`, in build order (nulls never join). -/
theorem build_int (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool) :
    ∀ (rows : List (Option K × Nat)) (it : IT) (dom : List Int),
      (∀ k s, (some k, s) ∈ rows → (keyAsI64 k).isSome) →
      ∃ it' dom', build cv sbo h ne (.int it dom) rows = .int it' dom' ∧
        ∀ n, it' n = it n ++ (rows.filterMap fun r => match r.1 with
          | some k => if keyAsI64 k = some n then some r.2 else none
          | none => none)
  | [], it, dom, _ => ⟨it, dom, rfl, fun n => by simp⟩
  | (none, s) :: rest, it, dom, hr => by
      obtain ⟨it', dom', h1, h2⟩ := build_int cv sbo h ne rest it dom (fun k s' hm => hr k s' (List.mem_cons_of_mem _ hm))
      exact ⟨it', dom', by simp [build, h1], fun n => by simp [h2]⟩
  | (some k, s) :: rest, it, dom, hr => by
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hr k s (List.mem_cons_self ..))
      obtain ⟨it', dom', h1, h2⟩ := build_int cv sbo h ne rest (intInsert it m s)
        (if dom.contains m then dom else dom ++ [m])
        (fun k s' hm' => hr k s' (List.mem_cons_of_mem _ hm'))
      refine ⟨it', dom', ?_, fun n => ?_⟩
      · simp only [build, hm]; exact h1
      rw [h2 n]
      by_cases e : n = m
      · subst e; simp [intInsert, hm]
      · have : ¬ m = n := fun x => e x.symm
        simp [intInsert, e, hm, this]

/-- `fill_matches`, once the origin's table is found: integer table probes `key_as_i64(key)` (a
key it rejects finds nothing, `keyAsI64_none_sep`); value table probes the key's bucket by
`keys_match`. -/
def fillMatches (cv : K → K → Ordering × Bool) (h : K → Nat) : JT → K → List Nat
  | .int it _, key => match keyAsI64 key with
    | some n => it n
    | none => []
  | .value v _, key => probe cv h v key

theorem fillMatches_int (cv : K → K → Ordering × Bool) (h : K → Nat) (it : IT) (d : List Int) (n : Int) (f : Int)
    (hn : n.natAbs < lim) (hf : f.natAbs < lim) :
    fillMatches cv h (.int it d) (.int n) = it n ∧ fillMatches cv h (.int it d) (.float f true) = it f ∧
    fillMatches cv h (.int it d) (.float f false) = [] := by
  simp [fillMatches, keyAsI64, hn, hf]

/-- A key `key_as_i64` rejects (e.g. 2^53 or more) finds nothing in the integer table. -/
theorem fillMatches_int_big (cv : K → K → Ordering × Bool) (h : K → Nat) (it : IT) (d : List Int) (k : K)
    (hk : keyAsI64 k = none) : fillMatches cv h (.int it d) k = [] := by
  simp [fillMatches, hk]


/-- `ValueHashJoinOp::new`: no table, no buffered rows, one shared table (`per_origin`
false until `set_argument_batch` sees a multi-row argument batch, batch.rs). -/
structure VhjSt where
  built : Option (Nat → JT)
  perOrigin : Bool
  leftPos : Nat
  matchPos : Nat
  found : List Nat

def VhjSt.new : VhjSt := ⟨none, false, 0, 0, []⟩

theorem vhjNew_spec : VhjSt.new.built = none ∧ VhjSt.new.leftPos = 0 ∧ VhjSt.new.found = [] ∧
    VhjSt.new.perOrigin = false := ⟨rfl, rfl, rfl, rfl⟩

/-- `next`, on one table: an empty table ends the stream at once; otherwise each left row (in
order) with a non-null key yields one merged row per matching right row, in the table's ref
order (batches of `BATCH_SIZE`, the row stream is what is modelled). -/
def vhjRows {L R O : Type} (cv : K → K → Ordering × Bool) (h : K → Nat) (t : JT) (rights : Nat → R)
    (merge : L → R → O) (lefts : List (L × Option K)) : List O :=
  if jtEmpty t then []
  else lefts.flatMap fun lk => match lk.2 with
    | none => []
    | some k => (fillMatches cv h t k).map fun s => merge lk.1 (rights s)

theorem vhjRows_spec {L R O : Type} (cv : K → K → Ordering × Bool) (h : K → Nat) (t : JT) (rights : Nat → R)
    (merge : L → R → O) (l : L) (k : K) (rest : List (L × Option K)) (hne : jtEmpty t = false) :
    vhjRows cv h t rights merge ((l, some k) :: rest) =
      (fillMatches cv h t k).map (fun s => merge l (rights s)) ++ vhjRows cv h t rights merge rest ∧
    vhjRows cv h t rights merge ((l, none) :: rest) = vhjRows cv h t rights merge rest := by
  simp [vhjRows, hne]

/-! ## Per-origin tables

`build_hash_table` keys each right row by `origin = if per_origin { origin_row } else { 0 }`
and runs the integer-then-value build above on that origin's own pair of tables; an origin
with no table is probed as nothing (`fill_matches`). An absent origin is modelled as the
fresh table `jtNew` (empty, probes to `[]`). -/

def jtNew : JT := .int (fun _ => []) []

theorem build_append (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool) :
    ∀ (xs ys : List (Option K × Nat)) (t : JT),
      build cv sbo h ne t (xs ++ ys) = build cv sbo h ne (build cv sbo h ne t xs) ys
  | [], _, _ => rfl
  | (none, _) :: xs, ys, t => by
      simp only [List.cons_append, build]; exact build_append cv sbo h ne xs ys t
  | (some k, s) :: xs, ys, .int it dom => by
      simp only [List.cons_append, build]
      split <;> exact build_append cv sbo h ne xs ys _
  | (some k, s) :: xs, ys, .value v dom => by
      simp only [List.cons_append, build]; exact build_append cv sbo h ne xs ys _

/-- `build_hash_table` with origins: each right row `(origin, key, slot)` updates only its
origin's table, in build order. -/
def buildPO (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool) :
    (Nat → JT) → List (Nat × (Option K × Nat)) → (Nat → JT)
  | ts, [] => ts
  | ts, (o, r) :: rest => buildPO cv sbo h ne (fun o' => if o' = o then build cv sbo h ne (ts o) [r] else ts o') rest

/-- **Isolation**: origin `o`'s table is exactly the single-table build over the right rows of
origin `o` (in order); rows of other origins never reach it. -/
theorem buildPO_origin (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool) :
    ∀ (rows : List (Nat × (Option K × Nat))) (ts : Nat → JT) (o : Nat),
      buildPO cv sbo h ne ts rows o = build cv sbo h ne (ts o) ((rows.filter fun r => r.1 = o).map (·.2))
  | [], _, _ => rfl
  | (o', r) :: rest, ts, o => by
      simp only [buildPO]
      rw [buildPO_origin cv sbo h ne rest _ o]
      by_cases e : o = o'
      · subst e
        simp only [ite_true, List.filter_cons, decide_true, List.map_cons]
        rw [show r :: (List.filter (fun r => decide (r.1 = o)) rest).map (·.2)
            = [r] ++ (List.filter (fun r => decide (r.1 = o)) rest).map (·.2) from rfl, build_append]
      · have e' : ¬ o' = o := fun x => e x.symm
        simp [e, e']

/-- `fill_matches` with origins: probe the left row's origin's table. -/
def fillMatchesPO (cv : K → K → Ordering × Bool) (h : K → Nat) (ts : Nat → JT) (o : Nat) (key : K) : List Nat :=
  fillMatches cv h (ts o) key

/-- An origin with no table (`get(&origin)` is `None`) yields no match. -/
theorem fillMatches_absent (cv : K → K → Ordering × Bool) (h : K → Nat) (key : K) :
    fillMatches cv h jtNew key = [] := by
  simp only [fillMatches, jtNew]; cases keyAsI64 key <;> rfl

/-- **Per-origin join** on an all-integer build side: a left row of origin `o` with key `n`
matches exactly the right rows of origin `o` whose key is `n`, in build order. -/
theorem buildPO_int_probe (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool)
    (rows : List (Nat × (Option K × Nat))) (o : Nat) (n : Int)
    (hint : ∀ o' k s, (o', (some k, s)) ∈ rows → (keyAsI64 k).isSome) :
    fillMatchesPO cv h (buildPO cv sbo h ne (fun _ => jtNew) rows) o (.int n) =
      ((rows.filter fun r => r.1 = o).map (·.2)).filterMap fun r => match r.1 with
        | some k => if keyAsI64 k = some n then some r.2 else none
        | none => none := by
  unfold fillMatchesPO
  rw [buildPO_origin]
  obtain ⟨it', dom', h1, h2⟩ := build_int cv sbo h ne ((rows.filter fun r => r.1 = o).map (·.2)) (fun _ => []) []
    (fun k s hm => by
      simp only [List.mem_map, List.mem_filter] at hm
      obtain ⟨⟨o', r⟩, ⟨hr, _⟩, he⟩ := hm
      subst he
      exact hint o' k s hr)
  simp only [jtNew] at h1 ⊢
  rw [h1]; simp only [fillMatches]
  by_cases hn : n.natAbs < lim
  · simp [keyAsI64, hn, h2]
  · have e : keyAsI64 (.int n) = none := by simp [keyAsI64, hn]
    rw [e]; symm
    apply List.filterMap_eq_nil_iff.2
    intro r _
    split
    · split
      · rename_i hk; exact absurd (keyAsI64_bound _ _ hk) hn
      · rfl
    · rfl

/-- `next` with origins: the stream ends at once when every built table is empty
(`os` = the origins seen on the build side); otherwise each left row `(l, origin, key)`
probes its origin's table. -/
def vhjRowsPO {L R O : Type} (cv : K → K → Ordering × Bool) (h : K → Nat) (ts : Nat → JT) (os : List Nat)
    (rights : Nat → R) (merge : L → R → O) (lefts : List (L × Nat × Option K)) : List O :=
  if os.all (fun o => jtEmpty (ts o)) then []
  else lefts.flatMap fun lk => match lk.2.2 with
    | none => []
    | some k => (fillMatchesPO cv h ts lk.2.1 k).map fun s => merge lk.1 (rights s)

theorem vhjRowsPO_spec {L R O : Type} (cv : K → K → Ordering × Bool) (h : K → Nat) (ts : Nat → JT)
    (os : List Nat) (rights : Nat → R) (merge : L → R → O) (l : L) (o : Nat) (k : K)
    (rest : List (L × Nat × Option K)) (hne : os.all (fun o => jtEmpty (ts o)) = false) :
    vhjRowsPO cv h ts os rights merge ((l, o, some k) :: rest) =
      (fillMatches cv h (ts o) k).map (fun s => merge l (rights s)) ++ vhjRowsPO cv h ts os rights merge rest ∧
    vhjRowsPO cv h ts os rights merge ((l, o, none) :: rest) = vhjRowsPO cv h ts os rights merge rest := by
  simp [vhjRowsPO, hne, fillMatchesPO]

/-- With `per_origin` off every row has origin 0: the per-origin join is the single-table join
of before #3035. -/
theorem vhjRowsPO_single {L R O : Type} (cv : K → K → Ordering × Bool) (h : K → Nat) (ts : Nat → JT)
    (rights : Nat → R) (merge : L → R → O) (lefts : List (L × Option K)) :
    vhjRowsPO cv h ts [0] rights merge (lefts.map fun lk => (lk.1, 0, lk.2)) =
      vhjRows cv h (ts 0) rights merge lefts := by
  simp [vhjRowsPO, vhjRows, fillMatchesPO, List.flatMap_map]

theorem buildPO_single (cv : K → K → Ordering × Bool) (sbo : Nat → Nat → Bool) (h : K → Nat) (ne : K → Bool)
    (rows : List (Option K × Nat)) :
    buildPO cv sbo h ne (fun _ => jtNew) (rows.map fun r => (0, r)) 0 = build cv sbo h ne jtNew rows := by
  rw [buildPO_origin]; congr 1; simp [List.filter_map, Function.comp_def]

/-- An encoding of the hasher input, for the concrete runs below (any function works). -/
def kEnc : K → Nat
  | .int n => n.natAbs
  | _ => 0

/-- **W2-rewrites-3 is fixed** (the join agrees with `=` near 2^53): build side `2^53`, `2^53 + 1`,
`2^53` (slots 0, 1, 2), as in `test_mixed_numeric_matches_keep_build_order`. The probe `2^53.0`
matches all three, in build order; `2^53` matches slots 0 and 2, `2^53 + 1` slot 1. And a build
side holding only `2^53.0` is found by the probe `2^53 + 1` (before #3034 the float went to the
exact integer table under key 2^53 and the probe looked up 2^53 + 1: no row). -/
theorem vhj_2pow53_fixed :
    let t := build numCv (fun a b => a == b) (joinHash kEnc) (fun _ => false) jtNew
      [(some (.int (2 ^ 53)), 0), (some (.int (2 ^ 53 + 1)), 1), (some (.int (2 ^ 53)), 2)]
    let tf := build numCv (fun a b => a == b) (joinHash kEnc) (fun _ => false) jtNew
      [(some (.float (2 ^ 53) true), 0)]
    fillMatches numCv (joinHash kEnc) t (.float (2 ^ 53) true) = [0, 1, 2] ∧
    fillMatches numCv (joinHash kEnc) t (.int (2 ^ 53)) = [0, 2] ∧
    fillMatches numCv (joinHash kEnc) t (.int (2 ^ 53 + 1)) = [1] ∧
    fillMatches numCv (joinHash kEnc) tf (.int (2 ^ 53 + 1)) = [0] := by
  decide

end CartVhj
