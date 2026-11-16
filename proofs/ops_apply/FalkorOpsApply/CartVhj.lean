/-
CartesianProduct and ValueHashJoin (graph/src/runtime/ops/cartesian_product.rs,
value_hash_join.rs).

| here | there |
| --- | --- |
| `adv`, `orbit`, `allTuples` | `RightSide::advance` (cartesian_product.rs:105): the odometer |
| `rsNew`, `rsEmpty`, `rewind` | `RightSide::new` (:80), `is_empty` (:94), `rewind` (:117) |
| `CpSt.new`, `materialize`, `cpRows` | `CartesianProductOp::new` (:123), `materialize_right` (:144), `next` (:162) |
| `JT`, `jtEmpty` | `JoinHashTable`, `is_empty` (value_hash_join.rs:98) |
| `hashValue`, `keysMatch`, `keyAsI64`, `intKeyLimit` | `hash_value` (:120, any function: `JoinKey::hash` :136 and `f32_join_bits` :193 not modelled), `keys_match` (:218), `key_as_i64` (:242), `INT_KEY_LIMIT` (:254) |
| `sameBuildKey`, `insertValue`, `promote` | `same_build_key` (:259), `insert_value` (:289), `promote_int_table` (:309) |
| `VhjSt.new`, `build`, `fillMatches`/`probe`, `vhjRows` | `ValueHashJoinOp::new` (:341), `build_hash_table` (:369), `fill_matches` (:442), `next` (:509) |
| `numEq` | Cypher `=` on numeric keys (`compare_value`; `toF` = `i as f64`) |
-/
namespace CartVhj

/-! ## The odometer (cartesian_product.rs:105-115) -/

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
  | other (tag : Nat)

/-- `INT_KEY_LIMIT` (value_hash_join.rs:254): 2^53, the integer-table magnitude bound. -/
def intKeyLimit : Nat := 2 ^ 53

/-- value_hash_join.rs:242 (PR #3034): an int, or a whole float, only when its magnitude is
below 2^53 (where `i as f64` is exact); everything else (incl. larger ints) is `None`. -/
def keyAsI64 : K → Option Int
  | .int n => if n.natAbs < intKeyLimit then some n else none
  | .float f true => if f.natAbs < intKeyLimit then some f else none
  | _ => none

theorem keyAsI64_spec (n : Int) (f : Int) (hn : n.natAbs < intKeyLimit) (hf : f.natAbs < intKeyLimit) :
    keyAsI64 (.int n) = some n ∧ keyAsI64 (.float f true) = some f ∧ keyAsI64 (.float f false) = none ∧
    keyAsI64 .null = none := by
  simp [keyAsI64, hn, hf]

/-- Keys of magnitude ≥ 2^53 never reach the integer table (they promote it, or miss it). -/
theorem keyAsI64_large (n : Int) (hn : intKeyLimit ≤ n.natAbs) :
    keyAsI64 (.int n) = none ∧ keyAsI64 (.float n true) = none := by
  simp [keyAsI64, Nat.not_lt.mpr hn]

/-- Cypher `=` on numeric keys (`compare_value`): Int/Int exactly, Int/Float through
`toF` (= `i as f64`, the f64 rounding of an integer), Float/Float on the value. A non-whole
float (`exact = false`) never equals an integer-valued key; non-numerics never equal numerics. -/
def numEq (toF : Int → Int) : K → K → Bool
  | .int a, .int b => a == b
  | .int a, .float f true => toF a == f
  | .float f true, .int a => toF a == f
  | .float f true, .float g true => f == g
  | _, _ => false

/-- **W2-rewrites-3 fixed**: the integer fast path agrees with Cypher `=`. For a key `k` stored
in the integer table under `n`, a probe `p` reaches entry `n` exactly when `p = k`, for any
f64 rounding `toF` that is exact below 2^53 and stays ≥ 2^53 in magnitude above it. -/
theorem keyAsI64_sound (toF : Int → Int) (hex : ∀ n : Int, n.natAbs < intKeyLimit → toF n = n)
    (hbig : ∀ n : Int, intKeyLimit ≤ n.natAbs → intKeyLimit ≤ (toF n).natAbs)
    (k p : K) (n : Int) (hk : keyAsI64 k = some n) :
    keyAsI64 p = some n ↔ numEq toF k p = true := by
  cases k with
  | null => simp [keyAsI64] at hk
  | other t => simp [keyAsI64] at hk
  | int a =>
    simp only [keyAsI64] at hk
    split at hk <;> simp at hk
    subst hk
    rename_i ha
    cases p with
    | null => simp [keyAsI64, numEq]
    | other t => simp [keyAsI64, numEq]
    | int b =>
      simp only [keyAsI64, numEq, beq_iff_eq]
      split <;> simp <;> omega
    | float g e =>
      cases e with
      | false => simp [keyAsI64, numEq]
      | true =>
        simp only [keyAsI64, numEq, beq_iff_eq, hex a ha]
        split <;> simp <;> omega
  | float f e =>
    cases e with
    | false => simp [keyAsI64] at hk
    | true =>
      simp only [keyAsI64] at hk
      split at hk <;> simp at hk
      subst hk
      rename_i hf
      cases p with
      | null => simp [keyAsI64, numEq]
      | other t => simp [keyAsI64, numEq]
      | int b =>
        simp only [keyAsI64, numEq, beq_iff_eq]
        by_cases hb : b.natAbs < intKeyLimit
        · simp [hb, hex b hb]
        · have h1 := hbig b (Nat.le_of_not_lt hb)
          simp [hb]; intro h; subst h; omega
      | float g e =>
        cases e with
        | false => simp [keyAsI64, numEq]
        | true =>
          simp only [keyAsI64, numEq, beq_iff_eq]
          split <;> simp <;> omega

/-- The code before PR #3034 (`Int(n) => Some(n)`, any whole float): it keyed `2^53 + 1` and
`2^53.0` apart although `2^53 + 1 = 2^53.0` (W2-rewrites-3). -/
def keyAsI64Old : K → Option Int
  | .int n => some n
  | .float f true => some f
  | _ => none

theorem keyAsI64Old_unsound (toF : Int → Int) (h : toF (2 ^ 53 + 1) = 2 ^ 53) :
    numEq toF (.int (2 ^ 53 + 1)) (.float (2 ^ 53) true) = true ∧
      keyAsI64Old (.int (2 ^ 53 + 1)) ≠ keyAsI64Old (.float (2 ^ 53) true) := by
  have h' : toF 9007199254740993 = 9007199254740992 := h
  simp [numEq, keyAsI64Old, h']

/-- `=` is not transitive across Int/Float above 2^53 (`2^53 = 2^53.0 = 2^53 + 1`, yet
`2^53 <> 2^53 + 1`): why build keys cannot be grouped by `keys_match`. -/
theorem numEq_not_transitive (toF : Int → Int) (h0 : toF (2 ^ 53) = 2 ^ 53) (h1 : toF (2 ^ 53 + 1) = 2 ^ 53) :
    numEq toF (.int (2 ^ 53)) (.float (2 ^ 53) true) = true ∧
      numEq toF (.float (2 ^ 53) true) (.int (2 ^ 53 + 1)) = true ∧
      numEq toF (.int (2 ^ 53)) (.int (2 ^ 53 + 1)) = false := by
  have h0' : toF 9007199254740992 = 9007199254740992 := h0
  have h1' : toF 9007199254740993 = 9007199254740992 := h1
  simp [numEq, h0', h1']

/-- value_hash_join.rs:218: equal AND not disjoint-or-null (`cv` = `compare_value`). -/
def keysMatch (cv : K → K → Ordering × Bool) (a b : K) : Bool := (cv a b).1 == .eq && !(cv a b).2

theorem keysMatch_spec (cv : K → K → Ordering × Bool) (a b : K) :
    keysMatch cv a b = true ↔ (cv a b).1 = .eq ∧ (cv a b).2 = false := by
  simp [keysMatch]

/-- value_hash_join.rs:120: the seeded hasher — any function of the key. -/
def hashValue (h : K → Nat) (k : K) : Nat := h k

theorem hashValue_det (h : K → Nat) (a b : K) (e : a = b) : hashValue h a = hashValue h b := by rw [e]

/-- The value table: hash ↦ bucket of `(key, refs)` entries. -/
abbrev VT := Nat → List (K × List Nat)

/-- `same_build_key` (value_hash_join.rs:259): Int/Int exactly, never Int with Float, else
`keys_match` (`km`). Lists/maps/paths (recursing element-wise) are abstracted into `other`. -/
def sameBuildKey (km : K → K → Bool) : K → K → Bool
  | .int a, .int b => a == b
  | .int _, .float _ _ => false
  | .float _ _, .int _ => false
  | a, b => km a b

theorem sameBuildKey_int_float (km : K → K → Bool) (a f : Int) (e : Bool) :
    sameBuildKey km (.int a) (.float f e) = false ∧ sameBuildKey km (.float f e) (.int a) = false := ⟨rfl, rfl⟩

/-- Two build keys share an entry only when every probe matching one matches the other
(with `keys_match` = Cypher `=` on numerics). -/
theorem sameBuildKey_sound (toF : Int → Int) (a b p : K) (h : sameBuildKey (numEq toF) a b = true) :
    numEq toF a p = numEq toF b p := by
  cases a with
  | null => cases b <;> simp [sameBuildKey, numEq] at h
  | other t => cases b <;> simp [sameBuildKey, numEq] at h
  | int x =>
    cases b with
    | int y => simp [sameBuildKey] at h; subst h; rfl
    | _ => simp [sameBuildKey, numEq] at h
  | float f e =>
    cases e <;> cases b <;> simp [sameBuildKey, numEq] at h
    rename_i g e'
    cases e' <;> simp [numEq] at h
    subst h; rfl

def addRef (cv : K → K → Ordering × Bool) (key : K) (slot : Nat) : List (K × List Nat) → List (K × List Nat)
  | [] => [(key, [slot])]
  | (k, rs) :: es =>
    if sameBuildKey (keysMatch cv) k key then (k, rs ++ [slot]) :: es else (k, rs) :: addRef cv key slot es

/-- value_hash_join.rs:289-303: never-equal keys (null, NaN) are skipped. -/
def insertValue (cv : K → K → Ordering × Bool) (h : K → Nat) (neverEq : K → Bool) (t : VT) (key : K)
    (slot : Nat) : VT :=
  if neverEq key then t else fun hk => if hk = h key then addRef cv key slot (t hk) else t hk

/-- Probe (`fill_matches`, value arm, :459-492): every entry of the key's bucket that
`keys_match`es; their refs merged back into build order (`sorted_by_key((batch,row))`, a ref
here being its build position). -/
def probe (cv : K → K → Ordering × Bool) (h : K → Nat) (t : VT) (key : K) : List Nat :=
  (((t (h key)).filter fun e => keysMatch cv e.1 key).flatMap (·.2)).mergeSort

/-- Inserting a never-equal key changes nothing; otherwise the slot is appended to the entry
whose key is `same_build_key` (or a new entry), in the key's hash bucket; other buckets untouched. -/
theorem insertValue_spec (cv : K → K → Ordering × Bool) (h : K → Nat) (ne : K → Bool) (t : VT) (key : K) (s : Nat) :
    (ne key = true → insertValue cv h ne t key s = t) ∧
    (ne key = false → insertValue cv h ne t key s (h key) = addRef cv key s (t (h key)) ∧
      ∀ hk, hk ≠ h key → insertValue cv h ne t key s hk = t hk) := by
  refine ⟨fun hn => by simp [insertValue, hn], fun hn => ⟨by simp [insertValue, hn], fun hk hne => by simp [insertValue, hn, hne]⟩⟩

/-- A fresh key's probe after insertion finds exactly the new slot (given `keys_match` is
reflexive on joinable keys). -/
theorem probe_insert_fresh (cv : K → K → Ordering × Bool) (h : K → Nat) (ne : K → Bool) (t : VT) (key : K) (s : Nat)
    (hn : ne key = false) (hfresh : t (h key) = []) (hrefl : keysMatch cv key key = true) :
    probe cv h (insertValue cv h ne t key s) key = [s] := by
  simp [probe, insertValue, hn, hfresh, addRef, hrefl]

/-- New in PR #3034: when several entries of the bucket match (e.g. `2^53` and `2^53.0` both
`=` a probe `2^53 + 1`), the probe returns all their rows, in build order. -/
theorem probe_merges (cv : K → K → Ordering × Bool) (h : K → Nat) (t : VT) (key k1 k2 : K) (r1 r2 : List Nat)
    (hb : t (h key) = [(k1, r1), (k2, r2)]) (m1 : keysMatch cv k1 key = true) (m2 : keysMatch cv k2 key = true) :
    probe cv h t key = (r1 ++ r2).mergeSort ∧ (probe cv h t key).Perm (r1 ++ r2) ∧
      (probe cv h t key).Pairwise (· ≤ ·) := by
  have e : probe cv h t key = (r1 ++ r2).mergeSort := by simp [probe, hb, m1, m2]
  refine ⟨e, e ▸ List.mergeSort_perm _ _, e ▸ ?_⟩
  exact (List.pairwise_mergeSort (le := fun a b => decide (a ≤ b))
    (fun a b c hab hbc => by simp at *; omega) (fun a b => by simp; omega) _).imp (by simp)

/-- The integer fast path: `int ↦ refs`. -/
abbrev IT := Int → List Nat

def intInsert (t : IT) (n : Int) (s : Nat) : IT := fun m => if m = n then t m ++ [s] else t m

/-- `promote_int_table` (:309): every integer key `n` (the drained domain `dom`) is re-keyed as
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

/-- value_hash_join.rs:98 (`dom` = the occupied keys / buckets). -/
def jtEmpty : JT → Bool
  | .int _ dom => dom.isEmpty
  | .value _ dom => dom.isEmpty

theorem jtEmpty_spec (t : IT) (v : VT) (d : List Int) (e : List Nat) :
    jtEmpty (.int t d) = d.isEmpty ∧ jtEmpty (.value v e) = e.isEmpty := ⟨rfl, rfl⟩

/-- `build_hash_table` (:369-431; an `Ints` column key goes to the integer table only below 2^53, i.e. exactly when `keyAsI64` is `some`), over the right rows' evaluated keys (`none` = null, skipped):
start on the integer table; the first key that is not integer-valued promotes it and switches
to the value table for the rest. -/
def build (cv : K → K → Ordering × Bool) (h : K → Nat) (ne : K → Bool) :
    JT → List (Option K × Nat) → JT
  | t, [] => t
  | t, (none, _) :: rest => build cv h ne t rest
  | .int it dom, (some k, s) :: rest =>
    match keyAsI64 k with
    | some n => build cv h ne (.int (intInsert it n s) (if dom.contains n then dom else dom ++ [n])) rest
    | none =>
      let v := insertValue cv h ne (promote h dom it) k s
      build cv h ne (.value v (dom.map (fun n => h (.int n)) ++ [h k])) rest
  | .value v dom, (some k, s) :: rest =>
    build cv h ne (.value (insertValue cv h ne v k s) (dom ++ [h k])) rest

/-- On an all-integer-valued build side the table is the integer table, and a probe for `n`
finds exactly the right rows whose key is `n`, in build order (nulls never join). -/
theorem build_int (cv : K → K → Ordering × Bool) (h : K → Nat) (ne : K → Bool) :
    ∀ (rows : List (Option K × Nat)) (it : IT) (dom : List Int),
      (∀ k s, (some k, s) ∈ rows → (keyAsI64 k).isSome) →
      ∃ it' dom', build cv h ne (.int it dom) rows = .int it' dom' ∧
        ∀ n, it' n = it n ++ (rows.filterMap fun r => match r.1 with
          | some k => if keyAsI64 k = some n then some r.2 else none
          | none => none)
  | [], it, dom, _ => ⟨it, dom, rfl, fun n => by simp⟩
  | (none, s) :: rest, it, dom, hr => by
      obtain ⟨it', dom', h1, h2⟩ := build_int cv h ne rest it dom (fun k s' hm => hr k s' (List.mem_cons_of_mem _ hm))
      exact ⟨it', dom', by simp [build, h1], fun n => by simp [h2]⟩
  | (some k, s) :: rest, it, dom, hr => by
      obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hr k s (List.mem_cons_self ..))
      obtain ⟨it', dom', h1, h2⟩ := build_int cv h ne rest (intInsert it m s)
        (if dom.contains m then dom else dom ++ [m])
        (fun k s' hm' => hr k s' (List.mem_cons_of_mem _ hm'))
      refine ⟨it', dom', ?_, fun n => ?_⟩
      · simp only [build, hm]; exact h1
      rw [h2 n]
      by_cases e : n = m
      · subst e; simp [intInsert, hm]
      · have : ¬ m = n := fun x => e x.symm
        simp [intInsert, e, hm, this]

theorem keyAsI64_bound (k : K) (n : Int) (h : keyAsI64 k = some n) : n.natAbs < intKeyLimit := by
  cases k with
  | int a => simp only [keyAsI64] at h; split at h <;> simp at h; subst h; assumption
  | float f e =>
    cases e <;> simp only [keyAsI64] at h
    · simp at h
    · split at h <;> simp at h; subst h; assumption
  | _ => simp [keyAsI64] at h

/-- New in PR #3034: the integer table never gains an entry at a magnitude ≥ 2^53. -/
theorem build_int_bounded (cv : K → K → Ordering × Bool) (h : K → Nat) (ne : K → Bool)
    (rows : List (Option K × Nat)) (it : IT) (dom : List Int)
    (hr : ∀ k s, (some k, s) ∈ rows → (keyAsI64 k).isSome) :
    ∃ it' dom', build cv h ne (.int it dom) rows = .int it' dom' ∧
      ∀ n : Int, intKeyLimit ≤ n.natAbs → it' n = it n := by
  obtain ⟨it', dom', h1, h2⟩ := build_int cv h ne rows it dom hr
  refine ⟨it', dom', h1, fun n hn => ?_⟩
  rw [h2 n]
  suffices H : (rows.filterMap fun r => match r.1 with
      | some k => if keyAsI64 k = some n then some r.2 else none
      | none => none) = [] by simp [H]
  rw [List.filterMap_eq_nil_iff]
  intro r _
  rcases r with ⟨_ | k, s⟩
  · rfl
  · have : keyAsI64 k ≠ some n := fun e => absurd (keyAsI64_bound k n e) (Nat.not_lt.mpr hn)
    simp [this]

/-- `fill_matches` (:442): integer table probes `key_as_i64(key)` (a non-integer key finds
nothing); value table probes the key's bucket by `keys_match`. -/
def fillMatches (cv : K → K → Ordering × Bool) (h : K → Nat) : JT → K → List Nat
  | .int it _, key => match keyAsI64 key with
    | some n => it n
    | none => []
  | .value v _, key => probe cv h v key

theorem fillMatches_int (cv : K → K → Ordering × Bool) (h : K → Nat) (it : IT) (d : List Int) (n : Int) (f : Int)
    (hn : n.natAbs < intKeyLimit) (hf : f.natAbs < intKeyLimit) :
    fillMatches cv h (.int it d) (.int n) = it n ∧ fillMatches cv h (.int it d) (.float f true) = it f ∧
    fillMatches cv h (.int it d) (.float f false) = [] := by
  simp [fillMatches, keyAsI64, hn, hf]

/-- A probe key of magnitude ≥ 2^53 finds nothing in the integer table (whose keys are all
below 2^53, `build_int_bounded`, and so never `=` it). -/
theorem fillMatches_int_large (cv : K → K → Ordering × Bool) (h : K → Nat) (it : IT) (d : List Int) (n : Int)
    (hn : intKeyLimit ≤ n.natAbs) :
    fillMatches cv h (.int it d) (.int n) = [] ∧ fillMatches cv h (.int it d) (.float n true) = [] := by
  simp [fillMatches, keyAsI64, Nat.not_lt.mpr hn]

/-- `ValueHashJoinOp::new` (:341): no table, no buffered rows. -/
structure VhjSt where
  built : Option JT
  leftPos : Nat
  matchPos : Nat
  found : List Nat

def VhjSt.new : VhjSt := ⟨none, 0, 0, []⟩

theorem vhjNew_spec : VhjSt.new.built = none ∧ VhjSt.new.leftPos = 0 ∧ VhjSt.new.found = [] := ⟨rfl, rfl, rfl⟩

/-- `next` (:509): an empty table ends the stream at once; otherwise each left row (in
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

end CartVhj
