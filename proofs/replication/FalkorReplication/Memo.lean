/-
# The memo loop of `digest_created_nodes` / `digest_deleted_nodes`

Copied verbatim from proofs/effects_emit_apply (`Basic.lean`, same author
project family; each Lake project is standalone): the `slots`/`index`/`last`
loop of `graph/src/effects/v3/emit.rs:525-563`, `:814-846`, and the proof that
it is first-appearance grouping. `FalkorReplication.lean` bridges this
`groupBy` to its own (`memo_refines_groupBy`).
-/
namespace FalkorMemo

/-! ## First-appearance grouping (the specification)

    A slot per key in first-appearance order, members appended in input order. -/

def addTo {K} [DecidableEq K] {A} (k : K) (x : A) : List (K × List A) → List (K × List A)
  | []              => [(k, [x])]
  | (k', g) :: rest => if k' = k then (k', g ++ [x]) :: rest else (k', g) :: addTo k x rest

def groupAux {K} [DecidableEq K] {A} (key : A → K) :
    List A → List (K × List A) → List (K × List A)
  | [],        acc => acc
  | x :: rest, acc => groupAux key rest (addTo (key x) x acc)

def groupBy {K} [DecidableEq K] {A} (key : A → K) (l : List A) : List (K × List A) :=
  groupAux key l []

/-! ## The memo loop, as `digest_created_nodes` / `digest_deleted_nodes` write it

    ```text
    let slot = match last {
        Some(i) if slots[i].0 == key => i,
        _ => { let i = index.get(&key) or push-and-insert; last = Some(i); i }
    };
    slots[slot].1.push(id);
    ```
    (`emit.rs:525-563`, `emit.rs:814-846`). `index` is an `FxHashMap<Shape,
    usize>`; here an association list, since only `get` and `insert` of a fresh
    key are used and the map's iteration order never leaks into the output. -/

structure Memo (K A : Type) where
  slots : List (K × List A)
  index : List (K × Nat)
  last  : Option Nat

/-- `index.get(&key)` -/
def lookupIdx {K} [DecidableEq K] (k : K) : List (K × Nat) → Option Nat
  | []            => none
  | (k', i) :: tl => if k' = k then some i else lookupIdx k tl

/-- The position of the slot holding key `k` — what a correct `index` returns. -/
def pos {K} [DecidableEq K] {A} (k : K) : List (K × List A) → Option Nat
  | []            => none
  | (k', _) :: tl => if k' = k then some 0 else (pos k tl).map (· + 1)

/-- `slots[i].0` (in bounds under `MemoInv`; Rust would panic otherwise). -/
def keyAt {K A} : Nat → List (K × List A) → Option K
  | _,     []            => none
  | 0,     (k, _) :: _   => some k
  | i + 1, _ :: tl       => keyAt i tl

/-- `slots[i].1.push(x)` -/
def pushAt {K A} (i : Nat) (x : A) : List (K × List A) → List (K × List A)
  | []            => []
  | (k, g) :: tl  => match i with
                     | 0     => (k, g ++ [x]) :: tl
                     | j + 1 => (k, g) :: pushAt j x tl

def memoMiss {K} [DecidableEq K] {A} (m : Memo K A) (k : K) (x : A) : Memo K A :=
  match lookupIdx k m.index with
  | some j => { m with slots := pushAt j x m.slots, last := some j }
  | none   =>
    let j := m.slots.length
    { slots := m.slots ++ [(k, [x])], index := m.index ++ [(k, j)], last := some j }

def memoStep {K} [DecidableEq K] {A} (key : A → K) (m : Memo K A) (x : A) : Memo K A :=
  match m.last with
  | some i => if keyAt i m.slots = some (key x) then { m with slots := pushAt i x m.slots }
              else memoMiss m (key x) x
  | none   => memoMiss m (key x) x

def memoGroup {K} [DecidableEq K] {A} (key : A → K) (l : List A) : List (K × List A) :=
  (l.foldl (memoStep key) ⟨[], [], none⟩).slots

/-- The memo's invariant: `index` answers exactly what `pos` does, `last` is a
    valid position, and slot keys are distinct. -/
structure MemoInv {K} [DecidableEq K] {A} (m : Memo K A) : Prop where
  idx  : ∀ k, lookupIdx k m.index = pos k m.slots
  last : ∀ i, m.last = some i → i < m.slots.length
  nd   : (m.slots.map Prod.fst).Nodup

theorem lookupIdx_append {K} [DecidableEq K] (k : K) (l : List (K × Nat)) (k' : K) (j : Nat) :
    lookupIdx k (l ++ [(k', j)]) = match lookupIdx k l with
      | some i => some i
      | none   => if k' = k then some j else none := by
  induction l with
  | nil => simp [lookupIdx]
  | cons hd tl ih =>
    obtain ⟨a, i⟩ := hd
    simp only [List.cons_append, lookupIdx]
    by_cases h : a = k <;> simp [h, ih]

theorem pos_append {K} [DecidableEq K] {A} (k k' : K) (g : List A) :
    ∀ l : List (K × List A), pos k (l ++ [(k', g)]) = match pos k l with
      | some i => some i
      | none   => if k' = k then some l.length else none := by
  intro l
  induction l with
  | nil => simp [pos]
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    simp only [List.cons_append, pos]
    by_cases h : a = k
    · simp [h]
    · simp only [h, ↓reduceIte, ih, List.length_cons]
      cases pos k tl with
      | some i => rfl
      | none => by_cases e : k' = k <;> simp [e]

theorem pos_pushAt {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ (i : Nat) (l : List (K × List A)), pos k (pushAt i x l) = pos k l := by
  intro i l
  induction l generalizing i with
  | nil => simp [pushAt]
  | cons hd tl ih => obtain ⟨a, b⟩ := hd; cases i <;> simp [pushAt, pos, ih]

theorem pushAt_length {K A} (x : A) : ∀ (i : Nat) (l : List (K × List A)),
    (pushAt i x l).length = l.length := by
  intro i l
  induction l generalizing i with
  | nil => simp [pushAt]
  | cons hd tl ih => obtain ⟨k, g⟩ := hd; cases i <;> simp [pushAt, ih]

theorem pushAt_fst {K A} (x : A) : ∀ (i : Nat) (l : List (K × List A)),
    (pushAt i x l).map Prod.fst = l.map Prod.fst := by
  intro i l
  induction l generalizing i with
  | nil => simp [pushAt]
  | cons hd tl ih => obtain ⟨k, g⟩ := hd; cases i <;> simp [pushAt, ih]

theorem pos_lt {K} [DecidableEq K] {A} (k : K) :
    ∀ (l : List (K × List A)) (i : Nat), pos k l = some i → i < l.length := by
  intro l
  induction l with
  | nil => intro i h; simp [pos] at h
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    intro i h
    simp only [pos] at h
    by_cases e : a = k
    · simp only [e, ↓reduceIte, Option.some.injEq] at h; subst h; simp
    · simp only [e, ↓reduceIte] at h
      cases hp : pos k tl with
      | none => simp [hp] at h
      | some j => simp only [hp, Option.map_some, Option.some.injEq] at h; subst h
                  have := ih j hp; simp; omega

theorem pos_none {K} [DecidableEq K] {A} (k : K) :
    ∀ (l : List (K × List A)), pos k l = none → k ∉ l.map Prod.fst := by
  intro l
  induction l with
  | nil => intro _; simp
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    intro h
    simp only [pos] at h
    by_cases e : a = k
    · simp [e] at h
    · simp only [e, ↓reduceIte] at h
      have h' : pos k tl = none := by cases hp : pos k tl <;> simp_all
      simp only [List.map_cons, List.mem_cons, not_or]
      exact ⟨fun h2 => e h2.symm, ih h'⟩

/-- `addTo` on the slot holding key `k` is `pushAt` at that slot's position. -/
theorem addTo_eq_pushAt {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ (l : List (K × List A)) (i : Nat), pos k l = some i → addTo k x l = pushAt i x l := by
  intro l
  induction l with
  | nil => intro i h; simp [pos] at h
  | cons hd tl ih =>
    obtain ⟨k', g⟩ := hd
    intro i h
    simp only [pos] at h
    by_cases hk : k' = k
    · simp only [hk, ↓reduceIte, Option.some.injEq] at h
      subst h; subst hk; simp [addTo, pushAt]
    · simp only [hk, ↓reduceIte] at h
      cases hj : pos k tl with
      | none => simp [hj] at h
      | some j =>
        simp only [hj, Option.map_some, Option.some.injEq] at h
        subst h
        simp only [addTo, hk, ↓reduceIte, pushAt]
        rw [ih j hj]

theorem addTo_eq_append {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ (l : List (K × List A)), pos k l = none → addTo k x l = l ++ [(k, [x])] := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons hd tl ih =>
    obtain ⟨k', g⟩ := hd
    intro h
    simp only [pos] at h
    by_cases hk : k' = k
    · simp [hk] at h
    · simp only [hk, ↓reduceIte] at h
      have h' : pos k tl = none := by cases hp : pos k tl <;> simp_all
      simp [addTo, hk, ih h']

theorem keyAt_mem {K A} : ∀ (i : Nat) (l : List (K × List A)) (k : K),
    keyAt i l = some k → k ∈ l.map Prod.fst := by
  intro i l
  induction l generalizing i with
  | nil => intro k h; simp [keyAt] at h
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    intro k h
    cases i with
    | zero => simp only [keyAt, Option.some.injEq] at h; subst h; simp
    | succ j => simp only [keyAt] at h; exact List.mem_cons_of_mem _ (ih j k h)

/-- With distinct keys, the `last` fast path picks the slot `pos` would. -/
theorem keyAt_pos {K} [DecidableEq K] {A} (k : K) :
    ∀ (l : List (K × List A)) (i : Nat), (l.map Prod.fst).Nodup →
      keyAt i l = some k → pos k l = some i := by
  intro l
  induction l with
  | nil => intro i _ h; simp [keyAt] at h
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    intro i hnd h
    simp only [List.map_cons, List.nodup_cons] at hnd
    cases i with
    | zero => simp only [keyAt, Option.some.injEq] at h; subst h; simp [pos]
    | succ j =>
      simp only [keyAt] at h
      have hk : a ≠ k := fun e => hnd.1 (e ▸ keyAt_mem j tl k h)
      simp [pos, hk, ih j hnd.2 h]

theorem memoMiss_spec {K} [DecidableEq K] {A} (m : Memo K A) (k : K) (x : A)
    (hinv : MemoInv m) :
    (memoMiss m k x).slots = addTo k x m.slots ∧ MemoInv (memoMiss m k x) := by
  unfold memoMiss
  cases hl : lookupIdx k m.index with
  | some j =>
    have hf : pos k m.slots = some j := by rw [← hinv.idx]; exact hl
    refine ⟨(addTo_eq_pushAt _ _ _ _ hf).symm, ⟨?_, ?_, ?_⟩⟩
    · intro k'; simp only; rw [hinv.idx, pos_pushAt]
    · intro i hi; simp only [Option.some.injEq] at hi; subst hi
      rw [pushAt_length]; exact pos_lt _ _ _ hf
    · simp only [pushAt_fst]; exact hinv.nd
  | none =>
    have hf : pos k m.slots = none := by rw [← hinv.idx]; exact hl
    refine ⟨(addTo_eq_append _ _ _ hf).symm, ⟨?_, ?_, ?_⟩⟩
    · intro k'
      simp only
      rw [lookupIdx_append, hinv.idx k', pos_append]
    · intro i hi; simp only [Option.some.injEq] at hi; subst hi; simp
    · simp only [List.map_append, List.map_cons, List.map_nil]
      rw [List.nodup_append]
      refine ⟨hinv.nd, by simp, ?_⟩
      intro a ha b hb e
      simp only [List.mem_singleton] at hb
      subst hb; subst e
      exact pos_none _ _ hf ha

/-- One step of the memo loop is one step of first-appearance grouping, and it
    keeps the memo's invariant. -/
theorem memoStep_spec {K} [DecidableEq K] {A} (key : A → K) (m : Memo K A) (x : A)
    (hinv : MemoInv m) :
    (memoStep key m x).slots = addTo (key x) x m.slots ∧ MemoInv (memoStep key m x) := by
  unfold memoStep
  cases hl : m.last with
  | none => exact memoMiss_spec m (key x) x hinv
  | some i =>
    simp only
    by_cases he : keyAt i m.slots = some (key x)
    · rw [if_pos he]
      have hf := keyAt_pos (key x) m.slots i hinv.nd he
      refine ⟨(addTo_eq_pushAt _ _ _ _ hf).symm, ⟨?_, ?_, ?_⟩⟩
      · intro k; simp only; rw [hinv.idx, pos_pushAt]
      · intro i' hi'; simp only at hi'; rw [pushAt_length]; simp only [Option.some.injEq] at hi'; subst hi'; exact hinv.last i hl
      · simp only [pushAt_fst]; exact hinv.nd
    · rw [if_neg he]; exact memoMiss_spec m (key x) x hinv

theorem memoFold_spec {K} [DecidableEq K] {A} (key : A → K) :
    ∀ (l : List A) (m : Memo K A), MemoInv m →
      (l.foldl (memoStep key) m).slots = groupAux key l m.slots := by
  intro l
  induction l with
  | nil => intro m _; rfl
  | cons x rest ih =>
    intro m hinv
    obtain ⟨h1, h2⟩ := memoStep_spec key m x hinv
    simp only [List.foldl_cons, groupAux]
    rw [ih _ h2, h1]

/-- **`memoGroup_eq_groupBy`: the memo is an optimisation over first-appearance
    grouping, exactly.** `digest_created_nodes`' and `digest_deleted_nodes'`
    `slots`/`index`/`last` loop yields the same slots, in the same order, with
    the same members in the same order, as plain first-appearance `groupBy`:
    the `last` fast path only skips a lookup that would have returned `last`,
    and a slot is pushed only for a key in no slot yet. -/
theorem memoGroup_eq_groupBy {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    memoGroup key l = groupBy key l := by
  unfold memoGroup groupBy
  exact memoFold_spec key l ⟨[], [], none⟩
    ⟨fun k => by simp [lookupIdx, pos], fun i h => by simp at h, by simp⟩

end FalkorMemo
