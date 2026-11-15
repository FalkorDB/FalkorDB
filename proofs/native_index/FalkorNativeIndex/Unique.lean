import FalkorNativeIndex.Query
set_option linter.unusedSimpArgs false

/-!
# `unique.rs`: where a unique-constraint check looks (`graph/src/index/falkordb/unique.rs`)

The check itself (comparing candidates' keys with the members') is `unique_violated` in
`runtime::pending` (project `pending_commit`); this file proves what the native side promises it:

* `nativeUniqueQuery`: one lookup per constrained property whose column can cover every holder of the value
  (`equal_candidates`: a column that is not `Building` and indexes the value; the lookup may return extra
  entities, which `unique_violated` re-checks), and `None` when no
  column can: a decline is never an empty answer (`nativeUniqueQuery_none_iff`).
* `fewest`/`fewest_candidates`: the loop reads one entity from every lookup per round and returns the
  lookup that runs out first, i.e. the **first shortest** one (`fewest_eq_minRow`), one of the lookups
  (`fewest_mem`), no longer than any other (`fewest_minimal`).
* `ReadOnce`: a set of lookups already read is not read again (`readOnce_repeat`), and a lookup set
  not seen before is read and recorded (`readOnce_first`).
-/

namespace NI

/-! ## `native_unique_query` (`unique.rs:20-33`) -/

/-- `native_unique_query`: the lookups of the properties a column can answer for (`properties.zip(values)`
    truncates to the shorter), or `None` when there are none. -/
def nativeUniqueQuery (x : FalkorDbIndex) (e : EntityType) (label : String) (props : List String)
    (values : List Value) : Option (List EqualCandidates) :=
  let lookups := (props.zip values).filterMap (fun av => x.equalCandidates e label av.1 av.2)
  if lookups.isEmpty then none else some lookups

theorem ite_isEmpty_none {α} (L : List α) : (if L.isEmpty = true then none else some L) = none ↔ L = [] := by
  cases L <;> simp

/-- Exactly the answerable lookups, and at least one. -/
theorem nativeUniqueQuery_some (x : FalkorDbIndex) (e : EntityType) (label : String) (props : List String)
    (values : List Value) (ls : List EqualCandidates) :
    nativeUniqueQuery x e label props values = some ls ↔
      ls = (props.zip values).filterMap (fun av => x.equalCandidates e label av.1 av.2) ∧ ls ≠ [] := by
  unfold nativeUniqueQuery
  generalize (props.zip values).filterMap (fun av => x.equalCandidates e label av.1 av.2) = L
  cases L with
  | nil => simp
  | cons c cs =>
    simp only [List.isEmpty_cons, Bool.false_eq_true, if_false, Option.some.injEq]
    constructor
    · rintro rfl; exact ⟨rfl, by simp⟩
    · rintro ⟨rfl, _⟩; rfl

/-- **Declines exactly when no column can answer exactly**: `None` iff for every property, there is no
    column, or it is `Building` (it has not seen every entity), or its kind does not index the value. -/
theorem nativeUniqueQuery_none_iff (x : FalkorDbIndex) (e : EntityType) (label : String) (props : List String)
    (values : List Value) :
    nativeUniqueQuery x e label props values = none ↔
      ∀ av ∈ props.zip values, x.equalCandidates e label av.1 av.2 = none := by
  unfold nativeUniqueQuery
  simp only
  rw [ite_isEmpty_none, List.filterMap_eq_nil_iff]

/-- A lookup is made for a property exactly when `equal_candidates` answers it. -/
theorem nativeUniqueQuery_mem (x : FalkorDbIndex) (e : EntityType) (label : String) (props : List String)
    (values : List Value) (ls : List EqualCandidates) (h : nativeUniqueQuery x e label props values = some ls)
    (c : EqualCandidates) :
    c ∈ ls ↔ ∃ av ∈ props.zip values, x.equalCandidates e label av.1 av.2 = some c := by
  rw [(nativeUniqueQuery_some x e label props values ls).1 h |>.1]
  simp [List.mem_filterMap]

/-! ## `fewest_candidates` (`unique.rs:69-85`) -/

/-- One round over `(remaining, read)` pairs, one per lookup: take one entity from each in order; the
    first whose lookup has run out returns what it read (`Some(id) => read[i].push(id)`,
    `None => return read[i]`). -/
def roundGo : List (List Nat × List Nat) → Except (List Nat) (List (List Nat × List Nat))
  | [] => .ok []
  | (rem, read) :: rest =>
    match rem with
    | [] => .error read
    | x :: xs => (roundGo rest).map (fun r => (xs, read ++ [x]) :: r)

/-- The `loop`, with fuel (every round consumes one entity of every lookup, so a lookup runs out within
    `max length + 1` rounds). -/
def fewestLoop : Nat → List (List Nat × List Nat) → List Nat
  | 0, _ => []
  | n + 1, ps =>
    match roundGo ps with
    | .error r => r
    | .ok ps' => fewestLoop n ps'

/-- `fewest_candidates`: one lookup returns its rows as they are; otherwise the round-robin loop.
    Precondition (the Rust loops forever on none): at least one lookup, which `native_unique_query`
    guarantees (`nativeUniqueQuery_some`). -/
def fewest (rows : List (List Nat)) : List Nat :=
  match rows with
  | [r] => r
  | _ => fewestLoop ((rows.map List.length).foldl max 0 + 1) (rows.map (fun r => (r, [])))

/-- The first shortest of the rows (ties go to the earlier). -/
def minAcc (m : List Nat) : List (List Nat) → List Nat
  | [] => m
  | r :: rs => minAcc (if r.length < m.length then r else m) rs

def minRow : List (List Nat) → List Nat
  | [] => []
  | r :: rs => minAcc r rs

theorem minAcc_const (m : List Nat) (rs : List (List Nat)) (h : ∀ r ∈ rs, m.length ≤ r.length) :
    minAcc m rs = m := by
  induction rs generalizing m with
  | nil => rfl
  | cons r rs ih =>
    have : ¬ r.length < m.length := by have := h r (by simp); omega
    simp only [minAcc, this, if_false]
    exact ih m (fun q hq => h q (List.mem_cons_of_mem _ hq))

theorem minAcc_split (pre : List (List Nat)) (r : List Nat) (post : List (List Nat)) :
    ∀ m : List Nat, r.length < m.length → (∀ q ∈ pre, r.length < q.length) → (∀ q ∈ post, r.length ≤ q.length) →
      minAcc m (pre ++ r :: post) = r := by
  induction pre with
  | nil =>
    intro m hm _ hpost
    simp only [List.nil_append, minAcc, hm, if_true]
    exact minAcc_const r post hpost
  | cons q pre ih =>
    intro m hm hpre hpost
    simp only [List.cons_append, minAcc]
    apply ih
    · by_cases h : q.length < m.length
      · simp only [h, if_true]; exact hpre q (by simp)
      · simp only [h, if_false]; exact hm
    · exact fun q' hq' => hpre q' (List.mem_cons_of_mem _ hq')
    · exact hpost

/-- `minRow` of a list whose element `r` is shorter than everything before it and no longer than
    everything after it is `r`. -/
theorem minRow_split (pre : List (List Nat)) (r : List Nat) (post : List (List Nat))
    (hpre : ∀ q ∈ pre, r.length < q.length) (hpost : ∀ q ∈ post, r.length ≤ q.length) :
    minRow (pre ++ r :: post) = r := by
  cases pre with
  | nil => simp only [List.nil_append, minRow]; exact minAcc_const r post hpost
  | cons p pre =>
    simp only [List.cons_append, minRow]
    exact minAcc_split pre r post p (hpre p (by simp)) (fun q hq => hpre q (List.mem_cons_of_mem _ hq)) hpost

theorem minAcc_mem1 (rs : List (List Nat)) : ∀ m : List Nat, minAcc m rs ∈ m :: rs := by
  induction rs with
  | nil => intro m; simp [minAcc]
  | cons r rs ih =>
    intro m
    simp only [minAcc]
    by_cases hlt : r.length < m.length
    · simp only [hlt, if_true]
      have := ih r
      simp only [List.mem_cons] at this ⊢
      exact Or.inr this
    · simp only [hlt, if_false]
      have := ih m
      simp only [List.mem_cons] at this ⊢
      rcases this with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inr h)

theorem minAcc_le (rs : List (List Nat)) :
    ∀ m : List Nat, ∀ q ∈ m :: rs, (minAcc m rs).length ≤ q.length := by
  induction rs with
  | nil => intro m q hq; simp at hq; simp [minAcc, hq]
  | cons r rs ih =>
    intro m q hq
    simp only [minAcc]
    by_cases hlt : r.length < m.length
    · simp only [hlt, if_true]
      rcases List.mem_cons.1 hq with rfl | hq
      · have := ih r r (by simp); omega
      · exact ih r q hq
    · simp only [hlt, if_false]
      rcases List.mem_cons.1 hq with rfl | hq
      · exact ih q q (by simp)
      · rcases List.mem_cons.1 hq with rfl | hq
        · have := ih m m (by simp); omega
        · exact ih m q (List.mem_cons_of_mem _ hq)

/-! The loop. A state is a list of `(remaining, read)`; `full p = read ++ remaining` is its lookup's
whole row list, unchanged by a round. -/

def full (p : List Nat × List Nat) : List Nat := p.2 ++ p.1

def stepP (p : List Nat × List Nat) : List Nat × List Nat :=
  match p.1 with
  | [] => p
  | x :: xs => (xs, p.2 ++ [x])

theorem roundGo_ok (ps : List (List Nat × List Nat)) (h : ∀ p ∈ ps, p.1 ≠ []) :
    roundGo ps = .ok (ps.map stepP) := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨rem, read⟩ := p
    cases rem with
    | nil => exact absurd rfl (h ([], read) (by simp))
    | cons x xs =>
      simp only [roundGo, ih (fun q hq => h q (List.mem_cons_of_mem _ hq)), Except.map, List.map_cons, stepP]

theorem roundGo_error (ps : List (List Nat × List Nat)) (h : ∃ p ∈ ps, p.1 = []) :
    ∃ pre p post, ps = pre ++ p :: post ∧ (∀ q ∈ pre, q.1 ≠ []) ∧ p.1 = [] ∧ roundGo ps = .error p.2 := by
  induction ps with
  | nil => obtain ⟨p, hp, _⟩ := h; simp at hp
  | cons p ps ih =>
    obtain ⟨rem, read⟩ := p
    cases rem with
    | nil => exact ⟨[], (([] : List Nat), read), ps, rfl, by simp, rfl, rfl⟩
    | cons x xs =>
      have h' : ∃ q ∈ ps, q.1 = [] := by
        obtain ⟨q, hq, hq0⟩ := h
        rcases List.mem_cons.1 hq with rfl | hq
        · simp at hq0
        · exact ⟨q, hq, hq0⟩
      obtain ⟨pre, q, post, rfl, hpre, hq, hr⟩ := ih h'
      refine ⟨(x :: xs, read) :: pre, q, post, rfl, ?_, hq, ?_⟩
      · intro z hz
        rcases List.mem_cons.1 hz with rfl | hz
        · simp
        · exact hpre z hz
      · simp only [roundGo, hr, Except.map]

theorem full_stepP (p : List Nat × List Nat) (h : p.1 ≠ []) : full (stepP p) = full p := by
  obtain ⟨rem, read⟩ := p
  cases rem with
  | nil => exact absurd rfl h
  | cons x xs => simp [full, stepP]

theorem fewestLoop_spec : ∀ (fuel : Nat) (ps : List (List Nat × List Nat)) (k : Nat),
    (∀ p ∈ ps, p.2.length = k) → (∃ p ∈ ps, p.1.length < fuel) →
    fewestLoop fuel ps = minRow (ps.map full) := by
  intro fuel
  induction fuel with
  | zero => intro ps k _ ⟨p, _, hp⟩; omega
  | succ n ih =>
    intro ps k hk hf
    by_cases hE : ∃ p ∈ ps, p.1 = []
    · obtain ⟨pre, p, post, rfl, hpre, hp, hr⟩ := roundGo_error ps hE
      simp only [fewestLoop, hr]
      rw [List.map_append, List.map_cons]
      have hfull : full p = p.2 := by simp [full, hp]
      rw [← hfull]
      have hlq : ∀ q' ∈ pre ++ p :: post, (full q').length = k + q'.1.length := by
        intro q' hq'; simp only [full, List.length_append, hk q' hq'] <;> omega
      have hlp : (full p).length = k := by
        have := hlq p (by simp); simp [hp] at this; exact this
      symm
      apply minRow_split
      · intro q hq
        obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
        have hq1 := hpre q' hq'
        have : 0 < q'.1.length := by
          cases h : q'.1 with
          | nil => exact absurd h hq1
          | cons _ _ => simp
        have := hlq q' (by simp [hq'])
        omega
      · intro q hq
        obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
        have := hlq q' (by simp [hq'])
        omega
    · have hne : ∀ p ∈ ps, p.1 ≠ [] := fun p hp h => hE ⟨p, hp, h⟩
      simp only [fewestLoop, roundGo_ok ps hne]
      have hmap : (ps.map stepP).map full = ps.map full := by
        rw [List.map_map]
        apply List.map_congr_left
        intro p hp; exact full_stepP p (hne p hp)
      rw [← hmap]
      apply ih _ (k + 1)
      · intro p hp
        obtain ⟨p0, hp0, rfl⟩ := List.mem_map.1 hp
        have := hk p0 hp0
        obtain ⟨rem, read⟩ := p0
        cases rem with
        | nil => exact absurd rfl (hne _ hp0)
        | cons x xs => simp [stepP] at this ⊢; omega
      · obtain ⟨p0, hp0, hl⟩ := hf
        refine ⟨stepP p0, List.mem_map.2 ⟨p0, hp0, rfl⟩, ?_⟩
        obtain ⟨rem, read⟩ := p0
        cases rem with
        | nil => exact absurd rfl (hne _ hp0)
        | cons x xs => simp [stepP] at hl ⊢; omega

theorem le_foldl_max (xs : List Nat) : ∀ init : Nat, init ≤ xs.foldl max init ∧ ∀ x ∈ xs, x ≤ xs.foldl max init := by
  induction xs with
  | nil => intro init; simp
  | cons y ys ih =>
    intro init
    obtain ⟨h1, h2⟩ := ih (max init y)
    simp only [List.foldl_cons]
    refine ⟨by omega, fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · omega
    · exact h2 x hx

/-- **`fewest_candidates` returns the first shortest lookup's rows.** -/
theorem fewest_eq_minRow (rows : List (List Nat)) (h : rows ≠ []) : fewest rows = minRow rows := by
  unfold fewest
  match rows, h with
  | [r], _ => simp [minRow, minAcc]
  | r1 :: r2 :: rs, _ =>
    simp only
    have hle := (le_foldl_max (((r1 :: r2 :: rs).map List.length)) 0).2 r1.length (by simp)
    have := fewestLoop_spec ((((r1 :: r2 :: rs).map List.length).foldl max 0) + 1)
      ((r1 :: r2 :: rs).map (fun r => (r, []))) 0
      (by intro p hp; obtain ⟨r, _, rfl⟩ := List.mem_map.1 hp; rfl)
      ⟨(r1, []), by simp, by show r1.length < _; omega⟩
    rw [this]
    congr 1
    simp [List.map_map, Function.comp_def, full]

/-- The result is one of the lookups' rows. -/
theorem fewest_mem (rows : List (List Nat)) (h : rows ≠ []) : fewest rows ∈ rows := by
  rw [fewest_eq_minRow rows h]
  cases rows with
  | nil => exact absurd rfl h
  | cons r rs => exact minAcc_mem1 rs r

/-- ... no longer than any other: so it is the smallest candidate set, read at most `n` times over. -/
theorem fewest_minimal (rows : List (List Nat)) (h : rows ≠ []) :
    ∀ r ∈ rows, (fewest rows).length ≤ r.length := by
  rw [fewest_eq_minRow rows h]
  cases rows with
  | nil => exact absurd rfl h
  | cons r0 rs => exact fun r hr => minAcc_le rs r0 r hr

/-! ## `ReadOnce` (`unique.rs:40-60`) -/

/-- The lookups of a unique-constraint check are read at most once per set. `ids.sort_unstable()` makes
    the id list canonical, so the `HashSet<Vec<(usize, u64)>>` holds a list up to reordering: a
    permutation test. -/
abbrev LookupId := (EntityType × String × String) × Nat

structure ReadOnce where
  seen : List (List LookupId)

/-- `ReadOnce::candidates`: the candidates of `lookups` (`fewest_candidates`), or none when an earlier
    member had the same lookups; the set is recorded (`HashSet::insert`). -/
def ReadOnce.candidates (st : ReadOnce) (lookups : List EqualCandidates) : List Nat × ReadOnce :=
  let ids := lookups.map EqualCandidates.id
  if st.seen.any (fun s => decide (s.Perm ids)) then ([], st)
  else (fewest (lookups.map EqualCandidates.rows), ⟨ids :: st.seen⟩)

/-- A lookup set not seen before is read (the smallest lookup) and recorded. -/
theorem readOnce_first (st : ReadOnce) (lookups : List EqualCandidates)
    (h : ∀ s ∈ st.seen, ¬ s.Perm (lookups.map EqualCandidates.id)) :
    st.candidates lookups = (fewest (lookups.map EqualCandidates.rows), ⟨lookups.map EqualCandidates.id :: st.seen⟩) := by
  unfold ReadOnce.candidates
  have : st.seen.any (fun s => decide (s.Perm (lookups.map EqualCandidates.id))) = false := by
    rw [Bool.eq_false_iff]; intro hh
    obtain ⟨s, hs, hp⟩ := List.any_eq_true.1 hh
    exact h s hs (of_decide_eq_true hp)
  simp [this]

/-- **A set of lookups read once is not read again**: a member whose lookups are the same as an
    earlier member's (in any order) gets no candidates, and the state is unchanged. This is what keeps
    a bulk insert with a shared first property (a tenant) from being quadratic. -/
theorem readOnce_repeat (st : ReadOnce) (lookups : List EqualCandidates) (s : List LookupId) (hs : s ∈ st.seen)
    (hp : s.Perm (lookups.map EqualCandidates.id)) : st.candidates lookups = ([], st) := by
  unfold ReadOnce.candidates
  have : st.seen.any (fun s => decide (s.Perm (lookups.map EqualCandidates.id))) = true :=
    List.any_eq_true.2 ⟨s, hs, decide_eq_true hp⟩
  simp [this]

/-- After a call the lookup set is recorded, so the same set read again returns nothing. -/
theorem readOnce_twice (st : ReadOnce) (lookups : List EqualCandidates) :
    ((st.candidates lookups).2.candidates lookups).1 = [] := by
  by_cases h : ∃ s ∈ st.seen, s.Perm (lookups.map EqualCandidates.id)
  · obtain ⟨s, hs, hp⟩ := h
    rw [readOnce_repeat st lookups s hs hp]
    exact congrArg Prod.fst (readOnce_repeat st lookups s hs hp)
  · have h : ∀ s ∈ st.seen, ¬ s.Perm (lookups.map EqualCandidates.id) := fun s hs hp => h ⟨s, hs, hp⟩
    rw [readOnce_first st lookups h]
    exact congrArg Prod.fst (readOnce_repeat _ lookups _ (List.mem_cons_self ..) (List.Perm.refl _))

/-- A lookup comes from one index: its slot holds exactly its column. -/
def FromIndex (x : FalkorDbIndex) (c : EqualCandidates) : Prop :=
  ∃ st, x.entry c.entity c.label c.attr = some ⟨c.column, st⟩

theorem fromIndex_equalCandidates (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value)
    (c : EqualCandidates) (h : x.equalCandidates e l a v = some c) : FromIndex x c := by
  obtain ⟨en, hen, _, _, hc⟩ := (equalCandidates_some x e l a v c).1 h
  refine ⟨en.state, ?_⟩
  rw [hc]
  simpa using hen

/-- **Two lookups with the same id answer with the same rows** (`EqualCandidates::id` doc): same slot, so
    the same column, and the same key. This is why a repeat can be skipped. -/
theorem rows_eq_of_id_eq (x : FalkorDbIndex) (c c' : EqualCandidates) (h : FromIndex x c) (h' : FromIndex x c')
    (hid : c.id = c'.id) : c.rows = c'.rows := by
  obtain ⟨st, hs⟩ := h
  obtain ⟨st', hs'⟩ := h'
  simp only [EqualCandidates.id, Prod.mk.injEq] at hid
  obtain ⟨⟨he, hl, ha⟩, hk⟩ := hid
  rw [← he, ← hl, ← ha] at hs'
  rw [hs] at hs'
  have := (Option.some.inj hs')
  have hc : c.column = c'.column := by simpa using congrArg IndexEntry.index this
  simp only [EqualCandidates.rows, hc, hk]

/-- **A skipped member loses nothing**: if its lookups have the same ids as an earlier set's, every lookup
    of it has the same rows as one of the earlier set's, so whichever it would have read was an option
    for the earlier read too (`unique_violated` re-checks each candidate against the whole key, so any
    complete lookup suffices). -/
theorem readOnce_skip_sound (x : FalkorDbIndex) (L L' : List EqualCandidates) (hL : ∀ c ∈ L, FromIndex x c)
    (hL' : ∀ c ∈ L', FromIndex x c) (hp : (L.map EqualCandidates.id).Perm (L'.map EqualCandidates.id)) :
    ∀ c' ∈ L', ∃ c ∈ L, c.rows = c'.rows := by
  intro c' hc'
  have : c'.id ∈ L.map EqualCandidates.id := hp.symm.subset (List.mem_map.2 ⟨c', hc', rfl⟩)
  obtain ⟨c, hc, hid⟩ := List.mem_map.1 this
  exact ⟨c, hc, rows_eq_of_id_eq x c c' (hL c hc) (hL' c' hc') hid⟩

end NI
