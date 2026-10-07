/-
# `BatchedResultEmitter`: pack-and-gather of per-row expansions

| here | there |
| --- | --- |
| `RowIter`, `RowIter.next` | `RowIter::{One,Spread,Many}`, `Iterator for RowIter`, `ops/batched_result_emitter.rs:455-488` |
| `St`                      | `BatchedResultEmitter { batch, pending, cursor, pack_ceiling }`, `:491-531`; `rows` = the batch's active rows from `cursor` on |
| `initCeiling`             | `with_binding`, `:553-569` |
| `refill`                  | `refill_from_cursor`, `:673-720` |
| `drain`                   | `drain_pending_entry`, `:619-644` |
| `loop`, `emit`            | `emit_lazy`, `:726-742` (loop + ceiling doubling) |
| `seed`, `reset`           | `seed`, `:575-586`; `reset`, `:747-751` |
| `drive`                   | the operator's pull loop: `emit_lazy` until `None`, then pull the next child batch and `seed` (e.g. `ExpandIntoOp::next`, `ops/expand_into.rs:258-304`) |

Abstractions. Output batches are the list of `(parent_row, item)` pairs that
`finish_batch` gathers (`batch.gather(indices)` replicates parent row `indices[i]` for
item `i`); the typed-lane `GatherItem` impls are not modelled. The per-row closure `f` is
pure (its `Err` short-circuit is not modelled). `emit_lazy`'s `while` loop is given fuel
`2·|rows| + 2`, which `loop_progress` shows is always enough.
-/
namespace RuntimeCore.Emitter

def BATCH_SIZE : Nat := 1024

inductive RowIter (β : Type) where
  | one (x : Option β)
  | spread (xs : List β)
  | many (xs : List β)

variable {β : Type}

def RowIter.toList : RowIter β → List β
  | .one x => x.toList
  | .spread xs => xs
  | .many xs => xs

/-- `Iterator for RowIter::next` (`:478-487`). -/
def RowIter.next : RowIter β → Option β × RowIter β
  | .many [] => (none, .many [])
  | .many (x :: xs) => (some x, .many xs)
  | .spread [] => (none, .spread [])
  | .spread (x :: xs) => (some x, .spread xs)
  | .one x => (x, .one none)

theorem next_none {it it' : RowIter β} (h : it.next = (none, it')) : it.toList = [] := by
  cases it with
  | one x => cases x <;> simp_all [RowIter.next, RowIter.toList]
  | spread xs => cases xs <;> simp_all [RowIter.next, RowIter.toList]
  | many xs => cases xs <;> simp_all [RowIter.next, RowIter.toList]

theorem next_some {it it' : RowIter β} {x : β} (h : it.next = (some x, it')) :
    it.toList = x :: it'.toList := by
  cases it with
  | one y => cases y <;> simp_all [RowIter.next, RowIter.toList] <;> (obtain ⟨rfl, rfl⟩ := h; rfl)
  | spread xs => cases xs <;> simp_all [RowIter.next, RowIter.toList] <;> (obtain ⟨rfl, rfl⟩ := h; simp [RowIter.toList])
  | many xs => cases xs <;> simp_all [RowIter.next, RowIter.toList] <;> (obtain ⟨rfl, rfl⟩ := h; simp [RowIter.toList])

structure St (β : Type) where
  rows : List Nat
  pending : Option (Nat × RowIter β)
  ceiling : Nat

/-- `with_binding` (`:553-569`). -/
def initCeiling : Option Nat → Nat
  | some c => if c < BATCH_SIZE then max c 1 else BATCH_SIZE
  | none => BATCH_SIZE

/-- `refill_from_cursor` (`:673-720`). -/
def refill (f : Nat → Option (RowIter β)) : List Nat → Option (Nat × RowIter β) × List Nat
  | [] => (none, [])
  | r :: rs =>
    match f r with
    | none => refill f rs
    | some (.spread []) => refill f rs
    | some (.spread [x]) => (some (r, .one (some x)), rs)
    | some it => (some (r, it), rs)

/-- `drain_pending_entry` (`:619-644`): `k = ceiling - count` slots left. `none` in the
second component is `self.pending = None` (the iterator was seen to be exhausted). -/
def drain (row : Nat) : Nat → RowIter β → List (Nat × β) × Option (RowIter β)
  | 0, it => ([], some it)
  | k + 1, it =>
    match it.next with
    | (none, _) => ([], none)
    | (some x, it') =>
      let r := drain row k it'
      ((row, x) :: r.1, r.2)

/-- The `while count < self.pack_ceiling` loop of `emit_lazy` (`:732-738`). -/
def loop (f : Nat → Option (RowIter β)) (ceil : Nat) :
    Nat → Nat → List (Nat × β) → St β → List (Nat × β) × St β
  | 0, _, acc, s => (acc, s)
  | fuel + 1, count, acc, s =>
    if count < ceil then
      match s.pending with
      | none =>
        match refill f s.rows with
        | (none, rows') => (acc, { s with rows := rows' })
        | (some (r, it), rows') =>
          let d := drain r (ceil - count) it
          loop f ceil fuel (count + d.1.length) (acc ++ d.1)
            { s with rows := rows', pending := d.2.map (fun it => (r, it)) }
      | some (r, it) =>
        let d := drain r (ceil - count) it
        loop f ceil fuel (count + d.1.length) (acc ++ d.1)
          { s with pending := d.2.map (fun it => (r, it)) }
    else (acc, s)

/-- `emit_lazy` (`:726-742`): pack, double the ceiling, `None` when nothing was packed. -/
def emit (f : Nat → Option (RowIter β)) (s : St β) : Option (List (Nat × β)) × St β :=
  let r := loop f s.ceiling (2 * s.rows.length + 2) 0 [] s
  let s' := { r.2 with ceiling := min (s.ceiling * 2) BATCH_SIZE }
  (if r.1 = [] then none else some r.1, s')

/-- `seed` (`:575-586`) installs a new parent batch and resets the cursor; `pending` is
left alone (the Rust `debug_assert!`s it is empty — `emit_none_pending` shows the
remaining work there is empty whenever `seed` is called). -/
def seed (s : St β) (rows : List Nat) : St β := { s with rows := rows }

/-- `reset` (`:747-751`). -/
def reset (s : St β) : St β := { s with rows := [], pending := none }

/-! ### What is left to emit -/

def rowOut (f : Nat → Option (RowIter β)) (r : Nat) : List (Nat × β) :=
  (((f r).map RowIter.toList).getD []).map (fun x => (r, x))

def flatRows (f : Nat → Option (RowIter β)) (rows : List Nat) : List (Nat × β) :=
  rows.flatMap (rowOut f)

def pendOut : Option (Nat × RowIter β) → List (Nat × β)
  | none => []
  | some (r, it) => it.toList.map (fun x => (r, x))

theorem pendOut_none : pendOut (none : Option (Nat × RowIter β)) = [] := rfl

def remaining (f : Nat → Option (RowIter β)) (s : St β) : List (Nat × β) :=
  pendOut s.pending ++ flatRows f s.rows

theorem refill_none {f : Nat → Option (RowIter β)} :
    ∀ {rows rows'}, refill f rows = (none, rows') → rows' = [] ∧ flatRows f rows = []
  | [], _, h => by simp [refill] at h; simp [h, flatRows]
  | r :: rs, rows', h => by
    simp only [refill] at h
    split at h
    · next hf =>
      obtain ⟨h1, h2⟩ := refill_none h
      refine ⟨h1, ?_⟩
      simp only [flatRows, List.flatMap_cons] at h2 ⊢
      simp [rowOut, hf, h2]
    · next hf =>
      obtain ⟨h1, h2⟩ := refill_none h
      refine ⟨h1, ?_⟩
      simp only [flatRows, List.flatMap_cons] at h2 ⊢
      simp [rowOut, hf, h2, RowIter.toList]
    · simp at h
    · simp at h

theorem refill_some {f : Nat → Option (RowIter β)} :
    ∀ {rows rows' r it}, refill f rows = (some (r, it), rows') →
      flatRows f rows = pendOut (some (r, it)) ++ flatRows f rows' ∧ rows'.length < rows.length
  | [], _, _, _, h => by simp [refill] at h
  | r0 :: rs, rows', r, it, h => by
    simp only [refill] at h
    split at h
    · next hf =>
      obtain ⟨h1, h2⟩ := refill_some h
      refine ⟨?_, by simp; omega⟩
      simp only [flatRows, List.flatMap_cons] at h1 ⊢
      simp [rowOut, hf, h1]
    · next hf =>
      obtain ⟨h1, h2⟩ := refill_some h
      refine ⟨?_, by simp; omega⟩
      simp only [flatRows, List.flatMap_cons] at h1 ⊢
      simp [rowOut, hf, h1, RowIter.toList]
    · next x hf =>
      simp at h; obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
      refine ⟨?_, by simp⟩
      simp [flatRows, rowOut, hf, pendOut, RowIter.toList]
    · next _ it1 _ _ hfr =>
      simp at h; obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
      refine ⟨?_, by simp⟩
      simp [flatRows, rowOut, hfr, pendOut]

theorem drain_spec (row : Nat) :
    ∀ (k : Nat) (it : RowIter β),
      (drain row k it).1 ++ pendOut ((drain row k it).2.map (fun it => (row, it))) =
        it.toList.map (fun x => (row, x)) ∧ (drain row k it).1.length ≤ k
  | 0, it => by simp [drain, pendOut]
  | k + 1, it => by
    simp only [drain]
    split
    · next it' h => simp [pendOut, next_none h]
    · next x it' h =>
      obtain ⟨h1, h2⟩ := drain_spec row k it'
      refine ⟨?_, by simp; omega⟩
      simp only [List.cons_append, h1, next_some h, List.map_cons]

theorem drain_nonempty (row : Nat) (k : Nat) (it : RowIter β) (hk : 0 < k) (x : β) (xs : List β)
    (h : it.toList = x :: xs) : ∃ y ys, (drain row k it).1 = y :: ys := by
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  simp only [drain]
  split
  · next it' hn => rw [next_none hn] at h; simp at h
  · exact ⟨_, _, rfl⟩

theorem drain_empty (row : Nat) (k : Nat) (it : RowIter β) (hk : 0 < k) (h : it.toList = []) :
    drain row k it = ([], none) := by
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  simp only [drain]
  split
  · rfl
  · next x it' hs => rw [next_some hs] at h; simp at h

/-- **PROVEN** (loop invariant): whatever `emit_lazy`'s loop packs, followed by what is
left, is exactly what was left before — nothing lost, duplicated or reordered. -/
theorem loop_inv (f : Nat → Option (RowIter β)) (ceil : Nat) :
    ∀ (fuel count : Nat) (acc : List (Nat × β)) (s : St β),
      (loop f ceil fuel count acc s).1 ++ remaining f (loop f ceil fuel count acc s).2 =
        acc ++ remaining f s
  | 0, _, _, _ => rfl
  | fuel + 1, count, acc, s => by
    simp only [loop]
    split
    · split
      · next hp =>
        split
        · next rows' hr =>
          obtain ⟨rfl, h2⟩ := refill_none hr
          simp only [remaining, hp, h2, pendOut_none]; rfl
        · next r it rows' hr =>
          rw [loop_inv f ceil fuel]
          obtain ⟨h1, -⟩ := refill_some hr
          obtain ⟨d1, -⟩ := drain_spec r (ceil - count) it
          simp only [remaining, hp, pendOut_none, List.nil_append, h1, List.append_assoc]
          rw [← List.append_assoc (drain r (ceil - count) it).1, d1]
          rfl
      · next r it hp =>
        rw [loop_inv f ceil fuel]
        obtain ⟨d1, -⟩ := drain_spec r (ceil - count) it
        simp only [remaining, hp, List.append_assoc]
        rw [← List.append_assoc (drain r (ceil - count) it).1, d1]
        rfl
    · rfl

/-- **PROVEN** (batch bound): one `emit_lazy` never packs more than the ceiling. -/
theorem loop_len (f : Nat → Option (RowIter β)) (ceil : Nat) :
    ∀ (fuel count : Nat) (acc : List (Nat × β)) (s : St β), acc.length = count → count ≤ ceil →
      (loop f ceil fuel count acc s).1.length ≤ ceil
  | 0, _, _, _, h1, h2 => by simp [loop]; omega
  | fuel + 1, count, acc, s, h1, h2 => by
    simp only [loop]
    split
    · split
      · split
        · simp; omega
        · next r it rows' _ =>
          have := (drain_spec r (ceil - count) it).2
          exact loop_len f ceil fuel _ _ _ (by simp; omega) (by omega)
      · next r it _ =>
        have := (drain_spec r (ceil - count) it).2
        exact loop_len f ceil fuel _ _ _ (by simp; omega) (by omega)
    · simp; omega

theorem loop_prefix (f : Nat → Option (RowIter β)) (ceil : Nat) :
    ∀ (fuel count : Nat) (acc : List (Nat × β)) (s : St β),
      ∃ out, (loop f ceil fuel count acc s).1 = acc ++ out
  | 0, _, acc, _ => ⟨[], by simp [loop]⟩
  | fuel + 1, count, acc, s => by
    simp only [loop]
    split
    · split
      · split
        · exact ⟨[], by simp⟩
        · next r it rows' _ =>
          obtain ⟨o, ho⟩ := loop_prefix f ceil fuel (count + (drain r (ceil - count) it).1.length)
            (acc ++ (drain r (ceil - count) it).1)
            { s with rows := rows', pending := (drain r (ceil - count) it).2.map (fun it => (r, it)) }
          exact ⟨(drain r (ceil - count) it).1 ++ o, by rw [ho, List.append_assoc]⟩
      · next r it _ =>
        obtain ⟨o, ho⟩ := loop_prefix f ceil fuel (count + (drain r (ceil - count) it).1.length)
          (acc ++ (drain r (ceil - count) it).1)
          { s with pending := (drain r (ceil - count) it).2.map (fun it => (r, it)) }
        exact ⟨(drain r (ceil - count) it).1 ++ o, by rw [ho, List.append_assoc]⟩
    · exact ⟨[], by simp⟩

def fuelNeed (s : St β) : Nat := 2 * s.rows.length + (if s.pending.isSome then 2 else 1)

/-- **PROVEN** (progress): with work left and room in the batch, the loop packs at least
one item, given fuel `fuelNeed` (≤ the `2·|rows|+2` that `emit` supplies). -/
theorem loop_progress (f : Nat → Option (RowIter β)) (ceil : Nat) :
    ∀ (fuel count : Nat) (acc : List (Nat × β)) (s : St β),
      count < ceil → remaining f s ≠ [] → fuelNeed s ≤ fuel →
      ∃ y ys, (loop f ceil fuel count acc s).1 = acc ++ y :: ys
  | 0, _, _, s, _, _, hf => by simp [fuelNeed] at hf; split at hf <;> omega
  | fuel + 1, count, acc, s, hc, hrem, hf => by
    simp only [loop, hc, ite_true]
    split
    · next hp =>
      split
      · next rows' hr =>
        exfalso; apply hrem
        simp [remaining, hp, pendOut, (refill_none hr).2]
      · next r it rows' hr =>
        obtain ⟨h1, hlen⟩ := refill_some hr
        cases hit : it.toList with
        | cons x xs =>
          obtain ⟨y, ys, hy⟩ := drain_nonempty r (ceil - count) it (by omega) x xs hit
          obtain ⟨o, ho⟩ := loop_prefix f ceil fuel (count + (drain r (ceil - count) it).1.length)
            (acc ++ (drain r (ceil - count) it).1)
            { s with rows := rows', pending := (drain r (ceil - count) it).2.map (fun it => (r, it)) }
          exact ⟨y, ys ++ o, by rw [ho, hy]; simp⟩
        | nil =>
          have hd := drain_empty r (ceil - count) it (by omega) hit
          simp only [hd, List.length_nil, Nat.add_zero, List.append_nil, Option.map_none]
          apply loop_progress f ceil fuel count acc _ hc
          · intro h; apply hrem
            simp only [remaining, hp, pendOut, List.nil_append, h1, hit, List.map_nil]
            simpa [remaining, pendOut] using h
          · simp [fuelNeed, hp] at hf ⊢; omega
    · next r it hp =>
      cases hit : it.toList with
      | cons x xs =>
        obtain ⟨y, ys, hy⟩ := drain_nonempty r (ceil - count) it (by omega) x xs hit
        obtain ⟨o, ho⟩ := loop_prefix f ceil fuel (count + (drain r (ceil - count) it).1.length)
          (acc ++ (drain r (ceil - count) it).1)
          { s with pending := (drain r (ceil - count) it).2.map (fun it => (r, it)) }
        exact ⟨y, ys ++ o, by rw [ho, hy]; simp⟩
      | nil =>
        have hd := drain_empty r (ceil - count) it (by omega) hit
        simp only [hd, List.length_nil, Nat.add_zero, List.append_nil, Option.map_none]
        apply loop_progress f ceil fuel count acc _ hc
        · intro h; apply hrem
          simp only [remaining, hp, pendOut, hit, List.map_nil, List.nil_append]
          simpa [remaining, pendOut] using h
        · simp [fuelNeed, hp] at hf ⊢; omega

/-! ### One `emit_lazy` call -/

theorem emit_inv (f : Nat → Option (RowIter β)) (s : St β) :
    ((emit f s).1.getD []) ++ remaining f (emit f s).2 = remaining f s := by
  have h := loop_inv f s.ceiling (2 * s.rows.length + 2) 0 [] s
  simp only [List.nil_append] at h
  simp only [emit]
  split
  · next he => rw [he] at h; simpa [remaining] using h
  · simpa [remaining] using h

/-- **PROVEN**: `emit_lazy` returns `None` only when nothing is left to emit (given a
ceiling ≥ 1, which `with_binding` guarantees — `initCeiling_pos`). -/
theorem emit_none_remaining (f : Nat → Option (RowIter β)) (s : St β) (hc : 1 ≤ s.ceiling)
    (h : (emit f s).1 = none) : remaining f s = [] ∧ remaining f (emit f s).2 = [] := by
  have hinv := emit_inv f s
  simp only [h, Option.getD_none, List.nil_append] at hinv
  have hs : remaining f s = [] := by
    apply Classical.byContradiction; intro hne
    obtain ⟨y, ys, hy⟩ := loop_progress f s.ceiling (2 * s.rows.length + 2) 0 [] s hc hne
      (by simp [fuelNeed]; split <;> omega)
    simp only [emit] at h
    rw [hy] at h
    simp at h
  exact ⟨hs, by rw [hinv]; exact hs⟩

theorem emit_some_ne (f : Nat → Option (RowIter β)) (s : St β) (out : List (Nat × β))
    (h : (emit f s).1 = some out) : out ≠ [] := by
  simp only [emit] at h; split at h <;> simp_all

/-- **PROVEN**: every packed batch holds at most the current ceiling (≤ `BATCH_SIZE`). -/
theorem emit_len (f : Nat → Option (RowIter β)) (s : St β) (out : List (Nat × β))
    (h : (emit f s).1 = some out) : out.length ≤ s.ceiling := by
  simp only [emit] at h; split at h
  · simp at h
  · simp at h; subst h
    exact loop_len f s.ceiling _ 0 [] s rfl (Nat.zero_le _)

theorem initCeiling_pos (c : Option Nat) : 1 ≤ initCeiling c ∧ initCeiling c ≤ BATCH_SIZE := by
  cases c with
  | none => simp [initCeiling, BATCH_SIZE]
  | some c => simp only [initCeiling]; split <;> simp [BATCH_SIZE] at * <;> omega

/-- **PROVEN** (ceiling schedule): the ceiling stays in `[1, BATCH_SIZE]` and doubles. -/
theorem emit_ceiling (f : Nat → Option (RowIter β)) (s : St β) :
    (emit f s).2.ceiling = min (s.ceiling * 2) BATCH_SIZE := rfl

theorem ceiling_bounds (c : Nat) (h1 : 1 ≤ c) (h2 : c ≤ BATCH_SIZE) :
    1 ≤ min (c * 2) BATCH_SIZE ∧ min (c * 2) BATCH_SIZE ≤ BATCH_SIZE := by
  simp [BATCH_SIZE] at *; omega

/-! ### The operator's pull loop over all child batches -/

/-- Pull loop: `emit_lazy` until `None`, then `seed` the next child batch. -/
def drive (f : Nat → Option (RowIter β)) : Nat → St β → List (List Nat) → List (List (Nat × β))
  | 0, _, _ => []
  | fuel + 1, s, bs =>
    match emit f s with
    | (some out, s') => out :: drive f fuel s' bs
    | (none, s') =>
      match bs with
      | [] => []
      | b :: bs' => drive f fuel (seed s' b) bs'

def flatAll (f : Nat → Option (RowIter β)) (bs : List (List Nat)) : List (Nat × β) :=
  bs.flatMap (flatRows f)

theorem remaining_seed (f : Nat → Option (RowIter β)) (s : St β) (b : List Nat)
    (h : remaining f s = []) : remaining f (seed s b) = flatRows f b := by
  simp only [remaining, List.append_eq_nil_iff] at h
  simp [remaining, seed, h.1]

/-- **PROVEN** (headline, emitter correctness): the batches an emitter-driven operator
yields, concatenated, are exactly the per-row expansions of every active input row, in
order — for any initial `record_cap`. So the pack ceiling is only a batch-size hint and
cannot change a query's rows (contrast `Budget.hard_cap_unsound_*`). -/
theorem drive_flatten (f : Nat → Option (RowIter β)) :
    ∀ (fuel : Nat) (s : St β) (bs : List (List Nat)),
      1 ≤ s.ceiling → s.ceiling ≤ BATCH_SIZE →
      (remaining f s).length + (flatAll f bs).length + bs.length + 1 ≤ fuel →
      (drive f fuel s bs).flatten = remaining f s ++ flatAll f bs
  | 0, _, _, _, _, h => by omega
  | fuel + 1, s, bs, h1, h2, hf => by
    have hinv := emit_inv f s
    obtain ⟨c1, c2⟩ := ceiling_bounds s.ceiling h1 h2
    simp only [drive]
    split
    · next out s' he =>
      have hne := emit_some_ne f s out (by rw [he])
      rw [he] at hinv
      simp only [Option.getD_some] at hinv
      have hc : s'.ceiling = min (s.ceiling * 2) BATCH_SIZE := by
        have := emit_ceiling f s; rw [he] at this; exact this
      have hl : (remaining f s').length < (remaining f s).length := by
        rw [← hinv, List.length_append]
        have : out.length ≠ 0 := by simpa using hne
        omega
      rw [List.flatten_cons, drive_flatten f fuel s' bs (by omega) (by omega) (by omega), ← hinv,
        List.append_assoc]
    · next s' he =>
      have hn := emit_none_remaining f s h1 (by rw [he])
      rw [he] at hn
      have hc : s'.ceiling = min (s.ceiling * 2) BATCH_SIZE := by
        have := emit_ceiling f s; rw [he] at this; exact this
      split
      · simp [hn.1, flatAll]
      · next b bs' =>
        rw [drive_flatten f fuel (seed s' b) bs' (by simp [seed]; omega) (by simp [seed]; omega)]
        · rw [remaining_seed f s' b hn.2, hn.1]; simp [flatAll]
        · rw [remaining_seed f s' b hn.2]
          simp [flatAll, hn.1] at hf ⊢; omega

/-- Corollary for a freshly constructed emitter fed batches `bs`. -/
theorem fresh_drive (f : Nat → Option (RowIter β)) (cap : Option Nat) (bs : List (List Nat))
    (fuel : Nat) (hf : (flatAll f bs).length + bs.length + 1 ≤ fuel) :
    (drive f fuel ⟨[], none, initCeiling cap⟩ bs).flatten = flatAll f bs := by
  have := drive_flatten f fuel ⟨[], none, initCeiling cap⟩ bs (initCeiling_pos cap).1
    (initCeiling_pos cap).2 (by simp [remaining, pendOut, flatRows]; omega)
  simpa [remaining, pendOut, flatRows] using this

/-- `reset` drops all queued work (`Apply` re-seeding cannot replay stale rows). -/
theorem reset_remaining (f : Nat → Option (RowIter β)) (s : St β) :
    remaining f (reset s) = [] := by simp [reset, remaining, pendOut, flatRows]

/-- The `zero_cap_clamps_to_one` / `cap_at_or_above_batch_size_is_ignored` unit tests
(`:819-844`) as theorems. -/
theorem initCeiling_cases :
    initCeiling (some 0) = 1 ∧ initCeiling (some 10) = 10 ∧ initCeiling (some 1024) = 1024 ∧
    initCeiling (some 5000) = 1024 ∧ initCeiling none = 1024 := by decide

end RuntimeCore.Emitter
