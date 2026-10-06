import Columnar.Batch

/-
# The batched result emitter (`runtime/ops/batched_result_emitter.rs`)

The emitter walks the active rows of a parent batch; each row yields a list
of results (`RowIter` One / Spread / Many — only the item sequence matters); it
packs `(parent_row, item)` pairs, at most `pack_ceiling` per output batch, and
gathers the parent columns by the packed parent rows.

| here | there |
| --- | --- |
| `withBinding` | `BatchedResultEmitter::with_binding` (`batched_result_emitter.rs:553-568`) — **source semantics**; #2922 is an LLVM miscompile of exactly this `match` guard on aarch64-apple-darwin |
| `grow` | `pack_ceiling.saturating_mul(2).min(BATCH_SIZE)` (`:741`) |
| `drain` | `drain_pending_entry` (`:619-644`) |
| `packRows` | the `while count < pack_ceiling` loop with `refill_from_cursor` (`:673-714`, `:735-740`) |
| `emitLazy` | `emit_lazy` (`:726-743`) |
| `stream` | the flattened `(row, item)` sequence of `pending` + unread rows |
| `emitAll` | repeated `emit_lazy` until `Ok(None)` |
| `finishLen` | `finish_batch` (`:648-666`) row count incl. the `should_expand` switch (`start_batch`, `:602-614`) |
| `expandTrim` | `ExpandIntoOp::next` record-cap trim (`expand_into.rs:311-320`) |
-/

namespace Columnar

def BATCH_SIZE : Nat := 1024

/-- `with_binding`: `Some(cap) if cap < BATCH_SIZE => cap.max(1)`, else `BATCH_SIZE`. -/
def withBinding (recordCap : Option Nat) : Nat :=
  match recordCap with
  | some cap => if cap < BATCH_SIZE then max cap 1 else BATCH_SIZE
  | none => BATCH_SIZE

/-- The doubling after each `emit_lazy`. -/
def grow (c : Nat) : Nat := min (c * 2) BATCH_SIZE

/-- **The ceiling is always in `[1, BATCH_SIZE]`** (in the source; #2922's
miscompiled binary yields `cap` itself for `cap ≥ BATCH_SIZE`). -/
theorem withBinding_le_batch (rc : Option Nat) : 1 ≤ withBinding rc ∧ withBinding rc ≤ BATCH_SIZE := by
  cases rc with
  | none => simp [withBinding, BATCH_SIZE]
  | some cap => simp only [withBinding, BATCH_SIZE]; by_cases h : cap < 1024 <;> simp [h] <;> omega

theorem grow_bounds {c : Nat} (h : 1 ≤ c ∧ c ≤ BATCH_SIZE) : 1 ≤ grow c ∧ grow c ≤ BATCH_SIZE := by
  unfold grow BATCH_SIZE at *; omega

/-- A lowered ceiling reaches `BATCH_SIZE` again (here: 10 → … → 1024 in 7 steps). -/
theorem grow_reaches : Nat.repeat grow 7 (withBinding (some 10)) = BATCH_SIZE := by decide

/-- The miscompile's effect, for the record: the guard dropped, `cap` passes through. -/
def withBindingMiscompiled (recordCap : Option Nat) : Nat :=
  match recordCap with
  | some cap => max cap 1
  | none => BATCH_SIZE

theorem miscompiled_differs : withBindingMiscompiled (some 1025) ≠ withBinding (some 1025) := by decide

variable {I : Type}

/-- `drain_pending_entry` with `room = ceiling - count`: pull items while there
is room; `none` = the iterator was exhausted (`pending = None`). -/
def drain (row : Nat) : Nat → List I → List (Nat × I) × Option (List I)
  | 0, l => ([], some l)
  | _ + 1, [] => ([], none)
  | n + 1, x :: xs => let r := drain row n xs; ((row, x) :: r.1, r.2)

/-- The `emit_lazy` loop over unread rows (each row's result list). -/
def packRows : Nat → List (Nat × List I) → List (Nat × I) × Option (Nat × List I) × List (Nat × List I)
  | 0, t => ([], none, t)
  | _ + 1, [] => ([], none, [])
  | n + 1, (r, xs) :: t =>
    match drain r (n + 1) xs with
    | (o, some l) => (o, some (r, l), t)
    | (o, none) => let res := packRows (n + 1 - o.length) t; (o ++ res.1, res.2)

/-- `emit_lazy` with ceiling `c`: drain `pending` first, then refill from the cursor. -/
def emitLazy (c : Nat) (pending : Option (Nat × List I)) (t : List (Nat × List I)) :
    List (Nat × I) × Option (Nat × List I) × List (Nat × List I) :=
  match pending with
  | none => packRows c t
  | some (r, xs) =>
    match drain r c xs with
    | (o, some l) => (o, some (r, l), t)
    | (o, none) => let res := packRows (c - o.length) t; (o ++ res.1, res.2)

def rowItems (p : Nat × List I) : List (Nat × I) := p.2.map (p.1, ·)

/-- Everything still to be emitted, in order. -/
def stream (pending : Option (Nat × List I)) (t : List (Nat × List I)) : List (Nat × I) :=
  (match pending with | none => [] | some p => rowItems p) ++ t.flatMap rowItems

theorem drain_spec (row : Nat) :
    ∀ (n : Nat) (l : List I),
      (match (drain row n l).2 with | none => [] | some l' => l'.map (row, ·)) =
        ((l.map (row, ·)).drop (drain row n l).1.length) ∧
      (drain row n l).1 = (l.map (row, ·)).take (drain row n l).1.length ∧
      (drain row n l).1.length = min n l.length ∧
      ((drain row n l).2 = none → l.length ≤ n) ∧
      (∀ l', (drain row n l).2 = some l' → (drain row n l).1.length = n)
  | 0, l => by simp [drain]
  | n + 1, [] => by simp [drain]
  | n + 1, x :: xs => by
    obtain ⟨h1, h2, h3, h4, h5⟩ := drain_spec row n xs
    simp only [drain, List.length_cons, List.map_cons]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa using h1
    · simp only [List.take_succ_cons, List.cons.injEq, true_and]; exact h2
    · rw [h3]; omega
    · intro h; have := h4 h; omega
    · intro l' h; have := h5 l' h; omega

theorem drain_conserve (row n : Nat) (l : List I) :
    (drain row n l).1 ++ (match (drain row n l).2 with | none => [] | some l' => l'.map (row, ·)) =
      l.map (row, ·) := by
  obtain ⟨h1, h2, -, -, -⟩ := drain_spec row n l
  rw [h1]; conv => rhs; rw [← List.take_append_drop (drain row n l).1.length (l.map (row, ·))]
  rw [← h2]

theorem packRows_spec :
    ∀ (n : Nat) (t : List (Nat × List I)),
      (packRows n t).1 ++ stream (packRows n t).2.1 (packRows n t).2.2 = t.flatMap rowItems ∧
      (packRows n t).1.length = min n (t.flatMap rowItems).length
  | 0, t => by simp [packRows, stream]
  | n + 1, [] => by simp [packRows, stream]
  | n + 1, (r, xs) :: t => by
    have hc := drain_conserve r (n + 1) xs
    obtain ⟨-, -, hl, hn, hs⟩ := drain_spec r (n + 1) xs
    unfold packRows
    cases hd : drain r (n + 1) xs with
    | mk o rest =>
      rw [hd] at hc hl hn hs
      simp only at hc hl hn hs
      cases rest with
      | some l =>
        simp only
        have hon := hs l rfl
        refine ⟨?_, ?_⟩
        · simp only [stream, rowItems, List.flatMap_cons]
          rw [← List.append_assoc, hc]
        · simp only [List.flatMap_cons, rowItems, List.length_append, List.length_map]
          omega
      | none =>
        simp only at hc ⊢
        have hle := hn rfl
        obtain ⟨ih1, ih2⟩ := packRows_spec (n + 1 - o.length) t
        refine ⟨?_, ?_⟩
        · rw [List.append_assoc, ih1]
          simp only [List.flatMap_cons, rowItems]
          rw [List.append_nil] at hc; rw [hc]
        · rw [List.length_append, ih2, hl]
          simp only [List.flatMap_cons, rowItems, List.length_append, List.length_map]
          omega

/-- **`emit_lazy` conserves the stream**: the packed items followed by what is
left (pending + unread rows) is exactly what was there before — no item is
lost, duplicated or reordered — and the batch is as full as possible:
`min(ceiling, remaining)` items. -/
theorem emitLazy_spec (c : Nat) (p : Option (Nat × List I)) (t : List (Nat × List I)) :
    (emitLazy c p t).1 ++ stream (emitLazy c p t).2.1 (emitLazy c p t).2.2 = stream p t ∧
    (emitLazy c p t).1.length = min c (stream p t).length := by
  cases p with
  | none =>
    simp only [emitLazy]
    obtain ⟨h1, h2⟩ := packRows_spec c t
    exact ⟨by rw [h1]; simp [stream], by rw [h2]; simp [stream]⟩
  | some q =>
    obtain ⟨r, xs⟩ := q
    have hc := drain_conserve r c xs
    obtain ⟨-, -, hl, hn, hs⟩ := drain_spec r c xs
    simp only [emitLazy]
    cases hd : drain r c xs with
    | mk o rest =>
      rw [hd] at hc hl hn hs
      simp only at hc hl hn hs
      cases rest with
      | some l =>
        simp only
        have hon := hs l rfl
        refine ⟨?_, ?_⟩
        · simp only [stream, rowItems]; rw [← List.append_assoc, hc]
        · simp only [stream, rowItems, List.length_append, List.length_map]; omega
      | none =>
        simp only at hc ⊢
        have hle := hn rfl
        obtain ⟨ih1, ih2⟩ := packRows_spec (c - o.length) t
        refine ⟨?_, ?_⟩
        · rw [List.append_assoc, ih1]
          simp only [stream, rowItems]
          rw [List.append_nil] at hc; rw [hc]
        · rw [List.length_append, ih2, hl]
          simp only [stream, rowItems, List.length_append, List.length_map]
          omega

/-- Repeated `emit_lazy` until it returns `None` (count 0), with the ceiling
growing after every call; `fuel` bounds the calls. -/
def emitAll : Nat → Nat → Option (Nat × List I) → List (Nat × List I) → List (List (Nat × I))
  | 0, _, _, _ => []
  | fuel + 1, c, p, t =>
    let r := emitLazy c p t
    if r.1 = [] then [] else r.1 :: emitAll fuel (grow c) r.2.1 r.2.2

/-- **The emitted batches, concatenated, are the per-row expansion in order**
(`flat_map` over active rows of `(row, item)`), each batch is non-empty and has
at most `BATCH_SIZE` rows — given enough fuel (one call per item suffices). -/
theorem emitAll_spec :
    ∀ (fuel c : Nat) (p : Option (Nat × List I)) (t : List (Nat × List I)),
      1 ≤ c ∧ c ≤ BATCH_SIZE → (stream p t).length ≤ fuel →
      (emitAll fuel c p t).flatten = stream p t ∧
      ∀ bt ∈ emitAll fuel c p t, bt ≠ [] ∧ bt.length ≤ BATCH_SIZE
  | 0, c, p, t, _, hf => by
    have : stream p t = [] := List.eq_nil_of_length_eq_zero (by omega)
    simp [emitAll, this]
  | fuel + 1, c, p, t, hc, hf => by
    obtain ⟨h1, h2⟩ := emitLazy_spec c p t
    simp only [emitAll]
    split
    · rename_i he
      rw [he, List.length_nil] at h2
      have : (stream p t).length = 0 := by omega
      simp [List.eq_nil_of_length_eq_zero this]
    · rename_i hne
      have hpos : 0 < (emitLazy c p t).1.length := Nat.pos_of_ne_zero (fun h0 => hne (List.eq_nil_of_length_eq_zero h0))
      have hrest : (stream (emitLazy c p t).2.1 (emitLazy c p t).2.2).length ≤ fuel := by
        have := congrArg List.length h1
        rw [List.length_append] at this; omega
      obtain ⟨ih1, ih2⟩ := emitAll_spec fuel (grow c) _ _ (grow_bounds hc) hrest
      refine ⟨by rw [List.flatten_cons, ih1, h1], ?_⟩
      intro bt hbt
      rcases List.mem_cons.mp hbt with rfl | hbt
      · exact ⟨hne, by rw [h2]; omega⟩
      · exact ih2 bt hbt

/-- The rows a parent batch hands the emitter: each active row with its results. -/
def seedRows {F : Type} (b : Batch F) (f : Nat → List I) : List (Nat × List I) :=
  b.active.map (fun r => (r, f r))

/-- **Every packed parent index is an active (hence physical, `< len`) row**,
so the `gather` of `finish_batch` never panics. -/
theorem stream_rows_active {F : Type} {b : Batch F} (h : b.WF) (f : Nat → List I) :
    ∀ x ∈ stream none (seedRows b f), x.1 ∈ b.active ∧ x.1 < b.len := by
  intro x hx
  simp only [stream, seedRows, List.nil_append, List.mem_flatMap, List.mem_map, rowItems] at hx
  obtain ⟨_, ⟨r, hr, rfl⟩, ⟨y, _, rfl⟩⟩ := hx
  exact ⟨hr, Batch.active_lt h r hr⟩

/-- A sub-list of the stream (one emitted batch) also indexes active rows. -/
theorem batch_rows_active {F : Type} {b : Batch F} (h : b.WF) (f : Nat → List I)
    {bt : List (Nat × I)} (hsub : ∀ x ∈ bt, x ∈ stream none (seedRows b f)) :
    (b.gather (bt.map (·.1))).isSome :=
  Batch.gather_isSome h (fun i hi => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hi
    exact (stream_rows_active h f x (hsub x hx)).2)

/-- `finish_batch`'s row count: gathered when `start_batch` (`:602-614`) sets
`should_expand` — the parent has a column **or carries correlation origins**
(`has_origins`, since #2845) — else a fresh `Batch::new(0)` whose length comes from
the first installed column (`set_column` on an empty batch), or stays 0 when the
binding installs none. -/
def finishLen (parentCols : Nat) (hasOrigins : Bool) (bindsColumn : Bool) (count : Nat) : Nat :=
  if parentCols > 0 ∨ hasOrigins = true then count else if bindsColumn then count else 0

theorem finishLen_eq (pc : Nat) (ho bc : Bool) (count : Nat) (h : pc > 0 ∨ ho = true ∨ bc = true) :
    finishLen pc ho bc count = count := by
  unfold finishLen; split <;> simp_all

/-- Since #2845 (eb470521b) a column-less parent that carries origins — the entry
projection of a `CALL {}` body importing nothing — is gathered, so every result
keeps its row (and its origin; `proofs/pr2845_review` `emit_origin_sound`). Before
#2845 `should_expand` read only `num_columns() > 0` and this case was a 0-row batch. -/
theorem finishLen_origins_kept : finishLen 0 true false 5 = 5 := rfl

/-- **Latent row loss**: a no-alias emitter over a parent with no column and no
origins emits `count` results as a 0-row batch. The only no-alias user, ExpandInto
(`expand_into.rs:96`), always has both endpoint columns bound, so this is not
reachable today. -/
theorem finishLen_drops : finishLen 0 false false 5 = 0 := rfl

/-! ## ExpandInto's record-cap trim -/

/-- `ExpandIntoOp::next` (`expand_into.rs:313-319`) on a dense output batch of
`n` rows: `(new selection, rows counted)`. -/
def expandTrim (cap produced n : Nat) : Option (List Nat) × Nat :=
  let remaining := cap - produced
  if n > remaining then (some ((List.range remaining).map Batch.toU16), produced + remaining)
  else (none, produced + n)

/-- **The trim suspicion is refuted**: the emitted batch is dense with `n` rows,
`n ≤ BATCH_SIZE` (or, under the #2922 miscompile, the first batch has
`n ≤ cap = remaining` so the trim does not fire). Whenever the trim fires, every
selected index is `< n` and survives the `u16` cast, the selection is the first
`remaining` rows, and `produced` never exceeds `cap`. -/
theorem expandTrim_ok {F : Type} (b : Batch F) (hw : b.WF) (hs : b.sel = none)
    (cap produced : Nat) (hp : produced ≤ cap)
    (hn : b.len ≤ BATCH_SIZE ∨ b.len ≤ cap - produced) :
    (∀ s, (expandTrim cap produced b.len).1 = some s →
      s = List.range (cap - produced) ∧ (b.setSel s).WF ∧
      (b.setSel s).active = b.active.take (cap - produced)) ∧
    (expandTrim cap produced b.len).2 ≤ cap := by
  have hact : b.active = List.range b.len := by simp [Batch.active, hs]
  unfold expandTrim
  simp only
  split
  · rename_i hgt
    have hsmall : cap - produced < 65536 := by unfold BATCH_SIZE at hn; omega
    have e : (List.range (cap - produced)).map Batch.toU16 = List.range (cap - produced) := by
      conv => rhs; rw [← List.map_id (List.range (cap - produced))]
      apply List.map_congr_left
      intro i hi
      exact Batch.toU16_id (by simp at hi; omega)
    refine ⟨fun s hs' => ?_, by omega⟩
    simp only [Option.some.injEq] at hs'
    subst hs'
    rw [e]
    have htake : b.active.take (cap - produced) = List.range (cap - produced) := by
      rw [hact, List.take_range]; congr 1; omega
    refine ⟨rfl, ?_, ?_⟩
    · apply Batch.setSel_wf hw
      rw [← htake]; exact List.take_sublist _ _
    · rw [htake]; rfl
  · refine ⟨fun s hs' => by simp at hs', by omega⟩

end Columnar
