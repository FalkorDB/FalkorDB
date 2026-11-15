/-
# FOREACH — `graph/src/runtime/ops/foreach.rs`

`ForEachOp::next` (`foreach.rs:124-208`) pulls input batches, and for every
active row: evaluates the list expression, runs the body sub-plan over the list
items in chunks of at most `BATCH_SIZE` loop envs (`execute_list`,
`foreach.rs:86-118`), and queues the row unchanged in `pending`; queued rows are
emitted in batches of at most `BATCH_SIZE` (`drain_pending`, `ops/mod.rs:127`).

The model keeps Rust's state exactly (`pending`, `current_batch` + `current_pos`
as the remaining suffix `cur`, the child iterator as the list of its remaining
results) and adds a ghost `log` of body invocations (the side effects).  The
headline theorem `run_ok`, for any batch size `B > 0` and no errors:

* the concatenation of all emitted batches is the input, row for row (FOREACH is
  a pass-through), every emitted batch has `1 ≤ size ≤ B`, and
* the body was invoked exactly on `calls input` — every list item of every row,
  in row order and item order, in chunks of size `≤ B` (`chunks_flatten`,
  `chunks_size`); a NULL list runs nothing (`foreach.rs:173-175`).
-/

namespace ForEach

variable {α ι : Type}

/-- `execute_list`'s chunking loop (`foreach.rs:104-116`): `drain(..BATCH_SIZE.min(len))`. -/
def chunks (B : Nat) (hB : 0 < B) (l : List ι) : List (List ι) :=
  if h : l = [] then [] else l.take B :: chunks B hB (l.drop B)
termination_by l.length
decreasing_by
  simp only [List.length_drop]
  have : l.length ≠ 0 := by simpa using h
  omega

theorem chunks_nil (B : Nat) (hB : 0 < B) : chunks B hB ([] : List ι) = [] := by
  rw [chunks]; simp

theorem chunks_flatten (B : Nat) (hB : 0 < B) (l : List ι) : (chunks B hB l).flatten = l := by
  induction l using chunks.induct B hB with
  | case1 => simp [chunks_nil]
  | case2 l h ih => rw [chunks, dif_neg h]; simp [ih]

theorem chunks_size (B : Nat) (hB : 0 < B) (l : List ι) :
    ∀ c ∈ chunks B hB l, 0 < c.length ∧ c.length ≤ B := by
  induction l using chunks.induct B hB with
  | case1 => simp [chunks_nil]
  | case2 l h ih =>
    rw [chunks, dif_neg h]
    intro c hc
    simp only [List.mem_cons] at hc
    rcases hc with rfl | hc
    · have : l.length ≠ 0 := by simpa using h
      simp; omega
    · exact ih c hc

/-- The environment of one FOREACH clause. -/
structure Env (α ι : Type) where
  B : Nat
  hB : 0 < B
  /-- list expression: `ok (some items)` (a List), `ok none` (Null), or an error
  (evaluation error or `Type mismatch: expected List`, `foreach.rs:156-181`). -/
  ev : α → Except String (Option (List ι))
  /-- running the body over one chunk of loop envs: `some e` if it failed (`result?`). -/
  body : α → List ι → Option String

variable (E : Env α ι)

/-- Running the body on the chunks in order, stopping at the first failure
(`foreach.rs:104-116`). -/
def runChunks (x : α) : List (List ι) → List (α × List ι) × Option String
  | [] => ([], none)
  | c :: cs => match E.body x c with
    | some e => ([(x, c)], some e)
    | none => let r := runChunks x cs; ((x, c) :: r.1, r.2)

/-- `execute_list` (`foreach.rs:86-118`). -/
def execList (x : α) (items : List ι) : List (α × List ι) × Option String :=
  if items.isEmpty then ([], none) else runChunks E x (chunks E.B E.hB items)

/-- The inner `while self.current_pos < active.len()` loop (`foreach.rs:150-190`).
Returns the unprocessed rows, the new pending queue, the log and an error. -/
def inner (pending : List α) (log : List (α × List ι)) :
    List α → List α × List α × List (α × List ι) × Option String
  | [] => ([], pending, log, none)
  | x :: xs =>
    match E.ev x with
    | .error e => (xs, pending, log, some e)
    | .ok none =>
      let p := pending ++ [x]
      if E.B ≤ p.length then (xs, p, log, none) else inner p log xs
    | .ok (some items) =>
      let r := execList E x items
      match r.2 with
      | some e => (xs, pending, log ++ r.1, some e)
      | none =>
        let p := pending ++ [x]
        if E.B ≤ p.length then (xs, p, log ++ r.1, none) else inner p (log ++ r.1) xs

theorem inner_rest_le (pending : List α) (log : List (α × List ι)) (rows : List α) :
    (inner E pending log rows).1.length ≤ rows.length := by
  induction rows generalizing pending log with
  | nil => simp [inner]
  | cons x xs ih =>
    simp only [inner]
    split
    · simp
    · split
      · simp
      · exact Nat.le_succ_of_le (ih _ _)
    · split
      · simp
      · split
        · simp
        · exact Nat.le_succ_of_le (ih _ _)

theorem inner_cons_le (pending : List α) (log : List (α × List ι)) (x : α) (xs : List α) :
    (inner E pending log (x :: xs)).1.length ≤ xs.length := by
  simp only [inner]
  split
  · simp
  · split
    · simp
    · exact inner_rest_le E _ _ xs
  · split
    · simp
    · split
      · simp
      · exact inner_rest_le E _ _ xs

theorem inner_rest_lt (pending : List α) (log : List (α × List ι)) (rows : List α)
    (h : (inner E pending log rows).1 ≠ []) : (inner E pending log rows).1.length < rows.length := by
  cases rows with
  | nil => simp [inner] at h
  | cons x xs => have := inner_cons_le E pending log x xs; simp; omega

/-- `drain_pending` (`ops/mod.rs:127-138`): move rows from the front of `pending`
into the builder until it holds `B` rows. -/
def drainP (builder pending : List α) : List α × List α :=
  (builder ++ pending.take (E.B - builder.length), pending.drop (E.B - builder.length))

/-- Operator state: `pending`, `current_batch`/`current_pos` (as the remaining
active rows), the child's remaining results, and the ghost side-effect log. -/
structure St (α ι : Type) where
  pending : List α
  cur : Option (List α)
  child : List (Except String (List α))
  log : List (α × List ι)

def finish (builder : List α) (st : St α ι) : Option (Except String (List α)) × St α ι :=
  if builder.isEmpty then (none, st) else (some (.ok builder), st)

/-- The body of `next`'s `loop` (`foreach.rs:130-201`) followed by the final
`builder.is_empty()` check (`foreach.rs:203-207`). -/
def go (builder : List α) (st : St α ι) : Option (Except String (List α)) × St α ι :=
  if E.B ≤ builder.length then finish builder st                             -- 131-133
  else match hc : st.cur with
    | none => match hch : st.child with
      | [] => finish builder st                                              -- 142
      | .error e :: cs => (some (.error e), { st with child := cs })         -- 141
      | .ok b :: cs => go builder { st with cur := some b, child := cs }     -- 137-139
    | some rows =>
      match hr : inner E st.pending st.log rows with
      | (rest, p, log, some e) => (some (.error e), ⟨p, some rest, st.child, log⟩)  -- 163, 170
      | (rest, p, log, none) =>
        let d := drainP E builder p                                           -- 193
        match hrest : rest with                                               -- 196-200
        | [] => go d.1 ⟨d.2, none, st.child, log⟩
        | y :: ys => go d.1 ⟨d.2, some (y :: ys), st.child, log⟩
termination_by (st.child.length, match st.cur with | none => 0 | some r => r.length + 1)
decreasing_by
  · simp only [hch, hc, List.length_cons]; exact Prod.Lex.left _ _ (by omega)
  · simp only [hc]; exact Prod.Lex.right _ (by omega)
  · have := inner_rest_lt E st.pending st.log rows (by rw [hr]; simp)
    rw [hr] at this
    simp only [hc]; exact Prod.Lex.right _ (by simpa using this)

/-- `ForEachOp::next` (`foreach.rs:124-208`). -/
def next (st : St α ι) : Option (Except String (List α)) × St α ι :=
  let d := drainP E [] st.pending                                             -- 128
  go E d.1 { st with pending := d.2 }

/-- Draining the operator. -/
def run : Nat → St α ι → List (Except String (List α)) × St α ι
  | 0, st => ([], st)
  | fuel + 1, st => match next E st with
    | (none, st') => ([], st')
    | (some r, st') => let rest := run fuel st'; (r :: rest.1, rest.2)

/-! ### Correctness without errors -/

/-- The list items of row `x` (NULL = no iteration). -/
def itemsOf (x : α) : List ι :=
  match E.ev x with
  | .ok (some l) => l
  | _ => []

/-- The body invocations FOREACH must perform for `rows`. -/
def calls (rows : List α) : List (α × List ι) :=
  rows.flatMap (fun x => (chunks E.B E.hB (itemsOf E x)).map (fun c => (x, c)))

/-- No error anywhere: every list evaluates to a list or NULL, the body never fails. -/
def NoErr : Prop := (∀ x, ∃ o, E.ev x = .ok o) ∧ (∀ x c, E.body x c = none)

def okRows : List (Except String (List α)) → List α
  | [] => []
  | .ok b :: cs => b ++ okRows cs
  | .error _ :: cs => okRows cs

def AllOk (cs : List (Except String (List α))) : Prop := ∀ c ∈ cs, ∃ b, c = .ok b

theorem runChunks_ok (h : NoErr E) (x : α) (cs : List (List ι)) :
    runChunks E x cs = (cs.map (fun c => (x, c)), none) := by
  induction cs with
  | nil => rfl
  | cons c cs ih => simp [runChunks, h.2 x c, ih]

theorem execList_ok (h : NoErr E) (x : α) (items : List ι) :
    execList E x items = ((chunks E.B E.hB items).map (fun c => (x, c)), none) := by
  unfold execList
  cases items with
  | nil => simp [chunks_nil]
  | cons i is => simp [runChunks_ok E h]

theorem calls_append (a b : List α) : calls E (a ++ b) = calls E a ++ calls E b := by
  simp [calls]

theorem calls_cons (x : α) (l : List α) : calls E (x :: l) = calls E [x] ++ calls E l := by
  simp [calls]

theorem calls_single (x : α) (h : NoErr E) :
    calls E [x] = (execList E x (itemsOf E x)).1 := by
  simp [calls, execList_ok E h]

/-- The inner loop, without errors: it moves a prefix of `rows` to `pending`,
logs exactly their body calls, and stops early only once `pending` holds `B` rows. -/
theorem inner_ok (h : NoErr E) (pending : List α) (log : List (α × List ι)) (rows : List α) :
    ∃ k, inner E pending log rows =
        (rows.drop k, pending ++ rows.take k, log ++ calls E (rows.take k), none) ∧
      (rows.drop k ≠ [] → E.B ≤ (pending ++ rows.take k).length) ∧ (rows ≠ [] → 0 < k) := by
  induction rows generalizing pending log with
  | nil => exact ⟨0, by simp [inner, calls], by simp, by simp⟩
  | cons x xs ih =>
    obtain ⟨o, ho⟩ := h.1 x
    have hcx : calls E [x] = (execList E x (itemsOf E x)).1 := calls_single E x h
    cases o with
    | none =>
      have hi : itemsOf E x = [] := by simp [itemsOf, ho]
      have hc0 : calls E [x] = [] := by rw [hcx, hi]; simp [execList]
      by_cases hB : E.B ≤ (pending ++ [x]).length
      · refine ⟨1, ?_, ?_, ?_⟩
        · simp only [inner, ho, hB, ite_true, List.drop_succ_cons, List.drop_zero,
            List.take_succ_cons, List.take_zero, hc0, List.append_nil]
        · intro _; simpa using hB
        · intro _; omega
      · obtain ⟨k, hk, hk2, -⟩ := ih (pending ++ [x]) log
        refine ⟨k + 1, ?_, ?_, ?_⟩
        · simp only [inner, ho, hB, ite_false, hk, List.drop_succ_cons, List.take_succ_cons]
          rw [calls_cons E x (List.take k xs), hc0]; simp
        · intro hne
          have := hk2 (by simpa using hne)
          simpa [List.append_assoc] using this
        · intro _; omega
    | some items =>
      have hi : itemsOf E x = items := by simp [itemsOf, ho]
      have hex := execList_ok E h x items
      rw [hi, hex] at hcx
      simp only at hcx
      by_cases hB : E.B ≤ (pending ++ [x]).length
      · refine ⟨1, ?_, ?_, ?_⟩
        · simp only [inner, ho, hex, hB, ite_true, List.drop_succ_cons, List.drop_zero,
            List.take_succ_cons, List.take_zero, hcx]
        · intro _; simpa using hB
        · intro _; omega
      · obtain ⟨k, hk, hk2, -⟩ := ih (pending ++ [x]) (log ++ (execList E x items).1)
        refine ⟨k + 1, ?_, ?_, ?_⟩
        · rw [hex] at hk
          simp only [inner, ho, hex, hB, ite_false, hk, List.drop_succ_cons, List.take_succ_cons]
          rw [calls_cons E x (List.take k xs), hcx]; simp
        · intro hne
          have := hk2 (by simpa using hne)
          simpa [List.append_assoc] using this
        · intro _; omega


theorem okRows_append_ok (b : List α) (cs : List (Except String (List α))) :
    okRows (.ok b :: cs) = b ++ okRows cs := rfl

/-- What one call of `go` guarantees, without errors: it returns an emitted batch
`out` (empty iff `None`) of at most `B` rows; rows are neither lost, duplicated
nor reordered; the log holds exactly the body calls of the rows handed on so far;
and `None` only when everything is consumed. -/
def GoPost (emitted input : List α) (res : Option (Except String (List α))) (st' : St α ι) : Prop :=
  ∃ out : List α,
    ((res = none ∧ out = []) ∨ (res = some (.ok out) ∧ out ≠ [])) ∧
    out.length ≤ E.B ∧
    emitted ++ out ++ st'.pending ++ st'.cur.getD [] ++ okRows st'.child = input ∧
    st'.log = calls E (emitted ++ out ++ st'.pending) ∧
    AllOk st'.child ∧
    (res = none → st'.pending = [] ∧ st'.cur.getD [] = [] ∧ okRows st'.child = [])

theorem allOk_tail {c : Except String (List α)} {cs} (h : AllOk (c :: cs)) : AllOk cs :=
  fun d hd => h d (by simp [hd])

theorem go_ok (h : NoErr E) (emitted input : List α) (builder : List α) (st : St α ι)
    (I1 : emitted ++ builder ++ st.pending ++ st.cur.getD [] ++ okRows st.child = input)
    (I2 : st.log = calls E (emitted ++ builder ++ st.pending))
    (I3 : builder.length ≤ E.B)
    (I4 : st.pending ≠ [] → E.B ≤ builder.length)
    (I5 : AllOk st.child) :
    GoPost E emitted input (go E builder st).1 (go E builder st).2 := by
  rw [go]
  by_cases hB : E.B ≤ builder.length
  · -- builder full: finish with a non-empty batch
    simp only [hB, ite_true, finish]
    have hne : builder ≠ [] := by
      intro h0; subst h0; have := E.hB; simp at hB; omega
    have : builder.isEmpty = false := by cases builder <;> simp_all
    simp only [this]
    exact ⟨builder, Or.inr ⟨rfl, hne⟩, I3, I1, I2, I5, by simp⟩
  · simp only [hB, ite_false]
    have hp : st.pending = [] := by
      cases hq : st.pending with
      | nil => rfl
      | cons a l => exact absurd (I4 (by simp [hq])) hB
    split
    · rename_i hc
      split
      · rename_i hch
        simp only [finish]
        cases hb : builder.isEmpty
        · simp only [Bool.false_eq_true, if_false]
          have hne : builder ≠ [] := by cases builder <;> simp_all
          exact ⟨builder, Or.inr ⟨rfl, hne⟩, I3, I1, I2, I5, by simp⟩
        · simp only [if_true]
          have h0 : builder = [] := by cases builder <;> simp_all
          subst h0
          refine ⟨[], Or.inl ⟨rfl, rfl⟩, by simp, ?_, ?_, I5, ?_⟩
          · simpa using I1
          · simpa using I2
          · intro _; simp [hp, hc, hch, okRows]
      · rename_i e cs hch
        obtain ⟨b, hb⟩ := I5 (.error e) (by simp [hch])
        cases hb
      · rename_i b cs hch
        have hdec0 : cs.length < st.child.length := by rw [hch]; simp
        apply go_ok h emitted input builder
        · simp only [Option.getD_some]; rw [← I1, hc, hch]; simp [okRows, hp]
        · exact I2
        · exact I3
        · intro hne; exact absurd hp hne
        · rw [hch] at I5; exact allOk_tail I5
    · rename_i rows hc
      obtain ⟨k, hk, hk2, hk3⟩ := inner_ok E h st.pending st.log rows
      split
      · rename_i rest p log e hr
        rw [hk] at hr; simp at hr
      · rename_i rest p log hr
        rw [hk] at hr
        simp only [Prod.mk.injEq] at hr
        obtain ⟨hr1, hr2, hr3, -⟩ := hr
        subst hr1 hr2 hr3
        have key1 : emitted ++ builder ++ (st.pending ++ rows.take k) ++ rows.drop k ++ okRows st.child
            = input := by
          rw [← I1, hc]; simp [hp]
        have hdec : ∀ l, List.drop k rows = l → l ≠ [] → l.length < rows.length := by
          intro l hl hne
          have hrne : rows ≠ [] := by intro h0; rw [h0] at hl; simp at hl; exact hne hl
          have hkp := hk3 hrne
          have hlp : 0 < rows.length := by cases rows <;> simp_all
          rw [← hl, List.length_drop]; omega
        split
        · rename_i _ _ _ hrest _
          apply go_ok h emitted input
          · simp only [Option.getD_none, List.append_nil, drainP]
            rw [← key1, hrest]; simp [List.append_assoc]
          · simp only [drainP]; rw [I2, hp]
            simp only [List.append_nil, List.nil_append, calls_append, List.append_assoc]
            rw [← calls_append E (List.take _ _), List.take_append_drop]
          · simp only [drainP, List.length_append, List.length_take]; omega
          · intro hne; simp only [drainP, List.length_append, List.length_take] at hne ⊢
            have : E.B - builder.length < (st.pending ++ List.take k rows).length := by
              apply Nat.lt_of_not_le; intro hc2; apply hne; apply List.drop_eq_nil_of_le; omega
            simp only [List.length_append, List.length_take] at this
            omega
          · exact I5
        · rename_i y ys _ _ _ hrest _
          apply go_ok h emitted input
          · simp only [Option.getD_some, drainP]
            rw [← key1, hrest]; simp [List.append_assoc]
          · simp only [drainP]; rw [I2, hp]
            simp only [List.append_nil, List.nil_append, calls_append, List.append_assoc]
            rw [← calls_append E (List.take _ _), List.take_append_drop]
          · simp only [drainP, List.length_append, List.length_take]; omega
          · intro hne; simp only [drainP, List.length_append, List.length_take] at hne ⊢
            have : E.B - builder.length < (st.pending ++ List.take k rows).length := by
              apply Nat.lt_of_not_le; intro hc2; apply hne; apply List.drop_eq_nil_of_le; omega
            simp only [List.length_append, List.length_take] at this
            omega
          · exact I5
termination_by (st.child.length, match st.cur with | none => 0 | some r => r.length + 1)
decreasing_by
  all_goals first
    | exact Prod.Lex.left _ _ hdec0
    | (simp only [‹st.cur = some _›]; exact Prod.Lex.right _ (by omega))
    | (simp only [‹st.cur = some _›]; exact Prod.Lex.right _
        (by have := hdec _ (by assumption) (by simp); simpa using this))


/-- The between-calls invariant: everything handed on so far (`emitted`), then
`pending`, the current batch's rest and the child's remaining rows, is the
input; the body was run exactly for the rows already queued or emitted. -/
def Inv (emitted input : List α) (st : St α ι) : Prop :=
  emitted ++ st.pending ++ st.cur.getD [] ++ okRows st.child = input ∧
  st.log = calls E (emitted ++ st.pending) ∧ AllOk st.child

theorem next_ok (h : NoErr E) (emitted input : List α) (st : St α ι) (hi : Inv E emitted input st) :
    ∃ out : List α,
      (((next E st).1 = none ∧ out = [] ∧ emitted = input ∧ (next E st).2.log = calls E input) ∨
       ((next E st).1 = some (.ok out) ∧ out ≠ [] ∧ out.length ≤ E.B ∧
         Inv E (emitted ++ out) input (next E st).2)) := by
  obtain ⟨h1, h2, h3⟩ := hi
  have hp := go_ok E h emitted input (drainP E [] st.pending).1 { st with pending := (drainP E [] st.pending).2 }
    (by simp only [drainP, List.nil_append, List.length_nil, Nat.sub_zero]
        rw [← h1]; simp [List.append_assoc])
    (by simp only [drainP, List.nil_append, List.length_nil, Nat.sub_zero]
        rw [h2, List.append_assoc emitted, List.take_append_drop])
    (by simp [drainP]; omega)
    (by intro hne; simp only [drainP, List.nil_append, List.length_nil, Nat.sub_zero] at hne ⊢
        simp only [List.length_take]
        have : E.B < st.pending.length := by
          apply Nat.lt_of_not_le; intro hc; apply hne; apply List.drop_eq_nil_of_le; omega
        omega)
    h3
  obtain ⟨out, hres, hlen, g1, g2, g3, g4⟩ := hp
  refine ⟨out, ?_⟩
  unfold next
  rcases hres with ⟨hn, rfl⟩ | ⟨hs, hne⟩
  · left
    obtain ⟨e1, e2, e3⟩ := g4 hn
    refine ⟨hn, rfl, ?_, ?_⟩
    · rw [e1, e2, e3] at g1; simpa using g1
    · rw [g2, e1]; rw [e1, e2, e3] at g1; simp at g1; simp [g1]
  · right
    refine ⟨hs, hne, hlen, ?_, ?_, g3⟩
    · simpa [List.append_assoc] using g1
    · simpa [List.append_assoc] using g2

def remaining (st : St α ι) : List α := st.pending ++ st.cur.getD [] ++ okRows st.child

/-- **FOREACH is a pass-through that runs the body on every item exactly once.**
From a fresh operator (`ForEachOp::new`, `foreach.rs:56-80`: nothing pending, no
current batch) over all-`Ok` child batches, with enough fuel, the operator emits
non-empty batches of at most `B` rows whose concatenation is the input, and the
body was invoked exactly for `calls input`. -/
theorem run_ok (h : NoErr E) (fuel : Nat) : ∀ (emitted input : List α) (st : St α ι),
    Inv E emitted input st → (remaining st).length < fuel →
    ∃ bs : List (List α),
      (run E fuel st).1 = bs.map Except.ok ∧ emitted ++ bs.flatten = input ∧
      (run E fuel st).2.log = calls E input ∧ ∀ b ∈ bs, b ≠ [] ∧ b.length ≤ E.B := by
  induction fuel with
  | zero => intro _ _ _ _ hf; simp at hf
  | succ n ih =>
    intro emitted input st hi hf
    obtain ⟨out, hcase⟩ := next_ok E h emitted input st hi
    unfold run
    rcases hcase with ⟨hn, _, he, hl⟩ | ⟨hs, hne, hlen, hi'⟩
    · rcases hnx : next E st with ⟨r, st'⟩
      rw [hnx] at hn hl; simp only at hn; subst hn
      exact ⟨[], rfl, by simpa using he, hl, by simp⟩
    · rcases hnx : next E st with ⟨r, st'⟩
      rw [hnx] at hs hi'; simp only at hs; subst hs
      have hrem : (remaining st').length < n := by
        have a1 := hi.1; have a2 := hi'.1
        have : emitted ++ remaining st = emitted ++ out ++ remaining st' := by
          simp only [remaining]; rw [List.append_assoc emitted out, ← List.append_assoc] at a2
          simp only [List.append_assoc] at a1 a2 ⊢; rw [a1, a2]
        have hl2 := congrArg List.length this
        simp at hl2
        have : 0 < out.length := by cases out <;> simp_all
        omega
      obtain ⟨bs, e1, e2, e3, e4⟩ := ih (emitted ++ out) input st' hi' hrem
      refine ⟨out :: bs, by simp [e1], by simpa using e2, e3, ?_⟩
      intro b hb
      simp only [List.mem_cons] at hb
      rcases hb with rfl | hb
      · exact ⟨hne, hlen⟩
      · exact e4 b hb

/-- The fresh operator over input batches `bs`. -/
def init (bs : List (List α)) : St α ι := ⟨[], none, bs.map Except.ok, []⟩

theorem okRows_map (bs : List (List α)) : okRows (bs.map Except.ok) = bs.flatten := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [okRows, ih]

theorem foreach_correct (h : NoErr E) (bs : List (List α)) :
    ∃ outs : List (List α),
      (run E (bs.flatten.length + 1) (init bs)).1 = outs.map Except.ok ∧
      outs.flatten = bs.flatten ∧
      (run E (bs.flatten.length + 1) (init bs)).2.log = calls E bs.flatten ∧
      ∀ o ∈ outs, o ≠ [] ∧ o.length ≤ E.B := by
  have hi : Inv E [] bs.flatten (init bs : St α ι) := by
    refine ⟨by simp [init, okRows_map], by simp [init, calls], ?_⟩
    intro c hc; simp [init] at hc; obtain ⟨b, _, rfl⟩ := hc; exact ⟨b, rfl⟩
  obtain ⟨outs, a, b, c, d⟩ := run_ok E h (bs.flatten.length + 1) [] bs.flatten (init bs) hi
    (by simp [remaining, init, okRows_map])
  exact ⟨outs, a, by simpa using b, c, d⟩

/-- `calls` really is "every item, in order": flattening the chunks of each row
gives back that row's list (`chunks_flatten`), each chunk `≤ B`. -/
theorem calls_items (rows : List α) :
    ((calls E rows).map Prod.snd).flatten = rows.flatMap (itemsOf E) := by
  induction rows with
  | nil => rfl
  | cons x xs ih =>
    rw [calls_cons, List.map_append, List.flatten_append, ih]
    simp [calls, Function.comp_def, chunks_flatten]

/-- Concrete check: `B = 2`, rows `[1,2,3]`, each row `x` iterates `[x, x, x]`;
batches `[[1,2],[3]]` — out `[[1,2],[3]]`, 6 body calls in 6 chunks. -/
def demoE : Env Nat Nat := ⟨2, by decide, fun x => .ok (some [x, x, x]), fun _ _ => none⟩

def okOrEmpty : Except String (List Nat) → List Nat
  | .ok b => b
  | .error _ => []

#guard (run demoE 10 (init [[1, 2], [3]])).1.map okOrEmpty = [[1, 2], [3]]
#guard (run demoE 10 (init [[1, 2], [3]])).2.log =
      [(1, [1, 1]), (1, [1]), (2, [2, 2]), (2, [2]), (3, [3, 3]), (3, [3])]
/- `B = 2` with rows `[1,2,3]` in ONE input batch: `pending` reaches `B` after
row 2, so the inner loop stops early and row 3 is processed on the next call. -/
#guard (run demoE 10 (init [[1, 2, 3]])).1.map okOrEmpty = [[1, 2], [3]]

end ForEach
