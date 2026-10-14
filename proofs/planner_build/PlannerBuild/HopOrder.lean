/-
# `self_loops_after_binders` (planner/hop_order.rs, #3102 61284ea4f; fixes #2999)

| Lean | Rust |
| --- | --- |
| `HopShape`, `touches`, `movable`, `isBinder` | `HopShape` hop_order.rs:5, the closures `touches` :35, `movable` :31 |
| `deferP` | the `if` condition :41-46 (`!x.self_loop && touches(x, s.from)` = `isBinder`) |
| `flush` | the latest-first `for k in (0..deferred.len()).rev()` loop :52-56 |
| `step` | one iteration of `for (i, (hop, s)) in hops.into_iter().zip(&shapes).enumerate()` :39-58 |
| `selfLoopsAfterBinders` | `self_loops_after_binders` :25 (early return :32-34, `foldl` = the loop) |
| `pre3102_loop`, `pre3102_reference` | the loop #3102's first commit used, kept as `tests::reference` hop_order.rs:84-107 |

Generic over the hop type `T` and the node type `N` (the planner instantiates `T = QR`,
`N = (id, scope_id)`, `bound = self.visited.contains`, mod.rs:1917-1926).

* `selfLoops_perm` — the output is a permutation of the input (no hop lost or doubled; the
  deferred list is always empty at the end, `run_D_nil`).
* `kept_sublist` — the hops that are not deferred keep their relative order.
* `deferred_after_binder` — a deferred self-loop comes after the first later hop that binds
  its node (with only other self-loops flushed by that binder between them).
* `single_pass` — the result is one left fold over the input (each hop is stepped exactly once,
  `length (zipIdx) = length hops`) — vs. the old loop, which revisits every hop it moves.
* **`eq_reference`** — it equals the pre-#3102 loop (`pre3102_reference`) whenever a hop flagged
  `self_loop` has `from = to` (`WF`; in the planner, an alias id names one variable per pattern).
  `wf_needed` shows the premise matters for the generic function.
* `early_return_sound` — the `!(1..len).any(movable)` fast path returns what the loop would.
-/
namespace PlannerBuild.HopOrder

local infix:50 " ~ " => List.Perm

structure HopShape (N : Type) where
  frm : N
  to : N
  selfLoop : Bool
  fixed : Bool
  deriving DecidableEq

variable {T N : Type} [DecidableEq N]

def touches (x : HopShape N) (n : N) : Bool := x.frm == n || x.to == n

def isBinder (n : N) (x : HopShape N) : Bool := !x.selfLoop && touches x n

/-- `movable(k)`: `k > 0 && shapes[k].self_loop && shapes[k].fixed`. -/
def movable (ss : List (HopShape N)) (k : Nat) : Bool :=
  0 < k && match ss[k]? with
    | some s => s.selfLoop && s.fixed
    | none => false

/-- The deferral test (hop_order.rs:41-46). -/
def deferP (ss : List (HopShape N)) (bound : N → Bool) (i : Nat) (s : HopShape N) : Bool :=
  movable ss i && !bound s.frm &&
  !((List.range i).any fun k => match ss[k]? with
    | some x => touches x s.frm && !movable ss k
    | none => false) &&
  (ss.drop (i + 1)).any (isBinder s.frm)

/-- Latest-first flush at a non-self-loop hop `s`: `(pushed, kept)`. The `rev()` loop removes
index `k` only after every higher index was examined, so recursing on the tail first and then
the head is the same order. -/
def flush (s : HopShape N) : List (N × T) → List T × List (N × T)
  | [] => ([], [])
  | d :: ds =>
    let r := flush s ds
    if touches s d.1 then (r.1 ++ [d.2], r.2) else (r.1, d :: r.2)

/-- One loop iteration on `(deferred, out)`. -/
def step (ss : List (HopShape N)) (bound : N → Bool) (st : List (N × T) × List T)
    (x : (T × HopShape N) × Nat) : List (N × T) × List T :=
  if deferP ss bound x.2 x.1.2 then (st.1 ++ [(x.1.2.frm, x.1.1)], st.2)
  else
    let out := st.2 ++ [x.1.1]
    if !x.1.2.selfLoop then ((flush x.1.2 st.1).2, out ++ (flush x.1.2 st.1).1) else (st.1, out)

/-- `self_loops_after_binders(hops, shape, bound)` (hop_order.rs:25). -/
def selfLoopsAfterBinders (hops : List T) (shape : T → HopShape N) (bound : N → Bool) : List T :=
  let ss := hops.map shape
  if !((List.range' 1 (ss.length - 1)).any (movable ss)) then hops
  else (((hops.zip ss).zipIdx).foldl (step ss bound) ([], [])).2

/-! ## Basic facts -/

theorem zip_map_self (hops : List T) (f : T → HopShape N) :
    hops.zip (hops.map f) = hops.map (fun h => (h, f h)) := by
  induction hops with
  | nil => rfl
  | cons h t ih => simp [ih]

theorem flush_perm (s : HopShape N) : ∀ D : List (N × T),
    ((flush s D).1 ++ (flush s D).2.map Prod.snd).Perm (D.map Prod.snd)
  | [] => by simp [flush]
  | d :: ds => by
    have ih := flush_perm s ds
    unfold flush
    by_cases h : touches s d.1 = true
    · simp only [h, if_true, List.map_cons, List.append_assoc, List.singleton_append]
      exact (List.perm_middle).trans (List.Perm.cons _ ih)
    · simp only [h, Bool.false_eq_true, if_false, List.map_cons]
      exact (List.perm_middle).trans (List.Perm.cons _ ih)

theorem flush_kept_sub (s : HopShape N) : ∀ (D : List (N × T)), ∀ d ∈ (flush s D).2, d ∈ D ∧ touches s d.1 = false
  | [] => by simp [flush]
  | d :: ds => by
    intro x hx
    have ih := flush_kept_sub s ds
    unfold flush at hx
    by_cases h : touches s d.1 = true
    · simp only [h, if_true] at hx
      exact ⟨List.mem_cons_of_mem _ (ih x hx).1, (ih x hx).2⟩
    · simp only [h, Bool.false_eq_true, if_false, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact ⟨List.mem_cons_self, by simpa using h⟩
      · exact ⟨List.mem_cons_of_mem _ (ih x hx).1, (ih x hx).2⟩

theorem flush_pushed (s : HopShape N) : ∀ (D : List (N × T)) (d : N × T), d ∈ D → touches s d.1 = true →
    d.2 ∈ (flush s D).1
  | [], _, h, _ => by simp at h
  | e :: es, d, hd, ht => by
    unfold flush
    rcases List.mem_cons.1 hd with rfl | hd
    · simp [ht]
    · have := flush_pushed s es d hd ht
      by_cases h : touches s e.1 = true <;> simp [h, this]

theorem flush_pushed_mem (s : HopShape N) : ∀ (D : List (N × T)) (x : T), x ∈ (flush s D).1 →
    ∃ d ∈ D, d.2 = x ∧ touches s d.1 = true
  | [], _, h => by simp [flush] at h
  | e :: es, x, hx => by
    unfold flush at hx
    by_cases h : touches s e.1 = true
    · simp only [h, if_true, List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · obtain ⟨d, hd, h1, h2⟩ := flush_pushed_mem s es x hx
        exact ⟨d, List.mem_cons_of_mem _ hd, h1, h2⟩
      · exact ⟨e, List.mem_cons_self, rfl, h⟩
    · simp only [h, Bool.false_eq_true, if_false] at hx
      obtain ⟨d, hd, h1, h2⟩ := flush_pushed_mem s es x hx
      exact ⟨d, List.mem_cons_of_mem _ hd, h1, h2⟩

theorem flush_nil (s : HopShape N) : flush s ([] : List (N × T)) = ([], []) := rfl

/-! ## The run over a suffix -/

/-- The loop from index `m` on the remaining hops `t`. -/
def run (shape : T → HopShape N) (ss : List (HopShape N)) (bound : N → Bool) (t : List T) (m : Nat)
    (st : List (N × T) × List T) : List (N × T) × List T :=
  ((t.map fun h => (h, shape h)).zipIdx m).foldl (step ss bound) st

theorem run_cons (shape : T → HopShape N) (ss) (bound : N → Bool) (h : T) (t : List T) (m : Nat) (st) :
    run shape ss bound (h :: t) m st = run shape ss bound t (m + 1) (step ss bound st ((h, shape h), m)) := by
  simp [run, List.zipIdx_cons]

theorem run_nil (shape : T → HopShape N) (ss) (bound : N → Bool) (m : Nat) (st) :
    run shape ss bound [] m st = st := rfl

/-- The loop is `run` from 0. -/
theorem loop_eq_run (hops : List T) (shape : T → HopShape N) (bound : N → Bool) :
    ((hops.zip (hops.map shape)).zipIdx).foldl (step (hops.map shape) bound) ([], []) =
      run shape (hops.map shape) bound hops 0 ([], []) := by
  simp [run, zip_map_self]

/-- `out` only grows. -/
theorem run_prefix (shape : T → HopShape N) (ss) (bound : N → Bool) :
    ∀ (t : List T) (m : Nat) (st : List (N × T) × List T), ∃ X, (run shape ss bound t m st).2 = st.2 ++ X := by
  intro t
  induction t with
  | nil => intro m st; exact ⟨[], by simp [run_nil]⟩
  | cons h t ih =>
    intro m st
    rw [run_cons]
    obtain ⟨X, hX⟩ := ih (m + 1) (step ss bound st ((h, shape h), m))
    rw [hX]
    unfold step
    split
    · exact ⟨X, rfl⟩
    · split
      · exact ⟨[h] ++ (flush (shape h) st.1).1 ++ X, by simp⟩
      · exact ⟨[h] ++ X, by simp⟩

/-! ## Invariant: alignment with the input -/

theorem getElem?_of_drop {α : Type} (l : List α) (m : Nat) (h : α) (t : List α) (ht : l.drop m = h :: t) :
    l[m]? = some h := by
  have := congrArg (·[0]?) ht
  simpa [List.getElem?_drop] using this

theorem drop_succ_of_drop {α : Type} (l : List α) (m : Nat) (h : α) (t : List α) (ht : l.drop m = h :: t) :
    l.drop (m + 1) = t := by
  rw [← List.drop_drop, ht]; rfl

theorem take_succ_of_drop {α : Type} (l : List α) (m : Nat) (h : α) (t : List α) (ht : l.drop m = h :: t) :
    l.take (m + 1) = l.take m ++ [h] := by
  rw [List.take_succ, getElem?_of_drop l m h t ht]; rfl

theorem movable_spec (ss : List (HopShape N)) (m : Nat) (s : HopShape N) (hs : ss[m]? = some s) :
    movable ss m = (decide (0 < m) && s.selfLoop && s.fixed) := by
  simp [movable, hs, Bool.and_assoc]

section
variable (hops : List T) (shape : T → HopShape N) (bound : N → Bool)

theorem step_defer (m : Nat) (h : T) (st : List (N × T) × List T)
    (hd : deferP (hops.map shape) bound m (shape h) = true) :
    step (hops.map shape) bound st ((h, shape h), m) = (st.1 ++ [((shape h).frm, h)], st.2) := by
  simp [step, hd]

theorem step_keep_loop (m : Nat) (h : T) (st : List (N × T) × List T)
    (hd : deferP (hops.map shape) bound m (shape h) = false) (hl : (shape h).selfLoop = true) :
    step (hops.map shape) bound st ((h, shape h), m) = (st.1, st.2 ++ [h]) := by
  simp [step, hd, hl]

theorem step_keep_bind (m : Nat) (h : T) (st : List (N × T) × List T)
    (hd : deferP (hops.map shape) bound m (shape h) = false) (hl : (shape h).selfLoop = false) :
    step (hops.map shape) bound st ((h, shape h), m) =
      ((flush (shape h) st.1).2, st.2 ++ [h] ++ (flush (shape h) st.1).1) := by
  simp [step, hd, hl]

/-- A deferred hop is a fixed self-loop past index 0 with a later binder. -/
theorem deferP_spec (m : Nat) (h : T) (t : List T) (ht : hops.drop m = h :: t)
    (hd : deferP (hops.map shape) bound m (shape h) = true) :
    (shape h).selfLoop = true ∧ (shape h).fixed = true ∧ 0 < m ∧ bound (shape h).frm = false ∧
    (∀ k < m, ∀ x, (hops.map shape)[k]? = some x → touches x (shape h).frm = true →
      movable (hops.map shape) k = true) ∧
    (t.map shape).any (isBinder (shape h).frm) = true := by
  have hs : (hops.map shape)[m]? = some (shape h) := by simp [getElem?_of_drop hops m h t ht]
  have hdr : (hops.map shape).drop (m + 1) = t.map shape := by
    rw [← List.map_drop, drop_succ_of_drop hops m h t ht]
  simp only [deferP, movable_spec _ m _ hs, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq,
    hdr] at hd
  obtain ⟨⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩, h5⟩ := hd
  refine ⟨h1, h2, h0, h3, ?_, h5⟩
  intro k hk x hx ht'
  have := List.any_eq_false.1 h4 k (List.mem_range.2 hk)
  simp only [hx, ht', Bool.true_and, Bool.not_eq_true'] at this
  cases hm : movable (hops.map shape) k
  · simp [hm] at this
  · rfl

/-- The perm/deferred-list invariant. -/
structure Inv (m : Nat) (D : List (N × T)) (out : List T) : Prop where
  perm : (out ++ D.map Prod.snd).Perm (hops.take m)
  dl : ∀ d ∈ D, (shape d.2).selfLoop = true ∧ d.1 = (shape d.2).frm ∧
    ((hops.drop m).map shape).any (isBinder d.1) = true

theorem inv_step (m : Nat) (h : T) (t : List T) (ht : hops.drop m = h :: t) (D : List (N × T)) (out : List T)
    (hi : Inv hops shape m D out) :
    Inv hops shape (m + 1) (step (hops.map shape) bound (D, out) ((h, shape h), m)).1
      (step (hops.map shape) bound (D, out) ((h, shape h), m)).2 := by
  have hdr : hops.drop (m + 1) = t := drop_succ_of_drop hops m h t ht
  have htk : hops.take (m + 1) = hops.take m ++ [h] := take_succ_of_drop hops m h t ht
  -- an old deferred hop's binder is still ahead unless `h` is it
  have old : ∀ d ∈ D, isBinder d.1 (shape h) = false → (t.map shape).any (isBinder d.1) = true := by
    intro d hd hb
    have := (hi.dl d hd).2.2
    rw [ht] at this
    simpa [hb] using this
  cases hdef : deferP (hops.map shape) bound m (shape h)
  · cases hl : (shape h).selfLoop
    · rw [step_keep_bind hops shape bound m h _ hdef hl]
      refine ⟨?_, ?_⟩
      · rw [htk]
        have hp := flush_perm (shape h) D
        calc (out ++ [h] ++ (flush (shape h) D).1 ++ (flush (shape h) D).2.map Prod.snd)
            = (out ++ [h]) ++ ((flush (shape h) D).1 ++ (flush (shape h) D).2.map Prod.snd) := by simp
          _ ~ (out ++ [h]) ++ D.map Prod.snd := List.Perm.append_left _ hp
          _ ~ (out ++ D.map Prod.snd) ++ [h] := by
            simp only [List.append_assoc]; exact List.Perm.append_left out List.perm_append_comm
          _ ~ hops.take m ++ [h] := List.Perm.append_right [h] hi.perm
      · intro d hd
        obtain ⟨hdD, hnt⟩ := flush_kept_sub (shape h) D d hd
        refine ⟨(hi.dl d hdD).1, (hi.dl d hdD).2.1, ?_⟩
        rw [hdr]; exact old d hdD (by simp [isBinder, hl, hnt])
    · rw [step_keep_loop hops shape bound m h _ hdef hl]
      refine ⟨?_, ?_⟩
      · rw [htk]
        calc (out ++ [h] ++ D.map Prod.snd)
            ~ (out ++ D.map Prod.snd) ++ [h] := by
              simp only [List.append_assoc]; exact List.Perm.append_left out List.perm_append_comm
          _ ~ hops.take m ++ [h] := List.Perm.append_right [h] hi.perm
      · intro d hd
        refine ⟨(hi.dl d hd).1, (hi.dl d hd).2.1, ?_⟩
        rw [hdr]; exact old d hd (by simp [isBinder, hl])
  · obtain ⟨hl, -, -, -, -, hb⟩ := deferP_spec hops shape bound m h t ht hdef
    rw [step_defer hops shape bound m h _ hdef]
    refine ⟨?_, ?_⟩
    · rw [htk]
      have := hi.perm
      simp only [List.map_append, List.map_cons, List.map_nil, ← List.append_assoc]
      exact List.Perm.append_right [h] this
    · intro d hd
      rcases List.mem_append.1 hd with hd | hd
      · refine ⟨(hi.dl d hd).1, (hi.dl d hd).2.1, ?_⟩
        rw [hdr]; exact old d hd (by simp [isBinder, hl])
      · simp only [List.mem_singleton] at hd
        subst hd
        exact ⟨hl, rfl, by rw [hdr]; exact hb⟩

theorem run_inv : ∀ (t : List T) (m : Nat) (D : List (N × T)) (out : List T), hops.drop m = t →
    Inv hops shape m D out →
    (run shape (hops.map shape) bound t m (D, out)).1 = [] ∧ (run shape (hops.map shape) bound t m (D, out)).2 ~ hops := by
  intro t
  induction t with
  | nil =>
    intro m D out ht hi
    have hlen : hops.length ≤ m := by simpa using ht
    have hD : D = [] := by
      cases D with
      | nil => rfl
      | cons d _ => have := (hi.dl d List.mem_cons_self).2.2; simp [ht] at this
    subst hD
    refine ⟨rfl, ?_⟩
    have := hi.perm
    simpa [run_nil, List.take_of_length_le hlen] using this
  | cons h t ih =>
    intro m D out ht hi
    rw [run_cons]
    have := inv_step hops shape bound m h t ht D out hi
    exact ih (m + 1) _ _ (drop_succ_of_drop hops m h t ht) this

/-- The early return is the loop's answer when nothing is movable. -/
theorem run_nothing_movable : ∀ (t : List T) (m : Nat) (out : List T), hops.drop m = t →
    (∀ i, m ≤ i → movable (hops.map shape) i = false) →
    run shape (hops.map shape) bound t m ([], out) = ([], out ++ t) := by
  intro t
  induction t with
  | nil => intro m out _ _; simp [run_nil]
  | cons h t ih =>
    intro m out ht hm
    have hd : deferP (hops.map shape) bound m (shape h) = false := by
      simp [deferP, hm m (Nat.le_refl _)]
    rw [run_cons]
    cases hl : (shape h).selfLoop
    · rw [step_keep_bind hops shape bound m h _ hd hl]
      simp only [flush_nil, List.append_nil]
      rw [ih (m + 1) _ (drop_succ_of_drop hops m h t ht) (fun i hi => hm i (by omega))]; simp
    · rw [step_keep_loop hops shape bound m h _ hd hl]
      rw [ih (m + 1) _ (drop_succ_of_drop hops m h t ht) (fun i hi => hm i (by omega))]; simp

theorem selfLoops_eq_run :
    selfLoopsAfterBinders hops shape bound =
      if !((List.range' 1 ((hops.map shape).length - 1)).any (movable (hops.map shape))) then hops
      else (run shape (hops.map shape) bound hops 0 ([], [])).2 := by
  simp only [selfLoopsAfterBinders, loop_eq_run]

theorem nothing_movable_of (h : (List.range' 1 ((hops.map shape).length - 1)).any (movable (hops.map shape)) = false) :
    ∀ i, 0 ≤ i → movable (hops.map shape) i = false := by
  intro i _
  rcases Nat.lt_or_ge i (hops.map shape).length with hi | hi
  · rcases Nat.eq_zero_or_pos i with rfl | hpos
    · simp [movable]
    · simpa using List.any_eq_false.1 h i (List.mem_range'_1.2 ⟨hpos, by omega⟩)
  · simp [movable, List.getElem?_eq_none hi]

/-- **`early_return_sound`**: the `!(1..len).any(movable)` fast path returns exactly what the
loop computes. -/
theorem early_return_sound :
    selfLoopsAfterBinders hops shape bound = (run shape (hops.map shape) bound hops 0 ([], [])).2 := by
  rw [selfLoops_eq_run]
  split
  · rename_i h
    rw [run_nothing_movable hops shape bound hops 0 [] (by simp)
      (nothing_movable_of hops shape (by simpa using h))]; simp
  · rfl

/-- **`selfLoops_perm`**: the output is a permutation of the input. -/
theorem selfLoops_perm : selfLoopsAfterBinders hops shape bound ~ hops := by
  rw [early_return_sound]
  exact (run_inv hops shape bound hops 0 [] [] (by simp) ⟨by simp, by simp⟩).2

/-- **`single_pass`**: one left fold over `zip(shapes).enumerate()` — one `step` per hop. -/
theorem single_pass :
    (selfLoopsAfterBinders hops shape bound = hops ∨
      selfLoopsAfterBinders hops shape bound =
        (((hops.zip (hops.map shape)).zipIdx).foldl (step (hops.map shape) bound) ([], [])).2) ∧
    ((hops.zip (hops.map shape)).zipIdx).length = hops.length := by
  refine ⟨?_, by simp⟩
  dsimp only [selfLoopsAfterBinders]
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

/-! ## Order -/

/-- The hops that stay in place, in input order. -/
def keptFrom (t : List T) (m : Nat) : List T :=
  ((t.map fun h => (h, shape h)).zipIdx m).filterMap fun x =>
    if deferP (hops.map shape) bound x.2 x.1.2 then none else some x.1.1

theorem kept_run : ∀ (t : List T) (m : Nat) (st : List (N × T) × List T) (K : List T), K.Sublist st.2 →
    (K ++ keptFrom hops shape bound t m).Sublist (run shape (hops.map shape) bound t m st).2 := by
  intro t
  induction t with
  | nil => intro m st K hK; simpa [keptFrom, run_nil] using hK
  | cons h t ih =>
    intro m st K hK
    rw [run_cons]
    have hk : keptFrom hops shape bound (h :: t) m =
        (if deferP (hops.map shape) bound m (shape h) then [] else [h]) ++ keptFrom hops shape bound t (m + 1) := by
      cases hd : deferP (hops.map shape) bound m (shape h) <;>
        simp [keptFrom, List.zipIdx_cons, hd]
    rw [hk]
    cases hd : deferP (hops.map shape) bound m (shape h)
    · simp only [Bool.false_eq_true, if_false, ← List.append_assoc]
      apply ih
      cases hl : (shape h).selfLoop
      · rw [step_keep_bind hops shape bound m h _ hd hl]
        exact (hK.append_right [h]).trans (List.sublist_append_left _ _)
      · rw [step_keep_loop hops shape bound m h _ hd hl]
        exact hK.append_right [h]
    · simp only [if_true, List.nil_append]
      apply ih
      rw [step_defer hops shape bound m h _ hd]
      exact hK

/-- **`kept_sublist`**: the hops not deferred appear in the output in their input order. -/
theorem kept_sublist :
    (keptFrom hops shape bound hops 0).Sublist (selfLoopsAfterBinders hops shape bound) := by
  rw [early_return_sound]
  simpa using kept_run hops shape bound hops 0 ([], []) [] (List.nil_sublist _)

theorem flush_keeps (s : HopShape N) : ∀ (D : List (N × T)) (d : N × T), d ∈ D → touches s d.1 = false →
    d ∈ (flush s D).2
  | [], _, h, _ => by simp at h
  | e :: es, d, hd, ht => by
    unfold flush
    rcases List.mem_cons.1 hd with rfl | hd
    · simp [ht]
    · have := flush_keeps s es d hd ht
      by_cases h : touches s e.1 = true <;> simp [h, this]

theorem run_append : ∀ (A B : List T) (m : Nat) (st : List (N × T) × List T),
    run shape (hops.map shape) bound (A ++ B) m st =
      run shape (hops.map shape) bound B (m + A.length) (run shape (hops.map shape) bound A m st) := by
  intro A
  induction A with
  | nil => intro B m st; simp [run_nil]
  | cons a A ih =>
    intro B m st
    simp only [List.cons_append, run_cons, ih, List.length_cons]
    rw [Nat.add_assoc, Nat.add_comm 1 A.length]

theorem deferP_not_selfloop (m : Nat) (s : HopShape N) (hl : s.selfLoop = false)
    (hs : (hops.map shape)[m]? = some s) : deferP (hops.map shape) bound m s = false := by
  simp [deferP, movable_spec _ m s hs, hl]

/-- A deferred `(n, h)` stays deferred until the first binder of `n`, and is flushed right
after it. -/
theorem track : ∀ (A : List T) (b : T) (C : List T) (m : Nat) (D : List (N × T)) (out : List T) (d : N × T),
    hops.drop m = A ++ b :: C → d ∈ D → (∀ a ∈ A, isBinder d.1 (shape a) = false) → isBinder d.1 (shape b) = true →
    ∃ X Y Z, (run shape (hops.map shape) bound (A ++ b :: C) m (D, out)).2 = X ++ [b] ++ Y ++ d.2 :: Z := by
  intro A
  induction A with
  | nil =>
    intro b C m D out d ht hd _ hb
    have hl : (shape b).selfLoop = false := by simp [isBinder] at hb; exact hb.1
    have hs : (hops.map shape)[m]? = some (shape b) := by simp [getElem?_of_drop hops m b C ht]
    have hdf := deferP_not_selfloop hops shape bound m (shape b) hl hs
    simp only [List.nil_append, run_cons, step_keep_bind hops shape bound m b _ hdf hl]
    have hP : d.2 ∈ (flush (shape b) D).1 :=
      flush_pushed (shape b) D d hd (by simp [isBinder] at hb; simpa [touches] using hb.2)
    obtain ⟨Y, Z, hYZ⟩ := List.append_of_mem hP
    obtain ⟨W, hW⟩ := run_prefix shape (hops.map shape) bound C (m + 1)
      ((flush (shape b) D).2, out ++ [b] ++ (flush (shape b) D).1)
    rw [hW, hYZ]
    exact ⟨out, Y, Z ++ W, by simp⟩
  | cons a A ih =>
    intro b C m D out d ht hd hA hb
    have ha := hA a List.mem_cons_self
    have hdr : hops.drop (m + 1) = A ++ b :: C := drop_succ_of_drop hops m a _ (by simpa using ht)
    simp only [List.cons_append, run_cons]
    cases hdef : deferP (hops.map shape) bound m (shape a)
    · cases hl : (shape a).selfLoop
      · rw [step_keep_bind hops shape bound m a _ hdef hl]
        have hnt : touches (shape a) d.1 = false := by simpa [isBinder, hl] using ha
        exact ih b C (m + 1) _ _ d hdr (flush_keeps (shape a) D d hd hnt) (fun x hx => hA x (by simp [hx])) hb
      · rw [step_keep_loop hops shape bound m a _ hdef hl]
        exact ih b C (m + 1) _ _ d hdr hd (fun x hx => hA x (by simp [hx])) hb
    · rw [step_defer hops shape bound m a _ hdef]
      exact ih b C (m + 1) _ _ d hdr (List.mem_append_left _ hd) (fun x hx => hA x (by simp [hx])) hb

/-- **`deferred_after_binder`**: a deferred self-loop `h` (index `m`) is emitted after the first
later hop `b` that binds its node; only hops flushed by `b` (other deferred self-loops, later
ones first) sit between them. -/
theorem deferred_after_binder (m : Nat) (h : T) (A : List T) (b : T) (C : List T)
    (ht : hops.drop m = h :: (A ++ b :: C))
    (hd : deferP (hops.map shape) bound m (shape h) = true)
    (hA : ∀ a ∈ A, isBinder (shape h).frm (shape a) = false) (hb : isBinder (shape h).frm (shape b) = true) :
    ∃ X Y Z, selfLoopsAfterBinders hops shape bound = X ++ [b] ++ Y ++ h :: Z := by
  rw [early_return_sound]
  have hsplit : hops = hops.take m ++ h :: (A ++ b :: C) := by rw [← ht, List.take_append_drop]
  have hlen : (hops.take m).length = m := by
    have : m < hops.length := by
      have := congrArg List.length ht; simp at this; omega
    simp; omega
  have e := congrArg (fun L => (run shape (hops.map shape) bound L 0 ([], [])).2) hsplit
  rw [e, run_append, hlen, Nat.zero_add, run_cons, step_defer hops shape bound m h _ hd]
  exact track hops shape bound A b C (m + 1) _ _ ((shape h).frm, h)
    (drop_succ_of_drop hops m h _ ht) (List.mem_append_right _ (by simp)) hA hb

end

/-! ## The pre-#3102 loop (`tests::reference`) -/

/-- The loop #3102's first commit planned with (hop_order.rs:84-107 keeps it verbatim):
`remove(i)` the self-loop and `insert(i + j + 1)` it right after its first later binder, then
look at index `i` again. Fuel bounds the `while`. -/
def pre3102_loop (shape : T → HopShape N) (bound : N → Bool) : Nat → List T → Nat → List T
  | 0, rels, _ => rels
  | fuel + 1, rels, i =>
    match rels[i]? with
    | none => rels
    | some r =>
      if (shape r).selfLoop && (shape r).fixed && !bound (shape r).frm &&
          !((rels.take i).any fun x => touches (shape x) (shape r).frm) then
        match (rels.drop (i + 1)).findIdx? (fun x => isBinder (shape r).frm (shape x)) with
        | some j => pre3102_loop shape bound fuel ((rels.eraseIdx i).insertIdx (i + j + 1) r) i
        | none => pre3102_loop shape bound fuel rels (i + 1)
      else pre3102_loop shape bound fuel rels (i + 1)

/-- `reference(rels, bound)` starting at `i = 1`; `2n + 1` iterations always suffice
(each hop is looked at once, plus once more for each move). -/
def pre3102_reference (shape : T → HopShape N) (bound : N → Bool) (hops : List T) : List T :=
  pre3102_loop shape bound (2 * hops.length + 1) hops 1

/-- The old loop's list while the new one has `(D, out)` and `t` left: every deferred hop
already sits right after its binder in `t`. -/
def ins (shape : T → HopShape N) : List (N × T) → List T → List T
  | D, [] => D.map Prod.snd
  | D, h :: t =>
    if (shape h).selfLoop then h :: ins shape D t
    else h :: ((flush (shape h) D).1 ++ ins shape (flush (shape h) D).2 t)

theorem flush_nil_list (s : HopShape N) : ∀ D : List (N × T), D = [] → flush s D = ([], [])
  | [], _ => rfl

theorem ins_nil (shape : T → HopShape N) : ∀ t : List T, ins shape [] t = t
  | [] => rfl
  | h :: t => by
    unfold ins
    split
    · rw [ins_nil shape t]
    · simp [flush_nil, ins_nil shape t]

theorem flush_snoc_touch (s : HopShape N) (x : N × T) (hx : touches s x.1 = true) :
    ∀ D : List (N × T), flush s (D ++ [x]) = (x.2 :: (flush s D).1, (flush s D).2)
  | [] => by simp [flush, hx]
  | d :: ds => by
    have ih := flush_snoc_touch s x hx ds
    simp only [List.cons_append]
    unfold flush
    rw [ih]
    split <;> simp

theorem flush_snoc_notouch (s : HopShape N) (x : N × T) (hx : touches s x.1 = false) :
    ∀ D : List (N × T), flush s (D ++ [x]) = ((flush s D).1, (flush s D).2 ++ [x])
  | [] => by simp [flush, hx]
  | d :: ds => by
    have ih := flush_snoc_notouch s x hx ds
    simp only [List.cons_append]
    unfold flush
    rw [ih]
    split <;> simp

theorem flush_pushed_loops (shape : T → HopShape N) (s : HopShape N) (D : List (N × T))
    (hD : ∀ d ∈ D, (shape d.2).selfLoop = true) : ∀ x ∈ (flush s D).1, (shape x).selfLoop = true := by
  intro x hx
  obtain ⟨d, hd, rfl, -⟩ := flush_pushed_mem s D x hx
  exact hD d hd

theorem flush_kept_loops (shape : T → HopShape N) (s : HopShape N) (D : List (N × T))
    (hD : ∀ d ∈ D, (shape d.2).selfLoop = true) : ∀ d ∈ (flush s D).2, (shape d.2).selfLoop = true :=
  fun d hd => hD d (flush_kept_sub s D d hd).1

theorem findIdx?_none_of {α : Type} (p : α → Bool) (l : List α) (h : ∀ x ∈ l, p x = false) :
    l.findIdx? p = none := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.findIdx?_cons, h a List.mem_cons_self]
    simp [ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

theorem insertIdx_append_left {α : Type} (A B : List α) (k : Nat) (x : α) :
    (A ++ B).insertIdx (A.length + k) x = A ++ B.insertIdx k x := by
  induction A with
  | nil => simp
  | cons a A ih =>
    rw [List.cons_append, List.length_cons, show A.length + 1 + k = (A.length + k) + 1 by omega,
      List.insertIdx_succ_cons, ih]; rfl

/-- **Insertion after the first binder** is deferral: `insert(i + j + 1)` in the old loop's
list equals the new loop's state with the hop appended to the deferred list. -/
theorem ins_insert (shape : T → HopShape N) (n : N) (h : T) : ∀ (t : List T) (D : List (N × T)),
    (∀ d ∈ D, (shape d.2).selfLoop = true) → (t.map shape).any (isBinder n) = true →
    ∃ j, (ins shape D t).findIdx? (fun x => isBinder n (shape x)) = some j ∧
      (ins shape D t).insertIdx (j + 1) h = ins shape (D ++ [(n, h)]) t := by
  intro t
  induction t with
  | nil => intro D _ hb; simp at hb
  | cons b t ih =>
    intro D hD hb
    by_cases hbb : isBinder n (shape b) = true
    · have hl : (shape b).selfLoop = false := by simp [isBinder] at hbb; exact hbb.1
      have ht : touches (shape b) n = true := by simp [isBinder] at hbb; simpa [touches] using hbb.2
      refine ⟨0, ?_, ?_⟩
      · simp [ins, hl, List.findIdx?_cons, hbb]
      · simp only [ins, hl, Bool.false_eq_true, if_false]
        rw [flush_snoc_touch (shape b) (n, h) ht D]
        simp [List.insertIdx_succ_cons]
    · have hbf : isBinder n (shape b) = false := by simpa using hbb
      have hb' : (t.map shape).any (isBinder n) = true := by simpa [hbf] using hb
      cases hl : (shape b).selfLoop
      · have hnt : touches (shape b) n = false := by simpa [isBinder, hl] using hbf
        obtain ⟨j, hj, hins⟩ := ih (flush (shape b) D).2 (flush_kept_loops shape _ D hD) hb'
        have hP : ∀ x ∈ (flush (shape b) D).1, isBinder n (shape x) = false := by
          intro x hx; simp [isBinder, flush_pushed_loops shape _ D hD x hx]
        refine ⟨1 + (flush (shape b) D).1.length + j, ?_, ?_⟩
        · simp only [ins, hl, Bool.false_eq_true, if_false, List.findIdx?_cons, hbf]
          rw [List.findIdx?_append, findIdx?_none_of _ _ hP, hj]
          simp; omega
        · simp only [ins, hl, Bool.false_eq_true, if_false]
          rw [flush_snoc_notouch (shape b) (n, h) hnt D]
          rw [show 1 + (flush (shape b) D).1.length + j + 1 = ((flush (shape b) D).1.length + (j + 1)) + 1 by omega,
            List.insertIdx_succ_cons, insertIdx_append_left, hins]
      · obtain ⟨j, hj, hins⟩ := ih D hD hb'
        refine ⟨j + 1, ?_, ?_⟩
        · simp [ins, hl, List.findIdx?_cons, hbf, hj]
        · simp only [ins, hl, if_true]
          rw [List.insertIdx_succ_cons, hins]

theorem mem_ins (shape : T → HopShape N) : ∀ (t : List T) (D : List (N × T)) (x : T), x ∈ ins shape D t →
    x ∈ t ∨ ∃ d ∈ D, d.2 = x := by
  intro t
  induction t with
  | nil =>
    intro D x hx
    simp only [ins, List.mem_map] at hx
    obtain ⟨d, hd, rfl⟩ := hx
    exact Or.inr ⟨d, hd, rfl⟩
  | cons b t ih =>
    intro D x hx
    unfold ins at hx
    split at hx
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact Or.inl List.mem_cons_self
      · rcases ih D x hx with h | h
        · exact Or.inl (List.mem_cons_of_mem _ h)
        · exact Or.inr h
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact Or.inl List.mem_cons_self
      · rcases List.mem_append.1 hx with hx | hx
        · obtain ⟨d, hd, h1, -⟩ := flush_pushed_mem _ D x hx
          exact Or.inr ⟨d, hd, h1⟩
        · rcases ih _ x hx with h | ⟨d, hd, h1⟩
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · exact Or.inr ⟨d, (flush_kept_sub _ D d hd).1, h1⟩

/-- The old loop walks over hops whose node an earlier hop already touches. -/
theorem skip_touched (shape : T → HopShape N) (bound : N → Bool) : ∀ (P pre rest : List T) (fuel : Nat),
    (∀ p ∈ P, ∃ x ∈ pre, touches (shape x) (shape p).frm = true) → P.length ≤ fuel →
    pre3102_loop shape bound fuel (pre ++ P ++ rest) pre.length =
      pre3102_loop shape bound (fuel - P.length) (pre ++ P ++ rest) (pre ++ P).length := by
  intro P
  induction P with
  | nil => intro pre rest fuel _ _; simp
  | cons p P ih =>
    intro pre rest fuel hP hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    obtain ⟨x, hx, htx⟩ := hP p List.mem_cons_self
    have hget : (pre ++ p :: P ++ rest)[pre.length]? = some p := by simp
    have htake : (pre ++ p :: P ++ rest).take pre.length = pre := by simp
    have hany : (pre.any fun y => touches (shape y) (shape p).frm) = true := List.any_eq_true.2 ⟨x, hx, htx⟩
    rw [pre3102_loop, hget]
    simp only [htake, hany, Bool.not_true, Bool.and_false, Bool.false_eq_true, if_false]
    have e : pre ++ p :: P ++ rest = (pre ++ [p]) ++ P ++ rest := by simp
    rw [e, show pre.length + 1 = (pre ++ [p]).length by simp,
      ih (pre ++ [p]) rest f (fun q hq => by
        obtain ⟨y, hy, h⟩ := hP q (List.mem_cons_of_mem _ hq)
        exact ⟨y, List.mem_append_left _ hy, h⟩) (by simp at hf; omega)]
    simp only [List.length_cons, List.append_assoc, List.singleton_append]
    rw [show f + 1 - (P.length + 1) = f - P.length by omega]

/-! ## The new loop equals the old one -/

section
variable (hops : List T) (shape : T → HopShape N) (bound : N → Bool)

/-- A hop flagged `self_loop` starts and ends at the same node. The planner sets
`self_loop = from.id == to.id` with nodes `(id, scope_id)`; this is the premise that an alias
id names one variable within a pattern. -/
def WF : Prop := ∀ x ∈ hops, (shape x).selfLoop = true → (shape x).frm = (shape x).to

/-- What the old loop's `rels[..i]` scan sees of `out`. -/
structure InvR (m : Nat) (out : List T) : Prop where
  A : ∀ k < m, ∀ y, hops[k]? = some y → movable (hops.map shape) k = false → y ∈ out
  B : ∀ x ∈ out, ∀ n, bound n = false → ((hops.drop m).map shape).any (isBinder n) = true →
      touches (shape x) n = true →
      ∃ k < m, ∃ y, hops[k]? = some y ∧ touches (shape y) n = true ∧ movable (hops.map shape) k = false

theorem wf_touch (hwf : WF hops shape) (x : T) (hx : x ∈ hops) (hl : (shape x).selfLoop = true) (n : N)
    (ht : touches (shape x) n = true) : n = (shape x).frm := by
  have := hwf x hx hl
  simp [touches] at ht
  rcases ht with h | h
  · exact h.symm
  · rw [this]; exact h.symm

theorem mem_of_inv (m : Nat) (D : List (N × T)) (out : List T) (hi : Inv hops shape m D out) :
    (∀ x ∈ out, x ∈ hops) ∧ (∀ d ∈ D, d.2 ∈ hops) := by
  have hp := hi.perm
  refine ⟨fun x hx => ?_, fun d hd => ?_⟩
  · exact List.mem_of_mem_take (hp.mem_iff.1 (List.mem_append_left _ hx))
  · exact List.mem_of_mem_take (hp.mem_iff.1 (List.mem_append_right _ (List.mem_map_of_mem hd)))

theorem invr_step (hwf : WF hops shape) (m : Nat) (h : T) (t : List T) (ht : hops.drop m = h :: t)
    (D : List (N × T)) (out : List T) (hi : Inv hops shape m D out) (hr : InvR hops shape bound m out) :
    InvR hops shape bound (m + 1) (step (hops.map shape) bound (D, out) ((h, shape h), m)).2 := by
  have hdr : hops.drop (m + 1) = t := drop_succ_of_drop hops m h t ht
  have hget : hops[m]? = some h := getElem?_of_drop hops m h t ht
  have hs : (hops.map shape)[m]? = some (shape h) := by simp [hget]
  have hmem := mem_of_inv hops shape m D out hi
  have hbinder : ∀ n, (t.map shape).any (isBinder n) = true → ((hops.drop m).map shape).any (isBinder n) = true := by
    intro n hn; rw [ht]; simp only [List.map_cons, List.any_cons, hn, Bool.or_true]
  -- old elements of `out`
  have oldB : ∀ x ∈ out, ∀ n, bound n = false → ((hops.drop (m + 1)).map shape).any (isBinder n) = true →
      touches (shape x) n = true →
      ∃ k < m + 1, ∃ y, hops[k]? = some y ∧ touches (shape y) n = true ∧ movable (hops.map shape) k = false := by
    intro x hx n hb hn htx
    rw [hdr] at hn
    obtain ⟨k, hk, y, h1, h2, h3⟩ := hr.B x hx n hb (hbinder n hn) htx
    exact ⟨k, by omega, y, h1, h2, h3⟩
  have oldA : ∀ out' : List T, (∀ x ∈ out, x ∈ out') → (h ∈ out' ∨ movable (hops.map shape) m = true) →
      ∀ k < m + 1, ∀ y, hops[k]? = some y → movable (hops.map shape) k = false → y ∈ out' := by
    intro out' hsub hh k hk y hy hm
    rcases Nat.lt_or_ge k m with hk' | hk'
    · exact hsub y (hr.A k hk' y hy hm)
    · have : k = m := by omega
      subst this
      rw [hget] at hy; cases hy
      rcases hh with hh | hh
      · exact hh
      · rw [hh] at hm; cases hm
  cases hdef : deferP (hops.map shape) bound m (shape h)
  · -- `h` is kept
    have hkeptB : ∀ n, bound n = false → ((hops.drop (m + 1)).map shape).any (isBinder n) = true →
        touches (shape h) n = true →
        ∃ k < m + 1, ∃ y, hops[k]? = some y ∧ touches (shape y) n = true ∧ movable (hops.map shape) k = false := by
      intro n hb hn htn
      cases hmv : movable (hops.map shape) m
      · exact ⟨m, by omega, h, hget, htn, hmv⟩
      · have hsl : (shape h).selfLoop = true := by
          rw [movable_spec _ m _ hs] at hmv
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hmv; exact hmv.1.2
        have hn' : n = (shape h).frm := wf_touch hops shape hwf h (List.mem_of_getElem? hget) hsl n htn
        subst hn'
        have hbt : ((hops.map shape).drop (m + 1)).any (isBinder (shape h).frm) = true := by
          rw [← List.map_drop]; exact hn
        have hany : ((List.range m).any fun k => match (hops.map shape)[k]? with
            | some x => touches x (shape h).frm && !movable (hops.map shape) k
            | none => false) = true := by
          cases hc : ((List.range m).any fun k => match (hops.map shape)[k]? with
            | some x => touches x (shape h).frm && !movable (hops.map shape) k
            | none => false)
          · have e : deferP (hops.map shape) bound m (shape h) =
                (movable (hops.map shape) m && !bound (shape h).frm &&
                  !((List.range m).any fun k => match (hops.map shape)[k]? with
                    | some x => touches x (shape h).frm && !movable (hops.map shape) k
                    | none => false) &&
                  ((hops.map shape).drop (m + 1)).any (isBinder (shape h).frm)) := rfl
            rw [e, hmv, hb, hc, hbt] at hdef
            simp at hdef
          · rfl
        obtain ⟨k, hk, hkx⟩ := List.any_eq_true.1 hany
        have hk' := List.mem_range.1 hk
        cases hx : (hops.map shape)[k]? with
        | none => simp [hx] at hkx
        | some x =>
          simp only [hx, Bool.and_eq_true, Bool.not_eq_true'] at hkx
          have : ∃ y, hops[k]? = some y ∧ shape y = x := by
            simp only [List.getElem?_map] at hx
            cases hy : hops[k]? with
            | none => simp [hy] at hx
            | some y => simp [hy] at hx; exact ⟨y, rfl, hx⟩
          obtain ⟨y, hy, rfl⟩ := this
          exact ⟨k, by omega, y, hy, hkx.1, hkx.2⟩
    cases hl : (shape h).selfLoop
    · rw [step_keep_bind hops shape bound m h _ hdef hl]
      refine ⟨oldA _ (fun x hx => by simp [hx]) (Or.inl (by simp)), ?_⟩
      intro x hx n hb hn htx
      simp only [List.mem_append, List.mem_singleton] at hx
      rcases hx with (hx | rfl) | hx
      · exact oldB x hx n hb hn htx
      · exact hkeptB n hb hn htx
      · obtain ⟨d, hd, rfl, hdt⟩ := flush_pushed_mem (shape h) D x hx
        have hdl := hi.dl d hd
        have hn' : n = (shape d.2).frm :=
          wf_touch hops shape hwf d.2 (hmem.2 d hd) hdl.1 n htx
        rw [hn', ← hdl.2.1]
        refine ⟨m, by omega, h, hget, hdt, ?_⟩
        rw [movable_spec _ m _ hs]; simp [hl]
    · rw [step_keep_loop hops shape bound m h _ hdef hl]
      refine ⟨oldA _ (fun x hx => by simp [hx]) (Or.inl (by simp)), ?_⟩
      intro x hx n hb hn htx
      simp only [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · exact oldB x hx n hb hn htx
      · exact hkeptB n hb hn htx
  · -- `h` is deferred
    rw [step_defer hops shape bound m h _ hdef]
    have hmv : movable (hops.map shape) m = true := by simp [deferP] at hdef; exact hdef.1.1.1
    exact ⟨oldA _ (fun x hx => hx) (Or.inr hmv), oldB⟩

theorem ins_cons_head (D : List (N × T)) (h : T) (t : List T) :
    ∃ rest, ins shape D (h :: t) = h :: rest := by
  unfold ins; split
  · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

/-- **Simulation**: from the state after index `m`, the old loop on `out ++ ins D t` at
`i = |out|` ends where the new loop ends. -/
theorem sim (hwf : WF hops shape) : ∀ (t : List T) (m : Nat) (D : List (N × T)) (out : List T) (fuel : Nat),
    hops.drop m = t → 1 ≤ m → Inv hops shape m D out → InvR hops shape bound m out →
    2 * t.length + D.length ≤ fuel →
    pre3102_loop shape bound fuel (out ++ ins shape D t) out.length =
      (run shape (hops.map shape) bound t m (D, out)).2 := by
  intro t
  induction t with
  | nil =>
    intro m D out fuel ht _ hi _ _
    have hD : D = [] := by
      cases D with
      | nil => rfl
      | cons d _ => have := (hi.dl d List.mem_cons_self).2.2; simp [ht] at this
    subst hD
    simp only [ins, List.map_nil, List.append_nil, run_nil]
    cases fuel with
    | zero => rfl
    | succ f => simp [pre3102_loop]
  | cons h t ih =>
    intro m D out fuel ht hm1 hi hr hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have hdr : hops.drop (m + 1) = t := drop_succ_of_drop hops m h t ht
    have hget : hops[m]? = some h := getElem?_of_drop hops m h t ht
    have hs : (hops.map shape)[m]? = some (shape h) := by simp [hget]
    have hi' := inv_step hops shape bound m h t ht D out hi
    have hr' := invr_step hops shape bound hwf m h t ht D out hi hr
    have hmem := mem_of_inv hops shape m D out hi
    have hDl : ∀ d ∈ D, (shape d.2).selfLoop = true := fun d hd => (hi.dl d hd).1
    obtain ⟨rest, hrest⟩ := ins_cons_head shape D h t
    have hgetR : (out ++ ins shape D (h :: t))[out.length]? = some h := by rw [hrest]; simp
    have htake : (out ++ ins shape D (h :: t)).take out.length = out := by simp
    rw [run_cons, pre3102_loop, hgetR]
    simp only [htake]
    cases hdef : deferP (hops.map shape) bound m (shape h)
    · -- kept: the old loop steps past `h`
      have hstep : (if (shape h).selfLoop && (shape h).fixed && !bound (shape h).frm &&
            !(out.any fun x => touches (shape x) (shape h).frm) then
            match ((out ++ ins shape D (h :: t)).drop (out.length + 1)).findIdx?
                (fun x => isBinder (shape h).frm (shape x)) with
            | some j => pre3102_loop shape bound f
                (((out ++ ins shape D (h :: t)).eraseIdx out.length).insertIdx (out.length + j + 1) h) out.length
            | none => pre3102_loop shape bound f (out ++ ins shape D (h :: t)) (out.length + 1)
          else pre3102_loop shape bound f (out ++ ins shape D (h :: t)) (out.length + 1)) =
          pre3102_loop shape bound f (out ++ ins shape D (h :: t)) (out.length + 1) := by
        by_cases hc : ((shape h).selfLoop && (shape h).fixed && !bound (shape h).frm &&
            !(out.any fun x => touches (shape x) (shape h).frm)) = true
        · rw [if_pos hc]
          simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
          obtain ⟨⟨⟨hsl, hfx⟩, hbd⟩, hout⟩ := hc
          have hdrop : (out ++ ins shape D (h :: t)).drop (out.length + 1) = ins shape D t := by
            simp [ins, hsl]
          rw [hdrop]
          cases hfi : (ins shape D t).findIdx? (fun x => isBinder (shape h).frm (shape x)) with
          | none => rfl
          | some j =>
            exfalso
            obtain ⟨x, hx, hxb⟩ : ∃ x ∈ ins shape D t, isBinder (shape h).frm (shape x) = true := by
              have := List.findIdx?_eq_some_iff_getElem.1 hfi
              obtain ⟨hj, hjb, -⟩ := this
              exact ⟨_, List.getElem_mem hj, hjb⟩
            have hxt : x ∈ t := by
              rcases mem_ins shape t D x hx with h1 | ⟨d, hd, rfl⟩
              · exact h1
              · simp [isBinder, hDl d hd] at hxb
            have hbt : ((hops.map shape).drop (m + 1)).any (isBinder (shape h).frm) = true := by
              rw [← List.map_drop, hdr]
              exact List.any_eq_true.2 ⟨shape x, List.mem_map_of_mem hxt, hxb⟩
            have hnone : ((List.range m).any fun k => match (hops.map shape)[k]? with
                | some x => touches x (shape h).frm && !movable (hops.map shape) k
                | none => false) = false := by
              apply List.any_eq_false.2
              intro k hk hkx
              have hk' := List.mem_range.1 hk
              cases hx' : (hops.map shape)[k]? with
              | none => simp [hx'] at hkx
              | some x' =>
                simp only [hx', Bool.and_eq_true, Bool.not_eq_true'] at hkx
                simp only [List.getElem?_map] at hx'
                cases hy : hops[k]? with
                | none => simp [hy] at hx'
                | some y =>
                  simp [hy] at hx'
                  subst hx'
                  have hyo := hr.A k hk' y hy hkx.2
                  have : (out.any fun x => touches (shape x) (shape h).frm) = true :=
                    List.any_eq_true.2 ⟨y, hyo, hkx.1⟩
                  rw [hout] at this; cases this
            have : deferP (hops.map shape) bound m (shape h) = true := by
              simp only [deferP, movable_spec _ m _ hs, hsl, hfx, hbd, hnone, hbt]
              simp; omega
            rw [hdef] at this; cases this
        · rw [if_neg hc]
      rw [hstep]
      cases hl : (shape h).selfLoop
      · rw [step_keep_bind hops shape bound m h _ hdef hl]
        have hR : out ++ ins shape D (h :: t) =
            (out ++ [h]) ++ (flush (shape h) D).1 ++ ins shape (flush (shape h) D).2 t := by
          simp [ins, hl]
        have hlen := (flush_perm (shape h) D).length_eq
        simp only [List.length_append, List.length_map] at hlen
        rw [hR, show out.length + 1 = (out ++ [h]).length by simp,
          skip_touched shape bound _ _ _ f (fun p hp => by
            obtain ⟨d, hd, rfl, hdt⟩ := flush_pushed_mem (shape h) D p hp
            exact ⟨h, by simp, by rw [← (hi.dl d hd).2.1]; exact hdt⟩) (by simp at hf; omega)]
        have := ih (m + 1) (flush (shape h) D).2 (out ++ [h] ++ (flush (shape h) D).1) (f - (flush (shape h) D).1.length)
          hdr (by omega) (by simpa [step_keep_bind hops shape bound m h _ hdef hl] using hi')
          (by simpa [step_keep_bind hops shape bound m h _ hdef hl] using hr') (by simp at hf; omega)
        simp only [List.append_assoc] at this ⊢
        exact this
      · rw [step_keep_loop hops shape bound m h _ hdef hl]
        have hR : out ++ ins shape D (h :: t) = (out ++ [h]) ++ ins shape D t := by simp [ins, hl]
        rw [hR, show out.length + 1 = (out ++ [h]).length by simp]
        exact ih (m + 1) D (out ++ [h]) f hdr (by omega)
          (by simpa [step_keep_loop hops shape bound m h _ hdef hl] using hi')
          (by simpa [step_keep_loop hops shape bound m h _ hdef hl] using hr') (by simp at hf; omega)
    · -- deferred: the old loop moves `h` behind its binder
      obtain ⟨hsl, hfx, -, hbd, hearly, hbt⟩ := deferP_spec hops shape bound m h t ht hdef
      have hout : (out.any fun x => touches (shape x) (shape h).frm) = false := by
        apply List.any_eq_false.2
        intro x hx htx
        obtain ⟨k, hk, y, hy, hty, hmk⟩ := hr.B x hx (shape h).frm hbd
          (by rw [ht]; simp only [List.map_cons, List.any_cons, hbt, Bool.or_true]) htx
        have := hearly k hk (shape y) (by simp [hy]) hty
        rw [hmk] at this; cases this
      have hc : ((shape h).selfLoop && (shape h).fixed && !bound (shape h).frm &&
          !(out.any fun x => touches (shape x) (shape h).frm)) = true := by
        simp [hsl, hfx, hbd, hout]
      rw [if_pos hc]
      have hdrop : (out ++ ins shape D (h :: t)).drop (out.length + 1) = ins shape D t := by
        simp [ins, hsl]
      rw [hdrop]
      obtain ⟨j, hj, hins⟩ := ins_insert shape (shape h).frm h t D hDl hbt
      rw [hj]
      simp only
      have hR : ((out ++ ins shape D (h :: t)).eraseIdx out.length).insertIdx (out.length + j + 1) h =
          out ++ ins shape (D ++ [((shape h).frm, h)]) t := by
        have : ins shape D (h :: t) = h :: ins shape D t := by simp [ins, hsl]
        rw [this, List.eraseIdx_append_of_length_le (Nat.le_refl _), Nat.sub_self]
        simp only [List.eraseIdx_cons_zero]
        rw [Nat.add_assoc, insertIdx_append_left, hins]
      rw [hR, step_defer hops shape bound m h _ hdef]
      exact ih (m + 1) _ out f hdr (by omega)
        (by simpa [step_defer hops shape bound m h _ hdef] using hi')
        (by simpa [step_defer hops shape bound m h _ hdef] using hr') (by simp at hf ⊢; omega)

/-- **`eq_reference`**: the one-pass reorder returns exactly what the pre-#3102 loop returned
(the property `matches_reference_on_random_chains` samples 20 000 times). -/
theorem eq_reference (hwf : WF hops shape) :
    selfLoopsAfterBinders hops shape bound = pre3102_reference shape bound hops := by
  rw [early_return_sound]
  cases hops with
  | nil => simp [run_nil, pre3102_reference, pre3102_loop]
  | cons h0 t =>
    have hd0 : deferP ((h0 :: t).map shape) bound 0 (shape h0) = false := by simp [deferP, movable]
    have hs0 : step ((h0 :: t).map shape) bound ([], []) ((h0, shape h0), 0) = ([], [h0]) := by
      cases hl : (shape h0).selfLoop
      · rw [step_keep_bind (h0 :: t) shape bound 0 h0 _ hd0 hl]; simp [flush_nil]
      · rw [step_keep_loop (h0 :: t) shape bound 0 h0 _ hd0 hl]; simp
    rw [run_cons, hs0]
    have := sim (h0 :: t) shape bound hwf t 1 [] [h0] (2 * (h0 :: t).length + 1) (by simp) (Nat.le_refl _)
      ⟨by simp, by simp⟩
      ⟨fun k hk y hy _ => by
          have : k = 0 := by omega
          subst this; simp at hy; simp [hy],
        fun x hx n _ _ htx => by
          simp at hx; subst hx
          exact ⟨0, by omega, x, by simp, htx, by simp [movable]⟩⟩
      (by simp; omega)
    rw [← this, ins_nil]
    rfl

end

/-- The `WF` premise is needed: a hop *flagged* `self_loop` whose endpoints differ (never
built by the planner when alias ids are unique per pattern) makes the two loops disagree. -/
theorem wf_needed :
    let hops : List (HopShape Nat) :=
      [⟨0, 1, false, true⟩, ⟨5, 2, true, true⟩, ⟨2, 2, true, true⟩, ⟨1, 2, false, false⟩]
    selfLoopsAfterBinders hops id (fun _ => false) ≠ pre3102_reference id (fun _ => false) hops := by
  decide

/-- The PR's case `(a)-[:R]->(b)-[:S*]->(c)-[:L]->(c)` sorted `R, L, S*` becomes `R, S*, L`
(hop_order.rs:133-136); two self-loops on `c` go latest-first. -/
theorem pr_cases :
    selfLoopsAfterBinders ([⟨0, 1, false, true⟩, ⟨2, 2, true, true⟩, ⟨1, 2, false, false⟩] : List (HopShape Nat))
        id (fun _ => false) = [⟨0, 1, false, true⟩, ⟨1, 2, false, false⟩, ⟨2, 2, true, true⟩] ∧
    selfLoopsAfterBinders ([⟨0, 1, false, true⟩, ⟨2, 2, true, true⟩, ⟨2, 2, true, true⟩, ⟨1, 2, false, false⟩] :
        List (HopShape Nat)) id (fun _ => false) =
      [⟨0, 1, false, true⟩, ⟨1, 2, false, false⟩, ⟨2, 2, true, true⟩, ⟨2, 2, true, true⟩] ∧
    selfLoopsAfterBinders ([⟨0, 1, false, true⟩, ⟨2, 2, true, true⟩, ⟨1, 2, false, false⟩] : List (HopShape Nat))
        id (fun n => n == 2) = [⟨0, 1, false, true⟩, ⟨2, 2, true, true⟩, ⟨1, 2, false, false⟩] := by
  decide

end PlannerBuild.HopOrder
