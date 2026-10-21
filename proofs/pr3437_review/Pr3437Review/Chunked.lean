import Pr3437Review.ListLoop

/-!
# PR #3437: the columnar half and the chunk loop (`list_loop_columnar`, `list_loop_chunk`)

`columnar` is `list_loop_columnar` (eval.rs:1550-1652) with the vector evaluator abstracted
to `Col` over the chunk's batch: `colP chunk` = `ve.eval(predicate, &batch, &all)` (:1569),
`colJ chunk passed` = `ve.eval(projection, &batch, &passed)` (:1630), `colJv chunk passed` =
`ve.eval_values(projection, &batch, &passed)` (:1643) — `passed` the row indices that passed
(`passedIdx`, :1617-1628), as in Rust (since 2026-10-21; the model used to hand the
projection the passed *elements*). Its only assumption is `Col.Sound`: a column the vector evaluator *returns* holds,
row by row, what the per-element evaluator returns (`Bridge.lean` derives it from columnar's
`Dispatch.eval_sound`). Nothing is assumed when it errors: `chunkStep` then re-runs the
chunk per element, which is main's loop (`perElem_eq`).

Every `?` in `list_loop_columnar` precedes the write to `sink` it guards (:1604-1606 counts
into `local` first; :1632-1634, :1642-1645 evaluate before `+=`/`extend`), so an `Err` leaves
`sink` as it was: `columnar` returns a new sink or an error, never a partial write.
-/

namespace Pr3437.ListLoop
open FalkorCore FalkorStrList

variable {F : Type}

/-- `ExprColumn` as read back by `ExprColumn::get`: the `Bools(mask, nulls)` lane cell by
cell (`(mask[i], nulls.is_null(i))`), or any other lane through its `get`. -/
inductive EC (F : Type) where
  | bools (cells : List (Bool × Bool))
  | vals (vs : List (V F))

def cell : Bool × Bool → V F
  | (b, nl) => if nl then .null else .bool b

def EC.vals' : EC F → List (V F)
  | .bools cs => cs.map cell
  | .vals vs => vs

structure Col (F : Type) where
  colP : List (V F) → Except String (EC F)
  colJ : List (V F) → List Nat → Except String (EC F)
  colJv : List (V F) → List Nat → Except String (List (V F))

/-- The vector evaluator's contract (`Bridge.colSound`): a column it returns over the chunk's
batch holds main's per-element values of the elements at those rows. -/
structure Col.Sound (c : Col F) (pe pj : V F → Except String (V F)) : Prop where
  pred : ∀ xs col, c.colP xs = .ok col → xs.mapM pe = .ok col.vals'
  proj : ∀ xs rows col, c.colJ xs rows = .ok col → (∀ r ∈ rows, r < xs.length) →
    ∃ ys, (rows.map (xs.getD · .null)).mapM pj = .ok ys
  projV : ∀ xs rows ys, c.colJv xs rows = .ok ys → (∀ r ∈ rows, r < xs.length) →
    (rows.map (xs.getD · .null)).mapM pj = .ok ys

/-- The elements whose predicate value is `Bool(true)`, in order. -/
def filterPass (chunk ps : List (V F)) : List (V F) :=
  ((chunk.zip ps).filter fun q => isTrue q.2).map (·.1)

/-- One quantifier count step on a predicate value (eval.rs:1585-1594). -/
def cstep : Nat × Nat × Nat → V F → Except String (Nat × Nat × Nat)
  | (t, f, n), v =>
    match kOf v with
    | some .t => .ok (t + 1, f, n)
    | some .f => .ok (t, f + 1, n)
    | some .n => .ok (t, f, n + 1)
    | none => .error (boolMismatch v)

/-- `local` (eval.rs:1573-1597). -/
def countLocal (len : Nat) : Option (EC F) → Except String (Nat × Nat × Nat)
  | none => .ok (len, 0, 0)
  | some (.bools cs) =>
    .ok (cs.foldl (fun (acc : Nat × Nat × Nat) (q : Bool × Bool) =>
      let (t, f, n) := acc
      if q.2 then (t, f, n + 1) else if q.1 then (t + 1, f, n) else (t, f + 1, n)) (0, 0, 0))
  | some (.vals vs) => vs.foldlM cstep (0, 0, 0)

/-- eval.rs:1604-1615. -/
def tallyNoProj (len : Nat) : Option (EC F) → Nat
  | none => len
  | some (.bools cs) => (cs.filter fun q => q.1 && !q.2).length
  | some (.vals vs) => (vs.filter isTrue).length

/-- `passed` as elements (`passed.iter().map(|&i| chunk[i])`, eval.rs:1639). -/
def passedOf (chunk : List (V F)) : Option (EC F) → List (V F)
  | none => chunk
  | some (.bools cs) => ((chunk.zip cs).filter fun q => q.2.1 && !q.2.2).map (·.1)
  | some (.vals vs) => filterPass chunk vs

/-- `passed` (eval.rs:1617-1628): `all.retain(|&i| mask[i] && !nulls.is_null(i))` /
`all.retain(|&i| matches!(column.get(i), Value::Bool(true)))`. -/
def passedIdx (len : Nat) : Option (EC F) → List Nat
  | none => List.range len
  | some (.bools cs) => (List.range len).filter fun i => match cs[i]? with
    | some q => q.1 && !q.2
    | none => false
  | some (.vals vs) => (List.range len).filter fun i => isTrue (vs.getD i .null)

/-- `list_loop_columnar` (eval.rs:1550-1652). -/
def columnar (b : Body F) (c : Col F) (s : Sink F) (chunk : List (V F)) : Except String (Sink F) :=
  (if b.hasPred then (c.colP chunk).map some else pure none) >>= fun tested =>
  match s with
  | .count t f n =>
    (countLocal chunk.length tested).map fun r => .count (t + r.1) (f + r.2.1) (n + r.2.2)
  | .tally k =>
    if !b.hasProj then pure (.tally (k + tallyNoProj chunk.length tested))
    else
      (if (passedIdx chunk.length tested).isEmpty then pure ()
       else (c.colJ chunk (passedIdx chunk.length tested)).map fun _ => ())
        >>= fun _ => pure (.tally (k + (passedIdx chunk.length tested).length))
  | .collect acc =>
    if !b.hasProj then
      if (passedIdx chunk.length tested).length = chunk.length then pure (.collect (acc ++ chunk))
      else pure (.collect (acc ++ (passedIdx chunk.length tested).map (chunk.getD · .null)))
    else if (passedIdx chunk.length tested).isEmpty then pure (.collect acc)
    else (c.colJv chunk (passedIdx chunk.length tested)).map fun ys => .collect (acc ++ ys)

/-! ## Main's loop over precomputed predicate values -/

section
variable (pe pj : V F → Except String (V F))

theorem mapM_cons_ok {v : V F} {vs : List (V F)} {ps : List (V F)}
    (h : (v :: vs).mapM pe = .ok ps) : ∃ p ps', ps = p :: ps' ∧ pe v = .ok p ∧ vs.mapM pe = .ok ps' := by
  simp only [List.mapM_cons] at h
  cases hv : pe v with
  | error e => simp [hv, bind, Except.bind] at h
  | ok p =>
    simp only [hv, bind, Except.bind] at h
    cases hvs : vs.mapM pe with
    | error e => simp [hvs] at h
    | ok ps' =>
      simp only [hvs, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨p, ps', h.symm, rfl, rfl⟩

theorem mapM_length {f : V F → Except String (V F)} : ∀ {xs ys : List (V F)}, xs.mapM f = .ok ys → ys.length = xs.length
  | [], ys, h => by simp [pure, Except.pure] at h; subst h; rfl
  | x :: xs, ys, h => by
    obtain ⟨p, ps', rfl, _, h'⟩ := mapM_cons_ok f h
    simp [mapM_length h']

theorem ref_collect_pre : ∀ (chunk ps acc : List (V F)), chunk.mapM pe = .ok ps →
    ref pe pj (.collect acc) chunk = ((filterPass chunk ps).mapM pj).map fun ys => .collect (acc ++ ys)
  | [], ps, acc, h => by
    simp [pure, Except.pure] at h; subst h; simp [ref_nil, filterPass, pure, Except.pure, Except.map]
  | v :: vs, ps, acc, h => by
    obtain ⟨p, ps', rfl, hp, hvs⟩ := mapM_cons_ok pe h
    rw [ref_cons]; simp only [refStep, hp, bind, Except.bind]
    by_cases ht : isTrue p = true
    · simp only [ht, ite_true, filterPass, List.zip_cons_cons, List.filter_cons, List.map_cons,
        List.mapM_cons]
      cases hj : pj v with
      | error e => simp [bind, Except.bind, Except.map]
      | ok y =>
        simp only [pure, Except.pure, bind, Except.bind]
        rw [ref_collect_pre vs ps' (acc ++ [y]) hvs]
        simp only [filterPass]
        cases (List.map (·.1) (List.filter (fun q => isTrue q.2) (vs.zip ps'))).mapM pj <;>
          simp [Except.map]
    · simp only [ht, Bool.false_eq_true, ite_false, pure, Except.pure, filterPass, List.zip_cons_cons,
        List.filter_cons]
      exact ref_collect_pre vs ps' acc hvs

theorem ref_tally_pre : ∀ (chunk ps : List (V F)) (k : Nat), chunk.mapM pe = .ok ps →
    ref pe pj (.tally k) chunk =
      ((filterPass chunk ps).mapM pj).map fun _ => .tally (k + (filterPass chunk ps).length)
  | [], ps, k, h => by
    simp [pure, Except.pure] at h; subst h; simp [ref_nil, filterPass, pure, Except.pure, Except.map]
  | v :: vs, ps, k, h => by
    obtain ⟨p, ps', rfl, hp, hvs⟩ := mapM_cons_ok pe h
    rw [ref_cons]; simp only [refStep, hp, bind, Except.bind]
    by_cases ht : isTrue p = true
    · simp only [ht, ite_true, filterPass, List.zip_cons_cons, List.filter_cons, List.map_cons,
        List.mapM_cons, List.length_cons]
      cases hj : pj v with
      | error e => simp [bind, Except.bind, Except.map]
      | ok y =>
        simp only [pure, Except.pure, bind, Except.bind]
        rw [ref_tally_pre vs ps' (k + 1) hvs]
        simp only [filterPass]
        cases (List.map (·.1) (List.filter (fun q => isTrue q.2) (vs.zip ps'))).mapM pj <;>
          simp [Except.map]; omega
    · simp only [ht, Bool.false_eq_true, ite_false, pure, Except.pure, filterPass, List.zip_cons_cons,
        List.filter_cons]
      exact ref_tally_pre vs ps' k hvs

theorem cstep_fold_shift : ∀ (ps : List (V F)) (t f n : Nat),
    ps.foldlM cstep (t, f, n) = (ps.foldlM cstep (0, 0, 0)).map fun r => (t + r.1, f + r.2.1, n + r.2.2)
  | [], t, f, n => by simp [pure, Except.pure, Except.map]
  | p :: ps, t, f, n => by
    simp only [List.foldlM_cons]
    cases hk : kOf p with
    | none => simp [cstep, hk, bind, Except.bind, Except.map]
    | some k =>
      cases k
      · simp only [cstep, hk, bind, Except.bind]
        rw [cstep_fold_shift ps (t + 1) f n, cstep_fold_shift ps 1 0 0]
        cases ps.foldlM cstep (0, 0, 0) <;> simp [Except.map] <;> omega
      · simp only [cstep, hk, bind, Except.bind]
        rw [cstep_fold_shift ps t (f + 1) n, cstep_fold_shift ps 0 1 0]
        cases ps.foldlM cstep (0, 0, 0) <;> simp [Except.map] <;> omega
      · simp only [cstep, hk, bind, Except.bind]
        rw [cstep_fold_shift ps t f (n + 1), cstep_fold_shift ps 0 0 1]
        cases ps.foldlM cstep (0, 0, 0) <;> simp [Except.map] <;> omega

theorem ref_count_pre : ∀ (chunk ps : List (V F)) (t f n : Nat), chunk.mapM pe = .ok ps →
    ref pe pj (.count t f n) chunk = (ps.foldlM cstep (t, f, n)).map fun r => .count r.1 r.2.1 r.2.2
  | [], ps, t, f, n, h => by
    simp [pure, Except.pure] at h; subst h; simp [ref_nil, pure, Except.pure, Except.map]
  | v :: vs, ps, t, f, n, h => by
    obtain ⟨p, ps', rfl, hp, hvs⟩ := mapM_cons_ok pe h
    rw [ref_cons]; simp only [refStep, hp, bind, Except.bind, List.foldlM_cons]
    cases hk : kOf p with
    | none => simp [cstep, hk, throw, throwThe, MonadExceptOf.throw, Except.map]
    | some k =>
      cases k <;> simp only [cstep, hk, bump, pure, Except.pure] <;> exact ref_count_pre vs ps' _ _ _ hvs

theorem mapM_id {g : V F → Except String (V F)} (hg : ∀ v, g v = .ok v) :
    ∀ xs : List (V F), xs.mapM g = .ok xs
  | [] => rfl
  | x :: xs => by simp [List.mapM_cons, hg x, mapM_id hg xs, bind, Except.bind, pure, Except.pure]

theorem filterPass_length : ∀ (chunk ps : List (V F)), chunk.length = ps.length →
    (filterPass chunk ps).length = (ps.filter isTrue).length
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | v :: vs, p :: ps, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    have ih := filterPass_length vs ps h
    simp only [filterPass] at ih ⊢
    simp only [List.zip_cons_cons, List.filter_cons]
    split <;> simp_all

theorem filterPass_le : ∀ (chunk ps : List (V F)), (filterPass chunk ps).length ≤ chunk.length
  | [], _ => by simp [filterPass]
  | _ :: _, [] => by simp [filterPass]
  | v :: vs, p :: ps => by
    have ih := filterPass_le vs ps
    simp only [filterPass] at ih ⊢
    simp only [List.zip_cons_cons, List.filter_cons]
    split <;> simp_all <;> omega

theorem filterPass_full : ∀ (chunk ps : List (V F)), (filterPass chunk ps).length = chunk.length →
    filterPass chunk ps = chunk
  | [], _, _ => by simp [filterPass]
  | v :: vs, [], h => by simp [filterPass] at h
  | v :: vs, p :: ps, h => by
    have hle := filterPass_le vs ps
    simp only [filterPass, List.zip_cons_cons, List.filter_cons] at h ⊢
    by_cases ht : isTrue p = true
    · simp only [ht, ite_true, List.map_cons, List.length_cons] at h ⊢
      have := filterPass_full vs ps (by simpa [filterPass] using h)
      simp only [filterPass] at this; rw [this]
    · simp only [ht, Bool.false_eq_true, ite_false, List.length_cons] at h
      simp only [filterPass] at hle; omega

theorem filterPass_trues : ∀ chunk : List (V F), filterPass chunk (chunk.map fun _ => .bool true) = chunk
  | [] => rfl
  | v :: vs => by
    have := filterPass_trues vs
    simp only [filterPass] at this ⊢
    simp only [List.map_cons, List.zip_cons_cons, List.filter_cons]
    have ht : isTrue (V.bool true : V F) = true := rfl
    simp only [ht, ite_true, List.map_cons]; rw [this]

theorem cstep_trues : ∀ chunk : List (V F),
    (chunk.map fun _ => (V.bool true : V F)).foldlM cstep (0, 0, 0) = .ok (chunk.length, 0, 0)
  | [] => rfl
  | v :: vs => by
    simp only [List.map_cons, List.foldlM_cons, cstep, kOf, bind, Except.bind]
    rw [cstep_fold_shift, cstep_trues vs]; simp [Except.map]; omega

theorem isTrue_cell (q : Bool × Bool) : isTrue (cell q : V F) = (q.1 && !q.2) := by
  rcases q with ⟨_ | _, _ | _⟩ <;> rfl

theorem passed_bools : ∀ (chunk : List (V F)) (cs : List (Bool × Bool)),
    ((chunk.zip cs).filter fun q => q.2.1 && !q.2.2).map (·.1) = filterPass chunk (cs.map cell)
  | [], _ => by simp [filterPass]
  | _ :: _, [] => by simp [filterPass]
  | v :: vs, c :: cs => by
    have ih := passed_bools vs cs
    simp only [filterPass] at ih ⊢
    simp only [List.map_cons, List.zip_cons_cons, List.filter_cons, isTrue_cell]
    split <;> simp_all

theorem count_bools : ∀ (cs : List (Bool × Bool)) (t f n : Nat),
    Except.ok (cs.foldl (fun (acc : Nat × Nat × Nat) (q : Bool × Bool) =>
      let (t, f, n) := acc
      if q.2 then (t, f, n + 1) else if q.1 then (t + 1, f, n) else (t, f + 1, n)) (t, f, n)) =
    ((cs.map cell : List (V F)).foldlM cstep (t, f, n))
  | [], t, f, n => rfl
  | (b, nl) :: cs, t, f, n => by
    simp only [List.foldl_cons, List.map_cons, List.foldlM_cons]
    rcases b with _ | _ <;> rcases nl with _ | _ <;>
      simp only [cell, cstep, kOf, bind, Except.bind, Bool.false_eq_true, ite_false, ite_true] <;>
      exact count_bools cs _ _ _

theorem tally_bools : ∀ (chunk : List (V F)) (cs : List (Bool × Bool)), chunk.length = cs.length →
    (cs.filter fun q => q.1 && !q.2).length = (filterPass chunk (cs.map cell)).length := by
  intro chunk cs h
  rw [filterPass_length chunk _ (by simp [h]), List.filter_map, List.length_map]
  congr 1; apply List.filter_congr; intro q _; simp [isTrue_cell]

end

end Pr3437.ListLoop
