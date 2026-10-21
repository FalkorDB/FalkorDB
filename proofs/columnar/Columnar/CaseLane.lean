import Columnar.AndOrLane

/-!
# `VectorEval::eval_case` (vector_expr.rs:477-555) against the per-row `eval_case`

The lane evaluates the subject over all rows, then each `WHEN` over the positions no arm has
claimed yet, its `THEN` over the positions it claims (scattered into `out`), and finally the
`ELSE` over what is still unclaimed.

| here | there |
| --- | --- |
| `splitHits` | the claim pass (:506-528): `compare_values(when, subject, Eq) == Some(true)` / not `false`/`null` (`caseMatch`) |
| `scatter` | `out[pos] = then.get(offset)` (:532-534, :547-549) |
| `armLoop` | `while arm + 1 < num_arms && !unclaimed.is_empty()` (:499-540) |
| `evalCase` | the whole lane, `Ok(ExprColumn::Values(out))` |

**`evalCase_sound`**: entry `i` of a returned column is the per-row `CASE` of row `rows[i]`
(expr_semantics' `evalNode (.case ..)`, eval.rs:495-528): a `WHEN` is evaluated columnarly for
exactly the rows still unclaimed — the rows whose per-row loop reaches it — and a `THEN` only
for the rows it claims.
-/

namespace Columnar.CaseLane
open FalkorCore (V)
open FalkorExpr.Eval (E Ops Compound Ev evalNode evalSubj evalArms caseHit)
open Columnar.CoreOps (RT coreOps)
open Columnar.AndOrLane (pairsOf pairsOf_fst pairsOf_mem)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

def updV (f : Nat → V F) (p : Nat) (v : V F) : Nat → V F := fun q => if q = p then v else f q

/-- `out[pos] = col.get(offset)` over `(pos, value)` pairs. -/
def scatter (out : Nat → V F) : List (Nat × V F) → Nat → V F
  | [] => out
  | (p, v) :: ps => scatter (updV out p v) ps

/-- The claim pass: positions whose `WHEN` value matches, and the rest. -/
def splitHits (rt : RT F) (subject : Option (EC F)) : List (Nat × V F) → List Nat × List Nat
  | [] => ([], [])
  | (pos, w) :: ps =>
    let r := splitHits rt subject ps
    if caseMatch (CoreOps.ocC rt) (subject.map (·.get pos)) w then (pos :: r.1, r.2) else (r.1, pos :: r.2)

structure CSt (F : Type) where
  out : Nat → V F
  unclaimed : List Nat

section
variable (rt : RT F) (ev : E (V F) → List Nat → Except String (EC F)) (rows : List Nat)

def rowOf (i : Nat) : Nat := rows[i]?.getD 0

/-- One `THEN` (or the `ELSE`) over the positions `ps`, scattered into `out`. -/
def fill (e : E (V F)) (out : Nat → V F) (ps : List Nat) : Except String (Nat → V F) :=
  if ps = [] then .ok out
  else do
    let col ← ev e (ps.map (rowOf rows))
    .ok (scatter out (pairsOf col.get ps 0))

/-- The arm loop (vector_expr.rs:499-540). -/
def armLoop (subject : Option (EC F)) : List (E (V F)) → CSt F → Except String (CSt F)
  | w :: t :: rest, st =>
    if st.unclaimed = [] then .ok st
    else do
      let wc ← ev w (st.unclaimed.map (rowOf rows))
      let hs := splitHits rt subject (pairsOf wc.get st.unclaimed 0)
      let out ← fill ev rows t st.out hs.1
      armLoop subject rest ⟨out, hs.2⟩
  | _, st => .ok st

/-- The subject column (vector_expr.rs:484-488). -/
def subjCol : Option (E (V F)) → Except String (Option (EC F))
  | none => .ok none
  | some s => (ev s rows).map some

/-- `eval_case` (vector_expr.rs:477). -/
def evalCase (subj : Option (E (V F))) (arms : List (E (V F))) (els : E (V F)) : Except String (EC F) := do
  let subject ← subjCol ev rows subj
  let st ← armLoop rt ev rows subject arms ⟨fun _ => .null, List.range rows.length⟩
  let out ← fill ev rows els st.out st.unclaimed
  .ok (.values ((List.range rows.length).map out))
end

/-! ## Lemmas on the passes -/

theorem scatter_out (out : Nat → V F) : ∀ (ps : List (Nat × V F)) (q : Nat), q ∉ ps.map Prod.fst →
    scatter out ps q = out q
  | [], _, _ => rfl
  | (p, v) :: ps, q, hq => by
    simp only [List.map_cons, List.mem_cons, not_or] at hq
    simp only [scatter]
    rw [scatter_out _ ps q hq.2]
    simp [updV, hq.1]

theorem scatter_in (out : Nat → V F) : ∀ (ps : List (Nat × V F)) (p : Nat) (v : V F),
    (ps.map Prod.fst).Nodup → (p, v) ∈ ps → scatter out ps p = v
  | [], _, _, _, h => by simp at h
  | (q, w) :: ps, p, v, hnd, hm => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    simp only [scatter]
    rcases List.mem_cons.1 hm with he | hm
    · simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      rw [scatter_out _ ps p hnd.1]; simp [updV]
    · exact scatter_in _ ps p v hnd.2 hm

theorem splitHits_spec (rt : RT F) (subject : Option (EC F)) : ∀ (ps : List (Nat × V F)),
    let r := splitHits rt subject ps
    (∀ p, p ∈ r.1 ↔ ∃ w, (p, w) ∈ ps ∧ caseMatch (CoreOps.ocC rt) (subject.map (·.get p)) w = true) ∧
    (∀ p, p ∈ r.2 ↔ ∃ w, (p, w) ∈ ps ∧ caseMatch (CoreOps.ocC rt) (subject.map (·.get p)) w = false) ∧
    r.1.Sublist (ps.map Prod.fst) ∧ r.2.Sublist (ps.map Prod.fst)
  | [] => by simp [splitHits]
  | (pos, w) :: ps => by
    obtain ⟨h1, h2, h3, h4⟩ := splitHits_spec rt subject ps
    simp only [splitHits]
    split
    · rename_i hm
      refine ⟨fun p => ?_, fun p => ?_, h3.cons_cons _, h4.cons _⟩
      · simp only [List.mem_cons, h1, Prod.mk.injEq]
        constructor
        · rintro (rfl | ⟨w', hw, hc⟩)
          · exact ⟨w, .inl ⟨rfl, rfl⟩, hm⟩
          · exact ⟨w', .inr hw, hc⟩
        · rintro ⟨w', (⟨rfl, rfl⟩ | hw), hc⟩
          · exact .inl rfl
          · exact .inr ⟨w', hw, hc⟩
      · simp only [h2, List.mem_cons, Prod.mk.injEq]
        constructor
        · rintro ⟨w', hw, hc⟩; exact ⟨w', .inr hw, hc⟩
        · rintro ⟨w', (⟨rfl, rfl⟩ | hw), hc⟩
          · simp [hm] at hc
          · exact ⟨w', hw, hc⟩
    · rename_i hm
      simp only [Bool.not_eq_true] at hm
      refine ⟨fun p => ?_, fun p => ?_, h3.cons _, h4.cons_cons _⟩
      · simp only [h1, List.mem_cons, Prod.mk.injEq]
        constructor
        · rintro ⟨w', hw, hc⟩; exact ⟨w', .inr hw, hc⟩
        · rintro ⟨w', (⟨rfl, rfl⟩ | hw), hc⟩
          · simp [hm] at hc
          · exact ⟨w', hw, hc⟩
      · simp only [List.mem_cons, h2, Prod.mk.injEq]
        constructor
        · rintro (rfl | ⟨w', hw, hc⟩)
          · exact ⟨w, .inl ⟨rfl, rfl⟩, hm⟩
          · exact ⟨w', .inr hw, hc⟩
        · rintro ⟨w', (⟨rfl, rfl⟩ | hw), hc⟩
          · exact .inl rfl
          · exact .inr ⟨w', hw, hc⟩

/-! ## Soundness -/

section
variable (rt : RT F) (ev0 : Ev (V F)) (c : Compound (V F)) (envOf : Nat → Option (Nat → Option (V F)))
  (ev : E (V F) → List Nat → Except String (EC F)) (rows : List Nat) (Rows : Nat → Prop)

abbrev KidOK := Columnar.AndOrLane.KidOK rt ev0 c envOf ev Rows

theorem pairsOf_vals (get : Nat → V F) (ps : List Nat) (e : E (V F))
    (hk : ∀ i (h : i < (ps.map (rowOf rows)).length),
      evalNode (coreOps rt) ev0 c (envOf (ps.map (rowOf rows))[i]) e = .ok (get i)) :
    ∀ p w, (p, w) ∈ pairsOf get ps 0 → evalNode (coreOps rt) ev0 c (envOf (rowOf rows p)) e = .ok w := by
  intro p w hm
  obtain ⟨j, hj, hpj, hw⟩ := pairsOf_mem get ps 0 p w hm
  have := hk j (by simpa using hj)
  simp only [List.getElem_map, hpj, Nat.zero_add] at this hw
  rw [hw]; exact this

theorem rows_ok (hrows : ∀ r ∈ rows, Rows r) (ps : List Nat) (hps : ps.Sublist (List.range rows.length)) :
    ∀ r ∈ ps.map (rowOf rows), Rows r := by
  intro r hr
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hr
  have := List.mem_range.1 (hps.subset hq)
  simp only [rowOf, List.getElem?_eq_getElem this, Option.getD_some]
  exact hrows _ (List.getElem_mem _)

theorem fill_spec (hrows : ∀ r ∈ rows, Rows r) (e : E (V F)) (he : KidOK rt ev0 c envOf ev Rows e)
    (out out' : Nat → V F) (ps : List Nat) (hps : ps.Sublist (List.range rows.length))
    (h : fill ev rows e out ps = .ok out') :
    ∀ q, (q ∉ ps → out' q = out q) ∧
      (q ∈ ps → evalNode (coreOps rt) ev0 c (envOf (rowOf rows q)) e = .ok (out' q)) := by
  unfold fill at h
  split at h
  · rename_i h0; simp only [Except.ok.injEq] at h; subst h; subst h0; simp
  · obtain ⟨col, hcol, h⟩ := VecEval.bind_ok h
    simp only [Except.ok.injEq] at h; subst h
    have hnd : ((pairsOf col.get ps 0).map Prod.fst).Nodup := by
      rw [pairsOf_fst]; exact hps.nodup List.nodup_range
    have hv := pairsOf_vals rt ev0 c envOf rows col.get ps e
      (he _ col (rows_ok rows Rows hrows ps hps) hcol)
    intro q
    constructor
    · intro hq; exact scatter_out out _ q (by rw [pairsOf_fst]; exact hq)
    · intro hq
      obtain ⟨⟨q', w⟩, hm, hq'⟩ := List.mem_map.1 (show q ∈ (pairsOf col.get ps 0).map Prod.fst by
        rw [pairsOf_fst]; exact hq)
      simp only at hq'; subst hq'
      rw [scatter_in out _ q' w hnd hm]; exact hv q' w hm

/-- The rest of the per-row `CASE` from arm list `ks` on, given the subject value. -/
def Rest (sv : Nat → Option (V F)) (els : E (V F)) (p : Nat) (ks : List (E (V F))) : Except String (V F) := do
  let r ← evalArms (coreOps rt) ev0 c (envOf (rowOf rows p)) (sv p) ks
  match r with
  | some v => .ok v
  | none => evalNode (coreOps rt) ev0 c (envOf (rowOf rows p)) els

theorem armLoop_sound (hrows : ∀ r ∈ rows, Rows r) (subject : Option (EC F)) (els : E (V F))
    (R0 : Nat → Except String (V F)) :
    ∀ (ks : List (E (V F))) (st st' : CSt F), (∀ ch ∈ ks, KidOK rt ev0 c envOf ev Rows ch) →
    st.unclaimed.Sublist (List.range rows.length) →
    (∀ p < rows.length,
      (p ∈ st.unclaimed → R0 p = Rest rt ev0 c envOf rows (fun p => subject.map (·.get p)) els p ks) ∧
      (p ∉ st.unclaimed → R0 p = .ok (st.out p))) →
    armLoop rt ev rows subject ks st = .ok st' →
    st'.unclaimed.Sublist (List.range rows.length) ∧
    ∀ p < rows.length,
      (p ∈ st'.unclaimed → R0 p = evalNode (coreOps rt) ev0 c (envOf (rowOf rows p)) els) ∧
      (p ∉ st'.unclaimed → R0 p = .ok (st'.out p))
  | [], st, st', _, hsub, hinv, h => by
    simp only [armLoop, Except.ok.injEq] at h; subst h
    refine ⟨hsub, fun p hp => ⟨fun hu => ?_, (hinv p hp).2⟩⟩
    rw [(hinv p hp).1 hu]; simp [Rest, evalArms, bind, Except.bind]
  | [w], st, st', _, hsub, hinv, h => by
    simp only [armLoop, Except.ok.injEq] at h; subst h
    refine ⟨hsub, fun p hp => ⟨fun hu => ?_, (hinv p hp).2⟩⟩
    rw [(hinv p hp).1 hu]; simp [Rest, evalArms, bind, Except.bind]
  | w :: t :: rest, st, st', hk, hsub, hinv, h => by
    simp only [armLoop] at h
    split at h
    · rename_i he
      simp only [Except.ok.injEq] at h; subst h
      refine ⟨hsub, fun p hp => ⟨fun hu => by rw [he] at hu; simp at hu, (hinv p hp).2⟩⟩
    · obtain ⟨wc, hwc, h⟩ := VecEval.bind_ok h
      obtain ⟨out, hout, h⟩ := VecEval.bind_ok h
      have hwv := pairsOf_vals rt ev0 c envOf rows wc.get st.unclaimed w
        (hk w (by simp) _ wc (rows_ok rows Rows hrows _ hsub) hwc)
      obtain ⟨hs1, hs2, hsub1, hsub2⟩ := splitHits_spec rt subject (pairsOf wc.get st.unclaimed 0)
      rw [pairsOf_fst] at hsub1 hsub2
      have hfill := fill_spec rt ev0 c envOf ev rows Rows hrows t (hk t (by simp)) st.out out _
        (hsub1.trans hsub) hout
      refine armLoop_sound hrows subject els R0 rest ⟨out, _⟩ st' (fun x hx => hk x (by simp [hx]))
        (hsub2.trans hsub) ?_ h
      intro p hp
      have hnd : ((pairsOf wc.get st.unclaimed 0).map Prod.fst).Nodup := by
        rw [pairsOf_fst]; exact hsub.nodup List.nodup_range
      -- the `WHEN` value at a position, and the claim verdict
      have hpair : ∀ q ∈ st.unclaimed, ∃ wv, (q, wv) ∈ pairsOf wc.get st.unclaimed 0 := by
        intro q hq
        obtain ⟨⟨q', wv⟩, hm, hq'⟩ := List.mem_map.1 (show q ∈ (pairsOf wc.get st.unclaimed 0).map Prod.fst by
          rw [pairsOf_fst]; exact hq)
        simp only at hq'; subst hq'; exact ⟨wv, hm⟩
      have huniq : ∀ q wv wv', (q, wv) ∈ pairsOf wc.get st.unclaimed 0 → (q, wv') ∈ pairsOf wc.get st.unclaimed 0 →
          wv = wv' := by
        intro q wv wv' h1 h2
        have := hwv q wv h1; rw [hwv q wv' h2] at this; injection this with this; exact this.symm
      have hrest : ∀ q ∈ st.unclaimed, ∀ wv, (q, wv) ∈ pairsOf wc.get st.unclaimed 0 →
          Rest rt ev0 c envOf rows (fun p => subject.map (·.get p)) els q (w :: t :: rest) =
            if caseMatch (CoreOps.ocC rt) (subject.map (·.get q)) wv then
              evalNode (coreOps rt) ev0 c (envOf (rowOf rows q)) t
            else Rest rt ev0 c envOf rows (fun p => subject.map (·.get p)) els q rest := by
        intro q _ wv hm
        have hcase := CoreOps.caseHit_caseMatch rt (subject.map (·.get q)) wv
        simp only [Rest, evalArms, hwv q wv hm, bind, Except.bind]
        rw [hcase]
        cases caseMatch (CoreOps.ocC rt) (subject.map (·.get q)) wv
        · rfl
        · simp only [ite_true]
          cases evalNode (coreOps rt) ev0 c (envOf (rowOf rows q)) t <;> rfl
      by_cases hu : p ∈ st.unclaimed
      · obtain ⟨wv, hm⟩ := hpair p hu
        have hR := (hinv p hp).1 hu
        rw [hrest p hu wv hm] at hR
        by_cases hc : caseMatch (CoreOps.ocC rt) (subject.map (·.get p)) wv
        · have hin1 : p ∈ (splitHits rt subject (pairsOf wc.get st.unclaimed 0)).1 := (hs1 p).2 ⟨wv, hm, hc⟩
          have hnot2 : p ∉ (splitHits rt subject (pairsOf wc.get st.unclaimed 0)).2 := by
            intro h2; obtain ⟨wv', hm', hc'⟩ := (hs2 p).1 h2
            rw [huniq p wv wv' hm hm'] at hc; simp [hc] at hc'
          refine ⟨fun h2 => absurd h2 hnot2, fun _ => ?_⟩
          rw [hR, if_pos hc]; exact (hfill p).2 hin1
        · simp only [Bool.not_eq_true] at hc
          have hin2 : p ∈ (splitHits rt subject (pairsOf wc.get st.unclaimed 0)).2 := (hs2 p).2 ⟨wv, hm, hc⟩
          refine ⟨fun _ => ?_, fun h2 => absurd hin2 h2⟩
          rw [hR, if_neg (by simp [hc])]
      · have hnot1 : p ∉ (splitHits rt subject (pairsOf wc.get st.unclaimed 0)).1 :=
          fun h1 => hu (hsub1.subset h1)
        have hnot2 : p ∉ (splitHits rt subject (pairsOf wc.get st.unclaimed 0)).2 :=
          fun h2 => hu (hsub2.subset h2)
        refine ⟨fun h2 => absurd h2 hnot2, fun _ => ?_⟩
        rw [(hinv p hp).2 hu, ← (hfill p).1 hnot1]

/-- **`eval_case` = the per-row `CASE`, at every row** (given sound sub-expressions). -/
theorem evalCase_sound (hrows : ∀ r ∈ rows, Rows r) (subj : Option (E (V F))) (arms : List (E (V F)))
    (els : E (V F)) (col : EC F)
    (hk : ∀ ch ∈ subj.toList ++ arms ++ [els], KidOK rt ev0 c envOf ev Rows ch)
    (h : evalCase rt ev rows subj arms els = .ok col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      evalNode (coreOps rt) ev0 c (envOf rows[i]) (.case subj arms els) = .ok (col.get i) := by
  unfold evalCase at h
  obtain ⟨subject, hsj, h⟩ := VecEval.bind_ok h
  obtain ⟨st, hst, h⟩ := VecEval.bind_ok h
  obtain ⟨out, hout, h⟩ := VecEval.bind_ok h
  simp only [Except.ok.injEq] at h; subst h
  -- the subject, per row
  have hsv : ∀ p < rows.length, evalSubj (coreOps rt) ev0 c (envOf (rowOf rows p)) subj =
      .ok (subject.map (·.get p)) := by
    intro p hp
    cases subj with
    | none => simp only [subjCol, Except.ok.injEq] at hsj; subst hsj; rfl
    | some s =>
      simp only [subjCol] at hsj
      cases hs : ev s rows with
      | error e => simp [hs, Except.map] at hsj
      | ok scol =>
        simp only [hs, Except.map, Except.ok.injEq] at hsj; subst hsj
        have := hk s (by simp) rows scol hrows hs p hp
        simp only [evalSubj, rowOf, List.getElem?_eq_getElem hp, Option.getD_some, this, bind, Except.bind]
        rfl
  have hR0 : ∀ p < rows.length, evalNode (coreOps rt) ev0 c (envOf (rowOf rows p)) (.case subj arms els) =
      Rest rt ev0 c envOf rows (fun p => subject.map (·.get p)) els p arms := by
    intro p hp
    simp only [evalNode, hsv p hp, Rest, bind, Except.bind]
    cases evalArms (coreOps rt) ev0 c (envOf (rowOf rows p)) (subject.map (·.get p)) arms with
    | error e => rfl
    | ok v => cases v <;> rfl
  obtain ⟨hsub, hfin⟩ := armLoop_sound rt ev0 c envOf ev rows Rows hrows subject els
    (fun p => evalNode (coreOps rt) ev0 c (envOf (rowOf rows p)) (.case subj arms els)) arms _ st
    (fun x hx => hk x (by simp [hx])) (List.Sublist.refl _)
    (fun p hp => ⟨fun _ => hR0 p hp, fun h' => absurd (List.mem_range.2 hp) h'⟩) hst
  have hfill := fill_spec rt ev0 c envOf ev rows Rows hrows els (hk els (by simp)) st.out out _ hsub hout
  refine ⟨by simp [EC.WF], fun i hi => ?_⟩
  have hrow : rowOf rows i = rows[i] := by simp [rowOf, List.getElem?_eq_getElem hi]
  simp only [EC.get, List.getElem?_map, List.getElem?_range hi, Option.map_some, Option.getD_some]
  rw [← hrow]
  by_cases hu : i ∈ st.unclaimed
  · rw [(hfin i hi).1 hu]; exact (hfill i).2 hu
  · rw [(hfin i hi).2 hu, (hfill i).1 hu]
end

end Columnar.CaseLane
