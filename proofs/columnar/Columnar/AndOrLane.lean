import Columnar.CoreOps

/-!
# `VectorEval::eval_and_or` (vector_expr.rs:391-475) against the per-row `AND` / `OR`

The lane keeps, per position into `rows`, the verdict `result[pos]` (starting at the neutral
element `!sc`), a null bit, and the `live` positions; each child is evaluated over the live
rows only, then one pass over its column settles the rows whose value is `Bool(sc)`, marks the
nulls, and drops the settled rows from `live` (a non-boolean, non-null value is a type error).

| here | there |
| --- | --- |
| `pairsOf` | `for (offset, &pos) in live.iter().enumerate()` with `vector.get(offset)` |
| `scanP` | the per-position pass (:454-470); the `Bools` fast path (:428-447) performs the same transition on `ExprColumn::get` (a `Bools` cell reads `Null` under its null bit and `Bool(mask[offset])` otherwise), so one pass models both |
| `loop` | the child loop (:409-472): `break` on an empty `live`; `child_rows` is `live.map(rows[·])` (the `live.len() == len` shortcut hands `rows` itself, the same list since `live` is then `0..len`) |
| `evalAndOr` | the whole lane: `Ok(ExprColumn::Bools(result, nulls))` |

**`evalAndOr_sound`**: if the lane returns a column and every child's columns are sound,
entry `i` is the per-row `AND`/`OR` of row `rows[i]` — expr_semantics' `andOrLoop` (eval.rs
:639-717), the short-circuiting loop — for every row: a row's child is evaluated columnarly
exactly when the per-row loop reaches it (the row is still live).
-/

namespace Columnar.AndOrLane
open FalkorCore (V)
open FalkorExpr.Eval (E Ops Compound Ev evalNode andOrLoop)
open Columnar.CoreOps (RT coreOps)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

def upd (f : Nat → Bool) (p : Nat) (b : Bool) : Nat → Bool := fun q => if q = p then b else f q

/-- `live.iter().enumerate()` paired with the child column's entries. -/
def pairsOf (get : Nat → V F) : List Nat → Nat → List (Nat × V F)
  | [], _ => []
  | p :: ps, k => (p, get k) :: pairsOf get ps (k + 1)

/-- The pass over the live positions (vector_expr.rs:428-470). -/
def scanP (sc : Bool) : List (Nat × V F) → (Nat → Bool) → (Nat → Bool) →
    Except String ((Nat → Bool) × (Nat → Bool) × List Nat)
  | [], res, nul => .ok (res, nul, [])
  | (pos, v) :: ps, res, nul =>
    match v with
    | .bool b =>
      if b = sc then scanP sc ps (upd res pos sc) (upd nul pos false)
      else (scanP sc ps res nul).map fun t => (t.1, t.2.1, pos :: t.2.2)
    | .null => (scanP sc ps res (upd nul pos true)).map fun t => (t.1, t.2.1, pos :: t.2.2)
    | v => .error s!"Type mismatch: expected Boolean or Null but was {v.name}"

structure St where
  res : Nat → Bool
  nul : Nat → Bool
  live : List Nat

/-- The child loop (vector_expr.rs:409-472). -/
def loop (ev : E (V F) → List Nat → Except String (EC F)) (sc : Bool) (rows : List Nat) :
    List (E (V F)) → St → Except String St
  | [], st => .ok st
  | ch :: cs, st =>
    if st.live = [] then .ok st
    else do
      let col ← ev ch (st.live.map fun i => rows[i]?.getD 0)
      let t ← scanP sc (pairsOf col.get st.live 0) st.res st.nul
      loop ev sc rows cs ⟨t.1, t.2.1, t.2.2⟩

/-- `eval_and_or` (vector_expr.rs:391). -/
def evalAndOr (ev : E (V F) → List Nat → Except String (EC F)) (cs : List (E (V F))) (rows : List Nat)
    (sc : Bool) : Except String (EC F) := do
  let st ← loop ev sc rows cs ⟨fun _ => !sc, fun _ => false, List.range rows.length⟩
  .ok (.bools ((List.range rows.length).map st.res) ((List.range rows.length).map st.nul))

/-! ## The pass -/

theorem pairsOf_fst (get : Nat → V F) : ∀ (L : List Nat) (k : Nat), (pairsOf get L k).map Prod.fst = L
  | [], _ => rfl
  | p :: ps, k => by simp [pairsOf, pairsOf_fst get ps (k + 1)]

theorem pairsOf_mem (get : Nat → V F) : ∀ (L : List Nat) (k : Nat) (p : Nat) (v : V F),
    (p, v) ∈ pairsOf get L k → ∃ j, ∃ h : j < L.length, L[j] = p ∧ v = get (k + j)
  | [], _, _, _, h => by simp [pairsOf] at h
  | q :: qs, k, p, v, h => by
    simp only [pairsOf, List.mem_cons, Prod.mk.injEq] at h
    rcases h with ⟨rfl, rfl⟩ | h
    · exact ⟨0, by simp, rfl, by simp⟩
    · obtain ⟨j, hj, h1, h2⟩ := pairsOf_mem get qs (k + 1) p v h
      exact ⟨j + 1, by simp; omega, by simpa using h1, by rw [h2]; congr 1; omega⟩

/-- What one pass does to each position. -/
theorem scanP_spec (sc : Bool) : ∀ (ps : List (Nat × V F)) (res nul res' nul' : Nat → Bool) (L : List Nat),
    (ps.map Prod.fst).Nodup → scanP sc ps res nul = .ok (res', nul', L) →
    L.Sublist (ps.map Prod.fst) ∧
    (∀ p, p ∉ ps.map Prod.fst → res' p = res p ∧ nul' p = nul p) ∧
    (∀ p v, (p, v) ∈ ps →
      (v = .bool sc ∧ p ∉ L ∧ res' p = sc ∧ nul' p = false) ∨
      ((∃ b, v = .bool b ∧ b ≠ sc) ∧ p ∈ L ∧ res' p = res p ∧ nul' p = nul p) ∨
      (v = .null ∧ p ∈ L ∧ res' p = res p ∧ nul' p = true))
  | [], res, nul, res', nul', L, _, h => by
    simp only [scanP, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨by simp, fun _ _ => ⟨rfl, rfl⟩, fun _ _ hm => by simp at hm⟩
  | (pos, v) :: ps, res, nul, res', nul', L, hnd, h => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    obtain ⟨hpos, hnd⟩ := hnd
    have hne : ∀ q v, (q, v) ∈ ps → q ≠ pos := fun q v hq heq =>
      hpos (heq ▸ List.mem_map.2 ⟨(q, v), hq, rfl⟩)
    cases v with
    | bool b =>
      simp only [scanP] at h
      by_cases hb : b = sc
      · subst hb
        simp only [ite_true] at h
        obtain ⟨hs, hout, hin⟩ := scanP_spec b ps _ _ res' nul' L hnd h
        refine ⟨hs.trans (List.sublist_cons_self _ _), fun p hp => ?_, fun p w hm => ?_⟩
        · simp only [List.map_cons, List.mem_cons, not_or] at hp
          have := hout p hp.2; simp [upd, hp.1] at this; exact this
        · rcases List.mem_cons.1 hm with he | hm
          · simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
            have := hout p hpos
            refine .inl ⟨rfl, fun hL => hpos (hs.subset hL), ?_, ?_⟩ <;> simp [this, upd]
          · have hq := hne p w hm
            rcases hin p w hm with h1 | h1 | h1
            · exact .inl h1
            · obtain ⟨hb', hL, hr, hn⟩ := h1
              exact .inr (.inl ⟨hb', hL, by simp [hr, upd, hq], by simp [hn, upd, hq]⟩)
            · obtain ⟨hb', hL, hr, hn⟩ := h1
              exact .inr (.inr ⟨hb', hL, by simp [hr, upd, hq], hn⟩)
      · simp only [hb, ite_false] at h
        cases hr : scanP sc ps res nul with
        | error e => simp [hr, Except.map] at h
        | ok t =>
          obtain ⟨r1, n1, L1⟩ := t
          simp only [hr, Except.map, Except.ok.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl, rfl⟩ := h
          obtain ⟨hs, hout, hin⟩ := scanP_spec sc ps res nul r1 n1 L1 hnd hr
          refine ⟨hs.cons_cons _, fun p hp => ?_, fun p w hm => ?_⟩
          · simp only [List.map_cons, List.mem_cons, not_or] at hp; exact hout p hp.2
          · rcases List.mem_cons.1 hm with he | hm
            · simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
              exact .inr (.inl ⟨⟨b, rfl, hb⟩, by simp, hout p hpos⟩)
            · rcases hin p w hm with h1 | h1 | h1
              · refine .inl ⟨h1.1, ?_, h1.2.2⟩
                simp only [List.mem_cons, not_or]; exact ⟨hne p w hm, h1.2.1⟩
              · exact .inr (.inl ⟨h1.1, by simp [h1.2.1], h1.2.2⟩)
              · exact .inr (.inr ⟨h1.1, by simp [h1.2.1], h1.2.2⟩)
    | null =>
      simp only [scanP] at h
      cases hr : scanP sc ps res (upd nul pos true) with
      | error e => simp [hr, Except.map] at h
      | ok t =>
        obtain ⟨r1, n1, L1⟩ := t
        simp only [hr, Except.map, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        obtain ⟨hs, hout, hin⟩ := scanP_spec sc ps res _ r1 n1 L1 hnd hr
        refine ⟨hs.cons_cons _, fun p hp => ?_, fun p w hm => ?_⟩
        · simp only [List.map_cons, List.mem_cons, not_or] at hp
          have := hout p hp.2; simp [upd, hp.1] at this; exact this
        · rcases List.mem_cons.1 hm with he | hm
          · simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
            have := hout p hpos
            exact .inr (.inr ⟨rfl, by simp, this.1, by simp [this.2, upd]⟩)
          · have hq := hne p w hm
            rcases hin p w hm with h1 | h1 | h1
            · refine .inl ⟨h1.1, ?_, h1.2.2⟩
              simp only [List.mem_cons, not_or]; exact ⟨hq, h1.2.1⟩
            · obtain ⟨hb', hL, hr', hn⟩ := h1
              exact .inr (.inl ⟨hb', by simp [hL], hr', by simp [hn, upd, hq]⟩)
            · exact .inr (.inr ⟨h1.1, by simp [h1.2.1], h1.2.2⟩)
    | _ => simp [scanP] at h

/-! ## The child loop against the per-row loop -/

section
variable (rt : RT F) (ev0 : Ev (V F)) (c : Compound (V F)) (envOf : Nat → Option (Nat → Option (V F)))
  (ev : E (V F) → List Nat → Except String (EC F)) (sc : Bool) (rows : List Nat)

/-- The per-row `AND` (`sc = false`) / `OR` (`sc = true`) loop of row `rows[p]`. -/
def R (p : Nat) (ks : List (E (V F))) (n : Bool) : Except String (V F) :=
  andOrLoop (coreOps rt) ev0 c (envOf (rows[p]?.getD 0)) sc n ks

/-- A child's columns are its per-row values (the `Sound` contract, values half). -/
def KidOK (Rows : Nat → Prop) (ch : E (V F)) : Prop :=
  ∀ rs col, (∀ r ∈ rs, Rows r) → ev ch rs = .ok col →
    ∀ i (h : i < rs.length), evalNode (coreOps rt) ev0 c (envOf rs[i]) ch = .ok (col.get i)

theorem loop_sound (Rows : Nat → Prop) (hrows : ∀ r ∈ rows, Rows r) (cs0 : List (E (V F))) :
    ∀ (ks : List (E (V F))) (st st' : St), (∀ ch ∈ ks, KidOK rt ev0 c envOf ev Rows ch) →
    st.live.Sublist (List.range rows.length) →
    (∀ p < rows.length,
      (p ∈ st.live → st.res p = !sc ∧ R rt ev0 c envOf sc rows p cs0 false = R rt ev0 c envOf sc rows p ks (st.nul p)) ∧
      (p ∉ st.live → st.res p = sc ∧ st.nul p = false ∧ R rt ev0 c envOf sc rows p cs0 false = .ok (.bool sc))) →
    loop ev sc rows ks st = .ok st' →
    ∀ p < rows.length, R rt ev0 c envOf sc rows p cs0 false = .ok (if st'.nul p then .null else .bool (st'.res p))
  | [], st, st', _, _, hinv, h => by
    simp only [loop, Except.ok.injEq] at h; subst h
    intro p hp
    by_cases hl : p ∈ st.live
    · obtain ⟨hr, hR⟩ := (hinv p hp).1 hl
      rw [hR]; simp only [R, andOrLoop, hr]
      cases st.nul p <;> rfl
    · obtain ⟨hr, hn, hR⟩ := (hinv p hp).2 hl
      rw [hR, hn, hr]; rfl
  | ch :: ks, st, st', hk, hsub, hinv, h => by
    simp only [loop] at h
    split at h
    · rename_i he
      simp only [Except.ok.injEq] at h; subst h
      intro p hp
      have hl : p ∉ st.live := by rw [he]; simp
      obtain ⟨hr, hn, hR⟩ := (hinv p hp).2 hl
      rw [hR, hn, hr]; rfl
    · obtain ⟨col, hcol, h⟩ := VecEval.bind_ok h
      obtain ⟨t, ht, h⟩ := VecEval.bind_ok h
      obtain ⟨r1, n1, L1⟩ := t
      have hnd : ((pairsOf col.get st.live 0).map Prod.fst).Nodup := by
        rw [pairsOf_fst]; exact hsub.nodup List.nodup_range
      obtain ⟨hs, hout, hin⟩ := scanP_spec sc _ _ _ r1 n1 L1 hnd ht
      rw [pairsOf_fst] at hs hout
      -- the child's value at every live position
      have hlt : ∀ q ∈ st.live, q < rows.length := fun q hq => List.mem_range.1 (hsub.subset hq)
      have hval : ∀ p v, (p, v) ∈ pairsOf col.get st.live 0 →
          evalNode (coreOps rt) ev0 c (envOf (rows[p]?.getD 0)) ch = .ok v := by
        intro p v hm
        obtain ⟨j, hj, hpj, hv⟩ := pairsOf_mem col.get st.live 0 p v hm
        have := hk ch (by simp) _ col (fun r hr => by
          obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hr
          exact hrows _ (by rw [List.getElem?_eq_getElem (hlt q hq)]; exact List.getElem_mem _)) hcol j
          (by simpa using hj)
        simp only [List.getElem_map, hpj, Nat.zero_add] at this hv
        rw [hv]; exact this
      have hmem : ∀ p ∈ st.live, ∃ v, (p, v) ∈ pairsOf col.get st.live 0 := by
        intro p hp
        obtain ⟨⟨q, v⟩, hm, hq⟩ := List.mem_map.1 (show p ∈ (pairsOf col.get st.live 0).map Prod.fst by
          rw [pairsOf_fst]; exact hp)
        simp only at hq; subst hq; exact ⟨v, hm⟩
      refine loop_sound Rows hrows cs0 ks ⟨r1, n1, L1⟩ st' (fun x hx => hk x (by simp [hx]))
        (hs.trans hsub) ?_ h
      intro p hp
      by_cases hl : p ∈ st.live
      · obtain ⟨v, hm⟩ := hmem p hl
        obtain ⟨hr0, hR0⟩ := (hinv p hp).1 hl
        have hv := hval p v hm
        rcases hin p v hm with ⟨rfl, hL, hr, hn⟩ | ⟨⟨b, rfl, hb⟩, hL, hr, hn⟩ | ⟨rfl, hL, hr, hn⟩
        · refine ⟨fun h' => absurd h' hL, fun _ => ⟨hr, hn, ?_⟩⟩
          rw [hR0]; simp [R, andOrLoop, hv, bind, Except.bind, CoreOps.boolOf]
        · refine ⟨fun _ => ⟨by dsimp only; rw [hr, hr0], ?_⟩, fun h' => absurd hL h'⟩
          dsimp only; rw [hR0, hn]; simp [R, andOrLoop, hv, bind, Except.bind, CoreOps.boolOf, hb]
        · refine ⟨fun _ => ⟨by dsimp only; rw [hr, hr0], ?_⟩, fun h' => absurd hL h'⟩
          dsimp only; rw [hR0, hn]; simp [R, andOrLoop, hv, bind, Except.bind, CoreOps.boolOf]
      · have hL : p ∉ L1 := fun h' => hl (hs.subset h')
        obtain ⟨hr, hn⟩ := hout p hl
        obtain ⟨hr0, hn0, hR0⟩ := (hinv p hp).2 hl
        exact ⟨fun h' => absurd h' hL, fun _ => ⟨by dsimp only; rw [hr, hr0], by dsimp only; rw [hn, hn0], hR0⟩⟩

/-- **`eval_and_or` = the per-row short-circuit loop, at every row** (given sound children). -/
theorem evalAndOr_sound (Rows : Nat → Prop) (cs : List (E (V F))) (col : EC F)
    (hk : ∀ ch ∈ cs, KidOK rt ev0 c envOf ev Rows ch) (hrows : ∀ r ∈ rows, Rows r)
    (h : evalAndOr ev cs rows sc = .ok col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      andOrLoop (coreOps rt) ev0 c (envOf rows[i]) sc false cs = .ok (col.get i) := by
  unfold evalAndOr at h
  obtain ⟨st, hst, h⟩ := VecEval.bind_ok h
  simp only [Except.ok.injEq] at h; subst h
  have := loop_sound rt ev0 c envOf ev sc rows Rows hrows cs cs _ st hk (List.Sublist.refl _)
    (fun p hp => ⟨fun _ => ⟨rfl, rfl⟩, fun h' => absurd (List.mem_range.2 hp) h'⟩) hst
  refine ⟨by simp [EC.WF], fun i hi => ?_⟩
  have h2 := this i hi
  simp only [R, List.getElem?_eq_getElem hi, Option.getD_some] at h2
  rw [h2, bools_get]
  simp [Bits.isNull, List.getElem?_range hi]

end

end Columnar.AndOrLane
