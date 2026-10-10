import FalkorIdList.Wire
/-! # Pushing onto a decoded `IdList` keeps every id in order (#2920)

`from_segments` (`id_list.rs:1183`) adopts decoded segments verbatim and, since
#2920 (`2ef102ae0`), starts the run *after* all of them. `Inv` (the builder's
invariant) need not hold there — the last decoded segment may be a range outside
the run — so this file carries a second invariant, `DInv`, for "the run is empty
and starts past the decoded segments; a direction, if claimed, belongs to the
last decoded segment, which a push extended in place". Every push from a `DInv`
state lands in `DInv` or in `Inv`, and from `Inv` `push_ok` takes over.

Headline: `decoded_then_pushed_keeps_order` — for every decision sequence of
`prefers_bitmap`, pushing `xs` onto any well-formed decoded segment list yields
exactly `flat segs ++ xs`, and the collapse `assert!` never fires.
-/
namespace IdListPush
open Seg

structure DInv (st : St) : Prop where
  start_eq : st.start = st.segs.length
  wf : ∀ s ∈ st.segs, WF s
  dir : st.desc = none ∨
    (st.desc = some false ∧ ∃ b l, st.segs.getLast? = some (.range b l) ∧ 2 ≤ l) ∨
    (st.desc = some true ∧ ∃ b l, st.segs.getLast? = some (.rdesc b l) ∧ 2 ≤ l)

/-- `from_segments` of well-formed segments satisfies `DInv`. -/
theorem fromSegments_dinv (segs : List Seg) (hw : ∀ s ∈ segs, WF s) : DInv (fromSegments segs) :=
  ⟨rfl, hw, .inl rfl⟩

/-- A fresh `Range{id,1}` appended past the end opens a run of its own. -/
theorem inv_fresh (init : List Seg) (len : Nat) (d : Option Bool) (hd : d ≠ none ∨ d = none)
    (hw : ∀ s ∈ init, WF s) (id : Nat) (hid : id < W) :
    Inv ⟨init ++ [.range id 1], len, init.length, d⟩ := by
  refine inv_of (Nat.le_refl _) hw (by simp [WF]; omega) (by simp [RL]) (by simp) ?_
  rw [List.drop_length, List.nil_append]
  cases d with
  | none => exact Or.inr ⟨id, rfl⟩
  | some d => exact runOk_single _ (by simp) id

/-- The tail of `push` from a state whose run is empty and starts at the end. -/
theorem pushTail_fresh (c : Bool) (init : List Seg) (last : Seg) (len : Nat) (desc : Option Bool)
    (hw : ∀ s ∈ init ++ [last], WF s) (id : Nat) (hid : id < W) :
    ∃ st', push.pushTail c ⟨init ++ [last], len, (init ++ [last]).length, desc⟩ last id = some st' ∧
      Inv st' ∧ flat st'.segs = flat (init ++ [last]) ++ [id] ∧ st'.len = len := by
  have cont : ∀ d : Bool,
      ∃ st', collapse c ⟨init ++ [last] ++ [.range id 1], len, (init ++ [last]).length, some d⟩ = some st' ∧
        Inv st' ∧ flat st'.segs = flat (init ++ [last]) ++ [id] ∧ st'.len = len := by
    intro d
    obtain ⟨st', h1, h2, h3, h4⟩ :=
      collapse_ok c _ (inv_fresh (init ++ [last]) len (some d) (.inl (by simp)) hw id hid) d rfl
    exact ⟨st', h1, h2, by rw [h3, flat_snoc]; rfl, h4⟩
  have restart : Inv ⟨init ++ [last] ++ [.range id 1], len, (init ++ [last]).length, none⟩ :=
    inv_fresh _ len none (.inr rfl) hw id hid
  have hf : flat (init ++ [last] ++ [.range id 1]) = flat (init ++ [last]) ++ [id] := by
    rw [flat_snoc]; rfl
  cases desc with
  | none =>
    simp only [push.pushTail]
    split
    · exact cont _
    · exact ⟨_, rfl, restart, hf, rfl⟩
  | some d =>
    cases d with
    | false =>
      simp only [push.pushTail]
      split
      · exact cont false
      · exact ⟨_, rfl, restart, hf, rfl⟩
    | true =>
      simp only [push.pushTail]
      split
      · exact cont true
      · exact ⟨_, rfl, restart, hf, rfl⟩

theorem range'_snoc (b l : Nat) : List.range' b (l + 1) = List.range' b l ++ [b + l] := by
  induction l generalizing b with
  | zero => simp
  | succ n ih => rw [List.range'_succ, ih (b + 1), List.range'_succ]; simp; omega

/-- **One push from a decoded state** keeps the ids in order and lands in `DInv`
or `Inv`. -/
theorem push_dinv (c : Bool) (st0 : St) (hd : DInv st0) (id : Nat) (hid : id < W) :
    ∃ st', push c st0 id = some st' ∧ (DInv st' ∨ Inv st') ∧
      flat st'.segs = flat st0.segs ++ [id] ∧ st'.len = st0.len + 1 := by
  obtain ⟨segs, len, start, desc⟩ := st0
  obtain ⟨hs, hw, hdir⟩ := hd
  dsimp only at hs hw hdir; subst hs
  rcases List.eq_nil_or_concat segs with h | ⟨init, last, h⟩
  · subst h
    have hp : push c ⟨[], len, 0, desc⟩ id = some ⟨[.range id 1], len + 1, 0, none⟩ := by
      simp [push, claimStep]
    refine ⟨_, hp, .inr ?_, by simp [flat, iter], rfl⟩
    have := @inv_of [] (.range id 1) (len + 1) 0 none (by simp) (by simp) (by simp [WF]; exact hid)
      (by simp [RL]) (by simp) (Or.inr ⟨id, rfl⟩)
    simpa using this
  rw [List.concat_eq_append] at h; subst h
  have hwl : WF last := hw last (by simp)
  have hwi : ∀ s ∈ init, WF s := fun s hs => hw s (by simp [hs])
  -- the tail, reached whenever `claimStep` did nothing
  have tail : ∀ st : St, st = ⟨init ++ [last], len + 1, (init ++ [last]).length, desc⟩ →
      ∃ st', push.pushTail c st last id = some st' ∧ (DInv st' ∨ Inv st') ∧
        flat st'.segs = flat (init ++ [last]) ++ [id] ∧ st'.len = len + 1 := by
    intro st hst; subst hst
    obtain ⟨st', h1, h2, h3, h4⟩ := pushTail_fresh c init last (len + 1) desc hw id hid
    exact ⟨st', h1, .inr h2, h3, h4⟩
  -- a direction is claimed only for a last segment of length ≥ 2
  have hnone : (∀ b l, last = .range b l → l < 2) → (∀ b l, last = .rdesc b l → l < 2) →
      desc = none := by
    intro h1 h2
    rcases hdir with h | ⟨_, b, l, hl, h2l⟩ | ⟨_, b, l, hl, h2l⟩
    · exact h
    · simp at hl; have := h1 b l hl; omega
    · simp at hl; have := h2 b l hl; omega
  have hlast : (init ++ [last]).getLast? = some last := by simp
  cases last with
  | range b l =>
    simp [WF] at hwl
    by_cases hl1 : l = 1
    · subst hl1
      have hdn : desc = none := hnone (fun b' l' h => by cases h; omega) (fun _ _ h => by cases h)
      subst hdn
      by_cases hb : b + 1 = id
      · -- claim ascending, extend to `Range{b,2}`
        subst hb
        refine ⟨⟨init ++ [.range b 2], len + 1, (init ++ [Seg.range b 1]).length, some false⟩,
          by simp [push, claimStep, claim, extend?], .inl ⟨by simp, ?_, .inr (.inl ⟨rfl, b, 2, by simp, by omega⟩)⟩,
          ?_, rfl⟩
        · intro s hs; simp at hs; rcases hs with hs | rfl
          · exact hwi s hs
          · simp [WF]; omega
        · simp only [flat_snoc, iter, List.append_assoc]; simp [List.range'_succ]
      · by_cases hd1 : id + 1 = b
        · -- `Range{b,1}` then `b-1`: rewritten to `RangeDescending{b,2}`, claim descending
          refine ⟨⟨init ++ [.rdesc b 2], len + 1, (init ++ [Seg.range b 1]).length, some true⟩,
            by simp [push, claimStep, extend?, hb, hd1, claim], .inl ⟨by simp, ?_, .inr (.inr ⟨rfl, b, 2, by simp, by omega⟩)⟩,
            ?_, rfl⟩
          · intro s hs; simp at hs; rcases hs with hs | rfl
            · exact hwi s hs
            · simp [WF]; omega
          · simp only [flat_snoc, iter, List.append_assoc]
            rw [show b - (2 - 1) = id by omega]
            simp [List.range'_succ]; omega
        · by_cases hb2 : b = id
          · subst hb2
            refine ⟨⟨init ++ [.rep b 2], len + 1, (init ++ [Seg.rep b 2]).length, none⟩,
              by simp [push, claimStep, extend?], .inl ⟨rfl, ?_, .inl rfl⟩, ?_, rfl⟩
            · intro s hs; simp at hs; rcases hs with hs | rfl
              · exact hwi s hs
              · simp [WF]; omega
            · simp only [flat_snoc, iter, List.append_assoc]; simp
          · have hp : push c ⟨init ++ [.range b 1], len, (init ++ [Seg.range b 1]).length, none⟩ id =
                push.pushTail c ⟨init ++ [.range b 1], len + 1, (init ++ [Seg.range b 1]).length, none⟩
                  (.range b 1) id := by
              simp [push, claimStep, extend?, hb, hd1, hb2]
            rw [hp]; exact tail _ rfl
    · by_cases he : b + l = id
      · subst he
        have hcs : claimStep ⟨init ++ [.range b l], len + 1, (init ++ [Seg.range b l]).length, desc⟩ (b + l)
            = ⟨init ++ [.range b l], len + 1, (init ++ [Seg.range b l]).length, desc⟩ := by
          simp [claimStep]; split <;> simp_all
        refine ⟨⟨init ++ [.range b (l + 1)], len + 1, (init ++ [Seg.range b l]).length, desc⟩, ?_,
          .inl ⟨by simp, ?_, ?_⟩, ?_, rfl⟩
        · simp only [push]; rw [hcs]; simp [extend?]
        · intro s hs; simp at hs; rcases hs with hs | rfl
          · exact hwi s hs
          · simp [WF]; omega
        · rcases hdir with h | ⟨h, b', l', hl', h2⟩ | ⟨h, b', l', hl', h2⟩
          · exact .inl h
          · simp at hl'; obtain ⟨rfl, rfl⟩ := hl'
            exact .inr (.inl ⟨h, b, l + 1, by simp, by omega⟩)
          · simp at hl'
        · simp only [flat_snoc, iter, List.append_assoc, range'_snoc]
      · have hn : extend? (.range b l) id = none := by simp [extend?, he]
        have hcs := claimStep_noop ⟨init ++ [.range b l], len + 1, (init ++ [Seg.range b l]).length, desc⟩
          id _ hlast hn
        have hp : push c ⟨init ++ [.range b l], len, (init ++ [Seg.range b l]).length, desc⟩ id =
            push.pushTail c ⟨init ++ [.range b l], len + 1, (init ++ [Seg.range b l]).length, desc⟩
              (.range b l) id := by
          simp only [push]; rw [hcs]; simp [hn]
          split
          · rename_i h; simp at h; omega
          · rfl
        rw [hp]; exact tail _ rfl
  | rdesc b l =>
    simp [WF] at hwl
    by_cases he : l ≤ b ∧ id = b - l
    · obtain ⟨hlb, rfl⟩ := he
      -- the claim (only for `len 1`, which leaves `desc = none` before it) and the extension
      have hres : push c ⟨init ++ [.rdesc b l], len, (init ++ [Seg.rdesc b l]).length, desc⟩ (b - l) =
          some ⟨init ++ [.rdesc b (l + 1)], len + 1, (init ++ [Seg.rdesc b l]).length,
            if l = 1 then some true else desc⟩ := by
        by_cases hl1 : l = 1
        · subst hl1
          have hdn : desc = none := hnone (fun _ _ h => by cases h) (fun b' l' h => by cases h; omega)
          subst hdn
          have e1 : b - 1 + 1 = b := by omega
          simp [push, claimStep, claim, extend?, e1, hlb]
        · simp [push, claimStep, extend?, hl1, hlb]
      refine ⟨_, hres, .inl ⟨by simp, ?_, ?_⟩, ?_, rfl⟩
      · intro s hs; simp at hs; rcases hs with hs | rfl
        · exact hwi s hs
        · simp [WF]; omega
      · by_cases hl1 : l = 1
        · subst hl1; exact .inr (.inr ⟨by simp, b, 2, by simp, by omega⟩)
        · simp only [hl1, ite_false]
          rcases hdir with h | ⟨h, b', l', hl', h2⟩ | ⟨h, b', l', hl', h2⟩
          · exact .inl h
          · simp at hl'
          · simp at hl'; obtain ⟨rfl, rfl⟩ := hl'
            exact .inr (.inr ⟨h, b, l + 1, by simp, by omega⟩)
      · simp only [flat_snoc, iter, List.append_assoc]
        rw [show b - (l + 1 - 1) = b - l by omega, show b - (l - 1) = b - l + 1 by omega]
        rw [List.range'_succ]; simp
    · have hn : extend? (.rdesc b l) id = none := by simp [extend?]; omega
      have hcs := claimStep_noop ⟨init ++ [.rdesc b l], len + 1, (init ++ [Seg.rdesc b l]).length, desc⟩
        id _ hlast hn
      have hp : push c ⟨init ++ [.rdesc b l], len, (init ++ [Seg.rdesc b l]).length, desc⟩ id =
          push.pushTail c ⟨init ++ [.rdesc b l], len + 1, (init ++ [Seg.rdesc b l]).length, desc⟩
            (.rdesc b l) id := by
        simp only [push]; rw [hcs]; simp [hn]
      rw [hp]; exact tail _ rfl
  | rep r k =>
    simp [WF] at hwl
    have hdn : desc = none := hnone (fun _ _ h => by cases h) (fun _ _ h => by cases h)
    subst hdn
    by_cases he : r = id
    · subst he
      refine ⟨⟨init ++ [.rep r (k + 1)], len + 1, (init ++ [Seg.rep r k]).length, none⟩,
        by simp [push, claimStep, extend?], .inl ⟨by simp, ?_, .inl rfl⟩, ?_, rfl⟩
      · intro s hs; simp at hs; rcases hs with hs | rfl
        · exact hwi s hs
        · simp [WF]; omega
      · simp only [flat_snoc, iter, List.append_assoc, List.replicate_succ']
    · have hp : push c ⟨init ++ [.rep r k], len, (init ++ [Seg.rep r k]).length, none⟩ id =
          push.pushTail c ⟨init ++ [.rep r k], len + 1, (init ++ [Seg.rep r k]).length, none⟩
            (.rep r k) id := by
        simp [push, claimStep, extend?, he]
      rw [hp]; exact tail _ rfl
  | asc bm =>
    obtain ⟨hne, hsorted, hlt⟩ := hwl
    have hdn : desc = none := hnone (fun _ _ h => by cases h) (fun _ _ h => by cases h)
    subst hdn
    by_cases he : bm.getLast?.getD 0 < id
    · have hall : ∀ y ∈ bm, y < id := fun y hy => Nat.lt_of_le_of_lt (le_getLast hsorted hy) he
      refine ⟨⟨init ++ [.asc (bm ++ [id])], len + 1, (init ++ [Seg.asc bm]).length, none⟩,
        by simp [push, claimStep, extend?, he, ins_snoc hall], .inl ⟨by simp, ?_, .inl rfl⟩, ?_, rfl⟩
      · intro s hs; simp at hs; rcases hs with hs | rfl
        · exact hwi s hs
        · refine ⟨by simp, ?_, ?_⟩
          · rw [← ins_snoc hall]; exact sorted_ins hsorted
          · intro x hx; simp at hx; rcases hx with hx | rfl
            · exact hlt x hx
            · exact hid
      · simp only [flat_snoc, iter, List.append_assoc]
    · have hp : push c ⟨init ++ [.asc bm], len, (init ++ [Seg.asc bm]).length, none⟩ id =
          push.pushTail c ⟨init ++ [.asc bm], len + 1, (init ++ [Seg.asc bm]).length, none⟩
            (.asc bm) id := by
        simp [push, claimStep, extend?, he]
      rw [hp]; exact tail _ rfl
  | dsc bm =>
    obtain ⟨hne, hsorted, hlt⟩ := hwl
    have hdn : desc = none := hnone (fun _ _ h => by cases h) (fun _ _ h => by cases h)
    subst hdn
    by_cases he : id < bm.head?.getD 0
    · have hall : ∀ y ∈ bm, id < y := fun y hy => Nat.lt_of_lt_of_le he (head_le hsorted hy)
      refine ⟨⟨init ++ [.dsc (id :: bm)], len + 1, (init ++ [Seg.dsc bm]).length, none⟩,
        by simp [push, claimStep, extend?, he, ins_cons hall], .inl ⟨by simp, ?_, .inl rfl⟩, ?_, rfl⟩
      · intro s hs; simp at hs; rcases hs with hs | rfl
        · exact hwi s hs
        · refine ⟨by simp, ?_, ?_⟩
          · rw [← ins_cons hall]; exact sorted_ins hsorted
          · intro x hx; simp at hx; rcases hx with rfl | hx
            · exact hid
            · exact hlt x hx
      · simp only [flat_snoc, iter, List.append_assoc]; simp
    · have hp : push c ⟨init ++ [.dsc bm], len, (init ++ [Seg.dsc bm]).length, none⟩ id =
          push.pushTail c ⟨init ++ [.dsc bm], len + 1, (init ++ [Seg.dsc bm]).length, none⟩
            (.dsc bm) id := by
        simp [push, claimStep, extend?, he]
      rw [hp]; exact tail _ rfl

/-- Any push sequence from a `DInv` or `Inv` state keeps every id, in order. -/
theorem pushAll_dinv : ∀ (cs : List Bool) (st : St) (xs : List Nat), (DInv st ∨ Inv st) →
    (∀ x ∈ xs, x < W) →
    ∃ st', pushAll cs st xs = some st' ∧ flat st'.segs = flat st.segs ++ xs ∧
      st'.len = st.len + xs.length
  | _, st, [], _, _ => ⟨st, rfl, by simp, rfl⟩
  | cs, st, x :: xs, hd, hx => by
    obtain ⟨st1, h1, hd1, hf1, hl1⟩ : ∃ st1, push (cs.headD false) st x = some st1 ∧
        (DInv st1 ∨ Inv st1) ∧ flat st1.segs = flat st.segs ++ [x] ∧ st1.len = st.len + 1 := by
      rcases hd with hd | hi
      · exact push_dinv _ st hd x (hx x (by simp))
      · obtain ⟨st1, h1, hi1, hf1, hl1⟩ := push_ok (cs.headD false) st hi x (hx x (by simp))
        exact ⟨st1, h1, .inr hi1, hf1, hl1⟩
    obtain ⟨st2, h2, hf2, hl2⟩ := pushAll_dinv cs.tail st1 xs hd1 (fun y hy => hx y (by simp [hy]))
    refine ⟨st2, ?_, ?_, ?_⟩
    · simp only [pushAll]; rw [h1]; exact h2
    · rw [hf2, hf1]; simp
    · rw [hl2, hl1]; simp; omega

/-- **#2920's correctness theorem.** Pushing any `xs` onto a decoded list (any
well-formed segments, whatever the last one is) yields exactly the decoded ids
followed by `xs`, with `len` counting all of them, for every sequence of collapse
decisions — no reorder, no `assert!`. (Before the fix: `pre2920_decoded_then_pushed_*`.) -/
theorem decoded_then_pushed_keeps_order (cs : List Bool) (segs : List Seg) (xs : List Nat)
    (hw : ∀ s ∈ segs, WF s) (hx : ∀ x ∈ xs, x < W) :
    ∃ st, pushAll cs (fromSegments segs) xs = some st ∧ flat st.segs = flat segs ++ xs ∧
      st.len = (flat segs).length + xs.length :=
  pushAll_dinv cs _ xs (.inl (fromSegments_dinv segs hw)) hx

end IdListPush

namespace IdListWire
open IdListPush

/-- **End to end**: whatever `read_ids` accepts off the wire, `from_segments` of it
can be extended by any push sequence with every id kept in order after the
`count` decoded ones (`readIds_safe` supplies the well-formedness). -/
theorem readIds_then_pushed_keeps_order (R : Roaring) (hR : RoaringOK R) (bs : List Nat)
    (count : Nat) (hb : ∀ b ∈ bs, b < 256) (hc : count < 4294967296) {segs : List Seg}
    {rest : List Nat} (h : readIds R bs count = .ok (segs, rest))
    (cs : List Bool) (xs : List Nat) (hx : ∀ x ∈ xs, x < W) :
    ∃ st, pushAll cs (fromSegments segs) xs = some st ∧ flat st.segs = flat segs ++ xs ∧
      st.len = count + xs.length := by
  obtain ⟨hw, hlen, _⟩ := readIds_safe R hR bs count hb hc h
  obtain ⟨st, h1, h2, h3⟩ := decoded_then_pushed_keeps_order cs segs xs hw hx
  exact ⟨st, h1, h2, by rw [h3, hlen]⟩

end IdListWire
