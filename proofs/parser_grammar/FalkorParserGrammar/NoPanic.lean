/-
# `parse_expr_inner` never panics

The four panic sites of `parse_expr_inner` are
`children().last().unwrap()` (cypher.rs:2300, 2305), `child(0)` of a
`Negate` (2526) and of a `Paren` (2598), and the two `unreachable!()`
(2614, 2617). They are all safe because of one stack invariant:

* every frame's level is at most 11;
* the stack is never empty inside the loop;
* a frame's tree has only well-formed children (`AlmostGood`), and the top
  frame's tree, when there is one, is itself well formed (`Good`): a
  `Negate`, `Paren` or comparison always has a child by the time it is
  popped with a tree, because it is only ever pushed *under* the frame that
  will produce that child.
-/
import FalkorParserGrammar.Model
import FalkorParserGrammar.Tactic

namespace FalkorParserGrammar

/-- Operators whose child the loop reads unconditionally. -/
def special : Op → Bool
  | .neg | .paren => true
  | o => isCmp o

/-- A tree in which every `Negate`, `Paren` and comparison has a child. -/
inductive Good : RT → Prop
  | mk (o : Op) (cs : List RT) :
      (special o = true → cs ≠ []) → (∀ c ∈ cs, Good c) → Good (.node o cs)

/-- The tree of a frame that is still waiting for its (next) child. -/
def AlmostGood (t : RT) : Prop := ∀ c ∈ t.kids, Good c

theorem Good.almost {t : RT} (h : Good t) : AlmostGood t := by
  cases h with | mk o cs _ hc => exact hc

theorem good_kid {o : Op} {cs : List RT} {c : RT} (h : Good (.node o cs)) (hc : c ∈ cs) : Good c := by
  cases h with | mk _ _ _ h2 => exact h2 c hc

theorem good_leaf {o : Op} (h : special o = false) : Good (lf o) :=
  .mk o [] (by simp [h]) (by simp)

theorem good_node {o : Op} {cs : List RT} (hne : cs ≠ []) (h : ∀ c ∈ cs, Good c) : Good (.node o cs) :=
  .mk o cs (fun _ => hne) h

theorem almost_leaf (o : Op) : AlmostGood (lf o) := by simp [AlmostGood, RT.kids]

theorem good_addChild {e t : RT} (he : AlmostGood e) (ht : Good t) : Good (e.addChild t) := by
  cases e with
  | node o cs =>
    simp only [RT.addChild]
    refine good_node (by simp) ?_
    intro c hc
    simp only [List.mem_append, List.mem_singleton] at hc
    rcases hc with hc | rfl
    · exact he c hc
    · exact ht

theorem good_last {t c : RT} (ht : Good t) (h : t.last? = some c) : Good c := by
  cases t with
  | node o cs =>
    simp only [RT.last?, RT.kids] at h
    exact good_kid ht (List.mem_of_getLast? h)

theorem last_some_of_good {t : RT} (ht : Good t) (hs : special t.root = true) : ∃ c, t.last? = some c := by
  cases ht with
  | mk o cs h1 _ =>
    have hne := h1 hs
    obtain ⟨c, hc⟩ := List.getLast?_isSome.mpr hne |> Option.isSome_iff_exists.mp
    exact ⟨c, hc⟩

/-! ## The invariant -/

def FrameOK : Frame → Prop
  | (l, o) => l ≤ 11 ∧ ∀ e, o = some e → AlmostGood e

def TopOK : List Frame → Prop
  | (_, some e) :: _ => Good e
  | _ => True

def Inv (K : List Frame) : Prop := K ≠ [] ∧ (∀ f ∈ K, FrameOK f) ∧ TopOK K

def StOK : St → Prop
  | .run K _ => Inv K
  | .done t _ => Good t
  | .err _ => True
  | .panic => False
  | .fuel => True

/-- What a sub-parser may return: never a panic, and only good trees. -/
def ResOK {α} (P : α → Prop) : Res α → Prop
  | .ok a _ => P a
  | .panic => False
  | _ => True

def PeOK (pe : List Tok → Out) : Prop := ∀ ts, ResOK Good (pe ts)

theorem frameOK_none {l : Nat} (h : l ≤ 11) : FrameOK (l, none) := ⟨h, by simp⟩

theorem frameOK_some {l : Nat} {e : RT} (h : l ≤ 11) (he : AlmostGood e) : FrameOK (l, some e) :=
  ⟨h, by intro e' h'; cases h'; exact he⟩

theorem ret_ok {t : RT} {K : List Frame} (ts : List Tok) (ht : Good t) (hK : ∀ f ∈ K, FrameOK f) :
    StOK (ret t K ts) := by
  match K, hK with
  | [], _ => exact ht
  | (l, some e) :: K', hK =>
    have hf := hK (l, some e) (by simp)
    obtain ⟨hl, he⟩ := hf
    have hg := good_addChild (he e rfl) ht
    refine ⟨by simp, ?_, hg⟩
    intro f hf'
    simp only [List.mem_cons] at hf'
    rcases hf' with rfl | hf'
    · exact frameOK_some hl hg.almost
    · exact hK f (by simp [hf'])
  | (l, none) :: K', hK =>
    have hl := (hK (l, none) (by simp)).1
    refine ⟨by simp, ?_, ht⟩
    intro f hf'
    simp only [List.mem_cons] at hf'
    rcases hf' with rfl | hf'
    · exact frameOK_some hl ht.almost
    · exact hK f (by simp [hf'])

/-- Pushing frames on a good stack. -/
theorem push_ok {K : List Frame} {F : List Frame} (ts : List Tok) (hK : ∀ f ∈ K, FrameOK f)
    (hF : ∀ f ∈ F, FrameOK f) (hne : F ≠ []) (htop : TopOK (F ++ K)) : StOK (.run (F ++ K) ts) := by
  refine ⟨by simp [hne], ?_, htop⟩
  intro f hf
  simp only [List.mem_append] at hf
  rcases hf with hf | hf
  · exact hF f hf
  · exact hK f hf

theorem tail_ok {f : Frame} {K : List Frame} (h : ∀ g ∈ f :: K, FrameOK g) : ∀ g ∈ K, FrameOK g :=
  fun g hg => h g (by simp [hg])

/-! ## Sub-parsers return good trees -/

theorem good2 {o : Op} {a b : RT} (ha : Good a) (hb : Good b) : Good (.node o [a, b]) :=
  good_node (by simp) (by intro c hc; simp at hc; rcases hc with rfl | rfl <;> assumption)

theorem good1 {o : Op} {a : RT} (ha : Good a) : Good (.node o [a]) :=
  good_node (by simp) (by intro c hc; simp at hc; subst hc; assumption)

theorem isLoop_ok (res : RT) (ts : List Tok) : Good res → ResOK Good (isLoop res ts) := by
  fun_induction isLoop res ts <;> intro h
  all_goals first
    | (simp_all [ResOK]; done)
    | (rename_i ih; exact ih (good2 (good_leaf rfl) h))


set_option linter.unusedSimpArgs false

theorem good_iff {o : Op} {cs : List RT} :
    Good (.node o cs) ↔ (special o = true → cs ≠ []) ∧ ∀ c ∈ cs, Good c :=
  ⟨fun h => by cases h with | mk _ _ h1 h2 => exact ⟨h1, h2⟩, fun ⟨h1, h2⟩ => .mk o cs h1 h2⟩

theorem parseIdent_np (ts : List Tok) : ResOK (fun _ => True) (parseIdent ts) := by
  unfold parseIdent; split
  · split <;> simp [ResOK]
  · simp [ResOK]

theorem dotted_np (acc : List Nat) (ts : List Tok) : ResOK (fun _ => True) (dotted acc ts) := by
  fun_induction dotted acc ts <;> simp_all [ResOK]

theorem labels_np (acc : List Nat) (ts : List Tok) : ResOK (fun _ => True) (labels acc ts) := by
  fun_induction labels acc ts <;> simp_all [ResOK]

/-- What a primary returns: a good tree, or (when the caller must continue
parsing into it) a `Paren`/`List` with good children. -/
def PrimOK (p : RT × Bool) : Prop := AlmostGood p.1 ∧ (p.2 = false → Good p.1)

/-- The closing simp set: goodness of concrete trees, list membership. -/
macro "gsimp" : tactic => `(tactic| simp_all [FalkorParserGrammar.ResOK, FalkorParserGrammar.good_iff,
  FalkorParserGrammar.special, FalkorParserGrammar.isCmp, or_imp, forall_eq, forall_and,
  FalkorParserGrammar.lf, FalkorParserGrammar.RT.kids, FalkorParserGrammar.AlmostGood,
  FalkorParserGrammar.PrimOK, FalkorParserGrammar.StOK])

/-- Split every `match`/`if`, add every sub-parser fact, close by simp. -/
macro "crush" hpe:ident pe:ident : tactic => `(tactic| (
  repeat' split
  all_goals (try pe_facts $hpe $pe)
  all_goals (try use_facts FalkorParserGrammar.parseIdent_np)
  all_goals (try use_facts FalkorParserGrammar.dotted_np)
  all_goals (try use_facts FalkorParserGrammar.labels_np)
  all_goals gsimp))

theorem PeOK_at {pe : List Tok → Out} (h : PeOK pe) (ts : List Tok) : ResOK Good (pe ts) := h ts

theorem listCompTail_ok {v : Nat} {l : RT} {r : Res RT} (hl : Good l) (hr : ResOK Good r) :
    ResOK PrimOK (listCompTail v l r) := by
  unfold listCompTail
  split <;> gsimp

section sub
variable {pe : List Tok → Out} (hpe : PeOK pe)
include hpe

theorem exprItems_ok (n : Nat) (acc : List RT) (ts : List Tok) :
    (∀ c ∈ acc, Good c) → ResOK (fun l => ∀ c ∈ l, Good c) (exprItems pe n acc ts) := by
  fun_induction exprItems pe n acc ts <;> intro hacc <;> crush hpe pe

theorem exprList_ok (n : Nat) (acc : List RT) (ts : List Tok) :
    (∀ c ∈ acc, Good c) → ResOK (fun l => ∀ c ∈ l, Good c) (exprList pe n acc ts) := by
  intro hacc
  unfold exprList
  split
  · exact hacc
  · exact exprItems_ok hpe n acc ts hacc

theorem exprList_nil_ok (n : Nat) (ts : List Tok) :
    ResOK (fun l => ∀ c ∈ l, Good c) (exprList pe n [] ts) :=
  exprList_ok hpe n [] ts (by simp)

theorem mapBody_ok (n : Nat) (acc : List RT) (ts : List Tok) :
    (∀ c ∈ acc, Good c) → ResOK Good (mapBody pe n acc ts) := by
  fun_induction mapBody pe n acc ts <;> intro hacc <;> crush hpe pe

theorem parseMap_ok (n : Nat) (ts : List Tok) : ResOK Good (parseMap pe n ts) := by
  unfold parseMap; split
  · gsimp
  · exact mapBody_ok hpe n [] _ (by simp)
  · gsimp

theorem listComp_ok (v : Nat) (ts : List Tok) : ResOK PrimOK (listComp pe v ts) := by
  unfold listComp
  split
  all_goals (try pe_facts hpe pe)
  all_goals (try use_goal FalkorParserGrammar.listCompTail_ok)
  all_goals (try exact hpe _)
  all_goals gsimp

theorem listLit_ok (ts : List Tok) : ResOK PrimOK (listLit pe ts) := by
  unfold listLit listDefault
  repeat' split
  all_goals (try use_goal FalkorParserGrammar.listComp_ok hpe)
  all_goals gsimp

theorem primary_ok (n : Nat) (ts : List Tok) : ResOK PrimOK (primary pe n ts) := by
  unfold primary
  repeat' split
  all_goals (try use_goal FalkorParserGrammar.listLit_ok hpe)
  all_goals (try use_facts FalkorParserGrammar.parseMap_ok hpe)
  all_goals (try use_facts FalkorParserGrammar.dotted_np)
  all_goals (try use_facts FalkorParserGrammar.exprList_nil_ok hpe)
  all_goals (try have := Good.almost ‹Good _›)
  all_goals gsimp

theorem listOp_ok (lhs : RT) (hl : Good lhs) (ts : List Tok) : ResOK Good (listOp pe lhs ts) := by
  unfold listOp listOpFrom
  repeat' split
  all_goals (try pe_facts hpe pe)
  all_goals (try (unfold listOpTo; repeat' split))
  all_goals (try (unfold orDefault))
  all_goals gsimp

theorem mapProjItem_ok (ts : List Tok) : ResOK Good (mapProjItem pe ts) := by
  unfold mapProjItem
  repeat' split
  all_goals (try pe_facts hpe pe)
  all_goals (try use_facts FalkorParserGrammar.parseIdent_np)
  all_goals gsimp

theorem mapProjBody_ok (n : Nat) (items : List RT) (ts : List Tok) :
    (∀ c ∈ items, Good c) → ResOK Good (mapProjBody pe n items ts) := by
  fun_induction mapProjBody pe n items ts <;> intro hi
  all_goals (try use_facts FalkorParserGrammar.mapProjItem_ok hpe)
  all_goals gsimp

theorem mapProj_ok (n : Nat) (base : RT) (hb : Good base) (ts : List Tok) :
    ResOK Good (mapProj pe n base ts) := by
  unfold mapProj; split
  · gsimp
  · exact mapProjBody_ok hpe n [base] _ (by simpa using hb)

theorem postLoop_ok (n : Nat) (res : RT) (ts : List Tok) :
    Good res → ResOK Good (postLoop pe n res ts) := by
  fun_induction postLoop pe n res ts <;> intro hr
  all_goals (try use_facts FalkorParserGrammar.listOp_ok hpe)
  all_goals (try use_facts FalkorParserGrammar.mapProj_ok hpe)
  all_goals (try use_facts FalkorParserGrammar.parseIdent_np)
  all_goals gsimp

theorem postStep_ok (n : Nat) (res : RT) (hr : Good res) (K : List Frame) (hK : ∀ f ∈ K, FrameOK f)
    (ts : List Tok) : StOK (postStep pe n res K ts) := by
  unfold postStep
  repeat' split
  all_goals (try use_facts FalkorParserGrammar.postLoop_ok hpe)
  all_goals (try use_facts FalkorParserGrammar.labels_np)
  all_goals (try exact ret_ok _ (by gsimp) hK)
  all_goals gsimp


end sub

/-- Unfold the stack invariant. -/
macro "isimp" : tactic => `(tactic| simp_all [FalkorParserGrammar.StOK, FalkorParserGrammar.Inv,
  FalkorParserGrammar.FrameOK, FalkorParserGrammar.TopOK, List.mem_cons, forall_eq_or_imp,
  FalkorParserGrammar.ResOK, FalkorParserGrammar.good_iff, FalkorParserGrammar.special,
  FalkorParserGrammar.isCmp, or_imp, forall_eq, forall_and, FalkorParserGrammar.lf,
  FalkorParserGrammar.RT.kids, FalkorParserGrammar.AlmostGood, Prod.forall])

theorem binStep_ok {l : Nat} (hl : l ≤ 8) {res : RT} (hr : Good res) {K : List Frame}
    (hK : ∀ f ∈ K, FrameOK f) (ts : List Tok) : StOK (binStep l res K ts) := by
  unfold binStep
  split
  · rename_i op _
    have ha : AlmostGood (if res.root = op then res else .node op [res]) := by
      split
      · exact hr.almost
      · intro c hc; simp [RT.kids] at hc; subst hc; exact hr
    refine ⟨by simp, ?_, trivial⟩
    intro f hf
    simp only [List.mem_cons] at hf
    rcases hf with rfl | rfl | hf
    · exact frameOK_none (by omega)
    · exact frameOK_some (by omega) ha
    · exact hK f hf
  · exact ret_ok ts hr hK

theorem push5_ok {res : RT} (hr : Good res) {K : List Frame} (hK : ∀ f ∈ K, FrameOK f)
    (op : Op) (ts : List Tok) : StOK (push5 (.node op [res]) K ts) := by
  refine ⟨by simp [push5], ?_, trivial⟩
  intro f hf
  simp only [push5, List.mem_cons] at hf
  rcases hf with rfl | rfl | hf
  · exact frameOK_none (by omega)
  · exact frameOK_some (by omega) (by intro c hc; simp [RT.kids] at hc; subst hc; exact hr)
  · exact hK f hf

theorem predStep_ok {res : RT} (hr : Good res) {K : List Frame} (hK : ∀ f ∈ K, FrameOK f)
    (ts : List Tok) : StOK (predStep res K ts) := by
  have hK' : ∀ f ∈ ((5, some (.node .not_ [])) :: K : List Frame), FrameOK f := by
    intro f hf; simp only [List.mem_cons] at hf
    rcases hf with rfl | hf
    · exact frameOK_some (by omega) (by simp [AlmostGood, RT.kids])
    · exact hK f hf
  have hrun : ∀ (r : RT) (ts' : List Tok), Good r → StOK (.run ((5, some r) :: K) ts') := by
    intro r ts' hg
    refine ⟨by simp, ?_, hg⟩
    intro f hf; simp only [List.mem_cons] at hf
    rcases hf with rfl | hf
    · exact frameOK_some (by omega) hg.almost
    · exact hK f hf
  unfold predStep
  repeat' split
  all_goals (try use_facts FalkorParserGrammar.isLoop_ok)
  all_goals first
    | exact push5_ok hr hK _ _
    | exact push5_ok hr hK' _ _
    | exact ret_ok _ ‹_› hK
    | exact hrun _ _ ‹_›
    | isimp


theorem frames_ok {F K : List Frame} (hF : ∀ f ∈ F, FrameOK f) (hK : ∀ f ∈ K, FrameOK f) :
    ∀ f ∈ F ++ K, FrameOK f := by
  intro f hf; simp only [List.mem_append] at hf; rcases hf with hf | hf
  · exact hF f hf
  · exact hK f hf

theorem cmpStep_ok {res : RT} (hr : Good res) {K : List Frame} (hK : ∀ f ∈ K, FrameOK f)
    (ts : List Tok) : StOK (cmpStep res K ts) := by
  unfold cmpStep
  split
  · exact ret_ok ts hr hK
  · rename_i op _
    simp only
    split
    · rename_i hord
      unfold lastCmpIsOrdering at hord
      -- last_cmp_node exists and has a last child
      have hlcn : ∃ lcn, (if res.root = .and_ then res.last? else some res) = some lcn ∧
          Good lcn ∧ special lcn.root = true := by
        by_cases hand : res.root = .and_
        · simp only [hand, beq_self_eq_true, Bool.true_and, Bool.or_eq_true] at hord
          rcases hord with h | h
          · simp [hand, isCmp] at h
          · revert h; cases hl : res.last? with
            | none => simp
            | some c =>
              intro h
              refine ⟨c, by simp [hand, hl], good_last hr hl, ?_⟩
              simp only [special]; split <;> simp_all
        · have hc : isCmp res.root = true := by
            simp only [Bool.or_eq_true, beq_iff_eq] at hord
            rcases hord with h | h
            · exact h
            · simp only [Bool.and_eq_true, beq_iff_eq] at h; exact absurd h.1 hand
          refine ⟨res, by simp [hand], hr, ?_⟩
          simp only [special]; split <;> simp_all
      obtain ⟨lcn, hl1, hl2, hl3⟩ := hlcn
      rw [hl1]
      obtain ⟨mid, hmid⟩ := last_some_of_good hl2 hl3
      simp only [hmid]
      have hgm := good_last hl2 hmid
      refine ⟨by simp, ?_, trivial⟩
      apply frames_ok (F := [(5, none), (4, _), (4, _)]) _ hK
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl
      · exact frameOK_none (by omega)
      · exact frameOK_some (by omega) (by intro c hc; simp [RT.kids] at hc; subst hc; exact hgm)
      · refine frameOK_some (by omega) ?_
        split
        · exact hr.almost
        · intro c hc; simp [RT.kids] at hc; subst hc; exact hr
    · refine ⟨by simp, ?_, trivial⟩
      apply frames_ok (F := [(5, none), (4, _)]) _ hK
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl
      · exact frameOK_none (by omega)
      · exact frameOK_some (by omega) (by intro c hc; simp [RT.kids] at hc; subst hc; exact hr)

theorem closeStep_ok {res : RT} (hr : Good res) {K : List Frame} (hK : ∀ f ∈ K, FrameOK f)
    (ts : List Tok) : StOK (closeStep res K ts) := by
  unfold closeStep
  split
  · rename_i cs
    split
    · rename_i r
      split
      · exact absurd rfl ((good_iff.mp hr).1 rfl)
      · rename_i gcs rest
        have hg := good_iff.mp hr
        have hp := good_iff.mp (hg.2 (.node .paren gcs) (by simp))
        refine ret_ok r (good_iff.mpr ⟨fun _ => ?_, ?_⟩) hK
        · have := hp.1 rfl
          simp [this]
        · intro c hc
          simp only [List.mem_append] at hc
          rcases hc with hc | hc
          · exact hp.2 c hc
          · exact hg.2 c (by simp [hc])
      · exact ret_ok _ hr hK
    · trivial
  · split
    · refine ⟨by simp, ?_, trivial⟩
      apply frames_ok (F := [(0, none), (11, _)]) _ hK
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl
      · exact frameOK_none (by omega)
      · exact frameOK_some (by omega) hr.almost
    · exact ret_ok _ hr hK
    · trivial
  · exact ret_ok _ hr hK

theorem notFrames_ok {K : List Frame} (hK : ∀ f ∈ K, FrameOK f) (n : Nat) :
    ∀ f ∈ notFrames n K, FrameOK f := by
  intro f hf
  unfold notFrames at hf
  split at hf
  · simp only [List.mem_cons] at hf
    rcases hf with rfl | hf
    · exact frameOK_none (by omega)
    · exact hK f hf
  · split at hf
    · simp only [List.mem_cons] at hf
      rcases hf with rfl | hf
      · exact frameOK_some (by omega) (almost_leaf _)
      · exact hK f hf
    · simp only [List.mem_cons] at hf
      rcases hf with rfl | rfl | hf
      · exact frameOK_some (by omega) (almost_leaf _)
      · exact frameOK_some (by omega) (almost_leaf _)
      · exact hK f hf

theorem step_ok {pe : List Tok → Out} (hpe : PeOK pe) (n : Nat) {K : List Frame} (hK : Inv K)
    (ts : List Tok) : StOK (step pe n K ts) := by
  obtain ⟨hne, hK, htop⟩ := hK
  match K, hne, hK, htop with
  | (l, none) :: K', _, hK, _ =>
    have hl := (hK (l, none) (by simp)).1
    have hK' := tail_ok hK
    simp only [step]
    split
    · refine ⟨by simp, ?_, trivial⟩
      apply frames_ok (F := [(l + 1, none), (l, none)]) _ hK'
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl
      · exact frameOK_none (by omega)
      · exact frameOK_none hl
    · split
      · refine ⟨by simp, ?_, trivial⟩
        intro f hf
        simp only [List.mem_cons] at hf
        rcases hf with rfl | hf
        · exact frameOK_none (by omega)
        · exact notFrames_ok hK' _ f hf
      · split
        · refine ⟨by simp, ?_, trivial⟩
          apply frames_ok (F := [(10, none), (9, _)]) _ hK'
          intro f hf
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
          rcases hf with rfl | rfl
          · exact frameOK_none (by omega)
          · refine ⟨by omega, ?_⟩
            intro e he; split at he <;> simp at he; subst he; exact almost_leaf _
        · have hp := primary_ok hpe n ts
          revert hp
          cases primary pe n ts with
          | ok p ts' =>
            obtain ⟨res, b⟩ := p
            intro hp
            cases b with
            | true =>
              refine ⟨by simp, ?_, trivial⟩
              apply frames_ok (F := [(0, none), (l, _)]) _ hK'
              intro f hf
              simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
              rcases hf with rfl | rfl
              · exact frameOK_none (by omega)
              · exact frameOK_some hl hp.1
            | false => exact ret_ok ts' (hp.2 rfl) hK'
          | err e => intro; trivial
          | panic => intro h; exact h
          | fuel => intro; trivial
  | (l, some res) :: K', _, hK, htop =>
    have hl := (hK (l, some res) (by simp)).1
    have hK' := tail_ok hK
    have hr : Good res := htop
    simp only [step]
    match l, hl with
    | 0, _ => exact binStep_ok (by omega) hr hK' ts
    | 1, _ => exact binStep_ok (by omega) hr hK' ts
    | 2, _ => exact binStep_ok (by omega) hr hK' ts
    | 3, _ => exact ret_ok ts hr hK'
    | 4, _ => exact cmpStep_ok hr hK' ts
    | 5, _ => exact predStep_ok hr hK' ts
    | 6, _ => exact binStep_ok (by omega) hr hK' ts
    | 7, _ => exact binStep_ok (by omega) hr hK' ts
    | 8, _ => exact binStep_ok (by omega) hr hK' ts
    | 9, _ =>
      simp only
      split
      · exact absurd rfl ((good_iff.mp hr).1 rfl)
      · exact ret_ok ts hr hK'
    | 10, _ => exact postStep_ok hpe n res hr K' hK' ts
    | 11, _ => exact closeStep_ok hr hK' ts

theorem loop_ok {pe : List Tok → Out} (hpe : PeOK pe) :
    ∀ (n : Nat) (s : St), StOK s → ResOK Good (loop pe n s)
  | n, .done t ts, h => by cases n <;> exact h
  | n, .err _, _ => by cases n <;> trivial
  | n, .panic, h => by cases n <;> exact h
  | n, .fuel, _ => by cases n <;> trivial
  | 0, .run _ _, _ => trivial
  | n + 1, .run K ts, h => loop_ok hpe n _ (step_ok hpe n h ts)

/-- **`parse_expr` never panics**: on every token list, at every budget, the
modelled `parse_expr_inner` returns a tree, an error, or runs out of budget —
never one of the `unwrap()`/`child(0)`/`unreachable!()` panics — and every
tree it returns is well formed. -/
theorem parseExpr_no_panic : ∀ (n : Nat), PeOK (parseExpr n)
  | 0 => fun _ => trivial
  | n + 1 => fun ts => by
    simp only [parseExpr]
    exact loop_ok (parseExpr_no_panic n) n _ ⟨by simp, by simp [FrameOK], trivial⟩

theorem parseExpr_ne_panic (n : Nat) (ts : List Tok) : parseExpr n ts ≠ .panic := by
  intro h; have := parseExpr_no_panic n ts; rw [h] at this; exact this

end FalkorParserGrammar
