import FalkorCowBTree.BatchWF
import FalkorCowBTree.Assert

/-!
# `insert_batch` and `from_sorted` build well-formed trees

`CowBTree::insert_batch` (`mod.rs:241`) is `build_root(root.apply_batch(batch))`, and
`CowBTree::from_sorted` (`mod.rs:190`) is `build_root` over the leaf chunks. Their contents are proven
in `proofs/index_layer` (`Leaf.insertBatch_contents`, `Leaf.fromSorted_contents`); this file proves
that the trees they build satisfy `TreeWF`, so every tree the Rust can build — `from_sorted`, then any
sequence of `insert`, `remove`, `insert_batch`, `remove_batch` — is well-formed (`Reachable_wf`), and
the content specs of `insert_spec`, `remove_spec`, `range_spec`, … apply to it.

* `mergeWalk_spec` — the leaf merge (`merge_walk`) of a strictly sorted page and a sorted batch is
  strictly sorted, holds exactly the union, and is no longer than both together.
* `chunks_spec'` — `<[T]>::chunks(n)`: non-empty pieces of at most `n`, concatenating to the input.
* `MinChain` — a list of well-formed sibling fragments whose separators are their minima (what
  `pack_branches` computes, `node.rs:80`). `reseparate`: any separator chain can be re-separated by
  minima. `packBranches_chain`: `pack_branches` turns a `MinChain` of `>= 2` nodes into a shorter
  `MinChain` one level up. `buildRoot_wf`: `build_root` ends in one well-formed node.
* `applyBatch_chain` — `Node::apply_batch` (`node.rs:259`) returns a `MinChain` of fragments holding the
  sorted union of the subtree and its batch slice.
* `insertBatch_wf`, `fromSorted_wf` (headlines), `Reachable_wf`.
-/

namespace CowBTree

/-! ## The leaf merge and `chunks` -/

/-- Every input is at or above the previous output `last`. -/
def AboveLast (last : Option E) (l : List E) : Prop := ∀ x, last = some x → ∀ y ∈ l, x ≤ y

theorem mem_append_cons_left {y a : E} {as bs : List E} (h : y ∈ as ++ bs) : y ∈ (a :: as) ++ bs := by
  rcases List.mem_append.1 h with h | h
  · exact List.mem_append.2 (Or.inl (List.mem_cons_of_mem _ h))
  · exact List.mem_append.2 (Or.inr h)

theorem mem_append_cons_right {y b : E} {as bs : List E} (h : y ∈ as ++ bs) : y ∈ as ++ (b :: bs) := by
  rcases List.mem_append.1 h with h | h
  · exact List.mem_append.2 (Or.inl h)
  · exact List.mem_append.2 (Or.inr (List.mem_cons_of_mem _ h))

theorem mergeWalk_spec : ∀ (last : Option E) (as bs : List E), Sorted as → SortedLe bs → AboveLast last (as ++ bs) →
    Sorted (mergeWalk last as bs) ∧ (∀ x, last = some x → ∀ e ∈ mergeWalk last as bs, x < e) ∧
    (∀ e, e ∈ mergeWalk last as bs ↔ (e ∈ as ∨ e ∈ bs) ∧ last ≠ some e) ∧
    (mergeWalk last as bs).length ≤ as.length + bs.length := by
  intro last as bs
  induction last, as, bs using mergeWalk.induct with
  | case1 x =>
    intro _ _ _
    rw [mergeWalk]
    simp [sorted_nil]
  | case2 a as ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_left rfl]
    have hs' := (sorted_cons.1 hs).2
    obtain ⟨r1, r2, r3, r4⟩ := ih hs' hb (fun x hx y hy => hab x hx y (by simp at hy ⊢; exact Or.inr hy))
    refine ⟨r1, r2, fun e => ?_, by simp at r4 ⊢; omega⟩
    rw [r3 e]; simp only [List.mem_cons, List.not_mem_nil, or_false, ne_eq, Option.some.injEq]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨Or.inr h1, h2⟩
    · rintro ⟨h1 | h1, h2⟩
      · exact absurd h1.symm h2
      · exact ⟨h1, h2⟩
  | case3 last a as hl ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_right hl]
    obtain ⟨hs1, hs'⟩ := sorted_cons.1 hs
    obtain ⟨r1, r2, r3, r4⟩ := ih hs' hb (fun x hx y hy => by
      cases hx; simp at hy; exact Nat.le_of_lt (hs1 y hy))
    refine ⟨sorted_cons.2 ⟨fun e he => r2 a rfl e he, r1⟩, fun x hx e he => ?_, fun e => ?_, by simp at r4 ⊢; omega⟩
    · have hxa := hab x hx a (by simp)
      have hxa' : x ≠ a := fun h => hl (by rw [hx, h])
      rcases List.mem_cons.1 he with rfl | he
      · omega
      · have := r2 a rfl e he; omega
    · rw [List.mem_cons, r3 e]
      simp only [List.mem_cons, List.not_mem_nil, or_false, ne_eq, Option.some.injEq]
      by_cases hea : e = a
      · subst hea; simp only [true_or, true_and]; exact ⟨fun _ => hl, fun _ => by simp⟩
      · constructor
        · rintro (h | ⟨h1, _⟩)
          · exact absurd h hea
          · refine ⟨Or.inr h1, fun h2 => ?_⟩
            have := hs1 e h1; have := hab e h2 a (by simp); omega
        · rintro ⟨h1 | h1, h2⟩
          · exact absurd h1 hea
          · exact Or.inr ⟨h1, fun h => hea h.symm⟩
  | case4 b bs ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_left rfl]
    have hb' := (List.pairwise_cons.1 hb).2
    obtain ⟨r1, r2, r3, r4⟩ := ih hs hb' (fun x hx y hy => hab x hx y (by simp at hy ⊢; exact Or.inr hy))
    refine ⟨r1, r2, fun e => ?_, by simp at r4 ⊢; omega⟩
    rw [r3 e]; simp only [List.mem_cons, List.not_mem_nil, false_or, ne_eq, Option.some.injEq]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨Or.inr h1, h2⟩
    · rintro ⟨h1 | h1, h2⟩
      · exact absurd h1.symm h2
      · exact ⟨h1, h2⟩
  | case5 last b bs hl ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_right hl]
    obtain ⟨hb1, hb'⟩ := List.pairwise_cons.1 hb
    obtain ⟨r1, r2, r3, r4⟩ := ih hs hb' (fun x hx y hy => by
      cases hx; simp at hy; exact hb1 y hy)
    refine ⟨sorted_cons.2 ⟨fun e he => r2 b rfl e he, r1⟩, fun x hx e he => ?_, fun e => ?_, by simp at r4 ⊢; omega⟩
    · have hxb := hab x hx b (by simp)
      have hxb' : x ≠ b := fun h => hl (by rw [hx, h])
      rcases List.mem_cons.1 he with rfl | he
      · omega
      · have := r2 b rfl e he; omega
    · rw [List.mem_cons, r3 e]
      simp only [List.mem_cons, List.not_mem_nil, false_or, ne_eq, Option.some.injEq]
      by_cases heb : e = b
      · subst heb; simp only [true_or, true_and]; exact ⟨fun _ => hl, fun _ => by simp⟩
      · constructor
        · rintro (h | ⟨h1, h2⟩)
          · exact absurd h heb
          · exact ⟨Or.inr h1, fun h3 => by have := hb1 e h1; have := hab e h3 b (by simp); omega⟩
        · rintro ⟨h1 | h1, h2⟩
          · exact absurd h1 heb
          · exact Or.inr ⟨h1, fun h => heb h.symm⟩
  | case6 a as b bs hab' ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_left hab', ite_eq_left rfl]
    have hs' := (sorted_cons.1 hs).2
    obtain ⟨r1, r2, r3, r4⟩ := ih hs' hb (fun x hx y hy => hab x hx y (mem_append_cons_left hy))
    refine ⟨r1, r2, fun e => ?_, by simp at r4 ⊢; omega⟩
    rw [r3 e]; simp only [List.mem_cons, ne_eq, Option.some.injEq]
    constructor
    · rintro ⟨h1, h2⟩
      rcases h1 with h1 | h1
      · exact ⟨Or.inl (Or.inr h1), h2⟩
      · exact ⟨Or.inr h1, h2⟩
    · rintro ⟨(h1 | h1) | h1, h2⟩
      · exact absurd h1.symm h2
      · exact ⟨Or.inl h1, h2⟩
      · exact ⟨Or.inr h1, h2⟩
  | case7 last a as b bs hab' hl ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_left hab', ite_eq_right hl]
    obtain ⟨hs1, hs'⟩ := sorted_cons.1 hs
    have hb1 := List.pairwise_cons.1 hb
    obtain ⟨r1, r2, r3, r4⟩ := ih hs' hb (fun x hx y hy => by
      cases hx; simp only [List.mem_append, List.mem_cons] at hy
      rcases hy with hy | hy | hy
      · exact Nat.le_of_lt (hs1 y hy)
      · omega
      · have := hb1.1 y hy; omega)
    refine ⟨sorted_cons.2 ⟨fun e he => r2 a rfl e he, r1⟩, fun x hx e he => ?_, fun e => ?_,
      by simp at r4 ⊢; omega⟩
    · have hxa := hab x hx a (by simp)
      have hxa' : x ≠ a := fun h => hl (by rw [hx, h])
      rcases List.mem_cons.1 he with rfl | he
      · omega
      · have := r2 a rfl e he; omega
    · rw [List.mem_cons, r3 e]
      simp only [List.mem_cons, ne_eq, Option.some.injEq]
      by_cases hea : e = a
      · subst hea; simp only [true_or, true_and]; exact ⟨fun _ => hl, fun _ => by simp⟩
      · constructor
        · rintro (h | ⟨h1, h2⟩)
          · exact absurd h hea
          · refine ⟨?_, fun h3 => ?_⟩
            · rcases h1 with h1 | h1
              · exact Or.inl (Or.inr h1)
              · exact Or.inr h1
            · have hea' := hab e h3 a (by simp)
              rcases h1 with h1 | h1
              · have := hs1 e h1; omega
              · have : b ≤ e := by
                  rcases h1 with h | h
                  · omega
                  · exact hb1.1 e h
                omega
        · rintro ⟨(h1 | h1) | h1, h2⟩
          · exact absurd h1 hea
          · exact Or.inr ⟨Or.inl h1, fun h => hea h.symm⟩
          · exact Or.inr ⟨Or.inr h1, fun h => hea h.symm⟩
  | case8 a as b bs hab' ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_right hab', ite_eq_left rfl]
    have hb' := (List.pairwise_cons.1 hb).2
    obtain ⟨r1, r2, r3, r4⟩ := ih hs hb' (fun x hx y hy => hab x hx y (mem_append_cons_right hy))
    refine ⟨r1, r2, fun e => ?_, by simp at r4 ⊢; omega⟩
    rw [r3 e]; simp only [List.mem_cons, ne_eq, Option.some.injEq]
    constructor
    · rintro ⟨h1, h2⟩
      rcases h1 with h1 | h1
      · exact ⟨Or.inl h1, h2⟩
      · exact ⟨Or.inr (Or.inr h1), h2⟩
    · rintro ⟨h1 | (h1 | h1), h2⟩
      · exact ⟨Or.inl h1, h2⟩
      · exact absurd h1.symm h2
      · exact ⟨Or.inr h1, h2⟩
  | case9 last a as b bs hab' hl ih =>
    intro hs hb hab
    rw [mergeWalk, ite_eq_right hab', ite_eq_right hl]
    have hs1 := sorted_cons.1 hs
    obtain ⟨hb1, hb'⟩ := List.pairwise_cons.1 hb
    obtain ⟨r1, r2, r3, r4⟩ := ih hs hb' (fun x hx y hy => by
      cases hx; simp only [List.mem_append, List.mem_cons] at hy
      rcases hy with (hy | hy) | hy
      · omega
      · have := hs1.1 y hy; omega
      · exact hb1 y hy)
    refine ⟨sorted_cons.2 ⟨fun e he => r2 b rfl e he, r1⟩, fun x hx e he => ?_, fun e => ?_,
      by simp at r4 ⊢; omega⟩
    · have hxb := hab x hx b (by simp)
      have hxb' : x ≠ b := fun h => hl (by rw [hx, h])
      rcases List.mem_cons.1 he with rfl | he
      · omega
      · have := r2 b rfl e he; omega
    · rw [List.mem_cons, r3 e]
      simp only [List.mem_cons, ne_eq, Option.some.injEq]
      by_cases heb : e = b
      · subst heb; simp only [true_or, or_true, true_and]; exact ⟨fun _ => hl, fun _ => by simp⟩
      · constructor
        · rintro (h | ⟨h1, h2⟩)
          · exact absurd h heb
          · refine ⟨?_, fun h3 => ?_⟩
            · rcases h1 with h1 | h1
              · exact Or.inl h1
              · exact Or.inr (Or.inr h1)
            · have heb' := hab e h3 b (by simp)
              rcases h1 with h1 | h1
              · have : a ≤ e := by
                  rcases h1 with h | h
                  · omega
                  · exact Nat.le_of_lt (hs1.1 e h)
                omega
              · have := hb1 e h1; omega
        · rintro ⟨h1 | (h1 | h1), h2⟩
          · exact Or.inr ⟨Or.inl h1, fun h => heb h.symm⟩
          · exact absurd h1 heb
          · exact Or.inr ⟨Or.inr h1, fun h => heb h.symm⟩

theorem chunks_spec' (n : Nat) (hn : 0 < n) : ∀ (l : List E),
    (chunks n hn l).flatten = l ∧ ∀ p ∈ chunks n hn l, p ≠ [] ∧ p.length ≤ n := by
  intro l
  induction l using chunks.induct n hn with
  | case1 => rw [chunks]; simp
  | case2 x xs ih =>
    rw [chunks]
    obtain ⟨i1, i2⟩ := ih
    refine ⟨by rw [List.flatten_cons, i1, List.take_append_drop], fun p hp => ?_⟩
    rcases List.mem_cons.1 hp with rfl | hp
    · refine ⟨?_, by simp [Nat.min_le_left]⟩
      cases n with
      | zero => omega
      | succ n => simp
    · exact i2 p hp

/-! ## Fragment chains whose separators are their minima -/

/-- `Node::min` as `pack_branches` uses it (`getD 0` is the panic on an empty page). -/
def minE (h : Nat) (n : Node) : E := (minOpt h n).getD 0

/-- Well-formed sibling fragments, separated by their minima (`pack_branches`, `node.rs:80`). -/
def MinChain (c : Cfg) (h : Nat) (lo : Nat) (ns : List Node) (hi : Nat) : Prop :=
  COK (fun a b n => WFb c false h n a b) lo (ns.tail.map (minE h)) ns hi

theorem minE_wf {c : Cfg} {h : Nat} {n : Node} {lo hi : Nat} (hw : WFb c false h n lo hi) :
    ∃ t, toList h n = minE h n :: t := by
  have hm := minOpt_head hw
  have hne := WFb_ne hw
  cases hl : toList h n with
  | nil => exact absurd hl hne
  | cons x t => refine ⟨t, ?_⟩; simp [minE, hm, hl]

theorem minE_mem {c : Cfg} {h : Nat} {n : Node} {lo hi : Nat} (hw : WFb c false h n lo hi) :
    minE h n ∈ toList h n := by
  obtain ⟨t, ht⟩ := minE_wf hw; rw [ht]; exact List.mem_cons_self

theorem COK_first {P : Nat → Nat → Node → Prop} {lo lo' hi : Nat} {ss : List E} {n : Node} {ns : List Node}
    (hk : COK P lo ss (n :: ns) hi) (hP : ∀ b, P lo b n → P lo' b n) : COK P lo' ss (n :: ns) hi := by
  match ss, ns, hk with
  | [], [], hk => exact hP _ hk
  | s :: ss, _ :: _, hk => exact ⟨hP _ hk.1, hk.2⟩
  | [], _ :: _, hk => simp [COK] at hk
  | _ :: _, [], hk => simp [COK] at hk

/-- A well-formed subtree stays well-formed with its lower bound raised to its own minimum. -/
theorem WFb_narrow {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, WFb c false h n lo hi →
    WFb c false h n (minE h n) hi
  | 0, .leaf es, lo, hi, hw => by
    obtain ⟨hlt, hs, hb, hl, hne⟩ := hw
    cases es with
    | nil => exact absurd rfl (hne rfl)
    | cons x t =>
      simp only [minE, minOpt, List.head?_cons, Option.getD_some]
      refine ⟨(hb x List.mem_cons_self).2, hs, fun e he => ⟨?_, (hb e he).2⟩, hl, hne⟩
      rcases List.mem_cons.1 he with rfl | he
      · exact Nat.le_refl _
      · exact Nat.le_of_lt ((sorted_cons.1 hs).1 e he)
  | h + 1, .branch seps cs, lo, hi, hw => by
    obtain ⟨hlt, h2, hB, hk⟩ := hw
    match cs, hk with
    | c0 :: cs', hk =>
      have hm : minE (h + 1) (.branch seps (c0 :: cs')) = minE h c0 := rfl
      rw [hm]
      obtain ⟨b, hb⟩ := COK_head hk
      have hlt' : minE h c0 < hi := by
        have hmem := minE_mem hb
        have := (COK_sb (toList h) (fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩) hk).2 (minE h c0)
          (by simp only [List.map_cons, List.flatten_cons]; exact List.mem_append_left _ hmem)
        exact this.2
      refine ⟨hlt', h2, hB, COK_first hk (fun b' hp => WFb_narrow hp)⟩
  | 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

/-- **Any separator chain of well-formed fragments can be re-separated by the fragments' minima.** -/
theorem reseparate {c : Cfg} {h : Nat} : ∀ {lo hi : Nat} {ss : List E} {ns : List Node},
    COK (fun a b n => WFb c false h n a b) lo ss ns hi → MinChain c h lo ns hi
  | _, _, [], [n], hk => hk
  | lo, hi, s :: ss, n :: m :: rest, hk => by
    obtain ⟨h1, h2⟩ : WFb c false h n lo s ∧ COK (fun a b n => WFb c false h n a b) s ss (m :: rest) hi := hk
    have ih : MinChain c h s (m :: rest) hi := reseparate h2
    obtain ⟨b, hb⟩ := COK_head h2
    have hsm : s ≤ minE h m := ((WFb_sb hb).2 _ (minE_mem hb)).1
    show COK _ lo (minE h m :: rest.map (minE h)) (n :: m :: rest) hi
    exact ⟨WFb_bounds_mono h1 (Nat.le_refl _) hsm (by have := WFb_lt h1; omega),
      COK_first ih (fun b' hp => WFb_narrow hp)⟩
  | _, _, [], [], hk => by simp [COK] at hk
  | _, _, [], _ :: _ :: _, hk => by simp [COK] at hk
  | _, _, _ :: _, [], hk => by simp [COK] at hk
  | _, _, _ :: _, [_], hk => by simp [COK] at hk

/-- Leaves cut from one sorted run, each non-empty and at most `LEAF_MAX`, form a `MinChain`. -/
theorem leafChain (c : Cfg) : ∀ (parts : List (List E)) (lo hi : Nat), parts ≠ [] →
    (∀ p ∈ parts, p ≠ [] ∧ p.length ≤ c.L) → Sorted parts.flatten →
    (∀ e ∈ parts.flatten, lo ≤ e ∧ e < hi) → lo < hi → MinChain c 0 lo (parts.map .leaf) hi
  | [p], lo, hi, _, hp, hs, hb, hlt => by
    have := hp p (by simp)
    show WFb c false 0 (.leaf p) lo hi
    exact ⟨hlt, by simpa using hs, fun e he => hb e (by simpa using he), this.2, fun _ => this.1⟩
  | p :: q :: rest, lo, hi, _, hp, hs, hb, hlt => by
    have hq := hp q (by simp)
    have hpp := hp p (by simp)
    obtain ⟨x, t, hxt⟩ : ∃ x t, q = x :: t := by
      cases q with
      | nil => exact absurd rfl hq.1
      | cons x t => exact ⟨x, t, rfl⟩
    subst hxt
    obtain ⟨hs1, hsq, hs12⟩ := sorted_append.1 (show Sorted (p ++ ((x :: t) ++ rest.flatten)) from hs)
    have hx : x ∈ (x :: t) ++ rest.flatten := List.mem_append_left _ List.mem_cons_self
    have ih := leafChain c ((x :: t) :: rest) x hi (by simp) (fun p hp' => hp p (List.mem_cons_of_mem _ hp'))
      (by simpa using hsq) (fun e he => by
        refine ⟨?_, (hb e (List.mem_append_right p he)).2⟩
        simp only [List.flatten_cons, List.mem_append, List.mem_cons] at he
        rcases he with (rfl | he) | he
        · exact Nat.le_refl _
        · exact Nat.le_of_lt ((sorted_cons.1 (sorted_append.1 hsq).1).1 e he)
        · exact Nat.le_of_lt ((sorted_append.1 hsq).2.2 x List.mem_cons_self e he))
      (hb x (by simp)).2
    show COK _ lo (minE 0 (.leaf (x :: t)) :: (rest.map Node.leaf).map (minE 0)) (.leaf p :: .leaf (x :: t) :: rest.map .leaf) hi
    have hmx : minE 0 (.leaf (x :: t)) = x := rfl
    rw [hmx]
    refine ⟨⟨(hb x (by simp)).1 |> fun h1 => ?_, hs1, fun e he => ⟨(hb e (by simp [he])).1, ?_⟩, hpp.2, fun _ => hpp.1⟩, ih⟩
    · obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hpp.1
      have := (hb y (by simp [hy])).1
      have := hs12 y hy x hx
      omega
    · exact hs12 e he x hx
  | [], _, _, h, _, _, _, _ => absurd rfl h

/-! ## `pack_branches` and `build_root` -/

theorem packBranches_cons (c : Cfg) (h : Nat) (ns : List Node) (hne : ns ≠ []) :
    packBranches c h ns =
      .branch (((ns.take (if ns.length = c.B + 1 then c.B - 1 else min ns.length c.B)).drop 1).map (minE h))
        (ns.take (if ns.length = c.B + 1 then c.B - 1 else min ns.length c.B)) ::
      packBranches c h (ns.drop (if ns.length = c.B + 1 then c.B - 1 else min ns.length c.B)) := by
  rw [packBranches, dite_eq_right hne]; rfl

theorem packBranches_head (c : Cfg) (h : Nat) (x : Node) (xs : List Node) :
    ∃ sp cs' tl, packBranches c h (x :: xs) = .branch sp (x :: cs') :: tl := by
  rw [packBranches_cons c h _ (by simp)]
  have hB := c.hB
  generalize hk : (if (x :: xs).length = c.B + 1 then c.B - 1 else min (x :: xs).length c.B) = k
  have hk1 : 1 ≤ k := by rw [← hk]; simp only [List.length_cons]; split <;> omega
  obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [List.take_succ_cons]
  exact ⟨_, _, _, rfl⟩

theorem minE_packBranches (c : Cfg) (h : Nat) (x : Node) (xs : List Node) (b : Node) (tl : List Node)
    (he : packBranches c h (x :: xs) = b :: tl) : minE (h + 1) b = minE h x := by
  obtain ⟨sp, cs', tl', he'⟩ := packBranches_head c h x xs
  rw [he] at he'
  cases he'
  rfl

/-- **`pack_branches` (`node.rs:65-100`)**: a `MinChain` of two or more fragments becomes a strictly
    shorter `MinChain` one level up, every branch with `2 ..= BRANCH_MAX` children, same entries. -/
theorem packBranches_chain (c : Cfg) (h : Nat) : ∀ (ns : List Node) (lo hi : Nat), 2 ≤ ns.length →
    MinChain c h lo ns hi →
    MinChain c (h + 1) lo (packBranches c h ns) hi ∧ (packBranches c h ns).length < ns.length ∧
      ((packBranches c h ns).map (toList (h + 1))).flatten = (ns.map (toList h)).flatten := by
  intro ns
  induction hn : ns.length using Nat.strongRecOn generalizing ns with
  | _ n ih =>
    intro lo hi h2 hk
    have hB := c.hB
    obtain ⟨x, xs, rfl⟩ : ∃ x xs, ns = x :: xs := by
      cases ns with
      | nil => simp at hn; omega
      | cons x xs => exact ⟨x, xs, rfl⟩
    have hl : (x :: xs).length = xs.length + 1 := rfl
    have hn' : xs.length + 1 = n := by rw [← hl]; exact hn
    rw [packBranches_cons c h _ (by simp)]
    generalize hkk : (if (x :: xs).length = c.B + 1 then c.B - 1 else min (x :: xs).length c.B) = k
    have hkb : 2 ≤ k ∧ k ≤ c.B ∧ k ≤ (x :: xs).length ∧ ((x :: xs).length - k = 0 ∨ 2 ≤ (x :: xs).length - k) := by
      rw [← hkk]; split <;> omega
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    rw [hl] at hkb
    -- the chain over `x :: xs`, separators `xs.map (minE h)`
    have hk' : COK (fun a b n => WFb c false h n a b) lo (xs.map (minE h)) (x :: xs) hi := hk
    have hbr : ∀ bnd, COK (fun a b n => WFb c false h n a b) lo ((xs.take k').map (minE h)) ((x :: xs).take (k' + 1)) bnd →
        lo < bnd → WFb c false (h + 1)
          (.branch ((((x :: xs).take (k' + 1)).drop 1).map (minE h)) ((x :: xs).take (k' + 1))) lo bnd := by
      intro bnd hc hlt
      refine ⟨hlt, by simp; omega, by simp; omega, ?_⟩
      simpa using hc
    have hflat : ((x :: xs).map (toList h)).flatten =
        (((x :: xs).take (k' + 1)).map (toList h)).flatten ++ (((x :: xs).drop (k' + 1)).map (toList h)).flatten := by
      rw [← List.flatten_append, ← List.map_append, List.take_append_drop]
    rcases hkb.2.2.2 with hz | h2'
    · -- the last chunk: everything fits in one branch
      have hdrop : (x :: xs).drop (k' + 1) = [] := List.drop_eq_nil_of_le (by simp; omega)
      have htake : (x :: xs).take (k' + 1) = x :: xs := List.take_of_length_le (by simp; omega)
      rw [hdrop]
      rw [packBranches, dite_eq_left rfl]
      refine ⟨?_, by simp; omega, ?_⟩
      · show WFb c false (h + 1) _ lo hi
        rw [htake]
        refine ⟨COK_lt (fun _ _ _ hp => WFb_lt hp) hk', by simp; omega, by simp; omega, ?_⟩
        simpa using hk'
      · simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil, toList]
        rw [htake]; rfl
    · obtain ⟨hs, w1, w2⟩ := COK_split hk' (k' + 1) (by omega) (by simp; omega)
      simp only [Nat.add_sub_cancel] at hs w1 w2
      have e1 : (xs.map (minE h)).take k' = (xs.take k').map (minE h) := (List.map_take ..).symm
      have hxk : k' < xs.length := by simpa using hs
      have e2 : (xs.map (minE h))[k'] = minE h xs[k'] := List.getElem_map ..
      have e3 : (xs.map (minE h)).drop (k' + 1) = (xs.drop (k' + 1)).map (minE h) := (List.map_drop ..).symm
      have hd : (x :: xs).drop (k' + 1) = xs[k'] :: xs.drop (k' + 1) := by
        rw [List.drop_succ_cons, List.drop_eq_getElem_cons hxk]
      rw [e1, e2] at w1
      rw [e2, e3] at w2
      -- the rest is a `MinChain` from its own minimum
      have hrest : MinChain c h (minE h xs[k']) ((x :: xs).drop (k' + 1)) hi := by
        rw [hd] at w2 ⊢; exact w2
      obtain ⟨r1, r2, r3⟩ := ih ((x :: xs).drop (k' + 1)).length (by simp; omega) _ rfl _ _ (by simp; omega) hrest
      refine ⟨?_, ?_, ?_⟩
      · rw [hd] at r1 ⊢
        obtain ⟨sp, cs', tl, hpe⟩ := packBranches_head c h xs[k'] (xs.drop (k' + 1))
        rw [hpe] at r1 ⊢
        have hm : minE (h + 1) (.branch sp (xs[k'] :: cs')) = minE h xs[k'] := rfl
        show COK _ lo (minE (h + 1) (.branch sp (xs[k'] :: cs')) :: tl.map (minE (h + 1))) _ hi
        rw [hm]
        exact ⟨hbr _ w1 (COK_lt (fun _ _ _ hp => WFb_lt hp) w1), r1⟩
      · simp only [List.length_cons, List.length_drop] at r2 ⊢; omega
      · rw [List.map_cons, List.flatten_cons, r3, hflat]; rfl

/-- **`build_root` (`node.rs:105-114`)** on a `MinChain`: one well-formed node with the same entries
    (`fuel` = the fragment count is enough: each `pack_branches` round shrinks the list). -/
theorem buildRoot_wf (c : Cfg) : ∀ (fuel h : Nat) (frags : List Node) (lo hi : Nat), frags ≠ [] →
    frags.length ≤ fuel + 1 → MinChain c h lo frags hi →
    WFb c false (buildRoot c fuel h frags).2 (buildRoot c fuel h frags).1 lo hi ∧
      toList (buildRoot c fuel h frags).2 (buildRoot c fuel h frags).1 = (frags.map (toList h)).flatten := by
  intro fuel h frags
  induction fuel, h, frags using buildRoot.induct c with
  | case1 fuel h a b tl ih =>
    intro lo hi _ hl hk
    rw [buildRoot]
    obtain ⟨p1, p2, p3⟩ := packBranches_chain c h (a :: b :: tl) lo hi (by simp) hk
    obtain ⟨r1, r2⟩ := ih lo hi (fun he => by rw [he] at p1; simp [MinChain, COK] at p1) (by simp at hl p2; omega) p1
    exact ⟨r1, r2.trans p3⟩
  | case2 t h n =>
    intro lo hi _ _ hk
    rw [buildRoot]
    exact ⟨hk, by simp⟩
  | case3 t h =>
    intro lo hi hne _ _
    exact absurd rfl hne
  | case4 h n tl htl =>
    intro lo hi _ hl _
    cases tl with
    | nil => exact absurd rfl htl
    | cons _ _ => simp at hl

/-! ## `apply_batch` -/

theorem length_le_flatMap {α β} (f : α → List β) : ∀ (l : List α), (∀ x ∈ l, f x ≠ []) → l.length ≤ (l.flatMap f).length
  | [], _ => by simp
  | x :: xs, hf => by
    have h1 := hf x List.mem_cons_self
    have h2 := length_le_flatMap f xs (fun y hy => hf y (List.mem_cons_of_mem _ hy))
    have : 1 ≤ (f x).length := by
      cases hfx : f x with
      | nil => exact absurd hfx h1
      | cons _ _ => simp
    simp only [List.flatMap_cons, List.length_append, List.length_cons]; omega

/-- **The routing sweep of `apply_batch` (`node.rs:284-304`) keeps a separator chain**: each touched
    child is replaced by its fragments, each untouched child kept. -/
theorem route_chain {P : Nat → Nat → Node → Prop} (f : Node → List E) (g : Node → List E → List Node)
    (hg : ∀ a b n sl, P a b n → SortedLe sl → sl ≠ [] → (∀ e ∈ sl, a ≤ e ∧ e < b) →
      g n sl ≠ [] ∧ (∃ ss, COK P a ss (g n sl) b) ∧ ∀ e, e ∈ ((g n sl).map f).flatten ↔ e ∈ f n ∨ e ∈ sl) :
    ∀ {lo hi : Nat} {seps : List E} {cs : List Node} (batch : List E), COK P lo seps cs hi →
      SortedLe batch → (∀ e ∈ batch, lo ≤ e ∧ e < hi) →
      cs.length ≤ ((route seps cs batch).flatMap (fun p => if p.2 = [] then [p.1] else g p.1 p.2)).length ∧
      (∃ ss, COK P lo ss ((route seps cs batch).flatMap (fun p => if p.2 = [] then [p.1] else g p.1 p.2)) hi) ∧
      ∀ e, e ∈ (((route seps cs batch).flatMap (fun p => if p.2 = [] then [p.1] else g p.1 p.2)).map f).flatten ↔
        e ∈ (cs.map f).flatten ∨ e ∈ batch
  | _, _, [], [n], batch, hk, hb, hbd => by
    simp only [route, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    split
    · next h0 =>
      subst h0
      exact ⟨Nat.le_refl _, ⟨[], hk⟩, fun e => by simp⟩
    · next h0 =>
      obtain ⟨g1, g2, g3⟩ := hg _ _ n batch hk hb h0 hbd
      refine ⟨?_, g2, fun e => by rw [g3 e]; simp⟩
      cases hgn : g n batch with
      | nil => exact absurd hgn g1
      | cons _ _ => simp
  | _, _, [], [], _, hk, _, _ => by simp [COK] at hk
  | _, _, [], _ :: _ :: _, _, hk, _, _ => by simp [COK] at hk
  | _, _, _ :: _, [], _, hk, _, _ => by simp [COK] at hk
  | _, _, _ :: _, [_], _, hk, _, _ => by
    simp only [COK] at hk
    exact hk.2.elim
  | lo, hi, s :: ss, n :: c2 :: cs, batch, hk, hb, hbd => by
    obtain ⟨hk1, hk2⟩ : P lo s n ∧ COK P s ss (c2 :: cs) hi := hk
    rw [route_cons2, List.flatMap_cons, drop_takeWhile_length]
    have hsl : SortedLe (batch.takeWhile (fun e => decide (e < s))) := hb.sublist (List.takeWhile_sublist _)
    have hrs : SortedLe (batch.dropWhile (fun e => decide (e < s))) := hb.sublist (List.dropWhile_sublist _)
    obtain ⟨r1, ⟨ss2, r2⟩, r3⟩ := route_chain f g hg (batch.dropWhile (fun e => decide (e < s))) hk2 hrs
      (fun e he => ⟨dropWhile_lt_ge_le s batch hb e he, (hbd e ((List.dropWhile_sublist _).subset he)).2⟩)
    -- the first child's fragments
    have hfirst : 1 ≤ (if batch.takeWhile (fun e => decide (e < s)) = [] then [n]
          else g n (batch.takeWhile (fun e => decide (e < s)))).length ∧
        (∃ ss1, COK P lo ss1 (if batch.takeWhile (fun e => decide (e < s)) = [] then [n]
          else g n (batch.takeWhile (fun e => decide (e < s)))) s) ∧
        ∀ e, e ∈ ((if batch.takeWhile (fun e => decide (e < s)) = [] then [n]
          else g n (batch.takeWhile (fun e => decide (e < s)))).map f).flatten ↔
            e ∈ f n ∨ e ∈ batch.takeWhile (fun e => decide (e < s)) := by
      split
      · next h0 => rw [h0]; exact ⟨by simp, ⟨[], hk1⟩, fun e => by simp⟩
      · next h0 =>
        obtain ⟨g1, g2, g3⟩ := hg _ _ n _ hk1 hsl h0 (fun e he =>
          ⟨(hbd e ((List.takeWhile_sublist _).subset he)).1, by have := mem_takeWhile_p he; simpa using this⟩)
        refine ⟨?_, g2, g3⟩
        cases hgn : g n (batch.takeWhile (fun e => decide (e < s))) with
        | nil => exact absurd hgn g1
        | cons _ _ => simp
    obtain ⟨f1, ⟨ss1, f2⟩, f3⟩ := hfirst
    refine ⟨?_, ⟨ss1 ++ s :: ss2, (COK_append (COK_len f2)).2 ⟨f2, r2⟩⟩, fun e => ?_⟩
    · simp only [List.length_append, List.length_cons] at r1 ⊢; omega
    · have hsplit := List.takeWhile_append_dropWhile (p := fun e => decide (e < s)) (l := batch)
      rw [List.map_append, List.flatten_append, List.mem_append, f3 e, r3 e]
      conv => rhs; rw [← hsplit]
      simp only [List.map_cons, List.flatten_cons, List.mem_append]
      constructor
      · rintro ((h | h) | (h | h))
        · exact Or.inl (Or.inl h)
        · exact Or.inr (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr (Or.inr h)
      · rintro ((h | h) | (h | h))
        · exact Or.inl (Or.inl h)
        · exact Or.inr (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr (Or.inr h)

/-- **`Node::apply_batch` (`node.rs:259-312`)**: on a well-formed subtree (the root leaf may be empty)
    and a non-empty sorted batch within its bounds, the fragments form a `MinChain` over the same
    bounds holding exactly the union of the subtree and the batch. -/
theorem applyBatch_chain (c : Cfg) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi : Nat) (batch : List E),
    WFb c r h n lo hi → SortedLe batch → batch ≠ [] → (∀ e ∈ batch, lo ≤ e ∧ e < hi) →
    applyBatch c h n batch ≠ [] ∧ MinChain c h lo (applyBatch c h n batch) hi ∧
      ∀ e, e ∈ ((applyBatch c h n batch).map (toList h)).flatten ↔ e ∈ toList h n ∨ e ∈ batch
  | 0, r, .leaf es, lo, hi, batch, hw, hb, hne, hbd => by
    obtain ⟨hlt, hs, hbe, hl, _⟩ := hw
    have hL := c.hL
    obtain ⟨m1, _, m3, m4⟩ := mergeWalk_spec none es batch hs hb (fun x hx => by cases hx)
    have hmem : ∀ e, e ∈ mergeWalk none es batch ↔ e ∈ es ∨ e ∈ batch := fun e => by
      rw [m3 e]; simp
    have hmne : mergeWalk none es batch ≠ [] := by
      obtain ⟨b, hbm⟩ := List.exists_mem_of_ne_nil _ hne
      intro h0; have := (hmem b).2 (Or.inr hbm); rw [h0] at this; simp at this
    have hparts : (mergeBatch c es batch).flatten = mergeWalk none es batch ∧
        ∀ p ∈ mergeBatch c es batch, p ≠ [] ∧ p.length ≤ c.L := by
      unfold mergeBatch
      split
      · next hfit => exact ⟨by simp, fun p hp => by simp at hp; subst hp; exact ⟨hmne, by omega⟩⟩
      · exact chunks_spec' c.L _ _
    have hpne : mergeBatch c es batch ≠ [] := by
      intro h0; rw [h0] at hparts; exact hmne hparts.1.symm
    have hflat : ((mergeBatch c es batch).map Node.leaf).map (toList 0) = mergeBatch c es batch := by
      simp [List.map_map, Function.comp_def, toList]
    simp only [applyBatch]
    refine ⟨by simpa using hpne, ?_, fun e => ?_⟩
    · apply leafChain c _ lo hi hpne hparts.2 (by rw [hparts.1]; exact m1) _ hlt
      intro e he
      rw [hparts.1, hmem] at he
      rcases he with he | he
      · exact hbe e he
      · exact hbd e he
    · rw [hflat, hparts.1, hmem]; rfl
  | h + 1, r, .branch seps cs, lo, hi, batch, hw, hb, hne, hbd => by
    obtain ⟨hlt, h2, hB, hk⟩ := hw
    obtain ⟨r1, ⟨ss, r2⟩, r3⟩ := route_chain (P := fun a b n => WFb c false h n a b) (toList h) (applyBatch c h)
      (fun a b n sl hp hs hne' hbd' => by
        obtain ⟨a1, a2, a3⟩ := applyBatch_chain c h false n a b sl hp hs hne' hbd'
        exact ⟨a1, ⟨_, a2⟩, a3⟩) batch hk hb hbd
    have hmc := reseparate r2
    obtain ⟨p1, _, p3⟩ := packBranches_chain c h _ lo hi (by omega) hmc
    simp only [applyBatch]
    refine ⟨fun h0 => by rw [h0] at p1; simp [MinChain, COK] at p1, p1, fun e => ?_⟩
    rw [p3, r3 e]; rfl
  | 0, _, .branch _ _, _, _, _, hw, _, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw, _, _, _ => by simp [WFb] at hw

/-! ## Headlines -/

/-- **`CowBTree::insert_batch` (`mod.rs:241`) keeps the tree well-formed** (any `BRANCH_MAX >= 3`), and
    the tree afterwards holds exactly the old entries and the batch, strictly sorted. -/
theorem insertBatch_wf (c : Cfg) (h : Nat) (t : Node) (batch : List E) (hw : TreeWF c h t) (hb : SortedLe batch)
    (hlt : ∀ e ∈ batch, e < HI) :
    TreeWF c (insertBatch c h t batch).2 (insertBatch c h t batch).1 ∧
      Sorted (toList (insertBatch c h t batch).2 (insertBatch c h t batch).1) ∧
      ∀ e, e ∈ toList (insertBatch c h t batch).2 (insertBatch c h t batch).1 ↔ e ∈ toList h t ∨ e ∈ batch := by
  by_cases h0 : batch = []
  · subst h0
    simp only [insertBatch, ↓reduceIte]
    exact ⟨hw, TreeWF_sorted hw, fun e => by simp⟩
  · have hch := applyBatch_chain c h true t 0 HI batch hw hb h0 (fun e he => ⟨Nat.zero_le _, hlt e he⟩)
    obtain ⟨a1, a2, a3⟩ := hch
    have hbr := buildRoot_wf c (applyBatch c h t batch).length h (applyBatch c h t batch) 0 HI a1 (by omega) a2
    simp only [insertBatch, h0, ↓reduceIte]
    have hw' : TreeWF c _ _ := WFb_weaken hbr.1
    exact ⟨hw', TreeWF_sorted hw', fun e => by rw [hbr.2, a3 e]⟩

/-- **`CowBTree::from_sorted` (`mod.rs:190`) builds a well-formed tree** holding exactly its input. -/
theorem fromSorted_wf (c : Cfg) (pairs : List E) (hs : Sorted pairs) (hlt : ∀ e ∈ pairs, e < HI) :
    TreeWF c (fromSorted c pairs).2 (fromSorted c pairs).1 ∧
      toList (fromSorted c pairs).2 (fromSorted c pairs).1 = pairs := by
  by_cases h0 : pairs = []
  · subst h0
    simp only [fromSorted, ↓reduceIte, TreeWF, WFb, toList]
    exact ⟨⟨by decide, sorted_nil, by simp, by simp, fun h => by cases h⟩, trivial⟩
  · have hL := c.hL
    obtain ⟨c1, c2⟩ := chunks_spec' c.L (by omega) pairs
    have hcne : chunks c.L (by omega) pairs ≠ [] := by
      intro he; rw [he] at c1; exact h0 c1.symm
    have hch := leafChain c (chunks c.L (by omega) pairs) 0 HI hcne c2 (by rw [c1]; exact hs)
      (fun e he => by rw [c1] at he; exact ⟨Nat.zero_le _, hlt e he⟩) (by decide)
    have hbr := buildRoot_wf c ((chunks c.L (by omega) pairs).map Node.leaf).length 0
      ((chunks c.L (by omega) pairs).map Node.leaf) 0 HI (by simpa using hcne) (by omega) hch
    simp only [fromSorted, h0, ↓reduceIte]
    refine ⟨WFb_weaken hbr.1, ?_⟩
    rw [hbr.2]
    simp [List.map_map, Function.comp_def, toList, c1]

/-- The trees the Rust API can build: `CowBTree::new` / `default` (an empty leaf), `from_sorted`, and
    any sequence of `insert`, `remove`, `insert_batch`, `remove_batch` (sorted batches, as the
    `assert!`s demand; entries are `(u64, u64)` pairs, i.e. below `HI`). -/
inductive Reachable (c : Cfg) : Nat → Node → Prop
  | empty : Reachable c 0 (.leaf [])
  | fromSorted (pairs : List E) (hs : Sorted pairs) (hlt : ∀ e ∈ pairs, e < HI) :
      Reachable c (CowBTree.fromSorted c pairs).2 (CowBTree.fromSorted c pairs).1
  | insert {h : Nat} {t : Node} (x : E) (hr : Reachable c h t) (hx : x < HI) :
      Reachable c (CowBTree.insert c h t x).2 (CowBTree.insert c h t x).1
  | remove {h : Nat} {t : Node} (x : E) (hr : Reachable c h t) (hx : x < HI) :
      Reachable c (CowBTree.remove c h t x).2 (CowBTree.remove c h t x).1
  | insertBatch {h : Nat} {t : Node} (batch : List E) (hr : Reachable c h t) (hb : SortedLe batch)
      (hlt : ∀ e ∈ batch, e < HI) :
      Reachable c (CowBTree.insertBatch c h t batch).2 (CowBTree.insertBatch c h t batch).1
  | removeBatch {h h' : Nat} {t t' : Node} (batch : List E) (hr : Reachable c h t) (hb : SortedLe batch)
      (he : removeBatchTree c h t batch = some (t', h')) : Reachable c h' t'

/-- **Every tree the Rust API can build is well-formed** (`BRANCH_MAX >= 4`, the const assert), so the
    content specs `insert_spec`, `remove_spec`, `removeBatchTree_spec`, `insertBatch_wf`, `range_spec`,
    `firstDoc_spec`, … hold for every tree a caller ever holds. -/
theorem Reachable_wf (c : Cfg) (hB4 : 4 ≤ c.B) {h : Nat} {t : Node} (hr : Reachable c h t) : TreeWF c h t := by
  induction hr with
  | empty => exact ⟨by decide, sorted_nil, by simp, by simp, fun h => by cases h⟩
  | fromSorted pairs hs hlt => exact (fromSorted_wf c pairs hs hlt).1
  | insert x _ hx ih => exact (insert_spec c _ _ x ih hx).1
  | remove x _ hx ih => exact (remove_spec c hB4 _ _ x ih hx).1
  | insertBatch batch _ hb hlt ih => exact (insertBatch_wf c _ _ batch ih hb hlt).1
  | removeBatch batch _ hb he ih =>
    obtain ⟨t'', h'', he', hw, _⟩ := removeBatchTree_wf c hB4 _ _ batch ih hb
    rw [he] at he'
    cases he'
    exact hw

/-- `Reachable_wf` on every configuration the const assert admits (`4 ≤ BRANCH_MAX` discharged). -/
theorem Reachable_wf_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) : TreeWF c h t :=
  Reachable_wf c (constAssert_B4 ha) hr

end CowBTree
