import FalkorCowBTree.Batch

/-!
# `remove_batch` after #3283 (W7-btree-1 fixed): the model, and `TreeWF` is preserved

#3283 (799968a8b) changed `merge_underfull` (`node.rs:171-191`): after each `combine_with_sibling`
step it calls the new `Node::merge_underfull_below` (`node.rs:560-573`) on every node the combine
produced (one after a merge, two after a borrow), and then re-examines index `left`. `merge_underfull_below`
re-runs `merge_underfull` over a produced branch's children when one of them is under-full. That
repairs the lone-child branch that a combine used to carry one level up untouched (Bug 3,
`BugsBatch.lean`, now pre-#3283 counterexamples (removed; in 17ccc5412)).

## Model

* `muStep` — one loop body: `combine_with_sibling` (`cws`, `muStep_id`) with the repair `f` applied to
  the produced node(s).
* `muLoop` — the `while` loop, for any repair `f` that never grows a node and keeps its under-flow
  flag unless it shrank it (`Mono`). That is what makes the loop terminate: a merge, or any repair that
  shrinks, drops the total node count (`lsize`); a borrow whose repairs do not shrink leaves two nodes
  that are not under-full, so the under-full count right of the cursor drops.
* `mubF h` — `merge_underfull_below` at height `h`, built by recursion on the height together with its
  `Mono` proof (`mubF (h+1)` runs `muLoop` with `mubF h` as the repair).
* `mergeUnderfullBelow`, `mergeUnderfull`, `removeBatch`, `removeBatchTree` — the Rust functions.
  `removeBatchTree` gives a drained root (a leaf) the ghost height 0 (`leafAt0`); in Rust the height
  is implicit.

## Proven

* `removeBatchTree_wf` / `removeBatchTree_wf_asserted` (headline): for `BRANCH_MAX >= 4` (the const
  assert), on a well-formed tree and a sorted batch, `CowBTree::remove_batch` returns a well-formed tree
  (`TreeWF`) holding exactly the entries not in the batch.
* The invariant behind it: `Good` — a node that is either well-formed, or an under-full branch with one
  child that is itself `Good` (the lone-child chain a batch can leave). `combine_good`: combining two
  `Good` siblings gives nodes whose children are `Good` (`PreG`); `mubF_spec`: the repair turns those
  into `Good` nodes; `muLoop_good`: the loop ends with every child `Good` and, if there are two or
  more, none under-full, hence well-formed; `removeBatch_good`: every subtree `remove_batch` returns is
  `Good` or an empty leaf; `collapse_good`: the root collapse turns a `Good` root into a well-formed one.
-/

namespace CowBTree

/-! ## Model -/

/-- Total node count of a child list (children of height `h`). -/
def lsize (h : Nat) (cs : List Node) : Nat := (cs.map (size h)).sum

/-- Number of under-full nodes in a list. -/
def ufCount (c : Cfg) (cs : List Node) : Nat := cs.countP (isUnderfull c)

/-- A repair at height `h` never grows a node, and keeps its under-flow flag unless it shrank it. -/
def Mono (c : Cfg) (h : Nat) (f : Node → Node) : Prop :=
  ∀ n, size h (f n) ≤ size h n ∧ (size h (f n) = size h n → isUnderfull c (f n) = isUnderfull c n)

/-- `left` in `merge_underfull` (`node.rs:173`) and `combine_with_sibling` (`node.rs:120-124`): the
    pair is `(i, i + 1)`, or `(i - 1, i)` when `i` is the last child. -/
def leftOf (len i : Nat) : Nat := if i + 1 < len then i else i - 1

/-- One body of the #3283 loop (`node.rs:172-179`): `combine_with_sibling` on the pair at `left`, then
    `merge_underfull_below` (`f`) on the one node a merge leaves, or the two a borrow leaves. -/
def muStep (c : Cfg) (f : Node → Node) (seps : List E) (cs : List Node) (i : Nat) : List E × List Node :=
  let li := leftOf cs.length i
  match combine c (cs.getD li default) (seps.getD li 0) (cs.getD (li + 1) default) with
  | .one m => (delAt seps li, delAt (setAt cs li (f m)) (li + 1))
  | .two l s r => (setAt seps li s, setAt (setAt cs li (f l)) (li + 1) (f r))

/-- With no repair, the step is exactly `combine_with_sibling` (`cws`, unchanged by #3283). -/
theorem muStep_id (c : Cfg) (seps : List E) (cs : List Node) (i : Nat) (h1 : 1 < cs.length) (hi : i < cs.length) :
    muStep c id seps cs i = ((cws c seps cs i).1, (cws c seps cs i).2.1) := by
  unfold muStep cws leftOf
  by_cases hl : i + 1 < cs.length
  · simp only [hl, ↓reduceIte]
    split <;> rename_i heq <;> simp only [heq] <;> rfl
  · simp only [hl, ↓reduceIte]
    have h1 : i - 1 + 1 = i := by omega
    rw [h1]
    split <;> rename_i heq <;> simp only [heq] <;> rfl

theorem leftOf_spec (len i : Nat) (h1 : 1 < len) (hi : i < len) :
    leftOf len i + 1 < len ∧ (leftOf len i = i ∨ (leftOf len i + 1 = i ∧ i + 1 = len)) := by
  unfold leftOf; split <;> omega

/-- The step in window form: the pair at `li, li + 1` is replaced by the produced node(s). -/
theorem muStep_window (c : Cfg) (f : Node → Node) (seps : List E) (cs : List Node) (i : Nat)
    (h1 : 1 < cs.length) (hi : i < cs.length) :
    (∃ m, combine c (cs.getD (leftOf cs.length i) default) (seps.getD (leftOf cs.length i) 0)
          (cs.getD (leftOf cs.length i + 1) default) = .one m ∧
        muStep c f seps cs i = (seps.take (leftOf cs.length i) ++ seps.drop (leftOf cs.length i + 1),
          cs.take (leftOf cs.length i) ++ [f m] ++ cs.drop (leftOf cs.length i + 2))) ∨
    (∃ l s r, combine c (cs.getD (leftOf cs.length i) default) (seps.getD (leftOf cs.length i) 0)
          (cs.getD (leftOf cs.length i + 1) default) = .two l s r ∧
        muStep c f seps cs i = (seps.take (leftOf cs.length i) ++ s :: seps.drop (leftOf cs.length i + 1),
          cs.take (leftOf cs.length i) ++ [f l, f r] ++ cs.drop (leftOf cs.length i + 2))) := by
  have hl := (leftOf_spec cs.length i h1 hi).1
  unfold muStep
  generalize leftOf cs.length i = li at hl ⊢
  dsimp only
  split
  · rename_i m heq
    left; refine ⟨m, heq, ?_⟩
    rw [delAt_setAt_succ _ _ _ hl]; rfl
  · rename_i l s r heq
    right; refine ⟨l, s, r, heq, ?_⟩
    rw [setAt_setAt_succ _ _ _ _ hl]; simp [setAt]

/-! ### Sizes -/

theorem size_pos : ∀ (h : Nat) (n : Node), 1 ≤ size h n
  | _, .leaf _ => by simp [size]
  | 0, .branch _ _ => by simp [size]
  | _ + 1, .branch _ _ => by simp [size]

theorem lsize_append (h : Nat) (a b : List Node) : lsize h (a ++ b) = lsize h a + lsize h b := by
  simp [lsize, List.map_append, List.sum_append]

theorem lsize_cons (h : Nat) (a : Node) (b : List Node) : lsize h (a :: b) = size h a + lsize h b := by
  simp [lsize]

theorem lsize_nil (h : Nat) : lsize h [] = 0 := rfl

theorem size_branch (h : Nat) (seps : List E) (cs : List Node) : size (h + 1) (.branch seps cs) = 1 + lsize h cs := rfl

theorem combine_size_one (c : Cfg) (h : Nat) (a b : Node) (s : E) (m : Node) (hc : combine c a s b = .one m) :
    size h m + 1 ≤ size h a + size h b := by
  unfold combine at hc
  split at hc
  · dsimp only at hc; split at hc
    · cases hc; cases h <;> simp [size]
    · cases hc
  · dsimp only at hc; split at hc
    · cases hc
      cases h with
      | zero => simp [size]
      | succ h =>
        simp only [size_branch]
        rename_i s1 c1 s2 c2 _
        simp only [lsize_append]; omega
    · cases hc
  · cases hc; have := size_pos h b; omega

theorem combine_size_two (c : Cfg) (h : Nat) (a b : Node) (s : E) (l : Node) (s' : E) (r : Node)
    (hc : combine c a s b = .two l s' r) : size h l + size h r = size h a + size h b := by
  unfold combine at hc
  split at hc
  · dsimp only at hc; split at hc
    · cases hc
    · cases hc; cases h <;> simp [size]
  · dsimp only at hc; split at hc
    · cases hc
    · cases hc
      cases h with
      | zero => simp [size]
      | succ h =>
        rename_i s1 c1 s2 c2 _
        simp only [size_branch]
        have := lsize_append h ((c1 ++ c2).take ((c1 ++ c2).length / 2)) ((c1 ++ c2).drop ((c1 ++ c2).length / 2))
        rw [List.take_append_drop, lsize_append] at this
        omega
  · cases hc

/-- The left node of a borrow is not under-full either (cf. `combine_two_not_uf` for the right). -/
theorem combine_two_left_not_uf (c : Cfg) (a b : Node) (sep : E) :
    ∀ l s r, combine c a sep b = .two l s r → isUnderfull c l = false := by
  intro l s r hc
  unfold combine at hc
  have hL := c.hL
  have hB := c.hB
  split at hc
  · dsimp only at hc
    split at hc
    · cases hc
    · cases hc; simp only [isUnderfull, List.length_take, decide_eq_false_iff_not]; omega
  · dsimp only at hc
    split at hc
    · cases hc
    · cases hc; simp only [isUnderfull, List.length_take, decide_eq_false_iff_not]; omega
  · cases hc

theorem ufCount_append (c : Cfg) (a b : List Node) : ufCount c (a ++ b) = ufCount c a + ufCount c b := by
  simp [ufCount, List.countP_append]

theorem ufCount_cons (c : Cfg) (a : Node) (b : List Node) :
    ufCount c (a :: b) = (if isUnderfull c a then 1 else 0) + ufCount c b := by
  simp only [ufCount, List.countP_cons]; omega

/-- **The loop measure drops on every repair step** (lexicographic: node count, under-full nodes from
    the cursor on, children left to scan). -/
theorem muStep_dec (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) (seps : List E) (cs : List Node)
    (i : Nat) (h1 : 1 < cs.length) (hi : i < cs.length)
    (hu : isUnderfull c (cs.getD i default) = true) :
    Prod.Lex (· < ·) (Prod.Lex (· < ·) (· < ·))
      (lsize h (muStep c f seps cs i).2, ufCount c ((muStep c f seps cs i).2.drop (leftOf cs.length i)),
        (muStep c f seps cs i).2.length - leftOf cs.length i)
      (lsize h cs, ufCount c (cs.drop i), cs.length - i) := by
  obtain ⟨hl, hcase⟩ := leftOf_spec cs.length i h1 hi
  have hwin := window_eq cs (leftOf cs.length i) hl default
  rcases muStep_window c f seps cs i h1 hi with ⟨m, hc, hst⟩ | ⟨l, s, r, hc, hst⟩
  · rw [hst]
    apply Prod.Lex.left
    have := combine_size_one c h _ _ _ _ hc
    have := (hf m).1
    conv => rhs; rw [hwin]
    simp only [lsize_append, lsize_cons, lsize_nil]
    omega
  · rw [hst]
    have e2 := combine_size_two c h _ _ _ _ _ _ hc
    have fl := hf l
    have fr := hf r
    rw [Prod.lex_def]
    by_cases hlt : size h (f l) + size h (f r) < size h l + size h r
    · left
      conv => rhs; rw [hwin]
      simp only [lsize_append, lsize_cons, lsize_nil]
      omega
    · right
      have el : size h (f l) = size h l := by omega
      have er : size h (f r) = size h r := by omega
      have ul : isUnderfull c (f l) = false := by rw [fl.2 el]; exact combine_two_left_not_uf c _ _ _ l s r hc
      have ur : isUnderfull c (f r) = false := by rw [fr.2 er]; exact combine_two_not_uf c _ _ _ l s r hc
      refine ⟨?_, ?_⟩
      · conv => rhs; rw [hwin]
        simp only [lsize_append, lsize_cons, lsize_nil]
        omega
      · apply Prod.Lex.left
        have htl : (cs.take (leftOf cs.length i)).length = leftOf cs.length i := by simp; omega
        rw [List.append_assoc, List.drop_left' htl, ufCount_append, ufCount_cons, ufCount_cons, ul, ur]
        simp only [Bool.false_eq_true, ↓reduceIte, Nat.zero_add]
        rcases hcase with he | ⟨he, hlast⟩
        · have hdi : cs.drop i = cs.getD i default :: cs.getD (i + 1) default :: cs.drop (i + 2) := by
            rw [List.drop_eq_getElem_cons hi, List.drop_eq_getElem_cons (by omega), getD_eq_getElem _ _ _ hi,
              getD_eq_getElem _ _ _ (by omega)]
          rw [hdi, he, ufCount_cons, ufCount_cons, hu]
          have : ufCount c [] = 0 := rfl
          simp only [↓reduceIte]
          omega
        · have hd : cs.drop (leftOf cs.length i + 2) = [] := List.drop_eq_nil_of_le (by omega)
          have hdi : cs.drop i = [cs.getD i default] := by
            rw [List.drop_eq_getElem_cons hi, List.drop_eq_nil_of_le (by omega), getD_eq_getElem _ _ _ hi]
          rw [hd, hdi, ufCount_cons, hu]
          simp [ufCount]

theorem muStep_len (c : Cfg) (f : Node → Node) (seps : List E) (cs : List Node) (i : Nat)
    (hk : cs.length = seps.length + 1) (h1 : 1 < cs.length) (hi : i < cs.length) :
    (muStep c f seps cs i).2.length = (muStep c f seps cs i).1.length + 1 := by
  have hl := (leftOf_spec cs.length i h1 hi).1
  rcases muStep_window c f seps cs i h1 hi with ⟨m, _, hst⟩ | ⟨l, s, r, _, hst⟩
  · rw [hst]; simp; omega
  · rw [hst]; simp; omega

/-- The `while` loop of `merge_underfull` (`node.rs:169-189`), for a repair `f`. -/
def muLoop (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) (seps : List E) (cs : List Node) (i : Nat) :
    List E × List Node :=
  if hc : 1 < cs.length ∧ i < cs.length then                                    -- line 172
    if hu : isUnderfull c (cs.getD i default) = true then                       -- line 173
      muLoop c h f hf (muStep c f seps cs i).1 (muStep c f seps cs i).2 (leftOf cs.length i)  -- lines 174-182
    else muLoop c h f hf seps cs (i + 1)                                        -- line 184
  else (seps, cs)
termination_by (lsize h cs, ufCount c (cs.drop i), cs.length - i)
decreasing_by
  · exact muStep_dec c h f hf seps cs i hc.1 hc.2 hu
  · have hd : cs.drop i = cs.getD i default :: cs.drop (i + 1) := by
      rw [List.drop_eq_getElem_cons hc.2, getD_eq_getElem _ _ _ hc.2]
    have hu' : isUnderfull c (cs.getD i default) = false := by simpa using hu
    have : ufCount c (cs.drop (i + 1)) = ufCount c (cs.drop i) := by
      rw [hd, ufCount_cons, hu']; simp
    rw [this]
    exact Prod.Lex.right _ (Prod.Lex.right _ (by omega))

theorem muStep_size (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) (seps : List E) (cs : List Node)
    (i : Nat) (h1 : 1 < cs.length) (hi : i < cs.length) :
    lsize h (muStep c f seps cs i).2 + (cs.length - (muStep c f seps cs i).2.length) ≤ lsize h cs ∧
      (muStep c f seps cs i).2.length ≤ cs.length := by
  obtain ⟨hl, _⟩ := leftOf_spec cs.length i h1 hi
  have hwin := window_eq cs (leftOf cs.length i) hl default
  have hst := muStep_window c f seps cs i h1 hi
  generalize leftOf cs.length i = li at hl hwin hst
  have htl : (cs.take li).length = li := by simp; omega
  have hsz : lsize h cs = lsize h (cs.take li) + size h (cs.getD li default) + size h (cs.getD (li + 1) default) +
      lsize h (cs.drop (li + 2)) := by
    conv => lhs; rw [hwin]
    simp only [lsize_append, lsize_cons, lsize_nil]; omega
  have hln : cs.length = li + 2 + (cs.drop (li + 2)).length := by simp; omega
  rcases hst with ⟨m, hc, hst⟩ | ⟨l, s, r, hc, hst⟩
  · rw [hst]
    have := combine_size_one c h _ _ _ _ hc
    have := (hf m).1
    simp only [lsize_append, lsize_cons, lsize_nil, List.length_append, List.length_cons, List.length_nil, htl]
    omega
  · rw [hst]
    have := combine_size_two c h _ _ _ _ _ _ hc
    have := (hf l).1
    have := (hf r).1
    simp only [lsize_append, lsize_cons, lsize_nil, List.length_append, List.length_cons, List.length_nil, htl]
    omega

/-- The loop removes at least one node per child it drops, and never grows. -/
theorem muLoop_size (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) :
    ∀ (seps : List E) (cs : List Node) (i : Nat),
      lsize h (muLoop c h f hf seps cs i).2 + (cs.length - (muLoop c h f hf seps cs i).2.length) ≤ lsize h cs ∧
        (muLoop c h f hf seps cs i).2.length ≤ cs.length := by
  intro seps cs i
  induction seps, cs, i using muLoop.induct c h f hf with
  | case1 seps cs i hc hu ih =>
    rw [muLoop, dite_eq_left hc, dite_eq_left hu]
    have := muStep_size c h f hf seps cs i hc.1 hc.2
    omega
  | case2 seps cs i hc hu ih =>
    rw [muLoop, dite_eq_left hc, dite_eq_right hu]
    exact ih
  | case3 seps cs i hc =>
    rw [muLoop, dite_eq_right hc]
    simp

/-- `Node::merge_underfull_below` (`node.rs:560-573`), given the repair `f` for the children's height:
    re-run the loop over a branch's children when one of them is under-full; a leaf is untouched. -/
def mubBody (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) : Node → Node
  | .branch seps cs =>
    if cs.any (isUnderfull c) then                                   -- line 565
      let r := muLoop c h f hf seps cs 0                             -- lines 570-571
      .branch r.1 r.2
    else .branch seps cs
  | n => n

theorem mubBody_mono (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) : Mono c (h + 1) (mubBody c h f hf) := by
  intro n
  cases n with
  | leaf es => simp [mubBody]
  | branch seps cs =>
    simp only [mubBody]
    split
    · have := muLoop_size c h f hf seps cs 0
      simp only [size_branch]
      refine ⟨by omega, fun he => ?_⟩
      have hl : (muLoop c h f hf seps cs 0).2.length = cs.length := by omega
      simp [isUnderfull, hl]
    · exact ⟨Nat.le_refl _, fun _ => rfl⟩

/-- `merge_underfull_below` at every height, built with its `Mono` proof by recursion on the height
    (a node of height 0 is a leaf, which the Rust leaves untouched). -/
def mubF (c : Cfg) : (h : Nat) → {f : Node → Node // Mono c h f}
  | 0 => ⟨id, fun _ => ⟨Nat.le_refl _, fun _ => rfl⟩⟩
  | h + 1 => ⟨mubBody c h (mubF c h).1 (mubF c h).2, mubBody_mono c h _ _⟩

/-- **`Node::merge_underfull_below`** (`node.rs:560-573`) on a node of height `h`. -/
def mergeUnderfullBelow (c : Cfg) (h : Nat) (n : Node) : Node := (mubF c h).1 n

/-- **`merge_underfull`** (`node.rs:167-191`, after #3283) on children of height `h`. -/
def mergeUnderfull (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) : List E × List Node :=
  muLoop c h (mubF c h).1 (mubF c h).2 seps cs 0

theorem mergeUnderfullBelow_branch (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) :
    mergeUnderfullBelow c (h + 1) (.branch seps cs) =
      if cs.any (isUnderfull c) then .branch (mergeUnderfull c h seps cs).1 (mergeUnderfull c h seps cs).2
      else .branch seps cs := rfl

/-- `Node::remove_batch` (`node.rs:498-555`), after #3283: the repair is
    the new `merge_underfull` (line 553). -/
def removeBatch (c : Cfg) : Nat → Node → List E → Node
  | _, .leaf es, batch => .leaf (subtract es batch)                   -- lines 503-516
  | 0, n@(.branch _ _), _ => n                                        -- unreachable (uniform height)
  | h + 1, .branch seps cs, batch =>
    let kids := (route seps cs batch).map
      (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)    -- lines 523-544
    let st := stripEmpty seps kids 0                                  -- line 547
    if st.2 = [] then .leaf []                                        -- lines 548-550
    else
      let mu := mergeUnderfull c h st.1 st.2                          -- line 552
      .branch mu.1 mu.2                                               -- line 553

/-- A drained root is a leaf; its ghost height is 0 (Rust tracks no height). -/
def leafAt0 : Node × Nat → Node × Nat
  | (.leaf es, _) => (.leaf es, 0)
  | p => p

/-- `CowBTree::remove_batch` (`mod.rs:286-309`). `none` is the `assert!` panic on unsorted input. -/
def removeBatchTree (c : Cfg) (h : Nat) (root : Node) (batch : List E) : Option (Node × Nat) :=
  if batch = [] then some (root, h)                                    -- lines 290-292
  else if batch.Pairwise (· ≤ ·) then some (leafAt0 (collapse h (removeBatch c h root batch)))  -- lines 295-308
  else none                                                            -- lines 295-298 (assert)

/-! ## The invariant a batch leaves behind -/

/-- A node `remove_batch` may hand to its parent: well-formed, or an under-full branch with a single
    child that is itself `Good` (the lone-child chain of Bug 3). A branch with two or more children has
    well-formed children. -/
def Good (c : Cfg) : Nat → Node → Nat → Nat → Prop
  | 0, n, lo, hi => WFb c false 0 n lo hi
  | h + 1, .branch seps cs, lo, hi =>
    lo < hi ∧ 1 ≤ cs.length ∧ cs.length ≤ c.B ∧ COK (fun a b n => Good c h n a b) lo seps cs hi ∧
      (2 ≤ cs.length → COK (fun a b n => WFb c false h n a b) lo seps cs hi)
  | _ + 1, .leaf _, _, _ => False

/-- What `combine` builds from two `Good` siblings: at least two `Good` children (a leaf: well-formed). -/
def PreG (c : Cfg) : Nat → Node → Nat → Nat → Prop
  | 0, n, lo, hi => WFb c false 0 n lo hi
  | h + 1, .branch seps cs, lo, hi =>
    lo < hi ∧ 2 ≤ cs.length ∧ cs.length ≤ c.B ∧ COK (fun a b n => Good c h n a b) lo seps cs hi
  | _ + 1, .leaf _, _, _ => False

section
variable {P Q : Nat → Nat → Node → Prop}

theorem COK_mono_mem :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, (∀ k ∈ ns, ∀ a b, P a b k → Q a b k) →
      COK P lo ss ns hi → COK Q lo ss ns hi
  | _, _, [], [_], hPQ, h => hPQ _ (by simp) _ _ h
  | _, _, [], [], _, h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h => by simp [COK] at h
  | _, _, _ :: _, [], _, h => by simp [COK] at h
  | _, _, _ :: _, _ :: _, hPQ, h => by
    simp only [COK] at h ⊢
    exact ⟨hPQ _ (by simp) _ _ h.1, COK_mono_mem (fun k hk => hPQ k (List.mem_cons_of_mem _ hk)) h.2⟩

theorem COK_widen (hP : ∀ a b a' b' n, P a b n → a' ≤ a → b ≤ b' → P a' b' n) :
    ∀ {lo hi lo' hi' : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → lo' ≤ lo → hi ≤ hi' →
      COK P lo' ss ns hi'
  | _, _, _, _, [], [_], h, h1, h2 => hP _ _ _ _ _ h h1 h2
  | _, _, _, _, [], [], h, _, _ => by simp [COK] at h
  | _, _, _, _, [], _ :: _ :: _, h, _, _ => by simp [COK] at h
  | _, _, _, _, _ :: _, [], h, _, _ => by simp [COK] at h
  | _, _, _, _, _ :: _, _ :: _, h, h1, h2 => by
    simp only [COK] at h ⊢
    exact ⟨hP _ _ _ _ _ h.1 h1 (Nat.le_refl _), COK_widen hP h.2 (Nat.le_refl _) h2⟩

/-- Cutting a chain in the middle (the generic form of `branch_split`). -/
theorem COK_halves {lo hi : Nat} {ss : List E} {cs : List Node}
    (hk : COK P lo ss cs hi) (h2 : 2 ≤ cs.length) :
    COK P lo (ss.take (cs.length / 2)).dropLast (cs.take (cs.length / 2)) ((ss.take (cs.length / 2)).getLast?.getD 0) ∧
    COK P ((ss.take (cs.length / 2)).getLast?.getD 0) (ss.drop (cs.length / 2)) (cs.drop (cs.length / 2)) hi := by
  have hm0 : 0 < cs.length / 2 := by omega
  have hm : cs.length / 2 < cs.length := by omega
  obtain ⟨hs, hk1, hk2⟩ := COK_split hk (cs.length / 2) hm0 hm
  have e1 : (ss.take (cs.length / 2)).dropLast = ss.take (cs.length / 2 - 1) :=
    List.dropLast_take (by omega)
  have e2 : (ss.take (cs.length / 2)).getLast?.getD 0 = ss[cs.length / 2 - 1] := by
    rw [List.getLast?_take]
    simp [show cs.length / 2 ≠ 0 by omega, List.getElem?_eq_getElem hs]
  rw [e1, e2]
  exact ⟨hk1, hk2⟩

end

theorem WFb_good {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, WFb c false h n lo hi → Good c h n lo hi
  | 0, _, _, _, hw => hw
  | _ + 1, .branch _ _, _, _, hw =>
    ⟨hw.1, by have := hw.2.1; omega, hw.2.2.1, COK_mono (fun _ _ _ hp => WFb_good hp) hw.2.2.2, fun _ => hw.2.2.2⟩
  | _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

theorem Good_lt {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, Good c h n lo hi → lo < hi
  | 0, _, _, _, hg => WFb_lt hg
  | _ + 1, .branch _ _, _, _, hg => hg.1
  | _ + 1, .leaf _, _, _, hg => by simp [Good] at hg

theorem PreG_lt {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, PreG c h n lo hi → lo < hi
  | 0, _, _, _, hg => WFb_lt hg
  | _ + 1, .branch _ _, _, _, hg => hg.1
  | _ + 1, .leaf _, _, _, hg => by simp [PreG] at hg

/-- **A `Good` node that is not under-full is well-formed** (`BRANCH_MAX >= 4`: not under-full means
    at least `BRANCH_MAX / 2 >= 2` children). -/
theorem Good_wf {c : Cfg} (hB4 : 4 ≤ c.B) : ∀ {h : Nat} {n : Node} {lo hi : Nat}, Good c h n lo hi →
    isUnderfull c n = false → WFb c false h n lo hi
  | 0, _, _, _, hg, _ => hg
  | _ + 1, .branch _ cs, _, _, hg, hu => by
    simp only [isUnderfull, decide_eq_false_iff_not, Nat.not_lt] at hu
    exact ⟨hg.1, by omega, hg.2.2.1, hg.2.2.2.2 (by omega)⟩
  | _ + 1, .leaf _, _, _, hg, _ => by simp [Good] at hg

theorem WFb_widen {c : Cfg} {r : Bool} {h : Nat} {n : Node} {a b a' b' : Nat} (hw : WFb c r h n a b)
    (h1 : a' ≤ a) (h2 : b ≤ b') : WFb c r h n a' b' :=
  WFb_bounds_mono hw h1 h2 (by have := WFb_lt hw; omega)

theorem Good_widen {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi lo' hi' : Nat}, Good c h n lo hi →
    lo' ≤ lo → hi ≤ hi' → Good c h n lo' hi'
  | 0, _, _, _, _, _, hg, h1, h2 => WFb_widen hg h1 h2
  | h + 1, .branch _ _, _, _, _, _, hg, h1, h2 =>
    ⟨by have := hg.1; omega, hg.2.1, hg.2.2.1,
      COK_widen (P := fun a b n => Good c h n a b) (fun _ _ _ _ _ hp h1 h2 => Good_widen hp h1 h2)
        hg.2.2.2.1 h1 h2,
      fun h2' => COK_widen (P := fun a b n => WFb c false h n a b)
        (fun _ _ _ _ _ hp h1 h2 => WFb_widen hp h1 h2) (hg.2.2.2.2 h2') h1 h2⟩
  | _ + 1, .leaf _, _, _, _, _, hg, _, _ => by simp [Good] at hg

/-! ## `combine` on two `Good` siblings -/

def CombG (c : Cfg) (h : Nat) (a b : Node) (m1 m2 : Nat) : Combined → Prop
  | .one m => PreG c h m m1 m2 ∧ toList h m = toList h a ++ toList h b
  | .two l s r => PreG c h l m1 s ∧ PreG c h r s m2 ∧ toList h l ++ toList h r = toList h a ++ toList h b

/-- **`Node::combine` (`node.rs:384`) of two `Good` siblings** — neither needs to be well-formed (two
    lone-child branches side by side): the result has at least two `Good` children per node. -/
theorem combine_good (c : Cfg) : ∀ (h : Nat) (a b : Node) (m1 sep m2 : Nat),
    Good c h a m1 sep → Good c h b sep m2 → CombG c h a b m1 m2 (combine c a sep b)
  | 0, .leaf p, .leaf q, m1, sep, m2, ha, hb => by
    have := combine_spec c 0 (.leaf p) (.leaf q) m1 sep m2 (WFb_weak ha) (WFb_weak hb) (Or.inl ha)
    revert this
    cases combine c (.leaf p) sep (.leaf q) with
    | one m => exact id
    | two l s r => exact id
  | h + 1, .branch s1 c1, .branch s2 c2, m1, sep, m2, ha, hb => by
    obtain ⟨h1, l1, u1, k1, _⟩ := ha
    obtain ⟨h2, l2, u2, k2, _⟩ := hb
    have hB := c.hB
    have hk := (COK_append (COK_len k1)).2 ⟨k1, k2⟩
    have hge : 2 ≤ (c1 ++ c2).length := by simp; omega
    have hle : (c1 ++ c2).length ≤ 2 * c.B := by simp; omega
    have hflat : ((c1 ++ c2).map (toList h)).flatten = toList (h + 1) (.branch s1 c1) ++ toList (h + 1) (.branch s2 c2) := by
      simp [toList, List.map_append, List.flatten_append]
    simp only [combine]
    generalize c1 ++ c2 = cs at hk hge hle hflat
    generalize s1 ++ sep :: s2 = ss at hk
    by_cases hfit : cs.length ≤ c.B
    · simp only [hfit, ↓reduceIte, CombG]
      exact ⟨⟨by omega, hge, hfit, hk⟩, by simpa [toList] using hflat⟩
    · simp only [hfit, ↓reduceIte, CombG]
      obtain ⟨w1, w2⟩ := COK_halves hk hge
      refine ⟨⟨COK_lt (fun _ _ _ hp => Good_lt hp) w1, by simp; omega, by simp; omega, w1⟩,
        ⟨COK_lt (fun _ _ _ hp => Good_lt hp) w2, by simp; omega, by simp; omega, w2⟩, ?_⟩
      rw [← hflat]
      simp only [toList]
      rw [← List.flatten_append, ← List.map_append, List.take_append_drop]
  | 0, .branch _ _, _, _, _, _, ha, _ => by simp [Good, WFb] at ha
  | 0, .leaf _, .branch _ _, _, _, _, _, hb => by simp [Good, WFb] at hb
  | _ + 1, .leaf _, _, _, _, _, ha, _ => by simp [Good] at ha
  | _ + 1, .branch _ _, .leaf _, _, _, _, _, hb => by simp [Good] at hb

/-! ## The loop and the repair -/

/-- What a repair must do: turn a node `combine` built from `Good` siblings into a `Good` node, keeping
    its entries. -/
def Spec (c : Cfg) (h : Nat) (f : Node → Node) : Prop :=
  ∀ n lo hi, PreG c h n lo hi → Good c h (f n) lo hi ∧ toList h (f n) = toList h n

theorem getD_take_append {α} (cs xs : List α) (li j : Nat) (d : α) (hj : j < li) (hl : li ≤ cs.length) :
    (cs.take li ++ xs).getD j d = cs.getD j d := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp; omega), List.getElem?_take_of_lt hj,
    ← List.getD_eq_getElem?_getD]

/-- **`merge_underfull` (`node.rs:167-191`)**: over a chain of `Good` children it ends with a chain of
    `Good` children, no longer, with the same entries, and — if two or more remain — none under-full. -/
theorem muLoop_good (c : Cfg) (h : Nat) (f : Node → Node) (hf : Mono c h f) (hs : Spec c h f) :
    ∀ (seps : List E) (cs : List Node) (i : Nat) (lo hi : Nat),
      COK (fun a b n => Good c h n a b) lo seps cs hi →
      (∀ j, j < i → j < cs.length → isUnderfull c (cs.getD j default) = false) →
      COK (fun a b n => Good c h n a b) lo (muLoop c h f hf seps cs i).1 (muLoop c h f hf seps cs i).2 hi ∧
      (muLoop c h f hf seps cs i).2.length ≤ cs.length ∧
      (2 ≤ (muLoop c h f hf seps cs i).2.length → ∀ k ∈ (muLoop c h f hf seps cs i).2, isUnderfull c k = false) ∧
      ((muLoop c h f hf seps cs i).2.map (toList h)).flatten = (cs.map (toList h)).flatten := by
  intro seps cs i
  induction seps, cs, i using muLoop.induct c h f hf with
  | case1 seps cs i hc hu ih =>
    intro lo hi hk hpre
    rw [muLoop, dite_eq_left hc, dite_eq_left hu]
    obtain ⟨hl, hcase⟩ := leftOf_spec cs.length i hc.1 hc.2
    have hsz := (muStep_size c h f hf seps cs i hc.1 hc.2).2
    have hst := muStep_window c f seps cs i hc.1 hc.2
    have hwin := window_eq cs (leftOf cs.length i) hl default
    have hli : leftOf cs.length i ≤ i := by rcases hcase with h' | h' <;> omega
    obtain ⟨m1, m2, w1, w2, hctx⟩ := COK_window (leftOf cs.length i) hk hl
    have hcg := combine_good c h _ _ m1 _ m2 w1 w2
    -- the step's facts, in terms of `muStep`
    have key : COK (fun a b n => Good c h n a b) lo (muStep c f seps cs i).1 (muStep c f seps cs i).2 hi ∧
        ((muStep c f seps cs i).2.map (toList h)).flatten = (cs.map (toList h)).flatten ∧
        ∀ j, j < leftOf cs.length i → (muStep c f seps cs i).2.getD j default = cs.getD j default := by
      generalize leftOf cs.length i = li at hl hst hwin hctx hcg hli ⊢
      rcases hst with ⟨m, hcm, hst⟩ | ⟨l, s, r, hcm, hst⟩
      · rw [hcm] at hcg
        obtain ⟨hm, tm⟩ := hcg
        obtain ⟨gm, tfm⟩ := hs m _ _ hm
        rw [hst]
        refine ⟨?_, ?_, fun j hj => by
          dsimp only; rw [List.append_assoc]; exact getD_take_append cs _ li j default hj (by omega)⟩
        · have := hctx [] [f m] gm
          simpa only [List.append_nil] using this
        · conv => rhs; rw [hwin]
          simp only [List.map_append, List.flatten_append, List.map_cons, List.map_nil, List.flatten_cons,
            List.flatten_nil, List.append_nil, tfm, tm, List.append_assoc]
      · rw [hcm] at hcg
        obtain ⟨hl', hr', tlr⟩ := hcg
        obtain ⟨gl, tfl⟩ := hs l _ _ hl'
        obtain ⟨gr, tfr⟩ := hs r _ _ hr'
        rw [hst]
        refine ⟨?_, ?_, fun j hj => ?_⟩
        · have := hctx [s] [f l, f r] ⟨gl, gr⟩
          simpa only [List.append_assoc, List.cons_append, List.nil_append, List.singleton_append] using this
        · conv => rhs; rw [hwin]
          simp only [List.map_append, List.flatten_append, List.map_cons, List.map_nil, List.flatten_cons,
            List.flatten_nil, List.append_nil, tfl, tfr, List.append_assoc]
          rw [← List.append_assoc (toList h l), tlr]
          simp only [List.append_assoc]
        · dsimp only; rw [List.append_assoc]; exact getD_take_append cs _ li j default hj (by omega)
    obtain ⟨k1, k2, k3⟩ := key
    obtain ⟨r1, r2, r3, r4⟩ := ih lo hi k1 (fun j hj hjl => by
      rw [k3 j hj]; exact hpre j (by omega) (by omega))
    exact ⟨r1, by omega, r3, r4.trans k2⟩
  | case2 seps cs i hc hu ih =>
    intro lo hi hk hpre
    rw [muLoop, dite_eq_left hc, dite_eq_right hu]
    exact ih lo hi hk (fun j hj hjl => by
      rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
      · exact hpre j hj hjl
      · simpa using hu)
  | case3 seps cs i hc =>
    intro lo hi hk hpre
    rw [muLoop, dite_eq_right hc]
    dsimp only
    refine ⟨hk, Nat.le_refl _, fun h2 k hk' => ?_, rfl⟩
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hk'
    have := hpre j (by omega) hj
    rwa [getD_eq_getElem _ _ _ hj] at this

/-- **`merge_underfull_below` (`node.rs:560-573`) repairs**: at every height it turns a node `combine`
    built from `Good` siblings into a `Good` node with the same entries. -/
theorem mubF_spec (c : Cfg) (hB4 : 4 ≤ c.B) : ∀ h, Spec c h (mubF c h).1
  | 0 => fun n lo hi hp => ⟨hp, rfl⟩
  | h + 1 => by
    intro n lo hi hp
    match n, hp with
    | .branch seps cs, hp =>
      obtain ⟨hlt, h2, hB, hk⟩ := hp
      show Good c (h + 1) (mubBody c h (mubF c h).1 (mubF c h).2 (.branch seps cs)) lo hi ∧
        toList (h + 1) (mubBody c h (mubF c h).1 (mubF c h).2 (.branch seps cs)) = toList (h + 1) (.branch seps cs)
      simp only [mubBody]
      split
      · obtain ⟨r1, r2, r3, r4⟩ := muLoop_good c h _ (mubF c h).2 (mubF_spec c hB4 h) seps cs 0 lo hi hk
          (fun _ hj _ => absurd hj (Nat.not_lt_zero _))
        have hl1 := COK_len r1
        refine ⟨⟨hlt, by omega, by omega, r1, fun h2' => ?_⟩, by simp only [toList]; exact r4⟩
        exact COK_mono_mem (fun k hk a b hg => Good_wf hB4 hg (r3 h2' k hk)) r1
      · rename_i hany
        simp only [Bool.not_eq_true, List.any_eq_false] at hany
        refine ⟨⟨hlt, by omega, hB, hk, fun _ => ?_⟩, rfl⟩
        exact COK_mono_mem (fun k hk a b hg => Good_wf hB4 hg (by simpa using hany k hk)) hk
    | .leaf _, hp => simp [PreG] at hp

theorem mergeUnderfull_good (c : Cfg) (hB4 : 4 ≤ c.B) (h : Nat) (seps : List E) (cs : List Node) (lo hi : Nat)
    (hk : COK (fun a b n => Good c h n a b) lo seps cs hi) :
    COK (fun a b n => Good c h n a b) lo (mergeUnderfull c h seps cs).1 (mergeUnderfull c h seps cs).2 hi ∧
    (mergeUnderfull c h seps cs).2.length ≤ cs.length ∧
    (2 ≤ (mergeUnderfull c h seps cs).2.length → ∀ k ∈ (mergeUnderfull c h seps cs).2, isUnderfull c k = false) ∧
    ((mergeUnderfull c h seps cs).2.map (toList h)).flatten = (cs.map (toList h)).flatten :=
  muLoop_good c h _ (mubF c h).2 (mubF_spec c hB4 h) seps cs 0 lo hi hk (fun _ hj _ => absurd hj (Nat.not_lt_zero _))

/-! ## `strip_empty` and the routing sweep keep the separator chain -/

section
variable {P Q : Nat → Nat → Node → Prop}

/-- Dropping child `i` and the near separator (`strip_empty`, `node.rs:200-205`) keeps a chain whose
    predicate tolerates wider bounds: the neighbour that absorbs the gap gets the wider range. -/
theorem COK_del (hP : ∀ a b a' b' n, P a b n → a' ≤ a → b ≤ b' → P a' b' n) (hlt : ∀ a b n, P a b n → a < b) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node} (i : Nat), COK P lo ss ns hi → 2 ≤ ns.length →
      i < ns.length → COK P lo (delAt ss (min i (ss.length - 1))) (delAt ns i) hi
  | lo, hi, s :: ss, n :: ns, 0, h, _, _ => by
    simp only [COK] at h
    simp only [Nat.zero_min, delAt, List.take_zero, List.nil_append]
    exact COK_widen hP h.2 (Nat.le_of_lt (hlt _ _ _ h.1)) (Nat.le_refl _)
  | lo, hi, [s], [n, n'], 1, h, _, _ => by
    simp only [COK] at h
    simp only [List.length_singleton, Nat.sub_self, Nat.min_zero, delAt, List.take_zero, List.append_nil,
      List.take_succ_cons, List.drop_succ_cons, List.drop_nil]
    exact hP _ _ _ _ _ h.1 (Nat.le_refl _) (Nat.le_of_lt (hlt _ _ _ h.2))
  | lo, hi, s :: s' :: ss, n :: n' :: ns, i + 1, h, _, hi' => by
    obtain ⟨h1, h2⟩ : P lo s n ∧ COK P s (s' :: ss) (n' :: ns) hi := h
    have hlen := COK_len h2
    have ih := COK_del hP hlt i h2 (by simp at hlen ⊢; omega) (by simp at hi' ⊢; omega)
    have e1 : delAt (s :: s' :: ss) (min (i + 1) ((s :: s' :: ss).length - 1)) =
        s :: delAt (s' :: ss) (min i ((s' :: ss).length - 1)) := by
      simp only [List.length_cons, Nat.add_sub_cancel]
      rw [show min (i + 1) (ss.length + 1) = min i ss.length + 1 by omega]
      simp [delAt]
    have e2 : delAt (n :: n' :: ns) (i + 1) = n :: delAt (n' :: ns) i := by simp [delAt]
    rw [e1, e2]
    exact ⟨h1, ih⟩
  | _, _, [], [_], _, _, h2, _ => by simp at h2
  | _, _, [], [], _, h, _, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h, _, _ => by simp [COK] at h
  | _, _, _ :: _, [], _, h, _, _ => by simp [COK] at h
  | _, _, [_], [_], _, h, _, _ => by simp [COK] at h
  | _, _, [_], _ :: _ :: _ :: _, _, h, _, _ => by simp [COK] at h
  | _, _, [_], [_, _], i + 2, _, _, hi' => by simp at hi'; omega
  | _, _, _ :: _ :: _, [_], _, h, _, _ => by simp [COK] at h

/-- **The routing sweep keeps the chain** (`node.rs:523-544`): replacing each touched child by `g`
    applied to its slice is fine when `g` maps a child with predicate `P` (and a non-empty slice at or
    above its lower bound) to one with predicate `Q`. -/
theorem route_cok (g : Node → List E → Node) (hPQ : ∀ a b n, P a b n → Q a b n)
    (hg : ∀ a b n sl, P a b n → SortedLe sl → sl ≠ [] → (∀ e ∈ sl, a ≤ e) → Q a b (g n sl)) :
    ∀ {lo hi : Nat} {seps : List E} {cs : List Node} (batch : List E), COK P lo seps cs hi →
      SortedLe batch → (∀ e ∈ batch, lo ≤ e) →
      COK Q lo seps ((route seps cs batch).map (fun (p : Node × List E) => if p.2 = [] then p.1 else g p.1 p.2)) hi
  | _, _, [], [n], batch, hk, hb, hlo => by
    simp only [route, List.map_cons, List.map_nil]
    show Q _ _ _
    split
    · exact hPQ _ _ _ hk
    · next h0 => exact hg _ _ n batch hk hb h0 hlo
  | _, _, [], [], _, hk, _, _ => by simp [COK] at hk
  | _, _, [], _ :: _ :: _, _, hk, _, _ => by simp [COK] at hk
  | _, _, _ :: _, [], _, hk, _, _ => by simp [COK] at hk
  | _, _, _ :: _, [_], _, hk, _, _ => by
    simp only [COK] at hk
    exact hk.2.elim
  | lo, hi, s :: ss, n :: c2 :: cs, batch, hk, hb, hlo => by
    simp only [COK] at hk
    rw [route_cons2]
    simp only [List.map_cons, COK]
    rw [drop_takeWhile_length]
    have hsl : SortedLe (batch.takeWhile (fun e => decide (e < s))) := hb.sublist (List.takeWhile_sublist _)
    have hrs : SortedLe (batch.dropWhile (fun e => decide (e < s))) := hb.sublist (List.dropWhile_sublist _)
    have ih := route_cok g hPQ hg (batch.dropWhile (fun e => decide (e < s))) hk.2 hrs
      (dropWhile_lt_ge_le s batch hb)
    refine ⟨?_, ih⟩
    split
    · exact hPQ _ _ _ hk.1
    · next h0 => exact hg _ _ n _ hk.1 hsl h0 (fun e he => hlo e ((List.takeWhile_sublist _).subset he))

end

/-- `Good`, or a drained child that `strip_empty` will drop. -/
def GoodOrEmpty (c : Cfg) (h : Nat) (a b : Nat) (n : Node) : Prop :=
  Good c h n a b ∨ (isEmptyNode n = true ∧ a < b)

theorem GoodOrEmpty_widen {c : Cfg} {h : Nat} {a b a' b' : Nat} {n : Node} (hg : GoodOrEmpty c h a b n)
    (h1 : a' ≤ a) (h2 : b ≤ b') : GoodOrEmpty c h a' b' n := by
  rcases hg with hg | ⟨he, hlt⟩
  · exact Or.inl (Good_widen hg h1 h2)
  · exact Or.inr ⟨he, by omega⟩

theorem GoodOrEmpty_lt {c : Cfg} {h : Nat} {a b : Nat} {n : Node} (hg : GoodOrEmpty c h a b n) : a < b := by
  rcases hg with hg | ⟨_, hlt⟩
  · exact Good_lt hg
  · exact hlt

/-- **`strip_empty` (`node.rs:195-210`)** leaves a chain of non-empty children (or nothing). -/
theorem stripEmpty_good (c : Cfg) (h : Nat) (lo hi : Nat) : ∀ (seps : List E) (cs : List Node) (i : Nat),
    (COK (GoodOrEmpty c h) lo seps cs hi ∨ (cs = [] ∧ seps = [])) →
    (∀ j, j < i → j < cs.length → isEmptyNode (cs.getD j default) = false) →
    (COK (GoodOrEmpty c h) lo (stripEmpty seps cs i).1 (stripEmpty seps cs i).2 hi ∨
      (stripEmpty seps cs i).2 = []) ∧
    (∀ k ∈ (stripEmpty seps cs i).2, isEmptyNode k = false) ∧ (stripEmpty seps cs i).2.length ≤ cs.length := by
  intro seps cs i
  induction seps, cs, i using stripEmpty.induct with
  | case1 seps cs i hi' he ih =>
    intro hk hpre
    rw [stripEmpty, dite_eq_left hi', ite_eq_left he]
    have hdl : (delAt cs i).length = cs.length - 1 := by simp [delAt]; omega
    have hpre' : ∀ j, j < i → j < (delAt cs i).length → isEmptyNode ((delAt cs i).getD j default) = false := by
      intro j hj hjl
      have : (delAt cs i).getD j default = cs.getD j default := by
        unfold delAt; exact getD_take_append cs _ i j default hj (by omega)
      rw [this]; exact hpre j hj (by omega)
    simp only [dite_eq_ite] at ih
    generalize hns : (if seps.isEmpty then seps else delAt seps (min i (seps.length - 1))) = ns at ih ⊢
    rcases hk with hk | ⟨rfl, rfl⟩
    · have hlen := COK_len hk
      by_cases h1 : cs.length = 1
      · have hs0 : seps = [] := by
          cases seps with
          | nil => rfl
          | cons _ _ => simp at hlen; omega
        subst hs0
        simp only [List.isEmpty_nil, ↓reduceIte] at hns
        subst hns
        have hd0 : delAt cs i = [] := by
          apply List.eq_nil_of_length_eq_zero; rw [hdl]; omega
        obtain ⟨r1, r2, r3⟩ := ih (Or.inr ⟨hd0, rfl⟩) hpre'
        exact ⟨r1, r2, by omega⟩
      · have hne : seps.isEmpty = false := by
          cases seps with
          | nil => simp at hlen; omega
          | cons _ _ => rfl
        simp only [hne, Bool.false_eq_true, ↓reduceIte] at hns
        subst hns
        have hk' := COK_del (P := GoodOrEmpty c h) (fun _ _ _ _ _ hp h1 h2 => GoodOrEmpty_widen hp h1 h2)
          (fun _ _ _ hp => GoodOrEmpty_lt hp) i hk (by omega) hi'
        obtain ⟨r1, r2, r3⟩ := ih (Or.inl hk') hpre'
        exact ⟨r1, r2, by omega⟩
    · simp at hi'
  | case2 seps cs i hi' he ih =>
    intro hk hpre
    rw [stripEmpty, dite_eq_left hi', ite_eq_right he]
    exact ih hk (fun j hj hjl => by
      rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
      · exact hpre j hj hjl
      · simpa using he)
  | case3 seps cs i hi' =>
    intro hk hpre
    rw [stripEmpty, dite_eq_right hi']
    dsimp only
    refine ⟨?_, fun k hk' => ?_, Nat.le_refl _⟩
    · rcases hk with hk | ⟨rfl, _⟩
      · exact Or.inl hk
      · exact Or.inr rfl
    · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hk'
      have := hpre j (by omega) hj
      rwa [getD_eq_getElem _ _ _ hj] at this

/-! ## `remove_batch` -/

theorem isEmptyNode_eq {n : Node} (he : isEmptyNode n = true) : n = .leaf [] := by
  cases n with
  | leaf es => simp [isEmptyNode] at he; rw [he]
  | branch _ _ => simp [isEmptyNode] at he

/-- **`Node::remove_batch` (`node.rs:498-555`) after #3283**: on a well-formed non-root subtree and a
    sorted batch at or above its lower bound, the result is `Good` (or a drained, empty leaf) and holds
    exactly the entries not in the batch. -/
theorem removeBatch_good (c : Cfg) (hB4 : 4 ≤ c.B) : ∀ (h : Nat) (n : Node) (lo hi : Nat) (batch : List E),
    WFb c false h n lo hi → SortedLe batch → (∀ e ∈ batch, lo ≤ e) →
    GoodOrEmpty c h lo hi (removeBatch c h n batch) ∧
      toList h (removeBatch c h n batch) = (toList h n).filter (fun e => decide (e ∉ batch))
  | 0, .leaf es, lo, hi, batch, hw, hb, _ => by
    obtain ⟨hlt, hs, hbd, hlen, _⟩ := hw
    simp only [removeBatch, toList]
    rw [subtract_eq es batch hs hb]
    refine ⟨?_, rfl⟩
    by_cases h0 : es.filter (fun e => decide (e ∉ batch)) = []
    · right; rw [h0]; exact ⟨rfl, hlt⟩
    · left
      exact ⟨hlt, Sorted.sublist List.filter_sublist hs, fun e he => hbd e (List.mem_filter.1 he).1,
        Nat.le_trans (List.length_filter_le _ _) hlen, fun _ => h0⟩
  | h + 1, .branch seps cs, lo, hi, batch, hw, hb, hlo => by
    obtain ⟨hlt, h2, hB, hk⟩ := hw
    have hP : ∀ a b n, WFb c false h n a b → a < b ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, (WFb_sb hp).2⟩
    -- the children after the sweep
    have hkids := route_cok (P := fun a b n => WFb c false h n a b) (Q := GoodOrEmpty c h)
      (fun n sl => removeBatch c h n sl) (fun _ _ _ hp => Or.inl (WFb_good hp))
      (fun a b n sl hp hs _ hge => (removeBatch_good c hB4 h n a b sl hp hs hge).1) batch hk hb hlo
    have hrc := route_content h (fun a b n => WFb c false h n a b)
      (fun n sl => toList h (removeBatch c h n sl)) hP
      (fun a b n sl hp hs _ hge => (removeBatch_good c hB4 h n a b sl hp hs hge).2) batch hk hb hlo
    have hkflat : (((route seps cs batch).map
        (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)).map (toList h)).flatten =
        (cs.map (fun n => (toList h n).filter (fun e => decide (e ∉ batch)))).flatten := by
      rw [← hrc, List.map_map]
      congr 1
      apply List.map_congr_left
      intro p _
      simp only [Function.comp]
      split <;> rfl
    have hfil : (cs.map (fun n => (toList h n).filter (fun e => decide (e ∉ batch)))).flatten =
        ((cs.map (toList h)).flatten).filter (fun e => decide (e ∉ batch)) := by
      rw [List.filter_flatten, List.map_map]; rfl
    have hklen : ((route seps cs batch).map
        (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)).length = cs.length := by
      rw [List.length_map, ← List.length_map (f := Prod.fst), route_fst]
    generalize hkd : (route seps cs batch).map
        (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2) = kids at hkids hkflat hklen
    obtain ⟨hst1, _⟩ := stripEmpty_content h seps kids 0
    obtain ⟨s1, s2, s3⟩ := stripEmpty_good c h lo hi seps kids 0 (Or.inl hkids)
      (fun _ hj _ => absurd hj (Nat.not_lt_zero _))
    simp only [removeBatch, toList]
    rw [hkd]
    split
    · next h0 =>
      refine ⟨Or.inr ⟨rfl, hlt⟩, ?_⟩
      rw [h0] at hst1
      simp only [toList]
      rw [← hfil, ← hkflat, ← hst1]; rfl
    · next h0 =>
      have hgk : COK (fun a b n => Good c h n a b) lo (stripEmpty seps kids 0).1 (stripEmpty seps kids 0).2 hi := by
        rcases s1 with s1 | s1
        · refine COK_mono_mem (fun k hk' a b hg => ?_) s1
          rcases hg with hg | ⟨he, _⟩
          · exact hg
          · rw [s2 k hk'] at he; cases he
        · exact absurd s1 h0
      obtain ⟨r1, r2, r3, r4⟩ := mergeUnderfull_good c hB4 h _ _ lo hi hgk
      have hl1 := COK_len r1
      refine ⟨Or.inl ⟨hlt, by omega, by omega, r1, fun h2' => ?_⟩, ?_⟩
      · exact COK_mono_mem (fun k hk' a b hg => Good_wf hB4 hg (r3 h2' k hk')) r1
      · simp only [toList]
        rw [r4, hst1, hkflat, hfil]
  | 0, .branch _ _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | _ + 1, .leaf _, _, _, _, hw, _, _ => by simp [WFb] at hw

/-- The root-collapse loop (`mod.rs:301-308`) turns a `Good` root into a well-formed one: it descends
    the lone-child chain to a node with two or more children, or a leaf. -/
theorem collapse_good (c : Cfg) : ∀ (h : Nat) (n : Node) (lo hi : Nat), Good c h n lo hi →
    WFb c true (collapse h n).2 (collapse h n).1 lo hi
  | 0, .leaf es, _, _, hg => WFb_weaken hg
  | 0, .branch _ _, _, _, hg => by simp [Good, WFb] at hg
  | h + 1, .branch seps [ch], lo, hi, hg => by
    obtain ⟨_, _, _, hk, _⟩ := hg
    match seps, hk with
    | [], hk => exact collapse_good c h ch lo hi hk
    | _ :: ss, hk =>
      obtain ⟨_, hk2⟩ := hk
      match ss, hk2 with
      | [], hk2 => simp [COK] at hk2
      | _ :: _, hk2 => simp [COK] at hk2
  | h + 1, .branch seps [], lo, hi, hg => by simp [Good] at hg
  | h + 1, .branch seps (a :: b :: cs), lo, hi, hg => by
    obtain ⟨h1, _, h3, _, hw⟩ := hg
    simp only [collapse]
    exact ⟨h1, by simp, h3, hw (by simp)⟩
  | _ + 1, .leaf _, _, _, hg => by simp [Good] at hg

theorem leafAt0_wf {c : Cfg} {r : Bool} {h : Nat} {n : Node} {lo hi : Nat} (hw : WFb c r h n lo hi) :
    leafAt0 (n, h) = (n, h) := by
  cases n with
  | leaf es => cases h with
    | zero => rfl
    | succ _ => simp [WFb] at hw
  | branch _ _ => rfl

theorem toList_leafAt0 (p : Node × Nat) : toList (leafAt0 p).2 (leafAt0 p).1 = toList p.2 p.1 := by
  obtain ⟨n, h⟩ := p
  cases n <;> simp [leafAt0, toList]

/-- **W7-btree-1 fixed (#3283): `CowBTree::remove_batch` (`mod.rs:286`) keeps the tree well-formed.**
    For `BRANCH_MAX >= 4`, on a well-formed tree and a batch sorted as the `assert!` demands, it does
    not panic, the result is `TreeWF`, and it holds exactly the entries not in the batch. -/
theorem removeBatchTree_wf (c : Cfg) (hB4 : 4 ≤ c.B) (h : Nat) (t : Node) (batch : List E) (hw : TreeWF c h t)
    (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧ TreeWF c h' t' ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) := by
  unfold removeBatchTree
  by_cases h0 : batch = []
  · subst h0
    exact ⟨t, h, by simp, hw, (List.filter_eq_self.2 (by simp)).symm⟩
  · simp only [h0, ↓reduceIte, hb]
    refine ⟨(leafAt0 (collapse h (removeBatch c h t batch))).1, (leafAt0 (collapse h (removeBatch c h t batch))).2,
      rfl, ?_, ?_⟩
    · match h, t, hw with
      | 0, .leaf es, hw =>
        obtain ⟨hlt, hs, hbd, hlen, _⟩ := hw
        simp only [removeBatch, collapse, leafAt0, TreeWF, WFb]
        rw [subtract_eq es batch hs hb]
        exact ⟨hlt, Sorted.sublist List.filter_sublist hs, fun e he => hbd e (List.mem_filter.1 he).1,
          Nat.le_trans (List.length_filter_le _ _) hlen, fun h => by cases h⟩
      | h + 1, .branch seps cs, hw =>
        rcases (removeBatch_good c hB4 (h + 1) (.branch seps cs) 0 HI batch hw hb (fun _ _ => Nat.zero_le _)).1
          with hg | ⟨he, hlt⟩
        · have := collapse_good c _ _ 0 HI hg
          rw [leafAt0_wf this]; exact this
        · rw [isEmptyNode_eq he]
          simp only [collapse, leafAt0, TreeWF, WFb]
          exact ⟨hlt, sorted_nil, by simp, by simp, fun h => by cases h⟩
      | 0, .branch _ _, hw => simp [TreeWF, WFb] at hw
      | _ + 1, .leaf _, hw => simp [TreeWF, WFb] at hw
    · rw [toList_leafAt0, collapse_toList]
      match h, t, hw with
      | 0, .leaf es, hw =>
        simp only [removeBatch, toList]
        exact subtract_eq es batch hw.2.1 hb
      | h + 1, .branch seps cs, hw =>
        exact (removeBatch_good c hB4 (h + 1) (.branch seps cs) 0 HI batch hw hb (fun _ _ => Nat.zero_le _)).2
      | 0, .branch _ _, hw => simp [TreeWF, WFb] at hw
      | _ + 1, .leaf _, hw => simp [TreeWF, WFb] at hw

/-- **Contents** (the pre-#3283 headline, re-proven on the current model): exactly `List.filter`, sorted. -/
theorem removeBatchTree_spec (c : Cfg) (hB4 : 4 ≤ c.B) (h : Nat) (t : Node) (batch : List E) (hw : TreeWF c h t)
    (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) ∧ Sorted (toList h' t') := by
  obtain ⟨t', h', he, hw', ht⟩ := removeBatchTree_wf c hB4 h t batch hw hb
  exact ⟨t', h', he, ht, TreeWF_sorted hw'⟩

/-- An unsorted batch is the `assert!` panic (`mod.rs:295`). -/
theorem removeBatchTree_unsorted (c : Cfg) (h : Nat) (t : Node) (batch : List E) (hne : batch ≠ [])
    (hb : ¬ SortedLe batch) : removeBatchTree c h t batch = none := by
  unfold removeBatchTree; simp [hne, hb]

end CowBTree
