import FalkorCowBTree.Bugs

/-!
# The batch paths after #2278 (3597b3a82): routing fix and `remove_batch`

* `route_keeps_all`, `route_routes_TOP`, `route_TOP_example` — the new sweep (last child takes the
  rest) loses nothing; in particular `(u64::MAX, u64::MAX)` now reaches the last child (Bug 1 fixed).
* `subtract_eq` — the leaf survivor walk of `remove_batch` (`node.rs:487-499`) is `List.filter`.
* `combine_with_sibling` (`cws`), `merge_underfull`, `strip_empty`, `Node::is_underfull`,
  `Branch::is_underfull`, `Node::is_empty_node` — modelled line by line, with `rebalance_eq_cws`
  (the single-key repair is exactly one `combine_with_sibling` step).
* `removeBatch_content`, `removeBatchTree_spec` (headline) — on a well-formed tree and a sorted batch,
  `CowBTree::remove_batch` leaves exactly the entries not in the batch, in order.
-/

namespace CowBTree

/-! ## Model -/

/-- `Node::is_underfull` (`node.rs:546`), reporting through `Branch::is_underfull` (`node.rs:197`). -/
def isUnderfull (c : Cfg) : Node → Bool
  | .leaf es => decide (es.length < c.L / 2)
  | .branch _ cs => decide (cs.length < c.B / 2)

/-- `Branch::is_underfull` (`node.rs:197`): `children.len() < BRANCH_MAX / 2`. -/
def branchUnderfull (c : Cfg) (cs : List Node) : Bool := decide (cs.length < c.B / 2)

/-- `Node::is_empty_node` (`node.rs:555`): a leaf with `count() == 0`. -/
def isEmptyNode : Node → Bool
  | .leaf es => es.isEmpty
  | .branch _ _ => false

/-- `combine_with_sibling` (`node.rs:113-140`): pair child `ci` with its right neighbour (its left one
    when it is the last), combine, and return the new `(seps, children)` and the index to examine next. -/
def cws (c : Cfg) (seps : List E) (cs : List Node) (ci : Nat) : List E × List Node × Nat :=
  let li := if ci + 1 < cs.length then ci else ci - 1          -- lines 119-123
  let ri := if ci + 1 < cs.length then ci + 1 else ci
  match combine c (cs.getD li default) (seps.getD li 0) (cs.getD ri default) with
  | .one m => (delAt seps li, delAt (setAt cs li m) ri, li)       -- lines 125-130
  | .two l s r => (setAt seps li s, setAt (setAt cs li l) ri r, ri) -- lines 132-137

/-- The `.two` arm of `combine` always hands back a right node that is *not* under-full. -/
theorem combine_two_not_uf (c : Cfg) (a b : Node) (sep : E) :
    ∀ l s r, combine c a sep b = .two l s r → isUnderfull c r = false := by
  intro l s r hc
  unfold combine at hc
  have hL := c.hL
  have hB := c.hB
  split at hc
  · dsimp only at hc
    split at hc
    · cases hc
    · cases hc; simp only [isUnderfull, List.length_drop, decide_eq_false_iff_not]; omega
  · dsimp only at hc
    split at hc
    · cases hc
    · cases hc; simp only [isUnderfull, List.length_drop, decide_eq_false_iff_not]; omega
  · cases hc

theorem length_delAt {α} (l : List α) (i : Nat) (h : i < l.length) : (delAt l i).length = l.length - 1 := by
  simp [delAt]; omega

/-- The termination measure of `merge_underfull`'s loop strictly drops on each repair step. -/
theorem cws_measure (c : Cfg) (seps : List E) (cs : List Node) (i : Nat) (h1 : 1 < cs.length) (hi : i < cs.length) :
    3 * (cws c seps cs i).2.1.length - 2 * (cws c seps cs i).2.2 +
      (if (cws c seps cs i).2.2 < (cws c seps cs i).2.1.length ∧
          isUnderfull c ((cws c seps cs i).2.1.getD (cws c seps cs i).2.2 default) = true then 1 else 0)
      < 3 * cs.length - 2 * i + 1 := by
  unfold cws
  by_cases hlast : i + 1 < cs.length
  · simp only [hlast, ↓reduceIte]
    generalize hres : combine c (cs.getD i default) (seps.getD i 0) (cs.getD (i + 1) default) = res
    cases res with
    | one m =>
      simp only
      rw [length_delAt _ _ (by rw [length_setAt _ _ _ hi]; omega), length_setAt _ _ _ hi]
      split <;> omega
    | two l s r =>
      simp only
      rw [length_setAt _ _ _ (by rw [length_setAt _ _ _ hi]; omega), length_setAt _ _ _ hi]
      split <;> omega
  · simp only [hlast, ↓reduceIte]
    generalize hres : combine c (cs.getD (i - 1) default) (seps.getD (i - 1) 0) (cs.getD i default) = res
    cases res with
    | one m =>
      simp only
      rw [length_delAt _ _ (by rw [length_setAt _ _ _ (by omega)]; omega), length_setAt _ _ _ (by omega)]
      split <;> omega
    | two l s r =>
      simp only
      have hr := combine_two_not_uf c _ _ _ l s r hres
      have hl1 : (setAt cs (i - 1) l).length = cs.length := length_setAt _ _ _ (by omega)
      rw [length_setAt _ _ _ (by omega), hl1, getD_setAt_self _ _ _ _ (by omega), hr]
      simp only [Bool.false_eq_true, and_false, ↓reduceIte]
      omega

/-- `merge_underfull` (`node.rs:148-160`): left to right, repair every under-full child with
    `combine_with_sibling` (re-checking the index it returns) while more than one child remains. -/
def mergeUnderfull (c : Cfg) (seps : List E) (cs : List Node) (i : Nat) : List E × List Node :=
  if hc : 1 < cs.length ∧ i < cs.length then
    if hu : isUnderfull c (cs.getD i default) = true then                       -- line 154
      mergeUnderfull c (cws c seps cs i).1 (cws c seps cs i).2.1 (cws c seps cs i).2.2  -- line 155
    else mergeUnderfull c seps cs (i + 1)                                        -- line 157
  else (seps, cs)
termination_by 3 * cs.length - 2 * i + (if i < cs.length ∧ isUnderfull c (cs.getD i default) = true then 1 else 0)
decreasing_by
  · have := cws_measure c seps cs i hc.1 hc.2
    simp only [hc.2, hu, and_self, ↓reduceIte]
    exact this
  · simp only [hc.2, hu, and_false, ↓reduceIte, Bool.false_eq_true]
    split <;> omega

/-- `strip_empty` (`node.rs:165-180`): drop every empty child, and with it the near separator
    (`seps[i]`, or the trailing one for the last child). -/
def stripEmpty (seps : List E) (cs : List Node) (i : Nat) : List E × List Node :=
  if hi : i < cs.length then
    if isEmptyNode (cs.getD i default) then                                       -- line 172
      stripEmpty (if seps.isEmpty then seps else delAt seps (min i (seps.length - 1)))  -- lines 173-175
        (delAt cs i) i                                                            -- line 176
    else stripEmpty seps cs (i + 1)
  else (seps, cs)
termination_by cs.length - i
decreasing_by
  · simp only [delAt, List.length_append, List.length_take, List.length_drop]; omega
  · omega

/-- The leaf arm of `Node::remove_batch` (`node.rs:487-499`): the survivor merge-walk. `bs` is the
    unconsumed batch suffix (`batch[bi..]`), which persists across entries. -/
def subtract : List E → List E → List E
  | [], _ => []
  | p :: ps, bs =>
    let bs' := bs.dropWhile (fun b => decide (b < p))                 -- lines 492-494
    if bs'.head? = some p then subtract ps bs'                         -- lines 495-497
    else p :: subtract ps bs'                                          -- line 498

/-- `Node::remove_batch` (`node.rs:483-538`). Untouched children are shared (`p.2 = []`, line 523). -/
def removeBatch (c : Cfg) : Nat → Node → List E → Node
  | _, .leaf es, batch => .leaf (subtract es batch)                   -- line 500 (`Leaf::from_pairs`)
  | 0, n@(.branch _ _), _ => n                                        -- unreachable (uniform height)
  | h + 1, .branch seps cs, batch =>
    let kids := (route seps cs batch).map
      (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)    -- lines 509-526
    let st := stripEmpty seps kids 0                                  -- line 529
    if st.2 = [] then .leaf []                                        -- lines 530-532
    else
      let mu := mergeUnderfull c st.1 st.2 0                          -- line 534
      .branch mu.1 mu.2                                               -- line 535

/-- `CowBTree::remove_batch` (`mod.rs:269-292`). `none` is the `assert!` panic on unsorted input. -/
def removeBatchTree (c : Cfg) (h : Nat) (root : Node) (batch : List E) : Option (Node × Nat) :=
  if batch = [] then some (root, h)                                    -- lines 273-275
  else if batch.Pairwise (· ≤ ·) then some (collapse h (removeBatch c h root batch))  -- lines 278-291
  else none                                                            -- line 278 (assert)

/-! ## The routing sweep (fix of Bug 1) -/

/-- **The new sweep loses nothing**: the slices handed to the children concatenate back to the batch. -/
theorem route_keeps_all : ∀ (seps : List E) (cs : List Node) (batch : List E), cs ≠ [] →
    ((route seps cs batch).map Prod.snd).flatten = batch
  | _, [], _, h => absurd rfl h
  | _, [_], batch, _ => by simp [route]
  | seps, ch :: c2 :: cs, batch, _ => by
    simp only [route, List.map_cons, List.flatten_cons]
    rw [route_keeps_all seps.tail (c2 :: cs) _ (by simp)]
    have := List.take_append_drop (batch.takeWhile (fun e => decide (e < seps.head?.getD 0))).length batch
    rwa [take_takeWhile_length] at this

/-- **Bug 1 is fixed** (#2278, 3597b3a82): every batch entry — `TOP = (u64::MAX, u64::MAX)` included —
    is handed to some child. Contrast `pre2278_route_never_routes_TOP`. -/
theorem route_routes_TOP (seps : List E) (cs : List Node) (batch : List E) (hcs : cs ≠ []) (hT : TOP ∈ batch) :
    ∃ p ∈ route seps cs batch, TOP ∈ p.2 := by
  rw [← route_keeps_all seps cs batch hcs] at hT
  obtain ⟨l, hl, hTl⟩ := List.mem_flatten.1 hT
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hl
  exact ⟨p, hp, hTl⟩

/-- The concrete Bug-1 input now routes `TOP` to the last child. -/
theorem route_TOP_example :
    route [enc 2 0] [.leaf [enc 0 0, enc 1 0], .leaf [enc 2 0, enc 3 0]] [TOP] =
      [(.leaf [enc 0 0, enc 1 0], []), (.leaf [enc 2 0, enc 3 0], [TOP])] := by
  simp [route, enc, TOP, W]

/-! ## Non-strictly sorted batches -/

/-- A batch as the Rust `assert!` demands it: `windows(2).all(|w| w[0] <= w[1])`. -/
abbrev SortedLe (l : List E) : Prop := l.Pairwise (· ≤ ·)

theorem dropWhile_lt_ge_le (x : E) : ∀ (l : List E), SortedLe l → ∀ e ∈ l.dropWhile (fun e => decide (e < x)), x ≤ e
  | [], _, e, he => by simp at he
  | a :: l, hs, e, he => by
    rw [List.dropWhile_cons] at he
    rw [SortedLe, List.pairwise_cons] at hs
    split at he
    · exact dropWhile_lt_ge_le x l hs.2 e he
    · rename_i hax
      simp at hax
      rcases List.mem_cons.1 he with he | he
      · omega
      · have := hs.1 e he; omega

theorem mem_takeWhile_lt_iff (x : E) : ∀ (l : List E), SortedLe l → ∀ e, e < x →
    (e ∈ l.takeWhile (fun e => decide (e < x)) ↔ e ∈ l)
  | [], _, _, _ => by simp
  | a :: l, hs, e, hex => by
    rw [SortedLe, List.pairwise_cons] at hs
    rw [List.takeWhile_cons]
    by_cases hax : a < x
    · simp only [hax, decide_true, ↓reduceIte, List.mem_cons]
      rw [mem_takeWhile_lt_iff x l hs.2 e hex]
    · simp only [hax, decide_false, Bool.false_eq_true, ↓reduceIte, List.not_mem_nil, List.mem_cons, false_iff]
      rintro (rfl | he)
      · omega
      · have := hs.1 e he; omega

theorem mem_dropWhile_lt_iff (x : E) (l : List E) (e : E) (hxe : x ≤ e) :
    e ∈ l.dropWhile (fun e => decide (e < x)) ↔ e ∈ l := by
  constructor
  · exact fun h => (List.dropWhile_sublist _).subset h
  · intro h
    rw [← List.takeWhile_append_dropWhile (p := fun e => decide (e < x)) (l := l)] at h
    rcases List.mem_append.1 h with h | h
    · have := mem_takeWhile_p h; simp at this; omega
    · exact h

/-- **The leaf survivor walk is `List.filter`**: on a sorted page and a sorted batch, `subtract`
    keeps exactly the entries not in the batch. -/
theorem subtract_eq : ∀ (es bs : List E), Sorted es → SortedLe bs →
    subtract es bs = es.filter (fun e => decide (e ∉ bs))
  | [], _, _, _ => rfl
  | p :: ps, bs, hs, hb => by
    rw [sorted_cons] at hs
    have hb' : SortedLe (bs.dropWhile (fun b => decide (b < p))) := hb.sublist (List.dropWhile_sublist _)
    have hge := dropWhile_lt_ge_le p bs hb
    have ih := subtract_eq ps _ hs.2 hb'
    have hrest : ps.filter (fun e => decide (e ∉ bs.dropWhile (fun b => decide (b < p)))) =
        ps.filter (fun e => decide (e ∉ bs)) := by
      apply List.filter_congr
      intro e he
      simp only [decide_eq_decide]
      rw [mem_dropWhile_lt_iff p bs e (by have := hs.1 e he; omega)]
    have hhead : (bs.dropWhile (fun b => decide (b < p))).head? = some p ↔ p ∈ bs := by
      rw [← mem_dropWhile_lt_iff p bs p (Nat.le_refl _)]
      generalize bs.dropWhile (fun b => decide (b < p)) = d at hb' hge
      cases d with
      | nil => simp
      | cons b d =>
        simp only [List.head?_cons, Option.some.injEq, List.mem_cons]
        constructor
        · intro h; exact Or.inl h.symm
        · rintro (h | h)
          · exact h.symm
          · have := (List.pairwise_cons.1 hb').1 p h; have := hge b List.mem_cons_self; omega
    simp only [subtract, List.filter_cons]
    by_cases hp : p ∈ bs
    · simp only [hhead.2 hp, ↓reduceIte, hp, not_true_eq_false, decide_false, Bool.false_eq_true]
      rw [ih, hrest]
    · have : ¬ (bs.dropWhile (fun b => decide (b < p))).head? = some p := fun h => hp (hhead.1 h)
      simp only [this, ↓reduceIte, hp, not_false_eq_true, decide_true]
      rw [ih, hrest]

/-! ## `Branch::rebalance` is one `combine_with_sibling` step -/

/-- The #2278 refactor of `Branch::rebalance` (`node.rs:204-218`) into `combine_with_sibling` keeps it
    the same function: the old model `rebalance` is the new code's first two components. -/
theorem rebalance_eq_cws (c : Cfg) (seps : List E) (cs : List Node) (ci : Nat) :
    rebalance c seps cs ci = if cs.length < 2 then (seps, cs) else ((cws c seps cs ci).1, (cws c seps cs ci).2.1) := by
  unfold rebalance cws
  split
  · rfl
  · dsimp only
    split
    · next m heq => rw [heq]
    · next l s r heq => rw [heq]

/-! ## Uniform-height shape (all that the content proof needs of the invariant) -/

/-- Leaves exactly at height 0, branches above (the "same kind" siblings `combine` relies on). -/
def Shape : Nat → Node → Prop
  | 0, .leaf _ => True
  | h + 1, .branch _ cs => ∀ k ∈ cs, Shape h k
  | _, _ => False

theorem COK_mem {P : Nat → Nat → Node → Prop} :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → ∀ k ∈ ns, ∃ a b, P a b k
  | _, _, [], [_], h, k, hk => by simp at hk; subst hk; exact ⟨_, _, h⟩
  | _, _, [], [], h, _, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h, _, _ => by simp [COK] at h
  | _, _, _ :: _, [], h, _, _ => by simp [COK] at h
  | _, _, _ :: _, _ :: _, h, k, hk => by
    simp only [COK] at h
    rcases List.mem_cons.1 hk with hk | hk
    · subst hk; exact ⟨_, _, h.1⟩
    · exact COK_mem h.2 k hk

theorem WFb_shape {c : Cfg} : ∀ {r : Bool} {h : Nat} {n : Node} {lo hi : Nat}, WFb c r h n lo hi → Shape h n
  | _, 0, .leaf _, _, _, _ => trivial
  | _, h + 1, .branch _ cs, _, _, hw => by
    intro k hk
    obtain ⟨a, b, hp⟩ := COK_mem hw.2.2.2 k hk
    exact WFb_shape hp
  | _, 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _, _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

/-- Content of `combine` on two same-kind siblings. -/
def CC (h : Nat) (a b : Node) : Combined → Prop
  | .one m => Shape h m ∧ toList h m = toList h a ++ toList h b
  | .two l _ r => Shape h l ∧ Shape h r ∧ toList h l ++ toList h r = toList h a ++ toList h b

theorem combine_content (c : Cfg) (sep : E) : ∀ (h : Nat) (a b : Node), Shape h a → Shape h b →
    CC h a b (combine c a sep b)
  | 0, .leaf p, .leaf q, _, _ => by
    unfold combine; dsimp only
    split
    · exact ⟨trivial, rfl⟩
    · exact ⟨trivial, trivial, by simp [toList]⟩
  | h + 1, .branch s1 c1, .branch s2 c2, ha, hb => by
    unfold combine; dsimp only
    have hall : ∀ k ∈ c1 ++ c2, Shape h k := by
      intro k hk; rcases List.mem_append.1 hk with hk | hk
      · exact ha k hk
      · exact hb k hk
    split
    · exact ⟨hall, by simp [toList, List.map_append, List.flatten_append]⟩
    · refine ⟨fun k hk => hall k (List.mem_of_mem_take hk), fun k hk => hall k (List.mem_of_mem_drop hk), ?_⟩
      simp only [toList]
      rw [← List.flatten_append, ← List.map_append, List.take_append_drop]
      simp [List.map_append, List.flatten_append]
  | 0, .branch _ _, _, ha, _ => by simp [Shape] at ha
  | _ + 1, .leaf _, _, ha, _ => by simp [Shape] at ha
  | 0, .leaf _, .branch _ _, _, hb => by simp [Shape] at hb
  | _ + 1, .branch _ _, .leaf _, _, hb => by simp [Shape] at hb

theorem window_eq {α} (cs : List α) (j : Nat) (hj : j + 1 < cs.length) (d : α) :
    cs = cs.take j ++ [cs.getD j d, cs.getD (j + 1) d] ++ cs.drop (j + 2) := by
  conv => lhs; rw [← List.take_append_drop j cs]
  rw [List.drop_eq_getElem_cons (by omega), List.drop_eq_getElem_cons hj,
    getD_eq_getElem _ _ _ (by omega), getD_eq_getElem _ _ _ hj]
  simp

theorem cws_window (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) (j : Nat) (hj : j + 1 < cs.length)
    (hs : ∀ k ∈ cs, Shape h k) (r : List Node)
    (hr : r = match combine c (cs.getD j default) (seps.getD j 0) (cs.getD (j + 1) default) with
      | .one m => delAt (setAt cs j m) (j + 1)
      | .two l _ x => setAt (setAt cs j l) (j + 1) x) :
    (∀ k ∈ r, Shape h k) ∧ (r.map (toList h)).flatten = (cs.map (toList h)).flatten := by
  have hA : Shape h (cs.getD j default) := hs _ (by rw [getD_eq_getElem _ _ _ (by omega)]; exact List.getElem_mem _)
  have hB : Shape h (cs.getD (j + 1) default) := hs _ (by rw [getD_eq_getElem _ _ _ hj]; exact List.getElem_mem _)
  have hcc := combine_content c (seps.getD j 0) h _ _ hA hB
  have hwin := window_eq cs j hj default
  have hout : ∀ k, k ∈ cs.take j ∨ k ∈ cs.drop (j + 2) → Shape h k := by
    rintro k (hk | hk)
    · exact hs k (List.mem_of_mem_take hk)
    · exact hs k (List.mem_of_mem_drop hk)
  generalize combine c (cs.getD j default) (seps.getD j 0) (cs.getD (j + 1) default) = res at hcc hr
  cases res with
  | one m =>
    obtain ⟨hm, tm⟩ := hcc
    simp only at hr
    rw [delAt_setAt_succ _ _ _ hj] at hr
    subst hr
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_append, List.mem_singleton] at hk
      rcases hk with (hk | rfl) | hk
      · exact hout k (Or.inl hk)
      · exact hm
      · exact hout k (Or.inr hk)
    · conv => rhs; rw [hwin]
      simp [List.map_append, List.flatten_append, tm, List.append_assoc]
  | two l s x =>
    obtain ⟨hl, hx, tlx⟩ := hcc
    simp only at hr
    rw [setAt_setAt_succ _ _ _ _ hj] at hr
    subst hr
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with (hk | rfl | rfl) | hk
      · exact hout k (Or.inl hk)
      · exact hl
      · exact hx
      · exact hout k (Or.inr hk)
    · conv => rhs; rw [hwin]
      rw [flat_window _ _ _ hj, flat_window _ _ _ hj]
      simp only [List.append_assoc] at tlx ⊢
      rw [← List.append_assoc (toList h l) (toList h x), tlx]
      simp only [List.append_assoc]

/-- **`combine_with_sibling` keeps the children's entries and their shape.** -/
theorem cws_content (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) (i : Nat)
    (h1 : 1 < cs.length) (hi : i < cs.length) (hs : ∀ k ∈ cs, Shape h k) :
    (∀ k ∈ (cws c seps cs i).2.1, Shape h k) ∧
      ((cws c seps cs i).2.1.map (toList h)).flatten = (cs.map (toList h)).flatten := by
  unfold cws
  by_cases hlast : i + 1 < cs.length
  · simp only [hlast, ↓reduceIte]
    apply cws_window c h seps cs i hlast hs
    split <;> rfl
  · simp only [hlast, ↓reduceIte]
    obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
    simp only [Nat.add_sub_cancel]
    apply cws_window c h seps cs j (by omega) hs
    split <;> rfl

/-- **`merge_underfull` keeps the children's entries and their shape.** -/
theorem mergeUnderfull_content (c : Cfg) (h : Nat) : ∀ (seps : List E) (cs : List Node) (i : Nat),
    (∀ k ∈ cs, Shape h k) →
    (∀ k ∈ (mergeUnderfull c seps cs i).2, Shape h k) ∧
      ((mergeUnderfull c seps cs i).2.map (toList h)).flatten = (cs.map (toList h)).flatten := by
  intro seps cs i
  induction seps, cs, i using mergeUnderfull.induct c with
  | case1 seps cs i hc hu ih =>
    intro hs
    rw [mergeUnderfull, dif_pos hc, dif_pos hu]
    obtain ⟨hs', hf⟩ := cws_content c h seps cs i hc.1 hc.2 hs
    obtain ⟨r1, r2⟩ := ih hs'
    exact ⟨r1, r2.trans hf⟩
  | case2 seps cs i hc hu ih =>
    intro hs
    rw [mergeUnderfull, dif_pos hc, dif_neg hu]
    exact ih hs
  | case3 seps cs i hc =>
    intro hs
    rw [mergeUnderfull, dif_neg hc]
    exact ⟨hs, rfl⟩

theorem isEmptyNode_toList (h : Nat) (k : Node) (he : isEmptyNode k = true) : toList h k = [] := by
  cases k with
  | leaf es => simp [isEmptyNode] at he; subst he; simp [toList]
  | branch _ _ => simp [isEmptyNode] at he

/-- **`strip_empty` drops only empty children**: the entries are unchanged, and every survivor is a
    child that was there before and is not empty. -/
theorem stripEmpty_content (h : Nat) : ∀ (seps : List E) (cs : List Node) (i : Nat),
    ((stripEmpty seps cs i).2.map (toList h)).flatten = (cs.map (toList h)).flatten ∧
    ∀ k ∈ (stripEmpty seps cs i).2, k ∈ cs.take i ∨ (k ∈ cs ∧ isEmptyNode k = false) := by
  intro seps cs i
  induction seps, cs, i using stripEmpty.induct with
  | case1 seps cs i hi he ih =>
    rw [stripEmpty, dif_pos hi, if_pos he]
    simp only [dite_eq_ite] at ih
    obtain ⟨ih1, ih2⟩ := ih
    have hsp := split_at cs i hi
    have hge : cs.getD i default = cs[i] := getD_eq_getElem _ _ _ hi
    have hdel : delAt cs i = cs.take i ++ cs.drop (i + 1) := rfl
    refine ⟨?_, fun k hk => ?_⟩
    · rw [hge] at he
      have hz : toList h cs[i] = [] := isEmptyNode_toList h _ he
      rw [ih1, hdel]
      conv => rhs; rw [hsp]
      rw [List.map_append, List.map_append, List.map_cons, List.flatten_append, List.flatten_append,
        List.flatten_cons, hz, List.nil_append]
    · rcases ih2 k hk with hk | ⟨hk, hne⟩
      · left; rw [hdel, List.take_append_of_le_length (by simp; omega)] at hk
        simpa [List.take_take] using hk
      · right; refine ⟨?_, hne⟩
        rw [hdel] at hk
        rcases List.mem_append.1 hk with hk | hk
        · exact List.mem_of_mem_take hk
        · exact List.mem_of_mem_drop hk
  | case2 seps cs i hi he ih =>
    rw [stripEmpty, dif_pos hi, if_neg he]
    obtain ⟨ih1, ih2⟩ := ih
    refine ⟨ih1, fun k hk => ?_⟩
    rcases ih2 k hk with hk | hk
    · rw [List.take_add_one, List.getElem?_eq_getElem hi] at hk
      rcases List.mem_append.1 hk with hk | hk
      · exact Or.inl hk
      · simp at hk; subst hk
        rw [getD_eq_getElem _ _ _ hi] at he
        exact Or.inr ⟨List.getElem_mem _, by simpa using he⟩
    · exact Or.inr hk
  | case3 seps cs i hi =>
    rw [stripEmpty, dif_neg hi]
    refine ⟨rfl, fun k hk => Or.inl ?_⟩
    rwa [List.take_of_length_le (by omega)]

/-! ## `remove_batch` content -/

theorem route_fst : ∀ (seps : List E) (cs : List Node) (batch : List E), (route seps cs batch).map Prod.fst = cs
  | _, [], _ => rfl
  | _, [_], _ => rfl
  | seps, _ :: c2 :: cs, batch => by simp only [route, List.map_cons]; rw [route_fst]

theorem COK_ge {P : Nat → Nat → Node → Prop} (f : Node → List E)
    (hP : ∀ a b n, P a b n → a < b ∧ ∀ e ∈ f n, a ≤ e ∧ e < b) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → ∀ k ∈ ns, ∀ e ∈ f k, lo ≤ e
  | _, _, [], [_], h, k, hk, e, he => by
    simp at hk; subst hk; exact ((hP _ _ _ h).2 e he).1
  | _, _, [], [], h, _, _, _, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h, _, _, _, _ => by simp [COK] at h
  | _, _, _ :: _, [], h, _, _, _, _ => by simp [COK] at h
  | _, _, _ :: _, _ :: _, h, k, hk, e, he => by
    simp only [COK] at h
    rcases List.mem_cons.1 hk with hk | hk
    · subst hk; exact ((hP _ _ _ h.1).2 e he).1
    · have := COK_ge f hP h.2 k hk e he; have := (hP _ _ _ h.1).1; omega

theorem route_cons2 (s : E) (ss : List E) (n c2 : Node) (cs : List Node) (batch : List E) :
    route (s :: ss) (n :: c2 :: cs) batch =
      (n, batch.takeWhile (fun e => decide (e < s))) ::
        route ss (c2 :: cs) (batch.drop (batch.takeWhile (fun e => decide (e < s))).length) := rfl

/-- **Routing a sorted batch over a separator chain**: each child is handed exactly the batch
    entries that could live in it, so "remove your slice" per child is "remove the batch" overall. -/
theorem route_content (h : Nat) (P : Nat → Nat → Node → Prop) (g : Node → List E → List E)
    (hP : ∀ a b n, P a b n → a < b ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b)
    (hg : ∀ a b n sl, P a b n → SortedLe sl → sl ≠ [] → (∀ e ∈ sl, a ≤ e) →
      g n sl = (toList h n).filter (fun e => decide (e ∉ sl))) :
    ∀ {lo hi : Nat} {seps : List E} {cs : List Node} (batch : List E), COK P lo seps cs hi →
      SortedLe batch → (∀ e ∈ batch, lo ≤ e) →
      ((route seps cs batch).map (fun (p : Node × List E) => if p.2 = [] then toList h p.1 else g p.1 p.2)).flatten =
        (cs.map (fun n => (toList h n).filter (fun e => decide (e ∉ batch)))).flatten
  | _, _, [], [n], batch, hk, hb, hlo => by
    simp only [route, List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
    split
    · next h0 => subst h0; exact (List.filter_eq_self.2 (by simp)).symm
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
    simp only [List.map_cons, List.flatten_cons]
    rw [drop_takeWhile_length]
    have hsl : SortedLe (batch.takeWhile (fun e => decide (e < s))) := hb.sublist (List.takeWhile_sublist _)
    have hrs : SortedLe (batch.dropWhile (fun e => decide (e < s))) := hb.sublist (List.dropWhile_sublist _)
    have ih := route_content h P g hP hg (batch.dropWhile (fun e => decide (e < s))) hk.2 hrs
      (dropWhile_lt_ge_le s batch hb)
    have hge := COK_ge (toList h) hP hk.2
    have hrest : ((c2 :: cs).map (fun n => (toList h n).filter
        (fun e => decide (e ∉ batch.dropWhile (fun e => decide (e < s)))))).flatten =
        ((c2 :: cs).map (fun n => (toList h n).filter (fun e => decide (e ∉ batch)))).flatten := by
      congr 1
      apply List.map_congr_left
      intro k hkm
      apply List.filter_congr
      intro e he
      simp only [decide_eq_decide]
      rw [mem_dropWhile_lt_iff s batch e (hge k hkm e he)]
    rw [hrest] at ih
    simp only [List.map_cons, List.flatten_cons] at ih
    rw [ih]
    congr 1
    -- the first child: its slice is the batch below `s`
    have hfirst : (toList h n).filter (fun e => decide (e ∉ batch.takeWhile (fun e => decide (e < s)))) =
        (toList h n).filter (fun e => decide (e ∉ batch)) := by
      apply List.filter_congr
      intro e he
      simp only [decide_eq_decide]
      rw [mem_takeWhile_lt_iff s batch hb e ((hP _ _ _ hk.1).2 e he).2]
    rw [← hfirst]
    split
    · next h0 => rw [h0]; exact (List.filter_eq_self.2 (by simp)).symm
    · next h0 => exact hg _ _ n _ hk.1 hsl h0 (fun e he => hlo e ((List.takeWhile_sublist _).subset he))

/-- `remove_batch` returns a node of the same height, or (for a fully drained subtree) an empty leaf. -/
theorem removeBatch_shape (c : Cfg) : ∀ (h : Nat) (n : Node) (batch : List E), Shape h n →
    Shape h (removeBatch c h n batch) ∨ removeBatch c h n batch = .leaf []
  | 0, .leaf es, batch, _ => Or.inl trivial
  | h + 1, .branch seps cs, batch, hs => by
    simp only [removeBatch]
    have hkids : ∀ k ∈ (route seps cs batch).map
        (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2), Shape h k ∨ k = .leaf [] := by
      intro k hk
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hk
      have hp1 : p.1 ∈ cs := by rw [← route_fst seps cs batch]; exact List.mem_map_of_mem hp
      split
      · exact Or.inl (hs _ hp1)
      · exact removeBatch_shape c h p.1 p.2 (hs _ hp1)
    obtain ⟨_, hst⟩ := stripEmpty_content h seps
      ((route seps cs batch).map (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)) 0
    split
    · exact Or.inr rfl
    · left
      apply (mergeUnderfull_content c h _ _ 0 _).1
      intro k hk
      rcases hst k hk with hk' | ⟨hk', hne⟩
      · simp at hk'
      · rcases hkids k hk' with h1 | h1
        · exact h1
        · subst h1; simp [isEmptyNode] at hne
  | 0, .branch _ _, _, hs => by simp [Shape] at hs
  | _ + 1, .leaf _, _, hs => by simp [Shape] at hs

/-- **Content of `Node::remove_batch` (`node.rs:483`)**: on a well-formed subtree and a sorted batch
    whose entries are at or above the subtree's lower bound, the result holds exactly the entries
    not in the batch, in order. -/
theorem removeBatch_content (c : Cfg) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi : Nat) (batch : List E),
    WFb c r h n lo hi → SortedLe batch → (∀ e ∈ batch, lo ≤ e) →
    toList h (removeBatch c h n batch) = (toList h n).filter (fun e => decide (e ∉ batch))
  | 0, _, .leaf es, _, _, batch, hw, hb, _ => by
    simp only [removeBatch, toList]
    exact subtract_eq es batch hw.2.1 hb
  | h + 1, r, .branch seps cs, lo, hi, batch, hw, hb, hlo => by
    obtain ⟨_, _, _, hk⟩ := hw
    have hP : ∀ a b n, WFb c false h n a b → a < b ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, (WFb_sb hp).2⟩
    have hrc := route_content h (fun a b n => WFb c false h n a b)
      (fun n sl => toList h (removeBatch c h n sl)) hP
      (fun a b n sl hp hs _ hge => removeBatch_content c h false n a b sl hp hs hge) batch hk hb hlo
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
    obtain ⟨hst1, hst2⟩ := stripEmpty_content h seps
      ((route seps cs batch).map (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)) 0
    simp only [removeBatch, toList]
    split
    · next h0 =>
      rw [h0] at hst1
      simp only [toList]
      rw [← hfil, ← hkflat, ← hst1]; rfl
    · next h0 =>
      simp only [toList]
      have hshape : ∀ k ∈ (stripEmpty seps ((route seps cs batch).map
          (fun (p : Node × List E) => if p.2 = [] then p.1 else removeBatch c h p.1 p.2)) 0).2, Shape h k := by
        intro k hkk
        rcases hst2 k hkk with hk' | ⟨hk', hne⟩
        · simp at hk'
        · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hk'
          have hp1 : p.1 ∈ cs := by rw [← route_fst seps cs batch]; exact List.mem_map_of_mem hp
          have hs1 : Shape h p.1 := by
            obtain ⟨a, b, hpw⟩ := COK_mem hk p.1 hp1
            exact WFb_shape hpw
          split
          · exact hs1
          · rcases removeBatch_shape c h p.1 p.2 hs1 with h1 | h1
            · exact h1
            · rename_i hne'
              simp only [hne', ↓reduceIte] at hne
              rw [h1] at hne; simp [isEmptyNode] at hne
      rw [(mergeUnderfull_content c h _ _ 0 hshape).2, hst1, hkflat, hfil]
  | 0, _, .branch _ _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw, _, _ => by simp [WFb] at hw

theorem collapse_toList : ∀ (h : Nat) (n : Node), toList (collapse h n).2 (collapse h n).1 = toList h n
  | h + 1, .branch _ [ch] => by simp only [collapse]; rw [collapse_toList h ch]; simp [toList]
  | 0, .leaf _ => rfl
  | 0, .branch _ _ => rfl
  | _ + 1, .leaf _ => rfl
  | _ + 1, .branch _ [] => rfl
  | _ + 1, .branch _ (_ :: _ :: _) => rfl

/-- **`CowBTree::remove_batch` (`mod.rs:269`), headline**: on a well-formed tree and a batch sorted as
    the `assert!` demands, it does not panic, and the tree afterwards holds exactly the entries not in
    the batch — the reference `List.filter` — still strictly sorted. -/
theorem removeBatchTree_spec (c : Cfg) (h : Nat) (t : Node) (batch : List E) (hw : TreeWF c h t)
    (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) ∧ Sorted (toList h' t') := by
  have hs := TreeWF_sorted hw
  unfold removeBatchTree
  by_cases h0 : batch = []
  · subst h0
    refine ⟨t, h, by simp, (List.filter_eq_self.2 (by simp)).symm, hs⟩
  · simp only [h0, ↓reduceIte, hb]
    refine ⟨_, _, rfl, ?_, ?_⟩
    · rw [collapse_toList]
      exact removeBatch_content c h true t 0 HI batch hw hb (fun _ _ => Nat.zero_le _)
    · rw [collapse_toList, removeBatch_content c h true t 0 HI batch hw hb (fun _ _ => Nat.zero_le _)]
      exact Sorted.sublist (List.filter_sublist) hs

/-- An unsorted batch is the `assert!` panic (`mod.rs:278`). -/
theorem removeBatchTree_unsorted (c : Cfg) (h : Nat) (t : Node) (batch : List E) (hne : batch ≠ [])
    (hb : ¬ SortedLe batch) : removeBatchTree c h t batch = none := by
  unfold removeBatchTree; simp [hne, hb]

end CowBTree
