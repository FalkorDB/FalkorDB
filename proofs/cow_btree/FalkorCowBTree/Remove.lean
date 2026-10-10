import FalkorCowBTree.Insert

/-!
# Single remove: `Node::remove_one` / `Branch::rebalance` / `Node::combine` / `CowBTree::remove`

The contract that makes remove work is the **underflow flag**: `remove_one` returns
`Some(false)` only if the returned node is again a *full* well-formed non-root node; with
`Some(true)` the node may be *weak* (an empty leaf, or a branch with a single child) and the parent
must repair it (`rebalance`). The flag is computed as `count < LEAF_MAX / 2` for a leaf and
`children < BRANCH_MAX / 2` for a branch (`node.rs:197`, reported at `:473`). A weak branch has one child, so the
contract needs `1 < BRANCH_MAX / 2`, i.e. **`BRANCH_MAX >= 4`**. The const assert only demands
`BRANCH_MAX >= 3` (`mod.rs:135-138`); `Bugs.lean` shows the contract (and the tree) breaking at 3.
-/

namespace CowBTree

/-- A node as `remove_one` may hand it back with the underflow flag set. -/
def Weak (c : Cfg) : Nat → Node → Nat → Nat → Prop
  | 0, .leaf es, lo, hi => WFb c true 0 (.leaf es) lo hi
  | h + 1, .branch seps cs, lo, hi =>
    lo < hi ∧ 1 ≤ cs.length ∧ cs.length ≤ c.B ∧ COK (fun a b n => WFb c false h n a b) lo seps cs hi
  | _, _, _, _ => False

theorem WFb_weak {c : Cfg} : ∀ {r : Bool} {h : Nat} {n : Node} {lo hi : Nat}, WFb c r h n lo hi → Weak c h n lo hi
  | true, 0, .leaf _, _, _, hw => hw
  | false, 0, .leaf _, _, _, hw => WFb_weaken hw
  | _, _ + 1, .branch _ _, _, _, hw => ⟨hw.1, by have := hw.2.1; omega, hw.2.2.1, hw.2.2.2⟩
  | _, 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _, _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

theorem Weak_lt {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, Weak c h n lo hi → lo < hi
  | 0, .leaf _, _, _, hw => hw.1
  | _ + 1, .branch _ _, _, _, hw => hw.1
  | 0, .branch _ _, _, _, hw => by simp [Weak] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [Weak] at hw

theorem Weak_sb {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, Weak c h n lo hi →
    Sorted (toList h n) ∧ ∀ e ∈ toList h n, lo ≤ e ∧ e < hi
  | 0, .leaf _, _, _, hw => WFb_sb hw
  | h + 1, .branch _ _, _, _, hw => by
    simp only [toList]
    exact COK_sb (toList h) (fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩) hw.2.2.2
  | 0, .branch _ _, _, _, hw => by simp [Weak] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [Weak] at hw

/-! ## More `COK` tools -/

section
variable {P : Nat → Nat → Node → Prop}

/-- One-hole context at any child index. -/
theorem COK_focus :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node} (i : Nat), COK P lo ss ns hi → i < ns.length →
      ∃ (a b : Nat), P a b (ns.getD i default) ∧
        ∀ (ts : List E) (ms : List Node), COK P a ts ms b →
          COK P lo (ss.take i ++ ts ++ ss.drop i) (ns.take i ++ ms ++ ns.drop (i + 1)) hi
  | _, _, [], [n], 0, h, _ => ⟨_, _, by simpa [COK] using h, fun ts ms hm => by simpa using hm⟩
  | _, _, [], [_], _ + 1, _, hi' => by simp at hi'
  | _, _, [], [], _, h, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h, _ => by simp [COK] at h
  | _, _, _ :: _, [], _, h, _ => by simp [COK] at h
  | lo, hi, s :: ss, n :: ns, 0, h, _ => by
    simp only [COK] at h
    refine ⟨lo, s, by simpa using h.1, fun ts ms hm => ?_⟩
    simp only [List.take_zero, List.nil_append, List.drop_zero, List.drop_succ_cons]
    exact (COK_append (COK_len hm)).2 ⟨hm, h.2⟩
  | lo, hi, s :: ss, n :: ns, i + 1, h, hi' => by
    simp only [COK] at h
    obtain ⟨a, b, hp, hctx⟩ := COK_focus i h.2 (by simpa using hi')
    refine ⟨a, b, by simpa using hp, fun ts ms hm => ?_⟩
    simp only [List.take_succ_cons, List.drop_succ_cons, List.cons_append, COK]
    exact ⟨h.1, hctx ts ms hm⟩

/-- Everything outside the routed child is strictly on the correct side of `x`. -/
theorem COK_route_outside (f : Node → List E)
    (hP : ∀ a b n, P a b n → a < b ∧ Sorted (f n) ∧ ∀ e ∈ f n, a ≤ e ∧ e < b) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node} (x : E), COK P lo ss ns hi →
      (∀ e ∈ ((ns.take (childIndex ss x)).map f).flatten, e < x) ∧
      (∀ e ∈ ((ns.drop (childIndex ss x + 1)).map f).flatten, x < e)
  | _, _, [], [n], x, h => by simp [childIndex_nil]
  | _, _, [], [], _, h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h => by simp [COK] at h
  | _, _, _ :: _, [], _, h => by simp [COK] at h
  | lo, hi, s :: ss, n :: ns, x, h => by
    simp only [COK] at h
    by_cases hsx : s ≤ x
    · have hci : childIndex (s :: ss) x = childIndex ss x + 1 := by rw [childIndex_cons]; simp [hsx]
      obtain ⟨ih1, ih2⟩ := COK_route_outside f hP x h.2
      rw [hci]
      simp only [List.take_succ_cons, List.drop_succ_cons, List.map_cons, List.flatten_cons, List.mem_append]
      refine ⟨fun e he => ?_, ih2⟩
      rcases he with he | he
      · have := ((hP _ _ _ h.1).2.2 e he).2; omega
      · exact ih1 e he
    · have hci : childIndex (s :: ss) x = 0 := by rw [childIndex_cons]; simp [hsx]
      rw [hci]
      simp only [List.take_zero, List.map_nil, List.flatten_nil, List.not_mem_nil, false_imp_iff,
        implies_true, List.drop_succ_cons, true_and]
      intro e he
      have := ((COK_sb f hP h.2).2 e he).1; omega

/-- A window of two adjacent children `li, li + 1` and its context (used by `rebalance`). -/
theorem COK_window :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node} (li : Nat), COK P lo ss ns hi → li + 1 < ns.length →
      ∃ (m1 m2 : Nat), P m1 (ss.getD li 0) (ns.getD li default) ∧ P (ss.getD li 0) m2 (ns.getD (li + 1) default) ∧
        ∀ (ts : List E) (ms : List Node), COK P m1 ts ms m2 →
          COK P lo (ss.take li ++ ts ++ ss.drop (li + 1)) (ns.take li ++ ms ++ ns.drop (li + 2)) hi
  | _, _, [], [_], _, _, hl => by simp at hl
  | _, _, [], [], _, h, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h, _ => by simp [COK] at h
  | _, _, _ :: _, [], _, h, _ => by simp [COK] at h
  | _, _, [_], [_], _, h, _ => by simp [COK] at h
  | _, _, _ :: _ :: _, [_], _, h, _ => by simp [COK] at h
  | lo, hi, [s0], n0 :: n1 :: C, 0, h, _ => by
    simp only [COK] at h
    match C, h with
    | [], h =>
      refine ⟨lo, hi, by simpa using h.1, by simpa [COK] using h.2, fun ts ms hm => ?_⟩
      simpa using hm
    | _ :: _, h => simp [COK] at h
  | lo, hi, s0 :: s1 :: S, n0 :: n1 :: C, 0, h, _ => by
    simp only [COK] at h
    match C, h with
    | [], h => simp [COK] at h
    | n2 :: C, h =>
      refine ⟨lo, s1, by simpa using h.1, by simpa using h.2.1, fun ts ms hm => ?_⟩
      simp only [List.take_zero, List.nil_append, List.drop_succ_cons, List.drop_zero]
      exact (COK_append (COK_len hm)).2 ⟨hm, h.2.2⟩
  | lo, hi, s :: ss, n :: ns, li + 1, h, hl => by
    simp only [COK] at h
    obtain ⟨m1, m2, h1, h2, hctx⟩ := COK_window li h.2 (by simpa using hl)
    refine ⟨m1, m2, by simpa using h1, by simpa using h2, fun ts ms hm => ?_⟩
    simp only [List.take_succ_cons, List.drop_succ_cons, List.cons_append, COK]
    exact ⟨h.1, hctx ts ms hm⟩

end

/-! ## List edits used by `rebalance` -/

theorem setAt_eq_set {α} (l : List α) (i : Nat) (x : α) (hi : i < l.length) : setAt l i x = l.set i x := by
  rw [List.set_eq_take_append_cons_drop]; simp [hi, setAt]

theorem take_setAt_of_le {α} (l : List α) (i j : Nat) (x : α) (hj : j ≤ i) (hi : i < l.length) :
    (setAt l i x).take j = l.take j := by
  rw [setAt_eq_set _ _ _ hi, List.take_set, List.set_eq_of_length_le (by simp; omega)]

theorem drop_setAt_of_lt {α} (l : List α) (i j : Nat) (x : α) (hj : i < j) (hi : i < l.length) :
    (setAt l i x).drop j = l.drop j := by
  rw [setAt_eq_set _ _ _ hi, List.drop_set]; simp [hj]

theorem length_setAt {α} (l : List α) (i : Nat) (x : α) (hi : i < l.length) : (setAt l i x).length = l.length := by
  rw [setAt_eq_set _ _ _ hi, List.length_set]

theorem getD_setAt_self {α} (l : List α) (i : Nat) (x d : α) (hi : i < l.length) : (setAt l i x).getD i d = x := by
  rw [setAt_eq_set _ _ _ hi, List.getD_eq_getElem?_getD, List.getElem?_set_self hi]; rfl

theorem getD_setAt_ne {α} (l : List α) (i j : Nat) (x d : α) (hi : i < l.length) (hij : j ≠ i) :
    (setAt l i x).getD j d = l.getD j d := by
  rw [setAt_eq_set _ _ _ hi, List.getD_eq_getElem?_getD, List.getElem?_set_ne (Ne.symm hij),
    ← List.getD_eq_getElem?_getD]

theorem take_succ_setAt {α} (l : List α) (li : Nat) (m : α) (hl : li < l.length) :
    (setAt l li m).take (li + 1) = l.take li ++ [m] := by
  unfold setAt
  have hlen : (l.take li).length = li := by simp; omega
  rw [List.take_append, hlen, List.take_of_length_le (by rw [hlen]; omega)]
  simp

theorem delAt_setAt_succ {α} (l : List α) (li : Nat) (m : α) (hl : li + 1 < l.length) :
    delAt (setAt l li m) (li + 1) = l.take li ++ [m] ++ l.drop (li + 2) := by
  unfold delAt
  rw [drop_setAt_of_lt l li (li + 1 + 1) m (by omega) (by omega), take_succ_setAt l li m (by omega)]

theorem setAt_setAt_succ {α} (l : List α) (li : Nat) (x y : α) (hl : li + 1 < l.length) :
    setAt (setAt l li x) (li + 1) y = l.take li ++ [x, y] ++ l.drop (li + 2) := by
  rw [setAt_eq, take_succ_setAt l li x (by omega), drop_setAt_of_lt l li (li + 1 + 1) x (by omega) (by omega)]
  simp

/-! ## Splitting an overflowing branch (shared by `insert_one` and `combine`) -/

theorem branch_split (c : Cfg) (h : Nat) {lo hi : Nat} {ss : List E} {cs : List Node}
    (hk : COK (fun a b n => WFb c false h n a b) lo ss cs hi) (hgt : c.B < cs.length) (hle : cs.length ≤ 2 * c.B) :
    WFb c false (h + 1) (.branch (ss.take (cs.length / 2)).dropLast (cs.take (cs.length / 2))) lo
      ((ss.take (cs.length / 2)).getLast?.getD 0) ∧
    WFb c false (h + 1) (.branch (ss.drop (cs.length / 2)) (cs.drop (cs.length / 2)))
      ((ss.take (cs.length / 2)).getLast?.getD 0) hi := by
  have hB := c.hB
  have hm0 : 0 < cs.length / 2 := by omega
  have hm : cs.length / 2 < cs.length := by omega
  obtain ⟨hs, hk1, hk2⟩ := COK_split hk (cs.length / 2) hm0 hm
  have e1 : (ss.take (cs.length / 2)).dropLast = ss.take (cs.length / 2 - 1) :=
    List.dropLast_take (by omega)
  have e2 : (ss.take (cs.length / 2)).getLast?.getD 0 = ss[cs.length / 2 - 1] := by
    rw [List.getLast?_take]
    simp [show cs.length / 2 ≠ 0 by omega, List.getElem?_eq_getElem hs]
  rw [e1, e2]
  exact ⟨⟨COK_lt (fun a b n hp => WFb_lt hp) hk1, by simp; omega, by simp; omega, hk1⟩,
         ⟨COK_lt (fun a b n hp => WFb_lt hp) hk2, by simp; omega, by simp; omega, hk2⟩⟩

/-! ## `Node::combine` -/

/-- Post-condition of `combine`: the merged node, or the two re-balanced ones, are full non-root nodes
    spanning the pair's bounds, holding exactly the pair's entries in order. -/
def CombPost (c : Cfg) (h : Nat) (a b : Node) (m1 m2 : Nat) : Combined → Prop
  | .one m => WFb c false h m m1 m2 ∧ toList h m = toList h a ++ toList h b
  | .two l s r => WFb c false h l m1 s ∧ WFb c false h r s m2 ∧ toList h l ++ toList h r = toList h a ++ toList h b

/-- **`Node::combine` (`node.rs:369`)** on an underflowed node and a full sibling (either order). -/
theorem combine_spec (c : Cfg) : ∀ (h : Nat) (a b : Node) (m1 sep m2 : Nat),
    Weak c h a m1 sep → Weak c h b sep m2 → (WFb c false h a m1 sep ∨ WFb c false h b sep m2) →
    CombPost c h a b m1 m2 (combine c a sep b)
  | 0, .leaf p, .leaf q, m1, sep, m2, ha, hb, hfull => by
    obtain ⟨h1, sp, bp, lp, _⟩ := ha
    obtain ⟨h2, sq, bq, lq, _⟩ := hb
    have hL := c.hL
    have srt : Sorted (p ++ q) := sorted_append.2 ⟨sp, sq, fun x hx y hy => by
      have := (bp x hx).2; have := (bq y hy).1; omega⟩
    have bnd : ∀ e ∈ p ++ q, m1 ≤ e ∧ e < m2 := by
      intro e he; rcases List.mem_append.1 he with he | he
      · have := bp e he; omega
      · have := bq e he; omega
    have ne : p ++ q ≠ [] := by
      rcases hfull with hf | hf
      · have := hf.2.2.2.2 rfl; simp [this]
      · have := hf.2.2.2.2 rfl; simp [this]
    have hlen : (p ++ q).length ≤ 2 * c.L := by simp; omega
    simp only [combine]
    generalize hps : p ++ q = ps at srt bnd ne hlen
    by_cases hfit : ps.length ≤ c.L
    · simp only [hfit, ↓reduceIte, CombPost, toList]
      exact ⟨⟨by omega, srt, bnd, hfit, fun _ => ne⟩, hps.symm⟩
    · simp only [hfit, ↓reduceIte, CombPost, toList]
      have hmid : ps.length / 2 < ps.length := by omega
      obtain ⟨hc1, hc2⟩ := sorted_cut ps srt _ hmid
      have hsp := List.take_append_drop (ps.length / 2) ps
      have hs' := srt; rw [← hsp] at hs'
      obtain ⟨st, sd, _⟩ := sorted_append.1 hs'
      have hne1 : ps.take (ps.length / 2) ≠ [] := by
        intro h0; have := congrArg List.length h0; rw [List.length_take, List.length_nil] at this; omega
      have hne2 : ps.drop (ps.length / 2) ≠ [] := by
        intro h0; have := congrArg List.length h0; rw [List.length_drop, List.length_nil] at this; omega
      obtain ⟨el, hel⟩ := List.exists_mem_of_ne_nil _ hne1
      obtain ⟨er, her⟩ := List.exists_mem_of_ne_nil _ hne2
      have bl := fun e (he : e ∈ ps.take (ps.length / 2)) => bnd e (List.mem_of_mem_take he)
      have br := fun e (he : e ∈ ps.drop (ps.length / 2)) => bnd e (List.mem_of_mem_drop he)
      refine ⟨⟨?_, st, fun e he => ⟨(bl e he).1, hc1 e he⟩, by simp; omega, fun _ => hne1⟩,
              ⟨?_, sd, fun e he => ⟨hc2 e he, (br e he).2⟩, by simp; omega, fun _ => hne2⟩, hsp.trans hps.symm⟩
      · have := (bl el hel).1; have := hc1 el hel; omega
      · have := hc2 er her; have := (br er her).2; omega
  | h + 1, .branch s1 c1, .branch s2 c2, m1, sep, m2, ha, hb, hfull => by
    obtain ⟨h1, l1, u1, k1⟩ := ha
    obtain ⟨h2, l2, u2, k2⟩ := hb
    have hk := (COK_append (COK_len k1)).2 ⟨k1, k2⟩
    have hge : 3 ≤ (c1 ++ c2).length := by
      rcases hfull with hf | hf
      · have := hf.2.1; simp; omega
      · have := hf.2.1; simp; omega
    have hle : (c1 ++ c2).length ≤ 2 * c.B := by simp; omega
    have hflat : ((c1 ++ c2).map (toList h)).flatten = toList (h + 1) (.branch s1 c1) ++ toList (h + 1) (.branch s2 c2) := by
      simp [toList, List.map_append, List.flatten_append]
    simp only [combine]
    generalize c1 ++ c2 = cs at hk hge hle hflat
    generalize s1 ++ sep :: s2 = ss at hk
    by_cases hfit : cs.length ≤ c.B
    · simp only [hfit, ↓reduceIte, CombPost]
      exact ⟨⟨by omega, by omega, hfit, hk⟩, by simpa [toList] using hflat⟩
    · simp only [hfit, ↓reduceIte, CombPost]
      obtain ⟨w1, w2⟩ := branch_split c h hk (by omega) hle
      refine ⟨w1, w2, ?_⟩
      rw [← hflat]
      simp only [toList]
      rw [← List.flatten_append, ← List.map_append, List.take_append_drop]
  | 0, .branch _ _, _, _, _, _, ha, _, _ => by simp [Weak] at ha
  | _ + 1, .leaf _, _, _, _, _, ha, _, _ => by simp [Weak] at ha
  | 0, .leaf _, .branch _ _, _, _, _, _, hb, _ => by simp [Weak] at hb
  | _ + 1, .branch _ _, .leaf _, _, _, _, _, hb, _ => by simp [Weak] at hb

/-! ## `Branch::rebalance` -/

theorem flat_window (f : Node → List E) (cs : List Node) (i : Nat) (hi : i + 1 < cs.length) (a b : Node) :
    ((cs.take i ++ [a, b] ++ cs.drop (i + 2)).map f).flatten =
      ((cs.take i).map f).flatten ++ f a ++ f b ++ ((cs.drop (i + 2)).map f).flatten := by
  simp [List.map_append, List.flatten_append, List.append_assoc]

theorem setAt_window {α} (cs : List α) (i : Nat) (hi : i + 1 < cs.length) (x d : α) :
    setAt cs i x = cs.take i ++ [x, cs.getD (i + 1) d] ++ cs.drop (i + 2) := by
  rw [setAt, List.drop_eq_getElem_cons hi, getD_eq_getElem _ _ _ hi]
  simp

theorem setAt_window_last {α} (cs : List α) (j : Nat) (hi : j + 1 < cs.length) (x d : α) :
    setAt cs (j + 1) x = cs.take j ++ [cs.getD j d, x] ++ cs.drop (j + 2) := by
  have ht : cs.take (j + 1) = cs.take j ++ [cs[j]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem (by omega : j < cs.length)]; rfl
  rw [setAt, ht, getD_eq_getElem _ _ _ (by omega)]
  simp only [List.append_assoc, List.singleton_append, List.cons_append]
  rfl

theorem flat_setAt (f : Node → List E) (cs : List Node) (i : Nat) (hi : i + 1 < cs.length) (x : Node) :
    ((setAt cs i x).map f).flatten =
      ((cs.take i).map f).flatten ++ f x ++ f (cs.getD (i + 1) default) ++ ((cs.drop (i + 2)).map f).flatten := by
  rw [setAt_window cs i hi x default, flat_window f cs i hi]

theorem flat_setAt_last (f : Node → List E) (cs : List Node) (j : Nat) (hi : j + 1 < cs.length) (x : Node) :
    ((setAt cs (j + 1) x).map f).flatten =
      ((cs.take j).map f).flatten ++ f (cs.getD j default) ++ f x ++ ((cs.drop (j + 2)).map f).flatten := by
  rw [setAt_window_last cs j hi x default, flat_window f cs j hi]

/-- **`Branch::rebalance` (`node.rs:204`)**: after child `i` came back weak (`c'`), combining it with a
    sibling yields a chain of full children over the same bounds and the same entries, one child
    shorter (merge) or the same length (borrow). -/
theorem rebalance_spec (c : Cfg) (h : Nat) {lo hi : Nat} {seps : List E} {cs : List Node} (i : Nat) (c' : Node)
    (hk : COK (fun a b n => WFb c false h n a b) lo seps cs hi) (h2 : 2 ≤ cs.length) (hi' : i < cs.length)
    (hc' : ∀ a b, WFb c false h (cs.getD i default) a b → Weak c h c' a b) :
    COK (fun a b n => WFb c false h n a b) lo (rebalance c seps (setAt cs i c') i).1
      (rebalance c seps (setAt cs i c') i).2 hi ∧
    ((rebalance c seps (setAt cs i c') i).2.map (toList h)).flatten = ((setAt cs i c').map (toList h)).flatten ∧
    cs.length ≤ (rebalance c seps (setAt cs i c') i).2.length + 1 ∧
    (rebalance c seps (setAt cs i c') i).2.length ≤ cs.length := by
  have hlen := length_setAt cs i c' hi'
  have hsl := COK_len hk
  unfold rebalance
  rw [hlen]
  simp only [show ¬ cs.length < 2 by omega, ↓reduceIte]
  by_cases hlast : i + 1 < cs.length
  · simp only [hlast, ↓reduceIte]
    rw [getD_setAt_self _ _ _ _ hi', getD_setAt_ne _ _ _ _ _ hi' (by omega)]
    obtain ⟨m1, m2, w1, w2, hctx⟩ := COK_window i hk hlast
    have spec := combine_spec c h c' (cs.getD (i + 1) default) m1 (seps.getD i 0) m2
      (hc' _ _ w1) (WFb_weak w2) (Or.inr w2)
    have hflat := flat_setAt (toList h) cs i hlast c'
    generalize combine c c' (seps.getD i 0) (cs.getD (i + 1) default) = res at spec
    match res, spec with
    | .one m, ⟨wm, tm⟩ =>
      simp only
      rw [delAt_setAt_succ _ _ _ (by omega), take_setAt_of_le _ _ _ _ (Nat.le_refl _) hi',
        drop_setAt_of_lt _ _ _ _ (by omega) hi']
      have hd : delAt seps i = seps.take i ++ [] ++ seps.drop (i + 1) := by simp [delAt]
      rw [hd]
      refine ⟨hctx [] [m] wm, ?_, by simp; omega, by simp; omega⟩
      rw [hflat]; simp [List.map_append, List.flatten_append, tm, List.append_assoc]
    | .two l s r, ⟨wl, wr, tlr⟩ =>
      simp only
      rw [setAt_setAt_succ _ _ _ _ (by omega), take_setAt_of_le _ _ _ _ (Nat.le_refl _) hi',
        drop_setAt_of_lt _ _ _ _ (by omega) hi', setAt_eq]
      refine ⟨hctx [s] [l, r] ⟨wl, wr⟩, ?_, by simp; omega, by simp; omega⟩
      rw [hflat, flat_window _ _ _ hlast]
      simp only [List.append_assoc] at tlr ⊢
      rw [← List.append_assoc (toList h l) (toList h r), tlr]
      simp only [List.append_assoc]
  · simp only [hlast, ↓reduceIte]
    obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
    simp only [Nat.add_sub_cancel]
    rw [getD_setAt_self _ _ _ _ hi', getD_setAt_ne _ _ _ _ _ hi' (by omega)]
    obtain ⟨m1, m2, w1, w2, hctx⟩ := COK_window j hk (by omega)
    have spec := combine_spec c h (cs.getD j default) c' m1 (seps.getD j 0) m2
      (WFb_weak w1) (hc' _ _ w2) (Or.inl w1)
    have hflat := flat_setAt_last (toList h) cs j (by omega) c'
    generalize combine c (cs.getD j default) (seps.getD j 0) c' = res at spec
    match res, spec with
    | .one m, ⟨wm, tm⟩ =>
      simp only
      rw [delAt_setAt_succ _ _ _ (by rw [hlen]; omega), take_setAt_of_le _ _ _ _ (by omega) hi',
        drop_setAt_of_lt _ _ _ _ (by omega) hi']
      have hd : delAt seps j = seps.take j ++ [] ++ seps.drop (j + 1) := by simp [delAt]
      rw [hd]
      refine ⟨hctx [] [m] wm, ?_, by simp; omega, by simp; omega⟩
      rw [hflat]; simp [List.map_append, List.flatten_append, tm, List.append_assoc]
    | .two l s r, ⟨wl, wr, tlr⟩ =>
      simp only
      rw [setAt_setAt_succ _ _ _ _ (by rw [hlen]; omega), take_setAt_of_le _ _ _ _ (by omega) hi',
        drop_setAt_of_lt _ _ _ _ (by omega) hi', setAt_eq]
      refine ⟨hctx [s] [l, r] ⟨wl, wr⟩, ?_, by simp; omega, by simp; omega⟩
      rw [hflat, flat_window _ _ _ (by omega)]
      simp only [List.append_assoc] at tlr ⊢
      rw [← List.append_assoc (toList h l) (toList h r), tlr]
      simp only [List.append_assoc]

/-! ## `Node::remove_one` -/

/-- **Shape of `Node::remove_one` (`node.rs:456`)**: the returned node is weak, and full whenever the
    underflow flag is clear. Needs `BRANCH_MAX >= 4` (see the module doc and `Bugs.lean`). -/
theorem removeOne_shape (c : Cfg) (hB4 : 4 ≤ c.B) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi x : Nat) (n' : Node) (u : Bool),
    WFb c r h n lo hi → removeOne c h n x = some (n', u) →
    Weak c h n' lo hi ∧ (u = false → WFb c r h n' lo hi)
  | 0, r, .leaf es, lo, hi, x, n', u, hw, hres => by
    obtain ⟨hlh, hs, hb, hl, hne⟩ := hw
    have spec := leafRemove_spec c es x hs
    simp only [removeOne] at hres
    generalize leafRemove c es x = res at spec hres
    match res, spec, hres with
    | some (es', u'), ⟨s', hmem, hlen, hu⟩, hres =>
      simp only [Option.some.injEq, Prod.mk.injEq] at hres
      obtain ⟨rfl, rfl⟩ := hres
      have hb' : ∀ e ∈ es', lo ≤ e ∧ e < hi := fun e he => hb e ((hmem e).1 he).1
      refine ⟨⟨hlh, s', hb', by omega, by simp⟩, fun hu0 => ⟨hlh, s', hb', by omega, fun _ => ?_⟩⟩
      have hL := c.hL
      rw [hu0] at hu
      have : c.L / 2 ≤ es'.length := by simpa using hu
      intro h0; rw [h0] at this; simp at this; omega
  | 0, _, .branch _ _, _, _, _, _, _, hw, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, _, _, hw, _ => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, lo, hi, x, n', u, hw, hres => by
    obtain ⟨hlh, hc2, hcB, hk⟩ := hw
    have hsl := COK_len hk
    simp only [removeOne] at hres
    have hi' : childIndex seps x < cs.length := by have := childIndex_le seps x; omega
    generalize childIndex seps x = i at hres hi'
    generalize hres' : removeOne c h (cs.getD i default) x = res at hres
    match res, hres with
    | some (c', u'), hres =>
      simp only [Option.some.injEq, Prod.mk.injEq] at hres
      obtain ⟨rfl, rfl⟩ := hres
      have ih : ∀ a b, WFb c false h (cs.getD i default) a b →
          Weak c h c' a b ∧ (u' = false → WFb c false h c' a b) :=
        fun a b hw' => removeOne_shape c hB4 h false _ a b x c' u' hw' hres'
      cases u'
      · -- no underflow below: the child is full again, the branch keeps its fan-out
        obtain ⟨a, b, hp, hctx⟩ := COK_focus i hk hi'
        have hk' := hctx [] [c'] ((ih a b hp).2 rfl)
        simp only [List.append_nil, List.take_append_drop] at hk'
        rw [← setAt_eq] at hk'
        have hl' := length_setAt cs i c' hi'
        simp only [Bool.false_eq_true, ↓reduceIte]
        refine ⟨⟨hlh, by omega, by omega, hk'⟩, fun _ => ⟨hlh, by omega, by omega, hk'⟩⟩
      · -- the child underflowed: `rebalance` repairs it with a sibling
        obtain ⟨hk', _, hl1, hl2⟩ := rebalance_spec c h i c' hk hc2 hi' (fun a b hw' => (ih a b hw').1)
        simp only [↓reduceIte]
        refine ⟨⟨hlh, by omega, by omega, hk'⟩, fun hu => ⟨hlh, ?_, by omega, hk'⟩⟩
        have : ¬ (rebalance c seps (setAt cs i c') i).2.length < c.B / 2 := by simpa using hu
        omega

/-- Membership post-condition of `remove_one` (for an `x` within the node's bounds). -/
def RemMem (h : Nat) (n : Node) (x : E) : Option (Node × Bool) → Prop
  | none => x ∉ toList h n
  | some (n', _) => ∀ e, e ∈ toList h n' ↔ e ∈ toList h n ∧ e ≠ x

/-- **Content of `Node::remove_one`**: absent ⇒ `None`; present ⇒ exactly that entry disappears. -/
theorem removeOne_mem (c : Cfg) (hB4 : 4 ≤ c.B) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi x : Nat),
    WFb c r h n lo hi → lo ≤ x → x < hi → RemMem h n x (removeOne c h n x)
  | 0, r, .leaf es, lo, hi, x, hw, _, _ => by
    have spec := leafRemove_spec c es x hw.2.1
    simp only [removeOne]
    generalize leafRemove c es x = res at spec
    match res, spec with
    | none, spec => exact spec
    | some (es', u'), ⟨_, hmem, _, _⟩ => exact hmem
  | 0, _, .branch _ _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, lo, hi, x, hw, hlo, hhi => by
    obtain ⟨hlh, hc2, hcB, hk⟩ := hw
    obtain ⟨hi', a, b, hax, hxb, hp, _⟩ := COK_route x hk hlo hhi
    have hPb : ∀ a b n, WFb c false h n a b → a < b ∧ Sorted (toList h n) ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩
    obtain ⟨out1, out2⟩ := COK_route_outside (toList h) hPb x hk
    have ih := removeOne_mem c hB4 h false _ a b x hp hax hxb
    have hsp := split_at cs (childIndex seps x) hi'
    simp only [removeOne]
    rw [getD_eq_getElem _ _ _ hi']
    generalize childIndex seps x = i at hi' hp ih hsp out1 out2 ⊢
    generalize hres' : removeOne c h cs[i] x = res at ih ⊢
    match res, ih with
    | none, ih =>
      simp only [RemMem, toList]
      rw [hsp]
      simp only [List.map_append, List.map_cons, List.flatten_append, List.flatten_cons, List.mem_append]
      rintro (h1 | h1 | h1)
      · have := out1 x h1; omega
      · exact ih h1
      · have := out2 x h1; omega
    | some (c', u'), ih =>
      -- the new branch's entries are those of `setAt cs i c'`, with or without a rebalance
      have hflat : ((if u' = true then rebalance c seps (setAt cs i c') i else (seps, setAt cs i c')).2.map
          (toList h)).flatten = ((setAt cs i c').map (toList h)).flatten := by
        cases u'
        · rfl
        · simp only [↓reduceIte]
          have hsh : ∀ a b, WFb c false h (cs.getD i default) a b → Weak c h c' a b := by
            intro a b hw'
            rw [getD_eq_getElem _ _ _ hi'] at hw'
            exact (removeOne_shape c hB4 h false _ a b x c' true hw' hres').1
          exact (rebalance_spec c h i c' hk hc2 hi' hsh).2.1
      simp only [RemMem, toList]
      rw [hflat, setAt_eq]
      intro e
      conv => rhs; rw [hsp]
      simp only [List.map_append, List.map_cons, List.map_nil, List.flatten_append, List.flatten_cons,
        List.flatten_nil, List.append_nil, List.mem_append]
      rw [ih e]
      constructor
      · rintro ((h1 | h1) | h1)
        · exact ⟨Or.inl h1, by have := out1 e h1; omega⟩
        · exact ⟨Or.inr (Or.inl h1.1), h1.2⟩
        · exact ⟨Or.inr (Or.inr h1), by have := out2 e h1; omega⟩
      · rintro ⟨h1 | h1 | h1, hne⟩
        · exact Or.inl (Or.inl h1)
        · exact Or.inl (Or.inr ⟨h1, hne⟩)
        · exact Or.inr h1

/-! ## `CowBTree::remove` -/

/-- The root-collapse loop (`mod.rs:250-257`) turns a weak root into a well-formed root. -/
theorem collapse_spec (c : Cfg) : ∀ (h : Nat) (n : Node) (lo hi : Nat), Weak c h n lo hi →
    WFb c true (collapse h n).2 (collapse h n).1 lo hi ∧ toList (collapse h n).2 (collapse h n).1 = toList h n
  | 0, .leaf es, lo, hi, hw => by simp only [collapse]; exact ⟨hw, trivial⟩
  | h + 1, .branch seps [ch], lo, hi, hw => by
    obtain ⟨_, _, _, hk⟩ := hw
    match seps, hk with
    | [], hk =>
      simp only [collapse]
      have := collapse_spec c h ch lo hi (WFb_weak hk)
      refine ⟨this.1, ?_⟩
      rw [this.2]; simp [toList]
    | _ :: ss, hk =>
      obtain ⟨_, hk2⟩ := hk
      match ss, hk2 with
      | [], hk2 => simp [COK] at hk2
      | _ :: _, hk2 => simp [COK] at hk2
  | h + 1, .branch seps [], lo, hi, hw => by simp [Weak] at hw
  | h + 1, .branch seps (a :: b :: cs), lo, hi, hw => by
    obtain ⟨h1, _, h3, hk⟩ := hw
    simp only [collapse]
    exact ⟨⟨h1, by simp, h3, hk⟩, trivial⟩
  | 0, .branch _ _, _, _, hw => by simp [Weak] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [Weak] at hw

/-- **`CowBTree::remove` (`mod.rs:243`)**, for `BRANCH_MAX >= 4`: a well-formed tree stays well-formed
    (losing root levels as needed) and its entries become exactly the reference `List.erase`. -/
theorem remove_spec (c : Cfg) (hB4 : 4 ≤ c.B) (h : Nat) (t : Node) (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    TreeWF c (remove c h t x).2 (remove c h t x).1 ∧
    toList (remove c h t x).2 (remove c h t x).1 = (toList h t).erase x := by
  have hmem := removeOne_mem c hB4 h true t 0 HI x hw (Nat.zero_le _) hx
  have hs := TreeWF_sorted hw
  unfold remove
  generalize hres : removeOne c h t x = res at hmem ⊢
  match res, hmem with
  | none, hmem =>
    exact ⟨hw, (List.erase_of_not_mem hmem).symm⟩
  | some (n', u), hmem =>
    have hsh := (removeOne_shape c hB4 h true t 0 HI x n' u hw hres).1
    obtain ⟨hw', ht⟩ := collapse_spec c h n' 0 HI hsh
    refine ⟨hw', ?_⟩
    rw [ht]
    exact sorted_ext (Weak_sb hsh).1 (sorted_erase x _ hs) (fun e => by
      rw [hmem e, mem_erase_sorted x _ hs e])

end CowBTree
