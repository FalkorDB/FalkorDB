import FalkorCowBTree.Remove

/-!
# The range cursor yields exactly the entries whose key is in `[lo, hi]`, in order

* `rest cur` — everything the cursor can still reach; `out cur` — its prefix with keys `<= hi`.
* `next_spec` — each `Iterator::next` stops exactly when `out` is empty, or yields `out`'s head.
* `seek_rest` — `RangeIter::new`'s descent lands on `(toList root).dropWhile (keyOf · < lo)`.
* `range_spec` — `range(lo, hi)` collects exactly the reference filter.
-/

namespace CowBTree

def restStack (stk : List Frame) : List E := (stk.map (fun f => (f.2.map (toList f.1)).flatten)).flatten

def rest (cur : Cursor) : List E :=
  match cur.leaf with
  | none => []
  | some es => es.drop cur.pos ++ restStack cur.stack

def keyLe (hi : Nat) (e : E) : Bool := decide (keyOf e ≤ hi)

def out (cur : Cursor) : List E := (rest cur).takeWhile (keyLe cur.hi)

def Inv (cur : Cursor) : Prop :=
  match cur.leaf with
  | none => True
  | some es => Sorted (es.drop cur.pos ++ restStack cur.stack) ∧
      (cur.whole = true → ∀ e ∈ es.drop cur.pos, keyOf e ≤ cur.hi)

theorem restStack_cons (h : Nat) (cs : List Node) (stk : List Frame) :
    restStack ((h, cs) :: stk) = (cs.map (toList h)).flatten ++ restStack stk := by
  simp [restStack]

theorem descendLeft_rest : ∀ (h : Nat) (n : Node) (stk : List Frame),
    (descendLeft h n stk).2 ++ restStack (descendLeft h n stk).1 = toList h n ++ restStack stk
  | 0, .leaf es, stk => by simp [descendLeft, toList]
  | 0, .branch _ _, stk => by simp [descendLeft, toList]
  | _ + 1, .leaf es, stk => by simp [descendLeft, toList]
  | _ + 1, .branch _ [], stk => by simp [descendLeft, toList]
  | h + 1, .branch _ (ch :: cs), stk => by
    simp only [descendLeft]
    rw [descendLeft_rest h ch, restStack_cons]
    simp [toList, List.append_assoc]

theorem advanceLeaf_rest : ∀ (stk : List Frame),
    match advanceLeaf stk with
    | none => restStack stk = []
    | some (stk', es) => es ++ restStack stk' = restStack stk
  | [] => by simp [advanceLeaf, restStack]
  | (h, []) :: stk => by
    have := advanceLeaf_rest stk
    simp only [advanceLeaf]
    rw [restStack_cons]
    simpa using this
  | (h, ch :: cs) :: stk => by
    simp only [advanceLeaf]
    rw [descendLeft_rest h ch, restStack_cons, restStack_cons]
    simp [List.append_assoc]

theorem keyOf_mono {a b : E} (h : a ≤ b) : keyOf a ≤ keyOf b := Nat.div_le_div_right h

theorem sorted_le_last : ∀ {l : List E} {a : E}, Sorted l → l.getLast? = some a → ∀ e ∈ l, e ≤ a
  | [], _, _, _, _, he => by simp at he
  | [x], a, _, h, e, he => by simp at h he; omega
  | x :: y :: l, a, hs, h, e, he => by
    rw [sorted_cons] at hs
    have ih := sorted_le_last hs.2 (by simpa using h)
    rcases List.mem_cons.1 he with he | he
    · subst he; have := hs.1 y List.mem_cons_self; have := ih y List.mem_cons_self; omega
    · exact ih e he

theorem wholeOf_sound (es : List E) (pos hi : Nat) (hs : Sorted (es.drop pos)) (hw : wholeOf es hi = true) :
    ∀ e ∈ es.drop pos, keyOf e ≤ hi := by
  intro e he
  have hlt : pos < es.length := by
    rcases Nat.lt_or_ge pos es.length with h | h
    · exact h
    · rw [List.drop_of_length_le h] at he; simp at he
  unfold wholeOf at hw
  match hl : es.getLast?, hw with
  | none, _ => simp at hl; subst hl; simp at hlt
  | some a, hw =>
    simp only [decide_eq_true_eq] at hw
    have hl' : (es.drop pos).getLast? = some a := by
      rw [List.getLast?_drop]; simp [show ¬ es.length ≤ pos by omega, hl]
    exact Nat.le_trans (keyOf_mono (sorted_le_last hs hl' e he)) hw

def NextPost (cur : Cursor) : Option (E × Cursor) → Prop
  | none => out cur = []
  | some (e, cur') => out cur = e :: out cur' ∧ Inv cur' ∧ cur'.hi = cur.hi

theorem next_none (cur : Cursor) (hl : cur.leaf = none) : cur.next = none := by
  rw [Cursor.next]; split
  · rfl
  · rename_i h; rw [hl] at h; simp at h

theorem next_in (cur : Cursor) (es : List E) (hl : cur.leaf = some es) (hp : cur.pos < es.length) :
    cur.next = if (!cur.whole && decide (keyOf es[cur.pos] > cur.hi)) = true then none
      else some (es[cur.pos], { cur with pos := cur.pos + 1 }) := by
  rw [Cursor.next]; split
  · rename_i h; rw [hl] at h; simp at h
  · rename_i es' h
    rw [hl] at h; simp only [Option.some.injEq] at h; subst h
    simp only [hp, ↓reduceDIte]

theorem next_adv (cur : Cursor) (es : List E) (hl : cur.leaf = some es) (hp : ¬ cur.pos < es.length) :
    cur.next = match advanceLeaf cur.stack with
      | none => none
      | some (stk', es') => Cursor.next { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi } := by
  rw [Cursor.next]; split
  · rename_i h; rw [hl] at h; simp at h
  · rename_i es' h
    rw [hl] at h; simp only [Option.some.injEq] at h; subst h
    simp only [hp, ↓reduceDIte]
    split <;> rename_i h1 <;> split <;> rename_i h2 <;> simp_all

/-- **`Iterator::next` (`cursor.rs:181`)** is a correct generator of `out` (any stack shape). -/
theorem next_spec : ∀ (w : Nat) (cur : Cursor), stackWeight cur.stack = w → Inv cur → NextPost cur cur.next := by
  intro w
  induction w using Nat.strongRecOn with
  | ind w ih =>
  intro cur hwt hinv
  match hl : cur.leaf with
  | none => rw [next_none cur hl]; simp [NextPost, out, rest, hl]
  | some es =>
    by_cases hp : cur.pos < es.length
    · rw [next_in cur es hl hp]
      simp only [Inv, hl] at hinv
      obtain ⟨hs, hwh⟩ := hinv
      by_cases hstop : (!cur.whole && decide (keyOf es[cur.pos] > cur.hi)) = true
      · simp only [hstop, ↓reduceIte, NextPost, out, rest, hl]
        rw [List.drop_eq_getElem_cons hp, List.cons_append, List.takeWhile_cons]
        simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at hstop
        simp [keyLe, show ¬ keyOf es[cur.pos] ≤ cur.hi by omega]
      · simp only [hstop, Bool.false_eq_true, ↓reduceIte, NextPost]
        have hkey : keyOf es[cur.pos] ≤ cur.hi := by
          cases hwc : cur.whole
          · simp [hwc] at hstop; omega
          · exact hwh hwc _ (by rw [List.drop_eq_getElem_cons hp]; exact List.mem_cons_self)
        rw [List.drop_eq_getElem_cons hp] at hs hwh
        refine ⟨?_, ?_, trivial⟩
        · simp only [out, rest, hl]
          rw [List.drop_eq_getElem_cons hp, List.cons_append, List.takeWhile_cons]
          simp [keyLe, hkey]
        · simp only [Inv, hl]
          refine ⟨?_, fun hw e he => hwh hw e (List.mem_cons_of_mem _ he)⟩
          simp only [List.cons_append] at hs; exact (sorted_cons.1 hs).2
    · rw [next_adv cur es hl hp]
      have hr := advanceLeaf_rest cur.stack
      have hd : es.drop cur.pos = [] := List.drop_of_length_le (Nat.le_of_not_lt hp)
      match ha : advanceLeaf cur.stack, hr with
      | none, hr => simp [NextPost, out, rest, hl, hd, hr]
      | some (stk', es'), hr =>
        simp only
        simp only [Inv, hl, hd, List.nil_append] at hinv
        rw [← hr] at hinv
        have hinv' : Inv { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi } := by
          simp only [Inv, List.drop_zero]
          refine ⟨hinv.1, fun hw => ?_⟩
          have := wholeOf_sound es' 0 cur.hi (by simpa using (sorted_append.1 hinv.1).1) hw
          simpa using this
        have hlt := advanceLeaf_weight _ _ _ ha
        have ih' := ih (stackWeight stk') (by omega) { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi } rfl hinv'
        have hout : out cur = out { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi } := by
          simp only [out, rest, hl, hd, List.nil_append, List.drop_zero, hr]
        revert ih'
        generalize Cursor.next _ = res
        match res with
        | none => intro ih'; simpa [NextPost, hout] using ih'
        | some (e, cur') => intro ih'; simpa [NextPost, hout] using ih'

theorem take_spec : ∀ (n : Nat) (cur : Cursor), Inv cur → cur.take n = (out cur).take n
  | 0, cur, _ => by simp [Cursor.take]
  | n + 1, cur, hinv => by
    have hn := next_spec _ cur rfl hinv
    simp only [Cursor.take]
    generalize cur.next = res at hn
    match res, hn with
    | none, hn => simp only [NextPost] at hn; simp [hn]
    | some (e, cur'), ⟨ho, hinv', _⟩ => simp only [ho, take_spec n cur' hinv', List.take_succ_cons]


/-! ## The seek descent -/

theorem dropWhile_all {α} (p : α → Bool) : ∀ (l : List α), (∀ x ∈ l, p x = true) → l.dropWhile p = []
  | [], _ => rfl
  | a :: l, h => by
    rw [List.dropWhile_cons, if_pos (h a List.mem_cons_self)]
    exact dropWhile_all p l (fun x hx => h x (List.mem_cons_of_mem _ hx))

theorem dropWhile_none {α} (p : α → Bool) : ∀ (l : List α), (∀ x ∈ l, p x = false) → l.dropWhile p = l
  | [], _ => rfl
  | a :: l, h => by rw [List.dropWhile_cons, if_neg (by simp [h a List.mem_cons_self])]

theorem dropWhile_route {α} (p : α → Bool) (A M C : List α) (hA : ∀ x ∈ A, p x = true) (hC : ∀ x ∈ C, p x = false) :
    (A ++ M ++ C).dropWhile p = M.dropWhile p ++ C := by
  rw [List.append_assoc, List.dropWhile_append, dropWhile_all p A hA]
  simp only [List.isEmpty_nil, ↓reduceIte]
  rw [List.dropWhile_append]
  split
  · rename_i h; simp at h; rw [h, dropWhile_none p C hC]; simp
  · rfl

def keyLt (lo : Nat) (e : E) : Bool := decide (keyOf e < lo)

/-- **`RangeIter::new`'s descent (`cursor.rs:101-115`)** lands on the first entry with key `>= lo`. -/
theorem seek_rest (c : Cfg) (lo : Nat) : ∀ (h : Nat) (r : Bool) (n : Node) (stk : List Frame) (a b : Nat),
    WFb c r h n a b →
    (seek (enc lo 0) lo h n stk).2.1.drop (seek (enc lo 0) lo h n stk).2.2 ++ restStack (seek (enc lo 0) lo h n stk).1 =
      (toList h n).dropWhile (keyLt lo) ++ restStack stk
  | 0, _, .leaf es, stk, _, _, _ => by
    simp only [seek, toList, lowerBound]
    rw [drop_takeWhile_length]; rfl
  | 0, _, .branch _ _, _, _, _, hw => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, stk, a, b, hw => by
    obtain ⟨_, _, _, hk⟩ := hw
    have hsl := COK_len hk
    have hi' : childIndex seps (enc lo 0) < cs.length := by have := childIndex_le seps (enc lo 0); omega
    have hPb : ∀ a b n, WFb c false h n a b → a < b ∧ Sorted (toList h n) ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩
    obtain ⟨out1, out2⟩ := COK_route_outside (toList h) hPb (enc lo 0) hk
    obtain ⟨a', b', hp, _⟩ := COK_focus (childIndex seps (enc lo 0)) hk hi'
    have hsp := split_at cs (childIndex seps (enc lo 0)) hi'
    simp only [seek]
    rw [seek_rest c lo h false _ _ a' b' hp, restStack_cons]
    generalize childIndex seps (enc lo 0) = i at hi' out1 out2 hp hsp ⊢
    rw [getD_eq_getElem _ _ _ hi']
    simp only [toList]
    conv => rhs; rw [hsp]
    simp only [List.map_append, List.map_cons, List.flatten_append, List.flatten_cons]
    rw [← List.append_assoc ((List.map (toList h) (List.take i cs)).flatten), dropWhile_route]
    · simp [List.append_assoc]
    · intro e he; have := out1 e he; simp only [keyLt, decide_eq_true_eq]; exact (lt_enc_lo_iff e lo).1 this
    · intro e he; have := out2 e he; simp only [keyLt, decide_eq_false_iff_not]
      intro hk'; have := (lt_enc_lo_iff e lo).2 hk'; omega

/-! ## `range` = reference filter -/

theorem filter_hi_only (lo hi : Nat) : ∀ (l : List E), Sorted l → (∀ e ∈ l, lo ≤ keyOf e) →
    l.takeWhile (keyLe hi) = l.filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))
  | [], _, _ => rfl
  | b :: l, hs, hge => by
    have hsb := sorted_cons.1 hs
    rw [List.takeWhile_cons, List.filter_cons]
    have hb := hge b List.mem_cons_self
    by_cases hbh : keyOf b ≤ hi
    · simp only [keyLe, hbh, decide_true, ↓reduceIte, show lo ≤ keyOf b by omega, and_self]
      congr 1
      exact filter_hi_only lo hi l hsb.2 (fun e he => hge e (List.mem_cons_of_mem _ he))
    · simp only [keyLe, hbh, decide_false, Bool.false_eq_true, ↓reduceIte, and_false]
      symm; rw [List.filter_eq_nil_iff]
      intro e he
      have := keyOf_mono (Nat.le_of_lt (hsb.1 e he)); simp; omega

/-- On a sorted list, "skip keys `< lo`, then take keys `<= hi`" is the `[lo, hi]` key filter. -/
theorem dropTake_filter (lo hi : Nat) : ∀ (l : List E), Sorted l →
    (l.dropWhile (keyLt lo)).takeWhile (keyLe hi) = l.filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))
  | [], _ => rfl
  | a :: l, hs => by
    have hs' := sorted_cons.1 hs
    rw [List.dropWhile_cons]
    by_cases hlo : keyOf a < lo
    · simp only [keyLt, hlo, decide_true, ↓reduceIte]
      rw [dropTake_filter lo hi l hs'.2, List.filter_cons]
      simp [show ¬ lo ≤ keyOf a by omega]
    · simp only [keyLt, hlo, decide_false, Bool.false_eq_true, ↓reduceIte]
      apply filter_hi_only lo hi (a :: l) hs
      intro e he
      rcases List.mem_cons.1 he with he | he
      · subst he; omega
      · exact Nat.le_trans (by omega) (keyOf_mono (Nat.le_of_lt (hs'.1 e he)))

theorem new_inv_rest (c : Cfg) (h : Nat) (root : Node) (lo hi : Nat) (hw : TreeWF c h root) :
    Inv (Cursor.new h root lo hi) ∧ rest (Cursor.new h root lo hi) = (toList h root).dropWhile (keyLt lo) ∧
    (Cursor.new h root lo hi).hi = hi := by
  have hr := seek_rest c lo h true root [] 0 HI hw
  have hs := TreeWF_sorted hw
  simp only [restStack, List.map_nil, List.flatten_nil, List.append_nil] at hr
  have hsr : Sorted ((toList h root).dropWhile (keyLt lo)) := Sorted.sublist (List.dropWhile_sublist _) hs
  simp only [Cursor.new, Inv, rest, restStack, List.map_nil, List.flatten_nil, List.append_nil]
  refine ⟨⟨?_, fun hwh => ?_⟩, ?_, trivial⟩
  · rw [hr]; exact hsr
  · apply wholeOf_sound _ _ _ _ hwh
    rw [← hr] at hsr; exact (sorted_append.1 hsr).1
  · exact hr

/-- **`CowBTree::range(lo, hi)` (`mod.rs:297`) collects exactly the entries whose key lies in the
    inclusive range `[lo, hi]`, in `(key, doc)` order**, on every well-formed tree, for all `lo`, `hi`
    (duplicates of a key all appear; `lo > hi` yields nothing). Its docs are `docOf` of these. -/
theorem range_spec (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) :
    range h root lo hi n = (toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi)) := by
  obtain ⟨hinv, hrest, hhi⟩ := new_inv_rest c h root lo hi hw
  unfold range
  rw [take_spec n _ hinv]
  have ho : out (Cursor.new h root lo hi) = (toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi)) := by
    simp only [out, hrest, hhi]
    exact dropTake_filter lo hi _ (TreeWF_sorted hw)
  rw [ho]
  exact List.take_of_length_le (Nat.le_trans (List.length_filter_le _ _) hn)

/-- `CowBTree::point(k)` = `range(k, k)`: exactly the entries with key `k`. -/
theorem point_spec (c : Cfg) (h : Nat) (root : Node) (k n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) :
    range h root k k n = (toList h root).filter (fun e => decide (keyOf e = k)) := by
  rw [range_spec c h root k k n hw hn]
  congr 1; funext e; simp only [decide_eq_decide]; omega

/-- An inverted range (`lo > hi`) is empty. -/
theorem range_empty (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) (hlh : hi < lo) : range h root lo hi n = [] := by
  rw [range_spec c h root lo hi n hw hn, List.filter_eq_nil_iff]
  intro e _; simp; omega

end CowBTree
