import FalkorCowBTree.Batch

/-!
# #2278 (3597b3a82) read-side and accounting additions

* `insertFlag_spec`, `removeFlag_spec` — `CowBTree::insert` / `remove` now return a `bool`; it is
  exactly "the tuple was absent" / "the tuple was present".
* `firstDoc_spec`, `containsKey_spec`, `firstDoc_eq_point` (headline) — the reference descent of
  `first_doc` (with its nearest-right-subtree fallback) returns the doc of the first entry with the
  key, i.e. the head of the point-lookup filter of the sorted multiset.
* `nextWith_eq`, `rangeWith_spec`, `rangeTuples_spec`, `rangeDocs_spec` — the generic cursor
  (`Extract`, `DocExtract`, `TupleExtract`) yields `make(key, doc)` of exactly the entries in range,
  in order; the key is read lazily but never wrongly.
* `heapBytes_eq`, `heapBytes_split`, `heapBytes_aos` — `heap_bytes` counts every page exactly once;
  its leaf part is the sum of the leaf blob lengths.
-/

namespace CowBTree

/-! ## `insert` / `remove` report whether they changed the tree -/

/-- The `inserted` half of `Node::insert_one`'s `(bool, Option<Split>)` (`node.rs:306-364`):
    `false` iff the leaf found the tuple already present (line 316); a branch forwards its child's. -/
def insertFlag (c : Cfg) : Nat → Node → E → Bool
  | _, .leaf es, x => (leafInsert c es x).isSome
  | 0, .branch _ _, _ => false
  | h + 1, .branch seps cs, x => insertFlag c h (cs.getD (childIndex seps x) default) x

/-- `CowBTree::insert` (`mod.rs:206-221`): the new tree and the returned `inserted`. -/
def insertRet (c : Cfg) (h : Nat) (root : Node) (x : E) : (Node × Nat) × Bool :=
  (insert c h root x, insertFlag c h root x)

/-- `CowBTree::remove` (`mod.rs:243-262`): the new tree and the returned `removed`
    (`self.root.remove_one(..).is_some()`, line 247). -/
def removeRet (c : Cfg) (h : Nat) (root : Node) (x : E) : (Node × Nat) × Bool :=
  (remove c h root x, (removeOne c h root x).isSome)

theorem insertFlag_spec (c : Cfg) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi x : Nat),
    WFb c r h n lo hi → lo ≤ x → x < hi → insertFlag c h n x = decide (x ∉ toList h n)
  | 0, _, .leaf es, _, _, x, hw, _, _ => by
    obtain ⟨_, _, h3, _⟩ := lbe_facts es x hw.2.1
    simp only [insertFlag, leafInsert, toList]
    by_cases hs : es[lowerBoundEntry es x]? = some x
    · simp [hs, h3.1 hs]
    · simp only [hs, ↓reduceIte]
      have : x ∉ es := fun hx => hs (h3.2 hx)
      split <;> simp [this]
  | 0, _, .branch _ _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, lo, hi, x, hw, hlo, hhi => by
    obtain ⟨_, _, _, hk⟩ := hw
    obtain ⟨hi', a, b, hax, hxb, hp, _⟩ := COK_route x hk hlo hhi
    have hPb : ∀ a b n, WFb c false h n a b → a < b ∧ Sorted (toList h n) ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩
    obtain ⟨out1, out2⟩ := COK_route_outside (toList h) hPb x hk
    have hsp := split_at cs (childIndex seps x) hi'
    simp only [insertFlag]
    rw [getD_eq_getElem _ _ _ hi', insertFlag_spec c h false _ a b x hp hax hxb]
    simp only [toList]
    conv => rhs; rw [hsp]
    simp only [List.map_append, List.map_cons, List.flatten_append, List.flatten_cons, List.mem_append,
      decide_eq_decide]
    constructor
    · intro hn; rintro (h1 | h1 | h1)
      · have := out1 x h1; omega
      · exact hn h1
      · have := out2 x h1; omega
    · intro hn h1; exact hn (Or.inr (Or.inl h1))

/-- **`CowBTree::insert` returns `true` iff the tuple was not in the tree** (`mod.rs:206`). -/
theorem insertRet_spec (c : Cfg) (h : Nat) (t : Node) (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    (insertRet c h t x).2 = decide (x ∉ toList h t) ∧
    toList (insertRet c h t x).1.2 (insertRet c h t x).1.1 = sins x (toList h t) :=
  ⟨insertFlag_spec c h true t 0 HI x hw (Nat.zero_le _) hx, (insert_spec c h t x hw hx).2⟩

theorem removeOne_some_mem (c : Cfg) : ∀ (h : Nat) (n : Node) (x : E) (n' : Node) (u : Bool),
    removeOne c h n x = some (n', u) → x ∈ toList h n
  | _, .leaf es, x, n', u, hr => by
    simp only [removeOne] at hr
    simp only [toList]
    by_cases hs : es[lowerBoundEntry es x]? = some x
    · exact List.mem_of_getElem? hs
    · simp [leafRemove, hs] at hr
  | 0, .branch _ _, _, _, _, hr => by simp [removeOne] at hr
  | h + 1, .branch seps cs, x, n', u, hr => by
    simp only [removeOne] at hr
    generalize hres : removeOne c h (cs.getD (childIndex seps x) default) x = res at hr
    match res, hr with
    | some (c', u'), _ =>
      by_cases hi : childIndex seps x < cs.length
      · have hx := removeOne_some_mem c h _ x c' u' hres
        rw [getD_eq_getElem _ _ _ hi] at hx
        simp only [toList]
        exact List.mem_flatten.2 ⟨_, List.mem_map_of_mem (List.getElem_mem hi), hx⟩
      · have hd : removeOne c h (default : Node) x = none := by cases h <;> rfl
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega), Option.getD_none, hd] at hres
        cases hres

/-- **`CowBTree::remove` returns `true` iff the tuple was in the tree** (`mod.rs:243`), for
    `BRANCH_MAX >= 4` (the range in which `remove` is correct at all; see `Bugs.lean`). -/
theorem removeRet_spec (c : Cfg) (hB4 : 4 ≤ c.B) (h : Nat) (t : Node) (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    (removeRet c h t x).2 = decide (x ∈ toList h t) ∧
    toList (removeRet c h t x).1.2 (removeRet c h t x).1.1 = (toList h t).erase x := by
  refine ⟨?_, (remove_spec c hB4 h t x hw hx).2⟩
  have hmem := removeOne_mem c hB4 h true t 0 HI x hw (Nat.zero_le _) hx
  simp only [removeRet]
  generalize hres : removeOne c h t x = res at hmem
  match res, hmem with
  | none, hmem => simp only [RemMem] at hmem; simp [hmem]
  | some (n', u), _ => simp [removeOne_some_mem c h t x n' u hres]

/-! ## The under-full flag is `Node::is_underfull` -/

theorem isUnderfull_leaf (c : Cfg) (es : List E) : isUnderfull c (.leaf es) = decide (es.length < c.L / 2) := rfl

theorem isUnderfull_branch (c : Cfg) (seps : List E) (cs : List Node) :
    isUnderfull c (.branch seps cs) = branchUnderfull c cs := rfl

/-- Since #2278 `Node::remove_one` reports `Some(branch.is_underfull())` (`node.rs:473`), and a leaf's
    `new_count < LEAF_MAX / 2`: in both arms the flag is exactly `Node::is_underfull` of the node it
    hands back. -/
theorem removeOne_flag (c : Cfg) : ∀ (h : Nat) (n : Node) (x : E) (n' : Node) (u : Bool),
    removeOne c h n x = some (n', u) → u = isUnderfull c n'
  | _, .leaf es, x, n', u, hr => by
    simp only [removeOne, leafRemove] at hr
    by_cases hs : es[lowerBoundEntry es x]? = some x
    · simp only [hs, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hr
      obtain ⟨rfl, rfl⟩ := hr
      have hlt : lowerBoundEntry es x < es.length := by
        rcases Nat.lt_or_ge (lowerBoundEntry es x) es.length with h1 | h1
        · exact h1
        · rw [List.getElem?_eq_none h1] at hs; simp at hs
      simp only [isUnderfull, length_delAt _ _ hlt]
    · simp [hs] at hr
  | 0, .branch _ _, _, _, _, hr => by simp [removeOne] at hr
  | h + 1, .branch seps cs, x, n', u, hr => by
    simp only [removeOne] at hr
    generalize removeOne c h (cs.getD (childIndex seps x) default) x = res at hr
    match res, hr with
    | some (c', u'), hr =>
      simp only [Option.some.injEq, Prod.mk.injEq] at hr
      obtain ⟨rfl, rfl⟩ := hr
      rfl

/-! ## `first_doc` / `contains_key` -/

/-- `CowBTree::first_doc`'s loop (`mod.rs:340-387`). `rs` is `right_subtree` with its (ghost) height;
    `minOpt = none` is `Node::min`'s panic on an empty subtree (unreachable on a well-formed tree). -/
def firstDocGo (k : Nat) : Nat → Node → Option (Nat × Node) → Option Nat
  | _, .leaf es, rs =>
    match es[lowerBound es k]? with                                       -- lines 364-365
    | some e =>
      if keyOf e = k then some (docOf e)                                  -- line 366
      else ((rs.bind (fun p => minOpt p.1 p.2)).filter (fun e => decide (keyOf e = k))).map docOf
    | none => ((rs.bind (fun p => minOpt p.1 p.2)).filter (fun e => decide (keyOf e = k))).map docOf  -- 369-372
  | 0, .branch _ _, _ => none
  | h + 1, .branch seps cs, rs =>
    let ci := childIndex seps (enc k 0)                                   -- line 375
    let rs' := if ci + 1 < cs.length then some (h, cs.getD (ci + 1) default) else rs  -- lines 376-380
    firstDocGo k h (cs.getD ci default) rs'                               -- line 381

/-- `CowBTree::first_doc` (`mod.rs:340`). -/
def firstDoc (h : Nat) (root : Node) (k : Nat) : Option Nat := firstDocGo k h root none

/-- `CowBTree::contains_key` (`mod.rs:326`). -/
def containsKey (h : Nat) (root : Node) (k : Nat) : Bool := (firstDoc h root k).isSome

theorem minOpt_head {c : Cfg} : ∀ {r : Bool} {h : Nat} {n : Node} {lo hi : Nat}, WFb c r h n lo hi →
    minOpt h n = (toList h n).head?
  | _, 0, .leaf _, _, _, _ => rfl
  | _, h + 1, .branch seps cs, _, _, hw => by
    obtain ⟨_, h2, _, hk⟩ := hw
    match cs, hk with
    | n :: ns, hk =>
      obtain ⟨b, hb⟩ := COK_head hk
      simp only [minOpt, toList, List.map_cons, List.flatten_cons]
      rw [minOpt_head hb]
      have hne := WFb_ne hb
      cases hl : toList h n with
      | nil => exact absurd hl hne
      | cons _ _ => rfl
  | _, 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _, _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

/-- On a sorted list lying strictly above `(k, 0)`, the first key-`k` entry, if any, is the head. -/
theorem find_above (k : Nat) : ∀ (A : List E), Sorted A → (∀ e ∈ A, enc k 0 < e) →
    A.find? (fun e => decide (keyOf e = k)) = A.head?.filter (fun e => decide (keyOf e = k))
  | [], _, _ => rfl
  | a :: A, hs, hA => by
    by_cases hka : keyOf a = k
    · simp [hka, Option.filter]
    · have hgt : k < keyOf a := by
        have := hA a List.mem_cons_self
        have h1 : ¬ keyOf a < k := fun h1 => by have := (lt_enc_lo_iff a k).2 h1; omega
        omega
      simp only [List.find?_cons, hka, decide_false, List.head?_cons, Option.filter_some]
      simp only [Bool.false_eq_true, ↓reduceIte]
      rw [List.find?_eq_none]
      intro e he
      have := keyOf_mono (Nat.le_of_lt ((sorted_cons.1 hs).1 e he))
      simp; omega

theorem find_prefix_none (k : Nat) (P S : List E) (hP : ∀ e ∈ P, keyOf e ≠ k) :
    (P ++ S).find? (fun e => decide (keyOf e = k)) = S.find? (fun e => decide (keyOf e = k)) := by
  rw [List.find?_append, List.find?_eq_none.2 (fun e he => by have := hP e he; simpa using this)]
  rfl

theorem head_dropWhile_false {α} (p : α → Bool) : ∀ (l : List α) (e : α) (t : List α),
    l.dropWhile p = e :: t → p e = false
  | [], _, _, h => by simp at h
  | a :: l, e, t, h => by
    rw [List.dropWhile_cons] at h
    split at h
    · exact head_dropWhile_false p l e t h
    · next ha => simp at h; rw [← h.1]; simpa using ha

theorem firstDocGo_spec (c : Cfg) (k : Nat) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi : Nat)
    (rs : Option (Nat × Node)) (A : List E), WFb c r h n lo hi → lo ≤ enc k 0 → enc k 0 < hi →
    Sorted (toList h n ++ A) → (∀ e ∈ A, enc k 0 < e) → rs.bind (fun p => minOpt p.1 p.2) = A.head? →
    firstDocGo k h n rs = ((toList h n ++ A).find? (fun e => decide (keyOf e = k))).map docOf
  | 0, _, .leaf es, lo, hi, rs, A, hw, hlo, hhi, hs, hA, hrs => by
    simp only [firstDocGo, toList] at hs ⊢
    have hpre : ∀ e ∈ es.take (lowerBound es k), keyOf e ≠ k := by
      intro e he
      rw [lowerBound, take_takeWhile_length] at he
      have := mem_takeWhile_p he; simp at this; omega
    have hsA : Sorted (es.drop (lowerBound es k) ++ A) :=
      Sorted.sublist ((List.drop_sublist _ _).append (List.Sublist.refl A)) hs
    have hfind : (es ++ A).find? (fun e => decide (keyOf e = k)) =
        (es.drop (lowerBound es k) ++ A).find? (fun e => decide (keyOf e = k)) := by
      conv => lhs; rw [← List.take_append_drop (lowerBound es k) es]
      rw [List.append_assoc, find_prefix_none k _ _ hpre]
    rw [hfind]
    have hAfind : A.find? (fun e => decide (keyOf e = k)) = (rs.bind (fun p => minOpt p.1 p.2)).filter
        (fun e => decide (keyOf e = k)) := by
      rw [hrs]; exact find_above k A (sorted_append.1 hsA).2.1 hA
    cases hpos : es[lowerBound es k]? with
    | none =>
      have hlen : es.length ≤ lowerBound es k := by
        rcases Nat.lt_or_ge (lowerBound es k) es.length with h1 | h1
        · rw [List.getElem?_eq_getElem h1] at hpos; simp at hpos
        · exact h1
      rw [List.drop_of_length_le hlen, List.nil_append, hAfind]
    | some e =>
      have hlt : lowerBound es k < es.length := by
        rcases Nat.lt_or_ge (lowerBound es k) es.length with h1 | h1
        · exact h1
        · rw [List.getElem?_eq_none h1] at hpos; simp at hpos
      have he : es[lowerBound es k] = e := by rw [List.getElem?_eq_getElem hlt] at hpos; simpa using hpos
      have hdc := List.drop_eq_getElem_cons hlt
      rw [he] at hdc
      rw [hdc, List.cons_append, List.find?_cons]
      by_cases hke : keyOf e = k
      · simp [hke]
      · simp only [hke, decide_false, ↓reduceIte]
        -- `e` is the first entry with key `>= k`, and its key is not `k`: so it is `> k`, and so is the rest
        have hge : k ≤ keyOf e := by
          have hd : es.drop (lowerBound es k) = es.dropWhile (fun e => decide (keyOf e < k)) :=
            drop_takeWhile_length _ es
          have := head_dropWhile_false _ es e _ (hd ▸ hdc)
          simp at this; omega
        have hrest : ∀ x ∈ es.drop (lowerBound es k + 1), keyOf x ≠ k := by
          intro x hx
          have hs' : Sorted (es.drop (lowerBound es k)) :=
            Sorted.sublist (List.drop_sublist _ _) (sorted_append.1 hs).1
          rw [hdc] at hs'
          have := keyOf_mono (Nat.le_of_lt ((sorted_cons.1 hs').1 x hx)); omega
        rw [find_prefix_none k _ _ hrest, hAfind]
  | 0, _, .branch _ _, _, _, _, _, hw, _, _, _, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, _, hw, _, _, _, _, _ => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, lo, hi, rs, A, hw, hlo, hhi, hs, hA, hrs => by
    obtain ⟨_, _, _, hk⟩ := hw
    have hsl := COK_len hk
    obtain ⟨hi', a, b, hax, hxb, hp, _⟩ := COK_route (enc k 0) hk hlo hhi
    have hPb : ∀ a b n, WFb c false h n a b → a < b ∧ Sorted (toList h n) ∧ ∀ e ∈ toList h n, a ≤ e ∧ e < b :=
      fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩
    obtain ⟨out1, out2⟩ := COK_route_outside (toList h) hPb (enc k 0) hk
    have hsp := split_at cs (childIndex seps (enc k 0)) hi'
    simp only [firstDocGo]
    generalize childIndex seps (enc k 0) = i at hi' hp out1 out2 hsp
    rw [getD_eq_getElem _ _ _ hi']
    have hflat : toList (h + 1) (.branch seps cs) ++ A =
        ((cs.take i).map (toList h)).flatten ++ (toList h cs[i] ++ (((cs.drop (i + 1)).map (toList h)).flatten ++ A)) := by
      simp only [toList]; conv => lhs; rw [hsp]
      simp only [List.map_append, List.map_cons, List.flatten_append, List.flatten_cons, List.append_assoc]
    rw [hflat, find_prefix_none k _ _ (fun e he => Nat.ne_of_lt ((lt_enc_lo_iff e k).1 (out1 e he)))]
    rw [hflat] at hs
    have hs2 := (sorted_append.1 hs).2.1
    apply firstDocGo_spec c k h false cs[i] a b _ _ hp hax hxb hs2
    · intro e he
      rcases List.mem_append.1 he with he | he
      · exact out2 e he
      · exact hA e he
    · split
      · next hlt =>
        simp only [Option.bind_some]
        rw [getD_eq_getElem _ _ _ hlt]
        obtain ⟨a', b', hp'⟩ := COK_mem hk cs[i + 1] (List.getElem_mem hlt)
        rw [minOpt_head hp', List.drop_eq_getElem_cons hlt, List.map_cons, List.flatten_cons, List.append_assoc]
        have hne := WFb_ne hp'
        cases hl : toList h cs[i + 1] with
        | nil => exact absurd hl hne
        | cons _ _ => rfl
      · next hlt =>
        rw [List.drop_of_length_le (by omega)]
        simpa using hrs

/-- **`CowBTree::first_doc` (`mod.rs:340`)**: the doc of the first entry with key `k` in `(key, doc)`
    order (the smallest doc under `k`), or `None` — on every well-formed tree. -/
theorem firstDoc_spec (c : Cfg) (h : Nat) (root : Node) (k : Nat) (hw : TreeWF c h root) (hk : k < W) :
    firstDoc h root k = ((toList h root).find? (fun e => decide (keyOf e = k))).map docOf := by
  have := firstDocGo_spec c k h true root 0 HI none [] hw (Nat.zero_le _)
    (by simp only [enc, HI, W] at hk ⊢; omega) (by simpa using TreeWF_sorted hw) (by simp) rfl
  simpa [firstDoc] using this

theorem find?_eq_head?_filter {α} (p : α → Bool) : ∀ (l : List α), l.find? p = (l.filter p).head?
  | [] => rfl
  | a :: l => by
    by_cases hp : p a = true
    · simp [List.find?_cons, List.filter_cons, hp]
    · simp only [List.find?_cons, List.filter_cons, hp, Bool.false_eq_true, ↓reduceIte]
      exact find?_eq_head?_filter p l

/-- **Point lookup = filter of the sorted multiset**: `first_doc(k)` is the doc of the head of the
    point query's result `range(k, k)` (= `point(k)`, `point_spec`). -/
theorem firstDoc_eq_point (c : Cfg) (h : Nat) (root : Node) (k n : Nat) (hw : TreeWF c h root) (hk : k < W)
    (hn : (toList h root).length ≤ n) :
    firstDoc h root k = (range h root k k n).head?.map docOf := by
  rw [firstDoc_spec c h root k hw hk, point_spec c h root k n hw hn, find?_eq_head?_filter]

/-- **`CowBTree::contains_key` (`mod.rs:326`)**: some entry has key `k`. -/
theorem containsKey_spec (c : Cfg) (h : Nat) (root : Node) (k : Nat) (hw : TreeWF c h root) (hk : k < W) :
    containsKey h root k = (toList h root).any (fun e => decide (keyOf e = k)) := by
  unfold containsKey
  rw [firstDoc_spec c h root k hw hk, Option.isSome_map]
  cases hf : (toList h root).find? (fun e => decide (keyOf e = k)) with
  | none => rw [List.find?_eq_none] at hf; simp only [Option.isSome_none]; symm; simpa using hf
  | some e =>
    have := List.find?_some hf
    simp only [Option.isSome_some]; symm
    exact List.any_eq_true.2 ⟨e, List.mem_of_find?_eq_some hf, this⟩

end CowBTree
