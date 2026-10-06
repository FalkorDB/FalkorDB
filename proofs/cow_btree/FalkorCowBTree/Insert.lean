import FalkorCowBTree.Leaf

/-!
# Single insert: `Node::insert_one` / `CowBTree::insert` preserve the invariant and implement
# sorted-set insertion
-/

namespace CowBTree

theorem mem_flat_append (f : Node → List E) (a b : List Node) (e : E) :
    e ∈ ((a ++ b).map f).flatten ↔ e ∈ (a.map f).flatten ∨ e ∈ (b.map f).flatten := by
  simp [List.map_append, List.flatten_append]

theorem mem_flat_cons (f : Node → List E) (n : Node) (b : List Node) (e : E) :
    e ∈ ((n :: b).map f).flatten ↔ e ∈ f n ∨ e ∈ (b.map f).flatten := by
  simp

theorem mem_flat_nil (f : Node → List E) (e : E) : e ∈ (([] : List Node).map f).flatten ↔ False := by
  simp

/-- The post-condition of `insert_one`: no split ⇒ same bounds; split ⇒ two non-root halves around
    the promoted separator. Either way the entries are the old ones plus `x`. -/
def InsPost (c : Cfg) (r : Bool) (h : Nat) (n : Node) (lo hi x : Nat) : Node × Option (E × Node) → Prop
  | (n', none) => WFb c r h n' lo hi ∧ ∀ e, e ∈ toList h n' ↔ e ∈ toList h n ∨ e = x
  | (n', some (s, m)) => WFb c false h n' lo s ∧ WFb c false h m s hi ∧
      ∀ e, e ∈ toList h n' ∨ e ∈ toList h m ↔ e ∈ toList h n ∨ e = x

theorem insertOne_leaf (c : Cfg) (r : Bool) (es : List E) (lo hi x : Nat)
    (hw : WFb c r 0 (.leaf es) lo hi) (hlo : lo ≤ x) (hhi : x < hi) :
    InsPost c r 0 (.leaf es) lo hi x (insertOne c 0 (.leaf es) x) := by
  obtain ⟨hlh, hs, hb, hl, hne⟩ := hw
  have spec := leafInsert_spec c es x hs hl
  simp only [insertOne]
  generalize leafInsert c es x = res at spec ⊢
  have bnd : ∀ e, e ∈ es ∨ e = x → lo ≤ e ∧ e < hi := by
    rintro e (he | rfl)
    · exact hb e he
    · exact ⟨hlo, hhi⟩
  match res, spec with
  | none, spec =>
    refine ⟨⟨hlh, hs, hb, hl, hne⟩, fun e => ?_⟩
    simp only [toList]
    constructor
    · exact Or.inl
    · rintro (h | rfl)
      · exact h
      · exact spec
  | some (.fit ps), ⟨hsrt, hmem, hlen, _⟩ =>
    refine ⟨⟨hlh, hsrt, fun e he => bnd e ((hmem e).1 he), hlen, fun _ => ?_⟩, fun e => by simpa [toList] using hmem e⟩
    intro h0; have := (hmem x).2 (Or.inr rfl); rw [h0] at this; simp at this
  | some (.split l s r'), ⟨sl, sr, hmem, hls, hrs, lne, rne, _, hll, _, hrl⟩ =>
    have bl : ∀ e ∈ l, lo ≤ e ∧ e < hi := fun e he => bnd e ((hmem e).1 (Or.inl he))
    have br : ∀ e ∈ r', lo ≤ e ∧ e < hi := fun e he => bnd e ((hmem e).1 (Or.inr he))
    obtain ⟨el, hel⟩ := List.exists_mem_of_ne_nil l lne
    obtain ⟨er, her⟩ := List.exists_mem_of_ne_nil r' rne
    refine ⟨⟨?_, sl, fun e he => ⟨(bl e he).1, hls e he⟩, hll, fun _ => lne⟩,
            ⟨?_, sr, fun e he => ⟨hrs e he, (br e he).2⟩, hrl, fun _ => rne⟩, fun e => by simpa [toList] using hmem e⟩
    · have := (bl el hel).1; have := hls el hel; omega
    · have := hrs er her; have := (br er her).2; omega

/-- **`Node::insert_one` (`node.rs:223`)** on a well-formed subtree, for an `x` within its bounds. -/
theorem insertOne_spec (c : Cfg) : ∀ (h : Nat) (r : Bool) (n : Node) (lo hi x : Nat),
    WFb c r h n lo hi → lo ≤ x → x < hi → InsPost c r h n lo hi x (insertOne c h n x)
  | 0, r, .leaf es, lo, hi, x, hw, hlo, hhi => insertOne_leaf c r es lo hi x hw hlo hhi
  | 0, _, .branch _ _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | _ + 1, _, .leaf _, _, _, _, hw, _, _ => by simp [WFb] at hw
  | h + 1, r, .branch seps cs, lo, hi, x, hw, hlo, hhi => by
    obtain ⟨hlh, hc2, hcB, hc⟩ := hw
    obtain ⟨hi', a, b, hax, hxb, hp, hctx⟩ := COK_route x hc hlo hhi
    have ih := insertOne_spec c h false cs[childIndex seps x] a b x hp hax hxb
    have hsp := split_at cs (childIndex seps x) hi'
    have hlen := COK_len hc
    simp only [insertOne]
    rw [getD_eq_getElem _ _ _ hi']
    generalize childIndex seps x = i at hi' hp hctx ih hsp ⊢
    generalize insertOne c h cs[i] x = res at ih ⊢
    match res, ih with
    | (c', none), ⟨hw', hmem⟩ =>
      have hk := hctx [] [c'] hw'
      simp only [List.append_nil, List.take_append_drop] at hk
      refine ⟨⟨hlh, by simp [setAt]; omega, by simp [setAt]; omega, by rw [setAt_eq]; exact hk⟩, fun e => ?_⟩
      simp only [toList]
      conv => rhs; rw [hsp]
      rw [setAt_eq]
      simp only [mem_flat_append, mem_flat_cons, List.map_cons, List.map_nil, List.flatten_cons,
        List.flatten_nil, List.append_nil, hmem e, List.mem_append]
      constructor
      · rintro ((h1 | h1 | h1) | h1)
        · exact Or.inl (Or.inl h1)
        · exact Or.inl (Or.inr (Or.inl h1))
        · exact Or.inr h1
        · exact Or.inl (Or.inr (Or.inr h1))
      · rintro ((h1 | h1 | h1) | h1)
        · exact Or.inl (Or.inl h1)
        · exact Or.inl (Or.inr (Or.inl h1))
        · exact Or.inr h1
        · exact Or.inl (Or.inr (Or.inr h1))
    | (c', some (s, m)), ⟨hw1, hw2, hmem⟩ =>
      have hk := hctx [s] [c', m] ⟨hw1, hw2⟩
      simp only
      rw [insAt_setAt _ _ _ _ hi']
      have hseps : seps.take i ++ [s] ++ seps.drop i = insAt seps i s := by simp [insAt]
      rw [hseps] at hk
      have hl' : (cs.take i ++ [c', m] ++ cs.drop (i + 1)).length = cs.length + 1 := by simp; omega
      -- the entries of the new child list
      have hmem' : ∀ e, e ∈ ((cs.take i ++ [c', m] ++ cs.drop (i + 1)).map (toList h)).flatten ↔
          e ∈ toList (h + 1) (.branch seps cs) ∨ e = x := by
        intro e
        simp only [toList]
        conv => rhs; rw [hsp]
        simp only [mem_flat_append, mem_flat_cons, List.map_cons, List.map_nil, List.flatten_cons,
          List.flatten_nil, List.append_nil, List.mem_append]
        have := hmem e
        constructor
        · rintro ((h1 | h1 | h1) | h1)
          · exact Or.inl (Or.inl h1)
          · rcases this.1 (Or.inl h1) with h2 | h2
            · exact Or.inl (Or.inr (Or.inl h2))
            · exact Or.inr h2
          · rcases this.1 (Or.inr h1) with h2 | h2
            · exact Or.inl (Or.inr (Or.inl h2))
            · exact Or.inr h2
          · exact Or.inl (Or.inr (Or.inr h1))
        · rintro ((h1 | h1 | h1) | h1)
          · exact Or.inl (Or.inl h1)
          · rcases this.2 (Or.inl h1) with h2 | h2
            · exact Or.inl (Or.inr (Or.inl h2))
            · exact Or.inl (Or.inr (Or.inr h2))
          · exact Or.inr h1
          · rcases this.2 (Or.inr h1) with h2 | h2
            · exact Or.inl (Or.inr (Or.inl h2))
            · exact Or.inl (Or.inr (Or.inr h2))
      generalize cs.take i ++ [c', m] ++ cs.drop (i + 1) = cs' at hk hl' hmem' ⊢
      generalize insAt seps i s = seps' at hk ⊢
      by_cases hfit : cs'.length ≤ c.B
      · simp only [hfit, ↓reduceIte]
        exact ⟨⟨hlh, by omega, hfit, hk⟩, fun e => by simpa [toList] using hmem' e⟩
      · simp only [hfit, ↓reduceIte]
        have hB := c.hB
        have hm0 : 0 < cs'.length / 2 := by omega
        have hm : cs'.length / 2 < cs'.length := by omega
        obtain ⟨hs, hk1, hk2⟩ := COK_split hk (cs'.length / 2) hm0 hm
        have hsl := COK_len hk
        have e1 : (seps'.take (cs'.length / 2)).dropLast = seps'.take (cs'.length / 2 - 1) :=
          List.dropLast_take (by omega)
        have e2 : (seps'.take (cs'.length / 2)).getLast?.getD 0 = seps'[cs'.length / 2 - 1] := by
          rw [List.getLast?_take]
          simp [show cs'.length / 2 ≠ 0 by omega, List.getElem?_eq_getElem hs]
        rw [e1, e2]
        refine ⟨⟨COK_lt (fun a b n hp => WFb_lt hp) hk1, by simp; omega, by simp; omega, hk1⟩,
                ⟨COK_lt (fun a b n hp => WFb_lt hp) hk2, by simp; omega, by simp; omega, hk2⟩, fun e => ?_⟩
        rw [← hmem' e]
        simp only [toList]
        conv => rhs; rw [← List.take_append_drop (cs'.length / 2) cs']
        rw [mem_flat_append]

/-- **`CowBTree::insert` (`mod.rs:169`)**: a well-formed tree stays well-formed (growing a level on a
    root split) and its entries become exactly the reference sorted-set insertion. -/
theorem insert_spec (c : Cfg) (h : Nat) (t : Node) (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    TreeWF c (insert c h t x).2 (insert c h t x).1 ∧
    toList (insert c h t x).2 (insert c h t x).1 = sins x (toList h t) := by
  have spec := insertOne_spec c h true t 0 HI x hw (Nat.zero_le _) hx
  have hs := TreeWF_sorted hw
  unfold insert
  generalize insertOne c h t x = res at spec ⊢
  match res, spec with
  | (n', none), ⟨hw', hmem⟩ =>
    exact ⟨hw', sorted_ext (WFb_sb hw').1 (sorted_sins x _ hs) (fun e => by rw [hmem e, mem_sins])⟩
  | (n', some (s, m)), ⟨hw1, hw2, hmem⟩ =>
    have hB := c.hB
    have hw' : TreeWF c (h + 1) (.branch [s] [n', m]) :=
      ⟨by unfold HI W; omega, by simp, by simp; omega, ⟨hw1, hw2⟩⟩
    refine ⟨hw', sorted_ext (WFb_sb hw').1 (sorted_sins x _ hs) (fun e => ?_)⟩
    rw [mem_sins, ← hmem e]
    simp [toList]

end CowBTree
