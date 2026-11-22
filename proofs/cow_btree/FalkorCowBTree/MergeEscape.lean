import FalkorCowBTree.BatchWF

/-!
# `merge_underfull`'s "unchanged borrow" escape (#3738, `node.rs`) never fires on the `u64` page

#3738 made the loop record the pair's fills before a step
(`before = (children[left].fill(), children[left + 1].fill())`, `node.rs`) and, when a borrow
(`produced == 2`) leaves both fills as they were, move past the pair (`left + 1` / `left + 2`)
instead of re-examining `left`. For the `StrLeaf` page a single oversized
string page can make such a borrow; `muLoop` models the old rule (always `left`).

Proven here (`muStep_never_unchanged`): on the modelled page — a leaf splits at `count / 2`
entries (`Page::combine`, `leaf/mod.rs`), a branch at `children.len() / 2` — a step on an
under-full child is never "unchanged": a merge shrinks the list, and a borrow of an under-full node
always changes a fill, also after `merge_underfull_below` (`fill_mubF_le`). So
`match (unchanged, i == left)` always takes `(false, _) => left`, and `muLoop` / `mergeUnderfull`
are exactly the new loop at `CowBTree = PageTree<Leaf<..>>`; every `BatchWF` theorem stands.
-/

namespace CowBTree

/-- `Node::fill`: a leaf's `count()`, a branch's `children.len()`. -/
def fill : Node → Nat
  | .leaf es => es.length
  | .branch _ cs => cs.length

/-- `merge_underfull_below` never grows a node's fill. -/
theorem fill_mubF_le (c : Cfg) : ∀ (h : Nat) (n : Node), fill ((mubF c h).1 n) ≤ fill n
  | 0, n => Nat.le_refl _
  | h + 1, n => by
    show fill (mubBody c h (mubF c h).1 (mubF c h).2 n) ≤ fill n
    cases n with
    | leaf es => simp [mubBody]
    | branch seps cs =>
      simp only [mubBody]
      split
      · simp only [fill]
        exact (muLoop_size c h (mubF c h).1 (mubF c h).2 seps cs 0).2
      · exact Nat.le_refl _

/-- A borrow (`combine` = `.two`) of a pair with an under-full member cannot hand back two nodes
    at least as full as the pair was: the split point is half the total, which an under-full node is
    below. -/
theorem combine_two_moves (c : Cfg) (a b : Node) (sep : E) (l : Node) (s : E) (r : Node)
    (hc : combine c a sep b = .two l s r) (hu : isUnderfull c a = true ∨ isUnderfull c b = true) :
    ¬ (fill a ≤ fill l ∧ fill b ≤ fill r) := by
  have hL := c.hL
  have hB := c.hB
  unfold combine at hc
  split at hc
  · rename_i p q
    dsimp only at hc
    split at hc
    · cases hc
    · rename_i hn
      cases hc
      simp only [fill, List.length_take, List.length_drop, List.length_append]
      simp only [isUnderfull, List.length_append, decide_eq_true_eq] at hu hn
      omega
  · rename_i s1 c1 s2 c2
    dsimp only at hc
    split at hc
    · cases hc
    · rename_i hn
      cases hc
      simp only [fill, List.length_take, List.length_drop, List.length_append]
      simp only [isUnderfull, List.length_append, decide_eq_true_eq] at hu hn
      omega
  · cases hc

/-- **A step of `merge_underfull` on an under-full child is never the "unchanged" borrow of
    `node.rs`**: either the children shrank (`produced == 1`), or a fill at `left` /
    `left + 1` differs from before the step. Hence the new index is always `left`, as `muLoop` has it. -/
theorem muStep_never_unchanged (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) (i : Nat)
    (h1 : 1 < cs.length) (hi : i < cs.length) (hu : isUnderfull c (cs.getD i default) = true) :
    let li := leftOf cs.length i
    let r := (muStep c (mubF c h).1 seps cs i).2
    ¬ (r.length = cs.length ∧
        (fill (r.getD li default), fill (r.getD (li + 1) default)) =
          (fill (cs.getD li default), fill (cs.getD (li + 1) default))) := by
  intro li r
  obtain ⟨hl, hlr⟩ := leftOf_spec cs.length i h1 hi
  have hpair : isUnderfull c (cs.getD li default) = true ∨ isUnderfull c (cs.getD (li + 1) default) = true := by
    rcases hlr with he | ⟨he, _⟩
    · left; show isUnderfull c (cs.getD (leftOf cs.length i) default) = true; rw [he]; exact hu
    · right; show isUnderfull c (cs.getD (leftOf cs.length i + 1) default) = true; rw [he]; exact hu
  rintro ⟨hlen, hfill⟩
  rcases muStep_window c (mubF c h).1 seps cs i h1 hi with ⟨m, _, hst⟩ | ⟨l, s, rr, hc, hst⟩
  · have : r = cs.take li ++ [(mubF c h).1 m] ++ cs.drop (li + 2) := by
      show (muStep c (mubF c h).1 seps cs i).2 = _; rw [hst]
    rw [this] at hlen
    simp only [List.length_append, List.length_take, List.length_drop, List.length_singleton] at hlen
    omega
  · have hr : r = cs.take li ++ [(mubF c h).1 l, (mubF c h).1 rr] ++ cs.drop (li + 2) := by
      show (muStep c (mubF c h).1 seps cs i).2 = _; rw [hst]
    have htl : (cs.take li).length = li := by simp; omega
    have g1 : r.getD li default = (mubF c h).1 l := by
      rw [hr, List.getD_eq_getElem?_getD, List.append_assoc, List.getElem?_append_right (by omega), htl]
      simp
    have g2 : r.getD (li + 1) default = (mubF c h).1 rr := by
      rw [hr, List.getD_eq_getElem?_getD, List.append_assoc, List.getElem?_append_right (by omega), htl]
      simp
    rw [g1, g2, Prod.mk.injEq] at hfill
    have k1 := fill_mubF_le c h l
    have k2 := fill_mubF_le c h rr
    have hc' : combine c (cs.getD li default) (seps.getD li 0) (cs.getD (li + 1) default) = .two l s rr := hc
    exact combine_two_moves c _ _ _ l s rr hc' hpair ⟨by omega, by omega⟩

/-- The new loop's next index on the modelled page is `left`, the index `muLoop`
    continues at. -/
theorem merge_underfull_next_index (c : Cfg) (h : Nat) (seps : List E) (cs : List Node) (i : Nat)
    (h1 : 1 < cs.length) (hi : i < cs.length) (hu : isUnderfull c (cs.getD i default) = true) :
    let li := leftOf cs.length i
    let r := (muStep c (mubF c h).1 seps cs i).2
    let unchanged := decide (r.length = cs.length ∧
        (fill (r.getD li default), fill (r.getD (li + 1) default)) =
          (fill (cs.getD li default), fill (cs.getD (li + 1) default)))
    (match unchanged, decide (i = li) with
      | false, _ => li
      | true, true => li + 1
      | true, false => li + 2) = li := by
  intro li r unchanged
  have : unchanged = false := by
    simp only [unchanged, decide_eq_false_iff_not]
    exact muStep_never_unchanged c h seps cs i h1 hi hu
  rw [this]

end CowBTree
