import FalkorCowBTree.Basic

/-!
# The B⁺-tree invariant

`WFb c r h n lo hi` — node `n` is a well-formed subtree of uniform height `h` whose entries all lie in
`[lo, hi)`:

* a leaf is sorted, holds `<= LEAF_MAX` entries, and is non-empty unless it is the root (`r = true`);
* a branch has `2 ..= BRANCH_MAX` children, and its separators form a chain
  `lo = b₀ < b₁ = seps[0] < … < b_k = hi` with child `i` well-formed within `[b_i, b_{i+1})`
  (`COK`). That is the Rust doc's "`max(left) < sep <= min(right)`" (`tests.rs:1229`), stated with
  explicit bounds so it survives the stale separators a remove leaves behind.

`lo`/`hi` range over `Nat`; the root is checked against `[0, 2^128)`, i.e. all `(u64, u64)` pairs.
-/

namespace CowBTree

/-- The separator chain: child `i` satisfies `P` within `[b_i, b_{i+1})`, `b = lo :: seps ++ [hi]`. -/
def COK (P : Nat → Nat → Node → Prop) : Nat → List E → List Node → Nat → Prop
  | lo, [], [n], hi => P lo hi n
  | lo, s :: ss, n :: ns, hi => P lo s n ∧ COK P s ss ns hi
  | _, _, _, _ => False

/-- The structural invariant (see the module doc). -/
def WFb (c : Cfg) (r : Bool) : Nat → Node → Nat → Nat → Prop
  | 0, .leaf es, lo, hi =>
    lo < hi ∧ Sorted es ∧ (∀ e ∈ es, lo ≤ e ∧ e < hi) ∧ es.length ≤ c.L ∧ (r = false → es ≠ [])
  | h + 1, .branch seps cs, lo, hi =>
    lo < hi ∧ 2 ≤ cs.length ∧ cs.length ≤ c.B ∧ COK (fun a b n => WFb c false h n a b) lo seps cs hi
  | _, _, _, _ => False

/-- Entries are `(u64, u64)` pairs: the root's bounds. -/
def HI : Nat := W * W

/-- A whole tree: root invariant within all `(u64, u64)` pairs. -/
def TreeWF (c : Cfg) (h : Nat) (root : Node) : Prop := WFb c true h root 0 HI

/-! ## `COK` toolkit -/

section COK
variable {P Q : Nat → Nat → Node → Prop}

theorem COK_len : ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → ns.length = ss.length + 1
  | _, _, [], [_], _ => rfl
  | _, _, [], [], h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h => by simp [COK] at h
  | _, _, _ :: _, [], h => by simp [COK] at h
  | _, _, _ :: _, _ :: _, h => by
    simp only [COK] at h
    simp [COK_len h.2]

theorem COK_mono (hPQ : ∀ a b n, P a b n → Q a b n) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → COK Q lo ss ns hi
  | _, _, [], [_], h => hPQ _ _ _ h
  | _, _, [], [], h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h => by simp [COK] at h
  | _, _, _ :: _, [], h => by simp [COK] at h
  | _, _, _ :: _, _ :: _, h => by
    simp only [COK] at h ⊢
    exact ⟨hPQ _ _ _ h.1, COK_mono hPQ h.2⟩

theorem COK_append : ∀ {lo hi m : Nat} {s1 s2 : List E} {c1 c2 : List Node},
    c1.length = s1.length + 1 →
    (COK P lo (s1 ++ m :: s2) (c1 ++ c2) hi ↔ COK P lo s1 c1 m ∧ COK P m s2 c2 hi)
  | _, _, _, [], _, [n], _, _ => by simp [COK]
  | _, _, _, [], _, [], _, h => by simp at h
  | _, _, _, [], _, _ :: _ :: _, _, h => by simp at h
  | _, _, _, _ :: _, _, [], _, h => by simp at h
  | _, _, _, t :: ts, _, n :: c1, _, h => by
    simp only [List.cons_append, COK]
    rw [COK_append (by simpa using h)]
    exact and_assoc.symm

theorem COK_lt (hP : ∀ a b n, P a b n → a < b) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi → lo < hi
  | _, _, [], [_], h => hP _ _ _ h
  | _, _, [], [], h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h => by simp [COK] at h
  | _, _, _ :: _, [], h => by simp [COK] at h
  | _, _, _ :: _, _ :: _, h => by
    simp only [COK] at h
    exact Nat.lt_trans (hP _ _ _ h.1) (COK_lt hP h.2)

/-- Separator chains give a sorted, bounded concatenation of the children. -/
theorem COK_sb (f : Node → List E) (hP : ∀ a b n, P a b n → a < b ∧ Sorted (f n) ∧ ∀ e ∈ f n, a ≤ e ∧ e < b) :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node}, COK P lo ss ns hi →
      Sorted (ns.map f).flatten ∧ ∀ e ∈ (ns.map f).flatten, lo ≤ e ∧ e < hi
  | _, _, [], [_], h => by
    have := hP _ _ _ h
    simpa using ⟨this.2.1, this.2.2⟩
  | _, _, [], [], h => by simp [COK] at h
  | _, _, [], _ :: _ :: _, h => by simp [COK] at h
  | _, _, _ :: _, [], h => by simp [COK] at h
  | lo, hi, s :: ss, n :: ns, h => by
    simp only [COK] at h
    have h1 := hP _ _ _ h.1
    have h2 := COK_sb f hP h.2
    have hlt : s < hi := COK_lt (fun a b n hp => (hP a b n hp).1) h.2
    simp only [List.map_cons, List.flatten_cons]
    refine ⟨sorted_append.2 ⟨h1.2.1, h2.1, ?_⟩, ?_⟩
    · intro a ha b hb
      have := (h1.2.2 a ha).2; have := (h2.2 b hb).1; omega
    · intro e he
      rcases List.mem_append.1 he with he | he
      · have := h1.2.2 e he; omega
      · have := h2.2 e he; have := h1.1; omega

theorem COK_head : ∀ {lo hi : Nat} {ss : List E} {n : Node} {ns : List Node},
    COK P lo ss (n :: ns) hi → ∃ b, P lo b n
  | _, hi, [], _, [], h => ⟨hi, h⟩
  | _, _, [], _, _ :: _, h => by simp [COK] at h
  | _, _, s :: _, _, _, h => ⟨s, h.1⟩

theorem childIndex_nil (x : E) : childIndex [] x = 0 := rfl

theorem childIndex_cons (s : E) (ss : List E) (x : E) :
    childIndex (s :: ss) x = if s ≤ x then childIndex ss x + 1 else 0 := by
  unfold childIndex
  rw [List.takeWhile_cons]
  by_cases h : s ≤ x <;> simp [h]

theorem childIndex_le (ss : List E) (x : E) : childIndex ss x ≤ ss.length := by
  induction ss with
  | nil => simp [childIndex_nil]
  | cons s ss ih => rw [childIndex_cons]; split <;> simp <;> omega

/-- **Routing** (`Branch::child_index`, `node.rs:186`). An `x` within the chain's bounds goes to a
    child whose own bounds contain it; replacing that child by any chain over the same bounds keeps
    the whole chain valid (the one-hole context used by insert and remove). -/
theorem COK_route :
    ∀ {lo hi : Nat} {ss : List E} {ns : List Node} (x : E), COK P lo ss ns hi → lo ≤ x → x < hi →
      ∃ (hi' : childIndex ss x < ns.length) (a b : Nat), a ≤ x ∧ x < b ∧ P a b ns[childIndex ss x] ∧
        ∀ (ts : List E) (ms : List Node), COK P a ts ms b →
          COK P lo (ss.take (childIndex ss x) ++ ts ++ ss.drop (childIndex ss x))
            (ns.take (childIndex ss x) ++ ms ++ ns.drop (childIndex ss x + 1)) hi
  | _, _, [], [n], x, h, hlo, hhi => by
    refine ⟨by simp [childIndex_nil], _, _, hlo, hhi, by simpa [childIndex_nil, COK] using h, ?_⟩
    intro ts ms hm; simpa [childIndex_nil] using hm
  | _, _, [], [], _, h, _, _ => by simp [COK] at h
  | _, _, [], _ :: _ :: _, _, h, _, _ => by simp [COK] at h
  | _, _, _ :: _, [], _, h, _, _ => by simp [COK] at h
  | lo, hi, s :: ss, n :: ns, x, h, hlo, hhi => by
    simp only [COK] at h
    by_cases hsx : s ≤ x
    · obtain ⟨hi', a, b, hax, hxb, hp, hctx⟩ := COK_route x h.2 hsx hhi
      have hci : childIndex (s :: ss) x = childIndex ss x + 1 := by rw [childIndex_cons]; simp [hsx]
      refine ⟨by simp [hci]; omega, a, b, hax, hxb, by simpa [hci] using hp, ?_⟩
      intro ts ms hm
      rw [hci]
      simp only [List.take_succ_cons, List.drop_succ_cons, List.cons_append, COK]
      exact ⟨h.1, hctx ts ms hm⟩
    · have hci : childIndex (s :: ss) x = 0 := by rw [childIndex_cons]; simp [hsx]
      refine ⟨by simp [hci], lo, s, hlo, by omega, by simpa [hci] using h.1, ?_⟩
      intro ts ms hm
      rw [hci]
      simp only [List.take_zero, List.nil_append, List.drop_zero, List.drop_succ_cons]
      exact (COK_append (COK_len hm)).2 ⟨hm, h.2⟩

/-- Cutting a chain at child `m` (used by every branch split: `node.rs:347-350`, `:405-410`). -/
theorem COK_split {lo hi : Nat} {ss : List E} {ns : List Node} (h : COK P lo ss ns hi)
    (m : Nat) (hm0 : 0 < m) (hm : m < ns.length) :
    ∃ (hs : m - 1 < ss.length),
      COK P lo (ss.take (m - 1)) (ns.take m) ss[m - 1] ∧ COK P ss[m - 1] (ss.drop m) (ns.drop m) hi := by
  have hl := COK_len h
  have hs : m - 1 < ss.length := by omega
  refine ⟨hs, ?_⟩
  have e1 : ss = ss.take (m - 1) ++ ss[m - 1] :: ss.drop m := by
    have := split_at ss (m - 1) hs
    rwa [show m - 1 + 1 = m by omega] at this
  have e2 : ns = ns.take m ++ ns.drop m := (List.take_append_drop m ns).symm
  rw [e1, e2] at h
  exact (COK_append (by simp; omega)).1 h

end COK

/-! ## Consequences of the invariant -/

theorem WFb_lt {c : Cfg} {r : Bool} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, WFb c r h n lo hi → lo < hi
  | 0, .leaf _, _, _, hw => hw.1
  | _ + 1, .branch _ _, _, _, hw => hw.1
  | 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

/-- A well-formed subtree's entries are sorted and within its bounds. -/
theorem WFb_sb {c : Cfg} : ∀ {r : Bool} {h : Nat} {n : Node} {lo hi : Nat}, WFb c r h n lo hi →
    Sorted (toList h n) ∧ ∀ e ∈ toList h n, lo ≤ e ∧ e < hi
  | _, 0, .leaf es, _, _, hw => by simp only [toList]; exact ⟨hw.2.1, hw.2.2.1⟩
  | _, h + 1, .branch seps cs, _, _, hw => by
    simp only [toList]
    exact COK_sb (toList h) (fun a b n hp => ⟨WFb_lt hp, WFb_sb hp⟩) hw.2.2.2
  | _, 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _, _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

theorem WFb_weaken {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, WFb c false h n lo hi → WFb c true h n lo hi
  | 0, .leaf _, _, _, hw => ⟨hw.1, hw.2.1, hw.2.2.1, hw.2.2.2.1, by simp⟩
  | _ + 1, .branch _ _, _, _, hw => hw
  | 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

/-- Non-root subtrees are non-empty. -/
theorem WFb_ne {c : Cfg} : ∀ {h : Nat} {n : Node} {lo hi : Nat}, WFb c false h n lo hi → toList h n ≠ []
  | 0, .leaf _, _, _, hw => by simpa [toList] using hw.2.2.2.2 rfl
  | h + 1, .branch seps cs, _, _, hw => by
    obtain ⟨_, h2, _, hc⟩ := hw
    match cs, seps, hc with
    | n :: _, ss, hc =>
      obtain ⟨b, hb⟩ := COK_head hc
      simp [toList, WFb_ne hb]
  | 0, .branch _ _, _, _, hw => by simp [WFb] at hw
  | _ + 1, .leaf _, _, _, hw => by simp [WFb] at hw

theorem WFb_bounds_mono {c : Cfg} : ∀ {r : Bool} {h : Nat} {n : Node} {lo hi lo' hi' : Nat},
    WFb c r h n lo hi → lo' ≤ lo → hi ≤ hi' → lo' < hi' → WFb c r h n lo' hi'
  | _, 0, .leaf _, _, _, _, _, hw, h1, h2, h3 =>
    ⟨h3, hw.2.1, fun e he => by have := hw.2.2.1 e he; omega, hw.2.2.2.1, hw.2.2.2.2⟩
  | _, _ + 1, .branch seps cs, lo, hi, lo', hi', hw, h1, h2, h3 => by
    obtain ⟨_, h2', h3', hc⟩ := hw
    refine ⟨h3, h2', h3', ?_⟩
    clear h2' h3'
    induction seps generalizing lo lo' cs with
    | nil =>
      match cs, hc with
      | [n], hc => exact WFb_bounds_mono hc h1 h2 h3
    | cons s ss ih =>
      match cs, hc with
      | n :: ns, hc =>
        simp only [COK] at hc ⊢
        have := COK_lt (fun a b n hp => WFb_lt hp) hc.2
        exact ⟨WFb_bounds_mono hc.1 h1 (Nat.le_refl _) (by have := WFb_lt hc.1; omega),
          by apply ih <;> first | exact hc.2 | exact Nat.le_refl _ | omega⟩
  | _, 0, .branch _ _, _, _, _, _, hw, _, _, _ => by simp [WFb] at hw
  | _, _ + 1, .leaf _, _, _, _, _, hw, _, _, _ => by simp [WFb] at hw

theorem TreeWF_sorted {c : Cfg} {h : Nat} {root : Node} (hw : TreeWF c h root) : Sorted (toList h root) :=
  (WFb_sb hw).1

end CowBTree
