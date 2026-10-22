import AlgoUdf.Dfs
/-! # `enumerate_paths` soundness: invariant of the frame machine (see `AlgoUdf.Dfs`) -/
namespace AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

/-! ## Invariant -/

/-- The frame stack is a weighted walk from the source. -/
def Chain (g : G) (src : Nat) : List Frame → Prop
  | [] => False
  | [b] => b.node = src ∧ b.pw = 0 ∧ b.pc = 0 ∧ b.rel = none
  | fr :: b :: rest =>
    (∃ r c, fr.rel = some r ∧ Edge g b.node fr.node r c ∧ fr.pw = b.pw + c ∧
       fr.pc = b.pc + g.cost r.2.2) ∧ Chain g src (b :: rest)

/-- Pending successors of each frame are successors of its node; a frame only
has pending work if its depth is `< maxLen`. -/
def LvOk (g : G) (maxLen : Nat) : List Frame → Prop
  | [] => True
  | fr :: below =>
    (∀ L, fr.lv = some L → (∀ p ∈ L, p ∈ succs g fr.node) ∧ (L = [] ∨ below.length < maxLen)) ∧
    LvOk g maxLen below

/-- A reported path. -/
def Good (g : G) (cfg : Cfg) (f : FP) : Prop :=
  ∃ v ns, Walk g cfg.src f.es v f.w ∧ NWalk g cfg.src f.es ns v ∧ (cfg.src :: ns).Nodup ∧
    cfg.tgt.all (· == v) = true ∧ f.c = costSum g f.es ∧ 1 ≤ f.es.length ∧
    f.es.length ≤ cfg.maxLen ∧ cfg.maxCost.all (f.c ≤ ·) = true

structure Inv (g : G) (cfg : Cfg) (st : DS) : Prop where
  chain : Chain g cfg.src st.frames
  lvok : LvOk g cfg.maxLen st.frames
  nodup : (st.frames.map (·.node)).Nodup
  onPath : ∀ x, st.onPath x = true ↔ x ∈ st.frames.map (·.node)
  depth : st.frames.length ≤ cfg.maxLen + 1
  costOk : ∀ fr ∈ st.frames, cfg.maxCost.all (fr.pc ≤ ·) = true
  results : ∀ f ∈ st.results, Good g cfg f

theorem chain_ne_nil {g : G} {src : Nat} {fs : List Frame} (h : Chain g src fs) : fs ≠ [] := by
  intro he; subst he; exact h

theorem chain_tail {g : G} {src : Nat} {fr : Frame} {below : List Frame}
    (h : Chain g src (fr :: below)) (hb : below ≠ []) : Chain g src below := by
  cases below with
  | nil => exact absurd rfl hb
  | cons b rest => exact h.2

/-- The live path spelled by the frames. -/
theorem chain_walk {g : G} {src : Nat} :
    ∀ (fr : Frame) (below : List Frame), Chain g src (fr :: below) →
      Walk g src (pathOf (fr :: below)) fr.node fr.pw ∧
      NWalk g src (pathOf (fr :: below)) ((fr :: below).map (·.node)).reverse.tail fr.node ∧
      fr.pc = costSum g (pathOf (fr :: below)) ∧
      ((fr :: below).map (·.node)).getLast? = some src ∧
      (pathOf (fr :: below)).length = below.length := by
  intro fr below
  induction below generalizing fr with
  | nil =>
    intro h
    obtain ⟨h1, h2, h3, h4⟩ := h
    simp only [pathOf, List.filterMap_cons, h4, List.filterMap_nil, List.reverse_nil, List.map_cons,
      List.map_nil, List.reverse_cons, List.nil_append, List.tail_cons]
    rw [h1, h2, h3]; exact ⟨.nil _, .nil _, rfl, rfl, rfl⟩
  | cons b rest ih =>
    intro h
    obtain ⟨⟨r, c, hr, he, hw, hc⟩, hrest⟩ := h
    obtain ⟨w1, n1, c1, l1, len1⟩ := ih b hrest
    have hp : pathOf (fr :: b :: rest) = pathOf (b :: rest) ++ [r] := by
      simp [pathOf, List.filterMap_cons, hr]
    rw [hp]
    refine ⟨?_, ?_, ?_, ?_, by simp [len1]⟩
    · rw [hw]; exact w1.snoc he
    · have : ((fr :: b :: rest).map (·.node)).reverse.tail =
          ((b :: rest).map (·.node)).reverse.tail ++ [fr.node] := by
        simp only [List.map_cons, List.reverse_cons]
        have hne : ((rest.map (·.node)).reverse ++ [b.node]) ≠ [] := by simp
        rw [List.tail_append_of_ne_nil hne]
      rw [this]; exact n1.snoc ⟨he.1, he.2.1⟩
    · rw [hc, c1]; simp [costSum]
    · simp only [List.map_cons] at l1 ⊢
      rw [List.getLast?_cons_cons]; exact l1

theorem inv_init (g : G) (cfg : Cfg) (b : Option Nat) : Inv g cfg (init g cfg b) := by
  refine ⟨⟨rfl, rfl, rfl, rfl⟩, ⟨?_, trivial⟩, by simp [init], ?_, by simp [init], ?_, by simp [init]⟩
  · intro L hL
    simp only [init] at hL
    split at hL
    · rename_i h; cases hL
      refine ⟨fun p hp => ?_, Or.inr (by simpa using h)⟩
      simpa [succsRev] using hp
    · cases hL
  · intro x; simp only [init, List.map_cons, List.map_nil, List.mem_singleton]
    by_cases h : x = cfg.src
    · subst h; simp
    · simp [upd_ne _ _ _ _ h, h]
  · intro fr hfr; simp [init] at hfr; subst hfr; cases cfg.maxCost <;> simp

theorem nodes_sub {fs : List Frame} {x : Nat} (h : x ∈ fs.reverse.tail.map (·.node)) :
    x ∈ fs.map (·.node) := by
  rw [List.mem_map] at h ⊢
  obtain ⟨a, ha, rfl⟩ := h
  exact ⟨a, List.mem_reverse.mp (List.mem_of_mem_tail ha), rfl⟩

theorem nodup_rev {l : List Nat} (h : l.Nodup) : l.reverse.Nodup :=
  List.pairwise_reverse.mpr (h.imp (fun h e => h e.symm))

theorem nodup_last {l : List Nat} {src : Nat} (hnd : l.Nodup) (hl : l.getLast? = some src) :
    (src :: l.reverse.tail).Nodup := by
  have h1 := nodup_rev hnd
  rw [List.getLast?_eq_head?_reverse] at hl
  cases h : l.reverse with
  | nil => rw [h] at hl; cases hl
  | cons a t => rw [h] at hl h1; simp at hl; subst hl; simpa using h1

/-- Pushing an accepted successor keeps the invariant. -/
theorem inv_push {g : G} {cfg : Cfg} {st : DS} {fr : Frame} {below : List Frame} {r : Rel}
    {next : Nat} {rest : List (Rel × Nat)} {x : Nat} (I : Inv g cfg st)
    (hfs : st.frames = fr :: below) (hlvEq : fr.lv = some ((r, next) :: rest))
    (hnotOn : st.onPath next = false) (hwt : g.wt r.2.2 = some x)
    (hmc : cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = false)
    {lv : Option (List (Rel × Nat))}
    (hlvN : ∀ L, lv = some L → (∀ p ∈ L, p ∈ succs g next) ∧ (L = [] ∨ (below.length + 1) < cfg.maxLen))
    {res : List FP} {bnd : Option Nat} {lg : List FP}
    (hres : ∀ f ∈ res, f ∈ st.results ∨
      (f = ⟨pathOf ({ fr with lv := some rest } :: below) ++ [r], fr.pw + x, fr.pc + g.cost r.2.2⟩ ∧
        cfg.tgt.all (· == next) = true)) :
    Inv g cfg ⟨⟨lv, next, fr.pw + x, fr.pc + g.cost r.2.2, some r⟩ :: { fr with lv := some rest } :: below,
      upd st.onPath next true, res, bnd, lg⟩ := by
  have hch := I.chain; rw [hfs] at hch
  have hlv := I.lvok; rw [hfs] at hlv
  have hnd := I.nodup; rw [hfs] at hnd
  have hsucc : (r, next) ∈ succs g fr.node := (hlv.1 _ hlvEq).1 _ (by simp)
  have hadj : Adj g fr.node next r := (mem_succs g fr.node r next).mp hsucc
  have hedge : Edge g fr.node next r x := ⟨hadj.1, hadj.2, hwt⟩
  have hnotin : next ∉ (fr :: below).map (·.node) := by
    intro hm; rw [← hfs, ← I.onPath] at hm; rw [hm] at hnotOn; cases hnotOn
  have hch' : Chain g cfg.src ({ fr with lv := some rest } :: below) := by
    cases below with
    | nil => exact hch
    | cons b rest' => exact hch
  have hchN : Chain g cfg.src
      (⟨lv, next, fr.pw + x, fr.pc + g.cost r.2.2, some r⟩ :: { fr with lv := some rest } :: below) :=
    ⟨⟨r, x, rfl, hedge, rfl, rfl⟩, hch'⟩
  have hcost : cfg.maxCost.all (fun m => fr.pc + g.cost r.2.2 ≤ m) = true := by
    cases hm : cfg.maxCost with
    | none => rfl
    | some m => rw [hm] at hmc; simp at hmc; simp; omega
  have hnd' : ((⟨lv, next, fr.pw + x, fr.pc + g.cost r.2.2, some r⟩ :: { fr with lv := some rest } :: below :
      List Frame).map (·.node)).Nodup := by
    simp only [List.map_cons, List.nodup_cons] at hnd ⊢
    exact ⟨by simpa using hnotin, hnd⟩
  refine ⟨hchN, ⟨hlvN, ?_⟩, hnd', ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, hlv.2⟩
    intro L hL; cases hL
    obtain ⟨h1, h2⟩ := hlv.1 _ hlvEq
    refine ⟨fun p hp => h1 p (List.mem_cons_of_mem _ hp), ?_⟩
    rcases h2 with h2 | h2
    · cases h2
    · exact Or.inr h2
  · intro y
    simp only [List.map_cons, List.mem_cons]
    by_cases hy : y = next
    · subst hy; simp
    · rw [upd_ne _ _ _ _ hy, I.onPath y, hfs]; simp [hy]
  · have hdep : below.length < cfg.maxLen := by
      rcases (hlv.1 _ hlvEq).2 with h2 | h2
      · cases h2
      · exact h2
    simp; omega
  · intro f hf
    simp only [List.mem_cons] at hf
    rcases hf with rfl | rfl | hf
    · exact hcost
    · exact I.costOk fr (by rw [hfs]; simp)
    · exact I.costOk f (by rw [hfs]; simp [hf])
  · intro f hf
    rcases hres f hf with hf | ⟨rfl, hat⟩
    · exact I.results f hf
    · obtain ⟨w1, n1, c1, l1, len1⟩ := chain_walk _ _ hchN
      have hdep : below.length < cfg.maxLen := by
        rcases (hlv.1 _ hlvEq).2 with h2 | h2
        · cases h2
        · exact h2
      refine ⟨next, ((⟨lv, next, fr.pw + x, fr.pc + g.cost r.2.2, some r⟩ :: { fr with lv := some rest } ::
          below : List Frame).map (·.node)).reverse.tail, ?_, ?_, ?_, hat, ?_, ?_, ?_, hcost⟩
      · simpa [pathOf] using w1
      · simpa [pathOf] using n1
      · exact nodup_last hnd' l1
      · simpa [pathOf] using c1
      · simp [pathOf] at len1 ⊢
      · have := len1; simp [pathOf] at this ⊢; omega

/-- One loop iteration keeps the invariant. -/
theorem inv_step {g : G} {cfg : Cfg} {st st' : DS} (I : Inv g cfg st)
    (h : step g cfg st = some st') : Inv g cfg st' := by
  unfold step at h
  split at h
  · cases h
  · rename_i fr below hfs
    have hch := I.chain; rw [hfs] at hch
    have hlv := I.lvok; rw [hfs] at hlv
    have hnd := I.nodup; rw [hfs] at hnd
    try dsimp only at h
    split at h
    · rename_i r next rest hlvEq
      -- facts common to every "skip this successor" outcome
      have skipInv : Inv g cfg { st with frames := { fr with lv := some rest } :: below } := by
        refine ⟨?_, ?_, ?_, ?_, ?_, ?_, I.results⟩
        · cases below with
          | nil => exact hch
          | cons b rest' => exact hch
        · refine ⟨?_, hlv.2⟩
          intro L hL; cases hL
          obtain ⟨h1, h2⟩ := hlv.1 _ hlvEq
          refine ⟨fun p hp => h1 p (List.mem_cons_of_mem _ hp), ?_⟩
          rcases h2 with h2 | h2
          · cases h2
          · exact Or.inr h2
        · simpa using hnd
        · intro x; rw [I.onPath x, hfs]; simp
        · have := I.depth; rw [hfs] at this; simpa using this
        · intro f hf
          simp only [List.mem_cons] at hf
          rcases hf with rfl | hf
          · exact I.costOk fr (by rw [hfs]; simp)
          · exact I.costOk f (by rw [hfs]; simp [hf])
      split at h
      · cases h; exact skipInv
      · rename_i hnotOn
        split at h
        · cases h; exact skipInv
        · rename_i x hwt
          split at h
          · cases h; exact skipInv
          · rename_i hb
            split at h
            · cases h; exact skipInv
            · rename_i hmc
              cases h
              apply inv_push I hfs hlvEq (by simpa using hnotOn) hwt (by simpa using hmc)
              · intro L hL
                split at hL
                · rename_i hexp; cases hL
                  refine ⟨fun p hp => by simpa [succsRev] using hp, Or.inr ?_⟩
                  simp only [Bool.and_eq_true, decide_eq_true_eq] at hexp
                  simpa using hexp.2
                · cases hL
              · intro f hf
                split at hf
                · rename_i hat
                  rcases recordB_mem _ _ _ _ f hf with rfl | hf
                  · exact Or.inr ⟨rfl, hat⟩
                  · exact Or.inl hf
                · exact Or.inl hf
    · -- backtrack
      split at h
      · cases h
      · rename_i hne
        cases h
        refine ⟨chain_tail hch hne, hlv.2, ?_, ?_, ?_, ?_, I.results⟩
        · simp only [List.map_cons, List.nodup_cons] at hnd; exact hnd.2
        · intro y
          simp only [List.map_cons, List.nodup_cons] at hnd
          by_cases hy : y = fr.node
          · subst hy; simp
            intro z hz he; exact hnd.1 (List.mem_map.mpr ⟨z, hz, he⟩)
          · show upd st.onPath fr.node false y = true ↔ y ∈ below.map (·.node)
            rw [upd_ne _ _ _ _ hy, I.onPath y, hfs]; simp [hy]
        · have := I.depth; rw [hfs] at this; simp at this ⊢; omega
        · intro f hf; exact I.costOk f (by rw [hfs]; simp [hf])

theorem inv_star {g : G} {cfg : Cfg} {b : Option Nat} {st : DS} (h : Star g cfg (init g cfg b) st) :
    Inv g cfg st := by
  generalize hi : init g cfg b = s0 at h
  induction h with
  | refl => subst hi; exact inv_init g cfg b
  | tail _ hs ih => exact inv_step ih hs

/-- **Soundness of `enumerate_paths`**: every kept result, at every point of the
search (in particular when the loop breaks), is a reportable path. -/
theorem dfs_sound (g : G) (cfg : Cfg) (b : Option Nat) (st : DS)
    (h : Star g cfg (init g cfg b) st) : ∀ f ∈ st.results, Good g cfg f :=
  (inv_star h).results

/-- **No panic**: the frame stack is never empty, so `levels[depth]`,
`edges.pop().expect(..)` and `nodes.pop().expect(..)` never fail, and the live
path never exceeds `maxLen` hops. -/
theorem dfs_no_panic (g : G) (cfg : Cfg) (b : Option Nat) (st : DS)
    (h : Star g cfg (init g cfg b) st) :
    st.frames ≠ [] ∧ st.frames.length ≤ cfg.maxLen + 1 :=
  ⟨chain_ne_nil (inv_star h).chain, (inv_star h).depth⟩

end AlgoUdf.Dfs
