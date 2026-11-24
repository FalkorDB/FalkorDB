import OpsTraverse.Enum
/-
# `enumerate_paths`: soundness and completeness of the recursive search `Enum.enumRec`

(`Enum.enum_eq_rec` ties it to the iterative loop.) `Valid` is what every call of
`record_found_path` is handed; `enum_complete` says that, for a bound that recording does not
move, exactly the `Cand` paths are handed over.
-/
namespace OpsTraverse.EnumSpec
open OpsTraverse.Search OpsTraverse.Enum

variable {ε F R : Type}

def atT (c : EC ε F) (v : Nat) : Bool := c.target.all (fun t => v == t)

theorem okStep_some (c : EC ε F) (b pw pc : F) (e : ε) (nw nc : F)
    (h : okStep c b pw pc e = some (nw, nc)) :
    nw = c.o.add pw (c.wf e) ∧ nc = c.o.add pc (c.cf e) ∧ c.o.fin nw = true ∧ c.o.gt nw b = false ∧
      c.o.fin nc = true ∧ c.maxCost.all (fun m => !c.o.gt nc m) = true := by
  unfold okStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rename_i hn1 hn2
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Bool.not_eq_true, Bool.not_eq_false] at hn1 hn2
      refine ⟨rfl, rfl, hn1.1, hn1.2, hn2.1, ?_⟩
      cases hm : c.maxCost with
      | none => rfl
      | some m =>
        rw [hm] at hn2
        simp only [Option.any_some, Bool.not_eq_true] at hn2
        simp [hn2.2]

theorem okStep_of (c : EC ε F) (b pw pc : F) (e : ε)
    (h1 : c.o.fin (c.o.add pw (c.wf e)) = true) (h2 : c.o.gt (c.o.add pw (c.wf e)) b = false)
    (h3 : c.o.fin (c.o.add pc (c.cf e)) = true)
    (h4 : c.maxCost.all (fun m => !c.o.gt (c.o.add pc (c.cf e)) m) = true) :
    okStep c b pw pc e = some (c.o.add pw (c.wf e), c.o.add pc (c.cf e)) := by
  unfold okStep
  have e1 : ¬ (c.o.fin (c.o.add pw (c.wf e)) = false || c.o.gt (c.o.add pw (c.wf e)) b) = true := by
    simp [h1, h2]
  have e2 : ¬ (c.o.fin (c.o.add pc (c.cf e)) = false ||
      c.maxCost.any (fun m => c.o.gt (c.o.add pc (c.cf e)) m)) = true := by
    cases hm : c.maxCost with
    | none => simp [h3]
    | some m => rw [hm] at h4; simp at h4; simp [h3, h4]
  rw [if_neg e1, if_neg e2]

/-- The state of the search at a node: the path to it, its prefix weight/cost. -/
structure Ctx (c : EC ε F) (on : List Nat) (pe : List (Nat × ε)) (pw pc : F) (cur : Nat) : Prop where
  walk : WalkTo c.succ c.source pe cur
  on_eq : on = (nodesOf c.source pe).reverse
  nodup : on.Nodup
  w : pw = fsum c.o.add c.o.zero c.wf pe
  cst : pc = fsum c.o.add c.o.zero c.cf pe

theorem ctx_root (c : EC ε F) : Ctx c [c.source] [] c.o.zero c.o.zero c.source where
  walk := rfl
  on_eq := by simp [nodesOf]
  nodup := by simp
  w := rfl
  cst := rfl

theorem ctx_child (c : EC ε F) (on pe pw pc cur) (x : Nat × ε) (nw nc : F) (hc : Ctx c on pe pw pc cur)
    (hx : x ∈ c.succ cur) (hn : x.1 ∉ on) (hnw : nw = c.o.add pw (c.wf x.2)) (hnc : nc = c.o.add pc (c.cf x.2)) :
    Ctx c (x.1 :: on) (pe ++ [x]) nw nc x.1 where
  walk := (walk_append c.succ c.source pe [x] x.1).mpr ⟨cur, hc.walk, by simpa [WalkTo] using hx⟩
  on_eq := by simp [nodesOf, hc.on_eq]
  nodup := List.nodup_cons.mpr ⟨hn, hc.nodup⟩
  w := by rw [hnw, hc.w, fsum_append]; rfl
  cst := by rw [hnc, hc.cst, fsum_append]; rfl

/-- What `record_found_path` is handed: a non-empty simple walk from the source ending where the
search may stop, within `max_len`, whose weight/cost are the left folds over its edges, finite, with
weight not above the bound current at the call and cost within `maxCost`. -/
def Valid (c : EC ε F) (f : Fnd ε F) (b : F) : Prop :=
  ∃ e, WalkTo c.succ c.source f.1 e ∧ f.1 ≠ [] ∧ (nodesOf c.source f.1).Nodup ∧ f.1.length ≤ c.maxLen ∧
    atT c e = true ∧ f.2.1 = fsum c.o.add c.o.zero c.wf f.1 ∧ f.2.2 = fsum c.o.add c.o.zero c.cf f.1 ∧
    c.o.fin f.2.1 = true ∧ c.o.fin f.2.2 = true ∧ c.o.gt f.2.1 b = false ∧
    c.maxCost.all (fun m => !c.o.gt f.2.2 m) = true

theorem visit_sound (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (Q : R → F → Prop)
    (hrec : ∀ r f b, Q r b → Valid c f b → Q (rec r f b).1 (rec r f b).2)
    (k : Nat)
    (ihk : ∀ on pe pw pc cur rem (rs : R × F), Ctx c on pe pw pc cur → k + on.length = c.maxLen + 1 →
      (∀ x ∈ rem, x ∈ c.succ cur) → Q rs.1 rs.2 →
      Q (ch c rec k on pe pw pc rem rs).1 (ch c rec k on pe pw pc rem rs).2)
    (on pe pw pc cur) (hc : Ctx c on pe pw pc cur) (hk : k + 1 + on.length = c.maxLen + 1)
    (x : Nat × ε) (hx : x ∈ c.succ cur) (rs : R × F) (hq : Q rs.1 rs.2) :
    Q (visit c rec (ch c rec k) on pe pw pc x rs).1 (visit c rec (ch c rec k) on pe pw pc x rs).2 := by
  unfold visit
  by_cases hn : x.1 ∈ on
  · simpa [hn] using hq
  · simp only [hn, ↓reduceIte]
    rcases hok : okStep c rs.2 pw pc x.2 with _ | ⟨nw, nc⟩
    · simpa using hq
    · simp only
      obtain ⟨hnw, hnc, hf1, hg1, hf2, hm⟩ := okStep_some c _ _ _ _ _ _ hok
      have hch := ctx_child c on pe pw pc cur x nw nc hc hx hn hnw hnc
      have hval : atT c x.1 = true → Valid c (pe ++ [x], nw, nc) rs.2 := by
        intro hat
        refine ⟨x.1, hch.walk, by simp, ?_, ?_, hat, hch.w, hch.cst, hf1, hf2, hg1, hm⟩
        · have := hch.nodup
          rw [hch.on_eq] at this
          exact ((List.reverse_perm _).nodup_iff).mp this
        · show (pe ++ [x]).length ≤ c.maxLen
          have h1 : (pe ++ [x]).length = on.length := by
            have := congrArg List.length hc.on_eq
            simp [nodesOf] at this ⊢; omega
          omega
      have hq1 : Q (if atT c x.1 = true then rec rs.1 (pe ++ [x], nw, nc) rs.2 else rs).1
          (if atT c x.1 = true then rec rs.1 (pe ++ [x], nw, nc) rs.2 else rs).2 := by
        by_cases hat : atT c x.1 = true
        · simp only [atT] at hat ⊢
          simp only [hat, ↓reduceIte]
          exact hrec _ _ _ hq (hval hat)
        · simp only [atT] at hat ⊢
          simp only [hat, ↓reduceIte]; exact hq
      split
      · refine ihk _ _ _ _ x.1 _ _ hch (by simp; omega) (fun y hy => ?_) hq1
        simpa using hy
      · exact hq1

theorem ch_sound (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (Q : R → F → Prop)
    (hrec : ∀ r f b, Q r b → Valid c f b → Q (rec r f b).1 (rec r f b).2) :
    ∀ (k : Nat) on pe pw pc cur rem (rs : R × F), Ctx c on pe pw pc cur → k + on.length = c.maxLen + 1 →
      (∀ x ∈ rem, x ∈ c.succ cur) → Q rs.1 rs.2 →
      Q (ch c rec k on pe pw pc rem rs).1 (ch c rec k on pe pw pc rem rs).2 := by
  intro k
  induction k with
  | zero => intro on pe pw pc cur rem rs _ _ _ hq; simpa [ch] using hq
  | succ k ih =>
    intro on pe pw pc cur rem rs hc hk hrem hq
    simp only [ch]
    induction rem generalizing rs with
    | nil => simpa using hq
    | cons x r ihr =>
      simp only [List.foldl_cons]
      refine ihr _ (fun y hy => hrem y (List.mem_cons_of_mem _ hy)) ?_
      exact visit_sound c rec Q hrec k ih on pe pw pc cur hc hk x (hrem x (by simp)) rs hq

/-- **PROVEN** (`enumerate_paths`, soundness): for any property `Q` of (results, bound) that
`record_found_path` (`rec`) keeps when handed a `Valid` path, `Q` holds of the search's output; in
particular every path ever handed to `rec` is `Valid`. -/
theorem enum_sound (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (Q : R → F → Prop)
    (hrec : ∀ r f b, Q r b → Valid c f b → Q (rec r f b).1 (rec r f b).2) (r0 : R) (b0 : F)
    (h0 : Q r0 b0) : Q (enumRec c rec r0 b0).1 (enumRec c rec r0 b0).2 := by
  unfold enumRec
  refine ch_sound c rec Q hrec c.maxLen _ _ _ _ c.source _ (r0, b0) (ctx_root c) (by simp) ?_ h0
  intro x hx
  by_cases hm : 0 < c.maxLen
  · simpa [hm] using hx
  · simp [hm] at hx

/-! ## Completeness (a recorder that keeps the bound) -/

def recApp (r : List (Fnd ε F)) (f : Fnd ε F) (b : F) : List (Fnd ε F) × F := (r ++ [f], b)

/-- `es` (the rest of a path below `cur`, first edge first) is a path the search hands to
`record_found_path` under the fixed bound `b`: every step is a successor, not on the path, passes
`okStep`, and the path either stops at the first target-hit or continues while the search expands. -/
def Cand (c : EC ε F) (b : F) : List Nat → Nat → F → F → List (Nat × ε) → Prop
  | _, _, _, _, [] => False
  | on, cur, pw, pc, x :: rest =>
    x ∈ c.succ cur ∧ x.1 ∉ on ∧ ∃ nw nc, okStep c b pw pc x.2 = some (nw, nc) ∧
      ((atT c x.1 = true ∧ rest = []) ∨
       ((!(atT c x.1 && c.target.isSome) && decide (on.length < c.maxLen)) = true ∧
          Cand c b (x.1 :: on) x.1 nw nc rest))

def WtOk (c : EC ε F) (P : Fnd ε F) : Prop :=
  P.2.1 = fsum c.o.add c.o.zero c.wf P.1 ∧ P.2.2 = fsum c.o.add c.o.zero c.cf P.1

theorem cand_nil (c : EC ε F) (b : F) (on cur pw pc) : ¬ Cand c b on cur pw pc [] := id

def ChMem (c : EC ε F) (b : F) (k : Nat) : Prop :=
  ∀ on pe pw pc cur rem (rs : List (Fnd ε F) × F), Ctx c on pe pw pc cur → k + on.length = c.maxLen + 1 →
    (k = 0 → rem = []) → (∀ x ∈ rem, x ∈ c.succ cur) → rs.2 = b → ∀ P : Fnd ε F,
    (P ∈ (ch c recApp k on pe pw pc rem rs).1 ↔
      (P ∈ rs.1 ∨ ∃ x ∈ rem, ∃ rest, P.1 = pe ++ x :: rest ∧ WtOk c P ∧ Cand c b on cur pw pc (x :: rest)))

theorem keep_bound (c : EC ε F) (b : F) :
    ∀ (k : Nat) on pe pw pc cur rem (rs : List (Fnd ε F) × F), Ctx c on pe pw pc cur →
      k + on.length = c.maxLen + 1 → (∀ x ∈ rem, x ∈ c.succ cur) → rs.2 = b →
      (ch c recApp k on pe pw pc rem rs).2 = b := by
  intro k on pe pw pc cur rem rs hc hk hrem hb
  exact ch_sound c recApp (fun _ b' => b' = b) (fun r f b' h _ => h) k on pe pw pc cur rem rs hc hk hrem hb

theorem visit_mem (c : EC ε F) (b : F) (k : Nat) (ihk : ChMem c b k)
    (on pe pw pc cur) (hc : Ctx c on pe pw pc cur) (hk : k + 1 + on.length = c.maxLen + 1)
    (x : Nat × ε) (hx : x ∈ c.succ cur) (rs : List (Fnd ε F) × F) (hb : rs.2 = b) (P : Fnd ε F) :
    P ∈ (visit c recApp (ch c recApp k) on pe pw pc x rs).1 ↔
      (P ∈ rs.1 ∨ ∃ rest, P.1 = pe ++ x :: rest ∧ WtOk c P ∧ Cand c b on cur pw pc (x :: rest)) := by
  unfold visit
  by_cases hn : x.1 ∈ on
  · simp only [hn, ↓reduceIte]
    constructor
    · intro h; exact Or.inl h
    · rintro (h | ⟨rest, _, _, hcd⟩)
      · exact h
      · exact absurd hn hcd.2.1
  · simp only [hn, ↓reduceIte]
    rcases hok : okStep c rs.2 pw pc x.2 with _ | ⟨nw, nc⟩
    · simp only
      constructor
      · intro h; exact Or.inl h
      · rintro (h | ⟨rest, _, _, _, _, nw', nc', hok2, _⟩)
        · exact h
        · rw [hb] at hok; rw [hok] at hok2; simp at hok2
    · simp only
      obtain ⟨hnw, hnc, hf1, hg1, hf2, hm⟩ := okStep_some c _ _ _ _ _ _ hok
      have hch := ctx_child c on pe pw pc cur x nw nc hc hx hn hnw hnc
      have hok' : okStep c b pw pc x.2 = some (nw, nc) := by rw [← hb]; exact hok
      have hF0 : ∀ P : Fnd ε F, P = (pe ++ [x], nw, nc) → P.1 = pe ++ x :: [] ∧ WtOk c P := by
        intro P hP
        subst hP
        exact ⟨rfl, hch.w, hch.cst⟩
      let rs1 : List (Fnd ε F) × F :=
        if Option.all (fun t => x.fst == t) c.target = true then recApp rs.fst (pe ++ [x], nw, nc) rs.snd else rs
      have hrs1b : rs1.2 = b := by
        by_cases hat : Option.all (fun t => x.fst == t) c.target = true
        · simp [rs1, hat, recApp, hb]
        · simp [rs1, hat, hb]
      have hrs1m : ∀ P : Fnd ε F, P ∈ rs1.1 ↔ P ∈ rs.1 ∨ (atT c x.1 = true ∧ P = (pe ++ [x], nw, nc)) := by
        intro P
        by_cases hat : Option.all (fun t => x.fst == t) c.target = true
        · have hat' : atT c x.1 = true := hat
          simp [rs1, hat, hat', recApp]
        · have hat' : ¬ atT c x.1 = true := hat
          simp [rs1, hat, hat']
      have hhead : P.1 = pe ++ x :: [] → WtOk c P → atT c x.1 = true → P ∈ rs1.1 := by
        intro h1 h2 hat
        refine (hrs1m P).mpr (Or.inr ⟨hat, ?_⟩)
        obtain ⟨P1, P2, P3⟩ := P
        simp only at h1 h2
        obtain ⟨hw, hcs⟩ := h2
        simp only at hw hcs
        rw [h1, ← hch.w] at hw
        rw [h1, ← hch.cst] at hcs
        simp [h1, hw, hcs]
      show P ∈ (if (!(Option.all (fun t => x.fst == t) c.target && c.target.isSome) && decide (on.length < c.maxLen)) = true
          then ch c recApp k (x.fst :: on) (pe ++ [x]) nw nc (c.succ x.fst).reverse rs1 else rs1).fst ↔ _
      by_cases hexp : (!(Option.all (fun t => x.fst == t) c.target && c.target.isSome) && decide (on.length < c.maxLen)) = true
      · rw [if_pos hexp]
        have hlt : on.length < c.maxLen := decide_eq_true_iff.mp (Bool.and_eq_true_iff.mp hexp).2
        have hmem := ihk (x.fst :: on) (pe ++ [x]) nw nc x.fst (c.succ x.fst).reverse rs1 hch
          (by simp only [List.length_cons]; omega) (fun h0 => by omega)
          (fun y hy => List.mem_reverse.mp hy) hrs1b P
        rw [hmem]
        constructor
        · rintro (h | ⟨y, hy, rest', h1, h2, h3⟩)
          · rcases (hrs1m P).mp h with h | ⟨hat, hP⟩
            · exact Or.inl h
            · obtain ⟨e1, e2⟩ := hF0 P hP
              exact Or.inr ⟨[], e1, e2, hx, hn, nw, nc, hok', Or.inl ⟨hat, rfl⟩⟩
          · refine Or.inr ⟨y :: rest', by simpa using h1, h2, hx, hn, nw, nc, hok', Or.inr ⟨hexp, h3⟩⟩
        · rintro (h | ⟨rest, h1, h2, hx', hn', nw', nc', hok2, hd⟩)
          · exact Or.inl ((hrs1m P).mpr (Or.inl h))
          · rw [hok'] at hok2
            simp only [Option.some.injEq, Prod.mk.injEq] at hok2
            obtain ⟨rfl, rfl⟩ := hok2
            rcases hd with ⟨hat, rfl⟩ | ⟨_, hcd⟩
            · exact Or.inl (hhead h1 h2 hat)
            · cases rest with
              | nil => exact absurd hcd (cand_nil _ _ _ _ _ _)
              | cons y rest' =>
                refine Or.inr ⟨y, List.mem_reverse.mpr hcd.1, rest', by simpa using h1, h2, hcd⟩
      · rw [if_neg hexp]
        rw [hrs1m P]
        constructor
        · rintro (h | ⟨hat, hP⟩)
          · exact Or.inl h
          · obtain ⟨e1, e2⟩ := hF0 P hP
            exact Or.inr ⟨[], e1, e2, hx, hn, nw, nc, hok', Or.inl ⟨hat, rfl⟩⟩
        · rintro (h | ⟨rest, h1, h2, hx', hn', nw', nc', hok2, hd⟩)
          · exact Or.inl h
          · rw [hok'] at hok2
            simp only [Option.some.injEq, Prod.mk.injEq] at hok2
            obtain ⟨rfl, rfl⟩ := hok2
            rcases hd with ⟨hat, rfl⟩ | ⟨hexp', _⟩
            · have := hhead h1 h2 hat
              exact (hrs1m P).mp this
            · exact absurd hexp' hexp

theorem ch_mem (c : EC ε F) (b : F) : ∀ k, ChMem c b k := by
  intro k
  induction k with
  | zero =>
    intro on pe pw pc cur rem rs _ _ h0 _ _ P
    have := h0 rfl
    subst this
    simp [ch]
  | succ k ih =>
    intro on pe pw pc cur rem rs hc hk _ hrem hb P
    simp only [ch]
    have key : ∀ (rem : List (Nat × ε)) (rs : List (Fnd ε F) × F), (∀ x ∈ rem, x ∈ c.succ cur) → rs.2 = b →
        (P ∈ (rem.foldl (fun rs x => visit c recApp (ch c recApp k) on pe pw pc x rs) rs).1 ↔
          (P ∈ rs.1 ∨ ∃ x ∈ rem, ∃ rest, P.1 = pe ++ x :: rest ∧ WtOk c P ∧ Cand c b on cur pw pc (x :: rest))) := by
      intro rem
      induction rem with
      | nil => intro rs _ _; simp
      | cons x r ihr =>
        intro rs hrem hb
        simp only [List.foldl_cons]
        have hx : x ∈ c.succ cur := hrem x (by simp)
        have hb' : (visit c recApp (ch c recApp k) on pe pw pc x rs).2 = b :=
          visit_sound c recApp (fun _ b' => b' = b) (fun r f b' h _ => h) k
            (fun on pe pw pc cur rem rs hc hk hrem hq => ch_sound c recApp (fun _ b' => b' = b) (fun r f b' h _ => h) k on pe pw pc cur rem rs hc hk hrem hq)
            on pe pw pc cur hc hk x hx rs hb
        rw [ihr _ (fun y hy => hrem y (List.mem_cons_of_mem _ hy)) hb', visit_mem c b k ih on pe pw pc cur hc hk x hx rs hb P]
        constructor
        · rintro ((h | ⟨rest, h1, h2, h3⟩) | ⟨y, hy, rest, h1, h2, h3⟩)
          · exact Or.inl h
          · exact Or.inr ⟨x, by simp, rest, h1, h2, h3⟩
          · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, rest, h1, h2, h3⟩
        · rintro (h | ⟨y, hy, rest, h1, h2, h3⟩)
          · exact Or.inl (Or.inl h)
          · rcases List.mem_cons.mp hy with rfl | hy
            · exact Or.inl (Or.inr ⟨rest, h1, h2, h3⟩)
            · exact Or.inr ⟨y, hy, rest, h1, h2, h3⟩
    exact key rem rs hrem hb

/-- **PROVEN** (`enumerate_paths`, exactness for a bound that does not move): with a recorder that
only appends (`recApp`: the bound `b` never changes), the paths the search hands over are exactly
`Cand`: non-empty paths from the source, each step a successor edge (`c.succ`, i.e. `node_successors`
with its direction/`relTypes`), no repeated node (the source included), every prefix passing
`okStep` (finite weight/cost, `weight <= b`, `cost <= maxCost`), ending at the target (any node for
SSpaths), not extended past a target hit (SPpaths), at most `max_len` edges. Their weight/cost are
the left folds over the edges (`WtOk`). -/
theorem enum_complete (c : EC ε F) (b : F) (P : Fnd ε F) :
    P ∈ (enumRec c recApp ([] : List (Fnd ε F)) b).1 ↔
      (0 < c.maxLen ∧ ∃ es, P.1 = es ∧ WtOk c P ∧ Cand c b [c.source] c.source c.o.zero c.o.zero es) := by
  unfold enumRec
  have := ch_mem c b c.maxLen [c.source] [] c.o.zero c.o.zero c.source
    (if 0 < c.maxLen then (c.succ c.source).reverse else []) ([], b) (ctx_root c) (by simp)
    (fun h0 => by simp [h0]) (by
      intro x hx
      by_cases hm : 0 < c.maxLen
      · simpa [hm] using hx
      · simp [hm] at hx) rfl P
  rw [this]
  by_cases hm : 0 < c.maxLen
  · simp only [hm, ↓reduceIte, List.not_mem_nil, false_or, List.mem_reverse, List.nil_append, true_and]
    constructor
    · rintro ⟨x, _, rest, h1, h2, h3⟩; exact ⟨x :: rest, h1, h2, h3⟩
    · rintro ⟨es, h1, h2, h3⟩
      cases es with
      | nil => exact absurd h3 (cand_nil _ _ _ _ _ _)
      | cons x rest => exact ⟨x, h3.1, rest, h1, h2, h3⟩
  · simp [hm]

end OpsTraverse.EnumSpec
