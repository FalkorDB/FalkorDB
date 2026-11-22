import OpsTraverse.Dijkstra
/-
# `dijkstra_single_path` over an exact cost domain (`Nat` weights)

The same `Dijkstra.dijkstra` (same loop, same lazy-deletion heap, same labels) instantiated at
`F = Nat` with `+`, `<` and `is_finite = true`: `f64` optimality is not provable (`f64` `+` is not
associative; see `Dijkstra.lean`), but over an exact domain the algorithm is optimal. Heap = `popMax`
under `DijkstraItem::cmp` (`AlgoPaths.itemCmp`) with `partial_cmp = compare`.

`dijkstraNat_spec`: a returned path is a cheapest walk from the source to the target (no walk is
cheaper), its weight is exactly the sum of its edges' weights, its cost the sum of its costs; and
`None` only when the target is the source or unreachable.
-/
namespace OpsTraverse.DijkstraNat
open OpsTraverse.Search OpsTraverse.BfsFind OpsTraverse.AlgoPaths OpsTraverse.Dijkstra

variable {ε : Type}

def fnat : FO Nat := ⟨0, (· + ·), fun a b => decide (a < b), fun _ => true⟩

def natP : Nat → Nat → Option Ordering := fun a b => some (compare a b)

def popNat : List (Item Nat) → Option (Item Nat × List (Item Nat)) := popMax (itemCmp natP)

theorem itemCmp_nat_gt (a b : Item Nat) :
    itemCmp natP a b = .gt ↔ a.weight < b.weight ∨ (a.weight = b.weight ∧ a.node < b.node) := by
  simp only [itemCmp, pc, natP, Option.getD_some]
  rcases Nat.lt_trichotomy a.weight b.weight with h | h | h
  · have : compare b.weight a.weight = .gt := Nat.compare_eq_gt.mpr h
    simp [this, Ordering.then, h]
  · have : compare b.weight a.weight = .eq := Nat.compare_eq_eq.mpr h.symm
    rw [this]
    show compare b.node a.node = .gt ↔ _
    rw [Nat.compare_eq_gt]
    omega
  · have : compare b.weight a.weight = .lt := Nat.compare_eq_lt.mpr h
    simp only [this, Ordering.then, reduceCtorEq, false_iff]
    omega

theorem popMax_max {α : Type} (cmp : α → α → Ordering)
    (htr : ∀ a b c, cmp a b ≠ .gt → cmp b c ≠ .gt → cmp a c ≠ .gt) (hrefl : ∀ a, cmp a a ≠ .gt)
    (hasy : ∀ a b, cmp a b = .gt → cmp b a ≠ .gt) :
    ∀ (h : List α) (x : α) (r : List α), popMax cmp h = some (x, r) → ∀ y ∈ h, cmp y x ≠ .gt := by
  intro h
  induction h with
  | nil => intro x r hp; simp [popMax] at hp
  | cons y0 t ih =>
    intro x r hp y hy
    simp only [popMax] at hp
    rcases hm : popMax cmp t with _ | ⟨m, r'⟩
    · rw [hm] at hp
      simp only [Option.some.injEq, Prod.mk.injEq] at hp
      obtain ⟨rfl, rfl⟩ := hp
      have ht : t = [] := popMax_none cmp t hm
      subst ht
      simp at hy; subst hy; exact hrefl _
    · rw [hm] at hp
      have hmax := ih m r' hm
      by_cases hc : cmp m y0 = .gt
      · simp only [hc, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hp
        obtain ⟨rfl, rfl⟩ := hp
        rcases List.mem_cons.mp hy with rfl | hy
        · exact hasy _ _ hc
        · exact hmax y hy
      · simp only [hc, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hp
        obtain ⟨rfl, rfl⟩ := hp
        rcases List.mem_cons.mp hy with rfl | hy
        · exact hrefl _
        · exact htr _ _ _ (hmax y hy) hc

theorem popNat_min (H : List (Item Nat)) (it : Item Nat) (rest : List (Item Nat))
    (h : popNat H = some (it, rest)) : ∀ y ∈ H, it.weight ≤ y.weight := by
  have := popMax_max (itemCmp natP) ?_ ?_ ?_ H it rest h
  · intro y hy
    have h1 := this y hy
    rw [Ne, itemCmp_nat_gt] at h1
    omega
  · intro a b c h1 h2
    rw [Ne, itemCmp_nat_gt] at *
    omega
  · intro a; rw [Ne, itemCmp_nat_gt]; omega
  · intro a b h1; rw [itemCmp_nat_gt] at h1; rw [Ne, itemCmp_nat_gt]; omega

theorem popNat_ok : PopOk popNat := popOk_popMax _

/-! ## Costs and optimality -/

def cwFrom (wf : ε → Nat) (a : Nat) (es : List (Nat × ε)) : Nat := es.foldl (fun acc x => acc + wf x.2) a

theorem cwFrom_add (wf : ε → Nat) (es : List (Nat × ε)) : ∀ a, cwFrom wf a es = a + cwFrom wf 0 es := by
  induction es with
  | nil => intro a; simp [cwFrom]
  | cons x r ih =>
    intro a
    simp only [cwFrom, List.foldl_cons] at *
    rw [ih (a + wf x.2), ih (0 + wf x.2)]
    omega

theorem fsum_nat (wf : ε → Nat) (es : List (Nat × ε)) : fsum fnat.add fnat.zero wf es = cwFrom wf 0 es := rfl

theorem cwFrom_append (wf : ε → Nat) (a b : List (Nat × ε)) :
    cwFrom wf 0 (a ++ b) = cwFrom wf 0 a + cwFrom wf 0 b := by
  simp only [cwFrom, List.foldl_append]
  have := cwFrom_add wf b (a.foldl (fun acc x => acc + wf x.2) 0)
  exact this

/-- `c` is the cheapest cost of a walk `s → x`. -/
def Opt (step : Step ε) (s : Nat) (wf : ε → Nat) (x c : Nat) : Prop :=
  (∃ es, WalkTo step s es x ∧ cwFrom wf 0 es = c) ∧ ∀ es, WalkTo step s es x → c ≤ cwFrom wf 0 es

theorem Opt_unique (step : Step ε) (s : Nat) (wf : ε → Nat) (x c c' : Nat)
    (h : Opt step s wf x c) (h' : Opt step s wf x c') : c = c' := by
  obtain ⟨⟨es, hw, hc⟩, hm⟩ := h
  obtain ⟨⟨es', hw', hc'⟩, hm'⟩ := h'
  have := hm es' hw'
  have := hm' es hw
  omega

theorem Opt_source (step : Step ε) (s : Nat) (wf : ε → Nat) : Opt step s wf s 0 :=
  ⟨⟨[], rfl, rfl⟩, fun _ _ => Nat.zero_le _⟩

theorem Opt_source_eq (step : Step ε) (s : Nat) (wf : ε → Nat) (c : Nat) (h : Opt step s wf s c) : c = 0 :=
  Opt_unique step s wf s c 0 h (Opt_source step s wf)


theorem fnat_fin (x : Nat) : fnat.fin x = true := rfl

theorem lblGet_cons_ne {F : Type} (L : List (Nat × Nat × ε × F)) (k p : Nat) (e : ε) (w : F) (x : Nat)
    (h : k ≠ x) : lblGet ((k, p, e, w) :: L) x = lblGet L x := by
  simp [lblGet, List.find?_cons, h]

theorem lblGet_cons_self {F : Type} (L : List (Nat × Nat × ε × F)) (k p : Nat) (e : ε) (w : F) :
    lblGet ((k, p, e, w) :: L) k = some (p, e, w) := by
  simp [lblGet, List.find?_cons]

/-- The Nat-specific invariant. `ex` is the node whose edges are still being relaxed. -/
structure OI (step : Step ε) (s : Nat) (wf : ε → Nat) (ex : List Nat) (st : DS ε Nat) : Prop where
  init : st.settled = [] → st.heap = [⟨0, s⟩]
  o1 : ∀ x ∈ st.settled, ∃ c, Opt step s wf x c ∧ (x ≠ s → ∃ p e, lblGet st.labels x = some (p, e, c))
  o2 : ∀ y p e lw, y ∉ st.settled → lblGet st.labels y = some (p, e, lw) →
        (∃ es, WalkTo step s es y ∧ cwFrom wf 0 es = lw) ∧ (⟨lw, y⟩ : Item Nat) ∈ st.heap
  o3 : ∀ x ∈ st.settled, x ∉ ex → ∀ c, Opt step s wf x c → ∀ y e, (y, e) ∈ step x →
        y ∈ st.settled ∨ ∃ p e' lw, lblGet st.labels y = some (p, e', lw) ∧ lw ≤ c + wf e
  o4 : ∀ it ∈ st.heap, ∀ p e lw, lblGet st.labels it.node = some (p, e, lw) → lw ≤ it.weight
  o6 : ∀ x p e w, (x, p, e, w) ∈ st.labels → ∃ cp, Opt step s wf p cp ∧ w = cp + wf e ∧
        (p ≠ s → ∃ p' e', lblGet st.labels p = some (p', e', cp))

theorem OI_init (step : Step ε) (s : Nat) (wf : ε → Nat) :
    OI step s wf [] (⟨[], [], [⟨fnat.zero, s⟩]⟩ : DS ε Nat) where
  init _ := rfl
  o1 x h := by simp at h
  o2 y p e lw h := by simp [lblGet]
  o3 x h := by simp at h
  o4 it h p e lw hl := by simp [lblGet] at hl
  o6 x p e w h := by simp at h

/-- Walks starting at a settled node meet an unsettled label that is on the heap. -/
theorem first_unsettled (step : Step ε) (s : Nat) (wf : ε → Nat) (st : DS ε Nat) (O : OI step s wf [] st)
    (u : Nat) (hu : u ∉ st.settled) :
    ∀ (es : List (Nat × ε)) (a c : Nat), a ∈ st.settled → Opt step s wf a c → WalkTo step a es u →
      ∃ y p e lw, y ∉ st.settled ∧ lblGet st.labels y = some (p, e, lw) ∧ (⟨lw, y⟩ : Item Nat) ∈ st.heap ∧
        lw ≤ c + cwFrom wf 0 es := by
  intro es
  induction es with
  | nil =>
    intro a c ha _ hw
    simp only [WalkTo] at hw
    subst hw; exact absurd ha hu
  | cons x r ih =>
    intro a c ha hopt hw
    obtain ⟨hx, hwr⟩ := hw
    obtain ⟨v, e⟩ := x
    simp only at hx hwr
    have hcw : cwFrom wf 0 ((v, e) :: r) = wf e + cwFrom wf 0 r := by
      have h := cwFrom_add wf r (0 + wf e)
      have h0 : cwFrom wf 0 ((v, e) :: r) = cwFrom wf (0 + wf e) r := rfl
      rw [h0, h]; omega
    by_cases hv : v ∈ st.settled
    · obtain ⟨cv, hoptv, _⟩ := O.o1 v hv
      have hle : cv ≤ c + wf e := by
        obtain ⟨⟨es0, hw0, hc0⟩, _⟩ := hopt
        have h1 := hoptv.2 (es0 ++ [(v, e)]) ((walk_append step s es0 _ v).mpr ⟨a, hw0, by simpa [WalkTo] using hx⟩)
        rw [cwFrom_append, hc0] at h1
        have : cwFrom wf 0 [(v, e)] = wf e := by simp [cwFrom]
        omega
      obtain ⟨y, p, e', lw, h1, h2, h3, h4⟩ := ih v cv hv hoptv hwr
      exact ⟨y, p, e', lw, h1, h2, h3, by rw [hcw]; omega⟩
    · rcases O.o3 a ha (by simp) c hopt v e hx with h | ⟨p, e', lw, hl, hle⟩
      · exact absurd h hv
      · obtain ⟨_, hit⟩ := O.o2 v p e' lw hv hl
        exact ⟨v, p, e', lw, hv, hl, hit, by rw [hcw]; omega⟩


theorem lt_of_match (L : List (Nat × Nat × ε × Nat)) (far fw : Nat)
    (h : (match lblGet L far with | none => true | some (_, _, lw) => fnat.lt fw lw) = true) :
    ∀ p e lw, lblGet L far = some (p, e, lw) → fw < lw := by
  intro p e lw hl
  rw [hl] at h
  simpa [fnat] using h

theorem OI_push (step : Step ε) (s : Nat) (wf : ε → Nat) (ex : List Nat) (st : DS ε Nat)
    (I : SI fnat step s st) (O : OI step s wf ex st) (node far : Nat) (e : ε) (w : Nat)
    (hopt : Opt step s wf node w) (hn : node ∈ st.settled) (hs : s ∈ st.settled)
    (hmem : (far, e) ∈ step node) (h1 : far ∉ st.settled)
    (hlt : (match lblGet st.labels far with | none => true | some (_, _, lw) => fnat.lt (fnat.add w (wf e)) lw) = true) :
    OI step s wf ex { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] } := by
  have hfw : fnat.add w (wf e) = w + wf e := rfl
  have hne : ∀ x ∈ st.settled, far ≠ x := fun x hx h => h1 (h ▸ hx)
  have lbl_old : ∀ x ∈ st.settled, lblGet ((far, node, e, fnat.add w (wf e)) :: st.labels) x = lblGet st.labels x :=
    fun x hx => lblGet_cons_ne _ _ _ _ _ _ (hne x hx)
  have lbl_far : lblGet ((far, node, e, fnat.add w (wf e)) :: st.labels) far = some (node, e, fnat.add w (wf e)) :=
    lblGet_cons_self _ _ _ _ _
  have lbl_other : ∀ y, y ≠ far → lblGet ((far, node, e, fnat.add w (wf e)) :: st.labels) y = lblGet st.labels y :=
    fun y hy => lblGet_cons_ne _ _ _ _ _ _ (fun h => hy h.symm)
  have hltd := lt_of_match st.labels far _ hlt
  obtain ⟨⟨es0, hw0, hc0⟩, hmin⟩ := hopt
  have hwalk : WalkTo step s (es0 ++ [(far, e)]) far :=
    (walk_append step s es0 _ far).mpr ⟨node, hw0, by simpa [WalkTo] using hmem⟩
  have hcw : cwFrom wf 0 (es0 ++ [(far, e)]) = w + wf e := by
    rw [cwFrom_append, hc0]; simp [cwFrom]
  have hnodeLbl : node ≠ s → ∃ p e', lblGet st.labels node = some (p, e', w) := by
    intro hns
    obtain ⟨c, hc, hl⟩ := O.o1 node hn
    have : c = w := Opt_unique step s wf node c w hc ⟨⟨es0, hw0, hc0⟩, hmin⟩
    subst this; exact hl hns
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro h
    have h' : st.settled = [] := h
    rw [h'] at hn; simp at hn
  · intro x hx
    dsimp only at hx ⊢
    obtain ⟨c, hc, hl⟩ := O.o1 x hx
    refine ⟨c, hc, fun hxs => ?_⟩
    obtain ⟨p, e', hl'⟩ := hl hxs
    exact ⟨p, e', by rw [lbl_old x hx]; exact hl'⟩
  · intro y p e' lw hy hl
    dsimp only at hy hl ⊢
    by_cases hyf : y = far
    · subst hyf
      rw [lbl_far] at hl
      simp only [Option.some.injEq, Prod.mk.injEq] at hl
      obtain ⟨_, _, rfl⟩ := hl
      exact ⟨⟨es0 ++ [(y, e)], hwalk, by rw [hcw]; rfl⟩, List.mem_append_right _ (by simp)⟩
    · rw [lbl_other y hyf] at hl
      obtain ⟨hw, hit⟩ := O.o2 y p e' lw hy hl
      exact ⟨hw, List.mem_append_left _ hit⟩
  · intro x hx hxe c hc y e' hed
    dsimp only at hx ⊢
    rcases O.o3 x hx hxe c hc y e' hed with h | ⟨p, e'', lw, hl, hle⟩
    · exact Or.inl h
    · by_cases hyf : y = far
      · subst hyf
        have := hltd p e'' lw hl
        exact Or.inr ⟨node, e, _, lbl_far, by rw [hfw]; omega⟩
      · exact Or.inr ⟨p, e'', lw, by rw [lbl_other y hyf]; exact hl, hle⟩
  · intro it hit p e' lw hl
    dsimp only at hit hl ⊢
    rcases List.mem_append.mp hit with hit | hit
    · by_cases hf : it.node = far
      · rw [hf, lbl_far] at hl
        simp only [Option.some.injEq, Prod.mk.injEq] at hl
        obtain ⟨_, _, rfl⟩ := hl
        have hnl : ∃ p0 e0 lw0, lblGet st.labels far = some (p0, e0, lw0) := by
          rcases I.heapN it hit with h0 | ⟨p0, e0, w0, hm0⟩
          · exact absurd (h0.symm.trans hf ▸ hs) h1
          · rw [hf] at hm0; exact lblGet_exists st.labels far ⟨p0, e0, w0, hm0⟩
        obtain ⟨p0, e0, lw0, hl0⟩ := hnl
        have := hltd p0 e0 lw0 hl0
        have h4 := O.o4 it hit p0 e0 lw0 (by rw [hf]; exact hl0)
        omega
      · rw [lbl_other _ hf] at hl
        exact O.o4 it hit p e' lw hl
    · simp only [List.mem_singleton] at hit
      subst hit
      rw [lbl_far] at hl
      simp only [Option.some.injEq, Prod.mk.injEq] at hl
      obtain ⟨_, _, rfl⟩ := hl
      exact Nat.le_refl _
  · intro x p e' w' hm
    dsimp only at hm ⊢
    rcases List.mem_cons.mp hm with h | h
    · simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      refine ⟨w, ⟨⟨es0, hw0, hc0⟩, hmin⟩, hfw, fun hns => ?_⟩
      obtain ⟨p', e', hl'⟩ := hnodeLbl hns
      exact ⟨p', e', by rw [lbl_old _ hn]; exact hl'⟩
    · obtain ⟨cp, hcp, hw', hl'⟩ := O.o6 x p e' w' h
      have hpS : p ∈ st.settled := (I.ent x p e' w' h).1
      refine ⟨cp, hcp, hw', fun hps => ?_⟩
      obtain ⟨p', e'', hl''⟩ := hl' hps
      exact ⟨p', e'', by rw [lbl_old p hpS]; exact hl''⟩


theorem relax_cons_unsettled (wf : ε → Nat) (node w : Nat) (st : DS ε Nat) (far : Nat) (e : ε)
    (r : List (Nat × ε)) (h1 : far ∉ st.settled) :
    relax fnat wf node w ((far, e) :: r) st =
      if (match lblGet st.labels far with | none => true | some (_, _, lw) => fnat.lt (fnat.add w (wf e)) lw) = true
      then relax fnat wf node w r { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] }
      else relax fnat wf node w r st := by
  simp only [relax, h1, ↓reduceIte, fnat_fin, Bool.true_eq_false]

theorem relax_OI (step : Step ε) (s : Nat) (wf : ε → Nat) (node w : Nat) (hopt : Opt step s wf node w) :
    ∀ (l pr : List (Nat × ε)) (st : DS ε Nat), step node = pr ++ l → SI fnat step s st →
      OI step s wf [node] st → node ∈ st.settled → s ∈ st.settled →
      (∀ y e, (y, e) ∈ pr → y ∈ st.settled ∨ ∃ p e' lw, lblGet st.labels y = some (p, e', lw) ∧ lw ≤ w + wf e) →
      SI fnat step s (relax fnat wf node w l st) ∧ OI step s wf [node] (relax fnat wf node w l st) ∧
        (relax fnat wf node w l st).settled = st.settled ∧
        ∀ y e, (y, e) ∈ step node → y ∈ st.settled ∨
          ∃ p e' lw, lblGet (relax fnat wf node w l st).labels y = some (p, e', lw) ∧ lw ≤ w + wf e := by
  intro l
  induction l with
  | nil =>
    intro pr st hl I O hn hs hprog
    simp only [relax]
    refine ⟨I, O, trivial, fun y e hm => hprog y e (by rw [hl] at hm; simpa using hm)⟩
  | cons y r ih =>
    intro pr st hl I O hn hs hprog
    obtain ⟨far, e⟩ := y
    have hl' : step node = (pr ++ [(far, e)]) ++ r := by rw [hl]; simp
    have hmem : (far, e) ∈ step node := by rw [hl]; simp
    simp only [relax]
    by_cases h1 : far ∈ st.settled
    · simp only [h1, ↓reduceIte]
      refine ih _ st hl' I O hn hs (fun y' e' hm => ?_)
      rcases List.mem_append.mp hm with hm | hm
      · exact hprog y' e' hm
      · simp at hm; obtain ⟨rfl, _⟩ := hm; exact Or.inl h1
    · rw [relax_cons_unsettled wf node w st far e r h1]
      have hpush : ∀ (hlt : (match lblGet st.labels far with | none => true | some (_, _, lw) => fnat.lt (fnat.add w (wf e)) lw) = true),
          SI fnat step s (relax fnat wf node w r { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] }) ∧
          OI step s wf [node] (relax fnat wf node w r { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] }) ∧
          (relax fnat wf node w r { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] }).settled = st.settled ∧
          ∀ y e'', (y, e'') ∈ step node → y ∈ st.settled ∨ ∃ p e' lw, lblGet (relax fnat wf node w r { st with labels := (far, node, e, fnat.add w (wf e)) :: st.labels, heap := st.heap ++ [⟨fnat.add w (wf e), far⟩] }).labels y = some (p, e', lw) ∧ lw ≤ w + wf e'' := by
        intro h3
        have hltd := lt_of_match st.labels far _ h3
        have I' := SI_push fnat step s st I node far e (fnat.add w (wf e)) hn hs hmem h1 (fnat_fin _)
        have O' := OI_push step s wf [node] st I O node far e w hopt hn hs hmem h1 h3
        refine ih _ _ hl' I' O' hn hs (fun y' e' hm => ?_)
        rcases List.mem_append.mp hm with hm | hm
        · rcases hprog y' e' hm with h | ⟨p, e'', lw, hl2, hle⟩
          · exact Or.inl h
          · by_cases hyf : y' = far
            · subst hyf
              have := hltd p e'' lw hl2
              refine Or.inr ⟨node, e, fnat.add w (wf e), ?_, ?_⟩
              · exact lblGet_cons_self _ _ _ _ _
              · have hh : fnat.add w (wf e) = w + wf e := rfl
                rw [hh]; omega
            · refine Or.inr ⟨p, e'', lw, ?_, hle⟩
              exact (lblGet_cons_ne _ _ _ _ _ _ (fun h => hyf h.symm)).trans hl2
        · simp at hm; obtain ⟨hy, he⟩ := hm
          rw [hy, he]
          exact Or.inr ⟨node, e, fnat.add w (wf e), lblGet_cons_self _ _ _ _ _, Nat.le_refl _⟩
      have hskip : ∀ (hlt : ¬ ((match lblGet st.labels far with | none => true | some (_, _, lw) => fnat.lt (fnat.add w (wf e)) lw) = true)),
          SI fnat step s (relax fnat wf node w r st) ∧ OI step s wf [node] (relax fnat wf node w r st) ∧
          (relax fnat wf node w r st).settled = st.settled ∧
          ∀ y e'', (y, e'') ∈ step node → y ∈ st.settled ∨ ∃ p e' lw, lblGet (relax fnat wf node w r st).labels y = some (p, e', lw) ∧ lw ≤ w + wf e'' := by
        intro h3
        refine ih _ st hl' I O hn hs (fun y' e' hm => ?_)
        rcases List.mem_append.mp hm with hm | hm
        · exact hprog y' e' hm
        · simp at hm; obtain ⟨hy, he⟩ := hm
          rw [hy, he]
          rcases hg : lblGet st.labels far with _ | ⟨p, e'', lw⟩
          · rw [hg] at h3; simp at h3
          · rw [hg] at h3
            simp only [fnat, Bool.not_eq_true, decide_eq_false_iff_not, Bool.not_eq_true] at h3
            exact Or.inr ⟨p, e'', lw, hg, by show lw ≤ w + wf e; omega⟩
      by_cases h3 : (match lblGet st.labels far with | none => true | some (_, _, lw) => fnat.lt (fnat.add w (wf e)) lw) = true
      · rw [if_pos h3]; exact hpush h3
      · rw [if_neg h3]; exact hskip h3

end OpsTraverse.DijkstraNat
