import OpsTraverse.AlgoPaths
import OpsTraverse.BfsFind
/-
# `dijkstra_single_path` (graph/src/runtime/functions/algo_procedures.rs:2320)

Lazy-deletion Dijkstra with one label per reached node. `step u` is `node_successors` (far endpoint
and edge, in iterator order). The `f64` operations are the record `FO` (`0.0`, `+`, `<`,
`is_finite`); `wf`/`cf` are `edge_numeric_attr` for `weightProp` (default `1.0`) and `costProp`
(default `1.0` when set, `0.0` otherwise). The `BinaryHeap` is the parameter `pop` (a std contract:
it returns a greatest element under `DijkstraItem::cmp`); `popMax` is a concrete instance and the
generic theorems only use that `pop` returns a member of the heap. The loop is fuelled (`fuel` pops),
a `Rust` `while let` has none; `dijkLoop` returns `none` when the fuel runs out and every theorem is
about the finished runs. Timeout polls are abstracted away.

| here | there |
| --- | --- |
| `DS.labels` (newest first), `DS.settled`, `DS.heap` | `labels: FxHashMap<NodeId, DijkstraLabel>` (insert = prepend, read = first match), `settled`, `heap` |
| `relax` | the `for … in get_node_relationships_by_type` body (settled skip, `is_finite` skip, `far_weight < label.weight`) |
| `dijkLoop` | the `while let Some(DijkstraItem {weight, node}) = heap.pop()` loop |
| `dijkstra` | the function: chain walk-back (`BfsFind.chain`, fuel = number settled), cost fold, `labels[&target].weight` |
-/
namespace OpsTraverse.Dijkstra
open OpsTraverse.Search OpsTraverse.BfsFind OpsTraverse.AlgoPaths

variable {ε F : Type}

structure FO (F : Type) where
  zero : F
  add : F → F → F
  lt : F → F → Bool
  fin : F → Bool

structure DS (ε F : Type) where
  labels : List (Nat × Nat × ε × F)
  settled : List Nat
  heap : List (Item F)

def lblGet (L : List (Nat × Nat × ε × F)) (x : Nat) : Option (Nat × ε × F) :=
  (L.find? (fun q => q.1 == x)).map (·.2)

/-- The parent view of the labels (what `BfsFind.chain` reads). -/
def parOf (L : List (Nat × Nat × ε × F)) : List (Nat × Nat × ε) := L.map (fun q => (q.1, q.2.1, q.2.2.1))

/-- The `for` over `node`'s edges, after `settled.insert(node)`; `w` is the popped weight. -/
def relax (o : FO F) (wf : ε → F) (node : Nat) (w : F) : List (Nat × ε) → DS ε F → DS ε F
  | [], st => st
  | (far, e) :: r, st =>
    if far ∈ st.settled then relax o wf node w r st
    else
      let fw := o.add w (wf e)
      if o.fin fw = false then relax o wf node w r st
      else if (match lblGet st.labels far with | none => true | some (_, _, lw) => o.lt fw lw) = true then
        relax o wf node w r { st with labels := (far, node, e, fw) :: st.labels, heap := st.heap ++ [⟨fw, far⟩] }
      else relax o wf node w r st

def dijkLoop (o : FO F) (pop : List (Item F) → Option (Item F × List (Item F))) (step : Step ε)
    (wf : ε → F) (t : Nat) : Nat → DS ε F → Option (DS ε F × Bool)
  | 0, _ => none
  | n + 1, st =>
    match pop st.heap with
    | none => some (st, false)
    | some (it, rest) =>
      let st1 : DS ε F := { st with heap := rest }
      if it.node ∈ st1.settled then dijkLoop o pop step wf t n st1
      else
        let st2 : DS ε F := { st1 with settled := it.node :: st1.settled }
        if it.node == t then some (st2, true)
        else dijkLoop o pop step wf t n (relax o wf it.node it.weight (step it.node) st2)

/-- Outcome: `none` = out of fuel; `some none` = no path (or the `labels[&target]` panic);
`some (some (edges, weight, cost))`. -/
def dijkstra (o : FO F) (pop : List (Item F) → Option (Item F × List (Item F))) (step : Step ε)
    (s t : Nat) (wf cf : ε → F) (fuel : Nat) : Option (Option (List (Nat × ε) × F × F)) :=
  match dijkLoop o pop step wf t fuel ⟨[], [], [⟨o.zero, s⟩]⟩ with
  | none => none
  | some (_, false) => some none
  | some (st, true) =>
    match chain s (parOf st.labels) st.settled.length t with
    | none => some none
    | some es =>
      match lblGet st.labels t with
      | none => some none
      | some (_, _, w) => some (some (es, w, fsum o.add o.zero cf es))

/-- A max-heap pop: a greatest element (first of equals), the rest in some order. -/
def popMax {α : Type} (cmp : α → α → Ordering) : List α → Option (α × List α)
  | [] => none
  | x :: r =>
    match popMax cmp r with
    | none => some (x, [])
    | some (m, r') => if cmp m x = .gt then some (m, x :: r') else some (x, r)


/-! ## Heap contract -/

/-- What the loop needs of `BinaryHeap::pop`: `None` only when empty, otherwise some member and the
remaining items. -/
def PopOk (pop : List (Item F) → Option (Item F × List (Item F))) : Prop :=
  (∀ h, pop h = none → h = []) ∧ (∀ h x r, pop h = some (x, r) → (x :: r).Perm h)

theorem popMax_none {α : Type} (cmp : α → α → Ordering) (h : List α) : popMax cmp h = none → h = [] := by
  cases h with
  | nil => intro _; rfl
  | cons y t =>
    intro hp
    simp only [popMax] at hp
    rcases hm : popMax cmp t with _ | ⟨m, r'⟩
    · rw [hm] at hp; simp at hp
    · rw [hm] at hp; simp only at hp; split at hp <;> simp at hp

theorem popMax_perm {α : Type} (cmp : α → α → Ordering) :
    ∀ (h : List α) (x : α) (r : List α), popMax cmp h = some (x, r) → (x :: r).Perm h := by
  intro h
  induction h with
  | nil => intro x r hp; simp [popMax] at hp
  | cons y t ih =>
    intro x r hp
    simp only [popMax] at hp
    rcases hm : popMax cmp t with _ | ⟨m, r'⟩
    · rw [hm] at hp
      simp only [Option.some.injEq, Prod.mk.injEq] at hp
      obtain ⟨rfl, rfl⟩ := hp
      have ht : t = [] := popMax_none cmp t hm
      subst ht; exact List.Perm.refl _
    · rw [hm] at hp
      have hperm := ih m r' hm
      by_cases hc : cmp m y = .gt
      · simp only [hc, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hp
        obtain ⟨rfl, rfl⟩ := hp
        exact (List.Perm.swap y m r').trans ((hperm.cons y))
      · simp only [hc, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hp
        obtain ⟨rfl, rfl⟩ := hp
        exact List.Perm.refl _

theorem popOk_popMax (cmp : Item F → Item F → Ordering) : PopOk (popMax cmp) :=
  ⟨popMax_none cmp, popMax_perm cmp⟩

/-! ## Structural invariant (any `F`, any heap) -/

structure SI (o : FO F) (step : Step ε) (s : Nat) (st : DS ε F) : Prop where
  nodup : st.settled.Nodup
  hs : st.settled = [] ∨ s ∈ st.settled
  init : st.settled = [] → st.labels = [] ∧ ∃ w, st.heap = [⟨w, s⟩]
  heapN : ∀ it ∈ st.heap, it.node = s ∨ ∃ p e w, (it.node, p, e, w) ∈ st.labels
  setLbl : ∀ x ∈ st.settled, x ≠ s → ∃ p e w, (x, p, e, w) ∈ st.labels
  ent : ∀ x p e w, (x, p, e, w) ∈ st.labels → p ∈ st.settled ∧ (x, e) ∈ step p ∧ x ≠ s ∧
        (x ∈ st.settled → st.settled.idxOf x < st.settled.idxOf p)
  fin : ∀ x p e w, (x, p, e, w) ∈ st.labels → o.fin w = true

theorem SI_init (o : FO F) (step : Step ε) (s : Nat) : SI o step s (⟨[], [], [⟨o.zero, s⟩]⟩ : DS ε F) where
  nodup := List.nodup_nil
  hs := Or.inl rfl
  init _ := ⟨rfl, o.zero, rfl⟩
  heapN it h := by simp at h; exact Or.inl (by rw [h])
  setLbl x h := by simp at h
  ent x p e w h := by simp at h
  fin x p e w h := by simp at h

theorem SI_push (o : FO F) (step : Step ε) (s : Nat) (st : DS ε F) (I : SI o step s st)
    (node far : Nat) (e : ε) (fw : F) (hn : node ∈ st.settled) (hs : s ∈ st.settled)
    (hmem : (far, e) ∈ step node) (h1 : far ∉ st.settled) (hfin : o.fin fw = true) :
    SI o step s { st with labels := (far, node, e, fw) :: st.labels, heap := st.heap ++ [⟨fw, far⟩] } := by
  refine ⟨I.nodup, I.hs, fun h => by have h' : st.settled = [] := h; rw [h'] at hn; simp at hn, ?_, ?_, ?_, ?_⟩
  · intro it hit
    rcases List.mem_append.mp hit with hit | hit
    · rcases I.heapN it hit with h | ⟨p, e', w', hm⟩
      · exact Or.inl h
      · exact Or.inr ⟨p, e', w', List.mem_cons_of_mem _ hm⟩
    · simp at hit; subst hit; exact Or.inr ⟨node, e, fw, by simp⟩
  · intro x hx hxs
    obtain ⟨p, e', w', hm⟩ := I.setLbl x hx hxs
    exact ⟨p, e', w', List.mem_cons_of_mem _ hm⟩
  · intro x p e' w' hm
    rcases List.mem_cons.mp hm with h | h
    · simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      refine ⟨hn, hmem, fun hx => h1 (hx ▸ hs), fun hx => absurd hx h1⟩
    · exact I.ent x p e' w' h
  · intro x p e' w' hm
    rcases List.mem_cons.mp hm with h | h
    · simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact hfin
    · exact I.fin x p e' w' h

theorem relax_SI (o : FO F) (step : Step ε) (s : Nat) (wf : ε → F) (node : Nat) (w : F) :
    ∀ (l pr : List (Nat × ε)) (st : DS ε F), step node = pr ++ l → SI o step s st →
      node ∈ st.settled → s ∈ st.settled →
      SI o step s (relax o wf node w l st) ∧ (relax o wf node w l st).settled = st.settled := by
  intro l
  induction l with
  | nil => intro pr st _ I _ _; exact ⟨I, rfl⟩
  | cons y r ih =>
    intro pr st hl I hn hs
    obtain ⟨far, e⟩ := y
    have hl' : step node = (pr ++ [(far, e)]) ++ r := by rw [hl]; simp
    have hmem : (far, e) ∈ step node := by rw [hl]; simp
    simp only [relax]
    by_cases h1 : far ∈ st.settled
    · simp only [h1, ↓reduceIte]; exact ih _ st hl' I hn hs
    · simp only [h1, ↓reduceIte]
      by_cases h2 : o.fin (o.add w (wf e)) = false
      · simp only [h2, ↓reduceIte]; exact ih _ st hl' I hn hs
      · have h2' : o.fin (o.add w (wf e)) = true := by simpa using h2
        simp only [h2, Bool.false_eq_true, ↓reduceIte]
        by_cases h3 : (match lblGet st.labels far with | none => true | some (_, _, lw) => o.lt (o.add w (wf e)) lw) = true
        · simp only [h3, ↓reduceIte]
          have I' := SI_push o step s st I node far e (o.add w (wf e)) hn hs hmem h1 h2'
          exact ih _ _ hl' I' hn hs
        · simp only [h3, Bool.false_eq_true, ↓reduceIte]
          exact ih _ st hl' I hn hs

theorem settle_SI (o : FO F) (step : Step ε) (s : Nat) (st : DS ε F) (I : SI o step s st)
    (it : Item F) (hit : it ∈ st.heap) (rest : List (Item F)) (hrest : ∀ y ∈ rest, y ∈ st.heap)
    (hn : it.node ∉ st.settled) :
    SI o step s { st with settled := it.node :: st.settled, heap := rest } := by
  have hnode : it.node = s ∨ ∃ p e w, (it.node, p, e, w) ∈ st.labels := I.heapN it hit
  refine ⟨List.nodup_cons.mpr ⟨hn, I.nodup⟩, ?_, fun h => by simp at h, ?_, ?_, ?_, ?_⟩
  · rcases I.hs with h0 | h0
    · obtain ⟨_, w, hh⟩ := I.init h0
      rw [hh] at hit
      simp at hit
      subst hit
      exact Or.inr (by simp)
    · exact Or.inr (List.mem_cons_of_mem _ h0)
  · intro y hy; exact I.heapN y (hrest y hy)
  · intro x hx hxs
    rcases List.mem_cons.mp hx with rfl | hx
    · rcases hnode with h | h
      · exact absurd h hxs
      · exact h
    · exact I.setLbl x hx hxs
  · intro x p e w hm
    obtain ⟨hp, hed, hxs, hidx⟩ := I.ent x p e w hm
    refine ⟨List.mem_cons_of_mem _ hp, hed, hxs, fun hx => ?_⟩
    have hpn : it.node ≠ p := fun h => hn (h ▸ hp)
    have hidp : (it.node :: st.settled).idxOf p = st.settled.idxOf p + 1 := by simp [List.idxOf_cons, hpn]
    rcases List.mem_cons.mp hx with rfl | hx
    · simp [List.idxOf_cons_self, hidp]
    · have hxn : it.node ≠ x := fun h => hn (h ▸ hx)
      have : (it.node :: st.settled).idxOf x = st.settled.idxOf x + 1 := by simp [List.idxOf_cons, hxn]
      rw [this, hidp]
      have := hidx hx
      omega
  · intro x p e w hm; exact I.fin x p e w hm

theorem loop_SI (o : FO F) (pop : List (Item F) → Option (Item F × List (Item F))) (hp : PopOk pop)
    (step : Step ε) (s : Nat) (wf : ε → F) (t : Nat) :
    ∀ (fuel : Nat) (st : DS ε F) (st' : DS ε F) (r : Bool), SI o step s st →
      dijkLoop o pop step wf t fuel st = some (st', r) →
      SI o step s st' ∧ (r = true → t ∈ st'.settled) ∧ (r = false → t ∉ st.settled → t ∉ st'.settled) := by
  intro fuel
  induction fuel with
  | zero => intro st st' r _ h; simp [dijkLoop] at h
  | succ n ih =>
    intro st st' r I h
    simp only [dijkLoop] at h
    rcases hq : pop st.heap with _ | ⟨it, rest⟩
    · rw [hq] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨I, by simp, fun _ h => h⟩
    · rw [hq] at h
      simp only at h
      have hperm := hp.2 _ _ _ hq
      have hit : it ∈ st.heap := hperm.subset (by simp)
      have hrest : ∀ y ∈ rest, y ∈ st.heap := fun y hy => hperm.subset (List.mem_cons_of_mem _ hy)
      by_cases hs : it.node ∈ st.settled
      · simp only [hs, ↓reduceIte] at h
        have I1 : SI o step s { st with heap := rest } :=
          ⟨I.nodup, I.hs, fun h0 => by
            have h0' : st.settled = [] := h0
            rw [h0'] at hs; simp at hs, fun y hy => I.heapN y (hrest y hy), I.setLbl, I.ent, I.fin⟩
        exact ih { st with heap := rest } st' r I1 h
      · simp only [hs, ↓reduceIte] at h
        have I2 := settle_SI o step s st I it hit rest hrest hs
        have hs2 : s ∈ it.node :: st.settled := by
          rcases I2.hs with h0 | h0
          · simp at h0
          · exact h0
        by_cases ht : (it.node == t) = true
        · simp only [ht, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          refine ⟨I2, fun _ => ?_, fun h => absurd h (by simp)⟩
          have : it.node = t := by simpa using ht
          simp [this]
        · simp only [ht, Bool.false_eq_true, ↓reduceIte] at h
          have hr := relax_SI o step s wf it.node it.weight (step it.node) [] _ (by simp) I2 (by simp) hs2
          obtain ⟨I3, hset3, hmem3⟩ := (ih _ st' r hr.1 h)
          refine ⟨I3, hset3, fun hf htn => hmem3 hf ?_⟩
          rw [hr.2]
          have : t ≠ it.node := fun e => ht (by simp [e])
          simp [this, htn]


/-! ## The reported path -/

theorem lblGet_mem (L : List (Nat × Nat × ε × F)) (x p : Nat) (e : ε) (w : F)
    (h : lblGet L x = some (p, e, w)) : (x, p, e, w) ∈ L := by
  unfold lblGet at h
  cases hf : L.find? (fun q => q.1 == x) with
  | none => rw [hf] at h; simp at h
  | some q =>
    rw [hf] at h
    obtain ⟨x', p', e', w'⟩ := q
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    have h2 := List.find?_some hf
    simp only [beq_iff_eq] at h2
    subst h2
    exact List.mem_of_find?_eq_some hf

theorem lblGet_exists (L : List (Nat × Nat × ε × F)) (x : Nat) (h : ∃ p e w, (x, p, e, w) ∈ L) :
    ∃ p e w, lblGet L x = some (p, e, w) := by
  obtain ⟨p, e, w, hm⟩ := h
  unfold lblGet
  cases hf : L.find? (fun q => q.1 == x) with
  | none => have := List.find?_eq_none.mp hf _ hm; simp at this
  | some q => exact ⟨q.2.1, q.2.2.1, q.2.2.2, by simp⟩

theorem find_comp (L : List (Nat × Nat × ε × F)) (x : Nat) :
    List.find? ((fun q : Nat × Nat × ε => q.1 == x) ∘ (fun q : Nat × Nat × ε × F => (q.1, q.2.1, q.2.2.1))) L
      = L.find? (fun q => q.1 == x) := rfl

theorem lookup_eq (L : List (Nat × Nat × ε × F)) (x p : Nat) (e : ε) (w : F)
    (h : lblGet L x = some (p, e, w)) : lookupP (parOf L) x = some (p, e) := by
  unfold lblGet at h
  unfold lookupP parOf
  rw [List.find?_map, find_comp]
  cases hf : L.find? (fun q => q.1 == x) with
  | none => rw [hf] at h; simp at h
  | some q =>
    rw [hf] at h
    obtain ⟨x', p', e', w'⟩ := q
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h ⊢
    exact ⟨h.1, h.2.1⟩

theorem lookup_eq_rev (L : List (Nat × Nat × ε × F)) (x p : Nat) (e : ε)
    (h : lookupP (parOf L) x = some (p, e)) : ∃ w, lblGet L x = some (p, e, w) := by
  unfold lookupP parOf at h
  rw [List.find?_map, find_comp] at h
  unfold lblGet
  cases hf : L.find? (fun q => q.1 == x) with
  | none => rw [hf] at h; simp at h
  | some q =>
    rw [hf] at h
    obtain ⟨x', p', e', w'⟩ := q
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    exact ⟨w', by simp [h.1, h.2]⟩

theorem chain_self (s : Nat) (par : List (Nat × Nat × ε)) (f : Nat) : chain s par f s = some [] := by
  cases f <;> simp [chain]

theorem chainD_ok (step : Step ε) (s : Nat) (L : List (Nat × Nat × ε × F)) (S : List Nat)
    (hent : ∀ x p e w, (x, p, e, w) ∈ L → p ∈ S ∧ (x, e) ∈ step p ∧ x ≠ s ∧
        (x ∈ S → S.idxOf x < S.idxOf p))
    (hset : ∀ x ∈ S, x ≠ s → ∃ p e w, (x, p, e, w) ∈ L) (hs : s ∈ S) :
    ∀ (n x : Nat), x ∈ S → S.length - S.idxOf x ≤ n → ∀ fuel, n ≤ fuel →
      ∃ es, chain s (parOf L) fuel x = some es ∧ WalkTo step s es x ∧
        (nodesOf s es).Pairwise (fun a b => S.idxOf b < S.idxOf a) ∧
        ∀ a ∈ nodesOf s es, S.idxOf x ≤ S.idxOf a := by
  intro n
  induction n with
  | zero =>
    intro x hx hn
    have := List.idxOf_lt_length_of_mem hx
    omega
  | succ n ih =>
    intro x hx hn fuel hf
    by_cases hxs : x = s
    · subst hxs
      refine ⟨[], chain_self _ _ _, rfl, by simp [nodesOf], by simp [nodesOf]⟩
    · obtain ⟨p0, e0, w0, hm0⟩ := hset x hx hxs
      obtain ⟨p, e, w, hl⟩ := lblGet_exists L x ⟨p0, e0, w0, hm0⟩
      have hm := lblGet_mem L x p e w hl
      obtain ⟨hpS, hed, _, hidx⟩ := hent x p e w hm
      have hlt := hidx hx
      have hplen := List.idxOf_lt_length_of_mem hpS
      obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      obtain ⟨es, hes, hw, hpw, hlast⟩ := ih p hpS (by omega) f (by omega)
      refine ⟨es ++ [(x, e)], ?_, ?_, ?_, ?_⟩
      · simp only [chain, hxs, ↓reduceIte, lookup_eq L x p e w hl, hes, Option.map_some]
      · exact (walk_append step s es _ x).mpr ⟨p, hw, by simpa [WalkTo] using hed⟩
      · have : nodesOf s (es ++ [(x, e)]) = nodesOf s es ++ [x] := by simp [nodesOf]
        rw [this, List.pairwise_append]
        refine ⟨hpw, by simp, fun a ha b hb => ?_⟩
        simp at hb; subst hb
        have := hlast a ha
        omega
      · have : nodesOf s (es ++ [(x, e)]) = nodesOf s es ++ [x] := by simp [nodesOf]
        rw [this]
        intro a ha
        rcases List.mem_append.mp ha with ha | ha
        · have := hlast a ha; omega
        · simp at ha; subst ha; omega

theorem chain_weight (o : FO F) (wf : ε → F) (s : Nat) (L : List (Nat × Nat × ε × F))
    (hnos : ∀ x p e w, (x, p, e, w) ∈ L → x ≠ s)
    (hW : ∀ x p e w, (x, p, e, w) ∈ L →
      (p = s ∧ w = o.add o.zero (wf e)) ∨ ∃ p' e' wp, lblGet L p = some (p', e', wp) ∧ w = o.add wp (wf e)) :
    ∀ (fuel x : Nat) (es : List (Nat × ε)), chain s (parOf L) fuel x = some es →
      (x = s → es = []) ∧ (x ≠ s → ∃ p e w, lblGet L x = some (p, e, w) ∧ fsum o.add o.zero wf es = w) := by
  intro fuel
  induction fuel with
  | zero =>
    intro x es h
    simp only [chain] at h
    by_cases hx : x = s
    · simp only [hx, ↓reduceIte, Option.some.injEq] at h; exact ⟨fun _ => h.symm, fun h' => absurd hx h'⟩
    · simp [hx] at h
  | succ f ih =>
    intro x es h
    by_cases hx : x = s
    · simp only [chain, hx, ↓reduceIte, Option.some.injEq] at h
      exact ⟨fun _ => h.symm, fun h' => absurd hx h'⟩
    · refine ⟨fun h' => absurd h' hx, fun _ => ?_⟩
      simp only [chain, hx, ↓reduceIte] at h
      rcases hq : lookupP (parOf L) x with _ | ⟨p, e⟩
      · rw [hq] at h; simp at h
      · rw [hq] at h
        simp only at h
        rcases hc : chain s (parOf L) f p with _ | es'
        · rw [hc] at h; simp at h
        · rw [hc] at h
          simp only [Option.map_some, Option.some.injEq] at h
          subst h
          obtain ⟨w, hl⟩ := lookup_eq_rev L x p e hq
          refine ⟨p, e, w, hl, ?_⟩
          have hm := lblGet_mem L x p e w hl
          rcases hW x p e w hm with ⟨hps, hw⟩ | ⟨p', e', wp, hlp, hw⟩
          · subst hps
            have := (ih _ _ hc).1 rfl
            subst this
            simp [fsum, hw]
          · have hps : p ≠ s := by
              intro hps
              subst hps
              have hm' := lblGet_mem L p p' e' wp hlp
              exact hnos p p' e' wp hm' rfl
            obtain ⟨pp, ee, ww, hl2, hf2⟩ := (ih _ _ hc).2 hps
            rw [hlp] at hl2
            simp only [Option.some.injEq, Prod.mk.injEq] at hl2
            obtain ⟨_, _, rfl⟩ := hl2
            rw [fsum_append, hf2, hw]
            simp [fsum]


/-- **PROVEN** (`dijkstra_single_path`, structure; any `F`, any heap honouring `PopOk`): a returned
`(edges, weight, cost)` is a walk `source → target` (`target ≠ source`: the panicking
`labels[&target]` index is `none` here) over edges of `step`, visiting distinct nodes; its `cost` is
the left fold `0.0 + c e₁ + c e₂ + …` over those edges in path order; its `weight` is the target's
label and is finite (`is_finite` gates every label). The function never looks at `maxLen` /
`maxCost` (the dispatch guard in `run_path_algo` only sends it `maxCost = ∅`, `maxLen = u32::MAX`). -/
theorem dijkstra_valid (o : FO F) (pop : List (Item F) → Option (Item F × List (Item F))) (hp : PopOk pop)
    (step : Step ε) (s t : Nat) (wf cf : ε → F) (fuel : Nat)
    (es : List (Nat × ε)) (w c : F)
    (h : dijkstra o pop step s t wf cf fuel = some (some (es, w, c))) :
    t ≠ s ∧ WalkTo step s es t ∧ (nodesOf s es).Nodup ∧ c = fsum o.add o.zero cf es ∧ o.fin w = true := by
  unfold dijkstra at h
  rcases hl : dijkLoop o pop step wf t fuel (⟨[], [], [⟨o.zero, s⟩]⟩ : DS ε F) with _ | ⟨st, r⟩
  · rw [hl] at h; simp at h
  · rw [hl] at h
    cases r with
    | false => simp at h
    | true =>
      simp only at h
      obtain ⟨I, htS, _⟩ := loop_SI o pop hp step s wf t fuel _ st true (SI_init o step s) hl
      have htS := htS rfl
      rcases hq : lblGet st.labels t with _ | ⟨p, e', w''⟩
      · rw [hq] at h
        rcases hc : chain s (parOf st.labels) st.settled.length t with _ | es0 <;> rw [hc] at h <;> simp at h
      · have hm := lblGet_mem st.labels t p e' w'' hq
        obtain ⟨_, _, hts, _⟩ := I.ent t p e' w'' hm
        have hs : s ∈ st.settled := by
          rcases I.hs with h0 | h0
          · rw [h0] at htS; simp at htS
          · exact h0
        obtain ⟨es0, hch, hw, hpw, _⟩ := chainD_ok step s st.labels st.settled I.ent I.setLbl hs
          st.settled.length t htS (by omega) st.settled.length (Nat.le_refl _)
        rw [hq, hch] at h
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        refine ⟨hts, hw, ?_, rfl, I.fin t p e' w'' hm⟩
        exact hpw.imp (fun {a b} h1 (hab : a = b) => by subst hab; omega)

end OpsTraverse.Dijkstra
