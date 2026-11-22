import OpsTraverse.SearchBase
/-
# `ExprEval::bfs_shortest_path` (graph/src/runtime/eval.rs): bidirectional BFS

`nf` / `nb` are `fwd_nbrs.neighbors` / `bwd_nbrs.neighbors` (`NeighborIter`, per node, already sorted
and deduplicated; the iterators are stateless between calls). `f_map`/`b_map` are `FxHashMap<u64,
(parent, depth)>`, modelled as lists, newest first, read = first match (a key is only inserted when
absent). The `while meet.is_none()` loop is fuelled (`fuel` levels); theorems cover finished runs.

| here | there |
| --- | --- |
| `scanN` | the inner `for &nb in nbrs.neighbors(cur)` (insert into `own`, meet check, else `next.push`) |
| `scanFront` | `for &cur in front` |
| `bbLoop` | `while meet.is_none() { … }` incl. the three early `return Null` conditions |
| `chainN` | the two reconstruction `while` loops (fuel `maxLevel`: the chains are at most `maxLevel` long, proved in `bspNodes_spec`) |
| `bspNodes` | everything up to the `min_hops` check |
-/
namespace OpsTraverse.BidirBfs
open OpsTraverse.Search

abbrev DM := List (Nat × Nat × Nat)

def dget (m : DM) (k : Nat) : Option (Nat × Nat) := (m.find? (fun q => q.1 == k)).map (·.2)

structure LS where
  own : DM
  next : List Nat
  meet : Option (Nat × Nat)

def scanN (other : DM) (depth cur : Nat) : List Nat → LS → LS
  | [], s => s
  | nb :: r, s =>
    if (dget s.own nb).isSome then scanN other depth cur r s
    else
      match dget other nb with
      | some (_, od) =>
        scanN other depth cur r { s with own := (nb, cur, depth) :: s.own, meet := if s.meet.all (fun bo => od < bo.1) then some (od, nb) else s.meet }
      | none => scanN other depth cur r { s with own := (nb, cur, depth) :: s.own, next := s.next ++ [nb] }

def scanFront (nbrs : Nat → List Nat) (other : DM) (depth : Nat) : List Nat → LS → LS
  | [], s => s
  | cur :: r, s => scanFront nbrs other depth r (scanN other depth cur (nbrs cur) s)

structure BB where
  fm : DM
  bm : DM
  ff : List Nat
  bf : List Nat
  df : Nat
  db : Nat

/-- `none` = out of fuel; `some none` = `Null`; `some (some (st, od, n))` = a meeting node `n` whose
other-side depth is `od`, with the maps after the last level. -/
def bbLoop (nf nb : Nat → List Nat) (maxLevel : Nat) : Nat → BB → Option (Option (BB × Nat × Nat))
  | 0, _ => none
  | n + 1, st =>
    if st.ff.isEmpty || st.bf.isEmpty || decide (maxLevel ≤ st.df + st.db) then some none
    else if st.ff.length ≤ st.bf.length then
      let r := scanFront nf st.bm (st.df + 1) st.ff ⟨st.fm, [], none⟩
      let st' : BB := { st with fm := r.own, ff := r.next, df := st.df + 1 }
      match r.meet with
      | some m => some (some (st', m))
      | none => bbLoop nf nb maxLevel n st'
    else
      let r := scanFront nb st.fm (st.db + 1) st.bf ⟨st.bm, [], none⟩
      let st' : BB := { st with bm := r.own, bf := r.next, db := st.db + 1 }
      match r.meet with
      | some m => some (some (st', m))
      | none => bbLoop nf nb maxLevel n st'

/-- Parent chain from `cur` to `stop` (nodes after `cur`, ending with `stop`). -/
def chainN (m : DM) (stop : Nat) : Nat → Nat → Option (List Nat)
  | 0, cur => if cur = stop then some [] else none
  | f + 1, cur =>
    if cur = stop then some [] else
    match dget m cur with
    | none => none
    | some (p, _) => (chainN m stop f p).map (p :: ·)

/-- `bfs_shortest_path` up to the node list: `none` is `Value::Null` (or out of fuel in the outer
`Option`). -/
def bspNodes (nf nb : Nat → List Nat) (src dst maxLevel fuel : Nat) : Option (Option (List Nat)) :=
  if src = dst then some none else
  match bbLoop nf nb maxLevel fuel ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ with
  | none => none
  | some none => some none
  | some (some (st, _, n)) =>
    match chainN st.fm src maxLevel n, chainN st.bm dst maxLevel n with
    | some fw, some bw => some (some ((n :: fw).reverse ++ bw))
    | _, _ => some none

/-! ## Invariants -/

/-- The adjacency as a `Step`. -/
def gs (g : Nat → List Nat) : Step Unit := fun u => (g u).map (fun v => (v, ()))

theorem mem_gs (g : Nat → List Nat) (u v : Nat) (e : Unit) : (v, e) ∈ gs g u ↔ v ∈ g u := by
  simp [gs]

theorem dget_mem (m : DM) (x p k : Nat) (h : dget m x = some (p, k)) : (x, p, k) ∈ m := by
  unfold dget at h
  cases hf : m.find? (fun q => q.1 == x) with
  | none => rw [hf] at h; simp at h
  | some q =>
    rw [hf] at h
    obtain ⟨x', p', k'⟩ := q
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have h2 := List.find?_some hf
    simp only [beq_iff_eq] at h2
    subst h2
    exact List.mem_of_find?_eq_some hf

theorem dget_isSome_iff (m : DM) (x : Nat) : (dget m x).isSome = true ↔ ∃ p k, (x, p, k) ∈ m := by
  constructor
  · intro h
    cases hg : dget m x with
    | none => rw [hg] at h; simp at h
    | some q => exact ⟨q.1, q.2, dget_mem m x q.1 q.2 hg⟩
  · rintro ⟨p, k, hm⟩
    unfold dget
    cases hf : m.find? (fun q => q.1 == x) with
    | none => have := List.find?_eq_none.mp hf _ hm; simp at this
    | some q => simp

theorem dget_isNone_iff (m : DM) (x : Nat) : (dget m x).isNone = true ↔ ¬ ∃ p k, (x, p, k) ∈ m := by
  rw [← dget_isSome_iff]; cases dget m x <;> simp

structure SI' (g : Nat → List Nat) (r : Nat) (m : DM) (fr : List Nat) (d : Nat) : Prop where
  keys : ∀ x, (∃ p k, (x, p, k) ∈ m) ↔ ∃ j, j ≤ d ∧ R (gs g) r j x
  front : ∀ x, x ∈ fr ↔ Dst (gs g) r x d
  ent : ∀ x p k, (x, p, k) ∈ m → Dst (gs g) r x k ∧
    ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1)))

theorem SI'_init (g : Nat → List Nat) (r : Nat) : SI' g r [(r, r, 0)] [r] 0 where
  keys x := by
    simp only [List.mem_singleton, Prod.mk.injEq]
    constructor
    · rintro ⟨p, k, h1, _⟩; exact ⟨0, Nat.le_refl _, (R_zero _ r x).mpr h1⟩
    · rintro ⟨j, hj, hr⟩
      have : j = 0 := by omega
      subst this
      exact ⟨r, 0, (R_zero _ r x).mp hr, rfl, rfl⟩
  front x := by simp [dist_zero]
  ent x p k h := by
    simp only [List.mem_singleton, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨(dist_zero _ _ _).mpr rfl, Or.inl ⟨rfl, rfl, rfl⟩⟩

def Old (g : Nat → List Nat) (r d : Nat) (x : Nat) : Prop := ∃ j, j ≤ d ∧ R (gs g) r j x

/-- Mid-scan state: `Disc` are the nodes some already-scanned frontier node points to. -/
structure MS (g : Nat → List Nat) (r d : Nat) (other : DM) (Disc : Nat → Prop) (s : LS) : Prop where
  keys : ∀ x, (∃ p k, (x, p, k) ∈ s.own) ↔ Old g r d x ∨ Disc x
  ent : ∀ x p k, (x, p, k) ∈ s.own → Dst (gs g) r x k ∧
    ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1)))
  next : ∀ x, x ∈ s.next ↔ (Disc x ∧ ¬ Old g r d x ∧ ¬ ∃ p k, (x, p, k) ∈ other)
  mnone : s.meet = none → ∀ x, Disc x → ¬ Old g r d x → ¬ ∃ p k, (x, p, k) ∈ other
  msome : ∀ od n, s.meet = some (od, n) → Disc n ∧ ¬ Old g r d n ∧ (∃ q, dget other n = some (q, od)) ∧
    ∀ x, Disc x → ¬ Old g r d x → ∀ q od', dget other x = some (q, od') → od ≤ od'

theorem MS_congr (g : Nat → List Nat) (r d : Nat) (other : DM) (D1 D2 : Nat → Prop) (s : LS)
    (h : ∀ x, ¬ Old g r d x → (D1 x ↔ D2 x)) (I : MS g r d other D1 s) : MS g r d other D2 s where
  keys x := by
    rw [I.keys x]
    by_cases ho : Old g r d x
    · simp [ho]
    · simp [ho, h x ho]
  ent := I.ent
  next x := by
    rw [I.next x]
    by_cases ho : Old g r d x
    · simp [ho]
    · simp [ho, h x ho]
  mnone hm x hx ho := I.mnone hm x ((h x ho).mpr hx) ho
  msome od n hm := by
    obtain ⟨h1, h2, h3, h4⟩ := I.msome od n hm
    exact ⟨(h n h2).mp h1, h2, h3, fun x hx ho => h4 x ((h x ho).mpr hx) ho⟩

theorem scanN_spec (g : Nat → List Nat) (r d : Nat) (other : DM) (cur : Nat) (hcur : Dst (gs g) r cur d)
    (Disc : Nat → Prop) :
    ∀ (l pr : List Nat) (s : LS), g cur = pr ++ l → MS g r d other (fun x => Disc x ∨ x ∈ pr) s →
      MS g r d other (fun x => Disc x ∨ x ∈ g cur) (scanN other (d + 1) cur l s) := by
  intro l
  induction l with
  | nil =>
    intro pr s hl I
    simp only [scanN]
    rw [hl]
    simpa using I
  | cons nb rest ih =>
    intro pr s hl I
    have hl' : g cur = (pr ++ [nb]) ++ rest := by rw [hl]; simp
    have hmem : nb ∈ g cur := by rw [hl]; simp
    by_cases hown : (dget s.own nb).isSome = true
    · simp only [scanN, hown, ↓reduceIte]
      refine ih (pr ++ [nb]) s hl' (MS_congr g r d other _ _ s ?_ I)
      intro x hx
      simp only [List.mem_append, List.mem_singleton]
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
      · rintro (h | h | h)
        · exact Or.inl h
        · exact Or.inr h
        · subst h
          obtain ⟨p, k, hm⟩ := (dget_isSome_iff _ _).mp hown
          rcases (I.keys x).mp ⟨p, k, hm⟩ with h | h
          · exact absurd h hx
          · exact h
    · have hnoown : ¬ ∃ p k, (nb, p, k) ∈ s.own := fun h => hown ((dget_isSome_iff _ _).mpr h)
      have hk := (I.keys nb)
      have hnn : ¬ (Old g r d nb ∨ (Disc nb ∨ nb ∈ pr)) := fun h => hnoown (hk.mpr h)
      simp only [not_or] at hnn
      obtain ⟨hnold, hnD, hnpr⟩ := hnn
      have hD : Dst (gs g) r nb (d + 1) := by
        refine dist_succ (gs g) r cur nb d () hcur ((mem_gs g cur nb ()).mpr hmem) ?_
        intro j hj hrj; exact hnold ⟨j, hj, hrj⟩
      have hent : Dst (gs g) r nb (d + 1) ∧ ((nb = r ∧ cur = r ∧ d + 1 = 0) ∨ (1 ≤ d + 1 ∧ nb ∈ g cur ∧ Dst (gs g) r cur (d + 1 - 1))) :=
        ⟨hD, Or.inr ⟨by omega, hmem, by simpa using hcur⟩⟩
      have hDd : ∀ x, (Disc x ∨ x ∈ pr ++ [nb]) ↔ ((Disc x ∨ x ∈ pr) ∨ x = nb) := by
        intro x; simp only [List.mem_append, List.mem_singleton]
        constructor
        · rintro (h | h | h)
          · exact Or.inl (Or.inl h)
          · exact Or.inl (Or.inr h)
          · exact Or.inr h
        · rintro ((h | h) | h)
          · exact Or.inl h
          · exact Or.inr (Or.inl h)
          · exact Or.inr (Or.inr h)
      simp only [scanN, hown, Bool.false_eq_true, ↓reduceIte]
      rcases hq : dget other nb with _ | ⟨q, od⟩
      · simp only
        refine ih (pr ++ [nb]) _ hl' ⟨?_, ?_, ?_, ?_, ?_⟩
        · intro x
          show (∃ p k, (x, p, k) ∈ (nb, cur, d + 1) :: s.own) ↔ _
          constructor
          · rintro ⟨p, k, hm⟩
            rcases List.mem_cons.mp hm with h | h
            · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, _, _⟩ := h
              exact Or.inr ((hDd _).mpr (Or.inr rfl))
            · rcases (I.keys x).mp ⟨p, k, h⟩ with h | h
              · exact Or.inl h
              · exact Or.inr ((hDd _).mpr (Or.inl h))
          · rintro (h | h)
            · obtain ⟨p, k, hm⟩ := (I.keys x).mpr (Or.inl h); exact ⟨p, k, List.mem_cons_of_mem _ hm⟩
            · rcases (hDd _).mp h with h | rfl
              · obtain ⟨p, k, hm⟩ := (I.keys x).mpr (Or.inr h); exact ⟨p, k, List.mem_cons_of_mem _ hm⟩
              · exact ⟨cur, d + 1, by simp⟩
        · intro x p k hm
          rcases List.mem_cons.mp hm with h | h
          · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact hent
          · exact I.ent x p k h
        · intro x
          show x ∈ s.next ++ [nb] ↔ _
          have hnone : ¬ ∃ p k, (nb, p, k) ∈ other := (dget_isNone_iff _ _).mp (by rw [hq]; rfl)
          rw [List.mem_append, List.mem_singleton, I.next x, hDd]
          constructor
          · rintro (⟨h1, h2, h3⟩ | rfl)
            · exact ⟨Or.inl h1, h2, h3⟩
            · exact ⟨Or.inr rfl, hnold, hnone⟩
          · rintro ⟨h1, h2, h3⟩
            rcases h1 with h1 | rfl
            · exact Or.inl ⟨h1, h2, h3⟩
            · exact Or.inr rfl
        · intro hm x hx ho
          rcases (hDd x).mp hx with h | rfl
          · exact I.mnone hm x h ho
          · exact (dget_isNone_iff _ _).mp (by rw [hq]; rfl)
        · intro od n hm
          obtain ⟨h1, h2, h3, h4⟩ := I.msome od n hm
          refine ⟨(hDd n).mpr (Or.inl h1), h2, h3, fun x hx ho q' od' hg => ?_⟩
          rcases (hDd x).mp hx with h | rfl
          · exact h4 x h ho q' od' hg
          · rw [hq] at hg; simp at hg
      · simp only
        have hnoth : ∃ p k, (nb, p, k) ∈ other := ⟨q, od, dget_mem other nb q od hq⟩
        have hmeet_some : ∀ (s : LS), (if s.meet.all (fun bo => od < bo.1) = true then some (od, nb) else s.meet) ≠ none := by
          intro s'
          by_cases hc : s'.meet.all (fun bo => od < bo.1) = true
          · simp [hc]
          · simp only [hc, ↓reduceIte]
            cases hsm : s'.meet with
            | none => rw [hsm] at hc; simp at hc
            | some _ => simp
        refine ih (pr ++ [nb]) _ hl' ⟨?_, ?_, ?_, ?_, ?_⟩
        · intro x
          show (∃ p k, (x, p, k) ∈ (nb, cur, d + 1) :: s.own) ↔ _
          constructor
          · rintro ⟨p, k, hm⟩
            rcases List.mem_cons.mp hm with h | h
            · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, _, _⟩ := h
              exact Or.inr ((hDd _).mpr (Or.inr rfl))
            · rcases (I.keys x).mp ⟨p, k, h⟩ with h | h
              · exact Or.inl h
              · exact Or.inr ((hDd _).mpr (Or.inl h))
          · rintro (h | h)
            · obtain ⟨p, k, hm⟩ := (I.keys x).mpr (Or.inl h); exact ⟨p, k, List.mem_cons_of_mem _ hm⟩
            · rcases (hDd _).mp h with h | rfl
              · obtain ⟨p, k, hm⟩ := (I.keys x).mpr (Or.inr h); exact ⟨p, k, List.mem_cons_of_mem _ hm⟩
              · exact ⟨cur, d + 1, by simp⟩
        · intro x p k hm
          rcases List.mem_cons.mp hm with h | h
          · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact hent
          · exact I.ent x p k h
        · intro x
          show x ∈ s.next ↔ _
          rw [I.next x, hDd]
          constructor
          · rintro ⟨h1, h2, h3⟩; exact ⟨Or.inl h1, h2, h3⟩
          · rintro ⟨h1, h2, h3⟩
            rcases h1 with h1 | rfl
            · exact ⟨h1, h2, h3⟩
            · exact absurd hnoth h3
        · intro hm
          exact absurd hm (hmeet_some s)
        · intro od' n hm
          have hgn : ∀ q' od'', dget other nb = some (q', od'') → od'' = od := by
            intro q' od'' hg; rw [hq] at hg; simp at hg; exact hg.2.symm
          cases hsm : s.meet with
          | none =>
            simp only [hsm, Option.all_none, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hm
            obtain ⟨rfl, rfl⟩ := hm
            refine ⟨(hDd _).mpr (Or.inr rfl), hnold, ⟨q, hq⟩, fun x hx ho q' od'' hg => ?_⟩
            rcases (hDd x).mp hx with h | rfl
            · exact absurd ⟨q', od'', dget_mem other x q' od'' hg⟩ (I.mnone hsm x h ho)
            · have := hgn q' od'' hg; omega
          | some bm =>
            obtain ⟨bo, n0⟩ := bm
            by_cases hlt : od < bo
            · simp only [hsm, Option.all_some, hlt, decide_true, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hm
              obtain ⟨rfl, rfl⟩ := hm
              refine ⟨(hDd _).mpr (Or.inr rfl), hnold, ⟨q, hq⟩, fun x hx ho q' od'' hg => ?_⟩
              obtain ⟨_, _, _, h4⟩ := I.msome bo n0 hsm
              rcases (hDd x).mp hx with h | rfl
              · have := h4 x h ho q' od'' hg; omega
              · have := hgn q' od'' hg; omega
            · simp only [hsm, Option.all_some, hlt, decide_false, Bool.false_eq_true, ↓reduceIte] at hm
              simp only [Option.some.injEq, Prod.mk.injEq] at hm
              obtain ⟨rfl, rfl⟩ := hm
              obtain ⟨h1, h2, h3, h4⟩ := I.msome bo n0 hsm
              refine ⟨(hDd _).mpr (Or.inl h1), h2, h3, fun x hx ho q' od'' hg => ?_⟩
              rcases (hDd x).mp hx with h | rfl
              · exact h4 x h ho q' od'' hg
              · have := hgn q' od'' hg; omega

def Dsc (g : Nat → List Nat) (dn : List Nat) (x : Nat) : Prop := ∃ u ∈ dn, x ∈ g u

theorem scanFront_spec (g : Nat → List Nat) (r d : Nat) (other : DM) :
    ∀ (fl dn : List Nat) (s : LS), (∀ u ∈ fl, Dst (gs g) r u d) → MS g r d other (Dsc g dn) s →
      MS g r d other (Dsc g (fl.reverse ++ dn)) (scanFront g other (d + 1) fl s) := by
  intro fl
  induction fl with
  | nil => intro dn s _ I; simpa [scanFront] using I
  | cons cur ns ih =>
    intro dn s hfl I
    simp only [scanFront]
    have hcur := hfl cur (by simp)
    have h1 := scanN_spec g r d other cur hcur (Dsc g dn) (g cur) [] s (by simp)
      (MS_congr g r d other _ _ s (fun x _ => by simp) I)
    have h2 := ih (cur :: dn) _ (fun u hu => hfl u (List.mem_cons_of_mem _ hu))
      (MS_congr g r d other _ _ _ (fun x _ => by
        simp only [Dsc, List.mem_cons]
        constructor
        · rintro (⟨u, hu, hx⟩ | hx)
          · exact ⟨u, Or.inr hu, hx⟩
          · exact ⟨cur, Or.inl rfl, hx⟩
        · rintro ⟨u, rfl | hu, hx⟩
          · exact Or.inr hx
          · exact Or.inl ⟨u, hu, hx⟩) h1)
    simpa [List.reverse_cons, List.append_assoc] using h2

theorem expand_keys (g : Nat → List Nat) (r d : Nat) (fr : List Nat) (hfr : ∀ x, x ∈ fr ↔ Dst (gs g) r x d)
    (other : DM) (s : LS) (I : MS g r d other (Dsc g fr) s) :
    (∀ x, (∃ p k, (x, p, k) ∈ s.own) ↔ ∃ j, j ≤ d + 1 ∧ R (gs g) r j x) ∧
    (∀ x p k, (x, p, k) ∈ s.own → Dst (gs g) r x k ∧
      ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1)))) := by
  refine ⟨fun x => ?_, I.ent⟩
  rw [I.keys x]
  constructor
  · rintro (⟨j, hj, hr⟩ | ⟨u, hu, hx⟩)
    · exact ⟨j, by omega, hr⟩
    · exact ⟨d + 1, Nat.le_refl _, (R_succ (gs g) r d x).mpr ⟨u, (), ((hfr u).mp hu).1, (mem_gs g u x ()).mpr hx⟩⟩
  · rintro ⟨j, hj, hr⟩
    obtain ⟨dd, hdd, hD⟩ := exists_dist (gs g) r j x hr
    by_cases hle : dd ≤ d
    · exact Or.inl ⟨dd, hle, hD.1⟩
    · have : dd = d + 1 := by omega
      subst this
      obtain ⟨u, e, he, hu⟩ := dist_pred (gs g) r x d hD
      exact Or.inr ⟨u, (hfr u).mpr hu, (mem_gs g u x e).mp he⟩

/-! ## Walk plumbing -/

theorem nodesOf_last (g : Nat → List Nat) : ∀ (es : List (Nat × Unit)) (a x : Nat), WalkTo (gs g) a es x →
    (nodesOf a es).reverse.head? = some x := by
  intro es
  induction es with
  | nil => intro a x h; simp only [WalkTo] at h; subst h; simp [nodesOf]
  | cons y r ih =>
    intro a x h
    obtain ⟨_, hw⟩ := h
    have h1 := ih y.1 x hw
    have h2 : (nodesOf a (y :: r)).reverse = (nodesOf y.1 r).reverse ++ [a] := by simp [nodesOf]
    rw [h2, List.head?_append, h1]; rfl

/-- Reversing a walk along edges that are converse in the other adjacency. -/
theorem walk_rev (g1 g2 : Nat → List Nat) (hrv : ∀ u v, v ∈ g1 u → u ∈ g2 v) :
    ∀ (es : List (Nat × Unit)) (a b : Nat), WalkTo (gs g1) a es b →
      ∃ es', WalkTo (gs g2) b es' a ∧ es'.length = es.length ∧ nodesOf b es' = (nodesOf a es).reverse := by
  intro es
  induction es with
  | nil =>
    intro a b h
    simp only [WalkTo] at h; subst h
    exact ⟨[], rfl, rfl, by simp [nodesOf]⟩
  | cons x r ih =>
    intro a b h
    obtain ⟨hx, hw⟩ := h
    obtain ⟨es', hw', hl, hn⟩ := ih x.1 b hw
    have hxa : a ∈ g2 x.1 := hrv a x.1 (by
      obtain ⟨x1, x2⟩ := x
      simpa [mem_gs] using hx)
    refine ⟨es' ++ [(a, ())], (walk_append (gs g2) b es' _ a).mpr ⟨x.1, hw', ?_⟩, by simp [hl], ?_⟩
    · simp [WalkTo, mem_gs, hxa]
    · have := congrArg (· ++ [a]) hn
      simp only [nodesOf, List.map_append, List.map_cons, List.map_nil, List.reverse_cons] at this ⊢
      simpa using this

theorem prefix_dist (g : Nat → List Nat) (s x m : Nat) (es1 es2 : List (Nat × Unit))
    (h1 : WalkTo (gs g) s es1 m) (h2 : WalkTo (gs g) m es2 x) (hd : Dst (gs g) s x (es1.length + es2.length)) :
    Dst (gs g) s m es1.length := by
  refine ⟨⟨es1, rfl, h1⟩, fun j hj hr => ?_⟩
  obtain ⟨e0, hl0, hw0⟩ := hr
  exact hd.2 (j + es2.length) (by omega)
    ⟨e0 ++ es2, by simp [hl0], (walk_append (gs g) s e0 es2 x).mpr ⟨m, hw0, h2⟩⟩

def Key (m : DM) (x : Nat) : Prop := ∃ p k, (x, p, k) ∈ m

theorem disj_lb (gA gB : Nat → List Nat) (rA rB : Nat) (mA mB : DM) (frA frB : List Nat) (dA dB : Nat)
    (hrev : ∀ a es b, WalkTo (gs gA) a es b → ∃ es', WalkTo (gs gB) b es' a ∧ es'.length = es.length)
    (IA : SI' gA rA mA frA dA) (IB : SI' gB rB mB frB dB) (hdisj : ∀ x, Key mA x → ¬ Key mB x)
    (es : List (Nat × Unit)) (hw : WalkTo (gs gA) rA es rB) : dA + dB < es.length := by
  refine Classical.byContradiction fun hle => ?_
  have hle : es.length ≤ dA + dB := by omega
  have hsplit : WalkTo (gs gA) rA (es.take (min dA es.length) ++ es.drop (min dA es.length)) rB := by
    rw [List.take_append_drop]; exact hw
  obtain ⟨u, h1, h2⟩ := (walk_append (gs gA) rA _ _ rB).mp hsplit
  have hl1 : (es.take (min dA es.length)).length = min dA es.length := by simp
  have hl2 : (es.drop (min dA es.length)).length = es.length - min dA es.length := by simp
  obtain ⟨es', hw', hl'⟩ := hrev u _ rB h2
  have kA : Key mA u := (IA.keys u).mpr ⟨min dA es.length, by omega, ⟨_, hl1, h1⟩⟩
  have kB : Key mB u := (IB.keys u).mpr ⟨es.length - min dA es.length, by omega, ⟨es', by omega, hw'⟩⟩
  exact hdisj u kA kB

theorem chain_ok (g : Nat → List Nat) (r : Nat) (m : DM) (d : Nat)
    (hkeys : ∀ x, Key m x ↔ ∃ j, j ≤ d ∧ R (gs g) r j x)
    (hent : ∀ x p k, (x, p, k) ∈ m → Dst (gs g) r x k ∧
      ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1)))) :
    ∀ k, k ≤ d → ∀ x, Dst (gs g) r x k → ∀ fuel, k ≤ fuel →
      ∃ es l, WalkTo (gs g) r es x ∧ es.length = k ∧ chainN m r fuel x = some l ∧
        x :: l = (nodesOf r es).reverse := by
  intro k
  induction k with
  | zero =>
    intro _ x hx fuel _
    have hxr : x = r := (dist_zero _ r x).mp hx
    subst hxr
    refine ⟨[], [], rfl, rfl, ?_, by simp [nodesOf]⟩
    cases fuel <;> simp [chainN]
  | succ k ih =>
    intro hk x hx fuel hf
    have hxr : x ≠ r := R_ne_source _ r x (k + 1) hx (by omega)
    obtain ⟨p0, k0, hm0⟩ := (hkeys x).mpr ⟨k + 1, hk, hx.1⟩
    obtain ⟨p, k', hg⟩ : ∃ p k', dget m x = some (p, k') := by
      have := (dget_isSome_iff m x).mpr ⟨p0, k0, hm0⟩
      cases h : dget m x with
      | none => rw [h] at this; simp at this
      | some q => exact ⟨q.1, q.2, rfl⟩
    obtain ⟨hD, hd⟩ := hent x p k' (dget_mem m x p k' hg)
    have hk' : k' = k + 1 := dist_unique _ r x hD hx
    subst hk'
    rcases hd with ⟨h1, _⟩ | ⟨_, hxp, hp⟩
    · exact absurd h1 hxr
    · simp only [Nat.add_sub_cancel] at hp
      obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      obtain ⟨es', l', hw, hlen, hc, hnl⟩ := ih (by omega) p hp f (by omega)
      refine ⟨es' ++ [(x, ())], p :: l', (walk_append (gs g) r es' _ x).mpr ⟨p, hw, ?_⟩, by simp [hlen], ?_, ?_⟩
      · simp [WalkTo, mem_gs, hxp]
      · simp [chainN, hxr, hg, hc]
      · have h2 : nodesOf r (es' ++ [(x, ())]) = nodesOf r es' ++ [x] := by simp [nodesOf]
        rw [h2, List.reverse_append, ← hnl]; rfl

theorem expand_spec (g : Nat → List Nat) (r d : Nat) (fr : List Nat) (m other : DM) (I : SI' g r m fr d) :
    ((scanFront g other (d + 1) fr ⟨m, [], none⟩).meet = none →
      SI' g r (scanFront g other (d + 1) fr ⟨m, [], none⟩).own (scanFront g other (d + 1) fr ⟨m, [], none⟩).next (d + 1)) ∧
    (∀ x, (∃ p k, (x, p, k) ∈ (scanFront g other (d + 1) fr ⟨m, [], none⟩).own) ↔ ∃ j, j ≤ d + 1 ∧ R (gs g) r j x) ∧
    (∀ x p k, (x, p, k) ∈ (scanFront g other (d + 1) fr ⟨m, [], none⟩).own → Dst (gs g) r x k ∧
      ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1)))) ∧
    (∀ od n, (scanFront g other (d + 1) fr ⟨m, [], none⟩).meet = some (od, n) →
      Dst (gs g) r n (d + 1) ∧ (∃ q, dget other n = some (q, od)) ∧
      ∀ x, Dst (gs g) r x (d + 1) → ∀ q od', dget other x = some (q, od') → od ≤ od') := by
  have hs0 : MS g r d other (Dsc g []) (⟨m, [], none⟩ : LS) :=
    ⟨fun x => by simp [Dsc, I.keys x, Old], I.ent, fun x => by simp [Dsc],
     fun _ x hx => by simp [Dsc] at hx, fun od n h => by simp at h⟩
  have hM := scanFront_spec g r d other fr [] _ (fun u hu => (I.front u).mp hu) hs0
  have hM' : MS g r d other (Dsc g fr) (scanFront g other (d + 1) fr ⟨m, [], none⟩) :=
    MS_congr g r d other _ _ _ (fun x _ => by simp [Dsc]) hM
  obtain ⟨hk, he⟩ := expand_keys g r d fr I.front other _ hM'
  refine ⟨fun hm => ⟨hk, ?_, he⟩, hk, he, ?_⟩
  · intro x
    rw [hM'.next x]
    constructor
    · rintro ⟨⟨u, hu, hx⟩, hno, _⟩
      obtain ⟨hr⟩ : Nonempty (R (gs g) r (d + 1) x) :=
        ⟨(R_succ (gs g) r d x).mpr ⟨u, (), ((I.front u).mp hu).1, (mem_gs g u x ()).mpr hx⟩⟩
      exact ⟨hr, fun j hj hrj => hno ⟨j, by omega, hrj⟩⟩
    · intro hx
      obtain ⟨u, e, he', hu⟩ := dist_pred (gs g) r x d hx
      have hno : ¬ Old g r d x := fun ⟨j, hj, hrj⟩ => hx.2 j (by omega) hrj
      exact ⟨⟨u, (I.front u).mpr hu, (mem_gs g u x e).mp he'⟩, hno, hM'.mnone hm x ⟨u, (I.front u).mpr hu, (mem_gs g u x e).mp he'⟩ hno⟩
  · intro od n hm
    obtain ⟨⟨u, hu, hx⟩, hno, hq, hmin⟩ := hM'.msome od n hm
    refine ⟨⟨(R_succ (gs g) r d n).mpr ⟨u, (), ((I.front u).mp hu).1, (mem_gs g u n ()).mpr hx⟩,
      fun j hj hrj => hno ⟨j, by omega, hrj⟩⟩, hq, fun x hx' q od' hg => ?_⟩
    obtain ⟨u', e, he', hu'⟩ := dist_pred (gs g) r x d hx'
    have hno' : ¬ Old g r d x := fun ⟨j, hj, hrj⟩ => hx'.2 j (by omega) hrj
    exact hmin x ⟨u', (I.front u').mpr hu', (mem_gs g u' x e).mp he'⟩ hno' q od' hg

theorem expand_MS (g : Nat → List Nat) (r d : Nat) (fr : List Nat) (m other : DM) (I : SI' g r m fr d) :
    MS g r d other (Dsc g fr) (scanFront g other (d + 1) fr ⟨m, [], none⟩) := by
  have hs0 : MS g r d other (Dsc g []) (⟨m, [], none⟩ : LS) :=
    ⟨fun x => by simp [Dsc, I.keys x, Old], I.ent, fun x => by simp [Dsc],
     fun _ x hx => by simp [Dsc] at hx, fun od n h => by simp at h⟩
  have hM := scanFront_spec g r d other fr [] _ (fun u hu => (I.front u).mp hu) hs0
  exact MS_congr g r d other _ _ _ (fun x _ => by simp [Dsc]) hM

theorem expand_disj (g : Nat → List Nat) (r d : Nat) (fr : List Nat) (m other : DM) (I : SI' g r m fr d)
    (hm : (scanFront g other (d + 1) fr ⟨m, [], none⟩).meet = none) :
    ∀ x, Key (scanFront g other (d + 1) fr ⟨m, [], none⟩).own x → Key m x ∨ ¬ Key other x := by
  have hM := expand_MS g r d fr m other I
  intro x hx
  rcases (hM.keys x).mp hx with ho | hd
  · exact Or.inl ((I.keys x).mpr ho)
  · by_cases ho : Old g r d x
    · exact Or.inl ((I.keys x).mpr ho)
    · exact Or.inr (hM.mnone hm x hd ho)

/-- Any walk `rA → rB` is at least as long as the path the meeting node gives. -/
theorem meet_min (gA gB : Nat → List Nat) (rA rB : Nat) (mA mB : DM) (frA frB : List Nat) (dA dB : Nat)
    (hrev : ∀ a es b, WalkTo (gs gA) a es b → ∃ es', WalkTo (gs gB) b es' a ∧ es'.length = es.length)
    (IA : SI' gA rA mA frA dA) (IB : SI' gB rB mB frB dB) (hdisj : ∀ x, Key mA x → ¬ Key mB x)
    (od n q : Nat) (hq : dget mB n = some (q, od))
    (hmin : ∀ x, Dst (gs gA) rA x (dA + 1) → ∀ q od', dget mB x = some (q, od') → od ≤ od')
    (es : List (Nat × Unit)) (hw : WalkTo (gs gA) rA es rB) : dA + 1 + od ≤ es.length := by
  obtain ⟨D, hD, hDD⟩ := exists_dist (gs gA) rA es.length rB ⟨es, rfl, hw⟩
  obtain ⟨esD, hlD, hwD⟩ := hDD.1
  have hlb := disj_lb gA gB rA rB mA mB frA frB dA dB hrev IA IB hdisj esD hwD
  rw [hlD] at hlb
  have hsplit : WalkTo (gs gA) rA (esD.take (dA + 1) ++ esD.drop (dA + 1)) rB := by
    rw [List.take_append_drop]; exact hwD
  obtain ⟨u, h1, h2⟩ := (walk_append (gs gA) rA _ _ rB).mp hsplit
  have hl1 : (esD.take (dA + 1)).length = dA + 1 := by simp; omega
  have hl2 : (esD.drop (dA + 1)).length = D - (dA + 1) := by simp [hlD]
  have hsum : (esD.take (dA + 1)).length + (esD.drop (dA + 1)).length = D := by omega
  have hpd : Dst (gs gA) rA u (dA + 1) := by
    have := prefix_dist gA rA rB u _ _ h1 h2 (by rw [hsum]; exact hDD)
    rwa [hl1] at this
  obtain ⟨es', hw', hl'⟩ := hrev u _ rB h2
  by_cases hc : D - (dA + 1) ≤ dB
  · have hku : Key mB u := (IB.keys u).mpr ⟨D - (dA + 1), hc, ⟨es', hl'.trans hl2, hw'⟩⟩
    obtain ⟨p', k', hg⟩ : ∃ p k', dget mB u = some (p, k') := by
      have := (dget_isSome_iff mB u).mpr hku
      cases h : dget mB u with
      | none => rw [h] at this; simp at this
      | some q => exact ⟨q.1, q.2, rfl⟩
    have hdk := (IB.ent u p' k' (dget_mem mB u p' k' hg)).1
    have hkle : k' ≤ D - (dA + 1) := by
      refine Classical.byContradiction fun hlt => hdk.2 (D - (dA + 1)) (by omega) ⟨es', hl'.trans hl2, hw'⟩
    have := hmin u hpd p' k' hg
    omega
  · have hn := (IB.ent n q od (dget_mem mB n q od hq)).1
    obtain ⟨p1, k1, hm1⟩ : ∃ p k, (n, p, k) ∈ mB := ⟨q, od, dget_mem mB n q od hq⟩
    obtain ⟨j, hj, hrj⟩ := (IB.keys n).mp ⟨p1, k1, hm1⟩
    have : od ≤ j := Classical.byContradiction fun hlt => hn.2 j (by omega) hrj
    omega

theorem depth_le (g : Nat → List Nat) (r : Nat) (m : DM) (fr : List Nat) (d : Nat) (I : SI' g r m fr d)
    (n q od : Nat) (hq : dget m n = some (q, od)) : od ≤ d := by
  have hn := (I.ent n q od (dget_mem m n q od hq)).1
  obtain ⟨j, hj, hrj⟩ := (I.keys n).mp ⟨q, od, dget_mem m n q od hq⟩
  exact Classical.byContradiction fun hlt => hn.2 j (by omega) hrj |>.elim

theorem dget_depth (g : Nat → List Nat) (r : Nat) (m : DM)
    (hent : ∀ x p k, (x, p, k) ∈ m → Dst (gs g) r x k ∧
      ((x = r ∧ p = r ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ g p ∧ Dst (gs g) r p (k - 1))))
    (n k : Nat) (hD : Dst (gs g) r n k) (hk : Key m n) : ∃ p, dget m n = some (p, k) := by
  obtain ⟨p', k', hg⟩ : ∃ p k', dget m n = some (p, k') := by
    have := (dget_isSome_iff m n).mpr hk
    cases h : dget m n with
    | none => rw [h] at this; simp at this
    | some q => exact ⟨q.1, q.2, rfl⟩
  have := dist_unique _ r n (hent n p' k' (dget_mem m n p' k' hg)).1 hD
  subst this
  exact ⟨p', hg⟩

theorem front_nonempty (gA gB : Nat → List Nat) (rA rB : Nat) (mA mB : DM) (frA frB : List Nat) (dA dB : Nat)
    (hrev : ∀ a es b, WalkTo (gs gA) a es b → ∃ es', WalkTo (gs gB) b es' a ∧ es'.length = es.length)
    (IA : SI' gA rA mA frA dA) (IB : SI' gB rB mB frB dB) (hdisj : ∀ x, Key mA x → ¬ Key mB x)
    (es : List (Nat × Unit)) (hw : WalkTo (gs gA) rA es rB) : ∃ u, u ∈ frA := by
  obtain ⟨D, hD, hDD⟩ := exists_dist (gs gA) rA es.length rB ⟨es, rfl, hw⟩
  obtain ⟨esD, hlD, hwD⟩ := hDD.1
  have hlb := disj_lb gA gB rA rB mA mB frA frB dA dB hrev IA IB hdisj esD hwD
  rw [hlD] at hlb
  have hsplit : WalkTo (gs gA) rA (esD.take dA ++ esD.drop dA) rB := by
    rw [List.take_append_drop]; exact hwD
  obtain ⟨u, h1, h2⟩ := (walk_append (gs gA) rA _ _ rB).mp hsplit
  have hl1 : (esD.take dA).length = dA := by simp; omega
  have hl2 : (esD.drop dA).length = D - dA := by simp [hlD]
  have hsum : (esD.take dA).length + (esD.drop dA).length = D := by omega
  have := prefix_dist gA rA rB u _ _ h1 h2 (by rw [hsum]; exact hDD)
  rw [hl1] at this
  exact ⟨u, (IA.front u).mpr this⟩

structure LInv (nf nb : Nat → List Nat) (src dst : Nat) (st : BB) : Prop where
  F : SI' nf src st.fm st.ff st.df
  B : SI' nb dst st.bm st.bf st.db
  disj : ∀ x, Key st.fm x → ¬ Key st.bm x

def FinOK (nf nb : Nat → List Nat) (src dst maxLevel : Nat) (st' : BB) (n : Nat) : Prop :=
  (∀ x, Key st'.fm x ↔ ∃ j, j ≤ st'.df ∧ R (gs nf) src j x) ∧
  (∀ x p k, (x, p, k) ∈ st'.fm → Dst (gs nf) src x k ∧
    ((x = src ∧ p = src ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ nf p ∧ Dst (gs nf) src p (k - 1)))) ∧
  (∀ x, Key st'.bm x ↔ ∃ j, j ≤ st'.db ∧ R (gs nb) dst j x) ∧
  (∀ x p k, (x, p, k) ∈ st'.bm → Dst (gs nb) dst x k ∧
    ((x = dst ∧ p = dst ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ nb p ∧ Dst (gs nb) dst p (k - 1)))) ∧
  ∃ A B pf pb, dget st'.fm n = some (pf, A) ∧ dget st'.bm n = some (pb, B) ∧ A ≤ st'.df ∧ B ≤ st'.db ∧
    A + B ≤ maxLevel ∧ ∀ es, WalkTo (gs nf) src es dst → A + B ≤ es.length

theorem step_gen (gA gB : Nat → List Nat) (rA rB : Nat) (mA mB : DM) (frA frB : List Nat) (dA dB : Nat)
    (hrev : ∀ a es b, WalkTo (gs gA) a es b → ∃ es', WalkTo (gs gB) b es' a ∧ es'.length = es.length)
    (IA : SI' gA rA mA frA dA) (IB : SI' gB rB mB frB dB) (hdisj : ∀ x, Key mA x → ¬ Key mB x) :
    let s' := scanFront gA mB (dA + 1) frA ⟨mA, [], none⟩
    (s'.meet = none → SI' gA rA s'.own s'.next (dA + 1) ∧ ∀ x, Key s'.own x → ¬ Key mB x) ∧
    (∀ od n, s'.meet = some (od, n) →
      (∀ x, Key s'.own x ↔ ∃ j, j ≤ dA + 1 ∧ R (gs gA) rA j x) ∧
      (∀ x p k, (x, p, k) ∈ s'.own → Dst (gs gA) rA x k ∧
        ((x = rA ∧ p = rA ∧ k = 0) ∨ (1 ≤ k ∧ x ∈ gA p ∧ Dst (gs gA) rA p (k - 1)))) ∧
      (∃ p, dget s'.own n = some (p, dA + 1)) ∧ (∃ q, dget mB n = some (q, od)) ∧ od ≤ dB ∧
      ∀ es, WalkTo (gs gA) rA es rB → dA + 1 + od ≤ es.length) := by
  intro s'
  obtain ⟨h1, h2, h3, h4⟩ := expand_spec gA rA dA frA mA mB IA
  refine ⟨fun hm => ⟨h1 hm, fun x hx => ?_⟩, fun od n hm => ?_⟩
  · rcases expand_disj gA rA dA frA mA mB IA hm x hx with h | h
    · exact hdisj x h
    · exact h
  · obtain ⟨hD, ⟨q, hq⟩, hmin⟩ := h4 od n hm
    refine ⟨h2, h3, ?_, ⟨q, hq⟩, depth_le gB rB mB frB dB IB n q od hq, fun es hw => ?_⟩
    · exact dget_depth gA rA s'.own h3 n (dA + 1) hD ((h2 n).mpr ⟨dA + 1, Nat.le_refl _, hD.1⟩)
    · exact meet_min gA gB rA rB mA mB frA frB dA dB hrev IA IB hdisj od n q hq hmin es hw

theorem bbLoop_spec (nf nb : Nat → List Nat) (hrv1 : ∀ u v, v ∈ nf u → u ∈ nb v)
    (hrv2 : ∀ u v, v ∈ nb u → u ∈ nf v) (src dst maxLevel : Nat) :
    ∀ (fuel : Nat) (st : BB) (res : Option (BB × Nat × Nat)), LInv nf nb src dst st →
      bbLoop nf nb maxLevel fuel st = some res →
      (res = none → ∀ es, WalkTo (gs nf) src es dst → maxLevel < es.length) ∧
      (∀ st' od n, res = some (st', od, n) → FinOK nf nb src dst maxLevel st' n) := by
  have hrevF : ∀ a es b, WalkTo (gs nf) a es b → ∃ es', WalkTo (gs nb) b es' a ∧ es'.length = es.length :=
    fun a es b hw => by obtain ⟨es', h, l, _⟩ := walk_rev nf nb hrv1 es a b hw; exact ⟨es', h, l⟩
  have hrevB : ∀ a es b, WalkTo (gs nb) a es b → ∃ es', WalkTo (gs nf) b es' a ∧ es'.length = es.length :=
    fun a es b hw => by obtain ⟨es', h, l, _⟩ := walk_rev nb nf hrv2 es a b hw; exact ⟨es', h, l⟩
  intro fuel
  induction fuel with
  | zero => intro st res _ h; simp [bbLoop] at h
  | succ n ih =>
    intro st res I h
    simp only [bbLoop] at h
    by_cases hg : (st.ff.isEmpty || st.bf.isEmpty || decide (maxLevel ≤ st.df + st.db)) = true
    · rw [if_pos hg] at h
      simp only [Option.some.injEq] at h
      subst h
      refine ⟨fun _ es hw => ?_, fun st' od n h => by simp at h⟩
      simp only [Bool.or_eq_true, decide_eq_true_eq, List.isEmpty_iff] at hg
      rcases hg with (hg | hg) | hg
      · obtain ⟨u, hu⟩ := front_nonempty nf nb src dst st.fm st.bm st.ff st.bf st.df st.db hrevF I.F I.B I.disj es hw
        rw [hg] at hu; simp at hu
      · obtain ⟨es', hw', _⟩ := hrevF src es dst hw
        obtain ⟨u, hu⟩ := front_nonempty nb nf dst src st.bm st.fm st.bf st.ff st.db st.df hrevB I.B I.F
          (fun x hb hf => I.disj x hf hb) es' hw'
        rw [hg] at hu; simp at hu
      · have := disj_lb nf nb src dst st.fm st.bm st.ff st.bf st.df st.db hrevF I.F I.B I.disj es hw
        omega
    · rw [if_neg hg] at h
      have hlev : st.df + st.db < maxLevel := by
        simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hg
        omega
      by_cases hl : st.ff.length ≤ st.bf.length
      · rw [if_pos hl] at h
        obtain ⟨s1, s2⟩ := step_gen nf nb src dst st.fm st.bm st.ff st.bf st.df st.db hrevF I.F I.B I.disj
        rcases hmt : (scanFront nf st.bm (st.df + 1) st.ff ⟨st.fm, [], none⟩).meet with _ | ⟨od, nn⟩
        · rw [hmt] at h
          simp only at h
          obtain ⟨hF', hdj⟩ := s1 hmt
          refine ih _ res ?_ h
          exact ⟨hF', I.B, fun x hx hb => hdj x hx hb⟩
        · rw [hmt] at h
          simp only [Option.some.injEq] at h
          subst h
          obtain ⟨k1, k2, k3, k4, k5, k6⟩ := s2 od nn hmt
          refine ⟨fun h => by simp at h, fun st' od' n' h' => ?_⟩
          simp only [Option.some.injEq, Prod.mk.injEq] at h'
          obtain ⟨rfl, rfl, rfl⟩ := h'
          obtain ⟨p, hp⟩ := k3
          obtain ⟨q, hq⟩ := k4
          exact ⟨k1, k2, I.B.keys, I.B.ent, st.df + 1, od, p, q, hp, hq, Nat.le_refl _, k5, by omega, k6⟩
      · rw [if_neg hl] at h
        obtain ⟨s1, s2⟩ := step_gen nb nf dst src st.bm st.fm st.bf st.ff st.db st.df hrevB I.B I.F
          (fun x hb hf => I.disj x hf hb)
        rcases hmt : (scanFront nb st.fm (st.db + 1) st.bf ⟨st.bm, [], none⟩).meet with _ | ⟨od, nn⟩
        · rw [hmt] at h
          simp only at h
          obtain ⟨hB', hdj⟩ := s1 hmt
          refine ih _ res ?_ h
          exact ⟨I.F, hB', fun x hx hb => hdj x hb hx⟩
        · rw [hmt] at h
          simp only [Option.some.injEq] at h
          subst h
          obtain ⟨k1, k2, k3, k4, k5, k6⟩ := s2 od nn hmt
          refine ⟨fun h => by simp at h, fun st' od' n' h' => ?_⟩
          simp only [Option.some.injEq, Prod.mk.injEq] at h'
          obtain ⟨rfl, rfl, rfl⟩ := h'
          obtain ⟨p, hp⟩ := k3
          obtain ⟨q, hq⟩ := k4
          refine ⟨I.F.keys, I.F.ent, k1, k2, od, st.db + 1, q, p, hq, hp, k5, Nat.le_refl _, by omega, fun es hw => ?_⟩
          obtain ⟨es', hw', hl'⟩ := hrevF src es dst hw
          have := k6 es' hw'
          omega

/-- **PROVEN** (`bfs_shortest_path`, node list; `NeighborIter` rows `nf`/`nb` are converse of each other,
`hrv1`/`hrv2`: the forward iterator over the adjacency and the reversed/transposed one): with `src = dst`
the answer is `Null`. Otherwise `Null` only when no walk `src → dst` has at most `max_level` edges (a
walk whose length is within the level cap is always found: the stop rule is complete), and a returned
node list is the node sequence of a walk `src → dst` along `nf` that is a *shortest* one (no walk with
fewer edges), at most `max_level` long, so it also visits distinct nodes (`Search.shortest_nodup`). -/
theorem bspNodes_spec (nf nb : Nat → List Nat) (hrv1 : ∀ u v, v ∈ nf u → u ∈ nb v)
    (hrv2 : ∀ u v, v ∈ nb u → u ∈ nf v) (src dst maxLevel fuel : Nat) (r : Option (List Nat))
    (h : bspNodes nf nb src dst maxLevel fuel = some r) :
    (src = dst → r = none) ∧ (src ≠ dst →
      (r = none → ∀ es, WalkTo (gs nf) src es dst → maxLevel < es.length) ∧
      (∀ P, r = some P → ∃ es, WalkTo (gs nf) src es dst ∧ nodesOf src es = P ∧
        Dst (gs nf) src dst es.length ∧ es.length ≤ maxLevel)) := by
  unfold bspNodes at h
  by_cases hsd : src = dst
  · simp only [hsd, ↓reduceIte, Option.some.injEq] at h
    exact ⟨fun _ => h.symm, fun h' => absurd hsd h'⟩
  · refine ⟨fun h' => absurd h' hsd, fun _ => ?_⟩
    simp only [hsd, ↓reduceIte] at h
    have hI : LInv nf nb src dst ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ :=
      ⟨SI'_init nf src, SI'_init nb dst, fun x hf hb => by
        obtain ⟨p, k, hm⟩ := hf
        obtain ⟨p', k', hm'⟩ := hb
        simp only [List.mem_singleton, Prod.mk.injEq] at hm hm'
        exact hsd (hm.1.symm.trans hm'.1)⟩
    rcases hl : bbLoop nf nb maxLevel fuel ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ with _ | _ | ⟨st, od, n⟩
    · rw [hl] at h; simp at h
    · rw [hl] at h
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨h1, _⟩ := bbLoop_spec nf nb hrv1 hrv2 src dst maxLevel fuel _ _ hI hl
      exact ⟨fun _ => h1 rfl, fun P hP => by simp at hP⟩
    · rw [hl] at h
      obtain ⟨_, h2⟩ := bbLoop_spec nf nb hrv1 hrv2 src dst maxLevel fuel _ _ hI hl
      obtain ⟨hkf, hef, hkb, heb, A, B, pf, pb, hgf, hgb, hAd, hBd, hAB, hmin⟩ := h2 st od n rfl
      have hDA := (hef n pf A (dget_mem _ n pf A hgf)).1
      have hDB := (heb n pb B (dget_mem _ n pb B hgb)).1
      obtain ⟨esf, lf, hwf, hlf, hcf, hnf⟩ := chain_ok nf src st.fm st.df hkf hef A hAd n hDA maxLevel (by omega)
      obtain ⟨esb, lb, hwb, hlb, hcb, hnb⟩ := chain_ok nb dst st.bm st.db hkb heb B hBd n hDB maxLevel (by omega)
      obtain ⟨esB', hwB', hlB', hnB'⟩ := walk_rev nb nf hrv2 esb dst n hwb
      have hlb' : esB'.map (·.1) = lb := by
        have : nodesOf n esB' = n :: lb := by rw [hnB', ← hnb]
        simpa [nodesOf] using this
      simp only [hcf, hcb] at h
      simp only [Option.some.injEq] at h
      subst h
      refine ⟨fun h' => by simp at h', fun P hP => ?_⟩
      simp only [Option.some.injEq] at hP
      subst hP
      have hwall : WalkTo (gs nf) src (esf ++ esB') dst := (walk_append _ src esf esB' dst).mpr ⟨n, hwf, hwB'⟩
      have hlen : (esf ++ esB').length = A + B := by simp [hlf, hlB', hlb]
      refine ⟨esf ++ esB', hwall, ?_, ?_, by omega⟩
      · have : (n :: lf).reverse = nodesOf src esf := by rw [hnf]; simp
        rw [this]
        simp [nodesOf, hlb']
      · rw [hlen]
        refine ⟨⟨esf ++ esB', hlen, hwall⟩, fun j hj hr => ?_⟩
        obtain ⟨e0, hl0, hw0⟩ := hr
        have := hmin e0 hw0
        omega

end OpsTraverse.BidirBfs
