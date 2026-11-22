import OpsTraverse.SearchBase
/-
# `bfs_find_bound` (graph/src/runtime/functions/algo_procedures.rs:2426)

Level-synchronised BFS from the source for the target, one parent per node, then the parent chain
folded source-to-target. `step u` is `node_successors` (far endpoint and the edge, in iterator
order), `ε` the edge. `wf`/`cf` are `edge_numeric_attr` with the weight/cost property (default `1.0`),
`add`/`zero` the `f64` `+`/`0.0`. Timeout / memory polls (`check_timeout`) are abstracted away: the
theorems are about the value returned when they do not fire.

| here | there |
| --- | --- |
| `BS` (`seen`, `parents`, `next`) | `seen`, `parents`, `next_frontier` |
| `scanEdges` | the inner `for (edge_src, edge_dst, edge_id)` loop, including `break 'levels` |
| `scanLevel` | the `for &node in &frontier` loop |
| `levels m` | `'levels: for _hop in 1..=max_len` (`m` hops left), with `swap` + `clear` |
| `chain` | the `while cur != config.source` loop + `chain.reverse()` (fuel = `max_len`, never short: `bfsFind_chain`) |
| `bfsFindBound` | the whole function |
-/
namespace OpsTraverse.BfsFind
open OpsTraverse.Search

variable {ε F : Type}

structure BS (ε : Type) where
  seen : List Nat
  parents : List (Nat × Nat × ε)
  next : List Nat

/-- The inner loop over `node`'s edges: a first-seen `far` records its parent; the target stops
everything (`break 'levels`), any other joins `next_frontier`. -/
def scanEdges (t node : Nat) : List (Nat × ε) → BS ε → BS ε × Bool
  | [], st => (st, false)
  | (far, e) :: r, st =>
    if far ∈ st.seen then scanEdges t node r st
    else
      let st' : BS ε := { st with seen := far :: st.seen, parents := (far, node, e) :: st.parents }
      if far == t then (st', true)
      else scanEdges t node r { st' with next := st'.next ++ [far] }

def scanLevel (step : Step ε) (t : Nat) : List Nat → BS ε → BS ε × Bool
  | [], st => (st, false)
  | n :: ns, st =>
    match scanEdges t n (step n) st with
    | (st', true) => (st', true)
    | (st', false) => scanLevel step t ns st'

/-- `m` hops left. `none` = the loop ended without finding the target. -/
def levels (step : Step ε) (t : Nat) : Nat → List Nat → BS ε → Option (List (Nat × Nat × ε))
  | 0, _, _ => none
  | m + 1, fr, st =>
    if fr.isEmpty then none else
    match scanLevel step t fr st with
    | (st', true) => some st'.parents
    | (st', false) => levels step t m st'.next { st' with next := [] }

def lookupP (par : List (Nat × Nat × ε)) (x : Nat) : Option (Nat × ε) :=
  (par.find? (fun q => q.1 == x)).map (·.2)

/-- Walk the parent chain back to `s` and flip to source→target order. -/
def chain (s : Nat) (par : List (Nat × Nat × ε)) : Nat → Nat → Option (List (Nat × ε))
  | 0, cur => if cur = s then some [] else none
  | f + 1, cur =>
    if cur = s then some [] else
    match lookupP par cur with
    | none => none
    | some (p, e) => (chain s par f p).map (· ++ [(cur, e)])

def bfsFindBound (step : Step ε) (s t maxLen : Nat) (wf cf : ε → F) (add : F → F → F) (zero : F) :
    Option (F × F) :=
  if maxLen = 0 then none else
  match levels step t maxLen [s] ⟨[s], [], []⟩ with
  | none => none
  | some par =>
    match chain s par maxLen t with
    | none => none
    | some es => some (fsum add zero wf es, fsum add zero cf es)

/-! ## Invariants -/

/-- Every recorded parent entry is a real edge from a node one level closer. -/
def PV (step : Step ε) (s : Nat) (par : List (Nat × Nat × ε)) : Prop :=
  ∀ x p e, (x, p, e) ∈ par → x ≠ s ∧ (x, e) ∈ step p ∧ ∃ k, Dst step s p k ∧ Dst step s x (k + 1)

/-- Every node at distance `1..h` has a parent entry. -/
def Cover (step : Step ε) (s h : Nat) (par : List (Nat × Nat × ε)) : Prop :=
  ∀ x j, Dst step s x j → 1 ≤ j → j ≤ h → ∃ p e, (x, p, e) ∈ par

/-- Mid-level state: `done` frontier nodes fully scanned, the level being `h → h+1`. -/
structure MI (step : Step ε) (s t h : Nat) (done : List Nat) (st : BS ε) : Prop where
  seen : ∀ x, x ∈ st.seen ↔ (∃ j, j ≤ h ∧ R step s j x) ∨ x ∈ st.next
  next : ∀ x ∈ st.next, Dst step s x (h + 1) ∧ ∃ p e, (x, p, e) ∈ st.parents
  pv : PV step s st.parents
  cov : Cover step s h st.parents
  prog : ∀ u ∈ done, ∀ y ∈ step u, y.1 ∈ st.seen
  tgt : t ∈ st.seen → t = s

/-- Level-start state. -/
structure LI (step : Step ε) (s t h : Nat) (fr : List Nat) (st : BS ε) : Prop where
  seen : ∀ x, x ∈ st.seen ↔ ∃ j, j ≤ h ∧ R step s j x
  fr : ∀ x, x ∈ fr ↔ Dst step s x h
  pv : PV step s st.parents
  cov : Cover step s h st.parents
  tgt : t ∈ st.seen → t = s
  nxt : st.next = []

/-- What a found target leaves behind. -/
def Found (step : Step ε) (s t h : Nat) (par : List (Nat × Nat × ε)) : Prop :=
  PV step s par ∧ Cover step s h par ∧ Dst step s t (h + 1) ∧ ∃ p e, (t, p, e) ∈ par

theorem scanEdges_spec (step : Step ε) (s t h node : Nat) (done : List Nat) (hn : Dst step s node h) :
    ∀ (l pr : List (Nat × ε)) (st : BS ε), step node = pr ++ l → MI step s t h done st →
      (∀ y ∈ pr, y.1 ∈ st.seen) →
      ((scanEdges t node l st).2 = false → MI step s t h (node :: done) (scanEdges t node l st).1) ∧
      ((scanEdges t node l st).2 = true → Found step s t h (scanEdges t node l st).1.parents) := by
  intro l
  induction l with
  | nil =>
    intro pr st hl I hpr
    simp only [scanEdges]
    refine ⟨fun _ => ?_, fun h => absurd h (by simp)⟩
    refine ⟨I.seen, I.next, I.pv, I.cov, fun u hu y hy => ?_, I.tgt⟩
    rcases List.mem_cons.mp hu with rfl | hu
    · apply hpr; rw [hl] at hy; simpa using hy
    · exact I.prog u hu y hy
  | cons y r ih =>
    intro pr st hl I hpr
    obtain ⟨far, e⟩ := y
    have hmem : (far, e) ∈ step node := by rw [hl]; simp
    have hl' : step node = (pr ++ [(far, e)]) ++ r := by rw [hl]; simp
    by_cases hs : far ∈ st.seen
    · simp only [scanEdges, hs, ↓reduceIte]
      exact ih (pr ++ [(far, e)]) st hl' I (fun z hz => by
        rcases List.mem_append.mp hz with hz | hz
        · exact hpr z hz
        · simp at hz; subst hz; exact hs)
    · have hns : ¬ ((∃ j, j ≤ h ∧ R step s j far) ∨ far ∈ st.next) := fun h => hs ((I.seen far).mpr h)
      simp only [not_or, not_exists, not_and] at hns
      obtain ⟨hnR, hnn⟩ := hns
      have hD : Dst step s far (h + 1) := dist_succ step s node far h e hn hmem (fun j hj => hnR j hj)
      have hfs : far ≠ s := R_ne_source step s far (h + 1) hD (by omega)
      have hpv : PV step s ((far, node, e) :: st.parents) := by
        intro x p e' hx
        rcases List.mem_cons.mp hx with h1 | h1
        · simp only [Prod.mk.injEq] at h1
          obtain ⟨rfl, rfl, rfl⟩ := h1
          exact ⟨hfs, hmem, h, hn, hD⟩
        · exact I.pv x p e' h1
      have hcov : Cover step s h ((far, node, e) :: st.parents) := fun x j hd h1 h2 => by
        obtain ⟨p, e', hp⟩ := I.cov x j hd h1 h2
        exact ⟨p, e', List.mem_cons_of_mem _ hp⟩
      by_cases ht : (far == t) = true
      · simp only [scanEdges, hs, ↓reduceIte, ht]
        have ht' : far = t := by simpa using ht
        subst ht'
        refine ⟨fun h => absurd h (by simp), fun _ => ?_⟩
        exact ⟨hpv, hcov, hD, node, e, by simp⟩
      · simp only [scanEdges, hs, ↓reduceIte, ht]
        have ht' : far ≠ t := by simpa using ht
        refine ih (pr ++ [(far, e)]) _ hl' ⟨?_, ?_, hpv, hcov, ?_, ?_⟩ ?_
        · intro x
          simp only [List.mem_cons, List.mem_append, List.not_mem_nil, or_false]
          constructor
          · rintro (rfl | hx)
            · exact Or.inr (Or.inr rfl)
            · rcases (I.seen x).mp hx with h1 | h1
              · exact Or.inl h1
              · exact Or.inr (Or.inl h1)
          · rintro (h1 | h1 | h1)
            · exact Or.inr ((I.seen x).mpr (Or.inl h1))
            · exact Or.inr ((I.seen x).mpr (Or.inr h1))
            · exact Or.inl h1
        · intro x hx
          simp only [List.mem_append, List.mem_singleton] at hx
          rcases hx with hx | rfl
          · obtain ⟨h1, p, e', h2⟩ := I.next x hx
            exact ⟨h1, p, e', List.mem_cons_of_mem _ h2⟩
          · exact ⟨hD, node, e, by simp⟩
        · intro u hu z hz
          exact List.mem_cons_of_mem _ (I.prog u hu z hz)
        · intro htm
          rcases List.mem_cons.mp htm with h1 | h1
          · exact absurd h1.symm ht'
          · exact I.tgt h1
        · intro z hz
          rcases List.mem_append.mp hz with hz | hz
          · exact List.mem_cons_of_mem _ (hpr z hz)
          · simp at hz; subst hz; simp


theorem scanLevel_spec (step : Step ε) (s t h : Nat) :
    ∀ (fl done : List Nat) (st : BS ε), (∀ u ∈ fl, Dst step s u h) → MI step s t h done st →
      ((scanLevel step t fl st).2 = false →
        MI step s t h (fl.reverse ++ done) (scanLevel step t fl st).1) ∧
      ((scanLevel step t fl st).2 = true → Found step s t h (scanLevel step t fl st).1.parents) := by
  intro fl
  induction fl with
  | nil =>
    intro done st _ I
    simp only [scanLevel, List.reverse_nil, List.nil_append]
    exact ⟨fun _ => I, fun h => absurd h (by simp)⟩
  | cons n ns ih =>
    intro done st hfl I
    have hn : Dst step s n h := hfl n (by simp)
    have hsp := scanEdges_spec step s t h n done hn (step n) [] st (by simp) I (by simp)
    simp only [scanLevel]
    rcases hr : scanEdges t n (step n) st with ⟨st', f⟩
    rw [hr] at hsp
    cases f with
    | true =>
      simp only
      exact ⟨fun h => absurd h (by simp), fun _ => hsp.2 rfl⟩
    | false =>
      simp only
      have I' := hsp.1 rfl
      have := ih (n :: done) st' (fun u hu => hfl u (List.mem_cons_of_mem _ hu)) I'
      simpa using this

theorem level_end (step : Step ε) (s t h : Nat) (fr done : List Nat) (st : BS ε)
    (hfr : ∀ u, u ∈ fr ↔ Dst step s u h) (I : MI step s t h done st) (hdone : ∀ u, u ∈ fr → u ∈ done) :
    LI step s t (h + 1) st.next { st with next := [] } := by
  have hnext : ∀ x, Dst step s x (h + 1) → x ∈ st.next := by
    intro x hx
    obtain ⟨u, e, he, hu⟩ := dist_pred step s x h hx
    have hseen : x ∈ st.seen := I.prog u (hdone u ((hfr u).mpr hu)) (x, e) he
    rcases (I.seen x).mp hseen with ⟨j, hj, hr⟩ | h1
    · exact absurd hr (hx.2 j (by omega))
    · exact h1
  refine ⟨?_, ?_, I.pv, ?_, I.tgt, rfl⟩
  · intro x
    show x ∈ st.seen ↔ _
    constructor
    · intro hx
      rcases (I.seen x).mp hx with ⟨j, hj, hr⟩ | h1
      · exact ⟨j, by omega, hr⟩
      · exact ⟨h + 1, Nat.le_refl _, (I.next x h1).1.1⟩
    · rintro ⟨j, hj, hr⟩
      obtain ⟨d, hd, hdd⟩ := exists_dist step s j x hr
      by_cases hdh : d ≤ h
      · exact (I.seen x).mpr (Or.inl ⟨d, hdh, hdd.1⟩)
      · have : d = h + 1 := by omega
        subst this
        exact (I.seen x).mpr (Or.inr (hnext x hdd))
  · intro x
    constructor
    · intro hx; exact (I.next x hx).1
    · exact hnext x
  · intro x j hd h1 h2
    by_cases hj : j ≤ h
    · exact I.cov x j hd h1 hj
    · have : j = h + 1 := by omega
      subst this
      obtain ⟨_, p, e, hp⟩ := I.next x (hnext x hd)
      exact ⟨p, e, hp⟩

theorem lookup_spec (par : List (Nat × Nat × ε)) (x : Nat) (h : ∃ p e, (x, p, e) ∈ par) :
    ∃ p e, lookupP par x = some (p, e) ∧ (x, p, e) ∈ par := by
  obtain ⟨p, e, hm⟩ := h
  unfold lookupP
  cases hf : par.find? (fun q => q.1 == x) with
  | none =>
    have := List.find?_eq_none.mp hf _ hm
    simp at this
  | some q =>
    have h1 := List.mem_of_find?_eq_some hf
    have h2 := List.find?_some hf
    obtain ⟨x', p', e'⟩ := q
    simp only [beq_iff_eq] at h2
    subst h2
    exact ⟨p', e', rfl, h1⟩

theorem chain_ok (step : Step ε) (s : Nat) (par : List (Nat × Nat × ε)) (hpv : PV step s par) :
    ∀ h, Cover step s h par → ∀ x, Dst step s x (h + 1) → (∃ p e, (x, p, e) ∈ par) →
      ∀ fuel, h + 1 ≤ fuel → ∃ es, chain s par fuel x = some es ∧ es.length = h + 1 ∧ WalkTo step s es x := by
  intro h
  induction h with
  | zero =>
    intro _ x hx hent fuel hf
    obtain ⟨p, e, hl, hm⟩ := lookup_spec par x hent
    obtain ⟨hxs, hed, k, hp, hx'⟩ := hpv x p e hm
    have hk : k = 0 := by
      have := dist_unique step s x hx hx'
      omega
    subst hk
    have hps : p = s := (dist_zero step s p).mp hp
    subst hps
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    refine ⟨[(x, e)], ?_, rfl, ?_⟩
    · simp only [chain, hxs, ↓reduceIte, hl]
      cases f <;> simp [chain]
    · simp only [WalkTo]; exact ⟨hed, trivial⟩
  | succ h ih =>
    intro hc x hx hent fuel hf
    obtain ⟨p, e, hl, hm⟩ := lookup_spec par x hent
    obtain ⟨hxs, hed, k, hp, hx'⟩ := hpv x p e hm
    have hk : k = h + 1 := by
      have := dist_unique step s x hx hx'
      omega
    subst hk
    obtain ⟨p2, e2, hm2⟩ := hc p (h + 1) hp (by omega) (Nat.le_refl _)
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    obtain ⟨es, hes, hlen, hw⟩ := ih (fun y j hd h1 h2 => hc y j hd h1 (by omega)) p hp ⟨p2, e2, hm2⟩ f (by omega)
    refine ⟨es ++ [(x, e)], ?_, by simp [hlen], ?_⟩
    · simp only [chain, hxs, ↓reduceIte, hl, hes, Option.map_some]
    · exact (walk_append step s es _ x).mpr ⟨p, hw, by simpa [WalkTo] using hed⟩

theorem levels_spec (step : Step ε) (s t : Nat) :
    ∀ (m h : Nat) (fr : List Nat) (st : BS ε), LI step s t h fr st →
      (levels step t m fr st = none → ∀ k, h < k → k ≤ h + m → ¬ Dst step s t k) ∧
      (∀ par, levels step t m fr st = some par →
        ∃ k, h < k ∧ k ≤ h + m ∧ Dst step s t k ∧ PV step s par ∧ Cover step s (k - 1) par ∧
          ∃ p e, (t, p, e) ∈ par) := by
  intro m
  induction m with
  | zero =>
    intro h fr st I
    simp only [levels]
    exact ⟨fun _ k h1 h2 => by omega, fun par h => by simp at h⟩
  | succ m ih =>
    intro h fr st I
    unfold levels
    by_cases hem : fr.isEmpty = true
    · simp only [hem, ↓reduceIte]
      refine ⟨fun _ k h1 h2 => ?_, fun par h => by simp at h⟩
      have hne : ∀ u, ¬ Dst step s u h := fun u hu => by
        have := (I.fr u).mpr hu
        simp [List.isEmpty_iff.mp hem] at this
      exact no_level step s h hne k (by omega) t
    · have hem' : fr.isEmpty = false := by simpa using hem
      simp only [hem', Bool.false_eq_true, ↓reduceIte]
      have hmi : MI step s t h [] st := by
        refine ⟨?_, ?_, I.pv, I.cov, ?_, I.tgt⟩
        · intro x; simp [I.nxt, I.seen]
        · intro x hx; simp [I.nxt] at hx
        · intro u hu; simp at hu
      have hsl := scanLevel_spec step s t h fr [] st (fun u hu => (I.fr u).mp hu) hmi
      rcases hr : scanLevel step t fr st with ⟨st', f⟩
      rw [hr] at hsl
      cases f with
      | true =>
        simp only
        obtain ⟨hpv, hcov, hD, hent⟩ := hsl.2 rfl
        refine ⟨fun h => by simp at h, fun par hp => ?_⟩
        simp only [Option.some.injEq] at hp
        subst hp
        exact ⟨h + 1, by omega, by omega, hD, hpv, by simpa using hcov, hent⟩
      | false =>
        simp only
        have hm := hsl.1 rfl
        simp only [List.append_nil] at hm
        have hle := level_end step s t h fr (fr.reverse ++ []) st' (fun u => I.fr u) (by simpa using hm)
          (fun u hu => by simpa using hu)
        have := ih (h + 1) st'.next _ hle
        refine ⟨fun hn k h1 h2 => ?_, fun par hp => ?_⟩
        · by_cases hk : k = h + 1
          · subst hk
            intro hD
            have hseen := (hle.seen t).mpr ⟨h + 1, Nat.le_refl _, hD.1⟩
            have := hle.tgt hseen
            subst this
            exact R_ne_source step t t (h + 1) hD (by omega) rfl
          · exact this.1 hn k (by omega) (by omega)
        · obtain ⟨k, hk1, hk2, rest⟩ := this.2 par hp
          exact ⟨k, by omega, by omega, rest⟩


theorem li_init (step : Step ε) (s t : Nat) : LI step s t 0 [s] (⟨[s], [], []⟩ : BS ε) where
  seen x := by
    simp only [List.mem_singleton]
    constructor
    · intro h; exact ⟨0, Nat.le_refl _, (R_zero step s x).mpr h⟩
    · rintro ⟨j, hj, hr⟩
      have : j = 0 := by omega
      subst this; exact (R_zero step s x).mp hr
  fr x := by simp [dist_zero]
  pv := by intro x p e h; simp at h
  cov := by intro x j _ h1 h2; omega
  tgt := by simp
  nxt := rfl

/-- **PROVEN** (`bfs_find_bound`): with `Reach` = a walk of `1..max_len` steps to a target other than
the source, the function returns `None` exactly when `Reach` fails (`max_len == 0` included, and
`target == source` is never found: the source is pre-seen). When it returns `(weight, cost)`, they
are the left folds `0.0 + w e₁ + w e₂ + …` / `0.0 + c e₁ + …` over the edges of a walk
source → target that is a shortest one (no walk of fewer steps exists), has `1 ≤ len ≤ max_len`, and
visits distinct nodes. -/
theorem bfsFind_spec (step : Step ε) (s t maxLen : Nat) (wf cf : ε → F) (add : F → F → F) (zero : F) :
    (bfsFindBound step s t maxLen wf cf add zero = none ↔
      ¬ (t ≠ s ∧ ∃ k, 1 ≤ k ∧ k ≤ maxLen ∧ R step s k t)) ∧
    (∀ w c, bfsFindBound step s t maxLen wf cf add zero = some (w, c) →
      ∃ es, WalkTo step s es t ∧ 1 ≤ es.length ∧ es.length ≤ maxLen ∧
        (∀ es', WalkTo step s es' t → es.length ≤ es'.length) ∧ (nodesOf s es).Nodup ∧
        w = fsum add zero wf es ∧ c = fsum add zero cf es) := by
  have hreach : ¬ (t ≠ s ∧ ∃ k, 1 ≤ k ∧ k ≤ maxLen ∧ R step s k t) ↔
      ∀ k, 1 ≤ k → k ≤ maxLen → ¬ Dst step s t k := by
    constructor
    · intro hn k h1 h2 hd
      exact hn ⟨R_ne_source step s t k hd h1, k, h1, h2, hd.1⟩
    · rintro hn ⟨hts, k, h1, h2, hr⟩
      obtain ⟨d, hd, hdd⟩ := exists_dist step s k t hr
      have : 1 ≤ d := by
        refine Nat.one_le_iff_ne_zero.mpr fun h0 => hts ?_
        subst h0; exact (dist_zero step s t).mp hdd
      exact hn d this (by omega) hdd
  by_cases hm : maxLen = 0
  · subst hm
    have h0 : bfsFindBound step s t 0 wf cf add zero = none := by simp [bfsFindBound]
    refine ⟨?_, fun w c h => by simp [h0] at h⟩
    rw [h0]
    simp only [true_iff]
    rintro ⟨_, k, h1, h2, _⟩
    omega
  · have hls := levels_spec step s t maxLen 0 [s] ⟨[s], [], []⟩ (li_init step s t)
    simp only [Nat.zero_add] at hls
    unfold bfsFindBound
    simp only [hm, ↓reduceIte]
    cases hl : levels step t maxLen [s] (⟨[s], [], []⟩ : BS ε) with
    | none =>
      simp only
      refine ⟨⟨fun _ => hreach.mpr (fun k h1 h2 => hls.1 hl k (by omega) h2), fun _ => trivial⟩, ?_⟩
      intro w c h; simp at h
    | some par =>
      obtain ⟨k, hk1, hk2, hD, hpv, hcov, hent⟩ := hls.2 par hl
      obtain ⟨es, hch, hlen, hw⟩ := chain_ok step s par hpv (k - 1) hcov t
        (by have : k - 1 + 1 = k := by omega
            rw [this]; exact hD) hent maxLen (by omega)
      have hk : k - 1 + 1 = k := by omega
      rw [hk] at hlen
      simp only [hch]
      refine ⟨iff_of_false (by simp) (fun hn => hn ⟨R_ne_source step s t k hD (by omega), k, by omega, by omega, hD.1⟩), ?_⟩
      intro w c h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨es, hw, by omega, by omega, fun es' hw' => ?_, ?_, rfl, rfl⟩
      · refine Classical.byContradiction fun hlt => ?_
        exact hD.2 es'.length (by omega) ⟨es', rfl, hw'⟩
      · have hD' : Dst step s t es.length := by rw [hlen]; exact hD
        exact (shortest_nodup step s es t hw hD').1

end OpsTraverse.BfsFind
