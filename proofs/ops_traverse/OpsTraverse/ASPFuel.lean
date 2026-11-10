import OpsTraverse.ASPMain
/-
allShortestPaths, part 6: the BFS drains its queue within `|nodes| + 1` iterations (each
iteration pops one node, and only never-seen nodes are enqueued), so the fuel hypothesis of
`asp_complete` is met by any fuel above the graph's node count — the Rust loop terminates.
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

/-- Every node a BFS can touch: `src` and the edge endpoints. -/
def univ (g : Graph) (src : Nat) : List Nat := src :: g.flatMap fun e => [e.src, e.dst]

def undisc (U : List Nat) (st : St) : Nat := (U.filter fun v => (st.dist v).isNone).length

theorem filter_len_le {α : Type} (l : List α) (p q : α → Bool) (hpq : ∀ v, p v = true → q v = true) :
    (l.filter p).length ≤ (l.filter q).length := by
  induction l with
  | nil => simp
  | cons y ys ih =>
    simp only [List.filter_cons]
    cases hp : p y <;> cases hq : q y <;>
      simp only [ite_true, ite_false, Bool.false_eq_true, List.length_cons] <;> try omega
    have := hpq y hp; rw [hq] at this; cases this

theorem filter_len_lt {α : Type} (U : List α) (p q : α → Bool) (hpq : ∀ v, p v = true → q v = true)
    (x : α) (hx : x ∈ U) (hqx : q x = true) (hpx : p x = false) : (U.filter p).length < (U.filter q).length := by
  induction U with
  | nil => simp at hx
  | cons u us ih =>
    have hle : ∀ (l : List α), (l.filter p).length ≤ (l.filter q).length := fun l => filter_len_le l p q hpq
    rcases List.mem_cons.mp hx with rfl | hx'
    · simp only [List.filter_cons, hqx, hpx, Bool.false_eq_true, ite_false, ite_true, List.length_cons]
      have := hle us; omega
    · have := ih hx'
      simp only [List.filter_cons]
      cases hp : p u <;> cases hq : q u <;>
        simp only [ite_true, ite_false, Bool.false_eq_true, List.length_cons] <;> try omega
      have := hpq u hp; rw [hq] at this; cases this

theorem count_drop {α : Type} [DecidableEq α] (U : List α) (p q : α → Bool) (hpq : ∀ v, p v = true → q v = true) :
    ∀ (news : List α), news.Nodup → (∀ v ∈ news, v ∈ U ∧ q v = true ∧ p v = false) →
      (U.filter p).length + news.length ≤ (U.filter q).length
  | [], _, _ => by
      simp only [List.length_nil, Nat.add_zero]
      exact filter_len_le U p q hpq
  | n :: ns, hn, hv => by
      have hn' := List.nodup_cons.mp hn
      obtain ⟨hnU, hqn, hpn⟩ := hv n (List.mem_cons_self ..)
      let q' : α → Bool := fun v => q v && v != n
      have ih := count_drop U p q' (fun v hp => by
          simp only [q', Bool.and_eq_true, bne_iff_ne, ne_eq]
          refine ⟨hpq v hp, fun h => ?_⟩; subst h; rw [hpn] at hp; cases hp) ns hn'.2
        (fun v hv' => by
          obtain ⟨a, b, c⟩ := hv v (List.mem_cons_of_mem _ hv')
          refine ⟨a, ?_, c⟩
          simp only [q', Bool.and_eq_true, bne_iff_ne, ne_eq]
          exact ⟨b, fun h => hn'.1 (h ▸ hv')⟩)
      have hlt := filter_len_lt U q' q (fun v h => by simp only [q', Bool.and_eq_true] at h; exact h.1) n hnU hqn
        (by simp [q'])
      simp only [List.length_cons]; omega

theorem step_mem_univ (g : Graph) (b : Bool) (src cur e v : Nat) (h : (e, v) ∈ step g b cur) : v ∈ univ g src := by
  obtain ⟨x, hx, _, hx1⟩ := mem_step.mp h
  simp only [univ, List.mem_cons, List.mem_flatMap]
  right
  rcases hx1 with ⟨_, rfl⟩ | ⟨_, _, _, rfl⟩
  · exact ⟨x, hx, by simp⟩
  · exact ⟨x, hx, by simp⟩

def UInv (g : Graph) (src : Nat) (st : St) : Prop := ∀ v, st.dist v ≠ none → v ∈ univ g src

def pot (g : Graph) (src : Nat) (st : St) : Nat := st.queue.length + undisc (univ g src) st

theorem bstep_pot (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St)
    (hi : Inv g b src dst maxH st) (hu : UInv g src st) (cur : Nat) (q : List Nat) (hq : st.queue = cur :: q) :
    UInv g src (bstep g b src dst maxH st) ∧ pot g src (bstep g b src dst maxH st) + 1 ≤ pot g src st := by
  obtain ⟨dcur, hc, _⟩ := hi.q_dist cur (by rw [hq]; simp)
  unfold bstep
  simp only [hq, hc, Option.getD_some]
  split
  · refine ⟨hu, ?_⟩
    simp [pot, undisc, hq]
    omega
  · have hGq : ∀ v ∈ ({ st with queue := q } : St).queue, ({ st with queue := q } : St).dist v ≠ none := by
      intro v hv; obtain ⟨d, hd, _⟩ := hi.q_dist v (by rw [hq]; exact List.mem_cons_of_mem _ hv)
      simp [hd]
    have R := expand_rel g b src dst maxH cur dcur { st with queue := q } hGq
    generalize List.foldl (relax src dst 1 maxH false cur dcur) { st with queue := q } (step g b cur) = X at R ⊢
    obtain ⟨news, hXq, hnd, hnews, _⟩ := R.queue_ext
    simp only at hXq hnews
    have hux : UInv g src X := by
      intro v hv
      cases hgv : st.dist v with
      | some d => exact hu v (by simp [hgv])
      | none =>
        obtain ⟨d, hd⟩ := Option.ne_none_iff_exists'.mp hv
        obtain ⟨_, e, he⟩ := R.dist_new v d hd hgv
        exact step_mem_univ g b src cur e v he
    refine ⟨hux, ?_⟩
    have hcnt := count_drop (univ g src) (fun v => (X.dist v).isNone) (fun v => (st.dist v).isNone)
      (fun v h => by
        simp only [Option.isNone_iff_eq_none] at h ⊢
        cases hs : st.dist v with
        | none => rfl
        | some d => rw [R.dist_keep v d hs] at h; cases h) news hnd
      (fun v hv => by
        obtain ⟨h1, h2, _⟩ := hnews v hv
        exact ⟨hux v (by rw [h2]; simp), by simp [h1], by simp [h2]⟩)
    simp only [pot, undisc, hXq, hq, List.length_append, List.length_cons]
    omega

theorem bfs_drains (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) :
    ∀ (n : Nat) (st : St), Inv g b src dst maxH st → UInv g src st → pot g src st ≤ n →
      (bfs g b src dst 1 maxH false n st).queue = []
  | 0, st, _, _, hp => by
      have : st.queue = [] := List.eq_nil_of_length_eq_zero (by unfold pot at hp; omega)
      simp [bfs, this]
  | n + 1, st, hi, hu, hp => by
      cases hq : st.queue with
      | nil => simp [bfs, hq]
      | cons cur q =>
        rw [bfs_unfold g b src dst maxH n st cur q hq]
        obtain ⟨hu', hp'⟩ := bstep_pot g hwf b src dst maxH st hi hu cur q hq
        exact bfs_drains g hwf b src dst maxH n _ (inv_bstep g hwf b src dst maxH st hi) hu' (by omega)

theorem bfs_stable (g : Graph) (b : Bool) (src dst maxH : Nat) :
    ∀ (n m : Nat) (st : St), (bfs g b src dst 1 maxH false n st).queue = [] → n ≤ m →
      bfs g b src dst 1 maxH false m st = bfs g b src dst 1 maxH false n st
  | 0, m, st, h, _ => by
      simp only [bfs] at h
      cases m with
      | zero => rfl
      | succ m => simp [bfs, h]
  | n + 1, m, st, h, hnm => by
      obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
      cases hq : st.queue with
      | nil => simp [bfs, hq]
      | cons cur q =>
        rw [bfs_unfold g b src dst maxH n st cur q hq] at h ⊢
        rw [bfs_unfold g b src dst maxH m' st cur q hq]
        exact bfs_stable g b src dst maxH n m' _ h (by omega)

/-- The BFS drains for every fuel above the node count. -/
theorem fin_drained (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH fuel : Nat) (hsd : src ≠ dst) (hm : 1 ≤ maxH)
    (hf : (univ g src).length + 1 ≤ fuel) : (bfs g b src dst 1 maxH false fuel (st0 src)).queue = [] := by
  have hu0 : UInv g src (st0 src) := by
    intro v hv; simp only [st0, upd] at hv
    split at hv
    · rename_i h; subst h; simp [univ]
    · exact absurd rfl hv
  have hp0 : pot g src (st0 src) ≤ (univ g src).length + 1 := by
    have := List.length_filter_le (fun v => ((st0 src).dist v).isNone) (univ g src)
    have hq0 : (st0 src).queue.length = 1 := rfl
    unfold pot undisc; omega
  have hd := bfs_drains g hwf b src dst maxH _ (st0 src) (inv_st0 g b src dst maxH hsd hm) hu0 hp0
  rw [bfs_stable g b src dst maxH _ fuel (st0 src) hd hf]; exact hd

/-- **Completeness, unconditional on the BFS**: with fuel above the node count and the path
length, every shortest walk of at most `max_hops` edges is returned. -/
theorem asp_complete' (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH fuel : Nat) (hsd : src ≠ dst)
    (hm : 1 ≤ maxH) (es : List Nat) (hw : Walk g b src es dst)
    (hmin : ∀ es', Walk g b src es' dst → es.length ≤ es'.length) (hmax : es.length ≤ maxH)
    (hf : (univ g src).length + 1 ≤ fuel) (hfuel : es.length + 1 ≤ fuel) :
    es ∈ asp g b src dst 1 maxH false fuel :=
  asp_complete g hwf b src dst maxH fuel hsd hm (fin_drained g hwf b src dst maxH fuel hsd hm hf) es hw hmin hmax hfuel

end OpsTraverse.ASP
