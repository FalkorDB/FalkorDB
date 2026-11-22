import FalkorNativeIndex.CchSearch
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# The route: predecessors, unpacking, loop erasure (`cch/query.rs:136-152`, `:206-229`, `:309-352`)
-/
namespace NI

/-- `Chain::arcs_up_to` (`query.rs:136-150`): follow the predecessors from `at` back to the start, then reverse;
    `fuel` bounds the walk (a predecessor is a lower rank). -/
def arcsUp (pr : Nat → Option Nat) : Nat → Nat → List (Nat × Nat)
  | 0, _ => []
  | f + 1, a =>
    match pr a with
    | none => []
    | some p => arcsUp pr f p ++ [(p, a)]

/-- A list of arcs `s -> ... -> a` going up, total weight `e` under `wf`. -/
inductive Arcs (up : Nat → List Nat) (wf : Nat → Nat → Wt) (s : Nat) : Nat → List (Nat × Nat) → Nat → Prop where
  | nil : Arcs up wf s s [] 0
  | snoc {p a : Nat} {L : List (Nat × Nat)} {e w : Nat} : Arcs up wf s p L e → a ∈ up p → wf p a = some w →
      Arcs up wf s a (L ++ [(p, a)]) (e + w)

/-- **The predecessors name a path**: for every reached rank, `arcs_up_to` returns the arcs of an ascending path
    from the start of exactly the distance found. -/
theorem arcsUp_spec {up wt A N} (h : Ctx up wt A N) {wf s done best ds} (inv : DPInv h wf s done best ds) :
    ∀ (a : Nat) (e : Nat), ds.d a = some e → ∀ f, a < f → Arcs up wf s a (arcsUp ds.pr f a) e := by
  intro a
  induction a using Nat.strongRecOn with
  | _ a ih =>
    intro e hd f hf
    cases f with
    | zero => omega
    | succ f =>
      cases hp : ds.pr a with
      | none =>
        by_cases has : a = s
        · subst has
          have := inv.z
          rw [this] at hd
          cases hd
          simp only [arcsUp, hp]
          exact Arcs.nil
        · obtain ⟨p, hp'⟩ := inv.reached a e hd has
          rw [hp] at hp'; cases hp'
      | some p =>
        obtain ⟨hpd, hpm, ep, w, hep, hw, he⟩ := inv.pred a e p hd hp
        have hlt := h.gt p a hpm
        have := ih p hlt ep hep f (by omega)
        simp only [arcsUp, hp]
        rw [he]
        exact Arcs.snoc this hpm hw

/-- `query`'s loop over the route's arcs (`query.rs:324-335`): unpack each in order; one missing arc fails. -/
def unpackArcs (up : Nat → List Nat) (M : Nat → Nat → Option Nat) (f : Nat) : List (Nat × Nat) → Option (List Nat)
  | [] => some []
  | (a, b) :: r =>
    match unpackRec up M f a b, unpackArcs up M f r with
    | some l, some l' => some (l ++ l')
    | _, _ => none

theorem unpackArcs_snoc (up M f) : ∀ (L : List (Nat × Nat)) (x : Nat × Nat),
    unpackArcs up M f (L ++ [x]) =
      (unpackArcs up M f L).bind (fun l => (unpackRec up M f x.1 x.2).map (fun l' => l ++ l')) := by
  intro L x
  induction L with
  | nil =>
    obtain ⟨a, b⟩ := x
    cases hu : unpackRec up M f a b <;> simp [unpackArcs, hu]
  | cons y L ih =>
    obtain ⟨a, b⟩ := y
    simp only [List.cons_append, unpackArcs, ih]
    cases hu : unpackRec up M f a b <;> cases hl : unpackArcs up M f L <;>
      cases hx : unpackRec up M f x.1 x.2 <;> simp [hl, hx, List.append_assoc]

/-- **The forward half unpacks to a walk of original edges** `s -> ... -> a` of exactly the path's weight. -/
theorem unpack_fwd {up wt A N} (h : Ctx up wt A N) (M : Nat → Nat → Option Nat)
    (hM : ∀ u v, ArcP up u v → (M u v = none → A u v = wt u v) ∧
      ∀ m, M u v = some m → m < min u v ∧ ArcP up u m ∧ ArcP up m v ∧
        ∃ e1 e2, A u m = some e1 ∧ A m v = some e2 ∧ A u v = some (e1 + e2))
    (s f : Nat) (hf : 2 * N < f) {a L e} (hA : Arcs up (fun x y => A x y) s a L e) :
    ∃ l, unpackArcs up M f L = some l ∧ lastOr s l = a ∧ EdgesFrom wt s l e := by
  induction hA with
  | nil => exact ⟨[], rfl, rfl, EdgesFrom.nil⟩
  | @snoc p a L e w hp hm hw ih =>
    obtain ⟨l, hl, hlast, hwalk⟩ := ih
    have harc : ArcP up p a := ⟨Nat.ne_of_lt (h.gt _ _ hm), Or.inl ⟨h.gt _ _ hm, hm⟩⟩
    have hpN := h.lt _ _ hm
    have := h.gt _ _ hm
    obtain ⟨l2, hl2, hv2, hw2⟩ := unpack_correct up wt A M hM f p a w (by omega) harc hw
    refine ⟨l ++ l2, ?_, ?_, ?_⟩
    · rw [unpackArcs_snoc, hl]; simp [hl2]
    · rw [lastOr_append, hlast]; exact hv2
    · exact hwalk.append (by rw [hlast]; exact hw2)

/-- **The backward half**, read from the meeting rank down to `t`: its arcs flipped and reversed unpack to a walk
    `a -> ... -> t` of exactly the path's weight. -/
theorem unpack_bwd {up wt A N} (h : Ctx up wt A N) (M : Nat → Nat → Option Nat)
    (hM : ∀ u v, ArcP up u v → (M u v = none → A u v = wt u v) ∧
      ∀ m, M u v = some m → m < min u v ∧ ArcP up u m ∧ ArcP up m v ∧
        ∃ e1 e2, A u m = some e1 ∧ A m v = some e2 ∧ A u v = some (e1 + e2))
    (t f : Nat) (hf : 2 * N < f) {a L e} (hA : Arcs up (fun x y => A y x) t a L e) :
    ∃ l, unpackArcs up M f (L.reverse.map (fun p => (p.2, p.1))) = some l ∧ lastOr a l = t ∧
      EdgesFrom wt a l e := by
  induction hA with
  | nil => exact ⟨[], rfl, rfl, EdgesFrom.nil⟩
  | @snoc p a L e w hp hm hw ih =>
    obtain ⟨l, hl, hlast, hwalk⟩ := ih
    have harc : ArcP up a p := ⟨(Nat.ne_of_lt (h.gt _ _ hm)).symm, Or.inr ⟨h.gt _ _ hm, hm⟩⟩
    have hpN := h.lt _ _ hm
    have := h.gt _ _ hm
    obtain ⟨l2, hl2, hv2, hw2⟩ := unpack_correct up wt A M hM f a p w (by omega) harc hw
    refine ⟨l2 ++ l, ?_, ?_, ?_⟩
    · simp only [List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append, List.singleton_append,
        List.map_cons, unpackArcs, hl2, hl]
    · rw [lastOr_append, hv2]; exact hlast
    · have := hw2.append (by rw [hv2]; exact hwalk)
      rwa [Nat.add_comm w e] at this

/-! ## `erase_loops` (`query.rs:352-368`)

The Rust keeps `at[node] = position in the output` and, on a repeated node, drops everything since its first visit
(removing the dropped nodes from `at`). With `at` = the position of each node of the output, that is: a node already
in the output truncates it just after that node, a new one is appended. -/

def eraseStep (out : List Nat) (r : Nat) : List Nat :=
  if r ∈ out then out.take (out.idxOf r + 1) else out ++ [r]

def eraseLoops (route : List Nat) : List Nat := route.foldl eraseStep []

/-- A walk given as its list of vertices. -/
def WalkR (wt : Nat → Nat → Wt) (l : List Nat) (c : Nat) : Prop := ∃ x l', l = x :: l' ∧ EdgesFrom wt x l' c

theorem edgesFrom_snoc {wt a l r c} (h : EdgesFrom wt a (l ++ [r]) c) :
    ∃ cp w, EdgesFrom wt a l cp ∧ wt (lastOr a l) r = some w ∧ c = cp + w := by
  induction l generalizing a c with
  | nil =>
    cases h with
    | @cons _ _ l w c' hw rest =>
      cases rest
      exact ⟨0, w, EdgesFrom.nil, hw, by omega⟩
  | cons y l ih =>
    cases h with
    | @cons _ _ l' w c' hw rest =>
      obtain ⟨cp, w2, h1, h2, h3⟩ := ih rest
      exact ⟨w + cp, w2, EdgesFrom.cons hw h1, h2, by omega⟩

theorem edgesFrom_snoc_intro {wt a l r cp w} (h : EdgesFrom wt a l cp) (hw : wt (lastOr a l) r = some w) :
    EdgesFrom wt a (l ++ [r]) (cp + w) := by
  have := h.append (EdgesFrom.cons hw (EdgesFrom.nil (wt := wt) (a := r)))
  simpa using this

theorem edgesFrom_take {wt a l c} (h : EdgesFrom wt a l c) : ∀ k, ∃ c' ≤ c, EdgesFrom wt a (l.take k) c' := by
  induction h with
  | nil => intro k; exact ⟨0, Nat.le_refl _, by simpa using EdgesFrom.nil⟩
  | @cons a x l w c hw rest ih =>
    intro k
    cases k with
    | zero => exact ⟨0, Nat.zero_le _, by simpa using EdgesFrom.nil⟩
    | succ k =>
      obtain ⟨c', hc', hr⟩ := ih k
      exact ⟨w + c', by omega, by simpa using EdgesFrom.cons hw hr⟩

theorem lastOr_take_succ (a : Nat) (l : List Nat) (k : Nat) (hk : k < l.length) :
    lastOr a (l.take (k + 1)) = l[k] := by
  induction l generalizing a k with
  | nil => simp at hk
  | cons y l ih =>
    cases k with
    | zero => simp [lastOr]
    | succ k => simp [lastOr]; exact ih y k (by simpa using hk)

/-- The invariant of the loop after the prefix `pre` (non-empty): the output is a walk of no greater weight, with
    the same first and last node, and no node twice. -/
def EraseInv (wt : Nat → Nat → Wt) (pre out : List Nat) : Prop :=
  pre ≠ [] ∧ out ≠ [] ∧ out.head? = pre.head? ∧ out.getLast? = pre.getLast? ∧ out.Nodup ∧
    ∀ c, WalkR wt pre c → ∃ c' ≤ c, WalkR wt out c'

theorem getLast?_cons_eq (a : Nat) (l : List Nat) : (a :: l).getLast? = some (lastOr a l) := by
  induction l generalizing a with
  | nil => simp [lastOr]
  | cons y l ih => simp [lastOr, List.getLast?_cons_cons, ih]

theorem eraseStep_inv {wt pre out r} (h : EraseInv wt pre out) : EraseInv wt (pre ++ [r]) (eraseStep out r) := by
  obtain ⟨hpre, hout, hhd, hlast, hnd, hw⟩ := h
  obtain ⟨y, pre', rfl⟩ := List.exists_cons_of_ne_nil hpre
  obtain ⟨x, o', rfl⟩ := List.exists_cons_of_ne_nil hout
  have hwalk : ∀ c, WalkR wt ((y :: pre') ++ [r]) c → ∃ cp w, EdgesFrom wt y pre' cp ∧
      wt (lastOr y pre') r = some w ∧ c = cp + w := by
    intro c ⟨x', l', hl, hed⟩
    simp only [List.cons_append, List.cons.injEq] at hl
    obtain ⟨rfl, hl⟩ := hl
    rw [← hl] at hed
    exact edgesFrom_snoc hed
  have hxy : x = y := by simpa using hhd
  subst hxy
  have hlo : lastOr x o' = lastOr x pre' := by
    rw [getLast?_cons_eq, getLast?_cons_eq] at hlast
    exact Option.some.inj hlast
  have hsnoc : ((x :: pre') ++ [r]).getLast? = some r := List.getLast?_concat ..
  have hsnoc2 : ((x :: pre') ++ [r]).head? = some x := by simp
  unfold eraseStep
  by_cases hr : r ∈ x :: o'
  · rw [ifP hr]
    have hidx : (x :: o').idxOf r < (x :: o').length := List.idxOf_lt_length_iff.2 hr
    have hget : (x :: o')[(x :: o').idxOf r] = r := by
      have := List.getElem_idxOf hidx
      simpa using this
    have hlen : ((x :: o').take ((x :: o').idxOf r + 1)).length = (x :: o').idxOf r + 1 := by
      rw [List.length_take]; omega
    refine ⟨by simp, ?_, ?_, ?_, ?_, ?_⟩
    · intro hnil
      have := congrArg List.length hnil
      rw [hlen] at this; simp at this
    · rw [hsnoc2]; simp [List.take_succ_cons]
    · rw [hsnoc, List.getLast?_eq_getElem?, hlen]
      simp only [Nat.add_sub_cancel, List.getElem?_take]
      rw [ifP (by omega), List.getElem?_eq_getElem hidx, hget]
    · exact List.Nodup.sublist (List.take_sublist _ _) hnd
    · intro c hc
      obtain ⟨cp, w, h1, h2, h3⟩ := hwalk c hc
      obtain ⟨c1, hc1, ⟨x', l1, hx, hed⟩⟩ := hw cp ⟨x, pre', rfl, h1⟩
      simp only [List.cons.injEq] at hx
      obtain ⟨rfl, rfl⟩ := hx
      obtain ⟨c2, hc2, hed2⟩ := edgesFrom_take hed ((x :: o').idxOf r)
      exact ⟨c2, by omega, x, _, by simp [List.take_succ_cons], hed2⟩
  · rw [ifN hr]
    refine ⟨by simp, by simp, ?_, ?_, ?_, ?_⟩
    · rw [hsnoc2]; simp
    · show ((x :: o') ++ [r]).getLast? = ((x :: pre') ++ [r]).getLast?
      rw [hsnoc]; exact List.getLast?_concat ..
    · exact List.nodup_append.2 ⟨hnd, by simp, fun a ha b hb hab => by
        simp at hb; subst hb; exact hr (hab ▸ ha)⟩
    · intro c hc
      obtain ⟨cp, w, h1, h2, h3⟩ := hwalk c hc
      obtain ⟨c1, hc1, ⟨x', l1, hx, hed⟩⟩ := hw cp ⟨x, pre', rfl, h1⟩
      simp only [List.cons.injEq] at hx
      obtain ⟨rfl, rfl⟩ := hx
      rw [← hlo] at h2
      exact ⟨c1 + w, by omega, x, o' ++ [r], by simp, edgesFrom_snoc_intro hed h2⟩

theorem eraseFold_inv (wt) : ∀ (rest pre out : List Nat), EraseInv wt pre out →
    EraseInv wt (pre ++ rest) (rest.foldl eraseStep out) := by
  intro rest
  induction rest with
  | nil => intro pre out h; simpa using h
  | cons r rest ih =>
    intro pre out h
    have := ih (pre ++ [r]) (eraseStep out r) (eraseStep_inv h)
    simpa using this

/-- **`erase_loops` keeps a valid walk**: from the same first node to the same last node, no node twice, over
    original edges only, and no dearer than the route it cleans (a cycle on a walk only adds weight). -/
theorem eraseLoops_spec (wt : Nat → Nat → Wt) (x : Nat) (rest : List Nat) (c : Nat) (hw : WalkR wt (x :: rest) c) :
    (eraseLoops (x :: rest)).head? = some x ∧ (eraseLoops (x :: rest)).getLast? = (x :: rest).getLast? ∧
      (eraseLoops (x :: rest)).Nodup ∧ ∃ c' ≤ c, WalkR wt (eraseLoops (x :: rest)) c' := by
  have h0 : EraseInv wt [x] [x] := by
    refine ⟨by simp, by simp, rfl, rfl, by simp, ?_⟩
    intro c ⟨y, l, hl, hed⟩
    simp only [List.cons.injEq] at hl
    obtain ⟨rfl, rfl⟩ := hl
    cases hed
    exact ⟨0, Nat.le_refl _, x, [], rfl, EdgesFrom.nil⟩
  have := eraseFold_inv wt rest [x] [x] h0
  have e : eraseLoops (x :: rest) = rest.foldl eraseStep [x] := by
    simp [eraseLoops, eraseStep]
  rw [e]
  obtain ⟨_, _, hhd, hlast, hnd, hwk⟩ := this
  refine ⟨by simpa using hhd, by simpa using hlast, hnd, ?_⟩
  exact hwk c (by simpa using hw)

theorem unpackArcs_append (up M f) : ∀ (L1 L2 : List (Nat × Nat)) (l1 l2 : List Nat),
    unpackArcs up M f L1 = some l1 → unpackArcs up M f L2 = some l2 → unpackArcs up M f (L1 ++ L2) = some (l1 ++ l2) := by
  intro L1
  induction L1 with
  | nil => intro L2 l1 l2 h1 h2; simp [unpackArcs] at h1; subst h1; simpa using h2
  | cons y L1 ih =>
    intro L2 l1 l2 h1 h2
    obtain ⟨a, b⟩ := y
    simp only [unpackArcs] at h1
    cases hu : unpackRec up M f a b with
    | none => rw [hu] at h1; simp at h1
    | some u =>
      cases hr : unpackArcs up M f L1 with
      | none => rw [hu, hr] at h1; simp at h1
      | some r =>
        rw [hu, hr] at h1
        simp only [Option.some.injEq] at h1
        subst h1
        have := ih L2 r l2 hr h2
        simp [unpackArcs, hu, this, List.append_assoc]

theorem edges_lt {up wt A N} (h : Ctx up wt A N) {a l c} (he : EdgesFrom wt a l c) (ha : a < N) :
    ∀ x ∈ l, x < N := by
  induction he with
  | nil => intro x hx; simp at hx
  | @cons a x l w c hw rest ih =>
    have harc := h.edgeArc a x w hw
    have hxN : x < N := by
      rcases harc.2 with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact h.lt _ _ h2
      · omega
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact hxN
    · exact ih hxN y hy

/-- A non-empty walk of original edges is a `WP` (all its vertices are below `N`). -/
theorem edges_wp {up wt A N} (h : Ctx up wt A N) {a l c} (he : EdgesFrom wt a l c) (ha : a < N) (hl : l ≠ []) :
    WP wt N a (lastOr a l) c := by
  induction he with
  | nil => exact absurd rfl hl
  | @cons a x l w c hw rest ih =>
    cases l with
    | nil =>
      cases rest
      simpa [lastOr] using (WP.edge hw : WP wt N a x w)
    | cons y l' =>
      have hxN := edges_lt h (EdgesFrom.cons hw rest) ha x (by simp)
      have := WP.step hw hxN (ih hxN (by simp))
      simpa [lastOr] using this

/-! ## `Cch::query` end to end (`query.rs:231-345`), in rank space -/

/-- The search state after both halves and the climb, exactly as `query` builds it: the chains, the shared count,
    phase one (`relax_from` over each chain below the lowest common ancestor), then the climb over the shared ranks. -/
def queryState {up wt A N} (h : Ctx up wt A N) (s t : Nat) : LS :=
  let f := chain h s
  let b := chain h t
  let c := commonCount f b
  let ls0 : LS := ⟨(f.take (f.length - c)).foldl (fun d o => relaxFrom up (fun x y => A x y) o d) (DS.init s),
    (b.take (b.length - c)).foldl (fun d o => relaxFrom up (fun x y => A y x) o d) (DS.init t), none, none⟩
  (f.drop (f.length - c)).foldl (fun l o => stepL up A o l) ls0

/-- The route and its unpacking (`query.rs:309-345`): the forward arcs, then the backward arcs reversed and flipped,
    each unpacked, the nodes joined and cleaned of loops. `None` if nothing met or an arc is missing. -/
def queryNodes (up : Nat → List Nat) (M : Nat → Nat → Option Nat) (N s : Nat) (ls : LS) : Option (List Nat) :=
  match ls.meet with
  | none => none
  | some m =>
    (unpackArcs up M (2 * N + 1)
      (arcsUp ls.F.pr (m + 1) m ++ (arcsUp ls.B.pr (m + 1) m).reverse.map (fun p => (p.2, p.1)))).map
      (fun l => eraseLoops (s :: l))

theorem queryState_eq {up wt A N} (h : Ctx up wt A N) (s t : Nat) :
    queryState h s t =
      let f := chain h s
      let b := chain h t
      let c := commonCount f b
      (f.drop (f.length - c)).foldl (fun l o => stepL up A o l)
        ⟨(f.take (f.length - c)).foldl (fun d o => gstep up (fun x y => A x y) none o d) (DS.init s),
          (b.take (b.length - c)).foldl (fun d o => gstep up (fun x y => A y x) none o d) (DS.init t), none, none⟩ := by
  unfold queryState
  have e1 : (fun d o => relaxFrom up (fun x y => A x y) o d) = (fun d o => gstep up (fun x y => A x y) none o d) := by
    funext d o; exact relaxFrom_eq_gstep _ _ _ _
  have e2 : (fun d o => relaxFrom up (fun x y => A y x) o d) = (fun d o => gstep up (fun x y => A y x) none o d) := by
    funext d o; exact relaxFrom_eq_gstep _ _ _ _
  simp only [e1, e2]

/-- **`Cch::query` returns a shortest path** (over `Nat` weights, rank space; `s ≠ t`, both below `N`).
    `best` is the true distance (`IsMin`: the cheapest walk of the original graph, none iff there is no walk), the
    returned nodes are a duplicate-free walk of original edges from `s` to `t` whose weight is exactly `best`, and
    `None` is returned exactly when no walk exists. The hypotheses are what Phases 1 and 2 establish
    (`Ctx`, `customize_correct`'s middle invariant `hM`). -/
theorem query_correct {up wt A N} (h : Ctx up wt A N) (M : Nat → Nat → Option Nat)
    (hM : ∀ u v, ArcP up u v → (M u v = none → A u v = wt u v) ∧
      ∀ m, M u v = some m → m < min u v ∧ ArcP up u m ∧ ArcP up m v ∧
        ∃ e1 e2, A u m = some e1 ∧ A m v = some e2 ∧ A u v = some (e1 + e2))
    (s t : Nat) (hst : s ≠ t) (hs : s < N) (ht : t < N) :
    let ls := queryState h s t
    IsMin wt N s t ls.best ∧
    (ls.best = none → queryNodes up M N s ls = none) ∧
    (∀ d, ls.best = some d → ∃ nodes l, queryNodes up M N s ls = some nodes ∧ nodes = s :: l ∧
      nodes.getLast? = some t ∧ nodes.Nodup ∧ EdgesFrom wt s l d) := by
  intro ls
  obtain ⟨hsplit1, hsplit2, hcomm⟩ := shared_spec h s t
  have hq : ls = (chain h s |>.drop ((chain h s).length - commonCount (chain h s) (chain h t))).foldl
      (fun l o => stepL up A o l)
      ⟨((chain h s).take ((chain h s).length - commonCount (chain h s) (chain h t))).foldl
          (fun d o => gstep up (fun x y => A x y) none o d) (DS.init s),
        ((chain h t).take ((chain h t).length - commonCount (chain h s) (chain h t))).foldl
          (fun d o => gstep up (fun x y => A y x) none o d) (DS.init t), none, none⟩ := by
    show queryState h s t = _
    rw [queryState_eq]
  obtain ⟨hmin, hmeet, inv⟩ := query_core h s t hst hs ht _ _ _ hsplit1 hsplit2 hcomm
  rw [← hq] at hmin hmeet inv
  refine ⟨hmin, fun hb => ?_, fun d hd => ?_⟩
  · have : ls.meet = none := hmeet.2 hb
    simp [queryNodes, this]
  · rcases inv.bi with ⟨hb, _⟩ | ⟨m, e1, e2, hb, hm, hmP, hf, hbm⟩
    · rw [hb] at hd; cases hd
    · rw [hb] at hd
      cases hd
      have arcF := arcsUp_spec h inv.F m e1 hf (m + 1) (by omega)
      have arcB := arcsUp_spec h inv.B m e2 hbm (m + 1) (by omega)
      have hf2 : 2 * N < 2 * N + 1 := by omega
      obtain ⟨l1, hl1, hv1, hw1⟩ := unpack_fwd h M hM s (2 * N + 1) hf2 arcF
      obtain ⟨l2, hl2, hv2, hw2⟩ := unpack_bwd h M hM t (2 * N + 1) hf2 arcB
      have hl : unpackArcs up M (2 * N + 1)
          (arcsUp ls.F.pr (m + 1) m ++ (arcsUp ls.B.pr (m + 1) m).reverse.map (fun p => (p.2, p.1))) =
          some (l1 ++ l2) := unpackArcs_append up M _ _ _ _ _ hl1 hl2
      have hwalk : EdgesFrom wt s (l1 ++ l2) (e1 + e2) := hw1.append (by rw [hv1]; exact hw2)
      have hlast : lastOr s (l1 ++ l2) = t := by rw [lastOr_append, hv1]; exact hv2
      have hnodes : queryNodes up M N s ls = some (eraseLoops (s :: (l1 ++ l2))) := by
        rw [queryNodes, hm]; dsimp only; rw [hl]; rfl
      obtain ⟨hh, hlt, hnd, c', hc', hwk⟩ := eraseLoops_spec wt s (l1 ++ l2) (e1 + e2) ⟨s, l1 ++ l2, rfl, hwalk⟩
      obtain ⟨x, l', hx, hed⟩ := hwk
      have hxs : x = s := by
        rw [hx] at hh; simpa using hh
      subst hxs
      have hlt' : (x :: l').getLast? = some t := by
        rw [← hx, hlt, getLast?_cons_eq, hlast]
      have hne : l' ≠ [] := by
        intro hl'
        subst hl'
        simp at hlt'
        exact hst hlt'
      have hwp := edges_wp h hed hs hne
      have hlast' : lastOr x l' = t := by
        rw [getLast?_cons_eq] at hlt'; exact Option.some.inj hlt'
      rw [hlast'] at hwp
      obtain ⟨e, he, hle⟩ := hmin.1 c' hwp
      have hee : e = e1 + e2 := by
        have := Option.some.inj (hb.symm.trans he)
        omega
      refine ⟨eraseLoops (x :: (l1 ++ l2)), l', hnodes, hx, ?_, ?_, ?_⟩
      · rw [hlt, getLast?_cons_eq, hlast]
      · exact hnd
      · have : c' = e1 + e2 := by omega
        rw [← this]; exact hed

/-! ## The whole pipeline: skeleton, order, customization, query -/

theorem pth_lt (adj : Nat → List Nat) (n : Nat) (hN : ∀ u v, v ∈ adj u → v < n) {b u v : Nat}
    (p : Pth adj b u v) : v < n := by
  induction p with
  | edge hv => exact hN _ _ hv
  | step _ _ _ ih => exact ih

/-- **Phases 1 and 2 give the query's context**: on a symmetric skeleton `adj` (every entry below `n`) and edge
    weights `w` inside it, the chordal rows, the original weights (diagonal dropped) and the customized arc
    weights satisfy `Ctx`, and the customization's middle invariant holds. -/
theorem ctx_of_customize (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (n : Nat)
    (hN : ∀ u v, v ∈ adj u → v < n) (w : Nat → Nat → Wt) (hcov : ∀ u v c, seedA w u v = some c → v ∈ adj u) :
    ∃ h : Ctx (rowOf adj) (seedA w) (customizeRun (rowOf adj) w n).1 n,
      ∀ u v, ArcP (rowOf adj) u v → ((customizeRun (rowOf adj) w n).2 u v = none →
          (customizeRun (rowOf adj) w n).1 u v = seedA w u v) ∧
        ∀ m, (customizeRun (rowOf adj) w n).2 u v = some m → m < min u v ∧ ArcP (rowOf adj) u m ∧
          ArcP (rowOf adj) m v ∧ ∃ e1 e2, (customizeRun (rowOf adj) w n).1 u m = some e1 ∧
            (customizeRun (rowOf adj) w n).1 m v = some e2 ∧
            (customizeRun (rowOf adj) w n).1 u v = some (e1 + e2) := by
  have hlt : ∀ r x, x ∈ rowOf adj r → x < n := fun r x hx =>
    pth_lt adj n hN ((row_sound adj hs r x hx).2)
  have harcN : ∀ u v, ArcP (rowOf adj) u v → u < n ∧ v < n := by
    intro u v ⟨_, hh⟩
    rcases hh with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact ⟨by have := hlt _ _ h2; omega, hlt _ _ h2⟩
    · exact ⟨hlt _ _ h2, by have := hlt _ _ h2; omega⟩
  have hc := customize_correct adj hs w hcov n
  refine ⟨⟨row_gt adj hs, hlt, fun x r w' hr hw hrw => up_closure adj (r - x) x r w' rfl
      (row_gt adj hs _ _ hr) hr hw hrw, ?_, ?_, rowOf_sorted adj⟩, ?_⟩
  · intro u v harc
    obtain ⟨hu, hv⟩ := harcN u v harc
    exact (hc u v hu hv harc).1
  · intro u v c he
    have hne : u ≠ v := by
      intro heq; subst heq; simp [seedA] at he
    have hv := hcov u v c he
    by_cases huv : u < v
    · exact ⟨hne, Or.inl ⟨huv, (mem_rowOf adj u v).2 (Or.inl ⟨hv, huv⟩)⟩⟩
    · have hvu : v < u := by omega
      exact ⟨hne, Or.inr ⟨hvu, (mem_rowOf adj v u).2 (Or.inl ⟨hs _ _ hv, hvu⟩)⟩⟩
  · intro u v harc
    obtain ⟨hu, hv⟩ := harcN u v harc
    have := hc u v hu hv harc
    exact ⟨this.2.1, this.2.2⟩

/-- **End to end: `Cch::query` on a customized hierarchy returns a shortest path.** For a symmetric skeleton `adj`,
    edge weights `w` inside it, and two distinct ranks `s`, `t` below `n`, the query's final `best` is the true
    shortest-path distance in the original graph, and the nodes it returns are a duplicate-free walk of original
    edges from `s` to `t` of exactly that weight (none iff there is no walk). Weights are `Nat`; ranks are the
    elimination order's positions (`Topology::rank_of`/`node_of` translate, `invert_spec`). -/
theorem cch_query_end_to_end (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (n : Nat)
    (hN : ∀ u v, v ∈ adj u → v < n) (w : Nat → Nat → Wt) (hcov : ∀ u v c, seedA w u v = some c → v ∈ adj u)
    (s t : Nat) (hst : s ≠ t) (hsn : s < n) (htn : t < n) :
    ∃ (h : Ctx (rowOf adj) (seedA w) (customizeRun (rowOf adj) w n).1 n),
      let ls := queryState h s t
      IsMin (seedA w) n s t ls.best ∧
      (ls.best = none → queryNodes (rowOf adj) (customizeRun (rowOf adj) w n).2 n s ls = none) ∧
      (∀ d, ls.best = some d → ∃ nodes l, queryNodes (rowOf adj) (customizeRun (rowOf adj) w n).2 n s ls = some nodes ∧
        nodes = s :: l ∧ nodes.getLast? = some t ∧ nodes.Nodup ∧ EdgesFrom (seedA w) s l d) := by
  obtain ⟨h, hM⟩ := ctx_of_customize adj hs n hN w hcov
  exact ⟨h, query_correct h _ hM s t hst hsn htn⟩

end NI
