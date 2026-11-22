import FalkorNativeIndex.CchOrder
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# CCH topology, as the Rust builds it (`cch/order.rs`, `cch/nd.rs`)
-/
namespace NI

/-! ## `chordal_supergraph` with its linked lists (`order.rs:381-428`) -/

/-- The loop's state: the finished rows, each rank's first elimination-tree child (`first_child`) and the next
    sibling of each rank (`next_sibling`). -/
structure CS where
  rows : List (List Nat)
  fc : Nat → Option Nat
  ns : Nat → Option Nat

/-- `while let Some(child) = next_child { ...; next_child = next_sibling[child] }`, as a list of the children
    visited. `fuel` bounds the walk (the lists are finite: `walk_chn`). -/
def walk (ns : Nat → Option Nat) : Nat → Option Nat → List Nat
  | 0, _ => []
  | _ + 1, none => []
  | f + 1, some c => c :: walk ns f (ns c)

/-- The row `chordal_supergraph` builds for rank `r` (`order.rs:403-421`): its higher skeleton neighbours, each
    child's row less `r` (the children walked from `first_child[r]`), sorted and deduplicated. -/
def csRow (adj : Nat → List Nat) (s : CS) (r : Nat) : List Nat :=
  sortDedup ((adj r).filter (fun h => decide (r < h)) ++
    (walk s.ns s.rows.length (s.fc r)).flatMap (fun c => (s.rows.getD c []).filter (fun h => h != r)))

/-- One iteration of the loop for rank `r` (`order.rs:403-428`): the row is pushed, and the rank becomes the first
    child of its parent (the first entry of its row), the old first child its next sibling. -/
def csStep (adj : Nat → List Nat) (s : CS) (r : Nat) : CS :=
  match (csRow adj s r).head? with
  | some p => ⟨s.rows ++ [csRow adj s r], fun q => if q = p then some r else s.fc q,
      fun q => if q = r then s.fc p else s.ns q⟩
  | none => ⟨s.rows ++ [csRow adj s r], s.fc, s.ns⟩

def csInit : CS := ⟨[], fun _ => none, fun _ => none⟩

/-- The loop over ranks `0 .. n-1`. -/
def csRun (adj : Nat → List Nat) (n : Nat) : CS := (List.range n).foldl (csStep adj) csInit

/-- A linked list: starting at `o`, following `ns`, visits exactly `L`. -/
inductive Chn (ns : Nat → Option Nat) : Option Nat → List Nat → Prop where
  | nil : Chn ns none []
  | cons {c : Nat} {L : List Nat} : Chn ns (ns c) L → Chn ns (some c) (c :: L)

theorem Chn.congr {ns ns' : Nat → Option Nat} {o : Option Nat} {L : List Nat} (h : Chn ns o L)
    (hag : ∀ c ∈ L, ns' c = ns c) : Chn ns' o L := by
  induction h with
  | nil => exact .nil
  | @cons c L _ ih =>
    refine .cons ?_
    rw [hag c (by simp)]
    exact ih (fun d hd => hag d (List.mem_cons_of_mem _ hd))

theorem walk_chn (ns : Nat → Option Nat) {o : Option Nat} {L : List Nat} (h : Chn ns o L) :
    ∀ f, L.length ≤ f → walk ns f o = L := by
  induction h with
  | nil => intro f _; cases f <;> rfl
  | @cons c L _ ih =>
    intro f hf
    cases f with
    | zero => simp at hf
    | succ f => simp only [walk]; rw [ih f (by simpa using hf)]

/-- The invariant of the loop after `k` ranks: the rows are `buildRows`, and each rank's list is exactly its
    children so far (the ranks below `k` whose finished row starts with it), each once. -/
def CSInv (adj : Nat → List Nat) (k : Nat) (s : CS) : Prop :=
  s.rows = buildRows adj k ∧
  ∀ p, ∃ L, Chn s.ns (s.fc p) L ∧ L.Nodup ∧ ∀ c, c ∈ L ↔ (c < k ∧ (rowOf adj c).head? = some p)

theorem csInv_init (adj : Nat → List Nat) : CSInv adj 0 csInit := by
  refine ⟨rfl, fun p => ⟨[], .nil, List.nodup_nil, fun c => by simp⟩⟩

theorem nodup_length_le : ∀ (k : Nat) (L : List Nat), L.Nodup → (∀ c ∈ L, c < k) → L.length ≤ k := by
  intro k
  induction k with
  | zero => intro L _ hs; cases L with | nil => simp | cons a t => exact absurd (hs a (by simp)) (by omega)
  | succ k ih =>
    intro L hn hs
    by_cases hk : k ∈ L
    · have h1 := List.length_erase_of_mem hk
      have h2 : (L.erase k).length ≤ k := ih _ (hn.erase k) (fun c hc => by
        have := (List.mem_erase_of_ne (by intro h; exact (List.Nodup.mem_erase_iff hn |>.1 hc).1 h)).1 hc
        have := hs c this
        have hne := (List.Nodup.mem_erase_iff hn |>.1 hc).1
        omega)
      have : 0 < L.length := List.length_pos_of_mem hk
      omega
    · have := ih L hn (fun c hc => by have := hs c hc; have : c ≠ k := fun h => hk (h ▸ hc); omega)
      omega

theorem sortDedup_congr (l l' : List Nat) (h : ∀ e, e ∈ l ↔ e ∈ l') : sortDedup l = sortDedup l' :=
  sortDedup_unique l' (sortDedup l) (sortDedup_spec l).1 (fun e => by rw [(sortDedup_spec l).2, h])

theorem csRow_eq (adj : Nat → List Nat) (k : Nat) (s : CS) (h : CSInv adj k s) : csRow adj s k = rowOf adj k := by
  obtain ⟨hrows, hch⟩ := h
  have hlen : s.rows.length = k := by rw [hrows, buildRows_length]
  obtain ⟨L, hL, hnd, hmem⟩ := hch k
  have hw : walk s.ns s.rows.length (s.fc k) = L :=
    walk_chn s.ns hL _ (by rw [hlen]; exact nodup_length_le k L hnd (fun c hc => ((hmem c).1 hc).1))
  unfold csRow rowOf rowAt
  rw [hw]
  apply sortDedup_congr; intro e
  simp only [List.mem_append, List.mem_flatMap, List.mem_filter, List.mem_range, decide_eq_true_eq, hrows]
  constructor
  · rintro (h | ⟨c, hc, he⟩)
    · exact Or.inl h
    · obtain ⟨hc1, hc2⟩ := (hmem c).1 hc
      refine Or.inr ⟨c, ⟨hc1, ?_⟩, he⟩
      rw [rows_getD adj k c hc1]; exact hc2
  · rintro (h | ⟨c, ⟨hc1, hc2⟩, he⟩)
    · exact Or.inl h
    · rw [rows_getD adj k c hc1] at hc2 he
      exact Or.inr ⟨c, (hmem c).2 ⟨hc1, hc2⟩, by rw [rows_getD adj k c hc1]; exact he⟩

theorem csStep_inv (adj : Nat → List Nat) (k : Nat) (s : CS) (h : CSInv adj k s) :
    CSInv adj (k + 1) (csStep adj s k) := by
  have hrow := csRow_eq adj k s h
  obtain ⟨hrows, hch⟩ := h
  have hrows' : s.rows ++ [rowOf adj k] = buildRows adj (k + 1) := by rw [hrows]; rfl
  unfold csStep
  rw [hrow]
  rcases hh : (rowOf adj k).head? with _ | p0
  · refine ⟨hrows', fun p => ?_⟩
    obtain ⟨L, hL, hnd, hmem⟩ := hch p
    refine ⟨L, hL, hnd, fun c => ?_⟩
    rw [hmem c]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨by omega, h2⟩
    · rintro ⟨h1, h2⟩
      refine ⟨?_, h2⟩
      by_cases hc : c = k
      · subst hc; rw [hh] at h2; simp at h2
      · omega
  · refine ⟨hrows', fun p => ?_⟩
    obtain ⟨L, hL, hnd, hmem⟩ := hch p
    have hlt : ∀ c ∈ L, c < k := fun c hc => ((hmem c).1 hc).1
    have hag : ∀ c ∈ L, (fun q => if q = k then s.fc p0 else s.ns q) c = s.ns c := by
      intro c hc; have := hlt c hc; simp only [ifN (by omega : c ≠ k)]
    by_cases hp : p = p0
    · subst hp
      refine ⟨k :: L, ?_, ?_, fun c => ?_⟩
      · simp only [ite_true]
        refine .cons ?_
        simp only [ite_true]
        exact hL.congr hag
      · exact List.nodup_cons.2 ⟨fun hk => by have := hlt k hk; omega, hnd⟩
      · rw [List.mem_cons, hmem c]
        constructor
        · rintro (rfl | ⟨h1, h2⟩)
          · exact ⟨by omega, hh⟩
          · exact ⟨by omega, h2⟩
        · rintro ⟨h1, h2⟩
          by_cases hc : c = k
          · exact Or.inl hc
          · exact Or.inr ⟨by omega, h2⟩
    · refine ⟨L, ?_, hnd, fun c => ?_⟩
      · simp only [ifN hp]
        exact hL.congr hag
      · rw [hmem c]
        constructor
        · rintro ⟨h1, h2⟩; exact ⟨by omega, h2⟩
        · rintro ⟨h1, h2⟩
          refine ⟨?_, h2⟩
          by_cases hc : c = k
          · subst hc; rw [hh] at h2; simp at h2; exact absurd h2.symm hp
          · omega

theorem csRun_inv (adj : Nat → List Nat) : ∀ n, CSInv adj n (csRun adj n) := by
  intro n
  induction n with
  | zero => exact csInv_init adj
  | succ n ih =>
    unfold csRun at *
    rw [List.range_succ, List.foldl_append]
    exact csStep_inv adj n _ ih

/-- **`chordal_supergraph`'s loop, with its linked lists, builds exactly `buildRows`**: rank `r`'s row is `rowOf r`,
    the model of `CchOrder.lean` (children = the ranks whose row starts with `r`), whose rows are the fill-in
    (`row_iff`). -/
theorem csRun_rows (adj : Nat → List Nat) (n : Nat) : (csRun adj n).rows = buildRows adj n := (csRun_inv adj n).1

/-! ## The flat layout: offsets and neighbours (`RankRows`, `row_slots`, `order.rs:152-165`, `:324`) -/

/-- The loop's two pushes after each row (`neighbors.extend_from_slice(&row); offsets.push(neighbors.len())`). -/
def pushRow (st : List Nat × List Nat) (row : List Nat) : List Nat × List Nat :=
  (st.1 ++ [st.2.length + row.length], st.2 ++ row)

/-- `offsets` and `neighbors` after all rows (`offsets` starts `[0]`). -/
def flatRows (rows : List (List Nat)) : List Nat × List Nat := rows.foldl pushRow ([0], [])

/-- The offset of row `r`: the entries before it. -/
def offAt {α : Type} (rows : List (List α)) (r : Nat) : Nat := ((rows.take r).flatten).length

def stateOf (done : List (List Nat)) : List Nat × List Nat :=
  ((List.range (done.length + 1)).map (offAt done), done.flatten)

theorem pushRow_stateOf (done : List (List Nat)) (x : List Nat) :
    pushRow (stateOf done) x = stateOf (done ++ [x]) := by
  have e1 : (List.range (done.length + 1)).map (offAt (done ++ [x])) =
      (List.range (done.length + 1)).map (offAt done) := by
    apply List.map_congr_left
    intro r hr
    have : r ≤ done.length := by have := List.mem_range.1 hr; omega
    simp [offAt, List.take_append_of_le_length this]
  have e2 : offAt (done ++ [x]) (done.length + 1) = done.flatten.length + x.length := by
    have : (done ++ [x]).take (done.length + 1) = done ++ [x] := List.take_of_length_le (by simp)
    simp [offAt, this, List.length_flatten]
  simp only [stateOf, pushRow, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil]
  refine Prod.ext ?_ rfl
  show _ ++ _ = List.map _ (List.range ((done ++ [x]).length + 1))
  rw [show (done ++ [x]).length + 1 = (done.length + 1) + 1 by simp, List.range_succ (n := done.length + 1),
    List.map_append, e1]
  simp only [List.map_cons, List.map_nil, e2]

theorem foldl_pushRow (rest : List (List Nat)) : ∀ done : List (List Nat),
    rest.foldl pushRow (stateOf done) = stateOf (done ++ rest) := by
  induction rest with
  | nil => intro done; simp
  | cons x xs ih =>
    intro done
    simp only [List.foldl_cons, pushRow_stateOf]
    rw [ih]; simp

theorem flatRows_eq (rows : List (List Nat)) : flatRows rows = stateOf rows := by
  have := foldl_pushRow rows []
  simpa [flatRows, stateOf, offAt] using this

theorem offAt_succ {α : Type} (rows : List (List α)) (r : Nat) (h : r < rows.length) :
    offAt rows (r + 1) = offAt rows r + rows[r].length := by
  unfold offAt
  rw [List.take_succ_eq_append_getElem h, List.flatten_append, List.length_append]
  simp

/-- **A row is the slice between its two offsets**: `neighbors[offsets[r]..offsets[r + 1]]` is row `r`. -/
theorem slice_row {α : Type} (rows : List (List α)) (r : Nat) (h : r < rows.length) :
    ((rows.flatten).drop (offAt rows r)).take (offAt rows (r + 1) - offAt rows r) = rows[r] := by
  have hsplit : rows = rows.take r ++ rows[r] :: rows.drop (r + 1) := by
    rw [← List.drop_eq_getElem_cons h, List.take_append_drop]
  have hflat : rows.flatten = (rows.take r).flatten ++ (rows[r] ++ (rows.drop (r + 1)).flatten) := by
    have := congrArg List.flatten hsplit
    rw [List.flatten_append, List.flatten_cons] at this
    exact this
  rw [offAt_succ rows r h, Nat.add_sub_cancel_left, hflat]
  have : (rows.take r).flatten.length = offAt rows r := rfl
  rw [← this, List.drop_left, List.take_left']
  rfl

/-! ## `invert` (`order.rs:333-358`): the order is a permutation, or it is refused

`node_to_rank: Vec<Option<Rank>>` is modelled as a function from a node to `Some slot` inside the vector
(`slot : Option Nat`, `None` = not yet set) and `None` outside it (the `get_mut` miss). -/

/-- `vec![None; node_count]`. -/
def slotsInit (n : Nat) : Nat → Option (Option Nat) := fun x => if x < n then some none else none

/-- The loop `for (index, &node) in rank_to_node.iter().enumerate()`; `None` is an `Err`. -/
def invertGo : (Nat → Option (Option Nat)) → Nat → List Nat → Option (Nat → Option (Option Nat))
  | sl, _, [] => some sl
  | sl, i, node :: rest =>
    match sl node with
    | some none => invertGo (fun x => if x = node then some (some i) else sl x) (i + 1) rest
    | some (some _) => none
    | none => none

/-- `invert`: the length check, the loop, then `rank.expect(..)` on every slot (`getD 0` stands for the
    panic, which `invert_slots_set` shows is never reached). -/
def invert (r2n : List Nat) (n : Nat) : Option (List Nat) :=
  if r2n.length ≠ n then none else
  (invertGo (slotsInit n) 0 r2n).map (fun sl => (List.range n).map (fun x => ((sl x).bind id).getD 0))

/-- After the nodes `pre` (distinct, in range), slot `x` holds its index in `pre`. -/
def InvertInv (n : Nat) (pre : List Nat) (sl : Nat → Option (Option Nat)) : Prop :=
  pre.Nodup ∧ (∀ x ∈ pre, x < n) ∧ ∀ x, sl x = if x < n then some (pre.findIdx? (fun y => y == x)) else none

theorem invertGo_spec (n : Nat) : ∀ (rest pre : List Nat) (sl : Nat → Option (Option Nat)), InvertInv n pre sl →
    (invertGo sl pre.length rest = none ↔ ¬((pre ++ rest).Nodup ∧ ∀ x ∈ rest, x < n)) ∧
    ∀ sl', invertGo sl pre.length rest = some sl' → InvertInv n (pre ++ rest) sl' := by
  intro rest
  induction rest with
  | nil =>
    intro pre sl hi
    refine ⟨?_, ?_⟩
    · simp only [invertGo, reduceCtorEq, false_iff, List.append_nil, List.not_mem_nil, imp_false, implies_true,
        and_true, Classical.not_not]
      exact ⟨hi.1, fun x hx => absurd hx (by simp)⟩
    · intro sl' h; simp only [invertGo, Option.some.injEq] at h; subst h; simpa using hi
  | cons node rest ih =>
    intro pre sl hi
    obtain ⟨hnd, hlt, hsl⟩ := hi
    by_cases hn : node < n
    · by_cases hm : node ∈ pre
      · have hj : ∃ j, pre.findIdx? (fun y => y == node) = some j := by
          cases h : pre.findIdx? (fun y => y == node) with
          | none => rw [List.findIdx?_eq_none_iff] at h; exact absurd (h node hm) (by simp)
          | some j => exact ⟨j, rfl⟩
        obtain ⟨j, hj⟩ := hj
        have hs : sl node = some (some j) := by rw [hsl, ite_eq_left_of_eq_true _ _ (by simp [hn]), hj]
        simp only [invertGo, hs]
        refine ⟨by simp only [true_iff]; intro ⟨h, _⟩; exact (List.nodup_append.1 h).2.2 node hm node (by simp) rfl, ?_⟩
        intro sl' h; simp at h
      · have hnone : pre.findIdx? (fun y => y == node) = none := by
          rw [List.findIdx?_eq_none_iff]; intro y hy; simp; intro h; exact hm (h ▸ hy)
        have hs : sl node = some none := by rw [hsl, ite_eq_left_of_eq_true _ _ (by simp [hn]), hnone]
        let sl' : Nat → Option (Option Nat) := fun x => if x = node then some (some pre.length) else sl x
        have hi' : InvertInv n (pre ++ [node]) sl' := by
          refine ⟨?_, ?_, ?_⟩
          · simp [List.nodup_append, hnd]
            intro a ha h; exact hm (h ▸ ha)
          · intro x hx; simp at hx; rcases hx with hx | rfl
            · exact hlt x hx
            · exact hn
          · intro x
            by_cases hx : x = node
            · subst hx
              simp [sl', List.findIdx?_append, hnone, hn]
            · have : [node].findIdx? (fun y => y == x) = none := by
                simp [List.findIdx?_cons]; intro h; exact hx h.symm
              simp only [sl', ite_eq_right_iff.2 (fun h => absurd h hx), hsl x, List.findIdx?_append, this]
              simp
        have hrec := ih (pre ++ [node]) sl' hi'
        have hlen : (pre ++ [node]).length = pre.length + 1 := by simp
        rw [hlen] at hrec
        have hrest : (pre ++ [node]) ++ rest = pre ++ node :: rest := by simp
        rw [hrest] at hrec
        simp only [invertGo, hs]
        refine ⟨?_, ?_⟩
        · rw [hrec.1]
          constructor
          · intro h ⟨h1, h2⟩; exact h ⟨h1, fun x hx => h2 x (List.mem_cons_of_mem _ hx)⟩
          · intro h ⟨h1, h2⟩; exact h ⟨h1, fun x hx => by
              rcases List.mem_cons.1 hx with rfl | hx
              · exact hn
              · exact h2 x hx⟩
        · intro s'' h; exact hrec.2 s'' h
    · have hs : sl node = none := by rw [hsl, ite_eq_right_of_eq_false _ _ (by simp [hn])]
      simp only [invertGo, hs]
      refine ⟨by simp only [true_iff]; intro ⟨_, h2⟩; exact hn (h2 node (by simp)), ?_⟩
      intro s'' h; simp at h

/-- A list of `n` distinct numbers below `n` holds every number below `n`. -/
theorem nodup_complete (L : List Nat) (n : Nat) (hl : L.length = n) (hn : L.Nodup) (hlt : ∀ x ∈ L, x < n) :
    ∀ x, x < n → x ∈ L := by
  intro x hx
  refine Classical.byContradiction (fun hxn => ?_)
  have h1 : (x :: L).Nodup := List.nodup_cons.2 ⟨hxn, hn⟩
  have h2 := List.Nodup.length_le_of_subset h1 (l₂ := List.range n) (fun y hy => by
    rcases List.mem_cons.1 hy with rfl | hy
    · exact List.mem_range.2 hx
    · exact List.mem_range.2 (hlt y hy))
  simp at h2; omega

theorem invert_none_iff (r2n : List Nat) (n : Nat) :
    invert r2n n = none ↔ ¬(r2n.length = n ∧ r2n.Nodup ∧ ∀ x ∈ r2n, x < n) := by
  by_cases hl : r2n.length = n
  · have hi : InvertInv n [] (slotsInit n) := ⟨List.nodup_nil, by simp, fun x => by simp [slotsInit]⟩
    have := (invertGo_spec n r2n [] (slotsInit n) hi)
    simp only [List.length_nil, List.nil_append] at this
    unfold invert
    simp only [hl, ne_eq, not_true_eq_false, ite_false, true_and]
    cases h : invertGo (slotsInit n) 0 r2n with
    | none => simp only [Option.map_none, true_iff]; exact this.1.1 h
    | some sl =>
      have hc := this.1
      rw [h] at hc
      simp only [Option.map_some, reduceCtorEq, false_iff, Classical.not_not]
      exact Classical.byContradiction (fun hn => by have := hc.2 hn; cases this)
  · unfold invert
    simp [hl]

/-- **`invert` accepts exactly the permutations of `0..n`**, and then returns the inverse: every node has a rank,
    `rank_to_node[node_to_rank[node]] = node`, and the `expect` on each slot never fires. -/
theorem invert_spec (r2n : List Nat) (n : Nat) (n2r : List Nat) (h : invert r2n n = some n2r) :
    r2n.length = n ∧ r2n.Nodup ∧ (∀ x ∈ r2n, x < n) ∧ r2n.Perm (List.range n) ∧ n2r.length = n ∧
    (∀ x, x < n → ∃ j, n2r[x]? = some j ∧ r2n[j]? = some x) ∧
    (∀ j, j < n → ∃ x, r2n[j]? = some x ∧ n2r[x]? = some j) := by
  have hne : invert r2n n ≠ none := by rw [h]; simp
  rw [Ne, invert_none_iff, Classical.not_not] at hne
  obtain ⟨hl, hnd, hlt⟩ := hne
  have hcomp := nodup_complete r2n n hl hnd hlt
  have hi : InvertInv n [] (slotsInit n) := ⟨List.nodup_nil, by simp, fun x => by simp [slotsInit]⟩
  have hs := invertGo_spec n r2n [] (slotsInit n) hi
  simp only [List.length_nil, List.nil_append] at hs
  unfold invert at h
  simp only [hl, ne_eq, not_true_eq_false, ite_false] at h
  cases hg : invertGo (slotsInit n) 0 r2n with
  | none => rw [hg] at h; simp at h
  | some sl =>
    rw [hg] at h
    simp only [Option.map_some, Option.some.injEq] at h
    subst h
    have hinv := hs.2 sl hg
    obtain ⟨_, _, hslot⟩ := hinv
    have hfind : ∀ x, x < n → ∃ j, r2n.findIdx? (fun y => y == x) = some j ∧ r2n[j]? = some x := by
      intro x hx
      have hm := hcomp x hx
      cases hf : r2n.findIdx? (fun y => y == x) with
      | none => rw [List.findIdx?_eq_none_iff] at hf; exact absurd (hf x hm) (by simp)
      | some j =>
        refine ⟨j, rfl, ?_⟩
        obtain ⟨hj, hjx, _⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
        simp only [beq_iff_eq] at hjx
        rw [List.getElem?_eq_getElem hj, hjx]
    have hget : ∀ x, x < n → ∃ j, ((List.range n).map (fun x => ((sl x).bind id).getD 0))[x]? = some j ∧
        r2n.findIdx? (fun y => y == x) = some j := by
      intro x hx
      obtain ⟨j, hj, _⟩ := hfind x hx
      refine ⟨j, ?_, hj⟩
      simp [List.getElem?_range, hx, hslot x, hj]
    refine ⟨hl, hnd, hlt, ?_, by simp, ?_, ?_⟩
    · rw [List.perm_ext_iff_of_nodup hnd List.nodup_range]
      intro x; constructor
      · intro hx; exact List.mem_range.2 (hlt x hx)
      · intro hx; exact hcomp x (List.mem_range.1 hx)
    · intro x hx
      obtain ⟨j, hj, hfj⟩ := hget x hx
      obtain ⟨j', hj', hr⟩ := hfind x hx
      rw [hfj] at hj'; cases hj'
      exact ⟨j, hj, hr⟩
    · intro j hj
      have hjl : j < r2n.length := by omega
      obtain ⟨x, hx⟩ : ∃ x, r2n[j]? = some x := ⟨r2n[j], List.getElem?_eq_getElem hjl⟩
      have hxn : x < n := hlt x (List.mem_of_getElem? hx)
      refine ⟨x, hx, ?_⟩
      obtain ⟨j', hj', hfj⟩ := hget x hxn
      -- the index of x is j (distinctness)
      have : j' = j := by
        obtain ⟨hj1, hj2, _⟩ := List.findIdx?_eq_some_iff_getElem.1 hfj
        have hxj : r2n[j'] = x := by simpa using hj2
        have e1 : r2n[j']? = some x := by rw [List.getElem?_eq_getElem hj1, hxj]
        have hxj' : r2n[j] = x := by
          have := List.getElem?_eq_some_iff.1 hx
          exact this.2
        have a1 := List.Nodup.idxOf_getElem hnd j' hj1
        have a2 := List.Nodup.idxOf_getElem hnd j hjl
        rw [hxj] at a1; rw [hxj'] at a2
        omega
      rw [← this]; exact hj'

/-! ## `transpose` (`order.rs:441-470`)

The counting sort writes each `(lower, higher)` pair of `upward`, in row order, at the next free slot of `higher`'s
downward row (`next_free[higher] = offsets[higher]` plus what is already placed there). Modelled per downward row:
the pair appends `lower` to `higher`'s row. -/

/-- A pair `(lower, higher)` is placed: `lower` goes on the end of `higher`'s row. -/
def place (b : Nat → List Nat) (lower higher : Nat) : Nat → List Nat :=
  fun h => if h = higher then b h ++ [lower] else b h

/-- The downward rows after row `lower` of `upward` is walked. -/
def walkRow (b : Nat → List Nat) (lower : Nat) (row : List Nat) : Nat → List Nat :=
  row.foldl (fun b h => place b lower h) b

/-- All downward rows: `upward`'s rows walked in order `lower = 0, 1, ..`. -/
def downRows (up : List (List Nat)) : Nat → List Nat :=
  (List.range up.length).foldl (fun b l => walkRow b l (up.getD l [])) (fun _ => [])

theorem walkRow_spec (lower : Nat) (row : List Nat) (hn : row.Nodup) : ∀ b : Nat → List Nat, ∀ h,
    walkRow b lower row h = if h ∈ row then b h ++ [lower] else b h := by
  induction row with
  | nil => intro b h; simp [walkRow]
  | cons x xs ih =>
    intro b h
    have hn' := (List.nodup_cons.1 hn)
    unfold walkRow at *
    simp only [List.foldl_cons]
    rw [ih hn'.2 (place b lower x) h]
    by_cases hx : h = x
    · subst hx; simp [place, hn'.1]
    · simp [place, hx]

/-- **Each downward row is the ascending list of the ranks whose upward row holds it**
    (given upward rows without repeats). -/
theorem downRows_spec (up : List (List Nat)) (hn : ∀ r ∈ up, r.Nodup) (h : Nat) :
    downRows up h = (List.range up.length).filter (fun l => decide (h ∈ up.getD l [])) := by
  unfold downRows
  have key : ∀ j, j ≤ up.length → ((List.range j).foldl (fun b l => walkRow b l (up.getD l [])) (fun _ => [])) h =
      (List.range j).filter (fun l => decide (h ∈ up.getD l [])) := by
    intro j
    induction j with
    | zero => intro _; simp
    | succ j ih =>
      intro hj
      rw [List.range_succ, List.foldl_append, List.filter_append]
      simp only [List.foldl_cons, List.foldl_nil]
      have hrow : (up.getD j []).Nodup := by
        have : j < up.length := by omega
        have e : up.getD j [] = up[j] := by simp [List.getD_eq_getElem?_getD, this]
        rw [e]; exact hn _ (List.getElem_mem this)
      rw [walkRow_spec j _ hrow, ih (by omega)]
      by_cases hm : h ∈ up.getD j []
      · rw [ifP hm]; simp only [List.filter_cons, List.filter_nil, hm, decide_true, ite_true]
      · rw [ifN hm]; simp only [List.filter_cons, List.filter_nil, hm, decide_false, ite_false]; simp
  exact key _ (Nat.le_refl _)

/-! ## `Topology` (`order.rs:100-310`) -/

/-- `Topology`: the order both ways, and the two `RankRows` (offsets and neighbours), as lists. -/
structure Topo where
  r2n : List Nat
  n2r : List Nat
  upOff : List Nat
  upNb : List Nat
  dnOff : List Nat
  dnNb : List Nat

/-- `row_slots`: `offsets[r]..offsets[r + 1]`. -/
def rowSlots (off : List Nat) (r : Nat) : Nat × Nat := (off.getD r 0, off.getD (r + 1) 0)

/-- `RankRows::row`: `neighbors[slots(r)]`. -/
def rowOfFlat (off nb : List Nat) (r : Nat) : List Nat :=
  let s := rowSlots off r
  (nb.drop s.1).take (s.2 - s.1)

/-- `Topology::up`, `up_range`, `down`, `arc_count`, `node_count`, `node_of`, `rank_of`, `ranks`. -/
def Topo.up (t : Topo) (r : Nat) : List Nat := rowOfFlat t.upOff t.upNb r
def Topo.upRange (t : Topo) (r : Nat) : Nat × Nat := rowSlots t.upOff r
def Topo.down (t : Topo) (r : Nat) : List Nat := rowOfFlat t.dnOff t.dnNb r
def Topo.arcCount (t : Topo) : Nat := t.upNb.length
def Topo.nodeCount (t : Topo) : Nat := t.r2n.length
def Topo.nodeOf (t : Topo) (r : Nat) : Nat := t.r2n.getD r 0
def Topo.rankOf (t : Topo) (n : Nat) : Nat := t.n2r.getD n 0
def Topo.ranks (t : Topo) : List Nat := List.range t.nodeCount

/-- `binary_search(&x)` on a slice as an index: std's contract on a strictly sorted slice (found at its only
    position, else an `Err`). -/
def BsIdxSpec (bs : List Nat → Nat → Option Nat) : Prop :=
  ∀ l x, l.Pairwise (· < ·) → bs l x = if x ∈ l then some (l.idxOf x) else none

/-- `Topology::arc_slot` (`order.rs:287-300`). -/
def Topo.arcSlot (bs : List Nat → Nat → Option Nat) (t : Topo) (e o : Nat) : Option Nat :=
  let lower := if e < o then e else o
  let higher := if e < o then o else e
  (bs (t.up lower) higher).map (fun off => (t.upRange lower).1 + off)

/-- **`arc_slot`**: the slot of the arc joining two ranks, in either order, is the lower end's first slot plus
    the position of the higher end in its row; `None` iff they are not adjacent. -/
theorem arcSlot_spec (bs : List Nat → Nat → Option Nat) (hbs : BsIdxSpec bs) (t : Topo) (e o : Nat)
    (hs : (t.up (min e o)).Pairwise (· < ·)) :
    t.arcSlot bs e o = if max e o ∈ t.up (min e o) then
      some ((t.upRange (min e o)).1 + (t.up (min e o)).idxOf (max e o)) else none := by
  unfold Topo.arcSlot
  by_cases h : e < o
  · simp only [h, if_true]
    rw [hbs _ _ (by simpa [Nat.min_eq_left (Nat.le_of_lt h)] using hs)]
    simp [Nat.min_eq_left (Nat.le_of_lt h), Nat.max_eq_right (Nat.le_of_lt h)]
  · simp only [h, if_false]
    have hle : o ≤ e := by omega
    rw [hbs _ _ (by simpa [Nat.min_eq_right hle] using hs)]
    simp [Nat.min_eq_right hle, Nat.max_eq_left hle]

theorem nodup_of_sorted {l : List Nat} (h : l.Pairwise (· < ·)) : l.Nodup :=
  List.Pairwise.imp (fun h => Nat.ne_of_lt h) h

/-- `Topology::from_order`'s result: `None` is an `Err`. The skeleton must be well formed and `r2n` a permutation
    (`invert`); the adjacency read in rank space is `adjR`. -/
def adjR (skel : Cch.Csr) (r2n n2r : List Nat) (r : Nat) : List Nat :=
  if r < skel.n then (skel.rowD (r2n.getD r 0)).map (fun x => n2r.getD x 0) else []

def fromOrder (skel : Cch.Csr) (r2n : List Nat) : Option Topo :=
  if skel.isWellFormed = false then none else
  match invert r2n skel.n with
  | none => none
  | some n2r =>
    let rows := (csRun (adjR skel r2n n2r) skel.n).rows
    let up := flatRows rows
    let dn := flatRows ((List.range skel.n).map (downRows rows))
    some ⟨r2n, n2r, up.1, up.2, dn.1, dn.2⟩

/-- **`RankRows::row` on the flat arrays is the row**: the slice `neighbors[offsets[r]..offsets[r+1]]`. -/
theorem rowOfFlat_flatRows (rows : List (List Nat)) (r : Nat) (hr : r < rows.length) :
    rowOfFlat (flatRows rows).1 (flatRows rows).2 r = rows[r] := by
  rw [flatRows_eq]
  unfold rowOfFlat rowSlots stateOf
  simp only
  have e1 : ((List.range (rows.length + 1)).map (offAt rows)).getD r 0 = offAt rows r := by
    simp [List.getD_eq_getElem?_getD, List.getElem?_range, show r < rows.length + 1 by omega]
  have e2 : ((List.range (rows.length + 1)).map (offAt rows)).getD (r + 1) 0 = offAt rows (r + 1) := by
    simp [List.getD_eq_getElem?_getD, List.getElem?_range, show r + 1 < rows.length + 1 by omega]
  rw [e1, e2]
  exact slice_row rows r hr

/-- `from_order` accepts exactly a well-formed skeleton and a permutation of its nodes. -/
theorem fromOrder_none_iff (skel : Cch.Csr) (r2n : List Nat) :
    fromOrder skel r2n = none ↔ ¬(skel.isWellFormed = true ∧ r2n.length = skel.n ∧ r2n.Nodup ∧ ∀ x ∈ r2n, x < skel.n) := by
  unfold fromOrder
  by_cases hw : skel.isWellFormed = true
  · simp only [hw, Bool.true_eq_false, ite_false, true_and]
    cases h : invert r2n skel.n with
    | none => simp only [true_iff]; exact (invert_none_iff _ _).1 h
    | some n2r =>
      simp only [reduceCtorEq, false_iff, Classical.not_not]
      have := invert_spec r2n skel.n n2r h
      exact ⟨this.1, this.2.1, this.2.2.1⟩
  · simp [hw]

theorem fromOrder_eq (skel : Cch.Csr) (r2n n2r : List Nat) (hw : skel.isWellFormed = true)
    (hinv : invert r2n skel.n = some n2r) :
    fromOrder skel r2n =
      some ⟨r2n, n2r, (flatRows (buildRows (adjR skel r2n n2r) skel.n)).1,
        (flatRows (buildRows (adjR skel r2n n2r) skel.n)).2,
        (flatRows ((List.range skel.n).map (downRows (buildRows (adjR skel r2n n2r) skel.n)))).1,
        (flatRows ((List.range skel.n).map (downRows (buildRows (adjR skel r2n n2r) skel.n)))).2⟩ := by
  unfold fromOrder
  simp only [hw, hinv, Bool.true_eq_false, ite_false, csRun_rows]

/-- **The built rows are the chordal rows**: for a well-formed skeleton and a permutation, `up(r)` is the model's
    `rowOf r` of the rank-space adjacency, for every rank, and `down(h)` is the ascending list of ranks whose upward
    row holds `h`. -/
theorem fromOrder_rows (skel : Cch.Csr) (r2n n2r : List Nat) (hw : skel.isWellFormed = true)
    (hinv : invert r2n skel.n = some n2r) (t : Topo) (h : fromOrder skel r2n = some t) :
    t.r2n = r2n ∧ t.n2r = n2r ∧ t.nodeCount = skel.n ∧
    (∀ r, r < skel.n → t.up r = rowOf (adjR skel r2n n2r) r) ∧
    (∀ r, r < skel.n → ∀ x, x ∈ t.down r ↔ x < skel.n ∧ r ∈ t.up x) ∧
    t.arcCount = ((buildRows (adjR skel r2n n2r) skel.n).flatten).length := by
  rw [fromOrder_eq skel r2n n2r hw hinv] at h
  cases h
  have hspec := invert_spec r2n skel.n n2r hinv
  have hlen : ∀ adj : Nat → List Nat, (buildRows adj skel.n).length = skel.n := fun adj => buildRows_length adj skel.n
  generalize adjR skel r2n n2r = adj
  have hup : ∀ r, r < skel.n → rowOfFlat (flatRows (buildRows adj skel.n)).1 (flatRows (buildRows adj skel.n)).2 r =
      rowOf adj r := by
    intro r hr
    rw [rowOfFlat_flatRows _ r (by rw [hlen]; exact hr)]
    have := buildRows_getD adj skel.n r hr
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by rw [hlen]; exact hr)] at this
    simpa [rowOf] using this
  refine ⟨rfl, rfl, by simp [Topo.nodeCount, hspec.1], fun r hr => hup r hr, ?_, ?_⟩
  · intro r hr x
    have hl2 : ((List.range skel.n).map (downRows (buildRows adj skel.n))).length = skel.n := by simp
    unfold Topo.down Topo.up
    simp only
    rw [rowOfFlat_flatRows _ r (by rw [hl2]; exact hr)]
    simp only [List.getElem_map, List.getElem_range]
    have hnd : ∀ q ∈ buildRows adj skel.n, q.Nodup := by
      intro q hq
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hq
      have := buildRows_getD adj skel.n j (by rw [hlen] at hj; exact hj)
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj] at this
      simp only [Option.getD_some] at this
      rw [this]
      exact nodup_of_sorted (rowOf_sorted adj j)
    rw [downRows_spec _ hnd r, List.mem_filter, List.mem_range, decide_eq_true_eq, hlen]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨h1, ?_⟩
      rw [hup x h1]
      have := buildRows_getD adj skel.n x h1
      rw [rows_getD adj skel.n x h1] at h2
      exact h2
    · rintro ⟨h1, h2⟩
      refine ⟨h1, ?_⟩
      rw [hup x h1] at h2
      rw [rows_getD adj skel.n x h1]
      exact h2
  · simp [Topo.arcCount, flatRows_eq, stateOf]

/-- The rank-space adjacency of a symmetric simple skeleton is symmetric on ranks. -/
theorem adjR_symm (skel : Cch.Csr) (r2n n2r : List Nat) (hw : skel.isWellFormed = true)
    (hsym : ∀ u, u < skel.n → ∀ v ∈ skel.rowD u, v ≠ u ∧ u ∈ skel.rowD v)
    (hinv : invert r2n skel.n = some n2r) :
    ∀ u v, u < skel.n → v ∈ adjR skel r2n n2r u → u ∈ adjR skel r2n n2r v ∧ v < skel.n := by
  intro u v hu hv
  obtain ⟨hl, hnd, hlt, _, hl2, hinv1, hinv2⟩ := invert_spec r2n skel.n n2r hinv
  obtain ⟨_, _, _, _, hid⟩ := (Cch.isWellFormed_iff skel).1 hw
  unfold adjR at hv
  rw [ifP hu] at hv
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
  obtain ⟨ru, hru⟩ : ∃ a, r2n[u]? = some a := ⟨r2n[u]'(by omega), List.getElem?_eq_getElem (by omega)⟩
  have hru' : r2n.getD u 0 = ru := by simp [List.getD_eq_getElem?_getD, hru]
  rw [hru'] at hx
  have hru_lt : ru < skel.n := hlt ru (List.mem_of_getElem? hru)
  have hxn : x < skel.n := hid x (Cch.rowD_subset skel ru x hx)
  obtain ⟨hne, hmir⟩ := hsym ru hru_lt x hx
  obtain ⟨j, hj1, hj2⟩ := hinv1 x hxn
  have hn2 : n2r.getD x 0 = j := by simp [List.getD_eq_getElem?_getD, hj1]
  rw [hn2]
  have hjn : j < skel.n := (List.getElem?_eq_some_iff.1 hj2).1 |> fun h => by omega
  refine ⟨?_, hjn⟩
  unfold adjR
  rw [ifP hjn]
  have hrj : r2n.getD j 0 = x := by simp [List.getD_eq_getElem?_getD, hj2]
  rw [hrj]
  refine List.mem_map.2 ⟨ru, hmir, ?_⟩
  obtain ⟨x', h1, h2⟩ := hinv2 u hu
  rw [hru] at h1
  cases h1
  simp [List.getD_eq_getElem?_getD, h2]

/-- **With a symmetric simple skeleton, `from_order`'s upward rows are exactly the fill-in** of the elimination
    game: `w` is in `up(r)` iff `w > r` and `w` is reachable from `r` through ranks below `r` in the skeleton. -/
theorem fromOrder_fill_in (skel : Cch.Csr) (r2n n2r : List Nat) (hw : skel.isWellFormed = true)
    (hsym : ∀ u, u < skel.n → ∀ v ∈ skel.rowD u, v ≠ u ∧ u ∈ skel.rowD v)
    (hinv : invert r2n skel.n = some n2r) (t : Topo) (h : fromOrder skel r2n = some t) (r w : Nat) (hr : r < skel.n) :
    w ∈ t.up r ↔ r < w ∧ Pth (adjR skel r2n n2r) r r w := by
  obtain ⟨_, _, _, hup, _⟩ := fromOrder_rows skel r2n n2r hw hinv t h
  rw [hup r hr]
  apply row_iff
  intro u v hv
  by_cases hu : u < skel.n
  · exact (adjR_symm skel r2n n2r hw hsym hinv u v hu hv).1
  · simp [adjR, hu] at hv

/-! ## `nested_dissection_order` (`nd.rs:46-125`)

The ordering backend (`feral_metis::metis_order`) is another crate; it is modelled as an arbitrary function from
the renumbered skeleton (`m`, offsets, neighbours) to an order, or a refusal, and every statement below holds
for every backend. -/

/-- `MAX_NODES` = `GrB_INDEX_MAX` = `2^60 - 1`. -/
def MAX_NODES : Nat := 2 ^ 60 - 1

/-- `connected.get(index).ok_or(..)` over the backend's ids, in order; `None` is the out-of-range error. -/
def mapIds (c : List Nat) : List Nat → Option (List Nat)
  | [] => some []
  | i :: r =>
    match c[i]?, mapIds c r with
    | some x, some xs => some (x :: xs)
    | _, _ => none

theorem mapIds_spec (c : List Nat) : ∀ ord : List Nat,
    mapIds c ord = if ∀ i ∈ ord, i < c.length then some (ord.map (fun i => c.getD i 0)) else none := by
  intro ord
  induction ord with
  | nil => simp [mapIds]
  | cons i r ih =>
    unfold mapIds
    rw [ih]
    by_cases hi : i < c.length
    · have e : c[i]? = some c[i] := List.getElem?_eq_getElem hi
      by_cases hr : ∀ j ∈ r, j < c.length
      · have this : ∀ j ∈ i :: r, j < c.length := by
          intro j hj; rcases List.mem_cons.1 hj with rfl | hj; exact hi; exact hr j hj
        rw [ifP hr, e, ifP this]
        simp [List.getD_eq_getElem?_getD, e]
      · have this : ¬ ∀ j ∈ i :: r, j < c.length := fun h => hr (fun j hj => h j (List.mem_cons_of_mem _ hj))
        rw [ifN hr, e, ifN this]
    · have e : c[i]? = none := List.getElem?_eq_none (by omega)
      have this : ¬ ∀ j ∈ i :: r, j < c.length := fun h => hi (h i (by simp))
      rw [e, ifN this]

/-- `nested_dissection_order` (`None` is an `Err`): the checks, the isolated nodes first, one backend call on the
    connected nodes renumbered `0..m`, its order mapped back. -/
def ndOrder (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (skel : Cch.Csr) : Option (List Nat) :=
  if skel.n > MAX_NODES then none
  else if skel.isWellFormed = false then none
  else if skel.isSymmetricSimple bs = false then none
  else
    let parts := (List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)
    if parts.2.isEmpty then some parts.1 else
    let flat := flatRows (parts.2.map (fun node => (skel.rowD node).map (fun nb => parts.2.idxOf nb)))
    match backend parts.2.length flat.1 flat.2 with
    | none => none
    | some ord =>
      if ord.length ≠ parts.2.length then none
      else (mapIds parts.2 ord).map (fun xs => parts.1 ++ xs)

theorem partition_perm (n : Nat) (p : Nat → Bool) :
    ((List.range n).partition p).1 ++ ((List.range n).partition p).2 |>.Perm (List.range n) := by
  rw [List.partition_eq_filter_filter]
  exact List.filter_append_perm _ _

/-- **`nested_dissection_order` returns an order of exactly the skeleton's nodes whenever the backend returns a
    permutation of the connected nodes' ids**, and whatever the backend does, a result is only produced with the
    right length, and every entry a node (`< n`); the checked order is then also verified by `from_order`'s
    `invert`. -/
theorem ndOrder_range (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (skel : Cch.Csr) (o : List Nat) (h : ndOrder backend bs skel = some o) : ∀ x ∈ o, x < skel.n := by
  unfold ndOrder at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  dsimp only at h
  have hperm := partition_perm skel.n (fun node => (skel.rowD node).isEmpty)
  have hsub : ∀ x, x ∈ ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).1 ∨
      x ∈ ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).2 → x < skel.n := by
    intro x hx
    have : x ∈ ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).1 ++
        ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).2 := List.mem_append.2 hx
    exact List.mem_range.1 (hperm.subset this)
  split at h
  · cases h
    intro x hx; exact hsub x (Or.inl hx)
  · split at h
    · cases h
    · rename_i ord hord
      split at h
      · cases h
      · intro x hx
        cases hm : mapIds ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).2 ord with
        | none => rw [hm] at h; simp at h
        | some xs =>
          rw [hm] at h
          simp only [Option.map_some, Option.some.injEq] at h
          subst h
          rcases List.mem_append.1 hx with hx | hx
          · exact hsub x (Or.inl hx)
          · rw [mapIds_spec] at hm
            by_cases hall : ∀ i ∈ ord, i < ((List.range skel.n).partition (fun node => (skel.rowD node).isEmpty)).2.length
            · rw [ifP hall] at hm
              simp only [Option.some.injEq] at hm
              subst hm
              obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
              have := hall i hi
              refine hsub _ (Or.inr ?_)
              rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem this]
              exact List.getElem_mem this
            · rw [ifN hall] at hm; cases hm

theorem ndOrder_checks (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (skel : Cch.Csr) (o : List Nat) (h : ndOrder backend bs skel = some o) :
    skel.isWellFormed = true ∧ skel.isSymmetricSimple bs = true ∧ ¬ skel.n > MAX_NODES := by
  unfold ndOrder at h
  by_cases h1 : skel.n > MAX_NODES
  · simp [h1] at h
  by_cases h2 : skel.isWellFormed = false
  · simp [h1, h2] at h
  by_cases h3 : skel.isSymmetricSimple bs = false
  · simp [h1, h2, h3] at h
  exact ⟨by simpa using h2, by simpa using h3, h1⟩

/-- `Topology::build` (`order.rs:180-185`): the order, then the elimination game. -/
def topoBuild (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (skel : Cch.Csr) : Option Topo :=
  (ndOrder backend bs skel).bind (fromOrder skel)

/-- **`Topology::build` is correct for every backend**: when it succeeds, the order is a permutation of the
    nodes (checked by `invert`, whatever the backend returned), and `up(r)` is exactly the fill-in of the
    elimination game, for every rank. The skeleton's symmetry is not assumed: `nested_dissection_order` checked it. -/
theorem topoBuild_correct (backend : Nat → List Nat → List Nat → Option (List Nat)) (bs : List Nat → Nat → Bool)
    (hbs : Cch.BsSpec bs) (skel : Cch.Csr) (t : Topo) (h : topoBuild backend bs skel = some t) :
    ∃ n2r, t.r2n.Perm (List.range skel.n) ∧ invert t.r2n skel.n = some n2r ∧ t.nodeCount = skel.n ∧
      ∀ r, r < skel.n → ∀ w, w ∈ t.up r ↔ r < w ∧ Pth (adjR skel t.r2n n2r) r r w := by
  unfold topoBuild at h
  cases ho : ndOrder backend bs skel with
  | none => rw [ho] at h; simp at h
  | some o =>
    rw [ho] at h
    simp only [Option.bind_some] at h
    obtain ⟨hw, hsy, _⟩ := ndOrder_checks backend bs skel o ho
    have hsym := ((Cch.isSymmetricSimple_iff bs hbs skel).1 hsy).2
    have hsym' : ∀ u, u < skel.n → ∀ v ∈ skel.rowD u, v ≠ u ∧ u ∈ skel.rowD v := fun u hu => (hsym u hu).2
    have hne : fromOrder skel o ≠ none := by rw [h]; simp
    rw [Ne, fromOrder_none_iff, Classical.not_not] at hne
    obtain ⟨_, hl, hnd, hlt⟩ := hne
    obtain ⟨n2r, hinv⟩ : ∃ n2r, invert o skel.n = some n2r := by
      cases hi : invert o skel.n with
      | none => exact absurd hi (by rw [invert_none_iff, Classical.not_not]; exact ⟨hl, hnd, hlt⟩)
      | some n2r => exact ⟨n2r, rfl⟩
    obtain ⟨hr, hn2, hnc, hup, _⟩ := fromOrder_rows skel o n2r hw hinv t h
    refine ⟨n2r, ?_, ?_, hnc, ?_⟩
    · rw [hr]; exact (invert_spec o skel.n n2r hinv).2.2.2.1
    · rw [hr]; exact hinv
    · intro r hr' w
      rw [hr, hn2] at *
      exact fromOrder_fill_in skel o n2r hw hsym' hinv t h r w hr'

/-! ## Index conversions (`mod.rs:58-66`, `order.rs:317`, `nd.rs:126`)

`id_index` (`usize::try_from(u64)`), `index_id` (`u64::try_from(usize)`), `node_index` and `backend_index`
(`i64::try_from(usize)`) only change the width of a number that is below `MAX_NODES < 2^60`, so no conversion
fails on a 64-bit target: each is the identity on `Nat` below that bound. -/

def idIndex (x : Nat) : Nat := x
def indexId (x : Nat) : Nat := x
def nodeIndex (x : Nat) : Nat := idIndex x
def backendIndex (x : Nat) : Nat := x

/-- The conversions compose to the identity, and every id below `MAX_NODES` fits an `i64`, a `u64`, a `usize`. -/
theorem conversions_roundtrip (x : Nat) (h : x < MAX_NODES) :
    idIndex (indexId x) = x ∧ nodeIndex x = x ∧ backendIndex x = x ∧ x < 2 ^ 63 ∧ x < 2 ^ 64 := by
  refine ⟨rfl, rfl, rfl, ?_, ?_⟩ <;> (unfold MAX_NODES at h; omega)

end NI
