import FalkorNativeIndex.Memory
import FalkorNativeIndex.Scan
import FalkorNativeIndex.Unique
import FalkorNativeIndex.Decline
import FalkorNativeIndex.Cch
import FalkorNativeIndex.Rebase
set_option linter.unusedSimpArgs false

namespace NI

theorem ctn_ite (p : Prop) [Decidable p] (A : KeySet) (k : Nat) :
    (if p then A else KeySet.empty).contains k = (decide p && A.contains k) := by
  by_cases h : p <;> simp [h, KeySet.contains]

theorem point_contains (kk k : Nat) : (KeySet.point kk).contains k = (kk == k) := rfl
theorem keys_contains (ks : List Nat) (k : Nat) : (KeySet.keys ks).contains k = decide (k ∈ ks) := rfl

theorem ofKeys_filter_contains (ks : List Nat) (o : KeySet) (k : Nat) :
    (KeySet.ofKeys (ks.filter (fun x => o.contains x))).contains k = (decide (k ∈ ks) && o.contains k) := by
  rw [ofKeys_contains]
  by_cases h1 : k ∈ ks <;> by_cases h2 : o.contains k = true <;> simp [List.mem_filter, h1, h2]

/-- The `Point` arm of `and`: the point, if the other side holds it. -/
theorem point_arm (o : KeySet) (ka k : Nat) :
    (if o.contains ka = true then KeySet.point ka else KeySet.empty).contains k = (ka == k && o.contains k) := by
  by_cases h2 : o.contains ka = true
  · rw [if_pos h2]
    by_cases h : ka = k
    · subst h; simp only [point_contains, beq_self_eq_true, h2, Bool.and_self]
    · simp [KeySet.contains, beq_false_of_ne h]
  · rw [if_neg h2]
    by_cases h : ka = k
    · subst h; simp only [Bool.not_eq_true] at h2
      rw [h2, beq_self_eq_true]; rfl
    · simp [KeySet.contains, beq_false_of_ne h]

/-- A key in a window. -/
def inWin (k : Nat) (w : Nat × Nat) : Bool := decide (w.1 ≤ k ∧ k ≤ w.2)

/-- **`of_windows` holds exactly the keys in its windows**, whichever shape it picks. -/
theorem ofWindows_contains (ws : List (Nat × Nat)) (k : Nat) :
    (KeySet.ofWindows ws).contains k = ws.any (inWin k) := by
  unfold KeySet.ofWindows
  split
  · rfl
  · rename_i lo hi
    split
    · rename_i h; subst h
      rw [Bool.eq_iff_iff]; simp [KeySet.contains, inWin]; omega
    · simp [KeySet.contains, inWin]
  · rfl

/-- Sorted, disjoint windows (`cover_of`'s and `of_windows`'s invariant): each window is a range and
    starts after the one before it ends. -/
def WinsSorted (ws : List (Nat × Nat)) : Prop := (∀ w ∈ ws, w.1 ≤ w.2) ∧ ws.Pairwise (fun a b => a.2 < b.1)

private theorem none_after (k : Nat) : ∀ (ws : List (Nat × Nat)), WinsSorted ws →
    ws.takeWhile (fun w => decide (w.1 ≤ k)) = [] → ws.any (inWin k) = false
  | [], _, _ => rfl
  | r :: rest, ⟨hw, hp⟩, ht => by
    have hr : ¬ r.1 ≤ k := by
      intro h; simp [List.takeWhile_cons, h] at ht
    rw [List.any_eq_false]
    intro x hx
    simp only [inWin, decide_eq_true_eq, not_and, Nat.not_le]
    rcases List.mem_cons.1 hx with rfl | hx
    · intro h; omega
    · have := List.rel_of_pairwise_cons hp hx
      have := hw r List.mem_cons_self
      intro h; omega

/-- **`KeySet::contains` on `Windows` (`partition_point`, then the last window starting at or below the
    key) is "some window holds the key"**, the model's `contains`, on sorted disjoint windows. -/
theorem containsW_eq_any (k : Nat) : ∀ (ws : List (Nat × Nat)), WinsSorted ws →
    KeySet.containsW ws k = (KeySet.windows ws).contains k
  | [], _ => rfl
  | w :: rest, ⟨hw, hp⟩ => by
    have hs : WinsSorted rest := ⟨fun x hx => hw x (List.mem_cons_of_mem _ hx), (List.pairwise_cons.1 hp).2⟩
    have ih := containsW_eq_any k rest hs
    simp only [KeySet.contains] at ih ⊢
    unfold KeySet.containsW at ih ⊢
    by_cases h1 : w.1 ≤ k
    · rw [List.takeWhile_cons, if_pos (by simpa using h1)]
      cases ht : rest.takeWhile (fun w => decide (w.1 ≤ k)) with
      | nil =>
        have : rest.any (fun w => decide (w.1 ≤ k ∧ k ≤ w.2)) = false := none_after k rest hs ht
        rw [List.any_cons, this, Bool.or_false]
        simp [h1]
      | cons r t =>
        rw [ht] at ih
        have hl : (w :: r :: t).getLast? = (r :: t).getLast? := by simp [List.getLast?_cons]
        rw [hl, ih, List.any_cons]
        have hrk : r ∈ rest ∧ r.1 ≤ k := by
          cases rest with
          | nil => simp at ht
          | cons r' rest' =>
            rw [List.takeWhile_cons] at ht
            by_cases h' : r'.1 ≤ k
            · rw [if_pos (by simpa using h')] at ht
              obtain ⟨rfl, -⟩ := List.cons.inj ht
              exact ⟨List.mem_cons_self, h'⟩
            · rw [if_neg (by simpa using h')] at ht; simp at ht
        obtain ⟨hr, hrk⟩ := hrk
        have := List.rel_of_pairwise_cons hp hr
        have hn : decide (w.1 ≤ k ∧ k ≤ w.2) = false := by simp; omega
        rw [hn, Bool.false_or]
    · rw [List.takeWhile_cons, if_neg (by simpa using h1)]
      have : (w :: rest).any (fun w => decide (w.1 ≤ k ∧ k ≤ w.2)) = false :=
        none_after k (w :: rest) ⟨hw, hp⟩ (by rw [List.takeWhile_cons, if_neg (by simpa using h1)])
      rw [this]; rfl

/-- **The pairwise overlaps hold exactly the keys both window lists hold.** -/
theorem overlaps_any (a b : List (Nat × Nat)) (k : Nat) :
    (overlaps a b).any (inWin k) = (a.any (inWin k) && b.any (inWin k)) := by
  rw [Bool.eq_iff_iff]
  simp only [overlaps, List.any_eq_true, List.mem_flatMap, List.mem_filterMap, Bool.and_eq_true, inWin,
    decide_eq_true_eq]
  constructor
  · rintro ⟨w, ⟨x, hx, y, hy, hw⟩, h1, h2⟩
    split at hw
    · cases hw; exact ⟨⟨x, hx, by simp at h1 h2; omega⟩, ⟨y, hy, by simp at h1 h2; omega⟩⟩
    · cases hw
  · rintro ⟨⟨x, hx, h1⟩, ⟨y, hy, h2⟩⟩
    refine ⟨(max x.1 y.1, min x.2 y.2), ⟨x, hx, y, hy, ?_⟩, ?_⟩
    · rw [if_pos (by omega)]
    · simp; omega

theorem wins_contains_range (lo hi k : Nat) : (KeySet.range lo hi).contains k = (KeySet.range lo hi).wins.any (inWin k) := by
  simp [KeySet.contains, KeySet.wins, inWin]
theorem wins_contains_windows (ws : List (Nat × Nat)) (k : Nat) :
    (KeySet.windows ws).contains k = (KeySet.windows ws).wins.any (inWin k) := rfl

/-- **`KeySet::and` holds exactly the keys both sides hold**, arm by arm (#2527: two sets of windows,
    a range being one, meet in their pairwise overlaps). -/
theorem keyset_and_contains (a b : KeySet) (k : Nat) : (a.and b).contains k = (a.contains k && b.contains k) := by
  cases a <;> cases b
  all_goals simp only [KeySet.and]
  all_goals first
    | (rw [point_arm]; simp [KeySet.contains, Bool.and_comm]; done)
    | (rw [point_arm]; rw [Bool.and_comm]; rfl)
    | (rw [ofKeys_filter_contains]; simp [KeySet.contains, Bool.and_comm]; done)
    | (rw [ofKeys_filter_contains]; simp only [KeySet.contains]; rw [Bool.and_comm]; done)
    | (rw [ofWindows_contains, overlaps_any, wins_contains_range, wins_contains_windows]; done)
    | (rw [ofWindows_contains, overlaps_any, wins_contains_windows, wins_contains_range]; done)
    | (rw [ofWindows_contains, overlaps_any, wins_contains_windows, wins_contains_windows]; done)
    | (simp [KeySet.contains]; done)
    | skip
  all_goals rename_i lo hi lo' hi'
  by_cases hc : max lo lo' ≤ min hi hi'
  · rw [if_pos hc, Bool.eq_iff_iff]
    simp only [KeySet.contains, Bool.and_eq_true, decide_eq_true_eq]; omega
  · rw [if_neg hc, Bool.eq_iff_iff]
    simp only [KeySet.contains, Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true]
    constructor
    · intro h; exact h.elim
    · rintro ⟨⟨h1, h2⟩, h3, h4⟩; omega

/-! ## `atoms` (`range.rs`) -/

mutual
theorem atomsGo_notSame (b : Bool) : ∀ (ms as : List IndexQuery), atomsGo b ms = some as →
    ∀ a ∈ as, sameCombinator b a = false
  | [], as, h => by
    simp only [atomsGo, Option.some.injEq] at h
    subst h; simp
  | m :: ms, as, h => by
    unfold atomsGo at h
    by_cases hs : sameCombinator b m = true
    · rw [if_pos hs] at h
      cases h1 : atoms m with
      | none => simp [h1] at h
      | some a1 =>
        cases h2 : atomsGo b ms with
        | none => simp [h1, h2] at h
        | some r =>
          simp only [h1, h2, Option.bind_some, Option.map_some, Option.some.injEq] at h
          subst h
          intro a ha
          rcases List.mem_append.1 ha with ha | ha
          · exact atoms_same m b a1 hs h1 a ha
          · exact atomsGo_notSame b ms r h2 a ha
    · rw [if_neg hs] at h
      cases h2 : atomsGo b ms with
      | none => simp [h2] at h
      | some r =>
        simp only [h2, Option.map_some, Option.some.injEq] at h
        subst h
        intro a ha
        rcases List.mem_cons.1 ha with rfl | ha
        · simpa using hs
        · exact atomsGo_notSame b ms r h2 a ha
theorem atoms_same : ∀ (m : IndexQuery) (b : Bool) (as : List IndexQuery), sameCombinator b m = true →
    atoms m = some as → ∀ a ∈ as, sameCombinator b a = false
  | .and ms, b, as, hs, h => by
    simp only [sameCombinator] at hs
    subst hs
    unfold atoms at h
    by_cases he : ms.isEmpty = true
    · simp [he] at h
    · rw [if_neg he] at h
      exact atomsGo_notSame true ms as h
  | .or ms, b, as, hs, h => by
    simp only [sameCombinator, Bool.not_eq_true'] at hs
    have hb : b = false := by simpa using hs
    subst hb
    unfold atoms at h
    by_cases he : ms.isEmpty = true
    · simp [he] at h
    · rw [if_neg he] at h
      exact atomsGo_notSame false ms as h
  | .equal _ _, b, as, hs, h => by simp [sameCombinator] at hs
  | .range .., b, as, hs, h => by simp [sameCombinator] at hs
  | .inList .., b, as, hs, h => by simp [sameCombinator] at hs
  | .arrayContains .., b, as, hs, h => by simp [sameCombinator] at hs
  | .point .., b, as, hs, h => by simp [sameCombinator] at hs
end

/-- **No atom of an `AND` is itself an `AND`** (`conjunction`'s `debug_assert!`, `range.rs`): the
    members that are `And`s are opened, so `Selection.ofAtom`'s decline of an `And` conjunct is never
    reached from `Selection.of`. -/
theorem atoms_and_no_and (ms as : List IndexQuery) (h : atoms (.and ms) = some as) :
    ∀ a ∈ as, sameCombinator true a = false := atoms_same (.and ms) true as rfl h

theorem atoms_or_no_or (ms as : List IndexQuery) (h : atoms (.or ms) = some as) :
    ∀ a ∈ as, sameCombinator false a = false := atoms_same (.or ms) false as rfl h

/-! ## `Selection::and` (`range.rs`) -/

theorem encodeNumeric_list (es : List (Option Nat)) : encodeNumeric (.list es) = none := by
  simp [encodeNumeric]

theorem arrayKeys_nl (v : Value) (h : ∀ es, v ≠ .list es) : arrayKeys v = [] := by
  cases v <;> first | exact absurd rfl (h _) | rfl

/-- **`Selection::and` is the intersection of what the two select**: an entity matches the conjunction
    iff it matches both. A value is a scalar or a list, never both, so a scalar and a list selection select
    nothing together. -/
theorem not_list_holding (v : Value) (hnl : ∀ es, v ≠ .list es) (key : Nat) : isListHolding key v = false := by
  cases hh : isListHolding key v with
  | false => rfl
  | true =>
    have := (isListHolding_iff key v).1 hh
    simp [arrayKeys_nl v hnl] at this

theorem selection_and_matches (a b : Selection) (v : Value) :
    (Selection.and a b).matches v = (a.matches v && b.matches v) := by
  cases a with
  | scalars ka =>
    cases b with
    | scalars kb =>
      simp only [Selection.and, Selection.matches, isScalarIn_eq, KeySet.matchesValue]
      cases encodeNumeric v with
      | none => rfl
      | some k => simp [keyset_and_contains]
    | listsHolding f r =>
      simp only [Selection.and, Selection.matches, Selection.empty, isScalarIn_eq, KeySet.matchesValue]
      cases hv : encodeNumeric v with
      | none => simp
      | some k =>
        have hnl : ∀ es, v ≠ .list es := by rintro es rfl; rw [encodeNumeric_list] at hv; cases hv
        simp [not_list_holding v hnl, KeySet.contains]
  | listsHolding f r =>
    cases b with
    | scalars kb =>
      simp only [Selection.and, Selection.matches, Selection.empty, isScalarIn_eq, KeySet.matchesValue]
      cases hv : encodeNumeric v with
      | none => simp
      | some k =>
        have hnl : ∀ es, v ≠ .list es := by rintro es rfl; rw [encodeNumeric_list] at hv; cases hv
        simp [not_list_holding v hnl, KeySet.contains]
    | listsHolding f' r' =>
      simp only [Selection.and, Selection.matches]
      simp [List.all_append, Bool.and_assoc]

/-! ## `DistinctIter` (`numeric.rs`)

The streams of a union across attributes, read one after another, each doc yielded once. State: the
streams not started, the rest of the current stream, the docs yielded so far. -/

/-- `DistinctIter::next`'s loop (`numeric.rs`), with a fuel that every iteration spends (each one
    consumes a doc of the current stream or starts a stream, so it terminates). -/
def distinctGo : Nat → List Nat → List (List Nat) → List Nat →
    Option (Nat × List (List Nat) × List Nat × List Nat)
  | 0, _, _, _ => none
  | n + 1, seen, streams, d :: ds =>
    if d ∈ seen then distinctGo n seen streams ds else some (d, streams, ds, d :: seen)
  | _ + 1, _, [], [] => none
  | n + 1, seen, s :: ss, [] => distinctGo n seen ss s

/-- What the iterator still has to yield: the rest of the current stream, then the later streams, each doc
    once and none already yielded. -/
def distinctRem (seen : List Nat) (streams : List (List Nat)) (cur : List Nat) : List Nat :=
  ((cur ++ streams.flatten).filter (fun x => !seen.contains x)).eraseDups

theorem distinctGo_spec : ∀ (fuel : Nat) (seen : List Nat) (streams : List (List Nat)) (cur : List Nat),
    cur.length + (streams.map (fun s => s.length + 1)).sum < fuel →
    (∀ d st cu se, distinctGo fuel seen streams cur = some (d, st, cu, se) →
      se = d :: seen ∧ distinctRem seen streams cur = d :: distinctRem se st cu) ∧
    (distinctGo fuel seen streams cur = none → distinctRem seen streams cur = []) := by
  intro fuel
  induction fuel with
  | zero => intro seen streams cur h; omega
  | succ n ih =>
    intro seen streams cur hf
    cases cur with
    | cons d ds =>
      by_cases hd : d ∈ seen
      · have hf' : ds.length + (streams.map (fun s => s.length + 1)).sum < n := by
          simp only [List.length_cons] at hf; omega
        obtain ⟨h1, h2⟩ := ih seen streams ds hf'
        have hrem : distinctRem seen streams (d :: ds) = distinctRem seen streams ds := by
          unfold distinctRem
          simp [List.cons_append, List.filter_cons, hd]
        simp only [distinctGo, hd, ite_true]
        refine ⟨fun d' st cu se h => ?_, fun h => ?_⟩
        · obtain ⟨a, b⟩ := h1 d' st cu se h
          exact ⟨a, by rw [hrem]; exact b⟩
        · rw [hrem]; exact h2 h
      · simp only [distinctGo, hd, ite_false]
        refine ⟨fun d' st cu se h => ?_, fun h => by simp at h⟩
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl, rfl⟩ := h
        refine ⟨rfl, ?_⟩
        unfold distinctRem
        have hc : seen.contains d = false := by simpa using hd
        rw [List.cons_append, List.filter_cons, hc]
        simp only [Bool.not_false, ite_true, List.eraseDups_cons]
        congr 1
        rw [List.filter_filter]
        congr 1
        apply List.filter_congr
        intro x _
        by_cases hx : x = d <;> simp [hx, List.contains_cons]
    | nil =>
      cases streams with
      | nil =>
        simp only [distinctGo]
        refine ⟨fun d st cu se h => by simp at h, fun _ => ?_⟩
        simp [distinctRem]
      | cons s ss =>
        have hf' : s.length + (ss.map (fun s => s.length + 1)).sum < n := by
          simp only [List.map_cons, List.sum_cons, List.length_nil] at hf; omega
        obtain ⟨h1, h2⟩ := ih seen ss s hf'
        have hrem : distinctRem seen (s :: ss) [] = distinctRem seen ss s := by
          unfold distinctRem; simp
        simp only [distinctGo]
        refine ⟨fun d' st cu se h => ?_, fun h => ?_⟩
        · obtain ⟨a, b⟩ := h1 d' st cu se h
          exact ⟨a, by rw [hrem]; exact b⟩
        · rw [hrem]; exact h2 h

/-- **`DistinctIter::next` yields the first not-yet-yielded doc and leaves the rest**; `None` only when
    nothing is left. So draining it enumerates each doc of the streams once, at its first appearance
    (`distinctDocs`). -/
theorem distinctIter_next_spec (seen : List Nat) (streams : List (List Nat)) (cur : List Nat) :
    (∀ d st cu se, distinctGo (cur.length + (streams.map (fun s => s.length + 1)).sum + 1) seen streams cur =
      some (d, st, cu, se) → distinctRem seen streams cur = d :: distinctRem se st cu) ∧
    (distinctGo (cur.length + (streams.map (fun s => s.length + 1)).sum + 1) seen streams cur = none →
      distinctRem seen streams cur = []) := by
  obtain ⟨h1, h2⟩ := distinctGo_spec (cur.length + (streams.map (fun s => s.length + 1)).sum + 1) seen streams cur
    (Nat.lt_succ_self _)
  exact ⟨fun d st cu se h => (h1 d st cu se h).2, h2⟩

/-- Started from nothing, the iterator's remaining output is the union `distinctDocs`. -/
theorem distinctRem_initial (streams : List (List Nat)) : distinctRem [] [] [] = [] ∧
    distinctRem [] streams [] = distinctDocs streams := by
  refine ⟨by simp [distinctRem], ?_⟩
  unfold distinctRem distinctDocs
  simp only [List.nil_append]
  congr 1
  exact List.filter_eq_self.2 (fun x _ => by simp)

/-! ## What the builder asks of a version (`versioned.rs`) -/

/-- `Graph::index_generation` (`versioned.rs`): the entry's generation and whether it is `Building`. -/
def indexGeneration (en : Option EN) : Option (Nat × Bool) := en.map (fun e => (e.generation, e.isBuilding))

/-- `Graph::building_since` (`versioned.rs`): the generation of an entry that is `Building` here and,
    in `older`, absent or another generation. -/
def buildingSince (ours older : Option EN) : Option Nat :=
  (ours.filter (fun o => o.isBuilding && older.all (fun t => t.generation != o.generation))).map (·.generation)

theorem buildingSince_iff (ours older : Option EN) (g : Nat) :
    buildingSince ours older = some g ↔
      ∃ o, ours = some o ∧ o.isBuilding = true ∧ o.generation = g ∧ ∀ t, older = some t → t.generation ≠ g := by
  unfold buildingSince
  cases ours with
  | none => simp
  | some o =>
    cases older with
    | none => simp [Option.filter]
    | some t =>
      by_cases hb : o.isBuilding = true <;> by_cases hg : t.generation = o.generation <;>
        simp [Option.filter, hb, hg] <;> omega

/-- `Graph::indexes_to_build_since` (`versioned.rs`): the indexes `ours` holds `Building` that `older`
    does not hold as the same index, with their generations. -/
def indexesToBuildSince {α : Type} (names : List α) (ours older : α → Option EN) : List (α × Nat) :=
  names.filterMap (fun n => (buildingSince (ours n) (older n)).map (fun g => (n, g)))

theorem mem_indexesToBuildSince {α : Type} (names : List α) (ours older : α → Option EN) (n : α) (g : Nat) :
    (n, g) ∈ indexesToBuildSince names ours older ↔ n ∈ names ∧ buildingSince (ours n) (older n) = some g := by
  simp only [indexesToBuildSince, List.mem_filterMap, Option.map_eq_some_iff, Prod.mk.injEq]
  constructor
  · rintro ⟨m, hm, g', hg', rfl, rfl⟩; exact ⟨hm, hg'⟩
  · rintro ⟨hn, hg⟩; exact ⟨n, hn, g, hg, rfl, rfl⟩

/-- With no entry `Building` the answer is empty (the early return of `indexes_to_build_since`). -/
theorem indexesToBuildSince_idle {α : Type} (names : List α) (ours older : α → Option EN)
    (h : ∀ n, ∀ o, ours n = some o → o.isBuilding = false) : indexesToBuildSince names ours older = [] := by
  rw [List.eq_nil_iff_forall_not_mem]
  rintro ⟨n, g⟩ hm
  obtain ⟨_, hg⟩ := (mem_indexesToBuildSince names ours older n g).1 hm
  obtain ⟨o, ho, hb, _⟩ := (buildingSince_iff _ _ _).1 hg
  rw [h n o ho] at hb; cases hb

/-! ## The accessors of the managed index and the builder's sizing -/

/-- `FalkorDbIndex::is_ready` (`falkordb_index.rs`): the column exists and is not `Building`. -/
def FalkorDbIndex.isReady (x : FalkorDbIndex) (e : EntityType) (l a : String) : Bool :=
  (x.entry e l a).any (fun en => !en.isBuilding)

theorem isReady_iff (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.isReady e l a = true ↔ ∃ en, x.entry e l a = some en ∧ en.isBuilding = false := by
  unfold FalkorDbIndex.isReady
  cases h : x.entry e l a <;> simp

/-- `is_ready` is the gate `query_column` answers rows behind: a `Ready` column's answer is rows. -/
theorem isReady_readyNumeric (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.isReady e l a = (x.readyNumeric e l a).isSome := by
  unfold FalkorDbIndex.isReady FalkorDbIndex.readyNumeric
  cases h : x.entry e l a with
  | none => simp
  | some en => cases hb : en.isBuilding <;> simp [hb]

/-- `FalkorDbIndex::has_building_columns` (`falkordb_index.rs`): some column of either kind is
    `Building`. -/
def FalkorDbIndex.hasBuildingColumns (x : FalkorDbIndex) : Bool :=
  x.nodeColumns.any (fun p => p.2.isBuilding) || x.edgeColumns.any (fun p => p.2.isBuilding)

theorem hasBuildingColumns_iff (x : FalkorDbIndex) :
    x.hasBuildingColumns = true ↔
      (∃ p ∈ x.nodeColumns, p.2.isBuilding = true) ∨ (∃ p ∈ x.edgeColumns, p.2.isBuilding = true) := by
  simp [FalkorDbIndex.hasBuildingColumns, List.any_eq_true]

/-- `FalkorDbIndex::column` / `column_mut` (`falkordb_index.rs`) reach the `RangeIndex` of the
    entry (`IndexColumn::Range`, the only variant): the model's `numeric`/`setNumeric`. -/
theorem column_is_numeric (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.numeric e l a = (x.entry e l a).map (fun en => en.index.ix) := rfl

/-- `Sizing::batch` (`online_build.rs`): `read`, between `floor` and `limit`. -/
def sizingBatch (floor read limit : Nat) : Nat := max (min read limit) floor

theorem sizingBatch_spec (floor read limit : Nat) :
    floor ≤ sizingBatch floor read limit ∧ sizingBatch floor read limit ≤ max limit floor ∧
    (floor ≤ read → read ≤ limit → sizingBatch floor read limit = read) := by
  unfold sizingBatch
  refine ⟨by omega, by omega, by intro h1 h2; omega⟩

/-- `RangeIndex::query` (`range.rs`): `Selection::of(query).map(|s| self.lookup(s))`. -/
def NumericIndex.queryRange (ix : NumericIndex) (q : IndexQuery) : Option (List Nat) :=
  (Selection.of q).map ix.lookupSel

theorem queryRange_isSome_iff (ix : NumericIndex) (q : IndexQuery) :
    (ix.queryRange q).isSome = (Selection.of q).isSome := by
  simp [NumericIndex.queryRange]

end NI
