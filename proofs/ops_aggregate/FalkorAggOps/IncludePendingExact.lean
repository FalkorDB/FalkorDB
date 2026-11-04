import FalkorAggOps.StreamOps
/-
# IncludePending composed with `Pending` (#3453, fixed by #3467)

`StreamOps.includePending_spec` proves Phase 1 (`childRows`) and Phase 2
(`pendingRows`) of `IncludePendingOp` separately, each against whatever node
lists it is handed. Here the lists are the ones main hands it, for an
uncorrelated MERGE node scan (`argument_rows` empty, one row per node; the row
is the node id):

| here | there (a477d9aa3) |
| --- | --- |
| `Gc` | the committed graph: live node ids, each node's committed labels |
| `Pd` | `Pending`'s `created_nodes`, `deleted_nodes`, `set_labels`, `remove_labels` (pending.rs) |
| `scan` | the wrapped `NodeByLabelScan` over the committed graph |
| `stage` | `stage_node_labels` (pending.rs), reached by `SetOp` `SetItem::Label` (ops/set.rs) without checking the committed matrix |
| `deletedNode` | `deleted_node`: inserts into `deleted_nodes`, `set_labels` untouched |
| `removedFor` | `nodes_with_pending_label_removes` |
| `pendingWith` | `get_pending_nodes_with_labels(label_ids, g)` (pending.rs:1114): live nodes gaining a pattern label |
| `includePendingMain` | `IncludePendingOp::next` (include_pending.rs:166-224) = `StreamOps.childRows` (skipping deleted, label-removed and `extra_nodes`) then `StreamOps.pendingRows` over `extra_nodes` |

Property `IncludePendingExact`: the emitted node rows are duplicate-free and
are exactly the live nodes (committed or created, not deleted) whose effective
labels — `(committed \ remove_labels) ∪ set_labels`, `set_labels` alone for a
created node — contain the pattern. On b84473a7f the three live repros refuted it
(#3453); #3467 made `get_pending_nodes_with_labels` return exactly the live nodes
that gain a pattern label and made Phase 1 skip them, and `includePendingMain_exact`
proves the property for main: for every non-empty pattern, given the pending maps
are keyed uniquely (`FxHashMap`) and name only committed-or-created nodes (a deleted node may still
appear in `set_labels`), the committed node ids are distinct and created ids are apart from them.
Those are hypotheses about `Pending`/the committed graph (other code), not discharged here, and the
composition covers an uncorrelated scan only (`argument_rows` empty). The three repros are
`fixed3453_deleted_node`, `fixed3453_duplicate_row`, `fixed3453_split_multilabel`.
-/
namespace FalkorAggOps.IncludePendingExact
open FalkorAggOps.StreamOps

/-- The committed graph. -/
structure Gc where
  nodes : List Nat
  labels : Nat → List Nat

/-- The four `Pending` fields the scan reads. -/
structure Pd where
  created : List Nat
  deleted : List Nat
  setL : List (Nat × List Nat)
  remL : List (Nat × List Nat)

def look (m : List (Nat × List Nat)) (n : Nat) : List Nat := ((m.find? (·.1 == n)).map (·.2)).getD []

/-- `NodeByLabelScan` over the committed graph (pattern labels non-empty). -/
def scan (g : Gc) (ls : List Nat) : List Nat := g.nodes.filter fun n => ls.all (· ∈ g.labels n)

/-- `stage_node_labels` (pending.rs:661-676): append to the node's `set_labels`
entry (created if absent), drop those labels from its `remove_labels`. -/
def stage (p : Pd) (n : Nat) (ls : List Nat) : Pd :=
  { p with
    setL := if p.setL.any (·.1 == n) then p.setL.map (fun e => if e.1 == n then (n, e.2 ++ ls) else e)
            else p.setL ++ [(n, ls)],
    remL := (p.remL.map fun e => if e.1 == n then (n, e.2.filter (· ∉ ls)) else e).filter
              fun e => !(e.1 == n && e.2.isEmpty) }

/-- `deleted_node` (pending.rs:772-777). -/
def deletedNode (p : Pd) (n : Nat) : Pd := { p with deleted := if n ∈ p.deleted then p.deleted else p.deleted ++ [n] }

/-- `nodes_with_pending_label_removes` (pending.rs:1060-1076). -/
def removedFor (p : Pd) (ls : List Nat) : List Nat :=
  if ls = [] then [] else (p.remL.filter fun e => ls.any (· ∈ e.2)).map (·.1)

/-- A pattern label `l` is committed on `n` and still there after the staged removals: `n` is not created,
`remove_labels` does not hold `l`, and the graph has it (`node_has_label_id`). -/
def committedL (g : Gc) (p : Pd) (n l : Nat) : Bool :=
  !decide (n ∈ p.created) && !decide (l ∈ look p.remL n) && decide (l ∈ g.labels n)

/-- `get_pending_nodes_with_labels(label_ids, g)` (pending.rs:1114-1146): created nodes for no label; else the
`set_labels` entries that are not pending-deleted, whose every pattern label is committed or staged, and not
all committed. -/
def pendingWith (g : Gc) (p : Pd) (ls : List Nat) : List Nat :=
  if ls = [] then p.created else
    (p.setL.filter fun e => !decide (e.1 ∈ p.deleted) &&
      ls.all (fun l => committedL g p e.1 l || decide (l ∈ e.2)) &&
      !ls.all (fun l => committedL g p e.1 l)).map (·.1)

/-- `IncludePendingOp::next` with no argument rows: Phase 1 = the scan minus deleted, label-removed and
`extra_nodes` (`childRows`), Phase 2 = `extra_nodes` (`pendingRows`, one empty row per node). -/
def includePendingMain (g : Gc) (p : Pd) (ls : List Nat) : List Nat :=
  childRows some p.deleted (removedFor p ls) (pendingWith g p ls) (scan g ls) ++
  pendingRows (fun _ n => n) 0 [] (pendingWith g p ls)

/-- Live after the pending changes. -/
def live (g : Gc) (p : Pd) (n : Nat) : Prop := (n ∈ g.nodes ∨ n ∈ p.created) ∧ n ∉ p.deleted

/-- Effective labels: what `update_node_labels` overlays (pending.rs:754-770). -/
def eff (g : Gc) (p : Pd) (n : Nat) : List Nat :=
  (if n ∈ p.created then [] else (g.labels n).filter (· ∉ look p.remL n)) ++ look p.setL n

/-- **Property `IncludePendingExact`**. -/
def IncludePendingExact (g : Gc) (p : Pd) (ls : List Nat) (rows : List Nat) : Prop :=
  rows.Nodup ∧ ∀ n, n ∈ rows ↔ live g p n ∧ ∀ l ∈ ls, l ∈ eff g p n

def A : Nat := 0
def B : Nat := 1
/-- `CREATE (:A)`: node 0 with label A, nothing pending. -/
def g1 : Gc := ⟨[0], fun n => if n = 0 then [A] else []⟩
def p0 : Pd := ⟨[], [], [], []⟩

/-- **#3453 (1), fixed** `MATCH (n:A) SET n:B DELETE n MERGE (m:B) RETURN m`: the deleted node is not
yielded, so MERGE creates a new `:B`, as C (before: node 0 was matched). -/
theorem fixed3453_deleted_node :
    includePendingMain g1 (deletedNode (stage p0 0 [B]) 0) [B] = [] := by decide

/-- **#3453 (2), fixed** `MATCH (n:A) SET n:A MERGE (m:A) …`: node 0 is emitted once (before: twice). -/
theorem fixed3453_duplicate_row :
    includePendingMain g1 (stage p0 0 [A]) [A] = [0] := by decide

/-- **#3453 (3), fixed** `MATCH (n:A) SET n:B MERGE (m:A:B)`: node 0 (A committed, B staged) is matched
(before: nothing, MERGE created a duplicate). -/
theorem fixed3453_split_multilabel :
    includePendingMain g1 (stage p0 0 [B]) [A, B] = [0] := by decide

/-! ## The proof for main -/

theorem contains_false (l : List Nat) (n : Nat) : l.contains n = false ↔ n ∉ l := by
  rw [← Bool.not_eq_true, List.contains_iff_mem]

theorem childRows_some (d r e rows : List Nat) :
    childRows some d r e rows = rows.filter fun n => !(d.contains n || r.contains n || e.contains n) := rfl

theorem pendingRows_id (l : List Nat) : pendingRows (fun _ n => n) 0 [] l = l := by
  simp [pendingRows]

theorem look_mem (m : List (Nat × List Nat)) (n l : Nat) (h : l ∈ look m n) : n ∈ m.map (·.1) := by
  unfold look at h
  cases hf : m.find? (·.1 == n) with
  | none => simp [hf] at h
  | some e =>
    have hm := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp at he
    exact List.mem_map.mpr ⟨e, hm, he⟩

theorem look_of_mem : ∀ (m : List (Nat × List Nat)), (m.map (·.1)).Nodup → ∀ e ∈ m, look m e.1 = e.2
  | [], _, e, he => by simp at he
  | x :: xs, hk, e, he => by
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hk
    rcases List.mem_cons.mp he with rfl | he
    · simp [look]
    · have hne : (x.1 == e.1) = false := by
        simp only [beq_eq_false_iff_ne, ne_eq]
        intro hx; exact hk.1 ⟨e, he, hx.symm⟩
      have := look_of_mem xs hk.2 e he
      simp only [look, List.find?_cons, hne] at this ⊢
      exact this

theorem mem_removedFor (p : Pd) (ls : List Nat) (n : Nat) (hls : ls ≠ []) (l : Nat) (hl : l ∈ ls)
    (hr : l ∈ look p.remL n) : n ∈ removedFor p ls := by
  unfold look at hr
  cases hf : p.remL.find? (·.1 == n) with
  | none => simp [hf] at hr
  | some e =>
    simp only [hf, Option.map_some, Option.getD_some] at hr
    have hm := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [beq_iff_eq] at he
    simp only [removedFor, hls, ite_false, List.mem_map, List.mem_filter, List.any_eq_true, decide_eq_true_eq]
    exact ⟨e, ⟨hm, l, hl, hr⟩, he⟩

theorem committedL_iff (g : Gc) (p : Pd) (n l : Nat) :
    committedL g p n l = true ↔ n ∉ p.created ∧ l ∉ look p.remL n ∧ l ∈ g.labels n := by
  simp [committedL, and_assoc]

/-- Membership in `pendingWith` for a non-empty pattern (set_labels keyed uniquely). -/
theorem mem_pendingWith (g : Gc) (p : Pd) (ls : List Nat) (hls : ls ≠ []) (hsk : (p.setL.map (·.1)).Nodup)
    (n : Nat) :
    n ∈ pendingWith g p ls ↔ (∃ e ∈ p.setL, e.1 = n) ∧ n ∉ p.deleted ∧
      (∀ l ∈ ls, committedL g p n l = true ∨ l ∈ look p.setL n) ∧ ∃ l ∈ ls, committedL g p n l = false := by
  unfold pendingWith
  simp only [hls, ite_false, List.mem_map, List.mem_filter, Bool.and_eq_true, Bool.not_eq_true',
    List.all_eq_true, decide_eq_true_eq, Bool.or_eq_true, decide_eq_false_iff_not]
  constructor
  · rintro ⟨e, ⟨he, ⟨hd, hl⟩, hnc⟩, rfl⟩
    have hlk := look_of_mem p.setL hsk e he
    refine ⟨⟨e, he, rfl⟩, hd, fun l hl' => ?_, ?_⟩
    · rw [hlk]; exact hl l hl'
    · apply Classical.byContradiction
      intro hno
      have : (ls.all fun l => committedL g p e.1 l) = true := by
        simp only [List.all_eq_true]; intro l hl'
        by_cases hc : committedL g p e.1 l = true
        · exact hc
        · exact absurd ⟨l, hl', by simpa using hc⟩ hno
      rw [this] at hnc; cases hnc
  · rintro ⟨hk, hd, hl, l0, hl0, hc0⟩
    obtain ⟨e, he, hen⟩ := hk
    subst hen
    have hlk := look_of_mem p.setL hsk e he
    refine ⟨e, ⟨he, ⟨hd, fun l hl' => ?_⟩, ?_⟩, rfl⟩
    · rw [← hlk]; exact hl l hl'
    · have : (ls.all fun l => committedL g p e.1 l) = false := by
        simp only [Bool.eq_false_iff, ne_eq, List.all_eq_true]
        intro hall; have := hall l0 hl0; simp [hc0] at this
      simp [this]

/-- **`includePendingMain_exact`** (#3453 fixed by #3467): for every non-empty pattern, when committed node
ids are distinct (`g.nodes.Nodup`), created ids are apart from the committed ones, `set_labels` and `remove_labels` are keyed uniquely (they are `FxHashMap`s) and
`set_labels` names only committed or created nodes, main's operator emits each live node whose effective
labels contain the pattern exactly once, and nothing else. -/
theorem includePendingMain_exact (g : Gc) (p : Pd) (ls : List Nat) (hls : ls ≠ [])
    (hgn : g.nodes.Nodup) (hapart : ∀ n ∈ p.created, n ∉ g.nodes)
    (hrk : (p.remL.map (·.1)).Nodup) (hsk : (p.setL.map (·.1)).Nodup)
    (hsub : ∀ n ∈ p.setL.map (·.1), n ∈ g.nodes ∨ n ∈ p.created) :
    IncludePendingExact g p ls (includePendingMain g p ls) := by
  have hX := mem_pendingWith g p ls hls hsk
  have h1mem : ∀ n, n ∈ childRows some p.deleted (removedFor p ls) (pendingWith g p ls) (scan g ls) ↔
      ((n ∈ g.nodes ∧ ∀ l ∈ ls, l ∈ g.labels n) ∧ n ∉ p.deleted) ∧ n ∉ removedFor p ls ∧ n ∉ pendingWith g p ls := by
    intro n
    simp only [childRows_some, scan, List.mem_filter, Bool.not_eq_true', Bool.or_eq_false_iff, contains_false,
      decide_eq_true_eq, List.all_eq_true, and_assoc]
  rw [includePendingMain, pendingRows_id]
  refine ⟨?_, fun n => ?_⟩
  · rw [List.nodup_append]
    refine ⟨?_, ?_, ?_⟩
    · rw [childRows_some]; exact (hgn.filter _).filter _
    · unfold pendingWith
      split
      · rename_i h; exact absurd h hls
      · exact (List.Nodup.sublist (List.filter_sublist.map _) hsk)
    · intro a ha b hb hab; subst hab
      exact ((h1mem a).1 ha).2.2 hb
  · rw [List.mem_append, h1mem]
    constructor
    · rintro (⟨⟨⟨hn, hl⟩, hd⟩, hr, hx⟩ | hx)
      · refine ⟨⟨.inl hn, hd⟩, fun l hl' => ?_⟩
        have hc : n ∉ p.created := fun hc => hapart n hc hn
        simp only [eff, hc, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq]
        exact .inl ⟨hl l hl', fun hrm => hr (mem_removedFor p ls n hls l hl' hrm)⟩
      · obtain ⟨hk, hd, hl, _⟩ := (hX n).1 hx
        refine ⟨⟨hsub n (List.mem_map.2 hk), hd⟩, fun l hl' => ?_⟩
        rcases hl l hl' with hc | hs
        · obtain ⟨hc1, hc2, hc3⟩ := (committedL_iff g p n l).1 hc
          simp only [eff, hc1, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq]
          exact .inl ⟨hc3, hc2⟩
        · simp only [eff, List.mem_append]; exact .inr hs
    · rintro ⟨⟨hlv, hd⟩, hl⟩
      by_cases hx : n ∈ pendingWith g p ls
      · exact .inr hx
      · left
        -- n is not created: a created node would be in the list
        have hc : n ∉ p.created := by
          intro hc
          apply hx
          obtain ⟨l0, hl0⟩ : ∃ l0, l0 ∈ ls := by
            cases ls with
            | nil => exact absurd rfl hls
            | cons a _ => exact ⟨a, List.mem_cons_self⟩
          have hel : ∀ l ∈ ls, l ∈ look p.setL n := by
            intro l hl'
            have := hl l hl'
            simpa [eff, hc] using this
          refine (hX n).2 ⟨List.mem_map.1 (look_mem _ _ _ (hel l0 hl0)), hd, fun l hl' => .inr (hel l hl'), l0, hl0, ?_⟩
          simp [committedL, hc]
        have hn : n ∈ g.nodes := hlv.resolve_right hc
        -- every pattern label is committed and not removed, or n would be in the list
        have hall : ∀ l ∈ ls, l ∈ g.labels n ∧ l ∉ look p.remL n := by
          intro l0 hl0
          apply Classical.byContradiction
          intro hbad
          apply hx
          have hel : l0 ∈ look p.setL n := by
            have := hl l0 hl0
            simp only [eff, hc, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq] at this
            rcases this with h | h
            · exact absurd ⟨h.1, h.2⟩ hbad
            · exact h
          refine (hX n).2 ⟨List.mem_map.1 (look_mem _ _ _ hel), hd, fun l hl' => ?_, l0, hl0, ?_⟩
          · have := hl l hl'
            simp only [eff, hc, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq] at this
            rcases this with h | h
            · exact .inl ((committedL_iff g p n l).2 ⟨hc, h.2, h.1⟩)
            · exact .inr h
          · apply Classical.byContradiction
            intro hct
            have := (committedL_iff g p n l0).1 (by simpa using hct)
            exact hbad ⟨this.2.2, this.2.1⟩
        refine ⟨⟨⟨hn, fun l hl' => (hall l hl').1⟩, hd⟩, fun hrf => ?_, hx⟩
        simp only [removedFor, hls, ite_false, List.mem_map, List.mem_filter, List.any_eq_true,
          decide_eq_true_eq] at hrf
        obtain ⟨e, ⟨he, l, hl', hle⟩, rfl⟩ := hrf
        exact (hall l hl').2 (by rw [look_of_mem p.remL hrk e he]; exact hle)

end FalkorAggOps.IncludePendingExact
