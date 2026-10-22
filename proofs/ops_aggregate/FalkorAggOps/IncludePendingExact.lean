import FalkorAggOps.StreamOps
/-
# IncludePending composed with `Pending` (#3453)

`StreamOps.includePending_spec` proves Phase 1 (`childRows`) and Phase 2
(`pendingRows`) of `IncludePendingOp` separately, each against whatever node
lists it is handed. Here the lists are the ones main hands it, for an
uncorrelated MERGE node scan (`argument_rows` empty, one row per node; the row
is the node id):

| here | there (b84473a7f) |
| --- | --- |
| `Gc` | the committed graph: live node ids, each node's committed labels |
| `Pd` | `Pending`'s `created_nodes`, `deleted_nodes`, `set_labels`, `remove_labels` (pending.rs) |
| `scan` | the wrapped `NodeByLabelScan` over the committed graph |
| `stage` | `stage_node_labels` (pending.rs:661-676), reached by `SetOp` `SetItem::Label` (ops/set.rs:403) without checking the committed matrix |
| `deletedNode` | `deleted_node` (pending.rs:772-777): inserts into `deleted_nodes`, `set_labels` untouched |
| `removedFor` | `nodes_with_pending_label_removes` (pending.rs:1060-1076) |
| `pendingWith` | `get_pending_nodes_with_labels` (pending.rs:1040-1055): `set_labels` entries carrying every label, no deleted filter, committed labels ignored |
| `includePendingMain` | `IncludePendingOp::next` (include_pending.rs:161-209) = `StreamOps.childRows` then `StreamOps.pendingRows` |

Property `IncludePendingExact`: the emitted node rows are duplicate-free and
are exactly the live nodes (committed or created, not deleted) whose effective
labels — `(committed \ remove_labels) ∪ set_labels`, `set_labels` alone for a
created node — contain the pattern. The three live repros refute it on main:
`bug3453_yields_deleted_node`, `bug3453_duplicate_row`,
`bug3453_misses_split_multilabel`. `includePendingFixed_exact` proves it for a
Phase 2 that filters by liveness and effective labels and skips what Phase 1
already emitted.
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

/-- `get_pending_nodes_with_labels` (pending.rs:1040-1055). -/
def pendingWith (p : Pd) (ls : List Nat) : List Nat :=
  if ls = [] then p.created else (p.setL.filter fun e => ls.all (· ∈ e.2)).map (·.1)

/-- `IncludePendingOp::next` with no argument rows: Phase 1 = the scan minus
deleted and label-removed nodes (`childRows`), Phase 2 = `pendingWith`
(`pendingRows`, one empty row per node). -/
def includePendingMain (g : Gc) (p : Pd) (ls : List Nat) : List Nat :=
  childRows some p.deleted (removedFor p ls) (scan g ls) ++
  pendingRows (fun _ n => n) 0 [] (pendingWith p ls)

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

/-- **#3453 (1)** `MATCH (n:A) SET n:B DELETE n MERGE (m:B) RETURN m`: Phase 2
yields the deleted node, so MERGE matches it (live: Rust returns node 0 and
creates nothing; C creates a new `:B`). -/
theorem bug3453_yields_deleted_node :
    includePendingMain g1 (deletedNode (stage p0 0 [B]) 0) [B] = [0] ∧
    ¬ IncludePendingExact g1 (deletedNode (stage p0 0 [B]) 0) [B] [0] := by
  refine ⟨by decide, fun ⟨_, h⟩ => ?_⟩
  have := ((h 0).1 (by simp)).1.2
  exact this (by decide)

/-- **#3453 (2)** `MATCH (n:A) SET n:A MERGE (m:A) …`: SET stages a label the
node already has, so Phase 1 and Phase 2 both yield node 0 (live: two rows, ON
MATCH runs twice; C one row). -/
theorem bug3453_duplicate_row :
    includePendingMain g1 (stage p0 0 [A]) [A] = [0, 0] ∧
    ¬ IncludePendingExact g1 (stage p0 0 [A]) [A] [0, 0] := by
  refine ⟨by decide, fun ⟨h, _⟩ => ?_⟩
  simp at h

/-- **#3453 (3)** `MATCH (n:A) SET n:B MERGE (m:A:B)`: node 0 has A committed and
B staged; Phase 1 sees only A, Phase 2 only B, so nothing is emitted and MERGE
creates a duplicate (live: count(n) = 2 after; C matches n). -/
theorem bug3453_misses_split_multilabel :
    includePendingMain g1 (stage p0 0 [B]) [A, B] = [] ∧
    ¬ IncludePendingExact g1 (stage p0 0 [B]) [A, B] [] := by
  refine ⟨by decide, fun ⟨_, h⟩ => ?_⟩
  have := (h 0).2 ⟨⟨.inl (by decide), by decide⟩, by decide⟩
  simp at this

/-! ## What a fix proves -/

/-- Phase 1's condition: a committed node the scan yields and the filter keeps. -/
def phase1 (g : Gc) (p : Pd) (ls : List Nat) (n : Nat) : Bool :=
  n ∈ g.nodes && ls.all (· ∈ g.labels n) && n ∉ p.deleted && n ∉ removedFor p ls

instance (g : Gc) (p : Pd) (n : Nat) : Decidable (live g p n) := by unfold live; infer_instance

/-- Keep the last copy of each id. -/
def dedupL : List Nat → List Nat
  | [] => []
  | x :: xs => if x ∈ xs then dedupL xs else x :: dedupL xs

theorem mem_dedupL : ∀ (l : List Nat) (a : Nat), a ∈ dedupL l ↔ a ∈ l
  | [], _ => by simp [dedupL]
  | x :: xs, a => by
    unfold dedupL; split
    · rename_i h; rw [mem_dedupL xs a]; constructor
      · exact List.mem_cons_of_mem _
      · intro h'; rcases List.mem_cons.mp h' with rfl | h'
        · exact h
        · exact h'
    · simp [mem_dedupL xs a]

theorem nodup_dedupL : ∀ (l : List Nat), (dedupL l).Nodup
  | [] => by simp [dedupL]
  | x :: xs => by
    unfold dedupL; split
    · exact nodup_dedupL xs
    · rename_i h; exact List.nodup_cons.2 ⟨by rw [mem_dedupL]; exact h, nodup_dedupL xs⟩

/-- A fixed Phase 2: candidates are created nodes and `set_labels` keys, kept when
live, matching on effective labels, and not already emitted by Phase 1. -/
def pendingFixed (g : Gc) (p : Pd) (ls : List Nat) : List Nat :=
  (dedupL (p.created ++ p.setL.map (·.1))).filter fun n =>
    decide (live g p n) && ls.all (· ∈ eff g p n) && !phase1 g p ls n

def includePendingFixed (g : Gc) (p : Pd) (ls : List Nat) : List Nat :=
  childRows some p.deleted (removedFor p ls) (scan g ls) ++ pendingRows (fun _ n => n) 0 [] (pendingFixed g p ls)

theorem contains_false (l : List Nat) (n : Nat) : l.contains n = false ↔ n ∉ l := by
  rw [← Bool.not_eq_true, List.contains_iff_mem]

theorem childRows_some (d r rows : List Nat) : childRows some d r rows = rows.filter fun n => !(d.contains n || r.contains n) := rfl

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

/-- **`includePendingFixed_exact`**: when committed and created ids are apart,
with `remove_labels` keyed uniquely (an `FxHashMap`), the fixed operator satisfies `IncludePendingExact` for every non-empty pattern. -/
theorem includePendingFixed_exact (g : Gc) (p : Pd) (ls : List Nat) (hls : ls ≠ [])
    (hgn : g.nodes.Nodup) (hapart : ∀ n ∈ p.created, n ∉ g.nodes) (hrk : (p.remL.map (·.1)).Nodup) :
    IncludePendingExact g p ls (includePendingFixed g p ls) := by
  have h1mem : ∀ n, n ∈ childRows some p.deleted (removedFor p ls) (scan g ls) ↔ phase1 g p ls n = true := by
    intro n
    simp only [childRows_some, scan, List.mem_filter, phase1, Bool.and_eq_true, decide_eq_true_eq,
      Bool.not_eq_true', Bool.or_eq_false_iff, contains_false]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3, h4⟩; exact ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩
    · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩; exact ⟨⟨h1, h2⟩, h3, h4⟩
  have p1live : ∀ n, phase1 g p ls n = true → live g p n ∧ ∀ l ∈ ls, l ∈ eff g p n := by
    intro n h
    simp only [phase1, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
    obtain ⟨⟨⟨hn, hl⟩, hd⟩, hr⟩ := h
    refine ⟨⟨.inl hn, hd⟩, fun l hl' => ?_⟩
    have hc : n ∉ p.created := fun hc => hapart n hc hn
    simp only [eff, hc, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq]
    refine .inl ⟨hl l hl', fun hrm => hr (mem_removedFor p ls n hls l hl' hrm)⟩
  rw [includePendingFixed, pendingRows_id]
  refine ⟨?_, fun n => ?_⟩
  · rw [List.nodup_append]
    refine ⟨?_, ?_, ?_⟩
    · rw [childRows_some]; exact (hgn.filter _).filter _
    · exact (nodup_dedupL _).filter _
    · intro a ha b hb hab; subst hab
      have := (h1mem a).1 ha
      simp only [pendingFixed, List.mem_filter, Bool.and_eq_true, Bool.not_eq_true'] at hb
      rw [this] at hb; exact absurd hb.2 (by simp)
  · rw [List.mem_append, h1mem]
    constructor
    · rintro (h | h)
      · exact p1live n h
      · simp only [pendingFixed, List.mem_filter, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
        exact ⟨h.2.1.1, h.2.1.2⟩
    · rintro ⟨hlv, hl⟩
      by_cases hp : phase1 g p ls n = true
      · exact .inl hp
      · right
        simp only [pendingFixed, List.mem_filter, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true,
          Bool.not_eq_true', mem_dedupL, List.mem_append]
        refine ⟨?_, ⟨hlv, hl⟩, by simpa using hp⟩
        -- a node not emitted by Phase 1 is created or carries a staged label
        by_cases hc : n ∈ p.created
        · exact .inl hc
        · right
          have hn : n ∈ g.nodes := hlv.1.resolve_right hc
          -- some pattern label is not committed-and-unremoved, so it is staged
          have : ∃ l ∈ ls, l ∈ look p.setL n := by
            apply Classical.byContradiction
            intro hno
            apply hp
            simp only [phase1, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
            have hall : ∀ l ∈ ls, l ∈ g.labels n ∧ l ∉ look p.remL n := by
              intro l hl'
              have := hl l hl'
              simp only [eff, hc, ite_false, List.mem_append, List.mem_filter, decide_eq_true_eq] at this
              rcases this with h | h
              · exact h
              · exact absurd ⟨l, hl', h⟩ hno
            refine ⟨⟨⟨hn, fun l hl' => (hall l hl').1⟩, hlv.2⟩, fun hrf => ?_⟩
            simp only [removedFor, hls, ite_false, List.mem_map, List.mem_filter, List.any_eq_true,
              decide_eq_true_eq] at hrf
            obtain ⟨e, ⟨he, l, hl', hle⟩, rfl⟩ := hrf
            -- `look` returns the first entry with this key; `e` may be a later one
            exact (hall l hl').2 (by rw [look_of_mem p.remL hrk e he]; exact hle)
          obtain ⟨l, _, hl'⟩ := this
          exact look_mem _ _ _ hl'

end FalkorAggOps.IncludePendingExact
