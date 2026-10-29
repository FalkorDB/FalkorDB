import FalkorNativeIndex.Writes
set_option linter.unusedSimpArgs false

/-!
# Indexer batches: how a `Building` column is filled (#2351)
# (`graph_writes.rs:528-632` `scan`/`scan_nodes`/`scan_edges`, `versioned.rs:161-217`,
# `falkordb_index.rs:197-213`)

A column is created `Building` (`create_building`) and filled by indexer batches. A batch
(`Graph::indexer_prepare`) reads the next `budget` entities from the column's cursor at the values
the version holds (`scan`), adds them (`add_encoded`) and moves the cursor past them, or flips the
column `Ready` when the scan reached the end. `Pending.lean` proves what a live write does to a
`Building` column (`writerRecord_eq_sequential`, `stateLog_*`); this file proves what a batch reads
and that the cursor it returns is exactly the line the writer's filter uses:

* `scanNodes_below_cursor` / `scanEdges_below_cursor`: with the matrix iterated in ascending order
  (GraphBLAS row order, a hypothesis about the C API), the entities a batch read are exactly those
  `precedes` the returned cursor, so an entity is in the index iff `place` puts it below the cursor.
* `edgeLoop_pair_complete` (inside the same theorem): a batch never stops inside a multi-edge pair.
* `scanNodes_tuples`, `scanEdges_tuples`: the tuples are the encoded pairs of the entities read.
* `prepareBatch_spec`: a batch changes nothing on a column that is not `Building`.
* `kind_*`: cursors never mix kinds, so `precedes`'s `unreachable!` never fires.
-/

namespace NI

/-- `usize::MAX`, the budget of `build_in_one_batch` (`graph_writes.rs:433`). -/
def usizeMax : Nat := 2 ^ 64 - 1

/-! ## Reading tuples -/

/-- The statement `if let Some(value) = <read attr> { column.push_encoded(&value, id, out) }` of
    `scan_nodes` (`graph_writes.rs:581-583`) and the `index` closure of `scan_edges` (`:603-607`). -/
def pushAttr (col : IndexColumn) (attrOf : Nat → Option Value) (out : EncodedTuples) (id : Nat) : EncodedTuples :=
  match attrOf id with
  | some v => col.pushEncoded v id out
  | none => out

/-- The tuples one entity contributes. -/
def tupleOf (col : IndexColumn) (attrOf : Nat → Option Value) (id : Nat) : List Nat :=
  match attrOf id with
  | some v => (match col.encodeValue v with | some k => [enc k id] | none => [])
  | none => []

theorem pushAttr_scalar (col : IndexColumn) (attrOf : Nat → Option Value) (out : EncodedTuples) (id : Nat) :
    (pushAttr col attrOf out id).scalar = out.scalar ++ tupleOf col attrOf id := by
  unfold pushAttr tupleOf
  cases attrOf id with
  | none => simp
  | some v =>
    simp only [IndexColumn.pushEncoded]
    cases h : col.encodeValue v <;> simp [h]

theorem foldPush_scalar (col : IndexColumn) (attrOf : Nat → Option Value) (ids : List Nat) :
    ∀ (out : EncodedTuples),
    (ids.foldl (pushAttr col attrOf) out).scalar = out.scalar ++ ids.flatMap (tupleOf col attrOf) := by
  induction ids with
  | nil => intro out; simp
  | cons i is ih => intro out; simp only [List.foldl_cons, ih, pushAttr_scalar, List.flatMap_cons, List.append_assoc]

/-! ## `scan_nodes` (`graph_writes.rs:553-587`) -/

/-- The node ids `lm.iter(from, u64::MAX)` yields: rows from `from` on, ascending. -/
def nodesFrom (lm : List Nat) (from_ : Nat) : List Nat := lm.filter (fun n => decide (from_ ≤ n))

/-- `scan_nodes`: up to `budget` `label` nodes from node `from` on, their tuples appended to `out`
    (unsorted), and the node to resume from (`nodes.peek()`), or `none` when no `label` node is left
    (also when the label or attribute does not exist). -/
def scanNodes (g : G) (label attr : String) (from_ budget : Nat) (col : IndexColumn) (out : EncodedTuples) :
    EncodedTuples × Option Nat :=
  match g.labelNodes label, g.attrId attr with
  | some lm, some idx =>
    let nodes := nodesFrom lm from_
    let out := (nodes.take budget).foldl (pushAttr col (fun n => g.nodeAttr n idx)) out
    (out, (nodes.drop budget).head?)
  | _, _ => (out, none)

theorem scanNodes_tuples (g : G) (label attr : String) (from_ budget : Nat) (col : IndexColumn) (out : EncodedTuples)
    (lm : List Nat) (idx : Nat) (hl : g.labelNodes label = some lm) (ha : g.attrId attr = some idx) :
    (scanNodes g label attr from_ budget col out).1.scalar =
      out.scalar ++ ((nodesFrom lm from_).take budget).flatMap (tupleOf col (fun n => g.nodeAttr n idx)) ∧
    (scanNodes g label attr from_ budget col out).2 = ((nodesFrom lm from_).drop budget).head? := by
  simp only [scanNodes, hl, ha]
  refine ⟨?_, ?_⟩
  · exact foldPush_scalar _ _ _ _
  · first | rfl | trivial

/-- A missing label or attribute: nothing read, at the end. -/
theorem scanNodes_missing (g : G) (label attr : String) (from_ budget : Nat) (col : IndexColumn) (out : EncodedTuples)
    (h : g.labelNodes label = none ∨ g.attrId attr = none) :
    scanNodes g label attr from_ budget col out = (out, none) := by
  unfold scanNodes
  rcases h with h | h <;> simp [h]

/-- **The nodes a batch read are exactly those below the cursor it returns.** With the label matrix
    iterated in strictly ascending order, a node at or after `from` was read iff it `precedes` the
    returned cursor; with no cursor returned, every node from `from` on was read. -/
theorem scanNodes_below_cursor (g : G) (label attr : String) (from_ budget : Nat) (col : IndexColumn)
    (out : EncodedTuples) (lm : List Nat) (idx : Nat) (hl : g.labelNodes label = some lm)
    (ha : g.attrId attr = some idx) (hasc : lm.Pairwise (· < ·)) (n : Nat) (hn : n ∈ lm) (hf : from_ ≤ n) :
    (∀ m, (scanNodes g label attr from_ budget col out).2 = some m →
        (n ∈ (nodesFrom lm from_).take budget ↔ (BuildCursor.node n).precedes (.node m) = true)) ∧
    ((scanNodes g label attr from_ budget col out).2 = none → n ∈ (nodesFrom lm from_).take budget) := by
  have hs : (nodesFrom lm from_).Pairwise (· < ·) := List.Pairwise.sublist List.filter_sublist hasc
  have hmem : n ∈ nodesFrom lm from_ := by simp [nodesFrom, hn, hf]
  have hsplit := List.take_append_drop budget (nodesFrom lm from_)
  have hpa := hs
  rw [← hsplit, List.pairwise_append] at hpa
  rw [(scanNodes_tuples g label attr from_ budget col out lm idx hl ha).2]
  constructor
  · intro m hm
    obtain ⟨rest, hrest⟩ : ∃ rest, (nodesFrom lm from_).drop budget = m :: rest := by
      cases h : (nodesFrom lm from_).drop budget with
      | nil => simp [h] at hm
      | cons a r => simp [h] at hm; exact ⟨r, by rw [hm]⟩
    have hdsorted : ((nodesFrom lm from_).drop budget).Pairwise (· < ·) := hpa.2.1
    rw [hrest] at hdsorted hpa
    simp only [BuildCursor.precedes, decide_eq_true_eq]
    rw [← hsplit] at hmem
    rcases List.mem_append.1 hmem with h1 | h1
    · exact ⟨fun _ => hpa.2.2 n h1 m (by simp), fun _ => h1⟩
    · rw [hrest] at h1
      have hmn : m ≤ n := by
        rcases List.mem_cons.1 h1 with h2 | h2
        · omega
        · have := List.rel_of_pairwise_cons hdsorted h2; omega
      constructor
      · intro htake
        have := hpa.2.2 n htake m (by simp)
        omega
      · intro hlt; omega
  · intro hm
    have : (nodesFrom lm from_).drop budget = [] := by
      cases h : (nodesFrom lm from_).drop budget with
      | nil => rfl
      | cons a r => simp [h] at hm
    rw [← hsplit] at hmem
    rcases List.mem_append.1 hmem with h1 | h1
    · exact h1
    · rw [this] at h1; simp at h1

/-! ## `scan_edges` (`graph_writes.rs:589-632`) -/

/-- The edge-reading loop of `scan_edges`: read edges one at a time while `read < budget`; when
    the budget is reached, also read the rest of the `(src, dst)` pair the last edge belongs to
    (`next_if` on the same pair), so no pair is split between batches. Returns the tuples and the
    edges left. -/
def edgeLoop (budget : Nat) (index : Nat → EncodedTuples → EncodedTuples) :
    Nat → List (Nat × Nat × Nat) → EncodedTuples → EncodedTuples × List (Nat × Nat × Nat)
  | _, [], out => (out, [])
  | read, (s, d, eid) :: es, out =>
    if read < budget then
      let out := index eid out
      if read + 1 = budget then
        let same := fun (e : Nat × Nat × Nat) => decide (e.1 = s ∧ e.2.1 = d)
        ((es.takeWhile same).foldl (fun o e => index e.2.2 o) out, es.dropWhile same)
      else edgeLoop budget index (read + 1) es out
    else (out, (s, d, eid) :: es)

/-- The `next_if` skip loop of `scan_edges` (`graph_writes.rs:609-613`): edges of `from_src` with a
    destination below `from_dst` are skipped, after starting at row `from_src`. -/
def skipP (fromSrc fromDst : Nat) (e : Nat × Nat × Nat) : Bool := decide (e.1 = fromSrc ∧ e.2.1 < fromDst)
def skipEdges (t : List (Nat × Nat × Nat)) (fromSrc fromDst : Nat) : List (Nat × Nat × Nat) :=
  (t.filter (fun e => decide (fromSrc ≤ e.1))).dropWhile (skipP fromSrc fromDst)

/-- `scan_edges`: up to `budget` `label`-typed edges from pair `(from_src, from_dst)` on, in
    `(src, dst)` order, plus the rest of the pair the batch stops in; edges of `from_src` with a
    destination below `from_dst` are skipped (an earlier batch indexed them). Returns the pair to
    resume from, or `none` when no edge is left. -/
def scanEdges (g : G) (label attr : String) (fromSrc fromDst budget : Nat) (col : IndexColumn)
    (out : EncodedTuples) : EncodedTuples × Option (Nat × Nat) :=
  match g.typeEdges label, g.attrId attr with
  | some t, some idx =>
    let index := fun (eid : Nat) (out : EncodedTuples) => pushAttr col (fun e => g.relAttr e idx) out eid
    let r := edgeLoop budget index 0 (skipEdges t fromSrc fromDst) out
    (r.1, r.2.head?.map (fun e => (e.1, e.2.1)))
  | _, _ => (out, none)

/-- `(src, dst)` order of two triples, strictly. -/
def pairLt (a b : Nat × Nat × Nat) : Prop := a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2.1 < b.2.1)
/-- ... and non-strictly. -/
def pairLe (a b : Nat × Nat × Nat) : Prop := a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2.1 ≤ b.2.1)

theorem pairLe_lt_trans {a b c : Nat × Nat × Nat} (h1 : pairLe a b) (h2 : pairLt b c) : pairLt a c := by
  unfold pairLe pairLt at *; omega

theorem pairLt_of_le_ne {a b : Nat × Nat × Nat} (h1 : pairLe a b) (h2 : ¬ (a.1 = b.1 ∧ a.2.1 = b.2.1)) :
    pairLt a b := by
  unfold pairLe pairLt at *; omega

theorem pairLe_lt_irrefl {a b : Nat × Nat × Nat} (h1 : pairLe a b) (h2 : pairLt b a) : False := by
  unfold pairLe pairLt at *; omega

theorem dropWhile_head {α} (p : α → Bool) : ∀ (l : List α) (h : α) (r : List α), l.dropWhile p = h :: r → p h = false
  | [], h, r, hh => by simp at hh
  | x :: xs, h, r, hh => by
    by_cases hp : p x = true
    · simp only [List.dropWhile_cons, hp, if_true] at hh; exact dropWhile_head p xs h r hh
    · simp only [List.dropWhile_cons, hp] at hh
      simp at hh
      obtain ⟨rfl, _⟩ := hh
      simpa using hp

/-- **What the edge loop reads.** It reads a prefix `c` of the edges (`c ++ left = es`), whose tuples
    are those of `c`; every edge read is strictly before the first one left (so a batch never stops
    inside a `(src, dst)` pair) when the edges are in `(src, dst)` order, and it reads something if
    there was budget and edges are left. -/
theorem edgeLoop_spec (budget : Nat) (index : Nat → EncodedTuples → EncodedTuples) (f : Nat → List Nat)
    (hidx : ∀ e out, (index e out).scalar = out.scalar ++ f e) :
    ∀ (es : List (Nat × Nat × Nat)), es.Pairwise pairLe → ∀ (read : Nat) (out : EncodedTuples),
    ∃ c, c ++ (edgeLoop budget index read es out).2 = es ∧
      (edgeLoop budget index read es out).1.scalar = out.scalar ++ c.flatMap (fun e => f e.2.2) ∧
      ((edgeLoop budget index read es out).2 ≠ [] → read < budget → c ≠ []) ∧
      (∀ h r', (edgeLoop budget index read es out).2 = h :: r' → ∀ e ∈ c, pairLt e h) := by
  intro es
  induction es with
  | nil => intro _ read out; exact ⟨[], by simp [edgeLoop]⟩
  | cons x es ih =>
    intro hs read out
    obtain ⟨s, d, eid⟩ := x
    have hs' : es.Pairwise pairLe := hs.of_cons
    have hle : ∀ y ∈ es, pairLe (s, d, eid) y := fun y hy => List.rel_of_pairwise_cons hs hy
    by_cases hr : read < budget
    · by_cases hb : read + 1 = budget
      · -- budget reached: finish the pair
        let same := fun (e : Nat × Nat × Nat) => decide (e.1 = s ∧ e.2.1 = d)
        have hloop : edgeLoop budget index read ((s, d, eid) :: es) out =
            ((es.takeWhile same).foldl (fun o e => index e.2.2 o) (index eid out), es.dropWhile same) := by
          simp only [edgeLoop, hr, hb, if_true]
          rfl
        refine ⟨(s, d, eid) :: es.takeWhile same, ?_⟩
        rw [hloop]
        refine ⟨?_, ?_, fun _ _ => by simp, ?_⟩
        · simp only [List.cons_append, List.cons.injEq, true_and]
          exact List.takeWhile_append_dropWhile
        · have key : ∀ (l : List (Nat × Nat × Nat)) (o : EncodedTuples),
              (l.foldl (fun o e => index e.2.2 o) o).scalar = o.scalar ++ l.flatMap (fun e => f e.2.2) := by
            intro l; induction l with
            | nil => intro o; simp
            | cons y ys ih2 => intro o; simp [ih2, hidx, List.append_assoc]
          rw [key, hidx]; simp [List.append_assoc]
        · intro h r' hh e he
          have hnot := dropWhile_head same es h r' hh
          have hsame : ¬ (h.1 = s ∧ h.2.1 = d) := by simpa [same] using hnot
          have hhmem : h ∈ es := by
            have hh' : es.dropWhile same = h :: r' := hh
            have := List.takeWhile_append_dropWhile (p := same) (l := es)
            rw [hh'] at this
            rw [← this]; simp
          have hxh : pairLt (s, d, eid) h := pairLt_of_le_ne (hle h hhmem) (by
            intro ⟨a, b⟩; exact hsame ⟨a.symm, b.symm⟩)
          rcases List.mem_cons.1 he with rfl | he
          · exact hxh
          · have := (List.all_eq_true.1 (List.all_takeWhile (p := same) (l := es))) e he
            have hp : e.1 = s ∧ e.2.1 = d := by simpa [same] using this
            simp only [pairLt] at hxh ⊢; omega
      · -- one edge read, loop on
        obtain ⟨c, h1, h2, h3, h4⟩ := ih hs' (read + 1) (index eid out)
        have hloop : edgeLoop budget index read ((s, d, eid) :: es) out =
            edgeLoop budget index (read + 1) es (index eid out) := by
          simp only [edgeLoop, hr, hb, if_true, if_false]
        refine ⟨(s, d, eid) :: c, ?_⟩
        rw [hloop]
        refine ⟨by simp [h1], ?_, fun _ _ => by simp, ?_⟩
        · rw [h2, hidx]; simp [List.append_assoc]
        · intro h r' hh e he
          have hcne : c ≠ [] := h3 (by rw [hh]; simp) (by omega)
          have hhmem : h ∈ es := by rw [← h1]; simp [hh]
          rcases List.mem_cons.1 he with rfl | he
          · obtain ⟨e', he'⟩ := List.exists_mem_of_ne_nil c hcne
            have he'es : e' ∈ es := by rw [← h1]; simp [he']
            exact pairLe_lt_trans (hle e' he'es) (h4 h r' hh e' he')
          · exact h4 h r' hh e he
    · have hloop : edgeLoop budget index read ((s, d, eid) :: es) out = (out, (s, d, eid) :: es) := by
        simp [edgeLoop, hr]
      refine ⟨[], ?_⟩
      rw [hloop]
      refine ⟨by simp, by simp, fun _ h => absurd h hr, by simp⟩

theorem scanEdges_missing (g : G) (label attr : String) (fromSrc fromDst budget : Nat) (col : IndexColumn)
    (out : EncodedTuples) (h : g.typeEdges label = none ∨ g.attrId attr = none) :
    scanEdges g label attr fromSrc fromDst budget col out = (out, none) := by
  unfold scanEdges
  rcases h with h | h <;> simp [h]

/-- **The edges a batch read are exactly those below the cursor it returns**, and their tuples are
    what it appended. With the tensor iterated in `(src, dst)` order, every edge at or after
    `(from_src, from_dst)` was read iff its pair `precedes` the returned pair (so a batch never ends
    inside a multi-edge pair); with no pair returned, every such edge was read. -/
theorem scanEdges_below_cursor (g : G) (label attr : String) (fromSrc fromDst budget : Nat) (col : IndexColumn)
    (out : EncodedTuples) (t : List (Nat × Nat × Nat)) (idx : Nat) (hl : g.typeEdges label = some t)
    (ha : g.attrId attr = some idx) (hasc : t.Pairwise pairLe) :
    ∃ c : List (Nat × Nat × Nat), (∀ e ∈ c, e ∈ t) ∧
      (scanEdges g label attr fromSrc fromDst budget col out).1.scalar =
        out.scalar ++ c.flatMap (fun e => tupleOf col (fun x => g.relAttr x idx) e.2.2) ∧
      (∀ m, (scanEdges g label attr fromSrc fromDst budget col out).2 = some m →
        ∀ e ∈ t, ¬ pairLt e (fromSrc, fromDst, 0) →
          (e ∈ c ↔ (BuildCursor.edge e.1 e.2.1).precedes (.edge m.1 m.2) = true)) ∧
      ((scanEdges g label attr fromSrc fromDst budget col out).2 = none →
        ∀ e ∈ t, ¬ pairLt e (fromSrc, fromDst, 0) → e ∈ c) := by
  simp only [scanEdges, hl, ha]
  have hsub : (skipEdges t fromSrc fromDst).Sublist t :=
    (List.dropWhile_sublist _).trans List.filter_sublist
  have hes : (skipEdges t fromSrc fromDst).Pairwise pairLe := hasc.sublist hsub
  have hf : ∀ x out, (pushAttr col (fun e => g.relAttr e idx) out x).scalar =
      out.scalar ++ tupleOf col (fun e => g.relAttr e idx) x := fun x out => pushAttr_scalar _ _ _ _
  obtain ⟨c, h1, h2, h3, h4⟩ := edgeLoop_spec budget
    (fun (eid : Nat) (out : EncodedTuples) => pushAttr col (fun e => g.relAttr e idx) out eid)
    (tupleOf col (fun e => g.relAttr e idx)) hf (skipEdges t fromSrc fromDst) hes 0 out
  -- every edge at or after the start is among `edges`
  have hin : ∀ e ∈ t, ¬ pairLt e (fromSrc, fromDst, 0) → e ∈ skipEdges t fromSrc fromDst := by
    intro e he hnl
    have h1' : e ∈ t.filter (fun e => decide (fromSrc ≤ e.1)) := by
      simp only [List.mem_filter, he, true_and, decide_eq_true_eq]
      simp only [pairLt] at hnl; omega
    have hnp : skipP fromSrc fromDst e = false := by
      simp only [skipP, decide_eq_false_iff_not]; simp only [pairLt] at hnl; omega
    have hsplit := List.takeWhile_append_dropWhile (p := skipP fromSrc fromDst)
      (l := t.filter (fun e => decide (fromSrc ≤ e.1)))
    rw [← hsplit] at h1'
    rcases List.mem_append.1 h1' with h | h
    · have := (List.all_eq_true.1 (List.all_takeWhile (p := skipP fromSrc fromDst)
        (l := t.filter (fun e => decide (fromSrc ≤ e.1))))) e h
      simp [hnp] at this
    · exact h
  refine ⟨c, fun e he => hsub.subset (by rw [← h1]; exact List.mem_append_left _ he), ?_, ?_, ?_⟩
  · simpa using h2
  · intro m hm e he hnl
    have hein := hin e he hnl
    rw [← h1] at hein
    obtain ⟨h, r', hh⟩ : ∃ h r', (edgeLoop budget (fun (eid : Nat) (out : EncodedTuples) =>
        pushAttr col (fun e => g.relAttr e idx) out eid) 0 (skipEdges t fromSrc fromDst) out).2 = h :: r' := by
      cases hr : (edgeLoop budget (fun (eid : Nat) (out : EncodedTuples) =>
          pushAttr col (fun e => g.relAttr e idx) out eid) 0 (skipEdges t fromSrc fromDst) out).2 with
      | nil => simp [hr] at hm
      | cons a r => exact ⟨a, r, rfl⟩
    have hmh : m = (h.1, h.2.1) := by simp [hh] at hm; exact hm.symm
    have hsortedr : (h :: r').Pairwise pairLe := by
      have : (c ++ (h :: r')).Pairwise pairLe := by rw [← hh, h1]; exact hes
      exact (List.pairwise_append.1 this).2.1
    simp only [BuildCursor.precedes, decide_eq_true_eq, hmh]
    rcases List.mem_append.1 hein with hc | hr
    · have := h4 h r' hh e hc
      unfold pairLt at this
      simp only [hc, true_iff]; exact this
    · rw [hh] at hr
      have hhe : pairLe h e := by
        rcases List.mem_cons.1 hr with rfl | hr
        · unfold pairLe; omega
        · exact List.rel_of_pairwise_cons hsortedr hr
      refine ⟨fun hc => ?_, fun hlt => ?_⟩
      · exact absurd (h4 h r' hh e hc) (fun hlt => pairLe_lt_irrefl hhe hlt)
      · exfalso; unfold pairLe at hhe; omega
  · intro hn e he hnl
    have hein := hin e he hnl
    rw [← h1] at hein
    have hr : (edgeLoop budget (fun (eid : Nat) (out : EncodedTuples) =>
        pushAttr col (fun e => g.relAttr e idx) out eid) 0 (skipEdges t fromSrc fromDst) out).2 = [] := by
      cases hr : (edgeLoop budget (fun (eid : Nat) (out : EncodedTuples) =>
          pushAttr col (fun e => g.relAttr e idx) out eid) 0 (skipEdges t fromSrc fromDst) out).2 with
      | nil => rfl
      | cons a r => simp [hr] at hn
    rw [hr] at hein
    simpa using hein

/-! ## `scan` and the batch (`graph_writes.rs:528-551`, `versioned.rs:187-217`) -/

/-- `scan` (`graph_writes.rs:528-551`): the sorted, deduplicated tuples of one batch and the cursor
    where the next one resumes (`none`: the scan reached the end). -/
def scan (g : G) (label attr : String) (from_ : BuildCursor) (entities : Nat) (col : IndexColumn) :
    EncodedTuples × Option BuildCursor :=
  let r : EncodedTuples × Option BuildCursor :=
    match from_ with
    | .node n =>
      let r := scanNodes g label attr n entities col EncodedTuples.empty
      (r.1, r.2.map BuildCursor.node)
    | .edge s d =>
      let r := scanEdges g label attr s d entities col EncodedTuples.empty
      (r.1, r.2.map (fun p => BuildCursor.edge p.1 p.2))
  (r.1.sortDedup, r.2)

/-- A batch is sorted and unique (the shape `from_sorted` needs) and holds the tuples read. -/
theorem scan_sorted (g : G) (label attr : String) (from_ : BuildCursor) (entities : Nat) (col : IndexColumn) :
    Sorted (scan g label attr from_ entities col).1.scalar ∧
    ∀ e, e ∈ (scan g label attr from_ entities col).1.scalar ↔
      e ∈ (match from_ with
        | .node n => (scanNodes g label attr n entities col EncodedTuples.empty).1.scalar
        | .edge s d => (scanEdges g label attr s d entities col EncodedTuples.empty).1.scalar) := by
  unfold scan
  cases from_ <;> simp only [EncodedTuples.sortDedup] <;> exact ⟨(sortDedup_spec _).1, (sortDedup_spec _).2⟩

/-- A cursor is always resumed in its own kind (so `precedes`'s `unreachable!` never fires). -/
theorem scan_kind (g : G) (label attr : String) (from_ : BuildCursor) (entities : Nat) (col : IndexColumn)
    (e : EntityType) (hk : BuildCursor.OfKind e from_) (m : BuildCursor)
    (hm : (scan g label attr from_ entities col).2 = some m) : BuildCursor.OfKind e m := by
  unfold scan at hm
  cases from_ with
  | node n =>
    cases e with
    | relationship => exact absurd hk (by simp [BuildCursor.OfKind])
    | node =>
      simp only [Option.map_eq_some_iff] at hm
      obtain ⟨_, _, rfl⟩ := hm; trivial
  | edge s d =>
    cases e with
    | node => exact absurd hk (by simp [BuildCursor.OfKind])
    | relationship =>
      simp only [Option.map_eq_some_iff] at hm
      obtain ⟨_, _, rfl⟩ := hm; trivial

theorem kind_new (e : EntityType) : BuildCursor.OfKind e (BuildCursor.new e) := by cases e <;> trivial

/-- The place the graph gives an entity (`Graph::apply_index_writes`, `graph.rs:3798-3807`): a node by
    its id, an edge by its endpoints, which an edge deleted in this unit no longer has. -/
def G.place (g : G) : EntityType → Nat → Option BuildCursor
  | .node, id => some (.node id)
  | .relationship, id => (g.endpoints id).map (fun p => .edge p.1 p.2)

theorem kind_place (g : G) (e : EntityType) (id : Nat) (c : BuildCursor) (h : g.place e id = some c) :
    BuildCursor.OfKind e c := by
  cases e with
  | node => simp [G.place] at h; subst h; trivial
  | relationship =>
    simp only [G.place, Option.map_eq_some_iff] at h
    obtain ⟨_, _, rfl⟩ := h; trivial

/-- `IndexColumn::indexer_read` (`falkordb_index.rs:197-205`). -/
def IndexColumn.indexerRead (c : IndexColumn) (g : G) (name : EntityType × IndexKey) (from_ : BuildCursor)
    (budget : Nat) : EncodedTuples × Option BuildCursor :=
  scan g name.2.1 name.2.2 from_ budget c

theorem indexerRead_spec (c : IndexColumn) (g : G) (e : EntityType) (k : IndexKey) (from_ : BuildCursor)
    (budget : Nat) : c.indexerRead g (e, k) from_ budget = scan g k.1 k.2 from_ budget c := rfl

/-- `IndexColumn::indexer_add` (`falkordb_index.rs:207-212`): `add_encoded`. -/
def IndexColumn.indexerAdd (c : IndexColumn) (batch : EncodedTuples) : IndexColumn × EncodedTuples :=
  c.addEncoded batch

theorem indexerAdd_spec (c : IndexColumn) (batch : EncodedTuples) : c.indexerAdd batch = c.addEncoded batch := rfl

theorem indexerAdd_mem (ix : NumericIndex) (batch : EncodedTuples) (e : Nat) :
    e ∈ ((IndexColumn.numeric ix).indexerAdd batch).1.ix.tree ↔ e ∈ ix.tree ∨ e ∈ batch.scalar :=
  addEncoded_spec ix batch e

/-- `IndexId` (`versioned.rs:47-50`). -/
inductive IndexId where
  | column (e : EntityType) (k : IndexKey)

/-- `Graph::prepare_batch::<IndexColumn>` (`versioned.rs:205-217`): read the batch from the column's
    cursor, then apply it to the same version; `false` (nothing changed) if the column is gone or
    not `Building`. -/
def G.prepareBatch (g : G) (e : EntityType) (k : IndexKey) (budget : Nat) : G × Bool :=
  match g.idx.entry e k.1 k.2 with
  | none => (g, false)
  | some en =>
    match en.cursor with
    | none => (g, false)
    | some from_ =>
      let r := en.index.indexerRead g (e, k) from_ budget
      let a := g.idx.applyIndexerBatch e k r.1 r.2
      ({ g with idx := a.1 }, a.2)

/-- `Graph::indexer_prepare` (`versioned.rs:191-201`). -/
def G.indexerPrepare (g : G) (id : IndexId) (budget : Nat) : G × Bool :=
  match id with
  | .column e k => g.prepareBatch e k budget

end NI
