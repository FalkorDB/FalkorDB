import OpsTraverse.ExpandIntoPending
import OpsTraverse.UnionGlue
/-
# CondTraverse over uncommitted deletes (#3577)

`CondTraverse.processPairs` / `expandRow` / `UnionGlue.expandUnionRow` model the operator over
one committed graph. #3577 makes `CondTraverseOp` read `Pending` too, as #3463 did for
ExpandInto (`ExpandIntoPending`): a pending-deleted node matches nothing, and a
pending-deleted edge is skipped.

| here | there |
| --- | --- |
| `pairRowsP`, `processPairsP` | `CondTraverseOp::process_pairs` (`cond_traverse.rs:1271-1367`): a pair with a pending-deleted endpoint is skipped (`:1293-1296`); the collapse (`:1330-1345`) and emit (`:1352-1365`) scans skip `is_relationship_deleted` edges before the sibling check |
| `expandRowPP`, `expandRowPend` | `expand_row` (`:931-1146`) passing `pending` (`:981`) to both `process_pairs` blocks (`:1060`, `:1105`) and to `expand_union_row` (`:1017`) |
| `expandUnionRowP` | `expand_union_row` (`:1154-1267`), the same two blocks over the union stream (`:1210`, `:1255`) |
| `batchedP` | `expand_batch` (`:605-924`): any pending-deleted relationship ⇒ fall back to the per-row path (`:717-722`); a pending-deleted source is never seeded (`:746-748`), a pending-deleted destination never emitted (`:816-818`) |

Every model here with empty `Pending` is literally the committed-graph model
(`processPairsP_empty`, `expandRowPend_empty`, `expandUnionRowP_empty`, `batchedP_empty`), so
every earlier CondTraverse theorem is its no-pending instance. With pending deletes, the
emit-mode match set is exactly the committed one restricted to live edges with live endpoints
(`emitP_directed_iff`, `emitP_bidir_iff`, `union_emitP_directed_iff`), collapse keeps one row
per pair (`collapseP_directed_keys_nodup`, `union_collapseP_directed_keys_nodup`), and the
batched path equals the per-row collapse (`batchedP_eq_collapse`). Property
`EIP.LiveCondTraverse` now holds (`expandRowPend_live`, `expandUnionRowP_live`): the #3452
CondTraverse half (`EIP.pre3577_condTraverse_violates`) is fixed by #3577.

The fused chain (`chain` with hops) checks only its first source and last destination; it
is not reached with pending deletes because `fuse_anonymous_traverse` no longer fuses a plan
that contains a `Delete` (`fuse_anonymous_traverse.rs:195-206`).
-/

namespace OpsTraverse.CTP

open OpsTraverse OpsTraverse.EIP OpsTraverse.UnionGlue

/-- Neither endpoint of the matrix pair is pending-deleted (`:1293-1296`). -/
def liveEnds (p : Pending) (q : Nat × Nat) : Bool :=
  !(p.delNodes.contains q.1 || p.delNodes.contains q.2)

/-- The edges of a pair `process_pairs` may use: not pending-deleted, not used by a sibling. -/
def pesP (g : Graph) (p : Pending) (r : RowIn) (q : Nat × Nat) : List Nat :=
  (edgesBetween g q.1 q.2).filter fun e => !(p.delRels.contains e) && !(r.used.contains e)

/-- One matrix pair of `process_pairs` (#3577). -/
def pairRowsP (g : Graph) (p : Pending) (isRev : Bool) (r : RowIn) (emit : Bool)
    (q : Nat × Nat) : List (Nat × Nat × Nat) :=
  if p.delNodes.contains q.1 || p.delNodes.contains q.2 then []
  else if pcond r isRev q then
    (if emit then pesP g p r q else (pesP g p r q).take 1).map fun e => (pf isRev q, pt isRev q, e)
  else []

def processPairsP (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (emit : Bool) : List (Nat × Nat × Nat) :=
  ps.flatMap (pairRowsP g p isRev r emit)

/-- `expand_row`'s forward and (undirected) reverse blocks over a pair stream, with `Pending`. -/
def expandRowPP (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  processPairsP g p ps false r emit ++
    (if bidir then processPairsP g p (ps.filter fun q => q.1 != q.2) true r emit else [])

/-- `CondTraverseOp::expand_row` (single type) on #3577. -/
def expandRowPend (g : Graph) (p : Pending) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  expandRowPP g p (pairs g) bidir emit r

/-- `expand_union_row` on #3577. -/
noncomputable def expandUnionRowP (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (p : Pending) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  expandRowPP gs.flatten p (unionStream S gs sk) bidir emit r

/-! ## Nothing pending: the committed-graph models -/

theorem pairRowsP_empty (g : Graph) (isRev : Bool) (r : RowIn) (emit : Bool) (q : Nat × Nat) :
    pairRowsP g ⟨[], []⟩ isRev r emit q = pairRows g isRev r emit q := by
  simp [pairRowsP, pairRows, pesP, pes]

theorem processPairsP_empty (g : Graph) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (emit : Bool) : processPairsP g ⟨[], []⟩ ps isRev r emit = processPairs g ps isRev r emit := by
  have : pairRowsP g ⟨[], []⟩ isRev r emit = pairRows g isRev r emit :=
    funext (pairRowsP_empty g isRev r emit)
  simp only [processPairsP, processPairs, this]

theorem expandRowPP_empty (g : Graph) (ps : List (Nat × Nat)) (bidir emit : Bool) (r : RowIn) :
    expandRowPP g ⟨[], []⟩ ps bidir emit r = expandRowP g ps bidir emit r := by
  simp only [expandRowPP, expandRowP, processPairsP_empty]

theorem expandRowPend_empty (g : Graph) (bidir emit : Bool) (r : RowIn) :
    expandRowPend g ⟨[], []⟩ bidir emit r = expandRow g bidir emit r := by
  rw [expandRowPend, expandRowPP_empty, expandRow_eq_expandRowP]

theorem expandUnionRowP_empty (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (bidir emit : Bool) (r : RowIn) :
    expandUnionRowP S gs sk ⟨[], []⟩ bidir emit r = expandUnionRow S gs sk bidir emit r := by
  rw [expandUnionRowP, expandRowPP_empty, expandUnionRow]

/-! ## Reduction: pending deletes = live pairs + deleted edges treated as used -/

/-- The row with every pending-deleted edge added to the sibling-used set. -/
def hide (p : Pending) (r : RowIn) : RowIn := { r with used := r.used ++ p.delRels }

theorem pesP_eq (g : Graph) (p : Pending) (r : RowIn) (q : Nat × Nat) :
    pesP g p r q = pes g (hide p r) q := by
  simp only [pesP, pes, hide, List.contains_append]
  congr 1
  funext e
  cases p.delRels.contains e <;> cases r.used.contains e <;> rfl

theorem pairRowsP_eq (g : Graph) (p : Pending) (isRev : Bool) (r : RowIn) (emit : Bool)
    (q : Nat × Nat) :
    pairRowsP g p isRev r emit q =
      if liveEnds p q then pairRows g isRev (hide p r) emit q else [] := by
  unfold pairRowsP pairRows liveEnds
  have hc : pcond (hide p r) isRev q = pcond r isRev q := rfl
  rw [hc, pesP_eq]
  cases p.delNodes.contains q.1 || p.delNodes.contains q.2 <;> simp

theorem processPairsP_eq (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (isRev : Bool)
    (r : RowIn) (emit : Bool) :
    processPairsP g p ps isRev r emit = processPairs g (ps.filter (liveEnds p)) isRev (hide p r) emit := by
  induction ps with
  | nil => rfl
  | cons q ps ih =>
    simp only [processPairsP, processPairs, List.flatMap_cons, List.filter_cons] at ih ⊢
    rw [pairRowsP_eq, ih]
    cases liveEnds p q <;> simp

theorem filter_comm {α : Type} (a b : α → Bool) (l : List α) :
    (l.filter a).filter b = (l.filter b).filter a := by
  simp only [List.filter_filter]
  congr 1
  funext x
  exact Bool.and_comm _ _

theorem expandRowPP_eq (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (bidir emit : Bool)
    (r : RowIn) :
    expandRowPP g p ps bidir emit r = expandRowP g (ps.filter (liveEnds p)) bidir emit (hide p r) := by
  simp only [expandRowPP, expandRowP, processPairsP_eq]
  rw [filter_comm]

/-! ## Collapse mode over an arbitrary pair stream -/

theorem collapse_iff_gen {g : Graph} {ps : List (Nat × Nat)} {r : RowIn} {f t : Nat}
    (Q : Nat → Nat → Prop)
    (hps : ∀ s d, (s, d) ∈ ps ↔ (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ Q s d) :
    (∃ e, (f, t, e) ∈ processPairs g ps false r false) ↔
      ∃ x ∈ g, x.src = f ∧ x.dst = t ∧ x.id ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t ∧ Q f t := by
  constructor
  · rintro ⟨e, he⟩
    obtain ⟨x, hx, hid, hu, hf, ht, hq, h1, h2⟩ := (mem_processPairs Q hps).1 (collapse_sub_emit he)
    subst h1 h2 hid
    exact ⟨x, hx, rfl, rfl, by simpa using hu, hf, ht, hq⟩
  · rintro ⟨x, hx, rfl, rfl, hu, hf, ht, hq⟩
    simp only [processPairs, List.mem_flatMap, mem_pairRows]
    have hne : x.id ∈ pes g r (x.src, x.dst) := mem_pes.2 ⟨⟨x, hx, rfl, rfl, rfl⟩, hu⟩
    have hc : pcond r false (x.src, x.dst) = true := by simp [pcond, pf, pt, hf, ht]
    revert hne
    cases hes : pes g r (x.src, x.dst) with
    | nil => intro hne; simp at hne
    | cons e0 rest =>
      intro _
      refine ⟨e0, (x.src, x.dst), (hps _ _).2 ⟨⟨x, hx, rfl, rfl⟩, hq⟩, hc, e0, ?_, rfl⟩
      simp [hes]

/-! ## Correctness with pending deletes -/

theorem liveEnds_iff (p : Pending) (s d : Nat) :
    liveEnds p (s, d) = true ↔ s ∉ p.delNodes ∧ d ∉ p.delNodes := by
  simp [liveEnds]

theorem livePairs_spec (g : Graph) (p : Pending) (s d : Nat) :
    (s, d) ∈ (pairs g).filter (liveEnds p) ↔
      (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ (s ∉ p.delNodes ∧ d ∉ p.delNodes) := by
  simp only [List.mem_filter, mem_pairs, liveEnds_iff]

theorem livePairs_filter_spec (g : Graph) (p : Pending) (s d : Nat) :
    (s, d) ∈ ((pairs g).filter (liveEnds p)).filter (fun q => q.1 != q.2) ↔
      (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ ((s ∉ p.delNodes ∧ d ∉ p.delNodes) ∧ s ≠ d) := by
  simp only [List.mem_filter, mem_pairs, liveEnds_iff, bne_iff_ne, ne_eq]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, h2, h3⟩
  · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h1, h2⟩, h3⟩

/-- **Directed, emit mode, with pending deletes**: `(f,t,e)` is produced iff `e` is an edge
`f→t` satisfying the row's bindings, unused by a sibling, not pending-deleted, with neither
endpoint pending-deleted (`emit_directed_iff` on the effective graph). -/
theorem emitP_directed_iff {g : Graph} {p : Pending} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowPend g p false true r ↔
      ∃ x ∈ g, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t ∧
        e ∉ p.delRels ∧ f ∉ p.delNodes ∧ t ∉ p.delNodes := by
  rw [expandRowPend, expandRowPP_eq]
  simp only [expandRowP, Bool.false_eq_true, ite_false, List.append_nil]
  rw [mem_processPairs _ (livePairs_spec g p)]
  simp only [Bool.false_eq_true, ite_false, hide, List.mem_append, not_or]
  constructor
  · rintro ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, h5, h6⟩
    subst h5 h6
    exact ⟨x, hx, h1, rfl, rfl, hu, h3, h4, hd, hn1, hn2⟩
  · rintro ⟨x, hx, h1, rfl, rfl, hu, h3, h4, hd, hn1, hn2⟩
    exact ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, rfl, rfl⟩

/-- **Undirected, emit mode, with pending deletes**: `emit_bidir_iff` on the effective graph. -/
theorem emitP_bidir_iff {g : Graph} {p : Pending} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowPend g p true true r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        e ∉ p.delRels ∧ f ∉ p.delNodes ∧ t ∉ p.delNodes ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  rw [expandRowPend, expandRowPP_eq]
  simp only [expandRowP, ite_true, List.mem_append]
  rw [mem_processPairs _ (livePairs_spec g p), mem_processPairs _ (livePairs_filter_spec g p)]
  simp only [Bool.false_eq_true, ite_false, ite_true, hide, List.mem_append, not_or]
  constructor
  · rintro (⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, h5, h6⟩ |
      ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨⟨hn1, hn2⟩, hne⟩, h5, h6⟩)
    · subst h5 h6
      exact ⟨x, hx, h1, hu, h3, h4, hd, hn1, hn2, Or.inl ⟨rfl, rfl⟩⟩
    · subst h5 h6
      exact ⟨x, hx, h1, hu, h3, h4, hd, hn2, hn1, Or.inr ⟨hne, rfl, rfl⟩⟩
  · rintro ⟨x, hx, h1, hu, h3, h4, hd, hn1, hn2, (⟨rfl, rfl⟩ | ⟨hne, rfl, rfl⟩)⟩
    · exact Or.inl ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, rfl, rfl⟩
    · exact Or.inr ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨⟨hn2, hn1⟩, hne⟩, rfl, rfl⟩

/-- **Directed collapse with pending deletes**: a row for `(f,t)` iff some live, unused edge
`f→t` joins two live nodes. -/
theorem collapseP_directed_iff {g : Graph} {p : Pending} {r : RowIn} {f t : Nat} :
    (∃ e, (f, t, e) ∈ expandRowPend g p false false r) ↔
      ∃ x ∈ g, x.src = f ∧ x.dst = t ∧ x.id ∉ r.used ∧ x.id ∉ p.delRels ∧
        okBound r.fromId f ∧ okBound r.toId t ∧ f ∉ p.delNodes ∧ t ∉ p.delNodes := by
  rw [expandRowPend]
  simp only [expandRowPP_eq, expandRowP, Bool.false_eq_true, ite_false, List.append_nil]
  rw [collapse_iff_gen _ (livePairs_spec g p)]
  simp only [hide, List.mem_append, not_or]
  constructor
  · rintro ⟨x, hx, h1, h2, ⟨hu, hd⟩, h3, h4, hn1, hn2⟩
    exact ⟨x, hx, h1, h2, hu, hd, h3, h4, hn1, hn2⟩
  · rintro ⟨x, hx, h1, h2, hu, hd, h3, h4, hn1, hn2⟩
    exact ⟨x, hx, h1, h2, ⟨hu, hd⟩, h3, h4, hn1, hn2⟩

/-- Directed collapse still emits each matrix pair at most once. -/
theorem collapseP_directed_keys_nodup (g : Graph) (p : Pending) (r : RowIn) :
    ((expandRowPend g p false false r).map fun x => (x.1, x.2.1)).Nodup := by
  simp only [expandRowPend, expandRowPP_eq, expandRowP, Bool.false_eq_true, ite_false,
    List.append_nil]
  exact ((nodup_pairs g).sublist List.filter_sublist).sublist (collapse_keys_sublist g _ _)

/-- Every row of `processPairsP` is carried by a live edge of `g` joining the row's endpoints. -/
theorem mem_processPairsP {g : Graph} {p : Pending} {ps : List (Nat × Nat)} {isRev emit : Bool}
    {r : RowIn} {y : Nat × Nat × Nat} (h : y ∈ processPairsP g p ps isRev r emit) :
    ∃ x ∈ g, x.id = y.2.2 ∧ y.2.2 ∉ p.delRels ∧ x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes := by
  rw [processPairsP_eq] at h
  have hT : y ∈ processPairs g (ps.filter (liveEnds p)) isRev (hide p r) true := by
    cases emit
    · exact collapse_sub_emit h
    · exact h
  simp only [processPairs, List.mem_flatMap, mem_pairRows, ite_true] at hT
  obtain ⟨q, hq, -, e, he, rfl⟩ := hT
  obtain ⟨⟨x, hx, hid, hs, hd⟩, hu⟩ := mem_pes.1 he
  have hl := (List.mem_filter.1 hq).2
  have hl' : liveEnds p (q.1, q.2) = true := hl
  rw [liveEnds_iff] at hl'
  simp only [hide, List.mem_append, not_or] at hu
  exact ⟨x, hx, hid, hu.2, hs ▸ hl'.1, hd ▸ hl'.2⟩

theorem expandRowPP_live {g : Graph} (hwf : g.WF) (p : Pending) (ps : List (Nat × Nat))
    (b em : Bool) (r : RowIn) : ∀ y ∈ expandRowPP g p ps b em r, Live g p y.2.2 := by
  intro y hy
  have hm : ∃ x ∈ g, x.id = y.2.2 ∧ y.2.2 ∉ p.delRels ∧ x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes := by
    simp only [expandRowPP, List.mem_append] at hy
    rcases hy with hy | hy
    · exact mem_processPairsP hy
    · split at hy
      · exact mem_processPairsP hy
      · simp at hy
  obtain ⟨x, hx, hid, hd, hs, ht⟩ := hm
  refine ⟨hd, fun z hz hzid => ?_⟩
  have : z = x := ASP.eq_of_mem_of_id hwf hz hx (hzid.trans hid.symm)
  subst this
  exact ⟨hs, ht⟩

/-- **#3577 fixes the CondTraverse half of #3452**: every edge `expand_row` matches is in the
effective graph (not pending-deleted, both endpoints live). -/
theorem expandRowPend_live : LiveCondTraverse expandRowPend := by
  intro g p b em r hwf y hy
  exact expandRowPP_live hwf p (pairs g) b em r y hy

/-- The same for `expand_union_row` over the types' edges. -/
theorem expandUnionRowP_live (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (hwf : Graph.WF gs.flatten) (p : Pending) (b em : Bool) (r : RowIn) :
    ∀ y ∈ expandUnionRowP S gs sk p b em r, Live gs.flatten p y.2.2 :=
  expandRowPP_live hwf p _ b em r

/-- **Multi-type, directed, emit mode, with pending deletes** (`union_emit_directed_iff` on the
effective union graph). -/
theorem union_emitP_directed_iff {S : Graph → List (Nat × Nat)} (hS : StructIter S)
    (gs : List Graph) {p : Pending} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandUnionRowP S gs (fun _ => true) p false true r ↔
      ∃ x ∈ gs.flatten, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t ∧
        e ∉ p.delRels ∧ f ∉ p.delNodes ∧ t ∉ p.delNodes := by
  have hs := (unionStream_spec hS gs (fun _ => true)).2.2
  have hps : ∀ s d, (s, d) ∈ (unionStream S gs (fun _ => true)).filter (liveEnds p) ↔
      (∃ x ∈ gs.flatten, x.src = s ∧ x.dst = d) ∧ (s ∉ p.delNodes ∧ d ∉ p.delNodes) := by
    intro s d; simp only [List.mem_filter, hs, liveEnds_iff, and_true]
  rw [expandUnionRowP, expandRowPP_eq]
  simp only [expandRowP, Bool.false_eq_true, ite_false, List.append_nil]
  rw [mem_processPairs _ hps]
  simp only [Bool.false_eq_true, ite_false, hide, List.mem_append, not_or]
  constructor
  · rintro ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, h5, h6⟩
    subst h5 h6
    exact ⟨x, hx, h1, rfl, rfl, hu, h3, h4, hd, hn1, hn2⟩
  · rintro ⟨x, hx, h1, rfl, rfl, hu, h3, h4, hd, hn1, hn2⟩
    exact ⟨x, hx, h1, ⟨hu, hd⟩, h3, h4, ⟨hn1, hn2⟩, rfl, rfl⟩

/-- Multi-type directed collapse with pending deletes: still one row per pair. -/
theorem union_collapseP_directed_keys_nodup {S : Graph → List (Nat × Nat)} (hS : StructIter S)
    (gs : List Graph) (sk : Nat × Nat → Bool) (p : Pending) (r : RowIn) :
    ((expandUnionRowP S gs sk p false false r).map fun x => (x.1, x.2.1)).Nodup := by
  simp only [expandUnionRowP, expandRowPP_eq, expandRowP, Bool.false_eq_true, ite_false,
    List.append_nil]
  exact ((unionStream_spec hS gs sk).2.1.sublist List.filter_sublist).sublist
    (collapse_keys_sublist _ _ _)

/-! ## `expand_batch` with pending deletes -/

/-- `expand_batch`: `none` = fall back to the per-row path (any pending-deleted relationship,
`:715-722`); otherwise sources and destinations that are pending-deleted are dropped
(`:746-748`, `:816-818`) around the `F·A₀·…` product. -/
def batchedP (mxm : Mat → Mat → Mat) (p : Pending) (F : Mat) (As : List Mat) : Option Mat :=
  if p.delRels.isEmpty then
    some ((chain mxm (F.filter fun q => !p.delNodes.contains q.2) As).filter
      fun q => !p.delNodes.contains q.2)
  else none

/-- Nothing pending: the committed batched path (`batched_eq_collapse`, `fused_chain_semantics`,
`chain_nodup` are its instances). -/
theorem batchedP_empty (mxm : Mat → Mat → Mat) (F : Mat) (As : List Mat) :
    batchedP mxm ⟨[], []⟩ F As = some (chain mxm F As) := by
  have hf : ∀ l : Mat, l.filter (fun _ => true) = l := fun l => List.filter_eq_self.2 (by simp)
  simp [batchedP, hf]

/-- **Batched path = per-row collapse, with pending deletes**: when the batched path runs,
row `i` of the result holds `d` iff the per-row collapsed expansion (#3577 model) of a seed
`s` of row `i` yields `(s, d)`. -/
theorem batchedP_eq_collapse {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) (g : Graph) (p : Pending)
    (F M : Mat) (hM : batchedP mxm p F [pairs g] = some M) (i d : Nat) :
    (i, d) ∈ M ↔
      ∃ s, (i, s) ∈ F ∧ ∃ e, (s, d, e) ∈ expandRowPend g p false false ⟨some s, none, []⟩ := by
  unfold batchedP at hM
  split at hM
  · rename_i hE
    have hR : p.delRels = [] := List.isEmpty_iff.1 hE
    cases hM
    rw [List.mem_filter, mem_chain h]
    simp only [Reach, mem_pairs, List.mem_filter, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not]
    constructor
    · rintro ⟨⟨s, ⟨hF, hs⟩, k, ⟨x, hx, h1, h2⟩, rfl⟩, hd⟩
      refine ⟨s, hF, collapseP_directed_iff.2 ⟨x, hx, h1, h2, by simp, by simp [hR], ?_, ?_, hs, hd⟩⟩
      · simp [okBound]
      · simp [okBound]
    · rintro ⟨s, hF, he⟩
      obtain ⟨x, hx, h1, h2, -, -, -, -, hs, hd⟩ := collapseP_directed_iff.1 he
      exact ⟨⟨s, ⟨hF, hs⟩, d, ⟨x, hx, h1, h2⟩, rfl⟩, hd⟩
  · cases hM

/-! ## #3452, CondTraverse half (fixed by #3577) -/

/-- `MATCH (a:A) DETACH DELETE a MERGE (a)-[r:R]->(x:B)`: #3577's CondTraverse matches nothing
for the pending-deleted `a`, so MERGE takes its create branch. -/
theorem main_3577_empty : expandRowPend gAB1 pDetach false true rowA = [] := by decide

/-- A pending-deleted target matches nothing either. -/
theorem main_3577_empty_dst : expandRowPend gAB1 ⟨[1], []⟩ false true rowA = [] := by decide

/-- An explicitly `DELETE`d edge is skipped. -/
theorem main_3577_empty_rel : expandRowPend gAB1 ⟨[], [0]⟩ false true rowA = [] := by decide

/-- …while before #3577 the same row matched the doomed edge (`EIP.pre3577_condTraverse_violates`). -/
theorem pre3577_vs_main :
    condTraversePre gAB1 pDetach false true rowA = [(0, 1, 0)] ∧
    expandRowPend gAB1 pDetach false true rowA = [] := by decide

end OpsTraverse.CTP
