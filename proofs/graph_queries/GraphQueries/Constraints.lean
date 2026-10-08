import GraphQueries.Index
/-
# Constraint bookkeeping (constraint.rs and graph.rs constraint management)

| here | there |
| --- | --- |
| `ctStr`, `csStr` | `Display for ConstraintType` / `ConstraintStatus` (constraint.rs:14,34) |
| `cnew`           | `Constraint::new` (constraint.rs:60), `NEXT_CONSTRAINT_ID` counter |
| `cmatches`       | `Constraint::matches` (constraint.rs:79) |
| `addRaw`         | `add_constraint_raw` (graph.rs:4178) |
| `upsert`         | `upsert_constraint_raw` (graph.rs:4194) |
| `entityCount`    | `get_constraint_entity_count` (graph.rs:4216) |
| `validate`       | `validate_constraint` (graph.rs:4228) |
| `validatePending`| `validate_pending_constraints` (graph.rs:4239) |
| `computePending` | `compute_pending_constraint_results` (graph.rs:4268) |
| `applyResults`   | `apply_constraint_validation_results` (graph.rs:4280), ids unique |
| `dropC`          | `drop_constraint` (graph.rs:4413) — `swap_remove` |
| `hasSupporting`  | `has_supporting_index` (graph.rs:4436) |
| `dependsOn`      | `constraint_depends_on_index` (graph.rs:4455) |
-/
namespace GQ
variable {V : Type}

def ctStr : CType → String | .unique => "UNIQUE" | .mandatory => "MANDATORY"
def csStr : CStatus → String
  | .underConstruction => "UNDER CONSTRUCTION" | .operational => "OPERATIONAL" | .failed => "FAILED"
theorem str_spec : ctStr .unique = "UNIQUE" ∧ ctStr .mandatory = "MANDATORY" ∧
    csStr .underConstruction = "UNDER CONSTRUCTION" ∧ csStr .operational = "OPERATIONAL" ∧
    csStr .failed = "FAILED" := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `Constraint::new`: fresh id from the process counter, UNDER CONSTRUCTION. -/
def cnew (ctr : Nat) (ct : CType) (et : EType) (label : String) (props : List String) : Constraint × Nat :=
  (⟨ctr, ct, et, label, props, .underConstruction⟩, ctr + 1)

theorem cnew_fresh (ctr : Nat) (ct : CType) (et : EType) (l : String) (p : List String) :
    (cnew ctr ct et l p).1.status = .underConstruction ∧ (cnew ctr ct et l p).1.id < (cnew ctr ct et l p).2 :=
  ⟨rfl, Nat.lt_succ_self _⟩

def cmatches (c : Constraint) (ct : CType) (et : EType) (label : String) (props : List String) : Bool :=
  c.ct == ct && c.et == et && c.label == label && c.props.length == props.length &&
  (c.props.zip props).all (fun (a, b) => a == b)

theorem zip_all_eq : ∀ (a b : List String), a.length = b.length →
    ((a.zip b).all (fun (x, y) => x == y) = true ↔ a = b)
  | [], [], _ => by simp
  | x :: a, y :: b, h => by
    simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true, beq_iff_eq, List.cons.injEq]
    rw [zip_all_eq a b (by simpa using h)]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

/-- `matches` is plain equality of all four fields — the property list
**in order**. -/
theorem cmatches_iff (c : Constraint) (ct : CType) (et : EType) (l : String) (p : List String) :
    cmatches c ct et l p = true ↔ c.ct = ct ∧ c.et = et ∧ c.label = l ∧ c.props = p := by
  unfold cmatches
  simp only [Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩; exact ⟨h1, h2, h3, (zip_all_eq _ _ h4).1 h5⟩
  · rintro ⟨h1, h2, h3, rfl⟩; exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, rfl⟩, (zip_all_eq _ _ rfl).2 rfl⟩

/-- CONFIRMED divergence from C: `UNIQUE (a, b)` and `UNIQUE (b, a)` are
different constraints to Rust (both get created), the same to C
("Constraint already exists"). -/
theorem cmatches_order_sensitive :
    cmatches ⟨1, .unique, .node, "L", ["a", "b"], .operational⟩ .unique .node "L" ["b", "a"] = false := by
  decide

def addRaw (cs : List Constraint) (c : Constraint) : List Constraint := cs ++ [c]

def sameKey (c : Constraint) (ct : CType) (et : EType) (l : String) (p : List String) : Bool :=
  c.ct == ct && c.et == et && c.label == l && c.props == p

def upsert (cs : List Constraint) (ctr : Nat) (ct : CType) (et : EType) (l : String) (p : List String)
    (st : CStatus) : List Constraint × Nat :=
  match cs.findIdx? (sameKey · ct et l p) with
  | some i => (cs.modify i (fun c => { c with status := st }), ctr)
  | none => let (c, ctr') := cnew ctr ct et l p; (cs ++ [{ c with status := st }], ctr')

/-- Re-announcing a constraint (the primary sends it twice) never duplicates it. -/
theorem upsert_twice (cs : List Constraint) (ctr : Nat) (ct : CType) (et : EType) (l : String)
    (p : List String) (s1 s2 : CStatus) :
    let r1 := upsert cs ctr ct et l p s1
    let r2 := upsert r1.1 r1.2 ct et l p s2
    r2.1.length = r1.1.length ∧ r2.2 = r1.2 := by
  cases h : cs.findIdx? (sameKey · ct et l p) with
  | some i =>
    have : (cs.modify i (fun c => { c with status := s1 })).findIdx? (sameKey · ct et l p) = some i := by
      rw [List.findIdx?_eq_some_iff_getElem] at h ⊢
      obtain ⟨hi, h1, h2⟩ := h
      refine ⟨by simpa using hi, ?_, ?_⟩
      · simp only [List.getElem_modify, ite_true]; simpa [sameKey] using h1
      · intro j hj; simp only [List.getElem_modify]; rw [if_neg (Nat.ne_of_gt hj)]
        exact h2 j hj
    simp only [upsert, h]; rw [this]; simp
  | none =>
    have : (cs ++ [({ id := ctr, ct := ct, et := et, label := l, props := p, status := s1 } : Constraint)]).findIdx?
        (sameKey · ct et l p) = some cs.length := by
      rw [List.findIdx?_eq_some_iff_getElem]
      refine ⟨by simp, by simp [sameKey], fun j hj => ?_⟩
      rw [List.getElem_append_left hj]
      have := List.findIdx?_eq_none_iff.1 h
      simpa using this _ (List.getElem_mem hj)
    simp only [upsert, h, cnew]; rw [this]; simp

def entityCount (g : G V) (et : EType) (label : String) : Nat :=
  match et with
  | .node => labelNodeCount g label
  | .rel => ((getRelMat g label).map Ten.edgeCount).getD 0

def validate (mand uniq : Constraint → Bool) (c : Constraint) : Bool :=
  match c.ct with | .mandatory => mand c | .unique => uniq c

def settle (v : Bool) : CStatus := if v then .operational else .failed

def validatePending (val : Constraint → Bool) (cs : List Constraint) : List Constraint :=
  cs.map (fun c => if c.status = .underConstruction then { c with status := settle (val c) } else c)

def computePending (val : Constraint → Bool) (cs : List Constraint) : List (Nat × Bool) :=
  (cs.filter (·.status = .underConstruction)).map (fun c => (c.id, val c))

def applyResults (res : List (Nat × Bool)) (cs : List Constraint) : List Constraint :=
  cs.map (fun c => if c.status = .underConstruction then
    match res.lookup c.id with | some v => { c with status := settle v } | none => c else c)

theorem lookup_compute (val : Constraint → Bool) :
    ∀ (cs : List Constraint) (c : Constraint), (cs.map (·.id)).Nodup → c ∈ cs →
    c.status = .underConstruction → (computePending val cs).lookup c.id = some (val c)
  | [], _, _, h, _ => by cases h
  | d :: cs, c, hn, hm, hu => by
    simp only [List.map_cons, List.nodup_cons] at hn
    simp only [computePending, List.filter_cons] at *
    rcases List.mem_cons.1 hm with rfl | hm'
    · simp [hu, List.lookup]
    · have hne : d.id ≠ c.id := fun h => hn.1 (h ▸ List.mem_map_of_mem hm')
      have ih := lookup_compute val cs c hn.2 hm' hu
      have hb : (c.id == d.id) = false := by simp [Ne.symm hne]
      split
      · simp only [List.map_cons, List.lookup, hb]; exact ih
      · exact ih

/-- The two-phase background path (read-only compute, then apply under the
write lock) reaches the same statuses as the one-shot path when nothing was
dropped in between (constraint ids are unique). -/
theorem two_phase_eq (val : Constraint → Bool) (cs : List Constraint) (hn : (cs.map (·.id)).Nodup) :
    applyResults (computePending val cs) cs = validatePending val cs := by
  unfold applyResults validatePending
  apply List.map_congr_left
  intro c hc
  split
  · rename_i hu; rw [lookup_compute val cs c hn hc hu]
  · rfl

/-- `swap_remove(i)`: the last element takes slot `i`. -/
def swapRemove {α} (l : List α) (i : Nat) : List α :=
  if i + 1 = l.length then l.dropLast else
  match l.getLast? with | some x => (l.set i x).dropLast | none => l

def dropC (cs : List Constraint) (ct : CType) (et : EType) (l : String)
    (p : List String) : Except String (List Constraint) :=
  match cs.findIdx? (cmatches · ct et l p) with
  | some i => .ok (swapRemove cs i)
  | none => .error "Unable to drop constraint, no such constraint."

theorem swapRemove_length {α} (l : List α) (i : Nat) (h : i < l.length) :
    (swapRemove l i).length = l.length - 1 := by
  unfold swapRemove; split
  · simp
  · cases hl : l.getLast? with
    | none => simp at hl; subst hl; simp at h
    | some x => simp

theorem dropC_length (cs : List Constraint) (ct : CType) (et : EType) (l : String) (p : List String)
    (r : List Constraint) (h : dropC cs ct et l p = .ok r) : r.length + 1 = cs.length := by
  unfold dropC at h
  split at h
  · rename_i i hi
    cases h
    have := (List.findIdx?_eq_some_iff_getElem.1 hi).1
    rw [swapRemove_length _ _ this]; omega
  · cases h

def hasSupporting (hasField : String → String → Bool) (label : String) (props : List String) : Bool :=
  props.all (hasField label)

def dependsOn (cs : List Constraint) (et : EType) (label attr : String) (isRange : Bool) : Bool :=
  isRange && cs.any (fun c => c.ct == .unique && c.et == et && c.label == label && c.props.contains attr)

/-- An index backing a UNIQUE constraint cannot be dropped. -/
theorem drop_blocked (cs : List Constraint) (et : EType) (label : String) (fieldsOfType attrs : List String)
    (dropped : Nat) (c : Constraint) (hc : c ∈ cs) (hu : c.ct = .unique) (he : c.et = et)
    (hl : c.label = label) (a : String) (ha : a ∈ (if attrs = [] then fieldsOfType else attrs))
    (hp : a ∈ c.props) :
    dropIndex (fun x => dependsOn cs et label x true) fieldsOfType attrs dropped =
      .error "Index supports constraint" := by
  apply dropIndex_protects _ _ _ _ a ha
  simp only [dependsOn, Bool.true_and, List.any_eq_true, Bool.and_eq_true, beq_iff_eq,
    List.contains_iff_mem]
  exact ⟨c, hc, ⟨⟨⟨hu, he⟩, hl⟩, hp⟩⟩

theorem hasSupporting_nil (hasField : String → String → Bool) (l : String) :
    hasSupporting hasField l [] = true := rfl

end GQ
