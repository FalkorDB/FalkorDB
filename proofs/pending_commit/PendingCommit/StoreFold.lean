import PendingCommit.Store
/-
# Batches of per-entity writes over distinct ids

`AttributeStore::insert_attrs` (:1315), `insert_attrs_rows` (:1266),
`import_attrs` (:1351), `import_attrs_resolved` (:1379), `remove_all` (:1246)
and `decode_with_count` (:1460) all loop "for each entity, one
`DataBlock` write" and sum the per-entity counts. `runOps_spec`: when the ids
are distinct, each entity ends up with exactly its own write's result, every
other entity is untouched, and the totals are the sums of the per-entity
counts — whatever order the loop visits them in (Rust sorts by id; the hash
map order of `import_attrs` is irrelevant).
-/
namespace PendingCommit.Content

def runOps {α : Type} (op : D → Nat → α → D × Nat × Nat) : D → List (Nat × α) → D × Nat × Nat
  | d, [] => (d, 0, 0)
  | d, (k, a) :: L =>
    let r := op d k a
    let r' := runOps op r.1 L
    (r'.1, r.2.1 + r'.2.1, r.2.2 + r'.2.2)

/-- The entity's read after the batch: its own write's result, or unchanged. -/
def after {α : Type} (f : List Entry → α → List Entry) (old : List Entry) : Option α → List Entry
  | some a => f old a
  | none => old

def lk {α : Type} : List (Nat × α) → Nat → Option α
  | [], _ => none
  | (k, a) :: L, id => if id = k then some a else lk L id

structure OpSpec {α : Type} (op : D → Nat → α → D × Nat × Nat) (f : List Entry → α → List Entry)
    (g : List Entry → α → Nat × Nat) (ok : List Entry → α → Prop) : Prop where
  inv : ∀ d id a, SInv d → ok (dread d id) a → SInv (op d id a).1
  self : ∀ d id a, SInv d → ok (dread d id) a → dread (op d id a).1 id = f (dread d id) a
  frame : ∀ d id a, SInv d → ok (dread d id) a → ∀ id', id' ≠ id → dread (op d id a).1 id' = dread d id'
  cnt : ∀ d id a, SInv d → ok (dread d id) a → (op d id a).2 = g (dread d id) a

theorem lk_none {α : Type} (L : List (Nat × α)) (id : Nat) (h : id ∉ L.map (·.1)) : lk L id = none := by
  induction L with
  | nil => rfl
  | cons p L ih =>
    obtain ⟨k, a⟩ := p
    simp only [List.map_cons, List.mem_cons, not_or] at h
    simp only [lk]; rw [if_neg h.1, ih h.2]

/-- **`runOps_spec`**. -/
theorem runOps_spec {α : Type} (op : D → Nat → α → D × Nat × Nat) (f : List Entry → α → List Entry)
    (g : List Entry → α → Nat × Nat) (ok : List Entry → α → Prop) (hs : OpSpec op f g ok) :
    ∀ (L : List (Nat × α)) (d : D), SInv d → (L.map (·.1)).Nodup → (∀ p ∈ L, ok (dread d p.1) p.2) →
    let r := runOps op d L
    SInv r.1 ∧ (∀ id, dread r.1 id = after f (dread d id) (lk L id)) ∧
      r.2.1 = (L.map (fun p => (g (dread d p.1) p.2).1)).sum ∧
      r.2.2 = (L.map (fun p => (g (dread d p.1) p.2).2)).sum
  | [], d, h, _, _ => ⟨h, fun id => rfl, rfl, rfl⟩
  | (k, a) :: L, d, h, hnd, hok => by
    have hk := hok (k, a) (by simp)
    simp only [List.map_cons, List.nodup_cons] at hnd
    have h1 := hs.inv d k a h hk
    have hfr := hs.frame d k a h hk
    have hok' : ∀ p ∈ L, ok (dread (op d k a).1 p.1) p.2 := by
      intro p hp
      have hne : p.1 ≠ k := fun e => hnd.1 (e ▸ List.mem_map_of_mem hp)
      rw [hfr _ hne]; exact hok p (by simp [hp])
    obtain ⟨r1, r2, r3, r4⟩ := runOps_spec op f g ok hs L (op d k a).1 h1 hnd.2 hok'
    have hsum : ∀ (h : Nat × Nat → Nat), (L.map (fun p => h (g (dread (op d k a).1 p.1) p.2))).sum =
        (L.map (fun p => h (g (dread d p.1) p.2))).sum := by
      intro h; congr 1; apply List.map_congr_left; intro p hp
      have hne : p.1 ≠ k := fun e => hnd.1 (e ▸ List.mem_map_of_mem hp)
      rw [hfr _ hne]
    have hc := hs.cnt d k a h hk
    refine ⟨r1, ?_, ?_, ?_⟩
    · intro id
      simp only [runOps, lk]
      rw [r2 id]
      by_cases hid : id = k
      · subst hid; rw [lk_none L id hnd.1, if_pos rfl]; exact hs.self d id a h hk
      · rw [if_neg hid]
        cases lk L id with
        | none => exact hfr id hid
        | some b => simp only [after]; rw [hfr id hid]
    · simp only [runOps, List.map_cons, List.sum_cons]; rw [r3, hsum (·.1), hc]
    · simp only [runOps, List.map_cons, List.sum_cons]; rw [r4, hsum (·.2), hc]

/-! ## The per-entity specs -/

theorem merge_spec : OpSpec (fun d id ps => dmerge d id ps) (fun old ps => (mergeSpan old ps).1)
    (fun old ps => (mergeSpan old ps).2) (fun old ps => (mergeSpan old ps).1.length ≤ 65535) :=
  ⟨fun d id a h ok => (dmergeSpan_ok d h id a ok).1, fun d id a h ok => (dmergeSpan_ok d h id a ok).2.1,
   fun d id a h ok => (dmergeSpan_ok d h id a ok).2.2.2, fun d id a h ok => (dmergeSpan_ok d h id a ok).2.2.1⟩

theorem set_spec : OpSpec (fun d id ps => (dsetSpan d id ps, 0, ps.length)) (fun _ ps => ps)
    (fun _ ps => (0, ps.length)) (fun _ ps => ps.length ≤ 65535) :=
  ⟨fun d id a h ok => (dsetSpan_ok d h id a ok).1, fun d id a h ok => (dsetSpan_ok d h id a ok).2.1,
   fun d id a h ok => (dsetSpan_ok d h id a ok).2.2, fun _ _ _ _ _ => rfl⟩

theorem remove_spec : OpSpec (fun d id (_ : Unit) => (dremove d id, 0, 0)) (fun _ _ => [])
    (fun _ _ => (0, 0)) (fun _ _ => True) :=
  ⟨fun d id _ h _ => (dremove_ok d h id).1, fun d id _ h _ => (dremove_ok d h id).2.1,
   fun d id _ h _ => (dremove_ok d h id).2.2, fun _ _ _ _ _ => rfl⟩

end PendingCommit.Content
