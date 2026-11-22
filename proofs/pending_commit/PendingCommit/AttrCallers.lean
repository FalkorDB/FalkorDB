import PendingCommit.AttrInv
/-!
# The callers of `set_node_attributes` / `set_relationship_attributes` pass sorted attributes

`Pending::set_*_attributes` documents "`attrs` must be attribute-id-resolved, sorted by id, and
unique" and checks it only with `debug_assert!` (pending.rs). `AttrInv.stageAll_inv`
takes `Sorted attrs` as its premise. The one production caller is `CreateOp`
(ops/create.rs), which builds each row's list from `build_attr_template`
(create.rs):

```rust
let mut template = [];                              // one entry per distinct attribute id, written order
... template.push((id, 0, value_node)) / entry.2 = value_node   // a repeated id overwrites (last wins)
let mut order: Vec<usize> = (0..template.len()).collect();
order.sort_unstable_by_key(|&i| template[i].0);     // ids are distinct, so the order is unique
for (pos, &i) in order.iter().enumerate() { template[i].1 = pos; }     // pos = #smaller ids
...
let mut out = vec![(0u16, Value::Null); t.len()];                      // (create.rs)
for (attr_id, pos, value_node) in t { out[*pos] = (*attr_id, eval(value_node)); }
```

The template's ids are distinct (proofs/ops_apply `CreateAttrs.tplBuild_nodup`) and `pos` is the rank of
the id among them. Here: `rank` is the number of smaller ids; `rank_surj` shows the ranks of `n`
distinct ids are exactly `0..n-1`, so every slot of `out` is written; `placed_sorted` shows the
resulting vector is strictly ascending by attribute id, i.e. `Sorted` (so `stageAll_inv`'s premise).

The other route to a non-sorted list, `resolve_map_attrs` (`sort_unstable_by_key`, which would keep two
names that both resolved to `ATTRIBUTE_ID_NONE`), is only reached for an attrs expression that is not a
map literal; `CREATE (n $map)` is refused by the parser (live: "Encountered unhandled type in inlined
properties."), so it is unreachable from Cypher. The effects staging that calls `set_node_attributes`
(`effects/v3/staging.rs`) is `#[cfg(test)]` code.
-/
namespace PendingCommit.PA
open PendingCommit

/-- `pos`: the number of smaller ids (create.rs). -/
def rank (ids : List Nat) (id : Nat) : Nat := ids.countP (· < id)

theorem rank_cons (x : Nat) (xs : List Nat) (a : Nat) :
    rank (x :: xs) a = rank xs a + (if x < a then 1 else 0) := by
  simp [rank, List.countP_cons]

theorem rank_mono (l : List Nat) (a b : Nat) (h : a ≤ b) : rank l a ≤ rank l b := by
  induction l with
  | nil => simp [rank]
  | cons x xs ih =>
    rw [rank_cons, rank_cons]
    have := ih
    by_cases h1 : x < a
    · have : x < b := by omega
      simp [h1, this]; omega
    · by_cases h2 : x < b <;> simp [h1, h2] <;> omega

theorem rank_strict (l : List Nat) (a b : Nat) (ha : a ∈ l) (h : a < b) : rank l a < rank l b := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    rw [rank_cons, rank_cons]
    rcases List.mem_cons.mp ha with rfl | ha'
    · have := rank_mono xs a b (by omega)
      simp only [Nat.lt_irrefl, h, ite_true, ite_false]; omega
    · have := ih ha'
      by_cases hx : x < a
      · have : x < b := by omega
        simp [hx, this]; omega
      · by_cases hy : x < b <;> simp [hx, hy] <;> omega

theorem rank_lt (l : List Nat) (a : Nat) (ha : a ∈ l) : rank l a < l.length := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    rw [rank_cons, List.length_cons]
    rcases List.mem_cons.mp ha with rfl | ha'
    · have : rank xs a ≤ xs.length := List.countP_le_length
      simp only [Nat.lt_irrefl, ite_false]; omega
    · have := ih ha'
      by_cases hx : x < a <;> simp only [hx, ite_true, ite_false] <;> omega

/-- Distinct ids have distinct ranks. -/
theorem rank_inj (l : List Nat) (a b : Nat) (ha : a ∈ l) (hb : b ∈ l) (h : rank l a = rank l b) : a = b := by
  by_cases hab : a < b
  · have := rank_strict l a b ha hab; omega
  · by_cases hba : b < a
    · have := rank_strict l b a hb hba; omega
    · omega

theorem nodup_map_of_inj {α β : Type} (f : α → β) : ∀ l : List α, l.Nodup →
    (∀ a ∈ l, ∀ b ∈ l, f a = f b → a = b) → (l.map f).Nodup
  | [], _, _ => List.nodup_nil
  | x :: xs, h, hi => by
    rw [List.map_cons, List.nodup_cons]
    refine ⟨fun hm => ?_, nodup_map_of_inj f xs (List.nodup_cons.1 h).2
      (fun a ha b hb e => hi a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb) e)⟩
    obtain ⟨y, hy, hxy⟩ := List.mem_map.1 hm
    have := hi y (List.mem_cons_of_mem _ hy) x (by simp) hxy
    exact (List.nodup_cons.1 h).1 (this ▸ hy)

/-- The ranks of `n` distinct ids are exactly `0..n-1`. -/
theorem rank_surj (ids : List Nat) (hnd : ids.Nodup) (p : Nat) (hp : p < ids.length) :
    ∃ i ∈ ids, rank ids i = p := by
  refine Classical.byContradiction fun hno => ?_
  have hno' : ∀ i ∈ ids, rank ids i ≠ p := fun i hi h => hno ⟨i, hi, h⟩
  have hsub : (ids.map (rank ids)) ⊆ (List.range ids.length).erase p := by
    intro x hx
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
    rw [List.mem_erase_of_ne (hno' i hi)]
    exact List.mem_range.2 (rank_lt ids i hi)
  have hnd2 : (ids.map (rank ids)).Nodup := by
    exact nodup_map_of_inj _ _ hnd (fun a ha b hb h => rank_inj ids a b ha hb h)
  have := List.Nodup.length_le_of_subset hnd2 hsub
  rw [List.length_map, List.length_erase_of_mem (List.mem_range.2 hp), List.length_range] at this
  omega

/-- Writes `out[pos] = x` in order (`Vec` indexing, `List.set`). -/
def writeAll {β : Type} (l : List β) (ws : List (Nat × β)) : List β :=
  ws.foldl (fun o w => o.set w.1 w.2) l

theorem writeAll_cons {β : Type} (l : List β) (w : Nat × β) (ws : List (Nat × β)) :
    writeAll l (w :: ws) = writeAll (l.set w.1 w.2) ws := rfl

theorem writeAll_length {β : Type} (ws : List (Nat × β)) : ∀ l : List β, (writeAll l ws).length = l.length := by
  induction ws with
  | nil => intro l; rfl
  | cons w ws ih => intro l; rw [writeAll_cons, ih]; simp

theorem writeAll_untouched {β : Type} (ws : List (Nat × β)) :
    ∀ (l : List β) (q : Nat), (∀ w ∈ ws, w.1 ≠ q) → (writeAll l ws)[q]? = l[q]? := by
  induction ws with
  | nil => intro l q _; rfl
  | cons w ws ih =>
    intro l q h
    rw [writeAll_cons, ih _ q (fun w' hw' => h w' (List.mem_cons_of_mem _ hw'))]
    exact List.getElem?_set_ne (h w (by simp))

/-- With pairwise distinct in-range positions, each written slot holds the value written there. -/
theorem writeAll_get {β : Type} (ws : List (Nat × β)) :
    ∀ (l : List β), (ws.map (·.1)).Nodup → (∀ w ∈ ws, w.1 < l.length) →
    ∀ w ∈ ws, (writeAll l ws)[w.1]? = some w.2 := by
  induction ws with
  | nil => intro l _ _ w hw; simp at hw
  | cons h ws ih =>
    intro l hnd hlt w hw
    simp only [List.map_cons, List.nodup_cons] at hnd
    rw [writeAll_cons]
    rcases List.mem_cons.1 hw with rfl | hw'
    · rw [writeAll_untouched ws _ _ (fun w' hw' e => hnd.1 (List.mem_map.2 ⟨w', hw', e⟩))]
      exact List.getElem?_set_self (hlt w (by simp))
    · exact ih (l.set h.1 h.2) hnd.2
        (fun w' hw' => by rw [List.length_set]; exact hlt w' (List.mem_cons_of_mem _ hw')) w hw'

/-- A list of `(attribute id, value)` strictly ascending by id. -/
def SortedP {V : Type} (l : List (Nat × V)) : Prop := List.Pairwise (fun a b => a.1 < b.1) l

/-- A template entry: attribute id, position, value. -/
abbrev Tpl (V : Type) := List (Nat × Nat × V)

/-- `out = vec![dflt; t.len()]; for (id, pos, v) in t { out[pos] = (id, v) }` (create.rs). -/
def placed {V : Type} (dflt : Nat × V) (t : Tpl V) : List (Nat × V) :=
  writeAll (List.replicate t.length dflt) (t.map (fun e => (e.2.1, (e.1, e.2.2))))

/-- **The `CreateOp` attribute lists are strictly ascending by id**: for a template whose ids are
distinct and whose positions are the ranks of the ids, the placed vector is sorted, for every row's
values and every default. With `V = Val` this is `Sorted`, `stageAll_inv`'s premise for `CreateOp`. -/
theorem placed_sorted {V : Type} (dflt : Nat × V) (t : Tpl V) (hnd : (t.map (·.1)).Nodup)
    (hpos : ∀ e ∈ t, e.2.1 = rank (t.map (·.1)) e.1) : SortedP (placed dflt t) := by
  have hlen : (List.replicate t.length dflt).length = t.length := by simp
  have hposnd : ((t.map (fun e => (e.2.1, (e.1, e.2.2)))).map (·.1)).Nodup := by
    simp only [List.map_map, Function.comp_def]
    have h1 : (t.map (fun e => e.2.1)) = t.map (fun e => rank (t.map (·.1)) e.1) :=
      List.map_congr_left (fun e he => hpos e he)
    rw [h1]
    have h2 : t.map (fun e => rank (t.map (·.1)) e.1) = (t.map (·.1)).map (rank (t.map (·.1))) := by
      rw [List.map_map]; rfl
    rw [h2]
    exact nodup_map_of_inj _ _ hnd (fun a ha b hb h => rank_inj _ a b ha hb h)
  have hlt : ∀ w ∈ t.map (fun e => (e.2.1, (e.1, e.2.2))), w.1 < (List.replicate t.length dflt).length := by
    intro w hw
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hw
    rw [hlen]; show e.2.1 < t.length
    rw [hpos e he]
    have := rank_lt (t.map (·.1)) e.1 (List.mem_map.2 ⟨e, he, rfl⟩)
    simpa using this
  have hslot : ∀ q (hq : q < t.length), ∃ e ∈ t, e.2.1 = q ∧ (placed dflt t)[q]? = some (e.1, e.2.2) := by
    intro q hq
    have hn : (t.map (·.1)).length = t.length := by simp
    obtain ⟨i, hi, hr⟩ := rank_surj (t.map (·.1)) hnd q (by omega)
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hi
    have hq' : e.2.1 = q := by rw [hpos e he]; exact hr
    refine ⟨e, he, hq', ?_⟩
    have := writeAll_get (t.map (fun e => (e.2.1, (e.1, e.2.2)))) (List.replicate t.length dflt) hposnd hlt
      (e.2.1, (e.1, e.2.2)) (List.mem_map.2 ⟨e, he, rfl⟩)
    unfold placed; rw [← hq']; exact this
  have hplen : (placed dflt t).length = t.length := by
    unfold placed; rw [writeAll_length]; simp
  unfold SortedP
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  rw [hplen] at hi hj
  obtain ⟨ei, hei, hpi, hgi⟩ := hslot i hi
  obtain ⟨ej, hej, hpj, hgj⟩ := hslot j hj
  have g1 : (placed dflt t)[i]? = some ((placed dflt t)[i]'(by rw [hplen]; exact hi)) := List.getElem?_eq_getElem _
  have g2 : (placed dflt t)[j]? = some ((placed dflt t)[j]'(by rw [hplen]; exact hj)) := List.getElem?_eq_getElem _
  rw [hgi] at g1; rw [hgj] at g2
  have e1 : (placed dflt t)[i]'(by rw [hplen]; exact hi) = (ei.1, ei.2.2) := (Option.some.inj g1).symm
  have e2 : (placed dflt t)[j]'(by rw [hplen]; exact hj) = (ej.1, ej.2.2) := (Option.some.inj g2).symm
  rw [e1, e2]
  show ei.1 < ej.1
  refine Classical.byContradiction fun hge => ?_
  have := rank_mono (t.map (·.1)) ej.1 ei.1 (by omega)
  have r1 := hpos ei hei; have r2 := hpos ej hej
  omega

/-- For `V = Val`: the `Sorted` premise of `AttrInv.stageAll_inv`. -/
theorem placed_Sorted (dflt : Entry) (t : Tpl Val) (hnd : (t.map (·.1)).Nodup)
    (hpos : ∀ e ∈ t, e.2.1 = rank (t.map (·.1)) e.1) : Sorted (placed dflt t) :=
  placed_sorted dflt t hnd hpos

end PendingCommit.PA
