import GraphQueries.Basic
/-
# Schema tables: labels, relationship types, capacity growth

| here | there (graph.rs) |
| --- | --- |
| `nextMul`      | `u64::next_multiple_of` (core) |
| `grbMax`, `ckAdd`, `ckNextMul` | `GrB_INDEX_MAX` (tensor.rs), `checked_add`, `checked_next_multiple_of` |
| `growStep`, `growCap` | `grow_cap`  (loop body :847-852) |
| `resizeNodeMs`/`resize` | `resize_node_matrices`  / `resize`  (#3967: no type matrix to resize) |
| `edgeTypeSlot`/`relTypeSlot`/`relTypesFit` | `edge_type_slot`  / `relationship_type_slot`  / `relationship_types_fit`  (#3967) |
| `idx`          | `NameMap::id_of` (schema.rs; first position, `FirstInv` in pending_commit `NameMap`) |
| `internLabel`  | `intern_label`  |
| `getLabelId`   | `get_label_id`  |
| `getLabelIdMut`| `get_label_id_mut`  |
| `getTypeId`    | `get_type_id`  |
| `getTypeIdMut`, `registerRelType` | `get_type_id_mut` , `register_relationship_type`  (resizes; a known type is returned as is) |
| `getRelMatMut` | `get_relationship_matrix_mut`  |
| `getRelMat`    | `get_relationship_matrix`  |
| `getLabelMat`/`getLabelMatMut` | `get_label_matrix`  / `get_label_matrix_mut`  |
| `nodeHasLabel`/`nodeHasLabelId`/`edgeHasType`/`edgeHasTypeId` | `node_has_label`  / `node_has_label_id`  / `edge_has_type`  / `edge_has_type_id`  |
| `labelNodeCount`/`..ByIdx`/`typeEdgeCount` | `label_node_count`  / `label_node_count_by_idx`  / `type_edge_count`  |
-/
namespace GQ
variable {V : Type}

/-! ## `grow_cap` -/

/-- `x.next_multiple_of(c)` (core: `r == 0 ? x : x + (c - r)`). -/
def nextMul (c x : Nat) : Nat := if x % c = 0 then x else x + (c - x % c)

theorem nextMul_ge (c x : Nat) : x ≤ nextMul c x := by
  unfold nextMul; split
  · exact Nat.le_refl _
  · exact Nat.le_add_right _ _

theorem nextMul_dvd (c x : Nat) (hc : 0 < c) : nextMul c x % c = 0 := by
  unfold nextMul; split
  · assumption
  · have h1 := Nat.mod_lt x hc
    have h0 := Nat.mod_le x c
    have h2 : x + (c - x % c) = (x - x % c) + c := by omega
    rw [h2, Nat.add_mod_right]
    have h3 := Nat.div_add_mod x c
    have h4 : x - x % c = c * (x / c) := by omega
    rw [h4, Nat.mul_mod_right]

/-- `GrB_INDEX_MAX` (`tensor.rs`): `2^60 - 1`, the largest dimension GraphBLAS accepts. -/
def grbMax : Nat := 2 ^ 60 - 1

/-- `u64::checked_add`. -/
def ckAdd (a b : Nat) : Option Nat := if a + b < 2 ^ 64 then some (a + b) else none

/-- `u64::checked_next_multiple_of(c)` (core: `r == 0 ? Some(x) : x.checked_add(c - r)`). -/
def ckNextMul (c x : Nat) : Option Nat := if nextMul c x < 2 ^ 64 then some (nextMul c x) else none

/-- One step of `grow_cap`'s loop (:871-876):
`cap.checked_add(max(cap/4, chunk)).and_then(|c| c.checked_next_multiple_of(chunk))
 .map_or(GrB_INDEX_MAX, |c| c.min(GrB_INDEX_MAX))`. -/
def growStep (chunk cap : Nat) : Nat :=
  match (ckAdd cap (max (cap / 4) chunk)).bind (ckNextMul chunk) with
  | some c => min c grbMax
  | none => grbMax

theorem growStep_le (chunk cap : Nat) : growStep chunk cap ≤ grbMax := by
  unfold growStep; split <;> omega

theorem growStep_gt (chunk cap : Nat) (hc : 0 < chunk) (h : cap < grbMax) : cap < growStep chunk cap := by
  unfold growStep
  split
  · rename_i c hcs
    unfold ckAdd at hcs; split at hcs
    · simp only [Option.bind_some, ckNextMul] at hcs; split at hcs
      · cases hcs
        have := nextMul_ge chunk (cap + max (cap / 4) chunk)
        have : chunk ≤ max (cap / 4) chunk := Nat.le_max_right _ _
        omega
      · cases hcs
    · cases hcs
  · exact h

/-- `grow_cap` (:866): `while needed > cap && cap < GrB_INDEX_MAX { cap = step }`.
`chunk` is `NODE_CREATION_BUFFER`, normalised to a power of two ≥ 128, so `0 < chunk`.
The model is total (`termination_by grbMax - cap`): the loop always ends. -/
def growCap (chunk cap needed : Nat) (hc : 0 < chunk) : Nat :=
  if h : needed > cap ∧ cap < grbMax then
    have : grbMax - growStep chunk cap < grbMax - cap := by
      have := growStep_gt chunk cap hc h.2; omega
    growCap chunk (growStep chunk cap) needed hc
  else cap
termination_by grbMax - cap

theorem growCap_ge_cap (chunk cap needed : Nat) (hc : 0 < chunk) :
    cap ≤ growCap chunk cap needed hc := by
  fun_induction growCap chunk cap needed hc with
  | case1 cap h _ ih => have := growStep_gt chunk cap hc h.2; omega
  | case2 cap h => omega

/-- **Never past `GrB_INDEX_MAX`**: from a dimension GraphBLAS accepts, every
grown dimension is one it accepts. -/
theorem growCap_le_max (chunk cap needed : Nat) (hc : 0 < chunk) (h0 : cap ≤ grbMax) :
    growCap chunk cap needed hc ≤ grbMax := by
  fun_induction growCap chunk cap needed hc with
  | case1 cap h _ ih => exact ih (growStep_le chunk cap)
  | case2 cap h => exact h0

/-- Grows to `needed` whenever `needed` is a dimension GraphBLAS accepts — always so for a
create, which `IdSpace::create` refuses past `ID_LIMIT = GrB_INDEX_MAX`. -/
theorem growCap_ge_needed (chunk cap needed : Nat) (hc : 0 < chunk) (hn : needed ≤ grbMax) :
    needed ≤ growCap chunk cap needed hc := by
  fun_induction growCap chunk cap needed hc with
  | case1 cap h _ ih => exact ih
  | case2 cap h => omega

/-- Otherwise it stops at `GrB_INDEX_MAX` (the Rust test
`growth_stops_at_the_largest_graphblas_dimension`). -/
theorem growCap_reaches (chunk cap needed : Nat) (hc : 0 < chunk) (h0 : cap ≤ grbMax) :
    needed ≤ growCap chunk cap needed hc ∨ growCap chunk cap needed hc = grbMax := by
  fun_induction growCap chunk cap needed hc with
  | case1 cap h _ ih => exact ih (growStep_le chunk cap)
  | case2 cap h => omega

theorem growCap_noop (chunk cap needed : Nat) (hc : 0 < chunk) (h : needed ≤ cap) :
    growCap chunk cap needed hc = cap := by
  rw [growCap, dite_eq_right_of_eq_false (by simp; omega)]

/-- Every grown capacity is a whole number of `NODE_CREATION_BUFFER` chunks, or the clamp. -/
theorem growCap_chunked (chunk cap needed : Nat) (hc : 0 < chunk) (h : needed > cap)
    (h0 : cap < grbMax) :
    growCap chunk cap needed hc % chunk = 0 ∨ growCap chunk cap needed hc = grbMax := by
  fun_induction growCap chunk cap needed hc with
  | case1 cap h' _ ih =>
    by_cases h2 : needed > growStep chunk cap ∧ growStep chunk cap < grbMax
    · exact ih h2.1 h2.2
    · rw [growCap, dite_eq_right_of_eq_false (by simpa using h2)]
      unfold growStep; split
      · rename_i c hcs
        unfold ckAdd at hcs; split at hcs
        · simp only [Option.bind_some, ckNextMul] at hcs; split at hcs
          · cases hcs
            by_cases hm : nextMul chunk (cap + max (cap / 4) chunk) ≤ grbMax
            · left; rw [Nat.min_eq_left hm]; exact nextMul_dvd _ _ hc
            · right; omega
          · cases hcs
        · cases hcs
      · exact .inr rfl
  | case2 cap h' => exact absurd ⟨h, h0⟩ h'

/-- **Ordinary growth is unchanged**: when no step reaches the top, a step is the old
unchecked `(cap + max(cap/4, chunk)).next_multiple_of(chunk)`. -/
theorem growStep_small (chunk cap : Nat) (h : nextMul chunk (cap + max (cap / 4) chunk) ≤ grbMax) :
    growStep chunk cap = nextMul chunk (cap + max (cap / 4) chunk) := by
  have := nextMul_ge chunk (cap + max (cap / 4) chunk)
  have hg : grbMax < 2 ^ 64 := by unfold grbMax; omega
  unfold growStep ckAdd
  rw [ite_eq_left (by omega)]
  simp only [Option.bind_some, ckNextMul]
  rw [ite_eq_left (by omega)]
  exact Nat.min_eq_left h

-- The Rust tests `grow_cap_tests` (the `mod grow_cap_tests` at the end of graph.rs), `chunk = DEFAULT_NODE_CREATION_BUFFER = 16384`.
#guard growCap 16384 16384 (2 ^ 64 - 2) (by decide) == grbMax
#guard growCap 16384 16384 grbMax (by decide) == grbMax
#guard growCap 16384 (grbMax - 1) grbMax (by decide) == grbMax
#guard growCap 16384 16384 16384 (by decide) == 16384
#guard growCap 16384 16384 (16384 + 1) (by decide) == 2 * 16384
#guard growCap 16384 (100 * 16384) (100 * 16384 + 1) (by decide) == 100 * 16384 + 100 * 16384 / 4

/-! ## `resize` -/

def resizeNodeMs (g : G V) : G V :=
  { g with adj := g.adj.resize g.nodeCap g.nodeCap
           nodeLabels := g.nodeLabels.resize g.nodeCap g.labelMs.length
           labelMs := g.labelMs.map (·.resize g.nodeCap g.nodeCap) }
  -- tensors: `Tensor::resize` keeps the logical triples (dims are not modelled on `Ten`)

def resize (chunk : Nat) (hc : 0 < chunk) (g : G V) : G V :=
  let g1 := if g.nodeCount > g.nodeCap then
      resizeNodeMs { g with nodeCap := growCap chunk g.nodeCap g.nodeCount hc } else g
  let g2 := if g1.labelMs.length > g1.nodeLabels.nc then
      { g1 with nodeLabels := g1.nodeLabels.resize g1.nodeCap g1.labelMs.length } else g1
  if g2.relCount > g2.relCap then
      { g2 with relCap := growCap chunk g2.relCap g2.relCount hc } else g2

/-- Post-condition of `resize`: every dimension covers what it must hold. -/
theorem resize_post (chunk : Nat) (hc : 0 < chunk) (g : G V) :
    let r := resize chunk hc g
    (g.nodeCount ≤ grbMax → r.nodeCount ≤ r.nodeCap) ∧ r.labelMs.length ≤ r.nodeLabels.nc ∧
    (g.relCount ≤ grbMax → r.relCount ≤ r.relCap) := by
  have gn := growCap_ge_needed chunk g.nodeCap g.nodeCount hc
  have gr := growCap_ge_needed chunk g.relCap g.relCount hc
  simp only [resize]
  by_cases h1 : g.nodeCount > g.nodeCap <;> simp only [h1, ite_true, ite_false] <;>
  (repeat' split) <;> simp_all [resizeNodeMs, Mat.resize] <;> omega

/-! ## Label / type id tables -/

/-- `iter().position(|t| t == s)`. -/
def idx : List String → String → Option Nat
  | [], _ => none
  | a :: l, s => if a = s then some 0 else (idx l s).map (· + 1)

theorem idx_get : ∀ (l : List String) s i, idx l s = some i → l[i]? = some s
  | [], _, _, h => by simp [idx] at h
  | a :: l, s, i, h => by
    simp only [idx] at h; split at h
    · cases h; simp_all
    · cases hh : idx l s with
      | none => simp [hh] at h
      | some k => simp [hh] at h; subst h; simp [idx_get l s k hh]

theorem idx_none : ∀ (l : List String) s, idx l s = none ↔ s ∉ l
  | [], _ => by simp [idx]
  | a :: l, s => by
    simp only [idx]; split
    · simp_all
    · rw [Option.map_eq_none_iff, idx_none l s]; simp_all [eq_comm]

theorem idx_append (l : List String) (t s : String) :
    idx (l ++ [t]) s = match idx l s with
      | some i => some i
      | none => if t = s then some l.length else none := by
  induction l with
  | nil => simp [idx]
  | cons a l ih =>
    simp only [List.cons_append, idx]; split
    · rfl
    · rw [ih]; cases idx l s <;> simp <;> split <;> simp_all

/-- The invariant `intern_label` maintains: `node_labels_index` is exactly the
inverse of `node_labels` (first position), and one label matrix per label. -/
def LInv (g : G V) : Prop := (∀ s, g.labelsIndex s = idx g.labels s) ∧ g.labelMs.length = g.labels.length

def getLabelId (g : G V) (s : String) : Option Nat := g.labelsIndex s

def internLabel (g : G V) (s : String) : G V × Nat :=
  match g.labelsIndex s with
  | some id => (g, id)
  | none =>
    let id := g.labels.length
    ({ g with labels := g.labels ++ [s],
              labelsIndex := fun t => if t = s then some id else g.labelsIndex t }, id)

theorem internLabel_spec (g : G V) (s : String) (h : ∀ t, g.labelsIndex t = idx g.labels t) :
    let r := internLabel g s
    (∀ t, r.1.labelsIndex t = idx r.1.labels t) ∧ r.1.labels[r.2]? = some s ∧
    getLabelId r.1 s = some r.2 := by
  cases hs : g.labelsIndex s with
  | some id =>
    simp only [internLabel, hs, getLabelId]; refine ⟨h, ?_, trivial⟩
    rw [h] at hs; exact idx_get _ _ _ hs
  | none =>
    simp only [internLabel, hs, getLabelId, ite_true]
    refine ⟨fun t => ?_, by simp, trivial⟩
    rw [idx_append]; by_cases ht : t = s
    · subst ht; rw [← h, hs]
    · simp only [ht, ite_false, ← h]; cases g.labelsIndex t <;> simp [Ne.symm ht]

/-- `get_label_id_mut` (:1262): existing id, or intern + push a
`node_cap × node_cap` matrix + `resize`. -/
def getLabelIdMut (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) : G V × Nat :=
  match getLabelId g s with
  | some id => (g, id)
  | none =>
    let (g1, id) := internLabel g s
    (resize chunk hc { g1 with labelMs := g1.labelMs ++ [Mat.empty g1.nodeCap g1.nodeCap] }, id)

theorem resize_keeps (chunk : Nat) (hc : 0 < chunk) (g : G V) :
    let r := resize chunk hc g
    r.labels = g.labels ∧ r.labelsIndex = g.labelsIndex ∧ r.labelMs.length = g.labelMs.length ∧
    r.types = g.types ∧ r.relMs = g.relMs := by
  simp only [resize]; (repeat' split) <;> simp [resizeNodeMs]

/-- `resize` changes no edge's type or endpoints (the type column lives in `edge_endpoints`,
which `resize` does not touch). -/
theorem resize_keeps_relTypes (chunk : Nat) (hc : 0 < chunk) (g : G V) :
    (resize chunk hc g).relTypes = g.relTypes ∧ (resize chunk hc g).endpoints = g.endpoints := by
  simp only [resize]; (repeat' split) <;> simp [resizeNodeMs]

theorem getLabelIdMut_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : LInv g) :
    let r := getLabelIdMut chunk hc g s
    LInv r.1 ∧ r.1.labels[r.2]? = some s ∧ getLabelId r.1 s = some r.2 := by
  have hi := internLabel_spec g s h.1
  cases hs : g.labelsIndex s with
  | some id =>
    simp only [getLabelIdMut, getLabelId, hs]; refine ⟨h, ?_, trivial⟩
    rw [h.1] at hs; exact idx_get _ _ _ hs
  | none =>
    simp only [internLabel, hs] at hi
    simp only [getLabelIdMut, getLabelId, hs, internLabel]
    obtain ⟨k1, k2, k3, -, -⟩ := resize_keeps chunk hc
      ({ g with labels := g.labels ++ [s],
                labelsIndex := fun t => if t = s then some g.labels.length else g.labelsIndex t,
                labelMs := g.labelMs ++ [Mat.empty g.nodeCap g.nodeCap] } : G V)
    simp only at k1 k2 k3
    simp only [LInv, k1, k2, k3]
    refine ⟨⟨hi.1, ?_⟩, hi.2.1, by simp⟩
    simp [h.2]

def getTypeId (g : G V) (s : String) : Option Nat := idx g.types s

/-- `Vec::insert(i, x)` for `i ≤ len` (it panics otherwise). -/
def vecInsert {α} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop i

theorem vecInsert_end {α} (l : List α) (x : α) : vecInsert l l.length x = l ++ [x] := by
  simp [vecInsert]

/-- `register_relationship_type` (:1366): push the name, insert its tensor, then
`self.resize()` — every registration path goes through it. It goes through
`get_or_insert`: a type already registered gets its id back and nothing else
(`registerRelType_known`). -/
def registerRelType (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) : G V × Nat :=
  match idx g.types s with
  | some p => (g, p)
  | none =>
    let types := g.types ++ [s]
    (resize chunk hc { g with types := types, relMs := vecInsert g.relMs (types.length - 1) [] },
     types.length - 1)

/-- `get_type_id_mut` (:1346): an existing type's position, else
`register_relationship_type`. -/
def getTypeIdMut (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) : G V × Nat :=
  match idx g.types s with
  | some p => (g, p)
  | none => registerRelType chunk hc g s

/-- `get_relationship_matrix_mut` (:1477): a new type goes through
`register_relationship_type` (`getRelMatMut_new`), which resizes. Returns the type index. -/
def getRelMatMut (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) : G V × Nat :=
  let g1 := if s ∈ g.types then g else
    resize chunk hc { g with types := g.types ++ [s],
                             relMs := vecInsert g.relMs (g.types ++ [s]).length.pred [] }
  (g1, (idx g1.types s).getD 0)

def getRelMat (g : G V) (s : String) : Option Ten :=
  if s ∈ g.types then (idx g.types s).bind (g.relMs[·]?) else none

def TInv (g : G V) : Prop := g.relMs.length = g.types.length

theorem getTypeIdMut_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : TInv g) :
    let r := getTypeIdMut chunk hc g s
    TInv r.1 ∧ r.1.types[r.2]? = some s ∧ getTypeId r.1 s = some r.2 := by
  simp only [getTypeIdMut, getTypeId]
  cases hs : idx g.types s with
  | some p => exact ⟨h, idx_get _ _ _ hs, by simpa using hs⟩
  | none =>
    obtain ⟨-, -, -, k4, k5⟩ := resize_keeps chunk hc
      ({ g with types := g.types ++ [s], relMs := vecInsert g.relMs ((g.types ++ [s]).length - 1) [] } : G V)
    simp only at k4 k5
    simp only [registerRelType, hs, k4, k5, TInv] at h ⊢
    simp only [List.length_append, List.length_singleton, Nat.add_sub_cancel]
    rw [← h, vecInsert_end, idx_append, hs]; simp [h]

theorem idx_some (l : List String) (s : String) (h : s ∈ l) : ∃ i, idx l s = some i := by
  cases hh : idx l s with
  | none => exact absurd h ((idx_none l s).1 hh)
  | some i => exact ⟨i, rfl⟩

/-- A *new* type gets its tensor and the next type index (`types.len()` before, which is
what `relationship_type_slot` checked); an already-registered one is returned as is. -/
theorem getRelMatMut_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : TInv g) :
    let r := getRelMatMut chunk hc g s
    TInv r.1 ∧ r.1.types[r.2]? = some s ∧
    (s ∉ g.types → r.2 = g.types.length) := by
  by_cases hm : s ∈ g.types
  · obtain ⟨i, hi⟩ := idx_some _ _ hm
    simp only [getRelMatMut, hm, ite_true, hi, Option.getD_some]
    exact ⟨h, idx_get _ _ _ hi, fun hn => (hn trivial).elim⟩
  · simp only [getRelMatMut, hm, ite_false]
    obtain ⟨-, -, -, k4, k5⟩ := resize_keeps chunk hc
      ({ g with types := g.types ++ [s], relMs := vecInsert g.relMs (g.types ++ [s]).length.pred [] } : G V)
    simp only at k4 k5
    have hn := (idx_none g.types s).2 hm
    simp only [k4, k5, idx_append, hn, ite_true, Option.getD_some, TInv]
    refine ⟨?_, by simp, fun _ => trivial⟩
    simp only [TInv] at h
    simp only [List.length_append, List.length_singleton, Nat.pred_eq_sub_one, Nat.add_sub_cancel]
    rw [← h, vecInsert_end]; simp

/-- `registerRelType` changes no edge's type or endpoints (#3967: there is no type matrix to
widen; the type column is per edge, so a new type needs no room in it). -/
theorem registerRelType_keeps_relTypes (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) :
    let r := (registerRelType chunk hc g s).1
    r.relTypes = g.relTypes ∧ r.endpoints = g.endpoints := by
  simp only [registerRelType]
  split
  · exact ⟨rfl, rfl⟩
  · exact resize_keeps_relTypes chunk hc _

/-- **`getTypeIdMut_keeps_relTypes`**: whatever `get_type_id_mut` registers, every edge keeps
its type, so a later delete still clears it (`type_cleared_after_registration`). -/
theorem getTypeIdMut_keeps_relTypes (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) :
    let r := (getTypeIdMut chunk hc g s).1
    r.relTypes = g.relTypes ∧ r.endpoints = g.endpoints := by
  simp only [getTypeIdMut]
  split
  · exact ⟨rfl, rfl⟩
  · exact registerRelType_keeps_relTypes chunk hc g s

/-- `get_relationship_matrix_mut` registers a new type through `register_relationship_type`. -/
theorem getRelMatMut_new (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (hn : s ∉ g.types) :
    (getRelMatMut chunk hc g s).1 = (registerRelType chunk hc g s).1 := by
  simp [getRelMatMut, registerRelType, hn, (idx_none g.types s).2 hn, Nat.pred_eq_sub_one]

/-- **#3757**: registering a type already registered changes nothing and returns its id,
so its tensor is never displaced. -/
theorem registerRelType_known (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (p : Nat)
    (hp : idx g.types s = some p) :
    registerRelType chunk hc g s = (g, p) := by
  simp [registerRelType, hp]

/-- `registerRelType` keeps `TInv` and returns `s`'s id, new or not. -/
theorem registerRelType_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : TInv g) :
    let r := registerRelType chunk hc g s
    TInv r.1 ∧ r.1.types[r.2]? = some s ∧ getTypeId r.1 s = some r.2 := by
  have := getTypeIdMut_spec chunk hc g s h
  simp only [getTypeIdMut] at this
  cases hs : idx g.types s with
  | some p => rw [registerRelType_known chunk hc g s p hs]; simp only [hs] at this; exact this
  | none => simp only [hs] at this; exact this

def getLabelMat (g : G V) (s : String) : Option Mat := (getLabelId g s).bind (g.labelMs[·]?)

/-- `get_label_matrix_mut` (:1462). Returns the label id. -/
def getLabelMatMut (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) : G V × Nat :=
  let (g1, id) := internLabel g s
  if id = g1.labelMs.length then
    (resize chunk hc { g1 with labelMs := vecInsert g1.labelMs id (Mat.empty g1.nodeCap g1.nodeCap) }, id)
  else (g1, id)

theorem getLabelMatMut_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : LInv g) :
    let r := getLabelMatMut chunk hc g s
    LInv r.1 ∧ getLabelId r.1 s = some r.2 := by
  have hi := internLabel_spec g s h.1
  cases hs : g.labelsIndex s with
  | some id =>
    simp only [internLabel, hs] at hi
    have hlt : id < g.labelMs.length := by
      rw [h.2]; have := hi.2.1; exact (List.getElem?_eq_some_iff.1 this).1
    simp only [getLabelMatMut, internLabel, hs, Nat.ne_of_lt hlt, ite_false]
    exact ⟨h, hi.2.2⟩
  | none =>
    simp only [internLabel, hs] at hi
    have he : g.labels.length = g.labelMs.length := h.2.symm
    simp only [getLabelMatMut, internLabel, hs, he, ite_true, vecInsert_end]
    obtain ⟨k1, k2, k3, -, -⟩ := resize_keeps chunk hc
      ({ g with labels := g.labels ++ [s],
                labelsIndex := fun t => if t = s then some g.labelMs.length else g.labelsIndex t,
                labelMs := g.labelMs ++ [Mat.empty g.nodeCap g.nodeCap] } : G V)
    simp only at k1 k2 k3
    simp only [LInv, getLabelId, k1, k2, k3]
    rw [he] at hi
    exact ⟨⟨hi.1, by simp [h.2]⟩, by simp⟩

def nodeHasLabelId (g : G V) (n l : Nat) : Bool := g.nodeLabels.get n l
def nodeHasLabel (g : G V) (n : Nat) (s : String) : Bool :=
  match getLabelId g s with | some l => g.nodeLabels.get n l | none => false
/-- `edge_has_type_id` (`graph.rs`): the type column's entry for the edge is `type_id`
(`get_type(e).is_some_and(|t| t == type_id)`), no string lookup. -/
def edgeHasTypeId (g : G V) (e t : Nat) : Bool := g.relTypes e == some t
/-- `edge_has_type` (`graph.rs`): the type's id by name, then `edge_has_type_id`. -/
def edgeHasType (g : G V) (e : Nat) (s : String) : Bool :=
  match getTypeId g s with | some t => edgeHasTypeId g e t | none => false

theorem edgeHasTypeId_iff (g : G V) (e t : Nat) : edgeHasTypeId g e t = true ↔ g.relTypes e = some t := by
  simp [edgeHasTypeId]

theorem nodeHasLabel_iff (g : G V) (n : Nat) (s : String) :
    nodeHasLabel g n s = true ↔ ∃ l, getLabelId g s = some l ∧ nodeHasLabelId g n l = true := by
  unfold nodeHasLabel nodeHasLabelId; cases getLabelId g s <;> simp

theorem edgeHasType_iff (g : G V) (e : Nat) (s : String) :
    edgeHasType g e s = true ↔ ∃ t, g.types[t]? = some s ∧ getTypeId g s = some t ∧
      g.relTypes e = some t := by
  unfold edgeHasType; cases ht : getTypeId g s with
  | none => simp
  | some t =>
    simp only [edgeHasTypeId_iff]
    exact ⟨fun h => ⟨t, idx_get _ _ _ ht, rfl, h⟩, fun ⟨_, _, h1, h2⟩ => by cases h1; exact h2⟩

def labelNodeCount (g : G V) (s : String) : Nat := ((getLabelMat g s).map Mat.nvals).getD 0
/-- `label_node_count_by_idx` / `type_edge_count`: `None` = index panic. -/
def labelNodeCountByIdx (g : G V) (i : Nat) : Option Nat := g.labelMs[i]?.map Mat.nvals
def typeEdgeCount (g : G V) (i : Nat) : Option Nat := g.relMs[i]?.map Ten.edgeCount

theorem labelNodeCount_eq (g : G V) (s : String) (i : Nat) (h : getLabelId g s = some i) :
    labelNodeCount g s = (labelNodeCountByIdx g i).getD 0 := by
  simp [labelNodeCount, labelNodeCountByIdx, getLabelMat, h]

theorem labelNodeCount_unknown (g : G V) (s : String) (h : getLabelId g s = none) :
    labelNodeCount g s = 0 := by simp [labelNodeCount, getLabelMat, h]

/-! ## The relationship type limit (#3967)

The endpoint index stores each edge's type as a `u16`, `u16::MAX` its empty marker, so
type indices `0..=MAX_TYPE_ID = 65534` can carry edges. Every edge-creating path checks the
type's slot before it writes anything. -/

/-- `MAX_TYPE_ID` (endpoint_index.rs). -/
def maxTypeId : Nat := 65534

/-- `edge_type_slot` (graph.rs): `u16::try_from(type_idx).ok().filter(|t| t <= MAX_TYPE_ID)`. -/
def edgeTypeSlot (i : Nat) : Option Nat := if i ≤ maxTypeId then some i else none

theorem edgeTypeSlot_spec (i t : Nat) : edgeTypeSlot i = some t ↔ t = i ∧ i ≤ maxTypeId := by
  unfold edgeTypeSlot; split <;> simp_all [eq_comm]
  intro h; subst h; assumption

/-- `relationship_type_slot` (graph.rs): the type's id, or `types.len()` (the id it would be
registered with); `none` = `Err(TooManyRelationshipTypes)`. -/
def relTypeSlot (g : G V) (s : String) : Option Nat :=
  edgeTypeSlot ((getTypeId g s).getD g.types.length)

/-- **The slot is the id the type has or will get**: when `relationship_type_slot` accepts,
`get_relationship_matrix_mut` returns that very index (`create_relationships_bulk`'s
`debug_assert_eq!(edge_type_slot(type_idx), Some(type_slot))`). -/
theorem relTypeSlot_getRelMatMut (chunk : Nat) (hc : 0 < chunk) (g : G V) (s : String) (h : TInv g)
    (t : Nat) (ht : relTypeSlot g s = some t) : (getRelMatMut chunk hc g s).2 = t := by
  unfold relTypeSlot at ht
  rw [edgeTypeSlot_spec] at ht
  obtain ⟨rfl, -⟩ := ht
  by_cases hm : s ∈ g.types
  · obtain ⟨i, hi⟩ := idx_some _ _ hm
    simp [getRelMatMut, hm, hi, getTypeId]
  · rw [(getRelMatMut_spec chunk hc g s h).2.2 hm]
    simp [getTypeId, (idx_none g.types s).2 hm]

/-- A type past the limit is refused, registered or not. -/
theorem relTypeSlot_refused (g : G V) (s : String) (h : maxTypeId < (getTypeId g s).getD g.types.length) :
    relTypeSlot g s = none := by
  unfold relTypeSlot edgeTypeSlot; rw [ite_eq_right (by omega)]

/-- `relationship_types_fit` (graph.rs): walk the names; a registered one is checked at its
id, a new one at `next` (then `next += 1`), a repeated new one skipped (`new` is an
`FxHashSet`; only membership is read, so the list `new` models it exactly). `none` = `Ok(())`,
`some n` = `Err(TooManyRelationshipTypes { name: n, .. })`. -/
def relTypesFitAux (g : G V) : List String → Nat → List String → Option String
  | [], _, _ => none
  | n :: ns, next, new =>
    match getTypeId g n with
    | some t => if edgeTypeSlot t = none then some n else relTypesFitAux g ns next new
    | none =>
      if n ∈ new then relTypesFitAux g ns next new
      else if edgeTypeSlot next = none then some n else relTypesFitAux g ns (next + 1) (new ++ [n])

def relTypesFit (g : G V) (names : List String) : Option String :=
  relTypesFitAux g names g.types.length []

/-- For one name the group check is `relationship_type_slot`. -/
theorem relTypesFit_single (g : G V) (n : String) :
    relTypesFit g [n] = none ↔ (relTypeSlot g n).isSome := by
  unfold relTypesFit relTypeSlot
  simp only [relTypesFitAux]
  cases h : getTypeId g n with
  | some t => simp only [Option.getD_some]; cases edgeTypeSlot t <;> simp
  | none => simp only [Option.getD_none, List.not_mem_nil, ite_false]; cases edgeTypeSlot g.types.length <;> simp

theorem relTypesFitAux_known (g : G V) : ∀ (ns : List String) (next : Nat) (new : List String),
    relTypesFitAux g ns next new = none → ∀ n ∈ ns, ∀ t, getTypeId g n = some t → t ≤ maxTypeId := by
  intro ns
  induction ns with
  | nil => intro _ _ _ n hn; cases hn
  | cons m ms ih =>
    intro next new h n hn t ht
    simp only [relTypesFitAux] at h
    cases hm : getTypeId g m with
    | some u =>
      rw [hm] at h; simp only at h
      split at h
      · cases h
      · rename_i hfit
        rcases List.mem_cons.1 hn with rfl | hn
        · rw [hm] at ht; cases ht
          unfold edgeTypeSlot at hfit; split at hfit <;> simp_all
        · exact ih _ _ h n hn t ht
    | none =>
      rw [hm] at h; simp only at h
      rcases List.mem_cons.1 hn with rfl | hn
      · rw [hm] at ht; cases ht
      · split at h
        · exact ih _ _ h n hn t ht
        · split at h
          · cases h
          · exact ih _ _ h n hn t ht

/-- **Accepted ⇒ every registered name fits.** -/
theorem relTypesFit_known (g : G V) (names : List String) (h : relTypesFit g names = none)
    (n : String) (hn : n ∈ names) (t : Nat) (ht : getTypeId g n = some t) : t ≤ maxTypeId :=
  relTypesFitAux_known g names _ _ h n hn t ht

theorem relTypesFitAux_full (g : G V) : ∀ (ns : List String) (next : Nat) (new : List String),
    maxTypeId < next → ∀ n ∈ ns, getTypeId g n = none → n ∉ new → relTypesFitAux g ns next new ≠ none := by
  intro ns
  induction ns with
  | nil => intro _ _ _ n hn; cases hn
  | cons m ms ih =>
    intro next new hfull n hn hnone hnew
    simp only [relTypesFitAux]
    cases hm : getTypeId g m with
    | some u =>
      simp only
      split
      · simp
      · rcases List.mem_cons.1 hn with rfl | hn
        · rw [hm] at hnone; cases hnone
        · exact ih _ _ hfull n hn hnone hnew
    | none =>
      simp only
      split
      · rename_i hmn
        rcases List.mem_cons.1 hn with rfl | hn
        · exact absurd hmn hnew
        · exact ih _ _ hfull n hn hnone hnew
      · rw [ite_eq_left_iff.2 (fun h => (h (by unfold edgeTypeSlot; rw [ite_eq_right (by omega)])).elim)]; simp

/-- **No room ⇒ any new name is refused**: with `MAX_TYPE_ID + 1` types already registered,
a group that would register one more is refused (before any write). -/
theorem relTypesFit_full (g : G V) (names : List String) (hfull : maxTypeId < g.types.length)
    (n : String) (hn : n ∈ names) (hnone : getTypeId g n = none) : relTypesFit g names ≠ none :=
  relTypesFitAux_full g names _ [] hfull n hn hnone (by simp)

end GQ
