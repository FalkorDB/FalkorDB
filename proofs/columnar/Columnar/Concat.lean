import Columnar.Row

/-
# `Batch::concat` against its row-at-a-time definition (`BatchBuilder`)

`concat` documents that it is "byte-for-byte the same result the per-row concat
produced" (`batch.rs:965-977`). The per-row concat is: push every active row of
every batch, in order, through a `BatchBuilder` (`push_row` of
`BatchRow::to_owned_row`) and `finish`. We model both per column slot and
compare what a consumer can observe at every (row, slot): the `value_at`
answer and the `is_bound_at` bit.

| here | there |
| --- | --- |
| `builderCol` | `BatchBuilder::push_row_with` (`batch.rs:577-651`, empty `extra`) + `finish` (`:750-796`), one slot |
| `rowVals` | the fallback loop `for r in b.active_indices() { col.get(r) }` (`batch.rs:1008-1021`) |
| `concatGeneric` | the fallback branch of `Batch::concat` (`batch.rs:994-1035`) |
| `typedScan`, `concatTyped` | `Batch::concat_typed_column` (`batch.rs:1064-1151`) |
| `concatCol` | one column of `Batch::concat` (`batch.rs:989-993`) |
| `concatOrigins` | the origin loop (`batch.rs:1039-1047`) |

Result: equal whenever every input batch has an active row
(`concat_obs_eq`); a batch with no active rows that binds a slot makes
`concat` bind it where the per-row definition leaves it unbound
(`empty_batch_binds`) — test `latent_concat_empty_batch_binds_column`.
-/

namespace Columnar

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

theorem flatMap_congr' {α β : Type} {f g : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = g a) → l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.flatMap_cons]
    rw [h a (by simp), flatMap_congr' (fun x hx => h x (by simp [hx]))]

theorem any_const_of_ne_nil {α : Type} {l : List α} (h : l ≠ []) (c : Bool) :
    l.any (fun _ => c) = c := by
  cases l with
  | nil => exact absurd rfl h
  | cons a l => cases c <;> simp

theorem any_congr' {α : Type} {p q : α → Bool} :
    ∀ {l : List α}, (∀ a ∈ l, p a = q a) → l.any p = l.any q
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.any_cons]
    rw [h a (by simp), any_congr' (fun x hx => h x (by simp [hx]))]

theorem fillRow_origin (f : Nat → Option (V F × Bool)) (acc : Row F) :
    ∀ n, (fillRow n f acc).origin = acc.origin
  | 0 => rfl
  | n + 1 => by
    rw [fillRow_succ']
    cases f n with
    | none => exact fillRow_origin f acc n
    | some p =>
      obtain ⟨v, keep⟩ := p
      cases keep <;> exact fillRow_origin f acc n

theorem any_or' {α : Type} (l : List α) (p q : α → Bool) :
    l.any (fun a => p a || q a) = (l.any p || l.any q) := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.any_cons, ih]; cases p a <;> cases q a <;> simp


/-- Observation of a finished column slot: (`value_at`, `is_bound_at`). -/
def colObs (c : Column F × Bool) (k : Nat) : Option (V F) × Bool :=
  (if c.1.isUnbound then none else c.1.get? k, !c.1.isUnbound && !c.2)

/-- One slot of `BatchBuilder` after pushing `rows` and `finish` (`vo` = the
value-only bit). -/
def builderCol (rows : List (Row F)) (i : Nat) : Column F × Bool :=
  let vals := rows.map (fun r => (r.get i).getD .null)
  let anyBound := rows.any (fun r => (r.get i).isSome && r.bound i)
  let present := rows.any (fun r => match r.get i with
    | some v => r.bound i || !v.isNull
    | none => false)
  if present then (if anyBound then (classifyStored vals, false) else (.values vals, true))
  else (.unbound, false)

/-- The per-row concat's rows. -/
def perRowRows (bs : List (Batch F)) : List (Row F) :=
  bs.flatMap (fun b => b.active.map (toOwned b))

def rowVals (b : Batch F) (i : Nat) : List (V F) :=
  b.active.map (fun r => ((b.column i).get? r).getD .null)

/-- `concat`'s fallback column. -/
def concatGeneric (bs : List (Batch F)) (i : Nat) : Column F × Bool :=
  let vals := bs.flatMap (rowVals · i)
  let boundAny := bs.any (fun b => !(b.column i).isUnbound && !b.vo i)
  let nonnull := vals.any (fun v => !v.isNull)
  if boundAny then (classifyStored vals, false)
  else if nonnull then (.values vals, true) else (.unbound, false)

/-- `Kind` of a primitive column (`batch.rs:1070-1075`). -/
def kindOf : Column F → Option Nat
  | .nodeIds _ => some 0
  | .relIds _ => some 1
  | .ints _ => some 2
  | .floats _ => some 3
  | _ => none

/-- The guard loop of `concat_typed_column` (`batch.rs:1081-1107`); outer
`none` = an early `return None`. -/
def typedScan (i : Nat) : List (Batch F) → Option Nat → Option (Option Nat)
  | [], kind => some kind
  | b :: bs, kind =>
    if b.activeLen = 0 then typedScan i bs kind
    else if b.vo i then none
    else match kindOf (b.column i) with
      | none => none
      | some k => match kind with
        | none => typedScan i bs (some k)
        | some p => if p = k then typedScan i bs kind else none

/-- The bulk copy of `concat_typed_column` (`batch.rs:1113-1150`). -/
def buildTyped (bs : List (Batch F)) (i : Nat) : Nat → Column F
  | 0 => .nodeIds (bs.flatMap (fun b => match b.column i with
      | .nodeIds v => (extendActive v b.sel).getD [] | _ => []))
  | 1 => .relIds (bs.flatMap (fun b => match b.column i with
      | .relIds v => (extendActive v b.sel).getD [] | _ => []))
  | 2 => .ints (bs.flatMap (fun b => match b.column i with
      | .ints v => (extendActive v b.sel).getD [] | _ => []))
  | _ => .floats (bs.flatMap (fun b => match b.column i with
      | .floats v => (extendActive v b.sel).getD [] | _ => []))

def concatTyped (bs : List (Batch F)) (i : Nat) : Option (Column F) :=
  match typedScan i bs none with
  | some (some k) => some (buildTyped bs i k)
  | _ => none

/-- One column of `Batch::concat`. -/
def concatCol (bs : List (Batch F)) (i : Nat) : Column F × Bool :=
  match concatTyped bs i with
  | some c => (c, false)
  | none => concatGeneric bs i

/-- `concat`'s origin sidecar, read at row `k` (`0` when elided). -/
def concatOrigins (bs : List (Batch F)) : List Nat :=
  bs.flatMap (fun b => b.active.map b.originRow)

/-! ## The typed path copies exactly the fallback's values -/

theorem typedScan_spec (i : Nat) :
    ∀ (bs : List (Batch F)) (kind : Option Nat) (k : Nat), typedScan i bs kind = some (some k) →
      (kind = none ∨ kind = some k) ∧
      (kind = none → ∃ b ∈ bs, 0 < b.activeLen) ∧
      ∀ b ∈ bs, 0 < b.activeLen → b.vo i = false ∧ kindOf (b.column i) = some k
  | [], kind, k, h => by
    simp [typedScan] at h; subst h; simp
  | b :: bs, kind, k, h => by
    unfold typedScan at h
    by_cases h0 : b.activeLen = 0
    · rw [if_pos h0] at h
      obtain ⟨h1, h2, h3⟩ := typedScan_spec i bs kind k h
      refine ⟨h1, fun hk => ?_, ?_⟩
      · obtain ⟨b', hb', hp⟩ := h2 hk; exact ⟨b', by simp [hb'], hp⟩
      · intro b' hb' hp
        rcases List.mem_cons.mp hb' with rfl | hb'
        · omega
        · exact h3 b' hb' hp
    · rw [if_neg h0] at h
      by_cases hv : b.vo i = true
      · rw [if_pos hv] at h; simp at h
      · rw [if_neg hv] at h
        cases hkb : kindOf (b.column i) with
        | none => rw [hkb] at h; simp at h
        | some k0 =>
          rw [hkb] at h
          cases kind with
          | none =>
            simp only at h
            obtain ⟨h1, -, h3⟩ := typedScan_spec i bs (some k0) k h
            have hk0 : k0 = k := by rcases h1 with h1 | h1 <;> simp_all
            subst hk0
            refine ⟨Or.inl rfl, fun _ => ⟨b, by simp, by omega⟩, ?_⟩
            intro b' hb' hp
            rcases List.mem_cons.mp hb' with rfl | hb'
            · exact ⟨by simpa using hv, hkb⟩
            · exact h3 b' hb' hp
          | some p =>
            simp only at h
            by_cases hp : p = k0
            · rw [if_pos hp] at h
              obtain ⟨h1, -, h3⟩ := typedScan_spec i bs (some p) k h
              have hpk : p = k := by rcases h1 with h1 | h1 <;> simp_all
              subst hpk; subst hp
              refine ⟨Or.inr rfl, fun h => by simp at h, ?_⟩
              intro b' hb' hq
              rcases List.mem_cons.mp hb' with rfl | hb'
              · exact ⟨by simpa using hv, hkb⟩
              · exact h3 b' hb' hq
            · rw [if_neg hp] at h; simp at h

/-- A typed lane's active slice, embedded as values, is the fallback's `rowVals`. -/
theorem extendActive_rowVals {α : Type} {b : Batch F} (hw : b.WF) {i : Nat} {v : List α}
    (emb : α → V F) (hcol : ∀ r, (b.column i).get? r = (v[r]?).map emb)
    (hlen : v.length = b.len) :
    ((extendActive v b.sel).getD []).map emb = rowVals b i := by
  have e1 : extendActive v b.sel = gatherL v b.active := by
    rw [extendActive_eq_gather, hlen]; rfl
  have hs : (gatherL v b.active).isSome :=
    gatherL_isSome_iff.mpr (fun r hr => hlen ▸ Batch.active_lt hw r hr)
  obtain ⟨w, hwv⟩ := Option.isSome_iff_exists.mp hs
  rw [e1, hwv, Option.getD_some]
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map, gatherL_getElem? hwv k, rowVals, List.getElem?_map]
  cases hk : b.active[k]? with
  | none => rfl
  | some r =>
    simp only [Option.bind_some, Option.map_some]
    have hr : r < v.length := hlen ▸ Batch.active_lt hw r (List.mem_of_getElem? hk)
    rw [hcol r, List.getElem?_eq_getElem hr]; rfl

theorem rowVals_empty {b : Batch F} (h : b.activeLen = 0) (i : Nat) : rowVals b i = [] := by
  simp [rowVals, Batch.activeLen] at h ⊢; exact h

theorem extendActive_empty {α : Type} {b : Batch F} (hw : b.WF) (h : b.activeLen = 0) {v : List α}
    (hlen : v.length = b.len) : (extendActive v b.sel).getD [] = [] := by
  have e1 : extendActive v b.sel = gatherL v b.active := by
    rw [extendActive_eq_gather, hlen]; rfl
  have ha : b.active = [] := by simpa [Batch.activeLen] using h
  rw [e1, ha]; rfl

/-- One batch's contribution to the typed copy, embedded as values. -/
def laneOf (b : Batch F) (i : Nat) : Nat → List (V F)
  | 0 => (match b.column i with | .nodeIds v => (extendActive v b.sel).getD [] | _ => []).map V.node
  | 1 => (match b.column i with | .relIds v => (extendActive v b.sel).getD [] | _ => []).map V.rel
  | 2 => (match b.column i with | .ints v => (extendActive v b.sel).getD [] | _ => []).map V.int
  | _ => (match b.column i with | .floats v => (extendActive v b.sel).getD [] | _ => []).map V.float

/-- Per batch, the typed copy contributes exactly `rowVals`. -/
theorem laneOf_eq {b : Batch F} (hw : b.WF) {i k : Nat}
    (hk : 0 < b.activeLen → kindOf (b.column i) = some k) : laneOf b i k = rowVals b i := by
  have hlen : ∀ c, b.column i = c → c.isUnbound = false → c.len = b.len := by
    intro c e hu; rw [← e]; exact Batch.column_len hw (by rw [e]; exact hu)
  by_cases h0 : b.activeLen = 0
  · rw [rowVals_empty h0]
    rcases k with _ | _ | _ | k <;> unfold laneOf <;> cases hc : b.column i <;>
      simp only [List.map_eq_nil_iff] <;>
      first
      | rfl
      | (apply extendActive_empty hw h0
         simpa [Column.len] using hlen _ hc rfl)
  · have hk' := hk (by omega)
    rcases k with _ | _ | _ | k <;> unfold laneOf <;> cases hc : b.column i <;>
      rw [hc] at hk' <;> simp [kindOf] at hk' <;>
      (simp only
       apply extendActive_rowVals hw _ (by intro r; rw [hc]; rfl)
       simpa [Column.len] using hlen _ hc rfl)

theorem buildTyped_toValues {bs : List (Batch F)} (hw : ∀ b ∈ bs, b.WF) {i k : Nat}
    (hk : ∀ b ∈ bs, 0 < b.activeLen → kindOf (b.column i) = some k) :
    (buildTyped bs i k).toValues = some (bs.flatMap (rowVals · i)) := by
  have hb := fun b (hbm : b ∈ bs) => laneOf_eq (hw b hbm) (hk b hbm)
  have e : (buildTyped bs i k).toValues = some (bs.flatMap (laneOf · i k)) := by
    rcases k with _ | _ | _ | k <;> simp [buildTyped, Column.toValues, List.map_flatMap, laneOf]
  rw [e, flatMap_congr' (fun b hbm => hb b hbm)]

/-- **The typed fast path is observationally the fallback**: same values at
every row, bound. -/
theorem concatCol_obs_generic {bs : List (Batch F)} (hw : ∀ b ∈ bs, b.WF) (i k : Nat) :
    colObs (concatCol bs i) k = colObs (concatGeneric bs i) k := by
  unfold concatCol concatTyped
  cases hs : typedScan i bs none with
  | none => rfl
  | some kind =>
    cases kind with
    | none => rfl
    | some kd =>
      simp only
      obtain ⟨-, hex, hall⟩ := typedScan_spec i bs none kd hs
      have hv := buildTyped_toValues hw (fun b hb hp => (hall b hb hp).2)
      obtain ⟨b0, hb0, hp0⟩ := hex rfl
      have hbound : bs.any (fun b => !(b.column i).isUnbound && !b.vo i) = true := by
        rw [List.any_eq_true]
        refine ⟨b0, hb0, ?_⟩
        obtain ⟨hvo, hkd⟩ := hall b0 hb0 hp0
        have : (b0.column i).isUnbound = false := by
          cases hc : b0.column i <;> rw [hc] at hkd <;> simp [kindOf] at hkd <;> rfl
        simp [this, hvo]
      have hbu : (buildTyped bs i kd).isUnbound = false := by
        rcases kd with _ | _ | _ | kd <;> rfl
      simp only [colObs, concatGeneric, hbound, if_true, hbu, classifyStored_isUnbound,
        Column.get?_eq_toValues hv, classifyStored_get?]

/-! ## The fallback is the per-row builder when no batch is empty -/

/-- What the owned snapshot of an active row holds in slot `i`. -/
theorem toOwned_slot {b : Batch F} (hw : b.WF) {r : Nat} (hr : r ∈ b.active) (i : Nat) :
    ((toOwned b r).get i).getD .null = ((b.column i).get? r).getD .null ∧
    (((toOwned b r).get i).isSome && (toOwned b r).bound i) = (!(b.column i).isUnbound && !b.vo i) ∧
    (match (toOwned b r).get i with
      | some v => (toOwned b r).bound i || !v.isNull
      | none => false) =
      ((!(b.column i).isUnbound && !b.vo i) || !(((b.column i).get? r).getD .null).isNull) := by
  cases hu : (b.column i).isUnbound
  · obtain ⟨g, bnd⟩ := toOwned_bound_slot hw hr hu
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (Batch.valueAt_isSome hw hr hu)
    have hget : (b.column i).get? r = some v := by simpa [Batch.valueAt, hu] using hv
    have hg : (toOwned b r).get i = some v := by rw [g]; simp [viewAt, hv]
    rw [hg, bnd, hget]
    simp
  · obtain ⟨bnd, g⟩ := toOwned_unbound_slot b r i hu
    have hnull : (b.column i).get? r = some .null := by
      cases hc : b.column i <;> rw [hc] at hu <;> simp [Column.isUnbound] at hu; rfl
    rw [hnull, bnd]
    rcases g with g | g <;> rw [g] <;> simp [V.isNull]

theorem perRow_vals (bs : List (Batch F)) (hw : ∀ b ∈ bs, b.WF) (i : Nat) :
    (perRowRows bs).map (fun r => (r.get i).getD .null) = bs.flatMap (rowVals · i) := by
  simp only [perRowRows, List.map_flatMap, List.map_map]
  apply flatMap_congr'
  intro b hb
  apply List.map_congr_left
  intro r hr
  exact (toOwned_slot (hw b hb) hr i).1

theorem perRow_anyBound (bs : List (Batch F)) (hw : ∀ b ∈ bs, b.WF) (hne : ∀ b ∈ bs, 0 < b.activeLen)
    (i : Nat) :
    (perRowRows bs).any (fun r => (r.get i).isSome && r.bound i) =
      bs.any (fun b => !(b.column i).isUnbound && !b.vo i) := by
  simp only [perRowRows, List.any_flatMap, List.any_map]
  apply any_congr'
  intro b hb
  have hne' : b.active ≠ [] := by
    have := hne b hb; intro e; simp [Batch.activeLen, e] at this
  rw [← any_const_of_ne_nil hne' (!(b.column i).isUnbound && !b.vo i)]
  apply any_congr'
  intro r hr
  exact (toOwned_slot (hw b hb) hr i).2.1

theorem perRow_present (bs : List (Batch F)) (hw : ∀ b ∈ bs, b.WF) (hne : ∀ b ∈ bs, 0 < b.activeLen)
    (i : Nat) :
    (perRowRows bs).any (fun r => match r.get i with
      | some v => r.bound i || !v.isNull
      | none => false) =
      (bs.any (fun b => !(b.column i).isUnbound && !b.vo i) ||
        (bs.flatMap (rowVals · i)).any (fun v => !v.isNull)) := by
  simp only [perRowRows, List.any_flatMap, List.any_map, rowVals]
  rw [← any_or']
  apply any_congr'
  intro b hb
  have hne' : b.active ≠ [] := by
    have := hne b hb; intro e; simp [Batch.activeLen, e] at this
  rw [← any_const_of_ne_nil hne' (!(b.column i).isUnbound && !b.vo i), ← any_or']
  apply any_congr'
  intro r hr
  exact (toOwned_slot (hw b hb) hr i).2.2

/-- **`concat` = per-row concat** (every slot, every row: value and bound bit),
provided every input batch has at least one active row. -/
theorem concat_obs_eq {bs : List (Batch F)} (hw : ∀ b ∈ bs, b.WF) (hne : ∀ b ∈ bs, 0 < b.activeLen)
    (i k : Nat) : colObs (concatCol bs i) k = colObs (builderCol (perRowRows bs) i) k := by
  rw [concatCol_obs_generic hw]
  unfold concatGeneric builderCol
  simp only
  rw [perRow_vals bs hw i, perRow_anyBound bs hw hne i, perRow_present bs hw hne i]
  cases bs.any (fun b => !(b.column i).isUnbound && !b.vo i) <;> simp

/-- Row count: both have `Σ active_len` rows. -/
theorem concat_len (bs : List (Batch F)) :
    (perRowRows bs).length = (bs.map Batch.activeLen).sum := by
  simp [perRowRows, List.length_flatMap]; rfl

/-- The origin sidecar concatenates each batch's active origins in order —
exactly the per-row builder's `origins` (`row.origin_row` of each owned row). -/
theorem concat_origins (bs : List (Batch F)) :
    concatOrigins bs = (perRowRows bs).map (·.origin) := by
  simp only [concatOrigins, perRowRows, List.map_flatMap, List.map_map]
  apply flatMap_congr'
  intro b _
  apply List.map_congr_left
  intro r _
  obtain ⟨o, ho⟩ := toOwned_row b r
  simp only [Function.comp, toOwned, Batch.originRow]
  cases b.origins with
  | some o => rfl
  | none => simp only; rw [fillRow_origin]; rfl

/-- **Empty batches break the equality**: a batch whose rows are all filtered
out but which binds slot 1 makes `concat` report slot 1 bound (reading
`Some(Null)`), where the per-row concat leaves it unbound (`None`). -/
theorem empty_batch_binds (s : V F) :
    let a : Batch F := { len := 1, sel := none, cols := [.ints [1]], origins := none,
                         vo := fun _ => false }
    let b : Batch F := { len := 1, sel := some [], cols := [.ints [2], .values [s]],
                         origins := none, vo := fun _ => false }
    (colObs (concatCol [a, b] 1) 0).2 = true ∧ (colObs (builderCol (perRowRows [a, b]) 1) 0).2 = false := by
  constructor
  · rfl
  · rfl

end Columnar
