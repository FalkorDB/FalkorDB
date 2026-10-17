import Columnar.Column

/-
# `Batch`: selection vectors, gather, compaction, set/write column

| here | there |
| --- | --- |
| `Batch` | `Batch` (`runtime/batch.rs:805-826`); `vo` = `value_only` BitSet as a predicate |
| `Batch.column` | `Batch::column` (`batch.rs:1213-1223`) |
| `Batch.active` | `Batch::active_indices` / `ActiveIndices::next` (`batch.rs:1204`, `:1458-1479`) |
| `Batch.activeLen` | `Batch::active_len` (`batch.rs:1178`) |
| `Batch.isBound` | `Batch::is_bound_at` (`batch.rs:1336-1343`) — column-level |
| `Batch.valueAt` | `Batch::value_at` (`batch.rs:1264-1273`) |
| `Batch.originRow` | `Batch::origin_row` (`batch.rs:1298-1303`) |
| `Batch.gather` | `Batch::gather` (`batch.rs:897-918`); `none` = panic |
| `Batch.compact` | `Batch::into_compacted` (`batch.rs:884-892`) |
| `Batch.setSel` | `Batch::set_selection` (`batch.rs:1195-1200`) |
| `filterSel`, `takeSel`, `dropSel` | selection producers: `filter.rs:69-112`, `limit.rs:78-84`, `skip.rs:72-78`, `runtime.rs:529-534` |
| `toU16` | the `as u16` cast every producer applies |
| `Batch.setColumn` | `Batch::set_column` (`batch.rs:1226-1256`) |
| `Batch.writeColumn`, `scatter` | `Batch::write_column` (`batch.rs:1347-1367`) |
| `Batch.WF` | the invariants the code relies on (documented at `batch.rs:806-818`, debug-asserted at `:1241`, `:1323`) |
-/

namespace Columnar

variable {F : Type}

/-- `Option`-monad `map` (panic = `none`); `Vec::iter().map(..).collect()` of a fallible step. -/
def mapO {α β : Type} (f : α → Option β) : List α → Option (List β)
  | [] => some []
  | a :: l =>
    match f a with
    | none => none
    | some b => (mapO f l).map (b :: ·)

theorem mapO_getElem? {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, mapO f l = some l' → ∀ i : Nat, l'[i]? = (l[i]?).bind f
  | [], l', h, i => by simp [mapO] at h; subst h; simp
  | a :: l, l', h, i => by
    simp only [mapO] at h
    split at h
    · simp at h
    · rename_i b hb
      obtain ⟨w, hw, rfl⟩ := Option.map_eq_some_iff.mp h
      cases i with
      | zero => simp [hb]
      | succ i => simp [mapO_getElem? hw i]

theorem mapO_length {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, mapO f l = some l' → l'.length = l.length
  | [], l', h => by simp [mapO] at h; subst h; rfl
  | a :: l, l', h => by
    simp only [mapO] at h
    split at h
    · simp at h
    · obtain ⟨w, hw, rfl⟩ := Option.map_eq_some_iff.mp h
      simp [mapO_length hw]

theorem mapO_isSome {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).isSome) → (mapO f l).isSome
  | [], _ => by simp [mapO]
  | a :: l, h => by
    simp only [mapO]
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (h a (by simp))
    rw [hb]
    simpa using mapO_isSome (l := l) (fun x hx => h x (by simp [hx]))

structure Batch (F : Type) where
  len : Nat
  sel : Option (List Nat)
  cols : List (Column F)
  origins : Option (List Nat)
  vo : Nat → Bool

namespace Batch

def column (b : Batch F) (i : Nat) : Column F := b.cols[i]?.getD .unbound

def active (b : Batch F) : List Nat := b.sel.getD (List.range b.len)

def activeLen (b : Batch F) : Nat := b.active.length

def isBound (b : Batch F) (i : Nat) : Bool := !(b.column i).isUnbound && !(b.vo i)

/-- `Batch::value_at`: `none` = unbound (or an index panic, excluded by `WF`). -/
def valueAt (b : Batch F) (i r : Nat) : Option (V F) :=
  if (b.column i).isUnbound then none else (b.column i).get? r

/-- What a row-level consumer can observe of slot `i` at row `r`. -/
def obs (b : Batch F) (i r : Nat) : Option (V F) × Bool := (b.valueAt i r, b.isBound i)

def originRow (b : Batch F) (r : Nat) : Nat :=
  match b.origins with
  | none => 0
  | some o => o[r]?.getD 0

/-- Invariants: every bound column has `len` rows; the selection lists
in-range, strictly increasing (sorted, deduplicated — `batch.rs:808`) rows;
the origin sidecar has `len` entries. -/
structure WF (b : Batch F) : Prop where
  cols_len : ∀ c ∈ b.cols, c.isUnbound = false → c.len = b.len
  sel_lt : ∀ s, b.sel = some s → ∀ i ∈ s, i < b.len
  sel_sorted : ∀ s, b.sel = some s → s.Pairwise (· < ·)
  origins_len : ∀ o, b.origins = some o → o.length = b.len

theorem column_len {b : Batch F} (h : b.WF) {i : Nat} (hu : (b.column i).isUnbound = false) :
    (b.column i).len = b.len := by
  unfold column at hu ⊢
  cases hc : b.cols[i]? with
  | none => simp [hc, Column.isUnbound] at hu
  | some c =>
    simp only [hc, Option.getD_some] at hu ⊢
    exact h.cols_len c (List.mem_of_getElem? hc) hu

/-- **Selection indices are physical rows `< len`.** -/
theorem active_lt {b : Batch F} (h : b.WF) : ∀ i ∈ b.active, i < b.len := by
  unfold active
  cases hs : b.sel with
  | none => simp
  | some s => simpa using h.sel_lt s hs

theorem active_sorted {b : Batch F} (h : b.WF) : b.active.Pairwise (· < ·) := by
  unfold active
  cases hs : b.sel with
  | none => simpa using List.pairwise_lt_range
  | some s => simpa using h.sel_sorted s hs

/-- **No read of an active row of a bound column panics.** -/
theorem valueAt_isSome {b : Batch F} (h : b.WF) {i r : Nat} (hr : r ∈ b.active)
    (hb : (b.column i).isUnbound = false) : (b.valueAt i r).isSome := by
  unfold valueAt
  rw [hb]
  exact Column.get?_isSome hb (by rw [column_len h hb]; exact active_lt h r hr)

/-! ## gather / into_compacted -/

/-- `Batch::gather`. -/
def gather (b : Batch F) (idx : List Nat) : Option (Batch F) := do
  let cols ← mapO (fun c => c.gather idx) b.cols
  let origins ← match b.origins with
    | none => some none
    | some o => (gatherL o idx).map (fun os => if os.any (· ≠ 0) then some os else none)
  pure { len := idx.length, sel := none, cols := cols, origins := origins, vo := b.vo }

/-- `Batch::into_compacted`. -/
def compact (b : Batch F) : Option (Batch F) :=
  match b.sel with
  | none => some b
  | some s => (b.gather s).map (fun b' => { b' with sel := none })

theorem gather_fields {b b' : Batch F} {idx : List Nat} (h : b.gather idx = some b') :
    b'.len = idx.length ∧ b'.sel = none ∧ b'.vo = b.vo ∧
      mapO (fun c => c.gather idx) b.cols = some b'.cols := by
  unfold gather at h
  cases hc : mapO (fun c => c.gather idx) b.cols with
  | none => simp [hc] at h
  | some cols =>
    cases ho : b.origins with
    | none => simp [hc, ho] at h; subst h; simp
    | some o =>
      cases hg : gatherL o idx with
      | none => simp [hc, ho, hg] at h
      | some os => simp [hc, ho, hg] at h; subst h; simp

theorem gather_column {b b' : Batch F} {idx : List Nat} (h : b.gather idx = some b') (i : Nat) :
    (b.column i).gather idx = some (b'.column i) := by
  obtain ⟨-, -, -, hc⟩ := gather_fields h
  have := mapO_getElem? hc i
  have hl := mapO_length hc
  unfold column
  cases hb : b.cols[i]? with
  | none =>
    rw [hb] at this
    simp [this, Column.gather]
  | some c =>
    rw [hb, Option.bind_some] at this
    cases hg : c.gather idx with
    | none =>
      rw [hg] at this
      have hi : i < b.cols.length := by
        rcases Nat.lt_or_ge i b.cols.length with h' | h'
        · exact h'
        · simp [List.getElem?_eq_none h'] at hb
      rw [List.getElem?_eq_getElem (by omega)] at this; simp at this
    | some c' => rw [hg] at this; simp [this, hg]

/-- **gather reads row `idx[k]`** for every slot: value and bound bit. -/
theorem gather_obs {b b' : Batch F} {idx : List Nat} (h : b.gather idx = some b') {k : Nat}
    (hk : k < idx.length) (i : Nat) : b'.obs i k = b.obs i idx[k] := by
  have hc := gather_column h i
  have hu := Column.gather_isUnbound hc
  obtain ⟨-, -, hvo, -⟩ := gather_fields h
  simp only [obs, valueAt, isBound, hu, hvo, Column.gather_get? hc k hk]

/-- gather's origin sidecar reads `origins[idx[k]]` (the `any(!= 0)` elision
keeps the all-zero default). -/
theorem gather_originRow {b b' : Batch F} {idx : List Nat} (h : b.gather idx = some b') {k : Nat}
    (hk : k < idx.length) : b'.originRow k = b.originRow idx[k] := by
  unfold gather at h
  cases hc : mapO (fun c => c.gather idx) b.cols with
  | none => simp [hc] at h
  | some cols =>
    cases ho : b.origins with
    | none => simp [hc, ho] at h; subst h; simp [originRow, ho]
    | some o =>
      cases hg : gatherL o idx with
      | none => simp [hc, ho, hg] at h
      | some os =>
        simp [hc, ho, hg] at h; subst h
        have hget := gatherL_getElem? hg k
        rw [List.getElem?_eq_getElem hk, Option.bind_some] at hget
        split
        · rename_i hany
          simp [originRow, ho, hget]
        · rename_i hany
          simp only [originRow, ho, Option.getD_none]
          simp only [List.any_eq_true, ne_eq, decide_eq_true_eq, not_exists, not_and,
            Decidable.not_not] at hany
          cases hx : o[idx[k]]? with
          | none => rfl
          | some x =>
            simp only [Option.getD_some]
            have : os[k]? = some x := hget.trans hx
            exact (hany x (List.mem_of_getElem? this)).symm

/-- gather never panics when every index is a physical row. -/
theorem gather_isSome {b : Batch F} (h : b.WF) {idx : List Nat} (hi : ∀ i ∈ idx, i < b.len) :
    (b.gather idx).isSome := by
  have hc : (mapO (fun c => c.gather idx) b.cols).isSome := by
    apply mapO_isSome
    intro c hc
    cases hu : c.isUnbound
    · exact Column.gather_isSome (by rw [h.cols_len c hc hu]; exact hi)
    · cases c <;> simp [Column.isUnbound] at hu; rfl
  obtain ⟨cols, hcols⟩ := Option.isSome_iff_exists.mp hc
  unfold gather
  rw [hcols]
  cases ho : b.origins with
  | none => simp
  | some o =>
    have : (gatherL o idx).isSome :=
      gatherL_isSome_iff.mpr (by rw [h.origins_len o ho]; exact hi)
    obtain ⟨os, hos⟩ := Option.isSome_iff_exists.mp this
    simp [hos]

theorem gather_wf {b b' : Batch F} (h : b.WF) {idx : List Nat} (hg : b.gather idx = some b') :
    b'.WF := by
  obtain ⟨hlen, hsel, -, hc⟩ := gather_fields hg
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro c' hc' hu
    obtain ⟨j, hj⟩ := List.getElem?_of_mem hc'
    have := mapO_getElem? hc j
    rw [hj] at this
    cases hbj : b.cols[j]? with
    | none => rw [hbj] at this; simp at this
    | some c =>
      rw [hbj, Option.bind_some] at this
      have hcu : c.isUnbound = false := by rw [← Column.gather_isUnbound this.symm]; exact hu
      rw [Column.gather_len this.symm hcu, hlen]
  · intro s hs; rw [hsel] at hs; simp at hs
  · intro s hs; rw [hsel] at hs; simp at hs
  · intro o ho
    unfold gather at hg
    cases hc2 : mapO (fun c => c.gather idx) b.cols with
    | none => simp [hc2] at hg
    | some cols =>
      cases hbo : b.origins with
      | none => simp [hc2, hbo] at hg; subst hg; simp at ho
      | some o0 =>
        cases hgo : gatherL o0 idx with
        | none => simp [hc2, hbo, hgo] at hg
        | some os =>
          simp [hc2, hbo, hgo] at hg; subst hg
          simp only at ho ⊢
          split at ho
          · cases ho; exact gatherL_length hgo
          · simp at ho

/-- **Compaction preserves the active row sequence**: `into_compacted` is dense,
has `active_len` rows, and its row `k` is the `k`-th active row of the input,
slot by slot (value, bound bit, origin). -/
theorem compact_spec {b : Batch F} (h : b.WF) :
    ∃ b', b.compact = some b' ∧ b'.sel = none ∧ b'.len = b.activeLen ∧ b'.WF ∧
      ∀ k (hk : k < b.activeLen) i, b'.obs i k = b.obs i (b.active[k]'hk) ∧
        b'.originRow k = b.originRow (b.active[k]'hk) := by
  cases hs : b.sel with
  | none =>
    refine ⟨b, by simp [compact, hs], hs, ?_, h, ?_⟩
    · simp [activeLen, active, hs]
    · intro k hk i
      have : b.active[k]'hk = k := by simp [active, hs]
      rw [this]; exact ⟨rfl, rfl⟩
  | some s =>
    have hact : b.active = s := by simp [active, hs]
    obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.mp
      (gather_isSome h (idx := s) (fun i hi => active_lt h i (by rw [hact]; exact hi)))
    obtain ⟨hlen, hsel, -, -⟩ := gather_fields hb1
    have hwf := gather_wf h hb1
    refine ⟨{ b1 with sel := none }, by simp [compact, hs, hb1], rfl, ?_, ?_, ?_⟩
    · simp [hlen, activeLen, hact]
    · exact ⟨hwf.cols_len, by simp, by simp, hwf.origins_len⟩
    · intro k hk i
      have hk' : k < s.length := by simpa [activeLen, hact] using hk
      have e : b.active[k]'hk = s[k]'hk' := by simp [hact]
      rw [e]
      exact ⟨gather_obs hb1 hk' i, gather_originRow hb1 hk'⟩

/-- gather ∘ gather reads through the composed index map (gathering twice is
gathering once by `j.map (i[·])`). -/
theorem gather_gather_obs {b b1 b2 : Batch F} {i j : List Nat} (h1 : b.gather i = some b1)
    (h2 : b1.gather j = some b2) {k : Nat} (hk : k < j.length) (hj : j[k] < i.length) (x : Nat) :
    b2.obs x k = b.obs x i[j[k]] := by
  rw [gather_obs h2 hk, gather_obs h1 hj]

/-! ## Selection vectors -/

def setSel (b : Batch F) (s : List Nat) : Batch F := { b with sel := some s }

/-- The `as u16` cast. -/
def toU16 (i : Nat) : Nat := i % 65536

/-- Filter / SemiApply / Distinct / OrApply: keep the active rows passing `p`. -/
def filterSel (b : Batch F) (p : Nat → Bool) : Batch F :=
  b.setSel ((b.active.filter p).map toU16)

/-- Limit / result-set cap: keep the first `n` active rows. -/
def takeSel (b : Batch F) (n : Nat) : Batch F := b.setSel ((b.active.take n).map toU16)

/-- Skip: drop the first `n` active rows. -/
def dropSel (b : Batch F) (n : Nat) : Batch F := b.setSel ((b.active.drop n).map toU16)

theorem toU16_id {i : Nat} (h : i < 65536) : toU16 i = i := Nat.mod_eq_of_lt h

theorem map_toU16_id {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) {s : List Nat}
    (hs : s ⊆ b.active) : s.map toU16 = s := by
  conv => rhs; rw [← List.map_id s]
  apply List.map_congr_left
  intro i hi
  exact toU16_id (Nat.lt_of_lt_of_le (active_lt h i (hs hi)) hl)

/-- A sublist of the active rows is a valid selection. -/
theorem setSel_wf {b : Batch F} (h : b.WF) {s : List Nat} (hs : s.Sublist b.active) :
    (b.setSel s).WF :=
  ⟨h.cols_len,
   fun s' e i hi => by cases e; exact active_lt h i (hs.subset hi),
   fun s' e => by cases e; exact (active_sorted h).sublist hs,
   h.origins_len⟩

/-- **Selection composition (Filter).** With at most 65536 rows the cast is the
identity, the result is well formed, and its active rows are exactly the
previously active rows that pass, in order. -/
theorem filterSel_spec {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) (p : Nat → Bool) :
    (b.filterSel p).WF ∧ (b.filterSel p).active = b.active.filter p := by
  have e : (b.active.filter p).map toU16 = b.active.filter p :=
    map_toU16_id h hl (List.filter_sublist).subset
  refine ⟨?_, ?_⟩
  · unfold filterSel; rw [e]; exact setSel_wf h (List.filter_sublist)
  · show (b.active.filter p).map toU16 = b.active.filter p; exact e

theorem takeSel_spec {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) (n : Nat) :
    (b.takeSel n).WF ∧ (b.takeSel n).active = b.active.take n := by
  have e : (b.active.take n).map toU16 = b.active.take n :=
    map_toU16_id h hl (List.take_sublist _ _).subset
  refine ⟨?_, ?_⟩
  · unfold takeSel; rw [e]; exact setSel_wf h (List.take_sublist _ _)
  · show (b.active.take n).map toU16 = b.active.take n; exact e

theorem dropSel_spec {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) (n : Nat) :
    (b.dropSel n).WF ∧ (b.dropSel n).active = b.active.drop n := by
  have e : (b.active.drop n).map toU16 = b.active.drop n :=
    map_toU16_id h hl (List.drop_sublist _ _).subset
  refine ⟨?_, ?_⟩
  · unfold dropSel; rw [e]; exact setSel_wf h (List.drop_sublist _ _)
  · show (b.active.drop n).map toU16 = b.active.drop n; exact e

theorem setSel_len (b : Batch F) (s : List Nat) : (b.setSel s).len = b.len := rfl

/-- `SKIP m LIMIT n` composes to `drop m` then `take n` of the active rows. -/
theorem skip_then_limit {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) (m n : Nat) :
    ((b.dropSel m).takeSel n).active = (b.active.drop m).take n := by
  obtain ⟨hw, ha⟩ := dropSel_spec h hl m
  rw [(takeSel_spec hw (by rw [dropSel, setSel_len]; exact hl) n).2, ha]

/-- Filters chain: filtering twice keeps rows passing both. -/
theorem filter_then_filter {b : Batch F} (h : b.WF) (hl : b.len ≤ 65536) (p q : Nat → Bool) :
    ((b.filterSel p).filterSel q).active = b.active.filter (fun i => p i && q i) := by
  obtain ⟨hw, ha⟩ := filterSel_spec h hl p
  rw [(filterSel_spec hw (by rw [filterSel, setSel_len]; exact hl) q).2, ha, List.filter_filter]
  congr 1; funext i; exact Bool.and_comm _ _

/-- **The `as u16` cast is unsound past 65536 rows**: skipping one row of a
dense 65537-row batch selects row 0 again (at position 65535) — the selection
is no longer sorted, row 65536 is lost and row 0 is duplicated. Only
`BATCH_SIZE`-bounded producers keep batches below the limit; nothing checks it. -/
theorem skip_u16_wraps (cols : List (Column F)) :
    let b : Batch F := { len := 65537, sel := none, cols := cols, origins := none, vo := fun _ => false }
    (b.dropSel 1).active[65535]? = some 0 ∧ (b.dropSel 1).active[0]? = some 1 := by
  simp [dropSel, setSel, active, toU16]

/-! ## set_column / write_column -/

/-- The column `set_column` actually stores: `Values` are re-classified. -/
def normCol [FloatModel F] : Column F → Column F
  | .values vs => classifyStored vs
  | other => other

/-- `set_column`: `Values` columns are re-classified, `len` is taken from the
first column installed into an empty batch, the value-only mark is cleared. -/
def setColumn [FloatModel F] (b : Batch F) (id : Nat) (c : Column F) : Batch F :=
  { b with
    len := if b.len = 0 then (normCol c).len else b.len
    cols := (b.cols ++ List.replicate (id + 1 - b.cols.length) (Column.unbound : Column F)).set id (normCol c)
    vo := fun j => if j = id then false else b.vo j }

theorem column_setColumn [FloatModel F] (b : Batch F) (id : Nat) (c : Column F) (j : Nat) :
    (b.setColumn id c).column j = if j = id then normCol c else b.column j := by
  unfold setColumn column
  simp only
  by_cases hj : j = id
  · subst hj
    rw [List.getElem?_set_self (by simp; omega)]
    simp
  · rw [if_neg hj, List.getElem?_set_ne (Ne.symm hj)]
    rcases Nat.lt_or_ge j b.cols.length with hl | hl
    · rw [List.getElem?_append_left hl]
    · rw [List.getElem?_append_right hl, List.getElem?_eq_none hl]
      cases h : (List.replicate (id + 1 - b.cols.length) (Column.unbound : Column F))[j - b.cols.length]? with
      | none => rfl
      | some x =>
        rw [List.getElem?_replicate] at h
        split at h
        · cases h; rfl
        · simp at h

theorem normCol_isUnbound [FloatModel F] (c : Column F) : (normCol c).isUnbound = c.isUnbound := by
  cases c <;> simp only [normCol] <;> first | exact classifyStored_isUnbound _ | rfl

theorem normCol_toValues [FloatModel F] (c : Column F) : (normCol c).toValues = c.toValues := by
  cases c <;> simp only [normCol] <;> first | exact classifyStored_toValues _ | rfl

/-- Installing a column binds exactly that slot; every other slot is unchanged. -/
theorem setColumn_obs [FloatModel F] (b : Batch F) (id : Nat) (c : Column F) (r : Nat) :
    (b.setColumn id c).obs id r =
      (if c.isUnbound then none else c.toValues.bind (·[r]?), !c.isUnbound) ∧
    ∀ j, j ≠ id → (b.setColumn id c).obs j r = b.obs j r := by
  constructor
  · simp only [obs, valueAt, isBound, column_setColumn, if_true, normCol_isUnbound]
    cases hu : c.isUnbound
    · obtain ⟨vs, hv⟩ := Column.toValues_isSome hu
      have hv' := normCol_toValues c
      rw [hv] at hv'
      simp [Column.get?_eq_toValues hv', hv, setColumn]
    · simp [setColumn]
  · intro j hj
    have hc := column_setColumn b id c j
    rw [if_neg hj] at hc
    have hv : (b.setColumn id c).vo j = b.vo j := by simp [setColumn, hj]
    simp only [obs, valueAt, isBound, hc, hv]

/-- Scatter `(value, row)` pairs into `l` (`full[row] = val`, `batch.rs:1359-1361`). -/
def scatter {α : Type} : List α → List (α × Nat) → List α
  | l, [] => l
  | l, (v, r) :: ps => scatter (l.set r v) ps

theorem scatter_length {α : Type} : ∀ (l : List α) (ps : List (α × Nat)), (scatter l ps).length = l.length
  | l, [] => rfl
  | l, (v, r) :: ps => by simp [scatter, scatter_length _ ps]

theorem scatter_getElem?_not_mem {α : Type} :
    ∀ (l : List α) (ps : List (α × Nat)) (j : Nat), j ∉ ps.map Prod.snd → (scatter l ps)[j]? = l[j]?
  | l, [], j, _ => rfl
  | l, (v, r) :: ps, j, h => by
    simp only [List.map_cons, List.mem_cons, not_or] at h
    simp only [scatter]
    rw [scatter_getElem?_not_mem _ ps j h.2, List.getElem?_set_ne (Ne.symm h.1)]

theorem scatter_getElem?_mem {α : Type} :
    ∀ (l : List α) (ps : List (α × Nat)), (ps.map Prod.snd).Nodup → (∀ p ∈ ps, p.2 < l.length) →
      ∀ p ∈ ps, (scatter l ps)[p.2]? = some p.1
  | l, [], _, _, p, hp => by simp at hp
  | l, (v, r) :: ps, hnd, hlt, p, hp => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    simp only [scatter]
    rcases List.mem_cons.mp hp with rfl | hp
    · rw [scatter_getElem?_not_mem _ ps _ hnd.1, List.getElem?_set_self (hlt (v, r) (by simp))]
    · apply scatter_getElem?_mem _ ps hnd.2 _ p hp
      intro q hq; rw [List.length_set]; exact hlt q (by simp [hq])

/-- `write_column` (`batch.rs:1347-1367`). -/
def writeColumn [FloatModel F] (b : Batch F) (id : Nat) (vals : List (V F)) : Batch F :=
  match b.sel with
  | some s =>
    let full := (List.range b.len).map (fun r => (b.valueAt id r).getD .null)
    b.setColumn id (.values (scatter full (vals.zip s)))
  | none => b.setColumn id (.values vals)

/-- **gather ∘ scatter**: after `write_column`, the `k`-th active row reads
`vals[k]`, every inactive row keeps its old value (`Null` if unbound), and the
slot is bound. -/
theorem writeColumn_spec [FloatModel F] {b : Batch F} (h : b.WF) {id : Nat} {vals : List (V F)}
    (hl : vals.length = b.activeLen) :
    (∀ k (hk : k < b.activeLen), (b.writeColumn id vals).valueAt id (b.active[k]'hk) =
        vals[k]? ∧ (b.writeColumn id vals).isBound id = true) ∧
    (∀ r, r < b.len → r ∉ b.active →
        (b.writeColumn id vals).valueAt id r = some ((b.valueAt id r).getD .null)) := by
  cases hs : b.sel with
  | none =>
    have hact : b.active = List.range b.len := by simp [active, hs]
    refine ⟨fun k hk => ?_, fun r hr hn => by rw [hact] at hn; simp at hn; omega⟩
    have hk' : k < b.len := by simpa [activeLen, hact] using hk
    have e : b.active[k]'hk = k := by simp [hact]
    rw [e]
    have := (setColumn_obs b id (.values vals) k).1
    simp only [writeColumn, hs]
    simp only [obs, Prod.mk.injEq] at this
    refine ⟨?_, ?_⟩
    · rw [this.1]; simp [Column.isUnbound, Column.toValues]
    · rw [this.2]; simp [Column.isUnbound]
  | some s =>
    have hact : b.active = s := by simp [active, hs]
    have hlen : vals.length = s.length := by simpa [activeLen, hact] using hl
    let full := (List.range b.len).map (fun r => (b.valueAt id r).getD .null)
    have hwc : b.writeColumn id vals = b.setColumn id (.values (scatter full (vals.zip s))) := by
      simp [writeColumn, hs, full]
    have hobs := fun r => (setColumn_obs b id (.values (scatter full (vals.zip s))) r).1
    have hnd : ((vals.zip s).map Prod.snd).Nodup := by
      rw [List.map_snd_zip (by omega)]
      exact ((h.sel_sorted s hs).imp (fun hab => Nat.ne_of_lt hab))
    have hlt : ∀ p ∈ vals.zip s, p.2 < full.length := by
      intro p hp
      simp only [full, List.length_map, List.length_range]
      exact h.sel_lt s hs _ (List.of_mem_zip hp).2
    refine ⟨fun k hk => ?_, fun r hr hn => ?_⟩
    · have hk' : k < s.length := by simpa [activeLen, hact] using hk
      have e : b.active[k]'hk = s[k]'hk' := by simp [hact]
      rw [e, hwc]
      have := hobs (s[k]'hk')
      simp only [obs, Prod.mk.injEq] at this
      refine ⟨?_, ?_⟩
      · rw [this.1]
        simp only [Column.isUnbound, Bool.false_eq_true, ↓reduceIte, Column.toValues,
          Option.bind_some]
        have hmem : (vals[k]'(by omega), s[k]'hk') ∈ vals.zip s := by
          rw [List.mem_iff_getElem]
          exact ⟨k, by simp; omega, by simp⟩
        rw [scatter_getElem?_mem full _ hnd hlt _ hmem, List.getElem?_eq_getElem (by omega)]
      · rw [this.2]; simp [Column.isUnbound]
    · rw [hwc]
      have := hobs r
      simp only [obs, Prod.mk.injEq] at this
      rw [this.1]
      simp only [Column.isUnbound, Bool.false_eq_true, ↓reduceIte, Column.toValues,
        Option.bind_some]
      rw [scatter_getElem?_not_mem]
      · simp [full, List.getElem?_range hr]
      · rw [List.map_snd_zip (by omega)]; rw [hact] at hn; exact hn

end Batch

end Columnar
