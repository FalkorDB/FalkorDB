import PendingCommit.PendingCommitOp
/-
# UNIQUE constraints enforced by index lookup (#3334, 4ace60ec7; pending.rs:148-256, :1482-1615)

Before #3334 (fixing #3330) a UNIQUE check scanned the whole label (type) once per
affected member: `PC.check` / `PC.scan_ok` / `PC.check_spec` (kept as the reference
"label-scan spec"). Now `unique_violated` (:178) gathers the members' composite keys
(an early `true` on two members sharing one), then looks each key up in the
constraint's supporting index (one `Equal`, or an `And` of them) — or, without an
operational index or for a value the index cannot match exactly, scans the label once —
and re-verifies every candidate against the graph.

| Lean | Rust |
| --- | --- |
| `EV`, `exactLookup`, `exactLookup_spec` | `is_value_valid_for_index_lookup` (:153) |
| `gather` | the member loop of `unique_violated` (:197-226): `keys`, `lookups`, `use_index` |
| `uniqueViolated` | `unique_violated` (:178) |
| `checkMandatory`, `checkUnique`, `checkNode2`, `checkEdge2` | `check_node_constraint` (:1482), `check_edge_constraint` (:1546) |

Modelling:
* Values are an abstract type `V` with decidable equality; `attr e q = none` is an absent
  or `Null` property (both filtered by Rust before keying). The composite key of `e` is
  `key e = props.mapM (attr e)`: `Graph::build_composite_key` (graph.rs:4607) returns the
  empty byte string iff a property is null/absent, and otherwise the `Debug` rendering of
  each value followed by `'|'`, which we take as injective (strings render quoted and
  escaped). With `props = []` both sides degenerate the same way (`some []` / `b""`).
* `FxHashSet`/`Vec` are lists; `RoaringTreemap` iteration is a duplicate-free list.
* The index (`lookup`) and the label/type scan (`scan`) are uninterpreted. The theorems
  that compare with a specification take, as explicit hypotheses, the contract the Rust
  doc comment states: the index lags only this commit (`hIdx`: every live candidate
  *not* written in this commit whose key has only exactly-matchable values is returned
  by the lookup of that key), and the scan lists every live candidate (`hScan`).

Results:
* `uniqueViolated_spec` — `unique_violated` is `true` iff two members share a full key,
  or a member's key is also carried by a live candidate outside the affected set.
* `uniqueViolated_agrees` — under the constraint's pre-commit invariant (no two live,
  unaffected candidates share a full key) that is exactly the old label-scan verdict:
  some member participates and the label's full keys are not pairwise distinct.
* `checkMandatory_eq_old` — MANDATORY is unchanged by the refactor.
-/
namespace PendingCommit.UQ
open PC

/-! ## `is_value_valid_for_index_lookup` -/

/-- The value shapes the function distinguishes. `loses` is
`Index::int_loses_f64_precision` (proven in proofs/index_layer). -/
inductive EV where
  | int (i : Int)
  | float (isNaN : Bool)
  | bool (b : Bool)
  | str (s : String)
  | other

def exactLookup (loses : Int → Bool) : EV → Bool
  | .int i => !loses i
  | .float n => !n
  | .bool _ => true
  | .str s => !s.isEmpty
  | .other => false

theorem exactLookup_spec (loses : Int → Bool) (v : EV) :
    exactLookup loses v = true ↔
      (∃ i, v = .int i ∧ loses i = false) ∨ v = .float false ∨ (∃ b, v = .bool b) ∨
      (∃ s, v = .str s ∧ s ≠ "") := by
  cases v with
  | int i => simp [exactLookup]
  | float n => cases n <;> simp [exactLookup]
  | bool b => simp [exactLookup]
  | str s => simp [exactLookup]
  | other => simp [exactLookup]

/-! ## `unique_violated` -/

variable {V : Type} [DecidableEq V]

/-- The member loop: `none` is the early `return true` (two members share a key). -/
def gather (props : List Nat) (attr : Nat → Nat → Option V) (exact : V → Bool) :
    List Nat → List (List V) → List (List V) → Bool → Option (List (List V) × List (List V) × Bool)
  | [], keys, lk, ui => some (keys, lk, ui)
  | m :: ms, keys, lk, ui =>
    match props.mapM (attr m) with
    | none => gather props attr exact ms keys lk ui
    | some vals =>
      if vals ∈ keys then none
      else if ui then
        if vals.all exact then gather props attr exact ms (vals :: keys) (lk ++ [vals]) true
        else gather props attr exact ms (vals :: keys) [] false
      else gather props attr exact ms (vals :: keys) lk false

/-- `unique_violated` (:178). `affected` is the whole affected set, `isCand` the
`is_candidate` closure, `lookup` one index query (the conjunction of `Equal`s),
`scan` the label matrix / relationship tensor listing. -/
def uniqueViolated (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (isCand : Nat → Bool) (indexed : Bool) (exact : V → Bool)
    (lookup : List (Nat × V) → List Nat) (scan : List Nat) : Bool :=
  match gather props attr exact members [] [] indexed with
  | none => true
  | some (keys, lks, ui) =>
    if keys = [] then false
    else
      let collides := fun o => !affected o && isCand o &&
        (match props.mapM (attr o) with | some k => decide (k ∈ keys) | none => false)
      if ui then lks.any (fun vals => (lookup (props.zip vals)).any collides)
      else scan.any collides

section
variable (props : List Nat) (attr : Nat → Nat → Option V) (exact : V → Bool)

local notation "key" => fun e => props.mapM (attr e)

theorem gather_none : ∀ (ms : List Nat) (keys lk : List (List V)) (ui : Bool), keys.Nodup →
    (gather props attr exact ms keys lk ui = none ↔ ¬ (ms.filterMap key ++ keys).Nodup)
  | [], keys, lk, ui, h => by simp [gather, h]
  | m :: ms, keys, lk, ui, h => by
    simp only [gather, List.filterMap_cons]
    cases hk : props.mapM (attr m) with
    | none => exact gather_none ms keys lk ui h
    | some vals =>
      simp only [List.cons_append, List.nodup_cons]
      by_cases hv : vals ∈ keys
      · simp [hv]
      · have h' : (vals :: keys).Nodup := List.nodup_cons.2 ⟨hv, h⟩
        have e : (ms.filterMap key ++ vals :: keys).Nodup ↔ (vals ∉ ms.filterMap key ++ keys ∧
            (ms.filterMap key ++ keys).Nodup) := by
          rw [List.perm_middle.nodup_iff, List.nodup_cons]
        rw [if_neg hv]
        split
        · split
          · rw [gather_none ms _ _ _ h', e]
          · rw [gather_none ms _ _ _ h', e]
        · rw [gather_none ms _ _ _ h', e]

/-- Invariant: while the index is still in use, every key gathered is queued for lookup
and exactly matchable. -/
def GInv (keys lk : List (List V)) (ui : Bool) : Prop :=
  ui = true → ∀ k ∈ keys, k ∈ lk ∧ k.all exact = true

theorem gather_some : ∀ (ms : List Nat) (keys lk : List (List V)) (ui : Bool) r,
    GInv exact keys lk ui → gather props attr exact ms keys lk ui = some r →
    (∀ k, k ∈ r.1 ↔ k ∈ keys ∨ k ∈ ms.filterMap key) ∧ GInv exact r.1 r.2.1 r.2.2 ∧
    (r.2.2 = true → ui = true)
  | [], keys, lk, ui, r, hi, h => by
    simp only [gather, Option.some.injEq] at h; subst h; simp [hi]
  | m :: ms, keys, lk, ui, r, hi, h => by
    simp only [gather] at h
    simp only [List.filterMap_cons]
    cases hk : props.mapM (attr m) with
    | none =>
      rw [hk] at h; simp only at h
      exact gather_some ms keys lk ui r hi h
    | some vals =>
      rw [hk] at h; simp only at h
      by_cases hv : vals ∈ keys
      · rw [if_pos hv] at h; cases h
      rw [if_neg hv] at h
      have fin : ∀ lk' ui', GInv exact (vals :: keys) lk' ui' →
          gather props attr exact ms (vals :: keys) lk' ui' = some r → (ui' = true → ui = true) →
          (∀ k, k ∈ r.1 ↔ k ∈ keys ∨ k ∈ vals :: ms.filterMap key) ∧ GInv exact r.1 r.2.1 r.2.2 ∧
          (r.2.2 = true → ui = true) := by
        intro lk' ui' hi' h' hu
        obtain ⟨a, b, c⟩ := gather_some ms (vals :: keys) lk' ui' r hi' h'
        refine ⟨fun k => ?_, b, fun x => hu (c x)⟩
        rw [a k]; simp only [List.mem_cons]; constructor
        · rintro ((h | h) | h) <;> simp_all
        · rintro (h | h | h) <;> simp_all
      split at h
      · split at h
        · rename_i hu ha
          refine fin _ _ (fun _ k hk => ?_) h (fun _ => hu)
          simp only [List.mem_cons] at hk
          rcases hk with rfl | hk
          · exact ⟨by simp, ha⟩
          · obtain ⟨a, b⟩ := hi hu k hk; exact ⟨by simp [a], b⟩
        · exact fin _ _ (fun h => by cases h) h (fun h => by cases h)
      · exact fin _ _ (fun h => by cases h) h (fun h => by cases h)
end

/-- **`uniqueViolated_spec`**: given the index/scan contract, `unique_violated` reports a
violation iff two members share a full key, or some member's key is carried by a live
candidate this commit did not write. -/
theorem uniqueViolated_spec (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (isCand : Nat → Bool) (indexed : Bool) (exact : V → Bool)
    (lookup : List (Nat × V) → List Nat) (scan : List Nat)
    (hIdx : indexed = true → ∀ vals : List V, vals.all exact = true → ∀ o, affected o = false →
      isCand o = true → props.mapM (attr o) = some vals → o ∈ lookup (props.zip vals))
    (hScan : ∀ o, affected o = false → isCand o = true → (props.mapM (attr o)).isSome → o ∈ scan) :
    uniqueViolated props members affected attr isCand indexed exact lookup scan = true ↔
      (¬ (members.filterMap fun e => props.mapM (attr e)).Nodup ∨
       ∃ m ∈ members, ∃ k, props.mapM (attr m) = some k ∧
         ∃ o, affected o = false ∧ isCand o = true ∧ props.mapM (attr o) = some k) := by
  unfold uniqueViolated
  cases hg : gather props attr exact members [] [] indexed with
  | none =>
    have := (gather_none props attr exact members [] [] indexed List.nodup_nil).1 hg
    simp only [List.append_nil] at this
    simp [this]
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    have hnd : (members.filterMap fun e => props.mapM (attr e)).Nodup := by
      have := gather_none props attr exact members [] [] indexed List.nodup_nil
      simp only [List.append_nil] at this
      exact Classical.byContradiction fun hc => by rw [this.2 hc] at hg; cases hg
    obtain ⟨hmem, hinv, hui⟩ := gather_some props attr exact members [] [] indexed _
      (fun _ k hk => by cases hk) hg
    simp only [List.not_mem_nil, false_or] at hmem
    dsimp only
    simp only [hnd, not_true_eq_false, false_or]
    have toM : ∀ k, k ∈ keys → ∃ m ∈ members, props.mapM (attr m) = some k := by
      intro k hk; have := (hmem k).1 hk; rw [List.mem_filterMap] at this; exact this
    have ofM : ∀ m ∈ members, ∀ k, props.mapM (attr m) = some k → k ∈ keys := by
      intro m hm k hk; exact (hmem k).2 (List.mem_filterMap.2 ⟨m, hm, hk⟩)
    have hcol : ∀ o, ((!affected o && isCand o &&
        (match props.mapM (attr o) with | some k => decide (k ∈ keys) | none => false)) = true ↔
        affected o = false ∧ isCand o = true ∧ ∃ k, props.mapM (attr o) = some k ∧ k ∈ keys) := by
      intro o
      cases h : props.mapM (attr o) <;> simp [and_assoc]
    by_cases hk : keys = []
    · simp only [hk, ite_true, Bool.false_eq_true, false_iff, not_exists, not_and]
      intro m hm k hmk
      have := ofM m hm k hmk; rw [hk] at this; cases this
    · simp only [hk, ite_false]
      constructor
      · intro h
        have : ∃ o, affected o = false ∧ isCand o = true ∧ ∃ k, props.mapM (attr o) = some k ∧ k ∈ keys := by
          split at h
          · obtain ⟨vals, -, h⟩ := List.any_eq_true.1 h
            obtain ⟨o, -, h⟩ := List.any_eq_true.1 h
            exact ⟨o, (hcol o).1 h⟩
          · obtain ⟨o, -, h⟩ := List.any_eq_true.1 h
            exact ⟨o, (hcol o).1 h⟩
        obtain ⟨o, ha, hc, k, hok, hkk⟩ := this
        obtain ⟨m, hm, hmk⟩ := toM k hkk
        exact ⟨m, hm, k, hmk, o, ha, hc, hok⟩
      · rintro ⟨m, hm, k, hmk, o, ha, hc, hok⟩
        have hkk := ofM m hm k hmk
        have hco := (hcol o).2 ⟨ha, hc, k, hok, hkk⟩
        split
        · rename_i hu
          obtain ⟨hl, hx⟩ := hinv hu k hkk
          have ho := hIdx (hui hu) k hx o ha hc hok
          exact List.any_eq_true.2 ⟨k, hl, List.any_eq_true.2 ⟨o, ho, hco⟩⟩
        · exact List.any_eq_true.2 ⟨o, hScan o ha hc (by simp [hok]), hco⟩

/-! ## Agreement with the label-scan spec -/

/-- Pairwise form of `(l.filterMap f).Nodup` over a duplicate-free `l`. -/
theorem filterMap_nodup_iff {α β : Type} (f : α → Option β) : ∀ (l : List α), l.Nodup →
    ((l.filterMap f).Nodup ↔ ∀ a ∈ l, ∀ b ∈ l, a ≠ b → ∀ k, f a = some k → f b ≠ some k)
  | [], _ => by simp
  | a :: l, h => by
    rw [List.nodup_cons] at h
    have ih := filterMap_nodup_iff f l h.2
    simp only [List.filterMap_cons, List.mem_cons]
    cases hfa : f a with
    | none =>
      rw [ih]
      constructor
      · intro H x hx y hy hxy k hk
        rcases hx with rfl | hx
        · rw [hfa] at hk; cases hk
        rcases hy with rfl | hy
        · rw [hfa]; simp
        · exact H x hx y hy hxy k hk
      · intro H x hx y hy hxy k hk; exact H x (.inr hx) y (.inr hy) hxy k hk
    | some k0 =>
      rw [List.nodup_cons, ih]
      have hn : k0 ∉ l.filterMap f ↔ ∀ b ∈ l, f b ≠ some k0 := by
        rw [List.mem_filterMap]; simp
      rw [hn]
      constructor
      · rintro ⟨H1, H2⟩ x hx y hy hxy k hk
        rcases hx with rfl | hx
        · rw [hfa] at hk; cases hk
          rcases hy with rfl | hy
          · exact absurd rfl hxy
          · exact H1 y hy
        rcases hy with rfl | hy
        · rw [hfa]; intro e; cases e; exact H1 x hx hk
        · exact H2 x hx y hy hxy k hk
      · intro H
        refine ⟨fun b hb => ?_, fun x hx y hy hxy k hk => H x (.inr hx) y (.inr hy) hxy k hk⟩
        have hab : a ≠ b := fun e => h.1 (e ▸ hb)
        exact H a (.inl rfl) b (.inr hb) hab k0 hfa

/-- **`uniqueViolated_agrees`**: with the scan contract and the constraint's pre-commit
invariant `hPre` (no two live candidates outside this commit share a full key), the
index-based verdict equals the old label-scan verdict (`PC.check_spec`'s UNIQUE clause:
some member participates and the full keys over the label, `all`, are not distinct). -/
theorem uniqueViolated_agrees (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (isCand : Nat → Bool) (indexed : Bool) (exact : V → Bool)
    (lookup : List (Nat × V) → List Nat) (all : List Nat)
    (hIdx : indexed = true → ∀ vals : List V, vals.all exact = true → ∀ o, affected o = false →
      isCand o = true → props.mapM (attr o) = some vals → o ∈ lookup (props.zip vals))
    (hnd : members.Nodup) (hallnd : all.Nodup) (hall : ∀ o, o ∈ all ↔ isCand o = true)
    (hmem : ∀ m ∈ members, isCand m = true ∧ affected m = true)
    (hAff : ∀ o, affected o = true → isCand o = true → o ∈ members)
    (hPre : ∀ o₁ o₂, affected o₁ = false → affected o₂ = false → isCand o₁ = true →
      isCand o₂ = true → o₁ ≠ o₂ → ∀ k, props.mapM (attr o₁) = some k → props.mapM (attr o₂) ≠ some k) :
    uniqueViolated props members affected attr isCand indexed exact lookup all = true ↔
      ((∃ m ∈ members, (props.mapM (attr m)).isSome) ∧
        ¬ (all.filterMap fun e => props.mapM (attr e)).Nodup) := by
  rw [uniqueViolated_spec props members affected attr isCand indexed exact lookup all hIdx
    (fun o _ hc _ => (hall o).2 hc)]
  rw [filterMap_nodup_iff _ _ hnd, filterMap_nodup_iff _ _ hallnd]
  simp only [Classical.not_forall, Classical.not_not, exists_prop]
  constructor
  · rintro (⟨a, ha, b, hb, hab, k, hka, hkb⟩ | ⟨m, hm, k, hmk, o, hoa, hoc, hok⟩)
    · exact ⟨⟨a, ha, by simp [hka]⟩, a, (hall a).2 (hmem a ha).1, b, (hall b).2 (hmem b hb).1, hab, k, hka, hkb⟩
    · have hmo : m ≠ o := fun e => by
        have := (hmem m hm).2; rw [e, hoa] at this; cases this
      exact ⟨⟨m, hm, by simp [hmk]⟩, m, (hall m).2 (hmem m hm).1, o, (hall o).2 hoc, hmo, k, hmk, hok⟩
  · rintro ⟨-, a, ha, b, hb, hab, k, hka, hkb⟩
    have hca := (hall a).1 ha
    have hcb := (hall b).1 hb
    cases hfa : affected a <;> cases hfb : affected b
    · exact absurd hkb (hPre a b hfa hfb hca hcb hab k hka)
    · exact .inr ⟨b, hAff b hfb hcb, k, hkb, a, hfa, hca, hka⟩
    · exact .inr ⟨a, hAff a hfa hca, k, hka, b, hfb, hcb, hkb⟩
    · exact .inl ⟨a, hAff a hfa hca, b, hAff b hfb hcb, hab, k, hka, hkb⟩

/-! ## `check_node_constraint` / `check_edge_constraint` since #3334 -/

/-- MANDATORY: every member has every property (the loop moved under the `match`). -/
def checkMandatory (props affected : List Nat) (member : Nat → Bool) (present : Nat → Nat → Bool) : Bool :=
  (affected.filter member).all fun e => props.all (present e)

theorem checkMandatory_eq_old (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (key : Nat → List Nat) (all : List Nat) :
    checkMandatory props affected member present = check .mandatory props affected member present key all := by
  unfold checkMandatory check
  induction affected with
  | nil => rfl
  | cons a l ih =>
    simp only [List.filter_cons, List.all_cons]
    cases hm : member a <;> simp [ih]

/-- UNIQUE: `!unique_violated` over the members `affected.filter member`. -/
def checkUnique (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option V)
    (isCand : Nat → Bool) (indexed : Bool) (exact : V → Bool)
    (lookup : List (Nat × V) → List Nat) (scan : List Nat) : Bool :=
  !uniqueViolated props (affected.filter member) (fun o => decide (o ∈ affected)) attr isCand indexed
    exact lookup scan

/-- `check_node_constraint` (:1482): no such label ⇒ `Ok`. `isCand` for nodes is
`!deleted_nodes.contains(n) && constraint_node_has_label(n)` (`PC.constraintHas`). -/
def checkNode2 (labelExists : Bool) (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option V) (isCand : Nat → Bool) (indexed : Bool)
    (exact : V → Bool) (lookup : List (Nat × V) → List Nat) (scan : List Nat) : Bool :=
  !labelExists || match ct with
    | .mandatory => checkMandatory props affected member present
    | .unique => checkUnique props affected member attr isCand indexed exact lookup scan

/-- `check_edge_constraint` (:1546): membership is `PC.edgeMember`; `isCand` is
`!deleted_relationships.contains(e)` (the index and the tensor hold only this type). -/
def checkEdge2 (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option V) (isCand : Nat → Bool) (indexed : Bool)
    (exact : V → Bool) (lookup : List (Nat × V) → List Nat) (scan : List Nat) : Bool :=
  match ct with
  | .mandatory => checkMandatory props affected member present
  | .unique => checkUnique props affected member attr isCand indexed exact lookup scan

/-- **`checkUnique_spec`**: the UNIQUE check passes iff no two members share a full key
and no member's key is carried by a live candidate outside the affected set (under the
index/scan contract). -/
theorem checkUnique_spec (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option V)
    (isCand : Nat → Bool) (indexed : Bool) (exact : V → Bool)
    (lookup : List (Nat × V) → List Nat) (scan : List Nat)
    (hIdx : indexed = true → ∀ vals : List V, vals.all exact = true → ∀ o, o ∉ affected →
      isCand o = true → props.mapM (attr o) = some vals → o ∈ lookup (props.zip vals))
    (hScan : ∀ o, o ∉ affected → isCand o = true → (props.mapM (attr o)).isSome → o ∈ scan) :
    checkUnique props affected member attr isCand indexed exact lookup scan = true ↔
      ((affected.filter member).filterMap fun e => props.mapM (attr e)).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, props.mapM (attr m) = some k →
        ∀ o, o ∉ affected → isCand o = true → props.mapM (attr o) ≠ some k := by
  unfold checkUnique
  rw [Bool.not_eq_true', ← Bool.not_eq_true,
    uniqueViolated_spec props _ _ attr isCand indexed exact lookup scan
      (fun hi vals hv o ho hc hk => hIdx hi vals hv o (by simpa using ho) hc hk)
      (fun o ho hc hs => hScan o (by simpa using ho) hc hs)]
  simp only [not_or, Classical.not_not, not_exists, not_and, decide_eq_false_iff_not]

end PendingCommit.UQ
