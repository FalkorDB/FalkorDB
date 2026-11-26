import PendingCommit.PendingCommitOp
/-
# UNIQUE constraints enforced by index lookup (#3547/#2386; pending.rs `unique_violated` and helpers, :1580-1700 constraint checks)

A UNIQUE check no longer scans the whole label (type) once per
affected member (`PC.check` / `PC.scan_ok` / `PC.check_spec` are kept as the reference
"label-scan spec"); `unique_violated` (:217) gathers the members' composite keys
(an early `true` on two members sharing one), then looks each key up through an index and
re-verifies every candidate against the graph. The index is abstracted by two
closures: `query_for(values) -> Option<Q>` gives the lookup for one member (or `None`: the
index cannot answer for it) and `lookup(Q)` runs one. `use_index` starts `true`; one member
for which `query_for` is `None` sends the whole set to the label scan, and the lookups
gathered so far are dropped (none runs). The RediSearch `query_for` is
`redisearch_unique_query` (:174); the native one is `native_unique_query`
(index/falkordb/unique.rs), which this project does not model.

Modelling:
* Values are an abstract type `V`; `attr e q = none` is an absent or `Null` property (both
  filtered by Rust before keying). The composite key of `e` is
  `keyOf e = (props.mapM (attr e)).bind ck`, where `ck : List V → Option K` is
  `Graph::build_composite_key` on non-null values, `none` standing for its empty key (a
  value the constraint does not enforce: list, point, temporal, vecf32, NaN). Such a member
  is skipped like a NULL (`if key.is_empty() { continue }`), and an empty key never
  collides. `ck` is arbitrary here, so key equality may identify distinct values (`1`, `1.0`
  and `true`); `query_for` still gets the member's raw values. graph_persist
  `Constraint.buildKey_eq_iff` characterises the real key (equal iff C calls the values
  duplicates). With `props = []` both sides degenerate the same way.
* `FxHashSet`/`Vec` are lists; `RoaringTreemap` iteration is a duplicate-free list.
* The lookup (`queryFor`, `lookup`) and the label/type scan (`scan`) are uninterpreted. The
  theorems that compare with a specification take, as explicit hypotheses, the contract the
  Rust doc comment states: a lookup only has to return a superset (`hIdx`: for a query the
  member's key gets, every live candidate *not* written in this commit carrying that key is
  returned, i.e. every one whose key equals the key of the values it was built from), and the
  scan lists every live candidate (`hScan`). `UniqueQuery.lean`
  discharges `hIdx` for RediSearch from the Equal-exactness of the index alone; the native
  lookup's `hIdx` is the undischarged hypothesis (unique.rs).

Results:
* `uniqueViolated_spec` — `unique_violated` is `true` iff two members share a key,
  or a member's key is also carried by a live candidate outside the affected set.
* `uniqueViolated_declined` — a member `query_for` declines sends the whole set to the
  scan: the verdict no longer depends on `lookup` at all (no lookup runs).
* `uniqueViolated_agrees` — under the constraint's pre-commit invariant (no two live,
  unaffected candidates share a key) that is exactly the old label-scan verdict:
  some member participates and the label's keys are not pairwise distinct.
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

variable {V K Q : Type} [DecidableEq K]

/-- The composite key of entity `e`: its constrained values (`none`: one is NULL or absent),
run through `ck` = `Graph::build_composite_key` (`none`: the empty key, a value the
constraint does not enforce). This is also the `collides` closure's
`build_composite_key(properties, |prop| attr(other, prop))`. -/
def keyOf (props : List Nat) (attr : Nat → Nat → Option V) (ck : List V → Option K) (e : Nat) :
    Option K :=
  (props.mapM (attr e)).bind ck

/-- The values of a member that participates (non-empty key): what `query_for` is given. -/
def pvals (props : List Nat) (attr : Nat → Nat → Option V) (ck : List V → Option K) (e : Nat) :
    Option (List V) :=
  (props.mapM (attr e)).filter fun vals => (ck vals).isSome

/-- The member loop: `none` is the early `return true` (two members share a key); a member
with a NULL/absent value or an empty key is skipped (`continue`).
`qf` is `query_for`, given the member's values; a `none` from it clears the queue and
`use_index` for good. -/
def gather (props : List Nat) (attr : Nat → Nat → Option V) (ck : List V → Option K)
    (qf : List V → Option Q) :
    List Nat → List K → List Q → Bool → Option (List K × List Q × Bool)
  | [], keys, lk, ui => some (keys, lk, ui)
  | m :: ms, keys, lk, ui =>
    match props.mapM (attr m) with
    | none => gather props attr ck qf ms keys lk ui
    | some vals =>
      match ck vals with
      | none => gather props attr ck qf ms keys lk ui
      | some k =>
        if k ∈ keys then none
        else if ui then
          match qf vals with
          | some q => gather props attr ck qf ms (k :: keys) (lk ++ [q]) true
          | none => gather props attr ck qf ms (k :: keys) [] false
        else gather props attr ck qf ms (k :: keys) lk false

/-- `unique_violated`. `affected` is the whole affected set, `isCand` the
`is_candidate` closure, `ck` is `build_composite_key`, `qf` is `query_for`, `lookup` runs
one query, `scan` the label matrix / relationship tensor listing. -/
def uniqueViolated (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (ck : List V → Option K) (isCand : Nat → Bool)
    (qf : List V → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  match gather props attr ck qf members [] [] true with
  | none => true
  | some (keys, lks, ui) =>
    if keys = [] then false
    else
      let collides := fun o => !affected o && isCand o &&
        (match keyOf props attr ck o with | some k => decide (k ∈ keys) | none => false)
      if ui then lks.any (fun q => (lookup q).any collides)
      else scan.any collides

section
variable (props : List Nat) (attr : Nat → Nat → Option V) (ck : List V → Option K)
  (qf : List V → Option Q)

local notation "key" => keyOf props attr ck
local notation "pv" => pvals props attr ck

/-- How one member enters the loop: skipped, or with its values and its key. -/
theorem member_cases (m : Nat) :
    (keyOf props attr ck m = none ∧ pvals props attr ck m = none ∧
      ∀ ms keys lk ui, gather props attr ck qf (m :: ms) keys lk ui = gather props attr ck qf ms keys lk ui) ∨
    (∃ vals k, ck vals = some k ∧ keyOf props attr ck m = some k ∧ pvals props attr ck m = some vals ∧
      ∀ ms keys lk ui, gather props attr ck qf (m :: ms) keys lk ui =
        if k ∈ keys then none
        else if ui then
          match qf vals with
          | some q => gather props attr ck qf ms (k :: keys) (lk ++ [q]) true
          | none => gather props attr ck qf ms (k :: keys) [] false
        else gather props attr ck qf ms (k :: keys) lk false) := by
  cases hk : props.mapM (attr m) with
  | none => exact .inl ⟨by simp [keyOf, hk], by simp [pvals, hk], fun _ _ _ _ => by simp [gather, hk]⟩
  | some vals =>
    cases hc : ck vals with
    | none =>
      exact .inl ⟨by simp [keyOf, hk, hc], by simp [pvals, hk, hc, Option.filter],
        fun _ _ _ _ => by simp [gather, hk, hc]⟩
    | some k =>
      exact .inr ⟨vals, k, hc, by simp [keyOf, hk, hc], by simp [pvals, hk, hc, Option.filter],
        fun _ _ _ _ => by simp only [gather, hk, hc]⟩

theorem gather_none : ∀ (ms : List Nat) (keys : List K) (lk : List Q) (ui : Bool), keys.Nodup →
    (gather props attr ck qf ms keys lk ui = none ↔ ¬ (ms.filterMap key ++ keys).Nodup)
  | [], keys, lk, ui, h => by simp [gather, h]
  | m :: ms, keys, lk, ui, h => by
    rcases member_cases props attr ck qf m with ⟨hkey, -, hg⟩ | ⟨vals, k, -, hkey, -, hg⟩
    · rw [hg, List.filterMap_cons, hkey]; exact gather_none ms keys lk ui h
    · rw [hg, List.filterMap_cons, hkey]
      simp only [List.cons_append, List.nodup_cons]
      by_cases hv : k ∈ keys
      · simp [hv]
      · have h' : (k :: keys).Nodup := List.nodup_cons.2 ⟨hv, h⟩
        have e : (ms.filterMap key ++ k :: keys).Nodup ↔ (k ∉ ms.filterMap key ++ keys ∧
            (ms.filterMap key ++ keys).Nodup) := by
          rw [List.perm_middle.nodup_iff, List.nodup_cons]
        rw [ite_eq_right hv]
        by_cases hu : ui = true
        · rw [ite_eq_left hu]
          cases hq : qf vals with
          | some q => dsimp only; rw [gather_none ms _ _ _ h', e]
          | none => dsimp only; rw [gather_none ms _ _ _ h', e]
        · rw [ite_eq_right hu, gather_none ms _ _ _ h', e]

/-- Invariant: while the index is still in use, every key gathered has the query of some
values carrying it queued. -/
def GInv (keys : List K) (lk : List Q) (ui : Bool) : Prop :=
  ui = true → ∀ k ∈ keys, ∃ vals q, ck vals = some k ∧ qf vals = some q ∧ q ∈ lk

theorem gather_some : ∀ (ms : List Nat) (keys : List K) (lk : List Q) (ui : Bool) r,
    GInv ck qf keys lk ui → gather props attr ck qf ms keys lk ui = some r →
    (∀ k, k ∈ r.1 ↔ k ∈ keys ∨ k ∈ ms.filterMap key) ∧ GInv ck qf r.1 r.2.1 r.2.2 ∧
    (r.2.2 = true → ui = true) ∧
    (r.2.2 = true ↔ ui = true ∧ ∀ vals ∈ ms.filterMap pv, ∃ q, qf vals = some q)
  | [], keys, lk, ui, r, hi, h => by
    simp only [gather, Option.some.injEq] at h; subst h; simp [hi]
  | m :: ms, keys, lk, ui, r, hi, h => by
    rcases member_cases props attr ck qf m with ⟨hkey, hpv, hg⟩ | ⟨vals, k, hc, hkey, hpv, hg⟩
    · rw [hg] at h
      rw [List.filterMap_cons, hkey, List.filterMap_cons, hpv]
      exact gather_some ms keys lk ui r hi h
    · rw [hg] at h
      clear hg
      rw [List.filterMap_cons, hkey, List.filterMap_cons, hpv]
      dsimp only
      by_cases hv : k ∈ keys
      · rw [ite_eq_left hv] at h; cases h
      rw [ite_eq_right hv] at h
      by_cases hu : ui = true
      · rw [ite_eq_left hu] at h
        cases hq : qf vals with
        | some q =>
          rw [hq] at h; dsimp only at h
          obtain ⟨a, b, c, d⟩ := gather_some ms (k :: keys) (lk ++ [q]) true r (fun _ k' hk => by
            simp only [List.mem_cons] at hk
            rcases hk with rfl | hk
            · exact ⟨vals, q, hc, hq, by simp⟩
            · obtain ⟨v', q', a, b, c⟩ := hi hu k' hk; exact ⟨v', q', a, b, by simp [c]⟩) h
          refine ⟨?_, b, fun _ => hu, ?_⟩
          · intro k'; rw [a k']; simp only [List.mem_cons]; constructor
            · rintro ((h | h) | h) <;> simp_all
            · rintro (h | h | h) <;> simp_all
          · rw [d]; simp only [hu, true_and, List.mem_cons, forall_eq_or_imp]
            exact ⟨fun h => ⟨⟨q, hq⟩, h⟩, fun h => h.2⟩
        | none =>
          rw [hq] at h; dsimp only at h
          obtain ⟨a, b, c, d⟩ := gather_some ms (k :: keys) [] false r (fun h => by cases h) h
          refine ⟨?_, b, fun h => absurd (c h) (by simp), ?_⟩
          · intro k'; rw [a k']; simp only [List.mem_cons]; constructor
            · rintro ((h | h) | h) <;> simp_all
            · rintro (h | h | h) <;> simp_all
          · constructor
            · intro h'; have := (d.1 h').1; simp at this
            · rintro ⟨-, h2⟩
              obtain ⟨q, hq'⟩ := h2 vals (by simp)
              rw [hq] at hq'; cases hq'
      · rw [ite_eq_right hu] at h
        obtain ⟨a, b, c, d⟩ := gather_some ms (k :: keys) lk false r (fun h => by cases h) h
        refine ⟨?_, b, fun h => absurd (c h) (by simp), ?_⟩
        · intro k'; rw [a k']; simp only [List.mem_cons]; constructor
          · rintro ((h | h) | h) <;> simp_all
          · rintro (h | h | h) <;> simp_all
        · constructor
          · intro h'; have := (d.1 h').1; simp at this
          · rintro ⟨h1, -⟩; exact absurd h1 hu
end

/-- **`uniqueViolated_spec`**: given the lookup/scan contract, `unique_violated` reports a
violation iff two members share a (non-empty) key, or some member's key is carried by a live
candidate this commit did not write. `hIdx`: a lookup built from values whose key is `k`
returns a superset of the live candidates outside the commit whose key is `k`. -/
theorem uniqueViolated_spec (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (ck : List V → Option K) (isCand : Nat → Bool)
    (qf : List V → Option Q) (lookup : Q → List Nat) (scan : List Nat)
    (hIdx : ∀ vals k q, ck vals = some k → qf vals = some q → ∀ o, affected o = false →
      isCand o = true → keyOf props attr ck o = some k → o ∈ lookup q)
    (hScan : ∀ o, affected o = false → isCand o = true → (keyOf props attr ck o).isSome → o ∈ scan) :
    uniqueViolated props members affected attr ck isCand qf lookup scan = true ↔
      (¬ (members.filterMap (keyOf props attr ck)).Nodup ∨
       ∃ m ∈ members, ∃ k, keyOf props attr ck m = some k ∧
         ∃ o, affected o = false ∧ isCand o = true ∧ keyOf props attr ck o = some k) := by
  unfold uniqueViolated
  cases hg : gather props attr ck qf members [] [] true with
  | none =>
    have := (gather_none props attr ck qf members [] [] true List.nodup_nil).1 hg
    simp only [List.append_nil] at this
    simp [this]
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    have hnd : (members.filterMap (keyOf props attr ck)).Nodup := by
      have := gather_none props attr ck qf members [] [] true List.nodup_nil
      simp only [List.append_nil] at this
      exact Classical.byContradiction fun hc => by rw [this.2 hc] at hg; cases hg
    obtain ⟨hmem, hinv, hui, -⟩ := gather_some props attr ck qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    simp only [List.not_mem_nil, false_or] at hmem
    dsimp only
    simp only [hnd, not_true_eq_false, false_or]
    have toM : ∀ k, k ∈ keys → ∃ m ∈ members, keyOf props attr ck m = some k := by
      intro k hk; have := (hmem k).1 hk; rw [List.mem_filterMap] at this; exact this
    have ofM : ∀ m ∈ members, ∀ k, keyOf props attr ck m = some k → k ∈ keys := by
      intro m hm k hk; exact (hmem k).2 (List.mem_filterMap.2 ⟨m, hm, hk⟩)
    have hcol : ∀ o, ((!affected o && isCand o &&
        (match keyOf props attr ck o with | some k => decide (k ∈ keys) | none => false)) = true ↔
        affected o = false ∧ isCand o = true ∧ ∃ k, keyOf props attr ck o = some k ∧ k ∈ keys) := by
      intro o
      cases h : keyOf props attr ck o <;> simp [and_assoc]
    by_cases hk : keys = []
    · simp only [hk, ite_true, Bool.false_eq_true, false_iff, not_exists, not_and]
      intro m hm k hmk
      have := ofM m hm k hmk; rw [hk] at this; cases this
    · simp only [hk, ite_false]
      constructor
      · intro h
        have : ∃ o, affected o = false ∧ isCand o = true ∧ ∃ k, keyOf props attr ck o = some k ∧ k ∈ keys := by
          split at h
          · obtain ⟨q, -, h⟩ := List.any_eq_true.1 h
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
          obtain ⟨vals, q, hck, hq, hl⟩ := hinv hu k hkk
          have ho := hIdx vals k q hck hq o ha hc hok
          exact List.any_eq_true.2 ⟨q, hl, List.any_eq_true.2 ⟨o, ho, hco⟩⟩
        · exact List.any_eq_true.2 ⟨o, hScan o ha hc (by simp [hok]), hco⟩

/-- **`uniqueViolated_declined`**: when `query_for` declines the values of some
participating member (that is reached: no two members shared a key before it), the whole
set goes to the scan and no lookup runs: the verdict is the same for every `lookup`. (The
lookups gathered before the declined member are dropped: `gather_some`'s last conjunct,
`use_index` ends `false`.) -/
theorem uniqueViolated_declined (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (ck : List V → Option K) (isCand : Nat → Bool)
    (qf : List V → Option Q) (lookup lookup' : Q → List Nat) (scan : List Nat)
    (hdecl : ∃ vals ∈ members.filterMap (pvals props attr ck), qf vals = none) :
    uniqueViolated props members affected attr ck isCand qf lookup scan =
      uniqueViolated props members affected attr ck isCand qf lookup' scan := by
  unfold uniqueViolated
  cases hg : gather props attr ck qf members [] [] true with
  | none => rfl
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    obtain ⟨-, -, -, hiff⟩ := gather_some props attr ck qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    have hui : ui = false := by
      cases hu : ui
      · rfl
      · obtain ⟨-, h2⟩ := hiff.1 hu
        obtain ⟨k, hk, hq⟩ := hdecl
        obtain ⟨q, hq'⟩ := h2 k hk
        rw [hq] at hq'; cases hq'
    subst hui
    simp

/-- The converse: with no declined member, `use_index` stays true, so every gathered key
runs its query (and the scan is not consulted). -/
theorem uniqueViolated_indexed (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (ck : List V → Option K) (isCand : Nat → Bool)
    (qf : List V → Option Q) (lookup : Q → List Nat) (scan scan' : List Nat)
    (hall : ∀ vals ∈ members.filterMap (pvals props attr ck), ∃ q, qf vals = some q) :
    uniqueViolated props members affected attr ck isCand qf lookup scan =
      uniqueViolated props members affected attr ck isCand qf lookup scan' := by
  unfold uniqueViolated
  cases hg : gather props attr ck qf members [] [] true with
  | none => rfl
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    obtain ⟨-, -, -, hiff⟩ := gather_some props attr ck qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    have hui : ui = true := hiff.2 ⟨rfl, hall⟩
    subst hui
    simp

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
invariant `hPre` (no two live candidates outside this commit share a key), the
index-based verdict equals the old label-scan verdict (`PC.check_spec`'s UNIQUE clause:
some member participates and the keys over the label, `all`, are not distinct). -/
theorem uniqueViolated_agrees (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option V) (ck : List V → Option K) (isCand : Nat → Bool)
    (qf : List V → Option Q) (lookup : Q → List Nat) (all : List Nat)
    (hIdx : ∀ vals k q, ck vals = some k → qf vals = some q → ∀ o, affected o = false →
      isCand o = true → keyOf props attr ck o = some k → o ∈ lookup q)
    (hnd : members.Nodup) (hallnd : all.Nodup) (hall : ∀ o, o ∈ all ↔ isCand o = true)
    (hmem : ∀ m ∈ members, isCand m = true ∧ affected m = true)
    (hAff : ∀ o, affected o = true → isCand o = true → o ∈ members)
    (hPre : ∀ o₁ o₂, affected o₁ = false → affected o₂ = false → isCand o₁ = true →
      isCand o₂ = true → o₁ ≠ o₂ → ∀ k, keyOf props attr ck o₁ = some k → keyOf props attr ck o₂ ≠ some k) :
    uniqueViolated props members affected attr ck isCand qf lookup all = true ↔
      ((∃ m ∈ members, (keyOf props attr ck m).isSome) ∧
        ¬ (all.filterMap (keyOf props attr ck)).Nodup) := by
  rw [uniqueViolated_spec props members affected attr ck isCand qf lookup all hIdx
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

/-! ## `check_node_constraint` / `check_edge_constraint` -/

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
    (ck : List V → Option K) (isCand : Nat → Bool) (qf : List V → Option Q)
    (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  !uniqueViolated props (affected.filter member) (fun o => decide (o ∈ affected)) attr ck isCand qf
    lookup scan

/-- `check_node_constraint` (:1580): no such label ⇒ `Ok`. `isCand` for nodes is
`!deleted_nodes.contains(n) && constraint_node_has_label(n)` (`PC.constraintHas`). `qf`/`lookup`
are the per-cfg `(query_for, lookup)` pair. -/
def checkNode2 (labelExists : Bool) (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option V) (ck : List V → Option K)
    (isCand : Nat → Bool) (qf : List V → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  !labelExists || match ct with
    | .mandatory => checkMandatory props affected member present
    | .unique => checkUnique props affected member attr ck isCand qf lookup scan

/-- `check_edge_constraint` (:1672): membership is `PC.edgeMember`; `isCand` is
`!deleted_relationships.contains(e)` (the index and the tensor hold only this type). -/
def checkEdge2 (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option V) (ck : List V → Option K)
    (isCand : Nat → Bool) (qf : List V → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  match ct with
  | .mandatory => checkMandatory props affected member present
  | .unique => checkUnique props affected member attr ck isCand qf lookup scan

/-- **`checkUnique_spec`**: the UNIQUE check passes iff no two members share a key
and no member's key is carried by a live candidate outside the affected set (under the
lookup/scan contract). -/
theorem checkUnique_spec (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option V)
    (ck : List V → Option K) (isCand : Nat → Bool) (qf : List V → Option Q)
    (lookup : Q → List Nat) (scan : List Nat)
    (hIdx : ∀ vals k q, ck vals = some k → qf vals = some q → ∀ o, o ∉ affected →
      isCand o = true → keyOf props attr ck o = some k → o ∈ lookup q)
    (hScan : ∀ o, o ∉ affected → isCand o = true → (keyOf props attr ck o).isSome → o ∈ scan) :
    checkUnique props affected member attr ck isCand qf lookup scan = true ↔
      ((affected.filter member).filterMap (keyOf props attr ck)).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, keyOf props attr ck m = some k →
        ∀ o, o ∉ affected → isCand o = true → keyOf props attr ck o ≠ some k := by
  unfold checkUnique
  rw [Bool.not_eq_true', ← Bool.not_eq_true,
    uniqueViolated_spec props _ _ attr ck isCand qf lookup scan
      (fun vals k q hk hq o ho hc hok => hIdx vals k q hk hq o (by simpa using ho) hc hok)
      (fun o ho hc hs => hScan o (by simpa using ho) hc hs)]
  simp only [not_or, Classical.not_not, not_exists, not_and, decide_eq_false_iff_not]

end PendingCommit.UQ
