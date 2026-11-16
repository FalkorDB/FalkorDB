import PendingCommit.PendingCommitOp
/-
# UNIQUE constraints enforced by index lookup (#3334, #3547/#2386; pending.rs:61-290 `unique_violated` and helpers, :1580-1700 constraint checks)

Before #3334 (fixing #3330) a UNIQUE check scanned the whole label (type) once per
affected member: `PC.check` / `PC.scan_ok` / `PC.check_spec` (kept as the reference
"label-scan spec"). Now `unique_violated` (:216) gathers the members' composite keys
(skipping a member with a `Null` or unenforced value; an early `true` on two members sharing one), then looks each key up through an index and
re-verifies every candidate against the graph. Since #3547 the index is abstracted by two
closures: `query_for(values) -> Option<Q>` gives the lookup for one member (or `None`: the
index cannot answer for it) and `lookup(Q)` runs one. `use_index` starts `true`; one member
for which `query_for` is `None` sends the whole set to the label scan, and the lookups
gathered so far are dropped (none runs). The RediSearch `query_for` is
`redisearch_unique_query` (:173); the native one is `native_unique_query`
(index/falkordb/unique.rs), which this project does not model.

| Lean | Rust |
| --- | --- |
| `EV`, `exactLookup`, `exactLookup_spec` | `is_value_valid_for_index_lookup` (:159) |
| `entOf`, `keyOf` | `values` / `Graph::build_composite_key` in `unique_violated` (:236-249) |
| `gather` | the member loop of `unique_violated` (:234-267): `keys`, `lookups`, `use_index` |
| `uniqueViolated` | `unique_violated` (:216) |
| `checkMandatory`, `checkUnique`, `checkNode2`, `checkEdge2` | `check_node_constraint` (:1580), `check_edge_constraint` (:1672) |
| `UniqueQuery.lean`: `Query`, `redisearchQuery` | `redisearch_unique_query` (:173) |

Modelling:
* Since #2991 values and keys are separate. Property values are an abstract type `W`;
  `attr e q = none` is an absent or `Null` property (both filtered by Rust before keying).
  `kv : W → Option V` is one component of `Graph::build_composite_key` (graph.rs:4866):
  `none` for a value the constraint does not enforce (list, point, temporal, vecf32, NaN:
  the empty key), otherwise the component's key class (`V` equality = equal key bytes:
  numbers as their folded `f64`, so `1`, `1.0`, `true` share one; strings byte for byte).
  `entOf` pairs a member's values `ws` (what `query_for` receives) with its key `vals`, and is
  `none` exactly when Rust skips the member (`collect` fails, or `key.is_empty()` at :245);
  `keyOf = (props.mapM (attr e)).bind (·.mapM kv)` (`keyOf_eq`) is the key `collides`
  recomputes. That the byte key is injective on key classes is proven in proofs/graph_persist
  (`composite_key_unambiguous`, `keyFrag_eq_iff`). With `props = []` both sides degenerate
  the same way (`some []` / `b""`).
* `FxHashSet`/`Vec` are lists; `RoaringTreemap` iteration is a duplicate-free list.
* The lookup (`queryFor`, `lookup`) and the label/type scan (`scan`) are uninterpreted. The
  theorems that compare with a specification take, as explicit hypotheses, the contract the
  Rust doc comment states: a lookup only has to return a superset (`hIdx`: for a query the
  member's key gets, every live candidate *not* written in this commit carrying that key is
  returned), and the scan lists every live candidate (`hScan`). `UniqueQuery.lean`
  discharges `hIdx` for RediSearch from the Equal-exactness of the index alone; the native
  lookup's `hIdx` is the undischarged hypothesis (unique.rs).

Results:
* `uniqueViolated_spec` — `unique_violated` is `true` iff two members share a full key,
  or a member's key is also carried by a live candidate outside the affected set.
* `uniqueViolated_declined` — a member `query_for` declines sends the whole set to the
  scan: the verdict no longer depends on `lookup` at all (no lookup runs).
* `uniqueViolated_vacuous` — (#2991) a member with a value the constraint does not enforce
  is ignored, like one with a `Null` property.
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

variable {W V Q : Type} [DecidableEq V]

/-- One member's values and key, or `none` when it does not participate: a property is
absent or `Null` (`attr` is `none`, the `collect::<Option<Vec<_>>>` at :236-242), or a value
is one the constraint does not enforce (`kv` is `none`, the `key.is_empty()` `continue` at
:245-249, #2991). `ws` is what `query_for` receives, `vals` the composite key. -/
def entOf (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) (e : Nat) :
    Option (List W × List V) :=
  match props.mapM (attr e) with
  | none => none
  | some ws =>
    match ws.mapM kv with
    | none => none
    | some vals => some (ws, vals)

/-- `Graph::build_composite_key` of an entity: `none` is the empty key. -/
def keyOf (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) (e : Nat) :
    Option (List V) :=
  (entOf props attr kv e).map Prod.snd

theorem keyOf_eq (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) (e : Nat) :
    keyOf props attr kv e = (props.mapM (attr e)).bind (·.mapM kv) := by
  unfold keyOf entOf
  cases h1 : props.mapM (attr e) with
  | none => rfl
  | some ws => cases h2 : ws.mapM kv <;> simp [h2]

/-- The member loop: `none` is the early `return true` (two members share a key).
`qf` is `query_for` (on the member's values); a `none` from it clears the queue and
`use_index` for good. -/
def gather (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V)
    (qf : List W → Option Q) :
    List Nat → List (List V) → List Q → Bool → Option (List (List V) × List Q × Bool)
  | [], keys, lk, ui => some (keys, lk, ui)
  | m :: ms, keys, lk, ui =>
    match entOf props attr kv m with
    | none => gather props attr kv qf ms keys lk ui
    | some (ws, vals) =>
      if vals ∈ keys then none
      else if ui then
        match qf ws with
        | some q => gather props attr kv qf ms (vals :: keys) (lk ++ [q]) true
        | none => gather props attr kv qf ms (vals :: keys) [] false
      else gather props attr kv qf ms (vals :: keys) lk false

/-- `unique_violated` (:216). `affected` is the whole affected set, `isCand` the
`is_candidate` closure, `qf` is `query_for`, `lookup` runs one query,
`scan` the label matrix / relationship tensor listing. -/
def uniqueViolated (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  match gather props attr kv qf members [] [] true with
  | none => true
  | some (keys, lks, ui) =>
    if keys = [] then false
    else
      let collides := fun o => !affected o && isCand o &&
        (match keyOf props attr kv o with | some k => decide (k ∈ keys) | none => false)
      if ui then lks.any (fun q => (lookup q).any collides)
      else scan.any collides

theorem keyOf_some_of_ent (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) {m : Nat} {ws : List W} {vals : List V}
    (h : entOf props attr kv m = some (ws, vals)) : keyOf props attr kv m = some vals := by
  simp [keyOf, h]

theorem keyOf_none_of_ent (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) {m : Nat} (h : entOf props attr kv m = none) :
    keyOf props attr kv m = none := by
  simp [keyOf, h]

theorem entOf_mapM (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V) {m : Nat} {ws : List W} {vals : List V}
    (h : entOf props attr kv m = some (ws, vals)) : ws.mapM kv = some vals := by
  unfold entOf at h
  cases h1 : props.mapM (attr m) with
  | none => rw [h1] at h; cases h
  | some ws' =>
    rw [h1] at h; dsimp only at h
    cases h2 : ws'.mapM kv with
    | none => rw [h2] at h; cases h
    | some v' => rw [h2] at h; cases h; exact h2

section
variable (props : List Nat) (attr : Nat → Nat → Option W) (kv : W → Option V)
  (qf : List W → Option Q)

local notation "key" => keyOf props attr kv

theorem gather_none : ∀ (ms : List Nat) (keys : List (List V)) (lk : List Q) (ui : Bool), keys.Nodup →
    (gather props attr kv qf ms keys lk ui = none ↔ ¬ (ms.filterMap key ++ keys).Nodup)
  | [], keys, lk, ui, h => by simp [gather, h]
  | m :: ms, keys, lk, ui, h => by
    simp only [gather, List.filterMap_cons]
    cases hk : entOf props attr kv m with
    | none => rw [keyOf_none_of_ent props attr kv hk]; exact gather_none ms keys lk ui h
    | some p =>
      obtain ⟨ws, vals⟩ := p
      rw [keyOf_some_of_ent props attr kv hk]
      simp only [List.cons_append, List.nodup_cons]
      by_cases hv : vals ∈ keys
      · simp [hv]
      · have h' : (vals :: keys).Nodup := List.nodup_cons.2 ⟨hv, h⟩
        have e : (ms.filterMap key ++ vals :: keys).Nodup ↔ (vals ∉ ms.filterMap key ++ keys ∧
            (ms.filterMap key ++ keys).Nodup) := by
          rw [List.perm_middle.nodup_iff, List.nodup_cons]
        rw [if_neg hv]
        by_cases hu : ui = true
        · rw [if_pos hu]
          cases hq : qf ws with
          | some q => dsimp only; rw [gather_none ms _ _ _ h', e]
          | none => dsimp only; rw [gather_none ms _ _ _ h', e]
        · rw [if_neg hu, gather_none ms _ _ _ h', e]

/-- Invariant: while the index is still in use, every key gathered has its query queued. -/
def GInv (keys : List (List V)) (lk : List Q) (ui : Bool) : Prop :=
  ui = true → ∀ k ∈ keys, ∃ ws q, ws.mapM kv = some k ∧ qf ws = some q ∧ q ∈ lk

theorem gather_some : ∀ (ms : List Nat) (keys : List (List V)) (lk : List Q) (ui : Bool) r,
    GInv kv qf keys lk ui → gather props attr kv qf ms keys lk ui = some r →
    (∀ k, k ∈ r.1 ↔ k ∈ keys ∨ k ∈ ms.filterMap key) ∧ GInv kv qf r.1 r.2.1 r.2.2 ∧
    (r.2.2 = true → ui = true) ∧
    (r.2.2 = true ↔ ui = true ∧ ∀ p ∈ ms.filterMap (entOf props attr kv), ∃ q, qf p.1 = some q)
  | [], keys, lk, ui, r, hi, h => by
    simp only [gather, Option.some.injEq] at h; subst h; simp [hi]
  | m :: ms, keys, lk, ui, r, hi, h => by
    simp only [gather] at h
    simp only [List.filterMap_cons]
    cases hk : entOf props attr kv m with
    | none =>
      rw [hk] at h; simp only at h
      rw [keyOf_none_of_ent props attr kv hk]
      exact gather_some ms keys lk ui r hi h
    | some p =>
      obtain ⟨ws, vals⟩ := p
      rw [hk] at h; simp only at h
      rw [keyOf_some_of_ent props attr kv hk]
      have hmk := entOf_mapM props attr kv hk
      clear hk
      by_cases hv : vals ∈ keys
      · rw [if_pos hv] at h; cases h
      rw [if_neg hv] at h
      by_cases hu : ui = true
      · rw [if_pos hu] at h
        cases hq : qf ws with
        | some q =>
          rw [hq] at h; dsimp only at h
          obtain ⟨a, b, c, d⟩ := gather_some ms (vals :: keys) (lk ++ [q]) true r (fun _ k hk => by
            simp only [List.mem_cons] at hk
            rcases hk with rfl | hk
            · exact ⟨ws, q, hmk, hq, by simp⟩
            · obtain ⟨ws', q', a, b, c⟩ := hi hu k hk; exact ⟨ws', q', a, b, by simp [c]⟩) h
          refine ⟨?_, b, fun _ => hu, ?_⟩
          · intro k; rw [a k]; simp only [List.mem_cons]; constructor
            · rintro ((h | h) | h) <;> simp_all
            · rintro (h | h | h) <;> simp_all
          · rw [d]; simp only [hu, true_and, List.mem_cons, forall_eq_or_imp]
            exact ⟨fun h => ⟨⟨q, hq⟩, h⟩, fun h => h.2⟩
        | none =>
          rw [hq] at h; dsimp only at h
          obtain ⟨a, b, c, d⟩ := gather_some ms (vals :: keys) [] false r (fun h => by cases h) h
          refine ⟨?_, b, fun h => absurd (c h) (by simp), ?_⟩
          · intro k; rw [a k]; simp only [List.mem_cons]; constructor
            · rintro ((h | h) | h) <;> simp_all
            · rintro (h | h | h) <;> simp_all
          · constructor
            · intro h'; have := (d.1 h').1; simp at this
            · rintro ⟨-, h2⟩
              obtain ⟨q, hq'⟩ := h2 (ws, vals) (by simp)
              rw [hq] at hq'; cases hq'
      · rw [if_neg hu] at h
        clear hmk
        obtain ⟨a, b, c, d⟩ := gather_some ms (vals :: keys) lk false r (fun h => by cases h) h
        refine ⟨?_, b, fun h => absurd (c h) (by simp), ?_⟩
        · intro k; rw [a k]; simp only [List.mem_cons]; constructor
          · rintro ((h | h) | h) <;> simp_all
          · rintro (h | h | h) <;> simp_all
        · constructor
          · intro h'; have := (d.1 h').1; simp at this
          · rintro ⟨h1, -⟩; exact absurd h1 hu
end

/-- **`uniqueViolated_spec`**: given the lookup/scan contract, `unique_violated` reports a
violation iff two members share a full key, or some member's key is carried by a live
candidate this commit did not write. `hIdx`: a lookup returns a superset of the live
candidates outside the commit that carry the key of the values it was built for. -/
theorem uniqueViolated_spec (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup : Q → List Nat) (scan : List Nat)
    (hIdx : ∀ ws q, qf ws = some q → ∀ vals, ws.mapM kv = some vals → ∀ o, affected o = false →
      isCand o = true → keyOf props attr kv o = some vals → o ∈ lookup q)
    (hScan : ∀ o, affected o = false → isCand o = true → (keyOf props attr kv o).isSome → o ∈ scan) :
    uniqueViolated props members affected attr kv isCand qf lookup scan = true ↔
      (¬ (members.filterMap (keyOf props attr kv)).Nodup ∨
       ∃ m ∈ members, ∃ k, keyOf props attr kv m = some k ∧
         ∃ o, affected o = false ∧ isCand o = true ∧ keyOf props attr kv o = some k) := by
  unfold uniqueViolated
  cases hg : gather props attr kv qf members [] [] true with
  | none =>
    have := (gather_none props attr kv qf members [] [] true List.nodup_nil).1 hg
    simp only [List.append_nil] at this
    simp [this]
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    have hnd : (members.filterMap (keyOf props attr kv)).Nodup := by
      have := gather_none props attr kv qf members [] [] true List.nodup_nil
      simp only [List.append_nil] at this
      exact Classical.byContradiction fun hc => by rw [this.2 hc] at hg; cases hg
    obtain ⟨hmem, hinv, hui, -⟩ := gather_some props attr kv qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    simp only [List.not_mem_nil, false_or] at hmem
    dsimp only
    simp only [hnd, not_true_eq_false, false_or]
    have toM : ∀ k, k ∈ keys → ∃ m ∈ members, keyOf props attr kv m = some k := by
      intro k hk; have := (hmem k).1 hk; rw [List.mem_filterMap] at this; exact this
    have ofM : ∀ m ∈ members, ∀ k, keyOf props attr kv m = some k → k ∈ keys := by
      intro m hm k hk; exact (hmem k).2 (List.mem_filterMap.2 ⟨m, hm, hk⟩)
    have hcol : ∀ o, ((!affected o && isCand o &&
        (match keyOf props attr kv o with | some k => decide (k ∈ keys) | none => false)) = true ↔
        affected o = false ∧ isCand o = true ∧ ∃ k, keyOf props attr kv o = some k ∧ k ∈ keys) := by
      intro o
      cases h : keyOf props attr kv o <;> simp [and_assoc]
    by_cases hk : keys = []
    · simp only [hk, ite_true, Bool.false_eq_true, false_iff, not_exists, not_and]
      intro m hm k hmk
      have := ofM m hm k hmk; rw [hk] at this; cases this
    · simp only [hk, ite_false]
      constructor
      · intro h
        have : ∃ o, affected o = false ∧ isCand o = true ∧ ∃ k, keyOf props attr kv o = some k ∧ k ∈ keys := by
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
          obtain ⟨ws, q, hws, hq, hl⟩ := hinv hu k hkk
          have ho := hIdx ws q hq k hws o ha hc hok
          exact List.any_eq_true.2 ⟨q, hl, List.any_eq_true.2 ⟨o, ho, hco⟩⟩
        · exact List.any_eq_true.2 ⟨o, hScan o ha hc (by simp [hok]), hco⟩

/-- **`uniqueViolated_declined`**: when `query_for` declines the values of some
participating member, the whole set goes to the scan and no lookup runs: the verdict is
the same for every `lookup`. (The lookups gathered before the declined member are dropped:
`gather_some`'s last conjunct, `use_index` ends `false`.) -/
theorem uniqueViolated_declined (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup lookup' : Q → List Nat) (scan : List Nat)
    (hdecl : ∃ p ∈ members.filterMap (entOf props attr kv), qf p.1 = none) :
    uniqueViolated props members affected attr kv isCand qf lookup scan =
      uniqueViolated props members affected attr kv isCand qf lookup' scan := by
  unfold uniqueViolated
  cases hg : gather props attr kv qf members [] [] true with
  | none => rfl
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    obtain ⟨-, -, -, hiff⟩ := gather_some props attr kv qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    have hui : ui = false := by
      cases hu : ui
      · rfl
      · obtain ⟨-, h2⟩ := hiff.1 hu
        obtain ⟨p, hp, hq⟩ := hdecl
        obtain ⟨q, hq'⟩ := h2 p hp
        rw [hq] at hq'; cases hq'
    subst hui
    simp

/-- The converse: with no declined member, `use_index` stays true, so every gathered key
runs its query (and the scan is not consulted). -/
theorem uniqueViolated_indexed (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup : Q → List Nat) (scan scan' : List Nat)
    (hall : ∀ p ∈ members.filterMap (entOf props attr kv), ∃ q, qf p.1 = some q) :
    uniqueViolated props members affected attr kv isCand qf lookup scan =
      uniqueViolated props members affected attr kv isCand qf lookup scan' := by
  unfold uniqueViolated
  cases hg : gather props attr kv qf members [] [] true with
  | none => rfl
  | some r =>
    obtain ⟨keys, lks, ui⟩ := r
    obtain ⟨-, -, -, hiff⟩ := gather_some props attr kv qf members [] [] true _
      (fun _ k hk => by cases hk) hg
    have hui : ui = true := hiff.2 ⟨rfl, hall⟩
    subst hui
    simp

/-- **`uniqueViolated_vacuous`** (#2991): a member whose values the constraint does not
enforce (a list, point, temporal, vecf32 or NaN: `kv` is `none` on one of them) changes
nothing, exactly as a member with a `Null` property: dropping it from `members` gives the
same verdict. -/
theorem uniqueViolated_vacuous (props : List Nat) (members : List Nat) (affected : Nat → Bool)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup : Q → List Nat) (scan : List Nat)
    (m : Nat) (hm : keyOf props attr kv m = none) :
    uniqueViolated props (m :: members) affected attr kv isCand qf lookup scan =
      uniqueViolated props members affected attr kv isCand qf lookup scan := by
  have he : entOf props attr kv m = none := by
    cases h : entOf props attr kv m with
    | none => rfl
    | some p => simp [keyOf, h] at hm
  simp [uniqueViolated, gather, he]

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
    (attr : Nat → Nat → Option W) (kv : W → Option V) (isCand : Nat → Bool)
    (qf : List W → Option Q) (lookup : Q → List Nat) (all : List Nat)
    (hIdx : ∀ ws q, qf ws = some q → ∀ vals, ws.mapM kv = some vals → ∀ o, affected o = false →
      isCand o = true → keyOf props attr kv o = some vals → o ∈ lookup q)
    (hnd : members.Nodup) (hallnd : all.Nodup) (hall : ∀ o, o ∈ all ↔ isCand o = true)
    (hmem : ∀ m ∈ members, isCand m = true ∧ affected m = true)
    (hAff : ∀ o, affected o = true → isCand o = true → o ∈ members)
    (hPre : ∀ o₁ o₂, affected o₁ = false → affected o₂ = false → isCand o₁ = true →
      isCand o₂ = true → o₁ ≠ o₂ → ∀ k, keyOf props attr kv o₁ = some k → keyOf props attr kv o₂ ≠ some k) :
    uniqueViolated props members affected attr kv isCand qf lookup all = true ↔
      ((∃ m ∈ members, (keyOf props attr kv m).isSome) ∧
        ¬ (all.filterMap (keyOf props attr kv)).Nodup) := by
  rw [uniqueViolated_spec props members affected attr kv isCand qf lookup all hIdx
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
def checkUnique (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option W)
    (kv : W → Option V) (isCand : Nat → Bool) (qf : List W → Option Q)
    (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  !uniqueViolated props (affected.filter member) (fun o => decide (o ∈ affected)) attr kv isCand qf
    lookup scan

/-- `check_node_constraint` (:1580): no such label ⇒ `Ok`. `isCand` for nodes is
`!deleted_nodes.contains(n) && constraint_node_has_label(n)` (`PC.constraintHas`). `qf`/`lookup`
are the per-cfg `(query_for, lookup)` pair. -/
def checkNode2 (labelExists : Bool) (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option W) (kv : W → Option V)
    (isCand : Nat → Bool) (qf : List W → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  !labelExists || match ct with
    | .mandatory => checkMandatory props affected member present
    | .unique => checkUnique props affected member attr kv isCand qf lookup scan

/-- `check_edge_constraint` (:1672): membership is `PC.edgeMember`; `isCand` is
`!deleted_relationships.contains(e)` (the index and the tensor hold only this type). -/
def checkEdge2 (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (attr : Nat → Nat → Option W) (kv : W → Option V)
    (isCand : Nat → Bool) (qf : List W → Option Q) (lookup : Q → List Nat) (scan : List Nat) : Bool :=
  match ct with
  | .mandatory => checkMandatory props affected member present
  | .unique => checkUnique props affected member attr kv isCand qf lookup scan

/-- **`checkUnique_spec`**: the UNIQUE check passes iff no two members share a full key
and no member's key is carried by a live candidate outside the affected set (under the
lookup/scan contract). -/
theorem checkUnique_spec (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option W)
    (kv : W → Option V) (isCand : Nat → Bool) (qf : List W → Option Q)
    (lookup : Q → List Nat) (scan : List Nat)
    (hIdx : ∀ ws q, qf ws = some q → ∀ vals, ws.mapM kv = some vals → ∀ o, o ∉ affected →
      isCand o = true → keyOf props attr kv o = some vals → o ∈ lookup q)
    (hScan : ∀ o, o ∉ affected → isCand o = true → (keyOf props attr kv o).isSome → o ∈ scan) :
    checkUnique props affected member attr kv isCand qf lookup scan = true ↔
      ((affected.filter member).filterMap (keyOf props attr kv)).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, keyOf props attr kv m = some k →
        ∀ o, o ∉ affected → isCand o = true → keyOf props attr kv o ≠ some k := by
  unfold checkUnique
  rw [Bool.not_eq_true', ← Bool.not_eq_true,
    uniqueViolated_spec props _ _ attr kv isCand qf lookup scan
      (fun ws q hq vals hv o ho hc hk => hIdx ws q hq vals hv o (by simpa using ho) hc hk)
      (fun o ho hc hs => hScan o (by simpa using ho) hc hs)]
  simp only [not_or, Classical.not_not, not_exists, not_and, decide_eq_false_iff_not]

end PendingCommit.UQ
