import PendingCommit.UniqueIndex
/-
# `redisearch_unique_query` (pending.rs) and the RediSearch `(query_for, lookup)` pair

`check_node_constraint` / `check_edge_constraint` without `index-falkordb` build
`query_for = |values| redisearch_unique_query(indexed, &constraint.properties, values)` and
`lookup = |query| g.get_indexed_nodes(label, query)` (edges: `get_indexed_edges`).

Results:
* `redisearchQuery_none_iff` — `None` iff the properties are not all Range-indexed or some
  value is not exactly matchable (`is_value_valid_for_index_lookup`).
* `redisearchQuery_single`, `redisearchQuery_and` — one `Equal` when the zip has exactly one
  pair (one constrained property), otherwise an `And` of every `Equal`.
* `redisearch_hIdx` — with the index's per-`Equal` exactness as the only hypothesis (an FFI
  fact about RediSearch, `hEq`), the lookup of the query this fn builds returns every live
  candidate outside the commit carrying the key: `uniqueViolated_spec`'s `hIdx`, discharged.

The key is `build_composite_key`: one component per value (`kc`, `none` for a value the
constraint does not enforce), the key empty (`none`) when any is; equal byte keys are equal
component lists (graph_persist `Constraint.buildKey_eq_iff`), so `ck vals = vals.mapM kc`.
`hEq` matches on components, not on raw values: an `Equal` on `1` returns the entities
holding `1`, `1.0` or `true`, as the numeric index does (the key equates them, #2774).
* `checkUnique_redisearch_spec` — the UNIQUE verdict on the RediSearch path.
* `redisearch_unindexed_scans` — unindexed ⇒ every member is declined ⇒ no lookup runs.
-/
namespace PendingCommit.UQ

variable {V : Type}

inductive Query (V : Type) where
  | equal (key : Nat) (value : V)
  | and (eqs : List (Nat × V))

/-- `redisearch_unique_query`. `eqs.len() == 1 ⇒ eqs.pop().unwrap()`. -/
def redisearchQuery (exact : V → Bool) (indexed : Bool) (props : List Nat) (vals : List V) :
    Option (Query V) :=
  if !indexed || !vals.all exact then none
  else
    some (match props.zip vals with
      | [(k, v)] => .equal k v
      | eqs => .and eqs)

theorem redisearchQuery_none_iff (exact : V → Bool) (indexed : Bool) (props : List Nat)
    (vals : List V) :
    redisearchQuery exact indexed props vals = none ↔
      (indexed = false ∨ vals.all exact = false) := by
  unfold redisearchQuery
  cases indexed <;> cases h : vals.all exact <;> simp

theorem redisearchQuery_single (exact : V → Bool) (props : List Nat) (vals : List V) (k : Nat)
    (v : V) (hz : props.zip vals = [(k, v)]) (hv : vals.all exact = true) :
    redisearchQuery exact true props vals = some (.equal k v) := by
  simp [redisearchQuery, hv, hz]

theorem redisearchQuery_and (exact : V → Bool) (props : List Nat) (vals : List V)
    (hn : (props.zip vals).length ≠ 1) (hv : vals.all exact = true) :
    redisearchQuery exact true props vals = some (.and (props.zip vals)) := by
  unfold redisearchQuery
  rw [ite_eq_right (by simp [hv])]
  congr 1
  split
  · rename_i k v h; rw [h] at hn; simp at hn
  · rfl

/-- The result of running a query on an index whose `Equal` lookups are `ix`; `And` is the
intersection of its `Equal`s. -/
def evalQ (ix : Nat × V → List Nat) : Query V → List Nat
  | .equal k v => ix (k, v)
  | .and eqs => (eqs.flatMap ix).filter fun o => eqs.all fun p => decide (o ∈ ix p)

/-- `mapM` length. -/
theorem mapM_length {α β : Type} (f : α → Option β) : ∀ (l : List α) (r : List β),
    l.mapM f = some r → r.length = l.length
  | [], r, h => by simp [List.mapM_nil] at h; subst h; rfl
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | none => rw [hb] at h; simp at h
    | some b =>
      cases hbs : l.mapM f with
      | none => rw [hb, hbs] at h; simp at h
      | some bs =>
        rw [hb, hbs] at h
        simp at h
        subst h
        simp [mapM_length f l bs hbs]

theorem mapM_cons_some {α β : Type} (f : α → Option β) (a : α) (l : List α) (r : List β)
    (h : (a :: l).mapM f = some r) : ∃ b bs, f a = some b ∧ l.mapM f = some bs ∧ r = b :: bs := by
  rw [List.mapM_cons] at h
  cases hb : f a with
  | none => rw [hb] at h; simp at h
  | some b =>
    cases hbs : l.mapM f with
    | none => rw [hb, hbs] at h; simp at h
    | some bs =>
      rw [hb, hbs] at h
      simp at h
      exact ⟨b, bs, rfl, rfl, h.symm⟩

/-- A candidate `o` with the member's key: each pair `(p, v)` of the member's query has, at
`o`, a value `w` with the same component as `v`. -/
theorem mapM_zip_key {C : Type} (attr : Nat → Option V) (kc : V → Option C) :
    ∀ (props : List Nat) (vals ws : List V) (k : List C), props.mapM attr = some ws →
      ws.mapM kc = some k → vals.mapM kc = some k →
      ∀ p ∈ props.zip vals, ∃ w c, attr p.1 = some w ∧ kc w = some c ∧ kc p.2 = some c
  | [], _, _, _, _, _, _, p, hp => by simp at hp
  | _ :: _, [], _, _, _, _, _, p, hp => by simp at hp
  | a :: props, v :: vals, ws, k, hw, hwk, hvk, p, hp => by
    obtain ⟨w, ws', hwa, hws', rfl⟩ := mapM_cons_some attr a props ws hw
    obtain ⟨c, k', hc, hk', rfl⟩ := mapM_cons_some kc w ws' k hwk
    obtain ⟨c2, k'', hc2, hk'', he⟩ := mapM_cons_some kc v vals _ hvk
    simp only [List.cons.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    simp only [List.zip_cons_cons, List.mem_cons] at hp
    rcases hp with rfl | hp
    · exact ⟨w, c, hwa, hc, hc2⟩
    · exact mapM_zip_key attr kc props vals ws' k' hws' hk' hk'' p hp

/-- **`redisearch_hIdx`**: `uniqueViolated_spec`'s lookup contract, for RediSearch. `hEq` is the
only assumption, about the index: an `Equal` on a constrained property (`p ∈ props`) and an
exactly matchable value `v` returns every live entity outside the commit whose value there has
`v`'s key component (an FFI fact: the index lags only this commit, matches such values
exactly, and holds Int, Float and Bool in one numeric field). `props ≠ []` (a constraint has
a property). -/
theorem redisearch_hIdx {C : Type} [DecidableEq C] (exact : V → Bool) (indexed : Bool)
    (props : List Nat) (attr : Nat → Nat → Option V) (kc : V → Option C)
    (affected isCand : Nat → Bool) (ix : Nat × V → List Nat)
    (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : V) c, exact v = true → kc v = some c → ∀ o, affected o = false →
      isCand o = true → ∀ w, attr o p = some w → kc w = some c → o ∈ ix (p, v)) :
    ∀ vals k q, vals.mapM kc = some k → redisearchQuery exact indexed props vals = some q →
      ∀ o, affected o = false → isCand o = true →
      keyOf props attr (·.mapM kc) o = some k → o ∈ evalQ ix q := by
  intro vals k q hvk hq o ha hc hk
  unfold keyOf at hk
  cases hws : props.mapM (attr o) with
  | none => rw [hws] at hk; cases hk
  | some ws =>
  rw [hws] at hk
  simp only [Option.bind_some] at hk
  have hz := mapM_zip_key (attr o) kc props vals ws k hws hk hvk
  have hlen : vals.length = props.length := by
    rw [← mapM_length kc vals k hvk, mapM_length kc ws k hk, mapM_length (attr o) props ws hws]
  unfold redisearchQuery at hq
  split at hq
  · cases hq
  rename_i hcond
  have hv : vals.all exact = true := by
    cases h : vals.all exact <;> simp_all
  have hex : ∀ p ∈ props.zip vals, exact p.2 = true := fun p hp =>
    List.all_eq_true.1 hv p.2 (List.of_mem_zip hp).2
  have hmem : ∀ p ∈ props.zip vals, o ∈ ix p := fun p hp => by
    obtain ⟨w, c, hw, hwc, hvc⟩ := hz p hp
    exact hEq p.1 (List.of_mem_zip hp).1 p.2 c (hex p hp) hvc o ha hc w hw hwc
  cases hs : props.zip vals with
  | nil =>
    exfalso; apply hprops
    cases props with
    | nil => rfl
    | cons a t =>
      cases vals with
      | nil => simp at hlen
      | cons b t' => simp at hs
  | cons p t =>
    rw [hs] at hq
    have hmem' : ∀ x ∈ p :: t, o ∈ ix x := fun x hx => hmem x (hs ▸ hx)
    cases t with
    | nil =>
      obtain ⟨k, v⟩ := p
      simp only at hq
      cases hq
      simpa [evalQ] using hmem' (k, v) (by simp)
    | cons p2 t2 =>
      simp only [Option.some.injEq] at hq
      subst hq
      simp only [evalQ, List.mem_filter, List.mem_flatMap, List.all_eq_true, decide_eq_true_eq]
      exact ⟨⟨p, by simp, hmem' p (by simp)⟩, fun x hx => hmem' x hx⟩

/-- **`checkUnique_redisearch_spec`**: the UNIQUE verdict on the RediSearch path is the
spec, assuming only the index's `Equal` exactness and that the scan lists every live
candidate with a key. -/
theorem checkUnique_redisearch_spec {C : Type} [DecidableEq C] (exact : V → Bool)
    (indexed : Bool) (props affected : List Nat)
    (member : Nat → Bool) (attr : Nat → Nat → Option V) (kc : V → Option C) (isCand : Nat → Bool)
    (ix : Nat × V → List Nat) (scan : List Nat) (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : V) c, exact v = true → kc v = some c → ∀ o, o ∉ affected →
      isCand o = true → ∀ w, attr o p = some w → kc w = some c → o ∈ ix (p, v))
    (hScan : ∀ o, o ∉ affected → isCand o = true →
      (keyOf props attr (·.mapM kc) o).isSome → o ∈ scan) :
    checkUnique props affected member attr (·.mapM kc) isCand
        (redisearchQuery exact indexed props) (evalQ ix) scan = true ↔
      ((affected.filter member).filterMap (keyOf props attr (·.mapM kc))).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, keyOf props attr (·.mapM kc) m = some k →
        ∀ o, o ∉ affected → isCand o = true → keyOf props attr (·.mapM kc) o ≠ some k :=
  checkUnique_spec props affected member attr _ isCand _ _ scan
    (fun vals k q hk hq o ho hc hok =>
      redisearch_hIdx exact indexed props attr kc (fun o => decide (o ∈ affected)) isCand ix
        hprops
        (fun p hpm v c hv hvc o ha hc' w hw hwc =>
          hEq p hpm v c hv hvc o (by simpa using ha) hc' w hw hwc)
        vals k q hk hq o (by simpa using ho) hc hok)
    hScan

/-- Not Range-indexed: every participating member is declined, so (when some member
participates) the label is scanned and no `get_indexed_*` call is made: the verdict is the
same for any lookup. -/
theorem redisearch_unindexed_scans {K : Type} [DecidableEq K] (exact : V → Bool)
    (props : List Nat) (members : List Nat)
    (affected : Nat → Bool) (attr : Nat → Nat → Option V) (ck : List V → Option K)
    (isCand : Nat → Bool) (lookup lookup' : Query V → List Nat) (scan : List Nat)
    (hm : ∃ m ∈ members, (pvals props attr ck m).isSome) :
    uniqueViolated props members affected attr ck isCand (redisearchQuery exact false props)
        lookup scan =
      uniqueViolated props members affected attr ck isCand (redisearchQuery exact false props)
        lookup' scan := by
  apply uniqueViolated_declined
  obtain ⟨m, hm, hs⟩ := hm
  obtain ⟨k, hk⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨k, List.mem_filterMap.2 ⟨m, hm, hk⟩,
    (redisearchQuery_none_iff exact false props k).2 (Or.inl rfl)⟩

end PendingCommit.UQ
