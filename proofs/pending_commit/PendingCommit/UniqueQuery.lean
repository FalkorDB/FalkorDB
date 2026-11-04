import PendingCommit.UniqueIndex
/-
# `redisearch_unique_query` (pending.rs:174) and the RediSearch `(query_for, lookup)` pair

`check_node_constraint` / `check_edge_constraint` without `index-falkordb` build
`query_for = |values| redisearch_unique_query(indexed, &constraint.properties, values)` and
`lookup = |query| g.get_indexed_nodes(label, query)` (edges: `get_indexed_edges`).

| Lean | Rust |
| --- | --- |
| `Query` (`equal` / `and`) | `IndexQuery::Equal` / `IndexQuery::And` |
| `redisearchQuery exact indexed props vals` | `redisearch_unique_query(indexed, properties, values)` (:174) |
| `evalQ ix` | `get_indexed_*` of a query: `Equal` is the index's exact-match set, `And` their intersection |

Results:
* `redisearchQuery_none_iff` — `None` iff the properties are not all Range-indexed or some
  value is not exactly matchable (`is_value_valid_for_index_lookup`).
* `redisearchQuery_single`, `redisearchQuery_and` — one `Equal` when the zip has exactly one
  pair (one constrained property), otherwise an `And` of every `Equal`.
* `redisearch_hIdx` — with the index's per-`Equal` exactness as the only hypothesis (an FFI
  fact about RediSearch, `hEq`), the lookup of the query this fn builds returns every live
  candidate outside the commit carrying the key: `uniqueViolated_spec`'s `hIdx`, discharged.
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
  rw [if_neg (by simp [hv])]
  congr 1
  split
  · rename_i k v h; rw [h] at hn; simp at hn
  · rfl

/-- The result of running a query on an index whose `Equal` lookups are `ix`; `And` is the
intersection of its `Equal`s. -/
def evalQ (ix : Nat × V → List Nat) : Query V → List Nat
  | .equal k v => ix (k, v)
  | .and eqs => (eqs.flatMap ix).filter fun o => eqs.all fun p => decide (o ∈ ix p)

/-- `mapM` over `props` pairs each property with the value read. -/
theorem mapM_zip (attr : Nat → Option V) : ∀ (props : List Nat) (vals : List V),
    props.mapM attr = some vals → ∀ p ∈ props.zip vals, attr p.1 = some p.2
  | [], vals, h, p, hp => by
    simp [List.mapM_nil] at h; subst h; simp at hp
  | a :: props, vals, h, p, hp => by
    rw [List.mapM_cons] at h
    cases hb : attr a with
    | none => rw [hb] at h; simp at h
    | some b =>
      cases hbs : props.mapM attr with
      | none => rw [hb, hbs] at h; simp at h
      | some bs =>
        rw [hb, hbs] at h
        simp at h
        subst h
        simp only [List.zip_cons_cons, List.mem_cons] at hp
        rcases hp with rfl | hp
        · exact hb
        · exact mapM_zip attr props bs hbs p hp

theorem mapM_length (attr : Nat → Option V) : ∀ (props : List Nat) (vals : List V),
    props.mapM attr = some vals → vals.length = props.length
  | [], vals, h => by simp [List.mapM_nil] at h; subst h; rfl
  | a :: props, vals, h => by
    rw [List.mapM_cons] at h
    cases hb : attr a with
    | none => rw [hb] at h; simp at h
    | some b =>
      cases hbs : props.mapM attr with
      | none => rw [hb, hbs] at h; simp at h
      | some bs =>
        rw [hb, hbs] at h
        simp at h
        subst h
        simp [mapM_length attr props bs hbs]

/-- **`redisearch_hIdx`**: `uniqueViolated_spec`'s lookup contract, for RediSearch. `hEq` is the
only assumption, about the index: an `Equal` on a constrained property (`p ∈ props`) and an exactly matchable value
returns every live entity outside the commit holding that value (an FFI fact: the index lags
only this commit and matches exactly such values). `props ≠ []` (a constraint has a property). -/
theorem redisearch_hIdx (exact : V → Bool) (indexed : Bool) (props : List Nat)
    (attr : Nat → Nat → Option V) (affected isCand : Nat → Bool) (ix : Nat × V → List Nat)
    (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : V), exact v = true → ∀ o, affected o = false → isCand o = true →
      attr o p = some v → o ∈ ix (p, v)) :
    ∀ vals q, redisearchQuery exact indexed props vals = some q → ∀ o, affected o = false →
      isCand o = true → props.mapM (attr o) = some vals → o ∈ evalQ ix q := by
  intro vals q hq o ha hc hk
  have hz := mapM_zip (attr o) props vals hk
  have hlen := mapM_length (attr o) props vals hk
  unfold redisearchQuery at hq
  split at hq
  · cases hq
  rename_i hcond
  have hv : vals.all exact = true := by
    cases h : vals.all exact <;> simp_all
  have hex : ∀ p ∈ props.zip vals, exact p.2 = true := fun p hp =>
    List.all_eq_true.1 hv p.2 (List.of_mem_zip hp).2
  have hmem : ∀ p ∈ props.zip vals, o ∈ ix p := fun p hp =>
    hEq p.1 (List.of_mem_zip hp).1 p.2 (hex p hp) o ha hc (hz p hp)
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
spec, assuming only the index's `Equal` exactness and that the scan lists every live member. -/
theorem checkUnique_redisearch_spec [DecidableEq V] (exact : V → Bool) (indexed : Bool) (props affected : List Nat)
    (member : Nat → Bool) (attr : Nat → Nat → Option V) (isCand : Nat → Bool)
    (ix : Nat × V → List Nat) (scan : List Nat) (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : V), exact v = true → ∀ o, o ∉ affected → isCand o = true →
      attr o p = some v → o ∈ ix (p, v))
    (hScan : ∀ o, o ∉ affected → isCand o = true → (props.mapM (attr o)).isSome → o ∈ scan) :
    checkUnique props affected member attr isCand (redisearchQuery exact indexed props) (evalQ ix) scan
      = true ↔
      ((affected.filter member).filterMap fun e => props.mapM (attr e)).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, props.mapM (attr m) = some k →
        ∀ o, o ∉ affected → isCand o = true → props.mapM (attr o) ≠ some k :=
  checkUnique_spec props affected member attr isCand _ _ scan
    (fun vals q hq o ho hc hk =>
      redisearch_hIdx exact indexed props attr (fun o => decide (o ∈ affected)) isCand ix hprops
        (fun p hpm v hv o ha hc' hp => hEq p hpm v hv o (by simpa using ha) hc' hp) vals q hq o
        (by simpa using ho) hc hk)
    hScan

/-- Not Range-indexed: every member is declined, so (when some member participates) the
label is scanned and no `get_indexed_*` call is made: the verdict is the same for any lookup. -/
theorem redisearch_unindexed_scans [DecidableEq V] (exact : V → Bool) (props : List Nat) (members : List Nat)
    (affected : Nat → Bool) (attr : Nat → Nat → Option V) (isCand : Nat → Bool)
    (lookup lookup' : Query V → List Nat) (scan : List Nat)
    (hm : ∃ m ∈ members, (props.mapM (attr m)).isSome) :
    uniqueViolated props members affected attr isCand (redisearchQuery exact false props) lookup scan =
      uniqueViolated props members affected attr isCand (redisearchQuery exact false props) lookup' scan := by
  apply uniqueViolated_declined
  obtain ⟨m, hm, hs⟩ := hm
  obtain ⟨k, hk⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨k, List.mem_filterMap.2 ⟨m, hm, hk⟩, (redisearchQuery_none_iff exact false props k).2 (Or.inl rfl)⟩

end PendingCommit.UQ
