import PendingCommit.UniqueIndex
/-
# `redisearch_unique_query` (pending.rs:173) and the RediSearch `(query_for, lookup)` pair

`check_node_constraint` / `check_edge_constraint` without `index-falkordb` build
`query_for = |values| redisearch_unique_query(indexed, &constraint.properties, values)` and
`lookup = |query| g.get_indexed_nodes(label, query)` (edges: `get_indexed_edges`).

| Lean | Rust |
| --- | --- |
| `Query` (`equal` / `and`) | `IndexQuery::Equal` / `IndexQuery::And` |
| `redisearchQuery exact indexed props vals` | `redisearch_unique_query(indexed, properties, values)` (:173) |
| `evalQ ix` | `get_indexed_*` of a query: `Equal` is the index's exact-match set, `And` their intersection |

Results:
* `redisearchQuery_none_iff` — `None` iff the properties are not all Range-indexed or some
  value is not exactly matchable (`is_value_valid_for_index_lookup`).
* `redisearchQuery_single`, `redisearchQuery_and` — one `Equal` when the zip has exactly one
  pair (one constrained property), otherwise an `And` of every `Equal`.
* `redisearch_hIdx` — with the index's per-`Equal` exactness as the only hypothesis (an FFI
  fact about RediSearch, `hEq`: an `Equal` on `v` returns every live entity outside the commit
  whose value has `v`'s key component, numbers being stored as `f64`), the lookup of the query
  this fn builds returns every live candidate outside the commit carrying the key:
  `uniqueViolated_spec`'s `hIdx`, discharged. `zip_key`, `mapM_len` are its list lemmas.
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

theorem mapM_len {α β : Type} (f : α → Option β) : ∀ (l : List α) (r : List β),
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
        simp [mapM_len f l bs hbs]

/-- Pairing the zip a query is built from with the candidate's own values: when the member's
values `ws` and the candidate's values `wo` have the same key, each `(p, v)` of
`props.zip ws` has a candidate value `w` at `p` with the same key component. -/
theorem zip_key {W V : Type} (attr : Nat → Option W) (kv : W → Option V) :
    ∀ (props : List Nat) (ws wo : List W) (vals : List V),
      props.mapM attr = some wo → ws.mapM kv = some vals → wo.mapM kv = some vals →
      ∀ p ∈ props.zip ws, ∃ w, attr p.1 = some w ∧ ∃ c, kv w = some c ∧ kv p.2 = some c
  | [], _, _, _, _, _, _, p, hp => by simp at hp
  | _ :: _, [], _, _, _, _, _, p, hp => by simp at hp
  | a :: props, v :: ws, wo, vals, h1, h2, h3, p, hp => by
    rw [List.mapM_cons] at h1 h2
    cases ha : attr a with
    | none => rw [ha] at h1; simp at h1
    | some w =>
    cases hr : props.mapM attr with
    | none => rw [ha, hr] at h1; simp at h1
    | some wo' =>
    rw [ha, hr] at h1; simp at h1; subst h1
    cases hv : kv v with
    | none => rw [hv] at h2; simp at h2
    | some c =>
    cases hvs : ws.mapM kv with
    | none => rw [hv, hvs] at h2; simp at h2
    | some vals' =>
    rw [hv, hvs] at h2; simp at h2; subst h2
    rw [List.mapM_cons] at h3
    cases hw : kv w with
    | none => rw [hw] at h3; simp at h3
    | some c' =>
    cases hws : wo'.mapM kv with
    | none => rw [hw, hws] at h3; simp at h3
    | some vals'' =>
    rw [hw, hws] at h3
    have h3' : c' = c ∧ vals'' = vals' := by simpa using h3
    simp only [List.zip_cons_cons, List.mem_cons] at hp
    rcases hp with rfl | hp
    · exact ⟨w, ha, c, by rw [hw, h3'.1], hv⟩
    · exact zip_key attr kv props ws wo' vals' hr hvs (by rw [hws, h3'.2]) p hp

/-- **`redisearch_hIdx`**: `uniqueViolated_spec`'s lookup contract, for RediSearch. `hEq` is the
only assumption, about the index: an `Equal` on a constrained property (`p ∈ props`) and an
exactly matchable value `v` returns every live entity outside the commit whose value at `p`
has the same key component as `v` (an FFI fact: the index lags only this commit; numbers are
stored and matched as `f64`, so `1`, `1.0` and `true` are one entry, the same equality
`build_composite_key` uses since #2991; a string tag matches at least the identical string).
`props ≠ []` (a constraint has a property). -/
theorem redisearch_hIdx {W : Type} [DecidableEq V] (exact : W → Bool) (indexed : Bool) (props : List Nat)
    (attr : Nat → Nat → Option W) (kv : W → Option V) (affected isCand : Nat → Bool)
    (ix : Nat × W → List Nat)
    (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : W), exact v = true → ∀ o, affected o = false → isCand o = true →
      ∀ w c, attr o p = some w → kv w = some c → kv v = some c → o ∈ ix (p, v)) :
    ∀ ws q, redisearchQuery exact indexed props ws = some q → ∀ vals, ws.mapM kv = some vals →
      ∀ o, affected o = false → isCand o = true → keyOf props attr kv o = some vals →
      o ∈ evalQ ix q := by
  intro ws q hq vals hvals o ha hc hk
  rw [keyOf_eq] at hk
  cases hwo : props.mapM (attr o) with
  | none => rw [hwo] at hk; cases hk
  | some wo =>
  rw [hwo] at hk
  have hz := zip_key (attr o) kv props ws wo vals hwo hvals hk
  have hlen := mapM_len (attr o) props wo hwo
  have hlen2 : ws.length = wo.length :=
    (mapM_len kv ws vals hvals).symm.trans (mapM_len kv wo vals hk)
  unfold redisearchQuery at hq
  split at hq
  · cases hq
  rename_i hcond
  have hv : ws.all exact = true := by
    cases h : ws.all exact <;> simp_all
  have hex : ∀ p ∈ props.zip ws, exact p.2 = true := fun p hp =>
    List.all_eq_true.1 hv p.2 (List.of_mem_zip hp).2
  have hmem : ∀ p ∈ props.zip ws, o ∈ ix p := fun p hp => by
    obtain ⟨w, hw, c, hwc, hvc⟩ := hz p hp
    exact hEq p.1 (List.of_mem_zip hp).1 p.2 (hex p hp) o ha hc w c hw hwc hvc
  cases hs : props.zip ws with
  | nil =>
    exfalso; apply hprops
    cases props with
    | nil => rfl
    | cons a t =>
      cases ws with
      | nil => simp at hlen hlen2; omega
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
theorem checkUnique_redisearch_spec {W : Type} [DecidableEq V] (exact : W → Bool) (indexed : Bool)
    (props affected : List Nat) (member : Nat → Bool) (attr : Nat → Nat → Option W)
    (kv : W → Option V) (isCand : Nat → Bool)
    (ix : Nat × W → List Nat) (scan : List Nat) (hprops : props ≠ [])
    (hEq : ∀ p ∈ props, ∀ (v : W), exact v = true → ∀ o, o ∉ affected → isCand o = true →
      ∀ w c, attr o p = some w → kv w = some c → kv v = some c → o ∈ ix (p, v))
    (hScan : ∀ o, o ∉ affected → isCand o = true → (keyOf props attr kv o).isSome → o ∈ scan) :
    checkUnique props affected member attr kv isCand (redisearchQuery exact indexed props) (evalQ ix)
      scan = true ↔
      ((affected.filter member).filterMap (keyOf props attr kv)).Nodup ∧
      ∀ m ∈ affected.filter member, ∀ k, keyOf props attr kv m = some k →
        ∀ o, o ∉ affected → isCand o = true → keyOf props attr kv o ≠ some k :=
  checkUnique_spec props affected member attr kv isCand _ _ scan
    (fun ws q hq vals hv o ho hc hk =>
      redisearch_hIdx exact indexed props attr kv (fun o => decide (o ∈ affected)) isCand ix hprops
        (fun p hpm v hv o ha hc' w c hw hwc hvc => hEq p hpm v hv o (by simpa using ha) hc' w c hw hwc hvc)
        ws q hq vals hv o (by simpa using ho) hc hk)
    hScan

/-- Not Range-indexed: every member is declined, so (when some member participates) the
label is scanned and no `get_indexed_*` call is made: the verdict is the same for any lookup. -/
theorem redisearch_unindexed_scans {W : Type} [DecidableEq V] (exact : W → Bool) (props : List Nat)
    (members : List Nat) (affected : Nat → Bool) (attr : Nat → Nat → Option W) (kv : W → Option V)
    (isCand : Nat → Bool) (lookup lookup' : Query W → List Nat) (scan : List Nat)
    (hm : ∃ m ∈ members, (keyOf props attr kv m).isSome) :
    uniqueViolated props members affected attr kv isCand (redisearchQuery exact false props) lookup scan =
      uniqueViolated props members affected attr kv isCand (redisearchQuery exact false props) lookup' scan := by
  apply uniqueViolated_declined
  obtain ⟨m, hm, hs⟩ := hm
  cases he : entOf props attr kv m with
  | none => simp [keyOf, he] at hs
  | some p =>
    exact ⟨p, List.mem_filterMap.2 ⟨m, hm, he⟩, (redisearchQuery_none_iff exact false props p.1).2 (Or.inl rfl)⟩

end PendingCommit.UQ
