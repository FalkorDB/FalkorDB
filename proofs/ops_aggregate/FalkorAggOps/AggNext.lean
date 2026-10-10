import FalkorAggOps.AggState
/-
`AggregateOp::next` (aggregate.rs:1111) and the bulk column extractors
`extract_key_columns` (:781) / `extract_agg_input_columns` (:830).

| here | there |
| --- | --- |
| `keyCol`, `extractKeys`   | `extract_key_columns` (aggregate.rs:781-822) |
| `aggCol`, `extractAggs`   | `extract_agg_input_columns` (aggregate.rs:830-864) |
| `finishGroup`             | the per-group closure of `next` (aggregate.rs:1141-1213) |
| `aggNext`                 | `Iterator::next` (aggregate.rs:1111-1222) after consumption |

`batch.value_at`, `extract_node_ids`, `materialize_node_property_values` and
`VectorEval::eval_values` are parameters; the expression evaluator (`ExprEval::eval`) is a
parameter `ev`. Group iteration order is the `HashMap`'s (any list).
-/
namespace FalkorAggOps.AggPlan

variable {Z : Type}

/-! ## Column extraction -/

section Extract
variable {N : Type} (null : Z) (valueAt : Nat → Nat → Option Z)
  (nodeIds : Nat → Option (Nat → N)) (props : List N → Nat → List Z)

/-- One key column (aggregate.rs:786-819). `.error ()` = "this batch cannot take the bulk path". -/
def keyCol (evalC : Ex Z → Except Unit (List Z)) (active : List Nat) : KeyKind Z → Except Unit (List Z)
  | .var v => .ok (active.map fun r => (valueAt v r).getD null)
  | .prop v attr => match nodeIds v with
    | none => .error ()
    | some ids => .ok (props (active.map ids) attr)
  | .computed e => evalC e

def extractKeys (evalC : Ex Z → Except Unit (List Z)) (active : List Nat) (ks : List (KeyKind Z)) :
    Except Unit (List (List Z)) :=
  ks.mapM (keyCol null valueAt nodeIds props evalC active)

/-- One aggregate input column (aggregate.rs:837-861); `count(*)` gets an empty placeholder. -/
def aggCol (evalC : Ex Z → Except String (List Z)) (active : List Nat) : Option (Input Z) → Except String (List Z)
  | none => .ok []
  | some (.var v) => .ok (active.map fun r => (valueAt v r).getD null)
  | some (.computed e) => evalC e

def extractAggs (evalC : Ex Z → Except String (List Z)) (active : List Nat) (as : List (VAgg Z)) :
    Except String (List (List Z)) :=
  as.mapM (fun a => aggCol null valueAt evalC active a.input)

theorem mapM_ok_get {α β ε : Type} (f : α → Except ε β) (xs : List α) (ys : List β)
    (h : xs.mapM f = .ok ys) : ys.length = xs.length ∧ ∀ i (h1 : i < xs.length) (h2 : i < ys.length),
      f xs[i] = .ok ys[i] := by
  induction xs generalizing ys with
  | nil => simp [List.mapM_nil, pure, Except.pure] at h; subst h; simp
  | cons x xs ih =>
    simp only [List.mapM_cons, bind, Except.bind] at h
    cases hx : f x with
    | error e => simp [hx] at h
    | ok y =>
      simp only [hx] at h
      cases hxs : xs.mapM f with
      | error e => simp [hxs] at h
      | ok ys' =>
        simp only [hxs, pure, Except.pure, Except.ok.injEq] at h
        subst h
        obtain ⟨hl, hg⟩ := ih ys' hxs
        refine ⟨by simp [hl], fun i h1 h2 => ?_⟩
        cases i with
        | zero => simpa using hx
        | succ i => simpa using hg i (by simpa using h1) (by simpa using h2)

/-- **`extract_key_columns`**: on success, one column per key kind, in order; a
variable key's column is that variable's value at each active row (`Null` if out of range);
a property key reads the node id at each active row. -/
theorem extractKeys_spec (evalC : Ex Z → Except Unit (List Z)) (active : List Nat)
    (ks : List (KeyKind Z)) (cols : List (List Z))
    (h : extractKeys null valueAt nodeIds props evalC active ks = .ok cols) :
    cols.length = ks.length ∧ ∀ i (h1 : i < ks.length) (h2 : i < cols.length),
      keyCol null valueAt nodeIds props evalC active ks[i] = .ok cols[i] ∧
      (∀ v, ks[i] = .var v → cols[i] = active.map fun r => (valueAt v r).getD null) := by
  obtain ⟨hl, hg⟩ := mapM_ok_get _ _ _ h
  refine ⟨hl, fun i h1 h2 => ⟨hg i h1 h2, fun v hv => ?_⟩⟩
  have := hg i h1 h2
  rw [hv] at this; simp [keyCol] at this; exact this.symm

/-- A failing key kind fails the whole batch (it then takes the per-row path). -/
theorem extractKeys_err (evalC : Ex Z → Except Unit (List Z)) (active : List Nat)
    (ks : List (KeyKind Z)) (i : Nat) (hi : i < ks.length)
    (he : keyCol null valueAt nodeIds props evalC active ks[i] = .error ()) :
    extractKeys null valueAt nodeIds props evalC active ks = .error () := by
  cases h : extractKeys null valueAt nodeIds props evalC active ks with
  | error e => cases e; rfl
  | ok cols =>
    obtain ⟨hl, hg⟩ := mapM_ok_get _ _ _ h
    have := hg i hi (by omega); rw [he] at this; cases this

/-- **`extract_agg_input_columns`**: one column per aggregate, `[]` for `count(*)`, the
variable's values for a variable input, the evaluator's column otherwise; an evaluation error
is returned as is. -/
theorem extractAggs_spec (evalC : Ex Z → Except String (List Z)) (active : List Nat)
    (as : List (VAgg Z)) (cols : List (List Z))
    (h : extractAggs null valueAt evalC active as = .ok cols) :
    cols.length = as.length ∧ ∀ i (h1 : i < as.length) (h2 : i < cols.length),
      aggCol null valueAt evalC active as[i].input = .ok cols[i] ∧
      (as[i].input = none → cols[i] = []) := by
  obtain ⟨hl, hg⟩ := mapM_ok_get _ _ _ h
  refine ⟨hl, fun i h1 h2 => ⟨hg i h1 h2, fun hn => ?_⟩⟩
  have := hg i h1 h2
  rw [hn] at this; simp [aggCol] at this; exact this

end Extract

/-! ## Per-group output row (aggregate.rs:1141-1213) -/

/-- `Row::merge`: every bound slot of `other` overwrites. -/
def mergeE (self other : Env Z) : Env Z := fun j => match other j with
  | some v => some v
  | none => self j

def updAll (env : Env Z) (l : List (Nat × Z)) : Env Z := l.foldl (fun a p => upd a p.1 p.2) env

/-- A key = (output name, `Some original_var` when the key tree is a bare variable). -/
structure KeySpec where
  name : Nat
  orig : Option Nat

def finishGroup {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc : Env Z) : Except E (Env Z) := do
  -- combined = key + pre-projection aliases, then acc on top
  let combined0 := keys.foldl (fun c k => match k.orig, key k.name with
    | some o, some v => upd c o v
    | _, _ => c) key
  let combined1 := mergeE combined0 acc
  -- evaluate every aggregation expression in order
  let (acc1, _, outs) ← aggs.foldlM (fun (st : Env Z × Env Z × List (Nat × Z)) (p : Nat × Ex Z) => do
      let val ← ev p.2 st.2.1
      pure (upd st.1 p.1 val, upd st.2.1 p.1 val, st.2.2 ++ [(p.1, val)]))
    (acc, combined1, [])
  -- pre-projection key aliases, unless an aggregate output owns the slot
  let acc2 := keys.foldl (fun a k => match k.orig, key k.name with
    | some o, some v => if aggs.any (·.1 == o) then a else upd a o v
    | _, _ => a) acc1
  let acc3 := mergeE acc2 key
  let acc4 := aggs.foldl (fun a p => unbindAcc p.2 a) acc3
  pure (updAll acc4 outs)

theorem updAll_get (env : Env Z) (l : List (Nat × Z)) (n : Nat) (v : Z)
    (hn : (l.map Prod.fst).Nodup) (hm : (n, v) ∈ l) : updAll env l n = some v := by
  induction l generalizing env with
  | nil => simp at hm
  | cons p ps ih =>
    simp only [updAll, List.foldl_cons] at *
    simp only [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.mp hm with rfl | hm'
    · -- p = (n, v); later writes never touch n
      have keep : ∀ (l : List (Nat × Z)) (e : Env Z), (∀ q ∈ l, q.1 ≠ n) → e n = some v →
          l.foldl (fun a p => upd a p.1 p.2) e n = some v := by
        intro l; induction l with
        | nil => intro e _ he; exact he
        | cons q qs ihq =>
          intro e hq he; simp only [List.foldl_cons]
          apply ihq _ (fun r hr => hq r (List.mem_cons_of_mem _ hr))
          have := hq q (List.mem_cons_self ..)
          simp [upd, Ne.symm this, he]
      apply keep ps _ (fun q hq he => hn.1 (by have := List.mem_map_of_mem (f := Prod.fst) hq; rw [he] at this; exact this))
      simp [upd]
    · exact ih _ hn.2 hm'

theorem foldlM_outs {E : Type} (ev : Ex Z → Env Z → Except E Z) (aggs : List (Nat × Ex Z))
    (st : Env Z × Env Z × List (Nat × Z)) (r : Env Z × Env Z × List (Nat × Z))
    (h : aggs.foldlM (fun (st : Env Z × Env Z × List (Nat × Z)) (p : Nat × Ex Z) => do
      let val ← ev p.2 st.2.1
      pure (upd st.1 p.1 val, upd st.2.1 p.1 val, st.2.2 ++ [(p.1, val)])) st = .ok r) :
    r.2.2.map Prod.fst = st.2.2.map Prod.fst ++ aggs.map Prod.fst := by
  induction aggs generalizing st with
  | nil => simp [List.foldlM_nil, pure, Except.pure] at h; subst h; simp
  | cons p ps ih =>
    simp only [List.foldlM_cons, bind, Except.bind] at h
    cases hv : ev p.2 st.2.1 with
    | error e => simp [hv] at h
    | ok val =>
      simp only [hv, pure, Except.pure] at h
      rw [ih _ h]; simp

/-- **Output bindings win** (aggregate.rs:1203-1210): on success, with distinct output
names, every aggregate output name is bound to the value computed for it — even when an
accumulator slot or a key alias shares its id. -/
theorem finishGroup_outputs {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (aggs : List (Nat × Ex Z)) (key acc out : Env Z)
    (hnd : (aggs.map Prod.fst).Nodup)
    (h : finishGroup ev keys aggs key acc = .ok out) :
    ∃ outs : List (Nat × Z), outs.map Prod.fst = aggs.map Prod.fst ∧
      ∀ n v, (n, v) ∈ outs → out n = some v := by
  unfold finishGroup at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i r hr
    simp only [pure, Except.pure, Except.ok.injEq] at h
    have hl := foldlM_outs ev aggs _ r hr
    simp only [List.map_nil, List.nil_append] at hl
    refine ⟨r.2.2, hl, fun n v hm => ?_⟩
    rw [← h]
    exact updAll_get _ _ n v (by rw [hl]; exact hnd) hm

/-- An evaluation error of any aggregation expression aborts the group with that error. -/
theorem finishGroup_err {E : Type} (ev : Ex Z → Env Z → Except E Z) (keys : List KeySpec)
    (key acc : Env Z) (n : Nat) (t : Ex Z) (e : E)
    (he : ∀ env, ev t env = .error e) :
    finishGroup ev keys [(n, t)] key acc = .error e := by
  simp [finishGroup, bind, Except.bind, List.foldlM_cons, he]

/-! ## `next` after consumption (aggregate.rs:1133-1222) -/

def BATCH_SIZE : Nat := 1024

/-- Pack up to `n` finished groups; the first failing group aborts the call
(`return Some(Err(e))`, aggregate.rs:1214). Returns (packed rows or error, rest). -/
def pack {G R E : Type} (fin : G → Except E R) : Nat → List G → List R → Except E (List R) × List G
  | 0, gs, acc => (.ok acc, gs)
  | _, [], acc => (.ok acc, [])
  | n + 1, g :: gs, acc => match fin g with
    | .ok r => pack fin n gs (acc ++ [r])
    | .error e => (.error e, gs)

structure NextSt (E G : Type) where
  errors : List E
  groups : List G

/-- One call: drain an error first, then emit up to `BATCH_SIZE` groups, `None` when empty. -/
def aggNext {G R E : Type} (fin : G → Except E R) (s : NextSt E G) :
    Option (Except E (List R)) × NextSt E G :=
  match s.errors with
  | e :: es => (some (.error e), { s with errors := es })
  | [] =>
    match pack fin BATCH_SIZE s.groups [] with
    | (.error e, gs) => (some (.error e), { s with groups := gs })
    | (.ok rows, gs) => (if rows.isEmpty then none else some (.ok rows), { s with groups := gs })

/-- Errors are reported first, one per call, groups untouched. -/
theorem aggNext_error_first {G R E : Type} (fin : G → Except E R) (e : E) (es : List E) (gs : List G) :
    aggNext fin ⟨e :: es, gs⟩ = (some (.error e), ⟨es, gs⟩) := rfl

theorem pack_ok {G R E : Type} (fin : G → Except E R) (f : G → R) (hok : ∀ g, fin g = .ok (f g)) :
    ∀ n gs acc, pack fin n gs acc =
      (.ok (acc ++ (gs.take n).map fun g => f g), gs.drop n)
  | 0, gs, acc => by simp [pack]
  | n + 1, [], acc => by simp [pack]
  | n + 1, g :: gs, acc => by
    simp only [pack, hok g]
    rw [pack_ok fin f hok n gs (acc ++ [f g])]
    simp

/-- With error-free finishing, a call emits exactly the next `min(1024, remaining)` groups
in iteration order and returns `None` only when no group remains. -/
theorem aggNext_ok {G R E : Type} (fin : G → Except E R) (f : G → R) (hok : ∀ g, fin g = .ok (f g))
    (gs : List G) :
    aggNext fin ⟨[], gs⟩ =
      (if gs.isEmpty then none
       else some (.ok ((gs.take BATCH_SIZE).map fun g => f g)),
       ⟨[], gs.drop BATCH_SIZE⟩) := by
  unfold aggNext
  simp only [pack_ok fin f hok, List.nil_append]
  cases gs with
  | nil => rfl
  | cons g gs => simp [BATCH_SIZE]

/-- Draining `next` to exhaustion yields every group exactly once, in order. -/
def drain {G R : Type} (f : G → R) : Nat → List G → List R
  | 0, _ => []
  | fuel + 1, gs => if gs.isEmpty then [] else
      (gs.take BATCH_SIZE).map (fun g => f g) ++ drain f fuel (gs.drop BATCH_SIZE)

theorem drain_all {G R : Type} (f : G → R) :
    ∀ fuel (gs : List G), gs.length ≤ fuel * BATCH_SIZE →
      drain f fuel gs = gs.map fun g => f g
  | 0, gs, h => by
    have : gs = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.mp (by simpa using h))
    subst this; rfl
  | fuel + 1, gs, h => by
    unfold drain
    cases gs with
    | nil => rfl
    | cons g gs' =>
      simp only [List.isEmpty_cons, Bool.false_eq_true, ite_false]
      have h' : (g :: gs').length ≤ fuel * BATCH_SIZE + BATCH_SIZE := by
        simpa [Nat.add_mul] using h
      rw [drain_all f fuel _ (by rw [List.length_drop]; omega)]
      rw [← List.map_append, List.take_append_drop]

end FalkorAggOps.AggPlan
