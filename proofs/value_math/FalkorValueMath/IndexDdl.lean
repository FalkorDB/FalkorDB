import FalkorValueMath.Value
/-!
# `graph/src/runtime/index_ddl.rs` (origin/main fe619ac5f)

`create_index` (:30), `drop_index` (:88), `emit_effect` (:116) as state transformers on
the parts of `Runtime` they touch (`RefCell`s are mutated in place and *not* rolled back
on `Err`, so a step returns its error together with the state reached). The graph's index
store, the options evaluator, `map_to_index_options`, `upgrade_to_write` and
`EffectsBuffer::build_index` are parameters (`DdlOps`).

Proved: read-only rejection leaves everything untouched; every failure before the write
upgrade leaves everything untouched; on success `indexes_created += attrs.len()` (create)
/ `indexes_dropped += dropped` (drop — the graph's count, not `attrs.len()`), exactly one
effect is appended iff `build_effects`, and `effects_count` counts appended effects
(`emit_counts`); a graph failure changes no statistic and emits no effect. Hazard
(`emit_buffer_created_on_error`): when `build_index` fails, the buffer has already been
created by `get_or_insert_with` (an empty buffer, no count) — harmless, it is discarded with
the failing query.

#3094 (`fe619ac5f`): a vector index whose options converted to nothing — no `OPTIONS`
map at all — is refused before the write upgrade (`:61-65`), so nothing is registered
(`create_vector_no_opts`). That is what closes W5-idx-2 (the half-created vector index
on error): live at fe619ac5f, `CREATE VECTOR INDEX FOR (n:L) ON (n.emb)` answers
"Invalid vector index configuration" on Rust and C alike and `db.indexes()` is unchanged.
-/

namespace ValueMath.Ddl

open ValueMath

variable {F G E O B : Type}

/-- The runtime state the DDL touches. `G` = graph index store, `B` = effects buffer. -/
structure St (G B : Type) where
  write : Bool
  buildEffects : Bool
  writer : Bool          -- write escalation reached
  created : Nat          -- stats.indexes_created
  dropped : Nat          -- stats.indexes_dropped
  buffer : Option B      -- effects_buffer
  effectsCount : Nat
  g : G

structure DdlOps (F G E O B : Type) where
  evalOpts : E → Except String (V F)
  toOpts : List (String × V F) → Except String (Option O)
  upgrade : Except String Unit
  gCreate : G → Option O → Except String G
  gDrop : G → Except String (G × Nat)
  newBuf : B
  buildIndex : B → Bool → Option (V F) → Except String B   -- create?, options

variable (D : DdlOps F G E O B)

def RO : String := "graph.RO_QUERY is to be executed only on read-only queries"

/-- `emit_effect` (:116). -/
def emit (s : St G B) (create : Bool) (opts : Option (V F)) : Option String × St G B :=
  if !s.buildEffects then (none, s) else
  let buf := s.buffer.getD D.newBuf                          -- get_or_insert_with
  let s1 := { s with buffer := some buf }
  match D.buildIndex buf create opts with
  | .error e => (some e, s1)
  | .ok buf' => (none, { s1 with buffer := some buf', effectsCount := s.effectsCount + 1 })

/-- The pure prefix of `create_index` (:46-60): evaluate `OPTIONS`, require a map, convert. -/
def prep (options : Option E) : Except String (Option (V F) × Option O) :=
  let ov : Except String (Option (V F)) := match options with
    | none => .ok none
    | some e => match D.evalOpts e with
      | .error err => .error err
      | .ok (.map m) => .ok (some (.map m))
      | .ok _ => .error "Index options must be a map"
  match ov with
  | .error err => .error err
  | .ok ov => match ov with
    | some (.map m) => (D.toOpts m).map fun io => (ov, io)
    | _ => .ok (ov, none)

def VEC : String := "Invalid vector index configuration"

/-- `create_index` (:30). `vec` is `*index_type == IndexType::Vector`. -/
def createIndex (s : St G B) (nAttrs : Nat) (vec : Bool) (options : Option E) : Option String × St G B :=
  if !s.write then (some RO, s) else
  match prep D options with
  | .error err => (some err, s)
  | .ok (ov, io) =>
  if vec && io.isNone then (some VEC, s) else
  match D.upgrade with
  | .error err => (some err, s)
  | .ok () =>
  match D.gCreate s.g io with
  | .error err => (some err, { s with writer := true })
  | .ok g' => emit D { s with writer := true, g := g', created := s.created + nAttrs } true ov

/-- `drop_index` (:88). -/
def dropIndex (s : St G B) : Option String × St G B :=
  if !s.write then (some RO, s) else
  match D.upgrade with
  | .error err => (some err, s)
  | .ok () =>
  let s := { s with writer := true }
  match D.gDrop s.g with
  | .error err => (some err, s)
  | .ok (g', n) => emit D { s with g := g', dropped := s.dropped + n } false none

/-! ## Theorems -/

theorem ro_rejects (s : St G B) (h : s.write = false) (n : Nat) (v : Bool) (o : Option E) :
    createIndex D s n v o = (some RO, s) ∧ dropIndex D s = (some RO, s) := by
  simp [createIndex, dropIndex, h]

theorem emit_off (s : St G B) (c : Bool) (o : Option (V F)) (h : s.buildEffects = false) :
    emit D s c o = (none, s) := by simp [emit, h]

/-- `effects_count` grows by one exactly when an effect was appended. -/
theorem emit_counts (s : St G B) (c : Bool) (o : Option (V F)) :
    (emit D s c o).2.effectsCount = s.effectsCount + (if s.buildEffects ∧ (emit D s c o).1 = none then 1 else 0) := by
  unfold emit
  cases hb : s.buildEffects <;> simp
  cases D.buildIndex (s.buffer.getD D.newBuf) c o <;> simp

/-- `emit_effect` touches only the buffer and the counter. -/
theorem emit_frame (s : St G B) (c : Bool) (o : Option (V F)) :
    let s' := (emit D s c o).2
    s'.g = s.g ∧ s'.created = s.created ∧ s'.dropped = s.dropped ∧ s'.writer = s.writer ∧ s'.write = s.write := by
  unfold emit; cases s.buildEffects <;> simp; cases D.buildIndex (s.buffer.getD D.newBuf) c o <;> simp

theorem emit_buffer_created_on_error (s : St G B) (c : Bool) (o : Option (V F)) (e : String)
    (hb : s.buildEffects = true) (he : D.buildIndex (s.buffer.getD D.newBuf) c o = .error e) :
    emit D s c o = (some e, { s with buffer := some (s.buffer.getD D.newBuf) }) := by
  simp [emit, hb, he]

/-- Failures before the graph is touched leave the state untouched: a non-map `OPTIONS`,
an evaluation error, an invalid option, a failed write upgrade. -/
theorem create_nonmap (s : St G B) (n : Nat) (vec : Bool) (e : E) (v : V F) (hw : s.write = true)
    (hv : D.evalOpts e = .ok v) (hm : ∀ m, v ≠ .map m) :
    createIndex D s n vec (some e) = (some "Index options must be a map", s) := by
  cases v <;> simp_all [createIndex, prep]

theorem create_eval_err (s : St G B) (n : Nat) (vec : Bool) (e : E) (err : String) (hw : s.write = true)
    (hv : D.evalOpts e = .error err) : createIndex D s n vec (some e) = (some err, s) := by
  simp [createIndex, prep, hw, hv]

theorem create_opts_err (s : St G B) (n : Nat) (vec : Bool) (e : E) (m : List (String × V F)) (err : String)
    (hw : s.write = true) (hv : D.evalOpts e = .ok (.map m)) (ho : D.toOpts m = .error err) :
    createIndex D s n vec (some e) = (some err, s) := by
  simp [createIndex, prep, hw, hv, ho, Except.map]

theorem create_upgrade_err (s : St G B) (n : Nat) (err : String) (hw : s.write = true)
    (hu : D.upgrade = .error err) : createIndex D s n false (none : Option E) = (some err, s) := by
  simp [createIndex, prep, hw, hu]

/-- **#3094**: a vector index with no options is refused before the write upgrade and
before `g.create_index` — the state, the graph's index store included, is untouched
(W5-idx-2 closed). -/
theorem create_vector_no_opts (s : St G B) (n : Nat) (hw : s.write = true) :
    createIndex D s n true (none : Option E) = (some VEC, s) := by
  simp [createIndex, prep, hw]

/-- …and so is any vector index whose options map converted to `None`. -/
theorem create_vector_none (s : St G B) (n : Nat) (e : E) (m : List (String × V F))
    (hw : s.write = true) (hv : D.evalOpts e = .ok (.map m)) (ho : D.toOpts m = .ok none) :
    createIndex D s n true (some e) = (some VEC, s) := by
  simp [createIndex, prep, hw, hv, ho, Except.map]

/-- On success: `created += attrs.len()`, the graph store is the `gCreate` result, the
query became a writer, and one effect is appended iff `build_effects`. -/
theorem create_ok (s : St G B) (n : Nat) (vec : Bool) (o : Option E) (h : (createIndex D s n vec o).1 = none) :
    let s' := (createIndex D s n vec o).2
    s.write = true ∧ s'.writer = true ∧ s'.created = s.created + n ∧ s'.dropped = s.dropped ∧
    s'.effectsCount = s.effectsCount + (if s.buildEffects then 1 else 0) := by
  intro s'
  unfold s' createIndex at *
  cases hw : s.write <;> simp only [hw, Bool.not_false, Bool.not_true, ite_true, ite_false, Bool.false_eq_true] at h ⊢
  · simp at h
  cases hp : prep D o with
  | error e => simp [hp] at h
  | ok p =>
    obtain ⟨ov, io⟩ := p
    simp only [hp] at h ⊢
    by_cases hvi : (vec && io.isNone) = true
    · simp [hvi] at h
    simp only [hvi, ite_false, Bool.false_eq_true] at h ⊢
    cases hu : D.upgrade with
    | error e => simp [hu] at h
    | ok u =>
      simp only [hu] at h ⊢
      cases hg : D.gCreate s.g io with
      | error e => simp [hg] at h
      | ok g' =>
        simp only [hg] at h ⊢
        have hf := emit_frame D ({ s with writer := true, g := g', created := s.created + n }) true ov
        have hc := emit_counts D ({ s with writer := true, g := g', created := s.created + n }) true ov
        simp_all

/-- A failing `g.create_index` changes no statistic and emits nothing (only the write
escalation has happened). -/
theorem create_graph_fail (s : St G B) (n : Nat) (err : String) (hw : s.write = true)
    (hu : D.upgrade = .ok ()) (hg : D.gCreate s.g none = .error err) :
    createIndex D s n false (none : Option E) = (some err, { s with writer := true }) := by
  simp [createIndex, prep, hw, hu, hg]

/-- `drop_index` success: `dropped += n` where `n` is what the graph reports. -/
theorem drop_ok (s : St G B) (g' : G) (k : Nat) (hw : s.write = true) (hu : D.upgrade = .ok ())
    (hg : D.gDrop s.g = .ok (g', k)) :
    (dropIndex D s).2.dropped = s.dropped + k ∧ (dropIndex D s).2.created = s.created ∧
    (dropIndex D s).2.g = g' := by
  simp only [dropIndex, hw, hu, hg, Bool.not_true, Bool.false_eq_true, if_false]
  have := emit_frame D ({ s with writer := true, g := g', dropped := s.dropped + k }) false none
  simp_all

end ValueMath.Ddl
