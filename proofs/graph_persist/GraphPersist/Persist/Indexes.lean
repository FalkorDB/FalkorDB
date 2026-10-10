/-! # `rebuild_indexes` (decoder/mod.rs:345-400)

After `Graph::restore`, each saved `IndexInfo` is replayed as `create_index_sync` calls: one
per `(attribute, field)` in `field_order` order (an attribute missing from `fields` is
skipped), the entity type taken from `entity_type == "RELATIONSHIP"`, vector fields with
their vector options, and the index-level language/stopwords riding on the **first**
full-text field only (`create_index` rejects them once the label has a full-text field).

* `rebuild_calls` — the calls are exactly the flattened `(attribute, field)` list.
* `rebuild_entity` — every call uses the saved entity type.
* `rebuild_meta_first` — the first non-vector full-text field carries the language and
  stopwords (when either was saved); every other field keeps its own options.
-/
namespace GraphPersist.Persist.Indexes

inductive Ty where
  | range | fulltext | vector
  deriving DecidableEq, Repr

inductive Ent where
  | node | relationship
  deriving DecidableEq, Repr

/-- `TextIndexOptions`: language and stopwords plus whatever else the field saved. -/
structure TOpts (L S X : Type) where
  language : Option L
  stopwords : Option S
  rest : X

structure Field (L S X V : Type) where
  ty : Ty
  vec : Option V              -- `field.vector_options()`
  opts : Option (TOpts L S X) -- `field.options()`

inductive Opt (L S X V : Type) where
  | text (t : TOpts L S X)
  | vector (v : V)

structure Info (A L S X V : Type) where
  label : String
  entityType : String
  language : Option L
  stopwords : Option S
  fieldOrder : List A
  fields : A → Option (List (Field L S X V))

structure Call (A L S X V : Type) where
  ty : Ty
  ent : Ent
  label : String
  attr : A
  opts : Option (Opt L S X V)

variable {A L S X V : Type} [Inhabited X]

def entOf (s : String) : Ent := if s = "RELATIONSHIP" then .relationship else .node

/-- `topts.language = info.language; topts.stopwords = info.stopwords` (`:379-380`). -/
def withMeta (info : Info A L S X V) (t : TOpts L S X) : TOpts L S X :=
  ⟨info.language, info.stopwords, t.rest⟩

/-- The options of one field, given whether the index-level text metadata is still pending;
returns the new pending flag (`:374-386`). -/
def fieldOpts (info : Info A L S X V) (pending : Bool) (f : Field L S X V) : Option (Opt L S X V) × Bool :=
  match f.vec with
  | some v => (some (.vector v), pending)
  | none =>
    if pending && f.ty == .fulltext then
      (some (.text (withMeta info (f.opts.getD ⟨none, none, default⟩))), false)
    else (f.opts.map .text, pending)

/-- The `(attribute, field)` pairs in visiting order. -/
def flat (info : Info A L S X V) : List (A × Field L S X V) :=
  info.fieldOrder.flatMap fun a => ((info.fields a).getD []).map fun f => (a, f)

def go (info : Info A L S X V) (ent : Ent) : Bool → List (A × Field L S X V) → List (Call A L S X V)
  | _, [] => []
  | p, (a, f) :: rest =>
    let r := fieldOpts info p f
    ⟨f.ty, ent, info.label, a, r.1⟩ :: go info ent r.2 rest

/-- `rebuild_indexes` for one `IndexInfo`. -/
def rebuild (info : Info A L S X V) : List (Call A L S X V) :=
  go info (entOf info.entityType) (info.language.isSome || info.stopwords.isSome) (flat info)

theorem go_shape (info : Info A L S X V) (ent : Ent) :
    ∀ (p : Bool) (l : List (A × Field L S X V)),
    (go info ent p l).map (fun c => (c.attr, c.ty, c.ent, c.label)) = l.map fun af => (af.1, af.2.ty, ent, info.label)
  | _, [] => rfl
  | p, (a, f) :: rest => by simp [go, go_shape info ent _ rest]

/-- **One call per saved (attribute, field)**, in `field_order`, each with the field's type,
the saved label and the saved entity type. -/
theorem rebuild_calls (info : Info A L S X V) :
    (rebuild info).map (fun c => (c.attr, c.ty, c.ent, c.label)) =
      (flat info).map fun af => (af.1, af.2.ty, entOf info.entityType, info.label) :=
  go_shape _ _ _ _

theorem rebuild_entity (info : Info A L S X V) (c : Call A L S X V) (hc : c ∈ rebuild info) :
    c.ent = (if info.entityType = "RELATIONSHIP" then .relationship else .node) := by
  have := congrArg (List.map fun x : A × Ty × Ent × String => x.2.2.1) (rebuild_calls info)
  simp only [List.map_map] at this
  have hm : c.ent ∈ (rebuild info).map (fun c => c.ent) := List.mem_map.2 ⟨c, hc, rfl⟩
  have e2 : (rebuild info).map (fun c => c.ent) = (flat info).map (fun _ => entOf info.entityType) := by
    simpa [Function.comp_def] using this
  rw [e2] at hm
  obtain ⟨_, _, h⟩ := List.mem_map.1 hm
  rw [← h]; rfl

/-- A field that takes the text metadata. -/
def Takes (f : Field L S X V) : Prop := f.vec = none ∧ f.ty = .fulltext

instance (f : Field L S X V) : Decidable (Takes f) := by unfold Takes; infer_instance

/-- What a field gets when the metadata is not (or no longer) pending: its own options. -/
def own (f : Field L S X V) : Option (Opt L S X V) :=
  match f.vec with
  | some v => some (.vector v)
  | none => f.opts.map .text

theorem go_not_pending (info : Info A L S X V) (ent : Ent) :
    ∀ l : List (A × Field L S X V), (go info ent false l).map (·.opts) = l.map fun af => own af.2
  | [] => rfl
  | (a, f) :: rest => by
    cases hv : f.vec <;> simp [go, fieldOpts, own, hv, go_not_pending info ent rest]

/-- **Language and stopwords go to the first full-text field only.** With metadata
pending, the fields before the first `Takes` field keep their own options, that field gets
its options with the index's language/stopwords, and every later field keeps its own. -/
theorem rebuild_meta_first (info : Info A L S X V) (ent : Ent) :
    ∀ (pre : List (A × Field L S X V)) (a : A) (f : Field L S X V) (post : List (A × Field L S X V)),
    (∀ af ∈ pre, ¬ Takes af.2) → Takes f →
    (go info ent true (pre ++ (a, f) :: post)).map (·.opts) =
      pre.map (fun af => own af.2) ++
        some (.text (withMeta info (f.opts.getD ⟨none, none, default⟩))) :: post.map fun af => own af.2
  | [], a, f, post, _, ⟨hv, ht⟩ => by
    simp [go, fieldOpts, hv, ht, go_not_pending info ent post]
  | (b, f0) :: pre, a, f, post, hpre, htk => by
    have h0 : ¬ Takes f0 := hpre (b, f0) (by simp)
    have ih := rebuild_meta_first info ent pre a f post (fun af h => hpre af (by simp [h])) htk
    cases hv : f0.vec with
    | some v => simp [go, fieldOpts, hv, ih, own]
    | none =>
      have hty : f0.ty ≠ .fulltext := fun e => h0 ⟨hv, e⟩
      have : (f0.ty == Ty.fulltext) = false := by simp [hty]
      simp [go, fieldOpts, hv, this, ih, own]

/-- Without saved language/stopwords, every field is rebuilt with its own options. -/
theorem rebuild_no_meta (info : Info A L S X V) (h : info.language = none ∧ info.stopwords = none) :
    (rebuild info).map (·.opts) = (flat info).map fun af => own af.2 := by
  simp [rebuild, h.1, h.2, go_not_pending]

end GraphPersist.Persist.Indexes
