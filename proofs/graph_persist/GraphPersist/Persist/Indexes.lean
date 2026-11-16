/-! # `rebuild_indexes` (decoder/mod.rs:345-441)

After `Graph::restore`, each saved `IndexInfo` is replayed as `create_index_sync` calls: one
per `(attribute, field)` in `field_order` order (an attribute missing from `fields` is
skipped), the entity type taken from `entity_type == "RELATIONSHIP"`, vector fields with
their vector options, and the index-level language/stopwords riding on the **first**
full-text field only (`create_index` rejects them once the label has a full-text field).

* `rebuild_calls` — the calls are exactly the flattened `(attribute, field)` list.
* `rebuild_entity` — every call uses the saved entity type.
* `rebuild_meta_first` — the first non-vector full-text field carries the language and
  stopwords (when either was saved); every other field keeps its own options.

Since #3680 the calls are **batched**: consecutive fields of one type with equal options
(not vector, not the field carrying the metadata, no repeated attribute) go in one
`create_index_sync` call (`rebuildB`). `rebuildB_expand` proves that expanding each batch
into one call per attribute gives exactly `rebuild`, so the three theorems above describe
what the batched calls create. `exec_retry` models the fallback: a failed batch of more than
one attribute is retried one attribute at a time.
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
  deriving DecidableEq

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

/-! ## Batched calls (#3680, `:368-438`) -/

/-- One `create_index_sync` call with several attributes. -/
structure BCall (A L S X V : Type) where
  ty : Ty
  ent : Ent
  label : String
  attrs : List A
  opts : Option (Opt L S X V)

section Batch
variable [DecidableEq A] [DecidableEq L] [DecidableEq S] [DecidableEq X]

/-- The run of following fields created exactly like `f` (`:412-421`): same type, no vector
options, equal options, attribute not yet in the batch (`seen.insert`). Returns the batched
attributes and what is left of `pending`. -/
def takeRun (f : Field L S X V) (seen : List A) : List (A × Field L S X V) → List A × List (A × Field L S X V)
  | [] => ([], [])
  | (a, g) :: rest =>
    if g.ty = f.ty ∧ g.vec.isNone ∧ g.opts = f.opts ∧ a ∉ seen then
      let r := takeRun f (a :: seen) rest
      (a :: r.1, r.2)
    else ([], (a, g) :: rest)

theorem takeRun_split (f : Field L S X V) (hf : f.vec = none) :
    ∀ (seen : List A) (l : List (A × Field L S X V)),
    l = (takeRun f seen l).1.map (fun a => (a, f)) ++ (takeRun f seen l).2
  | _, [] => rfl
  | seen, (a, g) :: rest => by
    unfold takeRun
    by_cases h : g.ty = f.ty ∧ g.vec.isNone ∧ g.opts = f.opts ∧ a ∉ seen
    · rw [if_pos h]
      obtain ⟨h1, h2, h3, -⟩ := h
      have hg : g = f := by
        cases g; cases f; simp only [Option.isNone_iff_eq_none] at h2; simp_all
      subst hg
      simp only [List.map_cons, List.cons_append]
      exact congrArg _ (takeRun_split g hf (a :: seen) rest)
    · rw [if_neg h]; rfl

theorem takeRun_length (f : Field L S X V) :
    ∀ (seen : List A) (l : List (A × Field L S X V)), (takeRun f seen l).2.length ≤ l.length
  | _, [] => Nat.le_refl _
  | seen, (a, g) :: rest => by
    unfold takeRun
    split
    · exact Nat.le_succ_of_le (takeRun_length f (a :: seen) rest)
    · exact Nat.le_refl _

/-- Whether the field carries the index-level language/stopwords (`carries_meta`). -/
def carries (p : Bool) (f : Field L S X V) : Bool := f.vec.isNone && p && f.ty == .fulltext

/-- The batched loop (`while let Some(..) = pending.pop()`). -/
def goB (info : Info A L S X V) (ent : Ent) (p : Bool) : List (A × Field L S X V) → List (BCall A L S X V)
  | [] => []
  | (a, f) :: rest =>
    let r := fieldOpts info p f
    if f.vec.isNone && !carries p f then
      ⟨f.ty, ent, info.label, a :: (takeRun f [a] rest).1, r.1⟩ :: goB info ent r.2 (takeRun f [a] rest).2
    else ⟨f.ty, ent, info.label, [a], r.1⟩ :: goB info ent r.2 rest
termination_by l => l.length
decreasing_by
  · have := takeRun_length f [a] rest; simp; omega
  · simp

/-- `rebuild_indexes` for one `IndexInfo`, batched. -/
def rebuildB (info : Info A L S X V) : List (BCall A L S X V) :=
  goB info (entOf info.entityType) (info.language.isSome || info.stopwords.isSome) (flat info)

/-- One call per attribute of each batch. -/
def expand (bs : List (BCall A L S X V)) : List (Call A L S X V) :=
  bs.flatMap fun b => b.attrs.map fun a => ⟨b.ty, b.ent, b.label, a, b.opts⟩

theorem go_rep (info : Info A L S X V) (ent : Ent) (p : Bool) (f : Field L S X V)
    (o : Option (Opt L S X V)) (hf : fieldOpts info p f = (o, p)) :
    ∀ (as : List A) (l : List (A × Field L S X V)),
    go info ent p (as.map (fun a => (a, f)) ++ l) =
      as.map (fun a => (⟨f.ty, ent, info.label, a, o⟩ : Call A L S X V)) ++ go info ent p l
  | [], _ => rfl
  | a :: as, l => by
    simp only [List.map_cons, List.cons_append, go, hf]
    exact congrArg _ (go_rep info ent p f o hf as l)

theorem goB_expand (info : Info A L S X V) (ent : Ent) :
    ∀ (n : Nat) (p : Bool) (l : List (A × Field L S X V)), l.length ≤ n →
    expand (goB info ent p l) = go info ent p l
  | _, _, [], _ => by simp [goB, go, expand]
  | 0, _, _ :: _, h => by simp at h
  | n + 1, p, (a, f) :: rest, h => by
    rw [goB]
    by_cases hb : (f.vec.isNone && !carries p f) = true
    · rw [if_pos hb]
      have hv : f.vec = none := by simp [Bool.and_eq_true] at hb; exact hb.1
      have hc : carries p f = false := by simp [Bool.and_eq_true] at hb; exact hb.2
      have hfo : fieldOpts info p f = (f.opts.map .text, p) := by
        have : (p && f.ty == .fulltext) = false := by simpa [carries, hv] using hc
        simp [fieldOpts, hv, this]
      have hl := takeRun_length f [a] rest
      have ih := goB_expand info ent n p (takeRun f [a] rest).2 (by simp at h; omega)
      rw [hfo]
      conv => rhs; rw [takeRun_split f hv [a] rest]
      simp only [go, hfo, go_rep info ent p f _ hfo, ← ih]
      simp [expand]
    · rw [if_neg hb]
      have ih := goB_expand info ent n (fieldOpts info p f).2 rest (by simp at h; omega)
      simp only [go, ← ih]
      simp [expand]

/-- **The batched calls create the same indexes**: expanding every batch into one call per
attribute gives exactly the per-field calls of `rebuild` (so `rebuild_calls`,
`rebuild_entity`, `rebuild_meta_first` and `rebuild_no_meta` describe `rebuildB`). -/
theorem rebuildB_expand (info : Info A L S X V) : expand (rebuildB info) = rebuild info :=
  goB_expand info _ _ _ _ (Nat.le_refl _)

/-- A batch never repeats an attribute (`seen.insert`). -/
theorem takeRun_nodup (f : Field L S X V) :
    ∀ (seen : List A) (l : List (A × Field L S X V)), ∀ a ∈ (takeRun f seen l).1, a ∉ seen
  | _, [], _, h => by simp [takeRun] at h
  | seen, (b, g) :: rest, a, h => by
    unfold takeRun at h
    split at h
    · rename_i hc
      rcases List.mem_cons.1 h with rfl | h
      · exact hc.2.2.2
      · exact fun hs => takeRun_nodup f (b :: seen) rest a h (List.mem_cons_of_mem _ hs)
    · simp at h

/-- The calls one batch makes (`:425-438`): the batch, and if it fails and has more than one
attribute, every attribute again on its own. -/
def exec (ok : List A → Bool) (attrs : List A) : List (List A) :=
  if ok attrs then [attrs]
  else if 1 < attrs.length then attrs :: attrs.map (fun a => [a]) else [attrs]

/-- **One bad field does not take the others with it**: a failed batch is retried attribute
by attribute. -/
theorem exec_retry (ok : List A → Bool) (attrs : List A) (h : ok attrs = false) :
    ∀ a ∈ attrs, [a] ∈ exec ok attrs := by
  intro a ha
  unfold exec
  rw [if_neg (by simp [h])]
  split
  · exact List.mem_cons_of_mem _ (List.mem_map.2 ⟨a, ha, rfl⟩)
  · match attrs, ha with
    | [b], ha => simp at ha; subst ha; simp
    | b :: c :: t, _ => simp at *

end Batch

end GraphPersist.Persist.Indexes
