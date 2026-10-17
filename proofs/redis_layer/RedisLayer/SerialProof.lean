import RedisLayer.SerialEntry
/-! # Schema round trip (proofs for `SerialEntry`) -/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

variable (U : Utf8) (one : Nat) (simCode : Option Bytes → Nat)

theorem pos_some (l : List Bytes) (p : Bytes) (h : p ∈ l) :
    ∃ k, pos l p = some k ∧ l[k]? = some p := by
  induction l with
  | nil => simp at h
  | cons a as ih =>
    by_cases hap : a = p
    · exact ⟨0, by simp [pos, hap]⟩
    · have : p ∈ as := by simp at h; rcases h with h | h; exact absurd h.symm hap; exact h
      obtain ⟨k, hk, hg⟩ := ih this
      exact ⟨k+1, by simp [pos, hap, hk], by simpa using hg⟩

theorem pos_get (l : List Bytes) (p : Bytes) (h : p ∈ l) : l[attrId l p]? = some p := by
  obtain ⟨k, hk, hg⟩ := pos_some l p h
  simp [attrId, hk, hg]

theorem lookup_attrId (attrs : List Bytes) (p : Bytes) (h : p ∈ attrs) :
    lookupAttr attrs (attrId attrs p) = p := by
  simp [lookupAttr, pos_get attrs p h]

theorem flatten_singletons {α β} (l : List α) (f : α → β) :
    (l.map fun a => [f a]).flatten = l.map f := by
  induction l <;> simp_all

theorem decCons1_enc (attrs : List Bytes) (name : Bytes) (c : Cons)
    (hp : ∀ p ∈ c.props, p ∈ attrs) (ts : List W) :
    decCons1 attrs name (([W.u (if c.unique then 0 else 1), .u c.props.length]
        ++ c.props.map fun p => W.u (attrId attrs p)) ++ ts)
      = some (⟨c.unique, false, name, c.props, true⟩, ts) := by
  have hm := rdMany_enc rdU (fun i => [W.u i]) (c.props.map (attrId attrs))
    (by intro x _ ts; rfl) ts
  rw [flatten_singletons, List.map_map] at hm
  simp only [List.length_map] at hm
  have hmap : c.props.map (lookupAttr attrs ∘ attrId attrs) = c.props := by
    conv => rhs; rw [← List.map_id c.props]
    apply List.map_congr_left
    intro p hp'; exact lookup_attrId attrs p (hp p hp')
  simp only [List.map_map, Function.comp_def] at hm
  simp only [decCons1, Dec.bind, rdU, List.cons_append, List.append_assoc, List.nil_append]
  rw [hm]
  simp only [pure', List.map_map, hmap]
  cases c.unique <;> simp

theorem rdMany_map {α β} (p : Dec α) (e : β → List W) (g : β → α) (ys : List β)
    (hp : ∀ y ∈ ys, ∀ ts, p (e y ++ ts) = some (g y, ts)) (ts : List W) :
    rdMany p ys.length ((ys.map e).flatten ++ ts) = some (ys.map g, ts) := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [rdMany, Dec.bind, List.length_cons, List.map_cons, List.flatten_cons,
      List.append_assoc]
    rw [hp y (by simp)]
    simp only
    rw [ih (fun z hz => hp z (by simp [hz]))]
    rfl

/-- What a well-formed field list needs: valid names, vector fields carry options. -/
def FieldsWF (fs : List (Bytes × Fld)) : Prop :=
  ∀ af ∈ fs, U.valid af.1 ∧ U.valid (af.2.phonetic.getD []) ∧ (af.2.ty = .vector → af.2.vopts.isSome)

def HeadWF : List Info → Prop
  | [] => True
  | i :: _ => U.valid (i.language.getD ENGLISH) ∧ ∀ s ∈ i.stopwords.getD [], U.valid s

theorem map_strip_nullTerm (l : List Bytes) (h : ∀ s ∈ l, U.valid s) :
    (l.map nullTerm).map (strip U) = l := by
  induction l with
  | nil => rfl
  | cons a as ih =>
    simp only [List.map_cons, strip_nullTerm U a (h a (by simp))]
    rw [ih (fun s hs => h s (by simp [hs]))]

theorem decIdx_enc (is : List Info) (hh : HeadWF U is) (hf : FieldsWF U (allFields is))
    (ts : List W) :
    decIdx U (encIdx one simCode is ++ ts) = some (mergeInfo one simCode [] false is, ts) := by
  cases is with
  | nil => simp [decIdx, encIdx, Dec.bind, rdU, pure', mergeInfo]
  | cons i rest =>
    obtain ⟨hl, hsw⟩ := hh
    have h1 := rdMany_map rdB (fun s => [W.buf (nullTerm s)]) nullTerm (i.stopwords.getD [])
      (by intro y _ ts; rfl)
      ([W.u (allFields (i :: rest)).length]
        ++ (((allFields (i :: rest)).map fun (a, f) => encField one simCode a f).flatten ++ ts))
    have h2 := rdMany_map (decField U) (fun af => encField one simCode af.1 af.2)
      (fun af => (attrBack af.1 af.2.ty, normField one simCode af.2)) (allFields (i :: rest))
      (by
        intro af haf ts
        obtain ⟨ha, hp, hv⟩ := hf af haf
        exact decField_encField U one simCode af.1 af.2 ha hp hv ts) ts
    simp only [decIdx, encIdx, List.cons_append, List.append_assoc, List.nil_append]
    rw [Dec.bind_some _ _ _ _ _ (rdU_u _ _)]
    simp only [Nat.one_ne_zero, ne_eq, not_false_eq_true, ite_true]
    rw [Dec.bind_some _ _ _ _ _ (rdB_buf _ _), Dec.bind_some _ _ _ _ _ (rdU_u _ _)]
    simp only [List.singleton_append] at h1
    rw [Dec.bind_some _ _ _ _ _ h1, Dec.bind_some _ _ _ _ _ (rdU_u _ _)]
    rw [Dec.bind_some _ _ _ _ _ h2]
    simp only [pure'_apply, mergeInfo, strip_nullTerm U _ hl, map_strip_nullTerm U _ hsw]
    by_cases he : i.stopwords.getD [] = [] <;> simp [he]

def decCons (l : Bytes) (c : Cons) : Cons := ⟨c.unique, false, l, c.props, true⟩

theorem decConsBlock {β} (attrs : List Bytes) (l : Bytes) (cs : List Cons)
    (hc : ∀ c ∈ cs, c.operational → ∀ p ∈ c.props, p ∈ attrs) (k : List Cons → Dec β) (ts : List W) :
    Dec.bind rdU (fun cc => Dec.bind (rdMany (decCons1 attrs l) cc) k) (encCons attrs cs ++ ts)
      = k ((cs.filter (·.operational)).map (decCons l)) ts := by
  have h := rdMany_map (decCons1 attrs l)
    (fun c => [W.u (if c.unique then 0 else 1), .u c.props.length]
        ++ c.props.map fun p => W.u (attrId attrs p))
    (decCons l) (cs.filter (·.operational))
    (by
      intro c hcm ts
      simp only [List.mem_filter] at hcm
      exact decCons1_enc attrs l c (hc c hcm.1 hcm.2) ts) ts
  simp only [encCons]
  rw [List.append_assoc, List.singleton_append, Dec.bind_some _ _ _ _ _ (rdU_u _ _)]
  rw [Dec.bind_some _ _ _ _ _ h]

theorem mergeInfo_stamp (l : Bytes) (rel : Bool) (is : List Info) :
    (mergeInfo one simCode [] false is).map (fun i => { i with label := l, isRel := rel })
      = mergeInfo one simCode l rel is := by
  cases is <;> rfl

/-- Well-formedness the round trip needs (all of it holds for schemas Rust builds). -/
structure WF (s : Schema) : Prop where
  attrs : ∀ a ∈ s.attrs, U.valid a
  names : ∀ l ∈ s.labels ++ s.types, U.valid l
  heads : ∀ l rel, HeadWF U (s.infos.filter fun i => i.label = l && i.isRel = rel)
  fields : ∀ l rel, FieldsWF U (allFields (s.infos.filter fun i => i.label = l && i.isRel = rel))
  cons : ∀ c ∈ s.cons, c.operational → ∀ p ∈ c.props, p ∈ s.attrs

def entryOut (s : Schema) (rel : Bool) (il : Nat × Bytes) : Bytes × Option Info × List Cons :=
  (il.2, mergeInfo one simCode [] false (s.infos.filter fun i => i.label = il.2 && i.isRel = rel),
   ((s.cons.filter fun c => c.isRel = rel && c.label = il.2).filter (·.operational)).map (decCons il.2))

theorem decEntry_enc (s : Schema) (hw : WF U s) (rel : Bool) (il : Nat × Bytes)
    (hl : U.valid il.2) (ts : List W) :
    decEntry U s.attrs (encEntry one simCode s rel il ++ ts) = some (entryOut one simCode s rel il, ts) := by
  obtain ⟨i, l⟩ := il
  simp only at hl
  have hc := decConsBlock s.attrs l (s.cons.filter fun c => c.isRel = rel && c.label = l)
    (fun c hcm hop => hw.cons c (List.mem_filter.1 hcm).1 hop)
    (fun cs => pure' (l, mergeInfo one simCode [] false
      (s.infos.filter fun i => i.label = l && i.isRel = rel), cs)) ts
  simp only [decEntry, encEntry, List.cons_append, List.append_assoc, List.nil_append]
  rw [Dec.bind_some _ _ _ _ _ (rdU_u _ _), Dec.bind_some _ _ _ _ _ (rdB_buf _ _)]
  simp only [strip_nullTerm U l hl]
  rw [Dec.bind_some _ _ _ _ _ (decIdx_enc U one simCode _ (hw.heads l rel) (hw.fields l rel) _)]
  exact hc

theorem enumFrom_length (n : Nat) (l : List Bytes) : (enumFrom n l).length = l.length := by
  induction l generalizing n <;> simp_all [enumFrom]

theorem enumFrom_map {β} (n : Nat) (l : List Bytes) (f : Bytes → β) :
    (enumFrom n l).map (fun il => f il.2) = l.map f := by
  induction l generalizing n <;> simp_all [enumFrom]

theorem enumFrom_mem (n : Nat) (l : List Bytes) (il : Nat × Bytes) (h : il ∈ enumFrom n l) :
    il.2 ∈ l := by
  induction l generalizing n with
  | nil => simp [enumFrom] at h
  | cons a as ih =>
    simp [enumFrom] at h
    rcases h with rfl | h
    · simp
    · exact List.mem_cons_of_mem _ (ih _ h)

theorem stamp_entry1 (s : Schema) (rel : Bool) (il : Nat × Bytes) :
    (stamp rel (entryOut one simCode s rel il)).1
      = (mergeInfo one simCode il.2 rel (s.infos.filter fun i => i.label = il.2 && i.isRel = rel)).toList := by
  simp only [stamp, entryOut]
  generalize (s.infos.filter fun i => i.label = il.2 && i.isRel = rel) = is
  cases is <;> rfl

theorem stamp_entry2 (s : Schema) (rel : Bool) (il : Nat × Bytes) :
    (stamp rel (entryOut one simCode s rel il)).2
      = ((s.cons.filter fun c => c.isRel = rel && c.label = il.2).filter (·.operational)).map normCons := by
  simp only [stamp, entryOut, List.map_map]
  apply List.map_congr_left
  intro c hc
  simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨_, hr, hlab⟩, hop⟩ := hc
  obtain ⟨u, r, lb, ps, op⟩ := c
  simp_all [decCons, normCons]

/-- **Schema round trip**: `Schema::decode ∘ Schema::encode = normSchema`. -/
theorem decSchema_encSchema (s : Schema) (hw : WF U s) (ts : List W) :
    decSchema U (encSchema one simCode s ++ ts) = some (normSchema one simCode s, ts) := by
  have hA := rdMany_map rdB (fun a => [W.buf (nullTerm a)]) nullTerm s.attrs (by intro _ _ _; rfl)
    ([W.u s.labels.length] ++ (((enumFrom 0 s.labels).map (encEntry one simCode s false)).flatten
      ++ ([W.u s.types.length] ++ (((enumFrom 0 s.types).map (encEntry one simCode s true)).flatten ++ ts))))
  have hL := rdMany_map (decEntry U s.attrs) (encEntry one simCode s false) (entryOut one simCode s false)
    (enumFrom 0 s.labels)
    (fun il hil ts => decEntry_enc U one simCode s hw false il
      (hw.names _ (List.mem_append_left _ (enumFrom_mem 0 _ il hil))) ts)
    ([W.u s.types.length] ++ (((enumFrom 0 s.types).map (encEntry one simCode s true)).flatten ++ ts))
  have hT := rdMany_map (decEntry U s.attrs) (encEntry one simCode s true) (entryOut one simCode s true)
    (enumFrom 0 s.types)
    (fun il hil ts => decEntry_enc U one simCode s hw true il
      (hw.names _ (List.mem_append_right _ (enumFrom_mem 0 _ il hil))) ts) ts
  rw [enumFrom_length] at hL hT
  have hstrip := map_strip_nullTerm U s.attrs hw.attrs
  simp only [decSchema, encSchema, List.append_assoc]
  rw [List.singleton_append, Dec.bind_some _ _ _ _ _ (rdU_u _ _), Dec.bind_some _ _ _ _ _ hA]
  rw [hstrip]
  rw [List.singleton_append, Dec.bind_some _ _ _ _ _ (rdU_u _ _), Dec.bind_some _ _ _ _ _ hL]
  rw [List.singleton_append, Dec.bind_some _ _ _ _ _ (rdU_u _ _), Dec.bind_some _ _ _ _ _ hT]
  simp only [pure'_apply, List.map_map, Option.some.injEq, Prod.mk.injEq, and_true]
  have e1 : ∀ rel (l : List Bytes), (enumFrom 0 l).map ((fun e => (stamp rel e).1) ∘ entryOut one simCode s rel)
      = l.map fun x => (mergeInfo one simCode x rel (s.infos.filter fun i => i.label = x && i.isRel = rel)).toList := by
    intro rel l
    rw [← enumFrom_map 0 l]; apply List.map_congr_left; intro il _
    exact stamp_entry1 one simCode s rel il
  have e2 : ∀ rel (l : List Bytes), (enumFrom 0 l).map ((fun e => (stamp rel e).2) ∘ entryOut one simCode s rel)
      = l.map fun x => ((s.cons.filter fun c => c.isRel = rel && c.label = x).filter (·.operational)).map normCons := by
    intro rel l
    rw [← enumFrom_map 0 l]; apply List.map_congr_left; intro il _
    exact stamp_entry2 one simCode s rel il
  have e3 : ∀ rel (l : List Bytes), (enumFrom 0 l).map ((fun x => x.1) ∘ entryOut one simCode s rel) = l := by
    intro rel l
    have := enumFrom_map 0 l id
    simpa [Function.comp_def, entryOut] using this
  rw [e1, e1, e2, e2, e3, e3]
  rfl

/-- `Schema::from_graph` (`:702-718`): the four graph lists and the caller's attribute
table, unchanged. -/
def schemaFromGraph (attrs labels types : List Bytes) (infos : List Info) (cons : List Cons) : Schema :=
  ⟨attrs, labels, types, infos, cons⟩

theorem schemaFromGraph_fields (attrs labels types : List Bytes) (infos : List Info) (cons : List Cons) :
    (schemaFromGraph attrs labels types infos cons).attrs = attrs ∧
    (schemaFromGraph attrs labels types infos cons).labels = labels ∧
    (schemaFromGraph attrs labels types infos cons).cons = cons := ⟨rfl, rfl, rfl⟩

theorem pos_none (l : List Bytes) (p : Bytes) (h : p ∉ l) : pos l p = none := by
  induction l with
  | nil => rfl
  | cons a as ih =>
    have hap : a ≠ p := fun e => h (by simp [e])
    have : p ∉ as := fun hm => h (by simp [hm])
    simp [pos, hap, ih this]

/-- **Finding (latent)**: a constraint property absent from the attribute table is written
as id 0 (`unwrap_or(0)`, `:460`) and so reloads as attribute 0. -/
theorem cons_unknown_prop (a : Bytes) (as : List Bytes) (p : Bytes) (h : p ∉ a :: as) :
    lookupAttr (a :: as) (attrId (a :: as) p) = a := by
  simp [attrId, pos_none _ _ h, lookupAttr]

end RedisLayer.Serial
