/-
SET and REMOVE (graph/src/runtime/ops/set.rs, remove.rs).

| here | there |
| --- | --- |
| `SetSt.new`, `RemoveSt.new`   | `SetOp::new` (set.rs:40), `RemoveOp::new` (remove.rs:30) |
| `opNext`                      | `SetOp::next` (set.rs:59), `RemoveOp::next` (remove.rs:48) |
| `perRow`                      | `Runtime::set_batch` (set.rs:76), `Runtime::remove_batch` (remove.rs:62) |
| `resolveItems`                | `Runtime::resolve_set_items` (set.rs:96) |
| `clearForReplace`                 | `clear_node_attrs_for_replace` (set.rs:125), `clear_relationship_attrs_for_replace` (set.rs:138), `Pending::clear_{node,relationship}_attributes` (pending.rs:591, :599) |
| `setProp`, `setFrom`, `setAttrItem` | `Runtime::set_inner` (set.rs:151) |
| `removeItem`, `removeInner`   | `Runtime::remove` (remove.rs:74) |

Attribute store of one entity kind: committed `cv e k` and this query's pending writes
`pv e k` (`some none` = pending null = removal); `get` = `get_{node,relationship}_attribute`
(pending overrides committed, null = absent). `ckeys e` = the committed keys
(`g.get_*_attrs(id)`), assumed complete.
-/
namespace SetRemove

variable {E K V : Type} [DecidableEq E] [DecidableEq K] [DecidableEq V]

structure Store (E K V : Type) where
  cv : E → K → Option V
  pv : E → K → Option (Option V)

def get (s : Store E K V) (e : E) (k : K) : Option V :=
  match s.pv e k with
  | some v => v
  | none => s.cv e k

/-- `set_pending_*_attr(id, k, v)` (runtime.rs:1371-1405). -/
def put (s : Store E K V) (e : E) (k : K) (v : Option V) : Store E K V :=
  { s with pv := fun e' k' => if e' = e ∧ k' = k then some v else s.pv e' k' }

/-- `clear_{node}_attributes(id)` (pending.rs). -/
def clearP (s : Store E K V) (e : E) : Store E K V :=
  { s with pv := fun e' k' => if e' = e then none else s.pv e' k' }

theorem get_put (s : Store E K V) (e : E) (k : K) (v : Option V) (e' : E) (k' : K) :
    get (put s e k v) e' k' = if e' = e ∧ k' = k then v else get s e' k' := by
  unfold get put; by_cases h : e' = e ∧ k' = k <;> simp [h]

/-- set.rs:203-217 / :279-296: skip when the stored value already equals the new one. -/
def setProp (s : Store E K V) (e : E) (k : K) (v : Option V) : Store E K V :=
  if get s e k = v ∧ v.isSome then s else put s e k v

theorem setProp_get (s : Store E K V) (e : E) (k : K) (v : Option V) (e' : E) (k' : K) :
    get (setProp s e k v) e' k' = if e' = e ∧ k' = k then v else get s e' k' := by
  unfold setProp; split
  · rename_i h; by_cases h' : e' = e ∧ k' = k
    · obtain ⟨rfl, rfl⟩ := h'; simp [h.1]
    · simp [h']
  · exact get_put s e k v e' k'

/-- `SET e.k = v`: afterwards `e.k` reads `v` (null removes), every other property unchanged. -/
theorem setProp_spec (s : Store E K V) (e : E) (k : K) (v : Option V) :
    get (setProp s e k v) e k = v ∧ ∀ e' k', ¬(e' = e ∧ k' = k) → get (setProp s e k v) e' k' = get s e' k' := by
  refine ⟨by simp [setProp_get], fun e' k' h => by simp [setProp_get, h]⟩

def setAll (s : Store E K V) (e : E) (kvs : List (K × Option V)) : Store E K V :=
  kvs.foldl (fun s kv => setProp s e kv.1 kv.2) s

theorem setAll_get (s : Store E K V) (e : E) (kvs : List (K × Option V)) (e' : E) (k : K) :
    get (setAll s e kvs) e' k =
      if e' = e then (kvs.foldl (fun acc kv => if kv.1 = k then kv.2 else acc) (get s e k))
      else get s e' k := by
  induction kvs generalizing s with
  | nil => by_cases he : e' = e <;> simp [setAll, he]
  | cons kv kvs ih =>
    simp only [setAll, List.foldl_cons] at ih ⊢
    rw [ih]
    by_cases he : e' = e
    · subst he; simp only [ite_true, setProp_get]; by_cases hk : kv.1 = k
      · simp [hk]
      · simp [hk, Ne.symm hk]
    · simp [he, setProp_get]

/-- `clear_node_attrs_for_replace` / `clear_relationship_attrs_for_replace` (set.rs:125-148):
drop the entity's pending attributes (`clearP`), then stage a null for every committed key. -/
def clearForReplace (ckeys : E → List K) (s : Store E K V) (e : E) : Store E K V :=
  setAll (clearP s e) e ((ckeys e).map (·, none))

/-- `SET e = src` (`replace`) / `SET e += src` for ANY target (node or relationship) and ANY
source (map, node, relationship; `m` is the source's attributes, read before the clear):
with `=` first `clearForReplace`, then write the entries. Since #3524 (af45ca776) every
branch of `set_inner` clears (set.rs:220, :242, :256, :300, :325, :345; Node←Node/Rel←Rel skip `tid = id`, :237, :340). -/
def setFrom (ckeys : E → List K) (s : Store E K V) (e : E) (replace : Bool)
    (m : List (K × Option V)) : Store E K V :=
  let s1 := if replace then clearForReplace ckeys s e else s
  setAll s1 e m

/-- The committed-key list is complete. -/
def KeysOk (ckeys : E → List K) (s : Store E K V) : Prop := ∀ e k, (s.cv e k).isSome → k ∈ ckeys e

theorem foldl_last (m : List (K × Option V)) (k : K) (init : Option V) :
    m.foldl (fun acc kv => if kv.1 = k then kv.2 else acc) init =
      ((m.reverse.find? (·.1 = k)).map (·.2)).getD init := by
  induction m generalizing init with
  | nil => rfl
  | cons kv m ih =>
    simp only [List.foldl_cons, List.reverse_cons, List.find?_append]
    rw [ih]
    cases h : m.reverse.find? (·.1 = k) with
    | some p => simp
    | none => by_cases hk : kv.1 = k <;> simp [hk]

/-- **`SET e = src`** (any target, any source, since #3524): afterwards `e.k` is the
source's value for `k` (last entry wins) and absent for every key the source does not
mention — committed or pending. -/
theorem setFrom_replace (ckeys : E → List K) (s : Store E K V) (e : E)
    (m : List (K × Option V)) (hk : KeysOk ckeys s) (k : K) :
    get (setFrom ckeys s e true m) e k = ((m.reverse.find? (·.1 = k)).map (·.2)).getD none := by
  unfold setFrom clearForReplace
  simp only [ite_true]
  rw [setAll_get, if_pos rfl, foldl_last, setAll_get, if_pos rfl, foldl_last]
  congr 1
  -- after clearing pending and nulling committed keys, `k` reads none
  cases hf : ((((ckeys e).map (·, (none : Option V))).reverse.find? (·.1 = k)).map (·.2)) with
  | some v =>
    simp only [Option.map_eq_some_iff] at hf
    obtain ⟨p, hp, rfl⟩ := hf
    have := List.mem_of_find?_eq_some hp
    simp at this; obtain ⟨_, _, rfl⟩ := this; rfl
  | none =>
    simp only [Option.getD_none]
    unfold get clearP; simp only [ite_true]
    cases hc : s.cv e k with
    | none => rfl
    | some v =>
      exfalso
      have hm := hk e k (by simp [hc])
      have hs : (((ckeys e).map (·, (none : Option V))).reverse.find? (·.1 = k)).isSome := by
        rw [List.find?_isSome]
        exact ⟨(k, none), by simp [hm], by simp⟩
      rw [Option.map_eq_none_iff] at hf
      rw [hf] at hs; cases hs

/-- `SET e += src` leaves every key the source does not mention as it was. -/
theorem setFrom_merge_keeps (ckeys : E → List K) (s : Store E K V) (e : E)
    (m : List (K × Option V)) (k : K) (hk : ∀ p ∈ m, p.1 ≠ k) :
    get (setFrom ckeys s e false m) e k = get s e k := by
  unfold setFrom
  simp only [Bool.false_eq_true, ite_false]
  rw [setAll_get, if_pos rfl]
  have : ∀ (init : Option V) (l : List (K × Option V)), (∀ p ∈ l, p.1 ≠ k) →
      l.foldl (fun acc kv => if kv.1 = k then kv.2 else acc) init = init := by
    intro init l; induction l generalizing init with
    | nil => intro _; rfl
    | cons p l ih =>
      intro h; simp only [List.foldl_cons, if_neg (h p (by simp))]
      exact ih _ (fun q hq => h q (by simp [hq]))
  exact this _ m hk

/-- **History (pre-#3524, fixed by #3524 af45ca776; FINDINGS "#2776, wider").** With `=`, the
no-clear branch kept a PENDING key the source lacks, e.g. `CREATE (a {x:1}), (b {y:2}) SET a = b`
(`a.x` survived: Rust `{x:1,y:2}`, C `{y:2}`) and `SET r.w=5 SET r={y:2}` on a relationship. -/
def demoStore : Store Nat Nat Nat := { cv := fun _ _ => none, pv := fun e k => if e = 0 ∧ k = 0 then some (some 1) else none }

/-- **Fixed by #3524**: the same state and source on main. The staged `x` is gone and only the
source's `y` remains. -/
theorem fixed3524_clears_pending :
    get (setFrom (fun _ => []) demoStore 0 true [(1, some 2)]) 0 0 = none ∧
    get (setFrom (fun _ => []) demoStore 0 true [(1, some 2)]) 0 1 = some 2 := by decide

/-! ## `set_inner` per item (set.rs:151-410) -/

inductive Ent (E : Type) where
  | node (e : E)
  | rel (e : E)
  | other     -- any other value, `Null` included: skipped (set.rs:365)

inductive Src (E K V : Type) where
  | map (m : List (K × Option V))
  | node (e : E)
  | rel (e : E)
  | other

/-- `SET target(.prop)? = value` for one row. `gone` = the deleted-entity check
(set.rs:196-201 node, :274-278 relationship), only consulted when `skip_delete_checks` is false. `attrsN`/`attrsR` =
`get_node_attrs` / `get_relationship_attrs` of a source entity. -/
def setAttrItem (ckeys : E → List K) (gone : Ent E → Bool) (skipDel : Bool)
    (attrsN attrsR : E → List (K × Option V))
    (sN sR : Store E K V) (target : Ent E) (prop : Option K) (val : Src E K V) (replace : Bool)
    (scalar : Option V) : Except String (Store E K V × Store E K V) :=
  let err := "Property values can only be of primitive types or arrays of primitive types"
  match target with
  | .other => .ok (sN, sR)
  | t@(.node e) =>
    if !skipDel && gone t then .ok (sN, sR) else
    match prop with
    | some k => .ok (setProp sN e k scalar, sR)
    | none => match val with
      | .map m => .ok (setFrom ckeys sN e replace m, sR)
      | .node tid => if tid = e then .ok (sN, sR) else .ok (setFrom ckeys sN e replace (attrsN tid), sR)
      | .rel r => .ok (setFrom ckeys sN e replace (attrsR r), sR)
      | .other => .error err
  | t@(.rel e) =>
    if !skipDel && gone t then .ok (sN, sR) else
    match prop with
    | some k => .ok (sN, setProp sR e k scalar)
    | none => match val with
      | .map m => .ok (sN, setFrom ckeys sR e replace m)
      | .node n => .ok (sN, setFrom ckeys sR e replace (attrsN n))
      | .rel r => if r = e then .ok (sN, sR) else .ok (sN, setFrom ckeys sR e replace (attrsR r))
      | .other => .error err

/-- Deleted targets are skipped; a non-entity target is a no-op; a property write lands. -/
theorem setAttrItem_spec (ckeys : E → List K) (gone : Ent E → Bool) (aN aR : E → List (K × Option V))
    (sN sR : Store E K V) (e : E) (k : K) (v : Option V) (val : Src E K V) (rep : Bool) :
    setAttrItem ckeys gone true aN aR sN sR (.node e) (some k) val rep v = .ok (setProp sN e k v, sR) ∧
    setAttrItem ckeys gone true aN aR sN sR (.rel e) (some k) val rep v = .ok (sN, setProp sR e k v) ∧
    setAttrItem ckeys gone false aN aR sN sR .other (some k) val rep v = .ok (sN, sR) ∧
    (gone (.node e) = true → setAttrItem ckeys gone false aN aR sN sR (.node e) (some k) val rep v = .ok (sN, sR)) := by
  refine ⟨rfl, rfl, rfl, fun h => by simp [setAttrItem, h]⟩

/-- **#3524 fixes #2776 for every shape** (Node←Map, Node←Node, Node←Rel, Rel←Map, Rel←Node,
Rel←Rel): with `=` the target ends with exactly the source's attributes, whatever was staged or
committed before. -/
theorem setAttrItem_replace_shapes (ckeys : E → List K) (gone : Ent E → Bool)
    (aN aR : E → List (K × Option V)) (sN sR : Store E K V) (e : E) (m : List (K × Option V))
    (tid r : E) (hk : KeysOk ckeys sN) (hk' : KeysOk ckeys sR) (k : K) :
    let rd : List (K × Option V) → Option V := fun l => ((l.reverse.find? (·.1 = k)).map (·.2)).getD none
    (∃ s, setAttrItem ckeys gone true aN aR sN sR (.node e) none (.map m) true none = .ok (s, sR) ∧
        get s e k = rd m) ∧
    (tid ≠ e → ∃ s, setAttrItem ckeys gone true aN aR sN sR (.node e) none (.node tid) true none = .ok (s, sR) ∧
        get s e k = rd (aN tid)) ∧
    (∃ s, setAttrItem ckeys gone true aN aR sN sR (.node e) none (.rel r) true none = .ok (s, sR) ∧
        get s e k = rd (aR r)) ∧
    (∃ s, setAttrItem ckeys gone true aN aR sN sR (.rel e) none (.map m) true none = .ok (sN, s) ∧
        get s e k = rd m) ∧
    (∃ s, setAttrItem ckeys gone true aN aR sN sR (.rel e) none (.node tid) true none = .ok (sN, s) ∧
        get s e k = rd (aN tid)) ∧
    (r ≠ e → ∃ s, setAttrItem ckeys gone true aN aR sN sR (.rel e) none (.rel r) true none = .ok (sN, s) ∧
        get s e k = rd (aR r)) := by
  intro rd
  refine ⟨⟨_, by simp [setAttrItem], setFrom_replace ckeys sN e m hk k⟩,
    fun h => ⟨_, by simp [setAttrItem, h], setFrom_replace ckeys sN e _ hk k⟩,
    ⟨_, by simp [setAttrItem], setFrom_replace ckeys sN e _ hk k⟩,
    ⟨_, by simp [setAttrItem], setFrom_replace ckeys sR e m hk' k⟩,
    ⟨_, by simp [setAttrItem], setFrom_replace ckeys sR e _ hk' k⟩,
    fun h => ⟨_, by simp [setAttrItem, h], setFrom_replace ckeys sR e _ hk' k⟩⟩

/-- `SET n:L` (set.rs:370-396): labels for a live node, null skipped, anything else errors. -/
def setLabelItem {L : Type} (gone : E → Bool) (skipDel : Bool) (labels : E → List L)
    (v : Option (Ent E)) (isNull : Bool) (ls : List L) : Except String (E → List L) :=
  match v with
  | some (.node e) =>
    if !skipDel && gone e then .ok labels
    else .ok fun e' => if e' = e then labels e ++ ls else labels e'
  | _ => if isNull then .ok labels else .error "Type mismatch: expected Node"

theorem setLabelItem_live {L : Type} (gone : E → Bool) (labels : E → List L) (e : E) (ls : List L) :
    (setLabelItem gone true labels (some (.node e)) false ls).map (· e) = .ok (labels e ++ ls) := by
  simp [setLabelItem, Except.map]

/-- set.rs:76-93 / remove.rs:62-71: every active row in order, the first error aborts. -/
def perRow {S R : Type} (f : S → R → Except String S) (s : S) (rows : List R) : Except String S :=
  rows.foldlM f s

theorem perRow_spec {S R : Type} (f : S → R → Except String S) (s : S) (r : R) (rs : List R) :
    perRow f s [] = .ok s ∧ perRow f s (r :: rs) = (f s r).bind fun s' => perRow f s' rs := by
  exact ⟨rfl, rfl⟩

/-- `resolve_set_items` (set.rs:96): label names → ids (same contract as `resolve_pattern`),
attribute items copied unchanged, order kept. -/
def resolveItems {A Tb : Type} (goc : Tb → String → Nat × Tb) : Tb →
    List (Sum (Nat × List String) A) → List (Sum (Nat × List Nat) A) × Tb
  | t, [] => ([], t)
  | t, .inr a :: is => let (r, t') := resolveItems goc t is; (.inr a :: r, t')
  | t, .inl (v, ls) :: is =>
    let (ids, t1) := ls.foldl (fun (acc : List Nat × Tb) l => let (i, t') := goc acc.2 l; (acc.1 ++ [i], t')) ([], t)
    let (r, t2) := resolveItems goc t1 is
    (.inl (v, ids) :: r, t2)

theorem resolveItems_length {A Tb : Type} (goc : Tb → String → Nat × Tb) (t : Tb)
    (is : List (Sum (Nat × List String) A)) : (resolveItems goc t is).1.length = is.length := by
  induction is generalizing t with
  | nil => rfl
  | cons i is ih => cases i <;> simp [resolveItems, ih]

theorem resolveItems_attr {A Tb : Type} (goc : Tb → String → Nat × Tb) (t : Tb) (a : A)
    (is : List (Sum (Nat × List String) A)) :
    (resolveItems goc t (.inr a :: is)).1 = .inr a :: (resolveItems goc t is).1 := rfl

/-! ## REMOVE (remove.rs:74) -/

/-- One REMOVE item: `e.k` → set null; `e:L…` on a node → remove only labels it has. -/
def removeItem {L : Type} [DecidableEq L] (gone : E → Bool) (s : Store E K V) (labels : E → List L)
    (target : Ent E) (isNull : Bool) (prop : Option K) (rmLabels : Option (List L)) :
    Except String (Store E K V × (E → List L)) :=
  match target with
  | .node e =>
    if gone e then .ok (s, labels) else
    let s1 := match prop with | some k => put s e k none | none => s
    let l1 := match rmLabels with
      | some ls => fun e' => if e' = e then (labels e).filter (fun l => !(ls.filter (· ∈ labels e)).contains l) else labels e'
      | none => labels
    .ok (s1, l1)
  | .rel e =>
    let s1 := match prop with | some k => put s e k none | none => s
    if rmLabels.isSome then .error "Type mismatch: expected Node but was Relationship" else .ok (s1, labels)
  | .other => if isNull then .ok (s, labels) else .error "Type mismatch: expected Node or Relationship"

/-- `REMOVE n.k` makes `n.k` absent; `REMOVE n:L` drops exactly the listed labels the node has;
a deleted node is untouched; `REMOVE r:L` on a relationship is a type error. -/
theorem removeItem_spec {L : Type} [DecidableEq L] (gone : E → Bool) (s : Store E K V)
    (labels : E → List L) (e : E) (k : K) (ls : List L) (hg : gone e = false) :
    (∃ r, removeItem gone s labels (.node e) false (some k) none = .ok r ∧ get r.1 e k = none) ∧
    (∃ r, removeItem gone s labels (.node e) false none (some ls) = .ok r ∧
      ∀ l, l ∈ r.2 e ↔ l ∈ labels e ∧ l ∉ ls) ∧
    removeItem gone s labels (.rel e) false none (some ls) = .error "Type mismatch: expected Node but was Relationship" := by
  refine ⟨⟨(put s e k none, labels), by simp [removeItem, hg], by simp [get_put]⟩,
    ⟨(s, fun e' => if e' = e then (labels e).filter (fun l => !(ls.filter (· ∈ labels e)).contains l)
      else labels e'), by simp [removeItem, hg], ?_⟩, rfl⟩
  intro l
  simp only [ite_true, List.mem_filter, Bool.not_eq_true', List.contains_eq_any_beq, List.any_eq_false,
    beq_iff_eq, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun h => h2 l ⟨h, h1⟩ rfl⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun x hx he => h2 (he ▸ hx.1)⟩

/-! ## `new` / `next` -/

structure OpSt (C T I : Type) where
  child : C
  items : T
  resolved : Option I

def SetSt.new {C T I : Type} (child : C) (items : T) : OpSt C T I := { child, items, resolved := none }
def RemoveSt.new {C T : Type} (child : C) (items : T) : OpSt C T Unit := { child, items, resolved := none }

theorem setNew_spec {C T I : Type} (c : C) (t : T) :
    (SetSt.new c t : OpSt C T I) = ⟨c, t, none⟩ := rfl
theorem removeNew_spec {C T : Type} (c : C) (t : T) : RemoveSt.new c t = ⟨c, t, none⟩ := rfl

/-- set.rs:59-73 / remove.rs:48-59: one child batch, the per-batch effect, the batch (or the
first error) passed on. -/
def opNext {B : Type} (eff : B → Except String Unit) : Option (Except String B) → Option (Except String B)
  | none => none
  | some (.error e) => some (.error e)
  | some (.ok b) => some ((eff b).map fun _ => b)

theorem opNext_spec {B : Type} (eff : B → Except String Unit) (x : Option (Except String B)) :
    opNext eff x = x.map (fun eb => eb.bind fun b => (eff b).map fun _ => b) := by
  cases x with
  | none => rfl
  | some eb => cases eb <;> rfl

end SetRemove
