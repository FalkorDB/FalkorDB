import FalkorExprSemantics.Eval
/-!
# Batch evaluation, UNWIND iteration, compiled regex, map projection
(runtime/eval.rs:86-95, :1102-1260, :1529-1610)
-/
namespace FalkorExpr.Eval

/-! ## `classify_join_keys` (eval.rs:86) over `classify_numeric(_, true, FloatLane::None)`
(batch.rs:191) -/

inductive JV (W : Type) where
  | null | int (i : Int) | other (w : W)

inductive Column (W : Type) where
  | ints (xs : List Int) | values (vs : List (JV W))

variable {W : Type}

def allIntOrNull : List (JV W) → Bool
  | [] => true
  | .int _ :: vs | .null :: vs => allIntOrNull vs
  | .other _ :: _ => false

def intsOf : List (JV W) → List Int
  | [] => []
  | .int i :: vs => i :: intsOf vs
  | _ :: vs => 0 :: intsOf vs        -- `Null` → placeholder 0

/-- `classify_join_keys`: `(column, NullBitmap::from_values)`. -/
def classifyJoinKeys (vs : List (JV W)) : Column W × List Bool :=
  (if allIntOrNull vs then .ints (intsOf vs) else .values vs,
   vs.map fun v => match v with | .null => true | _ => false)

/-- Reading a cell back: the null bit wins, else the column's value. -/
def decode : Column W × List Bool → List (JV W)
  | (.values vs, _) => vs
  | (.ints xs, ns) => (xs.zip ns).map fun (x, n) => if n then .null else .int x

/-- Lossless: decoding the classified column gives back exactly the input values (no
int→float promotion, so no key collisions past 2^53). -/
theorem classifyJoinKeys_lossless (vs : List (JV W)) : decode (classifyJoinKeys vs) = vs := by
  unfold classifyJoinKeys
  split
  · rename_i h
    simp only [decode]
    induction vs with
    | nil => rfl
    | cons v vs ih =>
      cases v with
      | null => simp [allIntOrNull] at h; simp [intsOf, ih h]
      | int i => simp [allIntOrNull] at h; simp [intsOf, ih h]
      | other w => simp [allIntOrNull] at h
  · rfl

/-! ## `eval_batch` (eval.rs:1102), `try_eval_batch_property` (:1118),
`eval_batch_per_row` (:1148) -/

variable {V : Type} (O : Ops V)

/-- The batch: per row, the env; plus the node-id column of a variable, if bound as one. -/
structure BatchM (V : Type) where
  rows : List (Nat → Option V)
  nodeCol : Nat → Option (List Nat)

/-- Row `i` binds variable `v` to node `ids[i]`. -/
def colMatches (O : Ops V) (v : Nat) : List Nat → List (Nat → Option V) → Prop
  | [], [] => True
  | i :: is, r :: rs => (∃ x, r v = some x ∧ O.asNode x = some i ∧ O.asRel x = none) ∧ colMatches O v is rs
  | _, _ => False

/-- The runtime law used by the fast path: `materialize_node_property_values(ids, a)` is
`get_node_attribute` per id (null when absent), and a node column holds the rows' nodes. -/
structure BatchLaws (b : BatchM V) (O : Ops V) (toJV : V → JV W) where
  materialize : List Nat → String → List V
  materialize_eq : ∀ ids a, materialize ids a = ids.map fun id => (O.nodeAttr id a).getD O.null
  col_rows : ∀ v ids, b.nodeCol v = some ids → colMatches O v ids b.rows

variable (toJV : V → JV W)

/-- `eval_batch_per_row`: `?` stops at the first error. -/
def evalBatchPerRow (ev : Ev V) (c : Nat → List V → Except String V) (b : BatchM V) (root : E V) :
    Except String (Column W × List Bool) := do
  let vs ← b.rows.mapM fun env => evalNode O ev c (some env) root
  .ok (classifyJoinKeys (vs.map toJV))

/-- `try_eval_batch_property`: only `var.attr` over a node column. -/
def tryEvalBatchProperty (ev : Ev V) (L : BatchLaws b O toJV) (root : E V) :
    Except String (Option (Column W × List Bool)) :=
  match root with
  | .prop attr (.var v _) => match b.nodeCol v with
    | some ids => do
      let _ ← ev.rt
      .ok (some (classifyJoinKeys ((L.materialize ids attr).map toJV)))
    | none => .ok none
  | _ => .ok none

def evalBatch (ev : Ev V) (c : Nat → List V → Except String V) (L : BatchLaws b O toJV) (root : E V) :
    Except String (Column W × List Bool) := do
  match ← tryEvalBatchProperty O toJV ev L root with
  | some r => .ok r
  | none => evalBatchPerRow O toJV ev c b root

theorem mapM_ok_of_eq {α β : Type} (f : α → Except String β) (g : α → β) (l : List α)
    (h : ∀ a ∈ l, f a = .ok (g a)) : l.mapM f = .ok (l.map g) := by
  induction l with
  | nil => rfl
  | cons a as ih =>
    simp only [List.mapM_cons, h a (by simp), ih (fun x hx => h x (by simp [hx]))]; rfl

/-- **Agreement**: the bulk property path gives exactly what per-row evaluation gives. -/
theorem evalBatch_eq_perRow (b : BatchM V) (params : String → Option V)
    (c : Nat → List V → Except String V) (L : BatchLaws b O toJV) (root : E V) :
    evalBatch O toJV (Ev.fromRuntime params) c L root =
      evalBatchPerRow O toJV (Ev.fromRuntime params) c b root := by
  unfold evalBatch tryEvalBatchProperty
  cases root with
  | prop attr e =>
    cases e with
    | var v n =>
      cases hc : b.nodeCol v with
      | none => simp [hc, bind, Except.bind]
      | some ids =>
        simp only [hc, Ev.fromRuntime, Ev.rt, bind, Except.bind]
        have hm := L.col_rows v ids hc
        have this : b.rows.mapM (fun env => evalNode O ⟨some params⟩ c (some env) (.prop attr (.var v n))) =
            .ok (L.materialize ids attr) := by
          rw [L.materialize_eq]
          clear hc
          generalize b.rows = rs at hm
          induction ids generalizing rs with
          | nil => cases rs with
            | nil => rfl
            | cons _ _ => exact absurd hm id
          | cons i is ih =>
            cases rs with
            | nil => exact absurd hm id
            | cons r rs =>
              obtain ⟨⟨x, hx1, hx2, _⟩, hrest⟩ := hm
              have e1 : evalNode O ⟨some params⟩ c (some r) (.prop attr (.var v n)) =
                  .ok ((O.nodeAttr i attr).getD O.null) := by
                simp [evalNode, resolveVar, hx1, hx2, Ev.rt, bind, Except.bind]
              rw [List.mapM_cons, e1, ih rs hrest]; rfl
        simp only [evalBatchPerRow, this, bind, Except.bind]
    | _ => simp [bind, Except.bind]
  | _ => simp [bind, Except.bind]

/-! ## `eval_iter_expr` (eval.rs:1178): what `UNWIND` iterates -/

/-- The emitted rows: `none` = no rows. `rangeVals` is `RangeIter`'s output (PROVEN
`rangeIter_ok_up`); here a parameter fixed by start/end/step. -/
inductive IterSrc (V : Type) where
  | range (start stop step : Int)
  | listLit (vs : List V)
  | value (v : V)

def evalIterExpr (isList : V → Option (List V)) (isNull : V → Bool)
    (rangeVals : Int → Int → Int → List V) : IterSrc V → Except String (Option (List V))
  | .range a e s =>
    if s = 0 then .error "ArgumentError: step argument to range() can't be 0"
    else if (a > e ∧ s > 0) ∨ (a < e ∧ s < 0) then .ok none
    else if (e - a).natAbs / s.natAbs + 1 > 2^32 - 1 then .error "Range too large"
    else .ok (some (rangeVals a e s))
  | .listLit vs => .ok (some vs)
  | .value v => match isList v with
    | some xs => .ok (some xs)
    | none => if isNull v then .ok none else .ok (some [v])

theorem evalIter_cases (isList : V → Option (List V)) (isNull : V → Bool) (rv : Int → Int → Int → List V) :
    evalIterExpr isList isNull rv (.range 1 5 0) = .error "ArgumentError: step argument to range() can't be 0" ∧
    evalIterExpr isList isNull rv (.range 5 1 1) = .ok none ∧
    (∀ vs, evalIterExpr isList isNull rv (.listLit vs) = .ok (some vs)) ∧
    (∀ v, isList v = none → isNull v = true → evalIterExpr isList isNull rv (.value v) = .ok none) ∧
    (∀ v, isList v = none → isNull v = false → evalIterExpr isList isNull rv (.value v) = .ok (some [v])) := by
  refine ⟨by simp [evalIterExpr], by simp [evalIterExpr], fun _ => rfl, fun v h1 h2 => by simp [evalIterExpr, h1, h2],
    fun v h1 h2 => by simp [evalIterExpr, h1, h2]⟩

theorem evalIter_list (isList : V → Option (List V)) (isNull : V → Bool) (rv : Int → Int → Int → List V)
    (v : V) (xs : List V) (h : isList v = some xs) :
    evalIterExpr isList isNull rv (.value v) = .ok (some xs) := by simp [evalIterExpr, h]

/-! ## `eval_compiled_regex` (eval.rs:1254) and `check_string_or_null` (:1261) -/

inductive RV where
  | null | str (s : String) | bool (b : Bool) | list (vs : List RV) | other (name : String)

def RV.name : RV → String
  | .null => "Null" | .str _ => "String" | .bool _ => "Boolean" | .list _ => "List"
  | .other n => n

/-- `check_string_or_null`: `value_of_type(Union[String, Null])` → error text. -/
def checkStringOrNull : RV → Except String Unit
  | .null | .str _ => .ok ()
  | v => .error s!"Type mismatch: expected String or Null but was {v.name}"

theorem checkStringOrNull_ok (v : RV) : checkStringOrNull v = .ok () ↔ (v = .null ∨ ∃ s, v = .str s) := by
  cases v <;> simp [checkStringOrNull]

inductive RxKind | matches | matchList | replace

structure Regex where
  isMatch : String → Bool
  captures : String → RV             -- `regex_captures_list` (PROVEN in functions_str_list)
  replaceAll : String → String → String

/-- `eval_compiled_regex` (eval.rs:1254) after the children are evaluated. -/
def evalCompiledRegex (re : Regex) (k : RxKind) (text : RV) (repl : Option RV) : Except String RV := do
  checkStringOrNull text
  match k with
  | .matches => match text with
    | .str s => .ok (.bool (re.isMatch s))
    | _ => .ok .null
  | .matchList => match text with
    | .str s => .ok (re.captures s)
    | _ => .ok (.list [])
  | .replace =>
    match repl with
    | some r => do
      checkStringOrNull r
      match text, r with
      | .str t, .null => let _ := t; .ok .null
      | .str t, .str r => .ok (.str (re.replaceAll t r))
      | _, _ => .ok .null
    | none => match text with
      | .str t => .ok (.str (re.replaceAll t ""))
      | _ => .ok .null

/-- The compiled form agrees with the generic function semantics: null text gives null
(`=~`, `replaceRegEx`) or `[]` (`matchRegEx`); a non-string is a type error; an omitted
replacement is `""`. -/
theorem compiledRegex_spec (re : Regex) :
    evalCompiledRegex re .matches .null none = .ok .null ∧
    evalCompiledRegex re .matchList .null none = .ok (.list []) ∧
    evalCompiledRegex re .replace .null (some (.str "x")) = .ok .null ∧
    (∀ t, evalCompiledRegex re .replace (.str t) none = evalCompiledRegex re .replace (.str t) (some (.str ""))) ∧
    evalCompiledRegex re .matches (.bool true) none =
      .error "Type mismatch: expected String or Null but was Boolean" := by
  refine ⟨rfl, rfl, rfl, fun t => rfl, rfl⟩

/-! ## `eval_map_projection` (eval.rs:1529) -/

inductive Item (V : Type) where
  | all                      -- `.*`
  | prop (k : String)        -- `.k`
  | lit (k : String) (v : V) -- `k: expr` (already evaluated)
  | bad

/-- `OrderMap::insert`: replace in place if present, else append. -/
def omInsert (m : List (String × V)) (k : String) (v : V) : List (String × V) :=
  if m.any (fun p => decide (p.1 = k)) then m.map (fun p => if p.1 = k then (k, v) else p)
  else m ++ [(k, v)]

inductive Base (V : Type) where
  | null | node (id : Nat) | rel (id : Nat) | map (m : List (String × V)) | other

def baseAttrs (nodeAttrs relAttrs : Nat → List (String × V)) : Base V → List (String × V)
  | .node id => nodeAttrs id
  | .rel id => relAttrs id
  | .map m => m
  | _ => []

def baseGet (nodeAttrs relAttrs : Nat → List (String × V)) (null : V) (b : Base V) (k : String) : V :=
  ((baseAttrs nodeAttrs relAttrs b).lookup k).getD null

def projItems (na ra : Nat → List (String × V)) (null : V) (b : Base V) :
    List (String × V) → List (Item V) → Except String (List (String × V))
  | acc, [] => .ok acc
  | acc, .all :: is => projItems na ra null b ((baseAttrs na ra b).foldl (fun m kv => omInsert m kv.1 kv.2) acc) is
  | acc, .prop k :: is => projItems na ra null b (omInsert acc k (baseGet na ra null b k)) is
  | acc, .lit k v :: is => projItems na ra null b (omInsert acc k v) is
  | _, .bad :: _ => .error "Encountered unhandled type evaluating map projection"

def evalMapProjection (na ra : Nat → List (String × V)) (null : V) (b : Base V) (items : List (Item V)) :
    Except String (Option (List (String × V))) :=
  match b with
  | .null => .ok none
  | .other => .error "Encountered unhandled type evaluating map projection"
  | _ => (projItems na ra null b [] items).map some

theorem mapProjection_null_other (na ra : Nat → List (String × V)) (n : V) (items : List (Item V)) :
    evalMapProjection na ra n .null items = .ok none ∧
    evalMapProjection na ra n .other items = .error "Encountered unhandled type evaluating map projection" :=
  ⟨rfl, rfl⟩

/-- `n {.k}` is `{k: n.k}` (null when absent), for nodes, relationships and maps alike. -/
theorem mapProjection_prop (na ra : Nat → List (String × V)) (n : V) (b : Base V) (k : String)
    (hb : b ≠ .null ∧ b ≠ .other) :
    evalMapProjection na ra n b [.prop k] = .ok (some [(k, baseGet na ra n b k)]) := by
  cases b <;> simp_all [evalMapProjection, projItems, omInsert, Except.map]

theorem lookup_map_set (m : List (String × V)) (k : String) (v : V)
    (h : m.any (fun p => decide (p.1 = k)) = true) :
    (m.map (fun p => if p.1 = k then (k, v) else p)).lookup k = some v := by
  induction m with
  | nil => simp at h
  | cons p ps ih =>
    obtain ⟨a, b⟩ := p
    by_cases e : a = k
    · subst e; simp [List.lookup]
    · have hb : (k == a) = false := by simp [Ne.symm e]
      simp only [List.any_cons, e, decide_false, Bool.false_or] at h
      simp only [List.map_cons, e, ite_false, List.lookup, hb]
      exact ih h

theorem lookup_append_new (m : List (String × V)) (k : String) (v : V)
    (h : m.any (fun p => decide (p.1 = k)) = false) : (m ++ [(k, v)]).lookup k = some v := by
  induction m with
  | nil => simp [List.lookup]
  | cons p ps ih =>
    obtain ⟨a, b⟩ := p
    have e : a ≠ k := by intro e; simp [e] at h
    have hb : (k == a) = false := by simp [Ne.symm e]
    simp only [List.any_cons, e, decide_false, Bool.false_or] at h
    simp only [List.cons_append, List.lookup, hb]
    exact ih h

/-- A later item overrides an earlier one with the same key. -/
theorem omInsert_lookup (m : List (String × V)) (k : String) (v : V) : (omInsert m k v).lookup k = some v := by
  unfold omInsert
  split
  · rename_i h; exact lookup_map_set m k v h
  · rename_i h; exact lookup_append_new m k v (Bool.eq_false_iff.mpr h)

end FalkorExpr.Eval
