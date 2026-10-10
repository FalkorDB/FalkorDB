/-
# Query parameters and the parser's small helpers
# (cypher.rs:155-358, 360-514)
-/
import FalkorParserGrammar.C.ClausesNP
import FalkorParserGrammar.C.ExprC4
import FalkorParserGrammar.C.Oracle0

namespace FalkorParserGrammar.C

/-! ## Error classification (cypher.rs:131-170, 274-279) -/

/-- `PARAM_VALUE_ERRORS` / `is_param_value_error`: membership in the list. -/
def paramValueErrors : List String :=
  ["Invalid parameter expression.", "Map parameter key must be a string",
   "ArgumentError: integer overflow in unary minus"]
def isParamValueError (e : String) : Bool := paramValueErrors.contains e
theorem isParamValueError_spec (e : String) : isParamValueError e = true ↔ e ∈ paramValueErrors := by
  simp [isParamValueError]

/-- `TOO_DEEP`, `too_deep` (message prefix + limit) and `is_too_deep` (prefix test). -/
def tooDeepPrefix : String := "Query nesting exceeds the maximum depth of"
def tooDeep (limit : Nat) : String := tooDeepPrefix ++ " " ++ toString limit
/-- `err.starts_with(TOO_DEEP)`, as a prefix test on characters. -/
def isTooDeep (e : String) : Bool := tooDeepPrefix.toList.isPrefixOf e.toList

theorem isPrefixOf_append_self (a b : List Char) : a.isPrefixOf (a ++ b) = true := by
  induction a with
  | nil => rfl
  | cons x a ih => simp [List.isPrefixOf, ih]
/-- Every `too_deep` message is recognised by `is_too_deep` (so it survives
the parameter-error summary, cypher.rs:385-391). `format_error` wraps the
message; that wrapper is `proofs/lexer`'s `format_error`, which keeps the
message as a prefix. -/
theorem isTooDeep_tooDeep (n : Nat) : isTooDeep (tooDeep n) = true := by
  simp only [isTooDeep, tooDeep, String.toList_append, List.append_assoc]
  exact isPrefixOf_append_self _ _

/-! ## `ExpressionListType::is_end_token` (cypher.rs:175-183) -/

inductive ELT | oneOrMore | closedBy (t : CT)
def ELT.isEnd : ELT → CT → Bool
  | .oneOrMore, _ => false
  | .closedBy t, c => c == t
theorem ELT.isEnd_spec (e : ELT) (c : CT) :
    e.isEnd c = true ↔ ∃ t, e = .closedBy t ∧ c = t := by
  cases e <;> simp [ELT.isEnd]

/-! ## Parser state (cypher.rs:262-358) -/

/-- `Parser::new`: the counter starts at 0 (and depth / heights at 0, the
pattern-comprehension flag unset). -/
def newState (ts : List CT) : PS := (ts, 0)
theorem newState_spec (ts : List CT) : (newState ts).1 = ts ∧ (newState ts).2 = 0 := ⟨rfl, rfl⟩

/-- `save_state` / `restore_state`: position and anonymous counter. -/
def saveState : P PS := getS
def restoreState (st : PS) : P Unit := setS st
/-- **Backtracking restores the counter**: `restore_state(st)` puts back
exactly the saved position and counter, whatever state it is run in. -/
theorem restore_state_spec (st s : PS) : restoreState st s = .ok () st := rfl
theorem save_state_spec (s : PS) : saveState s = .ok s s := rfl

/-- `with_child_height(parse)`: the outer tally is saved, the inner one starts
at 0 and is returned with the result, and the outer tally is put back. -/
def withChildHeight {α} (outer : Nat) (parse : Nat → α × Nat) : (α × Nat) × Nat :=
  let r := parse 0
  ((r.1, r.2), outer)
theorem withChildHeight_spec {α} (outer : Nat) (parse : Nat → α × Nat) :
    (withChildHeight outer parse).2 = outer ∧ (withChildHeight outer parse).1 = parse 0 := ⟨rfl, rfl⟩

/- The recursive height. -/
mutual
def height : ET → Nat
  | .node _ cs => 1 + heightL cs
def heightL : List ET → Nat
  | [] => 0
  | c :: cs => max (height c) (heightL cs)
end

/-- `tree_height` (cypher.rs:314-322): the explicit stack of (node, depth),
with a budget. -/
def thLoop : Nat → List (ET × Nat) → Nat → Option Nat
  | 0, _, _ => none
  | _ + 1, [], h => some h
  | f + 1, (n, d) :: rest, h => thLoop f (n.kids.map (fun c => (c, d + 1)) ++ rest) (max h d)

def treeHeight (f : Nat) (t : ET) : Option Nat := thLoop f [(t, 1)] 0

/-- The worklist's answer: the max over pending (node, depth) of depth + height − 1. -/
def pendMax : List (ET × Nat) → Nat
  | [] => 0
  | (n, d) :: rest => max (d + height n - 1) (pendMax rest)

theorem pendMax_append (a b : List (ET × Nat)) : pendMax (a ++ b) = max (pendMax a) (pendMax b) := by
  induction a with
  | nil => simp [pendMax]
  | cons x a ih => obtain ⟨n, d⟩ := x; simp [pendMax, ih, Nat.max_assoc]

theorem height_pos (t : ET) : 1 ≤ height t := by cases t; simp [height]

theorem pendMax_kids (cs : List ET) (d : Nat) (hd : 1 ≤ d) :
    pendMax (cs.map (fun c => (c, d + 1))) = if cs = [] then 0 else d + heightL cs := by
  induction cs with
  | nil => simp [pendMax]
  | cons c cs ih =>
    simp only [List.map_cons, pendMax, ih, heightL]
    have := height_pos c
    by_cases hcs : cs = []
    · subst hcs; simp [heightL]
    · simp only [hcs, ite_false, reduceCtorEq]
      rw [Nat.max_def, Nat.max_def]; split <;> split <;> omega

/-- **tree_height computes the height** (whenever the budget suffices): the
iterative walk returns `height t`. -/
theorem thLoop_spec : ∀ f (w : List (ET × Nat)) (h r : Nat), (∀ p ∈ w, 1 ≤ p.2) →
    thLoop f w h = some r → r = max h (pendMax w)
  | 0, _, _, _, _, hr => by simp [thLoop] at hr
  | _ + 1, [], h, r, _, hr => by simp [thLoop] at hr; simp [hr, pendMax]
  | f + 1, (n, d) :: rest, h, r, hw, hr => by
    simp only [thLoop] at hr
    have hd : 1 ≤ d := hw (n, d) (by simp)
    have ih := thLoop_spec f _ _ r (by
      intro p hp
      simp only [List.mem_append, List.mem_map] at hp
      rcases hp with ⟨c, _, rfl⟩ | hp
      · simp
      · exact hw p (by simp [hp])) hr
    rw [ih, pendMax_append, pendMax]
    obtain ⟨o, cs⟩ := n
    simp only [ET.kids, height]
    rw [pendMax_kids cs d hd]
    split
    · subst_vars; simp [heightL]
    · omega

theorem treeHeight_spec (f : Nat) (t : ET) (r : Nat) (h : treeHeight f t = some r) : r = height t := by
  have := thLoop_spec f [(t, 1)] 0 r (by simp) h
  simp [pendMax] at this; omega

/-! ## Literal parameters (cypher.rs:360-514) -/

/-- `Value` as far as literals go. -/
inductive LV
  | null | bool (b : Bool) | int (i : Int) | float | str (s : Nat)
  | list (vs : List LV) | map (es : List (Nat × LV))
  deriving Repr

def minI64 : Int := -9223372036854775808

/-- `OrderMap::insert` (ordermap.rs:76-89): replace in place, else append. -/
def omInsert {α} (k : Nat) (v : α) : List (Nat × α) → List (Nat × α)
  | [] => [(k, v)]
  | (k', v') :: rest => if k' = k then (k, v) :: rest else (k', v') :: omInsert k v rest

/-- Negation of a literal (cypher.rs:431-442); `Value::Float(-f)` is kept
abstract. -/
def negLit : LV → Option LV
  | .int i => if i = minI64 then some (.int minI64) else some (.int (-i))
  | .float => some .float
  | _ => some .null

mutual
/-- `parse_literal` (cypher.rs:429-453). -/
def litP : Nat → P LV
  | 0 => fuelP
  | n + 1 => do
    let neg ← opt .dash
    if neg then do
      let v ← ulitP n
      match negLit v with
      | some r => pure r
      | none => fail
    else do
      let v ← ulitP n
      match v with
      | .int i => if i = minI64 then fail else pure v
      | _ => pure v
/-- `parse_unsigned_literal` (cypher.rs:455-481). -/
def ulitP : Nat → P LV
  | 0 => fuelP
  | n + 1 => do
    let c ← peek
    match c with
    | .kw .null => do next; pure .null
    | .kw .true_ => do next; pure (.bool true)
    | .kw .false_ => do next; pure (.bool false)
    | .int i => do next; pure (.int i)
    | .float => do next; pure .float
    | .str s => do next; pure (.str s)
    | .lbrack => listLitV n
    | .lbrace => mapLitV n
    | _ => fail
/-- `parse_literal_list` (cypher.rs:483-496). -/
def listLitV : Nat → P LV
  | 0 => fuelP
  | n + 1 => do
    next
    let e ← opt .rbrack
    if e then pure (.list []) else listItems n []
def listItems : Nat → List LV → P LV
  | 0, _ => fuelP
  | n + 1, acc => do
    let v ← litP n
    let e ← opt .rbrack
    if e then pure (.list (acc ++ [v])) else do
      tok .comma
      listItems n (acc ++ [v])
/-- `parse_literal_map` (cypher.rs:498-514). -/
def mapLitV : Nat → P LV
  | 0 => fuelP
  | n + 1 => do
    next
    let e ← opt .rbrace
    if e then pure (.map []) else mapItems n []
def mapItems : Nat → List (Nat × LV) → P LV
  | 0, _ => fuelP
  | n + 1, acc => do
    let k ← ident
    tok .colon
    let v ← litP n
    let acc := omInsert k v acc
    let e ← opt .rbrace
    if e then pure (.map acc) else do
      tok .comma
      mapItems n acc
end

/-- **i64::MIN**: `-9223372036854775808` is read as i64::MIN (the lexer hands
back MIN for the digits), the unsigned digits alone are an overflow. -/
theorem lit_min_neg : (litP 5 ([.dash, .int minI64], 0)).isOk = true := by decide
theorem lit_min_pos : (litP 5 ([.int minI64], 0)).isOk = false := by decide
/-- Repeated map keys: the last value wins, in the first key's position. -/
theorem omInsert_replaces : omInsert 1 (2 : Nat) [(1, 1), (3, 3)] = [(1, 2), (3, 3)] := by decide

theorem np_lit : ∀ n, NP (litP n) ∧ NP (ulitP n) ∧ NP (listLitV n) ∧ (∀ acc, NP (listItems n acc)) ∧
    NP (mapLitV n) ∧ (∀ acc, NP (mapItems n acc))
  | 0 => ⟨np_fuelP, np_fuelP, np_fuelP, fun _ => np_fuelP, np_fuelP, fun _ => np_fuelP⟩
  | n + 1 => by
    obtain ⟨a, b, c, d, e, f⟩ := np_lit n
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · unfold litP; np_go
    · unfold ulitP; np_go
    · unfold listLitV; np_go
    · intro acc; unfold listItems; np_go
    · unfold mapLitV; np_go
    · intro acc; unfold mapItems; np_go

mutual
/-- A literal (Cypher.g4 `oC_Literal` restricted to constants), with the
value it denotes. -/
inductive GLit : PS → LV → PS → Prop
  | pos {s v s'} : GULit s v s' → (∀ i, v = .int i → i ≠ minI64) → GLit s v s'
  | neg {s v s' r} : s.cur = .dash → GULit s.adv v s' → negLit v = some r → GLit s r s'
inductive GULit : PS → LV → PS → Prop
  | null {s} : s.cur = .kw .null → GULit s .null s.adv
  | tru {s} : s.cur = .kw .true_ → GULit s (.bool true) s.adv
  | fls {s} : s.cur = .kw .false_ → GULit s (.bool false) s.adv
  | int {s i} : s.cur = .int i → GULit s (.int i) s.adv
  | float {s} : s.cur = .float → GULit s .float s.adv
  | str {s t} : s.cur = .str t → GULit s (.str t) s.adv
  | list {s vs s'} : s.cur = .lbrack → GLItems s.adv [] vs s' → GULit s (.list vs) s'
  | map {s es s'} : s.cur = .lbrace → GMItems s.adv [] es s' → GULit s (.map es) s'
inductive GLItems : PS → List LV → List LV → PS → Prop
  | empty {s acc} : acc = [] → s.cur = .rbrack → GLItems s acc acc s.adv
  | last {s acc v s1} : GLit s v s1 → s1.cur = .rbrack → GLItems s acc (acc ++ [v]) s1.adv
  | more {s acc v s1 out s'} : GLit s v s1 → s1.cur = .comma → GLItems s1.adv (acc ++ [v]) out s' →
      GLItems s acc out s'
inductive GMItems : PS → List (Nat × LV) → List (Nat × LV) → PS → Prop
  | empty {s acc} : acc = [] → s.cur = .rbrace → GMItems s acc acc s.adv
  | last {s acc k v s1} : identOf s.cur = some k → s.adv.cur = .colon → GLit s.adv.adv v s1 →
      s1.cur = .rbrace → GMItems s acc (omInsert k v acc) s1.adv
  | more {s acc k v s1 out s'} : identOf s.cur = some k → s.adv.cur = .colon → GLit s.adv.adv v s1 →
      s1.cur = .comma → GMItems s1.adv (omInsert k v acc) out s' → GMItems s acc out s'
end

theorem lit_sound : ∀ n,
    (∀ s v s', litP n s = .ok v s' → GLit s v s') ∧ (∀ s v s', ulitP n s = .ok v s' → GULit s v s') ∧
    (∀ s v s', listLitV n s = .ok v s' → ∃ vs, v = .list vs ∧ GLItems s.adv [] vs s') ∧
    (∀ acc s v s', listItems n acc s = .ok v s' → acc ≠ [] ∨ True →
      ∃ vs, v = .list vs ∧ (GLItems s acc vs s')) ∧
    (∀ s v s', mapLitV n s = .ok v s' → ∃ es, v = .map es ∧ GMItems s.adv [] es s') ∧
    (∀ acc s v s', mapItems n acc s = .ok v s' → ∃ es, v = .map es ∧ GMItems s acc es s')
  | 0 => by refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> intros <;> simp_all [litP, ulitP, listLitV, listItems, mapLitV, mapItems]
  | n + 1 => by
    obtain ⟨il, iu, ill, ili, iml, imi⟩ := lit_sound n
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro s v s' h
      simp only [litP] at h
      peel h => ng s1 h1
      cases ng
      · have := (opt_eq h1).2 rfl; subst s1
        simp only [Bool.false_eq_true, ite_false] at h
        peel h => u s2 h2
        have gu := iu _ _ _ h2
        cases u <;> (try dsimp only at h)
        case int i =>
          split at h
          · simp at h
          · obtain ⟨rfl, rfl⟩ := pure_ok h
            exact .pos gu (fun j hj => by cases hj; assumption)
        all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h; exact .pos gu (fun j hj => by cases hj))
      · obtain ⟨hd, rfl⟩ := (opt_eq h1).1 rfl
        simp only [ite_true] at h
        peel h => u s2 h2
        generalize hn : negLit u = r at h
        cases r with
        | none => simp at h
        | some r => obtain ⟨rfl, rfl⟩ := pure_ok h; exact .neg hd (iu _ _ _ h2) hn
    · intro s v s' h
      simp only [ulitP] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      generalize hc : s.cur = c at h
      cases c
      case kw k =>
        cases k <;> (try dsimp only at h)
        case null => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .null hc
        case true_ => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .tru hc
        case false_ => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .fls hc
        all_goals simp at h
      case int i => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .int hc
      case float => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .float hc
      case str t => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .str hc
      case lbrack => obtain ⟨vs, rfl, g⟩ := ill _ _ _ h; exact .list hc g
      case lbrace => obtain ⟨es, rfl, g⟩ := iml _ _ _ h; exact .map hc g
      all_goals simp at h
    · intro s v s' h
      simp only [listLitV] at h
      peel h => u s1 h1; have := next_ok h1; subst this
      peel h => e s2 h2
      cases e
      · have := (opt_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h
        exact ili [] _ _ _ h (.inr trivial)
      · obtain ⟨hr, rfl⟩ := (opt_eq h2).1 rfl
        simp only [ite_true] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨[], rfl, .empty rfl hr⟩
    · intro acc s v s' h _
      simp only [listItems] at h
      peel h => x s1 h1
      have gl := il _ _ _ h1
      peel h => e s2 h2
      cases e
      · have := (opt_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h
        peel h => u s3 h3; obtain ⟨hc, rfl⟩ := tok_ok h3
        obtain ⟨vs, rfl, g⟩ := ili _ _ _ _ h (.inr trivial)
        exact ⟨vs, rfl, .more gl hc g⟩
      · obtain ⟨hr, rfl⟩ := (opt_eq h2).1 rfl
        simp only [ite_true] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨_, rfl, .last gl hr⟩
    · intro s v s' h
      simp only [mapLitV] at h
      peel h => u s1 h1; have := next_ok h1; subst this
      peel h => e s2 h2
      cases e
      · have := (opt_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h
        exact imi [] _ _ _ h
      · obtain ⟨hr, rfl⟩ := (opt_eq h2).1 rfl
        simp only [ite_true] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨[], rfl, .empty rfl hr⟩
    · intro acc s v s' h
      simp only [mapItems] at h
      peel h => k s1 h1; obtain ⟨hk, rfl⟩ := ident_ok h1
      peel h => u s2 h2; obtain ⟨hcol, rfl⟩ := tok_ok h2
      peel h => x s3 h3
      have gl := il _ _ _ h3
      peel h => e s4 h4
      cases e
      · have := (opt_eq h4).2 rfl; subst s4
        simp only [Bool.false_eq_true, ite_false] at h
        peel h => u s5 h5; obtain ⟨hc, rfl⟩ := tok_ok h5
        obtain ⟨es, rfl, g⟩ := imi _ _ _ _ h
        exact ⟨es, rfl, .more hk hcol gl hc g⟩
      · obtain ⟨hr, rfl⟩ := (opt_eq h4).1 rfl
        simp only [ite_true] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨_, rfl, .last hk hcol gl hr⟩

/-! ## `parse_param_value` and `parse_parameters` (cypher.rs:360-427) -/

/-- The fast path is taken iff a literal parses and is followed by an
identifier (the next `name=` or the query) or the end. -/
def endsLiteral : CT → Bool
  | .ident _ | .kw _ | .eof => true
  | _ => false

/-- `parse_param_value`; `evp` is `evaluate_param` (runtime evaluation of a
non-literal value, outside this model). -/
def paramValueP (o : Oracle) (evp : ET → Option LV) (n : Nat) : P LV := do
  let s0 ← getS
  let r ← attempt (litP n)
  let c ← peek
  match r with
  | some v => if endsLiteral c then pure v else fallback s0
  | none => fallback s0
where
  fallback (s0 : PS) : P LV := do
    setS s0
    let e ← o.pe false
    match evp e with
    | some v => pure v
    | none => fail

/-- The `CYPHER` identifier (cypher.rs:369, not a keyword). -/
def nmCypher : Nat := 6000002

/-- `insert` into the parameter `HashMap`: a repeated name keeps the last value. -/
def hmInsert {α} (k : Nat) (v : α) (m : List (Nat × α)) : List (Nat × α) :=
  (m.filter (fun p => p.1 != k)) ++ [(k, v)]

/-- The inner `while let Some(id) = try_parse_ident()` (cypher.rs:372-394). -/
def paramPairs (o : Oracle) (evp : ET → Option LV) (n : Nat) :
    Nat → List (Nat × LV) → P (List (Nat × LV))
  | 0, _ => fuelP
  | f + 1, acc => do
    let st ← getS
    let a ← tryIdent
    match a with
    | none => pure acc
    | some id => do
      let eq ← opt .eq
      if !eq then do setS st; pure acc
      else do
        let v ← paramValueP o evp n
        paramPairs o evp n f (hmInsert id v acc)

/-- `parse_parameters` (cypher.rs:360-399): the parameters and the rest of
the query (here: the remaining tokens). -/
def paramsP (o : Oracle) (evp : ET → Option LV) (n : Nat) :
    Nat → List (Nat × LV) → P (List (Nat × LV))
  | 0, _ => fuelP
  | f + 1, acc => do
    let c ← peek
    if c = .ident nmCypher then do
      next
      let acc ← paramPairs o evp n f acc
      paramsP o evp n f acc
    else pure acc

theorem hmInsert_lookup {α} (k : Nat) (v : α) (m : List (Nat × α)) :
    ((hmInsert k v m).filter (fun p => p.1 == k)) = [(k, v)] := by
  simp [hmInsert, List.filter_append, List.filter_filter]

theorem paramValue_fast (o : Oracle) (evp : ET → Option LV) (n : Nat) (s : PS) v s1
    (h : litP n s = .ok v s1) (he : endsLiteral s1.cur = true) :
    paramValueP o evp n s = .ok v s1 := by
  simp only [paramValueP, run_bind, run_getS, attempt, h, peek, he, ite_true, run_pure]

section
variable (o : Oracle) (ho : o.Safe)
include ho

theorem np_paramValueP (evp : ET → Option LV) (n : Nat) : NP (paramValueP o evp n) := by
  have := (np_lit n).1; have := np_attempt this; have := np_pe o ho
  have hs : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
  unfold paramValueP paramValueP.fallback; np_go

theorem np_paramPairs (evp : ET → Option LV) (n : Nat) : ∀ f acc, NP (paramPairs o evp n f acc)
  | 0, _ => np_fuelP
  | f + 1, acc => by
    have := np_paramPairs evp n f; have := np_paramValueP o ho evp n
    have hs : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
    unfold paramPairs; np_go

/-- **parse_parameters never panics.** -/
theorem np_paramsP (evp : ET → Option LV) (n : Nat) : ∀ f acc, NP (paramsP o evp n f acc)
  | 0, _ => np_fuelP
  | f + 1, acc => by
    have := np_paramsP evp n f; have := np_paramPairs o ho evp n
    unfold paramsP; np_go

end

/-- `CYPHER a=1 MATCH ...`: the pair loop stops at `MATCH` (no `=` follows)
and leaves it for the query. -/
theorem params_stop_at_query :
    ((paramsP ⟨fun _ => fail, fail, fail, fun _ _ => 0, fun _ => none⟩ (fun _ => none) 5 5 [])
      ([.ident nmCypher, .ident 1, .eq, .int 7, .kw .match_], 0)).isOk = true := by decide

end FalkorParserGrammar.C
