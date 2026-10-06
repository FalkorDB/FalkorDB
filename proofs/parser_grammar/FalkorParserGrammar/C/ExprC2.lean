/-
# Pattern comprehensions, `[`-dispatch, shortestPath, and the
# pattern-comprehension guards (cypher.rs:1737-1924, 2752-2913, 3430-3440)
-/
import FalkorParserGrammar.C.ExprC

namespace FalkorParserGrammar.C

/-! ## The forbidden-pattern-comprehension flag (cypher.rs:2752-2768) -/

/-- `without_pattern_comprehensions(place, parse)`: run `parse` with the flag
set to `place`; the outer flag comes back whatever `parse` returned. The
flag is the second component of the result. -/
def withoutPC {α} (outer : Option Nat) (place : Nat) (parse : Option Nat → α × Option Nat) :
    α × Option Nat := ((parse (some place)).1, outer)

theorem withoutPC_spec {α} (outer : Option Nat) (place : Nat) (parse : Option Nat → α × Option Nat) :
    (withoutPC outer place parse).1 = (parse (some place)).1 ∧ (withoutPC outer place parse).2 = outer :=
  ⟨rfl, rfl⟩

/-- `reject_forbidden_pattern_comprehension` (cypher.rs:2763-2766). -/
def rejectForbidden (flag : Option Nat) : Except Nat Unit :=
  match flag with
  | some place => .error place
  | none => .ok ()

theorem rejectForbidden_spec (flag : Option Nat) :
    rejectForbidden flag = .ok () ↔ flag = none := by
  cases flag <;> simp [rejectForbidden]

/-- `reject_aggregate` (cypher.rs:3430-3440). -/
def rejectAggregate (t : ET) : P Unit := if (aggName t).isSome then fail else pure ()

theorem rejectAggregate_spec (t : ET) (s : PS) :
    (rejectAggregate t s = .ok () s ↔ ¬ HasAgg t) ∧ rejectAggregate t s ≠ .panic := by
  unfold rejectAggregate
  rw [← find_aggregate_name_spec]
  constructor
  · split <;> simp_all
  · split <;> simp

/-- `parse_inline_properties` (cypher.rs:2915-2929): the map's own error is
kept only when it is an unknown-function error. -/
def inlineErr (isUnknownFn : String → Bool) (msg : String) (r : Except String ET) : Except String ET :=
  match r with
  | .ok t => .ok t
  | .error e => .error (if isUnknownFn e then e else msg)

theorem inlineErr_spec (u : String → Bool) (msg : String) (r : Except String ET) :
    (∀ t, r = .ok t → inlineErr u msg r = .ok t) ∧
    (∀ e, r = .error e → inlineErr u msg r = .error (if u e then e else msg)) := by
  constructor <;> intro x h <;> subst h <;> rfl

/-! ## Pattern comprehension (cypher.rs:2867-2913; Cypher.g4:397-398) -/

def pcChain (o : Oracle) (F : Nat) : Nat → QNode Nat → QG Nat → P (QG Nat)
  | 0, _, _ => fuelP
  | n + 1, left, g => do
    let c ← peek
    if c = .dash ∨ c = .lt then do
      let rr ← relP o F .match_ left
      let g := ((g.addNode rr.2).2.addRel rr.1).2
      pcChain o F n rr.2 g
    else pure g

/-- `parse_pattern_comprehension(path_var)`, after `[` (and `p =`). -/
def patCompP (o : Oracle) (F : Nat) (pv : Option Nat) : P ET := do
  let first ← nodeP o F
  let c ← peek
  if !(c = .dash ∨ c = .lt) then fail else do
  let _g ← pcChain o F F first (QG.empty.addNode first).2
  let w ← optK .where_
  let cond ← (if w then o.pe false else pure (leaf (.bool true)))
  tok .pipe
  let res ← o.pe false
  tok .rbrack
  pure (.node .patComp [cond, res])

/-! ## `parse_list_literal_or_comprehension` (cypher.rs:2778-2822) -/

/-- Run a speculative parse: an error becomes `none` (the caller restores). -/
def attempt {α} (x : P α) : P (Option α) := fun s =>
  match x s with
  | .ok a s' => .ok (some a) s'
  | .err => .ok none s
  | .panic => .panic
  | .fuel => .fuel

/-- Alternatives 2-4 of `parse_list_literal_or_comprehension`, from the
restored state `saved`. -/
def listLitRest (o : Oracle) (F : Nat) (forb : Bool) (saved : PS) : P (ET × Bool) := do
  setS saved
  -- 2) `[p = (pattern) ... | expr]`
  let a2 ← tryIdent
  let eq ← (match a2 with | some _ => opt .eq | none => pure false)
  let c ← peek
  let r2 ← (match a2, eq, c with
    | some v, true, .lparen => attempt (patCompP o F (some v))
    | _, _, _ => pure none)
  match r2 with
  | some t => do
    if forb then fail else do
    rejectAggregate t
    pure (t, false)
  | none => do
    setS saved
    -- 3) `[(pattern) ... | expr]`
    let c3 ← peek
    let r3 ← (if c3 = .lparen then attempt (patCompP o F none) else pure none)
    match r3 with
    | some t => do
      if forb then fail else do
      rejectAggregate t
      pure (t, false)
    | none => do
      setS saved
      -- 4) a list literal
      let rb ← opt .rbrack
      pure (leaf .list, !rb)

/-- After `[`; `forb` is the forbidden-pattern-comprehension flag. -/
def listLitP (o : Oracle) (F : Nat) (forb : Bool) : P (ET × Bool) := do
  let saved ← getS
  -- 1) `[var IN ...`
  let a ← tryIdent
  let isIn ← (match a with | some _ => optK .in_ | none => pure false)
  match a, isIn with
  | some v, true => do let t ← listCompP (o.pe false) v; pure (t, false)
  | _, _ => listLitRest o F forb saved

/-! ## shortestPath (cypher.rs:1737-1924) -/

def spTypes : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let t ← ident
    let p ← opt .pipe
    let c ← opt .colon
    if p || c then spTypes n (acc ++ [t]) else pure (acc ++ [t])

/-- The edge-filter skipper (cypher.rs:1819-1846): `{` deepens, `}`
shallows and stops at depth ≤ 0, `]` stops before it; at end of input it
never stops (the model's budget runs out: `fuel`). -/
def skipP : Nat → Int → P Unit
  | 0, _ => fuelP
  | n + 1, d => do
    let c ← peek
    match c with
    | .lbrace => do next; skipP n (d + 1)
    | .rbrace => do next; if d - 1 ≤ 0 then pure () else skipP n (d - 1)
    | .rbrack => pure ()
    | _ => do next; skipP n d

/-- `Token::Parameter(_) | Token::LBracket` after the range (cypher.rs:1810-1814). -/
def isFilter : CT → Bool
  | .param _ => true
  | .lbrace => true
  | _ => false

/-- shortestPath's `*` range (cypher.rs:1784-1807). -/
def spRangeP : P (Nat × Option Nat) := do
  let st ← opt .star
  if st then do
    let start ← hopP
    let dd ← opt .dotdot
    if dd then do
      let e ← hopP
      pure ((start.map asU32).getD 1, e.map asU32)
    else match start with
      | some x => pure (asU32 x, some (asU32 x))
      | none => pure (1, none)
  else pure (1, some 1)

/-- `parse_shortest_path_expr`, after `shortestPath(`. -/
def shortestP (F : Nat) : P ET := do
  let c0 ← peek
  if c0 ≠ .lparen then fail else do
  tok .lparen
  let a ← tryIdent
  match a with
  | none => fail
  | some src => do
    let c1 ← peek
    if c1 = .lbrace then fail else do
    tok .rparen
    let inc ← opt .lt
    tok .dash
    let det ← opt .lbrack
    let d ← (if det then do
        let _ ← tryIdent
        let col ← opt .colon
        let ts ← (if col then spTypes F [] else pure [])
        let mm ← spRangeP
        let c2 ← peek
        let filt := isFilter c2
        if filt then skipP F 0 else pure ()
        tok .rbrack
        pure (ts, mm.1, mm.2, filt)
      else pure ([], 1, some 1, false))
    tok .dash
    let out ← opt .gt
    tok .lparen
    let b ← tryIdent
    match b with
    | none => fail
    | some dst => do
      let c3 ← peek
      if c3 = .lbrace then fail else do
      tok .rparen
      tok .rparen
      if d.2.1 > 1 then fail else do
      if d.2.2.2 then fail else do
      let ends := if inc && !out then (dst, src) else (src, dst)
      pure (.node (.shortest d.1 d.2.1 d.2.2.1 (inc || out)) [leaf (.var ends.1), leaf (.var ends.2)])

/-- **An edge filter is never accepted**: whenever `shortestPath` returns,
the relationship had no `{...}` / `$param` filter. With a filter the result
is an error, or (input ending inside the filter) no result at all — the
known hang, `skipFilter_eof_diverges`. -/
theorem skipP_eof : ∀ n (d : Int) (k : Nat), skipP n d ([], k) = .fuel
  | 0, _, _ => rfl
  | n + 1, d, k => by
    simp only [skipP, run_bind, peek, PS.cur, next, PS.adv, List.tail_nil]; exact skipP_eof n d k

end FalkorParserGrammar.C
