/-
# Pattern comprehensions, `[`-dispatch, shortestPath, and the
# pattern-comprehension guards (cypher.rs)
-/
import FalkorParserGrammar.C.ExprC

namespace FalkorParserGrammar.C

/-! ## The forbidden-pattern-comprehension flag (cypher.rs) -/

/-- `without_pattern_comprehensions(place, parse)`: run `parse` with the flag
set to `place`; the outer flag comes back whatever `parse` returned. The
flag is the second component of the result. -/
def withoutPC {α} (outer : Option Nat) (place : Nat) (parse : Option Nat → α × Option Nat) :
    α × Option Nat := ((parse (some place)).1, outer)

theorem withoutPC_spec {α} (outer : Option Nat) (place : Nat) (parse : Option Nat → α × Option Nat) :
    (withoutPC outer place parse).1 = (parse (some place)).1 ∧ (withoutPC outer place parse).2 = outer :=
  ⟨rfl, rfl⟩

/-- `reject_forbidden_pattern_comprehension` (cypher.rs). -/
def rejectForbidden (flag : Option Nat) : Except Nat Unit :=
  match flag with
  | some place => .error place
  | none => .ok ()

theorem rejectForbidden_spec (flag : Option Nat) :
    rejectForbidden flag = .ok () ↔ flag = none := by
  cases flag <;> simp [rejectForbidden]

/-- `reject_aggregate` (cypher.rs). -/
def rejectAggregate (t : ET) : P Unit := if (aggName t).isSome then fail else pure ()

theorem rejectAggregate_spec (t : ET) (s : PS) :
    (rejectAggregate t s = .ok () s ↔ ¬ HasAgg t) ∧ rejectAggregate t s ≠ .panic := by
  unfold rejectAggregate
  rw [← find_aggregate_name_spec]
  constructor
  · split <;> simp_all
  · split <;> simp

/-- `parse_inline_properties` (cypher.rs): the map's own error is
kept only when it is an unknown-function error. -/
def inlineErr (isUnknownFn : String → Bool) (msg : String) (r : Except String ET) : Except String ET :=
  match r with
  | .ok t => .ok t
  | .error e => .error (if isUnknownFn e then e else msg)

theorem inlineErr_spec (u : String → Bool) (msg : String) (r : Except String ET) :
    (∀ t, r = .ok t → inlineErr u msg r = .ok t) ∧
    (∀ e, r = .error e → inlineErr u msg r = .error (if u e then e else msg)) := by
  constructor <;> intro x h <;> subst h <;> rfl

/-! ## Pattern comprehension (cypher.rs; Cypher.g4:397-398) -/

/-- The chain of a pattern comprehension (cypher.rs). The state is the graph and
`repeated`: whether some relationship alias was already in it. Since #3289 a repeat does not stop
the loop (`repeated.get_or_insert(e)`): the parse goes on, and the caller reports it once the whole
comprehension has parsed. -/
def pcChain (o : Oracle) (F : Nat) : Nat → QNode Alias → QG Alias × Bool → P (QG Alias × Bool)
  | 0, _, _ => fuelP
  | n + 1, left, g => do
    let c ← peek
    if c = .dash ∨ c = .lt then do
      let rr ← relP o F .match_ left
      let ag := (g.1.addNode rr.2).2.addRel rr.1
      pcChain o F n rr.2 (ag.2, g.2 || !ag.1)
    else pure g

/-- `parse_pattern_comprehension(path_var)`, after `[` (and `p =`). Returns the tree and the
`repeated` flag (cypher.rs, #3289): `Some(error)` for a repeated relationship alias,
which the caller turns into a hard error after the attempt succeeded. -/
def patCompP (o : Oracle) (F : Nat) (pv : Option Nat) : P (ET × Bool) := do
  let first ← nodeP o F
  let c ← peek
  if !(c = .dash ∨ c = .lt) then fail else do
  let g ← pcChain o F F first ((QG.empty.addNode first).2, false)
  let w ← optK .where_
  let cond ← (if w then o.pe false else pure (leaf (.bool true)))
  tok .pipe
  let res ← o.pe false
  tok .rbrack
  pure (.node .patComp [cond, res], g.2)

/-! ## `parse_list_literal_or_comprehension` (cypher.rs) -/

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
  | some (t, rep) => do
    if rep then fail else do
    if forb then fail else do
    rejectAggregate t
    pure (t, false)
  | none => do
    setS saved
    -- 3) `[(pattern) ... | expr]`
    let c3 ← peek
    let r3 ← (if c3 = .lparen then attempt (patCompP o F none) else pure none)
    match r3 with
    | some (t, rep) => do
      if rep then fail else do
      if forb then fail else do
      rejectAggregate t
      pure (t, false)
    | none => do
      setS saved
      -- 4) a list literal
      let rb ← opt .rbrack
      pure (leaf .list, !rb)

/-- `list_has_top_level_pipe` (cypher.rs, #3104), on the tokens
after the `[`: one counter `open` for all three bracket kinds; a `|` at
`open = 0` is `true`, a `,` there or a closer that would take `open` below 0
is `false`, end of input (or a lexer error) is `false`. `[]` is
`EndOfFile`. -/
def pipeScan : Nat → List CT → Bool
  | _, [] => false
  | d, t :: r =>
    match t with
    | .pipe => if d = 0 then true else pipeScan d r
    | .comma => if d = 0 then false else pipeScan d r
    | .lparen | .lbrack | .lbrace => pipeScan (d + 1) r
    | .rparen | .rbrack | .rbrace => if d = 0 then false else pipeScan (d - 1) r
    | .eof => false
    | _ => pipeScan d r

/-- The gate of #3104 (cypher.rs) on the state after `[`: the
current token is `(` or an identifier/keyword, and `list_has_top_level_pipe`. -/
def gateOK (s : PS) : Bool :=
  (s.cur == .lparen || (identOf s.cur).isSome) && pipeScan 0 s.1

/-- Alternative 4, the list literal (cypher.rs). -/
def listLitLit (saved : PS) : P (ET × Bool) := do
  setS saved
  let rb ← opt .rbrack
  pure (leaf .list, !rb)

/-- `parse_list_literal_or_comprehension` after `[` (cypher.rs), as
merged by #3104 (`4a4f2a72f`): alternatives 2 and 3 are only tried when the
gate is open and the memo (`o.seen`) does not hold the attempt; `forb` is the
forbidden-pattern-comprehension flag. (`listLitRest` also restores `saved`
and falls through to alternative 4 when both attempts fail, which is where
the Rust inserts the key into the memo.) -/
def listLitP (o : Oracle) (F : Nat) (forb : Bool) : P (ET × Bool) := do
  let saved ← getS
  -- 1) `[var IN ...`
  let a ← tryIdent
  let isIn ← (match a with | some _ => optK .in_ | none => pure false)
  match a, isIn with
  | some v, true => do let t ← listCompP (o.pe false) v; pure (t, false)
  | _, _ =>
    if gateOK saved && !o.seen saved then listLitRest o F forb saved
    else listLitLit saved

/-- The same before #3104 (at `e8f8a3017`): alternatives 2 and 3 always
tried. `C.ListGate` compares the two. -/
def listLitPU (o : Oracle) (F : Nat) (forb : Bool) : P (ET × Bool) := do
  let saved ← getS
  let a ← tryIdent
  let isIn ← (match a with | some _ => optK .in_ | none => pure false)
  match a, isIn with
  | some v, true => do let t ← listCompP (o.pe false) v; pure (t, false)
  | _, _ => listLitRest o F forb saved

/-! ## shortestPath (cypher.rs) -/

/-- shortestPath's types: the same `parse_relationship_types` (cypher.rs) as a pattern's. -/
def spTypes : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let t ← ident
    let p ← opt .pipe
    if p then do
      let _ ← opt .colon
      spTypes n (acc ++ [t])
    else pure (acc ++ [t])

/-- `Token::Parameter(_) | Token::LBracket` after the range (cypher.rs). -/
def isFilter : CT → Bool
  | .param _ => true
  | .lbrace => true
  | _ => false

/-- The edge filter since #3019 (cypher.rs), entered on `isFilter`: `$p` is one
token, `{...}` is parsed by `parse_map` (an error there, e.g. at end of input, is an error). -/
def spFilterP (pmap : P ET) (c : CT) : P Unit :=
  match c with
  | .lbrace => do let _ ← pmap; pure ()
  | _ => next

/-- shortestPath's `*` range (cypher.rs). -/
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
def shortestP (pmap : P ET) (F : Nat) : P ET := do
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
        if filt then spFilterP pmap c2 else pure ()
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

/-- **#3019: an unterminated `$p` filter is a syntax error**: `shortestPath((a)-[$p` at
end of input fails (it used to hang). -/
theorem shortestP_param_eof_err (pmap : P ET) (F a p k : Nat) :
    shortestP pmap F ([.lparen, .ident a, .rparen, .dash, .lbrack, .param p], k) = .err := rfl

/-- **#3019: an unterminated `{...}` filter is a syntax error** whenever `parse_map`
rejects the rest of the input (as at end of input): `shortestPath((a)-[{` fails. -/
theorem shortestP_map_eof_err (pmap : P ET) (F a k : Nat) (h : pmap ([.lbrace], k) = .err) :
    shortestP pmap F ([.lparen, .ident a, .rparen, .dash, .lbrack, .lbrace], k) = .err := by
  simp [shortestP, spFilterP, run_bind, peek, PS.cur, PS.adv, tok, opt, tryIdent, spRangeP,
    isFilter, identOf, next, h]

end FalkorParserGrammar.C
