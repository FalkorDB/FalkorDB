/-
# `parse_primary_expr` in full (cypher.rs:1926-2117; Cypher.g4:361-374)

`Model.primary` covers the fragment of `Model.lean` (no CASE, quantifiers,
calls with DISTINCT / `*`, pattern predicates ...). This is the whole
dispatch over all tokens, with the sub-parsers of `ExprC*`.
-/
import FalkorParserGrammar.C.ExprC5

namespace FalkorParserGrammar.C

/-- A function registry entry (`get_functions().get(name, Function)` or
`Aggregation`, cypher.rs:1996-2007). -/
structure FnI where
  id : Nat
  lo : Nat
  hi : Nat
  agg : Bool
  count : Bool

def nmReduce : Nat := 4000003
def nmPlaceholder : Nat := 4000004

/-- The `loop` of `parse_expression_list` and its closing `)` (cypher.rs:2732-2747). -/
def argItemsP (pe : PE) : Nat → List ET → P (List ET)
  | 0, _ => fuelP
  | n + 1, acc => do
    let e ← pe
    let c2 ← peek
    if c2 = .comma then do next; argItemsP pe n (acc ++ [e])
    else do tok .rparen; pure (acc ++ [e])

/-- `parse_expression_list(ZeroOrMoreClosedBy(RParen), allow)` (cypher.rs:2723-2750); #3060:
only an empty list ends at once, so `f(1,)` is an error. -/
def argsP (pe : PE) (n : Nat) (acc : List ET) : P (List ET) := do
  let c ← peek
  if c = .rparen then do next; pure acc
  else argItemsP pe n acc

/-- The call part after `name(` (cypher.rs:2009-2072). -/
def callTail (pe : PE) (F : Nat) (fi : FnI) : P ET := do
  let dis ← optK .distinct
  if fi.agg then do
    let st ← opt .star
    if st then do
      if !fi.count then fail else do
      if dis then fail else do
      tok .rparen
      pure (.node (.func fi.id true) [leaf (.int 1), leaf (.var nmPlaceholder)])
    else do
      let args ← argsP pe F []
      if !(fi.lo ≤ args.length ∧ args.length ≤ fi.hi) then fail else do
      if args.any (fun a => (aggName a).isSome) then fail else do
      let args := if dis then [ET.node .distinct args] else args
      pure (.node (.func fi.id true) (args ++ [leaf (.var nmPlaceholder)]))
  else do
    let args ← argsP pe F []
    if !(fi.lo ≤ args.length ∧ args.length ≤ fi.hi) then fail else do
    if dis && args.isEmpty then fail else pure (.node (.func fi.id false) args)

/-- The identifier arm: a call, `reduce(`, `shortestPath(`, or a variable. -/
def identArm (o : Oracle) (fr : List Nat → Option FnI) (F : Nat) (allow : Bool) : P (ET × Bool) := do
  let s0 ← getS
  let v ← ident
  let name ← dottedP F [v]
  let lp ← opt .lparen
  if lp then
    if name = [nmReduce] then do let t ← reduceP (o.pe allow); pure (t, false)
    else if name = [nmShortest] then do let t ← shortestP F; pure (t, false)
    else if name = [nmAllShortest] then fail
    else match fr name with
      | none => fail
      | some fi => do let t ← callTail (o.pe allow) F fi; pure (t, false)
  else do
    setS s0
    let v ← ident
    pure (leaf (.var v), false)

def isPatG : Option (QG Nat) → Bool
  | some g => !g.rels.isEmpty
  | none => false

/-- The `(` arm: a pattern predicate when allowed and the pattern has a
relationship (cypher.rs:2098-2112), otherwise a parenthesised expression. -/
def parenArm (o : Oracle) (F : Nat) (allow : Bool) : P (ET × Bool) := do
  let s0 ← getS
  let pat ← (if allow then attempt (patternP o F .match_) else pure none)
  if isPatG pat then pure (leaf .pattern, false)
  else do
    setS s0
    next
    pure (leaf .paren, true)

/-- The quantifier-keyword arm (cypher.rs:1935-1951). -/
def quantArm (o : Oracle) (allow : Bool) : P (ET × Bool) := do
  let s0 ← getS
  next
  let c2 ← peek
  setS s0
  if c2 = .lparen then do let t ← quantP (o.pe allow); pure (t, false)
  else do let v ← ident; pure (leaf (.var v), false)

def isQuantKw : CT → Bool
  | .kw .all | .kw .any | .kw .none | .kw .single => true
  | _ => false

/-- `parse_primary_expr(allow_pattern_predicate)`; `forb` is the
forbidden-pattern-comprehension flag. -/
def primaryC (o : Oracle) (fr : List Nat → Option FnI) (F : Nat) (allow forb : Bool) : P (ET × Bool) := do
  let c ← peek
  match c with
  | .kw .case_ => do let t ← caseP (o.pe false) F; pure (t, false)
  | .kw .null => do next; pure (leaf .null, false)
  | .kw .true_ => do next; pure (leaf (.bool true), false)
  | .kw .false_ => do next; pure (leaf (.bool false), false)
  | .kw k => if isQuantKw (.kw k) then quantArm o allow else identArm o fr F allow
  | .ident _ => identArm o fr F allow
  | .param p => do next; pure (leaf (.param p), false)
  | .int i => do next; pure (leaf (.int i), false)
  | .float => do next; pure (leaf (.other 0), false)
  | .str t => do next; pure (leaf (.str t), false)
  | .lbrack => do next; listLitP o F forb
  | .lbrace => do let m ← o.pmap; pure (m, false)
  | .lparen => parenArm o F allow
  | _ => fail

/-! ## No panic -/

section
variable (o : Oracle) (ho : o.Safe)
include ho

theorem np_argItemsP (pe : PE) (hpe : NP pe) : ∀ n acc, NP (argItemsP pe n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_argItemsP pe hpe n; unfold argItemsP; np_go
theorem np_argsP (pe : PE) (hpe : NP pe) (n : Nat) (acc : List ET) : NP (argsP pe n acc) := by
  have := np_argItemsP o ho pe hpe n acc; unfold argsP; np_go

theorem np_callTail (pe : PE) (hpe : NP pe) (F : Nat) (fi : FnI) : NP (callTail pe F fi) := by
  have := np_argsP o ho pe hpe; unfold callTail; np_go

theorem np_identArm (fr : List Nat → Option FnI) (F : Nat) (allow : Bool) : NP (identArm o fr F allow) := by
  have := np_dottedP; have := np_reduceP (o.pe allow) (np_pe o ho allow)
  have := np_shortestP; have := np_callTail o ho (o.pe allow) (np_pe o ho allow)
  have hs : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
  unfold identArm; np_go

theorem np_parenArm (F : Nat) (allow : Bool) : NP (parenArm o F allow) := by
  have := np_attempt (np_patternP o ho F .match_)
  have hs : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
  unfold parenArm; np_go

/-- The quantifier arm only calls `parse_quantifier_expr` on a quantifier
keyword, so its `unreachable!()` is never reached. -/
theorem quantArm_np (allow : Bool) (s : PS) (hq : isQuantKw s.cur = true) : quantArm o allow s ≠ .panic := by
  unfold quantArm
  simp only [run_bind, run_getS, next, peek, run_setS]
  split
  · have hq' : (quantCode s.cur).isSome := by
      revert hq; cases s.cur <;> simp [isQuantKw]; rename_i k; cases k <;> simp [quantCode]
    have := quantP_np (o.pe allow) (np_pe o ho allow) s hq'
    simp only [run_bind]
    revert this; cases quantP (o.pe allow) s <;> simp
  · simp only [run_bind]
    have := (np_ident).run s
    revert this; cases ident s <;> simp

/-- **parse_primary_expr never panics** (given expression and map parsers
that do not), including its `unreachable!()` through the quantifier arm. -/
theorem primaryC_np (fr : List Nat → Option FnI) (F : Nat) (allow forb : Bool) :
    NP (primaryC o fr F allow forb) := by
  have hc := np_caseP (o.pe false) (np_pe o ho false) F
  have hi := np_identArm o ho fr F allow
  have hp := np_parenArm o ho F allow
  have hl := np_listLitP o ho F forb
  have hm := np_pmap o ho
  refine ⟨fun s => ?_⟩
  unfold primaryC; simp only [run_bind, peek]
  generalize hcur : s.cur = c
  cases c
  case kw k =>
    cases k
    all_goals first
      | exact (np_bind hc (fun _ => np_pure _)).run s
      | exact (np_bind np_next (fun _ => np_pure _)).run s
      | (dsimp only; split
         · exact quantArm_np o ho allow s (by rw [hcur]; assumption)
         · exact hi.run s)
  all_goals first
    | exact hi.run s
    | exact (np_bind np_next (fun _ => np_pure _)).run s
    | exact (np_bind np_next (fun _ => hl)).run s
    | exact (np_bind hm (fun _ => np_pure _)).run s
    | exact hp.run s
    | simp

end

end FalkorParserGrammar.C
