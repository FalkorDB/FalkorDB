/-
# The clause parsers of `graph/src/parser/cypher.rs` (lines 516-1340, 2680-2738, 3190-3400)

One model function per Rust function. All recursion (loops and the
`CALL {}` / `FOREACH` re-entries) runs on one budget `n`: every call made by
a function entered with `n + 1` gets `n`.
-/
import FalkorParserGrammar.C.Pattern
import FalkorParserGrammar.C.IR

namespace FalkorParserGrammar.C

/-- `optional_match_token!(lexer => K)`. -/
abbrev optK (k : CK) : P Bool := opt (.kw k)
/-- `match_token!(lexer => K)`. -/
abbrev tokK (k : CK) : P Unit := tok (.kw k)

/-- `parse_where` (cypher.rs:1198-1215). -/
def whereP (o : Oracle) : P (Option ET) := do
  let c ← peek
  if c = .kw .where_ then do
    next
    let e ← o.pe true
    if (aggName e).isSome then fail else pure (some e)
  else pure none

/-- `parse_orderby` after `ORDER` (cypher.rs:3201-3217). -/
def orderItems (o : Oracle) : Nat → List (ET × Bool) → P (List (ET × Bool))
  | 0, _ => fuelP
  | n + 1, acc => do
    let e ← o.pe false
    let a1 ← optK .asc
    let asc ← (if a1 then pure true else optK .ascending)
    let desc ← (if asc then pure false else do
      let d1 ← optK .desc
      if d1 then pure true else optK .descending)
    let more ← opt .comma
    if more then orderItems o n (acc ++ [(e, desc)]) else pure (acc ++ [(e, desc)])

def orderbyP (o : Oracle) (n : Nat) : P (List (ET × Bool)) := do
  tokK .by
  orderItems o n []

/-- The SKIP / LIMIT value check (cypher.rs:1233-1248, 1254-1269): a
non-negative integer literal or a parameter. -/
def countOk (e : ET) : Bool :=
  match e.root with
  | .int i => decide (0 ≤ i)
  | .param _ => true
  | _ => false

def countP (o : Oracle) (k : CK) : P (Option ET) := do
  let b ← optK k
  if b then do
    let e ← o.pe false
    if countOk e then pure (some e) else fail
  else pure none

/-- `parse_orderby_skip_limit` (cypher.rs:1217-1275). -/
def oslP (o : Oracle) (n : Nat) : P (List (ET × Bool) × Option ET × Option ET) := do
  let b ← optK .order
  let ob ← (if b then orderbyP o n else pure [])
  let sk ← countP o .skip
  let li ← countP o .limit
  pure (ob, sk, li)

/-- `parse_named_exprs` (cypher.rs:2687-2721). An unaliased non-variable
projection is named by its source text, `o.txt` of the span. -/
def namedExprs (o : Oracle) (mustAlias : Bool) : Nat → List (Nat × ET) → P (List (Nat × ET))
  | 0, _ => fuelP
  | n + 1, acc => do
    let s0 ← getS
    let e ← o.pe true
    let c ← peek
    let item ← (if c = .kw .as_ then do
        next
        let v ← ident
        pure (v, e)
      else match e.root with
        | .var v => pure (v, e)
        | _ => if mustAlias then fail else do
          let s1 ← getS
          pure (o.txt s0 s1, e))
    let c2 ← peek
    if c2 = .comma then do
      next
      namedExprs o mustAlias n (acc ++ [item])
    else pure (acc ++ [item])

/-- The projection body shared by WITH and RETURN (cypher.rs:1281-1291,
1311-1320). -/
def projP (o : Oracle) (mustAlias : Bool) (n : Nat) : P Proj := do
  let d ← optK .distinct
  let st ← opt .star
  let ae ← (if st then do
      let cm ← opt .comma
      if cm then do let es ← namedExprs o mustAlias n []; pure (true, es) else pure (true, [])
    else do let es ← namedExprs o mustAlias n []; pure (false, es))
  let osl ← oslP o n
  pure ⟨d, ae.1, ae.2, osl.1, osl.2.1, osl.2.2⟩

/-- `parse_with_clause` (cypher.rs:1277-1305). -/
def withP (o : Oracle) (write : Bool) (n : Nat) : P IR := do
  let p ← projP o true n
  let f ← whereP o
  pure (.with_ p f write)

/-- `parse_return_clause` (cypher.rs:1307-1341). -/
def returnP (o : Oracle) (write : Bool) (n : Nat) : P IR := do
  let p ← projP o false n
  pure (.return_ p write)

/-- `parse_unwind_clause` (cypher.rs:1121-1129). -/
def unwindP (o : Oracle) : P IR := do
  let e ← o.pe false
  tokK .as_
  let v ← ident
  pure (.unwind e v)

/-- `parse_match_clause` (cypher.rs:1110-1119). -/
def matchP (o : Oracle) (F : Nat) (optional : Bool) : P IR := do
  let p ← patternP o F .match_
  let f ← whereP o
  pure (.match_ p f optional)

/-- `parse_create_clause` (cypher.rs:1131-1133). -/
def createP (o : Oracle) (F : Nat) : P IR := do
  let p ← patternP o F .create
  pure (.create p)

/-- `parse_expression_list(OneOrMore)` (cypher.rs:2723-2749): at least one
expression, comma separated, no closing token. -/
def exprList1 (o : Oracle) : Nat → List ET → P (List ET)
  | 0, _ => fuelP
  | n + 1, acc => do
    let e ← o.pe false
    let c ← peek
    if c = .comma then do next; exprList1 o n (acc ++ [e]) else pure (acc ++ [e])

/-- The `loop` of `parse_expression_list` and its closing `)` (cypher.rs:2732-2747). -/
def exprItemsR (o : Oracle) : Nat → List ET → P (List ET)
  | 0, _ => fuelP
  | n + 1, acc => do
    let e ← o.pe false
    let c2 ← peek
    if c2 = .comma then do next; exprItemsR o n (acc ++ [e])
    else do tok .rparen; pure (acc ++ [e])

/-- `parse_expression_list(ZeroOrMoreClosedBy(RParen))` (cypher.rs:2723-2750). #3060: only an
empty list ends at once; after `,` an expression must follow (`f(1,)` is an error). -/
def exprListR (o : Oracle) (n : Nat) (acc : List ET) : P (List ET) := do
  let c ← peek
  if c = .rparen then do next; pure acc
  else exprItemsR o n acc

/-- `parse_delete_clause` (cypher.rs:1162-1196): consecutive DELETEs are
folded into one clause, `detach` if any of them is. -/
def deleteMore (o : Oracle) : Nat → List ET → Bool → P IR
  | 0, _, _ => fuelP
  | n + 1, es, det => do
    let c ← peek
    if c = .kw .delete ∨ c = .kw .detach then do
      let d ← optK .detach
      tokK .delete
      let more ← exprList1 o n []
      deleteMore o n (es ++ more) (det || d)
    else pure (.delete es det)

def deleteP (o : Oracle) (n : Nat) (isDetach : Bool) : P IR := do
  let es ← exprList1 o n []
  deleteMore o n es isDetach

/-- The `.prop.prop...` chain of a SET / REMOVE target (cypher.rs:3242-3246). -/
def dotChain : Nat → ET → P ET
  | 0, _ => fuelP
  | n + 1, e => do
    let c ← peek
    if c = .dot then do
      next
      let p ← ident
      dotChain n (.node (.prop p) [e])
    else pure e

/-- `reject_unparenthesized_target` (cypher.rs:3289-3298, #3060 / `fe619ac5f`): a target
that opened a nested expression must have opened it with `(` — the tree is a `Paren`. -/
def rejectUnparenP (t : ET) : P Unit := if t.root = .paren then pure () else fail

/-- The target of a SET / REMOVE item (cypher.rs:3236-3241, 3317-3322). -/
def targetP (o : Oracle) : P ET := do
  let r ← o.pp
  if r.2 then do
    rejectUnparenP r.1
    let e ← o.pe false
    tok .rparen
    pure e
  else pure r.1

/-- `parse_set_items` (cypher.rs:3231-3284). -/
def setItems (o : Oracle) (F : Nat) : Nat → List SetItem → P (List SetItem)
  | 0, _ => fuelP
  | n + 1, acc => do
    let t ← targetP o
    let c ← peek
    let item ← (if c = .dot then do
        let t' ← dotChain F t
        tok .eq
        let v ← o.pe false
        pure (SetItem.attr t' v false)
      else if c = .colon then
        match t.root with
        | .var v => do let ls ← labelsP F []; pure (SetItem.label v ls)
        | _ => fail
      else do
        let eq ← opt .eq
        if eq then do
          let v ← o.pe false
          pure (SetItem.attr t v true)
        else do
          tok .plusEq
          let v ← o.pe false
          pure (SetItem.attr t v false))
    let more ← opt .comma
    if more then setItems o F n (acc ++ [item]) else pure (acc ++ [item])

/-- `parse_set_clause` (cypher.rs:3219-3229): consecutive SETs fold. -/
def setMore (o : Oracle) (F : Nat) : Nat → List SetItem → P IR
  | 0, _ => fuelP
  | n + 1, acc => do
    let b ← optK .set
    if b then do let is ← setItems o F F acc; setMore o F n is else pure (.set acc)

def setP (o : Oracle) (F : Nat) : P IR := do
  let is ← setItems o F F []
  setMore o F F is

/-- The `hasLabels` function id (cypher.rs:3332). -/
def fnHasLabels : Nat := 5000001

/-- `parse_remove_items` (cypher.rs:3312-3346). -/
def removeItems (o : Oracle) (F : Nat) : Nat → List ET → P (List ET)
  | 0, _ => fuelP
  | n + 1, acc => do
    let t ← targetP o
    let c ← peek
    let item ← (if c = .dot then dotChain F t
      else if c = .colon then do
        let ls ← labelsP F []
        pure (ET.node (.func fnHasLabels false) [t, .node .list (ls.map (fun l => leaf (.str l)))])
      else fail)
    let more ← opt .comma
    if more then removeItems o F n (acc ++ [item]) else pure (acc ++ [item])

/-- `parse_remove_clause` (cypher.rs:3300-3310). -/
def removeMore (o : Oracle) (F : Nat) : Nat → List ET → P IR
  | 0, _ => fuelP
  | n + 1, acc => do
    let b ← optK .remove
    if b then do let is ← removeItems o F F acc; removeMore o F n is else pure (.remove acc)

def removeP (o : Oracle) (F : Nat) : P IR := do
  let is ← removeItems o F F []
  removeMore o F F is

/-- `parse_merge_clause` (cypher.rs:1135-1160). -/
def mergeActions (o : Oracle) (F : Nat) : Nat → List SetItem → List SetItem → P (List SetItem × List SetItem)
  | 0, _, _ => fuelP
  | n + 1, oc, om => do
    let on ← optK .on
    if on then do
      let m ← optK .match_
      if m then do
        tokK .set
        let om' ← setItems o F F om
        mergeActions o F n oc om'
      else do
        let c ← optK .create
        if c then do
          tokK .set
          let oc' ← setItems o F F oc
          mergeActions o F n oc' om
        else fail
    else pure (oc, om)

def mergeP (o : Oracle) (F : Nat) : P IR := do
  let p ← patternP o F .merge
  let a ← mergeActions o F F [] []
  pure (.merge p a.1 a.2)

/-- `ExprIR::Constant(Value::String(","))` (cypher.rs:953-955). -/
def strComma : Nat := 5000002

/-- The LOAD CSV arm of `parse_reading_clasue` (cypher.rs:935-963), after LOAD. -/
def loadCsvP (o : Oracle) : P IR := do
  tokK .csv
  let w ← optK .with_
  -- #3060: `WITH` must be followed by `HEADERS` (cypher.rs:939-942)
  let hd ← (if w then do tokK .headers; pure true else pure false)
  tokK .from
  let path ← o.pe false
  tokK .as_
  let v ← ident
  let ft ← optK .fieldterminator
  let dl ← (if ft then o.pe false else pure (leaf (.str strComma)))
  pure (.loadCsv path hd dl v)

/-- `parse_reading_clasue` (cypher.rs:913-964). -/
def readingP (o : Oracle) (F : Nat) : P IR := do
  let op ← optK .optional
  if op then do tokK .match_; matchP o F true
  else do
    let c ← peek
    match c with
    | .kw .match_ => do next; matchP o F false        -- #3060: no second `MATCH` (920-925)
    | .kw .unwind => do next; unwindP o
    | .kw .load => do next; loadCsvP o
    | _ => oops                                   -- `unreachable!()` (963)

/-- Reading-clause and updating-clause keywords (cypher.rs:784-795, 809-822). -/
def isReadKw : CT → Bool
  | .kw .optional | .kw .match_ | .kw .unwind | .kw .call | .kw .load => true
  | _ => false
def isWriteKw : CT → Bool
  | .kw .create | .kw .merge | .kw .delete | .kw .detach | .kw .set | .kw .remove
  | .kw .foreach => true
  | _ => false

/-- `parse_dotted_ident` (cypher.rs:1099-1108). -/
def dottedP : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let c ← peek
    if c = .dot then do next; let v ← ident; dottedP n (acc ++ [v]) else pure acc

/-- YIELD items (cypher.rs:1054-1075): `name [AS alias]`, comma separated. -/
def yieldItems : Nat → List Nat → List (Option Nat) → P (List Nat × List (Option Nat))
  | 0, _, _ => fuelP
  | n + 1, outs, als => do
    let v ← ident
    let a ← optK .as_
    let p ← (if a then do let al ← ident; pure (al, some v) else pure (v, none))
    let more ← opt .comma
    if more then yieldItems n (outs ++ [p.1]) (als ++ [p.2]) else pure (outs ++ [p.1], als ++ [p.2])

/-- `CALL procedure` (cypher.rs:1032-1086), after CALL, not `{`. -/
def callProcP (o : Oracle) (F : Nat) : P IR := do
  let v ← ident
  let name ← dottedP F [v]
  match o.proc name with
  | none => fail
  | some fi => do
    let lp ← opt .lparen
    let args ← (if lp then exprListR o F [] else pure [])
    if !(fi.lo ≤ args.length ∧ args.length ≤ fi.hi) then fail else do
    let y ← optK .yield_
    if y then do
      let ya ← yieldItems F [] []
      let f ← whereP o
      pure (.call name args ya.1 ya.2 f true)
    else if fi.isProc then pure (.call name args fi.outputs (fi.outputs.map (fun _ => none)) none false)
    else pure (.call name args [] [] none false)

-- `body_has_return` is `bodyHasReturn` (IR.lean).

mutual
/-- `parse_query` (cypher.rs:755-773). -/
def queryP (o : Oracle) (F : Nat) : Nat → P IR
  | 0 => fuelP
  | n + 1 => do
    let first ← singleP o F n
    let u ← optK .union
    if u then do
      let all ← optK .all
      let second ← singleP o F n
      unionMore o F n all [first, second]
    else pure first
/-- The `while UNION` loop (cypher.rs:762-770). -/
def unionMore (o : Oracle) (F : Nat) : Nat → Bool → List IR → P IR
  | 0, _, _ => fuelP
  | n + 1, all, bs => do
    let u ← optK .union
    if u then do
      let a ← optK .all
      if a != all then fail else do
      let b ← singleP o F n
      unionMore o F n all (bs ++ [b])
    else pure (.union bs all)
/-- `parse_single_query` (cypher.rs:780-911). -/
def singleP (o : Oracle) (F : Nat) : Nat → P IR
  | 0 => fuelP
  | n + 1 => do
    let r ← segments o F n [] false
    let cl := r.1
    let write := r.2
    let ret ← optK .return_
    let res ← (if ret then do
        let rc ← returnP o write n
        let c ← peek
        if c = .eof ∨ c = .kw .union ∨ c = .semi ∨ c = .rbrace then pure (cl ++ [rc], false)
        else fail
      else pure (cl, write))
    let c ← peek
    if c = .kw .union ∨ c = .semi ∨ c = .rbrace then pure (.query res.1 res.2)
    else do tok .eof; pure (.query res.1 res.2)
/-- The `loop { reading* updating* (WITH | break) }` (cypher.rs:783-880). -/
def segments (o : Oracle) (F : Nat) : Nat → List IR → Bool → P (List IR × Bool)
  | 0, _, _ => fuelP
  | n + 1, cl, _ => do
    let cl1 ← readings o F n cl
    let w ← writings o F n cl1 false
    let wi ← optK .with_
    if wi then do
      let wc ← withP o w.2 n
      segments o F n (w.1 ++ [wc]) false
    else
      if w.2 then do
        let c ← peek
        if isReadKw c then fail else pure w
      else pure w
/-- `while let MATCH|OPTIONAL|UNWIND|CALL|LOAD` (cypher.rs:784-807). -/
def readings (o : Oracle) (F : Nat) : Nat → List IR → P (List IR)
  | 0, _ => fuelP
  | n + 1, cl => do
    let c ← peek
    if isReadKw c then
      if c = .kw .call then do
        next
        let cs ← callP o F n
        readings o F n (cl ++ cs)
      else do
        let r ← readingP o F
        readings o F n (cl ++ [r])
    else pure cl
/-- `while let CREATE|MERGE|DELETE|DETACH|SET|REMOVE|FOREACH` (cypher.rs:808-826). -/
def writings (o : Oracle) (F : Nat) : Nat → List IR → Bool → P (List IR × Bool)
  | 0, _, _ => fuelP
  | n + 1, cl, w => do
    let c ← peek
    if isWriteKw c then do
      let r ← writingP o F n
      writings o F n (cl ++ [r]) true
    else pure (cl, w)
/-- `parse_writing_clause` (cypher.rs:966-1013). -/
def writingP (o : Oracle) (F : Nat) : Nat → P IR
  | 0 => fuelP
  | n + 1 => do
    let c ← peek
    match c with
    | .kw .create => do next; createP o F
    | .kw .merge => do next; mergeP o F
    | .kw .detach | .kw .delete => do
      let d ← optK .detach
      tokK .delete
      deleteP o n d
    | .kw .set => do next; setP o F
    | .kw .remove => do next; removeP o F
    | .kw .foreach => do next; foreachP o F n
    | _ => oops                                   -- `unreachable!()` (1012)
/-- `parse_call_clause` (cypher.rs:1015-1087), after CALL. -/
def callP (o : Oracle) (F : Nat) : Nat → P (List IR)
  | 0 => fuelP
  | n + 1 => do
    let c ← peek
    if c = .lbrace then do
      next
      let body ← queryP o F n
      tok .rbrace
      pure [.callSub body (bodyHasReturn body)]
    else do
      let r ← callProcP o F
      pure [r]
/-- `parse_foreach_clause` (cypher.rs:3348-3428), after FOREACH. -/
def foreachP (o : Oracle) (F : Nat) : Nat → P IR
  | 0 => fuelP
  | n + 1 => do
    tok .lparen
    let v ← ident
    tokK .in_
    let l ← o.pe false
    if (aggName l).isSome then fail else do
    tok .pipe
    let body ← foreachBody o F n []
    tok .rparen
    if body = [] then fail else pure (.forEach l v body)
/-- The FOREACH body loop (cypher.rs:3362-3413). -/
def foreachBody (o : Oracle) (F : Nat) : Nat → List IR → P (List IR)
  | 0, _ => fuelP
  | n + 1, body => do
    let c ← peek
    match c with
    | .kw .create => do next; let r ← createP o F; foreachBody o F n (body ++ [r])
    | .kw .merge => do next; let r ← mergeP o F; foreachBody o F n (body ++ [r])
    | .kw .detach | .kw .delete => do
      let d ← optK .detach
      tokK .delete
      let r ← deleteP o n d
      foreachBody o F n (body ++ [r])
    | .kw .set => do next; let r ← setP o F; foreachBody o F n (body ++ [r])
    | .kw .remove => do next; let r ← removeP o F; foreachBody o F n (body ++ [r])
    | .kw .foreach => do next; let r ← foreachP o F n; foreachBody o F n (body ++ [r])
    | _ => pure body
end

/-- `expect_end_of_input` (cypher.rs:516-524). -/
def endP : Nat → P Unit
  | 0 => fuelP
  | n + 1 => do
    let c ← peek
    if c = .semi then do next; endP n
    else if c = .eof then pure () else fail

end FalkorParserGrammar.C
