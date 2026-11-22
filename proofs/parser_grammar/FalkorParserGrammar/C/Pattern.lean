/-
# Patterns: `parse_labels`, `parse_node_pattern`, `parse_relationship_pattern`,
# `add_pattern_node`, `parse_pattern` (cypher.rs:1359-1581, 3016-3214)
-/
import FalkorParserGrammar.C.Graph

namespace FalkorParserGrammar.C

def fuelP {α} : P α := fun _ => .fuel
@[simp] theorem run_fuelP {α} (s : PS) : (fuelP : P α) s = .fuel := rfl

/-- `Parser::next_unnamed` (cypher.rs:3064-3069, #3303): `Alias::Anon(anon_counter)`, then
`anon_counter += 1`. The `u32` increment cannot overflow on any query Redis accepts (4·10⁹
anonymous entities). -/
def anonFresh : P Alias := fun s => .ok (.anon s.2) (s.1, s.2 + 1)

/-- `i as u32` for an `i64` hop bound: truncation mod 2³². -/
def asU32 (i : Int) : Nat := (i % 4294967296).toNat

/-- `OrderSet::insert` / `HashSet::insert`: append if absent. -/
def sins (l : Nat) (acc : List Nat) : List Nat := if l ∈ acc then acc else acc ++ [l]

/-- `parse_labels` (cypher.rs:3207-3214). -/
def labelsP : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let c ← peek
    if c = .colon then do
      next
      let l ← ident
      labelsP n (sins l acc)
    else pure acc

/-- The property part of a node / relationship pattern (cypher.rs:3042-3049,
3143-3150): `$param`, an inline map (`parse_inline_properties`, i.e.
`parse_map` with pattern comprehensions forbidden, which only adds
errors), or the empty map. -/
def propsP (o : Oracle) : P ET := do
  let c ← peek
  match c with
  | .param p => do next; pure (leaf (.param p))
  | .lbrace => o.pmap
  | _ => pure (leaf .map)

/-- `Parser::parse_alias_or_unnamed` (cypher.rs:3071-3075, #3303):
`try_parse_ident().map_or_else(|| next_unnamed(), Alias::Named)`. -/
def aliasOr : Option Nat → P Alias
  | some v => pure (.named v)
  | none => anonFresh

/-- `parse_node_pattern` (cypher.rs:3032-3055). -/
def nodeP (o : Oracle) (F : Nat) : P (QNode Alias) := do
  tok .lparen
  let a ← tryIdent
  let alias ← aliasOr a
  let ls ← labelsP F []
  let attrs ← propsP o
  tok .rparen
  pure ⟨alias, ls, attrs⟩

/-- The relationship-type list (`parse_relationship_types`, cypher.rs:3057-3064, #3300): a name, then
while there is a `|` an optional `:` (the legacy `A|:B`) and another name. A bare `:` between
two names (`A:B`) is not a separator: it is left for the caller (`detailP`), which accepts
only `*`, a property map or `$param`, or `]` there, so `[:A:B]` is a syntax error. -/
def typesP : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let t ← ident
    let p ← opt .pipe
    if p then do
      let _ ← opt .colon
      typesP n (sins t acc)
    else pure (sins t acc)

/-- An optional integer hop bound (cypher.rs:3094-3099, 3101-3106). -/
def hopP : P (Option Int) := do
  let c ← peek
  match c with
  | .int i => do next; pure (some i)
  | _ => pure none

/-- The `*`-range (cypher.rs:3093-3118). -/
def varLenP : P (Option (Nat × Option Nat)) := do
  let st ← opt .star
  if st then do
    let start ← hopP
    let dd ← opt .dotdot
    if dd then do
      let e ← hopP
      pure (some (asU32 (start.getD 1), e.map asU32))
    else match start with
      | some x => pure (some (asU32 x, some (asU32 x)))
      | none => pure (some (1, none))
  else pure none

/-- `[*1..1]` / `[*1]` collapse to a single hop (cypher.rs:3126-3132). -/
def normVL (vl : Option (Nat × Option Nat)) : Option (Nat × Option Nat) :=
  if vl = some (1, some 1) then none else vl

def badRange : Option (Nat × Option Nat) → Bool
  | some (mn, some mx) => decide (mn > mx)
  | _ => false

/-- The bracketed detail `[ ... ]` after `-[` (cypher.rs:3081-3152). -/
def detailP (o : Oracle) (F : Nat) (clause : CK) : P (Alias × List Nat × ET × Option (Nat × Option Nat)) := do
  let a ← tryIdent
  let alias ← aliasOr a
  let col ← opt .colon
  let types ← (if col then typesP F [] else pure [])
  let vl0 ← varLenP
  if badRange vl0 then fail else do
  if (normVL vl0).isSome && (clause == .create || clause == .merge) then fail else do
  let attrs ← propsP o
  tok .rbrack
  pure (alias, types, attrs, normVL vl0)

/-- `parse_relationship_pattern` (cypher.rs:3095-3228). Rust collects the
types through a `HashSet`, so their order is unspecified; the model keeps
first-occurrence order (only membership is observable downstream). #3460: the relationship
records `written_incoming = is_incoming && !is_outgoing` (cypher.rs:3225-3226), i.e. only the
swapped `<-` branch sets `winc`; Rust also returns that flag, which is `rr.1.winc` here. -/
def relP (o : Oracle) (F : Nat) (clause : CK) (src : QNode Alias) : P (QRel Alias × QNode Alias) := do
  let inc ← opt .lt
  tok .dash
  let det ← opt .lbrack
  let d ← (if det then detailP o F clause else do
    let a ← anonFresh
    pure (a, [], leaf .map, none))
  tok .dash
  let out ← opt .gt
  let dst ← nodeP o F
  let mn := d.2.2.2.map (·.1)
  let mx := d.2.2.2.bind (·.2)
  if inc == out then
    if clause = .create then fail
    else pure (QRel.new d.1 d.2.1 d.2.2.1 src dst true mn mx, dst)
  else if inc then pure ({ QRel.new d.1 d.2.1 d.2.2.1 dst src false mn mx with winc := true }, dst)
  else pure (QRel.new d.1 d.2.1 d.2.2.1 src dst false mn mx, dst)

/-- `add_pattern_node` (cypher.rs:1359-1370). -/
def addPatNode (clause : CK) (gs : QG Alias × List Alias) (n : QNode Alias) : QG Alias × List Alias :=
  if n.alias ∈ gs.2 then (if clause = .match_ then gs.1.mergeNode n else gs.1, gs.2)
  else ((gs.1.addNode n).2, gs.2 ++ [n.alias])

/-- The anonymous-path chain (cypher.rs:1559-1561). Every relationship goes through
`add_pattern_relationship` (cypher.rs:1378), which rejects an alias the pattern already holds
whatever the clause (#3289); before it only MATCH and CREATE did. -/
def chainA (o : Oracle) (F : Nat) (clause : CK) : Nat → QNode Alias → QG Alias × List Alias →
    P (QG Alias × List Alias)
  | 0, _, _ => fuelP
  | n + 1, left, gs => do
    let c ← peek
    if c = .dash ∨ c = .lt then do
      let rr ← relP o F clause left
      let ag := gs.1.addRel rr.1
      if !ag.1 then fail
      else chainA o F clause n rr.2 (addPatNode clause (ag.2, gs.2) rr.2)
    else pure gs

/-- The named-path chain (cypher.rs:1551-1571). A repeated relationship variable is an error in
every clause (#3289; it was MATCH only). #3460: `inc` collects each relationship's `<-` flag
(`incoming.push(is_incoming)`), stored in the path. -/
def chainN (o : Oracle) (F : Nat) (clause : CK) (p : Nat) : Nat → QNode Alias → List Alias →
    List Bool → QG Alias × List Alias → P (QG Alias × List Alias)
  | 0, _, _, _, _ => fuelP
  | n + 1, left, vars, inc, gs => do
    let c ← peek
    if c = .dash ∨ c = .lt then do
      let rr ← relP o F clause left
      let ag := gs.1.addRel rr.1
      if !ag.1 then fail
      else
        let gs' := addPatNode clause (ag.2, gs.2) rr.2
        chainN o F clause p n rr.2 (vars ++ [rr.1.alias, rr.2.alias]) (inc ++ [rr.1.winc]) gs'
    else pure (((gs.1.addPath ⟨.named p, vars, inc⟩).2), gs.2)

/-- Identifier ids of `shortestPath` / `allShortestPaths` (compared
case-insensitively at cypher.rs:1384, 1389; case folding is lexical). -/
def nmShortest : Nat := 4000001
def nmAllShortest : Nat := 4000002

/-- The rebuilt allShortestPaths relationship (cypher.rs:1480-1497). `inc` is whether the
relationship was written `<-` (`is_incoming`, #3112; since #3460 the flag
`parse_relationship_pattern` returns, `rr.1.winc`): once from/to are swapped a same-alias
`(a)<-[*]-(a)` has `src.alias = la`, so the arrow is what marks it reversed. The rebuilt
relationship comes from `QueryRelationship::new`, so its own `written_incoming` is `false`. -/
def aspRel (r : QRel Alias) (la : Alias) (inc : Bool) : QRel Alias :=
  { QRel.new r.alias r.types r.attrs r.src r.dst r.bidir (some 1) r.maxH with
    asp := if !r.bidir && (r.src.alias != la || inc) then .rev else .fwd }

/-- **Fixed by #3112**: a same-alias `(a)<-[*]-(a)` (directed, `src.alias = la` after the
swap) is marked `Reversed` because of its `<`; the outgoing `(a)-[*]->(a)` stays `Forward`. -/
theorem aspRel_same_alias (r : QRel Alias) (la : Alias) (hd : r.bidir = false)
    (hs : r.src.alias = la) :
    (aspRel r la true).asp = .rev ∧ (aspRel r la false).asp = .fwd := by
  simp [aspRel, hd, hs]

/-- #3460 reads `written_incoming` (`is_incoming && !is_outgoing`) where the code before read
`<` alone; they differ only for `<-[*]->`, which is bidirectional, where `inc` is not read. -/
theorem aspRel_bidir_inc (r : QRel Alias) (la : Alias) (hb : r.bidir = true) (i j : Bool) :
    aspRel r la i = aspRel r la j := by
  simp [aspRel, hb]

/-- `if let Some(min) = relationship.min_hops && min > 1` (cypher.rs:1433-1440). -/
def minTooBig (r : QRel Alias) : Bool := match r.minH with | some m => decide (m > 1) | none => false

/-- The allShortestPaths relationship loop (cypher.rs:1446-1520); `inc` collects the `<-` flags
(#3460). -/
def aspLoop (o : Oracle) (F : Nat) (clause : CK) (leftAlias : Alias) : Nat → QNode Alias → Bool →
    List Alias → List Bool → QG Alias × List Alias →
      P (Bool × List Alias × List Bool × (QG Alias × List Alias))
  | 0, _, _, _, _, _ => fuelP
  | n + 1, prev, found, vars, inc, gs => do
    let c ← peek
    if c = .dash ∨ c = .lt then do
      let rr ← relP o F clause prev
      let r := rr.1
      if r.minH.isSome || r.maxH.isSome then
        if found then fail
        else if minTooBig r then fail
        else
          let r' := aspRel r leftAlias r.winc
          let ag := gs.1.addRel r'
          if !ag.1 then fail
          else
            let gs' := addPatNode clause (ag.2, gs.2) rr.2
            aspLoop o F clause leftAlias n rr.2 true (vars ++ [r'.alias, rr.2.alias]) (inc ++ [r.winc]) gs'
      else
        let ag := gs.1.addRel r
        if !ag.1 then fail
        else
          let gs' := addPatNode clause (ag.2, gs.2) rr.2
          aspLoop o F clause leftAlias n rr.2 found (vars ++ [r.alias, rr.2.alias]) (inc ++ [r.winc]) gs'
    else pure (found, vars, inc, gs)

/-- How one pattern part ends: `break` out of the loop, `continue` straight
to the next part (allShortestPaths after its `,`), or fall through to the
separator check. -/
inductive PEnd | stop | cont | sep
  deriving DecidableEq

/-- `p = allShortestPaths(...)` (cypher.rs:1389-1513), after `p =`. -/
def aspPart (o : Oracle) (F : Nat) (clause : CK) (p : Nat) (gs : QG Alias × List Alias) :
    P (PEnd × (QG Alias × List Alias)) := do
  next
  tok .lparen
  let left ← nodeP o F
  let gs := addPatNode clause gs left
  let c ← peek
  if !(c = .dash ∨ c = .lt) then fail else do
  let res ← aspLoop o F clause left.alias F left false [left.alias] [] gs
  if !res.1 then fail else do
  tok .rparen
  let gs := (((res.2.2.2.1.addPath ⟨.named p, res.2.1, res.2.2.1⟩).2), res.2.2.2.2)
  let c ← peek
  if c = .comma then do next; pure (.cont, gs) else pure (.stop, gs)

/-- One pattern part (cypher.rs:1379-1565). -/
def partP (o : Oracle) (F : Nat) (clause : CK) (gs : QG Alias × List Alias) :
    P (PEnd × (QG Alias × List Alias)) := do
  let a ← tryIdent
  match a with
  | some p => do
    tok .eq
    let c ← peek
    if identOf c = some nmShortest then fail
    else if identOf c = some nmAllShortest then aspPart o F clause p gs
    else do
      let left ← nodeP o F
      let gs' ← chainN o F clause p F left [left.alias] [] (addPatNode clause gs left)
      pure (.sep, gs')
  | none => do
    let left ← nodeP o F
    let gs' ← chainA o F clause F left (addPatNode clause gs left)
    pure (.sep, gs')

/-- The part separator (cypher.rs:1573-1588): MERGE takes one part; `,`
continues; anything else ends the pattern. Since #2981 the clause's own
keyword ends it too (`MATCH a MATCH b` is two clauses, no longer folded). -/
def sepP (clause : CK) : P Bool := do
  if clause = .merge then pure false else do
  let c ← peek
  match c with
  | .comma => do next; pure true
  | _ => pure false

/-- `parse_pattern` (cypher.rs:1410-1592). -/
def patternLoop (o : Oracle) (F : Nat) (clause : CK) : Nat → QG Alias × List Alias → P (QG Alias)
  | 0, _ => fuelP
  | n + 1, gs => do
    let r ← partP o F clause gs
    match r.1 with
    | .stop => pure r.2.1
    | .cont => patternLoop o F clause n r.2
    | .sep => do
      let more ← sepP clause
      if more then patternLoop o F clause n r.2 else pure r.2.1

def patternP (o : Oracle) (F : Nat) (clause : CK) : P (QG Alias) :=
  patternLoop o F clause F (QG.empty, [])

end FalkorParserGrammar.C
