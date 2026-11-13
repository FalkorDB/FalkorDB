import CchRedesignReview.Log
import FalkorNativeIndex.Rebase
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# The global state and the step relation (03 §5, §6, §6b, §7)

One CCH index, as every published version holds it. The kind's data (topology, metric, ranks) is abstracted
to what the log's rules need:

* `res p`: the data version up to which the structure's view of pair `p` is current: every write of `p`
  with `seq ≤ res p` is reflected in it. A consumer that resolves `p` at its fork `F` sets it to `F`
  (03 §6b: "pair state = f(current graph)"); a writer that flushes `p` sets it to its own version.
  A fresh build starts at its birth version: its skeleton is read at forks at or after it (03 §5 Phases), so it
  reflects at least every write up to the birth. Slab reads and Seed (`V_c = V_sg`) do not raise it: the model
  credits them nothing, which is conservative.
* `lineage`: which hierarchy lineage the entry's base is (S6 and S7 start a new one, 03 §2 "New lineages").

The rest is the design's state: `Building{phase}` or `Built{upkeep}`, the log and its `pairs` counter, the
generation and the batch count (`indexer_commits`).
-/
namespace CchReview

/-- 03 §5 "Phases". -/
inductive Phase where
  | readSkeleton | order | supergraph | seed | relax | seal | repair
deriving DecidableEq, Repr

/-- 03 §2 `upkeep`: `Upkeep { kind: Fold | S6, ticket, since }`. -/
inductive UKind where
  | fold | s6
deriving DecidableEq, Repr

structure Upkeep where
  kind : UKind
  ticket : Nat
  since : Nat
deriving DecidableEq, Repr

/-- 03 §7 "Versions": `Built { cch, upkeep, log }` or `Building { phase state, log }`. -/
inductive St where
  | building (ph : Phase)
  | built (up : Option Upkeep)
deriving DecidableEq, Repr

structure Entry where
  st : St
  res : Pair → Nat
  lineage : Nat
  log : List Node
  pairs : Nat
  gen : Nat
  commits : Nat

namespace Entry
def isBuilt (e : Entry) : Bool := match e.st with | .built _ => true | .building _ => false
def upkeep (e : Entry) : Option Upkeep := match e.st with | .built u => u | .building _ => none

/-- The `(generation, indexer_commits)` projection as native_index's `SE` (`Batch.lean`), with an empty column
    standing for the kind's data: `discarded_against` and `overtaken` read only the projection and whether the
    entry is `Building` (`versioned.rs:466-480, 749-765` @08c). -/
def toSE (e : Entry) : NI.SE :=
  ⟨⟨.numeric NI.NumericIndex.new, if e.isBuilt then .ready else .building (.node 0)⟩, e.gen, e.commits, 0⟩
end Entry

/-- A published version: its data version, `batches_published` (`falkordb_index.rs`, the counter the rebase gate
    reads) and the index's entry. -/
structure Pub where
  ver : Nat
  batches : Nat
  e : Entry

/-- The whole system: the published version, every version published before it (any of them may be a fork
    still held by a writer or a job), every committed write `(seq, pair)`, and the shared counter that
    generations, tickets and lineages come from (03 §6: "`ticket` comes from the same counter as
    generations"). -/
structure Sys where
  cur : Pub
  hist : List Pub
  writes : List (Nat × Pair)
  next : Nat

/-- Readers (03 §5 "Readers while Building", §6b MVCC): a `Built` entry serves its hierarchy, a `Building` one
    the Dijkstra fallback (`none`). -/
def serve (S : Pub) : Option (Pair → Nat) :=
  match S.e.st with
  | .built _ => some S.e.res
  | .building _ => none

/-- `p` resolved at `v`: what a consumer or a flush does to the structure's view of the pairs it resolves. -/
def resolve (r : Pair → Nat) (ps : List Pair) (v : Nat) : Pair → Nat := fun p => if p ∈ ps then v else r p

/-! ## Readings that the findings vary

`D` is the design as read here. The other settings are the alternative readings and the pre-03 rules that the
findings and the "break" theorems run (`Findings.lean`). -/
structure Rules where
  /-- 03 §7: `overtaken` also fires for a `Built` entry whose `indexer_commits` moved (D6). `false`: today's
      rule, `ours.is_building()` required (`versioned.rs:759-761` @08c, `NI.overtakenSE`). -/
  d6 : Bool := true
  /-- 03 §5 rebase table, row 1: the fork's own nodes are relinked onto the published log. `false`: @08c, the
      published copy with only `by_this_version` carried, which is empty for CCH (`versioned.rs:786, 805`). -/
  relink : Bool := true
  /-- 03 §6 "Retention": past the bound the writer "clears upkeep and the log". `true` (reading R2): the
      retained nodes, its own nodes kept (§5: "A writer always keeps its own nodes in its private version");
      `false`: the literal words, own nodes included. -/
  keepOwn : Bool := true
  /-- 03 §5 "Bound": the writer checks the bound whenever it changes a Building copy's log (reading R5). `false`:
      only when it appends a segment. -/
  checkAll : Bool := true

def D : Rules := {}

/-- 03 §5 "Bound and Overflow": past the bound `B` the writer replaces the copy's whole unconsumed list with one
    `Overflow{seq = its version}` and resets `pairs`. -/
def checkB (B v : Nat) (e : Entry) : Entry :=
  if B < e.pairs then { e with log := [⟨v, .overflow⟩], pairs := 0 } else e

def checkIf (c : Bool) (B v : Nat) (e : Entry) : Entry := if c then checkB B v e else e

/-! ## Writers (03 §5 "The log", §6 flush, hand-off, decay, §5 rebase table, §7 retention) -/

/-- What one `Commit` segment's flush does on a `Built` entry (budget and decay outcomes are free choices). -/
inductive Out where
  /-- Within N_max and B_gil. -/
  | fits
  /-- Within budget, and the flush crosses a fold or S6 trigger: `upkeep = Some{kind, ticket, since = v}`. -/
  | setUp (k : UKind)
  /-- Over N_max or B_gil: the hand-off (03 §6). -/
  | abort
  /-- An S7 trigger: a new generation through the batched build, this segment as its log (03 §6 Decay). -/
  | s7
deriving DecidableEq, Repr

/-- 03 §6 flush of one segment on a `Built` copy at the writer's version `v`: resolve the pairs at `v`, keep the
    own node in the private version (§5), and, while `upkeep` is set, past the bound clear `upkeep` and the
    retained log instead of writing an Overflow (§6 Retention). -/
def flushB (R : Rules) (B v : Nat) (ps : List Pair) (up : Option Upkeep) (e : Entry) : Entry :=
  let l : List Node := ⟨v, .pairs ps⟩ :: e.log
  let e1 : Entry := { e with res := resolve e.res ps v, log := l, pairs := e.pairs + ps.length }
  match up with
  | none => { e1 with st := .built none }
  | some u =>
    if B < e1.pairs then
      let k := if R.keepOwn then own v l else []
      { e1 with st := .built none, log := k, pairs := sumPairs k }
    else { e1 with st := .built (some u) }

/-- A Building copy's log with the segment's node on top (03 §5: a Building copy only logs). -/
def appendE (v : Nat) (ps : List Pair) (e : Entry) : Entry :=
  { e with log := ⟨v, .pairs ps⟩ :: e.log, pairs := e.pairs + ps.length }

/-- 03 §6 Hand-off: `Building{Repair, base, log: [this segment]}` under `next_generation()`; the base is the
    private hierarchy at the abort. -/
def handoffE (v : Nat) (ps : List Pair) (e : Entry) (nx : Nat) : Entry :=
  { st := .building .repair, res := e.res, lineage := e.lineage, log := [⟨v, .pairs ps⟩], pairs := ps.length,
    gen := nx, commits := 0 }

/-- 03 §6 Decay, S7: a new generation (and lineage) through the batched build, this segment as its log. Its
    skeleton is read at or after `v`. -/
def s7E (v : Nat) (ps : List Pair) (nx : Nat) : Entry :=
  { st := .building .readSkeleton, res := fun _ => v, lineage := nx, log := [⟨v, .pairs ps⟩], pairs := ps.length,
    gen := nx, commits := 0 }

/-- One segment at version `v` on the private copy `e`; `nx` is the next fresh id. -/
def seg (R : Rules) (B v : Nat) (ps : List Pair) (o : Out) (e : Entry) (nx : Nat) : Entry × Nat :=
  match e.st with
  | .building _ => (checkB B v (appendE v ps e), nx)
  | .built up =>
    match o with
    | .fits => (flushB R B v ps up e, nx)
    | .setUp k => (flushB R B v ps (some ⟨k, nx, v⟩) e, nx + 1)
    | .abort => (checkIf R.checkAll B v (handoffE v ps e nx), nx + 1)
    | .s7 => (checkIf R.checkAll B v (s7E v ps nx), nx + 1)

def segs (R : Rules) (B v : Nat) : List (List Pair × Out) → Entry → Nat → Entry × Nat
  | [], e, nx => (e, nx)
  | (ps, o) :: rest, e, nx => segs R B v rest (seg R B v ps o e nx).1 (seg R B v ps o e nx).2

def segPairs : List (List Pair × Out) → List Pair
  | [] => []
  | (ps, _) :: rest => ps ++ segPairs rest

/-- `Graph::overtaken` (`versioned.rs:749-765` @08c) with 03 §7's D6 change. With `d6 = false` it is exactly
    native_index's model of today's rule (`NI.overtakenSE`). -/
def overtaken (R : Rules) (ours theirs : Entry) : Bool :=
  if R.d6 then ours.gen == theirs.gen && ours.commits != theirs.commits
  else NI.overtakenSE ours.toSE theirs.toSE

/-- Run the §6 flush over the own nodes onto `t` (rebase table, rows 2 and 3 S6): resolved at `v`. -/
def flushOwn (v : Nat) (o : List Node) (t : Entry) : Entry :=
  { t with res := resolve t.res (allPairs o) v, log := o ++ t.log, pairs := sumPairs o + t.pairs }

/-- `Absorbed::Restart` from a writer rebase: `next_generation()`, `Building{Repair, base = the published
    hierarchy, log = the own nodes}` (03 §5 rebase table; `versioned.rs:792-795` @08c keeps `theirs`' count). -/
def restartE (o : List Node) (t : Entry) (nx : Nat) : Entry :=
  { st := .building .repair, res := t.res, lineage := t.lineage, log := o, pairs := sumPairs o, gen := nx,
    commits := t.commits }

def restart (R : Rules) (B v : Nat) (o : List Node) (t : Entry) (nx : Nat) : Entry :=
  checkIf R.checkAll B v (restartE o t nx)

/-- Row 1's relink: the own nodes on top of the published log. -/
def relinkE (o : List Node) (t : Entry) : Entry := { t with log := o ++ t.log, pairs := sumPairs o + t.pairs }

/-- Row 3, fold: the fork's copy with the fold re-applied (reading R3): `upkeep` and the retained log cleared as
    the fold's publish clears them, the publish counted. -/
def refoldE (o : List Node) (ours theirs : Entry) : Entry :=
  { ours with st := .built none, log := o, pairs := sumPairs o, commits := theirs.commits }

/-- 03 §5 "Writer rebase, by published state", for a fork the batches or a job overtook. `ours` is the fork's
    copy after the segments flushed before escalation, `theirs` the published copy; `b` is the remaining
    B_gil. -/
def rebase (R : Rules) (B v : Nat) (b : Bool) (ours theirs : Entry) (nx : Nat) : Entry × Nat :=
  let o := own v ours.log
  match ours.st, theirs.st with
  | .building _, .building _ =>
    -- Row 1: relink the fork's own nodes onto the published log.
    if R.relink then (checkIf R.checkAll B v (relinkE o theirs), nx) else (theirs, nx)
  | .building _, .built _ =>
    -- Row 2: the §6 flush over the own nodes within B_gil; over budget or an own Overflow, Restart.
    if b && !hasOv o then (flushOwn v o theirs, nx) else (restart R B v o theirs nx, nx + 1)
  | .built _, .built _ =>
    if ours.lineage == theirs.lineage then
      -- Row 3, fold: re-apply the fold to the fork's copy (reading R3: as the fold's publish, upkeep and the
      -- retained log cleared and the publish counted).
      (refoldE o ours theirs, nx)
    else if b then (flushOwn v o theirs, nx)   -- Row 3, S6: replay the own nodes onto the new lineage
    else (restart R B v o theirs nx, nx + 1)
  | .built _, .building _ =>
    -- Row 4: unreachable (`row4_unreachable`).
    (theirs, nx)

/-- 03 §7 "Retention", at a writer's publish. -/
def retain (e : Entry) : Entry :=
  match e.st with
  | .building _ => e
  | .built none => { e with log := [], pairs := 0 }
  | .built (some u) =>
    let l := e.log.filter (fun n => decide (u.since < n.seq))
    { e with log := l, pairs := sumPairs l }

/-- A writer that forked at `fork` and commits on `cur` (`fork.ver = cur.ver`: not stale, `mvcc_graph.rs:155-160`):
    the segments before escalation on the fork's copy (`commit.rs:85-103` @08c: the first `Commit` segment's
    `pending.commit` runs before `upgrade_to_write`), the rebase if a batch or job overtook it
    (`overtaken_by_indexer_batches`'s gate, `NI.overtakenAny`), the rest of the segments, then retention. -/
def writerE (R : Rules) (B : Nat) (cur fork : Pub) (pre post : List (List Pair × Out)) (b : Bool) (nx : Nat) :
    Entry × Nat :=
  let v := cur.ver + 1
  let r1 := segs R B v pre fork.e nx
  let r2 := if NI.overtakenAny (cur.batches != fork.batches) [overtaken R r1.1 cur.e]
            then rebase R B v b r1.1 cur.e r1.2 else r1
  let r3 := segs R B v post r2.1 r2.2
  (retain r3.1, r3.2)

def writerSys (R : Rules) (B : Nat) (fork : Pub) (pre post : List (List Pair × Out)) (b : Bool) (σ : Sys) : Sys :=
  let r := writerE R B σ.cur fork pre post b σ.next
  { cur := ⟨σ.cur.ver + 1, σ.cur.batches, r.1⟩
    hist := σ.cur :: σ.hist
    writes := σ.writes ++ (segPairs (pre ++ post)).map (fun p => (σ.cur.ver + 1, p))
    next := r.2 }

/-! ## Batches on the builder (03 §5 "Consumption", "Publish rules for a Building entry") -/

def consumes : Phase → Bool
  | .supergraph | .seal | .repair => true
  | _ => false

def adopts : Phase → Bool
  | .seal | .repair => true
  | _ => false

/-- The phase after a batch (`adv`: the batch reached its phase's end). -/
def nextPhase : Phase → Bool → Phase
  | .readSkeleton, adv => if adv then .order else .readSkeleton
  | .order, adv => if adv then .supergraph else .order
  | .supergraph, _ => .seed
  | .seed, adv => if adv then .relax else .seed
  | .relax, adv => if adv then .seal else .relax
  | .seal, _ => .seal
  | .repair, _ => .repair

/-- 03 §5 "Consumption": the consumer resolves every node of its copy against its fork `F` (an Overflow: a rescan,
    every pair), then its copy sets `head = None` and `pairs = 0`. -/
def consume (F : Nat) (e : Entry) : Entry :=
  { e with res := fun p => if hasOv e.log = true ∨ p ∈ allPairs e.log then F else e.res p, log := [], pairs := 0 }

/-- The batch's copy after `indexer_apply` (`versioned.rs:423-453` @08c): its phase's work, one more commit. -/
def batchCopy (F : Nat) (ph : Phase) (adv : Bool) (e : Entry) : Entry :=
  let c := if consumes ph then consume F e else e
  { c with st := .building (nextPhase ph adv), commits := e.commits + 1 }

/-- `Graph::fit` (`versioned.rs:551-584` @08c) for a CCH batch read on `S0`: `discarded_against`
    (`NI.discardedAgainst`), then `CatchUp(Refresh)` if the data version moved, `Swap` if only batches
    published, else `Same`. -/
def fit (S0 cur : Pub) (e1 : Entry) : NI.IndexerCommit :=
  match NI.discardedAgainst e1.toSE (some cur.e.toSE) with
  | some why => .discarded why
  | none => if cur.ver ≠ S0.ver then .catchUp .refresh else if cur.batches ≠ S0.batches then .swap else .same

def IsDiscarded : NI.IndexerCommit → Bool
  | .discarded _ => true
  | _ => false

/-- What the batch publishes, if anything (03 §5 publish rules and adoption). `b`: adoption's remaining B_gil. -/
def batchResult (b adv : Bool) (S0 cur : Pub) : Option Entry :=
  match S0.e.st with
  | .built _ => none
  | .building ph =>
    let e1 := batchCopy S0.ver ph adv S0.e
    let f := fit S0 cur e1
    if IsDiscarded f then none
    else
      let new := if f = .catchUp .refresh then relinked S0.ver cur.e.log else []
      if adopts ph then
        if (b || new.isEmpty) && !hasOv new then
          -- absorbed under the lock by the §6 path, resolved against the published graph: Built, no log
          some { e1 with st := .built none, res := resolve e1.res (allPairs new) cur.ver, log := [], pairs := 0 }
        else
          -- `Absorbed::Restart` keeps the generation and the job: `Building{Repair, base, log = relinked}`
          some { e1 with st := .building .repair, log := new, pairs := sumPairs new }
      else some { e1 with log := new ++ e1.log, pairs := sumPairs new + e1.pairs }

def batchSys (b adv : Bool) (S0 : Pub) (σ : Sys) : Sys :=
  match batchResult b adv S0 σ.cur with
  | some e => { σ with cur := ⟨σ.cur.ver, σ.cur.batches + 1, e⟩, hist := σ.cur :: σ.hist }
  | none => σ

/-! ## Built-entry jobs (03 §6 "Built-entry jobs (D6)", "Fold job") -/

/-- S6's hierarchy: a new lineage read at the fork `F`, then the retained nodes with `seq > F` replayed against
    the published graph at `c`. -/
def s6Res (F c : Nat) (log : List Node) : Pair → Nat :=
  resolve (fun _ => F) (allPairs (log.filter (fun n => decide (F < n.seq)))) c

/-- A fold or S6 job forked at `S0` publishes onto `cur`, fitted against generation and ticket; a mismatch is
    `Gone` (`none`). S6 replays the retained nodes with `seq > F` onto a new lineage read at `F`, within B_gil
    (`b`), else publishes nothing. Either clears `upkeep` and the log and counts in `indexer_commits`. -/
def jobResult (b : Bool) (S0 cur : Pub) (nx : Nat) : Option (Entry × Nat) :=
  match S0.e.st, cur.e.st with
  | .built (some u0), .built (some u) =>
    if S0.e.gen = cur.e.gen ∧ u0.ticket = u.ticket then
      match u0.kind with
      | .fold => some ({ cur.e with st := .built none, log := [], pairs := 0, commits := cur.e.commits + 1 }, nx)
      | .s6 =>
        if b then
          let e : Entry := { cur.e with st := .built none, res := s6Res S0.ver cur.ver cur.e.log,
                                        lineage := nx, log := [], pairs := 0, commits := cur.e.commits + 1 }
          some (e, nx + 1)
        else none
    else none
  | _, _ => none

def jobSys (b : Bool) (S0 : Pub) (σ : Sys) : Sys :=
  match jobResult b S0 σ.cur σ.next with
  | some (e, n) => { σ with cur := ⟨σ.cur.ver, σ.cur.batches + 1, e⟩, hist := σ.cur :: σ.hist, next := n }
  | none => σ

/-- Another index's batch publishes (`Swap` for it): this index's entry is unchanged. -/
def otherSys (σ : Sys) : Sys :=
  { σ with cur := { σ.cur with batches := σ.cur.batches + 1 }, hist := σ.cur :: σ.hist }

/-! ## The step relation -/

inductive Step (R : Rules) (B : Nat) : Sys → Sys → Prop
  | writer (σ : Sys) (fork : Pub) (hf : fork ∈ σ.cur :: σ.hist) (hv : fork.ver = σ.cur.ver)
      (pre post : List (List Pair × Out)) (b : Bool) : Step R B σ (writerSys R B fork pre post b σ)
  | batch (σ : Sys) (S0 : Pub) (h0 : S0 ∈ σ.cur :: σ.hist) (b adv : Bool) : Step R B σ (batchSys b adv S0 σ)
  | job (σ : Sys) (S0 : Pub) (h0 : S0 ∈ σ.cur :: σ.hist) (b : Bool) : Step R B σ (jobSys b S0 σ)
  | other (σ : Sys) : Step R B σ (otherSys σ)

/-- `CREATE INDEX` at version 0: Building at ReadSkeleton, generation 0, empty log. -/
def init : Sys :=
  { cur := ⟨0, 0, { st := .building .readSkeleton, res := fun _ => 0, lineage := 0, log := [], pairs := 0,
                    gen := 0, commits := 0 }⟩,
    hist := [], writes := [], next := 1 }

inductive Reach (R : Rules) (B : Nat) : Sys → Prop
  | init : Reach R B init
  | step {σ σ' : Sys} : Reach R B σ → Step R B σ σ' → Reach R B σ'

end CchReview
