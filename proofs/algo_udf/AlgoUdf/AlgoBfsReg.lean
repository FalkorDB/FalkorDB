import AlgoUdf.Repo
import AlgoUdf.Config
/-! # `algo.BFS` result assembly and the procedure registration table

| here | there (`algo_procedures.rs`) |
| --- | --- |
| `algoBfs`         | `algo_bfs` :1063-1205; LAGraph's BFS output is the input `Lag` (AXIOMATISED: `LAGr_BreadthFirstSearch_Extended` level/parent vectors, extracted in ascending index order) |
| `relOf p c`       | `g.get_src_dest_relationships(parent, child, &rel_types).next()` :1144 |
| `Entry`, `addProc`| `Functions::add_procedure` (functions/mod.rs:943-965: panics on a duplicate lowercase name) |
| `registerX`       | `register_pagerank` :702 … `register_maxflow` :2844, via `cypher_fn!` (functions/mod.rs:328-352) |
| `register`        | `register` :687 |
-/
namespace AlgoUdf.AlgoBfsReg
open AlgoUdf AlgoUdf.Config

/-! ## algo.BFS -/

/-- LAGraph BFS output: `(index, parent)` and `(index, level)` entries. -/
structure Lag where
  parent : List (Nat × Nat)
  level : List (Nat × Nat)

inductive BfsOut | empty | rows (nodes : List Nat) (edges : List Nat)
  deriving DecidableEq, Repr

/-- The `want_edges` branch :1112-1140. -/
def edgesBranch (src : Nat) (hasDel : Bool) (del : Nat → Bool) (relOf : Nat → Nat → Option Nat) :
    List (Nat × Nat) → List Nat × List Nat
  | [] => ([], [])
  | (c, p) :: t =>
    let (ns, es) := edgesBranch src hasDel del relOf t
    if c = src then (ns, es)
    else if hasDel && (del c || del p) then (ns, es)
    else (c :: ns, (match relOf p c with | some e => [e] | none => []) ++ es)

/-- The level branch :1145-1158. -/
def levelBranch (src : Nat) (hasDel : Bool) (del : Nat → Bool) : List (Nat × Nat) → List Nat
  | [] => []
  | (c, _) :: t =>
    if c = src then levelBranch src hasDel del t
    else if hasDel && del c then levelBranch src hasDel del t
    else c :: levelBranch src hasDel del t

def algoBfs (args : List Val) (nodeCount : Nat) (del : Nat → Bool) (hasDel : Bool) (wantEdges : Bool)
    (lag : Lag) (relOf : Nat → Nat → Option Nat) : Except String BfsOut :=
  match args with
  | a0 :: a1 :: a2 :: _ =>
    match a0 with
    | .null => .ok .empty
    | .node src =>
      match a1 with
      | .int _ =>
        match optString a2 with
        | .error e => .error e
        | .ok _ =>
          if nodeCount = 0 then .ok .empty
          else if del src then .error "Source node not found in graph"
          else
            let (ns, es) := if wantEdges then edgesBranch src hasDel del relOf lag.parent
                            else (levelBranch src hasDel del lag.level, [])
            if ns = [] then .ok .empty else .ok (.rows ns es)
      | _ => .error "maxDepth must be an integer"
    | _ => .error "Source must be a node or null"
  | _ => .error "arity"

/-- The parent→child relationship exists for every reported child (LAGraph
contract: a parent is adjacent to its child in the typed adjacency matrix). -/
def ParentsAdjacent (src : Nat) (relOf : Nat → Nat → Option Nat) (parent : List (Nat × Nat)) : Prop :=
  ∀ c p, (c, p) ∈ parent → c ≠ src → relOf p c ≠ none

/-- **Aligned columns**: under the LAGraph contract each reported node comes
with exactly one edge, the parent→node relationship. -/
theorem edgesBranch_aligned (src : Nat) (hasDel : Bool) (del : Nat → Bool) (relOf : Nat → Nat → Option Nat) :
    ∀ parent, ParentsAdjacent src relOf parent →
      (edgesBranch src hasDel del relOf parent).2.length = (edgesBranch src hasDel del relOf parent).1.length := by
  intro parent
  induction parent with
  | nil => intro _; rfl
  | cons cp t ih =>
    obtain ⟨c, p⟩ := cp
    intro hadj
    have ih' := ih (fun c' p' h hc => hadj c' p' (List.mem_cons_of_mem _ h) hc)
    simp only [edgesBranch]
    split
    · exact ih'
    · split
      · exact ih'
      · rename_i hc _
        have := hadj c p (by simp) hc
        cases hr : relOf p c with
        | none => exact absurd hr this
        | some e => simp [ih']

/-- **The `nodes` column does not depend on the yields bitmask** when LAGraph's
parent and level vectors share their pattern and no reached node's parent is
deleted (deleted nodes have no edges). -/
theorem branches_agree (src : Nat) (hasDel : Bool) (del : Nat → Bool) (relOf : Nat → Nat → Option Nat) :
    ∀ (parent level : List (Nat × Nat)), parent.map (·.1) = level.map (·.1) →
      (∀ c p, (c, p) ∈ parent → c ≠ src → del p = false) →
      (edgesBranch src hasDel del relOf parent).1 = levelBranch src hasDel del level := by
  intro parent
  induction parent with
  | nil => intro level h _; cases level; rfl; simp at h
  | cons cp t ih =>
    intro level h hp
    obtain ⟨c, p⟩ := cp
    cases level with
    | nil => simp at h
    | cons cl lt =>
      obtain ⟨c', l⟩ := cl
      simp only [List.map_cons, List.cons.injEq] at h
      obtain ⟨rfl, ht⟩ := h
      have ih' := ih lt ht (fun c'' p'' hm hc => hp c'' p'' (List.mem_cons_of_mem _ hm) hc)
      simp only [edgesBranch, levelBranch]
      split
      · exact ih'
      · rename_i hcs
        have hpd := hp c p (by simp) hcs
        simp only [hpd, Bool.or_false]
        split
        · exact ih'
        · simp [ih']

theorem algoBfs_null (rest : List Val) (a1 a2 : Val) (n : Nat) (del : Nat → Bool) (hd w : Bool) (lag : Lag)
    (relOf : Nat → Nat → Option Nat) : algoBfs (.null :: a1 :: a2 :: rest) n del hd w lag relOf = .ok .empty := rfl

theorem algoBfs_deleted_src (src d : Nat) (rel : Val) (rest : List Val) (n : Nat) (del : Nat → Bool)
    (hd w : Bool) (lag : Lag) (relOf : Nat → Nat → Option Nat) (hn : n ≠ 0) (hdel : del src = true)
    (hr : rel = .null) :
    algoBfs (.node src :: .int d :: rel :: rest) n del hd w lag relOf = .error "Source node not found in graph" := by
  subst hr; simp [algoBfs, optString, hn, hdel]

theorem algoBfs_bad_source (s : String) (a1 a2 : Val) (rest : List Val) (n : Nat) (del : Nat → Bool)
    (hd w : Bool) (lag : Lag) (relOf : Nat → Nat → Option Nat) :
    algoBfs (.str s :: a1 :: a2 :: rest) n del hd w lag relOf = .error "Source must be a node or null" := rfl

/-- Rows never contain the source. -/
theorem levelBranch_no_src (src : Nat) (hasDel : Bool) (del : Nat → Bool) :
    ∀ level, src ∉ levelBranch src hasDel del level := by
  intro level
  induction level with
  | nil => simp [levelBranch]
  | cons cl t ih =>
    obtain ⟨c, _⟩ := cl
    simp only [levelBranch]
    split
    · exact ih
    · split
      · exact ih
      · rename_i h _; simp only [List.mem_cons, not_or]; exact ⟨fun e => h e.symm, ih⟩

/-! ## Registration -/

structure Entry where
  name : String
  args : Nat
  yields : List String
  deriving DecidableEq, Repr

/-- `add_procedure`: `none` = the duplicate-name `assert!` panic. -/
def addProc (fs : List Entry) (e : Entry) : Option (List Entry) :=
  if fs.all (fun e' => Repo.asciiLow e'.name != Repo.asciiLow e.name) then some (fs ++ [e]) else none

def ePagerank : Entry := ⟨"algo.pageRank", 2, ["node", "score"]⟩
def eWcc : Entry := ⟨"algo.WCC", 1, ["node", "componentId"]⟩
def eBetweenness : Entry := ⟨"algo.betweenness", 1, ["node", "score"]⟩
def eBfs : Entry := ⟨"algo.BFS", 3, ["nodes", "edges"]⟩
def eCdlp : Entry := ⟨"algo.labelPropagation", 1, ["node", "communityId"]⟩
def eMsf : Entry := ⟨"algo.MSF", 1, ["nodes", "edges"]⟩
def eSp : Entry := ⟨"algo.SPpaths", 1, ["path", "pathWeight", "pathCost"]⟩
def eSs : Entry := ⟨"algo.SSpaths", 1, ["path", "pathWeight", "pathCost"]⟩
def eHarmonic : Entry := ⟨"algo.HarmonicCentrality", 1, ["node", "score", "reachable"]⟩
def eMaxflow : Entry := ⟨"algo.maxFlow", 1, ["nodes", "edges", "edgeFlows", "maxFlow"]⟩

def table : List Entry :=
  [ePagerank, eWcc, eBetweenness, eBfs, eCdlp, eMsf, eSp, eSs, eHarmonic, eMaxflow]

def registerPagerank (fs : List Entry) := addProc fs ePagerank
def registerWcc (fs : List Entry) := addProc fs eWcc
def registerBetweenness (fs : List Entry) := addProc fs eBetweenness
def registerBfs (fs : List Entry) := addProc fs eBfs
def registerCdlp (fs : List Entry) := addProc fs eCdlp
def registerMsf (fs : List Entry) := addProc fs eMsf
def registerSpPaths (fs : List Entry) := addProc fs eSp
def registerSsPaths (fs : List Entry) := addProc fs eSs
def registerHarmonic (fs : List Entry) := addProc fs eHarmonic
def registerMaxflow (fs : List Entry) := addProc fs eMaxflow

/-- `register` :687 in its call order. -/
def register (fs : List Entry) : Option (List Entry) :=
  registerPagerank fs >>= registerWcc >>= registerBetweenness >>= registerBfs >>= registerCdlp >>=
    registerMsf >>= registerSpPaths >>= registerSsPaths >>= registerHarmonic >>= registerMaxflow

/-- The ten names are pairwise distinct after lowercasing, so registration into
an empty table never hits the duplicate-name panic. -/
theorem table_names_distinct : (table.map (fun e => Repo.asciiLow e.name)).Nodup := by decide

/-- Each `register_*` appends its entry unless the lowercased name is taken. -/
theorem addProc_spec (fs : List Entry) (e : Entry) :
    (addProc fs e = some (fs ++ [e]) ↔ ∀ e' ∈ fs, Repo.asciiLow e'.name ≠ Repo.asciiLow e.name) ∧
    (addProc fs e = none ↔ ∃ e' ∈ fs, Repo.asciiLow e'.name = Repo.asciiLow e.name) := by
  unfold addProc
  constructor
  · split <;> simp_all
  · split <;> simp_all

theorem register_empty : register [] = some table := by
  simp only [register, registerPagerank, registerWcc, registerBetweenness, registerBfs, registerCdlp,
    registerMsf, registerSpPaths, registerSsPaths, registerHarmonic, registerMaxflow]
  decide

end AlgoUdf.AlgoBfsReg
