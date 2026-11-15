/-
# EXPLAIN text (mod.rs:341-536)

| here | there |
| --- | --- |
| `FRel`, `typesStr`, `hopsStr`, `fmtVarLen` | `fmt_var_len_rel` mod.rs:347-372 |
| `fmtRelDir`, `fmtRel` | `fmt_rel_with_labels_dir` mod.rs:379-415, `fmt_rel_with_labels` mod.rs:375-377 |
| `IRK`, `irName`, `irDetail`, `irFmt` | `impl Display for IR` mod.rs:419-536 |

Strings are built exactly as the `format!` calls do (payload `Display`s are
parameters: the node/relationship/pattern text is passed in already rendered).
-/
namespace PlannerBuild.Fmt

structure FRel where
  alias : String
  types : List String
  minHops : Option Nat
  maxHops : Option Nat
  frm : String
  to : String
  toLabels : List String
  bidirectional : Bool

def typesStr (r : FRel) : String :=
  if r.types.isEmpty then "" else ":" ++ "|".intercalate r.types

def hopsStr (r : FRel) : String :=
  let mn := r.minHops.getD 1
  if mn = 1 ∧ r.maxHops = some 1 then ""
  else "*" ++ toString mn ++ ".." ++ (match r.maxHops with | some m => toString m | none => "INF")

/-- `(from)-[alias:T1|T2*min..max]->(to)` -/
def fmtVarLen (r : FRel) : String :=
  "(" ++ r.frm ++ ")-[" ++ r.alias ++ typesStr r ++ hopsStr r ++ "]->(" ++ r.to ++ ")"

/-- The hop range is printed unless it is exactly `*1..1`; a missing upper bound prints `INF`. -/
theorem hopsStr_empty (r : FRel) : hopsStr r = "" ↔ r.minHops.getD 1 = 1 ∧ r.maxHops = some 1 := by
  unfold hopsStr
  by_cases h : r.minHops.getD 1 = 1 ∧ r.maxHops = some 1
  · simp [h]
  · simp only [h, ↓reduceIte, iff_false]
    intro he
    have := congrArg String.length he
    simp [String.length_append] at this

theorem hopsStr_inf (r : FRel) (h : r.maxHops = none) :
    hopsStr r = "*" ++ toString (r.minHops.getD 1) ++ "..INF" := by
  simp [hopsStr, h, String.append_assoc]

def arrows (bidi transposed : Bool) : String × String :=
  if bidi then ("", "") else if transposed then ("<", "") else ("", ">")

def fmtNode (alias : String) (labels : List String) : String :=
  if labels.isEmpty then alias else alias ++ ":" ++ ":".intercalate labels

def fmtRelDir (r : FRel) (transposed : Bool) : String :=
  let (l, rt) := arrows r.bidirectional transposed
  let toS := fmtNode r.to r.toLabels
  if r.alias.startsWith "_anon" then "(" ++ r.frm ++ ")" ++ l ++ "-" ++ rt ++ "(" ++ toS ++ ")"
  else if r.types.isEmpty then
    "(" ++ r.frm ++ ")" ++ l ++ "-[" ++ r.alias ++ "]-" ++ rt ++ "(" ++ toS ++ ")"
  else "(" ++ r.frm ++ ")" ++ l ++ "-[" ++ r.alias ++ ":" ++ "|".intercalate r.types ++ "]-" ++ rt ++
    "(" ++ toS ++ ")"

def fmtRel (r : FRel) : String := fmtRelDir r false

theorem fmtRel_eq (r : FRel) : fmtRel r = fmtRelDir r false := rfl

theorem arrows_table (b t : Bool) :
    arrows b t = (if b then ("", "") else if t then ("<", "") else ("", ">")) := rfl

/-- Anonymous edges print no bracket at all. -/
theorem fmtRelDir_anon (r : FRel) (t : Bool) (h : r.alias.startsWith "_anon" = true) :
    fmtRelDir r t = "(" ++ r.frm ++ ")" ++ (arrows r.bidirectional t).1 ++ "-" ++
      (arrows r.bidirectional t).2 ++ "(" ++ fmtNode r.to r.toLabels ++ ")" := by
  simp [fmtRelDir, h]

/-! ## `Display for IR` -/

/-- IR variants with the payload their text shows (rendered). -/
inductive IRK
  | argument | optional | procedureCall | unwind
  | create (pat : String) | merge (pat : String) | delete | set | remove
  | allNodeScan (n : String) | nodeByLabelScan (n : String) | includePending (n : String)
  | nodeByIndexScan (n : String) | edgeByIndexScan (r : FRel)
  | nodeByFulltextScan | edgeByFulltextScan | nodeByVectorScan | edgeByVectorScan
  | nodeByLabelAndIdScan (n : String) | nodeByIdSeek
  | condTraverse (r : FRel) (transposed : Bool) (chain : Nat) (optional : Bool)
  | condVarLenTraverse (r : FRel) (expandInto : Bool)
  | allShortestPaths (r : String) | expandInto (r : FRel)
  | pathBuilder | filter | cartesianProduct | valueHashJoin | apply | semiApply | antiSemiApply
  | orApplyMultiplexer | loadCsv | sort | skip | limit | aggregate | project | commit
  | forEach (v : String) | union | nestedPlans | distinct
  | createIndex (label attrs : String) | dropIndex (label attrs : String)

def ctName (optional : Bool) : String :=
  if optional then "Optional Conditional Traverse" else "Conditional Traverse"

def irFmt : IRK → String
  | .argument => "Argument"
  | .optional => "Optional"
  | .procedureCall => "ProcedureCall"
  | .unwind => "Unwind"
  | .create p => "Create | " ++ p
  | .merge p => "Merge | " ++ p
  | .delete => "Delete"
  | .set | .remove => "Update"
  | .allNodeScan n => "All Node Scan | " ++ n
  | .nodeByLabelScan n => "Node By Label Scan | " ++ n
  | .includePending n => "Include Pending | " ++ n
  | .nodeByIndexScan n => "Node By Index Scan | " ++ n
  | .edgeByIndexScan r => "Edge By Index Scan | " ++ fmtRel r
  | .nodeByFulltextScan => "Node By Fulltext Index Scan"
  | .edgeByFulltextScan => "Edge By Fulltext Index Scan"
  | .nodeByVectorScan => "Node By Vector Index Scan"
  | .edgeByVectorScan => "Edge By Vector Index Scan"
  | .nodeByLabelAndIdScan n => "Node By Label and ID Scan | " ++ n
  | .nodeByIdSeek => "NodeByIdSeek"
  | .condTraverse r t ch o =>
    if ch = 0 then ctName o ++ " | " ++ fmtRelDir r t
    else ctName o ++ " | " ++ fmtRelDir r t ++ " (+ " ++ toString ch ++ " fused hop" ++
      (if ch = 1 then "" else "s") ++ ")"
  | .condVarLenTraverse r ei =>
    (if ei then "Conditional Variable Length Traverse (Expand Into)"
     else "Conditional Variable Length Traverse") ++ " | " ++ fmtVarLen r
  | .allShortestPaths r => "All Shortest Paths | " ++ r
  | .expandInto r => "Expand Into | " ++ fmtRel r
  | .pathBuilder => "PathBuilder"
  | .filter => "Filter"
  | .cartesianProduct => "Cartesian Product"
  | .valueHashJoin => "Value Hash Join"
  | .apply => "Apply"
  | .semiApply => "Semi Apply"
  | .antiSemiApply => "Anti Semi Apply"
  | .orApplyMultiplexer => "Or Apply Multiplexer"
  | .loadCsv => "Load CSV"
  | .sort => "Sort"
  | .skip => "Skip"
  | .limit => "Limit"
  | .aggregate => "Aggregate"
  | .project => "Project"
  | .commit => "Commit"
  | .forEach v => "ForEach | " ++ v
  | .union => "Union"
  | .nestedPlans => "Nested Plans"
  | .distinct => "Distinct"
  | .createIndex l a => "Create Index | :" ++ l ++ "(" ++ a ++ ")"
  | .dropIndex l a => "Drop Index | :" ++ l ++ "(" ++ a ++ ")"

/-- `SET` and `REMOVE` print the same line. -/
theorem fmt_set_remove : irFmt .set = irFmt .remove := rfl

/-- One fused hop is singular, more are plural. -/
theorem fmt_fused (r : FRel) (t o : Bool) (n : Nat) (h : 2 ≤ n) :
    irFmt (.condTraverse r t 1 o) = ctName o ++ " | " ++ fmtRelDir r t ++ " (+ 1 fused hop)" ∧
    irFmt (.condTraverse r t n o) =
      ctName o ++ " | " ++ fmtRelDir r t ++ " (+ " ++ toString n ++ " fused hops)" := by
  constructor
  · simp [irFmt, String.append_assoc]; rfl
  · have h1 : n ≠ 0 := by omega
    have h2 : n ≠ 1 := by omega
    simp [irFmt, h1, h2, String.append_assoc]

theorem fmt_condTraverse_plain (r : FRel) (t o : Bool) :
    irFmt (.condTraverse r t 0 o) = ctName o ++ " | " ++ fmtRelDir r t := rfl

end PlannerBuild.Fmt
