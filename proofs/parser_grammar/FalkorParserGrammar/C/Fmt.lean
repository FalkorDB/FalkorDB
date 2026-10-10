/-
# The `Display` impls of `graph/src/parser/ast.rs` and small parser glue

Names are ids, printed with `toString`. Each theorem states exactly what the
impl writes for the shape at hand.
-/
import FalkorParserGrammar.C.IR
import FalkorParserGrammar.C.Pattern
import FalkorParserGrammar.C.Clauses

namespace FalkorParserGrammar.C

def joinSep (sep : String) : List String → String
  | [] => ""
  | [a] => a
  | a :: as => a ++ sep ++ joinSep sep as

/-- `impl Display for ExprIR` (ast.rs:330-401), one arm per operator. -/
def fmtOp : EOp → String
  | .var v => toString v
  | .param p => "@" ++ toString p
  | .int i => toString i
  | .str s => toString s
  | .null => "null"
  | .bool b => toString b
  | .prop p => "property(" ++ toString p ++ ")"
  | .func f _ => toString f ++ "()"
  | .list => "[]"
  | .map => "{}"
  | .paren => "()"
  | .distinct => "distinct"
  | .case_ _ => "CASE"
  | .quant q v => (match q with | 0 => "all" | 1 => "any" | 2 => "none" | _ => "single") ++ " " ++ toString v
  | .listComp v => "list comp(" ++ toString v ++ ")"
  | .reduce a i => "reduce(" ++ toString a ++ ", " ++ toString i ++ ")"
  | .patComp => "pattern comp"
  | .shortest .. => "shortestPath()"
  | .pattern => "<pattern>"
  | .other _ => "?"

theorem fmtOp_var (v : Nat) : fmtOp (.var v) = toString v := rfl
theorem fmtOp_param (p : Nat) : fmtOp (.param p) = "@" ++ toString p := rfl
theorem fmtOp_prop (p : Nat) : fmtOp (.prop p) = "property(" ++ toString p ++ ")" := rfl

/-- `impl Display for QuantifierType` (ast.rs:424-436). -/
def fmtQuant : Nat → String
  | 0 => "all" | 1 => "any" | 2 => "none" | _ => "single"
theorem fmtQuant_spec : fmtQuant 0 = "all" ∧ fmtQuant 1 = "any" ∧ fmtQuant 2 = "none" ∧
    fmtQuant 3 = "single" := ⟨rfl, rfl, rfl, rfl⟩

/-- `impl Display for QueryNode` (ast.rs:481-491). -/
def fmtNode (n : QNode Nat) : String :=
  if n.labels.isEmpty then "(" ++ toString n.alias ++ ")"
  else "(" ++ toString n.alias ++ ":" ++ joinSep ":" (n.labels.map toString) ++ ")"
theorem fmtNode_nolabel (n : QNode Nat) (h : n.labels = []) : fmtNode n = "(" ++ toString n.alias ++ ")" := by
  simp [fmtNode, h]

/-- `impl Display for QueryRelationship` (ast.rs:542-566). -/
def fmtRel (r : QRel Nat) : String :=
  let dir := if r.bidir then "" else ">"
  if r.types.isEmpty then
    "(" ++ toString r.src.alias ++ ")-[" ++ toString r.alias ++ "]-" ++ dir ++ "(" ++ toString r.dst.alias ++ ")"
  else
    "(" ++ toString r.src.alias ++ ")-[" ++ toString r.alias ++ ":" ++ joinSep "|" (r.types.map toString) ++
      "]-" ++ dir ++ "(" ++ toString r.dst.alias ++ ")"
theorem fmtRel_untyped_undirected (r : QRel Nat) (ht : r.types = []) (hb : r.bidir = true) :
    fmtRel r = "(" ++ toString r.src.alias ++ ")-[" ++ toString r.alias ++ "]-(" ++ toString r.dst.alias ++ ")" := by
  simp [fmtRel, ht, hb, String.append_assoc]

/-- `impl Display for QueryGraph` (ast.rs:640-660): every node, relationship
and path variable followed by `", "`. -/
def fmtGraph (g : QG Nat) : String :=
  String.join (g.nodes.map (fun n => fmtNode n ++ ", ")) ++
  String.join (g.rels.map (fun r => fmtRel r ++ ", ")) ++
  String.join (g.paths.map (fun p => toString p.var ++ ", "))
theorem fmtGraph_empty : fmtGraph QG.empty = "" := rfl

/-- `impl Display for SetItem` (ast.rs:896-923). -/
def fmtSetItem (fe : ET → String) : SetItem → String
  | .attr t v r => fe t ++ " " ++ (if r then "=" else "+=") ++ " " ++ fe v
  | .label v ls => toString v ++ ":" ++ joinSep ":" (ls.map toString)
theorem fmtSetItem_attr (fe : ET → String) (t v : ET) (r : Bool) :
    fmtSetItem fe (.attr t v r) = fe t ++ " " ++ (if r then "=" else "+=") ++ " " ++ fe v := rfl

/-- `impl Display for QueryIR` (ast.rs:1041-1147), for the clause headers. -/
def fmtIR (fe : ET → String) : IR → String
  | .match_ p _ _ => "MATCH " ++ fmtGraph p ++ "\n"
  | .merge p _ _ => "MERGE " ++ fmtGraph p ++ "\n"
  | .create p => "CREATE " ++ fmtGraph p
  | .unwind e v => "UNWIND " ++ toString v ++ ":\n" ++ fe e
  | .delete es _ => "DELETE:\n" ++ String.join (es.map fe)
  | .set is => "SET:\n" ++ String.join (is.map (fmtSetItem fe))
  | .remove is => "REMOVE:\n" ++ String.join (is.map fe)
  | .with_ p _ _ => "WITH:\n" ++ String.join (p.exprs.map (fun e => toString e.1))
  | .return_ p _ => "RETURN:\n" ++ String.join (p.exprs.map (fun e => toString e.1))
  | _ => "…"
theorem fmtIR_return (fe : ET → String) (p : Proj) (w : Bool) :
    fmtIR fe (.return_ p w) = "RETURN:\n" ++ String.join (p.exprs.map (fun e => toString e.1)) := rfl

/-! ## Parser glue -/

/-- `add_pattern_node` (cypher.rs:1343-1354): a new alias is added; a
repeated one is merged into the existing node in MATCH and ignored in
CREATE / MERGE. -/
theorem addPatNode_spec (cl : CK) (g : QG Nat) (seen : List Nat) (n : QNode Nat) :
    (n.alias ∉ seen → addPatNode cl (g, seen) n = ((g.addNode n).2, seen ++ [n.alias])) ∧
    (n.alias ∈ seen → cl = .match_ → addPatNode cl (g, seen) n = (g.mergeNode n, seen)) ∧
    (n.alias ∈ seen → cl ≠ .match_ → addPatNode cl (g, seen) n = (g, seen)) := by
  refine ⟨fun h => ?_, fun h hc => ?_, fun h hc => ?_⟩ <;> simp [addPatNode, *]

/-- `match_dot_property_separator` (cypher.rs:2623-2640): consumes a `.`, fails otherwise. -/
theorem dotSep_spec (s : PS) : (s.cur = .dot → tok .dot s = .ok () s.adv) ∧
    (s.cur ≠ .dot → tok .dot s = .err) := by
  constructor <;> intro h <;> simp [tok, h]

/-- `parse_property_name` (cypher.rs:2673-2685) reads exactly like `parse_ident`. -/
theorem propName_spec : propName = ident := rfl

end FalkorParserGrammar.C
