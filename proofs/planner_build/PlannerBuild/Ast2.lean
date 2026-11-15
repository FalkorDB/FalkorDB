import PlannerBuild.Ast
/-
# parser/ast.rs: connected components and the `Display` impls

| here | there |
| --- | --- |
| `dfs`, `components` | `QueryGraph::dfs` ast.rs:844-883, `connected_components` ast.rs:823-842 |
| `exprFmt` | `Display for ExprIR` ast.rs:330-412 |
| `quantFmt` | `Display for QuantifierType` ast.rs:424-436 |
| `setItemFmt` | `Display for SetItem` ast.rs:896-923 |
| `clauseFmt` | `Display for QueryIR` ast.rs:1041-1146 |

`dfs` keeps ONE visited set of ids for nodes, relationships and paths (ids
only, as `HashSet<u32>`); the fuel bounds the recursion (each call marks a new node).
-/
namespace PlannerBuild.Ast

/-! ## Connected components -/

/-- `if visited.insert(rel.id) { component.add_relationship(rel) }` -/
def markRel (acc : List Nat × QG) (r : QR) : List Nat × QG :=
  if acc.1.contains r.alias.id then acc else (acc.1 ++ [r.alias.id], (addRel AV.eqv acc.2 r).1)

/-- `if !visited.contains(other) { dfs(other) }` -/
def follow (dfsF : QN → List Nat → QG → List Nat × QG) (m : QN) (x : List Nat × QG) : List Nat × QG :=
  if !x.1.contains m.alias.id then dfsF m x.1 x.2 else x

def relStep (dfsF : QN → List Nat → QG → List Nat × QG) (n : QN) (acc : List Nat × QG) (r : QR) :
    List Nat × QG :=
  if r.frm.alias.id == n.alias.id then follow dfsF r.to (markRel acc r)
  else if r.to.alias.id == n.alias.id then follow dfsF r.frm (markRel acc r)
  else acc

def pathStep (acc : List Nat × QG) (p : QP) : List Nat × QG :=
  if p.vars.all (fun v => acc.1.contains v.id) && !acc.1.contains p.var.id then
    (acc.1 ++ [p.var.id], (addPath AV.eqv acc.2 p).1)
  else acc

def dfs (g : QG) : Nat → QN → List Nat → QG → List Nat × QG
  | 0, _, vis, comp => (vis, comp)
  | f + 1, n, vis, comp =>
    let vis := vis ++ [n.alias.id]
    let comp := (addNode AV.eqv comp n).1
    let x := g.rels.foldl (relStep (dfs g f) n) (vis, comp)
    g.paths.foldl pathStep x

def components (g : QG) : List QG :=
  let fuel := g.nodes.length + 1
  (g.nodes.foldl (fun (acc : List Nat × List QG) n =>
    if acc.1.contains n.alias.id then acc
    else let x := dfs g fuel n acc.1 QG.default; (x.1, acc.2 ++ [x.2])) ([], [])).2

/-! ### Visited only grows -/

theorem pathStep_mono (acc : List Nat × QG) (p : QP) (i : Nat) (h : i ∈ acc.1) : i ∈ (pathStep acc p).1 := by
  unfold pathStep; split
  · exact List.mem_append_left _ h
  · exact h

theorem foldl_mono {β : Type} (step : List Nat × QG → β → List Nat × QG)
    (hs : ∀ acc b i, i ∈ acc.1 → i ∈ (step acc b).1) (l : List β) (acc : List Nat × QG) (i : Nat)
    (h : i ∈ acc.1) : i ∈ (l.foldl step acc).1 := by
  induction l generalizing acc with
  | nil => exact h
  | cons b bs ih => exact ih _ (hs acc b i h)

theorem markRel_mono (acc : List Nat × QG) (r : QR) (j : Nat) (h : j ∈ acc.1) : j ∈ (markRel acc r).1 := by
  unfold markRel; split
  · exact h
  · exact List.mem_append_left _ h

theorem relStep_mono (dfsF : QN → List Nat → QG → List Nat × QG)
    (hd : ∀ m vis comp j, j ∈ vis → j ∈ (dfsF m vis comp).1) (n : QN) (acc : List Nat × QG) (r : QR)
    (j : Nat) (h : j ∈ acc.1) : j ∈ (relStep dfsF n acc r).1 := by
  have hf : ∀ m x, j ∈ x.1 → j ∈ (follow dfsF m x).1 := by
    intro m x hx; unfold follow; split
    · exact hd _ _ _ _ hx
    · exact hx
  unfold relStep
  split
  · exact hf _ _ (markRel_mono acc r j h)
  · split
    · exact hf _ _ (markRel_mono acc r j h)
    · exact h

theorem dfs_mono (g : QG) : ∀ (f : Nat) (n : QN) (vis : List Nat) (comp : QG) (i : Nat),
    i ∈ vis → i ∈ (dfs g f n vis comp).1
  | 0, n, vis, comp, i, h => h
  | f + 1, n, vis, comp, i, h => by
    simp only [dfs]
    apply foldl_mono _ (fun acc p i h => pathStep_mono acc p i h)
    exact foldl_mono _ (fun acc r j hj => relStep_mono _ (fun m v c k hk => dfs_mono g f m v c k hk) n acc r j hj)
      _ _ _ (List.mem_append_left _ h)

/-- The start node is marked visited by its own DFS (with any fuel > 0). -/
theorem dfs_marks (g : QG) (f : Nat) (n : QN) (vis : List Nat) (comp : QG) :
    n.alias.id ∈ (dfs g (f + 1) n vis comp).1 := by
  simp only [dfs]
  apply foldl_mono _ (fun acc p i h => pathStep_mono acc p i h)
  exact foldl_mono _ (fun acc r j hj => relStep_mono _ (fun m v c k hk => dfs_mono g f m v c k hk) n acc r j hj)
    _ _ _ (List.mem_append_right _ (List.mem_singleton_self _))

/-- Every node's id is visited once `connected_components` is done: each node
either started a component or was reached before. -/
theorem components_cover (g : QG) :
    ∀ n ∈ g.nodes, n.alias.id ∈ (g.nodes.foldl (fun (acc : List Nat × List QG) n =>
      if acc.1.contains n.alias.id then acc
      else let x := dfs g (g.nodes.length + 1) n acc.1 QG.default; (x.1, acc.2 ++ [x.2])) ([], [])).1 := by
  intro n hn
  suffices ∀ (l : List QN) (acc : List Nat × List QG), (∀ m ∈ l, m ∈ g.nodes) →
      (∀ m ∈ l, m.alias.id ∈ (l.foldl (fun (acc : List Nat × List QG) n =>
        if acc.1.contains n.alias.id then acc
        else let x := dfs g (g.nodes.length + 1) n acc.1 QG.default; (x.1, acc.2 ++ [x.2])) acc).1) ∧
      (∀ i ∈ acc.1, i ∈ (l.foldl (fun (acc : List Nat × List QG) n =>
        if acc.1.contains n.alias.id then acc
        else let x := dfs g (g.nodes.length + 1) n acc.1 QG.default; (x.1, acc.2 ++ [x.2])) acc).1) from
    (this g.nodes ([], []) (fun m h => h)).1 n hn
  intro l
  induction l with
  | nil => intro acc _; exact ⟨fun m h => by simp at h, fun i h => h⟩
  | cons m ms ih =>
    intro acc hsub
    simp only [List.foldl_cons]
    have hsub' : ∀ x ∈ ms, x ∈ g.nodes := fun x hx => hsub x (List.mem_cons_of_mem _ hx)
    refine ⟨?_, ?_⟩
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · apply (ih _ hsub').2
        split
        · rename_i hc; simpa using hc
        · exact dfs_marks g _ x _ _
      · exact (ih _ hsub').1 x hx
    · intro i hi
      apply (ih _ hsub').2
      split
      · exact hi
      · exact dfs_mono g _ _ _ _ i hi

/-! ## `Display` impls -/

/-- What `Display for ExprIR` prints for each payload (`render` = the payload's own Display). -/
inductive EK
  | null | bool (b : String) | int (i : String) | float (f : String) | str (s : String) | otherConst (dbg : String)
  | list | map | var (v : String) | param (p : String) | length | getElement | getElements | isNode | isRel
  | or | xor | and | not | negate | eq | neq | lt | gt | le | ge | inOp | add | sub | mul | div | pow | modulo
  | distinct | property (p : String) | func (name : String) | quant (q v : String) | listComp (v : String)
  | reduce (a i : String) | patComp | paren | pattern | nested (id : Nat) | shortestPath | mapProjection
  | regexMatches | regexMatchList | regexReplace | case_

def exprFmt : EK → String
  | .null => "null" | .bool b => b | .int i => i | .float f => f | .str s => s | .otherConst d => "const(" ++ d ++ ")"
  | .list => "[]" | .map => "{}" | .var v => v | .param p => "@" ++ p
  | .length => "length()" | .getElement => "get_element()" | .getElements => "get_elements()"
  | .isNode => "is_node()" | .isRel => "is_relationship()"
  | .or => "or()" | .xor => "xor()" | .and => "and()" | .not => "not()" | .negate => "-negate()"
  | .eq => "=" | .neq => "<>" | .lt => "<" | .gt => ">" | .le => "<=" | .ge => ">=" | .inOp => "in()"
  | .add => "+" | .sub => "-" | .mul => "*" | .div => "/" | .pow => "^" | .modulo => "%"
  | .distinct => "distinct" | .property p => "property(" ++ p ++ ")" | .func n => n ++ "()"
  | .quant q v => q ++ " " ++ v | .listComp v => "list comp(" ++ v ++ ")"
  | .reduce a i => "reduce(" ++ a ++ ", " ++ i ++ ")" | .patComp => "pattern comp" | .paren => "()"
  | .pattern => "<pattern>" | .nested i => "nested plan #" ++ toString i | .shortestPath => "shortestPath()"
  | .mapProjection => "map_projection" | .regexMatches => "regex_matches()"
  | .regexMatchList => "string.matchRegEx()" | .regexReplace => "string.replaceRegEx()" | .case_ => "CASE"

/-- The comparison and arithmetic operators print pairwise-distinct symbols. -/
theorem exprFmt_ops_distinct :
    [EK.eq, .neq, .lt, .gt, .le, .ge, .add, .sub, .mul, .div, .pow, .modulo].map exprFmt =
      ["=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/", "^", "%"] ∧
    (["=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/", "^", "%"] : List String).Nodup := by
  constructor
  · rfl
  · decide

theorem exprFmt_param (p : String) : exprFmt (.param p) = "@" ++ p := rfl
theorem exprFmt_func (n : String) : exprFmt (.func n) = n ++ "()" := rfl

inductive QK | all | any | none | single

def quantFmt : QK → String
  | .all => "all" | .any => "any" | .none => "none" | .single => "single"

theorem quantFmt_injective (a b : QK) (h : quantFmt a = quantFmt b) : a = b := by
  cases a <;> cases b <;> simp_all [quantFmt] <;> contradiction

/-- `SET` items: `target = value` / `target += value`, or `var:L1:L2`. -/
def setItemFmt : (String × String × Bool) ⊕ (String × List String) → String
  | .inl (t, v, replace) => t ++ " " ++ (if replace then "=" else "+=") ++ " " ++ v
  | .inr (var, labels) => var ++ ":" ++ ":".intercalate labels

theorem setItemFmt_replace (t v : String) : setItemFmt (.inl (t, v, true)) = t ++ " = " ++ v := by
  simp [setItemFmt, String.append_assoc]
theorem setItemFmt_merge (t v : String) : setItemFmt (.inl (t, v, false)) = t ++ " += " ++ v := by
  simp [setItemFmt, String.append_assoc]

/-- Clause headers of `Display for QueryIR` (the debug dump); bodies are the parts' Displays. -/
inductive CK
  | call (name : String) (args : List String) | match_ (pat : String) | unwind (var e : String)
  | merge (pat : String) | create (pat : String) | delete (es : List String) | set (items : List String)
  | remove (items : List String) | loadCsv (path var : String) | with_ (names : List String)
  | return_ (names : List String) | query (cs : List String) | union (all : Bool) (bs : List String)
  | foreach (var list : String) (body : List String) | callSub (body : String)

def clauseFmt : CK → String
  | .call n as => n ++ "():\n" ++ String.join as
  | .match_ p => "MATCH " ++ p ++ "\n"
  | .unwind v e => "UNWIND " ++ v ++ ":\n" ++ e
  | .merge p => "MERGE " ++ p ++ "\n"
  | .create p => "CREATE " ++ p
  | .delete es => "DELETE:\n" ++ String.join es
  | .set is => "SET:\n" ++ String.join is
  | .remove is => "REMOVE:\n" ++ String.join is
  | .loadCsv p v => "LOAD CSV FROM " ++ p ++ " AS " ++ v ++ ":\n"
  | .with_ ns => "WITH:\n" ++ String.join ns
  | .return_ ns => "RETURN:\n" ++ String.join ns
  | .query cs => String.join cs
  | .union all bs => (if all then "\nUNION ALL\n" else "\nUNION\n").intercalate bs
  | .foreach v l b => "FOREACH(" ++ v ++ " IN " ++ l ++ " | " ++ String.join b ++ ")"
  | .callSub b => "CALL { " ++ b ++ " }"

theorem clauseFmt_union (all : Bool) (bs : List String) :
    clauseFmt (.union all bs) = (if all then "\nUNION ALL\n" else "\nUNION\n").intercalate bs := rfl

theorem clauseFmt_match (p : String) : clauseFmt (.match_ p) = "MATCH " ++ p ++ "\n" := rfl

end PlannerBuild.Ast
