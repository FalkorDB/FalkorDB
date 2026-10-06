import AlgoUdf.MsfFlow
import AlgoUdf.Config
import AlgoUdf.Dfs
/-! # Remaining helpers of `algo_procedures.rs`

| here | there |
| --- | --- |
| `newMsg`, `msgToString`  | `new_msg` :101, `msg_to_string` :105 (`CStr::from_ptr`: bytes up to the first NUL) |
| `Ctx`, `msfScore`        | `MsfWeightCtx`, `msf_score` :144 (attribute lookup as an `Option Attr`) |
| `indexOp`, `valueOp`     | `msf_scored_edge_index_op` :176, `msf_scored_edge_value_op` :200 (null guard = no write) |
| `scoreOf`                | `msf_score_of` :266 |
| `nodeValue`              | `node_value` :463 |
| `collectNodeIds`         | `collect_node_ids` :469 (label scans AXIOMATISED as `scan`; FxHashSet + `sort_unstable`) |
| `activeNodes`            | `active_node_set` :493 / `Graph::get_nodes(∅, 0)` (graph.rs:2177-2198) |
| `Own`, `lifecycle`       | `create_lagraph_graph` :372, `delete_lagraph_graph` :397, `_borrowed` :404, `_maybe_borrowed` :417 under the LAGraph ownership contract |
| `cmpKey`                 | `cmp_found_path` :2092 (= `Dfs.fpLt`) |
-/
namespace AlgoUdf.AlgoGlue
open AlgoUdf

/-! ## LAGraph message buffer -/

def newMsg : List Nat := List.replicate 256 0

/-- `CStr::from_ptr(..).to_string_lossy()`: the bytes before the first NUL. -/
def msgToString (m : List Nat) : List Nat := m.takeWhile (· ≠ 0)

theorem msgToString_new : msgToString newMsg = [] := by decide

theorem msgToString_prefix (m : List Nat) : ∃ rest, m = msgToString m ++ rest ∧
    (rest = [] ∨ rest.head? = some 0) := by
  refine ⟨m.dropWhile (· ≠ 0), (List.takeWhile_append_dropWhile).symm, ?_⟩
  induction m with
  | nil => simp
  | cons x t ih =>
    by_cases hx : x = 0
    · right; simp [List.dropWhile_cons, hx]
    · simp only [List.dropWhile_cons, ne_eq, hx, not_false_eq_true, decide_true, if_true]; exact ih

/-! ## MSF scoring callbacks -/

inductive Attr | float (f : Int) | int (k : Int) | other deriving DecidableEq, Repr

structure Ctx where
  attr : Nat → Option Attr
  maximize : Bool
  unit : Bool

/-- `msf_score`: `1.0` when unweighted, the numeric weight (negated when
maximizing), `±inf` (here `MsfFlow.inf`) when missing / non-numeric. Finite
f64 weights are modelled by their integer image. -/
def msfScore (ctx : Ctx) (e : Nat) : MsfFlow.Score :=
  if ctx.unit then .num 1 else
  let miss : Int := if ctx.maximize then -MsfFlow.inf else MsfFlow.inf
  let raw : Int := match ctx.attr e with
    | some (.float f) => f
    | some (.int k) => k
    | _ => miss
  .num (if ctx.maximize then -raw else raw)

/-- A missing weight scores `+inf` in both modes (negation flips the `-inf`
miss of the maximizing mode), so it never beats a finite edge. -/
theorem msfScore_missing (ctx : Ctx) (e : Nat) (hu : ctx.unit = false)
    (h : ctx.attr e = none ∨ ctx.attr e = some .other) : msfScore ctx e = .num MsfFlow.inf := by
  unfold msfScore
  rcases h with h | h <;> cases hm : ctx.maximize <;> simp [hu, h]

theorem msfScore_unit (ctx : Ctx) (e : Nat) (hu : ctx.unit = true) : msfScore ctx e = .num 1 := by
  simp [msfScore, hu]

theorem msfScore_max (ctx : Ctx) (e : Nat) (k : Int) (hu : ctx.unit = false) (hm : ctx.maximize = true)
    (h : ctx.attr e = some (.int k) ∨ ctx.attr e = some (.float k)) : msfScore ctx e = .num (-k) := by
  unfold msfScore; rcases h with h | h <;> simp [hu, hm, h]

/-- A finite (|k| < inf) edge beats a missing-weight edge under `keepMin`. -/
theorem missing_never_selected (k : Int) (e e' : Nat) (hk : k < MsfFlow.inf) :
    MsfFlow.keepMin ⟨.num MsfFlow.inf, e'⟩ ⟨.num k, e⟩ = ⟨.num k, e⟩ := by
  simp [MsfFlow.keepMin, MsfFlow.Score.lt, hk]

/-- `msf_scored_edge_index_op`: `none` = a null pointer (no write). -/
def indexOp (ctx : Option Ctx) (zNull : Bool) (j : Nat) : Option MsfFlow.SE :=
  if zNull then none else ctx.map fun c => ⟨msfScore c j, j⟩

def valueOp (ctx : Option Ctx) (zNull : Bool) (x : Option Nat) : Option MsfFlow.SE :=
  if zNull then none else
  match x, ctx with
  | some e, some c => some ⟨msfScore c e, e⟩
  | _, _ => none

theorem indexOp_spec (c : Ctx) (j : Nat) : indexOp (some c) false j = some ⟨msfScore c j, j⟩ := rfl
theorem valueOp_spec (c : Ctx) (e : Nat) : valueOp (some c) false (some e) = some ⟨msfScore c e, e⟩ := rfl
theorem ops_null_guard (c : Option Ctx) (j : Nat) (x : Option Nat) :
    indexOp c true j = none ∧ valueOp c true x = none ∧ indexOp none false j = none ∧
    valueOp c false none = none := by
  refine ⟨rfl, rfl, rfl, ?_⟩; cases c <;> rfl

/-- `msf_score_of`. -/
def scoreOf (x : Option MsfFlow.SE) : Option MsfFlow.Score := x.map (·.score)
theorem scoreOf_spec (s : MsfFlow.SE) : scoreOf (some s) = some s.score := rfl

/-- `node_value`. -/
def nodeValue (id : Nat) : Val := .node id
theorem nodeValue_spec (id : Nat) : nodeValue id = .node id := rfl

/-! ## Node collection -/

/-- `Graph::get_nodes(&∅, 0)` (graph.rs:2177-2198). -/
def activeNodes (nodeCount maxId : Nat) (deleted : Nat → Bool) : List Nat :=
  if nodeCount = 0 then [] else (List.range (maxId + 1)).filter (fun i => !deleted i)

theorem mem_activeNodes (nodeCount maxId : Nat) (deleted : Nat → Bool) (x : Nat) :
    x ∈ activeNodes nodeCount maxId deleted ↔ nodeCount ≠ 0 ∧ x ≤ maxId ∧ deleted x = false := by
  unfold activeNodes
  by_cases h : nodeCount = 0
  · simp [h]
  · simp [h, List.mem_range]; omega

/-- FxHashSet insertion order is irrelevant after `sort_unstable`; dedup by membership. -/
def insertSet (s : List Nat) (x : Nat) : List Nat := if x ∈ s then s else s ++ [x]

theorem mem_insertSet (s : List Nat) (x y : Nat) : y ∈ insertSet s x ↔ y ∈ s ∨ y = x := by
  unfold insertSet; split
  · rename_i hx; constructor
    · exact Or.inl
    · rintro (h | rfl); exact h; exact hx
  · simp

theorem nodup_insertSet (s : List Nat) (x : Nat) (h : s.Nodup) : (insertSet s x).Nodup := by
  unfold insertSet; split
  · exact h
  · rename_i hx
    rw [List.nodup_append]; refine ⟨h, by simp, ?_⟩
    intro a ha b hb; simp at hb; subst hb; intro he; subst he; exact hx ha

/-- `collect_node_ids`: `scan l` is `get_nodes({l}, 0)` (AXIOMATISED graph accessor). -/
def collectNodeIds (scan : String → List Nat) (all : List Nat) (labels : List String) : List Nat :=
  if labels = [] then all
  else ((labels.flatMap scan).foldl insertSet []).mergeSort (· ≤ ·)

theorem foldl_insertSet (l s : List Nat) (hs : s.Nodup) :
    (l.foldl insertSet s).Nodup ∧ ∀ y, y ∈ l.foldl insertSet s ↔ y ∈ s ∨ y ∈ l := by
  induction l generalizing s with
  | nil => simp [hs]
  | cons x t ih =>
    simp only [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih (insertSet s x) (nodup_insertSet s x hs)
    refine ⟨h1, fun y => ?_⟩
    rw [h2, mem_insertSet]; simp only [List.mem_cons]
    constructor
    · rintro ((h | h) | h)
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
    · rintro (h | h | h)
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr h)
      · exact Or.inr h

/-- The union over labels, sorted ascending, without duplicates. -/
theorem collectNodeIds_spec (scan : String → List Nat) (all : List Nat) (labels : List String)
    (hne : labels ≠ []) :
    let r := collectNodeIds scan all labels
    (∀ x, x ∈ r ↔ ∃ l ∈ labels, x ∈ scan l) ∧ r.Nodup ∧ r.Pairwise (· ≤ ·) := by
  simp only [collectNodeIds, hne, if_false]
  obtain ⟨hnd, hmem⟩ := foldl_insertSet (labels.flatMap scan) [] (by simp)
  refine ⟨fun x => ?_, ?_, ?_⟩
  · rw [List.mem_mergeSort, hmem]; simp
  · exact (List.mergeSort_perm _ _).nodup_iff.mpr hnd
  · have := List.pairwise_mergeSort (le := fun a b : Nat => decide (a ≤ b))
      (by intro a b c h1 h2; simp at *; omega) (by intro a b; simp; omega)
      ((labels.flatMap scan).foldl insertSet [])
    exact this.imp (by simp)

/-! ## LAGraph graph ownership -/

/-- What frees the adjacency handle along one procedure run. -/
structure Own where
  frees : Nat
  gA : Bool      -- `G->A` still points at the handle

/-- `create_lagraph_graph(adj, kind, borrowed)` given whether `LAGraph_New`
succeeded (LAGraph contract: on success it takes the handle into `G->A`; on
failure it has not taken it). `none` = `Err`. -/
def create (borrowed ok : Bool) : Option Own × Nat :=
  if ok then (some ⟨0, true⟩, 0) else (none, if borrowed then 0 else 1)

/-- `LAGraph_Delete` frees `G->A` if set. -/
def delete (o : Own) : Nat := o.frees + (if o.gA then 1 else 0)
def deleteBorrowed (o : Own) : Nat := delete { o with gA := false }
def deleteMaybe (o : Own) (borrowed : Bool) : Nat := if borrowed then deleteBorrowed o else delete o

/-- Whole lifecycle: create, (run), teardown, then the caller's `Matrix`
wrapper drops (it owns the handle iff it was lent). -/
def lifecycle (borrowed ok : Bool) : Nat :=
  let wrapper := if borrowed then 1 else 0
  match create borrowed ok with
  | (some o, f) => f + deleteMaybe o borrowed + wrapper
  | (none, f) => f + wrapper

/-- **Exactly one free** of the adjacency handle on every path. -/
theorem lifecycle_frees_once (borrowed ok : Bool) : lifecycle borrowed ok = 1 := by
  cases borrowed <;> cases ok <;> rfl

/-! ## `cmp_found_path` -/

/-- `cmp_found_path(a, b) == Less` is `Dfs.fpLt`; on finite weights it is a strict order. -/
theorem cmp_found_path_strict (a b c : Dfs.FP) :
    Dfs.fpLt a a = false ∧ (Dfs.fpLt a b = true → Dfs.fpLt b c = true → Dfs.fpLt a c = true) ∧
    (Dfs.fpLt a b = true → Dfs.fpLt b a = false) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [Dfs.fpLt]
  · intro h1 h2; simp [Dfs.fpLt] at *; omega
  · intro h; simp [Dfs.fpLt] at *; omega

/-- `parse_config`: no args or Null → empty map; a Map → itself; else error. -/
theorem parseConfig_spec (args : List Val) :
    (args = [] → Config.parseConfig args = .ok []) ∧
    (∀ rest, Config.parseConfig (.null :: rest) = .ok []) ∧
    (∀ m rest, Config.parseConfig (.map m :: rest) = .ok m) ∧
    (∀ s rest, Config.parseConfig (.str s :: rest) = .error "Invalid argument type: expected a map or null") :=
  ⟨fun h => by subst h; rfl, fun _ => rfl, fun _ _ => rfl, fun _ _ => rfl⟩

end AlgoUdf.AlgoGlue
