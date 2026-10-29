import AlgoUdf.Paths
/-! # SPpaths/SSpaths: `costProp` configured vs. absent on the edge (#3121) — gap closed 2026-10-20

`Paths.pathCostRust` / `dijkstraCostRust` take one `Option Int` per edge, where `none`
merges two different cases: *no `costProp` configured* and *`costProp` configured but
this edge lacks it*. C distinguishes them; `edge_numeric_attr` does not. This file keeps
the configuration separate.

| here | there |
| --- | --- |
| `edgeNumericAttr` | `edge_numeric_attr` `algo_procedures.rs:2104-2117` (`prop.map_or(default, …)`, one `default` for both cases) |
| `rustFastCost`    | `dijkstra_single_path`'s reported cost, default `0.0` (`:2360`) |
| `rustEnumCost`    | `bfs_find_bound` `:2451` and `enumerate_paths` `:2573`, default `1.0` |
| `cResolve`        | `GraphContext_GetAttributeID` (`proc_sp_paths.c:350`): an unknown name is `ATTRIBUTE_ID_NONE` |
| `cFastCost`       | C `_sum_path_cost` as run live (`c.so`): `ATTRIBUTE_ID_NONE` → 0, else missing → `SI_LongVal(1)` |
| `cEnumCost`       | C DFS `GraphEntity_GetNumericPropertyOrDefault(.., cost_prop, SI_LongVal(1))` (`proc_sp_paths.c:633,670`) |

`costProp` is `none` (not configured), `some true` (configured, the attribute name
exists in the graph) or `some false` (configured, a name no entity has).
Live (Rust main 9d3869aa8 vs C), `A-[:E {weight:1,cost:2}]->B-[:E2 {weight:1}]->C`:

| config | Rust | C |
| --- | --- | --- |
| `costProp:'cost'` (fast path) | 2 | **3** |
| no `costProp` (fast path) | 0 | 0 |
| `costProp:'nosuch'` (fast path) | 0 | 0 |
| `costProp:'cost', maxLen:5` (enumeration) | 3 | 3 |

(origin/master's newer `_sum_path_cost` charges 1 even with no `costProp`; the
configured-but-missing case, which is what the property below states, is 1 in both C
versions.)
-/
namespace AlgoUdf.CostProp

/-- `edge_numeric_attr(g, e, prop, default)`: no prop → default; prop set → the numeric
attribute, else default. -/
def edgeNumericAttr (prop : Option Bool) (attr : Option Int) (dflt : Int) : Int :=
  match prop with
  | none => dflt
  | some _ => attr.getD dflt

def sumCost (f : Option Int → Int) (attrs : List (Option Int)) : Int :=
  (attrs.map f).foldl (· + ·) 0

def rustFastCost (prop : Option Bool) (attrs : List (Option Int)) : Int :=
  sumCost (edgeNumericAttr prop · 0) attrs
def rustEnumCost (prop : Option Bool) (attrs : List (Option Int)) : Int :=
  sumCost (edgeNumericAttr prop · 1) attrs

/-- `GraphContext_GetAttributeID`: a configured name the graph has never seen resolves to
`ATTRIBUTE_ID_NONE`, i.e. behaves as not configured. -/
def cResolve : Option Bool → Option Unit
  | some true => some ()
  | _ => none

def cFastCost (prop : Option Bool) (attrs : List (Option Int)) : Int :=
  match cResolve prop with
  | none => 0
  | some _ => sumCost (·.getD 1) attrs
def cEnumCost (_prop : Option Bool) (attrs : List (Option Int)) : Int :=
  sumCost (·.getD 1) attrs

/-- **A configured `costProp` that an edge lacks costs 1** (C parity, both paths): when
`costProp` names an existing attribute, the reported path cost is the sum over edges of
the attribute, or 1 where the edge does not carry it. -/
def MissingCostIsOne (cost : Option Bool → List (Option Int) → Int) : Prop :=
  ∀ attrs, cost (some true) attrs = sumCost (·.getD 1) attrs

theorem c_fast_ok : MissingCostIsOne cFastCost := fun _ => rfl
theorem c_enum_ok : MissingCostIsOne cEnumCost := fun _ => rfl
/-- The enumeration path (#3113) satisfies it. -/
theorem rust_enum_ok : MissingCostIsOne rustEnumCost := fun _ => rfl

/-- The existing `Paths` models are the configured case of these. -/
theorem enum_is_paths (attrs : List (Option Int)) :
    rustEnumCost (some true) attrs = Paths.pathCostRust attrs := rfl
theorem fast_is_paths (attrs : List (Option Int)) :
    rustFastCost (some true) attrs = Paths.dijkstraCostRust attrs := rfl

/-- **BUG #3121**: on the Dijkstra fast path (`pathCount 1`, no `maxCost`, unbounded
`maxLen`), a configured `costProp` missing on an edge contributes 0. Live repro: path cost
2 on Rust (2 + 0), 3 on C (2 + 1). -/
theorem bug3121_fastpath_missing_cost_zero :
    rustFastCost (some true) [some 2, none] = 2 ∧ cFastCost (some true) [some 2, none] = 3 ∧
      ¬ MissingCostIsOne rustFastCost := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h [some 2, none]
  revert this; decide

/-- The other two fast-path rows of the live table agree (0 = 0): not configured, and a
configured name the graph does not know. -/
theorem fast_unconfigured_agree :
    rustFastCost none [none, none] = cFastCost none [none, none] ∧
    rustFastCost (some false) [none, none] = cFastCost (some false) [none, none] := by decide

end AlgoUdf.CostProp
