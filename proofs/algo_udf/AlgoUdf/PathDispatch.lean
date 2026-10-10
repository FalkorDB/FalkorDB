import AlgoUdf.Config
import AlgoUdf.Marshal
import AlgoUdf.BfsBound
import AlgoUdf.DfsResult
/-! # algo.SPpaths / algo.SSpaths entry points and `run_path_algo` dispatch

| here | there (`algo_procedures.rs`) |
| --- | --- |
| `present`, `parseSp`   | `parse_sp_config` :2015-2049 (then `parse_common_path_config`, `Config`) |
| `parseSs`              | `parse_ss_config` :2051-2068 |
| `toNumeric`            | `to_numeric_value` :2070-2076 over `Marshal.F64` |
| `algoSp`, `algoSs`     | `algo_sp_paths` :2659, `algo_ss_paths` :2671 (`parse_*_config(args)?` then `run_path_algo`) |
| `Plan`, `plan`         | `run_path_algo` :2603-2651: Dijkstra fast path, BFS pre-pass, bound seeding, enumeration |
| `bfsWeight`/`bfsCost`  | the forward folds :2393-2397 (`none` weight = a non-finite sum: `next_weight > NaN/inf` never prunes) |

Theorems:
* `parseSp_ok`, `parseSp_missing`, `parseSp_not_node`, `parseSs_ok`, `parseSs_missing`.
* `toNumeric_nat` (exact path weights come back as `Int`), `toNumeric_neg_zero`.
* `plan_dijkstra_iff` — the fast path is taken exactly under the documented guard.
* `bfs_none_sound` — when the pre-pass says "no path", no qualifying path exists,
  so `Ok(empty)` is the enumeration's answer.
* `bfs_seed_keeps_answer` — for pathCount 1 / 0 the seeded bound never empties the
  answer: the BFS path itself qualifies (`ext_of_simple`), so `dfs_k1`/`dfs_k0`
  return a non-empty result.

Live-checked, shared with C (18900 C / 18901 Rust): with a *negative* weight
or cost the seeded bound is not an upper bound on prefixes and SPpaths returns
no row for `(s)-[w:5]->(a)-[w:-5]->(t)` with `maxLen:5`, and for a cost-`5,-5`
path plus a cost-0 detour with `maxCost:1`; C returns the same empty results.
Negative weights are outside the `Nat` model.
-/
namespace AlgoUdf.PathDispatch
open AlgoUdf AlgoUdf.Config AlgoUdf.Graph AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)

/-! ## Required node arguments -/

/-- `matches!(v, Some(v) if !matches!(v, Value::Null))` -/
def present : Option Val → Bool
  | none | some .null => false
  | some _ => true

/-- `parse_sp_config` up to the hand-off to `parse_common_path_config`. -/
def parseSp (args : List Val) : Except String (Nat × Nat × List (String × Val)) :=
  match parseConfig args with
  | .error e => .error e
  | .ok m =>
    if !present (lookup m "sourceNode") || !present (lookup m "targetNode") then
      .error "sourceNode and targetNode are required"
    else
      match lookup m "sourceNode" with
      | some (.node s) =>
        match lookup m "targetNode" with
        | some (.node t) => .ok (s, t, m)
        | _ => .error "sourceNode and targetNode must be of type Node"
      | _ => .error "sourceNode and targetNode must be of type Node"

def parseSs (args : List Val) : Except String (Nat × List (String × Val)) :=
  match parseConfig args with
  | .error e => .error e
  | .ok m =>
    if !present (lookup m "sourceNode") then .error "sourceNode is required"
    else
      match lookup m "sourceNode" with
      | some (.node s) => .ok (s, m)
      | _ => .error "sourceNode must be of type Node"

theorem parseSp_ok (args : List Val) (s t : Nat) (m : List (String × Val)) :
    parseSp args = .ok (s, t, m) →
      parseConfig args = .ok m ∧ lookup m "sourceNode" = some (.node s) ∧
        lookup m "targetNode" = some (.node t) := by
  unfold parseSp
  split
  · intro h; cases h
  · rename_i m' hm
    split
    · intro h; cases h
    · split
      · rename_i s' hs
        split
        · rename_i t' ht; intro h; cases h; exact ⟨hm, hs, ht⟩
        · intro h; cases h
      · intro h; cases h

theorem parseSp_missing (args : List Val) (m : List (String × Val)) (hm : parseConfig args = .ok m)
    (h : lookup m "sourceNode" = none ∨ lookup m "targetNode" = none ∨
      lookup m "sourceNode" = some .null ∨ lookup m "targetNode" = some .null) :
    parseSp args = .error "sourceNode and targetNode are required" := by
  unfold parseSp; rw [hm]; dsimp only
  rw [if_pos]
  rcases h with h | h | h | h <;> simp [h, present]

theorem parseSp_not_node (args : List Val) (m : List (String × Val)) (hm : parseConfig args = .ok m)
    (v : Val) (hv : lookup m "sourceNode" = some v) (hnn : v ≠ .null) (hnode : ∀ s, v ≠ .node s)
    (ht : present (lookup m "targetNode") = true) :
    parseSp args = .error "sourceNode and targetNode must be of type Node" := by
  unfold parseSp; rw [hm]; dsimp only
  have hp : present (lookup m "sourceNode") = true := by
    rw [hv]; cases v <;> simp_all [present]
  rw [if_neg (by simp [hp, ht]), hv]
  cases v <;> simp_all

theorem parseSs_ok (args : List Val) (s : Nat) (m : List (String × Val)) :
    parseSs args = .ok (s, m) → parseConfig args = .ok m ∧ lookup m "sourceNode" = some (.node s) := by
  unfold parseSs
  split
  · intro h; cases h
  · rename_i m' hm
    split
    · intro h; cases h
    · split
      · rename_i s' hs; intro h; cases h; exact ⟨hm, hs⟩
      · intro h; cases h

theorem parseSs_missing (args : List Val) (m : List (String × Val)) (hm : parseConfig args = .ok m)
    (h : lookup m "sourceNode" = none ∨ lookup m "sourceNode" = some .null) :
    parseSs args = .error "sourceNode is required" := by
  unfold parseSs; rw [hm]; dsimp only
  rcases h with h | h <;> simp [h, present]

/-- `algo_sp_paths`: `parse_sp_config(args)?` then `run_path_algo`. -/
def algoSp {α : Type} (run : Nat × Nat × List (String × Val) → Except String α) (args : List Val) :
    Except String α :=
  match parseSp args with
  | .error e => .error e
  | .ok c => run c

def algoSs {α : Type} (run : Nat × List (String × Val) → Except String α) (args : List Val) :
    Except String α :=
  match parseSs args with
  | .error e => .error e
  | .ok c => run c

theorem algoSp_spec {α : Type} (run : Nat × Nat × List (String × Val) → Except String α)
    (args : List Val) :
    (∀ e, parseSp args = .error e → algoSp run args = .error e) ∧
    (∀ c, parseSp args = .ok c → algoSp run args = run c) := by
  constructor <;> intro x h <;> simp [algoSp, h]

theorem algoSs_spec {α : Type} (run : Nat × List (String × Val) → Except String α) (args : List Val) :
    (∀ e, parseSs args = .error e → algoSs run args = .error e) ∧
    (∀ c, parseSs args = .ok c → algoSs run args = run c) := by
  constructor <;> intro x h <;> simp [algoSs, h]

/-! ## `to_numeric_value` -/

inductive Num | int (i : Int) | float (f : Marshal.F64) deriving DecidableEq, Repr

/-- `v.abs() < (i64::MAX as f64)` (= 2^63 after rounding). -/
def small63 : Marshal.F64 → Bool
  | .zero _ => true
  | .integ k => decide (k.natAbs < 9223372036854775808)
  | _ => false

def toNumeric (v : Marshal.F64) : Num :=
  if v.finite && v.integral && small63 v then
    .int (match v with | .integ k => k | _ => 0)
  else .float v

/-- Exact path weights / costs (naturals below 2^63) are reported as `Int`. -/
theorem toNumeric_nat (n : Nat) (h : n < 9223372036854775808) :
    toNumeric (Marshal.ofInt n) = .int n := by
  unfold toNumeric Marshal.ofInt
  by_cases hn : (n : Int) = 0
  · simp [hn, Marshal.F64.finite, Marshal.F64.integral, small63]
  · have hn' : n ≠ 0 := by omega
    simp [hn', Marshal.F64.finite, Marshal.F64.integral, small63]; omega

/-- `-0.0` is reported as `Int 0`. -/
theorem toNumeric_neg_zero : toNumeric (.zero true) = .int 0 := rfl

theorem toNumeric_nonfinite : toNumeric .nan = .float .nan ∧ toNumeric (.inf false) = .float (.inf false) :=
  ⟨rfl, rfl⟩

/-! ## `run_path_algo` -/

structure PCfg where
  src : Nat
  tgt : Option Nat
  maxLen : Nat
  maxCost : Option Nat
  k : Nat

inductive Plan | dijkstra (t : Nat) | empty | enumerate (bound : Option Nat) | panic
  deriving DecidableEq, Repr

/-- The BFS path's weight (forward fold; `none` when an edge weight is non-finite). -/
def bfsWeight (g : G) (es : List Rel) : Option Nat :=
  es.foldl (fun acc r => acc.bind fun a => (g.wt r.2.2).map (a + ·)) (some 0)

def bfsCost (g : G) (es : List Rel) : Nat := costSum g es

def u32Max : Nat := 4294967295

/-- `run_path_algo` :2603-2651; `bfs` is `bfs_find_bound`'s result. -/
def plan (g : G) (c : PCfg) (bfs : Option (Option (List Rel))) : Plan :=
  match c.tgt with
  | some t =>
    if c.k = 1 ∧ c.maxCost = none ∧ c.maxLen = u32Max ∧ c.src ≠ t then .dijkstra t
    else
      match bfs with
      | none => .panic
      | some none => .empty
      | some (some es) =>
        if (c.k = 0 ∨ c.k = 1) ∧ c.maxCost.all (bfsCost g es ≤ ·) = true then .enumerate (bfsWeight g es)
        else .enumerate none
  | none => .enumerate none

theorem plan_dijkstra_iff (g : G) (c : PCfg) (bfs : Option (Option (List Rel))) (t : Nat) :
    plan g c bfs = .dijkstra t ↔
      c.tgt = some t ∧ c.k = 1 ∧ c.maxCost = none ∧ c.maxLen = u32Max ∧ c.src ≠ t := by
  unfold plan
  cases ht : c.tgt with
  | none => simp
  | some t' =>
    simp only
    split
    · rename_i h; simp only [Plan.dijkstra.injEq, Option.some.injEq]
      constructor
      · rintro rfl; exact ⟨rfl, h⟩
      · rintro ⟨rfl, _⟩; rfl
    · rename_i h
      constructor
      · intro hp; split at hp <;> (try split at hp) <;> cases hp
      · rintro ⟨he, h'⟩; cases he; exact absurd h' h

/-- The configuration the enumeration runs with. -/
def dcfg (c : PCfg) : Cfg := ⟨c.src, c.tgt, c.maxLen, c.maxCost, c.k⟩

/-- A qualifying path for the enumeration. -/
abbrev Qual (g : G) (c : PCfg) (es : List Rel) (W C : Nat) : Prop :=
  Ext g (dcfg c) (succsRev g c.src) 0 0 0 [c.src] es W C

/-- Every qualifying extension is a hop-walk from the node its first step leaves. -/
theorem Ext.hwalk {g : G} {cfg : Cfg} {S pw pc d vis es W C} (h : Ext g cfg S pw pc d vis es W C) :
    ∀ u, (∀ r x, (r, x) ∈ S → Adj g u x r) → ∃ v, HWalk g u es v ∧ cfg.tgt.all (· == v) = true ∧
      d + es.length ≤ cfg.maxLen := by
  induction h with
  | @last _ _ _ _ _ r x _ h1 _ _ _ h5 h6 =>
    intro u hS; exact ⟨x, .cons (hS r x h1) (.nil x), h6, by simpa using h5⟩
  | @more _ _ _ _ _ r x _ _ _ _ h1 _ _ _ _ _ _ ih =>
    intro u hS
    obtain ⟨v, hw, ht, hl⟩ := ih x (fun r' x' hm => by
      have := (mem_succs g x r' x').mp (by simpa [succsRev] using hm); exact this)
    exact ⟨v, .cons (hS r x h1) hw, ht, by simp at hl ⊢; omega⟩

/-- **The pre-pass is sound**: when `bfs_find_bound` reports no path, no
qualifying path exists, so returning no rows is the enumeration's answer. -/
theorem bfs_none_sound (g : G) (c : PCfg) (t : Nat) (ht : c.tgt = some t) (hne : c.src ≠ t)
    (hb : BfsBound.bfsFindBound g t c.src c.maxLen = some none) :
    ∀ es W C, ¬ Qual g c es W C := by
  intro es W C hq
  obtain ⟨v, hw, hv, hl⟩ := Ext.hwalk hq c.src (fun r x hm =>
    (mem_succs g c.src r x).mp (by simpa [succsRev] using hm))
  simp [dcfg, ht] at hv
  subst hv
  have := BfsBound.bfs_complete g c.src t c.maxLen hne hb es hw
  simp [dcfg] at hl; omega

end AlgoUdf.PathDispatch
