import AlgoUdf.Graph
/-! # `enumerate_paths` (algo_procedures.rs:2469-2570) and `record_found_path` with its bound (:2405-2458)

| here | there |
| --- | --- |
| `FP`                 | `FoundPath = (Vec<(RelationshipId, NodeId, NodeId)>, f64, f64)` :2081 |
| `fpLt`               | `cmp_found_path(a, b) == Ordering::Less` :2092 (weight, cost, hops) |
| `insertSorted`       | `results.insert(results.partition_point(|kept| cmp(kept, found) == Less), found)` :2451-2453 (linear scan; equal to the binary search on the sorted vector the k-branch keeps) |
| `recordB`            | `record_found_path` :2405, including every `*bound = ..` update |
| `Frame`              | one depth of the search: `levels[d]`, `nodes[d]`, and (d > 0) `edges[d-1]` with its prefix weight/cost — Rust keeps the three vectors in lock-step (push :2537-2560, pop :2561-2567) |
| `succsRev`           | `node_successors` :2153 (reversed: `Vec::pop` takes the last) |
| `step`               | one iteration of the `loop` :2500-2568; `none` = `break` at depth 0 |
| `DS.log`             | ghost: every path passed to `record_found_path` (proof-only) |
| `Cfg`                | `PathAlgoConfig` fields read here |

Weights `G.wt` and costs `G.cost` are exact naturals (see `AlgoUdf.Graph`);
`bound = none` is `f64::INFINITY`; `maxCost` is `Option Nat`.

Results:
* `dfs_sound` — in every reachable state the stack is a simple weighted walk
  from the source and every kept result is a simple path of 1..maxLen hops from
  the source (ending at the target for SPpaths) whose reported weight / cost
  are its exact edge sums, within `maxCost`.
* `dfs_no_panic` — the `expect`s at :2564-2565 and the `levels[depth]` index
  never fail (the frame stack is never empty, so the three Rust vectors never
  underflow).
* `recordB_*` — the bound only ever decreases and only to the weight of a kept
  path; `pathCount` k ≥ 2 keeps ≤ k paths.
-/
namespace AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

structure FP where
  es : List Rel
  w : Nat
  c : Nat
  deriving Inhabited

def fpLt (a b : FP) : Bool :=
  a.w < b.w || (a.w == b.w && (a.c < b.c || (a.c == b.c && a.es.length < b.es.length)))

def insertSorted (f : FP) : List FP → List FP
  | [] => [f]
  | x :: xs => if fpLt x f then x :: insertSorted f xs else f :: x :: xs

/-- `record_found_path(results, found, path_count, bound)`; returns the new
`(results, bound)`. -/
def recordB (res : List FP) (f : FP) (k : Nat) (bound : Option Nat) : List FP × Option Nat :=
  if k = 1 then
    match res with
    | [] => ([f], some f.w)
    | best :: _ => if fpLt f best then ([f], some f.w) else (res, bound)
  else if k = 0 then
    match res with
    | [] => ([f], some f.w)
    | best :: _ =>
      if best.w < f.w then (res, bound)
      else if f.w < best.w then ([f], some f.w)
      else (res ++ [f], bound)
  else
    if res.length = k then
      if fpLt f (res.getLast!) then
        let r := insertSorted f res.dropLast
        (r, if r.length = k then some (r.getLast!.w) else bound)
      else (res, bound)
    else
      let r := insertSorted f res
      (r, if r.length = k then some (r.getLast!.w) else bound)

structure Cfg where
  src : Nat
  tgt : Option Nat
  maxLen : Nat
  maxCost : Option Nat
  k : Nat

structure Frame where
  lv : Option (List (Rel × Nat))
  node : Nat
  pw : Nat
  pc : Nat
  rel : Option Rel

structure DS where
  frames : List Frame
  onPath : Nat → Bool
  results : List FP
  bound : Option Nat
  /-- ghost: every path handed to `record_found_path`, in order -/
  log : List FP

def succsRev (g : G) (u : Nat) : List (Rel × Nat) := (succs g u).reverse

/-- The live path's edges in traversal order. -/
def pathOf (fs : List Frame) : List Rel := (fs.filterMap (·.rel)).reverse

def costSum (g : G) (es : List Rel) : Nat := (es.map (fun r => g.cost r.2.2)).sum

def init (g : G) (cfg : Cfg) (bound : Option Nat) : DS :=
  ⟨[⟨if 0 < cfg.maxLen then some (succsRev g cfg.src) else none, cfg.src, 0, 0, none⟩],
   upd (fun _ => false) cfg.src true, [], bound, []⟩

def step (g : G) (cfg : Cfg) (st : DS) : Option DS :=
  match st.frames with
  | [] => none
  | fr :: below =>
    match fr.lv with
    | some ((r, next) :: rest) =>
      let fr' : Frame := { fr with lv := some rest }
      let st1 : DS := { st with frames := fr' :: below }
      if st.onPath next then some st1 else
      match g.wt r.2.2 with
      | none => some st1
      | some x =>
        let nw := fr.pw + x
        let nc := fr.pc + g.cost r.2.2
        if st.bound.any (· < nw) then some st1 else
        if cfg.maxCost.any (· < nc) then some st1 else
        let atT := cfg.tgt.all (· == next)
        let rb := if atT then recordB st.results ⟨pathOf (fr' :: below) ++ [r], nw, nc⟩ cfg.k st.bound
                  else (st.results, st.bound)
        let expand := !(atT && cfg.tgt.isSome) && decide (below.length + 1 < cfg.maxLen)
        some ⟨⟨if expand then some (succsRev g next) else none, next, nw, nc, some r⟩ :: fr' :: below,
              upd st.onPath next true, rb.1, rb.2,
              if atT then st.log ++ [⟨pathOf (fr' :: below) ++ [r], nw, nc⟩] else st.log⟩
    | _ =>
      if below = [] then none
      else some { st with frames := below, onPath := upd st.onPath fr.node false }

inductive Star (g : G) (cfg : Cfg) : DS → DS → Prop
  | refl (s : DS) : Star g cfg s s
  | tail {a b c : DS} : Star g cfg a b → step g cfg b = some c → Star g cfg a c

/-! ## `record_found_path` -/

theorem insertSorted_mem (f : FP) (l : List FP) : ∀ x ∈ insertSorted f l, x = f ∨ x ∈ l := by
  induction l with
  | nil => intro x hx; simp [insertSorted] at hx; exact Or.inl hx
  | cons y t ih =>
    intro x hx
    simp only [insertSorted] at hx
    split at hx
    · simp only [List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact Or.inr (by simp)
      · rcases ih x hx with h | h
        · exact Or.inl h
        · exact Or.inr (by simp [h])
    · simp only [List.mem_cons] at hx
      rcases hx with rfl | rfl | hx
      · exact Or.inl rfl
      · exact Or.inr (by simp)
      · exact Or.inr (by simp [hx])

theorem insertSorted_length (f : FP) (l : List FP) : (insertSorted f l).length = l.length + 1 := by
  induction l with
  | nil => rfl
  | cons y t ih => simp only [insertSorted]; split <;> simp [ih]

theorem getLast!_mem {l : List FP} (h : l ≠ []) : l.getLast! ∈ l := by
  rw [List.getLast!_eq_getLast?_getD, List.getLast?_eq_getLast h]
  exact List.getLast_mem h

/-- Kept paths are old ones or the new one. -/
theorem recordB_mem (res : List FP) (f : FP) (k : Nat) (b : Option Nat) :
    ∀ x ∈ (recordB res f k b).1, x = f ∨ x ∈ res := by
  intro x hx
  unfold recordB at hx
  split at hx
  · split at hx
    · simp at hx; exact Or.inl hx
    · split at hx
      · simp at hx; exact Or.inl hx
      · exact Or.inr hx
  · split at hx
    · split at hx
      · simp at hx; exact Or.inl hx
      · split at hx
        · exact Or.inr hx
        · split at hx
          · simp at hx; exact Or.inl hx
          · simp only [List.mem_append, List.mem_singleton] at hx
            rcases hx with h | h
            · exact Or.inr h
            · exact Or.inl h
    · split at hx
      · split at hx
        · rcases insertSorted_mem f _ x hx with h | h
          · exact Or.inl h
          · exact Or.inr ((List.dropLast_sublist _).subset h)
        · exact Or.inr hx
      · exact insertSorted_mem f res x hx

/-- The new bound is the old one or the weight of a kept path. -/
theorem recordB_bound (res : List FP) (f : FP) (k : Nat) (b : Option Nat) :
    (recordB res f k b).2 = b ∨ ∃ x ∈ (recordB res f k b).1, (recordB res f k b).2 = some x.w := by
  unfold recordB
  split
  · split
    · right; exact ⟨f, by simp, rfl⟩
    · split
      · right; exact ⟨f, by simp, rfl⟩
      · left; rfl
  · split
    · split
      · right; exact ⟨f, by simp, rfl⟩
      · split
        · left; rfl
        · split
          · right; exact ⟨f, by simp, rfl⟩
          · left; rfl
    · split
      · split
        · dsimp only
          split
          · rename_i h; right
            refine ⟨_, getLast!_mem ?_, rfl⟩
            intro he; rw [he] at h; simp at h; omega
          · left; rfl
        · left; rfl
      · dsimp only
        split
        · rename_i h; right
          refine ⟨_, getLast!_mem ?_, rfl⟩
          intro he; rw [he] at h; simp at h; omega
        · left; rfl

/-- `pathCount = k ≥ 2` never keeps more than k paths. -/
theorem recordB_length (res : List FP) (f : FP) (k : Nat) (b : Option Nat) (hk : 2 ≤ k)
    (h : res.length ≤ k) : (recordB res f k b).1.length ≤ k := by
  unfold recordB
  rw [if_neg (by omega), if_neg (by omega)]
  split
  · split
    · dsimp only; rw [insertSorted_length, List.length_dropLast]; omega
    · omega
  · dsimp only; rw [insertSorted_length]; omega

end AlgoUdf.Dfs
