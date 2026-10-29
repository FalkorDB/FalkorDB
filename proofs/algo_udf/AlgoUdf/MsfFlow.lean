/-! # algo.MSF edge selection and algo.maxFlow result assembly

| here | there (`graph/src/runtime/functions/algo_procedures.rs`) |
| --- | --- |
| `Score`               | f64 score from `msf_score` :144 (abstracted: `num k` finite/±inf ordered, `nan`) |
| `SE`, `keepMin`       | `ScoredEdge`, `msf_keep_min_score` :238 (binary op + monoid for `GrB_Matrix_reduce_Monoid`) |
| `identity`            | the monoid identity `{score: +inf, edge: u64::MAX}` :1552 |
| `componentKeys`       | MSF grouping :1888-1909 (component key = compact rep; isolated fallback keyed by *original* id) |
| `flowAlign`           | maxFlow identity-compaction :3069-3125 and flow read-back :3297-3330 |
| `superFresh`          | super source/sink ids `base_dim`, `base_dim + multi_srcs` :3078-3081 |
| `buildTbl`, `denseTable`, `compactOf`, `bsearch` | the original-id → compact-id map and `compact_of` :1442-1465 (#3586) |

GraphBLAS requires a monoid's operator to be associative and commutative with
an identity (GrB spec §3.5, `GrB_Monoid_new`); `keepMin` is that operator.
-/
namespace AlgoUdf.MsfFlow

/-- An f64 score as far as `<`/`==` can see it: ordered values (finite and the
infinities, encoded as integers with ±inf at the ends) and NaN. -/
inductive Score | num (k : Int) | nan deriving DecidableEq, Repr

def Score.lt : Score → Score → Bool
  | .num a, .num b => decide (a < b)
  | _, _ => false

def Score.eq : Score → Score → Bool
  | .num a, .num b => decide (a = b)
  | _, _ => false

structure SE where
  score : Score
  edge : Nat
  deriving DecidableEq, Repr

/-- `msf_keep_min_score(x, y)`: `y` if `y.score < x.score || (== && y.edge < x.edge)`, else `x`. -/
def keepMin (x y : SE) : SE :=
  if y.score.lt x.score || (y.score.eq x.score && decide (y.edge < x.edge)) then y else x

/-- `+inf` stands above every finite score; `u64::MAX` above every real edge id. -/
def inf : Int := 1000000000000000000000000
def identity : SE := ⟨.num inf, 2 ^ 64 - 1⟩

/-- Commutative on NaN-free inputs with distinct edge ids (the tie-break the
comment at :255 relies on). -/
theorem keepMin_comm (a b : Int) (e f : Nat) (hef : e ≠ f) :
    keepMin ⟨.num a, e⟩ ⟨.num b, f⟩ = keepMin ⟨.num b, f⟩ ⟨.num a, e⟩ := by
  simp only [keepMin, Score.lt, Score.eq, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]
  split <;> split <;> rename_i h1 h2 <;> simp only [SE.mk.injEq, Score.num.injEq] <;> omega

/-- The identity is neutral for every NaN-free scored edge below it. -/
theorem identity_neutral (a : Int) (e : Nat) (ha : a < inf) :
    keepMin identity ⟨.num a, e⟩ = ⟨.num a, e⟩ ∧ keepMin ⟨.num a, e⟩ identity = ⟨.num a, e⟩ := by
  simp [keepMin, identity, Score.lt, Score.eq, ha]
  omega

/-- SUSPICION (not reproduced live): a NaN weight breaks the monoid laws, so the
winner of a multi-edge pair depends on the order GraphBLAS worker threads
combine operands, and the identity swallows a NaN-scored edge. -/
theorem keepMin_nan_not_comm :
    keepMin ⟨.nan, 1⟩ ⟨.num 5, 2⟩ = ⟨.nan, 1⟩ ∧
    keepMin ⟨.num 5, 2⟩ ⟨.nan, 1⟩ = ⟨.num 5, 2⟩ ∧
    keepMin identity ⟨.nan, 1⟩ = identity := by decide

/-! ## MSF component grouping -/

/-- Keys used for the result rows: compact representative for every node in the
component vector, the *original id* for a node missing from it. -/
def componentKeys (ids : List Nat) (comp : Nat → Option Nat) : List (Nat × Nat) :=
  (List.range ids.length).map fun i =>
    match comp i with
    | some r => (r, ids.getD i 0)
    | none => (ids.getD i 0, ids.getD i 0)

/-- Under LAGraph_msf's contract (the component vector is dense: every compact
index has an entry) the original-id fallback is dead code. -/
theorem msf_dense_no_fallback (ids : List Nat) (comp : Nat → Option Nat)
    (hd : ∀ i, i < ids.length → ∃ r, comp i = some r) :
    componentKeys ids comp =
      (List.range ids.length).map fun i => ((comp i).getD 0, ids.getD i 0) := by
  simp only [componentKeys]
  apply List.map_congr_left
  intro i hi
  obtain ⟨r, hr⟩ := hd i (List.mem_range.mp hi)
  simp [hr]

/-- If the vector were ever sparse, the two key spaces collide: node 5 (compact
index 1, missing) is keyed 5 ... and so is any component whose compact rep is 5. -/
theorem msf_fallback_key_space_mixed :
    componentKeys [0, 5] (fun i => if i = 0 then some 0 else none) = [(0, 0), (5, 5)] := by
  decide

/-! ## MSF original-id → compact-id map (#3586, fixes #3444) -/

/-- The dense table `map[orig] = compact` filled by the loop at :1442-1445:
`buildTbl ids 0 (replicate (max_id + 1) none)` (`u64::MAX` = `none`). -/
def buildTbl : List Nat → Nat → List (Option Nat) → List (Option Nat)
  | [], _, t => t
  | o :: os, c, t => buildTbl os (c + 1) (t.set o (some c))

theorem buildTbl_length (ids : List Nat) (c : Nat) (t : List (Option Nat)) :
    (buildTbl ids c t).length = t.length := by
  induction ids generalizing c t with
  | nil => rfl
  | cons o os ih => simp [buildTbl, ih]

theorem buildTbl_get (ids : List Nat) (c : Nat) (t : List (Option Nat))
    (hlt : ∀ o ∈ ids, o < t.length) (hnd : ids.Nodup) (x : Nat) :
    (buildTbl ids c t)[x]? = if x ∈ ids then some (some (ids.idxOf x + c)) else t[x]? := by
  induction ids generalizing c t with
  | nil => simp [buildTbl]
  | cons o os ih =>
    have hnd' := List.nodup_cons.mp hnd
    have hlt' : ∀ y ∈ os, y < (t.set o (some c)).length := by
      intro y hy; simpa using hlt y (List.mem_cons_of_mem _ hy)
    simp only [buildTbl]
    rw [ih (c + 1) _ hlt' hnd'.2]
    by_cases hx : x = o
    · subst hx
      simp only [List.mem_cons_self, if_true, hnd'.1]
      simp [List.getElem?_set_self (hlt x (List.mem_cons_self ..))]
    · by_cases hm : x ∈ os
      · have hox : (o == x) = false := by simpa using Ne.symm hx
        simp [hx, hm, List.idxOf_cons, hox]; omega
      · simp [hx, hm, List.getElem?_set_ne, Ne.symm hx]

/-- `compact_of` (:1450-1455) on main: the table when `max_id < 4 n`, else binary search of
the sorted ids. `bsearch`: `slice::binary_search` on a sorted, duplicate-free list is
`Ok(i)` exactly when `ids[i] = x` (std; AXIOMATISED as this spec). -/
def bsearch (ids : List Nat) (x : Nat) : Option Nat := if x ∈ ids then some (ids.idxOf x) else none

/-- The dense table exists only when the id range is within 4x the active count. -/
def denseTable (ids : List Nat) (maxId : Nat) : Option (List (Option Nat)) :=
  if maxId < ids.length * 4 then some (buildTbl ids 0 (List.replicate (maxId + 1) none)) else none

def compactOf (ids : List Nat) (maxId : Nat) (x : Nat) : Option Nat :=
  match denseTable ids maxId with
  | some tbl => (tbl[x]?).bind id
  | none => bsearch ids x

/-- **#3444 fixed, correctness**: both branches answer the same: the compact id is the
position in `sorted_ids` (every id is `≤ max_id`, ids distinct). -/
theorem compactOf_eq (ids : List Nat) (maxId : Nat) (hmax : ∀ o ∈ ids, o ≤ maxId)
    (hnd : ids.Nodup) (x : Nat) : compactOf ids maxId x = bsearch ids x := by
  unfold compactOf
  cases hd : denseTable ids maxId with
  | none => rfl
  | some tbl =>
    unfold denseTable at hd
    split at hd
    · cases hd
      simp only
      rw [buildTbl_get ids 0 _ (by intro o ho; simpa [Nat.lt_succ_iff] using hmax o ho) hnd]
      unfold bsearch
      by_cases hx : x ∈ ids
      · simp [hx]
      · by_cases hle : x ≤ maxId
        · simp [hx, List.getElem?_replicate, hle, Nat.lt_succ_iff]
        · simp [hx, List.getElem?_replicate, hle, Nat.lt_succ_iff]
    · cases hd

/-- **#3444 fixed, memory**: when a table is built its length is at most `4 n + ... `: at
most 4 slots per active node (so at most 32 bytes/node); a sparse id space builds none. -/
theorem denseTable_len (ids : List Nat) (maxId : Nat) (tbl : List (Option Nat))
    (h : denseTable ids maxId = some tbl) : tbl.length ≤ ids.length * 4 := by
  unfold denseTable at h
  split at h
  · cases h; simp [buildTbl_length]; omega
  · cases h

/-- One surviving high id (`#3444`): ids `[0, 10^9]` gave a 10^9-entry table before; now
no table is built. -/
theorem sparse_builds_no_table : denseTable [0, 1000000000] 1000000000 = none := by decide

/-! ## maxFlow: flow read-back is aligned with edge metadata -/

/-- identity-compaction path: keep edges with positive capacity; `compact` and
`meta` are built in lock-step, then super edges are appended to `compact`. -/
def flowAlign (es : List (Nat × Nat × Nat × Int)) (supers : List (Nat × Nat)) :
    List (Nat × Nat) × List (Nat × Nat × Nat) :=
  let kept := es.filter (fun e => decide (0 < e.2.2.2))
  (kept.map (fun e => (e.1, e.2.1)) ++ supers, kept.map (fun e => (e.1, e.2.1, e.2.2.1)))

theorem flowAlign_aligned (es : List (Nat × Nat × Nat × Int)) (supers : List (Nat × Nat))
    (i : Nat) (hi : i < (flowAlign es supers).2.length) :
    (flowAlign es supers).1[i]? =
      ((flowAlign es supers).2[i]?).map (fun m => (m.1, m.2.1)) := by
  simp only [flowAlign, List.length_map] at hi ⊢
  rw [List.getElem?_append_left (by simpa using hi)]
  simp [List.getElem?_map]
  cases (List.filter (fun e => decide (0 < e.2.2.2)) es)[i]? <;> rfl

/-- Super source/sink ids never collide with a graph node (`base_dim = max_node_id + 1`). -/
theorem superFresh (maxId x : Nat) (ms : Bool) (hx : x ≤ maxId) :
    x ≠ maxId + 1 ∧ x ≠ maxId + 1 + (if ms then 1 else 0) := by
  constructor <;> (try split) <;> omega

end AlgoUdf.MsfFlow
