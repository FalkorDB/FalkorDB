/-!
# `shortestPath`: `NeighborIter` and the bidirectional BFS (runtime/eval.rs:139-262, :1315-1527)

Matrix rows are abstract (`Nat → List Nat`); the GraphBLAS reads are the FFI boundary and
enter only through their row contents. The BFS is modelled level by level, line by line
(parent maps as association lists — `FxHashMap` insert-if-absent; the frontier swap;
the level-synchronised stop). The `while` loop is given fuel; the theorems are about every
result it can return.

Main theorem `bfs_sound`: whenever the BFS returns a path, it starts at `src`, ends at `dst`,
and every consecutive pair is joined by an edge the search followed; in particular the
`expect("BFS parent chain broken")` calls never fire.
-/
namespace FalkorExpr.BFS

/-! ## `NeighborIter` -/

/-- The graph's matrices as rows. `adj` is the adjacency matrix, `rel t` the matrix of
relationship type `t` (if the type exists), `relT t` its transpose. -/
structure G where
  adj : Nat → List Nat
  types : List String
  rel : String → Option (Nat → List Nat)
  relT : String → Option (Nat → List Nat)
  /-- `adjacency_matrix` is the pair-level union of the relationship matrices. -/
  adj_union : ∀ n x, x ∈ adj n ↔ ∃ t ∈ types, ∃ M, rel t = some M ∧ x ∈ M n
  /-- `matrix_t()` is the transpose. -/
  transpose : ∀ t M MT, rel t = some M → relT t = some MT → ∀ n x, x ∈ MT n ↔ n ∈ M x
  relT_some : ∀ t, (rel t).isSome = (relT t).isSome

structure NI where
  fwd : List (Nat → List Nat)
  bwd : List (Nat → List Nat)
  dedup : Bool

/-- `NeighborIter::new` (eval.rs:153). -/
def NI.new (g : G) (relTypes : List String) (directed : Bool) : NI :=
  let (fwd, bwd) :=
    if relTypes = [] then
      ([g.adj], if directed then [] else g.types.filterMap g.relT)
    else
      (relTypes.filterMap g.rel, if directed then [] else relTypes.filterMap g.relT)
  ⟨fwd, bwd, fwd.length + bwd.length > 1⟩

/-- `NeighborIter::new_reversed` (eval.rs:193). -/
def NI.newReversed (g : G) (relTypes : List String) : NI :=
  let fwd := (if relTypes = [] then g.types else relTypes).filterMap g.relT
  ⟨fwd, [], fwd.length > 1⟩

/-- `neighbors` (eval.rs:221): concatenate the rows; `sort_unstable` + `dedup` when several
iterators may repeat a neighbour. `sortDedup` is any membership-preserving normaliser. -/
def NI.neighbors (sortDedup : List Nat → List Nat) (ni : NI) (n : Nat) : List Nat :=
  let buf := (ni.fwd.map (· n)).flatten ++ (ni.bwd.map (· n)).flatten
  if ni.dedup then sortDedup buf else buf

theorem neighbors_mem (sd : List Nat → List Nat) (hsd : ∀ l x, x ∈ sd l ↔ x ∈ l) (ni : NI) (n x : Nat) :
    x ∈ ni.neighbors sd n ↔ (∃ M ∈ ni.fwd, x ∈ M n) ∨ (∃ M ∈ ni.bwd, x ∈ M n) := by
  have key : ∀ L : List (Nat → List Nat),
      (∃ l, (∃ a, a ∈ L ∧ a n = l) ∧ x ∈ l) ↔ ∃ M, M ∈ L ∧ x ∈ M n :=
    fun L => ⟨fun ⟨_, ⟨a, ha, e⟩, hx⟩ => ⟨a, ha, e ▸ hx⟩, fun ⟨a, ha, hx⟩ => ⟨_, ⟨a, ha, rfl⟩, hx⟩⟩
  unfold NI.neighbors
  split <;> simp only [hsd, List.mem_append, List.mem_flatten, List.mem_map] <;> rw [key, key]

/-- Directed, typed: the neighbours are exactly the out-neighbours over the given types. -/
theorem new_directed_typed (sd : List Nat → List Nat) (hsd : ∀ l x, x ∈ sd l ↔ x ∈ l) (g : G)
    (ts : List String) (h : ts ≠ []) (n x : Nat) :
    x ∈ (NI.new g ts true).neighbors sd n ↔ ∃ t ∈ ts, ∃ M, g.rel t = some M ∧ x ∈ M n := by
  have hf : (NI.new g ts true).fwd = ts.filterMap g.rel := by simp [NI.new, h]
  have hb : (NI.new g ts true).bwd = [] := by simp [NI.new, h]
  rw [neighbors_mem sd hsd, hf, hb]
  simp only [List.mem_filterMap, List.not_mem_nil, false_and, exists_false, or_false]
  constructor
  · rintro ⟨M, ⟨t, ht, hM⟩, hx⟩; exact ⟨t, ht, M, hM, hx⟩
  · rintro ⟨t, ht, M, hM, hx⟩; exact ⟨M, ⟨t, ht, hM⟩, hx⟩

/-- Directed, untyped: the adjacency matrix row, i.e. out-neighbours over every type. -/
theorem new_directed_untyped (sd : List Nat → List Nat) (hsd : ∀ l x, x ∈ sd l ↔ x ∈ l) (g : G) (n x : Nat) :
    x ∈ (NI.new g [] true).neighbors sd n ↔ ∃ t ∈ g.types, ∃ M, g.rel t = some M ∧ x ∈ M n := by
  have hf : (NI.new g [] true).fwd = [g.adj] := by simp [NI.new]
  have hb : (NI.new g [] true).bwd = [] := by simp [NI.new]
  rw [neighbors_mem sd hsd, ← g.adj_union, hf, hb]
  simp

/-- `new_reversed`, typed: exactly the in-neighbours (predecessors) over the given types. -/
theorem newReversed_typed (sd : List Nat → List Nat) (hsd : ∀ l x, x ∈ sd l ↔ x ∈ l) (g : G)
    (ts : List String) (h : ts ≠ []) (n x : Nat) :
    x ∈ (NI.newReversed g ts).neighbors sd n ↔ ∃ t ∈ ts, ∃ M, g.rel t = some M ∧ n ∈ M x := by
  have hf : (NI.newReversed g ts).fwd = ts.filterMap g.relT := by simp [NI.newReversed, h]
  have hb : (NI.newReversed g ts).bwd = [] := rfl
  rw [neighbors_mem sd hsd, hf, hb]
  simp only [List.mem_filterMap, List.not_mem_nil, false_and, exists_false, or_false]
  constructor
  · rintro ⟨MT, ⟨t, ht, hMT⟩, hx⟩
    have hs := g.relT_some t
    rw [hMT] at hs
    cases hM : g.rel t with
    | none => simp [hM] at hs
    | some M => exact ⟨t, ht, M, hM, (g.transpose t M MT hM hMT n x).mp hx⟩
  · rintro ⟨t, ht, M, hM, hx⟩
    have hs := g.relT_some t
    rw [hM] at hs
    cases hMT : g.relT t with
    | none => simp [hMT] at hs
    | some MT => exact ⟨MT, ⟨t, ht, hMT⟩, (g.transpose t M MT hM hMT n x).mpr hx⟩

/-! ## `bfs_shortest_path` (eval.rs:1400) -/

/-- Parent map: `key ↦ (parent, depth)`; inserted only when absent. -/
abbrev PMap := List (Nat × Nat × Nat)

def look (m : PMap) (x : Nat) : Option (Nat × Nat) := (m.find? (·.1 == x)).map (·.2)

/-- `if meet.is_none_or(|(best_od, _)| od < best_od) { meet = Some((od, nb)) }` -/
def bestMeet (meet : Option (Nat × Nat)) (od nb : Nat) : Option (Nat × Nat) :=
  match meet with
  | none => some (od, nb)
  | some (bod, bn) => if od < bod then some (od, nb) else some (bod, bn)

theorem bestMeet_cases (meet : Option (Nat × Nat)) (od nb o x : Nat)
    (h : bestMeet meet od nb = some (o, x)) : (o = od ∧ x = nb) ∨ meet = some (o, x) := by
  unfold bestMeet at h
  split at h
  · cases h; exact Or.inl ⟨rfl, rfl⟩
  · split at h <;> cases h
    · exact Or.inl ⟨rfl, rfl⟩
    · exact Or.inr rfl

/-- Expanding one frontier node: the inner `for &nb in nbrs.neighbors(cur)` loop. -/
def scan (other : PMap) (cur depth : Nat) :
    List Nat → PMap × List Nat × Option (Nat × Nat) → PMap × List Nat × Option (Nat × Nat)
  | [], st => st
  | nb :: nbs, (own, next, meet) =>
    if (look own nb).isSome then scan other cur depth nbs (own, next, meet)
    else
      let own := (nb, cur, depth) :: own
      match look other nb with
      | some (_, od) => scan other cur depth nbs (own, next, bestMeet meet od nb)
      | none => scan other cur depth nbs (own, next ++ [nb], meet)

/-- One level: `for &cur in front`. -/
def level (nbrs : Nat → List Nat) (other : PMap) (depth : Nat) :
    List Nat → PMap × List Nat × Option (Nat × Nat) → PMap × List Nat × Option (Nat × Nat)
  | [], st => st
  | cur :: cs, st => level nbrs other depth cs (scan other cur depth (nbrs cur) st)

structure St where
  fmap : PMap
  bmap : PMap
  ffront : List Nat
  bfront : List Nat
  df : Nat
  db : Nat

/-- The `while meet.is_none()` loop; `none` = `return Value::Null`. -/
def search (fN bN : Nat → List Nat) (maxLevel : Nat) : Nat → St → Option (St × Nat)
  | 0, _ => none
  | fuel + 1, s =>
    if s.ffront.isEmpty || s.bfront.isEmpty || s.df + s.db ≥ maxLevel then none
    else if s.ffront.length ≤ s.bfront.length then
      let (own, next, meet) := level fN s.bmap (s.df + 1) s.ffront (s.fmap, [], none)
      let s' := { s with fmap := own, ffront := next, df := s.df + 1 }
      match meet with
      | some (_, m) => some (s', m)
      | none => search fN bN maxLevel fuel s'
    else
      let (own, next, meet) := level bN s.fmap (s.db + 1) s.bfront (s.bmap, [], none)
      let s' := { s with bmap := own, bfront := next, db := s.db + 1 }
      match meet with
      | some (_, m) => some (s', m)
      | none => search fN bN maxLevel fuel s'

/-- Follow parents from `x` until `root` (`while cur != root { cur = map[cur].0 }`), giving
the visited nodes in order `x, parent(x), …, root`; `none` = the `expect` panic. -/
def chain (m : PMap) (root : Nat) : Nat → Nat → Option (List Nat)
  | 0, _ => none
  | fuel + 1, x =>
    if x = root then some [x]
    else match look m x with
      | some (p, _) => (chain m root fuel p).map (x :: ·)
      | none => none

/-- Fuel for the parent walk: the deepest entry (the Rust `while` needs no fuel; depths
strictly decrease along parents, so this many steps always reach the root). -/
def maxDepth (m : PMap) : Nat := (m.map (·.2.2)).foldl max 0

/-- The node sequence `bfs_shortest_path` builds before the min-hops check. -/
def bfsNodes (fN bN : Nat → List Nat) (maxLevel fuel : Nat) (src dst : Nat) : Option (List Nat) :=
  if src = dst then none else
  match search fN bN maxLevel fuel ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ with
  | none => none
  | some (s, m) => do
    let fwd ← chain s.fmap src (maxDepth s.fmap + 1) m
    let bwd ← chain s.bmap dst (maxDepth s.bmap + 1) m
    some (fwd.reverse ++ bwd.tail)

/-! ### Invariants -/

/-- Every key other than the root has a parent it was reached from by `R`, at a smaller
depth, and that parent is itself in the map. -/
def Inv (m : PMap) (root : Nat) (R : Nat → Nat → Prop) : Prop :=
  look m root = some (root, 0) ∧
  ∀ x p d, look m x = some (p, d) → x = root ∨ (R p x ∧ ∃ p' d', look m p = some (p', d') ∧ d' < d)

theorem look_cons (k p d x : Nat) (m : PMap) :
    look ((k, p, d) :: m) x = if k = x then some (p, d) else look m x := by
  by_cases h : k = x
  · subst h; simp [look]
  · have hb : (k == x) = false := by simp [h]
    simp [look, List.find?, hb, h]

theorem insert_pres (own : PMap) (nb cur depth x : Nat) (v : Nat × Nat) (hnb : look own nb = none)
    (hx : look own x = some v) : look ((nb, cur, depth) :: own) x = some v := by
  rw [look_cons]
  have : nb ≠ x := by intro e; subst e; rw [hx] at hnb; cases hnb
  simp [this, hx]

theorem insert_inv (R : Nat → Nat → Prop) (root : Nat) (own : PMap) (nb cur depth : Nat)
    (hI : Inv own root R) (hnb : look own nb = none) (hr : R cur nb)
    (hc : ∃ p d, look own cur = some (p, d) ∧ d < depth) : Inv ((nb, cur, depth) :: own) root R := by
  refine ⟨insert_pres own nb cur depth root _ hnb hI.1, ?_⟩
  intro x p d hx
  by_cases e : x = nb
  · subst e
    rw [look_cons] at hx; simp at hx; obtain ⟨rfl, rfl⟩ := hx
    obtain ⟨p', d', hp, hd⟩ := hc
    exact Or.inr ⟨hr, p', d', insert_pres own x cur depth cur (p', d') hnb hp, hd⟩
  · rw [look_cons] at hx
    have hne : nb ≠ x := Ne.symm e
    simp only [hne, ite_false] at hx
    rcases hI.2 x p d hx with h | ⟨hr', p', d', hp, hd⟩
    · exact Or.inl h
    · exact Or.inr ⟨hr', p', d', insert_pres own nb cur depth p (p', d') hnb hp, hd⟩

/-- Frontier nodes are in the map at depth < the depth being assigned. -/
def FInv (m : PMap) (front : List Nat) (depth : Nat) : Prop :=
  ∀ c ∈ front, ∃ p d, look m c = some (p, d) ∧ d < depth

def MInv (own other : PMap) (meet : Option (Nat × Nat)) : Prop :=
  ∀ od x, meet = some (od, x) → (look own x).isSome ∧ (look other x).isSome

theorem scan_inv (R : Nat → Nat → Prop) (root : Nat) (other : PMap) (cur depth : Nat)
    (nbs : List Nat) (hR : ∀ nb ∈ nbs, R cur nb) :
    ∀ own next meet, Inv own root R → (∃ p d, look own cur = some (p, d) ∧ d < depth) →
      MInv own other meet → FInv own next (depth + 1) →
      let r := scan other cur depth nbs (own, next, meet)
      Inv r.1 root R ∧ MInv r.1 other r.2.2 ∧ FInv r.1 r.2.1 (depth + 1) ∧
        (∃ p d, look r.1 cur = some (p, d) ∧ d < depth) ∧
        (∀ x v, look own x = some v → look r.1 x = some v) := by
  induction nbs with
  | nil => intro own next meet h1 h2 h3 h4; exact ⟨h1, h3, h4, h2, fun _ _ h => h⟩
  | cons nb nbs ih =>
    intro own next meet hI hc hM hF
    have hR' : ∀ x ∈ nbs, R cur x := fun x hx => hR x (by simp [hx])
    simp only [scan]
    split
    · exact ih hR' own next meet hI hc hM hF
    · rename_i hnone
      have hnb : look own nb = none := by simpa using hnone
      have hI' := insert_inv R root own nb cur depth hI hnb (hR nb (by simp)) hc
      have hpres : ∀ x v, look own x = some v → look ((nb, cur, depth) :: own) x = some v :=
        fun x v hx => insert_pres own nb cur depth x v hnb hx
      have hc' : ∃ p d, look ((nb, cur, depth) :: own) cur = some (p, d) ∧ d < depth := by
        obtain ⟨p, d, hp, hd⟩ := hc; exact ⟨p, d, hpres _ _ hp, hd⟩
      have hF' : ∀ nx, FInv own nx (depth + 1) → FInv ((nb, cur, depth) :: own) nx (depth + 1) := by
        intro nx h c hcm; obtain ⟨p, d, hp, hd⟩ := h c hcm; exact ⟨p, d, hpres _ _ hp, hd⟩
      have hMlift : ∀ o x, meet = some (o, x) → (look ((nb, cur, depth) :: own) x).isSome ∧
          (look other x).isSome := by
        intro o x hx; obtain ⟨h1, h2⟩ := hM o x hx; refine ⟨?_, h2⟩
        cases e : look own x with
        | none => simp [e] at h1
        | some v => simp [hpres x v e]
      split
      · rename_i pp od hod
        have hM' : MInv ((nb, cur, depth) :: own) other (bestMeet meet od nb) := by
          intro o x hx
          rcases bestMeet_cases meet od nb o x hx with ⟨_, rfl⟩ | hm
          · exact ⟨by rw [look_cons]; simp, by simp [hod]⟩
          · exact hMlift o x hm
        obtain ⟨a, b, c, d, e⟩ := ih hR' _ next _ hI' hc' hM' (hF' next hF)
        exact ⟨a, b, c, d, fun x v hx => e x v (hpres x v hx)⟩
      · have hM' : MInv ((nb, cur, depth) :: own) other meet := hMlift
        have hF2 : FInv ((nb, cur, depth) :: own) (next ++ [nb]) (depth + 1) := by
          intro c hcm
          simp only [List.mem_append, List.mem_singleton] at hcm
          rcases hcm with hcm | rfl
          · exact hF' next hF c hcm
          · exact ⟨cur, depth, by rw [look_cons]; simp, by omega⟩
        obtain ⟨a, b, c, d, e⟩ := ih hR' _ _ _ hI' hc' hM' hF2
        exact ⟨a, b, c, d, fun x v hx => e x v (hpres x v hx)⟩

theorem level_inv (R : Nat → Nat → Prop) (root : Nat) (nbrs : Nat → List Nat)
    (hR : ∀ c, ∀ nb ∈ nbrs c, R c nb) (other : PMap) (depth : Nat) :
    ∀ (front : List Nat) own next meet, Inv own root R → FInv own front depth →
      MInv own other meet → FInv own next (depth + 1) →
      let r := level nbrs other depth front (own, next, meet)
      Inv r.1 root R ∧ MInv r.1 other r.2.2 ∧ FInv r.1 r.2.1 (depth + 1) := by
  intro front
  induction front with
  | nil => intro own next meet h1 _ h3 h4; exact ⟨h1, h3, h4⟩
  | cons c cs ih =>
    intro own next meet hI hFr hM hF
    simp only [level]
    obtain ⟨a, b, d, _, e⟩ := scan_inv R root other c depth (nbrs c) (hR c) own next meet hI
      (hFr c (by simp)) hM hF
    have hFr' : FInv (scan other c depth (nbrs c) (own, next, meet)).1 cs depth := by
      intro x hx; obtain ⟨p, dd, hp, hd⟩ := hFr x (by simp [hx]); exact ⟨p, dd, e _ _ hp, hd⟩
    exact ih _ _ _ a hFr' b d

/-- `MInv` w.r.t. the *other* map is preserved when the other side grows monotonically. -/
def Mono (m m' : PMap) : Prop := ∀ x v, look m x = some v → look m' x = some v

/-! ### Chains -/

/-- A list whose consecutive pairs `(a, b)` satisfy `R b a` (child, then parent). -/
def UpChain (R : Nat → Nat → Prop) : List Nat → Prop
  | [] | [_] => True
  | a :: b :: rest => R b a ∧ UpChain R (b :: rest)

theorem chain_sound (m : PMap) (root : Nat) (R : Nat → Nat → Prop) (hI : Inv m root R) :
    ∀ fuel x l, chain m root fuel x = some l →
      l.head? = some x ∧ l.getLast? = some root ∧ UpChain R l := by
  intro fuel
  induction fuel with
  | zero => intro x l h; cases h
  | succ n ih =>
    intro x l h
    simp only [chain] at h
    split at h
    · rename_i e; cases h; subst e; simp [UpChain]
    · rename_i hne
      split at h
      · rename_i p d hx
        cases hc : chain m root n p with
        | none => simp [hc] at h
        | some l' =>
          simp [hc] at h; subst h
          obtain ⟨h1, h2, h3⟩ := ih p l' hc
          rcases hI.2 x p d hx with e | ⟨hr, _⟩
          · exact absurd e hne
          · refine ⟨rfl, ?_, ?_⟩
            · cases l' with
              | nil => simp at h1
              | cons a as => simpa [List.getLast?_cons_cons] using h2
            · cases l' with
              | nil => simp at h1
              | cons a as => simp at h1; subst h1; exact ⟨hr, h3⟩
      · cases h

/-- Fuel suffices: depths strictly decrease along parents. -/
theorem chain_total (m : PMap) (root : Nat) (R : Nat → Nat → Prop) (hI : Inv m root R) :
    ∀ d x p fuel, look m x = some (p, d) → d < fuel → (chain m root fuel x).isSome := by
  intro d
  induction d using Nat.strongRecOn with
  | ind d ih =>
    intro x p fuel hx hf
    cases fuel with
    | zero => omega
    | succ n =>
      simp only [chain]
      split
      · rfl
      · rename_i hne
        rw [hx]; simp only
        rcases hI.2 x p d hx with e | ⟨_, p', d', hp, hd⟩
        · exact absurd e hne
        · have := ih d' hd p p' n hp (by omega)
          cases hc : chain m root n p with
          | none => simp [hc] at this
          | some l => simp

theorem foldl_max_ge (l : List Nat) (a : Nat) : a ≤ l.foldl max a ∧ ∀ x ∈ l, x ≤ l.foldl max a := by
  induction l generalizing a with
  | nil => simp
  | cons y ys ih =>
    simp only [List.foldl_cons, List.mem_cons]
    obtain ⟨h1, h2⟩ := ih (max a y)
    refine ⟨Nat.le_trans (Nat.le_max_left a y) h1, ?_⟩
    rintro x (rfl | hx)
    · exact Nat.le_trans (Nat.le_max_right a x) h1
    · exact h2 x hx

theorem look_le_maxDepth (m : PMap) (x p d : Nat) (h : look m x = some (p, d)) : d ≤ maxDepth m := by
  unfold look at h
  cases hf : m.find? (·.1 == x) with
  | none => simp [hf] at h
  | some e =>
    simp [hf] at h
    have hm := List.mem_of_find?_eq_some hf
    obtain ⟨k, p', d'⟩ := e
    simp at h; obtain ⟨rfl, rfl⟩ := h
    exact (foldl_max_ge _ 0).2 _ (List.mem_map.mpr ⟨_, hm, rfl⟩)

/-! ### The search loop -/

def Rf (fN : Nat → List Nat) (p x : Nat) : Prop := x ∈ fN p
def Rb (bN : Nat → List Nat) (p x : Nat) : Prop := x ∈ bN p

def SInv (fN bN : Nat → List Nat) (src dst : Nat) (s : St) : Prop :=
  Inv s.fmap src (Rf fN) ∧ Inv s.bmap dst (Rb bN) ∧ FInv s.fmap s.ffront (s.df + 1) ∧
  FInv s.bmap s.bfront (s.db + 1)

theorem search_sound (fN bN : Nat → List Nat) (src dst maxLevel : Nat) :
    ∀ fuel s s' m, SInv fN bN src dst s → search fN bN maxLevel fuel s = some (s', m) →
      SInv fN bN src dst s' ∧ (look s'.fmap m).isSome ∧ (look s'.bmap m).isSome := by
  intro fuel
  induction fuel with
  | zero => intro s s' m _ h; cases h
  | succ n ih =>
    intro s s' m hS h
    obtain ⟨hIf, hIb, hFf, hFb⟩ := hS
    simp only [search] at h
    split at h
    · cases h
    split at h
    · have hl := level_inv (Rf fN) src fN (fun c nb h => h) s.bmap (s.df + 1) s.ffront s.fmap []
        none hIf hFf (by intro _ _ h; cases h) (by intro _ h; cases h)
      revert hl h
      generalize level fN s.bmap (s.df + 1) s.ffront (s.fmap, [], none) = r
      obtain ⟨own, next, meet⟩ := r
      intro h ⟨a, b, c⟩
      have hS' : SInv fN bN src dst { s with fmap := own, ffront := next, df := s.df + 1 } :=
        ⟨a, hIb, c, hFb⟩
      cases meet with
      | none => exact ih _ s' m hS' h
      | some mm =>
        obtain ⟨od, mn⟩ := mm
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨hS', (b od mn rfl).1, (b od mn rfl).2⟩
    · have hl := level_inv (Rb bN) dst bN (fun c nb h => h) s.fmap (s.db + 1) s.bfront s.bmap []
        none hIb hFb (by intro _ _ h; cases h) (by intro _ h; cases h)
      revert hl h
      generalize level bN s.fmap (s.db + 1) s.bfront (s.bmap, [], none) = r
      obtain ⟨own, next, meet⟩ := r
      intro h ⟨a, b, c⟩
      have hS' : SInv fN bN src dst { s with bmap := own, bfront := next, db := s.db + 1 } :=
        ⟨hIf, a, hFf, c⟩
      cases meet with
      | none => exact ih _ s' m hS' h
      | some mm =>
        obtain ⟨od, mn⟩ := mm
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨hS', (b od mn rfl).2, (b od mn rfl).1⟩

/-! ### Soundness of the returned path -/

/-- Consecutive pairs satisfy `E`. -/
def Walk (E : Nat → Nat → Prop) : List Nat → Prop
  | [] | [_] => True
  | a :: b :: rest => E a b ∧ Walk E (b :: rest)

theorem walk_append (E : Nat → Nat → Prop) :
    ∀ (l1 l2 : List Nat) (x : Nat), Walk E (l1 ++ [x]) → Walk E (x :: l2) → Walk E (l1 ++ x :: l2) := by
  intro l1
  induction l1 with
  | nil => intro l2 x _ h; simpa using h
  | cons a as ih =>
    intro l2 x h1 h2
    cases as with
    | nil => simp [Walk] at h1 ⊢; exact ⟨h1, h2⟩
    | cons b bs =>
      simp only [List.cons_append, Walk] at h1 ⊢
      exact ⟨h1.1, by simpa using ih l2 x (by simpa using h1.2) h2⟩

theorem upChain_reverse (R : Nat → Nat → Prop) : ∀ l : List Nat, UpChain R l → Walk R l.reverse := by
  intro l
  induction l with
  | nil => intro _; trivial
  | cons a as ih =>
    intro h
    cases as with
    | nil => trivial
    | cons b bs =>
      obtain ⟨hr, hrest⟩ := h
      have := ih hrest
      simp only [List.reverse_cons] at this ⊢
      rw [List.append_assoc]
      apply walk_append R _ _ b
      · simpa using this
      · simp [Walk, hr]

theorem upChain_walk (R : Nat → Nat → Prop) : ∀ l : List Nat, UpChain R l → Walk (fun a b => R b a) l := by
  intro l
  induction l with
  | nil => intro _; trivial
  | cons a as ih =>
    intro h
    cases as with
    | nil => trivial
    | cons b bs => exact ⟨h.1, ih h.2⟩

theorem walk_mono (E E' : Nat → Nat → Prop) (hE : ∀ a b, E a b → E' a b) :
    ∀ l, Walk E l → Walk E' l := by
  intro l
  induction l with
  | nil => intro _; trivial
  | cons a as ih =>
    intro h
    cases as with
    | nil => trivial
    | cons b bs => exact ⟨hE _ _ h.1, ih h.2⟩

/-- **`bfs_sound`**: if the BFS meets, the reconstruction never hits the
`expect("BFS parent chain broken")` panic, and the node sequence starts at `src`, ends at
`dst`, and every step follows a forward edge (`v ∈ fN u`) or a backward-side edge
(`u ∈ bN v`, i.e. an edge `u → v` for directed search where `bN` is the transpose). -/
theorem bfs_sound (fN bN : Nat → List Nat) (maxLevel fuel src dst : Nat) (hne : src ≠ dst)
    (s : St) (m : Nat)
    (hs : search fN bN maxLevel fuel ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ = some (s, m)) :
    ∃ path, bfsNodes fN bN maxLevel fuel src dst = some path ∧ path.head? = some src ∧
      path.getLast? = some dst ∧ Walk (fun u v => v ∈ fN u ∨ u ∈ bN v) path := by
  have hS0 : SInv fN bN src dst ⟨[(src, src, 0)], [(dst, dst, 0)], [src], [dst], 0, 0⟩ := by
    refine ⟨⟨by simp [look], ?_⟩, ⟨by simp [look], ?_⟩, ?_, ?_⟩
    · intro x p d h; left; simp [look] at h; by_cases e : src = x <;> simp_all
    · intro x p d h; left; simp [look] at h; by_cases e : dst = x <;> simp_all
    · intro c hc; simp at hc; subst hc; exact ⟨c, 0, by simp [look], by omega⟩
    · intro c hc; simp at hc; subst hc; exact ⟨c, 0, by simp [look], by omega⟩
  obtain ⟨⟨hIf, hIb, _, _⟩, hmf, hmb⟩ := search_sound fN bN src dst maxLevel fuel _ s m hS0 hs
  -- both chains exist
  obtain ⟨⟨pf, df⟩, hlf⟩ := Option.isSome_iff_exists.mp hmf
  obtain ⟨⟨pb, db⟩, hlb⟩ := Option.isSome_iff_exists.mp hmb
  have cf := chain_total s.fmap src (Rf fN) hIf df m pf (maxDepth s.fmap + 1) hlf
    (Nat.lt_succ_of_le (look_le_maxDepth _ _ _ _ hlf))
  have cb := chain_total s.bmap dst (Rb bN) hIb db m pb (maxDepth s.bmap + 1) hlb
    (Nat.lt_succ_of_le (look_le_maxDepth _ _ _ _ hlb))
  obtain ⟨l1, h1⟩ := Option.isSome_iff_exists.mp cf
  obtain ⟨l2, h2⟩ := Option.isSome_iff_exists.mp cb
  obtain ⟨a1, b1, c1⟩ := chain_sound s.fmap src (Rf fN) hIf _ m l1 h1
  obtain ⟨a2, b2, c2⟩ := chain_sound s.bmap dst (Rb bN) hIb _ m l2 h2
  refine ⟨l1.reverse ++ l2.tail, by simp [bfsNodes, hne, hs, h1, h2, bind, Option.bind], ?_, ?_, ?_⟩
  · cases l1 with
    | nil => simp at a1
    | cons x xs =>
      have : (x :: xs).reverse.head? = (x :: xs).getLast? := by
        rw [List.head?_reverse]
      simp only [List.head?_append, this, b1, Option.some_or]
  · cases l2 with
    | nil => simp at a2
    | cons y ys =>
      simp at a2; subst a2
      cases ys with
      | nil =>
        simp at b2; subst b2
        cases l1 with
        | nil => simp at a1
        | cons x xs => simp at a1; subst a1; simp [List.getLast?_reverse]
      | cons z zs =>
        simp only [List.tail_cons, List.getLast?_append]
        simp only [List.getLast?_cons_cons] at b2
        simp [b2]
  · cases l1 with
    | nil => simp at a1
    | cons x xs =>
      simp at a1; subst a1
      cases l2 with
      | nil => simp at a2
      | cons y ys =>
        simp at a2; subst a2
        have w1 : Walk (fun u v => v ∈ fN u ∨ u ∈ bN v) (y :: xs).reverse :=
          walk_mono _ _ (fun a b h => Or.inl h) _ (upChain_reverse (Rf fN) _ c1)
        have w2 : Walk (fun u v => v ∈ fN u ∨ u ∈ bN v) (y :: ys) :=
          walk_mono _ _ (fun a b h => Or.inr h) _ (upChain_walk (Rb bN) _ c2)
        rw [List.reverse_cons] at w1 ⊢
        simp only [List.tail_cons]
        simpa using walk_append _ _ ys y w1 w2

/-! ### `eval_shortest_path` (eval.rs:1315) and the path value (eval.rs:1500-1525) -/

inductive PV where
  | null | node (id : Nat) | rel (id : Nat) | other

/-- Interleave relationships: `get_src_dest_relationships(from, to).next()` or the reverse
direction; a missing edge is skipped (cannot happen for a sound walk, see below). -/
def buildPath (relFor : Nat → Nat → Option Nat) : List Nat → List PV
  | [] => []
  | [a] => [.node a]
  | a :: b :: rest => .node a :: ((relFor a b).map PV.rel).toList ++ buildPath relFor (b :: rest)

def evalShortestPath (fN bN : Nat → List Nat) (relFor : Nat → Nat → Option Nat) (maxLevel fuel : Nat)
    (minHops : Nat) (a b : PV) : Except String (Option (List PV)) :=
  match a, b with
  | .null, _ => .ok none
  | .node _, .null => .ok none
  | .node s, .node d =>
    if minHops = 0 ∧ s = d then .ok (some [.node s])
    else match bfsNodes fN bN maxLevel fuel s d with
      | none => .ok none
      | some ns => if ns.length - 1 < minHops then .ok none else .ok (some (buildPath relFor ns))
  | _, _ => .error "A shortestPath requires bound nodes"

theorem shortestPath_args (fN bN : Nat → List Nat) (rf : Nat → Nat → Option Nat) (ml fu mh : Nat) (s : Nat) :
    evalShortestPath fN bN rf ml fu mh .null (.node s) = .ok none ∧
    evalShortestPath fN bN rf ml fu mh (.node s) .null = .ok none ∧
    evalShortestPath fN bN rf ml fu mh .other (.node s) = .error "A shortestPath requires bound nodes" ∧
    evalShortestPath fN bN rf ml fu 0 (.node s) (.node s) = .ok (some [.node s]) := by
  refine ⟨rfl, rfl, rfl, by simp [evalShortestPath]⟩

/-- `src = dst` with `min_hops > 0` is `Null` (no cyclic path). -/
theorem shortestPath_self (fN bN : Nat → List Nat) (rf : Nat → Nat → Option Nat) (ml fu mh s : Nat)
    (h : mh ≠ 0) : evalShortestPath fN bN rf ml fu mh (.node s) (.node s) = .ok none := by
  simp [evalShortestPath, h, bfsNodes]

/-- When every walk step has a relationship, the path value alternates node/rel and has
`2k+1` entries for a `k`-edge walk. -/
theorem buildPath_length (rf : Nat → Nat → Option Nat) :
    ∀ ns : List Nat, ns ≠ [] → (∀ a b, (rf a b).isSome) → (buildPath rf ns).length = 2 * ns.length - 1 := by
  intro ns
  induction ns with
  | nil => intro h; exact absurd rfl h
  | cons a as ih =>
    intro _ hr
    cases as with
    | nil => rfl
    | cons b bs =>
      have := ih (by simp) hr
      obtain ⟨r, hr'⟩ := Option.isSome_iff_exists.mp (hr a b)
      simp only [buildPath, hr', Option.map_some, Option.toList_some, List.cons_append,
        List.length_cons] at this ⊢
      simp only [List.nil_append, List.length_cons] at this ⊢
      omega

end FalkorExpr.BFS
