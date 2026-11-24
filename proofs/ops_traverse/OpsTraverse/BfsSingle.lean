import OpsTraverse.SearchBase
/-
# `bfs_single_path` (algo_procedures.rs): bidirectional BFS for an unweighted `shortestPath`

`run_path_algo` sends a single-path query without `weightProp` here instead of to Dijkstra. Two
`BfsSide`s grow level by level, forward from the source along `sf` (`node_successors` in the
configured direction) and backward from the target along `sb` (the reversed direction). `seen`
(`FxHashMap<NodeId, (depth, Option<(prev, edge)>)>`) is a function; it is only ever written at an
absent key. The side to grow is picked from the `work` estimates (`BfsSide::measure`, sums of
`degree_capped` for frontiers of at most `BFS_DEGREE_PROBE_MAX` nodes; `deg` is `degree_capped`),
else by frontier size. The `while` loop and `BfsSide::chain` are fuelled; theorems cover finished
runs. The periodic `check_timeout` / `check_mem_capacity` errors are not modelled (they return no
path). Reported edges carry their far node (`(node, edge)`); the Rust keeps the edge triple only.

| here | there |
| --- | --- |
| `Side`, `Side.new` | `BfsSide`, `BfsSide::new` |
| `measure` | `BfsSide::measure` |
| `scanE` | the inner `for … in g.get_node_relationships_by_type(node, types, side.dir)` |
| `scanFront` | `for &node in &side.frontier` |
| `grow` | one level: the scan, `side.depth = depth`, the frontier swap and `measure` |
| `fwdFirst` | the `fwd_first` match |
| `loop` | `while best.is_none() && … { … }` |
| `chain` | `BfsSide::chain` (with the node each edge leads from) |
| `bfsSingle` | `bfs_single_path` up to the edge list |
-/
namespace OpsTraverse.BfsSingle
open OpsTraverse.Search

variable {ε : Type}

/-- `seen`: node ↦ (depth, the node it was reached from and the edge). -/
abbrev SMap (ε : Type) := Nat → Option (Nat × Option (Nat × ε))

def upd (m : SMap ε) (k : Nat) (v : Nat × Option (Nat × ε)) : SMap ε :=
  fun y => if y = k then some v else m y

structure Side (ε : Type) where
  seen : SMap ε
  frontier : List Nat
  work : Option Nat
  depth : Nat

/-- `BFS_DEGREE_PROBE_MAX` -/
def probeMax : Nat := 8

/-- `BfsSide::measure`: `deg` is `degree_capped(…, BFS_DEGREE_PROBE_CAP)`. -/
def measure (deg : Nat → Nat) (fr : List Nat) : Option Nat :=
  if fr.length ≤ probeMax then some (fr.map deg).sum else none

/-- `BfsSide::new` -/
def Side.new (deg : Nat → Nat) (root : Nat) : Side ε :=
  ⟨fun y => if y = root then some (0, none) else none, [root], some (deg root), 0⟩

/-- Scan state: `side.seen`, `next`, `best` (path length, meeting node). -/
structure SS (ε : Type) where
  seen : SMap ε
  next : List Nat
  best : Option (Nat × Nat)

/-- The relationships of one frontier node (`node_successors` order), `other` the other side's map. -/
def scanE (other : SMap ε) (depth node : Nat) : List (Nat × ε) → SS ε → SS ε
  | [], s => s
  | x :: r, s =>
    match s.seen x.1 with
    | some _ => scanE other depth node r s
    | none =>
      let best := match other x.1 with
        | some (od, _) => if s.best.all (fun b => decide (depth + od < b.1)) then some (depth + od, x.1)
            else s.best
        | none => s.best
      scanE other depth node r ⟨upd s.seen x.1 (depth, some (node, x.2)), s.next ++ [x.1], best⟩

def scanFront (st : Step ε) (other : SMap ε) (depth : Nat) : List Nat → SS ε → SS ε
  | [], s => s
  | node :: r, s => scanFront st other depth r (scanE other depth node (st node) s)

/-- One level of `side` (entered with `best = None`). -/
def grow (st : Step ε) (deg : Nat → Nat) (side : Side ε) (other : SMap ε) : Side ε × Option (Nat × Nat) :=
  let s := scanFront st other (side.depth + 1) side.frontier ⟨side.seen, [], none⟩
  (⟨s.seen, s.next, measure deg s.next, side.depth + 1⟩, s.best)

def fwdFirst (f b : Side ε) : Bool :=
  match f.work, b.work with
  | some x, some y => decide (x ≤ y)
  | _, _ => decide (f.frontier.length ≤ b.frontier.length)

def loop (sf sb : Step ε) (degF degB : Nat → Nat) :
    Nat → Side ε → Side ε → Option (Nat × Nat) → Option (Side ε × Side ε × Option (Nat × Nat))
  | 0, _, _, _ => none
  | n + 1, f, b, best =>
    if best.isNone && !f.frontier.isEmpty && !b.frontier.isEmpty then
      if fwdFirst f b then
        let r := grow sf degF f b.seen
        loop sf sb degF degB n r.1 b r.2
      else
        let r := grow sb degB b f.seen
        loop sf sb degF degB n f r.1 r.2
    else some (f, b, best)

/-- `BfsSide::chain`: `(node, prev, edge)` from `x` back to the root, nearest `x` first. -/
def chain (m : SMap ε) : Nat → Nat → Option (List (Nat × Nat × ε))
  | 0, _ => none
  | f + 1, x =>
    match m x with
    | some (_, some (p, e)) => (chain m f p).map ((x, p, e) :: ·)
    | _ => some []

/-- `bfs_single_path` up to the edge list: `none` = out of fuel, `some none` = `Ok(None)`, else the
path `source → target` as `(far node, edge)` in walk order (`fwd.chain(meet)` reversed, then
`bwd.chain(meet)`). -/
def bfsSingle (sf sb : Step ε) (degF degB : Nat → Nat) (src tgt fuel : Nat) :
    Option (Option (List (Nat × ε))) :=
  match loop sf sb degF degB fuel (Side.new degF src) (Side.new degB tgt) none with
  | none => none
  | some (_, _, none) => some none
  | some (f, b, some (_, meet)) =>
    match chain f.seen fuel meet, chain b.seen fuel meet with
    | some cf, some cb =>
      some (some ((cf.map (fun q => (q.1, q.2.2))).reverse ++ cb.map (fun q => (q.2.1, q.2.2))))
    | _, _ => none

/-! ## Invariants -/

/-- Every entry but the root's links to a parent one level shallower by a `st` edge. -/
structure SideInv (st : Step ε) (r : Nat) (m : SMap ε) : Prop where
  root : m r = some (0, none)
  noLink : ∀ x k, m x = some (k, none) → x = r ∧ k = 0
  link : ∀ x k p e, m x = some (k, some (p, e)) → 1 ≤ k ∧ (x, e) ∈ st p ∧ ∃ l', m p = some (k - 1, l')

/-- `best` against the two maps: none = disjoint; some = a common node of depth sum `L`, the least. -/
def Bst (m1 m2 : SMap ε) : Option (Nat × Nat) → Prop
  | none => ∀ y v, m1 y = some v → m2 y = none
  | some (L, mt) => (∃ a la b lb, m1 mt = some (a, la) ∧ m2 mt = some (b, lb) ∧ a + b = L) ∧
      ∀ y a la b lb, m1 y = some (a, la) → m2 y = some (b, lb) → L ≤ a + b

theorem Bst_symm (m1 m2 : SMap ε) : ∀ best, Bst m1 m2 best → Bst m2 m1 best
  | none, h => by
    intro y v hv
    cases h1 : m1 y with
    | none => rfl
    | some w => rw [h y w h1] at hv; cases hv
  | some (L, mt), h => by
    obtain ⟨⟨a, la, b, lb, h1, h2, h3⟩, h4⟩ := h
    exact ⟨⟨b, lb, a, la, h2, h1, by omega⟩, fun y a' la' b' lb' h5 h6 => by
      have := h4 y b' lb' a' la' h6 h5; omega⟩

def Ext (m m' : SMap ε) : Prop := ∀ y v, m y = some v → m' y = some v

theorem upd_eq (m : SMap ε) (k : Nat) (v) : upd m k v k = some v := by simp [upd]
theorem upd_ne (m : SMap ε) (k y : Nat) (v) (h : y ≠ k) : upd m k v y = m y := by simp [upd, h]

theorem ext_upd (m : SMap ε) (k : Nat) (v) (hk : m k = none) : Ext m (upd m k v) := by
  intro y w hy
  have : y ≠ k := by rintro rfl; rw [hk] at hy; cases hy
  rw [upd_ne m k y v this]; exact hy

theorem SideInv_upd (st : Step ε) (r : Nat) (m : SMap ε) (I : SideInv st r m) (x : Nat) (e : ε)
    (node d : Nat) (l0 : Option (Nat × ε)) (hx : m x = none) (hn : m node = some (d, l0))
    (he : (x, e) ∈ st node) : SideInv st r (upd m x (d + 1, some (node, e))) where
  root := by
    have : r ≠ x := by rintro rfl; rw [I.root] at hx; cases hx
    rw [upd_ne _ _ _ _ this]; exact I.root
  noLink y k h := by
    by_cases hy : y = x
    · subst hy; rw [upd_eq] at h; cases h
    · rw [upd_ne _ _ _ _ hy] at h; exact I.noLink y k h
  link y k p e' h := by
    have hnx : node ≠ x := by rintro rfl; rw [hn] at hx; cases hx
    by_cases hy : y = x
    · subst hy
      rw [upd_eq] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      refine ⟨by omega, he, l0, ?_⟩
      rw [upd_ne _ _ _ _ hnx]; simpa using hn
    · rw [upd_ne _ _ _ _ hy] at h
      obtain ⟨h1, h2, l', h3⟩ := I.link y k p e' h
      have hpx : p ≠ x := by rintro rfl; rw [h3] at hx; cases hx
      exact ⟨h1, h2, l', by rw [upd_ne _ _ _ _ hpx]; exact h3⟩

/-- Inserting a node of depth `d` at an absent key keeps `Bst` with the updated `best`. -/
theorem Bst_upd (m other : SMap ε) (best : Option (Nat × Nat)) (hB : Bst m other best) (x d : Nat)
    (v : Option (Nat × ε)) (hx : m x = none) :
    Bst (upd m x (d, v)) other (match other x with
        | some (od, _) => if best.all (fun b => decide (d + od < b.1)) then some (d + od, x) else best
        | none => best) := by
  have hext := ext_upd m x (d, v) hx
  cases ho : other x with
  | none =>
    simp only
    cases best with
    | none =>
      intro y w hy
      by_cases hyx : y = x
      · subst hyx; exact ho
      · rw [upd_ne _ _ _ _ hyx] at hy; exact hB y w hy
    | some q =>
      obtain ⟨L, mt⟩ := q
      obtain ⟨⟨a, la, b, lb, h1, h2, h3⟩, h4⟩ := hB
      refine ⟨⟨a, la, b, lb, hext _ _ h1, h2, h3⟩, fun y a' la' b' lb' h5 h6 => ?_⟩
      by_cases hyx : y = x
      · subst hyx; rw [ho] at h6; cases h6
      · rw [upd_ne _ _ _ _ hyx] at h5; exact h4 y a' la' b' lb' h5 h6
  | some w =>
    obtain ⟨od, lo⟩ := w
    simp only
    cases best with
    | none =>
      simp only [Option.all_none, ↓reduceIte]
      refine ⟨⟨d, v, od, lo, upd_eq _ _ _, ho, rfl⟩, fun y a' la' b' lb' h5 h6 => ?_⟩
      by_cases hyx : y = x
      · subst hyx
        rw [upd_eq] at h5; rw [ho] at h6
        simp only [Option.some.injEq, Prod.mk.injEq] at h5 h6
        omega
      · rw [upd_ne _ _ _ _ hyx] at h5
        rw [hB y _ h5] at h6; cases h6
    | some q =>
      obtain ⟨L, mt⟩ := q
      obtain ⟨⟨a, la, b, lb, h1, h2, h3⟩, h4⟩ := hB
      simp only [Option.all_some, decide_eq_true_eq]
      by_cases hlt : d + od < L
      · simp only [hlt, ↓reduceIte]
        refine ⟨⟨d, v, od, lo, upd_eq _ _ _, ho, rfl⟩, fun y a' la' b' lb' h5 h6 => ?_⟩
        by_cases hyx : y = x
        · subst hyx
          rw [upd_eq] at h5; rw [ho] at h6
          simp only [Option.some.injEq, Prod.mk.injEq] at h5 h6
          omega
        · rw [upd_ne _ _ _ _ hyx] at h5
          have := h4 y a' la' b' lb' h5 h6; omega
      · simp only [hlt, ↓reduceIte]
        refine ⟨⟨a, la, b, lb, hext _ _ h1, h2, h3⟩, fun y a' la' b' lb' h5 h6 => ?_⟩
        by_cases hyx : y = x
        · subst hyx
          rw [upd_eq] at h5; rw [ho] at h6
          simp only [Option.some.injEq, Prod.mk.injEq] at h5 h6
          omega
        · rw [upd_ne _ _ _ _ hyx] at h5; exact h4 y a' la' b' lb' h5 h6

/-- An entry of depth `k` is reached by a walk of `k` edges from the root. -/
theorem entry_R (st : Step ε) (r : Nat) (m : SMap ε) (I : SideInv st r m) :
    ∀ k x l, m x = some (k, l) → R st r k x
  | 0, x, l, h => by
    cases l with
    | none => rw [(I.noLink x 0 h).1]; exact (R_zero st r r).mpr rfl
    | some pe => have := (I.link x 0 pe.1 pe.2 h).1; omega
  | k + 1, x, l, h => by
    cases l with
    | none => have := (I.noLink x (k + 1) h).2; omega
    | some pe =>
      obtain ⟨_, he, l', hp⟩ := I.link x (k + 1) pe.1 pe.2 h
      exact (R_succ st r k x).mpr ⟨pe.1, pe.2, entry_R st r m I k pe.1 l' (by simpa using hp), he⟩

/-- Mid-level scan invariant (`d` = the new depth). -/
structure SInv (st : Step ε) (r : Nat) (other : SMap ε) (d : Nat) (s : SS ε) : Prop where
  side : SideInv st r s.seen
  next : ∀ x ∈ s.next, ∃ l, s.seen x = some (d, l)
  bst : Bst s.seen other s.best
  bound : ∀ x k l, s.seen x = some (k, l) → k ≤ d
  all : ∀ x l, s.seen x = some (d, l) → x ∈ s.next

theorem scanE_inv (st : Step ε) (r : Nat) (other : SMap ε) (d node : Nat) (l0 : Option (Nat × ε)) :
    ∀ (es : List (Nat × ε)) (s : SS ε), (∀ x ∈ es, x ∈ st node) → s.seen node = some (d, l0) →
      SInv st r other (d + 1) s →
      SInv st r other (d + 1) (scanE other (d + 1) node es s) ∧ Ext s.seen (scanE other (d + 1) node es s).seen ∧
      ∀ v e, (v, e) ∈ es → ∃ w, (scanE other (d + 1) node es s).seen v = some w
  | [], s, _, _, I => ⟨I, fun _ _ h => h, fun _ _ h => by simp at h⟩
  | x :: rest, s, hes, hn, I => by
    unfold scanE
    cases hs : s.seen x.1 with
    | some w0 =>
      obtain ⟨h1, h2, h3⟩ := scanE_inv st r other d node l0 rest s (fun y hy => hes y (by simp [hy])) hn I
      refine ⟨h1, h2, fun v e hv => ?_⟩
      rcases List.mem_cons.mp hv with hv | hv
      · subst hv; exact ⟨w0, h2 _ _ hs⟩
      · exact h3 v e hv
    | none =>
      simp only
      have hext := ext_upd s.seen x.1 (d + 1, some (node, x.2)) hs
      have I' : SInv st r other (d + 1) ⟨upd s.seen x.1 (d + 1, some (node, x.2)), s.next ++ [x.1],
          match other x.1 with
          | some (od, _) => if s.best.all (fun b => decide (d + 1 + od < b.1)) then some (d + 1 + od, x.1)
              else s.best
          | none => s.best⟩ :=
        ⟨SideInv_upd st r s.seen I.side x.1 x.2 node d l0 hs hn (hes x (by simp)),
         fun y hy => by
          rcases List.mem_append.mp hy with hy | hy
          · obtain ⟨l, hl⟩ := I.next y hy; exact ⟨l, hext _ _ hl⟩
          · simp only [List.mem_singleton] at hy; subst hy; exact ⟨_, upd_eq _ _ _⟩,
         Bst_upd s.seen other s.best I.bst x.1 (d + 1) (some (node, x.2)) hs,
         fun y k l hy => by
          dsimp only at hy
          by_cases hyx : y = x.1
          · subst hyx; rw [upd_eq] at hy; simp only [Option.some.injEq, Prod.mk.injEq] at hy; omega
          · rw [upd_ne _ _ _ _ hyx] at hy; exact I.bound y k l hy,
         fun y l hy => by
          dsimp only at hy ⊢
          by_cases hyx : y = x.1
          · subst hyx; simp
          · rw [upd_ne _ _ _ _ hyx] at hy; exact List.mem_append_left _ (I.all y l hy)⟩
      obtain ⟨h1, h2, h3⟩ := scanE_inv st r other d node l0 rest _ (fun y hy => hes y (by simp [hy]))
        (hext _ _ hn) I'
      refine ⟨h1, fun y v hy => h2 y v (hext y v hy), fun v e hv => ?_⟩
      rcases List.mem_cons.mp hv with hv | hv
      · subst hv; exact ⟨_, h2 _ _ (upd_eq _ _ _)⟩
      · exact h3 v e hv

theorem scanFront_inv (st : Step ε) (r : Nat) (other : SMap ε) (d : Nat) :
    ∀ (fr : List Nat) (s : SS ε), (∀ x ∈ fr, ∃ l, s.seen x = some (d, l)) → SInv st r other (d + 1) s →
      SInv st r other (d + 1) (scanFront st other (d + 1) fr s) ∧ Ext s.seen (scanFront st other (d + 1) fr s).seen ∧
      ∀ u ∈ fr, ∀ v e, (v, e) ∈ st u → ∃ w, (scanFront st other (d + 1) fr s).seen v = some w
  | [], s, _, I => ⟨I, fun _ _ h => h, fun _ h => by simp at h⟩
  | node :: rest, s, hfr, I => by
    unfold scanFront
    obtain ⟨l0, hn⟩ := hfr node (by simp)
    obtain ⟨I', hext, hsc⟩ := scanE_inv st r other d node l0 (st node) s (fun _ h => h) hn I
    obtain ⟨J, hext2, hsc2⟩ := scanFront_inv st r other d rest _ (fun y hy => by
      obtain ⟨l, hl⟩ := hfr y (by simp [hy])
      exact ⟨l, hext _ _ hl⟩) I'
    refine ⟨J, fun y v hy => hext2 y v (hext y v hy), fun u hu v e hv => ?_⟩
    rcases List.mem_cons.mp hu with rfl | hu
    · obtain ⟨w, hw⟩ := hsc v e hv; exact ⟨w, hext2 _ _ hw⟩
    · exact hsc2 u hu v e hv

/-- One side's invariant after a finished level: the `SideInv` links, frontier = the entries of the
side's depth, all entries within it, and every node within `depth` edges of the root is in the map
with a depth no larger than its walk. -/
structure SideOK (st : Step ε) (r : Nat) (side : Side ε) : Prop where
  inv : SideInv st r side.seen
  fr : ∀ x ∈ side.frontier, ∃ l, side.seen x = some (side.depth, l)
  bound : ∀ x k l, side.seen x = some (k, l) → k ≤ side.depth
  all : ∀ x l, side.seen x = some (side.depth, l) → x ∈ side.frontier
  cov : ∀ x j, j ≤ side.depth → R st r j x → ∃ k l, side.seen x = some (k, l) ∧ k ≤ j

theorem grow_inv (st : Step ε) (deg : Nat → Nat) (r : Nat) (side : Side ε) (other : SMap ε)
    (I0 : SideOK st r side) (hb : Bst side.seen other none) :
    SideOK st r (grow st deg side other).1 ∧ Bst (grow st deg side other).1.seen other (grow st deg side other).2 := by
  obtain ⟨I, hext, hsc⟩ := scanFront_inv st r other side.depth side.frontier ⟨side.seen, [], none⟩ I0.fr
    ⟨I0.inv, fun x hx => by simp at hx, hb, fun x k l h => by have := I0.bound x k l h; omega,
     fun x l h => by have := I0.bound x _ l h; omega⟩
  refine ⟨⟨I.side, I.next, I.bound, I.all, fun x j hj hR => ?_⟩, I.bst⟩
  show ∃ k l, (scanFront st other (side.depth + 1) side.frontier ⟨side.seen, [], none⟩).seen x = some (k, l) ∧ k ≤ j
  change j ≤ side.depth + 1 at hj
  by_cases hjD : j ≤ side.depth
  · obtain ⟨k, l, hk, hkj⟩ := I0.cov x j hjD hR
    exact ⟨k, l, hext _ _ hk, hkj⟩
  · obtain rfl : j = side.depth + 1 := by omega
    obtain ⟨u, e, hu, he⟩ := (R_succ st r side.depth x).mp hR
    obtain ⟨k, l, hk, hkD⟩ := I0.cov u side.depth (Nat.le_refl _) hu
    by_cases hkd : k = side.depth
    · subst hkd
      obtain ⟨⟨k', l'⟩, hw⟩ := hsc u (I0.all u l hk) x e he
      exact ⟨k', l', hw, I.bound x k' l' hw⟩
    · have hR' : R st r (k + 1) x := (R_succ st r k x).mpr ⟨u, e, entry_R st r _ I0.inv k u l hk, he⟩
      obtain ⟨k2, l2, hk2, hk2j⟩ := I0.cov x (k + 1) (by omega) hR'
      exact ⟨k2, l2, hext _ _ hk2, by omega⟩

structure LI (sf sb : Step ε) (src tgt : Nat) (f b : Side ε) (best : Option (Nat × Nat)) : Prop where
  fok : SideOK sf src f
  bok : SideOK sb tgt b
  bst : Bst f.seen b.seen best

/-- The loop keeps `LI`, and stops with `best = None` only when a frontier is empty. -/
theorem loop_inv (sf sb : Step ε) (degF degB : Nat → Nat) (src tgt : Nat) :
    ∀ n f b best res, LI sf sb src tgt f b best → loop sf sb degF degB n f b best = some res →
      LI sf sb src tgt res.1 res.2.1 res.2.2 ∧ (res.2.2 = none → res.1.frontier = [] ∨ res.2.1.frontier = [])
  | 0, _, _, _, _, _, h => by simp [loop] at h
  | n + 1, f, b, best, res, I, h => by
    unfold loop at h
    by_cases hc : (best.isNone && !f.frontier.isEmpty && !b.frontier.isEmpty) = true
    · simp only [hc, ↓reduceIte] at h
      have hbn : best = none := by
        cases best with
        | none => rfl
        | some _ => simp at hc
      subst hbn
      by_cases hd : fwdFirst f b = true
      · simp only [hd, ↓reduceIte] at h
        obtain ⟨g1, g2⟩ := grow_inv sf degF src f b.seen I.fok I.bst
        exact loop_inv sf sb degF degB src tgt n _ b _ res ⟨g1, I.bok, g2⟩ h
      · simp only [Bool.not_eq_true] at hd
        simp only [hd, Bool.false_eq_true, ↓reduceIte] at h
        obtain ⟨g1, g2⟩ := grow_inv sb degB tgt b f.seen I.bok (Bst_symm _ _ none I.bst)
        exact loop_inv sf sb degF degB src tgt n f _ _ res ⟨I.fok, g1, Bst_symm _ _ _ g2⟩ h
    · simp only [Bool.not_eq_true] at hc
      simp only [hc, Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at h
      subst h
      refine ⟨I, fun hn => ?_⟩
      simp only [Bool.and_eq_false_iff, Bool.not_eq_false', List.isEmpty_iff] at hc
      change best = none at hn
      subst hn
      simp only [Option.isNone_none, Bool.true_eq_false, false_or] at hc
      exact hc

/-- What `chain` returns from an entry of depth `k`: `k` links; read backwards, a walk from the root
to `x`; the nodes it ends on (`·.1`) have depths in `1..k`, those it starts from (`·.2.1`) depths
below `k`, each list without repeats. -/
theorem chain_ok (st : Step ε) (r : Nat) (m : SMap ε) (I : SideInv st r m) :
    ∀ f x k l cs, m x = some (k, l) → chain m f x = some cs →
      cs.length = k ∧ WalkTo st r ((cs.map (fun q => (q.1, q.2.2))).reverse) x ∧
      (∀ y ∈ cs.map (·.1), ∃ j l, m y = some (j, l) ∧ 1 ≤ j ∧ j ≤ k) ∧ (cs.map (·.1)).Nodup ∧
      (∀ y ∈ cs.map (·.2.1), ∃ j l, m y = some (j, l) ∧ j + 1 ≤ k) ∧ (cs.map (·.2.1)).Nodup
  | 0, _, _, _, _, _, h => by simp [chain] at h
  | f + 1, x, k, l, cs, hx, h => by
    unfold chain at h
    rw [hx] at h
    cases l with
    | none =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨rfl, rfl⟩ := I.noLink x k hx
      simp [WalkTo]
    | some pe =>
      obtain ⟨p, e⟩ := pe
      simp only at h
      obtain ⟨hk, hst, l', hp⟩ := I.link x k p e hx
      cases hc : chain m f p with
      | none => rw [hc] at h; simp at h
      | some cs' =>
        rw [hc] at h
        simp only [Option.map_some, Option.some.injEq] at h
        subst h
        obtain ⟨c1, c2, c3, c4, c5, c6⟩ := chain_ok st r m I f p (k - 1) l' cs' hp hc
        refine ⟨by simp; omega, ?_, ?_, ?_, ?_, ?_⟩
        · simp only [List.map_cons, List.reverse_cons]
          exact (walk_append st r _ _ x).mpr ⟨p, c2, by simp [WalkTo, hst]⟩
        · intro y hy
          simp only [List.map_cons, List.mem_cons] at hy
          rcases hy with rfl | hy
          · exact ⟨k, _, hx, hk, Nat.le_refl _⟩
          · obtain ⟨j, l2, h1, h2, h3⟩ := c3 y hy; exact ⟨j, l2, h1, h2, by omega⟩
        · simp only [List.map_cons, List.nodup_cons]
          refine ⟨fun hm => ?_, c4⟩
          obtain ⟨j, l2, h1, _, h3⟩ := c3 x hm
          rw [hx] at h1; simp only [Option.some.injEq, Prod.mk.injEq] at h1; omega
        · intro y hy
          simp only [List.map_cons, List.mem_cons] at hy
          rcases hy with rfl | hy
          · exact ⟨k - 1, l', hp, by omega⟩
          · obtain ⟨j, l2, h1, h3⟩ := c5 y hy; exact ⟨j, l2, h1, by omega⟩
        · simp only [List.map_cons, List.nodup_cons]
          refine ⟨fun hm => ?_, c6⟩
          obtain ⟨j, l2, h1, h3⟩ := c5 p hm
          rw [hp] at h1; simp only [Option.some.injEq, Prod.mk.injEq] at h1; omega

/-- A backward side's chain, read forwards with each edge's parent as its far node, is a walk along
`sf` from `x` to the backward root, when `sb` is the converse of `sf` (`hrev`). -/
theorem chain_rev (sf sb : Step ε) (hrev : ∀ u v e, (v, e) ∈ sb u → (u, e) ∈ sf v) (r : Nat) (m : SMap ε)
    (I : SideInv sb r m) :
    ∀ f x k l cs, m x = some (k, l) → chain m f x = some cs →
      WalkTo sf x (cs.map (fun q => (q.2.1, q.2.2))) r
  | 0, _, _, _, _, _, h => by simp [chain] at h
  | f + 1, x, k, l, cs, hx, h => by
    unfold chain at h
    rw [hx] at h
    cases l with
    | none =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨rfl, rfl⟩ := I.noLink x k hx
      rfl
    | some pe =>
      obtain ⟨p, e⟩ := pe
      simp only at h
      obtain ⟨_, hst, l', hp⟩ := I.link x k p e hx
      cases hc : chain m f p with
      | none => rw [hc] at h; simp at h
      | some cs' =>
        rw [hc] at h
        simp only [Option.map_some, Option.some.injEq] at h
        subst h
        exact ⟨hrev p x e hst, chain_rev sf sb hrev r m I f p (k - 1) l' cs' hp hc⟩

theorem SideOK_new (st : Step ε) (deg : Nat → Nat) (r : Nat) : SideOK st r (Side.new deg r) where
  inv := ⟨by simp [Side.new], fun x k h => by simp [Side.new] at h; exact ⟨h.1, h.2.symm⟩,
    fun x k p e h => by simp [Side.new] at h⟩
  fr x hx := by simp [Side.new] at hx; subst hx; simp [Side.new]
  bound x k l h := by simp [Side.new] at h; simp [Side.new]; omega
  all x l h := by simp [Side.new] at h; simp [Side.new, h.1]
  cov x j hj hR := by
    simp only [Side.new] at hj
    obtain rfl : j = 0 := by omega
    rw [(R_zero st r x).mp hR]
    exact ⟨0, none, by simp [Side.new], Nat.le_refl _⟩

theorem LI_init (sf sb : Step ε) (degF degB : Nat → Nat) (src tgt : Nat) (hsd : src ≠ tgt) :
    LI sf sb src tgt (Side.new degF src) (Side.new degB tgt) none where
  fok := SideOK_new sf degF src
  bok := SideOK_new sb degB tgt
  bst y v h := by
    simp only [Side.new] at h ⊢
    split at h
    · next hy => subst hy; simp [hsd]
    · cases h

/-- **PROVEN** (`bfs_single_path`, soundness): for `source ≠ target`, a returned edge list is a
non-empty walk `source → target` along `sf` (the forward `node_successors`) that visits distinct
nodes; `hrev`: the backward side's successors are the converse of the forward ones (a property of
`get_node_relationships_by_type`, see `AlgoSearch.stepOf_rev`). -/
theorem bfsSingle_sound (sf sb : Step ε) (hrev : ∀ u v e, (v, e) ∈ sb u → (u, e) ∈ sf v)
    (degF degB : Nat → Nat) (src tgt fuel : Nat) (hsd : src ≠ tgt) (es : List (Nat × ε))
    (h : bfsSingle sf sb degF degB src tgt fuel = some (some es)) :
    WalkTo sf src es tgt ∧ (nodesOf src es).Nodup ∧ es ≠ [] := by
  unfold bfsSingle at h
  rcases hl : loop sf sb degF degB fuel (Side.new degF src) (Side.new degB tgt) none with _ | ⟨f, b, _ | ⟨L, mt⟩⟩
  · rw [hl] at h; simp at h
  · rw [hl] at h; simp at h
  · rw [hl] at h
    simp only at h
    have I : LI sf sb src tgt f b (some (L, mt)) :=
      (loop_inv sf sb degF degB src tgt fuel _ _ none _ (LI_init sf sb degF degB src tgt hsd) hl).1
    obtain ⟨⟨a, la, bd, lb, hfa, hbb, hL⟩, hmin⟩ := I.bst
    cases hcf : chain f.seen fuel mt with
    | none => rw [hcf] at h; simp at h
    | some cf =>
      cases hcb : chain b.seen fuel mt with
      | none => rw [hcf, hcb] at h; simp at h
      | some cb =>
        rw [hcf, hcb] at h
        simp only [Option.some.injEq] at h
        subst h
        obtain ⟨f1, f2, f3, f4, _, _⟩ := chain_ok sf src f.seen I.fok.inv fuel mt a la cf hfa hcf
        obtain ⟨b1, _, _, _, b5, b6⟩ := chain_ok sb tgt b.seen I.bok.inv fuel mt bd lb cb hbb hcb
        have b2 := chain_rev sf sb hrev tgt b.seen I.bok.inv fuel mt bd lb cb hbb hcb
        refine ⟨(walk_append sf src _ _ tgt).mpr ⟨mt, f2, b2⟩, ?_, ?_⟩
        · have hn : nodesOf src ((cf.map (fun q => (q.1, q.2.2))).reverse ++ cb.map (fun q => (q.2.1, q.2.2))) =
              (src :: (cf.map (·.1)).reverse) ++ cb.map (·.2.1) := by
            simp [nodesOf, List.map_reverse, Function.comp_def]
          rw [hn, List.nodup_append]
          have hsrc : f.seen src = some (0, none) := I.fok.inv.root
          have hA : ∀ y ∈ src :: (cf.map (·.1)).reverse, ∃ j l, f.seen y = some (j, l) ∧ j ≤ a := by
            intro y hy
            simp only [List.mem_cons, List.mem_reverse] at hy
            rcases hy with rfl | hy
            · exact ⟨0, none, hsrc, Nat.zero_le _⟩
            · obtain ⟨j, l, h1, _, h3⟩ := f3 y hy; exact ⟨j, l, h1, h3⟩
          refine ⟨?_, b6, ?_⟩
          · rw [List.nodup_cons]
            refine ⟨fun hm => ?_, (by unfold List.Nodup; rw [List.pairwise_reverse]; exact f4.imp (fun h => Ne.symm h))⟩
            rw [List.mem_reverse] at hm
            obtain ⟨j, l, h1, h2, _⟩ := f3 src hm
            rw [hsrc] at h1; simp only [Option.some.injEq, Prod.mk.injEq] at h1; omega
          · intro y hy z hz hyz
            subst hyz
            obtain ⟨j1, l1, h1, h2⟩ := hA y hy
            obtain ⟨j2, l2, h3, h4⟩ := b5 y hz
            have := hmin y j1 l1 j2 l2 h1 h3
            omega
        · intro hnil
          have hlen := congrArg List.length hnil
          simp only [List.length_append, List.length_reverse, List.length_map, List.length_nil] at hlen
          have ha : a = 0 := by omega
          have hb : bd = 0 := by omega
          subst ha hb
          have e1 : mt = src := by
            cases la with
            | none => exact (I.fok.inv.noLink mt 0 hfa).1
            | some pe => have := (I.fok.inv.link mt 0 pe.1 pe.2 hfa).1; omega
          have e2 : mt = tgt := by
            cases lb with
            | none => exact (I.bok.inv.noLink mt 0 hbb).1
            | some pe => have := (I.bok.inv.link mt 0 pe.1 pe.2 hbb).1; omega
          exact hsd (e1.symm.trans e2)

/-- A forward walk `y → t` read backwards is a walk `t → y` along `sb` when `sb` contains the converse
of `sf` (`hrev2`). -/
theorem walk_revR (sf sb : Step ε) (hrev2 : ∀ u v e, (v, e) ∈ sf u → (u, e) ∈ sb v) (t : Nat) :
    ∀ (w : List (Nat × ε)) (y : Nat), WalkTo sf y w t → R sb t w.length y
  | [], y, h => by simp only [WalkTo] at h; subst h; exact (R_zero sb y y).mpr rfl
  | x :: w, y, h => by
    obtain ⟨hx, hw⟩ := h
    exact (R_succ sb t w.length y).mpr ⟨x.1, x.2, walk_revR sf sb hrev2 t w x.1 hw, hrev2 y x.1 x.2 hx⟩

/-- A side whose frontier ran out has every node reachable from its root in its map. -/
theorem cov_all (st : Step ε) (r : Nat) (side : Side ε) (I : SideOK st r side) (he : side.frontier = []) :
    ∀ j x, R st r j x → ∃ v, side.seen x = some v
  | 0, x, h => by
    obtain ⟨k, l, hk, _⟩ := I.cov x 0 (Nat.zero_le _) h
    exact ⟨_, hk⟩
  | j + 1, x, h => by
    obtain ⟨u, e, hu, hx⟩ := (R_succ st r j x).mp h
    obtain ⟨⟨k, l⟩, hk⟩ := cov_all st r side I he j u hu
    have hkD := I.bound u k l hk
    have hne : k ≠ side.depth := by
      rintro rfl; have := I.all u l hk; rw [he] at this; simp at this
    have hR : R st r (k + 1) x := (R_succ st r k x).mpr ⟨u, e, entry_R st r _ I.inv k u l hk, hx⟩
    obtain ⟨k2, l2, hk2, _⟩ := I.cov x (k + 1) (by omega) hR
    exact ⟨_, hk2⟩

/-- **PROVEN** (`bfs_single_path`, optimality): for `source ≠ target` the returned path has no more
edges than any walk `source → target` along `sf`: the level is finished after the first meeting and
the shortest join kept, so the hop count is the least. `hrev2`: `sb` contains the converse of `sf`. -/
theorem bfsSingle_min (sf sb : Step ε) (hrev2 : ∀ u v e, (v, e) ∈ sf u → (u, e) ∈ sb v)
    (degF degB : Nat → Nat) (src tgt fuel : Nat) (hsd : src ≠ tgt) (es : List (Nat × ε))
    (h : bfsSingle sf sb degF degB src tgt fuel = some (some es)) :
    ∀ w, WalkTo sf src w tgt → es.length ≤ w.length := by
  intro w hw
  unfold bfsSingle at h
  rcases hl : loop sf sb degF degB fuel (Side.new degF src) (Side.new degB tgt) none with _ | ⟨f, b, _ | ⟨L, mt⟩⟩
  · rw [hl] at h; simp at h
  · rw [hl] at h; simp at h
  · rw [hl] at h
    simp only at h
    have I : LI sf sb src tgt f b (some (L, mt)) :=
      (loop_inv sf sb degF degB src tgt fuel _ _ none _ (LI_init sf sb degF degB src tgt hsd) hl).1
    obtain ⟨⟨a, la, bd, lb, hfa, hbb, hL⟩, hmin⟩ := I.bst
    cases hcf : chain f.seen fuel mt with
    | none => rw [hcf] at h; simp at h
    | some cf =>
      cases hcb : chain b.seen fuel mt with
      | none => rw [hcf, hcb] at h; simp at h
      | some cb =>
        rw [hcf, hcb] at h
        simp only [Option.some.injEq] at h
        subst h
        have f1 := (chain_ok sf src f.seen I.fok.inv fuel mt a la cf hfa hcf).1
        have b1 := (chain_ok sb tgt b.seen I.bok.inv fuel mt bd lb cb hbb hcb).1
        simp only [List.length_append, List.length_reverse, List.length_map, f1, b1]
        have ha := I.fok.bound mt a la hfa
        have hb := I.bok.bound mt bd lb hbb
        let i := min w.length f.depth
        have hsplit := List.take_append_drop i w
        rw [← hsplit] at hw
        obtain ⟨y, hw1, hw2⟩ := (walk_append sf src _ _ tgt).mp hw
        have hl1 : (w.take i).length = i := by simp [List.length_take]; omega
        have hl2 : (w.drop i).length = w.length - i := by simp
        obtain ⟨k1, l1, hk1, hk1i⟩ := I.fok.cov y i (by omega) ⟨_, hl1, hw1⟩
        by_cases hni : w.length - i ≤ b.depth
        · have hR := walk_revR sf sb hrev2 tgt _ y hw2
          rw [hl2] at hR
          obtain ⟨k2, l2, hk2, hk2i⟩ := I.bok.cov y _ hni hR
          have := hmin y k1 l1 k2 l2 hk1 hk2
          omega
        · omega

/-- **PROVEN** (`bfs_single_path`, completeness): for `source ≠ target`, `Ok(None)` only when no walk
`source → target` exists: the loop stops without a meeting only when a frontier ran out, and then
that side has seen everything reachable, the other side's root included. -/
theorem bfsSingle_none (sf sb : Step ε) (hrev2 : ∀ u v e, (v, e) ∈ sf u → (u, e) ∈ sb v)
    (degF degB : Nat → Nat) (src tgt fuel : Nat) (hsd : src ≠ tgt)
    (h : bfsSingle sf sb degF degB src tgt fuel = some none) :
    ∀ w, ¬ WalkTo sf src w tgt := by
  intro w hw
  unfold bfsSingle at h
  rcases hl : loop sf sb degF degB fuel (Side.new degF src) (Side.new degB tgt) none with _ | ⟨f, b, _ | ⟨L, mt⟩⟩
  · rw [hl] at h; simp at h
  · obtain ⟨I, hex⟩ := loop_inv sf sb degF degB src tgt fuel _ _ none _ (LI_init sf sb degF degB src tgt hsd) hl
    have hB : Bst f.seen b.seen none := I.bst
    rcases hex rfl with he | he
    · obtain ⟨v, hv⟩ := cov_all sf src f I.fok he w.length tgt ⟨w, rfl, hw⟩
      have := hB tgt v hv
      rw [I.bok.inv.root] at this; cases this
    · obtain ⟨v, hv⟩ := cov_all sb tgt b I.bok he w.length src (walk_revR sf sb hrev2 tgt w src hw)
      have := hB src _ I.fok.inv.root
      rw [hv] at this; cases this
  · rw [hl] at h
    simp only at h
    split at h <;> simp at h

end OpsTraverse.BfsSingle
