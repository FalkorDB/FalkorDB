import FalkorIndexLayer.LeafOps
/-
# Tree-level glue: `pack_branches`, `build_root`, `make_private` (`cow_btree/node.rs`)
# and `CowBTree::{default,new,from_sorted,insert_batch}` + the test walkers
# (`cow_btree/mod.rs`), origin/main 8743953a8.
-/
namespace IndexLayer.Leaf

inductive TNode where
  | leaf (l : LeafV)
  | branch (seps : List P) (children : List TNode)

mutual
def TNode.contents : TNode → List P
  | .leaf l => l.toPairs
  | .branch _ cs => TNode.contentsL cs
def TNode.contentsL : List TNode → List P
  | [] => []
  | c :: cs => c.contents ++ TNode.contentsL cs
end

/-- `Node::min`: walk the left spine to the first leaf. -/
def TNode.min : TNode → P
  | .leaf l => (l.key 0, l.doc 0)
  | .branch _ [] => (0, 0)
  | .branch _ (c :: _) => c.min

theorem contentsL_append (a b : List TNode) : TNode.contentsL (a ++ b) = TNode.contentsL a ++ TNode.contentsL b := by
  induction a with
  | nil => simp [TNode.contentsL]
  | cons x xs ih => simp [TNode.contentsL, ih]

/-- `pack_branches`: branches of at most `BRANCH_MAX` children; a remainder of
exactly `BRANCH_MAX + 1` is split `BRANCH_MAX - 1` + 2 so no branch is a singleton. -/
def takeOf (bmax len : Nat) : Nat := if len = bmax + 1 then bmax - 1 else min len bmax

def packBranches (bmax : Nat) (rest : List TNode) : List TNode :=
  if h : rest = [] then [] else
  if ht : takeOf bmax rest.length = 0 then [.branch [] rest] else
  .branch ((rest.take (takeOf bmax rest.length)).tail.map TNode.min) (rest.take (takeOf bmax rest.length)) ::
    packBranches bmax (rest.drop (takeOf bmax rest.length))
termination_by rest.length
decreasing_by
  simp_wf
  have : rest.length ≠ 0 := by simpa using h
  have ht' : takeOf bmax rest.length ≠ 0 := ht
  omega

theorem takeOf_bounds (bmax len : Nat) (hb : 3 ≤ bmax) (hl : 2 ≤ len) :
    2 ≤ takeOf bmax len ∧ takeOf bmax len ≤ bmax ∧ takeOf bmax len ≤ len ∧
    (len - takeOf bmax len = 0 ∨ 2 ≤ len - takeOf bmax len) := by
  unfold takeOf; split <;> omega

theorem packBranches_contents (bmax : Nat) : ∀ (rest : List TNode),
    TNode.contentsL (packBranches bmax rest) = TNode.contentsL rest := by
  intro rest
  induction h : rest.length using Nat.strongRecOn generalizing rest with
  | _ n ih =>
    rw [packBranches]
    split
    · next he => subst he; rfl
    · next he =>
      split
      · simp [TNode.contentsL, TNode.contents]
      · next ht =>
        have hlen : rest.length ≠ 0 := by simpa using he
        rw [TNode.contentsL, TNode.contents]
        rw [ih _ (by simp; omega) _ rfl, ← contentsL_append, List.take_append_drop]

/-- Every packed branch has between 2 and `BRANCH_MAX` children (for `BRANCH_MAX >= 3`
and at least two inputs) — the invariant `rebalance` relies on. -/
theorem packBranches_arity (bmax : Nat) (hb : 3 ≤ bmax) : ∀ (rest : List TNode), 2 ≤ rest.length →
    ∀ t ∈ packBranches bmax rest, ∃ seps cs, t = .branch seps cs ∧ 2 ≤ cs.length ∧ cs.length ≤ bmax := by
  intro rest
  induction h : rest.length using Nat.strongRecOn generalizing rest with
  | _ n ih =>
    intro h2 t ht
    obtain ⟨b1, b2, b3, b4⟩ := takeOf_bounds bmax rest.length hb (by omega)
    rw [packBranches] at ht
    split at ht
    · simp at ht
    · split at ht
      · omega
      · rcases List.mem_cons.mp ht with rfl | ht
        · exact ⟨_, _, rfl, by simp; omega, by simp; omega⟩
        · rcases b4 with hz | hz
          · have : (rest.drop (takeOf bmax rest.length)).length = 0 := by simp; omega
            rw [List.length_eq_zero_iff.mp this, packBranches] at ht; simp at ht
          · exact ih _ (by simp; omega) _ rfl (by simp; omega) t ht

def TNode.kids : TNode → List TNode
  | .leaf _ => []
  | .branch _ cs => cs

theorem packBranches_kids (bmax : Nat) : ∀ (rest : List TNode), (packBranches bmax rest).flatMap TNode.kids = rest := by
  intro rest
  induction h : rest.length using Nat.strongRecOn generalizing rest with
  | _ n ih =>
    rw [packBranches]
    split
    · next he => subst he; rfl
    · next he =>
      split
      · simp [TNode.kids]
      · next ht =>
        have hlen : rest.length ≠ 0 := by simpa using he
        rw [List.flatMap_cons, ih _ (by simp; omega) _ rfl]
        simp [TNode.kids]

theorem packBranches_shrinks (bmax : Nat) (hb : 3 ≤ bmax) (rest : List TNode) (h2 : 2 ≤ rest.length) :
    (packBranches bmax rest).length < rest.length := by
  have hk := packBranches_kids bmax rest
  have ha := packBranches_arity bmax hb rest h2
  have : 2 * (packBranches bmax rest).length ≤ ((packBranches bmax rest).flatMap TNode.kids).length := by
    generalize packBranches bmax rest = ps at ha
    induction ps with
    | nil => simp
    | cons t ts ih =>
      obtain ⟨seps, cs, rfl, h1, -⟩ := ha t (by simp)
      simp [TNode.kids] at ih ⊢
      have := ih (fun t' ht' => ha t' (by simp [ht']))
      omega
  rw [hk] at this
  have hpos : 0 < (packBranches bmax rest).length := by
    rw [packBranches]; split
    · next he => subst he; simp at h2
    · split <;> simp
  omega

/-- `build_root`, with fuel for its `while fragments.len() > 1` loop. -/
def buildRoot (bmax : Nat) : Nat → List TNode → TNode
  | 0, frags => (frags.getLast?).getD (.leaf (fromPairs []))
  | f + 1, frags => if frags.length > 1 then buildRoot bmax f (packBranches bmax frags)
                    else (frags.getLast?).getD (.leaf (fromPairs []))

theorem fromPairs_nil : (fromPairs []).toPairs = [] :=
  fromPairs_roundtrip [] List.Pairwise.nil (fun _ h => by simp at h) (by simp)

/-- **PROVEN** (`build_root` keeps every entry, in order): with `BRANCH_MAX >= 3`
the loop terminates within `fragments.len()` rounds and the root holds the
fragments' entries; no fragments gives an empty leaf. -/
theorem buildRoot_contents (bmax : Nat) (hb : 3 ≤ bmax) : ∀ (f : Nat) (frags : List TNode), frags.length ≤ f + 1 →
    (buildRoot bmax f frags).contents = TNode.contentsL frags
  | 0, frags, hf => by
    simp only [buildRoot]
    match frags, hf with
    | [], _ => simp [TNode.contentsL, fromPairs_nil, TNode.contents]
    | [x], _ => simp [TNode.contentsL]
  | f + 1, frags, hf => by
    simp only [buildRoot]
    split
    · next h1 =>
      rw [buildRoot_contents bmax hb f _ (by have := packBranches_shrinks bmax hb frags (by omega); omega),
        packBranches_contents]
    · match frags with
      | [] => simp [TNode.contentsL, fromPairs_nil, TNode.contents]
      | [x] => simp [TNode.contentsL]
      | _ :: _ :: _ => simp at *

/-- `make_private`: replace the `Arc` with a fresh clone (always, unlike
`Arc::make_mut`). Modelled as a versioned store: the new version is a copy, and
the old version a reader holds is untouched by any later write to the copy. -/
def makePrivate {α : Type} (store : List α) (i : Nat) (hi : i < store.length) : List α × Nat :=
  (store ++ [store[i]], store.length)

/-- **PROVEN** (copy-on-write): the private copy starts equal to the shared
node, and any write through it leaves every existing version unchanged. -/
theorem makePrivate_cow {α : Type} (store : List α) (i : Nat) (hi : i < store.length) (v : α) :
    makePrivate store i hi = (store ++ [store[i]], store.length) ∧
    (store ++ [store[i]])[store.length]'(by simp) = store[i] ∧
    ∀ j (hj : j < store.length), ((store ++ [store[i]]).set store.length v)[j]'(by simp; omega) = store[j] := by
  refine ⟨rfl, by simp, fun j hj => ?_⟩
  rw [List.getElem_set_ne (by omega), List.getElem_append_left hj]

/-! ## `CowBTree` entry points (`cow_btree/mod.rs`) -/

/-- `CowBTree::default` / `new`: a single empty leaf. -/
def emptyTree : TNode := .leaf (fromPairs [])

theorem emptyTree_contents : emptyTree.contents = [] := fromPairs_nil

/-- `CowBTree::from_sorted` (the `assert!` on sorted-unique input is a precondition). -/
def fromSorted (leafMax bmax : Nat) (ps : List P) : TNode :=
  if ps = [] then emptyTree
  else buildRoot bmax (chunks leafMax ps).length ((chunks leafMax ps).map (fun c => .leaf (fromPairs c)))

theorem contentsL_leaves (cs : List (List P)) (hc : ∀ c ∈ cs, (fromPairs c).toPairs = c) :
    TNode.contentsL (cs.map (fun c => TNode.leaf (fromPairs c))) = cs.flatten := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.map_cons, TNode.contentsL, TNode.contents, List.flatten_cons]
    rw [hc c (by simp), ih (fun c' h => hc c' (by simp [h]))]

/-- **PROVEN** (`from_sorted`): the bulk-built tree holds exactly the input. -/
theorem fromSorted_contents (leafMax bmax : Nat) (h1 : 0 < leafMax) (h2 : leafMax ≤ 256) (hb : 3 ≤ bmax)
    (ps : List P) (hs : Sorted2 ps) (hu : AllU64 ps) : (fromSorted leafMax bmax ps).contents = ps := by
  unfold fromSorted
  split
  · next he => subst he; exact emptyTree_contents
  · obtain ⟨hfl, hsub⟩ := chunks_spec leafMax h1 ps
    rw [buildRoot_contents bmax hb _ _ (by simp), contentsL_leaves _ (fun c hc =>
      fromPairs_roundtrip c (hs.sublist (hsub c hc).2) (fun p hp => hu p ((hsub c hc).2.subset hp))
        (by have := (hsub c hc).1; omega)), hfl]

/-- `CowBTree::insert_batch`: empty batch is a no-op; otherwise rebuild the root
from `apply_batch`'s fragments. `apply` is `Node::apply_batch` (proved in
`proofs/cow_btree`); `fuel` bounds the `build_root` loop. -/
def insertBatch (bmax fuel : Nat) (apply : TNode → List P → List TNode) (root : TNode) (batch : List P) : TNode :=
  if batch = [] then root else buildRoot bmax fuel (apply root batch)

/-- **PROVEN** (`insert_batch` = `build_root` ∘ `apply_batch`): the new root holds
exactly what `apply_batch`'s fragments hold. -/
theorem insertBatch_contents (bmax fuel : Nat) (hb : 3 ≤ bmax) (apply : TNode → List P → List TNode)
    (root : TNode) (batch : List P) (hf : (apply root batch).length ≤ fuel + 1) :
    (insertBatch bmax fuel apply root batch).contents =
      if batch = [] then root.contents else TNode.contentsL (apply root batch) := by
  unfold insertBatch
  split
  · rfl
  · exact buildRoot_contents bmax hb fuel _ hf

-- The test helpers `len()` (inner `count`) and `leaves()` (inner `walk`).
mutual
def TNode.count : TNode → Nat
  | .leaf l => l.count
  | .branch _ cs => TNode.countL cs
def TNode.countL : List TNode → Nat
  | [] => 0
  | c :: cs => c.count + TNode.countL cs
end

mutual
def TNode.walk : TNode → List LeafV
  | .leaf l => [l]
  | .branch _ cs => TNode.walkL cs
def TNode.walkL : List TNode → List LeafV
  | [] => []
  | c :: cs => c.walk ++ TNode.walkL cs
end

mutual
theorem walk_spec : ∀ t : TNode, (t.walk.flatMap LeafV.toPairs = t.contents) ∧ (t.walk.map LeafV.count).sum = t.count
  | .leaf l => by simp [TNode.walk, TNode.contents, TNode.count]
  | .branch _ cs => by
    have := walkL_spec cs
    simpa [TNode.walk, TNode.contents, TNode.count] using this
theorem walkL_spec : ∀ cs : List TNode,
    ((TNode.walkL cs).flatMap LeafV.toPairs = TNode.contentsL cs) ∧ ((TNode.walkL cs).map LeafV.count).sum = TNode.countL cs
  | [] => by simp [TNode.walkL, TNode.contentsL, TNode.countL]
  | c :: cs => by
    obtain ⟨a1, a2⟩ := walk_spec c
    obtain ⟨b1, b2⟩ := walkL_spec cs
    simp [TNode.walkL, TNode.contentsL, TNode.countL, List.flatMap_append, a1, b1, ← a2, ← b2]
end

end IndexLayer.Leaf
