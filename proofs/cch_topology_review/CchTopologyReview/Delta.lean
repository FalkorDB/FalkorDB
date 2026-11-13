import CchTopologyReview.Props
import FalkorCowBTree

/-!
# The deltas are `cow_btree`s (03 §2): refinement, row reads, snapshots

03 §2: "`Up.delta`, `Down.delta`: `CowBTree`: key `(row ≪ 32) | col`, doc = id". This file ties the abstract
delta (`List Ent`, key-sorted) to `proofs/cow_btree`'s model of `graph/src/index/falkordb/data_structures/
cow_btree/` on main:

* `delta_insertBatch`: `CowBTree::insert_batch` (`mod.rs:241-255`) of a key-sorted batch with fresh keys
  leaves exactly the abstract `insAll` (via cow_btree's `insertBatch_wf`), and keeps `TreeWF`. That covers
  S2's `Up` batch, which is sorted by key. `delta_insertBatch_sorted`: any fresh, key-distinct batch, sorted
  by key first (`insAll [] b`), leaves exactly `insAll d b`. That covers S2's `Down` batch, which is not
  sorted (T1); `s2_batches` gives both batches' premises.
* `delta_row_read`: a `range_tuples` read of keys `[r ≪ 32, r ≪ 32 | u32::MAX]` (`mod.rs:408`; cow_btree's
  `rangeTuples_spec`) is exactly `dRow`, the delta row that `up_ids` merges: the "delta cursor" of 03 §6b
  step 3. **PR 3 note**: the cursor needs `range_tuples`, which yields `(key, doc)`. `CowBTree::range`
  (`mod.rs:314`) yields docs only (cow_btree's `rangeDocs_spec`), so it cannot give the column. On main,
  `range_tuples` says "No caller today".
* `delta_snapshot_isolated`, `delta_path_copy`: cow_btree's `snapshot_reads_committed` (which is its
  `range_spec`) and `insertOne_shares_siblings`. In the value model a reader's root is a value, so its reads
  cannot change. That the Rust never writes a shared page in place is cow_btree's Snapshot.lean argument
  (argued, with `make_private` proven in proofs/index_layer), not a theorem here.
* `down_batch_unsorted`, `fold_concat_unsorted`: **finding T1** (design-level). See the root file.

Bounds (`Fits`): a rank is a `NonZeroU32` (03 §2) and an id is a `u32` (`Ax: Arc<[u32]>`), so row, column
and id are below `2^32`; the key `(row ≪ 32) | col` is then `row * 2^32 + col` (`key_eq`).
-/

namespace CchTopo.Delta

open CchTopo

/-- `2^32`. -/
@[irreducible] def K : Nat := 4294967296

/-- `(row ≪ 32) | col`. -/
def keyE (e : Ent) : Nat := e.row * K + e.col

/-- The `(key, doc)` tuple an entry is stored as, in cow_btree's encoding. -/
def encE (e : Ent) : Nat := CowBTree.enc (keyE e) e.id

/-- The u32 bounds. -/
def Fits (e : Ent) : Prop := e.row < K ∧ e.col < K ∧ e.id < K

/-- `(row ≪ 32) | col` is `row * 2^32 + col` when `col < 2^32`. -/
theorem key_eq (row col : Nat) (hc : col < K) : (row <<< 32) ||| col = row * K + col := by
  have h32 : (2 : Nat) ^ 32 = 4294967296 := rfl
  rw [← Nat.shiftLeft_add_eq_or_of_lt (i := 32) (by rw [h32]; rw [K] at hc; exact hc), Nat.shiftLeft_eq, h32, K]

theorem fits_doc (e : Ent) (h : Fits e) : e.id < CowBTree.W := by
  unfold Fits K at h; unfold CowBTree.W; omega

theorem encE_lt {a b : Ent} (ha : Fits a) (hb : Fits b) (h : keyLt a b) : encE a < encE b := by
  unfold encE
  rw [CowBTree.enc_lt_iff _ _ _ _ (fits_doc a ha) (fits_doc b hb)]
  left
  unfold keyE; unfold Fits at ha hb; unfold keyLt at h
  rcases h with h | ⟨h1, h2⟩
  · have : a.row + 1 ≤ b.row := h
    have : (a.row + 1) * K ≤ b.row * K := Nat.mul_le_mul_right K this
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · rw [h1]; omega

theorem keyOf_encE (e : Ent) (h : Fits e) : CowBTree.keyOf (encE e) = keyE e :=
  CowBTree.keyOf_enc _ _ (fits_doc e h)

theorem docOf_encE (e : Ent) (h : Fits e) : CowBTree.docOf (encE e) = e.id :=
  CowBTree.docOf_enc _ _ (fits_doc e h)

theorem encE_inj {a b : Ent} (ha : Fits a) (hb : Fits b) (h : encE a = encE b) : a = b := by
  have k := keyOf_encE a ha; have k' := keyOf_encE b hb
  have d := docOf_encE a ha; have d' := docOf_encE b hb
  rw [h] at k d
  have hk : keyE a = keyE b := k.symm.trans k'
  have hid : a.id = b.id := d.symm.trans d'
  unfold keyE at hk; unfold Fits K at ha hb
  have hr : a.row = b.row := by
    rcases Nat.lt_trichotomy a.row b.row with h | h | h
    · have : (a.row + 1) * 4294967296 ≤ b.row * 4294967296 := Nat.mul_le_mul_right _ h
      rw [Nat.add_mul, Nat.one_mul] at this; unfold K at hk; omega
    · exact h
    · have : (b.row + 1) * 4294967296 ≤ a.row * 4294967296 := Nat.mul_le_mul_right _ h
      rw [Nat.add_mul, Nat.one_mul] at this; unfold K at hk; omega
  have hc : a.col = b.col := by rw [hr] at hk; unfold K at hk; omega
  cases a; cases b; simp_all

theorem K_mul_K : K * K = CowBTree.W := by unfold K CowBTree.W; rfl

theorem keyE_lt_W (e : Ent) (h : Fits e) : keyE e < CowBTree.W := by
  have h1 : (e.row + 1) * K ≤ K * K := Nat.mul_le_mul_right K h.1
  rw [Nat.add_mul, Nat.one_mul, K_mul_K] at h1
  unfold keyE
  have := h.2.1
  omega

theorem encE_lt_HI (e : Ent) (h : Fits e) : encE e < CowBTree.HI := by
  have hk := keyE_lt_W e h
  have hd := fits_doc e h
  have h1 : (keyE e + 1) * CowBTree.W ≤ CowBTree.W * CowBTree.W := Nat.mul_le_mul_right _ hk
  rw [Nat.add_mul, Nat.one_mul] at h1
  unfold encE CowBTree.enc CowBTree.HI
  omega

theorem sorted_map_encE {l : List Ent} (hl : KSorted l) (hf : ∀ e ∈ l, Fits e) : CowBTree.Sorted (l.map encE) := by
  unfold CowBTree.Sorted
  rw [List.pairwise_map]
  exact hl.imp_of_mem (fun ha hb h => encE_lt (hf _ ha) (hf _ hb) h)

/-- **`insert_batch` on a delta is the abstract insert.** A key-sorted batch of fresh keys, encoded, is a
    sorted batch (`insert_batch`'s `assert!`, `mod.rs:250-253`, passes), and the tree afterwards holds
    exactly `insAll d b`, encoded, and stays well formed. -/
theorem delta_insertBatch (c : CowBTree.Cfg) (h : Nat) (t : CowBTree.Node) (d b : List Ent)
    (hw : CowBTree.TreeWF c h t) (ht : CowBTree.toList h t = d.map encE)
    (hd : KSorted d) (hb : KSorted b) (hfresh : ∀ x ∈ b, ∀ y ∈ d, ¬ sameKey x y)
    (hfd : ∀ e ∈ d, Fits e) (hfb : ∀ e ∈ b, Fits e) :
    CowBTree.SortedLe (b.map encE) ∧
    CowBTree.TreeWF c (CowBTree.insertBatch c h t (b.map encE)).2 (CowBTree.insertBatch c h t (b.map encE)).1 ∧
    CowBTree.toList (CowBTree.insertBatch c h t (b.map encE)).2 (CowBTree.insertBatch c h t (b.map encE)).1 =
      (insAll d b).map encE := by
  have hsb : CowBTree.SortedLe (b.map encE) := (sorted_map_encE hb hfb).imp Nat.le_of_lt
  have hlt : ∀ e ∈ b.map encE, e < CowBTree.HI := by
    intro e he; obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he; exact encE_lt_HI x (hfb x hx)
  obtain ⟨hw', hs', hm⟩ := CowBTree.insertBatch_wf c h t (b.map encE) hw hsb hlt
  refine ⟨hsb, hw', CowBTree.sorted_ext hs' ?_ (fun e => ?_)⟩
  · refine sorted_map_encE (ksorted_insAll b d hd) (fun e he => ?_)
    rcases mem_insAll_sub b d e he with h | h
    · exact hfd e h
    · exact hfb e h
  · have hb' : b.Pairwise (fun a c => ¬ sameKey a c) := hb.imp (fun h hs => keyLt_irrefl_of_same hs h)
    rw [hm, ht]
    simp only [List.mem_map]
    constructor
    · rintro (⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩)
      · exact ⟨x, (mem_insAll_fresh b d hb' hfresh x).2 (Or.inl hx), rfl⟩
      · exact ⟨x, (mem_insAll_fresh b d hb' hfresh x).2 (Or.inr hx), rfl⟩
    · rintro ⟨x, hx, rfl⟩
      rcases (mem_insAll_fresh b d hb' hfresh x).1 hx with h | h
      · exact Or.inl ⟨x, h, rfl⟩
      · exact Or.inr ⟨x, h, rfl⟩

/-- **Any fresh, key-distinct batch, sorted first, is the abstract insert** (S2's `Down` batch, T1's fix):
    `insert_batch` of `insAll [] b`, encoded, leaves exactly `insAll d b`. -/
theorem delta_insertBatch_sorted (c : CowBTree.Cfg) (h : Nat) (t : CowBTree.Node) (d b : List Ent)
    (hw : CowBTree.TreeWF c h t) (ht : CowBTree.toList h t = d.map encE)
    (hd : KSorted d) (hb : b.Pairwise (fun a c => ¬ sameKey a c)) (hfresh : ∀ x ∈ b, ∀ y ∈ d, ¬ sameKey x y)
    (hfd : ∀ e ∈ d, Fits e) (hfb : ∀ e ∈ b, Fits e) :
    CowBTree.SortedLe ((insAll [] b).map encE) ∧
    CowBTree.TreeWF c (CowBTree.insertBatch c h t ((insAll [] b).map encE)).2
      (CowBTree.insertBatch c h t ((insAll [] b).map encE)).1 ∧
    CowBTree.toList (CowBTree.insertBatch c h t ((insAll [] b).map encE)).2
      (CowBTree.insertBatch c h t ((insAll [] b).map encE)).1 = (insAll d b).map encE := by
  have hm0 : ∀ e, e ∈ insAll [] b ↔ e ∈ b := fun e => by
    rw [mem_insAll_fresh b [] hb (by simp)]; simp
  obtain ⟨h1, h2, h3⟩ := delta_insertBatch c h t d (insAll [] b) hw ht hd (ksorted_insAll b [] (by simp [KSorted]))
    (fun x hx => hfresh x ((hm0 x).1 hx)) hfd (fun e he => hfb e ((hm0 e).1 he))
  exact ⟨h1, h2, by rw [h3, insAll_sorted_batch d b hd hb hfresh]⟩

/-- Decode a `(key, doc)` tuple back to `(col, id)`. -/
def dec (p : Nat × Nat) : RE := (p.1 % K, p.2)

/-- A key is in row `r`'s range exactly when its row is `r`. -/
theorem in_row_iff (e : Ent) (hf : Fits e) (r : Nat) :
    (r * K ≤ keyE e ∧ keyE e ≤ r * K + (K - 1)) ↔ e.row = r := by
  unfold keyE
  have hc := hf.2.1
  constructor
  · rintro ⟨h1, h2⟩
    rcases Nat.lt_trichotomy e.row r with h | h | h
    · have : (e.row + 1) * K ≤ r * K := Nat.mul_le_mul_right _ h
      rw [Nat.add_mul, Nat.one_mul] at this; omega
    · exact h
    · have : (r + 1) * K ≤ e.row * K := Nat.mul_le_mul_right _ h
      rw [Nat.add_mul, Nat.one_mul] at this; omega
  · intro h; subst h; omega

theorem dec_encE (e : Ent) (hf : Fits e) :
    dec (CowBTree.keyOf (encE e), CowBTree.docOf (encE e)) = (e.col, e.id) := by
  unfold dec
  rw [keyOf_encE e hf, docOf_encE e hf]
  unfold keyE
  rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hf.2.1]

/-- **A delta row read is a `range_tuples` of the tree** (`CowBTree::range_tuples`, `mod.rs:408` @main; the
    `(key, doc)` cursor, cow_btree's `rangeTuples_spec`): keys `[r ≪ 32, (r ≪ 32) | u32::MAX]`, decoded, are
    exactly `dRow d r`, in the same order. -/
theorem delta_row_read (c : CowBTree.Cfg) (h : Nat) (t : CowBTree.Node) (d : List Ent)
    (hw : CowBTree.TreeWF c h t) (ht : CowBTree.toList h t = d.map encE) (hf : ∀ e ∈ d, Fits e)
    (r n : Nat) (hn : d.length ≤ n) :
    (CowBTree.rangeWith CowBTree.TupleExtract h t (r * K) (r * K + (K - 1)) n).map dec = dRow d r := by
  have hl : (CowBTree.toList h t).length ≤ n := by rw [ht, List.length_map]; exact hn
  rw [CowBTree.rangeTuples_spec c h t (r * K) (r * K + (K - 1)) n hw hl, ht, List.filter_map, List.map_map,
    List.map_map]
  unfold dRow
  have hfil : d.filter ((fun e => decide (r * K ≤ CowBTree.keyOf e ∧ CowBTree.keyOf e ≤ r * K + (K - 1))) ∘ encE) =
      d.filter (fun e => decide (e.row = r)) := by
    apply List.filter_congr
    intro e he
    rw [Function.comp_apply, keyOf_encE e (hf e he)]
    exact decide_eq_decide.2 (in_row_iff e (hf e he) r)
  rw [hfil]
  apply List.map_congr_left
  intro e he
  rw [Function.comp_apply, Function.comp_apply]
  exact dec_encE e (hf e (List.mem_filter.1 he).1)

/-- **Snapshots of a delta**: a reader's root reads exactly its own entries in range. This is cow_btree's
    `snapshot_reads_committed`, which is its `range_spec`: in the value model the writer's operations never
    reach the reader's root. That the Rust never writes a shared page in place is cow_btree's argued Snapshot
    fact (its Snapshot.lean), on which this rests. -/
theorem delta_snapshot_isolated (c : CowBTree.Cfg) (t : CowBTree.Node × Nat) (lo hi n : Nat)
    (hw : CowBTree.TreeWF c t.2 t.1) (hn : (CowBTree.toList t.2 t.1).length ≤ n) :
    CowBTree.range t.2 t.1 lo hi n =
      (CowBTree.toList t.2 t.1).filter (fun e => decide (lo ≤ CowBTree.keyOf e ∧ CowBTree.keyOf e ≤ hi)) :=
  CowBTree.snapshot_reads_committed c t [] lo hi n hw hn

/-- **Path copying** (cow_btree's `insertOne_shares_siblings`): an insert that does not split replaces only
    the routed child and shares every other child. -/
theorem delta_path_copy (c : CowBTree.Cfg) (h : Nat) (seps : List Nat) (cs : List CowBTree.Node) (x : Nat)
    (c' : CowBTree.Node)
    (hc : (CowBTree.insertOne c h (cs.getD (CowBTree.childIndex seps x) default) x).2 = none)
    (hc' : (CowBTree.insertOne c h (cs.getD (CowBTree.childIndex seps x) default) x).1 = c') :
    CowBTree.insertOne c (h + 1) (.branch seps cs) x =
      (.branch seps (CowBTree.setAt cs (CowBTree.childIndex seps x) c'), none) :=
  CowBTree.insertOne_shares_siblings c h seps cs x c' hc hc'

/-! ## Finding T1 (design-level, no code to reproduce) -/

theorem encE_val (e : Ent) : encE e = (e.row * 4294967296 + e.col) * 18446744073709551616 + e.id := by
  unfold encE keyE CowBTree.enc CowBTree.W; rw [K]

/-- S2 sorts its new arcs by `(lo, hi)` (03 §6). Their `Down` entries, keyed `(hi ≪ 32) | lo`, are then not
    sorted: `insert_batch`'s `assert!` (`cow_btree/mod.rs:250-253` @main) would panic on them. This matters
    only if PR 3 inserts with `insert_batch` (or builds with `from_sorted`); 03's costs ("`cow_btree` inserts
    (O(new arcs × log n))", §6b) read as single `insert`s, which take any order. Here
    `N = [(0, 3), (1, 2)]`, sorted by `(lo, hi)`; Down's keys are `(3, 0)` then `(2, 1)`. -/
theorem down_batch_unsorted :
    [((0 : Nat), (3 : Nat)), (1, 2)].Pairwise (fun p q => p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 < q.2)) ∧
    ¬ CowBTree.SortedLe (((upEnts 0 [(0, 3), (1, 2)]).map tr).map encE) := by
  refine ⟨by decide, ?_⟩
  simp only [upEnts, List.map, tr, encE_val]
  decide

/-- The fold's new deltas "inserted into empty `cow_btree`s" from `added_since_fold`: two passes, each sorted,
    concatenate unsorted, which `from_sorted`'s `assert!` (`mod.rs:204-207`) and `insert_batch`'s refuse. -/
theorem fold_concat_unsorted :
    ¬ CowBTree.SortedLe ((ents [(5, upEnts 10 [(1, 2)]), (6, upEnts 11 [(0, 3)])]).map encE) := by
  have he : ents [(5, upEnts 10 [(1, 2)]), (6, upEnts 11 [(0, 3)])] = [⟨1, 2, 10⟩, ⟨0, 3, 11⟩] := rfl
  rw [he]
  simp only [List.map, encE_val]
  decide

end CchTopo.Delta
