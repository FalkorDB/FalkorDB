import PendingCommit.AttrInv
import PendingCommit.Snapshot
/-!
# A reserved id has no staged existing attributes (the premise of `AttrInv.created_inv`)

`created_nodes(ids)` (pending.rs:557) adds ids to `created`, which routes later writes to
`new_nodes_attrs`. `AttrInv.created_inv` keeps the staged maps disjoint provided the id has no entry in
`existing_nodes_attrs`. The ids come from `IdSpace::reserve` (create.rs:176), modelled by
`Snapshot.createRust` (`Wf`: a reserved id is neither live nor freed by this segment, nor quarantined,
#3641). Here the staged-existing map joins that model:

* `ExLive` — every id with a staged existing entry is a live entity or one this segment deleted
  (`freed`): writes to `existing_*_attrs` only target a live entity, because `SET`/`REMOVE` skip a
  committed-deleted id (unless pending-created, which routes to `new`) and a pending-deleted one
  (ops_apply `SetRemove.setAttrItem`, `removeItem_spec`: the `is_node_deleted`/`is_node_deleted` guards,
  set.rs:196-201, remove.rs:117-122); that guard is `stageEx`'s premise;
* `del` keeps it (the id moves from `live` to `freed`), `commit` keeps it (`end_segment` clears the
  pending maps, `ex := []`), `createRust` keeps it (live only grows), `stageEx` keeps it (premise);
* **`reserved_not_ex`** — under `Wf` and `ExLive` a reserved id has no staged existing entry: the premise of
  `created_inv`, so the staged maps stay disjoint across `CREATE`.

Before #3641 this failed for the same reason #2876 did (a freed id reissued within the query): the id of a node
deleted in an earlier segment is a *free* id the next segment may take, but the pending maps are cleared at the
segment end, so even then `ExLive` holds; `reserved_not_ex` needs only `Wf`'s "not freed by this segment".
-/
namespace PendingCommit.Snapshot
open PendingCommit.PA

structure JS where
  rs : RS
  ex : FMap (List Entry)

/-- Every id with a staged existing entry is live or freed by this segment. -/
def ExLive (js : JS) : Prop :=
  ∀ id, (fget js.ex id).isSome = true → id ∈ keys js.rs.live ∨ id ∈ js.rs.freed

def delJS (js : JS) (id : Nat) : JS := { js with rs := del js.rs id }
/-- `Pending::commit` then `end_segment`: ids freed move to the bin and quarantine, the pending maps are cleared. -/
def commitJS (js : JS) : JS := { rs := commit js.rs, ex := [] }
def createJS (js : JS) (d : Nat) : JS × Nat := let r := createRust js.rs d; ({ js with rs := r.1 }, r.2)
/-- A staged write into `existing_*_attrs` for an id (premise: the id is live). -/
def stageEx (js : JS) (id : Nat) (l : List Entry) : JS := { js with ex := fset js.ex id l }

theorem exLive_del (js : JS) (h : ExLive js) (id : Nat) : ExLive (delJS js id) := by
  intro j hj
  simp only [delJS, del] at hj ⊢
  cases hf : find js.rs.live id with
  | none => simpa [hf] using h j hj
  | some e =>
    simp only [hf]
    rcases h j hj with hl | hf'
    · by_cases hji : j = id
      · right; simp [hji]
      · left
        simp only [keys, List.mem_map, List.mem_filter, bne_iff_ne, ne_eq] at hl ⊢
        obtain ⟨p, hp, rfl⟩ := hl
        exact ⟨p, ⟨hp, hji⟩, rfl⟩
    · right; simp [hf']

theorem exLive_commit (js : JS) : ExLive (commitJS js) := by
  intro j hj; simp [commitJS, fget] at hj

theorem exLive_create (js : JS) (h : ExLive js) (d : Nat) : ExLive (createJS js d).1 := by
  intro j hj
  simp only [createJS] at hj ⊢
  rcases h j hj with hl | hf
  · left
    unfold createRust
    split
    · simp only [keys, List.map_cons, List.mem_cons]; exact Or.inr hl
    · simp only [keys, List.map_cons, List.mem_cons]; exact Or.inr hl
  · right
    unfold createRust
    split <;> exact hf

theorem exLive_stageEx (js : JS) (h : ExLive js) (id : Nat) (l : List Entry) (hlive : id ∈ keys js.rs.live) :
    ExLive (stageEx js id l) := by
  intro j hj
  simp only [stageEx, fget_fset] at hj ⊢
  by_cases hji : j = id
  · left; subst hji; exact hlive
  · simp only [hji, ite_false] at hj; exact h j hj

/-- A reserved id is neither live nor freed by this segment. -/
theorem createRust_id_fresh (s : RS) (h : Wf s) (d : Nat) :
    (createRust s d).2 ∉ keys s.live ∧ (createRust s d).2 ∉ s.freed := by
  unfold createRust
  split
  · rename_i id rest hav
    have hm : id ∈ s.bin ∧ id ∉ s.quar := (mem_avail s id).1 (by rw [hav]; simp)
    exact ⟨fun hl => (h.live id hl).1 hm.1, fun hf => (h.freed id hf).1 hm.1⟩
  · constructor
    · intro hl; have := (h.live _ hl).2.2; omega
    · intro hf; have := (h.freed _ hf).2; omega

/-- **`reserved_not_ex`**: the id `reserve` hands `created_nodes` has no staged existing entry, so
`AttrInv.created_inv` applies to `CREATE`. -/
theorem reserved_not_ex (js : JS) (hw : Wf js.rs) (h : ExLive js) (d : Nat) :
    fget js.ex (createJS js d).2 = none := by
  have hf := createRust_id_fresh js.rs hw d
  show fget js.ex (createRust js.rs d).2 = none
  cases hg : fget js.ex (createRust js.rs d).2 with
  | none => rfl
  | some l =>
    have := h _ (by rw [hg]; rfl)
    rcases this with hl | hfr
    · exact absurd hl hf.1
    · exact absurd hfr hf.2

end PendingCommit.Snapshot
