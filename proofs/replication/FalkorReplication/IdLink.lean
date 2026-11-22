import FalkorReplication
import FalkorIdSpace.Callers
/-!
# `IdSpace::release`'s `live -= freed.len()` on the replica's delete, discharged

`Faithful.Ids.release1` subtracts in `Nat`; Rust subtracts in `u64` (`self.live -= freed.len()`,
`id_space.rs:739`). `deleteOne_refines` took `1 ≤ live` as a premise. `proofs/id_space` proves the
count can never wrap on a reachable id space (`release_no_underflow`, whose `freed ⊆ requested` premise
is discharged at all three call sites by `release_call_sites`); here the node delete is `release({id}, {id})`
(`Graph::delete_nodes`, `graph.rs:2320` passes one set twice), so `1 ≤ live` follows for any id space
satisfying id_space's `Inv` whose refusals pass, whether the id is below the entry boundary or was taken by
the batch (the case `deleteOne_refines_live` could not reach).
-/
namespace Falkor
open IdSpaceModel

/-- `|{id}|` over `[0, N)` is 1 when `id < N`. -/
theorem cnt_single (id : Nat) : ∀ N, id < N → cnt N (fun i => decide (i = id)) = 1
  | 0, h => by omega
  | N + 1, h => by
    rw [cnt_succ]
    by_cases e : id = N
    · subst e
      have : ∀ k, k ≤ id → cnt k (fun i => decide (i = id)) = 0 := by
        intro k
        induction k with
        | zero => intro _; rfl
        | succ k ih => intro hk; rw [cnt_succ, ih (by omega)]; simp; omega
      rw [this id (Nat.le_refl _)]; simp
    · rw [cnt_single id N (by omega)]; simp [Ne.symm e]

/-- **`1 ≤ live` for a live id of a reachable id space.** -/
theorem live_pos_of_release (N : Nat) (sp : IdSpace) (h : IdSpaceModel.Inv N sp) (id : Nat) (hid : id < N)
    (hok : ReleaseOk N sp (fun i => decide (i = id))) : 1 ≤ sp.live := by
  have := release_no_underflow N sp _ _ h hok (fun _ _ x => x)
  unfold sLen at this
  rwa [cnt_single id N hid] at this

/-- `deleteOne_refines` with its `1 ≤ live` premise discharged from id_space, for every id the delete
may name (below the entry boundary or created by this batch). -/
theorem deleteOne_refines_idspace (σ : GState) (s : FalkorFaithful.Ids) (N : Nat) (sp : IdSpace)
    (hI : IdSpaceModel.Inv N sp) (hlive : s.live = sp.live)
    (hb : σ.bound = s.bound) (hf : ∀ i, σ.free i = s.bin.contains i) (id : Nat)
    (hid : id < N) (hok : ReleaseOk N sp (fun i => decide (i = id))) (hnf : id ∉ s.bin) :
    (deleteOne σ id).bound = (s.release1 id).bound ∧
    ∀ i, (deleteOne σ id).free i = (s.release1 id).bin.contains i ∧
      (s.release1 id).bin.Nodup = s.bin.Nodup :=
  deleteOne_refines σ s hb hf id (hlive ▸ live_pos_of_release N sp hI id hid hok) hnf

end Falkor
