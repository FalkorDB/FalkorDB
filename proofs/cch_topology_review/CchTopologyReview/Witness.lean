import CchTopologyReview.Findings
import CchTopologyReview.Chunked

/-!
# The premises are satisfiable together (a witness)

A fresh build of 4 ranks (ranks 0, 1, 2 held by nodes 10, 11, 12; rank 3 a pool rank), with game rows
`[[1, 2], [2], [], []]`, so arcs `(0,1)`, `(0,2)`, `(1,2)` with ids 0, 1, 2:

* `w_fresh_ok`, `w_transpose`, `w_inv0`: `FreshOk`, `TransposeSpec` and `fresh_inv` hold.
* `w_alloc`: a writer forks, and the allocation gives node 13 (degree 1) the pool rank 3.
* `w_s2ok`: S2 adds arc `(2, 3)`; `S2Ok` holds and the arc gets id 3 (`w_s2_id`).
* `w_lin`, `w_inv2`: the state is reached by `Lin` and satisfies `Inv`.
* `w_addUp`, `w_addDown`: `AddSpec` holds for the fold's bases `[[(1,0),(2,1)],[(2,2)],[(3,3)],[]]` and their
  transpose.
* `w_fold_rows`: `fold_rows` instantiates on it.
* `w_chunked`: a `WF` `WeightArr` state with two versions sharing one table, on which `setW` succeeds (both
  the table and the chunk are copied); the other version still reads 2, the written one reads 7.
-/

namespace CchTopo.Witness

open CchTopo

def fi : FreshIn := { ver := 0, reserve := 0, holders := [some 10, some 11, some 12, none], pat := [[1, 2], [2], [], []] }

def up0 : Rows := [[(1, 0), (2, 1)], [(2, 2)], [], []]
def down0 : Rows := [[], [(0, 0)], [(0, 1), (1, 2)], []]

theorem w_freshRows : freshRows 0 fi.pat = up0 := rfl

theorem mem_up0 (r c i : Nat) : (c, i) ∈ bRow up0 r ↔
    (r = 0 ∧ c = 1 ∧ i = 0) ∨ (r = 0 ∧ c = 2 ∧ i = 1) ∨ (r = 1 ∧ c = 2 ∧ i = 2) := by
  match r with
  | 0 => simp [bRow, up0]
  | 1 => simp [bRow, up0]
  | 2 => simp [bRow, up0]
  | 3 => simp [bRow, up0]
  | n + 4 => simp [bRow, up0]

theorem mem_down0 (r c i : Nat) : (c, i) ∈ bRow down0 r ↔
    (r = 1 ∧ c = 0 ∧ i = 0) ∨ (r = 2 ∧ c = 0 ∧ i = 1) ∨ (r = 2 ∧ c = 1 ∧ i = 2) := by
  match r with
  | 0 => simp [bRow, down0]
  | 1 => simp [bRow, down0]
  | 2 => simp [bRow, down0]
  | 3 => simp [bRow, down0]
  | n + 4 => simp [bRow, down0]

theorem holders_iff (r x : Nat) : fi.holders[r]? = some (some x) ↔ (r = 0 ∧ x = 10) ∨ (r = 1 ∧ x = 11) ∨ (r = 2 ∧ x = 12) := by
  match r with
  | 0 => simp [fi]; omega
  | 1 => simp [fi]; omega
  | 2 => simp [fi]; omega
  | 3 => simp [fi]
  | n + 4 => simp [fi]

theorem w_fresh_ok : FreshOk fi := by
  refine ⟨fun r s x hr hs => ?_, fun r => ?_, fun r h hh => ?_⟩
  · rw [holders_iff] at hr hs; omega
  · match r with
    | 0 => simp [fi]
    | 1 => simp [fi]
    | 2 => simp [fi]
    | 3 => simp [fi]
    | n + 4 => simp [fi]
  · match r, hh with
    | 0, hh => simp [fi] at hh; rcases hh with rfl | rfl <;> simp [fi]
    | 1, hh => simp [fi] at hh; subst hh; simp [fi]
    | 2, hh => simp [fi] at hh
    | 3, hh => simp [fi] at hh
    | n + 4, hh => simp [fi] at hh

theorem w_transpose : TransposeSpec (freshRows 0 fi.pat) down0 := by
  rw [w_freshRows]
  refine ⟨fun ⟨r, c, i⟩ => ?_, fun r => ?_⟩
  · simp only [inB, tr]
    rw [mem_up0, mem_down0]; omega
  · match r with
    | 0 => simp [bRow, down0, StrictCol]
    | 1 => simp [bRow, down0, StrictCol]
    | 2 => simp [bRow, down0, StrictCol]
    | 3 => simp [bRow, down0, StrictCol]
    | n + 4 => simp [bRow, down0, StrictCol]

def t0 : Cch Nat := fresh 0 fi down0

theorem w_inv0 : Inv t0 := fresh_inv 0 fi down0 w_fresh_ok w_transpose

/-- A writer forks and allocates node 13 (degree 1). -/
def t1 : Cch Nat := alloc (fork t0) [(13, 1)]

theorem w_alloc : t1.rankOf 13 = some 3 ∧ t1.nodeOf 3 = some 13 ∧ t1.free = [] ∧ t1.nodeOf 2 = some 12 := by
  refine ⟨rfl, rfl, rfl, rfl⟩

theorem t1_upB : t1.upB = up0 := rfl
theorem t1_upD : t1.upD = [] := rfl

theorem w_s2ok : S2Ok t1 [(2, 3)] := by
  refine ⟨by simp [PLt], by simp, ?_, ?_⟩
  · intro p hp i hu
    simp only [List.mem_singleton] at hp
    subst hp
    rcases hu with h | h
    · have := (mem_up0 2 3 i).1 (by simpa [inB, t1_upB] using h); omega
    · rw [t1_upD] at h; simp at h
  · intro p hp
    simp only [List.mem_singleton] at hp
    subst hp
    exact ⟨rfl, rfl⟩

def t2 : Cch Nat := s2 0 t1 [(2, 3)]

theorem w_s2_id : t2.upD = [⟨2, 3, 3⟩] ∧ t2.downD = [⟨3, 2, 3⟩] ∧ t2.cap = 4 ∧ t2.upB = up0 ∧ t2.downB = down0 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem w_lin : Lin 0 t0 t2 :=
  Lin.step (Lin.step (Lin.fork (Lin.refl t0)) (Step.alloc _ _)) (Step.s2 _ _ w_s2ok)

theorem w_inv2 : Inv t2 := (lin_all 0 w_lin w_inv0).inv

def nu : Rows := [[(1, 0), (2, 1)], [(2, 2)], [(3, 3)], []]
def nd : Rows := [[], [(0, 0)], [(0, 1), (1, 2)], [(2, 3)]]

theorem mem_nu (r c i : Nat) : (c, i) ∈ bRow nu r ↔
    (r = 0 ∧ c = 1 ∧ i = 0) ∨ (r = 0 ∧ c = 2 ∧ i = 1) ∨ (r = 1 ∧ c = 2 ∧ i = 2) ∨ (r = 2 ∧ c = 3 ∧ i = 3) := by
  match r with
  | 0 => simp [bRow, nu]
  | 1 => simp [bRow, nu]
  | 2 => simp [bRow, nu]
  | 3 => simp [bRow, nu]
  | n + 4 => simp [bRow, nu]

theorem mem_nd (r c i : Nat) : (c, i) ∈ bRow nd r ↔
    (r = 1 ∧ c = 0 ∧ i = 0) ∨ (r = 2 ∧ c = 0 ∧ i = 1) ∨ (r = 2 ∧ c = 1 ∧ i = 2) ∨ (r = 3 ∧ c = 2 ∧ i = 3) := by
  match r with
  | 0 => simp [bRow, nd]
  | 1 => simp [bRow, nd]
  | 2 => simp [bRow, nd]
  | 3 => simp [bRow, nd]
  | n + 4 => simp [bRow, nd]

theorem w_addUp : AddSpec t2.upB t2.upD nu := by
  obtain ⟨h1, _, _, h4, _⟩ := w_s2_id
  rw [h1, h4]
  refine ⟨fun _ ⟨r, c, i⟩ => ?_, fun r => ?_⟩
  · simp only [inB, List.mem_singleton, Ent.mk.injEq]
    rw [mem_nu, mem_up0]; omega
  · match r with
    | 0 => simp [bRow, nu, StrictCol]
    | 1 => simp [bRow, nu, StrictCol]
    | 2 => simp [bRow, nu, StrictCol]
    | 3 => simp [bRow, nu, StrictCol]
    | n + 4 => simp [bRow, nu, StrictCol]

theorem w_addDown : AddSpec t2.downB t2.downD nd := by
  obtain ⟨_, h2, _, _, h5⟩ := w_s2_id
  rw [h2, h5]
  refine ⟨fun _ ⟨r, c, i⟩ => ?_, fun r => ?_⟩
  · simp only [inB, List.mem_singleton, Ent.mk.injEq]
    rw [mem_nd, mem_down0]; omega
  · match r with
    | 0 => simp [bRow, nd, StrictCol]
    | 1 => simp [bRow, nd, StrictCol]
    | 2 => simp [bRow, nd, StrictCol]
    | 3 => simp [bRow, nd, StrictCol]
    | n + 4 => simp [bRow, nd, StrictCol]

/-- `fold_rows` on the witness: the fold forked at `t2` and published onto it. -/
theorem w_fold_rows :
    (∀ r, upIds (fold t2.ver nu nd t2) r = upIds t2 r) ∧ (∀ r, downIds (fold t2.ver nu nd t2) r = downIds t2 r) :=
  fold_rows 0 w_inv2 WLin.refl w_addUp w_addDown

/-! ## A `WeightArr` state on which `set` succeeds -/

open Chunked in
/-- Two versions share table 0, which holds chunk 0 = `[1, 2]`. -/
def st0 : Chunked.St Nat :=
  ⟨⟨fun p => if p = 0 then [0] else [], fun c => if c = 0 then [1, 2] else [], 1, 1⟩, [⟨0, 2⟩, ⟨0, 2⟩], [0]⟩

theorem w_st0_wf : Chunked.WF st0 := by
  refine ⟨by simp [st0], fun p => ?_, fun p hp => ?_, fun p hp c hc => ?_⟩
  · simp only [st0, List.mem_cons, List.not_mem_nil, or_false]
    constructor
    · intro h; exact ⟨⟨0, 2⟩, Or.inl rfl, h.symm⟩
    · rintro ⟨v, hv | hv, hp⟩ <;> subst hv <;> exact hp.symm
  · simp [st0] at hp; simp [st0, hp]
  · simp [st0] at hp; subst hp; simp [st0] at hc; simp [st0, hc]

theorem w_chunked :
    (Chunked.setW st0 1 7).map (fun st => (st.vs, Chunked.read st.h ⟨0, 2⟩ 1, Chunked.read st.h ⟨1, 2⟩ 1)) =
      some ([⟨1, 2⟩, ⟨0, 2⟩], some 2, some 7) ∧
    Chunked.read st0.h ⟨0, 2⟩ 1 = some 2 := by
  refine ⟨rfl, rfl⟩

end CchTopo.Witness
