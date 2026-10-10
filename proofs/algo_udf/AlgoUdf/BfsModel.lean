import AlgoUdf.Graph
/-! # `bfs_find_bound` (algo_procedures.rs:2322-2400)

| here | there |
| --- | --- |
| `BS.parents`, `BS.seen`, `BS.next` | `parents`, `seen`, `next_frontier` :2332-2337 |
| `BS.dist`                | ghost: the hop level at which a node was first seen |
| `scanRels`               | the inner `for (edge_src, edge_dst, edge_id)` loop :2350-2365; `inl` = `found = true; break 'levels` |
| `scanFrontier`           | `for &node in &frontier` :2343-2366 |
| `levels h`               | `'levels: for _hop in 1..=config.max_len` :2340-2369, `h + 1 = _hop` (`swap` + `clear` = the new frontier is `next`) |
| `chain`                  | the parent walk :2383-2390 (outer `none` = a `parents[&cur]` panic) |
| `bfsFindBound`           | the whole function; `some none` = `Ok(None)`, `some (some es)` = `Ok(Some(..))` where the reported `(weight, cost)` are the forward folds over `es` :2393-2397 |

`parents` records the whole stored relationship (the Rust records its id,
`r.2.2`). Timeout / memory polls are not modelled.

Results:
* `bfs_sound` — `Some` is a hop-walk source→target of 1..maxLen edges through
  pairwise distinct nodes (so it is a simple path the enumeration can find), and
  the parent walk never panics.
* `bfs_complete` — `None` (with `source ≠ target`) means no hop-walk
  source→target of at most `maxLen` edges exists, so skipping the enumeration
  (run_path_algo :2633-2635) loses no path.
* `bfs_src_eq_tgt` — `source = target` always gives `None`.
-/
namespace AlgoUdf.BfsBound
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

structure BS where
  parents : Nat → Option (Nat × Rel)
  seen : Nat → Bool
  dist : Nat → Nat
  next : List Nat

section defs
variable (g : G) (tgt : Nat)

/-- Mark `far` seen from `u` at level `h`. -/
def mark (st : BS) (u h far : Nat) (r : Rel) : BS :=
  { st with seen := upd st.seen far true, parents := upd st.parents far (some (u, r)),
            dist := upd st.dist far h }

/-- `next_frontier.push(far)`. -/
def push (st : BS) (x : Nat) : BS := { st with next := st.next ++ [x] }

def scanRels (u h : Nat) : List Rel → BS → BS ⊕ BS
  | [], st => .inr st
  | r :: rs, st =>
    match farEndpoint u r.1 r.2.1 g.dir with
    | none => scanRels u h rs st
    | some far =>
      if st.seen far then scanRels u h rs st else
      if far = tgt then .inl (mark st u h far r)
      else scanRels u h rs (push (mark st u h far r) far)

def scanFrontier (h : Nat) : List Nat → BS → BS ⊕ BS
  | [], st => .inr st
  | u :: us, st =>
    match scanRels g tgt u h (g.rels u) st with
    | .inl st' => .inl st'
    | .inr st' => scanFrontier h us st'

def levels : Nat → Nat → List Nat → BS → Option BS
  | _, 0, _, _ => none
  | h, k + 1, fr, st =>
    if fr = [] then none else
    match scanFrontier g tgt (h + 1) fr { st with next := [] } with
    | .inl st' => some st'
    | .inr st' => levels (h + 1) k st'.next st'

def init (src : Nat) : BS := ⟨fun _ => none, upd (fun _ => false) src true, fun _ => 0, []⟩

def chain (src : Nat) (P : Nat → Option (Nat × Rel)) : Nat → Nat → List Rel → Option (List Rel)
  | 0, _, _ => none
  | n + 1, cur, acc =>
    if cur = src then some acc else
    match P cur with
    | none => none
    | some (p, r) => chain src P n p (r :: acc)

def bfsFindBound (src maxLen : Nat) : Option (Option (List Rel)) :=
  if maxLen = 0 then some none else
  match levels g tgt 0 maxLen [src] (init src) with
  | none => some none
  | some st => (chain src st.parents (st.dist tgt + 1) tgt []).map some

end defs

/-! ## Invariants -/

/-- Parent-pointer invariant: every seen node but the source has a parent one
level closer that reaches it in one step. -/
structure PInv (g : G) (src : Nat) (st : BS) : Prop where
  srcSeen : st.seen src = true
  srcDist : st.dist src = 0
  par : ∀ x, st.seen x = true → x ≠ src → ∃ p r, st.parents x = some (p, r) ∧ Adj g p x r ∧
          st.seen p = true ∧ st.dist p + 1 = st.dist x

/-- Level invariant after `h` levels, with `fr` the current frontier. -/
structure LInv (g : G) (src tgt h : Nat) (fr : List Nat) (st : BS) : Prop extends PInv g src st where
  distLe : ∀ x, st.seen x = true → st.dist x ≤ h
  closed : ∀ y, st.seen y = true → st.dist y < h → ∀ x r, Adj g y x r →
             st.seen x = true ∧ st.dist x ≤ st.dist y + 1
  front : ∀ x, st.seen x = true → st.dist x = h → x ∈ fr
  frontSeen : ∀ x ∈ fr, st.seen x = true ∧ st.dist x = h
  tgtUnseen : st.seen tgt = false

/-- Invariant while scanning level `h` → `h + 1`: `done` nodes and the
relationships `P` of the current node `u` are already expanded. -/
structure SInv (g : G) (src tgt h : Nat) (fr done : List Nat) (u : Nat) (P : List Rel) (st : BS) : Prop
    extends PInv g src st where
  distLe : ∀ x, st.seen x = true → st.dist x ≤ h + 1
  closedOld : ∀ y, st.seen y = true → st.dist y < h → ∀ x r, Adj g y x r →
             st.seen x = true ∧ st.dist x ≤ st.dist y + 1
  closedDone : ∀ y ∈ done, ∀ x r, Adj g y x r → st.seen x = true ∧ st.dist x ≤ h + 1
  closedCur : ∀ r ∈ P, ∀ x, Adj g u x r → st.seen x = true ∧ st.dist x ≤ h + 1
  nextSpec : ∀ x, (st.seen x = true ∧ st.dist x = h + 1) ↔ x ∈ st.next
  front : ∀ x, st.seen x = true → st.dist x = h → x ∈ fr
  frontSeen : ∀ x ∈ fr, st.seen x = true ∧ st.dist x = h
  tgtUnseen : st.seen tgt = false

@[simp] theorem mark_seen (st : BS) (u h far : Nat) (r : Rel) :
    (mark st u h far r).seen = upd st.seen far true := rfl
@[simp] theorem mark_dist (st : BS) (u h far : Nat) (r : Rel) :
    (mark st u h far r).dist = upd st.dist far h := rfl
@[simp] theorem mark_parents (st : BS) (u h far : Nat) (r : Rel) :
    (mark st u h far r).parents = upd st.parents far (some (u, r)) := rfl
@[simp] theorem mark_next (st : BS) (u h far : Nat) (r : Rel) :
    (mark st u h far r).next = st.next := rfl
@[simp] theorem push_seen (st : BS) (x : Nat) : (push st x).seen = st.seen := rfl
@[simp] theorem push_dist (st : BS) (x : Nat) : (push st x).dist = st.dist := rfl
@[simp] theorem push_parents (st : BS) (x : Nat) : (push st x).parents = st.parents := rfl
@[simp] theorem push_next (st : BS) (x : Nat) : (push st x).next = st.next ++ [x] := rfl

theorem mark_seen_old {st : BS} {u h far : Nat} {r : Rel} {x : Nat} (hs : st.seen far = false)
    (hx : st.seen x = true) :
    upd st.seen far true x = true ∧ upd st.dist far h x = st.dist x ∧
      upd st.parents far (some (u, r)) x = st.parents x := by
  have : x ≠ far := by intro h; rw [h, hs] at hx; cases hx
  simp [upd_ne _ _ _ _ this, hx]

/-- One level: either the target is found (with the parent invariant and the
target at level `h + 1`), or the level invariant moves to `h + 1`. -/
theorem scanRels_spec {g : G} {src tgt h : Nat} {fr done : List Nat} {u : Nat} (hufr : u ∈ fr) :
    ∀ (rs P : List Rel) (st : BS), P ++ rs = g.rels u → SInv g src tgt h fr done u P st →
      match scanRels g tgt u (h + 1) rs st with
      | .inl st' => PInv g src st' ∧ st'.seen tgt = true ∧ st'.dist tgt = h + 1
      | .inr st' => SInv g src tgt h fr done u (g.rels u) st' := by
  intro rs
  induction rs with
  | nil => intro P st hP I; simp at hP; subst hP; exact I
  | cons r rs ih =>
    intro P st hP I
    have hr : r ∈ g.rels u := by rw [← hP]; simp
    have hP' : (P ++ [r]) ++ rs = g.rels u := by simpa using hP
    -- the current node is seen at level h (it is in the frontier)
    have huS : st.seen u = true ∧ st.dist u = h := I.frontSeen u hufr
    -- the invariant with `r` also discharged, given `r`'s far end is now seen at ≤ h+1
    have extend : ∀ st', SInv g src tgt h fr done u P st' →
        (∀ x, Adj g u x r → st'.seen x = true ∧ st'.dist x ≤ h + 1) →
        SInv g src tgt h fr done u (P ++ [r]) st' := by
      intro st' I' hx
      refine ⟨I'.toPInv, I'.distLe, I'.closedOld, I'.closedDone, ?_, I'.nextSpec, I'.front,
        I'.frontSeen, I'.tgtUnseen⟩
      intro r' hr' x hadj
      simp only [List.mem_append, List.mem_singleton] at hr'
      rcases hr' with hr' | rfl
      · exact I'.closedCur r' hr' x hadj
      · exact hx x hadj
    cases hf : farEndpoint u r.1 r.2.1 g.dir with
    | none =>
      have e : scanRels g tgt u (h + 1) (r :: rs) st = scanRels g tgt u (h + 1) rs st := by
        simp [scanRels, hf]
      rw [e]
      apply ih _ st hP' (extend st I ?_)
      intro x hadj; rw [hadj.2] at hf; cases hf
    | some far =>
      by_cases hsf : st.seen far = true
      · have e : scanRels g tgt u (h + 1) (r :: rs) st = scanRels g tgt u (h + 1) rs st := by
          simp [scanRels, hf, hsf]
        rw [e]
        apply ih _ st hP' (extend st I ?_)
        intro x hadj
        have : x = far := Option.some.inj (hadj.2.symm.trans hf)
        subst this; exact ⟨hsf, I.distLe _ hsf⟩
      · have hsf' : st.seen far = false := by simpa using hsf
        have hfs : far ≠ src := by intro h; rw [h, I.srcSeen] at hsf'; cases hsf'
        -- parent invariant for the marked state
        have PI : PInv g src (mark st u (h + 1) far r) := by
          refine ⟨(mark_seen_old (u := u) (r := r) (h := h + 1) hsf' I.srcSeen).1,
            by simp only [mark_dist]; rw [(mark_seen_old (u := u) (r := r) hsf' I.srcSeen).2.1]; exact I.srcDist, ?_⟩
          intro x hx hxs
          by_cases hxf : x = far
          · subst hxf
            refine ⟨u, r, by simp [mark], ⟨hr, hf⟩, (mark_seen_old (u := u) (r := r) (h := h + 1) hsf' huS.1).1, ?_⟩
            simp only [mark_dist]
            rw [(mark_seen_old (u := u) (r := r) hsf' huS.1).2.1, huS.2]; simp
          · simp only [mark, upd_ne _ _ _ _ hxf] at hx ⊢
            obtain ⟨p, r', h1, h2, h3, h4⟩ := I.par x hx hxs
            have hpf : p ≠ far := by intro h; rw [h, hsf'] at h3; cases h3
            exact ⟨p, r', h1, h2, by simp [upd_ne _ _ _ _ hpf, h3], by simp [upd_ne _ _ _ _ hpf, h4]⟩
        by_cases hft : far = tgt
        · subst hft
          have e : scanRels g far u (h + 1) (r :: rs) st = .inl (mark st u (h + 1) far r) := by
            simp [scanRels, hf, hsf]
          rw [e]
          exact ⟨PI, by simp [mark], by simp [mark]⟩
        · have e : scanRels g tgt u (h + 1) (r :: rs) st =
              scanRels g tgt u (h + 1) rs (push (mark st u (h + 1) far r) far) := by
            simp [scanRels, hf, hsf, hft]
          rw [e]
          apply ih _ _ hP'
          -- the invariant for the marked + pushed state
          refine ⟨⟨PI.srcSeen, PI.srcDist, PI.par⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · intro x hx
            by_cases hxf : x = far
            · subst hxf; simp
            · simp only [push_seen, push_dist, mark_seen, mark_dist, upd_ne _ _ _ _ hxf] at hx ⊢; exact I.distLe x hx
          · intro y hy hdy x r' hadj
            have hyf : y ≠ far := by intro h'; subst h'; simp at hdy <;> omega
            simp only [push_seen, push_dist, mark_seen, mark_dist, upd_ne _ _ _ _ hyf] at hy hdy ⊢
            obtain ⟨h1, h2⟩ := I.closedOld y hy hdy x r' hadj
            exact ⟨(mark_seen_old (u := u) (r := r) (h := h + 1) hsf' h1).1, by (try simp only [push_dist, mark_dist]); rw [(mark_seen_old (u := u) (r := r) hsf' h1).2.1]; exact h2⟩
          · intro y hy x r' hadj
            obtain ⟨h1, h2⟩ := I.closedDone y hy x r' hadj
            exact ⟨(mark_seen_old (u := u) (r := r) (h := h + 1) hsf' h1).1, by (try simp only [push_dist, mark_dist]); rw [(mark_seen_old (u := u) (r := r) hsf' h1).2.1]; exact h2⟩
          · intro r' hr' x hadj
            simp only [List.mem_append, List.mem_singleton] at hr'
            rcases hr' with hr' | rfl
            · obtain ⟨h1, h2⟩ := I.closedCur r' hr' x hadj
              exact ⟨(mark_seen_old (u := u) (r := r) (h := h + 1) hsf' h1).1, by (try simp only [push_dist, mark_dist]); rw [(mark_seen_old (u := u) (r := r) hsf' h1).2.1]; exact h2⟩
            · have : x = far := Option.some.inj (hadj.2.symm.trans hf)
              subst this; simp
          · intro x
            by_cases hxf : x = far
            · subst hxf; simp
            · simp only [push_seen, push_dist, push_next, mark_seen, mark_dist, mark_next, upd_ne _ _ _ _ hxf, List.mem_append, List.mem_singleton, hxf, or_false]; exact I.nextSpec x
          · intro x hx hdx
            by_cases hxf : x = far
            · subst hxf; simp [mark] at hdx
            · simp only [push_seen, push_dist, mark_seen, mark_dist, upd_ne _ _ _ _ hxf] at hx hdx; exact I.front x hx hdx
          · intro x hx
            obtain ⟨h1, h2⟩ := I.frontSeen x hx
            exact ⟨(mark_seen_old (u := u) (r := r) (h := h + 1) hsf' h1).1, by (try simp only [push_dist, mark_dist]); rw [(mark_seen_old (u := u) (r := r) hsf' h1).2.1]; exact h2⟩
          · have : tgt ≠ far := Ne.symm hft
            simp [mark, upd_ne _ _ _ _ this, I.tgtUnseen]

theorem scanFrontier_spec {g : G} {src tgt h : Nat} {fr : List Nat} :
    ∀ (rest done : List Nat) (st : BS), done ++ rest = fr →
      SInv g src tgt h fr done 0 [] st →
      match scanFrontier g tgt (h + 1) rest st with
      | .inl st' => PInv g src st' ∧ st'.seen tgt = true ∧ st'.dist tgt = h + 1
      | .inr st' => SInv g src tgt h fr fr 0 [] st' := by
  intro rest
  induction rest with
  | nil => intro done st hd I; simp at hd; subst hd; exact I
  | cons u us ih =>
    intro done st hd I
    have hufr : u ∈ fr := by rw [← hd]; simp
    have I0 : SInv g src tgt h fr done u [] st :=
      ⟨I.toPInv, I.distLe, I.closedOld, I.closedDone, by simp, I.nextSpec, I.front, I.frontSeen,
        I.tgtUnseen⟩
    have := scanRels_spec (fr := fr) (done := done) hufr (g.rels u) [] st rfl I0
    cases hres : scanRels g tgt u (h + 1) (g.rels u) st with
    | inl st' => rw [hres] at this; simp only [scanFrontier, hres]; exact this
    | inr st' =>
      rw [hres] at this
      simp only [scanFrontier, hres]
      apply ih (done ++ [u]) st' (by simpa using hd)
      refine ⟨this.toPInv, this.distLe, this.closedOld, ?_, by simp, this.nextSpec, this.front,
        this.frontSeen, this.tgtUnseen⟩
      intro y hy x r hadj
      simp only [List.mem_append, List.mem_singleton] at hy
      rcases hy with hy | rfl
      · exact this.closedDone y hy x r hadj
      · exact this.closedCur r hadj.1 x hadj

end AlgoUdf.BfsBound
