import FalkorNativeIndex.Rows
set_option linter.unusedSimpArgs false

/-!
# The background index builder as a transition system (`online_build.rs`, #2355 / #3585)

A job builds one generation of one index: it reads a batch on a new build version, publishes it through the
host's global lock, and submits itself again while the generation is `Building`. Threads are an interleaving
parameter: the pool runs any queued job next (`PoolStep`), and what a publish does is a parameter
(`Outcome`), so every statement holds for every interleaving. The published entry of the index is the
model `EN` (`Rebase.lean`: generation, state, `read`), a batch's commit is `EN.indexerApply`.
-/

namespace NI

/-- `Step` (`online_build.rs:~165`): what a job's batch did. -/
inductive Step where
  | more | tooLarge | failed | done | cancelled
deriving DecidableEq, Repr

/-- `standing`: `None` while the job's generation is `Building` here; `Done` once it is `Ready`;
    `Cancelled` when the index is gone or is another generation. -/
def standing (en : Option EN) (generation : Nat) : Option Step :=
  match en.map (fun e => (e.generation, e.isBuilding)) with
  | some (g, true) => if g = generation then none else some .cancelled
  | some (g, false) => if g = generation then some .done else some .cancelled
  | none => some .cancelled

/-- `Sizing`. -/
structure Sizing where
  floor : Nat
  cap : Nat
  catchUpMax : Nat

/-- A job: the generation it builds and its batch limit (starts at `cap`). -/
structure Job where
  generation : Nat
  limit : Nat
deriving DecidableEq

def Job.new (sz : Sizing) (generation : Nat) : Job := ⟨generation, sz.cap⟩

/-- The system a job acts on: the published entry of its index, whether the builder was stopped, whether the
    graph is still alive (`Weak::upgrade`). -/
structure World where
  pub : Option EN
  stopped : Bool
  alive : Bool

/-- `Job::read`: the batch size, or the step that ends the job before reading. -/
def readPhase (sz : Sizing) (w : World) (j : Job) : Except Step Nat :=
  if w.stopped then .error .cancelled
  else if !w.alive then .error .cancelled
  else match standing w.pub j.generation with
    | some st => .error st
    | none => .ok (sizingBatch sz.floor ((indexBuildRead w.pub).getD 0) j.limit)

/-- What a publish does (a parameter: the host's commit): the batch was discarded as a catch-up too large, it
    was published (another commit, or this one), or it was refused. -/
inductive Outcome where
  | tooLarge | published | refused
deriving DecidableEq, Repr

/-- The entry after a published batch of `entities` that read to `next` (`commit_indexer_batch` on this
    index: the entry's `indexer_apply`). -/
def commitEntry (work : EncodedTuples) (next : Option BuildCursor) (entities : Nat) (w : World) : World :=
  { w with pub := w.pub.map (fun e => (e.indexerApply work next entities).1) }

/-- `Job::step` after the read: publish holding the host's lock. After the stop, nothing is
    published (`builder.stopped()` is checked again under the lock). -/
def publishPhase (work : EncodedTuples) (next : Option BuildCursor) (entities : Nat) (generation : Nat)
    (o : Outcome) (w : World) : Step × World :=
  if w.stopped then (.cancelled, w)
  else match o with
    | .tooLarge => (.tooLarge, w)
    | .published =>
      let w' := commitEntry work next entities w
      ((standing w'.pub generation).getD .more, w')
    | .refused => (.failed, w)

/-- The new limit. -/
def nextLimit (sz : Sizing) (st : Step) (entities limit : Nat) : Nat :=
  match st with
  | .more => min (limit * 2) sz.cap
  | .tooLarge => max (entities / 2) sz.floor
  | _ => limit

/-- One whole `Job::step`. -/
def jobStep (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome) (w : World) (j : Job) :
    Step × World × Job :=
  match readPhase sz w j with
  | .error st => (st, w, j)
  | .ok entities =>
    let r := publishPhase work next entities j.generation o w
    (r.1, r.2, { j with limit := nextLimit sz r.1 entities j.limit })

/-! ## What a step does and does not do -/

/-- **A job whose generation is not the index's current `Building` one reads and publishes nothing**: the index
    is dropped or recreated (`Cancelled`), or already built (`Done`); the world and the job are unchanged. -/
theorem jobStep_standing (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (st : Step) (h : standing w.pub j.generation = some st) (hs : w.stopped = false)
    (ha : w.alive = true) : jobStep sz work next o w j = (st, w, j) := by
  simp [jobStep, readPhase, hs, ha, h]

/-- After the stop nothing is read and nothing published. -/
theorem jobStep_stopped (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (h : w.stopped = true) : jobStep sz work next o w j = (.cancelled, w, j) := by
  simp [jobStep, readPhase, h]

/-- A graph that is gone ends its job. -/
theorem jobStep_dead (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (hs : w.stopped = false) (h : w.alive = false) :
    jobStep sz work next o w j = (.cancelled, w, j) := by
  simp [jobStep, readPhase, hs, h]

/-- A refused or too-large batch changes the published entry not at all. -/
theorem publish_noop (work : EncodedTuples) (next : Option BuildCursor) (entities generation : Nat) (w : World) :
    (publishPhase work next entities generation .refused w).2 = w ∧
    (publishPhase work next entities generation .tooLarge w).2 = w := by
  unfold publishPhase; split <;> simp_all

/-- **A step only ever changes the entry of its own generation, and only by a batch's `indexer_apply`**: when
    the job's standing check passes the entry exists and is `Building` at the job's generation; a published
    batch keeps the generation. -/
theorem commitEntry_generation (work : EncodedTuples) (next : Option BuildCursor) (entities : Nat) (w : World)
    (e : EN) (h : w.pub = some e) :
    ∃ e', (commitEntry work next entities w).pub = some e' ∧ e'.generation = e.generation := by
  refine ⟨(e.indexerApply work next entities).1, by simp [commitEntry, h], ?_⟩
  cases e with | mk i s g c r =>
    cases s <;> simp [EN.indexerApply]

/-- The batch the read phase sizes is at least the floor and at most the larger of the limit and the floor. -/
theorem readPhase_size (sz : Sizing) (w : World) (j : Job) (n : Nat) (h : readPhase sz w j = .ok n) :
    sz.floor ≤ n ∧ n ≤ max j.limit sz.floor := by
  unfold readPhase at h
  split at h
  · cases h
  · split at h
    · cases h
    · cases hs : standing w.pub j.generation <;> simp [hs] at h
      subst h
      exact ⟨(sizingBatch_spec _ _ _).1, (sizingBatch_spec _ _ _).2.1⟩

/-- **The limit stays within `[?, cap]`**: a `More` step doubles up to the cap, a `TooLarge` one halves the
    batch, never below the floor and never past the cap (given `floor ≤ cap` and `limit ≤ cap`). -/
theorem nextLimit_le_cap (sz : Sizing) (hf : sz.floor ≤ sz.cap) (st : Step) (entities limit : Nat)
    (hl : limit ≤ sz.cap) (he : entities ≤ max limit sz.floor) : nextLimit sz st entities limit ≤ sz.cap := by
  cases st <;> simp [nextLimit] <;> omega

/-- **Each worker step keeps the job's limit within the cap.** -/
theorem jobStep_limit_le_cap (sz : Sizing) (hf : sz.floor ≤ sz.cap) (work : EncodedTuples)
    (next : Option BuildCursor) (o : Outcome) (w : World) (j : Job) (hl : j.limit ≤ sz.cap) :
    (jobStep sz work next o w j).2.2.limit ≤ sz.cap := by
  unfold jobStep
  cases hr : readPhase sz w j with
  | error st => simpa using hl
  | ok n =>
    obtain ⟨_, h2⟩ := readPhase_size sz w j n hr
    exact nextLimit_le_cap sz hf _ _ _ hl h2

/-- A published batch that did not reach the end adds its size to the entry's `read` count (saturating), so
    the next batch can be as large as everything read so far. -/
theorem publish_read (work : EncodedTuples) (n' : BuildCursor) (entities generation : Nat) (w : World)
    (e : EN) (hs : w.stopped = false) (he : w.pub = some e) (hb : e.isBuilding = true) :
    indexBuildRead (commitEntry work (some n') entities w).pub = some (min (e.read + entities) (W - 1)) := by
  have := (indexerApply_read e work n' entities hb).1
  simp [indexBuildRead, commitEntry, he, this]

/-! ## The pool: any queued job runs next -/

/-- The key of a job in the queue: its index and generation. -/
abbrev JobKey := Nat × Nat

/-- A queued job: a key (index id, generation) and its state. -/
structure QJob where
  key : JobKey
  job : Job
deriving DecidableEq

/-- One pool step (`Job::run`): any queued job (the interleaving) runs a step; `more`/`tooLarge`
    submit it again, the others end it. -/
def PoolStep (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome) (q : List QJob)
    (w : World) (qj : QJob) : List QJob × World :=
  let r := jobStep sz work next o w qj.job
  let rest := q.erase qj
  match r.1 with
  | .more | .tooLarge => (rest ++ [⟨qj.key, r.2.2⟩], r.2.1)
  | _ => (rest, r.2.1)

/-- **No two jobs build one index**: the queue's keys stay distinct under every pool step (a job submits itself
    again under its own key; the other keys are untouched). (Each generation is started once: the commit
    hands the builder `indexesToBuildSince`, `Rows.lean`.) -/
theorem poolStep_keys_nodup (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (q : List QJob) (w : World) (qj : QJob) (hm : qj ∈ q) (h : (q.map (·.key)).Nodup) :
    (((PoolStep sz work next o q w qj).1).map (·.key)).Nodup := by
  have hperm : (q.map (·.key)).Perm (qj.key :: (q.erase qj).map (·.key)) := by
    have := (List.perm_cons_erase hm).map (·.key)
    simpa using this
  have hn : (qj.key :: (q.erase qj).map (·.key)).Nodup := hperm.nodup_iff.1 h
  unfold PoolStep
  cases hst : (jobStep sz work next o w qj.job).1 <;> simp only [hst]
  all_goals first
    | (simp only [List.map_append, List.map_cons, List.map_nil]
       exact (List.perm_append_comm.nodup_iff.2 (by simpa using hn)))
    | exact (List.nodup_cons.1 hn).2

/-! ## Starting, submitting and stopping -/

/-- `Builder::submit`: the job joins the pool's queue. -/
def submit (q : List QJob) (qj : QJob) : List QJob := q ++ [qj]

/-- `Builder::enqueue`: a new job at the index's generation, limit `cap`, submitted. -/
def enqueue (sz : Sizing) (q : List QJob) (id generation : Nat) : List QJob :=
  submit q ⟨(id, generation), Job.new sz generation⟩

/-- `Builder::stop_within`: the stop is for good. (The bounded wait for the pool's workers is a
    timing statement and is not modelled.) -/
def stopWorld (w : World) : World := { w with stopped := true }

theorem enqueue_keys (sz : Sizing) (q : List QJob) (id generation : Nat)
    (h : (q.map (·.key)).Nodup) (hnew : (id, generation) ∉ q.map (·.key)) :
    ((enqueue sz q id generation).map (·.key)).Nodup := by
  simp only [enqueue, submit, List.map_append, List.map_cons, List.map_nil]
  exact List.nodup_append.2 ⟨h, (List.nodup_cons.2 ⟨by simp, List.nodup_nil⟩), by simpa using hnew⟩

/-- **The stop is for good**: no step undoes it, and a stopped builder publishes nothing. -/
theorem stopped_persists (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (h : w.stopped = true) : (jobStep sz work next o w j).2.1.stopped = true := by
  rw [jobStep_stopped sz work next o w j h]; exact h

theorem stopWorld_stops (w : World) : (stopWorld w).stopped = true := rfl

/-! ## Drop and re-create leave the old job harmless -/

/-- **A dropped index ends its job**: no entry, so `Cancelled`, nothing read or published. -/
theorem jobStep_dropped (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (hs : w.stopped = false) (ha : w.alive = true) (h : w.pub = none) :
    jobStep sz work next o w j = (.cancelled, w, j) :=
  jobStep_standing sz work next o w j .cancelled (by simp [standing, h]) hs ha

/-- **An index dropped and created again is another generation, so the old job ends** (and the new one's
    job, started by the commit that created it, builds it). -/
theorem jobStep_recreated (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (e : EN) (hs : w.stopped = false) (ha : w.alive = true) (h : w.pub = some e)
    (hg : e.generation ≠ j.generation) : jobStep sz work next o w j = (.cancelled, w, j) := by
  apply jobStep_standing sz work next o w j .cancelled _ hs ha
  unfold standing
  rw [h]
  simp only [Option.map_some]
  cases hb : e.isBuilding <;> simp [hg]

/-- A finished index (`Ready`, same generation) ends its job as `Done`, changing nothing. -/
theorem jobStep_done (sz : Sizing) (work : EncodedTuples) (next : Option BuildCursor) (o : Outcome)
    (w : World) (j : Job) (e : EN) (hs : w.stopped = false) (ha : w.alive = true) (h : w.pub = some e)
    (hg : e.generation = j.generation) (hb : e.isBuilding = false) :
    jobStep sz work next o w j = (.done, w, j) := by
  apply jobStep_standing sz work next o w j .done _ hs ha
  simp [standing, h, hb, hg]

/-! ## Completed batches compose to the full build

A column's build reads its label's nodes in ascending order, `budget` of them from the cursor (`scan_nodes`,
`Build.lean`: `scanNodes_tuples`, `scanNodes_below_cursor`), whatever the batch sizes a job chose. Over any
sequence of budgets, the batches that end the build (the last returns no cursor) read every node exactly once. -/

/-- The cursor `scan_nodes` returns (`nodes.peek()`) and the nodes it read. -/
def nodeBatch (lm : List Nat) (m budget : Nat) : List Nat × Option Nat :=
  ((nodesFrom lm m).take budget, ((nodesFrom lm m).drop budget).head?)

/-- The nodes read by successive batches, with the budgets the job chose, from cursor `m`; the last cursor. -/
def runBatches (lm : List Nat) : Nat → List Nat → List Nat × Option Nat
  | m, [] => ([], some m)
  | m, b :: bs =>
    match nodeBatch lm m b with
    | (r, none) => (r, none)
    | (r, some m') => ((r ++ (runBatches lm m' bs).1), (runBatches lm m' bs).2)

theorem nodesFrom_head (lm : List Nat) (hs : lm.Pairwise (· < ·)) (m b : Nat) (m' : Nat)
    (hm : ((nodesFrom lm m).drop b).head? = some m') :
    nodesFrom lm m' = (nodesFrom lm m).drop b := by
  have hs' : (nodesFrom lm m).Pairwise (· < ·) := List.Pairwise.sublist List.filter_sublist hs
  have hsplit := List.take_append_drop b (nodesFrom lm m)
  obtain ⟨rest, hrest⟩ : ∃ rest, (nodesFrom lm m).drop b = m' :: rest := by
    cases h : (nodesFrom lm m).drop b with
    | nil => simp [h] at hm
    | cons a r => simp [h] at hm; exact ⟨r, by rw [hm]⟩
  have hmem : m' ∈ nodesFrom lm m := by
    rw [← hsplit]; exact List.mem_append_right _ (by rw [hrest]; simp)
  have hmge : m ≤ m' := by simpa [nodesFrom] using (List.mem_filter.1 hmem).2
  have hpa := hs'
  rw [← hsplit, List.pairwise_append, hrest] at hpa
  -- nodesFrom lm m' = filter (m' ≤) (filter (m ≤) lm)
  have h1 : nodesFrom lm m' = (nodesFrom lm m).filter (fun n => decide (m' ≤ n)) := by
    simp only [nodesFrom, List.filter_filter]
    apply List.filter_congr
    intro n _
    by_cases h : m' ≤ n <;> simp [h] <;> omega
  have ht : ((nodesFrom lm m).take b).filter (fun n => decide (m' ≤ n)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro n hn
    have := hpa.2.2 n hn m' (by simp)
    simp; omega
  have hd : (m' :: rest).filter (fun n => decide (m' ≤ n)) = m' :: rest := by
    rw [List.filter_eq_self]
    intro n hn
    rcases List.mem_cons.1 hn with rfl | hn
    · simp
    · have := List.rel_of_pairwise_cons hpa.2.1 hn
      simp; omega
  have hf : (nodesFrom lm m).filter (fun n => decide (m' ≤ n)) =
      ((nodesFrom lm m).take b).filter (fun n => decide (m' ≤ n)) ++
      ((nodesFrom lm m).drop b).filter (fun n => decide (m' ≤ n)) := by
    rw [← List.filter_append, hsplit]
  rw [h1, hf, ht, List.nil_append, hrest]
  exact hd

/-- **Completed batches compose to the full build**: over any sequence of budgets, if the batches reach the
    end (the last returns no cursor), the nodes they read are exactly the label's nodes from the start
    cursor, each once and in order. -/
theorem runBatches_cover (lm : List Nat) (hs : lm.Pairwise (· < ·)) : ∀ (bs : List Nat) (m : Nat) (r : List Nat),
    runBatches lm m bs = (r, none) → r = nodesFrom lm m := by
  intro bs
  induction bs with
  | nil => intro m r h; simp [runBatches] at h
  | cons b bs ih =>
    intro m r h
    unfold runBatches at h
    cases hb : nodeBatch lm m b with
    | mk r0 c =>
      cases c with
      | none =>
        simp only [hb] at h
        simp only [Prod.mk.injEq, and_true] at h
        subst h
        have hn : ((nodesFrom lm m).drop b).head? = none := by simpa [nodeBatch, Prod.ext_iff] using (Prod.ext_iff.1 hb).2
        have hd : (nodesFrom lm m).drop b = [] := by
          cases h : (nodesFrom lm m).drop b with
          | nil => rfl
          | cons a t => simp [h] at hn
        have h0 : r0 = (nodesFrom lm m).take b := (Prod.ext_iff.1 hb).1.symm
        rw [h0]
        have := List.take_append_drop b (nodesFrom lm m)
        rw [hd, List.append_nil] at this
        exact this
      | some m' =>
        simp only [hb] at h
        have hm' : ((nodesFrom lm m).drop b).head? = some m' := (Prod.ext_iff.1 hb).2
        have h0 : r0 = (nodesFrom lm m).take b := (Prod.ext_iff.1 hb).1.symm
        obtain ⟨hr, hc⟩ := Prod.ext_iff.1 h
        simp only at hr hc
        have hrec := ih m' (runBatches lm m' bs).1 (Prod.ext rfl hc)
        rw [← hr, h0, hrec, nodesFrom_head lm hs m b m' hm']
        exact List.take_append_drop _ _

/-! ### The edge analogue

A relationship column's batch is `scan_edges` (`Build.lean`: `edgeLoop`, `skipEdges`): from the cursor pair
`(from_src, from_dst)` it reads edges in `(src, dst)` order up to the budget, then the rest of the pair it stopped in,
and returns the next pair. Over any sequence of budgets, the batches that end the build read every edge from the
start pair on exactly once, in order. The read edges are recorded here as their ids (`recIdx`: what
`push_encoded`'s result is, for `edgeLoop_spec`, abstracted to "the id was read"). -/

/-- Whether an edge's `(src, dst)` pair is at or after the pair `(s, d)`: the edges `scan_edges` starts with. -/
def atOrAfter (s d : Nat) (e : Nat × Nat × Nat) : Bool := decide (s < e.1 ∨ (s = e.1 ∧ d ≤ e.2.1))

/-- Reading an edge, recorded: its id is appended. -/
def recIdx (eid : Nat) (out : EncodedTuples) : EncodedTuples := ⟨out.scalar ++ [eid], out.array⟩

/-- One `scan_edges` batch from pair `(s, d)` with `budget`: the ids read and the pair to resume from. -/
def edgeBatch (t : List (Nat × Nat × Nat)) (s d budget : Nat) : List Nat × Option (Nat × Nat) :=
  let r := edgeLoop budget recIdx 0 (skipEdges t s d) ⟨[], []⟩
  (r.1.scalar, r.2.head?.map (fun e => (e.1, e.2.1)))

/-- The edges read by successive batches, with the budgets the job chose, from pair `(s, d)`; the last cursor. -/
def runEdgeBatches (t : List (Nat × Nat × Nat)) : Nat → Nat → List Nat → List Nat × Option (Nat × Nat)
  | s, d, [] => ([], some (s, d))
  | s, d, b :: bs =>
    match edgeBatch t s d b with
    | (r, none) => (r, none)
    | (r, some (s', d')) => (r ++ (runEdgeBatches t s' d' bs).1, (runEdgeBatches t s' d' bs).2)

/-- On edges in `(src, dst)` order, `skipEdges` is the filter "pair at or after the cursor". -/
theorem skipEdges_eq (t : List (Nat × Nat × Nat)) (hs : t.Pairwise pairLe) (s d : Nat) :
    skipEdges t s d = t.filter (atOrAfter s d) := by
  unfold skipEdges
  have hf : (t.filter (fun e => decide (s ≤ e.1))).Pairwise pairLe := List.Pairwise.sublist List.filter_sublist hs
  have key : ∀ l : List (Nat × Nat × Nat), l.Pairwise pairLe → (∀ e ∈ l, s ≤ e.1) →
      l.dropWhile (skipP s d) = l.filter (fun e => !skipP s d e) := by
    intro l
    induction l with
    | nil => intro _ _; rfl
    | cons x xs ih =>
      intro hp hge
      by_cases hx : skipP s d x = true
      · simp only [List.dropWhile_cons, hx, ite_true, List.filter_cons, Bool.not_true, Bool.false_eq_true,
          ite_false]
        exact ih hp.of_cons (fun e he => hge e (List.mem_cons_of_mem _ he))
      · have hx' : skipP s d x = false := by simpa using hx
        simp only [List.dropWhile_cons, hx', Bool.false_eq_true, ite_false, List.filter_cons, Bool.not_false,
          ite_true]
        congr 1
        symm
        rw [List.filter_eq_self]
        intro y hy
        have hxy := List.rel_of_pairwise_cons hp hy
        have hsx := hge x (List.mem_cons_self ..)
        simp only [skipP, decide_eq_true_eq, not_and, Nat.not_lt, decide_eq_false_iff_not] at hx'
        simp only [pairLe] at hxy
        simp only [skipP, Bool.not_eq_true', decide_eq_false_iff_not, not_and, Nat.not_lt]
        intro h1
        by_cases h2 : x.1 = s
        · have := hx' h2; omega
        · omega
  rw [key _ hf (fun e he => by simpa using (List.mem_filter.1 he).2), List.filter_filter]
  apply List.filter_congr
  intro e _
  simp only [skipP, atOrAfter]
  by_cases h1 : s ≤ e.1 <;> by_cases h2 : e.1 = s <;> by_cases h3 : e.2.1 < d <;> simp [h1, h2, h3] <;> omega

/-- **Completed edge batches compose to the full build** (the edge analogue of `runBatches_cover`): over any
    sequence of budgets, if the batches reach the end (the last returns no cursor), the edges they read are
    exactly those of the type from the start pair on, each once and in `(src, dst)` order. -/
theorem runEdgeBatches_cover (t : List (Nat × Nat × Nat)) (hs : t.Pairwise pairLe) :
    ∀ (bs : List Nat) (s d : Nat) (r : List Nat), runEdgeBatches t s d bs = (r, none) →
      r = ((t.filter (atOrAfter s d)).map (·.2.2)) := by
  intro bs
  induction bs with
  | nil => intro s d r h; simp [runEdgeBatches] at h
  | cons b bs ih =>
    intro s d r h
    have hF : skipEdges t s d = t.filter (atOrAfter s d) := skipEdges_eq t hs s d
    have hFs : (skipEdges t s d).Pairwise pairLe := by rw [hF]; exact List.Pairwise.sublist List.filter_sublist hs
    obtain ⟨c, hc1, hc2, _, hc4⟩ := edgeLoop_spec b recIdx (fun e => [e]) (fun e out => rfl) (skipEdges t s d) hFs 0 ⟨[], []⟩
    have hfl : ∀ l : List (Nat × Nat × Nat), l.flatMap (fun e => [e.2.2]) = l.map (·.2.2) := by
      intro l; induction l with
      | nil => rfl
      | cons x xs ih => simp [ih]
    have hcm : (edgeLoop b recIdx 0 (skipEdges t s d) ⟨[], []⟩).1.scalar = c.map (·.2.2) := by
      rw [hc2, hfl]; rfl
    rcases hL : (edgeLoop b recIdx 0 (skipEdges t s d) ⟨[], []⟩).2 with _ | ⟨h0, r'⟩
    · rw [hL] at hc1
      have hEB : edgeBatch t s d b = (c.map (·.2.2), none) := by simp [edgeBatch, hL, hcm]
      rw [runEdgeBatches, hEB] at h
      simp only [Prod.mk.injEq, and_true] at h
      subst h
      rw [← hF, ← hc1]; simp
    · rw [hL] at hc1 hc4
      have hEB : edgeBatch t s d b = (c.map (·.2.2), some (h0.1, h0.2.1)) := by simp [edgeBatch, hL, hcm]
      rw [runEdgeBatches, hEB] at h
      simp only [] at h
      have hc4' := hc4 h0 r' rfl
      have hmem : h0 ∈ skipEdges t s d := by rw [← hc1]; simp
      have hp : atOrAfter s d h0 = true := by rw [hF] at hmem; exact (List.mem_filter.1 hmem).2
      have hclaim : t.filter (atOrAfter h0.1 h0.2.1) = h0 :: r' := by
        have step1 : t.filter (atOrAfter h0.1 h0.2.1) = (t.filter (atOrAfter s d)).filter (atOrAfter h0.1 h0.2.1) := by
          rw [List.filter_filter]
          apply List.filter_congr
          intro e _
          simp only [atOrAfter, Bool.and_eq_true, decide_eq_true_eq] at hp ⊢
          by_cases h1 : (h0.1 < e.1 ∨ (h0.1 = e.1 ∧ h0.2.1 ≤ e.2.1)) <;> simp [h1] <;> omega
        rw [step1, ← hF, ← hc1, List.filter_append]
        have e1 : c.filter (atOrAfter h0.1 h0.2.1) = [] := by
          rw [List.filter_eq_nil_iff]
          intro e he
          have := hc4' e he
          simp only [pairLt] at this
          simp only [atOrAfter, decide_eq_true_eq, not_or, not_and, Nat.not_le]
          omega
        have hsorted : (c ++ h0 :: r').Pairwise pairLe := by rw [hc1]; exact hFs
        have e2 : (h0 :: r').filter (atOrAfter h0.1 h0.2.1) = h0 :: r' := by
          rw [List.filter_eq_self]
          intro e he
          simp only [atOrAfter, decide_eq_true_eq]
          rcases List.mem_cons.1 he with rfl | he
          · omega
          · have := List.rel_of_pairwise_cons (List.pairwise_append.1 hsorted).2.1 he
            simp only [pairLe] at this; omega
        rw [e1, e2]; rfl
      have hrec := ih h0.1 h0.2.1 (runEdgeBatches t h0.1 h0.2.1 bs).1
        (Prod.ext rfl (by have := (Prod.ext_iff.1 h).2; simpa using this))
      have hr := (Prod.ext_iff.1 h).1
      simp only at hr
      rw [← hr, hrec, hclaim, ← hF, ← hc1]
      simp

end NI
