/-
CREATE execution: `Runtime::create_batch` (create.rs:139), `CreateOp::new` (:102) and
`CreateOp::next` (:121).

A batch = its active rows (`Nat → Option Val`, slot ↦ value); `write_column(alias, vals)`
writes `vals[k]` into active row `k` (`write_scatter`, columnar `write_column` theorem).
`Pending` is an event log. FFI-free parameters: the id-space `reserve` (proved in
proofs/id_space: `n` fresh ids). Since #2846 it is the *graph's* batch that answers —
`g.node_id_space().reserve(n, &pending.created_nodes)` (create.rs:158) and
`g.relationship_id_space().reserve(n, &pending.taken_relationship_ids)` (:280) — and the
query lends only what it still holds (`reserveN`/`reserveR` read the log: the created,
not-yet-cancelled ids); cancelled ids are excluded by the space itself. The per-row attribute evaluator (template or map path,
CreateAttrs.lean), property validation, and the "endpoint gone" predicate
(`g.is_node_deleted && !pending.is_node_created || pending.is_node_deleted`).
-/
namespace CreateBatch

inductive Val (W : Type) where
  | node (n : Nat)
  | rel (n : Nat)
  | other (w : W)

abbrev Row (W : Type) := Nat → Option (Val W)

def upd {W : Type} (r : Row W) (k : Nat) (v : Val W) : Row W := fun j => if j = k then some v else r j

inductive Ev (W L : Type) where
  | createdNodes (ids : List Nat)
  | labels (ids : List Nat) (ls : List L)
  | nodeAttrs (id : Nat) (a : List (Nat × Val W))
  | createdRel (id src dst : Nat) (ty : String)
  | relAttrs (id : Nat) (a : List (Nat × Val W))

structure PNode (L A : Type) where
  alias : Nat
  labels : List L
  attrs : A

structure PRel (A : Type) where
  alias : Nat
  ty : String          -- `rel.types.first().unwrap()`
  src : Nat
  dst : Nat
  attrs : A

structure Ctx (W L A : Type) where
  reserveN : List (Ev W L) → Nat → Except String (List Nat)
  reserveR : List (Ev W L) → Nat → Except String (List Nat)
  /-- per-row attributes; `none` = empty template (no evaluation, no set). -/
  evAttrs : A → Row W → Except String (Option (List (Nat × Val W)))
  valid : Val W → Except String Unit
  gone : List (Ev W L) → Nat → Bool

def chkAll {W : Type} (valid : Val W → Except String Unit) : List (Nat × Val W) → Except String Unit
  | [] => .ok ()
  | (_, v) :: as => do valid v; chkAll valid as

/-- `pending.set_{node,relationship}_attributes(id, attrs)?` for every row (create.rs:225-230):
validate, then record unless empty. -/
def setAttrs {W L : Type} (valid : Val W → Except String Unit) (mk : Nat → List (Nat × Val W) → Ev W L) :
    List (Ev W L) → List (Nat × List (Nat × Val W)) → Except String (List (Ev W L))
  | log, [] => .ok log
  | log, (id, a) :: rest => do
    chkAll valid a
    setAttrs valid mk (if a.isEmpty then log else log ++ [mk id a]) rest

/-- One pattern node over the batch (create.rs:150-236). -/
def createNode {W L A : Type} (c : Ctx W L A) (log : List (Ev W L)) (rows : List (Row W))
    (n : PNode L A) : Except String (List (Ev W L) × List (Row W)) := do
  let ids ← c.reserveN log rows.length
  let log1 := log ++ [.createdNodes ids, .labels ids n.labels]
  let attrs ← rows.mapM (c.evAttrs n.attrs)
  let log2 ← setAttrs c.valid .nodeAttrs log1
    ((ids.zip attrs).filterMap fun p => p.2.map (p.1, ·))
  pure (log2, (rows.zip ids).map fun p => upd p.1 n.alias (.node p.2))

/-- create.rs:246-251 / :260-265: both endpoints must be node values. -/
def endpoint {W : Type} (r : Row W) (src dst : Nat) : Except String (Nat × Nat) :=
  match r src, r dst with
  | some (.node a), some (.node b) => .ok (a, b)
  | _, _ => .error "Invalid node id"

/-- One pattern relationship (create.rs:238-324). `skip` = both endpoints are created by this
very pattern (create.rs:239-240), which skips the deleted-endpoint check. -/
def createRel {W L A : Type} (c : Ctx W L A) (skip : Bool) (log : List (Ev W L)) (rows : List (Row W))
    (r : PRel A) : Except String (List (Ev W L) × List (Row W)) := do
  let eps ← rows.mapM fun row => do
    let (a, b) ← endpoint row r.src r.dst
    if !skip && (c.gone log a || c.gone log b) then
      throw "Failed to create relationship; endpoint was not found."
    pure (a, b)
  let ids ← c.reserveR log eps.length
  let log1 := log ++ (ids.zip eps).map fun p => .createdRel p.1 p.2.1 p.2.2 r.ty
  let attrs ← rows.mapM (c.evAttrs r.attrs)
  let log2 ← setAttrs c.valid .relAttrs log1
    ((ids.zip attrs).filterMap fun p => p.2.map (p.1, ·))
  pure (log2, (rows.zip ids).map fun p => upd p.1 r.alias (.rel p.2))

def foldNodes {W L A : Type} (c : Ctx W L A) : List (PNode L A) → List (Ev W L) × List (Row W) →
    Except String (List (Ev W L) × List (Row W))
  | [], s => .ok s
  | n :: ns, (log, rows) => do let s ← createNode c log rows n; foldNodes c ns s

def foldRels {W L A : Type} (c : Ctx W L A) (created : List Nat) : List (PRel A) →
    List (Ev W L) × List (Row W) → Except String (List (Ev W L) × List (Row W))
  | [], s => .ok s
  | r :: rs, (log, rows) => do
    let s ← createRel c (created.contains r.src && created.contains r.dst) log rows r
    foldRels c created rs s

/-- `Runtime::create_batch` (create.rs:139-339): all nodes, then all relationships. -/
def createBatch {W L A : Type} (c : Ctx W L A) (nodes : List (PNode L A)) (rels : List (PRel A))
    (log : List (Ev W L)) (rows : List (Row W)) : Except String (List (Ev W L) × List (Row W)) := do
  let s ← foldNodes c nodes (log, rows)
  foldRels c (nodes.map (·.alias)) rels s

/-! ## Theorems -/

theorem setAttrs_prefix {W L : Type} (valid : Val W → Except String Unit) (mk : Nat → List (Nat × Val W) → Ev W L)
    (log : List (Ev W L)) (xs : List (Nat × List (Nat × Val W))) (out : List (Ev W L))
    (h : setAttrs valid mk log xs = .ok out) : ∃ ext, out = log ++ ext := by
  induction xs generalizing log with
  | nil => simp [setAttrs] at h; exact ⟨[], by simp [h]⟩
  | cons x xs ih =>
    obtain ⟨id, a⟩ := x
    simp only [setAttrs, bind, Except.bind] at h
    split at h
    · cases h
    · obtain ⟨ext, he⟩ := ih _ h
      split at he
      · exact ⟨ext, he⟩
      · exact ⟨mk id a :: ext, by simp [he]⟩

/-- **CREATE (node)**: with `reserve` returning one id per active row, every active row `k`
gets its own new node `ids[k]` bound at the alias (other slots untouched), the ids are
logged as created (once) and labelled, before any attribute event. -/
theorem createNode_spec {W L A : Type} (c : Ctx W L A) (log : List (Ev W L)) (rows : List (Row W))
    (n : PNode L A) (out : List (Ev W L) × List (Row W)) (ids : List Nat)
    (hr : c.reserveN log rows.length = .ok ids) (hlen : ids.length = rows.length)
    (h : createNode c log rows n = .ok out) :
    (∃ ext, out.1 = log ++ [.createdNodes ids, .labels ids n.labels] ++ ext) ∧
    out.2.length = rows.length ∧
    ∀ k (hk : k < rows.length) (hk' : k < out.2.length),
      out.2[k] n.alias = some (.node (ids[k]'(by omega))) ∧
      ∀ j, j ≠ n.alias → out.2[k] j = rows[k] j := by
  unfold createNode at h
  simp only [hr, bind, Except.bind] at h
  split at h
  · cases h
  · rename_i attrs _
    split at h
    · cases h
    · rename_i log2 hs
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨ext, he⟩ := setAttrs_prefix _ _ _ _ _ hs
      refine ⟨⟨ext, by simp [he]⟩, by simp [hlen], fun k hk hk' => ?_⟩
      simp only [List.getElem_map, List.getElem_zip]
      exact ⟨by simp [upd], fun j hj => by simp [upd, hj]⟩

/-- A row whose endpoint slot is not a node fails the whole batch with "Invalid node id". -/
theorem endpoint_spec {W : Type} (r : Row W) (s d : Nat) (a b : Nat) :
    endpoint r s d = .ok (a, b) ↔ r s = some (.node a) ∧ r d = some (.node b) := by
  unfold endpoint
  split
  · rename_i a' b' h1 h2; simp [h1, h2]
  · rename_i hn
    simp only [reduceCtorEq, false_iff]
    rintro ⟨h1, h2⟩; exact hn a b h1 h2

theorem mapM_ok_get {α β ε : Type} (f : α → Except ε β) (xs : List α) (ys : List β)
    (h : xs.mapM f = .ok ys) : ys.length = xs.length ∧ ∀ i (h1 : i < xs.length) (h2 : i < ys.length),
      f xs[i] = .ok ys[i] := by
  induction xs generalizing ys with
  | nil => simp [List.mapM_nil, pure, Except.pure] at h; subst h; simp
  | cons x xs ih =>
    simp only [List.mapM_cons, bind, Except.bind] at h
    cases hx : f x with
    | error e => simp [hx] at h
    | ok y =>
      simp only [hx] at h
      cases hxs : xs.mapM f with
      | error e => simp [hxs] at h
      | ok ys' =>
        simp only [hxs, pure, Except.pure, Except.ok.injEq] at h
        subst h
        obtain ⟨hl, hg⟩ := ih ys' hxs
        refine ⟨by simp [hl], fun i h1 h2 => ?_⟩
        cases i with
        | zero => simpa using hx
        | succ i => simpa using hg i (by simpa using h1) (by simpa using h2)

/-- **CREATE (relationship)**: on success one relationship per active row is logged with that
row's endpoints and the pattern's type, and row `k` binds `ids[k]`; when not both endpoints
are created by this pattern, none of them is a deleted node. -/
theorem createRel_spec {W L A : Type} (c : Ctx W L A) (skip : Bool) (log : List (Ev W L))
    (rows : List (Row W)) (r : PRel A) (out : List (Ev W L) × List (Row W))
    (h : createRel c skip log rows r = .ok out) :
    ∃ (ids : List Nat) (eps : List (Nat × Nat)), c.reserveR log eps.length = .ok ids ∧ eps.length = rows.length ∧
      (∀ k (h1 : k < rows.length) (h2 : k < eps.length),
        rows[k] r.src = some (.node eps[k].1) ∧ rows[k] r.dst = some (.node eps[k].2) ∧
        (skip = false → c.gone log eps[k].1 = false ∧ c.gone log eps[k].2 = false)) ∧
      (∃ ext, out.1 = log ++ (ids.zip eps).map (fun p => .createdRel p.1 p.2.1 p.2.2 r.ty) ++ ext) ∧
      out.2 = (rows.zip ids).map fun p => upd p.1 r.alias (.rel p.2) := by
  unfold createRel at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i eps heps
    split at h
    · cases h
    · rename_i ids hids
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i log2 hs
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          obtain ⟨hl, hg⟩ := mapM_ok_get _ _ _ heps
          obtain ⟨ext, he⟩ := setAttrs_prefix _ _ _ _ _ hs
          refine ⟨ids, eps, hids, hl, fun k h1 h2 => ?_, ⟨ext, by simp [he]⟩, rfl⟩
          have := hg k h1 h2
          cases hep : endpoint rows[k] r.src r.dst with
          | error e => simp [hep, bind, Except.bind] at this
          | ok ab =>
            obtain ⟨a, b⟩ := ab
            have hab := (endpoint_spec _ _ _ a b).mp hep
            simp only [hep, bind, Except.bind] at this
            split at this
            · cases this
            · rename_i hg'
              simp only [pure, Except.pure, Except.ok.injEq] at this
              rw [← this]
              refine ⟨hab.1, hab.2, fun hsk => ?_⟩
              subst hsk
              simp only [Bool.not_false, Bool.true_and, Bool.or_eq_true, not_or] at hg'
              exact ⟨by simpa using hg'.1, by simpa using hg'.2⟩

/-- A failing endpoint check aborts with the C engine's message. -/
theorem createRel_gone {W L A : Type} (c : Ctx W L A) (log : List (Ev W L)) (row : Row W)
    (r : PRel A) (a b : Nat) (ha : row r.src = some (.node a)) (hb : row r.dst = some (.node b))
    (hg : c.gone log a = true) :
    createRel c false log [row] r = .error "Failed to create relationship; endpoint was not found." := by
  simp [createRel, endpoint, ha, hb, hg, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-! ## `CreateOp::new` / `next` (create.rs:102, :121) -/

structure CreateSt (C P R : Type) where
  child : C
  pattern : P
  resolved : Option R

def CreateSt.new {C P R : Type} (child : C) (pattern : P) : CreateSt C P R :=
  { child, pattern, resolved := none }

theorem createNew_spec {C P R : Type} (c : C) (p : P) :
    (CreateSt.new c p : CreateSt C P R).resolved = none ∧ (CreateSt.new c p : CreateSt C P R).child = c ∧
    (CreateSt.new c p : CreateSt C P R).pattern = p := ⟨rfl, rfl, rfl⟩

/-- create.rs:121-136: pull one child batch, pass its error through, resolve the pattern once
(`OnceCell`), run `create_batch` on it, pass its error through. -/
def createNext {B R : Type} (resolve : Unit → R) (create : R → B → Except String B) (cell : Option R) :
    Option (Except String B) → Option (Except String B) × Option R
  | none => (none, cell)
  | some (.error e) => (some (.error e), cell)
  | some (.ok b) =>
    let r := match cell with | some r => r | none => resolve ()
    (some (create r b), some r)

/-- The output stream is the child stream with `create_batch` applied to each batch
(errors propagated), using the once-resolved pattern. -/
theorem createNext_spec {B R : Type} (resolve : Unit → R) (create : R → B → Except String B)
    (cell : Option R) (hc : ∀ r, cell = some r → r = resolve ()) (x : Option (Except String B)) :
    (createNext resolve create cell x).1 = x.map (fun eb => eb.bind (create (resolve ()))) := by
  cases x with
  | none => rfl
  | some eb =>
    cases eb with
    | error e => rfl
    | ok b =>
      cases cell with
      | none => rfl
      | some r => have := hc r rfl; subst this; rfl

end CreateBatch
