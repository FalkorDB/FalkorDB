import GraphPersist.Persist.KeyRT
/-! # Folding payloads in any order

`fold_char` describes the accumulated locals after *any* list of decoded payloads, as a
function of which ids / matrices the list carries — not of their order. The multi-key
load (`Pending`) relies on it: Redis hands the keys of one graph over in keyspace order,
which need not be key order.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

attribute [ext] Acc

def nodeIds : List (PR M Tn) → List Nat
  | [] => []
  | .nodes es :: ps => es.map Prod.fst ++ nodeIds ps
  | _ :: ps => nodeIds ps

def edgeIds : List (PR M Tn) → List Nat
  | [] => []
  | .edges es :: ps => es.map Prod.fst ++ edgeIds ps
  | _ :: ps => edgeIds ps

def delNIds : List (PR M Tn) → List Nat
  | [] => []
  | .delN l :: ps => l ++ delNIds ps
  | _ :: ps => delNIds ps

def delEIds : List (PR M Tn) → List Nat
  | [] => []
  | .delE l :: ps => l ++ delEIds ps
  | _ :: ps => delEIds ps

def lmsOf : List (PR M Tn) → List M
  | [] => []
  | .lm ms :: ps => ms ++ lmsOf ps
  | _ :: ps => lmsOf ps

def tnsOf : List (PR M Tn) → List Tn
  | [] => []
  | .rt ts :: ps => ts ++ tnsOf ps
  | _ :: ps => tnsOf ps

/-- The last adjacency matrix in the list, else `d`. -/
def adjOf : List (PR M Tn) → M → M
  | [], d => d
  | .adj m :: ps, _ => adjOf ps m
  | _ :: ps, d => adjOf ps d

def lblsOf : List (PR M Tn) → M → M
  | [], d => d
  | .lbls m :: ps, _ => lblsOf ps m
  | _ :: ps, d => lblsOf ps d

/-- Every entity payload carries real spans of the graph. -/
def Faithful (g : G M Tn Nm Ix Cn) : PR M Tn → Prop
  | .nodes es => ∃ ids : List Nat, es = ids.map (rawOf g.nodes)
  | .edges es => ∃ ids : List Nat, es = ids.map (rawOf g.edges)
  | _ => True

theorem map_fst_rawOf (S : Store) (ids : List Nat) : (ids.map (rawOf S)).map Prod.fst = ids := by
  induction ids with
  | nil => rfl
  | cons i is ih => simp only [List.map_cons, ih]; rfl

/-- **The fold, characterised.** -/
theorem fold_char (g : G M Tn Nm Ix Cn) (limit : Nat) (hl : limit ≤ 65536)
    (hN : ∀ i sp, g.nodes i = some sp → GoodSpan limit sp) (hE : ∀ i sp, g.edges i = some sp → GoodSpan limit sp) :
    ∀ (prs : List (PR M Tn)) (a : Acc M Tn), (∀ p ∈ prs, Faithful g p) →
    prs.foldl (applyPR limit) a =
      ⟨fun j => if j ∈ nodeIds prs ∧ g.nodes j ≠ none then g.nodes j else a.nodes j,
       fun j => if j ∈ edgeIds prs ∧ g.edges j ≠ none then g.edges j else a.edges j,
       fun j => a.delN j || decide (j ∈ delNIds prs),
       fun j => a.delE j || decide (j ∈ delEIds prs),
       a.lms ++ lmsOf prs, a.tns ++ tnsOf prs, adjOf prs a.adj, lblsOf prs a.lbls⟩
  | [], a, _ => by obtain ⟨_, _, _, _, _, _, _, _⟩ := a; simp [nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf]
  | p :: ps, a, hf => by
    have ih := fold_char g limit hl hN hE ps
    have hps : ∀ q ∈ ps, Faithful g q := fun q hq => hf q (by simp [hq])
    have hp := hf p (by simp)
    simp only [List.foldl_cons]
    rw [ih _ hps]
    cases p with
    | nodes es =>
      obtain ⟨ids, rfl⟩ := hp
      have this : List.foldl (applyEnt limit) a.nodes (ids.map (rawOf g.nodes)) =
          fun j => if j ∈ ids ∧ g.nodes j ≠ none then g.nodes j else a.nodes j :=
        fold_ents limit hl g.nodes ids (fun i _ sp h => hN i sp h) a.nodes
      apply Acc.ext <;> (try rfl)
      show (fun j => if j ∈ nodeIds ps ∧ g.nodes j ≠ none then g.nodes j
          else (List.foldl (applyEnt limit) a.nodes (ids.map (rawOf g.nodes))) j) =
        fun j => if j ∈ nodeIds (PR.nodes (ids.map (rawOf g.nodes)) :: ps) ∧ g.nodes j ≠ none then g.nodes j else a.nodes j
      rw [this]
      funext j
      simp only [nodeIds, map_fst_rawOf, List.mem_append]
      by_cases h1 : j ∈ ids <;> by_cases h2 : j ∈ nodeIds ps <;> by_cases h3 : g.nodes j = none <;>
        simp [h1, h2, h3]
    | edges es =>
      obtain ⟨ids, rfl⟩ := hp
      have this : List.foldl (applyEnt limit) a.edges (ids.map (rawOf g.edges)) =
          fun j => if j ∈ ids ∧ g.edges j ≠ none then g.edges j else a.edges j :=
        fold_ents limit hl g.edges ids (fun i _ sp h => hE i sp h) a.edges
      apply Acc.ext <;> (try rfl)
      show (fun j => if j ∈ edgeIds ps ∧ g.edges j ≠ none then g.edges j
          else (List.foldl (applyEnt limit) a.edges (ids.map (rawOf g.edges))) j) =
        fun j => if j ∈ edgeIds (PR.edges (ids.map (rawOf g.edges)) :: ps) ∧ g.edges j ≠ none then g.edges j else a.edges j
      rw [this]
      funext j
      simp only [edgeIds, map_fst_rawOf, List.mem_append]
      by_cases h1 : j ∈ ids <;> by_cases h2 : j ∈ edgeIds ps <;> by_cases h3 : g.edges j = none <;>
        simp [h1, h2, h3]
    | delN l => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | delE l => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | lm ms => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | rt ts => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | adj m => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | lbls m => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)
    | skip => apply Acc.ext <;> (try funext j) <;> (try simp [applyPR, insAll_spec, nodeIds, edgeIds, delNIds, delEIds, lmsOf, tnsOf, adjOf, lblsOf, Bool.or_assoc]) <;> (try rfl)

/-! ## Order does not matter -/

theorem nodeIds_append (a b : List (PR M Tn)) : nodeIds (a ++ b) = nodeIds a ++ nodeIds b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [nodeIds, ih]
theorem edgeIds_append (a b : List (PR M Tn)) : edgeIds (a ++ b) = edgeIds a ++ edgeIds b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [edgeIds, ih]
theorem delNIds_append (a b : List (PR M Tn)) : delNIds (a ++ b) = delNIds a ++ delNIds b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [delNIds, ih]
theorem delEIds_append (a b : List (PR M Tn)) : delEIds (a ++ b) = delEIds a ++ delEIds b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [delEIds, ih]
theorem lmsOf_append (a b : List (PR M Tn)) : lmsOf (a ++ b) = lmsOf a ++ lmsOf b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [lmsOf, ih]
theorem tnsOf_append (a b : List (PR M Tn)) : tnsOf (a ++ b) = tnsOf a ++ tnsOf b := by
  induction a with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [tnsOf, ih]
theorem adjOf_append (a b : List (PR M Tn)) (d : M) : adjOf (a ++ b) d = adjOf b (adjOf a d) := by
  induction a generalizing d with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [adjOf, ih]
theorem lblsOf_append (a b : List (PR M Tn)) (d : M) : lblsOf (a ++ b) d = lblsOf b (lblsOf a d) := by
  induction a generalizing d with
  | nil => rfl
  | cons p ps ih => cases p <;> simp [lblsOf, ih]

/-- A list-valued field contributed by exactly one key (key 0) comes out the same in any
order of distinct keys that includes it. -/
theorem flatMap_single {β : Type} (f : Nat → List β) (ord : List Nat) (hnd : ord.Nodup) (h0 : 0 ∈ ord)
    (hz : ∀ k, k ≠ 0 → f k = []) : ord.flatMap f = f 0 := by
  induction ord with
  | nil => simp at h0
  | cons k ks ih =>
    rw [List.flatMap_cons]
    have hk := List.nodup_cons.1 hnd
    by_cases hk0 : k = 0
    · subst hk0
      have : ks.flatMap f = [] := by
        rw [List.flatMap_eq_nil_iff]; intro x hx; exact hz x (fun e => hk.1 (e ▸ hx))
      simp [this]
    · rw [hz k hk0, ih hk.2 (by simpa [Ne.symm hk0] using h0)]; rfl

theorem fold_single {β : Type} (step : Nat → β → β) (ord : List Nat) (hnd : ord.Nodup) (h0 : 0 ∈ ord)
    (hz : ∀ k, k ≠ 0 → ∀ d, step k d = d) (hc : ∀ d d', step 0 d = step 0 d') (d : β) :
    ord.foldl (fun b k => step k b) d = step 0 d := by
  induction ord generalizing d with
  | nil => simp at h0
  | cons k ks ih =>
    have hk := List.nodup_cons.1 hnd
    simp only [List.foldl_cons]
    by_cases hk0 : k = 0
    · subst hk0
      have : ∀ e, ks.foldl (fun b k => step k b) e = e := by
        intro e
        have hks : ∀ x ∈ ks, x ≠ 0 := fun x hx e' => hk.1 (e' ▸ hx)
        clear ih hnd hk h0
        induction ks generalizing e with
        | nil => rfl
        | cons x xs ihx =>
          simp only [List.foldl_cons]
          rw [hz x (hks x (by simp)) e, ihx _ (fun y hy => hks y (by simp [hy]))]
      rw [this]
    · rw [hz k hk0 d, ih hk.2 (by simpa [Ne.symm hk0] using h0)]

end GraphPersist.Persist
