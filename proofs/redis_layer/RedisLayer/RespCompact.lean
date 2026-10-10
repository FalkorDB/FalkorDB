import RedisLayer.Resp
/-!
# `reply_compact_value` as a call stream (`src/reply.rs:134-343`)

`encC v` is the exact sequence of `RedisModule_Reply*` calls `reply_compact_value` makes
(tag, then payload; the caller supplies the enclosing `ReplyWithArray(2)`).
`compact_emits`: that sequence emits precisely `Compact.encPair E (proj v)` — the tree the
falkordb-py decoder model in `Compact.lean` is proven against (`dec_enc`) — for every value
kind, live or deleted entities, postponed label arrays included. With `bytes_of_emits` this
is the RESP byte stream.

Graph-API facts used (not code in `src/`): for a live entity, `get_node_attr_count(id)`
equals the number of pairs `get_node_all_attrs_by_id(id)` yields (`reply.rs:232` — if not,
`count_mismatch_breaks` shows the stream desynchronises); for a deleted one the attribute and
type ids are recovered by name (`get_node_attribute_id`, `get_type_id`, `:213`, `:249`),
which we take to be the ids themselves.
-/
namespace RedisLayer.Resp
open RedisLayer.Compact (R Val Env encPair encList encMap encProps)

inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (x : F)
  | str (s : String)
  | datetime (t : Int) | date (t : Int) | time (t : Int) | duration (t : Int)
  | list (xs : List (V F))
  | map (kvs : List (String × V F))
  | node (id : Nat) (deleted : Bool) (labels : List Nat) (props : List (Nat × V F))
  | edge (id ty src dst : Nat) (deleted : Bool) (props : List (Nat × V F))
  | path (items : List (V F))
  | vec (xs : List F)
  | point (lat lon : F)

variable {F : Type} (E : Env F)

def V.isNode : V F → Bool
  | .node .. => true
  | _ => false
def V.isRel : V F → Bool
  | .edge .. => true
  | _ => false

def countNodes : List (V F) → Nat
  | [] => 0
  | v :: vs => (if v.isNode then 1 else 0) + countNodes vs
def countRels : List (V F) → Nat
  | [] => 0
  | v :: vs => (if v.isRel then 1 else 0) + countRels vs

mutual
/-- `reply_compact_value` (`reply.rs:134-343`). -/
def encC : V F → List (Call F)
  | .null => [.ll 1, .null]
  | .bool x => [.ll 4, .str (if x then "true" else "false")]
  | .int x => [.ll 3, .ll x]
  | .float x => [.ll 5, .str (E.fmt x)]
  | .str s => [.ll 2, .str s]
  | .datetime t => [.ll 13, .ll t]
  | .date t => [.ll 14, .ll t]
  | .time t => [.ll 15, .ll t]
  | .duration t => [.ll 16, .ll t]
  | .list xs => [.ll 6, .arr xs.length] ++ encCList xs
  | .map kvs => [.ll 10, .arr (kvs.length * 2)] ++ encCMap kvs
  | .node id del ls ps =>
    [.ll 8, .arr 3, .ll id]
      ++ (if del then .arr ls.length :: ls.map (fun (l : Nat) => Call.ll (l : Int))
          else Call.arrP :: ls.map (fun (l : Nat) => Call.ll (l : Int)) ++ [.setLen ls.length])
      ++ [.arr ps.length] ++ encCProps ps
  | .edge id ty s d _ ps =>
    [.ll 7, .arr 5, .ll id, .ll ty, .ll s, .ll d, .arr ps.length] ++ encCProps ps
  | .path items =>
    [.ll 9, .arr 2, .arr 2, .ll 6, .arr (countNodes items)] ++ encCNodes items
      ++ [.arr 2, .ll 6, .arr (countRels items)] ++ encCRels items
  | .vec xs => [.ll 12, .arr xs.length] ++ xs.map Call.dbl
  | .point a o => [.ll 11, .arr 2, .str (E.fmt a), .str (E.fmt o)]
def encCList : List (V F) → List (Call F)
  | [] => []
  | v :: vs => (.arr 2 :: encC v) ++ encCList vs
def encCMap : List (String × V F) → List (Call F)
  | [] => []
  | (k, v) :: kvs => (.str k :: .arr 2 :: encC v) ++ encCMap kvs
def encCProps : List (Nat × V F) → List (Call F)
  | [] => []
  | (k, v) :: ps => (.arr 3 :: .ll k :: encC v) ++ encCProps ps
def encCNodes : List (V F) → List (Call F)
  | [] => []
  | v :: vs => (if v.isNode then .arr 2 :: encC v else []) ++ encCNodes vs
def encCRels : List (V F) → List (Call F)
  | [] => []
  | v :: vs => (if v.isRel then .arr 2 :: encC v else []) ++ encCRels vs
end

mutual
/-- The value the reply denotes, in the `Compact` model. -/
def proj : V F → Val F
  | .null => .null
  | .bool b => .bool b
  | .int i => .int i
  | .float x => .float x
  | .str s => .str s
  | .datetime t => .datetime t
  | .date t => .date t
  | .time t => .time t
  | .duration t => .duration t
  | .list xs => .list (projList xs)
  | .map kvs => .map (projMap kvs)
  | .node id _ ls ps => .node id ls (projProps ps)
  | .edge id ty s d _ ps => .edge id ty s d (projProps ps)
  | .path items => .path (projNodes items) (projRels items)
  | .vec xs => .vec xs
  | .point a o => .point a o
def projList : List (V F) → List (Val F)
  | [] => []
  | v :: vs => proj v :: projList vs
def projMap : List (String × V F) → List (String × Val F)
  | [] => []
  | (k, v) :: kvs => (k, proj v) :: projMap kvs
def projProps : List (Nat × V F) → List (Nat × Val F)
  | [] => []
  | (k, v) :: ps => (k, proj v) :: projProps ps
def projNodes : List (V F) → List (Val F)
  | [] => []
  | v :: vs => if v.isNode then proj v :: projNodes vs else projNodes vs
def projRels : List (V F) → List (Val F)
  | [] => []
  | v :: vs => if v.isRel then proj v :: projRels vs else projRels vs
end

theorem encPair_len2 (v : Val F) : (encPair E v).length = 2 := by
  cases v <;> rfl

theorem list_len (xs : List (V F)) : (encList E (projList xs)).length = xs.length := by
  induction xs <;> simp_all [encList, projList]
theorem map_len (kvs : List (String × V F)) : (encMap E (projMap kvs)).length = kvs.length * 2 := by
  induction kvs with
  | nil => rfl
  | cons kv kvs ih => obtain ⟨k, v⟩ := kv; simp [encMap, projMap, ih]; omega
theorem props_len (ps : List (Nat × V F)) : (encProps E (projProps ps)).length = ps.length := by
  induction ps with
  | nil => rfl
  | cons p ps ih => obtain ⟨k, v⟩ := p; simp [encProps, projProps, ih]
theorem nodes_len (xs : List (V F)) : (encList E (projNodes xs)).length = countNodes xs := by
  induction xs with
  | nil => rfl
  | cons v vs ih => by_cases h : v.isNode <;> simp [projNodes, countNodes, h, encList, ih]; omega
theorem rels_len (xs : List (V F)) : (encList E (projRels xs)).length = countRels xs := by
  induction xs with
  | nil => rfl
  | cons v vs ih => by_cases h : v.isRel <;> simp [projRels, countRels, h, encList, ih]; omega

theorem emits_lls (ls : List Nat) :
    Emits E.vfmt (ls.map fun (l : Nat) => Call.ll (F := F) (l : Int)) (ls.map fun (l : Nat) => R.int (l : Int)) := by
  induction ls with
  | nil => exact Emits.nil _
  | cons l ls ih =>
    have := (Emits.leaf_ll E.vfmt (F := F) l).append _ ih
    simpa using this

theorem emits_dbls (xs : List F) :
    Emits E.vfmt (xs.map Call.dbl) (xs.map fun x => R.bulk (E.vfmt x)) := by
  induction xs with
  | nil => exact Emits.nil _
  | cons x xs ih =>
    have := (Emits.leaf_dbl E.vfmt x).append _ ih
    simpa using this

theorem emits_cons {c : Call F} {x : R} {cs : List (Call F)} {xs : List R}
    (h1 : Emits E.vfmt [c] [x]) (h2 : Emits E.vfmt cs xs) :
    Emits E.vfmt (c :: cs) (x :: xs) := by
  have := h1.append _ h2; simpa using this

mutual
/-- **Compact encoder = spec tree**, for every value kind. -/
theorem compact_emits : ∀ v : V F, Emits E.vfmt (encC E v) (encPair E (proj v))
  | .null => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_null _)
  | .bool x => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_str _ _)
  | .int x => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_ll _ _)
  | .float x => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_str _ _)
  | .str s => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_str _ _)
  | .datetime t => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_ll _ _)
  | .date t => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_ll _ _)
  | .time t => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_ll _ _)
  | .duration t => emits_cons E (Emits.leaf_ll _ _) (Emits.leaf_ll _ _)
  | .list xs => by
    have h := (list_emits xs).array
    rw [list_len] at h
    exact emits_cons E (Emits.leaf_ll _ _) h
  | .map kvs => by
    have h := (map_emits kvs).array
    rw [map_len] at h
    exact emits_cons E (Emits.leaf_ll _ _) h
  | .node id del ls ps => by
    have hl : Emits E.vfmt
        (if del then .arr ls.length :: ls.map (fun (l : Nat) => Call.ll (l : Int))
         else Call.arrP :: ls.map (fun (l : Nat) => Call.ll (l : Int)) ++ [.setLen ls.length])
        [.arr (ls.map fun (l : Nat) => R.int (l : Int))] := by
      have h := emits_lls E ls
      cases del
      · have := h.postponed; simpa using this
      · have := h.array; simpa using this
    have hp := (props_emits ps).array
    rw [props_len] at hp
    have hin := (((Emits.leaf_ll E.vfmt (F := F) id).append _ hl).append _ hp).array
    have := emits_cons E (Emits.leaf_ll E.vfmt (F := F) 8) hin
    simpa [encC, encPair, proj] using this
  | .edge id ty s d del ps => by
    have hp := (props_emits ps).array
    rw [props_len] at hp
    have h5 := (emits_cons E (Emits.leaf_ll _ (id : Int)) (emits_cons E (Emits.leaf_ll _ (ty : Int))
      (emits_cons E (Emits.leaf_ll _ (s : Int)) (emits_cons E (Emits.leaf_ll _ (d : Int)) hp)))).array
    have := emits_cons E (Emits.leaf_ll E.vfmt (F := F) 7) h5
    simpa [encC, encPair, proj] using this
  | .path items => by
    have hn := (nodes_emits items).array
    rw [nodes_len] at hn
    have hr := (rels_emits items).array
    rw [rels_len] at hr
    have a1 := (emits_cons E (Emits.leaf_ll E.vfmt (F := F) 6) hn).array
    have a2 := (emits_cons E (Emits.leaf_ll E.vfmt (F := F) 6) hr).array
    have hb := (a1.append _ a2).array
    have := emits_cons E (Emits.leaf_ll E.vfmt (F := F) 9) hb
    simpa [encC, encPair, proj] using this
  | .vec xs => by
    have h := (emits_dbls E xs).array
    have := emits_cons E (Emits.leaf_ll E.vfmt (F := F) 12) h
    simpa [encC, encPair, proj] using this
  | .point a o => by
    have h := (emits_cons E (Emits.leaf_str E.vfmt (F := F) (E.fmt a)) (Emits.leaf_str _ (E.fmt o))).array
    have := emits_cons E (Emits.leaf_ll E.vfmt (F := F) 11) h
    simpa [encC, encPair, proj] using this
theorem list_emits : ∀ xs : List (V F), Emits E.vfmt (encCList E xs) (encList E (projList xs))
  | [] => Emits.nil _
  | v :: vs => by
    have h := (compact_emits v).array
    rw [encPair_len2] at h
    have := h.append _ (list_emits vs)
    simpa [encCList, encList, projList] using this
theorem map_emits : ∀ kvs : List (String × V F), Emits E.vfmt (encCMap E kvs) (encMap E (projMap kvs))
  | [] => Emits.nil _
  | (k, v) :: kvs => by
    have h := (compact_emits v).array
    rw [encPair_len2] at h
    have := (emits_cons E (Emits.leaf_str E.vfmt k) h).append _ (map_emits kvs)
    simpa [encCMap, encMap, projMap] using this
theorem props_emits : ∀ ps : List (Nat × V F), Emits E.vfmt (encCProps E ps) (encProps E (projProps ps))
  | [] => Emits.nil _
  | (k, v) :: ps => by
    have h := emits_cons E (Emits.leaf_ll E.vfmt (F := F) k) (compact_emits v)
    have h3 := h.array
    rw [show (R.int (k : Int) :: encPair E (proj v)).length = 3 by
      simp [encPair_len2]] at h3
    have := h3.append _ (props_emits ps)
    simpa [encCProps, encProps, projProps] using this
theorem nodes_emits : ∀ xs : List (V F), Emits E.vfmt (encCNodes E xs) (encList E (projNodes xs))
  | [] => Emits.nil _
  | v :: vs => by
    by_cases hv : v.isNode
    · have h := (compact_emits v).array
      rw [encPair_len2] at h
      have := h.append _ (nodes_emits vs)
      simpa [encCNodes, encList, projNodes, hv] using this
    · simpa [encCNodes, projNodes, hv] using nodes_emits vs
theorem rels_emits : ∀ xs : List (V F), Emits E.vfmt (encCRels E xs) (encList E (projRels xs))
  | [] => Emits.nil _
  | v :: vs => by
    by_cases hv : v.isRel
    · have h := (compact_emits v).array
      rw [encPair_len2] at h
      have := h.append _ (rels_emits vs)
      simpa [encCRels, encList, projRels, hv] using this
    · simpa [encCRels, projRels, hv] using rels_emits vs
end

end RedisLayer.Resp
