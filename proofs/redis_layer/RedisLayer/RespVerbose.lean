import RedisLayer.RespCompact
/-!
# Verbose values, statistics and whole replies (`src/reply.rs`)

| here | there |
| --- | --- |
| `fmtS` / `fmtList` / `fmtMap` | `format_value_to_string` `:36-100` |
| `encVb`, `specVb`             | `reply_verbose_value` `:346-528` |
| `statCalls`, `statLines`      | `reply_stats` `:530-617` |
| `resultCalls`                 | `reply_result::<COMPACT>` `:619-666`, `reply_verbose` `:668`, `reply_compact` `:676` |
| `Fmt.g15`                     | `format_g(x, 15)` `:112-131` (libc `snprintf`, AXIOMATISED as an abstract function) |

Number/temporal formatting is abstract (`Fmt`): `{:.6}`, `%.15g`, `Value::format_*`.
-/
namespace RedisLayer.Resp
open RedisLayer.Compact (R Val Env encPair encReply)

variable {F : Type}

structure Fmt (F : Type) where
  f6 : F → String            -- `{:.6}`
  dt : Int → String          -- `Value::format_datetime`
  date : Int → String
  time : Int → String
  dur : Int → String
  label : Nat → String       -- `get_label_by_id` / `get_node_labels`
  prop : Nat → String        -- attribute name
  rel : Nat → String         -- `get_type(type_id).unwrap_or("")`

variable (E : Env F) (M : Fmt F)

def joinSep : List String → String
  | [] => ""
  | [s] => s
  | s :: ss => s ++ ", " ++ joinSep ss

mutual
/-- `format_value_to_string` (`:36-100`). -/
def fmtS : V F → String
  | .null => "NULL"
  | .bool b => if b then "true" else "false"
  | .int i => toString i
  | .float x => M.f6 x
  | .str s => s
  | .datetime t => M.dt t
  | .date t => M.date t
  | .time t => M.time t
  | .duration t => M.dur t
  | .list xs => "[" ++ joinSep (fmtList xs) ++ "]"
  | .path xs => "[" ++ joinSep (fmtList xs) ++ "]"
  | .map kvs => "{" ++ joinSep (fmtMap kvs) ++ "}"
  | .node id .. => "(" ++ toString id ++ ")"
  | .edge id .. => "[" ++ toString id ++ "]"
  | .vec xs => "<" ++ joinSep (xs.map M.f6) ++ ">"
  | .point a o => "point({latitude: " ++ M.f6 a ++ ", longitude: " ++ M.f6 o ++ "})"
def fmtList : List (V F) → List String
  | [] => []
  | v :: vs => fmtS v :: fmtList vs
def fmtMap : List (String × V F) → List String
  | [] => []
  | (k, v) :: kvs => (k ++ ": " ++ fmtS v) :: fmtMap kvs
end

def pointStr (a o : F) : String :=
  "point({latitude: " ++ M.f6 a ++ ", longitude: " ++ M.f6 o ++ "})"

mutual
/-- `reply_verbose_value` (`:346-528`). -/
def encVb : V F → List (Call F)
  | .null => [.null]
  | .bool x => [.str (if x then "true" else "false")]
  | .int x => [.ll x]
  | .float x => [.str (E.fmt x)]
  | .str s => [.str s]
  | .datetime t => [.str (M.dt t)]
  | .date t => [.str (M.date t)]
  | .time t => [.str (M.time t)]
  | .duration t => [.str (M.dur t)]
  | .list xs => [.str (fmtS M (.list xs))]
  | .map kvs => [.str (fmtS M (.map kvs))]
  | .path xs => [.str (fmtS M (.path xs))]
  | .vec xs => [.str (fmtS M (.vec xs))]
  | .node id del ls ps =>
    [.arr 3, .arr 2, .str "id", .ll id, .arr 2, .str "labels"]
      ++ (if del then .arr ls.length :: ls.map (fun l => Call.str (M.label l))
          else Call.arrP :: ls.map (fun l => Call.str (M.label l)) ++ [.setLen ls.length])
      ++ [.arr 2, .str "properties"]
      ++ (if del then .arr ps.length :: encVbProps ps
          else Call.arrP :: encVbProps ps ++ [.setLen ps.length])
  | .edge id ty s d del ps =>
    [.arr 5, .arr 2, .str "id", .ll id, .arr 2, .str "type", .str (M.rel ty),
     .arr 2, .str "src_node", .ll s, .arr 2, .str "dest_node", .ll d, .arr 2, .str "properties"]
      ++ (if del then .arr ps.length :: encVbProps ps
          else Call.arrP :: encVbProps ps ++ [.setLen ps.length])
  | .point a o => [.str (pointStr M a o)]
def encVbProps : List (Nat × V F) → List (Call F)
  | [] => []
  | (k, v) :: ps => (.arr 2 :: .str (M.prop k) :: encVb v) ++ encVbProps ps
end

mutual
/-- The documented verbose shape. -/
def specVb : V F → R
  | .null => .nil
  | .bool x => .bulk (if x then "true" else "false")
  | .int x => .int x
  | .float x => .bulk (E.fmt x)
  | .str s => .bulk s
  | .datetime t => .bulk (M.dt t)
  | .date t => .bulk (M.date t)
  | .time t => .bulk (M.time t)
  | .duration t => .bulk (M.dur t)
  | .list xs => .bulk (fmtS M (.list xs))
  | .map kvs => .bulk (fmtS M (.map kvs))
  | .path xs => .bulk (fmtS M (.path xs))
  | .vec xs => .bulk (fmtS M (.vec xs))
  | .node id _ ls ps =>
    .arr [.arr [.bulk "id", .int id], .arr [.bulk "labels", .arr (ls.map fun l => .bulk (M.label l))],
          .arr [.bulk "properties", .arr (specVbProps ps)]]
  | .edge id ty s d _ ps =>
    .arr [.arr [.bulk "id", .int id], .arr [.bulk "type", .bulk (M.rel ty)],
          .arr [.bulk "src_node", .int s], .arr [.bulk "dest_node", .int d],
          .arr [.bulk "properties", .arr (specVbProps ps)]]
  | .point a o => .bulk (pointStr M a o)
def specVbProps : List (Nat × V F) → List R
  | [] => []
  | (k, v) :: ps => .arr [.bulk (M.prop k), specVb v] :: specVbProps ps
end

theorem specVbProps_len (ps : List (Nat × V F)) : (specVbProps E M ps).length = ps.length := by
  induction ps with
  | nil => rfl
  | cons p ps ih => obtain ⟨k, v⟩ := p; simp [specVbProps, ih]

theorem emits_strs (ls : List Nat) (f : Nat → String) :
    Emits E.vfmt (ls.map fun l => Call.str (F := F) (f l)) (ls.map fun l => R.bulk (f l)) := by
  induction ls with
  | nil => exact Emits.nil _
  | cons l ls ih => exact emits_cons E (Emits.leaf_str _ _) ih

/-- A fixed- or postponed-length array, as the `deleted` branch chooses. -/
theorem emits_either (del : Bool) {cs : List (Call F)} {ys : List R} (h : Emits E.vfmt cs ys) :
    Emits E.vfmt (if del then .arr ys.length :: cs else Call.arrP :: cs ++ [.setLen ys.length])
      [.arr ys] := by
  cases del
  · exact h.postponed
  · exact h.array

/-- Two leaves in a 2-array: `[name, value]`. -/
theorem emits_pair {cs : List (Call F)} {x y : R} (t : String)
    (h : Emits E.vfmt cs [y]) (hx : x = .bulk t) :
    Emits E.vfmt (.arr 2 :: .str t :: cs) [.arr [x, y]] := by
  subst hx
  exact (emits_cons E (Emits.leaf_str _ t) h).array

mutual
/-- **Verbose encoder = documented shape** for every value kind; the postponed label and
property arrays are closed with exactly their element counts. -/
theorem verbose_emits : ∀ v : V F, Emits E.vfmt (encVb E M v) [specVb E M v]
  | .null => Emits.leaf_null _
  | .bool _ => Emits.leaf_str _ _
  | .int _ => Emits.leaf_ll _ _
  | .float _ => Emits.leaf_str _ _
  | .str _ => Emits.leaf_str _ _
  | .datetime _ => Emits.leaf_str _ _
  | .date _ => Emits.leaf_str _ _
  | .time _ => Emits.leaf_str _ _
  | .duration _ => Emits.leaf_str _ _
  | .list _ => Emits.leaf_str _ _
  | .map _ => Emits.leaf_str _ _
  | .path _ => Emits.leaf_str _ _
  | .vec _ => Emits.leaf_str _ _
  | .point _ _ => Emits.leaf_str _ _
  | .node id del ls ps => by
    have hl := emits_either E del (emits_strs E ls M.label)
    rw [List.length_map] at hl
    have hp := emits_either E del (vprops_emits ps)
    rw [specVbProps_len] at hp
    have a1 := emits_pair E "id" (Emits.leaf_ll E.vfmt (F := F) (id : Int)) rfl
    have a2 := emits_pair E "labels" hl rfl
    have a3 := emits_pair E "properties" hp rfl
    have := ((a1.append _ a2).append _ a3).array
    simpa [encVb, specVb] using this
  | .edge id ty s d del ps => by
    have hp := emits_either E del (vprops_emits ps)
    rw [specVbProps_len] at hp
    have a1 := emits_pair E "id" (Emits.leaf_ll E.vfmt (F := F) (id : Int)) rfl
    have a2 := emits_pair E "type" (Emits.leaf_str E.vfmt (F := F) (M.rel ty)) rfl
    have a3 := emits_pair E "src_node" (Emits.leaf_ll E.vfmt (F := F) (s : Int)) rfl
    have a4 := emits_pair E "dest_node" (Emits.leaf_ll E.vfmt (F := F) (d : Int)) rfl
    have a5 := emits_pair E "properties" hp rfl
    have := ((((a1.append _ a2).append _ a3).append _ a4).append _ a5).array
    simpa [encVb, specVb] using this
theorem vprops_emits : ∀ ps : List (Nat × V F), Emits E.vfmt (encVbProps E M ps) (specVbProps E M ps)
  | [] => Emits.nil _
  | (k, v) :: ps => by
    have h := emits_pair E (M.prop k) (verbose_emits v) rfl
    have := h.append _ (vprops_emits ps)
    simpa [encVbProps, specVbProps] using this
end

/-! ## Statistics (`reply_stats`, `:530-617`) -/

structure Stats (F : Type) where
  labelsAdded : Nat
  labelsRemoved : Nat
  nodesCreated : Nat
  nodesDeleted : Nat
  propertiesSet : Nat
  propertiesRemoved : Nat
  relationshipsCreated : Nat
  relationshipsDeleted : Nat
  indexesCreated : Nat
  indexesDropped : Nat
  cached : Bool
  execTime : F

def opt (n : Nat) (label : String) : List String :=
  if n > 0 then [label ++ toString n] else []

/-- The lines in emission order (`:568-616`). -/
def statLines (st : Stats F) (version : Nat) : List String :=
  opt st.labelsAdded "Labels added: " ++ opt st.labelsRemoved "Labels removed: "
  ++ opt st.nodesCreated "Nodes created: " ++ opt st.propertiesSet "Properties set: "
  ++ opt st.propertiesRemoved "Properties removed: "
  ++ opt st.relationshipsCreated "Relationships created: "
  ++ opt st.nodesDeleted "Nodes deleted: " ++ opt st.relationshipsDeleted "Relationships deleted: "
  ++ opt st.indexesCreated "Indices created: " ++ opt st.indexesDropped "Indices deleted: "
  ++ ["Cached execution: " ++ (if st.cached then "1" else "0"),
      "Query internal execution time: " ++ M.f6 st.execTime ++ " milliseconds",
      "Graph version: " ++ toString version]

def c1 (n : Nat) : Nat := if n > 0 then 1 else 0

/-- `stats_len`, computed in the *counting* order (`:535-565`). -/
def statsLen (st : Stats F) : Nat :=
  3 + c1 st.labelsAdded + c1 st.labelsRemoved + c1 st.nodesCreated + c1 st.nodesDeleted
    + c1 st.propertiesSet + c1 st.propertiesRemoved + c1 st.relationshipsCreated
    + c1 st.relationshipsDeleted + c1 st.indexesCreated + c1 st.indexesDropped

def statCalls (st : Stats F) (version : Nat) : List (Call F) :=
  .arr (statsLen st) :: (statLines M st version).map Call.str

theorem opt_len (n : Nat) (l : String) : (opt n l).length = c1 n := by
  unfold opt c1; split <;> rfl

/-- The announced length is the number of lines sent, although the two are computed in
different orders (`Nodes deleted` is counted 4th and sent 7th). -/
theorem statsLen_eq (st : Stats F) (version : Nat) :
    statsLen st = (statLines M st version).length := by
  simp only [statsLen, statLines, List.length_append, opt_len, List.length_cons, List.length_nil]
  omega

theorem emits_strList (ls : List String) :
    Emits E.vfmt (ls.map Call.str) (ls.map R.bulk) := by
  induction ls with
  | nil => exact Emits.nil _
  | cons l ls ih => exact emits_cons E (Emits.leaf_str _ l) ih

theorem stats_emits (st : Stats F) (version : Nat) :
    Emits E.vfmt (statCalls M st version) [.arr ((statLines M st version).map R.bulk)] := by
  have h := (emits_strList E (statLines M st version)).array
  rw [List.length_map, ← statsLen_eq] at h
  exact h

/-! ## Whole replies (`reply_result`, `:619-666`) -/

def rowCalls (compact : Bool) (row : List (V F)) : List (Call F) :=
  .arr row.length :: (row.map fun v => if compact then .arr 2 :: encC E v else encVb E M v).flatten

def resultCalls (compact : Bool) (names : List String) (rows : List (List (V F)))
    (st : Stats F) (version : Nat) : List (Call F) :=
  if names = [] then .arr 1 :: statCalls M st version
  else [.arr 3, .arr names.length]
    ++ (names.map fun n => if compact then [Call.arr 2, .ll 1, .str n] else [.str n]).flatten
    ++ [.arr rows.length] ++ (rows.map (rowCalls E M compact)).flatten
    ++ statCalls M st version

theorem emits_flatten {α} (l : List α) (f : α → List (Call F)) (g : α → List R)
    (h : ∀ a ∈ l, Emits E.vfmt (f a) (g a)) :
    Emits E.vfmt (l.map f).flatten (l.map g).flatten := by
  induction l with
  | nil => exact Emits.nil _
  | cons a as ih =>
    have := (h a (by simp)).append _ (ih fun b hb => h b (by simp [hb]))
    simpa using this

theorem emits_flatten1 {α} (l : List α) (f : α → List (Call F)) (g : α → R)
    (h : ∀ a ∈ l, Emits E.vfmt (f a) [g a]) :
    Emits E.vfmt (l.map f).flatten (l.map g) := by
  induction l with
  | nil => exact Emits.nil _
  | cons a as ih =>
    have := (h a (by simp)).append _ (ih fun b hb => h b (by simp [hb]))
    simpa using this

theorem flat_single {α β} (l : List α) (f : α → β) : (l.map fun a => [f a]).flatten = l.map f := by
  induction l <;> simp_all

/-- **Compact reply**: `GRAPH.QUERY … --compact` sends exactly `Compact.encReply`, which
`Compact.decReply_enc` proves the client decodes. Rows have one cell per return name
(`batch.value_at(name.id, row)` for each name, `:653-662`). -/
theorem compact_reply (names : List String) (rows : List (List (V F))) (st : Stats F)
    (version : Nat) (hn : names ≠ []) :
    Emits E.vfmt (resultCalls E M true names rows st version)
      [encReply E names (rows.map (·.map proj)) (statLines M st version)] := by
  have hh := (emits_flatten1 E names (fun n => [Call.arr 2, .ll 1, .str n])
      (fun n => R.arr [.int 1, .bulk n])
      (fun n _ => (emits_cons E (Emits.leaf_ll _ 1) (Emits.leaf_str _ n)).array)).array
  rw [List.length_map] at hh
  have hr := (emits_flatten1 E rows (rowCalls E M true)
      (fun row => R.arr (row.map (Compact.encV E ∘ proj)))
      (fun row _ => by
        have := (emits_flatten1 E row (fun v => Call.arr 2 :: encC E v)
          (fun v => Compact.encV E (proj v))
          (fun v _ => by
            have h := (compact_emits E v).array
            rw [encPair_len2] at h; exact h)).array
        rw [List.length_map] at this
        simpa [rowCalls, Function.comp_def] using this)).array
  rw [List.length_map] at hr
  have := (((hh.append _ hr).append _ (stats_emits E M st version))).array
  simpa [resultCalls, hn, encReply, Function.comp_def, List.map_map] using this

/-- A query without `RETURN` gets `[stats]` only (`:624-629`). -/
theorem noreturn_reply (compact : Bool) (rows : List (List (V F))) (st : Stats F) (version : Nat) :
    Emits E.vfmt (resultCalls E M compact [] rows st version)
      [.arr [.arr ((statLines M st version).map R.bulk)]] := by
  have := (stats_emits E M st version).array
  simpa [resultCalls] using this

/-- **Verbose reply** shape. -/
theorem verbose_reply (names : List String) (rows : List (List (V F))) (st : Stats F)
    (version : Nat) (hn : names ≠ []) :
    Emits E.vfmt (resultCalls E M false names rows st version)
      [.arr [.arr (names.map R.bulk), .arr (rows.map fun row => .arr (row.map (specVb E M))),
             .arr ((statLines M st version).map R.bulk)]] := by
  have hh := (emits_strList E names).array
  rw [List.length_map] at hh
  have hr := (emits_flatten1 E rows (rowCalls E M false)
      (fun row => R.arr (row.map (specVb E M)))
      (fun row _ => by
        have := (emits_flatten1 E row (encVb E M) (specVb E M)
          (fun v _ => verbose_emits E M v)).array
        rw [List.length_map] at this
        simpa [rowCalls] using this)).array
  rw [List.length_map] at hr
  have := (((hh.append _ hr).append _ (stats_emits E M st version))).array
  simpa [resultCalls, hn, flat_single] using this

end RedisLayer.Resp
