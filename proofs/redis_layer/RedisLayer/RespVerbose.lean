import RedisLayer.RespCompact
/-!
# Verbose values, statistics and whole replies (`src/reply.rs`)

| here | there |
| --- | --- |
| `wf`                          | `write_f` `:36-45` (#3043) |
| `fmtS` / `fmtList` / `fmtMap` | `format_value_to_string` `:49-111` |
| `encVb`, `specVb`, `pointStr` | `reply_verbose_value` `:357-541` (top-level point `:531-540`, #3043) |
| `statCalls`, `statLines`      | `reply_stats` `:543-630` |
| `resultCalls`                 | `reply_result::<COMPACT>` `:632-679`, `reply_verbose` `:681`, `reply_compact` `:689` |
| `Fmt.g15`                     | `format_g(x, 15)` `:123-142` (libc `snprintf`, AXIOMATISED as an abstract function) |

Number/temporal formatting is abstract (`Fmt`): `{:.6}`, `%.15g`, `Value::format_*`.

**#3043 (`7cde92c1d`, closes #3042).** C prints NaN with `%f` as `nan`; Rust's `{:.6}` spells
it `NaN`, so a NaN inside a list or vector printed `[NaN]` (C `[nan]`). `write_f` now prints
`nan`. C's top-level verbose point is `point({latitude:%f, longitude:%f})`
(`resultset_replyverbose.c:239`), without the space after the colons that the nested
`SIValue_ToString` form has (`value.c:504`); the top-level form now matches. Against C's
`%f` (`CPf`, a hypothesis about libc): `wf_c`, `fmtS_floats_c`, `fmtS_vec_c`, `pointStr_c`.
History: `pre3043_point_space`, `pre3043_nan_in_list`.
-/
namespace RedisLayer.Resp
open RedisLayer.Compact (R Val Env encPair encReply)

variable {F : Type}

structure Fmt (F : Type) where
  f6 : F → String            -- `{:.6}`
  nan : F → Bool             -- `f64::is_nan`
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

/-- `write_f` (`:36-45`, #3043): C's `%f` spelling of NaN, `{:.6}` otherwise. -/
def wf (x : F) : String := if M.nan x then "nan" else M.f6 x

mutual
/-- `format_value_to_string` (`:49-111`). -/
def fmtS : V F → String
  | .null => "NULL"
  | .bool b => if b then "true" else "false"
  | .int i => toString i
  | .float x => wf M x
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
  | .vec xs => "<" ++ joinSep (xs.map (wf M)) ++ ">"
  | .point a o => "point({latitude: " ++ M.f6 a ++ ", longitude: " ++ M.f6 o ++ "})"
def fmtList : List (V F) → List String
  | [] => []
  | v :: vs => fmtS v :: fmtList vs
def fmtMap : List (String × V F) → List String
  | [] => []
  | (k, v) :: kvs => (k ++ ": " ++ fmtS v) :: fmtMap kvs
end

/-- The top-level verbose point (`:531-540`, #3043): no space after the colons. -/
def pointStr (a o : F) : String :=
  "point({latitude:" ++ M.f6 a ++ ", longitude:" ++ M.f6 o ++ "})"

mutual
/-- `reply_verbose_value` (`:357-541`). -/
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

/-! ## Statistics (`reply_stats`, `:543-630`) -/

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

/-- The lines in emission order (`:581-629`). -/
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

/-- `stats_len`, computed in the *counting* order (`:548-578`). -/
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

/-! ## Whole replies (`reply_result`, `:632-679`) -/

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
(`batch.value_at(name.id, row)` for each name, `:666-675`). -/
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

/-- A query without `RETURN` gets `[stats]` only (`:637-642`). -/
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

/-! ## Agreement with C's `%f` (#3043) -/

/-- C's `%f` (glibc / Apple libc `printf`), as a hypothesis: it agrees with Rust's `{:.6}` on
every non-NaN value (finite values and `inf`/`-inf`), and prints NaN as `nan`. The second
field holds for a NaN with the sign bit clear; glibc prints `-nan` for one with it set. -/
structure CPf (M : Fmt F) where
  cf : F → String
  f6_agree : ∀ x, M.nan x = false → cf x = M.f6 x
  nan_text : ∀ x, M.nan x = true → cf x = "nan"

/-- C's top-level verbose point, `_ResultSet_VerboseReplyWithPoint`
(`resultset_replyverbose.c:239`). -/
def cPointStr (cf : F → String) (a o : F) : String :=
  "point({latitude:" ++ cf a ++ ", longitude:" ++ cf o ++ "})"

theorem wf_c (H : CPf M) (x : F) : wf M x = H.cf x := by
  unfold wf
  cases h : M.nan x
  · simp [H.f6_agree x h]
  · simp [H.nan_text x h]

theorem fmtS_float_c (H : CPf M) (x : F) : fmtS M (.float x) = H.cf x := by
  simp [fmtS, wf_c M H]

theorem fmtList_floats_c (H : CPf M) (xs : List F) :
    fmtList M (xs.map V.float) = xs.map H.cf := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [fmtList, fmtS_float_c M H, ih]

/-- **A list of floats prints as C prints it**, NaN included (`RETURN [0.0/0.0]` → `[nan]`). -/
theorem fmtS_floats_c (H : CPf M) (xs : List F) :
    fmtS M (.list (xs.map V.float)) = "[" ++ joinSep (xs.map H.cf) ++ "]" := by
  simp [fmtS, fmtList_floats_c M H]

/-- **A vector prints as C prints it**, NaN included. -/
theorem fmtS_vec_c (H : CPf M) (xs : List F) :
    fmtS M (.vec xs) = "<" ++ joinSep (xs.map H.cf) ++ ">" := by
  have : xs.map (wf M) = xs.map H.cf := List.map_congr_left (fun x _ => wf_c M H x)
  simp [fmtS, this]

/-- **The top-level verbose point is C's text.** Point coordinates are finite
(`Point::validate`, `value.rs:177-190`), hence not NaN. -/
theorem pointStr_c (H : CPf M) (a o : F) (ha : M.nan a = false) (ho : M.nan o = false) :
    pointStr M a o = cPointStr H.cf a o := by
  simp [pointStr, cPointStr, H.f6_agree a ha, H.f6_agree o ho]

/-- The verbose reply of a top-level point is the bulk string C sends. -/
theorem verbose_point_c (E : Env F) (H : CPf M) (a o : F) (ha : M.nan a = false)
    (ho : M.nan o = false) : specVb E M (.point a o) = .bulk (cPointStr H.cf a o) := by
  simp [specVb, pointStr_c M H a o ha ho]

/-! ### History: before #3043 (`289c201fe`) -/

/-- The top-level point before #3043 (`reply.rs:522-525` then): a space after each colon. -/
def pre3043_pointStr (a o : F) : String :=
  "point({latitude: " ++ M.f6 a ++ ", longitude: " ++ M.f6 o ++ "})"

/-- The float leaf of `format_value_to_string` before #3043: plain `{:.6}`. -/
def pre3043_fmtFloat (x : F) : String := M.f6 x

/-- **Divergence before #3043 (confirmed live, `repro_live.py` V1)**: every top-level point
differed from C's text (two characters longer). -/
theorem pre3043_point_space (H : CPf M) (a o : F) (ha : M.nan a = false)
    (ho : M.nan o = false) : pre3043_pointStr M a o ≠ cPointStr H.cf a o := by
  intro h
  have := congrArg String.length h
  have l1 : "point({latitude: ".length = "point({latitude:".length + 1 := by decide
  have l2 : ", longitude: ".length = ", longitude:".length + 1 := by decide
  simp only [pre3043_pointStr, cPointStr, H.f6_agree a ha, H.f6_agree o ho,
    String.length_append, l1, l2] at this
  omega

/-- **Divergence before #3043 (confirmed live, `repro_live.py` V2)**: Rust's `{:.6}` spells
NaN `NaN`, so `RETURN [0.0/0.0]` printed `[NaN]`; C `[nan]`. -/
theorem pre3043_nan_in_list (H : CPf M) (x : F) (hn : M.nan x = true) (hR : M.f6 x = "NaN") :
    pre3043_fmtFloat M x ≠ H.cf x := by
  simp [pre3043_fmtFloat, hR, H.nan_text x hn]

/-- And on main the same NaN prints as C does. -/
theorem fixed3043_nan_in_list (x : F) (hn : M.nan x = true) :
    fmtS M (.list [.float x]) = "[nan]" := by
  simp [fmtS, fmtList, joinSep, wf, hn]

end RedisLayer.Resp
