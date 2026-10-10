/-!
# Remaining string / list / internal functions (string.rs, list.rs, internal.rs)

Strings are `List Char` (Rust `&str` byte operations on valid UTF-8 are in bijection with
char operations for `starts_with`/`ends_with`/`contains`/`replace`). External crates are
structures with stated laws, not axioms: `regex` (`Rx`), the string pool (`intern`),
`slice::sort_by` (`Sorter`: returns a permutation; the NaN panic is a known bug).
-/
namespace FalkorStrList.Misc

/-- Values these functions inspect. -/
inductive SV where
  | null | bool (b : Bool) | int (i : Int) | str (s : List Char) | list (vs : List SV)
  | other (name : String)

def SV.name : SV → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .str _ => "String"
  | .list _ => "List" | .other n => n

inductive Res where
  | ok (v : SV) | err (s : String) | unreachable

/-! ## internal.rs -/

/-- `ends_with` / `contains` (internal.rs:57, :75): `Null` unless both are strings. -/
def internalEndsWith : List SV → Res
  | .str s :: .str t :: _ => .ok (.bool (t.isSuffixOf s))
  | _ => .ok .null

def internalContains : List SV → Res
  | .str s :: .str t :: _ => .ok (.bool (decide (t <:+: s)))
  | _ => .ok .null

theorem internalEndsWith_spec (s t : List Char) :
    internalEndsWith [.str s, .str t] = .ok (.bool true) ↔ ∃ p, s = p ++ t := by
  simp only [internalEndsWith, Res.ok.injEq, SV.bool.injEq]
  rw [List.isSuffixOf_iff_suffix]
  exact ⟨fun ⟨p, h⟩ => ⟨p, h.symm⟩, fun ⟨p, h⟩ => ⟨p, h.symm⟩⟩

theorem internalContains_spec (s t : List Char) :
    internalContains [.str s, .str t] = .ok (.bool true) ↔ ∃ p q, s = p ++ t ++ q := by
  simp only [internalContains, Res.ok.injEq, SV.bool.injEq, decide_eq_true_eq]
  exact ⟨fun ⟨p, q, h⟩ => ⟨p, q, h.symm⟩, fun ⟨p, q, h⟩ => ⟨p, q, h.symm⟩⟩

theorem internal_null (a b : SV) (h : ∀ s t, ¬ (a = .str s ∧ b = .str t)) :
    internalEndsWith [a, b] = .ok .null ∧ internalContains [a, b] = .ok .null := by
  cases a <;> cases b <;> simp_all [internalEndsWith, internalContains]

def SV.isNull : SV → Bool
  | .null => true
  | _ => false

/-- `is_null` (internal.rs:90): `x IS NULL` is `is_null(false, x)`, `IS NOT NULL` is
`is_null(true, x)`. -/
def internalIsNull : List SV → Res
  | .bool isNot :: .null :: _ => .ok (.bool (!isNot))
  | .bool isNot :: _ :: _ => .ok (.bool isNot)
  | _ => .unreachable

theorem internalIsNull_spec (isNot : Bool) (v : SV) :
    internalIsNull [.bool isNot, v] =
      .ok (.bool (if isNot then !v.isNull else v.isNull)) := by
  cases v <;> cases isNot <;> rfl

/-- `case` / `add` (internal.rs:130, :145): enumeration-only stubs that always fail. -/
def internalCase (_ : List SV) : Res := .err "Internal function 'case' should not be called directly"
def internalAdd (_ : List SV) : Res := .err "Internal function 'add' should not be called directly"

theorem internal_stubs (a b : List SV) :
    internalCase a = internalCase b ∧ internalAdd a = internalAdd b ∧
    (∀ v, internalCase a ≠ .ok v) ∧ (∀ v, internalAdd a ≠ .ok v) := by
  simp [internalCase, internalAdd]

/-- `register` (internal.rs:31): names, all flagged `internal`. -/
def internalRegistered : List String :=
  ["starts_with", "ends_with", "contains", "is_null", "regex_matches", "case", "add"]

theorem internalRegistered_nodup : internalRegistered.Nodup := by decide

/-! ## string.rs -/

/-- `string_pool::global().intern(s)`: returns an `Arc` with the same content. -/
structure Pool where
  intern : List Char → List Char
  intern_eq : ∀ s, intern s = s

/-- `intern` (string.rs:40). -/
def intern (P : Pool) : List SV → Res
  | .str s :: _ => .ok (.str (P.intern s))
  | .null :: _ => .ok .null
  | _ => .unreachable

theorem intern_id (P : Pool) (v : SV) (h : v = .null ∨ ∃ s, v = .str s) : intern P [v] = .ok v := by
  rcases h with rfl | ⟨s, rfl⟩
  · rfl
  · simp [intern, P.intern_eq]

/-- `str::replace` (string.rs:197) for a non-empty pattern: left-to-right, non-overlapping. -/
def replaceNE (pat rep : List Char) (hp : pat ≠ []) : List Char → List Char
  | [] => []
  | c :: cs =>
    if pat.isPrefixOf (c :: cs) then rep ++ replaceNE pat rep hp ((c :: cs).drop pat.length)
    else c :: replaceNE pat rep hp cs
termination_by s => s.length
decreasing_by
  · have : pat.length ≥ 1 := by cases pat with | nil => exact absurd rfl hp | cons _ _ => simp
    simp; omega
  · simp

/-- Empty pattern: `rep` before every char and at the end (Rust `"ab".replace("", "-")`
is `"-a-b-"`). -/
def replaceEmpty (rep : List Char) : List Char → List Char
  | [] => rep
  | c :: cs => rep ++ c :: replaceEmpty rep cs

def strReplace (s pat rep : List Char) : List Char :=
  if h : pat = [] then replaceEmpty rep s else replaceNE pat rep h s

/-- Replacing a pattern by itself is the identity. -/
theorem replaceNE_self (pat : List Char) (hp : pat ≠ []) : ∀ s, replaceNE pat pat hp s = s := by
  intro s
  induction h : s.length using Nat.strongRecOn generalizing s with
  | ind n ih =>
    cases s with
    | nil => simp [replaceNE]
    | cons c cs =>
      rw [replaceNE]
      split
      · rename_i hpre
        obtain ⟨t, ht⟩ := (List.isPrefixOf_iff_prefix.mp hpre)
        have hl : pat.length ≥ 1 := by cases pat with | nil => exact absurd rfl hp | cons _ _ => simp
        have hd : (c :: cs).drop pat.length = t := by rw [← ht]; simp
        rw [hd, ih t.length (by rw [← h, ← ht]; simp; omega) t rfl, ht]
      · rw [ih cs.length (by simp [← h]) cs rfl]

theorem strReplace_self (s pat : List Char) : strReplace s pat pat = s := by
  unfold strReplace; split
  · subst_vars; induction s with
    | nil => rfl
    | cons c cs ih => simp [replaceEmpty, ih]
  · exact replaceNE_self _ _ s

/-- `replace` (string.rs:190): strings only, any `null` argument gives `null`. -/
def stringReplace : List SV → Res
  | [.str s, .str p, .str r] => .ok (.str (strReplace s p r))
  | [.null, _, _] | [_, .null, _] | [_, _, .null] => .ok .null
  | _ => .unreachable

theorem stringReplace_null (a b : SV) :
    stringReplace [.null, a, b] = .ok .null ∧ stringReplace [a, .null, b] = .ok .null ∧
    stringReplace [a, b, .null] = .ok .null := by
  refine ⟨rfl, ?_, ?_⟩ <;> cases a <;> cases b <;> rfl

/-- `to_string_vec` (string.rs:309), the `string.join` argument check. -/
def toStringVec : List SV → Except String (List (List Char))
  | [] => .ok []
  | .str s :: vs => (toStringVec vs).map (s :: ·)
  | v :: _ => .error s!"Type mismatch: expected String but was {v.name}"

theorem toStringVec_strs (ss : List (List Char)) : toStringVec (ss.map .str) = .ok ss := by
  induction ss with
  | nil => rfl
  | cons s ss ih => simp [toStringVec, ih, Except.map]

theorem toStringVec_err (pre : List (List Char)) (v : SV) (rest : List SV)
    (hv : ∀ s, v ≠ .str s) :
    toStringVec (pre.map .str ++ v :: rest) = .error s!"Type mismatch: expected String but was {v.name}" := by
  induction pre with
  | nil => cases v <;> simp_all [toStringVec]
  | cons s ss ih => simp [toStringVec, ih, Except.map]

/-- The `regex` crate. A match is a list of capture slots (slot 0 = whole match). -/
structure Rx (R : Type) where
  compile : List Char → Except String R
  captures : R → List Char → List (List (Option (List Char)))
  replaceAll : R → List Char → List Char → List Char

variable {R : Type} (X : Rx R)

/-- `regex_captures_list` (string.rs:490). -/
def capturesList (re : R) (text : List Char) : SV :=
  .list ((X.captures re text).map fun caps =>
    .list (caps.map fun c => match c with | some m => .str m | none => .null))

/-- One sub-list per match, one entry per capture slot, `null` exactly for the
non-participating groups. -/
theorem capturesList_shape (re : R) (text : List Char) :
    capturesList X re text = .list ((X.captures re text).map fun caps =>
      .list (caps.map fun c => c.elim .null .str)) := by
  simp only [capturesList]; congr; funext caps; congr; funext c; cases c <;> rfl

/-- `string.matchRegEx` (string.rs:417). -/
def matchRegEx : List SV → Res
  | [.str t, .str p] => match X.compile p with
    | .ok re => .ok (capturesList X re t)
    | .error e => .err s!"Invalid regex, {e}"
  | [.null, _] | [_, .null] => .ok (.list [])
  | _ => .unreachable

theorem matchRegEx_null (v : SV) : matchRegEx X [.null, v] = .ok (.list []) ∧
    matchRegEx X [v, .null] = .ok (.list []) := by
  constructor <;> cases v <;> rfl

/-- `string.replaceRegEx` (string.rs:440). -/
def replaceRegEx (args : List SV) : Res :=
  match args with
  | .null :: _ :: _ => .ok .null
  | _ :: .null :: _ => .ok .null
  | .str t :: .str p :: rest =>
    match X.compile p with
    | .error e => .err s!"Invalid regex, {e}"
    | .ok re => match rest with
      | [] => .ok (.str (X.replaceAll re t []))
      | .null :: _ => .ok .null
      | .str r :: _ => .ok (.str (X.replaceAll re t r))
      | v :: _ => .err s!"Type mismatch: expected String or Null but was {v.name}"
  | _ => .unreachable

/-- An omitted replacement is the empty string. -/
theorem replaceRegEx_default (t p : List Char) :
    replaceRegEx X [.str t, .str p] = replaceRegEx X [.str t, .str p, .str []] := by
  simp only [replaceRegEx]

theorem replaceRegEx_null (t p : List Char) (re : R) (h : X.compile p = .ok re) :
    replaceRegEx X [.str t, .str p, .null] = .ok .null := by simp [replaceRegEx, h]

/-- `register` (string.rs:35). -/
def stringRegistered : List String :=
  ["intern", "substring", "split", "tolower", "toupper", "replace", "left", "ltrim", "rtrim", "trim",
   "right", "string.join", "string.matchRegEx", "string.replaceRegEx"]

theorem stringRegistered_nodup : stringRegistered.Nodup := by decide

/-! ## list.rs -/

/-- `slice::sort_by`: we rely only on "returns a permutation". -/
structure Sorter where
  sortBy : (SV → SV → Ordering) → List SV → List SV
  perm : ∀ c l, (sortBy c l).Perm l

/-- `list.sort` (list.rs:200). -/
def listSort (S : Sorter) (cmp : SV → SV → Ordering) : List SV → Res
  | .null :: _ => .ok .null
  | .list vs :: rest =>
    let asc : Option Bool := match rest with
      | [] => some true
      | .bool b :: _ => some b
      | _ => none
    match asc with
    | none => .ok .null
    | some a => let s := S.sortBy cmp vs; .ok (.list (if a then s else s.reverse))
  | _ => .unreachable

theorem listSort_perm (S : Sorter) (cmp : SV → SV → Ordering) (vs : List SV) (b : Bool) :
    ∃ out, listSort S cmp [.list vs, .bool b] = .ok (.list out) ∧ out.Perm vs ∧
      (b = false → out = (S.sortBy cmp vs).reverse) := by
  refine ⟨if b then S.sortBy cmp vs else (S.sortBy cmp vs).reverse, rfl, ?_, ?_⟩
  · split
    · exact S.perm _ _
    · exact (List.reverse_perm _).trans (S.perm _ _)
  · intro h; simp [h]

theorem listSort_default (S : Sorter) (cmp : SV → SV → Ordering) (vs : List SV) :
    listSort S cmp [.list vs] = listSort S cmp [.list vs, .bool true] := rfl

theorem listSort_badAsc (S : Sorter) (cmp : SV → SV → Ordering) (vs : List SV) (i : Int) :
    listSort S cmp [.list vs, .int i] = .ok .null := rfl

/-- `list.dedup` (list.rs:341): `seen.contains(v)` is `PartialEq` (`compare_value == Equal`). -/
def dedupLoop (eq : SV → SV → Bool) : List SV → List SV → List SV
  | [], seen => seen
  | v :: vs, seen => dedupLoop eq vs (if seen.any (fun s => eq s v) then seen else seen ++ [v])

def listDedup (eq : SV → SV → Bool) : List SV → Res
  | .null :: _ => .ok .null
  | .list vs :: _ => .ok (.list (dedupLoop eq vs []))
  | _ => .unreachable

theorem dedupLoop_sublist (eq : SV → SV → Bool) :
    ∀ (vs seen : List SV), ∃ ks, dedupLoop eq vs seen = seen ++ ks ∧ ks.Sublist vs := by
  intro vs
  induction vs with
  | nil => intro seen; exact ⟨[], by simp [dedupLoop], List.Sublist.slnil⟩
  | cons v vs ih =>
    intro seen
    simp only [dedupLoop]
    split
    · obtain ⟨ks, h1, h2⟩ := ih seen; exact ⟨ks, h1, h2.cons v⟩
    · obtain ⟨ks, h1, h2⟩ := ih (seen ++ [v])
      exact ⟨v :: ks, by simp [h1], h2.cons_cons v⟩

/-- Output elements are pairwise "not equal" (earlier vs later), given the seed is. -/
theorem dedupLoop_pairwise (eq : SV → SV → Bool) :
    ∀ (vs seen : List SV), seen.Pairwise (fun a b => eq a b = false) →
      (dedupLoop eq vs seen).Pairwise (fun a b => eq a b = false) := by
  intro vs
  induction vs with
  | nil => intro seen h; simpa [dedupLoop] using h
  | cons v vs ih =>
    intro seen h
    simp only [dedupLoop]
    split
    · exact ih seen h
    · rename_i hn
      apply ih
      rw [List.pairwise_append]
      refine ⟨h, by simp, ?_⟩
      intro a ha b hb
      simp at hb; subst hb
      simp only [List.any_eq_true, not_exists, not_and] at hn
      simpa using hn a ha

/-- `list.dedup` keeps a subsequence of the input with no `=`-equal pair. (A NaN is never
`=` to anything, so every NaN is kept — matching `=` semantics.) -/
theorem listDedup_spec (eq : SV → SV → Bool) (vs : List SV) :
    ∃ out, listDedup eq [.list vs] = .ok (.list out) ∧ out.Sublist vs ∧
      out.Pairwise (fun a b => eq a b = false) := by
  obtain ⟨ks, h1, h2⟩ := dedupLoop_sublist eq vs []
  refine ⟨dedupLoop eq vs [], rfl, by simpa [h1] using h2, dedupLoop_pairwise eq vs [] (by simp)⟩

/-- `register` (list.rs:56). -/
def listRegistered : List String :=
  ["size", "head", "last", "tail", "reverse", "list.remove", "list.sort", "list.insert",
   "list.insertListElements", "list.dedup"]

theorem listRegistered_nodup : listRegistered.Nodup := by decide

end FalkorStrList.Misc
