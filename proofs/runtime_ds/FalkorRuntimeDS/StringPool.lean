/-
# `StringPool` (`graph/src/runtime/string_pool.rs`): interning is injective and stable

A pooled `ArcStr` is a pointer `Nat` into an allocation counter `next`
(every `ArcStr::from` of a non-empty string is a fresh allocation). The pool is the list of its
`(content, ptr)` entries; `ext p` is the number of strong refs held *outside*
the pool (`Arc::strong_count - 1`). All operations run under the `Mutex`, so
they are modelled sequentially.

| here | there |
| --- | --- |
| `Pool`        | `StringPool { inner: Mutex<HashSet<ArcStr>> }` (string_pool.rs) |
| `internArc`   | `StringPool::intern` (string_pool.rs) — a static `ArcStr` (since #3966) is returned unpooled, else `intern` |
| `intern`      | `StringPool::intern` past its `is_static` return — hit: clone existing; miss: `ArcStr::from(a.as_str())` |
| `isInternedArc` / `isInterned` | `StringPool::is_interned` (string_pool.rs) — lookup by content, then `ArcStr::ptr_eq` |
| `prune`       | the `retain(|k| ArcStr::strong_count(k).is_some_and(|n| n > 1))` in `StringPool::stats` (string_pool.rs) |
| `release`     | dropping an outside `ArcStr` |

The only static `ArcStr` a query can hand `intern` is the empty string
(`ArcStr::from("")` is arcstr's static `ArcStr::new()`; nothing builds a `literal!`
value), so `internArc` takes "static" to mean `s = ""` and returns the handle `none`.
-/
namespace FalkorRuntimeDS.StringPoolModel

structure Pool where
  entries : List (String × Nat)
  next : Nat
  ext : Nat → Nat

def lookup (es : List (String × Nat)) (s : String) : Option Nat :=
  (es.find? (fun e => e.1 == s)).map Prod.snd

def bump (f : Nat → Nat) (p : Nat) : Nat → Nat := fun q => if q = p then f q + 1 else f q

/-- string_pool.rs — returns the canonical pointer; the caller now holds one ref. -/
def intern (P : Pool) (s : String) : Pool × Nat :=
  match lookup P.entries s with
  | some p => ({ P with ext := bump P.ext p }, p)
  | none =>
    let p := P.next
    ({ entries := P.entries ++ [(s, p)], next := p + 1, ext := bump P.ext p }, p)

/-- string_pool.rs -/
def isInterned (P : Pool) (p : Nat) (s : String) : Bool := lookup P.entries s == some p

/-- string_pool.rs -/
def prune (P : Pool) : Pool := { P with entries := P.entries.filter (fun e => decide (P.ext e.2 > 0)) }

def release (P : Pool) (p : Nat) : Pool := { P with ext := fun q => if q = p then P.ext q - 1 else P.ext q }

/-- Invariant: one entry per content, one content per pointer, all pointers already allocated. -/
structure WF (P : Pool) : Prop where
  keys : (P.entries.map Prod.fst).Nodup
  ptrs : (P.entries.map Prod.snd).Nodup
  alloc : ∀ e ∈ P.entries, e.2 < P.next

theorem wf_empty : WF ⟨[], 0, fun _ => 0⟩ := ⟨by simp, by simp, by simp⟩

theorem lookup_append (es : List (String × Nat)) (s t : String) (p : Nat) :
    lookup (es ++ [(t, p)]) s = match lookup es s with
      | some q => some q
      | none => if t = s then some p else none := by
  unfold lookup
  rw [List.find?_append]
  cases h : es.find? (fun e => e.1 == s) with
  | some e => simp
  | none => simp [List.find?_cons]; split <;> simp_all

theorem lookup_mem {es : List (String × Nat)} {s : String} {p : Nat} (h : lookup es s = some p) :
    (s, p) ∈ es := by
  unfold lookup at h
  cases hf : es.find? (fun e => e.1 == s) with
  | none => simp [hf] at h
  | some e =>
    simp [hf] at h
    have hm := List.mem_of_find?_eq_some hf
    have hk := List.find?_some hf
    simp at hk
    obtain ⟨a, b⟩ := e
    simp at hk h; subst hk h; exact hm

theorem lookup_of_mem {es : List (String × Nat)} (hk : (es.map Prod.fst).Nodup) {s : String} {p : Nat}
    (h : (s, p) ∈ es) : lookup es s = some p := by
  induction es with
  | nil => simp at h
  | cons e t ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hk
    unfold lookup; simp only [List.find?_cons]
    rcases List.mem_cons.mp h with h | h
    · subst h; simp
    · have : e.1 ≠ s := fun he => hk.1 ⟨(s, p), h, he.symm⟩
      have hb : (e.1 == s) = false := by simpa using this
      simp only [hb]
      exact ih hk.2 h

theorem lookup_none {es : List (String × Nat)} {s : String} (h : lookup es s = none) :
    ∀ e ∈ es, e.1 ≠ s := by
  intro e he hs
  unfold lookup at h
  simp only [Option.map_eq_none_iff, List.find?_eq_none] at h
  exact absurd (h e he) (by simp [hs])

theorem wf_intern {P : Pool} (hP : WF P) (s : String) : WF (intern P s).1 := by
  unfold intern
  split
  · exact ⟨hP.keys, hP.ptrs, hP.alloc⟩
  · rename_i hl
    refine ⟨?_, ?_, ?_⟩
    · simp only [List.map_append, List.map_cons, List.map_nil]
      rw [List.nodup_append]
      refine ⟨hP.keys, by simp, ?_⟩
      intro a ha b hb; simp at hb; subst hb
      simp only [List.mem_map] at ha
      obtain ⟨e, he, rfl⟩ := ha
      exact lookup_none hl e he
    · simp only [List.map_append, List.map_cons, List.map_nil]
      rw [List.nodup_append]
      refine ⟨hP.ptrs, by simp, ?_⟩
      intro a ha b hb; simp at hb; subst hb
      simp only [List.mem_map] at ha
      obtain ⟨e, he, rfl⟩ := ha
      exact Nat.ne_of_lt (hP.alloc e he)
    · intro e he
      rcases List.mem_append.mp he with he | he
      · exact Nat.lt_succ_of_lt (hP.alloc e he)
      · simp at he; subst he; simp

theorem wf_prune {P : Pool} (hP : WF P) : WF (prune P) :=
  ⟨List.Pairwise.sublist ((List.filter_sublist).map _) hP.keys,
   List.Pairwise.sublist ((List.filter_sublist).map _) hP.ptrs,
   fun e he => hP.alloc e ((List.mem_filter.mp he).1)⟩

theorem wf_release {P : Pool} (hP : WF P) (p : Nat) : WF (release P p) := ⟨hP.keys, hP.ptrs, hP.alloc⟩

theorem ptr_unique : ∀ {es : List (String × Nat)}, (es.map Prod.snd).Nodup →
    ∀ {a b : String} {p : Nat}, (a, p) ∈ es → (b, p) ∈ es → a = b
  | [], _, _, _, _, ha, _ => by simp at ha
  | e :: t, hn, a, b, p, ha, hb => by
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    rcases List.mem_cons.mp ha with ha | ha <;> rcases List.mem_cons.mp hb with hb | hb
    · rw [← ha] at hb; exact (Prod.mk.inj hb).1.symm
    · exact absurd ⟨(b, p), hb, by rw [← ha]⟩ hn.1
    · exact absurd ⟨(a, p), ha, by rw [← hb]⟩ hn.1
    · exact ptr_unique hn.2 ha hb

/-- `intern(s)` returns the canonical pointer for `s`. -/
theorem intern_isInterned {P : Pool} (hP : WF P) (s : String) :
    isInterned (intern P s).1 (intern P s).2 s = true := by
  unfold isInterned intern
  split
  · rename_i p hp; simp [hp]
  · rename_i hl; simp [lookup_append, hl]

/-- Interning the same content twice (no pruning in between) returns the same pointer. -/
theorem intern_idem {P : Pool} (s : String) :
    (intern (intern P s).1 s).2 = (intern P s).2 := by
  cases h : lookup P.entries s with
  | some p => simp [intern, h]
  | none => simp [intern, h, lookup_append]

/-- Injective: two contents share a pointer only if they are equal. -/
theorem intern_injective {P : Pool} (hP : WF P) (s t : String) :
    let r1 := intern P s
    let r2 := intern r1.1 t
    r2.2 = r1.2 → t = s := by
  intro r1 r2 heq
  have hw1 := wf_intern hP s
  have h1 : (s, r1.2) ∈ r1.1.entries := lookup_mem (by
    have := intern_isInterned hP s; unfold isInterned at this; simpa using this)
  have h2 : (t, r2.2) ∈ r2.1.entries := lookup_mem (by
    have := intern_isInterned hw1 t; unfold isInterned at this; simpa using this)
  have hsub : ∀ e ∈ r1.1.entries, e ∈ r2.1.entries := by
    intro e he; show e ∈ (intern r1.1 t).1.entries
    unfold intern; split
    · exact he
    · exact List.mem_append_left _ he
  have hw2 := wf_intern hw1 t
  rw [heq] at h2
  exact ptr_unique hw2.ptrs h2 (hsub _ h1)

/-- Stable: pruning never evicts an entry someone still holds, so a held pointer
stays canonical. -/
theorem prune_keeps_held {P : Pool} (s : String) (p : Nat) (h : isInterned P p s = true)
    (hheld : P.ext p > 0) (hP : WF P) : isInterned (prune P) p s = true := by
  unfold isInterned at *
  simp at h
  have hm := lookup_mem h
  have : (s, p) ∈ (prune P).entries := List.mem_filter.mpr ⟨hm, by simpa using hheld⟩
  simp [lookup_of_mem (wf_prune hP).keys this]

/-- No ABA: a string re-interned after its entry was pruned gets a pointer that
was never handed out before. -/
theorem reintern_fresh {P : Pool} (hP : WF P) (s : String) (h : lookup P.entries s = none) :
    (intern P s).2 = P.next ∧ ∀ e ∈ P.entries, e.2 ≠ (intern P s).2 := by
  unfold intern; rw [h]
  exact ⟨rfl, fun e he => Nat.ne_of_lt (hP.alloc e he)⟩

/-! ## The whole `intern` (#3966): the static empty string is never pooled -/

/-- string_pool.rs `StringPool::intern`: `none` is the static `ArcStr` handed back
as is (`ArcStr::is_static`), `some p` a pool pointer. -/
def internArc (P : Pool) (s : String) : Pool × Option Nat :=
  if s = "" then (P, none) else let r := intern P s; (r.1, some r.2)

/-- string_pool.rs `StringPool::is_interned`: a static is never `ptr_eq` to a pool
entry (every entry is a heap `ArcStr::from`). -/
def isInternedArc (P : Pool) : Option Nat → String → Bool
  | none, _ => false
  | some p, s => isInterned P p s

/-- Counterexample to `intern_isInterned` for the whole fn since #3966: interning the
empty string does not return a pool entry, so `is_interned` is false for it (and
`GRAPH.INFO ObjectPool` no longer counts it). -/
theorem internArc_empty_not_interned (P : Pool) :
    isInternedArc (internArc P "").1 (internArc P "").2 "" = false := by
  simp [internArc, isInternedArc]

/-- `intern_isInterned`'s statement, on the whole fn, is refuted (`s = ""`). -/
theorem internArc_isInterned_fails :
    ¬ ∀ (P : Pool), WF P → ∀ s, isInternedArc (internArc P s).1 (internArc P s).2 s = true := by
  intro h
  have := h _ wf_empty ""
  rw [internArc_empty_not_interned] at this
  exact Bool.false_ne_true this

/-- Every non-empty string interns to its canonical pool entry, as before. -/
theorem internArc_isInterned {P : Pool} (hP : WF P) {s : String} (hs : s ≠ "") :
    isInternedArc (internArc P s).1 (internArc P s).2 s = true := by
  simp only [internArc, hs, if_false, isInternedArc]
  exact intern_isInterned hP s

theorem wf_internArc {P : Pool} (hP : WF P) (s : String) : WF (internArc P s).1 := by
  unfold internArc; split
  · exact hP
  · exact wf_intern hP s

/-- Interning the same content twice returns the same handle. -/
theorem internArc_idem (P : Pool) (s : String) :
    (internArc (internArc P s).1 s).2 = (internArc P s).2 := by
  unfold internArc; split
  · simp_all
  · simp only [*, if_false]; rw [intern_idem]

/-- Injective: two contents get the same handle only if they are equal. -/
theorem internArc_injective {P : Pool} (hP : WF P) (s t : String) :
    let r1 := internArc P s
    let r2 := internArc r1.1 t
    r2.2 = r1.2 → t = s := by
  intro r1 r2 heq
  simp only [r1, r2, internArc] at heq
  by_cases hs : s = "" <;> by_cases ht : t = ""
  · rw [hs, ht]
  · simp [hs, ht] at heq
  · simp [hs, ht] at heq
  · simp only [hs, ht, if_false, Option.some.injEq] at heq
    exact intern_injective hP s t heq

/-- The pool never holds the empty string: `intern` keeps it out, `prune` only drops. -/
def NoEmpty (P : Pool) : Prop := ∀ e ∈ P.entries, e.1 ≠ ""

theorem noEmpty_internArc {P : Pool} (h : NoEmpty P) (s : String) : NoEmpty (internArc P s).1 := by
  unfold internArc; split
  · exact h
  · rename_i hs
    intro e he
    simp only [intern] at he
    split at he
    · exact h e he
    · rcases List.mem_append.mp he with he | he
      · exact h e he
      · simp at he; subst he; exact hs

theorem noEmpty_prune {P : Pool} (h : NoEmpty P) : NoEmpty (prune P) :=
  fun e he => h e (List.mem_filter.mp he).1

end FalkorRuntimeDS.StringPoolModel
