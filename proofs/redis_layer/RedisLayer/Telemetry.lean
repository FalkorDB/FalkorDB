/-!
# `src/telemetry.rs` — stream naming, the query registry, batching and per-graph grouping

Line numbers: `origin/main` (`3fec7d7c9`).

| here | there |
| --- | --- |
| `truncate`             | `truncate` `:34`, `truncate_arc` `:50` |
| `upToNul`, `streamName`| `graph_core::up_to_nul` `:87`, `stream_name` `:66` |
| `hashTag`              | Redis Cluster `keyHashSlot` tag rule (documentation of the key scheme) |
| `Reg`, `register*` … `snapshot*` | `shard_for` `:404`, `register_running` `:409` … `snapshot_waiting` `:534` |
| `WE.*`                 | `WaitingEntry::register/promote/drop` `:481-502` |
| `enqueueOk`            | `enqueue_entry` gates `:892-937` |
| `drain`                | `drain_queued` `:832` |
| `capDeferred`          | `deferred.drain(..len - DEFERRED_XADD_MAX)` (`flusher_loop`) |
| `resolve`              | `resolve_current_names` `:682-742` |
| `groupOrder`           | `deferred.sort_by(graph_name)` + `chunk_by` (`flusher_loop`) |
| `fieldValues`          | `StreamTemplate::add` value array `:176-201` |
| `failMsg`              | `StreamTemplate::report_failures` `:228-240` |
-/
namespace RedisLayer.Telemetry

/-! ## Truncation -/

def STR_MAX_LEN := 2048

/-- `truncate` / `truncate_arc`: at most `STR_MAX_LEN` chars, `"..."` appended if cut.
(`char_indices().nth(N)` exists iff there are more than `N` chars.) -/
def truncate (s : List Char) : List Char :=
  if STR_MAX_LEN < s.length then s.take STR_MAX_LEN ++ "...".toList else s

theorem truncate_short (s : List Char) (h : s.length ≤ STR_MAX_LEN) : truncate s = s := by
  unfold truncate; rw [if_neg (by omega)]

theorem truncate_len (s : List Char) : (truncate s).length ≤ STR_MAX_LEN + 3 := by
  unfold truncate; split
  · simp [STR_MAX_LEN]; omega
  · omega

theorem truncate_prefix (s : List Char) : (truncate s).take (min s.length STR_MAX_LEN) = s.take STR_MAX_LEN := by
  unfold truncate; split
  · rename_i h; have : min s.length STR_MAX_LEN = STR_MAX_LEN := by rw [Nat.min_def]; split <;> omega
    rw [this, List.take_append_of_le_length (by simp [STR_MAX_LEN] at h ⊢; omega)]
    simp [List.take_take]
  · rename_i h; have : min s.length STR_MAX_LEN = s.length := by rw [Nat.min_def]; split <;> omega
    rw [this, List.take_length, List.take_of_length_le (by omega)]

/-- `truncate_arc` is the same function (it only avoids the copy when nothing is cut). -/
def truncateArc := truncate
theorem truncateArc_eq : truncateArc = truncate := rfl

/-! ## Stream names -/

/-- `up_to_nul` (`graph_core.rs:87`): the prefix before the first NUL. -/
def upToNul : List Char → List Char
  | [] => []
  | c :: cs => if c = '\x00' then [] else c :: upToNul cs

def streamName (g : List Char) : List Char := "telemetry{".toList ++ upToNul g ++ "}".toList

theorem upToNul_idem (s : List Char) : upToNul (upToNul s) = upToNul s := by
  induction s with
  | nil => rfl
  | cons c cs ih => by_cases h : c = '\x00' <;> simp [upToNul, h, ih]

/-- The key a graph lives at and its name stream to the same key (`delete_stream` relies on
this: "`graph_name` may be either the graph's name or the key it lives at"). -/
theorem streamName_key_name (g : List Char) : streamName (upToNul g) = streamName g := by
  simp [streamName, upToNul_idem]

theorem upToNul_nofree (s : List Char) (h : '\x00' ∉ s) : upToNul s = s := by
  induction s with
  | nil => rfl
  | cons c cs ih =>
    have hc : c ≠ '\x00' := fun e => h (by simp [e])
    have : '\x00' ∉ cs := fun m => h (by simp [m])
    simp [upToNul, hc, ih this]

/-- Distinct NUL-free graph names get distinct streams. -/
theorem streamName_inj (a b : List Char) (ha : '\x00' ∉ a) (hb : '\x00' ∉ b)
    (h : streamName a = streamName b) : a = b := by
  simp only [streamName, upToNul_nofree a ha, upToNul_nofree b hb] at h
  rw [List.append_assoc] at h
  have := List.append_cancel_left h
  exact List.append_cancel_right this

/-- …and two keys that agree up to their first NUL share one (C: `telemetry{%s}` of the
`rm_strdup`ed name). -/
theorem streamName_nul (a b c : List Char) (hb : '\x00' ∉ a) :
    streamName (a ++ '\x00' :: b) = streamName (a ++ '\x00' :: c) := by
  have : ∀ x, upToNul (a ++ '\x00' :: x) = a := by
    intro x; induction a with
    | nil => simp [upToNul]
    | cons d ds ih =>
      have hd : d ≠ '\x00' := fun e => hb (by simp [e])
      simp [upToNul, hd]; exact ih (fun m => hb (by simp [m]))
  simp [streamName, this]

/-- From the first `d` on (`[]` if none). -/
def dropTo (d : Char) : List Char → List Char
  | [] => []
  | c :: cs => if c = d then c :: cs else dropTo d cs
/-- Up to the first `d`, and whether one was found. -/
def takeTo (d : Char) : List Char → List Char × Bool
  | [] => ([], false)
  | c :: cs => if c = d then ([], true) else let r := takeTo d cs; (c :: r.1, r.2)

/-- Redis Cluster hash tag (`keyHashSlot`): the part between the first `{` and the next
`}` if that is non-empty, else the whole key. -/
def hashTag (k : List Char) : List Char :=
  match dropTo '{' k with
  | [] => k
  | _ :: rest =>
    let t := takeTo '}' rest
    if t.2 ∧ t.1 ≠ [] then t.1 else k

theorem dropTo_none (d : Char) (x : List Char) (h : d ∉ x) : dropTo d x = [] := by
  induction x with
  | nil => rfl
  | cons c cs ih =>
    have : c ≠ d := fun e => h (by simp [e])
    simp [dropTo, this, ih (fun m => h (by simp [m]))]

theorem takeTo_close (d : Char) (x y : List Char) (h : d ∉ x) : takeTo d (x ++ d :: y) = (x, true) := by
  induction x with
  | nil => simp [takeTo]
  | cons c cs ih =>
    have : c ≠ d := fun e => h (by simp [e])
    simp [takeTo, this, ih (fun m => h (by simp [m]))]

/-- For a brace-free, NUL-free, non-empty graph name the stream hashes to the graph key's
slot. -/
theorem stream_same_slot (g : List Char) (h1 : '{' ∉ g) (h2 : '}' ∉ g) (h0 : '\x00' ∉ g)
    (hne : g ≠ []) : hashTag (streamName g) = g ∧ hashTag g = g := by
  refine ⟨?_, by simp [hashTag, dropTo_none '{' g h1]⟩
  have e : streamName g = "telemetry".toList ++ '{' :: (g ++ '}' :: []) := by
    simp [streamName, upToNul_nofree g h0]
  have hd : dropTo '{' ("telemetry".toList ++ '{' :: (g ++ '}' :: [])) = '{' :: (g ++ '}' :: []) := by
    simp [dropTo]
  rw [e, hashTag, hd]
  simp only [takeTo_close '}' g [] h2]
  simp [hne]

/-- A graph name with braces does not: `a{b}c` lives in slot(`b`), its stream in
slot(`a{b`). (Same in C, which builds the identical `telemetry{%s}`.) -/
theorem stream_other_slot :
    hashTag (streamName "a{b}c".toList) = "a{b".toList ∧ hashTag "a{b}c".toList = "b".toList := by
  decide

end RedisLayer.Telemetry
