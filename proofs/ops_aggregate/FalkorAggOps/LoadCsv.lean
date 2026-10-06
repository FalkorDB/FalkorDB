/-
LOAD CSV (graph/src/runtime/ops/load_csv.rs).

| here | there |
| --- | --- |
| `v4Forbidden`        | `ipv4_is_forbidden` (:48) |
| `validateRemote`     | `validate_remote_url` (:81) — DNS (`to_socket_addrs`) is the FFI parameter `resolve` |
| `ER`, `erNew`, `erRead` | `EnforcingReader::new` (:153), `read` (:167) |
| `pinned`             | `PinnedResolver::resolve` (:194) |
| `httpConfig`         | `http_config` (:218), a `OnceLock` |
| `csvNext`            | `CsvRecordIter::next` (:253) |
| `LcSt.new`           | `LoadCsvOp::new` (:321) |
| `headerPlan`, `openSource` | `open_csv_records` (:350) |
| `resolvePath`, `lcRow` | the per-row closure of `LoadCsvOp::next` (:438) |
-/
namespace FalkorAggOps.LoadCsv

/-! ## SSRF filter -/

/-- An IPv4 address as its four octets. -/
structure V4 where
  a : Nat
  b : Nat
  c : Nat
  d : Nat

/-- load_csv.rs:48-63 (std `is_loopback`/`is_private`/… spelled out on the octets). -/
def v4Forbidden (ip : V4) : Bool :=
  ip.a == 127 ||                                                   -- loopback
  ip.a == 10 || (ip.a == 172 && 16 ≤ ip.b && ip.b ≤ 31) || (ip.a == 192 && ip.b == 168) || -- private
  (ip.a == 169 && ip.b == 254) ||                                  -- link-local
  (224 ≤ ip.a && ip.a ≤ 239) ||                                    -- multicast
  (ip.a == 255 && ip.b == 255 && ip.c == 255 && ip.d == 255) ||    -- broadcast
  (ip.a == 0 && ip.b == 0 && ip.c == 0 && ip.d == 0) ||            -- unspecified
  (ip.a == 192 && ip.b == 0 && ip.c == 2) || (ip.a == 198 && ip.b == 51 && ip.c == 100) ||
  (ip.a == 203 && ip.b == 0 && ip.c == 113) ||                     -- documentation
  (ip.a == 100 && (ip.b &&& 0xc0) == 0x40) ||                      -- 100.64/10 shared
  (ip.a == 198 && (ip.b &&& 0xfe) == 0x12) ||                      -- 198.18/15 benchmarking
  (ip.a == 192 && ip.b == 0 && ip.c == 0) ||                       -- 192.0.0/24 IETF
  240 ≤ ip.a                                                       -- reserved

theorem v4Forbidden_examples :
    v4Forbidden ⟨127, 0, 0, 1⟩ = true ∧ v4Forbidden ⟨10, 1, 2, 3⟩ = true ∧ v4Forbidden ⟨169, 254, 169, 254⟩ = true ∧
    v4Forbidden ⟨100, 64, 0, 1⟩ = true ∧ v4Forbidden ⟨100, 128, 0, 1⟩ = false ∧ v4Forbidden ⟨8, 8, 8, 8⟩ = false ∧
    v4Forbidden ⟨172, 32, 0, 1⟩ = false ∧ v4Forbidden ⟨198, 19, 0, 1⟩ = true := by decide

inductive IP where
  | v4 (ip : V4)
  | v6 (forbidden : Bool)   -- the v6 checks of :124-130 (incl. a forbidden v4-mapped address)

def ipForbidden : IP → Bool
  | .v4 ip => v4Forbidden ip
  | .v6 f => f

/-- load_csv.rs:81-136 after parsing: https only, a host, a port, a non-empty DNS answer, and
EVERY resolved address public. `parse` = the host/port split (`none` = malformed). -/
def validateRemote (parse : String → Except String (String × Nat)) (resolve : String → Nat → Except String (List IP))
    (url : String) : Except String (List IP) := do
  if !url.startsWith "https://" then throw "Only https:// URLs are allowed for LOAD CSV"
  let (host, port) ← parse url
  let addrs ← resolve host port
  if addrs.isEmpty then throw s!"DNS resolution returned no addresses for '{host}'"
  if addrs.any ipForbidden then throw s!"LOAD CSV refused: host '{host}' resolves to a non-public address"
  pure addrs

/-- **SSRF**: an accepted URL is https and every address the request may use is public — the
pinned resolver (`pinned`) only ever hands these addresses to the connector. -/
theorem validateRemote_spec (parse : String → Except String (String × Nat)) (resolve : String → Nat → Except String (List IP))
    (url : String) (addrs : List IP) (h : validateRemote parse resolve url = .ok addrs) :
    url.startsWith "https://" = true ∧ addrs ≠ [] ∧ ∀ a ∈ addrs, ipForbidden a = false := by
  unfold validateRemote at h
  by_cases hs : url.startsWith "https://" = true
  · simp only [hs, Bool.not_true, Bool.false_eq_true, ite_false] at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    · rename_i hp
      split at h
      · cases h
      · rename_i as _
        split at h
        · cases h
        · rename_i hne
          split at h
          · cases h
          · rename_i hf
            cases h
            refine ⟨hs, fun e => by simp [e] at hne, fun a ha => ?_⟩
            cases hfa : ipForbidden a
            · rfl
            · exact absurd (List.any_eq_true.mpr ⟨a, ha, hfa⟩) hf
  · simp [hs, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-! ## Size cap (`EnforcingReader`) -/

structure ER where
  remaining : Nat   -- `Take` budget: `limit + 1`
  limit : Nat

def erNew (limit : Nat) : ER := ⟨limit + 1, limit⟩

/-- One `read` of up to `want` bytes from a source that has `avail`: an error once the budget
(limit + 1) is exhausted by a non-empty read, i.e. once more than `limit` bytes were seen. -/
def erRead (r : ER) (want avail : Nat) : Except String (Nat × ER) :=
  let n := min want (min avail r.remaining)
  let r' := { r with remaining := r.remaining - n }
  if n > 0 && r'.remaining == 0 then .error s!"CSV payload exceeds the {r.limit} byte limit" else .ok (n, r')

def erRun (r : ER) : List (Nat × Nat) → Except String Nat
  | [] => .ok 0
  | (w, a) :: rs => do
    let (n, r') ← erRead r w a
    let m ← erRun r' rs
    pure (n + m)

/-- **Never more than `limit` bytes are delivered**: every successful run of reads stays
within the budget. -/
theorem erRun_le (rs : List (Nat × Nat)) : ∀ (r : ER) (n : Nat), erRun r rs = .ok n → n < r.remaining ∨ n = 0 := by
  induction rs with
  | nil => intro r n h; simp [erRun, pure, Except.pure] at h; omega
  | cons x xs ih =>
    intro r n h
    obtain ⟨w, a⟩ := x
    simp only [erRun, bind, Except.bind] at h
    cases hp : erRead r w a with
    | error e => rw [hp] at h; cases h
    | ok p =>
      rw [hp] at h
      obtain ⟨k, r'⟩ := p
      cases hm : erRun r' xs with
      | error e => simp only [hm] at h; cases h
      | ok m =>
        simp only [hm, pure, Except.pure, Except.ok.injEq] at h
        subst h
        unfold erRead at hp
        simp only at hp
        have hkR : min w (min a r.remaining) ≤ r.remaining :=
          Nat.le_trans (Nat.min_le_right _ _) (Nat.min_le_right _ _)
        split at hp
        · cases hp
        · rename_i hc
          simp only [Except.ok.injEq, Prod.mk.injEq] at hp
          obtain ⟨rfl, rfl⟩ := hp
          have := ih _ m hm
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, not_and] at hc
          simp only at this
          by_cases hk : min w (min a r.remaining) > 0
          · have h2 := hc hk; left; omega
          · omega

theorem erNew_spec (l : Nat) : (erNew l).remaining = l + 1 ∧ (erNew l).limit = l := ⟨rfl, rfl⟩

/-! ## Pinned resolver, client config -/

/-- load_csv.rs:194-215: at most 16 of the validated addresses; none → `HostNotFound`. -/
def pinned (addrs : List IP) : Except String (List IP) :=
  let out := addrs.take 16
  if out.isEmpty then .error "HostNotFound" else .ok out

theorem pinned_spec (addrs : List IP) (out : List IP) (h : pinned addrs = .ok out) :
    out.length ≤ 16 ∧ ∀ a ∈ out, a ∈ addrs := by
  unfold pinned at h
  simp only at h
  split at h
  · cases h
  · cases h; exact ⟨List.length_take_le _ _, fun a ha => List.mem_of_mem_take ha⟩

/-- load_csv.rs:218: `OnceLock::get_or_init` — built once, then shared. -/
def httpConfig {C : Type} (init : C) : Option C → C × Option C
  | none => (init, some init)
  | some c => (c, some c)

theorem httpConfig_spec {C : Type} (init : C) (cell : Option C) :
    (httpConfig init (httpConfig init cell).2).1 = (httpConfig init cell).1 := by
  cases cell <;> rfl

/-! ## Records -/

inductive CV where
  | null
  | str (s : String)
  | list (vs : List CV)
  | map (kvs : List (String × CV))

/-- load_csv.rs:350-380: header name ↦ column index; a duplicated name keeps its FIRST position
in the plan but reads the LAST column carrying it. -/
def headerPlan (hdr : List String) : List (String × Nat) :=
  (hdr.zipIdx).foldl (fun acc (n, i) =>
    if acc.any (·.1 == n) then acc.map (fun p => if p.1 == n then (n, i) else p) else acc ++ [(n, i)]) []

theorem headerPlan_example : headerPlan ["a", "b", "a"] = [("a", 2), ("b", 1)] := by decide

/-- load_csv.rs:253-301: parked error ⇒ stop; a record becomes a list (empty field ⇒ null) or a
map over the header plan (empty fields omitted); a parse error is parked and ends the stream. -/
def csvNext (plan : Option (List (String × Nat))) (err : Option String) (rec : Option (Except String (List String))) :
    Option CV × Option String :=
  match err with
  | some e => (none, some e)
  | none => match rec with
    | none => (none, none)
    | some (.error e) => (none, some s!"Failed to read CSV record: {e}")
    | some (.ok fields) => match plan with
      | none => (some (.list (fields.map fun f => if f.isEmpty then .null else .str f)), none)
      | some cols => (some (.map (cols.filterMap fun (n, i) => match fields[i]? with
          | some f => if f.isEmpty then none else some (n, .str f)
          | none => none)), none)

theorem csvNext_spec (fields : List String) (e : String) :
    csvNext none none (some (.ok fields)) = (some (.list (fields.map fun f => if f.isEmpty then .null else .str f)), none) ∧
    csvNext none none (some (.error e)) = (none, some s!"Failed to read CSV record: {e}") ∧
    (∀ p r, csvNext p (some e) r = (none, some e)) := ⟨rfl, rfl, fun _ _ => rfl⟩

/-- load_csv.rs:321-339 -/
structure LcSt where
  col : Nat
  cap : Option Nat
  err : Option String

def LcSt.new (var : Nat) : LcSt := ⟨var, none, none⟩
theorem lcNew_spec (v : Nat) : LcSt.new v = ⟨v, none, none⟩ := rfl

/-- load_csv.rs:357-379: the byte source — remote via the validated, pinned fetch; else a local
file, both under the 100 MiB cap. -/
def openSource (validated : String → Except String (List IP)) (openLocal : String → Except String Unit)
    (path : String) : Except String Bool :=
  if path.startsWith "https://" then (validated path).map fun _ => true
  else (openLocal path).map fun _ => false

theorem openSource_remote (v : String → Except String (List IP)) (o : String → Except String Unit) (p : String)
    (hp : p.startsWith "https://" = true) (e : String) (he : v p = .error e) : openSource v o p = .error e := by
  simp [openSource, hp, he, Except.map]

/-! ## Path resolution (`next`, :455-500) -/

/-- `file://` paths are joined under the import folder and must stay inside it after
canonicalisation; `https://` passes to the remote path; anything else is refused. -/
def resolvePath (canon : String → Except String String) (isWithin : String → String → Bool) (folder : String)
    (path : String) : Except String String :=
  if path.startsWith "file://" then do
    let base ← canon folder
    let c ← canon (folder ++ "/" ++ ((path.drop 7).dropWhile (· == '/')).toString)
    if isWithin c base then pure c
    else throw s!"File path is not within the import folder '{folder}'"
  else if path.startsWith "https://" then pure path
  else throw "File path must start with 'file://' prefix"

/-- **Path containment**: an accepted `file://` path resolves inside the canonical import
folder; only `file://` and `https://` are accepted. -/
theorem resolvePath_spec (canon : String → Except String String) (isWithin : String → String → Bool)
    (folder path out : String) (h : resolvePath canon isWithin folder path = .ok out) :
    (path.startsWith "file://" = true → ∃ base, canon folder = .ok base ∧ isWithin out base = true) ∧
    (path.startsWith "file://" = false → path.startsWith "https://" = true ∧ out = path) := by
  unfold resolvePath at h
  constructor
  · intro hf
    simp only [hf, ite_true, bind, Except.bind] at h
    split at h
    · cases h
    · rename_i base hb
      split at h
      · cases h
      · rename_i c hc
        split at h
        · rename_i hw; simp [pure, Except.pure] at h; subst h; exact ⟨base, hb, hw⟩
        · simp [throw, throwThe, MonadExceptOf.throw] at h
  · intro hf
    simp only [hf, Bool.false_eq_true, ite_false] at h
    split at h
    · rename_i hs; simp [pure, Except.pure] at h; exact ⟨hs, h.symm⟩
    · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- The per-row checks before opening (:443-454): a string delimiter of one byte, a string path. -/
def lcRowArgs (delim : Option String) (path : Option String) : Except String (String × String) :=
  match delim with
  | none => .error "Delimiter must be a string"
  | some d => if d.utf8ByteSize ≠ 1 then .error "CSV field terminator can only be one character wide"
    else match path with
      | none => .error "File path must be a string"
      | some p => .ok (d, p)

theorem lcRowArgs_spec (p : String) :
    lcRowArgs (some ",") (some p) = .ok (",", p) ∧ lcRowArgs none (some p) = .error "Delimiter must be a string" ∧
    lcRowArgs (some "ab") (some p) = .error "CSV field terminator can only be one character wide" := by
  have h1 : ",".utf8ByteSize = 1 := by decide
  have h2 : "ab".utf8ByteSize = 2 := by decide
  refine ⟨by simp [lcRowArgs, h1], rfl, by simp [lcRowArgs, h2]⟩

end FalkorAggOps.LoadCsv
