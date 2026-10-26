import FalkorAggOps.LoadCsv
/-
# LOAD CSV redirects (#3443, fixed by #3458): every hop of a remote fetch, not just the first

`LoadCsv.lean` treats a remote LOAD CSV as one request to the URL that
`validate_remote_url` accepted. The request is a ureq 3.4.2 agent
(`open_csv_records`, load_csv.rs:376-398) built from `http_config()`
(:224-234). On b84473a7f `http_config` set only the two timeouts, so the agent
kept ureq's defaults `max_redirects = 10` and `https_only = false`, redirects
included, while its resolver `PinnedResolver` (:193-214) ignores the `_uri` it
is asked about and returns the addresses validated for the first URL. That was
#3443. On origin/main (5e475e6c9, #3458) `http_config` also sets
`https_only(true)` and `max_redirects(0)`; `open_csv_records` turns a 3xx reply
into the error "LOAD CSV does not follow redirects" (:388-399), as C (libcurl
without `CURLOPT_FOLLOWLOCATION`) does.

| here | there |
| --- | --- |
| `Url` | the request URL; `scheme = .https` is `starts_with("https://")`, hosts are names (as `Nat`) |
| `validateU` | `validate_remote_url` (:81-142) on a parsed URL; `dns` = `to_socket_addrs` |
| `Cfg` | `ureq::config::Config` (`max_redirects`, `https_only`) |
| `httpCfgMain` | `http_config` (:224-234) on main: `max_redirects = 0`, `https_only = true` |
| `httpCfgPre3443` | `http_config` on b84473a7f: ureq 3.4.2's defaults (10, false) |
| `pinnedRes` | `PinnedResolver::resolve` (:193-214): `pinned addrs` for every URI |
| `fetch` | the agent's request loop: resolve, connect, follow a 3xx `Location` while redirects remain; with `https_only` a non-https hop is refused before connecting |
| `loadRemote` | the remote arm of `open_csv_records` on main: the agent, then a 3xx answer is an error |
| `loadRemotePre3443` | the same arm on b84473a7f (no 3xx check, `httpCfgPre3443`) |

Property `EachHopValidated`: every connection the fetch makes is an https
request whose addresses were validated for *that request's* host.
`pre3443_redirect_uses_first_host_ip` (cross-host redirect connects to the
first host's IP) and `pre3443_redirect_downgrades_to_http` (an http hop is
attempted) refute it on b84473a7f. `fixed3443_eachHopValidated` proves it for
main's `loadRemote` for every server and DNS answer; `fixed3443_redirect_is_error`
says a redirecting first URL ends in an error, never a second connection.
`fixed_eachHopValidated` (https_only + per-URL validate-then-pin resolver,
any budget) and `noRedirect_single` stay as the general fix-shape lemmas.
-/
namespace FalkorAggOps.LoadCsv

inductive Scheme where
  | https
  | http
  deriving DecidableEq

structure Url where
  scheme : Scheme
  host : Nat
  path : Nat
  deriving DecidableEq

/-- `validate_remote_url` (:81-142) on a parsed URL, as `validateRemote`: https
only, a non-empty DNS answer, every address public. -/
def validateU (dns : Nat → List IP) (u : Url) : Except String (List IP) :=
  if u.scheme ≠ .https then .error "Only https:// URLs are allowed for LOAD CSV"
  else if (dns u.host).isEmpty then .error "DNS resolution returned no addresses"
  else if (dns u.host).any ipForbidden then .error "LOAD CSV refused: non-public address"
  else .ok (dns u.host)

theorem validateU_spec (dns : Nat → List IP) (u : Url) (as : List IP) (h : validateU dns u = .ok as) :
    u.scheme = .https ∧ as = dns u.host ∧ as ≠ [] ∧ ∀ a ∈ as, ipForbidden a = false := by
  unfold validateU at h
  split at h
  · cases h
  · rename_i hs
    split at h
    · cases h
    · rename_i hne
      split at h
      · cases h
      · rename_i hf
        cases h
        refine ⟨by cases hu : u.scheme <;> simp_all, rfl, fun e => by simp [e] at hne, fun a ha => ?_⟩
        cases hfa : ipForbidden a
        · rfl
        · exact absurd (List.any_eq_true.mpr ⟨a, ha, hfa⟩) hf

/-- The two `ureq::config::Config` fields the redirect path reads. -/
structure Cfg where
  maxRedirects : Nat
  httpsOnly : Bool

/-- `http_config()` on b84473a7f: only timeouts are set, so ureq 3.4.2's
defaults stand: `max_redirects = 10`, `https_only = false`. -/
def httpCfgPre3443 : Cfg := ⟨10, false⟩

/-- `http_config()` (:224-234) on main (#3458): `https_only(true)`,
`max_redirects(0)`. -/
def httpCfgMain : Cfg := ⟨0, true⟩

/-- `PinnedResolver::resolve` (:193-214): the URI is ignored. -/
def pinnedRes (addrs : List IP) : Url → Except String (List IP) := fun _ => pinned addrs

/-- The agent's request loop. `serve u = some loc` is a 3xx with `Location: loc`.
Each element is one connection: the request URL and the addresses the resolver
handed the connector for it. -/
def fetch (cfg : Cfg) (res : Url → Except String (List IP)) (serve : Url → Option Url) :
    Nat → Url → List (Url × List IP)
  | 0, u =>
    if cfg.httpsOnly && u.scheme != .https then []
    else match res u with
      | .error _ => []
      | .ok as => [(u, as)]
  | f + 1, u =>
    if cfg.httpsOnly && u.scheme != .https then []
    else match res u with
      | .error _ => []
      | .ok as => (u, as) :: match serve u with
        | some loc => fetch cfg res serve f loc
        | none => []

/-- The remote arm of `open_csv_records` on b84473a7f (:359-378): validate the
first URL, pin its addresses into the agent, `get(path).call()`; redirects
followed up to `maxRedirects`. -/
def loadRemotePre3443 (dns : Nat → List IP) (serve : Url → Option Url) (u : Url) :
    Except String (List (Url × List IP)) :=
  (validateU dns u).map fun addrs =>
    fetch httpCfgPre3443 (pinnedRes addrs) serve httpCfgPre3443.maxRedirects u

/-- The remote arm of `open_csv_records` on main (:376-399): the same agent with
`httpCfgMain`; when the first URL answers 3xx (`serve u = some _`) the reply is
returned as is and the function errors with "does not follow redirects". -/
def loadRemote (dns : Nat → List IP) (serve : Url → Option Url) (u : Url) :
    Except String (List (Url × List IP)) :=
  match validateU dns u with
  | .error e => .error e
  | .ok addrs =>
    match fetch httpCfgMain (pinnedRes addrs) serve httpCfgMain.maxRedirects u with
    | [] => .ok []
    | c :: cs =>
      match serve u with
      | some _ => .error "LOAD CSV does not follow redirects"
      | none => .ok (c :: cs)

/-- **Property**: every connection is https and uses addresses validated for its
own host. -/
def EachHopValidated (dns : Nat → List IP) (conns : List (Url × List IP)) : Prop :=
  ∀ c ∈ conns, c.1.scheme = .https ∧ ∃ as, validateU dns c.1 = .ok as ∧ ∀ a ∈ c.2, a ∈ as

/-! ## Counterexamples on b84473a7f (history) -/

def ipA : IP := .v4 ⟨140, 82, 112, 3⟩     -- public, first host
def ipB : IP := .v4 ⟨185, 199, 108, 133⟩  -- public, redirect target
def dnsAB : Nat → List IP := fun h => if h = 1 then [ipA] else if h = 2 then [ipB] else []
def u0 : Url := ⟨.https, 1, 0⟩

/-- **#3443, cross-host redirect (b84473a7f)** (live: httpbin `redirect-to` a
raw.githubusercontent.com CSV fails with "certificate not valid for name"):
`https://a/start` answers `302 Location: https://b/data.csv`; the second
connection is to `b` but at `a`'s pinned address. -/
theorem pre3443_redirect_uses_first_host_ip :
    let serve : Url → Option Url := fun u => if u = u0 then some ⟨.https, 2, 1⟩ else none
    loadRemotePre3443 dnsAB serve u0 = .ok [(u0, [ipA]), (⟨.https, 2, 1⟩, [ipA])] ∧
    ¬ EachHopValidated dnsAB [(u0, [ipA]), (⟨.https, 2, 1⟩, [ipA])] := by
  refine ⟨rfl, fun h => ?_⟩
  obtain ⟨_, as, hv, hm⟩ := h (⟨.https, 2, 1⟩, [ipA]) (by simp)
  have : validateU dnsAB ⟨.https, 2, 1⟩ = .ok [ipB] := rfl
  rw [this] at hv; cases hv
  have := hm ipA (by simp)
  simp [ipA, ipB] at this

/-- **#3443, downgrade (b84473a7f)** (live: a redirect to `http://httpbin.org/get` is
followed and fails with `http status: 400`, plain HTTP sent to the pinned :443):
`https_only = false`, so an http hop is connected. -/
theorem pre3443_redirect_downgrades_to_http :
    let serve : Url → Option Url := fun u => if u = u0 then some ⟨.http, 1, 1⟩ else none
    loadRemotePre3443 dnsAB serve u0 = .ok [(u0, [ipA]), (⟨.http, 1, 1⟩, [ipA])] ∧
    ¬ EachHopValidated dnsAB [(u0, [ipA]), (⟨.http, 1, 1⟩, [ipA])] := by
  refine ⟨rfl, fun h => ?_⟩
  have := (h (⟨.http, 1, 1⟩, [ipA]) (by simp)).1
  cases this

/-- With no redirects followed (C: libcurl without `CURLOPT_FOLLOWLOCATION`; main
since #3458) the fetch is the one validated request, so `EachHopValidated` holds
with `pinnedRes`, whatever `https_only` is (the validated URL is https). -/
theorem noRedirect_single (dns : Nat → List IP) (serve : Url → Option Url) (u : Url) (addrs : List IP)
    (b : Bool) (hv : validateU dns u = .ok addrs) :
    EachHopValidated dns (fetch ⟨0, b⟩ (pinnedRes addrs) serve 0 u) := by
  have hsch := (validateU_spec dns u addrs hv).1
  intro c hc
  simp only [fetch, hsch, bne_self_eq_false, Bool.and_false, Bool.false_eq_true, ite_false, pinnedRes] at hc
  split at hc
  · cases hc
  · rename_i as hr
    rw [List.mem_singleton] at hc; subst hc
    exact ⟨hsch, addrs, hv, fun a ha => (pinned_spec addrs as hr).2 a ha⟩

/-! ## Main (#3458) -/

/-- **`fixed3443_eachHopValidated`**: main's `loadRemote` (https_only, no
redirects, pinned resolver) makes only connections that are https to an address
validated for their own host, for every DNS answer and every server. -/
theorem fixed3443_eachHopValidated (dns : Nat → List IP) (serve : Url → Option Url) (u : Url)
    (conns : List (Url × List IP)) (h : loadRemote dns serve u = .ok conns) :
    EachHopValidated dns conns := by
  unfold loadRemote at h
  split at h
  · cases h
  · rename_i addrs hv
    split at h
    · cases h; intro c hc; cases hc
    · rename_i c cs hf
      split at h
      · cases h
      · cases h
        rw [← hf]
        exact noRedirect_single dns serve u addrs true hv

/-- **`fixed3443_redirect_is_error`**: a first URL that answers 3xx is an error
(never a second connection), as in C. -/
theorem fixed3443_redirect_is_error (dns : Nat → List IP) (serve : Url → Option Url) (u : Url)
    (loc : Url) (addrs : List IP) (hv : validateU dns u = .ok addrs) (hs : serve u = some loc) :
    loadRemote dns serve u = .error "LOAD CSV does not follow redirects" := by
  obtain ⟨hsch, -, hne, -⟩ := validateU_spec dns u addrs hv
  have hp : pinned addrs = .ok (addrs.take 16) := by
    unfold pinned
    have : (addrs.take 16).isEmpty = false := by
      cases addrs with
      | nil => exact absurd rfl hne
      | cons a t => simp
    simp only [this]
    simp
  simp [loadRemote, hv, fetch, pinnedRes, httpCfgMain, hs, hp, hsch]

/-! ## What a fix proves -/

/-- Validate each hop and pin its own addresses (`pinned` keeps a subset). -/
def perUrlRes (dns : Nat → List IP) : Url → Except String (List IP) :=
  fun u => match validateU dns u with
    | .ok as => pinned as
    | .error e => .error e

/-- **`fixed_eachHopValidated`**: with `https_only` and a resolver that validates
and pins per request, every connection of every redirect chain satisfies
`EachHopValidated`, whatever the servers answer and the redirect budget. -/
theorem fixed_eachHopValidated (dns : Nat → List IP) (serve : Url → Option Url) (n : Nat) (fuel : Nat) (u : Url) :
    EachHopValidated dns (fetch ⟨n, true⟩ (perUrlRes dns) serve fuel u) := by
  have head : ∀ (u : Url) (as : List IP), perUrlRes dns u = .ok as →
      u.scheme = .https ∧ ∃ vs, validateU dns u = .ok vs ∧ ∀ a ∈ as, a ∈ vs := by
    intro u as hr
    unfold perUrlRes at hr
    split at hr
    · rename_i vs hv
      exact ⟨(validateU_spec dns u vs hv).1, vs, hv, fun a ha => (pinned_spec vs as hr).2 a ha⟩
    · cases hr
  induction fuel generalizing u with
  | zero =>
    intro c hc
    simp only [fetch] at hc
    split at hc
    · cases hc
    · split at hc
      · cases hc
      · rename_i as hr
        rw [List.mem_singleton] at hc; subst hc; exact head u as hr
  | succ f ih =>
    intro c hc
    simp only [fetch] at hc
    split at hc
    · cases hc
    · split at hc
      · cases hc
      · rename_i as hr
        rcases List.mem_cons.mp hc with rfl | hc
        · exact head u as hr
        · split at hc
          · rename_i loc _
            exact ih loc c hc
          · cases hc

end FalkorAggOps.LoadCsv
