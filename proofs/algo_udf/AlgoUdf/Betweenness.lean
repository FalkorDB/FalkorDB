import AlgoUdf.Config
import AlgoUdf.Compact
/-! # algo.betweenness source sampling vs C (#3380) — gap closed 2026-10-20

Extends `Config.samplingSize` and `Compact.seed0Sources` (which only bounded the
seed-0 path) with the *seeded* draw loop and the C reference, so the C-parity
defaults and the "one source per requested sample" property can be stated.

| here | there |
| --- | --- |
| `Config.samplingSize` (existing) | default `16` at `algo_procedures.rs:928` |
| `samplingSeed`  | `samplingSeed` parse `algo_procedures.rs:938-943` (absent/null → `0`, `i64 as u64`) |
| `lcg`, `drawLoop`, `rustSources` | source selection `algo_procedures.rs:991-1012` (seed 0: `i % n`; else the 64-bit LCG, `(rng >> 33) % n`, *deduplicated through `used`*) |
| `cDefaultSize`  | C `BETWEENNESS_DEFAULT_SAMPLE_SIZE 32` (`src/procedures/proc_betweenness.c:21`, origin/master d7cf86675) |
| `cEffSeed`      | C `if(samplingSeed == 0) samplingSeed = (uint32_t)time(NULL)` (`proc_betweenness.c:318-319`) |
| `xorshift`, `cDraws` | C `_Random_Sources` (`proc_betweenness.c:46-96`) + `simple_rand` (`src/util/simple_rand.h`): exactly `MIN(nvals, samplingSize)` draws, duplicates kept |

C draws positions in `AT`'s index array (one per stored entry); the model keeps the
draw *count* and the seed handling, which is what the properties are about, and
abstracts which node a position names.
-/
namespace AlgoUdf.Betweenness
open AlgoUdf

/-! ## Rust (fb3d516d0) -/

/-- `samplingSeed`: absent/null → 0, an Int → `*n as u64`, anything else errors. -/
def samplingSeed (m : List (String × Val)) : Except String Nat :=
  match lookup m "samplingSeed" with
  | none | some .null => .ok 0
  | some (.int n) => .ok (n % 18446744073709551616).toNat
  | some _ => .error "samplingSeed must be an integer"

def two64 : Nat := 18446744073709551616

/-- `rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)`. -/
def lcg (r : Nat) : Nat := (r * 6364136223846793005 + 1442695040888963407) % two64

/-- The seeded loop: `k` draws left, LCG state `r`, `acc` = `srcs` (pushed only when
`used.insert(idx)` is true). -/
def drawLoop (n : Nat) : Nat → Nat → List Nat → List Nat
  | 0, _, acc => acc
  | k + 1, r, acc =>
    let r' := lcg r
    let i := (r' / 8589934592) % n
    drawLoop n k r' (if i ∈ acc then acc else acc ++ [i])

/-- The seed-0 loop: index `i % n` for `i < s`, deduplicated. -/
def seed0Loop (n : Nat) (s : Nat) : List Nat :=
  (List.range s).foldl (fun acc i => if i % n ∈ acc then acc else acc ++ [i % n]) []

/-- `sources` (`algo_procedures.rs:991-1012`) for `samplingSize s`, `n` compact
indices and seed `seed`. -/
def rustSources (s n seed : Nat) : List Nat :=
  if n = 0 then []
  else if s ≥ n then List.range n
  else if seed = 0 then seed0Loop n s
  else drawLoop n s seed []

/-- The Rust run with no `samplingSeed`: nothing reads a clock (`:966` gives 0 and the
seed-0 branch `:1035` is `i % n_nodes`), so the clock argument is ignored. -/
def rustSeedless (s n : Nat) (_clock : Nat) : List Nat := rustSources s n 0

/-! ## C (origin/master) -/

def cDefaultSize : Nat := 32

/-- `samplingSeed == 0` (absent or given as 0) is replaced by the wall clock. -/
def cEffSeed (seed clock : Nat) : Nat := if seed = 0 then clock % 4294967296 else seed

/-- `simple_rand`: 64-bit xorshift (13, 7, 17). -/
def xorshift (s : Nat) : Nat :=
  let s := Nat.xor s ((s <<< 13) % two64)
  let s := Nat.xor s (s >>> 7)
  Nat.xor s ((s <<< 17) % two64)

/-- `_Random_Sources`: `MIN(nvals, samplingSize)` positions `simple_rand % nvals`,
duplicates kept. -/
def cDrawsAux (nvals : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | k + 1, st => let st' := xorshift st; (st' % nvals) :: cDrawsAux nvals k st'

def cDraws (s nvals seed : Nat) : List Nat := cDrawsAux nvals (min nvals s) seed

/-- The C run with no `samplingSeed`, at wall-clock time `clock`. -/
def cSeedless (s nvals : Nat) (clock : Nat) : List Nat := cDraws s nvals (cEffSeed 0 clock)

/-! ## Properties (C parity) -/

/-- **Default sample size matches C**: with no `samplingSize`, betweenness samples
`BETWEENNESS_DEFAULT_SAMPLE_SIZE = 32` sources. -/
def DefaultSizeMatchesC : Prop := Config.samplingSize [] = .ok cDefaultSize

/-- **A seedless run is randomised**: with no `samplingSeed`, the chosen sources depend
on the clock (C seeds from `time(NULL)`), i.e. two clocks give two different samples.
`run clock` is the sample drawn at that clock. -/
def SeedlessRunVaries (run : Nat → List Nat) : Prop := ∃ t₁ t₂, run t₁ ≠ run t₂

/-- **One source per requested sample**: whenever `samplingSize ≤ n`, exactly
`samplingSize` sources are handed to `LAGr_Betweenness` (C: `MIN(nvals, samplingSize)`
draws). A fix that samples without replacement satisfies it too. -/
def SourcesExact (sampler : Nat → Nat → Nat → List Nat) : Prop :=
  ∀ s n seed, s ≤ n → (sampler s n seed).length = s

/-! ## C satisfies them (on the 40-cycle of the live repro) -/

theorem c_default_32 : cDefaultSize = 32 := rfl

theorem c_draw_count (s nvals seed : Nat) : (cDraws s nvals seed).length = min nvals s := by
  unfold cDraws
  generalize min nvals s = k
  induction k generalizing seed with
  | zero => rfl
  | succ k ih => simp [cDrawsAux, ih]

theorem c_sources_exact : SourcesExact (fun s n seed => cDraws s n seed) := by
  intro s n seed h; rw [c_draw_count]; omega

/-- C's seedless 32-draw sample on 40 entries differs between clocks 1 and 2. -/
theorem c_seedless_varies : SeedlessRunVaries (cSeedless 32 40) :=
  ⟨1, 2, by decide⟩

/-! ## Rust violates them: BUG #3380 -/

/-- **BUG #3380 (default size)**: no `samplingSize` gives 16, not C's 32. Live, 40-node
directed cycle: `CALL algo.betweenness()` sums to 11856 = 16·741 on Rust, 23712 = 32·741
on C (each source contributes 741 on that cycle). -/
theorem bug3380_default_size_16 :
    Config.samplingSize [] = .ok 16 ∧ ¬ DefaultSizeMatchesC := by
  refine ⟨rfl, ?_⟩
  intro h; simp [DefaultSizeMatchesC, Config.samplingSize, lookup, cDefaultSize] at h

/-- **BUG #3380 (seed)**: with no `samplingSeed` the Rust sample is the same at every
clock (`0, 1, …, 15` on 40 nodes), so it is never randomised. -/
theorem bug3380_seedless_deterministic (n : Nat) : ¬ SeedlessRunVaries (rustSeedless 16 n) := by
  rintro ⟨t₁, t₂, h⟩; exact h rfl

theorem rust_seedless_first16 : rustSeedless 16 40 0 = List.range 16 := by decide

/-- **BUG #3380 (fewer sources)**: `samplingSize: 20, samplingSeed: 7` on 40 nodes
yields 15 distinct sources, not 20, because colliding LCG draws are dropped by `used`.
Live: Rust sums to 11115 = 15·741, C to 14820 = 20·741. -/
theorem bug3380_seeded_fewer_sources :
    (rustSources 20 40 7).length = 15 ∧ ¬ SourcesExact rustSources := by
  have h : (rustSources 20 40 7).length = 15 := by decide
  refine ⟨h, fun hs => ?_⟩
  have := hs 20 40 7 (by decide); omega

theorem seeded_draws_value :
    rustSources 20 40 7 = [38, 31, 33, 25, 19, 24, 4, 6, 35, 32, 0, 29, 2, 5, 36] := by decide

end AlgoUdf.Betweenness
