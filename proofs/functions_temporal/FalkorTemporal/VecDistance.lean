/-!
# `graph/src/runtime/vec_distance.rs` (origin/main 3fec7d7c9)

The simsimd 6.5.16 Rust wrappers (`rust/lib.rs:659-685`, `impl SpatialSimilarity for f32`
:966-1006) are modelled exactly: `sqeuclidean = l2sq`, `cosine = cos`, and each of
`l2sq`/`cos`/`dot` is `if a.len() != b.len() { None } else { Some(kernel(a, b)) }`, the
kernel being the C FFI routine (`simsimd_l2sq_f32` etc.) — a parameter (`Kernels`), its
numeric value is not modelled. `f64::sqrt` and negation are parameters too.

Proved: each metric's exact composition (`euclidean = sqrt ∘ l2sq`, `inner_product = -dot`,
`cosine = cos`), the dispatch table of `distance` (`None` ⇒ euclidean, unknown name ⇒
`None`, names are case-sensitive), and **`distance_none_iff`**: the result is `None`
exactly when the metric is unknown or the lengths differ — the only two reasons the
callers report (`vec.euclideanDistance` / vector KNN).
-/

namespace FalkorTemporal.VecDist

variable {F32 F64 : Type}

/-- The FFI kernels (`simsimd_*_f32`) and the two `f64` operations used. -/
structure Kernels (F32 F64 : Type) where
  l2sq : List F32 → List F32 → F64
  cos : List F32 → List F32 → F64
  dot : List F32 → List F32 → F64
  sqrt : F64 → F64
  neg : F64 → F64

variable (K : Kernels F32 F64)

/-- simsimd's length-checked wrapper. -/
def checked (k : List F32 → List F32 → F64) (a b : List F32) : Option F64 :=
  if a.length ≠ b.length then none else some (k a b)

/-- `euclidean` (:30): `f32::sqeuclidean(a, b).map(f64::sqrt)`. -/
def euclidean (a b : List F32) : Option F64 := (checked K.l2sq a b).map K.sqrt
/-- `cosine` (:40): `f32::cosine(a, b)`. -/
def cosine (a b : List F32) : Option F64 := checked K.cos a b
/-- `inner_product` (:49): `f32::dot(a, b).map(|d| -d)`. -/
def innerProduct (a b : List F32) : Option F64 := (checked K.dot a b).map K.neg

/-- `distance` (:59). -/
def distance (metric : Option String) (a b : List F32) : Option F64 :=
  match metric.getD "euclidean" with
  | "euclidean" => euclidean K a b
  | "cosine" => cosine K a b
  | "ip" => innerProduct K a b
  | _ => none

def known (m : String) : Bool := m == "euclidean" || m == "cosine" || m == "ip"

theorem euclidean_spec (a b : List F32) (h : a.length = b.length) :
    euclidean K a b = some (K.sqrt (K.l2sq a b)) := by simp [euclidean, checked, h]
theorem cosine_spec (a b : List F32) (h : a.length = b.length) :
    cosine K a b = some (K.cos a b) := by simp [cosine, checked, h]
theorem innerProduct_spec (a b : List F32) (h : a.length = b.length) :
    innerProduct K a b = some (K.neg (K.dot a b)) := by simp [innerProduct, checked, h]

theorem mismatch (a b : List F32) (h : a.length ≠ b.length) :
    euclidean K a b = none ∧ cosine K a b = none ∧ innerProduct K a b = none := by
  simp [euclidean, cosine, innerProduct, checked, h]

/-- `metric == None` defaults to euclidean (mirrors `VectorIndexOptions` default). -/
theorem distance_default (a b : List F32) : distance K none a b = euclidean K a b := rfl

theorem distance_table (a b : List F32) :
    distance K (some "euclidean") a b = euclidean K a b ∧ distance K (some "cosine") a b = cosine K a b ∧
    distance K (some "ip") a b = innerProduct K a b ∧ distance K (some "IP") a b = none ∧
    distance K (some "Cosine") a b = none := ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem distance_none_iff (m : Option String) (a b : List F32) :
    distance K m a b = none ↔ (known (m.getD "euclidean") = false ∨ a.length ≠ b.length) := by
  unfold distance
  generalize m.getD "euclidean" = s
  by_cases h : a.length = b.length
  · by_cases h1 : s = "euclidean"
    · subst h1; simp [euclidean, checked, h, known]
    by_cases h2 : s = "cosine"
    · subst h2; simp [cosine, checked, h, known]
    by_cases h3 : s = "ip"
    · subst h3; simp [innerProduct, checked, h, known]
    simp [known, h, h1, h2, h3]
  · by_cases h1 : s = "euclidean"
    · subst h1; simp [euclidean, checked, h]
    by_cases h2 : s = "cosine"
    · subst h2; simp [cosine, checked, h]
    by_cases h3 : s = "ip"
    · subst h3; simp [innerProduct, checked, h]
    simp [h, h1, h2, h3]

end FalkorTemporal.VecDist
