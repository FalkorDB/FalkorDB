/-
# `Tensor::decode` trusts the tensor section → `edge_count` underflow-panics

`Tensor::decode` (tensor.rs:1530) rebuilds the inline forward matrix `m` from the
on-disk forward layers, then fills the multi-edge store `me` from the tensor
section (tensor.rs:1573-1611). It never checks the *promotion-completeness*
invariant that ties them together:

> `eff_get (src,dst) = MULTI_EDGE`  ⟺  row `κ(src,dst)` of `me` is non-empty
> (tensor.rs module docs; `single_me` / `multi_me` in `proofs/versioned_matrix`).

A crafted payload can put a pair in the tensor section (giving it an `me` row)
without a matching `MULTI_EDGE` forward entry. Then `edge_count` (tensor.rs:1221),
which is *checked* precisely because a decoded blob can break an invariant,
computes `multi` (number of `me` rows) greater than the effective forward
pattern and the subtraction underflows — a deliberate `panic!`. Reached from the
very next `Tensor::encode` (BGSAVE / GRAPH.COPY), so a restored graph can no
longer be saved.

`edge_count`'s identity (tensor.rs:1221):
`|E| = |m| + |dp| − |dm| − |dp∩m| − multi + |me|`, each step checked.

Confirmed: `bug_tensor_decode_accepts_orphan_me_row_then_edge_count_panics`
(`graph/tests/lean_graphblas_wrappers.rs`) prints
`Tensor::edge_count: multi exceeds the effective pattern
(promotion-completeness is broken). |m|=0 |dp|=0 |dm|=0 |dp∩m|=0 multi=1 |me|=1`.
-/

namespace GBW

/-- Checked `u64` subtraction, matching `checked_sub().unwrap_or_else(panic)`
(tensor.rs:1255-1275): `none` = the panic. -/
def csub (x y : Nat) : Option Nat := if y ≤ x then some (x - y) else none

/-- Checked add — the two additions in `edge_count`. Overflow needs an `nvals`
that is not a real matrix size (`> u64`); modelled as always-`some` at the `Nat`
level, since the additions cannot underflow. -/
def cadd (x y : Nat) : Option Nat := some (x + y)

/-- `Tensor::edge_count` (tensor.rs:1221), the exact left-to-right checked
sequence. `some n` = returns `n`; `none` = panics (server crash). -/
def edgeCount (m dp dm shadow multi me : Nat) : Option Nat :=
  (cadd m dp).bind fun a =>
  (csub a dm).bind fun a =>
  (csub a shadow).bind fun a =>
  (csub a multi).bind fun a =>
  cadd a me

/-- The delta/promotion invariants `edge_count`'s doc-comment lists as bounding
each subtraction (tensor.rs:1221-1255):
`dm ⊆ m` ⇒ `dm ≤ m`; `dp ∩ dm = ∅` bounds the pattern so `shadow ≤ m+dp−dm`;
promotion-completeness ⇒ `multi ≤ effective pattern`. -/
structure Wf (m dp dm shadow multi me : Nat) : Prop where
  dm_le : dm ≤ m
  shadow_le : shadow ≤ m + dp - dm
  multi_le : multi ≤ m + dp - dm - shadow

/-- **No underflow while the invariant holds.** For a tensor whose layers
satisfy `Wf`, `edge_count` returns a value and never panics — the machine-checked
core of the `# Panics` doc-comment. -/
theorem edgeCount_no_panic (m dp dm shadow multi me : Nat) (h : Wf m dp dm shadow multi me) :
    (edgeCount m dp dm shadow multi me).isSome := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold edgeCount cadd csub
  have e1 : dm ≤ m + dp := by omega
  simp only [Option.bind, if_pos e1]
  have e2 : shadow ≤ m + dp - dm := h2
  simp only [if_pos e2]
  have e3 : multi ≤ m + dp - dm - shadow := h3
  simp only [if_pos e3]
  rfl

/-- The value it returns is the edge-count identity. -/
theorem edgeCount_value (m dp dm shadow multi me : Nat) (h : Wf m dp dm shadow multi me) :
    edgeCount m dp dm shadow multi me = some (m + dp - dm - shadow - multi + me) := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold edgeCount cadd csub
  have e1 : dm ≤ m + dp := by omega
  simp only [Option.bind, if_pos e1, if_pos h2, if_pos h3]

/-- **The decode gap.** An orphan `me` row — a multi-edge pair the decoder
accepted with no forward entry — has `m = dp = dm = shadow = 0`, `multi = 1`,
`me = 1`. That violates `Wf.multi_le` (`1 ≤ 0` is false) and `edge_count`
panics. This is the exact state the confirmed repro prints. -/
theorem orphan_row_breaks_wf : ¬ Wf 0 0 0 0 1 1 := by
  intro h
  have := h.multi_le
  omega

theorem orphan_row_panics : edgeCount 0 0 0 0 1 1 = none := by decide

/-- More generally, any decoded tensor with more `me` rows than its effective
forward pattern panics — the class of the bug, not just the one witness. -/
theorem more_me_rows_than_pattern_panics
    (m dp dm shadow multi me : Nat)
    (hdm : dm ≤ m + dp) (hsh : shadow ≤ m + dp - dm)
    (hbad : m + dp - dm - shadow < multi) :
    edgeCount m dp dm shadow multi me = none := by
  unfold edgeCount cadd csub
  simp only [Option.bind, if_pos hdm, if_pos hsh]
  have : ¬ multi ≤ m + dp - dm - shadow := by omega
  simp only [if_neg this]

end GBW
