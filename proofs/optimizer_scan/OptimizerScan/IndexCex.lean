import OptimizerScan.Index
/-!
# Counterexamples to "`utilize_index` preserves results"

Each `example` evaluates both the rewritten plan (`utilize`) and the reference
(`Filter` over `NodeByLabelScan`) on one node, by `decide`. Every one of them was then
reproduced against the real engine (Cypher, with vs without the index, and vs C) — see
the header of `OptimizerScan.lean` for queries and outputs.

Keys: 0 = `v`, 1 = `a`, 2 = `b`, 3 = `arr`. Labels: 0 = `L`/`A`, 1 = `B`.
-/
namespace OptimizerScan.Index.Cex

def node (ls : List Nat) (props : List (Nat × V)) : Node :=
  { labels := ls, prop := fun k => (props.find? (·.1 == k)).map (·.2) |>.getD .null }

def noOpq : Nat → Node → Bool := fun _ _ => true
/-- An unindexable conjunct that is always true (`n.k > 0` on an unindexed key). -/
def opqTrue : Nat → Node → Bool := fun _ _ => true

/-- Label 0 indexes `v`, `a`, `arr`; label 1 indexes `b`. -/
def idx1 : Nat → List Nat | 0 => [0, 1, 3] | 1 => [2] | _ => []

def agree (ls : List Nat) (f : F) (n : Node) : Bool :=
  (utilize idx1 ls f).sel idx1 noOpq n == (reference ls f).sel idx1 noOpq n

/-- The rewrite is right on plain cases (sanity): `n.v = 1`, `n.v > 1`, `n.v IN [1,2]`,
`1 <= n.v AND n.v < 5`, `n.v = 1 OR n.v = 7`. -/
example : agree [0] (.atom (.cmp .eq (.prop 0) (.lit (.i 1)))) (node [0] [(0, .i 1)]) := by decide
example : agree [0] (.atom (.cmp .gt (.prop 0) (.lit (.i 1)))) (node [0] [(0, .i 1)]) := by decide
example : agree [0] (.atom (.inn (.prop 0) (.list [.lit (.i 1), .lit (.i 2)]))) (node [0] [(0, .i 2)]) := by decide
example : agree [0] (.and [.cmp .le (.lit (.i 1)) (.prop 0), .cmp .lt (.prop 0) (.lit (.i 5))])
    (node [0] [(0, .i 3)]) := by decide
example : agree [0] (.or [.cmp .eq (.prop 0) (.lit (.i 1)), .cmp .eq (.prop 0) (.lit (.i 7))])
    (node [0] [(0, .i 7)]) := by decide

/-- C1 (shared with C): Bool and Int share the NUMERIC field — `n.v = 1` matches `v: true`. -/
example : agree [0] (.atom (.cmp .eq (.prop 0) (.lit (.i 1)))) (node [0] [(0, .b true)]) = false := by
  decide

/-- C2 — FIXED by #3076 (e20300436). Temporal values used to be indexed as NUMERIC timestamps, so
`n.v > 0` matched `v: date(...)` with the Filter gone (W2-index-5). Temporals are no longer indexed: the
scan now agrees. -/
example : agree [0] (.atom (.cmp .gt (.prop 0) (.lit (.i 0)))) (node [0] [(0, .date 5)]) = true := by
  decide

/-- C3 — FIXED by #3064 (W2-index-1). `n.v = date(...)` is folded to a `Constant(Date)`, which
`is_non_indexable_subexpr` used not to flag: the Filter was dropped, the index rejected the Date and
the op fell back to a full label scan, returning every node. `is_exact_index_literal` now flags it
(`litFlag`): the Filter stays and the scan agrees. -/
example : agree [0] (.atom (.cmp .eq (.prop 0) (.lit (.date 3)))) (node [0] [(0, .i 1)]) = true := by
  decide

/-- C4 — FIXED by #3064 (W2-scan-3). A constant expression that evaluates to an int ≥ 2^52
(`4503599627370495 + 4503599627370495 + 3`) used to keep no Filter (no literal is lossy), and the
lossy runtime value fell back to a label scan returning every node. `Add` is now outside the
allow-list of `is_non_indexable_subexpr` (`nonIdxT`): the Filter stays and the scan agrees. -/
example : agree [0] (.atom (.cmp .eq (.prop 0)
    (.add (.add (.lit (.i 4503599627370495)) (.lit (.i 4503599627370495))) (.lit (.i 3)))))
    (node [0] [(0, .i 1)]) = true := by decide

/-- C5 — FIXED by #3064 (W2-index-4). Multi-label AND `(n:A:B) WHERE n.a = 1 AND n.b = 2` with
`A(a)`, `B(b)`: both conjuncts used to be merged into one query on A's index, which has no `b`
field → null → nothing. Only A's conjuncts are merged now (`pushAnd`); `n.b = 2` stays in the
leftover filter: the scan agrees. -/
example : agree [0, 1] (.and [.cmp .eq (.prop 1) (.lit (.i 1)), .cmp .eq (.prop 2) (.lit (.i 2))])
    (node [0, 1] [(1, .i 1), (2, .i 2)]) = true := by decide

/-- C6 — FIXED by #3064 (W2-index-4). Multi-label OR: the `b` disjunct used to be a null child of
the union, silently skipped. An OR across labels is no longer pushed (`pushOr`): the scan agrees. -/
example : agree [0, 1] (.or [.cmp .eq (.prop 1) (.lit (.i 1)), .cmp .eq (.prop 2) (.lit (.i 9))])
    (node [0, 1] [(1, .i 5), (2, .i 9)]) = true := by decide

/-- …and the OR across labels is refused outright. -/
example : tryPushdown idx1 [0, 1] (.or [.cmp .eq (.prop 1) (.lit (.i 1)), .cmp .eq (.prop 2) (.lit (.i 9))])
    = none := rfl

/-- C7 — FIXED by #3068. `n.a + 1 IN [2]` used to be pushed as `a IN [2]` (the IN path only
asked that the left side *contain* a property). `bare_property_of` now requires `n.a` itself:
the Filter stays and the scan agrees. -/
example : agree [0] (.atom (.inn (.add (.prop 1) (.lit (.i 1))) (.list [.lit (.i 2)])))
    (node [0] [(1, .i 1)]) = true := by decide

/-- C7' — FIXED by #3068. `abs(n.a) IN [1]` used to be pushed as `a IN [1]`, missing `a = -1`. -/
example : agree [0] (.atom (.inn (.abs (.prop 1)) (.list [.lit (.i 1)])))
    (node [0] [(1, .i (-1))]) = true := by decide

/-- C8 — FIXED by #3068. `2 IN [n.a, n.b]` used to become an array-contains query on `a`. -/
example : agree [0] (.atom (.inn (.lit (.i 2)) (.list [.prop 1, .prop 2])))
    (node [0] [(1, .i 2), (2, .i 7)]) = true := by decide

/-- C9 — FIXED by #3064 (W2-index-3). `n.v IN [1, date(...)]`: the runtime used to drop the Date
item, so the index never returned the node. `IndexQuery::in_list` keeps it, `can_be_served`
refuses the list, the op falls back under the kept Filter: the scan agrees. -/
example : agree [0] (.atom (.inn (.prop 0) (.list [.lit (.i 1), .lit (.date 3)])))
    (node [0] [(0, .date 3)]) = true := by decide

/-- C10 — FIXED by #3736. `1 IN n.arr AND …`: array-contains used to keep its Filter only at the
root, so inside AND it was dropped, and Bool/Int share the numeric array field (shared with C). An
array-contains conjunct now keeps the whole Filter (`pushedNeedsPost`, #3064): the scan agrees. -/
example : agree [0] (.and [.inn (.lit (.i 1)) (.term (.prop 3)), .opq 0])
    (node [0] [(3, .arr [.b true])]) = true := by decide

/-- C11 — FIXED by #3072 (1c9994e37). `n.v > 'B' AND n.v < 'B'` merges into
`Range{min:'B', max:'B', exclusive}`, which `build_string_range_node` used to turn into an exact match
on 'B' (W2-index-6). It is now the empty node:
the scan agrees. -/
example : agree [0] (.and [.cmp .gt (.prop 0) (.lit (.s 66)), .cmp .lt (.prop 0) (.lit (.s 66))])
    (node [0] [(0, .s 66)]) = true := by decide

/-- …while the numeric analogue is right (empty `RediSearch` numeric range). -/
example : agree [0] (.and [.cmp .gt (.prop 0) (.lit (.i 66)), .cmp .lt (.prop 0) (.lit (.i 66))])
    (node [0] [(0, .i 66)]) := by decide

end OptimizerScan.Index.Cex
