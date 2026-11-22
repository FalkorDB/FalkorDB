import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days),
one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so Lake checks the chunks in parallel.
Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_0_0 : allRange bwdOk 0 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_0_1 : allRange bwdOk 2500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_0_2 : allRange bwdOk 5000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_0_3 : allRange bwdOk 7500 2500 = true := by decide +kernel

theorem bwd_0 : allRange bwdOk 0 10000 = true := allRange_append bwdOk 0 7500 2500 (allRange_append bwdOk 0 5000 2500 (allRange_append bwdOk 0 2500 2500 (bwd_0_0) bwd_0_1) bwd_0_2) bwd_0_3

end FalkorTemporal
